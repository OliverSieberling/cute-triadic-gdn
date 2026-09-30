"""Blackwell (sm100/sm103) backward, first stage: the reverse recurrence (tcgen05 MMAs into tensor memory,
warp-specialized, one CTA per batch row or document, head and value block of VC columns), then the state gradients
of sm100_joint_state_grads.  Same outputs as sm90_joint_bwd_recurrent.

G = [G_1 ... G_E] (K x N, N = E*VC) is the state gradient dL/dS_{n+1} while chunk n is processed (last chunk first):
    R   = [K; Q] G                        rows 0-63: K G_e, all slices at once (the Q rows are not used)
    dVd = sum_e diag(r_e) (K G_e) + X     X = Pm^T dO, Pm = scale * tril(Q K^T * R')     (r_e = k2_e exp(gL_e - gc_e))
    dW  = (A diag(beta))^T dVd            A operand = the transposed tile read twice, so dW lands in all 128 lanes
    dG  = [K^T | Q^T] [VN; VR]            VN_e = -diag(a_e) dW,  VR_e = diag(c_e) dO     (a_e = k2_e exp(gc_e),
    G  <- diag(exp(gL)) G + dG                                                            c_e = scale q2_e exp(gc_e))
Roles (512 threads):
  warps 0-7   STATE: G (two warpgroups, N/2 columns each; thread t = key row t), its BF16 copy for the MMA and the
              saved dS (TMA), VN from dW, dV = dW (BF16)
  warps 8-11  RED: K side (t < 64, row t) dVd; Q side (row t-64) Pm, dO, VR; both A diag(beta)
  warp 12     MMA issuer: per chunk R, dW, dG, then P and X for the next chunk
  warp 13     TMA loads of K, Q (two stages) and A, R' (one stage)
  warps 14-15 gate tables (two stages)
Tensor memory: R and dG share [0, N); P at 256, X at 320, dW at 384.
Shared memory: the BF16 copy of G and the [VN; VR] tile share one buffer (their lifetimes do not overlap).
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from .sm100_tc import (idesc, mma128, commit, fence_before, fence_after, fence_async, wait_ld, ld32, f32_of_i32,
                       smem_desc, off64, off128, mbar_arrive, mbar_wait, pack2, unpack_lo, unpack_hi, sts128, stg128,
                       lds128, ldg128, saddr, swz_bytes, SW128, SW64, SW32)
from .sm90_joint_packed import token_chunk, valid_rows, keep_singleton_tma_axis
from .sm100_joint_fwd import _value_cols
from .sm100_joint_state_grads import joint_state_grads_sm100

C, KD, DV = 64, 128, 128
NT = 512
T_R, T_P, T_X, T_W = 0, 256, 320, 384
NREG_S, NREG_R, NREG_M = 184, 104, 40
(KQ_FULL, KQ_EMPTY, GT_FULL, GT_EMPTY, AM_FULL, AM_EMPTY, SG_FULL, SG_FREE, P_FULL, PM_READY, X_FULL, R_FULL,
 R_EMPTY, DVD_READY, DW_FULL, VN_FULL, DG_FULL) = (0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
NBAR = 21


def _adv128k(ki):
    return ((ki >> 2) * 16384 + (ki & 3) * 32) >> 4


def _advmn(ki):
    return (ki * 2048) >> 4


def _bar(sbar, k):
    return cutlass.Int32((sbar.iterator + k).toint())


@cute.kernel
def rec_kernel(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor, gDO: cute.Tensor,
               gDv: cute.Tensor, gDvd: cute.Tensor,
               tmaK: cute.CopyAtom, tKt: cute.Tensor, tmaQ: cute.CopyAtom, tQt: cute.Tensor,
               tmaA: cute.CopyAtom, tAt: cute.Tensor, tmaM: cute.CopyAtom, tMt: cute.Tensor,
               tmaS: cute.CopyAtom, tSt: cute.Tensor, lBox: cute.ComposedLayout, lG: cute.ComposedLayout,
               gOffsets: cute.Tensor, mapping: cute.Tensor,
               H: cutlass.Constexpr, E: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr,
               SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    N = E * VC
    NH = N // 2
    NQ = NH // 2
    NV = DV // VC
    NS = N if N >= 64 else 64
    SWB = 1 if VC == 16 else (2 if VC == 32 else 3)
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    vb, h, b = bid % NV, (bid // NV) % H, bid // (NV * H)
    first, count = cutlass.Int32(0), cutlass.Int32(NC)
    if cutlass.const_expr(DOCS):
        # packed documents: chunks [first, first + count) of the padded chunk space, as in sm100_joint_fwd
        if cutlass.const_expr(NATIVE):
            b = cutlass.Int32(mapping[2, b])
        first = cutlass.Int32(gOffsets[b])
        count = cutlass.Int32(gOffsets[b + 1]) - first
        b = cutlass.Int32(0)
    bf = cutlass.BFloat16
    f32 = cutlass.Float32

    smem = utils.SmemAllocator()
    sKQ = smem.allocate_tensor(bf, cute.make_layout((2 * 128 * KD,)), byte_alignment=1024)
    sAm = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)
    sMm = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)
    sAb = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)          # A diag(beta)
    sPm = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)
    sDO = smem.allocate_tensor(bf, cute.make_layout((C * 64,)), byte_alignment=1024)         # dO block (cols >= VC zero)
    sDVD = smem.allocate_tensor(bf, cute.make_layout((C * 64,)), byte_alignment=1024)        # dVd (cols >= VC zero)
    sGV = smem.allocate_tensor(bf, cute.make_layout((128 * (N if N >= 128 else 128),)), byte_alignment=1024)  # G / [VN; VR]
    sa = smem.allocate_tensor(f32, cute.make_layout((2, E, C), stride=(E * C, C, 1)), byte_alignment=16)
    sr = smem.allocate_tensor(f32, cute.make_layout((2, E, C), stride=(E * C, C, 1)), byte_alignment=16)
    sc = smem.allocate_tensor(f32, cute.make_layout((2, E, C), stride=(E * C, C, 1)), byte_alignment=16)
    sb = smem.allocate_tensor(f32, cute.make_layout((2, C), stride=(C, 1)), byte_alignment=16)
    sd = smem.allocate_tensor(f32, cute.make_layout((2, 16), stride=(16, 1)), byte_alignment=16)
    sbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NBAR,)), byte_alignment=8)
    thold = smem.allocate_tensor(cutlass.Int32, cute.make_layout((1,)), byte_alignment=16)

    if warp == 0:
        with cute.arch.elect_one():
            for st in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(sbar.iterator + KQ_FULL + st, 1)
                cute.arch.mbarrier_init(sbar.iterator + KQ_EMPTY + st, 1)
                cute.arch.mbarrier_init(sbar.iterator + GT_FULL + st, 64)
                cute.arch.mbarrier_init(sbar.iterator + GT_EMPTY + st, 128 + 256)
            cute.arch.mbarrier_init(sbar.iterator + AM_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + AM_EMPTY, 128)
            cute.arch.mbarrier_init(sbar.iterator + SG_FULL, 256)
            cute.arch.mbarrier_init(sbar.iterator + SG_FREE, 1)
            cute.arch.mbarrier_init(sbar.iterator + P_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + PM_READY, 64)
            cute.arch.mbarrier_init(sbar.iterator + X_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + R_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + R_EMPTY, 64)
            cute.arch.mbarrier_init(sbar.iterator + DVD_READY, 64)
            cute.arch.mbarrier_init(sbar.iterator + DW_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + VN_FULL, 256 + 64)
            cute.arch.mbarrier_init(sbar.iterator + DG_FULL, 1)
    cute.arch.mbarrier_init_fence()
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(thold.iterator, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(f32)
    tb = cutlass.Int32(tmem_ptr.toint())
    z = cutlass.Int32(0)
    r0, c80 = tid // 8, (tid % 8) * 8
    if cutlass.const_expr(VC < 64):                   # zero padding columns [VC, 64) of dO and dVd once
        if c80 >= VC:
            sts128(saddr(sDO.iterator, off64(r0, c80)), z, z, z, z)
            sts128(saddr(sDVD.iterator, off64(r0, c80)), z, z, z, z)
    fence_async()
    cute.arch.barrier()

    one = cute.make_layout(1)
    kq_ptr = sKQ.iterator

    def box(ptr, off):
        return cute.make_tensor(cute.recast_ptr(ptr + off, lBox.inner, dtype=bf), lBox.outer)

    if cutlass.const_expr(NATIVE):
        # raw token view from the document's first token (tile row = chunk index within the document).  Rows
        # past the document end: dO is loaded as zero and k2, q2, beta are masked, so they add nothing to X,
        # dVd, dW, [VN; VR] or dG (Pm is lower triangular, so their columns never meet a valid row either)
        start = mapping[0, first]
        tKt = cute.domain_offset((start, 0, 0, 0, 0), tKt)
        tQt = cute.domain_offset((start, 0, 0, 0, 0), tQt)
    gK = cute.group_modes(cute.local_tile(tKt, (C, 64), (None, None, None, None, None)), 0, 2)
    gQ = cute.group_modes(cute.local_tile(tQt, (C, 64), (None, None, None, None, None)), 0, 2)
    gA = cute.group_modes(cute.local_tile(tAt, (C, 64), (None, None, None, None, None)), 0, 2)
    gM = cute.group_modes(cute.local_tile(tMt, (C, 64), (None, None, None, None, None)), 0, 2)
    dK00, tgK = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 0), 0, 2), gK)
    dK01, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 8192), 0, 2), gK)
    dK10, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 16384), 0, 2), gK)
    dK11, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 24576), 0, 2), gK)
    dQ00, tgQ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 4096), 0, 2), gQ)
    dQ01, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 12288), 0, 2), gQ)
    dQ10, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 20480), 0, 2), gQ)
    dQ11, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 28672), 0, 2), gQ)
    dA, tgA = cpasync.tma_partition(tmaA, 0, one, cute.group_modes(box(sAm.iterator, 0), 0, 2), gA)
    dM, tgM = cpasync.tma_partition(tmaM, 0, one, cute.group_modes(box(sMm.iterator, 0), 0, 2), gM)
    sGv = cute.make_tensor(cute.recast_ptr(sGV.iterator, lG.inner, dtype=bf), lG.outer)          # (VC, KD, E)
    tGs, tgS = cpasync.tma_partition(tmaS, 0, one, cute.group_modes(sGv, 0, 2),
                                     cute.group_modes(cute.local_tile(tSt, (VC, KD), (None, None, None, None, None)), 0, 2))

    if tid < 256:
        # ================================================================ STATE
        cute.arch.setmaxregister_increase(NREG_S)
        g = tid // 128
        t = tid % 128
        col0 = g * NH
        r = t % 64
        qc0 = col0 + (t // 64) * NQ
        G = cute.make_rmem_tensor((NH,), f32)
        w = cute.make_rmem_tensor((VC,), f32)
        x = cute.make_rmem_tensor((8,), f32)
        for j in cutlass.range_constexpr(NH):
            G[j] = cutlass.Float32(0.)
        gbase = cutlass.Int32(sGV.iterator.toint())
        for it in cutlass.range(count, unroll=1):
            n = first + count - 1 - it
            st = it % 2
            ph = it & 1
            # BF16 copy of G -> per-slice MN-major regions (the saved dS of chunk n, and the B operand of R)
            for c8 in cutlass.range_constexpr(NH // 8):
                cc = 8 * c8
                col = col0 + cc
                boff = 2 * ((col % VC) + t * VC + (col // VC) * (VC * KD))
                sts128(gbase + swz_bytes(boff, SWB), pack2(G[cc + 1], G[cc]), pack2(G[cc + 3], G[cc + 2]),
                       pack2(G[cc + 5], G[cc + 4]), pack2(G[cc + 7], G[cc + 6]))
            fence_async()
            fence_before()
            mbar_arrive(_bar(sbar, SG_FULL))
            cute.arch.barrier(barrier_id=3, number_of_threads=256)
            if tid < 32:
                for e in cutlass.range_constexpr(E):
                    cute.copy(tmaS, tGs[(None, e)], tgS[(None, vb, e, b, n, h)])
                cute.arch.cp_async_bulk_commit_group()
            # ---- dW (all 128 lanes hold row lane % 64): VN_e = -a_e dW into rows 0-63 of [VN; VR]
            mbar_wait(_bar(sbar, DW_FULL), ph)
            fence_after()
            for s_ in cutlass.range_constexpr(VC // 16):
                vals = ld32(tb + (T_W + 16 * s_), 16)
                for j in cutlass.range_constexpr(16):
                    w[16 * s_ + j] = f32_of_i32(vals[j])
            wait_ld()
            if tid < 32:
                cute.arch.cp_async_bulk_wait_group(0, read=True)      # the dS save has read the buffer
                with cute.arch.elect_one():
                    mbar_arrive(_bar(sbar, SG_FREE))
            mbar_wait(_bar(sbar, SG_FREE), ph)                          # G copy dead (R done, save done)
            if cutlass.const_expr(NQ >= VC):
                for c8 in cutlass.range_constexpr(NQ // 8):
                    col = qc0 + 8 * c8
                    aw = -sa[st, col // VC, r]
                    for j in cutlass.range_constexpr(8):
                        x[j] = aw * w[(8 * c8) % VC + j]
                    sts128(gbase + 2 * off128(r, col), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
            else:
                cb = qc0 % VC
                wq = cute.make_rmem_tensor((NQ,), f32)
                for j in cutlass.range_constexpr(NQ):
                    wq[j] = w[j]
                for opt in cutlass.range_constexpr(1, VC // NQ):
                    if cb == opt * NQ:
                        for j in cutlass.range_constexpr(NQ):
                            wq[j] = w[opt * NQ + j]
                aw = -sa[st, qc0 // VC, r]
                for c8 in cutlass.range_constexpr(NQ // 8):
                    for j in cutlass.range_constexpr(8):
                        x[j] = aw * wq[8 * c8 + j]
                    sts128(gbase + 2 * off128(r, qc0 + 8 * c8), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
            if cutlass.const_expr(N < 64):
                if t < 64:                     # the dG MMA spans 64 columns: zero VN columns [N, 64) of row r
                    for c8 in cutlass.range_constexpr((64 - N) // 8):
                        sts128(gbase + 2 * off128(r, N + 8 * c8), cutlass.Int32(0), cutlass.Int32(0),
                               cutlass.Int32(0), cutlass.Int32(0))
            fence_async()
            fence_before()
            mbar_arrive(_bar(sbar, VN_FULL))
            if tid < 64:                       # dV = dW (bf16), row r
                dvc = token_chunk(gDv, n, mapping, NATIVE)
                if r < valid_rows(n, mapping, NATIVE):
                    for c8 in cutlass.range_constexpr(VC // 8):
                        cc = 8 * c8
                        stg128((dvc.iterator + dvc.layout((b, n, r, h, vb * VC + cc))).toint(),
                               pack2(w[cc + 1], w[cc]), pack2(w[cc + 3], w[cc + 2]),
                               pack2(w[cc + 5], w[cc + 4]), pack2(w[cc + 7], w[cc + 6]))
            # ---- G <- exp(gL) G + dG
            mbar_wait(_bar(sbar, DG_FULL), ph)
            fence_after()
            for s_ in cutlass.range_constexpr(NH // 16):
                dec = sd[st, (col0 + 16 * s_) // VC]
                vals = ld32(tb + (T_R + col0 + 16 * s_), 16)
                for j in cutlass.range_constexpr(16):
                    G[16 * s_ + j] = G[16 * s_ + j] * dec + f32_of_i32(vals[j])
            wait_ld()
            mbar_arrive(_bar(sbar, GT_EMPTY + st))
        if tid < 32:
            cute.arch.cp_async_bulk_wait_group(0)
    elif tid < 384:
        # ================================================================ RED
        cute.arch.setmaxregister_decrease(NREG_R)
        t = tid - 256
        kside = t < 64
        r = t % 64
        acc = cute.make_rmem_tensor((VC,), f32)
        dov = cute.make_rmem_tensor((VC,), f32)
        x = cute.make_rmem_tensor((8,), f32)
        gbase = cutlass.Int32(sGV.iterator.toint())
        for it in cutlass.range(count, unroll=1):
            n = first + count - 1 - it
            st = it % 2
            ph = it & 1
            ph2 = (it // 2) & 1
            if not kside:
                if cutlass.const_expr(NATIVE):
                    doc_ = token_chunk(gDO, n, mapping, NATIVE)
                    for j in cutlass.range_constexpr(VC):
                        dov[j] = cutlass.Float32(0.)
                    if r < valid_rows(n, mapping, NATIVE):
                        for c8 in cutlass.range_constexpr(VC // 8):
                            wv = ldg128((doc_.iterator + doc_.layout((b, n, r, h, vb * VC + 8 * c8))).toint())
                            for j in cutlass.range_constexpr(4):
                                dov[8 * c8 + 2 * j] = unpack_lo(wv[j])
                                dov[8 * c8 + 2 * j + 1] = unpack_hi(wv[j])
                else:
                    for c8 in cutlass.range_constexpr(VC // 8):
                        wv = ldg128((gDO.iterator + gDO.layout((b, n, r, h, vb * VC + 8 * c8))).toint())
                        for j in cutlass.range_constexpr(4):
                            dov[8 * c8 + 2 * j] = unpack_lo(wv[j])
                            dov[8 * c8 + 2 * j + 1] = unpack_hi(wv[j])
            mbar_wait(_bar(sbar, GT_FULL + st), ph2)
            mbar_wait(_bar(sbar, AM_FULL), ph)
            if kside:
                # A diag(beta), row r (the last chunk's dW MMA has finished reading the tile)
                if it > 0:
                    mbar_wait(_bar(sbar, DW_FULL), (it - 1) & 1)
                for c8 in cutlass.range_constexpr(8):
                    wa = lds128(saddr(sAm.iterator, off64(r, 8 * c8)))
                    for j in cutlass.range_constexpr(4):
                        x[2 * j] = unpack_lo(wa[j]) * sb[st, 8 * c8 + 2 * j]
                        x[2 * j + 1] = unpack_hi(wa[j]) * sb[st, 8 * c8 + 2 * j + 1]
                    sts128(saddr(sAb.iterator, off64(r, 8 * c8)), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
                fence_async()
                mbar_arrive(_bar(sbar, AM_EMPTY))
                # dVd = sum_e r_e (K G_e) + X
                for j in cutlass.range_constexpr(VC):
                    acc[j] = cutlass.Float32(0.)
                mbar_wait(_bar(sbar, R_FULL), ph)
                fence_after()
                for e in cutlass.range_constexpr(E):
                    rw = sr[st, e, r]
                    for s_ in cutlass.range_constexpr(VC // 16):
                        vals = ld32(tb + (T_R + e * VC + 16 * s_), 16)
                        for j in cutlass.range_constexpr(16):
                            acc[16 * s_ + j] = acc[16 * s_ + j] + rw * f32_of_i32(vals[j])
                wait_ld()
                fence_before()
                mbar_arrive(_bar(sbar, R_EMPTY))
                mbar_wait(_bar(sbar, X_FULL), ph)
                fence_after()
                for s_ in cutlass.range_constexpr(VC // 16):
                    vals = ld32(tb + (T_X + 16 * s_), 16)
                    for j in cutlass.range_constexpr(16):
                        acc[16 * s_ + j] = acc[16 * s_ + j] + f32_of_i32(vals[j])
                wait_ld()
                fence_before()
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    sts128(saddr(sDVD.iterator, off64(r, cc)), pack2(acc[cc + 1], acc[cc]), pack2(acc[cc + 3], acc[cc + 2]),
                           pack2(acc[cc + 5], acc[cc + 4]), pack2(acc[cc + 7], acc[cc + 6]))
                fence_async()
                mbar_arrive(_bar(sbar, DVD_READY))
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    stg128((gDvd.iterator + gDvd.layout((b, n, r, h, vb * VC + cc))).toint(),
                           pack2(acc[cc + 1], acc[cc]), pack2(acc[cc + 3], acc[cc + 2]),
                           pack2(acc[cc + 5], acc[cc + 4]), pack2(acc[cc + 7], acc[cc + 6]))
            else:
                # Pm = scale * tril(Q K^T * R') row r, and the dO block (B operand of X = Pm^T dO)
                mbar_wait(_bar(sbar, P_FULL), ph)
                fence_after()
                for s_ in cutlass.range_constexpr(4):
                    vals = ld32(tb + (T_P + 16 * s_), 16)
                    for c8 in cutlass.range_constexpr(2):
                        c0 = 16 * s_ + 8 * c8
                        wm = lds128(saddr(sMm.iterator, off64(r, c0)))
                        for j in cutlass.range_constexpr(4):
                            x[2 * j] = SCALE * f32_of_i32(vals[8 * c8 + 2 * j]) * unpack_lo(wm[j])
                            x[2 * j + 1] = SCALE * f32_of_i32(vals[8 * c8 + 2 * j + 1]) * unpack_hi(wm[j])
                        for j in cutlass.range_constexpr(8):
                            if c0 + j > r:
                                x[j] = cutlass.Float32(0.)
                        sts128(saddr(sPm.iterator, off64(r, c0)), pack2(x[1], x[0]), pack2(x[3], x[2]),
                               pack2(x[5], x[4]), pack2(x[7], x[6]))
                wait_ld()
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    sts128(saddr(sDO.iterator, off64(r, cc)), pack2(dov[cc + 1], dov[cc]), pack2(dov[cc + 3], dov[cc + 2]),
                           pack2(dov[cc + 5], dov[cc + 4]), pack2(dov[cc + 7], dov[cc + 6]))
                fence_async()
                fence_before()
                mbar_arrive(_bar(sbar, PM_READY))
                mbar_arrive(_bar(sbar, AM_EMPTY))
                # VR_e = c_e dO into rows 64-127 of [VN; VR] (once the G copy is dead)
                mbar_wait(_bar(sbar, SG_FREE), ph)
                for e in cutlass.range_constexpr(E):
                    cw = sc[st, e, r]
                    for c8 in cutlass.range_constexpr(VC // 8):
                        cc = 8 * c8
                        for j in cutlass.range_constexpr(8):
                            x[j] = cw * dov[cc + j]
                        sts128(gbase + 2 * off128(64 + r, e * VC + cc), pack2(x[1], x[0]), pack2(x[3], x[2]),
                               pack2(x[5], x[4]), pack2(x[7], x[6]))
                if cutlass.const_expr(N < 64):
                    for c8 in cutlass.range_constexpr((64 - N) // 8):
                        sts128(gbase + 2 * off128(64 + r, N + 8 * c8), z, z, z, z)
                fence_async()
                fence_before()
                mbar_arrive(_bar(sbar, VN_FULL))
            mbar_arrive(_bar(sbar, GT_EMPTY + st))
    else:
        cute.arch.setmaxregister_decrease(NREG_M)
        if warp == 12:
            # ================================================================ MMA issuer
            I_P = idesc(128, 64, 0, 0)
            I_R = idesc(128, N, 0, 1)
            I_T = idesc(128, 64, 1, 1)          # (X, dW): A^T read MN-major (twice), B MN-major
            I_G = idesc(128, NS, 1, 1)
            akq = cutlass.Int32(kq_ptr.toint())
            HLAY = SW32 if VC == 16 else (SW64 if VC == 32 else SW128)
            ag = cutlass.Int32(sGV.iterator.toint())
            dG = smem_desc(ag, VC * KD * 2, 8 * VC * 2, HLAY)
            dVR = smem_desc(ag, 16384, 1024)
            dPmT = smem_desc(cutlass.Int32(sPm.iterator.toint()), 0, 1024)
            dAbT = smem_desc(cutlass.Int32(sAb.iterator.toint()), 0, 1024)
            dDO = smem_desc(cutlass.Int32(sDO.iterator.toint()), 0, 1024)
            dDVD = smem_desc(cutlass.Int32(sDVD.iterator.toint()), 0, 1024)
            # prologue: P and X of the first chunk processed (n = NC-1)
            mbar_wait(_bar(sbar, KQ_FULL), 0)
            fence_after()
            for ki in cutlass.range_constexpr(8):
                mma128(tb + T_P, smem_desc(akq, 16, 1024) + _adv128k(ki), smem_desc(akq, 16, 1024) + _adv128k(ki),
                       cutlass.Int32(I_P), cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, P_FULL))
            mbar_wait(_bar(sbar, PM_READY), 0)
            fence_after()
            for ki in cutlass.range_constexpr(4):
                mma128(tb + T_X, dPmT + _advmn(ki), dDO + _advmn(ki), cutlass.Int32(I_T), cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, X_FULL))
            for it in cutlass.range(count, unroll=1):
                st = it % 2
                ph = it & 1
                base = akq + st * 32768
                dKQ = smem_desc(base, 16, 1024)
                dKt = smem_desc(base, 16384, 1024)
                mbar_wait(_bar(sbar, SG_FULL), ph)
                fence_after()
                for ki in cutlass.range_constexpr(8):                                   # R = [K;Q] G
                    mma128(tb + T_R, dKQ + _adv128k(ki), dG + ((ki * 32 * VC) >> 4), cutlass.Int32(I_R),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, R_FULL))
                mbar_wait(_bar(sbar, DVD_READY), ph)
                fence_after()
                for ki in cutlass.range_constexpr(4):                                   # dW = (A diag b)^T dVd
                    mma128(tb + T_W, dAbT + _advmn(ki), dDVD + _advmn(ki), cutlass.Int32(I_T),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, DW_FULL))
                mbar_wait(_bar(sbar, R_EMPTY), ph)
                mbar_wait(_bar(sbar, VN_FULL), ph)
                fence_after()
                for ki in cutlass.range_constexpr(8):                                   # dG = [K^T | Q^T] [VN; VR]
                    mma128(tb + T_R, dKt + _advmn(ki), dVR + _advmn(ki), cutlass.Int32(I_G),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, DG_FULL))
                commit(_bar(sbar, KQ_EMPTY + st))
                if it + 1 < count:                                                      # P, X of the next chunk
                    st1 = (it + 1) % 2
                    base1 = akq + st1 * 32768
                    mbar_wait(_bar(sbar, KQ_FULL + st1), ((it + 1) // 2) & 1)
                    fence_after()
                    for ki in cutlass.range_constexpr(8):
                        mma128(tb + T_P, smem_desc(base1, 16, 1024) + _adv128k(ki), smem_desc(base1, 16, 1024) + _adv128k(ki),
                               cutlass.Int32(I_P), cutlass.Int32(1 if ki > 0 else 0))
                    commit(_bar(sbar, P_FULL))
                    mbar_wait(_bar(sbar, PM_READY), (it + 1) & 1)
                    fence_after()
                    for ki in cutlass.range_constexpr(4):
                        mma128(tb + T_X, dPmT + _advmn(ki), dDO + _advmn(ki), cutlass.Int32(I_T),
                               cutlass.Int32(1 if ki > 0 else 0))
                    commit(_bar(sbar, X_FULL))
        elif warp == 13:
            # ================================================================ TMA loader (chunks in reverse)
            for it in cutlass.range(count, unroll=1):
                n = first + count - 1 - it
                if cutlass.const_expr(NATIVE):
                    kr, kn = count - 1 - it, 0
                else:
                    kr, kn = 0, n
                st = it % 2
                if it >= 2:
                    mbar_wait(_bar(sbar, KQ_EMPTY + st), ((it // 2) - 1) & 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + KQ_FULL + st, 2 * 128 * KD)
                bar = sbar.iterator + KQ_FULL + st
                if st == 0:
                    cute.copy(tmaK, tgK[(None, kr, 0, b, kn, h)], dK00, tma_bar_ptr=bar)
                    cute.copy(tmaK, tgK[(None, kr, 1, b, kn, h)], dK01, tma_bar_ptr=bar)
                    cute.copy(tmaQ, tgQ[(None, kr, 0, b, kn, h)], dQ00, tma_bar_ptr=bar)
                    cute.copy(tmaQ, tgQ[(None, kr, 1, b, kn, h)], dQ01, tma_bar_ptr=bar)
                else:
                    cute.copy(tmaK, tgK[(None, kr, 0, b, kn, h)], dK10, tma_bar_ptr=bar)
                    cute.copy(tmaK, tgK[(None, kr, 1, b, kn, h)], dK11, tma_bar_ptr=bar)
                    cute.copy(tmaQ, tgQ[(None, kr, 0, b, kn, h)], dQ10, tma_bar_ptr=bar)
                    cute.copy(tmaQ, tgQ[(None, kr, 1, b, kn, h)], dQ11, tma_bar_ptr=bar)
                if it >= 1:
                    mbar_wait(_bar(sbar, AM_EMPTY), (it - 1) & 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + AM_FULL, 2 * C * C * 2)
                cute.copy(tmaA, tgA[(None, 0, 0, b, n, h)], dA, tma_bar_ptr=sbar.iterator + AM_FULL)
                cute.copy(tmaM, tgM[(None, 0, 0, b, n, h)], dM, tma_bar_ptr=sbar.iterator + AM_FULL)
        elif warp >= 14:
            # ================================================================ gate tables
            u_ = tid - 448
            EV = 4 if E >= 4 else E
            kv = cute.make_rmem_tensor((EV,), f32)
            qv = cute.make_rmem_tensor((EV,), f32)
            for it in cutlass.range(count, unroll=1):
                n = first + count - 1 - it
                st = it % 2
                rows = valid_rows(n, mapping, NATIVE)
                k2c = token_chunk(gK2, n, mapping, NATIVE)
                q2c = token_chunk(gQ2, n, mapping, NATIVE)
                btc = token_chunk(gBt, n, mapping, NATIVE)
                if it >= 2:
                    mbar_wait(_bar(sbar, GT_EMPTY + st), ((it // 2) - 1) & 1)
                for e0 in cutlass.range_constexpr(0, E, EV):
                    gv = cute.make_tensor((gGc.iterator + gGc.layout((b, n, u_, h, e0))).align(4 * EV), cute.make_layout((EV,))).load()
                    lv = cute.make_tensor((gGc.iterator + gGc.layout((b, n, C - 1, h, e0))).align(4 * EV), cute.make_layout((EV,))).load()
                    if cutlass.const_expr(NATIVE):
                        kv.fill(0.)
                        qv.fill(0.)
                        if u_ < rows:
                            kv.store(cute.make_tensor((k2c.iterator + k2c.layout((b, n, u_, h, e0))).align(4 * EV), cute.make_layout((EV,))).load())
                            qv.store(cute.make_tensor((q2c.iterator + q2c.layout((b, n, u_, h, e0))).align(4 * EV), cute.make_layout((EV,))).load())
                    else:
                        kv.store(cute.make_tensor((gK2.iterator + gK2.layout((b, n, u_, h, e0))).align(4 * EV), cute.make_layout((EV,))).load())
                        qv.store(cute.make_tensor((gQ2.iterator + gQ2.layout((b, n, u_, h, e0))).align(4 * EV), cute.make_layout((EV,))).load())
                    for j in cutlass.range_constexpr(EV):
                        eg = cute.math.exp(gv[j], fastmath=True)
                        sa[st, e0 + j, u_] = kv[j] * eg
                        sr[st, e0 + j, u_] = kv[j] * cute.math.exp(lv[j] - gv[j], fastmath=True)
                        sc[st, e0 + j, u_] = SCALE * qv[j] * eg
                        if u_ == 0:
                            sd[st, e0 + j] = cute.math.exp(lv[j], fastmath=True)
                bv = cutlass.Float32(0.)
                if u_ < rows:
                    bv = btc[b, n, u_, h]
                sb[st, u_] = bv
                mbar_arrive(_bar(sbar, GT_FULL + st))
    cute.arch.barrier()
    tmem.free(tmem_ptr, 512)


@cute.jit
def launch_rec(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor, gDO: cute.Tensor,
               gDv: cute.Tensor, gDvd: cute.Tensor,
               gKT: cute.Tensor, gQT: cute.Tensor, gAT: cute.Tensor, gMT: cute.Tensor, gST: cute.Tensor,
               gOffsets: cute.Tensor, mapping: cute.Tensor,
               H: cutlass.Constexpr, E: cutlass.Constexpr, VC: cutlass.Constexpr,
               NC: cutlass.Constexpr, SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr,
               NBLK: cutlass.Int32, stream: cuda.CUstream):
    bf = cutlass.BFloat16
    lBox = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, 64, 64), bf, 1), (None, None, 0))
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKT, lBox, (C, 64), 1)
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQT, lBox, (C, 64), 1)
    tmaA, tAt = cpasync.make_tiled_tma_atom(op, gAT, lBox, (C, 64), 1)
    tmaM, tMt = cpasync.make_tiled_tma_atom(op, gMT, lBox, (C, 64), 1)
    lG = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, KD), bf, E)
    tmaS, tSt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), gST, cute.slice_(lG, (None, None, 0)), (VC, KD), 1)
    rec_kernel(gGc, gK2, gQ2, gBt, gDO, gDv, gDvd, tmaK, tKt, tmaQ, tQt, tmaA, tAt, tmaM, tMt, tmaS, tSt,
               lBox, lG, gOffsets, mapping, H, E, VC, NC, SCALE, DOCS, NATIVE).launch(
        grid=(NBLK * H * (DV // VC), 1, 1), block=(NT, 1, 1), stream=stream, min_blocks_per_mp=1)


_compiled = {}


def joint_bwd_recurrent_sm100(q, k, do, A, Mp, gc, k2, q2, beta, h, Vd, scale=None, groups=None, value_cols=None,
                              chunk_starts=None, token_map=None):
    """Same contract as sm90_joint_bwd_recurrent.joint_bwd_recurrent (groups and value_cols, the sm90 kernels'
    tuning, are not used).  Returns dqp, dkp, dv, dvd, dgv, dk2v, dgo, dq2o."""
    B, T, H, D_ = q.shape
    E = k2.shape[-1]
    native = token_map is not None
    assert D_ == KD and (native or T % C == 0)
    docs = chunk_starts.numel() - 1 if chunk_starts is not None else 0
    assert not docs or B == 1
    dev = q.device
    VC = _value_cols(docs if docs else B, H, E, dev)
    NC = gc.shape[1] // C
    PT = NC * C
    scale = KD ** -0.5 if scale is None else float(scale)
    A, Mp, do = A.contiguous(), Mp.contiguous(), do.contiguous()
    ds = torch.empty((B, NC, H, E, KD, DV), device=dev, dtype=torch.bfloat16)
    dv = torch.empty_like(do)
    dvd = torch.empty((B, PT, H, DV), device=dev, dtype=torch.bfloat16)

    def ch(t):
        return t.view(B, NC, C, *t.shape[2:])

    def raw(t):
        return t.view(B, 1, T, *t.shape[2:]) if native else ch(t)

    def mat(t):
        return t.view(B, NC, C, H, C).permute(2, 4, 0, 1, 3)

    tl = (ch(gc), raw(k2), raw(q2), raw(beta), raw(do), raw(dv), ch(dvd),
          raw(k).permute(2, 4, 0, 1, 3), raw(q).permute(2, 4, 0, 1, 3), mat(A), mat(Mp),
          ds.view(B, NC, H, E * KD, DV).permute(4, 3, 0, 1, 2))
    args = [from_dlpack(t.detach(), assumed_align=16) for t in tl]
    if native and T == 1 and H == 1:
        for index in (7, 8):
            keep_singleton_tma_axis(args[index], 0)
    args.append(from_dlpack(chunk_starts.detach(), assumed_align=16).mark_layout_dynamic() if docs else args[0])
    args.append(from_dlpack(token_map.detach(), assumed_align=16) if native else args[0])
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    key = (B, T, H, E, VC, scale, bool(docs), native, NC)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_rec, *args, H, E, VC, NC, scale, bool(docs), native,
                                      docs if docs else B, stream)
    _compiled[key](*args, docs if docs else B, stream)      # NBLK is a runtime argument
    dqp, dkp = (torch.empty((B, PT, H, 1, KD), device=dev, dtype=torch.float32) for _ in range(2))
    dgv, dk2v, dgo, dq2o = (torch.empty((B, PT, H, 1, E), device=dev, dtype=torch.float32) for _ in range(4))
    joint_state_grads_sm100(q, k, do, Vd, dv, gc, k2, q2, h, ds, scale, (dqp, dkp, dgv, dk2v, dgo, dq2o),
                            chunk_starts=chunk_starts, token_map=token_map)
    return dqp, dkp, dv, dvd, dgv, dk2v, dgo, dq2o
