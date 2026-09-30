"""Blackwell (sm100/sm103) forward recurrence of Triadic GDN (Eq. 11): tcgen05 MMAs into tensor memory,
warp-specialized, one CTA per (batch row, head, value block of VC columns).

The E state slices of the CTA's value block sit side by side as one K x N matrix S = [S_1 ... S_E], N = E*VC <= 256,
so every per-slice product of a chunk is a single M=128 MMA:
    R  = [K; Q] S                          rows 0-63: K S_e, rows 64-127: Q S_e, all slices at once
    W  = V - sum_e diag(a_e) (K S_e)       reduced in registers, a_e = k2_e exp(gc_e)
    U  = A (diag(beta) W)                  A tile [A; A] (TMA, twice): U lands in all 128 lanes
    dS = K^T [Vn_1 ... Vn_E]               Vn_e = diag(k2_e exp(gL_e - gc_e)) U
    S  <- diag(exp(gL)) S + dS             state in registers
    O  = sum_e diag(c_e) (Q S_e) + Pm U    A tile [A diag(beta); Pm], Pm = scale * tril(Q K^T * R'), rows 64-127
The two A tiles are rows 0-127 and rows 64-191 of one 192-row buffer [A; A; Pm], all TMA-loaded (A once the last
chunk's U has read the tile, Pm once the last chunk's O has); Pm comes from the masks kernel.
Roles (512 threads):
  warps 0-7   STATE: the state (two warpgroups, N/2 columns each; thread t = key row t), its BF16 copy for the MMA
              and the saved state (TMA), and Vn, Vd from U (every thread one row half of its columns)
  warps 8-11  RED: reduce R into W (thread t < 64: K row t) and the readout (t >= 64: Q row t-64); A diag(beta),
              Pm; the output
  warp 12     MMA issuer (order per chunk: R, U, dS, O)
  warp 13     TMA loads of K, Q (two stages), A (one stage) and Pm (into the A buffer)
  warps 14-15 gate tables of each chunk (two stages)
Tensor memory: R and dS share columns [0, N); U at 320, O at 384 (64 columns each).
Saves the state at the start of every chunk, W and U = Vd in BF16 for the backward, like sm90_joint_fwd_split.
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90          # smem layout helpers only (plain swizzled layouts)
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from .sm100_tc import (idesc, mma128, commit, fence_before, fence_after, fence_async, wait_ld, ld32, f32_of_i32,
                       smem_desc, off64, mbar_arrive, mbar_wait, pack2, unpack_lo, unpack_hi, sts128, stg128,
                       ldg128, saddr, swz_bytes, SW128, SW64, SW32)
from .sm90_joint_packed import token_chunk, valid_rows, keep_singleton_tma_axis

C, KD, DV = 64, 128, 128
NT = 512
T_R, T_U, T_O = 0, 320, 384
NREG_S, NREG_R, NREG_M = 184, 104, 40        # registers per thread: STATE, RED, the other warpgroup
# mbarrier slots
(KQ_FULL, KQ_EMPTY, GT_FULL, GT_EMPTY, AM_FULL, SH_FULL, R_FULL, R_EMPTY, U_GO, U_FULL, VN_FULL, DS_FULL,
 PM_READY, O_FULL) = (0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
NBAR = 18


def _adv128k(ki):     # K-major tile of >= 128 rows, K = 128: two K-blocks of 128 rows (16 KB)
    return ((ki >> 2) * 16384 + (ki & 3) * 32) >> 4


def _adv64k(ki):      # K-major tile with K = 64: one K-block
    return (ki * 32) >> 4


def _advmn(ki):       # MN-major, 128 B rows: 16 K-rows per step
    return (ki * 2048) >> 4


def _bar(sbar, k):
    return cutlass.Int32((sbar.iterator + k).toint())


@cute.kernel
def fwd_kernel(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor, gV: cute.Tensor,
               gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor,
               tmaK: cute.CopyAtom, tKt: cute.Tensor, tmaQ: cute.CopyAtom, tQt: cute.Tensor,
               tmaA: cute.CopyAtom, tAt: cute.Tensor, tmaM: cute.CopyAtom, tMt: cute.Tensor,
               tmaH: cute.CopyAtom, tHt: cute.Tensor, lBox: cute.ComposedLayout, lH: cute.ComposedLayout,
               gOffsets: cute.Tensor, mapping: cute.Tensor,
               H: cutlass.Constexpr, E: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr,
               SAVE: cutlass.Constexpr, SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    N = E * VC
    NH = N // 2           # state columns per STATE warpgroup
    NQ = NH // 2          # Vn columns per STATE thread (one row half)
    NS = N if N >= 64 else 64   # the dS MMA spans whole 64-column MN atoms of [Vn]; columns >= N stay zero
    NV = DV // VC
    SWB = 1 if VC == 16 else (2 if VC == 32 else 3)
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    vb, h, b = bid % NV, (bid // NV) % H, bid // (NV * H)
    first, count = cutlass.Int32(0), cutlass.Int32(NC)
    if cutlass.const_expr(DOCS):
        # packed documents: one recurrence per document over its chunks [first, first + count) of the padded
        # chunk space (gc, A, Pm, the saved h, W, Vd); raw token tensors are read through token_chunk
        if cutlass.const_expr(NATIVE):
            b = cutlass.Int32(mapping[2, b])      # documents longest first (see metadata_kernel)
        first = cutlass.Int32(gOffsets[b])
        count = cutlass.Int32(gOffsets[b + 1]) - first
        b = cutlass.Int32(0)
    bf = cutlass.BFloat16
    f32 = cutlass.Float32

    smem = utils.SmemAllocator()
    sKQ = smem.allocate_tensor(bf, cute.make_layout((2 * 128 * KD,)), byte_alignment=1024)   # [K; Q] x 2 stages
    sA3 = smem.allocate_tensor(bf, cute.make_layout((192 * C,)), byte_alignment=1024)        # [Ab; Ab; Pm]
    sH = smem.allocate_tensor(bf, cute.make_layout((KD * N,)), byte_alignment=1024)          # S (bf16), per slice
    sVn = smem.allocate_tensor(bf, cute.make_layout((C * NS,)), byte_alignment=1024)         # [Vn_1 .. Vn_E]
    sW = smem.allocate_tensor(bf, cute.make_layout((C * 64,)), byte_alignment=1024)          # W  (cols >= VC zero)
    sVd = smem.allocate_tensor(bf, cute.make_layout((C * 64,)), byte_alignment=1024)         # Vd (cols >= VC zero)
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
            cute.arch.mbarrier_init(sbar.iterator + SH_FULL, 256)
            cute.arch.mbarrier_init(sbar.iterator + R_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + R_EMPTY, 128)
            cute.arch.mbarrier_init(sbar.iterator + U_GO, 64)
            cute.arch.mbarrier_init(sbar.iterator + U_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + VN_FULL, 256)
            cute.arch.mbarrier_init(sbar.iterator + DS_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + PM_READY, 1)          # Pm TMA (tx)
            cute.arch.mbarrier_init(sbar.iterator + O_FULL, 1)
    cute.arch.mbarrier_init_fence()
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(thold.iterator, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(f32)
    tb = cutlass.Int32(tmem_ptr.toint())
    if cutlass.const_expr(VC < 64):                   # zero the padding columns [VC, 64) of the W/Vd tile once
        z = cutlass.Int32(0)
        r0, c80 = tid // 8, (tid % 8) * 8
        if c80 >= VC:
            sts128(saddr(sW.iterator, off64(r0, c80)), z, z, z, z)
            sts128(saddr(sVd.iterator, off64(r0, c80)), z, z, z, z)
        if cutlass.const_expr(N < 64):
            if c80 >= N:
                sts128(saddr(sVn.iterator, off64(r0, c80)), z, z, z, z)
        fence_async()
    cute.arch.barrier()

    # ---------------------------------------------------------------- TMA plumbing
    one = cute.make_layout(1)
    kq_ptr = sKQ.iterator

    def box(ptr, off):
        return cute.make_tensor(cute.recast_ptr(ptr + off, lBox.inner, dtype=bf), lBox.outer)

    if cutlass.const_expr(NATIVE):
        # raw token view (T, KD, 1, 1, H) from the document's first token; chunk i is tile row i. Rows past the
        # document end hold the next document's tokens (or TMA zeros past T): their gates k2, q2, beta are
        # masked to zero, so they drop out of W, Vn and the readout; their outputs are not stored.
        start = mapping[0, first]
        tKt = cute.domain_offset((start, 0, 0, 0, 0), tKt)
        tQt = cute.domain_offset((start, 0, 0, 0, 0), tQt)
    gK = cute.group_modes(cute.local_tile(tKt, (C, 64), (None, None, None, None, None)), 0, 2)
    gQ = cute.group_modes(cute.local_tile(tQt, (C, 64), (None, None, None, None, None)), 0, 2)
    gA = cute.group_modes(cute.local_tile(tAt, (C, 64), (None, None, None, None, None)), 0, 2)
    gM = cute.group_modes(cute.local_tile(tMt, (C, 64), (None, None, None, None, None)), 0, 2)
    # stage st, K column block j at element st*16384 + j*8192, Q at + 4096
    dK00, tgK = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 0), 0, 2), gK)
    dK01, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 8192), 0, 2), gK)
    dK10, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 16384), 0, 2), gK)
    dK11, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(kq_ptr, 24576), 0, 2), gK)
    dQ00, tgQ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 4096), 0, 2), gQ)
    dQ01, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 12288), 0, 2), gQ)
    dQ10, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 20480), 0, 2), gQ)
    dQ11, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(kq_ptr, 28672), 0, 2), gQ)
    dA, tgA = cpasync.tma_partition(tmaA, 0, one, cute.group_modes(box(sA3.iterator, 0), 0, 2), gA)            # rows 0-63
    dA2, _ = cpasync.tma_partition(tmaA, 0, one, cute.group_modes(box(sA3.iterator, 64 * C), 0, 2), gA)       # rows 64-127
    dM, tgM = cpasync.tma_partition(tmaM, 0, one, cute.group_modes(box(sA3.iterator, 128 * C), 0, 2), gM)   # Pm -> rows 128-191
    sHv = cute.make_tensor(cute.recast_ptr(sH.iterator, lH.inner, dtype=bf), lH.outer)          # (VC, KD, E)
    tHs, tgH = cpasync.tma_partition(tmaH, 0, one, cute.group_modes(sHv, 0, 2),
                                     cute.group_modes(cute.local_tile(tHt, (VC, KD), (None, None, None, None, None)), 0, 2))

    if tid < 256:
        # ================================================================ STATE (two warpgroups)
        cute.arch.setmaxregister_increase(NREG_S)
        g = tid // 128
        t = tid % 128                    # key row
        col0 = g * NH
        r = t % 64                       # Vn row
        qc0 = col0 + (t // 64) * NQ      # first of this thread's NQ Vn columns
        S = cute.make_rmem_tensor((NH,), f32)
        u = cute.make_rmem_tensor((VC,), f32)
        for j in cutlass.range_constexpr(NH):
            S[j] = cutlass.Float32(0.)
        for i in cutlass.range(count, unroll=1):
            n = first + i
            st = i % 2
            if i > 0:
                stp = (i - 1) % 2
                mbar_wait(_bar(sbar, DS_FULL), (i - 1) & 1)
                fence_after()
                for s_ in cutlass.range_constexpr(NH // 16):
                    dec = sd[stp, (col0 + 16 * s_) // VC]
                    vals = ld32(tb + (T_R + col0 + 16 * s_), 16)
                    for j in cutlass.range_constexpr(16):
                        S[16 * s_ + j] = S[16 * s_ + j] * dec + f32_of_i32(vals[j])
                wait_ld()
                mbar_arrive(_bar(sbar, GT_EMPTY + stp))
                if cutlass.const_expr(SAVE):
                    if tid < 32:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)     # last chunk's state save has read sH
                    cute.arch.barrier(barrier_id=2, number_of_threads=256)
            # BF16 copy of S_[i] -> sH (row t of every slice this warpgroup holds), then the saved state (TMA)
            hbase = cutlass.Int32(sH.iterator.toint())
            for c8 in cutlass.range_constexpr(NH // 8):
                cc = 8 * c8
                col = col0 + cc
                boff = 2 * ((col % VC) + t * VC + (col // VC) * (VC * KD))
                sts128(hbase + swz_bytes(boff, SWB),
                       pack2(S[cc + 1], S[cc]), pack2(S[cc + 3], S[cc + 2]),
                       pack2(S[cc + 5], S[cc + 4]), pack2(S[cc + 7], S[cc + 6]))
            fence_async()
            fence_before()
            mbar_arrive(_bar(sbar, SH_FULL))
            if cutlass.const_expr(SAVE):
                cute.arch.barrier(barrier_id=3, number_of_threads=256)
                if tid < 32:
                    for e in cutlass.range_constexpr(E):
                        cute.copy(tmaH, tHs[(None, e)], tgH[(None, vb, e, b, n, h)])
                    cute.arch.cp_async_bulk_commit_group()
            # ---- Vn and Vd from U (row r, this thread's NQ columns)
            mbar_wait(_bar(sbar, U_FULL), i & 1)
            fence_after()
            for s_ in cutlass.range_constexpr(VC // 16):
                vals = ld32(tb + (T_U + 16 * s_), 16)
                for j in cutlass.range_constexpr(16):
                    u[16 * s_ + j] = f32_of_i32(vals[j])
            wait_ld()
            vbase = cutlass.Int32(sVn.iterator.toint())
            x = cute.make_rmem_tensor((8,), f32)
            if cutlass.const_expr(NQ >= VC):
                # whole slices: qc0 is a multiple of VC, so the U column of Vn column qc0 + 8*c8 + j is static
                for c8 in cutlass.range_constexpr(NQ // 8):
                    col = qc0 + 8 * c8
                    rw = sr[st, col // VC, r]
                    for j in cutlass.range_constexpr(8):
                        x[j] = rw * u[(8 * c8) % VC + j]
                    sts128(vbase + 2 * off64(r, col), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
            else:
                # part of one slice: pick the NQ U columns starting at qc0 % VC with static selects
                cb = qc0 % VC
                uq = cute.make_rmem_tensor((NQ,), f32)
                for j in cutlass.range_constexpr(NQ):
                    uq[j] = u[j]
                for opt in cutlass.range_constexpr(1, VC // NQ):
                    if cb == opt * NQ:
                        for j in cutlass.range_constexpr(NQ):
                            uq[j] = u[opt * NQ + j]
                rw = sr[st, qc0 // VC, r]
                for c8 in cutlass.range_constexpr(NQ // 8):
                    for j in cutlass.range_constexpr(8):
                        x[j] = rw * uq[8 * c8 + j]
                    sts128(vbase + 2 * off64(r, qc0 + 8 * c8), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
            if tid < 64:                    # Vd = U, row r
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    sts128(saddr(sVd.iterator, off64(r, cc)), pack2(u[cc + 1], u[cc]), pack2(u[cc + 3], u[cc + 2]),
                           pack2(u[cc + 5], u[cc + 4]), pack2(u[cc + 7], u[cc + 6]))
            fence_async()
            fence_before()
            mbar_arrive(_bar(sbar, VN_FULL))
            if cutlass.const_expr(SAVE):
                if tid < 64:
                    for c8 in cutlass.range_constexpr(VC // 8):
                        cc = 8 * c8
                        stg128((gVdo.iterator + gVdo.layout((b, n, r, h, vb * VC + cc))).toint(),
                               pack2(u[cc + 1], u[cc]), pack2(u[cc + 3], u[cc + 2]),
                               pack2(u[cc + 5], u[cc + 4]), pack2(u[cc + 7], u[cc + 6]))
        if cutlass.const_expr(SAVE):
            if tid < 32:
                cute.arch.cp_async_bulk_wait_group(0)
    elif tid < 384:
        # ================================================================ RED
        cute.arch.setmaxregister_decrease(NREG_R)
        t = tid - 256
        kside = t < 64
        r = t % 64
        acc = cute.make_rmem_tensor((VC,), f32)
        vv = cute.make_rmem_tensor((VC // 2,), cutlass.Int32)
        x = cute.make_rmem_tensor((8,), f32)
        for i in cutlass.range(count, unroll=1):
            n = first + i
            st = i % 2
            ph2 = (i // 2) & 1
            if kside:
                if cutlass.const_expr(NATIVE):
                    vch = token_chunk(gV, n, mapping, NATIVE)
                    for j in cutlass.range_constexpr(VC // 2):
                        vv[j] = cutlass.Int32(0)
                    if r < valid_rows(n, mapping, NATIVE):
                        for c8 in cutlass.range_constexpr(VC // 8):
                            w = ldg128((vch.iterator + vch.layout((b, n, r, h, vb * VC + 8 * c8))).toint())
                            for j in cutlass.range_constexpr(4):
                                vv[4 * c8 + j] = w[j]
                else:
                    for c8 in cutlass.range_constexpr(VC // 8):
                        w = ldg128((gV.iterator + gV.layout((b, n, r, h, vb * VC + 8 * c8))).toint())
                        for j in cutlass.range_constexpr(4):
                            vv[4 * c8 + j] = w[j]
                mbar_wait(_bar(sbar, GT_FULL + st), ph2)
                # W = V - sum_e a_e (K S_e); V (prefetched from global) is added after the reduction so its load
                # latency hides behind the wait for R
                for j in cutlass.range_constexpr(VC):
                    acc[j] = cutlass.Float32(0.)
                mbar_wait(_bar(sbar, R_FULL), i & 1)
                fence_after()
                for e in cutlass.range_constexpr(E):
                    wgt = -sa[st, e, r]
                    for s_ in cutlass.range_constexpr(VC // 16):
                        vals = ld32(tb + (T_R + e * VC + 16 * s_), 16)
                        for j in cutlass.range_constexpr(16):
                            acc[16 * s_ + j] = acc[16 * s_ + j] + wgt * f32_of_i32(vals[j])
                wait_ld()
                fence_before()
                mbar_arrive(_bar(sbar, R_EMPTY))
                mbar_arrive(_bar(sbar, GT_EMPTY + st))
                for j in cutlass.range_constexpr(VC // 2):
                    acc[2 * j] = unpack_lo(vv[j]) + acc[2 * j]
                    acc[2 * j + 1] = unpack_hi(vv[j]) + acc[2 * j + 1]
                bw = sb[st, r]                                   # U = A (diag(beta) W): row r of W scaled by beta_r
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    for j in cutlass.range_constexpr(8):
                        x[j] = bw * acc[cc + j]
                    sts128(saddr(sW.iterator, off64(r, cc)), pack2(x[1], x[0]), pack2(x[3], x[2]),
                           pack2(x[5], x[4]), pack2(x[7], x[6]))
                fence_async()
                mbar_arrive(_bar(sbar, U_GO))
                if cutlass.const_expr(SAVE):
                    for c8 in cutlass.range_constexpr(VC // 8):
                        cc = 8 * c8
                        stg128((gWo.iterator + gWo.layout((b, n, r, h, vb * VC + cc))).toint(),
                               pack2(acc[cc + 1], acc[cc]), pack2(acc[cc + 3], acc[cc + 2]),
                               pack2(acc[cc + 5], acc[cc + 4]), pack2(acc[cc + 7], acc[cc + 6]))
            else:
                # the previous chunk's output: O(i-1) is issued right after R(i)
                if i > 0:
                    mbar_wait(_bar(sbar, O_FULL), (i - 1) & 1)
                    fence_after()
                    for s_ in cutlass.range_constexpr(VC // 16):
                        vals = ld32(tb + (T_O + 16 * s_), 16)
                        for j in cutlass.range_constexpr(16):
                            acc[16 * s_ + j] = acc[16 * s_ + j] + f32_of_i32(vals[j])
                    wait_ld()
                    fence_before()
                    och = token_chunk(gO, n - 1, mapping, NATIVE)
                    if r < valid_rows(n - 1, mapping, NATIVE):
                        for c8 in cutlass.range_constexpr(VC // 8):
                            cc = 8 * c8
                            stg128((och.iterator + och.layout((b, n - 1, r, h, vb * VC + cc))).toint(),
                                   pack2(acc[cc + 1], acc[cc]), pack2(acc[cc + 3], acc[cc + 2]),
                                   pack2(acc[cc + 5], acc[cc + 4]), pack2(acc[cc + 7], acc[cc + 6]))
                # ao = sum_e c_e (Q S_e)
                for j in cutlass.range_constexpr(VC):
                    acc[j] = cutlass.Float32(0.)
                mbar_wait(_bar(sbar, GT_FULL + st), ph2)
                mbar_wait(_bar(sbar, R_FULL), i & 1)
                fence_after()
                for e in cutlass.range_constexpr(E):
                    wgt = sc[st, e, r]
                    for s_ in cutlass.range_constexpr(VC // 16):
                        vals = ld32(tb + (T_R + e * VC + 16 * s_), 16)
                        for j in cutlass.range_constexpr(16):
                            acc[16 * s_ + j] = acc[16 * s_ + j] + wgt * f32_of_i32(vals[j])
                wait_ld()
                fence_before()
                mbar_arrive(_bar(sbar, R_EMPTY))
                mbar_arrive(_bar(sbar, GT_EMPTY + st))
        if not kside:
            mbar_wait(_bar(sbar, O_FULL), (count - 1) & 1)
            fence_after()
            for s_ in cutlass.range_constexpr(VC // 16):
                vals = ld32(tb + (T_O + 16 * s_), 16)
                for j in cutlass.range_constexpr(16):
                    acc[16 * s_ + j] = acc[16 * s_ + j] + f32_of_i32(vals[j])
            wait_ld()
            nl = first + count - 1
            och = token_chunk(gO, nl, mapping, NATIVE)
            if r < valid_rows(nl, mapping, NATIVE):
                for c8 in cutlass.range_constexpr(VC // 8):
                    cc = 8 * c8
                    stg128((och.iterator + och.layout((b, nl, r, h, vb * VC + cc))).toint(),
                           pack2(acc[cc + 1], acc[cc]), pack2(acc[cc + 3], acc[cc + 2]),
                           pack2(acc[cc + 5], acc[cc + 4]), pack2(acc[cc + 7], acc[cc + 6]))
    else:
        cute.arch.setmaxregister_decrease(NREG_M)
        if warp == 12:
            # ================================================================ MMA issuer
            I_R = idesc(128, N, 0, 1)
            I_U = idesc(128, 64, 0, 1)
            I_S = idesc(128, NS, 1, 1)
            akq = cutlass.Int32(kq_ptr.toint())
            HLAY = SW32 if VC == 16 else (SW64 if VC == 32 else SW128)
            dH = smem_desc(cutlass.Int32(sH.iterator.toint()), VC * KD * 2, 8 * VC * 2, HLAY)
            dVn = smem_desc(cutlass.Int32(sVn.iterator.toint()), 8192, 1024)
            a3 = cutlass.Int32(sA3.iterator.toint())
            dAU = smem_desc(a3, 16, 1024)                  # rows 0-127   [Ab; Ab]
            dAO = smem_desc(a3 + 8192, 16, 1024)           # rows 64-191  [Ab; Pm]
            dW = smem_desc(cutlass.Int32(sW.iterator.toint()), 0, 1024)
            dVd = smem_desc(cutlass.Int32(sVd.iterator.toint()), 0, 1024)
            for i in cutlass.range(count, unroll=1):
                st = i % 2
                ph2 = (i // 2) & 1
                base = akq + st * 32768
                dKQ = smem_desc(base, 16, 1024)
                dKt = smem_desc(base, 16384, 1024)
                mbar_wait(_bar(sbar, KQ_FULL + st), ph2)
                mbar_wait(_bar(sbar, SH_FULL), i & 1)
                fence_after()
                for ki in cutlass.range_constexpr(8):                                   # R = [K;Q] S
                    mma128(tb + T_R, dKQ + _adv128k(ki), dH + ((ki * 32 * VC) >> 4), cutlass.Int32(I_R),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, R_FULL))
                if i > 0:                                                               # O(i-1) = [Ab; Pm] Vd
                    mbar_wait(_bar(sbar, PM_READY), (i - 1) & 1)
                    fence_after()
                    for ki in cutlass.range_constexpr(4):
                        mma128(tb + T_O, dAO + _adv64k(ki), dVd + _advmn(ki), cutlass.Int32(I_U),
                               cutlass.Int32(1 if ki > 0 else 0))
                    commit(_bar(sbar, O_FULL))
                mbar_wait(_bar(sbar, U_GO), i & 1)
                mbar_wait(_bar(sbar, AM_FULL), i & 1)                                  # A of chunk i in [A; A]
                fence_after()
                for ki in cutlass.range_constexpr(4):                                   # U = [Ab; Ab] W
                    mma128(tb + T_U, dAU + _adv64k(ki), dW + _advmn(ki), cutlass.Int32(I_U),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, U_FULL))
                mbar_wait(_bar(sbar, R_EMPTY), i & 1)
                mbar_wait(_bar(sbar, VN_FULL), i & 1)
                fence_after()
                for ki in cutlass.range_constexpr(4):                                   # dS = K^T [Vn_e]
                    mma128(tb + T_R, dKt + _advmn(ki), dVn + _advmn(ki), cutlass.Int32(I_S),
                           cutlass.Int32(1 if ki > 0 else 0))
                commit(_bar(sbar, DS_FULL))
                commit(_bar(sbar, KQ_EMPTY + st))
            mbar_wait(_bar(sbar, PM_READY), (count - 1) & 1)                           # O of the last chunk
            fence_after()
            for ki in cutlass.range_constexpr(4):
                mma128(tb + T_O, dAO + _adv64k(ki), dVd + _advmn(ki), cutlass.Int32(I_U), cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, O_FULL))
        elif warp == 13:
            # ================================================================ TMA loader
            for i in cutlass.range(count, unroll=1):
                n = first + i
                if cutlass.const_expr(NATIVE):
                    kr, kn = i, 0                       # tile row i of the document's token view
                else:
                    kr, kn = 0, n
                st = i % 2
                if i >= 2:
                    mbar_wait(_bar(sbar, KQ_EMPTY + st), ((i // 2) - 1) & 1)
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
                if i >= 1:
                    mbar_wait(_bar(sbar, U_FULL), (i - 1) & 1)          # U(i-1) has read [A; A]
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + AM_FULL, 2 * C * C * 2)
                cute.copy(tmaA, tgA[(None, 0, 0, b, n, h)], dA, tma_bar_ptr=sbar.iterator + AM_FULL)
                cute.copy(tmaA, tgA[(None, 0, 0, b, n, h)], dA2, tma_bar_ptr=sbar.iterator + AM_FULL)
                if i >= 1:
                    mbar_wait(_bar(sbar, O_FULL), (i - 1) & 1)          # O(i-1) has read the Pm rows
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + PM_READY, C * C * 2)
                cute.copy(tmaM, tgM[(None, 0, 0, b, n, h)], dM, tma_bar_ptr=sbar.iterator + PM_READY)
        elif warp >= 14:
            # ================================================================ gate tables (vector loads)
            u_ = tid - 448          # chunk row
            EV = 4 if E >= 4 else E
            kv = cute.make_rmem_tensor((EV,), f32)
            qv = cute.make_rmem_tensor((EV,), f32)
            for i in cutlass.range(count, unroll=1):
                n = first + i
                st = i % 2
                rows = valid_rows(n, mapping, NATIVE)
                k2c = token_chunk(gK2, n, mapping, NATIVE)
                q2c = token_chunk(gQ2, n, mapping, NATIVE)
                btc = token_chunk(gBt, n, mapping, NATIVE)
                if i >= 2:
                    mbar_wait(_bar(sbar, GT_EMPTY + st), ((i // 2) - 1) & 1)
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
def launch_fwd(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor, gV: cute.Tensor,
               gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor,
               gKT: cute.Tensor, gQT: cute.Tensor, gAT: cute.Tensor, gMT: cute.Tensor, gHT: cute.Tensor,
               gOffsets: cute.Tensor, mapping: cute.Tensor,
               H: cutlass.Constexpr, E: cutlass.Constexpr, VC: cutlass.Constexpr,
               NC: cutlass.Constexpr, SAVE: cutlass.Constexpr, SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr,
               NATIVE: cutlass.Constexpr, NBLK: cutlass.Int32, stream: cuda.CUstream):
    bf = cutlass.BFloat16
    lBox = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, 64, 64), bf, 1), (None, None, 0))
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKT, lBox, (C, 64), 1)
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQT, lBox, (C, 64), 1)
    tmaA, tAt = cpasync.make_tiled_tma_atom(op, gAT, lBox, (C, 64), 1)
    tmaM, tMt = cpasync.make_tiled_tma_atom(op, gMT, lBox, (C, 64), 1)
    lH = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, KD), bf, E)
    tmaH, tHt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), gHT, cute.slice_(lH, (None, None, 0)), (VC, KD), 1)
    fwd_kernel(gGc, gK2, gQ2, gBt, gV, gO, gHs, gWo, gVdo, tmaK, tKt, tmaQ, tQt, tmaA, tAt, tmaM, tMt, tmaH, tHt,
               lBox, lH, gOffsets, mapping, H, E, VC, NC, SAVE, SCALE, DOCS, NATIVE).launch(
        grid=(NBLK * H * (DV // VC), 1, 1), block=(NT, 1, 1), stream=stream, min_blocks_per_mp=1)


_compiled = {}
_SMS = {}


def _value_cols(rows, H, E, dev):
    """Value-block width: 32 columns (64 at E = 1); 16 when too few CTAs would fill about 70% of the SMs."""
    if E == 1:
        return 64
    if E >= 12:
        return 16
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    if rows * H * (DV // 32) < 0.7 * _SMS[dev]:
        return 16
    return 32


def joint_fwd_sm100(q, k, v, k2, q2, gc, beta, A, Mp, scale=None, value_cols=None, save=False,
                    chunk_starts=None, token_map=None):
    """Same contract as sm90_joint_fwd_split.joint_fwd_split, with Mp = the Pm tile of joint_masks_sm100.

    Batch rows (T % 64 == 0) or packed documents (B = 1): chunk_starts holds every document's first chunk and the
    total; with token_map the raw token tensors are read in place (documents of any length).
    Returns o, or (o, states, W, Vd) when save=True."""
    B, T, H, D_ = q.shape
    E = k2.shape[-1]
    native = token_map is not None
    assert D_ == KD and v.shape[-1] == DV and (native or T % C == 0)
    docs = chunk_starts.numel() - 1 if chunk_starts is not None else 0
    assert not docs or B == 1
    dev = q.device
    VC = value_cols if value_cols is not None else _value_cols(docs if docs else B, H, E, dev)
    assert VC in (16, 32, 64) and (E * VC) % 32 == 0 and E * VC <= 256, (E, VC)
    NC = gc.shape[1] // C
    scale = KD ** -0.5 if scale is None else float(scale)
    A, Mp = A.contiguous(), Mp.contiguous()
    o = torch.empty(B, T, H, DV, device=dev, dtype=torch.bfloat16)
    hs = torch.empty((B, NC, H, E, KD, DV), device=dev, dtype=torch.bfloat16) if save else o
    wo, vdo = (torch.empty((B, NC * C, H, DV), device=dev, dtype=torch.bfloat16) for _ in range(2)) if save else (o, o)

    def ch(t):              # padded chunk space
        return t.view(B, NC, C, *t.shape[2:])

    def raw(t):             # token tensors: (B, 1, T, ...) read through token_chunk when packed
        return t.view(B, 1, T, *t.shape[2:]) if native else ch(t)

    def mat(t):
        return t.view(B, NC, C, H, C).permute(2, 4, 0, 1, 3)

    tl = (ch(gc), raw(k2), raw(q2), raw(beta), raw(v), raw(o),
          hs if save else raw(o), ch(wo) if save else raw(o), ch(vdo) if save else raw(o),
          raw(k).permute(2, 4, 0, 1, 3), raw(q).permute(2, 4, 0, 1, 3), mat(A), mat(Mp),
          hs.view(B, NC, H, E * KD, DV).permute(4, 3, 0, 1, 2) if save
          else torch.empty(1, 1, 1, 1, 1, device=dev, dtype=torch.bfloat16).expand(DV, E * KD, B, NC, H))
    args = [from_dlpack(t.detach(), assumed_align=16) for t in tl]
    if native and T == 1 and H == 1:
        for index in (9, 10):
            keep_singleton_tma_axis(args[index], 0)
    args.append(from_dlpack(chunk_starts.detach(), assumed_align=16).mark_layout_dynamic() if docs else args[0])
    args.append(from_dlpack(token_map.detach(), assumed_align=16) if native else args[0])
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    key = (B, T, H, E, VC, bool(save), scale, bool(docs), native, NC)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_fwd, *args, H, E, VC, NC, bool(save), scale, bool(docs), native,
                                      docs if docs else B, stream)
    _compiled[key](*args, docs if docs else B, stream)      # NBLK is a runtime argument
    return (o, hs, wo, vdo) if save else o
