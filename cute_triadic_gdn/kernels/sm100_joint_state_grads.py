"""Blackwell (sm100/sm103) state-gradient kernel of the Triadic GDN backward (the second kernel of
sm90_joint_bwd_recurrent): persistent, one CTA per SM walking (chunk, head) items, tcgen05 MMAs into tensor memory.

Per item, every slice e contributes (h_e = state at the chunk start, G_e = dS at the chunk end, both KD x DV):
    T1 = [dW; dO] h_e^T      rows 0-63: dW h_e^T (dK),  rows 64-127: dO h_e^T (dQ)
    T2 = [Vd; dW] G_e^T      rows 0-63: Vd G_e^T (dK)
    dKp += r_e T2 - a_e T1[0:64],   dQp += c_e T1[64:128]
    d1 = -rowsum(T1[0:64] * K),  d2 = rowsum(T2 * K),  d3 = rowsum(T1[64:] * Q),  dot = sum(h_e * G_e)
and the per-row second-key / decay terms follow from d1, d2, d3, dot exactly as in sm90 (dgv, dk2v, dgo, dq2o).
The two A operands are rows 64-191 and rows 0-127 of one 192-row tile [Vd; dW; dO].
Roles (576 threads): warps 0-15 epilogue (thread t: accumulator row t % 128, column quarter t // 128), warp 16 MMA,
warp 17 TMA.  h_e / G_e stream through a two-stage ring; tensor memory is double-buffered across slices.
Packed documents: the items are the chunks of the padded chunk space up to its device-side total; dW (the BF16 dV of
the reverse recurrence), dO, K and Q are read in place from each chunk's first token.  Rows past a document end hold
the next document's tokens: every per-row output is multiplied by a masked gate or zeroed there, as on sm90.
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
                       lds128, saddr)
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, keep_singleton_tma_axis

C, KD, DV = 64, 128, 128
NT = 576          # 16 epilogue warps (4 per tensor-memory lane quadrant), MMA warp 16, TMA warp 17
NE = 512
# A3: the [Vd; dW; dO] tile (read by the MMAs); KQ: the K and Q tiles (read by the epilogue).  Separate barriers let
# the next item's A3 tile and first h/G stages load while the epilogue finishes the current item.
(A3_FULL, A3_EMPTY, KQ_FULL, KQ_EMPTY, HD_FULL, HD_EMPTY, TM_FULL, TM_EMPTY) = (0, 1, 2, 3, 4, 6, 8, 10)
NBAR = 12


def _adv192k(ki):     # K-major 192-row tile, K = 128: K-blocks of 192 rows (24 KB)
    return ((ki >> 2) * 24576 + (ki & 3) * 32) >> 4


def _adv128k(ki):
    return ((ki >> 2) * 16384 + (ki & 3) * 32) >> 4


def _bar(sbar, k):
    return cutlass.Int32((sbar.iterator + k).toint())


@cute.kernel
def sg_kernel(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor,
              gDqp: cute.Tensor, gDkp: cute.Tensor, gDgv: cute.Tensor, gDk2v: cute.Tensor, gDgo: cute.Tensor,
              gDq2o: cute.Tensor,
              tmaV: cute.CopyAtom, tVt: cute.Tensor, tmaW: cute.CopyAtom, tWt: cute.Tensor,
              tmaO: cute.CopyAtom, tOt: cute.Tensor, tmaK: cute.CopyAtom, tKt: cute.Tensor,
              tmaQ: cute.CopyAtom, tQt: cute.Tensor, tmaH: cute.CopyAtom, tHt: cute.Tensor,
              tmaG: cute.CopyAtom, tGt: cute.Tensor, lBox: cute.ComposedLayout, lBox128: cute.ComposedLayout,
              gOffsets: cute.Tensor, mapping: cute.Tensor,
              H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr, SCALE: cutlass.Constexpr,
              NITEMS: cutlass.Constexpr, NBLK: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bf = cutlass.BFloat16
    f32 = cutlass.Float32
    nitems = cutlass.Int32(NITEMS)
    if cutlass.const_expr(DOCS):
        nitems = cutlass.Int32(gOffsets[cute.size(gOffsets) - 1]) * H     # chunks in use x heads
    nloc = (nitems - bid + NBLK - 1) // NBLK

    smem = utils.SmemAllocator()
    sA3 = smem.allocate_tensor(bf, cute.make_layout((192 * DV,)), byte_alignment=1024)        # [Vd; dW; dO]
    sK = smem.allocate_tensor(bf, cute.make_layout((C * KD,)), byte_alignment=1024)
    sQ = smem.allocate_tensor(bf, cute.make_layout((C * KD,)), byte_alignment=1024)
    sHG = smem.allocate_tensor(bf, cute.make_layout((2 * 2 * KD * DV,)), byte_alignment=1024)   # (h, G) x 2 stages
    # quarter partial sums of the row sums, double-buffered by slice.  E >= 16 needs the compact form (quarters 1-3
    # only, quarter 0 keeps its own in registers; no d2 slot for Q rows: 4.5 KB instead of 8) to fit in shared memory;
    # below that the plain form is faster
    COMPACT = E >= 16
    if cutlass.const_expr(COMPACT):
        sXk = smem.allocate_tensor(f32, cute.make_layout((2, 2, 3, 64), stride=(384, 192, 64, 1)), byte_alignment=16)
        sXq = smem.allocate_tensor(f32, cute.make_layout((2, 3, 64), stride=(192, 64, 1)), byte_alignment=16)
    else:
        sXc = smem.allocate_tensor(f32, cute.make_layout((2, 2, 4, 128), stride=(1024, 512, 128, 1)), byte_alignment=16)
    sDot = smem.allocate_tensor(f32, cute.make_layout((E, 16), stride=(16, 1)), byte_alignment=16)
    sRed = smem.allocate_tensor(f32, cute.make_layout((E, 2), stride=(2, 1)), byte_alignment=16)
    sT = smem.allocate_tensor(f32, cute.make_layout((3, E, C), stride=(E * C, C, 1)), byte_alignment=16)
    sbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NBAR,)), byte_alignment=8)
    thold = smem.allocate_tensor(cutlass.Int32, cute.make_layout((1,)), byte_alignment=16)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sbar.iterator + A3_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + A3_EMPTY, 1)
            cute.arch.mbarrier_init(sbar.iterator + KQ_FULL, 1)
            cute.arch.mbarrier_init(sbar.iterator + KQ_EMPTY, NE)
            for s in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(sbar.iterator + HD_FULL + s, 1)
                cute.arch.mbarrier_init(sbar.iterator + HD_EMPTY + s, 1 + NE)
                cute.arch.mbarrier_init(sbar.iterator + TM_FULL + s, 1)
                cute.arch.mbarrier_init(sbar.iterator + TM_EMPTY + s, NE)
    cute.arch.mbarrier_init_fence()
    tmem = utils.TmemAllocator(thold.iterator, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=NT),
                               allocator_warp_id=0)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(f32)
    tb = cutlass.Int32(tptr.toint())
    cute.arch.barrier()

    one = cute.make_layout(1)

    def box(ptr, off, lay):
        return cute.make_tensor(cute.recast_ptr(ptr + off, lay.inner, dtype=bf), lay.outer)

    def gtile(t, tile):
        return cute.group_modes(cute.local_tile(t, tile, (None,) * cute.rank(t)), 0, 2)

    gV, gW, gO = gtile(tVt, (C, 64)), gtile(tWt, (C, 64)), gtile(tOt, (C, 64))
    gK, gQ = gtile(tKt, (C, 64)), gtile(tQt, (C, 64))
    gH, gG = gtile(tHt, (128, 64)), gtile(tGt, (128, 64))
    a3 = sA3.iterator
    # [Vd; dW; dO]: tile j, column block kb at element 12288*kb + 4096*j
    dV0, tgV = cpasync.tma_partition(tmaV, 0, one, cute.group_modes(box(a3, 0, lBox), 0, 2), gV)
    dV1, _ = cpasync.tma_partition(tmaV, 0, one, cute.group_modes(box(a3, 12288, lBox), 0, 2), gV)
    dW0, tgW = cpasync.tma_partition(tmaW, 0, one, cute.group_modes(box(a3, 4096, lBox), 0, 2), gW)
    dW1, _ = cpasync.tma_partition(tmaW, 0, one, cute.group_modes(box(a3, 16384, lBox), 0, 2), gW)
    dO0, tgO = cpasync.tma_partition(tmaO, 0, one, cute.group_modes(box(a3, 8192, lBox), 0, 2), gO)
    dO1, _ = cpasync.tma_partition(tmaO, 0, one, cute.group_modes(box(a3, 20480, lBox), 0, 2), gO)
    dK0, tgK = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(sK.iterator, 0, lBox), 0, 2), gK)
    dK1, _ = cpasync.tma_partition(tmaK, 0, one, cute.group_modes(box(sK.iterator, 4096, lBox), 0, 2), gK)
    dQ0, tgQ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(sQ.iterator, 0, lBox), 0, 2), gQ)
    dQ1, _ = cpasync.tma_partition(tmaQ, 0, one, cute.group_modes(box(sQ.iterator, 4096, lBox), 0, 2), gQ)
    hg = sHG.iterator
    # stage s: h at element s*32768 (+ 8192 for its second column block), G at + 16384
    dH00, tgH = cpasync.tma_partition(tmaH, 0, one, cute.group_modes(box(hg, 0, lBox128), 0, 2), gH)
    dH01, _ = cpasync.tma_partition(tmaH, 0, one, cute.group_modes(box(hg, 8192, lBox128), 0, 2), gH)
    dG00, tgG = cpasync.tma_partition(tmaG, 0, one, cute.group_modes(box(hg, 16384, lBox128), 0, 2), gG)
    dG01, _ = cpasync.tma_partition(tmaG, 0, one, cute.group_modes(box(hg, 24576, lBox128), 0, 2), gG)
    dH10, _ = cpasync.tma_partition(tmaH, 0, one, cute.group_modes(box(hg, 32768, lBox128), 0, 2), gH)
    dH11, _ = cpasync.tma_partition(tmaH, 0, one, cute.group_modes(box(hg, 40960, lBox128), 0, 2), gH)
    dG10, _ = cpasync.tma_partition(tmaG, 0, one, cute.group_modes(box(hg, 49152, lBox128), 0, 2), gG)
    dG11, _ = cpasync.tma_partition(tmaG, 0, one, cute.group_modes(box(hg, 57344, lBox128), 0, 2), gG)
    smW = cute.group_modes(box(a3, 4096, lBox), 0, 2)          # first boxes of dW, dO, K, Q (packed re-partition)
    smO = cute.group_modes(box(a3, 8192, lBox), 0, 2)
    smK = cute.group_modes(box(sK.iterator, 0, lBox), 0, 2)
    smQ = cute.group_modes(box(sQ.iterator, 0, lBox), 0, 2)

    if tid < NE:
        # ================================================================ epilogue
        t = tid % 128            # accumulator row (tensor-memory lane)
        cq = tid // 128          # column quarter
        kside = t < 64
        r = t % 64
        acc = cute.make_rmem_tensor((32,), f32)       # dK or dQ partial, row t, columns [32 cq, 32 cq + 32)
        dd = cute.make_rmem_tensor((2 * E,), f32)     # quarter 0: d1_e, d2_e (K rows) or d3_e (Q rows)
        own = cute.make_rmem_tensor((2,), f32)        # this thread's quarter of the current slice's row sums
        src = sK.iterator if kside else sQ.iterator
        qd = t // 32             # lane quadrant: warps qd, qd + 4, qd + 8, qd + 12 hold the four quarters of its rows
        EV = 4 if E >= 4 else E
        # gate vectors: layout and register buffers made here, outside every branch (a layout first made inside
        # one branch and reused in another fails IR dominance)
        lev = cute.make_layout((EV,))
        gvr = cute.make_rmem_tensor((EV,), f32)
        lvr = cute.make_rmem_tensor((EV,), f32)
        xv = cute.make_rmem_tensor((EV,), f32)
        for li in cutlass.range(nloc, unroll=1):
            item = bid + li * NBLK
            h = item % H
            n = (item // H) % NC
            b = item // (H * NC)
            # per-row gate tables of this item: a_e, r_e (K rows), c_e (Q rows).  Vector loads, all independent
            rows = valid_rows(n, mapping, NATIVE)
            if cq == 0:
                k2c = token_chunk(gK2, n, mapping, NATIVE)
                q2c = token_chunk(gQ2, n, mapping, NATIVE)
                for e0 in cutlass.range_constexpr(0, E, EV):
                    gvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, r, h, e0))).align(4 * EV), lev).load())
                    lvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, C - 1, h, e0))).align(4 * EV), lev).load())
                    xv.fill(0.)
                    if (r < rows) | cutlass.const_expr(not NATIVE):
                        if kside:
                            xv.store(cute.make_tensor((k2c.iterator + k2c.layout((b, n, r, h, e0))).align(4 * EV), lev).load())
                        else:
                            xv.store(cute.make_tensor((q2c.iterator + q2c.layout((b, n, r, h, e0))).align(4 * EV), lev).load())
                    for j in cutlass.range_constexpr(EV):
                        ex = cute.math.exp(gvr[j], fastmath=True)
                        if kside:
                            sT[0, e0 + j, r] = xv[j] * ex
                            sT[1, e0 + j, r] = xv[j] * cute.math.exp(lvr[j] - gvr[j], fastmath=True)
                        else:
                            sT[2, e0 + j, r] = SCALE * xv[j] * ex
            for j in cutlass.range_constexpr(32):
                acc[j] = cutlass.Float32(0.)
            mbar_wait(_bar(sbar, KQ_FULL), li & 1)
            cute.arch.barrier(barrier_id=2, number_of_threads=NE)
            for e in cutlass.range_constexpr(E):
                slot = (li * E + e) % 2
                sph = ((li * E + e) // 2) & 1
                mbar_wait(_bar(sbar, TM_FULL + slot), sph)
                fence_after()
                if kside:
                    a_ = sT[0, e, r]
                    r_ = sT[1, e, r]
                    p1 = cute.make_rmem_tensor((4,), f32)
                    p2 = cute.make_rmem_tensor((4,), f32)
                    for j in cutlass.range_constexpr(4):
                        p1[j] = cutlass.Float32(0.)
                        p2[j] = cutlass.Float32(0.)
                    for s_ in cutlass.range_constexpr(2):
                        m0 = lds128(saddr(src, off64(r, 32 * cq + 16 * s_)))
                        m1 = lds128(saddr(src, off64(r, 32 * cq + 16 * s_ + 8)))
                        v1 = ld32(tb + (slot * 256 + 32 * cq + 16 * s_), 16)
                        v2 = ld32(tb + (slot * 256 + 128 + 32 * cq + 16 * s_), 16)
                        wait_ld()
                        for j in cutlass.range_constexpr(16):
                            jj = 16 * s_ + j
                            w_ = m0[j // 2] if j < 8 else m1[(j - 8) // 2]
                            m_ = unpack_lo(w_) if j % 2 == 0 else unpack_hi(w_)
                            t1 = f32_of_i32(v1[j])
                            t2 = f32_of_i32(v2[j])
                            p1[j % 4] = p1[j % 4] - t1 * m_
                            p2[j % 4] = p2[j % 4] + t2 * m_
                            acc[jj] = acc[jj] + r_ * t2 - a_ * t1
                    fence_before()
                    mbar_arrive(_bar(sbar, TM_EMPTY + slot))
                    if cutlass.const_expr(COMPACT):
                        own[0] = (p1[0] + p1[1]) + (p1[2] + p1[3])
                        own[1] = (p2[0] + p2[1]) + (p2[2] + p2[3])
                        if cq > 0:
                            sXk[e % 2, 0, cq - 1, r] = own[0]
                            sXk[e % 2, 1, cq - 1, r] = own[1]
                    else:
                        sXc[e % 2, 0, cq, t] = (p1[0] + p1[1]) + (p1[2] + p1[3])
                        sXc[e % 2, 1, cq, t] = (p2[0] + p2[1]) + (p2[2] + p2[3])
                else:
                    c_ = sT[2, e, r]
                    p3 = cute.make_rmem_tensor((4,), f32)
                    for j in cutlass.range_constexpr(4):
                        p3[j] = cutlass.Float32(0.)
                    for s_ in cutlass.range_constexpr(2):
                        m0 = lds128(saddr(src, off64(r, 32 * cq + 16 * s_)))
                        m1 = lds128(saddr(src, off64(r, 32 * cq + 16 * s_ + 8)))
                        v1 = ld32(tb + (slot * 256 + 32 * cq + 16 * s_), 16)
                        wait_ld()
                        for j in cutlass.range_constexpr(16):
                            jj = 16 * s_ + j
                            w_ = m0[j // 2] if j < 8 else m1[(j - 8) // 2]
                            m_ = unpack_lo(w_) if j % 2 == 0 else unpack_hi(w_)
                            t1 = f32_of_i32(v1[j])
                            p3[j % 4] = p3[j % 4] + t1 * m_
                            acc[jj] = acc[jj] + c_ * t1
                    fence_before()
                    mbar_arrive(_bar(sbar, TM_EMPTY + slot))
                    if cutlass.const_expr(COMPACT):
                        own[0] = (p3[0] + p3[1]) + (p3[2] + p3[3])
                        if cq > 0:
                            sXq[e % 2, cq - 1, r] = own[0]
                    else:
                        sXc[e % 2, 0, cq, t] = (p3[0] + p3[1]) + (p3[2] + p3[3])
                # dot(h_e, G_e): row t, columns [32 cq, 32 cq + 32) of the stage
                hs_ = (li * E + e) % 2
                mbar_wait(_bar(sbar, HD_FULL + hs_), ((li * E + e) // 2) & 1)
                base = hs_ * 32768
                q4 = cute.make_rmem_tensor((4,), f32)
                for j in cutlass.range_constexpr(4):
                    q4[j] = cutlass.Float32(0.)
                for c8 in cutlass.range_constexpr(4):
                    wh = lds128(saddr(hg, base + off128(t, 32 * cq + 8 * c8)))
                    wgg = lds128(saddr(hg, base + 16384 + off128(t, 32 * cq + 8 * c8)))
                    for j in cutlass.range_constexpr(4):
                        q4[j] = q4[j] + unpack_lo(wh[j]) * unpack_lo(wgg[j]) + unpack_hi(wh[j]) * unpack_hi(wgg[j])
                mbar_arrive(_bar(sbar, HD_EMPTY + hs_))
                dotp = cute.arch.warp_reduction_sum((q4[0] + q4[1]) + (q4[2] + q4[3]))
                if tid % 32 == 0:
                    sDot[e, tid // 32] = dotp                   # summed once per item, after the item barrier
                cute.arch.barrier(barrier_id=4 + qd, number_of_threads=128)     # this row's four quarter sums
                if cq == 0:
                    if cutlass.const_expr(COMPACT):
                        if kside:
                            dd[2 * e] = (own[0] + sXk[e % 2, 0, 0, r]) + (sXk[e % 2, 0, 1, r] + sXk[e % 2, 0, 2, r])
                            dd[2 * e + 1] = (own[1] + sXk[e % 2, 1, 0, r]) + (sXk[e % 2, 1, 1, r] + sXk[e % 2, 1, 2, r])
                        else:
                            dd[2 * e] = (own[0] + sXq[e % 2, 0, r]) + (sXq[e % 2, 1, r] + sXq[e % 2, 2, r])
                    else:
                        dd[2 * e] = (sXc[e % 2, 0, 0, t] + sXc[e % 2, 0, 1, t]) + (sXc[e % 2, 0, 2, t] + sXc[e % 2, 0, 3, t])
                        if kside:
                            dd[2 * e + 1] = (sXc[e % 2, 1, 0, t] + sXc[e % 2, 1, 1, t]) + (sXc[e % 2, 1, 2, t] + sXc[e % 2, 1, 3, t])
            mbar_arrive(_bar(sbar, KQ_EMPTY))
            # ---- per-row outputs and the dK / dQ partials
            cute.arch.barrier(barrier_id=2, number_of_threads=NE)
            if cq == 0:
                if kside:
                    for e in cutlass.range_constexpr(E):
                        tv = sT[1, e, r] * dd[2 * e + 1]
                        tv = cute.arch.warp_reduction_sum(tv)
                        if tid % 32 == 0:
                            sRed[e, tid // 32] = tv
                    cute.arch.barrier(barrier_id=3, number_of_threads=64)
                    for e0 in cutlass.range_constexpr(0, E, EV):
                        gvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, r, h, e0))).align(4 * EV), lev).load())
                        lvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, C - 1, h, e0))).align(4 * EV), lev).load())
                        for j in cutlass.range_constexpr(EV):
                            e = e0 + j
                            ex = cute.math.exp(gvr[j], fastmath=True)
                            er = cute.math.exp(lvr[j] - gvr[j], fastmath=True)
                            d1 = dd[2 * e]
                            d2 = dd[2 * e + 1]
                            vg = sT[0, e, r] * d1 - sT[1, e, r] * d2
                            if r == C - 1:
                                vg = vg + (sRed[e, 0] + sRed[e, 1])
                            gDgv[b, n, r, h, 0, e] = vg
                            vk = ex * d1 + er * d2
                            if cutlass.const_expr(NATIVE):
                                if r >= rows:
                                    vk = cutlass.Float32(0.)
                            gDk2v[b, n, r, h, 0, e] = vk
                else:
                    for e0 in cutlass.range_constexpr(0, E, EV):
                        gvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, r, h, e0))).align(4 * EV), lev).load())
                        lvr.store(cute.make_tensor((gGc.iterator + gGc.layout((b, n, C - 1, h, e0))).align(4 * EV), lev).load())
                        for j in cutlass.range_constexpr(EV):
                            e = e0 + j
                            ex = cute.math.exp(gvr[j], fastmath=True)
                            d3 = dd[2 * e]
                            vgo = sT[2, e, r] * d3
                            if r == C - 1:
                                dall = cutlass.Float32(0.)
                                for z in cutlass.range_constexpr(16):
                                    dall = dall + sDot[e, z]
                                vgo = vgo + cute.math.exp(lvr[j], fastmath=True) * dall
                            gDgo[b, n, r, h, 0, e] = vgo
                            vq = ex * (SCALE * d3)
                            if cutlass.const_expr(NATIVE):
                                if r >= rows:
                                    vq = cutlass.Float32(0.)
                            gDq2o[b, n, r, h, 0, e] = vq
            for c4 in cutlass.range_constexpr(8):
                cc = 4 * c4
                v4 = cute.make_rmem_tensor((4,), f32)
                for j in cutlass.range_constexpr(4):
                    v4[j] = acc[cc + j]
                if kside:
                    cute.make_tensor((gDkp.iterator + gDkp.layout((b, n, r, h, 0, 32 * cq + cc))).align(16),
                                     cute.make_layout((4,))).store(v4.load())
                else:
                    cute.make_tensor((gDqp.iterator + gDqp.layout((b, n, r, h, 0, 32 * cq + cc))).align(16),
                                     cute.make_layout((4,))).store(v4.load())
            cute.arch.barrier(barrier_id=2, number_of_threads=NE)      # tables / sDd / sDot / sRed reuse
    else:
        if warp == 16:
            # ================================================================ MMA issuer
            I_T = idesc(128, 128, 0, 0)
            aa3 = cutlass.Int32(a3.toint())
            d1A = smem_desc(aa3 + 8192, 16, 1024)          # rows 64-191  [dW; dO]
            d2A = smem_desc(aa3, 16, 1024)                 # rows 0-127   [Vd; dW]
            ahg = cutlass.Int32(hg.toint())
            for li in cutlass.range(nloc, unroll=1):
                mbar_wait(_bar(sbar, A3_FULL), li & 1)
                fence_after()
                for e in cutlass.range_constexpr(E):
                    k = li * E + e
                    s = k % 2
                    ph = (k // 2) & 1
                    mbar_wait(_bar(sbar, HD_FULL + s), ph)
                    if k >= 2:
                        mbar_wait(_bar(sbar, TM_EMPTY + s), ((k // 2) - 1) & 1)
                    fence_after()
                    dH = smem_desc(ahg + s * 65536, 16, 1024)
                    dG = smem_desc(ahg + s * 65536 + 32768, 16, 1024)
                    for ki in cutlass.range_constexpr(8):
                        mma128(tb + s * 256, d1A + _adv192k(ki), dH + _adv128k(ki), cutlass.Int32(I_T),
                               cutlass.Int32(1 if ki > 0 else 0))
                    for ki in cutlass.range_constexpr(8):
                        mma128(tb + s * 256 + 128, d2A + _adv192k(ki), dG + _adv128k(ki), cutlass.Int32(I_T),
                               cutlass.Int32(1 if ki > 0 else 0))
                    commit(_bar(sbar, TM_FULL + s))
                    commit(_bar(sbar, HD_EMPTY + s))
                commit(_bar(sbar, A3_EMPTY))
        elif warp == 17:
            # ================================================================ TMA loads
            for li in cutlass.range(nloc, unroll=1):
                item = bid + li * NBLK
                h = item % H
                n = (item // H) % NC
                b = item // (H * NC)
                if li > 0:
                    mbar_wait(_bar(sbar, A3_EMPTY), (li - 1) & 1)          # the last item's MMAs are done
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + A3_FULL, 3 * C * KD * 2)
                bar = sbar.iterator + A3_FULL
                cute.copy(tmaV, tgV[(None, 0, 0, b, n, h)], dV0, tma_bar_ptr=bar)
                cute.copy(tmaV, tgV[(None, 0, 1, b, n, h)], dV1, tma_bar_ptr=bar)
                if cutlass.const_expr(NATIVE):
                    # token tensors (T, D, 1, 1, H) from this chunk's first token
                    tok = mapping[0, n]
                    _, tgWn = cpasync.tma_partition(tmaW, 0, one, smW, gtile(cute.domain_offset((tok, 0, 0, 0, 0), tWt), (C, 64)))
                    _, tgOn = cpasync.tma_partition(tmaO, 0, one, smO, gtile(cute.domain_offset((tok, 0, 0, 0, 0), tOt), (C, 64)))
                    cute.copy(tmaW, tgWn[(None, 0, 0, b, 0, h)], dW0, tma_bar_ptr=bar)
                    cute.copy(tmaW, tgWn[(None, 0, 1, b, 0, h)], dW1, tma_bar_ptr=bar)
                    cute.copy(tmaO, tgOn[(None, 0, 0, b, 0, h)], dO0, tma_bar_ptr=bar)
                    cute.copy(tmaO, tgOn[(None, 0, 1, b, 0, h)], dO1, tma_bar_ptr=bar)
                else:
                    cute.copy(tmaW, tgW[(None, 0, 0, b, n, h)], dW0, tma_bar_ptr=bar)
                    cute.copy(tmaW, tgW[(None, 0, 1, b, n, h)], dW1, tma_bar_ptr=bar)
                    cute.copy(tmaO, tgO[(None, 0, 0, b, n, h)], dO0, tma_bar_ptr=bar)
                    cute.copy(tmaO, tgO[(None, 0, 1, b, n, h)], dO1, tma_bar_ptr=bar)
                for e in cutlass.range_constexpr(E):
                    k = li * E + e
                    s = k % 2
                    if k >= 2:
                        mbar_wait(_bar(sbar, HD_EMPTY + s), ((k // 2) - 1) & 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + HD_FULL + s, 2 * KD * DV * 2)
                    hb = sbar.iterator + HD_FULL + s
                    if s == 0:
                        cute.copy(tmaH, tgH[(None, e, 0, b, n, h)], dH00, tma_bar_ptr=hb)
                        cute.copy(tmaH, tgH[(None, e, 1, b, n, h)], dH01, tma_bar_ptr=hb)
                        cute.copy(tmaG, tgG[(None, e, 0, b, n, h)], dG00, tma_bar_ptr=hb)
                        cute.copy(tmaG, tgG[(None, e, 1, b, n, h)], dG01, tma_bar_ptr=hb)
                    else:
                        cute.copy(tmaH, tgH[(None, e, 0, b, n, h)], dH10, tma_bar_ptr=hb)
                        cute.copy(tmaH, tgH[(None, e, 1, b, n, h)], dH11, tma_bar_ptr=hb)
                        cute.copy(tmaG, tgG[(None, e, 0, b, n, h)], dG10, tma_bar_ptr=hb)
                        cute.copy(tmaG, tgG[(None, e, 1, b, n, h)], dG11, tma_bar_ptr=hb)
                    if cutlass.const_expr(e == 0):
                        # K, Q of this item once the epilogue has finished the last one
                        if li > 0:
                            mbar_wait(_bar(sbar, KQ_EMPTY), (li - 1) & 1)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + KQ_FULL, 2 * C * KD * 2)
                        kb_ = sbar.iterator + KQ_FULL
                        if cutlass.const_expr(NATIVE):
                            tok = mapping[0, n]
                            _, tgKn = cpasync.tma_partition(tmaK, 0, one, smK, gtile(cute.domain_offset((tok, 0, 0, 0, 0), tKt), (C, 64)))
                            _, tgQn = cpasync.tma_partition(tmaQ, 0, one, smQ, gtile(cute.domain_offset((tok, 0, 0, 0, 0), tQt), (C, 64)))
                            cute.copy(tmaK, tgKn[(None, 0, 0, b, 0, h)], dK0, tma_bar_ptr=kb_)
                            cute.copy(tmaK, tgKn[(None, 0, 1, b, 0, h)], dK1, tma_bar_ptr=kb_)
                            cute.copy(tmaQ, tgQn[(None, 0, 0, b, 0, h)], dQ0, tma_bar_ptr=kb_)
                            cute.copy(tmaQ, tgQn[(None, 0, 1, b, 0, h)], dQ1, tma_bar_ptr=kb_)
                        else:
                            cute.copy(tmaK, tgK[(None, 0, 0, b, n, h)], dK0, tma_bar_ptr=kb_)
                            cute.copy(tmaK, tgK[(None, 0, 1, b, n, h)], dK1, tma_bar_ptr=kb_)
                            cute.copy(tmaQ, tgQ[(None, 0, 0, b, n, h)], dQ0, tma_bar_ptr=kb_)
                            cute.copy(tmaQ, tgQ[(None, 0, 1, b, n, h)], dQ1, tma_bar_ptr=kb_)
    cute.arch.barrier()
    tmem.free(tptr, 512)


@cute.jit
def launch_sg(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gDqp: cute.Tensor, gDkp: cute.Tensor,
              gDgv: cute.Tensor, gDk2v: cute.Tensor, gDgo: cute.Tensor, gDq2o: cute.Tensor,
              gVT: cute.Tensor, gWT: cute.Tensor, gOT: cute.Tensor, gKT: cute.Tensor, gQT: cute.Tensor,
              gHT: cute.Tensor, gGT: cute.Tensor, gOffsets: cute.Tensor, mapping: cute.Tensor,
              H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr, SCALE: cutlass.Constexpr,
              NITEMS: cutlass.Constexpr, NBLK: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr,
              stream: cuda.CUstream):
    bf = cutlass.BFloat16
    lBox = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, 64, 64), bf, 1), (None, None, 0))
    lBox128 = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (128, 64, 64), bf, 1), (None, None, 0))
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaV, tVt = cpasync.make_tiled_tma_atom(op, gVT, lBox, (C, 64), 1)
    tmaW, tWt = cpasync.make_tiled_tma_atom(op, gWT, lBox, (C, 64), 1)
    tmaO, tOt = cpasync.make_tiled_tma_atom(op, gOT, lBox, (C, 64), 1)
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKT, lBox, (C, 64), 1)
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQT, lBox, (C, 64), 1)
    tmaH, tHt = cpasync.make_tiled_tma_atom(op, gHT, lBox128, (128, 64), 1)
    tmaG, tGt = cpasync.make_tiled_tma_atom(op, gGT, lBox128, (128, 64), 1)
    sg_kernel(gGc, gK2, gQ2, gDqp, gDkp, gDgv, gDk2v, gDgo, gDq2o, tmaV, tVt, tmaW, tWt, tmaO, tOt, tmaK, tKt,
              tmaQ, tQt, tmaH, tHt, tmaG, tGt, lBox, lBox128, gOffsets, mapping, H, E, NC, SCALE, NITEMS, NBLK, DOCS,
              NATIVE).launch(
        grid=(NBLK, 1, 1), block=(NT, 1, 1), stream=stream, min_blocks_per_mp=1)


_compiled = {}


def joint_state_grads_sm100(q, k, do, vd, dw_bf16, gc, k2, q2, h, ds, scale, out, chunk_starts=None, token_map=None):
    """out = (dqp, dkp, dgv, dk2v, dgo, dq2o): dqp, dkp (B,NC*64,H,1,D) f32, the others (B,NC*64,H,1,E) f32.
    dw_bf16 is dW rounded to bf16 (the dV output of the reverse recurrence, token layout).  Batch rows, or packed
    documents (B = 1) with chunk_starts / token_map as in joint_fwd_sm100."""
    B, T, H, D_ = q.shape
    E = k2.shape[-1]
    NC = gc.shape[1] // C
    native = token_map is not None
    docs = chunk_starts.numel() - 1 if chunk_starts is not None else 0
    assert not docs or B == 1
    dev = q.device
    dqp, dkp, dgv, dk2v, dgo, dq2o = out

    def ch(t):
        return t.view(B, NC, C, *t.shape[2:])

    def raw(t):
        return t.view(B, 1, T, *t.shape[2:]) if native else ch(t)

    def rows(t):          # (C, D, B, NC, H), or (T, D, B, 1, H) for token tensors of packed documents
        return t.permute(2, 4, 0, 1, 3)

    def states(t):        # (B, NC, H, E, KD, DV) -> (E*KD rows, DV cols, B, NC, H): TMA takes at most 5 modes;
        return t.reshape(B, NC, H, E * KD, DV).permute(3, 4, 0, 1, 2)     # slice e = row tile e of (128, 64) boxes

    nitems = B * NC * H
    nblk = min(nitems, torch.cuda.get_device_properties(dev).multi_processor_count)
    tl = (ch(gc), raw(k2), raw(q2), ch(dqp), ch(dkp), ch(dgv), ch(dk2v), ch(dgo), ch(dq2o),
          rows(ch(vd)), rows(raw(dw_bf16)), rows(raw(do)), rows(raw(k)), rows(raw(q)), states(h), states(ds))
    args = [from_dlpack(t.detach(), assumed_align=16) for t in tl]
    if native and T == 1 and H == 1:
        for index in (10, 11, 12, 13):
            keep_singleton_tma_axis(args[index], 0)
    args.append(from_dlpack(chunk_starts.detach(), assumed_align=16).mark_layout_dynamic() if docs else args[0])
    args.append(from_dlpack(token_map.detach(), assumed_align=16) if native else args[0])
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    key = (B, T, H, E, float(scale), nblk, bool(docs), native, NC)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_sg, *args, H, E, NC, float(scale), nitems, nblk, bool(docs), native,
                                      stream)
    _compiled[key](*args, stream)
