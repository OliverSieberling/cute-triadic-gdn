"""Blackwell (sm100/sm103) backward, second stage: every chunk in parallel.

The matrix kernel forms the chunk-local parts of dQ, dK, dbeta and the derivatives of the masks R' (dmp) and R (dm),
one CTA per (chunk, head), with tcgen05 MMAs into tensor memory; sm90_joint_bwd_parallel's mask-gradient kernel then
reduces them into the second-key, second-query and decay gradients.  Same arithmetic as sm90's matrix kernel:
    P = Q K^T, dP = dO Vd^T      dmp = scale tril(P * dP),     X = scale tril(R' * dP)
    dAb = dVd W^T                X2 = dAb diag(beta),           dbeta_col = colsum(dAb * A)
    Y = A^T X2, dp = Y A^T       dl = -strict_tril(dp),  dm = diag(beta) dl * (K K^T),  X3 = diag(beta) dl * R
                                 dbeta_row = rowsum(dl * (K K^T) * R)
    dQ = X K + dQ_state          dK = X^T Q + (X3 + X3^T) K + dK_state
Each 64x64 product is one M=128 MMA whose useful rows land in tensor-memory lanes 0-63 (the other 64 rows of the A
window are whatever follows in shared memory); warps 0-1 run those epilogues.  The output pair puts dQ in lanes 0-63
and dK in lanes 64-127: [X; Z] K with Z = X3 + X3^T (the two triangles are disjoint), then an MN-major A whose first
64 rows are a zero tile and whose last 64 are X^T, times Q, accumulated.
Shared memory: a 256-row stack of 64-column K-blocks (rows 0-63 Q, 64-127 K, 128-191 dO then dVd then R,
192-255 Vd then W), the R' tile (later X2, then Y), the A tile and [0; X; Z]: 104 KB, two CTAs per SM.
Packed documents: rows past a document end hold the next document's tokens; every output there is multiplied by a
masked gate except dmp, which is zeroed explicitly (as zero-filled operands would give).
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
                       i32_of_f32, smem_desc, mbar_wait, pack2, unpack_lo, unpack_hi, sts128, stg128, lds128, ldg128)
from .asm import _selp_f32
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, keep_singleton_tma_axis
from .sm90_joint_bwd_parallel import mask_grad_kernel
from cutlass._mlir.dialects import llvm

C, D, NT = 64, 128, 128
KB = 256 * 64                       # elements in one 64-column K-block of the row stack
R_Q, R_K, R_D, R_V = 0, 64, 128, 192
(L1, L2, L3, M1, M2, M3, M4, M5) = range(8)
NBAR = 8


def _sw(r, c):
    """Element offset of (row r, column c < 64) in a 128-byte-swizzled K-major tile of 64 columns."""
    return r * 64 + 8 * (cutlass.Int32(r % 8) ^ cutlass.Int32(c // 8)) + c % 8


def _adv2(ki):      # K-major operand, K = 128 over two K-blocks of the stack (32 KB apart)
    return ((ki >> 2) * (2 * KB) + (ki & 3) * 32) >> 4


def _adv1(ki):      # K-major operand, K = 64 (one K-block)
    return (ki * 32) >> 4


def _advmn(ki):     # MN-major operand: 16 K-rows of 128 bytes
    return (ki * 2048) >> 4


def _bar(sbar, k):
    return cutlass.Int32((sbar.iterator + k).toint())


def _sts16(addr, v32):
    """Store the low 16 bits of v32 at shared address addr."""
    llvm.inline_asm(None, [addr.ir_value(), v32.ir_value()],
                    "{\n.reg .b16 t;\ncvt.u16.u32 t, $1;\nst.shared.b16 [$0], t;\n}", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def _box(ptr, lBox):
    return cute.group_modes(cute.make_tensor(cute.recast_ptr(ptr, lBox.inner, dtype=cutlass.BFloat16), lBox.outer), 0, 2)


def _gt(t):
    return cute.group_modes(cute.local_tile(t, (C, 64), (None, None, None, None, None)), 0, 2)


@cute.kernel
def matrix_kernel(rawbeta: cute.Tensor, qp: cute.Tensor, kp: cute.Tensor,
                  rawdq: cute.Tensor, rawdk: cute.Tensor, rawdb: cute.Tensor, dmp: cute.Tensor, dm: cute.Tensor,
                  offsets: cute.Tensor, mapping: cute.Tensor,
                  tmaQ: cute.CopyAtom, tQt: cute.Tensor, tmaK: cute.CopyAtom, tKt: cute.Tensor,
                  tmaO: cute.CopyAtom, tOt: cute.Tensor, tmaV: cute.CopyAtom, tVt: cute.Tensor,
                  tmaDV: cute.CopyAtom, tDVt: cute.Tensor, tmaW: cute.CopyAtom, tWt: cute.Tensor,
                  tmaA: cute.CopyAtom, tAt: cute.Tensor, tmaMp: cute.CopyAtom, tMpt: cute.Tensor,
                  tmaM: cute.CopyAtom, tMt: cute.Tensor, lBox: cute.ComposedLayout,
                  H: cutlass.Constexpr, NC: cutlass.Constexpr, NV: cutlass.Constexpr, SCALE: cutlass.Constexpr,
                  PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bh, n, b = bid % H, (bid // H) % NC, bid // (H * NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets) - 1]
    if active:
        rows = valid_rows(n, mapping, NATIVE)
        beta = token_chunk(rawbeta, n, mapping, NATIVE)
        dqout = token_chunk(rawdq, n, mapping, NATIVE)
        dkout = token_chunk(rawdk, n, mapping, NATIVE)
        dbout = token_chunk(rawdb, n, mapping, NATIVE)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        bf = cutlass.BFloat16
        f32 = cutlass.Float32
        smem = utils.SmemAllocator()
        sS = smem.allocate_tensor(bf, cute.make_layout((2 * KB,)), byte_alignment=1024)      # row stack, 64 KB
        sMp = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)      # R', then X2, then Y
        sAm = smem.allocate_tensor(bf, cute.make_layout((C * C,)), byte_alignment=1024)      # A (UT inverse)
        sZX = smem.allocate_tensor(bf, cute.make_layout((3 * C * C,)), byte_alignment=1024)  # [0; X; Z]
        sBeta = smem.allocate_tensor(f32, cute.make_layout((C,)), byte_alignment=16)
        sDbp = smem.allocate_tensor(f32, cute.make_layout((2, C), stride=(C, 1)), byte_alignment=16)
        sbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NBAR,)), byte_alignment=8)
        thold = smem.allocate_tensor(cutlass.Int32, cute.make_layout((1,)), byte_alignment=16)
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(NBAR):
                    cute.arch.mbarrier_init(sbar.iterator + i, 1)
        cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(thold.iterator, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=NT),
                                   allocator_warp_id=0)
        tmem.allocate(128)
        if warp == 0:
            cute.arch.relinquish_tmem_alloc_permit()      # let the other CTAs on this SM allocate
        tmem.wait_for_alloc()
        tptr = tmem.retrieve_ptr(f32)
        tb = cutlass.Int32(tptr.toint())
        aS = cutlass.Int32(sS.iterator.toint())
        aMp = cutlass.Int32(sMp.iterator.toint())
        aAm = cutlass.Int32(sAm.iterator.toint())
        aZX = cutlass.Int32(sZX.iterator.toint())
        z = cutlass.Int32(0)
        for j in cutlass.range_constexpr(C * C // (NT * 8)):          # the zero tile
            sts128(aZX + 16 * (j * NT + tid), z, z, z, z)
        if tid < C:
            sBeta[tid] = masked_scalar(beta, (b, n, tid, bh), tid, rows)
        fence_async()
        cute.arch.barrier()

        # ---------------------------------------------------------------- TMA
        one = cute.make_layout(1)
        if cutlass.const_expr(NATIVE):
            tok = mapping[0, n]
            tQu = cute.domain_offset((tok, 0, 0, 0, 0), tQt)
            tKu = cute.domain_offset((tok, 0, 0, 0, 0), tKt)
            tOu = cute.domain_offset((tok, 0, 0, 0, 0), tOt)
            cn = 0
        else:
            tQu, tKu, tOu = tQt, tKt, tOt
            cn = n
        pS = sS.iterator
        dQ0, gQ = cpasync.tma_partition(tmaQ, 0, one, _box(pS + R_Q * 64, lBox), _gt(tQu))
        dQ1, _ = cpasync.tma_partition(tmaQ, 0, one, _box(pS + KB + R_Q * 64, lBox), _gt(tQu))
        dK0, gK = cpasync.tma_partition(tmaK, 0, one, _box(pS + R_K * 64, lBox), _gt(tKu))
        dK1, _ = cpasync.tma_partition(tmaK, 0, one, _box(pS + KB + R_K * 64, lBox), _gt(tKu))
        dO0, gO = cpasync.tma_partition(tmaO, 0, one, _box(pS + R_D * 64, lBox), _gt(tOu))
        dO1, _ = cpasync.tma_partition(tmaO, 0, one, _box(pS + KB + R_D * 64, lBox), _gt(tOu))
        dV0, gV = cpasync.tma_partition(tmaV, 0, one, _box(pS + R_V * 64, lBox), _gt(tVt))
        dV1, _ = cpasync.tma_partition(tmaV, 0, one, _box(pS + KB + R_V * 64, lBox), _gt(tVt))
        dD0, gDV = cpasync.tma_partition(tmaDV, 0, one, _box(pS + R_D * 64, lBox), _gt(tDVt))
        dD1, _ = cpasync.tma_partition(tmaDV, 0, one, _box(pS + KB + R_D * 64, lBox), _gt(tDVt))
        dW0, gW = cpasync.tma_partition(tmaW, 0, one, _box(pS + R_V * 64, lBox), _gt(tWt))
        dW1, _ = cpasync.tma_partition(tmaW, 0, one, _box(pS + KB + R_V * 64, lBox), _gt(tWt))
        dA, gA = cpasync.tma_partition(tmaA, 0, one, _box(sAm.iterator, lBox), _gt(tAt))
        dMp, gMp = cpasync.tma_partition(tmaMp, 0, one, _box(sMp.iterator, lBox), _gt(tMpt))
        dM, gM = cpasync.tma_partition(tmaM, 0, one, _box(pS + R_D * 64, lBox), _gt(tMt))
        I64 = cutlass.Int32(idesc(128, 64, 0, 0))
        if warp == 3:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + L1, (4 * C * D + 2 * C * C) * 2)
            bar = sbar.iterator + L1
            cute.copy(tmaQ, gQ[(None, 0, 0, b, cn, bh)], dQ0, tma_bar_ptr=bar)
            cute.copy(tmaQ, gQ[(None, 0, 1, b, cn, bh)], dQ1, tma_bar_ptr=bar)
            cute.copy(tmaK, gK[(None, 0, 0, b, cn, bh)], dK0, tma_bar_ptr=bar)
            cute.copy(tmaK, gK[(None, 0, 1, b, cn, bh)], dK1, tma_bar_ptr=bar)
            cute.copy(tmaO, gO[(None, 0, 0, b, cn, bh)], dO0, tma_bar_ptr=bar)
            cute.copy(tmaO, gO[(None, 0, 1, b, cn, bh)], dO1, tma_bar_ptr=bar)
            cute.copy(tmaV, gV[(None, 0, 0, b, n, bh)], dV0, tma_bar_ptr=bar)
            cute.copy(tmaV, gV[(None, 0, 1, b, n, bh)], dV1, tma_bar_ptr=bar)
            cute.copy(tmaA, gA[(None, 0, 0, b, n, bh)], dA, tma_bar_ptr=bar)
            cute.copy(tmaMp, gMp[(None, 0, 0, b, n, bh)], dMp, tma_bar_ptr=bar)
            # P = Q K^T -> columns [0, 64), dP = dO Vd^T -> [64, 128), both in lanes 0-63
            mbar_wait(_bar(sbar, L1), 0)
            fence_after()
            for ki in cutlass.range_constexpr(8):
                mma128(tb, smem_desc(aS + R_Q * 128, 16, 1024) + _adv2(ki), smem_desc(aS + R_K * 128, 16, 1024) + _adv2(ki),
                       I64, cutlass.Int32(1 if ki > 0 else 0))
            for ki in cutlass.range_constexpr(8):
                mma128(tb + 64, smem_desc(aS + R_D * 128, 16, 1024) + _adv2(ki), smem_desc(aS + R_V * 128, 16, 1024) + _adv2(ki),
                       I64, cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, M1))
            mbar_wait(_bar(sbar, M1), 0)                     # dO and Vd read: dVd and W take their rows
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + L2, 2 * C * D * 2)
            bar2 = sbar.iterator + L2
            cute.copy(tmaDV, gDV[(None, 0, 0, b, n, bh)], dD0, tma_bar_ptr=bar2)
            cute.copy(tmaDV, gDV[(None, 0, 1, b, n, bh)], dD1, tma_bar_ptr=bar2)
            cute.copy(tmaW, gW[(None, 0, 0, b, n, bh)], dW0, tma_bar_ptr=bar2)
            cute.copy(tmaW, gW[(None, 0, 1, b, n, bh)], dW1, tma_bar_ptr=bar2)

        vals = cute.make_rmem_tensor((16,), f32)
        wv = cute.make_rmem_tensor((16,), f32)
        x8 = cute.make_rmem_tensor((8,), f32)
        red = cute.make_rmem_tensor((32,), f32)
        r = tid % C
        # ---------------------------------------------------------------- epilogue 1: dmp, X
        mbar_wait(_bar(sbar, M1), 0)
        fence_after()
        if tid < C:
            for s_ in cutlass.range_constexpr(4):
                pv = ld32(tb + 16 * s_, 16)
                dv = ld32(tb + 64 + 16 * s_, 16)
                wait_ld()
                for h_ in cutlass.range_constexpr(2):
                    c0 = 16 * s_ + 8 * h_
                    mw = lds128(aMp + 2 * _sw(r, c0))
                    for j in cutlass.range_constexpr(8):
                        c = c0 + j
                        dpv = f32_of_i32(dv[8 * h_ + j])
                        mpv = unpack_lo(mw[j // 2]) if j % 2 == 0 else unpack_hi(mw[j // 2])
                        a_ = SCALE * f32_of_i32(pv[8 * h_ + j]) * dpv
                        x_ = SCALE * mpv * dpv
                        if c > r:
                            a_ = cutlass.Float32(0.)
                            x_ = cutlass.Float32(0.)
                        vals[8 * h_ + j] = a_
                        x8[j] = x_
                    sts128(aZX + 2 * (C * C + _sw(r, c0)), pack2(x8[1], x8[0]), pack2(x8[3], x8[2]),
                           pack2(x8[5], x8[4]), pack2(x8[7], x8[6]))
                if cutlass.const_expr(NATIVE):
                    if r >= rows:
                        for j in cutlass.range_constexpr(16):
                            vals[j] = cutlass.Float32(0.)
                for q4 in cutlass.range_constexpr(4):
                    cc = 16 * s_ + 4 * q4
                    stg128((dmp.iterator + dmp.layout((b, n, bh, r, cc))).toint(),
                           i32_of_f32(vals[4 * q4]), i32_of_f32(vals[4 * q4 + 1]),
                           i32_of_f32(vals[4 * q4 + 2]), i32_of_f32(vals[4 * q4 + 3]))
        fence_before()
        fence_async()
        cute.arch.barrier()

        # ---------------------------------------------------------------- dAb = dVd W^T, K K^T
        if warp == 3:
            mbar_wait(_bar(sbar, L2), 0)
            fence_after()
            for ki in cutlass.range_constexpr(8):
                mma128(tb, smem_desc(aS + R_D * 128, 16, 1024) + _adv2(ki), smem_desc(aS + R_V * 128, 16, 1024) + _adv2(ki),
                       I64, cutlass.Int32(1 if ki > 0 else 0))
            for ki in cutlass.range_constexpr(8):
                mma128(tb + 64, smem_desc(aS + R_K * 128, 16, 1024) + _adv2(ki), smem_desc(aS + R_K * 128, 16, 1024) + _adv2(ki),
                       I64, cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, M2))
            mbar_wait(_bar(sbar, M2), 0)                     # dVd read: R takes its rows (K-block 0)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator + L3, C * C * 2)
            cute.copy(tmaM, gM[(None, 0, 0, b, n, bh)], dM, tma_bar_ptr=sbar.iterator + L3)
        # ---------------------------------------------------------------- epilogue 2: X2, column sums of dAb * A
        mbar_wait(_bar(sbar, M2), 0)
        fence_after()
        if tid < C:
            for hh in cutlass.range_constexpr(2):
                for s_ in cutlass.range_constexpr(2):
                    c0 = 32 * hh + 16 * s_
                    dv = ld32(tb + c0, 16)
                    wait_ld()
                    for h_ in cutlass.range_constexpr(2):
                        c1 = c0 + 8 * h_
                        aw = lds128(aAm + 2 * _sw(r, c1))
                        for j in cutlass.range_constexpr(8):
                            c = c1 + j
                            d_ = f32_of_i32(dv[8 * h_ + j])
                            av = unpack_lo(aw[j // 2]) if j % 2 == 0 else unpack_hi(aw[j // 2])
                            x8[j] = d_ * sBeta[c]
                            red[16 * s_ + 8 * h_ + j] = d_ * av
                        sts128(aMp + 2 * _sw(r, c1), pack2(x8[1], x8[0]), pack2(x8[3], x8[2]),
                               pack2(x8[5], x8[4]), pack2(x8[7], x8[6]))
                # reduce-scatter over the warp's 32 rows: lane l ends with the sum of column 32 hh + l
                for st in cutlass.range_constexpr(5):
                    s = 16 >> st
                    up = cutlass.Int32(lane & s)
                    for j in cutlass.range_constexpr(s):
                        send = _selp_f32(red[j], red[j + s], up)
                        keep = _selp_f32(red[j + s], red[j], up)
                        red[j] = keep + cute.arch.shuffle_sync_bfly(send, s)
                sDbp[tid // 32, 32 * hh + lane] = red[0]
        fence_before()
        fence_async()
        cute.arch.barrier()
        # ---------------------------------------------------------------- Y = A^T X2
        IM = cutlass.Int32(idesc(128, 64, 1, 1))
        if warp == 3:
            fence_after()
            for ki in cutlass.range_constexpr(4):
                mma128(tb, smem_desc(aAm, 8192, 1024) + _advmn(ki), smem_desc(aMp, 8192, 1024) + _advmn(ki),
                       IM, cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, M3))
        mbar_wait(_bar(sbar, M3), 0)
        fence_after()
        if tid < C:                                         # Y (bf16) over X2
            for s_ in cutlass.range_constexpr(4):
                yv = ld32(tb + 16 * s_, 16)
                wait_ld()
                for h_ in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(8):
                        x8[j] = f32_of_i32(yv[8 * h_ + j])
                    sts128(aMp + 2 * _sw(r, 16 * s_ + 8 * h_), pack2(x8[1], x8[0]), pack2(x8[3], x8[2]),
                           pack2(x8[5], x8[4]), pack2(x8[7], x8[6]))
        fence_before()
        fence_async()
        cute.arch.barrier()
        # ---------------------------------------------------------------- dp = Y A^T
        if warp == 3:
            fence_after()
            for ki in cutlass.range_constexpr(4):
                mma128(tb, smem_desc(aMp, 16, 1024) + _adv1(ki), smem_desc(aAm, 16, 1024) + _adv1(ki),
                       I64, cutlass.Int32(1 if ki > 0 else 0))
            commit(_bar(sbar, M4))
        mbar_wait(_bar(sbar, M4), 0)
        mbar_wait(_bar(sbar, L3), 0)
        fence_after()
        # ---------------------------------------------------------------- epilogue 3: dm, Z, dbeta
        if tid < C:
            br = sBeta[r]
            rs = cutlass.Float32(0.)
            for s_ in cutlass.range_constexpr(4):
                pv = ld32(tb + 16 * s_, 16)
                kv = ld32(tb + 64 + 16 * s_, 16)
                wait_ld()
                for h_ in cutlass.range_constexpr(2):
                    c0 = 16 * s_ + 8 * h_
                    mw = lds128(aS + 2 * _sw(R_D + r, c0))
                    for j in cutlass.range_constexpr(8):
                        c = c0 + j
                        mv = unpack_lo(mw[j // 2]) if j % 2 == 0 else unpack_hi(mw[j // 2])
                        dl = cutlass.Float32(0.)
                        if c < r:
                            dl = -f32_of_i32(pv[8 * h_ + j])
                        kk = f32_of_i32(kv[8 * h_ + j])
                        vals[8 * h_ + j] = dl * br * kk
                        x8[j] = dl * br * mv
                        rs = rs + dl * kk * mv
                    sts128(aZX + 2 * (2 * C * C + _sw(r, c0)), pack2(x8[1], x8[0]), pack2(x8[3], x8[2]),
                           pack2(x8[5], x8[4]), pack2(x8[7], x8[6]))
                for q4 in cutlass.range_constexpr(4):
                    cc = 16 * s_ + 4 * q4
                    stg128((dm.iterator + dm.layout((b, n, bh, r, cc))).toint(),
                           i32_of_f32(vals[4 * q4]), i32_of_f32(vals[4 * q4 + 1]),
                           i32_of_f32(vals[4 * q4 + 2]), i32_of_f32(vals[4 * q4 + 3]))
            if r < rows:
                dbout[b, n, r, bh] = (sDbp[0, r] + sDbp[1, r]) + rs
            # Z = X3 + X3^T: the rows above were written with zeros above the diagonal; now each thread mirrors its
            # row's strictly lower part into its column
            cute.arch.barrier(barrier_id=2, number_of_threads=C)
            for c8 in cutlass.range_constexpr(8):
                zw = lds128(aZX + 2 * (2 * C * C + _sw(r, 8 * c8)))
                for j in cutlass.range_constexpr(8):
                    c = 8 * c8 + j
                    if c < r:
                        hv = zw[j // 2] if j % 2 == 0 else (zw[j // 2] >> 16)
                        _sts16(aZX + 2 * (2 * C * C + _sw(c, r)), hv)
        fence_before()
        fence_async()
        cute.arch.barrier()
        # ---------------------------------------------------------------- dQ, dK
        if warp == 3:
            fence_after()
            for ki in cutlass.range_constexpr(4):          # [X; Z] K: lanes 0-63 X K, lanes 64-127 Z K
                mma128(tb, smem_desc(aZX + 2 * C * C, 16, 1024) + _adv1(ki),
                       smem_desc(aS + R_K * 128, 2 * KB, 1024) + _advmn(ki),
                       cutlass.Int32(idesc(128, 128, 0, 1)), cutlass.Int32(1 if ki > 0 else 0))
            for ki in cutlass.range_constexpr(4):          # + [0 | X]^T Q: lanes 64-127 get X^T Q
                mma128(tb, smem_desc(aZX, 8192, 1024) + _advmn(ki),
                       smem_desc(aS + R_Q * 128, 2 * KB, 1024) + _advmn(ki),
                       cutlass.Int32(idesc(128, 128, 1, 1)), cutlass.Int32(1))
            commit(_bar(sbar, M5))
        mbar_wait(_bar(sbar, M5), 0)
        fence_after()
        kside = tid >= C
        for s_ in cutlass.range_constexpr(8):
            av = ld32(tb + 16 * s_, 16)
            wait_ld()
            for j in cutlass.range_constexpr(16):
                vals[j] = f32_of_i32(av[j])
            for part in cutlass.range_constexpr(NV):
                for q4 in cutlass.range_constexpr(4):
                    cc = 16 * s_ + 4 * q4
                    src = (qp.iterator + qp.layout((b, n, r, bh, part, cc))).toint()
                    if kside:
                        src = (kp.iterator + kp.layout((b, n, r, bh, part, cc))).toint()
                    w4 = ldg128(src)
                    for j in cutlass.range_constexpr(4):
                        vals[4 * q4 + j] = vals[4 * q4 + j] + f32_of_i32(w4[j])
            if (r < rows) | cutlass.const_expr(not NATIVE):
                for h_ in cutlass.range_constexpr(2):
                    cc = 16 * s_ + 8 * h_
                    dst = (dqout.iterator + dqout.layout((b, n, r, bh, cc))).toint()
                    if kside:
                        dst = (dkout.iterator + dkout.layout((b, n, r, bh, cc))).toint()
                    stg128(dst, pack2(vals[8 * h_ + 1], vals[8 * h_]), pack2(vals[8 * h_ + 3], vals[8 * h_ + 2]),
                           pack2(vals[8 * h_ + 5], vals[8 * h_ + 4]), pack2(vals[8 * h_ + 7], vals[8 * h_ + 6]))
        fence_before()
        cute.arch.barrier()
        tmem.free(tptr, 128)


@cute.jit
def launch_matrix(beta: cute.Tensor, qp: cute.Tensor, kp: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor,
                  db: cute.Tensor, dmp: cute.Tensor, dm: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                  gQT: cute.Tensor, gKT: cute.Tensor, gOT: cute.Tensor, gVT: cute.Tensor, gDVT: cute.Tensor,
                  gWT: cute.Tensor, gAT: cute.Tensor, gMpT: cute.Tensor, gMT: cute.Tensor,
                  B: cutlass.Constexpr, H: cutlass.Constexpr, NC: cutlass.Constexpr, NV: cutlass.Constexpr,
                  SCALE: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr,
                  stream: cuda.CUstream):
    lBox = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, 64, 64), cutlass.BFloat16, 1), (None, None, 0))
    op = cpasync.CopyBulkTensorTileG2SOp()
    atoms = [cpasync.make_tiled_tma_atom(op, t, lBox, (C, 64), 1) for t in (gQT, gKT, gOT, gVT, gDVT, gWT, gAT, gMpT, gMT)]
    (tmaQ, tQt), (tmaK, tKt), (tmaO, tOt), (tmaV, tVt), (tmaDV, tDVt), (tmaW, tWt), (tmaA, tAt), (tmaMp, tMpt), (tmaM, tMt) = atoms
    matrix_kernel(beta, qp, kp, dq, dk, db, dmp, dm, offsets, mapping,
                  tmaQ, tQt, tmaK, tKt, tmaO, tOt, tmaV, tVt, tmaDV, tDVt, tmaW, tWt, tmaA, tAt, tmaMp, tMpt,
                  tmaM, tMt, lBox, H, NC, NV, SCALE, PACKED, NATIVE).launch(
        grid=(B * NC * H, 1, 1), block=(NT, 1, 1), stream=stream)



@cute.jit
def launch_mask_grad(dmp: cute.Tensor, dm: cute.Tensor, gc: cute.Tensor, k2: cute.Tensor, q2: cute.Tensor,
                     dgv: cute.Tensor, dk2v: cute.Tensor, dgo: cute.Tensor, dq2o: cute.Tensor,
                     dg: cute.Tensor, dk2: cute.Tensor, dq2: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                     B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr,
                     NV: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr, stream: cuda.CUstream):
    # two CTAs per SM (at most 128 registers per thread) instead of one: the kernel is latency-bound
    mask_grad_kernel(dmp, dm, gc, k2, q2, dgv, dk2v, dgo, dq2o, dg, dk2, dq2, offsets, mapping,
                     H, E, NC, NV, PACKED, NATIVE).launch(
        grid=(B * NC * H, (E + 7) // 8, 1), block=(256, 1, 1), stream=stream, min_blocks_per_mp=2)


_compiled = {}


def joint_bwd_parallel_sm100(q, k, do, A, Mp, M, gc, k2, q2, beta, W, Vd, dvd,
                             dq_p, dk_p, dgv_p, dk2v_p, dgo_p, dq2o_p, scale=None, chunk_offsets=None, token_map=None):
    """Same contract as sm90_joint_bwd_parallel.joint_bwd_parallel.  Returns dq, dk, dg, dk2, dq2, db."""
    B, T, H, D_ = q.shape
    native = token_map is not None
    E, NC = k2.shape[-1], gc.shape[1] // C
    NV, NR = dq_p.shape[-2], dgv_p.shape[-2]
    assert D_ == D and (native or T % C == 0)
    scale = D ** -0.5 if scale is None else float(scale)
    dev = q.device
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dg, dk2, dq2 = torch.empty_like(gc), torch.empty_like(k2), torch.empty_like(q2)
    db = torch.empty_like(beta)
    dmp = torch.empty((B, NC, H, C, C), device=dev, dtype=torch.float32)
    dm = torch.empty_like(dmp)

    def ch(t):
        return t.view(B, NC, C, *t.shape[2:])

    def raw(t):
        return t.view(B, 1, T, *t.shape[2:]) if native else ch(t)

    def mat(t):
        return t.view(B, NC, C, H, C).permute(2, 4, 0, 1, 3)

    A, Mp, M = A.contiguous(), Mp.contiguous(), M.contiguous()
    packed = chunk_offsets is not None
    ma = [from_dlpack(t.detach(), assumed_align=16) for t in (raw(beta), ch(dq_p), ch(dk_p), raw(dq), raw(dk), raw(db))]
    masks = [from_dlpack(t.detach(), assumed_align=16) for t in (dmp, dm)]
    offsets = from_dlpack(chunk_offsets.detach(), assumed_align=16).mark_layout_dynamic() if packed else masks[0]
    mapping = from_dlpack(token_map.detach(), assumed_align=16) if native else masks[0]
    tma = [from_dlpack(t.detach(), assumed_align=16) for t in
           [raw(t).permute(2, 4, 0, 1, 3) for t in (q, k, do)] + [ch(t).permute(2, 4, 0, 1, 3) for t in (Vd, dvd, W)]
           + [mat(t) for t in (A, Mp, M)]]
    if native and T == 1 and H == 1:
        for a_ in tma[:3]:
            keep_singleton_tma_axis(a_, 0)
    ga = masks + [from_dlpack((raw(t) if i in (1, 2, 8, 9) else ch(t)).detach(), assumed_align=16)
                  for i, t in enumerate((gc, k2, q2, dgv_p, dk2v_p, dgo_p, dq2o_p, dg, dk2, dq2))]
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    key = (B, T, H, E, scale, NV, NR, packed, native, NC)
    if key not in _compiled:
        _compiled[key] = (cute.compile(launch_matrix, *ma, *masks, offsets, mapping, *tma, B, H, NC, NV, scale,
                                       packed, native, stream),
                          cute.compile(launch_mask_grad, *ga, offsets, mapping, B, H, E, NC, NR, packed, native,
                                       stream))
    matrix, mask_grad = _compiled[key]
    matrix(*ma, *masks, offsets, mapping, *tma, stream)
    mask_grad(*ga, offsets, mapping, stream)
    return dq, dk, dg, dk2, dq2, db
