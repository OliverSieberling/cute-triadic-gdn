"""Backward, second stage: every chunk in parallel.

A matrix kernel forms dQ, dK, dbeta and the derivatives of the masks R and R'. A second kernel reduces those
into the second-key, second-query and decay gradients with one stable exponential per (t, s, e). A final
vectorized kernel combines the state-dependent and the chunk-local q/k gradients.
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu import OperandMajorMode as OMM
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, store_pair

C, D, NT = 64, 128, 128


@cute.kernel
def matrix_kernel(rawq: cute.Tensor, rawk: cute.Tensor, rawdo: cute.Tensor,
                  a: cute.Tensor, mp: cute.Tensor, m: cute.Tensor, beta: cute.Tensor,
                  wv: cute.Tensor, vd: cute.Tensor, dvd: cute.Tensor,
                  qp: cute.Tensor, kp: cute.Tensor,
                  dq: cute.Tensor, dk: cute.Tensor, db: cute.Tensor,
                  dmp: cute.Tensor, dm: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                  gram: cute.TiledMma, left_t: cute.TiledMma,
                  out: cute.TiledMma, out_t: cute.TiledMma,
                  lk: cute.ComposedLayout, lkt: cute.ComposedLayout,
                  la: cute.ComposedLayout, lat: cute.ComposedLayout,
                  H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr, NV: cutlass.Constexpr,
                  SCALE: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bh, n, b = bid % H, (bid // H) % NC, bid // (H * NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets)-1]
    if active:
        rows = valid_rows(n,mapping,NATIVE)
        q = token_chunk(rawq,n,mapping,NATIVE)
        k = token_chunk(rawk,n,mapping,NATIVE)
        do = token_chunk(rawdo,n,mapping,NATIVE)
        bi = token_chunk(beta,n,mapping,NATIVE)
        dbout = token_chunk(db,n,mapping,NATIVE)
        dqout = token_chunk(dq,n,mapping,NATIVE)
        dkout = token_chunk(dk,n,mapping,NATIVE)
        smem = utils.SmemAllocator()
        def alloc(layout):
            return smem.allocate_tensor(cutlass.BFloat16, layout.outer, 128, swizzle=layout.inner)
        def transpose(t, layout):
            return cute.make_tensor(cute.recast_ptr(t.iterator, layout.inner, dtype=cutlass.BFloat16), layout.outer)
        sQ, sK, sD0, sD1 = alloc(lk), alloc(lk), alloc(lk), alloc(lk)
        sA, sMp, sM, sX = alloc(la), alloc(la), alloc(la), alloc(la)
        sF = smem.allocate_tensor(cutlass.Float32, cute.make_layout((C, C), stride=(C + 1, 1)), byte_alignment=16)
        sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), byte_alignment=16)
        sDb = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), byte_alignment=16)
        sQt, sKt = transpose(sQ, lkt), transpose(sK, lkt)
        sAt, sXt = transpose(sA, lat), transpose(sX, lat)
        # sMp becomes the intermediate A^T dA once the readout derivatives are formed.
        def load128(src, dst, limit):
            for j in cutlass.range_constexpr(C * D // (NT * 8)):
                p = j * NT + tid
                r,c = p // (D//8),(p % (D//8))*8
                view = cute.make_tensor((src.iterator+src.layout((b,n,r,bh,c))).align(16),cute.make_layout(8))
                vec = cute.make_fragment_like(view)
                vec.fill(0.)
                if r < limit:
                    cute.autovec_copy(view,vec)
                # Swizzle<3,4,3> acts on byte addresses. Apply it explicitly before
                # a 16-byte store; a vector tensor over a swizzled pointer skips it.
                off=(c//64)*4096+r*64+8*((c//8)%8 ^ (r%8))+c%8
                raw=cute.recast_ptr(dst.iterator,dtype=cutlass.BFloat16)
                packed=cute.make_tensor((raw+off).align(16),cute.make_layout(8))
                packed.store(vec.load())
        load128(q, sQ, rows)
        load128(k, sK, rows)
        load128(do, sD0, rows)
        load128(vd, sD1, C)
        for j in cutlass.range_constexpr(C * C // NT):
            p = j * NT + tid
            r, c = p // C, p % C
            sA[r, c, 0] = a[b, n, r, bh, c]
            sMp[r, c, 0] = mp[b, n, r, bh, c]
            sM[r, c, 0] = m[b, n, r, bh, c]
        if tid < C:
            sBeta[tid] = masked_scalar(bi,(b,n,tid,bh),tid,rows)
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        tp, to = gram.get_slice(tid), out.get_slice(tid)
        cp = tp.partition_C(cute.make_identity_tensor((C, C)))
        co = to.partition_C(cute.make_identity_tensor((C, D)))
        def gemm(mma, acc, sa, sb, accumulate=False):
            th = mma.get_slice(tid)
            fa = mma.make_fragment_A(th.partition_A(sa[None, None, 0]))
            fb = mma.make_fragment_B(th.partition_B(sb[None, None, 0]))
            warpgroup.fence()
            for j in cutlass.range(cute.size(fa, mode=[2]), unroll_full=True):
                mma.set(warpgroup.Field.ACCUMULATE, cutlass.Boolean(accumulate or j != 0))
                cute.gemm(mma, acc, fa[None, None, j], fb[None, None, j], acc)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
        p = gram.make_fragment_C(gram.partition_shape_C((C, C)))
        dp = gram.make_fragment_C(gram.partition_shape_C((C, C)))
        gemm(gram, p, sQ, sK)
        gemm(gram, dp, sD0, sD1)
        # epilogues in pairs of consecutive columns (same per-element arithmetic; masked elements are
        # computed and then zeroed, as the branch left them)
        pairf = cute.make_rmem_tensor((2,), cutlass.Float32)
        pairh = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        for jp in cutlass.range_constexpr(C * C // NT // 2):
            i = 2 * jp
            r, c = cp[i][0], cp[i][1]
            mv = cute.make_tensor((sMp.iterator + sMp.layout((r, c, 0))).align(4), cute.make_layout((2,))).load()
            m0 = cutlass.Float32(mv[0])
            m1 = cutlass.Float32(mv[1])
            dmpr0 = SCALE * p[i] * dp[i]
            dpr0 = SCALE * m0 * dp[i]
            dmpr1 = SCALE * p[i + 1] * dp[i + 1]
            dpr1 = SCALE * m1 * dp[i + 1]
            if c > r:
                dmpr0 = cutlass.Float32(0.)
                dpr0 = cutlass.Float32(0.)
            if c + 1 > r:
                dmpr1 = cutlass.Float32(0.)
                dpr1 = cutlass.Float32(0.)
            pairf[0], pairf[1] = dmpr0, dmpr1
            cute.make_tensor((dmp.iterator + dmp.layout((b, n, bh, r, c))).align(8), cute.make_layout((2,))).store(pairf.load())
            pairh[0], pairh[1] = cutlass.BFloat16(dpr0), cutlass.BFloat16(dpr1)
            cute.make_tensor((sX.iterator + sX.layout((r, c, 0))).align(4), cute.make_layout((2,))).store(pairh.load())
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        aq = out.make_fragment_C(out.partition_shape_C((C, D)))
        ak = out.make_fragment_C(out.partition_shape_C((C, D)))
        gemm(out, aq, sX, sKt)
        gemm(out_t, ak, sXt, sQt)
        # Fold state contributions in their original order, then round once.  The partials are loaded
        # eight pairs at a time before any of them is added and stored: one global-load latency per
        # eight pairs instead of one per pair (the compiler cannot move a load over the preceding store).
        pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
        vq=cute.make_rmem_tensor((8,NV,2),cutlass.Float32)
        for gq in cutlass.range_constexpr(C*D//(NT*2)//8):
            for jj in cutlass.range_constexpr(8):
                i=2*(gq*8+jj)
                r,c=co[i][0],co[i][1]
                for part in cutlass.range_constexpr(NV):
                    src=cute.make_tensor((qp.iterator+qp.layout((b,n,r,bh,part,c))).align(8),cute.make_layout(2))
                    vals=src.load()
                    vq[jj,part,0]=vals[0]
                    vq[jj,part,1]=vals[1]
            for jj in cutlass.range_constexpr(8):
                i=2*(gq*8+jj)
                r,c=co[i][0],co[i][1]
                v0,v1=aq[i],aq[i+1]
                for part in cutlass.range_constexpr(NV):
                    v0,v1=v0+vq[jj,part,0],v1+vq[jj,part,1]
                pack[0],pack[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                store_pair(dqout,b,n,r,bh,c,pack.load(),rows)
        # dAb = dVd W^T; dbeta_col = column_sum(dAb * A).
        load128(dvd, sD0, C)
        load128(wv, sD1, C)
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        gemm(gram, dp, sD0, sD1)
        pairh2 = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        for jp in cutlass.range_constexpr(C * C // NT // 2):
            i = 2 * jp
            r, c = cp[i][0], cp[i][1]
            pairh2[0], pairh2[1] = cutlass.BFloat16(dp[i] * sBeta[c]), cutlass.BFloat16(dp[i + 1] * sBeta[c + 1])
            cute.make_tensor((sX.iterator + sX.layout((r, c, 0))).align(4), cute.make_layout((2,))).store(pairh2.load())
        for j in cutlass.range_constexpr(C//8):
            i = 4*j
            r0,r1,c = cp[i][0],cp[i+2][0],cp[i][1]
            d0 = dp[i]*cutlass.Float32(sA[r0,c,0])+dp[i+2]*cutlass.Float32(sA[r1,c,0])
            d1 = dp[i+1]*cutlass.Float32(sA[r0,c+1,0])+dp[i+3]*cutlass.Float32(sA[r1,c+1,0])
            for off in (4,8,16):
                d0 = d0+cute.arch.shuffle_sync_bfly(d0,off)
                d1 = d1+cute.arch.shuffle_sync_bfly(d1,off)
            if tid%32<4:
                sF[tid//32,c] = d0
                sF[tid//32,c+1] = d1
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        if tid < C:
            val = cutlass.Float32(0.)
            for j in cutlass.range_constexpr(4):
                val = val + sF[j, tid]
            sDb[tid] = val
        # dLm = -A^T dA A^T (strict lower). sMp is now scratch Y.
        gemm(left_t, p, sAt, sXt)
        pairh3 = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        for jp in cutlass.range_constexpr(C * C // NT // 2):
            i = 2 * jp
            pairh3[0], pairh3[1] = cutlass.BFloat16(p[i]), cutlass.BFloat16(p[i + 1])
            cute.make_tensor((sMp.iterator + sMp.layout((cp[i][0], cp[i][1], 0))).align(4), cute.make_layout((2,))).store(pairh3.load())
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        gemm(gram, dp, sMp, sA)
        gemm(gram, p, sK, sK)
        pairf4 = cute.make_rmem_tensor((2,), cutlass.Float32)
        pairh4 = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        for jp in cutlass.range_constexpr(C * C // NT // 2):
            i = 2 * jp
            r, c = cp[i][0], cp[i][1]
            mv4 = cute.make_tensor((sM.iterator + sM.layout((r, c, 0))).align(4), cute.make_layout((2,))).load()
            sm0 = cutlass.Float32(mv4[0])
            sm1 = cutlass.Float32(mv4[1])
            dl0 = cutlass.Float32(0.)
            dl1 = cutlass.Float32(0.)
            if c < r:
                dl0 = -dp[i]
            if c + 1 < r:
                dl1 = -dp[i + 1]
            pairf4[0], pairf4[1] = dl0 * sBeta[r] * p[i], dl1 * sBeta[r] * p[i + 1]
            cute.make_tensor((dm.iterator + dm.layout((b, n, bh, r, c))).align(8), cute.make_layout((2,))).store(pairf4.load())
            pairh4[0], pairh4[1] = cutlass.BFloat16(dl0 * sBeta[r] * sm0), cutlass.BFloat16(dl1 * sBeta[r] * sm1)
            cute.make_tensor((sX.iterator + sX.layout((r, c, 0))).align(4), cute.make_layout((2,))).store(pairh4.load())
            p[i] = dl0 * p[i] * sm0
            p[i + 1] = dl1 * p[i + 1] * sm1
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.barrier()
        d0,d1 = cutlass.Float32(0.),cutlass.Float32(0.)
        for i in cutlass.range_constexpr(C*C//NT):
            if cutlass.const_expr((i//2)%2==0):
                d0 = d0+p[i]
            else:
                d1 = d1+p[i]
        d0 = d0+cute.arch.shuffle_sync_bfly(d0,1)
        d0 = d0+cute.arch.shuffle_sync_bfly(d0,2)
        d1 = d1+cute.arch.shuffle_sync_bfly(d1,1)
        d1 = d1+cute.arch.shuffle_sync_bfly(d1,2)
        if tid%4==0:
            if cp[0][0] < rows:
                dbout[b,n,cp[0][0],bh] = sDb[cp[0][0]]+d0
            if cp[2][0] < rows:
                dbout[b,n,cp[2][0],bh] = sDb[cp[2][0]]+d1
        gemm(out, ak, sX, sKt, True)
        gemm(out_t, ak, sXt, sKt, True)
        packk=cute.make_rmem_tensor((2,),cutlass.BFloat16)
        vk=cute.make_rmem_tensor((8,NV,2),cutlass.Float32)
        for gk in cutlass.range_constexpr(C*D//(NT*2)//8):
            for jj in cutlass.range_constexpr(8):
                i=2*(gk*8+jj)
                r,c=co[i][0],co[i][1]
                for part in cutlass.range_constexpr(NV):
                    srck=cute.make_tensor((kp.iterator+kp.layout((b,n,r,bh,part,c))).align(8),cute.make_layout(2))
                    valsk=srck.load()
                    vk[jj,part,0]=valsk[0]
                    vk[jj,part,1]=valsk[1]
            for jj in cutlass.range_constexpr(8):
                i=2*(gk*8+jj)
                r,c=co[i][0],co[i][1]
                v0,v1=ak[i],ak[i+1]
                for part in cutlass.range_constexpr(NV):
                    v0,v1=v0+vk[jj,part,0],v1+vk[jj,part,1]
                packk[0],packk[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                store_pair(dkout,b,n,r,bh,c,packk.load(),rows)


def _bit_reverse5(m):
    return sum(((m >> i) & 1) << (4 - i) for i in range(5))


def _tree_add(stack, m, x):
    # Trace-time helper (plain Python): binary-counter evaluation of the balanced pairwise tree over
    # 32 leaves fed in order m = 0..31; partial sums are left + right like the butterfly reduction.
    # Returns the total when the last leaf arrives (m == 31), else None.
    for lvl in range(5):
        if (m >> lvl) & 1 == 0:
            stack[lvl] = x
            return None
        x = stack[lvl] + x
    return x


@cute.kernel
def mask_grad_kernel(dmp: cute.Tensor, dm: cute.Tensor, gc: cute.Tensor,
                   k2: cute.Tensor, q2: cute.Tensor, dgv: cute.Tensor,
                   dk2v: cute.Tensor, dgo: cute.Tensor, dq2o: cute.Tensor,
                   dg: cute.Tensor, dk2: cute.Tensor, dq2: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                   H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr, NV: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    # One lane per row: the lane walks the columns itself and reproduces the exact addition tree of a
    # warp butterfly reduction (offsets 16, 8, 4, 2, 1: a balanced pairwise tree over the lanes in
    # bit-reversed order), so every sum is bitwise reproducible.
    tid, _, _ = cute.arch.thread_idx()
    bid, slice_block, _ = cute.arch.block_idx()
    bh = bid % H
    n, b = (bid // H) % NC, bid // (H * NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets)-1]
    if active:
        rows = valid_rows(n,mapping,NATIVE)
        k2c = token_chunk(k2,n,mapping,NATIVE)
        qi = token_chunk(q2,n,mapping,NATIVE)
        dk2out = token_chunk(dk2,n,mapping,NATIVE)
        dqout = token_chunk(dq2,n,mapping,NATIVE)
        lane, wid = tid % 32, tid // 32
        SLICES = min(E, 8)
        sl = slice_block * SLICES + wid % SLICES
        ROW_GROUPS = 8 // SLICES
        RPW = C // ROW_GROUPS                  # rows per warp
        RPL = max(1, RPW // 32)                # rows per lane
        LPR = min(32, RPW)                     # lanes that own a row
        smem = utils.SmemAllocator()
        layout = cute.make_layout((C, C), stride=(C + 1, 1))
        sMP = smem.allocate_tensor(cutlass.Float32, layout, byte_alignment=16)
        sM = smem.allocate_tensor(cutlass.Float32, layout, byte_alignment=16)
        sg = smem.allocate_tensor(cutlass.Float32, cute.make_layout((E,C),stride=(C,1)), byte_alignment=16)
        sk2 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((E,C),stride=(C,1)), byte_alignment=16)
        sq = smem.allocate_tensor(cutlass.Float32, cute.make_layout((E,C),stride=(C,1)), byte_alignment=16)
        for j in cutlass.range_constexpr(C * C // 256):
            p = j * 256 + tid
            sMP[p // C, p % C] = dmp[b, n, bh, p // C, p % C]
            sM[p // C, p % C] = dm[b, n, bh, p // C, p % C]
        for j in cutlass.range_constexpr((C * E + 255) // 256):
            pos = j * 256 + tid
            if pos < C * E:
                row_, sl_ = pos // E, pos % E
                sg[sl_, row_] = gc[b, n, row_, bh, sl_]
                sk2[sl_, row_] = masked_scalar(k2c,(b,n,row_,bh,sl_),row_,rows)
                sq[sl_, row_] = masked_scalar(qi,(b,n,row_,bh,sl_),row_,rows)
        cute.arch.barrier()
        if sl < E:
            if lane < LPR:
                base = (wid // SLICES) * RPW
                for t in cutlass.range_constexpr(RPL):
                    row = base + lane + 32 * t
                    gr = sg[sl, row]
                    s1, s2, s3, s4 = [None] * 5, [None] * 5, [None] * 5, [None] * 5
                    r1, c1, r2, c2 = cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.)
                    for m in cutlass.range_constexpr(32):
                        l = _bit_reverse5(m)
                        p1, p2, p3, p4 = cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.)
                        for half in cutlass.range_constexpr(2):
                            col = l + half * 32
                            cgc, ck2, cqc = sg[sl, col], sk2[sl, col], sq[sl, col]
                            delta = gr - cgc
                            # For monotone cumulative log-decay, both causal orientations use
                            # exp(-abs(delta)); upper-triangular matrix entries are already zero.
                            decay = cute.math.exp(-cute.math.absf(delta), fastmath=True)
                            p1 = p1 + sMP[row, col] * decay * ck2
                            p2 = p2 + sMP[col, row] * decay * cqc
                            p3 = p3 + sM[row, col] * decay * ck2
                            p4 = p4 + sM[col, row] * decay * ck2
                        o1, o2, o3, o4 = _tree_add(s1, m, p1), _tree_add(s2, m, p2), _tree_add(s3, m, p3), _tree_add(s4, m, p4)
                        if cutlass.const_expr(m == 31):
                            r1, c1, r2, c2 = o1, o2, o3, o4
                    pg, pk2, pq = cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.)
                    if cutlass.const_expr(NV == 1):
                        pg = dgv[b, n, row, bh, 0, sl] + dgo[b, n, row, bh, 0, sl]
                        pk2 = dk2v[b, n, row, bh, 0, sl]
                        pq = dq2o[b, n, row, bh, 0, sl]
                    else:
                        # the warp-reduced partials over NV lanes, same tree (zero leaves beyond NV)
                        sg_, sk2_, sq_ = [None] * 5, [None] * 5, [None] * 5
                        for m in cutlass.range_constexpr(32):
                            l = _bit_reverse5(m)
                            vg, vk2, vq = cutlass.Float32(0.), cutlass.Float32(0.), cutlass.Float32(0.)
                            if cutlass.const_expr(l < NV):
                                vg = dgv[b, n, row, bh, l, sl] + dgo[b, n, row, bh, l, sl]
                                vk2 = dk2v[b, n, row, bh, l, sl]
                                vq = dq2o[b, n, row, bh, l, sl]
                            og, ok2, oq = _tree_add(sg_, m, vg), _tree_add(sk2_, m, vk2), _tree_add(sq_, m, vq)
                            if cutlass.const_expr(m == 31):
                                pg, pk2, pq = og, ok2, oq
                    dg[b, n, row, bh, sl] = pg + sq[sl, row] * r1 + sk2[sl, row] * (r2 - c1 - c2)
                    if row < rows:
                        dk2out[b, n, row, bh, sl] = pk2 + c1 + r2 + c2
                        dqout[b, n, row, bh, sl] = pq + r1


@cute.jit
def launch_matrix(q: cute.Tensor, k: cute.Tensor, do: cute.Tensor,
                  a: cute.Tensor, mp: cute.Tensor, m: cute.Tensor, beta: cute.Tensor,
                  wv: cute.Tensor, vd: cute.Tensor, dvd: cute.Tensor,
                  qp: cute.Tensor, kp: cute.Tensor,
                  dq: cute.Tensor, dk: cute.Tensor, db: cute.Tensor,
                  dmp: cute.Tensor, dm: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                  B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                  NC: cutlass.Constexpr, NV: cutlass.Constexpr, SCALE: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr, stream: cuda.CUstream):
    bf, ff = cutlass.BFloat16, cutlass.Float32
    def mma(n, am, bm):
        return sm90.make_trivial_tiled_mma(bf, bf, am, bm, ff, (1, 1, 1), (C, n))
    gram = mma(C, OMM.K, OMM.K)
    left_t = mma(C, OMM.MN, OMM.MN)
    out, out_t = mma(D, OMM.K, OMM.MN), mma(D, OMM.MN, OMM.MN)
    lk = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, D), bf, 1)
    lkt = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, D, C), bf, 1)
    la = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, C), bf, 1)
    lat = sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR, (C, C, C), bf, 1)
    matrix_kernel(q,k,do,a,mp,m,beta,wv,vd,dvd,qp,kp,dq,dk,db,dmp,dm,offsets,mapping,
        gram,left_t,out,out_t,lk,lkt,la,lat,H,E,NC,NV,SCALE,PACKED,NATIVE).launch(
            grid=(B * NC * H, 1, 1), block=(NT, 1, 1), stream=stream)


@cute.jit
def launch_mask_grad(dmp: cute.Tensor, dm: cute.Tensor, gc: cute.Tensor,
                   k2: cute.Tensor, q2: cute.Tensor, dgv: cute.Tensor,
                   dk2v: cute.Tensor, dgo: cute.Tensor, dq2o: cute.Tensor,
                   dg: cute.Tensor, dk2: cute.Tensor, dq2: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                   B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                   NC: cutlass.Constexpr, NV: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr, stream: cuda.CUstream):
    mask_grad_kernel(dmp,dm,gc,k2,q2,dgv,dk2v,dgo,dq2o,dg,dk2,dq2,offsets,mapping,H,E,NC,NV,PACKED,NATIVE).launch(
        grid=(B * NC * H, (E+7)//8, 1), block=(256, 1, 1), stream=stream)


_compiled = {}


def joint_bwd_parallel(q,k,do,A,Mp,M,gc,k2,q2,beta,W,Vd,dvd,
                       dq_p,dk_p,dgv_p,dk2v_p,dgo_p,dq2o_p,scale=None,chunk_offsets=None,token_map=None):
    B,T,H,D_ = q.shape
    native=token_map is not None
    E,NC = k2.shape[-1], gc.shape[1] // C
    NV = dq_p.shape[-2]
    NR = dgv_p.shape[-2]
    assert NV in (1, E)
    assert D_ == D and (native or T % C == 0) and E in (1, 2, 4, 8, 12, 16)
    scale = D ** -0.5 if scale is None else float(scale)
    dq,dk = torch.empty_like(q),torch.empty_like(k)
    dg,dk2,dq2 = torch.empty_like(gc),torch.empty_like(k2),torch.empty_like(q2)
    db = torch.empty_like(beta)
    dmp = torch.empty((B,NC,H,C,C),device=q.device,dtype=torch.float32)
    dm = torch.empty_like(dmp)
    def chunks(t):
        return t.view(B,NC,C,*t.shape[2:])
    raw_matrix = (q,k,do,A,Mp,M,beta,W,Vd,dvd,dq_p,dk_p,dq,dk,db)
    def raw(t):
        return t.view(B,1,T,*t.shape[2:]) if native else chunks(t)
    ma = [from_dlpack((raw(t) if i in (0,1,2,6,12,13,14) else chunks(t)).detach(),assumed_align=16)
          for i,t in enumerate(raw_matrix)]
    masks = [from_dlpack(t.detach(),assumed_align=16) for t in (dmp,dm)]
    ra = masks + [from_dlpack((raw(t) if i in (1,2,8,9) else chunks(t)).detach(),assumed_align=16)
                  for i,t in enumerate((gc,k2,q2,dgv_p,dk2v_p,dgo_p,dq2o_p,dg,dk2,dq2))]
    packed = chunk_offsets is not None
    offset_arg = from_dlpack(chunk_offsets.detach(),assumed_align=16).mark_layout_dynamic() if packed else ma[0]
    mapping=from_dlpack(token_map.detach(),assumed_align=16) if native else ma[0]
    ra.extend((offset_arg,mapping))
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key=(B,T,H,E,scale,NV,NR,packed,native,NC)
    if key not in _compiled:
        mat=cute.compile(launch_matrix,*ma,*masks,offset_arg,mapping,B,H,E,NC,NV,scale,packed,native,stream)
        mask_grad=cute.compile(launch_mask_grad,*ra,B,H,E,NC,NR,packed,native,stream)
        _compiled[key]=(mat,mask_grad)
    mat,mask_grad=_compiled[key]
    mat(*ma,*masks,offset_arg,mapping,stream)
    mask_grad(*ra,stream)
    return dq,dk,dg,dk2,dq2,db
