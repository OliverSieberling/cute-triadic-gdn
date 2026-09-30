"""Blackwell (sm100/sm103) chunk masks: the second-key masks R and R' (Eq. 9), the UT transform (Eq. 10) and, for the
forward, Pm = scale * tril(Q K^T * R'), one CTA per (chunk, head).

[K; Q] arrives by TMA as one 128-row K-major tile; one tcgen05 MMA [K; Q] K^T puts K K^T in tensor-memory lanes 0-63
and Q K^T in lanes 64-127.  The mask sums (E exponentials per element) are spread evenly over the 128 threads into
shared memory; then thread t combines its tensor-memory row with them: rows 0-63 the UT input and R, rows 64-127 R'
and Pm.  The blocked inverse is sm90_joint_masks's.  The [K; Q] tile is dead after the MMA, so the inverse's FP32
buffers reuse it: about 45 KB of shared memory, four CTAs per SM.
Packed documents: rows past a document end hold the next document's tokens; their k2, q2, beta are masked to zero,
so their mask, UT-input and Pm entries are zero (the diagonal of the UT input stays one).
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90          # smem layout helper only (a plain swizzled layout)
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from .sm90_joint_masks import fma, _inv16_cols, _mma_tf32
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, keep_singleton_tma_axis
from .sm100_tc import (idesc, mma128, commit, fence_after, wait_ld, ld32, f32_of_i32, i32_of_f32, smem_desc,
                       mbar_wait, pack2, unpack_lo, unpack_hi, sts128, stg128, lds128, saddr)

C,D,NT=64,128,128


def _box(ptr, lBox):
    return cute.group_modes(cute.make_tensor(cute.recast_ptr(ptr, lBox.inner, dtype=cutlass.BFloat16), lBox.outer), 0, 2)


@cute.kernel
def masks_kernel(rawk2: cute.Tensor, rawq2: cute.Tensor, gc: cute.Tensor, rawbeta: cute.Tensor,
                    A: cute.Tensor, Mp: cute.Tensor, M: cute.Tensor, Pm: cute.Tensor,
                    offsets: cute.Tensor, mapping: cute.Tensor,
                    tmaK: cute.CopyAtom, tKt: cute.Tensor, tmaQ: cute.CopyAtom, tQt: cute.Tensor,
                    lBox: cute.ComposedLayout,
                    H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr, PACKED: cutlass.Constexpr,
                    NATIVE: cutlass.Constexpr, WITH_PM: cutlass.Constexpr, SCALE: cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    bh,n,b=bid%H,(bid//H)%NC,bid//(H*NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets)-1]
    if active:
        rows = valid_rows(n, mapping, NATIVE)
        k2 = token_chunk(rawk2,n,mapping,NATIVE)
        q2 = token_chunk(rawq2,n,mapping,NATIVE)
        beta = token_chunk(rawbeta,n,mapping,NATIVE)
        lane,wid=tid%32,tid//32
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bf = cutlass.BFloat16
        f32 = cutlass.Float32
        smem=utils.SmemAllocator()
        sKQ=smem.allocate_tensor(bf,cute.make_layout((128*D,)),byte_alignment=1024)        # [K; Q], 32 KB
        sMp=smem.allocate_tensor(bf,cute.make_layout((C,C+8),stride=(C+8,1)),byte_alignment=16)
        sg=smem.allocate_tensor(f32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sk2=smem.allocate_tensor(f32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sq=smem.allocate_tensor(f32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sb=smem.allocate_tensor(f32,cute.make_layout(C),byte_alignment=16)
        sbar=smem.allocate_tensor(cutlass.Int64,cute.make_layout((2,)),byte_alignment=8)
        thold=smem.allocate_tensor(cutlass.Int32,cute.make_layout((1,)),byte_alignment=16)
        # after the MMA: the FP32 UT input / inverse and the merge scratch over the dead [K; Q] tile
        sf=cute.make_tensor(cute.recast_ptr(sKQ.iterator,dtype=f32),cute.make_layout((C,C),stride=(C+4,1)))
        tmp=cute.make_tensor(cute.recast_ptr(sKQ.iterator+2*C*(C+4),dtype=f32),cute.make_layout((32,32),stride=(36,1)))
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(sbar.iterator, 1)
                cute.arch.mbarrier_init(sbar.iterator + 1, 1)
        cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(thold.iterator, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=NT),
                                   allocator_warp_id=0)
        tmem.allocate(64)
        if warp == 0:
            # without this no other CTA can allocate tensor memory on this SM until this one exits (one CTA per SM)
            cute.arch.relinquish_tmem_alloc_permit()
        tmem.wait_for_alloc()
        tptr = tmem.retrieve_ptr(f32)
        tb = cutlass.Int32(tptr.toint())
        one = cute.make_layout(1)
        kq = sKQ.iterator
        if cutlass.const_expr(NATIVE):
            tok = mapping[0, n]
            tKu = cute.domain_offset((tok, 0, 0, 0, 0), tKt)
            tQu = cute.domain_offset((tok, 0, 0, 0, 0), tQt)
            cn = 0
        else:
            tKu, tQu = tKt, tQt
            cn = n
        gK = cute.group_modes(cute.local_tile(tKu, (C, 64), (None, None, None, None, None)), 0, 2)
        gQ = cute.group_modes(cute.local_tile(tQu, (C, 64), (None, None, None, None, None)), 0, 2)
        # K rows 0-63 and Q rows 64-127 of each 64-column block (128 rows, 16 KB)
        dK0, tgK = cpasync.tma_partition(tmaK, 0, one, _box(kq, lBox), gK)
        dK1, _ = cpasync.tma_partition(tmaK, 0, one, _box(kq + 8192, lBox), gK)
        dQ0, tgQ = cpasync.tma_partition(tmaQ, 0, one, _box(kq + 4096, lBox), gQ)
        dQ1, _ = cpasync.tma_partition(tmaQ, 0, one, _box(kq + 12288, lBox), gQ)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(sbar.iterator, (2 if WITH_PM else 1) * C * D * 2)
            cute.copy(tmaK, tgK[(None, 0, 0, b, cn, bh)], dK0, tma_bar_ptr=sbar.iterator)
            cute.copy(tmaK, tgK[(None, 0, 1, b, cn, bh)], dK1, tma_bar_ptr=sbar.iterator)
            if cutlass.const_expr(WITH_PM):
                cute.copy(tmaQ, tgQ[(None, 0, 0, b, cn, bh)], dQ0, tma_bar_ptr=sbar.iterator)
                cute.copy(tmaQ, tgQ[(None, 0, 1, b, cn, bh)], dQ1, tma_bar_ptr=sbar.iterator)
        for j in cutlass.range_constexpr((E*C+NT-1)//NT):
            pos=j*NT+tid
            if pos<E*C:
                r,e=pos//E,pos%E
                sg[e,r]=gc[b,n,r,bh,e]
                sk2[e,r]=masked_scalar(k2,(b,n,r,bh,e),r,rows)
                sq[e,r]=masked_scalar(q2,(b,n,r,bh,e),r,rows)
        if tid<C:
            sb[tid]=masked_scalar(beta,(b,n,tid,bh),tid,rows)
        cute.arch.barrier()                                        # gate tables visible
        if warp == 0:                                              # [K; Q] K^T
            mbar_wait(cutlass.Int32(sbar.iterator.toint()), 0)
            fence_after()
            akq = cutlass.Int32(kq.toint())
            for ki in cutlass.range_constexpr(8):
                adv = ((ki >> 2) * 16384 + (ki & 3) * 32) >> 4
                mma128(tb, smem_desc(akq, 16, 1024) + adv, smem_desc(akq, 16, 1024) + adv,
                       cutlass.Int32(idesc(128, 64, 0, 0)), cutlass.Int32(1 if ki > 0 else 0))
            commit(cutlass.Int32((sbar.iterator + 1).toint()))
        mbar_wait(cutlass.Int32((sbar.iterator + 1).toint()), 0)
        fence_after()
        # mask sums, balanced: rows p and 63 - p (65 lower-triangle entries) shared by four threads
        pp, sub = tid // 4, tid % 4
        for i in cutlass.range_constexpr(17):
            jj = sub + 4 * i
            if jj < C + 1:
                r = pp
                c = jj
                if jj > pp:
                    r = C - 1 - pp
                    c = jj - pp - 1
                m, mp = cutlass.Float32(0.), cutlass.Float32(0.)
                for e in cutlass.range_constexpr(E):
                    decay=cute.math.exp(sg[e,r]-sg[e,c],fastmath=True)
                    m=m+sk2[e,r]*sk2[e,c]*decay
                    mp=mp+sq[e,r]*sk2[e,c]*decay
                sf[r,c]=m
                sMp[r,c]=cutlass.BFloat16(mp)
        vals = cute.make_rmem_tensor((C,), f32)
        for s_ in cutlass.range_constexpr(4):
            v16 = ld32(tb + 16 * s_, 16)
            for j in cutlass.range_constexpr(16):
                vals[16 * s_ + j] = f32_of_i32(v16[j])
        wait_ld()
        cute.arch.barrier()
        r = tid % C
        x = cute.make_rmem_tensor((8,), f32)
        if tid < C:
            # UT input row r: beta_r (K K^T)[r, c] m[r, c] below the diagonal, 1 on it; M row r (strictly lower)
            br = sb[r]
            y = cute.make_rmem_tensor((4,), f32)
            for c8 in cutlass.range_constexpr(8):
                for h4 in cutlass.range_constexpr(2):
                    c0 = 8 * c8 + 4 * h4
                    fa = saddr(sf.iterator, r * (C + 4) + c0)       # 16 B per thread; rows 272 B apart: no conflicts
                    w = lds128(fa)
                    for j in cutlass.range_constexpr(4):
                        c = c0 + j
                        mv = f32_of_i32(w[j])
                        val = cutlass.Float32(0.)
                        mo = cutlass.Float32(0.)
                        if c < r:
                            val = br * vals[c] * mv
                            mo = mv
                        if c == r:
                            val = cutlass.Float32(1.)
                        x[4 * h4 + j] = mo
                        y[j] = val
                    sts128(fa, i32_of_f32(y[0]), i32_of_f32(y[1]), i32_of_f32(y[2]), i32_of_f32(y[3]))
                stg128((M.iterator + M.layout((b, n, r, bh, 8 * c8))).toint(),
                       pack2(x[1], x[0]), pack2(x[3], x[2]), pack2(x[5], x[4]), pack2(x[7], x[6]))
        else:
            # Mp row r (lower, with the diagonal) and Pm = scale * (Q K^T) * Mp as stored
            for c8 in cutlass.range_constexpr(8):
                w = lds128(saddr(sMp.iterator, r * (C + 8) + 8 * c8))
                for j in cutlass.range_constexpr(8):
                    c = 8 * c8 + j
                    mv = unpack_lo(w[j // 2]) if j % 2 == 0 else unpack_hi(w[j // 2])
                    if c > r:
                        mv = cutlass.Float32(0.)
                    x[j] = mv
                stg128((Mp.iterator + Mp.layout((b, n, r, bh, 8 * c8))).toint(),
                       pack2(x[1], x[0]), pack2(x[3], x[2]), pack2(x[5], x[4]), pack2(x[7], x[6]))
                if cutlass.const_expr(WITH_PM):
                    for j in cutlass.range_constexpr(8):
                        x[j] = SCALE * vals[8 * c8 + j] * x[j]
                    stg128((Pm.iterator + Pm.layout((b, n, r, bh, 8 * c8))).toint(),
                           pack2(x[1], x[0]), pack2(x[3], x[2]), pack2(x[5], x[4]), pack2(x[7], x[6]))
        cute.arch.barrier()
        cols=[]
        for r in cutlass.range_constexpr(16):
            val=cutlass.Float32(0.)
            if lane<16:
                val=sf[wid*16+r,wid*16+lane]
            cols.append(val)
        cols=_inv16_cols(cols,lane)
        for r in cutlass.range_constexpr(16):
            if lane<16:
                sf[wid*16+r,wid*16+lane]=cols[r]
        cute.arch.barrier()
        # Inv([[L,0],[X,R]]) = [[Linv,0],[-Rinv X Linv,Rinv]].
        # Keep first-level products in FP32; match TF32 operands at the final level.
        for size in cutlass.range_constexpr(16,17,16):
            for j in cutlass.range_constexpr((C//(2*size))*size*size//NT):
                pos=j*NT+tid
                block,r,c=pos//(size*size),(pos//size)%size,pos%size
                offset=block*2*size
                value=cutlass.Float32(0.)
                for t in cutlass.range_constexpr(size):
                    left,right=sf[offset+size+r,offset+size+t],sf[offset+size+t,offset+c]
                    value=fma(-left,right,value)
                tmp[block*size+r,c]=value
            cute.arch.barrier()
            for j in cutlass.range_constexpr((C//(2*size))*size*size//NT):
                pos=j*NT+tid
                block,r,c=pos//(size*size),(pos//size)%size,pos%size
                offset=block*2*size
                value=cutlass.Float32(0.)
                for t in cutlass.range_constexpr(size):
                    left,right=tmp[block*size+r,t],sf[offset+t,offset+c]
                    value=fma(left,right,value)
                sf[offset+size+r,offset+c]=value
            cute.arch.barrier()
        # The last block merge uses the same TF32 MMA reduction as the torch reference.
        # Scalar FMA with truncated operands rounds at different points in the sum.
        mr, mc = (wid % 2) * 16, (wid // 2) * 16
        row, col = lane // 4, lane % 4
        for phase in cutlass.range_constexpr(2):
            cc0 = [cutlass.Float32(0.) for _ in range(4)]
            cc1 = [cutlass.Float32(0.) for _ in range(4)]
            for kk in cutlass.range_constexpr(0,32,8):
                if cutlass.const_expr(phase == 0):
                    aa = [sf[32+mr+row,32+kk+col], sf[32+mr+row+8,32+kk+col],
                          sf[32+mr+row,32+kk+col+4], sf[32+mr+row+8,32+kk+col+4]]
                    bb0 = [-sf[32+kk+col,mc+row], -sf[32+kk+col+4,mc+row]]
                    bb1 = [-sf[32+kk+col,mc+row+8], -sf[32+kk+col+4,mc+row+8]]
                else:
                    aa = [tmp[mr+row,kk+col], tmp[mr+row+8,kk+col],
                          tmp[mr+row,kk+col+4], tmp[mr+row+8,kk+col+4]]
                    bb0 = [sf[kk+col,mc+row], sf[kk+col+4,mc+row]]
                    bb1 = [sf[kk+col,mc+row+8], sf[kk+col+4,mc+row+8]]
                cc0 = _mma_tf32(aa,bb0,cc0)
                cc1 = _mma_tf32(aa,bb1,cc1)
            for j in cutlass.range_constexpr(4):
                rr, c0 = mr+row+(j//2)*8, mc+col*2+j%2
                if cutlass.const_expr(phase == 0):
                    tmp[rr,c0],tmp[rr,c0+8]=cc0[j],cc1[j]
                else:
                    sf[32+rr,c0],sf[32+rr,c0+8]=cc0[j],cc1[j]
            cute.arch.barrier()
        # A row ra, columns [32 * half, 32 * half + 32)
        ra, half = tid % C, tid // C
        for c8 in cutlass.range_constexpr(4):
            c0 = 32 * half + 8 * c8
            for j in cutlass.range_constexpr(8):
                x[j] = sf[ra, c0 + j]
            stg128((A.iterator + A.layout((b, n, ra, bh, c0))).toint(),
                   pack2(x[1], x[0]), pack2(x[3], x[2]), pack2(x[5], x[4]), pack2(x[7], x[6]))
        cute.arch.barrier()
        tmem.free(tptr, 64)


@cute.jit
def launch_masks(k2: cute.Tensor, q2: cute.Tensor, gc: cute.Tensor, beta: cute.Tensor,
                    A: cute.Tensor, Mp: cute.Tensor, M: cute.Tensor, Pm: cute.Tensor,
                    offsets: cute.Tensor, mapping: cute.Tensor, gKT: cute.Tensor, gQT: cute.Tensor,
                    B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr, NC: cutlass.Constexpr,
                    PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr, WITH_PM: cutlass.Constexpr,
                    SCALE: cutlass.Constexpr, stream: cuda.CUstream):
    bf = cutlass.BFloat16
    lBox = cute.slice_(sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, 64, 64), bf, 1), (None, None, 0))
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKT, lBox, (C, 64), 1)
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQT, lBox, (C, 64), 1)
    masks_kernel(k2, q2, gc, beta, A, Mp, M, Pm, offsets, mapping, tmaK, tKt, tmaQ, tQt, lBox,
                    H, E, NC, PACKED, NATIVE, WITH_PM, SCALE).launch(
        grid=(B * NC * H, 1, 1), block=(NT, 1, 1), stream=stream)



_compiled={}


def joint_masks_sm100(k,k2,q2,gc,beta,chunk_offsets=None,token_map=None,q=None,scale=None):
    """A, Mp, M (B,T,H,64) bf16 like sm90_joint_masks.joint_masks; with q also Pm = scale * tril(Q K^T * Mp) (as
    stored), returned as a fourth tensor."""
    B,T,H,D_=k.shape
    native=token_map is not None
    E,NC=k2.shape[-1],gc.shape[1]//C
    assert D_==D and (native or T%C==0)
    with_pm=q is not None
    scale=float(D**-0.5 if scale is None else scale)
    dev=k.device
    A,Mp,M=(torch.empty((B,NC*C,H,C),dtype=torch.bfloat16,device=dev) for _ in range(3))
    Pm=torch.empty((B,NC*C,H,C),dtype=torch.bfloat16,device=dev) if with_pm else A
    def ch(t):
        return t.view(B,NC,C,*t.shape[2:])
    def raw(t):
        return t.view(B,1,T,*t.shape[2:]) if native else ch(t)
    qq=q if with_pm else k
    tl=(raw(k2),raw(q2),ch(gc),raw(beta),ch(A),ch(Mp),ch(M),ch(Pm))
    args=[from_dlpack(t.detach(),assumed_align=16) for t in tl]
    packed=chunk_offsets is not None
    args.append(from_dlpack(chunk_offsets.detach(),assumed_align=16).mark_layout_dynamic() if packed else args[0])
    args.append(from_dlpack(token_map.detach(),assumed_align=16) if native else args[0])
    kq=[from_dlpack(raw(t).detach().permute(2,4,0,1,3),assumed_align=16) for t in (k,qq)]
    if native and T == 1 and H == 1:
        for a_ in kq:
            keep_singleton_tma_axis(a_, 0)
    args+=kq
    stream=cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    key=(B,T,H,E,packed,native,NC,with_pm,scale)
    if key not in _compiled:
        _compiled[key]=cute.compile(launch_masks,*args,B,H,E,NC,packed,native,with_pm,scale,stream)
    _compiled[key](*args,stream)
    return (A,Mp,M,Pm) if with_pm else (A,Mp,M)
