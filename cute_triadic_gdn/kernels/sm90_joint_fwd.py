"""Forward recurrence for E <= 4 (Eq. 11), warp-specialized: producer, state, value and output warpgroups; stores the
state at the start of every chunk in BF16 for the backward."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import warpgroup, cpasync
from cutlass.cute.nvgpu import OperandMajorMode as OMM
from cutlass.cute.runtime import from_dlpack
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, zero_tail, store_pair, keep_singleton_tma_axis

from cutlass._mlir.dialects import llvm
from .asm import _f32, _PURE


def fma(a,b,c):
    return cutlass.Float32(llvm.inline_asm(_f32(), [a.ir_value(), b.ir_value(), c.ir_value()],
        'fma.rn.f32 $0, $1, $2, $3;', '=f,f,f,f', **_PURE))


C = 64          # chunk
WG = 128        # threads per warpgroup



def _pack_bf16(hi, lo):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type,
        [hi.ir_value(), lo.ir_value()], "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f", **_PURE))


@cute.jit
def _store_mma_bf16(acc: cute.Tensor, smem: cute.Tensor, layout: cute.ComposedLayout,
                    ROWS: cutlass.Constexpr, COLS: cutlass.Constexpr,
                    stage: cutlass.Int32, lane: cutlass.Int32,
                    row_major: cutlass.Constexpr = False):
    # WGMMA's C fragment holds pairs at (row,col), (row+8,col),
    # (row,col+8), (row+8,col+8). STMatrix writes these four 8x8 tiles
    # directly, with the same single FP32 -> BF16 rounding as scalar stores.
    raw = cute.recast_ptr(smem.iterator, dtype=cutlass.Int32)
    for m in cutlass.range_constexpr(ROWS // 64):
        for d in cutlass.range_constexpr(COLS // 16):
            i = 8*d + (COLS//2)*m
            w00 = _pack_bf16(acc[i+1], acc[i])
            w10 = _pack_bf16(acc[i+3], acc[i+2])
            w01 = _pack_bf16(acc[i+5], acc[i+4])
            w11 = _pack_bf16(acc[i+7], acc[i+6])
            row = (lane//32)*16 + lane%8 + 8*((lane%32)//8%2) + 64*m
            col = 16*d + 8*((lane%32)//16)
            offset = (smem.layout((row,col,stage)) if cutlass.const_expr(row_major)
                      else smem.layout((col,row,stage))) * 2
            # Swizzles act on bytes; inline assembly uses an unswizzled pointer.
            shift = cutlass.const_expr(layout.inner.num_shift)
            mask = cutlass.const_expr(((1<<layout.inner.num_bits)-1)
                                     << (layout.inner.num_base+shift))
            offset = offset ^ ((offset & mask) >> shift)
            address = cutlass.Int32((raw + offset//4).toint())
            llvm.inline_asm(None, [address.ir_value(), w00.ir_value(), w10.ir_value(),
                                  w01.ir_value(), w11.ir_value()],
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1,$2,$3,$4};",
                "r,r,r,r,r", has_side_effects=True, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.kernel
def fwd_kernel(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor,
                  gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor,
                  tmaQ: cute.CopyAtom, rawQt: cute.Tensor,
                  tmaK: cute.CopyAtom, rawKt: cute.Tensor,
                  tmaV: cute.CopyAtom, rawVt: cute.Tensor,
                  tmaA: cute.CopyAtom, tAt: cute.Tensor,
                  tmaM: cute.CopyAtom, tMt: cute.Tensor,
                  tmaH: cute.CopyAtom, tHt: cute.Tensor,
                  gOffsets: cute.Tensor, mapping: cute.Tensor,
                  mmaP: cute.TiledMma, mmaO: cute.TiledMma, mmaS: cute.TiledMma,
                  lQ: cute.ComposedLayout, lK: cute.ComposedLayout, lKa: cute.ComposedLayout,
                  lKq: cute.ComposedLayout, lV: cute.ComposedLayout, lH: cute.ComposedLayout,
                  lVn: cute.ComposedLayout, lP: cute.ComposedLayout, lA: cute.ComposedLayout,
                  B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                  KD: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr, SAVE: cutlass.Constexpr,
                  SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    lane=tid%WG
    group=cute.arch.make_warp_uniform(tid//WG)
    warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
    NV=128//VC
    vb,bh,b=bid%NV,(bid//NV)%H,bid//(NV*H)
    if cutlass.const_expr(DOCS and NATIVE):
        b = cutlass.Int32(mapping[2, b])      # documents longest first (see metadata_kernel)
    first, count = cutlass.Int32(0), cutlass.Int32(NC)
    if cutlass.const_expr(DOCS):
        first = cutlass.Int32(gOffsets[b])
        count = cutlass.Int32(gOffsets[b+1]) - first
        b = cutlass.Int32(0)
    smem=utils.SmemAllocator()
    def alloc(layout):
        return smem.allocate_tensor(cutlass.BFloat16,layout.outer,128,swizzle=layout.inner)
    def view(t,layout):
        return cute.make_tensor(cute.recast_ptr(t.iterator,layout.inner,dtype=cutlass.BFloat16),layout.outer)
    def fp(shape,stride=None):
        return smem.allocate_tensor(cutlass.Float32,cute.make_layout(shape,stride=stride),byte_alignment=16)
    sQ,sK,sV,sA,sM=alloc(lQ),alloc(lK),alloc(lV),alloc(lA),alloc(lA)
    sH,sVn,sWt,sVd,sP=alloc(lH),alloc(lVn),alloc(lV),alloc(lV),alloc(lP)
    sKa,sKq,sQo=view(sK,lKa),view(sK,lKq),view(sQ,lKq)
    sa,sr,sc=[fp((2,E,C),(E*(C+32//E),C+32//E,1)) for _ in range(3)]
    sb,sd=fp((2,C),(C,1)),fp((2,E),(E,1))
    def bar(n=1):
        return smem.allocate_tensor(cutlass.Int64,cute.make_layout(n),byte_alignment=8)
    ready,free=bar(2),bar(2)
    tailReady=bar(2) if cutlass.const_expr(NATIVE == 1) else ready
    hReady,hFree,vnReady,vdReady,vdFree=bar(),bar(),bar(),bar(),bar()
    if tid<32:
        with cute.arch.elect_one():
            for st in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(ready.iterator+st,WG+1)
                if cutlass.const_expr(NATIVE == 1):
                    cute.arch.mbarrier_init(tailReady.iterator+st,WG)
                cute.arch.mbarrier_init(free.iterator+st,3*WG)
            cute.arch.mbarrier_init(hReady.iterator,WG)
            cute.arch.mbarrier_init(hFree.iterator,2*WG)
            cute.arch.mbarrier_init(vnReady.iterator,WG)
            cute.arch.mbarrier_init(vdReady.iterator,WG)
            cute.arch.mbarrier_init(vdFree.iterator,WG)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    def tma_part(atom,ten,sm,mma,tile,is_a):
        gt=cute.local_tile(ten,tile,(None,None,None,None,None))
        th=mma.get_slice(0)
        part=th.partition_A(gt) if is_a else th.partition_B(gt)
        return cpasync.tma_partition(atom,0,cute.make_layout(1),cute.group_modes(sm,0,2),cute.group_modes(part,0,3))
    if cutlass.const_expr(NATIVE):
        start = mapping[0, first]
        tQt = cute.domain_offset((start,0,0,0,0), rawQt)
        tKt = cute.domain_offset((start,0,0,0,0), rawKt)
        tVt = cute.domain_offset((0,start,0,0,0), rawVt)
    else:
        tQt, tKt, tVt = rawQt, rawKt, rawVt
    tQs,tQg=tma_part(tmaQ,tQt,sQ,mmaP,(C,KD),True)
    tKs,tKg=tma_part(tmaK,tKt,sK,mmaP,(C,KD),False)
    tVs,tVg=tma_part(tmaV,tVt,sV,mmaO,(VC,C),False)
    tAs,tAg=tma_part(tmaA,tAt,sA,mmaO,(C,C),True)
    tMs,tMg=tma_part(tmaM,tMt,sM,mmaO,(C,C),True)
    if cutlass.const_expr(SAVE):
        tHs,tHg=tma_part(tmaH,tHt,sH,mmaO,(VC,KD),False)
    tu,tp,ts=mmaO.get_slice(lane),mmaP.get_slice(lane),mmaS.get_slice(lane)
    cu=tu.partition_C(cute.make_identity_tensor((C,VC)))
    cp=tp.partition_C(cute.make_identity_tensor((C,C)))
    cs=ts.partition_C(cute.make_identity_tensor((KD,VC)))
    NU,NS,NP=C*VC//WG,KD*VC//WG,C*C//WG
    def sync(group):
        cute.arch.barrier(barrier_id=group+1,number_of_threads=WG)
    def gemm_async(mma,acc,fa,fb,add=False):
        warpgroup.fence()
        for k in cutlass.range(cute.size(fa,mode=[2]),unroll_full=True):
            mma.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(add or k!=0))
            cute.gemm(mma,acc,fa[None,None,k],fb[None,None,k],acc)
        warpgroup.commit_group()
    if group==3:
        cute.arch.warpgroup_reg_dealloc(64)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            if step>=2:
                cute.arch.mbarrier_wait(free.iterator+st,((step-2)//2)%2)
            rows = valid_rows(n, mapping, NATIVE)
            k2c = token_chunk(gK2, n, mapping, NATIVE)
            qq = token_chunk(gQ2, n, mapping, NATIVE)
            qb = token_chunk(gBt, n, mapping, NATIVE)
            if warp==12:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ready.iterator+st,4*C*KD+2*C*VC+4*C*C)
                if cutlass.const_expr(NATIVE):
                    cute.copy(tmaQ,tQg[(None,step,0,b,0,bh)],tQs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaK,tKg[(None,step,0,b,0,bh)],tKs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaV,tVg[(None,vb,step,b,0,bh)],tVs[(None,st)],tma_bar_ptr=ready.iterator+st)
                else:
                    cute.copy(tmaQ,tQg[(None,0,0,b,n,bh)],tQs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaK,tKg[(None,0,0,b,n,bh)],tKs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaV,tVg[(None,vb,0,b,n,bh)],tVs[(None,st)],tma_bar_ptr=ready.iterator+st)
                cute.copy(tmaA,tAg[(None,0,0,b,n,bh)],tAs[(None,st)],tma_bar_ptr=ready.iterator+st)
                cute.copy(tmaM,tMg[(None,0,0,b,n,bh)],tMs[(None,st)],tma_bar_ptr=ready.iterator+st)
            for j in cutlass.range_constexpr((E*C+WG-1)//WG):
                pos=j*WG+lane
                if pos<E*C:
                    e,r=pos%E,pos//E
                    g= gGc[b,n,r,bh,e]
                    gL=gGc[b,n,C-1,bh,e]
                    weight=masked_scalar(k2c,(b,n,r,bh,e),r,rows)
                    eg=cute.math.exp(g,fastmath=True)
                    sa[st,e,r]=weight*eg
                    sr[st,e,r]=weight*cute.math.exp(gL-g,fastmath=True)
                    sc[st,e,r]=SCALE*masked_scalar(qq,(b,n,r,bh,e),r,rows)*eg
            if lane<C:
                sb[st,lane]=masked_scalar(qb,(b,n,lane,bh),lane,rows)
            if lane<E:
                sd[st,lane]=cute.math.exp(gGc[b,n,C-1,bh,lane],fastmath=True)
            cute.arch.mbarrier_arrive(ready.iterator+st)
            if cutlass.const_expr(NATIVE):
                if rows < C:
                    cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
                    zero_tail(sQ,rows,lane,WG,KD,st,False)
                    zero_tail(sK,rows,lane,WG,KD,st,False)
                    zero_tail(sV,rows,lane,WG,VC,st,True)
                    cute.arch.fence_proxy('async.shared',space='cta')
                    sync(group)
                    cute.arch.mbarrier_arrive(tailReady.iterator+st)
    elif group==0:
        cute.arch.warpgroup_reg_alloc(160)
        states=[mmaS.make_fragment_C(mmaS.partition_shape_C((KD,VC))) for _ in range(E)]
        for e in cutlass.range_constexpr(E):
            states[e].fill(0.)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            if step>0:
                cute.arch.mbarrier_wait(hFree.iterator,(step-1)%2)
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for e in cutlass.range_constexpr(E):
                _store_mma_bf16(states[e],sH,lH,KD,VC,e,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            if cutlass.const_expr(SAVE):
                if warp==0:
                    for e in cutlass.range_constexpr(E):
                        cute.copy(tmaH,tHs[(None,e)],tHg[(None,vb,e,b,n,bh)])
                    cute.arch.cp_async_bulk_commit_group()
            cute.arch.mbarrier_arrive(hReady.iterator)
            cute.arch.mbarrier_wait(vnReady.iterator,step%2)
            fa=mmaS.make_fragment_A(ts.partition_A(sKa[None,None,st]))
            for e in cutlass.range_constexpr(E):
                decay=sd[st,e]
                for i in cutlass.range_constexpr(NS):
                    states[e][i]=states[e][i]*decay
                fb=mmaS.make_fragment_B(ts.partition_B(sVn[None,None,e]))
                gemm_async(mmaS,states[e],fa,fb,True)
            warpgroup.wait_group(0)
            if cutlass.const_expr(SAVE):
                if warp==0:
                    cute.arch.cp_async_bulk_wait_group(0,read=True)
            cute.arch.mbarrier_arrive(free.iterator+st)
        if cutlass.const_expr(SAVE):
            if warp==0:
                cute.arch.cp_async_bulk_wait_group(0)
    elif group==1:
        cute.arch.warpgroup_reg_alloc(144)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            cute.arch.mbarrier_wait(hReady.iterator,step%2)
            accs=[mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC))) for _ in range(E)]
            fa=mmaO.make_fragment_A(tu.partition_A(sKq[None,None,st]))
            for e in cutlass.range_constexpr(E):
                fb=mmaO.make_fragment_B(tu.partition_B(sH[None,None,e]))
                gemm_async(mmaO,accs[e],fa,fb)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(hFree.iterator)
            gates=cute.make_rmem_tensor((E,2),cutlass.Float32)
            for e in cutlass.range_constexpr(E):
                gates[e,0],gates[e,1]=sa[st,e,cu[0][0]],sa[st,e,cu[2][0]]
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                r,c=cu[i][0],cu[i][1]
                v0,v1=cutlass.Float32(sV[c,r,st]),cutlass.Float32(sV[c+1,r,st])
                for e in cutlass.range_constexpr(E):
                    weight=gates[e,j%2]
                    v0,v1=fma(-weight,accs[e][i],v0),fma(-weight,accs[e][i+1],v1)
                if cutlass.const_expr(SAVE):
                    pack[0],pack[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                    dst=cute.make_tensor((gWo.iterator+gWo.layout((b,n,r,bh,vb*VC+c))).align(4),cute.make_layout((2,)))
                    dst.store(pack.load())
                pack[0],pack[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                accs[0][i],accs[0][i+1]=v0,v1
            _store_mma_bf16(accs[0],sWt,lV,C,VC,0,lane)
            # Match the reference's BF16 factorization: (A diag(beta)) @ W.
            # Scaling W instead is algebraically equivalent but rounds differently
            # at every recurrent layer. Keep the saved, global A unscaled.
            ab=mmaP.make_fragment_C(mmaP.partition_shape_C((C,C)))
            for i in cutlass.range_constexpr(NP):
                r,c=cp[i][0],cp[i][1]
                ab[i]=cutlass.Float32(sA[r,c,st])*sb[st,c]
            _store_mma_bf16(ab,sA,lA,C,C,st,lane,True)
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            fa=mmaO.make_fragment_A(tu.partition_A(sA[None,None,st]))
            fb=mmaO.make_fragment_B(tu.partition_B(sWt[None,None,0]))
            gemm_async(mmaO,accs[0],fa,fb)
            warpgroup.wait_group(0)
            if step>0:
                cute.arch.mbarrier_wait(vdFree.iterator,(step-1)%2)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                r,c=cu[i][0],cu[i][1]
                pack[0],pack[1]=cutlass.BFloat16(accs[0][i]),cutlass.BFloat16(accs[0][i+1])

                if cutlass.const_expr(SAVE):
                    dst=cute.make_tensor((gVdo.iterator+gVdo.layout((b,n,r,bh,vb*VC+c))).align(4),cute.make_layout((2,)))
                    dst.store(pack.load())
            _store_mma_bf16(accs[0],sVd,lV,C,VC,0,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            cute.arch.mbarrier_arrive(vdReady.iterator)
            vn=mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC)))
            for e in cutlass.range_constexpr(E):
                weight0=sr[st,e,cu[0][0]]
                weight1=sr[st,e,cu[2][0]]
                for j in cutlass.range_constexpr(NU//2):
                    i=2*j
                    weight=weight0 if cutlass.const_expr(j%2==0) else weight1
                    vn[i],vn[i+1]=accs[0][i]*weight,accs[0][i+1]*weight
                _store_mma_bf16(vn,sVn,lVn,C,VC,e,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            cute.arch.mbarrier_arrive(vnReady.iterator)
            cute.arch.mbarrier_arrive(free.iterator+st)
    else:
        cute.arch.warpgroup_reg_alloc(144)
        for step in cutlass.range(count):
            n = first + step
            rows_out = valid_rows(n,mapping,NATIVE)
            out_chunk = token_chunk(gO,n,mapping,NATIVE)
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            ap=mmaP.make_fragment_C(mmaP.partition_shape_C((C,C)))
            fa=mmaP.make_fragment_A(tp.partition_A(sQ[None,None,st]))
            fb=mmaP.make_fragment_B(tp.partition_B(sK[None,None,st]))
            gemm_async(mmaP,ap,fa,fb)
            warpgroup.wait_group(0)
            for i in cutlass.range_constexpr(NP):
                r,c=cp[i][0],cp[i][1]
                val=cutlass.Float32(0.)
                if c<=r:
                    val=ap[i]*(SCALE*cutlass.Float32(sM[r,c,st]))
                ap[i]=val
            _store_mma_bf16(ap,sP,lP,C,C,0,lane,True)
            cute.arch.mbarrier_wait(hReady.iterator,step%2)
            accs=[mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC))) for _ in range(E)]
            ao=mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC)))
            fa=mmaO.make_fragment_A(tu.partition_A(sQo[None,None,st]))
            for e in cutlass.range_constexpr(E):
                fb=mmaO.make_fragment_B(tu.partition_B(sH[None,None,e]))
                gemm_async(mmaO,accs[e],fa,fb)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(hFree.iterator)
            for i in cutlass.range_constexpr(NU):
                val=cutlass.Float32(0.)
                # The first readout addition contracts c0*u0 into c1*u1.
                for e in cutlass.range_constexpr(E):
                    sl=(1-e if e<2 else e) if E>1 else e
                    val=fma(sc[st,sl,cu[i][0]],accs[sl][i],val)
                ao[i]=val
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            cute.arch.mbarrier_arrive(free.iterator+st)
            cute.arch.mbarrier_wait(vdReady.iterator,step%2)
            fa=mmaO.make_fragment_A(tu.partition_A(sP[None,None,0]))
            fb=mmaO.make_fragment_B(tu.partition_B(sVd[None,None,0]))
            gemm_async(mmaO,ao,fa,fb,True)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(vdFree.iterator)
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                pack[0],pack[1]=cutlass.BFloat16(ao[i]),cutlass.BFloat16(ao[i+1])
                store_pair(out_chunk,b,n,cu[i][0],bh,vb*VC+cu[i][1],pack.load(),rows_out)


@cute.jit
def launch_fwd(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor,
                        gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor, gQA: cute.Tensor, gKB: cute.Tensor, gVB: cute.Tensor,
                        gAA: cute.Tensor, gMA: cute.Tensor, gHT: cute.Tensor, gOffsets: cute.Tensor, mapping: cute.Tensor,
                        B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                        KD: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr, SAVE: cutlass.Constexpr,
                        SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NBLK: cutlass.Int32, NATIVE: cutlass.Constexpr, stream: cuda.CUstream):
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    one = (1, 1, 1)
    mmaP = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.K, OMM.K, f32, one, (C, C))      # Q K^T
    mmaO = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.K, OMM.MN, f32, one, (C, VC))    # K@S, A@W, P@Vd, Q@S
    mmaS = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.MN, OMM.MN, f32, one, (C, VC))   # K^T@vn (M=KD via M-iters)
    lQ = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, KD), bf16, 2)
    lK = sm90.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (C, C, KD), bf16, 2)
    lKa = sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR, (KD, VC, C), bf16, 2)
    lKq = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, KD), bf16, 2)
    lV = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, C), bf16, 2)
    lH = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, KD), bf16, E)
    lVn = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, C), bf16, E)
    lP = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, C), bf16, 1)
    lA = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, C), bf16, 2)
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQA, cute.slice_(lQ, (None, None, 0)), (C, KD), 1)
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKB, cute.slice_(lK, (None, None, 0)), (C, KD), 1)
    tmaV, tVt = cpasync.make_tiled_tma_atom(op, gVB, cute.slice_(lV, (None, None, 0)), (VC, C), 1)
    tmaA, tAt = cpasync.make_tiled_tma_atom(op, gAA, cute.slice_(lA, (None, None, 0)), (C, C), 1)
    tmaM, tMt = cpasync.make_tiled_tma_atom(op, gMA, cute.slice_(lA, (None, None, 0)), (C, C), 1)
    tmaH,tHt=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),gHT,
        cute.slice_(lH,(None,None,0)),(VC,KD),1)
    fwd_kernel(gGc, gK2, gQ2, gBt, gO, gHs, gWo, gVdo, tmaQ, tQt, tmaK, tKt, tmaV, tVt, tmaA, tAt, tmaM, tMt, tmaH, tHt, gOffsets, mapping,
                  mmaP, mmaO, mmaS, lQ, lK, lKa, lKq, lV, lH, lVn, lP, lA,
                  B, H, E, KD, VC, NC, SAVE, SCALE, DOCS, NATIVE).launch(
        grid=(NBLK * H * (128 // VC), 1, 1), block=(4 * WG, 1, 1), stream=stream, min_blocks_per_mp=1)


_compiled = {}


def joint_fwd(q, k, v, k2, q2, gc, beta, A, Mp, scale=None, value_cols=None, save=False,
                         chunk_starts=None, token_map=None):
    """Return output, or (output, states, W, Vd) when save=True.

    q,k,v (B,T,H,128) bf16; k2,q2,gc (B,T,H,E) f32 (gc = chunk-local cumsum); beta (B,T,H) f32;
    A, Mp (B,T,H,64) or (B,NC,H,64,64) contiguous bf16 from joint_masks.  Returns o (B,T,H,128) bf16."""
    B, Tn, H, KD = q.shape
    DV = v.shape[-1]
    E = k2.shape[-1]
    native = token_map is not None
    assert KD == 128 and DV == 128 and (native or Tn % C == 0) and DV % E == 0
    assert E in (1, 2, 4, 8), "wgmma requires at least 8 value columns per CTA"
    VC = 128//E if value_cols is None else value_cols
    assert VC in (16,32,64) and E*VC <= 128
    NC = gc.shape[1] // C
    scale = KD ** -0.5 if scale is None else float(scale)
    dev = q.device
    # TMA needs row-major A and Mp (the torch reference's linalg.inv output is column-major)
    A, Mp = A.contiguous(), Mp.contiguous()
    o = torch.empty(B, Tn, H, DV, device=dev, dtype=torch.bfloat16)
    hs = torch.empty((B,NC,H,E,KD,DV),device=dev,dtype=q.dtype) if save else o
    wo,vdo = (torch.empty((B,NC*C,H,DV),device=dev,dtype=o.dtype) for _ in range(2)) if save else (o,o)
    def matrix_view(t):
        if t.ndim == 4:
            return t.view(B,NC,C,H,C).permute(2,4,0,1,3)
        return t.permute(3,4,0,1,2)
    def raw(t):
        return t.view(B,1,Tn,*t.shape[2:]) if native else t.view(B,NC,C,*t.shape[2:])
    q5, k5, v5 = (raw(t) for t in (q,k,v))
    args = (gc.view(B, NC, C, H, E), raw(k2), raw(q2),
            raw(beta), raw(o),
            hs, wo.view(B,NC,C,H,DV) if save else raw(o), vdo.view(B,NC,C,H,DV) if save else raw(o),
            q5.permute(2, 4, 0, 1, 3),        # (C, KD, B, NC, H)   A operand of P/O
            k5.permute(2, 4, 0, 1, 3),        # (C, KD, B, NC, H)
            v5.permute(4, 2, 0, 1, 3),        # (DV, C, B, NC, H)   MN-major
            matrix_view(A),         # (64, 64, B, NC, H)
            matrix_view(Mp),
            hs.view(B,NC,H,E*KD,DV).permute(4,3,0,1,2) if save else q5.permute(4,2,0,1,3))
    ta = [from_dlpack(t.detach(), assumed_align=16) for t in args]
    if native and Tn == 1 and H == 1:
        for index, axis in ((8, 0), (9, 0), (10, 1)):
            keep_singleton_tma_axis(ta[index], axis)
        if not save:
            keep_singleton_tma_axis(ta[13], 1)  # unused state-descriptor placeholder
    # packed documents: chunk_starts holds every document's first chunk and the total, one recurrence per document
    docs = chunk_starts.numel() - 1 if chunk_starts is not None else 0
    assert not docs or B == 1
    ta.append(ta[0] if chunk_starts is None else from_dlpack(chunk_starts.detach(), assumed_align=16).mark_layout_dynamic())
    ta.append(from_dlpack(token_map.detach(), assumed_align=16) if native else ta[0])
    key = (B, Tn, H, E, scale, VC, bool(save), bool(docs), native, NC, tuple(A.stride()), tuple(Mp.stride()))
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_fwd, *ta, B, H, E, KD, VC, NC, bool(save), float(scale), bool(docs), docs if docs else B, int(native), stream)
    _compiled[key](*ta, docs if docs else B, stream)      # NBLK is a runtime argument
    return (o,hs,wo,vdo) if save else o
