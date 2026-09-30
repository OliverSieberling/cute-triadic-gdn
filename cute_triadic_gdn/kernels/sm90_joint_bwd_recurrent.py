"""Backward, first stage: a short reverse recurrence over the chunks that stores the gradient of every state
slice in BF16, so that the remaining gradients can be formed for every chunk in parallel (sm90_joint_bwd_parallel).
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cutlass.cute.nvgpu import warpgroup, cpasync
from cutlass.cute.nvgpu import OperandMajorMode as OMM
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, zero_tail, store_pair, keep_singleton_tma_axis

C,D,WG=64,128,128


@cute.kernel
def recurrent_kernel(q:cute.Tensor,k:cute.Tensor,do:cute.Tensor,A:cute.Tensor,Mp:cute.Tensor,
                     gc:cute.Tensor,k2:cute.Tensor,q2:cute.Tensor,beta:cute.Tensor,
                     dsout:cute.Tensor,dwf:cute.Tensor,dv:cute.Tensor,dvd:cute.Tensor,
                     tmaDS:cute.CopyAtom,tDSt:cute.Tensor,
                     tmaQ:cute.CopyAtom,rawQt:cute.Tensor,tmaK:cute.CopyAtom,rawKt:cute.Tensor,
                     tmaDO:cute.CopyAtom,rawDOt:cute.Tensor,tmaA:cute.CopyAtom,tAt:cute.Tensor,tmaM:cute.CopyAtom,tMt:cute.Tensor,
                     gOffsets:cute.Tensor,mapping:cute.Tensor,
                     mmaP:cute.TiledMma,mmaU:cute.TiledMma,mmaT:cute.TiledMma,mmaS:cute.TiledMma,
                     lK:cute.ComposedLayout,lKt:cute.ComposedLayout,lA:cute.ComposedLayout,lAt:cute.ComposedLayout,
                     lH:cute.ComposedLayout,lV:cute.ComposedLayout,lVs:cute.ComposedLayout,
                     H:cutlass.Constexpr,E:cutlass.Constexpr,NC:cutlass.Constexpr,VC:cutlass.Constexpr,
                     NW:cutlass.Constexpr,SCALE:cutlass.Constexpr,DOCS:cutlass.Constexpr,NATIVE:cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    NV=D//VC
    vb,bh,b=bid%NV,(bid//NV)%H,bid//(NV*H)
    if cutlass.const_expr(DOCS and NATIVE):
        b = cutlass.Int32(mapping[2, b])      # documents longest first (see metadata_kernel)
    first, count = cutlass.Int32(0), cutlass.Int32(NC)
    if cutlass.const_expr(DOCS):
        first = cutlass.Int32(gOffsets[b])
        count = cutlass.Int32(gOffsets[b+1]) - first
        b = cutlass.Int32(0)
    group=cute.arch.make_warp_uniform(tid//WG)
    lane=tid%WG
    EP=E//NW
    smem=utils.SmemAllocator()
    def alloc(layout):
        return smem.allocate_tensor(cutlass.BFloat16,layout.outer,128,swizzle=layout.inner)
    def view(t,layout):
        return cute.make_tensor(cute.recast_ptr(t.iterator,layout.inner,dtype=cutlass.BFloat16),layout.outer)
    def fp(shape,stride=None):
        return smem.allocate_tensor(cutlass.Float32,cute.make_layout(shape,stride=stride),byte_alignment=16)
    sQ,sK=alloc(lK),alloc(lK)
    sQt,sKt=view(sQ,lKt),view(sK,lKt)
    sA,sM=alloc(lA),alloc(lA)
    sP=sM  # Each QK result overwrites only its own element of R'.
    sAt,sPt=view(sA,lAt),view(sP,lAt)
    sDO,sDVD=alloc(lV),alloc(lV)
    sDS=alloc(lH)
    sVN,sVR=alloc(lVs),alloc(lVs)
    sa,sr,sc=fp((E,C),(C+32//E,1)),fp((E,C),(C+32//E,1)),fp((E,C),(C+32//E,1))
    sd=fp((E,))
    sdw=fp((C,VC),(VC+4,1))
    sb=fp((2,C))                      # beta of the current and the next chunk (masked), staged a step ahead
    ready=smem.allocate_tensor(cutlass.Int64,cute.make_layout(1),byte_alignment=8)
    if tid<32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(ready.iterator,1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    def tma_part(atom,ten,sm,mma,tile,is_a):
        gt=cute.local_tile(ten,tile,(None,None,None,None,None))
        part=mma.get_slice(0).partition_A(gt) if is_a else mma.get_slice(0).partition_B(gt)
        return cpasync.tma_partition(atom,0,cute.make_layout(1),cute.group_modes(sm,0,2),cute.group_modes(part,0,3))
    if cutlass.const_expr(NATIVE):
        start = mapping[0,first]
        tQt = cute.domain_offset((start,0,0,0,0),rawQt)
        tKt = cute.domain_offset((start,0,0,0,0),rawKt)
        tDOt = cute.domain_offset((0,start,0,0,0),rawDOt)
    else:
        tQt,tKt,tDOt = rawQt,rawKt,rawDOt
    tQs,tQg=tma_part(tmaQ,tQt,sQ,mmaP,(C,D),True)
    tKs,tKg=tma_part(tmaK,tKt,sK,mmaP,(C,D),True)
    tDOs,tDOg=tma_part(tmaDO,tDOt,sDO,mmaU,(VC,C),False)
    tAs,tAg=tma_part(tmaA,tAt,sA,mmaP,(C,C),True)
    tMs,tMg=tma_part(tmaM,tMt,sM,mmaP,(C,C),True)
    tp,tu,ts=mmaP.get_slice(lane),mmaU.get_slice(lane),mmaS.get_slice(lane)
    cp=tp.partition_C(cute.make_identity_tensor((C,C)))
    cu=tu.partition_C(cute.make_identity_tensor((C,VC)))
    cs=ts.partition_C(cute.make_identity_tensor((D,VC)))
    gDS=cute.local_tile(tDSt,(VC,D),(None,None,None,None,None))
    tgDS=mmaU.get_slice(0).partition_B(gDS)
    tDSs,tDSg=cpasync.tma_partition(tmaDS,0,cute.make_layout(1),
        cute.group_modes(sDS,0,2),cute.group_modes(tgDS,0,3))
    NS,NU=D*VC//WG,C*VC//WG
    state=[mmaS.make_fragment_C(mmaS.partition_shape_C((D,VC))) for _ in range(EP)]
    for e in cutlass.range_constexpr(EP):
        state[e].fill(0.)
    def sync(group):
        cute.arch.barrier(barrier_id=group+1,number_of_threads=WG)
    def gemm(lane,mma,acc,aa,bb,accumulate=False):
        th=mma.get_slice(lane)
        fa=mma.make_fragment_A(th.partition_A(aa))
        fb=mma.make_fragment_B(th.partition_B(bb))
        warpgroup.fence()
        for j in cutlass.range(cute.size(fa,mode=[2]),unroll_full=True):
            mma.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(accumulate or j!=0))
            cute.gemm(mma,acc,fa[None,None,j],fb[None,None,j],acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    for it in cutlass.range(count):
        n=first+count-1-it
        rows = valid_rows(n,mapping,NATIVE)
        k2c = token_chunk(k2,n,mapping,NATIVE)
        qq = token_chunk(q2,n,mapping,NATIVE)
        bb = token_chunk(beta,n,mapping,NATIVE)
        dvout = token_chunk(dv,n,mapping,NATIVE)
        if it==0:
            if tid<C:
                sb[0,tid]=masked_scalar(bb,(b,n,tid,bh),tid,rows)
            cute.arch.barrier()
        if tid<32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ready.iterator,4*C*D+2*C*VC+4*C*C)
            if cutlass.const_expr(NATIVE):
                step = count-1-it
                cute.copy(tmaQ,tQg[(None,step,0,b,0,bh)],tQs[(None,0)],tma_bar_ptr=ready.iterator)
                cute.copy(tmaK,tKg[(None,step,0,b,0,bh)],tKs[(None,0)],tma_bar_ptr=ready.iterator)
                cute.copy(tmaDO,tDOg[(None,vb,step,b,0,bh)],tDOs[(None,0)],tma_bar_ptr=ready.iterator)
            else:
                cute.copy(tmaQ,tQg[(None,0,0,b,n,bh)],tQs[(None,0)],tma_bar_ptr=ready.iterator)
                cute.copy(tmaK,tKg[(None,0,0,b,n,bh)],tKs[(None,0)],tma_bar_ptr=ready.iterator)
                cute.copy(tmaDO,tDOg[(None,vb,0,b,n,bh)],tDOs[(None,0)],tma_bar_ptr=ready.iterator)
            cute.copy(tmaA,tAg[(None,0,0,b,n,bh)],tAs[(None,0)],tma_bar_ptr=ready.iterator)
            cute.copy(tmaM,tMg[(None,0,0,b,n,bh)],tMs[(None,0)],tma_bar_ptr=ready.iterator)
        for j in cutlass.range_constexpr((E*C+NW*WG-1)//(NW*WG)):
            pos=j*NW*WG+tid
            if pos<E*C:
                e,r=pos%E,pos//E
                g=gc[b,n,r,bh,e]
                gL=gc[b,n,C-1,bh,e]
                eg=cute.math.exp(g,fastmath=True)
                sa[e,r]=masked_scalar(k2c,(b,n,r,bh,e),r,rows)*eg
                sr[e,r]=masked_scalar(k2c,(b,n,r,bh,e),r,rows)*cute.math.exp(gL-g,fastmath=True)
                sc[e,r]=SCALE*masked_scalar(qq,(b,n,r,bh,e),r,rows)*eg
                if r==C-1:
                    sd[e]=eg
        for e in cutlass.range_constexpr(EP):
            sl=group*EP+e
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for j in cutlass.range_constexpr(NS//2):
                i=2*j
                pack[0],pack[1]=cutlass.BFloat16(state[e][i]),cutlass.BFloat16(state[e][i+1])
                dst=cute.make_tensor((sDS.iterator+sDS.layout((cs[i][1],cs[i][0],sl))).align(4),cute.make_layout((2,)))
                dst.store(pack.load())
        cute.arch.mbarrier_wait(ready.iterator,it%2)
        if cutlass.const_expr(NATIVE):
            if rows < C:
                zero_tail(sQ,rows,tid,NW*WG,D,0,False)
                zero_tail(sK,rows,tid,NW*WG,D,0,False)
                zero_tail(sDO,rows,tid,NW*WG,VC,0,True)
        for j in cutlass.range_constexpr(C*C//(NW*WG)):
            pos=j*NW*WG+tid
            r,c=pos//C,pos%C
            sA[r,c,0]=cutlass.BFloat16(cutlass.Float32(sA[r,c,0])*sb[it%2,c])
        cute.arch.fence_proxy('async.shared',space='cta')
        cute.arch.barrier()
        # One dS-slice TMA store per warp: warp 0 issuing all E of them held its warpgroup's
        # wgmma (warpgroup-collective) back by about a microsecond per chunk.
        for sl in cutlass.range_constexpr(E):
            if tid//32==sl%(NW*4):
                cute.copy(tmaDS,tDSs[(None,sl)],tDSg[(None,vb,sl,b,n,bh)])
        cute.arch.cp_async_bulk_commit_group()
        if group==NW-1:
            # P on the last warpgroup while warpgroup 0 forms every K dS_e term
            ap=mmaP.make_fragment_C(mmaP.partition_shape_C((C,C)))
            gemm(lane,mmaP,ap,sQ[None,None,0],sK[None,None,0])
            # masked, scaled P in bf16 pairs: one 4-byte load of M' and one 4-byte store per pair
            # (the two lanes are extracted before the staged masks: the DSL cannot use a vector
            # element defined outside an if inside it)
            packp=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for jp in cutlass.range_constexpr(C*C//WG//2):
                i=2*jp
                r,c=cp[i][0],cp[i][1]
                mv=cute.make_tensor((sM.iterator+sM.layout((r,c,0))).align(4),cute.make_layout((2,))).load()
                m0=cutlass.Float32(mv[0])
                m1=cutlass.Float32(mv[1])
                v0=SCALE*ap[i]*m0
                v1=SCALE*ap[i+1]*m1
                if c>r:
                    v0=cutlass.Float32(0.)
                if c+1>r:
                    v1=cutlass.Float32(0.)
                packp[0],packp[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                dstp=cute.make_tensor((sP.iterator+sP.layout((r,c,0))).align(4),cute.make_layout((2,)))
                dstp.store(packp.load())
            cute.arch.fence_proxy('async.shared',space='cta')
        au=mmaU.make_fragment_C(mmaU.partition_shape_C((C,VC)))
        av=mmaU.make_fragment_C(mmaU.partition_shape_C((C,VC)))
        av.fill(0.)
        av2=mmaU.make_fragment_C(mmaU.partition_shape_C((C,VC)))
        av2.fill(0.)
        # Warpgroup 0 forms every slice's K dS_e term: the first half of the slices into av and the
        # second into av2, added as the two warpgroups' partials were ((0 + first) + second).
        if group==0:
            for e in cutlass.range_constexpr(EP):
                gemm(lane,mmaU,au,sK[None,None,0],sDS[None,None,e])
                for i in cutlass.range_constexpr(NU):
                    av[i]=av[i]+sr[e,cu[i][0]]*au[i]
            if cutlass.const_expr(NW>1):
                for e in cutlass.range_constexpr(E-EP):
                    gemm(lane,mmaU,au,sK[None,None,0],sDS[None,None,EP+e])
                    for i in cutlass.range_constexpr(NU):
                        av2[i]=av2[i]+sr[EP+e,cu[i][0]]*au[i]
        cute.arch.barrier()
        if group==0:
            for i in cutlass.range_constexpr(NU):
                vsum=cutlass.Float32(0.)+av[i]
                if cutlass.const_expr(NW>1):
                    vsum=vsum+av2[i]
                av[i]=vsum
            gemm(lane,mmaT,av,sPt[None,None,0],sDO[None,None,0],True)
            packv=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for jp in cutlass.range_constexpr(NU//2):
                i=2*jp
                r,c=cu[i][0],cu[i][1]
                packv[0],packv[1]=cutlass.BFloat16(av[i]),cutlass.BFloat16(av[i+1])
                sDVD[c,r,0]=packv[0]
                sDVD[c+1,r,0]=packv[1]
                dstv=cute.make_tensor((dvd.iterator+dvd.layout((b,n,r,bh,vb*VC+c))).align(4),cute.make_layout((2,)))
                dstv.store(packv.load())
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            gemm(lane,mmaT,au,sAt[None,None,0],sDVD[None,None,0])
            pairw=cute.make_rmem_tensor((2,),cutlass.Float32)
            packw=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for jp in cutlass.range_constexpr(NU//2):
                i=2*jp
                r,c=cu[i][0],cu[i][1]
                pairw[0],pairw[1]=au[i],au[i+1]
                cute.make_tensor((sdw.iterator+sdw.layout((r,c))).align(8),cute.make_layout((2,))).store(pairw.load())
                cute.make_tensor((dwf.iterator+dwf.layout((b,n,r,bh,vb*VC+c))).align(8),cute.make_layout((2,))).store(pairw.load())
                packw[0],packw[1]=cutlass.BFloat16(au[i]),cutlass.BFloat16(au[i+1])
                store_pair(dvout,b,n,r,bh,vb*VC+c,packw.load(),rows)
        cute.arch.barrier()
        # VN_e = -a_e (x) dW and VR_e = c_e (x) dO for this warpgroup's slices: each dW pair and dO
        # pair is loaded once and used for every slice (same per-element arithmetic and stores).
        pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
        for j in cutlass.range_constexpr(NU//2):
            i=2*j
            r,c=cu[i][0],cu[i][1]
            vals=cute.make_tensor((sdw.iterator+sdw.layout((r,c))).align(8),cute.make_layout((2,))).load()
            w0=vals[0]
            w1=vals[1]
            o0=cutlass.Float32(sDO[c,r,0])
            o1=cutlass.Float32(sDO[c+1,r,0])
            for e in cutlass.range_constexpr(EP):
                sl=group*EP+e
                a_,c_=sa[sl,r],sc[sl,r]
                pack[0],pack[1]=cutlass.BFloat16(-a_*w0),cutlass.BFloat16(-a_*w1)
                dst=cute.make_tensor((sVN.iterator+sVN.layout((c,r,sl))).align(4),cute.make_layout((2,)))
                dst.store(pack.load())
                pack[0]=cutlass.BFloat16(c_*o0)
                pack[1]=cutlass.BFloat16(c_*o1)
                dst=cute.make_tensor((sVR.iterator+sVR.layout((c,r,sl))).align(4),cute.make_layout((2,)))
                dst.store(pack.load())
        for e in cutlass.range_constexpr(EP):
            sl=group*EP+e
            for i in cutlass.range_constexpr(NS):
                state[e][i]=state[e][i]*sd[sl]
        cute.arch.fence_proxy('async.shared',space='cta')
        sync(group)
        warpgroup.fence()
        faK=mmaS.make_fragment_A(ts.partition_A(sKt[None,None,0]))
        faQ=mmaS.make_fragment_A(ts.partition_A(sQt[None,None,0]))
        for e in cutlass.range_constexpr(EP):
            sl=group*EP+e
            fbN=mmaS.make_fragment_B(ts.partition_B(sVN[None,None,sl]))
            fbR=mmaS.make_fragment_B(ts.partition_B(sVR[None,None,sl]))
            for j in cutlass.range(cute.size(faK,mode=[2]),unroll_full=True):
                mmaS.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(True))
                cute.gemm(mmaS,state[e],faK[None,None,j],fbN[None,None,j],state[e])
            for j in cutlass.range(cute.size(faQ,mode=[2]),unroll_full=True):
                mmaS.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(True))
                cute.gemm(mmaS,state[e],faQ[None,None,j],fbR[None,None,j],state[e])
        warpgroup.commit_group()
        if it+1<count:
            # next chunk's beta, staged while the state gemms run (visible after the step's barrier)
            nn=n-1
            rows_next=valid_rows(nn,mapping,NATIVE)
            bb_next=token_chunk(beta,nn,mapping,NATIVE)
            if tid<C:
                sb[(it+1)%2,tid]=masked_scalar(bb_next,(b,nn,tid,bh),tid,rows_next)
        warpgroup.wait_group(0)
        cute.arch.cp_async_bulk_wait_group(0,read=True)     # every thread waits for its own (possibly empty) groups
        cute.arch.barrier()
    cute.arch.cp_async_bulk_wait_group(0)


@cute.kernel
def state_grads_kernel(q:cute.Tensor,k:cute.Tensor,do:cute.Tensor,vd:cute.Tensor,dwf:cute.Tensor,
                       gc:cute.Tensor,k2:cute.Tensor,q2:cute.Tensor,h:cute.Tensor,ds:cute.Tensor,
                       dqp:cute.Tensor,dkp:cute.Tensor,dgv:cute.Tensor,dk2v:cute.Tensor,dgo:cute.Tensor,dq2o:cute.Tensor,
                       tmaH:cute.CopyAtom,tHt:cute.Tensor,tmaDS:cute.CopyAtom,tDSt:cute.Tensor,offsets:cute.Tensor,mapping:cute.Tensor,
                       mmaU:cute.TiledMma,mmaG:cute.TiledMma,
                       lK:cute.ComposedLayout,lH:cute.ComposedLayout,lHg:cute.ComposedLayout,
                       H:cutlass.Constexpr,E:cutlass.Constexpr,NC:cutlass.Constexpr,SCALE:cutlass.Constexpr,PACKED:cutlass.Constexpr,NATIVE:cutlass.Constexpr,NBLK:cutlass.Constexpr,NITEMS:cutlass.Constexpr):
    # Persistent: each CTA walks chunk-heads bid, bid+NBLK, ...; the two-stage h/ds ring runs across them,
    # and the last slot of a chunk-head issues the first slot of the next one, so the next prologue's
    # operand loads overlap that transfer instead of following it.
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    group=cute.arch.make_warp_uniform(tid//WG)
    lane=tid%WG
    smem=utils.SmemAllocator()
    def alloc(layout):
        return smem.allocate_tensor(cutlass.BFloat16,layout.outer,128,swizzle=layout.inner)
    def view(t,layout):
        return cute.make_tensor(cute.recast_ptr(t.iterator,layout.inner,dtype=cutlass.BFloat16),layout.outer)
    def fp(shape,stride=None):
        return smem.allocate_tensor(cutlass.Float32,cute.make_layout(shape,stride=stride),byte_alignment=16)
    sQ,sK,sDO=alloc(lK),alloc(lK),alloc(lK)
    sH,sDS=alloc(lH),alloc(lH)
    sHg,sDSg=view(sH,lHg),view(sDS,lHg)
    sDW,sVD=alloc(lK),alloc(lK)
    sd1,sd2,sd3=fp((C,)),fp((C,)),fp((C,))
    sg,sk2,sq=fp((E,C),(C,1)),fp((E,C),(C,1)),fp((E,C),(C,1))
    sa,sr,sc=fp((C,)),fp((C,)),fp((C,))
    sdot=fp((8,))
    ready=smem.allocate_tensor(cutlass.Int64,cute.make_layout(2),byte_alignment=8)
    if tid<32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(ready.iterator,1)
            cute.arch.mbarrier_init(ready.iterator+1,1)
    cute.arch.mbarrier_init_fence()
    def tma_parts(atom,ten,sm):
        gt=cute.local_tile(ten,(D,D),(None,None,None,None,None))
        part=mmaU.get_slice(0).partition_B(gt)
        return cpasync.tma_partition(atom,0,cute.make_layout(1),
            cute.group_modes(sm,0,2),cute.group_modes(part,0,3))
    tHs,tHg=tma_parts(tmaH,tHt,sH)
    tDSs,tDSg=tma_parts(tmaDS,tDSt,sDS)
    def issue_slot(tid,tmaH,tHg,tHs,tmaDS,tDSg,tDSs,ready,stage,e,b,n,bh):
        if tid<32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ready.iterator+stage,4*D*D)
            cute.copy(tmaH,tHg[(None,0,e,b,n,bh)],tHs[(None,stage)],tma_bar_ptr=ready.iterator+stage)
            cute.copy(tmaDS,tDSg[(None,0,e,b,n,bh)],tDSs[(None,stage)],tma_bar_ptr=ready.iterator+stage)

    def load128(src,dst,limit,b,n,bh,tid):
        # all of a thread's 16-byte loads of the tile are issued before any of its shared-memory
        # stores (a store after each load left one global-load latency per row)
        vecs=cute.make_rmem_tensor((C*D//(2*WG*8),8),cutlass.BFloat16)
        for j in cutlass.range_constexpr(C*D//(2*WG*8)):
            pos=j*2*WG+tid
            r,c=pos//(D//8),(pos%(D//8))*8
            ptr=cute.make_tensor((src.iterator+src.layout((b,n,r,bh,c))).align(16),cute.make_layout((8,)))
            vec=cute.make_fragment_like(ptr)
            vec.fill(0.)
            if r < limit:
                cute.autovec_copy(ptr,vec)
            for z in cutlass.range_constexpr(8):
                vecs[j,z]=vec[z]
        for j in cutlass.range_constexpr(C*D//(2*WG*8)):
            pos=j*2*WG+tid
            r,c=pos//(D//8),(pos%(D//8))*8
            off=(c//64)*4096+r*64+8*((c//8)%8 ^ (r%8))+c%8
            raw=cute.recast_ptr(dst.iterator,dtype=cutlass.BFloat16)
            packed=cute.make_tensor((raw+off).align(16),cute.make_layout(8))
            packed.store(vecs[j,None].load())
    tu,tg=mmaU.get_slice(lane),mmaG.get_slice(lane)
    cu=tu.partition_C(cute.make_identity_tensor((C,D)))
    cg=tg.partition_C(cute.make_identity_tensor((C,D)))
    NU=C*D//WG
    ag=mmaG.make_fragment_C(mmaG.partition_shape_C((C,D)))
    au=mmaU.make_fragment_C(mmaU.partition_shape_C((C,D)))
    def gemm(lane,mma,acc,aa,bb,stage,accumulate=False):
        th=mma.get_slice(lane)
        fa=mma.make_fragment_A(th.partition_A(aa[None,None,0]))
        fb=mma.make_fragment_B(th.partition_B(bb[None,None,stage]))
        warpgroup.fence()
        for j in cutlass.range(cute.size(fa,mode=[2]),unroll_full=True):
            mma.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(accumulate or j!=0))
            cute.gemm(mma,acc,fa[None,None,j],fb[None,None,j],acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    def sync(group):
        cute.arch.barrier(barrier_id=group+1,number_of_threads=WG)
    def accumulate_dot(au,ag,mat,weight,dest,sign,cg,lane,group):
        # WGMMA's four-lane quads own two rows. Sum each row in registers,
        # then exchange the four column partitions within its quad.
        dot0,dot1=cutlass.Float32(0.),cutlass.Float32(0.)
        for i in cutlass.range_constexpr(C*D//WG):
            r,c=cg[i][0],cg[i][1]
            term=sign*au[i]*cutlass.Float32(mat[r,c,0])
            if cutlass.const_expr((i//2)%2==0):
                dot0=dot0+term
            else:
                dot1=dot1+term
            ag[i]=ag[i]+sign*weight[r]*au[i]
        dot0=dot0+cute.arch.shuffle_sync_bfly(dot0,1)
        dot0=dot0+cute.arch.shuffle_sync_bfly(dot0,2)
        dot1=dot1+cute.arch.shuffle_sync_bfly(dot1,1)
        dot1=dot1+cute.arch.shuffle_sync_bfly(dot1,2)
        if lane%4==0:
            dest[cg[0][0]]=dot0
            dest[cg[2][0]]=dot1
        sync(group)
    pcount=cutlass.Int32(0)
    prefetched=cutlass.Boolean(False)
    for t in cutlass.range((NITEMS-bid+NBLK-1)//NBLK):
        item=bid+t*NBLK
        bh,n,b=item%H,(item//H)%NC,item//(H*NC)
        active=cutlass.Boolean(True)
        if cutlass.const_expr(PACKED):
            active=n<offsets[cute.size(offsets)-1]
        if active:
            rows = valid_rows(n,mapping,NATIVE)
            qi = token_chunk(q,n,mapping,NATIVE)
            ki = token_chunk(k,n,mapping,NATIVE)
            doi = token_chunk(do,n,mapping,NATIVE)
            k2c = token_chunk(k2,n,mapping,NATIVE)
            q2i = token_chunk(q2,n,mapping,NATIVE)
            cute.arch.barrier()          # the previous chunk-head's readers are done with the operand tiles
            if not prefetched:
                issue_slot(tid,tmaH,tHg,tHs,tmaDS,tDSg,tDSs,ready,pcount%2,0,b,n,bh)
            load128(qi,sQ,rows,b,n,bh,tid)
            load128(ki,sK,rows,b,n,bh,tid)
            load128(doi,sDO,rows,b,n,bh,tid)
            load128(vd,sVD,C,b,n,bh,tid)
            for j in cutlass.range_constexpr(C*D//(2*WG*8)):
                pos=j*2*WG+tid
                r,c=pos//(D//8),(pos%(D//8))*8
                src=cute.make_tensor((dwf.iterator+dwf.layout((b,n,r,bh,c))).align(16),cute.make_layout(8))
                values=src.load().to(cutlass.BFloat16)
                off=(c//64)*4096+r*64+8*((c//8)%8 ^ (r%8))+c%8
                raw=cute.recast_ptr(sDW.iterator,dtype=cutlass.BFloat16)
                dst=cute.make_tensor((raw+off).align(16),cute.make_layout(8))
                dst.store(values)
            for j in cutlass.range_constexpr((E*C+2*WG-1)//(2*WG)):
                pos=j*2*WG+tid
                if pos<E*C:
                    e,r=pos//C,pos%C
                    sg[e,r]=gc[b,n,r,bh,e]
                    sk2[e,r]=masked_scalar(k2c,(b,n,r,bh,e),r,rows)
                    sq[e,r]=masked_scalar(q2i,(b,n,r,bh,e),r,rows)
            cute.arch.fence_proxy('async.shared',space='cta')
            cute.arch.barrier()
            ag.fill(0.)
            for e in cutlass.range(E,unroll=1):
                stage=(pcount+e)%2
                cute.arch.mbarrier_wait(ready.iterator+stage,((pcount+e)//2)%2)
                if e+1<E:
                    issue_slot(tid,tmaH,tHg,tHs,tmaDS,tDSg,tDSs,ready,1-stage,e+1,b,n,bh)
                else:
                    # last slot: the next chunk-head's first slot, if there is one and it is active
                    prefetched=cutlass.Boolean(False)
                    if item+NBLK<NITEMS:
                        bhx,nx,bx=(item+NBLK)%H,((item+NBLK)//H)%NC,(item+NBLK)//(H*NC)
                        activex=cutlass.Boolean(True)
                        if cutlass.const_expr(PACKED):
                            activex=nx<offsets[cute.size(offsets)-1]
                        if activex:
                            issue_slot(tid,tmaH,tHg,tHs,tmaDS,tDSg,tDSs,ready,1-stage,0,bx,nx,bhx)
                            prefetched=cutlass.Boolean(True)
                dot=cutlass.Float32(0.)
                for j in cutlass.range_constexpr(D*D//(2*WG*8)):
                    pos=j*2*WG+tid
                    row,col=pos//(D//8),(pos%(D//8))*8
                    byteoff=sH.layout((col,row,stage))*2
                    shift=cutlass.const_expr(lH.inner.num_shift)
                    mask=cutlass.const_expr(((1<<lH.inner.num_bits)-1)<<(lH.inner.num_base+shift))
                    off=(byteoff ^ ((byteoff & mask)>>shift))//2
                    vh=cute.make_tensor((cute.recast_ptr(sH.iterator,dtype=cutlass.BFloat16)+off).align(16),cute.make_layout(8)).load()
                    vs=cute.make_tensor((cute.recast_ptr(sDS.iterator,dtype=cutlass.BFloat16)+off).align(16),cute.make_layout(8)).load()
                    for z in cutlass.range_constexpr(8):
                        dot=dot+cutlass.Float32(vh[z])*cutlass.Float32(vs[z])
                dot=cute.arch.warp_reduction_sum(dot)
                if tid%32==0:
                    sdot[tid//32]=dot
                if tid<C:
                    eg=cute.math.exp(sg[e,tid],fastmath=True)
                    sa[tid]=sk2[e,tid]*eg
                    sr[tid]=sk2[e,tid]*cute.math.exp(sg[e,C-1]-sg[e,tid],fastmath=True)
                    sc[tid]=SCALE*sq[e,tid]*eg
                cute.arch.fence_proxy('async.shared',space='cta')
                cute.arch.barrier()
                # Trace identities form both the matrix gradient and its per-slice dots (second key, decay)
                # from the same product. Row weights are applied in FP32 after WGMMA.
                if group==0:
                    gemm(lane,mmaG,au,sVD,sDSg,stage)
                    accumulate_dot(au,ag,sK,sr,sd2,1.,cg,lane,group)
                    gemm(lane,mmaG,au,sDW,sHg,stage)
                    accumulate_dot(au,ag,sK,sa,sd1,-1.,cg,lane,group)
                    if lane<C:
                        ex=cute.math.exp(sg[e,lane],fastmath=True)
                        er=cute.math.exp(sg[e,C-1]-sg[e,lane],fastmath=True)
                        d1,d2=sd1[lane],sd2[lane]
                        vg=sa[lane]*d1-sr[lane]*d2
                        if lane==C-1:
                            tot=cutlass.Float32(0.)
                            for r in cutlass.range(C):
                                tot=tot+sr[r]*sd2[r]
                            vg=vg+tot
                        dgv[b,n,lane,bh,0,e]=vg
                        dk2v[b,n,lane,bh,0,e]=ex*d1+er*d2
                else:
                    gemm(lane,mmaG,au,sDO,sHg,stage)
                    accumulate_dot(au,ag,sQ,sc,sd3,1.,cg,lane,group)
                    if lane<C:
                        ex=cute.math.exp(sg[e,lane],fastmath=True)
                        d3=SCALE*sd3[lane]
                        vgo=sq[e,lane]*ex*d3
                        if lane==C-1:
                            dotall=cutlass.Float32(0.)
                            for z in cutlass.range_constexpr(8):
                                dotall=dotall+sdot[z]
                            vgo=vgo+cute.math.exp(sg[e,C-1],fastmath=True)*dotall
                        dgo[b,n,lane,bh,0,e]=vgo
                        dq2o[b,n,lane,bh,0,e]=ex*d3
                cute.arch.barrier()
            pairp=cute.make_rmem_tensor((2,),cutlass.Float32)
            for jp in cutlass.range_constexpr(NU//2):
                i=2*jp
                r,c=cg[i][0],cg[i][1]
                pairp[0],pairp[1]=ag[i],ag[i+1]
                if group==0:
                    cute.make_tensor((dkp.iterator+dkp.layout((b,n,r,bh,0,c))).align(8),cute.make_layout((2,))).store(pairp.load())
                else:
                    cute.make_tensor((dqp.iterator+dqp.layout((b,n,r,bh,0,c))).align(8),cute.make_layout((2,))).store(pairp.load())


            pcount=pcount+E
        else:
            prefetched=cutlass.Boolean(False)
@cute.jit
def launch_recurrent(q:cute.Tensor,k:cute.Tensor,do:cute.Tensor,A:cute.Tensor,Mp:cute.Tensor,
                     gc:cute.Tensor,k2:cute.Tensor,q2:cute.Tensor,beta:cute.Tensor,
                     dsout:cute.Tensor,dwf:cute.Tensor,dv:cute.Tensor,dvd:cute.Tensor,gDST:cute.Tensor,
                     gQT:cute.Tensor,gKT:cute.Tensor,gDOT:cute.Tensor,gAT:cute.Tensor,gMT:cute.Tensor,gOffsets:cute.Tensor,mapping:cute.Tensor,
                     B:cutlass.Constexpr,H:cutlass.Constexpr,E:cutlass.Constexpr,NC:cutlass.Constexpr,
                     VC:cutlass.Constexpr,NW:cutlass.Constexpr,SCALE:cutlass.Constexpr,DOCS:cutlass.Constexpr,NBLK:cutlass.Int32,NATIVE:cutlass.Constexpr,stream:cuda.CUstream):
    bf,ff=cutlass.BFloat16,cutlass.Float32
    def mma(n,am,bm):
        return sm90.make_trivial_tiled_mma(bf,bf,am,bm,ff,(1,1,1),(C,n))
    mmaP,mmaU=mma(C,OMM.K,OMM.K),mma(VC,OMM.K,OMM.MN)
    mmaT,mmaS=mma(VC,OMM.MN,OMM.MN),mma(VC,OMM.MN,OMM.MN)
    lK=sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR,(C,C,D),bf,1)
    lKt=sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR,(D,VC,C),bf,1)
    lA=sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR,(C,C,C),bf,1)
    lAt=sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR,(C,C,C),bf,1)
    lH=sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR,(C,VC,D),bf,E)
    lV=sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR,(C,VC,C),bf,1)
    lVs=sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR,(C,VC,C),bf,E)
    tmaDS,tDSt=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),gDST,
        cute.slice_(lH,(None,None,0)),(VC,D),1)
    op=cpasync.CopyBulkTensorTileG2SOp()
    tmaQ,tQt=cpasync.make_tiled_tma_atom(op,gQT,cute.slice_(lK,(None,None,0)),(C,D),1)
    tmaK,tKt=cpasync.make_tiled_tma_atom(op,gKT,cute.slice_(lK,(None,None,0)),(C,D),1)
    tmaDO,tDOt=cpasync.make_tiled_tma_atom(op,gDOT,cute.slice_(lV,(None,None,0)),(VC,C),1)
    tmaA,tAt=cpasync.make_tiled_tma_atom(op,gAT,cute.slice_(lA,(None,None,0)),(C,C),1)
    tmaM,tMt=cpasync.make_tiled_tma_atom(op,gMT,cute.slice_(lA,(None,None,0)),(C,C),1)
    recurrent_kernel(q,k,do,A,Mp,gc,k2,q2,beta,dsout,dwf,dv,dvd,tmaDS,tDSt,
        tmaQ,tQt,tmaK,tKt,tmaDO,tDOt,tmaA,tAt,tmaM,tMt,gOffsets,mapping,
        mmaP,mmaU,mmaT,mmaS,lK,lKt,lA,lAt,lH,lV,lVs,H,E,NC,VC,NW,SCALE,DOCS,NATIVE).launch(
            grid=(NBLK*H*(D//VC),1,1),block=(NW*WG,1,1),stream=stream)


@cute.jit
def launch_state_grads(q:cute.Tensor,k:cute.Tensor,do:cute.Tensor,vd:cute.Tensor,dwf:cute.Tensor,
                       gc:cute.Tensor,k2:cute.Tensor,q2:cute.Tensor,h:cute.Tensor,ds:cute.Tensor,
                       dqp:cute.Tensor,dkp:cute.Tensor,dgv:cute.Tensor,dk2v:cute.Tensor,dgo:cute.Tensor,dq2o:cute.Tensor,gHT:cute.Tensor,gDST:cute.Tensor,offsets:cute.Tensor,mapping:cute.Tensor,
                       B:cutlass.Constexpr,H:cutlass.Constexpr,E:cutlass.Constexpr,NC:cutlass.Constexpr,
                       SCALE:cutlass.Constexpr,PACKED:cutlass.Constexpr,NATIVE:cutlass.Constexpr,stream:cuda.CUstream):
    bf,ff=cutlass.BFloat16,cutlass.Float32
    mmaU=sm90.make_trivial_tiled_mma(bf,bf,OMM.K,OMM.MN,ff,(1,1,1),(C,D))
    mmaG=sm90.make_trivial_tiled_mma(bf,bf,OMM.K,OMM.K,ff,(1,1,1),(C,D))
    lK=sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR,(C,D,D),bf,1)
    lH=sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR,(C,D,D),bf,2)
    lHg=sm90.make_smem_layout_b(LayoutEnum.ROW_MAJOR,(C,D,D),bf,2)
    tmaH,tHt=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),gHT,
        cute.slice_(lH,(None,None,0)),(D,D),1)
    tmaDS,tDSt=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),gDST,
        cute.slice_(lH,(None,None,0)),(D,D),1)
    state_grads_kernel(q,k,do,vd,dwf,gc,k2,q2,h,ds,dqp,dkp,dgv,dk2v,dgo,dq2o,
        tmaH,tHt,tmaDS,tDSt,offsets,mapping,mmaU,mmaG,lK,lH,lHg,H,E,NC,SCALE,PACKED,NATIVE,min(B*NC*H,132),B*NC*H).launch(grid=(min(B*NC*H,132),1,1),block=(2*WG,1,1),stream=stream)


_compiled={}


def joint_bwd_recurrent(q,k,do,A,Mp,gc,k2,q2,beta,h,Vd,scale=None,groups=2,value_cols=None,chunk_starts=None,token_map=None):
    B,T,H,D_=q.shape
    native=token_map is not None
    E,NC=k2.shape[-1],gc.shape[1]//C
    PT=NC*C
    assert D_==D and (native or T%C==0) and E in (1,2,4,8,12,16) and groups in (1,2,4) and E%groups==0
    VC=(16 if E>=12 else 32) if value_cols is None else value_cols
    assert VC in (8,16,32)
    scale=D**-.5 if scale is None else float(scale)
    device=q.device
    def empty(shape,dtype=torch.float32):
        return torch.empty(shape,device=device,dtype=dtype)
    ds=empty((B,NC,H,E,D,D),q.dtype)
    dwf=empty((B,PT,H,D))
    dv,dvd=torch.empty_like(do),empty((B,PT,H,D),do.dtype)
    docs=chunk_starts.numel()-1 if chunk_starts is not None else 0
    assert not docs or B==1
    dqp,dkp=(empty((B,PT,H,1,D)) for _ in range(2))
    dgv,dk2v,dgo,dq2o=(empty((B,PT,H,1,E)) for _ in range(4))
    def chunks(t):
        return t.view(B,NC,C,*t.shape[2:])
    def raw(t):
        return t.view(B,1,T,*t.shape[2:]) if native else chunks(t)
    recurrent=tuple(raw(t) if i in (0,1,2,6,7,8) else chunks(t)
                    for i,t in enumerate((q,k,do,A,Mp,gc,k2,q2,beta)))+(ds,chunks(dwf),raw(dv),chunks(dvd),ds.view(B,NC,H,E*D,D).permute(4,3,0,1,2))
    recurrent+=tuple(raw(t).permute(2,4,0,1,3) for t in (q,k))
    recurrent+=(raw(do).permute(4,2,0,1,3),)
    recurrent+=tuple(chunks(t).permute(2,4,0,1,3) for t in (A,Mp))
    parallel=tuple(raw(t) if i in (0,1,2,6,7) else chunks(t) for i,t in enumerate((q,k,do,Vd,dwf,gc,k2,q2)))+(h.view_as(ds),ds)+tuple(chunks(t) for t in (dqp,dkp,dgv,dk2v,dgo,dq2o))
    parallel+=tuple(t.view(B,NC,H,E*D,D).permute(4,3,0,1,2) for t in (h,ds))
    ra=[from_dlpack(t.detach(),assumed_align=16) for t in recurrent]
    if native and T == 1 and H == 1:
        for index, axis in ((14, 0), (15, 0), (16, 1)):
            keep_singleton_tma_axis(ra[index], axis)
    pa=[from_dlpack(t.detach(),assumed_align=16) for t in parallel]
    ra.append(ra[0] if chunk_starts is None else from_dlpack(chunk_starts.detach(),assumed_align=16).mark_layout_dynamic())
    pa.append(ra[-1] if docs else pa[0])
    mapping=from_dlpack(token_map.detach(),assumed_align=16) if native else ra[0]
    ra.append(mapping)
    pa.append(mapping)
    stream=cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key=(B,T,H,E,scale,groups,VC,bool(docs),native,NC)
    if key not in _compiled:
        recurrent=cute.compile(launch_recurrent,*ra,B,H,E,NC,VC,groups,scale,bool(docs),docs if docs else B,int(native),stream)
        par=cute.compile(launch_state_grads,*pa,B,H,E,NC,scale,bool(docs),native,stream)
        _compiled[key]=recurrent,par
    recurrent,par=_compiled[key]
    recurrent(*ra,docs if docs else B,stream)      # NBLK is a runtime argument
    par(*pa,stream)
    return dqp,dkp,dv,dvd,dgv,dk2v,dgo,dq2o
