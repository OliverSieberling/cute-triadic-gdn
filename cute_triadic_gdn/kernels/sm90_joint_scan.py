"""Chunk-local cumulative log decay (log gamma of the paper) with two 32-token parallel prefix sums per chunk."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from .asm import _f32, _PURE, _shfl_idx
from .sm90_joint_packed import token_chunk, valid_rows


def shift(x, distance, reverse):
    opcode = 'down' if reverse else 'up'
    clamp = 31 if reverse else 0
    return cutlass.Float32(llvm.inline_asm(
        _f32(), [x.ir_value(), cutlass.Int32(distance).ir_value()],
        f'shfl.sync.{opcode}.b32 $0, $1, $2, {clamp}, 0xffffffff;', '=f,f,r', **_PURE))


@cute.kernel
def scan_kernel(g: cute.Tensor, out: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor,
                B: cutlass.Constexpr, NC: cutlass.Constexpr, HE: cutlass.Constexpr,
                REVERSE: cutlass.Constexpr, PACKED: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    blocks=(HE+15)//16
    group,n,b=bid%blocks,(bid//blocks)%NC,bid//(blocks*NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets)-1]
    if active:
        rows = valid_rows(n, mapping, NATIVE)
        if cutlass.const_expr(NATIVE and not REVERSE):
            gin = token_chunk(g, n, mapping, True)
        else:
            gin = g
        if cutlass.const_expr(NATIVE and REVERSE):
            gout = token_chunk(out, n, mapping, True)
        else:
            gout = out
        lane,warp=tid%32,tid//32
        smem=cutlass.utils.SmemAllocator()
        tile=smem.allocate_tensor(cutlass.Float32,cute.make_layout((64,16),stride=(17,1)),byte_alignment=16)
        for j in cutlass.range_constexpr(8):
            pos=j*128+tid
            r,e=pos//16,pos%16
            value=cutlass.Float32(0.)
            if group*16+e<HE:
                if cutlass.const_expr(REVERSE):
                    value=gin[b,n,r,group*16+e]
                elif r < rows:
                    value=gin[b,n,r,group*16+e]
            tile[r,e]=value
        cute.arch.barrier()
        for j in cutlass.range_constexpr(4):
            e=j*4+warp
            carry=cutlass.Float32(0.)
            for seg in cutlass.range_constexpr(2):
                r=(1-seg)*32+lane if cutlass.const_expr(REVERSE) else seg*32+lane
                value=tile[r,e]
                for i in cutlass.range_constexpr(5):
                    distance=1<<i
                    other=shift(value,distance,REVERSE)
                    if cutlass.const_expr(REVERSE):
                        if lane<32-distance:
                            value=value+other
                    else:
                        if lane>=distance:
                            value=value+other
                value=value+carry
                tile[r,e]=value
                carry=_shfl_idx(value,0 if cutlass.const_expr(REVERSE) else 31)
        cute.arch.barrier()
        for j in cutlass.range_constexpr(8):
            pos=j*128+tid
            r,e=pos//16,pos%16
            if group*16+e<HE:
                if cutlass.const_expr(not REVERSE) or r < rows:
                    gout[b,n,r,group*16+e]=tile[r,e]


@cute.jit
def launch(g: cute.Tensor,out: cute.Tensor,offsets: cute.Tensor,mapping: cute.Tensor,B: cutlass.Constexpr,NC: cutlass.Constexpr,
           HE: cutlass.Constexpr,REVERSE: cutlass.Constexpr,PACKED: cutlass.Constexpr,NATIVE: cutlass.Constexpr,stream: cuda.CUstream):
    scan_kernel(g,out,offsets,mapping,B,NC,HE,REVERSE,PACKED,NATIVE).launch(grid=(B*NC*((HE+15)//16),1,1),block=(128,1,1),stream=stream)


_compiled={}


def joint_scan(g,reverse=False,chunk_offsets=None,token_map=None,output_tokens=None):
    B,T,H,E=g.shape
    native=token_map is not None
    NC=token_map.shape[1] if native else T//64
    assert (native or T%64==0) and g.dtype==torch.float32 and g.is_contiguous()
    out_T=(output_tokens if reverse else NC*64) if native else T
    out=torch.empty((B,out_T,H,E),device=g.device,dtype=g.dtype)
    shapes=((B,NC,64,H*E),(B,1,out_T,H*E)) if native and reverse else (
        ((B,1,T,H*E),(B,NC,64,H*E)) if native else ((B,NC,64,H*E),)*2)
    args=[from_dlpack(t.detach().view(shape),assumed_align=16) for t,shape in zip((g,out),shapes)]
    packed=chunk_offsets is not None
    args.append(from_dlpack(chunk_offsets.detach(),assumed_align=16).mark_layout_dynamic() if packed else args[0])
    args.append(from_dlpack(token_map.detach(),assumed_align=16) if native else args[0])
    stream=cuda.CUstream(torch.cuda.current_stream(g.device).cuda_stream)
    key=(B,T,out_T,H*E,bool(reverse),packed,native,NC)
    if key not in _compiled:
        _compiled[key]=cute.compile(launch,*args,B,NC,H*E,bool(reverse),packed,native,stream)
    _compiled[key](*args,stream)
    return out
