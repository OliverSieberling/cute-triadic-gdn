"""Native packed-document metadata and masked chunk views (no token copies)."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..ops.joint_packing import chunk_capacity


@cute.jit
def token_chunk(t: cute.Tensor, n, mapping: cute.Tensor, native: cutlass.Constexpr):
    if cutlass.const_expr(native):
        # Raw tensors have shape (1, 1, T, ...); padded intermediates keep
        # (1, NC, 64, ...). Rebase only the raw tensors for this logical chunk.
        # Ignore the logical chunk coordinate in raw storage. Subtracting
        # n*T*stride here and adding it back during indexing can overflow
        # 32-bit intermediates even when the real tensor is much smaller.
        offset = cutlass.Int64(mapping[0, n]) * t.stride[2]
        layout = cute.make_layout(t.shape, stride=(t.stride[0], 0, *t.stride[2:]))
        return cute.make_tensor(t.iterator + offset, layout)
    else:
        return t


@cute.jit
def valid_rows(n, mapping: cute.Tensor, native: cutlass.Constexpr):
    # native 0: fixed/padded tensors (whole chunks); 1: packed documents, whose last chunk may be partial
    rows = cutlass.Int32(64)
    if cutlass.const_expr(native == 1):
        rows = mapping[1, n]
    return rows


@cute.jit
def masked_scalar(t: cute.Tensor, coord: tuple, row, rows):
    value = cutlass.Float32(0.)
    if row < rows:
        value = t[coord]
    return value


@cute.jit
def zero_tail(dst: cute.Tensor, rows, tid, threads: cutlass.Constexpr,
              width: cutlass.Constexpr, stage, transpose: cutlass.Constexpr):
    for pos in cutlass.range(tid, 64*width, threads):
        r, c = pos // width, pos % width
        if r >= rows:
            if cutlass.const_expr(transpose):
                dst[c,r,stage] = cutlass.BFloat16(0.)
            else:
                dst[r,c,stage] = cutlass.BFloat16(0.)


@cute.jit
def store_pair(dst: cute.Tensor, b, n, r, head, col, values,
               rows):
    if r < rows:
        pair = cute.make_tensor((dst.iterator + dst.layout((b,n,r,head,col))).align(4),
                                cute.make_layout(2))
        pair.store(values)


@cute.kernel
def metadata_kernel(cu: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor, valid: cute.Tensor,
                    T: cutlass.Constexpr, N: cutlass.Int32, NC: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    doc, _, _ = cute.arch.block_idx()
    start, end = cutlass.Int32(cu[doc]), cutlass.Int32(cu[doc + 1])
    safe = (cu[doc] >= 0) & (cu[doc+1] <= T) & (cu[doc+1] > cu[doc])
    first = cutlass.Int32(0)
    for i in cutlass.range(doc):
        safe = safe & (cu[i] >= 0) & (cu[i+1] <= T) & (cu[i+1] > cu[i])
        first += (cutlass.Int32(cu[i + 1]) - cutlass.Int32(cu[i]) + 63) // 64
    count = (end - start + 63) // 64
    if tid == 0:
        offsets[doc] = first
        if doc == N - 1:
            offsets[N] = first + count
        # Row 2 of the mapping lists the documents longest first: the recurrences give one CTA per
        # (document, head, value block) and run at most one CTA per SM, so launching the long chains
        # first shortens the makespan (list scheduling of unequal chains).  Ties keep document order.
        rank = cutlass.Int32(0)
        for i in cutlass.range(N):
            other = (cutlass.Int32(cu[i + 1]) - cutlass.Int32(cu[i]) + 63) // 64
            if (other > count) | ((other == count) & (i < doc)):
                rank += 1
        mapping[2, rank] = doc
    # Invalid offsets must never cause an OOB access before the following
    # stream-ordered torch device assertion sees the validation result.
    if safe & (first >= 0) & (first + count <= NC):
        for step in cutlass.range(tid, count, 128):
            mapping[0, first + step] = start + step * 64
            mapping[1, first + step] = cutlass.min(64, end - start - step * 64)
    if (doc == 0) & (tid == 0):
        correct = (cu[0] == 0) & (cu[N] == T)
        for i in cutlass.range(N):
            correct = correct & (cu[i] >= 0) & (cu[i+1] <= T) & (cu[i+1] > cu[i])
        valid[0] = correct


@cute.jit
def launch_metadata(cu: cute.Tensor, offsets: cute.Tensor, mapping: cute.Tensor, valid: cute.Tensor,
                    T: cutlass.Constexpr, N: cutlass.Int32, NC: cutlass.Int32,
                    stream: cuda.CUstream):
    metadata_kernel(cu, offsets, mapping, valid, T, N, NC).launch(
        grid=(N, 1, 1), block=(128, 1, 1), stream=stream)


_compiled = {}


def joint_metadata(cu, tokens):
    documents = cu.numel() - 1
    capacity = chunk_capacity(tokens, documents)
    offsets = torch.empty(documents + 1, dtype=torch.int32, device=cu.device)
    mapping = torch.empty((3, capacity), dtype=torch.int32, device=cu.device)   # rows: chunk start, valid rows, document order
    valid = torch.empty(1, dtype=torch.bool, device=cu.device)
    # A from_dlpack tensor's shape is part of the compiled kernel unless it is marked dynamic: the
    # compiled code would silently keep the first call's document count.  Everything whose size
    # follows the document count is marked here, so one compile serves every count.
    args = [from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic() for t in (cu, offsets, mapping)]
    args.append(from_dlpack(valid.detach(), assumed_align=16))
    stream = cuda.CUstream(torch.cuda.current_stream(cu.device).cuda_stream)
    key = (tokens, cu.dtype)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_metadata, *args, tokens, documents, capacity, stream)
    _compiled[key](*args, documents, capacity, stream)
    return offsets, mapping, valid


def keep_singleton_tma_axis(t, token_axis):
    """Keep a singleton token axis in a 2D TMA chunk descriptor.

    With T=H=1, CuTe otherwise collapses the raw input to a 1D tensor. The
    generated 1D loads are not valid for our swizzled 2D chunk copy. A runtime
    extent of one preserves the axis and hardware OOB zero-fill, without
    padding or copying storage. This changes descriptor metadata only.
    """
    order = (2, 3, 0, 4, 1) if token_axis == 0 else (2, 3, 1, 4, 0)
    return t.mark_compact_shape_dynamic(token_axis, stride_order=order)
