"""Packed documents: metadata for the Hopper kernels, and the copying path of the torch reference.

The kernels read the original token storage through metadata only (no copies); the allocation depends
on T and the number of documents, never on the boundary values, so no host read is needed.  The torch
reference pads every document to whole chunks and resets the state at each document start.
"""
import torch
from torch.library import custom_op

CHUNK = 64
# Chunk-map capacity is rounded up to a multiple of this, so the kernels' compile keys (which include
# the capacity) take only a few values as the document count changes from batch to batch.
NC_BUCKET = 16


def chunk_capacity(tokens, documents):
    """Chunk-map capacity for `documents` packed documents over `tokens` tokens:
    sum(ceil(len_i / 64)) <= ceil(tokens / 64) + documents - 1, rounded up to a multiple of NC_BUCKET."""
    exact = (tokens + CHUNK - 1) // CHUNK + documents - 1
    return -(-exact // NC_BUCKET) * NC_BUCKET


@custom_op("cute_triadic_gdn::joint_metadata3", mutates_args=(), device_types="cuda")
def _metadata(cu_seqlens: torch.Tensor, tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ..kernels.sm90_joint_packed import joint_metadata
    return joint_metadata(cu_seqlens, tokens)


@_metadata.register_fake
def _metadata_fake(cu_seqlens, tokens):
    documents = cu_seqlens.numel() - 1
    capacity = chunk_capacity(tokens, documents)
    return (cu_seqlens.new_empty(documents + 1, dtype=torch.int32),
            cu_seqlens.new_empty((3, capacity), dtype=torch.int32),
            cu_seqlens.new_empty(1, dtype=torch.bool))


def _validate(inputs, cu_seqlens):
    q = inputs[0]
    if q.shape[0] != 1 or q.shape[1] <= 0:
        raise ValueError("cu_seqlens requires a nonempty packed row: B=1, T>0")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a vector with at least two offsets")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must have dtype int32 or int64")
    if cu_seqlens.device != q.device or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous and on the input device")
    if q.shape[1] > 2**31 - 1:
        raise ValueError("packed token offsets must fit int32")


def prepare_documents(inputs, cu_seqlens):
    """Validate the offsets and build the kernels' metadata on the device without copying tokens."""
    _validate(inputs, cu_seqlens)
    q = inputs[0]
    message = "cu_seqlens must start at 0, end at T, and strictly increase"
    if cu_seqlens.numel() == 2 and q.shape[1] % CHUNK == 0:
        torch._assert_async((cu_seqlens[0] == 0) & (cu_seqlens[1] == q.shape[1]), message)
        return (cu_seqlens // CHUNK).to(torch.int32), None
    offsets, mapping, valid = _metadata(cu_seqlens, q.shape[1])
    torch._assert_async(valid, message)
    return offsets, mapping


def pack_documents(inputs, cu_seqlens):
    """The torch reference's packing: every document padded to whole chunks, bool (1, NC) reset flags,
    and the indices that restore the original token order.  No host read of the boundary values."""
    _validate(inputs, cu_seqlens)
    q = inputs[0]
    T, N = q.shape[1], cu_seqlens.numel() - 1
    offsets = cu_seqlens.to(torch.int64)
    lengths = offsets[1:] - offsets[:-1]
    torch._assert_async((offsets[0] == 0) & (offsets[-1] == T) & (lengths > 0).all(),
                        "cu_seqlens must start at 0, end at T, and strictly increase")
    padded_lengths = (lengths + CHUNK - 1) // CHUNK * CHUNK
    ends = padded_lengths.cumsum(0)
    starts = ends - padded_lengths
    capacity = ((T + CHUNK - 1) // CHUNK + N - 1) * CHUNK   # sum(ceil(len_i / C)) <= ceil(T / C) + N - 1
    pos = torch.arange(capacity, device=q.device)
    doc = torch.searchsorted(ends, pos, right=True).clamp_max(N - 1)
    local = pos - starts[doc]
    valid = (local < lengths[doc]) & (pos < ends[-1])
    source = (offsets[doc] + local).clamp(0, T - 1)
    chunk_starts = (local[::CHUNK] == 0).view(1, -1).contiguous()
    tokens = torch.arange(T, device=q.device)
    token_doc = torch.searchsorted(offsets[1:], tokens, right=True)
    restore = starts[token_doc] + tokens - offsets[token_doc]
    packed = [torch.where(valid.view(1, capacity, *([1] * (t.ndim - 2))),
                          t.index_select(1, source), 0).contiguous() for t in inputs]
    return packed, chunk_starts, restore
