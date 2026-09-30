"""Causal depthwise short convolution whose leading channels get SiLU and the rest no activation.

The triadic layer convolves q, k, v (SiLU) and its second key and query (no activation, softplus afterwards)
in one call on one projection's output: channels [0, ACT_CH) get SiLU, the others none.  With cu_seqlens the
input is one packed row of documents and the convolution restarts at every document.

Kernels adapted from flash-linear-attention's causal_conv1d training kernels (fla/modules/conv/triton/kernels.py,
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license); per channel the arithmetic is theirs.
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import custom_op, register_autograd

_CONFIGS = [triton.Config({'BD': BD}, num_warps=num_warps) for BD in [16, 32, 64, 128] for num_warps in [4, 8, 16, 32]]


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=_CONFIGS, key=['D', 'W', 'NB', 'ACT_CH'])
@triton.jit
def _fwd_kernel(x, y, weight, cu_seqlens, chunk_indices, B, T, stride_x_n, stride_x_t, stride_x_d,
                D: tl.constexpr, W: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr,
                NB: tl.constexpr, ACT_CH: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        bos = (i_b * T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
        b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
        b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
        b_y += b_yi
    if ACT_CH > 0:
        b_y = tl.where((o_d < ACT_CH)[None, :], b_y * tl.sigmoid(b_y), b_y)

    p_y = y + bos * D + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=_CONFIGS, key=['D', 'W', 'NB', 'ACT_CH'])
@triton.jit
def _bwd_kernel(x, y, weight, dy, dx, dw, cu_seqlens, chunk_indices, B, T,
                stride_x_n, stride_x_t, stride_x_d, stride_dx_n, stride_dx_t, stride_dx_d,
                stride_dy_n, stride_dy_t, stride_dy_d,
                D: tl.constexpr, W: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr,
                NB: tl.constexpr, ACT_CH: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
        p_dy = dy + bos * stride_dy_t
        p_dx = dx + bos * stride_dx_t
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        bos = (i_b * T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n
        p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_x = (o_t < T)[:, None] & m_d[None, :]

    b_x = tl.load(p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d, mask=m_x, other=0.0)
    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = (o_dy < T)[:, None] & m_d[None, :]
        b_dy = tl.load(p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d, mask=m_dy, other=0.0).to(tl.float32)
        if ACT_CH > 0:
            m_act = (o_d < ACT_CH)[None, :]
            b_y = tl.load(y + bos * D + o_dy[:, None] * D + o_d[None, :], mask=m_dy & m_act, other=0.0).to(tl.float32)
            b_ys = tl.sigmoid(b_y)
            b_dy = tl.where(m_act, b_dy * b_ys * (1 + b_y * (1 - b_ys)), b_dy)
        b_wdy = b_dy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
        b_dw = tl.sum(b_dy * b_x, 0)
        tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
        b_dx += b_wdy

    p_dx = p_dx + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_x)


_BT = 64


def _chunk_indices(cu_seqlens, T):
    """(NT, 2) int32 rows [document, chunk within the document] over every chunk of every packed document, built on the
    device without reading the offsets: NT = ceil(T / BT) + documents - 1 bounds the chunk count, and the surplus rows
    point past the end of the last document, so their programs load and store nothing."""
    N = cu_seqlens.numel() - 1
    counts = (cu_seqlens[1:] - cu_seqlens[:-1] + _BT - 1) // _BT
    ends = counts.cumsum(0)
    pos = torch.arange(triton.cdiv(T, _BT) + N - 1, device=cu_seqlens.device, dtype=ends.dtype)
    doc = torch.searchsorted(ends, pos, right=True).clamp_max(N - 1)
    return torch.stack([doc, pos - (ends - counts)[doc]], 1).to(torch.int32)


def _layout(x, weight, cu_seqlens):
    B, T, D = x.shape
    W = weight.shape[1]
    chunk_indices = _chunk_indices(cu_seqlens, T) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, _BT)
    return B, T, D, W, triton.next_power_of_2(W), triton.cdiv(B * T, 1024), NT, chunk_indices


def _forward(x, weight, act_ch, cu_seqlens):
    B, T, D, W, BW, NB, NT, chunk_indices = _layout(x, weight, cu_seqlens)
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    _fwd_kernel[lambda meta: (triton.cdiv(D, meta['BD']), NT, B)](
        x=x, y=y, weight=weight, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B=B, T=T,
        stride_x_n=x.stride(0), stride_x_t=x.stride(1), stride_x_d=x.stride(2),
        D=D, W=W, BT=_BT, BW=BW, NB=NB, ACT_CH=act_ch)
    return y


@custom_op("cute_triadic_gdn::conv_split_act_fwd", mutates_args=(), device_types="cuda")
def conv_split_act_fwd(x: torch.Tensor, weight: torch.Tensor, act_channels: int,
                       cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _forward(x, weight, act_channels, cu_seqlens)


@conv_split_act_fwd.register_fake
def _(x, weight, act_channels, cu_seqlens=None):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@custom_op("cute_triadic_gdn::conv_split_act_bwd", mutates_args=(), device_types="cuda")
def conv_split_act_bwd(x: torch.Tensor, dy: torch.Tensor, weight: torch.Tensor, act_channels: int,
                       cu_seqlens: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D, W, BW, NB, NT, chunk_indices = _layout(x, weight, cu_seqlens)
    # the SiLU derivative needs the pre-activation output: recomputed
    y = _forward(x, weight, 0, cu_seqlens) if act_channels > 0 else None
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dw = weight.new_empty(B * NT, D, W, dtype=torch.float)
    _bwd_kernel[lambda meta: (triton.cdiv(D, meta['BD']), NT, B)](
        x=x, y=y, weight=weight, dy=dy, dx=dx, dw=dw, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B=B, T=T,
        stride_x_n=x.stride(0), stride_x_t=x.stride(1), stride_x_d=x.stride(2),
        stride_dx_n=dx.stride(0), stride_dx_t=dx.stride(1), stride_dx_d=dx.stride(2),
        stride_dy_n=dy.stride(0), stride_dy_t=dy.stride(1), stride_dy_d=dy.stride(2),
        D=D, W=W, BT=_BT, BW=BW, NB=NB, ACT_CH=act_channels)
    return dx, dw.sum(0).to(weight)


@conv_split_act_bwd.register_fake
def _(x, dy, weight, act_channels, cu_seqlens=None):
    return torch.empty_like(x, memory_format=torch.contiguous_format), torch.empty_like(weight)


def _setup(ctx, inputs, output):
    x, weight, act_channels, cu_seqlens = inputs
    ctx.save_for_backward(x, weight, *([cu_seqlens] if cu_seqlens is not None else []))
    ctx.act_channels = act_channels
    ctx.has_cu = cu_seqlens is not None


def _backward(ctx, dy):
    saved = ctx.saved_tensors
    x, weight = saved[0], saved[1]
    cu_seqlens = saved[2] if ctx.has_cu else None
    dx, dw = conv_split_act_bwd(x, dy, weight, ctx.act_channels, cu_seqlens)
    return dx, dw, None, None


register_autograd("cute_triadic_gdn::conv_split_act_fwd", _backward, setup_context=_setup)


def conv_split_act_call(x, weight, act_channels, cu_seqlens=None):
    """x [B, T, D], weight [D, W]; channels [0, act_channels) get SiLU, the rest none.  cu_seqlens: document
    boundaries of a packed row (the conv restarts at each document)."""
    return conv_split_act_fwd(x, weight.contiguous(), int(act_channels), cu_seqlens)
