"""Triadic Gated DeltaNet: the gated delta rule over the joint key kappa_t = k2_t (x) k_t (paper, Section 2.4).

    S_e <- alpha_te S_e                     every slice e (alpha = exp(g), the decay gate)
    S   <- S + beta_t kappa_t (v_t - S^T kappa_t)^T
    o_t  = S^T (q2_t (x) q_t) * scale

S is (E*D, V), slice e is rows e*D:(e+1)*D.  The erase term sees every slice at once, so the chunk quantities
below are shared by all E slices.  Names of the chunkwise form (Appendix A) in this code:
    gc      chunk-local cumulative log decay (log gamma)         A       the UT transform T of Eq. 10 without diag(beta)
    M, Mp   the masks R and R' of Eq. 9                          W       the residual V - sum_e diag(k2_e gamma_e) K S_e
    Vd      U = T W of Eq. 11 (delta-corrected values)           h       the state S at the start of every chunk

Inputs (all contiguous, one CUDA device):
  q, k    (B, T, H, D)  bf16, L2-normalised along D
  v       (B, T, H, V)  bf16
  k2, q2  (B, T, H, E)  f32, the second key and second query, L2-normalised along E
  g       (B, T, H, E)  f32, log of the decay gate of each slice, <= 0
  beta    (B, T, H)     f32, the write strength, one per head
  scale   float, default D ** -0.5
Output o (B, T, H, V) bf16.  D = V = 128, E in {1, 2, 4, 8, 12, 16} on Hopper (E = 1 is Gated DeltaNet:
k2 = q2 = 1).
Without cu_seqlens, T % 64 == 0.  With cu_seqlens, B = 1 and document lengths are arbitrary.

Every exponential the kernels take has argument <= 0 (the chunk-local cumsum of a non-positive log gate is
non-positive), so the decay span needs no cap.
"""
import os as _os

import torch

from .dispatch import sm_arch

CHUNK = 64


def _cumsum(g):
    """chunk-local inclusive cumsum over T, (B,T,H,E) -> (B,T,H,E)"""
    B, T, H, E = g.shape
    return g.view(B, T // CHUNK, CHUNK, H, E).cumsum(2).view(B, T, H, E)


# ----------------------------------------------------------------------------------------------
# torch reference (differentiable through autograd): what the kernels are tested against, and the path off Hopper.
# ----------------------------------------------------------------------------------------------
def masks_torch(k, k2, q2, gc, beta, chunk=CHUNK):
    """A (B,NC,H,64,64), M (same), Mp (same), all f32 here (the kernels emit bf16).
    M = R, Mp = R' (Eq. 9); A = (I + StrictLower(beta_t <k_t,k_s> M_ts))^-1, the UT transform T without diag(beta)."""
    B, T, H, D = k.shape
    E = k2.shape[-1]
    NC = T // chunk
    r = lambda t, last: t.view(B, NC, chunk, H, last).transpose(2, 3)            # (B,NC,H,chunk,last)
    kc, wc, q2c, gcc = (r(t.float(), t.shape[-1]) for t in (k, k2, q2, gc))
    bc = beta.float().view(B, NC, chunk, H).transpose(2, 3)                      # (B,NC,H,chunk)
    dif = (gcc[..., :, None, :] - gcc[..., None, :, :]).clamp(max=0.0).exp()     # (B,NC,H,t,s,E)
    M = torch.einsum('bnhte,bnhse,bnhtse->bnhts', wc, wc, dif)
    Mp = torch.einsum('bnhte,bnhse,bnhtse->bnhts', q2c, wc, dif)
    P = torch.einsum('bnhtd,bnhsd->bnhts', kc, kc)
    I = torch.eye(chunk, device=k.device, dtype=torch.float32)
    A = torch.linalg.inv(I + (bc[..., None] * P * M).tril(-1))
    return A, M, Mp


def fwd_torch(q, k, v, k2, q2, gc, beta, A, Mp, scale, chunk=CHUNK, chunk_starts=None):
    """Eq. 11 in torch: the output o (f32)."""
    B, T, H, D = q.shape
    V, E = v.shape[-1], k2.shape[-1]
    NC = T // chunk
    dev = q.device
    S = torch.zeros(B, H, E, D, V, device=dev, dtype=torch.float32)
    o = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    for n in range(NC):
        if chunk_starts is not None:
            S = torch.where(chunk_starts[:, n, None, None, None, None], 0., S)
        s0 = n * chunk
        sl = slice(s0, s0 + chunk)
        kc, qc, vc = (t[:, sl].float() for t in (k, q, v))
        wc, q2c, gcc = (t[:, sl].float() for t in (k2, q2, gc))
        bc = beta[:, sl].float()
        a, b_ = wc * gcc.exp(), q2c * gcc.exp()
        U = torch.einsum('bthd,bhedv->bthev', kc, S)
        Uq = torch.einsum('bthd,bhedv->bthev', qc, S)
        Wm = vc - torch.einsum('bthe,bthev->bthv', a, U)
        Vd = torch.einsum('bhts,bshv->bthv', A[:, n].float(), bc[..., None] * Wm)
        Pq = torch.einsum('bthd,bshd->bhts', qc, kc)
        o[:, sl] = scale * (torch.einsum('bthe,bthev->bthv', b_, Uq)
                            + torch.einsum('bhts,bshv->bthv', (Pq * Mp[:, n].float()).tril(0), Vd))
        gL = gcc[:, -1]
        S = gL.exp()[..., None, None] * S + torch.einsum(
            'bthe,bthd,bthv->bhedv', wc * (gL[:, None] - gcc).exp(), kc, Vd)
    return o


class _JointTorch(torch.autograd.Function):
    """Differentiable torch path: the reference the kernels are tested against, and the fallback off Hopper."""

    @staticmethod
    def forward(ctx, q, k, v, k2, q2, g, beta, scale, chunk_starts=None):
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(t.requires_grad) for t in (q, k, v, k2, q2, g, beta)]
            gc = _cumsum(leaves[5])
            A, _, Mp = masks_torch(leaves[1], leaves[3], leaves[4], gc, leaves[6])
            o = fwd_torch(leaves[0], leaves[1], leaves[2], leaves[3], leaves[4],
                                      gc, leaves[6], A, Mp, scale, chunk_starts=chunk_starts)
        ctx.save_for_backward(*leaves)
        ctx._o, ctx._leaves = o, leaves
        # Keep the internal autograd graph separate even when to() is a no-op.
        return o.to(q.dtype).detach()

    @staticmethod
    def backward(ctx, do):
        gr = torch.autograd.grad(ctx._o, [t for t in ctx._leaves if t.requires_grad], do.float(),
                                 allow_unused=True)
        it = iter(gr)
        out = [next(it) if t.requires_grad else None for t in ctx._leaves]
        return (*out, None, None)[:len(ctx.needs_input_grad)]


# ----------------------------------------------------------------------------------------------
# CuTe pipeline (Hopper)
# ----------------------------------------------------------------------------------------------
def _scan_cute(g, reverse=False, chunk_offsets=None, token_map=None, output_tokens=None):
    from ..kernels.sm90_joint_scan import joint_scan
    return joint_scan(g, reverse=reverse, chunk_offsets=chunk_offsets, token_map=token_map, output_tokens=output_tokens)


_TARGET_CTAS = None


def _value_cols(q, E, chunk_starts):
    """Value-axis block width of the recurrent kernels.  One thread block runs per document (or batch row),
    head and value block; when too few documents are in flight to fill about 70% of the SMs, halve the block
    from 32 to 16 columns so a single long document still occupies the GPU."""
    global _TARGET_CTAS
    width = 16 if E >= 12 else 32
    if width != 32:
        return width
    if _TARGET_CTAS is None:
        _TARGET_CTAS = int(0.7 * torch.cuda.get_device_properties(q.device).multi_processor_count)
    rows = chunk_starts.numel() - 1 if chunk_starts is not None else q.shape[0]
    H = q.shape[2]
    return 16 if rows * H * (128 // width) < _TARGET_CTAS else width


def _cute_forward(q, k, v, k2, q2, g, beta, scale, save, chunk_starts=None, token_map=None):
    from ..kernels.sm90_joint_masks import joint_masks
    if k2.shape[-1] in (1, 2, 4):
        from ..kernels.sm90_joint_fwd import joint_fwd
    else:
        from ..kernels.sm90_joint_fwd_split import joint_fwd_split as joint_fwd
    gc = _scan_cute(g, chunk_offsets=chunk_starts, token_map=token_map)
    A, Mp, M = joint_masks(k, k2, q2, gc, beta, chunk_offsets=chunk_starts, token_map=token_map)
    result = joint_fwd(q, k, v, k2, q2, gc, beta, A, Mp, scale=scale, save=save,
                          value_cols=_value_cols(q, k2.shape[-1], chunk_starts),
                          chunk_starts=chunk_starts, token_map=token_map)
    return result, gc, A, Mp, M


def _save_masks():
    """Hold the chunk masks R, R' and the UT transform from the forward (default) instead of recomputing them in the backward.

    GJ_SAVE_MASKS=0 recomputes them with the forward's own kernel (bitwise identical): one extra mask kernel per
    call, and three (B, T, H, 64) bf16 tensors less per layer, 768 MiB at 128k tokens.  Worth it only where
    memory limits the sequence length."""
    return _os.environ.get("GJ_SAVE_MASKS", "1") == "1"


def _masks(k, k2, q2, gc, beta, chunk_starts=None, token_map=None):
    from ..kernels.sm90_joint_masks import joint_masks
    return joint_masks(k, k2, q2, gc, beta, chunk_offsets=chunk_starts, token_map=token_map)


def _unpack_saved(saved):
    """(q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map) from either save set."""
    if len(saved) == 14:
        return saved
    q, k, k2, q2, gc, beta, h, W, Vd, *extra = saved
    chunk_starts = extra[0] if extra else None
    token_map = extra[1] if len(extra) > 1 else None
    A, Mp, M = _masks(k, k2, q2, gc, beta, chunk_starts, token_map)
    return (q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map)


def _save_list(q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map):
    if _save_masks():
        return (q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map)
    return (q, k, k2, q2, gc, beta, h, W, Vd, chunk_starts, token_map)


class _JointCute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k2, q2, g, beta, scale, chunk_starts=None, token_map=None):
        (o, h, W, Vd), gc, A, Mp, M = _cute_forward(q, k, v, k2, q2, g, beta, scale, True, chunk_starts, token_map)
        ctx.save_for_backward(*_save_list(q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map))
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        from ..kernels.sm90_joint_bwd_recurrent import joint_bwd_recurrent
        from ..kernels.sm90_joint_bwd_parallel import joint_bwd_parallel
        q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map = _unpack_saved(ctx.saved_tensors)
        do = do.contiguous()
        kwargs = {"token_map": token_map, "value_cols": _value_cols(q, k2.shape[-1], chunk_starts)}
        if k2.shape[-1] == 1:
            kwargs["groups"] = 1
        if chunk_starts is not None:
            kwargs["chunk_starts"] = chunk_starts
        dqp, dkp, dv, dvd, dgv, dk2v, dgo, dq2o = joint_bwd_recurrent(
            q, k, do, A, Mp, gc, k2, q2, beta, h, Vd, scale=ctx.scale, **kwargs)
        dq, dk, dgc, dk2, dq2, db = joint_bwd_parallel(q, k, do, A, Mp, M, gc, k2, q2, beta, W, Vd, dvd,
                                                      dqp, dkp, dgv, dk2v, dgo, dq2o, scale=ctx.scale,
                                                      chunk_offsets=chunk_starts, token_map=token_map)
        dg = _scan_cute(dgc, reverse=True, chunk_offsets=chunk_starts, token_map=token_map, output_tokens=q.shape[1])
        grads = (dq, dk, dv, dk2, dq2, dg, db)
        return (*(grad if needed else None for grad, needed in zip(grads, ctx.needs_input_grad)),
                *([None] * (len(ctx.needs_input_grad) - 7)))


def _check(q, k, v, k2, q2, g, beta, packed=False):
    B, T, H, D = q.shape
    E = k2.shape[-1]
    assert q.shape == k.shape and v.shape[:3] == q.shape[:3], "q, k, v must be (B, T, H, D)"
    assert D == 128 and v.shape[-1] == 128, f"head dim must be 128 (got {D}, {v.shape[-1]})"
    assert T > 0 and (packed or T % CHUNK == 0), f"T must be a multiple of {CHUNK} (got {T})"
    assert E > 0, f"E must be positive (got E={E})"
    assert k2.shape == q2.shape == g.shape == (B, T, H, E), "k2, q2, g must be (B, T, H, E)"
    assert beta.shape == (B, T, H), "beta is one write strength per head: (B, T, H)"
    for n, t in (("q", q), ("k", k), ("v", v), ("k2", k2), ("q2", q2), ("g", g), ("beta", beta)):
        assert t.is_contiguous(), f"{n} must be contiguous"
        assert t.is_cuda and t.device == q.device, f"{n} must be on the same CUDA device as q"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be bf16"
    for n, t in (("k2", k2), ("q2", q2), ("g", g), ("beta", beta)):
        assert t.dtype == torch.float32, f"{n} must be f32"


def chunk_gdn_joint(q, k, v, k2, q2, g, beta, scale=None, cu_seqlens=None, reference=False):
    """Triadic GDN with autograd for all seven tensor inputs.

    Hopper (sm90) runs the CuTe kernels at E = 1/2/4/8/12/16; `reference=True` or any other shape or
    architecture runs the torch reference pipeline.  With cu_seqlens the inputs are one packed row
    (1, T, ...): contiguous int32/int64 CUDA offsets that start at 0, end at T and strictly increase;
    document lengths are arbitrary and the state resets at every document start.
    """
    _check(q, k, v, k2, q2, g, beta, packed=cu_seqlens is not None)
    use_cute = not reference and sm_arch(q.device) == "sm90" and k2.shape[-1] in (1, 2, 4, 8, 12, 16)
    chunk_starts = restore = token_map = None
    if cu_seqlens is not None:
        from .joint_packing import pack_documents, prepare_documents
        if use_cute:
            chunk_starts, token_map = prepare_documents((q, k, v, k2, q2, g, beta), cu_seqlens)
        else:
            (q, k, v, k2, q2, g, beta), chunk_starts, restore = pack_documents((q, k, v, k2, q2, g, beta), cu_seqlens)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if use_cute:
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, k2, q2, g, beta)):
            o = _JointCute.apply(q, k, v, k2, q2, g, beta, float(scale), chunk_starts, token_map)
        else:
            o = _cute_forward(q, k, v, k2, q2, g, beta, float(scale), False, chunk_starts, token_map)[0]
    else:
        o = _JointTorch.apply(q, k, v, k2, q2, g, beta, scale, chunk_starts)
    return o if restore is None else o.index_select(1, restore)
