"""Triadic GDN behind torch.library custom ops, so torch.compile records one opaque node per call."""
import torch
from torch.library import custom_op, register_autograd

from .gdn_joint import _check, _cute_forward, _JointCute, _save_list


@custom_op("cute_triadic_gdn::joint_fwd", mutates_args=(), device_types="cuda")
def joint_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k2: torch.Tensor,
              q2: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
              scale: float, chunk_starts: torch.Tensor | None = None,
              token_map: torch.Tensor | None = None) -> list[torch.Tensor]:
    _check(q, k, v, k2, q2, g, beta, packed=token_map is not None)
    if torch.cuda.get_device_capability(q.device)[0] != 9 or k2.shape[-1] not in (1, 2, 4, 8, 12, 16):
        raise ValueError("gdn_joint_call requires sm90 and E=1/2/4/8/12/16")
    (o, h, W, Vd), gc, A, Mp, M = _cute_forward(q, k, v, k2, q2, g, beta, scale, True, chunk_starts, token_map)
    return [o, gc, A, Mp, M, h, W, Vd]


@joint_fwd.register_fake
def _joint_fwd_fake(q, k, v, k2, q2, g, beta, scale, chunk_starts=None, token_map=None):
    B, T, H, D = q.shape
    E = k2.shape[-1]
    PT = token_map.shape[1] * 64 if token_map is not None else T
    masks = [q.new_empty(B, PT, H, 64) for _ in range(3)]
    return [torch.empty_like(v), g.new_empty(B, PT, H, E), *masks,
            q.new_empty(B, PT // 64, H, E, D, v.shape[-1]),
            v.new_empty(B, PT, H, v.shape[-1]), v.new_empty(B, PT, H, v.shape[-1])]


class _BwdContext:
    """Stands in for an autograd ctx while the backward runs.  A plain instance with slots dies with its
    last reference, so the saved tensors are freed when the layer's backward finishes."""

    __slots__ = ('saved_tensors', 'needs_input_grad', 'scale')


@custom_op("cute_triadic_gdn::joint_bwd", mutates_args=(), device_types="cuda")
def joint_bwd(saved: list[torch.Tensor], do: torch.Tensor, scale: float,
              chunk_starts: torch.Tensor | None = None,
              token_map: torch.Tensor | None = None) -> list[torch.Tensor]:
    ctx = _BwdContext()
    ctx.saved_tensors = tuple(t.detach() for t in saved) + (chunk_starts, token_map)
    ctx.needs_input_grad = (True,) * 7 + (False,)
    ctx.scale = scale
    return list(_JointCute.backward(ctx, do.contiguous())[:-1])


@joint_bwd.register_fake
def _joint_bwd_fake(saved, do, scale, chunk_starts=None, token_map=None):
    q, k, k2, q2, gc, beta = saved[:6]
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(do),
            torch.empty_like(k2), torch.empty_like(q2), torch.empty_like(k2), torch.empty_like(beta)]


def _setup(ctx, inputs, output):
    q, k, v, k2, q2, g, beta, scale, chunk_starts, token_map = inputs
    _, gc, A, Mp, M, h, W, Vd = output
    ctx.save_for_backward(*_save_list(q, k, k2, q2, gc, beta, A, Mp, M, h, W, Vd, chunk_starts, token_map))
    ctx.scale = scale
    ctx.mark_non_differentiable(*output[1:])
    # Without this, autograd materialises a zero gradient for every non-differentiable output on each
    # backward (600 MB for the saved states alone at E=8 and the 1.3B shape).
    ctx.set_materialize_grads(False)


def _backward(ctx, grads):
    if grads[0] is None:
        return (None,) * len(ctx.needs_input_grad)
    # one read of saved_tensors: activation checkpointing (use_reentrant=False) unpacks each saved
    # tensor exactly once, so a second read of the tuple raises CheckpointError
    saved = ctx.saved_tensors
    dq, dk, dv, dk2, dq2, dg, db = joint_bwd(list(saved[:-2]), grads[0].contiguous(),
                                            ctx.scale, saved[-2], saved[-1])
    # The dispatcher may omit a trailing argument equal to its default (None),
    # even though setup_context receives the complete, default-filled schema.
    return (dq, dk, dv, dk2, dq2, dg, db, None, None, None)[:len(ctx.needs_input_grad)]


register_autograd("cute_triadic_gdn::joint_fwd", _backward, setup_context=_setup)


def gdn_joint_call(q, k, v, k2, q2, g, beta, scale=None, cu_seqlens=None):
    """Traceable Triadic GDN for training; sm90, E=1/2/4/8/12/16, D=128.

    Same inputs and packed-document contract as `chunk_gdn_joint`.  Document metadata is built on the
    device; inputs and outputs are not repacked.
    """
    q, k, v, k2, q2, g, beta = (t.contiguous() for t in (q, k, v, k2, q2, g, beta))
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    chunk_starts = token_map = None
    if cu_seqlens is not None:
        from .joint_packing import prepare_documents
        chunk_starts, token_map = prepare_documents((q, k, v, k2, q2, g, beta), cu_seqlens)
    return joint_fwd(q, k, v, k2, q2, g, beta, scale, chunk_starts, token_map)[0]
