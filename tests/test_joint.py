"""The CuTe kernels against the torch reference pipeline: forward output and the gradients of all seven inputs,
E in 1/2/4/8, batch rows and packed documents, on one Hopper GPU.  `pytest tests` or `python tests/test_joint.py`."""
import torch
try:
    import pytest
except ImportError:   # `python tests/test_joint.py` without pytest
    class pytest:
        class mark:
            parametrize = staticmethod(lambda *a, **k: (lambda f: f))
        skip = staticmethod(lambda msg: (_ for _ in ()).throw(RuntimeError(msg)))

from cute_triadic_gdn import chunk_gdn_joint, gdn_joint_call


def l2norm(t):
    tf = t.float()
    return (tf * torch.rsqrt((tf * tf).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)


def make_inputs(B, T, H, E, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    q = l2norm(r(B, T, H, 128)).to(torch.bfloat16)
    k = l2norm(r(B, T, H, 128)).to(torch.bfloat16)
    v = (0.5 * r(B, T, H, 128)).to(torch.bfloat16)
    k2 = l2norm(torch.nn.functional.softplus(r(B, T, H, E)))
    q2 = l2norm(torch.nn.functional.softplus(r(B, T, H, E)))
    log_alpha = -torch.nn.functional.softplus(r(B, T, H, E) + 3.0) * torch.rand(H, E, device="cuda", generator=g) * 0.2
    beta = torch.sigmoid(r(B, T, H))
    return [t.contiguous().requires_grad_(True) for t in (q, k, v, k2, q2, log_alpha.float(), beta)]


def run(fn, inputs, cu, seed):
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    o = fn(*leaves, cu_seqlens=cu)
    do = torch.randn(o.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)).to(o.dtype)
    o.backward(do)
    return o.float(), [t.grad.float() for t in leaves]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


CASES = [(2, 256, 4, 1, None), (2, 256, 4, 2, None), (2, 256, 4, 4, None), (2, 256, 4, 8, None),
         (1, 640, 4, 8, [0, 70, 133, 400, 640]), (1, 320, 4, 2, [0, 64, 65, 320]), (1, 512, 4, 4, [0, 512])]


@pytest.mark.parametrize("B,T,H,E,docs", CASES)
def test_against_reference(B, T, H, E, docs):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the kernels are built for sm90")
    inputs = make_inputs(B, T, H, E, seed=E * 1000 + T)
    cu = torch.tensor(docs, dtype=torch.int32, device="cuda") if docs else None
    ref = lambda *a, cu_seqlens=None: chunk_gdn_joint(*a, cu_seqlens=cu_seqlens, reference=True)
    o_ref, g_ref = run(ref, inputs, cu, 7)
    o_ker, g_ker = run(chunk_gdn_joint, inputs, cu, 7)
    o_op, g_op = run(gdn_joint_call, inputs, cu, 7)
    assert rel(o_ker, o_ref) < 2e-2, f"forward {rel(o_ker, o_ref):.3e}"
    for name, a, b in zip("q k v k2 q2 g beta".split(), g_ker, g_ref):
        assert rel(a, b) < 5e-2, f"grad {name}: {rel(a, b):.3e}"
    assert torch.equal(o_op, o_ker), "custom-op call differs from the direct call"
    for name, a, b in zip("q k v k2 q2 g beta".split(), g_op, g_ker):
        assert torch.equal(a, b), f"custom-op grad {name} differs from the direct call"


if __name__ == "__main__":
    for case in CASES:
        test_against_reference(*case)
        print("ok", case)
