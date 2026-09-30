# cute-triadic-gdn

CuTe DSL kernels for Triadic Gated DeltaNet, the main model of
[Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling](https://arxiv.org/abs/2609.36529).
Triadic GDN is the gated delta rule over the joint key `k2 (x) k`, with `E` state slices per head. `E = 1` is
Gated DeltaNet.

Hopper (H100, sm90) and Blackwell (sm100/sm103, tested on B300). The kernels compile at first use, there is
nothing to build.

```
pip install git+https://github.com/OliverSieberling/cute-triadic-gdn
```

## Usage

```python
from cute_triadic_gdn import gdn_joint_call

o = gdn_joint_call(q, k, v, k2, q2, g, beta, scale=None, cu_seqlens=None)
```

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `q`, `k` | `(B, T, H, 128)` | bf16 | queries and keys, L2-normalised along the last axis |
| `v` | `(B, T, H, 128)` | bf16 | values |
| `k2`, `q2` | `(B, T, H, E)` | f32 | second key and second query, L2-normalised along `E` |
| `g` | `(B, T, H, E)` | f32 | log of the decay gate of each state slice, `<= 0` |
| `beta` | `(B, T, H)` | f32 | write strength of the delta rule, in `(0, 1)` |

The output `o` is `(B, T, H, 128)` bf16, with gradients for all seven inputs. `E` can be 1, 2, 4, 8, 12 or 16.
Without `cu_seqlens`, `T` must be a multiple of 64. With `cu_seqlens` (int32 offsets `0 = c_0 < ... < c_N = T`)
the batch is one packed row `(1, T, ...)` of `N` documents of any length, and the state resets at every document
start.

Per head, with the joint key `kappa_t = k2_t (x) k_t` and state slices `S_e`:

```
S_e <- exp(g_te) S_e
S   <- S + beta_t kappa_t (v_t - S^T kappa_t)^T
o_t  = S^T (q2_t (x) q_t) * scale
```

`gdn_joint_call` runs behind `torch.library` custom ops, so it is one node under `torch.compile`.
`chunk_gdn_joint` is the same function as a plain autograd call. On other GPUs, or with `reference=True`, it runs
the torch reference instead of the kernels. `conv_split_act_call(x, weight, act_channels, cu_seqlens)` is the causal
depthwise convolution (width 4, SiLU on the first `act_channels` channels) that the Triadic layer applies to q, k,
v, k2 and q2 in one pass. `GJ_SAVE_MASKS=0` makes the backward recompute the chunk masks instead of storing them.
The numbers stay the same and every layer needs 768 MiB less at 128k tokens.

## Tests

`python tests/test_joint.py` (or `pytest tests`) checks the kernels against the torch reference on one GPU, for
the forward pass and all gradients.

## License

MIT. The chunkwise schedule follows the Hopper kernels of [FlashQLA](https://github.com/QwenLM/FlashQLA), and the convolution kernel adapts the one from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), both MIT-licensed. See `NOTICE`.

## Citation

```bibtex
@misc{sieberling2026triadic,
  title        = {Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling},
  author       = {Sieberling, Oliver and Runwal, Bharat and Jin, David and Chin, Ryan and Panda, Rameswar and Kim, Yoon},
  year         = {2026},
  eprint       = {2609.36529},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2609.36529}
}
```
