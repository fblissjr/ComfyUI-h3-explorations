# comfy-kitchen: upstream main merged, and two open upstream fixes carried (2026-11-11)

The owner asked for the kitchen fork to be brought up to upstream, and for any open upstream pull request
that fixes a real problem on this card. This is what was taken, what was left, and what was measured on the
RTX 4090 before and after.

## What changed

`h3-frontier` in the kitchen fork moved from `b9b85e0` to `b63d8c7`, and the venv's build from
`0.2.41+sol.b9b85e0.up.cbc44a5` to `0.2.41+sol.b63d8c7.up.7d6a22a` (`bash vendor/rebuild_kernel.sh --check`).
Nothing was pushed. The previous wheel is still in the clone's `dist/`.

| commit | what it is |
|---|---|
| `46fa022` | merge of upstream main at `7d6a22a`: one commit past v0.2.41, a HIP port of the mesh operations. Nothing in it is compiled into an sm_89 CUDA build |
| `506ae00` | Comfy-Org/comfy-kitchen#242, as its author wrote it: with `token_aug` on, a query group with more candidate tokens than the budget is redone in one pass in a fixed order, so the tokens kept no longer depend on thread scheduling. One conflict, in a declaration the fork had also widened |
| `6f67ef7` | the CUDA half of Comfy-Org/comfy-kitchen#124: on architectures below sm_90 the row quantizer rounds `silu(gate) * up` to the storage dtype before quantizing, as the eager chain does. Upstream's quantizer for sm_90 and up already did (`8e96d65`). The pull request's HIP half is left out: this fork builds and tests CUDA only |
| `b63d8c7` | ours: that loader calls `torch_silu`, the helper upstream's sm_90 quantizer uses, in place of the pull request's own formula. The formula matched torch for bfloat16 and not for float16 |

Neither pull request had a maintainer's review on this day.

## What a render sees

- **An INT8 ConvRot layer that core calls as `linear_input_act(..., "swiglu")` changes when its activated width
  is one the fused quantizer serves**: a multiple of 256, no wider than `_CONVROT_FUSED_MAX_K` in
  `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py`, and fitting shared memory. A wider layer has
  its activation applied in torch first, on every build, and is the same before and after. By file:

  | file | layer | changes |
  |---|---|---|
  | the INT8 DiTs | `comfy/ldm/minimax/model.py::MLP`, `fc2`, every block | yes: the same seed gives a different sample |
  | the INT8 video VAE | `comfy/ldm/minimax/vae.py::FeedForward` in the decoder | yes: the same latent decodes to slightly different pixels |
  | the P-H3-Image INT8 text encoder (Qwen3-VL 4B) and prompt upsampler (Qwen3.5 9B) | `comfy/text_encoders/llama.py::MLP` | yes: their output differs |
  | **the INT8 32B encoder**, which most shipped graphs load | the same `MLP`, too wide for the fused path | **no**: its conditioning is bit for bit what it was |

  The table is the "widths" output below. For the 32B encoder it agrees with what the Pika image workflow
  session saw on the server: a control vector built by `MiniMaxH3MakeControlVector` through the INT8 32B encoder
  before the restart and again after it came out equal under `torch.equal`. A bf16 encoder never took this path.
  Core's other callers (Qwen-Image 2.1, FIBO 2, Flux 3, the Wan VSR model, MiniMax Music) follow the same width
  rule on an INT8 file.

  *Corrected twice on 2026-11-11. This item first named only the DiT's `fc2`; the Gemini session's review found
  the VAE and the encoders. The first correction then said every INT8 encoder changes, which is wrong for the
  32B encoder: the Pika image workflow session's result did not fit it, and the width rule is why.*

  A record of INT8 renders made earlier this day is for the earlier build:
  [`2026-11-11_h3_image_weights.md`](2026-11-11_h3_image_weights.md), whose render comparisons cannot be taken
  again because their bf16 reference file was deleted. Its weight-level half does not go through this path.
- **Token routing repeats.** Only with `token_routing` on; the node's default is off.
- A running server keeps the build it started with. The change takes effect at its next restart.

## What was measured

`bench/probe_kitchen_swiglu_and_token_overflow.py`; its docstring says what each figure is and what it cannot
see. The "before" half was produced by the same code before it was given that name and its `build:` line, on
the build named in its first line here. Both halves: this box, the card idle, no render in the queue.

Before, `0.2.41+sol.b9b85e0.up.cbc44a5`:

```
build: 0.2.41+sol.b9b85e0.up.cbc44a5 | torch 2.15.0+cu134 | NVIDIA GeForce RTX 4090 | full
has tokOverflow slot: False

[SwiGLU] every finite 16-bit gate value x 256 up values per row, K=256 groups of 256
  torch.bfloat16 : INT8 codes differing 19194/143360, rows touched 558
  torch.float16  : INT8 codes differing 57642/230400, rows touched 822

[SwiGLU] H3 fc2 shape, int8_linear fused vs chain output
  torch.bfloat16  M=  1024: outputs differing 4392607/5505024, fused call 0.43 ms
  torch.bfloat16  M=  4096: outputs differing 17582559/22020096, fused call 2.33 ms
  torch.bfloat16  M= 32700: outputs differing 140439460/175795200, fused call 10.80 ms
  torch.float16   M=  1024: outputs differing 5080094/5505024, fused call 0.43 ms
  torch.float16   M=  4096: outputs differing 20327493/22020096, fused call 2.17 ms
  torch.float16   M= 32700: outputs differing 162265745/175795200, fused call 11.47 ms

[token routing] repeatability over 12 repeats, error against dense attention, time per call
  overflow T=  4096 H=  2 aug= 64: 12/12 repeats differ (max 2.81e-03),     0.17 ms (no tokens     0.12 ms), error vs dense 0.1253 (no tokens 4.5686)
  overflow T=  4096 H=  2 aug=256: 11/12 repeats differ (max 1.77e-03),     0.18 ms (no tokens     0.12 ms), error vs dense 0.1253 (no tokens 4.5686)
  overflow T= 16384 H=  8 aug= 64: 12/12 repeats differ (max 3.16e-03),     0.70 ms (no tokens     0.51 ms), error vs dense 0.0624 (no tokens 9.0972)
  overflow T= 16384 H=  8 aug=256: 12/12 repeats differ (max 8.99e-04),     0.76 ms (no tokens     0.51 ms), error vs dense 0.0624 (no tokens 9.0972)
  overflow T= 36445 H= 56 aug= 64: 12/12 repeats differ (max 2.94e-03),    10.58 ms (no tokens     6.37 ms)
  overflow T= 36445 H= 56 aug=256: 12/12 repeats differ (max 2.62e-03),    11.30 ms (no tokens     6.37 ms)
  random   T=  4096 H=  2 aug= 64:  0/12 repeats differ (max 0.00e+00),     0.17 ms (no tokens     0.12 ms), error vs dense 0.7875 (no tokens 0.7918)
  random   T=  4096 H=  2 aug=256:  0/12 repeats differ (max 0.00e+00),     0.18 ms (no tokens     0.12 ms), error vs dense 0.7875 (no tokens 0.7918)
  random   T= 16384 H=  8 aug= 64:  0/12 repeats differ (max 0.00e+00),     0.70 ms (no tokens     0.51 ms), error vs dense 0.7883 (no tokens 0.7902)
  random   T= 16384 H=  8 aug=256:  0/12 repeats differ (max 0.00e+00),     0.73 ms (no tokens     0.51 ms), error vs dense 0.7868 (no tokens 0.7902)
  random   T= 36445 H= 56 aug= 64:  0/12 repeats differ (max 0.00e+00),    10.57 ms (no tokens     6.26 ms)
  random   T= 36445 H= 56 aug=256:  0/12 repeats differ (max 0.00e+00),    11.06 ms (no tokens     6.26 ms)
```

After, `0.2.41+sol.b63d8c7.up.7d6a22a`:

```
build: 0.2.41+sol.b63d8c7.up.7d6a22a | torch 2.15.0+cu134 | NVIDIA GeForce RTX 4090 | full
has tokOverflow slot: True

[SwiGLU] row quantizer, every 16-bit gate value under 1e4, rows of 256, four spreads of up
  torch.bfloat16 : INT8 codes differing 0/143360, rows touched 0
  torch.float16  : INT8 codes differing 0/230400, rows touched 0

[SwiGLU] H3 fc2 shape, int8_linear fused vs chain output
  torch.bfloat16  M=  1024: outputs differing 0/5505024, fused call 0.43 ms
  torch.bfloat16  M=  4096: outputs differing 0/22020096, fused call 2.31 ms
  torch.bfloat16  M= 32700: outputs differing 0/175795200, fused call 10.84 ms
  torch.float16   M=  1024: outputs differing 0/5505024, fused call 0.43 ms
  torch.float16   M=  4096: outputs differing 0/22020096, fused call 2.16 ms
  torch.float16   M= 32700: outputs differing 0/175795200, fused call 11.67 ms

[token routing] repeatability over 12 repeats, error against dense attention, time per call
  overflow T=  4096 H=  2 aug= 64:  0/12 repeats differ (max 0.00e+00),     0.35 ms (no tokens     0.12 ms), error vs dense 0.1253 (no tokens 4.5686)
  overflow T=  4096 H=  2 aug=256:  0/12 repeats differ (max 0.00e+00),     0.47 ms (no tokens     0.12 ms), error vs dense 0.1253 (no tokens 4.5686)
  overflow T= 16384 H=  8 aug= 64:  0/12 repeats differ (max 0.00e+00),     1.42 ms (no tokens     0.62 ms), error vs dense 0.0624 (no tokens 9.0972)
  overflow T= 16384 H=  8 aug=256:  0/12 repeats differ (max 0.00e+00),     1.48 ms (no tokens     0.62 ms), error vs dense 0.0624 (no tokens 9.0972)
  overflow T= 36445 H= 56 aug= 64:  0/12 repeats differ (max 0.00e+00),    14.80 ms (no tokens     6.37 ms)
  overflow T= 36445 H= 56 aug=256:  0/12 repeats differ (max 0.00e+00),    15.57 ms (no tokens     6.37 ms)
  random   T=  4096 H=  2 aug= 64:  0/12 repeats differ (max 0.00e+00),     0.18 ms (no tokens     0.11 ms), error vs dense 0.7875 (no tokens 0.7918)
  random   T=  4096 H=  2 aug=256:  0/12 repeats differ (max 0.00e+00),     0.18 ms (no tokens     0.11 ms), error vs dense 0.7875 (no tokens 0.7918)
  random   T= 16384 H=  8 aug= 64:  0/12 repeats differ (max 0.00e+00),     0.71 ms (no tokens     0.51 ms), error vs dense 0.7883 (no tokens 0.7902)
  random   T= 16384 H=  8 aug=256:  0/12 repeats differ (max 0.00e+00),     0.75 ms (no tokens     0.51 ms), error vs dense 0.7868 (no tokens 0.7902)
  random   T= 36445 H= 56 aug= 64:  0/12 repeats differ (max 0.00e+00),    10.77 ms (no tokens     6.28 ms)
  random   T= 36445 H= 56 aug=256:  0/12 repeats differ (max 0.00e+00),    11.34 ms (no tokens     6.28 ms)
```

Which layer widths the fused quantizer serves, the probe's "widths" section, added after the first run. A
hash that differs between the two builds is a layer whose output changed; `--small` here only because the
server was holding most of the card.

Before, from the `b9b85e0` wheel unpacked beside the venv's build:

```
build: 0.2.41+sol.b9b85e0.up.cbc44a5 | torch 2.15.0+cu134 | NVIDIA GeForce RTX 4090 | small
[widths] bfloat16, int8_linear fused vs chain, and a hash of the fused output; fused limit 16384
  K= 14336 M=  512: outputs differing   104648/131072, fused output f8e96e0ad837 | H3 DiT MLP fc2 (comfy/ldm/minimax/model.py::MLP)
  K= 14336 M= 4096: outputs differing   837634/1048576, fused output 4892c8fedd8b | H3 DiT MLP fc2 (comfy/ldm/minimax/model.py::MLP)
  K=  8192 M=  512: outputs differing   103954/131072, fused output 55cadb9b7f50 | H3 video VAE decoder FeedForward w2 (comfy/ldm/minimax/vae.py::FeedForward)
  K=  8192 M= 4096: outputs differing   833333/1048576, fused output b308669cddde | H3 video VAE decoder FeedForward w2 (comfy/ldm/minimax/vae.py::FeedForward)
  K=  9728 M=  512: outputs differing   104463/131072, fused output f8f1d87f4fde | Qwen3-VL 4B encoder MLP down_proj (comfy/text_encoders/llama.py::MLP)
  K=  9728 M= 4096: outputs differing   834762/1048576, fused output 9d7e721f0e2f | Qwen3-VL 4B encoder MLP down_proj (comfy/text_encoders/llama.py::MLP)
  K=  6144 M=  512: outputs differing   104103/131072, fused output a33285b4e292 | Qwen3.5 9B upsampler MLP down_proj
  K=  6144 M= 4096: outputs differing   832919/1048576, fused output 778079f9e1e1 | Qwen3.5 9B upsampler MLP down_proj
  K= 25600 M=  512: outputs differing        0/131072, fused output 4c352ada256f | Qwen3-VL 32B encoder MLP down_proj
  K= 25600 M= 4096: outputs differing        0/1048576, fused output 9ee39467e734 | Qwen3-VL 32B encoder MLP down_proj
```

After, the installed build:

```
build: 0.2.41+sol.b63d8c7.up.7d6a22a | torch 2.15.0+cu134 | NVIDIA GeForce RTX 4090 | small
[widths] bfloat16, int8_linear fused vs chain, and a hash of the fused output; fused limit 16384
  K= 14336 M=  512: outputs differing        0/131072, fused output f6374a2f6f20 | H3 DiT MLP fc2 (comfy/ldm/minimax/model.py::MLP)
  K= 14336 M= 4096: outputs differing        0/1048576, fused output 17e5cba1d3b3 | H3 DiT MLP fc2 (comfy/ldm/minimax/model.py::MLP)
  K=  8192 M=  512: outputs differing        0/131072, fused output 5f3372643f10 | H3 video VAE decoder FeedForward w2 (comfy/ldm/minimax/vae.py::FeedForward)
  K=  8192 M= 4096: outputs differing        0/1048576, fused output 36adbc5bc72e | H3 video VAE decoder FeedForward w2 (comfy/ldm/minimax/vae.py::FeedForward)
  K=  9728 M=  512: outputs differing        0/131072, fused output dffb8be8e7df | Qwen3-VL 4B encoder MLP down_proj (comfy/text_encoders/llama.py::MLP)
  K=  9728 M= 4096: outputs differing        0/1048576, fused output 06866cb58fa5 | Qwen3-VL 4B encoder MLP down_proj (comfy/text_encoders/llama.py::MLP)
  K=  6144 M=  512: outputs differing        0/131072, fused output 28802fd70020 | Qwen3.5 9B upsampler MLP down_proj
  K=  6144 M= 4096: outputs differing        0/1048576, fused output 0900017edd31 | Qwen3.5 9B upsampler MLP down_proj
  K= 25600 M=  512: outputs differing        0/131072, fused output 4c352ada256f | Qwen3-VL 32B encoder MLP down_proj
  K= 25600 M= 4096: outputs differing        0/1048576, fused output 9ee39467e734 | Qwen3-VL 32B encoder MLP down_proj
```

Read as data:

- Fused SwiGLU against the unfused chain: most outputs differed before, none do after, for bfloat16 and
  float16, at every size run. The fused call's time is the same to within the spread between sizes.
- Token routing on the overflow case: every repeat differed before, none after, and the error against dense
  attention is the same figure in both. On random inputs, where no group overflows, nothing differed before or
  after and the time is unchanged.
- The retry pass costs time where groups overflow: on the overflow case, built so that every group does, the
  call with token routing took about four tenths longer at the largest size. **Not measured: how many groups
  overflow on real H3 activations**, so the cost on a render lies between the two cases and is not known.
  A capture run through `sol_attn` with `token_aug` on and off, timed, would say.

Between `6f67ef7` and `b63d8c7` a test build was probed as well: bfloat16 was already bit-identical, and
float16 outputs still differed in a few percent of elements (4% at the largest size), which is what `b63d8c7`
removed.

## The fork's own tests

`tests/test_sol_attn.py` and `tests/test_int8_input_act.py` from the fork, run against a test build of the
same source with the `test_bindings_*` cases left out: all pass but one,
`test_topk_ties_over_select`, which fails identically on the build before (cosine 0.983 against a 0.99 bar).
It is upstream's test and was not looked into. The `test_bindings_*` cases fail on the build before as well,
with an illegal memory access that poisons the rest of the process; that is the subject of the open
Comfy-Org/comfy-kitchen#236, and they were left out for that reason. `bench/check_sol_kernel.py --require`
and `bench/check_solattn_correctness.py` pass on the installed build.

## Looked at and not taken

Open on this day, each a fix, none on a path this install runs:

| pull request | what it fixes | why not |
|---|---|---|
| #269, #279 | flash decode rejects a size-1 dimension whose stride a DLPack export normalized | torch 2.15 here exports the strides unchanged |
| #306, #283 | the first call inside a CUDA graph capture fails | nothing here captures CUDA graphs |
| #236 | the CUDA Sol bindings do not check dtype, layout or extents | reachable only by calling `_C` directly; it conflicts with the fork's widened bindings |
| #224 | negative attention scale | no caller passes one |
| #205 | a one-element INT8 scale with extra leading dimensions | no file here is known to store one |
| #300 | eager INT4 linear overflows in fp16 | only where the fused kernel is missing |
| #294, #135 | type hints older torch rejects | kitchen imports and runs on this torch |
| #244 | CMake cache not invalidated on a toolchain change | `vendor/rebuild_kernel.sh` already clears the build tree on one |

The performance pull requests (#331, #333 to #336) and the format drafts (#303, #330) were not considered:
the ask was for fixes.

## A rotate-only branch, prepared for upstream and not sent

`rotate` is the one fork option nearly every Sol graph uses, and upstream Sol has no rotation, so it is what
keeps the fork a fork. The kitchen clone now has a local branch `sol-rotate-pr`: upstream main at `7d6a22a`
plus two commits, `63ad5f7` (the option, direct and chunked, eager reference, HIP accepting it and refusing
True) and `6a24b29` (its tests). It carries no `blk_cnt`, `qk_balance` or `tau_map`, and not the serial
reference form of the rotation the fork still holds. Nothing was pushed and no pull request was opened.

It was written from upstream's files, not picked from the fork's commits, because those were written on top
of `qk_balance` and do not separate by cherry-pick. What says it is still the fork's kernel:

- `bench/probe_kitchen_sol_build_parity.py` hashes `sol_attn` and `sol_attn_chunked` outputs over a grid of
  options. Run on the fork's wheel from before this day, on the installed build, and on a wheel built from
  `sol-rotate-pr`: no line differs between any two of the three. So with `rotate` on or off the branch
  computes, on that grid, exactly what the fork does.
- The fork's `tests/test_sol_attn.py` cases for `rotate`, carried over without the ones that need `qk_balance`:
  all pass on the branch's build. The whole file passes but `test_topk_ties_over_select`, the same upstream
  case that fails on the fork.

Not checked: a build of stock upstream beside it (the off path is upstream's text, with the option's branches
added), any HIP build, and the branch on captured H3 activations (the figures in its commit message are the
fork's, measured before on the same arithmetic).

## Going back

Through the script, so the build record beside the venv stays true: make a worktree of the kitchen clone at
`b9b85e0` outside this repo, run `SRC=<that worktree> bash vendor/rebuild_kernel.sh`, and restart. This route
was not run on this day; the script's header documents `SRC`.

The shortcut is the `b9b85e0` wheel still in the clone's `dist/`:
`uv pip install --python <comfy-venv-python> --force-reinstall --no-deps <wheel>`, **run from the kitchen
clone's folder, never from this repo** (`AGENTS.md`: no `uv` here), then a restart. It leaves
`comfy_kitchen_build.json` naming the newer build, and `vendor/rebuild_kernel.sh --check` will say the venv and
the source differ, until the script next builds. *Corrected 2026-11-11: this section gave the shortcut alone,
without where to run it or what it leaves stale; found by the Gemini session's review.*

Either way `h3-frontier` still holds the three commits. It is a fast-forward of what `origin/h3-frontier`
holds, so the fork's history is untouched until the owner pushes; to drop the ports for good, revert them on
the branch.
