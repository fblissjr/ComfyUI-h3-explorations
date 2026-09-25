# sglang's PDD builder merges the fc1 LoRA delta with gate and value swapped, 2026-09-25

**Result: shown, one block's MLP at a time, on CPU.** sglang's offline PDD
builder maps the diffusers LoRA module `ff.net.0.proj` onto the native
checkpoint's `mlp.fc1` as a plain rename. The two layouts pack the SwiGLU
halves in opposite orders. So the fc1 half of alibaba-pai's adapter lands on the
wrong rows, and the merged MLP misses the adapter's intended output. The error
is the same order as the adapter's whole effect on that MLP.

This is the probable defect the upstream survey recorded by reading
(`docs/research/sglang_comparison.md`, "Seventh read"); this record measures
it. Whether to report it upstream is the owner's call.

## What was compared

- Source read at sglang `2f5c9ac43d`:
  `coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_pdd_weights.py::_target_of`
  maps `.ff.net.0.proj` to `.mlp.fc1`. The file imports only torch and
  safetensors, and nothing in it swaps halves. sglang's own runtime LoRA path
  does swap (`runtime/pipelines_core/lora/pipeline.py::_swap_peft_swiglu_fc1_lora_b`),
  and the builder's only test merges `attn.to_out.0`.
- Layouts, from the release on disk (`MiniMaxAI/MiniMax-H3`: `transformer/`
  is diffusers, `FL2VA/transformer/` is native):
  - diffusers `FeedForward(activation_fn="swiglu")` splits `[value; gate]`
    and returns `value * silu(gate)`
    (`coderef/diffusers/src/diffusers/models/activations.py`, `SwiGLU.forward`);
  - native `mlp.fc1` is `[gate; up]` and returns `silu(gate) * up`
    (ComfyUI checkout `comfy/ops.py::_swiglu_eager`).
- The adapter: alibaba-pai `MiniMax-H3-FL2VA-Acc-8Step.safetensors`, with its
  metadata's alpha over rank as the scale.
- Three MLPs, fp32, the same random input rows:
  - `ref`, diffusers layout with the delta added, which is the space the
    adapter was trained in;
  - `rename`, native layout with the delta added as the builder does;
  - `swapped`, native layout with the delta's halves swapped first.

  The control is the two base MLPs with no delta, which must agree.

## Result

Blocks 0, 7, 25 and 49 of FL2VA:

| block | control: native vs diffusers base | native fc1 == swap(diffusers fc1) | adapter's own effect (ref vs base) | swapped vs ref | rename (sglang) vs ref |
|---|---|---|---|---|---|
| 0 | 0 | yes | 1.142e-2 | 0 | 7.272e-3 |
| 7 | 0 | yes | 8.235e-3 | 0 | 1.024e-2 |
| 25 | 0 | yes | 6.795e-3 | 0 | 8.533e-3 |
| 49 | 0 | yes | 7.849e-2 | 0 | 4.266e-2 |

Values are relative L2 of the MLP output. The swapped merge reproduces the
reference exactly, and the plain rename does not. Its error is the same order
as everything the adapter contributes to that MLP.

## What this does not establish

- It does not show how much an end-to-end sglang PDD render degrades. That
  needs sglang, which does not run on this card. The fc2, attention and AdaLN
  merges are not affected.
- It is not a statement about this pack. Our converter swaps
  (`docs/h3_pdd.md`, the `bench/convert_pdd_lora.py` section), and the
  in-repo bake inherits that.

## Reproduce

CPU only, in the ComfyUI venv. The arguments are block indices. Set `H3_RELEASE` and `PDD_LORA`.

```python
"""Does a plain-rename merge of alibaba-pai's PDD LoRA onto native mlp.fc1 land correctly?

Reimplements (does not import) sglang's build_minimax_h3_pdd_weights mapping for one
block's MLP, and compares three MLPs on the same inputs, all in fp32 on CPU:
  ref      diffusers layout, base + delta           (the space the LoRA was trained in)
  rename   native layout, base + delta as-is         (sglang's builder: plain rename)
  swapped  native layout, base + delta, halves swapped
Plus a control: native base vs diffusers base with no delta, which must agree.
"""
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from safetensors import safe_open
R = os.environ["H3_RELEASE"]          # the MiniMaxAI/MiniMax-H3 download
L = os.environ["PDD_LORA"]            # coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/MiniMax-H3-FL2VA-Acc-8Step.safetensors
blocks = [int(b) for b in (sys.argv[1:] or ["7"])]

def get(path_glob_dir, key):
    import json, glob
    idx = json.load(open(glob.glob(f"{path_glob_dir}/*.index.json")[0]))["weight_map"]
    with safe_open(f"{path_glob_dir}/{idx[key]}", "pt") as f:
        return f.get_tensor(key).float()

def mlp_diffusers(x, w1, w2):          # FeedForward(swiglu): [value; gate] -> value * silu(gate)
    h, g = (x @ w1.T).chunk(2, dim=-1)
    return (h * torch.nn.functional.silu(g)) @ w2.T

def mlp_native(x, w1, w2):             # core: [gate; up] -> silu(gate) * up
    g, u = (x @ w1.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(g) * u) @ w2.T

def swap_halves(w):
    a, b = w.chunk(2, dim=0)
    return torch.cat([b, a], dim=0)

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

with safe_open(L, "pt") as f:
    meta = f.metadata()
    lora = {k: f.get_tensor(k).float() for k in f.keys() if ".ff.net." in k and any(f"transformer_blocks.{b}." in k for b in blocks)}
scale = float(meta["lora_alpha"]) / int(meta["lora_rank"])
torch.manual_seed(0)
for b in blocks:
    wd1 = get(f"{R}/transformer", f"transformer_blocks.{b}.ff.net.0.proj.weight")
    wd2 = get(f"{R}/transformer", f"transformer_blocks.{b}.ff.net.2.weight")
    wn1 = get(f"{R}/FL2VA/transformer", f"blocks.{b}.mlp.fc1.weight")
    wn2 = get(f"{R}/FL2VA/transformer", f"blocks.{b}.mlp.fc2.weight")
    p = f"transformer_blocks.{b}.ff.net."
    d1 = (lora[p + "0.proj.lora_up"] @ lora[p + "0.proj.lora_down"]) * scale
    d2 = (lora[p + "2.lora_up"] @ lora[p + "2.lora_down"]) * scale
    x = torch.randn(64, wd1.shape[1]) * wd1.std() ** 0 * 0.5
    base_ref = mlp_diffusers(x, wd1, wd2)
    print(f"block {b}: layout control  native base vs diffusers base  rel {rel(mlp_native(x, wn1, wn2), base_ref):.2e}"
          f"   weights: native fc1 == swap(diffusers)? {torch.equal(wn1, swap_halves(wd1))}  same order? {torch.equal(wn1, wd1)}")
    ref = mlp_diffusers(x, wd1 + d1, wd2 + d2)
    rename = mlp_native(x, wn1 + d1, wn2 + d2)
    swapped = mlp_native(x, wn1 + swap_halves(d1), wn2 + d2)
    lora_effect = rel(ref, base_ref)
    print(f"  LoRA's own effect on the MLP output (ref vs base)    rel {lora_effect:.3e}")
    print(f"  swapped merge vs ref                                 rel {rel(swapped, ref):.3e}")
    print(f"  plain-rename merge (sglang builder) vs ref           rel {rel(rename, ref):.3e}")
    print(f"  plain-rename merge vs base (how much of the LoRA it keeps is not the point; this is its distance from base) rel {rel(rename, base_ref):.3e}")
```
