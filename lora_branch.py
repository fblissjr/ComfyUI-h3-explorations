"""Apply an H3 LoRA as a separate low-rank branch, so none of it is lost to the int8 grid.

`LoraLoaderModelOnly` merges a LoRA into the weight. On the int8 convrot
checkpoints, ComfyUI does that per layer in
`comfy/ops.py::resolve_cast_module_with_vbar`: dequantize, add the delta,
requantize at a recalculated per-row scale with stochastic rounding, and keep
the result resident. A delta smaller than one int8 step survives only in
expectation, under rounding noise several times its own size.
`bench/results/2026-09-26_int8_lora_requant.json` measures it on real layers:

- FlashGen's delta is about a hundredth of a step, and LightX2V Turbo's about
  a thousandth, in the middle blocks, so the merge keeps almost none of either.
- PDD's and TaoMate's deltas are ten times FlashGen's, and survive in part.

This node leaves the quantized weight alone and adds the LoRA at the call:
`y = W x + s * B (A x)`, exact in the compute dtype. It is `pdd_lora.py`'s
`unmerged_blocks` mechanism (`_make_unmerged_forward`), generalized to any
LoRA and to every module an H3 LoRA targets:

- **A plain Linear** (`qkv_proj`, `out_proj`, `fc1`, `adaln_proj.linear`) takes
  a `forward` object patch.
- **`mlp.fc2` cannot**, because `MLP.forward` calls
  `comfy.ops.linear_input_act(self.fc2, ...)`, which on the int8 path does the
  matmul without calling the module (`pdd_lora.py`, `UNMERGED_KINDS`). So the
  MLP's own `forward` is patched: the stock call, plus `fc2`'s branch applied
  to `swiglu(fc1(x))`.

**Costs.** Two small matmuls per targeted module per call, added into the
base output in place. The A and B matrices stay in pinned host RAM and are
copied to the device per call, `pdd_lora.py`'s reasoning: a rank-64 file
resident on a 24 GB card is about a gigabyte ComfyUI does not account for.

**Formats.** `<prefix>.lora_A.weight` / `.lora_B.weight` (or
`lora_down` / `lora_up`), an optional `.alpha`, and an optional `.diff_b`, under
`diffusion_model.`. Scale is ComfyUI's: `strength * alpha / rank`, or
`strength` with no alpha. Any other key is refused, not skipped.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

import comfy.ops
import comfy.utils
import folder_paths

log = logging.getLogger(__name__)

PREFIX = "diffusion_model."
_A = (".lora_A.weight", ".lora_down.weight")
_B = (".lora_B.weight", ".lora_up.weight")


def _host(t):
    """Contiguous, and pinned when there is a card to copy to, so the per-call
    copy can run without blocking. Pinned host memory is page-locked RAM, not
    VRAM: a rank-64 file pins about a gigabyte of the host's RAM."""
    if t is None:
        return None
    t = t.contiguous()
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:     # pinning refused (limits): pageable still works
            pass
    return t


class _Branch:
    """One module's low-rank delta, `scale * B (A x) + diff_b`, added into the base output.

    Added in place with `addmm_`: the delta never exists as its own
    output-sized tensor. On a 24 GB card under dynamic VRAM an extra
    `[tokens, 21504]` temporary per qkv call is room ComfyUI then evicts
    weights to find. The matrices live in host RAM and are copied per call.
    """

    def __init__(self, a, b, scale, diff_b):
        self.a = _host(a)
        self.b = _host(b * scale if (b is not None and scale != 1.0) else b)
        self.diff_b = _host(diff_b)

    @staticmethod
    def _dev(t, like):
        return t.to(like.device, non_blocking=True).to(like.dtype)

    def add_into(self, x, out):
        """`out += branch(x)`, in place; `out` is the base forward's fresh output."""
        flat_out = out.view(-1, out.shape[-1])
        if self.a is not None:
            flat_x = x.reshape(-1, x.shape[-1]).to(out.dtype)
            flat_out.addmm_(flat_x @ self._dev(self.a, out).T, self._dev(self.b, out).T)
        if self.diff_b is not None:
            flat_out.add_(self._dev(self.diff_b, out))
        return out


def parse_lora(sd, strength):
    """{module path under the diffusion model: _Branch}. Refuses keys it cannot place."""
    groups = {}
    unknown = []
    for key, t in sd.items():
        if not key.startswith(PREFIX):
            unknown.append(key)
            continue
        body = key[len(PREFIX):]
        for suf, slot in [(s, "a") for s in _A] + [(s, "b") for s in _B] + \
                         [(".alpha", "alpha"), (".diff_b", "diff_b")]:
            if body.endswith(suf):
                groups.setdefault(body[:-len(suf)], {})[slot] = t
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"LoRA keys this node cannot place ({len(unknown)}), e.g. "
                         f"{unknown[:3]}. Use LoraLoaderModelOnly for this file.")
    branches = {}
    for path, g in groups.items():
        a, b = g.get("a"), g.get("b")
        if (a is None) != (b is None):
            raise ValueError(f"{path}: lora_A and lora_B must come together")
        rank = a.shape[0] if a is not None else 1
        alpha = float(g["alpha"]) if "alpha" in g else None
        scale = strength * (alpha / rank if alpha is not None else 1.0)
        diff_b = g["diff_b"] * strength if "diff_b" in g else None
        branches[path] = _Branch(a, b, scale, diff_b)
    return branches


def _linear_forward(base_forward, branch):
    def forward(x):
        out = base_forward(x)
        if not out.is_contiguous():
            out = out.contiguous()
        return branch.add_into(x, out)
    return forward


#: Rows per chunk of `fc2`'s branch. The int8 path fuses swiglu into the
#: matmul so the activation is never written out; the branch needs it, so it
#: is materialized a chunk at a time. **Reasoned**: 16k rows of a 14336-wide
#: activation is under half a gigabyte in bf16.
FC2_CHUNK_ROWS = 16384


def _mlp_forward(mlp, fc1_forward, fc2_branch):
    """Core's `MLP.forward` plus `fc2`'s branch on the activation it consumes."""
    swiglu = comfy.ops.INPUT_ACT_EAGER["swiglu"]

    def forward(x):
        h = fc1_forward(x)
        out = comfy.ops.linear_input_act(mlp.fc2, h, "swiglu")
        flat_h = h.reshape(-1, h.shape[-1])
        if not out.is_contiguous():
            out = out.contiguous()
        flat_out = out.view(-1, out.shape[-1])
        for a in range(0, flat_h.shape[0], FC2_CHUNK_ROWS):
            b = min(a + FC2_CHUNK_ROWS, flat_h.shape[0])
            fc2_branch.add_into(swiglu(flat_h[a:b]), flat_out[a:b])
        return out
    return forward


def attach(model, branches):
    """Clone `model` with every branch installed as an object patch."""
    m = model.clone()
    taken = []
    patches = {}
    fc2 = {p for p in branches if p.endswith("mlp.fc2")}
    for path, branch in branches.items():
        if path in fc2:
            continue
        mod = m.get_model_object(PREFIX + path)
        # Duck-typed: the int8 checkpoints load through
        # `comfy.ops.mixed_precision_ops`, whose Linear is not a torch.nn.Linear.
        if getattr(getattr(mod, "weight", None), "ndim", 0) != 2:
            raise ValueError(f"{path} is a {type(mod).__name__} with no 2-D weight; this "
                             f"node only branches linear modules and MLP.fc2")
        patches[f"{PREFIX}{path}.forward"] = _linear_forward(mod.forward, branch)
    for path in fc2:
        parent = path[:-len(".fc2")]
        mlp = m.get_model_object(PREFIX + parent)
        fc1_key = f"{PREFIX}{parent}.fc1.forward"
        fc1_forward = patches.get(fc1_key, mlp.fc1.forward)
        patches[f"{PREFIX}{parent}.forward"] = _mlp_forward(mlp, fc1_forward, branches[path])
    for key in patches:
        if key in m.object_patches:
            taken.append(key)
    if taken:
        raise ValueError(f"object patches already taken ({len(taken)}), e.g. {taken[:3]}: "
                         f"another node patches these forwards, and the last writer "
                         f"would silently win")
    for key, fn in patches.items():
        m.add_object_patch(key, fn)
    return m


class MiniMaxH3LoRABranch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoRABranch",
            display_name="MiniMax H3 LoRA (exact branch)",
            category="MiniMax H3/loaders",
            description=(
                "Loads a LoRA for MiniMax H3 and applies it at the call, y = W x + B (A x), "
                "instead of merging it into the weight. On the int8 checkpoints a merged "
                "LoRA is requantized, and a delta below one int8 step (FlashGen, LightX2V "
                "Turbo) survives only as rounding noise. Replaces LoraLoaderModelOnly; "
                "do not use both for one file."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="ComfyUI's LoRA strength: the delta is strength * alpha / rank * B A."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        branches = parse_lora(comfy.utils.load_torch_file(path, safe_load=True), strength)
        m = attach(model, branches)
        log.info("[h3] LoRA branch: %s at strength %g, %d module(s) applied at the call",
                 lora_name, strength, len(branches))
        return io.NodeOutput(m)
