"""Apply an H3 LoRA as a separate low-rank branch, so none of it is lost to the int8 grid.

`LoraLoaderModelOnly` merges a LoRA into the weight. On the int8 convrot
checkpoints, ComfyUI does that per layer in
`comfy/ops.py::resolve_cast_module_with_vbar`: dequantize, add the delta,
requantize at a recalculated per-row scale with stochastic rounding, and keep
the result resident. A delta smaller than one int8 step survives only in
expectation, under rounding noise several times its own size.
`data/results/2026-09-26_int8_lora_requant.json` measures it on real layers:

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

**Other key names** (owner, 2026-10-05). A Kohya export names a module
`lora_unet_blocks_0_attn_qkv_proj`, and some files drop the `diffusion_model.`
prefix. `native_keys` renames both through core's own table for the loaded
model, so this node places exactly the names `LoraLoaderModelOnly` would and
keeps no module list of its own. **Not verified:** that a Kohya trainer's
`qkv_proj` rows are in core's order. The stock loader assumes the same, and no
Kohya-trained H3 file has been run here.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

import comfy.lora
import comfy.ops
import comfy.utils
import comfy.ldm.minimax.model as mm_h3
import folder_paths
from comfy.patcher_extension import WrappersMP

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

    def __init__(self, a, b, scale, diff_b, gate=None):
        self.gate = gate
        self.a = _host(a)
        self.b = _host(b * scale if (b is not None and scale != 1.0) else b)
        self.diff_b = _host(diff_b)

    @staticmethod
    def _dev(t, like):
        return t.to(like.device, non_blocking=True).to(like.dtype)

    def device_factors(self, like):
        """`(a, b, diff_b)` on `like`'s device and dtype, `None` where absent.

        A caller that adds into several slices of one output fetches these
        once and hands them to each `add_into`, so the host copy is made once
        per call and not once per slice."""
        return tuple(None if t is None else self._dev(t, like)
                     for t in (self.a, self.b, self.diff_b))

    def add_into(self, x, out, factors=None):
        """`out += branch(x)`, in place; `out` is the base forward's fresh output.
        `factors` is `device_factors(out)` when the caller already holds it."""
        if self.gate is not None and not self.gate.active:
            return out
        a, b, diff_b = self.device_factors(out) if factors is None else factors
        flat_out = out.view(-1, out.shape[-1])
        if a is not None:
            flat_x = x.reshape(-1, x.shape[-1]).to(out.dtype)
            flat_out.addmm_(flat_x @ a.T, b.T)
        if diff_b is not None:
            flat_out.add_(diff_b)
        return out


_SUFFIXES = _A + _B + (".alpha", ".diff_b")


def native_keys(sd, model):
    """`sd` with every key core's loader can place renamed under `diffusion_model.`.

    `comfy.lora.model_lora_keys_unet` is the table `LoraLoaderModelOnly`
    resolves names through, built from the loaded model's own weights: a
    Kohya-style `lora_unet_<module path with underscores>` and, on H3, the
    bare module path. A key whose name is in that table is renamed to the
    weight's module path, keeping its suffix; one already under
    `diffusion_model.` passes through; anything else is left as it is, for
    `parse_lora` to refuse. A file that is all native costs nothing.
    """
    if all(k.startswith(PREFIX) for k in sd):
        return sd
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    out = {}
    for key, t in sd.items():
        new = key
        if not key.startswith(PREFIX):
            for suf in _SUFFIXES:
                if not key.endswith(suf):
                    continue
                target = key_map.get(key[:-len(suf)])
                if isinstance(target, str) and target.startswith(PREFIX) and target.endswith(".weight"):
                    new = target[:-len(".weight")] + suf
                break
        if new in out:
            raise ValueError(f"LoRA key {key} names the same tensor as another key in the file ({new})")
        out[new] = t
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

    def forward(x, residual=None, gate=None, segments=None):
        h = fc1_forward(x)
        out = comfy.ops.linear_input_act(mlp.fc2, h, "swiglu")
        flat_h = h.reshape(-1, h.shape[-1])
        if not out.is_contiguous():
            out = out.contiguous()
        flat_out = out.view(-1, out.shape[-1])
        if fc2_branch.gate is None or fc2_branch.gate.active:
            # One host copy of fc2's factors for every chunk, and no
            # activation written out on a step the window excludes.
            factors = fc2_branch.device_factors(out)
            for a in range(0, flat_h.shape[0], FC2_CHUNK_ROWS):
                b = min(a + FC2_CHUNK_ROWS, flat_h.shape[0])
                fc2_branch.add_into(swiglu(flat_h[a:b]), flat_out[a:b], factors)
        if residual is None:
            return out
        # Core PR 16681's convention (open as of 2026-10-02): the block hands
        # the MLP its residual add, `residual + gate * mlp(h)`, which is
        # `_mod_gate`. Accepting it keeps fc2's branch when that PR merges;
        # `bench/check_lora_branch.py` runs both conventions.
        return mm_h3._mod_gate(residual, gate, out, segments)
    return forward


#: `modules` choices: which LoRA modules the branch applies. **Reasoned** from
#: how an H3 LoRA targets the model: attention is `attn.*`, the MLP is `mlp.*`,
#: and the timestep modulation is `adaln_proj.linear` (every block and the
#: final layer). The owner, 2026-09-26: vary "what's running for each adapter
#: and what it runs or skips or whatever and when at granular levels".
MODULE_CHOICES = {
    "all": lambda p: True,
    "no adaln": lambda p: "adaln_proj" not in p,
    "adaln only": lambda p: "adaln_proj" in p,
    "attention only": lambda p: ".attn." in p,
    "mlp only": lambda p: ".mlp." in p,
}


def parse_blocks(spec: str) -> set[int] | None:
    """`all`, or a list of DiT block indices and ranges like `0-24,40`. None means all.

    A named block list keeps only those DiT blocks; the token refiner and the
    final layer are applied only under `all`.
    """
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"blocks {spec!r} names no block; write 'all' for every module")
    return out


def select(branches, modules="all", blocks=None):
    """The branches a `modules` choice and a block set keep."""
    keep = MODULE_CHOICES[modules]
    out = {}
    for path, br in branches.items():
        if not keep(path):
            continue
        if blocks is not None:
            if not path.startswith("blocks."):
                continue
            if int(path.split(".")[1]) not in blocks:
                continue
        out[path] = br
    return out


class _Window:
    """Whether the branches apply at the current step: a percent window over the
    sampler's own schedule, the step's index over the schedule's length, as
    Sol's `start_percent` / `end_percent` read. Updated by a model wrapper each
    call from `transformer_options`."""

    def __init__(self, start: float, end: float):
        self.start, self.end = float(start), float(end)
        self.active = True

    def update(self, transformer_options):
        sched = transformer_options.get("sample_sigmas")
        cur = transformer_options.get("sigmas")
        if sched is None or cur is None or len(sched) < 2:
            self.active = True
            return
        i = int((sched - float(cur.flatten()[0])).abs().argmin())
        frac = i / (len(sched) - 1)
        self.active = self.start <= frac < self.end


def _window_wrapper(window):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        window.update(transformer_options)
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


def attach(model, branches):
    """Clone `model` with every branch installed as an object patch."""
    m = model.clone()
    install(m, branches)
    return m


def install(m, branches):
    """Install every branch on the ModelPatcher `m` itself, as object patches.

    `attach` is this on a clone. `MiniMaxH3PDDLoRA` calls it directly, on the
    clone it already holds, to apply its backbone at the call."""
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
        # The ORIGINAL forward, through `get_model_object` on the `.forward` key,
        # which reads the backup the clones of one checkpoint share. `mod.forward`
        # is whatever is applied right now: another model's branch, if it is
        # still loaded, which this one would then wrap (2026-09-26: FlashGen ran
        # base + Turbo + PDD + FlashGen; `check_lora_branch.py` reproduces it).
        key = f"{PREFIX}{path}.forward"
        patches[key] = _linear_forward(m.get_model_object(key), branch)
    for path in fc2:
        parent = path[:-len(".fc2")]
        mlp = m.get_model_object(PREFIX + parent)
        fc1_key = f"{PREFIX}{parent}.fc1.forward"
        fc1_forward = patches.get(fc1_key, m.get_model_object(fc1_key))
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
                io.Combo.Input("modules", options=list(MODULE_CHOICES), default="all", optional=True,
                               tooltip="Which of the LoRA's modules apply: all, everything but the "
                                       "timestep modulation (adaln), adaln alone, attention alone, or "
                                       "the MLP alone."),
                io.String.Input("blocks", default="all", optional=True,
                                tooltip="'all', or DiT block indices and ranges such as '0-24,40'. A "
                                        "list keeps only those blocks; the token refiner and the final "
                                        "layer apply only under 'all'."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="The branch applies from this fraction of the sampler's steps."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="The branch applies until this fraction of the sampler's steps. "
                                       "A few-step distill usually needs its LoRA on every step."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, modules="all", blocks="all",
                start_percent=0.0, end_percent=1.0) -> io.NodeOutput:
        if not start_percent < end_percent:
            raise ValueError(f"start_percent {start_percent} must be below end_percent {end_percent}")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = native_keys(comfy.utils.load_torch_file(path, safe_load=True), model)
        branches = parse_lora(sd, strength)
        branches = select(branches, modules, parse_blocks(blocks))
        if not branches:
            raise ValueError(f"modules={modules!r}, blocks={blocks!r} keep no module of {lora_name}")
        windowed = (start_percent, end_percent) != (0.0, 1.0)
        if windowed:
            window = _Window(start_percent, end_percent)
            for br in branches.values():
                br.gate = window
        m = attach(model, branches)
        if windowed:
            m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "h3_lora_branch_window",
                                   _window_wrapper(window))
        log.info("[h3] LoRA branch: %s at strength %g, %d module(s) applied at the call "
                 "(modules %s, blocks %s, steps %g-%g)", lora_name, strength, len(branches),
                 modules, blocks, start_percent, end_percent)
        return io.NodeOutput(m)
