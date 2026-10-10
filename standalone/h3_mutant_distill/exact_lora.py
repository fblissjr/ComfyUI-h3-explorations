"""Apply an H3 distill LoRA at the call, so none of it is lost to the int8 grid.

One node for both files the recipes load: a FlashGen LoRA and a converted PDD8
sidecar. It is the source pack's `MiniMaxH3LoRABranch` (`lora_branch.py`) and
the part of its `MiniMaxH3PDDLoRA` (`pdd_lora.py`) these files reach, at that
pack's settings `backbone_apply="exact branch"`, heads patched, strength 1.
Parity with those two nodes is checked by rendering, not assumed:
`bench/check_mutant_parity.py` in ComfyUI-h3-explorations.

Why not `LoraLoaderModelOnly`: on the int8 convrot checkpoints ComfyUI merges a
LoRA by dequantizing, adding the delta and requantizing. FlashGen's delta is
about a hundredth of one int8 step, so the merge keeps almost none of it, and
PDD's survives only in part (`data/results/2026-09-26_int8_lora_requant.json`
there). This node leaves the int8 weight alone and adds `B (A x)` at the call.

A PDD file carries three more things, all handled here:
- its adaln update, pre-solved into the pruned checkpoint's 8-column curve
  basis: an ordinary `diff` weight patch on an fp16 layer, through core;
- a bank of 32 per-interval output heads: each step uses the step-size
  weighted mean of the heads its sigma interval spans, fused once in fp64;
- the checkpoint fingerprints it was converted against, checked before
  anything is patched.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
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

#: A PDD file converted against another partition sits about 0.05 away on
#: `final_layer.video_out`, a dtype cast a few thousandths. Inherited from
#: `pdd_lora.PARTITION_TOLERANCE`.
PARTITION_TOLERANCE = 0.015
#: The curve table the adaln bake was solved against, same reasoning.
#: Inherited from `pdd_lora.TABLE_TOLERANCE`.
TABLE_TOLERANCE = 5e-3
#: Rows per chunk of `fc2`'s branch; 16k rows of the 14336-wide activation is
#: under half a gigabyte in bf16. Inherited from `lora_branch.FC2_CHUNK_ROWS`.
FC2_CHUNK_ROWS = 16384


# ---- the branch: y = W x + s * B (A x) ------------------------------------

def _host(t):
    """Contiguous, and pinned when there is a card, so the per-call copy does not block."""
    if t is None:
        return None
    t = t.contiguous()
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


class _Branch:
    """One module's delta, `scale * B (A x) + diff_b`, added into the base output in place."""

    def __init__(self, a, b, scale, diff_b):
        self.a = _host(a)
        self.b = _host(b * scale if (b is not None and scale != 1.0) else b)
        self.diff_b = _host(diff_b)

    @staticmethod
    def _dev(t, like):
        return t.to(like.device, non_blocking=True).to(like.dtype)

    def add_into(self, x, out):
        flat_out = out.view(-1, out.shape[-1])
        if self.a is not None:
            flat_x = x.reshape(-1, x.shape[-1]).to(out.dtype)
            flat_out.addmm_(flat_x @ self._dev(self.a, out).T, self._dev(self.b, out).T)
        if self.diff_b is not None:
            flat_out.add_(self._dev(self.diff_b, out))
        return out


def parse_lora(sd, strength):
    """{module path under the diffusion model: _Branch} for every `diffusion_model.*` key."""
    groups, unknown = {}, []
    for key, t in sd.items():
        body = key[len(PREFIX):]
        for suf, slot in [(s, "a") for s in _A] + [(s, "b") for s in _B] + \
                         [(".alpha", "alpha"), (".diff_b", "diff_b")]:
            if body.endswith(suf):
                groups.setdefault(body[:-len(suf)], {})[slot] = t
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"LoRA keys this node cannot place ({len(unknown)}), e.g. {unknown[:3]}")
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


def parse_blocks(spec: str):
    """`all` (None), or DiT block indices and ranges such as `34-49`."""
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    out = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    if not out:
        raise ValueError(f"blocks {spec!r} names no block; write 'all' for every module")
    return out


def select(branches, blocks):
    """A block list keeps only those DiT blocks; the refiner and final layer only under 'all'."""
    if blocks is None:
        return branches
    return {p: br for p, br in branches.items()
            if p.startswith("blocks.") and int(p.split(".")[1]) in blocks}


def _linear_forward(base_forward, branch):
    def forward(x):
        out = base_forward(x)
        if not out.is_contiguous():
            out = out.contiguous()
        return branch.add_into(x, out)
    return forward


def _mlp_forward(mlp, fc1_forward, fc2_branch):
    """Core's `MLP.forward` plus `fc2`'s branch. The int8 path fuses swiglu into
    fc2's matmul without calling the module, so fc2 cannot take its own patch."""
    swiglu = comfy.ops.INPUT_ACT_EAGER["swiglu"]

    def forward(x, residual=None, gate=None, segments=None):
        h = fc1_forward(x)
        out = comfy.ops.linear_input_act(mlp.fc2, h, "swiglu")
        flat_h = h.reshape(-1, h.shape[-1])
        if not out.is_contiguous():
            out = out.contiguous()
        flat_out = out.view(-1, out.shape[-1])
        for a in range(0, flat_h.shape[0], FC2_CHUNK_ROWS):
            b = min(a + FC2_CHUNK_ROWS, flat_h.shape[0])
            fc2_branch.add_into(swiglu(flat_h[a:b]), flat_out[a:b])
        if residual is None:
            return out
        # Core PR 16681's convention (open as of 2026-10-02): the block hands
        # the MLP its residual add, `residual + gate * mlp(h)`, which is
        # `_mod_gate`. Accepting it keeps fc2's branch when that PR merges;
        # `bench/check_lora_branch.py` runs both conventions.
        return mm_h3._mod_gate(residual, gate, out, segments)
    return forward


def install(m, branches):
    """Every branch as an object patch on the ModelPatcher `m`.

    Bases come through `get_model_object`, which reads the backup the clones of
    one checkpoint share, never the live attribute: while another graph's
    patcher is still applied, the live forward is that graph's branch, and
    wrapping it stacks two LoRAs."""
    patches = {}
    fc2 = {p for p in branches if p.endswith("mlp.fc2")}
    for path, branch in branches.items():
        if path in fc2:
            continue
        mod = m.get_model_object(PREFIX + path)
        if getattr(getattr(mod, "weight", None), "ndim", 0) != 2:
            raise ValueError(f"{path} is a {type(mod).__name__} with no 2-D weight")
        key = f"{PREFIX}{path}.forward"
        patches[key] = _linear_forward(m.get_model_object(key), branch)
    for path in fc2:
        parent = path[:-len(".fc2")]
        mlp = m.get_model_object(PREFIX + parent)
        fc1_key = f"{PREFIX}{parent}.fc1.forward"
        fc1_forward = patches.get(fc1_key, m.get_model_object(fc1_key))
        patches[f"{PREFIX}{parent}.forward"] = _mlp_forward(mlp, fc1_forward, branches[path])
    taken = [k for k in patches if k in m.object_patches]
    if taken:
        raise ValueError(f"object patches already taken ({len(taken)}), e.g. {taken[:3]}: "
                         f"another LoRA node in this chain patches these forwards")
    for key, fn in patches.items():
        m.add_object_patch(key, fn)


# ---- PDD: the per-interval head bank ---------------------------------------

def base_sigma(sigma, shift):
    """Undo the flow shift: the point of the unshifted grid a sigma came from."""
    return sigma / (shift + sigma * (1.0 - shift))


def grid_index(sigma: float, shift: float, num_steps: int) -> int:
    s = min(max(float(sigma), 0.0), 1.0)
    return int(round((1.0 - float(base_sigma(torch.tensor(s, dtype=torch.float64), shift))) * num_steps))


def fuse_block(stack, shift, num_steps, start, stop):
    """The block `[start, stop)`'s fused head: the step-size weighted mean of its
    heads, in fp64. `pdd_math.fuse_block` in the source pack."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    t = 1.0 - shift * sigma / (1 + (shift - 1) * sigma)
    steps = t.diff()
    plan = torch.zeros(steps.shape[0], dtype=torch.float64)
    plan[start:stop] = steps[start:stop] / steps[start:stop].sum()
    block = stack[start:stop].detach().to("cpu", torch.float64)
    fused = torch.tensordot(plan[start:stop], block, dims=([0], [0]))
    return fused.to(torch.float32).to(stack.device)


class _Schedule:
    """The grid block this step spans, from the sampler's own sigmas.

    Updated by a diffusion-model wrapper on every call: `sample_sigmas` is the
    schedule this sampler run evaluates, `sigmas` the current one. The block
    runs from the current sigma's grid point to the next scheduled sigma's, so
    a partial schedule (PDD8 stopped at 0.8) selects the same blocks as the
    full one."""

    def __init__(self, shift, num_steps):
        self.shift, self.num_steps = shift, num_steps
        self.block = (0, 1)

    def update(self, transformer_options):
        sched = transformer_options.get("sample_sigmas")
        cur = transformer_options.get("sigmas")
        if sched is None or cur is None:
            raise RuntimeError("PDD heads need the sampler's sigma schedule "
                               "(transformer_options['sample_sigmas']); this sampler gives none")
        sched = sched.detach().flatten().to("cpu", torch.float64)
        i = int((sched - float(cur.flatten()[0])).abs().argmin())
        nxt = sched[min(i + 1, sched.shape[0] - 1)]
        start = min(grid_index(sched[i], self.shift, self.num_steps), self.num_steps - 1)
        stop = max(grid_index(nxt, self.shift, self.num_steps), start + 1)
        self.block = (start, stop)


class _FusedHeads:
    """One stream's fused head per block, cached; `base + strength * (fused - base)`."""

    def __init__(self, bank_w, bank_b, base_w, base_b, shift, num_steps, strength):
        self.bank_w, self.bank_b = bank_w, bank_b
        self.base_w, self.base_b = base_w, base_b
        self.shift, self.num_steps, self.strength = shift, num_steps, strength
        self._master, self._cast = {}, {}

    def get(self, block, device, dtype):
        key = (block, str(device), dtype)
        hit = self._cast.get(key)
        if hit is None:
            master = self._master.get(block)
            if master is None:
                w = fuse_block(self.bank_w, self.shift, self.num_steps, *block)
                b = fuse_block(self.bank_b, self.shift, self.num_steps, *block)
                master = (self.base_w + self.strength * (w - self.base_w),
                          self.base_b + self.strength * (b - self.base_b))
                self._master[block] = master
            hit = self._cast[key] = (master[0].to(device, dtype), master[1].to(device, dtype))
        return hit


def _head_forward(heads, schedule):
    def forward(inp):
        w, b = heads.get(schedule.block, inp.device, inp.dtype)
        return F.linear(inp, w, b)
    return forward


def _schedule_wrapper(schedule, shifts):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        for name, want in shifts.items():
            got = transformer_options.get(f"minimax_h3_sigma_shift_{name}")
            if got is not None and abs(float(got) - want) > 1e-6:
                raise RuntimeError(f"this graph sets shift_{name}={got}, but the PDD heads "
                                   f"were fused at {want}")
        schedule.update(transformer_options)
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


def apply_pdd(m, sd, meta, strength, name):
    """The adaln bake as core weight patches, and the head bank, on the clone `m`."""
    dm = m.get_model_object("diffusion_model")
    if not getattr(dm, "use_adaln_curves", False):
        raise RuntimeError(f"{name} is for the pruned (curve-form) H3 checkpoints; this one is not")

    ref = sd["h3_pdd.base_video_out"].to(torch.float32)
    live = dm.final_layer.video_out.weight.detach().to(torch.float32).cpu()
    if live.shape != ref.shape:
        raise RuntimeError(f"final_layer.video_out is {tuple(live.shape)}, not {tuple(ref.shape)}: "
                           f"another PDD implementation enlarged it on the cached model. "
                           f"Restart ComfyUI.")
    dist = float((live - ref).norm() / ref.norm())
    if dist > PARTITION_TOLERANCE:
        raise RuntimeError(f"{name} was converted for {meta.get('h3_pdd_pruned_base', '?')}; "
                           f"this checkpoint is {dist:.4f} away from it (fl2va and ref2va "
                           f"differ by about 0.05)")
    table = sd["h3_pdd.adaln_table"].to(torch.float32)
    live_t = dm.adaln_t_table.detach().to(torch.float32).cpu()
    if live_t.shape != table.shape or \
            float((live_t - table).norm() / table.norm()) > TABLE_TOLERANCE:
        raise RuntimeError(f"{name}'s adaln update was solved against a different curve table")

    n = int(meta["adaln_modules"])
    patch = {}
    for i in range(n):
        base_key = f"{PREFIX}blocks.{i}.adaln_proj.linear"
        patch[f"{base_key}.diff"] = sd[f"h3_pdd.adaln_baked.blocks.{i}.diff"]
        patch[f"{base_key}.diff_b"] = sd[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"]
    loaded = comfy.lora.load_lora(patch, comfy.lora.model_lora_keys_unet(m.model, {}))
    applied = m.add_patches(loaded, strength)
    if len(applied) != len(loaded) or len(loaded) != 2 * n:
        raise RuntimeError(f"{name}: {len(applied)} of {2 * n} adaln patches matched this model")
    return n


def install_heads(m, sd, meta, strength):
    shift_v, shift_a = float(meta["pdd_shift_video"]), float(meta["pdd_shift_audio"])
    num_steps = int(meta["pdd_num_steps"])
    final_layer = m.get_model_object("diffusion_model.final_layer")
    schedule = _Schedule(shift_v, num_steps)
    for stream, out_name, shift in (("video", "video_out", shift_v), ("audio", "audio_out", shift_a)):
        live = getattr(final_layer, out_name)
        heads = _FusedHeads(sd[f"h3_pdd.bank.{stream}.weight"].to(torch.float32),
                            sd[f"h3_pdd.bank.{stream}.bias"].to(torch.float32),
                            live.weight.detach().to(torch.float32).cpu(),
                            live.bias.detach().to(torch.float32).cpu(),
                            shift, num_steps, strength)
        key = f"diffusion_model.final_layer.{out_name}.forward"
        if key in m.object_patches:
            raise RuntimeError(f"{key} is already patched: two PDD nodes in one chain")
        m.add_object_patch(key, _head_forward(heads, schedule))
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "h3_mutant_pdd_schedule",
                           _schedule_wrapper(schedule, {"video": shift_v, "audio": shift_a}))


# ---- the node ----------------------------------------------------------------

class H3ExactLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ExactLoRA",
            display_name="H3 Exact LoRA (FlashGen / PDD)",
            category="MiniMax H3/loaders",
            description=(
                "Applies a MiniMax H3 distill LoRA at the call instead of merging it into "
                "the int8 weights, where most of it would be rounded away. Loads a FlashGen "
                "LoRA or a converted PDD8 sidecar; for PDD it also installs the adaln update "
                "and the per-step output heads. Experimental."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.String.Input("blocks", default="all",
                                tooltip="'all', or DiT blocks such as '34-49' (FlashGen only). A list "
                                        "skips the token refiner and final layer."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, blocks="all") -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        meta = meta or {}
        strength = float(strength)
        pdd = "h3_pdd.bank.video.weight" in sd
        stray = [k for k in sd if not k.startswith((PREFIX, "h3_pdd."))]
        if stray:
            raise ValueError(f"{lora_name} has keys this node cannot place, e.g. {stray[:3]}")
        block_set = parse_blocks(blocks)
        if pdd and block_set is not None:
            raise ValueError("blocks must be 'all' for a PDD file: its heads and adaln are whole-model")

        branches = select(parse_lora({k: v for k, v in sd.items() if k.startswith(PREFIX)}, strength),
                          block_set)
        if not branches:
            raise ValueError(f"blocks={blocks!r} keeps no module of {lora_name}")
        m = model.clone()
        n_adaln = apply_pdd(m, sd, meta, strength, lora_name) if pdd else 0
        install(m, branches)
        if pdd:
            install_heads(m, sd, meta, strength)
        log.info("[h3-mutant] %s at strength %g: %d module(s) at the call (blocks %s)%s",
                 lora_name, strength, len(branches), blocks,
                 f", {n_adaln} adaln patches, heads fused per step" if pdd else "")
        return io.NodeOutput(m)
