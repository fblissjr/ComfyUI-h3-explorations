"""Apply a converted MiniMax-H3 Parallel Decoding Distillation LoRA.

PDD (arXiv 2607.26004, released for H3 by alibaba-pai) reaches the model on
three surfaces at once, and only one of them is a LoRA in the sense
`LoraLoaderModelOnly` understands. `bench/convert_pdd_lora.py` does everything
that can be done offline; this node does the rest, which is the part that needs
the loaded checkpoint in front of it.

    backbone   208 modules   weight patch, through comfy.lora
    adaln       50 modules   weight patch on an unpruned base;
                             runtime injection on a pruned one
    heads        8 pairs     per-step swap of final_layer's two output linears

## Why the head swap is keyed on the timestep

The vendor arms its parallel heads from a `register_forward_hook` that
increments a counter once per forward and wraps at `nfe`. Nothing ties that
counter to the schedule. One extra evaluation -- a CFG uncond pass, a warmup, a
shape probe, `torch.compile` tracing, an offload dry run -- desyncs it from the
trajectory for the rest of the render, and the wrap-around hides it instead of
raising.

We derive the step from `t_emb`, which is what the model was actually called
with, so it cannot desync. `final_layer.forward` receives separate rows for the
video and audio streams, and PDD runs those on separate schedules (shift 12 and
3), so the per-stream split falls out of the model's own signature rather than
having to be threaded through.

`step_for_t` picks by interval membership, not by nearest boundary, so a run at
a step count or shift the file was not fused for degrades to the closest
available head rather than an arbitrary one -- and `boundary_residual` says so
in the log, once, because nothing else about such a render looks wrong.

## Three traps this node is shaped around

**Patch `.forward` attributes; never wrap a module.** A wrapper `nn.Module`
holding the original under `.base` injects `.base.linear.weight` into the
parameter tree. ComfyUI's dynamic-VRAM streaming loader records every such path
in its backup and restores it by that path on unload -- by which time the
object patch has reverted the module and the path no longer resolves. Pattern
and diagnosis borrowed from `ComfyUI-MiniMax-H3-Turbo` (`_make_adaln_forward`,
its issue #4); the code here is ours.

**LoRA tensors are captured locals, never registered.** Same reason, plus it
keeps them off the offload bookkeeping entirely. They are cast to the
activation's device and dtype per call, which is also what makes the
CPU-offload case work while the projection runs on GPU.

**The adaln injection is partition-specific.** On a pruned checkpoint the
2688-dim time-embedding space the adaln delta lives in has been replaced by an
8-column curve, so the update is re-injected at run time from a grid of
`silu(t_emb)`. That grid differs between fl2va and ref2va by 7.8% relative --
which is why ours is derived at conversion time from the same checkpoint that
supplies the partition fingerprint, rather than bundled once. The turbo pack
ships a single fl2va grid; reusing it for ref2va would feed the injection a
7.8%-wrong input and render without complaint.

## Strength

Interpolates all three mechanisms together: the backbone and adaln deltas scale
linearly, and each fused head is `base + strength * (fused - base)` against the
checkpoint's own head. So `strength=0.0` is exactly the base model on every
surface, which is what makes it a usable control here -- unlike the weight-patch
path, where 0.0 short-circuits the dequantise/add/requantise round trip and is
therefore not like-for-like (see `workflows/h3_config.py`).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io

try:                                     # loaded as a package by ComfyUI
    from .pdd_math import block_bounds, boundary_residual, silu_temb_grid, step_for_t
except ImportError:                      # loaded as a bare module by a script
    from pdd_math import block_bounds, boundary_residual, silu_temb_grid, step_for_t

logger = logging.getLogger(__name__)

#: Log a warning once per patch when the recovered `t` sits this far from any
#: block boundary. On the schedule the heads were fused for the residual is
#: ~1e-9; the tightest boundary gap at 32/4 shift 12 is 0.0118, so this is well
#: inside "wrong schedule" and well outside "float noise".
BOUNDARY_TOLERANCE = 2e-3

#: Relative-Frobenius distance allowed between the loaded checkpoint's
#: `final_layer.video_out` and the one the LoRA was converted against. The
#: fl2va and ref2va partitions sit ~0.05 apart (measured 2026-08-26 across the
#: six H3 checkpoints on this box) and a dtype cast on load moves it a few
#: thousandths, so anything in between is unambiguous. Set an order of
#: magnitude below the partition gap and an order above a cast.
PARTITION_TOLERANCE = 0.015


def _is_minimax_h3(diffusion_model) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def _row_index(row) -> int:
    """The `t_emb` row a stream's segment refers to.

    `final_layer` is handed an int normally, and a per-token LongTensor when a
    denoise mask puts rows at different strengths. In that case the stream's
    own timestep is the SMALLEST row value: masked rows are pinned toward the
    conditioning timestep (`rows_t = (1 - m * sigma).clamp(max=t_pin)`), so the
    fully-denoised rows -- the ones on the sampler's actual trajectory -- carry
    the minimum. Taking the max or the mean would read a pinned conditioning
    row and select a head for a time the sampler never visits.
    """
    if torch.is_tensor(row):
        return int(row.min().item())
    return int(row)


class _StepTracker:
    """Recovers the sampling step from `t_emb`, per stream, per forward.

    Holds the `[rows, D]` table `t_emb` is drawn from -- the model's own
    `adaln_t_table` when pruned, the time embedder's output grid when not -- so
    a nearest-row lookup turns an embedding back into a `t`. The table is the
    live model's, so this cannot be pointed at the wrong partition.
    """

    def __init__(self, table, bounds_v, bounds_a, nfe, label):
        self.table = table
        self.bounds_v = bounds_v
        self.bounds_a = bounds_a
        self.nfe = nfe
        self.label = label
        self.video = 0
        self.audio = 0
        self.warned = False

    def _t(self, t_emb, row) -> float:
        e = t_emb[_row_index(row)].detach().float().reshape(1, -1)
        table = self.table.to(e.device, torch.float32)
        j = int(torch.cdist(e, table).argmin())
        return j / (table.shape[0] - 1)

    def update(self, t_emb, video_seg, audio_seg) -> None:
        tv = self._t(t_emb, video_seg[2])
        ta = self._t(t_emb, audio_seg[2])
        # Snap at the same tolerance the residual check calls "on schedule",
        # so the two cannot disagree about which regime this render is in.
        self.video = step_for_t(tv, self.bounds_v, self.nfe, BOUNDARY_TOLERANCE)
        self.audio = step_for_t(ta, self.bounds_a, self.nfe, BOUNDARY_TOLERANCE)
        if not self.warned:
            rv = boundary_residual(tv, self.bounds_v)
            ra = boundary_residual(ta, self.bounds_a)
            if max(rv, ra) > BOUNDARY_TOLERANCE:
                self.warned = True
                logger.warning(
                    "[h3-pdd] %s: this render is NOT on the schedule the heads "
                    "were fused for. video t=%.5f is %.4f from the nearest "
                    "block boundary, audio t=%.5f is %.4f (tolerance %.4f). "
                    "The fused output heads assume %d steps at the shifts "
                    "recorded in the file; check the sampler's step count and "
                    "MiniMaxH3SigmaShift, or reconvert at the shifts you want. "
                    "Sampling continues on the nearest available head.",
                    self.label, tv, rv, ta, ra, BOUNDARY_TOLERANCE, self.nfe)


def _make_final_layer_forward(base_forward, tracker):
    """Bookkeeping only, then the stock forward.

    Deliberately does not reimplement the modulation maths: the two head swaps
    below are separate patches on the output linears, so `FinalLayer.forward`
    stays upstream's and a change to it does not silently diverge here.
    """
    def forward(x, t_emb, video_seg, audio_seg):
        tracker.update(t_emb, video_seg, audio_seg)
        return base_forward(x, t_emb, video_seg, audio_seg)
    return forward


def _make_head_forward(weights, biases, tracker, stream):
    """Replace one output linear with the fused head for the current step.

    The masters stay on CPU and each (device, dtype) pair is materialised once,
    not per call -- the per-device cache pattern is borrowed from
    `ComfyUI-MiniMaxH3-PDD-Mamad8::PDDHeads.for_device`, which patches the same
    two projections for a different PDD artifact family. Like theirs, the
    tensors are held in a closure and never registered on a module, so they
    stay out of the streaming loader's backup bookkeeping.
    """
    cache: dict[tuple, tuple] = {}

    def forward(inp):
        k = tracker.video if stream == "video" else tracker.audio
        key = (str(inp.device), inp.dtype)
        entry = cache.get(key)
        if entry is None:
            entry = (weights.to(inp.device, inp.dtype),
                     biases.to(inp.device, inp.dtype))
            cache[key] = entry
        return F.linear(inp, entry[0][k], entry[1][k])
    return forward


def _make_adaln_forward(base, a, b, grid, table, strength):
    """Curve-mode adaln injection: add `strength * B @ A @ silu(t_emb)`.

    On a pruned checkpoint `t_emb` is an 8-column curve coordinate and the
    adaln linear consumes it directly (`apply_silu` is False). The LoRA update
    lives in the 2688-dim `silu(t_emb)` space that was collapsed away, so it is
    recovered per row: nearest row of the model's own `adaln_t_table`, then the
    matching row of the grid this file was converted with.

    Approach borrowed from `ComfyUI-MiniMax-H3-Turbo::_make_adaln_forward`,
    which solved this for the v4 turbo pack. Reimplemented rather than imported:
    that package bundles an fl2va-only grid, and ref2va is the arm this exists
    for.

    `a` and `b` are cast per call and deliberately NOT cached per device, unlike
    the output heads. Caching would pin roughly 25 MB of fp32 per block across
    50 blocks -- over a gigabyte of VRAM standing next to a 24 GB checkpoint on
    a 24 GB card, to save transfers that cost about a second across a whole
    render. The heads are cached because there are two of them, not a hundred.
    """
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        tb = table.to(t_emb.device, torch.float32)
        idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
        st = grid.to(x.device, x.dtype)[idx]                   # [M, 2688]
        av = a.to(x.device, x.dtype)
        bv = b.to(x.device, x.dtype)
        x = x + strength * (bv @ (av @ st.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


class MiniMaxH3PDDLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return io.Schema(
            node_id="MiniMaxH3PDDLoRA",
            display_name="MiniMax H3 PDD LoRA (parallel decoding)",
            category="loaders/minimax",
            description=(
                "Loads a Parallel Decoding Distillation LoRA converted by "
                "bench/convert_pdd_lora.py. PDD is not step distillation: the "
                "trajectory is a 32-point grid whose final output head is "
                "replicated per interval, and each sampling step decodes a "
                "block of those heads as one mean velocity. The block "
                "boundaries are exactly the plain 8-step shifted schedule, so "
                "a PDD arm changes the sampler's step count and NOTHING else "
                "-- the shift stays at the checkpoint's own 12/3. Place where "
                "a LoRA loader goes: before MiniMaxH3SigmaShift and before the "
                "attention nodes."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    options=[n for n in folder_paths.get_filename_list("loras")
                             if "pdd" in n.lower()],
                    tooltip=(
                        "A CONVERTED PDD file. The published alibaba-pai "
                        "weights do not load here or anywhere else in "
                        "ComfyUI: their keys are diffusers-side and their "
                        "suffixes are bare `lora_down`/`lora_up`, which no "
                        "ComfyUI adapter matches, so every one of the 728 "
                        "tensors would be skipped with a log line and the "
                        "render would come out as an undistilled 8-step pass. "
                        "Run bench/convert_pdd_lora.py first."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip=(
                        "Scales all three mechanisms together. 1.0 is the "
                        "vendor's own default and what their demo clips were "
                        "rendered at. 0.0 is exactly the base model here -- "
                        "the heads fall back to the checkpoint's own, not just "
                        "the weight deltas to zero -- so unlike the plain LoRA "
                        "path it is a real control."
                    ),
                ),
                # APPENDED, not inserted. Saved graphs match widget values by
                # index, so a new widget ahead of `strength` would land an old
                # graph's float on this boolean. Same rule as the head_chunks
                # input on MiniMaxH3SageAttention.
                io.Boolean.Input(
                    "patch_heads", default=True, optional=True,
                    tooltip=(
                        "Off runs the backbone and adaln updates against the "
                        "checkpoint's OWN output heads -- PDD's whole "
                        "mechanism disabled, everything else applied. That is "
                        "the control for whether the per-interval heads earn "
                        "their complexity: measured against the base head the "
                        "fused heads sit 0.005 apart early and 0.015 apart at "
                        "the last step, so if this arm is indistinguishable "
                        "the head machinery is not what is doing the work. "
                        "On by default; turning it off is an experiment, not "
                        "a fallback."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0,
                patch_heads=True) -> io.NodeOutput:
        import comfy.lora
        import comfy.utils
        import folder_paths

        dm = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(dm):
            raise RuntimeError(
                f"This node only patches MiniMax H3; got {type(dm).__name__}.")

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd, meta = comfy.utils.load_torch_file(path, return_metadata=True)
        meta = meta or {}
        if "h3_pdd_converter_version" not in meta:
            raise RuntimeError(
                f"{lora_name} was not produced by bench/convert_pdd_lora.py "
                f"(no h3_pdd_converter_version in its metadata). The published "
                f"alibaba-pai files must be converted first; loading one "
                f"directly applies nothing at all.")

        nfe = int(meta["pdd_nfe"])
        num_steps = int(meta["pdd_num_steps"])
        block_size = int(meta["pdd_block_size"])
        shift_v = float(meta["pdd_shift_video"])
        shift_a = float(meta["pdd_shift_audio"])

        # Partition check. fl2va and ref2va ship identical key sets, so a
        # mismatched pair loads with zero unmatched keys and renders -- the
        # silent-success failure docs/h3_ref2v_distillation.md records.
        #
        # Compared BY DISTANCE, not by hash. The first version hashed this
        # tensor and fired on the first real render against the correct
        # checkpoint: ComfyUI casts on load, and a cast changes every bit while
        # moving the value a fraction of a percent. The two partitions are 5%
        # apart and a cast is a few tenths of one, so the separation is an
        # order of magnitude and the threshold does not need to be delicate --
        # but an exact hash had no separation at all.
        ref = sd.get("h3_pdd.base_video_out")
        if ref is not None:
            live = dm.final_layer.video_out.weight.detach().to(torch.float32).cpu()
            dist = float((live - ref.to(torch.float32)).norm() / ref.norm())
            if dist > PARTITION_TOLERANCE:
                raise RuntimeError(
                    f"{lora_name} was converted against a different checkpoint "
                    f"partition than the one loaded: final_layer.video_out is "
                    f"{dist:.4f} away from the one it was built against "
                    f"({meta.get('h3_pdd_base', '?')}), tolerance "
                    f"{PARTITION_TOLERANCE}. The fl2va and ref2va partitions "
                    f"sit about 0.05 apart and a dtype cast moves this a few "
                    f"thousandths, so this is a partition mismatch and not a "
                    f"loader artifact. Their key sets are identical, so this "
                    f"would otherwise render without one unmatched key and "
                    f"merely be wrong.")
            logger.info("[h3-pdd] partition check ok: final_layer.video_out is "
                        "%.5f from %s", dist, meta.get("h3_pdd_base", "?"))

        pruned = bool(getattr(dm, "use_adaln_curves", False))

        backbone = {k: v for k, v in sd.items() if k.startswith("diffusion_model.")}
        adaln = {k: v for k, v in sd.items() if k.startswith("h3_pdd.adaln.")}
        n_adaln = len({k.rsplit(".", 1)[0] for k in adaln})

        if not pruned:
            # Unpruned base: the adaln update is an ordinary weight patch, so
            # hand it to comfy.lora with the rest rather than injecting it.
            for k, v in adaln.items():
                i = k.split(".")[3]
                slot = "lora_A" if k.endswith("lora_A") else "lora_B"
                backbone[f"diffusion_model.blocks.{i}.adaln_proj.linear."
                         f"{slot}.weight"] = v

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(backbone, key_map, log_missing=True)
        if not loaded:
            raise RuntimeError(
                f"{lora_name} matched no module on this model. Expected "
                f"ComfyUI generic-LoRA keys under `diffusion_model.`; the "
                f"conversion may predate a checkpoint layout change.")

        m = model.clone()
        m.add_patches(loaded, strength)

        # --- the two runtime surfaces ---------------------------------------
        if pruned:
            grid = sd["h3_pdd.silu_temb_grid"]
            table = dm.adaln_t_table
            for i in range(n_adaln):
                base = m.get_model_object(f"diffusion_model.blocks.{i}.adaln_proj")
                m.add_object_patch(
                    f"diffusion_model.blocks.{i}.adaln_proj.forward",
                    _make_adaln_forward(
                        base,
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_A"],
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_B"],
                        grid, table, strength))
            step_table = table
        else:
            te = dm.time_embedder
            step_table = silu_temb_grid(
                te.proj_in.weight, te.proj_in.bias,
                te.proj_out.weight, te.proj_out.bias,
                rows=int(meta.get("pdd_grid_rows", 1025)), apply_silu=False)

        tracker = _StepTracker(
            step_table,
            block_bounds(shift_v, num_steps, block_size),
            block_bounds(shift_a, num_steps, block_size),
            nfe, lora_name)

        final_layer = m.get_model_object("diffusion_model.final_layer")
        if patch_heads:
            m.add_object_patch(
                "diffusion_model.final_layer.forward",
                _make_final_layer_forward(final_layer.forward, tracker))
        for stream, out_name in (
                (("video", "video_out"), ("audio", "audio_out"))
                if patch_heads else ()):
            live = getattr(final_layer, out_name)
            base_w = live.weight.detach().to(torch.float32).cpu()
            base_b = live.bias.detach().to(torch.float32).cpu()
            fused_w = sd[f"h3_pdd.head.{stream}.weight"]
            fused_b = sd[f"h3_pdd.head.{stream}.bias"]
            # strength interpolates toward the fused head, so 0.0 is the base
            # head exactly rather than a zeroed projection.
            w = base_w[None] + strength * (fused_w - base_w[None])
            b = base_b[None] + strength * (fused_b - base_b[None])
            m.add_object_patch(
                f"diffusion_model.final_layer.{out_name}.forward",
                _make_head_forward(w, b, tracker, stream))

        logger.info(
            "[h3-pdd] %s at strength %.3f: %d backbone modules patched, "
            "%d adaln %s, %s (grid %d/%d -> nfe %d, shifts %g/%g). Base is %s.",
            lora_name, strength, len(loaded), n_adaln,
            "re-injected at run time (pruned base)" if pruned
            else "applied as weight patches (unpruned base)",
            f"{nfe} fused head pairs per stream" if patch_heads
            else "HEADS NOT PATCHED (control arm: the checkpoint's own heads)",
            num_steps, block_size, nfe, shift_v, shift_a,
            "pruned/curve-form" if pruned else "full-width")
        return io.NodeOutput(m)
