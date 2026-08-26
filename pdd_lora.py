"""Apply a converted MiniMax-H3 Parallel Decoding Distillation LoRA.

PDD (arXiv 2607.26004, released for H3 by alibaba-pai) reaches the model on
three surfaces at once, and only one of them is a LoRA in the sense
`LoraLoaderModelOnly` understands. `bench/convert_pdd_lora.py` does everything
that can be done offline; this node does the rest, which is the part that needs
the loaded checkpoint in front of it.

    backbone   208 modules   weight patch, through comfy.lora
    adaln       50 modules   weight patch on an unpruned base; a weight patch
                             in the curve basis on a pruned one when the file
                             carries a bake; runtime injection otherwise
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

**The adaln path is partition-specific, and usually is not an injection at
all.** On a pruned checkpoint the 2688-dim space the adaln delta lives in has
been replaced by an 8-column curve. `bench/convert_pdd_lora.py --pruned`
pre-solves the delta into that same basis, so it becomes an ordinary weight
patch and this node installs no adaln forward patches whatever -- no per-forward
`cdist`, no per-call casts, and strength composes through ComfyUI's own `diff`
path instead of a closure. The delta's time curve fits that basis to about
1e-4, measured per block at conversion time and refused there if it does not.

The runtime injection below is the fallback for a file converted without
`--pruned`, or one whose bake was solved against a different table. It is
partition-specific for the same reason the bake is: the fl2va and ref2va time
curves differ by 7.8% relative, which is why our grid is derived from the same
checkpoint that supplies the partition fingerprint rather than bundled once.
The turbo pack ships a single fl2va grid; reusing it for ref2va would feed the
injection a 7.8%-wrong input and render without complaint.

## Strength, and how it composes

Every surface scales through the path that owns it, so nothing here reimplements
weighting that ComfyUI already does:

  backbone       `add_patches(..., strength)`  -- native
  adaln, baked   `diff` / `diff_b` in the same dict -- native
  adaln, injected  captured in the closure and applied to the delta
  heads          `base + strength * (fused - base)`

`strength=0.0` installs nothing on any of them, so it is exactly the base
model. That is deliberately a different control from the one
`workflows/h3_config.py` recommends for plain LoRAs, where 0.0 short-circuits
the dequantise/add/requantise round trip and 0.01 is the like-for-like
baseline -- use 0.01 here to price the backbone's numerical cost, 0.0 to get
the base model back.

## What the timestep keying buys downstream

Deriving the step from `t_emb` rather than counting forwards is not only
robustness against an extra evaluation. It is what lets this node sit in the
graphs the repo already ships:

  * **the step cache** skips forwards on reused steps. A counter would drift by
    exactly the number of skips and never recover; a time lookup simply reads
    whatever step the next real forward is at.
  * **`split_at` two-pass sampling** runs part of the trajectory on one model
    and part on another. A pass covering only the tail still selects the heads
    for the tail, because the heads are indexed by time and not by how many
    calls this particular model object has seen.
  * **CFG**, if a graph ever runs it, doubles the forwards per step without
    moving time at all.

None of those needed special handling. They are the reason not to port the
vendor's forward-hook counter, stated as capabilities rather than as hazards.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io

try:                                     # loaded as a package by ComfyUI
    from .pdd_math import block_bounds, fuse_heads, silu_temb_grid
except ImportError:                      # loaded as a bare module by a script
    from pdd_math import block_bounds, fuse_heads, silu_temb_grid

logger = logging.getLogger(__name__)

# Dtypes, and why each is what it is. They are not interchangeable here.
#
#   fused heads, baked adaln diffs   fp32 on disk. `final_layer`'s two output
#       projections are the checkpoint's fp32 island -- `comfy/ldm/minimax/
#       model.py` builds them with an explicit `dtype=torch.float32` while the
#       rest of the block is model dtype -- and the vendor's own fusion loses
#       ~1.7e-3 by casting its plan to bf16 before the einsum. Storing fp32
#       keeps the precision the island exists for.
#   backbone / adaln LoRA pairs      bf16, as published. `calculate_weight`
#       casts them itself, so nothing is gained by widening on disk.
#   curve table and time grid        fp32, and compared in fp32. The step
#       lookup is a `cdist` against 1025 rows spanning [0, 1]; in bf16 the row
#       spacing would be at the edge of representable and the recovered `t`
#       would quantise further than the 1e-3 the boundary snap already has to
#       absorb.

#: Warn once per patch when the timestep embedding sits this far from every
#: block boundary, in EMBEDDING distance rather than in `t`. Selection no
#: longer needs a tolerance -- the nearest boundary is the answer -- so this
#: guards one thing only: whether the render is on the schedule the heads were
#: fused for at all. On schedule the distance is ~0; a step count the file was
#: not fused for puts it orders of magnitude above this.
BOUNDARY_TOLERANCE = 1e-2

#: Relative-Frobenius distance allowed between the loaded checkpoint's
#: `final_layer.video_out` and the one the LoRA was converted against. The
#: fl2va and ref2va partitions sit ~0.05 apart (measured 2026-08-26 across the
#: six H3 checkpoints on this box) and a dtype cast on load moves it a few
#: thousandths, so anything in between is unambiguous. Set an order of
#: magnitude below the partition gap and an order above a cast.
PARTITION_TOLERANCE = 0.015

#: Relative distance allowed between the loaded checkpoint's `adaln_t_table`
#: and the one a bake was solved against. A partition's `int8_convrot` and
#: `fp8_scaled` builds carry byte-identical tables (verified 2026-08-26), so
#: this only has to separate a dtype cast from a different partition's basis.
#: Measured the same day: a bf16 cast of a table is 0.00164 away, and the fl2va
#: and ref2va tables are 0.01835 apart. This sits between them -- three times
#: above the cast, three and a half below the gap.
#:
#: It was 1e-3 for about an hour, which is BELOW the cast. That would have
#: rejected every correct bake the moment the loader cast the buffer and fallen
#: back to the injection -- slower, and silent, because the fallback is correct.
#: The same shape as hashing `video_out`: a tolerance under the noise it has to
#: tolerate.
#:
#: A mismatch FALLS BACK rather than raising. The injection is correct on any
#: pruned base, so refusing the render would be worse than taking the slow path.
#: What it must not do is bake anyway: a bake solved against the wrong basis is
#: 0.0205 wrong at runtime against 0.0001 for the right one, and nothing
#: downstream would say so.
TABLE_TOLERANCE = 5e-3


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
    """Which fused head each stream wants, from the embedding it was called with.

    Matches `t_emb` against the `nfe + 1` BLOCK BOUNDARY embeddings, not against
    the 1025-row curve table. That is the whole selector: the nearest boundary
    is the block, directly.

    The earlier version recovered a `t` by nearest row of the full table and
    then bucketed it, which is how a `t` sitting exactly ON a boundary came back
    a fraction below it and selected the previous block -- wrong at two of eight
    steps, silent, and it took a deliberate drive against real inputs to find.
    It was fixed by snapping, i.e. by a tolerance that then had to be justified
    against the table's own quantisation.

    Matching the boundaries directly deletes that problem rather than guarding
    it. There is no intermediate `t`, so nothing to quantise; there are exactly
    `nfe` answers, so nothing to fall between; and the selector needs no
    tolerance at all. The tolerance that remains is only for the WARNING, which
    is a different question -- is this render even on the schedule the heads
    were fused for.

    Costs `nfe + 1` distances per stream per forward against 1025 before, which
    is not why: at ~70 s a step neither is measurable. It is smaller and it
    cannot be wrong in the way the other one was.
    """

    def __init__(self, boundary_emb_v, boundary_emb_a, bounds_v, bounds_a,
                 nfe, label):
        self.emb_v = boundary_emb_v          # [nfe+1, D]
        self.emb_a = boundary_emb_a
        self.bounds_v = bounds_v
        self.bounds_a = bounds_a
        self.nfe = nfe
        self.label = label
        self.video = 0
        self.audio = 0
        self.warned = False

    def _pick(self, t_emb, row, table):
        e = t_emb[_row_index(row)].detach().float().reshape(1, -1)
        d = torch.cdist(e, table.to(e.device, torch.float32))[0]
        j = int(d.argmin())
        return min(j, self.nfe - 1), float(d[j])

    def update(self, t_emb, video_seg, audio_seg) -> None:
        self.video, dv = self._pick(t_emb, video_seg[2], self.emb_v)
        self.audio, da = self._pick(t_emb, audio_seg[2], self.emb_a)
        if not self.warned and max(dv, da) > BOUNDARY_TOLERANCE:
            self.warned = True
            logger.warning(
                "[h3-pdd] %s: this render is NOT on the schedule the heads "
                "were fused for. The timestep embedding sits %.4f (video) and "
                "%.4f (audio) from the nearest block boundary, tolerance "
                "%.4f. The heads assume %d evaluations at the shifts recorded "
                "in the file; set the sampler's step count to %d, or "
                "reconvert. Sampling continues on the nearest boundary.",
                self.label, dv, da, BOUNDARY_TOLERANCE, self.nfe, self.nfe)


def boundary_embeddings(bounds, table, time_embedder=None, rows=1025):
    """The `t_emb` the model will produce AT each block boundary.

    Built the same two ways the model builds `t_emb`, chosen by the same
    observable the rest of this node branches on -- a curve table when the
    checkpoint is pruned, the time embedder when it is not -- so the thing being
    matched against is constructed by the model's own arithmetic rather than
    approximated.
    """
    out = []
    for t in bounds.tolist():
        if time_embedder is None:
            pos = min(max(float(t), 0.0), 1.0) * (table.shape[0] - 1)
            i0 = min(int(pos), table.shape[0] - 2)
            out.append(torch.lerp(table[i0].float(), table[i0 + 1].float(),
                                  pos - i0))
        else:
            out.append(table[min(int(round(float(t) * (rows - 1))), rows - 1)])
    return torch.stack(out)


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
                        "Scales all three mechanisms together, and each one "
                        "through the path that owns it: the backbone and the "
                        "adaln update are ordinary ComfyUI weight patches, so "
                        "they take strength natively, and each fused head is "
                        "base + strength * (fused - base) against the "
                        "checkpoint's own head. 1.0 is the vendor's default "
                        "and what their published clips were rendered at.\n\n"
                        "0.0 installs nothing at all and is exactly the base "
                        "model, heads included. Note that is a DIFFERENT "
                        "control from the one h3_config recommends for plain "
                        "LoRAs: there 0.0 short-circuits the "
                        "dequantise/add/requantise round trip and 0.01 is the "
                        "like-for-like baseline. Use 0.01 to isolate the "
                        "backbone's numerical cost, 0.0 to get the base model."
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
                # APPENDED. See the note on patch_heads.
                io.Int.Input(
                    "nfe", default=0, min=0, max=64, optional=True,
                    tooltip=(
                        "Transformer evaluations, i.e. the sampler's step "
                        "count. 0 means the count the file was converted for.\n\n"
                        "The published grid is 32 points, so ANY divisor is a "
                        "legal arm from the same weights: 8 (block 4) is what "
                        "the file records, and 4 (block 8) is the other count "
                        "the vendor's README reports rendering at. Every "
                        "divisor lands exactly on the plain shifted schedule "
                        "for its own step count, so changing this changes the "
                        "sampler's steps and nothing else -- not the shift, not "
                        "the scheduler.\n\n"
                        "The heads are fused here, at load, for whichever "
                        "count you ask. Set BasicScheduler to the same number: "
                        "a mismatch is what the boundary warning reports."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0,
                patch_heads=True, nfe=0) -> io.NodeOutput:
        import comfy.lora
        import comfy.utils
        import folder_paths

        # Coerced, not assumed. The schema declares FLOAT and BOOLEAN and the
        # generator writes 1.0 / True, but an API prompt is JSON a person can
        # hand-write, and `strength` reaches tensor arithmetic and an equality
        # against 0.0 that decides whether anything is installed at all.
        strength = float(strength)
        patch_heads = bool(patch_heads)

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

        num_steps = int(meta["pdd_num_steps"])
        nfe = int(nfe) or int(meta["pdd_nfe"])
        if nfe < 1 or num_steps % nfe:
            raise RuntimeError(
                f"nfe={nfe} does not divide the file's {num_steps}-point grid. "
                f"A block has to tile the grid exactly or the fused heads "
                f"decode intervals that are not the ones being stepped over. "
                f"Legal here: {sorted(n for n in range(1, num_steps + 1) if num_steps % n == 0)}.")
        block_size = num_steps // nfe
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

        # Which of the three adaln paths this checkpoint gets. Branching on
        # `use_adaln_curves` and on the live table -- observables of the loaded
        # model -- rather than on anything in the filename.
        baked = None
        if pruned and "h3_pdd.adaln_table" in sd:
            live_table = dm.adaln_t_table.detach().to(torch.float32).cpu()
            ref_table = sd["h3_pdd.adaln_table"].to(torch.float32)
            if live_table.shape == ref_table.shape:
                d = float((live_table - ref_table).norm() / ref_table.norm())
                if d <= TABLE_TOLERANCE:
                    baked = d
                else:
                    logger.warning(
                        "[h3-pdd] %s was baked against a different curve table "
                        "(%.5f away, tolerance %g); falling back to the runtime "
                        "adaln injection, which is correct on any pruned base.",
                        lora_name, d, TABLE_TOLERANCE)

        if not pruned:
            # Unpruned base: the adaln update is an ordinary weight patch in the
            # 2688-dim time space, so hand it to comfy.lora with the rest.
            # The baked tensors are NOT offered here -- they are 8 columns wide
            # and `calculate_weight` only WARNS on a shape mismatch before
            # skipping, so a wrong-width diff would drop 50 modules with a log
            # line rather than an error.
            for k, v in adaln.items():
                i = k.split(".")[3]
                slot = "lora_A" if k.endswith("lora_A") else "lora_B"
                backbone[f"diffusion_model.blocks.{i}.adaln_proj.linear."
                         f"{slot}.weight"] = v
        elif baked is not None:
            # Pruned, with a bake solved against this checkpoint's own basis.
            # `diff` / `diff_b` are comfy.lora's own patch kinds and take
            # strength like any other patch.
            for i in range(n_adaln):
                base_key = f"diffusion_model.blocks.{i}.adaln_proj.linear"
                backbone[f"{base_key}.diff"] = \
                    sd[f"h3_pdd.adaln_baked.blocks.{i}.diff"]
                backbone[f"{base_key}.diff_b"] = \
                    sd[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"]

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(backbone, key_map, log_missing=True)
        if not loaded:
            raise RuntimeError(
                f"{lora_name} matched no module on this model. Expected "
                f"ComfyUI generic-LoRA keys under `diffusion_model.`; the "
                f"conversion may predate a checkpoint layout change.")

        m = model.clone()
        m.add_patches(loaded, strength)

        # --- the runtime surfaces -------------------------------------------
        # The step tracker needs a table in whatever space `t_emb` lives in,
        # and that follows `pruned` alone -- NOT whether the adaln update was
        # baked. Those two were briefly conflated while adding the bake, which
        # left `step_table` unset on the pruned-and-baked path, i.e. on the
        # default configuration.
        if pruned:
            table = dm.adaln_t_table
            step_table = table
        else:
            te = dm.time_embedder
            step_table = silu_temb_grid(
                te.proj_in.weight, te.proj_in.bias,
                te.proj_out.weight, te.proj_out.bias,
                rows=int(meta.get("pdd_grid_rows", 1025)), apply_silu=False)

        if pruned and baked is None:
            # Fallback only: a file with no bake, or one solved against another
            # table. Installs 50 forward patches the baked path does not need.
            grid = sd["h3_pdd.silu_temb_grid"]
            for i in range(n_adaln):
                base = m.get_model_object(f"diffusion_model.blocks.{i}.adaln_proj")
                m.add_object_patch(
                    f"diffusion_model.blocks.{i}.adaln_proj.forward",
                    _make_adaln_forward(
                        base,
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_A"],
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_B"],
                        grid, table, strength))

        bounds_v = block_bounds(shift_v, num_steps, block_size)
        bounds_a = block_bounds(shift_a, num_steps, block_size)
        # Built once at load, from the model's own arithmetic, for the nfe this
        # render will use. The two streams run different shifts, so they get
        # different boundary times and therefore different embeddings.
        tracker = _StepTracker(
            boundary_embeddings(bounds_v, step_table,
                                None if pruned else dm.time_embedder),
            boundary_embeddings(bounds_a, step_table,
                                None if pruned else dm.time_embedder),
            bounds_v, bounds_a, nfe, lora_name)

        final_layer = m.get_model_object("diffusion_model.final_layer")
        # strength 0 installs NOTHING on the head path. Interpolating to the
        # base head would be arithmetically identical, but it would still route
        # the projection through this module's `F.linear` instead of the
        # checkpoint's own `operations.Linear`, which owns the casting and
        # offload handling. "Exactly the base model" has to mean the base
        # model's own code, or the claim is only nearly true and the control
        # arm is only nearly a control.
        if patch_heads and strength != 0.0:
            m.add_object_patch(
                "diffusion_model.final_layer.forward",
                _make_final_layer_forward(final_layer.forward, tracker))
        for stream, out_name in (
                (("video", "video_out"), ("audio", "audio_out"))
                if (patch_heads and strength != 0.0) else ()):
            live = getattr(final_layer, out_name)
            base_w = live.weight.detach().to(torch.float32).cpu()
            base_b = live.bias.detach().to(torch.float32).cpu()
            # Fused HERE, from the published per-interval bank, for whatever
            # nfe was asked for. The paper's section 3.1 is explicit that this
            # belongs at inference setup rather than inside the forward:
            # "we only need to hold one fused linear layer per block in
            # memory". Doing it at load also means one file serves every legal
            # step count, where precomputing pinned it into the artifact.
            bank_w = sd.get(f"h3_pdd.bank.{stream}.weight")
            if bank_w is None:
                raise RuntimeError(
                    f"{lora_name} carries no per-interval head bank. It was "
                    f"converted before the bank was stored; reconvert with "
                    f"bench/convert_pdd_lora.py.")
            shift = shift_v if stream == "video" else shift_a
            fused_w = fuse_heads(bank_w, shift, num_steps, block_size)
            fused_b = fuse_heads(sd[f"h3_pdd.bank.{stream}.bias"], shift,
                                 num_steps, block_size)
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
            ("baked into the curve basis, applied as weight patches"
             if baked is not None else
             "re-injected at run time (pruned base, no bake in this file)"
             if pruned else
             "applied as weight patches (unpruned base)"),
            f"{nfe} fused head pairs per stream" if patch_heads
            else "HEADS NOT PATCHED (control arm: the checkpoint's own heads)",
            num_steps, block_size, nfe, shift_v, shift_a,
            "pruned/curve-form" if pruned else "full-width")
        return io.NodeOutput(m)
