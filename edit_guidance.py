"""Steer a render with an edited copy of its own prompt, with no control vector.

Two nodes. `MiniMaxH3PromptEdit` applies `old wording -> new wording` lines to a prompt and refuses
anything it cannot place exactly. `MiniMaxH3EditGuider` takes the original and the edited conditioning and,
each step, moves the model's prediction from the original toward the edited by a scale:

    prediction = original + scale * (edited - original)

A scale of 0 is the original prompt, 1 is the edited prompt, above 1 exaggerates the edit and between 0 and
1 gives part of it. Before `first_step` only the original prompt is run, in one model pass, so the first
steps can settle the layout before the edit comes in.

What it estimates: what this edit means for this shot at each step, computed by the model itself. It needs
no pair file and no vector. What it costs: two model passes for every guided step.

Its limit: the two prompts usually differ in length, and the text length sets where the video and audio
rows start in position space (`comfy/ldm/minimax/model.py::PackedLayout`), so the two passes see those rows
at slightly different positions. `align_positions` gives both passes the longer prompt's origin. That
alignment is untested on renders, so it is off unless asked for.

The guider is a subclass of core's `comfy.samplers.CFGGuider` with its own `predict_noise`, the way core's
`Guider_DualCFG` is made. Nothing in core is patched.

Provenance of the defaults: scale 1.5 and the two test scales came from the owner's look at
`bench/edit_guidance_arms.py` renders on 2026-11-09 (two shots, two seeds), where 2.5 added color beyond the
edit; they are starting values, not measured optima.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import comfy.nested_tensor
import comfy.samplers
import torch
from comfy_api.latest import io

try:
    from . import h3_prompt_edit as _pe
    from . import h3_rules as _rules
except ImportError:  # run from the repo root by a check
    import h3_prompt_edit as _pe
    import h3_rules as _rules

logger = logging.getLogger(__name__)

#: The node's defaults and their reasons are in `h3_rules.EDIT_GUIDANCE_DEFAULTS`, one copy.
DEFAULT_SCALE = _rules.EDIT_GUIDANCE_DEFAULTS["scale"]
DEFAULT_AUDIO_SCALE = _rules.EDIT_GUIDANCE_DEFAULTS["audio_scale"]
EDIT_GUIDANCE_STREAMS = tuple(_rules.EDIT_GUIDANCE_STREAMS)
MAX_SCALE = 4.0
PASS_KEY = "h3_edit_pass"

#: The edit grammar and `apply_edits` live in `h3_prompt_edit.py`, which loads no model library, so the
#: static graph walkers read an edit node's text with the same code this node runs.
apply_edits = _pe.apply_edits


def _one(conditioning: Any, label: str) -> dict:
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError(f"{label} conditioning must be a single prompt (one entry); "
                         "prompt lists and scheduled conditioning are not supported here")
    meta = conditioning[0][1]
    if not isinstance(meta, dict) or meta.get("minimax_token_tags") is None:
        raise ValueError(f"{label} conditioning carries no H3 token tags; wire it from an H3 conditioning node")
    return meta


def _count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else (0 if value is None else 1)


def check_same_but_text(original: Any, edited: Any) -> tuple[int, int]:
    """Refuse unless the two conditionings carry H3 token tags and share `minimax_refs`.

    `minimax_keyframes` may differ so `MiniMaxH3GuideLatent` can anchor visual or audio
    guide islands on `edited` alone against an unguided `original` pass (`EditGuider._prepare`
    marks `PASS_KEY` on both payloads so the two passes build their own `PackedLayout`).
    Returns `(len_original, len_edited)` in text tokens.
    """
    a, b = _one(original, "original"), _one(edited, "edited")
    if _count(a.get("minimax_refs")) != _count(b.get("minimax_refs")):
        raise ValueError(
            "the original and edited conditioning differ in minimax_refs; both must be made with "
            "the same references"
        )
    return int(original[0][0].shape[1]), int(edited[0][0].shape[1])


def step_index(model_options: dict, timestep: Any) -> int | None:
    """The index of the current step in the schedule the sampler is running, or None when it is not known."""
    opts = (model_options or {}).get("transformer_options", {})
    sched = opts.get("sample_sigmas")
    if sched is None or timestep is None:
        return None
    sched_t = torch.as_tensor(sched, dtype=torch.float64).flatten().cpu()
    if int(sched_t.numel()) < 2:
        return None
    cur = float(torch.as_tensor(timestep).flatten()[0])
    return int((sched_t[:-1] - cur).abs().argmin())


class EditGuider(comfy.samplers.CFGGuider):
    """Per-stream `original + scale * (edited - original)`, one pass before `first_step` and one `[edited]` pass after `last_step` (or `[original]` when `original` and `edited` share identical text and `edited` alone carries guides)."""

    def set_conds(self, original, edited):
        self.inner_set_conds({"positive": edited, "negative": original})
        same_text = (
            isinstance(original, list)
            and len(original) == 1
            and isinstance(edited, list)
            and len(edited) == 1
            and torch.is_tensor(original[0][0])
            and torch.is_tensor(edited[0][0])
            and original[0][0].shape == edited[0][0].shape
            and bool(torch.equal(original[0][0], edited[0][0]))
        )
        kf_orig = _count(original[0][1].get("minimax_keyframes")) if same_text else 0
        kf_edit = _count(edited[0][1].get("minimax_keyframes")) if same_text else 0
        self._release_to_original_after = bool(same_text and kf_orig == 0 and kf_edit > 0)

    def set_edit(
        self,
        scale: float,
        first_step: int,
        align_positions: bool,
        last_step: int = 1000,
        streams: str = "both",
        audio_scale: float = 1.0,
    ):
        f_step = int(first_step)
        l_step = int(last_step)
        if l_step < f_step:
            raise ValueError(f"last_step ({l_step}) must be >= first_step ({f_step})")
        if streams not in EDIT_GUIDANCE_STREAMS:
            raise ValueError(f"streams must be one of {EDIT_GUIDANCE_STREAMS}; got {streams!r}")
        if streams == "both":
            self.edit_scale = float(scale)
            self.edit_audio_scale = float(scale)
        elif streams == "audio_only":
            self.edit_scale = 0.0
            self.edit_audio_scale = float(scale)
        elif streams == "video_only":
            self.edit_scale = float(scale)
            self.edit_audio_scale = 0.0
        else:  # "independent"
            self.edit_scale = float(scale)
            self.edit_audio_scale = float(audio_scale)
        self.streams = streams
        self.first_step = f_step
        self.last_step = l_step
        self.align_positions = bool(align_positions)
        self._prepared = False

    def _prepare(self) -> None:
        """Mark each pass, so the two are never batched together and a wrapper can tell them apart, and
        (when asked) give both passes the longer prompt's position origin."""
        if self._prepared:
            return
        entries = {name: self.conds.get(name) or [] for name in ("negative", "positive")}
        lengths = {}
        for name, conds in entries.items():
            for cond in conds:
                wrapped = cond.get("model_conds", {}).get("minimax_payload")
                if wrapped is None:
                    continue
                payload = dict(wrapped.cond)
                payload[PASS_KEY] = "original" if name == "negative" else "edited"
                layout = payload.get("layout")
                if layout is not None:
                    lengths[name] = int(layout.signature[0])
                cond["model_conds"] = dict(cond["model_conds"])
                cond["model_conds"]["minimax_payload"] = wrapped.__class__(payload)
        if self.align_positions and len(set(lengths.values())) > 1:
            longest = max(lengths.values())
            for name, conds in entries.items():
                for cond in conds:
                    wrapped = cond["model_conds"].get("minimax_payload")
                    layout = None if wrapped is None else wrapped.cond.get("layout")
                    if layout is None or int(layout.signature[0]) == longest:
                        continue
                    text_len = int(layout.signature[0])
                    moved = copy.copy(layout)
                    moved.position_ids = layout.position_ids.clone()
                    moved.position_ids[text_len:, 0] += float(longest - text_len)
                    payload = dict(wrapped.cond)
                    payload["layout"] = moved
                    cond["model_conds"]["minimax_payload"] = wrapped.__class__(payload)
            logger.info("[h3] edit guidance: media rows of the shorter prompt moved to the longer prompt's "
                        "origin (text lengths %s)", lengths)
        self._prepared = True

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        self._prepare()
        original, edited = self.conds.get("negative"), self.conds.get("positive")
        step = step_index(model_options, timestep)
        before = step is not None and step < self.first_step
        after = step is not None and step > self.last_step
        if before or (after and getattr(self, "_release_to_original_after", False)):
            eff_v, eff_a = 0.0, 0.0
        elif after:
            eff_v = 0.0 if self.edit_scale == 0.0 else 1.0
            eff_a = 0.0 if self.edit_audio_scale == 0.0 else 1.0
        else:
            eff_v, eff_a = self.edit_scale, self.edit_audio_scale
        has_audio = bool(getattr(x, "is_nested", False)) and len(x.unbind()) == 2
        if not has_audio:
            eff_a = eff_v
        if eff_v == 0.0 and eff_a == 0.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [original], x, timestep, model_options)[0]
        if eff_v == 1.0 and eff_a == 1.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [edited], x, timestep, model_options)[0]
        out = comfy.samplers.calc_cond_batch(self.inner_model, [original, edited], x, timestep, model_options)
        if eff_v != eff_a and getattr(out[0], "is_nested", False) and len(out[0].unbind()) == 2:
            v0, a0 = out[0].unbind()
            v1, a1 = out[1].unbind()
            v_out = v0 if eff_v == 0.0 else (v1 if eff_v == 1.0 else v0 + (v1 - v0) * eff_v)
            a_out = a0 if eff_a == 0.0 else (a1 if eff_a == 1.0 else a0 + (a1 - a0) * eff_a)
            return comfy.nested_tensor.NestedTensor((v_out, a_out))
        return out[0] + (out[1] - out[0]) * eff_v


class MiniMaxH3PromptEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptEdit",
            display_name="MiniMax H3 Prompt Edit",
            category="conditioning/minimax",
            description="Apply 'old wording -> new wording' lines to a prompt, refusing anything it cannot "
                        "place exactly. Wire the edited prompt into a second conditioning node.",
            inputs=[
                # The shipped graphs write the prompt once, on a text node wired to this input and to
                # the first conditioning node (`workflows/build_workflows.py::_wire_edit_guidance`). It
                # stays a box so the node also works alone. Named `original`, as on the guider, and not
                # `prompt`: the conditioning node is what hands a prompt to the encoder and what every
                # grader reads (`h3_config.PROMPT_INPUTS`).
                io.String.Input("original", multiline=True, default="",
                                tooltip="The prompt as written. Wire it from the same text node that feeds the "
                                        "first conditioning node, so the two cannot differ."),
                io.String.Input("edits", multiline=True, default="",
                                tooltip="One edit a line: old wording -> new wording. The old wording must occur "
                                        "exactly once in the prompt (or inside the named section). Start a line with "
                                        "'subject_definitions:', 'summary:', 'retention_analysis:', 'description:', "
                                        "'soundscape:', or 'music:' to edit that section. Dialogue, tags and headers "
                                        "cannot be edited inside 'old -> new'."),
            ],
            outputs=[io.String.Output(display_name="edited_prompt"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, original, edits) -> io.NodeOutput:
        edited, records = apply_edits(original, edits)
        lines = [
            f"{'soundscape' if r['soundscape'] else (r.get('section') or 'description')}: "
            f"{r['old_words']} word(s) -> {r['new_words']} word(s): {r['old']!r} -> {r['new']!r}"
            for r in records
        ]
        return io.NodeOutput(edited, "\n".join(lines))


class MiniMaxH3EditGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EditGuider",
            display_name="MiniMax H3 Edit Guider",
            category="sampling/custom_sampling/guiders",
            description="Steer with an edited copy of the prompt: each step, move the prediction from the "
                        "original prompt toward the edited one by a scale (on both streams, video only, audio "
                        "only, or independent video and audio scales). Use in place of BasicGuider.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("original", tooltip="Conditioning of the prompt as written."),
                io.Conditioning.Input("edited", tooltip="Conditioning of the edited prompt, made with the same "
                                                        "encoder, references and canvas."),
                io.Float.Input("scale", default=DEFAULT_SCALE, min=0.0, max=MAX_SCALE, step=0.05,
                               tooltip="0 is the original prompt, 1 is the edited prompt, above 1 exaggerates "
                                       "the edit. Above about 2.5 the picture gains color the edit did not ask for."),
                io.Int.Input("first_step", default=_rules.EDIT_GUIDANCE_DEFAULTS["first_step"], min=0, max=1000,
                             tooltip="The first sampling step that uses the edit. Earlier steps run the original "
                                     "prompt alone, which keeps its framing and pose."),
                io.Boolean.Input("align_positions", default=_rules.EDIT_GUIDANCE_DEFAULTS["align_positions"],
                                 optional=True, advanced=True,
                                 tooltip="Give both prompts the same starting position for the video and audio "
                                         "rows when their lengths differ. Untested on renders."),
                io.Int.Input("last_step", default=_rules.EDIT_GUIDANCE_DEFAULTS["last_step"], min=0, max=1000,
                             optional=True,
                             tooltip="The last sampling step that runs 2-pass edit guidance (default 1000 = all steps). "
                                     "Steps after last_step run 1-pass [edited] (scale = 1.0) on active streams while "
                                     "keeping any 0-locked stream on [original], halving late-step DiT cost."),
                # APPENDED 2026-11-12: stream targeting and independent audio scale (0 extra DiT passes).
                io.Combo.Input("streams", options=list(EDIT_GUIDANCE_STREAMS),
                               default=_rules.EDIT_GUIDANCE_DEFAULTS["streams"], optional=True,
                               tooltip="Which latent streams receive edit guidance: 'both' steers video and audio "
                                       "at scale; 'audio_only' locks video to original (0.0) and steers audio at "
                                       "scale; 'video_only' steers video at scale and locks audio to original (0.0); "
                                       "'independent' steers video at scale and audio at audio_scale."),
                io.Float.Input("audio_scale", default=DEFAULT_AUDIO_SCALE, min=0.0, max=MAX_SCALE, step=0.05,
                               optional=True,
                               tooltip="Audio edit scale when streams = 'independent' (ignored in other modes). "
                                       "Keep at 1.0 when pushing visual scale >= 1.65 to adopt edited audio without "
                                       "> 0 dBFS transient clipping, or raise above 1.0 to punch a quiet visual scene's "
                                       "soundscape edit."),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        original,
        edited,
        scale,
        first_step,
        align_positions=False,
        last_step=_rules.EDIT_GUIDANCE_DEFAULTS["last_step"],
        streams=_rules.EDIT_GUIDANCE_DEFAULTS["streams"],
        audio_scale=DEFAULT_AUDIO_SCALE,
    ) -> io.NodeOutput:
        len_a, len_b = check_same_but_text(original, edited)
        kf_a = _count(original[0][1].get("minimax_keyframes"))
        kf_b = _count(edited[0][1].get("minimax_keyframes"))
        guider = EditGuider(model)
        guider.set_conds(original, edited)
        guider.set_edit(
            scale,
            first_step,
            align_positions,
            last_step=last_step,
            streams=streams,
            audio_scale=audio_scale,
        )
        kf_note = f"; keyframes {kf_a} (original) and {kf_b} (edited)" if (kf_a or kf_b) else ""
        logger.info(
            "[h3] edit guidance: streams=%s (video=%g, audio=%g) across steps %d..%d; "
            "text length %d (original) and %d (edited) tokens%s",
            streams,
            guider.edit_scale,
            guider.edit_audio_scale,
            int(first_step),
            int(last_step),
            len_a,
            len_b,
            kf_note,
        )
        return io.NodeOutput(guider)
