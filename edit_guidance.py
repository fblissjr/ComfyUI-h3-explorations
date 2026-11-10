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
import re
from typing import Any

import comfy.samplers
import torch
from comfy_api.latest import io

try:
    from . import control_vectors as _cv
    from . import h3_pair_compiler as _pc
except ImportError:  # run from the repo root by a check
    import control_vectors as _cv
    import h3_pair_compiler as _pc

logger = logging.getLogger(__name__)

#: Reasoned: the owner's 2026-11-09 look found 1.5 a touch stronger than the rewritten prompt and 2.5
#: adding color the edit did not ask for.
DEFAULT_SCALE = 1.5
MAX_SCALE = 4.0
PASS_KEY = "h3_edit_pass"

_DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.S)
_TAG_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>|\(S\d+(?:,\s*S\d+)*\)|</?d>|<\|[a-z_]+\|>")


def _inside_dialogue(text: str, start: int, end: int) -> bool:
    return any(m.start() < end and start < m.end() for m in _DIALOGUE_RE.finditer(text))


def apply_edits(prompt: str, edits: str) -> tuple[str, list[dict[str, Any]]]:
    """Apply every `old -> new` line of `edits` to `prompt`. Returns the edited prompt and one record a line.

    Refuses, with the line named: a line with no arrow or an empty side; old wording that is absent or
    occurs more than once; an edit that touches dialogue, a tag, a speaker id, a section header or a shot
    header; a `soundscape:` line whose old wording is not in the soundscape section."""
    text = str(prompt or "")
    if not text.strip():
        raise ValueError("the prompt is empty")
    lines = _pc._split_clause_statements(edits)
    if not lines:
        raise ValueError("no edit given; write one 'old wording -> new wording' per line")
    records: list[dict[str, Any]] = []
    for raw in lines:
        sound = _pc._SOUNDSCAPE_PREFIX_RE.match(raw)
        body = sound.group(1) if sound else raw
        old, new = _pc._split_pole_pair(body)
        if not old or not new:
            raise ValueError(f"edit {raw!r}: write it as 'old wording -> new wording'")
        if old == new:
            raise ValueError(f"edit {raw!r}: both sides are the same")
        for side, wording in (("old", old), ("new", new)):
            if _TAG_RE.search(wording) or _cv.HEADER_TOKEN_RE.search(wording):
                raise ValueError(f"edit {raw!r}: the {side} wording contains a tag, a speaker id or a header; "
                                 "edit only the prose around them")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"edit {raw!r}: the old wording occurs {count} time(s) in the prompt; it must "
                             "occur exactly once (add the words that follow it to make it unique)")
        at = text.index(old)
        if _inside_dialogue(text, at, at + len(old)):
            raise ValueError(f"edit {raw!r}: the old wording is inside <d>...</d>; dialogue is not edited here")
        if sound:
            marker = text.find("overall_soundscape:")
            music = text.find("non_diegetic_music:")
            if marker < 0 or at < marker or (0 <= music < at):
                raise ValueError(f"edit {raw!r}: a 'soundscape:' edit must replace wording inside the "
                                 "overall_soundscape section")
        text = text[:at] + new + text[at + len(old):]
        records.append({"old": old, "new": new, "soundscape": bool(sound),
                        "old_words": len(old.split()), "new_words": len(new.split())})
    return text, records


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
    """Refuse unless the two conditionings differ in text only. Returns their text lengths in tokens."""
    a, b = _one(original, "original"), _one(edited, "edited")
    for key in ("minimax_refs", "minimax_keyframes"):
        if _count(a.get(key)) != _count(b.get(key)):
            raise ValueError(f"the original and edited conditioning differ in {key}; both must be made with "
                             "the same references and keyframes")
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
    """`original + scale * (edited - original)`, one pass before `first_step`."""

    def set_conds(self, original, edited):
        self.inner_set_conds({"positive": edited, "negative": original})

    def set_edit(self, scale: float, first_step: int, align_positions: bool):
        self.edit_scale = float(scale)
        self.first_step = int(first_step)
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
        if before or self.edit_scale == 0.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [original], x, timestep, model_options)[0]
        if self.edit_scale == 1.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [edited], x, timestep, model_options)[0]
        out = comfy.samplers.calc_cond_batch(self.inner_model, [original, edited], x, timestep, model_options)
        return out[0] + (out[1] - out[0]) * self.edit_scale


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
                io.String.Input("prompt", multiline=True, force_input=True,
                                tooltip="The prompt as written. Wire the same text the conditioning node gets."),
                io.String.Input("edits", multiline=True, default="",
                                tooltip="One edit a line: old wording -> new wording. The old wording must occur "
                                        "exactly once in the prompt. Start a line with 'soundscape:' to edit the "
                                        "soundscape section. Dialogue, tags and headers cannot be edited."),
            ],
            outputs=[io.String.Output(display_name="edited_prompt"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, prompt, edits) -> io.NodeOutput:
        edited, records = apply_edits(prompt, edits)
        lines = [f"{'soundscape' if r['soundscape'] else 'description'}: {r['old_words']} word(s) -> "
                 f"{r['new_words']} word(s): {r['old']!r} -> {r['new']!r}" for r in records]
        return io.NodeOutput(edited, "\n".join(lines))


class MiniMaxH3EditGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EditGuider",
            display_name="MiniMax H3 Edit Guider",
            category="sampling/custom_sampling/guiders",
            description="Steer with an edited copy of the prompt: each step, move the prediction from the "
                        "original prompt toward the edited one by a scale. Use in place of BasicGuider.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("original", tooltip="Conditioning of the prompt as written."),
                io.Conditioning.Input("edited", tooltip="Conditioning of the edited prompt, made with the same "
                                                        "encoder, references and canvas."),
                io.Float.Input("scale", default=DEFAULT_SCALE, min=0.0, max=MAX_SCALE, step=0.05,
                               tooltip="0 is the original prompt, 1 is the edited prompt, above 1 exaggerates "
                                       "the edit. Above about 2.5 the picture gains color the edit did not ask for."),
                io.Int.Input("first_step", default=0, min=0, max=1000,
                             tooltip="The first sampling step that uses the edit. Earlier steps run the original "
                                     "prompt alone, which keeps its framing and pose."),
                io.Boolean.Input("align_positions", default=False, optional=True,
                                 tooltip="Give both prompts the same starting position for the video and audio "
                                         "rows when their lengths differ. Untested on renders."),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(cls, model, original, edited, scale, first_step, align_positions=False) -> io.NodeOutput:
        len_a, len_b = check_same_but_text(original, edited)
        guider = EditGuider(model)
        guider.set_conds(original, edited)
        guider.set_edit(scale, first_step, align_positions)
        logger.info("[h3] edit guidance: scale %g from step %d; text length %d (original) and %d (edited) tokens",
                    float(scale), int(first_step), len_a, len_b)
        return io.NodeOutput(guider)
