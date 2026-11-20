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
import math
from collections.abc import Mapping
from typing import Any

from comfy.ldm.minimax.model import FRAME_PER_TOKEN
import comfy.nested_tensor
import comfy.samplers
import torch
import torch.nn.functional as F
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
TEMPORAL_CURVES: tuple[str, ...] = ("whole_clip", "ramp_in", "sudden_midclip", "second_half")
MAX_SCALE = 4.0
PASS_KEY = "h3_edit_pass"
H3EditPlan = io.Custom("MINIMAX_H3_EDIT_PLAN")

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


def _same_opt_tensor(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return bool(torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b))


def _keyframes_extend_prefix(orig_kfs: Any, edit_kfs: Any) -> bool:
    orig_seq = list(orig_kfs) if isinstance(orig_kfs, (list, tuple)) else ([] if orig_kfs is None else [orig_kfs])
    edit_seq = list(edit_kfs) if isinstance(edit_kfs, (list, tuple)) else ([] if edit_kfs is None else [edit_kfs])
    if len(edit_seq) <= len(orig_seq):
        return False
    for ko, ke in zip(orig_seq, edit_seq):
        if not (isinstance(ko, Mapping) and isinstance(ke, Mapping)):
            return False
        if ko.get("resolved_frame_index") != ke.get("resolved_frame_index"):
            return False
        if not _same_opt_tensor(ko.get("latent"), ke.get("latent")):
            return False
        if not _same_opt_tensor(ko.get("audio_latent"), ke.get("audio_latent")):
            return False
    return True


def _normalize_edit_mask(mask: torch.Tensor) -> torch.Tensor:
    """Validate and normalize an incoming ComfyUI MASK to CPU float32 `[T, H, W]` in `[0, 1]`."""
    if not torch.is_tensor(mask):
        raise ValueError(f"MiniMaxH3EditGuider: mask must be a Tensor, got {type(mask).__name__}")
    if mask.ndim == 2:
        m = mask.unsqueeze(0)
    elif mask.ndim == 3:
        m = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[3] == 1:
        m = mask[..., 0]
    else:
        raise ValueError(
            f"MiniMaxH3EditGuider: mask must have shape [H, W], [T, H, W], [T, 1, H, W], or [T, H, W, 1], "
            f"got {tuple(mask.shape)}"
        )
    if m.numel() == 0 or any(int(d) < 1 for d in m.shape):
        raise ValueError(f"MiniMaxH3EditGuider: mask is empty ({tuple(mask.shape)})")
    if not bool(torch.isfinite(m).all()):
        raise ValueError("MiniMaxH3EditGuider: mask contains NaN or Inf values")
    m = m.detach().to(dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
    if not bool((m > 0.0).any()):
        raise ValueError(
            "MiniMaxH3EditGuider: mask is all zero (1 applies the edit, 0 keeps the original; "
            "check LoadImage alpha, SAM detections, or wire InvertMask upstream)"
        )
    return m


def _fit_crop_hw(m: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop `[T, H, W]` onto `(target_h, target_w)` aspect ratio matching `video_mask._fit_crop`."""
    in_h, in_w = int(m.shape[-2]), int(m.shape[-1])
    if in_h < 1 or in_w < 1 or target_h < 1 or target_w < 1:
        raise ValueError(
            f"MiniMaxH3EditGuider: invalid mask/target dims {(in_h, in_w)} -> {(target_h, target_w)}"
        )
    if in_w * target_h == in_h * target_w:
        return m
    old_aspect = in_w / in_h
    new_aspect = target_w / target_h
    if old_aspect > new_aspect:
        x = max(0, round((in_w - in_w * (new_aspect / old_aspect)) / 2))
        if 0 < 2 * x < in_w:
            return m[..., :, x:in_w - x]
    elif old_aspect < new_aspect:
        y = max(0, round((in_h - in_h * (old_aspect / new_aspect)) / 2))
        if 0 < 2 * y < in_h:
            return m[..., y:in_h - y, :]
    return m


def _pool_spatial_continuous(m: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Aspect-fit crop and resample `[T, H, W]` to `(target_h, target_w)` keeping continuous `[0, 1]` values."""
    m = _fit_crop_hw(m, target_h, target_w)
    in_h, in_w = int(m.shape[-2]), int(m.shape[-1])
    if (in_h, in_w) == (target_h, target_w):
        return m
    downscale = in_h >= target_h and in_w >= target_w
    chunk_size = 48
    chunks = []
    for start in range(0, int(m.shape[0]), chunk_size):
        chunk = m[start:start + chunk_size].unsqueeze(1)
        if downscale:
            out = F.adaptive_max_pool2d(chunk, (target_h, target_w))
        else:
            out = F.interpolate(chunk, size=(target_h, target_w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        chunks.append(out.squeeze(1))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def _pool_causal_temporal(m: torch.Tensor, runs: list[int], *, mode: str = "max") -> torch.Tensor:
    """Causal `[1, 4, 4, 4, 4]` pooling (`max` for spatial masks, `mean` for 1D continuous ramps) into `[len(runs), ...]`."""
    steps = []
    pos = 0
    for run in runs:
        seg = m[pos:pos + run]
        steps.append(seg.mean(dim=0) if mode == "mean" else seg.amax(dim=0))
        pos += run
    return torch.stack(steps, dim=0)


def _plan_field(edit_plan: Any, name: str, default: Any = None) -> Any:
    """Read `name` from either an `EditPlan` dataclass or a mapping."""
    if isinstance(edit_plan, Mapping):
        return edit_plan.get(name, default)
    return getattr(edit_plan, name, default)


def _has_visual_refs_meta(meta: Mapping[str, Any]) -> bool:
    """Return True if conditioning metadata carries visual references or visual keyframes."""
    raw_refs = meta.get("minimax_refs")
    refs = list(raw_refs) if isinstance(raw_refs, (list, tuple)) else ([] if raw_refs is None else [raw_refs])
    for r in refs:
        if r is None:
            continue
        if isinstance(r, Mapping):
            if "latent" in r or r.get("kind") in ("image", "video", "video_audio") or r.get("type") == "image":
                return True
        else:
            return True
    raw_kfs = meta.get("minimax_keyframes")
    kfs = list(raw_kfs) if isinstance(raw_kfs, (list, tuple)) else ([] if raw_kfs is None else [raw_kfs])
    for kf in kfs:
        if isinstance(kf, Mapping) and kf.get("latent") is not None:
            return True
    return False


def _model_ref_vae_end_percent(model: Any) -> float:
    """Return the tightest active `ref_vae_end_percent` attached to `model`, or `1.0`."""
    model_options = getattr(model, "model_options", None)
    t_opts = model_options.get("transformer_options") if isinstance(model_options, Mapping) else None
    specs = t_opts.get("h3_control_vectors") if isinstance(t_opts, Mapping) else None
    if not specs:
        return 1.0
    active: list[float] = []
    for sp in specs:
        raw_strength = sp.get("strength", 0.0) if isinstance(sp, Mapping) else getattr(sp, "strength", 0.0)
        strength = float(0.0 if raw_strength is None else raw_strength)
        vec_name = sp.get("vector_name", "none") if isinstance(sp, Mapping) else getattr(sp, "vector_name", "none")
        if strength != 0.0 or vec_name == "none":
            raw_rvep = (
                sp.get("ref_vae_end_percent", 1.0)
                if isinstance(sp, Mapping)
                else getattr(sp, "ref_vae_end_percent", 1.0)
            )
            active.append(float(1.0 if raw_rvep is None else raw_rvep))
    return min(active, default=1.0)


def build_temporal_curve(curve: str, n_frames: int) -> torch.Tensor | None:
    """Build a normalized `[n_frames, 1, 1]` float32 temporal curve, or `None` when inert."""
    if curve not in TEMPORAL_CURVES:
        raise ValueError(f"MiniMaxH3EditGuider: temporal_curve must be one of {TEMPORAL_CURVES}; got {curve!r}")
    n = int(n_frames)
    if n <= 1 or curve == "whole_clip":
        return None
    if curve == "ramp_in":
        weights = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    elif curve == "sudden_midclip":
        weights = torch.zeros(n, dtype=torch.float32)
        weights[n // 2:] = 1.0
    else:  # "second_half"
        mid = n // 2
        rem = n - mid
        weights = torch.zeros(n, dtype=torch.float32)
        weights[mid:] = torch.linspace(0.0, 1.0, rem + 1, dtype=torch.float32)[1:]
    return weights.reshape(n, 1, 1)


class EditGuider(comfy.samplers.CFGGuider):
    """Per-stream `original + scale * (edited - original)`, one pass before `first_step` and one `[edited]` pass after `last_step` (or `[original]` when `original` and `edited` share identical text and `edited` appends extra guide keyframes onto `original`)."""

    _mask: torch.Tensor | None = None
    _temporal_curve: str = "whole_clip"
    _mask_cache: dict[tuple[Any, ...], tuple[torch.Tensor | None, torch.Tensor | None]] | None = None

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
        self._release_to_original_after = bool(
            same_text
            and _keyframes_extend_prefix(
                original[0][1].get("minimax_keyframes"),
                edited[0][1].get("minimax_keyframes"),
            )
        )

    def set_edit(
        self,
        scale: float,
        first_step: int,
        align_positions: bool,
        last_step: int = 1000,
        streams: str = "both",
        audio_scale: float = 1.0,
        mask: torch.Tensor | None = None,
        temporal_curve: str = "whole_clip",
    ):
        f_step = int(first_step)
        l_step = int(last_step)
        if l_step < f_step:
            raise ValueError(f"last_step ({l_step}) must be >= first_step ({f_step})")
        if streams not in EDIT_GUIDANCE_STREAMS:
            raise ValueError(f"streams must be one of {EDIT_GUIDANCE_STREAMS}; got {streams!r}")
        if temporal_curve not in TEMPORAL_CURVES:
            raise ValueError(
                f"temporal_curve must be one of {TEMPORAL_CURVES}; got {temporal_curve!r}"
            )
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
        self._mask = None if mask is None else _normalize_edit_mask(mask)
        self._temporal_curve = str(temporal_curve)
        self._mask_cache = {}
        self._prepared = False

    def _has_active_temporal_curve(self, x, is_nested: bool, latent_shapes) -> bool:
        curve = getattr(self, "_temporal_curve", "whole_clip")
        if curve == "whole_clip":
            return False
        if is_nested:
            v_lat = x.unbind()[0]
            t_lat = int(v_lat.shape[2]) if v_lat.ndim == 5 else 1
        elif latent_shapes is not None and len(latent_shapes) >= 1 and len(latent_shapes[0]) == 5:
            t_lat = int(latent_shapes[0][2])
        elif torch.is_tensor(x) and x.ndim == 5:
            t_lat = int(x.shape[2])
        else:
            return False
        return t_lat > 1

    def _prepare_masks_for_step(
        self,
        video_shape: tuple[int, ...],
        audio_shape: tuple[int, ...] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        raw = getattr(self, "_mask", None)
        t_curve_name = getattr(self, "_temporal_curve", "whole_clip")
        if raw is None and t_curve_name == "whole_clip":
            return (None, None)
        v_key = tuple(int(d) for d in video_shape)
        a_key = None if audio_shape is None else tuple(int(d) for d in audio_shape)
        cache_key = (v_key, a_key, t_curve_name, device, dtype)
        cache = getattr(self, "_mask_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if len(v_key) != 5:
            raise ValueError(f"MiniMaxH3EditGuider: expected 5D video latent shape, got {v_key}")
        _, _, t_lat, h_lat, w_lat = v_key
        runs = [1] if t_lat == 1 else [FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(t_lat)]
        f_expected = sum(runs)
        t_curve = build_temporal_curve(t_curve_name, f_expected)
        if raw is None and t_curve is None:
            result = (None, None)
            if cache is None:
                cache = {}
                self._mask_cache = cache
            cache[cache_key] = result
            return result
        if raw is not None:
            t_in, h_in, w_in = int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2])
            if t_in not in (1, f_expected):
                raise ValueError(
                    f"MiniMaxH3EditGuider: mask has {t_in} frame(s), expected 1 or {f_expected} "
                    f"for {t_lat} video latent step(s)"
                )
        else:
            t_in, h_in, w_in = f_expected, 1, 1

        if h_in == 1 and w_in == 1:
            if raw is None:
                eff_raw = t_curve
            elif t_curve is None:
                eff_raw = raw
            else:
                if not bool(((raw[:, 0, 0] > 0.0) & (t_curve[:, 0, 0] > 0.0)).any()):
                    raise ValueError(
                        "MiniMaxH3EditGuider: spatial mask and timing curve do not overlap on any frame"
                    )
                eff_raw = raw * t_curve
            t_eff = int(eff_raw.shape[0])
            if t_eff == 1:
                v_m = eff_raw.reshape(1, 1, 1, 1, 1).expand(1, 1, t_lat, 1, 1).contiguous().to(device=device, dtype=dtype)
                a_m = None if a_key is None else eff_raw.reshape((1,) * len(a_key)).to(device=device, dtype=dtype)
            else:
                v_m = _pool_causal_temporal(eff_raw, runs, mode="mean").reshape(1, 1, t_lat, 1, 1).to(device=device, dtype=dtype)
                if a_key is None:
                    a_m = None
                else:
                    t_aud = int(a_key[-1])
                    a_1d = F.interpolate(eff_raw.reshape(1, 1, t_eff), size=t_aud, mode="linear", align_corners=False)
                    a_m = a_1d.reshape((1,) * (len(a_key) - 1) + (t_aud,)).to(device=device, dtype=dtype)
        else:
            if t_curve is not None:
                if not bool(((raw.amax(dim=(1, 2)) > 0.0) & (t_curve[:, 0, 0] > 0.0)).any()):
                    raise ValueError(
                        "MiniMaxH3EditGuider: spatial mask and timing curve do not overlap on any frame"
                    )
            pooled = _pool_spatial_continuous(raw, h_lat, w_lat)
            if t_curve is not None:
                pooled = pooled * t_curve
                v_m = _pool_causal_temporal(pooled, runs).reshape(1, 1, t_lat, h_lat, w_lat).to(device=device, dtype=dtype)
                if a_key is None:
                    a_m = None
                else:
                    t_aud = int(a_key[-1])
                    t_env = (raw.amax(dim=(1, 2)) * t_curve.reshape(-1)).reshape(1, 1, f_expected)
                    a_1d = F.interpolate(t_env, size=t_aud, mode="linear", align_corners=False)
                    a_m = a_1d.reshape((1,) * (len(a_key) - 1) + (t_aud,)).to(device=device, dtype=dtype)
            elif t_in == 1:
                a_m = None
                v_m = (
                    pooled.reshape(1, 1, 1, h_lat, w_lat)
                    .expand(1, 1, t_lat, h_lat, w_lat)
                    .contiguous()
                    .to(device=device, dtype=dtype)
                )
            else:
                v_m = _pool_causal_temporal(pooled, runs).reshape(1, 1, t_lat, h_lat, w_lat).to(device=device, dtype=dtype)
                if a_key is None:
                    a_m = None
                else:
                    t_aud = int(a_key[-1])
                    t_env = raw.amax(dim=(1, 2)).reshape(1, 1, t_in)
                    a_1d = F.interpolate(t_env, size=t_aud, mode="linear", align_corners=False)
                    a_m = a_1d.reshape((1,) * (len(a_key) - 1) + (t_aud,)).to(device=device, dtype=dtype)
        result = (v_m, a_m)
        if cache is None:
            cache = {}
            self._mask_cache = cache
        cache[cache_key] = result
        return result

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
        is_nested = bool(getattr(x, "is_nested", False)) and len(x.unbind()) == 2
        latent_shapes = getattr(self.inner_model, "latent_shapes", None)
        packed_split = (
            math.prod(latent_shapes[0][1:])
            if (not is_nested and latent_shapes is not None and len(latent_shapes) >= 2)
            else None
        )
        has_audio = is_nested or (packed_split is not None)
        if not has_audio:
            eff_a = eff_v
        has_mask = (getattr(self, "_mask", None) is not None) or self._has_active_temporal_curve(
            x, is_nested, latent_shapes
        )
        if eff_v == 0.0 and eff_a == 0.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [original], x, timestep, model_options)[0]
        if eff_v == 1.0 and eff_a == 1.0 and not has_mask:
            return comfy.samplers.calc_cond_batch(self.inner_model, [edited], x, timestep, model_options)[0]
        out = comfy.samplers.calc_cond_batch(self.inner_model, [original, edited], x, timestep, model_options)
        if (eff_v != eff_a or has_mask) and getattr(out[0], "is_nested", False) and len(out[0].unbind()) == 2:
            v0, a0 = out[0].unbind()
            v1, a1 = out[1].unbind()
            v_m, a_m = (
                self._prepare_masks_for_step(tuple(v0.shape), tuple(a0.shape), v0.device, v0.dtype)
                if has_mask
                else (None, None)
            )
            if eff_v == 0.0:
                v_out = v0
            elif v_m is None:
                v_out = v1 if eff_v == 1.0 else v0 + (v1 - v0) * eff_v
            else:
                v_out = v0 + (v1 - v0) * (eff_v * v_m)
            if eff_a == 0.0:
                a_out = a0
            elif a_m is None:
                a_out = a1 if eff_a == 1.0 else a0 + (a1 - a0) * eff_a
            else:
                a_out = a0 + (a1 - a0) * (eff_a * a_m)
            return comfy.nested_tensor.NestedTensor((v_out, a_out))
        if (
            (eff_v != eff_a or has_mask)
            and packed_split is not None
            and torch.is_tensor(out[0])
            and out[0].shape[-1] > packed_split
        ):
            v0, a0 = out[0][..., :packed_split], out[0][..., packed_split:]
            v1, a1 = out[1][..., :packed_split], out[1][..., packed_split:]
            v_m, a_m = (
                self._prepare_masks_for_step(
                    (int(v0.shape[0]), *latent_shapes[0][1:]),
                    tuple(int(d) for d in latent_shapes[1]),
                    out[0].device,
                    out[0].dtype,
                )
                if has_mask
                else (None, None)
            )
            if eff_v == 0.0:
                v_out = v0
            elif v_m is None:
                v_out = v1 if eff_v == 1.0 else v0 + (v1 - v0) * eff_v
            else:
                v_shape_5d = (int(v0.shape[0]), *latent_shapes[0][1:])
                v0_5d = v0.reshape(v_shape_5d)
                v1_5d = v1.reshape(v_shape_5d)
                v_out = (v0_5d + (v1_5d - v0_5d) * (eff_v * v_m)).reshape(v0.shape)
            if eff_a == 0.0:
                a_out = a0
            elif a_m is None:
                a_out = a1 if eff_a == 1.0 else a0 + (a1 - a0) * eff_a
            else:
                a_shape_nd = (int(a0.shape[0]), *latent_shapes[1][1:])
                a0_nd = a0.reshape(a_shape_nd)
                a1_nd = a1.reshape(a_shape_nd)
                a_out = (a0_nd + (a1_nd - a0_nd) * (eff_a * a_m)).reshape(a0.shape)
            return torch.cat((v_out, a_out), dim=-1)
        if has_mask:
            if not (torch.is_tensor(out[0]) and out[0].ndim == 5):
                raise ValueError(
                    "MiniMaxH3EditGuider: masked edit requires a 5D video tensor or AV latent, "
                    f"got {tuple(getattr(out[0], 'shape', ()))}"
                )
            if eff_v == 0.0:
                return out[0]
            v_m, _ = self._prepare_masks_for_step(tuple(out[0].shape), None, out[0].device, out[0].dtype)
            return out[0] + (out[1] - out[0]) * (eff_v * v_m)
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
                io.Conditioning.Input(
                    "original",
                    lazy=True,
                    tooltip="Conditioning of the prompt as written (lazily skipped on unmasked 1x fast_1x / auto_fast_1x runs).",
                ),
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
                                     "Steps after last_step run 1-pass [edited] (scale = 1.0) on active streams when "
                                     "unmasked; when prompts differ and a mask is attached (or a stream is 0-locked), "
                                     "steps after last_step run 2 passes so unmasked pixels (or the 0-locked stream) "
                                     "stay on [original], while same-text keyframe extensions release to 1-pass [original]."),
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
                # APPENDED 2026-11-14: optional spatial/temporal edit mask (link-only).
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional regional or temporal edit weight in [0, 1]: 1 applies the edit at scale, "
                            "0 keeps the original prediction (wire InvertMask upstream to protect a tracked subject, "
                            "and GrowMask before InvertMask to add a motion margin). Accepts [H, W] or [1, H, W] "
                            "(video-only across frames, audio stays on its stream scale), [F, H, W] matching frame "
                            "count (max-pooled over causal [1, 4, 4, 4, 4] video steps + 40 Hz max(H, W) audio "
                            "gating when F > 1), or [F, 1, 1] temporal envelopes; an all-zero mask is refused.",
                ),
                # APPENDED 2026-11-19: optional 4-axis EditPlan from MiniMaxH3AutoEditPrompt (link-only).
                H3EditPlan.Input(
                    "edit_plan",
                    optional=True,
                    tooltip="Optional 4-axis EditPlan from MiniMaxH3AutoEditPrompt. Auto-configures "
                            "first_step, last_step, streams, audio_scale, [F, 1, 1] temporal curve, and "
                            "1-pass fast_1x (explicit fast_1x or auto whole-clip overhauls when unmasked) "
                            "unless manual_widgets is set; scale remains your master dial on guided/masked edits.",
                ),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def _resolve_effective_settings(
        cls,
        *,
        scale,
        first_step,
        last_step,
        streams,
        audio_scale,
        mask=None,
        edit_plan=None,
    ) -> tuple[float, int, int, str, float, str, bool]:
        eff_scale = float(scale)
        eff_first_step = int(first_step)
        eff_last_step = int(last_step)
        eff_streams = str(streams)
        eff_audio_scale = float(audio_scale)
        temporal_curve = "whole_clip"
        if edit_plan is not None and not bool(_plan_field(edit_plan, "manual_widgets", False)):
            force_scale = _plan_field(edit_plan, "force_scale", None)
            auto_fast_1x = bool(_plan_field(edit_plan, "auto_fast_1x", False))
            if force_scale is not None:
                if mask is not None:
                    raise ValueError(
                        "MiniMaxH3EditGuider: take_mode='fast_1x' cannot be combined with a spatial mask "
                        "(masking requires 2-pass guidance; switch take_mode to 'auto' or 'lock_take')"
                    )
                eff_scale = float(force_scale)
                eff_first_step = 0
                eff_last_step = 3
                eff_streams = "both"
                eff_audio_scale = 1.0
                temporal_curve = "whole_clip"
            elif auto_fast_1x and mask is None:
                eff_scale = 1.0
                eff_first_step = 0
                eff_last_step = 3
                eff_streams = "both"
                eff_audio_scale = 1.0
                temporal_curve = "whole_clip"
            else:
                eff_first_step = int(_plan_field(edit_plan, "first_step", eff_first_step))
                eff_last_step = int(_plan_field(edit_plan, "last_step", eff_last_step))
                eff_streams = str(_plan_field(edit_plan, "streams", eff_streams))
                eff_audio_scale = float(_plan_field(edit_plan, "audio_scale", eff_audio_scale))
                temporal_curve = str(_plan_field(edit_plan, "temporal_curve", "whole_clip"))
        pure_edited_1x = (
            eff_first_step == 0
            and abs(eff_scale - 1.0) <= 1e-9
            and (eff_streams == "both" or (eff_streams == "independent" and abs(eff_audio_scale - 1.0) <= 1e-9))
            and mask is None
            and temporal_curve == "whole_clip"
        )
        return eff_scale, eff_first_step, eff_last_step, eff_streams, eff_audio_scale, temporal_curve, pure_edited_1x

    @classmethod
    def check_lazy_status(
        cls,
        model=None,
        original=None,
        edited=None,
        scale=DEFAULT_SCALE,
        first_step=_rules.EDIT_GUIDANCE_DEFAULTS["first_step"],
        align_positions=False,
        last_step=_rules.EDIT_GUIDANCE_DEFAULTS["last_step"],
        streams=_rules.EDIT_GUIDANCE_DEFAULTS["streams"],
        audio_scale=DEFAULT_AUDIO_SCALE,
        mask=None,
        edit_plan=None,
    ):
        if original is not None:
            return []
        try:
            _s, _fs, _ls, _st, _as, _tc, pure_edited_1x = cls._resolve_effective_settings(
                scale=scale,
                first_step=first_step,
                last_step=last_step,
                streams=streams,
                audio_scale=audio_scale,
                mask=mask,
                edit_plan=edit_plan,
            )
        except Exception:
            return ["original"]
        if pure_edited_1x:
            return []
        return ["original"]

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
        mask=None,
        edit_plan=None,
    ) -> io.NodeOutput:
        eff_scale, eff_first_step, eff_last_step, eff_streams, eff_audio_scale, temporal_curve, pure_edited_1x = (
            cls._resolve_effective_settings(
                scale=scale,
                first_step=first_step,
                last_step=last_step,
                streams=streams,
                audio_scale=audio_scale,
                mask=mask,
                edit_plan=edit_plan,
            )
        )
        skipped_original = original is None
        if skipped_original:
            if not pure_edited_1x:
                raise ValueError(
                    "MiniMaxH3EditGuider requires 'original' conditioning when running 2-pass, masked, or delayed edit guidance"
                )
            eff_original = edited
            len_b = int(edited[0][0].shape[1]) if (edited and torch.is_tensor(edited[0][0]) and edited[0][0].ndim >= 2) else 0
            len_a = len_b
            kf_b = _count(edited[0][1].get("minimax_keyframes"))
            kf_a = kf_b
        else:
            eff_original = original
            len_a, len_b = check_same_but_text(original, edited)
            kf_a = _count(original[0][1].get("minimax_keyframes"))
            kf_b = _count(edited[0][1].get("minimax_keyframes"))
        if edit_plan is not None and not bool(_plan_field(edit_plan, "manual_widgets", False)):
            raw_plan_ref_end = _plan_field(edit_plan, "ref_end_percent", 1.0)
            plan_ref_end = float(1.0 if raw_plan_ref_end is None else raw_plan_ref_end)
            ref_meta_src = edited[0][1] if skipped_original else original[0][1]
            if plan_ref_end < 1.0 - 1e-6 and _has_visual_refs_meta(ref_meta_src):
                model_ref_end = _model_ref_vae_end_percent(model)
                if model_ref_end >= 1.0 - 1e-6:
                    logger.warning(
                        "[h3] edit_plan requests ref_end_percent=%.2f because the edit deforms the "
                        "subject silhouette, and visual references/keyframes are attached, but no "
                        "upstream MiniMaxH3ReleaseReferences lowers ref_vae_end_percent below 1.0 "
                        "(active=%.2f). Wire MiniMaxH3AutoEditPrompt.edit_plan -> "
                        "MiniMaxH3ReleaseReferences.edit_plan on the model chain so late steps can "
                        "release reference rows.",
                        plan_ref_end,
                        model_ref_end,
                    )
        guider = EditGuider(model)
        guider.set_conds(eff_original, edited)
        guider.set_edit(
            eff_scale,
            eff_first_step,
            align_positions,
            last_step=eff_last_step,
            streams=eff_streams,
            audio_scale=eff_audio_scale,
            mask=mask,
            temporal_curve=temporal_curve,
        )
        kf_note = f"; keyframes {kf_a} (original) and {kf_b} (edited)" if (kf_a or kf_b) else ""
        mask_note = f"; mask={tuple(mask.shape)}" if mask is not None else ""
        curve_note = f"; temporal_curve={temporal_curve}" if temporal_curve != "whole_clip" else ""
        lazy_note = "; original skipped (lazy 1x)" if skipped_original else ""
        logger.info(
            "[h3] edit guidance: streams=%s (video=%g, audio=%g) across steps %d..%d; "
            "text length %d (original) and %d (edited) tokens%s%s%s%s",
            eff_streams,
            guider.edit_scale,
            guider.edit_audio_scale,
            eff_first_step,
            eff_last_step,
            len_a,
            len_b,
            kf_note,
            mask_note,
            curve_note,
            lazy_note,
        )
        return io.NodeOutput(guider)
