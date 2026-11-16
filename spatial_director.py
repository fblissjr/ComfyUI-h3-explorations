"""Spatial World Director (`MiniMaxH3SpatialDirector`) & Director HUD Export (`MiniMaxH3DirectorHUD`).

Turns 2D + temporal (`x, y, t`) stage actions (draggable Reality Portals, moving
vertical wipes, and Point-and-Click World Targets paired with Genie D-Pad sprites
and 3-frame sound cues) into:
  1. An `[F, H, W]` `MASK` tensor for `MiniMaxH3EditGuider.mask` (`first_step=0` gives
     bitwise 0.0 Step-0 velocity leak outside the mask with localized physics &
     lighting inside).
  2. Validated `3f`-snapped cue lines for `MiniMaxH3SoundCues`.
  3. Validated `(x, y, face)` waypoint/hop lines for `MiniMaxH3AppendSpriteTrajectory`.
  4. A per-frame `director_plan` JSON payload consumed by `MiniMaxH3DirectorHUD` to
     burn a sleek 2D cursor/reticle + Stage Radar + Multi-Lane Sequencer + optional
     Same-Seed Baseline PiP directly into showcase `.mp4` exports.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

try:
    from . import h3_rules as _rules
    from .sound_cues import SOUND_CUE_PRESETS
    from .sprite_trajectory import SpriteTrajectorySpec, parse_sprite_trajectory, evaluate_sprite_pose
except ImportError:  # standalone bench check loaded without package context
    import h3_rules as _rules
    from sound_cues import SOUND_CUE_PRESETS
    from sprite_trajectory import SpriteTrajectorySpec, parse_sprite_trajectory, evaluate_sprite_pose


VIDEO_FPS = 24.0
DIRECTOR_MODES = ("portal_path", "point_targets")
PORTAL_SHAPES = ("circle", "ellipse", "vertical_split")
HUD_LAYOUTS = ("director_card", "canvas_reticle_only", "split_ab_card")
ALL_SOUND_PRESETS = tuple(sorted(SOUND_CUE_PRESETS))
DEFAULT_DIRECTOR_EVENTS_JSON = _rules.DEFAULT_DIRECTOR_EVENTS_JSON
_SD_DEFAULTS = _rules.SPATIAL_DIRECTOR_DEFAULTS
_HUD_DEFAULTS = _rules.DIRECTOR_HUD_DEFAULTS


@dataclass(frozen=True)
class PortalKeyframe:
    frame: int
    x: float
    y: float
    rx: float
    ry: float
    shape: str
    label: str
    weight: float = 1.0
    halo_radius: float = 0.0
    halo_weight: float = 0.30


@dataclass(frozen=True)
class WorldTargetEvent:
    target_id: int
    label: str
    start: int
    end: int
    ramp: int
    x: float
    y: float
    radius: float
    cue: str
    gain: float
    weight: float = 1.0
    halo_radius: float = 0.0
    halo_weight: float = 0.30


def _cosine_ease(u: float) -> float:
    c = max(0.0, min(1.0, float(u)))
    return 0.5 - 0.5 * math.cos(math.pi * c)


def _snap_to_3f(frame: int, length: int) -> int:
    max_mult = ((length - 1) // 3) * 3
    return min(max_mult, max(0, int(round(frame / 3.0)) * 3))


def _resolve_frame_value(val: Any, length: int, *, field_name: str) -> int:
    if isinstance(val, bool):
        raise ValueError(f"MiniMaxH3SpatialDirector: `{field_name}` expected integer frame or '<sec>s', got {val!r}")
    if isinstance(val, int):
        f = val if val >= 0 else length + val
    elif isinstance(val, str):
        tok = val.strip()
        if tok.endswith("s"):
            sec = float(tok[:-1])
            if sec < 0.0:
                raise ValueError(f"MiniMaxH3SpatialDirector: `{field_name}` seconds must be >= 0, got {val!r}")
            f = int(round(sec * VIDEO_FPS))
        else:
            if "." in tok:
                raise ValueError(
                    f"MiniMaxH3SpatialDirector: `{field_name}` fractional value {val!r} requires an `s` suffix"
                )
            iv = int(tok)
            f = iv if iv >= 0 else length + iv
    else:
        raise ValueError(
            f"MiniMaxH3SpatialDirector: `{field_name}` must be int frame or '<sec>s' string, got {val!r}"
        )
    if f < 0 or f >= length:
        raise ValueError(
            f"MiniMaxH3SpatialDirector: `{field_name}` resolved to frame {f}, outside [0, {length - 1}]"
        )
    return f


def _validate_coord(val: Any, *, field_name: str, lo: float = -0.25, hi: float = 1.25) -> float:
    if isinstance(val, bool):
        raise ValueError(f"MiniMaxH3SpatialDirector: `{field_name}` expected a number, got {val!r}")
    fval = float(val)
    if not (math.isfinite(fval) and lo <= fval <= hi):
        raise ValueError(
            f"MiniMaxH3SpatialDirector: `{field_name}`={fval} outside [{lo}, {hi}]"
        )
    return fval


def _validate_halo_fields(item: dict[str, Any], *, scope: str) -> tuple[float, float, float]:
    raw_weight = item.get("weight")
    raw_halo_r = item.get("halo_radius")
    raw_halo_w = item.get("halo_weight")
    weight = _validate_coord(
        1.0 if raw_weight is None else raw_weight,
        field_name=f"{scope}.weight",
        lo=0.05,
        hi=1.0,
    )
    halo_radius = _validate_coord(
        0.0 if raw_halo_r is None else raw_halo_r,
        field_name=f"{scope}.halo_radius",
        lo=0.0,
        hi=1.0,
    )
    halo_weight = _validate_coord(
        0.30 if raw_halo_w is None else raw_halo_w,
        field_name=f"{scope}.halo_weight",
        lo=0.0,
        hi=1.0,
    )
    return weight, halo_radius, halo_weight


def parse_director_events(
    events_json: str,
    *,
    mode: str,
    length: int,
    return_portal_window: bool = False,
) -> (
    tuple[
        tuple[PortalKeyframe, ...],
        tuple[WorldTargetEvent, ...],
        list[tuple[int, str, float]],
        str,
    ]
    | tuple[
        tuple[PortalKeyframe, ...],
        tuple[WorldTargetEvent, ...],
        list[tuple[int, str, float]],
        str,
        tuple[int, int, int],
    ]
):
    """Validate and normalize `events_json` into portals, targets, portal cues, and sprite text."""
    if mode not in DIRECTOR_MODES:
        raise ValueError(f"MiniMaxH3SpatialDirector: unknown mode {mode!r}; expected {DIRECTOR_MODES}")
    try:
        doc = json.loads(events_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"MiniMaxH3SpatialDirector: invalid `events_json`: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("MiniMaxH3SpatialDirector: `events_json` must decode to a JSON object")

    raw_p_start = doc.get("portal_start")
    raw_p_end = doc.get("portal_end")
    raw_p_ramp = doc.get("portal_ramp")
    portal_start = 0 if raw_p_start is None else _resolve_frame_value(raw_p_start, length, field_name="portal_start")
    portal_end = (
        length - 1
        if raw_p_end is None
        else _resolve_frame_value(raw_p_end, length, field_name="portal_end")
    )
    if portal_end <= portal_start:
        raise ValueError(
            f"MiniMaxH3SpatialDirector: `portal_end` ({portal_end}) must exceed `portal_start` ({portal_start})"
        )
    if raw_p_ramp is None:
        portal_ramp = 0
    elif isinstance(raw_p_ramp, bool):
        raise ValueError(f"MiniMaxH3SpatialDirector: `portal_ramp` expected integer >= 0, got {raw_p_ramp!r}")
    else:
        portal_ramp = int(raw_p_ramp)
        if portal_ramp < 0:
            raise ValueError(f"MiniMaxH3SpatialDirector: `portal_ramp`={portal_ramp} must be >= 0")

    raw_portals = doc.get("portal") or []
    if not isinstance(raw_portals, list):
        raise ValueError("MiniMaxH3SpatialDirector: `portal` must be a list of keyframes")
    portals: list[PortalKeyframe] = []
    for idx, item in enumerate(raw_portals):
        if not isinstance(item, dict):
            raise ValueError(f"MiniMaxH3SpatialDirector: `portal[{idx}]` must be an object")
        frame = _resolve_frame_value(item.get("frame", 0), length, field_name=f"portal[{idx}].frame")
        x = _validate_coord(item.get("x", 0.5), field_name=f"portal[{idx}].x")
        y = _validate_coord(item.get("y", 0.5), field_name=f"portal[{idx}].y")
        rx = _validate_coord(item.get("rx", item.get("radius", 0.24)), field_name=f"portal[{idx}].rx", lo=0.01, hi=1.50)
        ry = _validate_coord(item.get("ry", rx), field_name=f"portal[{idx}].ry", lo=0.01, hi=1.50)
        shape = str(item.get("shape", "circle")).strip().lower()
        if shape not in PORTAL_SHAPES:
            raise ValueError(
                f"MiniMaxH3SpatialDirector: `portal[{idx}].shape`={shape!r} must be one of {PORTAL_SHAPES}"
            )
        label = str(item.get("label", "REALITY LENS")).strip() or "REALITY LENS"
        p_weight, p_halo_r, p_halo_w = _validate_halo_fields(item, scope=f"portal[{idx}]")
        portals.append(
            PortalKeyframe(
                frame=frame,
                x=x,
                y=y,
                rx=rx,
                ry=ry,
                shape=shape,
                label=label[:36],
                weight=p_weight,
                halo_radius=p_halo_r,
                halo_weight=p_halo_w,
            )
        )
    portals.sort(key=lambda p: p.frame)

    raw_targets = doc.get("targets") or []
    if not isinstance(raw_targets, list):
        raise ValueError("MiniMaxH3SpatialDirector: `targets` must be a list of target objects")
    targets: list[WorldTargetEvent] = []
    for idx, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise ValueError(f"MiniMaxH3SpatialDirector: `targets[{idx}]` must be an object")
        tid = int(item.get("id", idx + 1))
        label = str(item.get("label", f"TARGET {tid}")).strip() or f"TARGET {tid}"
        start = _resolve_frame_value(item.get("start", 0), length, field_name=f"targets[{idx}].start")
        end = _resolve_frame_value(item.get("end", length - 1), length, field_name=f"targets[{idx}].end")
        if end <= start:
            raise ValueError(
                f"MiniMaxH3SpatialDirector: `targets[{idx}]` end ({end}) must exceed start ({start})"
            )
        ramp = max(0, int(item.get("ramp", 4)))
        x = _validate_coord(item.get("x", 0.5), field_name=f"targets[{idx}].x")
        y = _validate_coord(item.get("y", 0.5), field_name=f"targets[{idx}].y")
        radius = _validate_coord(
            item.get("radius", item.get("rx", 0.18)),
            field_name=f"targets[{idx}].radius",
            lo=0.02,
            hi=1.20,
        )
        cue = str(item.get("cue", "slam")).strip().lower()
        if cue != "none" and cue not in SOUND_CUE_PRESETS:
            raise ValueError(
                f"MiniMaxH3SpatialDirector: `targets[{idx}].cue`={cue!r} unknown; expected 'none' or {ALL_SOUND_PRESETS}"
            )
        gain = float(item.get("gain", 1.15))
        if not (0.05 <= gain <= 2.5):
            raise ValueError(f"MiniMaxH3SpatialDirector: `targets[{idx}].gain`={gain} outside [0.05, 2.5]")
        t_weight, t_halo_r, t_halo_w = _validate_halo_fields(item, scope=f"targets[{idx}]")
        targets.append(
            WorldTargetEvent(
                target_id=tid,
                label=label[:36],
                start=start,
                end=end,
                ramp=ramp,
                x=x,
                y=y,
                radius=radius,
                cue=cue,
                gain=gain,
                weight=t_weight,
                halo_radius=t_halo_r,
                halo_weight=t_halo_w,
            )
        )
    targets.sort(key=lambda t: (t.start, t.target_id))

    if mode == "portal_path" and not portals:
        raise ValueError("MiniMaxH3SpatialDirector: `mode='portal_path'` requires at least one keyframe in `portal`")
    if mode == "point_targets" and not targets:
        raise ValueError("MiniMaxH3SpatialDirector: `mode='point_targets'` requires at least one entry in `targets`")

    portal_cues: list[tuple[int, str, float]] = []
    raw_portal_cues = doc.get("portal_cues") or []
    if not isinstance(raw_portal_cues, list):
        raise ValueError("MiniMaxH3SpatialDirector: `portal_cues` must be a list if supplied")
    for idx, item in enumerate(raw_portal_cues):
        if not isinstance(item, dict):
            raise ValueError(f"MiniMaxH3SpatialDirector: `portal_cues[{idx}]` must be an object")
        c_frame = _resolve_frame_value(item.get("frame", 0), length, field_name=f"portal_cues[{idx}].frame")
        preset = str(item.get("preset", item.get("cue", "click"))).strip().lower()
        if preset not in SOUND_CUE_PRESETS:
            raise ValueError(
                f"MiniMaxH3SpatialDirector: `portal_cues[{idx}].preset`={preset!r} unknown; expected {ALL_SOUND_PRESETS}"
            )
        gain = float(item.get("gain", 0.8))
        if not (0.05 <= gain <= 2.5):
            raise ValueError(f"MiniMaxH3SpatialDirector: `portal_cues[{idx}].gain`={gain} outside [0.05, 2.5]")
        portal_cues.append((c_frame, preset, gain))

    dpad = doc.get("dpad")
    sprite_lines: list[str] = []
    if isinstance(dpad, dict):
        wps = dpad.get("waypoints") or []
        hops = dpad.get("hops") or []
        if not isinstance(wps, list) or not isinstance(hops, list):
            raise ValueError("MiniMaxH3SpatialDirector: `dpad.waypoints` and `dpad.hops` must be lists")
        for idx, wp in enumerate(wps):
            wf = _resolve_frame_value(wp.get("frame", 0), length, field_name=f"dpad.waypoints[{idx}].frame")
            wx = _validate_coord(wp.get("x", 0.5), field_name=f"dpad.waypoints[{idx}].x")
            wy = _validate_coord(wp.get("y", 0.72), field_name=f"dpad.waypoints[{idx}].y", lo=0.0, hi=1.25)
            face = str(wp.get("face", "right")).strip().lower()
            if face not in ("right", "left", "auto"):
                raise ValueError(f"MiniMaxH3SpatialDirector: `dpad.waypoints[{idx}].face`={face!r} invalid")
            scale = float(wp.get("scale", 1.0))
            if not (0.20 <= scale <= 3.0):
                raise ValueError(f"MiniMaxH3SpatialDirector: `dpad.waypoints[{idx}].scale`={scale} outside [0.20, 3.0]")
            sprite_lines.append(f"{wf}: x={wx:.3f}, y={wy:.3f}, face={face}, scale={scale:.2f}")
        for idx, hp in enumerate(hops):
            hs = _resolve_frame_value(hp.get("start", 0), length, field_name=f"dpad.hops[{idx}].start")
            he = _resolve_frame_value(hp.get("end", hs + 6), length, field_name=f"dpad.hops[{idx}].end")
            if he <= hs:
                raise ValueError(f"MiniMaxH3SpatialDirector: `dpad.hops[{idx}]` end ({he}) must exceed start ({hs})")
            lift = float(hp.get("lift", 0.24))
            if not (0.0 <= lift <= 0.85):
                raise ValueError(f"MiniMaxH3SpatialDirector: `dpad.hops[{idx}].lift`={lift} outside [0.0, 0.85]")
            sprite_lines.append(f"hop {hs}..{he}: lift={lift:.2f}")

    if not sprite_lines:
        sprite_lines = [f"0: x=0.500, y=0.720, face=right, scale=1.00"]
    sprite_keyframes = "\n".join(sprite_lines)
    # Validate round-trip compatibility with MiniMaxH3AppendSpriteTrajectory immediately.
    parse_sprite_trajectory(sprite_keyframes, length)

    if return_portal_window:
        return (
            tuple(portals),
            tuple(targets),
            portal_cues,
            sprite_keyframes,
            (portal_start, portal_end, portal_ramp),
        )
    return tuple(portals), tuple(targets), portal_cues, sprite_keyframes


def _eval_portal_frame(
    portals: tuple[PortalKeyframe, ...],
    frame: int,
) -> tuple[float, float, float, float, str, str, float, float, float]:
    if frame <= portals[0].frame:
        p = portals[0]
        return p.x, p.y, p.rx, p.ry, p.shape, p.label, p.weight, p.halo_radius, p.halo_weight
    if frame >= portals[-1].frame:
        p = portals[-1]
        return p.x, p.y, p.rx, p.ry, p.shape, p.label, p.weight, p.halo_radius, p.halo_weight
    for i in range(len(portals) - 1):
        p0, p1 = portals[i], portals[i + 1]
        if p0.frame <= frame <= p1.frame:
            span = max(1, p1.frame - p0.frame)
            u = _cosine_ease((frame - p0.frame) / float(span))
            x = p0.x + (p1.x - p0.x) * u
            y = p0.y + (p1.y - p0.y) * u
            rx = p0.rx + (p1.rx - p0.rx) * u
            ry = p0.ry + (p1.ry - p0.ry) * u
            weight = p0.weight + (p1.weight - p0.weight) * u
            halo_radius = p0.halo_radius + (p1.halo_radius - p0.halo_radius) * u
            halo_weight = p0.halo_weight + (p1.halo_weight - p0.halo_weight) * u
            src = p0 if u < 0.5 else p1
            return x, y, rx, ry, src.shape, src.label, weight, halo_radius, halo_weight
    p = portals[-1]
    return p.x, p.y, p.rx, p.ry, p.shape, p.label, p.weight, p.halo_radius, p.halo_weight


def _span_temporal_weight(start: int, end: int, ramp: int, frame: int) -> float:
    if frame < start or frame > end:
        return 0.0
    span = end - start
    if ramp < 1 or span < 2:
        return 1.0
    r_eff = min(ramp, max(1, span // 2))
    rel_in = frame - start
    rel_out = end - frame
    if rel_in < r_eff:
        return _cosine_ease((rel_in + 1) / float(r_eff + 1))
    if rel_out < r_eff:
        return _cosine_ease((rel_out + 1) / float(r_eff + 1))
    return 1.0


def _target_temporal_weight(target: WorldTargetEvent, frame: int) -> float:
    return _span_temporal_weight(target.start, target.end, target.ramp, frame)


def _soft_edge_from_signed_dist(dist_px: np.ndarray, feather_px: float) -> np.ndarray:
    """Cosine-feathered indicator where negative `dist_px` is inside (`1.0`)."""
    if feather_px < 1.0:
        return (dist_px <= 0.0).astype(np.float32)
    half_f = 0.5 * feather_px
    t = np.clip((dist_px + half_f) / (2.0 * half_f), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(math.pi * t))).astype(np.float32)


def _mask_with_halo(
    dist_px: np.ndarray,
    *,
    feather_px: float,
    short_edge: float,
    weight: float,
    halo_radius: float,
    halo_weight: float,
) -> np.ndarray:
    """Combine weighted core feather with optional concentric outer cosine halo."""
    core = _soft_edge_from_signed_dist(dist_px, feather_px)
    if weight != 1.0:
        core = (np.float32(weight) * core).astype(np.float32, copy=False)
    halo_px = float(halo_radius) * short_edge
    if halo_px > 1.0 and halo_weight > 0.0:
        outer = (dist_px > 0.0) & (dist_px <= halo_px)
        if np.any(outer):
            amp = float(weight) * float(halo_weight) * 0.5
            t_halo = np.clip(dist_px[outer] / float(halo_px), 0.0, 1.0)
            halo_vals = (amp * (1.0 + np.cos(math.pi * t_halo))).astype(np.float32)
            core = core.copy()
            core[outer] = np.maximum(core[outer], halo_vals)
    return core


def _compute_sound_cues_text(
    *,
    mode: str,
    length: int,
    portals: tuple[PortalKeyframe, ...],
    targets: tuple[WorldTargetEvent, ...],
    portal_cues: list[tuple[int, str, float]],
) -> tuple[str, list[dict[str, Any]]]:
    entries: list[tuple[int, str, float]] = []
    if mode == "point_targets":
        for t in targets:
            if t.cue != "none":
                entries.append((_snap_to_3f(t.start, length), t.cue, t.gain))
    for frame, preset, gain in portal_cues:
        entries.append((_snap_to_3f(frame, length), preset, gain))
    if not entries:
        anchor = _snap_to_3f(portals[0].frame if portals else 3, length)
        entries.append((max(3, min(anchor, max(3, ((length - 1) // 3) * 3))), "step", 0.25))
    lines = [
        f"{f}: {preset} @{gain:.2f}" if abs(gain - 1.0) > 1e-4 else f"{f}: {preset}"
        for f, preset, gain in entries
    ]
    cue_meta = [
        {"frame": int(f), "preset": preset, "gain": round(float(gain), 3)}
        for f, preset, gain in entries
    ]
    return "\n".join(lines), cue_meta


def _compute_sprite_states(
    sprite_keyframes: str,
    length: int,
) -> tuple[bool, list[dict[str, Any]]]:
    spec: SpriteTrajectorySpec = parse_sprite_trajectory(sprite_keyframes, length)
    has_motion = (
        len(spec.waypoints) > 1
        or len(spec.hops) > 0
        or any(abs(w.x - spec.waypoints[0].x) > 1e-4 for w in spec.waypoints)
    )
    states: list[dict[str, Any]] = []
    prev_flip = False
    xs: list[float] = []
    for f in range(length):
        x_n, y_n, _gy, lift_f, flip_x = evaluate_sprite_pose(spec, f, prev_flip_x=prev_flip)
        prev_flip = flip_x
        xs.append(x_n)
        vx = 0.0 if f == 0 else x_n - xs[f - 1]
        if abs(vx) < 5e-4:
            move_dir = "idle"
        elif vx > 0.0:
            move_dir = "right"
        else:
            move_dir = "left"
        in_hop = any(h.start_frame <= f <= h.end_frame for h in spec.hops)
        moonwalk = (move_dir == "left" and not flip_x) or (move_dir == "right" and flip_x)
        states.append(
            {
                "x": round(x_n, 4),
                "y": round(y_n, 4),
                "lift": round(lift_f, 4),
                "face": "left" if flip_x else "right",
                "move_dir": move_dir,
                "hop": in_hop,
                "moonwalk": moonwalk,
            }
        )
    return has_motion, states


def _compute_target_cursor_path(
    targets: tuple[WorldTargetEvent, ...],
    length: int,
) -> list[dict[str, Any]]:
    """Precompute an animated pointer glide -> click -> hold path across world targets."""
    if not targets:
        return [{"x": 0.5, "y": 0.5, "clicking": False, "active_ids": [], "ripple": 0.0} for _ in range(length)]

    # Anchor rests at first target before first approach so viewer anticipates cursor motion.
    cursor_frames: list[dict[str, Any]] = []
    prev_x, prev_y = targets[0].x, targets[0].y
    segments: list[tuple[int, int, float, float, float, float]] = []
    # Build glide windows leading into each target's `start`.
    glide_len = 10
    for idx, t in enumerate(targets):
        if idx == 0:
            g_start = max(0, t.start - glide_len)
            start_x, start_y = max(0.08, t.x - 0.14), max(0.10, t.y - 0.14)
            segments.append((g_start, t.start, start_x, start_y, t.x, t.y))
        else:
            p = targets[idx - 1]
            g_start = max(p.start + 4, t.start - glide_len)
            segments.append((g_start, t.start, p.x, p.y, t.x, t.y))
        prev_x, prev_y = t.x, t.y

    for f in range(length):
        cx, cy = targets[0].x, targets[0].y
        if f < segments[0][0]:
            cx, cy = segments[0][2], segments[0][3]
        else:
            cx, cy = prev_x, prev_y
            for s_idx, (gs, ge, sx, sy, ex, ey) in enumerate(segments):
                if f <= gs:
                    cx, cy = sx, sy
                    break
                if gs < f <= ge:
                    u = _cosine_ease((f - gs) / float(max(1, ge - gs)))
                    cx = sx + (ex - sx) * u
                    cy = sy + (ey - sy) * u
                    break
                if s_idx + 1 < len(segments) and f < segments[s_idx + 1][0]:
                    cx, cy = ex, ey
                    break

        active_ids: list[int] = []
        clicking = False
        ripple = 0.0
        for t in targets:
            if t.start <= f <= t.end:
                active_ids.append(t.target_id)
            if t.start <= f <= min(length - 1, t.start + 10):
                clicking = True
                ripple = max(ripple, (f - t.start + 1) / 11.0)
        cursor_frames.append(
            {
                "x": round(cx, 4),
                "y": round(cy, 4),
                "clicking": clicking,
                "active_ids": active_ids,
                "ripple": round(ripple, 3),
            }
        )
    return cursor_frames


class MiniMaxH3SpatialDirector(io.ComfyNode):
    """Compile 2D+time portals, wipes, or clicked world targets into `[F, H, W]` velocity masks + cues + HUD plans."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SpatialDirector",
            display_name="MiniMax H3 Spatial World Director",
            category="conditioning/minimax",
            description=(
                "Interactive 2D+time world director for MiniMax H3 velocity steering. "
                "Compiles draggable Reality Portals or Genie + Point-and-Click World Targets "
                "into an `[F, H, W]` `MASK` for `MiniMaxH3EditGuider.mask` (keep `first_step=0` "
                "for localized Step-0 physics with 0.0 velocity leak outside the mask), "
                "`3f`-snapped `sound_cues`, `(x, y, face)` `sprite_keyframes`, and a "
                "`director_plan` for `MiniMaxH3DirectorHUD`."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=list(DIRECTOR_MODES),
                    default=_SD_DEFAULTS["mode"],
                    tooltip="`portal_path`: moving Reality Lens or vertical wipe across keyframes. "
                            "`point_targets`: timed `(x, y)` click zones paired with SoundCues & Genie D-pad.",
                ),
                io.Int.Input("width", default=_SD_DEFAULTS["width"], min=64, max=4096, step=32),
                io.Int.Input("height", default=_SD_DEFAULTS["height"], min=64, max=4096, step=32),
                io.Int.Input(
                    "length",
                    default=_SD_DEFAULTS["length"],
                    min=5,
                    max=362,
                    step=1,
                    tooltip="Exact `17n + 5` frame count matching the video latent (`90` or `141` on both clocks).",
                ),
                io.Float.Input(
                    "feather",
                    default=_SD_DEFAULTS["feather"],
                    min=0.0,
                    max=0.50,
                    step=0.01,
                    tooltip="Cosine transition width as a fraction of `min(width, height)`.",
                ),
                io.Boolean.Input("invert_mask", default=_SD_DEFAULTS["invert_mask"]),
                io.String.Input("events_json", default=DEFAULT_DIRECTOR_EVENTS_JSON, multiline=True),
                io.Image.Input("backdrop", optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="sound_cues"),
                io.String.Output(display_name="sprite_keyframes"),
                io.String.Output(display_name="director_plan"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mode: str = _SD_DEFAULTS["mode"],
        width: int = _SD_DEFAULTS["width"],
        height: int = _SD_DEFAULTS["height"],
        length: int = _SD_DEFAULTS["length"],
        feather: float = _SD_DEFAULTS["feather"],
        invert_mask: bool = _SD_DEFAULTS["invert_mask"],
        events_json: str = DEFAULT_DIRECTOR_EVENTS_JSON,
        backdrop=None,
    ) -> io.NodeOutput:
        del backdrop  # Optional ComfyUI preview backdrop for interactive HTML5 stage
        w = max(64, int(width))
        h = max(64, int(height))
        n_frames = int(length)
        if n_frames < 5 or _rules.snap_length(n_frames) != n_frames:
            raise ValueError(
                f"MiniMaxH3SpatialDirector: `length={n_frames}` must obey H3's `17n + 5` rule "
                f"(nearest valid: {_rules.snap_length(max(5, n_frames))})"
            )

        portals, targets, portal_cues, sprite_keyframes, (portal_start, portal_end, portal_ramp) = parse_director_events(
            events_json,
            mode=mode,
            length=n_frames,
            return_portal_window=True,
        )
        sound_cues_text, cue_meta = _compute_sound_cues_text(
            mode=mode,
            length=n_frames,
            portals=portals,
            targets=targets,
            portal_cues=portal_cues,
        )
        has_dpad, sprite_states = _compute_sprite_states(sprite_keyframes, n_frames)

        short_edge = float(min(w, h))
        feather_px = max(0.0, float(feather)) * short_edge
        px = np.arange(w, dtype=np.float32)[None, :] + 0.5
        py = np.arange(h, dtype=np.float32)[:, None] + 0.5

        mask_arr = np.zeros((n_frames, h, w), dtype=np.float32)
        portal_frames_meta: list[dict[str, Any]] = []
        portal_bounded = portal_start > 0 or portal_end < (n_frames - 1) or portal_ramp > 0

        if mode == "portal_path":
            for f in range(n_frames):
                x, y, rx, ry, shape, label, weight, halo_radius, halo_weight = _eval_portal_frame(portals, f)
                cx = float(x) * float(w)
                cy = float(y) * float(h)
                if shape == "vertical_split":
                    dist = cx - px  # right of split_x -> negative -> 1.0
                    dist_2d = np.broadcast_to(dist, (h, w))
                elif shape == "ellipse":
                    rad_x = max(2.0, float(rx) * float(w))
                    rad_y = max(2.0, float(ry) * float(h))
                    norm_r = np.sqrt(((px - cx) / rad_x) ** 2 + ((py - cy) / rad_y) ** 2)
                    dist_2d = (norm_r - 1.0) * (0.5 * (rad_x + rad_y))
                else:  # "circle"
                    rad_px = max(2.0, float(rx) * short_edge)
                    dist_2d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - rad_px
                frame_mask = _mask_with_halo(
                    dist_2d,
                    feather_px=feather_px,
                    short_edge=short_edge,
                    weight=weight,
                    halo_radius=halo_radius,
                    halo_weight=halo_weight,
                )
                if portal_bounded:
                    tw = _span_temporal_weight(portal_start, portal_end, portal_ramp, f)
                    if tw <= 1e-5:
                        frame_mask = np.zeros((h, w), dtype=np.float32)
                    elif tw < 1.0 - 1e-6:
                        frame_mask = (frame_mask * np.float32(tw)).astype(np.float32, copy=False)
                mask_arr[f] = frame_mask
                portal_frames_meta.append(
                    {
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "rx": round(rx, 4),
                        "ry": round(ry, 4),
                        "shape": shape,
                        "label": label,
                        "weight": round(weight, 4),
                        "halo_radius": round(halo_radius, 4),
                        "halo_weight": round(halo_weight, 4),
                    }
                )
        else:
            spatial_discs: list[tuple[WorldTargetEvent, np.ndarray]] = []
            for t in targets:
                cx = float(t.x) * float(w)
                cy = float(t.y) * float(h)
                rad_px = max(2.0, float(t.radius) * short_edge)
                dist_2d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - rad_px
                spatial_discs.append(
                    (
                        t,
                        _mask_with_halo(
                            dist_2d,
                            feather_px=feather_px,
                            short_edge=short_edge,
                            weight=t.weight,
                            halo_radius=t.halo_radius,
                            halo_weight=t.halo_weight,
                        ),
                    )
                )
            for f in range(n_frames):
                acc = np.zeros((h, w), dtype=np.float32)
                for t, s_disc in spatial_discs:
                    tw = _target_temporal_weight(t, f)
                    if tw > 1e-5:
                        np.maximum(acc, s_disc * np.float32(tw), out=acc)
                mask_arr[f] = acc

        if invert_mask:
            mask_arr = 1.0 - mask_arr
        mask_arr = np.clip(mask_arr, 0.0, 1.0).astype(np.float32, copy=False)

        if float(mask_arr.max()) <= 1e-6:
            raise ValueError(
                "MiniMaxH3SpatialDirector produced an all-zero mask; adjust target/portal size or invert_mask."
            )

        cursor_frames = _compute_target_cursor_path(targets, n_frames) if mode == "point_targets" else []
        plan = {
            "version": 1,
            "mode": mode,
            "width": w,
            "height": h,
            "length": n_frames,
            "fps": VIDEO_FPS,
            "feather": round(float(feather), 4),
            "invert_mask": bool(invert_mask),
            "portal_start": portal_start,
            "portal_end": portal_end,
            "portal_ramp": portal_ramp,
            "portals": [
                {
                    "frame": p.frame,
                    "x": round(p.x, 4),
                    "y": round(p.y, 4),
                    "rx": round(p.rx, 4),
                    "ry": round(p.ry, 4),
                    "shape": p.shape,
                    "label": p.label,
                    "weight": round(p.weight, 4),
                    "halo_radius": round(p.halo_radius, 4),
                    "halo_weight": round(p.halo_weight, 4),
                }
                for p in portals
            ],
            "portal_frames": portal_frames_meta,
            "targets": [
                {
                    "id": t.target_id,
                    "label": t.label,
                    "start": t.start,
                    "end": t.end,
                    "ramp": t.ramp,
                    "x": round(t.x, 4),
                    "y": round(t.y, 4),
                    "radius": round(t.radius, 4),
                    "cue": t.cue,
                    "gain": round(t.gain, 3),
                    "weight": round(t.weight, 4),
                    "halo_radius": round(t.halo_radius, 4),
                    "halo_weight": round(t.halo_weight, 4),
                }
                for t in targets
            ],
            "cursor_frames": cursor_frames,
            "cues": cue_meta,
            "has_dpad": has_dpad,
            "sprite_frames": sprite_states if has_dpad else [],
        }
        plan_json = json.dumps(plan, separators=(",", ":"))
        mean_cov = float(mask_arr.mean()) * 100.0
        peak_cov = float((mask_arr > 0.5).mean()) * 100.0
        summary = (
            f"Spatial World Director ({mode}): {n_frames}f @ {w}x{h} | "
            f"mean_mask={mean_cov:.1f}%, >0.5_area={peak_cov:.1f}%, "
            f"cues={len(cue_meta)}, dpad={'on' if has_dpad else 'off'}"
        )
        return io.NodeOutput(
            torch.from_numpy(mask_arr).contiguous(),
            sound_cues_text,
            sprite_keyframes,
            plan_json,
            ui=ui.PreviewText(summary),
        )


# ---------------------------------------------------------------------------
# HUD Export Helpers (`MiniMaxH3DirectorHUD`)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_MONO_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=32)
def _load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    pool = _MONO_FONT_CANDIDATES if mono else _FONT_CANDIDATES
    for p in pool:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1]))


def _draw_cursor_arrow(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    scale: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    pts = [
        (x, y),
        (x + 2.0 * scale, y + 20.0 * scale),
        (x + 7.2 * scale, y + 15.2 * scale),
        (x + 12.2 * scale, y + 24.0 * scale),
        (x + 15.6 * scale, y + 22.0 * scale),
        (x + 10.6 * scale, y + 13.4 * scale),
        (x + 17.5 * scale, y + 13.0 * scale),
    ]
    draw.polygon(pts, fill=fill, outline=outline)


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    bg: tuple[int, int, int, int],
    fg: tuple[int, int, int, int],
    border: tuple[int, int, int, int] | None = None,
    pad_x: int = 10,
    pad_y: int = 5,
) -> tuple[int, int]:
    tw, th = _text_wh(draw, text, font)
    pw, ph = tw + 2 * pad_x, th + 2 * pad_y
    x0, y0 = int(xy[0]), int(xy[1])
    draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=max(6, ph // 3), fill=bg, outline=border, width=1)
    draw.text((x0 + pad_x, y0 + pad_y - 1), text, font=font, fill=fg)
    return pw, ph


def _draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    *,
    outline: tuple[int, int, int, int],
    width: int,
    dash_deg: float = 10.0,
    gap_deg: float = 8.0,
) -> None:
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return
    step = dash_deg + gap_deg
    deg = 0.0
    while deg < 360.0:
        draw.arc(bbox, start=deg, end=min(360.0, deg + dash_deg), fill=outline, width=width)
        deg += step


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash_px: float = 14.0,
    gap_px: float = 10.0,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    step = dash_px + gap_px
    while pos < dist:
        end = min(dist, pos + dash_px)
        draw.line(
            [(p0[0] + ux * pos, p0[1] + uy * pos), (p0[0] + ux * end, p0[1] + uy * end)],
            fill=fill,
            width=width,
        )
        pos += step


def _render_scene_overlay(
    frame_rgb_u8: np.ndarray,
    *,
    frame_idx: int,
    total_frames: int,
    plan: dict[str, Any],
    title: str,
    subtitle: str,
    ring_opacity: float,
    show_top_bar: bool = True,
) -> Image.Image:
    """Composite crisp 2x-supersampled spatial agency rings, cursors, and Genie overlay onto one frame."""
    h, w = frame_rgb_u8.shape[:2]
    short_edge = float(min(w, h))
    alpha_scale = max(0.05, min(1.0, float(ring_opacity)))

    base = Image.fromarray(frame_rgb_u8, mode="RGB").convert("RGBA")
    ss = 2
    ow, oh = w * ss, h * ss
    ov = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov, "RGBA")

    mode = plan.get("mode", "portal_path")
    inv = bool(plan.get("invert_mask", False))
    pframes = plan.get("portal_frames") or []
    targets = plan.get("targets") or []
    cframes = plan.get("cursor_frames") or []
    sframes = plan.get("sprite_frames") or []

    f_clamp = min(max(0, frame_idx), max(0, total_frames - 1))
    a_main = int(round(235 * alpha_scale))
    a_soft = int(round(125 * alpha_scale))

    font_tag = _load_font(14 * ss, mono=True)
    font_small = _load_font(12 * ss, mono=True)
    font_head = _load_font(15 * ss, mono=False)

    if mode == "portal_path" and pframes:
        pf = pframes[min(f_clamp, len(pframes) - 1)]
        cx = float(pf["x"]) * ow
        cy = float(pf["y"]) * oh
        shape = pf.get("shape", "circle")
        label = pf.get("label", "REALITY LENS")
        p_start = int(plan.get("portal_start", 0))
        p_end = int(plan.get("portal_end", max(0, total_frames - 1)))
        portal_active = p_start <= f_clamp <= p_end
        p_main = a_main if portal_active else int(round(90 * alpha_scale))
        p_soft = a_soft if portal_active else int(round(50 * alpha_scale))
        p_halo_px = float(pf.get("halo_radius", 0.0)) * short_edge * ss
        closed_suffix = "" if portal_active else " [CLOSED]"

        # Breadcrumb trail behind portal center
        trail_pts: list[tuple[float, float]] = []
        for tf in range(max(0, f_clamp - 18), f_clamp + 1):
            tp = pframes[min(tf, len(pframes) - 1)]
            trail_pts.append((float(tp["x"]) * ow, float(tp["y"]) * oh))
        if len(trail_pts) >= 2:
            for ti in range(len(trail_pts) - 1):
                frac = (ti + 1) / float(len(trail_pts))
                draw.line(
                    [trail_pts[ti], trail_pts[ti + 1]],
                    fill=(92, 225, 230, int(round((160 if portal_active else 60) * frac * alpha_scale))),
                    width=max(2, int(round(3.5 * ss))),
                )

        if shape == "vertical_split":
            if p_halo_px > 1.0:
                _draw_dashed_line(
                    draw,
                    (cx - p_halo_px, 0.0),
                    (cx - p_halo_px, float(oh)),
                    fill=(125, 249, 255, int(round((175 if portal_active else 65) * alpha_scale))),
                    width=max(1, int(round(2.0 * ss))),
                )
            draw.line([(cx, 0), (cx, oh)], fill=(92, 225, 230, int(round((85 if portal_active else 35) * alpha_scale))), width=10 * ss)
            draw.line([(cx, 0), (cx, oh)], fill=(125, 249, 255, p_main), width=3 * ss)
            _draw_cursor_arrow(
                draw,
                cx + 4 * ss,
                cy - 8 * ss,
                scale=1.15 * ss,
                fill=(255, 255, 255, 245 if portal_active else 140),
                outline=(10, 14, 24, 245),
            )
            tag = f"{label} · SPLIT x={pf['x']:.2f}" + (" [INV]" if inv else "") + closed_suffix
            bx = min(ow - 280 * ss, max(16 * ss, int(cx + 18 * ss)))
            _draw_pill(
                draw,
                (bx, max(54 * ss, int(cy - 18 * ss))),
                tag,
                font=font_tag,
                bg=(11, 16, 26, 205),
                fg=(125, 249, 255, 255 if portal_active else 170),
                border=(92, 225, 230, p_main),
                pad_x=10 * ss,
                pad_y=5 * ss,
            )
        else:
            rx_px = float(pf["rx"]) * (ow if shape == "ellipse" else short_edge * ss)
            ry_px = float(pf["ry"]) * (oh if shape == "ellipse" else short_edge * ss)
            feather_r = max(6.0 * ss, float(plan.get("feather", 0.08)) * 0.5 * short_edge * ss)
            if p_halo_px > 1.0:
                avg_r = max(2.0, 0.5 * (rx_px + ry_px))
                halo_rx = rx_px * (1.0 + p_halo_px / avg_r)
                halo_ry = ry_px * (1.0 + p_halo_px / avg_r)
                _draw_dashed_ellipse(
                    draw,
                    [cx - halo_rx, cy - halo_ry, cx + halo_rx, cy + halo_ry],
                    outline=(125, 249, 255, int(round((175 if portal_active else 65) * alpha_scale))),
                    width=max(1, int(round(2.0 * ss))),
                )
            draw.ellipse(
                [cx - rx_px - feather_r, cy - ry_px - feather_r, cx + rx_px + feather_r, cy + ry_px + feather_r],
                outline=(92, 225, 230, p_soft),
                width=max(1, int(round(1.5 * ss))),
            )
            draw.ellipse(
                [cx - rx_px, cy - ry_px, cx + rx_px, cy + ry_px],
                outline=(125, 249, 255, p_main),
                width=max(2, int(round(3.0 * ss))),
            )
            tick = 9 * ss
            for tx, ty in ((cx - rx_px, cy), (cx + rx_px, cy), (cx, cy - ry_px), (cx, cy + ry_px)):
                draw.ellipse([tx - 3 * ss, ty - 3 * ss, tx + 3 * ss, ty + 3 * ss], fill=(255, 209, 102, p_main))
            draw.line([(cx - tick, cy), (cx + tick, cy)], fill=(125, 249, 255, p_soft), width=ss)
            draw.line([(cx, cy - tick), (cx, cy + tick)], fill=(125, 249, 255, p_soft), width=ss)

            grab_x = cx + rx_px * 0.70
            grab_y = cy + ry_px * 0.65
            _draw_cursor_arrow(
                draw,
                grab_x,
                grab_y,
                scale=1.15 * ss,
                fill=(255, 255, 255, 245 if portal_active else 140),
                outline=(10, 14, 24, 245),
            )
            pill_txt = f"{label} ({pf['x']:.2f}, {pf['y']:.2f})" + (" [INV]" if inv else "") + closed_suffix
            pw, ph = _text_wh(draw, pill_txt, font_tag)
            px_pos = int(max(14 * ss, min(ow - pw - 28 * ss, cx - pw * 0.5)))
            py_pos = int(max(52 * ss, cy - ry_px - ph - 20 * ss))
            _draw_pill(
                draw,
                (px_pos, py_pos),
                pill_txt,
                font=font_tag,
                bg=(11, 16, 26, 205),
                fg=(125, 249, 255, 255 if portal_active else 170),
                border=(92, 225, 230, p_main),
                pad_x=9 * ss,
                pad_y=4 * ss,
            )

    elif mode == "point_targets" and targets:
        # Target reticles: queued (slate) -> active click + hold (amber/gold) -> done (muted green)
        for t in targets:
            tx = float(t["x"]) * ow
            ty = float(t["y"]) * oh
            r_px = float(t["radius"]) * short_edge * ss
            st = int(t["start"])
            en = int(t["end"])
            tid = int(t["id"])
            lbl = str(t.get("label", f"TARGET {tid}"))
            if st <= f_clamp <= en:
                t_halo_px = float(t.get("halo_radius", 0.0)) * short_edge * ss
                if t_halo_px > 1.0:
                    _draw_dashed_ellipse(
                        draw,
                        [
                            tx - r_px - t_halo_px,
                            ty - r_px - t_halo_px,
                            tx + r_px + t_halo_px,
                            ty + r_px + t_halo_px,
                        ],
                        outline=(255, 195, 77, int(round(175 * alpha_scale))),
                        width=max(1, int(round(2.0 * ss))),
                    )
                # Active zone + shockwave on click start
                draw.ellipse(
                    [tx - r_px, ty - r_px, tx + r_px, ty + r_px],
                    outline=(255, 195, 77, a_main),
                    width=3 * ss,
                )
                age = f_clamp - st
                if age <= 12:
                    expand = r_px * (1.0 + 0.55 * (age / 12.0))
                    r_alpha = int(round(230 * (1.0 - age / 13.0) * alpha_scale))
                    draw.ellipse(
                        [tx - expand, ty - expand, tx + expand, ty + expand],
                        outline=(255, 220, 115, r_alpha),
                        width=max(2, 4 * ss),
                    )
                pill_txt = f"CLICK #{tid} · {lbl} [{t.get('cue', '')}]"
                pill_fg = (255, 214, 112, 255)
                pill_border = (255, 195, 77, a_main)
            elif f_clamp < st:
                draw.ellipse(
                    [tx - r_px * 0.78, ty - r_px * 0.78, tx + r_px * 0.78, ty + r_px * 0.78],
                    outline=(148, 163, 184, int(round(145 * alpha_scale))),
                    width=2 * ss,
                )
                draw.ellipse([tx - 5 * ss, ty - 5 * ss, tx + 5 * ss, ty + 5 * ss], fill=(148, 163, 184, 185))
                pill_txt = f"#{tid} {lbl} @ {st / VIDEO_FPS:.2f}s"
                pill_fg = (203, 213, 225, 230)
                pill_border = (100, 116, 139, 175)
            else:
                draw.ellipse(
                    [tx - r_px * 0.85, ty - r_px * 0.85, tx + r_px * 0.85, ty + r_px * 0.85],
                    outline=(74, 222, 128, int(round(135 * alpha_scale))),
                    width=2 * ss,
                )
                pill_txt = f"#{tid} {lbl} DONE"
                pill_fg = (134, 239, 172, 230)
                pill_border = (74, 222, 128, 170)

            tw_p, th_p = _text_wh(draw, pill_txt, font_small)
            px_pos = int(max(12 * ss, min(ow - tw_p - 24 * ss, tx - tw_p * 0.5)))
            py_pos = int(max(50 * ss, ty - r_px - th_p - 16 * ss))
            _draw_pill(
                draw,
                (px_pos, py_pos),
                pill_txt,
                font=font_small,
                bg=(11, 16, 26, 200),
                fg=pill_fg,
                border=pill_border,
                pad_x=8 * ss,
                pad_y=4 * ss,
            )

        if cframes:
            cf = cframes[min(f_clamp, len(cframes) - 1)]
            cx = float(cf["x"]) * ow
            cy = float(cf["y"]) * oh
            if cf.get("clicking"):
                cr = 18 * ss * (0.6 + float(cf.get("ripple", 0.3)))
                draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=(255, 215, 100, 220), width=2 * ss)
            _draw_cursor_arrow(
                draw,
                cx,
                cy,
                scale=1.25 * ss,
                fill=(255, 215, 100, 250) if cf.get("clicking") else (255, 255, 255, 245),
                outline=(9, 13, 22, 250),
            )

    # Genie Mini-Gamepad Overlay (lower-left) when D-Pad sprite trajectory is active
    if sframes:
        sf = sframes[min(f_clamp, len(sframes) - 1)]
        gx0, gy0 = 20 * ss, oh - 76 * ss
        gw, gh = 290 * ss, 56 * ss
        draw.rounded_rectangle(
            [gx0, gy0, gx0 + gw, gy0 + gh],
            radius=12 * ss,
            fill=(10, 14, 23, 212),
            outline=(71, 85, 105, 195),
            width=ss,
        )
        mdir = sf.get("move_dir", "idle")
        left_on = mdir == "left"
        right_on = mdir == "right"
        hop_on = bool(sf.get("hop"))
        aim_on = bool(cframes and cframes[min(f_clamp, len(cframes) - 1)].get("active_ids"))
        # Left / Right buttons
        btn_y = gy0 + 11 * ss
        draw.rounded_rectangle(
            [gx0 + 12 * ss, btn_y, gx0 + 48 * ss, btn_y + 34 * ss],
            radius=8 * ss,
            fill=(34, 211, 238, 220) if left_on else (30, 41, 59, 210),
            outline=(125, 249, 255, 240) if left_on else (71, 85, 105, 180),
            width=ss,
        )
        draw.text((gx0 + 21 * ss, btn_y + 7 * ss), "←", font=font_head, fill=(9, 13, 22, 255) if left_on else (148, 163, 184, 220))
        draw.rounded_rectangle(
            [gx0 + 54 * ss, btn_y, gx0 + 90 * ss, btn_y + 34 * ss],
            radius=8 * ss,
            fill=(34, 211, 238, 220) if right_on else (30, 41, 59, 210),
            outline=(125, 249, 255, 240) if right_on else (71, 85, 105, 180),
            width=ss,
        )
        draw.text((gx0 + 63 * ss, btn_y + 7 * ss), "→", font=font_head, fill=(9, 13, 22, 255) if right_on else (148, 163, 184, 220))
        # [A: HOP] & [B: AIM]
        draw.rounded_rectangle(
            [gx0 + 104 * ss, btn_y, gx0 + 188 * ss, btn_y + 34 * ss],
            radius=17 * ss,
            fill=(249, 115, 22, 235) if hop_on else (30, 41, 59, 210),
            outline=(253, 186, 116, 240) if hop_on else (71, 85, 105, 180),
            width=ss,
        )
        draw.text((gx0 + 117 * ss, btn_y + 9 * ss), "A: HOP", font=font_small, fill=(9, 13, 22, 255) if hop_on else (186, 200, 218, 220))
        draw.rounded_rectangle(
            [gx0 + 196 * ss, btn_y, gx0 + 278 * ss, btn_y + 34 * ss],
            radius=17 * ss,
            fill=(250, 204, 21, 235) if aim_on else (30, 41, 59, 210),
            outline=(254, 240, 138, 240) if aim_on else (71, 85, 105, 180),
            width=ss,
        )
        draw.text((gx0 + 208 * ss, btn_y + 9 * ss), "B: AIM", font=font_small, fill=(9, 13, 22, 255) if aim_on else (186, 200, 218, 220))

        if sf.get("moonwalk"):
            _draw_pill(
                draw,
                (gx0, gy0 - 30 * ss),
                "BACKPEDAL · FACE LOCKED RIGHT",
                font=font_small,
                bg=(15, 23, 42, 215),
                fg=(103, 232, 249, 255),
                border=(34, 211, 238, 210),
                pad_x=8 * ss,
                pad_y=3 * ss,
            )

    if show_top_bar:
        clean_title = (title or "SPATIAL WORLD DIRECTOR · 2D VELOCITY STEERING").strip()
        _draw_pill(
            draw,
            (16 * ss, 14 * ss),
            clean_title,
            font=font_head,
            bg=(9, 13, 22, 198),
            fg=(241, 245, 249, 255),
            border=(71, 85, 105, 175),
            pad_x=11 * ss,
            pad_y=5 * ss,
        )
        time_sec = f_clamp / VIDEO_FPS
        extra = (subtitle or "").strip()
        status_txt = f"t={time_sec:4.2f}s  f{f_clamp:03d}/{max(0, total_frames - 1):03d}"
        if extra:
            status_txt = f"{extra}  ·  {status_txt}"
        stw, _ = _text_wh(draw, status_txt, font_tag)
        _draw_pill(
            draw,
            (ow - stw - 38 * ss, 14 * ss),
            status_txt,
            font=font_tag,
            bg=(9, 13, 22, 198),
            fg=(125, 249, 255, 255),
            border=(92, 225, 230, 175),
            pad_x=11 * ss,
            pad_y=5 * ss,
        )

    ov_1x = ov.resize((w, h), Image.Resampling.LANCZOS)
    return Image.alpha_composite(base, ov_1x).convert("RGB")


def _render_director_footer(
    width: int,
    *,
    frame_idx: int,
    total_frames: int,
    plan: dict[str, Any],
    baseline_rgb_u8: np.ndarray | None,
) -> np.ndarray:
    """Render 144 px tall Director Telemetry Footer: Stage Radar + Multi-Lane Sequencer + Baseline PiP."""
    fh = 144
    footer = Image.new("RGB", (width, fh), (10, 14, 23))
    draw = ImageDraw.Draw(footer, "RGBA")

    f_clamp = min(max(0, frame_idx), max(0, total_frames - 1))
    mode = plan.get("mode", "portal_path")
    inv = bool(plan.get("invert_mask", False))
    pframes = plan.get("portal_frames") or []
    targets = plan.get("targets") or []
    cframes = plan.get("cursor_frames") or []
    sframes = plan.get("sprite_frames") or []
    cues = plan.get("cues") or []

    font_lbl = _load_font(11, mono=True)
    font_val = _load_font(12, mono=True)

    # Top divider line
    draw.line([(0, 0), (width, 0)], fill=(45, 58, 82, 255), width=2)

    # 1. Left panel: 2D STAGE RADAR (16:9 minimap proving 2D (x, y) control)
    pad = 12
    radar_h = 104
    radar_w = int(round(radar_h * 16.0 / 9.0))  # 185 px
    rx0, ry0 = pad, 26
    draw.text((rx0, 7), "2D STAGE RADAR (x, y)", font=font_lbl, fill=(148, 163, 184, 255))
    draw.rounded_rectangle(
        [rx0, ry0, rx0 + radar_w, ry0 + radar_h],
        radius=6,
        fill=(15, 22, 36, 255),
        outline=(51, 65, 85, 255),
        width=1,
    )
    # Faint grid inside radar
    for gx in (0.25, 0.5, 0.75):
        lx = int(round(rx0 + gx * radar_w))
        draw.line([(lx, ry0 + 2), (lx, ry0 + radar_h - 2)], fill=(30, 41, 59, 180), width=1)
    for gy in (0.5,):
        ly = int(round(ry0 + gy * radar_h))
        draw.line([(rx0 + 2, ly), (rx0 + radar_w - 2, ly)], fill=(30, 41, 59, 180), width=1)

    if mode == "portal_path" and pframes:
        pf = pframes[min(f_clamp, len(pframes) - 1)]
        mx = rx0 + float(pf["x"]) * radar_w
        my = ry0 + float(pf["y"]) * radar_h
        shape = pf.get("shape", "circle")
        if inv:
            draw.rectangle([rx0 + 2, ry0 + 2, rx0 + radar_w - 2, ry0 + radar_h - 2], fill=(24, 96, 128, 110))
        if shape == "vertical_split":
            sx = int(round(mx))
            if not inv:
                draw.rectangle([sx, ry0 + 2, rx0 + radar_w - 2, ry0 + radar_h - 2], fill=(34, 211, 238, 110))
            else:
                draw.rectangle([sx, ry0 + 2, rx0 + radar_w - 2, ry0 + radar_h - 2], fill=(15, 22, 36, 255))
            draw.line([(sx, ry0 + 2), (sx, ry0 + radar_h - 2)], fill=(125, 249, 255, 255), width=2)
        else:
            rr_x = float(pf["rx"]) * (radar_w if shape == "ellipse" else radar_h)
            rr_y = float(pf["ry"]) * radar_h
            draw.ellipse(
                [mx - rr_x, my - rr_y, mx + rr_x, my + rr_y],
                fill=(15, 22, 36, 255) if inv else (34, 211, 238, 110),
                outline=(125, 249, 255, 255),
                width=2,
            )
        # Path polyline
        path_pts = [(rx0 + float(p["x"]) * radar_w, ry0 + float(p["y"]) * radar_h) for p in pframes[::3]]
        if len(path_pts) >= 2:
            draw.line(path_pts, fill=(92, 225, 230, 150), width=1)
        draw.ellipse([mx - 3, my - 3, mx + 3, my + 3], fill=(255, 255, 255, 255))
    elif mode == "point_targets":
        for t in targets:
            tx = rx0 + float(t["x"]) * radar_w
            ty = ry0 + float(t["y"]) * radar_h
            tr = float(t["radius"]) * radar_h
            active = int(t["start"]) <= f_clamp <= int(t["end"])
            draw.ellipse(
                [tx - tr, ty - tr, tx + tr, ty + tr],
                fill=(251, 191, 36, 130) if active else None,
                outline=(251, 191, 36, 255) if active else (100, 116, 139, 175),
                width=2 if active else 1,
            )
        if cframes:
            cf = cframes[min(f_clamp, len(cframes) - 1)]
            cx = rx0 + float(cf["x"]) * radar_w
            cy = ry0 + float(cf["y"]) * radar_h
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 215, 100, 255))
        if sframes:
            sf = sframes[min(f_clamp, len(sframes) - 1)]
            sx = rx0 + float(sf["x"]) * radar_w
            sy = ry0 + float(sf["y"]) * radar_h
            draw.rectangle([sx - 3, sy - 3, sx + 3, sy + 3], fill=(74, 222, 128, 255))

    # 3. Right panel: Synchronized Same-Seed Baseline PiP OR Static Velocity Telemetry Card
    right_w = 196 if baseline_rgb_u8 is not None else 210
    bx0 = width - right_w - pad
    if baseline_rgb_u8 is not None:
        pip_h = 104
        pip_w = int(round(pip_h * 16.0 / 9.0))
        bx0 = width - pip_w - pad
        draw.text((bx0, 7), "SAME-SEED BASELINE (NO MASK)", font=font_lbl, fill=(248, 113, 113, 255))
        pip_img = Image.fromarray(baseline_rgb_u8, mode="RGB").resize((pip_w, pip_h), Image.Resampling.BILINEAR)
        footer.paste(pip_img, (bx0, 26))
        draw.rectangle([bx0, 26, bx0 + pip_w - 1, 26 + pip_h - 1], outline=(248, 113, 113, 215), width=2)
    else:
        draw.text((bx0, 7), "VELOCITY STEERING SPEC", font=font_lbl, fill=(148, 163, 184, 255))
        draw.rounded_rectangle(
            [bx0, 26, width - pad, 130],
            radius=6,
            fill=(15, 22, 36, 255),
            outline=(51, 65, 85, 255),
            width=1,
        )
        draw.text((bx0 + 10, 36), "STEP-0 LEAK: 0.0 (m==0)", font=font_lbl, fill=(74, 222, 128, 255))
        draw.text((bx0 + 10, 56), f"MODE: {mode.upper()}", font=font_lbl, fill=(125, 249, 255, 255))
        draw.text((bx0 + 10, 76), f"FEATHER: {float(plan.get('feather', 0.08)):.2f}  INV: {inv}", font=font_lbl, fill=(203, 213, 225, 255))
        draw.text((bx0 + 10, 96), "ATTENTION: FULL GLOBAL", font=font_lbl, fill=(251, 191, 36, 255))

    # 2. Center panel: SPATIAL EVENT SEQUENCER (Multi-Lane Playhead Timeline)
    seq_x0 = rx0 + radar_w + 18
    seq_x1 = bx0 - 18
    if seq_x1 > seq_x0 + 140:
        draw.text((seq_x0, 7), "SPATIAL EVENT SEQUENCER", font=font_lbl, fill=(148, 163, 184, 255))
        total_span = max(1, total_frames - 1)
        play_x = int(round(seq_x0 + (f_clamp / float(total_span)) * (seq_x1 - seq_x0)))

        # Lanes depend on mode + dpad/cues
        lanes: list[tuple[str, list[tuple[int, int, tuple[int, int, int, int], str]]]] = []
        if mode == "portal_path":
            p_list = plan.get("portals") or []
            default_start = int(p_list[0]["frame"]) if p_list else 0
            default_end = int(p_list[-1]["frame"]) if p_list else total_span
            p_start = int(plan.get("portal_start", default_start))
            p_end = int(plan.get("portal_end", default_end))
            lanes.append(("PORTAL", [(p_start, p_end, (34, 211, 238, 210), "LENS PATH")]))
        else:
            t_blocks = [
                (int(t["start"]), int(t["end"]), (245, 158, 11, 215), f"#{t['id']} {t.get('label', '')[:12]}")
                for t in targets
            ]
            lanes.append(("TARGETS", t_blocks))

        if sframes:
            move_runs: list[tuple[int, int, tuple[int, int, int, int], str]] = []
            # Highlight active travel & hops
            cur_s: int | None = None
            for fi, sf in enumerate(sframes):
                act = sf.get("move_dir", "idle") != "idle" or sf.get("hop", False)
                if act and cur_s is None:
                    cur_s = fi
                elif not act and cur_s is not None:
                    move_runs.append((cur_s, fi - 1, (52, 211, 153, 210), "D-PAD"))
                    cur_s = None
            if cur_s is not None:
                move_runs.append((cur_s, len(sframes) - 1, (52, 211, 153, 210), "D-PAD"))
            lanes.append(("GENIE", move_runs))

        if cues:
            c_blocks = [
                (
                    int(c["frame"]),
                    min(total_span, int(c["frame"]) + 5),
                    (192, 132, 252, 225),
                    f"{c['preset']}",
                )
                for c in cues
            ]
            lanes.append(("SOUNDCUES", c_blocks))

        n_lanes = max(1, len(lanes))
        top_y, bot_y = 26, 128
        draw.rounded_rectangle([seq_x0, top_y, seq_x1, bot_y], radius=6, fill=(15, 22, 36, 255), outline=(51, 65, 85, 255), width=1)
        # Second markers
        max_sec = int(total_span // VIDEO_FPS)
        for sec in range(0, max_sec + 1):
            tx = int(round(seq_x0 + ((sec * VIDEO_FPS) / float(total_span)) * (seq_x1 - seq_x0)))
            draw.line([(tx, top_y + 2), (tx, bot_y - 2)], fill=(30, 41, 59, 210), width=1)
            draw.text((tx + 3, bot_y - 14), f"{sec}s", font=font_lbl, fill=(100, 116, 139, 220))

        lane_h = max(20, (bot_y - top_y - 18) // n_lanes)
        for l_idx, (lane_name, blocks) in enumerate(lanes):
            ly0 = top_y + 4 + l_idx * lane_h
            ly1 = min(bot_y - 18, ly0 + lane_h - 4)
            draw.text((seq_x0 + 6, ly0 + max(1, (ly1 - ly0 - 11) // 2)), lane_name, font=font_lbl, fill=(148, 163, 184, 215))
            bar_x0 = seq_x0 + 78
            bar_w = max(20, seq_x1 - bar_x0 - 8)
            for bs, be, bcol, blabel in blocks:
                x_a = int(round(bar_x0 + (bs / float(total_span)) * bar_w))
                x_b = int(round(bar_x0 + (be / float(total_span)) * bar_w))
                x_b = max(x_a + 10, x_b)
                draw.rounded_rectangle([x_a, ly0 + 1, min(seq_x1 - 4, x_b), ly1 - 1], radius=4, fill=bcol)
                if x_b - x_a >= 38:
                    draw.text((x_a + 5, ly0 + max(1, (ly1 - ly0 - 11) // 2)), blabel, font=font_lbl, fill=(9, 13, 22, 255))

            if l_idx == 0:
                p_line_x = int(round(bar_x0 + (f_clamp / float(total_span)) * bar_w))
                draw.line([(p_line_x, top_y + 2), (p_line_x, bot_y - 2)], fill=(255, 255, 255, 235), width=2)
                draw.polygon(
                    [(p_line_x - 5, top_y + 1), (p_line_x + 5, top_y + 1), (p_line_x, top_y + 8)],
                    fill=(255, 209, 102, 255),
                )

        # Readout line inside sequencer right edge
        if mode == "portal_path" and pframes:
            pf = pframes[min(f_clamp, len(pframes) - 1)]
            info = f"pos=({pf['x']:.2f}, {pf['y']:.2f}) r={pf['rx']:.2f}"
        elif cframes:
            cf = cframes[min(f_clamp, len(cframes) - 1)]
            info = f"cursor=({cf['x']:.2f}, {cf['y']:.2f}) active={cf['active_ids']}"
        else:
            info = f"frame={f_clamp}"
        iw, _ = _text_wh(draw, info, font_val)
        draw.text((max(seq_x0, seq_x1 - iw), 7), info, font=font_val, fill=(226, 232, 240, 255))

    return np.asarray(footer, dtype=np.uint8)


class MiniMaxH3DirectorHUD(io.ComfyNode):
    """Overlay crisp 2D portal/cursor/Genie visuals and Director Telemetry Footer onto rendered video frames."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3DirectorHUD",
            display_name="MiniMax H3 Director HUD Export",
            category="conditioning/minimax",
            description=(
                "Renders a crisp spatial agency overlay (moving Portal Lens / animated "
                "crosshair clicks + Genie gamepad + 2D Stage Radar + Multi-Lane Event "
                "Sequencer + optional Same-Seed Baseline comparison) over decoded frames, "
                "outputting even dimensions ready for `VHS_VideoCombine` h264-mp4."
            ),
            inputs=[
                io.Image.Input("images"),
                io.String.Input("director_plan", default="", multiline=False),
                io.Combo.Input("hud_layout", options=list(HUD_LAYOUTS), default=_HUD_DEFAULTS["hud_layout"]),
                io.String.Input(
                    "title",
                    default=_HUD_DEFAULTS["title"],
                    optional=True,
                ),
                io.String.Input("subtitle", default=_HUD_DEFAULTS["subtitle"], optional=True),
                io.Float.Input("ring_opacity", default=_HUD_DEFAULTS["ring_opacity"], min=0.05, max=1.0, step=0.05, optional=True),
                io.Image.Input("baseline_images", optional=True),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        images,
        director_plan: str = "",
        hud_layout: str = _HUD_DEFAULTS["hud_layout"],
        title: str = _HUD_DEFAULTS["title"],
        subtitle: str = _HUD_DEFAULTS["subtitle"],
        ring_opacity: float = _HUD_DEFAULTS["ring_opacity"],
        baseline_images=None,
    ) -> io.NodeOutput:
        if hud_layout not in HUD_LAYOUTS:
            raise ValueError(f"MiniMaxH3DirectorHUD: unknown hud_layout {hud_layout!r}")
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError(
                f"MiniMaxH3DirectorHUD: expected `[F, H, W, 3]` tensor, got {tuple(getattr(images, 'shape', ()))}"
            )
        vid = images[..., :3].detach().cpu().clamp(0.0, 1.0)
        n_frames, h_in, w_in, _ = (int(d) for d in vid.shape)
        # Ensure even video dimensions for yuv420p H.264 encoding
        h_even = h_in - (h_in % 2)
        w_even = w_in - (w_in % 2)
        if h_even != h_in or w_even != w_in:
            vid = vid[:, :h_even, :w_even, :]
        vid_u8 = (vid.numpy() * 255.0 + 0.5).astype(np.uint8)

        if director_plan.strip():
            try:
                plan = json.loads(director_plan)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MiniMaxH3DirectorHUD: invalid `director_plan` JSON: {exc}") from exc
            if not isinstance(plan, dict):
                raise ValueError("MiniMaxH3DirectorHUD: `director_plan` must decode to a JSON object")
        else:
            plan = {"mode": "portal_path", "length": n_frames, "feather": 0.08, "invert_mask": False}

        base_u8: np.ndarray | None = None
        if baseline_images is not None:
            if not torch.is_tensor(baseline_images) or baseline_images.ndim != 4 or baseline_images.shape[-1] < 3:
                raise ValueError("MiniMaxH3DirectorHUD: `baseline_images` must be a `[F, H, W, 3]` image tensor")
            if int(baseline_images.shape[0]) != n_frames:
                raise ValueError(
                    f"MiniMaxH3DirectorHUD: `baseline_images` has {int(baseline_images.shape[0])} frames, "
                    f"expected {n_frames}"
                )
            b_clip = baseline_images[..., :3].detach().cpu().clamp(0.0, 1.0).numpy()
            base_u8 = (b_clip * 255.0 + 0.5).astype(np.uint8)

        opacity = max(0.05, min(1.0, float(ring_opacity)))

        if hud_layout == "canvas_reticle_only":
            out_frames = np.empty((n_frames, h_even, w_even, 3), dtype=np.float32)
            for f in range(n_frames):
                ov_img = _render_scene_overlay(
                    vid_u8[f],
                    frame_idx=f,
                    total_frames=n_frames,
                    plan=plan,
                    title=title,
                    subtitle=subtitle,
                    ring_opacity=opacity,
                    show_top_bar=True,
                )
                out_frames[f] = np.asarray(ov_img, dtype=np.float32) / 255.0
            return io.NodeOutput(torch.from_numpy(out_frames).contiguous())

        if hud_layout == "split_ab_card":
            # Synchronized Side-by-Side Comparison: Unsteered Baseline (Left) vs Spatially Directed HUD (Right)
            gap = 12
            panel_w = min(940, w_even)
            panel_w -= panel_w % 2
            panel_h = int(round(panel_w * float(h_even) / float(w_even)))
            panel_h -= panel_h % 2
            total_w = panel_w * 2 + gap
            total_w -= total_w % 2
            top_header_h = 38
            footer_h = 144
            total_h = top_header_h + panel_h + footer_h
            total_h -= total_h % 2

            font_hdr = _load_font(15, mono=False)
            font_tag = _load_font(12, mono=True)
            out_frames = np.empty((n_frames, total_h, total_w, 3), dtype=np.float32)
            for f in range(n_frames):
                canvas = Image.new("RGB", (total_w, top_header_h + panel_h), (10, 14, 23))
                cdraw = ImageDraw.Draw(canvas, "RGBA")
                cdraw.text((14, 9), (title or "SPATIAL WORLD DIRECTOR").strip(), font=font_hdr, fill=(241, 245, 249, 255))
                time_s = f / VIDEO_FPS
                r_txt = f"t={time_s:4.2f}s  f{f:03d}/{max(0, n_frames - 1):03d}"
                rw, _ = _text_wh(cdraw, r_txt, font_tag)
                cdraw.text((total_w - rw - 14, 11), r_txt, font=font_tag, fill=(125, 249, 255, 255))

                right_annotated = _render_scene_overlay(
                    vid_u8[f],
                    frame_idx=f,
                    total_frames=n_frames,
                    plan=plan,
                    title="",
                    subtitle="",
                    ring_opacity=opacity,
                    show_top_bar=False,
                ).resize((panel_w, panel_h), Image.Resampling.LANCZOS)

                if base_u8 is not None:
                    left_src = Image.fromarray(base_u8[f], mode="RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
                    left_lbl = "SAME-SEED BASELINE (NO MASK / NO EDIT)"
                    left_col = (248, 113, 113, 255)
                else:
                    left_src = Image.fromarray(vid_u8[f], mode="RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
                    left_lbl = "CLEAN OUTPUT (NO OVERLAY)"
                    left_col = (148, 163, 184, 255)

                canvas.paste(left_src, (0, top_header_h))
                canvas.paste(right_annotated, (panel_w + gap, top_header_h))
                cdraw = ImageDraw.Draw(canvas, "RGBA")
                _draw_pill(
                    cdraw,
                    (12, top_header_h + 10),
                    left_lbl,
                    font=font_tag,
                    bg=(9, 13, 22, 205),
                    fg=left_col,
                    border=left_col,
                )
                _draw_pill(
                    cdraw,
                    (panel_w + gap + 12, top_header_h + 10),
                    f"DIRECTED (2D VELOCITY MASK · {plan.get('mode', '').upper()})",
                    font=font_tag,
                    bg=(9, 13, 22, 205),
                    fg=(125, 249, 255, 255),
                    border=(92, 225, 230, 230),
                )

                footer_np = _render_director_footer(
                    total_w,
                    frame_idx=f,
                    total_frames=n_frames,
                    plan=plan,
                    baseline_rgb_u8=None,
                )
                full = np.vstack([np.asarray(canvas, dtype=np.uint8), footer_np])
                out_frames[f] = full[:total_h, :total_w].astype(np.float32) / 255.0
            return io.NodeOutput(torch.from_numpy(out_frames).contiguous())

        # Default: "director_card" (native 1:1 scene + on-canvas agency + 144 px Director Telemetry Footer)
        footer_h = 144
        out_h = h_even + footer_h
        out_h -= out_h % 2
        out_frames = np.empty((n_frames, out_h, w_even, 3), dtype=np.float32)
        for f in range(n_frames):
            scene_img = _render_scene_overlay(
                vid_u8[f],
                frame_idx=f,
                total_frames=n_frames,
                plan=plan,
                title=title,
                subtitle=subtitle,
                ring_opacity=opacity,
                show_top_bar=True,
            )
            footer_np = _render_director_footer(
                w_even,
                frame_idx=f,
                total_frames=n_frames,
                plan=plan,
                baseline_rgb_u8=base_u8[f] if base_u8 is not None else None,
            )
            combined = np.vstack([np.asarray(scene_img, dtype=np.uint8), footer_np])
            out_frames[f] = combined[:out_h, :w_even].astype(np.float32) / 255.0

        return io.NodeOutput(torch.from_numpy(out_frames).contiguous())
