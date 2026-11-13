"""Zero-file 2D kinematic cutout trajectory reference (`MiniMaxH3AppendSpriteTrajectory`).

Extracts a foreground cutout from `image[0]` (automatically via deterministic
border-seeded OpenCV GrabCut, or from an explicit `mask` input such as SAM3),
animates it across a low-cost neutral grey canvas (`384x224` default, `+3,108`
VAE rows vs `+21,312` at `1024x576` on 124 frames) along cosine-eased `(x, y)` waypoints and
parabolic `hop` arcs with independent facing/orientation polarity, and appends
the resulting clip directly to `H3References` in memory as `<Video N>` without
writing temporary `.mp4` files to disk or wiring `VHS_LoadVideo`.

Why this exists (`docs/wiki/research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`):
- On `unet_ref2va`, a moving 2D cutout of `<Picture 1>` acts as a **continuous
  frame-exact kinematic D-pad**: horizontal `cx(t)` reversal transfers cleanly
  (`r = +0.97`), explicit sprite orientation (`face=right` while moving `-x` vs
  `face=left` while moving `-x`) switches backward backpedal vs physical
  turn-around (`-350 px` vs `-571 px`), and parabolic `cy(t)` arcs trigger upward
  jumps.
- Pair with `MiniMaxH3ReleaseReferences(end_percent=1.0, video_end_percent=0.33)`
  so `<Picture 1>` holds subject identity across all 6 steps while the rigid
  cutout `<Video N>` releases at the `b = 0.50` flow-time midpoint (after step 1,
  `sigma_v = 0.9231`), unlocking articulated native leg strides on steps `2..5`
  instead of a frozen static-cutout glide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from comfy_api.latest import io, ui

try:
    from . import h3_rules as _rules
    from .reference_conditioning import (
        H3References,
        REFERENCES_CHAIN_TOOLTIP,
        RuntimeVideoReference,
        _appended_preview,
        _image_shape,
        _reference_tuple,
    )
except ImportError:  # run from the repo root by a check
    import h3_rules as _rules
    from reference_conditioning import (
        H3References,
        REFERENCES_CHAIN_TOOLTIP,
        RuntimeVideoReference,
        _appended_preview,
        _image_shape,
        _reference_tuple,
    )


VIDEO_FPS = 24.0

_HOP_LINE_RE = re.compile(
    r"^\s*hop\s+([+-]?\d+(?:\.\d+)?s?)\s*\.\.\s*([+-]?\d+(?:\.\d+)?s?)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_WAYPOINT_LINE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?s?)\s*:\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class SpriteWaypoint:
    frame: int
    x: float
    y: float
    face: str  # "right" (flip_x=False), "left" (flip_x=True), or "auto"
    scale: float


@dataclass(frozen=True)
class SpriteHopArc:
    start_frame: int
    end_frame: int
    lift: float


@dataclass(frozen=True)
class SpriteTrajectorySpec:
    waypoints: tuple[SpriteWaypoint, ...]
    hops: tuple[SpriteHopArc, ...]


def _parse_frame_token(token: str, length: int, *, line_no: int) -> int:
    tok = token.strip()
    if tok.endswith("s"):
        sec = float(tok[:-1])
        if sec < 0.0:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: negative seconds {tok!r} are not supported"
            )
        resolved = int(round(sec * VIDEO_FPS))
    else:
        if "." in tok:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: fractional time {tok!r} requires an `s` suffix"
            )
        f_val = int(tok)
        resolved = f_val if f_val >= 0 else length + f_val
    if resolved < 0 or resolved >= length:
        raise ValueError(
            f"MiniMaxH3AppendSpriteTrajectory line {line_no}: frame {resolved} ({tok!r}) "
            f"is outside [0, {length - 1}]"
        )
    return resolved


def _parse_kv_pairs(body: str, *, line_no: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in re.split(r"[,\s]+", body.strip()):
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: expected `key=value`, got {chunk!r}"
            )
        k, v = chunk.split("=", 1)
        key = k.strip().lower()
        val = v.strip().lower()
        if not key or not val:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: malformed `key=value` pair {chunk!r}"
            )
        out[key] = val
    return out


def parse_sprite_trajectory(keyframes_text: str, length: int) -> SpriteTrajectorySpec:
    """Parse multiline waypoint + `hop a..b: lift=...` schedule into `SpriteTrajectorySpec`."""
    if length < 5:
        raise ValueError(
            f"MiniMaxH3AppendSpriteTrajectory requires length >= 5 frames (got {length})"
        )
    if not isinstance(keyframes_text, str) or not keyframes_text.strip():
        raise ValueError("MiniMaxH3AppendSpriteTrajectory: `keyframes` schedule is empty")

    waypoints_by_frame: dict[int, SpriteWaypoint] = {}
    hops: list[SpriteHopArc] = []

    for line_no, raw_line in enumerate(keyframes_text.splitlines(), 1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue

        m_hop = _HOP_LINE_RE.match(stripped)
        if m_hop is not None:
            f0 = _parse_frame_token(m_hop.group(1), length, line_no=line_no)
            f1 = _parse_frame_token(m_hop.group(2), length, line_no=line_no)
            if f1 <= f0:
                raise ValueError(
                    f"MiniMaxH3AppendSpriteTrajectory line {line_no}: hop end frame {f1} must exceed start frame {f0}"
                )
            kv = _parse_kv_pairs(m_hop.group(3), line_no=line_no)
            unknown = set(kv) - {"lift"}
            if unknown or "lift" not in kv:
                raise ValueError(
                    f"MiniMaxH3AppendSpriteTrajectory line {line_no}: hop expects `lift=<0..0.8>`, got {sorted(kv)}"
                )
            lift = float(kv["lift"])
            if not (0.0 <= lift <= 0.85):
                raise ValueError(
                    f"MiniMaxH3AppendSpriteTrajectory line {line_no}: hop lift {lift} outside [0.0, 0.85]"
                )
            hops.append(SpriteHopArc(start_frame=f0, end_frame=f1, lift=lift))
            continue

        m_wp = _WAYPOINT_LINE_RE.match(stripped)
        if m_wp is None:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: could not parse {raw_line!r}; "
                f"expected `<frame>: x=0.20, y=0.70[, face=right|left|auto][, scale=1.0]` or `hop a..b: lift=0.29`"
            )
        f_idx = _parse_frame_token(m_wp.group(1), length, line_no=line_no)
        kv = _parse_kv_pairs(m_wp.group(2), line_no=line_no)
        unknown = set(kv) - {"x", "y", "face", "flip", "scale"}
        if unknown:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: unknown waypoint keys {sorted(unknown)}"
            )
        if "x" not in kv or "y" not in kv:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: waypoint requires both `x=` and `y=`"
            )
        x = float(kv["x"])
        y = float(kv["y"])
        if not (-0.25 <= x <= 1.25) or not (0.0 <= y <= 1.25):
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: (x={x}, y={y}) outside valid normalized canvas range"
            )
        if "face" in kv and "flip" in kv:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: specify either `face=` or `flip=`, not both"
            )
        if "flip" in kv:
            if kv["flip"] in ("true", "1", "yes"):
                face = "left"
            elif kv["flip"] in ("false", "0", "no"):
                face = "right"
            else:
                raise ValueError(
                    f"MiniMaxH3AppendSpriteTrajectory line {line_no}: invalid `flip={kv['flip']!r}`"
                )
        else:
            face = kv.get("face", "right")
            if face not in ("right", "left", "auto"):
                raise ValueError(
                    f"MiniMaxH3AppendSpriteTrajectory line {line_no}: `face` must be `right`, `left`, or `auto`, got {face!r}"
                )
        scale = float(kv.get("scale", "1.0"))
        if not (0.20 <= scale <= 3.0):
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory line {line_no}: `scale={scale}` outside [0.20, 3.0]"
            )
        waypoints_by_frame[f_idx] = SpriteWaypoint(
            frame=f_idx, x=x, y=y, face=face, scale=scale
        )

    if not waypoints_by_frame:
        raise ValueError(
            "MiniMaxH3AppendSpriteTrajectory: `keyframes` requires at least one `<frame>: x=..., y=...` waypoint"
        )

    ordered_wps = tuple(waypoints_by_frame[k] for k in sorted(waypoints_by_frame))
    ordered_hops = tuple(sorted(hops, key=lambda h: (h.start_frame, h.end_frame)))
    return SpriteTrajectorySpec(waypoints=ordered_wps, hops=ordered_hops)


def _cosine_ease(u: float) -> float:
    u_clamped = max(0.0, min(1.0, float(u)))
    return 0.5 - 0.5 * math.cos(math.pi * u_clamped)


def evaluate_sprite_pose(
    spec: SpriteTrajectorySpec,
    frame_idx: int,
    prev_flip_x: bool = False,
) -> tuple[float, float, float, float, bool]:
    """Return `(x_norm, y_norm, ground_y_norm, lift_frac, flip_x)` at `frame_idx`."""
    wps = spec.waypoints
    if frame_idx <= wps[0].frame:
        base_x = wps[0].x
        base_y = wps[0].y
        scale_mult = wps[0].scale
        face_mode = wps[0].face
        dx = 0.0
    elif frame_idx >= wps[-1].frame:
        base_x = wps[-1].x
        base_y = wps[-1].y
        scale_mult = wps[-1].scale
        face_mode = wps[-1].face
        dx = 0.0
    else:
        seg_i = 0
        for idx in range(len(wps) - 1):
            if wps[idx].frame <= frame_idx <= wps[idx + 1].frame:
                seg_i = idx
                break
        w0, w1 = wps[seg_i], wps[seg_i + 1]
        span = max(1, w1.frame - w0.frame)
        eased = _cosine_ease((frame_idx - w0.frame) / float(span))
        base_x = w0.x + (w1.x - w0.x) * eased
        base_y = w0.y + (w1.y - w0.y) * eased
        scale_mult = w0.scale + (w1.scale - w0.scale) * eased
        # Segment destination owns facing during transition (so e.g. 48 -> 92 uses waypoint 92's face).
        face_mode = w1.face if frame_idx > w0.frame else w0.face
        dx = w1.x - w0.x

    if face_mode == "left":
        flip_x = True
    elif face_mode == "right":
        flip_x = False
    else:  # auto
        if dx < -1e-4:
            flip_x = True
        elif dx > 1e-4:
            flip_x = False
        else:
            flip_x = prev_flip_x

    total_lift = 0.0
    peak_lift_frac = 0.0
    for hop in spec.hops:
        if hop.start_frame <= frame_idx <= hop.end_frame:
            phase = (frame_idx - hop.start_frame) / float(max(1, hop.end_frame - hop.start_frame))
            arc = math.sin(math.pi * phase)
            total_lift += hop.lift * arc
            peak_lift_frac = max(peak_lift_frac, arc)

    y_norm = base_y - total_lift
    return base_x, y_norm, base_y, peak_lift_frac, flip_x


def extract_sprite_rgba(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tight `(rgb_u8 [Hs, Ws, 3], alpha_f32 [Hs, Ws])` from `image[0]`."""
    _frames, h, w = _image_shape(image, "image")
    rgb_f32 = image[0, :, :, :3].detach().cpu().float().clamp(0.0, 1.0).numpy()
    rgb_u8 = np.clip(np.round(rgb_f32 * 255.0), 0, 255).astype(np.uint8)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise ValueError("MiniMaxH3AppendSpriteTrajectory: `mask` must be a torch.Tensor")
        m = mask.detach().cpu().float()
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[0, 0]
        elif m.ndim == 3:
            m = m[0]
        elif m.ndim != 2:
            raise ValueError(
                f"MiniMaxH3AppendSpriteTrajectory: expected `mask` with shape [H, W] or [B, H, W], got {tuple(mask.shape)}"
            )
        alpha = m.clamp(0.0, 1.0).numpy()
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        if float(alpha.max()) <= 0.05:
            raise ValueError(
                "MiniMaxH3AppendSpriteTrajectory: provided `mask` has no foreground pixels (> 0.05)"
            )
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0.8)
    else:
        # Deterministic border-seeded GrabCut on a bounded working resolution for fast CPU execution.
        cv2.setRNGSeed(0)
        max_side = 512
        scale = min(1.0, float(max_side) / float(max(h, w)))
        gh = max(32, int(round(h * scale)))
        gw = max(32, int(round(w * scale)))
        work_bgr = cv2.cvtColor(
            rgb_u8 if (gh == h and gw == w) else cv2.resize(rgb_u8, (gw, gh), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2BGR,
        )
        gc_mask = np.full((gh, gw), cv2.GC_PR_BGD, dtype=np.uint8)
        bx = max(2, int(0.06 * gw))
        by = max(2, int(0.06 * gh))
        gc_mask[:by, :] = cv2.GC_BGD
        gc_mask[-by:, :] = cv2.GC_BGD
        gc_mask[:, :bx] = cv2.GC_BGD
        gc_mask[:, -bx:] = cv2.GC_BGD
        gc_mask[int(0.18 * gh) : int(0.90 * gh), int(0.16 * gw) : int(0.84 * gw)] = cv2.GC_PR_FGD
        gc_mask[int(0.38 * gh) : int(0.75 * gh), int(0.30 * gw) : int(0.70 * gw)] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        except cv2.error:
            fg = np.zeros((gh, gw), dtype=np.float32)

        # Preserve thin bright upper-center features attached above the body (such as a lit candle or crest).
        work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
        upper_zone = np.zeros((gh, gw), dtype=bool)
        upper_zone[int(0.05 * gh) : int(0.42 * gh), int(0.34 * gw) : int(0.66 * gw)] = True
        bright_feature = upper_zone & (
            ((work_rgb[:, :, 0] > 185) & (work_rgb[:, :, 1] > 175) & (work_rgb[:, :, 2] > 150))
            | ((hsv[:, :, 2] > 210) & (hsv[:, :, 1] > 90))
        )
        fg = np.maximum(fg, bright_feature.astype(np.float32))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.GaussianBlur(fg, (5, 5), 1.1)

        if float(fg.sum()) < 0.01 * float(gh * gw):
            # Fallback for uniform/borderless studio crops where GrabCut finds no contrast.
            fg = np.ones((gh, gw), dtype=np.float32)

        alpha = fg if (gh == h and gw == w) else cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)

    ys, xs = np.where(alpha > 0.05)
    if len(ys) == 0:
        return rgb_u8, np.ones((h, w), dtype=np.float32)
    pad_y = max(2, int(0.015 * h))
    pad_x = max(2, int(0.015 * w))
    y0, y1 = max(0, int(ys.min()) - pad_y), min(h, int(ys.max()) + pad_y + 1)
    x0, x1 = max(0, int(xs.min()) - pad_x), min(w, int(xs.max()) + pad_x + 1)
    return rgb_u8[y0:y1, x0:x1], np.clip(alpha[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32)


def _neutral_horizon_background(canvas_w: int, canvas_h: int) -> np.ndarray:
    """Neutral grey vertical gradient + ground plane (`[canvas_h, canvas_w, 3]` float32 in `0..255`)."""
    y_grad = np.linspace(134.0, 114.0, canvas_h, dtype=np.float32)[:, None]
    bg = np.repeat(y_grad[:, :, None], 3, axis=2)
    bg = np.repeat(bg, canvas_w, axis=1)
    horizon_y = int(round(0.68 * canvas_h))
    bg[horizon_y:, :, :] *= 0.93
    bg[horizon_y : min(canvas_h, horizon_y + 2), :, :] = 98.0
    return bg


def render_sprite_trajectory(
    sprite_rgb_u8: np.ndarray,
    sprite_alpha: np.ndarray,
    spec: SpriteTrajectorySpec,
    *,
    length: int,
    canvas_width: int = 384,
    canvas_height: int = 224,
    sprite_scale: float = 0.31,
) -> torch.Tensor:
    """Render `[length, canvas_height, canvas_width, 3]` float32 tensor in `[0, 1]`."""
    sh0, sw0 = sprite_rgb_u8.shape[:2]
    aspect = float(sw0) / float(max(1, sh0))
    bg_base = _neutral_horizon_background(canvas_width, canvas_height)
    yy_grid, xx_grid = np.ogrid[:canvas_height, :canvas_width]

    frames = np.empty((length, canvas_height, canvas_width, 3), dtype=np.float32)
    prev_flip = False
    scaled_cache: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]] = {}

    for f in range(length):
        x_norm, y_norm, ground_y_norm, lift_frac, flip_x = evaluate_sprite_pose(
            spec, f, prev_flip_x=prev_flip
        )
        prev_flip = flip_x

        target_w = max(16, int(round(float(canvas_width) * float(sprite_scale) * (
            spec.waypoints[0].scale if len(spec.waypoints) == 1 else 1.0
        ))))
        # Re-evaluate interpolated waypoint scale cleanly:
        if f <= spec.waypoints[0].frame:
            wp_scale = spec.waypoints[0].scale
        elif f >= spec.waypoints[-1].frame:
            wp_scale = spec.waypoints[-1].scale
        else:
            wp_scale = 1.0
            for idx in range(len(spec.waypoints) - 1):
                w0, w1 = spec.waypoints[idx], spec.waypoints[idx + 1]
                if w0.frame <= f <= w1.frame:
                    eased = _cosine_ease((f - w0.frame) / float(max(1, w1.frame - w0.frame)))
                    wp_scale = w0.scale + (w1.scale - w0.scale) * eased
                    break
        sw = max(16, int(round(float(canvas_width) * float(sprite_scale) * wp_scale)))
        sh = max(16, int(round(float(sw) / max(aspect, 1e-4))))

        cache_key = (sw, sh, flip_x)
        cached = scaled_cache.get(cache_key)
        if cached is None:
            interp = cv2.INTER_AREA if sw <= sw0 else cv2.INTER_LINEAR
            rgb_s = cv2.resize(sprite_rgb_u8, (sw, sh), interpolation=interp).astype(np.float32)
            alpha_s = np.clip(
                cv2.resize(sprite_alpha, (sw, sh), interpolation=cv2.INTER_LINEAR),
                0.0,
                1.0,
            ).astype(np.float32)
            if flip_x:
                rgb_s = rgb_s[:, ::-1, :].copy()
                alpha_s = alpha_s[:, ::-1].copy()
            scaled_cache[cache_key] = (rgb_s, alpha_s)
        else:
            rgb_s, alpha_s = cached

        canvas = bg_base.copy()
        cx = float(x_norm) * float(canvas_width)
        cy = float(y_norm) * float(canvas_height)
        ground_y = float(ground_y_norm) * float(canvas_height)

        # Ground contact shadow stays anchored on the ground line and attenuates during upward hops.
        shadow_y = int(round(ground_y + 0.36 * sh))
        shadow_rx = max(4.0, 0.38 * sw * (1.0 - 0.40 * lift_frac))
        shadow_ry = max(2.0, 0.08 * sh * (1.0 - 0.30 * lift_frac))
        shadow_strength = 0.22 * (1.0 - 0.65 * lift_frac)
        dist2 = ((xx_grid - cx) / shadow_rx) ** 2 + ((yy_grid - shadow_y) / shadow_ry) ** 2
        shadow_mask = np.clip(1.0 - dist2, 0.0, 1.0) * shadow_strength
        canvas *= (1.0 - shadow_mask[:, :, None])

        x0 = int(round(cx - sw / 2.0))
        y0 = int(round(cy - sh / 2.0))
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        copy_w = min(sw - sx0, canvas_width - dx0)
        copy_h = min(sh - sy0, canvas_height - dy0)
        if copy_w > 0 and copy_h > 0:
            sub_rgb = rgb_s[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
            sub_a = alpha_s[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w, None]
            roi = canvas[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w]
            canvas[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = roi * (1.0 - sub_a) + sub_rgb * sub_a

        frames[f] = np.clip(canvas / 255.0, 0.0, 1.0)

    return torch.from_numpy(frames).contiguous()


class MiniMaxH3AppendSpriteTrajectory(io.ComfyNode):
    """Append an in-memory 2D cutout kinematic D-pad trajectory to `H3References` as `<Video N>`."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3AppendSpriteTrajectory",
            display_name="Append MiniMax H3 Sprite Trajectory",
            category="MiniMaxH3/references",
            description=(
                "Extract a 2D cutout from `image` (automatically via border-seeded GrabCut or "
                "from an optional `mask`), animate it along `(x, y, face)` waypoints and parabolic "
                "`hop` arcs on a compact neutral canvas (`384x224` default), and append the silent "
                "motion guide to `references` as `<Video N>` with zero `.mp4` files on disk. "
                "Pair with `MiniMaxH3ReleaseReferences(end_percent=1.0, video_end_percent=0.33)` "
                "so `<Picture 1>` holds identity across all steps while the rigid cutout releases "
                "after step 1 to unlock articulated leg strides."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Source still (`<Picture 1>`) whose foreground subject is cut out and animated.",
                ),
                io.Int.Input(
                    "length",
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["length"],
                    min=5,
                    max=4097,
                    tooltip="Frame count at 24 fps (wire to `MiniMaxH3Resolution` `length` so duration stays matched).",
                ),
                io.String.Input(
                    "keyframes",
                    multiline=True,
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["keyframes"],
                    tooltip=(
                        "Waypoints `<frame_or_s>: x=<0..1>, y=<0..1>[, face=right|left|auto][, scale=1.0]` "
                        "plus optional parabolic jumps `hop <start>..<end>: lift=<0..0.85>`. "
                        "Holding `face=right` while moving `-x` drives a backward backpedal; "
                        "`face=left` flips the cutout horizontally to drive a physical turn-around."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional foreground mask (`1.0` = subject, `0.0` = background, e.g. from SAM3). "
                        "When omitted, extracts the subject automatically via deterministic border-seeded GrabCut."
                    ),
                ),
                io.Int.Input(
                    "canvas_width",
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["canvas_width"],
                    min=128,
                    max=1024,
                    step=32,
                    optional=True,
                    tooltip="Width of the synthetic reference canvas (`384` default = `+3,108` VAE rows at 124f).",
                ),
                io.Int.Input(
                    "canvas_height",
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["canvas_height"],
                    min=128,
                    max=1024,
                    step=32,
                    optional=True,
                    tooltip="Height of the synthetic reference canvas (`224` default).",
                ),
                io.Float.Input(
                    "sprite_scale",
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["sprite_scale"],
                    min=0.10,
                    max=0.85,
                    step=0.01,
                    optional=True,
                    tooltip="Sprite width as a fraction of `canvas_width` (`0.31` default).",
                ),
                H3References.Input("references", optional=True, tooltip=REFERENCES_CHAIN_TOOLTIP),
                io.Boolean.Input(
                    "use_vae",
                    default=_rules.SPRITE_TRAJECTORY_DEFAULTS["use_vae"],
                    optional=True,
                    tooltip=(
                        "On (default): encode into 24 fps Video VAE `ref_img` tokens (needed for frame-exact "
                        "jump timing and trajectory transfer; release after step 1 with `video_end_percent=0.33`). "
                        "Off: 2 fps Qwen3-VL vision tower only."
                    ),
                ),
            ],
            outputs=[
                H3References.Output(display_name="references"),
                io.Image.Output(display_name="preview_frames"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        length: int = _rules.SPRITE_TRAJECTORY_DEFAULTS["length"],
        keyframes: str = _rules.SPRITE_TRAJECTORY_DEFAULTS["keyframes"],
        mask=None,
        canvas_width: int = _rules.SPRITE_TRAJECTORY_DEFAULTS["canvas_width"],
        canvas_height: int = _rules.SPRITE_TRAJECTORY_DEFAULTS["canvas_height"],
        sprite_scale: float = _rules.SPRITE_TRAJECTORY_DEFAULTS["sprite_scale"],
        references=None,
        use_vae: bool = _rules.SPRITE_TRAJECTORY_DEFAULTS["use_vae"],
    ) -> io.NodeOutput:
        cw = max(128, (int(canvas_width) // 32) * 32)
        ch = max(128, (int(canvas_height) // 32) * 32)
        n_frames = int(length)
        spec = parse_sprite_trajectory(keyframes, n_frames)
        sprite_rgb, sprite_alpha = extract_sprite_rgba(image, mask=mask)
        frames = render_sprite_trajectory(
            sprite_rgb,
            sprite_alpha,
            spec,
            length=n_frames,
            canvas_width=cw,
            canvas_height=ch,
            sprite_scale=float(sprite_scale),
        )
        records = _reference_tuple(references) + (
            RuntimeVideoReference(
                frames=frames,
                loaded_fps=VIDEO_FPS,
                soundtrack=None,
                use_vae=bool(use_vae),
                allow_upscale=False,
            ),
        )
        hop_note = (
            f", {len(spec.hops)} hop arc(s)"
            if spec.hops
            else ""
        )
        detail = (
            f"sprite trajectory, {cw}x{ch}, {n_frames} frames "
            f"({len(spec.waypoints)} waypoint(s){hop_note}"
            + (", text encoder only (use_vae off)" if not use_vae else "")
            + ")"
        )
        return io.NodeOutput(
            records,
            frames,
            ui=ui.PreviewText(_appended_preview(records, 1, detail)),
        )
