#!/usr/bin/env python3
"""Measure visible 16 px blocks on a laid clip or render piece and a saved window region.

    <comfy venv python> bench/measure_blocks.py --clip CLIP.mp4 --region WINDOW_1_region.npz \\
        [--first-frame 751] [--region-first 751] [--only 751-785] [--band-cells 2] \\
        [--source SOURCE.mov] [--out blocks.json]

Inside the subject's region and a band of `--band-cells` 16x16 cells around it (plus any cells
in the window's carried or kept masks), measures per frame and per 16x16 cell:
1. The brightness jump across each 16 px grid line (`grid_jump`: mean `|Y(16k) - Y(16k - 1)|`
   along the 16 px cell edge) against the jump one pixel either side of it (`side_jump`: mean of
   `|Y(16k - 1) - Y(16k - 2)|` and `|Y(16k + 1) - Y(16k)|`). A square's edge stands out when
   `grid_jump` exceeds `side_jump` (`grid_excess = grid_jump - side_jump`,
   `grid_ratio = (grid_jump + 1) / (side_jump + 1)`), while a real scene gradient across the grid
   has a comparable `side_jump`, and a straight scene line on the grid runs unclosed along a single
   axis without forming a 16x16 square (`real_edges_on_grid`).
2. The temporal step across frames (`f - 1` to `f`): each 16x16 square's interior mean frame step
   (`cell_step = mean(|Y_f - Y_{f-1}|)`) against the median step of its 8 neighbouring 16x16
   squares (`neighbour_step`), `step_excess = cell_step - neighbour_step`, and
   `step_ratio = (cell_step + 1) / (neighbour_step + 1)` (a square that changes in one frame step
   while its neighbours do not).

Prints per-frame and per-square data with what was measured on and what it cannot see. No verdict:
exits 0 on data, and exits non-zero only on bad or missing inputs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap, home_relative  # noqa: E402
from _lib.frames import LOADER_CROP, probe  # noqa: E402

#: Latent cell width and height in canvas pixels (8x spatial VAE stride * 2x patch size).
#: Inherited from `video_mask.CELL` (`video_mask.py:39`).
CELL = 16

#: Default band of 16x16 cells dilated around the subject's region on each frame (`--band-cells`).
#: Reasoned: 2 cells (32 px) covers the subject's regenerated region, its outer boundary ring, and
#: adjacent kept-out cells (`grown & ~tokens`) that lie just outside the token region.
DEFAULT_BAND_CELLS = 2

#: Temporal reach in frames (+/- one 4-frame latent step) unioned into the measured region mask so
#: a cell carved out of the region for a single latent step stays inside the measured band on those
#: frames. Inherited from `video_mask.TIME_STRIDE` (`4` frames per latent step).
TEMPORAL_REACH_FRAMES = 4

#: Minimum luma jump excess (`grid_jump - side_jump`, in 0..255 BT.709 Y) along a 16 px cell edge
#: for that edge segment to count as an active 1-pixel step on the 16 px grid. Reasoned and measured
#: on `lead_fp_0751` (source frames 751-785, 2026-10-18), where visible kept/boundary squares have
#: per-side `grid_excess` of 4.0 to 24.5 luma levels while smooth interior cells sit below 1.5.
GRID_EXCESS_MIN = 3.5

#: Minimum ratio `(grid_jump + 1.0) / (side_jump + 1.0)` along a 16 px cell edge for that edge to
#: count as a sharp grid step rather than a multi-pixel scene gradient crossing the grid line.
#: Reasoned and measured on `lead_fp_0751` (2026-10-18).
GRID_RATIO_MIN = 1.8

#: Minimum `grid_jump` (in 0..255 BT.709 Y) along a 16 px grid segment to record a real edge on the
#: grid (`real_edges_on_grid`) when the edge is either a multi-pixel gradient (`grid_ratio < GRID_RATIO_MIN`
#: or `grid_excess < GRID_EXCESS_MIN`) or an unclosed single line on the grid. Reasoned: 6.0 luma levels
#: is above sensor/codec grain (~1-2 Y) and marks a genuine contrast edge crossing the 16 px line.
REAL_EDGE_JUMP_MIN = 6.0

#: Minimum run length in consecutive cells along a single 16 px grid line (with no parallel opposite
#: cell wall 16 px away) for a sharp 1-pixel grid step to be recognised as a long collinear line in
#: the picture (`real_edges_on_grid`) rather than a 16x16 square side. Reasoned: 4 cells = 64 px.
COLLINEAR_LINE_CELLS = 4

#: Minimum across-frame luma step excess (`cell_step - neighbour_step`, in 0..255 BT.709 Y, where
#: `neighbour_step` is the median `cell_step` of the 8 surrounding 16x16 squares) for a square to
#: stand out as a temporal step. Reasoned and measured on `lead_fp_0751` (source frames 751-785,
#: 2026-10-18), where 1-frame slip and keep-out carve squares step by +4.0 to +28.0 Y above their
#: neighbours.
STEP_EXCESS_MIN = 3.5

#: Minimum ratio `(cell_step + 1.0) / (neighbour_step + 1.0)` for a 16x16 square's frame-to-frame
#: luma change against the median of its 8 neighbours. Reasoned and measured on `lead_fp_0751`
#: (2026-10-18).
STEP_RATIO_MIN = 1.8

#: Minimum 16 px boundary jump excess on the temporal difference image `|Y_f - Y_{f-1}|` (or spatial
#: `grid_excess` on frame `f` or `f - 1`) required alongside `STEP_EXCESS_MIN` so a smooth moving
#: feature that does not stop on a 16 px boundary is not miscounted as a stepping 16x16 square.
#: Reasoned: 2.0 luma levels across the 16 px line.
DIFF_GRID_EXCESS_MIN = 2.0

#: Minimum share of the 16 pixels along a 16 px cell edge segment that must individually step across
#: the grid line (`|Y(16k) - Y(16k-1)| - side_px >= 2.0`) for the segment to count as a square wall
#: rather than a narrow diagonal contour crossing the grid line at a few pixels. Reasoned: 0.5
#: (8 of the 16 pixels along the cell wall).
WALL_COVERAGE_MIN = 0.5


def _pack(name: str):
    """Load a root module of the pack on CPU without opening a CUDA context."""
    comfy_cand = Path(sys.executable).parents[2]
    if not (REPO.parents[1] / "comfy").is_dir() and (comfy_cand / "comfy").is_dir():
        if str(comfy_cand) not in sys.path:
            sys.path.insert(0, str(comfy_cand))
    bootstrap(cpu=True, front=False)
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    return importlib.import_module(f"_h3pack.{name}")


def _loader_fit(w: int, h: int) -> str:
    """The lane video loader's centre crop and bicubic scale to `(w, h)` (`bench/assemble_delivery.py::loader_fit`)."""
    return LOADER_CROP.format(ar=float(w) / float(h)) + f",scale={w}:{h}:flags=bicubic"


def _parse_only(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    s = text.strip()
    sep = ":" if ":" in s else "-"
    parts = s.split(sep)
    if len(parts) not in (1, 2):
        raise SystemExit(f"REFUSED: --only must be FIRST-LAST, FIRST:LAST, or FRAME, got {text!r}")
    try:
        lo = int(parts[0])
        hi = int(parts[-1])
    except ValueError:
        raise SystemExit(f"REFUSED: --only must hold integer frame numbers, got {text!r}") from None
    if lo < 0 or hi < lo:
        raise SystemExit(f"REFUSED: --only range must satisfy 0 <= FIRST <= LAST, got {text!r}")
    return lo, hi


def _discover_region_files(paths: list[str]) -> list[Path]:
    """Resolve `--region` arguments (files or folders of `*_window_*_region.npz`) into an ordered list of `.npz` paths."""
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise SystemExit(f"REFUSED: --region {raw}: file or directory not found")
        if p.is_dir():
            win = sorted(
                [f for f in p.glob("*_window_*_region.npz") if not f.name.endswith("_planned_region.npz")],
                key=lambda f: int(f.stem.split("_window_")[1].split("_")[0]) if "_window_" in f.stem else 0,
            )
            if not win:
                win = sorted(
                    p.glob("*_window_*_planned_region.npz"),
                    key=lambda f: int(f.stem.split("_window_")[1].split("_")[0]) if "_window_" in f.stem else 0,
                )
            if not win and (p / "region.npz").is_file():
                win = [p / "region.npz"]
            if not win:
                raise SystemExit(f"REFUSED: --region directory {raw} holds no *_window_*_region.npz or region.npz files")
            resolved.extend(win)
        else:
            if p.suffix.lower() != ".npz":
                raise SystemExit(f"REFUSED: --region {p.name}: expected a .npz file")
            resolved.append(p)
    if not resolved:
        raise SystemExit("REFUSED: give at least one --region .npz file or windows directory")
    return resolved


def load_region_data(paths: list[str]) -> dict:
    """Load one or more window `*_region.npz` files (or a capture's `region.npz`) onto the 16 px cell grid.

    Returns a dict with:
      - `region_cells`: `bool` array `[F, lh, lw]` of regenerated token cells per frame
      - `subject_cells`: `bool` array `[F, lh, lw]` of cells covered by the window's carried/grown mask
      - `kept_carved_cells`: `bool` array `[F, lh, lw]` of cells in the grown subject mask or `kept` mask
        that were kept out of `region_cells` (`~region_cells`)
      - `canvas_size`: `(W, H)` in pixels (`(lw * CELL, lh * CELL)`)
      - `zoom`: zoom dict if the window was rendered zoomed in, else `None`
      - `files`: list of file names loaded
      - `kind`: `"window_region"` or `"capture_region"`
      - `inferred_first_frame`: `int | None` (from capture manifest when available)
    """
    files = _discover_region_files(paths)
    try:
        first_z = np.load(files[0])
    except Exception as exc:
        raise SystemExit(f"REFUSED: --region {files[0].name}: could not read .npz ({exc})") from None

    # Case A: a capture's `runs/<run>/region.npz` (`region` [F, lh, lw] bool, optional `carried`, `given_back__*`)
    if "region" in first_z.files and "tokens" not in first_z.files:
        if len(files) > 1:
            raise SystemExit("REFUSED: a capture region.npz already spans the run; pass one capture region.npz or window *_region.npz files")
        reg = np.asarray(first_z["region"], dtype=bool)
        if reg.ndim != 3:
            raise SystemExit(f"REFUSED: --region {files[0].name}: `region` must be 3D [F, lh, lw], got shape {reg.shape}")
        f_count, lh, lw = reg.shape
        w, h = lw * CELL, lh * CELL
        subj = reg.copy()
        if "carried" in first_z.files:
            carried = np.unpackbits(first_z["carried"], axis=-1)[..., :w].astype(bool)
            if carried.shape == (f_count, h, w):
                subj |= carried.reshape(f_count, lh, CELL, lw, CELL).any(axis=(2, 4))
        kept_carved = np.zeros_like(reg)
        for k in first_z.files:
            if k.startswith("given_back__"):
                gb = np.asarray(first_z[k], dtype=bool)
                if gb.shape == reg.shape:
                    kept_carved |= gb
        inferred_first = None
        manifest_path = files[0].parent.parent.parent / "manifest.json"
        if manifest_path.is_file():
            try:
                m = json.loads(manifest_path.read_text())
                inferred_first = int(m.get("first_frame", 0))
            except Exception:
                inferred_first = None
        return {
            "region_cells": reg,
            "subject_cells": subj | kept_carved,
            "kept_carved_cells": kept_carved | (subj & ~reg),
            "canvas_size": (w, h),
            "zoom": None,
            "files": [files[0].name],
            "kind": "capture_region",
            "inferred_first_frame": inferred_first,
        }

    # Case B: one or more window `*_region.npz` files saved by `video_mask.save_window_region`
    for f in files:
        try:
            z = np.load(f)
        except Exception as exc:
            raise SystemExit(f"REFUSED: --region {f.name}: could not read .npz ({exc})") from None
        lacking = [k for k in ("mask", "tokens", "margin", "meta") if k not in z.files]
        if lacking:
            raise SystemExit(f"REFUSED: --region {f.name}: missing {', '.join(lacking)} (expected window *_region.npz or capture region.npz)")

    import torch
    vm = _pack("video_mask")
    reg_pieces: list[np.ndarray] = []
    subj_pieces: list[np.ndarray] = []
    kept_pieces: list[np.ndarray] = []
    end = 0
    canvas_size: tuple[int, int] | None = None
    zoom = None
    names: list[str] = []

    for f in files:
        got = vm.load_window_region(str(f))
        tokens = got["tokens"]
        mask = got["mask"]
        if tokens.ndim != 3 or mask.ndim != 3:
            raise SystemExit(f"REFUSED: --region {f.name}: invalid tensor shapes (tokens {tuple(tokens.shape)}, mask {tuple(mask.shape)})")
        lh, lw = int(tokens.shape[1]), int(tokens.shape[2])
        w, h = lw * CELL, lh * CELL
        if canvas_size is None:
            canvas_size = (w, h)
            zoom = got.get("zoom")
        elif canvas_size != (w, h):
            raise SystemExit(f"REFUSED: --region {f.name} has canvas {w}x{h}, after {canvas_size[0]}x{canvas_size[1]}")
        start = int(got.get("first_frame", 0))
        trim = int(got.get("trim", 0))
        count = int(mask.shape[0])
        if names and start + trim != end:
            raise SystemExit(f"REFUSED: --region {f.name} starts its written frames on {start + trim}, after the previous window ended on {end}")
        if not names and start + trim > 0:
            end = start + trim
        lt = int(tokens.shape[0])
        gate = (vm.cut_gate(mask, lt, got["source"].get("cuts"), start) > 0.5).numpy()
        cells = (vm.pixel_alpha(tokens, lh, lw, 0) > 0.5).numpy()
        if tuple(mask.shape[1:]) != (h, w):
            mask_hw = torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="nearest")[:, 0]
        else:
            mask_hw = mask
        given_any = (mask_hw > 0.5).numpy().reshape(count, lh, CELL, lw, CELL).any(axis=(2, 4))
        # Use the node's own step-pooled `token_mask(grow(mask, margin))` to find cells carved out of the
        # grown region (`grown & ~(tokens > 0.5)`), matching `bench/stack_delivery.py::KeptSquares`.
        whole = (got["source"].get("edge") or vm.EDGE_TOKENS) == vm.EDGE_TOKENS
        grown_tok = vm.token_mask(vm.grow(mask, got["margin"]), lt, lh, lw, whole) > 0.5
        taken_tok = grown_tok & ~(tokens > 0.5)
        grown_cells = (vm.pixel_alpha(grown_tok.to(torch.float32), lh, lw, 0) > 0.5).numpy()
        taken_cells = (vm.pixel_alpha(taken_tok.to(torch.float32), lh, lw, 0) > 0.5).numpy()

        win_reg = np.zeros((count - trim, lh, lw), dtype=bool)
        win_subj = np.zeros((count - trim, lh, lw), dtype=bool)
        win_kept = np.zeros((count - trim, lh, lw), dtype=bool)
        for k in range(trim, count):
            empty = not bool(given_any[k].any())
            unlaid = empty and bool(cells[k].any())
            r_k = cells[k] & bool(gate[k]) & (not unlaid)
            win_reg[k - trim] = r_k
            win_subj[k - trim] = given_any[k] | grown_cells[k]
            win_kept[k - trim] = taken_cells[k] & bool(gate[k])
        reg_pieces.append(win_reg)
        subj_pieces.append(win_subj)
        kept_pieces.append(win_kept)
        end = start + count
        names.append(f.name)

    assert canvas_size is not None
    return {
        "region_cells": np.concatenate(reg_pieces, axis=0),
        "subject_cells": np.concatenate(subj_pieces, axis=0),
        "kept_carved_cells": np.concatenate(kept_pieces, axis=0),
        "canvas_size": canvas_size,
        "zoom": zoom,
        "files": names,
        "kind": "window_region",
        "inferred_first_frame": None,
    }


def read_luma_at_canvas(
    clip_path: str | Path,
    canvas_size: tuple[int, int],
    first_frame: int,
    count: int,
    zoom: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Read `count` frames from `clip_path` starting at `first_frame` onto `canvas_size` `(W, H)` as float32 BT.709 Y (`0..255`).

    Uses `yuv420p`'s Y plane directly (`bench/assemble_delivery.py::luma_at_canvas`), never ffmpeg's `gray`
    filter which expands limited-range 16..235 to 0..255. When `clip_path` is already at `canvas_size`,
    reads its pixels untouched (`vf=None`); when `clip_path` is at source resolution, fits it to the canvas
    using the lane's `LOADER_CROP` + bicubic scale (and crops to `zoom['box']` if the region was zoomed in
    and `clip_path` is the full-frame laid clip).
    """
    p = Path(clip_path)
    if not p.is_file():
        raise SystemExit(f"REFUSED: clip not found: {clip_path}")
    clip_w, clip_h, total = probe(p)
    if first_frame < 0 or first_frame >= total:
        raise SystemExit(f"REFUSED: frame {first_frame} is out of range for {p.name} ({total} frames, 0..{total - 1})")
    if first_frame + count > total:
        raise SystemExit(
            f"REFUSED: requested {count} frames from frame {first_frame} ({first_frame}..{first_frame + count - 1}), "
            f"but {p.name} holds {total} frames (0..{total - 1})"
        )
    w, h = canvas_size
    if (clip_w, clip_h) == (w, h):
        vf = None
        fit_how = f"native {w}x{h} (no scaling)"
    elif zoom and zoom.get("box") and zoom.get("source_size"):
        x0, y0, x1, y1 = [int(v) for v in zoom["box"]]
        sw, sh = [int(v) for v in zoom["source_size"]]
        vf = _loader_fit(sw, sh) + f",crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={w}:{h}:flags=bicubic"
        fit_how = f"fitted {clip_w}x{clip_h} -> {sw}x{sh}, cropped to zoom box {[x0, y0, x1, y1]}, scaled to {w}x{h}"
    else:
        vf = _loader_fit(w, h)
        fit_how = f"fitted {clip_w}x{clip_h} -> {w}x{h} ({vf})"

    filters = ([f"select='gte(n\\,{first_frame})'"] if first_frame else []) + ([vf] if vf else [])
    cmd = [
        "ffmpeg", "-v", "error", "-an", "-i", str(p),
        *(["-vf", ",".join(filters)] if filters else []),
        "-frames:v", str(count),
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ]
    frame_bytes = w * h * 3 // 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
    assert proc.stdout is not None and proc.stderr is not None
    frames: list[np.ndarray] = []
    try:
        while len(frames) < count:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(np.frombuffer(buf, np.uint8, w * h).astype(np.float32).reshape(h, w))
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
    if len(frames) != count:
        raise SystemExit(f"REFUSED: expected {count} frames from {p.name}, decoded {len(frames)}")
    luma = np.stack(frames, axis=0)
    return luma, {"clip_size": [clip_w, clip_h], "clip_total_frames": total, "canvas_size": [w, h], "fit": fit_how}


def build_measured_band(
    region_cells: np.ndarray,
    subject_cells: np.ndarray | None = None,
    kept_carved_cells: np.ndarray | None = None,
    band_cells: int = DEFAULT_BAND_CELLS,
) -> np.ndarray:
    """Return `[F, lh, lw]` bool mask of cells inside the subject's region, its +/-1 latent-step temporal
    union, `subject_cells` / `kept_carved_cells`, and a spatial band of `band_cells` 16x16 squares around them.
    """
    f_count, lh, lw = region_cells.shape
    base = region_cells.copy()
    if subject_cells is not None and subject_cells.shape == base.shape:
        base |= subject_cells
    if kept_carved_cells is not None and kept_carved_cells.shape == base.shape:
        base |= kept_carved_cells

    # Include cells active within +/- TEMPORAL_REACH_FRAMES so a square carved out on a single latent step
    # stays inside the measured zone on the frames where it was carved out.
    temporal = base.copy()
    for f in range(f_count):
        lo = max(0, f - TEMPORAL_REACH_FRAMES)
        hi = min(f_count, f + TEMPORAL_REACH_FRAMES + 1)
        temporal[f] = base[lo:hi].any(axis=0)

    if band_cells <= 0:
        return temporal
    ksize = 2 * int(band_cells) + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    out = np.zeros_like(temporal)
    for f in range(f_count):
        if temporal[f].any():
            out[f] = cv2.dilate(temporal[f].astype(np.uint8), kernel).astype(bool)
    return out


def _grid_jumps(luma: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-segment `grid_jump`, `side_jump`, `grid_excess`, `grid_ratio`, and `wall_cov` across all 16 px
    vertical and horizontal cell boundaries of `luma` (`[F, H, W]`, `H = lh * 16`, `W = lw * 16`).

    - Vertical boundaries sit at `x = 16 * c` for `c in 1..lw-1` along each cell row `r in 0..lh-1`
      (`y in 16*r .. 16*(r+1)`), shape `[F, lh, lw - 1]`.
    - Horizontal boundaries sit at `y = 16 * r` for `r in 1..lh-1` along each cell col `c in 0..lw-1`
      (`x in 16*c .. 16*(c+1)`), shape `[F, lh - 1, lw]`.
    """
    f_count, h, w = luma.shape
    lh, lw = h // CELL, w // CELL

    # Vertical grid lines at x = 16*c (c = 1..lw-1): jump between x = 16*c - 1 and x = 16*c
    v_grid_px = np.abs(luma[:, :, CELL::CELL] - luma[:, :, CELL - 1 : w - 1 : CELL]).reshape(f_count, lh, CELL, lw - 1)
    v_left_px = np.abs(luma[:, :, CELL - 1 : w - 1 : CELL] - luma[:, :, CELL - 2 : w - 2 : CELL]).reshape(f_count, lh, CELL, lw - 1)
    v_right_px = np.abs(luma[:, :, CELL + 1 :: CELL] - luma[:, :, CELL::CELL]).reshape(f_count, lh, CELL, lw - 1)
    v_side_px = 0.5 * (v_left_px + v_right_px)
    v_grid = v_grid_px.mean(axis=2)
    v_side = v_side_px.mean(axis=2)
    v_excess = v_grid - v_side
    v_ratio = (v_grid + 1.0) / (v_side + 1.0)
    v_cov = ((v_grid_px - v_side_px >= DIFF_GRID_EXCESS_MIN) & (v_grid_px >= 1.5 * v_side_px)).mean(axis=2)

    # Horizontal grid lines at y = 16*r (r = 1..lh-1): jump between y = 16*r - 1 and y = 16*r
    h_grid_px = np.abs(luma[:, CELL::CELL, :] - luma[:, CELL - 1 : h - 1 : CELL, :]).reshape(f_count, lh - 1, lw, CELL)
    h_top_px = np.abs(luma[:, CELL - 1 : h - 1 : CELL, :] - luma[:, CELL - 2 : h - 2 : CELL, :]).reshape(f_count, lh - 1, lw, CELL)
    h_bot_px = np.abs(luma[:, CELL + 1 :: CELL, :] - luma[:, CELL::CELL, :]).reshape(f_count, lh - 1, lw, CELL)
    h_side_px = 0.5 * (h_top_px + h_bot_px)
    h_grid = h_grid_px.mean(axis=3)
    h_side = h_side_px.mean(axis=3)
    h_excess = h_grid - h_side
    h_ratio = (h_grid + 1.0) / (h_side + 1.0)
    h_cov = ((h_grid_px - h_side_px >= DIFF_GRID_EXCESS_MIN) & (h_grid_px >= 1.5 * h_side_px)).mean(axis=3)

    return {
        "v_grid": v_grid,
        "v_side": v_side,
        "v_excess": v_excess,
        "v_ratio": v_ratio,
        "v_cov": v_cov,
        "h_grid": h_grid,
        "h_side": h_side,
        "h_excess": h_excess,
        "h_ratio": h_ratio,
        "h_cov": h_cov,
    }


def _collinear_run_lengths_vertical(active_v: np.ndarray) -> np.ndarray:
    """For `active_v` `[lh, lw - 1]` bool, return `[lh, lw - 1]` int run length of consecutive True along `r`."""
    lh, nw = active_v.shape
    runs = np.zeros((lh, nw), dtype=np.int32)
    for c in range(nw):
        r = 0
        while r < lh:
            if not active_v[r, c]:
                r += 1
                continue
            r2 = r
            while r2 < lh and active_v[r2, c]:
                r2 += 1
            runs[r:r2, c] = r2 - r
            r = r2
    return runs


def _collinear_run_lengths_horizontal(active_h: np.ndarray) -> np.ndarray:
    """For `active_h` `[lh - 1, lw]` bool, return `[lh - 1, lw]` int run length of consecutive True along `c`."""
    nh, lw = active_h.shape
    runs = np.zeros((nh, lw), dtype=np.int32)
    for r in range(nh):
        c = 0
        while c < lw:
            if not active_h[r, c]:
                c += 1
                continue
            c2 = c
            while c2 < lw and active_h[r, c2]:
                c2 += 1
            runs[r, c:c2] = c2 - c
            c = c2
    return runs


def _neighbour_median_step(cell_step: np.ndarray) -> np.ndarray:
    """For `cell_step` `[F, lh, lw]`, return the median of the 8 surrounding cells' `cell_step` at each `(f, r, c)`."""
    f_count, lh, lw = cell_step.shape
    padded = np.pad(cell_step, ((0, 0), (1, 1), (1, 1)), mode="edge")
    shifts = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifts.append(padded[:, 1 + dr : 1 + dr + lh, 1 + dc : 1 + dc + lw])
    stack = np.stack(shifts, axis=0)  # [8, F, lh, lw]
    return np.median(stack, axis=0)


def measure_luma_blocks(
    luma: np.ndarray,
    region_cells: np.ndarray,
    subject_cells: np.ndarray | None = None,
    kept_carved_cells: np.ndarray | None = None,
    source_luma: np.ndarray | None = None,
    first_frame: int = 0,
    band_cells: int = DEFAULT_BAND_CELLS,
) -> dict:
    """Measure spatial 16 px grid jumps and across-frame cell steps on `luma` `[F, H, W]` (`0..255` float32).

    Separates:
      - `squares_found`: 16x16 cells in the measured band that stand out as pasted/regenerated/kept blocks
        either spatially (`>= 2` cell walls with sharp 16 px grid excess not explained by a long scene line
        or `--source`) or temporally (`step_excess >= STEP_EXCESS_MIN` and `step_ratio >= STEP_RATIO_MIN`
        over the median of its 8 neighbours, with a 16 px boundary jump on the frame or temporal difference).
      - `real_edges_on_grid`: 16 px grid segments in the measured band with `grid_jump >= REAL_EDGE_JUMP_MIN`
        that are real edges in the picture rather than pasted 16x16 squares:
          * `gradient_across_grid`: multi-pixel scene edge where `side_jump` one pixel either side of the
            grid line is also high (`grid_ratio < GRID_RATIO_MIN` or `grid_excess < GRID_EXCESS_MIN` or
            `wall_cov < WALL_COVERAGE_MIN`),
          * `single_grid_line`: a sharp 1-pixel step along a single grid axis that does not close into a
            2-sided 16x16 square (or runs straight across `>= COLLINEAR_LINE_CELLS` cells without a parallel
            opposite cell wall 16 px away),
          * `present_in_source`: a grid step already present on the same 16 px segment in `--source`.
    """
    if luma.ndim != 3:
        raise ValueError(f"luma must be 3D [F, H, W], got shape {luma.shape}")
    f_count, h, w = luma.shape
    if h % CELL != 0 or w % CELL != 0:
        raise ValueError(f"canvas {w}x{h} must be a multiple of CELL={CELL}")
    lh, lw = h // CELL, w // CELL
    if region_cells.shape != (f_count, lh, lw):
        raise ValueError(f"region_cells shape {region_cells.shape} does not match luma cell grid {(f_count, lh, lw)}")

    band_mask = build_measured_band(region_cells, subject_cells, kept_carved_cells, band_cells=band_cells)
    kept_mask = kept_carved_cells if (kept_carved_cells is not None and kept_carved_cells.shape == region_cells.shape) else np.zeros_like(region_cells)

    jumps = _grid_jumps(luma)
    src_jumps = _grid_jumps(source_luma) if source_luma is not None else None

    # Active 1-px grid step segments on vertical [F, lh, lw-1] and horizontal [F, lh-1, lw] lines
    active_v = (jumps["v_excess"] >= GRID_EXCESS_MIN) & (jumps["v_ratio"] >= GRID_RATIO_MIN) & (jumps["v_cov"] >= WALL_COVERAGE_MIN)
    active_h = (jumps["h_excess"] >= GRID_EXCESS_MIN) & (jumps["h_ratio"] >= GRID_RATIO_MIN) & (jumps["h_cov"] >= WALL_COVERAGE_MIN)

    if src_jumps is not None:
        src_active_v = (src_jumps["v_excess"] >= GRID_EXCESS_MIN * 0.8) & (
            np.abs(jumps["v_excess"] - src_jumps["v_excess"]) < GRID_EXCESS_MIN
        )
        src_active_h = (src_jumps["h_excess"] >= GRID_EXCESS_MIN * 0.8) & (
            np.abs(jumps["h_excess"] - src_jumps["h_excess"]) < GRID_EXCESS_MIN
        )
    else:
        src_active_v = np.zeros_like(active_v)
        src_active_h = np.zeros_like(active_h)

    # Temporal steps across frames (f - 1 -> f)
    cell_step = np.zeros((f_count, lh, lw), dtype=np.float32)
    neighbour_step = np.zeros((f_count, lh, lw), dtype=np.float32)
    step_excess = np.zeros((f_count, lh, lw), dtype=np.float32)
    step_ratio = np.ones((f_count, lh, lw), dtype=np.float32)
    diff_active_sides = np.zeros((f_count, lh, lw), dtype=np.int32)

    if f_count > 1:
        abs_diff = np.abs(luma[1:] - luma[:-1])  # [F - 1, H, W]
        cs = abs_diff.reshape(f_count - 1, lh, CELL, lw, CELL).mean(axis=(2, 4))
        ns = _neighbour_median_step(cs)
        cell_step[1:] = cs
        neighbour_step[1:] = ns
        step_excess[1:] = cs - ns
        step_ratio[1:] = (cs + 1.0) / (ns + 1.0)

        diff_jumps = _grid_jumps(abs_diff)
        dv_act = (diff_jumps["v_excess"] >= GRID_EXCESS_MIN) & (diff_jumps["v_ratio"] >= GRID_RATIO_MIN) & (diff_jumps["v_cov"] >= WALL_COVERAGE_MIN)
        dh_act = (diff_jumps["h_excess"] >= GRID_EXCESS_MIN) & (diff_jumps["h_ratio"] >= GRID_RATIO_MIN) & (diff_jumps["h_cov"] >= WALL_COVERAGE_MIN)
        dl_a = np.pad(dv_act, ((0, 0), (0, 0), (1, 0)), constant_values=False)
        dr_a = np.pad(dv_act, ((0, 0), (0, 0), (0, 1)), constant_values=False)
        dt_a = np.pad(dh_act, ((0, 0), (1, 0), (0, 0)), constant_values=False)
        db_a = np.pad(dh_act, ((0, 0), (0, 1), (0, 0)), constant_values=False)
        diff_active_sides[1:] = dl_a.astype(np.int32) + dr_a.astype(np.int32) + dt_a.astype(np.int32) + db_a.astype(np.int32)

    squares_found: list[dict] = []
    real_edges_on_grid: list[dict] = []
    per_frame: list[dict] = []

    for f in range(f_count):
        src_f = first_frame + f
        bm = band_mask[f]

        # Identify long collinear straight lines on frame f that have no parallel opposite wall 16 px away
        v_runs = _collinear_run_lengths_vertical(active_v[f])
        h_runs = _collinear_run_lengths_horizontal(active_h[f])

        # A vertical grid line at c (between col c-1 and col c) has a parallel wall 16 px away if c-1 or c+1 is also active at row r
        v_has_parallel = np.zeros_like(active_v[f])
        if lw - 1 > 1:
            v_has_parallel[:, :-1] |= active_v[f, :, 1:]
            v_has_parallel[:, 1:] |= active_v[f, :, :-1]
        v_long_line = (v_runs >= COLLINEAR_LINE_CELLS) & ~v_has_parallel

        h_has_parallel = np.zeros_like(active_h[f])
        if lh - 1 > 1:
            h_has_parallel[:-1, :] |= active_h[f, 1:, :]
            h_has_parallel[1:, :] |= active_h[f, :-1, :]
        h_long_line = (h_runs >= COLLINEAR_LINE_CELLS) & ~h_has_parallel

        # Effective square-wall active flags: active, not in source, and not an unclosed long scene line
        sq_v = active_v[f] & ~src_active_v[f] & ~v_long_line
        sq_h = active_h[f] & ~src_active_h[f] & ~h_long_line

        # Per-cell 4 sides (left, right, top, bottom)
        cell_left_ex = np.zeros((lh, lw), dtype=np.float32)
        cell_right_ex = np.zeros((lh, lw), dtype=np.float32)
        cell_top_ex = np.zeros((lh, lw), dtype=np.float32)
        cell_bot_ex = np.zeros((lh, lw), dtype=np.float32)

        cell_left_ex[:, 1:] = jumps["v_excess"][f]
        cell_right_ex[:, :-1] = jumps["v_excess"][f]
        cell_top_ex[1:, :] = jumps["h_excess"][f]
        cell_bot_ex[:-1, :] = jumps["h_excess"][f]

        cell_left_gr = np.zeros((lh, lw), dtype=np.float32)
        cell_right_gr = np.zeros((lh, lw), dtype=np.float32)
        cell_top_gr = np.zeros((lh, lw), dtype=np.float32)
        cell_bot_gr = np.zeros((lh, lw), dtype=np.float32)

        cell_left_gr[:, 1:] = jumps["v_grid"][f]
        cell_right_gr[:, :-1] = jumps["v_grid"][f]
        cell_top_gr[1:, :] = jumps["h_grid"][f]
        cell_bot_gr[:-1, :] = jumps["h_grid"][f]

        cell_left_sd = np.zeros((lh, lw), dtype=np.float32)
        cell_right_sd = np.zeros((lh, lw), dtype=np.float32)
        cell_top_sd = np.zeros((lh, lw), dtype=np.float32)
        cell_bot_sd = np.zeros((lh, lw), dtype=np.float32)

        cell_left_sd[:, 1:] = jumps["v_side"][f]
        cell_right_sd[:, :-1] = jumps["v_side"][f]
        cell_top_sd[1:, :] = jumps["h_side"][f]
        cell_bot_sd[:-1, :] = jumps["h_side"][f]

        s_left = np.zeros((lh, lw), dtype=bool)
        s_right = np.zeros((lh, lw), dtype=bool)
        s_top = np.zeros((lh, lw), dtype=bool)
        s_bot = np.zeros((lh, lw), dtype=bool)

        s_left[:, 1:] = sq_v
        s_right[:, :-1] = sq_v
        s_top[1:, :] = sq_h
        s_bot[:-1, :] = sq_h

        active_sides_count = s_left.astype(np.int32) + s_right.astype(np.int32) + s_top.astype(np.int32) + s_bot.astype(np.int32)
        max_side_excess = np.maximum.reduce([cell_left_ex, cell_right_ex, cell_top_ex, cell_bot_ex])
        mean_side_excess = 0.25 * (cell_left_ex + cell_right_ex + cell_top_ex + cell_bot_ex)
        max_side_grid = np.maximum.reduce([cell_left_gr, cell_right_gr, cell_top_gr, cell_bot_gr])
        mean_side_side = 0.25 * (cell_left_sd + cell_right_sd + cell_top_sd + cell_bot_sd)

        # Spatial square criterion: inside measured band and >= 2 active square sides
        spatial_hit = bm & (active_sides_count >= 2)
        # Temporal square criterion: inside measured band, strong step_excess & step_ratio over 8 neighbours,
        # and either >= 2 active 16 px square walls on |Y_f - Y_{f-1}| (with a 16 px wall on f or f-1)
        # or a kept_carved cell stepping against its regenerated neighbours.
        prev_active_sides = np.zeros((lh, lw), dtype=np.int32)
        if f > 0:
            prev_sq_v = active_v[f - 1] & ~src_active_v[f - 1]
            prev_sq_h = active_h[f - 1] & ~src_active_h[f - 1]
            prev_active_sides[:, 1:] += prev_sq_v.astype(np.int32)
            prev_active_sides[:, :-1] += prev_sq_v.astype(np.int32)
            prev_active_sides[1:, :] += prev_sq_h.astype(np.int32)
            prev_active_sides[:-1, :] += prev_sq_h.astype(np.int32)
        sharp_temporal_box = (diff_active_sides[f] >= 2) & ((active_sides_count >= 1) | (prev_active_sides >= 1))
        temporal_hit = (
            bm
            & (step_excess[f] >= STEP_EXCESS_MIN)
            & (step_ratio[f] >= STEP_RATIO_MIN)
            & (sharp_temporal_box | kept_mask[f])
        )

        hit_cells = spatial_hit | temporal_hit
        frame_squares: list[dict] = []

        for r, c in np.argwhere(hit_cells):
            r_i, c_i = int(r), int(c)
            sides = []
            if s_top[r_i, c_i]:
                sides.append("top")
            if s_bot[r_i, c_i]:
                sides.append("bottom")
            if s_left[r_i, c_i]:
                sides.append("left")
            if s_right[r_i, c_i]:
                sides.append("right")
            if spatial_hit[r_i, c_i] and temporal_hit[r_i, c_i]:
                kind = "spatial_and_temporal"
            elif spatial_hit[r_i, c_i]:
                kind = "spatial_grid"
            else:
                kind = "temporal_step"
            sq_entry = {
                "frame": f,
                "source_frame": src_f,
                "cell": [r_i, c_i],
                "box_px": [c_i * CELL, r_i * CELL, (c_i + 1) * CELL, (r_i + 1) * CELL],
                "kind": kind,
                "in_region": bool(region_cells[f, r_i, c_i]),
                "in_kept_carved": bool(kept_mask[f, r_i, c_i]),
                "active_sides": sides,
                "active_sides_count": int(active_sides_count[r_i, c_i]),
                "max_grid_jump": round(float(max_side_grid[r_i, c_i]), 3),
                "mean_side_jump": round(float(mean_side_side[r_i, c_i]), 3),
                "max_grid_excess": round(float(max_side_excess[r_i, c_i]), 3),
                "mean_grid_excess": round(float(mean_side_excess[r_i, c_i]), 3),
                "cell_step": round(float(cell_step[f, r_i, c_i]), 3),
                "neighbour_step": round(float(neighbour_step[f, r_i, c_i]), 3),
                "step_excess": round(float(step_excess[f, r_i, c_i]), 3),
                "step_ratio": round(float(step_ratio[f, r_i, c_i]), 3),
            }
            frame_squares.append(sq_entry)
            squares_found.append(sq_entry)

        # Real edges on the 16 px grid inside the measured band (not part of a detected spatial square's active walls)
        frame_real_edges: list[dict] = []
        # Check vertical grid segments at x = 16*c (c = 1..lw-1)
        v_in_band = bm[:, :-1] | bm[:, 1:]
        v_high_jump = v_in_band & (jumps["v_grid"][f] >= REAL_EDGE_JUMP_MIN)
        for r, c_idx in np.argwhere(v_high_jump):
            r_i, c_i = int(r), int(c_idx)
            # Segment lies between (r_i, c_i) and (r_i, c_i + 1) at x = (c_i + 1) * 16
            if sq_v[r_i, c_i] and (spatial_hit[r_i, c_i] or spatial_hit[r_i, c_i + 1]):
                continue
            g_jump = float(jumps["v_grid"][f, r_i, c_i])
            s_jump = float(jumps["v_side"][f, r_i, c_i])
            g_ex = float(jumps["v_excess"][f, r_i, c_i])
            g_rat = float(jumps["v_ratio"][f, r_i, c_i])
            if src_active_v[f, r_i, c_i]:
                why = "present_in_source"
            elif not active_v[f, r_i, c_i]:
                why = "gradient_across_grid"
            else:
                why = "single_grid_line"
            edge_entry = {
                "frame": f,
                "source_frame": src_f,
                "axis": "vertical",
                "grid_px": (c_i + 1) * CELL,
                "span_px": [r_i * CELL, (r_i + 1) * CELL],
                "cell_pair": [[r_i, c_i], [r_i, c_i + 1]],
                "grid_jump": round(g_jump, 3),
                "side_jump": round(s_jump, 3),
                "grid_excess": round(g_ex, 3),
                "grid_ratio": round(g_rat, 3),
                "collinear_cells": int(v_runs[r_i, c_i]),
                "why": why,
            }
            frame_real_edges.append(edge_entry)
            real_edges_on_grid.append(edge_entry)

        # Check horizontal grid segments at y = 16*r (r = 1..lh-1)
        h_in_band = bm[:-1, :] | bm[1:, :]
        h_high_jump = h_in_band & (jumps["h_grid"][f] >= REAL_EDGE_JUMP_MIN)
        for r_idx, c in np.argwhere(h_high_jump):
            r_i, c_i = int(r_idx), int(c)
            # Segment lies between (r_i, c_i) and (r_i + 1, c_i) at y = (r_i + 1) * 16
            if sq_h[r_i, c_i] and (spatial_hit[r_i, c_i] or spatial_hit[r_i + 1, c_i]):
                continue
            g_jump = float(jumps["h_grid"][f, r_i, c_i])
            s_jump = float(jumps["h_side"][f, r_i, c_i])
            g_ex = float(jumps["h_excess"][f, r_i, c_i])
            g_rat = float(jumps["h_ratio"][f, r_i, c_i])
            if src_active_h[f, r_i, c_i]:
                why = "present_in_source"
            elif not active_h[f, r_i, c_i]:
                why = "gradient_across_grid"
            else:
                why = "single_grid_line"
            edge_entry = {
                "frame": f,
                "source_frame": src_f,
                "axis": "horizontal",
                "grid_px": (r_i + 1) * CELL,
                "span_px": [c_i * CELL, (c_i + 1) * CELL],
                "cell_pair": [[r_i, c_i], [r_i + 1, c_i]],
                "grid_jump": round(g_jump, 3),
                "side_jump": round(s_jump, 3),
                "grid_excess": round(g_ex, 3),
                "grid_ratio": round(g_rat, 3),
                "collinear_cells": int(h_runs[r_i, c_i]),
                "why": why,
            }
            frame_real_edges.append(edge_entry)
            real_edges_on_grid.append(edge_entry)

        bm_max_ex = float(max_side_excess[bm].max()) if bm.any() else 0.0
        bm_max_step = float(step_excess[f][bm].max()) if bm.any() else 0.0
        per_frame.append({
            "frame": f,
            "source_frame": src_f,
            "region_cells": int(region_cells[f].sum()),
            "measured_band_cells": int(bm.sum()),
            "kept_carved_cells": int(kept_mask[f].sum()),
            "max_grid_excess": round(bm_max_ex, 3),
            "max_step_excess": round(bm_max_step, 3),
            "squares_count": len(frame_squares),
            "real_edges_on_grid_count": len(frame_real_edges),
            "squares": frame_squares,
            "real_edges_on_grid": frame_real_edges,
        })

    frames_with_squares = sorted({sq["source_frame"] for sq in squares_found})
    frames_with_real_edges = sorted({ed["source_frame"] for ed in real_edges_on_grid})

    return {
        "cell_px": CELL,
        "band_cells": int(band_cells),
        "constants": {
            "GRID_EXCESS_MIN": GRID_EXCESS_MIN,
            "GRID_RATIO_MIN": GRID_RATIO_MIN,
            "REAL_EDGE_JUMP_MIN": REAL_EDGE_JUMP_MIN,
            "COLLINEAR_LINE_CELLS": COLLINEAR_LINE_CELLS,
            "STEP_EXCESS_MIN": STEP_EXCESS_MIN,
            "STEP_RATIO_MIN": STEP_RATIO_MIN,
            "DIFF_GRID_EXCESS_MIN": DIFF_GRID_EXCESS_MIN,
            "WALL_COVERAGE_MIN": WALL_COVERAGE_MIN,
        },
        "summary": {
            "frames_measured": f_count,
            "source_frame_span": [first_frame, first_frame + f_count - 1] if f_count else [],
            "squares_found_count": len(squares_found),
            "frames_with_squares": frames_with_squares,
            "real_edges_on_grid_count": len(real_edges_on_grid),
            "frames_with_real_edges_on_grid": frames_with_real_edges,
            "max_grid_excess": round(max((r["max_grid_excess"] for r in per_frame), default=0.0), 3),
            "max_step_excess": round(max((r["max_step_excess"] for r in per_frame), default=0.0), 3),
        },
        "squares_found": squares_found,
        "real_edges_on_grid": real_edges_on_grid,
        "per_frame": per_frame,
        "what_measured": (
            f"BT.709 Y luma (0..255) on {w}x{h} canvas ({lw}x{lh} grid of {CELL}x{CELL} px cells) over {f_count} "
            f"frame(s) (source frames {first_frame}..{first_frame + f_count - 1}), inside the saved token region "
            f"and a {band_cells}-cell ({band_cells * CELL} px) spatial band around it (plus +/-{TEMPORAL_REACH_FRAMES} "
            "frames of latent-step reach)."
        ),
        "cannot_see": (
            "Chroma-only steps at constant BT.709 Y luma; blocks wholly outside the subject's region and its "
            f"{band_cells}-cell band; sub-cell seams feathered wider than 2 px across a 16 px grid line where "
            "side_jump equals grid_jump; whether a stepping square looks acceptable in motion to a person."
        ),
    }


def format_report(report: dict) -> str:
    """Format a human-readable summary of `report` for stdout (data only, no verdict)."""
    s = report["summary"]
    span = f"{s['source_frame_span'][0]}-{s['source_frame_span'][1]}" if s["source_frame_span"] else "empty"
    lines = [
        f"measure_blocks: clip={report.get('clip', '?')} region={', '.join(report.get('region_files', []))}",
        f"  measured_on: {report['what_measured']}",
        f"  cannot_see:  {report['cannot_see']}",
        f"  span={span} ({s['frames_measured']} frames)  "
        f"squares_found={s['squares_found_count']} on {len(s['frames_with_squares'])} frame(s) {s['frames_with_squares']}  "
        f"real_edges_on_grid={s['real_edges_on_grid_count']} on {len(s['frames_with_real_edges_on_grid'])} frame(s)  "
        f"max_grid_excess={s['max_grid_excess']:.2f}  max_step_excess={s['max_step_excess']:.2f}",
    ]
    if report["squares_found"]:
        lines.append("  squares (top 20 by max(max_grid_excess, step_excess)):")
        top = sorted(
            report["squares_found"],
            key=lambda x: max(x["max_grid_excess"], x["step_excess"]),
            reverse=True,
        )[:20]
        for sq in top:
            lines.append(
                f"    source_frame={sq['source_frame']} (f={sq['frame']}) cell=({sq['cell'][0]},{sq['cell'][1]}) "
                f"box={sq['box_px']} kind={sq['kind']} in_region={sq['in_region']} kept_carved={sq['in_kept_carved']} "
                f"sides={sq['active_sides']} max_grid_ex={sq['max_grid_excess']:.2f} "
                f"step_ex={sq['step_excess']:.2f} (cell_step={sq['cell_step']:.2f} vs neigh={sq['neighbour_step']:.2f})"
            )
    if report["real_edges_on_grid"]:
        lines.append("  real_edges_on_grid (top 10 by grid_jump):")
        top_e = sorted(report["real_edges_on_grid"], key=lambda x: x["grid_jump"], reverse=True)[:10]
        for ed in top_e:
            lines.append(
                f"    source_frame={ed['source_frame']} (f={ed['frame']}) {ed['axis']}@{ed['grid_px']}px "
                f"span={ed['span_px']} why={ed['why']} grid_jump={ed['grid_jump']:.2f} "
                f"side_jump={ed['side_jump']:.2f} grid_ex={ed['grid_excess']:.2f} ratio={ed['grid_ratio']:.2f}"
            )
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description="Measure visible 16 px blocks on a laid clip or render piece and a window region file."
    )
    ap.add_argument("--clip", required=True, help="Laid clip or render piece video path")
    ap.add_argument(
        "--region",
        required=True,
        nargs="+",
        help="One or more *_window_*_region.npz files, a windows directory, or a capture region.npz",
    )
    ap.add_argument(
        "--first-frame",
        type=int,
        default=None,
        help="Source frame number of frame 0 of --clip (default: 0, or inferred from capture manifest)",
    )
    ap.add_argument(
        "--region-first",
        type=int,
        default=None,
        help="Source frame number of frame 0 of --region (default: same as --first-frame, or inferred from capture)",
    )
    ap.add_argument("--only", default=None, help="Optional inclusive source frame range FIRST-LAST to measure")
    ap.add_argument(
        "--band-cells",
        type=int,
        default=DEFAULT_BAND_CELLS,
        help=f"Band of 16x16 cells around the region to include (default: {DEFAULT_BAND_CELLS})",
    )
    ap.add_argument("--source", default=None, help="Optional source video to identify grid edges already in the source")
    ap.add_argument("--source-first", type=int, default=0, help="Source frame number of frame 0 of --source (default: 0)")
    ap.add_argument("--out", default=None, help="Optional JSON path to write the full report")
    args = ap.parse_args(argv)

    if args.band_cells < 0:
        raise SystemExit(f"REFUSED: --band-cells must be >= 0, got {args.band_cells}")

    reg_data = load_region_data(args.region)
    reg_cells = reg_data["region_cells"]
    subj_cells = reg_data["subject_cells"]
    kept_cells = reg_data["kept_carved_cells"]
    canvas_size = reg_data["canvas_size"]
    zoom = reg_data["zoom"]
    reg_total = int(reg_cells.shape[0])

    clip_p = Path(args.clip)
    if not clip_p.is_file():
        raise SystemExit(f"REFUSED: --clip not found: {args.clip}")
    _, _, clip_total = probe(clip_p)

    inferred = reg_data["inferred_first_frame"]
    clip_first = args.first_frame if args.first_frame is not None else (inferred if inferred is not None else 0)
    reg_first = args.region_first if args.region_first is not None else (inferred if inferred is not None else clip_first)

    # Determine overlapping source frame span [span_lo, span_hi]
    span_lo = max(clip_first, reg_first)
    span_hi = min(clip_first + clip_total - 1, reg_first + reg_total - 1)
    only = _parse_only(args.only)
    if only is not None:
        span_lo = max(span_lo, only[0])
        span_hi = min(span_hi, only[1])
    if span_hi < span_lo:
        raise SystemExit(
            f"REFUSED: no overlapping frames between --clip ({clip_first}..{clip_first + clip_total - 1}), "
            f"--region ({reg_first}..{reg_first + reg_total - 1})"
            + (f", and --only {args.only}" if args.only else "")
        )

    count = span_hi - span_lo + 1
    clip_offset = span_lo - clip_first
    reg_offset = span_lo - reg_first

    luma, clip_info = read_luma_at_canvas(clip_p, canvas_size, first_frame=clip_offset, count=count, zoom=zoom)
    src_luma = None
    if args.source:
        src_offset = span_lo - int(args.source_first)
        src_luma, _ = read_luma_at_canvas(args.source, canvas_size, first_frame=src_offset, count=count, zoom=zoom)

    res = measure_luma_blocks(
        luma=luma,
        region_cells=reg_cells[reg_offset : reg_offset + count],
        subject_cells=subj_cells[reg_offset : reg_offset + count],
        kept_carved_cells=kept_cells[reg_offset : reg_offset + count],
        source_luma=src_luma,
        first_frame=span_lo,
        band_cells=args.band_cells,
    )
    report = {
        "clip": home_relative(clip_p.resolve()),
        "clip_info": clip_info,
        "region_files": reg_data["files"],
        "region_kind": reg_data["kind"],
        **({"zoom": zoom} if zoom else {}),
        **res,
    }
    print(format_report(report), flush=True)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    run_cli()
