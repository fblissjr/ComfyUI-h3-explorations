#!/usr/bin/env python3
"""Measure per-frame metric depth and camera intrinsics on video frames with HyDen MoGe-2.

    <comfy venv python> bench/measure_moge2_depth.py --source VIDEO.mov --frames 755,764,772,780 \\
        --out-dir data/fun/moge2/span_0755_0780 [--box x0,y0,x1,y1] [--size 1024x768] \\
        [--region NAME=MASK.mkv[@FIRST][:COLOUR]] [--device cpu]

Reads chosen source frame numbers via `bench/_lib/frames.py::stream` (fitted to `--size` as the
lane's video loader fits a source, or scaled when `--no-fit` is given), runs `HyDenMoGe` on raw
`[0, 1]` `float32` RGB frames, and writes to `--out-dir`:
- `depth.npz`: `frames`, `depth` `(N, H, W)` in meters, `mask` `(N, H, W)` bool, `intrinsics`
  `(N, 3, 3)` float32, `fov_x_deg` `(N,)`, `fov_y_deg` `(N,)`, `box` `[x0, y0, x1, y1]`.
- `record.json`: model metadata (`input_range: [0, 1]`), source, canvas, box, per-frame intrinsics
  and metric depth statistics inside `box` (plus any named `--region` masks), and shot-level
  intrinsics variation summary (`fov_x_deg_least_median_most`, `fov_x_deg_spread`,
  `fov_x_deg_std`, `fx_least_median_most`).
- `sheet.jpg`: side-by-side visual frame sheet with each frame's RGB on the left (`box` and any
  named `--region` masks outlined) and inverse-depth Turbo colormap on the right (near=warm/bright,
  far=cool/dark, with metric depth legend and per-region median depth in meters printed on every row).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, home_relative  # noqa: E402
from _lib.frames import FIT, probe, read_mask, stream  # noqa: E402

sys.path.insert(0, str(REPO))
from vendor.meta_moge2.hyden_moge2 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    HyDenMoGe,
    fov_deg_from_intrinsics,
    load_hyden_moge2,
)

MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
HEADER_H = 28
COLOURS: dict[str, tuple[int, int, int]] = {
    "cyan": (60, 220, 255),
    "yellow": (255, 235, 0),
    "magenta": (255, 70, 220),
    "green": (60, 255, 90),
    "red": (255, 60, 60),
    "white": (255, 255, 255),
}
COLOUR_ORDER = ("cyan", "yellow", "magenta", "green", "red", "white")


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(f"REFUSED: --size must be WxH (e.g. 1024x768), got {text!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"REFUSED: --size must hold integer width and height, got {text!r}") from None
    if w < 14 or h < 14:
        raise SystemExit(f"REFUSED: --size must be at least 14x14, got {w}x{h}")
    return w, h


def _parse_frames(text: str, total_frames: int) -> list[int]:
    if not text or not text.strip():
        raise SystemExit("REFUSED: --frames must list at least one frame number")
    frames: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            f = int(piece)
        except ValueError:
            raise SystemExit(f"REFUSED: invalid frame number {piece!r} in --frames") from None
        if f < 0 or f >= total_frames:
            raise SystemExit(
                f"REFUSED: frame {f} is out of range for source with {total_frames} frames (0..{total_frames - 1})"
            )
        if f not in frames:
            frames.append(f)
    if not frames:
        raise SystemExit("REFUSED: --frames resolved to an empty list")
    return sorted(frames)


def _parse_box(text: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not text:
        return 0, 0, width, height
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"REFUSED: --box must be x0,y0,x1,y1 (4 integers), got {text!r}")
    try:
        x0, y0, x1, y1 = (int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"REFUSED: --box coordinates must be integers, got {text!r}") from None
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 - x0 < 14 or y1 - y0 < 14:
        raise SystemExit(
            f"REFUSED: --box {x0},{y0},{x1},{y1} is outside canvas {width}x{height} or smaller than 14x14"
        )
    return x0, y0, x1, y1


def _parse_region(spec: str, idx: int) -> tuple[str, Path, int, str]:
    """Parse `NAME=MASK.mkv[@FIRST][:COLOUR]`."""
    if "=" not in spec:
        raise SystemExit(f"REFUSED: --region must be NAME=MASK[@FIRST][:COLOUR], got {spec!r}")
    name, _, rest = spec.partition("=")
    name = name.strip()
    if not name:
        raise SystemExit(f"REFUSED: empty region name in {spec!r}")
    colour_name = COLOUR_ORDER[idx % len(COLOUR_ORDER)]
    for cand in COLOURS:
        if rest.endswith(f":{cand}"):
            colour_name = cand
            rest = rest[: -len(f":{cand}")]
            break
    path_str, _, at_str = rest.partition("@")
    try:
        at = int(at_str) if at_str else 0
    except ValueError:
        raise SystemExit(f"REFUSED: invalid @FIRST frame offset in --region {spec!r}") from None
    path = Path(path_str)
    if not path.is_file():
        raise SystemExit(f"REFUSED: region mask file not found: {home_relative(path)}")
    return name, path, at, colour_name


def _grab_frames(
    source: Path,
    frames: list[int],
    size: tuple[int, int],
    fit: bool,
) -> dict[int, np.ndarray]:
    w, h = size
    pick = "select=" + "+".join(f"eq(n\\,{f})" for f in frames)
    vf = pick + "," + (FIT.format(w=w, h=h) if fit else f"scale={w}:{h}")
    got = {f: img.copy() for f, img in zip(frames, stream(str(source), size, vf=vf, pix="rgb24"))}
    if len(got) != len(frames):
        missing = [f for f in frames if f not in got]
        raise SystemExit(f"REFUSED: could not decode source frame(s) {missing} from {home_relative(source)}")
    return got


def _depth_stats(depth: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | int | None]:
    vals = depth[valid_mask & np.isfinite(depth) & (depth > 0)]
    if vals.size == 0:
        return {
            "pixels": 0,
            "min_m": None,
            "p10_m": None,
            "median_m": None,
            "p90_m": None,
            "max_m": None,
        }
    return {
        "pixels": int(vals.size),
        "min_m": round(float(np.min(vals)), 4),
        "p10_m": round(float(np.percentile(vals, 10)), 4),
        "median_m": round(float(np.median(vals)), 4),
        "p90_m": round(float(np.percentile(vals, 90)), 4),
        "max_m": round(float(np.max(vals)), 4),
    }


def _colorize_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Render metric depth as Turbo RGB where near (smaller m) is warm/red-yellow and far is cool/blue."""
    inv_near = 1.0 / max(near_m, 1e-3)
    inv_far = 1.0 / max(far_m, near_m + 1e-3)
    safe_d = np.clip(np.where(valid & np.isfinite(depth), depth, far_m), near_m, far_m)
    inv_d = 1.0 / safe_d
    norm = np.clip((inv_d - inv_far) / max(inv_near - inv_far, 1e-6), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def _render_sheet(
    frames: list[int],
    rgb_by_frame: dict[int, np.ndarray],
    depth_by_frame: dict[int, np.ndarray],
    mask_by_frame: dict[int, np.ndarray],
    box: tuple[int, int, int, int],
    per_frame_rows: list[dict],
    regions_by_name: dict[str, tuple[str, dict[int, np.ndarray]]],
    cell_width: int,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    first_rgb = rgb_by_frame[frames[0]]
    canvas_h, canvas_w = first_rgb.shape[:2]
    cell_h = max(1, int(round(cell_width * canvas_h / canvas_w)))

    # Shared metric depth window across the shot (5th..95th percentile inside box) for consistent colour scale
    box_vals: list[np.ndarray] = []
    for f in frames:
        d_box = depth_by_frame[f][y0:y1, x0:x1]
        m_box = mask_by_frame[f][y0:y1, x0:x1] & np.isfinite(d_box) & (d_box > 0)
        if np.any(m_box):
            box_vals.append(d_box[m_box])
    if box_vals:
        all_v = np.concatenate(box_vals)
        near_m = float(np.percentile(all_v, 2))
        far_m = float(np.percentile(all_v, 98))
        if far_m <= near_m + 1e-3:
            far_m = near_m + 0.5
    else:
        near_m, far_m = 0.5, 5.0

    row_h = HEADER_H + cell_h
    sheet_img = np.zeros((len(frames) * row_h, cell_width * 2, 3), dtype=np.uint8)
    labels: list[tuple[int, int, str, tuple[int, int, int]]] = []

    for r_idx, (f, f_rec) in enumerate(zip(frames, per_frame_rows)):
        top = r_idx * row_h + HEADER_H
        rgb_tile = rgb_by_frame[f].copy()
        depth_tile = _colorize_depth(depth_by_frame[f], mask_by_frame[f], near_m, far_m)

        # Draw box if sub-canvas
        if (x0, y0, x1, y1) != (0, 0, canvas_w, canvas_h):
            cv2.rectangle(rgb_tile, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
            cv2.rectangle(depth_tile, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)

        # Draw named region outlines on both RGB and depth tiles
        for reg_name, (col_name, masks_f) in regions_by_name.items():
            if f in masks_f and np.any(masks_f[f]):
                colour = COLOURS[col_name]
                cont, _ = cv2.findContours(masks_f[f].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb_tile, cont, -1, colour, 2)
                cv2.drawContours(depth_tile, cont, -1, (255, 255, 255), 2)

        interp = cv2.INTER_AREA if canvas_w >= cell_width else cv2.INTER_CUBIC
        sheet_img[top : top + cell_h, 0:cell_width] = cv2.resize(rgb_tile, (cell_width, cell_h), interpolation=interp)
        sheet_img[top : top + cell_h, cell_width : cell_width * 2] = cv2.resize(
            depth_tile, (cell_width, cell_h), interpolation=interp
        )

        left_hdr = (
            f"frame {f}  fov_x={f_rec['fov_x_deg']:.2f}deg  fx={f_rec['fx']:.3f}"
        )
        right_hdr = (
            f"depth [near={near_m:.2f}m warm .. far={far_m:.2f}m cool]  "
            f"med={f_rec['depth_median_m']:.2f}m ({f_rec['depth_min_m']:.2f}..{f_rec['depth_max_m']:.2f}m)"
        )
        labels.append((6, r_idx * row_h + 6, left_hdr, (255, 235, 0)))
        labels.append((cell_width + 6, r_idx * row_h + 6, right_hdr, (255, 235, 0)))

        # Overlay per-region median depth in meters at the bottom-left of the depth panel
        if f_rec.get("regions"):
            reg_parts = []
            for r_name, r_stat in f_rec["regions"].items():
                if r_stat["median_m"] is not None:
                    reg_parts.append(f"{r_name}:{r_stat['median_m']:.2f}m(p10={r_stat['p10_m']:.2f})")
            if reg_parts:
                reg_line = "  ".join(reg_parts)
                labels.append((cell_width + 6, top + cell_h - 20, reg_line, (255, 255, 255)))

    pil_img = Image.fromarray(sheet_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(MONO_FONT, 13) if Path(MONO_FONT).is_file() else ImageFont.load_default()
    for lx, ly, text, col in labels:
        draw.text((lx + 1, ly + 1), text, font=font, fill=(0, 0, 0))
        draw.text((lx, ly), text, font=font, fill=col)
    return np.asarray(pil_img)


def run_measurement(
    *,
    source: Path,
    frames: list[int],
    out_dir: Path,
    size: tuple[int, int] = (1024, 768),
    box: tuple[int, int, int, int] | None = None,
    fit: bool = True,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    regions: list[tuple[str, Path, int, str]] | None = None,
    cell_width: int = 512,
    crop_to_box: bool = False,
) -> dict:
    """Measure metric depth and camera intrinsics on `frames` of `source` and write outputs to `out_dir`."""
    if not source.is_file():
        raise SystemExit(f"REFUSED: source video not found: {home_relative(source)}")

    npz_path = out_dir / "depth.npz"
    json_path = out_dir / "record.json"
    sheet_path = out_dir / "sheet.jpg"
    for existing in (npz_path, json_path, sheet_path):
        if existing.exists():
            raise SystemExit(f"REFUSED: output file already exists: {home_relative(existing)}")

    w, h = size
    x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
    rgb_by_frame = _grab_frames(source, frames, size, fit=fit)

    # Load optional region masks on the canvas clock
    regions_by_name: dict[str, tuple[str, dict[int, np.ndarray]]] = {}
    if regions:
        lo_f, hi_f = min(frames), max(frames)
        span_len = hi_f - lo_f + 1
        for r_name, r_path, r_at, r_col in regions:
            mask_arr, cov_arr = read_mask(str(r_path), size, r_at, lo_f, span_len)
            regions_by_name[r_name] = (
                r_col,
                {f: mask_arr[f - lo_f] for f in frames if cov_arr[f - lo_f]},
            )

    if str(checkpoint) == "tiny":
        model = HyDenMoGe.tiny_for_tests().to(device)
        model_info = {
            "checkpoint": "tiny_for_tests",
            "bytes": 0,
            "input_range": [0.0, 1.0],
            "vit_internal_resolution": list(model.vit_internal_resolution or []),
        }
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise SystemExit(f"REFUSED: checkpoint not found: {home_relative(ckpt_path)}")
        model = load_hyden_moge2(ckpt_path, device=device)
        model_info = {
            "checkpoint": ckpt_path.name,
            "bytes": int(ckpt_path.stat().st_size),
            "input_range": [0.0, 1.0],
            "vit_internal_resolution": list(model.vit_internal_resolution or []),
        }

    depth_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    intr_list: list[np.ndarray] = []
    fov_x_list: list[float] = []
    fov_y_list: list[float] = []
    per_frame_rows: list[dict] = []
    depth_by_frame: dict[int, np.ndarray] = {}
    mask_by_frame: dict[int, np.ndarray] = {}

    for f in frames:
        rgb = rgb_by_frame[f]
        inp_rgb = rgb[y0:y1, x0:x1] if crop_to_box else rgb
        tensor = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32, device=device) / 255.0
        out = model.infer(tensor, apply_mask=False)

        depth_f = out["depth"][0].detach().cpu().numpy().astype(np.float32)
        mask_f = out["mask"][0].detach().cpu().numpy().astype(bool)
        intr_f = out["intrinsics"][0].detach().cpu().numpy().astype(np.float32)
        fov_x_t, fov_y_t = fov_deg_from_intrinsics(out["intrinsics"])
        fov_x = float(fov_x_t[0].item())
        fov_y = float(fov_y_t[0].item())

        if crop_to_box and (x0, y0, x1, y1) != (0, 0, w, h):
            full_depth = np.full((h, w), np.inf, dtype=np.float32)
            full_mask = np.zeros((h, w), dtype=bool)
            full_depth[y0:y1, x0:x1] = depth_f
            full_mask[y0:y1, x0:x1] = mask_f
            depth_canvas, mask_canvas = full_depth, full_mask
        else:
            depth_canvas, mask_canvas = depth_f, mask_f

        depth_by_frame[f] = depth_canvas
        mask_by_frame[f] = mask_canvas
        depth_list.append(depth_f)
        mask_list.append(mask_f)
        intr_list.append(intr_f)
        fov_x_list.append(fov_x)
        fov_y_list.append(fov_y)

        box_depth = depth_canvas[y0:y1, x0:x1]
        box_mask = mask_canvas[y0:y1, x0:x1]
        b_stats = _depth_stats(box_depth, box_mask)

        reg_stats: dict[str, dict] = {}
        for r_name, (_col, masks_f) in regions_by_name.items():
            if f in masks_f:
                reg_stats[r_name] = _depth_stats(depth_canvas, mask_canvas & masks_f[f])

        row = {
            "frame": int(f),
            "fov_x_deg": round(fov_x, 4),
            "fov_y_deg": round(fov_y, 4),
            "fx": round(float(intr_f[0, 0]), 5),
            "fy": round(float(intr_f[1, 1]), 5),
            "cx": round(float(intr_f[0, 2]), 5),
            "cy": round(float(intr_f[1, 2]), 5),
            "metric_scale": round(float(out["metric_scale"][0].item()), 5),
            "depth_min_m": b_stats["min_m"],
            "depth_p10_m": b_stats["p10_m"],
            "depth_median_m": b_stats["median_m"],
            "depth_p90_m": b_stats["p90_m"],
            "depth_max_m": b_stats["max_m"],
            "valid_pixels_in_box": b_stats["pixels"],
        }
        if reg_stats:
            row["regions"] = reg_stats
        per_frame_rows.append(row)

    fov_arr = np.asarray(fov_x_list, dtype=np.float64)
    fx_arr = np.asarray([r["fx"] for r in per_frame_rows], dtype=np.float64)
    summary = {
        "fov_x_deg_least_median_most": [
            round(float(np.min(fov_arr)), 4),
            round(float(np.median(fov_arr)), 4),
            round(float(np.max(fov_arr)), 4),
        ],
        "fov_x_deg_spread": round(float(np.max(fov_arr) - np.min(fov_arr)), 4),
        "fov_x_deg_std": round(float(np.std(fov_arr)), 4),
        "fx_least_median_most": [
            round(float(np.min(fx_arr)), 5),
            round(float(np.median(fx_arr)), 5),
            round(float(np.max(fx_arr)), 5),
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        frames=np.asarray(frames, dtype=np.int32),
        depth=np.stack(depth_list, axis=0),
        mask=np.stack(mask_list, axis=0),
        intrinsics=np.stack(intr_list, axis=0),
        fov_x_deg=np.asarray(fov_x_list, dtype=np.float32),
        fov_y_deg=np.asarray(fov_y_list, dtype=np.float32),
        box=np.asarray([x0, y0, x1, y1], dtype=np.int32),
    )

    sheet_arr = _render_sheet(
        frames=frames,
        rgb_by_frame=rgb_by_frame,
        depth_by_frame=depth_by_frame,
        mask_by_frame=mask_by_frame,
        box=(x0, y0, x1, y1),
        per_frame_rows=per_frame_rows,
        regions_by_name=regions_by_name,
        cell_width=cell_width,
    )
    Image.fromarray(sheet_arr).save(sheet_path, quality=92)

    record = {
        "what": "HyDen MoGe-2 per-frame metric depth and camera intrinsics measurement",
        "model": model_info,
        "source": home_relative(source),
        "canvas": [w, h],
        "fit": bool(fit),
        "box": [x0, y0, x1, y1],
        "crop_to_box": bool(crop_to_box),
        "frames": frames,
        "summary": summary,
        "per_frame": per_frame_rows,
        "outputs": {
            "depth_npz": home_relative(npz_path),
            "record_json": home_relative(json_path),
            "sheet_jpg": home_relative(sheet_path),
        },
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--source", required=True, help="source video path")
    p.add_argument("--frames", required=True, help="comma-separated source frame numbers")
    p.add_argument("--out-dir", required=True, help="output directory for depth.npz, record.json, sheet.jpg")
    p.add_argument("--size", default="1024x768", help="canvas size WxH (default: 1024x768)")
    p.add_argument("--box", default=None, help="optional x0,y0,x1,y1 box on canvas for stats/outline")
    p.add_argument("--crop-to-box", action="store_true", help="run MoGe-2 on the cropped box instead of full canvas")
    p.add_argument("--no-fit", action="store_true", help="plain scale to canvas instead of loader FIT")
    p.add_argument("--region", action="append", default=[], metavar="NAME=MASK.mkv[@FIRST][:COLOUR]",
                   help="optional mask video to outline and measure metric depth on")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                   help="path to HyDen MoGe-2 .pth checkpoint (or 'tiny' for synthetic check)")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--cell", type=int, default=512, help="tile width in pixels for sheet.jpg")
    a = p.parse_args(argv)

    source = Path(a.source)
    if not source.is_file():
        raise SystemExit(f"REFUSED: source video not found: {home_relative(source)}")
    w, h = _parse_size(a.size)
    box = _parse_box(a.box, w, h)
    _, _, total_frames = probe(str(source))
    frames = _parse_frames(a.frames, total_frames)
    regions = [_parse_region(spec, idx) for idx, spec in enumerate(a.region)]

    rec = run_measurement(
        source=source,
        frames=frames,
        out_dir=Path(a.out_dir),
        size=(w, h),
        box=box,
        fit=not a.no_fit,
        checkpoint=a.checkpoint,
        device=a.device,
        regions=regions,
        cell_width=a.cell,
        crop_to_box=a.crop_to_box,
    )
    s = rec["summary"]
    print(
        f"wrote {a.out_dir} ({len(frames)} frames): "
        f"fov_x={s['fov_x_deg_least_median_most']} deg (spread {s['fov_x_deg_spread']} deg), "
        f"fx={s['fx_least_median_most']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
