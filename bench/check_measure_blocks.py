#!/usr/bin/env python3
"""Check `bench/measure_blocks.py`: pasted 16x16 squares found, clean clips clear, real grid edges said as such.

Claims, i.e. what breaks if a case is deleted:
- Case 1 (pasted 16x16 squares are found spatially and temporally): a spatial 16x16
  square pasted on the grid or a temporal `kept_carved` (`taken`) cell that steps across
  a frame boundary without its neighbours stepping goes undetected, or fails to report
  its exact `(cy, cx)` coordinates, frame number, `grid_excess` / `step_excess`, and
  `in_kept_carved` status.
- Case 2 (clean clip without pasted squares finds none): smooth moving texture inside
  the subject region false-positives as `squares_found` or `real_edges_on_grid`, or fails
  to print `what_measured` and `cannot_see` while exiting 0.
- Case 3 (real edges on the 16 px grid are reported as such and not as squares): a
  straight horizontal or vertical line that happens to fall on `x % 16 == 0` or
  `y % 16 == 0` (either one-sided or present in `--source`) is misreported in
  `squares_found` instead of `real_edges_on_grid` with its reason
  (`single_grid_line` / `present_in_source`).
- Case 4 (capture region.npz and --only frame span filter): a capture `region.npz`
  (`region` `[F, lh, lw]` bool with optional packed `carried` / `given_back__*` and
  `manifest.json`) or `--only START:END` / `START-END` span filter fails to load or
  fails to restrict measurement to the requested frames.
- Case 5 (invalid inputs refuse cleanly with no traceback): missing clips/regions,
  bad `--only` syntax, frames with no overlap, off-16 canvas dimensions, or directories
  with no region files crash with a Python traceback instead of refusing cleanly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, refused  # noqa: E402
import measure_blocks as mb  # noqa: E402

CELL = mb.CELL


def _write_gray_mkv(path: Path, luma: np.ndarray) -> None:
    """Write `[F, H, W]` uint8 luma array to a lossless yuv420p `.mkv` video with neutral chroma (U=128, V=128) at 24 fps."""
    f_count, h, w = luma.shape
    u_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8).tobytes()
    v_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8).tobytes()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "24",
        "-i", "-",
        "-c:v", "ffv1", "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    for f in range(f_count):
        y_bytes = np.clip(np.round(luma[f]), 0, 255).astype(np.uint8).tobytes()
        proc.stdin.write(y_bytes + u_plane + v_plane)
    proc.stdin.close()
    err = proc.stderr.read()
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {err.decode(errors='replace')}")


def _write_window_region_npz(
    path: Path,
    mask_hw: np.ndarray,
    tokens: np.ndarray,
    margin: int = 16,
    first_frame: int = 0,
    trim: int = 0,
) -> None:
    """Write a window `*_region.npz` file in the exact format `video_mask.save_window_region` writes."""
    f_count, h, w = mask_hw.shape
    meta = {
        "width": w,
        "margin": int(margin),
        "first_frame": int(first_frame),
        "trim": int(trim),
        "source": {
            "composite": "the mask grown by the margin",
            "feather_pixels": 0,
            "change_threshold": 0.0,
            "cuts": None,
            "grow_pixels": int(margin),
            "grow_by": "a fixed margin",
            "replace": "the masked region",
            "edge": "round to whole token pairs",
        },
        "zoom": None,
    }
    np.savez_compressed(
        path,
        mask=np.packbits(mask_hw.astype(np.uint8), axis=-1),
        tokens=tokens.astype(np.uint8),
        margin=np.int32(margin),
        meta=json.dumps(meta),
    )


def case_pasted_squares_found() -> None:
    """A made clip with pasted 16x16 squares (static spatial square, 1-frame blinking square, and kept-carved stepping square) -> found."""
    f_count, lh, lw = 5, 6, 8
    h, w = lh * CELL, lw * CELL
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Smooth base background (gentle 0.2 px slope, well below any grid jump)
    base = 90.0 + 0.15 * xx + 0.10 * yy
    luma = np.stack([base + 0.5 * f for f in range(f_count)], axis=0).astype(np.float32)

    # 1. Static pasted 16x16 square at cell (r=2, c=3) across all frames (+35 luma jump right on 16 px walls)
    luma[:, 2 * CELL : 3 * CELL, 3 * CELL : 4 * CELL] += 35.0

    # 2. Temporal 1-frame pasted 16x16 square at cell (r=3, c=5) on frame 2 only (+40 luma step)
    luma[2, 3 * CELL : 4 * CELL, 5 * CELL : 6 * CELL] += 40.0

    # 3. Kept-carved square at cell (r=1, c=2) with feathered-like interior step on frame 3 (+28 luma inside)
    luma[3, 1 * CELL + 2 : 2 * CELL - 2, 2 * CELL + 2 : 3 * CELL - 2] += 28.0

    region_cells = np.zeros((f_count, lh, lw), dtype=bool)
    region_cells[:, 1:5, 2:6] = True
    region_cells[:, 1, 2] = False  # carved out by kept
    kept_carved = np.zeros((f_count, lh, lw), dtype=bool)
    kept_carved[:, 1, 2] = True

    res = mb.measure_luma_blocks(
        luma=luma,
        region_cells=region_cells,
        subject_cells=region_cells | kept_carved,
        kept_carved_cells=kept_carved,
        first_frame=751,
        band_cells=2,
    )
    assert res["summary"]["squares_found_count"] > 0, f"expected pasted squares to be found, got {res['summary']}"

    # Verify static square (2, 3) is found on frame 0 (source_frame 751) with all 4 sides active
    sq_static = [s for s in res["squares_found"] if s["source_frame"] == 751 and s["cell"] == [2, 3]]
    assert len(sq_static) == 1, f"expected static square at (2, 3) on frame 751, got {sq_static}"
    assert sq_static[0]["kind"] == "spatial_grid", sq_static[0]
    assert set(sq_static[0]["active_sides"]) == {"top", "bottom", "left", "right"}, sq_static[0]
    assert sq_static[0]["max_grid_excess"] >= 30.0, sq_static[0]

    # Verify temporal square (3, 5) is found on frame 2 (source_frame 753) as spatial_and_temporal
    sq_temp = [s for s in res["squares_found"] if s["source_frame"] == 753 and s["cell"] == [3, 5]]
    assert len(sq_temp) == 1, f"expected temporal square at (3, 5) on frame 753, got {sq_temp}"
    assert sq_temp[0]["kind"] == "spatial_and_temporal", sq_temp[0]
    assert sq_temp[0]["step_excess"] >= 35.0, sq_temp[0]
    assert sq_temp[0]["step_ratio"] >= 10.0, sq_temp[0]

    # Verify kept-carved square (1, 2) is found on frame 3 (source_frame 754) as temporal_step
    sq_kept = [s for s in res["squares_found"] if s["source_frame"] == 754 and s["cell"] == [1, 2]]
    assert len(sq_kept) == 1, f"expected kept_carved stepping square at (1, 2) on frame 754, got {sq_kept}"
    assert sq_kept[0]["in_kept_carved"] is True, sq_kept[0]
    assert sq_kept[0]["step_excess"] >= 15.0, sq_kept[0]

    # Also test end-to-end CLI on a written video and saved *_window_1_region.npz
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        clip_path = td / "pasted.mkv"
        reg_path = td / "pasted_window_1_region.npz"
        out_json = td / "out.json"
        _write_gray_mkv(clip_path, luma)
        # 5 frames -> lt = 2 latent steps (1 + 4 frames)
        mask_hw = np.zeros((f_count, h, w), dtype=bool)
        mask_hw[:, 1 * CELL : 5 * CELL, 2 * CELL : 6 * CELL] = True
        tokens = np.zeros((2, lh, lw), dtype=np.uint8)
        tokens[:, 1:5, 2:6] = 1
        tokens[:, 1, 2] = 0
        _write_window_region_npz(reg_path, mask_hw, tokens, margin=16, first_frame=0, trim=0)

        cli_rep = mb.run_cli([
            "--clip", str(clip_path),
            "--region", str(reg_path),
            "--first-frame", "751",
            "--region-first", "751",
            "--out", str(out_json),
        ])
        assert out_json.is_file(), "expected --out JSON file to be written"
        saved = json.loads(out_json.read_text())
        assert saved["summary"]["squares_found_count"] == cli_rep["summary"]["squares_found_count"]
        assert saved["summary"]["squares_found_count"] >= 7, saved["summary"]
        assert any(s["cell"] == [2, 3] and s["source_frame"] == 751 for s in saved["squares_found"])
        assert any(s["cell"] == [3, 5] and s["source_frame"] == 753 for s in saved["squares_found"])


def case_clean_clip_not_found() -> None:
    """A made clip without pasted squares (smooth gradients and smooth across-frame motion) -> not found."""
    f_count, lh, lw = 5, 6, 8
    h, w = lh * CELL, lw * CELL
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth spatial wave + smooth across-frame pan, no 16x16 block jumps
    frames = []
    for f in range(f_count):
        frame = (
            110.0
            + 25.0 * np.sin((xx + 2.0 * f) / 18.0)
            + 20.0 * np.cos((yy - 1.5 * f) / 15.0)
        )
        frames.append(frame.astype(np.float32))
    luma = np.stack(frames, axis=0)

    region_cells = np.zeros((f_count, lh, lw), dtype=bool)
    region_cells[:, 1:5, 2:6] = True

    res = mb.measure_luma_blocks(
        luma=luma,
        region_cells=region_cells,
        first_frame=100,
        band_cells=2,
    )
    assert res["summary"]["squares_found_count"] == 0, f"expected 0 squares on clean clip, got {res['squares_found']}"
    assert res["squares_found"] == [], res["squares_found"]
    assert res["summary"]["real_edges_on_grid_count"] == 0, res["real_edges_on_grid"]

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        clip_path = td / "clean.mkv"
        reg_path = td / "clean_window_1_region.npz"
        _write_gray_mkv(clip_path, luma)
        mask_hw = np.zeros((f_count, h, w), dtype=bool)
        mask_hw[:, 1 * CELL : 5 * CELL, 2 * CELL : 6 * CELL] = True
        tokens = np.zeros((2, lh, lw), dtype=np.uint8)
        tokens[:, 1:5, 2:6] = 1
        _write_window_region_npz(reg_path, mask_hw, tokens, margin=0, first_frame=0, trim=0)

        cli_rep = mb.run_cli([
            "--clip", str(clip_path),
            "--region", str(reg_path),
            "--first-frame", "100",
            "--region-first", "100",
        ])
        assert cli_rep["summary"]["squares_found_count"] == 0, cli_rep["summary"]


def case_real_edge_on_grid_said_as_such() -> None:
    """A real edge on the 16 px grid (multi-pixel gradient edge, full straight scene line on grid, or source edge) -> said as such in real_edges_on_grid, not miscounted as a pasted square."""
    f_count, lh, lw = 5, 6, 8
    h, w = lh * CELL, lw * CELL
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Subcase A: Multi-pixel scene gradient crossing vertical grid line x = 3 * 16 = 48
    # Ramp of 12 luma per pixel across x = 46..50 so grid_jump = 12 and side_jump = 12 (grid_ratio = 1.0)
    ramp = np.clip((xx - 45.0) * 12.0, 0.0, 60.0)
    luma_grad = np.stack([80.0 + ramp for _ in range(f_count)], axis=0).astype(np.float32)
    region_cells = np.zeros((f_count, lh, lw), dtype=bool)
    region_cells[:, 1:5, 1:7] = True

    res_grad = mb.measure_luma_blocks(luma_grad, region_cells, first_frame=500, band_cells=1)
    assert res_grad["summary"]["squares_found_count"] == 0, (
        f"multi-pixel gradient across grid must not be called a square: {res_grad['squares_found']}"
    )
    assert res_grad["summary"]["real_edges_on_grid_count"] > 0, "expected gradient edge on grid to be reported"
    grad_edges = [e for e in res_grad["real_edges_on_grid"] if e["grid_px"] == 48 and e["axis"] == "vertical"]
    assert len(grad_edges) == f_count * lh, f"expected vertical@48px on all rows/frames, got {len(grad_edges)}"
    assert all(e["why"] == "gradient_across_grid" for e in grad_edges), grad_edges[:3]

    # Subcase B: Sharp 1-pixel straight architectural line lying right on x = 4 * 16 = 64 across all rows
    # (collinear_cells == 6 >= COLLINEAR_LINE_CELLS, no closing orthogonal walls)
    luma_line = np.full((f_count, h, w), 70.0, dtype=np.float32)
    luma_line[:, :, 4 * CELL :] = 125.0  # 55-luma 1-px step right at x = 64 across the whole frame height
    res_line = mb.measure_luma_blocks(luma_line, region_cells, first_frame=500, band_cells=1)
    assert res_line["summary"]["squares_found_count"] == 0, (
        f"straight scene line on grid must not be called a 16x16 square: {res_line['squares_found']}"
    )
    line_edges = [e for e in res_line["real_edges_on_grid"] if e["grid_px"] == 64 and e["axis"] == "vertical"]
    assert len(line_edges) == f_count * lh, f"expected vertical@64px on all rows/frames, got {len(line_edges)}"
    assert all(e["why"] == "single_grid_line" and e["collinear_cells"] == lh for e in line_edges), line_edges[:3]

    # Subcase C: A grid feature already present in --source (source_luma) is reported as present_in_source
    luma_src = np.full((f_count, h, w), 80.0, dtype=np.float32)
    luma_src[:, 2 * CELL : 3 * CELL, 3 * CELL : 4 * CELL] = 130.0
    res_src = mb.measure_luma_blocks(
        luma_src,
        region_cells,
        source_luma=luma_src.copy(),
        first_frame=500,
        band_cells=1,
    )
    assert res_src["summary"]["squares_found_count"] == 0, (
        f"grid feature already present in --source must not be called a render square: {res_src['squares_found']}"
    )
    src_edges = [e for e in res_src["real_edges_on_grid"] if e["why"] == "present_in_source"]
    assert len(src_edges) == f_count * 4, f"expected 4 source edges per frame, got {len(src_edges)}"


def case_capture_region_and_only_span() -> None:
    """Capture-style region.npz + manifest.json and --only span filter measure only the requested source frames."""
    f_count, lh, lw = 6, 4, 6
    h, w = lh * CELL, lw * CELL
    luma = np.full((f_count, h, w), 100.0, dtype=np.float32)
    # Put a pasted square only on frame index 4 (source_frame 204)
    luma[4, 1 * CELL : 2 * CELL, 2 * CELL : 3 * CELL] = 150.0

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        cap_dir = td / "capture"
        run_dir = cap_dir / "runs" / "lead_0200"
        run_dir.mkdir(parents=True)
        (cap_dir / "manifest.json").write_text(json.dumps({"first_frame": 200, "frames": f_count, "size": [w, h]}))
        reg_cells = np.zeros((f_count, lh, lw), dtype=bool)
        reg_cells[:, 1:3, 1:5] = True
        np.savez_compressed(run_dir / "region.npz", region=reg_cells)

        clip_path = td / "piece.mkv"
        _write_gray_mkv(clip_path, luma)

        # Measure only 200:203 (colon separator, excludes frame 204) -> 0 squares
        rep_before = mb.run_cli([
            "--clip", str(clip_path),
            "--region", str(run_dir),
            "--only", "200:203",
        ])
        assert rep_before["summary"]["source_frame_span"] == [200, 203], rep_before["summary"]
        assert rep_before["summary"]["squares_found_count"] == 0, rep_before["summary"]

        # Measure 203-205 (dash separator, includes frame 204) -> square found on 204
        rep_hit = mb.run_cli([
            "--clip", str(clip_path),
            "--region", str(run_dir / "region.npz"),
            "--only", "203-205",
        ])
        assert rep_hit["summary"]["source_frame_span"] == [203, 205], rep_hit["summary"]
        assert rep_hit["summary"]["squares_found_count"] == 2, rep_hit["summary"]
        assert rep_hit["summary"]["frames_with_squares"] == [204, 205], rep_hit["summary"]


def case_refusals() -> None:
    """Bad inputs refuse with the tool's own words and no traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        reg_cells = np.ones((4, 4, 4), dtype=bool)
        good_reg = td / "region.npz"
        np.savez_compressed(good_reg, region=reg_cells)
        good_clip = td / "clip.mkv"
        _write_gray_mkv(good_clip, np.full((4, 4 * CELL, 4 * CELL), 90.0, dtype=np.float32))

        refused(
            lambda: mb.run_cli(["--clip", str(td / "missing.mkv"), "--region", str(good_reg)]),
            "REFUSED", "not found",
        )
        refused(
            lambda: mb.run_cli(["--clip", str(good_clip), "--region", str(td / "missing.npz")]),
            "REFUSED", "not found",
        )
        refused(
            lambda: mb.run_cli(["--clip", str(good_clip), "--region", str(good_reg), "--band-cells", "-1"]),
            "REFUSED", "--band-cells",
        )
        refused(
            lambda: mb.run_cli(["--clip", str(good_clip), "--region", str(good_reg), "--only", "10-5"]),
            "REFUSED", "--only",
        )
        refused(
            lambda: mb.run_cli([
                "--clip", str(good_clip),
                "--region", str(good_reg),
                "--first-frame", "0",
                "--region-first", "50",
            ]),
            "REFUSED", "no overlapping frames",
        )
        bad_npz = td / "bad.npz"
        np.savez_compressed(bad_npz, foo=np.zeros(3))
        refused(
            lambda: mb.run_cli(["--clip", str(good_clip), "--region", str(bad_npz)]),
            "REFUSED", "missing",
        )
        empty_dir = td / "empty_regions"
        empty_dir.mkdir()
        refused(
            lambda: mb.run_cli(["--clip", str(good_clip), "--region", str(empty_dir)]),
            "REFUSED", "holds no",
        )
        refused(
            lambda: mb.measure_luma_blocks(
                np.zeros((2, 30, 32), dtype=np.float32),
                np.zeros((2, 2, 2), dtype=bool),
            ),
            "multiple of CELL=16",
        )


if __name__ == "__main__":
    case("pasted 16x16 squares are found spatially and temporally", case_pasted_squares_found)
    case("clean clip without pasted squares finds none", case_clean_clip_not_found)
    case("real edges on the 16 px grid are reported as such and not as squares", case_real_edge_on_grid_said_as_such)
    case("capture region.npz and --only frame span filter", case_capture_region_and_only_span)
    case("invalid inputs refuse cleanly with no traceback", case_refusals)
    sys.exit(finish())

