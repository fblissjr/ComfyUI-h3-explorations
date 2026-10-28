#!/usr/bin/env python3
"""Deterministic CLI helper for `.gemini/skills/video-understanding/`.

Enforces the Native-First (`agy`) Gate and wraps `gemini-api` (using its
built-in credential store) for five multimodal workflows in `docs/masked_v2v/`:

1. `boxes`: 2D bounding-box (`box_2d` `[ymin, xmin, ymax, xmax]` in `0..1000`)
   and polygon segmentation (`mask` `[[x, y], ...]` in `0..1000`) grounding,
   descaling to `1024x768` `[left, top, width, height]` (`cast.jsonl` format)
   and snapping to SAM 3.1 detections in `<preview>_people.npz` (`IoU >= 0.50`).
2. `slice`: Server-side shot-window slicing (`start_offset` / `end_offset` in
   seconds from `24.0 fps` `[first..last]` frame spans) in static `24 fps`
   high-resolution mode.
3. `agentic-video`: Opt-in `"processing": "agentic"` timeline navigation on
   long single-canvas videos (`>100 MB` master cuts), or mixed `agentic`
   (master cut) + `static` (`24 fps` candidate shot via `--compare-shot`)
   continuity checks. Mechanically refuses `*_STACKED*` and `.1fps.` strips.
4. `zoom`: Agentic Vision (`tools: [{"type": "code_execution"}]` +
   `"resolution": "ultra_high"`) on dense diagnostic strips (`--strips`) or
   contact sheets when local `$PY` cropping is not used.
5. `transcribe-spans`: Speaker and lyric transcription (audio-only or
   `--visual-speakers` at `24 fps` high resolution) with automatic conversion
   of `MM:SS(.mmm)` timestamps into `24.0 fps` `[start_frame, end_frame]` spans
   to complement `bench/voice_spans.py`.

Native-First (`agy`) Gate:
  When a local video `<100 MB` is passed to `slice`, `agentic-video`, or
  `transcribe-spans` and a matching `<stem>.1fps.*.mp4` companion exists on
  disk, this tool refuses to upload the video unless `--force-api` is explicitly
  passed, instructing the caller to use `agy`'s built-in `read_file` tool at
  `$0` API spend.

Run `--self-test` to verify coordinate math, SAM `<preview>_people.npz` IoU
matching, timestamp-to-frame conversion, Native-First / `*_STACKED` refusal
guards, and `gemini-api --dry-run` payload schemas without API spend.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get("GEMINI_VIDEO_MODEL", "gemini-3.8-flash")
INLINE_MAX_BYTES = 12 * 1024 * 1024  # Safe raw ceiling under 20 MB base64 limit
READ_FILE_MAX_BYTES = 100 * 1024 * 1024  # agy read_file ceiling (100 MB)


def _mime_of(path: Path, default: str = "image/png") -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or default


def _image_size(path: Path) -> tuple[int, int]:
    """Return `(width, height)` of an image using PIL or cv2."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cannot read image dimensions from {path}")
        h, w = img.shape[:2]
        return int(w), int(h)


def _find_1fps_companions(video_path: Path) -> list[Path]:
    """Return any `<stem>.1fps.*.mp4` companions sitting alongside `video_path`."""
    if not video_path.parent.is_dir():
        return []
    stem = video_path.stem
    if ".1fps." in video_path.name:
        return [video_path]
    return sorted(video_path.parent.glob(f"{stem}.1fps.*.mp4"))


def _check_native_first(video_target: str, force_api: bool, dry_run: bool) -> None:
    """Refuse uploading a local `<100 MB` video when a `.1fps.*.mp4` companion exists."""
    if force_api or dry_run or video_target.startswith("files/"):
        return
    p = Path(video_target)
    if not p.is_file():
        return
    size_bytes = p.stat().st_size
    if size_bytes >= READ_FILE_MAX_BYTES:
        return
    companions = _find_1fps_companions(p)
    if companions:
        comp_names = ", ".join(c.name for c in companions[:3])
        raise ValueError(
            f"Native-First (agy) Gate refused uploading {p.name} ({size_bytes / (1024 * 1024):.1f} MB < 100 MB): "
            f"companion .1fps file(s) exist on disk ({comp_names}). "
            f"Use agy's native `read_file` tool on {comp_names} (and {p.name} for audio) at $0 API spend, "
            f"or pass --force-api if an explicit gemini-api call is required."
        )


def _check_not_stacked_for_agentic(video_target: str) -> None:
    """Refuse `*_STACKED*` or `.1fps.` diagnostic strips on `agentic-video`."""
    name_upper = Path(video_target).name.upper()
    if "STACKED" in name_upper or ".1FPS." in name_upper:
        raise ValueError(
            f"Path 2B (`processing: agentic`) is BANNED on multi-panel/stacked or .1fps strips ({video_target}). "
            f"Server-side dynamic cropping misreads multi-row diagnostic tiles. "
            f"Use Path 1 (`read_file` on `.1fps.mp4`) or Path 2A (`gemini_media.py slice` at `fps: 24, resolution: high`)."
        )


def descale_box_2d(box_2d: list[int | float], width: int, height: int) -> dict[str, Any]:
    """Convert Gemini `[ymin, xmin, ymax, xmax]` in `[0, 1000]` to pixel coordinates.

    Returns both `box_xyxy_px` (`[x1, y1, x2, y2]`) and `box_ltwh_px`
    (`[left, top, width, height]`, the format used by `bench/cast_file.py`).
    """
    if len(box_2d) != 4:
        raise ValueError(f"expected [ymin, xmin, ymax, xmax] of length 4, got {box_2d}")
    ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
    x1 = round(xmin * width / 1000.0, 2)
    y1 = round(ymin * height / 1000.0, 2)
    x2 = round(xmax * width / 1000.0, 2)
    y2 = round(ymax * height / 1000.0, 2)
    w = round(max(0.0, x2 - x1), 2)
    h = round(max(0.0, y2 - y1), 2)
    return {
        "box_2d_1000": [ymin, xmin, ymax, xmax],
        "canvas": [int(width), int(height)],
        "box_xyxy_px": [x1, y1, x2, y2],
        "box_ltwh_px": [x1, y1, w, h],
    }


def descale_polygon(mask_1000: list[list[int | float]], width: int, height: int) -> list[list[float]]:
    """Convert Gemini `[[x, y], ...]` polygon in `[0, 1000]` to pixel `[[x_px, y_px], ...]`."""
    out: list[list[float]] = []
    for pt in mask_1000:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append([
                round(float(pt[0]) * width / 1000.0, 2),
                round(float(pt[1]) * height / 1000.0, 2),
            ])
    return out


def match_sam_people(
    box_ltwh: list[float],
    target_size: tuple[int, int],
    people_npz: Path,
    source_frame: int,
) -> dict[str, Any]:
    """Match a `box_ltwh` (`[left, top, width, height]`) against `<preview>_people.npz` on `source_frame`.

    Mirrors `bench/cast_file.py::drawn` (`L318-L358`) so Group 3 `cast.jsonl`
    seed verification uses the exact same bit-unpacked SAM 3.1 bounding boxes.
    """
    import numpy as np

    z = np.load(people_npz)
    if "masks" not in z.files or "frame" not in z.files or "first" not in z.files:
        raise ValueError(f"{people_npz} is missing required keys (`masks`, `frame`, `first`)")
    pf = int(z["first"])
    p_json = people_npz.with_suffix(".json")
    if p_json.is_file():
        pw, ph = [int(v) for v in json.loads(p_json.read_text())["canvas"]]
    else:
        ph = int(z["masks"].shape[1])
        pw = int(round(ph * (target_size[0] / float(target_size[1]))))
    sx, sy = pw / float(target_size[0]), ph / float(target_size[1])
    rel = int(source_frame) - pf
    z_frames = np.asarray(z["frame"], dtype=np.int32)
    idxs = np.nonzero(z_frames == rel)[0]
    if len(idxs) == 0:
        return {
            "source_frame": int(source_frame),
            "relative_frame": int(rel),
            "detections_on_frame": 0,
            "best_iou": 0.0,
            "passes_0_50_gate": False,
            "sam_box_ltwh": None,
        }

    bx, by0, bw, bh = [float(v) for v in box_ltwh]
    ux0, uy0, ux1, uy1 = bx * sx, by0 * sy, (bx + bw) * sx, (by0 + bh) * sy
    best_iou = 0.0
    best_det_box: list[float] | None = None
    best_det_idx: int | None = None

    for det_rank, idx in enumerate(idxs):
        m_2d = np.unpackbits(z["masks"][int(idx)], axis=-1)[:, :pw].astype(bool)
        ys, xs = np.nonzero(m_2d)
        if len(ys) == 0:
            continue
        dx0, dy0 = float(xs.min()), float(ys.min())
        dx1, dy1 = float(xs.max() + 1), float(ys.max() + 1)
        ix0, iy0 = max(ux0, dx0), max(uy0, dy0)
        ix1, iy1 = min(ux1, dx1), min(uy1, dy1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = (
            max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0)
            + max(0.0, dx1 - dx0) * max(0.0, dy1 - dy0)
            - inter
        )
        if union > 0:
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_det_idx = int(det_rank)
                best_det_box = [
                    round(dx0 / sx, 2),
                    round(dy0 / sy, 2),
                    round((dx1 - dx0) / sx, 2),
                    round((dy1 - dy0) / sy, 2),
                ]

    fmt_box = (
        [int(v) if float(v).is_integer() else float(v) for v in best_det_box]
        if best_det_box is not None
        else None
    )
    cast_arg = (
        f"{int(source_frame)}:{fmt_box[0]},{fmt_box[1]},{fmt_box[2]},{fmt_box[3]}"
        if fmt_box is not None
        else None
    )
    return {
        "source_frame": int(source_frame),
        "relative_frame": int(rel),
        "detections_on_frame": int(len(idxs)),
        "best_person_rank": best_det_idx,
        "best_iou": round(float(best_iou), 4),
        "passes_0_50_gate": bool(best_iou >= 0.5 and fmt_box is not None),
        "sam_box_ltwh": fmt_box,
        "cast_cli_box_arg": cast_arg,
    }


def frame_span_to_offsets(first: int, last: int, fps: float = 24.0) -> tuple[float, float]:
    """Convert inclusive frame span `[first..last]` to `(start_offset, end_offset)` seconds."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if last < first or first < 0:
        raise ValueError(f"invalid frame span [{first}..{last}]")
    return round(first / fps, 3), round((last + 1) / fps, 3)


_TS_RE = re.compile(
    r"^\s*(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)\s*$"
)


def parse_timestamp_seconds(ts: str | float | int) -> float:
    """Parse `HH:MM:SS(.mmm)`, `MM:SS(.mmm)`, or numeric seconds into float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if not s:
        raise ValueError("empty timestamp")
    m = _TS_RE.match(s)
    if m:
        hours = int(m.group("h") or 0)
        minutes = int(m.group("m"))
        seconds = float(m.group("s"))
        return hours * 3600.0 + minutes * 60.0 + seconds
    if s.endswith("s"):
        return float(s[:-1].strip())
    return float(s)


def timestamp_span_to_frames(
    start_ts: str | float | int,
    end_ts: str | float | int,
    shot_first: int = 0,
    fps: float = 24.0,
    shot_last: int | None = None,
) -> tuple[int, int]:
    """Convert `(start_ts, end_ts)` clip-relative timestamps into inclusive `[start_frame, end_frame]`."""
    t0 = max(0.0, parse_timestamp_seconds(start_ts))
    t1 = max(t0, parse_timestamp_seconds(end_ts))
    f0 = int(shot_first) + int(round(t0 * fps))
    f1 = int(shot_first) + int(round(t1 * fps))
    if f1 < f0:
        f1 = f0
    if shot_last is not None:
        f0 = min(max(int(shot_first), f0), int(shot_last))
        f1 = min(max(f0, f1), int(shot_last))
    return f0, f1


def _upload_file(path: Path) -> str:
    res = subprocess.run(
        ["gemini-api", "files", "upload", str(path), "--wait"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gemini-api files upload failed ({res.returncode}): {res.stderr.strip()}")
    return res.stdout.strip().splitlines()[-1].strip()


def _delete_file(file_id: str) -> None:
    subprocess.run(
        ["gemini-api", "files", "delete", file_id],
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_video_id(target: str, dry_run: bool, uploaded: list[str], tag: str = "vid01") -> str:
    if target.startswith("files/"):
        return target
    if dry_run:
        clean_id = "".join(c for c in Path(target).stem.lower() if c.isalnum()) or tag
        return f"files/dryrun{clean_id}"
    p = Path(target)
    if not p.is_file():
        raise FileNotFoundError(f"video file not found and not a files/<id> URI: {target}")
    fid = _upload_file(p)
    uploaded.append(fid)
    return fid


def _image_part(path: Path, resolution: str, dry_run: bool, uploaded: list[str]) -> dict[str, Any]:
    mime = _mime_of(path, "image/png")
    if dry_run:
        clean_id = "".join(c for c in path.stem.lower() if c.isalnum()) or "img01"
        return {
            "type": "image",
            "uri": f"https://generativelanguage.googleapis.com/v1beta/files/dryrun{clean_id}",
            "mime_type": mime,
            "resolution": resolution,
        }
    if path.stat().st_size <= INLINE_MAX_BYTES:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "data": b64,
            "mime_type": mime,
            "resolution": resolution,
        }
    fid = _upload_file(path)
    uploaded.append(fid)
    return {
        "type": "image",
        "uri": f"https://generativelanguage.googleapis.com/v1beta/{fid}",
        "mime_type": mime,
        "resolution": resolution,
    }


def _run_agent(
    model: str,
    body: dict[str, Any],
    dry_run: bool,
    previous_interaction_id: str | None = None,
) -> dict[str, Any]:
    cmd = [
        "gemini-api",
        "agent",
        "run",
        "-m",
        model,
        "--stream=false",
        "-o",
        "json",
    ]
    if previous_interaction_id:
        cmd.extend(["--previous-interaction-id", previous_interaction_id])
    if dry_run:
        cmd.append("--dry-run")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(body, tf)
        body_path = tf.name
    try:
        cmd.extend(["--body", f"@{body_path}"])
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"gemini-api agent run failed ({res.returncode}): {res.stderr.strip()}")
        return json.loads(res.stdout)
    finally:
        Path(body_path).unlink(missing_ok=True)


def _extract_output_text(resp: dict[str, Any]) -> str:
    if "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"]
    for step in reversed(resp.get("steps") or []):
        if step.get("type") == "model_output":
            for block in step.get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    return str(block["text"])
    return ""


def cmd_boxes(args: argparse.Namespace) -> int:
    img_path = Path(args.image)
    if not args.dry_run and not img_path.is_file():
        raise FileNotFoundError(f"image not found: {img_path}")
    if args.target_size:
        w_str, h_str = args.target_size.lower().split("x", 1)
        target_w, target_h = int(w_str), int(h_str)
    elif args.people:
        target_w, target_h = 1024, 768
    elif img_path.is_file():
        target_w, target_h = _image_size(img_path)
    else:
        target_w, target_h = 1024, 768

    uploaded: list[str] = []
    try:
        inputs: list[dict[str, Any]] = []
        if args.ref:
            ref_path = Path(args.ref)
            inputs.append(_image_part(ref_path, "low", args.dry_run, uploaded))
        inputs.append(_image_part(img_path, args.resolution, args.dry_run, uploaded))

        schema_props: dict[str, Any] = {
            "box_2d": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "The 2D bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000.",
            },
            "label": {
                "type": "string",
                "description": "Descriptive label for the detected subject or part.",
            },
        }
        req_fields = ["box_2d", "label"]
        if args.with_masks:
            schema_props["mask"] = {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Polygon segmentation mask as [[x, y], ...] normalized to 0-1000.",
            }
            req_fields.append("mask")

        prompt_suffix = (
            " Return JSON with `boxes`, where each entry has `box_2d` as "
            "[ymin, xmin, ymax, xmax] normalized to 0-1000"
            + (", `mask` as a polygon of [x, y] normalized to 0-1000," if args.with_masks else "")
            + " and `label`."
        )
        inputs.append({"type": "text", "text": args.prompt.strip() + prompt_suffix})

        body: dict[str, Any] = {
            "input": inputs,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "boxes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": schema_props,
                                "required": req_fields,
                            },
                        }
                    },
                    "required": ["boxes"],
                },
            },
        }
        if args.with_masks:
            body["generation_config"] = {"thinking_level": "minimal"}

        resp = _run_agent(args.model, body, args.dry_run)
        if args.dry_run:
            print(json.dumps(resp, indent=2))
            return 0

        raw_text = _extract_output_text(resp)
        parsed = json.loads(raw_text) if raw_text else {"boxes": []}
        enriched = []
        for item in parsed.get("boxes") or []:
            entry = descale_box_2d(item["box_2d"], target_w, target_h)
            entry["label"] = item.get("label", "")
            if args.with_masks and "mask" in item:
                entry["mask_polygon_px"] = descale_polygon(item["mask"], target_w, target_h)
            if args.people and args.frame is not None:
                entry["sam_match"] = match_sam_people(
                    entry["box_ltwh_px"],
                    (target_w, target_h),
                    Path(args.people),
                    int(args.frame),
                )
            enriched.append(entry)

        print(json.dumps({"canvas": [target_w, target_h], "boxes": enriched}, indent=2))
        return 0
    finally:
        for fid in uploaded:
            _delete_file(fid)


def cmd_slice(args: argparse.Namespace) -> int:
    _check_native_first(args.video, args.force_api, args.dry_run)
    start_s, end_s = frame_span_to_offsets(args.first, args.last, args.fps)
    uploaded: list[str] = []
    try:
        file_id = _resolve_video_id(args.video, args.dry_run, uploaded, tag="slice01")
        body = {
            "input": [
                {
                    "type": "video",
                    "uri": f"https://generativelanguage.googleapis.com/v1beta/{file_id}",
                    "mime_type": "video/mp4",
                    "resolution": args.resolution,
                    "processing": {
                        "type": "static",
                        "start_offset": start_s,
                        "end_offset": end_s,
                        "fps": args.fps,
                    },
                },
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ]
        }
        resp = _run_agent(args.model, body, args.dry_run)
        print(json.dumps(resp, indent=2))
        return 0
    finally:
        for fid in uploaded:
            _delete_file(fid)


def cmd_agentic_video(args: argparse.Namespace) -> int:
    _check_not_stacked_for_agentic(args.video)
    _check_native_first(args.video, args.force_api, args.dry_run)
    if args.compare_shot and not args.compare_shot.startswith("files/"):
        _check_native_first(args.compare_shot, args.force_api, args.dry_run)

    uploaded: list[str] = []
    try:
        master_id = _resolve_video_id(args.video, args.dry_run, uploaded, tag="master01")
        inputs: list[dict[str, Any]] = [
            {
                "type": "video",
                "uri": f"https://generativelanguage.googleapis.com/v1beta/{master_id}",
                "mime_type": "video/mp4",
                "processing": "agentic",
            }
        ]
        if args.compare_shot:
            shot_id = _resolve_video_id(args.compare_shot, args.dry_run, uploaded, tag="shot01")
            shot_proc: dict[str, Any] = {"type": "static", "fps": args.compare_fps}
            if args.compare_first is not None and args.compare_last is not None:
                s0, s1 = frame_span_to_offsets(args.compare_first, args.compare_last, args.compare_fps)
                shot_proc["start_offset"] = s0
                shot_proc["end_offset"] = s1
            inputs.append(
                {
                    "type": "video",
                    "uri": f"https://generativelanguage.googleapis.com/v1beta/{shot_id}",
                    "mime_type": "video/mp4",
                    "resolution": args.compare_resolution,
                    "processing": shot_proc,
                }
            )
        inputs.append({"type": "text", "text": args.prompt})
        body: dict[str, Any] = {"input": inputs}
        resp = _run_agent(
            args.model,
            body,
            args.dry_run,
            previous_interaction_id=args.previous_interaction_id,
        )
        if args.dry_run:
            print(json.dumps(resp, indent=2))
            return 0

        steps = resp.get("steps") or []
        proc_calls = [s for s in steps if s.get("type") in ("processing_call", "processing_result")]
        summary = {
            "interaction_id": resp.get("id"),
            "processing_steps_count": len(proc_calls),
            "processing_steps": proc_calls,
            "output_text": _extract_output_text(resp),
            "usage": resp.get("usage"),
        }
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        for fid in uploaded:
            _delete_file(fid)


def cmd_zoom(args: argparse.Namespace) -> int:
    img_path = Path(args.image)
    uploaded: list[str] = []
    try:
        img_part = _image_part(img_path, args.resolution, args.dry_run, uploaded)
        body = {
            "input": [
                img_part,
                {"type": "text", "text": args.prompt},
            ],
            "tools": [{"type": "code_execution"}],
        }
        resp = _run_agent(args.model, body, args.dry_run)
        print(json.dumps(resp, indent=2))
        return 0
    finally:
        for fid in uploaded:
            _delete_file(fid)


def cmd_transcribe_spans(args: argparse.Namespace) -> int:
    _check_native_first(args.video, args.force_api, args.dry_run)
    uploaded: list[str] = []
    try:
        file_id = _resolve_video_id(args.video, args.dry_run, uploaded, tag="audio01")
        video_part: dict[str, Any] = {
            "type": "video",
            "uri": f"https://generativelanguage.googleapis.com/v1beta/{file_id}",
            "mime_type": _mime_of(Path(args.video), "video/mp4"),
        }
        proc: dict[str, Any] = {"type": "static"}
        if args.visual_speakers:
            video_part["resolution"] = "high"
            proc["fps"] = args.fps
        if args.slice_master and args.shot_last is not None:
            s0, s1 = frame_span_to_offsets(args.shot_first, args.shot_last, args.fps)
            proc["start_offset"] = s0
            proc["end_offset"] = s1
        video_part["processing"] = proc

        prompt_text = (
            args.prompt.strip()
            + " Return JSON with `segments`, where each segment has `speaker` (describing "
            "which subject or voice is singing/speaking), `start_time` and `end_time` in "
            "`MM:SS.mmm` format (relative to the start of the clip/slice), `text` (exact "
            "lyrics or spoken words), and `mouth_visibly_open` (boolean)."
        )
        body: dict[str, Any] = {
            "input": [
                video_part,
                {"type": "text", "text": prompt_text},
            ],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "start_time": {"type": "string"},
                                    "end_time": {"type": "string"},
                                    "text": {"type": "string"},
                                    "mouth_visibly_open": {"type": "boolean"},
                                },
                                "required": [
                                    "speaker",
                                    "start_time",
                                    "end_time",
                                    "text",
                                    "mouth_visibly_open",
                                ],
                            },
                        }
                    },
                    "required": ["segments"],
                },
            },
        }
        resp = _run_agent(args.model, body, args.dry_run)
        if args.dry_run:
            print(json.dumps(resp, indent=2))
            return 0

        raw_text = _extract_output_text(resp)
        parsed = json.loads(raw_text) if raw_text else {"segments": []}
        enriched_segments: list[dict[str, Any]] = []
        for seg in parsed.get("segments") or []:
            f0, f1 = timestamp_span_to_frames(
                seg.get("start_time", "00:00"),
                seg.get("end_time", "00:00"),
                shot_first=args.shot_first,
                fps=args.fps,
                shot_last=args.shot_last,
            )
            enriched_segments.append(
                {
                    "speaker": seg.get("speaker", ""),
                    "start_time": seg.get("start_time", ""),
                    "end_time": seg.get("end_time", ""),
                    "start_frame": f0,
                    "end_frame": f1,
                    "frame_count": f1 - f0 + 1,
                    "mouth_visibly_open": bool(seg.get("mouth_visibly_open", True)),
                    "text": seg.get("text", ""),
                }
            )
        out = {
            "shot_first": args.shot_first,
            "shot_last": args.shot_last,
            "fps": args.fps,
            "visual_speakers": bool(args.visual_speakers),
            "segments": enriched_segments,
        }
        print(json.dumps(out, indent=2))
        return 0
    finally:
        for fid in uploaded:
            _delete_file(fid)


def run_self_test() -> int:
    import numpy as np

    # 1. Verify [ymin, xmin, ymax, xmax] -> [x1, y1, x2, y2] and [left, top, width, height]
    d = descale_box_2d([250, 500, 750, 750], 1024, 768)
    assert d["box_xyxy_px"] == [512.0, 192.0, 768.0, 576.0], d
    assert d["box_ltwh_px"] == [512.0, 192.0, 256.0, 384.0], d

    # 2. Verify 24.0 fps frame-span offset math and timestamp-to-frame math
    s0, s1 = frame_span_to_offsets(4912, 4946, 24.0)
    assert s0 == 204.667 and s1 == 206.125, (s0, s1)
    assert parse_timestamp_seconds("01:02.500") == 62.5
    f0, f1 = timestamp_span_to_frames("00:00.500", "00:01.250", shot_first=4912, fps=24.0, shot_last=4946)
    assert (f0, f1) == (4924, 4942), (f0, f1)

    # 3. Verify SAM 3.1 <preview>_people.npz bit-packed mask IoU matching
    #    and Native-First / *_STACKED refusal guards
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "preview_people.npz"
        mask = np.zeros((768, 1024), dtype=bool)
        mask[192:576, 512:768] = True
        packed = np.packbits(mask, axis=-1)[None, ...]
        np.savez_compressed(
            npz_path,
            masks=packed,
            frame=np.array([10], dtype=np.int32),
            first=np.array(4900, dtype=np.int32),
        )
        (Path(td) / "preview_people.json").write_text(
            json.dumps({"canvas": [1024, 768]}), encoding="utf-8"
        )
        m = match_sam_people(d["box_ltwh_px"], (1024, 768), npz_path, 4910)
        assert m["passes_0_50_gate"] is True, m
        assert m["best_iou"] == 1.0, m
        assert m["sam_box_ltwh"] == [512, 192, 256, 384], m
        assert m["cast_cli_box_arg"] == "4910:512,192,256,384", m

        # Native-First Gate must refuse a <100 MB local .mp4 when a .1fps.*.mp4 sibling exists
        clip_24 = Path(td) / "shot_4912.mp4"
        clip_1fps = Path(td) / "shot_4912.1fps.04912-04946.mp4"
        clip_24.write_bytes(b"00" * 1024)
        clip_1fps.write_bytes(b"00" * 1024)
        try:
            _check_native_first(str(clip_24), force_api=False, dry_run=False)
            raise AssertionError("expected _check_native_first to refuse when .1fps sibling exists")
        except ValueError as exc:
            assert "Native-First (agy) Gate refused" in str(exc), exc

        # *_STACKED refusal guard on agentic-video
        try:
            _check_not_stacked_for_agentic("fun_STACKED.mp4")
            raise AssertionError("expected _check_not_stacked_for_agentic to refuse *_STACKED*")
        except ValueError as exc:
            assert "BANNED" in str(exc), exc

    # 4. Verify gemini-api --dry-run schemas across all 5 subcommands
    for sub_argv in [
        ["boxes", "frame.png", "--prompt", "Detect head", "--with-masks", "--dry-run"],
        ["slice", "files/abc123", "--first", "4912", "--last", "4946", "--prompt", "Inspect shot", "--dry-run"],
        [
            "agentic-video",
            "master_cut.mp4",
            "--compare-shot",
            "candidate_shot.mp4",
            "--prompt",
            "Compare continuity",
            "--dry-run",
        ],
        ["zoom", "strips.jpg", "--prompt", "Zoom into tiles", "--dry-run"],
        [
            "transcribe-spans",
            "shot_4912.mp4",
            "--shot-first",
            "4912",
            "--shot-last",
            "4946",
            "--visual-speakers",
            "--dry-run",
        ],
    ]:
        p = build_parser().parse_args(sub_argv)
        rc = p.func(p)
        assert rc == 0, f"dry-run failed for {sub_argv}"

    print(
        "ok    gemini_media.py self-test passed "
        "(descaling, 24fps offsets, timestamps, SAM people.npz IoU, Native-First & STACKED guards, 5 CLI dry-runs)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run deterministic unit and dry-run tests")
    sub = ap.add_subparsers(dest="subcommand")

    p_boxes = sub.add_parser("boxes", help="2D box / polygon mask grounding + optional SAM 3.1 _people.npz snap")
    p_boxes.add_argument("image", help="Scene frame image path")
    p_boxes.add_argument("--ref", help="Optional reference still (<Picture 1>) to identify a specific cast subject")
    p_boxes.add_argument("--prompt", required=True, help="Detection prompt")
    p_boxes.add_argument("--people", help="Optional <preview>_people.npz path to snap boxes to SAM 3.1 detections")
    p_boxes.add_argument("--frame", type=int, help="Source frame number when --people is passed")
    p_boxes.add_argument("--target-size", help="Target WxH coordinate space (default: 1024x768 with --people, else image size)")
    p_boxes.add_argument("--with-masks", action="store_true", help="Also request polygon segmentation masks")
    p_boxes.add_argument("--resolution", default="high", choices=["low", "medium", "high", "ultra_high"])
    p_boxes.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p_boxes.add_argument("--dry-run", action="store_true", help="Preview request JSON via gemini-api --dry-run")
    p_boxes.set_defaults(func=cmd_boxes)

    p_slice = sub.add_parser("slice", help="Inspect a [first..last] frame slice on a video via static start_offset/end_offset")
    p_slice.add_argument("video", help="Uploaded files/<id> or local .mp4 path")
    p_slice.add_argument("--first", type=int, required=True, help="Inclusive start frame number")
    p_slice.add_argument("--last", type=int, required=True, help="Inclusive end frame number")
    p_slice.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24.0)")
    p_slice.add_argument("--prompt", required=True, help="Inspection prompt")
    p_slice.add_argument("--resolution", default="high", choices=["low", "medium", "high"])
    p_slice.add_argument("--force-api", action="store_true", help="Override Native-First Gate even if a .1fps.mp4 sibling exists")
    p_slice.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p_slice.add_argument("--dry-run", action="store_true", help="Preview request JSON via gemini-api --dry-run")
    p_slice.set_defaults(func=cmd_slice)

    p_agentic = sub.add_parser(
        "agentic-video",
        help="Opt-in agentic video timeline navigation on >100 MB single-canvas master cuts (+ optional --compare-shot)",
    )
    p_agentic.add_argument("video", help="Uploaded files/<id> or local single-canvas master cut .mp4")
    p_agentic.add_argument("--compare-shot", help="Optional second video (candidate shot) inspected in 24 fps static high-res mode")
    p_agentic.add_argument("--compare-first", type=int, help="Optional start frame on --compare-shot")
    p_agentic.add_argument("--compare-last", type=int, help="Optional end frame on --compare-shot")
    p_agentic.add_argument("--compare-fps", type=float, default=24.0, help="Frame rate for --compare-shot (default: 24.0)")
    p_agentic.add_argument("--compare-resolution", default="high", choices=["low", "medium", "high"])
    p_agentic.add_argument("--previous-interaction-id", help="Stateful multi-turn follow-up interaction ID")
    p_agentic.add_argument("--prompt", required=True, help="Timeline query or continuity comparison prompt")
    p_agentic.add_argument("--force-api", action="store_true", help="Override Native-First Gate even if a .1fps.mp4 sibling exists")
    p_agentic.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p_agentic.add_argument("--dry-run", action="store_true", help="Preview request JSON via gemini-api --dry-run")
    p_agentic.set_defaults(func=cmd_agentic_video)

    p_zoom = sub.add_parser("zoom", help="Agentic code-execution zoom-and-inspect on diagnostic strips or contact sheets")
    p_zoom.add_argument("image", help="Diagnostic strip (--strips) or contact sheet image path")
    p_zoom.add_argument("--prompt", required=True, help="Inspection prompt")
    p_zoom.add_argument("--resolution", default="ultra_high", choices=["high", "ultra_high"])
    p_zoom.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p_zoom.add_argument("--dry-run", action="store_true", help="Preview request JSON via gemini-api --dry-run")
    p_zoom.set_defaults(func=cmd_zoom)

    p_trans = sub.add_parser(
        "transcribe-spans",
        help="Speaker/lyric transcription converted to 24.0 fps [start_frame, end_frame] spans",
    )
    p_trans.add_argument("video", help="Uploaded files/<id> or local .mp4/.wav path")
    p_trans.add_argument("--shot-first", type=int, default=0, help="Source frame corresponding to clip start (default: 0)")
    p_trans.add_argument("--shot-last", type=int, help="Optional inclusive end frame of the shot")
    p_trans.add_argument("--slice-master", action="store_true", help="Apply start_offset/end_offset using --shot-first/--shot-last on a full master cut")
    p_trans.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24.0)")
    p_trans.add_argument(
        "--visual-speakers",
        action="store_true",
        help="Inspect video at 24 fps high resolution alongside audio to attribute vocals to specific on-screen subjects",
    )
    p_trans.add_argument(
        "--prompt",
        default="Transcribe all singing and spoken lines with accurate timestamps and identify which subject is active.",
        help="Transcription prompt",
    )
    p_trans.add_argument("--force-api", action="store_true", help="Override Native-First Gate even if a .1fps.mp4 sibling exists")
    p_trans.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p_trans.add_argument("--dry-run", action="store_true", help="Preview request JSON via gemini-api --dry-run")
    p_trans.set_defaults(func=cmd_transcribe_spans)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.self_test:
        return run_self_test()
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
