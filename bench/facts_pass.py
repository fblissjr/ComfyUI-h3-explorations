#!/usr/bin/env python3
"""Emit a per-frame deterministic geometry and mask facts table for a masked-run capture.

Reads a ``capture_masked_run.py`` directory (``manifest.json``, ``classes__<by>.npz``,
``subjects/<label>/masks__<by>.npz``, ``subjects/<label>/head_tube__<by>.npz``,
``runs/<run>/region.npz``, optional window ``*_region.npz`` matte files, and optional
``<reference>.json`` record from ``head_reference.py``) and produces a JSON + Markdown
report with one row per source frame in the window and a notable-spans summary block.

Every measurement is reported as data with what it was measured on and what it
cannot see; no notable span is a pass/fail gate. Hard structural errors (missing
files, unreadable archives, mismatched frame counts) refuse cleanly via SystemExit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Inherited from vendor/meta_sapiens2/seg_classes29.json ("names"[3] == "Face_Neck").
SAPIENS_FACE_NECK_CLASS = 3

#: Inherited from vendor/meta_sapiens2/seg_classes29.json ("names"[4] == "Hair").
SAPIENS_HAIR_CLASS = 4

#: Inherited from video_mask.py::_DEFAULT_CELL_PX (16 px = 2 VAE cells at stride 8).
DEFAULT_CELL_PX = 16

#: Inherited from video_mask.py (matte threshold > 0.01 marks active soft-matte pixels).
MATTE_ACTIVE_THRESHOLD = 0.01

#: Inherited from docs/wiki/handover.md item 1c (12-frame rolling median window, 0.5 s at 24 fps).
ROLLING_MEDIAN_WINDOW = 12

#: Inherited from docs/wiki/handover.md item 1c (parts_held drop < 0.5 * rolling 12f median).
PARTS_HELD_DROP_RATIO = 0.5

#: Reasoned description of what facts_pass.py cannot see from geometry and mask arrays alone.
CANNOT_SEE = (
    "pixel appearance, identity likeness, lip-sync accuracy, temporal flicker, "
    "and whether a mouth mode or region boundary looks natural in the rendered video"
)


def _die(msg: str) -> None:
    raise SystemExit(f"facts_pass.py: {msg}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _die(f"{label} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _die(f"failed to parse {label} ({path}): {exc}")
    if not isinstance(data, dict):
        _die(f"{label} ({path}) must be a JSON object")
    return data


def _load_npz(path: Path, label: str) -> Any:
    if not path.is_file():
        _die(f"{label} not found: {path}")
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        _die(f"failed to load {label} ({path}): {exc}")


def _unpack_mask(arr: np.ndarray, frames: int, h: int, w: int, label: str) -> np.ndarray:
    """Unpack a [frames, h, w] bool/uint8 or bit-packed [frames, h, ceil(w/8)] uint8 mask array."""
    if arr.ndim != 3 or arr.shape[0] != frames or arr.shape[1] != h:
        _die(
            f"{label} shape {arr.shape} does not match expected "
            f"({frames}, {h}, {w}) or packed width ({frames}, {h}, {(w + 7) // 8})"
        )
    if arr.dtype == np.bool_:
        if arr.shape[2] != w:
            _die(f"{label} bool width {arr.shape[2]} != {w}")
        return arr
    if arr.dtype == np.uint8:
        packed_w = (w + 7) // 8
        if arr.shape[2] == packed_w and w != packed_w:
            return np.unpackbits(arr, axis=-1)[..., :w].astype(bool)
        if arr.shape[2] == w:
            return arr > 0
        if arr.shape[2] == packed_w:
            return np.unpackbits(arr, axis=-1)[..., :w].astype(bool)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.shape[2] != w:
            _die(f"{label} float width {arr.shape[2]} != {w}")
        return arr > MATTE_ACTIVE_THRESHOLD
    _die(f"unsupported array dtype {arr.dtype} and shape {arr.shape} for {label}")
    raise AssertionError("unreachable")


def _cells_up(cells: np.ndarray, cell_px: int, h: int, w: int, label: str) -> np.ndarray:
    """Expand a [frames, h // cell_px, w // cell_px] cell grid to [frames, h, w] pixel resolution."""
    if cells.ndim != 3:
        _die(f"{label} must be 3D, got shape {cells.shape}")
    if cells.shape[1] == h and cells.shape[2] == w:
        return cells.astype(bool)
    expected_hc = h // cell_px
    expected_wc = w // cell_px
    if cells.shape[1] != expected_hc or cells.shape[2] != expected_wc:
        _die(
            f"{label} shape {cells.shape} does not match cell grid "
            f"({cells.shape[0]}, {expected_hc}, {expected_wc}) or pixel grid ({cells.shape[0]}, {h}, {w})"
        )
    up = np.repeat(np.repeat(cells.astype(bool), cell_px, axis=1), cell_px, axis=2)
    return up[:, :h, :w]


def _resolve_face_hair_classes(manifest: dict[str, Any]) -> tuple[int, int]:
    """Return (face_neck_class_idx, hair_class_idx) from manifest or Sapiens-2 defaults."""
    face_idx = SAPIENS_FACE_NECK_CLASS
    hair_idx = SAPIENS_HAIR_CLASS
    names = manifest.get("class_names")
    if isinstance(names, list):
        if "Face_Neck" in names:
            face_idx = int(names.index("Face_Neck"))
        if "Hair" in names:
            hair_idx = int(names.index("Hair"))
    return face_idx, hair_idx


def _normalize_sightings(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized sighting dicts (each with 'label' and 'by') from manifest.json.

    Supports both the real capture_masked_run.py layout:
      manifest["subjects"] = [{"label": "<lbl>", "sightings": [{"by": "<by>", ...}]}, ...]
    and flat synthetic fixtures:
      manifest["sightings"] = [{"label": "<lbl>", "by": "<by>", ...}, ...]
    """
    normalized: list[dict[str, Any]] = []
    subjects_list = manifest.get("subjects")
    if isinstance(subjects_list, list):
        for subj_entry in subjects_list:
            if not isinstance(subj_entry, dict):
                continue
            lbl = str(subj_entry.get("label") or "")
            subj_sightings = subj_entry.get("sightings")
            if isinstance(subj_sightings, list) and subj_sightings:
                for s in subj_sightings:
                    if isinstance(s, dict):
                        merged = dict(s)
                        if lbl and not merged.get("label"):
                            merged["label"] = lbl
                        normalized.append(merged)
            elif lbl:
                normalized.append(dict(subj_entry))

    top_sightings = manifest.get("sightings")
    if isinstance(top_sightings, list):
        for s in top_sightings:
            if isinstance(s, dict):
                normalized.append(dict(s))

    return normalized


def _find_sighting_and_run(
    manifest: dict[str, Any],
    subject: str | None,
    run_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str, list[dict[str, Any]]]:
    """Resolve the target subject sighting and run entry from manifest.json."""
    sightings = _normalize_sightings(manifest)
    if not sightings:
        _die(
            "manifest.json must contain a non-empty 'subjects' (with 'sightings') "
            "or 'sightings' list"
        )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        _die("manifest.json must contain a non-empty 'runs' list")

    chosen_run: dict[str, Any] | None = None
    if run_name is not None:
        for r in runs:
            if isinstance(r, dict) and r.get("name") == run_name:
                chosen_run = r
                break
        if chosen_run is None:
            available = [r.get("name") for r in runs if isinstance(r, dict)]
            _die(f"run {run_name!r} not found in manifest.json (available: {available})")
    elif subject is not None:
        matching = [r for r in runs if isinstance(r, dict) and r.get("subject") == subject]
        if not matching:
            available = [r.get("name") for r in runs if isinstance(r, dict)]
            _die(f"no run for subject {subject!r} in manifest.json (available runs: {available})")
        chosen_run = matching[0]
    else:
        chosen_run = runs[0] if isinstance(runs[0], dict) else None
        if chosen_run is None:
            _die("first entry in manifest.json 'runs' is not an object")

    resolved_run_name = str(chosen_run.get("name") or "")
    resolved_subject = str(subject or chosen_run.get("subject") or "")
    if not resolved_subject:
        _die(f"run {resolved_run_name!r} does not specify a 'subject'")

    chosen_sighting: dict[str, Any] | None = None
    for s in sightings:
        if s.get("label") == resolved_subject:
            chosen_sighting = s
            break
    if chosen_sighting is None:
        available_s = [s.get("label") for s in sightings]
        _die(
            f"subject {resolved_subject!r} not found in manifest.json subjects/sightings "
            f"(available: {available_s})"
        )
    if not resolved_run_name:
        resolved_run_name = resolved_subject

    return chosen_sighting, chosen_run, resolved_subject, resolved_run_name, sightings


def _load_classes_array(
    capture_dir: Path,
    manifest: dict[str, Any],
    sighting: dict[str, Any],
    resolved_subject: str,
    frames: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Locate and load the per-frame Sapiens class map [frames, h, w] uint8."""
    by = str(sighting.get("by") or "yolo_sam2")
    part_by = str(manifest.get("part_by") or "sapiens2")
    files_dict = sighting.get("files") if isinstance(sighting.get("files"), dict) else {}

    rel_candidates: list[str] = []
    if isinstance(manifest.get("classes_file"), str):
        rel_candidates.append(manifest["classes_file"])
    if isinstance(files_dict, dict) and isinstance(files_dict.get("classes"), str):
        rel_candidates.append(files_dict["classes"])
    rel_candidates.extend(
        [
            f"subjects/{resolved_subject}/classes__{by}.npz",
            f"classes__{by}.npz",
            f"classes__{part_by}.npz",
            f"subjects/{resolved_subject}/head_tube__{by}.npz",
        ]
    )

    cand_paths: list[Path] = [
        (Path(r) if Path(r).is_absolute() else (capture_dir / r)) for r in rel_candidates
    ]
    subj_dir = capture_dir / "subjects" / resolved_subject
    if subj_dir.is_dir():
        cand_paths.extend(sorted(subj_dir.glob("classes__*.npz")))
        cand_paths.extend(sorted(subj_dir.glob("head_tube__*.npz")))
    cand_paths.extend(sorted(capture_dir.glob("classes__*.npz")))

    for cand in cand_paths:
        if not cand.is_file():
            continue
        npz = _load_npz(cand, f"classes npz ({cand})")
        for key in ("classes", "class_map"):
            if key in npz.files:
                arr = npz[key]
                if arr.shape != (frames, h, w):
                    _die(
                        f"{cand}[{key!r}] shape {arr.shape} != expected ({frames}, {h}, {w})"
                    )
                return arr

    _die(
        f"classes npz not found for subject {resolved_subject!r} (by={by!r}) in {capture_dir}"
    )
    raise AssertionError("unreachable")


def _load_head_track(
    capture_dir: Path,
    sighting: dict[str, Any],
    subj_masks_npz: Any,
    frames: int,
    h: int,
    w: int,
) -> tuple[np.ndarray | None, str | None]:
    """Load optional per-frame head_track / head_tube boolean mask [frames, h, w] if available."""
    for key in ("head_track", "head_tube", "tube"):
        if key in subj_masks_npz.files:
            return (
                _unpack_mask(
                    subj_masks_npz[key],
                    frames,
                    h,
                    w,
                    f"masks.npz[{key!r}]",
                ),
                f"masks.npz:{key}",
            )

    files_dict = sighting.get("files") if isinstance(sighting.get("files"), dict) else {}
    rel_candidates: list[str] = []
    if isinstance(files_dict, dict):
        for k in ("head_track", "head_tube"):
            if isinstance(files_dict.get(k), str):
                rel_candidates.append(files_dict[k])
    for k in ("head_track", "head_tube"):
        if isinstance(sighting.get(k), str):
            rel_candidates.append(sighting[k])
    inputs_dict = sighting.get("inputs") if isinstance(sighting.get("inputs"), dict) else {}
    if isinstance(inputs_dict, dict):
        for k in ("head_track", "head_tube"):
            ht_in = inputs_dict.get(k)
            if isinstance(ht_in, dict) and isinstance(ht_in.get("file"), str):
                rel_candidates.append(ht_in["file"])

    label = str(sighting.get("label", ""))
    by = str(sighting.get("by", "yolo_sam2"))
    if label:
        rel_candidates.extend(
            [
                f"subjects/{label}/head_tube__{by}.npz",
                f"subjects/{label}/head_track__{by}.npz",
                f"subjects/{label}/head_track__{by}.mkv",
            ]
        )

    cand_paths: list[Path] = [
        (Path(r) if Path(r).is_absolute() else (capture_dir / r)) for r in rel_candidates
    ]
    if label and (capture_dir / "subjects" / label).is_dir():
        subj_dir = capture_dir / "subjects" / label
        cand_paths.extend(sorted(subj_dir.glob("head_tube__*.npz")))
        cand_paths.extend(sorted(subj_dir.glob("head_track__*.npz")))

    for cand in cand_paths:
        if not cand.is_file():
            continue
        if cand.suffix == ".npz":
            npz = _load_npz(cand, f"head_track ({cand})")
            key = ""
            for k in ("head_track", "tube", "head_tube"):
                if k in npz.files:
                    key = k
                    break
            if not key and npz.files:
                key = npz.files[0]
            if not key:
                _die(f"head_track npz {cand} has no arrays")
            return _unpack_mask(npz[key], frames, h, w, f"{cand}[{key!r}]"), f"{cand.name}:{key}"
        if cand.suffix in (".mkv", ".mp4"):
            from _lib.frames import read_mask

            arr = read_mask(cand)
            if arr.shape[0] != frames or arr.shape[1] != h or arr.shape[2] != w:
                _die(
                    f"head_track video {cand} shape {arr.shape} != ({frames}, {h}, {w})"
                )
            return arr > MATTE_ACTIVE_THRESHOLD, str(cand.name)

    return None, None


def _load_window_matte(
    capture_dir: Path,
    run_entry: dict[str, Any],
    run_npz: Any,
    frames: int,
    h: int,
    w: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None, str | None]:
    """Locate and load optional soft matte from run_npz or window *_region.npz."""
    for key in ("matte", "soft_matte"):
        if key in run_npz.files:
            arr = run_npz[key]
            if arr.ndim == 3 and arr.shape == (frames, h, w):
                if arr.dtype == np.uint8:
                    return (arr.astype(np.float32) / 255.0) > MATTE_ACTIVE_THRESHOLD, None, f"region.npz:{key}"
                return arr > MATTE_ACTIVE_THRESHOLD, None, f"region.npz:{key}"

    run_name = str(run_entry.get("name") or "")
    run_dir = (capture_dir / "runs" / run_name) if run_name else capture_dir
    files_field = run_entry.get("files")
    candidates: list[Path] = []
    if isinstance(files_field, dict):
        for k in ("window_region", "matte", "window_matte"):
            val = files_field.get(k)
            if isinstance(val, str):
                candidates.append(Path(val) if Path(val).is_absolute() else (capture_dir / val))
    elif isinstance(files_field, list):
        for item in files_field:
            if isinstance(item, str):
                for base in (run_dir / "predicted", run_dir, capture_dir):
                    candidates.append(base / item)

    for k in ("window_region", "matte", "window_matte"):
        val_top = run_entry.get(k)
        if isinstance(val_top, str):
            candidates.append(Path(val_top) if Path(val_top).is_absolute() else (capture_dir / val_top))

    if run_dir.is_dir():
        candidates.extend(sorted(run_dir.glob("*_window_*_region.npz")))
        if (run_dir / "predicted").is_dir():
            candidates.extend(sorted((run_dir / "predicted").glob("*_window_*_region.npz")))
    candidates.extend(sorted(capture_dir.glob("*_window_*_region.npz")))

    first_meta_obj: dict[str, Any] | None = None
    for cand in candidates:
        if not cand.is_file():
            continue
        npz = _load_npz(cand, f"window region ({cand})")
        meta_obj: dict[str, Any] | None = None
        if "meta" in npz.files:
            try:
                raw_meta = npz["meta"]
                meta_str = str(raw_meta.item() if raw_meta.ndim == 0 else raw_meta[0])
                parsed = json.loads(meta_str)
                if isinstance(parsed, dict):
                    meta_obj = parsed
                    if first_meta_obj is None:
                        first_meta_obj = parsed
            except Exception:
                meta_obj = None
        for key in ("matte", "mask", "carried"):
            if key in npz.files:
                arr = npz[key]
                if arr.ndim == 3 and arr.shape[0] == frames and arr.shape[1] == h:
                    if arr.dtype == np.uint8 and arr.shape[2] == w:
                        mask = (arr.astype(np.float32) / 255.0) > MATTE_ACTIVE_THRESHOLD
                    else:
                        mask = _unpack_mask(arr, frames, h, w, f"{cand}[{key!r}]")
                    return mask, meta_obj, str(cand.name)
    return None, first_meta_obj, None


def _discover_reference_json(
    capture_dir: Path,
    manifest: dict[str, Any],
    run_entry: dict[str, Any],
    reference_arg: Path | None,
    job_arg: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Locate and load the optional head_reference.py .json record."""
    if reference_arg is not None:
        ref_path = reference_arg
        if ref_path.suffix in (".mp4", ".mkv", ".png"):
            json_cand = ref_path.with_suffix(".json")
            if json_cand.is_file():
                ref_path = json_cand
        return _load_json(ref_path, "reference record"), str(ref_path)

    candidates: list[Path] = []
    if job_arg is not None:
        job_data = _load_json(job_arg, "job JSON")
        _collect_ref_candidates_from_job(job_data, job_arg.parent, candidates)

    for key in ("reference_json", "reference"):
        for container in (run_entry, manifest):
            val = container.get(key)
            if isinstance(val, str):
                p = Path(val) if Path(val).is_absolute() else (capture_dir / val)
                if p.suffix in (".mp4", ".mkv", ".png"):
                    p = p.with_suffix(".json")
                candidates.append(p)
            elif isinstance(val, dict) and isinstance(val.get("file"), str):
                p = Path(val["file"]) if Path(val["file"]).is_absolute() else (capture_dir / val["file"])
                if p.suffix in (".mp4", ".mkv", ".png"):
                    p = p.with_suffix(".json")
                candidates.append(p)

    # Check sibling <capture_dir>_job directory for job.render.json / job.plan.json / job.json
    run_name = str(run_entry.get("name") or "")
    sibling_job_dir = capture_dir.parent / f"{capture_dir.name}_job"
    if sibling_job_dir.is_dir():
        job_files = [
            sibling_job_dir / "job.render.json",
            sibling_job_dir / "job.plan.json",
            sibling_job_dir / "job.json",
        ]
        if run_name:
            job_files.extend(
                [
                    sibling_job_dir / f"{run_name}.render.json",
                    sibling_job_dir / f"{run_name}.plan.json",
                ]
            )
        for jf in job_files:
            if jf.is_file():
                try:
                    jd = json.loads(jf.read_text(encoding="utf-8"))
                    if isinstance(jd, dict):
                        _collect_ref_candidates_from_job(jd, jf.parent, candidates)
                except Exception:
                    pass

    for cand in candidates:
        if cand.is_file():
            return _load_json(cand, "reference record"), str(cand)

    return None, None


def _collect_ref_candidates_from_job(
    job_data: dict[str, Any], base_dir: Path, out: list[Path]
) -> None:
    for key in ("reference_json", "reference"):
        val = job_data.get(key)
        if isinstance(val, str):
            p = Path(val) if Path(val).is_absolute() else (base_dir / val)
            if p.suffix in (".mp4", ".mkv", ".png"):
                p = p.with_suffix(".json")
            out.append(p)
    loads = job_data.get("loads")
    if isinstance(loads, list):
        for load in loads:
            if isinstance(load, dict):
                mf = load.get("motion_file") or load.get("reference_file")
                if isinstance(mf, str):
                    p = Path(mf) if Path(mf).is_absolute() else (base_dir / mf)
                    out.append(p.with_suffix(".json"))


def _expand_spans(spans_obj: Any, offset: int = 0) -> set[int]:
    """Expand a list of ints or [start, end] inclusive spans into a set of frame indices."""
    out: set[int] = set()
    if not isinstance(spans_obj, list):
        return out
    for item in spans_obj:
        if isinstance(item, int):
            out.add(item + offset)
        elif (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            for f in range(item[0], item[1] + 1):
                out.add(f + offset)
    return out


def _resolve_mouth_modes(
    ref_record: dict[str, Any] | None,
    source_frames: list[int],
) -> dict[int, str]:
    """Map each source_frame in source_frames to its mouth_mode string."""
    if ref_record is None:
        return {sf: "unknown" for sf in source_frames}

    # Direct per-frame override map or list (supported for synthetic or pre-indexed records).
    direct = ref_record.get("mouth_modes") or ref_record.get("mouth_mode_by_frame")
    if isinstance(direct, dict):
        result: dict[int, str] = {}
        for sf in source_frames:
            val = direct.get(str(sf), direct.get(sf))
            result[sf] = str(val) if val is not None else "unknown"
        return result
    if isinstance(direct, list) and len(direct) == len(source_frames):
        return {sf: str(direct[i]) for i, sf in enumerate(source_frames)}

    drawn = ref_record.get("drawn")
    if not isinstance(drawn, dict):
        return {sf: "unknown" for sf in source_frames}

    first_frame = ref_record.get("first_frame")
    if not isinstance(first_frame, int):
        first_frame = ref_record.get("its_first_frame_is_source_frame")
    offset = int(first_frame) if isinstance(first_frame, int) else 0

    def _to_source_set(raw: Any) -> set[int]:
        raw_set = _expand_spans(raw, offset=0)
        if not raw_set:
            return set()
        sf_set = set(source_frames)
        if offset > 0 and not (raw_set & sf_set) and ({x + offset for x in raw_set} & sf_set):
            return {x + offset for x in raw_set}
        return raw_set

    bare_frames = _to_source_set(drawn.get("frames_left_bare"))
    for nf_key in ("no_face_on", "no_draw_face_on", "no_head_on"):
        no_face_on = drawn.get(nf_key)
        if isinstance(no_face_on, dict):
            if "frames" in no_face_on:
                bare_frames |= _to_source_set(no_face_on.get("frames"))
            else:
                for v in no_face_on.values():
                    bare_frames |= _to_source_set(v)
        elif isinstance(no_face_on, list):
            bare_frames |= _to_source_set(no_face_on)

    mouth_info = drawn.get("mouth")
    if not isinstance(mouth_info, dict):
        parts_left_out = drawn.get("parts_left_out")
        if isinstance(parts_left_out, list) and "mouth" in parts_left_out:
            return {
                sf: ("no_face" if sf in bare_frames else "no_mouth")
                for sf in source_frames
            }
        return {
            sf: ("no_face" if sf in bare_frames else "unknown")
            for sf in source_frames
        }

    mouth_drawn_flag = bool(mouth_info.get("drawn", True))
    no_mouth_frames: set[int] = set()
    no_mouth_on = mouth_info.get("no_mouth_on")
    if isinstance(no_mouth_on, dict):
        for v in no_mouth_on.values():
            no_mouth_frames |= _to_source_set(v)
    elif isinstance(no_mouth_on, list):
        no_mouth_frames |= _to_source_set(no_mouth_on)

    cm_hole = mouth_info.get("class_mouth_needs_hole")
    if isinstance(cm_hole, dict):
        no_mouth_frames |= _to_source_set(cm_hole.get("no_mouth_on"))

    closed_frames: set[int] = set()
    cll = mouth_info.get("closed_lip_line")
    if isinstance(cll, dict):
        closed_frames |= _to_source_set(cll.get("closed_frames"))
    elif isinstance(cll, list):
        closed_frames |= _to_source_set(cll)

    worded_frames: set[int] = set()
    mouth_on = mouth_info.get("mouth_on")
    if isinstance(mouth_on, dict):
        worded_frames |= _to_source_set(mouth_on.get("drawn_on_the_word"))
    elif isinstance(mouth_on, list):
        worded_frames |= _to_source_set(mouth_on)

    class_map_frames = _to_source_set(mouth_info.get("frames_by_the_class_map"))
    lip_line_frames = _to_source_set(mouth_info.get("frames_by_the_lip_lines"))

    modes: dict[int, str] = {}
    for sf in source_frames:
        if sf in bare_frames:
            modes[sf] = "no_face"
        elif not mouth_drawn_flag or sf in no_mouth_frames:
            modes[sf] = "no_mouth"
        elif sf in closed_frames:
            modes[sf] = "closed_lip_line"
        elif sf in worded_frames:
            modes[sf] = "worded"
        elif sf in class_map_frames:
            modes[sf] = "class_map"
        elif sf in lip_line_frames:
            modes[sf] = "lip_lines"
        elif mouth_drawn_flag:
            modes[sf] = "lip_lines"
        else:
            modes[sf] = "unknown"
    return modes


def _extract_ref_crop_and_scale(
    ref_record: dict[str, Any] | None,
) -> tuple[list[int] | None, float | list[float] | None]:
    """Extract (ref_crop_box, ref_scale) from a head_reference.py or crop_reference_to_head.py JSON record."""
    if ref_record is None:
        return None, None
    box = ref_record.get("box") or ref_record.get("crop_box") or ref_record.get("ref_crop_box")
    crop_box: list[int] | None = None
    if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        crop_box = [int(v) for v in box]

    scale_raw = ref_record.get("scale") if "scale" in ref_record else ref_record.get("ref_scale")
    ref_scale: float | list[float] | None = None
    if isinstance(scale_raw, (int, float)):
        ref_scale = round(float(scale_raw), 6)
    elif (
        isinstance(scale_raw, list)
        and len(scale_raw) == 2
        and all(isinstance(v, (int, float)) for v in scale_raw)
    ):
        ref_scale = [round(float(scale_raw[0]), 6), round(float(scale_raw[1]), 6)]
    return crop_box, ref_scale


def _resolve_zoom_box(
    run_entry: dict[str, Any],
    window_meta: dict[str, Any] | None,
    w: int,
    h: int,
) -> list[int]:
    """Resolve zoom_box [x0, y0, x1, y1] from run entry or window metadata, defaulting to full canvas."""
    for src in (run_entry, window_meta or {}):
        z = src.get("zoom")
        if isinstance(z, dict):
            box = z.get("box")
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                return [int(v) for v in box]
        if isinstance(z, list) and len(z) == 4 and all(isinstance(v, (int, float)) for v in z):
            return [int(v) for v in z]
        zb = src.get("zoom_box")
        if isinstance(zb, list) and len(zb) == 4 and all(isinstance(v, (int, float)) for v in zb):
            return [int(v) for v in zb]
    return [0, 0, w, h]


def _contiguous_spans(
    rows: list[dict[str, Any]],
    predicate: Any,
    value_key: str,
) -> list[dict[str, Any]]:
    """Group consecutive rows satisfying predicate(row) into [first_source_frame, last_source_frame] spans."""
    spans: list[dict[str, Any]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_count = 0
    cur_max = 0
    cur_sum = 0

    for r in rows:
        sf = int(r["source_frame"])
        val = int(r[value_key])
        if predicate(r):
            if cur_start is None:
                cur_start = sf
                cur_end = sf
                cur_count = 1
                cur_max = val
                cur_sum = val
            else:
                cur_end = sf
                cur_count += 1
                if val > cur_max:
                    cur_max = val
                cur_sum += val
        else:
            if cur_start is not None and cur_end is not None:
                spans.append(
                    {
                        "source_frames": [cur_start, cur_end],
                        "frame_count": cur_count,
                        f"max_{value_key}": cur_max,
                        f"sum_{value_key}": cur_sum,
                    }
                )
                cur_start = None
                cur_end = None
    if cur_start is not None and cur_end is not None:
        spans.append(
            {
                "source_frames": [cur_start, cur_end],
                "frame_count": cur_count,
                f"max_{value_key}": cur_max,
                f"sum_{value_key}": cur_sum,
            }
        )
    return spans


def _rolling_median(values: list[int], window: int) -> list[float]:
    """Compute centered rolling median over `window` frames (clipped at boundaries)."""
    n = len(values)
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        slice_vals = values[lo:hi]
        out.append(float(np.median(slice_vals)) if slice_vals else 0.0)
    return out


def compute_facts(
    capture_dir: Path,
    *,
    subject: str | None = None,
    run_name: str | None = None,
    first_frame: int | None = None,
    last_frame: int | None = None,
    reference_path: Path | None = None,
    job_path: Path | None = None,
) -> dict[str, Any]:
    """Compute per-frame geometry and mask facts for a capture directory."""
    if not capture_dir.is_dir():
        _die(f"capture directory not found: {capture_dir}")

    manifest_path = capture_dir / "manifest.json"
    manifest = _load_json(manifest_path, "manifest.json")

    frames_total = manifest.get("frames")
    size = manifest.get("size")
    if not isinstance(frames_total, int) or frames_total <= 0:
        _die(f"manifest.json 'frames' must be a positive integer, got {frames_total!r}")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(v, int) and v > 0 for v in size)
    ):
        _die(f"manifest.json 'size' must be [width, height], got {size!r}")
    w, h = int(size[0]), int(size[1])

    cell_px = int(manifest.get("cell_px") or DEFAULT_CELL_PX)
    if cell_px <= 0 or (h % cell_px != 0) or (w % cell_px != 0):
        _die(f"invalid cell_px {cell_px} for canvas size {w}x{h}")

    sighting, run_entry, resolved_subject, resolved_run_name, all_sightings = _find_sighting_and_run(
        manifest, subject, run_name
    )

    classes_arr = _load_classes_array(
        capture_dir, manifest, sighting, resolved_subject, frames_total, h, w
    )
    face_cls, hair_cls = _resolve_face_hair_classes(manifest)

    # Resolve subject masks__<by>.npz
    sighting_files = sighting.get("files") if isinstance(sighting.get("files"), dict) else {}
    masks_rel = sighting_files.get("masks") if isinstance(sighting_files, dict) else None
    if not isinstance(masks_rel, str):
        by = str(sighting.get("by") or "yolo_sam2")
        masks_rel = f"subjects/{resolved_subject}/masks__{by}.npz"
    masks_path = Path(masks_rel) if Path(masks_rel).is_absolute() else (capture_dir / masks_rel)
    subj_masks_npz = _load_npz(masks_path, f"subject {resolved_subject!r} masks npz")
    if "track" not in subj_masks_npz.files:
        _die(f"subject masks npz ({masks_path}) is missing 'track' array")
    track_mask = _unpack_mask(
        subj_masks_npz["track"], frames_total, h, w, f"{masks_path}['track']"
    )

    head_track_mask, head_track_source = _load_head_track(
        capture_dir, sighting, subj_masks_npz, frames_total, h, w
    )

    # Resolve run region.npz
    run_files = run_entry.get("files") if isinstance(run_entry.get("files"), dict) else {}
    region_rel = run_files.get("region") if isinstance(run_files, dict) else None
    if not isinstance(region_rel, str):
        region_rel = f"runs/{resolved_run_name}/region.npz"
    region_path = (
        Path(region_rel) if Path(region_rel).is_absolute() else (capture_dir / region_rel)
    )
    run_npz = _load_npz(region_path, f"run {resolved_run_name!r} region npz")
    if "region" not in run_npz.files:
        _die(f"run region npz ({region_path}) is missing 'region' array")
    if "carried" not in run_npz.files:
        _die(f"run region npz ({region_path}) is missing 'carried' array")

    raw_region = run_npz["region"]
    if raw_region.ndim != 3 or raw_region.shape[0] != frames_total:
        _die(
            f"region array shape {raw_region.shape} does not have {frames_total} frames"
        )
    if raw_region.shape[1] == h // cell_px and raw_region.shape[2] == w // cell_px:
        region_cells_arr = raw_region.astype(bool)
        region_px_mask = _cells_up(region_cells_arr, cell_px, h, w, "region")
    elif raw_region.shape[1] == h and raw_region.shape[2] == w:
        region_px_mask = raw_region.astype(bool)
        region_cells_arr = (
            region_px_mask.reshape(frames_total, h // cell_px, cell_px, w // cell_px, cell_px)
            .any(axis=(2, 4))
        )
    else:
        _die(
            f"region array shape {raw_region.shape} matches neither cell grid "
            f"({frames_total}, {h // cell_px}, {w // cell_px}) nor pixel grid ({frames_total}, {h}, {w})"
        )

    carried_mask = _unpack_mask(
        run_npz["carried"], frames_total, h, w, f"{region_path}['carried']"
    )

    if "parts_held" in subj_masks_npz.files:
        parts_held_mask = _unpack_mask(
            subj_masks_npz["parts_held"],
            frames_total,
            h,
            w,
            f"{masks_path}['parts_held']",
        )
    elif "parts" in subj_masks_npz.files:
        parts_held_mask = _unpack_mask(
            subj_masks_npz["parts"],
            frames_total,
            h,
            w,
            f"{masks_path}['parts']",
        )
    else:
        parts_held_mask = carried_mask

    # Load other subjects' tracks and given_back__<other> arrays for other_overlap_px
    other_tracks: dict[str, np.ndarray] = {}
    for s in all_sightings:
        lbl = str(s.get("label") or "")
        if not lbl or lbl == resolved_subject or lbl in other_tracks:
            continue
        s_files = s.get("files") if isinstance(s.get("files"), dict) else {}
        s_masks_rel = s_files.get("masks") if isinstance(s_files, dict) else None
        if not isinstance(s_masks_rel, str):
            s_by = str(s.get("by") or "yolo_sam2")
            s_masks_rel = f"subjects/{lbl}/masks__{s_by}.npz"
        s_path = (
            Path(s_masks_rel) if Path(s_masks_rel).is_absolute() else (capture_dir / s_masks_rel)
        )
        if s_path.is_file():
            s_npz = _load_npz(s_path, f"other subject {lbl!r} masks npz")
            if "track" in s_npz.files:
                other_tracks[lbl] = _unpack_mask(
                    s_npz["track"], frames_total, h, w, f"{s_path}['track']"
                )

    given_back_px_by_other: dict[str, np.ndarray] = {}
    for key in run_npz.files:
        if key.startswith("given_back__"):
            other_lbl = key[len("given_back__") :]
            gb_arr = run_npz[key]
            if gb_arr.shape[0] != frames_total:
                _die(f"{region_path}[{key!r}] frame count {gb_arr.shape[0]} != {frames_total}")
            given_back_px_by_other[other_lbl] = _cells_up(
                gb_arr, cell_px, h, w, f"{region_path}[{key!r}]"
            )

    window_matte_mask, window_meta, matte_source = _load_window_matte(
        capture_dir, run_entry, run_npz, frames_total, h, w
    )

    ref_record, ref_source = _discover_reference_json(
        capture_dir, manifest, run_entry, reference_path, job_path
    )
    ref_crop_box, ref_scale = _extract_ref_crop_and_scale(ref_record)
    zoom_box = _resolve_zoom_box(run_entry, window_meta, w, h)

    # Resolve window frame bounds (0-based indices into capture arrays, and source_frame numbers)
    base_first_source = int(manifest.get("first_frame") or 0)
    run_first_default = (
        run_entry.get("first_frame")
        if run_entry.get("first_frame") is not None
        else run_entry.get("first_source_frame")
    )
    win_first = (
        int(first_frame)
        if first_frame is not None
        else int(run_first_default if run_first_default is not None else base_first_source)
    )
    win_last = (
        int(last_frame)
        if last_frame is not None
        else int(
            run_entry.get("last_frame")
            if run_entry.get("last_frame") is not None
            else (base_first_source + frames_total - 1)
        )
    )

    idx_start = win_first - base_first_source
    idx_end = win_last - base_first_source
    if idx_start < 0 or idx_end >= frames_total or idx_start > idx_end:
        _die(
            f"requested source frame window [{win_first}, {win_last}] is outside "
            f"capture range [{base_first_source}, {base_first_source + frames_total - 1}]"
        )

    source_frames = list(range(win_first, win_last + 1))
    mouth_modes = _resolve_mouth_modes(ref_record, source_frames)

    rows: list[dict[str, Any]] = []
    for window_idx, sf in enumerate(source_frames):
        n = sf - base_first_source
        trk_n = track_mask[n]
        cls_n = classes_arr[n]
        s_face_n = trk_n & (cls_n == face_cls)
        s_hair_n = trk_n & (cls_n == hair_cls)
        if head_track_mask is not None:
            ht_n = head_track_mask[n]
            face_combined_n = s_face_n | (ht_n & ~s_hair_n)
            head_track_px_val: int | None = int(ht_n.sum())
        else:
            face_combined_n = s_face_n
            head_track_px_val = None

        reg_cells_n = region_cells_arr[n]
        reg_px_n = region_px_mask[n]
        carried_n = carried_mask[n]
        parts_held_n = parts_held_mask[n]

        face_out_n = face_combined_n & ~reg_px_n
        if window_matte_mask is not None:
            matte_px_val = int(window_matte_mask[n].sum())
        else:
            matte_px_val = int(carried_n.sum())

        other_union_n = np.zeros((h, w), dtype=bool)
        for other_lbl, o_trk in other_tracks.items():
            other_union_n |= trk_n & o_trk[n]
            other_union_n |= reg_px_n & o_trk[n]
        for other_lbl, gb_px in given_back_px_by_other.items():
            other_union_n |= gb_px[n]

        row: dict[str, Any] = {
            "window_frame": window_idx,
            "source_frame": sf,
            "source_track_px": int(trk_n.sum()),
            "sapiens_face_px": int(s_face_n.sum()),
            "sapiens_hair_px": int(s_hair_n.sum()),
            "head_track_px": head_track_px_val,
            "parts_held_px": int(parts_held_n.sum()),
            "region_cells": int(reg_cells_n.sum()),
            "region_px": int(reg_px_n.sum()),
            "face_outside_region_px": int(face_out_n.sum()),
            "matte_px": matte_px_val,
            "other_overlap_px": int(other_union_n.sum()),
            "mouth_mode": mouth_modes.get(sf, "unknown"),
            "ref_crop_box": ref_crop_box,
            "ref_scale": ref_scale,
            "zoom_box": zoom_box,
        }
        rows.append(row)

    # Compute notable spans (data only, never a pass/fail gate)
    face_outside_spans = _contiguous_spans(
        rows, lambda r: int(r["face_outside_region_px"]) > 0, "face_outside_region_px"
    )
    other_overlap_spans = _contiguous_spans(
        rows, lambda r: int(r["other_overlap_px"]) > 0, "other_overlap_px"
    )

    parts_vals = [int(r["parts_held_px"]) for r in rows]
    medians = _rolling_median(parts_vals, ROLLING_MEDIAN_WINDOW)
    for r, med in zip(rows, medians):
        r["parts_held_rolling_12f_median"] = round(med, 2)

    parts_drop_spans = _contiguous_spans(
        rows,
        lambda r: (
            int(r["source_track_px"]) > 0
            and float(r["parts_held_rolling_12f_median"]) > 0.0
            and int(r["parts_held_px"]) < PARTS_HELD_DROP_RATIO * float(r["parts_held_rolling_12f_median"])
        ),
        "parts_held_px",
    )

    mouth_transitions: list[dict[str, Any]] = []
    for i in range(1, len(rows)):
        prev_m = str(rows[i - 1]["mouth_mode"])
        cur_m = str(rows[i]["mouth_mode"])
        if prev_m != cur_m:
            mouth_transitions.append(
                {
                    "from_source_frame": int(rows[i - 1]["source_frame"]),
                    "to_source_frame": int(rows[i]["source_frame"]),
                    "from_mode": prev_m,
                    "to_mode": cur_m,
                }
            )

    notable_spans = {
        "face_outside_region": face_outside_spans,
        "parts_held_drop_below_half_rolling_12f_median": parts_drop_spans,
        "other_overlap": other_overlap_spans,
        "mouth_mode_transitions": mouth_transitions,
    }

    return {
        "schema": "h3.facts_pass.v1",
        "what_measured": {
            "capture_dir": str(capture_dir.name),
            "subject": resolved_subject,
            "run": resolved_run_name,
            "canvas_size": [w, h],
            "cell_px": cell_px,
            "window_source_frames": [win_first, win_last],
            "frame_count": len(rows),
            "head_track_source": head_track_source,
            "matte_source": matte_source or "region.npz:carried",
            "reference_source": Path(ref_source).name if ref_source else None,
            "other_subjects": sorted(set(other_tracks.keys()) | set(given_back_px_by_other.keys())),
        },
        "cannot_see": CANNOT_SEE,
        "notable_spans": notable_spans,
        "frames": rows,
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Format a compute_facts() report dictionary as a concise Markdown document."""
    wm = report["what_measured"]
    ns = report["notable_spans"]
    rows = report["frames"]

    lines: list[str] = [
        f"# Facts Pass: `{wm['subject']}` (`{wm['run']}`)",
        "",
        "## What was measured",
        "",
        f"- **Capture**: `{wm['capture_dir']}`",
        f"- **Subject / run**: `{wm['subject']}` / `{wm['run']}`",
        f"- **Canvas**: `{wm['canvas_size'][0]}x{wm['canvas_size'][1]}` (`cell_px={wm['cell_px']}`)",
        f"- **Source frames**: `{wm['window_source_frames'][0]}..{wm['window_source_frames'][1]}` (`{wm['frame_count']}` frames)",
        f"- **Head track source**: `{wm['head_track_source'] or 'none'}`",
        f"- **Matte source**: `{wm['matte_source']}`",
        f"- **Reference record**: `{wm['reference_source'] or 'none'}`",
        f"- **Other subjects**: `{', '.join(wm['other_subjects']) if wm['other_subjects'] else 'none'}`",
        f"- **Cannot see**: {report['cannot_see']}",
        "",
        "## Notable spans",
        "",
    ]

    fo = ns["face_outside_region"]
    if fo:
        spans_str = ", ".join(
            f"`{s['source_frames'][0]}..{s['source_frames'][1]}` (max `{s['max_face_outside_region_px']}` px)"
            for s in fo
        )
        lines.append(f"- **`face_outside_region_px > 0`**: {spans_str}")
    else:
        lines.append("- **`face_outside_region_px > 0`**: none")

    pd = ns["parts_held_drop_below_half_rolling_12f_median"]
    if pd:
        spans_str = ", ".join(
            f"`{s['source_frames'][0]}..{s['source_frames'][1]}` ({s['frame_count']}f)"
            for s in pd
        )
        lines.append(f"- **`parts_held_px < 0.5 * rolling_12f_median`**: {spans_str}")
    else:
        lines.append("- **`parts_held_px < 0.5 * rolling_12f_median`**: none")

    oo = ns["other_overlap"]
    if oo:
        spans_str = ", ".join(
            f"`{s['source_frames'][0]}..{s['source_frames'][1]}` (max `{s['max_other_overlap_px']}` px)"
            for s in oo
        )
        lines.append(f"- **`other_overlap_px > 0`**: {spans_str}")
    else:
        lines.append("- **`other_overlap_px > 0`**: none")

    mt = ns["mouth_mode_transitions"]
    if mt:
        trans_str = ", ".join(
            f"`{t['from_source_frame']}->{t['to_source_frame']}` (`{t['from_mode']}->{t['to_mode']}`)"
            for t in mt
        )
        lines.append(f"- **`mouth_mode` transitions**: {trans_str}")
    else:
        lines.append("- **`mouth_mode` transitions**: none")

    lines.extend(
        [
            "",
            "## Per-frame table",
            "",
            "| win_f | src_f | track_px | face_px | hair_px | head_trk_px | parts_held_px | reg_cells | reg_px | face_out_px | matte_px | other_ovl_px | mouth_mode | ref_crop_box | ref_scale | zoom_box |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
        ]
    )

    for r in rows:
        ht_str = str(r["head_track_px"]) if r["head_track_px"] is not None else "null"
        rc_str = json.dumps(r["ref_crop_box"]) if r["ref_crop_box"] is not None else "null"
        rs_str = json.dumps(r["ref_scale"]) if r["ref_scale"] is not None else "null"
        zb_str = json.dumps(r["zoom_box"])
        lines.append(
            f"| {r['window_frame']} | {r['source_frame']} | {r['source_track_px']} | "
            f"{r['sapiens_face_px']} | {r['sapiens_hair_px']} | {ht_str} | "
            f"{r['parts_held_px']} | {r['region_cells']} | {r['region_px']} | "
            f"{r['face_outside_region_px']} | {r['matte_px']} | {r['other_overlap_px']} | "
            f"{r['mouth_mode']} | {rc_str} | {rs_str} | {zb_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "capture_dir",
        type=Path,
        help="Path to the capture_masked_run.py directory containing manifest.json",
    )
    ap.add_argument("--subject", type=str, default=None, help="Subject label in manifest.json")
    ap.add_argument("--run", type=str, default=None, help="Run name in manifest.json")
    ap.add_argument("--first-frame", type=int, default=None, help="First source frame (inclusive)")
    ap.add_argument("--last-frame", type=int, default=None, help="Last source frame (inclusive)")
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional path to head_reference.py .json record (or .mp4 sibling)",
    )
    ap.add_argument(
        "--job",
        type=Path,
        default=None,
        help="Optional path to job.json to discover reference record",
    )
    ap.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON report to this path",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Write Markdown report to this path",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compute_facts(
        args.capture_dir,
        subject=args.subject,
        run_name=args.run,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        reference_path=args.reference,
        job_path=args.job,
    )
    md_text = format_markdown(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_text, encoding="utf-8")

    if args.output_json is None and args.output_md is None:
        print(md_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
