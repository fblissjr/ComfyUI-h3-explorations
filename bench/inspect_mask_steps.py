#!/usr/bin/env python3
"""Forensic per-frame and per-latent-step mask, region, provenance, and keep-out inspector for a masked v2v capture.

    <python> bench/inspect_mask_steps.py --capture data/<day>_<run> [--run <name>] [--frames 4920-4927,4939-4946] [--problems-only] [--strips <out.jpg>] [--json <out.json>]

Audits eight forensic categories across single-pass, multi-pass (Pass 1/2/3), single-window, and multi-window chained captures:
0. **Input & pass-dependency provenance** (`provenance`): checks `laid.jsonl` chronologically (separating passes laid *before*
   this capture was built from passes laid *after*), checks `preview` and `strangers` source videos against `manifest["clip"]`,
   distinguishes inputs that were already stale at build time (`stale_at_build`) from code updated after the capture ran
   (`code_updated_since_capture`), and measures prior-pass DiT region overlap and SAM track collapse on Pass 2+ loads.
1. **Latent step vs. frame mapping & held-padding bleed** (`pad_bleed_steps`): maps every `(1, 4, 4, 4, 4)` latent step per window
   to its window frames and source frames (`first..last`, `TRIM_k`, `PAD_k`), and flags any boundary latent step where held `PAD`
   frames carry a mask (`> 0 px`) while real shot frames in the same latent step have `0 px`.
2. **Blue-only frames** (`blue_only_frames`): frames with `0 px` red mask (`carried`) inside an active blue latent step (left unlaid
   by `video_mask.unlaid_frames` while the DiT regenerates those tokens across the 4-frame step).
3. **Speck-in-blue frames** (`speck_in_region_frames`): frames with a small red mask sliver (`< 25%` of the step's max red mask and
   below the subject's part floor) inside an active blue latent step, which causes `unlaid_frames` to composite only that speck.
4. **Edge-shrunk whole-head frames** (`edge_shrunk_whole_frames`): frames inside `whole_head_frames` whose mask area drops below
   `75%` of an immediate neighbour (`i - 1` or `i + 1`) right before or after an occlusion gap.
5. **Strangers self-swallow frames** (`strangers_self_swallow_frames`): frames where `carried < 50%` of the subject's median and
   `raw_strangers` covers `>= 30%` of the nearest clean reference mask (excluding `Apparel` and other people's faces).
6. **Missed SAM 3.1 strangers inside blue region** (`missed_sam_strangers_in_region`): non-subject people detected by SAM 3.1
   (`_people.npz`, `score >= 0.70`, excluding all tracked subjects in `manifest["subjects"]`) that overlap the active blue region
   by `>= 256 px` while whole-frame Sapiens2 `strangers.mkv` covers `< 25%` of that overlap inside the region.
7. **Per-frame gap recovery & visual crop strips** (`gap_recovery`, `--strips`): for every frame where `carried < 50%` of median
   (and not intentionally dropped via `--drop` / `--no-face-on`), reports what a stationary reference from the nearest non-shrunk
   clean frame recovers (`replaces_hair=True` uses `head_track | carried`; face-only uses `carried`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# H3 VideoVAE temporal token cadence (`comfy.ldm.minimax.model.FRAME_PER_TOKEN` and `video_mask.run_lengths`):
# Cyclic 5-step / 17-frame cadence (1, 4, 4, 4, 4).
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def run_lengths(latent_t: int) -> list[int]:
    return [FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t))]


def _parse_frames(spec: str | None, first: int, n: int) -> list[int]:
    if not spec:
        return list(range(first, first + n))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            for f in range(int(a_s), int(b_s) + 1):
                if first <= f < first + n:
                    out.add(f)
        else:
            f = int(part)
            if first <= f < first + n:
                out.add(f)
    return sorted(out)


def _read_mkv_bool(path: Path, n: int, h: int, w: int, start_offset: int = 0) -> np.ndarray | None:
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    if start_offset > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        g = frame[:, :, 0] if frame.ndim == 3 else frame
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(g > 127)
    cap.release()
    if len(frames) != n:
        return None
    return np.stack(frames, axis=0)


def _unpack_npz_mask(npz_path: Path, key: str, width: int) -> np.ndarray | None:
    if not npz_path.is_file():
        return None
    z = np.load(npz_path)
    if key not in z.files:
        return None
    arr = z[key]
    if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[-1] == (width + 7) // 8:
        return np.unpackbits(arr, axis=-1)[..., :width].astype(bool)
    return (arr > 0).astype(bool)


def _read_head_mask(subj_dir: Path, by: str, n: int, h: int, w: int) -> np.ndarray | None:
    ht_npz = subj_dir / f"head_tube__{by}.npz"
    if not ht_npz.is_file() and subj_dir.is_dir():
        cands = sorted(subj_dir.glob("head_tube__*.npz"))
        if cands:
            ht_npz = cands[0]
    if ht_npz.is_file():
        z = np.load(ht_npz)
        if "tube" in z.files:
            arr = z["tube"].astype(bool)
            if arr.shape == (n, h, w):
                return arr
    return _read_mkv_bool(subj_dir / f"head_track__{by}.mkv", n, h, w)


def _project_tokens_to_canvas(
    w_tok: np.ndarray,
    zoom_box: list[int] | None,
    h: int,
    w: int,
) -> np.ndarray:
    """Project zoomed `[T, 48, 64]` token grid back onto `[T, h, w]` canvas boolean mask."""
    t_steps, gh, gw = w_tok.shape
    out = np.zeros((t_steps, h, w), dtype=bool)
    if zoom_box and len(zoom_box) == 4:
        x0, y0, x1, y1 = [int(v) for v in zoom_box]
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    for t in range(t_steps):
        if not w_tok[t].any():
            continue
        u8 = w_tok[t].astype(np.uint8) * 255
        resized = cv2.resize(u8, (bw, bh), interpolation=cv2.INTER_NEAREST)
        out[t, y0:y1, x0:x1] = resized > 127
    return out


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            local_tz = dt.datetime.now().astimezone().tzinfo
            d = d.replace(tzinfo=local_tz)
        return d.timestamp()
    except Exception:
        return None


PRODUCER_CODE_GROUPS: dict[str, list[str]] = {
    "track": ["subject_track.py", "sam31_corrections.py", "shot_table.py"],
    "head_track": ["subject_track.py", "sam31_corrections.py"],
    "parts": ["sapiens2_parts.py", "head_tube_parts.py", "bench/head_tube_parts.py"],
    "classes": ["sapiens2_parts.py"],
    "pose": ["sapiens2_Pose.py", "bench/pose_along_shot.py"],
    "points": ["bench/pose_along_shot.py", "bench/head_reference.py", "bench/draw_face_points.py"],
    "strangers": ["bench/strangers.py", "sapiens2_parts.py"],
}


def _stat_dates(path: Path | None) -> dict[str, Any]:
    """Return on-disk creation/birth time (%w) and modification time (%y, %Y) via `stat`."""
    if path is None or not path.exists():
        return {"created": None, "modified": None, "mtime_epoch": None}
    try:
        out = subprocess.check_output(
            ["stat", "-c", "%w|%y|%Y", str(path)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        b_s, m_s, ep_s = out.split("|", 2)
        return {
            "created": None if b_s.startswith("-") else b_s[:19],
            "modified": m_s[:19],
            "mtime_epoch": float(ep_s),
        }
    except Exception:
        st = path.stat()
        m_iso = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return {"created": None, "modified": m_iso, "mtime_epoch": float(st.st_mtime)}


def _git_latest_commit(paths: list[str], before_epoch: float | None = None) -> tuple[str, float, str]:
    existing = [p for p in paths if (REPO / p).is_file()]
    if not existing:
        return ("", 0.0, "")
    try:
        cmd = ["git", "log", "-n", "1", "--format=%h %ct %cs"]
        if before_epoch is not None:
            cmd.append(f"--before={int(before_epoch)}")
        cmd.append("--")
        cmd.extend(existing)
        out = subprocess.check_output(cmd, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            return ("", 0.0, "")
        parts = out.split(" ", 2)
        return (parts[0], float(parts[1]), parts[2] if len(parts) > 2 else "")
    except Exception:
        return ("", 0.0, "")


def inspect_capture(capture_dir: Path, run_name: str | None = None) -> dict[str, Any]:
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        if capture_dir.name.endswith("_first"):
            alt = capture_dir.with_name(capture_dir.name[:-6])
            raise ValueError(f"missing {manifest_path} (try on-plan capture directory: {alt})")
        raise ValueError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    first = int(manifest["first_frame"])
    n = int(manifest["frames"])
    w, h = int(manifest["size"][0]), int(manifest["size"][1])

    runs = manifest.get("runs") or []
    if not runs:
        if (capture_dir.parent / f"{capture_dir.name}_first").is_dir():
            raise ValueError(f"no runs in {manifest_path}")
        raise ValueError(f"no runs in {manifest_path} (is this a _first pre-plan directory? pass the on-plan capture directory)")
    run_entry = runs[0] if run_name is None else next((r for r in runs if r["name"] == run_name), None)
    if run_entry is None:
        raise ValueError(f"run {run_name!r} not found in {manifest_path}")
    run_name = run_entry["name"]
    subj = run_entry["subject"]
    run_dir = capture_dir / "runs" / run_name

    subj_entry = next((s for s in manifest.get("subjects", []) if s["label"] == subj), None)
    if subj_entry is None or not subj_entry.get("sightings"):
        raise ValueError(f"subject {subj!r} has no sightings in {manifest_path}")
    seen = subj_entry["sightings"][0]
    by = seen["by"]
    shots_first = int(seen.get("shots_first_source_frame", first))
    part_classes = list(seen.get("part_is_made_of") or [])
    replaces_hair = ("Hair" in part_classes) or (seen.get("hair_in_hole") is not None)

    subj_dir = capture_dir / "subjects" / subj
    masks_npz = subj_dir / f"masks__{by}.npz"
    track = _unpack_npz_mask(masks_npz, "track", w)
    parts = _unpack_npz_mask(masks_npz, "parts", w)
    head_track = _read_head_mask(subj_dir, by, n, h, w)
    parts_held = _read_mkv_bool(subj_dir / f"parts_held__{by}.mkv", n, h, w)
    carried_mkv = _read_mkv_bool(run_dir / "carried.mkv", n, h, w)
    region_mkv = _read_mkv_bool(run_dir / "region.mkv", n, h, w)

    region_npz_path = run_dir / "region.npz"
    carried_npz = _unpack_npz_mask(region_npz_path, "carried", w) if region_npz_path.is_file() else None
    if track is None:
        raise ValueError(f"missing track in {masks_npz}")
    carried_mask = (
        carried_mkv
        if carried_mkv is not None
        else (carried_npz if carried_npz is not None else (parts_held if parts_held is not None else parts))
    )
    if carried_mask is None:
        raise ValueError(f"missing carried/parts_held mask for {subj}")

    # Load all other named subjects' tracks so [6] never confuses another tracked subject with an unnamed stranger
    other_subject_tracks: dict[str, np.ndarray] = {}
    for s_other in manifest.get("subjects", []):
        lbl_o = s_other.get("label", "")
        if lbl_o in ("", subj, "strangers") or not s_other.get("sightings"):
            continue
        by_o = s_other["sightings"][0].get("by", "")
        tr_o = _unpack_npz_mask(capture_dir / "subjects" / lbl_o / f"masks__{by_o}.npz", "track", w)
        if tr_o is not None and len(tr_o) == n:
            other_subject_tracks[lbl_o] = tr_o

    if region_npz_path.is_file():
        r_npz = np.load(region_npz_path)
        r_arr = (r_npz["region"] > 0).astype(bool)
        if r_arr.shape[1:] == (h // 16, w // 16):
            region_cells = r_arr.reshape(n, -1).sum(axis=-1)
            region_mask = np.repeat(np.repeat(r_arr, 16, axis=1), 16, axis=2)[:, :h, :w]
        else:
            region_mask = r_arr
            region_cells = r_arr.reshape(n, h // 16, 16, w // 16, 16).any(axis=(2, 4)).sum(axis=(1, 2))
    elif region_mkv is not None:
        region_mask = region_mkv
        region_cells = region_mkv.reshape(n, h // 16, 16, w // 16, 16).any(axis=(2, 4)).sum(axis=(1, 2))
    else:
        raise ValueError(f"missing region.npz and region.mkv in {run_dir}")

    # Derive video project name from clip path (e.g. "fun" from "fun_0000_6394...")
    clip_name = Path(manifest.get("clip") or "").name
    video_proj = clip_name.split("_")[0] if "_" in clip_name else "fun"
    out_root = Path(os.environ.get("H3_OUTPUT_DIR", "output")) / "Video"
    vid_out_root = out_root / video_proj if (out_root / video_proj).is_dir() else out_root

    # Load strangers mask (both carved in capture and raw on share)
    str_subj = next((s for s in manifest.get("subjects", []) if s["label"] == "strangers"), None)
    str_carved: np.ndarray | None = None
    raw_strangers: np.ndarray | None = None
    whole_classes: np.ndarray | None = None
    people_npz_path: Path | None = None
    strangers_rec_path: Path | None = None
    str_first = shots_first
    if str_subj and str_subj.get("sightings"):
        s_seen = str_subj["sightings"][0]
        s_by = s_seen.get("by", "")
        str_first = int(s_seen.get("shots_first_source_frame", s_seen.get("first_source_frame", shots_first)))
        str_carved = _unpack_npz_mask(capture_dir / "subjects" / "strangers" / f"masks__{s_by}.npz", "track", w)
        tr_name = s_seen.get("track", "")
        if tr_name and vid_out_root.is_dir():
            candidates = list(vid_out_root.glob(f"keep_outs/{s_by}/{tr_name}")) or list(vid_out_root.glob(f"keep_outs/{tr_name}"))
            if not candidates:
                candidates = list(out_root.glob(f"*/keep_outs/{s_by}/{tr_name}")) or list(out_root.glob(f"*/keep_outs/{tr_name}"))
            if candidates:
                tr_p = candidates[0]
                str_offset = max(0, first - str_first)
                raw_strangers = _read_mkv_bool(tr_p, n, h, w, start_offset=str_offset)
                rec_p = tr_p.with_suffix(".json")
                if rec_p.is_file():
                    strangers_rec_path = rec_p
                    rec = json.loads(rec_p.read_text())
                    rec_f = rec.get("first_source_frame", rec.get("first"))
                    if rec_f is not None:
                        str_first = int(rec_f)
                        str_offset = max(0, first - str_first)
                        raw_strangers = _read_mkv_bool(tr_p, n, h, w, start_offset=str_offset)
                    wc_info = rec.get("whole_frame_classes")
                    wc_file = wc_info if isinstance(wc_info, str) else (wc_info.get("file") if isinstance(wc_info, dict) else None)
                    if wc_file:
                        wc_p = Path(wc_file) if Path(wc_file).is_file() else (tr_p.parent / Path(wc_file).name)
                        if wc_p.is_file():
                            wc_z = np.load(wc_p)
                            wc_first = int(wc_z["first"]) if "first" in wc_z.files else str_first
                            wc_arr = wc_z["classes"]
                            wc_off = first - wc_first
                            if 0 <= wc_off and wc_off + n <= len(wc_arr):
                                whole_classes = wc_arr[wc_off : wc_off + n]
                    pp_info = rec.get("people_file") or rec.get("people")
                    pp_name = pp_info if isinstance(pp_info, str) else (pp_info.get("file") if isinstance(pp_info, dict) else None)
                    if pp_name:
                        pp_cands = [Path(pp_name), tr_p.parent / Path(pp_name).name] + list(vid_out_root.glob(f"*/{s_by}/{Path(pp_name).name}"))
                        for pp_c in pp_cands:
                            if pp_c.is_file():
                                people_npz_path = pp_c
                                break
                    if people_npz_path is None and vid_out_root.is_dir():
                        pp_cands = list(vid_out_root.glob(f"*/{s_by}/{s_by}_people.npz"))
                        if pp_cands:
                            people_npz_path = pp_cands[0]

    # --- 0. Input & Pass-Dependency Provenance Audit ---
    data_root = capture_dir.resolve().parent
    laid_path = data_root / video_proj / "laid.jsonl"
    if not laid_path.is_file():
        laid_path = REPO / "data" / "fun" / "laid.jsonl"

    # Estimate when this capture was built from input timestamps or manifest mtime
    input_epochs: list[float] = []
    for v_in in (seen.get("inputs") or {}).values():
        if isinstance(v_in, dict):
            ep = _iso_to_epoch(v_in.get("written"))
            if ep is not None:
                input_epochs.append(ep)
    cap_mtime = manifest_path.stat().st_mtime
    cap_build_epoch = max(input_epochs) if input_epochs else cap_mtime

    prior_laid_passes: list[dict[str, Any]] = []
    later_laid_passes: list[dict[str, Any]] = []
    self_laid_epoch: float | None = None
    if laid_path.is_file():
        all_laid = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        for r_laid in all_laid:
            if r_laid.get("load") == run_name:
                self_laid_epoch = _iso_to_epoch(r_laid.get("added"))
        ref_epoch = min(cap_build_epoch, self_laid_epoch) if self_laid_epoch is not None else cap_build_epoch
        for r_laid in all_laid:
            if r_laid.get("subject") == subj:
                continue
            spans = r_laid.get("frames") or [[int(r_laid.get("first", -1)), int(r_laid.get("last", -1))]]
            overlaps_span = any(int(sp[0]) <= first + n - 1 and int(sp[1]) >= first for sp in spans if len(sp) == 2)
            if not overlaps_span:
                continue
            added_str = r_laid.get("added")
            added_ep = _iso_to_epoch(added_str)
            entry = {
                "load": r_laid.get("load"),
                "subject": r_laid.get("subject"),
                "frames": spans,
                "added": added_str,
                "file": r_laid.get("file"),
            }
            # A laid pass is only a PRIOR pass for this capture if it was laid before this capture was built
            # (with a 600s grace window if built right as the cut was being assembled).
            if added_ep is not None and added_ep > ref_epoch + 600:
                later_laid_passes.append(entry)
            else:
                prior_laid_passes.append(entry)

    # Find preview job folder to check what source video the preview ran on
    preview_source: str | None = None
    preview_queued_at: str | None = None
    for p_job in sorted(data_root.glob(f"*_{by}_job/job.json")):
        try:
            pj = json.loads(p_job.read_text())
            src_raw = pj.get("video") or (pj.get("job") or {}).get("source") or ""
            if src_raw:
                preview_source = Path(str(src_raw)).name
            pq = p_job.parent / "preview.queued.json"
            if pq.is_file():
                pq_j = json.loads(pq.read_text())
                preview_queued_at = pq_j.get("completed_at") or pq_j.get("queued_at")
        except Exception:
            pass
    if preview_source is None and vid_out_root.is_dir():
        png_name = f"{subj}_{by}_00001_.png"
        for p_png in vid_out_root.glob(f"*/{by}/{png_name}"):
            wf_p = p_png.with_name(p_png.stem + "_workflow.json")
            if wf_p.is_file():
                try:
                    wf = json.loads(wf_p.read_text())
                    for nd in wf.values():
                        if isinstance(nd, dict) and "video" in (nd.get("inputs") or {}):
                            v_val = nd["inputs"]["video"]
                            if isinstance(v_val, str):
                                preview_source = Path(v_val).name
                                break
                except Exception:
                    pass
            break

    strangers_source: str | None = None
    if strangers_rec_path and strangers_rec_path.is_file():
        try:
            sr = json.loads(strangers_rec_path.read_text())
            strangers_source = Path(sr.get("source") or "").name
        except Exception:
            pass

    provenance_mismatches: list[str] = []
    if preview_source and clip_name and preview_source != clip_name:
        provenance_mismatches.append(f"capture clip={clip_name!r} != preview source={preview_source!r}")
    if strangers_source and clip_name and strangers_source != clip_name:
        provenance_mismatches.append(f"capture clip={clip_name!r} != strangers source={strangers_source!r}")
    if prior_laid_passes and clip_name and "from_laid" not in clip_name and "joined" not in clip_name:
        prior_names = [f"{p['load']} ({p['subject']})" for p in prior_laid_passes]
        provenance_mismatches.append(
            f"prior pass(es) {prior_names} were laid before this capture, "
            f"but capture clip is {clip_name!r} (expected joined post-pass plate)"
        )

    # Check input timestamps (recorded `written` AND on-disk create/modify dates) vs manifest and producer code git commits
    manifest_dates = _stat_dates(manifest_path)
    in_root = Path(os.environ.get("H3_INPUT_DIR", "input"))
    clip_path_on_disk: Path | None = None
    if manifest.get("clip") and in_root.is_dir():
        c_p = in_root / str(manifest["clip"])
        if c_p.is_file():
            clip_path_on_disk = c_p
        else:
            cands = list(in_root.rglob(clip_name))
            if cands:
                clip_path_on_disk = cands[0]
    clip_dates = _stat_dates(clip_path_on_disk)

    stale_at_build: list[dict[str, Any]] = []
    code_updated_since_capture: list[dict[str, Any]] = []
    input_file_dates: dict[str, dict[str, Any]] = {}
    inputs_dict = seen.get("inputs") or {}
    for k_in, v_in in inputs_dict.items():
        if not isinstance(v_in, dict) or not v_in.get("written"):
            continue
        f_name = str(v_in.get("file") or "")
        w_str = str(v_in["written"])
        w_ep = _iso_to_epoch(w_str)
        # Locate input file on share to inspect its creation (%w) and modification (%y) dates
        disk_p: Path | None = None
        if f_name and vid_out_root.is_dir():
            f_cands = (
                list(vid_out_root.glob(f"*/{by}/{f_name}"))
                + list(vid_out_root.glob(f"keep_outs/{by}/{f_name}"))
                + list(vid_out_root.glob(f"keep_outs/{f_name}"))
            )
            if f_cands:
                disk_p = f_cands[0]
        d_dates = _stat_dates(disk_p)
        input_file_dates[k_in] = {
            "file": f_name,
            "manifest_written": w_str,
            "disk_created": d_dates["created"],
            "disk_modified": d_dates["modified"],
        }
        if (
            d_dates["mtime_epoch"] is not None
            and manifest_dates["mtime_epoch"] is not None
            and d_dates["mtime_epoch"] > manifest_dates["mtime_epoch"] + 5
        ):
            provenance_mismatches.append(
                f"input[{k_in}] {f_name!r} modified on disk ({d_dates['modified']}) "
                f"AFTER manifest.json was last updated ({manifest_dates['modified']})"
            )
        if w_ep is None:
            continue
        code_files = PRODUCER_CODE_GROUPS.get(k_in, [])
        if not code_files:
            continue
        # 1. Was the input already older than a producer commit when this capture was built?
        b_hash, b_ep, b_date = _git_latest_commit(code_files, before_epoch=cap_build_epoch)
        if b_ep > 0 and b_ep > w_ep + 120:
            stale_at_build.append({
                "input": k_in,
                "file": f_name,
                "input_written": w_str,
                "disk_created": d_dates["created"],
                "disk_modified": d_dates["modified"],
                "latest_code_commit": b_hash,
                "latest_code_date": b_date,
                "code_files": code_files,
            })
        else:
            # 2. Did producer code change on HEAD after this capture was built?
            h_hash, h_ep, h_date = _git_latest_commit(code_files, before_epoch=None)
            if h_ep > 0 and h_ep > max(w_ep, cap_build_epoch) + 120:
                code_updated_since_capture.append({
                    "input": k_in,
                    "file": f_name,
                    "input_written": w_str,
                    "disk_created": d_dates["created"],
                    "disk_modified": d_dates["modified"],
                    "latest_code_commit": h_hash,
                    "latest_code_date": h_date,
                    "code_files": code_files,
                })

    # Audit MoGe-2 depth (facebook_hyden-mogev2-metric-point via measure_moge2_depth.py -> depth.npz) & contested owner pixels
    owners_info = manifest.get("owners") or {}
    claimed_multi_px = int(owners_info.get("claimed_by_more_than_one_px", 0) or 0)
    contested_px = int(owners_info.get("contested_px", 0) or 0)
    depth_meta = owners_info.get("depth")
    depth_npz_path = capture_dir / "depth.npz"
    depth_dates = _stat_dates(depth_npz_path) if depth_npz_path.is_file() else None
    carved_keep_out_px = 0
    for s_o in manifest.get("subjects", []):
        for st_o in s_o.get("sightings", []):
            co = st_o.get("carved_out_of_own_track")
            if isinstance(co, dict):
                carved_keep_out_px += int(co.get("px", 0) or 0)
    depth_status = {
        "depth_used": bool(depth_meta or depth_npz_path.is_file()),
        "depth_meta": depth_meta,
        "depth_dates": depth_dates,
        "claimed_by_more_than_one_px": claimed_multi_px,
        "contested_px": contested_px,
        "carved_keep_out_px": carved_keep_out_px,
    }

    # If this is Pass 2+, check whether a prior pass's capture exists on disk and whether its region overlapped `subj`
    prior_pass_drift: list[dict[str, Any]] = []
    for p_laid in prior_laid_passes:
        p_load = p_laid.get("load")
        if not p_load:
            continue
        p_caps = sorted(
            d for d in data_root.glob(f"*_{p_load}")
            if d.is_dir() and not d.name.endswith("_first") and (d / "manifest.json").is_file()
        )
        if not p_caps:
            continue
        p_cap = p_caps[-1]
        p_man = json.loads((p_cap / "manifest.json").read_text())
        p_first = int(p_man.get("first_frame", -1))
        p_n = int(p_man.get("frames", -1))
        ov_start = max(first, p_first)
        ov_end = min(first + n, p_first + p_n)
        if ov_start >= ov_end:
            continue
        p_reg_p = p_cap / "runs" / p_load / "region.npz"
        p_subj_in_prior = next((s for s in p_man.get("subjects", []) if s["label"] == subj), None)
        if p_reg_p.is_file() and p_subj_in_prior and p_subj_in_prior.get("sightings"):
            p_reg_raw = (np.load(p_reg_p)["region"] > 0).astype(bool)
            if p_reg_raw.shape[1:] == (h // 16, w // 16):
                p_reg = np.repeat(np.repeat(p_reg_raw, 16, axis=1), 16, axis=2)[:, :h, :w]
            else:
                p_reg = p_reg_raw
            p_by = p_subj_in_prior["sightings"][0]["by"]
            p_tr_raw = _unpack_npz_mask(p_cap / "subjects" / subj / f"masks__{p_by}.npz", "track", w)
            if p_tr_raw is not None:
                overlap_frames = 0
                max_overlap_px = 0
                drifted_frames: list[dict[str, Any]] = []
                for sf in range(ov_start, ov_end):
                    i_cur = sf - first
                    i_pri = sf - p_first
                    ov_raw = int((p_reg[i_pri] & p_tr_raw[i_pri]).sum())
                    ov_cur = int((p_reg[i_pri] & track[i_cur]).sum())
                    tr_raw = int(p_tr_raw[i_pri].sum())
                    tr_cur = int(track[i_cur].sum())
                    if ov_raw > 0 or ov_cur > 0:
                        overlap_frames += 1
                        max_overlap_px = max(max_overlap_px, ov_raw, ov_cur)
                    # Flag significant SAM track collapse or drift where prior pass overlapped or track dropped >30%
                    if (tr_raw >= 500 and tr_cur < 0.5 * tr_raw) or (
                        (ov_raw > 0 or ov_cur > 0) and abs(tr_raw - tr_cur) > max(500, int(0.20 * max(tr_raw, tr_cur)))
                    ):
                        drifted_frames.append({
                            "source_frame": sf,
                            "prior_pass_region_overlap_raw_px": ov_raw,
                            "prior_pass_region_overlap_cur_px": ov_cur,
                            "raw_track_px": tr_raw,
                            "pass2_track_px": tr_cur,
                        })
                prior_pass_drift.append({
                    "prior_load": p_load,
                    "prior_capture": p_cap.name,
                    "prior_preview": p_by,
                    "current_preview": by,
                    "prior_region_overlap_frames": overlap_frames,
                    "prior_region_max_overlap_px": max_overlap_px,
                    "drifted_frames_count": len(drifted_frames),
                    "drifted_frames": drifted_frames,
                })

    pass_number = len({p["subject"] for p in prior_laid_passes}) + 1
    provenance_summary = {
        "pass_number": pass_number,
        "replaces_hair": replaces_hair,
        "part_classes": part_classes,
        "prior_laid_passes": prior_laid_passes,
        "later_laid_passes": later_laid_passes,
        "capture_clip": clip_name,
        "manifest_dates": manifest_dates,
        "clip_dates": clip_dates,
        "input_file_dates": input_file_dates,
        "preview_by": by,
        "preview_source": preview_source,
        "preview_queued_at": preview_queued_at,
        "strangers_source": strangers_source,
        "depth_status": depth_status,
        "provenance_mismatches": provenance_mismatches,
        "stale_inputs": stale_at_build,
        "stale_at_build": stale_at_build,
        "code_updated_since_capture": code_updated_since_capture,
        "prior_pass_drift": prior_pass_drift,
    }

    # --- 1. Latent Step Mapping & Held-Padding Bleed (Multi-Window Aware) ---
    plan_info = manifest.get("plan") if isinstance(manifest.get("plan"), dict) else {}
    plan_file: Path | None = Path(plan_info["file"]) if plan_info.get("file") else None
    reg_files: list[Path] = [Path(p) for p in (plan_info.get("region_files") or [])]

    def _resolve_pred_file(fname: str) -> Path | None:
        if not fname:
            return None
        p_direct = Path(fname)
        if p_direct.is_file():
            return p_direct
        for cand in (
            capture_dir / "predicted" / fname,
            capture_dir / "runs" / run_name / fname,
            vid_out_root / run_name / "predicted" / fname,
            vid_out_root / run_name / "_windows" / fname,
            vid_out_root / run_name / f"{run_name}_windows" / fname,
            vid_out_root / run_name / f"{run_name}_plan" / f"{run_name}_windows" / fname,
            vid_out_root / run_name / fname,
        ):
            if cand.is_file():
                return cand
        if vid_out_root.is_dir():
            for pat in (
                f"*/{run_name}/{run_name}_plan/{run_name}_windows/{fname}",
                f"*/{run_name}/{run_name}/{run_name}_windows/{fname}",
                f"*/{run_name}/predicted/{fname}",
                f"*/{run_name}/_windows/{fname}",
                f"*/predicted/{fname}",
                f"*/_windows/{fname}",
                f"*/{run_name}_windows/{fname}",
            ):
                g = list(vid_out_root.glob(pat))
                if g:
                    return g[0]
        return None

    if (plan_file is None or not plan_file.is_file()) and isinstance(run_entry.get("plan"), str):
        plan_file = _resolve_pred_file(run_entry["plan"])
    if not reg_files and run_entry.get("files"):
        for rf_str in run_entry["files"]:
            rp = _resolve_pred_file(str(rf_str))
            if rp is not None:
                reg_files.append(rp)

    latent_steps: list[dict[str, Any]] = []
    pad_bleed_steps: list[dict[str, Any]] = []
    step_canvas_masks: dict[tuple[int, int], np.ndarray] = {}

    plan_windows: list[dict[str, Any]] = []
    if plan_file and plan_file.is_file():
        try:
            plan_windows = json.loads(plan_file.read_text()).get("windows") or []
        except Exception:
            plan_windows = []

    for w_idx, reg_p in enumerate(reg_files):
        if not reg_p.is_file():
            continue
        pz = np.load(reg_p)
        w_tok = pz["tokens"] > 0
        w_mask = _unpack_npz_mask(reg_p, "mask", w)
        meta = json.loads(str(pz["meta"])) if "meta" in pz.files else {}
        w_row = plan_windows[w_idx] if w_idx < len(plan_windows) else {}

        w_first = int(meta.get("first_frame", w_row.get("first_frame", 0)))
        pad_b = int(meta.get("pad_before", w_row.get("pad_before", 0)))
        pad_a = int(meta.get("pad_after", w_row.get("pad_after", 0)))
        trim = int(w_row.get("trim", meta.get("trim", 0)))
        w_first_written = int(w_row.get("first_written_frame", w_first + trim))
        w_last_written = int(w_row.get("last_written_frame", n - 1))
        zoom_box = (meta.get("zoom") or w_row.get("zoom") or {}).get("box")

        proj_canvas = _project_tokens_to_canvas(w_tok, zoom_box, h, w)
        total_wf = len(w_mask) if w_mask is not None else int(meta.get("window_frames", w_row.get("window_frames", n)))
        r_lens = run_lengths(len(w_tok))

        cur = 0
        for t_idx, length in enumerate(r_lens):
            step_canvas_masks[(w_idx, t_idx)] = proj_canvas[t_idx]
            w_frames = list(range(cur, min(cur + length, total_wf)))
            cur += length
            tok_cells = int(w_tok[t_idx].sum()) if t_idx < len(w_tok) else 0
            canvas_cells = int(proj_canvas[t_idx].reshape(h // 16, 16, w // 16, 16).any(axis=(1, 3)).sum())
            ys, xs = np.where(w_tok[t_idx]) if t_idx < len(w_tok) else (np.array([]), np.array([]))
            tok_bbox = [int(xs.min() * 16), int(ys.min() * 16), int((xs.max() + 1) * 16), int((ys.max() + 1) * 16)] if len(xs) else None
            c_ys, c_xs = np.where(proj_canvas[t_idx])
            canvas_bbox = [int(c_xs.min()), int(c_ys.min()), int(c_xs.max() + 1), int(c_ys.max() + 1)] if len(c_xs) else None

            src_labels: list[str] = []
            mask_px_per_wf: list[int] = []
            real_wf_px: list[int] = []
            pad_wf_px: list[int] = []
            written_indices: list[int] = []
            for wf in w_frames:
                m_px = int(w_mask[wf].sum()) if (w_mask is not None and wf < len(w_mask)) else 0
                mask_px_per_wf.append(m_px)
                rel_idx = w_first + (wf - pad_b)
                is_pad_before = wf < pad_b
                is_pad_after = (pad_a > 0 and wf >= total_wf - pad_a) or (rel_idx >= n)
                is_trim = (not is_pad_before) and (not is_pad_after) and (wf < pad_b + trim or rel_idx < w_first_written or rel_idx > w_last_written)
                if is_pad_before:
                    src_labels.append(f"PAD_B{wf}")
                    pad_wf_px.append(m_px)
                elif is_pad_after:
                    src_labels.append(f"PAD_{wf}")
                    pad_wf_px.append(m_px)
                elif is_trim:
                    src_labels.append(f"TRIM_{first + rel_idx}")
                else:
                    src_labels.append(str(first + rel_idx))
                    real_wf_px.append(m_px)
                    written_indices.append(rel_idx)

            step_rec = {
                "window": w_idx,
                "step": t_idx,
                "window_frames": w_frames,
                "frames": src_labels,
                "written_indices": written_indices,
                "token_cells": tok_cells,
                "canvas_cells": canvas_cells,
                "token_bbox": tok_bbox,
                "canvas_bbox": canvas_bbox,
                "mask_px_per_wf": mask_px_per_wf,
            }
            latent_steps.append(step_rec)
            if real_wf_px and pad_wf_px and max(real_wf_px) == 0 and max(pad_wf_px) > 0 and tok_cells > 0:
                pad_bleed_steps.append(step_rec)

    # --- 2 & 3. Blue-Only Frames and Speck-in-Blue Frames (Keyed by (window, step)) ---
    carried_px = carried_mask.reshape(n, -1).sum(axis=-1)
    track_px = track.reshape(n, -1).sum(axis=-1)
    head_track_px = head_track.reshape(n, -1).sum(axis=-1) if head_track is not None else np.zeros(n, dtype=int)
    part_floor_px = int((seen.get("part_sizes") or {}).get("floor_px", 256))

    frame_to_step: dict[int, tuple[int, int]] = {}
    step_to_frames: dict[tuple[int, int], list[int]] = {}
    step_to_tok_cells: dict[tuple[int, int], int] = {}
    step_to_canvas_cells: dict[tuple[int, int], int] = {}
    for st in latent_steps:
        key = (int(st["window"]), int(st["step"]))
        step_to_tok_cells[key] = int(st["token_cells"])
        step_to_canvas_cells[key] = int(st["canvas_cells"])
        for f_idx in st["written_indices"]:
            if 0 <= f_idx < n:
                frame_to_step[f_idx] = key
                step_to_frames.setdefault(key, []).append(f_idx)

    blue_only_frames: list[dict[str, Any]] = []
    speck_in_region_frames: list[dict[str, Any]] = []
    for i in range(n):
        sf = first + i
        c_px = int(carried_px[i])
        st_key = frame_to_step.get(i, (0, -1))
        r_canvas_cells = int(region_cells[i]) or step_to_canvas_cells.get(st_key, 0)
        r_tok_cells = step_to_tok_cells.get(st_key, 0)
        peers = step_to_frames.get(st_key, [i])
        step_max_c = max(int(carried_px[p]) for p in peers) if peers else c_px
        peer_str = [f"{first + p}:{int(carried_px[p])}px" for p in peers]
        if c_px == 0 and (r_canvas_cells > 0 or r_tok_cells > 0):
            blue_only_frames.append({
                "source_frame": sf,
                "window": st_key[0],
                "latent_step": st_key[1],
                "carried_px": 0,
                "region_cells": r_canvas_cells,
                "zoomed_token_cells": r_tok_cells,
                "step_peers_carried_px": peer_str,
            })
        elif 0 < c_px < 0.25 * step_max_c and c_px < max(256, part_floor_px) and (r_canvas_cells > 0 or r_tok_cells > 0):
            speck_in_region_frames.append({
                "source_frame": sf,
                "window": st_key[0],
                "latent_step": st_key[1],
                "carried_px": c_px,
                "step_max_carried_px": step_max_c,
                "ratio_to_step_max": round(c_px / max(1, step_max_c), 4),
                "region_cells": r_canvas_cells,
                "zoomed_token_cells": r_tok_cells,
                "step_peers_carried_px": peer_str,
            })

    # --- 4, 5, & 7. Reference Mask Selection (Hair+Face vs Face-Only), Edge-Shrink, Self-Swallow, & Gap Recovery ---
    # Crucial distinction: `second` replaces `Face_Neck + Hair` (`replaces_hair=True`), so her target mask is `head_track | carried_mask`.
    # Face-only subjects (`lead`, `third`, `replaces_hair=False`) do NOT replace hair (`58%-76%` of `head_track` is unreplaced hair),
    # so their reference mask for [4], [5], and [7] must be `carried_mask` (face only), never full-cranium `head_track`!
    ref_head = (head_track | carried_mask) if (replaces_hair and head_track is not None) else carried_mask
    ref_px = ref_head.reshape(n, -1).sum(axis=-1)
    nonzero_ref = ref_px[ref_px > 0]
    med_px = float(np.median(nonzero_ref)) if len(nonzero_ref) else 0.0
    nonzero_carried = carried_px[carried_px > 0]
    med_carried_px = float(np.median(nonzero_carried)) if len(nonzero_carried) else 0.0
    nonzero_ht = head_track_px[head_track_px > 0]
    med_ht_px = float(np.median(nonzero_ht)) if len(nonzero_ht) else 0.0

    # Intentional empty/dropped frames (--drop or --no-face-on on a face-only subject)
    intentional_empty_sfs: set[int] = set()
    for k_emp in ("parts_emptied_on_source_frames", "parts_face_dropped_on_a_word_on_source_frames"):
        for f_emp in (seen.get(k_emp) or []):
            if isinstance(f_emp, int):
                intentional_empty_sfs.add(f_emp)

    other_union = np.zeros((n, h, w), dtype=bool)
    for lbl_o, tr_o in other_subject_tracks.items():
        other_union |= tr_o
    if str_carved is not None:
        other_union |= str_carved

    whole_head_frames: list[int] = []
    for i in range(n):
        # A frame can only be a whole reference head if its carried mask is also >= 50% of med_carried_px
        if ref_px[i] >= 0.5 * med_px and carried_px[i] >= 0.5 * med_carried_px and not (ref_head[i] & other_union[i]).any():
            whole_head_frames.append(i)

    edge_shrunk_whole_frames: list[dict[str, Any]] = []
    clean_whole: list[int] = []
    for i in whole_head_frames:
        prev_px = int(ref_px[i - 1]) if i > 0 else 0
        next_px = int(ref_px[i + 1]) if i + 1 < n else 0
        nbr_max = max(prev_px, next_px)
        if nbr_max > 0 and int(ref_px[i]) < 0.75 * nbr_max:
            edge_shrunk_whole_frames.append({
                "source_frame": first + i,
                "ref_head_px": int(ref_px[i]),
                "prev_px": prev_px,
                "next_px": next_px,
                "ratio_to_neighbor": round(int(ref_px[i]) / nbr_max, 3),
            })
        else:
            clean_whole.append(i)
    if not clean_whole:
        clean_whole = whole_head_frames

    # --- 5. Strangers Self-Swallow Audit ---
    strangers_self_swallow_frames: list[dict[str, Any]] = []
    if raw_strangers is not None and clean_whole:
        face_ids = (2, 3, 23, 24, 25, 26, 27)
        for i in range(n):
            if int(carried_px[i]) >= 0.5 * med_carried_px or (first + i) in intentional_empty_sfs:
                continue
            src = min(clean_whole, key=lambda j: (abs(j - i), j))
            h_src = ref_head[src]
            h_px = int(h_src.sum())
            if h_px == 0:
                continue
            raw_ov = int((h_src & raw_strangers[i]).sum())
            carved_ov = int((h_src & str_carved[i]).sum()) if str_carved is not None else 0
            if whole_classes is not None:
                other_p = raw_strangers[i] & ((whole_classes[i] == 8) | (np.isin(whole_classes[i], face_ids) & ~track[i]))
            else:
                other_p = str_carved[i] if str_carved is not None else raw_strangers[i]
            kept_after_fix = int((h_src & ~other_p).sum())
            if raw_ov >= 0.30 * h_px and int(carried_px[i]) < 0.25 * h_px:
                strangers_self_swallow_frames.append({
                    "source_frame": first + i,
                    "carried_px": int(carried_px[i]),
                    "track_px": int(track_px[i]),
                    "nearest_clean_whole": first + src,
                    "ref_head_px": h_px,
                    "swallowed_by_raw_strangers_px": raw_ov,
                    "swallowed_share": round(raw_ov / h_px, 3),
                    "in_carved_strangers_px": carved_ov,
                    "recoverable_with_class_filter_px": kept_after_fix,
                })

    # --- 7. Per-Frame Gap Recovery Table ---
    hih_map: dict[int, dict[str, Any]] = {}
    hih_info = seen.get("hair_in_hole")
    if isinstance(hih_info, dict) and hih_info.get("file"):
        hih_p = capture_dir / hih_info["file"]
        if hih_p.is_file():
            for r_row in json.loads(hih_p.read_text()).get("frames", []):
                hih_map[int(r_row["source_frame"])] = r_row

    gap_recovery: list[dict[str, Any]] = []
    clean_set = set(clean_whole)
    for i in range(n):
        sf = first + i
        if sf in intentional_empty_sfs:
            continue
        if i in clean_set and int(carried_px[i]) >= 0.5 * med_carried_px:
            continue
        if int(carried_px[i]) < 0.5 * med_carried_px or i in {
            r["source_frame"] - first for r in edge_shrunk_whole_frames
        }:
            src = min(clean_whole, key=lambda j: (abs(j - i), j)) if clean_whole else i
            h_src = ref_head[src]
            if whole_classes is not None and raw_strangers is not None:
                face_ids = (2, 3, 23, 24, 25, 26, 27)
                other_p = raw_strangers[i] & ((whole_classes[i] == 8) | (np.isin(whole_classes[i], face_ids) & ~track[i]))
            else:
                other_p = str_carved[i] if str_carved is not None else np.zeros((h, w), dtype=bool)
            for tr_o in other_subject_tracks.values():
                other_p = other_p | tr_o[i]
            rec_px = int((h_src & ~other_p).sum())
            hih_row = hih_map.get(sf, {})
            head_out_px = int((head_track[i] & ~region_mask[i]).sum()) if (replaces_hair and head_track is not None) else int((carried_mask[i] & ~region_mask[i]).sum())
            gap_recovery.append({
                "source_frame": sf,
                "track_px": int(track_px[i]),
                "head_track_px": int(head_track_px[i]),
                "carried_px": int(carried_px[i]),
                "region_cells": int(region_cells[i]),
                "head_outside_region_px": head_out_px,
                "hih_state": hih_row.get("state", "none"),
                "hih_added_px": int(hih_row.get("added_px", 0) or 0),
                "nearest_clean_whole": first + src,
                "recoverable_px": rec_px,
            })

    # --- 6. Missed SAM 3.1 Strangers Inside Blue Region ---
    missed_sam_strangers: list[dict[str, Any]] = []
    if people_npz_path is not None and people_npz_path.is_file():
        pz = np.load(people_npz_path)
        p_masks = np.unpackbits(pz["masks"], axis=-1)[..., :w].astype(bool)
        p_frames = pz["frame"]
        p_scores = pz["score"]
        p_nums = pz["person"]
        str_mask_all = raw_strangers if raw_strangers is not None else (str_carved if str_carved is not None else np.zeros((n, h, w), dtype=bool))
        p_offset = max(0, first - shots_first)
        for i in range(n):
            idx = np.where(p_frames == (i + p_offset))[0]
            for k in idx:
                sc = float(p_scores[k])
                if sc < 0.70:
                    continue
                m_det = p_masks[k]
                det_px = int(m_det.sum())
                if det_px == 0:
                    continue
                # Skip if this SAM 3.1 person detection is the target subject OR any other tracked subject
                if int((m_det & (track[i] | carried_mask[i])).sum()) >= 0.20 * det_px:
                    continue
                if any(int((m_det & tr_o[i]).sum()) >= 0.20 * det_px for tr_o in other_subject_tracks.values()):
                    continue
                in_reg_m = m_det & region_mask[i] & ~carried_mask[i]
                in_reg_px = int(in_reg_m.sum())
                in_reg_covered_by_str = int((in_reg_m & str_mask_all[i]).sum())
                unprotected_in_reg = in_reg_px - in_reg_covered_by_str
                if unprotected_in_reg >= 256 and in_reg_covered_by_str < 0.25 * max(1, in_reg_px):
                    ys, xs = np.where(m_det)
                    missed_sam_strangers.append({
                        "source_frame": first + i,
                        "person_id": int(p_nums[k]),
                        "sam_score": round(sc, 3),
                        "person_area_px": det_px,
                        "in_blue_region_px": unprotected_in_reg,
                        "in_strangers_mkv_px": in_reg_covered_by_str,
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    })

    return {
        "capture": str(capture_dir),
        "run": run_name,
        "subject": subj,
        "replaces_hair": replaces_hair,
        "part_classes": part_classes,
        "first_frame": first,
        "frames": n,
        "size": [w, h],
        "median_head_px": round(med_px, 1),
        "median_carried_px": round(med_carried_px, 1),
        "median_head_track_px": round(med_ht_px, 1),
        "whole_head_frames_count": len(whole_head_frames),
        "clean_whole_head_frames_count": len(clean_whole),
        "provenance": provenance_summary,
        "pad_bleed_steps": pad_bleed_steps,
        "blue_only_frames": blue_only_frames,
        "speck_in_region_frames": speck_in_region_frames,
        "edge_shrunk_whole_frames": edge_shrunk_whole_frames,
        "strangers_self_swallow_frames": strangers_self_swallow_frames,
        "missed_sam_strangers_in_region": missed_sam_strangers,
        "gap_recovery": gap_recovery,
        "latent_steps": latent_steps,
        "_step_canvas_masks": step_canvas_masks,
        "_frame_to_step": frame_to_step,
    }


def render_diagnostic_strips(
    capture_dir: Path,
    report: dict[str, Any],
    frames_to_draw: list[int],
    out_path: Path,
) -> None:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    first = int(manifest["first_frame"])
    n = int(manifest["frames"])
    w, h = int(manifest["size"][0]), int(manifest["size"][1])
    run_name = report["run"]
    subj = report["subject"]
    run_dir = capture_dir / "runs" / run_name
    subj_dir = capture_dir / "subjects" / subj
    by = next(s for s in manifest["subjects"] if s["label"] == subj)["sightings"][0]["by"]

    # Resolve source video
    clip_rel = manifest.get("clip") or ""
    in_root = Path(os.environ.get("H3_INPUT_DIR", "input"))
    vid_path = in_root / clip_rel
    if not vid_path.is_file():
        cands = list(in_root.rglob(Path(clip_rel).name))
        if not cands:
            raise ValueError(f"cannot find source clip {clip_rel} under {in_root}")
        vid_path = cands[0]

    track = _unpack_npz_mask(subj_dir / f"masks__{by}.npz", "track", w)
    head_track = _read_head_mask(subj_dir, by, n, h, w)
    carried = _read_mkv_bool(run_dir / "carried.mkv", n, h, w)
    if carried is None and (run_dir / "region.npz").is_file():
        carried = _unpack_npz_mask(run_dir / "region.npz", "carried", w)
    if carried is None:
        parts_held = _read_mkv_bool(subj_dir / f"parts_held__{by}.mkv", n, h, w)
        carried = parts_held if parts_held is not None else track
    reg_m: np.ndarray | None = None
    if (run_dir / "region.npz").is_file():
        r_raw = (np.load(run_dir / "region.npz")["region"] > 0).astype(bool)
        reg_m = np.repeat(np.repeat(r_raw, 16, axis=1), 16, axis=2)[:, :h, :w] if r_raw.shape[1:] == (h // 16, w // 16) else r_raw
    else:
        reg_m = _read_mkv_bool(run_dir / "region.mkv", n, h, w)
    step_canvas_masks: dict[tuple[int, int], np.ndarray] = report.get("_step_canvas_masks") or {}
    frame_to_step: dict[int, tuple[int, int]] = report.get("_frame_to_step") or {}

    # Compute crop box around the subject across the selected frames
    union = np.zeros((h, w), dtype=bool)
    for sf in frames_to_draw:
        idx = sf - first
        if 0 <= idx < n:
            if track is not None:
                union |= track[idx]
            if carried is not None:
                union |= carried[idx]
            if reg_m is not None:
                union |= reg_m[idx]
            st_m = step_canvas_masks.get(frame_to_step.get(idx, (0, -1)))
            if st_m is not None:
                union |= st_m
    ys, xs = np.where(union)
    if len(xs):
        x0, x1 = max(0, int(xs.min()) - 40), min(w, int(xs.max()) + 40)
        y0, y1 = max(0, int(ys.min()) - 40), min(h, int(ys.max()) + 40)
    else:
        x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4

    blue_set = {r["source_frame"] for r in report["blue_only_frames"]}
    speck_set = {r["source_frame"] for r in report["speck_in_region_frames"]}
    shrink_set = {r["source_frame"] for r in report["edge_shrunk_whole_frames"]}
    swallow_set = {r["source_frame"] for r in report["strangers_self_swallow_frames"]}
    pad_set = {
        int(lbl)
        for s in report["pad_bleed_steps"]
        for lbl in s["frames"]
        if lbl.isdigit()
    }

    cap = cv2.VideoCapture(str(vid_path))
    tiles: list[np.ndarray] = []
    for sf in frames_to_draw:
        idx = sf - first
        if not (0 <= idx < n):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        if frame_bgr.shape[:2] != (h, w):
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        canvas = frame_bgr.copy()
        st_key = frame_to_step.get(idx, (0, -1))
        st_m = step_canvas_masks.get(st_key)
        laid_reg = reg_m[idx] if reg_m is not None else np.zeros((h, w), dtype=bool)
        if laid_reg.any():
            canvas[laid_reg] = (canvas[laid_reg].astype(np.float32) * 0.55 + np.array([255, 110, 40], dtype=np.float32) * 0.45).astype(np.uint8)
        elif st_m is not None and st_m.any():
            # Unlaid blue-only or pad-bleed step: draw projected step region in purple-blue tint + contour
            canvas[st_m] = (canvas[st_m].astype(np.float32) * 0.55 + np.array([255, 60, 180], dtype=np.float32) * 0.45).astype(np.uint8)
            st_u8 = st_m.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(st_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, (255, 100, 220), 1)
        if carried is not None and carried[idx].any():
            m = carried[idx]
            canvas[m] = (canvas[m].astype(np.float32) * 0.45 + np.array([40, 40, 255], dtype=np.float32) * 0.55).astype(np.uint8)
        if track is not None and track[idx].any():
            tr_u8 = track[idx].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(tr_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, (0, 230, 0), 1)
        if head_track is not None and head_track[idx].any():
            ht_u8 = head_track[idx].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(ht_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, (255, 255, 0), 1)

        crop = canvas[y0:y1, x0:x1].copy()
        tags: list[str] = []
        if sf in blue_set:
            tags.append("BLUE-ONLY")
        if sf in speck_set:
            tags.append("SPECK")
        if sf in shrink_set:
            tags.append("SHRINK")
        if sf in swallow_set:
            tags.append("SWALLOW")
        if sf in pad_set:
            tags.append("PAD-BLEED")
        tag_str = ",".join(tags) if tags else "OK"
        c_px = int(carried[idx].sum()) if carried is not None else 0
        b_cells = int(laid_reg.sum()) // 256 if laid_reg.any() else (int(st_m.sum()) // 256 if st_m is not None else 0)
        hdr = f"{sf} r:{c_px} b:{b_cells}c {tag_str}"
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 22), (0, 0, 0), -1)
        col = (80, 255, 80) if tag_str == "OK" else (80, 180, 255)
        cv2.putText(crop, hdr, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        tiles.append(crop)
    cap.release()

    if not tiles:
        return
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    legend_h = 26
    grid = np.zeros((legend_h + rows * th, cols * tw, 3), dtype=np.uint8)
    legend_txt = (
        f"{run_name} ({subj}) | RED=carried mask  BLUE=laid step region  "
        f"PURPLE=unlaid step tokens  GREEN=person track  CYAN=head_track"
    )
    cv2.putText(grid, legend_txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[legend_h + r * th : legend_h + (r + 1) * th, c * tw : (c + 1) * tw] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def format_report(rep: dict[str, Any], problems_only: bool = False) -> str:
    lines: list[str] = []
    mode_lbl = "hair+face (ref=head_track|carried)" if rep.get("replaces_hair") else "face-only (ref=carried)"
    lines.append(
        f"=== MASK & LATENT-STEP FORENSIC REPORT: {rep['run']} (subject={rep['subject']}, {mode_lbl}, "
        f"frames={rep['first_frame']}..{rep['first_frame'] + rep['frames'] - 1}, N={rep['frames']}) ==="
    )
    lines.append(
        f"median_carried_px={rep.get('median_carried_px', rep['median_head_px'])} | "
        f"median_head_track_px={rep.get('median_head_track_px', 0.0)} | "
        f"whole_head_frames={rep['whole_head_frames_count']} | "
        f"clean_whole_head_frames={rep['clean_whole_head_frames_count']} "
        f"(shrunk_edges_dropped={len(rep['edge_shrunk_whole_frames'])})"
    )
    lines.append(
        "legend: RED (carried_px) = per-frame mask composited by lay_back | "
        "BLUE (region_cells / zoomed_token_cells) = 4-frame latent step regenerated by H3 DiT"
    )

    prov = rep.get("provenance") or {}
    if prov:
        p_num = prov.get("pass_number", 1)
        priors = [f"{p['load']}:{p['subject']}@{p['frames']}" for p in (prov.get("prior_laid_passes") or [])]
        laters = [f"{p['load']}:{p['subject']}@{p['frames']}" for p in (prov.get("later_laid_passes") or [])]
        lines.append(
            f"\n[0] INPUT & PASS-DEPENDENCY PROVENANCE: Pass #{p_num} "
            f"(prior_laid_before_capture={priors or 'none'}"
            + (f", laid_later={laters}" if laters else "")
            + ")"
        )
        m_dates = prov.get("manifest_dates") or {}
        c_dates = prov.get("clip_dates") or {}
        d_stat = prov.get("depth_status") or {}
        lines.append(
            f"  - manifest.json  : created={m_dates.get('created')} | modified={m_dates.get('modified')}\n"
            f"  - capture clip   : {prov.get('capture_clip')} (created={c_dates.get('created')}, modified={c_dates.get('modified')})\n"
            f"  - preview ({prov.get('preview_by')}) : source={prov.get('preview_source')} (queued={prov.get('preview_queued_at')})\n"
            f"  - strangers      : source={prov.get('strangers_source')}"
        )
        if d_stat:
            if d_stat.get("depth_used"):
                dm = d_stat.get("depth_meta") or {}
                lines.append(
                    f"  - moge2 depth    : USED (depth.npz settled {dm.get('settled_by_depth_px', 0)} / "
                    f"{d_stat.get('claimed_by_more_than_one_px', 0)} multi-claimed px; remaining contested={d_stat.get('contested_px', 0)} px)"
                )
            elif d_stat.get("claimed_by_more_than_one_px", 0) > 0 or d_stat.get("carved_keep_out_px", 0) > 0:
                lines.append(
                    f"  ! DEPTH NOT RUN (measure_moge2_depth.py / facebook_hyden-mogev2-metric-point): "
                    f"claimed_by_more_than_one={d_stat.get('claimed_by_more_than_one_px', 0)} px "
                    f"(contested_px={d_stat.get('contested_px', 0)} px unsettled) and "
                    f"carved_keep_out={d_stat.get('carved_keep_out_px', 0)} px carved without foreground depth check"
                )
        for mm in prov.get("provenance_mismatches") or []:
            lines.append(f"  ! MISMATCH: {mm}")
        for st in prov.get("stale_at_build") or []:
            lines.append(
                f"  ! STALE AT BUILD [{st['input']}]: {st['file']} (written={st['input_written']}, "
                f"disk_created={st.get('disk_created')}, disk_mod={st.get('disk_modified')}) "
                f"< code commit {st['latest_code_commit']} ({st['latest_code_date']}) in {st['code_files']}"
            )
        if not problems_only:
            for st in prov.get("code_updated_since_capture") or []:
                lines.append(
                    f"  * INFO (code updated after capture built) [{st['input']}]: {st['file']} (written={st['input_written']}) "
                    f"< HEAD commit {st['latest_code_commit']} ({st['latest_code_date']})"
                )
        for pd in prov.get("prior_pass_drift") or []:
            lines.append(
                f"  ! PRIOR PASS CONTAMINATION ({pd['prior_load']} -> {rep['run']}): "
                f"prior pass blue region overlapped {rep['subject']} on {pd['prior_region_overlap_frames']} frames "
                f"(up to {pd['prior_region_max_overlap_px']} px); "
                f"SAM track collapsed/drifted between {pd['prior_preview']} (raw) and {pd['current_preview']} (Pass-{p_num}) "
                f"on {pd['drifted_frames_count']} frames:"
            )
            for df in pd["drifted_frames"][:12]:
                lines.append(
                    f"      sf={df['source_frame']}: prior_reg_on_raw_tr={df['prior_pass_region_overlap_raw_px']} px "
                    f"(on_cur_tr={df.get('prior_pass_region_overlap_cur_px', 0)} px) | "
                    f"raw_track={df['raw_track_px']} px -> pass2_track={df['pass2_track_px']} px"
                )

    lines.append(f"\n[1] HELD-PADDING BLEED INTO KEPT SHOT STEPS: {len(rep['pad_bleed_steps'])}")
    for s in rep["pad_bleed_steps"]:
        lines.append(
            f"  - win {s['window']} step {s['step']:2d} frames={s['frames']} mask_px={s['mask_px_per_wf']} "
            f"canvas_cells={s['canvas_cells']} zoomed_tokens={s['token_cells']} canvas_bbox={s['canvas_bbox']}"
        )

    lines.append(f"\n[2] BLUE-ONLY FRAMES (0 px red mask inside active blue latent step): {len(rep['blue_only_frames'])}")
    for r in rep["blue_only_frames"]:
        lines.append(
            f"  - sf={r['source_frame']} (win {r['window']} step {r['latent_step']:2d}): carried=0 px, "
            f"canvas_cells={r['region_cells']} (zoomed_tokens={r['zoomed_token_cells']}), peers={r['step_peers_carried_px']}"
        )

    lines.append(f"\n[3] SPECK-IN-BLUE FRAMES (<25% of step max & below floor inside active blue step): {len(rep['speck_in_region_frames'])}")
    for r in rep["speck_in_region_frames"]:
        lines.append(
            f"  - sf={r['source_frame']} (win {r['window']} step {r['latent_step']:2d}): carried={r['carried_px']} px "
            f"({r['ratio_to_step_max']*100:.1f}% of step max {r['step_max_carried_px']} px), "
            f"canvas_cells={r['region_cells']} (zoomed_tokens={r['zoomed_token_cells']}), peers={r['step_peers_carried_px']}"
        )

    lines.append(f"\n[4] EDGE-SHRUNK WHOLE-HEAD FRAMES (<75% of immediate neighbor): {len(rep['edge_shrunk_whole_frames'])}")
    for r in rep["edge_shrunk_whole_frames"]:
        lines.append(
            f"  - sf={r['source_frame']}: ref_px={r['ref_head_px']} (prev={r['prev_px']}, next={r['next_px']}, "
            f"ratio={r['ratio_to_neighbor']})"
        )

    lines.append(f"\n[5] STRANGERS SELF-SWALLOW FRAMES (subject head swallowed by raw_strangers): {len(rep['strangers_self_swallow_frames'])}")
    for r in rep["strangers_self_swallow_frames"]:
        lines.append(
            f"  - sf={r['source_frame']}: carried={r['carried_px']} px, ref={r['nearest_clean_whole']} ({r['ref_head_px']} px), "
            f"swallowed_by_raw_strangers={r['swallowed_by_raw_strangers_px']} px ({r['swallowed_share']*100:.1f}%), "
            f"recoverable_with_class_filter={r['recoverable_with_class_filter_px']} px"
        )

    lines.append(f"\n[6] MISSED SAM 3.1 STRANGERS IN BLUE REGION (unprotected >=256 px): {len(rep['missed_sam_strangers_in_region'])}")
    for r in rep["missed_sam_strangers_in_region"]:
        lines.append(
            f"  - sf={r['source_frame']} person#{r['person_id']} (score={r['sam_score']}): "
            f"unprotected_in_blue_region={r['in_blue_region_px']} px, covered_by_strangers_mkv={r['in_strangers_mkv_px']} px, bbox={r['bbox']}"
        )

    if not problems_only or rep["gap_recovery"]:
        lines.append(f"\n[7] GAP & OCCLUSION RECOVERY TABLE ({len(rep['gap_recovery'])} frames below 50% median or edge-shrunk):")
        for r in rep["gap_recovery"]:
            lines.append(
                f"  - sf={r['source_frame']}: tr={r['track_px']:5d} ht={r['head_track_px']:5d} "
                f"carried={r['carried_px']:5d} reg_cells={r['region_cells']:3d} out_reg={r['head_outside_region_px']:4d} "
                f"hih={r['hih_state']} (+{r['hih_added_px']}) -> nearest_clean={r['nearest_clean_whole']} "
                f"recoverable={r['recoverable_px']:5d} px"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Forensic per-frame and per-latent-step mask & region inspector.")
    ap.add_argument("--capture", required=True, type=Path, help="capture folder data/<day>_<run>")
    ap.add_argument("--run", default=None, help="run name inside capture/runs/<run>")
    ap.add_argument("--frames", default=None, help="comma-separated frames or spans (e.g. 4920-4927,4939-4946)")
    ap.add_argument("--problems-only", action="store_true", help="only print problem summaries")
    ap.add_argument("--strips", type=Path, default=None, help="write annotated crop strip JPEG to this path")
    ap.add_argument("--json", type=Path, default=None, help="write full JSON report to this path")
    args = ap.parse_args(argv)

    try:
        rep = inspect_capture(args.capture, args.run)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    print(format_report(rep, problems_only=args.problems_only))
    if args.json:
        out_rep = {k: v for k, v in rep.items() if not k.startswith("_")}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out_rep, indent=2) + "\n")
    if args.strips:
        if args.frames:
            f_list = _parse_frames(args.frames, rep["first_frame"], rep["frames"])
        else:
            f_set: set[int] = set()
            for k in ("blue_only_frames", "speck_in_region_frames", "edge_shrunk_whole_frames",
                      "strangers_self_swallow_frames", "missed_sam_strangers_in_region", "gap_recovery"):
                for r in rep[k]:
                    f_set.add(int(r["source_frame"]))
            for s in rep["pad_bleed_steps"]:
                for lbl in s["frames"]:
                    if lbl.isdigit():
                        f_set.add(int(lbl))
            f_list = sorted(f_set) or list(range(rep["first_frame"], rep["first_frame"] + min(12, rep["frames"])))
        try:
            render_diagnostic_strips(args.capture, rep, f_list, args.strips)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
        print(f"Wrote diagnostic strip ({len(f_list)} frames) -> {args.strips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
