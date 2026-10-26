#!/usr/bin/env python3
"""Print every pre-render flag and carving/hold diagnostic for a capture, and optionally record a reader's initial.

Usage:
    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/flags_before_render.py <capture folder> [--load LOAD] [--initial WHO]

Prints four blocks of data from the capture folder (and its reference record):
1. `flags.json`: every flag the capture wrote (id, level, rule, subject/segment/other, source_frames, key figures,
   and whether it is blocking or overridden in `outcomes.json`).
2. Reference record (`marks_against_the_body_model.flagged_frames` and `face_on_the_track`): every frame flagged by
   the 2D-vs-3D body-model check (computing it live when an older reference record on disk lacks the key) and the
   off-track / neighbour-filled / bare-drawn face-point frame lists.
3. `runs/<run>/step_carves.json`: per-step keep-out cells carved from the grown region (strangers and other subjects,
   with cells touching the head), one line per carved step plus totals.
4. Held or emptied frames: frames held from a neighbour or by the floor rule, frames emptied by the grade or floor,
   doubted frames, and frames named empty by the job (`drop=`, `left_out`), one line each with frame spans.

With `--initial WHO`, records who read the printout and when (in UTC, with `status.json`'s `written` and `flags.json`'s
sha256) into `looks/<load>.shown.json` (under `"flags_read"`) and `looks/<load>.flags_read.json` beside it.
`bench/load_chain.py render` refuses when no initial later than the capture's `status.json` `written` is recorded.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class Refused(Exception):
    """A pre-condition that stops the tool with a message and no traceback."""


def now_iso() -> str:
    """This machine's clock in the same naive ISO format `capture_masked_run.py` writes to `status.json` and `looks.json`."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def as_time(text: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(str(text))
    except ValueError as e:
        raise Refused(f"{text!r} is not an ISO time") from e


def later(a: str, b: str) -> bool:
    """Is time `a` later than time `b`, as times. Both with an offset, or both without (two times one tool wrote on
    one machine's clock): comparable. One with and one without: refused, since they cannot be set in one zone."""
    ta, tb = as_time(a), as_time(b)
    if (ta.tzinfo is None) != (tb.tzinfo is None):
        raise Refused(f"the times {a!r} and {b!r} cannot be set in one zone: one says its offset and the other does not")
    return ta > tb


def in_utc(text: str) -> str:
    """A record's time in UTC. One written without an offset is this machine's local time, as the capture tool writes it."""
    t = as_time(text)
    return (t.astimezone(datetime.timezone.utc) if t.tzinfo else t.astimezone().astimezone(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_spans(spans_or_frames: list | dict | int) -> str:
    """Format a dict, list of `[first, last]` pairs, dicts, or integer frame numbers into compact `a-b, c` text."""
    if not spans_or_frames:
        return "none"
    if isinstance(spans_or_frames, int):
        return str(spans_or_frames)
    if isinstance(spans_or_frames, dict):
        if "source_frames" in spans_or_frames:
            return format_spans(spans_or_frames["source_frames"])
        if "frames" in spans_or_frames and isinstance(spans_or_frames["frames"], list):
            return format_spans(spans_or_frames["frames"])
        if "first" in spans_or_frames and "last" in spans_or_frames:
            return format_spans([[int(spans_or_frames["first"]), int(spans_or_frames["last"])]])
        return "none"
    if isinstance(spans_or_frames[0], dict):
        extracted = []
        for item in spans_or_frames:
            if "source_frame" in item:
                extracted.append(int(item["source_frame"]))
            elif "frame" in item:
                extracted.append(int(item["frame"]))
            elif "first" in item and "last" in item:
                extracted.append([int(item["first"]), int(item["last"])])
            elif "source_frames" in item and isinstance(item["source_frames"], list):
                extracted.extend(item["source_frames"])
        return format_spans(extracted)
    if isinstance(spans_or_frames[0], list | tuple):
        parts = []
        for item in spans_or_frames:
            if len(item) >= 2:
                a, b = int(item[0]), int(item[1])
                parts.append(str(a) if a == b else f"{a}-{b}")
            elif len(item) == 1:
                parts.append(str(int(item[0])))
        return ", ".join(parts) if parts else "none"
    nums = sorted({int(x) for x in spans_or_frames})
    runs = _runs_of_frames(nums)
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _runs_of_frames(nums: list[int] | set[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for n in sorted({int(x) for x in nums}):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return runs


def _extract_frame_set(spans_or_frames: list | dict | int) -> set[int]:
    out: set[int] = set()
    if not spans_or_frames:
        return out
    if isinstance(spans_or_frames, int):
        return {int(spans_or_frames)}
    if isinstance(spans_or_frames, list | tuple):
        for item in spans_or_frames:
            if isinstance(item, int):
                out.add(int(item))
            elif isinstance(item, list | tuple) and len(item) >= 2:
                out.update(range(int(item[0]), int(item[1]) + 1))
            elif isinstance(item, list | tuple) and len(item) == 1:
                out.add(int(item[0]))
            elif isinstance(item, dict):
                if "source_frame" in item:
                    out.add(int(item["source_frame"]))
                elif "frame" in item:
                    out.add(int(item["frame"]))
                elif "first" in item and "last" in item:
                    out.update(range(int(item["first"]), int(item["last"]) + 1))
    return out


def _manifest_meta(capture: Path) -> tuple[int, int, dict[str, str]]:
    """Return `(first_frame, canvas_width, {run_name: subject})` from `manifest.json` when present."""
    mfile = capture / "manifest.json"
    if not mfile.is_file():
        return 0, 0, {}
    try:
        m = json.loads(mfile.read_text())
    except Exception:
        return 0, 0, {}
    first = int(m.get("first_frame") or 0)
    size = m.get("size") or [0, 0]
    w = int(size[0]) if isinstance(size, list) and size else 0
    run_subj = {
        str(r["name"]): str(r["subject"])
        for r in (m.get("runs") or [])
        if isinstance(r, dict) and "name" in r and "subject" in r
    }
    return first, w, run_subj


def unlaid_while_tracked_of(capture: Path, loads: list[str] | None = None) -> dict[str, list[list[int]]]:
    """Return `{subject: [[first, last], ...]}` for frames inside the load where `track_px > 0` and `carried_px == 0`.

    When `runs/<load>/region.npz` is on disk, its `carried` and `read` arrays are authoritative for `carried_px == 0`
    (reflecting post-`her_hair_in_hole` and post-grade masks), and `frame_table*.jsonl` (written before `her_hair_in_hole`)
    is only consulted for subjects not covered by `runs/<load>/region.npz` or `runs/<load>/per_frame.json`.
    """
    loads_list = loads if loads is not None else loads_of_capture(capture)
    by_subj: dict[str, set[int]] = {}
    authoritative_subjs: set[str] = set()
    first, w, run_subj = _manifest_meta(capture)

    subj_tracked: dict[str, set[int]] = {}
    if (capture / "subjects").is_dir():
        for spf in sorted((capture / "subjects").glob("*/per_frame.json")):
            subj = spf.parent.name
            try:
                sdata = json.loads(spf.read_text())
            except Exception:
                continue
            tset: set[int] = set()
            for r in sdata.get("rows") or []:
                if not isinstance(r, dict) or "source_frame" not in r:
                    continue
                if float(r.get("track_share") or 0.0) > 0.0 or int(r.get("track_px") or 0) > 0:
                    tset.add(int(r["source_frame"]))
            subj_tracked[subj] = tset

    if (capture / "runs").is_dir():
        run_dirs = [capture / "runs" / ld for ld in loads_list if (capture / "runs" / ld).is_dir()]
        if not run_dirs:
            run_dirs = sorted(p for p in (capture / "runs").iterdir() if p.is_dir())
        for rdir in run_dirs:
            reg_path = rdir / "region.npz"
            rpf = rdir / "per_frame.json"
            if reg_path.is_file():
                try:
                    reg = np.load(reg_path)
                    carried_bits = np.unpackbits(reg["carried"], axis=-1)
                    if w > 0:
                        carried_bits = carried_bits[..., :w]
                    carried = carried_bits.astype(bool)
                    read_arr = reg["read"].astype(bool) if "read" in reg.files else np.ones(carried.shape[0], dtype=bool)
                    subj = run_subj.get(rdir.name, rdir.name)
                    authoritative_subjs.add(subj)
                    tset = subj_tracked.get(subj, set())
                    for idx in range(carried.shape[0]):
                        if not bool(read_arr[idx]):
                            continue
                        sf = first + idx
                        if int(carried[idx].sum()) == 0 and sf in tset:
                            by_subj.setdefault(subj, set()).add(sf)
                    continue
                except Exception:
                    pass
            if not rpf.is_file():
                continue
            try:
                rdata = json.loads(rpf.read_text())
            except Exception:
                continue
            for r in rdata.get("rows") or []:
                if not isinstance(r, dict) or "source_frame" not in r:
                    continue
                if int(r.get("read", 1)) != 1:
                    continue
                sf = int(r["source_frame"])
                subj = str(r.get("subject") or run_subj.get(rdir.name, rdir.name))
                authoritative_subjs.add(subj)
                carried_empty = float(r.get("carried_share") or 0.0) == 0.0 and int(r.get("carried_px") or 0) == 0
                is_tracked = (
                    float(r.get("track_share") or 0.0) > 0.0
                    or int(r.get("track_px") or 0) > 0
                    or sf in subj_tracked.get(subj, set())
                )
                if carried_empty and is_tracked:
                    by_subj.setdefault(subj, set()).add(sf)

    for ft_path in sorted(capture.glob("frame_table*.jsonl")):
        try:
            lines = ft_path.read_text().splitlines()
        except Exception:
            continue
        if not lines:
            continue
        try:
            hdr = json.loads(lines[0])
        except Exception:
            continue
        if loads is not None and hdr.get("run") and str(hdr["run"]) not in loads_list:
            continue
        subj = str(hdr.get("subject") or "subject")
        if subj in authoritative_subjs:
            continue
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "load_frame" in row and row["load_frame"] is None:
                continue
            if "source_frame" not in row:
                continue
            sf = int(row["source_frame"])
            t_px = int((row.get("tracked") or {}).get("px") or 0)
            c_px = int(row.get("loaded_part_px") or 0)
            if t_px > 0 and c_px == 0:
                by_subj.setdefault(subj, set()).add(sf)

    flags_file = capture / "flags.json"
    if flags_file.is_file():
        try:
            fdata = json.loads(flags_file.read_text())
            uwt = fdata.get("unlaid_while_tracked")
            if isinstance(uwt, dict):
                for subj, frames in uwt.items():
                    fset = _extract_frame_set(frames)
                    if fset:
                        by_subj.setdefault(str(subj), set()).update(fset)
        except Exception:
            pass

    return {subj: _runs_of_frames(fset) for subj, fset in sorted(by_subj.items()) if fset}


def unresolved_hair_in_hole_of(capture: Path, loads: list[str] | None = None) -> dict[str, dict]:
    """Return `{subject: {'lost_asked': [[a, b], ...], 'hair_undecided': [[a, b], ...]}}` for unresolved
    `hair_in_hole__*.json` frames inside the active load window(s).

    Overriding `her_head_lost_in_view` or `unclaimed_hair_touches_another` in `outcomes.json` adds 0 px to
    `parts_held.mkv`; these frames must be resolved in `plan.json` (`subjects.<subj>.head_marks`: `'hers'`,
    `'hidden'`, or `'not hers'`) or dropped with `--drop` and the capture rebuilt.
    """
    if not (capture / "subjects").is_dir():
        return {}
    loads_list = loads if loads is not None else loads_of_capture(capture)
    first, _w, run_subj = _manifest_meta(capture)
    target_subjs = {run_subj[ld] for ld in loads_list if ld in run_subj}

    read_frames_by_subj: dict[str, set[int]] = {}
    if (capture / "runs").is_dir():
        run_dirs = [capture / "runs" / ld for ld in loads_list if (capture / "runs" / ld).is_dir()]
        if not run_dirs:
            run_dirs = sorted(p for p in (capture / "runs").iterdir() if p.is_dir())
        for rdir in run_dirs:
            reg_path = rdir / "region.npz"
            if not reg_path.is_file():
                continue
            try:
                reg = np.load(reg_path)
                if "read" in reg.files:
                    read_arr = reg["read"].astype(bool)
                    subj = run_subj.get(rdir.name, rdir.name)
                    rset = {first + i for i in range(len(read_arr)) if bool(read_arr[i])}
                    read_frames_by_subj.setdefault(subj, set()).update(rset)
            except Exception:
                pass

    out: dict[str, dict] = {}
    for hh_path in sorted((capture / "subjects").glob("*/hair_in_hole__*.json")):
        subj = hh_path.parent.name
        if target_subjs and subj not in target_subjs:
            continue
        try:
            hh = json.loads(hh_path.read_text())
        except Exception:
            continue
        lost_set = _extract_frame_set(hh.get("lost_asked_on_source_frames") or [])
        undec_set = _extract_frame_set(hh.get("hair_undecided_on_source_frames") or [])
        if subj in read_frames_by_subj and read_frames_by_subj[subj]:
            lost_set &= read_frames_by_subj[subj]
            undec_set &= read_frames_by_subj[subj]
        if lost_set or undec_set:
            entry = out.setdefault(subj, {"lost_asked": set(), "hair_undecided": set()})
            entry["lost_asked"].update(lost_set)
            entry["hair_undecided"].update(undec_set)

    return {
        subj: {
            "lost_asked": _runs_of_frames(vals["lost_asked"]),
            "hair_undecided": _runs_of_frames(vals["hair_undecided"]),
        }
        for subj, vals in sorted(out.items())
        if vals["lost_asked"] or vals["hair_undecided"]
    }


def unmasked_frame_diagnostics_of(capture: Path, loads: list[str] | None = None) -> dict[str, dict]:
    """Join every unmasked (`carried_px == 0`) or unresolved (`hair_in_hole`) frame in the load against
    `region.npz`, `subjects/<subj>/per_frame.json`, `head_tube__*.json`, `part_grades.json`, and `hair_in_hole__*.json`.
    """
    loads_list = loads if loads is not None else loads_of_capture(capture)
    first, w, run_subj = _manifest_meta(capture)
    uwt = unlaid_while_tracked_of(capture, loads_list)
    uhh = unresolved_hair_in_hole_of(capture, loads_list)

    carried_by_subj: dict[str, dict[int, int]] = {}
    if (capture / "runs").is_dir():
        run_dirs = [capture / "runs" / ld for ld in loads_list if (capture / "runs" / ld).is_dir()]
        if not run_dirs:
            run_dirs = sorted(p for p in (capture / "runs").iterdir() if p.is_dir())
        for rdir in run_dirs:
            reg_path = rdir / "region.npz"
            if not reg_path.is_file():
                continue
            try:
                reg = np.load(reg_path)
                carried_bits = np.unpackbits(reg["carried"], axis=-1)
                if w > 0:
                    carried_bits = carried_bits[..., :w]
                carried = carried_bits.astype(bool)
                read_arr = reg["read"].astype(bool) if "read" in reg.files else np.ones(carried.shape[0], dtype=bool)
                subj = run_subj.get(rdir.name, rdir.name)
                cmap = carried_by_subj.setdefault(subj, {})
                for idx in range(carried.shape[0]):
                    if bool(read_arr[idx]):
                        sf = first + idx
                        cmap[sf] = max(cmap.get(sf, 0), int(carried[idx].sum()))
            except Exception:
                pass

    all_subjs = sorted(set(uwt.keys()) | set(uhh.keys()) | set(carried_by_subj.keys()))
    out: dict[str, dict] = {}
    for subj in all_subjs:
        cmap = carried_by_subj.get(subj, {})
        positive_cpx = [cpx for cpx in cmap.values() if cpx > 0]
        median_cpx = float(np.median(positive_cpx)) if positive_cpx else 0.0
        shrink_floor = 0.5 * median_cpx
        zero_or_shrunk_sfs = {
            sf for sf, cpx in cmap.items() if cpx == 0 or (median_cpx > 0 and cpx < shrink_floor)
        }
        uwt_sfs = _extract_frame_set(uwt.get(subj) or [])
        uhh_entry = uhh.get(subj) or {}
        uhh_sfs = _extract_frame_set(uhh_entry.get("lost_asked") or []) | _extract_frame_set(uhh_entry.get("hair_undecided") or [])
        target_sfs = sorted(zero_or_shrunk_sfs | uwt_sfs | uhh_sfs)
        if not target_sfs:
            continue

        sub_dir = capture / "subjects" / subj
        track_info: dict[int, dict] = {}
        if (sub_dir / "per_frame.json").is_file():
            try:
                pf = json.loads((sub_dir / "per_frame.json").read_text())
                for r in pf.get("rows") or []:
                    if isinstance(r, dict) and "source_frame" in r:
                        track_info[int(r["source_frame"])] = r
            except Exception:
                pass

        facing_set: set[int] = set()
        side_eye_set: set[int] = set()
        back_head_set: set[int] = set()
        grade_by_sf: dict[int, dict] = {}
        if (sub_dir / "part_grades.json").is_file():
            try:
                pg = json.loads((sub_dir / "part_grades.json").read_text())
                facing_set = {int(x) for x in (pg.get("facing_frames") or [])}
                side_eye_set = {int(x) for x in (pg.get("side_with_eye_frames") or [])}
                back_head_set = {int(x) for x in (pg.get("back_of_head_frames") or [])}
                for r in pg.get("frames") or []:
                    if isinstance(r, dict) and "source_frame" in r:
                        grade_by_sf[int(r["source_frame"])] = r
            except Exception:
                pass

        tube_by_sf: dict[int, int] = {}
        raw_part_by_sf: dict[int, int] = {}
        off_tube_by_sf: dict[int, int] = {}
        fp_by_sf: dict[int, int] = {}
        refused_by_sf: dict[int, str] = {}
        ht_files = sorted(sub_dir.glob("head_tube__*.json")) if sub_dir.is_dir() else []
        if ht_files:
            try:
                ht = json.loads(ht_files[0].read_text())
                tube_arr = ht.get("tube_px") or []
                raw_arr = ht.get("raw_part_px") or []
                off_arr = ht.get("off_tube_suppressed_px") or []
                fp_arr = ht.get("face_points_in_part") or []
                for idx, val in enumerate(tube_arr):
                    tube_by_sf[first + idx] = int(val or 0)
                for idx, val in enumerate(raw_arr):
                    raw_part_by_sf[first + idx] = int(val or 0)
                for idx, val in enumerate(off_arr):
                    if int(val or 0) > 0:
                        off_tube_by_sf[first + idx] = int(val)
                for idx, val in enumerate(fp_arr):
                    if int(val or 0) > 0:
                        fp_by_sf[first + idx] = int(val)
                for r in ht.get("regain_refused") or []:
                    sf = first + int(r.get("frame", -1))
                    refused_by_sf[sf] = str(r.get("why") or "regain_refused")
                    if int(r.get("face_points_on_own_part") or 0) > 0:
                        fp_by_sf[sf] = max(fp_by_sf.get(sf, 0), int(r["face_points_on_own_part"]))
            except Exception:
                pass

        hh_by_sf: dict[int, dict] = {}
        hh_files = sorted(sub_dir.glob("hair_in_hole__*.json")) if sub_dir.is_dir() else []
        if hh_files:
            try:
                hh = json.loads(hh_files[0].read_text())
                for r in hh.get("frames") or []:
                    if isinstance(r, dict) and "source_frame" in r:
                        hh_by_sf[int(r["source_frame"])] = r
            except Exception:
                pass

        rows = []
        has_visible_signal = False
        for sf in target_sfs:
            cpx = cmap.get(sf, 0)
            trow = track_info.get(sf) or {}
            t_share = float(trow.get("track_share") or 0.0)
            t_px = int(trow.get("track_px") or 0)
            if cpx > 0 and t_share == 0.0 and t_px == 0 and sf not in uhh_sfs:
                continue
            pose = (
                "facing" if sf in facing_set
                else "side_with_eye" if sf in side_eye_set
                else "back_of_head" if sf in back_head_set
                else "-"
            )
            tube_px = tube_by_sf.get(sf, 0)
            raw_px = raw_part_by_sf.get(sf, 0)
            off_px = off_tube_by_sf.get(sf, 0)
            fp_pts = fp_by_sf.get(sf, 0)
            refused = refused_by_sf.get(sf)
            grow = grade_by_sf.get(sf) or {}
            gdid = str(grow.get("did") or ("regain_refused" if refused else "-"))
            hhrow = hh_by_sf.get(sf) or {}
            lost_val = hhrow.get("lost")
            lost = bool(lost_val)
            lost_state = str(lost_val) if lost_val else "none"
            lost_held_px = int(hhrow.get("lost_held_px") or 0)
            undec_px = int(hhrow.get("hair_undecided_px") or 0)
            uncl_px = int(hhrow.get("hair_unclaimed_px") or 0)
            if cpx == 0 and (pose in ("facing", "side_with_eye") or off_px > 0 or fp_pts > 0 or lost or undec_px > 0):
                has_visible_signal = True
            elif lost_state == "asked" or undec_px > 0:
                has_visible_signal = True
            rows.append({
                "source_frame": sf,
                "carried_px": cpx,
                "median_carried_px": round(median_cpx, 1),
                "tube_px": tube_px,
                "raw_part_px": raw_px,
                "track_px": t_px,
                "track_share": t_share,
                "off_tube_suppressed_px": off_px,
                "pose": pose,
                "face_points": fp_pts,
                "grade_did": gdid,
                "lost": lost,
                "lost_state": lost_state,
                "lost_held_px": lost_held_px,
                "hair_unclaimed_px": uncl_px,
                "hair_undecided_px": undec_px,
            })
        out[subj] = {
            "rows": rows,
            "median_carried_px": round(median_cpx, 1),
            "has_visible_signal": has_visible_signal,
            "unresolved_hair_in_hole": uhh.get(subj) or {},
        }
    return out


def peer_look_covering(capture: Path, loads: list[str], status: dict) -> dict | None:
    """Return a valid peer look from `capture/looks.json` covering `loads` and later than `status['written']`, or None."""
    looks_file = capture / "looks.json"
    if not looks_file.is_file():
        return None
    try:
        looks = json.loads(looks_file.read_text()).get("looks") or []
    except Exception:
        return None
    written = status.get("written")
    made_by = status.get("made_by")
    for lk in reversed(looks):
        if not isinstance(lk, dict):
            continue
        lk_loads = lk.get("loads") or []
        if not all(ld in lk_loads for ld in loads):
            continue
        if not lk.get("by") or lk.get("by") == made_by:
            continue
        if written and lk.get("at"):
            try:
                if not later(lk["at"], written):
                    continue
            except Refused:
                continue
        return lk
    return None


def flag_key(flag: dict) -> str:
    about = [flag.get(k) for k in ("rule", "subject", "run", "other", "seen_by", "source_frames")]
    return hashlib.sha256(json.dumps(about, sort_keys=True).encode()).hexdigest()[:16]


def loads_of_capture(capture: Path, load_arg: str | None = None) -> list[str]:
    if load_arg:
        return [x.strip() for x in load_arg.split("+") if x.strip()]
    shown = sorted(p.name[: -len(".shown.json")] for p in (capture / "looks").glob("*.shown.json")) if (capture / "looks").is_dir() else []
    if shown:
        return shown
    runs = sorted(p.name for p in (capture / "runs").iterdir() if p.is_dir()) if (capture / "runs").is_dir() else []
    if runs:
        return runs
    name = capture.name
    if len(name) > 11 and name[4] == "-" and name[7] == "-" and name[10] == "_":
        return [name[11:]]
    return [name]


def _find_job_docs(capture: Path, job_file: Path | None = None) -> list[tuple[Path, dict]]:
    job_candidates: list[Path] = []
    if job_file is not None:
        job_candidates.append(job_file if job_file.is_absolute() else REPO / job_file)
    job_candidates.extend([
        capture.with_name(capture.name + "_job") / "job.plan.json",
        capture.with_name(capture.name + "_job") / "job.render.json",
        capture.with_name(capture.name + "_job") / "job.json",
        capture / "job.json",
    ])
    ft_file = capture / "frame_table.jsonl"
    if ft_file.is_file():
        try:
            first_line = ft_file.open().readline()
            hdr = json.loads(first_line)
            jf = hdr.get("job_file") or ((hdr.get("input_sha256") or {}).get("job") or {}).get("file")
            if jf:
                job_candidates.append(Path(jf) if Path(jf).is_absolute() else REPO / jf)
        except Exception:
            pass
    out: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    for jf in job_candidates:
        if jf.is_file() and jf.resolve() not in seen:
            try:
                jd = json.loads(jf.read_text())
                if isinstance(jd, dict):
                    seen.add(jf.resolve())
                    out.append((jf, jd))
            except Exception:
                pass
    return out


def recipe_check_of(
    capture: Path,
    loads: list[str] | None = None,
    flags_data: dict | None = None,
    job_file: Path | None = None,
) -> dict:
    """Return `{"rows": [...], "mouthless_references": [...]}` for `[0] recipe check` and `[2] reference`."""
    loads_list = loads if loads is not None else loads_of_capture(capture)
    if flags_data is None:
        flags_file = capture / "flags.json"
        flags_data = json.loads(flags_file.read_text()) if flags_file.is_file() else {}
    job_docs = _find_job_docs(capture, job_file=job_file)
    rows: list[dict] = []
    expected_closed_mouth: str | None = None
    for _jf, jd in job_docs:
        job = jd.get("job") or {}
        defaults = jd.get("defaults") or {}
        rec = job.get("recipe")
        r_diff = job.get("recipe_diff")
        ref_cfg = job.get("reference") or {}
        ref_mouth = ref_cfg.get("mouth") or {}
        if isinstance(rec, dict) and rec.get("closed_mouth"):
            expected_closed_mouth = str(rec.get("closed_mouth"))
        elif isinstance(ref_mouth.get("closed_mouth"), dict) and ref_mouth["closed_mouth"].get("form"):
            expected_closed_mouth = str(ref_mouth["closed_mouth"]["form"])
        if not isinstance(rec, dict) and not isinstance(r_diff, list):
            continue
        actual_margin = defaults.get("margin")
        margin_by = defaults.get("margin_by")
        actual_zoom = defaults.get("zoom_up_to", 2.0)
        zoom_by = defaults.get("zoom_up_to_by")
        if ref_cfg.get("form") == "none":
            actual_crop = "none"
            actual_open_mouth = "none"
            actual_closed_mouth = "none"
        else:
            actual_crop = str((ref_cfg.get("crop") or {}).get("form") or "head")
            cl_on = bool((ref_mouth.get("closed_lip_line") or {}).get("on", True))
            if ref_mouth.get("form") == "left out":
                actual_open_mouth = "none"
                actual_closed_mouth = "none"
            elif not cl_on:
                actual_open_mouth = "off"
                actual_closed_mouth = "off"
            else:
                actual_open_mouth = str((ref_mouth.get("open_mouth") or {}).get("form") or "inner outline")
                actual_closed_mouth = str((ref_mouth.get("closed_mouth") or {}).get("form") or "sculpted")
        actual_map = {
            "margin": actual_margin,
            "zoom_up_to": float(actual_zoom) if actual_zoom is not None else 2.0,
            "crop": actual_crop,
            "open_mouth": actual_open_mouth,
            "closed_mouth": actual_closed_mouth,
        }
        has_ov_map = {
            "margin": bool(isinstance(margin_by, dict) and str(margin_by.get("why") or "").strip()),
            "zoom_up_to": bool(
                (isinstance(zoom_by, dict) and str(zoom_by.get("why") or "").strip())
                or (isinstance(job.get("whole_frame"), dict) and str(job["whole_frame"].get("why") or "").strip())
            ),
            "crop": bool(ref_cfg.get("form") == "none"),
            "open_mouth": bool(
                ref_cfg.get("form") == "none"
                or ref_mouth.get("form") == "left out"
                or not bool((ref_mouth.get("closed_lip_line") or {}).get("on", True))
            ),
            "closed_mouth": bool(
                ref_cfg.get("form") == "none"
                or ref_mouth.get("form") == "left out"
                or not bool((ref_mouth.get("closed_lip_line") or {}).get("on", True))
            ),
        }
        if isinstance(r_diff, list) and r_diff:
            for item in r_diff:
                if not isinstance(item, dict) or "knob" not in item:
                    continue
                k = str(item["knob"])
                rec_v = item.get("recipe")
                cur_v = actual_map.get(k, item.get("this_load"))
                if cur_v is None:
                    cur_v = item.get("this_load")
                why_s = str(item.get("why") or "").strip()
                if cur_v == rec_v:
                    st = "match"
                elif cur_v == item.get("this_load") and item.get("status") in ("overridden", "unexplained"):
                    st = str(item["status"])
                elif has_ov_map.get(k, False) and why_s:
                    st = "overridden"
                else:
                    st = "unexplained"
                rows.append({"knob": k, "recipe": rec_v, "this_load": cur_v, "status": st, "why": why_s})
            break
        if isinstance(rec, dict):
            rec_why = str(rec.get("why") or rec.get("by") or "subject recipe in plan.json").strip()
            for k in ("margin", "zoom_up_to", "crop", "open_mouth", "closed_mouth"):
                if k not in rec:
                    continue
                rec_v = rec[k]
                cur_v = actual_map.get(k)
                if k == "margin" and isinstance(margin_by, dict):
                    why_s = str(margin_by.get("why") or rec_why).strip()
                elif k == "zoom_up_to" and isinstance(zoom_by, dict):
                    why_s = str(zoom_by.get("why") or rec_why).strip()
                else:
                    why_s = rec_why if cur_v == rec_v else ""
                st = "match" if cur_v == rec_v else ("overridden" if (has_ov_map.get(k, False) and why_s) else "unexplained")
                rows.append({"knob": k, "recipe": rec_v, "this_load": cur_v, "status": st, "why": why_s})
            break

    mouthless: list[str] = []
    ref_records = find_reference_records(capture, loads_list, flags_data, job_file=job_file)
    for _label, ref_path in ref_records:
        if not ref_path.is_file():
            continue
        try:
            rec_data = json.loads(ref_path.read_text())
        except Exception:
            continue
        cll = ((rec_data.get("drawn") or {}).get("mouth") or {}).get("closed_lip_line") or {}
        cm_dict = (
            cll.get("closed_mouth")
            if isinstance(cll.get("closed_mouth"), dict)
            else (rec_data.get("closed_mouth") if isinstance(rec_data.get("closed_mouth"), dict) else {})
        )
        cm_form = (
            cm_dict.get("form")
            or (rec_data.get("closed_mouth") if isinstance(rec_data.get("closed_mouth"), str) else None)
            or expected_closed_mouth
        )
        has_open_key = "open_frames" in cll or "open_frames" in rec_data
        has_closed_key = (
            "frames_drawn_closed" in cm_dict
            or "closed_drawn_frames" in cm_dict
            or "closed_drawn_frames" in cll
            or "closed_drawn_frames" in rec_data
        )
        open_f = cll.get("open_frames") if "open_frames" in cll else rec_data.get("open_frames")
        closed_f = (
            cm_dict.get("frames_drawn_closed")
            if "frames_drawn_closed" in cm_dict
            else cm_dict.get("closed_drawn_frames")
            if "closed_drawn_frames" in cm_dict
            else cll.get("closed_drawn_frames")
            if "closed_drawn_frames" in cll
            else rec_data.get("closed_drawn_frames")
        )
        if (
            cm_form == "outlined"
            and (has_open_key or has_closed_key)
            and len(_extract_frame_set(open_f or [])) == 0
            and len(_extract_frame_set(closed_f or [])) == 0
        ):
            mouthless.append(ref_path.name)

    return {"rows": rows, "mouthless_references": mouthless}


def find_reference_records(
    capture: Path,
    loads: list[str],
    flags_data: dict,
    job_file: Path | None = None,
) -> list[tuple[str, Path]]:
    """Return `(label, record_path)` pairs for every motion reference record associated with `capture`."""
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def _add(label: str, path_str: str | Path | None) -> None:
        if not path_str:
            return
        p = Path(path_str)
        if p.suffix.lower() in (".mp4", ".mkv", ".mov"):
            p = p.with_suffix(".json")
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if p not in seen:
            seen.add(p)
            found.append((label, p))

    for ld in loads:
        shown_file = capture / "looks" / f"{ld}.shown.json"
        if shown_file.is_file():
            try:
                shown = json.loads(shown_file.read_text())
                for m in shown.get("motion") or []:
                    if isinstance(m, dict) and m.get("file"):
                        _add(ld, m["file"])
            except Exception:
                pass

    told = flags_data.get("told_by_the_job") or {}
    for ref in told.get("references") or []:
        if isinstance(ref, dict) and ref.get("video"):
            _add(str(ref.get("subject") or "reference"), ref["video"])

    job_candidates: list[Path] = []
    if job_file is not None:
        job_candidates.append(job_file if job_file.is_absolute() else REPO / job_file)
    job_candidates.extend([
        capture.with_name(capture.name + "_job") / "job.plan.json",
        capture.with_name(capture.name + "_job") / "job.render.json",
    ])
    ft_file = capture / "frame_table.jsonl"
    if ft_file.is_file():
        try:
            first_line = ft_file.open().readline()
            hdr = json.loads(first_line)
            jf = hdr.get("job_file") or ((hdr.get("input_sha256") or {}).get("job") or {}).get("file")
            if jf:
                job_candidates.append(Path(jf) if Path(jf).is_absolute() else REPO / jf)
        except Exception:
            pass

    for jf in job_candidates:
        if jf.is_file():
            try:
                jd = json.loads(jf.read_text())
                job = jd.get("job") or {}
                for k in ("reference", "reference_2"):
                    rec = (job.get(k) or {}).get("record")
                    if rec:
                        _add(k, rec)
                for ld_obj in jd.get("loads") or []:
                    if not loads or ld_obj.get("name") in loads:
                        for mk in ("motion_file", "motion_file_2"):
                            if ld_obj.get(mk):
                                _add(str(ld_obj.get("name") or mk), ld_obj[mk])
            except Exception:
                pass

    return found


def _compute_live_marks(rec: dict, capture: Path) -> dict | None:
    """When an older reference `.json` record lacks `marks_against_the_body_model`, compute it live if its inputs exist."""
    inputs = rec.get("inputs") or {}
    mesh = inputs.get("body_model_video") or ((inputs.get("body_model") or {}).get("file") if isinstance(inputs.get("body_model"), dict) else None)
    points = inputs.get("points") if isinstance(inputs.get("points"), str) else ((inputs.get("points") or {}).get("file") if isinstance(inputs.get("points"), dict) else None)
    frames = rec.get("frames")
    if not (mesh and points and isinstance(frames, list) and len(frames) == 2):
        return None
    if not Path(points).is_file():
        return None
    try:
        import head_reference as hr
        first, last = int(frames[0]), int(frames[1])
        faceless = sorted({int(x) for k in ("no_face_frames_abs", "no_draw_face_frames_abs") for x in (rec.get(k) or [])})
        res = hr.marks_against_the_body_model(mesh, points, first, last - first + 1, faceless)
        if not res.get("known") and (capture / "subjects").is_dir():
            for pt_file in sorted((capture / "subjects").glob("*/pose_table__*.json")):
                fake_mesh = pt_file.parent / pt_file.name.replace("_pose_table.json", "_mesh_00001.mp4")
                res2 = hr.marks_against_the_body_model(fake_mesh, points, first, last - first + 1, faceless)
                if res2.get("known"):
                    return res2
        return res
    except Exception:
        return None


def print_capture_report(capture: Path, load_arg: str | None = None, out=sys.stdout) -> dict:
    """Read and print the four pre-render data sections for `capture`, returning a summary dict."""
    if not capture.is_dir():
        raise Refused(f"{capture} is not a folder")
    for req in ("status.json", "manifest.json", "flags.json"):
        if not (capture / req).is_file():
            raise Refused(f"{capture} holds no {req}")

    status = json.loads((capture / "status.json").read_text())
    manifest = json.loads((capture / "manifest.json").read_text())
    flags_data = json.loads((capture / "flags.json").read_text())
    outcomes_file = capture / "outcomes.json"
    outcomes = json.loads(outcomes_file.read_text()).get("outcomes", []) if outcomes_file.is_file() else []

    loads = loads_of_capture(capture, load_arg)
    span = flags_data.get("span") or [manifest.get("first_frame", 0), manifest.get("first_frame", 0) + manifest.get("frames", 0) - 1]
    verdict = flags_data.get("verdict", "unknown")
    blocking_ids = set(flags_data.get("blocking") or [])
    overridden_ids = set(flags_data.get("overridden") or [])
    override_by_key = {o.get("key"): o for o in outcomes if o.get("overridden") and o.get("key")}
    override_by_id = {o.get("flag"): o for o in outcomes if o.get("overridden") and o.get("flag")}

    flags_list = flags_data.get("flags") or []
    recipe_info = recipe_check_of(capture, loads, flags_data)
    recipe_rows = recipe_info["rows"]
    mouthless_refs = set(recipe_info["mouthless_references"])
    print(
        f"=== capture: {capture.name} | load(s): {', '.join(loads)} | frames: {span[0]}-{span[1]} "
        f"| written: {status.get('written')} by {status.get('made_by')} | gate: {verdict} ({len(flags_list)} flag(s)) ===",
        file=out,
    )

    if recipe_rows:
        unexp_n = sum(1 for r in recipe_rows if r.get("status") == "unexplained")
        print(
            f"\n[0] recipe check (knob | recipe | this load | status | why; unexplained={unexp_n}):",
            file=out,
        )
        for r in recipe_rows:
            print(
                f"  {str(r.get('knob', '')):<12} | {str(r.get('recipe', '')):<8} | {str(r.get('this_load', '')):<9} "
                f"| {str(r.get('status', '')):<11} | {r.get('why') or '-'}",
                file=out,
            )

    # 1. Every flag in flags.json
    print(f"\n[1] gate flags ({capture.name}/flags.json: {len(flags_list)} flag(s), blocking={sorted(blocking_ids)}, overridden={sorted(overridden_ids)}):", file=out)
    if not flags_list:
        print("  flags: none (0 flags)", file=out)
    for fl in flags_list:
        fid = fl.get("id", "?")
        lvl = fl.get("level", "?")
        rule = fl.get("rule", "?")
        who = fl.get("segment") or fl.get("subject") or fl.get("run") or "-"
        if fl.get("other"):
            who = f"{who} vs {fl['other']}"
        fspans = format_spans(fl.get("source_frames") or [])
        figs = fl.get("figures") or {}
        fig_str = ", ".join(f"{k}={v}" for k, v in figs.items() if k not in ("source_frames",))
        fk = flag_key(fl)
        ov = override_by_key.get(fk) or (override_by_id.get(fid) if fid in overridden_ids else None)
        if fid in blocking_ids:
            state = "BLOCKING"
        elif fid in overridden_ids or ov is not None:
            by_who = ov.get("by", "?") if ov else "?"
            state = f"OVERRIDDEN by {by_who}"
        else:
            state = "open"
        print(
            f"  flag {fid} [{lvl}] ({state}) {rule} ({who}): frames {fspans}"
            + (f" | {fig_str}" if fig_str else ""),
            file=out,
        )

    # 2. Reference record's marks_against_the_body_model.flagged_frames and face_on_the_track
    ref_records = find_reference_records(capture, loads, flags_data)
    print(f"\n[2] reference marks_against_the_body_model & face_on_the_track ({len(ref_records)} reference record(s)):", file=out)
    if not ref_records:
        print("  reference: none found for this capture", file=out)
    for label, ref_path in ref_records:
        if not ref_path.is_file():
            print(f"  reference ({label}): {ref_path} (file not on disk)", file=out)
            continue
        rec = json.loads(ref_path.read_text())
        mabm = rec.get("marks_against_the_body_model")
        source_note = ""
        if mabm is None:
            mabm = _compute_live_marks(rec, capture)
            source_note = " (absent in saved record; computed live)" if mabm is not None else " (absent in saved record)"
        if not mabm:
            print(f"  marks_against_the_body_model ({ref_path.name}){source_note}: not available", file=out)
        elif not mabm.get("known"):
            print(f"  marks_against_the_body_model ({ref_path.name}){source_note}: not known ({mabm.get('why')})", file=out)
        else:
            flagged = mabm.get("flagged_frames") or []
            f_nums = [int(r["frame"]) for r in flagged if isinstance(r, dict) and "frame" in r]
            spans_txt = format_spans(f_nums)
            print(
                f"  marks_against_the_body_model ({ref_path.name}, read_from={mabm.get('read_from')}){source_note}: "
                f"{len(flagged)} flagged frame(s) [{spans_txt}]",
                file=out,
            )
            for row in flagged:
                fr = row.get("frame")
                details = ", ".join(f"{k}={v}" for k, v in row.items() if k != "frame")
                print(f"    flagged_frame {fr}: {details}", file=out)

        fot = rec.get("face_on_the_track")
        if not fot or not fot.get("known"):
            print(f"  face_on_the_track ({ref_path.name}): not known", file=out)
        else:
            off_mid = fot.get("frames_whose_face_middle_is_off_the_track") or []
            filled_map = fot.get("frames_filled_from_their_neighbours") or {}
            filled_frames = sorted(int(k) for k in filled_map.keys()) if isinstance(filled_map, dict) else list(filled_map)
            not_filled = fot.get("frames_off_the_track_not_filled") or []
            no_face_drawn = fot.get("frames_off_the_track_with_no_face_drawn") or []
            print(
                f"  face_on_the_track ({ref_path.name}): "
                f"middle_off_track={len(off_mid)} [{format_spans(off_mid)}] | "
                f"filled_from_neighbours={len(filled_frames)} [{format_spans(filled_frames)}] | "
                f"off_track_not_filled={len(not_filled)} [{format_spans(not_filled)}] | "
                f"off_track_no_face_drawn={len(no_face_drawn)} [{format_spans(no_face_drawn)}]",
                file=out,
            )
        if ref_path.name in mouthless_refs:
            print(
                f"  mouthless_reference ({ref_path.name}): 0 open-mouth frame(s) and 0 closed-mouth drawn frame(s) "
                "while closed_mouth='outlined' was requested",
                file=out,
            )

    # 3. Strangers and other keep-out cells pinned per step (runs/<run>/step_carves.json)
    carve_files = sorted((capture / "runs").glob("*/step_carves.json")) if (capture / "runs").is_dir() else []
    print(f"\n[3] step_carves ({len(carve_files)} run file(s)):", file=out)
    if not carve_files:
        print("  step_carves: none (no runs/<run>/step_carves.json)", file=out)
    for cf in carve_files:
        sc = json.loads(cf.read_text())
        run_name = sc.get("run") or cf.parent.name
        steps = sc.get("steps") or []
        others = sc.get("others") or []
        carved_steps = [st for st in steps if (st.get("carved_units_total") or 0) > 0 or any((v or 0) > 0 for v in (st.get("carved_by_other") or {}).values())]
        strangers_steps = [st for st in steps if ((st.get("carved_by_other") or {}).get("strangers") or 0) > 0]
        total_carved = sum(int(st.get("carved_units_total") or 0) for st in steps)
        total_strangers = sum(int((st.get("carved_by_other") or {}).get("strangers") or 0) for st in steps)
        total_touching_head = sum(int(st.get("carved_touching_head_units") or 0) for st in steps)
        print(
            f"  run {run_name} (others={others}): {len(carved_steps)}/{len(steps)} step(s) carved ({total_carved} total cells, "
            f"{total_strangers} strangers cells across {len(strangers_steps)} step(s), {total_touching_head} touching head)",
            file=out,
        )
        for st in carved_steps:
            sf = st.get("source_frames") or []
            sf_txt = format_spans([sf]) if sf else "?"
            by_other = st.get("carved_by_other") or {}
            by_txt = ", ".join(f"{k}={v}" for k, v in by_other.items() if v)
            strangers_n = int(by_other.get("strangers") or 0)
            print(
                f"    window {st.get('window', 1)} step {st.get('step')}: frames {sf_txt} | "
                f"strangers={strangers_n}, carved_total={st.get('carved_units_total', 0)} ({by_txt}), "
                f"touching_head={st.get('carved_touching_head_units', 0)}, grown={st.get('grown_units')}->final={st.get('final_units')}",
                file=out,
            )

    # 4. Held or emptied frames
    print("\n[4] held or emptied frames:", file=out)
    any_held_or_empty = False
    unlaid_while_tracked = unlaid_while_tracked_of(capture, loads)
    for subj, spans in sorted(unlaid_while_tracked.items()):
        if spans:
            any_held_or_empty = True
            n_frames = sum(int(b) - int(a) + 1 for a, b in spans)
            print(
                f"  unlaid_while_tracked ({subj}): {n_frames} frame(s) [{format_spans(spans)}] (tracked > 0 px, carried == 0 px)",
                file=out,
            )

    unresolved_hh = unresolved_hair_in_hole_of(capture, loads)
    for subj, hh_vals in sorted(unresolved_hh.items()):
        any_held_or_empty = True
        if hh_vals.get("lost_asked"):
            print(
                f"  unresolved_hair_in_hole ({subj}) lost_asked_on_source_frames: frames {format_spans(hh_vals['lost_asked'])}",
                file=out,
            )
        if hh_vals.get("hair_undecided"):
            print(
                f"  unresolved_hair_in_hole ({subj}) hair_undecided_on_source_frames: frames {format_spans(hh_vals['hair_undecided'])}",
                file=out,
            )

    unmasked_diag = unmasked_frame_diagnostics_of(capture, loads)
    for subj, dinfo in sorted(unmasked_diag.items()):
        drows = dinfo.get("rows") or []
        if drows:
            any_held_or_empty = True
            med_cpx = dinfo.get("median_carried_px") or 0
            print(
                f"  tracker_and_mask_gap_table ({subj}, median_carried={med_cpx} px, {len(drows)} frame(s) empty or <50% median):",
                file=out,
            )
            for r in drows:
                t_str = f"{r['track_px']} px" if r["track_px"] else f"share={r['track_share']:.5f}"
                tag = "unmasked_frame" if r["carried_px"] == 0 else "shrunk_mask_frame"
                print(
                    f"  {tag} ({subj}) {r['source_frame']}: carried={r['carried_px']} px (tube={r.get('tube_px', 0)} px, raw_part={r.get('raw_part_px', 0)} px), "
                    f"track={t_str}, off_tube_suppressed={r['off_tube_suppressed_px']} px, "
                    f"pose={r['pose']}, face_pts={r['face_points']}, grade={r['grade_did']}, "
                    f"lost={r.get('lost_state', r['lost'])}, hair_undecided={r['hair_undecided_px']} px",
                    file=out,
                )
        if dinfo.get("has_visible_signal"):
            print(
                f"  PAUSE — VISIBLE HEAD/HAIR SIGNALS ON UNMASKED FRAME(S) ({subj}): "
                "overriding her_head_lost_in_view / unclaimed_hair_touches_another in flags.json adds 0 px to parts_held.mkv; "
                "mark head_marks ('hers'/'hidden'/'not hers') in plan.json or cut the load window with --drop and rebuild",
                file=out,
            )

    emptied_grade = flags_data.get("emptied_by_the_grade") or {}
    for subj, frames in sorted(emptied_grade.items()):
        if frames:
            any_held_or_empty = True
            if isinstance(frames, int):
                print(f"  flags.emptied_by_the_grade ({subj}): {frames} frame(s)", file=out)
            else:
                print(f"  flags.emptied_by_the_grade ({subj}): frames {format_spans(frames)}", file=out)
    held_top = flags_data.get("held_frames") or {}
    for subj, frames in sorted(held_top.items()):
        if frames:
            any_held_or_empty = True
            print(f"  flags.held_frames ({subj}): frames {format_spans(frames)}", file=out)

    named_empty = (flags_data.get("told_by_the_job") or {}).get("named_empty") or {}
    for subj, entries in sorted(named_empty.items()):
        for item in entries:
            if isinstance(item, list) and len(item) >= 2:
                any_held_or_empty = True
                why = item[2] if len(item) >= 3 else ""
                print(f"  named_empty ({subj}): frames {format_spans([[item[0], item[1]]])} | {why}", file=out)

    pg_files = sorted((capture / "subjects").glob("*/part_grades.json")) if (capture / "subjects").is_dir() else []
    for pgf in pg_files:
        pg = json.loads(pgf.read_text())
        subj = pg.get("subject") or pgf.parent.name
        ps = pg.get("part_sizes") or {}
        under_floor = ps.get("source_frames_under_the_floor") or []
        if under_floor:
            any_held_or_empty = True
            print(f"  part_grades ({subj}) under_the_floor (floor_px={ps.get('floor_px')}): frames {format_spans(under_floor)}", file=out)
        emptied_floor = (ps.get("emptied_by_the_floor") or {}).get("source_frames") or []
        if emptied_floor:
            any_held_or_empty = True
            print(f"  part_grades ({subj}) emptied_by_the_floor: frames {format_spans(emptied_floor)}", file=out)
        for key in ("side_with_eye_frames", "back_of_head_frames", "proposed_drop_frames", "overridden_by_word_frames"):
            vals = pg.get(key) or []
            if vals:
                any_held_or_empty = True
                print(f"  part_grades ({subj}) {key}: frames {format_spans(vals)}", file=out)
        by_did: dict[str, list[int]] = {}
        for fr in pg.get("frames") or []:
            did = fr.get("did")
            if did and did != "kept" and "source_frame" in fr:
                by_did.setdefault(str(did), []).append(int(fr["source_frame"]))
        for did, fnums in sorted(by_did.items()):
            any_held_or_empty = True
            print(f"  part_grades ({subj}) {did} ({len(fnums)} frame(s)): frames {format_spans(fnums)}", file=out)

    if not any_held_or_empty:
        print("  held_or_emptied: none (0 frames held or emptied)", file=out)

    return {
        "capture": str(capture),
        "loads": loads,
        "written": status.get("written"),
        "made_by": status.get("made_by"),
        "verdict": verdict,
        "flags_count": len(flags_list),
        "unlaid_while_tracked": unlaid_while_tracked,
        "unresolved_hair_in_hole": unresolved_hh,
        "unmasked_diagnostics": unmasked_diag,
        "recipe_check": recipe_info,
    }


def _format_unresolved_hair_in_hole(unresolved_hh: dict[str, dict]) -> str:
    bits = []
    for subj, vals in sorted(unresolved_hh.items()):
        s_bits = []
        if vals.get("lost_asked"):
            s_bits.append(f"lost_asked={format_spans(vals['lost_asked'])}")
        if vals.get("hair_undecided"):
            s_bits.append(f"hair_undecided={format_spans(vals['hair_undecided'])}")
        if s_bits:
            bits.append(f"{subj} ({', '.join(s_bits)})")
    return "; ".join(bits)


def record_initial(
    capture: Path,
    who: str,
    loads: list[str],
    ack_unlaid: str | None = None,
    ack_recipe: str | None = None,
    out=sys.stdout,
) -> dict:
    """Record `who` as having read the flags printout on `capture` for `loads`."""
    who_clean = (who or "").strip()
    if not who_clean:
        raise Refused("--initial WHO requires a non-empty reader name")
    status = json.loads((capture / "status.json").read_text())
    flags_bytes = (capture / "flags.json").read_bytes()
    flags_sha = hashlib.sha256(flags_bytes).hexdigest()
    now = now_iso()
    unresolved_hh = unresolved_hair_in_hole_of(capture, loads)
    if unresolved_hh:
        hh_summary = _format_unresolved_hair_in_hole(unresolved_hh)
        raise Refused(
            f"unresolved_hair_in_hole on {capture.name} ({hh_summary}): overriding her_head_lost_in_view / "
            "unclaimed_hair_touches_another in flags.json does NOT recover the mask (0 px in parts_held.mkv). "
            "Mark the frames in plan.json subjects.<subj>.head_marks ('hers' to recover head/hair into the mask, "
            "'hidden' if occluded, or 'not hers') or cut the load window with --drop, and rebuild the capture"
        )
    unlaid = unlaid_while_tracked_of(capture, loads)
    ack_clean = (ack_unlaid or "").strip() if ack_unlaid is not None else ""
    if ack_unlaid is not None and (not ack_clean or ack_clean.startswith("-")):
        raise Refused("--ack-unlaid WHY requires a non-empty reason")
    ack_recipe_clean = (ack_recipe or "").strip() if ack_recipe is not None else ""
    if ack_recipe is not None and (not ack_recipe_clean or ack_recipe_clean.startswith("-")):
        raise Refused("--ack-recipe WHY requires a non-empty reason")
    recipe_info = recipe_check_of(capture, loads)
    unexplained_recipe = [r for r in recipe_info["rows"] if r.get("status") == "unexplained"]
    mouthless_refs = recipe_info["mouthless_references"]
    entry = {
        "by": who_clean,
        "at": now,
        "loads": list(loads),
        "capture_written": status.get("written"),
        "capture_made_by": status.get("made_by"),
        "flags_sha256": flags_sha,
    }
    if unexplained_recipe or mouthless_refs:
        if not ack_recipe_clean:
            parts_msg = []
            if unexplained_recipe:
                parts_msg.append(
                    "unexplained recipe deviation(s): "
                    + ", ".join(f"{r['knob']} (recipe={r['recipe']}, this_load={r['this_load']})" for r in unexplained_recipe)
                )
            if mouthless_refs:
                parts_msg.append(
                    "mouthless_reference (0 open-mouth and 0 closed-mouth drawn frames with closed_mouth='outlined'): "
                    + ", ".join(mouthless_refs)
                )
            raise Refused(
                f"recipe check failed on {capture.name} ({'; '.join(parts_msg)}). "
                "Fix the job/reference to match the subject's recipe in plan.json, or pass --ack-recipe WHY to acknowledge"
            )
        entry["recipe_issues"] = {
            "unexplained": unexplained_recipe,
            "mouthless_references": mouthless_refs,
        }
        entry["recipe_ack"] = {"why": ack_recipe_clean, "by": who_clean, "at": now}
    if unlaid:
        peer_look = peer_look_covering(capture, loads, status)
        if not ack_clean and peer_look is None:
            unlaid_summary = "; ".join(f"{subj}: frames {format_spans(spans)}" for subj, spans in sorted(unlaid.items()))
            raise Refused(
                f"unlaid_while_tracked on {capture.name} ({unlaid_summary}): frame(s) have track_px > 0 and carried_px == 0. "
                f"Pass --ack-unlaid WHY to acknowledge these unlaid tracked frames, or record a peer look in {capture.name}/looks.json first"
            )
        entry["unlaid_while_tracked"] = unlaid
        if ack_clean:
            entry["unlaid_while_tracked_ack"] = {"why": ack_clean, "by": who_clean, "at": now}
        else:
            entry["unlaid_while_tracked_ack"] = {
                "covered_by_look": peer_look.get("by"),
                "look_at": peer_look.get("at"),
                "by": who_clean,
                "at": now,
            }
    looks_dir = capture / "looks"
    looks_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for ld in loads:
        shown_file = looks_dir / f"{ld}.shown.json"
        if shown_file.is_file():
            shown = json.loads(shown_file.read_text())
            shown["flags_read"] = entry
            shown_file.write_text(json.dumps(shown, indent=1) + "\n")
            written_paths.append(shown_file)
        beside_file = looks_dir / f"{ld}.flags_read.json"
        beside_file.write_text(json.dumps({"load": ld, **entry}, indent=1) + "\n")
        written_paths.append(beside_file)
    print(f"\nrecorded flags_read initial: {who_clean} at {now} for {', '.join(loads)} ({', '.join(p.name for p in written_paths)})", file=out)
    return entry


def check_flags_read(capture: Path, load_name: str, status: dict | None = None) -> dict:
    """Verify that `capture` has a valid `flags_read` initial for `load_name` later than `status.json`'s `written`.

    Raises `Refused` when missing, stale against `status.json`, or stale against `flags.json`.
    """
    if status is None:
        status_file = capture / "status.json"
        if not status_file.is_file():
            raise Refused(f"{capture.name} holds no status.json")
        status = json.loads(status_file.read_text())
    unresolved_hh = unresolved_hair_in_hole_of(capture, [load_name])
    if unresolved_hh:
        hh_summary = _format_unresolved_hair_in_hole(unresolved_hh)
        raise Refused(
            f"unresolved_hair_in_hole on {capture.name} ({hh_summary}): overriding her_head_lost_in_view / "
            "unclaimed_hair_touches_another in flags.json does NOT recover the mask (0 px in parts_held.mkv). "
            "Mark the frames in plan.json subjects.<subj>.head_marks ('hers' to recover head/hair into the mask, "
            "'hidden' if occluded, or 'not hers') or cut the load window with --drop, and rebuild the capture"
        )
    how = f"run `bench/flags_before_render.py {capture} --initial WHO`"
    shown_file = capture / "looks" / f"{load_name}.shown.json"
    beside_file = capture / "looks" / f"{load_name}.flags_read.json"
    rec = None
    if shown_file.is_file():
        try:
            rec = json.loads(shown_file.read_text()).get("flags_read")
        except Exception:
            rec = None
    if not isinstance(rec, dict) and beside_file.is_file():
        try:
            rec = json.loads(beside_file.read_text())
        except Exception:
            rec = None
    if not isinstance(rec, dict) or not str(rec.get("by") or "").strip() or not rec.get("at"):
        raise Refused(f"nobody has initialed the pre-render flags printout for {load_name!r} on {capture.name}: {how}")
    rec_loads = rec.get("loads")
    if isinstance(rec_loads, list) and load_name not in rec_loads:
        raise Refused(f"the flags-read initial on {capture.name} is for {rec_loads}, not {load_name!r}: {how}")
    written = status.get("written")
    if written and not later(rec["at"], written):
        raise Refused(
            f"the flags-read initial on {capture.name} ({rec['by']}, {in_utc(rec['at'])}) is not later than the capture "
            f"({in_utc(written)}): it is of an earlier capture; {how}"
        )
    flags_file = capture / "flags.json"
    if rec.get("flags_sha256") and flags_file.is_file():
        cur_sha = hashlib.sha256(flags_file.read_bytes()).hexdigest()
        if rec["flags_sha256"] != cur_sha:
            raise Refused(
                f"{capture.name}/flags.json changed since {rec['by']} initialed it at {in_utc(rec['at'])}: {how}"
            )
    recipe_info = recipe_check_of(capture, [load_name])
    unexplained_recipe = [r for r in recipe_info["rows"] if r.get("status") == "unexplained"]
    mouthless_refs = recipe_info["mouthless_references"]
    if unexplained_recipe or mouthless_refs:
        r_ack = rec.get("recipe_ack")
        has_valid_r_ack = isinstance(r_ack, dict) and bool(str(r_ack.get("why") or "").strip())
        if not has_valid_r_ack:
            parts_msg = []
            if unexplained_recipe:
                parts_msg.append(
                    "unexplained recipe deviation(s): "
                    + ", ".join(f"{r['knob']} (recipe={r['recipe']}, this_load={r['this_load']})" for r in unexplained_recipe)
                )
            if mouthless_refs:
                parts_msg.append("mouthless_reference: " + ", ".join(mouthless_refs))
            raise Refused(
                f"recipe check failed on {capture.name} ({'; '.join(parts_msg)}) with no recorded --ack-recipe WHY in the flags-read initial for {load_name!r}: "
                f"run `bench/flags_before_render.py {capture} --initial WHO --ack-recipe WHY`"
            )
    unlaid = unlaid_while_tracked_of(capture, [load_name])
    if unlaid:
        ack = rec.get("unlaid_while_tracked_ack")
        peer_look = peer_look_covering(capture, [load_name], status)
        has_valid_ack = (
            isinstance(ack, dict)
            and bool(str(ack.get("why") or ack.get("covered_by_look") or "").strip())
            and rec.get("unlaid_while_tracked") == unlaid
        )
        if not has_valid_ack and peer_look is None:
            unlaid_summary = "; ".join(f"{subj}: frames {format_spans(spans)}" for subj, spans in sorted(unlaid.items()))
            raise Refused(
                f"unlaid_while_tracked on {capture.name} ({unlaid_summary}) has no recorded --ack-unlaid WHY in the flags-read initial for {load_name!r} "
                f"(and no peer look in looks.json): run `bench/flags_before_render.py {capture} --initial WHO --ack-unlaid WHY`"
            )
    return {
        "by": str(rec["by"]).strip(),
        "at": rec["at"],
        "at_utc": in_utc(rec["at"]),
        "capture_written": written,
        "capture_written_utc": in_utc(written) if written else None,
        "flags_sha256": rec.get("flags_sha256"),
        "unlaid_while_tracked": unlaid,
        "unlaid_while_tracked_ack": rec.get("unlaid_while_tracked_ack"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", type=Path, help="path to the capture folder (e.g. data/2026-10-24_lead_fp_4310)")
    p.add_argument("--load", default=None, help="load name(s) separated by '+', defaulting to the capture's load(s)")
    p.add_argument("--initial", default=None, metavar="WHO", help="record WHO as having read this printout into looks/<load>.shown.json and looks/<load>.flags_read.json")
    p.add_argument("--ack-unlaid", default=None, metavar="WHY", help="reason acknowledging unlaid_while_tracked frames when recording --initial WHO")
    p.add_argument("--ack-recipe", default=None, metavar="WHY", help="reason acknowledging unexplained recipe deviations or mouthless_reference when recording --initial WHO")
    a = p.parse_args(argv)
    try:
        info = print_capture_report(a.capture, load_arg=a.load)
        if a.initial is not None:
            record_initial(a.capture, a.initial, info["loads"], ack_unlaid=a.ack_unlaid, ack_recipe=a.ack_recipe)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
