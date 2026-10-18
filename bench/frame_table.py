#!/usr/bin/env python3
"""Build one per-frame table (`frame_table.jsonl` and `frame_table.md`) for a load's capture folder.

    <python> bench/frame_table.py <capture_folder> [--job <job_file>] [--subject <label>] [--out-dir <dir>]

For every source frame of the shot/load, joins the four capture records (`frames.csv`, `subjects/<person>/per_frame.csv`,
`manifest.json` + `subjects/<person>/part_grades.json` + `shots__<preview>.json`, and `flags.json`) with the preview's
saved graph (`preview.json`), the who-is-who record, the job file, and the owner's shot presence authority
(`data/<video>/shots.json` and `data/<video>/cast/cast.jsonl`).

At the top of the table (the first line of `frame_table.jsonl` and the header section of `frame_table.md`):
- `sam_inputs`: what the SAM 3.1 tracking node (`MiniMaxH3SubjectTrack`) was given in the preview's saved graph,
  its wired `MiniMaxH3SAM31Corrections` inputs and `CheckpointLoaderSimple.inputs.ckpt_name` (beside `h3_config.SEGMENTER`),
  its `others` wiring, and which tracked id became this person (`who_is_who.json` / `shots__<preview>.json`), on whose word;
- `in_view_spans` and `out_of_view_spans`: `[first, last]` source frame runs where her SAM 3.1 track is present
  (`in_view == True`, from `shots__<preview>.json` and `per_frame.csv` `track_px`, never comparing `track_score` with
  `detection_threshold`) and where it is empty or on others (`frames_without_subject` / `frames_on_others` / `track_px == 0`);
- `in_the_shot_by_the_owner`: whether the owner says this person is in this shot (`"yes"`, `"no"`, or `"not answered"`),
  read from `shots.json` (and `cast/cast.jsonl`);
- `owner_says_in_shot_track_empty`: `[first, last]` source frame spans where the owner says `"yes"` in `shots.json`
  and her SAM 3.1 track is empty or on another person (`in_view == False`);
- `flag_spans`: the gate flags from `flags.json` with their bridged `[first, last]` spans (`frame_spans` uses `join=2`).

Then one JSON line per source frame in `frame_table.jsonl`:
- `source_frame`, `load_frame` (`null` when `not_in_load`);
- `tracked`: `{"share": float, "px": int}` from `per_frame.csv` (and exact mask px when `masks__<by>.npz` is present),
  or `false` with `tracked_why`;
- `loaded_part_px`, `saved_part_px`, `saved_part_height_px`, plus `part_px` (`part_px_measures`) and `part_height_px`
  (`part_height_px_measures`) stating what each size column measures on each outcome;
- `outcome`: one of `replaced`, `held` (with `from_frame` when the record says), `dropped` (with `reason` from `drop_by`
  only), `emptied` (with `cause`: `floor`, `grade`, or `nothing saved`), `not_in_load` (with `reason` when named in
  `left_out`), or `no per-frame record` (when `part_grades.json` is absent on a part load);
- `masks`: the track and part mask files this frame is read from, each with `frame_index` and `sha256` (hashed once per
  file);
- `others_given_back`: the other people's region reach on this frame (from `frames.csv`);
- `inside_flag_spans`: `["inside the span of fNNN", ...]` from `flags.json`'s bridged spans;
- `sam`: `{"track_score": float | None, "track_px": int | None, "in_view": bool, "in_view_from": str}`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workflows import h3_config  # noqa: E402

#: Schema tag for the top-of-table header line in `frame_table.jsonl`. Reasoned: version 1 of the per-load frame table.
SCHEMA = "h3_frame_table/1"
#: Exhaustive set of per-frame outcomes permitted in a frame table row (brief of 2026-10-16 + D10 item 4).
OUTCOMES = ("replaced", "held", "dropped", "emptied", "not_in_load", "no per-frame record")
#: Exhaustive set of causes for `outcome == "emptied"` (brief of 2026-10-16).
EMPTIED_CAUSES = ("floor", "grade", "nothing saved")
#: Closed vocabulary of `did` values written to `part_grades.json` by `bench/capture_masked_run.py::grade_parts` (lines 769-814).
CLOSED_DID_VOCAB = frozenset({
    "filled",
    "filled where nothing was labelled",
    "filled on a person's word",
    "held by the floor rule",
    "saved part kept, fill refused",
    "emptied",
    "left empty: the word not held",
    "left empty: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
    "emptied: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
    "emptied: no standing part in the load to hold it from",
})
#: Subset of `CLOSED_DID_VOCAB` that produces `outcome == "held"`.
HELD_DIDS = frozenset({
    "filled",
    "filled where nothing was labelled",
    "filled on a person's word",
    # the floor holds (build (B), bench/capture_masked_run.py grade_holds, 2026-10-16)
    "held by the floor rule",
})
#: Subset of `CLOSED_DID_VOCAB` that produces `outcome == "emptied"`.
EMPTIED_DIDS = frozenset({
    "emptied",
    "left empty: the word not held",
    # a hold whose shape lies mostly off her track, and a fragment with nothing standing to hold it from (build (B), 2026-10-16)
    "left empty: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
    "emptied: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
    "emptied: no standing part in the load to hold it from",
})
#: Set-bit lookup table per byte (0..255), used to count packed bool mask pixels in `masks__<by>.npz` without unpacking.
#: Inherited from `bench/frame_coverage.py::POP`.
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)

_SEED_RE = re.compile(
    r"^frame\s*(\d+)\s*[:=]\s*(?:(none)|(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?))?)$",
    re.IGNORECASE,
)
_CORR_SHOT_RE = re.compile(r"^shot\s*(\d+)\s*[:=]?\s*(?:person\s*(\d+)|(none))$", re.IGNORECASE)
_CORR_FRAME_RE = re.compile(
    r"^frame\s*(\d+)\s*[:=]\s*(?:([a-z_]+)\s+)?(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?))?$",
    re.IGNORECASE,
)


class Refused(Exception):
    """A hard failure: a missing mask file, a frame of the load with no row, a dropped frame with no reason, or an invalid record."""


def _rel(path: Path, base: Path = REPO) -> str:
    for b in (base, base.resolve(), (base / "data").resolve().parent):
        for p in (path, path.resolve()):
            try:
                return str(p.relative_to(b))
            except ValueError:
                pass
    return str(path)


def _sha256_once(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        if not path.is_file():
            raise Refused(f"mask file is missing: {path}")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[resolved] = h.hexdigest()
    return cache[resolved]


def _expand_spans(spans: list | None) -> set[int]:
    out: set[int] = set()
    for item in spans or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.update(range(int(item[0]), int(item[1]) + 1))
        elif isinstance(item, int):
            out.add(item)
    return out


def _to_spans(frames: list[int] | set[int]) -> list[list[int]]:
    ordered = sorted(set(int(x) for x in frames))
    if not ordered:
        return []
    spans: list[list[int]] = [[ordered[0], ordered[0]]]
    for f in ordered[1:]:
        if f == spans[-1][1] + 1:
            spans[-1][1] = f
        else:
            spans.append([f, f])
    return spans


def _parse_drop_spans(spec: str | None) -> set[int]:
    out: set[int] = set()
    for part in str(spec or "").split("+"):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s = (part.split("-") * 2)[:2]
        out.update(range(int(lo_s), int(hi_s) + 1))
    return out


def _find_job_file(capture_dir: Path, job_path: Path | None, load_name: str) -> tuple[Path | None, dict | None]:
    if job_path is not None:
        if not job_path.is_file():
            raise Refused(f"job file {job_path} does not exist")
        raw = json.loads(job_path.read_text())
        return job_path, raw
    job_dir = capture_dir.with_name(capture_dir.name + "_job")
    for candidate in (
        job_dir / "job.plan.json",
        job_dir / "job.render.json",
        job_dir / "job.json",
        capture_dir / "job.json",
    ):
        if candidate.is_file():
            raw = json.loads(candidate.read_text())
            built_file = job_dir / f"{load_name}.plan.built.json"
            if built_file.is_file():
                built = json.loads(built_file.read_text())
                named = built.get("job_file")
                if named:
                    p = Path(named) if Path(named).is_absolute() else REPO / named
                    if p.is_file():
                        return p, json.loads(p.read_text())
            return candidate, raw
    return None, None


def _parse_hand_entries(text: str, shots_first: int) -> list[dict]:
    out: list[dict] = []
    for raw_piece in re.split(r"[\n;]", str(text or "")):
        piece = raw_piece.strip()
        if not piece:
            continue
        m_seed = _SEED_RE.match(piece)
        if m_seed is not None:
            p_frame = int(m_seed[1])
            s_frame = shots_first + p_frame
            if m_seed[2]:
                out.append({"preview_frame": p_frame, "source_frame": s_frame, "kind": "none", "raw": piece})
            elif m_seed[5] is not None and m_seed[6] is not None:
                box = [float(m_seed[3]), float(m_seed[4]), float(m_seed[5]), float(m_seed[6])]
                out.append({"preview_frame": p_frame, "source_frame": s_frame, "kind": "box", "box": box, "raw": piece})
            else:
                pt = [float(m_seed[3]), float(m_seed[4])]
                out.append({"preview_frame": p_frame, "source_frame": s_frame, "kind": "point", "point": pt, "raw": piece})
            continue
        m_shot = _CORR_SHOT_RE.match(piece)
        if m_shot is not None:
            shot_num = int(m_shot[1])
            person = None if m_shot[3] else int(m_shot[2])
            out.append({"shot": shot_num, "person": person, "kind": "shot_person", "raw": piece})
            continue
        m_frame = _CORR_FRAME_RE.match(piece)
        if m_frame is not None:
            p_frame = int(m_frame[1])
            s_frame = shots_first + p_frame
            action = (m_frame[2] or "add").lower()
            if m_frame[5] is not None and m_frame[6] is not None:
                box = [float(m_frame[3]), float(m_frame[4]), float(m_frame[5]), float(m_frame[6])]
                out.append({"preview_frame": p_frame, "source_frame": s_frame, "action": action, "kind": "box", "box": box, "raw": piece})
            else:
                pt = [float(m_frame[3]), float(m_frame[4])]
                out.append({"preview_frame": p_frame, "source_frame": s_frame, "action": action, "kind": "point", "point": pt, "raw": piece})
            continue
        out.append({"kind": "raw", "raw": piece})
    return out


def _resolve_wired_segmenter(graph: dict, track_node: dict) -> tuple[str, dict | None]:
    """Follow `track_node.inputs.segmenter` through optional `MiniMaxH3SAM31Corrections` to `CheckpointLoaderSimple`."""
    inp = track_node.get("inputs", {})
    seg_link = inp.get("segmenter")
    corrections_info: dict | None = None
    visited: set[str] = set()
    while isinstance(seg_link, list) and len(seg_link) == 2:
        nid = str(seg_link[0])
        if nid in visited or nid not in graph:
            break
        visited.add(nid)
        upstream = graph[nid]
        if not isinstance(upstream, dict):
            break
        ct = upstream.get("class_type")
        up_inp = upstream.get("inputs", {})
        if ct == "MiniMaxH3SAM31Corrections":
            corrections_info = {
                "node_id": nid,
                "correct_image_range": up_inp.get("correct_image_range"),
                "correct_text_activation": up_inp.get("correct_text_activation"),
                # None on a graph made before the node had the input (2026-10-17): the tracker then kept ComfyUI's memory
                "memory_selection": up_inp.get("memory_selection"),
            }
            seg_link = up_inp.get("segmenter")
            continue
        if ct == "CheckpointLoaderSimple":
            ckpt = up_inp.get("ckpt_name")
            if not ckpt:
                raise Refused(f"CheckpointLoaderSimple node {nid!r} has no ckpt_name")
            return str(ckpt), corrections_info
        break
    raise Refused("could not resolve CheckpointLoaderSimple wired to MiniMaxH3SubjectTrack.inputs.segmenter")


def _read_sam_inputs(
    capture_dir: Path,
    job: dict | None,
    load_name: str,
    subject: str,
    sighting: dict,
    shots_table: dict,
) -> dict:
    by = sighting["by"]
    shots_first_raw = sighting.get("shots_first_source_frame")
    if shots_first_raw is None:
        raise Refused(f"{capture_dir}: sighting for subject {subject!r} in manifest.json has no shots_first_source_frame")
    shots_first = int(shots_first_raw)

    preview_dir: Path | None = None
    if job and job.get("preview"):
        p = Path(job["preview"])
        for root in (capture_dir.parent.parent, REPO):
            cand = p if p.is_absolute() else root / p
            if cand.is_dir():
                preview_dir = cand
                break
    elif (capture_dir / "preview.json").is_file():
        preview_dir = capture_dir
    else:
        raise Refused(f"cannot locate preview.json for sighting {by!r}: job has no 'preview' folder (no glob fallback)")

    if preview_dir is None or not (preview_dir / "preview.json").is_file():
        raise Refused(f"preview graph preview.json not found for sighting {by!r} (looked in {preview_dir})")

    preview_graph_path = preview_dir / "preview.json"
    raw_graph = json.loads(preview_graph_path.read_text())
    graph = raw_graph.get("prompt", raw_graph)

    track_nodes = [
        (str(nid), node)
        for nid, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3SubjectTrack"
    ]
    chosen_id: str | None = None
    chosen_node: dict | None = None
    if len(track_nodes) == 1:
        chosen_id, chosen_node = track_nodes[0]
    else:
        target_node_ids: set[str] = set()
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type", "")
            prefix = str(node.get("inputs", {}).get("filename_prefix", ""))
            if ct in ("MiniMaxH3SaveShotTable", "VHS_VideoCombine") and (
                prefix.endswith(f"_{subject}_shots") or prefix.endswith(f"_{subject}_track")
            ):
                for val in node.get("inputs", {}).values():
                    if isinstance(val, list) and len(val) == 2:
                        target_node_ids.add(str(val[0]))
        for nid, node in track_nodes:
            if str(nid) in target_node_ids:
                chosen_id, chosen_node = nid, node
                break
    if chosen_node is None:
        raise Refused(f"no MiniMaxH3SubjectTrack node for subject {subject!r} in {preview_graph_path}")

    model_file, corrections_node_info = _resolve_wired_segmenter(graph, chosen_node)

    inp = chosen_node.get("inputs", {})
    if inp.get("detection_threshold") is None:
        raise Refused(f"MiniMaxH3SubjectTrack node {chosen_id!r} in {preview_graph_path} has no detection_threshold")
    seeds_raw = str(inp.get("seeds", "") or "")
    corrections_raw = str(inp.get("corrections", "") or "")
    parsed_seeds = _parse_hand_entries(seeds_raw, shots_first)
    parsed_corrections = _parse_hand_entries(corrections_raw, shots_first)

    others_link = inp.get("others")
    others_wired = (
        {"node_id": str(others_link[0]), "output_slot": int(others_link[1])}
        if isinstance(others_link, list) and len(others_link) == 2
        else None
    )

    # Read who_is_who.json (required; never fabricate answered_by or fall back to shot table)
    job_dir = capture_dir.with_name(capture_dir.name + "_job")
    who_candidates: list[Path] = []
    if job and job.get("who_is_who"):
        wp = Path(job["who_is_who"])
        who_candidates.append(wp if wp.is_absolute() else capture_dir.parent / wp)
    who_candidates.extend([
        job_dir / f"{load_name}.who_is_who.json",
        capture_dir / "who_is_who.json",
    ])
    who_file = next((p for p in who_candidates if p.is_file()), None)
    if who_file is None:
        raise Refused(
            f"no who_is_who.json found for load {load_name!r} (looked in {[str(p) for p in who_candidates]})"
        )
    shot_0 = (shots_table.get("shots") or [{}])[0]
    subj_block = shot_0.get("subject") or {}
    wdata = json.loads(who_file.read_text())
    row0 = (wdata.get("rows") or [{}])[0]
    t_entry = (row0.get("trackers") or {}).get(subject) or {}
    who_info = {
        "file": _rel(who_file),
        "tracked_person_id": t_entry.get("called_person"),
        "called": t_entry.get("called"),
        "answered_by": t_entry.get("answered_by"),
        "tile_frame": t_entry.get("tile_frame"),
        "on_whose_word": row0.get("note"),
        "shot_table_why": subj_block.get("why"),
    }

    return {
        "preview": by,
        "preview_graph": _rel(preview_graph_path),
        "track_node_id": chosen_id,
        "shots_first_source_frame": shots_first,
        "model_file": model_file,
        "h3_config_segmenter": h3_config.SEGMENTER,
        "sam31_corrections": corrections_node_info,
        "others_wired": others_wired,
        "subject_phrase": inp.get("subject_phrase"),
        "pick": inp.get("pick"),
        "pick_on": inp.get("pick_on"),
        "match": inp.get("match"),
        "detection_threshold": float(inp["detection_threshold"]),
        "max_people": inp.get("max_people"),
        "head_phrase": inp.get("head_phrase"),
        "cuts": inp.get("cuts", ""),
        "seeds_raw": seeds_raw,
        "seeds": parsed_seeds,
        "corrections_raw": corrections_raw,
        "corrections": parsed_corrections,
        "who_is_who": who_info,
    }


def _read_owner_shot_presence(
    capture_dir: Path,
    manifest: dict,
    job: dict | None,
    subject: str,
    load_first: int,
    load_last: int,
) -> tuple[dict, dict | None, Path | None]:
    """Read the owner's presence ruling for `subject` in this shot from `data/<video>/shots.json` and `cast/cast.jsonl`."""
    import clip_folder

    data_dir = capture_dir.parent
    shots_path: Path | None = None
    if (capture_dir / "shots.json").is_file():
        shots_path = capture_dir / "shots.json"
    else:
        clip_name = manifest.get("clip") or (Path(job["source"]).name if job and job.get("source") else None)
        if not clip_name:
            return {"present": "not answered", "why": "neither manifest['clip'] nor job['source'] names a clip"}, None, None
        folder = clip_folder.folder_of_file(clip_name, data_dir)
        if (folder / "shots.json").is_file():
            shots_path = folder / "shots.json"
        else:
            return {"present": "not answered", "why": f"no shots.json at {_rel(folder / 'shots.json')}"}, None, None

    shots_data = json.loads(shots_path.read_text())
    matched_shot = None
    shot_num = None
    for idx, s in enumerate(shots_data.get("shots", []), 1):
        if int(s["first"]) <= load_first and load_last <= int(s["last"]):
            matched_shot, shot_num = s, idx
            break
    if matched_shot is None:
        overlapping = [
            (idx, s)
            for idx, s in enumerate(shots_data.get("shots", []), 1)
            if int(s["first"]) <= load_last and int(s["last"]) >= load_first
        ]
        if len(overlapping) > 1:
            raise Refused(
                f"load {load_first}-{load_last} spans {len(overlapping)} shots in {shots_path.name}: "
                f"{[(s.get('id', idx), int(s['first']), int(s['last'])) for idx, s in overlapping]}"
            )
        if len(overlapping) == 1:
            shot_num, matched_shot = overlapping[0]
    if matched_shot is None:
        return (
            {"present": "not answered", "shots_file": _rel(shots_path), "why": f"no shot in {shots_path.name} covers {load_first}-{load_last}"},
            None,
            shots_path,
        )

    s_first, s_last = int(matched_shot["first"]), int(matched_shot["last"])
    who_map = matched_shot.get("who") or {}
    who_entry = who_map.get(subject) if isinstance(who_map.get(subject), dict) else {}
    raw_present = who_entry.get("present")
    present_val = str(raw_present) if raw_present in ("yes", "no") else "not answered"

    cast_records: list[dict] = []
    cast_path = shots_path.parent / "cast" / "cast.jsonl"
    if cast_path.is_file():
        for line in cast_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            r_frame = rec.get("frame")
            in_span = isinstance(r_frame, int) and s_first <= r_frame <= s_last
            same_shot = rec.get("shot") is not None and rec.get("shot") in (matched_shot.get("id"), shot_num)
            if (in_span or same_shot) and (rec.get("subject") in (None, subject) and rec.get("person") in (None, subject)):
                cast_records.append(rec)

    info = {
        "shots_file": _rel(shots_path),
        "shot_id": matched_shot.get("id", shot_num),
        "shot_span": [s_first, s_last],
        "present": present_val,
        "how": who_entry.get("how"),
        "by": who_entry.get("by"),
        "at": who_entry.get("at") or (cast_records[-1].get("at") if cast_records else None),
        "frames_boxed": who_entry.get("frames_boxed"),
        "cast_file": _rel(cast_path) if cast_path.is_file() else None,
        "cast_records": len(cast_records),
    }
    return info, matched_shot, shots_path


def build_frame_table(
    capture_dir: Path,
    *,
    job_path: Path | None = None,
    out_dir: Path | None = None,
    subject_arg: str | None = None,
) -> dict[str, Path]:
    """Build `frame_table.jsonl` and `frame_table.md` for `capture_dir`; return their paths."""
    capture_dir = Path(capture_dir)
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise Refused(f"{capture_dir}: no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs") or []
    if not runs:
        raise Refused(f"{capture_dir}: manifest.json has no runs")
    run_entry = runs[0]
    load_name = run_entry.get("name")
    if not load_name:
        raise Refused(f"{capture_dir}: manifest.json runs[0] has no 'name'")
    primary_subject = run_entry.get("subject")
    if not primary_subject:
        raise Refused(f"{capture_dir}: manifest.json runs[0] has no 'subject'")
    subject = subject_arg or primary_subject
    others = list(run_entry.get("others") or [])

    job_file, job_doc = _find_job_file(capture_dir, job_path, load_name)
    job = job_doc.get("job") if isinstance(job_doc, dict) else None
    loads_list = job_doc.get("loads") if isinstance(job_doc, dict) else None
    load_cfg = next((ld for ld in (loads_list or []) if ld.get("name") == load_name), None)

    if "first_frame" not in manifest or "frames" not in manifest:
        raise Refused(f"{capture_dir}: manifest.json is missing 'first_frame' or 'frames'")
    cap_first = int(manifest["first_frame"])
    cap_frames = int(manifest["frames"])
    cap_last = cap_first + cap_frames - 1

    if load_cfg is not None and "first" in load_cfg and "frames" in load_cfg:
        load_first = int(load_cfg["first"])
        load_frames = int(load_cfg["frames"])
    elif job is not None and "first" in job and "frames" in job:
        load_first = int(job["first"])
        load_frames = int(job["frames"])
    else:
        raise Refused(f"{capture_dir}: no load bounds ('first' and 'frames') found in job file for load {load_name!r}")
    load_last = load_first + load_frames - 1

    size_raw = manifest.get("size")
    if not isinstance(size_raw, list) or len(size_raw) < 2:
        raise Refused(f"{capture_dir}: manifest.json is missing 'size' [width, height]")
    width, height = int(size_raw[0]), int(size_raw[1])
    canvas_area = width * height

    subj_manifest = next((s for s in manifest.get("subjects", []) if s.get("label") == subject), None)
    if subj_manifest is None or not subj_manifest.get("sightings"):
        raise Refused(f"{capture_dir}: manifest.json has no sighting for subject {subject!r}")
    sighting = subj_manifest["sightings"][0]
    by = sighting["by"]
    subj_dir = capture_dir / "subjects" / subject

    # Verify required mask files exist and hash once per file
    sha_cache: dict[Path, str] = {}
    track_mask_path = subj_dir / f"track__{by}.mkv"
    if not track_mask_path.is_file() or track_mask_path.stat().st_size == 0:
        raise Refused(f"track mask file is missing or empty: {track_mask_path}")
    track_sha = _sha256_once(track_mask_path, sha_cache)

    is_whole_person_load = "parts" in sighting and sighting.get("parts") is None
    part_mask_path: Path | None = None
    part_sha: str | None = None
    part_mask_source: str
    if is_whole_person_load:
        part_mask_source = "whole-person track-only load (sighting.parts is null)"
    else:
        expects_held_part = (subj_dir / "part_grades.json").is_file() or ("parts_held_on_source_frames" in sighting)
        if expects_held_part:
            part_mask_path = subj_dir / f"parts_held__{by}.mkv"
            part_mask_source = f"parts_held__{by}.mkv (graded/held part mask in capture)"
        else:
            part_mask_path = subj_dir / f"parts__{by}.mkv"
            part_mask_source = f"parts__{by}.mkv (ungraded part mask in capture)"
        if not part_mask_path.is_file() or part_mask_path.stat().st_size == 0:
            raise Refused(f"part mask file is missing or empty: {part_mask_path}")
        part_sha = _sha256_once(part_mask_path, sha_cache)

    # Read shots__<by>.json for SAM 3.1 per-frame track_score, frames_without_subject, and frames_on_others
    shots_table_path = subj_dir / f"shots__{by}.json"
    if not shots_table_path.is_file():
        raise Refused(f"shot table is missing: {shots_table_path}")
    shots_table = json.loads(shots_table_path.read_text())

    sam_inputs = _read_sam_inputs(capture_dir, job, load_name, subject, sighting, shots_table)
    shots_first = int(sam_inputs["shots_first_source_frame"])

    sam_scores: dict[int, float | None] = {}
    without_subject_sfs: set[int] = set()
    on_others_sfs: set[int] = set()
    offscreen_shot_sfs: set[int] = set()
    shot_covered_sfs: set[int] = set()
    for shot in shots_table.get("shots", []):
        s_first = int(shot.get("first_frame", 0))
        t_scores = shot.get("track_score") or []
        shot_offscreen = (
            shot.get("on_screen") is False
            or shot.get("frames_with_subject") == 0
        )
        for k, sc in enumerate(t_scores):
            sf = shots_first + s_first + k
            sam_scores[sf] = float(sc) if sc is not None else None
            shot_covered_sfs.add(sf)
            if shot_offscreen:
                offscreen_shot_sfs.add(sf)
        if not t_scores and shot_offscreen and shot.get("frames"):
            for k in range(int(shot["frames"])):
                offscreen_shot_sfs.add(shots_first + s_first + k)
        for rel_f in _expand_spans(shot.get("frames_without_subject")):
            without_subject_sfs.add(shots_first + rel_f)
        for rel_f in _expand_spans(shot.get("frames_on_others")):
            on_others_sfs.add(shots_first + rel_f)

    # Optional packed masks npz for exact pixel counts
    npz_path = subj_dir / f"masks__{by}.npz"
    npz_track_px: dict[int, int] = {}
    npz_part_px: dict[int, int] = {}
    if npz_path.is_file():
        z = np.load(npz_path)
        if "track" in z.files:
            t_counts = POP[z["track"].reshape(len(z["track"]), -1)].sum(1, dtype=np.int64)
            for idx, val in enumerate(t_counts):
                npz_track_px[cap_first + idx] = int(val)
        if "parts" in z.files:
            p_counts = POP[z["parts"].reshape(len(z["parts"]), -1)].sum(1, dtype=np.int64)
            for idx, val in enumerate(p_counts):
                npz_part_px[cap_first + idx] = int(val)

    # Read subjects/<person>/per_frame.csv
    per_frame_csv = subj_dir / "per_frame.csv"
    if not per_frame_csv.is_file():
        raise Refused(f"per_frame.csv is missing: {per_frame_csv}")
    per_frame_rows: dict[int, dict[str, str]] = {}
    with open(per_frame_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("seen_by") and row["seen_by"] != by:
                continue
            per_frame_rows[int(row["source_frame"])] = row

    # Read frames.csv for others_given_back
    frames_csv = capture_dir / "frames.csv"
    if not frames_csv.is_file():
        raise Refused(f"frames.csv is missing: {frames_csv}")
    frames_rows: dict[int, dict[str, str]] = {}
    with open(frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            frames_rows[int(row["source_frame"])] = row

    # Ensure every frame of the load has a row in per_frame.csv and frames.csv
    missing_in_load = [
        sf for sf in range(load_first, load_last + 1)
        if sf not in per_frame_rows or sf not in frames_rows
    ]
    if missing_in_load:
        raise Refused(
            f"load {load_name!r} frames {load_first}-{load_last} missing row(s) in per_frame.csv or frames.csv: "
            f"{missing_in_load[:10]}"
        )

    # Read part_grades.json (unbridged per-frame records; manifest.json's frame_spans joins spans <= 2 frames apart,
    # and part_sizes.emptied_by_the_floor is also bridged via frame_spans so we never read it for per-frame causes)
    grades_path = subj_dir / "part_grades.json"
    has_grades_file = grades_path.is_file()
    grades_by_sf: dict[int, dict] = {}
    if has_grades_file:
        gdata = json.loads(grades_path.read_text())
        for g in gdata.get("frames", []):
            sf_g = int(g["source_frame"])
            did_val = g.get("did")
            if did_val not in CLOSED_DID_VOCAB:
                raise Refused(
                    f"frame {sf_g}: part_grades.json has unknown did={did_val!r} "
                    f"(expected one of {sorted(CLOSED_DID_VOCAB)})"
                )
            grades_by_sf[sf_g] = g

    cap_dropped_sfs = _expand_spans(sighting.get("parts_emptied_on_source_frames"))
    no_mask_sfs = _expand_spans(run_entry.get("left_as_the_source_with_no_mask"))

    # Read drop= and drop_by from job (drop_by only, never falling back to note)
    subj_job_cfg = next((s for s in (job.get("subjects") or []) if s.get("label") == subject), {}) if job else {}
    if job is not None and ("drop" in subj_job_cfg or "parts_emptied_on_source_frames" in sighting):
        job_dropped_sfs = _parse_drop_spans(subj_job_cfg.get("drop"))
        if job_dropped_sfs != cap_dropped_sfs:
            raise Refused(
                f"job drop={_to_spans(job_dropped_sfs)} and capture parts_emptied_on_source_frames={_to_spans(cap_dropped_sfs)} "
                f"disagree for subject {subject!r}"
            )
    dropped_sfs = cap_dropped_sfs
    drop_reason = str(subj_job_cfg.get("drop_by") or "").strip()
    if dropped_sfs and not drop_reason:
        raise Refused(
            f"frames {sorted(dropped_sfs)} are dropped but no drop reason (`drop_by`) is on subject {subject!r} in the job"
        )

    left_out_reasons: dict[int, str] = {}
    for lo in (job.get("left_out") or []) if job else []:
        if lo.get("subject") in (None, subject):
            lo_first = int(lo["first"])
            lo_frames = int(lo["frames"])
            why = str(lo.get("why") or "").strip()
            for sf in range(lo_first, lo_first + lo_frames):
                left_out_reasons[sf] = why

    # Read flags.json (bridged spans with join=2)
    flags_path = capture_dir / "flags.json"
    header_flag_spans: list[dict] = []
    inside_flags_by_sf: dict[int, list[str]] = {}
    if flags_path.is_file():
        fdata = json.loads(flags_path.read_text())
        for fl in fdata.get("flags", []):
            fid = fl.get("id")
            if not fid:
                continue
            if fl.get("subject") not in (None, subject):
                continue
            raw_spans = fl.get("source_frames") or []
            expanded = _expand_spans(raw_spans)
            header_flag_spans.append({
                "id": fid,
                "rule": fl.get("rule"),
                "source_frames": _to_spans(expanded),
                "note": "spans from flags.json (written by frame_spans with join=2, which bridges gaps of <= 2 frames)",
            })
            tag = f"inside the span of {fid}"
            for sf in expanded:
                if tag not in inside_flags_by_sf.setdefault(sf, []):
                    inside_flags_by_sf[sf].append(tag)

    owner_presence, _, shots_path = _read_owner_shot_presence(
        capture_dir, manifest, job, subject, load_first, load_last
    )

    # All source frames in the table: union of load frames, capture frames, and left_out frames for this subject
    all_sfs = sorted(
        set(range(load_first, load_last + 1))
        | set(range(cap_first, cap_last + 1))
        | set(left_out_reasons.keys())
    )

    rows: list[dict] = []
    in_view_sfs: list[int] = []
    out_of_view_sfs: list[int] = []
    outcome_counts: dict[str, int] = {k: 0 for k in OUTCOMES}

    for sf in all_sfs:
        in_load = load_first <= sf <= load_last
        in_cap = cap_first <= sf <= cap_last
        load_frame = (sf - load_first) if in_load else None
        cap_idx = (sf - cap_first) if in_cap else None

        pf = per_frame_rows.get(sf)
        fr = frames_rows.get(sf)
        g = grades_by_sf.get(sf)

        # Tracked & SAM fields (from shots__<by>.json frames_without_subject / frames_on_others and track_px, never detection_threshold)
        sam_score = sam_scores.get(sf)
        if pf is not None:
            t_share_raw = pf.get("track_share", "")
            if t_share_raw == "":
                raise Refused(
                    f"frame {sf}: track_share is empty in {per_frame_csv} for subject {subject!r}"
                )
            t_share = float(t_share_raw)
            t_px = int(npz_track_px.get(sf, round(t_share * canvas_area)))
            if t_share > 0.0 or t_px > 0:
                tracked: dict | bool = {"share": t_share, "px": t_px}
                tracked_why = None
            else:
                tracked = False
                tracked_why = (
                    f"track_share is 0 in per_frame.csv (SAM 3.1 track_score={sam_score})"
                    if sam_score is not None
                    else "track_share is 0 in per_frame.csv"
                )
            if sf in without_subject_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:frames_without_subject"
            elif sf in on_others_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:frames_on_others"
            elif sf in offscreen_shot_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:on_screen=false"
            elif t_px > 0:
                in_view = True
                in_view_from = f"per_frame.csv:track_px={t_px} (and shots__{by}.json:with_subject)"
            else:
                in_view = False
                in_view_from = "per_frame.csv:track_px=0"
            sam_info = {
                "track_score": sam_score,
                "track_px": t_px,
                "in_view": in_view,
                "in_view_from": in_view_from,
            }
        else:
            tracked = False
            if sf in without_subject_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:frames_without_subject"
            elif sf in on_others_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:frames_on_others"
            elif sf in offscreen_shot_sfs:
                in_view = False
                in_view_from = f"shots__{by}.json:on_screen=false"
            elif sf in shot_covered_sfs:
                in_view = True
                in_view_from = f"shots__{by}.json:with_subject"
            else:
                in_view = False
                in_view_from = f"not in shots__{by}.json"
            tracked_why = (
                f"frame not in capture (not_in_load); preview SAM 3.1 track_score={sam_score} "
                f"({'in_view' if in_view else 'out_of_view'} via {in_view_from})"
            )
            sam_info = {
                "track_score": sam_score,
                "track_px": None,
                "in_view": in_view,
                "in_view_from": in_view_from,
            }

        if in_view:
            in_view_sfs.append(sf)
        else:
            out_of_view_sfs.append(sf)

        # Saved part px & height (before fill/drop)
        if pf is not None:
            p_share_str = pf.get("parts_share", "")
            if not is_whole_person_load and p_share_str == "":
                raise Refused(
                    f"frame {sf}: parts_share is empty in {per_frame_csv} for part load subject {subject!r}"
                )
            p_share = float(p_share_str) if p_share_str != "" else 0.0
            saved_part_px: int | None = int(npz_part_px.get(sf, round(p_share * canvas_area)))
            ph_str = pf.get("parts_height_px", "")
            if not is_whole_person_load and saved_part_px > 0 and ph_str == "":
                raise Refused(
                    f"frame {sf}: parts_height_px is empty in {per_frame_csv} when saved_part_px={saved_part_px} > 0"
                )
            saved_part_height_px: int | None = int(ph_str) if ph_str != "" else (0 if saved_part_px == 0 else None)
        else:
            saved_part_px = None
            saved_part_height_px = None

        # Outcome & size columns (from closed `did` vocabulary in part_grades.json)
        outcome_extra: dict = {}
        loaded_part_px: int | None
        part_px: int | None
        part_px_measures: str
        part_height_px: int | None = saved_part_height_px
        part_height_px_measures = "saved_part_before_fill" if in_cap else "not_in_load"

        if not in_load:
            outcome = "not_in_load"
            loaded_part_px = None
            part_px = saved_part_px if in_cap else None
            part_px_measures = "not_in_load"
            if sf in left_out_reasons and left_out_reasons[sf]:
                outcome_extra["reason"] = left_out_reasons[sf]
        elif sf in dropped_sfs:
            outcome = "dropped"
            loaded_part_px = 0
            part_px = saved_part_px
            part_px_measures = "removed_part (dropped, carried=0)"
            outcome_extra["reason"] = drop_reason
        elif is_whole_person_load:
            t_px_val = tracked["px"] if isinstance(tracked, dict) else 0
            if t_px_val > 0 and sf not in no_mask_sfs:
                outcome = "replaced"
                loaded_part_px = t_px_val
                part_px = t_px_val
                part_px_measures = "whole_person_track (carried)"
            else:
                outcome = "emptied"
                outcome_extra["cause"] = "nothing saved"
                loaded_part_px = 0
                part_px = 0
                part_px_measures = "removed_part (carried=0)"
        elif not has_grades_file:
            outcome = "no per-frame record"
            loaded_part_px = None
            part_px = saved_part_px
            part_px_measures = "saved_part (no_per_frame_record)"
            outcome_extra["reason"] = (
                f"subjects/{subject}/part_grades.json is absent; manifest.json spans are bridged (join=2) and not per-frame exact"
            )
        elif g is not None:
            did = g["did"]
            if did in HELD_DIDS:
                outcome = "held"
                from_frames_raw = g.get("from_frames")
                if isinstance(from_frames_raw, list) and from_frames_raw:
                    from_sfs = [cap_first + int(x) for x in from_frames_raw]
                    outcome_extra["from_frame"] = from_sfs[0] if len(from_sfs) == 1 else from_sfs
                else:
                    outcome_extra["from_frame"] = None
                if g.get("px") is not None:
                    held_px = int(g["px"])
                    loaded_part_px = held_px
                    part_px = held_px
                    part_px_measures = "held_fill (carried)"
                else:
                    loaded_part_px = None
                    part_px = None
                    part_px_measures = f"not in the record (did={did!r} in part_grades.json writes no held fill px)"
                if g.get("px_on_the_track") is not None:
                    outcome_extra["px_on_the_track"] = int(g["px_on_the_track"])
                if g.get("source_px") is not None:
                    outcome_extra["source_px"] = int(g["source_px"])
                if g.get("face_points_inside") is not None:
                    outcome_extra["face_points_inside"] = g["face_points_inside"]
                outcome_extra["did"] = did
            elif did == "saved part kept, fill refused":
                outcome = "replaced"
                loaded_part_px = int(saved_part_px or 0)
                part_px = int(saved_part_px or 0)
                part_px_measures = "saved_part (carried)"
                outcome_extra["did"] = did
            elif did in EMPTIED_DIDS:
                outcome = "emptied"
                loaded_part_px = 0
                # Only "least_px" or "collapsed_against_px" in the per-frame entry marks the post-grade floor pass (never bridged part_sizes)
                is_floor = "least_px" in g or "collapsed_against_px" in g
                if is_floor:
                    outcome_extra["cause"] = "floor"
                elif g.get("saved_on_its_classes") is None and (saved_part_px or 0) == 0:
                    outcome_extra["cause"] = "nothing saved"
                else:
                    outcome_extra["cause"] = "grade"
                removed_px = int(g["px"]) if g.get("px") is not None else int(saved_part_px or 0)
                if outcome_extra["cause"] == "nothing saved":
                    removed_px = 0
                part_px = removed_px
                part_px_measures = "removed_part (carried=0)"
                outcome_extra["did"] = did
                if g.get("why"):
                    outcome_extra["why"] = g["why"]
                if g.get("word_not_held"):
                    outcome_extra["word_not_held"] = g["word_not_held"]
            else:
                raise Refused(f"frame {sf}: unhandled did={did!r}")
        else:
            # Frame not listed in part_grades.json["frames"]: untouched by grade_parts (not offered & saved >= HOLD_INSIDE, not < least_px)
            if (saved_part_px or 0) == 0 or sf in no_mask_sfs:
                outcome = "emptied"
                outcome_extra["cause"] = "nothing saved"
                loaded_part_px = 0
                part_px = 0
                part_px_measures = "removed_part (carried=0)"
            else:
                outcome = "replaced"
                loaded_part_px = int(saved_part_px or 0)
                part_px = int(saved_part_px or 0)
                part_px_measures = "saved_part (carried)"

        if outcome not in OUTCOMES:
            raise Refused(f"frame {sf}: invalid outcome {outcome!r}")
        if outcome == "emptied" and outcome_extra.get("cause") not in EMPTIED_CAUSES:
            raise Refused(f"frame {sf}: invalid emptied cause {outcome_extra.get('cause')!r}")
        outcome_counts[outcome] += 1

        # Masks provenance for this frame
        masks_info = {
            "track": {
                "file": _rel(track_mask_path),
                "frame_index": cap_idx,
                "sha256": track_sha,
            },
            "part": (
                {
                    "file": _rel(part_mask_path),
                    "source": part_mask_source,
                    "frame_index": cap_idx,
                    "sha256": part_sha,
                }
                if part_mask_path is not None
                else {"file": None, "source": part_mask_source, "frame_index": cap_idx, "sha256": None}
            ),
        }

        # Others given back from frames.csv
        others_given_back: dict[str, dict] = {}
        if fr is not None:
            for other in others:
                r_on = fr.get(f"region_on__{load_name}__{other}", "")
                r_cells = fr.get(f"region_cells_holding__{load_name}__{other}", "")
                m_reach = fr.get(f"margin_reached__{load_name}__{other}", "")
                c_back = fr.get(f"cells_given_back__{load_name}__{other}", "")
                if any(x != "" for x in (r_on, r_cells, m_reach, c_back)):
                    others_given_back[other] = {
                        "region_on": float(r_on) if r_on != "" else 0.0,
                        "region_cells_holding": int(r_cells) if r_cells != "" else 0,
                        "margin_reached": float(m_reach) if m_reach != "" else 0.0,
                        "cells_given_back": int(c_back) if c_back != "" else 0,
                    }

        row_dict: dict = {
            "source_frame": sf,
            "load_frame": load_frame,
            "tracked": tracked,
        }
        if tracked is False and tracked_why:
            row_dict["tracked_why"] = tracked_why
        row_dict["loaded_part_px"] = loaded_part_px
        row_dict["saved_part_px"] = saved_part_px
        row_dict["saved_part_height_px"] = saved_part_height_px
        row_dict["part_px"] = part_px
        row_dict["part_px_measures"] = part_px_measures
        row_dict["part_height_px"] = part_height_px
        row_dict["part_height_px_measures"] = part_height_px_measures
        row_dict["outcome"] = outcome
        row_dict.update(outcome_extra)
        row_dict["masks"] = masks_info
        row_dict["others_given_back"] = others_given_back
        row_dict["inside_flag_spans"] = inside_flags_by_sf.get(sf, [])
        row_dict["sam"] = sam_info
        rows.append(row_dict)

    # Check that every frame of the load produced a row
    row_sfs = {r["source_frame"] for r in rows}
    for sf in range(load_first, load_last + 1):
        if sf not in row_sfs:
            raise Refused(f"frame {sf} of load {load_name!r} has no row in the frame table")

    in_view_spans = _to_spans(in_view_sfs)
    out_of_view_spans = _to_spans(out_of_view_sfs)
    owner_empty_spans = out_of_view_spans if owner_presence.get("present") == "yes" else []

    # Check if this load is a pair load in the job file
    pair_subjects = [s.get("subject") for s in (load_cfg.get("subjects") or []) if s.get("subject")] if load_cfg else []
    also_subjects = [s for s in pair_subjects if s != subject]
    pair_info: dict | None = None
    if also_subjects:
        also_sub = also_subjects[0]
        has_also_sighting = any(s.get("label") == also_sub and s.get("sightings") for s in manifest.get("subjects", []))
        also_stem = "frame_table" if also_sub == primary_subject else f"frame_table__{also_sub}"
        also_table_name = f"{also_stem}.jsonl"
        pair_info = {
            "also_subject": also_sub,
            "table_file": also_table_name if has_also_sighting else None,
            "note": (
                f"pair load's other subject {also_sub!r} has its own table {also_table_name}"
                if has_also_sighting
                else f"pair load's second subject {also_sub!r} has no sighting in manifest.json"
            ),
        }

    input_sha256 = {
        "manifest": {"file": _rel(manifest_path), "sha256": _sha256_once(manifest_path, sha_cache)},
        "part_grades": (
            {"file": _rel(grades_path), "sha256": _sha256_once(grades_path, sha_cache)}
            if has_grades_file
            else None
        ),
        "flags": (
            {"file": _rel(flags_path), "sha256": _sha256_once(flags_path, sha_cache)}
            if flags_path.is_file()
            else None
        ),
        "job": (
            {"file": _rel(job_file), "sha256": _sha256_once(job_file, sha_cache)}
            if job_file and job_file.is_file()
            else None
        ),
        "shots": {"file": _rel(shots_path), "sha256": _sha256_once(shots_path, sha_cache)},
    }

    header = {
        "schema": SCHEMA,
        "subject": subject,
        "capture": _rel(capture_dir),
        "job_file": _rel(job_file) if job_file else None,
        "input_sha256": input_sha256,
        "load": {
            "name": load_name,
            "subject": subject,
            "others": others,
            "first": load_first,
            "last": load_last,
            "frames": load_frames,
            "pair": pair_info,
        },
        "size_columns_are": {
            "loaded_part_px": "pixels carried into the load on this frame (saved_part_px on replaced, held_fill px on held, 0 on emptied and dropped, null on not_in_load)",
            "saved_part_px": "pixels in the saved part mask before grading/holding/dropping (from masks__<by>.npz or per_frame.csv:parts_share)",
            "saved_part_height_px": "bounding-box height of the saved part mask before fill (from per_frame.csv:parts_height_px)",
            "part_px": "outcome-specific pixel count paired with part_px_measures (saved_part on replaced, held_fill on held, removed_part on emptied and dropped)",
            "part_height_px": "saved_part_height_px (paired with part_height_px_measures='saved_part_before_fill')",
        },
        "sam_inputs": sam_inputs,
        "in_view_spans": in_view_spans,
        "out_of_view_spans": out_of_view_spans,
        "in_the_shot_by_the_owner": owner_presence,
        "owner_says_in_shot_track_empty": owner_empty_spans,
        "flag_spans": header_flag_spans,
        "outcome_counts": outcome_counts,
        "outcome_spans": {
            k: _to_spans([r["source_frame"] for r in rows if r["outcome"] == k])
            for k in OUTCOMES
        },
    }

    target_dir = Path(out_dir) if out_dir is not None else capture_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = "frame_table" if subject == primary_subject else f"frame_table__{subject}"
    jsonl_path = target_dir / f"{stem}.jsonl"
    md_path = target_dir / f"{stem}.md"

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md_path.write_text(_render_markdown(header, rows))
    return {"jsonl": jsonl_path, "md": md_path}


def _resolve_recorded_path(rel_or_abs: str | None, capture_dir: Path) -> Path | None:
    if not rel_or_abs:
        return None
    cand = Path(rel_or_abs)
    if cand.is_absolute():
        return cand.resolve() if cand.is_file() else None
    for base in (capture_dir.parent.parent, REPO):
        if (base / cand).is_file():
            return (base / cand).resolve()
    return None


def verify_table_fresh(
    capture_dir: Path,
    load: dict,
    *,
    job_file: Path | None = None,
    by: str = "<session>",
) -> dict[str, str]:
    """Verify that every replaced subject's frame_table in capture_dir exists, matches the load, and has fresh input_sha256.

    Returns a mapping {subject: sha256_of_table_file} or raises Refused with the exact command to rebuild.
    """
    sha_cache: dict[Path, str] = {}
    capture_dir = Path(capture_dir).resolve()
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise Refused(f"manifest.json is missing in {capture_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs") or []
    explicit_primary = (
        load.get("subject")
        or (runs[0].get("subject") if runs and runs[0].get("subject") else None)
    )
    primary_subject = (
        explicit_primary
        or (manifest.get("subjects")[0]["label"] if manifest.get("subjects") else None)
    )
    pair_entries = [
        s for s in (load.get("subjects") or []) if isinstance(s, dict) and s.get("subject")
    ]
    tables_to_check: list[tuple[str, str, bool]] = []
    if pair_entries:
        if not explicit_primary and pair_entries[0].get("subject"):
            explicit_primary = pair_entries[0]["subject"]
            primary_subject = explicit_primary
        for idx, entry in enumerate(pair_entries):
            sub = entry["subject"]
            has_sighting = any(
                ms.get("label") == sub and ms.get("sightings")
                for ms in (manifest.get("subjects") or [])
            )
            if has_sighting or idx == 0:
                stem = "frame_table" if idx == 0 else f"frame_table__{sub}"
                tables_to_check.append((sub, stem, True))
    elif primary_subject:
        tables_to_check.append((primary_subject, "frame_table", explicit_primary is not None))
    else:
        raise Refused(f"load {load.get('name')!r} names no subject")

    also_cfg = load.get("also")
    if isinstance(also_cfg, dict) and also_cfg.get("subject"):
        also_sub = also_cfg["subject"]
        also_stem = f"frame_table__{also_sub}"
        if not any(st == also_stem for _, st, _ in tables_to_check):
            tables_to_check.append((also_sub, also_stem, True))

    load_name = load["name"]
    has_bounds = load.get("first") is not None and load.get("frames") is not None
    load_first = int(load["first"]) if has_bounds else None
    load_frames = int(load["frames"]) if has_bounds else None
    load_last = (load_first + load_frames - 1) if (load_first is not None and load_frames is not None) else None

    rebuild_cmd = (
        f"python bench/load_chain.py table {job_file} --by {by}"
        if job_file is not None
        else f"python bench/load_chain.py table <job.json> --by {by} (or python bench/frame_table.py {capture_dir})"
    )

    table_shas: dict[str, str] = {}
    for sub, stem, is_explicit in tables_to_check:
        ft_path = capture_dir / f"{stem}.jsonl"
        if not ft_path.is_file():
            raise Refused(
                f"{ft_path.name} is missing in {capture_dir} for subject {sub!r} — "
                f"build it first with:\n  {rebuild_cmd}"
            )
        lines = [ln for ln in ft_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise Refused(f"{ft_path.name} is empty — rebuild with:\n  {rebuild_cmd}")
        try:
            hdr = json.loads(lines[0])
            rows = [json.loads(ln) for ln in lines[1:]]
        except Exception as exc:
            raise Refused(f"{ft_path.name} is not valid JSONL ({exc}) — rebuild with:\n  {rebuild_cmd}") from exc

        if hdr.get("schema") not in (SCHEMA, None) or (
            hdr.get("schema") is None and hdr.get("frame_table_schema") != 1
        ):
            raise Refused(
                f"{ft_path.name} has schema {hdr.get('schema')!r}, expected {SCHEMA!r} — "
                f"rebuild with:\n  {rebuild_cmd}"
            )
        hdr_load = hdr.get("load") or {}
        hdr_sub = hdr.get("subject") or hdr_load.get("subject")
        expected_sub = sub if is_explicit else (hdr_sub or sub)
        if hdr_sub != expected_sub:
            raise Refused(
                f"{ft_path.name} has subject={hdr_sub!r}, expected {expected_sub!r} — "
                f"rebuild with:\n  {rebuild_cmd}"
            )
        if hdr_load and has_bounds:
            if (
                hdr_load.get("name") != load_name
                or hdr_load.get("first") != load_first
                or hdr_load.get("last") != load_last
                or hdr_load.get("frames") != load_frames
            ):
                raise Refused(
                    f"{ft_path.name} header load {hdr_load} does not match "
                    f"load {load_name!r} ({load_first}..{load_last}, {load_frames} frames) — "
                    f"rebuild with:\n  {rebuild_cmd}"
                )

        in_sha = hdr.get("input_sha256")
        if not isinstance(in_sha, dict):
            raise Refused(
                f"{ft_path.name} header lacks input_sha256 — rebuild with:\n  {rebuild_cmd}"
            )

        def _rec_sha(key: str) -> tuple[str | None, str | None]:
            val = in_sha.get(key)
            if isinstance(val, dict):
                return val.get("sha256"), val.get("file")
            if isinstance(val, str):
                return val, None
            return None, None

        grades_path = capture_dir / "subjects" / sub / "part_grades.json"
        flags_path = capture_dir / "flags.json"
        rec_job_sha, rec_job_file = _rec_sha("job")
        resolved_job = (
            Path(job_file).resolve()
            if job_file is not None and Path(job_file).is_file()
            else _resolve_recorded_path(rec_job_file or hdr.get("job_file"), capture_dir)
        )
        rec_shots_sha, rec_shots_file = _rec_sha("shots")
        shots_rel = rec_shots_file or (hdr.get("in_the_shot_by_the_owner") or {}).get("shots_file")
        shots_path = _resolve_recorded_path(shots_rel, capture_dir)

        expected_actual: list[tuple[str, str | None, str | None]] = [
            ("manifest", _rec_sha("manifest")[0], _sha256_once(manifest_path, sha_cache)),
            (
                "part_grades",
                _rec_sha("part_grades")[0],
                _sha256_once(grades_path, sha_cache) if grades_path.is_file() else None,
            ),
            (
                "flags",
                _rec_sha("flags")[0],
                _sha256_once(flags_path, sha_cache) if flags_path.is_file() else None,
            ),
        ]
        if resolved_job is not None or rec_job_sha is not None:
            expected_actual.append(
                (
                    "job",
                    rec_job_sha,
                    _sha256_once(resolved_job, sha_cache) if resolved_job and resolved_job.is_file() else None,
                )
            )
        if shots_path is not None or rec_shots_sha is not None:
            expected_actual.append(
                (
                    "shots",
                    rec_shots_sha,
                    _sha256_once(shots_path, sha_cache) if shots_path and shots_path.is_file() else None,
                )
            )

        stale = [k for k, rec_s, cur_s in expected_actual if rec_s != cur_s]
        if stale:
            raise Refused(
                f"{ft_path.name} is stale: {', '.join(stale)} changed after the table was built — "
                f"rebuild with:\n  {rebuild_cmd}"
            )

        if rows and has_bounds and load_first is not None and load_last is not None:
            row_sfs = {r.get("source_frame") for r in rows if isinstance(r, dict)}
            missing = [sf for sf in range(load_first, load_last + 1) if sf not in row_sfs]
            if missing:
                raise Refused(
                    f"{ft_path.name} is missing row(s) for load {load_name!r} frames {missing[:10]} — "
                    f"rebuild with:\n  {rebuild_cmd}"
                )
        table_shas[sub] = _sha256_once(ft_path, sha_cache)

    return table_shas


def _fmt_spans(spans: list[list[int]]) -> str:
    if not spans:
        return "none"
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def _render_markdown(header: dict, rows: list[dict]) -> str:
    ld = header["load"]
    sam = header["sam_inputs"]
    who = sam["who_is_who"]
    own = header["in_the_shot_by_the_owner"]
    lines = [
        f"# Frame table: `{ld['name']}` (`{ld['subject']}`, source frames {ld['first']}–{ld['last']})",
        "",
        f"- **Capture**: `{header['capture']}`",
        f"- **Job file**: `{header['job_file']}`",
        f"- **Load span**: `{ld['first']}..{ld['last']}` ({ld['frames']} frames); table rows `{rows[0]['source_frame']}..{rows[-1]['source_frame']}` ({len(rows)} frames)",
    ]
    if ld.get("pair"):
        lines.append(f"- **Pair load**: {ld['pair']['note']}")
    lines.extend([
        "",
        "## Owner shot presence & SAM 3.1 visibility",
        "",
        f"- **`owner_says_in_shot_track_empty`**: `{_fmt_spans(header['owner_says_in_shot_track_empty'])}`",
        f"- **`in_the_shot_by_the_owner`**: `{own.get('present')}` (shot `{own.get('shot_id')}` `{own.get('shot_span')}`, how=`{own.get('how')}`, by=`{own.get('by')}`, at=`{own.get('at')}`, frames_boxed=`{own.get('frames_boxed')}`, from `{own.get('shots_file')}`)",
        f"- **`in_view_spans`** (SAM 3.1 track present in `shots__<by>.json` & `per_frame.csv`): `{_fmt_spans(header['in_view_spans'])}`",
        f"- **`out_of_view_spans`** (`frames_without_subject`, `frames_on_others`, or `track_px == 0`): `{_fmt_spans(header['out_of_view_spans'])}`",
        "",
        "## SAM 3.1 inputs (`sam_inputs`)",
        "",
        f"- **Preview**: `{sam['preview']}` (`{sam['preview_graph']}`, node `{sam.get('track_node_id')}`, `shots_first_source_frame={sam['shots_first_source_frame']}`)",
        f"- **Model file**: `{sam['model_file']}` (`h3_config.SEGMENTER`: `{sam['h3_config_segmenter']}`)",
        f"- **SAM 3.1 corrections node**: `{json.dumps(sam.get('sam31_corrections'))}` | **others wired**: `{json.dumps(sam.get('others_wired'))}`",
        f"- **Subject phrase**: `{sam['subject_phrase']!r}` | **head_phrase**: `{sam['head_phrase']!r}`",
        f"- **Pick**: `{sam['pick']!r}` (`pick_on={sam['pick_on']!r}`, `match={sam['match']}`, `detection_threshold={sam['detection_threshold']}`, `max_people={sam['max_people']}`)",
        f"- **Cuts**: `{sam['cuts']!r}`",
        f"- **Seeds**: `{sam['seeds_raw']!r}` -> `{json.dumps(sam['seeds'])}`",
        f"- **Corrections**: `{sam['corrections_raw']!r}` -> `{json.dumps(sam['corrections'])}`",
        f"- **Who is who**: tracked person `{who.get('tracked_person_id')}`, called `{who.get('called')}`, answered_by `{who.get('answered_by')}`, on whose word: `{who.get('on_whose_word')}` (`{who.get('file')}`)",
        "",
        "## Outcomes summary",
        "",
    ])
    for k in OUTCOMES:
        cnt = header["outcome_counts"].get(k, 0)
        if cnt == 0 and k == "no per-frame record":
            continue
        sp = _fmt_spans(header["outcome_spans"].get(k, []))
        lines.append(f"- **`{k}`**: {cnt} frame(s) ({sp})")

    lines.extend([
        "",
        "## Gate flag spans (`flags.json`, bridged with `join=2`)",
        "",
    ])
    if header.get("flag_spans"):
        for fl in header["flag_spans"]:
            lines.append(f"- **`{fl['id']}`** (`{fl.get('rule')}`): `{_fmt_spans(fl['source_frames'])}`")
    else:
        lines.append("- none")

    if rows:
        m0 = rows[0]["masks"]
        lines.extend([
            "",
            "## Mask files",
            "",
            f"- **Track mask**: `{m0['track']['file']}` (`sha256={m0['track']['sha256']}`)",
            f"- **Part mask**: `{m0['part']['file']}` (`source={m0['part'].get('source')}`, `sha256={m0['part']['sha256']}`)",
        ])

    lines.extend([
        "",
        "## Per-frame table",
        "",
        "- **Size columns**: `loaded_part_px` is what is carried into the load (`saved_part_px` on `replaced`, `held_fill` on `held`, `0` on `emptied`/`dropped`); `saved_part_px` and `saved_part_height_px` are the saved part before grading/holding/dropping; `part_px (measured)` shows `part_px` with what it measures (`saved_part`, `held_fill`, or `removed_part`).",
        "",
        "| source_frame | load_frame | tracked (share / px) | loaded_part_px | part_px (measured) | saved_part_height_px | outcome | detail | sam (score / in_view / from) | others_given_back | inside_flag_spans |",
        "|---:|---:|---|---:|---|---:|---|---|---|---|---|",
    ])
    for r in rows:
        tr = r["tracked"]
        tr_str = f"{tr['share']:.5f} ({tr['px']} px)" if isinstance(tr, dict) else f"false ({r.get('tracked_why', '')})"
        detail_parts = []
        if r["outcome"] == "held":
            detail_parts.append(f"from={r.get('from_frame')}")
            if "px_on_the_track" in r:
                detail_parts.append(f"on_track={r['px_on_the_track']}px")
            if r.get("saved_part_px") is not None:
                detail_parts.append(f"saved={r['saved_part_px']}px")
        elif r["outcome"] == "dropped":
            detail_parts.append(str(r.get("reason", "")))
        elif r["outcome"] == "emptied":
            detail_parts.append(f"cause={r.get('cause')}")
            if r.get("word_not_held"):
                detail_parts.append(str(r["word_not_held"]))
            elif r.get("why"):
                detail_parts.append(str(r["why"]))
        elif r["outcome"] in ("not_in_load", "no per-frame record") and r.get("reason"):
            detail_parts.append(str(r["reason"]))
        sam_r = r["sam"]
        sc_str = f"{sam_r['track_score']:.2f}" if isinstance(sam_r["track_score"], (int, float)) else str(sam_r["track_score"])
        sam_str = f"{sc_str} / {str(sam_r['in_view']).lower()} ({sam_r.get('in_view_from', '')})"
        others_str = ", ".join(
            f"{o}: cells_back={v['cells_given_back']}, reg={v['region_on']}"
            for o, v in r["others_given_back"].items()
            if v["cells_given_back"] or v["region_on"] or v["margin_reached"]
        ) or "-"
        flags_str = ", ".join(r["inside_flag_spans"]) or "-"
        loaded_str = str(r["loaded_part_px"]) if r["loaded_part_px"] is not None else "-"
        part_px_str = f"{r['part_px']} ({r['part_px_measures']})" if r["part_px"] is not None else "-"
        ph_str = f"{r['saved_part_height_px']} (saved)" if r["saved_part_height_px"] is not None else "-"
        lines.append(
            f"| {r['source_frame']} | {r['load_frame'] if r['load_frame'] is not None else '-'} | {tr_str} | "
            f"{loaded_str} | {part_px_str} | {ph_str} | "
            f"`{r['outcome']}` | {'; '.join(detail_parts) or '-'} | {sam_str} | {others_str} | {flags_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture", type=Path, help="the load's capture folder (data/<date>_<load>)")
    p.add_argument("--job", type=Path, default=None, help="the load's job file (found from <capture>_job/ when omitted)")
    p.add_argument("--subject", default=None, help="build the table for this subject label (defaults to the run's primary subject)")
    p.add_argument("--out-dir", type=Path, default=None, help="write frame_table.jsonl and frame_table.md here instead of inside the capture folder")
    a = p.parse_args(argv)
    try:
        wrote = build_frame_table(a.capture, job_path=a.job, out_dir=a.out_dir, subject_arg=a.subject)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"wrote {wrote['jsonl']} and {wrote['md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
