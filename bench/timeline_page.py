#!/usr/bin/env python3
"""Build the whole-video NLE timeline data file (`--out-json`) and interactive page (`--out-html`) from the rebuild's records.

    <python> bench/timeline_page.py --video <stem> --tables <delivery_tables_dir> --stamp <YYYY-MM-DDTHHMMZ> [--data data] [--share <output_dir>] --out-json <timeline_json> --out-html <TIMELINE.html>

**What it buys.** Gives the owner, Claude, and Gemini one holistic whole-video view across all frames (`0..N`), one lane
per subject, showing at every frame what stands there (which piece, its render day and age tier `pre-10/20`, `10/20-10/22`,
or `10/23+`, and whether it rendered zoomed or whole-frame), what a look left out (`left_out_by_a_look`), what is owed and
why (`no_capture`, `captured_not_rendered`, `rendered_not_kept`, `emptied_inside`, `laid_but_barely_changed`,
`region_far_below_its_own_size`), where two pieces of one subject meet inside a shot (`seams`), and every note or verdict
the owner has recorded on that shot or frame (`owner_answers.jsonl` and `verdicts.jsonl`, via `bench/said_on_shot.py`).
Notes pinned in `TIMELINE.html` append directly to `owner_answers.jsonl` through `serve.py`'s `POST /api/answer` (`row:
"timeline"`, `what: "timeline_note"`), preserving the one home for the owner's words.

**What it cannot see.** The picture itself (`bench/frame_coverage.py::CANNOT`): whether a face shows on a frame or whether
a laid piece looks right on playback. Colors represent record state and render age tier as data, never a pass/fail verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frame_coverage as fc  # noqa: E402
import said_on_shot as sos  # noqa: E402


class Refused(Exception):
    """A missing, malformed, or mismatched input record refused in the tool's own words with no traceback."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, label: str):
    if not path.is_file():
        raise Refused(f"REFUSED: {label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Refused(f"REFUSED: {label} is not valid JSON ({path}): {e}") from None


_HOST_PATH_RE = re.compile(
    r"(?:"
    + "/" + "home" + r"/[A-Za-z0-9._-]+/"
    + r"|"
    + "/" + "mnt" + r"/[A-Za-z0-9._-]+/"
    + r"|"
    + "/" + "Users" + r"/[A-Za-z0-9._-]+/"
    + r")[^\s\"')\]>]*"
)


def _scrub_rel_path(raw: str | None, video: str) -> str | None:
    """Strip any machine-specific leading path before `Video/<video>/` or `data/<video>/`."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    for marker in (f"Video/{video}/", f"data/{video}/", "Video/", "data/"):
        idx = s.find(marker)
        if idx >= 0:
            return s[idx:]
    # If it is an absolute path outside Video/ or data/, keep only the basename
    if s.startswith("/"):
        return Path(s).name
    return _scrub_text(s, video)


def _scrub_text(text: str, video: str) -> str:
    """Scrub any embedded machine-owner or mount path substrings inside free text."""
    return _HOST_PATH_RE.sub(
        lambda m: _scrub_rel_path(m.group(0), video) or Path(m.group(0)).name,
        text,
    )


def _scrub_obj(val, video: str):
    """Recursively scrub machine-specific paths from strings in dicts, lists, and tuples."""
    if isinstance(val, str):
        return _scrub_text(val, video)
    if isinstance(val, list):
        return [_scrub_obj(x, video) for x in val]
    if isinstance(val, tuple):
        return [_scrub_obj(x, video) for x in val]
    if isinstance(val, dict):
        return {k: _scrub_obj(v, video) for k, v in val.items()}
    return val


def _page_media_url(rel_path: str | None, video: str, out_html_dir: Path) -> str | None:
    """Convert a repo/share-relative media path (`Video/<video>/...` or `data/<video>/...`) into a relative URL from `out_html_dir`."""
    clean = _scrub_rel_path(rel_path, video)
    if not clean:
        return None
    prefix_video = f"Video/{video}/"
    prefix_data_poc = f"data/{video}/poc/"
    prefix_data = f"data/{video}/"
    use_share_link = out_html_dir.name == "poc" or (out_html_dir / f"{video}_share").exists()
    if clean.startswith(prefix_video):
        rest = clean[len(prefix_video):]
        return f"{video}_share/{rest}" if use_share_link else rest
    if clean.startswith(prefix_data_poc):
        rest = clean[len(prefix_data_poc):]
        return rest if use_share_link else f"poc/{rest}"
    if clean.startswith(prefix_data):
        rest = clean[len(prefix_data):]
        return rest
    return clean


def classify_age_tier(render_day: str | None) -> str | None:
    """Classify a `MM-DD` (or `MM/DD`) render day into one of the three timeline age tiers."""
    if not render_day or render_day == "?":
        return None
    norm = render_day.strip().replace("/", "-")
    if not re.fullmatch(r"\d{2}-\d{2}", norm):
        return None
    if norm < "10-20":
        return "pre-10/20"
    if norm <= "10-22":
        return "10/20-10/22"
    return "10/23+"


def _extract_table_laid_sha(made_line: str) -> str | None:
    m = re.search(r"laid\.jsonl[^()]*\(sha256\s+([0-9a-f]+)", str(made_line or ""))
    return m.group(1) if m else None


def _extract_out_path_from_cmd(cmd: str) -> str | None:
    m = re.search(r"--out\s+([^\s'\"]+)", str(cmd or ""))
    return m.group(1) if m else None


def _extract_source_path_from_cmd(cmd: str) -> str | None:
    m = re.search(r"--source\s+([^\s'\"]+)", str(cmd or ""))
    return m.group(1) if m else None


def _subject_order(shots: list[dict], plan_j: dict | None) -> list[str]:
    canonical = ["lead", "second", "third"]
    seen: list[str] = []
    if isinstance(plan_j, dict) and isinstance(plan_j.get("subjects"), dict):
        for s in plan_j["subjects"]:
            if s not in seen:
                seen.append(s)
    for r in shots:
        for s in (r.get("who") or {}):
            if s not in seen:
                seen.append(s)
    ordered = [s for s in canonical if s in seen]
    for s in seen:
        if s not in ordered:
            ordered.append(s)
    return ordered or canonical


def build_timeline_model(
    video: str,
    data_dir: Path,
    tables_dir: Path,
    stamp: str,
    out_html_dir: Path,
    share_dir: Path | None = None,
) -> dict:
    if not video or not video.strip():
        raise Refused("REFUSED: --video must be non-empty")
    if not stamp or not stamp.strip():
        raise Refused("REFUSED: --stamp must be non-empty")
    if not tables_dir.is_dir():
        raise Refused(f"REFUSED: --tables directory not found: {tables_dir}")

    vdir = data_dir / video
    shots_path = vdir / "shots.json"
    shots_j = _read_json(shots_path, "shots.json")
    shots_raw = shots_j.get("shots") if isinstance(shots_j, dict) and "shots" in shots_j else shots_j
    shots = [
        r for r in (shots_raw if isinstance(shots_raw, list) else (shots_raw.values() if isinstance(shots_raw, dict) else []))
        if isinstance(r, dict) and "first" in r and "last" in r
    ]
    if not shots:
        raise Refused(f"REFUSED: no shots found in {shots_path}")
    shots.sort(key=lambda r: int(r["first"]))

    laid_path = vdir / "laid.jsonl"
    if not laid_path.is_file():
        raise Refused(f"REFUSED: laid.jsonl not found: {laid_path}")
    laid_sha256 = _sha256_file(laid_path)
    laid_rows: list[dict] = []
    for line_no, raw in enumerate(laid_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as e:
            raise Refused(f"REFUSED: {laid_path} line {line_no} is not valid JSON: {e}") from None
        if isinstance(obj, dict):
            laid_rows.append(obj)

    plan_path = vdir / "plan.json"
    plan_j = _read_json(plan_path, "plan.json") if plan_path.is_file() else {}

    table_matches = sorted(tables_dir.glob(f"t_{video}_*_{stamp}.json"))
    if not table_matches:
        raise Refused(f"REFUSED: no delivery table t_{video}_*_{stamp}.json found in {tables_dir}")
    table_path = table_matches[-1]
    table_sha256 = _sha256_file(table_path)
    table_j = _read_json(table_path, "delivery table JSON")

    ledger_path = tables_dir / f"ledger_{video}_{stamp}.json"
    ledger_j = _read_json(ledger_path, "ledger JSON")
    renders = ledger_j.get("renders") if isinstance(ledger_j, dict) else None
    if not isinstance(renders, dict):
        raise Refused(f"REFUSED: ledger JSON missing 'renders' dict: {ledger_path}")
    renders_by_name = {r["name"]: r for r in renders.values() if isinstance(r, dict) and "name" in r}

    cov_matches = sorted(tables_dir.glob(f"frame_coverage_{video}_*_{stamp}.json"))
    if not cov_matches:
        raise Refused(f"REFUSED: no frame_coverage_{video}_*_{stamp}.json found in {tables_dir}")
    cov_path = cov_matches[-1]
    cov_j = _read_json(cov_path, "frame_coverage JSON")
    if not isinstance(cov_j, dict) or not isinstance(cov_j.get("shots"), list):
        raise Refused(f"REFUSED: frame_coverage JSON missing 'shots' list: {cov_path}")

    cov_table_sha = str(cov_j.get("table_sha256") or "")
    if cov_table_sha and cov_table_sha != table_sha256:
        raise Refused(
            f"REFUSED: frame_coverage {cov_path.name} records table_sha256 {cov_table_sha[:16]}, "
            f"which does not match {table_path.name} ({table_sha256[:16]})"
        )

    table_laid_sha = _extract_table_laid_sha(str(table_j.get("made") or ""))
    laid_matches_table = bool(table_laid_sha and (laid_sha256 == table_laid_sha or laid_sha256.startswith(table_laid_sha)))
    stale_banner = ""
    if not laid_matches_table:
        stale_banner = (
            f"laid.jsonl has moved since rebuild {stamp}: table records sha256 {table_laid_sha or 'unknown'}, "
            f"current laid.jsonl is {laid_sha256[:16]}. Run bench/rebuild_delivery.py --video {video} --clip to refresh."
        )

    # Map review items by id / piece name
    items_path = vdir / "review" / "items.json"
    review_items_by_id: dict[str, dict] = {}
    if items_path.is_file():
        try:
            items_doc = json.loads(items_path.read_text(encoding="utf-8"))
            items_iter = items_doc.get("items", items_doc) if isinstance(items_doc, dict) else items_doc
            for it in (items_iter.values() if isinstance(items_iter, dict) else items_iter):
                if isinstance(it, dict) and it.get("id"):
                    review_items_by_id[str(it["id"])] = it
        except ValueError:
            pass

    subjects = _subject_order(shots, plan_j)
    subject_info: list[dict] = []
    plan_subjects = plan_j.get("subjects") if isinstance(plan_j, dict) and isinstance(plan_j.get("subjects"), dict) else {}
    for sub in subjects:
        scfg = plan_subjects.get(sub) if isinstance(plan_subjects.get(sub), dict) else {}
        subject_info.append({
            "subject": sub,
            "parts": [str(p) for p in (scfg.get("parts") or [])],
            "region": str(scfg.get("region") or ""),
        })

    first_frame = min(int(s["first"]) for s in shots)
    last_frame = max(int(s["last"]) for s in shots)
    total_frames = last_frame - first_frame + 1

    # Collect owner answers + terminal verdicts per shot
    answers_path = vdir / "owner_answers.jsonl"
    verdicts_path = vdir / "verdicts" / "verdicts.jsonl"
    v_arg = verdicts_path if verdicts_path.is_file() else None

    all_notes: list[dict] = []
    seen_note_keys: set[tuple] = set()
    shot_records: list[dict] = []

    for idx, sh in enumerate(shots):
        s_first, s_last = int(sh["first"]), int(sh["last"])
        who_dict = sh.get("who") if isinstance(sh.get("who"), dict) else {}
        present_subs = [sub for sub in subjects if isinstance(who_dict.get(sub), dict) and who_dict[sub].get("present") == "yes"]
        seen_looks_by_sub: dict[str, list[dict]] = {}
        for sub, w in who_dict.items():
            if isinstance(w, dict) and w.get("seen_looks"):
                seen_looks_by_sub[sub] = [
                    {
                        "first": int(lk["first"]),
                        "last": int(lk["last"]),
                        "part": str(lk.get("part") or "head"),
                        "is": str(lk.get("is") or ""),
                        "by": str(lk.get("by") or ""),
                        "why": str(lk.get("why") or ""),
                    }
                    for lk in w["seen_looks"]
                    if isinstance(lk, dict) and "first" in lk and "last" in lk
                ]

        shot_notes: list[dict] = []
        if answers_path.is_file():
            raw_notes = sos.collect_answers(answers_path, s_first, s_last, verdicts_path=v_arg)
            for n in raw_notes:
                n_clean = {
                    "source": n["source"],
                    "line": n["line"],
                    "at": n["at"],
                    "date": n["date"],
                    "row": n["row"],
                    "span": n["span"],
                    "frame": n["frame"],
                    "shot": [s_first, s_last],
                    "person": n.get("person"),
                    "piece": n.get("piece"),
                    "tag": n.get("tag"),
                    "what": n["what"],
                    "said": n["said"],
                    "where": sos.where_label(n),
                }
                shot_notes.append(n_clean)
                nk = (n["source"], n["line"])
                if nk not in seen_note_keys:
                    seen_note_keys.add(nk)
                    all_notes.append(n_clean)

        shot_records.append({
            "index": idx,
            "first": s_first,
            "last": s_last,
            "frames": s_last - s_first + 1,
            "order": sh.get("order"),
            "scene": str(sh.get("scene") or sh.get("label") or f"{s_first}-{s_last}"),
            "present_subjects": present_subs,
            "seen_looks": seen_looks_by_sub,
            "notes": shot_notes,
        })

    all_notes.sort(key=lambda r: (r["at"], r["line"]))

    # Map coverage shots by ((first, last), subject)
    cov_map: dict[tuple[tuple[int, int], str], dict] = {}
    for c_shot in cov_j["shots"]:
        if isinstance(c_shot, dict) and isinstance(c_shot.get("shot"), list) and len(c_shot["shot"]) == 2 and "subject" in c_shot:
            cov_map[((int(c_shot["shot"][0]), int(c_shot["shot"][1])), str(c_shot["subject"]))] = c_shot

    # Map table rows for piece file URLs
    piece_file_by_name: dict[str, str] = {}
    for trow in (table_j.get("rows") or []):
        if isinstance(trow, dict) and trow.get("name") and trow.get("file"):
            piece_file_by_name[str(trow["name"])] = str(trow["file"])

    def _piece_meta(piece_name: str | None) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        if not piece_name:
            return None, None, None, None, None
        rec = renders_by_name.get(piece_name)
        day_s = str(rec.get("written") or "")[5:10] if rec and rec.get("written") else None
        tier = classify_age_tier(day_s)
        region_s = str((rec or {}).get("graph", {}).get("render_region", ""))
        zoom_s = "zoom" if "zoom" in region_s else ("whole" if rec else None)
        it = review_items_by_id.get(piece_name)
        item_id = piece_name if it else None
        media_rel = None
        if it and it.get("file"):
            media_rel = f"data/{video}/review/media/{it['file']}"
        elif piece_name in piece_file_by_name:
            media_rel = piece_file_by_name[piece_name]
        elif rec and rec.get("file"):
            media_rel = str(rec["file"])
        return day_s, tier, zoom_s, item_id, _page_media_url(media_rel, video, out_html_dir)

    # Build contiguous lane segments per subject across 0..last_frame and collect owed callouts
    lanes: dict[str, list[dict]] = {sub: [] for sub in subjects}
    owed_callouts: list[dict] = []
    coverage_shots_out: list[dict] = []
    owed_frame_subjects = 0
    left_out_frame_subjects = 0

    for sh in shot_records:
        s_first, s_last = sh["first"], sh["last"]
        for sub in subjects:
            if sub not in sh["present_subjects"]:
                lanes[sub].append({
                    "first": s_first,
                    "last": s_last,
                    "frames": s_last - s_first + 1,
                    "shot": [s_first, s_last],
                    "status": "not_present",
                    "step": 0,
                    "step_name": "not_present",
                    "cause": "not in shot",
                    "piece": None,
                    "piece_url": None,
                    "review_item_id": None,
                    "render_day": None,
                    "age_tier": None,
                    "zoom": None,
                    "track_px_min": 0,
                    "track_px_max": 0,
                    "carried_px_min": 0,
                    "carried_px_max": 0,
                    "changed_share_min": None,
                })
                continue

            c = cov_map.get(((s_first, s_last), sub))
            if c is None:
                # Fallback if shot/subject missing from coverage_json
                frames_list = list(range(s_first, s_last + 1))
                steps_list = [1] * len(frames_list)
                step_names = ["no_capture"] * len(frames_list)
                states_list = ["no capture"] * len(frames_list)
                pieces_list = [None] * len(frames_list)
                track_px_list = [0] * len(frames_list)
                carried_px_list = [0] * len(frames_list)
                changed_list = [None] * len(frames_list)
            else:
                frames_list = [int(x) for x in c["frames"]]
                steps_list = [int(x) for x in c["step"]]
                step_names = [str(x) for x in (c.get("step_name") or [""] * len(frames_list))]
                states_list = [str(x or "") for x in (c.get("state") or [""] * len(frames_list))]
                pieces_list = [str(x) if x else None for x in (c.get("piece") or [None] * len(frames_list))]
                track_px_list = [int(x or 0) for x in (c.get("track_px") or [0] * len(frames_list))]
                carried_px_list = [int(x or 0) for x in (c.get("carried_px") or [0] * len(frames_list))]
                changed_list = [float(x) if x is not None else None for x in (c.get("changed_share") or [None] * len(frames_list))]

            coverage_shots_out.append({
                "shot": [s_first, s_last],
                "subject": sub,
                "frames": frames_list,
                "step": steps_list,
                "step_name": step_names,
                "state": states_list,
                "piece": pieces_list,
                "track_px": track_px_list,
                "carried_px": carried_px_list,
                "changed_share": changed_list,
            })

            # Group contiguous frames with identical (status, cause, piece, step)
            cur_run: dict | None = None
            for f_num, st_num, st_name, state_s, piece_nm, t_px, c_px, ch_sh in zip(
                frames_list, steps_list, step_names, states_list, pieces_list, track_px_list, carried_px_list, changed_list
            ):
                if state_s.startswith("left out by a look"):
                    status = "left_out_by_a_look"
                    cause = state_s
                    left_out_frame_subjects += 1
                elif st_num < 4 or "emptied" in state_s or "barely changed" in state_s or "far below" in state_s:
                    status = "owed"
                    cause = state_s or fc.KIND_WORDS.get(st_name, st_name or "owed")
                    owed_frame_subjects += 1
                elif state_s.startswith("held"):
                    status = "held"
                    cause = state_s
                else:
                    status = "carried"
                    cause = state_s or "carried"

                if (
                    cur_run is not None
                    and f_num == cur_run["last"] + 1
                    and cur_run["status"] == status
                    and cur_run["cause"] == cause
                    and cur_run["piece"] == piece_nm
                    and cur_run["step"] == st_num
                ):
                    cur_run["last"] = f_num
                    cur_run["frames"] = cur_run["last"] - cur_run["first"] + 1
                    cur_run["track_px_min"] = min(cur_run["track_px_min"], t_px)
                    cur_run["track_px_max"] = max(cur_run["track_px_max"], t_px)
                    cur_run["carried_px_min"] = min(cur_run["carried_px_min"], c_px)
                    cur_run["carried_px_max"] = max(cur_run["carried_px_max"], c_px)
                    if ch_sh is not None:
                        cur_run["changed_share_min"] = (
                            ch_sh if cur_run["changed_share_min"] is None else min(cur_run["changed_share_min"], ch_sh)
                        )
                else:
                    if cur_run is not None:
                        lanes[sub].append(cur_run)
                        if cur_run["status"] == "owed":
                            owed_callouts.append({
                                "shot": [s_first, s_last],
                                "subject": sub,
                                "first": cur_run["first"],
                                "last": cur_run["last"],
                                "frames": cur_run["frames"],
                                "step": cur_run["step"],
                                "step_name": cur_run["step_name"],
                                "cause": cur_run["cause"],
                                "piece": cur_run["piece"],
                                "track_px_max": cur_run["track_px_max"],
                            })
                    r_day, r_tier, r_zoom, r_item, r_url = _piece_meta(piece_nm)
                    cur_run = {
                        "first": f_num,
                        "last": f_num,
                        "frames": 1,
                        "shot": [s_first, s_last],
                        "status": status,
                        "step": st_num,
                        "step_name": st_name,
                        "cause": cause,
                        "piece": piece_nm,
                        "piece_url": r_url,
                        "review_item_id": r_item,
                        "render_day": r_day,
                        "age_tier": r_tier,
                        "zoom": r_zoom,
                        "track_px_min": t_px,
                        "track_px_max": t_px,
                        "carried_px_min": c_px,
                        "carried_px_max": c_px,
                        "changed_share_min": ch_sh,
                    }
            if cur_run is not None:
                lanes[sub].append(cur_run)
                if cur_run["status"] == "owed":
                    owed_callouts.append({
                        "shot": [s_first, s_last],
                        "subject": sub,
                        "first": cur_run["first"],
                        "last": cur_run["last"],
                        "frames": cur_run["frames"],
                        "step": cur_run["step"],
                        "step_name": cur_run["step_name"],
                        "cause": cur_run["cause"],
                        "piece": cur_run["piece"],
                        "track_px_max": cur_run["track_px_max"],
                    })

    # Clean seams list (scrubbing any absolute paths)
    seams_out: list[dict] = []
    for sm in (cov_j.get("seams") or []):
        if not isinstance(sm, dict):
            continue
        b_info = sm.get("before") if isinstance(sm.get("before"), dict) else {}
        a_info = sm.get("after") if isinstance(sm.get("after"), dict) else {}
        seams_out.append({
            "shot": sm.get("shot"),
            "subject": sm.get("subject"),
            "at": sm.get("at"),
            "before": {
                "piece": b_info.get("piece"),
                "day": b_info.get("day"),
                "written": b_info.get("written"),
                "file": _page_media_url(b_info.get("file"), video, out_html_dir),
            },
            "after": {
                "piece": a_info.get("piece"),
                "day": a_info.get("day"),
                "written": a_info.get("written"),
                "file": _page_media_url(a_info.get("file"), video, out_html_dir),
            },
        })

    # Discover whole-video cuts (assembled, stacked, raw) as relative URLs
    asm = table_j.get("assembly") if isinstance(table_j.get("assembly"), dict) else {}
    asm_cmd = str(asm.get("command") or "")
    assembled_raw = _extract_out_path_from_cmd(asm_cmd)
    source_raw = _extract_source_path_from_cmd(asm_cmd)
    stacked_raw = None
    stack_info = asm.get("stack") if isinstance(asm.get("stack"), dict) else {}
    if stack_info.get("clip"):
        stacked_raw = str(stack_info["clip"])
    elif assembled_raw and assembled_raw.endswith(".mp4"):
        cand_view = assembled_raw[:-4] + "_VIEW.mp4"
        stacked_raw = cand_view

    cuts_urls = {
        "assembled": _page_media_url(assembled_raw, video, out_html_dir),
        "stacked": _page_media_url(stacked_raw, video, out_html_dir),
        "raw": _page_media_url(source_raw, video, out_html_dir),
    }

    tier_counts: Counter = Counter()
    for sub in subjects:
        for seg in lanes[sub]:
            if seg["status"] in ("carried", "held") and seg["age_tier"]:
                tier_counts[seg["age_tier"]] += seg["frames"]

    raw_model = {
        "schema": "h3_timeline/1",
        "meta": {
            "video": video,
            "stamp": stamp,
            "span": [first_frame, last_frame],
            "total_frames": total_frames,
            "fps": 24.0,
            "laid_sha256": laid_sha256,
            "table_laid_sha256": table_laid_sha,
            "laid_matches_table": laid_matches_table,
            "table_sha256": table_sha256,
            "stale_banner": stale_banner,
            "measured_on": {
                "shots": f"data/{video}/shots.json",
                "laid": f"data/{video}/laid.jsonl",
                "table": f"{tables_dir.name}/{table_path.name}",
                "ledger": f"{tables_dir.name}/{ledger_path.name}",
                "frame_coverage": f"{tables_dir.name}/{cov_path.name}",
                "owner_answers": f"data/{video}/owner_answers.jsonl",
            },
            "cannot_see": fc.CANNOT,
            "cuts": cuts_urls,
            "totals": {
                "shots": len(shot_records),
                "subjects": len(subjects),
                "laid_pieces": len(laid_rows),
                "owed_frame_subjects": owed_frame_subjects,
                "owed_spans": len(owed_callouts),
                "left_out_frame_subjects": left_out_frame_subjects,
                "seams": len(seams_out),
                "notes": len(all_notes),
                "laid_frames_by_age_tier": dict(tier_counts),
            },
        },
        "subjects": subject_info,
        "shots": shot_records,
        "lanes": lanes,
        "coverage_shots": coverage_shots_out,
        "owed_callouts": owed_callouts,
        "seams": seams_out,
        "notes": all_notes,
    }
    return _scrub_obj(raw_model, video)


def render_timeline_html(model: dict) -> str:
    meta = model["meta"]
    video = html.escape(str(meta["video"]))
    stamp = html.escape(str(meta["stamp"]))
    laid_sha = html.escape(str(meta["laid_sha256"]))
    span_a, span_b = int(meta["span"][0]), int(meta["span"][1])
    stale_html = ""
    if meta.get("stale_banner"):
        stale_html = f'<div class="stale-banner" role="status">{html.escape(str(meta["stale_banner"]))}</div>\n'

    json_payload = json.dumps(model, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{video} ({span_a}–{span_b}) — Whole-Video NLE Timeline ({stamp})</title>
<style>
  :root {{
    --bg: #f6f5f0;
    --panel: #ffffff;
    --panel-alt: #eeece4;
    --line: #d0cdc2;
    --ink: #1a1c1e;
    --dim: #5c6066;
    --accent: #2b5b84;
    --accent-soft: #d9e6f2;
    --cursor: #d97706;
    --tier-new: #3b82f6;
    --tier-mid: #8b5cf6;
    --tier-old: #64748b;
    --held-bg: #0284c7;
    --look-bg: #78716c;
    --owed-nocap: #b45309;
    --owed-notren: #c2410c;
    --owed-notkept: #9a3412;
    --owed-emptied: #a16207;
    --absent-bg: #e5e4dc;
    --pin-bg: #eab308;
    --seam-bg: #ec4899;
    --banner-bg: #fef3c7;
    --banner-ink: #78350f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #121417;
      --panel: #1b1e23;
      --panel-alt: #23272e;
      --line: #333842;
      --ink: #e6e4df;
      --dim: #9aa0a6;
      --accent: #5899d6;
      --accent-soft: #1e3448;
      --cursor: #f59e0b;
      --tier-new: #3b82f6;
      --tier-mid: #8b5cf6;
      --tier-old: #64748b;
      --held-bg: #0284c7;
      --look-bg: #57534e;
      --owed-nocap: #d97706;
      --owed-notren: #ea580c;
      --owed-notkept: #c2410c;
      --owed-emptied: #ca8a04;
      --absent-bg: #16191d;
      --pin-bg: #facc15;
      --seam-bg: #f472b6;
      --banner-bg: #452b07;
      --banner-ink: #fde68a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #121417;
    --panel: #1b1e23;
    --panel-alt: #23272e;
    --line: #333842;
    --ink: #e6e4df;
    --dim: #9aa0a6;
    --accent: #5899d6;
    --accent-soft: #1e3448;
    --cursor: #f59e0b;
    --tier-new: #3b82f6;
    --tier-mid: #8b5cf6;
    --tier-old: #64748b;
    --held-bg: #0284c7;
    --look-bg: #57534e;
    --owed-nocap: #d97706;
    --owed-notren: #ea580c;
    --owed-notkept: #c2410c;
    --owed-emptied: #ca8a04;
    --absent-bg: #16191d;
    --pin-bg: #facc15;
    --seam-bg: #f472b6;
    --banner-bg: #452b07;
    --banner-ink: #fde68a;
  }}
  :root[data-theme="light"] {{
    --bg: #f6f5f0;
    --panel: #ffffff;
    --panel-alt: #eeece4;
    --line: #d0cdc2;
    --ink: #1a1c1e;
    --dim: #5c6066;
    --accent: #2b5b84;
    --accent-soft: #d9e6f2;
    --cursor: #d97706;
    --tier-new: #3b82f6;
    --tier-mid: #8b5cf6;
    --tier-old: #64748b;
    --held-bg: #0284c7;
    --look-bg: #78716c;
    --owed-nocap: #b45309;
    --owed-notren: #c2410c;
    --owed-notkept: #9a3412;
    --owed-emptied: #a16207;
    --absent-bg: #e5e4dc;
    --pin-bg: #eab308;
    --seam-bg: #ec4899;
    --banner-bg: #fef3c7;
    --banner-ink: #78350f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 14px/1.45 system-ui, -apple-system, sans-serif;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .stale-banner {{
    background: var(--banner-bg);
    color: var(--banner-ink);
    border-bottom: 1px solid var(--cursor);
    padding: 8px 14px;
    font-weight: 600;
  }}
  header.topbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
    z-index: 20;
  }}
  .topbar-title {{ font-size: 16px; font-weight: 700; }}
  .topbar-sub {{ font-size: 12.5px; color: var(--dim); }}
  .nav-links {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .btn {{
    background: var(--panel-alt);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 5px 10px;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }}
  .btn:hover {{ border-color: var(--accent); }}
  .btn.active {{ background: var(--accent-soft); border-color: var(--accent); font-weight: 600; }}
  .workspace {{
    display: grid;
    grid-template-columns: minmax(360px, 1.15fr) minmax(360px, 1fr);
    gap: 12px;
    padding: 12px 16px;
  }}
  @media (max-width: 1100px) {{
    .workspace {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 12px;
  }}
  .card h2 {{
    margin: 0 0 8px;
    font-size: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  video#mainPlayer {{
    width: 100%;
    max-height: 48vh;
    background: #000;
    border-radius: 5px;
    display: block;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 8px;
  }}
  .readout {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    background: var(--panel-alt);
    border: 1px solid var(--line);
  }}
  .timeline-section {{
    padding: 0 16px 16px;
  }}
  .nle-scroll {{
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--panel);
    position: relative;
  }}
  .nle-canvas {{
    position: relative;
    min-width: 100%;
    user-select: none;
  }}
  .nle-row {{
    display: grid;
    grid-template-columns: 120px 1fr;
    border-bottom: 1px solid var(--line);
    min-height: 34px;
  }}
  .nle-row:last-child {{ border-bottom: 0; }}
  .nle-label {{
    padding: 6px 8px;
    font-weight: 600;
    font-size: 12.5px;
    background: var(--panel-alt);
    border-right: 1px solid var(--line);
    position: sticky;
    left: 0;
    z-index: 6;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .nle-label small {{ font-weight: 400; color: var(--dim); font-size: 11px; }}
  .nle-track {{
    position: relative;
    height: 36px;
    background: var(--absent-bg);
    cursor: pointer;
  }}
  .nle-seg {{
    position: absolute;
    top: 3px;
    bottom: 3px;
    border-radius: 3px;
    overflow: hidden;
    font: 600 11px/28px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    padding: 0 4px;
    white-space: nowrap;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.18);
  }}
  .nle-seg.tier-new {{ background: var(--tier-new); }}
  .nle-seg.tier-mid {{ background: var(--tier-mid); }}
  .nle-seg.tier-old {{ background: var(--tier-old); }}
  .nle-seg.status-held {{ background: var(--held-bg); }}
  .nle-seg.status-look {{
    background: repeating-linear-gradient(
      -45deg,
      var(--look-bg),
      var(--look-bg) 5px,
      var(--panel-alt) 5px,
      var(--panel-alt) 10px
    );
    color: var(--ink);
  }}
  .nle-seg.status-owed-nocap {{ background: var(--owed-nocap); }}
  .nle-seg.status-owed-notren {{ background: var(--owed-notren); }}
  .nle-seg.status-owed-notkept {{ background: var(--owed-notkept); }}
  .nle-seg.status-owed-emptied {{ background: var(--owed-emptied); }}
  .nle-shot {{
    position: absolute;
    top: 2px;
    bottom: 2px;
    border-right: 1px solid var(--line);
    font: 500 10.5px/28px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    padding-left: 3px;
    color: var(--dim);
    overflow: hidden;
    white-space: nowrap;
  }}
  .nle-pin {{
    position: absolute;
    top: 4px;
    width: 8px;
    height: 24px;
    margin-left: -4px;
    background: var(--pin-bg);
    border: 1px solid var(--ink);
    border-radius: 3px;
    z-index: 4;
  }}
  .nle-seam {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 3px;
    margin-left: -1px;
    background: var(--seam-bg);
    z-index: 5;
  }}
  .playhead-line {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--cursor);
    pointer-events: none;
    z-index: 10;
  }}
  .legend-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 12px;
    color: var(--dim);
    margin: 8px 0;
  }}
  .swatch {{
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 2px;
    vertical-align: -1px;
    margin-right: 4px;
  }}
  .inspector-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 10px;
  }}
  .inspector-table th, .inspector-table td {{
    border-bottom: 1px solid var(--line);
    padding: 5px 6px;
    text-align: left;
  }}
  .inspector-table th {{ color: var(--dim); font-weight: 600; }}
  .notes-list {{
    max-height: 180px;
    overflow-y: auto;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 6px 8px;
    background: var(--panel-alt);
    margin-bottom: 10px;
  }}
  .note-item {{
    padding: 4px 0;
    border-bottom: 1px dashed var(--line);
    font-size: 12.5px;
  }}
  .note-item:last-child {{ border-bottom: 0; }}
  .form-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 6px;
    margin-bottom: 6px;
  }}
  input[type="text"], select, textarea {{
    width: 100%;
    background: var(--panel-alt);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 5px 7px;
    font: inherit;
    font-size: 13px;
  }}
  textarea {{ min-height: 54px; resize: vertical; }}
  .callout-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    max-height: 110px;
    overflow-y: auto;
    margin-top: 6px;
  }}
</style>
</head>
<body data-laid-sha256="{laid_sha}">
{stale_html}<header class="topbar">
  <div>
    <div class="topbar-title">{video} — Whole-Video NLE Timeline ({span_a}–{span_b})</div>
    <div class="topbar-sub">Rebuild stamp <code>{stamp}</code> · <code>laid.jsonl</code> sha256 <code>{laid_sha[:12]}</code> · Colors show state &amp; render age tier (data, not a verdict)</div>
  </div>
  <div class="nav-links">
    <a class="btn" href="index.html">Review Queue (index.html)</a>
    <a class="btn" href="TO_REVIEW.html">TO_REVIEW.html</a>
    <a class="btn" href="PROVENANCE.html">PROVENANCE.html</a>
    <button class="btn" id="themeBtn" type="button">Theme: Dark/Light</button>
  </div>
</header>

<section class="workspace">
  <div class="card">
    <h2>
      <span>Video Monitor</span>
      <span class="readout" id="frameReadout">Frame {span_a} / {span_b}</span>
    </h2>
    <video id="mainPlayer" controls playsinline preload="metadata"></video>
    <div class="toolbar" id="sourceSwitcher">
      <span class="topbar-sub">Source:</span>
      <button class="btn active" data-src-kind="assembled" type="button">Assembled Cut</button>
      <button class="btn" data-src-kind="stacked" type="button">Stacked Cut</button>
      <button class="btn" data-src-kind="piece" type="button">Active Piece Clip</button>
      <button class="btn" data-src-kind="raw" type="button">Raw Source</button>
    </div>
    <div class="toolbar">
      <button class="btn" id="stepBack10" type="button" title="Shift+Left">-10f</button>
      <button class="btn" id="stepBack1" type="button" title="Left Arrow">← -1f</button>
      <button class="btn" id="playPauseBtn" type="button" title="Space">Play / Pause</button>
      <button class="btn" id="stepFwd1" type="button" title="Right Arrow">+1f →</button>
      <button class="btn" id="stepFwd10" type="button" title="Shift+Right">+10f</button>
      <button class="btn" id="loopInBtn" type="button" title="Bracket [">[ In</button>
      <button class="btn" id="loopOutBtn" type="button" title="Bracket ]">] Out</button>
      <button class="btn" id="loopClearBtn" type="button" title="Backslash">Clear Loop</button>
      <span class="topbar-sub" id="loopStatus">No loop</span>
    </div>
    <div class="toolbar">
      <button class="btn" id="nextGapBtn" type="button" title="Hotkey G">Next Owed Gap (G)</button>
      <button class="btn" id="nextCutBtn" type="button" title="Hotkey C">Next Shot Cut (C)</button>
      <button class="btn" id="nextNoteBtn" type="button" title="Hotkey N">Next Pinned Note (N)</button>
      <span class="topbar-sub">Jump to frame:</span>
      <input type="text" id="jumpInput" placeholder="e.g. 2455" style="width:90px">
      <button class="btn" id="jumpBtn" type="button">Go</button>
    </div>
  </div>

  <div class="card">
    <h2>
      <span>Frame Inspector &amp; Shot Notes</span>
      <span class="readout" id="shotReadout">Shot —</span>
    </h2>
    <table class="inspector-table">
      <thead>
        <tr>
          <th>Subject</th>
          <th>State / Cause</th>
          <th>Piece (Day, Tier, Zoom)</th>
          <th>Track / Carried px</th>
        </tr>
      </thead>
      <tbody id="inspectorBody"></tbody>
    </table>

    <div class="topbar-sub" style="font-weight:600; margin-bottom:4px;">Prior Owner Words on This Shot (chronological):</div>
    <div class="notes-list" id="shotNotesBox"></div>

    <form id="pinNoteForm">
      <div class="topbar-sub" style="font-weight:600; margin-bottom:4px;">Pin a Note to <code>owner_answers.jsonl</code> (<code>row: "timeline"</code>):</div>
      <div class="form-grid">
        <input type="text" id="noteWho" value="owner" placeholder="Who">
        <input type="text" id="noteFrames" placeholder="Frame or FIRST-LAST">
        <select id="noteSubject">
          <option value="">All / General</option>
        </select>
        <input type="text" id="notePiece" placeholder="Piece (optional)">
        <select id="noteTag">
          <option value="">No tag</option>
          <option value="gap_flash">gap_flash</option>
          <option value="identity_drift">identity_drift</option>
          <option value="seam_jump">seam_jump</option>
          <option value="mouth_gaze">mouth_gaze</option>
          <option value="hair_border">hair_border</option>
          <option value="keep_note">keep_note</option>
          <option value="remake">remake</option>
        </select>
      </div>
      <textarea id="noteWords" placeholder="Describe what happens at this frame or span..."></textarea>
      <div class="toolbar">
        <button class="btn active" type="submit">Pin Note to owner_answers.jsonl</button>
        <span class="topbar-sub" id="saveStatus"></span>
      </div>
    </form>
  </div>
</section>

<section class="timeline-section">
  <div class="card">
    <h2>
      <span>Multi-Track Timeline ({span_a}–{span_b})</span>
      <div class="toolbar" style="margin-top:0" id="ageFilterBar">
        <span class="topbar-sub">Age Filter:</span>
        <button class="btn active" data-age-filter="all" type="button">All</button>
        <button class="btn" data-age-filter="24h" type="button">24h (10/23+)</button>
        <button class="btn" data-age-filter="48h" type="button">48h (10/20–10/22)</button>
        <button class="btn" data-age-filter="72h" type="button">72h (10/20+)</button>
        <button class="btn" data-age-filter="older" type="button">Older (pre-10/20)</button>
        <button class="btn" data-age-filter="owed" type="button">Owed</button>
      </div>
      <div class="toolbar" style="margin-top:0" id="zoomBar">
        <span class="topbar-sub">Zoom:</span>
        <button class="btn active" data-zoom="whole" type="button">Whole</button>
        <button class="btn" data-zoom="quarter" type="button">Quarter</button>
        <button class="btn" data-zoom="400" type="button">400f Window</button>
        <button class="btn" data-zoom="100" type="button">100f Detail</button>
      </div>
    </h2>
    <div class="legend-bar">
      <span><i class="swatch" style="background:var(--tier-new)"></i>Laid 10/23+</span>
      <span><i class="swatch" style="background:var(--tier-mid)"></i>Laid 10/20–10/22</span>
      <span><i class="swatch" style="background:var(--tier-old)"></i>Laid pre-10/20</span>
      <span><i class="swatch" style="background:var(--held-bg)"></i>Held in piece</span>
      <span><i class="swatch" style="background:var(--look-bg)"></i>Left out by a look (hatched)</span>
      <span><i class="swatch" style="background:var(--owed-nocap)"></i>Owed: no capture</span>
      <span><i class="swatch" style="background:var(--owed-notren)"></i>Owed: captured, not rendered</span>
      <span><i class="swatch" style="background:var(--owed-notkept)"></i>Owed: rendered, not kept</span>
      <span><i class="swatch" style="background:var(--owed-emptied)"></i>Owed: emptied / barely changed</span>
      <span><i class="swatch" style="background:var(--pin-bg)"></i>Pinned owner note</span>
      <span><i class="swatch" style="background:var(--seam-bg)"></i>Mid-shot piece seam</span>
    </div>

    <div class="nle-scroll" id="nleScroll">
      <div class="nle-canvas" id="nleCanvas">
        <div id="nleRows"></div>
        <div class="playhead-line" id="playheadLine" style="left:120px"></div>
      </div>
    </div>

    <div class="topbar-sub" style="margin-top:10px; font-weight:600;">
      Owed Spans ({meta["totals"]["owed_spans"]} spans, {meta["totals"]["owed_frame_subjects"]} frame-subjects) &amp; Mid-Shot Seams ({meta["totals"]["seams"]}):
    </div>
    <div class="callout-strip" id="calloutStrip"></div>
    <div class="topbar-sub" style="margin-top:8px;">
      <b>Cannot see:</b> {html.escape(str(meta["cannot_see"]))}
    </div>
  </div>
</section>

<script id="timeline-data" type="application/json">{json_payload}</script>
<script>
(() => {{
  const DATA = JSON.parse(document.getElementById('timeline-data').textContent);
  const META = DATA.meta;
  const SPAN_A = META.span[0];
  const SPAN_B = META.span[1];
  const TOTAL = META.total_frames;
  const FPS = META.fps || 24.0;

  let currentFrame = SPAN_A;
  let loopIn = null;
  let loopOut = null;
  let activeSourceKind = 'assembled';
  let activeClipFirst = SPAN_A;
  let zoomMode = 'whole';
  let ageFilter = 'all';

  function matchesAgeFilter(seg) {{
    if (ageFilter === 'all') return true;
    if (ageFilter === '24h') return seg.age_tier === '10/23+';
    if (ageFilter === '48h') return seg.age_tier === '10/20-10/22';
    if (ageFilter === '72h') return seg.age_tier === '10/23+' || seg.age_tier === '10/20-10/22';
    if (ageFilter === 'older') return seg.age_tier === 'pre-10/20';
    if (ageFilter === 'owed') return seg.status === 'owed';
    return true;
  }}

  // Build fast per-frame lookup: frame -> {{ shot, subjects: {{ sub -> info }} }}
  const shotByFrame = new Array(SPAN_B + 1);
  for (const sh of DATA.shots) {{
    for (let f = sh.first; f <= sh.last; f++) shotByFrame[f] = sh;
  }}

  const covBySubFrame = {{}};
  for (const s of DATA.subjects) covBySubFrame[s.subject] = new Array(SPAN_B + 1);
  for (const cs of DATA.coverage_shots) {{
    const arr = covBySubFrame[cs.subject];
    if (!arr) continue;
    for (let i = 0; i < cs.frames.length; i++) {{
      const f = cs.frames[i];
      arr[f] = {{
        step: cs.step[i],
        step_name: cs.step_name[i],
        state: cs.state[i],
        piece: cs.piece[i],
        track_px: cs.track_px[i],
        carried_px: cs.carried_px[i],
        changed_share: cs.changed_share[i]
      }};
    }}
  }}

  const segBySubFrame = {{}};
  for (const s of DATA.subjects) {{
    const arr = new Array(SPAN_B + 1);
    for (const seg of (DATA.lanes[s.subject] || [])) {{
      for (let f = seg.first; f <= seg.last; f++) arr[f] = seg;
    }}
    segBySubFrame[s.subject] = arr;
  }}

  // Populate subject selector
  const subSelect = document.getElementById('noteSubject');
  for (const s of DATA.subjects) {{
    const opt = document.createElement('option');
    opt.value = s.subject;
    opt.textContent = s.subject + (s.parts.length ? ' (' + s.parts.join(', ') + ')' : '');
    subSelect.appendChild(opt);
  }}

  const player = document.getElementById('mainPlayer');
  function setPlayerSource(kind) {{
    activeSourceKind = kind;
    document.querySelectorAll('#sourceSwitcher button').forEach(b => {{
      b.classList.toggle('active', b.getAttribute('data-src-kind') === kind);
    }});
    let url = null;
    activeClipFirst = SPAN_A;
    if (kind === 'assembled') url = META.cuts.assembled;
    else if (kind === 'stacked') url = META.cuts.stacked || META.cuts.assembled;
    else if (kind === 'raw') url = META.cuts.raw;
    else if (kind === 'piece') {{
      for (const s of DATA.subjects) {{
        const seg = segBySubFrame[s.subject][currentFrame];
        if (seg && seg.piece_url) {{
          url = seg.piece_url;
          activeClipFirst = seg.shot[0];
          break;
        }}
      }}
    }}
    if (url && player.getAttribute('src') !== url) {{
      player.setAttribute('src', url);
      player.currentTime = Math.max(0, (currentFrame - activeClipFirst) / FPS);
    }}
  }}

  function fmtTimecode(f) {{
    const sec = f / FPS;
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(2).padStart(5, '0');
    return `${{m}}:${{s}}`;
  }}

  function seekFrame(f, syncVideo = true) {{
    currentFrame = Math.max(SPAN_A, Math.min(SPAN_B, Math.round(f)));
    if (syncVideo && player.getAttribute('src')) {{
      const targetSec = Math.max(0, (currentFrame - activeClipFirst + 0.25) / FPS);
      if (Math.abs(player.currentTime - targetSec) > 0.4 / FPS) {{
        player.currentTime = targetSec;
      }}
    }}
    updateInspector();
    updatePlayheadPosition();
  }}

  function updateInspector() {{
    const sh = shotByFrame[currentFrame];
    document.getElementById('frameReadout').textContent = `Frame ${{currentFrame}} / ${{SPAN_B}} (${{fmtTimecode(currentFrame)}})`;
    document.getElementById('shotReadout').textContent = sh ? `Shot ${{sh.first}}–${{sh.last}} (${{sh.frames}}f)` : 'Shot —';
    const noteFramesInput = document.getElementById('noteFrames');
    if (loopIn !== null && loopOut !== null) {{
      noteFramesInput.value = `${{Math.min(loopIn, loopOut)}}-${{Math.max(loopIn, loopOut)}}`;
    }} else {{
      noteFramesInput.value = String(currentFrame);
    }}

    const tbody = document.getElementById('inspectorBody');
    tbody.replaceChildren();
    let defaultPiece = '';
    for (const s of DATA.subjects) {{
      const sub = s.subject;
      const tr = document.createElement('tr');
      const seg = segBySubFrame[sub][currentFrame];
      const cov = covBySubFrame[sub][currentFrame];
      const tdSub = document.createElement('td');
      tdSub.innerHTML = `<b>${{sub}}</b>`;
      const tdState = document.createElement('td');
      const tdPiece = document.createElement('td');
      const tdPx = document.createElement('td');
      if (!seg || seg.status === 'not_present') {{
        tdState.textContent = 'not in shot';
        tdPiece.textContent = '—';
        tdPx.textContent = '—';
      }} else {{
        tdState.textContent = `${{seg.status}}: ${{cov && cov.state ? cov.state : seg.cause}}`;
        if (seg.piece) {{
          if (!defaultPiece) defaultPiece = seg.piece;
          const info = `${{seg.piece}} (${{seg.render_day || '?'}}, ${{seg.age_tier || '?'}}, ${{seg.zoom || '?'}})`;
          if (seg.review_item_id) {{
            const a = document.createElement('a');
            a.href = `index.html#${{encodeURIComponent(seg.review_item_id)}}`;
            a.textContent = info;
            tdPiece.appendChild(a);
          }} else {{
            tdPiece.textContent = info;
          }}
        }} else {{
          tdPiece.textContent = 'none laid';
        }}
        const tpx = cov ? cov.track_px : seg.track_px_max;
        const cpx = cov ? cov.carried_px : seg.carried_px_max;
        const chs = cov && cov.changed_share != null ? `, chg ${{cov.changed_share.toFixed(2)}}` : '';
        tdPx.textContent = `track ${{tpx}} px / carried ${{cpx}} px${{chs}}`;
      }}
      tr.append(tdSub, tdState, tdPiece, tdPx);
      tbody.appendChild(tr);
    }}
    const notePieceInput = document.getElementById('notePiece');
    if (!notePieceInput.dataset.userEdited) notePieceInput.value = defaultPiece;

    // Update shot notes
    const notesBox = document.getElementById('shotNotesBox');
    notesBox.replaceChildren();
    const notes = sh ? sh.notes : [];
    if (!notes.length) {{
      const empty = document.createElement('div');
      empty.className = 'topbar-sub';
      empty.textContent = 'No owner notes recorded on this shot yet.';
      notesBox.appendChild(empty);
    }} else {{
      for (const n of notes) {{
        const div = document.createElement('div');
        div.className = 'note-item';
        const b = document.createElement('b');
        b.textContent = `${{n.date}} [${{n.row}} · ${{n.where || 'shot'}} · ${{n.what}}]: `;
        const sp = document.createElement('span');
        sp.textContent = n.said;
        div.append(b, sp);
        notesBox.appendChild(div);
      }}
    }}
  }}

  function segColorClass(seg) {{
    if (seg.status === 'not_present') return '';
    if (seg.status === 'left_out_by_a_look') return 'status-look';
    if (seg.status === 'held') return 'status-held';
    if (seg.status === 'owed') {{
      if (seg.step_name === 'no_capture') return 'status-owed-nocap';
      if (seg.step_name === 'captured_not_rendered') return 'status-owed-notren';
      if (seg.step_name === 'rendered_not_kept') return 'status-owed-notkept';
      return 'status-owed-emptied';
    }}
    if (seg.age_tier === '10/23+') return 'tier-new';
    if (seg.age_tier === '10/20-10/22') return 'tier-mid';
    return 'tier-old';
  }}

  function buildTracks() {{
    const container = document.getElementById('nleRows');
    container.replaceChildren();
    const canvas = document.getElementById('nleCanvas');
    let trackWidthPx = 1200;
    if (zoomMode === 'whole') trackWidthPx = Math.max(960, document.getElementById('nleScroll').clientWidth - 130);
    else if (zoomMode === 'quarter') trackWidthPx = 3600;
    else if (zoomMode === '400') trackWidthPx = Math.round((TOTAL / 400) * 1000);
    else if (zoomMode === '100') trackWidthPx = Math.round((TOTAL / 100) * 1000);
    canvas.style.width = (120 + trackWidthPx) + 'px';

    const pctLeft = f => (((f - SPAN_A) / TOTAL) * 100).toFixed(4) + '%';
    const pctWidth = n => Math.max(0.08, (n / TOTAL) * 100).toFixed(4) + '%';

    // 1. Shots ruler row
    const rulerRow = document.createElement('div');
    rulerRow.className = 'nle-row';
    const rulerLabel = document.createElement('div');
    rulerLabel.className = 'nle-label';
    rulerLabel.innerHTML = `<span>Shots (${{DATA.shots.length}})</span><small>Click to seek</small>`;
    const rulerTrack = document.createElement('div');
    rulerTrack.className = 'nle-track';
    for (const sh of DATA.shots) {{
      const blk = document.createElement('div');
      blk.className = 'nle-shot';
      blk.style.left = pctLeft(sh.first);
      blk.style.width = pctWidth(sh.frames);
      blk.title = `Shot ${{sh.first}}-${{sh.last}} (${{sh.frames}}f)`;
      blk.textContent = sh.first;
      rulerTrack.appendChild(blk);
    }}
    rulerRow.append(rulerLabel, rulerTrack);
    container.appendChild(rulerRow);

    // 2. Subject lanes
    for (const s of DATA.subjects) {{
      const sub = s.subject;
      const row = document.createElement('div');
      row.className = 'nle-row';
      const lab = document.createElement('div');
      lab.className = 'nle-label';
      lab.innerHTML = `<span>${{sub}}</span><small>${{s.parts.join(', ')}}</small>`;
      const trk = document.createElement('div');
      trk.className = 'nle-track';
      for (const seg of (DATA.lanes[sub] || [])) {{
        if (seg.status === 'not_present') continue;
        const el = document.createElement('div');
        el.className = 'nle-seg ' + segColorClass(seg);
        el.style.left = pctLeft(seg.first);
        el.style.width = pctWidth(seg.frames);
        el.style.opacity = matchesAgeFilter(seg) ? '1' : '0.18';
        const tag = seg.piece ? `${{seg.piece}} (${{seg.render_day || '?'}}, ${{seg.zoom || '?'}})` : seg.cause;
        el.title = `${{sub}} ${{seg.first}}-${{seg.last}} (${{seg.frames}}f): ${{seg.status}} — ${{tag}}`;
        el.textContent = seg.piece ? `${{seg.piece}} · ${{seg.render_day || ''}}` : seg.cause;
        trk.appendChild(el);
      }}
      for (const sm of DATA.seams) {{
        if (sm.subject !== sub || !sm.at) continue;
        const smEl = document.createElement('div');
        smEl.className = 'nle-seam';
        smEl.style.left = pctLeft(sm.at[1]);
        smEl.title = `Seam on ${{sub}} at ${{sm.at[0]}}/${{sm.at[1]}}: ${{sm.before.piece}} -> ${{sm.after.piece}}`;
        trk.appendChild(smEl);
      }}
      row.append(lab, trk);
      container.appendChild(row);
    }}

    // 3. Notes & Pins row
    const pinsRow = document.createElement('div');
    pinsRow.className = 'nle-row';
    const pinsLabel = document.createElement('div');
    pinsLabel.className = 'nle-label';
    pinsLabel.innerHTML = `<span>Owner Notes</span><small>${{DATA.notes.length}} pin(s)</small>`;
    const pinsTrack = document.createElement('div');
    pinsTrack.className = 'nle-track';
    for (const n of DATA.notes) {{
      const fPin = n.frame != null ? n.frame : (n.span ? n.span[0] : (n.shot ? n.shot[0] : SPAN_A));
      const pin = document.createElement('div');
      pin.className = 'nle-pin';
      pin.style.left = pctLeft(fPin);
      pin.title = `Frame ${{fPin}} (${{n.date}}, ${{n.row}}): ${{n.said}}`;
      pin.addEventListener('click', e => {{
        e.stopPropagation();
        seekFrame(fPin, true);
      }});
      pinsTrack.appendChild(pin);
    }}
    pinsRow.append(pinsLabel, pinsTrack);
    container.appendChild(pinsRow);

    // Click anywhere on a track to seek
    container.querySelectorAll('.nle-track').forEach(trk => {{
      trk.addEventListener('click', e => {{
        const rect = trk.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        seekFrame(SPAN_A + Math.round(ratio * TOTAL), true);
      }});
    }});

    updatePlayheadPosition();
  }}

  function updatePlayheadPosition() {{
    const tracks = document.querySelectorAll('.nle-track');
    if (!tracks.length) return;
    const trackWidth = tracks[0].clientWidth;
    const x = 120 + ((currentFrame - SPAN_A) / TOTAL) * trackWidth;
    const line = document.getElementById('playheadLine');
    line.style.left = `${{x}}px`;
  }}

  // Build Owed & Seam callout strip
  function buildCallouts() {{
    const strip = document.getElementById('calloutStrip');
    strip.replaceChildren();
    for (const ow of DATA.owed_callouts) {{
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      const sp = ow.first === ow.last ? String(ow.first) : `${{ow.first}}-${{ow.last}}`;
      b.textContent = `Owed ${{ow.subject}} ${{sp}} (${{ow.frames}}f: ${{ow.cause}})`;
      b.addEventListener('click', () => seekFrame(ow.first, true));
      strip.appendChild(b);
    }}
    for (const sm of DATA.seams) {{
      if (!sm.at) continue;
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      b.textContent = `Seam ${{sm.subject}} ${{sm.at[0]}}/${{sm.at[1]}} (${{sm.before.piece}} → ${{sm.after.piece}})`;
      b.addEventListener('click', () => seekFrame(sm.at[1], true));
      strip.appendChild(b);
    }}
  }}

  // Controls & Hotkeys
  document.getElementById('stepBack10').addEventListener('click', () => {{ player.pause(); seekFrame(currentFrame - 10, true); }});
  document.getElementById('stepBack1').addEventListener('click', () => {{ player.pause(); seekFrame(currentFrame - 1, true); }});
  document.getElementById('stepFwd1').addEventListener('click', () => {{ player.pause(); seekFrame(currentFrame + 1, true); }});
  document.getElementById('stepFwd10').addEventListener('click', () => {{ player.pause(); seekFrame(currentFrame + 10, true); }});
  document.getElementById('playPauseBtn').addEventListener('click', () => {{
    if (player.paused) player.play(); else player.pause();
  }});

  function updateLoopUI() {{
    const el = document.getElementById('loopStatus');
    if (loopIn !== null && loopOut !== null) el.textContent = `Loop ${{Math.min(loopIn, loopOut)}}–${{Math.max(loopIn, loopOut)}}`;
    else if (loopIn !== null) el.textContent = `In: ${{loopIn}}`;
    else el.textContent = 'No loop';
    updateInspector();
  }}
  document.getElementById('loopInBtn').addEventListener('click', () => {{ loopIn = currentFrame; updateLoopUI(); }});
  document.getElementById('loopOutBtn').addEventListener('click', () => {{ loopOut = currentFrame; updateLoopUI(); }});
  document.getElementById('loopClearBtn').addEventListener('click', () => {{ loopIn = null; loopOut = null; updateLoopUI(); }});

  function jumpNextGap() {{
    const next = DATA.owed_callouts.find(o => o.first > currentFrame) || DATA.owed_callouts[0];
    if (next) seekFrame(next.first, true);
  }}
  function jumpNextCut() {{
    const next = DATA.shots.find(s => s.first > currentFrame) || DATA.shots[0];
    if (next) seekFrame(next.first, true);
  }}
  function jumpNextNote() {{
    const frames = DATA.notes.map(n => n.frame != null ? n.frame : (n.span ? n.span[0] : n.shot[0])).sort((a, b) => a - b);
    const next = frames.find(f => f > currentFrame) ?? frames[0];
    if (next != null) seekFrame(next, true);
  }}

  document.getElementById('nextGapBtn').addEventListener('click', jumpNextGap);
  document.getElementById('nextCutBtn').addEventListener('click', jumpNextCut);
  document.getElementById('nextNoteBtn').addEventListener('click', jumpNextNote);
  document.getElementById('jumpBtn').addEventListener('click', () => {{
    const v = parseInt(document.getElementById('jumpInput').value, 10);
    if (!Number.isNaN(v)) seekFrame(v, true);
  }});

  document.querySelectorAll('#sourceSwitcher button').forEach(btn => {{
    btn.addEventListener('click', () => setPlayerSource(btn.getAttribute('data-src-kind')));
  }});
  document.querySelectorAll('#zoomBar button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      zoomMode = btn.getAttribute('data-zoom');
      document.querySelectorAll('#zoomBar button').forEach(b => b.classList.toggle('active', b === btn));
      buildTracks();
    }});
  }});
  document.querySelectorAll('#ageFilterBar button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      ageFilter = btn.getAttribute('data-age-filter') || 'all';
      document.querySelectorAll('#ageFilterBar button').forEach(b => b.classList.toggle('active', b === btn));
      buildTracks();
    }});
  }});

  document.getElementById('themeBtn').addEventListener('click', () => {{
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  }});

  window.addEventListener('keydown', e => {{
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); player.pause(); seekFrame(currentFrame - (e.shiftKey ? 10 : 1), true); }}
    else if (e.key === 'ArrowRight') {{ e.preventDefault(); player.pause(); seekFrame(currentFrame + (e.shiftKey ? 10 : 1), true); }}
    else if (e.key === ' ') {{ e.preventDefault(); if (player.paused) player.play(); else player.pause(); }}
    else if (e.key === '[') {{ loopIn = currentFrame; updateLoopUI(); }}
    else if (e.key === ']') {{ loopOut = currentFrame; updateLoopUI(); }}
    else if (e.key === '\\\\') {{ loopIn = null; loopOut = null; updateLoopUI(); }}
    else if (e.key === 'g' || e.key === 'G') {{ jumpNextGap(); }}
    else if (e.key === 'c' || e.key === 'C') {{ jumpNextCut(); }}
    else if (e.key === 'n' || e.key === 'N') {{ jumpNextNote(); }}
  }});

  player.addEventListener('timeupdate', () => {{
    const f = activeClipFirst + Math.floor(player.currentTime * FPS + 0.001);
    if (loopIn !== null && loopOut !== null) {{
      const lo = Math.min(loopIn, loopOut);
      const hi = Math.max(loopIn, loopOut);
      if (f > hi) {{
        player.currentTime = Math.max(0, (lo - activeClipFirst) / FPS);
        return;
      }}
    }}
    seekFrame(f, false);
  }});

  // Merge live notes from /api/answers and handle Pin Note form submission
  function integrateLiveAnswer(row, lineNo) {{
    if (!row || (row.who && row.who !== 'owner' && row.who !== 'not said')) return;
    const words = [row.chosen, row.words].filter(Boolean).join(' — ').trim();
    if (!words) return;
    const fPin = row.frame != null ? Number(row.frame) : (row.source_frame != null ? Number(row.source_frame) : null);
    let span = Array.isArray(row.frames) && row.frames.length === 2 ? [Number(row.frames[0]), Number(row.frames[1])] : null;
    const targetFrame = fPin != null ? fPin : (span ? span[0] : null);
    if (targetFrame == null || Number.isNaN(targetFrame)) return;
    const sh = shotByFrame[targetFrame];
    if (!sh) return;
    const already = sh.notes.some(n => n.at === row.at && n.said === words);
    if (already) return;
    const entry = {{
      source: 'owner_answers.jsonl',
      line: lineNo || (DATA.notes.length + 1),
      at: row.at || '',
      date: (row.at || '').slice(0, 16).replace('T', ' ') || 'just now',
      row: row.row || 'timeline',
      span: span,
      frame: fPin,
      shot: [sh.first, sh.last],
      person: row.person || null,
      piece: row.piece || null,
      tag: row.tag || null,
      what: row.what || 'timeline_note',
      said: words,
      where: [
        span ? (span[0] !== span[1] ? `frames ${{span[0]}}-${{span[1]}}` : `frame ${{span[0]}}`) : null,
        fPin != null && (!span || span[0] !== fPin || span[1] !== fPin) ? `frame ${{fPin}}` : null,
        row.person ? row.person : null,
        row.piece ? `piece ${{row.piece}}` : null,
        row.tag ? `tag ${{row.tag}}` : null
      ].filter(Boolean).join(', ') || 'shot'
    }};
    sh.notes.push(entry);
    DATA.notes.push(entry);
  }}

  fetch('/api/answers').then(r => r.ok ? r.json() : []).then(list => {{
    if (Array.isArray(list)) {{
      list.forEach((r, idx) => integrateLiveAnswer(r, idx + 1));
      buildTracks();
      updateInspector();
    }}
  }}).catch(() => {{}});

  document.getElementById('pinNoteForm').addEventListener('submit', async e => {{
    e.preventDefault();
    const words = document.getElementById('noteWords').value.trim();
    const statusEl = document.getElementById('saveStatus');
    if (!words) {{
      statusEl.textContent = 'Enter note words first.';
      return;
    }}
    const rawSpan = document.getElementById('noteFrames').value.trim();
    let f0 = currentFrame, f1 = currentFrame;
    const m = rawSpan.match(/^(\\d+)(?:-(\\d+))?$/);
    if (m) {{
      f0 = parseInt(m[1], 10);
      f1 = m[2] != null ? parseInt(m[2], 10) : f0;
    }}
    const sh = shotByFrame[f0] || shotByFrame[currentFrame];
    const payload = {{
      who: document.getElementById('noteWho').value.trim() || 'owner',
      video: META.video,
      row: 'timeline',
      what: 'timeline_note',
      frames: [Math.min(f0, f1), Math.max(f0, f1)],
      frame: f0,
      source_frame: f0,
      shot: sh ? `${{sh.first}}-${{sh.last}}` : `${{f0}}-${{f1}}`,
      person: document.getElementById('noteSubject').value || undefined,
      piece: document.getElementById('notePiece').value.trim() || undefined,
      tag: document.getElementById('noteTag').value || undefined,
      words: words,
      laid_sha256: META.laid_sha256,
      table_sha256: META.table_sha256
    }};
    statusEl.textContent = 'Saving...';
    try {{
      const res = await fetch('/api/answer', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.status);
      integrateLiveAnswer(j.saved, j.lines);
      document.getElementById('noteWords').value = '';
      statusEl.textContent = `Saved line ${{j.lines}} at ${{j.saved.at}}`;
      buildTracks();
      updateInspector();
    }} catch (err) {{
      statusEl.textContent = `Save failed (${{err.message}}). Is serve.py running?`;
    }}
  }});

  setPlayerSource('assembled');
  buildTracks();
  buildCallouts();
  seekFrame(SPAN_A, false);
  window.addEventListener('resize', () => {{ if (zoomMode === 'whole') buildTracks(); }});
}})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--video", required=True, help="video stem under --data")
    ap.add_argument("--data", default="data", help="data root directory (default: data)")
    ap.add_argument("--tables", required=True, help="the rebuild's delivery tables folder")
    ap.add_argument("--stamp", required=True, help="the rebuild's stamp, e.g. 2026-10-24T2341Z")
    ap.add_argument("--share", help="optional share output root")
    ap.add_argument("--out-json", required=True, help="output path for timeline_<video>_<stamp>.json")
    ap.add_argument("--out-html", required=True, help="output path for TIMELINE.html")
    a = ap.parse_args(argv)

    out_json_path = Path(a.out_json)
    out_html_path = Path(a.out_html)
    share_path = Path(a.share) if a.share else None
    try:
        model = build_timeline_model(
            video=a.video,
            data_dir=Path(a.data),
            tables_dir=Path(a.tables),
            stamp=a.stamp,
            out_html_dir=out_html_path.resolve().parent,
            share_dir=share_path,
        )
    except Refused as e:
        sys.stderr.write(f"{e}\n")
        return 1

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_html_path.write_text(render_timeline_html(model), encoding="utf-8")
    t = model["meta"]["totals"]
    print(
        f"{out_html_path} + {out_json_path}: {t['shots']} shots, {t['laid_pieces']} laid pieces, "
        f"{t['owed_spans']} owed spans ({t['owed_frame_subjects']} frame-subjects), {t['notes']} notes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
