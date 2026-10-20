#!/usr/bin/env python3
"""Build a clean, chronological Cut & Hierarchical Data Flow Lineage Map (PROVENANCE.html, CUT_MAP.html, CUT_MAP.md) for fun (0000-6394).

Reads directly from the live sources of truth:
  - data/fun/shots.json (chronological shots 0..6394 and active subjects per shot)
  - data/fun/plan.json (subject stills, parts, clothing spans, voice spans, departures)
  - data/fun/cast/cast.jsonl (owner cast seed boxes per frame & subject)
  - data/fun/laid.jsonl (currently laid render MP4 per shot + subject)
  - data/*_job/*.render.built.json & *.render.queued.json (lineage: input job JSON, preview, capture folder, source video, digests, windows)
  - data/2026-*_<load>/manifest.json & flags.json (capture inputs, tracker/part files, gate status, held/emptied frames)
  - data/2026-10-20_fun_phase2_inputs/ & data/2026-10-20_fun_phase3_inputs/ (queued/pre-built inputs for remaining/updated shots)

Writes:
  - data/fun/CUT_MAP.md
  - data/fun/poc/PROVENANCE.html (served at http://localhost:8765/PROVENANCE.html)
  - data/fun/poc/CUT_MAP.html (served at http://localhost:8765/CUT_MAP.html)
  - H3_OUTPUT_DIR/Video/fun/PROVENANCE.html
  - H3_OUTPUT_DIR/Video/fun/CUT_MAP.html
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
FUN_DIR = DATA_DIR / "fun"
SHARE_DIR = Path(os.environ.get("H3_OUTPUT_DIR", "output"))
INPUT_DIR = Path(os.environ.get("H3_INPUT_DIR", str(SHARE_DIR.parent / "input")))


def local_tz_abbr() -> str:
    return datetime.datetime.now().astimezone().strftime("%Z") or "local"


def fmt_local_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")


def fmt_iso_to_local(iso_str: str | None) -> str:
    if not iso_str:
        return "—"
    try:
        s = iso_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception:
        return iso_str


_MTIME_CACHE: dict[str, str] = {}


def path_mtime_local(p: Path | None) -> str:
    if p is None:
        return "missing"
    key = str(p)
    if key in _MTIME_CACHE:
        return _MTIME_CACHE[key]
    if not key.startswith(str(REPO)):
        _MTIME_CACHE[key] = "on share (H3_OUTPUT_DIR)"
        return _MTIME_CACHE[key]
    try:
        res = fmt_local_ts(p.stat().st_mtime) if p.exists() else "missing"
    except Exception:
        res = "missing"
    _MTIME_CACHE[key] = res
    return res


def load_mtime_local_for_laid(r: dict, built_idx: dict[str, dict]) -> str:
    """Resolve render timestamp from local render.queued.json, render.built.json, or laid.jsonl 'added'."""
    lname = r.get("load", "")
    binfo = built_idx.get(lname)
    if binfo:
        if binfo.get("queued_path") and binfo["queued_path"].exists():
            return path_mtime_local(binfo["queued_path"])
        if binfo.get("built_path") and binfo["built_path"].exists():
            return path_mtime_local(binfo["built_path"])
    if r.get("added"):
        return fmt_iso_to_local(r.get("added"))
    return "—"


def fmt_tc(frame: int, fps: float = 24.0) -> str:
    sec = frame / fps
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:04.1f}"


def rel_or_str(p: Path | str | None) -> str:
    if not p:
        return "—"
    path_obj = Path(p)
    try:
        return str(path_obj.relative_to(REPO))
    except Exception:
        return str(path_obj)


def load_built_index() -> dict[str, dict]:
    """Map load_name -> {job_dir, built_path, queued_path, built, queued}."""
    idx: dict[str, dict] = {}
    for built_path in sorted(DATA_DIR.glob("*_job/*.render.built.json")):
        try:
            b = json.loads(built_path.read_text())
        except Exception:
            continue
        load_name = ((b.get("load") or {}).get("name")) or built_path.name.split(".render.built.json")[0]
        queued_path = built_path.with_name(built_path.name.replace(".built.json", ".queued.json"))
        q = None
        if queued_path.is_file():
            try:
                q = json.loads(queued_path.read_text())
            except Exception:
                q = None
        idx[load_name] = {
            "job_dir": built_path.parent,
            "built_path": built_path,
            "queued_path": queued_path if queued_path.is_file() else None,
            "built": b,
            "queued": q,
        }
    return idx


def find_input_json_for_load(load_name: str, job_dir: Path, built_data: dict | None = None) -> Path | None:
    if built_data and built_data.get("job_file"):
        jf = REPO / str(built_data["job_file"])
        if jf.is_file():
            return jf
    cand1 = job_dir / "job.render.json"
    for p in sorted(DATA_DIR.glob(f"*/*{load_name}*.json")):
        if p.name == f"{load_name}.json" and "_job" not in p.parent.name:
            return p
    if cand1.is_file():
        return cand1
    return None


def find_capture_dir_for_load(load_name: str, built_data: dict | None = None) -> Path | None:
    if built_data and built_data.get("capture"):
        cp = Path(str(built_data["capture"]))
        if not cp.is_absolute():
            cp = REPO / cp
        if cp.is_dir():
            return cp
    matches = sorted(
        p for p in DATA_DIR.glob(f"*_{load_name}")
        if p.is_dir() and (p / "manifest.json").is_file()
    )
    return matches[-1] if matches else None


def find_preview_job_dir(preview_ref: str | None) -> Path | None:
    if not preview_ref:
        return None
    p = Path(preview_ref)
    if not p.is_absolute():
        p = REPO / p
    if p.is_dir():
        return p
    matches = sorted(DATA_DIR.glob(f"*{preview_ref}*_job"))
    return matches[-1] if matches else None


def load_today_previews() -> dict[tuple[int, int], list[dict]]:
    """Map (first, last) -> list of pre-built preview jobs from phase2 and phase3 folders."""
    out: dict[tuple[int, int], list[dict]] = {}
    for phase_label, folder_name in [
        ("Phase 2 (Subject 2 re-run)", "2026-10-20_fun_phase2_inputs"),
        ("Phase 3/4 (Unfinished shots)", "2026-10-20_fun_phase3_inputs"),
    ]:
        folder = DATA_DIR / folder_name
        if not folder.is_dir():
            continue
        for jpath in sorted(folder.glob("shot_c_*.json")):
            try:
                d = json.loads(jpath.read_text())
            except Exception:
                continue
            j = d.get("job") or {}
            first = int(j.get("first", 0))
            last = first + int(j.get("frames", 0)) - 1
            pname = j.get("name", jpath.stem)
            job_folder = DATA_DIR / f"2026-10-20_{pname}_job"
            queued_file = job_folder / "preview.queued.json"
            qdata = None
            found_files: dict[str, list[dict]] = {}
            if queued_file.is_file():
                q_mtime = path_mtime_local(queued_file)
                try:
                    qdata = json.loads(queued_file.read_text())
                    for subj_k, sdict in (qdata.get("found") or {}).items():
                        flist = []
                        if isinstance(sdict, dict):
                            for fkind, fpaths in sdict.items():
                                for fp in (fpaths if isinstance(fpaths, list) else [fpaths]):
                                    flist.append({
                                        "kind": fkind,
                                        "path": str(fp),
                                        "mtime": q_mtime,
                                    })
                        found_files[subj_k] = flist
                except Exception:
                    qdata = None
            out.setdefault((first, last), []).append({
                "phase": phase_label,
                "name": pname,
                "source": j.get("source", "input/fun_0000_6394_24p_sound_in_step.mov"),
                "input_json": jpath.relative_to(REPO),
                "input_mtime": path_mtime_local(jpath),
                "job_folder": job_folder.relative_to(REPO) if job_folder.exists() else None,
                "preview_done": queued_file.is_file(),
                "preview_done_mtime": path_mtime_local(queued_file) if queued_file.is_file() else None,
                "preview_commit": (qdata or {}).get("commit", "—"),
                "preview_seconds": (qdata or {}).get("seconds"),
                "found_files": found_files,
            })
    return out


def load_cast_index() -> list[dict]:
    cast_file = FUN_DIR / "cast" / "cast.jsonl"
    if not cast_file.is_file():
        return []
    out = []
    for line in cast_file.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def get_clothing_for_span(plan_data: dict, subject: str, first: int, last: int) -> dict | None:
    s_cfg = ((plan_data.get("subjects") or {}).get(subject)) or {}
    for c in (s_cfg.get("clothing") or []):
        c_first = int(c.get("first", -1))
        c_last = int(c.get("last", -1))
        if c_first <= last and c_last >= first:
            return c
    return None


def build_dag_steps_for_row(
    sh: dict,
    subject: str,
    lrows: list[dict],
    previews_here: list[dict],
    plan_data: dict,
    cast_rows: list[dict],
    built_idx: dict[str, dict],
    latest_joined: Path | None,
) -> list[dict]:
    """Build the 6-step hierarchical data flow lineage tree (DAG) for a (shot, subject) row."""
    first, last = int(sh["first"]), int(sh["last"])
    n_frames = last - first + 1
    who_info = ((sh.get("who") or {}).get(subject)) or {}

    if subject == "—":
        src_mov = INPUT_DIR / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov"
        return [
            {
                "step_num": 1,
                "title": "Step 1: Project Sources & Shot Table (`data/fun/shots.json`)",
                "code": "bench/shots_record.py · bench/page_answers_to_cast.py",
                "status": "done",
                "status_label": "No Subject Present — Original Source Kept",
                "items": [
                    {"k": "Shot Span", "v": f"Frames {first}–{last} ({n_frames}f, {fmt_tc(first)}–{fmt_tc(last)})"},
                    {"k": "Subject Presence (`shots.json`)", "v": "lead: no, second: no, third: no"},
                    {"k": "Source Video Plate", "v": f"{src_mov} (mtime: {path_mtime_local(src_mov)})"},
                    {"k": "Shot Note", "v": str(sh.get("note") or "—")},
                ],
            },
            {
                "step_num": 6,
                "title": "Step 6: Final Cut Assembly (`bench/rebuild_delivery.py` → `bench/assemble_delivery.py`)",
                "code": "bench/rebuild_delivery.py · bench/assemble_delivery.py",
                "status": "done",
                "status_label": "Passed Through Unchanged to Final Cut",
                "items": [
                    {"k": "Action at Assembly", "v": "Original source frames copied directly into final cut"},
                    {"k": "Latest Joined Cut MP4", "v": f"{latest_joined.name if latest_joined else '—'} (mtime: {path_mtime_local(latest_joined)})"},
                ],
            },
        ]

    subj_plan = ((plan_data.get("subjects") or {}).get(subject)) or {}
    still_rel = subj_plan.get("still", "—")
    still_abs = INPUT_DIR / still_rel if still_rel != "—" else None
    parts_list = subj_plan.get("parts") or []
    clothing_entry = get_clothing_for_span(plan_data, subject, first, last)

    # Matching cast seed boxes in [first, last]
    cast_matches = [
        c for c in cast_rows
        if c.get("subject") == subject and c.get("box") and first <= int(c.get("frame", -1)) <= last
    ]

    # Determine primary load if laid
    primary_laid = lrows[-1] if lrows else None
    primary_load_name = primary_laid["load"] if primary_laid else None
    binfo = built_idx.get(primary_load_name) if primary_load_name else None
    bdata = (binfo or {}).get("built") or {}
    qdata = (binfo or {}).get("queued") or {}
    in_json_path = find_input_json_for_load(primary_load_name, binfo["job_dir"], bdata) if (primary_load_name and binfo) else None
    in_json_data = {}
    if in_json_path and in_json_path.is_file():
        try:
            in_json_data = json.loads(in_json_path.read_text())
        except Exception:
            in_json_data = {}
    job_block = in_json_data.get("job") or {}

    # Source video used
    src_video_str = (
        job_block.get("source")
        or (bdata.get("job") or {}).get("source")
        or (bdata.get("load") or {}).get("source")
        or (previews_here[-1]["source"] if previews_here else "input/fun_0000_6394_24p_sound_in_step.mov")
    )
    src_video_path = Path(src_video_str)

    # ---------------- STEP 1: Project Sources & Plan ----------------
    step1_items = [
        {
            "k": "Source Video Fed In",
            "v": f"{src_video_path.name} ({src_video_path})",
            "ts": path_mtime_local(src_video_path),
        },
        {
            "k": "Subject Reference Still (`<Picture 1>`)",
            "v": f"{still_rel} ({still_abs})",
            "ts": path_mtime_local(still_abs),
        },
        {
            "k": "Target Region & Parts (`plan.json`)",
            "v": f"region={subj_plan.get('region', '—')} · parts={', '.join(parts_list)}",
            "ts": path_mtime_local(FUN_DIR / "plan.json"),
        },
        {
            "k": "Shot Presence (`data/fun/shots.json`)",
            "v": f"present={who_info.get('present', '—')} · how: {who_info.get('how', '—')} · boxed frames: {who_info.get('frames_boxed', [])}",
            "ts": path_mtime_local(FUN_DIR / "shots.json"),
        },
    ]
    if clothing_entry:
        step1_items.append({
            "k": "Clothing Words (`data/fun/plan.json`)",
            "v": f"\"{clothing_entry.get('words')}\" (spans {clothing_entry.get('first')}–{clothing_entry.get('last')})",
            "ts": fmt_iso_to_local(clothing_entry.get("by", "").split(",")[1].strip() if "," in clothing_entry.get("by", "") else None),
        })
    if cast_matches:
        boxes_str = " ; ".join(
            f"frame {c['frame']}: box={c['box']} (table={c.get('table', '—')})"
            for c in cast_matches
        )
        step1_items.append({
            "k": "Cast Seed Boxes (`data/fun/cast/cast.jsonl`)",
            "v": boxes_str,
            "ts": path_mtime_local(FUN_DIR / "cast" / "cast.jsonl"),
        })

    step1 = {
        "step_num": 1,
        "title": "Step 1: Project Sources, Cast & Plan (`shots.json` · `cast.jsonl` · `plan.json`)",
        "code": "data/fun/shots.json · data/fun/cast/cast.jsonl · data/fun/plan.json",
        "status": "done",
        "status_label": "Completed (Source & Plan Locked)",
        "items": step1_items,
    }

    # ---------------- STEP 2: Shot Preview & Tracker/Segmentation ----------------
    cap_dir = find_capture_dir_for_load(primary_load_name, bdata) if primary_load_name else None
    manifest_data = {}
    if cap_dir and (cap_dir / "manifest.json").is_file():
        try:
            manifest_data = json.loads((cap_dir / "manifest.json").read_text())
        except Exception:
            manifest_data = {}

    step2_items = []
    step2_status = "pending"
    step2_label = "Pending Preview"

    # Check preview from job_block / manifest / today_previews
    preview_ref = job_block.get("preview")
    preview_jdir = find_preview_job_dir(preview_ref)
    if preview_jdir or manifest_data:
        step2_status = "done"
        p_name = preview_jdir.name if preview_jdir else (preview_ref or "preview")
        step2_label = f"Completed ({p_name})"
        if preview_jdir:
            pq = preview_jdir / "preview.queued.json"
            pb = preview_jdir / "preview.built.json"
            pq_data = {}
            if pq.is_file():
                try:
                    pq_data = json.loads(pq.read_text())
                except Exception:
                    pq_data = {}
            step2_items.append({
                "k": "Preview Job Folder",
                "v": f"{rel_or_str(preview_jdir)} (commit: {pq_data.get('commit', '—')}, duration: {pq_data.get('seconds', '—')}s)",
                "ts": path_mtime_local(pq if pq.is_file() else pb),
            })
        # Extract sightings / tracker files from manifest.json
        for s_entry in (manifest_data.get("subjects") or []):
            s_lbl = s_entry.get("label")
            for sight in (s_entry.get("sightings") or []):
                by_prev = sight.get("by", "—")
                inputs_dict = sight.get("inputs") or {}
                for ikind, idata in inputs_dict.items():
                    if isinstance(idata, dict) and idata.get("file"):
                        step2_items.append({
                            "k": f"Tracker/Part Output (`{s_lbl}` · `{ikind}`)",
                            "v": f"{idata['file']} (preview: {by_prev}, frames: {idata.get('frames', '—')})",
                            "ts": fmt_iso_to_local(idata.get("written")),
                        })
                if sight.get("pose") and isinstance(sight["pose"], dict) and sight["pose"].get("file"):
                    step2_items.append({
                        "k": f"3D Body/Hand Pose Table (`{s_lbl}`)",
                        "v": f"{sight['pose']['file']} (hand_refinement={sight['pose'].get('hand_refinement')})",
                        "ts": path_mtime_local(cap_dir / "manifest.json") if cap_dir else "—",
                    })

    # Also include any today's Phase 2 / Phase 3 preview jobs on this shot
    for qn in previews_here:
        pstat = f"Preview Finished ({qn['preview_done_mtime']})" if qn["preview_done"] else "Preview Input Built (Queued)"
        if step2_status != "done":
            step2_status = "done" if qn["preview_done"] else "ready"
            step2_label = pstat
        step2_items.append({
            "k": f"{qn['phase']} Preview Job (`{qn['name']}`)",
            "v": f"input: {qn['input_json']} → job_dir: {qn['job_folder'] or 'not yet queued'} ({pstat})",
            "ts": qn["preview_done_mtime"] or qn["input_mtime"],
        })
        for subj_k, flist in (qn.get("found_files") or {}).items():
            for fitem in flist:
                step2_items.append({
                    "k": f"New Preview File (`{subj_k}` · `{fitem['kind']}`)",
                    "v": fitem["path"],
                    "ts": fitem["mtime"],
                })

    step2 = {
        "step_num": 2,
        "title": "Step 2: Shot Preview, SAM 3.1 Tracking & SapiensSegmentation (`shot_to_load.py preview` → `masked_job.py queue preview`)",
        "code": "bench/shot_to_load.py preview · bench/masked_job.py build/queue preview (SAM31Tracker, Sapiens2, SAM3DBody)",
        "status": step2_status,
        "status_label": step2_label,
        "items": step2_items if step2_items else [{"k": "Status", "v": "No preview record linked", "ts": "—"}],
    }

    # ---------------- STEP 3: Load Job, Head Reference <Video 1> & Prompt ----------------
    step3_items = []
    step3_status = "pending"
    step3_label = "Pending Load Job (`shot_to_load.py job → reference → text`)"
    prompt_text_full = None

    if primary_load_name:
        step3_status = "done"
        step3_label = f"Completed (`{primary_load_name}`)"
        load_cfg = bdata.get("load") or {}
        if in_json_path:
            step3_items.append({
                "k": "Load Job JSON",
                "v": rel_or_str(in_json_path),
                "ts": path_mtime_local(in_json_path),
            })
        # Find subject config in job_block
        subj_job_cfg = next(
            (s for s in (job_block.get("subjects") or []) if s.get("label") == subject),
            {},
        )
        rules_bits = []
        for rk in ("least", "drop", "word", "parts_held_to_track_px"):
            if subj_job_cfg.get(rk) is not None:
                rules_bits.append(f"{rk}={subj_job_cfg[rk]}")
        for lk in ("carried", "render_region", "margin", "feather", "edge", "others"):
            if load_cfg.get(lk) is not None:
                rules_bits.append(f"{lk}={load_cfg[lk]}")
        if rules_bits:
            step3_items.append({
                "k": "Load Rules & Mask Conditioning",
                "v": " · ".join(rules_bits),
                "ts": path_mtime_local(in_json_path) if in_json_path else "—",
            })

        motion_file = load_cfg.get("motion_file")
        if motion_file:
            mf_path = Path(motion_file)
            ref_json = mf_path.with_suffix(".json")
            step3_items.append({
                "k": "Head Reference Video (`<Video 1>`)",
                "v": f"{motion_file} (motion_edge={load_cfg.get('motion_edge', 768)}, motion_first={load_cfg.get('motion_first', first)})",
                "ts": path_mtime_local(in_json_path) if in_json_path else "—",
            })
            step3_items.append({
                "k": "Head Reference Provenance JSON",
                "v": str(ref_json),
                "ts": path_mtime_local(in_json_path) if in_json_path else "—",
            })

        text_block = bdata.get("text") or {}
        txt_file_rel = text_block.get("file")
        txt_path = (REPO / txt_file_rel) if txt_file_rel else (in_json_path.with_suffix(".txt") if in_json_path else None)
        if txt_path and txt_path.is_file():
            try:
                prompt_text_full = txt_path.read_text()
            except Exception:
                prompt_text_full = text_block.get("text")
            step3_items.append({
                "k": "Prompt Text File (`<load>.txt`)",
                "v": f"{rel_or_str(txt_path)} (sha256: {(text_block.get('sha256') or '')[:16]}...)",
                "ts": path_mtime_local(txt_path),
            })
            if text_block.get("by"):
                step3_items.append({
                    "k": "Prompt Provenance (`by`)",
                    "v": str(text_block.get("by")),
                    "ts": "—",
                })
        elif text_block.get("text"):
            prompt_text_full = text_block.get("text")

    step3 = {
        "step_num": 3,
        "title": "Step 3: Load Job, 3D Head Reference `<Video 1>` & Prompt (`shot_to_load.py job` → `reference` → `text`)",
        "code": "bench/shot_to_load.py · bench/pose_along_shot.py · bench/head_reference.py · bench/load_text.py",
        "status": step3_status,
        "status_label": step3_label,
        "items": step3_items if step3_items else [{"k": "Next Pipeline Step", "v": "Run `shot_to_load.py job → reference → text` on completed preview", "ts": "—"}],
        "prompt_text": prompt_text_full,
    }

    # ---------------- STEP 4: Capture, Window Plan & Gate Verification ----------------
    step4_items = []
    step4_status = "pending"
    step4_label = "Pending Capture & Plan (`load_chain.py first → plan → on-plan → table`)"

    if cap_dir and cap_dir.is_dir():
        step4_status = "done"
        flags_path = cap_dir / "flags.json"
        flags_data = {}
        if flags_path.is_file():
            try:
                flags_data = json.loads(flags_path.read_text())
            except Exception:
                flags_data = {}
        verdict = flags_data.get("verdict") or ((qdata.get("gate") or {}).get("verdict")) or "clear"
        overridden = flags_data.get("overridden") or ((qdata.get("gate") or {}).get("overridden")) or []
        step4_label = f"Completed (`{cap_dir.name}` · gate={verdict})"
        step4_items.append({
            "k": "Capture Directory & Manifest",
            "v": f"{rel_or_str(cap_dir / 'manifest.json')} (span: {manifest_data.get('first_frame', first)}..{int(manifest_data.get('first_frame', first)) + int(manifest_data.get('frames', n_frames)) - 1}, {manifest_data.get('frames', n_frames)}f)",
            "ts": path_mtime_local(cap_dir / "manifest.json"),
        })
        if flags_path.is_file():
            held_cnt = len((flags_data.get("held_frames") or {}).get(subject) or [])
            emptied_cnt = (flags_data.get("emptied_by_the_grade") or {}).get(subject, 0)
            step4_items.append({
                "k": "Gate Verification (`flags.json`)",
                "v": f"verdict={verdict} · overridden={overridden} · held_frames={held_cnt} · emptied_by_grade={emptied_cnt}",
                "ts": path_mtime_local(flags_path),
            })
        ft_md = cap_dir / "frame_table.md"
        if ft_md.is_file():
            step4_items.append({
                "k": "Per-Frame Audit Table (`frame_table.md`)",
                "v": rel_or_str(ft_md),
                "ts": path_mtime_local(ft_md),
            })
        for f_mkv in sorted((cap_dir / "subjects").glob("*/*.mkv")) if (cap_dir / "subjects").is_dir() else []:
            step4_items.append({
                "k": f"Prepared Capture Matte (`{f_mkv.parent.name}`)",
                "v": rel_or_str(f_mkv),
                "ts": path_mtime_local(f_mkv),
            })

    step4 = {
        "step_num": 4,
        "title": "Step 4: Capture Matte Preparation, Window Plan & Gate (`load_chain.py first` → `plan` → `on-plan` → `table`)",
        "code": "bench/load_chain.py · bench/capture_masked_run.py files · bench/masked_job.py queue plan · bench/frame_table.py",
        "status": step4_status,
        "status_label": step4_label,
        "items": step4_items if step4_items else [{"k": "Next Pipeline Step", "v": "Run `load_chain.py first → plan → on-plan → table`", "ts": "—"}],
    }

    # ---------------- STEP 5: H3 DiT Render & Landing ----------------
    step5_items = []
    step5_status = "pending"
    step5_label = "Pending H3 DiT Render (`load_chain.py render → land`)"

    if primary_laid:
        step5_status = "done"
        for lr in lrows:
            lname = lr["load"]
            lbinfo = built_idx.get(lname)
            mp4_abs = SHARE_DIR / lr["file"]
            step5_items.append({
                "k": f"Final Rendered Clip MP4 (`{lname}`)",
                "v": f"{lr['file']} (sha256: {lr.get('sha256', '')[:16]}...)",
                "ts": path_mtime_local(mp4_abs),
            })
            if lbinfo:
                lb = lbinfo.get("built") or {}
                lq = lbinfo.get("queued") or {}
                step5_items.append({
                    "k": f"Render Build Record (`{lname}.render.built.json`)",
                    "v": f"{rel_or_str(lbinfo['built_path'])} (commit: {lb.get('commit', '—')}, seed: {(lb.get('load') or {}).get('seed', '—')})",
                    "ts": path_mtime_local(lbinfo["built_path"]),
                })
                if lbinfo.get("queued_path"):
                    step5_items.append({
                        "k": f"Render Queue Record (`{lname}.render.queued.json`)",
                        "v": f"{rel_or_str(lbinfo['queued_path'])} (status: {lq.get('status', '—')}, wall_time: {lq.get('seconds', '—')}s, prompt_id: {lq.get('prompt_id', '—')})",
                        "ts": path_mtime_local(lbinfo["queued_path"]),
                    })
                if (lq.get("look") or {}).get("by"):
                    lk = lq["look"]
                    step5_items.append({
                        "k": "Pre-Render 1fps Strip Inspection (`look`)",
                        "v": f"by {lk.get('by')} at {fmt_iso_to_local(lk.get('at'))}: {lk.get('note', '')[:180]}...",
                        "ts": fmt_iso_to_local(lk.get("at")),
                    })
                win_files = lq.get("windows_files") or []
                if win_files:
                    step5_items.append({
                        "k": "Per-Window Outputs & Region NPZs",
                        "v": ", ".join(win_files),
                        "ts": path_mtime_local(mp4_abs),
                    })
                digests = lq.get("digests") or {}
                if digests:
                    dig_str = " · ".join(f"{Path(k).name}: {v[:12]}" for k, v in digests.items())
                    step5_items.append({
                        "k": "Verified Input SHA256 Digests at Queue Time",
                        "v": dig_str,
                        "ts": fmt_iso_to_local(lq.get("queued_at")),
                    })
        step5_label = f"Rendered ({', '.join(r['load'] for r in lrows)})"

    step5 = {
        "step_num": 5,
        "title": "Step 5: MiniMax H3 DiT Render & Land (`load_chain.py render` → `land`)",
        "code": "MiniMaxH3V2VSong · bench/masked_job.py build/queue render · bench/capture_masked_run.py video",
        "status": step5_status,
        "status_label": step5_label,
        "items": step5_items if step5_items else [{"k": "Next Pipeline Step", "v": "Inspect `.1fps.` strip (`what_h3_got.py`), then `load_chain.py render → land`", "ts": "—"}],
    }

    # ---------------- STEP 6: Laid in Final Cut ----------------
    step6_items = []
    step6_status = "pending"
    step6_label = "Not Yet Laid in Final Cut (`laid.jsonl`)"

    if lrows:
        step6_status = "done"
        step6_label = f"Laid in Cut ({lrows[-1].get('owner', 'in cut')})"
        for lr in lrows:
            step6_items.append({
                "k": f"Ledger Entry (`data/fun/laid.jsonl` · `{lr['load']}`)",
                "v": f"frames={lr.get('frames')} · owner={lr.get('owner')} · added_by={lr.get('added_by')}",
                "ts": fmt_iso_to_local(lr.get("added")),
            })
            step6_items.append({
                "k": f"Verdict & Provenance (`by`)",
                "v": str(lr.get("by", "—")),
                "ts": fmt_iso_to_local(lr.get("added")),
            })
            if lr.get("replaces"):
                step6_items.append({
                    "k": "Replaces Earlier Render",
                    "v": str(lr.get("replaces")),
                    "ts": "—",
                })
        if latest_joined:
            step6_items.append({
                "k": "Assembled Full Delivery Cut MP4",
                "v": f"{latest_joined.name} ({latest_joined})",
                "ts": path_mtime_local(latest_joined),
            })

    step6 = {
        "step_num": 6,
        "title": "Step 6: Laid in Ledger & Full Cut Assembly (`bench/laid.py` → `bench/rebuild_delivery.py` → `bench/assemble_delivery.py`)",
        "code": "data/fun/laid.jsonl · bench/laid.py · bench/rebuild_delivery.py · bench/assemble_delivery.py",
        "status": step6_status,
        "status_label": step6_label,
        "items": step6_items if step6_items else [{"k": "Next Pipeline Step", "v": "Add to `data/fun/laid.jsonl` and rebuild delivery cut", "ts": "—"}],
    }

    return [step1, step2, step3, step4, step5, step6]


def render_dag_html(steps: list[dict], row_id: str) -> str:
    """Render the visual 6-step DAG lineage tree HTML for a row."""
    blocks = []
    for st in steps:
        st_cls = {
            "done": "dag-done",
            "ready": "dag-ready",
            "pending": "dag-pending",
        }.get(st["status"], "dag-pending")
        icon = "✓" if st["status"] == "done" else ("⏳" if st["status"] == "ready" else "○")

        kv_rows = []
        for it in st.get("items") or []:
            ts_html = f"<span class='dag-ts'>{html.escape(str(it['ts']))}</span>" if it.get("ts") and it["ts"] != "—" else ""
            kv_rows.append(
                f"<div class='dag-kv'>"
                f"<div class='dag-k'>{html.escape(str(it['k']))}</div>"
                f"<div class='dag-v'><code>{html.escape(str(it['v']))}</code> {ts_html}</div>"
                f"</div>"
            )

        prompt_html = ""
        if st.get("prompt_text"):
            prompt_html = (
                f"<details class='prompt-details'>"
                f"<summary>▶ View Full H3 Prompt Text (<code>&lt;load&gt;.txt</code>)</summary>"
                f"<pre class='prompt-pre'>{html.escape(str(st['prompt_text']))}</pre>"
                f"</details>"
            )

        blocks.append(
            f"<div class='dag-step {st_cls}'>"
            f"  <div class='dag-node-col'>"
            f"    <div class='dag-circle'>{icon}</div>"
            f"    <div class='dag-line'></div>"
            f"  </div>"
            f"  <div class='dag-card'>"
            f"    <div class='dag-card-hdr'>"
            f"      <span class='dag-step-title'>{html.escape(st['title'])}</span>"
            f"      <span class='dag-step-badge {st_cls}-badge'>{html.escape(st['status_label'])}</span>"
            f"    </div>"
            f"    <div class='dag-code-line'>Tools / Files: <code>{html.escape(st['code'])}</code></div>"
            f"    <div class='dag-kv-list'>{''.join(kv_rows)}</div>"
            f"    {prompt_html}"
            f"  </div>"
            f"</div>"
        )
    return f"<div class='dag-tree' id='dag-{row_id}'>{''.join(blocks)}</div>"


def load_frame_coverage_index() -> tuple[dict[tuple[int, int, str], dict], Path | None]:
    """Load the newest frame_coverage_fun_0000_6394_*.json keyed by (first, last, subject)."""
    files = sorted(
        DATA_DIR.glob("*_fun_delivery_tables/frame_coverage_fun_0000_6394_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return {}, None
    latest_path = files[-1]
    try:
        data = json.loads(latest_path.read_text())
    except Exception:
        return {}, latest_path
    idx: dict[tuple[int, int, str], dict] = {}
    for sh in data.get("shots") or []:
        shot_pair = sh.get("shot") or [0, 0]
        if len(shot_pair) == 2:
            key = (int(shot_pair[0]), int(shot_pair[1]), str(sh.get("subject")))
            idx[key] = sh
    return idx, latest_path


def classify_frame_cat(step_name: str, state_str: str) -> str:
    st = str(state_str or "")
    sn = str(step_name or "")
    if st.startswith("carried (held"):
        return "held"
    if st.startswith("carried"):
        return "carried"
    if "emptied inside" in st or "far below" in st or "barely changed" in st or sn in ("emptied_inside", "region_far_below_its_own_size", "laid_but_barely_changed"):
        return "emptied"
    if sn in ("captured_not_rendered", "rendered_not_kept") or st.startswith("rendered, not kept"):
        return "unrendered_gap"
    return "no_capture"


def build_frame_coverage_for_row(
    first: int,
    last: int,
    subject: str,
    lrows: list[dict],
    cov_idx: dict[tuple[int, int, str], dict],
    built_idx: dict[str, dict],
) -> dict:
    """Build per-frame coverage array, contiguous spans, and gap summary for (first, last, subject)."""
    n_frames = last - first + 1
    if subject == "—":
        return {
            "has_data": False,
            "frames": [],
            "spans": [],
            "counts": {"carried": 0, "held": 0, "emptied": 0, "unrendered_gap": 0, "no_capture": n_frames},
            "gap_summary_parts": [],
            "short_summary": f"{n_frames}f source pass-through",
        }

    sh_cov = cov_idx.get((first, last, subject))
    laid_loads = [r["load"] for r in lrows]
    cov_pieces = sorted({str(p) for p in (sh_cov.get("piece") or []) if p}) if sh_cov else []

    # Check if laid.jsonl has a newer load than what was in frame_coverage_*.json
    use_live_overlay = False
    if lrows and (not sh_cov or set(laid_loads) != set(cov_pieces)):
        use_live_overlay = True

    per_frame: list[dict] = []
    if sh_cov and not use_live_overlay:
        frames_list = sh_cov.get("frames") or list(range(first, last + 1))
        step_names = sh_cov.get("step_name") or ["no_capture"] * len(frames_list)
        states = sh_cov.get("state") or ["original: no piece laid"] * len(frames_list)
        pieces = sh_cov.get("piece") or [None] * len(frames_list)
        track_pxs = sh_cov.get("track_px") or [None] * len(frames_list)
        track_froms = sh_cov.get("track_from") or ["—"] * len(frames_list)
        carried_pxs = sh_cov.get("carried_px") or [None] * len(frames_list)
        changed_shares = sh_cov.get("changed_share") or [None] * len(frames_list)
        for i, fnum in enumerate(frames_list):
            sn = str(step_names[i]) if i < len(step_names) else "no_capture"
            st = str(states[i]) if i < len(states) else "—"
            cat = classify_frame_cat(sn, st)
            per_frame.append({
                "f": int(fnum),
                "step_name": sn,
                "state": st,
                "piece": pieces[i] if i < len(pieces) and pieces[i] else "—",
                "track_px": track_pxs[i] if i < len(track_pxs) else None,
                "track_from": str(track_froms[i]) if i < len(track_froms) and track_froms[i] else "—",
                "carried_px": carried_pxs[i] if i < len(carried_pxs) else None,
                "changed_share": changed_shares[i] if i < len(changed_shares) else None,
                "cat": cat,
            })
    else:
        # Synthesize or overlay live manifest.json + flags.json from current lrows on top of sh_cov
        base_map: dict[int, dict] = {}
        if sh_cov:
            frames_list = sh_cov.get("frames") or list(range(first, last + 1))
            step_names = sh_cov.get("step_name") or ["no_capture"] * len(frames_list)
            states = sh_cov.get("state") or ["original: no piece laid"] * len(frames_list)
            pieces = sh_cov.get("piece") or [None] * len(frames_list)
            track_pxs = sh_cov.get("track_px") or [None] * len(frames_list)
            track_froms = sh_cov.get("track_from") or ["—"] * len(frames_list)
            for i, fnum in enumerate(frames_list):
                base_map[int(fnum)] = {
                    "f": int(fnum),
                    "step_name": str(step_names[i]),
                    "state": str(states[i]),
                    "piece": pieces[i] or "—",
                    "track_px": track_pxs[i],
                    "track_from": str(track_froms[i] or "—"),
                    "carried_px": None,
                    "changed_share": None,
                }
        for fnum in range(first, last + 1):
            if fnum not in base_map:
                base_map[fnum] = {
                    "f": fnum,
                    "step_name": "no_capture",
                    "state": "original: no piece laid",
                    "piece": "—",
                    "track_px": None,
                    "track_from": "—",
                    "carried_px": None,
                    "changed_share": None,
                }
            elif lrows:
                # Reset old piece before overlaying current lrows
                base_map[fnum]["piece"] = "—"
                if base_map[fnum]["step_name"] == "kept_and_laid":
                    base_map[fnum]["step_name"] = "captured_not_rendered"
                    base_map[fnum]["state"] = "original: no piece laid"

        for r in lrows:
            lname = r["load"]
            binfo = built_idx.get(lname)
            cap_dir = find_capture_dir_for_load(lname, (binfo or {}).get("built"))
            c_first = int(r.get("first", first))
            c_last = int(r.get("last", last))
            emptied_set: set[int] = set()
            held_set: set[int] = set()
            if cap_dir and (cap_dir / "manifest.json").is_file():
                try:
                    m = json.loads((cap_dir / "manifest.json").read_text())
                    c_first = int(m.get("first_frame", m.get("first", c_first)))
                    c_last = c_first + int(m.get("frames", c_last - c_first + 1)) - 1
                except Exception:
                    pass
                if (cap_dir / "flags.json").is_file():
                    try:
                        fl = json.loads((cap_dir / "flags.json").read_text())
                        hf = fl.get("held_frames")
                        if isinstance(hf, dict):
                            held_set = {int(x) for x in (hf.get(subject) or [])}
                        elif isinstance(hf, list):
                            held_set = {int(x) for x in hf}
                    except Exception:
                        pass
            # If this load intentionally starts at c_first > first or ends at c_last < last, frames outside c_first..c_last are left_out_by_a_look (off-screen at shot edge)
            if c_first > first or c_last < last:
                for fnum in range(first, last + 1):
                    if fnum < c_first or fnum > c_last:
                        base_map[fnum]["step_name"] = "left_out_by_a_look"
                        base_map[fnum]["state"] = "left out by a look (subject off-screen at shot edge)"
            for fnum in range(max(first, c_first), min(last, c_last) + 1):
                entry = base_map[fnum]
                entry["piece"] = lname
                entry["step_name"] = "kept_and_laid"
                if fnum in emptied_set:
                    entry["state"] = "original: emptied inside the piece (flags.json)"
                elif fnum in held_set:
                    entry["state"] = "carried (held from a neighbouring frame)"
                else:
                    entry["state"] = "carried"
                if cap_dir:
                    entry["track_from"] = cap_dir.name

        for fnum in range(first, last + 1):
            entry = base_map[fnum]
            entry["cat"] = classify_frame_cat(entry["step_name"], entry["state"])
            per_frame.append(entry)

    # Group into contiguous spans
    spans: list[dict] = []
    for pf in per_frame:
        if (
            spans
            and spans[-1]["cat"] == pf["cat"]
            and spans[-1]["step_name"] == pf["step_name"]
            and spans[-1]["state"] == pf["state"]
            and spans[-1]["piece"] == pf["piece"]
        ):
            sp = spans[-1]
            sp["f1"] = pf["f"]
            sp["count"] += 1
            if pf["track_px"] is not None:
                sp["track_px_vals"].append(int(pf["track_px"]))
            if pf["carried_px"] is not None:
                sp["carried_px_vals"].append(int(pf["carried_px"]))
            if pf["changed_share"] is not None:
                sp["changed_vals"].append(float(pf["changed_share"]))
        else:
            spans.append({
                "f0": pf["f"],
                "f1": pf["f"],
                "count": 1,
                "cat": pf["cat"],
                "step_name": pf["step_name"],
                "state": pf["state"],
                "piece": pf["piece"],
                "track_from": pf["track_from"],
                "track_px_vals": [int(pf["track_px"])] if pf["track_px"] is not None else [],
                "carried_px_vals": [int(pf["carried_px"])] if pf["carried_px"] is not None else [],
                "changed_vals": [float(pf["changed_share"])] if pf["changed_share"] is not None else [],
            })

    counts = {"carried": 0, "held": 0, "emptied": 0, "unrendered_gap": 0, "no_capture": 0}
    for pf in per_frame:
        counts[pf["cat"]] = counts.get(pf["cat"], 0) + 1

    # Gap summary parts (human readable with exact frame numbers)
    gap_parts = []
    unren_spans = [f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}" for sp in spans if sp["cat"] == "unrendered_gap"]
    emptied_spans = [f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}" for sp in spans if sp["cat"] == "emptied"]
    nocap_spans = [f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}" for sp in spans if sp["cat"] == "no_capture"]

    if counts["unrendered_gap"] > 0:
        gap_parts.append(f"{counts['unrendered_gap']}f unrendered gap ({', '.join(unren_spans[:3])}{'…' if len(unren_spans) > 3 else ''})")
    if counts["emptied"] > 0:
        gap_parts.append(f"{counts['emptied']}f emptied inside ({', '.join(emptied_spans[:3])}{'…' if len(emptied_spans) > 3 else ''})")
    if counts["no_capture"] > 0 and lrows:
        gap_parts.append(f"{counts['no_capture']}f no track/off-screen ({', '.join(nocap_spans[:2])}{'…' if len(nocap_spans) > 2 else ''})")

    total_carried = counts["carried"] + counts["held"]
    short_parts = [f"{total_carried}/{n_frames}f carried"]
    if counts["unrendered_gap"] > 0:
        short_parts.append(f"{counts['unrendered_gap']}f unrendered ({', '.join(unren_spans[:2])})")
    if counts["emptied"] > 0:
        short_parts.append(f"{counts['emptied']}f emptied")
    if counts["no_capture"] > 0 and lrows:
        short_parts.append(f"{counts['no_capture']}f untrk ({', '.join(nocap_spans[:2])})")

    return {
        "has_data": True,
        "live_overlay": use_live_overlay,
        "frames": per_frame,
        "spans": spans,
        "counts": counts,
        "gap_summary_parts": gap_parts,
        "short_summary": " · ".join(short_parts),
    }


def render_mini_barcode_html(cov: dict, n_frames: int, row_id: str) -> str:
    """Render a compact horizontal frame-coverage barcode + 1-line summary for the main table cell."""
    if not cov.get("has_data") or not cov.get("spans"):
        return ""
    segs = []
    for sp in cov["spans"]:
        pct = max(0.8, (sp["count"] / max(1, n_frames)) * 100.0)
        cat_cls = f"fcat-{sp['cat']}"
        rng = f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}"
        tip = f"Frames {rng} ({sp['count']}f): {sp['step_name']} — {sp['state']} [piece: {sp['piece']}]"
        segs.append(
            f"<div class='fbar-seg {cat_cls}' style='width:{pct:.2f}%' title='{html.escape(tip)}'></div>"
        )
    return (
        f"<div class='mini-cov-wrap' onclick=\"toggleDrawer('{row_id}')\" title='Click to open Frame-Level Coverage &amp; Lineage Drill-Down'>"
        f"  <div class='fbar-mini'>{''.join(segs)}</div>"
        f"  <div class='mini-cov-txt mono'>{html.escape(cov['short_summary'])}</div>"
        f"</div>"
    )


def render_frame_drilldown_html(cov: dict, first: int, last: int, subject: str, row_id: str) -> str:
    """Render the full Frame-Level Coverage & Gap Drill-Down panel (visual timeline + span table + per-frame table)."""
    if not cov.get("has_data") or not cov.get("spans"):
        return ""
    n_frames = last - first + 1
    c = cov["counts"]
    total_carried = c["carried"] + c["held"]

    # Build visual timeline strip
    segs = []
    for sp in cov["spans"]:
        pct = max(1.0, (sp["count"] / max(1, n_frames)) * 100.0)
        cat_cls = f"fcat-{sp['cat']}"
        rng = f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}"
        tip = (
            f"Frames {rng} ({sp['count']}f, {fmt_tc(sp['f0'])}–{fmt_tc(sp['f1'])})\n"
            f"Step: {sp['step_name']}\n"
            f"State: {sp['state']}\n"
            f"Laid Piece: {sp['piece']}"
        )
        label_inside = rng if sp["count"] / max(1, n_frames) >= 0.12 else ""
        segs.append(
            f"<div class='fbar-lg-seg {cat_cls}' style='width:{pct:.2f}%' title='{html.escape(tip)}'>"
            f"<span>{html.escape(label_inside)}</span>"
            f"</div>"
        )

    # Build contiguous span rows
    span_rows = []
    for sp in cov["spans"]:
        rng = f"{sp['f0']}–{sp['f1']}" if sp["f0"] != sp["f1"] else f"{sp['f0']}"
        tc_rng = f"{fmt_tc(sp['f0'])}–{fmt_tc(sp['f1'])}" if sp["f0"] != sp["f1"] else fmt_tc(sp["f0"])
        cat_badge_cls = {
            "carried": "badge-green",
            "held": "badge-blue",
            "emptied": "badge-amber",
            "unrendered_gap": "badge-red",
            "no_capture": "badge-gray",
        }.get(sp["cat"], "badge-gray")
        cat_label = {
            "carried": "CARRIED IN CUT",
            "held": "CARRIED (HELD)",
            "emptied": "EMPTIED INSIDE PIECE",
            "unrendered_gap": "GAP: TRACKED / NOT IN CUT",
            "no_capture": "NO TRACK / OFF-SCREEN",
        }.get(sp["cat"], sp["cat"])

        trk_str = "—"
        if sp["track_px_vals"]:
            mn, mx = min(sp["track_px_vals"]), max(sp["track_px_vals"])
            trk_str = f"{mn:,} px" if mn == mx else f"{mn:,}–{mx:,} px"
        car_str = "—"
        if sp["carried_px_vals"]:
            mn, mx = min(sp["carried_px_vals"]), max(sp["carried_px_vals"])
            car_str = f"{mn:,} px" if mn == mx else f"{mn:,}–{mx:,} px"
        chg_str = "—"
        if sp["changed_vals"]:
            avg_c = sum(sp["changed_vals"]) / len(sp["changed_vals"])
            chg_str = f"{avg_c:.2f}%"

        span_rows.append(
            f"<tr>"
            f"<td class='mono'><strong>{rng}</strong> <span class='sub'>({tc_rng})</span></td>"
            f"<td class='mono'>{sp['count']}f</td>"
            f"<td><span class='badge {cat_badge_cls}'>{html.escape(cat_label)}</span> <code style='margin-left:4px'>{html.escape(sp['step_name'])}</code></td>"
            f"<td>{html.escape(sp['state'])}</td>"
            f"<td class='mono'><code>{html.escape(str(sp['piece']))}</code></td>"
            f"<td class='mono'>{html.escape(trk_str)} <span class='sub'>({html.escape(sp['track_from'])})</span></td>"
            f"<td class='mono'>{html.escape(car_str)}</td>"
            f"<td class='mono'>{html.escape(chg_str)}</td>"
            f"</tr>"
        )

    # Build expandable per-frame table
    frame_rows = []
    for pf in cov["frames"]:
        cat_badge_cls = {
            "carried": "badge-green",
            "held": "badge-blue",
            "emptied": "badge-amber",
            "unrendered_gap": "badge-red",
            "no_capture": "badge-gray",
        }.get(pf["cat"], "badge-gray")
        trk_s = f"{int(pf['track_px']):,} px" if pf["track_px"] is not None else "—"
        car_s = f"{int(pf['carried_px']):,} px" if pf["carried_px"] is not None else "—"
        chg_s = f"{float(pf['changed_share']):.2f}%" if pf["changed_share"] is not None else "—"
        frame_rows.append(
            f"<tr>"
            f"<td class='mono'><strong>{pf['f']}</strong> <span class='sub'>({fmt_tc(pf['f'])})</span></td>"
            f"<td><span class='badge {cat_badge_cls}'>{html.escape(pf['step_name'])}</span></td>"
            f"<td>{html.escape(pf['state'])}</td>"
            f"<td class='mono'><code>{html.escape(str(pf['piece']))}</code></td>"
            f"<td class='mono'>{trk_s} <span class='sub'>({html.escape(pf['track_from'])})</span></td>"
            f"<td class='mono'>{car_s}</td>"
            f"<td class='mono'>{chg_s}</td>"
            f"</tr>"
        )

    overlay_note = (
        " <span class='badge badge-blue' style='margin-left:6px'>Live Overlay from Today's Laid Manifest</span>"
        if cov.get("live_overlay")
        else ""
    )

    return (
        f"<div class='cov-panel' id='cov-{row_id}'>"
        f"  <div class='cov-hdr'>"
        f"    <div>"
        f"      <strong>Frame-Level Coverage &amp; Gap Drill-Down — Frames <code>{first}–{last}</code> ({n_frames}f) · Subject: <code>{html.escape(subject)}</code></strong>{overlay_note}"
        f"      <div class='sub'>Exact same per-frame audit used by <code>bench/frame_coverage_report.py</code> &amp; <code>bench/rebuild_delivery.py</code></div>"
        f"    </div>"
        f"    <div class='cov-legend'>"
        f"      <span class='leg-item'><i class='leg-dot fcat-carried'></i> Carried ({total_carried}f)</span>"
        f"      <span class='leg-item'><i class='leg-dot fcat-emptied'></i> Emptied Inside ({c['emptied']}f)</span>"
        f"      <span class='leg-item'><i class='leg-dot fcat-unrendered_gap'></i> Unrendered / Unkept Gap ({c['unrendered_gap']}f)</span>"
        f"      <span class='leg-item'><i class='leg-dot fcat-no_capture'></i> No Track / Off-Screen ({c['no_capture']}f)</span>"
        f"    </div>"
        f"  </div>"
        f"  <div class='fbar-lg-wrap'>"
        f"    <div class='fbar-lg-scale'><span>f{first} ({fmt_tc(first)})</span><span>Hover any segment for frame details</span><span>f{last} ({fmt_tc(last)})</span></div>"
        f"    <div class='fbar-lg'>{''.join(segs)}</div>"
        f"  </div>"
        f"  <table class='cov-span-table'>"
        f"    <thead>"
        f"      <tr>"
        f"        <th style='width:150px;'>Frame Span &amp; TC</th>"
        f"        <th style='width:55px;'>Len</th>"
        f"        <th style='width:240px;'>Coverage Status &amp; Step</th>"
        f"        <th>Detailed Frame State / Gap Reason</th>"
        f"        <th style='width:140px;'>Laid Piece</th>"
        f"        <th style='width:210px;'>Tracked Mask Area (<code>track_px</code>)</th>"
        f"        <th style='width:110px;'>Carried Area</th>"
        f"        <th style='width:80px;'>Changed</th>"
        f"      </tr>"
        f"    </thead>"
        f"    <tbody>{''.join(span_rows)}</tbody>"
        f"  </table>"
        f"  <details class='prompt-details' style='margin-top:10px;'>"
        f"    <summary>▶ Expand Single-Frame Table (All {n_frames} Individual Frames {first}..{last})</summary>"
        f"    <div style='max-height:360px; overflow-y:auto; margin-top:8px;'>"
        f"      <table class='cov-span-table'>"
        f"        <thead>"
        f"          <tr>"
        f"            <th>Frame &amp; TC</th>"
        f"            <th>Step</th>"
        f"            <th>State / Reason</th>"
        f"            <th>Laid Piece</th>"
        f"            <th>Tracked Mask (<code>track_px</code>)</th>"
        f"            <th>Carried Area</th>"
        f"            <th>Changed</th>"
        f"          </tr>"
        f"        </thead>"
        f"        <tbody>{''.join(frame_rows)}</tbody>"
        f"      </table>"
        f"    </div>"
        f"  </details>"
        f"</div>"
    )



def file_mtime_ts(p: Path | None) -> float:
    if p is None:
        return 0.0
    try:
        return p.stat().st_mtime if p.exists() else 0.0
    except Exception:
        return 0.0


def find_job_folder_any_date(load_name: str | None, built_idx: dict[str, dict]) -> Path | None:
    if not load_name:
        return None
    if load_name in built_idx and built_idx[load_name].get("job_dir"):
        return built_idx[load_name]["job_dir"]
    matches = sorted(DATA_DIR.glob(f"*_{load_name}_job"), key=file_mtime_ts)
    return matches[-1] if matches else None


def find_cap_dir_any_date(load_name: str | None, built_idx: dict[str, dict]) -> Path | None:
    if not load_name:
        return None
    if load_name in built_idx:
        c = find_capture_dir_for_load(load_name, built_idx[load_name].get("built"))
        if c:
            return c
    matches = sorted(
        (p for p in DATA_DIR.glob(f"*_{load_name}") if p.is_dir() and not p.name.endswith(("_job", "_first"))),
        key=file_mtime_ts,
    )
    return matches[-1] if matches else None


def discover_candidate_loads(shots: list[dict]) -> dict[tuple[int, int, str], list[dict]]:
    """Discover all candidate load JSONs on disk mapped to their enclosing shot (shot_first, shot_last, subject), sorted by (mtime, name)."""
    out: dict[tuple[int, int, str], list[dict]] = {}
    seen_names: dict[tuple[int, int, str, str], dict] = {}
    shot_spans = [(int(sh["first"]), int(sh["last"])) for sh in shots]

    for pat in ("*_fun_phase*_inputs/*.json", "*_job/*.json"):
        for p in sorted(DATA_DIR.glob(pat)):
            if p.name.endswith((
                ".spec.json", ".built.json", ".queued.json", ".chain.json",
                ".masks.json", ".who_is_who.json", ".plan.json", ".render.json",
            )) or p.name.startswith(("shot_c_", "items.")):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            j = d.get("job") or {}
            loads = d.get("loads") or []
            if not loads or not isinstance(j, dict) or "first" not in j or "frames" not in j:
                continue
            l = loads[0]
            lname = str(l.get("name") or p.stem)
            subj = str(l.get("subject") or "")
            l_first = int(j["first"])
            l_last = l_first + int(j["frames"]) - 1
            if not subj:
                continue
            # Map sub-shot or full-shot load to its enclosing shot span
            matched_span = (l_first, l_last)
            for sf, sl in shot_spans:
                if l_first >= sf and l_last <= sl:
                    matched_span = (sf, sl)
                    break
            s_first, s_last = matched_span
            existing = seen_names.get((s_first, s_last, subj, lname))
            if existing and "_job" in p.parent.name and "_job" not in existing["json_path"].parent.name:
                continue
            seen_names[(s_first, s_last, subj, lname)] = {
                "name": lname,
                "subject": subj,
                "first": l_first,
                "last": l_last,
                "shot_first": s_first,
                "shot_last": s_last,
                "json_path": p,
                "mtime_ts": file_mtime_ts(p),
                "data": d,
            }
    for (s_first, s_last, subj, _), item in sorted(seen_names.items(), key=lambda x: (x[1]["mtime_ts"], x[0][3])):
        out.setdefault((s_first, s_last, subj), []).append(item)
    return out


def inspect_load_etl_steps(
    cand: dict | None,
    laid_rows_here: list[dict],
    previews_here: list[dict],
    first: int,
    last: int,
    subject: str,
    present_subjs: list[str],
    upstream_parent_states: list[dict],
    built_idx: dict[str, dict],
    has_frame_gaps: bool,
    gap_summary_str: str,
    latest_joined: Path | None,
    latest_joined_ts: float,
) -> dict:
    """Compute the 8-node Parent -> Child ETL DAG state, measured_on, cannot_see, and next command for (first, last, subject)."""
    is_multi = len(present_subjs) > 1
    has_second = "second" in present_subjs

    if subject == "second":
        etl_stage = "stage_a"
        stage_label = "Stage A · Pass 1 (second)"
        pass_badge = "Pass 1 (Base)"
    elif is_multi and has_second:
        etl_stage = "stage_c"
        if len(present_subjs) == 3 and subject == "lead":
            stage_label = "Stage C · Pass 3 (lead after second+third)"
            pass_badge = "Pass 3 (after S2+S3)"
        else:
            stage_label = f"Stage C · Pass 2 ({subject} after second)"
            pass_badge = "Pass 2 (after S2)"
    elif is_multi:
        etl_stage = "stage_c"
        stage_label = f"Stage C · Multi-Pass ({subject})"
        pass_badge = "Multi-Pass"
    else:
        etl_stage = "stage_d"
        stage_label = f"Stage D · Single-Subject ({subject})"
        pass_badge = "Single-Pass"

    laid_latest = laid_rows_here[-1] if laid_rows_here else None
    laid_name = laid_latest["load"] if laid_latest else None
    cand_name = cand["name"] if cand else laid_name
    cand_json_path = cand["json_path"] if cand else None

    if not cand_json_path and laid_name and laid_name in built_idx:
        cand_json_path = find_input_json_for_load(laid_name, built_idx[laid_name]["job_dir"], built_idx[laid_name].get("built"))

    job_folder = find_job_folder_any_date(cand_name, built_idx)
    cap_dir = find_cap_dir_any_date(cand_name, built_idx)

    cand_data = (cand or {}).get("data") or {}
    if not cand_data and cand_json_path and cand_json_path.is_file():
        try:
            cand_data = json.loads(cand_json_path.read_text())
        except Exception:
            cand_data = {}

    jblock = cand_data.get("job") or {}
    lblock = ((cand_data.get("loads") or [{}])[0]) if cand_data.get("loads") else {}

    rq_path = (job_folder / f"{cand_name}.render.queued.json") if (job_folder and cand_name) else None
    if not (rq_path and rq_path.is_file()) and cand_name and cand_name in built_idx and built_idx[cand_name].get("queued_path"):
        rq_path = built_idx[cand_name]["queued_path"]
    this_render_ts = file_mtime_ts(rq_path)

    is_laid_current = bool(laid_latest and laid_name == cand_name)
    laid_clean_shortcut = bool(is_laid_current and not has_frame_gaps and etl_stage == "stage_d")

    # Node 1: Upstream Source / Pass Dependency (using real parent mtime_ts!)
    n1_status = "done"
    n1_detail = "Original source plate (fun_0000_6394_24p_sound_in_step.mov)"
    max_parent_render_ts = max((p.get("render_ts", 0.0) for p in upstream_parent_states), default=0.0)
    if etl_stage == "stage_c" and upstream_parent_states:
        unlaid_parents = [p["active_load"] for p in upstream_parent_states if p.get("overall_state") != "complete_fresh"]
        if unlaid_parents:
            n1_status = "blocked"
            n1_detail = f"Waiting on upstream pass ({', '.join(unlaid_parents)}) to finish & be laid in Post-Pass-1 Cut (Stage B)"
        else:
            src_used = str(jblock.get("source") or "")
            on_joined_cut = ("joined_" in src_used or "from_laid" in src_used)
            if this_render_ts < max_parent_render_ts or not on_joined_cut or has_frame_gaps:
                n1_status = "stale"
                n1_detail = (
                    f"Upstream pass rendered at {fmt_local_ts(max_parent_render_ts)} "
                    f"(this row render={fmt_local_ts(this_render_ts)}, source={Path(src_used).name if src_used else 'original'}); "
                    f"needs Pass-2 preview+render on fresh Post-Pass-1 Cut"
                )
            else:
                n1_status = "done"
                n1_detail = f"Built on Post-Pass-1 Cut ({Path(src_used).name}) after upstream pass"

    # Node 2: Preview (SAM3 + 3DBody)
    prev_ref = jblock.get("preview")
    prev_jdir = find_preview_job_dir(prev_ref)
    latest_prev = previews_here[-1] if previews_here else None
    if not prev_jdir and latest_prev and latest_prev.get("job_folder"):
        prev_jdir = REPO / latest_prev["job_folder"]
    pq_file = (prev_jdir / "preview.queued.json") if prev_jdir else None
    prev_ts = file_mtime_ts(pq_file)

    n2_status = "pending"
    n2_detail = "No preview job yet"
    if n1_status == "blocked":
        n2_status = "blocked"
        n2_detail = "Blocked until upstream pass is laid in Post-Pass-1 Cut"
    elif n1_status == "stale":
        n2_status = "stale"
        n2_detail = "Needs `shot_to_load.py preview --source <post_pass1_cut.mp4> --remake --run`"
    elif pq_file and pq_file.is_file():
        n2_status = "done"
        n2_detail = f"{prev_jdir.name} ({path_mtime_local(pq_file)})"
    elif latest_prev and latest_prev.get("preview_done"):
        n2_status = "done"
        n2_detail = f"{latest_prev['name']} ({latest_prev['preview_done_mtime']})"
    elif (prev_jdir and (prev_jdir / "preview.built.json").is_file()) or (latest_prev and not latest_prev.get("preview_done")):
        n2_status = "ready"
        pname = prev_jdir.name if prev_jdir else latest_prev["name"]
        n2_detail = f"Pre-built ({pname}); ready to queue on :8188"
    elif laid_clean_shortcut:
        n2_status = "done"
        n2_detail = f"Completed in earlier run ({laid_name})"

    # Node 3: Pose (Sapiens-2 1B points.npz)
    ref_block = jblock.get("reference") or {}
    pts_file = Path(ref_block["points"]) if ref_block.get("points") else None
    pts_ts = file_mtime_ts(pts_file)
    n3_status = "pending"
    n3_detail = "Pending pose_along_shot.py"
    if n2_status in ("blocked", "stale"):
        n3_status = n2_status
        n3_detail = "Waiting on fresh preview"
    elif pts_file and pts_file.exists():
        n3_status = "done"
        n3_detail = f"{pts_file.name}"
    elif job_folder and (job_folder.parent / "chain" / f"{cand_name}.pose_pass.out").is_file():
        n3_status = "done"
        n3_detail = f"{cand_name}.pose_pass.out"
    elif laid_clean_shortcut:
        n3_status = "done"
        n3_detail = f"Completed ({laid_name})"
    elif n2_status == "done":
        n3_status = "ready"
        n3_detail = "Ready for `shot_to_load.py reference`"

    # Node 4: Job JSON & Head Reference <Video 1>
    motion_file = Path(lblock["motion_file"]) if lblock.get("motion_file") else None
    motion_ts = file_mtime_ts(motion_file)
    json_ts = file_mtime_ts(cand_json_path)
    n4_status = "pending"
    n4_detail = "Pending `shot_to_load.py job -> reference`"
    if n2_status in ("blocked", "stale"):
        n4_status = n2_status
        n4_detail = "Waiting on upstream preview"
    elif cand_json_path and cand_json_path.is_file() and motion_file and motion_file.exists():
        if pts_ts > 0 and motion_ts > 0 and motion_ts + 2.0 < pts_ts:
            n4_status = "stale"
            n4_detail = f"{motion_file.name} is older than {pts_file.name if pts_file else 'points.npz'}; rerun `shot_to_load.py reference`"
        else:
            n4_status = "done"
            n4_detail = f"{cand_json_path.name} + {motion_file.name} ({path_mtime_local(cand_json_path)})"
    elif laid_clean_shortcut:
        n4_status = "done"
        n4_detail = f"Completed ({laid_name})"
    elif cand_json_path and cand_json_path.is_file():
        n4_status = "ready"
        n4_detail = f"{cand_json_path.name} built; needs `shot_to_load.py reference`"
    elif n2_status == "done":
        n4_status = "ready"
        n4_detail = "Ready for `shot_to_load.py job -> reference`"

    # Node 5: Prompt Text (<load>.txt)
    txt_path = cand_json_path.with_suffix(".txt") if cand_json_path else None
    if txt_path and not txt_path.is_file() and job_folder:
        alt_txt = job_folder / f"{cand_name}.txt"
        if alt_txt.is_file():
            txt_path = alt_txt
    txt_ts = file_mtime_ts(txt_path)
    n5_status = "pending"
    n5_detail = "Pending `shot_to_load.py text`"
    if n2_status in ("blocked", "stale"):
        n5_status = n2_status
        n5_detail = "Waiting on upstream preview"
    elif txt_path and txt_path.is_file():
        n5_status = "done"
        n5_detail = f"{txt_path.name} ({path_mtime_local(txt_path)})"
    elif laid_clean_shortcut:
        n5_status = "done"
        n5_detail = f"Completed ({laid_name})"
    elif n4_status in ("done", "ready") and cand_json_path:
        n5_status = "ready"
        n5_detail = "Ready for `shot_to_load.py text`"

    # Node 6: Capture, Window Plan & Gate (first -> plan -> on-plan -> table -> looked)
    n6_status = "pending"
    n6_detail = "Pending `load_chain.py first -> plan -> on-plan -> table`"
    if n2_status in ("blocked", "stale") or n4_status == "stale":
        n6_status = "stale" if (n2_status == "stale" or n4_status == "stale") else "blocked"
        n6_detail = "Waiting on fresh upstream preview/reference"
    elif cap_dir and cap_dir.is_dir():
        v_ok = False
        g_ok = False
        v_path = cap_dir / "verify.json"
        f_path = cap_dir / "flags.json"
        ft_path = cap_dir / "frame_table.jsonl"
        lk_path = cap_dir / "looks.json"
        if v_path.is_file():
            try:
                v_ok = json.loads(v_path.read_text()).get("verdict") == "reads what the nodes wrote"
            except Exception:
                v_ok = False
        if f_path.is_file():
            try:
                g_ok = json.loads(f_path.read_text()).get("verdict") == "clear"
            except Exception:
                g_ok = False
        cap_ts = max(file_mtime_ts(v_path), file_mtime_ts(f_path), file_mtime_ts(ft_path))
        if v_ok and g_ok and ft_path.is_file() and lk_path.is_file():
            if txt_ts > 0 and cap_ts > 0 and cap_ts + 2.0 < txt_ts:
                n6_status = "stale"
                n6_detail = f"{cap_dir.name} gate predates {txt_path.name if txt_path else 'prompt'}; rerun `on-plan -> table -> looked`"
            else:
                n6_status = "done"
                n6_detail = f"{cap_dir.name} (verify=ok, gate=clear, looked)"
        elif (v_path.is_file() and not v_ok) or (f_path.is_file() and not g_ok):
            n6_status = "failed"
            n6_detail = f"{cap_dir.name} (verify_ok={v_ok}, gate_clear={g_ok})"
        else:
            n6_status = "ready"
            n6_detail = f"In progress in {cap_dir.name}"
    elif laid_clean_shortcut:
        n6_status = "done"
        n6_detail = f"Completed ({laid_name})"
    elif n4_status == "done" and n5_status == "done":
        n6_status = "ready"
        n6_detail = "Ready for `load_chain.py first -> plan -> on-plan -> table`"

    # Node 7: H3 DiT Render
    n7_status = "pending"
    n7_detail = "Pending `load_chain.py render`"
    if n2_status in ("blocked", "stale") or n4_status == "stale" or n6_status == "stale":
        n7_status = "stale" if "stale" in (n2_status, n4_status, n6_status) else "blocked"
        n7_detail = "Waiting on fresh upstream inputs/plan"
    elif rq_path and rq_path.is_file():
        try:
            rq_d = json.loads(rq_path.read_text())
        except Exception:
            rq_d = {}
        if rq_d.get("status") == "success":
            if (motion_ts > 0 and this_render_ts + 2.0 < motion_ts) or (txt_ts > 0 and this_render_ts + 2.0 < txt_ts):
                n7_status = "stale"
                n7_detail = f"{cand_name} render ({path_mtime_local(rq_path)}) is older than reference/prompt text!"
            else:
                n7_status = "done"
                n7_detail = f"{cand_name} rendered in {rq_d.get('seconds', '—')}s ({path_mtime_local(rq_path)})"
        else:
            n7_status = "failed"
            n7_detail = f"{cand_name} render status={rq_d.get('status')}"
    elif laid_clean_shortcut:
        n7_status = "done"
        n7_detail = f"Rendered ({laid_name})"
    elif n6_status == "done":
        n7_status = "ready"
        n7_detail = f"Ready for `what_h3_got.py` check + `load_chain.py render` ({cand_name})"

    # Node 8: Land, Page & Laid in Cut
    rpts_path = (cap_dir / f"{cand_name}_render_points.npz") if (cap_dir and cand_name) else None
    n8_status = "pending"
    n8_detail = "Not yet landed / laid in cut"
    if n2_status in ("blocked", "stale") or n7_status == "stale":
        n8_status = "stale" if "stale" in (n2_status, n7_status) else "blocked"
        n8_detail = "Waiting on fresh render"
    elif is_laid_current and not has_frame_gaps and n7_status == "done":
        n8_status = "done"
        n8_detail = f"Laid in cut ({laid_name}, 100% carried)"
    elif rpts_path and rpts_path.is_file() and not is_laid_current and n7_status == "done":
        n8_status = "ready"
        n8_detail = f"Landed ({cand_name}); inspect stacked review clip, then lay into `data/fun/laid.jsonl`"
    elif n7_status == "done" and not (rpts_path and rpts_path.is_file()):
        n8_status = "ready"
        n8_detail = f"Rendered ({cand_name}); ready for `load_chain.py land && page`"
    elif is_laid_current and has_frame_gaps:
        n8_status = "stale"
        n8_detail = f"Laid ({laid_name}), but has frame gap: {gap_summary_str}"

    nodes = [
        {"num": 1, "short": "1. Pass/Source", "status": n1_status, "detail": n1_detail},
        {"num": 2, "short": "2. Preview", "status": n2_status, "detail": n2_detail},
        {"num": 3, "short": "3. Pose", "status": n3_status, "detail": n3_detail},
        {"num": 4, "short": "4. Job+Ref", "status": n4_status, "detail": n4_detail},
        {"num": 5, "short": "5. Prompt", "status": n5_status, "detail": n5_detail},
        {"num": 6, "short": "6. Plan+Gate", "status": n6_status, "detail": n6_detail},
        {"num": 7, "short": "7. H3 Render", "status": n7_status, "detail": n7_detail},
        {"num": 8, "short": "8. Land+Laid", "status": n8_status, "detail": n8_detail},
    ]

    # Build explicit measured_on and cannot_see strings (AGENTS.md rule: print what was measured on and what it cannot see)
    measured_bits = []
    if cand_json_path:
        measured_bits.append(f"job={rel_or_str(cand_json_path)} ({path_mtime_local(cand_json_path)})")
    if pq_file and pq_file.is_file():
        measured_bits.append(f"preview={rel_or_str(pq_file)} ({path_mtime_local(pq_file)})")
    elif latest_prev:
        measured_bits.append(f"preview_input={latest_prev['input_json']} ({latest_prev['input_mtime']})")
    if motion_file:
        measured_bits.append(f"head_ref={motion_file.name}")
    if txt_path and txt_path.is_file():
        measured_bits.append(f"prompt={rel_or_str(txt_path)} ({path_mtime_local(txt_path)})")
    if cap_dir and cap_dir.is_dir():
        measured_bits.append(f"capture={rel_or_str(cap_dir)}")
    if rq_path and rq_path.is_file():
        measured_bits.append(f"render_queued={rel_or_str(rq_path)} ({path_mtime_local(rq_path)})")
    if laid_name:
        measured_bits.append(f"laid={laid_name}")
    measured_on_str = " · ".join(measured_bits) if measured_bits else "shots.json + laid.jsonl (no candidate job on disk yet)"

    sheet_hint = str(motion_file).replace(".mp4", "_sheet.jpg") if motion_file else "<head_reference>_sheet.jpg"
    cannot_see_str = (
        f"Visual & prompt fidelity cannot be verified by mtime: inspect `{sheet_hint}` (3D head ref & face-not-shown frames), "
        f"`{rel_or_str(cap_dir) + '/frame_table.md' if cap_dir else 'data/<cap>/frame_table.md'}` + `.1fps.` strip, "
        f"full prompt via `bench/what_h3_got.py`, and stacked review clip `/media/{cand_name or laid_name or '<load>'}_{first:04d}_{last:04d}_stacked.mp4` before laying in `data/fun/laid.jsonl`."
    )

    jf_rel = rel_or_str(cand_json_path) if cand_json_path else f"data/2026-10-20_fun_phase3_inputs/{subject}_hfp_{first:04d}a.json"
    orig_src_mov = str(INPUT_DIR / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov")
    if etl_stage == "stage_c":
        rerun_src = str(latest_joined) if (latest_joined and latest_joined_ts >= max_parent_render_ts) else "<RUN_STAGE_B_FIRST_FOR_POST_PASS1_CUT.mp4>"
    else:
        rerun_src = orig_src_mov

    if n8_status == "done" and n1_status == "done":
        overall_state = "complete_fresh"
        overall_label = "✅ COMPLETE & FRESH IN CUT"
        reason = f"Laid in cut ({cand_name}), 100% carried frames, all parents fresh."
        next_cmd = f"# Complete ({cand_name})"
        runner_cmd = "—"
    elif n6_status == "failed" or n7_status == "failed":
        overall_state = "failed_gate"
        overall_label = "❌ GATE / VERIFY BLOCKED"
        reason = n6_detail if n6_status == "failed" else n7_detail
        next_cmd = f"python bench/load_chain.py on-plan {jf_rel} --by mrgem"
        runner_cmd = f"python bench/build_cut_map.py --next --shot {first} --subject {subject}"
    elif n1_status == "blocked":
        overall_state = "blocked_by_parent"
        overall_label = "🔒 BLOCKED BY PASS-1 PARENT"
        reason = n1_detail
        next_cmd = f"# Wait for Stage A (second_{first}) + Stage B (laid.jsonl + rebuild_delivery.py --clip), then run Pass 2"
        runner_cmd = f"python bench/build_cut_map.py --next --shot {first} --subject {subject}"
    elif rpts_path and rpts_path.is_file() and not is_laid_current and n7_status == "done":
        overall_state = "ready_to_lay"
        overall_label = "📥 LANDED — READY TO LAY IN CUT"
        reason = f"{cand_name} finished render + land + pose pass; verify stacked review clip & lay into data/fun/laid.jsonl (Stage B)."
        next_cmd = f"python bench/rebuild_delivery.py --video fun --clip && python bench/build_cut_map.py"
        runner_cmd = f"python bench/build_cut_map.py --next --stage stage-a"
    elif n1_status == "stale" or n8_status == "stale" or n4_status == "stale" or n6_status == "stale" or n7_status == "stale":
        overall_state = "stale_needs_rerun"
        overall_label = "🔄 NEEDS RERUN (STALE PARENT / GAP)"
        reason = next((d for st, d in [(n1_status, n1_detail), (n4_status, n4_detail), (n6_status, n6_detail), (n7_status, n7_detail), (n8_status, n8_detail)] if st == "stale"), gap_summary_str)
        if latest_prev and not latest_prev.get("preview_done") and etl_stage == "stage_d":
            p_jdir = latest_prev.get("job_folder") or f"data/2026-10-20_{latest_prev['name']}_job"
            next_cmd = f"python bench/masked_job.py queue {p_jdir} preview --wait"
        else:
            next_cmd = (
                f"python bench/shot_to_load.py preview --source {rerun_src} "
                f"--first {first} --last {last} --cast data/fun/cast/cast.jsonl --remake --run --by mrgem"
            )
        runner_cmd = f"python bench/build_cut_map.py --next --shot {first} --subject {subject}"
    else:
        overall_state = "ready_to_run"
        if n2_status == "ready" and latest_prev:
            overall_label = f"⚡ READY: STEP 2 PREVIEW ({latest_prev['name']})"
            reason = f"Preview job {latest_prev['name']} is pre-built; ready to queue on :8188."
            p_jdir = latest_prev.get("job_folder") or f"data/2026-10-20_{latest_prev['name']}_job"
            next_cmd = f"python bench/masked_job.py queue {p_jdir} preview --wait"
        elif n4_status == "ready" or n5_status == "ready":
            overall_label = f"⚡ READY: STEP 3–5 JOB/REF/TEXT ({cand_name or subject})"
            reason = f"Preview complete; ready to generate job/head_reference/text for {cand_name or f'{subject}_{first:04d}'}."
            next_cmd = f"python bench/shot_to_load.py reference {jf_rel} --by mrgem && python bench/shot_to_load.py text {jf_rel} --by mrgem --plan data/fun/plan.json"
        elif n6_status == "ready":
            overall_label = f"⚡ READY: STEP 6 PLAN & GATE ({cand_name})"
            reason = f"Reference & prompt text ready; run first -> plan -> on-plan -> table for {cand_name}."
            next_cmd = f"python bench/load_chain.py first {jf_rel} --by mrgem && python bench/load_chain.py plan {jf_rel} --by mrgem && python bench/load_chain.py on-plan {jf_rel} --by mrgem"
        elif n7_status == "ready":
            overall_label = f"⚡ READY: STEP 7 H3 RENDER ({cand_name})"
            reason = f"Plan & gate clear and looked; verify prompt via what_h3_got.py, then render + land + page {cand_name}."
            jfolder_rel = rel_or_str(job_folder) if job_folder else f"data/2026-10-20_{cand_name}_job"
            next_cmd = f"python bench/what_h3_got.py --job-folder {jfolder_rel} --load {cand_name} --stage plan && python bench/load_chain.py render {jf_rel} --by mrgem && python bench/load_chain.py land {jf_rel} --by mrgem"
        elif n8_status == "ready":
            overall_label = f"⚡ READY: STEP 8 LAND & PAGE ({cand_name})"
            reason = f"Render finished; run land & page for {cand_name}."
            next_cmd = f"python bench/load_chain.py land {jf_rel} --by mrgem"
        else:
            overall_label = f"⚡ READY TO BUILD ({subject} {first}–{last})"
            reason = gap_summary_str or "Pending build"
            next_cmd = f"python bench/build_cut_map.py --next --shot {first} --subject {subject}"
        runner_cmd = f"python bench/build_cut_map.py --next --shot {first} --subject {subject}"

    return {
        "etl_stage": etl_stage,
        "stage_label": stage_label,
        "pass_badge": pass_badge,
        "active_load": cand_name or laid_name or "—",
        "laid_load": laid_name or "—",
        "json_path": jf_rel if cand_json_path else None,
        "render_ts": this_render_ts,
        "overall_state": overall_state,
        "overall_label": overall_label,
        "reason": reason,
        "measured_on": measured_on_str,
        "cannot_see": cannot_see_str,
        "next_cmd": next_cmd,
        "runner_cmd": runner_cmd,
        "nodes": nodes,
    }


def main() -> None:
    tz_abbr = local_tz_abbr()
    now_local = fmt_local_ts(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())
    shots_data = json.loads((FUN_DIR / "shots.json").read_text())
    shots = shots_data["shots"]
    plan_data = json.loads((FUN_DIR / "plan.json").read_text())
    cast_rows = load_cast_index()

    laid_rows = [
        json.loads(line)
        for line in (FUN_DIR / "laid.jsonl").read_text().splitlines()
        if line.strip()
    ]
    laid_by_shot_subj: dict[tuple[int, int, str], list[dict]] = {}
    for r in laid_rows:
        key = (int(r["first"]), int(r["last"]), str(r["subject"]))
        laid_by_shot_subj.setdefault(key, []).append(r)

    built_idx = load_built_index()
    today_previews = load_today_previews()
    cov_idx, cov_json_path = load_frame_coverage_index()
    cand_idx = discover_candidate_loads(shots)
    laid_sha256 = hashlib.sha256((FUN_DIR / "laid.jsonl").read_bytes()).hexdigest() if (FUN_DIR / "laid.jsonl").is_file() else ""

    # Latest full joined video (resolved from local delivery table so we never block on SMB)
    delivery_tables = sorted(
        DATA_DIR.glob("*_fun_delivery_tables/t_fun_0000_6394_*.txt"),
        key=lambda p: p.stat().st_mtime,
    )
    if delivery_tables:
        dt_latest = delivery_tables[-1]
        ts_tag = dt_latest.stem.replace("t_fun_0000_6394_", "")
        latest_joined = SHARE_DIR / "Video" / "fun" / f"fun_0000_6394_joined_{ts_tag}_from_laid.mp4"
        latest_joined_mtime = path_mtime_local(dt_latest)
        latest_joined_ts = file_mtime_ts(dt_latest)
    else:
        latest_joined = None
        latest_joined_mtime = "—"
        latest_joined_ts = 0.0

    rows = []
    counts = {
        "total_subj_shots": 0,
        "in_cut": 0,
        "clean_in_cut": 0,
        "today_updated": 0,
        "in_cut_rerun": 0,
        "pending": 0,
        "no_subject": 0,
    }

    for sh in shots:
        first, last = int(sh["first"]), int(sh["last"])
        n_frames = last - first + 1
        tc = f"{fmt_tc(first)}–{fmt_tc(last)}"
        who = sh.get("who") or {}
        present_subjs = [
            s for s in ("lead", "second", "third")
            if (who.get(s) or {}).get("present") == "yes"
        ]
        if not present_subjs:
            counts["no_subject"] += 1
            row_id = f"r_{first}_{last}_none"
            dag_steps = build_dag_steps_for_row(sh, "—", [], [], plan_data, cast_rows, built_idx, latest_joined)
            cov_data = build_frame_coverage_for_row(first, last, "—", [], cov_idx, built_idx)
            rows.append({
                "row_id": row_id,
                "first": first,
                "last": last,
                "n_frames": n_frames,
                "tc": tc,
                "subject": "—",
                "status_kind": "no_subject",
                "status_label": "No Subject (Source Kept)",
                "load": "—",
                "mp4_rel": "— (original source video)",
                "mp4_abs": None,
                "mp4_mtime": "—",
                "review_clip": None,
                "lineage_items": [],
                "queued_next": [],
                "dag_steps": dag_steps,
                "cov_data": cov_data,
            })
            continue

        for s in present_subjs:
            counts["total_subj_shots"] += 1
            row_id = f"r_{first}_{last}_{s}"
            lrows = laid_by_shot_subj.get((first, last, s), [])
            previews_here = today_previews.get((first, last), [])
            dag_steps = build_dag_steps_for_row(
                sh, s, lrows, previews_here, plan_data, cast_rows, built_idx, latest_joined
            )
            cov_data = build_frame_coverage_for_row(first, last, s, lrows, cov_idx, built_idx)
            cc = cov_data["counts"]

            if lrows:
                counts["in_cut"] += 1
                loads_str = ", ".join(r["load"] for r in lrows)
                mp4_rels = [r["file"] for r in lrows]
                mp4_abs_list = [SHARE_DIR / r["file"] for r in lrows]
                mp4_mtimes = [load_mtime_local_for_laid(r, built_idx) for r in lrows]
                rendered_today = any(
                    "2026-10-20" in m or "0278i" in r["load"] or "0604h" in r["load"] or "0606d" in r["load"]
                    for r, m in zip(lrows, mp4_mtimes)
                )
                if rendered_today:
                    counts["today_updated"] += 1

                # Is this (shot, subject) slated for a Phase-2 or Phase-3/4 re-run?
                slated_phase2 = (
                    s == "second"
                    and any("Phase 2" in p["phase"] for p in previews_here)
                    and not rendered_today
                )
                slated_phase3 = (
                    any("Phase 3" in p["phase"] for p in previews_here)
                    and (cc["unrendered_gap"] > 0 or cc["emptied"] > 0)
                    and not rendered_today
                )
                # Or Pass-2 multi-subject re-run when Pass-1 second is being re-run on the same shot
                slated_pass2_after_second = (
                    s in ("lead", "third")
                    and "second" in present_subjs
                    and any("Phase 2" in p["phase"] for p in previews_here)
                    and (cc["unrendered_gap"] > 0 or cc["emptied"] > 0 or first == 1010)
                    and not rendered_today
                )

                has_frame_gaps = (cc["unrendered_gap"] > 0 or cc["emptied"] > 0)
                gap_str = " · ".join(cov_data["gap_summary_parts"][:2]) if cov_data["gap_summary_parts"] else ""

                if slated_phase2 or slated_phase3 or slated_pass2_after_second or has_frame_gaps:
                    counts["in_cut_rerun"] += 1
                    status_kind = "in_cut_rerun"
                    reasons = []
                    if gap_str:
                        reasons.append(gap_str)
                    if slated_phase2:
                        reasons.append("Queued Phase-2 Re-run")
                    elif slated_phase3:
                        reasons.append("Queued Phase-3/4 Re-run")
                    elif slated_pass2_after_second:
                        reasons.append("Queued Pass-2 Re-run")
                    reason_suffix = " · ".join(reasons) if reasons else "Needs Review / Gap Check"
                    status_label = f"IN CUT ({lrows[-1]['owner']}) · {reason_suffix}"
                elif rendered_today:
                    counts["clean_in_cut"] += 1
                    status_kind = "in_cut_today"
                    status_label = f"IN CUT (Rendered Today · 100% Carried)"
                else:
                    counts["clean_in_cut"] += 1
                    status_kind = "in_cut"
                    status_label = f"IN CUT ({lrows[-1]['owner']} · 100% Carried)"

                lineage_items = []
                review_clip_url = None
                for r in lrows:
                    lname = r["load"]
                    binfo = built_idx.get(lname)
                    cap_dir = find_capture_dir_for_load(lname, (binfo or {}).get("built"))
                    if binfo:
                        jdir = binfo["job_dir"]
                        in_json = find_input_json_for_load(lname, jdir, binfo.get("built"))
                        if in_json:
                            lineage_items.append({
                                "kind": "Job Input JSON",
                                "path": str(in_json.relative_to(REPO)),
                                "mtime": path_mtime_local(in_json),
                            })
                        lineage_items.append({
                            "kind": "Built/Queued Job",
                            "path": str(jdir.relative_to(REPO)),
                            "mtime": path_mtime_local(binfo["built_path"]),
                        })
                        bdata = binfo.get("built") or {}
                        src_vid = (bdata.get("job") or {}).get("source") or (bdata.get("load") or {}).get("source")
                        if src_vid:
                            src_p = Path(src_vid)
                            lineage_items.append({
                                "kind": "Source Video Fed In",
                                "path": src_p.name,
                                "mtime": path_mtime_local(src_p),
                            })
                    if cap_dir:
                        lineage_items.append({
                            "kind": "Capture & Matte Folder",
                            "path": str(cap_dir.relative_to(REPO)),
                            "mtime": path_mtime_local(cap_dir / "manifest.json"),
                        })
                    for rc in (FUN_DIR / "review" / "media").glob(f"{lname}_*_stacked.mp4"):
                        review_clip_url = f"/media/{rc.name}"

                rows.append({
                    "row_id": row_id,
                    "first": first,
                    "last": last,
                    "n_frames": n_frames,
                    "tc": tc,
                    "subject": s,
                    "status_kind": status_kind,
                    "status_label": status_label,
                    "load": loads_str,
                    "mp4_rel": " ; ".join(mp4_rels),
                    "mp4_abs": " ; ".join(str(p) for p in mp4_abs_list),
                    "mp4_mtime": " ; ".join(mp4_mtimes),
                    "review_clip": review_clip_url,
                    "lineage_items": lineage_items,
                    "queued_next": previews_here,
                    "dag_steps": dag_steps,
                    "cov_data": cov_data,
                })
            else:
                counts["pending"] += 1
                gap_str = " · ".join(cov_data["gap_summary_parts"][:2]) if cov_data["gap_summary_parts"] else ""
                p_label = f"PENDING RENDER · {gap_str}" if gap_str else "PENDING RENDER"
                rows.append({
                    "row_id": row_id,
                    "first": first,
                    "last": last,
                    "n_frames": n_frames,
                    "tc": tc,
                    "subject": s,
                    "status_kind": "pending",
                    "status_label": p_label,
                    "load": "—",
                    "mp4_rel": "Not yet in cut",
                    "mp4_abs": None,
                    "mp4_mtime": "—",
                    "review_clip": None,
                    "lineage_items": [],
                    "queued_next": previews_here,
                    "dag_steps": dag_steps,
                    "cov_data": cov_data,
                })

    # Compute 8-node Parent -> Child ETL DAG state for every row (Pass 1 second first, then Pass 2/3 & single-pass)
    present_by_shot: dict[tuple[int, int], list[str]] = {}
    for r in rows:
        if r["subject"] != "—":
            present_by_shot.setdefault((r["first"], r["last"]), []).append(r["subject"])

    second_etl_by_shot: dict[tuple[int, int], dict] = {}
    third_etl_by_shot: dict[tuple[int, int], dict] = {}
    for r in rows:
        if r["subject"] == "second":
            key3 = (r["first"], r["last"], "second")
            cands_here = cand_idx.get(key3, [])
            latest_cand = cands_here[-1] if cands_here else None
            lrows = laid_by_shot_subj.get(key3, [])
            previews_here = today_previews.get((r["first"], r["last"]), [])
            cc = r["cov_data"]["counts"]
            has_gaps = (cc["unrendered_gap"] > 0 or cc["emptied"] > 0)
            gap_str = " · ".join(r["cov_data"]["gap_summary_parts"][:2]) if r["cov_data"]["gap_summary_parts"] else ""
            etl_info = inspect_load_etl_steps(
                latest_cand, lrows, previews_here, r["first"], r["last"], "second",
                present_by_shot.get((r["first"], r["last"]), ["second"]),
                [], built_idx, has_gaps, gap_str, latest_joined, latest_joined_ts,
            )
            r["etl"] = etl_info
            second_etl_by_shot[(r["first"], r["last"])] = etl_info

    for r in rows:
        if r["subject"] == "third":
            key3 = (r["first"], r["last"], "third")
            cands_here = cand_idx.get(key3, [])
            latest_cand = cands_here[-1] if cands_here else None
            lrows = laid_by_shot_subj.get(key3, [])
            previews_here = today_previews.get((r["first"], r["last"]), [])
            cc = r["cov_data"]["counts"]
            has_gaps = (cc["unrendered_gap"] > 0 or cc["emptied"] > 0)
            gap_str = " · ".join(r["cov_data"]["gap_summary_parts"][:2]) if r["cov_data"]["gap_summary_parts"] else ""
            up_parents = [second_etl_by_shot[(r["first"], r["last"])]] if (r["first"], r["last"]) in second_etl_by_shot else []
            etl_info = inspect_load_etl_steps(
                latest_cand, lrows, previews_here, r["first"], r["last"], "third",
                present_by_shot.get((r["first"], r["last"]), ["third"]),
                up_parents, built_idx, has_gaps, gap_str, latest_joined, latest_joined_ts,
            )
            r["etl"] = etl_info
            third_etl_by_shot[(r["first"], r["last"])] = etl_info

    for r in rows:
        if r["subject"] == "—":
            r["etl"] = {
                "etl_stage": "no_subject",
                "stage_label": "Source Pass-Through",
                "pass_badge": "Source Only",
                "active_load": "—",
                "laid_load": "—",
                "json_path": None,
                "render_ts": 0.0,
                "overall_state": "no_subject",
                "overall_label": "SOURCE PASS-THROUGH",
                "reason": "No tracked subject in shot; original source video kept unchanged.",
                "measured_on": "data/fun/shots.json",
                "cannot_see": "—",
                "next_cmd": "—",
                "runner_cmd": "—",
                "nodes": [
                    {"num": 1, "short": "1. Source Plate", "status": "done", "detail": "Original source kept"},
                    {"num": 8, "short": "8. In Cut", "status": "done", "detail": "Passed through to final cut"},
                ],
            }
        elif r["subject"] == "lead":
            key3 = (r["first"], r["last"], "lead")
            cands_here = cand_idx.get(key3, [])
            latest_cand = cands_here[-1] if cands_here else None
            lrows = laid_by_shot_subj.get(key3, [])
            previews_here = today_previews.get((r["first"], r["last"]), [])
            cc = r["cov_data"]["counts"]
            has_gaps = (cc["unrendered_gap"] > 0 or cc["emptied"] > 0)
            gap_str = " · ".join(r["cov_data"]["gap_summary_parts"][:2]) if r["cov_data"]["gap_summary_parts"] else ""
            up_parents = []
            if (r["first"], r["last"]) in second_etl_by_shot:
                up_parents.append(second_etl_by_shot[(r["first"], r["last"])])
            if (r["first"], r["last"]) in third_etl_by_shot and (r["first"], r["last"]) in second_etl_by_shot:
                up_parents.append(third_etl_by_shot[(r["first"], r["last"])])
            r["etl"] = inspect_load_etl_steps(
                latest_cand, lrows, previews_here, r["first"], r["last"], "lead",
                present_by_shot.get((r["first"], r["last"]), ["lead"]),
                up_parents, built_idx, has_gaps, gap_str, latest_joined, latest_joined_ts,
            )

    # Compute ETL summary counts across stages
    etl_counts = {
        "complete_fresh": sum(1 for r in rows if r["etl"]["overall_state"] == "complete_fresh"),
        "ready_to_lay": sum(1 for r in rows if r["etl"]["overall_state"] == "ready_to_lay"),
        "ready_to_run": sum(1 for r in rows if r["etl"]["overall_state"] == "ready_to_run"),
        "stale_needs_rerun": sum(1 for r in rows if r["etl"]["overall_state"] == "stale_needs_rerun"),
        "blocked_by_parent": sum(1 for r in rows if r["etl"]["overall_state"] == "blocked_by_parent"),
        "failed_gate": sum(1 for r in rows if r["etl"]["overall_state"] == "failed_gate"),
        "stage_a_total": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_a"),
        "stage_a_done": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_a" and r["etl"]["overall_state"] in ("complete_fresh", "ready_to_lay")),
        "stage_a_laid": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_a" and r["etl"]["overall_state"] == "complete_fresh"),
        "stage_c_total": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_c"),
        "stage_c_done": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_c" and r["etl"]["overall_state"] == "complete_fresh"),
        "stage_d_total": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_d"),
        "stage_d_done": sum(1 for r in rows if r["etl"]["etl_stage"] == "stage_d" and r["etl"]["overall_state"] == "complete_fresh"),
    }

    # Write DAG_STATUS.json and DAG_STATUS.md
    dag_status_payload = {
        "video": "fun",
        "span": [0, 6394],
        "last_updated_local": now_local,
        "last_updated_utc": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "laid_sha256": laid_sha256,
        "latest_joined_mp4": str(latest_joined) if latest_joined else None,
        "latest_joined_mtime": latest_joined_mtime,
        "summary": etl_counts,
        "stage_order": [
            {
                "stage": "Stage A — Pass 1 (second, 19 shots)",
                "progress": f"{etl_counts['stage_a_done']} / {etl_counts['stage_a_total']} rendered & landed ({etl_counts['stage_a_laid']} laid in cut)",
                "command": "python bench/build_cut_map.py --next --stage stage-a",
            },
            {
                "stage": "Stage B — Lay Pass 1 & Rebuild 24.0 fps Post-Pass-1 Cut",
                "progress": "Ready once all 19 Stage A loads are landed" if etl_counts["stage_a_done"] < etl_counts["stage_a_total"] else "READY TO RUN NOW",
                "command": "python bench/rebuild_delivery.py --video fun --clip && python bench/build_cut_map.py",
            },
            {
                "stage": "Stage C — Multi-Pass Pass 2/3 (lead & third on multi-subject shots)",
                "progress": f"{etl_counts['stage_c_done']} / {etl_counts['stage_c_total']} complete in cut",
                "command": "python bench/build_cut_map.py --next --stage stage-c",
            },
            {
                "stage": "Stage D — Single-Subject lead & third Shots (0000–6394)",
                "progress": f"{etl_counts['stage_d_done']} / {etl_counts['stage_d_total']} complete in cut",
                "command": "python bench/build_cut_map.py --next --stage stage-d",
            },
        ],
        "rows": [
            {
                "row_id": r["row_id"],
                "first": r["first"],
                "last": r["last"],
                "frames": r["n_frames"],
                "tc": r["tc"],
                "subject": r["subject"],
                **r["etl"],
            }
            for r in rows
            if r["subject"] != "—"
        ],
    }
    (FUN_DIR / "DAG_STATUS.json").write_text(json.dumps(dag_status_payload, indent=2) + "\n")

    dag_md = [
        "# `fun` (0000–6394) — Parent -> Child ETL Build Queue & Execution DAG Status",
        "",
        f"- **Last Updated**: `{now_local}` (`{dag_status_payload['last_updated_utc']}`)",
        f"- **Browser Interactive View (Tab 3 of Provenance)**: `http://localhost:8765/PROVENANCE.html#etl` (`data/fun/poc/PROVENANCE.html`)",
        f"- **Machine-Readable Record**: `data/fun/DAG_STATUS.json`",
        f"- **Refresh Status & HTML Pages**: `python bench/build_cut_map.py`",
        f"- **Show Next Runnable Queue**: `python bench/build_cut_map.py --next`",
        f"- **Query by Stage / Shot / Subject**: `python bench/build_cut_map.py --next [--stage stage-a|stage-b|stage-c|stage-d] [--shot <FIRST>] [--subject <SUBJ>]`",
        "",
        "## 1. Macro-Stage Topological Order & Live Progress",
        "",
        "| Stage | Description | Progress | CLI Runner Command |",
        "|---|---|---|---|",
        f"| **Stage A** | **Pass 1 (`second`, 19 shots)** on original source (`fun_0000_6394_24p_sound_in_step.mov`) | **{etl_counts['stage_a_done']} / {etl_counts['stage_a_total']}** landed (**{etl_counts['stage_a_laid']}** laid in cut) | `python bench/build_cut_map.py --run --stage stage-a` |",
        f"| **Stage B** | **Lay Pass 1 & Assemble 24.0 fps Post-Pass-1 Cut** (`laid.jsonl` + `rebuild_delivery.py --clip`) | Latest cut: `{latest_joined.name if latest_joined else 'none'}` (`{latest_joined_mtime}`) | `python bench/build_cut_map.py --run --stage stage-b` |",
        f"| **Stage C** | **Multi-Pass Pass 2 & Pass 3 (`lead` & `third`)** on the Post-Pass-1 Cut | **{etl_counts['stage_c_done']} / {etl_counts['stage_c_total']}** complete & fresh | `python bench/build_cut_map.py --run --stage stage-c` |",
        f"| **Stage D** | **Single-Subject `lead` & `third` Shots** (`0000–6394`) on original source | **{etl_counts['stage_d_done']} / {etl_counts['stage_d_total']}** complete & fresh | `python bench/build_cut_map.py --run --stage stage-d` |",
        "",
        "## 2. Summary Counts Across All 121 Subject-Shots",
        "",
        f"- ✅ **Complete & Fresh in Cut**: `{etl_counts['complete_fresh']}`",
        f"- 📥 **Landed — Ready to Lay in Cut (`laid.jsonl`)**: `{etl_counts['ready_to_lay']}`",
        f"- ⚡ **Ready to Run Now**: `{etl_counts['ready_to_run']}`",
        f"- 🔄 **Needs Rerun (Stale Parent / Frame Gap)**: `{etl_counts['stale_needs_rerun']}`",
        f"- 🔒 **Blocked by Upstream Pass-1 Parent**: `{etl_counts['blocked_by_parent']}`",
        f"- ❌ **Gate / Verify Blocked**: `{etl_counts['failed_gate']}`",
        "",
        "## 3. Active Work Queue (All Non-Complete Subject-Shots in Execution Order)",
        "",
        "| Stage & Pass | Shot (`TC`) | Subject | Active Load | Overall ETL State | Why / Dependency Reason | Next Step Command |",
        "|---|---|---|---|---|---|---|",
    ]
    stage_rank = {"stage_a": 1, "stage_c": 2, "stage_d": 3, "no_subject": 9}
    actionable_rows = [r for r in rows if r["subject"] != "—" and r["etl"]["overall_state"] != "complete_fresh"]
    actionable_rows.sort(key=lambda r: (stage_rank.get(r["etl"]["etl_stage"], 9), r["first"], r["subject"]))
    for r in actionable_rows:
        e = r["etl"]
        dag_md.append(
            f"| `{e['stage_label']}` | `{r['first']}–{r['last']}` ({r['tc']}, {r['n_frames']}f) | `{r['subject']}` | `{e['active_load']}` | **{e['overall_label']}** | {e['reason']} | `{e['next_cmd']}` |"
        )
    (FUN_DIR / "DAG_STATUS.md").write_text("\n".join(dag_md) + "\n")

    # Write Markdown (data/fun/CUT_MAP.md)
    md_lines = [
        "# `fun` End-to-End Cut & Lineage Map (`0000–6394`)",
        "",
        f"- **Generated**: `{now_local}` by `data/fun/build_cut_map.py` from `data/fun/shots.json`, `data/fun/plan.json`, `data/fun/cast/cast.jsonl`, `data/fun/laid.jsonl`, `frame_coverage_fun_0000_6394_*.json`, and `data/*_job/`",
        f"- **Browser View (Table + Frame-Level Coverage Drill-Down + Hierarchical Data Flow DAG)**: `http://localhost:8765/PROVENANCE.html` (`data/fun/poc/PROVENANCE.html`)",
        f"- **Latest Full Joined Cut MP4**: `{latest_joined}` (created `{path_mtime_local(latest_joined)}`)",
        f"- **Subject-Shot Coverage**: **{counts['in_cut']} / {counts['total_subj_shots']}** in cut (**{counts['clean_in_cut']}** clean 100% carried, **{counts['in_cut_rerun']}** in cut with frame gaps or queued re-run, **{counts['pending']}** pending render, `{counts['no_subject']}` shots with no tracked subject)",
        "",
        f"| Frames (`TC`) | Len | Subject | Status & Frame Coverage | Final Rendered MP4 (under `H3_OUTPUT_DIR/`) | MP4 Created ({tz_abbr}) | Data Folders / Input Files & Timestamps ({tz_abbr}) |",
        "|---|---:|---|---|---|---|---|",
    ]
    for r in rows:
        lin_parts = []
        for li in r["lineage_items"]:
            lin_parts.append(f"**{li['kind']}**: `{li['path']}` ({li['mtime']})")
        for qn in r["queued_next"]:
            pstat = f"preview done {qn['preview_done_mtime']}" if qn["preview_done"] else "preview pre-built"
            lin_parts.append(f"**Next ({qn['phase']})**: `{qn['input_json']}` ({qn['input_mtime']}, {pstat})")
        lin_str = "<br>".join(lin_parts) if lin_parts else "—"
        md_lines.append(
            f"| `{r['first']}–{r['last']}` ({r['tc']}) | {r['n_frames']}f | `{r['subject']}` | {r['status_label']} (`{r['cov_data']['short_summary']}`) | `{r['mp4_rel']}` | {r['mp4_mtime']} | {lin_str} |"
        )

    (FUN_DIR / "CUT_MAP.md").write_text("\n".join(md_lines) + "\n")

    # Build HTML rows (Tab 1) and Master-Detail items (Tab 2)
    html_rows = []
    sidebar_items = []
    detail_panes = []

    for idx_i, r in enumerate(rows):
        row_id = r["row_id"]
        badge_cls = {
            "in_cut": "badge-green",
            "in_cut_today": "badge-blue",
            "in_cut_rerun": "badge-amber",
            "pending": "badge-red",
            "no_subject": "badge-gray",
        }.get(r["status_kind"], "badge-gray")

        short_badge = {
            "in_cut": "IN CUT 100%",
            "in_cut_today": "TODAY 100%",
            "in_cut_rerun": "GAP / RERUN",
            "pending": "PENDING",
            "no_subject": "SOURCE",
        }.get(r["status_kind"], "—")

        lin_html = []
        for li in r["lineage_items"]:
            lin_html.append(
                f"<div class='lin-row'><span class='lin-k'>{html.escape(li['kind'])}:</span> "
                f"<code>{html.escape(li['path'])}</code> <span class='ts'>({html.escape(li['mtime'])})</span></div>"
            )
        for qn in r["queued_next"]:
            pstat = f"Preview completed {qn['preview_done_mtime']}" if qn["preview_done"] else "Pre-built & ready to queue"
            lin_html.append(
                f"<div class='lin-row next-up'><span class='lin-k'>Queued Input ({html.escape(qn['phase'])}):</span> "
                f"<code>{html.escape(str(qn['input_json']))}</code> <span class='ts'>(created {html.escape(qn['input_mtime'])} · {html.escape(pstat)})</span></div>"
            )

        mp4_cell = f"<code>{html.escape(r['mp4_rel'])}</code>"
        if r["review_clip"] and r["review_clip"].startswith("/media/"):
            mp4_cell += f" <div style='margin-top:4px'><a href='{html.escape(r['review_clip'])}' target='_blank' class='clip-link'>▶ Watch Stacked Review Clip</a></div>"

        mini_cov_html = render_mini_barcode_html(r["cov_data"], r["n_frames"], row_id)
        frame_drilldown_html = render_frame_drilldown_html(r["cov_data"], r["first"], r["last"], r["subject"], row_id)
        dag_html = render_dag_html(r["dag_steps"], row_id)

        action_btns = (
            f"<div style='margin-top:6px; display:flex; gap:6px; flex-wrap:wrap;'>"
            f"<button class='btn-sm' onclick=\"toggleDrawer('{row_id}', this)\">▶ Frame Gaps &amp; Lineage Flow</button>"
            f"<button class='btn-sm btn-sm-accent' onclick=\"openInExplorer('{row_id}')\">☍ Open in Lineage Explorer</button>"
            f"</div>"
        )

        html_rows.append(
            f"<tr class='main-row' id='tr-{row_id}' data-row-id='{row_id}' data-status='{r['status_kind']}' data-subject='{r['subject']}'>"
            f"<td class='mono'><strong>{r['first']}–{r['last']}</strong><div class='sub'>{r['tc']} ({r['n_frames']}f)</div></td>"
            f"<td><span class='subj subj-{r['subject']}'>{html.escape(r['subject'])}</span></td>"
            f"<td><span class='badge {badge_cls}'>{html.escape(r['status_label'])}</span><div class='sub mono'>{html.escape(r['load'])}</div>{mini_cov_html}{action_btns}</td>"
            f"<td class='mp4-col'>{mp4_cell}</td>"
            f"<td class='mono ts-col'>{html.escape(r['mp4_mtime'])}</td>"
            f"<td class='lin-col'>{''.join(lin_html) if lin_html else '<span class=\"sub\">—</span>'}</td>"
            f"</tr>"
            f"<tr class='drawer-row' id='drawer-{row_id}' style='display:none;'>"
            f"<td colspan='6' class='drawer-cell'>"
            f"  <div class='drawer-hdr'>"
            f"    <strong>Frame-Level Coverage Drill-Down &amp; Hierarchical Data Flow Lineage — Frames {r['first']}–{r['last']} ({r['tc']}) · Subject: <code>{html.escape(r['subject'])}</code> · Load: <code>{html.escape(r['load'])}</code></strong>"
            f"    <button class='btn-sm' onclick=\"toggleDrawer('{row_id}')\">✕ Close Drawer</button>"
            f"  </div>"
            f"  {frame_drilldown_html}"
            f"  <div style='margin-top:14px; margin-bottom:8px; font-weight:600; font-size:13.5px; color:#93c5fd;'>6-Stage Hierarchical Data Flow Lineage (Source → Cast/Plan → Preview → Job/Text → Capture/Render → Cut)</div>"
            f"  {dag_html}"
            f"</td>"
            f"</tr>"
        )

        active_cls = " active" if idx_i == 1 else ""
        sidebar_items.append(
            f"<div class='sb-item{active_cls}' id='sb-{row_id}' data-row-id='{row_id}' data-status='{r['status_kind']}' data-subject='{r['subject']}' onclick=\"selectExplorerRow('{row_id}')\">"
            f"  <div class='sb-top'>"
            f"    <span class='mono'><strong>{r['first']}–{r['last']}</strong> <span class='sub'>({r['tc']})</span></span>"
            f"    <span class='badge {badge_cls}'>{short_badge}</span>"
            f"  </div>"
            f"  <div class='sb-bot'>"
            f"    <span class='subj subj-{r['subject']}'>{html.escape(r['subject'])}</span>"
            f"    <span class='mono sub'>{html.escape(r['cov_data']['short_summary'])}</span>"
            f"  </div>"
            f"</div>"
        )

        review_btn_html = ""
        if r["review_clip"] and r["review_clip"].startswith("/media/"):
            review_btn_html = f"<a href='{html.escape(r['review_clip'])}' target='_blank' class='btn btn-active-link'>▶ Watch Stacked Review Clip</a>"

        detail_panes.append(
            f"<div class='exp-pane' id='pane-{row_id}' style='display:{'block' if idx_i == 1 else 'none'};'>"
            f"  <div class='exp-pane-hdr'>"
            f"    <div>"
            f"      <h2 style='margin:0 0 4px 0; font-size:18px;'>Frames <code>{r['first']}–{r['last']}</code> ({r['tc']}, {r['n_frames']} frames) — Subject: <span class='subj subj-{r['subject']}'>{html.escape(r['subject'])}</span></h2>"
            f"      <div class='sub'>Status: <span class='badge {badge_cls}'>{html.escape(r['status_label'])}</span> · Load: <code>{html.escape(r['load'])}</code> · Output MP4: <code>{html.escape(r['mp4_rel'])}</code> ({html.escape(r['mp4_mtime'])})</div>"
            f"    </div>"
            f"    <div>{review_btn_html}</div>"
            f"  </div>"
            f"  {frame_drilldown_html}"
            f"  <div style='margin-top:16px; margin-bottom:8px; font-weight:600; font-size:14px; color:#93c5fd;'>6-Stage Hierarchical Data Flow Lineage (Source → Cast/Plan → Preview → Job/Text → Capture/Render → Cut)</div>"
            f"  {dag_html}"
            f"</div>"
        )

    # Build Tab 3 HTML rows (ETL Build Queue & Parent -> Child Execution DAG)
    etl_html_rows = []
    for r in rows:
        if r["subject"] == "—":
            continue
        row_id = r["row_id"]
        e = r["etl"]
        ost = e["overall_state"]
        ost_badge_cls = {
            "complete_fresh": "badge-green",
            "ready_to_lay": "badge-blue",
            "ready_to_run": "badge-blue",
            "stale_needs_rerun": "badge-amber",
            "blocked_by_parent": "badge-gray",
            "failed_gate": "badge-red",
        }.get(ost, "badge-gray")

        pill_html = []
        for nd in e["nodes"]:
            nd_cls = {
                "done": "etl-pill-done",
                "ready": "etl-pill-ready",
                "stale": "etl-pill-stale",
                "blocked": "etl-pill-blocked",
                "failed": "etl-pill-failed",
                "pending": "etl-pill-pending",
            }.get(nd["status"], "etl-pill-pending")
            nd_icon = {
                "done": "✓",
                "ready": "▶",
                "stale": "↻",
                "blocked": "🔒",
                "failed": "✕",
                "pending": "○",
            }.get(nd["status"], "○")
            pill_html.append(
                f"<div class='etl-pill {nd_cls}' title='{html.escape(nd['short'] + ': ' + nd['detail'])}'>"
                f"<span class='etl-pill-ic'>{nd_icon}</span><span>{html.escape(nd['short'])}</span>"
                f"</div>"
            )

        cmd_html = f"<code class='etl-cmd-box'>{html.escape(e['next_cmd'])}</code>"
        if e["runner_cmd"] != "—":
            cmd_html += (
                f"<div style='margin-top:5px; display:flex; gap:6px; align-items:center; flex-wrap:wrap;'>"
                f"<span class='sub'>Runner: <code>{html.escape(e['runner_cmd'])}</code></span>"
                f"<button class='btn-sm' onclick=\"navigator.clipboard.writeText({html.escape(json.dumps(e['next_cmd']))}); this.textContent='✓ Copied!'\">📋 Copy Cmd</button>"
                f"<button class='btn-sm btn-sm-accent' onclick=\"openInExplorer('{row_id}')\">☍ Open Lineage</button>"
                f"</div>"
            )
        else:
            cmd_html += (
                f"<div style='margin-top:5px;'>"
                f"<button class='btn-sm btn-sm-accent' onclick=\"openInExplorer('{row_id}')\">☍ Open Lineage</button>"
                f"</div>"
            )

        etl_html_rows.append(
            f"<tr class='etl-row' data-row-id='{row_id}' data-etl-state='{ost}' data-etl-stage='{e['etl_stage']}' data-subject='{r['subject']}'>"
            f"<td class='mono'><strong>{r['first']}–{r['last']}</strong><div class='sub'>{r['tc']} ({r['n_frames']}f)</div></td>"
            f"<td><span class='subj subj-{r['subject']}'>{html.escape(r['subject'])}</span><div class='sub' style='margin-top:4px;'><code>{html.escape(e['pass_badge'])}</code></div></td>"
            f"<td class='mono'><code>{html.escape(e['active_load'])}</code><div class='sub'>Laid: <code>{html.escape(e['laid_load'])}</code></div></td>"
            f"<td><span class='badge {ost_badge_cls}'>{html.escape(e['overall_label'])}</span>"
            f"<div class='sub' style='margin-top:4px;'>{html.escape(e['reason'])}</div>"
            f"<div class='sub' style='margin-top:3px; color:#93c5fd;'><strong>Measured on:</strong> <code>{html.escape(e['measured_on'])}</code></div></td>"
            f"<td><div class='etl-strip'>{''.join(pill_html)}</div>"
            f"<div class='sub' style='margin-top:5px;'><strong>Verify in data (`cannot_see`):</strong> {html.escape(e['cannot_see'])}</div></td>"
            f"<td>{cmd_html}</td>"
            f"</tr>"
        )

    joined_name = latest_joined.name if latest_joined else "none"
    joined_mtime = latest_joined_mtime
    cov_name = cov_json_path.name if cov_json_path else "none"
    cov_mtime = path_mtime_local(cov_json_path)

    page_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fun (0000–6394) — Cut & Hierarchical Data Flow Provenance</title>
<style>
  :root {{
    --bg: #0f1115;
    --panel: #161920;
    --panel2: #1c2029;
    --border: #262b36;
    --text: #e6e9ef;
    --muted: #9aa3b2;
    --accent: #60a5fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 20px 28px;
    line-height: 1.45;
  }}
  .top-nav {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 14px;
    font-size: 13.5px;
    flex-wrap: wrap;
  }}
  .top-nav a {{
    color: var(--accent);
    text-decoration: none;
    padding: 6px 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  .top-nav a:hover {{ background: #1f2430; }}
  h1 {{ margin: 0 0 6px 0; font-size: 21px; }}
  .summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin: 14px 0 18px 0;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }}
  .card .k {{ color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card .v {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
  .card .sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}

  /* View Mode Tabs */
  .view-tabs {{
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }}
  .view-tab {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 9px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
  }}
  .view-tab.active {{
    background: #1d4ed8;
    border-color: #3b82f6;
    color: #fff;
  }}

  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
    align-items: center;
  }}
  .btn {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    text-decoration: none;
  }}
  .btn.active, .btn-active-link {{
    background: #1d4ed8;
    border-color: #3b82f6;
    color: #fff;
  }}
  .btn-sm {{
    background: #1e2330;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 3px 8px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 11.5px;
  }}
  .btn-sm:hover {{ background: #283042; color: #fff; }}
  .btn-sm-accent {{
    background: rgba(37, 99, 235, 0.2);
    border-color: rgba(59, 130, 246, 0.5);
    color: #93c5fd;
  }}
  input.search {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    min-width: 250px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    font-size: 13px;
  }}
  th, td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background: #1a1e27;
    color: var(--muted);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    position: sticky;
    top: 0;
    z-index: 2;
  }}
  tr.main-row:hover td {{ background: #1b202b; }}
  .drawer-cell {{
    background: #12151c;
    padding: 16px 22px;
    border-bottom: 2px solid #3b82f6;
  }}
  .drawer-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #0b0d12;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 12px;
    color: #d1d5db;
    word-break: break-all;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge-green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
  .badge-blue {{ background: rgba(59, 130, 246, 0.18); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }}
  .badge-amber {{ background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }}
  .badge-red {{ background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.35); }}
  .badge-gray {{ background: rgba(156, 163, 175, 0.12); color: #9ca3af; border: 1px solid rgba(156, 163, 175, 0.25); }}
  .subj {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11.5px;
  }}
  .subj-lead {{ background: #1e3a8a; color: #bfdbfe; }}
  .subj-second {{ background: #701a75; color: #f5d0fe; }}
  .subj-third {{ background: #065f46; color: #a7f3d0; }}
  .lin-row {{ margin-bottom: 4px; }}
  .lin-k {{ color: var(--muted); font-weight: 500; }}
  .next-up .lin-k {{ color: #fcd34d; }}
  .ts {{ color: var(--muted); font-size: 11.5px; }}
  .clip-link {{ color: var(--accent); text-decoration: none; font-size: 12px; font-weight: 500; }}
  .clip-link:hover {{ text-decoration: underline; }}

  /* Hierarchical DAG Tree Styles */
  .dag-tree {{
    display: flex;
    flex-direction: column;
    gap: 0;
  }}
  .dag-step {{
    display: flex;
    gap: 14px;
    position: relative;
  }}
  .dag-node-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 28px;
    flex-shrink: 0;
  }}
  .dag-circle {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    background: #1f2937;
    border: 2px solid #4b5563;
    color: #9ca3af;
    z-index: 1;
  }}
  .dag-done .dag-circle {{
    background: rgba(16, 185, 129, 0.2);
    border-color: #10b981;
    color: #34d399;
  }}
  .dag-ready .dag-circle {{
    background: rgba(245, 158, 11, 0.2);
    border-color: #f59e0b;
    color: #fcd34d;
  }}
  .dag-line {{
    width: 2px;
    flex-grow: 1;
    background: #2d3748;
    min-height: 14px;
  }}
  .dag-done .dag-line {{ background: rgba(16, 185, 129, 0.45); }}
  .dag-step:last-child .dag-line {{ display: none; }}
  .dag-card {{
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }}
  .dag-done .dag-card {{ border-left: 3px solid #10b981; }}
  .dag-ready .dag-card {{ border-left: 3px solid #f59e0b; }}
  .dag-pending .dag-card {{ border-left: 3px solid #4b5563; opacity: 0.82; }}
  .dag-card-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 4px;
  }}
  .dag-step-title {{ font-weight: 600; font-size: 14px; color: #f3f4f6; }}
  .dag-step-badge {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
  }}
  .dag-done-badge {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
  .dag-ready-badge {{ background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }}
  .dag-pending-badge {{ background: rgba(156, 163, 175, 0.12); color: #9ca3af; border: 1px solid rgba(156, 163, 175, 0.25); }}
  .dag-code-line {{
    font-size: 11.5px;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .dag-kv-list {{
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .dag-kv {{
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 10px;
    font-size: 12.5px;
    padding: 4px 0;
    border-top: 1px dashed rgba(255,255,255,0.06);
  }}
  .dag-k {{ color: var(--muted); font-weight: 500; }}
  .dag-v {{ color: var(--text); word-break: break-word; }}
  .dag-ts {{ color: #93c5fd; font-size: 11.5px; margin-left: 6px; font-family: ui-monospace, monospace; }}
  .prompt-details {{
    margin-top: 8px;
    background: #0b0d12;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
  }}
  .prompt-details summary {{
    cursor: pointer;
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
  }}
  .prompt-pre {{
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    color: #d1d5db;
    margin: 8px 0 2px 0;
    line-height: 1.4;
  }}

  /* Explorer Split Layout (Tab 2) */
  .explorer-layout {{
    display: grid;
    grid-template-columns: 330px 1fr;
    gap: 18px;
    align-items: start;
  }}
  .explorer-sidebar {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    max-height: 78vh;
    overflow-y: auto;
    padding: 8px;
    position: sticky;
    top: 12px;
  }}
  .sb-item {{
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    margin-bottom: 4px;
  }}
  .sb-item:hover {{ background: #1c222e; }}
  .sb-item.active {{
    background: rgba(37, 99, 235, 0.18);
    border-color: #3b82f6;
  }}
  .sb-top, .sb-bot {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
  }}
  .sb-bot {{ margin-top: 4px; }}
  .exp-pane-hdr {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }}

  /* Frame-Level Coverage Barcode & Drill-Down Styles */
  .mini-cov-wrap {{
    margin-top: 6px;
    cursor: pointer;
    background: #0d1016;
    border: 1px solid #262c3a;
    border-radius: 5px;
    padding: 4px 6px;
  }}
  .mini-cov-wrap:hover {{ border-color: #3b82f6; }}
  .fbar-mini {{
    display: flex;
    height: 7px;
    border-radius: 3px;
    overflow: hidden;
    background: #1f2430;
    gap: 1px;
  }}
  .fbar-seg {{ height: 100%; min-width: 2px; }}
  .mini-cov-txt {{
    font-size: 10.5px;
    color: var(--muted);
    margin-top: 3px;
  }}
  .fcat-carried {{ background: #10b981; }}
  .fcat-held {{ background: #06b6d4; }}
  .fcat-emptied {{ background: #f59e0b; }}
  .fcat-unrendered_gap {{ background: #ef4444; }}
  .fcat-no_capture {{ background: #374151; }}

  .cov-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
  }}
  .cov-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }}
  .cov-legend {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 11.5px;
    color: var(--muted);
  }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; }}
  .leg-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }}
  .fbar-lg-wrap {{
    margin: 8px 0 12px 0;
    background: #0b0d12;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
  }}
  .fbar-lg-scale {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    font-family: ui-monospace, monospace;
    margin-bottom: 4px;
  }}
  .fbar-lg {{
    display: flex;
    height: 22px;
    border-radius: 4px;
    overflow: hidden;
    background: #1f2430;
    gap: 1px;
  }}
  .fbar-lg-seg {{
    height: 100%;
    min-width: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-family: ui-monospace, monospace;
    color: #0b0d12;
    font-weight: 700;
    overflow: hidden;
    white-space: nowrap;
  }}
  .fbar-lg-seg.fcat-no_capture, .fbar-lg-seg.fcat-unrendered_gap {{ color: #fff; }}
  .cov-span-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    background: #11141b;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .cov-span-table th, .cov-span-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
  }}
  .cov-span-table th {{
    background: #171b24;
    font-size: 11px;
  }}
  /* Tab 3: ETL Build Queue & Parent -> Child Execution DAG */
  .etl-stage-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .etl-stage-card {{
    background: #151923;
    border: 1px solid #2b3346;
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .etl-stage-card h3 {{
    margin: 0 0 6px 0;
    font-size: 13.5px;
    color: #93c5fd;
  }}
  .etl-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: center;
  }}
  .etl-pill {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 7px;
    border-radius: 5px;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    border: 1px solid transparent;
    cursor: help;
    white-space: nowrap;
  }}
  .etl-pill-ic {{ font-weight: 700; }}
  .etl-pill-done {{ background: rgba(16, 185, 129, 0.14); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.35); }}
  .etl-pill-ready {{ background: rgba(59, 130, 246, 0.22); color: #93c5fd; border-color: rgba(59, 130, 246, 0.55); font-weight: 600; }}
  .etl-pill-stale {{ background: rgba(245, 158, 11, 0.18); color: #fcd34d; border-color: rgba(245, 158, 11, 0.45); }}
  .etl-pill-blocked {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border-color: rgba(148, 163, 184, 0.3); }}
  .etl-pill-failed {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border-color: rgba(239, 68, 68, 0.5); font-weight: 600; }}
  .etl-pill-pending {{ background: #12151e; color: #64748b; border-color: #222734; }}
  .etl-cmd-box {{
    display: block;
    background: #0b0d12;
    border: 1px solid #262b36;
    padding: 5px 8px;
    border-radius: 5px;
    font-size: 11.5px;
    color: #e2e8f0;
    word-break: break-all;
  }}
</style>
</head>
<body data-laid-sha256="{laid_sha256}">
  <div class="top-nav">
    <a href="/index.html">← Review Queue (TO_REVIEW)</a>
    <a href="/PROVENANCE.html"><strong>Cut, Frame Coverage &amp; Hierarchical Data Flow Provenance</strong></a>
    <a href="/CUT_MAP.html">Cut Map Alias</a>
    <span style="color:var(--muted); margin-left:auto;">Last Updated: <strong>{html.escape(now_local)}</strong> · <code>data/fun/DAG_STATUS.md</code></span>
  </div>

  <h1><code>fun</code> (Frames 0–6394) — Chronological Cut, Frame-Level Gap Audit &amp; Parent → Child Execution DAG</h1>
  <div style="color:var(--muted); font-size:13px;">
    Live provenance, frame-level gap audit, and topological ETL build queue generated directly from <code>data/fun/shots.json</code>, <code>data/fun/plan.json</code>, <code>data/fun/cast/cast.jsonl</code>, <code>data/fun/laid.jsonl</code>, <code>{html.escape(cov_name)}</code>, and <code>data/*_job/</code>. All timestamps in {html.escape(tz_abbr)}.
  </div>

  <div class="summary-cards">
    <div class="card">
      <div class="k">Subject-Shots In Cut</div>
      <div class="v">{counts['in_cut']} / {counts['total_subj_shots']}</div>
      <div class="sub"><span style="color:#34d399">{counts['clean_in_cut']} clean 100%</span> · <span style="color:#fcd34d">{counts['in_cut_rerun']} with frame gap / queued re-run</span> · <span style="color:#fca5a5">{counts['pending']} pending</span></div>
    </div>
    <div class="card">
      <div class="k">ETL Build Queue (Parent → Child DAG)</div>
      <div class="v"><span style="color:#60a5fa">{etl_counts['ready_to_run'] + etl_counts['ready_to_lay']} Ready</span> · <span style="color:#fcd34d">{etl_counts['stale_needs_rerun']} Rerun</span> · <span style="color:#94a3b8">{etl_counts['blocked_by_parent']} Blocked</span></div>
      <div class="sub">Stage A (S2): {etl_counts['stage_a_done']}/{etl_counts['stage_a_total']} · Stage C (Multi): {etl_counts['stage_c_done']}/{etl_counts['stage_c_total']} · Stage D (Single): {etl_counts['stage_d_done']}/{etl_counts['stage_d_total']}</div>
    </div>
    <div class="card">
      <div class="k">Latest Full Joined Cut MP4</div>
      <div class="v" style="font-size:13px; word-break:break-all;"><code>{html.escape(joined_name)}</code></div>
      <div class="sub">Created: {html.escape(joined_mtime)}</div>
    </div>
    <div class="card">
      <div class="k">Agent &amp; CLI DAG Runbook</div>
      <div class="v" style="font-size:13px;"><code>data/fun/DAG_STATUS.md</code></div>
      <div class="sub">Run: <code>python bench/build_cut_map.py --next</code> or <code>--run</code></div>
    </div>
  </div>

  <div class="view-tabs">
    <button class="view-tab active" id="tabBtnTable" onclick="switchMainTab('table')">1. Chronological Cut Table (0–6394) + Frame-Level Coverage &amp; Lineage Drawers</button>
    <button class="view-tab" id="tabBtnExplorer" onclick="switchMainTab('explorer')">2. Hierarchical Data Flow &amp; Frame Coverage Explorer (Master-Detail DAG)</button>
    <button class="view-tab" id="tabBtnEtl" onclick="switchMainTab('etl')">3. ETL Build Queue &amp; Parent → Child Execution DAG ({etl_counts['total_subj_shots'] if 'total_subj_shots' in etl_counts else counts['total_subj_shots']} Subject-Shots)</button>
  </div>

  <div class="controls" id="mainControls">
    <button class="btn active" onclick="filterRows('all', this)">All Rows ({len(rows)})</button>
    <button class="btn" onclick="filterRows('in_cut_clean', this)">Clean In Cut 100% ({counts['clean_in_cut']})</button>
    <button class="btn" onclick="filterRows('in_cut_rerun', this)">In Cut w/ Gap or Queued Re-run ({counts['in_cut_rerun']})</button>
    <button class="btn" onclick="filterRows('pending', this)">Pending Render ({counts['pending']})</button>
    <button class="btn" onclick="filterRows('todo', this)">All Needs Work: Gap / Re-run / Pending ({counts['in_cut_rerun'] + counts['pending']})</button>
    <button class="btn" onclick="filterRows('in_cut_today', this)">Rendered Today ({counts['today_updated']})</button>
    <button class="btn" onclick="filterRows('second', this)">Subject 2 Only</button>
    <input type="text" class="search" id="searchBox" placeholder="Filter by frame, load name, status, or file..." oninput="applySearch()">
  </div>

  <!-- TAB 1: CHRONOLOGICAL TABLE WITH EXPANDABLE LINEAGE DRAWERS -->
  <div id="viewTable">
    <table id="cutTable">
      <thead>
        <tr>
          <th style="width:120px;">Frames &amp; TC</th>
          <th style="width:75px;">Subject</th>
          <th style="width:270px;">Status, Frame Coverage Barcode &amp; Actions</th>
          <th style="width:280px;">Final Rendered MP4 (under <code>H3_OUTPUT_DIR/</code>)</th>
          <th style="width:170px;">MP4 Created ({html.escape(tz_abbr)})</th>
          <th>Summary Data Folders, Input JSONs &amp; Timestamps ({html.escape(tz_abbr)})</th>
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
  </div>

  <!-- TAB 2: INTERACTIVE HIERARCHICAL DATA FLOW LINEAGE EXPLORER -->
  <div id="viewExplorer" style="display:none;">
    <div class="explorer-layout">
      <div class="explorer-sidebar" id="explorerSidebar">
        {''.join(sidebar_items)}
      </div>
      <div class="explorer-main" id="explorerMain">
        {''.join(detail_panes)}
      </div>
    </div>
  </div>

  <!-- TAB 3: ETL BUILD QUEUE & PARENT -> CHILD EXECUTION DAG -->
  <div id="viewEtl" style="display:none;">
    <div class="etl-stage-grid">
      <div class="etl-stage-card">
        <h3>Stage A · Pass 1 (<code>second</code>, 19 Shots)</h3>
        <div style="font-size:16px; font-weight:700; margin-bottom:4px;">{etl_counts['stage_a_done']} / {etl_counts['stage_a_total']} Landed ({etl_counts['stage_a_laid']} Laid)</div>
        <div class="sub">Whole-region <code>face_and_neck + hair</code> on original source plate. Must finish before Stage B &amp; C.</div>
        <div style="margin-top:6px;"><code>python bench/build_cut_map.py --run --stage stage-a</code></div>
      </div>
      <div class="etl-stage-card">
        <h3>Stage B · Lay Pass 1 &amp; Build 24p Post-Pass-1 Cut</h3>
        <div style="font-size:16px; font-weight:700; margin-bottom:4px;">{"Ready to Run Now" if etl_counts['stage_a_done'] == etl_counts['stage_a_total'] else f"Waiting on {etl_counts['stage_a_total'] - etl_counts['stage_a_done']} Stage-A Load(s)"}</div>
        <div class="sub">Updates <code>data/fun/laid.jsonl</code> &amp; runs <code>rebuild_delivery.py --clip</code> to create the 24.0 fps source plate for Pass 2.</div>
        <div style="margin-top:6px;"><code>python bench/build_cut_map.py --run --stage stage-b</code></div>
      </div>
      <div class="etl-stage-card">
        <h3>Stage C · Multi-Pass Pass 2/3 (<code>lead</code> &amp; <code>third</code>)</h3>
        <div style="font-size:16px; font-weight:700; margin-bottom:4px;">{etl_counts['stage_c_done']} / {etl_counts['stage_c_total']} Complete in Cut</div>
        <div class="sub">Runs <code>preview --remake</code> → <code>job</code> → <code>reference</code> → <code>text</code> → <code>render</code> on the Post-Pass-1 Cut MP4.</div>
        <div style="margin-top:6px;"><code>python bench/build_cut_map.py --run --stage stage-c</code></div>
      </div>
      <div class="etl-stage-card">
        <h3>Stage D · Single-Subject <code>lead</code> &amp; <code>third</code> Shots</h3>
        <div style="font-size:16px; font-weight:700; margin-bottom:4px;">{etl_counts['stage_d_done']} / {etl_counts['stage_d_total']} Complete in Cut</div>
        <div class="sub">Independent single-subject shots on original source plate (<code>data/2026-10-20_fun_phase3_inputs/</code>).</div>
        <div style="margin-top:6px;"><code>python bench/build_cut_map.py --run --stage stage-d</code></div>
      </div>
    </div>

    <div class="controls" id="etlControls">
      <button class="btn active" onclick="filterEtlRows('all', this)">All Subject-Shots ({counts['total_subj_shots']})</button>
      <button class="btn" onclick="filterEtlRows('actionable', this)">All Remaining / Needs Action ({counts['total_subj_shots'] - etl_counts['complete_fresh']})</button>
      <button class="btn" onclick="filterEtlRows('ready_to_run', this)">⚡ Ready to Run Now ({etl_counts['ready_to_run']})</button>
      <button class="btn" onclick="filterEtlRows('ready_to_lay', this)">📥 Landed — Ready to Lay ({etl_counts['ready_to_lay']})</button>
      <button class="btn" onclick="filterEtlRows('stale_needs_rerun', this)">🔄 Needs Rerun / Stale / Gap ({etl_counts['stale_needs_rerun']})</button>
      <button class="btn" onclick="filterEtlRows('blocked_by_parent', this)">🔒 Blocked by Pass 1 ({etl_counts['blocked_by_parent']})</button>
      <button class="btn" onclick="filterEtlRows('failed_gate', this)">❌ Gate Blocked ({etl_counts['failed_gate']})</button>
      <button class="btn" onclick="filterEtlRows('complete_fresh', this)">✅ Complete &amp; Fresh ({etl_counts['complete_fresh']})</button>
      <button class="btn" onclick="filterEtlRows('stage_a', this)">Stage A (Pass 1 S2)</button>
      <button class="btn" onclick="filterEtlRows('stage_c', this)">Stage C (Multi-Pass)</button>
      <button class="btn" onclick="filterEtlRows('stage_d', this)">Stage D (Single-Subj)</button>
    </div>

    <table id="etlTable">
      <thead>
        <tr>
          <th style="width:115px;">Shot &amp; TC</th>
          <th style="width:110px;">Subject &amp; Pass</th>
          <th style="width:165px;">Active Load &amp; Cut Load</th>
          <th style="width:260px;">Overall ETL State &amp; Parent/Gap Reason</th>
          <th>8-Stage ETL Pipeline Status (Hover any pill for file, timestamp &amp; parent details)</th>
          <th style="width:360px;">What Needs to Run Next (CLI Command)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(etl_html_rows)}
      </tbody>
    </table>
  </div>

<script>
let currentMode = 'all';
let currentEtlMode = 'all';

function switchMainTab(which) {{
  document.getElementById('viewTable').style.display = (which === 'table') ? 'block' : 'none';
  document.getElementById('viewExplorer').style.display = (which === 'explorer') ? 'block' : 'none';
  document.getElementById('viewEtl').style.display = (which === 'etl') ? 'block' : 'none';
  document.getElementById('mainControls').style.display = (which === 'etl') ? 'none' : 'flex';
  document.getElementById('tabBtnTable').classList.toggle('active', which === 'table');
  document.getElementById('tabBtnExplorer').classList.toggle('active', which === 'explorer');
  document.getElementById('tabBtnEtl').classList.toggle('active', which === 'etl');
  if (which === 'etl') {{
    window.location.hash = 'etl';
  }}
}}

function filterEtlRows(mode, btn) {{
  currentEtlMode = mode;
  document.querySelectorAll('#etlControls .btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  applyEtlSearch();
}}

function applyEtlSearch() {{
  const q = (document.getElementById('searchBox').value || '').toLowerCase();
  document.querySelectorAll('#etlTable tbody tr.etl-row').forEach(tr => {{
    const ost = tr.getAttribute('data-etl-state');
    const stg = tr.getAttribute('data-etl-stage');
    let show = true;
    if (currentEtlMode === 'actionable') show = (ost !== 'complete_fresh');
    else if (currentEtlMode === 'stage_a' || currentEtlMode === 'stage_c' || currentEtlMode === 'stage_d') show = (stg === currentEtlMode);
    else if (currentEtlMode !== 'all') show = (ost === currentEtlMode);
    if (show && q) {{
      show = tr.innerText.toLowerCase().includes(q);
    }}
    tr.style.display = show ? '' : 'none';
  }});
}}

function toggleDrawer(rowId, btn) {{
  const dr = document.getElementById('drawer-' + rowId);
  if (!dr) return;
  const open = dr.style.display === 'none';
  dr.style.display = open ? '' : 'none';
  if (btn) {{
    btn.textContent = open ? '▼ Hide Frame Gaps & Lineage' : '▶ Frame Gaps & Lineage Flow';
  }}
}}

function selectExplorerRow(rowId) {{
  document.querySelectorAll('.sb-item').forEach(el => el.classList.remove('active'));
  const sb = document.getElementById('sb-' + rowId);
  if (sb) {{
    sb.classList.add('active');
    sb.scrollIntoView({{ block: 'nearest' }});
  }}
  document.querySelectorAll('.exp-pane').forEach(el => el.style.display = 'none');
  const pane = document.getElementById('pane-' + rowId);
  if (pane) pane.style.display = 'block';
}}

function openInExplorer(rowId) {{
  switchMainTab('explorer');
  selectExplorerRow(rowId);
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function filterRows(mode, btn) {{
  currentMode = mode;
  document.querySelectorAll('#mainControls .btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  applySearch();
}}

function rowMatchesFilter(st, subj, textContent, q) {{
  let show = true;
  if (currentMode === 'in_cut_all') show = st.startsWith('in_cut');
  else if (currentMode === 'in_cut_clean') show = (st === 'in_cut' || st === 'in_cut_today');
  else if (currentMode === 'in_cut_rerun') show = (st === 'in_cut_rerun');
  else if (currentMode === 'pending') show = (st === 'pending');
  else if (currentMode === 'in_cut_today') show = (st === 'in_cut_today');
  else if (currentMode === 'todo') show = (st === 'pending' || st === 'in_cut_rerun');
  else if (currentMode === 'second') show = (subj === 'second');
  if (show && q) {{
    show = textContent.toLowerCase().includes(q);
  }}
  return show;
}}

function applySearch() {{
  const q = (document.getElementById('searchBox').value || '').toLowerCase();
  document.querySelectorAll('#cutTable tbody tr.main-row').forEach(tr => {{
    const rowId = tr.getAttribute('data-row-id');
    const st = tr.getAttribute('data-status');
    const subj = tr.getAttribute('data-subject');
    const show = rowMatchesFilter(st, subj, tr.innerText, q);
    tr.style.display = show ? '' : 'none';
    if (!show) {{
      const dr = document.getElementById('drawer-' + rowId);
      if (dr) dr.style.display = 'none';
    }}
  }});
  document.querySelectorAll('#explorerSidebar .sb-item').forEach(sb => {{
    const rowId = sb.getAttribute('data-row-id');
    const st = sb.getAttribute('data-status');
    const subj = sb.getAttribute('data-subject');
    const pane = document.getElementById('pane-' + rowId);
    const searchTxt = sb.innerText + ' ' + (pane ? pane.innerText : '');
    const show = rowMatchesFilter(st, subj, searchTxt, q);
    sb.style.display = show ? '' : 'none';
  }});
  applyEtlSearch();
}}

if (window.location.hash === '#etl') {{
  switchMainTab('etl');
}}
</script>
</body>
</html>
"""
    (FUN_DIR / "poc" / "CUT_MAP.html").write_text(page_html)
    (FUN_DIR / "poc" / "PROVENANCE.html").write_text(page_html)
    share_fun = SHARE_DIR / "Video" / "fun"
    if share_fun.is_dir():
        try:
            (share_fun / "PROVENANCE.html").write_text(page_html)
            (share_fun / "CUT_MAP.html").write_text(page_html)
        except Exception:
            pass
    print(
        f"Wrote {FUN_DIR / 'DAG_STATUS.md'}, {FUN_DIR / 'DAG_STATUS.json'}, {FUN_DIR / 'CUT_MAP.md'}, "
        f"{FUN_DIR / 'poc' / 'PROVENANCE.html'}, and {FUN_DIR / 'poc' / 'CUT_MAP.html'} ({len(rows)} rows)"
    )

    ap = argparse.ArgumentParser(description="Build PROVENANCE.html + DAG_STATUS.md (read-only execution & artifact ledger) and inspect actionable DAG items.")
    ap.add_argument("--video", default="fun", help="Video project name (default: fun)")
    ap.add_argument("--next", action="store_true", help="Print the ordered ETL work queue with exact disk paths, timestamps, and cannot_see verification notes")
    ap.add_argument("--stage", choices=["stage-a", "stage-b", "stage-c", "stage-d"], help="Filter by macro stage")
    ap.add_argument("--shot", type=int, help="Filter by shot first frame (e.g. 1093)")
    ap.add_argument("--subject", choices=["lead", "second", "third"], help="Filter by subject")
    ap.add_argument("--load", help="Filter by load name")
    args, _ = ap.parse_known_args()

    if args.next or args.stage or args.shot is not None or args.subject or args.load:
        target_rows = list(actionable_rows) if not (args.shot is not None or args.load) else [r for r in rows if r["subject"] != "—"]
        if args.stage:
            st_key = args.stage.replace("-", "_")
            target_rows = [r for r in target_rows if r["etl"]["etl_stage"] == st_key]
        if args.shot is not None:
            target_rows = [r for r in target_rows if r["first"] == args.shot]
        if args.subject:
            target_rows = [r for r in target_rows if r["subject"] == args.subject]
        if args.load:
            target_rows = [r for r in target_rows if r["etl"]["active_load"] == args.load]

        print(f"\n=== ETL EXECUTION & ARTIFACT LEDGER ({len(target_rows)} matching rows, updated {now_local}) ===")
        for i, r in enumerate(target_rows, 1):
            e = r["etl"]
            print(f"[{i:02d}] {e['stage_label']} | Shot {r['first']}–{r['last']} ({r['subject']}) | Load: {e['active_load']} | {e['overall_label']}")
            print(f"     State:       {e['reason']}")
            print(f"     Measured on: {e['measured_on']}")
            print(f"     Cannot see:  {e['cannot_see']}")
            print(f"     Next Cmd:    {e['next_cmd']}")


if __name__ == "__main__":
    main()
