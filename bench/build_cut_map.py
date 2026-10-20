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

import datetime
import html
import json
import os
import re
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
    else:
        latest_joined = None
        latest_joined_mtime = "—"

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
</style>
</head>
<body>
  <div class="top-nav">
    <a href="/index.html">← Review Queue (TO_REVIEW)</a>
    <a href="/PROVENANCE.html"><strong>Cut, Frame Coverage &amp; Hierarchical Data Flow Provenance</strong></a>
    <a href="/CUT_MAP.html">Cut Map Alias</a>
    <span style="color:var(--muted); margin-left:auto;">Generated: {html.escape(now_local)}</span>
  </div>

  <h1><code>fun</code> (Frames 0–6394) — Chronological Cut, Frame-Level Gap Audit &amp; Hierarchical Data Flow Lineage</h1>
  <div style="color:var(--muted); font-size:13px;">
    Live provenance &amp; frame-level gap audit generated directly from <code>data/fun/shots.json</code>, <code>data/fun/plan.json</code>, <code>data/fun/cast/cast.jsonl</code>, <code>data/fun/laid.jsonl</code>, <code>{html.escape(cov_name)}</code>, and <code>data/*_job/</code>. All timestamps in {html.escape(tz_abbr)}.
  </div>

  <div class="summary-cards">
    <div class="card">
      <div class="k">Subject-Shots In Cut</div>
      <div class="v">{counts['in_cut']} / {counts['total_subj_shots']}</div>
      <div class="sub"><span style="color:#34d399">{counts['clean_in_cut']} clean 100%</span> · <span style="color:#fcd34d">{counts['in_cut_rerun']} with frame gap / queued re-run</span> · <span style="color:#fca5a5">{counts['pending']} pending</span></div>
    </div>
    <div class="card">
      <div class="k">Latest Full Joined Cut MP4</div>
      <div class="v" style="font-size:13px; word-break:break-all;"><code>{html.escape(joined_name)}</code></div>
      <div class="sub">Created: {html.escape(joined_mtime)}</div>
    </div>
    <div class="card">
      <div class="k">Frame Coverage &amp; Runbook</div>
      <div class="v" style="font-size:13px;"><code>{html.escape(cov_name)}</code></div>
      <div class="sub">Created: {html.escape(cov_mtime)} · Plus <code>data/fun/CUT_MAP.md</code></div>
    </div>
  </div>

  <div class="view-tabs">
    <button class="view-tab active" id="tabBtnTable" onclick="switchMainTab('table')">1. Chronological Cut Table (0–6394) + Frame-Level Coverage &amp; Lineage Drawers</button>
    <button class="view-tab" id="tabBtnExplorer" onclick="switchMainTab('explorer')">2. Hierarchical Data Flow &amp; Frame Coverage Explorer (Master-Detail DAG)</button>
  </div>

  <div class="controls">
    <button class="btn active" onclick="filterRows('all', this)">All Rows ({len(rows)})</button>
    <button class="btn" onclick="filterRows('in_cut_clean', this)">Clean In Cut 100% ({counts['clean_in_cut']})</button>
    <button class="btn" onclick="filterRows('in_cut_rerun', this)">In Cut w/ Gap or Queued Re-run ({counts['in_cut_rerun']})</button>
    <button class="btn" onclick="filterRows('pending', this)">Pending Render ({counts['pending']})</button>
    <button class="btn" onclick="filterRows('todo', this)">All Needs Work: Gap / Re-run / Pending ({counts['in_cut_rerun'] + counts['pending']})</button>
    <button class="btn" onclick="filterRows('in_cut_today', this)">Rendered Today ({counts['today_updated']})</button>
    <button class="btn" onclick="filterRows('second', this)">Subject 2 Only</button>
    <input type="text" class="search" id="searchBox" placeholder="Filter by frame, load name, or file..." oninput="applySearch()">
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

<script>
let currentMode = 'all';

function switchMainTab(which) {{
  const isTable = (which === 'table');
  document.getElementById('viewTable').style.display = isTable ? 'block' : 'none';
  document.getElementById('viewExplorer').style.display = isTable ? 'none' : 'block';
  document.getElementById('tabBtnTable').classList.toggle('active', isTable);
  document.getElementById('tabBtnExplorer').classList.toggle('active', !isTable);
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
  document.querySelectorAll('.controls .btn').forEach(b => b.classList.remove('active'));
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
}}
</script>
</body>
</html>
"""
    (FUN_DIR / "poc" / "CUT_MAP.html").write_text(page_html)
    (FUN_DIR / "poc" / "PROVENANCE.html").write_text(page_html)
    print(f"Wrote {FUN_DIR / 'CUT_MAP.md'}, {FUN_DIR / 'poc' / 'PROVENANCE.html'}, and {FUN_DIR / 'poc' / 'CUT_MAP.html'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
