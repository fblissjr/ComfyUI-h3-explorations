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
    runs: list[list[int]] = []
    for n in nums:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


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


def find_reference_records(capture: Path, loads: list[str], flags_data: dict) -> list[tuple[str, Path]]:
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

    job_candidates = [
        capture.with_name(capture.name + "_job") / "job.plan.json",
        capture.with_name(capture.name + "_job") / "job.render.json",
    ]
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
    print(
        f"=== capture: {capture.name} | load(s): {', '.join(loads)} | frames: {span[0]}-{span[1]} "
        f"| written: {status.get('written')} by {status.get('made_by')} | gate: {verdict} ({len(flags_list)} flag(s)) ===",
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
    }


def record_initial(capture: Path, who: str, loads: list[str], out=sys.stdout) -> dict:
    """Record `who` as having read the flags printout on `capture` for `loads`."""
    who_clean = (who or "").strip()
    if not who_clean:
        raise Refused("--initial WHO requires a non-empty reader name")
    status = json.loads((capture / "status.json").read_text())
    flags_bytes = (capture / "flags.json").read_bytes()
    flags_sha = hashlib.sha256(flags_bytes).hexdigest()
    now = now_iso()
    entry = {
        "by": who_clean,
        "at": now,
        "loads": list(loads),
        "capture_written": status.get("written"),
        "capture_made_by": status.get("made_by"),
        "flags_sha256": flags_sha,
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
    return {
        "by": str(rec["by"]).strip(),
        "at": rec["at"],
        "at_utc": in_utc(rec["at"]),
        "capture_written": written,
        "capture_written_utc": in_utc(written) if written else None,
        "flags_sha256": rec.get("flags_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", type=Path, help="path to the capture folder (e.g. data/2026-10-24_lead_fp_4310)")
    p.add_argument("--load", default=None, help="load name(s) separated by '+', defaulting to the capture's load(s)")
    p.add_argument("--initial", default=None, metavar="WHO", help="record WHO as having read this printout into looks/<load>.shown.json and looks/<load>.flags_read.json")
    a = p.parse_args(argv)
    try:
        info = print_capture_report(a.capture, load_arg=a.load)
        if a.initial is not None:
            record_initial(a.capture, a.initial, info["loads"])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
