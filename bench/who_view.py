#!/usr/bin/env python3
"""Read-only structured view of who is in each shot, generated from `shots.json`, `cast.jsonl`, and `owner_answers.jsonl`.

    <python> bench/who_view.py --shots data/<stem>/shots.json --cast data/<stem>/cast/cast.jsonl \\
        --answers data/<stem>/owner_answers.jsonl [--plan data/<stem>/plan.json] [--manifest data/<stem>/manifest.json] \\
        --json data/<stem>/who_v2.json [--md data/<stem>/who_v2.md]

**What it buys.** `data/<stem>/shots.json` (`h3_shots/1`), `data/<stem>/cast/cast.jsonl` (`h3_cast/1`), and
`data/<stem>/owner_answers.jsonl` are the one home for each kind of fact. Fifteen tools read `shots.json` as
`h3_shots/1`. This tool generates the structured, read-only view (`h3_who_view/1`, JSON and Markdown) that joins
all three files with `plan.json` and `manifest.json`, carrying their `sha256` digests in `made_from`, without
mutating any input file or duplicating state across editable files.

**What it fixes over the initial `who_v2.py` survey (`G20` / `C53`).**
1. Preserves every shot-level field (`order`, `order_by`, `scene_score`, `others_present`, and `history`:
   `merged_from`, `split_from`, `moved_from`, `answered_over`).
2. Preserves `box`, `size`, `table`, and `standing: bool` (from `cast_file.current()`) on every cast entry
   (`cast_entries` and `boxes`), plus `standing_boxes` for standing `"here"` boxes.
3. Expands `line(s) A-B` ranges in `line_numbers()` and stops before ISO timestamps (`YYYY-MM-DD`), so date
   components (`10`, `12`, `2026`) never pollute `answer_lines`.
4. Matches `owner_answers` across `WHO_ROWS = ("b20", "who")` by the frame shown (`source_frame` inside
   `[first, last]`, and `word_target` for split-shot notes), never by stale pre-split `frames` spans, and
   includes early `person`/`chosen` (`"the lead"`, etc.) lines as well as `table` and `person`.
5. Resolves `cut_from` by full containment (`a <= first and last <= b`) before partial overlap, and lists all
   contributing ranges in `cut_from_all` (so merged shot `1898-1944` carries `["1039-1919", "1920-6394"]`).
6. Parses `"the owner"` and `"entered by shots_record.py answer for <session>"` in `attribution()`.
7. Computes `seen_standing` per part (where a later `"is": "seen"` look takes back an earlier `"not seen"` look on
   its frames, matching `bench/shot_coverage.py::nothing_to_make`), flags `owner_hedged` on shots where the owner's
   typed words express uncertainty (`maybe`, `i think`, `cannot tell`, `not sure`), and separates `copy_fps` (`24`)
   from `source_fps` (`manifest.json`).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cast_file  # noqa: E402
import page_answers_to_cast as patc  # noqa: E402
import shot_coverage  # noqa: E402

SCHEMA = "h3_who_view/1"
COPY_FPS = 24
WHO_ROWS = ("b20", "who")
EARLY_CHOSEN_SUBJECT = {
    "the lead": "lead",
    "the second subject": "second",
    "the third subject": "third",
}
ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z")
DAY = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SPAN = re.compile(r"^(\d+)-(\d+): ")
SESSION = re.compile(r"\bmr[a-z0-9]+(?:-[a-z0-9]+)?\b")
VIA = (
    ("who page", "who page"),
    ("tile answer", "tile page"),
    ("terminal", "terminal"),
    ("screenshot", "screenshot"),
    ("delivery table", "delivery table"),
)
CLAUSE_RE = re.compile(
    r"\b(?:(?:owner_)?answers\.jsonl\s+)?lines?\s+"
    r"(\d+(?:\s*-\s*\d+)?(?:\s*(?:,|and)\s*(?!\d{4}-\d{2}-\d{2})\d+(?:\s*-\s*\d+)?)*)\b"
    r"|(?:owner_)?answers\.jsonl:(\d+)"
)
TOKEN_RE = re.compile(r"(\d+)(?:\s*-\s*(\d+))?")
HEDGE_RE = re.compile(r"\b(?:maybe|i\s+think|can'?t\s+tell|cannot\s+tell|not\s+sure|hard\s+to\s+tell)\b", re.IGNORECASE)
HISTORY_KEYS = ("merged_from", "split_from", "moved_from", "answered_over")


class Refused(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_numbers(text: str | None) -> list[int]:
    """Every answers.jsonl line number a string cites, with `A-B` ranges expanded and ISO dates excluded."""
    out: set[int] = set()
    for m in CLAUSE_RE.finditer(text or ""):
        if m.group(2):
            out.add(int(m.group(2)))
        else:
            for tm in TOKEN_RE.finditer(m.group(1)):
                a = int(tm.group(1))
                b = int(tm.group(2)) if tm.group(2) else a
                if a <= b:
                    out.update(range(a, b + 1))
    return sorted(out)


def attribution(raw: str | None) -> list[dict]:
    """Split `by` into `; `-separated clause dicts with `said_by`, `at`, `via`, `answer_lines`, `entered_by`, `relayed_by`."""
    if not raw:
        return []
    parts = []
    for clause in [c.strip() for c in str(raw).split("; ") if c.strip()]:
        e: dict = {"raw": clause}
        m = SPAN.match(clause)
        if m:
            e["frames"] = [int(m.group(1)), int(m.group(2))]
            clause = clause[m.end():]
        low = clause.lower()
        m_ent = re.search(r"entered by ([\w.]+(?:\s+answer)?)(?: for ([\w-]+))?", clause)
        for_sess = m_ent.group(2) if m_ent else None
        if low.startswith("owner") or low.startswith("the owner") or "owner's" in low or "the owner" in low:
            e["said_by"] = "owner"
        elif for_sess:
            e["said_by"] = for_sess
        else:
            e["said_by"] = (SESSION.findall(clause) or [None])[0]
        t = ISO.findall(clause) or DAY.findall(clause)
        if t:
            e["at"] = t[0]
        for word, via in VIA:
            if word in low:
                e["via"] = via
                break
        nums = line_numbers(clause)
        if nums:
            e["answer_lines"] = nums
        if m_ent:
            e["entered_by"] = m_ent.group(1) + (f" for {m_ent.group(2)}" if m_ent.group(2) else "")
        m_rel = re.search(r"relayed by ([\w-]+)", clause)
        if m_rel:
            e["relayed_by"] = m_rel.group(1)
        others = [s for s in SESSION.findall(clause) if s not in (e.get("said_by"), for_sess, e.get("relayed_by"))]
        if others:
            e["sessions_named"] = others
        parts.append(e)
    return parts


def cut_from_for(cuts_from: dict | None, first: int, last: int) -> tuple[str | None, list[str]]:
    """Return `(primary_cut_from, all_overlapping_cuts_from)` for a shot `first..last`."""
    if not isinstance(cuts_from, dict):
        return None, []
    parsed: list[tuple[str, int, int]] = []
    for k in cuts_from:
        if "-" in k:
            a_str, _, b_str = k.partition("-")
            if a_str.isdigit() and b_str.isdigit():
                parsed.append((k, int(a_str), int(b_str)))
    containing = [k for k, a, b in parsed if a <= first and last <= b]
    overlapping = [k for k, a, b in parsed if a <= last and b >= first]
    primary = containing[0] if containing else (overlapping[0] if overlapping else None)
    return primary, overlapping


def frames_to_spans(frames: set[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and out[-1][1] == f - 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def seen_standing_for(who_entry: dict, first: int, last: int, plan_subject: dict | None = None) -> dict:
    """Compute effective standing `not seen` spans per part after applying `seen_looks` in order (`seen` overrides earlier `not seen`)."""
    looks = who_entry.get("seen_looks") or []
    parts_seen: list[str] = []
    for lk in looks:
        if isinstance(lk, dict) and lk.get("part") and lk["part"] not in parts_seen:
            parts_seen.append(str(lk["part"]))
    for p in (plan_subject or {}).get("parts") or []:
        if p not in parts_seen:
            parts_seen.append(str(p))
    not_seen_by_part: dict[str, list[list[int]]] = {}
    overruled: list[dict] = []
    for part in parts_seen:
        not_seen: set[int] = set()
        had_not_seen: set[int] = set()
        for lk in looks:
            if not isinstance(lk, dict) or lk.get("part") != part:
                continue
            a = max(first, int(lk.get("first", first)))
            b = min(last, int(lk.get("last", last)))
            span_frames = set(range(a, b + 1))
            if lk.get("is") == "not seen":
                not_seen |= span_frames
                had_not_seen |= span_frames
            elif lk.get("is") == "seen":
                cleared = not_seen & span_frames
                not_seen -= span_frames
                if cleared:
                    overruled.append({
                        "part": part,
                        "frames": frames_to_spans(cleared),
                        "by": lk.get("by"),
                        "at": lk.get("at"),
                        "why": lk.get("why"),
                    })
        if not_seen:
            not_seen_by_part[part] = frames_to_spans(not_seen)
        elif had_not_seen:
            not_seen_by_part[part] = []
    nothing = shot_coverage.nothing_to_make(who_entry, plan_subject, first, last) if plan_subject else []
    return {
        "not_seen_spans": not_seen_by_part,
        "overruled_looks": overruled,
        "nothing_to_make": nothing,
    }


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise Refused(f"no {what} at {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise Refused(f"{path} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise Refused(f"{path}: expected a JSON object")
    return data


def _read_jsonl(path: Path, what: str) -> list[tuple[int, dict]]:
    if not path.is_file():
        raise Refused(f"no {what} at {path}")
    out: list[tuple[int, dict]] = []
    for idx, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception as e:
            raise Refused(f"{path} line {idx} is not valid JSON: {e}") from None
        if not isinstance(row, dict):
            raise Refused(f"{path} line {idx}: expected a JSON object")
        out.append((idx, row))
    return out


def owner_subject_for(row: dict) -> str | None:
    """Subject named by an owner answer line (`subject` on current rows, or `EARLY_CHOSEN_SUBJECT` on step-0 / early rows)."""
    if row.get("subject"):
        return str(row["subject"])
    chosen = str(row.get("chosen") or "").strip().lower()
    return EARLY_CHOSEN_SUBJECT.get(chosen)


def build_view(
    shots_path: str | Path,
    cast_path: str | Path | None = None,
    answers_path: str | Path | None = None,
    plan_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """Build the read-only structured view dict (`h3_who_view/1`) from the authoritative files."""
    shots_p = Path(shots_path)
    old = _read_json(shots_p, "shots record")
    shot_coverage.load_shots(str(shots_p))

    cast_p = Path(cast_path) if cast_path else None
    ans_p = Path(answers_path) if answers_path else None
    plan_p = Path(plan_path) if plan_path else None
    man_p = Path(manifest_path) if manifest_path else (shots_p.parent / "manifest.json" if (shots_p.parent / "manifest.json").is_file() else None)

    cast_rows = _read_jsonl(cast_p, "cast file") if cast_p else []
    if cast_p:
        cast_file.read(cast_p)
    ans_rows = _read_jsonl(ans_p, "owner answers file") if ans_p else []
    plan = _read_json(plan_p, "plan file") if plan_p else {}
    manifest = _read_json(man_p, "manifest file") if man_p else {}

    made_from: dict[str, dict[str, str]] = {
        "shots": {"path": str(shots_p), "sha256": sha256_file(shots_p)},
    }
    if cast_p:
        made_from["cast"] = {"path": str(cast_p), "sha256": sha256_file(cast_p)}
    if ans_p:
        made_from["owner_answers"] = {"path": str(ans_p), "sha256": sha256_file(ans_p)}
    if plan_p:
        made_from["plan"] = {"path": str(plan_p), "sha256": sha256_file(plan_p)}
    if man_p:
        made_from["manifest"] = {"path": str(man_p), "sha256": sha256_file(man_p)}

    # Determine which line index in cast.jsonl currently stands per (subject, frame)
    standing_line_by_key: dict[tuple[str, int], int] = {}
    for idx, c in cast_rows:
        key = (str(c.get("subject")), int(c.get("frame", -1)))
        if c.get("answer") == "withdrawn":
            standing_line_by_key.pop(key, None)
        else:
            standing_line_by_key[key] = idx

    # Pre-route owner word lines to their target shot and effective frame (handling split shots)
    words_by_shot_first: dict[int, list[dict]] = {int(z["first"]): [] for z in old["shots"]}
    choice_rows: list[tuple[int, int, str | None, dict]] = []
    for idx, r in ans_rows:
        if r.get("who") != "owner" or r.get("row") not in WHO_ROWS:
            continue
        if r.get("video") not in (None, old.get("video"), Path(str(old.get("video", ""))).stem):
            continue
        if not str(r.get("source_frame", "")).isdigit():
            continue
        if r.get("what") == "words" and str(r.get("words", "")).strip():
            eff_frame, target_shot = patc.word_target(old["shots"], r)
            if target_shot is not None:
                words_by_shot_first[int(target_shot["first"])].append({
                    "line": idx,
                    "at": r.get("at"),
                    "row": r.get("row"),
                    "what": "words",
                    "frame": eff_frame,
                    "recorded_source_frame": int(r["source_frame"]),
                    "words": str(r["words"]).strip(),
                })
        elif r.get("what") == "choice":
            choice_rows.append((idx, int(r["source_frame"]), owner_subject_for(r), r))

    def view_spans(span_list, first: int, last: int) -> list[dict]:
        return [x for x in (span_list or []) if int(x.get("first", -1)) <= last and int(x.get("last", -1)) >= first]

    shots_out: list[dict] = []
    unparsed_attributions: list[str] = []

    for shot_idx, z in enumerate(old["shots"]):
        f, l = int(z["first"]), int(z["last"])
        shot_words = words_by_shot_first.get(f, [])
        note_raw = z.get("note")
        note_lines = line_numbers(note_raw)
        hedged = bool(
            HEDGE_RE.search(str(note_raw or ""))
            or any(HEDGE_RE.search(w["words"]) for w in shot_words)
        )
        new_subjects: dict[str, dict] = {}
        for s in old["subjects"]:
            v = z["who"][s]
            att = attribution(v.get("by"))
            if not att or not any(p.get("said_by") for p in att):
                unparsed_attributions.append(f"{f}-{l} {s}: by {v.get('by')!r}")
            mine = [
                {
                    "line": idx,
                    "at": r.get("at"),
                    "row": r.get("row"),
                    "what": r.get("what"),
                    "table": r.get("table"),
                    "chosen": r.get("chosen"),
                    "person": r.get("person"),
                    "frame": sf,
                    "words": r.get("words"),
                }
                for idx, sf, sub, r in choice_rows
                if f <= sf <= l and (sub == s or (sub is None and r.get("subject") is None))
            ]
            cited = set(line_numbers(v.get("how", ""))) | {n for p in att for n in p.get("answer_lines", [])} | set(note_lines)
            cast_entries = []
            for idx, c in cast_rows:
                if c.get("subject") == s and isinstance(c.get("frame"), int) and f <= c["frame"] <= l:
                    is_standing = standing_line_by_key.get((s, c["frame"])) == idx
                    cast_entries.append({
                        "line": idx,
                        "frame": c["frame"],
                        "answer": c["answer"],
                        "box": c.get("box"),
                        "size": c.get("size"),
                        "table": c.get("table"),
                        "by": c.get("by"),
                        "at": c.get("at"),
                        "standing": is_standing,
                    })
                    cited.update(line_numbers(c.get("by", "")))
            standing_boxes = [c for c in cast_entries if c["standing"] and c["answer"] == "here" and c["box"] is not None]
            plan_sub = (plan.get("subjects") or {}).get(s) or {}
            seen_st = seen_standing_for(v, f, l, plan_sub if plan_sub else None)
            new_subjects[s] = {
                "present": v.get("present"),
                "how": v.get("how"),
                "answer": att,
                "owner_answers": mine,
                "owner_answers_cited": sorted(cited),
                "owner_answers_not_cited": sorted({m["line"] for m in mine} - cited),
                "cast_entries": cast_entries,
                "standing_boxes": standing_boxes,
                "boxes": cast_entries,
                "evidence": {k: v[k] for k in ("frames_boxed", "frames_on", "frames_from") if k in v},
                "seen": v.get("seen_looks", []),
                "seen_standing": seen_st,
                "from_plan": {
                    "voice": view_spans(plan_sub.get("voice"), f, l),
                    "clothing": view_spans(plan_sub.get("clothing"), f, l),
                    "departures": [d for d in (plan.get("departures") or []) if d.get("subject") == s and int(d["first"]) <= l and int(d["last"]) >= f],
                },
                "raw": v,
            }

        cut_primary, cut_all = cut_from_for(old.get("cuts_from"), f, l)
        history = {k: z[k] for k in HISTORY_KEYS if k in z}
        shot_entry: dict = {
            "index": shot_idx,
            "first": f,
            "last": l,
            "frames": l - f + 1,
            "start_s": round(f / COPY_FPS, 3),
            "end_s": round((l + 1) / COPY_FPS, 3),
            "cut_from": cut_primary,
            "cut_from_all": cut_all,
            "scene_score": z.get("scene_score"),
            "order": z.get("order"),
            "order_by": z.get("order_by"),
            "others_present": z.get("others_present"),
            "history": history,
            "note": {
                "raw": note_raw,
                "answer_lines": note_lines,
                "owner_words": shot_words,
                "owner_hedged": hedged,
            },
            "not_people": [n["what"] for n in (plan.get("not_people") or []) if any(int(a_) <= l and int(b_) >= f for a_, b_ in n.get("shots", []))],
            "subjects": new_subjects,
        }
        shots_out.append(shot_entry)

    source_frames = int(old.get("source_frames") or (int(old["span"][1]) - int(old["span"][0]) + 1))
    raw_src_fps = (manifest.get("source") or {}).get("rate") or manifest.get("fps")
    src_fps = float(raw_src_fps) if raw_src_fps else None
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"

    return {
        "schema": SCHEMA,
        "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "made_by": "bench/who_view.py",
        "tool_commit": commit,
        "video": {
            "file": old.get("video"),
            "copy": old.get("copy"),
            "frames": source_frames,
            "span": old.get("span"),
            "copy_fps": COPY_FPS,
            "copy_seconds": round(source_frames / COPY_FPS, 3),
            "source_fps": src_fps,
            "source_seconds": round(source_frames / src_fps, 3) if src_fps else None,
        },
        "subjects": [
            {
                "label": s,
                "aliases": [k for k, v in (plan.get("aliases") or {}).items() if v == s],
                "region": ((plan.get("subjects") or {}).get(s) or {}).get("region"),
            }
            for s in old["subjects"]
        ],
        "made_from": made_from,
        "record": {
            "written_by": old.get("written_by"),
            "cuts_from": old.get("cuts_from"),
            "not_cuts": old.get("not_cuts"),
        },
        "summary": {
            "shots": len(shots_out),
            "shot_subject_entries": sum(len(z["subjects"]) for z in shots_out),
            "unparsed_attributions": unparsed_attributions,
            "hedged_shots": [f"{z['first']}-{z['last']}" for z in shots_out if z["note"]["owner_hedged"]],
        },
        "shots": shots_out,
    }


def verify_view_fresh(
    view: dict,
    shots_path: str | Path | None = None,
    cast_path: str | Path | None = None,
    answers_path: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> None:
    """Refuse a pre-built view whose `made_from` sha256 does not match the files handed in."""
    if view.get("schema") != SCHEMA:
        raise Refused(f"view schema is {view.get('schema')!r}, expected {SCHEMA!r}")
    mf = view.get("made_from") or {}
    for key, p in (("shots", shots_path), ("cast", cast_path), ("owner_answers", answers_path), ("plan", plan_path)):
        if p is None:
            continue
        path = Path(p)
        if not path.is_file():
            raise Refused(f"no {key} file at {path}")
        actual = sha256_file(path)
        recorded = (mf.get(key) or {}).get("sha256")
        if recorded != actual:
            raise Refused(f"view is stale for {key} ({path}): view has sha256 {recorded}, file on disk is {actual}")


def as_md(view: dict) -> str:
    """Render a human-readable Markdown table and summary from `view` (`h3_who_view/1`)."""
    v = view["video"]
    subs = [s["label"] for s in view["subjects"]]
    lines = [
        f"# Who is in each shot: `{v['file']}` (`{v['span'][0]}-{v['span'][1]}`)",
        "",
        f"Generated `{view['made_at']}` by `{view['made_by']}` at `{view['tool_commit']}` (`{view['schema']}`). Regenerated, never edited by hand.",
        "",
        "## Made from",
        "",
    ]
    for k, info in view["made_from"].items():
        lines.append(f"- **`{k}`**: `{info['path']}` (`sha256`: `{info['sha256']}`)")
    lines += [
        "",
        f"- **Working copy**: `{v.get('copy')}` (`{v['frames']}` frames at `{v['copy_fps']}` fps, `{v['copy_seconds']}s`"
        + (f"; source `{v['file']}` at `{v['source_fps']}` fps, `{v['source_seconds']}s`" if v.get("source_fps") else "")
        + ")",
        f"- **Shots**: `{view['summary']['shots']}` (`{view['summary']['shot_subject_entries']}` shot-subject entries)",
        f"- **Hedged notes**: `{len(view['summary']['hedged_shots'])}` ({', '.join(view['summary']['hedged_shots']) or 'none'})",
        "",
        "## Shots",
        "",
        "| Shot | Frames | Cut from | " + " | ".join(subs) + " | Standing boxes | History / Order | Owner words / Note |",
        "|---|---:|---|" + "|".join(["---"] * len(subs)) + "|---|---|---|",
    ]
    for z in view["shots"]:
        span = f"`{z['first']}-{z['last']}`"
        cut_str = ", ".join(z["cut_from_all"]) if z.get("cut_from_all") else (z.get("cut_from") or "")
        sub_cells = []
        boxes_summary = []
        for s in subs:
            sv = z["subjects"][s]
            cell = f"**{sv['present']}**"
            if sv["seen_standing"]["not_seen_spans"]:
                ns = ", ".join(f"{p}:{','.join(f'{a}-{b}' for a, b in sp)}" for p, sp in sv["seen_standing"]["not_seen_spans"].items() if sp)
                if ns:
                    cell += f" (not seen: `{ns}`)"
            if sv["seen_standing"]["overruled_looks"]:
                cell += " (overruled look: seen)"
            sub_cells.append(cell)
            for b in sv["standing_boxes"]:
                boxes_summary.append(f"{s}@{b['frame']}")
        hist_parts = list(z.get("history") or {})
        if z.get("order"):
            hist_parts.append(f"order:{','.join(z['order'])}")
        words_parts = [f"L{w['line']}@{w['frame']}: {w['words']}" for w in z["note"]["owner_words"]]
        if z["note"]["owner_hedged"]:
            words_parts.append("[HEDGED]")
        note_cell = "; ".join(words_parts).replace("|", "\\|")
        lines.append(
            f"| {span} | {z['frames']} | `{cut_str}` | "
            + " | ".join(sub_cells)
            + f" | {', '.join(boxes_summary) or 'none'} | {', '.join(hist_parts) or 'none'} | {note_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--shots", required=True, help="the video's shots record (data/<stem>/shots.json, schema h3_shots/1)")
    ap.add_argument("--cast", required=True, help="the video's cast file (data/<stem>/cast/cast.jsonl, schema h3_cast/1)")
    ap.add_argument("--answers", required=True, help="the owner's raw answers file (data/<stem>/owner_answers.jsonl)")
    ap.add_argument("--plan", help="the video's plan file (data/<stem>/plan.json)")
    ap.add_argument("--manifest", help="the video's manifest file (data/<stem>/manifest.json)")
    ap.add_argument("--json", "--out", dest="json_out", required=True, help="where to write the structured JSON view (e.g. data/<stem>/who_v2.json)")
    ap.add_argument("--md", dest="md_out", help="where to write the Markdown view (defaults to --json with .md suffix unless --no-md)")
    ap.add_argument("--no-md", action="store_true", help="write only the JSON view")
    a = ap.parse_args(argv)
    try:
        view = build_view(a.shots, a.cast, a.answers, a.plan, a.manifest)
    except (Refused, SystemExit) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    json_p = Path(a.json_out)
    json_p.parent.mkdir(parents=True, exist_ok=True)
    json_p.write_text(json.dumps(view, indent=1, ensure_ascii=False) + "\n")
    md_p = None if a.no_md else (Path(a.md_out) if a.md_out else json_p.with_suffix(".md"))
    if md_p is not None:
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md_p.write_text(as_md(view))
    print(
        f"written: {json_p}"
        + (f" and {md_p}" if md_p else "")
        + f"; {view['summary']['shots']} shots, {view['summary']['shot_subject_entries']} shot-subject entries; "
        f"attributions with no speaker: {len(view['summary']['unparsed_attributions'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
