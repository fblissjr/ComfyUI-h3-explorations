#!/usr/bin/env python3
"""What the owner already said on a shot, from `owner_answers.jsonl` (plus `verdicts.jsonl` and optional inbox files), oldest first.

    <python> bench/said_on_shot.py --answers data/<stem>/owner_answers.jsonl --shot FIRST-LAST [--verdicts FILE] [--inbox FILE ...] [--for-question]

**What it buys.** Before a session builds a load on a shot or writes a question on the review page or Open List, it must
read every answer and note the owner has already given on those frames (`docs/wiki/working_with_the_owner.md`: never
re-ask what was already answered). `owner_answers.jsonl` holds who-is-who clicks, gap answers, clip verdicts, frame notes,
timeline notes, and relayed inbox answers under varying `row` names (`who`, `segments`, `gap_<first>_<sub...>`, `<load>`,
`shot_<first>`, `timeline`, `fun_0000_6394_joined_*`), alongside unpinned whole-cut remarks on `[0, 6394]` that drown a
plain grep, while `verdicts/verdicts.jsonl` also holds terminal verdicts recorded directly from the lead's terminal.
This tool prints every line on `--shot FIRST-LAST` (`collect_answers`, `collect_verdicts`, `collect_inbox`: rows that pin
a `frame` / `source_frame` inside `FIRST-LAST`, whose span lies inside the shot, whose `row == "timeline"` span overlaps
the shot, whose overlapping pre-split shot span starts on `FIRST`, or whose `row` / `item` names a frame inside
`FIRST-LAST` after stripping ISO timestamps via `STAMP_RE` / `row_named_frames`, while dropping rows with a pinned frame
outside `FIRST-LAST`, multi-shot `correction_` rows, and unpinned whole-cut remarks via `is_whole_cut_row`, and repairing
JS string-concatenated spans via `parse_row_span`), in chronological order (oldest first), with date, line number, row,
where (including `person`/`subject`, `piece`, and `tag` when present), what, and full words.
`--for-question` (`format_for_question`) formats the output as a ready-to-paste block under a question so no question is
ever put to the owner without what they already said on that shot.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Whole-cut or multi-scene spans wider than this (in source frames) are not single-shot answers.
#: Reasoned: the longest H3 window is 362 frames (`h3_rules.MAX_LENGTH`) and the longest single shot in `fun` is under 700 frames,
#: whereas whole-cut review items span `[0, 6394]` or multi-shot correction spans `[2050, 4389]`.
MAX_SHOT_SPAN_FRAMES = 1000

STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{4,6}Z)?")
ROW_FRAME_RE = re.compile(r"(?:^|_|\b)0*(\d{3,5})(?:[a-z]?(?:_|$|\b))")


def parse_shot_arg(spec: str) -> tuple[int, int]:
    s = str(spec or "").strip()
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", s)
    if not m:
        raise SystemExit(f"--shot {spec!r}: expected FIRST-LAST or FIRST (non-negative integers)")
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) is not None else first
    if first > last:
        raise SystemExit(f"--shot {spec!r}: first frame {first} is after last frame {last}")
    return first, last


def parse_row_span(row: dict) -> tuple[int, int] | None:
    fr = row.get("frames")
    if isinstance(fr, (list, tuple)) and len(fr) == 2:
        try:
            a, b = int(fr[0]), int(fr[1])
            # Early review page lines (2026-10-12) saved `["415", 41522]` from JS string concatenation `"415" + 22`
            if isinstance(fr[0], str) and str(b).startswith(fr[0].strip()) and b - a > MAX_SHOT_SPAN_FRAMES:
                tail = str(b)[len(fr[0].strip()):]
                if tail.isdigit():
                    b = a + int(tail)
            if a <= b:
                return a, b
        except (ValueError, TypeError):
            pass
    sh = row.get("shot")
    if isinstance(sh, str) and re.fullmatch(r"\d+-\d+", sh.strip()):
        a_s, b_s = sh.strip().split("-")
        a, b = int(a_s), int(b_s)
        if a <= b:
            return a, b
    elif isinstance(sh, (list, tuple)) and len(sh) == 2:
        try:
            a, b = int(sh[0]), int(sh[1])
            if a <= b:
                return a, b
        except (ValueError, TypeError):
            pass
    if row.get("first") is not None and row.get("frames") is not None and not isinstance(row.get("frames"), (list, tuple)):
        try:
            f0 = int(row["first"])
            cnt = int(row["frames"])
            if cnt > 0:
                return f0, f0 + cnt - 1
        except (ValueError, TypeError):
            pass
    return None


def pinned_frame(row: dict) -> int | None:
    for key in ("frame", "source_frame"):
        val = row.get(key)
        if val is not None and val != "" and not isinstance(val, bool):
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def row_named_frames(row_name: str) -> list[int]:
    cleaned = STAMP_RE.sub("", row_name)
    out: list[int] = []
    for m in ROW_FRAME_RE.finditer(cleaned):
        out.append(int(m.group(1)))
    return out


def is_whole_cut_row(row: dict, span: tuple[int, int] | None) -> bool:
    row_name = str(row.get("row") or row.get("item") or "").strip()
    if (
        re.match(r"^[a-z0-9]+_0000_\d{3,5}_joined", row_name)
        or row_name.startswith("correction_")
        or "_full_" in row_name
        or "_cut_" in row_name
        or row_name.endswith(("_full", "_cut"))
    ):
        return True
    for key in ("clip", "render"):
        v = str(row.get(key) or "").strip()
        if re.search(r"(?:^|/)[^/]+_0000_\d{3,5}_joined", v) or "_full_" in v or "_cut_" in v:
            return True
    if span is not None and (span[1] - span[0] > MAX_SHOT_SPAN_FRAMES):
        return True
    return False


def said_text_and_what(row: dict) -> tuple[str, str] | None:
    what = str(row.get("what") or "").strip()
    chosen = " ".join(str(row.get("chosen") or "").split()).strip()
    words = " ".join(str(row.get("words") or "").split()).strip()
    if not what:
        if words and chosen:
            what = "choice+words"
        elif words:
            what = "words"
        elif chosen:
            what = "choice"
        elif row.get("parts") is not None:
            what = "parts"
    if chosen and words:
        return what or "choice+words", f"{chosen} — {words}"
    if words:
        return what or "words", words
    if chosen:
        return what or "choice", chosen
    if row.get("parts") is not None:
        return what or "parts", json.dumps(row["parts"], ensure_ascii=False)
    return None


def matches_shot(row: dict, first: int, last: int) -> bool:
    if row.get("who") not in ("owner", "not said", None, ""):
        return False
    if said_text_and_what(row) is None:
        return False
    row_name = str(row.get("row") or row.get("item") or "").strip()
    if row_name.startswith("correction_"):
        return False
    span = parse_row_span(row)
    pin = pinned_frame(row)
    if pin is not None:
        return first <= pin <= last
    if is_whole_cut_row(row, span):
        return False
    # 1. Span lies inside [first, last]
    if span is not None and first <= span[0] and span[1] <= last:
        return True
    # 2. Timeline span note overlapping [first, last]
    if row_name == "timeline" and span is not None and not (span[1] < first or span[0] > last):
        return True
    # 3. Pre-split or overlapping shot span that starts at `first`
    if span is not None and not (span[1] < first or span[0] > last) and span[0] == first:
        return True
    # 4. `row` names a load, gap, or shot on `FIRST-LAST` (provided its span, if any, is not disjoint from `FIRST-LAST`)
    if span is None or not (span[1] < first or span[0] > last):
        if any(first <= f <= last for f in row_named_frames(row_name)):
            return True
    return False


def _row_to_entry(row: dict, source_name: str, line_no: int) -> dict:
    tw = said_text_and_what(row)
    assert tw is not None
    what, text = tw
    span = parse_row_span(row)
    pin = pinned_frame(row)
    row_name = str(row.get("row") or row.get("item") or "").strip()
    at = str(row.get("at") or "").strip()
    entry: dict = {
        "source": source_name,
        "line": line_no,
        "at": at,
        "date": at[:16].replace("T", " ") if at else "date unknown",
        "row": row_name,
        "span": list(span) if span is not None else None,
        "frame": pin,
        "what": what,
        "said": text,
    }
    person = str(row.get("person") or row.get("subject") or "").strip()
    if person:
        entry["person"] = person
    piece = str(row.get("piece") or "").strip()
    if piece:
        entry["piece"] = piece
    tag = str(row.get("tag") or "").strip()
    if tag:
        entry["tag"] = tag
    return entry


def collect_verdicts(verdicts_path: Path, first: int, last: int) -> list[dict]:
    if not verdicts_path.is_file():
        raise SystemExit(f"--verdicts {verdicts_path}: file not found")
    matched: list[dict] = []
    for line_no, raw in enumerate(verdicts_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as e:
            raise SystemExit(f"--verdicts {verdicts_path} line {line_no}: invalid JSON ({e})") from None
        if not isinstance(row, dict):
            raise SystemExit(f"--verdicts {verdicts_path} line {line_no}: expected a JSON object")
        src = str(row.get("source") or "").strip()
        if src.startswith(("owner_answers.jsonl", "answers.jsonl", "comments.jsonl")):
            continue
        if not matches_shot(row, first, last):
            continue
        matched.append(_row_to_entry(row, verdicts_path.name, line_no))
    return matched


def collect_answers(answers_path: Path, first: int, last: int, verdicts_path: Path | None = None) -> list[dict]:
    if not answers_path.is_file():
        raise SystemExit(f"--answers {answers_path}: file not found")
    matched: list[dict] = []
    for line_no, raw in enumerate(answers_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as e:
            raise SystemExit(f"--answers {answers_path} line {line_no}: invalid JSON ({e})") from None
        if not isinstance(row, dict):
            raise SystemExit(f"--answers {answers_path} line {line_no}: expected a JSON object")
        if not matches_shot(row, first, last):
            continue
        matched.append(_row_to_entry(row, answers_path.name, line_no))
    v_path = verdicts_path
    if v_path is None:
        cand = answers_path.parent / "verdicts" / "verdicts.jsonl"
        if cand.is_file():
            v_path = cand
    if v_path is not None:
        seen_row_said = {(e["row"], e["said"]) for e in matched}
        for ve in collect_verdicts(v_path, first, last):
            if (ve["row"], ve["said"]) not in seen_row_said:
                seen_row_said.add((ve["row"], ve["said"]))
                matched.append(ve)
    matched.sort(key=lambda r: (r["at"], r["line"]))
    return matched


def collect_inbox(inbox_paths: list[Path], first: int, last: int) -> list[dict]:
    out: list[dict] = []
    for p in inbox_paths:
        if not p.is_file():
            raise SystemExit(f"--inbox {p}: file not found")
        raw_text = p.read_text(encoding="utf-8")
        # JSONL or JSON list support, plus line-by-line markdown/text fallback
        lines = raw_text.splitlines()
        for line_no, ln in enumerate(lines, 1):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                except ValueError:
                    obj = None
                if isinstance(obj, dict):
                    if matches_shot(obj, first, last):
                        tw = said_text_and_what(obj)
                        if tw:
                            at = str(obj.get("at") or "").strip()
                            out.append({
                                "source": p.name,
                                "line": line_no,
                                "at": at,
                                "date": at[:16].replace("T", " ") if at else "inbox",
                                "row": str(obj.get("row") or obj.get("id") or "inbox"),
                                "span": list(parse_row_span(obj)) if parse_row_span(obj) else None,
                                "frame": pinned_frame(obj),
                                "what": tw[0],
                                "said": tw[1],
                            })
                    continue
            nums = [int(m.group(1)) for m in re.finditer(r"(?<!\d)0*(\d{3,5})(?!\d)", STAMP_RE.sub("", s))]
            if any(first <= n <= last for n in nums):
                out.append({
                    "source": p.name,
                    "line": line_no,
                    "at": "",
                    "date": "inbox",
                    "row": p.name,
                    "span": None,
                    "frame": None,
                    "what": "inbox",
                    "said": " ".join(s.split()),
                })
    return out


def where_label(entry: dict) -> str:
    sp = entry.get("span")
    pin = entry.get("frame")
    parts: list[str] = []
    if isinstance(sp, list) and len(sp) == 2:
        parts.append(f"frames {sp[0]}-{sp[1]}" if sp[0] != sp[1] else f"frame {sp[0]}")
    if pin is not None:
        pin_s = f"frame {pin}"
        if pin_s not in parts:
            parts.append(pin_s)
    if entry.get("person"):
        parts.append(str(entry["person"]))
    if entry.get("piece"):
        parts.append(f"piece {entry['piece']}")
    if entry.get("tag"):
        parts.append(f"tag {entry['tag']}")
    return ", ".join(parts) if parts else "shot"


def format_plain(entries: list[dict], first: int, last: int, answers_name: str) -> str:
    span_s = f"{first}-{last}" if first != last else str(first)
    if not entries:
        return f"said_on_shot {span_s} ({answers_name}): 0 lines\n"
    out = [f"said_on_shot {span_s} ({answers_name}): {len(entries)} line(s), oldest first"]
    for e in entries:
        loc = f"{e['source']}:{e['line']}" if e.get("source") != answers_name else f"line {e['line']}"
        out.append(f"  {e['date']}  {loc:<9}  {e['row']}  ({where_label(e)})  {e['what']}: {e['said']}")
    return "\n".join(out) + "\n"


def format_for_question(entries: list[dict], first: int, last: int, answers_name: str) -> str:
    span_s = f"{first}-{last}" if first != last else str(first)
    if not entries:
        return f"What the owner already said on {span_s} (from {answers_name}): nothing recorded.\n"
    out = [f"What the owner already said on {span_s} (from {answers_name}, {len(entries)} line(s), oldest first):"]
    for e in entries:
        loc = f"{e['source']}:{e['line']}" if e.get("source") != answers_name else f"line {e['line']}"
        out.append(f"- {loc} ({e['date']}, `{e['row']}`, {where_label(e)}, {e['what']}): {e['said']}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--answers", required=True, help="path to data/<stem>/owner_answers.jsonl")
    ap.add_argument("--shot", required=True, help="shot span FIRST-LAST (or FIRST)")
    ap.add_argument("--verdicts", help="optional path to data/<stem>/verdicts/verdicts.jsonl")
    ap.add_argument("--inbox", action="append", default=[], help="optional inbox file(s) to check alongside owner_answers.jsonl")
    ap.add_argument("--for-question", action="store_true", help="format as a ready-to-paste block under a question")
    a = ap.parse_args(argv)
    first, last = parse_shot_arg(a.shot)
    ans_path = Path(a.answers)
    verdicts_path = Path(a.verdicts) if a.verdicts else None
    entries = collect_answers(ans_path, first, last, verdicts_path=verdicts_path)
    if a.inbox:
        entries.extend(collect_inbox([Path(p) for p in a.inbox], first, last))
    text = (
        format_for_question(entries, first, last, ans_path.name)
        if a.for_question
        else format_plain(entries, first, last, ans_path.name)
    )
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
