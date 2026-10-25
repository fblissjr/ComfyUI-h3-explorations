#!/usr/bin/env python3
"""Check `bench/said_on_shot.py` against synthetic and real `owner_answers.jsonl` rows."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import said_on_shot as sos  # noqa: E402
from _lib import case, finish  # noqa: E402


def span_and_load_matching() -> str:
    """Span inside the shot, pre-split overlapping shot span, and load name on the shot match; whole-cut and timestamp-in-row-name rows do not."""
    with tempfile.TemporaryDirectory() as tmp:
        ans = Path(tmp) / "owner_answers.jsonl"
        rows = [
            # 1. Exact span inside shot
            {"at": "2026-10-12T20:50:14Z", "page": "who", "shot": "4310-4398", "frames": [4310, 4398], "frame": 4310, "what": "words", "words": "lead = red dress"},
            # 2. Pre-split span whose frame is outside the narrowed sub-shot (516-589 vs frame 597)
            {"at": "2026-10-24T16:00:00Z", "row": "lead_fp_0516c", "frames": [516, 603], "frame": 597, "what": "words", "words": "frame 597 held"},
            # 3. Pre-split span starting on 516 with frame inside 516-589
            {"at": "2026-10-11T17:53:48Z", "page": "segments", "shot": "516-603", "frames": [516, 603], "frame": 516, "what": "words", "words": "just change the face"},
            # 4. Unpinned whole-cut joined clip (must be dropped)
            {"at": "2026-10-15T19:01:41Z", "row": "fun_0000_6394_joined_2026-10-14T0033Z_wholeshot", "frames": [0, 6394], "what": "words", "words": "whole cut remark"},
            # 5. Multi-shot correction row (must be dropped even with a frame field)
            {"at": "2026-10-24T17:20:00Z", "row": "correction_rows_1602_1603", "frames": [2050, 4389], "frame": 2050, "what": "note", "words": "multi shot correction"},
            # 6. Timestamp T2140Z in row name of another shot (must NOT match 2050-2247)
            {"at": "2026-10-15T22:55:24Z", "row": "fun_1419_1465_joined_2026-10-15T2140Z_wholeshot", "frames": [1419, 1465], "what": "choice", "chosen": "keep this one"},
            # 7. Load name with no frames field matching 4310-4389
            {"at": "2026-10-24T17:07:00Z", "row": "lead_fp_4310", "what": "choice", "chosen": "wrong, and I say where"},
            # 8. Oct-12 string-concatenated frame array bug (["415", 415100] -> (415, 515))
            {"at": "2026-10-12T20:43:20Z", "page": "who", "shot": 415, "frames": ["415", 415100], "frame": 437, "what": "choice", "chosen": "not outlined"},
            # 9. Frame-pinned whole-cut note (matches 604-1009, not 4310-4389)
            {"at": "2026-10-16T17:13:49Z", "row": "fun_0000_6394_joined_2026-10-16T1252Z_from_laid", "frames": [0, 6394], "frame": 681, "source_frame": 681, "what": "words", "words": "fix: the lead is gone from 681"},
            # 10. Timeline note with person, piece, and tag
            {"at": "2026-10-25T18:00:00Z", "row": "timeline", "frames": [2623, 2624], "frame": 2623, "source_frame": 2623, "shot": "2623-2711", "person": "lead", "piece": "lead_fp_2625", "tag": "gap_flash", "what": "timeline_note", "words": "2-frame original singer flash"},
            # 11. Unpinned whole-cut _full_ / _cut_ row with span <= 1000 starting at 0 (must NOT match shot 0-99)
            {"at": "2026-10-25T18:05:00Z", "row": "clip_full_cut_v1", "frames": [0, 500], "what": "words", "words": "short whole cut remark"},
        ]
        ans.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        got_0 = sos.collect_answers(ans, 0, 99)
        assert got_0 == [], got_0

        got_4310 = sos.collect_answers(ans, 4310, 4389)
        assert [r["line"] for r in got_4310] == [1, 7], got_4310

        got_589 = sos.collect_answers(ans, 516, 589)
        assert [r["line"] for r in got_589] == [3], got_589

        got_603 = sos.collect_answers(ans, 516, 603)
        assert [r["line"] for r in got_603] == [3, 2], got_603

        got_2050 = sos.collect_answers(ans, 2050, 2247)
        assert got_2050 == [], got_2050

        got_415 = sos.collect_answers(ans, 415, 515)
        assert [r["line"] for r in got_415] == [8] and got_415[0]["span"] == [415, 515], got_415

        got_604 = sos.collect_answers(ans, 604, 1009)
        assert [r["line"] for r in got_604] == [9] and got_604[0]["frame"] == 681, got_604

        got_2623 = sos.collect_answers(ans, 2623, 2711)
        assert [r["line"] for r in got_2623] == [10], got_2623
        assert sos.where_label(got_2623[0]) == "frames 2623-2624, frame 2623, lead, piece lead_fp_2625, tag gap_flash", sos.where_label(got_2623[0])

        vdir = Path(tmp) / "verdicts"
        vdir.mkdir()
        verd = vdir / "verdicts.jsonl"
        verd.write_text(
            json.dumps({"at": "2026-10-12T20:50:14Z", "row": "who", "first": 4310, "frames": 89, "frame": 4310, "what": "words", "words": "dup from owner_answers", "source": "owner_answers.jsonl:1"}) + "\n"
            + json.dumps({"at": "2026-10-23T19:32:42Z", "row": "third_fp_5114b", "first": 5114, "frames": 16, "frame": 5129, "what": "words", "words": "terminal beard note", "source": "the owner in the Claude lead's terminal (mrtx), 2026-10-23"}) + "\n"
            + json.dumps({"at": "2026-10-24T17:06:55Z", "row": "lead_fp_4310", "first": 4310, "frames": 80, "what": "choice", "chosen": "wrong, and I say where", "source": "the owner in the terminal"}) + "\n"
        )
        got_5114 = sos.collect_answers(ans, 5114, 5129)
        assert len(got_5114) == 1 and got_5114[0]["source"] == "verdicts.jsonl" and got_5114[0]["said"] == "terminal beard note", got_5114
        got_4310_v = sos.collect_answers(ans, 4310, 4389)
        assert [r["line"] for r in got_4310_v] == [1, 7], got_4310_v

    return ("exact sub-spans, pre-split shot spans whose frame is inside the shot, frame-pinned whole-cut notes, timeline notes with person/piece/tag, "
            "terminal verdicts from verdicts.jsonl, load names on the shot, and Oct-12 string-concatenated spans match; "
            "unpinned whole-cut remarks ([0, 6394]), correction_ rows, out-of-shot frames on a narrowed sub-shot, and ISO timestamps in row names are excluded")


def formatting_inbox_and_refusals() -> str:
    """`--for-question`, `--inbox`, and CLI refusals work with no traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        ans = Path(tmp) / "owner_answers.jsonl"
        ans.write_text(
            json.dumps({"at": "2026-10-24T17:07:00Z", "row": "lead_fp_4310", "frames": [4310, 4389], "what": "choice", "chosen": "wrong, and I say where"}) + "\n"
            + json.dumps({"at": "2026-10-24T20:00:00Z", "row": "lead_fp_4310c", "frames": [4310, 4389], "what": "choice", "chosen": "keep this one"}) + "\n"
        )
        inbox = Path(tmp) / "inbox.md"
        inbox.write_text(
            "- **2026-10-24 18:15** — On 4310, check 4368 where her head goes back.\n"
            "- **2026-10-24 18:20** — Unrelated remark on 1306.\n"
        )

        plain = subprocess.run(
            [sys.executable, str(HERE / "said_on_shot.py"), "--answers", str(ans), "--shot", "4310-4389", "--inbox", str(inbox)],
            capture_output=True, text=True, check=False,
        )
        assert plain.returncode == 0, plain.stderr
        assert "said_on_shot 4310-4389 (owner_answers.jsonl): 3 line(s), oldest first" in plain.stdout, plain.stdout
        assert "line 1     lead_fp_4310" in plain.stdout and "line 2     lead_fp_4310c" in plain.stdout, plain.stdout
        assert "inbox.md:1" in plain.stdout and "check 4368" in plain.stdout and "1306" not in plain.stdout, plain.stdout

        fq = subprocess.run(
            [sys.executable, str(HERE / "said_on_shot.py"), "--answers", str(ans), "--shot", "4310-4389", "--inbox", str(inbox), "--for-question"],
            capture_output=True, text=True, check=False,
        )
        assert fq.returncode == 0, fq.stderr
        assert "What the owner already said on 4310-4389 (from owner_answers.jsonl, 3 line(s), oldest first):" in fq.stdout, fq.stdout
        assert "- line 1 (2026-10-24 17:07, `lead_fp_4310`, frames 4310-4389, choice): wrong, and I say where" in fq.stdout, fq.stdout

        for bad_args, want in (
            (["--answers", str(ans), "--shot", "bad"], "--shot 'bad': expected FIRST-LAST or FIRST"),
            (["--answers", str(ans), "--shot", "500-400"], "--shot '500-400': first frame 500 is after last frame 400"),
            (["--answers", str(Path(tmp) / "missing.jsonl"), "--shot", "4310-4389"], "file not found"),
        ):
            r = subprocess.run(
                [sys.executable, str(HERE / "said_on_shot.py"), *bad_args],
                capture_output=True, text=True, check=False,
            )
            assert r.returncode == 1 and want in r.stderr and "Traceback" not in r.stderr, (bad_args, r.returncode, r.stderr)

    return "plain and --for-question output format oldest-first with line numbers and --inbox matches; invalid --shot or missing files refuse with exit 1 and no traceback"


case("span inside the shot, pre-split overlapping shot span, and load name match; whole-cut and timestamp-in-row-name rows do not", span_and_load_matching)
case("plain and --for-question output, --inbox matching, and CLI refusals with no traceback", formatting_inbox_and_refusals)

if __name__ == "__main__":
    sys.exit(finish())
