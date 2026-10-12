#!/usr/bin/env python3
"""`bench/page_answers_ledger.py` lays each owner line beside the record that cites it; each case is a way a citation could be missed or invented.

  a_cast_range_counts            `lines 3-5` in a cast line's `by` cites lines 3, 4 and 5
  a_verdict_source_counts        `answers.jsonl:7` in a verdict's `source` cites line 7
  a_shot_note_counts             `answers.jsonl line 2` in a shot's note cites line 2
  a_plan_note_counts             `line 9` in a plan departure's note cites line 9
  another_persons_line_is_left_out   a line by a tester is not in the ledger
  an_uncited_line_is_listed_whole    a words line no record cites is in the uncited section with its words
  nothing_is_written_to_a_record     the four input files are byte for byte what they were

    <python> bench/check_page_answers_ledger.py

No model, no card; the standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "page_answers_ledger.py"


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lines = [
            {"who": "owner", "row": "who", "what": "choice", "chosen": "1", "source_frame": 10, "frames": [0, 20], "at": "2026-10-12T10:00:00Z"},   # 1
            {"who": "owner", "row": "who", "what": "words", "words": "SHOT NOTE WORDS", "source_frame": 10, "frames": [0, 20]},                  # 2
            {"who": "owner", "row": "b20", "what": "choice", "chosen": "2", "source_frame": 30, "frames": [21, 40]},                              # 3
            {"who": "owner", "row": "b20", "what": "choice", "chosen": "not sure", "source_frame": 30, "frames": [21, 40]},                       # 4
            {"who": "owner", "row": "b20", "what": "words", "words": "cast words", "source_frame": 30, "frames": [21, 40]},                        # 5
            {"who": "mrx", "row": "who", "what": "words", "words": "TESTER LINE", "source_frame": 10, "frames": [0, 20]},                          # 6
            {"who": "owner", "row": "r__s__clip", "what": "choice", "chosen": "kept", "frames": [0, 20]},                                          # 7
            {"who": "owner", "row": "segments", "what": "words", "words": "NOWHERE WORDS", "source_frame": 50, "frames": [41, 60]},                # 8
            {"who": "owner", "row": "a04", "what": "words", "words": "plan words", "frames": [41, 60]},                                             # 9
        ]
        (d / "answers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (d / "cast.jsonl").write_text(json.dumps({"subject": "a", "frame": 30, "by": "owner, on the page at  (answers.jsonl lines 3-5), entered by page_answers_to_cast"}) + "\n")
        (d / "verdicts.jsonl").write_text(json.dumps({"row": "r__s__clip", "source": "answers.jsonl:7", "chosen": "kept"}) + "\n")
        (d / "shots.json").write_text(json.dumps({"shots": [{"first": 0, "last": 20, "note": "the owner, answers.jsonl line 2: SHOT NOTE WORDS", "who": {}}]}))
        (d / "plan.json").write_text(json.dumps({"departures": [{"first": 41, "last": 60, "note": "'plan words' (line 9)"}], "subjects": {}}))
        before = {p.name: p.read_bytes() for p in d.iterdir()}
        r = subprocess.run([sys.executable, str(TOOL), "--answers", str(d / "answers.jsonl"), "--cast", str(d / "cast.jsonl"), "--verdicts", str(d / "verdicts.jsonl"),
                            "--shots", str(d / "shots.json"), "--plan", str(d / "plan.json"), "--out", str(d / "out" / "ledger.md")], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        led = json.loads((d / "out" / "ledger.json").read_text())
        by = {row["line"]: row["cited_by"] for row in led["rows"]}
        case("a_cast_range_counts", by[3] == ["cast"] and by[4] == ["cast"] and by[5] == ["cast"], str(by))
        case("a_verdict_source_counts", by[7] == ["verdicts"], str(by.get(7)))
        case("a_shot_note_counts", by[2] == ["shots"], str(by.get(2)))
        case("a_plan_note_counts", by[9] == ["plan"], str(by.get(9)))
        case("another_persons_line_is_left_out", 6 not in by and led["owners"] == 8, str(sorted(by)))
        md = (d / "out" / "ledger.md").read_text()
        unc = [row["line"] for row in led["uncited"]]
        case("an_uncited_line_is_listed_whole", unc == [1, 8] and "NOWHERE WORDS" in md.split("## Lines no record cites")[1].split("## Every line")[0], str(unc))
        after = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}
        case("nothing_is_written_to_a_record", after == before, str(sorted(after)))

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
