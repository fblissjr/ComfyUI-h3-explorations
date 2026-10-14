#!/usr/bin/env python3
"""A person's look written into the shots record (`bench/shots_look.py`): what it writes, and what it refuses.

1. **A look is one line** under `who[LABEL].seen_looks` of the shot, with its part, frames (the whole shot when none
   are given), `is`, why, by, when and who entered it; the record reads back through `shot_coverage.load_shots`, a
   copy of it before is kept, and `shot_coverage.nothing_to_make` reads the line.
2. **Nothing is removed**: a later `seen` is a second line.
3. **Refused, with the record unchanged**: no shot begins there, a subject the record does not have, a subject not
   present `yes`, a word that is neither a part nor a class of the part model, frames outside the shot, no why or no
   by; and a dry run writes nothing.

    <python> bench/check_shots_look.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import shot_coverage  # noqa: E402
import shots_look as sl  # noqa: E402

SHOTS = {"schema": "h3_shots/1", "video": "t.mp4", "span": [100, 199], "subjects": ["a", "b"], "shots": [
    {"first": 100, "last": 149, "who": {"a": {"present": "yes"}, "b": {"present": "unsure"}}},
    {"first": 150, "last": 199, "who": {"a": {"present": "no"}, "b": {"present": "yes"}}}]}
WANT = {"region": "face", "parts": ["face_and_neck"]}


def run(path: Path, *more: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = sl.main(["add", "--shots", str(path), "--subject", "a", "--part", "face_and_neck", "--why", "A LOOK AT EVERY FRAME", "--by", "a reader",
                        "--entered-by", "the session", *more])
    return code, out.getvalue()


def a_look_is_one_line() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shots.json"
        path.write_text(json.dumps(SHOTS, indent=1) + "\n")
        before = path.read_text()
        code, said = run(path, "--shot", "100", "--is", "not seen", "--dry")
        assert code == 0 and path.read_text() == before and "dry: nothing written" in said, said
        code, said = run(path, "--shot", "100", "--is", "not seen")
        assert code == 0, said
        d = shot_coverage.load_shots(str(path))
        who = d["shots"][0]["who"]["a"]
        line = who["seen_looks"][0]
        assert (line["part"], line["first"], line["last"], line["is"], line["why"], line["by"], line["entered_by"]) == \
            ("face_and_neck", 100, 149, "not seen", "A LOOK AT EVERY FRAME", "a reader", "the session") and line["at"], line
        assert who["present"] == "yes" and len(list(Path(tmp).glob("shots.before_look_*.json"))) == 1, list(Path(tmp).iterdir())
        assert shot_coverage.nothing_to_make(who, WANT, 100, 149), "the coverage tool does not read the line"
        # nothing is removed: a later `seen` is a second line, and takes the earlier back on its frames
        code, said = run(path, "--shot", "100", "--is", "seen", "--frames", "140-149")
        who = shot_coverage.load_shots(str(path))["shots"][0]["who"]["a"]
        assert code == 0 and len(who["seen_looks"]) == 2 and not shot_coverage.nothing_to_make(who, WANT, 100, 149), who["seen_looks"]
    return "one line with its part, frames, is, why, by, when and who entered it; a copy before; read back; a later seen is a second line; a dry run writes nothing"


def what_it_refuses() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shots.json"
        path.write_text(json.dumps(SHOTS, indent=1) + "\n")
        before = path.read_text()
        for more, words in ((["--shot", "101", "--is", "not seen"], "no shot of the record begins on 101"),
                            (["--shot", "150", "--is", "not seen"], "presence is the owner's"),
                            (["--shot", "100", "--is", "not seen", "--frames", "140-160"], "not inside the shot"),
                            (["--shot", "100", "--is", "not seen", "--frames", "140"], "is not A-B"),
                            (["--shot", "100", "--is", "not seen", "--part", "nose"], "neither a part nor a class"),
                            (["--shot", "100", "--is", "not seen", "--why", " "], "says what was looked at"),
                            (["--shot", "100", "--is", "not seen", "--subject", "z"], "is not a subject of the record")):
            code, said = run(path, *more)
            assert code == 5 and words in said and path.read_text() == before, (more, said)
        # a class of the part model is a word a look may name
        code, said = run(path, "--shot", "100", "--is", "not seen", "--part", "Lower_Clothing")
        assert code == 0, said
    return "no such shot, a subject not present yes or not in the record, frames outside the shot or not A-B, a word that is no part or class, and no why are refused, unchanged"


case("a look is one line, and nothing is removed", a_look_is_one_line)
case("what it refuses", what_it_refuses)
sys.exit(finish())
