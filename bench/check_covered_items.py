#!/usr/bin/env python3
"""What a kept cut answers (`bench/covered_items.py`), on a made list, made verdicts and a made table.

1. **A kept cut covers the pieces its own table lays**, and no piece it does not lay, even one on its frames: a piece
   that landed after the keep stays waiting (2026-10-15: lead_fp_0751b).
2. **Superseded**: an earlier build of the kept cut's frames, and an older joined item whose frames hold the kept
   cut's, a date-only stamp read as that day's start; a joined item built after the keep is not.
3. **What keeps it the owner's**: a line of theirs on an item after the keep leaves it out; a joined item whose last
   choice is not a keep answers nothing; a kept cut with no table is said and answers nothing.

No card, no server.

    <python> bench/check_covered_items.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import covered_items as ci  # noqa: E402


def world(tables: Path) -> tuple[list[dict], list[dict]]:
    kept = "v_0100_0199_joined_2026-10-14T2023Z_stacked"
    items = [{"id": kept, "first": "100", "frames": "100"},
             {"id": "v_0100_0199_joined_2026-10-14T0033Z_stacked", "first": "100", "frames": "100"},     # earlier build: superseded
             {"id": "v_0100_0199_joined_2026-10-14T2257Z_stacked", "first": "100", "frames": "100"},     # later build: new
             {"id": "v_0000_0999_joined_2026-10-13_stacked", "first": "0", "frames": "1000"},           # older, holds it: superseded
             {"id": "v_0150_0999_joined_2026-10-13T1900Z_stacked", "first": "150", "frames": "850"},    # older, does not hold it
             {"id": "p_laid"}, {"id": "r__x__p_named", "name": "p_named"}, {"id": "p_after"}, {"id": "p_spoken"},
             {"id": "v_0300_0399_joined_2026-10-14T1000Z_stacked", "first": "300", "frames": "100"},     # wronged: answers nothing
             {"id": "p_300"},
             {"id": "v_0500_0599_joined_2026-10-14T1000Z_stacked", "first": "500", "frames": "100"}]     # kept, no table
    lines = [{"what": "choice", "row": kept, "chosen": "keep this one", "at": "2026-10-14T20:31:53Z"},
             {"what": "words", "row": "p_spoken", "words": "the owner, later", "at": "2026-10-14T21:00:00Z"},
             {"what": "choice", "row": "v_0300_0399_joined_2026-10-14T1000Z_stacked", "chosen": "keep this one", "at": "2026-10-14T10:00:00Z"},
             {"what": "choice", "row": "v_0300_0399_joined_2026-10-14T1000Z_stacked", "chosen": "wrong, and I say where", "at": "2026-10-14T11:00:00Z"},
             {"what": "choice", "row": "v_0500_0599_joined_2026-10-14T1000Z_stacked", "chosen": "keep this one", "at": "2026-10-14T10:00:00Z"}]
    (tables / "t_v_0100_0199_joined_2026-10-14T2023Z.json").write_text(json.dumps({"rows": [
        {"first": 100, "last": 149, "name": "p_laid", "subject": "a"}, {"first": 150, "last": 199, "name": "p_named", "subject": "a"},
        {"first": 120, "last": 140, "name": "p_spoken", "subject": "b"}]}))
    (tables / "t_v_0300_0399_joined_2026-10-14T1000Z.json").write_text(json.dumps({"rows": [{"first": 300, "last": 399, "name": "p_300", "subject": "a"}]}))
    return items, lines


def what_a_kept_cut_answers() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        items, lines = world(t)
        got = ci.covered(items, lines, [t])
    ids = {x["id"]: x["why"] for x in got["covered"]}
    want = {"p_laid", "r__x__p_named", "v_0100_0199_joined_2026-10-14T0033Z_stacked", "v_0000_0999_joined_2026-10-13_stacked"}
    assert set(ids) == want, f"covered {sorted(ids)}, wanted {sorted(want)}"
    assert ids["p_laid"].startswith("laid in the cut you kept") and ids["v_0100_0199_joined_2026-10-14T0033Z_stacked"].startswith("an earlier build")
    assert "which its frames hold" in ids["v_0000_0999_joined_2026-10-13_stacked"], ids
    assert [k["id"] for k in got["kept"]] == ["v_0100_0199_joined_2026-10-14T2023Z_stacked"], got["kept"]
    assert got["said"] and "v_0500_0599" in got["said"][0] and "answers nothing" in got["said"][0], got["said"]
    return ("the pieces its table lays, an earlier build and an older join holding it covered; a piece it does not lay, a later build, a "
            "piece the owner spoke of since, a cut wronged after its keep, and a kept cut with no table answer nothing")


case("what a kept cut answers", what_a_kept_cut_answers)

if __name__ == "__main__":
    sys.exit(finish())
