#!/usr/bin/env python3
"""`bench/cast_file.py` keeps a person's answers about who is on a frame, and each case is a way one could be lost or misread.

Why it exists: on 2026-10-10 the same answer ("that subject is not here") was typed in four places in four forms,
a tile number was read against the wrong list, a later tracker's tile did not show everybody, and "not in the
shot" was said of subjects who were in part of it. The file is the one place an answer is kept; these cases hold
the rules that make it worth keeping.

  two_writers_lose_nothing                 two processes appending at once: every line is there, whole, in each writer's order
  a_cut_short_last_line_stops_an_append    a file a dead writer left is not added to
  a_line_that_cannot_be_read_is_named      and is not skipped
  an_answer_holds_a_box_and_a_frame        the tile number becomes that person's box; no number is stored; nobody is "not on this frame"
  a_tile_that_does_not_show_everybody_is_refused   a table whose shot left another tracker's people out
  a_number_not_on_the_tile_is_refused
  the_latest_line_for_a_frame_stands       and a withdrawal leaves no answer
  seeds_are_read_by_the_trackers_own_parser    the text for a load goes through `subject_track.parse_seeds` and names the right shots
  not_on_this_frame_needs_a_choice         no default; `leave` writes no seed, `empty` writes `none`, and the result says which in words
  a_box_is_used_before_not_on_this_frame   in one shot; the other answer is named as unused
  another_picture_shape_is_refused         the same shape scales the box
  an_answer_outside_the_load_is_not_moved

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_cast_file.py

No model, no card. The parser case imports the tracker's module, so it needs ComfyUI's Python.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(HERE))

from _lib import case, finish  # noqa: E402

import cast_file as C  # noqa: E402

SIZE = [1024, 768]
#: A census table as a preview saves it: two shots, the first with two people on its tile.
TABLE = {"size": SIZE, "shots": [
    {"shot": 1, "first_frame": 0, "last_frame": 39, "shown_frame": 20, "others_left_out": None,
     "people": [{"person": 1, "box": [100, 50, 200, 600]}, {"person": 2, "box": [600, 80, 300, 500]}]},
    {"shot": 2, "first_frame": 40, "last_frame": 99, "shown_frame": 70, "others_left_out": {"on_shown_frame": 0, "on_frames_looked_at": 0},
     "people": [{"person": 1, "box": [400, 300, 60, 90]}]},
]}
SHOTS = [(0, 40), (40, 100)]
FIRST = 1000
WRITER = """
import sys
sys.path.insert(0, sys.argv[1])
import cast_file as C
for k in range(int(sys.argv[4])):
    C.append(sys.argv[2], C.entry(sys.argv[3], k, C.HERE, by=sys.argv[3], box=[1, 2, 30, 40], size=[1024, 768], table="t.json"))
"""


def _raises(fn, *words) -> str:
    try:
        fn()
    except (ValueError, SystemExit) as e:
        assert all(w in str(e) for w in words), f"refused, but the message does not say {words}: {e}"
        return str(e)
    raise AssertionError(f"was not refused (wanted a message with {words})")


def two_writers_lose_nothing():
    with tempfile.TemporaryDirectory() as d:
        path, each = Path(d) / "cast.jsonl", 300
        running = [subprocess.Popen([sys.executable, "-c", WRITER, str(HERE), str(path), who, str(each)]) for who in ("one", "two")]
        assert [p.wait() for p in running] == [0, 0], "a writer failed"
        raw = path.read_text().splitlines()
        assert len(raw) == 2 * each, f"{len(raw)} lines from two writers of {each}"
        lines = C.read(path)                    # every line parses, or this raises and names it
        for who in ("one", "two"):
            got = [line["frame"] for line in lines if line["subject"] == who]
            assert got == list(range(each)), f"writer {who}'s lines are not all there in its own order"
        return f"{len(raw)} whole lines"


def a_cut_short_last_line_stops_an_append():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, C.entry("lead", 5, C.NOT_ON_FRAME, by="a"))
        with open(path, "a") as f:
            f.write('{"schema": "h3_cast/1", "subj')
        before = path.read_bytes()
        _raises(lambda: C.append(path, C.entry("lead", 6, C.NOT_ON_FRAME, by="a")), "does not end with a whole line")
        assert path.read_bytes() == before, "the refused append changed the file"


def a_line_that_cannot_be_read_is_named():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, C.entry("lead", 5, C.NOT_ON_FRAME, by="a"))
        with open(path, "a") as f:
            f.write("not a line\n")
        _raises(lambda: C.read(path), "line 2")


def an_answer_holds_a_box_and_a_frame():
    line = C.from_table(TABLE, "census_shots.json", FIRST, 1, "lead", 2, "a person")
    assert (line["frame"], line["answer"], line["box"], line["size"]) == (1020, C.HERE, [600.0, 80.0, 300.0, 500.0], SIZE), line
    assert line["table"] == "census_shots.json" and "person" not in line and "shot" not in line, f"a number or a shot was stored: {sorted(line)}"
    nobody = C.from_table(TABLE, "census_shots.json", FIRST, 2, "lead", None, "a person")
    assert (nobody["frame"], nobody["answer"], nobody["box"]) == (1070, C.NOT_ON_FRAME, None), nobody
    assert C.NOT_ON_FRAME == "not on this frame", "the words say a frame, never a shot"


def a_tile_that_does_not_show_everybody_is_refused():
    table = json.loads(json.dumps(TABLE))
    table["shots"][1]["others_left_out"]["on_shown_frame"] = 1
    _raises(lambda: C.from_table(table, "t.json", FIRST, 2, "lead", 1, "a"), "not everybody")
    _raises(lambda: C.from_table(table, "t.json", FIRST, 2, "lead", None, "a"), "not everybody")
    # a table made before the field existed does not SAY that nobody was left out, and is not read as if it did
    old = json.loads(json.dumps(TABLE))
    del old["shots"][0]["others_left_out"]
    _raises(lambda: C.from_table(old, "t.json", FIRST, 1, "lead", 1, "a"), "does not say whether anybody was left out")
    _raises(lambda: C.from_table(old, "t.json", FIRST, 1, "lead", None, "a"), "does not say whether anybody was left out")
    # None is what the table writes when the tracker had nobody to leave out: that IS everybody (shot 1 of TABLE)
    assert C.from_table(TABLE, "t.json", FIRST, 1, "lead", 1, "a")["answer"] == C.HERE


def a_number_not_on_the_tile_is_refused():
    _raises(lambda: C.from_table(TABLE, "t.json", FIRST, 1, "lead", 3, "a"), "no person 3")
    _raises(lambda: C.from_table(TABLE, "t.json", FIRST, 9, "lead", 1, "a"), "no shot 9")


def the_latest_line_for_a_frame_stands():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, C.from_table(TABLE, "t.json", FIRST, 1, "lead", 1, "a"))
        C.append(path, C.from_table(TABLE, "t.json", FIRST, 1, "lead", 2, "b"))
        C.append(path, C.from_table(TABLE, "t.json", FIRST, 2, "lead", 1, "a"))
        now = C.current(C.read(path))
        assert now[("lead", 1020)]["by"] == "b" and now[("lead", 1020)]["box"][0] == 600.0, now[("lead", 1020)]
        C.append(path, C.entry("lead", 1070, C.WITHDRAWN, by="a"))
        assert ("lead", 1070) not in C.current(C.read(path)), "a withdrawn answer still stands"
        assert len(C.read(path)) == 4, "nothing is rewritten: every line given is still in the file"


def _tracker():
    """`subject_track` as `bench/check_subject_track.py` loads it: a module of a stand-in package, on the CPU."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(COMFY))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.subject_track", REPO / "subject_track.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_track"] = module
    spec.loader.exec_module(module)
    return module


def _answers() -> list[dict]:
    return [C.from_table(TABLE, "t.json", FIRST, 1, "lead", 2, "a"), C.from_table(TABLE, "t.json", FIRST, 2, "lead", None, "a"),
            C.from_table(TABLE, "t.json", FIRST, 1, "third", 1, "a")]


def seeds_are_read_by_the_trackers_own_parser():
    out = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "frame 20: 600, 80, 300, 500\nframe 70: none", repr(out["text"])
    parsed = _tracker().parse_seeds(out["text"], SHOTS)
    assert parsed == {1: (20, (600.0, 80.0, 300.0, 500.0)), 2: (70, None)}, parsed
    assert "third" not in json.dumps(out) and C.seeds(_answers(), "third", FIRST, SIZE, SHOTS, C.LEAVE)["text"] == "frame 20: 100, 50, 200, 600"


def not_on_this_frame_needs_a_choice():
    _raises(lambda: C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, None), "say what")
    _raises(lambda: C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, ""), "say what")
    left = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert left["text"] == "frame 20: 600, 80, 300, 500" and left["shots"][1]["seed"] is None, left["text"]
    assert "does not say not in the shot" in left["shots"][1]["why"] and left["not_on_frame_means"] == C.NOT_ON_FRAME_MEANS[C.LEAVE]
    emptied = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert emptied["shots"][1]["seed"] == "frame 70: none" and "emptied" in emptied["not_on_frame_means"], emptied["shots"][1]


def a_box_is_used_before_not_on_this_frame():
    # the subject walks out of shot 1: on its frame 20 she is person 2, and on frame 35 a person says she is not there
    lines = [C.from_table(TABLE, "t.json", FIRST, 1, "lead", 2, "a"), C.entry("lead", FIRST + 35, C.NOT_ON_FRAME, by="a", at="2099-01-01T00:00:00Z")]
    out = C.seeds(lines, "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "frame 20: 600, 80, 300, 500", f"a later `not on this frame` emptied a shot the subject is in: {out['text']!r}"
    assert [u["frame"] for u in out["unused"]] == [FIRST + 35] and "not wrong, only not used" in out["unused"][0]["why"], out["unused"]
    # two boxes in one shot: the later given is the one seed, and the other is named
    two = [C.entry("lead", FIRST + 5, C.HERE, by="a", box=[1, 1, 50, 50], size=SIZE, table="t.json", at="2026-01-01T00:00:00Z"),
           C.entry("lead", FIRST + 30, C.HERE, by="a", box=[2, 2, 60, 60], size=SIZE, table="t.json", at="2026-01-02T00:00:00Z")]
    out = C.seeds(two, "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert out["text"] == "frame 30: 2, 2, 60, 60" and [u["frame"] for u in out["unused"]] == [FIRST + 5], out
    # "later" is the file's order and nothing else: the same two lines the other way round, the later LINE carrying
    # the earlier time, and it is the later line that is used
    out = C.seeds(list(reversed(two)), "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert out["text"] == "frame 5: 1, 1, 50, 50" and [u["frame"] for u in out["unused"]] == [FIRST + 30], f"sorted on a typed time: {out['text']!r}"


def another_picture_shape_is_refused():
    lines = [C.from_table(TABLE, "t.json", FIRST, 1, "lead", 2, "a")]
    _raises(lambda: C.seeds(lines, "lead", FIRST, [1344, 768], SHOTS, C.LEAVE), "another shape")
    half = C.seeds(lines, "lead", FIRST, [512, 384], SHOTS, C.LEAVE)
    assert half["text"] == "frame 20: 300, 40, 150, 250", half["text"]
    # an answer of another shape that is not in this load stops nothing
    far = lines + [C.entry("lead", 5000, C.HERE, by="a", box=[1, 1, 5, 5], size=[1344, 768], table="t.json")]
    assert C.seeds(far, "lead", FIRST, SIZE, SHOTS, C.LEAVE)["outside_this_load"] == [5000]


def an_answer_outside_the_load_is_not_moved():
    lines = [C.entry("lead", FIRST - 1, C.HERE, by="a", box=[1, 1, 50, 50], size=SIZE, table="t.json"),
             C.entry("lead", FIRST + 100, C.NOT_ON_FRAME, by="a")]
    out = C.seeds(lines, "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "" and out["outside_this_load"] == [FIRST - 1, FIRST + 100], out
    assert all(row["seed"] is None and row["why"] == "no answer inside this shot" for row in out["shots"]), out["shots"]
    # every answer is in exactly one list: with a hole between two shots, an answer in the hole is named, not dropped
    holed = C.seeds([C.entry("lead", FIRST + 15, C.NOT_ON_FRAME, by="a")], "lead", FIRST, SIZE, [(0, 10), (20, 30)], C.EMPTY)
    assert holed["inside_the_load_in_no_shot"] == [FIRST + 15] and holed["text"] == "" and not holed["outside_this_load"], holed


def main() -> int:
    print("bench/check_cast_file.py\n")
    for fn in (two_writers_lose_nothing, a_cut_short_last_line_stops_an_append, a_line_that_cannot_be_read_is_named,
               an_answer_holds_a_box_and_a_frame, a_tile_that_does_not_show_everybody_is_refused, a_number_not_on_the_tile_is_refused,
               the_latest_line_for_a_frame_stands, seeds_are_read_by_the_trackers_own_parser, not_on_this_frame_needs_a_choice,
               a_box_is_used_before_not_on_this_frame, another_picture_shape_is_refused, an_answer_outside_the_load_is_not_moved):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
