#!/usr/bin/env python3
"""`bench/cast_file.py` keeps a person's answers about who the outlined people on a frame are, and each case is a way one could be lost or misread.

Why it exists: on 2026-10-10 the same answer ("that subject is not here") was typed in four places in four forms,
a tile number was read against the wrong list, a later tracker's tile did not show everybody, "not in the shot"
was said of subjects who were in part of it, and a default drawn from the trackers' takes would have been wrong on
the two shots where a wrong take hid a subject. The file is the one place an answer is kept; these cases hold the
rules that make it worth keeping.

  two_writers_lose_nothing                 two processes appending at once: every line is there, whole, in each writer's order
  a_cut_short_last_line_stops_an_append    a file a dead writer left is not added to
  a_line_that_cannot_be_read_is_named      and is not skipped
  a_tile_is_answered_by_naming_its_people  a number becomes that person's box; no number is stored; a subject not named is
                                           `none of the outlined people`, never "not on this frame"; one time for the tile
  a_tile_is_stored_whole_or_not_at_all     one bad pair and nothing is written; the pairs that cannot be are refused by name
  a_tile_that_does_not_show_everybody_is_refused   a table whose shot left another tracker's people out, or does not say
  the_latest_line_for_a_frame_stands       and a withdrawal leaves no answer
  a_standing_box_is_not_passed_over        a second tile of the same frame that does not outline the subject cannot erase a box
  seeds_are_read_by_the_trackers_own_parser    the text for a load goes through `subject_track.parse_seeds` and names the right shots
  none_outlined_needs_a_choice             no default; `leave` writes no seed, `empty` writes `none` and is called the strong reading
  a_box_is_used_before_none_outlined       in one shot; the other answer is named as unused; later means later in the file
  another_picture_shape_is_refused         the same shape scales the box
  an_answer_outside_the_load_is_not_moved  and one inside the load in no shot is named

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
JOB = ["lead", "second", "third"]
#: A census table as a version 1 tracker saved it: two shots, the first with two people on its tile.
TABLE = {"version": 1, "size": SIZE, "shots": [
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
    C.append(sys.argv[2], [C.entry(sys.argv[3], k, C.HERE, by=sys.argv[3], box=[1, 2, 30, 40], size=[1024, 768], table="t.json"),
                           C.entry(sys.argv[3] + "_b", k, C.NONE_OUTLINED, by=sys.argv[3])])
"""


def _raises(fn, *words) -> str:
    try:
        fn()
    except (ValueError, SystemExit) as e:
        assert all(w in str(e) for w in words), f"refused, but the message does not say {words}: {e}"
        return str(e)
    raise AssertionError(f"was not refused (wanted a message with {words})")


def _tile(shot: int, *named, table=TABLE, by="a person") -> list[dict]:
    return C.from_tile(table, "census_shots.json", FIRST, shot, JOB, list(named), by)


def two_writers_lose_nothing():
    with tempfile.TemporaryDirectory() as d:
        path, each = Path(d) / "cast.jsonl", 300
        running = [subprocess.Popen([sys.executable, "-c", WRITER, str(HERE), str(path), who, str(each)]) for who in ("one", "two")]
        assert [p.wait() for p in running] == [0, 0], "a writer failed"
        raw = path.read_text().splitlines()
        assert len(raw) == 4 * each, f"{len(raw)} lines from two writers of {each} answers of two lines"
        lines = C.read(path)                    # every line parses, or this raises and names it
        for who in ("one", "two"):
            got = [line["frame"] for line in lines if line["subject"] == who]
            assert got == list(range(each)), f"writer {who}'s lines are not all there in its own order"
        # the two lines of one answer are never parted by the other writer's
        pairs = [(lines[k]["subject"], lines[k + 1]["subject"]) for k in range(0, len(lines), 2)]
        assert all(b == a + "_b" for a, b in pairs), "an answer's lines were parted by another writer's"
        return f"{len(raw)} whole lines, each answer's two together"


def a_cut_short_last_line_stops_an_append():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, [C.entry("lead", 5, C.NONE_OUTLINED, by="a")])
        with open(path, "a") as f:
            f.write('{"schema": "h3_cast/1", "subj')
        before = path.read_bytes()
        _raises(lambda: C.append(path, [C.entry("lead", 6, C.NONE_OUTLINED, by="a")]), "does not end with a whole line")
        assert path.read_bytes() == before, "the refused append changed the file"


def a_line_that_cannot_be_read_is_named():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, [C.entry("lead", 5, C.NONE_OUTLINED, by="a")])
        with open(path, "a") as f:
            f.write("not a line\n")
        _raises(lambda: C.read(path), "line 2")


def a_tile_is_answered_by_naming_its_people():
    lines = _tile(1, (2, "lead"))
    by_subject = {line["subject"]: line for line in lines}
    assert list(by_subject) == JOB, f"every subject the job names gets a line: {list(by_subject)}"
    lead = by_subject["lead"]
    assert (lead["frame"], lead["answer"], lead["box"], lead["size"]) == (1020, C.HERE, [600.0, 80.0, 300.0, 500.0], SIZE), lead
    assert lead["table"] == "census_shots.json" and "person" not in lead and "shot" not in lead, f"a number or a shot was stored: {sorted(lead)}"
    for other in ("second", "third"):
        assert (by_subject[other]["frame"], by_subject[other]["answer"], by_subject[other]["box"]) == (1020, C.NONE_OUTLINED, None), by_subject[other]
    assert len({line["at"] for line in lines}) == 1, "a tile's lines carry one time"
    # the words are what the person said. They were shown outlines; a subject on the frame with no outline is not ruled out
    assert C.NONE_OUTLINED == "none of the outlined people" and "frame" not in C.NONE_OUTLINED and "shot" not in C.NONE_OUTLINED
    nobody = _tile(2)
    assert [line["answer"] for line in nobody] == [C.NONE_OUTLINED] * 3 and {line["frame"] for line in nobody} == {1070}, nobody
    both = {line["subject"]: line["answer"] for line in _tile(1, (1, "third"), (2, "lead"))}
    assert both == {"lead": C.HERE, "second": C.NONE_OUTLINED, "third": C.HERE}, both


def a_tile_is_stored_whole_or_not_at_all():
    _raises(lambda: _tile(1, (3, "lead")), "no person 3")
    _raises(lambda: _tile(9, (1, "lead")), "no shot 9")
    _raises(lambda: _tile(1, (1, "lead"), (1, "third")), "given to two subjects")
    _raises(lambda: _tile(1, (1, "lead"), (2, "lead")), "given two numbered people")
    _raises(lambda: _tile(1, (1, "somebody")), "not one of the job's subjects")
    _raises(lambda: C.from_tile(TABLE, "t.json", FIRST, 1, ["lead", "lead"], [], "a"), "each once")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        good = _tile(1, (2, "lead"))
        bad = [dict(line) for line in good]
        bad[2]["by"] = ""                              # the third line of the tile cannot be stored
        _raises(lambda: C.append(path, bad), "who gave it")
        assert not path.exists() or path.read_bytes() == b"", "part of a tile was stored"
        C.append(path, good)
        assert len(C.read(path)) == 3
        _raises(lambda: C.append(path, []), "nothing to add")


def a_tile_that_does_not_show_everybody_is_refused():
    table = json.loads(json.dumps(TABLE))
    table["shots"][1]["others_left_out"]["on_shown_frame"] = 1
    _raises(lambda: _tile(2, (1, "lead"), table=table), "not everybody")
    _raises(lambda: _tile(2, table=table), "not everybody")
    # a table made before the field existed does not SAY that nobody was left out, and is not read as if it did
    old = json.loads(json.dumps(TABLE))
    del old["shots"][0]["others_left_out"]
    _raises(lambda: _tile(1, (1, "lead"), table=old), "does not say whether anybody was left out")
    _raises(lambda: _tile(1, table=old), "does not say whether anybody was left out")
    # None is what the table writes when the tracker had nobody to leave out: that IS everybody (shot 1 of TABLE)
    assert _tile(1, (1, "lead"))[0]["answer"] == C.HERE
    # a version 2 table lists everybody by construction and carries no such field: accepted as it is
    new = json.loads(json.dumps(old))
    new["version"] = 2
    del new["shots"][1]["others_left_out"]
    assert _tile(1, (1, "lead"), table=new)[0]["answer"] == C.HERE and len(_tile(2, table=new)) == 3
    # a version this tool does not know is refused, never read as complete; so is a table with no version
    for unknown in (3, None, "2"):
        later = json.loads(json.dumps(new))
        later["version"] = unknown
        _raises(lambda: _tile(1, (1, "lead"), table=later), "does not know")


def the_latest_line_for_a_frame_stands():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, _tile(1, (1, "lead"), by="a"))
        C.append(path, _tile(1, (2, "lead"), by="b"))
        C.append(path, _tile(2, (1, "lead"), by="a"))
        now = C.current(C.read(path))
        assert now[("lead", 1020)]["by"] == "b" and now[("lead", 1020)]["box"][0] == 600.0, now[("lead", 1020)]
        C.append(path, [C.entry("lead", 1070, C.WITHDRAWN, by="a")])
        assert ("lead", 1070) not in C.current(C.read(path)), "a withdrawn answer still stands"
        assert ("second", 1070) in C.current(C.read(path)), "a withdrawal took another subject's answer with it"
        assert len(C.read(path)) == 10, "nothing is rewritten: every line given is still in the file"


def a_standing_box_is_not_passed_over():
    # census B of the same clip: the same shown frame, and the lead has no outline on it
    other = json.loads(json.dumps(TABLE))
    other["shots"][0]["people"] = [{"person": 1, "box": [100, 50, 200, 600]}]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cast.jsonl"
        C.append(path, _tile(1, (2, "lead"), (1, "second")))
        before = path.read_bytes()
        said = _raises(lambda: C.append(path, _tile(1, (1, "second"), table=other, by="b")), "lead has a confirmed box on frame 1020", "withdraw it first")
        assert "census_shots.json" in said and "a person" in said, f"the refusal does not say whose box, from which table: {said}"
        assert path.read_bytes() == before, "a refused tile added a line: the second's line of the same tile went in"
        assert C.seeds(C.read(path), "lead", FIRST, SIZE, SHOTS, C.LEAVE)["text"] == "frame 20: 600, 80, 300, 500"
        # the two commands a person means: take the box back, then the tile goes in
        C.append(path, [C.entry("lead", 1020, C.WITHDRAWN, by="b")])
        C.append(path, _tile(1, (1, "second"), table=other, by="b"))
        assert C.current(C.read(path))[("lead", 1020)]["answer"] == C.NONE_OUTLINED
        # a box replacing a box, and a box replacing a `none`, go in
        C.append(path, _tile(1, (2, "lead"), (1, "second"), by="c"))
        C.append(path, _tile(1, (1, "lead"), (2, "second"), by="d"))
        now = C.current(C.read(path))
        assert (now[("lead", 1020)]["by"], now[("lead", 1020)]["box"][0], now[("second", 1020)]["box"][0]) == ("d", 100.0, 600.0), now
        # and the same answer with the second left unnamed is refused: her box would be passed over
        _raises(lambda: C.append(path, _tile(1, (1, "lead"), by="e")), "second has a confirmed box")


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
    # shot 1: person 2 is the lead and person 1 the third; shot 2: nobody of ours
    return _tile(1, (2, "lead"), (1, "third")) + _tile(2)


def seeds_are_read_by_the_trackers_own_parser():
    out = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "frame 20: 600, 80, 300, 500\nframe 70: none", repr(out["text"])
    parsed = _tracker().parse_seeds(out["text"], SHOTS)
    assert parsed == {1: (20, (600.0, 80.0, 300.0, 500.0)), 2: (70, None)}, parsed
    assert "third" not in json.dumps(out) and C.seeds(_answers(), "third", FIRST, SIZE, SHOTS, C.LEAVE)["text"] == "frame 20: 100, 50, 200, 600"


def none_outlined_needs_a_choice():
    _raises(lambda: C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, None), "say what")
    _raises(lambda: C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, ""), "say what")
    left = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert left["text"] == "frame 20: 600, 80, 300, 500" and left["shots"][1]["seed"] is None, left["text"]
    assert "does not say lead is not in the shot" in left["shots"][1]["why"] and left["none_outlined_means"] == C.NONE_OUTLINED_MEANS[C.LEAVE]
    emptied = C.seeds(_answers(), "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert emptied["shots"][1]["seed"] == "frame 70: none", emptied["shots"][1]
    assert "STRONG" in emptied["none_outlined_means"] and "strong reading" in emptied["shots"][1]["why"], "emptying a shot is not called the strong reading"
    # second was named on neither tile: with `leave`, which is the safe reading, nothing at all is seeded for her
    assert C.seeds(_answers(), "second", FIRST, SIZE, SHOTS, C.LEAVE)["text"] == ""


def a_box_is_used_before_none_outlined():
    # the subject walks out of shot 1: on its frame 20 she is person 2, and on frame 35 a person says none outlined is her
    lines = _tile(1, (2, "lead")) + [C.entry("lead", FIRST + 35, C.NONE_OUTLINED, by="a")]
    out = C.seeds(lines, "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "frame 20: 600, 80, 300, 500", f"a later `none of the outlined people` emptied a shot the subject is in: {out['text']!r}"
    assert [u["frame"] for u in out["unused"]] == [FIRST + 35] and "not wrong, only not used" in out["unused"][0]["why"], out["unused"]
    # two boxes in one shot: the later in the file is the one seed, and the other is named
    two = [C.entry("lead", FIRST + 5, C.HERE, by="a", box=[1, 1, 50, 50], size=SIZE, table="t.json", at="2026-01-01T00:00:00Z"),
           C.entry("lead", FIRST + 30, C.HERE, by="a", box=[2, 2, 60, 60], size=SIZE, table="t.json", at="2026-01-02T00:00:00Z")]
    out = C.seeds(two, "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert out["text"] == "frame 30: 2, 2, 60, 60" and [u["frame"] for u in out["unused"]] == [FIRST + 5], out
    # "later" is the file's order and nothing else: the same two lines the other way round, the later LINE carrying
    # the earlier time, and it is the later line that is used
    out = C.seeds(list(reversed(two)), "lead", FIRST, SIZE, SHOTS, C.LEAVE)
    assert out["text"] == "frame 5: 1, 1, 50, 50" and [u["frame"] for u in out["unused"]] == [FIRST + 30], f"sorted on a typed time: {out['text']!r}"


def another_picture_shape_is_refused():
    lines = _tile(1, (2, "lead"))
    _raises(lambda: C.seeds(lines, "lead", FIRST, [1344, 768], SHOTS, C.LEAVE), "another shape")
    half = C.seeds(lines, "lead", FIRST, [512, 384], SHOTS, C.LEAVE)
    assert half["text"] == "frame 20: 300, 40, 150, 250", half["text"]
    # an answer of another shape that is not in this load stops nothing
    far = lines + [C.entry("lead", 5000, C.HERE, by="a", box=[1, 1, 5, 5], size=[1344, 768], table="t.json")]
    assert C.seeds(far, "lead", FIRST, SIZE, SHOTS, C.LEAVE)["outside_this_load"] == [5000]


def an_answer_outside_the_load_is_not_moved():
    lines = [C.entry("lead", FIRST - 1, C.HERE, by="a", box=[1, 1, 50, 50], size=SIZE, table="t.json"),
             C.entry("lead", FIRST + 100, C.NONE_OUTLINED, by="a")]
    out = C.seeds(lines, "lead", FIRST, SIZE, SHOTS, C.EMPTY)
    assert out["text"] == "" and out["outside_this_load"] == [FIRST - 1, FIRST + 100], out
    assert all(row["seed"] is None and row["why"] == "no answer inside this shot" for row in out["shots"]), out["shots"]
    # every answer is in exactly one list: with a hole between two shots, an answer in the hole is named, not dropped
    holed = C.seeds([C.entry("lead", FIRST + 15, C.NONE_OUTLINED, by="a")], "lead", FIRST, SIZE, [(0, 10), (20, 30)], C.EMPTY)
    assert holed["inside_the_load_in_no_shot"] == [FIRST + 15] and holed["text"] == "" and not holed["outside_this_load"], holed


def main() -> int:
    print("bench/check_cast_file.py\n")
    for fn in (two_writers_lose_nothing, a_cut_short_last_line_stops_an_append, a_line_that_cannot_be_read_is_named,
               a_tile_is_answered_by_naming_its_people, a_tile_is_stored_whole_or_not_at_all,
               a_tile_that_does_not_show_everybody_is_refused, the_latest_line_for_a_frame_stands, a_standing_box_is_not_passed_over,
               seeds_are_read_by_the_trackers_own_parser, none_outlined_needs_a_choice, a_box_is_used_before_none_outlined,
               another_picture_shape_is_refused, an_answer_outside_the_load_is_not_moved):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
