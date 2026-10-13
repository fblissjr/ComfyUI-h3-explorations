#!/usr/bin/env python3
"""The generated delivery table (`bench/delivery_table.py`) on a small made-up ledger: each step of its rule, with
the render that must NOT stand beside the one that does.

No file of a render is made: the rule reads the ledger's JSON, so the fixture is that JSON, written by hand here
with only the fields the rule reads, a shots file of three shots and a plan of two subjects.

1. **The owner's keep stands over a newer render in the frozen form**, when its file agrees with its record. A
   keep the owner later changed is not a keep; a keep said of a joined clip is not a choice on the render; typed
   words that say "keep" are not a choice.
2. **Else the newest agreeing render in the frozen form**: the plan's region for the subject and a reference whose
   records hold the cut to the head and the face points. An older one is passed over as taken; one with lid
   marks, one whose region is not the plan's, and one whose reference is not cut to the head are passed over
   with which. The plain cut stands by the form's exception and its line says the reason was not read.
3. **What never stands**, kept or not: no build record, a queue record that names no file, a file that disagrees
   with its record (the reading carries the ledger's own line), a whole-person render, a render of two subjects.
4. **Nothing left** is a `source` line over the whole shot naming the newest render passed over; a subject the
   shots file is unsure of says so; a subject not in a shot has no line.
5. **Several renders on one shot** stand on different frames, each later one only on frames no earlier one
   took; frames no render takes inside a shot that has a piece are a comment and not a `source` line.
6. **`restore=`**: every other subject the shots file has in the shot, `:whole` on a face row and bare on a face
   and hair row; none where the subject is alone.
7. **`--prefer`** goes first among the renders that may stand, with its reason printed; naming a render that never
   stands, a load the ledger lacks, or frames that are no shot is refused.
8. **The table reads back** through the assembler's own two readers to the rows and `source` lines decided.

No video, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_delivery_table.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table as dt  # noqa: E402

SHOTS = {"subjects": ["a", "b"], "span": [0, 149], "shots": [
    {"first": 0, "last": 49, "who": {"a": {"present": "yes"}, "b": {"present": "no"}}},
    {"first": 50, "last": 99, "who": {"a": {"present": "yes"}, "b": {"present": "yes"}}},
    {"first": 100, "last": 149, "who": {"a": {"present": "no"}, "b": {"present": "unsure"}}}]}
PLAN = {"subjects": {"a": {"region": "face"}, "b": {"region": "face and hair"}}, "aliases": {"singer": "a", "_by": "a note"}}
FROZEN = {"cut_to_the_head": True, "face_points": True, "lid_marks": False}
PLAIN = {"cut_to_the_head": True, "face_points": False, "lid_marks": False}
LIDS = {"cut_to_the_head": True, "face_points": False, "lid_marks": True}
WHOLE_FRAME = {"cut_to_the_head": False, "face_points": True, "lid_marks": False}
AGREES, DISAGREES, NO_FILE, NO_RECORD = dt.AGREES, "recorded, THE FILE DISAGREES", "a build record for its load, WHOSE QUEUE RECORD NAMES NO FILE AND NO OUTCOME", "NO BUILD RECORD"


def render(name: str, span: tuple[int, int], written: str, *, subject: str = "a", region: str | None = "face", has=FROZEN, making: str = AGREES,
           owner: tuple = (), also: tuple = ()) -> dict:
    """A render as the ledger's JSON holds one, with the fields the rule reads. `owner` is (time, what, said[, on a joined clip])."""
    return {"file": f"Video/s/{name}/{name}_00001.mp4", "name": name, "written": f"2026-01-{written}T00:00:00Z", "span": list(span), "making": making,
            "subjects": [subject, *also], "records": [{"disagrees": ["the record's load is 9 frames from 1; the file holds 3"]}] if making == DISAGREES else [],
            "form": {"region": region, "reference": {"has": has, "says": "what the ledger said of the reference"}},
            "owner": [{"at": f"2026-02-{o[0]}T00:00:00Z", "what": o[1], "said": o[2], "row": name, **({"on_the_joined_clip": o[3]} if len(o) > 3 else {})} for o in owner]}


def ledger(*renders: dict) -> dict:
    return {"schema": dt.LEDGER_SCHEMA, "share": "/share", "renders": {r["file"]: r for r in renders}}


def decide(*renders: dict, prefers=()) -> dict:
    return dt.decide(ledger(*renders), SHOTS, PLAN, [dt.parse_prefer(p) for p in prefers])


def entry(out: dict, first: int, subject: str) -> dict:
    return next(e for e in out["reading"] if (e["first"], e["subject"]) == (first, subject))


def the_owners_keep() -> str:
    kept = render("kept", (0, 49), "01", has=LIDS, owner=(("02", "choice", "keep this one"),))
    newer = render("newer_frozen", (0, 49), "09")
    e = entry(decide(kept, newer), 0, "a")
    assert [s["render"] for s in e["stands"]] == ["kept"] and e["stands"][0]["why"].startswith("step 1: the owner's keep of 2026-02-02"), e["stands"]
    assert [(p["render"], p["why"].startswith("may stand (step 2)")) for p in e["passed_over"]] == [("newer_frozen", True)], e["passed_over"]
    # a keep the owner changed afterwards, a keep said of a joined clip, and typed words are none of them a keep
    for name, owner in (("changed", (("02", "choice", "keep this one"), ("03", "choice", "wrong, and I say where"))),
                        ("of_a_joined_clip", (("02", "choice", "keep this one", "the joined clip"),)),
                        ("typed_words", (("02", "words", "keep this one"),))):
        e = entry(decide(render(name, (0, 49), "01", has=LIDS, owner=owner), newer), 0, "a")
        assert [s["render"] for s in e["stands"]] == ["newer_frozen"], (name, e["stands"])
        assert "has no keep and is not in the frozen form" in e["passed_over"][0]["why"], (name, e["passed_over"])
    # and the newest frozen one says what the owner's last choice on it was, where it stands in spite of one
    said_wrong = render("frozen_called_wrong", (0, 49), "09", owner=(("04", "choice", "wrong, and I say where"),))
    stands = entry(decide(said_wrong), 0, "a")["stands"][0]
    assert stands["why"].startswith("step 2") and stands["the_owners_last_choice"]["said"] == "wrong, and I say where", stands
    return ("the owner's keep on an agreeing render stands over a newer frozen one; a keep changed later, one said of a joined clip and typed "
            "words are not a keep; a frozen render the owner called wrong still stands by step 2 and its line says so")


def the_frozen_form() -> str:
    old, new = render("old", (0, 49), "01"), render("new", (0, 49), "05")
    e = entry(decide(old, new, render("lids", (0, 49), "08", has=LIDS), render("other_region", (0, 49), "08", region="face and hair"),
                     render("whole_frame_reference", (0, 49), "08", has=WHOLE_FRAME), render("no_reference", (0, 49), "08", has=None)), 0, "a")
    assert [s["render"] for s in e["stands"]] == ["new"] and "the body model cut to the head with the face points drawn" in e["stands"][0]["why"], e["stands"]
    why = {p["render"]: p["why"] for p in e["passed_over"]}
    assert "every frame of it on this shot is taken" in why["old"] and "marks drawn on the lids" in why["lids"], why
    assert "it replaces face and hair, where the plan's region for this subject is face" in why["other_region"], why["other_region"]
    assert "not the body model cut to the head" in why["whole_frame_reference"] and "not the body model cut to the head" in why["no_reference"], why
    # b's plan is face and hair: the same reference on a face render is not b's frozen form
    assert entry(decide(render("b_face", (50, 99), "05", subject="b", region="face")), 50, "b")["stands"] == []
    plain = entry(decide(render("plain", (0, 49), "05", has=PLAIN)), 0, "a")["stands"][0]
    assert "the form's exception" in plain["why"] and "in no field read here" in plain["why"], plain["why"]
    return ("the newest agreeing render in the frozen form stands; an older one, one with lid marks, one of another region and one whose "
            "reference is not cut to the head are passed over, each with which; the plain cut stands by the exception and says its reason was not read")


def what_never_stands() -> str:
    keep = (("02", "choice", "keep this one"),)
    never = [render("no_record", (0, 49), "09", making=NO_RECORD, owner=keep), render("names_no_file", (0, 49), "09", making=NO_FILE, owner=keep),
             render("disagrees", (0, 49), "09", making=DISAGREES, owner=keep), render("whole_person", (0, 49), "09", region="whole", owner=keep),
             render("two_subjects", (0, 49), "09", owner=keep, also=("b",))]
    out = decide(*never)
    e = entry(out, 0, "a")
    why = {p["render"]: p["why"] for p in e["passed_over"]}
    assert e["stands"] == [] and all(w.startswith("never stands: ") for w in why.values()) and len(why) == 5, why
    assert "no build record" in why["no_record"] and "names no file and no outcome" in why["names_no_file"] and "a whole-person render" in why["whole_person"]
    assert "the record's load is 9 frames from 1; the file holds 3" in why["disagrees"] and "two subjects in one pass" in why["two_subjects"], why
    assert out["rows"] == [] and {"first": 0, "last": 49, "subject": "a", "why": dt.NO_SOURCE_YET + " (the newest render passed over: " + e["passed_over"][0]["render"] + ")"} in out["sources"]
    # the control: the same render with none of those stands
    assert [s["render"] for s in entry(decide(render("fine", (0, 49), "09", has=LIDS, owner=keep)), 0, "a")["stands"]] == ["fine"]
    return "a render with no record, one whose record names no file, one whose file disagrees, a whole person and two subjects never stand, kept or not"


def nothing_left_and_several_on_a_shot() -> str:
    out = decide()
    assert out["rows"] == [] and [(s["first"], s["subject"]) for s in out["sources"]] == [(0, "a"), (50, "a"), (50, "b"), (100, "b")], out["sources"]
    assert out["sources"][0]["why"] == dt.NO_SOURCE_YET + " (no render of this subject covers these frames)"
    assert out["sources"][3]["why"].startswith("the shots file is unsure this subject is in the shot; " + dt.NO_SOURCE_YET), out["sources"][3]
    # two renders on one shot: the kept one takes its frames, the newer frozen one only what is left, and the rest is a gap
    kept = render("kept", (60, 79), "01", has=LIDS, owner=(("02", "choice", "keep this one"),))
    wide = render("wide", (55, 89), "09")
    out = decide(kept, wide)
    rows = [(r["first"], r["last"], r["name"], r["piece_first"]) for r in out["rows"]]
    assert rows == [(55, 59, "wide", 55), (60, 79, "kept", 60), (80, 89, "wide", 55)], rows
    assert [(g["first"], g["last"], g["subject"]) for g in out["gaps"]] == [(50, 54, "a"), (90, 99, "a")], out["gaps"]
    assert not any(s["subject"] == "a" and s["first"] == 50 for s in out["sources"]), "a shot with a piece was given a source line"
    assert entry(out, 50, "a")["left_as_the_source"] == [(50, 54), (90, 99)]
    # a render over two shots is one row where its restores are the same, and two where they differ
    across = decide(render("across", (40, 59), "05"))
    assert [(r["first"], r["last"], r["restore"]) for r in across["rows"]] == [(40, 49, []), (50, 59, ["b:whole"])], across["rows"]
    return ("a shot nobody may stand on is a `source` line with the newest render passed over; an unsure presence says so; two renders on one "
            "shot take different frames, the later only what is left, and what neither takes is a gap and no `source` line")


def restores() -> str:
    out = decide(render("a_face", (50, 99), "05"), render("b_hair", (50, 99), "05", subject="b", region="face and hair"),
                 render("a_alone", (0, 49), "05"), render("b_unsure", (100, 149), "05", subject="b", region="face and hair"))
    by = {r["name"]: r for r in out["rows"]}
    assert by["a_face"]["restore"] == ["b:whole"] and by["b_hair"]["restore"] == ["a"], (by["a_face"], by["b_hair"])
    assert by["a_alone"]["restore"] == [] and by["b_unsure"]["restore"] == [], "a subject alone in its shot was given a restore"
    # on shared frames the face row comes after the other
    shared = [r["name"] for r in out["rows"] if r["first"] == 50]
    assert shared == ["b_hair", "a_face"], shared
    return "every other subject in the shot is given back, `:whole` on a face row and bare on a face and hair row; none where the subject is alone"


def a_preference() -> str:
    kept = render("kept", (0, 49), "01", has=LIDS, owner=(("02", "choice", "keep this one"),))
    plain_old = render("older_frozen", (0, 49), "03")
    out = decide(kept, plain_old, prefers=["0-49:a=older_frozen:the lead's ruling on its strip"])
    stands = entry(out, 0, "a")["stands"]
    assert [s["render"] for s in stands] == ["older_frozen"] and stands[0]["why"] == "step 0: named by --prefer: the lead's ruling on its strip", stands
    for bad, words in ((["0-49:a=gone:why"], "the ledger has no render of"), (["3-9:a=kept:why"], "not a shot of the shots file"),
                       (["0-49:a=kept"], "is not `FIRST-LAST:SUBJECT=LOAD:the reason`"), (["50-99:a=kept:why"], "no render of that load covers those frames")):
        try:
            decide(kept, plain_old, prefers=bad)
        except SystemExit as e:
            assert words in str(e), (bad, str(e))
        else:
            raise AssertionError(f"--prefer {bad} was taken")
    try:
        decide(render("no_record", (0, 49), "09", making=NO_RECORD), prefers=["0-49:a=no_record:the lead wants it"])
    except SystemExit as e:
        assert "it never stands: no build record" in str(e), str(e)
    else:
        raise AssertionError("--prefer made a render with no build record stand")
    return "a preferred load goes first with its reason printed; one that never stands, a load the ledger lacks and frames that are no shot are refused"


def the_table_reads_back() -> str:
    import importlib.util
    spec = importlib.util.spec_from_file_location("assemble_delivery", Path(__file__).resolve().parent / "assemble_delivery.py")
    assert spec is not None and spec.loader is not None
    assembler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembler)
    out = decide(render("a_face", (50, 99), "05"), render("b_hair", (55, 99), "05", subject="b", region="face and hair"))
    text = dt.table_text(out, "/share", "Made for the check.", PLAN, "t_check")
    assert "--alias singer=a" in text and "_by" not in text and "#   50-54 b: no render that may stand covers these frames" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.txt"
        path.write_text(text)
        rows, intents = assembler.read_table(path), assembler.read_intents(path)
    assert [(r["first"], r["last"], r["piece"], r["piece_first"], r.get("subject"), r["restore"]) for r in rows] == [
        (50, 99, "/share/Video/s/a_face/a_face_00001.mp4", 50, "a", [("b", "whole")]),
        (55, 99, "/share/Video/s/b_hair/b_hair_00001.mp4", 55, "b", [("a", None)])], rows
    assert [(i["first"], i["last"], i["subject"]) for i in intents] == [(0, 49, "a"), (100, 149, "b")] and intents[0]["why"].startswith(dt.NO_SOURCE_YET), intents
    reading = dt.reading_text(out, "Made for the check.", "t.txt")
    assert "- STANDS on 50-99: **`a_face`**" in reading and "- NOTHING STANDS. The table's line: `source 0-49 subject=a" in reading, reading
    # a ledger of the schema before the fields this reads is refused, not read wrongly
    try:
        dt.decide({"schema": "h3_render_ledger/1", "renders": {}}, SHOTS, PLAN)
    except SystemExit as e:
        assert "make it again" in str(e)
    else:
        raise AssertionError("a ledger of an older schema was read")
    return "the table reads back through the assembler's two readers to the rows, restores and `source` lines decided; an older ledger is refused"


case("the owner's keep", the_owners_keep)
case("the frozen form", the_frozen_form)
case("what never stands", what_never_stands)
case("nothing left, and several renders on one shot", nothing_left_and_several_on_a_shot)
case("restore: the other subjects of the shot", restores)
case("a preference of the lead's", a_preference)
case("the table reads back through the assembler", the_table_reads_back)

if __name__ == "__main__":
    sys.exit(finish())
