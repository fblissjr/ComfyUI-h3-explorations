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
   shots file is unsure of says so; a subject the shots file says is NOT in a shot has a `source` line of its own
   kind, saying so and who said it (the assembler does not read the shots file), with a `#` in that name taken
   out, since the assembler's reader ends a line at one.
5. **Several renders on one shot** stand on different frames, each later one only on frames no earlier one
   took; frames no render takes inside a shot that has a piece are a comment and not a `source` line.
6. **`restore=`**: every other subject the shots file has in the shot, `:whole` on a face row and bare on a face
   and hair row; none where the subject is alone.
7b. **`--hold`** is its mirror (2026-10-13): the named render is not laid on that shot for that subject, and its
   reason is printed where it is passed over, in the table's header and in the reading, each time as a session's
   word and not the owner's. A shot it leaves with nothing standing says a render is HELD, not that none exists.
   An older render that may stand then does. Refused: a hold on a render the owner's last choice keeps; a hold
   and a preference on one render; frames that are no shot, a load the ledger lacks, a load that covers none of
   the shot, a hold with no reason. With no hold, nothing of it is in the table.
7. **`--prefer`** goes first among the renders that may stand, with its reason printed; naming a render that never
   stands, a load the ledger lacks, or frames that are no shot is refused.
8. **The table reads back** through the assembler's own two readers to the rows and `source` lines decided.
9. **A render whose last choice by the owner says it is not the one** does not stand by step 2 (the next newest
   frozen render is tried and the passed-over line says the choice was why) and cannot be preferred; `not sure`,
   a choice taken back, typed words and a choice said of a joined clip bar nothing.
10. **The commands are whole, each one line**: the ledger's own where the ledger keeps it (and said where it does
   not), this tool's own, the assembler's (the standing pieces' own captures from their build records, those
   handed in, an `--alias` only for an older name a listed capture uses, the caller's arguments, an argument
   with a space quoted) and the stack's build and check, made from the caller's `--source` and `--out`; not
   whole without those two, and the stack's not whole without its scratch folder.
11. **A row is tied to its capture's run from records**: the run of the capture its build record names that is
   named as the record's load and is of the row's subject; written on the row as `run=` and read back by the
   assembler's reader. No tie, with which lack, for a capture with no such run, a run of another subject, and a
   record that names no load.
12. **Each `restore=` has three counts** of its row's frames: a track, covered with an empty track, covered by no
   capture. They add up to the row, and the reading says the assembler flags only the third.

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
    {"first": 0, "last": 49, "who": {"a": {"present": "yes"}, "b": {"present": "no", "by": "a person, on a page # of 2026-01-01", "how": "words that are not carried"}}},
    {"first": 50, "last": 99, "who": {"a": {"present": "yes"}, "b": {"present": "yes"}}},
    {"first": 100, "last": 149, "who": {"a": {"present": "no"}, "b": {"present": "unsure"}}}]}
PLAN = {"subjects": {"a": {"region": "face", "parts": ["face_and_neck"]}, "b": {"region": "face and hair", "parts": ["hair", "face_and_neck"]}},
        "aliases": {"singer": "a", "_by": "a note"}}
FROZEN = {"cut_to_the_head": True, "face_points": True, "lid_marks": False}
PLAIN = {"cut_to_the_head": True, "face_points": False, "lid_marks": False}
LIDS = {"cut_to_the_head": True, "face_points": False, "lid_marks": True}
WHOLE_FRAME = {"cut_to_the_head": False, "face_points": True, "lid_marks": False}
AGREES, DISAGREES, NO_FILE, NO_RECORD = dt.AGREES, "recorded, THE FILE DISAGREES", "a build record for its load, WHOSE QUEUE RECORD NAMES NO FILE AND NO OUTCOME", "NO BUILD RECORD"


def render(name: str, span: tuple[int, int], written: str, *, subject: str = "a", region: str | None = "face", has=FROZEN, making: str = AGREES,
           owner: tuple = (), also: tuple = (), named: dict | None = None) -> dict:
    """A render as the ledger's JSON holds one, with the fields the rule reads. `owner` is (time, what, said[, on a joined clip])."""
    return {"file": f"Video/s/{name}/{name}_00001.mp4", "name": name, "written": f"2026-01-{written}T00:00:00Z", "span": list(span), "making": making,
            "subjects": [subject, *also], "records": [{"disagrees": ["the record's load is 9 frames from 1; the file holds 3"]}] if making == DISAGREES else [],
            "form": {"region": region, "part_named": named, "reference": {"has": has, "says": "what the ledger said of the reference"}},
            "owner": [{"at": f"2026-02-{o[0]}T00:00:00Z", "what": o[1], "said": o[2], "row": name, "from": f"verdicts (answers.jsonl:{10 + n})",
                       **({"on_the_joined_clip": o[3]} if len(o) > 3 else {})} for n, o in enumerate(owner)]}


def ledger(*renders: dict) -> dict:
    return {"schema": dt.LEDGER_SCHEMA, "share": "/share", "renders": {r["file"]: r for r in renders}}


def decide(*renders: dict, prefers=(), holds=()) -> dict:
    return dt.decide(ledger(*renders), SHOTS, PLAN, [dt.parse_prefer(p) for p in prefers], holds=[dt.parse_hold(h) for h in holds])


def entry(out: dict, first: int, subject: str) -> dict:
    return next(e for e in out["reading"] if (e["first"], e["subject"]) == (first, subject))


def the_owners_keep() -> str:
    kept = render("kept", (0, 49), "01", has=LIDS, owner=(("02", "choice", "keep this one"),))
    newer = render("newer_frozen", (0, 49), "09")
    e = entry(decide(kept, newer), 0, "a")
    assert [s["render"] for s in e["stands"]] == ["kept"] and e["stands"][0]["why"].startswith("step 1: the owner's keep of 2026-02-02"), e["stands"]
    assert [(p["render"], p["why"].startswith("may stand (step 2)")) for p in e["passed_over"]] == [("newer_frozen", True)], e["passed_over"]
    # the page's own positive button is a keep too, and of two kept renders on one shot the newest keep is first
    good = render("good_to_go", (0, 49), "02", has=LIDS, owner=(("06", "choice", "good to go"),))
    e = entry(decide(kept, good, newer), 0, "a")
    assert [s["render"] for s in e["stands"]] == ["good_to_go"] and "the owner's keep of 2026-02-06" in e["stands"][0]["why"], e["stands"]
    # a keep the owner changed afterwards, a keep said of a joined clip, and typed words are none of them a keep
    for name, owner in (("changed", (("02", "choice", "keep this one"), ("03", "choice", "wrong, and I say where"))),
                        ("of_a_joined_clip", (("02", "choice", "keep this one", "the joined clip"),)),
                        ("typed_words", (("02", "words", "keep this one"),))):
        e = entry(decide(render(name, (0, 49), "01", has=LIDS, owner=owner), newer), 0, "a")
        assert [s["render"] for s in e["stands"]] == ["newer_frozen"], (name, e["stands"])
        assert "has no keep and is not in the frozen form" in e["passed_over"][0]["why"], (name, e["passed_over"])
    # A FROZEN RENDER THE OWNER'S LAST CHOICE SAYS IS NOT THE ONE does not stand by step 2: the next newest is tried,
    # and the passed-over line says the choice was why. `not sure` bars nothing and is printed; a choice taken back
    # is no choice; a barring choice said of a joined clip is not on the render
    older = render("older_frozen", (0, 49), "03")
    for said in dt.NOT_THE_ONE:
        e = entry(decide(render("called_it", (0, 49), "09", owner=(("04", "choice", said),)), older), 0, "a")
        assert [s["render"] for s in e["stands"]] == ["older_frozen"], (said, e["stands"])
        assert f'the owner\'s last choice on it was "{said}"' in e["passed_over"][0]["why"] and "not the one" in e["passed_over"][0]["why"], e["passed_over"]
    for owner, last in (((("04", "choice", "not sure"),), "not sure"), ((("04", "choice", "wrong"), ("05", "choice", dt.TAKEN_BACK)), None),
                        ((("04", "choice", "wrong", "the joined clip"),), None), ((("04", "words", "wrong"),), None)):
        stands = entry(decide(render("newest", (0, 49), "09", owner=owner), older), 0, "a")["stands"][0]
        assert stands["render"] == "newest" and stands["why"].startswith("step 2"), (owner, stands)
        assert (stands["the_owners_last_choice"] or {}).get("said") == last, (owner, stands["the_owners_last_choice"])
    # with nothing else to try, a barred render leaves the shot as the source, and it is the one named as passed over
    alone = decide(render("called_it", (0, 49), "09", owner=(("04", "choice", "wrong"),)))
    assert entry(alone, 0, "a")["stands"] == [] and "the newest render passed over: called_it" in alone["sources"][0]["why"], alone["sources"][0]
    return ("the owner's keep on an agreeing render stands over a newer frozen one; a keep changed later, one said of a joined clip and typed "
            "words are not a keep; a frozen render whose last choice says it is not the one does not stand and the next newest is tried, while "
            "`not sure`, a choice taken back and one said of a joined clip bar nothing")


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


def a_part_the_job_named() -> str:
    """The hair alone where no face shows (the owner's answer of 2026-10-13): its capture's part reads `a part`, and the
    table laid the source there for want of the plan's region."""
    by = {"parts": ["hair"], "by": "the lead, on the owner's answer"}
    hair = render("b_hair", (60, 99), "09", subject="b", region="a part", named=by)
    e = entry(decide(hair), 50, "b")
    assert [s["render"] for s in e["stands"]] == ["b_hair"] and [tuple(f) for f in e["stands"][0]["frames"]] == [(60, 99)], e["stands"]
    why = e["stands"][0]["why"]
    assert why.startswith("step 2: a part of the plan's region (face and hair) named for this load by its job: hair alone, on the lead, on the owner's answer; the frozen form") , why
    # a part nobody named, a named part the plan does not have, and every part of the plan named are held to the plan's region as before
    for name, more in (("unnamed", {}), ("not_the_plans", {"named": {"parts": ["upper_clothing"], "by": "x"}}), ("all_of_them", {"named": {"parts": ["hair", "face_and_neck"], "by": "x"}})):
        got = entry(decide(render(name, (60, 99), "09", subject="b", region="a part", **more)), 50, "b")
        assert got["stands"] == [] and "it replaces a part, where the plan's region for this subject is face and hair" in got["passed_over"][0]["why"], (name, got)
    # a's plan has one part: nothing is some of it
    assert entry(decide(render("a_named", (0, 49), "09", region="a part", named={"parts": ["face_and_neck"], "by": "x"})), 0, "a")["stands"] == []
    # the named part still needs the reference of the form, and a whole person never stands
    assert entry(decide(render("b_hair_lids", (60, 99), "09", subject="b", region="a part", named=by, has=LIDS)), 50, "b")["stands"] == []
    assert entry(decide(render("b_whole", (60, 99), "09", subject="b", region="whole", named=by)), 50, "b")["stands"] == []
    # AFTER THE OWNER'S KEEP, not before it: the kept piece keeps the frame the two share, which a --prefer would have taken
    kept = render("b_kept", (50, 80), "02", subject="b", region="face and hair", owner=(("01", "choice", "keep this one"),))
    both = entry(decide(kept, render("b_hair", (80, 99), "09", subject="b", region="a part", named=by)), 50, "b")
    laid = {s["render"]: [tuple(f) for f in s["frames"]] for s in both["stands"]}
    assert laid == {"b_kept": [(50, 80)], "b_hair": [(81, 99)]}, laid
    return "a render whose job named its part as some of the plan's stands at step 2 as the plan's region, after a keep, and its row says which part and on whose word"


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
    # a subject the shots file says is not in a shot: a line of its own kind, who said so and never what they said
    assert [(n["first"], n["subject"], n["why"]) for n in out["not_in_the_shot"]] == [
        (0, "b", dt.NOT_IN_THE_SHOT + " (a person, on a page of 2026-01-01)"), (100, "a", dt.NOT_IN_THE_SHOT + " (it does not say who said so)")], out["not_in_the_shot"]
    assert not any((e["first"], e["subject"]) in ((0, "b"), (100, "a")) for e in out["reading"]), "a subject not in a shot was looked for"
    # two renders on one shot: the kept one takes its frames, the newer frozen one only what is left, and the rest is a gap
    kept = render("kept", (60, 79), "01", has=LIDS, owner=(("02", "choice", "keep this one"),))
    wide = render("wide", (55, 89), "09")
    out = decide(kept, wide)
    rows = [(r["first"], r["last"], r["name"], r["piece_first"]) for r in out["rows"]]
    assert rows == [(55, 59, "wide", 55), (60, 79, "kept", 60), (80, 89, "wide", 55)], rows
    assert [(g["first"], g["last"], g["subject"]) for g in out["gaps"]] == [(50, 54, "a"), (90, 99, "a")], out["gaps"]
    assert not any(s["subject"] == "a" and s["first"] == 50 for s in out["sources"]), "a shot with a piece was given a source line"
    assert entry(out, 50, "a")["left_as_the_source"] == [(50, 54), (90, 99)]
    # a span of the shots file: the shots inside it and no other, and a span that cuts a shot is refused
    part = dt.decide(ledger(kept, wide), SHOTS, PLAN, span=(50, 149))
    assert {e["first"] for e in part["reading"]} == {50, 100} and all(s["first"] >= 50 for s in part["sources"]), part["reading"]
    try:
        dt.decide(ledger(), SHOTS, PLAN, span=(0, 60))
    except SystemExit as e:
        assert "cuts the shot 50-99" in str(e), str(e)
    else:
        raise AssertionError("a span that cuts a shot was taken")
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


def the_owners_words_are_in_the_reading() -> str:
    """The owner asked (2026-10-13) that their typed words are read with their clicks: the reading prints them, whole,
    under the render they are on, whether it stands or is passed over. They decide nothing."""
    kept = render("kept", (0, 49), "09", owner=(("02", "words", "love this. great.\n\nthe other one is not replaced though, fyi."), ("03", "choice", "keep this one"),
                                                 ("04", "words", "SAID OF A JOINED CLIP", "the joined clip")))
    kept["owner"][0]["frame"] = 15
    older = render("older", (0, 49), "03", owner=(("05", "choice", "not sure"), ("06", "words", "these two look similar?")))
    out = decide(kept, older)
    e = entry(out, 0, "a")
    assert [s["render"] for s in e["stands"]] == ["kept"] and [w["said"] for w in e["stands"][0]["the_owners_words"]] == ["love this. great.\n\nthe other one is not replaced though, fyi."]
    assert [w["said"] for w in e["passed_over"][0]["the_owners_words"]] == ["these two look similar?"], e["passed_over"][0]
    text = dt.reading_text(out, "MADE", "t.txt")
    stands_at = text.index("- STANDS on 0-49: **`kept`**")
    words_at = text.index("  - the owner's words on it, 2026-02-02T00:00 at frame 15: love this. great. /  / the other one is not replaced though, fyi.  [verdicts (answers.jsonl:10), row kept]")
    passed_at = text.index("- passed over: `older`")
    assert stands_at < words_at < passed_at, "the words are not under the render they are on"
    assert "  - the owner's words on it, 2026-02-06T00:00: these two look similar?  [verdicts (answers.jsonl:11), row older]" in text[passed_at:], text[passed_at:passed_at + 600]
    assert "SAID OF A JOINED CLIP" not in text, "words said of a joined clip were printed as words on the render"
    # the words decide nothing: the same renders with no words stand and are passed over the same
    bare = decide(render("kept", (0, 49), "09", owner=(("03", "choice", "keep this one"),)), render("older", (0, 49), "03", owner=(("05", "choice", "not sure"),)))
    assert [(s["render"], s["why"]) for s in entry(bare, 0, "a")["stands"]] == [(s["render"], s["why"]) for s in e["stands"]]
    assert "the owner's words on it" not in dt.reading_text(bare, "MADE", "t.txt")
    return "the owner's typed words on a render are printed whole under it in the reading, with when, the frame and where the line is; words of a joined clip are not; they decide nothing"


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
    # the owner's choice is over the lead's preference
    try:
        decide(render("called_wrong", (0, 49), "09", owner=(("04", "choice", "wrong"),)), prefers=["0-49:a=called_wrong:the lead wants it"])
    except SystemExit as e:
        assert 'the owner\'s last choice on it was "wrong"' in str(e) and "over the lead's preference" in str(e), str(e)
    else:
        raise AssertionError("--prefer made a render the owner called wrong stand")
    return "a preferred load goes first with its reason printed; one that never stands, a load the ledger lacks and frames that are no shot are refused"


def a_hold_of_the_leads() -> str:
    """A render the lead says is not to be laid, before the owner has judged it (2026-10-13: a render read as not one
    face across its shot was the only render of that shot, and stood by the rule)."""
    only = render("only_one", (0, 49), "09")
    why = "a blind reader's reading, reading.md sections 4 and 8: it is not one face across the shot; the owner has not judged it"
    out = decide(only, holds=[f"0-49:a={only['name']}:{why}"])
    e = entry(out, 0, "a")
    assert e["stands"] == [] and not any(r["name"] == "only_one" for r in out["rows"]), "a held render was laid"
    over = e["passed_over"][0]
    assert over["render"] == "only_one" and over["held"] == why and over["why"].startswith(f"HELD BACK by the lead (--hold), a session's word and not the owner's: {why}") \
        and "Without the hold: it may stand (step 2)" in over["why"], over
    # the shot's line says a render is held, not that none exists
    assert e["source_line"] == f"{dt.HELD_BACK} (only_one)" and dt.NO_SOURCE_YET not in e["source_line"], e["source_line"]
    assert out["held"] == [{"first": 0, "last": 49, "subject": "a", "load": "only_one", "reason": why}], out["held"]
    text = dt.table_text(out, "/share", "MADE", PLAN, "t")
    assert f"# HELD BACK by the lead (--hold), a session's word and not the owner's: only_one on 0-49, a: {why}" in text and "only_one_00001.mp4" not in text.split("# HELD BACK")[1].split("\n", 1)[1], text
    assert "HELD BACK by the lead" in dt.reading_text(out, "MADE", "t.txt") and why in dt.reading_text(out, "MADE", "t.txt")
    # with an older render that may stand, the hold of the newer lets the older stand; with none held, the newer does
    older = render("older", (0, 49), "03")
    assert [s["render"] for s in entry(decide(only, older), 0, "a")["stands"]] == ["only_one"]
    stands = entry(decide(only, older, holds=[f"0-49:a=only_one:{why}"]), 0, "a")["stands"]
    assert [s["render"] for s in stands] == ["older"], stands
    # a hold on a render that would not have stood says what it would have been, and the shot's line stays the rule's own
    lids = render("with_lids", (0, 49), "09", has=LIDS)
    e = entry(decide(lids, holds=["0-49:a=with_lids:the lead's reason"]), 0, "a")
    assert "Without the hold: it is not in the frozen form" in e["passed_over"][0]["why"] and e["source_line"].startswith(dt.NO_SOURCE_YET), e
    # THE OWNER'S WORD IS OVER THE LEAD'S: a hold on a render the owner kept is refused; and so are the malformed ones
    kept = render("kept", (0, 49), "09", owner=(("04", "choice", "keep this one"),))
    for renders, kwargs, words in (((kept,), dict(holds=["0-49:a=kept:the lead does not like it"]), "a keep. The owner's word is over the lead's hold"),
                                   ((only,), dict(holds=["0-49:a=only_one:r"], prefers=["0-49:a=only_one:r"]), "put first or held back, not both"),
                                   ((only,), dict(holds=["0-48:a=only_one:r"]), "--hold names frames 0-48, which is not a shot"),
                                   ((only,), dict(holds=["0-49:a=nobody:r"]), "--hold names the load 'nobody', which the ledger has no render of"),
                                   ((only, render("elsewhere", (50, 99), "02")), dict(holds=["0-49:a=elsewhere:r"]), "no render of that load covers those frames"),
                                   ((only,), dict(holds=["0-49:a=only_one:"]), "--hold '0-49:a=only_one:' is not")):
        try:
            decide(*renders, **kwargs)
        except SystemExit as stop:
            assert words in str(stop), f"refused without saying {words!r}: {stop}"
        else:
            raise AssertionError(f"not refused: {kwargs}")
    # no hold: nothing of it in the table or the data
    assert decide(only)["held"] == [] and "HELD BACK" not in dt.table_text(decide(only), "/share", "MADE", PLAN, "t")
    return ("a held render is not laid and its reason is printed where it is passed over, in the table's header and in the reading, as a session's word; the "
            "shot's line says a render is held; an older render then stands; a hold on the owner's keep, with a preference, or on no such shot or load is refused")


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
    assert [(i["first"], i["last"], i["subject"]) for i in intents] == [(0, 49, "a"), (100, 149, "b"), (0, 49, "b"), (100, 149, "a")] \
        and intents[0]["why"].startswith(dt.NO_SOURCE_YET), intents
    # the absent subject's line reads back whole: the `#` in who said so did not end it
    assert intents[2]["why"] == dt.NOT_IN_THE_SHOT + " (a person, on a page of 2026-01-01)", intents[2]
    assert not any("run" in r for r in rows), "a row with no tie handed in was given one"
    reading = dt.reading_text(out, "Made for the check.", "t.txt")
    assert "- STANDS on 50-99: **`a_face`**" in reading and "- NOTHING STANDS. The table's line: `source 0-49 subject=a" in reading, reading
    assert "## Subjects the shots file says are not in a shot" in reading and "- `source 0-49 subject=b`: " + dt.NOT_IN_THE_SHOT in reading, reading
    # a ledger of the schema before the fields this reads is refused, not read wrongly
    try:
        dt.decide({"schema": "h3_render_ledger/1", "renders": {}}, SHOTS, PLAN)
    except SystemExit as e:
        assert "make it again" in str(e)
    else:
        raise AssertionError("a ledger of an older schema was read")
    return "the table reads back through the assembler's two readers to the rows, restores and `source` lines decided; an older ledger is refused"


def _assembler():
    import importlib.util
    spec = importlib.util.spec_from_file_location("assemble_delivery", Path(__file__).resolve().parent / "assemble_delivery.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture(repo: Path, name: str, first: int, frames: int, tracks: dict, runs=(), not_covered: dict | None = None) -> str:
    """A capture folder as the capture tool leaves one: a manifest with its runs, and each subject's packed track
    with the frames it covers (every frame unless `not_covered[label]` names some)."""
    import json

    import numpy as np
    folder = repo / "data" / name
    for label, on in tracks.items():
        (folder / "subjects" / label).mkdir(parents=True)
        track = np.zeros((frames, 4, 8), bool)
        track[[f - first for f in on]] = True
        covered = np.ones(frames, bool)
        covered[[f - first for f in (not_covered or {}).get(label, ())]] = False
        np.savez_compressed(folder / "subjects" / label / "masks__p.npz", track=np.packbits(track, axis=-1), covered=covered)
    (folder / "manifest.json").write_text(json.dumps({"first_frame": first, "frames": frames, "runs": list(runs), "subjects": [
        {"label": label, "sightings": [{"by": "p"}]} for label in tracks]}))
    return str(folder)


def the_commands() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        face = render("a_face", (50, 99), "05")
        hair = render("b_hair", (50, 99), "05", subject="b", region="face and hair")
        face["records"] = [{"capture": _capture(repo, "cap_a", 40, 70, {"a": range(40, 110), "b": range(40, 75)})}]
        hair["records"] = [{"capture": _capture(repo, "cap_b", 50, 50, {"b": range(50, 100), "singer": range(50, 100)})}]
        made = {**ledger(face, hair), "command": {"line": "py bench/render_ledger.py --share /share"}}
        out = dt.decide(made, SHOTS, PLAN)
        built = dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149), "--source '/in/a clip.mp4' --out /out/clip.mp4 --size source", (str(repo / "data" / "preview"),),
                            python="py", own="py bench/delivery_table.py --ledger L", stack_work="/scratch/work")
        # in the table's own row order (b's row is before a's face row on the frames they share), then those handed in
        assert built["captures"] == ["data/cap_b", "data/cap_a", "data/preview"] and built["captures_handed_in"] == ["data/preview"], built["captures"]
        assert built["whole"] and built["aliases_used_by_a_capture"] == ["singer"], built
        # one line, with the caller's argument that holds a space quoted again
        assert built["command"] == ("py bench/assemble_delivery.py --table data/t.txt --span 0-149 --source '/in/a clip.mp4' --out /out/clip.mp4 --size source "
                                    "--capture data/cap_b --capture data/cap_a --capture data/preview --alias singer=a"), built["command"]
        with_record = dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149), "--source '/in/a clip.mp4' --out /out/clip.mp4", (), python="py", shots_file="data/v/shots.json")
        assert " --out /out/clip.mp4 --shots data/v/shots.json --capture " in with_record["command"] and "--shots" not in built["command"], with_record["command"]
        assert built["stack"] == {"build": "py bench/stack_delivery.py build --source '/in/a clip.mp4' --table data/t.txt --joined /out/clip.mp4 "
                                           "--out /out/clip_STACKED.mp4 --work /scratch/work",
                                  "check": "py bench/stack_delivery.py check --source '/in/a clip.mp4' --joined /out/clip.mp4 --out /out/clip_STACKED.mp4",
                                  "whole": True, "out": "/out/clip_STACKED.mp4"}, built["stack"]
        assert built["ledger_command"] == "py bench/render_ledger.py --share /share" and built["own_command"] == "py bench/delivery_table.py --ledger L"
        text = dt.reading_text(out, "Made for the check.", "data/t.txt", built)
        for line in (built["ledger_command"], built["own_command"], built["command"], built["stack"]["build"], built["stack"]["check"]):
            assert f"\n{line}\n" in text, f"the reading does not hold this command as one line: {line}"
        assert "NOT WHOLE" not in text, text
        # with nothing of the caller's: not whole, no stack; a ledger that does not keep its command is said; no scratch folder is said
        bare = dt.assembly(out, ledger(face, hair), PLAN, repo, "data/t.txt", (0, 149))
        assert not bare["whole"] and bare["stack"] is None and bare["ledger_command"] is None, bare
        text = dt.reading_text(out, "Made for the check.", "data/t.txt", bare)
        assert "NOT WHOLE: no `--source` and `--out`" in text and "this ledger does not keep its command" in text, text
        no_work = dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149), "--source /in/clip.mp4 --out /out/clip.mp4")
        assert no_work["stack"]["whole"] is False and "NOT WHOLE: no `--stack-work`" in dt.reading_text(out, "m", "data/t.txt", no_work), no_work["stack"]
    return ("the ledger's, the table's, the clip's and the stack's commands are each one line, an argument with a space quoted; not whole without a source "
            "and an out, or without the stack's scratch folder; a ledger that keeps no command is said")


def a_row_is_tied_to_its_run() -> str:
    assembler = _assembler()
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        face = render("a_face", (50, 99), "05")
        hair = render("b_hair", (50, 99), "05", subject="b", region="face and hair")
        planned = lambda name, subject: {"name": name, "planned": True, "render": None, "subject": subject}   # noqa: E731
        # a's capture holds the run its record's load names, under the plan's OLDER name for a, beside another subject's run
        face["records"] = [{"capture": _capture(repo, "cap_a", 50, 50, {"singer": range(50, 100)}, [planned("other", "b"), planned("a_face", "singer")]), "load": {"name": "a_face"}}]
        hair["records"] = [{"capture": _capture(repo, "cap_b", 50, 50, {"b": range(50, 100)}, [planned("not_this_load", "b")]), "load": {"name": "b_hair"}}]
        made = ledger(face, hair)
        out = dt.decide(made, SHOTS, PLAN)
        built = dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))
        ties = {t["row"]: t for t in built["runs"]}
        assert ties["50-99 a_face (a)"]["run"] == "data/cap_a:a_face", ties
        assert ties["50-99 b_hair (b)"]["run"] is None and "holds no run named b_hair" in ties["50-99 b_hair (b)"]["why_not"], ties
        path = repo / "t.txt"
        path.write_text(dt.table_text(out, "/share", "Made for the check.", PLAN, "t_check"))
        rows = {r["subject"]: r for r in assembler.read_table(path)}
        assert rows["a"]["run"] == ("data/cap_a", "a_face") and "run" not in rows["b"], rows
        text = dt.reading_text(out, "m", "data/t.txt", built)
        assert "1 of 2 rows are tied" in text and "- `50-99 b_hair (b)`: the capture its build record names holds no run named b_hair" in text, text
        # the lacks, each by name: a run of another subject, a record with no load, no record at all
        hair["records"] = [{"capture": _capture(repo, "cap_b2", 50, 50, {"b": range(50, 100)}, [planned("b_hair", "a")]), "load": {"name": "b_hair"}}]
        face["records"] = [{"capture": str(repo / "data" / "cap_a")}]
        why = {t["row"]: t["why_not"] for t in dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))["runs"]}
        assert "is of a, not of this subject" in why["50-99 b_hair (b)"] and why["50-99 a_face (a)"] == "its build record names no load", why
        assert not any("run" in r for r in out["rows"]), "a tie of an earlier pass was left on a row"
        hair["records"] = []
        assert {t["row"]: t["why_not"] for t in dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))["runs"]}["50-99 b_hair (b)"] == "the ledger has no build record of it"
    return ("a row is tied to the run its build record's load names in the capture that record names, of its own subject under any of its names, and the "
            "assembler's reader takes it; three lacks are each said and leave the row untied")


def what_a_restore_has_to_go_on() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        face = render("a_face", (50, 99), "05")
        hair = render("b_hair", (50, 99), "05", subject="b", region="face and hair")
        # a's capture is of a longer stretch than the row. It holds b on the row's first 25 frames (and on frames before
        # the row, which are not the row's), covers the next 15 with an empty track, and does not cover the last 10
        face["records"] = [{"capture": _capture(repo, "cap_a", 40, 70, {"a": range(40, 110), "b": range(40, 75)}, not_covered={"b": range(90, 110)})}]
        hair["records"] = [{"capture": _capture(repo, "cap_b", 50, 50, {"b": range(50, 100), "singer": range(50, 100)})}]
        made = ledger(face, hair)
        out = dt.decide(made, SHOTS, PLAN)
        counts = lambda x: (x["frames_with_a_track"], x["frames_covered_with_an_empty_track"], x["frames_no_capture_covers"])   # noqa: E731
        by = {(x["row"], x["restore"]): x for x in dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))["restores"]}
        # b given back on a's face row: tracked on every frame by cap_b; a given back on b's row: by cap_a, and by cap_b under its older name
        assert counts(by[("50-99 a_face (a)", "b:whole")]) == (50, 0, 0) and by[("50-99 a_face (a)", "b:whole")]["in_the_captures"] == ["data/cap_b", "data/cap_a"], by
        assert by[("50-99 b_hair (b)", "a")]["in_the_captures"] == ["data/cap_b", "data/cap_a"], by
        # with b's own capture gone from the list the three counts are a's capture's alone, and they add up to the row
        hair["records"] = []
        half = {x["restore"]: x for x in dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))["restores"]}
        assert counts(half["b:whole"]) == (25, 15, 10) and sum(counts(half["b:whole"])) == half["b:whole"]["frames"] == 50, half["b:whole"]
        assert counts(half["a"]) == (50, 0, 0), half["a"]
        # a capture that covers the row and never saw b: every frame covered by no capture, and it is said
        face["records"] = [{"capture": _capture(repo, "cap_c", 50, 50, {"a": range(50, 100)})}]
        none = dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))
        nobody = next(x for x in none["restores"] if x["restore"] == "b:whole")
        assert (counts(nobody), nobody["in_the_captures"]) == ((0, 0, 50), []), nobody
        # one that covers every frame of the row with b's track empty: nothing to give back, and by the assembler's rule no flag
        face["records"] = [{"capture": _capture(repo, "cap_d", 50, 50, {"a": range(50, 100), "b": ()})}]
        empty = next(x for x in dt.assembly(out, made, PLAN, repo, "data/t.txt", (0, 149))["restores"] if x["restore"] == "b:whole")
        assert counts(empty) == (0, 50, 0), empty
        text = dt.reading_text(out, "Made for the check.", "data/t.txt", none)
        assert "NO listed capture holds that subject's track on these frames" in text and "covered by no capture on 50" in text \
            and "the assembler raises no flag" in text and "`restore_has_no_mask`" in text, text
    return ("each restore says on how many of its row's frames a listed capture holds the track, covers the frame with an empty one, or does not cover "
            "it; the three add up to the row; the reading says the assembler flags only the last")


case("the owner's keep", the_owners_keep)
case("the frozen form", the_frozen_form)
case("a part the job named stands as the plan's region", a_part_the_job_named)
case("what never stands", what_never_stands)
case("nothing left, and several renders on one shot", nothing_left_and_several_on_a_shot)
case("restore: the other subjects of the shot", restores)
case("the owner's words, under the render they are on", the_owners_words_are_in_the_reading)
case("a preference of the lead's", a_preference)
case("a hold of the lead's: not this one, and why", a_hold_of_the_leads)
case("the table reads back through the assembler", the_table_reads_back)
case("the commands, each whole and one line", the_commands)
case("a row is tied to its capture's run, from records", a_row_is_tied_to_its_run)
case("what a restore has to go on, as three counts", what_a_restore_has_to_go_on)

if __name__ == "__main__":
    sys.exit(finish())
