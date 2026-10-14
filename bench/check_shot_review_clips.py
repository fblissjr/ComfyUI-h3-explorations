#!/usr/bin/env python3
"""`bench/shot_review_clips.py` on made tables, records and lists: each case is a way a shot's review clip could say the wrong day, lose a piece or a subject, hide a session's word, or write over what stands.

  the_words_of_an_item              every standing piece with its frames, the day and hour on the owner's clock (03:10Z is 22:10 the day
                                    before in Central), the owner's keep (either word the table counts as one), last choice or none, the
                                    frames its load left as the original or NOT KNOWN; a subject unsure with and without a piece, said once;
                                    one the table lists as not in the shot; one with nothing standing, its whole shot not said twice, with
                                    how many were passed over and the newest named, a long reason cut and said so
  the_leads_word                    a piece that stands by the lead's --prefer says so with the reason whole; the owner's keep it set aside
                                    is named; a render held back (--hold) is named with its reason whole
  a_span_of_several_shots           every sentence begins with its shot, in frame order, and a subject in one shot and not the next is not
                                    said to be both on one span
  the_item_itself                   the buttons are the table's own, the rate the record's, who put it on the list, a note whole; an empty
                                    table says no piece is laid
  the_table_is_handed_what_it_needs the table's command carries the span, the source, the clip's name, the ledger, shots and plan, and every
                                    handed flag
  build_writes_over_nothing         a kept output or a span given twice is refused before anything runs; a capture that holds an earlier
                                    record of the name is refused after the table; a stack with a hard fault (a frame count off the span,
                                    even with both counts equal; sound that is not the clip's or none) stops the build
  items_at_the_top_and_a_copy       the items at the top with a copy first; dry, a stray note, two notes, a span twice, an unbuilt span, a
                                    listed id, a failing record, a stack fault and a media link to another build are refused
  by_the_day                        a row a piece (one render on two shots is one row) with the table's own words, the lead's --prefer said as
                                    the lead's; the day's renders this table does not lay, never one that stands in it; the ledger must be the one
                                    the table names; a kept output stays

    <comfy venv python> bench/check_shot_review_clips.py

The standard library only; no model, no card, no video. `build`'s steps are stood in for where it runs them (the
assembler's and the stack's are held by their own checks). What this cannot see: a build on a real ledger (its runs
are throwaways, read by someone who did not write them).
"""
from __future__ import annotations

import builtins
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table as dt  # noqa: E402
import shot_review_clips as src  # noqa: E402


def stand(render, frames, written, choice=None, left=(), why="step 2: the plan's region; its file agrees with its build record", step=2, led=None):
    return {"render": render, "frames": [list(f) for f in frames], "written": written, "why": why, "step": step, "the_leads_prefer": led,
            "leaves_as_the_original": None if left is None else [list(x) for x in left], "the_owners_last_choice": choice}


def passed(render, written, why, choice=None, held=None):
    return {"render": render, "written": written, "why": why, **({"held": held} if held else {}), "the_owners_last_choice": choice}


def entry(first, last, subject, present="yes", stands=(), passed_over=(), left=None):
    # as bench/delivery_table.decide makes one: with nothing standing, the whole shot is left as the source
    return {"first": first, "last": last, "subject": subject, "present": present, "stands": list(stands), "passed_over": list(passed_over),
            "left_as_the_source": left if left is not None else ([] if stands else [[first, last]])}


LONG = "never stands: no build record: its making cannot be said, and the words go on well past the hundred and ten characters an item shows of one"


def made_table() -> dict:
    return {"reading": [
        entry(100, 140, "lead", stands=[stand("lead_a", [(100, 120)], "2026-10-14T03:10:00Z", {"said": dt.KEEP[0], "at": "2026-10-14T04:00:00Z"}, step=1,
                                              why="step 1: the owner's keep"),
                                        stand("lead_b", [(121, 140)], "2026-10-12T15:00:00Z", None, [[125, 127]])]),
        entry(100, 140, "second", stands=[stand("second_a", [(100, 129)], "2026-10-13T18:00:00Z", {"said": dt.NOT_SURE, "at": "2026-10-13T20:00:00Z"})],
              left=[[130, 140]]),
        entry(100, 140, "third", passed_over=[passed("t_new", "2026-10-13T12:00:00Z", LONG), passed("t_mid", "2026-10-12T12:00:00Z", "short"),
                                              passed("t_old", "2026-10-11T12:00:00Z", "short"), passed("t_older", "2026-10-10T12:00:00Z", "short")]),
        entry(100, 140, "fourth", present="unsure"),
        entry(100, 140, "sixth", present="unsure", stands=[stand("sixth_a", [(100, 130)], "2026-10-13T18:00:00Z", {"said": dt.KEEP[1], "at": "2026-10-13T19:00:00Z"},
                                                                 None, step=1, why="step 1: the owner's keep")], left=[[131, 140]])],
        "not_in_the_shot": [{"first": 100, "last": 140, "subject": "fifth", "why": "the shots file says this subject is not in the shot (owner, on the who page)"}]}


RECORD = {"verdict": "passes", "verdict_line": "passes; every named subject has something laid on every shot", "video": {"rate": "24000/1001"}}


def reading(frames: int, same: bool = True, clip_frames: int | None = None, sound: bool = True) -> dict:
    """A stack's reading as `stack_delivery.read_back` returns one."""
    row = {"frames": frames, "own_frame_fits_best": frames, "ties_a_neighbour": 0, "a_neighbour_fits_better": [], "mean_to_its_own_frame": 0.7,
           "greatest_to_its_own_frame": 0.8, "mean_to_the_nearer_neighbour": 9.0}
    return {"frames": frames, "clip_frames": frames if clip_frames is None else clip_frames, "size": [1024, 2464], "canvas": [1024, 768],
            "sound_packets": {"stacked": "MD5=a" if sound else "", "clip": ("MD5=a" if same else "MD5=b") if sound else "", "the_same": same and sound},
            "the_clip_begins_on_source_frame": 0, "the_original_row": row, "the_clip_row": row}


def refused(fn, *words):
    try:
        fn()
    except SystemExit as stop:
        assert all(w in str(stop) for w in words), (str(stop), words)
        return
    raise AssertionError(f"not refused: {words}")


def words_of_an_item() -> str:
    pieces, said = src.item_words(made_table())
    text = ". ".join(said)
    assert pieces == 4, pieces
    assert "lead: lead_a on 100-120, rendered 10/13 (22:10 Central); your keep of 10/13" in text, text
    assert "lead: lead_b on 121-140, rendered 10/12 (10:00 Central); not judged by you yet; inside it the piece leaves the original on 125-127" in text, text
    assert f"second: second_a on 100-129, rendered 10/13 (13:00 Central); your last choice on it: \"{dt.NOT_SURE}\"." in text \
        and "second: the original on 130-140, where no piece of theirs is laid" in text, text
    assert "fourth: the shots record is unsure they are in this shot (yours to answer on the who page); the original throughout" in text, text
    assert "sixth: the shots record is unsure they are in this shot (yours to answer on the who page). sixth: sixth_a on 100-130" in text \
        and "sixth: the original on 131-140" in text and text.count("the original throughout") == 1, text
    # the table's second word for a keep is a keep too, and a ledger that does not say what a piece leaves is NOT KNOWN
    assert "sixth_a on 100-130, rendered 10/13 (13:00 Central); your keep of 10/13; what it leaves as the original inside it is NOT KNOWN" in text, text
    assert "fifth: not in this shot by the shots record (the shots file says" in text, text
    # nothing stands: the whole shot once (the table also lists it as left as the source), the newest three named, a long reason cut
    assert "third: the original on the whole shot: no piece of theirs stands here (4 other render(s) passed over, the newest 3: t_new, " + LONG[:src.REASON_CHARS] + "...; t_mid, short; t_old, short)" in text, text
    assert "third: the original on 100-140" not in text and "t_older" not in text, text
    return ("four pieces each with its day on the owner's clock, either keep word, choice or none, frames left or NOT KNOWN; unsure with and without a piece, "
            "said once; not in the shot from the table's list; nothing standing said once, with the count and the newest three, a long reason cut")


def the_leads_word() -> str:
    reason = "the remake of today replaces her piece of 10/12 on the lead's ruling, which is a reason far longer than an item's cut of a passed-over reason"
    t = {"reading": [entry(100, 140, "lead", stands=[stand("lead_new", [(100, 140)], "2026-10-14T15:00:00Z", None, step=0, why=f"step 0: named by --prefer: {reason}",
                                                           led=reason)],
                           passed_over=[passed("lead_kept", "2026-10-12T15:00:00Z", "may stand (step 1) and every frame of it on this shot is taken by a render before it",
                                               {"said": dt.KEEP[0], "at": "2026-10-12T20:00:00Z"}),
                                        passed("lead_held", "2026-10-13T15:00:00Z", "HELD BACK ...", held="its apron is a white top on all 50 frames, a reader's word")])],
         "not_in_the_shot": []}
    text = ". ".join(src.item_words(t)[1])
    assert f"lead_new on 100-140, rendered 10/14 (10:00 Central); not judged by you yet; IT STANDS BY THE LEAD'S CHOICE (--prefer), a session's word and not yours: {reason}" in text, text
    assert "lead: YOUR KEEP lead_kept (of 10/12) is not laid here: may stand (step 1)" in text, text
    assert "lead: lead_held is HELD BACK by the lead (--hold), a session's word and not yours: its apron is a white top on all 50 frames, a reader's word" in text, text
    return "the lead's --prefer said as the lead's choice with its reason whole; the owner's keep it set aside named; a hold named with its reason whole"


def a_span_of_several_shots() -> str:
    t = {"reading": [entry(100, 140, "lead", stands=[stand("lead_a", [(100, 140)], "2026-10-13T15:00:00Z")]),
                     entry(141, 160, "second", stands=[stand("second_b", [(141, 160)], "2026-10-13T16:00:00Z")]), entry(100, 140, "second", present="unsure")],
         "not_in_the_shot": [{"first": 141, "last": 160, "subject": "lead", "why": "the shots file says this subject is not in the shot"}]}
    said = src.item_words(t)[1]
    assert said[0].startswith("on 100-140, lead: lead_a on 100-140") and said[-1].startswith("on 141-160, ") and all(s.startswith("on 1") for s in said), said
    assert "on 141-160, lead: not in this shot by the shots record" in said[-2:] or "on 141-160, lead: not in this shot by the shots record (the shots file says this subject is not in the shot)" in said, said
    assert [int(s.split("-")[0][3:]) for s in said] == sorted(int(s.split("-")[0][3:]) for s in said), said
    item = src.make_item("v_0100_0160_joined_T", (100, 160), t, RECORD, "m/x.mp4", "T", "tester", None)
    assert "the 2 shots of 100-160" in item["title"] and item["what"].startswith("The 2 shots of 100-160"), item
    one = src.item_words(made_table())[1]
    assert not any(s.startswith("on ") for s in one), "a span of one shot names its shot on every sentence"
    return "on a span of two shots every sentence begins with its shot, in frame order, and the title says two shots; one shot's span names none"


def the_item_itself() -> str:
    item = src.make_item("fun_0100_0140_joined_T", (100, 140), made_table(), RECORD, "data/fun/review/media/fun_0100_0140_joined_T_STACKED.mp4",
                         "T", "tester", "A person's words, whole.")
    assert item["options"] == list(dt.RENDER_ITEM_BUTTONS) and item["fps"] == "23.976" and item["frames"] == "41" and item["first"] == "100", item
    assert item["what"].endswith(" Words of tester, who put this item on the list: A person's words, whole.") and "from the build of T, put on this list by tester" in item["what"] \
        and "4 piece(s) laid" in item["what"] and item["title"].startswith("fun_0100_0140_joined_T_stacked: the shot 100-140 "), item
    empty = src.make_item("fun_0100_0140_joined_T", (100, 140), {"reading": [], "not_in_the_shot": []}, RECORD, "m/x.mp4", "T", "tester", None)
    assert "No piece is laid: ." not in empty["what"] and "No piece is laid" in empty["what"], empty["what"]
    return "the table's buttons, the record's rate, who put it on the list, the note at the end; an empty table says no piece is laid"


def the_table_is_handed_what_it_needs() -> str:
    a = src.main.__globals__["argparse"].Namespace(ledger="L.json", shots="S.json", plan="P.json", source="/in/fun.mp4", also_capture=["c1", "c2"],
                                                   prefer=["1-2:lead=x:why"], hold=["3-4:second=y:why"])
    f = src.files_of("fun_0001_0009_joined_T", Path("/clips"), Path("/tables"))
    cmd = src.table_command(a, (1, 9), f, Path("/work/w"))
    given = cmd[cmd.index("--assemble-args") + 1]
    assert cmd[cmd.index("--span") + 1] == "1-9" and given == "--source /in/fun.mp4 --size source --out /clips/fun_0001_0009_joined_T.mp4", (cmd, given)
    assert cmd[cmd.index("--stack-work") + 1] == "/work/w" and cmd[cmd.index("--json") + 1] == "/tables/t_fun_0001_0009_joined_T.json", cmd
    assert (cmd[cmd.index("--ledger") + 1], cmd[cmd.index("--shots") + 1], cmd[cmd.index("--plan") + 1]) == ("L.json", "S.json", "P.json"), cmd
    assert [cmd[k + 1] for k, w in enumerate(cmd) if w == "--also-capture"] == ["c1", "c2"] and "1-2:lead=x:why" in cmd and "3-4:second=y:why" in cmd, cmd
    return "span, source, clip name, scratch folder, ledger, shots, plan and every handed flag reach the table"


class Steps:
    """`build`'s subprocesses stood in for: the table writes its files and names a capture; the two steps write the clip, its record and the stack."""

    def __init__(self, capture: Path, got: dict):
        self.capture, self.got, self.ran, self.clip = capture, got, [], Path()

    def run(self, argv, **_):
        argv = [str(x) for x in argv]
        self.ran.append(argv[1] if len(argv) > 1 else argv[0])
        if any(x.endswith("delivery_table.py") for x in argv):
            j = Path(argv[argv.index("--json") + 1])
            j.write_text(json.dumps({"assembly": {"command": "STEP build", "stack": {"build": "STEP stack"}, "captures": [str(self.capture)]}}))
            Path(argv[argv.index("--table") + 1]).write_text("t")
        elif argv[:2] == ["STEP", "build"]:
            self.clip.write_bytes(b"clip")
            Path(str(self.clip) + ".check.json").write_text(json.dumps({**RECORD, "failures": [], "frames_expected": 11}))
        elif argv[:2] == ["STEP", "stack"]:
            self.clip.with_name(self.clip.stem + "_STACKED.mp4").write_bytes(b"stack")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def build_writes_over_nothing() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for role in ("stacked", "table"):
            clips, tables = d / role / "clips", d / role / "tables"
            target = src.files_of("fun_0001_0009_joined_T", clips, tables)[role]
            target.parent.mkdir(parents=True)
            target.write_text("kept")
            refused(lambda: src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(clips), "--tables",
                                      str(tables), "--stack-work", str(d / "w"), "--stamp", "T", "20-30", "1-9"]), "a kept output is not replaced", str(target))
            assert target.read_text() == "kept" and not (tables / "t_fun_0020_0030_joined_T.txt").exists(), "something was written before the refusal"
        refused(lambda: src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(d / "c"), "--tables",
                                  str(d / "t"), "--stack-work", str(d / "w"), "--stamp", "T", "1-9", "1-9"]), "given twice")
        # the steps stood in for: a capture that holds an earlier build's record of the name, then a stack with a hard fault
        capture = d / "capture"
        capture.mkdir()
        real_run, real_read = src.subprocess.run, src.stack_delivery.read_back
        for stamp, earlier, got, want in (("A", True, reading(11), "REFUSED after the table"), ("B", False, reading(11, clip_frames=10), "HARD FAULT"),
                                          ("C", False, reading(11), None)):
            steps = Steps(capture, got)
            steps.clip = d / "clips" / f"fun_0200_0210_joined_{stamp}.mp4"
            (d / "clips").mkdir(exist_ok=True)
            if earlier:
                (capture / f"delivery__fun_0200_0210_joined_{stamp}.json").write_text("{}")
            src.subprocess.run, src.stack_delivery.read_back = steps.run, (lambda *a, **k: steps.got)
            said = []
            real_print = builtins.print
            try:
                builtins.print = lambda *a, **k: said.append(" ".join(str(x) for x in a))
                code = src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(d / "clips"), "--tables",
                                 str(d / "tables"), "--stack-work", str(d / "w"), "--stamp", stamp, "200-210"])
            finally:
                builtins.print = real_print
                src.subprocess.run, src.stack_delivery.read_back = real_run, real_read
            text = "\n".join(said)
            if want:
                assert code == 1 and want in text, (stamp, code, text)
            else:
                assert code == 0 and "frames: the stacked file 11, the clip 11" in text and "ORIGINAL row" in text, (stamp, text)
            if earlier:
                assert "build" not in steps.ran and (capture / "delivery__fun_0200_0210_joined_A.json").read_text() == "{}", steps.ran
    # the stack's hard faults: a frame count not the span's (both counts equal included), sound not the clip's or none; a row's fit is not one
    assert src.stack_faults(reading(11), (200, 210)) == []
    assert src.stack_faults({**reading(11), "the_original_row": {**reading(11)["the_original_row"], "a_neighbour_fits_better": [3, 4]}}, (200, 210)) == []
    assert src.stack_faults(reading(11, same=False), (200, 210)) == ["the stacked file's sound is NOT the clip's"]
    assert src.stack_faults(reading(11, sound=False), (200, 210)) == ["neither the stacked file nor the clip carries sound"]
    assert src.stack_faults(reading(11, clip_frames=10), (200, 210))[0].startswith("the stacked file has 11 frames and the clip 10")
    assert src.stack_faults(reading(12), (200, 210))[0].startswith("the stacked file has 12 frames and the clip 12, where the span is 11")
    return ("a kept output or a span twice is refused before anything runs; an earlier record in a capture after the table, before the assembler; a stack "
            "whose counts are off the span (even equal) or whose sound is not the clip's or none stops the build; a clean build prints the stack whole")


def items_at_the_top() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page, clips, tables = d / "review" / "items.json", d / "clips", d / "tables"
        (page.parent / "media").mkdir(parents=True)
        clips.mkdir()
        tables.mkdir()
        before = {"updated": "x", "by": "y", "items": [{"id": "old"}]}
        page.write_text(json.dumps(before, indent=1) + "\n")
        for span in ((100, 140), (200, 210)):
            f = src.files_of(src.clip_name("fun", *span, "T"), clips, tables)
            f["stacked"].write_bytes(b"clip")
            f["record"].write_text(json.dumps(RECORD))
            f["json"].write_text(json.dumps(made_table()))
            f["stack_reading"].write_text(json.dumps(reading(span[1] - span[0] + 1)))
        base = ["items", "--items", str(page), "--clips", str(clips), "--tables", str(tables), "--stem", "fun", "--stamp", "T", "--by", "tester"]
        assert src.main([*base, "--dry", "100-140", "200-210"]) == 0 and json.loads(page.read_text()) == before and not list(tables.glob("items.before*")), "dry wrote"
        for argv, said in (([*base, "--note", "300-310=words", "100-140"], "not FIRST-LAST=words for one of the spans"),
                           ([*base, "--note", "100-140=one", "--note", "100-140=two", "100-140"], "two for 100-140"),
                           ([*base, "100-140", "100-140"], "given twice"),
                           ([*base, "100-140", "400-410"], "not built")):
            refused(lambda: src.main(argv), said)
        assert json.loads(page.read_text()) == before, "a refused run wrote"
        # a media link of the clip's name that points at another build is not reused
        stale = page.parent / "media" / "fun_0100_0140_joined_T_STACKED.mp4"
        (d / "another_build.mp4").write_bytes(b"another")
        os.symlink(d / "another_build.mp4", stale)
        refused(lambda: src.main([*base, "100-140"]), "not to this build's", "nothing written")
        assert json.loads(page.read_text()) == before and not list(tables.glob("items.before*")), "a refused link left a copy or a write"
        stale.unlink()
        # a table made before the table said why each piece stands, in fields of its own, is refused: its words would hide the lead's word
        old_shape = made_table()
        for e in old_shape["reading"]:
            for st in e["stands"]:
                del st["step"]
        f_old = src.files_of(src.clip_name("fun", 100, 140, "T"), clips, tables)
        kept_json = f_old["json"].read_text()
        f_old["json"].write_text(json.dumps(old_shape))
        refused(lambda: src.main([*base, "100-140"]), "made before the table said why", "Build it again")
        f_old["json"].write_text(kept_json)
        assert src.main([*base, "--note", "200-210=the lead's words", "100-140", "200-210"]) == 0
        after = json.loads(page.read_text())
        assert [x["id"] for x in after["items"]] == ["fun_0100_0140_joined_T_stacked", "fun_0200_0210_joined_T_stacked", "old"] and after["by"] == "tester", after["items"]
        copies = sorted(tables.glob("items.before_tester_T_*.json"))
        assert len(copies) == 1 and json.loads(copies[0].read_text()) == before and after["items"][1]["what"].endswith(" the lead's words"), "copy or note"
        assert stale.is_symlink() and stale.resolve() == (clips / "fun_0100_0140_joined_T_STACKED.mp4").resolve(), stale
        refused(lambda: src.main([*base, "100-140"]), "already has an item")
        f = src.files_of(src.clip_name("fun", 300, 310, "T"), clips, tables)
        f["stacked"].write_bytes(b"clip")
        f["record"].write_text(json.dumps({**RECORD, "verdict": "FAILS", "verdict_line": "FAILS: nothing laid"}))
        f["json"].write_text(json.dumps(made_table()))
        f["stack_reading"].write_text(json.dumps(reading(11)))
        refused(lambda: src.main([*base, "300-310"]), "no item for a clip that fails")
        f["record"].write_text(json.dumps(RECORD))
        f["stack_reading"].write_text(json.dumps(reading(11, same=False)))
        refused(lambda: src.main([*base, "300-310"]), "hard fault", "NOT the clip's")
        f["stack_reading"].write_text(json.dumps(reading(11)))
        here = os.getcwd()
        os.chdir(d)
        try:
            assert src.main(["items", "--items", "review/items.json", "--clips", "clips", "--tables", "tables", "--stem", "fun", "--stamp", "T", "--by", "tester",
                             "300-310"]) == 0
        finally:
            os.chdir(here)
        link = page.parent / "media" / "fun_0300_0310_joined_T_STACKED.mp4"
        assert link.is_symlink() and link.read_bytes() == b"clip", "a dangling link"
    return ("two items at the top in span order, the copy before named by its time, the link, the note; dry, a stray note, two notes, a span twice, an "
            "unbuilt span, a listed id, a failing record, a stack fault and a link to another build refused; a relative --clips links to the clip")


def by_the_day() -> str:
    t = made_table()
    t["made"] = "Made at commit x from the ledger `data/l.json` (...)"
    t["reading"][0]["stands"].append(stand("lead_pref", [(141, 141)], "2026-10-13T16:00:00Z", None, step=0, led="the lead's own reason, whole"))
    t["reading"].append(entry(141, 160, "lead", stands=[stand("lead_b", [(141, 160)], "2026-10-12T15:00:00Z", step=0, led="the lead's word on its second shot")]))
    t["not_in_the_shot"].append({"first": 141, "last": 160, "subject": "ghost", "why": "not in the shot"})
    ledger = {"renders": {
        "lead_a": {"name": "lead_a", "written": "2026-10-14T03:10:00Z", "graph": {"first": 100, "frames": 21}},
        "lead_pref": {"name": "lead_pref", "written": "2026-10-13T16:00:00Z", "graph": {"first": 141, "frames": 1}},
        "second_a": {"name": "second_a", "written": "2026-10-13T18:00:00Z", "graph": {"first": 100, "frames": 30}},
        "t_new": {"name": "t_new", "written": "2026-10-13T12:00:00Z", "graph": {}},
        "elsewhere": {"name": "elsewhere", "written": "2026-10-13T13:00:00Z", "graph": {"first": 900, "frames": 10}},
        "ghost_r": {"name": "ghost_r", "subjects": ["ghost"], "written": "2026-10-13T14:00:00Z", "graph": {"first": 145, "frames": 10}},
        "next_day": {"name": "next_day", "written": "2026-10-14T06:00:00Z", "graph": {"first": 5, "frames": 10}}}}
    text = src.by_day_text(t, ledger, "2026-10-13", "t.json", "l.json")
    rows = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| frames")]
    pieces = [r for r in rows if r.count("|") == 6]
    assert [r.split(" | ")[2] for r in pieces] == ["lead_a", "second_a", "sixth_a", "lead_b", "lead_pref"], pieces
    assert "| 121-140,141-160 | lead | lead_b | 10-12 10:00 | THE LEAD'S CHOICE (--prefer), a session's word: the lead's word on its second shot |" in text \
        and "| 145-154 | ghost_r | 10-13 09:00 | its subject ghost is not in the shot 141-160 by the shots record |" in text and "| lead_pref | 10-13 11:00 | THE LEAD'S CHOICE (--prefer), a session's word: the lead's own reason, whole |" in text, text
    assert "5 pieces stand: 4 rendered on 2026-10-13, 1 on other days." in text, text
    assert "## Renders of 2026-10-13 in the ledger that this table does not lay" in text and "This table covers 100-160" in text and "`l.json`" in text, text
    assert "| not in its record | t_new | 10-13 07:00 | " + LONG + " |" in text and "| 900-909 | elsewhere | 10-13 08:00 | on no shot of this table |" in text, text
    tail = text.split("## Renders of 2026-10-13")[1]
    assert all(n not in tail for n in ("lead_pref", "second_a", "next_day")) and "6 renders of 2026-10-13 in the ledger; 3 of them stand in this table." in text, tail
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kept.md"
        out.write_text("kept")
        (Path(tmp) / "t.json").write_text(json.dumps(t))
        (Path(tmp) / "l.json").write_text(json.dumps(ledger))
        (Path(tmp) / "other.json").write_text(json.dumps(ledger))
        refused(lambda: src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "other.json"), "--day", "2026-10-13",
                                  "--out", str(Path(tmp) / "new.md")]), "made from another ledger")
        refused(lambda: src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "l.json"), "--day", "2026-10-13",
                                  "--out", str(out)]), "a kept output is not replaced")
        assert out.read_text() == "kept" and not (Path(tmp) / "new.md").exists()
    return ("a row a piece, one render on two shots one row, the table's words and the lead's --prefer said as the lead's; the day's renders this table does not "
            "lay and none that stands in it; another ledger and a kept file refused")


case("the words of an item", words_of_an_item)
case("the lead's word is said as the lead's", the_leads_word)
case("a span of several shots", a_span_of_several_shots)
case("the item itself", the_item_itself)
case("the table is handed what it needs", the_table_is_handed_what_it_needs)
case("a build writes over nothing, and stops on a stack's hard fault", build_writes_over_nothing)
case("items at the top, and a copy first", items_at_the_top)
case("by the day", by_the_day)

if __name__ == "__main__":
    sys.exit(finish())
