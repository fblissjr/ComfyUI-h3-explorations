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
  sentences_in_order                inside a shot, subject by subject and a subject's sentences by frame: unsure first, passed over last
  the_owners_words_after_a_keep     a keep the owner wrote after is said with their words in the item, a passed-over keep and by-day; a
                                    passed-over wrong is never called a keep
  the_sixth_readers_findings        the owner's words with no click and after a not sure; the lead's word beside a click; a piece of no frame
                                    not counted; nothing to make; unsure passed over; Central in November; by-day's newest click and words alone;
                                    an older build, an overlapping item, a dangling link, the owner as --by and a peer's write refused or named
  the_seventh_readers_findings      a passed-over render's clicks and words with its reason whole; words sent with the click; reasons not cut;
                                    the cut's length; the shots counted; by-day's pieces that show nothing; an escaped |
  the_eighth_readers_findings       words up to 10 s before a click go with it, with their frame; a held render says the owner's lines; --by
                                    never the owner's name; a ledger without --joined-tables refused in every step
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
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table as dt  # noqa: E402
import shot_review_clips as src  # noqa: E402


def stand(render, frames, written, choice=None, left=(), why=None, step=2, led=None):
    # as bench/delivery_table.decide words it: a piece the lead put first stands by step 0, its reason the lead's
    why = why or (f"step 0: named by --prefer: {led}" if led else "step 2: the plan's region; its file agrees with its build record")
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


def table_of(a: int, b: int) -> dict:
    """The made table moved to the span a-b, as a build of that span writes it."""
    if (a, b) == (100, 140):
        return made_table()
    return {"reading": [entry(a, b, "lead", stands=[stand(f"lead_{a}", [(a, b)], "2026-10-13T15:00:00Z")])], "not_in_the_shot": []}


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
    # a stamp with no zone is UTC, never the machine's clock; a render standing on two shots is one piece
    assert f"{src.owners_clock('2026-10-14T03:10:00'):%m/%d %H:%M}" == "10/13 22:10"
    two = {"reading": [entry(100, 140, "lead", stands=[stand("lead_a", [(100, 140)], "2026-10-13T15:00:00Z")]),
                       entry(141, 160, "lead", stands=[stand("lead_a", [(141, 160)], "2026-10-13T15:00:00Z")])], "not_in_the_shot": []}
    assert src.item_words(two)[0] == 1, "a render on two shots counted twice"
    return ("four pieces each with its day on the owner's clock, either keep word, choice or none, frames left or NOT KNOWN; unsure with and without a piece, "
            "said once; not in the shot from the table's list; nothing standing said once, with the count and the newest three, a long reason cut")


def the_leads_word() -> str:
    reason = "the remake of today replaces her piece of 10/12 on the lead's ruling, which is a reason far longer than an item's cut of a passed-over reason"
    t = {"reading": [entry(100, 140, "lead", stands=[stand("lead_new", [(100, 140)], "2026-10-14T15:00:00Z", None, step=0, why=f"step 0: named by --prefer: {reason}",
                                                           led=reason)],
                           passed_over=[passed("lead_kept", "2026-10-12T15:00:00Z", "may stand (step 1) and every frame of it on this shot is taken by a render before it",
                                               {"said": dt.KEEP[0], "at": "2026-10-12T20:00:00Z"}),
                                        passed("lead_held", "2026-10-13T15:00:00Z", "HELD BACK ...", held="its apron is a white top on all 50 frames, a reader's word"),
                                        passed("lead_good", "2026-10-11T15:00:00Z", "never stands: no build record", {"said": dt.KEEP[1], "at": "2026-10-11T20:00:00Z"})]),
                     entry(100, 140, "second", passed_over=[passed("s_held", "2026-10-13T15:00:00Z", "HELD", held="a reason"),
                                                            passed("s_other", "2026-10-12T15:00:00Z", "never stands: no build record")])],
         "not_in_the_shot": []}
    text = ". ".join(src.item_words(t)[1])
    assert f"lead_new on 100-140, rendered 10/14 (10:00 Central); not judged by you yet; IT STANDS BY THE LEAD'S CHOICE (--prefer), a session's word and not yours: {reason}" in text, text
    assert "lead: YOUR KEEP lead_kept (of 10/12) is not laid here: may stand (step 1)" in text, text
    assert "lead: lead_held is HELD BACK by the lead (--hold), a session's word and not yours: its apron is a white top on all 50 frames, a reader's word" in text, text
    assert "lead: YOUR KEEP lead_good (of 10/11) is not laid here: never stands" in text, "the table's second word for a keep, set aside, unsaid"
    assert "second: the original on the whole shot: no piece of theirs stands here (1 other render(s) passed over, the newest 1: s_other" in text, text
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
    assert item["seconds"] == "about 4.2 s to 5.9 s of the video", item["seconds"]
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
        self.capture, self.got, self.ran, self.clip, self.table_argv = capture, got, [], Path(), []
        self.sidecar = True                               # the assembler writes the clip's sidecar (bench/sidecar.py); False: it did not;
                                                          # "stale": it was written, and the clip written again after it

    def run(self, argv, **_):
        argv = [str(x) for x in argv]
        self.ran.append(argv[1] if len(argv) > 1 else argv[0])
        if any(x.endswith("delivery_table.py") for x in argv):
            self.table_argv = argv
            j = Path(argv[argv.index("--json") + 1])
            j.write_text(json.dumps({"assembly": {"command": "STEP build", "stack": {"coverage": "STEP coverage", "build": "STEP stack"},
                                                  "captures": [str(self.capture)]}}))
            Path(argv[argv.index("--table") + 1]).write_text("t")
        elif argv[:2] == ["STEP", "build"]:
            self.clip.write_bytes(b"clip")
            Path(str(self.clip) + ".check.json").write_text(json.dumps({**RECORD, "failures": [], "frames_expected": 11}))
            if self.sidecar:
                src.sidecar.write(self.clip, ["STEP", "build"], {})
            if self.sidecar == "stale":
                self.clip.write_bytes(b"clip, written again")
        elif argv[:2] == ["STEP", "stack"]:
            self.clip.with_name(self.clip.stem + "_STACKED.mp4").write_bytes(b"stack")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def build_writes_over_nothing() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for role in src.files_of("x", Path("c"), Path("t")):
            clips, tables = d / role / "clips", d / role / "tables"
            target = src.files_of("fun_0001_0009_joined_T", clips, tables)[role]
            target.parent.mkdir(parents=True)
            target.write_text("kept")
            refused(lambda: src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(clips), "--tables",
                                      str(tables), "--stack-work", str(d / "w"), "--stamp", "T", "20-30", "1-9"]), "a kept output is not replaced", str(target))
            assert target.read_text() == "kept" and not (tables / "t_fun_0020_0030_joined_T.txt").exists(), "something was written before the refusal"
        refused(lambda: src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(d / "c"), "--tables",
                                  str(d / "t"), "--stack-work", str(d / "w"), "--stamp", "T", "1-9", "1-9"]), "given twice")
        # a ledger made without the joined clips' tables reads no click on one: refused before anything runs (a seventh reader)
        ledger = d / "L.json"
        ledger.write_text(json.dumps({"command": {"argv": ["--shots", "S"]}}))
        refused(lambda: src.main(["build", "--ledger", str(ledger), "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(d / "c"), "--tables",
                                  str(d / "t"), "--stack-work", str(d / "w"), "--stamp", "T", "1-9"]), "made without --joined-tables")
        assert not (d / "c").exists(), "something was written before the refusal"
        ledger.write_text(json.dumps({"command": {"argv": ["--shots", "S", "--joined-tables", "tables"]}}))
        # the steps stood in for: a capture that holds an earlier build's record of the name, then a stack with a hard fault
        capture = d / "capture"
        capture.mkdir()
        real_run, real_read = src.subprocess.run, src.stack_delivery.read_back
        for stamp, earlier, got, want in (("A", True, reading(11), "REFUSED after the table"), ("B", False, reading(11, clip_frames=10), "HARD FAULT"),
                                          ("C", False, reading(11), None), ("D", False, reading(11), "dry"), ("E", "relative", reading(11), "REFUSED after the table"),
                                          ("F", "csv", reading(11), "REFUSED after the table"), ("G", False, reading(11), "SIDECAR DOES NOT HOLD"),
                                          ("H", False, reading(11), "the video changed after its sidecar")):
            steps = Steps(capture, got)
            steps.sidecar = "stale" if stamp == "H" else want != "SIDECAR DOES NOT HOLD"
            steps.clip = d / "clips" / f"fun_0200_0210_joined_{stamp}.mp4"
            (d / "clips").mkdir(exist_ok=True)
            real_here = src.HERE
            if earlier == "relative":                 # a capture named relative to the repo, as real tables name them
                (d / "repo" / "bench").mkdir(parents=True, exist_ok=True)
                (d / "repo" / "cap").mkdir(exist_ok=True)
                (d / "repo" / "cap" / f"delivery__fun_0200_0210_joined_{stamp}.json").write_text("{}")
                steps.capture = Path("cap")
                src.HERE = d / "repo" / "bench"
            elif earlier == "csv":                    # the assembler's other write: a run's changed cells under the build's name (a seventh reader)
                (capture / "runs" / "r").mkdir(parents=True, exist_ok=True)
                (capture / "runs" / "r" / f"changed__fun_0200_0210_joined_{stamp}.csv").write_text("x")
            elif earlier:
                (capture / f"delivery__fun_0200_0210_joined_{stamp}.json").write_text("{}")
            src.subprocess.run, src.stack_delivery.read_back = steps.run, (lambda *a, **k: steps.got)
            said = []
            real_print = builtins.print
            try:
                builtins.print = lambda *a, **k: said.append(" ".join(str(x) for x in a))
                code = src.main(["build", "--ledger", str(ledger), "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(d / "clips"), "--tables",
                                 str(d / "tables"), "--stack-work", str(d / "w"), "--stamp", stamp, *(["--dry"] if want == "dry" else []), "200-210"])
            finally:
                builtins.print = real_print
                src.subprocess.run, src.stack_delivery.read_back = real_run, real_read
                src.HERE = real_here
            given = steps.table_argv
            assert all(Path(given[given.index(k) + 1]).is_absolute() for k in ("--ledger", "--shots", "--plan")), "a path handed on relative to the caller"
            if want == "dry":
                assert code == 0 and "build" not in steps.ran and "stack" not in steps.ran and not steps.clip.exists(), ("a dry build ran a step", steps.ran)
                continue
            text = "\n".join(said)
            if want:
                assert code == 1 and want in text, (stamp, code, text)
            else:
                assert code == 0 and "frames: the stacked file 11, the clip 11" in text and "ORIGINAL row" in text, (stamp, text)
                # THE CASE THAT GOES RED when the stack is built with no gap check: the table's per-frame gap check runs after the clip and before the stack
                assert [x for x in steps.ran if x in ("build", "coverage", "stack")] == ["build", "coverage", "stack"], steps.ran
                assert json.loads(src.files_of(f"fun_0200_0210_joined_{stamp}", d / "clips", d / "tables")["stack_reading"].read_text())["frames"] == 11
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
            f["json"].write_text(json.dumps(table_of(*span)))
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
        # A LATER BUILD OF THE SAME SPAN retires the earlier open joined item, and leaves one in another state as it is
        later = [*base[:-4], "--stamp", "U", "--by", "tester"]
        f_u = src.files_of(src.clip_name("fun", 100, 140, "U"), clips, tables)
        f_u["stacked"].write_bytes(b"clip")
        f_u["record"].write_text(json.dumps(RECORD))
        f_u["json"].write_text(json.dumps(table_of(100, 140)))
        f_u["stack_reading"].write_text(json.dumps(reading(41)))
        listed = json.loads(page.read_text())
        listed["items"].append({"id": "fun_0100_0140_joined_S_stacked", "state": "ok"})
        listed["items"].append({"id": "fun_0100_0141_joined_S_stacked", "state": "open"})
        page.write_text(json.dumps(listed, indent=1) + "\n")
        assert src.main([*later, "--dry", "100-140"]) == 0 and json.loads(page.read_text()) == listed, "dry wrote"
        assert src.main([*later, "100-140"]) == 0
        state = {x["id"]: x for x in json.loads(page.read_text())["items"]}
        # THE CASE THAT GOES RED when two open joined items of one span stand on the page at once
        assert state["fun_0100_0140_joined_T_stacked"]["state"] == "superseded" and state["fun_0100_0140_joined_T_stacked"]["superseded_by"] == "fun_0100_0140_joined_U_stacked", state
        assert "fun_0100_0140_joined_U_stacked is the one to watch" in state["fun_0100_0140_joined_T_stacked"]["state_words"]
        assert state["fun_0100_0140_joined_U_stacked"]["state"] == "open" and state["fun_0100_0140_joined_S_stacked"]["state"] == "ok", state
        assert state["fun_0100_0141_joined_S_stacked"]["state"] == "open" and state["fun_0200_0210_joined_T_stacked"]["state"] == "open", "another span was retired"
        # a table that is not of the span is refused
        f_wrong = src.files_of(src.clip_name("fun", 400, 410, "T"), clips, tables)
        f_wrong["stacked"].write_bytes(b"clip")
        f_wrong["record"].write_text(json.dumps(RECORD))
        f_wrong["json"].write_text(json.dumps(table_of(500, 510)))
        f_wrong["stack_reading"].write_text(json.dumps(reading(11)))
        refused(lambda: src.main([*base, "400-410"]), "is of 500-510, not of this span")
        # a link of the clip's name in the media folder that points at another build's file of the SAME name is not reused
        other = d / "another_build"
        other.mkdir()
        (other / "fun_0400_0410_joined_T_STACKED.mp4").write_bytes(b"another")
        f_wrong["json"].write_text(json.dumps(table_of(400, 410)))
        os.symlink(other / "fun_0400_0410_joined_T_STACKED.mp4", page.parent / "media" / "fun_0400_0410_joined_T_STACKED.mp4")
        refused(lambda: src.main([*base, "400-410"]), "not to this build's")
        f = src.files_of(src.clip_name("fun", 300, 310, "T"), clips, tables)
        f["stacked"].write_bytes(b"clip")
        f["record"].write_text(json.dumps({**RECORD, "verdict": "FAILS", "verdict_line": "FAILS: nothing laid"}))
        f["json"].write_text(json.dumps(table_of(300, 310)))
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


def sentences_in_order() -> str:
    """A fourth reader (2026-10-14): inside a shot, a subject's sentences came in the table's order, not the frames'."""
    t = {"reading": [entry(100, 160, "lead", stands=[stand("lead_late", [(140, 160)], "2026-10-13T15:00:00Z"), stand("lead_early", [(110, 139)], "2026-10-13T15:00:00Z")],
                           passed_over=[passed("lead_kept", "2026-10-12T15:00:00Z", "set aside", {"said": dt.KEEP[0], "at": "2026-10-12T16:00:00Z"})],
                           left=[[100, 109]]),
                     entry(100, 160, "second", present="unsure", stands=[stand("second_a", [(100, 160)], "2026-10-13T15:00:00Z")]),
                     entry(161, 170, "lead", stands=[stand("lead_next", [(161, 170)], "2026-10-13T15:00:00Z")])],
         "not_in_the_shot": [{"first": 100, "last": 160, "subject": "third", "why": "the record"}]}
    _, said = src.item_words(t)
    order = [next(k for k in ("the original on 100-109", "lead_early", "lead_late", "YOUR KEEP lead_kept", "second: the shots record is unsure", "second_a",
                              "third: not in this shot", "lead_next") if k in w) for w in said]
    # THE CASE THAT GOES RED when a subject's sentences inside a shot follow the table's order and not the frames'
    assert order == ["the original on 100-109", "lead_early", "lead_late", "YOUR KEEP lead_kept", "second: the shots record is unsure", "second_a",
                     "third: not in this shot", "lead_next"], said
    return "shot by shot, subject by subject in the table's order, a subject's sentences by frame: unsure first, passed over and held last"


def the_owners_words_after_a_keep() -> str:
    """A fifth reader (2026-10-14): third_fp_1466b's last click is a keep and six seconds later the owner wrote "i dont know i cant
    choose"; the item said "your keep of 10/13". Their words rule over the button, wherever a keep is said."""
    words = [{"at": "2026-10-13T20:00:06Z", "said": "i dont know i cant choose"}]
    keep = {"said": dt.KEEP[0], "at": "2026-10-13T20:00:00Z"}
    stood = dict(stand("lead_kept", [(100, 120)], "2026-10-13T15:00:00Z", keep, step=1, why="step 1: the owner's keep"), the_owners_words=words)
    t = {"reading": [entry(100, 120, "lead", stands=[stood], passed_over=[
        dict(passed("lead_old", "2026-10-12T15:00:00Z", "set aside", keep), the_owners_words=words),
        passed("lead_wrong", "2026-10-11T15:00:00Z", "never stands", {"said": "wrong", "at": "2026-10-12T16:00:00Z"})])], "not_in_the_shot": []}
    _, said = src.item_words(t)
    text = " ".join(said)
    # THE CASES THAT GO RED when a keep is said plainly with the owner's words after it, or a passed-over wrong is called a keep
    assert "your keep of 10/13, AND YOUR WORDS AFTER IT: \"i dont know i cant choose\"" in text, text
    assert "YOUR KEEP lead_old (of 10/13, and your words after it: \"i dont know i cant choose\"" in text and "lead_wrong" not in text.split("YOUR KEEP")[-1].split(")")[0], text
    assert not any("YOUR KEEP lead_wrong" in x for x in said), said
    day = src.by_day_text(t, {"renders": {}}, "2026-10-13", "t.json", "l.json")
    assert "lead_old ('keep this one', 10-13 15:00 Central; your words after it: \"i dont know i cant choose\"" in day, day
    assert "| 'keep this one', 10-13 15:00; your words after it: \"i dont know i cant choose\" (10/13 15:00 Central) |" in day, day
    return "a keep the owner wrote after is said with their words, in the item, in a passed-over keep and in by-day; a passed-over wrong is never called a keep"

def the_sixth_readers_findings() -> str:
    """A sixth reader (2026-10-14), its findings and the changes that left every case green, each held here."""
    beard = [{"at": "2026-10-14T00:15:55Z", "said": "i dont like beard"}]
    t = {"reading": [
        entry(100, 120, "lead", stands=[dict(stand("lead_words", [(100, 120)], "2026-10-13T15:00:00Z"), the_owners_words=beard)]),
        entry(100, 120, "second", stands=[dict(stand("second_ns", [(100, 120)], "2026-10-13T15:00:00Z", {"said": dt.NOT_SURE, "at": "2026-10-13T16:00:00Z"},
                                                     led="THE LEAD'S REASON"), the_owners_words=[{"at": "2026-10-13T16:05:00Z", "said": "her eyes look right"}])]),
        entry(100, 120, "third", stands=[stand("third_gone", [(100, 120)], "2026-10-13T15:00:00Z", left=[[100, 120]])]),
        dict(entry(100, 120, "fourth", passed_over=[passed("f_kept", "2026-10-12T15:00:00Z", "set aside", {"said": dt.KEEP[1], "at": "2026-10-12T16:00:00Z"}),
                                                     passed("f_other", "2026-10-12T15:00:00Z", "R" * src.REASON_CHARS)]),
             nothing_to_make=["A LOOK: HER LEGS ONLY (mrx)"]),
        entry(100, 120, "fifth", present="unsure", passed_over=[passed("u_old", "2026-10-12T15:00:00Z", "never stands")])], "not_in_the_shot": []}
    pieces, said = src.item_words(t)
    text = " ".join(said)
    # THE CASES THAT GO RED: the owner's words with no click, or after a not sure, hidden; a piece of no frame counted; the table's reason cut
    assert "lead_words" in text and "not judged by you yet, BUT YOUR WORDS ON IT: \"i dont like beard\" (10/13 19:15 Central)" in text, text
    assert "your last choice on it: \"not sure\", AND YOUR WORDS AFTER IT: \"her eyes look right\" (10/13 11:05 Central)" in text, text
    assert "IT STANDS BY THE LEAD'S CHOICE (--prefer), a session's word and not yours: THE LEAD'S REASON" in text, "the lead's word hidden under the owner's click"
    assert pieces == 2 and "that is all its frames, so nothing of it shows and it is not counted as laid" in text, (pieces, text)
    assert "present, nothing to make under this plan, by the table: A LOOK: HER LEGS ONLY (mrx)" in text, text
    assert text.count("f_kept") == 1 and "(1 other render(s) passed over, the newest 1: f_other, " + "R" * src.REASON_CHARS + ")" in text, text
    assert "the original throughout (1 other render(s) passed over, the newest 1: u_old, never stands)" in text, text
    assert sum("the original on the whole shot" in x for x in said) == 1, "an unsure subject said twice to be the original"
    # the owner's clock is Central with its daylight time, not a fixed offset: in November it is six hours behind UTC
    assert f"{src.owners_clock('2026-11-05T15:00:00Z'):%H:%M}" == "09:00" and f"{src.owners_clock('2026-10-05T15:00:00Z'):%H:%M}" == "10:00"
    # by-day: the newest click on any shot of a piece; words with no click; a set-aside "good to go"
    t2 = {"made": "", "reading": [
        entry(100, 120, "lead", stands=[stand("two_shots", [(100, 120)], "2026-10-13T15:00:00Z", {"said": dt.NOT_SURE, "at": "2026-10-13T20:00:00Z"})],
              passed_over=[passed("gtg", "2026-10-12T15:00:00Z", "set aside", {"said": dt.KEEP[1], "at": "2026-10-12T16:00:00Z"})]),
        entry(121, 140, "lead", stands=[stand("two_shots", [(121, 140)], "2026-10-13T15:00:00Z", {"said": dt.KEEP[0], "at": "2026-10-13T21:00:00Z"})]),
        entry(141, 150, "lead", stands=[dict(stand("words_only", [(141, 150)], "2026-10-13T15:00:00Z"), the_owners_words=beard)])], "not_in_the_shot": []}
    day = src.by_day_text(t2, {"renders": {}}, "2026-10-13", "t.json", "l.json")
    assert "| two_shots | 10-13 10:00 | 'keep this one', 10-13 16:00 |" in day, day
    assert "| words_only | 10-13 10:00 | no click; your words on it: \"i dont like beard\" (10/13 19:15 Central) |" in day, day
    assert "gtg ('good to go'" in day, "a set-aside good-to-go keep is hidden"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page, clips, tables = d / "review" / "items.json", d / "clips", d / "tables"
        (page.parent / "media").mkdir(parents=True)
        clips.mkdir()
        tables.mkdir()
        f = src.files_of(src.clip_name("fun", 100, 140, "T2"), clips, tables)
        f["stacked"].write_bytes(b"clip")
        f["record"].write_text(json.dumps(RECORD))
        laid = dict(made_table(), rows=[{"first": 100, "last": 140, "name": "lead_a", "subject": "lead"}])
        f["json"].write_text(json.dumps(laid))
        f["stack_reading"].write_text(json.dumps(reading(41)))
        # an older build of the same pieces is retired into this one; one of other pieces is left open, its lines its own (the lead's ruling)
        src.files_of(src.clip_name("fun", 100, 140, "T1"), clips, tables)["json"].write_text(json.dumps(laid))
        src.files_of(src.clip_name("fun", 100, 140, "T0"), clips, tables)["json"].write_text(json.dumps(dict(laid, rows=[dict(laid["rows"][0], name="lead_old")])))
        # an eighth reader's gaps: the same piece on other frames, of another subject, or past the span's edges; no table; not open
        variants = {"T05": [dict(laid["rows"][0], first=101)], "T06": [dict(laid["rows"][0], subject="second")],
                    "T07": [dict(laid["rows"][0], first=90, last=150)]}
        for st_, rows_ in variants.items():
            src.files_of(src.clip_name("fun", 100, 140, st_), clips, tables)["json"].write_text(json.dumps(dict(laid, rows=rows_)))
        src.files_of(src.clip_name("fun", 100, 140, "T09"), clips, tables)["json"].write_text(json.dumps(laid))
        listed = {"updated": "x", "by": "y", "items": [
            {"id": "keep_me", "state": "open"},
            {"id": "fun_0100_0140_joined_T0_stacked", "first": "100", "frames": "41", "state": "open"},            # other pieces: left open
            {"id": "fun_0100_0140_joined_T05_stacked", "first": "100", "frames": "41", "state": "open"},           # the piece on other frames: left open
            {"id": "fun_0100_0140_joined_T06_stacked", "first": "100", "frames": "41", "state": "open"},           # of another subject: left open
            {"id": "fun_0100_0140_joined_T07_stacked", "first": "100", "frames": "41", "state": "open"},           # the same inside the span: retired
            {"id": "fun_0100_0140_joined_T08_stacked", "first": "100", "frames": "41", "state": "open"},           # no table: left open
            {"id": "fun_0100_0140_joined_T09_stacked", "first": "100", "frames": "41", "state": "ok"},             # not open: left as it is
            {"id": "fun_0100_0140_joined_T1_stacked", "first": "100", "frames": "41", "state": "open"},            # older, second on the list: retired
            {"id": "fun_0100_0140_joined_T1_check", "state": "open"},                                               # not a stacked item: left
            {"id": "fun_0090_0110_joined_T1_stacked", "first": "90", "frames": "21", "state": "open"}]}             # overlapping: named, left
        page.write_text(json.dumps(listed, indent=1) + "\n")
        base = ["items", "--items", str(page), "--clips", str(clips), "--tables", str(tables), "--stem", "fun", "--by", "tester"]
        for who in ("the owner", "Owner", "the_owner", "fblissjr"):
            refused(lambda w=who: src.main([*base, "--stamp", "T2", "--by", w, "100-140"]), "never the owner's")
        newer = json.loads(page.read_text())
        newer["items"].append({"id": "fun_0100_0140_joined_T3_stacked", "first": "100", "frames": "41", "state": "superseded"})   # a later build blocks in any state
        page.write_text(json.dumps(newer, indent=1) + "\n")
        refused(lambda: src.main([*base, "--stamp", "T2", "100-140"]), "a LATER build of the same frames", "T3", "nothing written")
        page.write_text(json.dumps(listed, indent=1) + "\n")
        # a dangling media link of the clip's name is not taken as this build's
        os.symlink(d / "nowhere.mp4", page.parent / "media" / f["stacked"].name)
        refused(lambda: src.main([*base, "--stamp", "T2", "100-140"]), "not to this build's")
        (page.parent / "media" / f["stacked"].name).unlink()
        # another session writes the list while this run works: refused, its change kept
        real = src.make_item

        def meanwhile(*a, **k):
            page.write_text(json.dumps({**listed, "items": [{"id": "a_peers_item"}, *listed["items"]]}, indent=1) + "\n")
            return real(*a, **k)
        src.make_item = meanwhile
        try:
            refused(lambda: src.main([*base, "--stamp", "T2", "100-140"]), "changed since this run read it", "nothing written")
        finally:
            src.make_item = real
        assert json.loads(page.read_text())["items"][0]["id"] == "a_peers_item", "a peer's write was lost"
        page.write_text(json.dumps(listed, indent=1) + "\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert src.main([*base, "--stamp", "T2", "100-140"]) == 0
        after = {x["id"]: x for x in json.loads(page.read_text())["items"]}
        assert after["fun_0100_0140_joined_T1_stacked"]["state"] == "superseded" and after["fun_0100_0140_joined_T1_check"]["state"] == "open", after
        # THE CASE THAT GOES RED when a build of other pieces takes an older item's place, and the owner's lines on it with it
        assert after["fun_0100_0140_joined_T0_stacked"]["state"] == "open" and "leaves fun_0100_0140_joined_T0_stacked open: it lays other pieces" in out.getvalue()
        assert [after[f"fun_0100_0140_joined_{x}_stacked"]["state"] for x in ("T05", "T06", "T07", "T08", "T09")] == ["open", "open", "superseded", "open", "ok"], after
        assert "leaves fun_0100_0140_joined_T08_stacked open: its table" in out.getvalue(), out.getvalue()
        assert after["fun_0090_0110_joined_T1_stacked"]["state"] == "open" and "overlaps it and stays open: fun_0090_0110_joined_T1_stacked (90-110)" in out.getvalue()
        assert after["fun_0100_0140_joined_T2_stacked"]["seconds"] == "about 4.2 s to 5.9 s of the video", after["fun_0100_0140_joined_T2_stacked"]["seconds"]
        # the seconds of a span's first frame are its own, not the next one's: on 200 the two round apart
        assert src.make_item("x", (200, 210), table_of(200, 210), RECORD, "l", "T", "s", None)["seconds"] == "about 8.3 s to 8.8 s of the video"
        # a table of more frames than the span is refused
        f3 = src.files_of(src.clip_name("fun", 100, 140, "T4"), clips, tables)
        for k in ("stacked", "record", "stack_reading"):
            f3[k].write_bytes(f[k].read_bytes())
        wide = made_table()
        wide["reading"][0]["last"] = 150
        f3["json"].write_text(json.dumps(wide))
        refused(lambda: src.main([*base, "--stamp", "T4", "100-140"]), "not of this span")
    return ("the owner's words said with no click and after a not sure; the lead's word beside the owner's click; a piece of no frame not counted; nothing to make "
            "said; an unsure subject's passed-over renders; a passed-over keep named once; Central in November; by-day's newest click, words alone and a "
            "good-to-go set aside; an older build refused, an overlapping item named, a non-stacked item left, an older item anywhere on the list retired, a "
            "dangling link, the owner as --by and a peer's write in between refused; the seconds; a wider table refused")


def the_seventh_readers_findings() -> str:
    """A seventh reader (2026-10-14), its findings and the changes that left every case green."""
    long = "A REASON WHOLE " + "x" * 130
    t = {"reading": [
        entry(100, 120, "lead", stands=[stand("lead_a", [(100, 120)], "2026-10-13T15:00:00Z")], passed_over=[
            dict(passed("lead_other", "2026-10-12T15:00:00Z", long, {"said": "wrong", "at": "2026-10-13T16:00:00Z"}),
                 the_owners_words=[{"at": "2026-10-13T16:00:00Z", "said": "SENT WITH THE CLICK"}, {"at": "2026-10-13T17:00:00Z", "said": "LINE ONE\nLINE TWO"}]),
            dict(passed("lead_words", "2026-10-12T15:00:00Z", "set aside"), the_owners_words=[{"at": "2026-10-13T18:00:00Z", "said": "I LIKE THIS ONE"}]),
            passed("lead_kept", "2026-10-12T15:00:00Z", long, {"said": dt.KEEP[0], "at": "2026-10-12T16:00:00Z"})]),
        dict(entry(100, 120, "second", present="unsure"), nothing_to_make=[long]),
        dict(entry(100, 120, "third"), nothing_to_make=[long])],
        "not_in_the_shot": [{"first": 121, "last": 130, "subject": "lead", "why": "not in the shot"}]}
    pieces, said = src.item_words(t)
    text = " ".join(said)
    # THE CASES THAT GO RED: a passed-over render's clicks and words only counted; words sent with the click hidden; a reason cut
    assert ("lead_other is not laid here (" + long + "); your last choice on it: \"wrong\" (10/13), and your words: \"SENT WITH THE CLICK\" "
            "(10/13 11:00 Central); \"LINE ONE / LINE TWO\" (10/13 12:00 Central)") in text, text
    assert "lead_words is not laid here (set aside); no click of yours on it, and your words: \"I LIKE THIS ONE\"" in text, text
    assert "YOUR KEEP lead_kept (of 10/12) is not laid here: " + long in text, "a passed-over keep's reason cut"
    assert "the original throughout; present, nothing to make under this plan, by the table: " + long in text, "an unsure subject's nothing-to-make reason lost or cut"
    assert "present, nothing to make under this plan, by the table: " + long in text.split("third:")[1], "the nothing-to-make reason cut"
    assert src.REASON_CHARS == 110 and src.cut("y" * 110) == "y" * 110 and src.cut("y" * 111) == "y" * 110 + "...", "the display cut is not 110 characters"
    item = src.make_item("x", (100, 130), t, RECORD, "l", "T", "s", None)
    assert item["what"].startswith("The 2 shots of 100-130"), "a shot only the not-in list holds is not counted"
    day = src.by_day_text(dict(t, reading=t["reading"] + [entry(140, 150, "lead", stands=[stand("all_gone", [(140, 150)], "2026-10-13T15:00:00Z", left=[[140, 150]],
                                                                                                why="a | in a reason")])]), {"renders": {}}, "2026-10-13", "t.json", "l.json")
    assert "| all_gone (NOTHING OF IT SHOWS: it leaves the original on all its frames) |" in day and "1 of them show nothing" in day, day
    assert "a \\| in a reason" in day, "a | in a cell breaks the table"
    tail = day.split("## Renders this table does not lay that carry your clicks or words")[1]
    assert "| 100-120 | lead | lead_other | 'wrong', 10-13 11:00 |" in tail and "| lead_words | none | \"I LIKE THIS ONE\"" in tail, tail
    return ("a passed-over render's clicks and words said with its reason whole, in the item and by-day; words sent with the click; multi-line words whole; "
            "a passed-over keep's and a nothing-to-make reason whole, an unsure subject's too; the cut at 110; the shots counted from the not-in list; "
            "by-day's pieces that show nothing and an escaped |")


def the_eighth_readers_findings() -> str:
    """An eighth reader (2026-10-14), and the lead's ruling that words sent up to 10 s before a click go with it."""
    import subprocess
    click = {"said": dt.NOT_SURE, "at": "2026-10-13T16:00:00Z"}
    words = [{"at": "2026-10-13T15:59:57Z", "said": "SENT WITH THE CLICK", "frame": 433}, {"at": "2026-10-13T15:59:00Z", "said": "A MINUTE BEFORE"}]
    t = {"reading": [entry(100, 120, "lead", stands=[dict(stand("lead_a", [(100, 120)], "2026-10-13T15:00:00Z", click), the_owners_words=words)],
                           passed_over=[dict(passed("lead_held", "2026-10-12T15:00:00Z", "held", {"said": dt.KEEP[0], "at": "2026-10-12T16:00:00Z"}, held="THE LEAD'S HOLD"),
                                             the_owners_words=[{"at": "2026-10-12T17:00:00Z", "said": "BEST ONE"}])])], "not_in_the_shot": []}
    text = " ".join(src.item_words(t)[1])
    # THE CASES THAT GO RED: words sent just before the click hidden; a word's frame dropped; a held render hiding the owner's keep
    assert "AND YOUR WORDS AFTER IT: \"SENT WITH THE CLICK\" (10/13 10:59 Central, at frame 433)" in text and "A MINUTE BEFORE" not in text, text
    assert "lead_held is HELD BACK by the lead (--hold), a session's word and not yours: THE LEAD'S HOLD; your last choice on it: \"keep this one\" (10/12), and your words: \"BEST ONE\"" in text, text
    assert src.WITH_THE_CLICK_S == 10
    # --by never the owner's name: read here from git as the tool reads it, never written down
    name = subprocess.run(["git", "-C", str(src.HERE.parent), "config", "user.name"], capture_output=True, text=True).stdout.strip()
    if name:
        assert src.names_the_owner(name) and src.names_the_owner(name.split()[0].upper() + " the session") and not src.names_the_owner("mrdrill"), "the owner's name"
        refused(lambda: src.main(["items", "--items", "x", "--clips", "c", "--tables", "t", "--stem", "fun", "--stamp", "T", "--by", name.split()[0], "1-2"]),
                "never the owner's")
    with tempfile.TemporaryDirectory() as tmp:
        led = Path(tmp) / "l.json"
        led.write_text(json.dumps({"command": {"argv": ["--joined-tablesX", "x"]}}))
        refused(lambda: src.joined_ledger(str(led)), "made without --joined-tables")
        led.write_text(json.dumps({"command": {"argv": ["--joined-tables", "x"]}}))
        src.joined_ledger(str(led))
    return ("words up to 10 s before the click go with it, with their frame; a held render says the owner's click and words; --by never the owner's "
            "name; a ledger without --joined-tables refused, the flag by its whole name")


def by_the_day() -> str:
    t = made_table()
    t["made"] = "Made at commit x from the ledger `LEDGER` (...)"
    t["reading"][0]["stands"].append(stand("lead_pref", [(141, 141)], "2026-10-13T16:00:00Z", None, step=0, led="the lead's own reason, whole"))
    t["reading"].append(entry(141, 160, "lead", stands=[stand("lead_b", [(141, 160)], "2026-10-12T15:00:00Z", step=0, led="the lead's word on its second shot")]))
    t["not_in_the_shot"].append({"first": 141, "last": 170, "subject": "ghost", "why": "not in the shot"})
    # the owner kept a piece the lead's --prefer set aside on 141-160: said beside what stands, not hidden under the lead's word
    t["reading"][-1]["passed_over"] = [{"render": "lead_owner", "why": "set aside by --prefer", "the_owners_last_choice": {"said": "keep this one", "at": "2026-10-13T01:00:00Z"}},
                                       {"render": "elsewhere", "why": "a reason on this shot", "the_owners_last_choice": None}]
    t["reading"][0].setdefault("passed_over", []).append({"render": "elsewhere", "why": "another reason on the first shot", "the_owners_last_choice": None})
    t["reading"][-1]["stands"][0]["the_owners_last_choice"] = {"said": "not sure", "at": "2026-10-13T02:00:00Z"}
    ledger = {"renders": {
        "lead_a": {"name": "lead_a", "written": "2026-10-14T03:10:00Z", "graph": {"first": 100, "frames": 21}},
        "lead_pref": {"name": "lead_pref", "written": "2026-10-13T16:00:00Z", "graph": {"first": 141, "frames": 1}},
        "second_a": {"name": "second_a", "written": "2026-10-13T18:00:00Z", "graph": {"first": 100, "frames": 30}},
        "t_new": {"name": "t_new", "written": "2026-10-13T12:00:00Z", "graph": {}},
        "elsewhere": {"name": "elsewhere", "written": "2026-10-13T13:00:00Z", "graph": {"first": 900, "frames": 10}},
        "ghost_r": {"name": "ghost_r", "subjects": ["ghost"], "written": "2026-10-13T14:00:00Z", "graph": {"first": 145, "frames": 10}},
        "next_day": {"name": "next_day", "written": "2026-10-14T06:00:00Z", "graph": {"first": 5, "frames": 10}}}}
    text = src.by_day_text(t, ledger, "2026-10-13", "t.json", "l.json")
    rows = [x for x in text.split("## Renders this table does not lay that carry")[0].splitlines() if x.startswith("| ") and not x.startswith("| frames")]
    pieces = [r for r in rows if r.count("|") == 7]
    assert [r.split(" | ")[2] for r in pieces] == ["lead_a", "second_a", "sixth_a", "lead_b", "lead_pref"], pieces
    # THE CASE THAT GOES RED when an owner's keep the lead's --prefer set aside is hidden under the lead's word
    assert ("| 121-140,141-160 | lead | lead_b | 10-12 10:00 | 'not sure', 10-12 21:00 | THE LEAD'S CHOICE (--prefer), a session's word: the lead's word on its "
            "second shot; on its other shots, the table's: step 2: the plan's region; its file agrees with its build record. THE OWNER'S OTHER KEEP(S) on these "
            "frames, not laid by this table: lead_owner ('keep this one', 10-12 20:00 Central) |") in text, text
    assert "| 145-154 | ghost_r | 10-13 09:00 | its subject ghost is not in the shot 141-170 by the shots record |" in text \
        and "| none | THE LEAD'S CHOICE (--prefer), a session's word: the lead's own reason, whole |" in text, text
    assert "5 pieces stand: 4 rendered on 2026-10-13, 1 on other days." in text, text
    assert "## Renders of 2026-10-13 in the ledger that this table does not lay" in text and "This table covers 100-170" in text and "`l.json`" in text, text
    assert "| not in its record | t_new | 10-13 07:00 | on 100-140: " + LONG + " |" in text, text
    assert "| 900-909 | elsewhere | 10-13 08:00 | on 100-140: another reason on the first shot; on 141-160: a reason on this shot |" in text, text
    tail = text.split("## Renders of 2026-10-13")[1]
    assert all(n not in tail for n in ("lead_pref", "second_a", "next_day")) and "6 renders of 2026-10-13 in the ledger; 3 of them stand in this table." in text, tail
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kept.md"
        out.write_text("kept")
        (Path(tmp) / "a").mkdir()
        t["made"] = t["made"].replace("LEDGER", str(Path(tmp) / "l.json"))
        (Path(tmp) / "t.json").write_text(json.dumps(t))
        for f in ("l.json", "other.json", "a/l.json"):
            (Path(tmp) / f).write_text(json.dumps(dict(ledger, command={"argv": ["--joined-tables", "tables"]})))
        # another ledger, and one of the same name in another folder, are refused: the table's ledger is held by its path
        for wrong in ("other.json", "a/l.json"):
            refused(lambda w=wrong: src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / w), "--day", "2026-10-13",
                                              "--out", str(Path(tmp) / "new.md")]), "made from another ledger")
        assert src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "l.json"), "--day", "2026-10-13",
                         "--out", str(Path(tmp) / "new.md")]) == 0 and (Path(tmp) / "new.md").is_file()
        # by-day refuses a ledger made without the joined clips' tables, as build does (an eighth reader)
        (Path(tmp) / "l.json").write_text(json.dumps(ledger))
        refused(lambda: src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "l.json"), "--day", "2026-10-13",
                                  "--out", str(Path(tmp) / "new2.md")]), "made without --joined-tables")
        (Path(tmp) / "l.json").write_text(json.dumps(dict(ledger, command={"argv": ["--joined-tables", "tables"]})))
        (Path(tmp) / "new.md").unlink()
        refused(lambda: src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "l.json"), "--day", "2026-10-13",
                                  "--out", str(out)]), "a kept output is not replaced")
        assert out.read_text() == "kept" and not (Path(tmp) / "new.md").exists()
    return ("a row a piece, one render on two shots one row, the table's words and the lead's --prefer said as the lead's, with the owner's last choice on it "
            "and an owner's keep the lead set aside beside it; the span counts the shots a subject is not in; a render passed over on two shots says both "
            "reasons; the day's renders this table does not lay and none that stands in it; another ledger, one of the same name elsewhere and a kept file refused")


case("the words of an item", words_of_an_item)
case("the lead's word is said as the lead's", the_leads_word)
case("a span of several shots", a_span_of_several_shots)
case("the item itself", the_item_itself)
case("the table is handed what it needs", the_table_is_handed_what_it_needs)
case("a build writes over nothing, and stops on a stack's hard fault", build_writes_over_nothing)
case("items at the top, and a copy first", items_at_the_top)
case("an item's sentences in order", sentences_in_order)
case("the owner's words after a keep", the_owners_words_after_a_keep)
case("by the day", by_the_day)
case("an eighth reader's findings", the_eighth_readers_findings)
case("a sixth reader's findings", the_sixth_readers_findings)
case("a seventh reader's findings", the_seventh_readers_findings)

if __name__ == "__main__":
    sys.exit(finish())
