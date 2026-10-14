#!/usr/bin/env python3
"""`bench/shot_review_clips.py` on made tables, records and lists: each case is a way a shot's review clip could say the wrong day, lose a piece or a subject, or write over what stands.

  the_words_of_an_item              every standing piece with its frames, the day and hour it was rendered on the owner's clock (a render
                                    written after midnight UTC is the day before in Central), the owner's keep or last choice or none,
                                    the frames its load left as the original; a subject the record is unsure of, one not in the shot,
                                    one with nothing standing (with what was passed over), and frames of a subject left as the source
  the_item_itself                   the buttons are the table's own, the rate is the assembler's record's, a person's note is added whole
  the_table_is_handed_what_it_needs the table's command carries the span, the source and the clip's name in --assemble-args, the stack's
                                    scratch folder, and every --also-capture, --prefer and --hold as given
  build_writes_over_nothing         a span whose clip, stack, record or table is there already is refused before anything runs
  items_at_the_top_and_a_copy       the items go to the top in span order with a copy of the list written first and the clip linked into
                                    the list's media folder; an id already listed, a clip whose record fails, a span not built and a
                                    note for another span are refused; --dry writes nothing
  by_the_day                        the pieces in frame order with why each stands in words, the count on the day asked about, and the
                                    day's renders that do not stand with why; an output that is there is not replaced

    <comfy venv python> bench/check_shot_review_clips.py

The standard library only; no model, no card, no video: the build's own steps are the assembler's and the stack's,
each held by its own check. What this cannot see: a build on a real ledger (its first run was a throwaway, read by
someone who did not write it).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table as dt  # noqa: E402
import shot_review_clips as src  # noqa: E402


def stand(render, frames, written, choice=None, left=None, why="step 2: the newest render"):
    return {"render": render, "frames": [list(f) for f in frames], "written": written, "why": why, "leaves_as_the_original": left,
            "the_owners_last_choice": choice}


def made_table() -> dict:
    return {"reading": [
        {"first": 100, "last": 140, "subject": "lead", "present": "yes", "left_as_the_source": [], "passed_over": [],
         "stands": [stand("lead_a", [(100, 120)], "2026-10-14T03:10:00Z", {"said": dt.KEEP[0], "at": "2026-10-14T04:00:00Z"}, why="step 1: the owner's keep"),
                    stand("lead_b", [(121, 140)], "2026-10-12T15:00:00Z", None, [[125, 127]])]},
        {"first": 100, "last": 140, "subject": "second", "present": "yes", "left_as_the_source": [[130, 140]], "passed_over": [],
         "stands": [stand("second_a", [(100, 129)], "2026-10-13T18:00:00Z", {"said": dt.NOT_SURE, "at": "2026-10-13T20:00:00Z"})]},
        {"first": 100, "last": 140, "subject": "third", "present": "yes", "left_as_the_source": [], "stands": [],
         "passed_over": [{"render": "third_old", "why": "never stands: no build record"}]},
        {"first": 100, "last": 140, "subject": "fourth", "present": "unsure", "left_as_the_source": [], "stands": [], "passed_over": []},
        {"first": 100, "last": 140, "subject": "fifth", "present": "no", "left_as_the_source": [], "stands": [], "passed_over": []}]}


RECORD = {"verdict": "passes", "verdict_line": "passes; every named subject has something laid on every shot", "video": {"rate": "24000/1001"}}


def words_of_an_item() -> str:
    pieces, said = src.item_words(made_table())
    text = ". ".join(said)
    assert pieces == 3, pieces
    assert "lead: lead_a on 100-120, rendered 10/13 (22:10 Central); your keep of 10/13" in text, text
    assert "lead: lead_b on 121-140, rendered 10/12 (10:00 Central); not judged by you yet; inside it the piece leaves the original on 125-127" in text, text
    assert f"second: second_a on 100-129, rendered 10/13 (13:00 Central); your last choice on it: \"{dt.NOT_SURE}\"" in text \
        and "second: the original on 130-140, where no piece of theirs is laid" in text, text
    assert "third: the original on the whole shot: no piece of theirs stands here (passed over: third_old, never stands: no build record)" in text, text
    assert "fourth: the shots record is unsure they are in this shot" in text and "fifth: not in this shot by the shots record" in text, text
    return "three pieces each with its day on the owner's clock (03:10Z is 22:10 the day before), keep, choice or none, frames left; unsure, absent, none standing"


def the_item_itself() -> str:
    item = src.make_item("fun_0100_0140_joined_T", (100, 140), made_table(), RECORD, "data/fun/review/media/fun_0100_0140_joined_T_STACKED.mp4",
                         "T", "tester", "A person's words, whole.")
    assert item["options"] == list(dt.RENDER_ITEM_BUTTONS) and item["fps"] == "23.976" and item["frames"] == "41" and item["first"] == "100", item
    assert item["what"].endswith(" A person's words, whole.") and "built by tester from the table of T" in item["what"] and "3 piece(s) laid" in item["what"], item["what"]
    assert item["id"] == "fun_0100_0140_joined_T_stacked" and item["file"] == "fun_0100_0140_joined_T_STACKED.mp4", item
    return "the table's buttons, the record's rate (23.976), the note at the end"


def the_table_is_handed_what_it_needs() -> str:
    a = src.main.__globals__["argparse"].Namespace(ledger="L.json", shots="S.json", plan="P.json", source="/in/fun.mp4", also_capture=["c1", "c2"],
                                                   prefer=["1-2:lead=x:why"], hold=["3-4:second=y:why"])
    f = src.files_of("fun_0001_0009_joined_T", Path("/clips"), Path("/tables"))
    cmd = src.table_command(a, (1, 9), f, Path("/work/w"))
    given = cmd[cmd.index("--assemble-args") + 1]
    assert cmd[cmd.index("--span") + 1] == "1-9" and given == "--source /in/fun.mp4 --size source --out /clips/fun_0001_0009_joined_T.mp4", (cmd, given)
    assert cmd[cmd.index("--stack-work") + 1] == "/work/w" and cmd[cmd.index("--json") + 1] == "/tables/t_fun_0001_0009_joined_T.json", cmd
    assert [cmd[k + 1] for k, w in enumerate(cmd) if w == "--also-capture"] == ["c1", "c2"] and "1-2:lead=x:why" in cmd and "3-4:second=y:why" in cmd, cmd
    return "span, source, clip name, scratch folder and every handed flag reach the table"


def build_writes_over_nothing() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for role in ("stacked", "table"):
            clips, tables = d / role / "clips", d / role / "tables"
            target = src.files_of("fun_0001_0009_joined_T", clips, tables)[role]
            target.parent.mkdir(parents=True)
            target.write_text("kept")
            try:
                src.main(["build", "--ledger", "L", "--shots", "S", "--plan", "P", "--source", "/in/fun.mp4", "--clips", str(clips), "--tables", str(tables),
                          "--stack-work", str(d / "w"), "--stamp", "T", "20-30", "1-9"])
                raise AssertionError(f"a build over a kept {role} ran")
            except SystemExit as stop:
                assert "a kept output is not replaced" in str(stop) and str(target) in str(stop), stop
            assert target.read_text() == "kept" and not (tables / "t_fun_0020_0030_joined_T.txt").exists(), "something was written before the refusal"
    return "a kept stack or table refuses the whole run before its first span is touched"


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
        base = ["items", "--items", str(page), "--clips", str(clips), "--tables", str(tables), "--stem", "fun", "--stamp", "T", "--by", "tester"]
        assert src.main([*base, "--dry", "100-140", "200-210"]) == 0 and json.loads(page.read_text()) == before and not list(tables.glob("items.before*")), "dry wrote"
        for argv, said in (([*base, "--note", "300-310=words", "100-140"], "not FIRST-LAST=words for one of the spans"),
                           ([*base, "100-140", "400-410"], "not built")):
            try:
                src.main(argv)
                raise AssertionError(f"not refused: {argv}")
            except SystemExit as stop:
                assert said in str(stop), stop
        assert json.loads(page.read_text()) == before, "a refused run wrote"
        assert src.main([*base, "--note", "200-210=the lead's words", "100-140", "200-210"]) == 0
        after = json.loads(page.read_text())
        assert [x["id"] for x in after["items"]] == ["fun_0100_0140_joined_T_stacked", "fun_0200_0210_joined_T_stacked", "old"] and after["by"] == "tester", after["items"]
        assert json.loads((tables / "items.before_tester_T.json").read_text()) == before and after["items"][1]["what"].endswith(" the lead's words"), "copy or note"
        link = page.parent / "media" / "fun_0100_0140_joined_T_STACKED.mp4"
        assert link.is_symlink() and link.resolve() == (clips / "fun_0100_0140_joined_T_STACKED.mp4").resolve(), link
        try:
            src.main([*base, "100-140"])
            raise AssertionError("an id already listed was added again")
        except SystemExit as stop:
            assert "already has an item" in str(stop), stop
        f = src.files_of(src.clip_name("fun", 300, 310, "T"), clips, tables)
        f["stacked"].write_bytes(b"clip")
        f["record"].write_text(json.dumps({**RECORD, "verdict": "FAILS", "verdict_line": "FAILS: nothing laid"}))
        f["json"].write_text(json.dumps(made_table()))
        try:
            src.main([*base, "300-310"])
            raise AssertionError("a clip whose record fails was put on the list")
        except SystemExit as stop:
            assert "no item for a clip that fails" in str(stop), stop
    return "two items at the top in span order, the copy before, the link, the note; dry, a stray note, an unbuilt span, a listed id and a failing record refused"


def by_the_day() -> str:
    t = made_table()
    t["reading"][2]["passed_over"] = [{"render": "third_old", "why": "never stands: no build record"}]
    ledger = {"renders": {
        "lead_a": {"name": "lead_a", "written": "2026-10-14T03:10:00Z", "graph": {"first": 100, "frames": 21}},
        "second_a": {"name": "second_a", "written": "2026-10-13T18:00:00Z", "graph": {"first": 100, "frames": 30}},
        "third_old": {"name": "third_old", "written": "2026-10-13T12:00:00Z", "graph": {}},
        "elsewhere": {"name": "elsewhere", "written": "2026-10-13T13:00:00Z", "graph": {"first": 900, "frames": 10}},
        "next_day": {"name": "next_day", "written": "2026-10-14T06:00:00Z", "graph": {"first": 5, "frames": 10}}}}
    text = src.by_day_text(t, ledger, "2026-10-13", "t.json")
    rows = [x for x in text.splitlines() if x.startswith("| ")]
    assert rows[1].startswith("| 100-120 | lead | lead_a | 10-13 22:10 | the owner's keep |") and rows[2].startswith("| 100-129 | second | second_a | 10-13 13:00 |") \
        and rows[3].startswith("| 121-140 | lead | lead_b | 10-12 10:00 | the newest render in the frozen form"), rows
    assert "3 pieces stand: 2 rendered on 2026-10-13, 1 on other days." in text, text
    assert "| not in its record | third_old | 10-13 07:00 | never stands: no build record |" in text and "| 900-909 | elsewhere | 10-13 08:00 | on no shot of this table |" in text, text
    assert "next_day" not in text and "4 renders of 2026-10-13 in the ledger; 2 of them stand." in text, text
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kept.md"
        out.write_text("kept")
        (Path(tmp) / "t.json").write_text(json.dumps(t))
        (Path(tmp) / "l.json").write_text(json.dumps(ledger))
        try:
            src.main(["by-day", "--table", str(Path(tmp) / "t.json"), "--ledger", str(Path(tmp) / "l.json"), "--day", "2026-10-13", "--out", str(out)])
            raise AssertionError("a kept output was replaced")
        except SystemExit as stop:
            assert "a kept output is not replaced" in str(stop) and out.read_text() == "kept", stop
    return "frame order with why in words; 22:10 Central counts on the 13th; the day's renders that do not stand with why; a kept file stays"


case("the words of an item", words_of_an_item)
case("the item itself", the_item_itself)
case("the table is handed what it needs", the_table_is_handed_what_it_needs)
case("a build writes over nothing", build_writes_over_nothing)
case("items at the top, and a copy first", items_at_the_top)
case("by the day", by_the_day)

if __name__ == "__main__":
    sys.exit(finish())
