#!/usr/bin/env python3
"""`bench/review_pages.py` on a made table, ledger, review list and verdicts: each case is a way the owner's two local pages could hide a live piece, show a dead one, put a session's word in the owner's mouth, cut a reason, lose a stopped load, or be unreadable in either mode.

  both_modes_read               no colour anywhere on either page but the tokens (no hex, rgb, hsl or named colour in a
                                rule, a style= or an SVG attribute); every token used set for light and for dark, the dark
                                set under the media query and under data-theme="dark"; every text token at least 4.5:1
                                against the page and the card in both modes (WCAG)
  to_review_is_the_tables       only what the current table lays and the owner has not settled (a keep or "not the one"); a
                                "not sure" stays, with the owner's words since; a joined clip shown only when its own table
                                lays the same pieces on the shot, else, and where none is built, each piece on its own with
                                why; a live piece never vanishes; an item found by its `name`; the card's look line said as
                                a session's words; what a piece replaces from the previous table, or not known without one
  since_the_last_page           against the previous table: pieces added and left, the frames a subject gained and lost
  provenance_reads_the_records  the stopped loads are the ledger's, each note whole with what stands on which of its frames;
                                a ledger made before it read them says so; the lead's --prefer said as a session's word in the
                                stretches and on the card; no reason cut; a frame left as the original says the ledger's
                                reason where no look's word covers it; the folder link relative to the page and its absence
                                said; the verdicts record's words labelled as the record's; a lane a subject, in order
  another_ledger_is_refused     a ledger the table does not name
  nothing_of_a_video_in_code    the tool's source names no address, share, machine folder or video

    <comfy venv python> bench/check_review_pages.py

Needs ffmpeg and ffprobe (a two-frame clip gives the source's rate). No model, no card, no server.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

import review_pages as rp  # noqa: E402

TOOL = Path(rp.__file__)
LONG = ("the lead's --prefer of a remake over the owner's keep of 10/12, a reason long enough that any cut of it would show: "
        "the owner's word of 2026-10-14, about 14:43Z, inbox note owner_terminal, the 10/12 pieces are not trusted and each is made again, "
        "so it stands over the owner's keep until the owner says otherwise, which they have not done yet, said whole to the end")
NAMED = ["white", "black", "red", "blue", "green", "gray", "grey", "yellow", "orange", "purple", "silver"]


def render(d: Path, name, subject, first, last, written, file=None, left=None, left_out=None):
    job = d / "data" / f"x_{name}_job"
    job.mkdir(parents=True, exist_ok=True)
    (job / "job.render.json").write_text(json.dumps({"job": {"left_out": left_out or []}}))
    return {"name": name, "file": file or f"Video/v/s/{name}/{name}_00001.mp4", "written": written, "subjects": [subject], "subject": subject,
            "span": [first, last], "graph": {"first": first, "frames": last - first + 1}, "form": {}, "making": "recorded, the file agrees",
            "records": [{"built_file": str(job / f"{name}.render.built.json")}],
            "leaves_as_the_original": {"source_frames": left or [], "read_from": "the capture's manifest: the frames on which it emptied this subject's part"}}


def world(d: Path, stopped=True, joined_table="same", verdicts=None, ledger_name="l.json"):
    """Shots: 100-149 (a and b), 150-199 (a), 200-219 (a), 220-239 (a and b, no joined clip), 240-259 (a, b: a joined clip of an older table)."""
    rows = [{"first": 100, "last": 149, "name": "a_new", "subject": "a", "file": "x"}, {"first": 100, "last": 149, "name": "b_new", "subject": "b", "file": "x"},
            {"first": 150, "last": 199, "name": "a_solo", "subject": "a", "file": "x"}, {"first": 200, "last": 219, "name": "a_kept", "subject": "a", "file": "x"},
            {"first": 220, "last": 239, "name": "a_x", "subject": "a", "file": "x"}, {"first": 240, "last": 259, "name": "a_y", "subject": "a", "file": "x"}]
    stand = lambda n, f, l, step=2, led=None: {"render": n, "frames": [[f, l]], "written": "2026-10-14T15:00:00Z", "step": step, "the_leads_prefer": led,
                                               "why": ("step 0: named by --prefer: " + led) if led else "step 2: the plan's region"}
    reading = [{"first": 100, "last": 149, "subject": "a", "present": "yes", "stands": [stand("a_new", 100, 149, 0, LONG)]},
               {"first": 100, "last": 149, "subject": "b", "present": "unsure", "stands": [stand("b_new", 100, 149)]},
               {"first": 150, "last": 199, "subject": "a", "present": "yes", "stands": [stand("a_solo", 150, 199)]},
               {"first": 150, "last": 199, "subject": "b", "present": "no", "stands": []},
               {"first": 200, "last": 219, "subject": "a", "present": "yes", "stands": [stand("a_kept", 200, 219)]},
               {"first": 220, "last": 239, "subject": "a", "present": "yes", "stands": [stand("a_x", 220, 239)]},
               {"first": 220, "last": 239, "subject": "b", "present": "yes", "stands": [],
                "passed_over": [{"render": "b_owner_kept", "why": "THE TABLE'S REASON FOR B_OWNER_KEPT",
                                 "the_owners_last_choice": {"said": "keep this one", "at": "2026-10-12T20:00:00Z"}}]},
               {"first": 240, "last": 259, "subject": "a", "present": "yes", "stands": [stand("a_y", 240, 259)]},
               {"first": 240, "last": 259, "subject": "b", "present": "yes", "stands": []}]
    table = {"made": f"from the ledger `{d / ledger_name}` (...)", "rows": rows, "sources": [], "gaps": [], "not_in_the_shot": [], "reading": reading,
             "held": [{"first": 220, "last": 239, "subject": "b", "load": "b_held_load", "reason": "THE LEAD'S HOLD, whole"}]}
    renders = [render(d, "a_new", "a", 100, 149, "2026-10-14T15:00:00Z", left=[[120, 122]]), render(d, "b_new", "b", 100, 149, "2026-10-13T15:00:00Z"),
               render(d, "a_solo", "a", 150, 199, "2026-10-12T15:00:00Z", left=[[160, 161]], left_out=[{"subject": "a", "first": 161, "frames": 1, "why": "A LOOK'S WORD"}]),
               render(d, "a_kept", "a", 200, 219, "2026-10-12T15:00:00Z"), render(d, "a_x", "a", 220, 239, "2026-10-14T15:00:00Z"),
               render(d, "a_y", "a", 240, 259, "2026-10-14T15:00:00Z"), render(d, "a_old", "a", 90, 130, "2026-10-10T15:00:00Z"),
               render(d, "a_older", "a", 100, 149, "2026-10-09T15:00:00Z"), render(d, "kitchen", "b", 100, 149, "2026-10-10T15:00:00Z", file="Video/v/kitchen.mp4")]
    ledger = {"renders": {r["file"]: r for r in renders}}
    if stopped:
        ledger["stopped"] = [{"load": "b_fp_0220", "subject": "b", "frames": [220, 239], "job_file": "j", "note": "data/x/b_fp_0220.STOPPED.txt",
                              "note_written": "2026-10-14T14:00:00Z", "note_says": "STOPPED AT ITS GATE.\nThe whole note, its second line too."},
                             {"load": "a_fp_0140", "subject": "a", "frames": [140, 160], "job_file": "j", "note": "n", "note_written": "t", "note_says": "an older stop"}]
    items = [{"id": "v_0100_0149_joined_NEW_stacked", "first": "100", "frames": "50", "state": "open"},
             {"id": "v_0100_0149_joined_OLD_stacked", "first": "100", "frames": "50", "state": "open"},
             {"id": "v_0240_0259_joined_STALE_stacked", "first": "240", "frames": "20", "state": "open"},
             {"id": "a_new", "first": "100", "frames": "50", "state": "open", "look": "her mouth on 120-130"},
             {"id": "r__s__kitchen", "name": "kitchen", "first": "100", "frames": "50", "state": "ok"},
             {"id": "r__s__a_solo", "name": "a_solo", "first": "150", "frames": "50", "state": "open", "look": "her eyes on 160"},
             {"id": "a_kept", "first": "200", "frames": "20", "state": "open"}, {"id": "a_x", "first": "220", "frames": "20", "state": "open"},
             {"id": "a_y", "first": "240", "frames": "20", "state": "open", "look": "THE LOOK OF A_Y"},
             {"id": "dead_of_10_11", "first": "150", "frames": "50", "state": "open", "look": "A DEAD ITEM"}]
    lines = verdicts if verdicts is not None else [
        {"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
        {"what": "choice", "row": "a_x", "chosen": "not sure", "at": "2026-10-13T10:00:00Z"},
        {"what": "words", "row": "a_x", "words": "i cant choose between them", "at": "2026-10-13T11:00:00Z"},
        {"what": "words", "row": "a_new", "words": "SESSION: is this right? OWNER: yes", "at": "2026-10-13T12:00:00Z"}]
    previous = {"rows": [{"first": 100, "last": 149, "name": "a_older", "subject": "a"}, {"first": 100, "last": 149, "name": "kitchen", "subject": "b"},
                         {"first": 150, "last": 209, "name": "a_solo", "subject": "a"}, {"first": 300, "last": 309, "name": "a_gone", "subject": "a"}], "reading": []}
    tables = d / "tables"
    tables.mkdir(exist_ok=True)
    lays = [r for r in rows if r["first"] <= 149 and r["last"] >= 100]
    (tables / "t_v_0100_0149_joined_NEW.json").write_text(json.dumps({"rows": lays if joined_table == "same" else lays[:1]}))
    (tables / "t_v_0240_0259_joined_STALE.json").write_text(json.dumps({"rows": [{"first": 240, "last": 259, "name": "a_older_y", "subject": "a"}]}))
    (d / "t.json").write_text(json.dumps(table))
    (d / ledger_name).write_text(json.dumps(ledger))
    (d / "items.json").write_text(json.dumps({"items": items}))
    (d / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in lines))
    (d / "t_before.json").write_text(json.dumps(previous))
    src = d / "v.mp4"
    if not src.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x48:r=24000/1001", "-frames:v", "2", "-y", str(src)], check=True)
    (d / "share").mkdir(exist_ok=True)
    (d / "day" / "0150_a_solo").mkdir(parents=True, exist_ok=True)
    (d / "day" / "0150_a_solo" / "text.txt").write_text("a text")
    return ["--table", str(d / "t.json"), "--ledger", str(d / ledger_name), "--verdicts", str(d / "verdicts.jsonl"), "--items", str(d / "items.json"),
            "--source", str(src), "--share", str(d / "share")]


def to_review(d: Path, more=(), **kw) -> str:
    out = d / "day" / "TO_REVIEW.html"
    assert rp.main(["to-review", *world(d, **kw), "--page-url", "p#", "--open-list", "https://the-open-list", "--tables", str(d / "tables"),
                    "--out", str(out), *more]) == 0
    return out.read_text()


def provenance(d: Path, stopped=True, out_at="day") -> str:
    out = d / out_at / "PROVENANCE.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert rp.main(["provenance", *world(d, stopped), "--day-folder", str(d / "day"), "--up-to", "259", "--watched-table", str(d / "t_before.json"),
                    "--out", str(out)]) == 0
    return out.read_text()


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", page, flags=re.S))))


def luminance(hexa: str) -> float:
    c = [int(hexa.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def both_modes_read() -> str:
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        for page in (to_review(Path(t1)), provenance(Path(t2))):
            css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
            blocks = re.findall(r'(:root(?::not\(\[data-theme="light"\]\)|\[data-theme="dark"\])?)\{([^{}]*)\}', css)
            light = next(b for sel, b in blocks if sel == ":root")
            darks = [b for sel, b in blocks if sel != ":root"]
            assert '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){' in css and len(darks) == 2 and darks[0] == darks[1], blocks
            defined = lambda b: dict(re.findall(r"--([\w-]+):([^;]+)", b))
            used = set(re.findall(r"var\(--([\w-]+)\)", page))
            assert used <= set(defined(light)) and used <= set(defined(darks[0])), used
            rest = css
            for _, b in blocks:
                rest = rest.replace(b, "")
            stray = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", rest) \
                + re.findall(r'(?:style|fill|stroke|color)="[^"]*(?:#[0-9a-fA-F]{3,8}|rgba?\(|\b(?:' + "|".join(NAMED) + r')\b)[^"]*"', page) \
                + re.findall(r"(?:color|background|fill|stroke|border[\w-]*):\s*(?:" + "|".join(NAMED) + r")\b", css)
            assert not stray, f"a colour of its own: {stray}"
    for k in ("fg", "mut", "link", "gap", *rp.DAY_TOKENS):
        for mode in (0, 1):
            for ground in ("bg", "card"):
                r = contrast(rp.TOKENS[k][mode], rp.TOKENS[ground][mode])
                assert r >= 4.5, f"{k} on {ground} in {'dark' if mode else 'light'} mode: {r:.2f}:1"
        assert rp.TOKENS[k][0] != rp.TOKENS[k][1], f"{k} is the same in both modes"
    # every token a timeline shape is drawn in, at least 3:1 on the page and the card, and drawn solid (a reader, 2026-10-14: `left` at .55 opacity)
    for k in ("left", "bare", "gap", *rp.DAY_TOKENS):
        for mode in (0, 1):
            for ground in ("bg", "card"):
                r = contrast(rp.TOKENS[k][mode], rp.TOKENS[ground][mode])
                assert r >= 3, f"the shape token {k} on {ground} in {'dark' if mode else 'light'} mode: {r:.2f}:1"
    assert "opacity" not in rp.PROVENANCE_CSS + rp.BASE_CSS, "a shape is drawn see-through: its contrast is not its token's"
    return ("nothing but tokens names a colour; every text token at least 4.5:1 on the page and the card in light and dark, and differs between them; "
            "every shape token at least 3:1, drawn solid")


def to_review_is_the_tables() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page = to_review(d, ["--previous-table", str(d / "t_before.json")])
        text = text_of(page)
        assert 'href="p#v_0100_0149_joined_NEW_stacked"' in page and "joined_OLD" not in page, "not the newest joined clip of this table"
        assert "A DEAD ITEM" not in text and "dead_of_10_11" not in page, "an open item the table does not lay is shown"
        assert "a_kept" not in text.split("Shots with two or more")[1], "a piece the owner kept is shown"
        # "not sure" stays, with the owner's words since; a shot with no joined clip shows its piece on its own, with why
        assert ("your last click: not sure (10/13 05:00); the words typed in its box (they may quote a session's question) since (10/13 06:00): "
                "\"i cant choose between them\"") in text, text
        assert "The shot 220-239" in text and "no joined clip of this shot is on the list" in text, text
        # a joined clip of an older table: said, and its piece shown on its own with its look line
        assert "v_0240_0259_joined_STALE_stacked) lays a_older_y, and this table lays a_y" in text and "THE LOOK OF A_Y" in text, text
        # an item found by its name; the look line said as the card's; what it replaces, from the previous table
        assert 'href="p#r__s__a_solo"' in page and "the card's look line (a session's words): her eyes on 160" in text, text
        assert "it replaces, in the previous table: a_older (rendered 10/09) on 100-149" in text, text
        assert "it replaces, in the previous table: nothing: this same piece stood here in the previous table on 150-199" in text, text
        assert 'it replaces, in the previous table: <a href="p#r__s__kitchen">kitchen</a> (rendered 10/10) on 100-149' in page, page
        assert '<a href="https://the-open-list">the Open List</a>' in page
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(to_review(Path(tmp)))
        assert "it replaces, in the previous table: not known here: no previous table was given" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        # THE VANISHING PIECE (a reader, 2026-10-14): the joined clip of 100-149 is of another table and the owner kept it: a_new must still show
        text = text_of(to_review(Path(tmp), joined_table="other",
                                 verdicts=[{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}]))
        assert "a_new" in text and "lays a_new, and this table lays a_new, b_new" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(to_review(Path(tmp), verdicts=[{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}]))
        assert "joined_NEW" not in text and "The shot 100-149" not in text, "a joined clip of this table the owner kept is shown"
    return ("only what this table lays and the owner has not settled; not sure stays with the words since; a joined clip only when its own table lays the "
            "same pieces, else each piece on its own with why, so a piece never vanishes; items by name; the look line a session's; what each replaces")


def the_owners_words_rule() -> str:
    """A fourth reader (2026-10-14): words after a keep were ignored, a line a session took down was labelled as typed in
    the box, a "wrong" on a joined clip hid the shot's pieces, and the lead's --hold and the owner's keeps the table
    does not lay were on neither page."""
    words_after_keep = [{"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
                        {"what": "words", "row": "a_kept", "words": "ACTUALLY HER CHIN IS OFF", "at": "2026-10-13T11:00:00Z", "source": "answers.jsonl:9"},
                        {"what": "words", "row": "a_solo", "words": "SAID IN THE TERMINAL", "at": "2026-10-13T12:00:00Z",
                         "source": "the lead's terminal (mrx), 2026-10-13", "note": "taken down whole by mrx"},
                        {"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "wrong", "at": "2026-10-14T16:00:00Z"}]
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(to_review(Path(tmp), verdicts=words_after_keep))
        # THE CASES THAT GO RED: a keep followed by words is to review again; a session's taking-down is said as such
        assert "a_kept" in text and "ACTUALLY HER CHIN IS OFF" in text, "words after a keep were ignored"
        assert ("your words as a session took them down (the lead's terminal (mrx), 2026-10-13; taken down whole by mrx) since (10/13 07:00): "
                "\"SAID IN THE TERMINAL\"") in text and "typed in its box (they may quote a session's question) since (10/13 07:00): \"SAID" not in text, text
        assert "The shot 100-149" in text and "your last click: wrong" in text and "a_new" in text.split("The shot 100-149")[1], "a wrong on the joined clip hid the shot"
        assert "HELD BACK BY THE LEAD, a session's word and not yours: THE LEAD'S HOLD, whole" in text and "b_held_load" in text, text
        assert "your keep of b_owner_kept (keep this one, 10/12 15:00) is not laid here. The table's reason: THE TABLE'S REASON FOR B_OWNER_KEPT" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp)))
        assert "HELD BACK BY THE LEAD" in text and "your keep of b_owner_kept" in text, "provenance does not say the hold or the keep"
        assert "a session's word in the load's job file (its `left_out`, a look's): A LOOK'S WORD" in text, text
        assert "The note, whole, a session's words (data/x/b_fp_0220.STOPPED.txt" in text, text
    return ("a keep followed by the owner's words is shown again with them; a line a session took down says so and who; a wrong on a joined clip leaves the "
            "shot with its pieces; the lead's holds and the owner's keeps the table does not lay are on both pages; a look's word and a stopped note say whose")

def since_the_last_page() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        text = text_of(to_review(d, ["--previous-table", str(d / "t_before.json")]))
        part = text.split("Since the last page")[1].split("Shots with two")[0]
        for said in ("added: a_new (a)", "added: b_new (b)", "added: a_kept (a)", "added: a_x (a)", "left: a_older (a)", "a: a piece now on 210-259",
                     "on the frames both tables cover: 100-259 (this one covers 100-259, that one 100-309)"):
            assert said in part, (said, part)
        # THE CASE THAT GOES RED when a table of other frames makes what lies outside this one's frames "left" (a reader, 2026-10-14)
        assert "a_solo" not in part and "a_gone" not in part and "300-309" not in part, part
        # against a narrower table, nothing outside its frames is "added"
        (d / "t_narrow.json").write_text(json.dumps({"rows": [{"first": 150, "last": 199, "name": "a_solo", "subject": "a"}],
                                                     "reading": [{"first": 150, "last": 199, "subject": "a"}]}))
        part = text_of(to_review(d, ["--previous-table", str(d / "t_narrow.json")])).split("Since the last page")[1].split("Shots with two")[0]
        assert "no piece added" in part and "no piece left" in part and "150-199 (this one covers 100-259, that one 150-199)" in part, part
    with tempfile.TemporaryDirectory() as tmp:
        assert "Since the last page" not in text_of(to_review(Path(tmp))), "the part is made with no previous table"
    return ("pieces added and left, and the frames each subject gained and lost, on the frames both tables cover and said so; against a narrower table "
            "nothing outside it is added; no part without one")


def provenance_reads_the_records() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page = provenance(d, out_at="pages")
        text = text_of(page)
        assert "b_fp_0220" in text and "What of b stands on those frames now: nothing" in text and "The whole note, its second line too." in page, text
        assert "What of a stands on those frames now: a_new on 140-149, a_solo on 150-160" in text, text
        changed = text.split("What changes in the next clip")[1].split("Loads stopped before a render")[0]
        stretches = text.split("Every stretch, a person at a time")[1].split("Each piece and where it came from")[0]
        cards = text.split("Each piece and where it came from")[1]
        for where in (changed, stretches, cards):
            assert LONG in where, "a reason was cut"
        for where in (stretches, cards):
            assert f"THE LEAD'S CHOICE (--prefer), a session's word: {LONG}" in where, where
        # frames left as the original: a look's word where one covers them, else the ledger's reason
        assert "A LOOK'S WORD" in stretches and "its load emptied the part here, by the ledger" in stretches, stretches
        assert 'href="../day/0150_a_solo/"' in page and "no folder of it in the day's folder" in text, "the folder link"
        assert "The verdicts record's words on it, whole" in text and "Your words on it" not in text and "SESSION: is this right? OWNER: yes" in text, text
        lanes = re.findall(r'<div style="height:\d+px;line-height:\d+px">([^<]+)</div>', page)
        assert lanes == ["a", "b", "inside no piece"], lanes
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp), stopped=False))
        assert "made before it read the stopped loads: none is shown, and that is not none" in text, text
    return ("stopped notes whole with which frames of what stands; the lead's word said as a session's in stretches and cards; no reason cut; a look's "
            "word or the ledger's reason for frames left; links relative to the page; the record's words labelled; the lanes in order")


def another_ledger_is_refused() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        args = world(d)
        (d / "other.json").write_text((d / "l.json").read_text())
        args[args.index("--ledger") + 1] = str(d / "other.json")
        try:
            rp.main(["to-review", *args, "--page-url", "p#", "--open-list", "o", "--out", str(d / "x.html")])
        except SystemExit as stop:
            assert "made from another ledger" in str(stop), stop
        else:
            raise AssertionError("a ledger the table does not name was read")
        # one of the same name in another folder is another ledger
        (d / "elsewhere").mkdir()
        (d / "elsewhere" / "l.json").write_text((d / "l.json").read_text())
        args[args.index("--ledger") + 1] = str(d / "elsewhere" / "l.json")
        try:
            rp.main(["to-review", *args, "--page-url", "p#", "--open-list", "o", "--out", str(d / "x.html")])
        except SystemExit as stop:
            assert "made from another ledger" in str(stop), stop
        else:
            raise AssertionError("a ledger of the same name in another folder was read")
    return "a ledger the table was not made from is refused, one of the same name in another folder too"


def nothing_of_a_video_in_code() -> str:
    code = TOOL.read_text()
    for word in ("/mnt/", "/home/", "http://", "https://", ":8765", "100.87.", "fun_", '"fun"', "day_2026"):
        assert word not in code, f"the tool's source holds {word!r}"
    return "no address, share, machine folder, day or video in the tool's source"


case("both modes read", both_modes_read)
case("to-review is the table's, and hides no live piece", to_review_is_the_tables)
case("since the last page", since_the_last_page)
case("the owner's words rule, and whose words are whose", the_owners_words_rule)
case("provenance reads the records", provenance_reads_the_records)
case("another ledger is refused", another_ledger_is_refused)
case("nothing of a video in the code", nothing_of_a_video_in_code)

if __name__ == "__main__":
    sys.exit(finish())
