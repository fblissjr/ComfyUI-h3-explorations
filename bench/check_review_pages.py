#!/usr/bin/env python3
"""`bench/review_pages.py` on a made table, ledger, review list and verdicts: each case is a way the owner's two local pages could show a dead item, cut a session's word, lose a stopped load, or be unreadable in dark mode.

  both_read_in_dark_mode        no rule of either page names a colour of its own: every colour literal is in the token blocks,
                                every token used is set for light and for dark, and the dark set stands under the media query
                                (unless the page says light) and under data-theme="dark"
  to_review_is_the_tables       only what the current table lays: the newest joined clip of exactly a two-subject shot's frames,
                                and a piece of a one-subject shot, each while the owner has made no choice on it; an open item of
                                an earlier day is not shown; a two-subject shot with no joined clip is said so; each piece with its
                                day, its item's look line and "compare with" the newest older render of that subject on its frames
                                (its item's link, or the file by a relative path); the Open List's link; no typed words
  since_the_last_page           against the previous table: pieces added and left, and the frames a subject gained and lost
  provenance_reads_the_ledger   the stopped loads are the ledger's, each note whole with what of that subject stands there now; a
                                ledger made before it read them says so and not "none"; a reason is never cut; a lane a subject
                                and one for frames inside no piece
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


def render(name, subject, first, last, written, file=None):
    return {"name": name, "file": file or f"Video/v/s/{name}/{name}_00001.mp4", "written": written, "subjects": [subject], "subject": subject,
            "span": [first, last], "graph": {"first": first, "frames": last - first + 1}, "form": {}, "records": [], "making": "recorded, the file agrees"}


def world(d: Path, stopped=True):
    """A table of two shots (100-149: a and b; 150-199: a alone), its ledger, the review list, the verdicts and a source."""
    rows = [{"first": 100, "last": 149, "name": "a_new", "subject": "a", "file": "x"}, {"first": 100, "last": 149, "name": "b_new", "subject": "b", "file": "x"},
            {"first": 150, "last": 199, "name": "a_solo", "subject": "a", "file": "x"}, {"first": 200, "last": 219, "name": "a_kept", "subject": "a", "file": "x"},
            {"first": 220, "last": 239, "name": "a_x", "subject": "a", "file": "x"}]
    stand = lambda n, f, l, why="step 2: the plan's region": {"render": n, "frames": [[f, l]], "written": "2026-10-14T15:00:00Z", "why": why}
    table = {"made": "from the ledger `l.json`", "rows": rows, "sources": [], "gaps": [], "not_in_the_shot": [],
             "reading": [{"first": 100, "last": 149, "subject": "a", "present": "yes", "stands": [stand("a_new", 100, 149, "step 0: named by --prefer: " + LONG)]},
                         {"first": 100, "last": 149, "subject": "b", "present": "unsure", "stands": [stand("b_new", 100, 149)]},
                         {"first": 150, "last": 199, "subject": "a", "present": "yes", "stands": [stand("a_solo", 150, 199)]},
                         {"first": 150, "last": 199, "subject": "b", "present": "no", "stands": []},
                         {"first": 200, "last": 219, "subject": "a", "present": "yes", "stands": [stand("a_kept", 200, 219)]},
                         {"first": 220, "last": 239, "subject": "a", "present": "yes", "stands": [stand("a_x", 220, 239)]},
                         {"first": 220, "last": 239, "subject": "b", "present": "yes", "stands": []}]}
    renders = [render("a_new", "a", 100, 149, "2026-10-14T15:00:00Z"), render("b_new", "b", 100, 149, "2026-10-13T15:00:00Z"),
               render("a_solo", "a", 150, 199, "2026-10-12T15:00:00Z"), render("a_kept", "a", 200, 219, "2026-10-12T15:00:00Z"),
               render("a_x", "a", 220, 239, "2026-10-14T15:00:00Z"), render("a_old", "a", 90, 130, "2026-10-10T15:00:00Z"), render("a_older", "a", 100, 149, "2026-10-09T15:00:00Z"),
               render("b_old", "b", 100, 149, "2026-10-11T15:00:00Z", file="Video/v/s/b_old.mp4")]
    ledger = {"renders": {r["file"]: r for r in renders}}
    if stopped:
        ledger["stopped"] = [{"load": "b_fp_0220", "subject": "b", "frames": [220, 239], "job_file": "data/x/b_fp_0220.json", "note": "data/x/b_fp_0220.STOPPED.txt",
                              "note_written": "2026-10-14T14:00:00Z", "note_says": "STOPPED AT ITS GATE.\nThe whole note, its second line too."},
                             {"load": "a_fp_0100", "subject": "a", "frames": [100, 149], "job_file": "j", "note": "n", "note_written": "t", "note_says": "an older stop"}]
    items = [{"id": "v_0100_0149_joined_NEW_stacked", "first": "100", "frames": "50", "state": "open"},
             {"id": "v_0100_0149_joined_OLD_stacked", "first": "100", "frames": "50", "state": "open"},
             {"id": "a_new", "first": "100", "frames": "50", "state": "open", "look": "her mouth on 120-130"},
             {"id": "a_old", "first": "90", "frames": "41", "state": "open"},
             {"id": "a_solo", "first": "150", "frames": "50", "state": "open", "look": "her eyes on 160"},
             {"id": "a_kept", "first": "200", "frames": "20", "state": "open"},
             {"id": "dead_of_10_11", "first": "10", "frames": "5", "state": "open", "look": "A DEAD ITEM"}]
    verdicts = [{"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"}]
    previous = {"rows": [{"first": 100, "last": 149, "name": "a_older", "subject": "a"}, {"first": 150, "last": 209, "name": "a_solo", "subject": "a"},
                         {"first": 300, "last": 309, "name": "a_gone", "subject": "a"}], "reading": []}
    (d / "t.json").write_text(json.dumps(table))
    (d / "l.json").write_text(json.dumps(ledger))
    (d / "items.json").write_text(json.dumps({"items": items}))
    (d / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    (d / "t_before.json").write_text(json.dumps(previous))
    src = d / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x48:r=24000/1001", "-frames:v", "2", "-y", str(src)], check=True)
    (d / "share").mkdir(exist_ok=True)
    (d / "day").mkdir(exist_ok=True)
    return ["--table", str(d / "t.json"), "--ledger", str(d / "l.json"), "--verdicts", str(d / "verdicts.jsonl"), "--items", str(d / "items.json"),
            "--source", str(src), "--share", str(d / "share")]


def to_review(d: Path, more=()) -> str:
    out = d / "day" / "TO_REVIEW.html"
    assert rp.main(["to-review", *world(d), "--page-url", "http://the-page/#", "--open-list", "https://the-open-list", "--out", str(out), *more]) == 0
    return out.read_text()


def provenance(d: Path, stopped=True) -> str:
    out = d / "day" / "PROVENANCE.html"
    assert rp.main(["provenance", *world(d, stopped), "--day-folder", str(d / "day"), "--up-to", "239", "--watched-table", str(d / "t_before.json"),
                    "--out", str(out)]) == 0
    return out.read_text()


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", page, flags=re.S))))


def both_read_in_dark_mode() -> str:
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        for page in (to_review(Path(t1)), provenance(Path(t2))):
            css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
            blocks = re.findall(r'(:root(?::not\(\[data-theme="light"\]\)|\[data-theme="dark"\])?)\{([^{}]*)\}', css)
            light = next(b for sel, b in blocks if sel == ":root")
            darks = [b for sel, b in blocks if sel != ":root"]
            assert '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){' in css and len(darks) == 2 and darks[0] == darks[1], blocks
            defined = lambda b: set(re.findall(r"--([\w-]+):", b))
            used = set(re.findall(r"var\(--([\w-]+)\)", css))
            assert used <= defined(light) and used <= defined(darks[0]), used - (defined(light) & defined(darks[0]))
            rest = css
            for _, b in blocks:
                rest = rest.replace(b, "")
            stray = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", rest)
            assert not stray, f"a rule names a colour of its own: {stray}"
    return "every colour of both pages is a token, every token used is set for light and for dark, and the dark set stands under both guards"


def to_review_is_the_tables() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page = to_review(d)
        text = text_of(page)
        assert 'href="http://the-page/#v_0100_0149_joined_NEW_stacked"' in page and "v_0100_0149_joined_OLD" not in page, "not the newest joined clip"
        assert "A DEAD ITEM" not in text and "dead_of_10_11" not in page, "an open item the table does not lay is shown"
        assert "a_kept" not in text.split("Pieces in a shot of one of them")[1], "a piece the owner chose on is shown"
        assert 'href="http://the-page/#a_solo"' in page and "look: her eyes on 160" in text, text
        assert "look: her mouth on 120-130" in text and "220-239 (a, b): no joined clip of this shot is on the list" in text, text
        # compare with: the newest older render of the subject on its frames; its item where the list has one, else its file by a relative path
        assert re.search(r'a_new.*?compare with: <a href="http://the-page/#a_old">a_old</a> \(rendered 10/10\)', page, re.S), page
        assert 'compare with: <a href="../share/Video/v/s/b_old.mp4">b_old</a> (rendered 10/11)' in page, page
        assert '<a href="https://the-open-list">the Open List</a>' in page and "10/14" in text and "10/13" in text and "10/12" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        world(d)
        (d / "verdicts.jsonl").write_text(json.dumps({"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}) + "\n")
        args = world(d)
        (d / "verdicts.jsonl").write_text(json.dumps({"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}) + "\n")
        out = d / "day" / "TO_REVIEW.html"
        rp.main(["to-review", *args, "--page-url", "p#", "--open-list", "o", "--out", str(out)])
        assert "joined_NEW" not in out.read_text() and "None waiting on you." in text_of(out.read_text()), "a joined clip the owner chose on is shown"
    return ("the newest joined clip of a two-subject shot and the one-subject pieces the owner has not chosen on, nothing older; an unbuilt joined "
            "clip said; each piece's look line and what it replaces (its item's link, else its file); the Open List's link")


def since_the_last_page() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        text = text_of(to_review(d, ["--previous-table", str(d / "t_before.json")]))
        part = text.split("Since the last page")[1].split("Shots with two")[0]
        for said in ("added: a_new (a)", "added: b_new (b)", "added: a_kept (a)", "added: a_x (a)", "left: a_older (a)", "left: a_gone (a)",
                     "a: a piece now on 210-239", "a: no piece now on 300-309", "b: a piece now on 100-149"):
            assert said in part, (said, part)
        assert "a_solo" not in part, part
        assert "Since the last page" not in text_of(to_review(Path(tempfile.mkdtemp()))), "the part is made with no previous table"
    return "pieces added and left, and the frames each subject gained and lost, against the previous table; no part without one"


def provenance_reads_the_ledger() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        page = provenance(d)
        text = text_of(page)
        assert "b_fp_0220" in text and "What of b stands on those frames now: nothing" in text and "The whole note, its second line too." in page, text
        assert "What of a stands on those frames now: a_new" in text, text
        changed = text.split("What changes in the next clip")[1].split("Loads stopped before a render")[0]
        stretches = text.split("Every stretch, a person at a time")[1].split("Each piece and where it came from")[0]
        assert LONG in changed and LONG in stretches, "a reason was cut"
        assert page.count('class="tl">') >= 2 and "a</div><div" in page.replace("\n", "") and "inside no piece" in text, "the lanes"
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp), stopped=False))
        assert "made before it read the stopped loads: none is shown, and that is not none" in text, text
    return "each stopped load's note whole with what stands there now; a ledger without them says not known; reasons whole; a lane a subject"


def nothing_of_a_video_in_code() -> str:
    code = TOOL.read_text()
    for word in ("/mnt/", "/home/", "http://", "https://", ":8765", "100.87.", "fun_", '"fun"', "day_2026"):
        assert word not in code, f"the tool's source holds {word!r}"
    return "no address, share, machine folder, day or video in the tool's source"


case("both read in dark mode", both_read_in_dark_mode)
case("to-review is the table's", to_review_is_the_tables)
case("since the last page", since_the_last_page)
case("provenance reads the ledger", provenance_reads_the_ledger)
case("nothing of a video in the code", nothing_of_a_video_in_code)

if __name__ == "__main__":
    sys.exit(finish())
