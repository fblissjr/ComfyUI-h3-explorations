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
  waiting_on_you_and_done       an entry with no click of the owner's waits on them; one with a click is under Done, a closed
                                fold, with the click; a piece settled on --today is under Done, one settled before on neither
  the_sixth_readers_findings    a wrong and words on an older joined clip of the same pieces; words alone and a praised keep under
                                Done on their day and gone after; a joined keep hides no later line on a piece; Central days; a joined
                                clip on other frames; section headings; where words came from; the stacked clip from any folder
  the_seventh_readers_findings  a not sure on a joined clip after a keep stays; words on a joined clip; a joined clip's last line, in
                                Central; the counts; which piece; no source; a piece across a shot's edge; the newest look line
  the_eighth_readers_findings   a line's frame; the folders read; every piece kept with the clip unwatched; the last line's day; a
                                wrong on a clip; times in Central; a session's source quoting the answers file
  every_heading_is_its_way_in   each entry's heading links the joined clip or each standing piece, named; a piece with no item says so
  the_owners_words_rule         a keep followed by the owner's words is shown again; a line a session took down said so; a wrong
                                on a joined clip keeps the shot; the lead's holds and the owner's keeps not laid on both pages
  the_fifth_readers_findings    words just before the last click go with it; an older line written later does not undo a keep; a
                                line on an item's own id is read; a keep on a joined clip counts on each piece it lays; unsure
                                subjects marked; a passed-over wrong never called a keep
  a_fifth_readers_mutations     a joined clip of fewer frames on the same first frame is not the shot's; the newest day is the
                                first colour; the change table keeps each reason
  another_ledger_is_refused     a ledger the table does not name, by its path
  nothing_of_a_video_in_code    the tool's source names no address, share, machine folder or video
  the_gap_check_is_on_both_pages both pages read `assembly.stack.coverage_json` (below Start here on to-review, at the top
                                of provenance) and show `remake` and `capture` lists apart, worst shots first, with all
                                nine kinds, per-subject gap spans, seam counts, and string `failures` read and never recomputed
  missing_or_stale_gap_check_is_refused both pages refuse when `coverage_json` is omitted or missing, when `table_sha256`,
                                `by_shot`, or `span` mismatches, when `assembly.command` lacks `--out`, or when `clip_check`
                                is missing, missing on disk, or mismatches `--out` or its file's sha256, writing nothing
  compare_clips_say_who_is_new_and_who_is_the_old_cut above each compare clip's player, one generated line per person in the
                                shot (`new here: <load>`, `the current cut: <load>, laid <day>`, or `the original: no piece`),
                                read from `.stack.json`/`.stack.frames.jsonl` or sidecars and the table/ledger, never typed

    <comfy venv python> bench/check_review_pages.py

Needs ffmpeg and ffprobe (a two-frame clip gives the source's rate). No model, no card, no server.
"""
from __future__ import annotations

import html
import json
import os
import re
import hashlib
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


def sync_gap_check(d: Path, by_shot: dict | None = None, runs: list | None = None, failures: list | None = None, span=None) -> Path:
    """Write or refresh `d / 'frame_coverage_t.json'` so its `table_sha256` and `clip_check.sha256` match `d / 't.json'` and `d / 'joined.mp4.check.json'`."""
    cc_file = d / "joined.mp4.check.json"
    if not cc_file.is_file():
        cc_file.write_text(json.dumps({"table": []}))
    cov_path = d / "frame_coverage_t.json"
    prev = json.loads(cov_path.read_text()) if cov_path.is_file() else {}
    default_by_shot = {
        "remake": [
            {"shot": [220, 239], "frames": 14, "subjects": ["a", "b"],
             "by_kind": {"captured_not_rendered": 3, "rendered_not_kept": 2, "emptied_inside": 4, "laid_but_barely_changed": 3, "region_far_below_its_own_size": 2}},
            {"shot": [150, 199], "frames": 4, "subjects": ["a"],
             "by_kind": {"captured_not_rendered": 2, "rendered_not_kept": 0, "emptied_inside": 2, "laid_but_barely_changed": 0, "region_far_below_its_own_size": 0}},
        ],
        "capture": [
            {"shot": [240, 259], "frames": 20, "subjects": ["a", "b"], "by_kind": {"no_capture": 20}},
            {"shot": [220, 239], "frames": 10, "subjects": ["b"], "by_kind": {"no_capture": 10}},
        ],
    }
    default_runs = [
        {"first": 225, "last": 227, "shot": [220, 239], "subject": "b", "kind": "captured_not_rendered", "pieces": [], "captures": ["b_cap"], "mask_files": ["data/b_cap/subjects/b/masks__sam.npz"], "cause": "captured, not rendered: b_cap"},
        {"first": 228, "last": 229, "shot": [220, 239], "subject": "b", "kind": "rendered_not_kept", "pieces": [], "unkept_renders": [{"name": "b_unkept", "window": [220, 239]}], "cause": "rendered, not kept: b_unkept (220-239)"},
        {"first": 230, "last": 233, "shot": [220, 239], "subject": "a", "kind": "emptied_inside", "pieces": ["a_x"], "cause": "floor"},
        {"first": 234, "last": 236, "shot": [220, 239], "subject": "a", "kind": "laid_but_barely_changed", "pieces": ["a_x"], "cause": "0.150 to 0.210"},
        {"first": 237, "last": 238, "shot": [220, 239], "subject": "a", "kind": "region_far_below_its_own_size", "pieces": ["a_x"], "cause": "carried 300 px"},
        {"first": 160, "last": 161, "shot": [150, 199], "subject": "a", "kind": "captured_not_rendered", "pieces": [], "captures": ["a_cap"], "mask_files": ["data/a_cap/subjects/a/masks__sam.npz"], "cause": "captured, not rendered: a_cap"},
        {"first": 170, "last": 170, "shot": [150, 199], "subject": "a", "kind": "emptied_inside", "pieces": ["a_solo"], "cause": "drop="},
        {"first": 172, "last": 172, "shot": [150, 199], "subject": "a", "kind": "emptied_inside", "pieces": ["a_solo"], "cause": "floor"},
        {"first": 240, "last": 249, "shot": [240, 259], "subject": "a", "kind": "no_capture", "pieces": [], "cause": "no capture"},
        {"first": 240, "last": 249, "shot": [240, 259], "subject": "b", "kind": "no_capture", "pieces": [], "cause": "no capture"},
        {"first": 220, "last": 229, "shot": [220, 239], "subject": "b", "kind": "no_capture", "pieces": [], "cause": "no capture"},
        {"first": 184, "last": 185, "shot": [150, 199], "subject": "a", "kind": "left_out_by_a_look", "pieces": [], "cause": "back turned", "who": "reader"},
        {"first": 124, "last": 125, "shot": [100, 149], "subject": "a", "kind": "seam", "pieces": ["a_part1", "a_part2"], "cause": None},
    ]
    cov = {
        "table": str(d / "t.json"),
        "table_sha256": hashlib.sha256((d / "t.json").read_bytes()).hexdigest(),
        "span": span if span is not None else prev.get("span"),
        "failures": failures if failures is not None else prev.get("failures", []),
        "clip_check": {"file": str(cc_file), "sha256": hashlib.sha256(cc_file.read_bytes()).hexdigest()},
        "by_shot": by_shot if by_shot is not None else prev.get("by_shot", default_by_shot),
        "runs": runs if runs is not None else prev.get("runs", default_runs),
    }
    cov_path.write_text(json.dumps(cov))
    return cov_path


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
             "held": [{"first": 220, "last": 239, "subject": "b", "load": "b_held_load", "reason": "THE LEAD'S HOLD, whole"}],
             "assembly": {"command": f"python bench/assemble_delivery.py --out {d / 'joined.mp4'}",
                          "stack": {"coverage_json": str(d / "frame_coverage_t.json")}}}
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
    sync_gap_check(d)
    (d / ledger_name).write_text(json.dumps(ledger))
    (d / "items.json").write_text(json.dumps({"items": items}))
    # each line as the page's server writes it, with the answers file's line as its source, unless a case gives its own
    (d / "verdicts.jsonl").write_text("".join(json.dumps({"source": f"answers.jsonl:{k + 1}", **v}) + "\n" for k, v in enumerate(lines)))
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
                    "--out", str(out), *(more if "--today" in more else [*more, "--today", "2026-10-20"])]) == 0
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
                + re.findall(r'(?:style|fill|stroke|color)="[^"]*(?:#[0-9a-fA-F]{3,8}|rgba?\(|\b(?:' + "|".join(NAMED) + r')\b)[^"]*"', page, re.I)
            # every value of a colour-bearing property, in the CSS and in a style= or an SVG attribute, is tokens and plain keywords only,
            # in any case (a sixth reader, 2026-10-14: `color:navy` and `color:Gray` passed a short list of names)
            values = re.findall(r"(?<![\w-])(?:color|background(?:-color)?|fill|stroke|outline(?:-color)?|border(?:-[\w-]+)?)\s*:\s*([^;}\"]+)", rest + " ".join(
                re.findall(r'style="([^"]*)"', page)), re.I) + re.findall(r'(?:fill|stroke)="([^"]*)"', page, re.I)
            for v in values:
                words = re.findall(r"[a-z]+", re.sub(r"var\(--[\w-]+\)|\d+(?:\.\d+)?(?:px|em|rem|%)?", "", v.lower()))
                stray += [v for w in words if w not in ("none", "transparent", "inherit", "currentcolor", "solid", "dashed", "dotted", "initial", "unset", "collapse", "separate")]
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
        # THE VANISHING PIECE (a reader, 2026-10-14): the joined clip of 100-149 is of another table, laying a_new alone, and the owner kept it.
        # By the lead's ruling of that day the keep counts on a_new, which it lays; b_new, which it does not lay, must still show
        text = text_of(to_review(Path(tmp), joined_table="other",
                                 verdicts=[{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}]))
        assert "b , 100-149: b_new" in text and "a , 100-149: a_new" not in text and "lays a_new, and this table lays a_new, b_new" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(to_review(Path(tmp), verdicts=[{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}]))
        assert "joined_NEW" not in text and "The shot 100-149" not in text, "a joined clip of this table the owner kept is shown"
    return ("only what this table lays and the owner has not settled; not sure stays with the words since; a joined clip only when its own table lays the "
            "same pieces, else each piece on its own with why, so a piece never vanishes; items by name; the look line a session's; what each replaces")


def waiting_on_you_and_done() -> str:
    """The lead's page of 2026-10-14, at the owner's ask of which items were still theirs: an entry with no click of the
    owner's is WAITING ON YOU; one with a click is under DONE, a closed fold, with the click; a piece settled on the
    owner's today is there too, and one settled before today is on neither."""
    lines = [{"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-14T15:00:00Z"},           # 10/14 10:00 Central: today
             {"what": "choice", "row": "r__s__a_solo", "chosen": "wrong, and I say where", "at": "2026-10-13T15:00:00Z"},  # settled before today
             {"what": "choice", "row": "a_x", "chosen": "not sure", "at": "2026-10-13T10:00:00Z"}]                 # clicked, not settled
    with tempfile.TemporaryDirectory() as tmp:
        page = to_review(Path(tmp), ["--today", "2026-10-14"], verdicts=lines)
        wait, done = page.split("<details><summary><b>Done:")
        wt, dt = text_of(wait).split("Waiting on you")[1], text_of(done)
        # THE CASE THAT GOES RED when a clicked entry is said to wait on the owner, or an unclicked one is folded away as done
        assert "The shot 100-149" in wt and "your last click" not in wt.split("The shot 100-149")[1].split("Pieces in a shot")[0], wt
        assert "a_y" in wt and "a_x" not in wt and "a_kept" not in wt, wt
        assert "a_x" in dt and "your last click: not sure" in dt and "a_kept" in dt and "your last click: keep this one (10/14 10:00)" in dt, dt
        assert "a_solo" not in wt and "a_solo" not in dt, "a piece settled before today is on the page"
        assert "Waiting on you: 2 with no click or words of yours yet" in text_of(wait) and "Done: 2 you have answered" in text_of(page), text_of(page)
    with tempfile.TemporaryDirectory() as tmp:
        dt = text_of(to_review(Path(tmp), ["--today", "2026-10-15"], verdicts=lines).split("<details><summary><b>Done:")[1])
        assert "a_kept" not in dt and "a_x" in dt, "a keep of another day is under Done, or a not sure left it"
    return "an entry with no click waits on the owner; a click puts it under Done with the click; a piece settled today is under Done, and one settled before is on neither"


def page_of(d: Path, more=(), extra=None, **kw) -> str:
    """to-review on the made world, with `extra(d)` run on it before the page is made."""
    args = world(d, **kw)
    if extra:
        extra(d)
    out = d / "day" / "TO_REVIEW.html"
    assert rp.main(["to-review", *args, "--page-url", "p#", "--open-list", "o", "--tables", str(d / "tables"), "--out", str(out), *more]) == 0
    return out.read_text()


def the_sixth_readers_findings() -> str:
    """A sixth reader (2026-10-14), and the lead's three rulings on it that day: clicks and words on an older joined clip of the
    same pieces count on them; a keep with words after it and words with no click are answers, under Done on the day of the
    last line and gone after it; a keep on the joined clip does not hide a piece with a later line of the owner's."""
    def old_joined(d: Path) -> None:          # the older joined clip of 100-149, of a table that lays the same pieces
        (d / "tables" / "t_v_0100_0149_joined_OLD.json").write_text((d / "tables" / "t_v_0100_0149_joined_NEW.json").read_text())
    lines = [{"what": "choice", "row": "v_0100_0149_joined_OLD_stacked", "chosen": "wrong, and I say where", "at": "2026-10-14T17:03:00Z"},
             {"what": "words", "row": "v_0100_0149_joined_OLD_stacked", "words": "HAIR CLIPS AT THE FOREHEAD", "at": "2026-10-14T17:04:00Z"},
             {"what": "words", "row": "r__s__a_solo", "words": "I DONT LIKE BEARD", "at": "2026-10-15T02:00:00Z"},           # 10/14 21:00 Central
             {"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-14T15:00:00Z"},
             {"what": "words", "row": "a_kept", "words": "LOOKS GREAT", "at": "2026-10-14T15:00:05Z"}]
    with tempfile.TemporaryDirectory() as tmp:
        page = page_of(Path(tmp), ["--today", "2026-10-14"], old_joined, verdicts=lines)
        wait, done = (text_of(x) for x in page.split("<details><summary><b>Done:"))
        # THE CASES THAT GO RED: a wrong and words on an older joined clip lost; words with no click left waiting; a praised keep lost on its day
        # the newest joined clip lays the same pieces on the same frames as the older one: it inherits the older clip's lines and the
        # shot is under Done, the older build named (the lead's ruling, on a seventh reader's 1222-1305); each piece's line says them too
        assert "The shot 100-149" not in wait and "The shot 100-149" in done, "a newer build of the same pieces did not inherit the older one's lines"
        # each line says the build it was said on (an eighth reader: an older build's words read as said of the newest)
        assert "the joined clip: your last click: wrong, and I say where (10/14 12:03, on the earlier build v_0100_0149_joined_OLD_stacked)" in done, done
        assert "earlier build(s) of the same pieces on the same frames: v_0100_0149_joined_OLD_stacked" in done, done
        assert "your last click: wrong, and I say where (10/14 12:03), on the whole joined clip v_0100_0149_joined_OLD_stacked" in done, done
        assert "on the whole joined clip v_0100_0149_joined_OLD_stacked: \"HAIR CLIPS AT THE FOREHEAD\"" in done, done
        assert "a_solo" not in wait and "I DONT LIKE BEARD" in done and "a_kept" in done and "LOOKS GREAT" in done, (wait, done)
        # each piece under its own section heading (M16)
        assert "a , 150-199: a_solo" in done.split("Pieces in a shot of one of them")[1] and "a_solo" not in done.split("Pieces in a shot of one of them")[0], done
        # a shot's times run to the frame after its last (M12)
        assert "0:04.2 to 0:06.3 of the video" in done, done
    with tempfile.TemporaryDirectory() as tmp:
        # the next day: the praised keep and the words alone are gone; the wrong on a joined clip, which names no piece, stays
        page = page_of(Path(tmp), ["--today", "2026-10-15"], old_joined, verdicts=lines)
        text = text_of(page)
        assert "a_kept" not in text.split("Held back")[0] and "LOOKS GREAT" not in text and "I DONT LIKE BEARD" not in text, text
        assert "HAIR CLIPS AT THE FOREHEAD" in text, "a wrong on a joined clip left the page"
    with tempfile.TemporaryDirectory() as tmp:
        # a keep on the joined clip, then a not sure on one of its pieces: the shot stays, under Done, with the piece's line (finding 2)
        keep_then = [{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-13T16:00:00Z"},
                     {"what": "choice", "row": "b_new", "chosen": "not sure", "at": "2026-10-13T17:00:00Z"}]
        page = page_of(Path(tmp), ["--today", "2026-10-15"], verdicts=keep_then)
        done = text_of(page.split("<details><summary><b>Done:")[1])
        assert "The shot 100-149" in done and "your last click: not sure (10/13 12:00)" in done, done
        # kept on the joined clip with nothing after, on its day it is under Done; the next day it is gone (M4, M6)
        page = page_of(Path(tmp) / "x", ["--today", "2026-10-13"], verdicts=keep_then[:1]) if (Path(tmp) / "x").mkdir() is None else ""
        assert "The shot 100-149" in text_of(page.split("<details><summary><b>Done:")[1]), "a joined clip kept today is not under Done"
        assert "which counts on each piece it lays" in text_of(page), "a keep on the joined clip is said as a click on the piece itself"
    with tempfile.TemporaryDirectory() as tmp:
        # the day is the owner's: a keep at 02:00Z on the 15th is the 14th in Central (M1, M17)
        page = page_of(Path(tmp), ["--today", "2026-10-14"], verdicts=[{"what": "choice", "row": "a_x", "chosen": "keep this one", "at": "2026-10-15T02:00:00Z"}])
        assert "a_x" in text_of(page.split("<details><summary><b>Done:")[1]), "an evening keep put on the next day"
    with tempfile.TemporaryDirectory() as tmp:
        # a joined clip of the same pieces on other frames is not this table's (M13)
        def shifted(d: Path) -> None:
            t = json.loads((d / "tables" / "t_v_0100_0149_joined_NEW.json").read_text())
            for r in t["rows"]:
                r["first"] += 1
            (d / "tables" / "t_v_0100_0149_joined_NEW.json").write_text(json.dumps(t))
        text = text_of(page_of(Path(tmp), extra=shifted))
        assert "it is of an earlier table" in text and "The shot 100-149 a, b" not in text.split("no joined clip of this table")[0].split("each as its joined clip")[1], text
    # the owner's clock is Central with its daylight time, not a fixed offset (M3): six hours behind UTC in November
    assert rp.owners("2026-11-05T15:00:00Z", "%H:%M") == "09:00" and rp.owners("2026-10-15T02:00:00Z", "%m/%d") == "10/14"
    # where the owner's words came from: a line with a source and no note is not said to be a session's, nor typed in the box (M40)
    assert rp.how_said({"source": "somewhere.jsonl:3"}) == "words from somewhere.jsonl:3, where who wrote them is not said"
    assert rp.how_said({"source": "comments.jsonl:6"}).startswith("the words typed in an earlier page's comment box")
    with tempfile.TemporaryDirectory() as tmp:
        # the stacked clip's path is the repo's, not the working folder's (finding 6); none is said
        d = Path(tmp)
        (d / "media").mkdir()
        (d / "media" / "a_solo_stacked.mp4").write_bytes(b"x")
        here = os.getcwd()
        os.chdir("/")
        try:
            def with_path(dd: Path) -> None:
                items = json.loads((dd / "items.json").read_text())
                next(i for i in items["items"] if i["id"] == "r__s__a_solo")["path"] = "media/a_solo_stacked.mp4"
                (dd / "items.json").write_text(json.dumps(items))
            args = world(d)
            with_path(d)
            out = d / "day" / "PROVENANCE.html"
            assert rp.main(["provenance", *args, "--day-folder", str(d / "day"), "--up-to", "259", "--out", str(out)]) == 0
        finally:
            os.chdir(here)
        text = text_of(out.read_text())
        assert "its stacked clip" in text and "no stacked clip of it: no item on the list names a file of it" in text, text
    return ("a wrong and words on an older joined clip of the same pieces said under Done; words with no click and a praised keep under Done on their day and "
            "gone after; a joined keep does not hide a later line on a piece; the owner's day in Central; a joined clip on other frames not this table's; "
            "each piece under its heading; times to the next frame; where words came from; the stacked clip found from any working folder")


def the_seventh_readers_findings() -> str:
    """A seventh reader (2026-10-14): what the rulings say, held where the page filed by other lines than it showed."""
    def over_a_x(d: Path) -> None:            # a joined clip of 200-239 (not the shot's own frames) whose table lays a_x
        items = json.loads((d / "items.json").read_text())
        items["items"].insert(0, {"id": "v_0200_0239_joined_X_stacked", "first": "200", "frames": "40", "state": "open"})
        items["items"].insert(0, {"id": "r__s__a_solo_newer", "name": "a_solo", "first": "150", "frames": "50", "state": "open", "look": "THE NEWER LOOK"})
        (d / "items.json").write_text(json.dumps(items))
        (d / "tables" / "t_v_0200_0239_joined_X.json").write_text(json.dumps({"rows": [{"first": 220, "last": 239, "name": "a_x", "subject": "a"}]}))
    lines = [{"what": "choice", "row": "a_x", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
             {"what": "choice", "row": "v_0200_0239_joined_X_stacked", "chosen": "not sure", "at": "2026-10-13T11:00:00Z"}]
    with tempfile.TemporaryDirectory() as tmp:
        page = page_of(Path(tmp), ["--today", "2026-10-15"], over_a_x, verdicts=lines)
        done = text_of(page.split("<details><summary><b>Done:")[1])
        # THE CASE THAT GOES RED (F1): a not sure on a joined clip after the piece's own keep is filed by the keep and leaves the page
        assert "a_x" in done.split("Held back")[0] and "your last click: not sure (10/13 06:00), on the whole joined clip v_0200_0239_joined_X_stacked" in done, done
        # the newest item of a render gives its look line (M25)
        assert "THE NEWER LOOK" in text_of(page) and "her eyes on 160" not in text_of(page), "an older item's look line"
    joined = "v_0100_0149_joined_NEW_stacked"
    with tempfile.TemporaryDirectory() as tmp:
        # F2: words with no click on the joined clip leave the page after their day, and are under Done on it (M05)
        words = [{"what": "words", "row": joined, "words": "WORDS ON THE CLIP", "at": "2026-10-13T10:00:00Z"}]
        assert "WORDS ON THE CLIP" not in text_of(page_of(Path(tmp) / "a", ["--today", "2026-10-20"], verdicts=words)) if (Path(tmp) / "a").mkdir() is None else 0
        on_day = page_of(Path(tmp) / "b", ["--today", "2026-10-13"], verdicts=words) if (Path(tmp) / "b").mkdir() is None else ""
        assert "The shot 100-149" in text_of(on_day.split("<details><summary><b>Done:")[1]), "words on a joined clip, on their day, not under Done"
        # the joined clip leaves on the day of its LAST line (M04), on the owner's clock (M09): a keep, then words at 02:00Z on the 15th
        late = [{"what": "choice", "row": joined, "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
                {"what": "words", "row": joined, "words": "AN EVENING LINE", "at": "2026-10-15T02:00:00Z"}]
        page = page_of(Path(tmp) / "c", ["--today", "2026-10-14"], verdicts=late) if (Path(tmp) / "c").mkdir() is None else ""
        wait, done = (text_of(x) for x in page.split("<details><summary><b>Done:"))
        assert "The shot 100-149" in done, "a joined clip with a line of the owner's evening is not under Done that day"
        # the two counts are each their own (M37)
        n_wait, n_done = len(re.findall(r'<div class="e">', page.split("<details>")[0])), len(re.findall(r'<div class="e">', page.split("<details>")[1].split("</details>")[0]))
        assert f"Waiting on you: {n_wait} with no click" in wait and f"Done: {n_done} you have answered" in text_of(page) and n_wait != n_done, (n_wait, n_done)
        # a not sure on the whole joined clip asks which piece (M32)
        ns = page_of(Path(tmp) / "d", verdicts=[{"what": "choice", "row": joined, "chosen": "not sure", "at": "2026-10-13T10:00:00Z"}]) if (Path(tmp) / "d").mkdir() is None else ""
        assert "Your click on the whole clip names no piece: which piece is it?" in text_of(ns)
    # F5: a line with no source is nobody's typing; F4's passed-over words say where they came from
    assert rp.how_said({}) == "your words, where they came from is not said"
    # a piece that crosses a shot's edge is a row of both shots (M01)
    sh = rp.shots_of_table({"reading": [{"first": 0, "last": 9, "subject": "a", "present": "yes"}, {"first": 10, "last": 19, "subject": "a", "present": "yes"}],
                            "rows": [{"first": 5, "last": 14, "name": "across", "subject": "a"}]})
    assert [[r["name"] for r in x["rows"]] for x in sh] == [["across"], ["across"]], sh
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp)))
        page = (Path(tmp) / "day" / "PROVENANCE.html").read_text()
        # an unsure subject's pieces are on the timeline, marked (M10); every link is relative to the page (M22)
        assert "(the record is unsure they are in the shot)" in text and not re.findall(r'href="/', page), "an unsure subject's stretch unmarked, or an absolute link"
    return ("a not sure on a joined clip after the piece's own keep stays under Done; words on a joined clip on their day and gone after; a joined clip on "
            "the day of its last line, in Central; the counts each their own; which piece on a not sure; a line with no source; a piece across a shot's edge; "
            "the newest item's look line; an unsure subject's stretch marked; links relative")


def the_eighth_readers_findings() -> str:
    """An eighth reader (2026-10-14), and the lead's ruling that a joined clip nobody watched whose every piece the owner kept is
    not waiting on them."""
    joined = "v_0100_0149_joined_NEW_stacked"
    with tempfile.TemporaryDirectory() as tmp:
        lines = [{"what": "words", "row": "a_x", "words": "THERE IS HAIR HERE", "at": "2026-10-14T15:00:00Z", "frame": 223},
                 {"what": "choice", "row": "a_x", "chosen": "not sure", "at": "2026-10-14T15:00:05Z"}]
        page = page_of(Path(tmp), ["--today", "2026-10-14"], verdicts=lines)
        text = text_of(page)
        # THE CASES THAT GO RED: a line's frame dropped; the folders the joined tables came from unsaid
        assert "(10/14 10:00, at frame 223): \"THERE IS HAIR HERE\"" in text, text
        assert "The joined clips' tables were read from: " + str(Path(tmp) / "tables") in text, text
    with tempfile.TemporaryDirectory() as tmp:
        # every piece of the shot kept on a day before, its joined clip never watched: not waiting on the owner, and off the page after the day
        kept = [{"what": "choice", "row": "a_new", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
                {"what": "choice", "row": "b_new", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"}]
        wait = text_of(page_of(Path(tmp) / "x", ["--today", "2026-10-13"], verdicts=kept).split("<details><summary><b>Done:")[0]) if (Path(tmp) / "x").mkdir() is None else ""
        assert "The shot 100-149" not in wait.split("Waiting on you")[1], "a shot whose every piece the owner kept waits on them"
        page = page_of(Path(tmp) / "y", ["--today", "2026-10-13"], verdicts=kept) if (Path(tmp) / "y").mkdir() is None else ""
        assert "every piece kept; the joined clip not watched" in text_of(page.split("<details><summary><b>Done:")[1]), "not under Done on the day"
        later = page_of(Path(tmp) / "z", ["--today", "2026-10-20"], verdicts=kept) if (Path(tmp) / "z").mkdir() is None else ""
        assert "The shot 100-149" not in text_of(later).split("Held back")[0], "kept pieces stayed past their day"
        # a keep on the piece the day before, words the next day: under Done on the day of the LAST line (M64)
        late = page_of(Path(tmp) / "w", ["--today", "2026-10-14"], verdicts=[{"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
                                                                       {"what": "words", "row": "a_kept", "words": "LATER WORDS", "at": "2026-10-14T15:00:00Z"}]) \
            if (Path(tmp) / "w").mkdir() is None else ""
        assert "LATER WORDS" in text_of(late.split("<details><summary><b>Done:")[1]), "filed by the first line's day"
        # a wrong on a joined clip that lays a piece, the next day: still under Done, since it names no piece (M12)
        wrong = page_of(Path(tmp) / "v", ["--today", "2026-10-15"], verdicts=[{"what": "choice", "row": joined, "chosen": "wrong, and I say where", "at": "2026-10-13T10:00:00Z"}]) \
            if (Path(tmp) / "v").mkdir() is None else ""
        assert "which names no piece" in text_of(wrong) and "a , 100-149: a_new" in text_of(wrong.split("<details><summary><b>Done:")[1]), "a wrong on the clip closed the piece"
    with tempfile.TemporaryDirectory() as tmp:
        # a wrong on a joined clip laying a piece shown on its own (the shot 220-239 has no joined clip): the next day still under Done (M12)
        def over_a_x(d: Path) -> None:
            items = json.loads((d / "items.json").read_text())
            items["items"].insert(0, {"id": "v_0200_0239_joined_X_stacked", "first": "200", "frames": "40", "state": "open"})
            (d / "items.json").write_text(json.dumps(items))
            (d / "tables" / "t_v_0200_0239_joined_X.json").write_text(json.dumps({"rows": [{"first": 220, "last": 239, "name": "a_x", "subject": "a"}]}))
        page = page_of(Path(tmp), ["--today", "2026-10-15"], over_a_x,
                       verdicts=[{"what": "choice", "row": "v_0200_0239_joined_X_stacked", "chosen": "wrong, and I say where", "at": "2026-10-13T10:00:00Z"}])
        assert "a_x" in text_of(page.split("<details><summary><b>Done:")[1]).split("Held back")[0], "a wrong on a joined clip closed the piece it lays"
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp)))
        # times on the owner's clock (finding 4)
        assert "written 2026-10-14 09:00 US Central (2026-10-14T14:00:00Z)" in text, "a stopped note's time in UTC"
    with tempfile.TemporaryDirectory() as tmp:
        # an assembler's record speaks only for the pieces it assembled: the rest of the table takes the ledger's word (finding 3)
        d = Path(tmp)
        args = world(d)
        record = d / "clip.mp4.check.json"
        record.write_text(json.dumps({"table": [{"piece": "/share/Video/v/s/a_new/a_new_00001.mp4", "frames": [100, 149]}],
                                      "flags": [{"rule": "piece_changes_nothing", "run": "a_new", "source_frames": [[140, 141]]},
                                                {"rule": "piece_changes_nothing", "run": "a_solo", "source_frames": [[190, 191]]}]}))
        out = d / "day" / "P.html"
        assert rp.main(["provenance", *args, "--day-folder", str(d / "day"), "--up-to", "259", "--record", str(record), "--out", str(out)]) == 0
        text = text_of(out.read_text())
        # THE CASE THAT GOES RED when a record of another table speaks for pieces it never assembled
        assert "the assembler's record clip.mp4.check.json for the 1 piece(s) of this table it assembled, and the ledger for the 5 it did not" in text, text
        assert "160-161" in text and "190-191" not in text, "a_solo's frames came from a record that never assembled it"
    # where a line came from: a session's source that names the answers file mid-way is not typed in the box (M04)
    assert rp.how_said({"source": "the lead's terminal, quoting answers.jsonl:4", "note": "taken down"}).startswith("your words as a session took them down")
    return ("a line's frame; the folders read; every piece kept with the joined clip unwatched under Done on the day and gone after; the day of the last "
            "line; a wrong on a clip names no piece and stays; times in Central; a session's source quoting the answers file")


def every_heading_is_its_way_in() -> str:
    """The owner, 2026-10-14: the links were not at the top. Every entry's heading is its link: the joined clip's item, or else
    each standing piece's item, named; a piece with no item says so."""
    with tempfile.TemporaryDirectory() as tmp:
        page = page_of(Path(tmp), ["--today", "2026-10-20"])
        # THE CASE THAT GOES RED when a two-person shot with no joined clip, or a one-person shot, has a heading with no link
        assert '<h3>The shot 220-239: <a href="p#a_x">a_x</a> <span>' in page, re.findall(r"<h3>The shot 220-239.*?</h3>", page)
        assert '<h3>a, 150-199: <a href="p#r__s__a_solo">a_solo</a> <span>' in page, re.findall(r"<h3>a, 150-199.*?</h3>", page)
        assert '<h3><a href="p#v_0100_0149_joined_NEW_stacked">The shot 100-149</a>' in page
        def no_item(d: Path) -> None:
            items = json.loads((d / "items.json").read_text())
            items["items"] = [i for i in items["items"] if i["id"] != "a_x"]
            (d / "items.json").write_text(json.dumps(items))
        page = page_of(Path(tmp) / "x", ["--today", "2026-10-20"], no_item) if (Path(tmp) / "x").mkdir() is None else ""
        assert "<h3>The shot 220-239: a_x (no item of it on the review list yet) <span>" in page, re.findall(r"<h3>The shot 220-239.*?</h3>", page)
    return "a two-person shot with no joined clip and a one-person shot each name their pieces' links in the heading; a piece with no item says so"


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
        text = text_of(to_review(Path(tmp), ["--today", "2026-10-13"], verdicts=words_after_keep))
        # THE CASES THAT GO RED: a keep followed by words is to review again; a session's taking-down is said as such
        assert "a_kept" in text and "ACTUALLY HER CHIN IS OFF" in text, "words after a keep were ignored"
        assert ("your words as a session took them down (the lead's terminal (mrx), 2026-10-13; the session's note on the line: taken down whole by mrx) since (10/13 07:00): "
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


def the_fifth_readers_findings() -> str:
    """A fifth reader (2026-10-14), each a change that left every case green, and the lead's ruling on a joined click."""
    lines = [{"what": "words", "row": "a_x", "words": "WORDS JUST BEFORE THE CLICK", "at": "2026-10-13T09:59:58Z"},
             {"what": "choice", "row": "a_x", "chosen": "not sure", "at": "2026-10-13T10:00:00Z"},
             {"what": "choice", "row": "a_kept", "chosen": "keep this one", "at": "2026-10-13T10:00:00Z"},
             {"what": "choice", "row": "a_kept", "chosen": "wrong", "at": "2026-10-13T09:00:00Z"},          # an older line written later in the file
             {"what": "words", "row": "r__s__a_solo", "words": "ON THE ITEM'S OWN ID", "at": "2026-10-13T11:00:00Z"},
             {"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "wrong", "at": "2026-10-14T16:00:00Z"}]
    with tempfile.TemporaryDirectory() as tmp:
        page = to_review(Path(tmp), ["--today", "2026-10-13"], verdicts=lines)      # the day of the owner's lines: under Done, not gone
        text = text_of(page)
        # words typed just before the last click go with it (B); an older line written later in the file does not undo a keep (M3)
        assert "since the click before it" not in text and "before it (10/13 04:59): \"WORDS JUST BEFORE THE CLICK\"" in text, text
        assert "a_kept" not in text.split("Pieces in a shot of one of them")[1].split("Done:")[0], "a keep was undone by an older line"
        # a line under the item's own id is read (M8); a wrong on the joined clip asks which piece and hides none (the lead's ruling)
        assert "ON THE ITEM'S OWN ID" in text, text
        assert "Your click on the whole clip names no piece: which piece is it?" in text and "a , 100-149: a_new" in text, text
        # the shot's header says which subject the record is unsure of
        assert "a, b (the record is unsure)" in text, text
        # a passed-over render whose last choice is not a keep is not called a keep (M1)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        to_review(d)
        t = json.loads((d / "t.json").read_text())
        t["reading"][6]["passed_over"].append({"render": "b_wrong", "why": "WHY", "the_owners_last_choice": {"said": "wrong", "at": "2026-10-12T20:00:00Z"}})
        t["reading"][6]["passed_over"][0]["the_owners_words"] = [{"at": "2026-10-12T21:00:00Z", "said": "WORDS AFTER THE KEEP"}]
        (d / "t.json").write_text(json.dumps(t))
        sync_gap_check(d)
        out = d / "day" / "again.html"
        args = ["--table", str(d / "t.json"), "--ledger", str(d / "l.json"), "--verdicts", str(d / "verdicts.jsonl"), "--items", str(d / "items.json"),
                "--source", str(d / "v.mp4"), "--share", str(d / "share"), "--page-url", "p#", "--open-list", "o", "--tables", str(d / "tables"), "--out", str(out)]
        assert rp.main(["to-review", *args]) == 0
        text = text_of(out.read_text())
        assert "your keep of b_wrong" not in text and "Your words after the keep (10/12 16:00, where they came from is not said): \"WORDS AFTER THE KEEP\"" in text, text
    with tempfile.TemporaryDirectory() as tmp:
        # a keep on the joined clip counts on each piece it lays: said so under the piece (the lead's ruling)
        page = provenance(Path(tmp)) if False else None
        text = text_of(to_review(Path(tmp), verdicts=[{"what": "choice", "row": "v_0100_0149_joined_NEW_stacked", "chosen": "keep this one", "at": "2026-10-14T16:00:00Z"}],
                                 joined_table="other"))
        assert "a , 100-149: a_new" not in text, "a keep on the joined clip did not count on the piece it lays"
    return ("words just before the last click go with it; an older line written later in the file does not undo a keep; a line on an item's own id is read; "
            "a wrong on a joined clip asks which piece and hides none, a keep there counts on each piece it lays; unsure subjects are marked in a shot's "
            "header; a passed-over render is called a keep only when it is one, with the owner's words after it")


def a_fifth_readers_mutations() -> str:
    """Three changes a fifth reader made that left every case green (2026-10-14): a joined clip matched on its first frame only,
    the reason dropped from the provenance page's change table, and the day colours run oldest first."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        args = world(d)
        rec = rp.Records(args[args.index("--table") + 1], args[args.index("--ledger") + 1], args[args.index("--verdicts") + 1],
                         args[args.index("--items") + 1], [str(d / "tables")])
        # a newer joined clip that begins on the shot's first frame and is of fewer frames is not this shot's
        rec.items_in_order.insert(0, {"id": "v_0100_0120_joined_SHORT_stacked", "first": "100", "frames": "21", "state": "open"})
        (d / "tables" / "t_v_0100_0120_joined_SHORT.json").write_text((d / "tables" / "t_v_0100_0149_joined_NEW.json").read_text())
        it, why = rp.joined_item(rec, 100, 149, [str(d / "tables")])
        assert it and it["id"] == "v_0100_0149_joined_NEW_stacked", (it, why)
        # the newest day takes the first day colour
        assert rec.day_token[max(rec.day_token)] == rp.DAY_TOKENS[0] and rec.day_token[min(rec.day_token)] != rp.DAY_TOKENS[0], rec.day_token
    with tempfile.TemporaryDirectory() as tmp:
        text = text_of(provenance(Path(tmp)))
        part = text.split("What changes in the next clip")[1].split("Loads stopped")[0]
        assert "a_older: no line in the reading" in part, part[:800]
    return "a joined clip of fewer frames on the same first frame is not taken for the shot; the newest day is the first colour; the change table keeps each reason"

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
        # a piece's folder is linked by the files it holds, never as a folder: the review server serves files (2026-10-15)
        assert 'href="../day/0150_a_solo/text.txt"' in page and 'href="../day/0150_a_solo/"' not in page \
            and "no folder of it in the day's folder" in text, "the folder's files are not linked, or the folder is"
        assert not re.search(r'href="[^"]*/"', page), f"a link to a folder: {re.findall(r'href=\"[^\"]*/\"', page)}"
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


def a_one_subject_shot_and_a_cut_piece() -> str:
    """2026-10-15: a one-subject shot whose joined clip is on the list is shown by it, not by its piece's rows; and one piece cut
    into rows by its restore spans is one entry, named once (the page showed 2823b three times, "no item of it on the review list yet")."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        args = world(d)
        T = json.loads((d / "t.json").read_text())
        cut = [{"first": 300, "last": 319, "name": "c_one", "subject": "a", "file": "x"},
               {"first": 320, "last": 322, "name": "c_one", "subject": "a", "file": "x", "restore": ["a.Left_Hand"]},
               {"first": 323, "last": 339, "name": "c_one", "subject": "a", "file": "x", "restore": ["a.Eyeglass"]},
               {"first": 340, "last": 349, "name": "d_one", "subject": "a", "file": "x"},
               {"first": 350, "last": 359, "name": "d_one", "subject": "a", "file": "x", "restore": ["a.Left_Hand"]}]
        T["rows"] += cut
        stand = lambda n, f, l: {"render": n, "frames": [[f, l]], "written": "2026-10-15T15:00:00Z", "step": 2, "the_leads_prefer": None, "why": "step 2"}  # noqa: E731
        T["reading"] += [{"first": 300, "last": 339, "subject": "a", "present": "yes", "stands": [stand("c_one", 300, 339)]},
                         {"first": 300, "last": 339, "subject": "b", "present": "no", "stands": []},
                         {"first": 340, "last": 359, "subject": "a", "present": "yes", "stands": [stand("d_one", 340, 359)]},
                         {"first": 340, "last": 359, "subject": "b", "present": "no", "stands": []}]
        (d / "t.json").write_text(json.dumps(T))
        sync_gap_check(d)
        ledger = json.loads((d / "l.json").read_text())
        for r in (render(d, "c_one", "a", 300, 339, "2026-10-15T15:00:00Z"), render(d, "d_one", "a", 340, 359, "2026-10-15T15:00:00Z")):
            ledger["renders"][r["file"]] = r
        (d / "l.json").write_text(json.dumps(ledger))
        items = json.loads((d / "items.json").read_text())
        items["items"].insert(0, {"id": "v_0300_0339_joined_ONE_stacked", "first": "300", "frames": "40", "state": "open"})
        (d / "items.json").write_text(json.dumps(items))
        (d / "tables" / "t_v_0300_0339_joined_ONE.json").write_text(json.dumps({"rows": cut[:3]}))
        out = d / "day" / "TO_REVIEW.html"
        assert rp.main(["to-review", *args, "--page-url", "p#", "--open-list", "x", "--tables", str(d / "tables"), "--out", str(out),
                        "--today", "2026-10-20"]) == 0
        text = text_of(out.read_text())
        one = text.split("Shots of one of them, as its joined clip")[1].split("Pieces in a shot of one of them")[0]
        assert "The shot 300-339" in one and "c_one" in one, one[:400]
        singles = text.split("Pieces in a shot of one of them")[1].split("Done:")[0]
        assert "c_one" not in singles, "a one-subject shot with its joined clip on the list is also shown by its piece"
        assert singles.count("d_one (no item of it on the review list yet)") == 1, "a piece cut by its restore spans is named more than once"
        assert "a, 340-359: d_one" in singles, singles[:600]
    return "the one-subject shot with its joined clip shown by it, under its own heading; a piece cut into two rows by a restore span named once, over 340-359"


def start_here() -> str:
    """The owner, 2026-10-15 & 2026-10-18: the most important first. --start-here draws the lead's list before everything, in its order,
    each entry linking what the page shows below (a shot, an item) or a clip that is no item, with its why and optional reader_file /
    reader_files read from playback_reading_<reader>.md (4 fields: Sees, Cannot see, Verdict, Why) in a closed fold whose <summary>
    never shows the verdict, plus the owner's verdict and agree/differ when answered; inline reader_notes or bad cards are refused."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        clip = d / "share" / "pair" / "hair_A.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"x")
        (clip.with_suffix(".json")).write_text(json.dumps({"span": [100, 149], "subjects": ["a"]}))
        other = clip.with_name("hair_B.mp4")
        other.write_bytes(b"x")
        (other.with_suffix(".json")).write_text(json.dumps({"span": [150, 199], "subjects": ["a"]}))

        cmp_clip = clip.with_name("load_b_0100_0149_compare.mp4")
        cmp_clip.write_bytes(b"x")
        (cmp_clip.with_suffix(".json")).write_text(json.dumps({"span": [100, 149], "subjects": ["a"]}))

        load_dir = d / "2026-10-18_load_a"
        load_dir.mkdir(parents=True)
        md_mrgemsol = load_dir / "playback_reading_mrgemsol.md"
        md_mrgemsol.write_text(
            "# Playback Reading\n\n"
            "- **Sees**:\n"
            "  - 100-125: <diff> collar sits cleanly against neck\n"
            "  - 126-149: hair edge holds through turn\n"
            "- **Cannot see**: source audio sync on silent comparison\n"
            "- **Verdict**: keep\n"
            "- **Why**: Clean boundary and stable identity across the full span.\n\n"
            "---\n\n"
            "## Detailed notes\n"
            "Long prose below the card block that is not copied into the compact card.\n"
        )
        load_b_dir = d / "2026-10-18_load_b"
        load_b_dir.mkdir(parents=True)
        md_gemread1 = load_b_dir / "playback_reading_gemread1.md"
        md_gemread1.write_text(
            "# Native Playback Reading: `load_b` (`100-149`)\n\n"
            "- **Reader**: `gemread1` (independent Gemini reader)\n"
            "- **Date**: `2026-10-18`\n"
            "- **Watched and listened natively (`view_file`)**:\n"
            "  - `share/pair/load_b_0100_0149_compare.mp4` (realtime 24 fps with audio)\n\n"
            "---\n\n"
            "## Reader Card — Clip 1: `load_b_0100_0149_compare.mp4` (`100-149`)\n\n"
            "- **Sees**:\n"
            "  - **Gap-line frames (`0` gaps across `100-149`)**: Continuous replacement across 100-149.\n"
            "  - **What the gap line does not count**:\n"
            "    - **`110-120` (rising hand)**: Unbroken blonde bob dome on 110-120.\n"
            "- **Cannot see**:\n"
            "  - Cut transition into 150.\n"
            "- **Verdict**: `keep`\n"
            "- **Why**: Retiring keep-outs removes the lower-left notch across 100-149.\n\n"
            "---\n\n"
            "## Detailed Frame-by-Frame & Audio Reading\n\n"
            "1. Body details ignored by card parser.\n\n"
            "---\n\n"
            "## Notes for Start here (reader_notes)\n\n"
            "### Clip 1 (`load_b_0100_0149_compare.mp4`)\n"
            "- Legacy notes after card.\n"
        )
        md_mrpro = load_dir / "playback_reading_mrpro.md"
        md_mrpro.write_text(
            "### 150-199\n"
            "- **Sees**: 150-175: face holds; 176-199:chin flicker at 182\n"
            "- **Cannot see**: full-cut transition into 200\n"
            "- **Verdict**: `remake`\n"
            "- **Why**: Chin flicker at frame 182 breaks the solo close-up.\n\n"
            "### row-7\n"
            "- **Reader**: fresh-eyes reader\n"
            "- **Sees**: 106: 2 px feather on hair_A; 160: softer edge on hair_B\n"
            "- **Cannot see**: whether the owner prefers the tighter crop\n"
            "- **Verdict**: unsure\n"
            "- **Why**: Both arms hold identity; choice is aesthetic.\n"
        )
        ans_file = d / "owner_answers.jsonl"
        ans_file.write_text(
            "\n".join([
                json.dumps({"who": "owner", "item": "hair_A.mp4", "chosen": "Keep this one", "at": "2026-10-18T12:00:00Z"}),
                json.dumps({"who": "owner", "shot": "100-149", "chosen": "wrong, and I say where", "at": "2026-10-18T12:01:00Z"}),
                json.dumps({"who": "owner", "shot": "150-199", "chosen": "Good to go", "at": "2026-10-18T12:05:00Z"}),
                json.dumps({"who": "owner", "item": "https://the-open-list/#row-7", "chosen": "Keep", "at": "2026-10-18T12:06:00Z"}),
                json.dumps({"who": "owner", "item": "https://the-open-list/#row-7", "chosen": "taken back", "at": "2026-10-18T12:07:00Z"}),
                json.dumps({"who": "owner", "item": "open-list-policy", "chosen": "Change the default everywhere", "at": "2026-10-18T12:08:00Z"}),
            ]) + "\n"
        )
        assert rp.normalize_owner_clip_verdict("not this one") == "remake"
        order = [
            {"item": str(clip), "frames": [100, 149], "why": "WHY THE PAIR", "reader_file": str(md_mrgemsol)},
            {"item": str(cmp_clip), "frames": [100, 149], "why": "WHY LOAD_B COMPARE", "reader_file": str(md_gemread1)},
            {"item": "150-199", "why": "WHY A_SOLO'S SHOT", "reader_file": f"{md_mrpro}#150-199"},
            {"item": "100-149", "why": "WHY THE JOINED SHOT"},
            {"link": "https://the-open-list/#row-7", "label": "THE HAIR QUESTION", "why": "WHY THE LINK", "clips": [str(clip), str(other)],
             "reader_files": [f"{md_mrpro}#row-7"]},
        ]
        f = d / "start.json"
        f.write_text(json.dumps(order))
        # load_b's owner verdict is in verdicts.jsonl with what="words" and chosen="keep this one" (C50 item 2)
        verdicts_for_start = [
            {"what": "words", "row": "load_b", "chosen": "keep this one", "words": "looks good on 100-149", "at": "2026-10-18T12:02:00Z"},
        ]
        page = to_review(d, verdicts=verdicts_for_start, more=("--start-here", str(f), "--answers", str(ans_file)))
        text = text_of(page)
        top = text.split("Start here, in this order")[1].split("Where the original still shows: the gap check")[0]
        assert top.index("WHY THE PAIR") < top.index("WHY LOAD_B COMPARE") < top.index("WHY A_SOLO'S SHOT") < top.index("WHY THE JOINED SHOT"), top
        assert "hair_A.mp4" in top and "r__s__a_solo" in top and "v_0100_0149_joined_NEW_stacked" in top, top
        assert '<a href="../share/pair/hair_A.mp4">' in page, "the clip is not linked relative to the page"
        assert text.index("Start here, in this order") < text.index("Where the original still shows: the gap check") < text.index("Waiting on you"), "Start here is not above the gap check and Waiting on you"
        # a question whose buttons are on the Open List: the entry goes there, its clips beneath it as plain links (the owner, 2026-10-15)
        assert '<a href="https://the-open-list/#row-7">THE HAIR QUESTION</a>: WHY THE LINK' in page, "the link entry"
        link_li = page.split("THE HAIR QUESTION")[1].split("</li></ol>")[0]
        assert '<a href="../share/pair/hair_B.mp4">hair_B.mp4</a>' in link_li, "the clips beneath it"
        # reader card: folded closed (<details class="why reader-card"> with no open attribute), <summary> never shows the verdict
        pair_li = page.split("WHY THE PAIR")[1].split("WHY LOAD_B COMPARE")[0]
        loadb_li = page.split("WHY LOAD_B COMPARE")[1].split("WHY A_SOLO&#x27;S SHOT")[0]
        solo_li = page.split("WHY A_SOLO&#x27;S SHOT")[1].split("WHY THE JOINED SHOT")[0]
        for seg in (pair_li, loadb_li, solo_li, link_li):
            assert '<details class="why reader-card"><summary>' in seg and "<details open" not in seg, seg
            sum_txt = seg.split("<summary>")[1].split("</summary>")[0].lower()
            for v_word in ("keep", "remake", "unsure", "agree", "differ"):
                assert v_word not in sum_txt, f"summary leaked {v_word!r}: {sum_txt}"
        assert "Reader card (mrgemsol) — a session&#x27;s reading, not yours" in pair_li, pair_li
        assert "<dt>Sees</dt><dd>100-125: &lt;diff&gt; collar sits cleanly against neck; 126-149: hair edge holds through turn</dd>" in pair_li, pair_li
        assert "<dt>Cannot see</dt><dd>source audio sync on silent comparison</dd>" in pair_li, pair_li
        # pair_li has frames=[100,149] and a later shot="100-149" remake answer in owner_answers.jsonl, which must NOT override hair_A.mp4's own keep answer
        assert '<dt>Verdict</dt><dd>reader: <b>keep</b> &middot; owner: <b>keep</b> ("Keep this one") &mdash; <b class="d">agree</b></dd>' in pair_li, pair_li
        assert "<dt>Why</dt><dd>Clean boundary and stable identity across the full span.</dd>" in pair_li, pair_li
        assert "Long prose below the card block" not in pair_li, "body prose below --- leaked into card"
        # loadb_li: parsed from ## Reader Card — Clip 1 after preamble --- and metadata bullets, markdown stripped, matched row="load_b" in verdicts.jsonl
        assert "Reader card (gemread1) — a session&#x27;s reading, not yours" in loadb_li, loadb_li
        assert "**" not in loadb_li and "`" not in loadb_li, f"markdown punctuation leaked into reader card: {loadb_li}"
        assert '<dt>Verdict</dt><dd>reader: <b>keep</b> &middot; owner: <b>keep</b> ("keep this one") &mdash; <b class="d">agree</b></dd>' in loadb_li, loadb_li
        # solo_li: reader says remake, owner says keep ("Good to go") -> differ
        assert '<dt>Verdict</dt><dd>reader: <b>remake</b> &middot; owner: <b>keep</b> ("Good to go") &mdash; <b class="d">differ</b></dd>' in solo_li, solo_li
        # link_li: owner's prior click was taken back, so only reader's verdict shows (no owner, no agree/differ)
        assert "Reader card (fresh-eyes reader) — a session&#x27;s reading, not yours" in link_li, link_li
        assert "<dt>Verdict</dt><dd>reader: <b>unsure</b></dd>" in link_li, link_li

        # Bad card files and inline JSON fields are refused
        bad_md_missing = load_dir / "bad_missing.md"
        bad_md_missing.write_text("- **Sees**: 100-110: ok\n- **Cannot see**: audio\n- **Verdict**: keep\n")
        bad_md_no_frames = load_dir / "bad_no_frames.md"
        bad_md_no_frames.write_text("- **Sees**: looks good overall: no numbers\n- **Cannot see**: audio\n- **Verdict**: keep\n- **Why**: one line\n")
        bad_md_verdict = load_dir / "bad_verdict.md"
        bad_md_verdict.write_text("- **Sees**: 100: ok\n- **Cannot see**: audio\n- **Verdict**: pass\n- **Why**: one line\n")
        bad_md_why = load_dir / "bad_why.md"
        bad_md_why.write_text("- **Sees**: 100: ok\n- **Cannot see**: audio\n- **Verdict**: keep\n- **Why**: line one\n  line two\n")
        bad_md_unknown = load_dir / "bad_unknown.md"
        bad_md_unknown.write_text("- **Sees**: 100: ok\n- **Cannot see**: audio\n- **Verdict**: keep\n- **Why**: one line\n- **Extra**: no\n")
        bad_md_long_sees = load_dir / "bad_long_sees.md"
        bad_md_long_sees.write_text(
            "- **Sees**:\n  - 100: line 1\n  - 110: line 2\n  - 120: line 3\n  - 130: line 4\n"
            "- **Cannot see**: audio\n- **Verdict**: keep\n- **Why**: one line\n"
        )
        bad_md_long_why = load_dir / "bad_long_why.md"
        bad_md_long_why.write_text(
            "- **Sees**: 100: ok\n- **Cannot see**: audio\n- **Verdict**: keep\n"
            f"- **Why**: {'x' * 225}\n"
        )

        for bad, said in (
            ([{"item": "a_not_here", "why": "w"}], "no item this page shows"),
            ([{"item": "500-599", "why": "w"}], "is not on this page"),
            ([{"item": str(d / "gone.mp4"), "why": "w"}], "is not there"),
            ([{"item": "150-199"}], "says in one line"),
            ([], "not empty"),
            ([{"link": "file:///x", "label": "l", "why": "w"}], "http(s) URL"),
            ([{"link": "https://x", "why": "w"}], "with a label"),
            ([{"link": "https://x", "label": "l", "why": "w", "item": "150-199"}], "names no item"),
            ([{"link": "https://x", "label": "l", "why": "w", "clips": [str(d / "gone.mp4")]}], "is not there"),
            ([{"item": "150-199", "why": "w", "reader_notes": "inline note"}], "never typed into start_here.json"),
            ([{"item": "150-199", "why": "w", "verdict": "keep"}], "never typed into start_here.json"),
            ([{"item": "150-199", "why": "w", "reader_file": ""}], "reader_file must be a non-empty"),
            ([{"item": "150-199", "why": "w", "reader_file": str(d / "missing_reading.md")}], "is not there"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_missing)}], "missing required card field"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_no_frames)}], "Sees in"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_verdict)}], "Verdict in"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_why)}], "Why in"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_unknown)}], "unknown card field"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_long_sees)}], "Sees in"),
            ([{"item": "150-199", "why": "w", "reader_file": str(bad_md_long_why)}], "Why in"),
        ):
            f.write_text(json.dumps(bad))
            try:
                to_review(d, more=("--start-here", str(f)))
            except SystemExit as e:
                assert said in str(e), (bad, str(e))
            else:
                raise AssertionError(f"a start-here list the page cannot hold was drawn: {bad}")
        f.write_text(json.dumps(order))
        assert "Start here" not in text_of(to_review(d)), "the list drawn with no --start-here"
    return ("the lead's list first and in its order, a shot, an item and a clip each linked with its why, a link to where a question's buttons "
            "are with its clips beneath, and each concise plain-text reader card read from playback_reading_<reader>.md in a closed fold whose summary hides the verdict "
            "and whose Verdict row shows owner agree/differ from both owner_answers.jsonl and verdicts.jsonl; twenty bad lists or cards refused")


def nothing_of_a_video_in_code() -> str:
    code = TOOL.read_text()
    for word in ("/mnt/", "/home/", "http://", "https://", ":8765", "fun_", '"fun"', "day_2026"):
        assert word not in code, f"the tool's source holds {word!r}"
    ip = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", code)
    assert ip is None, f"the tool's source holds a machine's address, {ip.group(0) if ip else ''!r}"
    return "no address (of any machine), share, machine folder, day or video in the tool's source"


def the_page_says_what_is_laid() -> str:
    """A table made with --laid says on both pages which list, by its sha256; one made by the older rule says so."""
    made = "Made x from the ledger `l.json`; and ONLY the list of what is laid, `data/v/laid.jsonl` (sha256 abcdef0123456789, 43 lines, every file's bytes checked)."
    assert "data/v/laid.jsonl (sha256 abcdef0123456789, 43 lines)" in rp.laid_from({"made": made}), rp.laid_from({"made": made})
    assert "older rule" in rp.laid_from({"made": "Made x from the ledger `l.json`; no --prefer"})
    return "the list and its sha256 are named; a table by the older rule says so"


def the_gap_check_is_on_both_pages() -> str:
    """2026-10-16: both pages (TO_REVIEW and PROVENANCE) read the gap check's JSON named by `assembly.stack.coverage_json`
    and show per shot, worst first, the count of frame-subjects where the original shows by kind (`captured_not_rendered`,
    `rendered_not_kept`, `emptied_inside`, `laid_but_barely_changed`, `region_far_below_its_own_size`; and, apart, `no_capture`),
    in two lists (`remake` and `capture`) with each subject's gap spans and clocks (1-frame gaps as one frame, never zero-length,
    and no repeated spans), plus the other run kinds (`left_out_by_a_look`, and `seam` counted as seams) and any string `failures`,
    read and never recomputed."""
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        rev = text_of(to_review(Path(t1)))
        prov = text_of(provenance(Path(t2)))
        for which_page, text, before in (("to-review", rev, "Waiting on you"), ("provenance", prov, "What changes in the next clip")):
            assert "Where the original still shows: the gap check" in text, f"{which_page} lacks the gap check heading"
            block = text.split("Where the original still shows: the gap check")[1].split(before)[0]
            assert "frame(s)" not in block and "frame-subject(s)" in block, (which_page, block)
            cap_heading = "Capture (2 shot(s), 30 frame-subject(s)): no capture"
            assert cap_heading in block, (which_page, block)
            remake_part, capture_part = block.split(cap_heading)
            assert "Remake (2 shot(s), 18 frame-subject(s)):" in remake_part, (which_page, remake_part)
            assert remake_part.index("220-239") < remake_part.index("150-199"), (which_page, remake_part)
            assert ("captured_not_rendered: 3, rendered_not_kept: 2, emptied_inside: 4, laid_but_barely_changed: 3, region_far_below_its_own_size: 2"
                    in remake_part), (which_page, remake_part)
            assert "captured_not_rendered: 2, emptied_inside: 2" in remake_part, (which_page, remake_part)
            # per-subject gap spans, 1-frame gap clock as one frame (170 (0:07.1), not 170-170 (0:07.1–0:07.1)), and not the whole shot's clock
            assert "a: 230-233 (" in remake_part and "b: 225-227 (" in remake_part and "228-229 (" in remake_part, (which_page, remake_part)
            assert "a: 160-161 (" in remake_part and "170 (0:07.1)" in remake_part and "172 (0:07.2)" in remake_part, (which_page, remake_part)
            assert "170-170" not in remake_part and "0:07.1–0:07.1" not in remake_part, (which_page, remake_part)
            assert "0:06.3–0:08.3" not in remake_part, (which_page, remake_part)
            assert capture_part.index("240-259") < capture_part.index("220-239"), (which_page, capture_part)
            assert "no_capture: 20" in capture_part and "no_capture: 10" in capture_part, (which_page, capture_part)
            # capture gap spans: each span says whose it is, once, with no repeated spans, and not the whole shot's span (N9)
            capture_list_only = capture_part.split("Other runs from the gap check")[0]
            assert "a, b: 240-249 (" in capture_list_only and capture_list_only.count("240-249") == 1, (which_page, capture_list_only)
            assert "b: 220-229 (" in capture_list_only and "0:09.2–0:10.0" not in capture_list_only, (which_page, capture_list_only)
            # other run kinds and the words for all ladder kinds; seam counted as seam(s), not frame-subject(s)
            assert "Other runs from the gap check" in capture_part, (which_page, capture_part)
            for kind_word in (
                "captured_not_rendered (captured, not rendered)",
                "rendered_not_kept (rendered, not kept)",
                "emptied_inside (emptied inside a piece)",
                "laid_but_barely_changed (laid piece barely changed its region)",
                "region_far_below_its_own_size (carried region far below its own size)",
                "no_capture (no capture)",
                "left_out_by_a_look (left out by a look): 2 frame-subject(s) on 184-185 (a)",
                "seam (seam where two laid pieces of the same person meet): 1 seam(s) on 124-125 (a)",
            ):
                assert kind_word in block, (which_page, kind_word, block)
        # read, never recomputed: reversing `by_shot["remake"]` in the JSON reverses the page's order
        d = Path(t1)
        cov = json.loads((d / "frame_coverage_t.json").read_text())
        cov["by_shot"]["remake"] = list(reversed(cov["by_shot"]["remake"]))
        cov["runs"] = []
        (d / "frame_coverage_t.json").write_text(json.dumps(cov))
        out = d / "day" / "rev2.html"
        assert rp.main(["to-review", "--table", str(d / "t.json"), "--ledger", str(d / "l.json"), "--verdicts", str(d / "verdicts.jsonl"),
                        "--items", str(d / "items.json"), "--source", str(d / "v.mp4"), "--share", str(d / "share"), "--page-url", "p#",
                        "--open-list", "o", "--tables", str(d / "tables"), "--out", str(out), "--today", "2026-10-20"]) == 0
        rev2_txt = text_of(out.read_text()).split("Where the original still shows: the gap check")[1].split("Waiting on you")[0]
        remake2 = rev2_txt.split("Capture (2 shot(s)")[0]
        assert remake2.index("150-199") < remake2.index("220-239"), "the page re-sorted or recomputed by_shot instead of reading it"
        assert "a, b: 220-239 (0:09.2–0:10.0)" in remake2 and "a, b: 240-259 (0:10.0–0:10.8)" in rev2_txt, rev2_txt
        # a gap check with non-empty string `failures` (as frame_coverage.py writes) never renders as clean "None."
        cov_fail = json.loads((d / "frame_coverage_t.json").read_text())
        cov_fail["by_shot"] = {"remake": [], "capture": []}
        cov_fail["runs"] = []
        cov_fail["failures"] = ["150-199 a (a_solo): run='bad' does not resolve"]
        (d / "frame_coverage_t.json").write_text(json.dumps(cov_fail))
        out_fail_rev = d / "day" / "rev_fail.html"
        out_fail_prov = d / "day" / "prov_fail.html"
        assert rp.main(["to-review", "--table", str(d / "t.json"), "--ledger", str(d / "l.json"), "--verdicts", str(d / "verdicts.jsonl"),
                        "--items", str(d / "items.json"), "--source", str(d / "v.mp4"), "--share", str(d / "share"), "--page-url", "p#",
                        "--open-list", "o", "--tables", str(d / "tables"), "--out", str(out_fail_rev), "--today", "2026-10-20"]) == 0
        assert rp.main(["provenance", "--table", str(d / "t.json"), "--ledger", str(d / "l.json"), "--verdicts", str(d / "verdicts.jsonl"),
                        "--items", str(d / "items.json"), "--source", str(d / "v.mp4"), "--share", str(d / "share"),
                        "--day-folder", str(d / "day"), "--up-to", "259", "--out", str(out_fail_prov)]) == 0
        for wf_page, wf_html, wf_before in (("to-review", out_fail_rev, "Waiting on you"), ("provenance", out_fail_prov, "What changes in the next clip")):
            fblock = text_of(wf_html.read_text()).split("Where the original still shows: the gap check")[1].split(wf_before)[0]
            assert "FAILED GAP CHECK" in fblock and "150-199 a (a_solo): run='bad' does not resolve" in fblock and "None." not in fblock, (wf_page, fblock)
    return "both pages show remake, capture, and the other run kinds with their words, per-subject gap spans/clocks, seam counts, and string failures"


def missing_or_stale_gap_check_is_refused() -> str:
    """2026-10-16: both page builds refuse when `assembly.stack.coverage_json` is missing from the table, when the file it
    names is missing on disk, when its `table_sha256` is not the table's, when `by_shot` is missing, when `span` does not
    cover the table's span, when `assembly.command` lacks `--out`, or when `clip_check` is missing, missing on disk, or
    mismatches `--out` or its file's sha256 on disk."""
    for mode in ("to-review", "provenance"):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            args = world(d)
            tail = (["--page-url", "p#", "--open-list", "o", "--tables", str(d / "tables")] if mode == "to-review"
                    else ["--day-folder", str(d / "day"), "--up-to", "259"])
            good_cov = json.loads((d / "frame_coverage_t.json").read_text())
            # 1. by_shot missing or malformed
            bad_cov = dict(good_cov, by_shot={"remake": "not-a-list"})
            (d / "frame_coverage_t.json").write_text(json.dumps(bad_cov))
            out_shape = d / "day" / f"{mode}_shape.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_shape)])
            except SystemExit as stop:
                assert "by_shot" in str(stop) and not out_shape.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built with malformed by_shot")
            # 2. partial --span gap check that does not match the table's span
            (d / "frame_coverage_t.json").write_text(json.dumps(dict(good_cov, span=[100, 199])))
            out_span = d / "day" / f"{mode}_span.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_span)])
            except SystemExit as stop:
                assert "span" in str(stop) and not out_span.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built with a partial --span gap check")
            # 3a. clip_check missing in JSON
            (d / "frame_coverage_t.json").write_text(json.dumps(dict(good_cov, clip_check=None)))
            out_noclip = d / "day" / f"{mode}_noclip.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_noclip)])
            except SystemExit as stop:
                assert "clip_check" in str(stop) and not out_noclip.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built with no clip_check in the gap check")
            # 3b. clip_check file missing on disk (N3)
            (d / "frame_coverage_t.json").write_text(json.dumps(good_cov))
            (d / "joined.mp4.check.json").unlink()
            out_missing_cc = d / "day" / f"{mode}_missing_cc.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_missing_cc)])
            except SystemExit as stop:
                assert "clip_check" in str(stop) and "missing" in str(stop) and not out_missing_cc.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when clip_check file was missing on disk")
            # 3c. clip_check sha256 stale (clip re-assembled after gap check ran)
            (d / "joined.mp4.check.json").write_text(json.dumps({"ok": True, "reassembled": True}))
            out_stale_clip = d / "day" / f"{mode}_stale_clip.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_stale_clip)])
            except SystemExit as stop:
                assert "clip_check" in str(stop) and not out_stale_clip.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when the clip check on disk changed after the gap check")
            sync_gap_check(d)
            # 3d. clip_check points to another clip's valid check file, not the one assembly.command --out names (N5)
            other_cc = d / "other.mp4.check.json"
            other_cc.write_text(json.dumps({"table": ["other"]}))
            cov_other_cc = json.loads((d / "frame_coverage_t.json").read_text())
            cov_other_cc["clip_check"] = {"file": str(other_cc), "sha256": hashlib.sha256(other_cc.read_bytes()).hexdigest()}
            (d / "frame_coverage_t.json").write_text(json.dumps(cov_other_cc))
            out_wrong_cc = d / "day" / f"{mode}_wrong_cc.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_wrong_cc)])
            except SystemExit as stop:
                assert "clip_check" in str(stop) and not out_wrong_cc.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when clip_check pointed to another clip's check file")
            # 3e. assembly.command lacks --out (Item 6)
            t_no_out = json.loads((d / "t.json").read_text())
            t_no_out["assembly"]["command"] = "python bench/assemble_delivery.py"
            (d / "t.json").write_text(json.dumps(t_no_out))
            sync_gap_check(d)
            out_no_out = d / "day" / f"{mode}_no_out.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out_no_out)])
            except SystemExit as stop:
                assert "--out" in str(stop) and not out_no_out.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when assembly.command lacked --out")
            t_restore = json.loads((d / "t.json").read_text())
            t_restore["assembly"]["command"] = f"python bench/assemble_delivery.py --out {d / 'joined.mp4'}"
            (d / "t.json").write_text(json.dumps(t_restore))
            sync_gap_check(d)
            # 4. table_sha256 mismatch (table edited after gap check was written)
            t = json.loads((d / "t.json").read_text())
            t["held"].append({"first": 100, "last": 149, "subject": "a", "load": "x", "reason": "r"})
            (d / "t.json").write_text(json.dumps(t))
            out = d / "day" / f"{mode}_stale.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out)])
            except SystemExit as stop:
                assert "table_sha256" in str(stop) and not out.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built with a stale gap check (table_sha256 mismatch)")
            # 5. missing gap check file on disk
            (d / "frame_coverage_t.json").unlink()
            out2 = d / "day" / f"{mode}_missing.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out2)])
            except SystemExit as stop:
                assert "missing" in str(stop) and not out2.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when the gap check file was missing")
            # 6. table JSON omits assembly.stack.coverage_json
            t["assembly"] = {}
            (d / "t.json").write_text(json.dumps(t))
            out3 = d / "day" / f"{mode}_no_key.html"
            try:
                rp.main([mode, *args, *tail, "--out", str(out3)])
            except SystemExit as stop:
                assert "coverage_json" in str(stop) and not out3.exists(), (mode, stop)
            else:
                raise AssertionError(f"{mode} built when the table named no coverage_json")
    return "both to-review and provenance refuse missing/malformed coverage_json, by_shot, span, table_sha256, --out, or clip_check, writing nothing"


def per_clip_gap_lines_and_start_here_refusal() -> str:
    """C131: right above each player on to-review (Start here, Waiting on you, Done), review_pages.py prints the gap check's
    counts for that clip's own frames: per person, each kind with frame counts and spans, in the check's own words; a whole
    clip has no line; and a clip on Start here whose gap-check line cannot be computed or is suppressed is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        raw_clip = d / "share" / "pair" / "clip_0150_0199.mp4"
        raw_clip.parent.mkdir(parents=True)
        raw_clip.write_bytes(b"x")
        whole_clip = d / "share" / "pair" / "clip_0100_0149.mp4"
        whole_clip.write_bytes(b"x")
        order = [
            {"item": str(raw_clip), "why": "RAW CLIP WITH GAPS"},
            {"item": str(whole_clip), "why": "WHOLE RAW CLIP"},
            {"item": "220-239", "why": "SHOT WITH GAPS"},
            {"item": "100-149", "why": "WHOLE SHOT"},
        ]
        f = d / "start_gaps.json"
        f.write_text(json.dumps(order))
        page = to_review(d, more=("--start-here", str(f)))
        # 1. Start here entry for 150-199 raw clip has its gap-check line right above the clip link
        expected_150 = ("Gaps on 150-199 — a: captured, not rendered: 2 (160-161); "
                        "emptied inside a piece: 2 (170, 172); left out by a look: 2 (184-185)")
        assert f'<div class="why gap-line">{expected_150}</div><a href="../share/pair/clip_0150_0199.mp4">' in page, page
        # 2. Whole clip on 100-149 has NO gap-line banner above it
        assert '<li><a href="../share/pair/clip_0100_0149.mp4">clip_0100_0149.mp4</a> (a clip, not on the review list): WHOLE RAW CLIP</li>' in page, page
        # 3. Shot 220-239 on Start here has its per-person gap-check line right above the shot link
        expected_220 = ("Gaps on 220-239 — a: emptied inside a piece: 4 (230-233); "
                        "laid piece barely changed its region: 3 (234-236); "
                        "carried region far below its own size: 2 (237-238) · "
                        "b: no capture: 10 (220-229); captured, not rendered: 3 (225-227); rendered, not kept: 2 (228-229)")
        assert f'<li><div class="why gap-line">{expected_220}</div>the shot 220-239:' in page, page
        # 4. Waiting on you / Done entries have `<p class="why gap-line">` right above `<h3>` inside `<div class="e">`
        assert f'<div class="e"><p class="why gap-line">{expected_150}</p><h3>a, 150-199:' in page, page
        assert f'<div class="e"><p class="why gap-line">{expected_220}</p><h3>The shot 220-239:' in page, page
        # Whole shot 100-149 has no gap-line inside its `<div class="e">`
        assert '<div class="e"><h3><a href="p#v_0100_0149_joined_NEW_stacked">The shot 100-149</a>' in page, page
        # 5. Refusal when a clip on Start here has no resolvable frame span (so its gap-check line cannot be computed)
        mystery_clip = d / "share" / "pair" / "mystery_clip.mp4"
        mystery_clip.write_bytes(b"x")
        for bad_start, bad_msg in (
            ([{"item": str(mystery_clip), "why": "no span"}], "no frame span for its gap-check line"),
            ([{"item": "150-199", "why": "suppressed", "no_gap_line": True}], "no gap-check line does not go on Start here"),
            ([{"item": "150-199", "why": "overridden", "gap_line": ""}], "no gap-check line does not go on Start here"),
        ):
            f.write_text(json.dumps(bad_start))
            try:
                to_review(d, more=("--start-here", str(f)))
            except SystemExit as stop:
                assert bad_msg in str(stop), (bad_start, str(stop))
            else:
                raise AssertionError(f"Start here accepted an entry without a valid gap-check line: {bad_start}")
    return "per-clip gap lines printed right above each player in Start here, Waiting on you, and Done; clips without a gap-check line refused on Start here"


def compare_clips_say_who_is_new_and_who_is_the_old_cut() -> str:
    """C29: above each compare clip's player, review_pages.py generates one line per person in the shot:
    'new here: <load>', 'the current cut: <load>, laid <day>', or 'the original: no piece', read from
    .stack.json/.stack.frames.jsonl or sidecars and the table/ledger, and refuses inline compare lines or
    compare clips whose candidate load/subject cannot be resolved."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pair_dir = d / "share" / "pair"
        pair_dir.mkdir(parents=True)

        # 1. Stack-delivery compare clip on 100-149 with .stack.json + .stack.frames.jsonl:
        #    b is new (b_cand_0100), while a has two pieces in the current cut across the shot (a_p1 laid 10/15, a_p2 laid 10/13).
        cmp_stack = pair_dir / "b_cand_0100_0100_0149_compare.mp4"
        cmp_stack.write_bytes(b"x")
        frames_jsonl = cmp_stack.with_suffix(".stack.frames.jsonl")
        frames_jsonl.write_text(
            "\n".join([
                json.dumps({
                    "source_frame": 100,
                    "columns": {
                        "current:b": {"side": "current", "subject": "b", "piece": "b_new"},
                        "candidate:b": {"side": "candidate", "subject": "b", "piece": "b_cand_0100"},
                    },
                    "legend": {
                        "b": "b: current b_new (10/13) -> candidate b_cand_0100 (10/16)",
                        "a": "a: a_p1 (10/15) (both)",
                    },
                }),
                json.dumps({
                    "source_frame": 130,
                    "columns": {
                        "current:b": {"side": "current", "subject": "b", "piece": "b_new"},
                        "candidate:b": {"side": "candidate", "subject": "b", "piece": "b_cand_0100"},
                    },
                    "legend": {
                        "b": "b: current b_new (10/13) -> candidate b_cand_0100 (10/16)",
                        "a": "a: a_p2 (10/13) (both)",
                    },
                }),
            ]) + "\n"
        )
        cmp_stack.with_suffix(".stack.json").write_text(json.dumps({
            "layout": "compare",
            "frames": [100, 149],
            "compared_subjects": ["b"],
            "unchanged_subjects": ["a"],
            "days": {
                "a_p1_00001.mp4": {"day": "10/15"},
                "a_p2_00001.mp4": {"day": "10/13"},
                "b_new_00001.mp4": {"day": "10/13"},
                "b_cand_0100_00001.mp4": {"day": "10/16"},
            },
            "records": frames_jsonl.name,
        }))

        # 2. Shot 220-239 has a (a_x laid 10/14) and b (in the shot, no piece laid -> the original: no piece).
        #    a_cand_0220_0220_0239_compare.mp4 tests 'a: new here: a_cand_0220' + 'b: the original: no piece'.
        cmp_orig = pair_dir / "a_cand_0220_0220_0239_compare.mp4"
        cmp_orig.write_bytes(b"x")
        #    b_cand_0220_0220_0239_compare.mp4 tests 'a: the current cut: a_x, laid 10/14' + 'b: new here: b_cand_0220'.
        cmp_b220 = pair_dir / "b_cand_0220_0220_0239_compare.mp4"
        cmp_b220.write_bytes(b"x")

        # 3. Raw 2x2 panel compare clip (extra.panels, no table/joined clip in inputs) on 100-149:
        #    only the compared subject is laid; the other person in the shot is 'the original: no piece'.
        cmp_panels = pair_dir / "compare_a_old_vs_a_cand_panel_0100_0149.mp4"
        cmp_panels.write_bytes(b"x")
        (Path(str(cmp_panels) + ".json")).write_text(json.dumps({
            "inputs": {"source": "v.mp4"},
            "extra": {"panels": {"top_right": "a_old", "bottom_left": "a_cand_panel"}},
        }))

        order = [
            {"item": str(cmp_stack), "why": "STACK COMPARE B NEW A CURRENT"},
            {"item": str(cmp_orig), "why": "COMPARE A NEW B ORIGINAL"},
            {"item": str(cmp_b220), "why": "COMPARE B NEW A CURRENT"},
            {"link": "https://the-open-list/#cmp-q", "label": "PANEL QUESTION", "why": "WHY PANEL", "clips": [str(cmp_panels)]},
        ]
        f = d / "start_cmp.json"
        f.write_text(json.dumps(order))
        page = to_review(d, more=("--start-here", str(f)))

        # Verify 1: stack compare clip with two current-cut pieces on unchanged subject 'a'
        assert (
            '<li>'
            '<div class="why compare-line">b: new here: b_cand_0100</div>'
            '<div class="why compare-line">a: the current cut: a_p1, laid 10/15; a_p2, laid 10/13</div>'
            '<a href="../share/pair/b_cand_0100_0100_0149_compare.mp4">'
        ) in page, page

        # Verify 2a: a is new here, b is the original: no piece, and gap-line labels each person's cut role (C50 item 3)
        assert (
            '<li>'
            '<div class="why compare-line">a: new here: a_cand_0220</div>'
            '<div class="why compare-line">b: the original: no piece</div>'
            '<div class="why gap-line">Gaps on 220-239 — a (new here): emptied inside a piece: 4 (230-233); '
            'laid piece barely changed its region: 3 (234-236); carried region far below its own size: 2 (237-238) · '
            'b (the original): no capture: 10 (220-229); captured, not rendered: 3 (225-227); rendered, not kept: 2 (228-229)</div>'
        ) in page, page

        # Verify 2b: a is the current cut (a_x, laid 10/14), b is new here (b_cand_0220), and gap-line labels both
        assert (
            '<li>'
            '<div class="why compare-line">a: the current cut: a_x, laid 10/14</div>'
            '<div class="why compare-line">b: new here: b_cand_0220</div>'
            '<div class="why gap-line">Gaps on 220-239 — a (the current cut): emptied inside a piece: 4 (230-233); '
            'laid piece barely changed its region: 3 (234-236); carried region far below its own size: 2 (237-238) · '
            'b (new here): no capture: 10 (220-229); captured, not rendered: 3 (225-227); rendered, not kept: 2 (228-229)</div>'
        ) in page, page

        # Verify 3: panel compare clip inside a link entry's clips list
        assert (
            '<div class="why compare-line">a: new here: a_cand_panel</div>'
            '<div class="why compare-line">b: the original: no piece</div>'
            '<a href="../share/pair/compare_a_old_vs_a_cand_panel_0100_0149.mp4">'
        ) in page, page

        # Verify 4: refusals for inline compare keys and unresolvable compare clips
        ambig = pair_dir / "mystery_0100_0149_compare.mp4"
        ambig.write_bytes(b"x")
        for bad_start, bad_msg in (
            ([{"item": str(cmp_stack), "why": "typed", "compare_line": "b: new here: x"}], "never typed into start_here.json"),
            ([{"item": str(cmp_stack), "why": "typed", "compare_lines": ["b: new here: x"]}], "never typed into start_here.json"),
            ([{"item": str(cmp_stack), "why": "typed", "who_is_new": "b"}], "never typed into start_here.json"),
            ([{"item": str(cmp_stack), "why": "typed", "no_compare_line": True}], "never typed into start_here.json"),
            ([{"item": str(ambig), "why": "ambiguous"}], "names no candidate load or subject"),
        ):
            f.write_text(json.dumps(bad_start))
            try:
                to_review(d, more=("--start-here", str(f)))
            except SystemExit as stop:
                assert bad_msg in str(stop), (bad_start, str(stop))
            else:
                raise AssertionError(f"Start here accepted invalid compare entry: {bad_start}")
    return "every compare clip prints one generated line per person in the shot and labels each person in the gap line with '(new here)' / '(the current cut)' / '(the original)', refusing inline or unresolvable compare clips"


case("both modes read", both_modes_read)
case("the page says what is laid", the_page_says_what_is_laid)
case("to-review is the table's, and hides no live piece", to_review_is_the_tables)
case("since the last page", since_the_last_page)
case("a fifth reader's mutations, held", a_fifth_readers_mutations)
case("a fifth reader's findings, and a joined click", the_fifth_readers_findings)
case("waiting on you, and done", waiting_on_you_and_done)
case("a sixth reader's findings, and the lead's rulings on them", the_sixth_readers_findings)
case("a seventh reader's findings", the_seventh_readers_findings)
case("an eighth reader's findings", the_eighth_readers_findings)
case("every heading is its way in", every_heading_is_its_way_in)
case("the owner's words rule, and whose words are whose", the_owners_words_rule)
case("provenance reads the records", provenance_reads_the_records)
case("another ledger is refused", another_ledger_is_refused)
case("a one-subject shot by its joined clip, and a piece cut by restore spans", a_one_subject_shot_and_a_cut_piece)
case("start here, in the lead's order", start_here)
case("nothing of a video in the code", nothing_of_a_video_in_code)
case("the gap check is on both pages, worst first", the_gap_check_is_on_both_pages)
case("a missing or stale gap check is refused", missing_or_stale_gap_check_is_refused)
case("per-clip gap lines right above each player, and Start here refusal", per_clip_gap_lines_and_start_here_refusal)
case("every compare clip says who is new and who is the old cut", compare_clips_say_who_is_new_and_who_is_the_old_cut)

if __name__ == "__main__":
    sys.exit(finish())
