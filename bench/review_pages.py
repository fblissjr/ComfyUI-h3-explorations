#!/usr/bin/env python3
"""The owner's two local pages in the video's folder on the share (`Video/<video>/`, one copy each), made from the delivery table and the ledger: what is to review now (`to-review`), and what every stretch of the video is made of and from which day (`provenance`).

    <python> bench/review_pages.py to-review --table T.json --ledger LEDGER.json [--previous-table T_BEFORE.json] \\
        --verdicts data/<stem>/verdicts/verdicts.jsonl --items data/<stem>/review/items.json --source SOURCE.mp4 \\
        --share <the server's output folder> --page-url URL --open-list URL [--tables FOLDER] [--provenance PROVENANCE.html] --out TO_REVIEW.html
    <python> bench/review_pages.py provenance --table T.json --ledger LEDGER.json [--record CLIP.mp4.check.json] \\
        --verdicts data/<stem>/verdicts/verdicts.jsonl --items data/<stem>/review/items.json --source SOURCE.mp4 \\
        --share <the server's output folder> --day-folder <share>/Video/<stem>/day_<date> --up-to N \\
        [--watched-table T_WATCHED.json] --out PROVENANCE.html

**NOT MADE** (2026-10-14): four blind readers of the per-shot clip tool and two of the pages each found real faults; the
lead ruled that neither tool takes a new feature until it passes one more read. What the last read found and is not
mended is in docs/checks.md's row for its check ("NOT HELD") and in the lead's notes. Read its output against the
table and the ledger it was made from.

**What it buys.** The owner's rule (2026-10-14, `AGENTS.md`, "One ledger, made new on every landing"): their pages
and the sessions read the same files, and nothing is typed or written twice. On 2026-10-13 the two pages were the
lead's session scripts (`internal/claude/2026-10-13_mrtower_scripts/to_review_page.py`, `provenance_page.py`), with
the video, the server's address and the day's folder in the code, the lead's words and decisions typed into one, and
the stopped loads a typed list in the other. Here every fact is read from the delivery table
(`bench/delivery_table.py --json`), the ledger it was made from (`bench/render_ledger.py --json`, which also holds
the stopped loads, read from the card's notes), the verdicts record, the review list and the source's frame rate;
provenance's cards also read each piece's job record (`job.render.json`, for a look's `left_out`) and its folder in the
day's folder (its text). The address, the folders and the video are arguments. Beside a record's words the pages say
whose they are: the owner's, a session's (the lead's --prefer, a card's look line) or the table's rule.
Decisions are not here: they are on the Open List (`--open-list`), which the page links to.

**Both read in dark mode.** Every colour is a token on `:root`, set again for dark under
`@media (prefers-color-scheme: dark)` and under `:root[data-theme="dark"]`, and no rule names a colour of its own.
The first local page was unreadable in dark mode because its dark rule sat before the rule it meant to override
(the owner's screenshot, 2026-10-13); with tokens there is no such order to get wrong.

**`to-review`: what the CURRENT table lays and the owner has not settled**, never every open item of the review
list (it holds the dead items of earlier days). Settled is a keep or a "not the one" (`SETTLED`, the table's own
words) with no words of the owner's after it: their words rule over the button, so a keep followed by words is shown
again with them; a "not sure" leaves a piece to review. Each line of words says where it came from: typed in the
page's box, or said elsewhere and taken down by a session (the verdicts record's `source` and `note`).
A record shot of the table with two or more of the subjects in it (the table's reading: present or unsure) is shown
as its joined clip, the newest item on the list of exactly that shot's frames, ONLY when the table that clip was
built from (`--tables`) lays the same pieces on the shot as this one, until the owner KEEPS that clip (a "wrong" on
it leaves the shot here with its pieces); otherwise, and where no joined clip is built, each unsettled piece of the
shot is shown on its own, with why. A shot of one subject with a joined clip of this table on the list is shown by
it too, under its own heading (2026-10-15); with none, its piece is shown on its own, once, however many rows its restore
spans cut it into. A
piece's item is the one whose id or `name` is the render's. Each piece carries the day it was rendered on the
owner's clock, its item's look line (said as the card's words), and what it replaces: the rows of its subject that
`--previous-table` laid on its frames. With `--previous-table`, a part "since the last page", on the frames both
tables cover (said, with each one's): the pieces added and the pieces that left, and for each subject the frames
that gained a piece (a gap closed) or lost one (opened). Each entry is put under WAITING ON YOU when it has no click of
the owner's, and under DONE, a closed fold, when it has one, with a piece they settled on `--today` (their clock) shown
there too (the lead's page of 2026-10-14, at the owner's ask of which items were still theirs). The lead's rulings the
same day, on a sixth reader's findings: a keep with words after it, and words with no click, are answers, under Done on the
day of the last line and off the page after it; the clicks and words on any joined clip that lays a piece count as
answers on it (a keep also settles it). Both pages include a short block from the table's per-frame gap check
(`assembly.stack.coverage_json`, from `bench/delivery_table.py`, held by `table_sha256` and never recomputed; a build
refuses when it is missing or not this table's), worst shots first in the `remake` and `capture` lists apart, by kind
(`original_while_tracked`, `emptied_inside`, `laid_but_barely_changed`, `region_far_below_its_own_size`; and, apart,
`original_tracking_not_known`), and end, or begin, with the lead's holds (the table's
`held`, a session's word, its reason whole) and the owner's keeps this table does not lay, with the table's reason.
Links are `--page-url` + an item's id.

**`provenance`**: a timeline a subject (the table's subjects in the order it reads them), each stretch a piece, a
stretch a piece covers and its load left as the original, or a subject in the shot with nothing there; a lane of
frames inside no piece at all; every stretch sorted by frame with the record's reason; the loads stopped before a
render (the ledger's `stopped`: each note whole, its frames and who, and what of that subject stands there now); a
card a piece with where it came from and links by relative path to its render, its stacked clip and its folder of
`bench/day_pieces.py`. With `--watched-table`, what changes from the clip the owner watched and why. The frames a
piece leaves as the original are the assembler's record's (`--record`) where one is given, else the ledger's.

**Colours of the days.** The newest day a piece was rendered on takes the first day colour, the one before it
the second, and so on; the fourth day and every older one share the last colour. The day is always written beside it.
Every link on either page is relative to where the page is written (`--out`).

**What it cannot see.** The picture: whether a face shows on a frame, whether a piece is good. A reason is what a
record says. Whether the review server is running when the owner opens a link. A joined clip built from an earlier
table than this one (its item's frames are the shot's, and its pieces may differ: the item's own words say which).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delivery_table  # noqa: E402
from piece_gaps import OWNERS_CLOCK  # noqa: E402

E = html.escape

# Light and dark values of every colour either page uses. Picked by eye on both pages (2026-10-13, the lead's session
# script); the day colours are the newest day first. The light day0 and day1 were darkened on 2026-10-14 to at least
# 4.5:1 against the page and the card (a reader measured 4.08 and 3.44); `check_review_pages.py` holds every text
# token to that, and every token a timeline shape is drawn in to 3:1, drawn solid: `left` was drawn at .55 opacity and
# came to 1.8:1 in light and 2.3:1 in dark (a reader, 2026-10-14). Nothing below names a colour of its own.
TOKENS = {
    "bg": ("#fafafa", "#161616"), "fg": ("#1a1a1a", "#e8e8e8"), "mut": ("#555555", "#b0b0b0"), "line": ("#d0d0d0", "#3a3a3a"),
    "card": ("#ffffff", "#222222"), "link": ("#0b57d0", "#8ab4f8"), "gap": ("#c0392b", "#e8604c"), "left": ("#808080", "#8c8c8c"),
    "bare": ("#444444", "#bbbbbb"),
    "day0": ("#17705a", "#35b596"), "day1": ("#8a5a00", "#e0a83c"), "day2": ("#6a4fb3", "#a990f0"), "day3": ("#5f6b7a", "#9aa7b8"),
}
DAY_TOKENS = ("day0", "day1", "day2", "day3")
REMAKE_GAP_KINDS = ("original_while_tracked", "emptied_inside", "laid_but_barely_changed", "region_far_below_its_own_size")
CAPTURE_GAP_KINDS = ("original_tracking_not_known",)
OTHER_RUN_KINDS = ("carried_not_known", "original_her_track_empty", "left_out_by_a_look", "seam")
KIND_WORDS = {
    "original_while_tracked": "original while tracked, no piece laid",
    "emptied_inside": "emptied inside a laid piece",
    "laid_but_barely_changed": "laid piece barely changed its region",
    "region_far_below_its_own_size": "carried region far below its own size",
    "original_tracking_not_known": "the original: no piece made for these frames (no capture here, or a capture that holds no track of her)",
    "carried_not_known": "tracked, whether the laid piece carried a mask is not known",
    "original_her_track_empty": "original, her tied track is empty (0 px)",
    "left_out_by_a_look": "original, left out by a person's look",
    "seam": "seam where two laid pieces of the same person meet",
}


def theme_css() -> str:
    light = ";".join(f"--{k}:{v[0]}" for k, v in TOKENS.items())
    dark = ";".join(f"--{k}:{v[1]}" for k, v in TOKENS.items())
    return (f":root{{color-scheme:light dark;{light}}}\n"
            f"@media (prefers-color-scheme: dark){{:root:not([data-theme=\"light\"]){{{dark}}}}}\n"
            f":root[data-theme=\"dark\"]{{{dark}}}\n"
            + "".join(f".{t}{{color:var(--{t})}} .k.{t},rect.{t}{{background:var(--{t});fill:var(--{t})}} .card.{t}{{border-left-color:var(--{t})}}\n" for t in DAY_TOKENS))


def rate_of(source: str) -> Fraction:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", source],
                         capture_output=True, text=True)
    if out.returncode or "/" not in out.stdout:
        raise SystemExit(f"--source {source}: ffprobe gives no frame rate: {out.stderr.strip()[:200]}")
    return Fraction(out.stdout.strip())


def clock(frame: int, rate: Fraction) -> str:
    tenths = int(round(float(frame / rate) * 10))            # whole tenths first, so a time never reads 0:60.0
    return f"{tenths // 600}:{tenths % 600 / 10:04.1f}"


def owners(utc: str, fmt: str = "%m/%d") -> str:
    t = datetime.datetime.fromisoformat(utc.replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)).astimezone(OWNERS_CLOCK).strftime(fmt)


def central(utc) -> str:
    """A record's UTC stamp on the owner's clock, with the stamp itself beside it; the stamp as it is where it is not one."""
    try:
        return f"{owners(str(utc), '%Y-%m-%d %H:%M')} US Central ({utc})"
    except (ValueError, TypeError):
        return str(utc)


def runs(frames) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def read_verdicts(path: str) -> dict[str, list[dict]]:
    """Every line of the verdicts record by its row, in the record's own order: a choice (a button) or words."""
    out: dict[str, list[dict]] = {}
    for line in open(path):
        v = json.loads(line)
        if v.get("row") and (v.get("what") == "choice" and v.get("chosen") or v.get("words")):
            out.setdefault(v["row"], []).append(v)
    return out


def how_said(v: dict) -> str:
    """Where a line of the owner's words came from, by the verdicts record's `source`: typed in the review page's box
    (the page's own answers file), or said elsewhere and taken down by a session (its `source` says where, and its
    `note` who took it down)."""
    src = str(v.get("source") or "")
    if not src:
        return "your words, where they came from is not said"
    if src.startswith("answers.jsonl"):
        return "the words typed in its box (they may quote a session's question)"
    if src.startswith("comments.jsonl"):                   # an earlier page's comment box (a sixth reader, 2026-10-14)
        return f"the words typed in an earlier page's comment box ({src})"
    if not v.get("note"):
        return f"words from {src}, where who wrote them is not said"
    return f"your words as a session took them down ({src}; the session's note on the line: {v['note']})"


def in_repo(path: str, near: str) -> Path:
    """A path a record holds, absolute or relative to the repo its record sits in: the first folder above `near` that holds
    it, else this repo (a sixth reader, 2026-10-14: read against the folder the tool ran from, the pages changed silently)."""
    if Path(path).is_absolute():
        return Path(path)
    for anc in [*Path(near).resolve().parents, HERE.parent]:
        if (anc / path).exists():
            return anc / path
    return HERE.parent / path


def table_span(TT: dict) -> tuple[int, int] | None:
    shots = list(TT.get("reading") or []) + list(TT.get("not_in_the_shot") or [])
    rr = shots or list(TT.get("rows") or [])
    return (min(int(r["first"]) for r in rr), max(int(r["last"]) for r in rr)) if rr else None


def read_gap_check(T: dict, table: str) -> tuple[dict, Path]:
    """The gap check (`bench/frame_coverage.py --json`) named by `assembly.stack.coverage_json`, verified by `table_sha256`,
    `span`, and `clip_check`."""
    import shlex
    cov_rel = ((T.get("assembly") or {}).get("stack") or {}).get("coverage_json")
    if not cov_rel:
        raise SystemExit(f"--table {table}: the table names no gap check (assembly.stack.coverage_json is missing)")
    cov_path = in_repo(str(cov_rel), table)
    if not cov_path.is_file():
        raise SystemExit(f"--table {table}: the gap check {cov_rel} is missing (looked at {cov_path})")
    cov = json.load(open(cov_path))
    want_sha = hashlib.sha256(Path(table).read_bytes()).hexdigest()
    got_sha = cov.get("table_sha256")
    if got_sha != want_sha:
        raise SystemExit(f"--table {table}: the gap check {cov_rel} has table_sha256 {got_sha!r}, not this table's {want_sha}")
    by_shot = cov.get("by_shot")
    if not isinstance(by_shot, dict) or not isinstance(by_shot.get("remake"), list) or not isinstance(by_shot.get("capture"), list):
        raise SystemExit(f"--table {table}: the gap check {cov_rel} has no by_shot remake and capture lists")
    t_span = table_span(T)
    c_span = cov.get("span")
    if c_span is not None and t_span is not None:
        if not isinstance(c_span, list) or len(c_span) != 2 or (int(c_span[0]), int(c_span[1])) != t_span:
            raise SystemExit(f"--table {table}: the gap check {cov_rel} is for span {c_span!r}, not this table's span {list(t_span)}")
    cc = cov.get("clip_check")
    if not isinstance(cc, dict) or not cc.get("file") or not cc.get("sha256"):
        raise SystemExit(f"--table {table}: the gap check {cov_rel} names no clip_check (file and sha256 required)")
    cc_path = in_repo(str(cc["file"]), table)
    if not cc_path.is_file():
        raise SystemExit(f"--table {table}: the gap check {cov_rel}'s clip_check {cc['file']} is missing on disk")
    cc_sha = hashlib.sha256(cc_path.read_bytes()).hexdigest()
    if cc.get("sha256") != cc_sha:
        raise SystemExit(f"--table {table}: the gap check {cov_rel} has clip_check sha256 {cc.get('sha256')!r}, not {cc_path}'s {cc_sha}")
    cmd_parts = shlex.split(str(((T.get("assembly") or {}).get("command")) or ""))
    if "--out" not in cmd_parts[:-1]:
        raise SystemExit(f"--table {table}: assembly.command names no --out clip to tie clip_check to")
    out_arg = cmd_parts[cmd_parts.index("--out") + 1]
    want_cc = in_repo(out_arg if out_arg.endswith(".check.json") else out_arg + ".check.json", table)
    if cc_path.resolve() != want_cc.resolve():
        raise SystemExit(f"--table {table}: the gap check {cov_rel} read clip_check {cc_path}, not the table's {want_cc}")
    return cov, cov_path


def fmt_clock_span(lo: int, hi: int, rate: Fraction) -> str:
    return f"{lo} ({clock(lo, rate)})" if lo == hi else f"{lo}-{hi} ({clock(lo, rate)}&ndash;{clock(hi + 1, rate)})"


def fmt_shot_spans(x: dict, shot_runs: list[dict], s0: int, s1: int, rate: Fraction) -> str:
    subs_order = list(dict.fromkeys(str(s) for s in (x.get("subjects") or [r.get("subject") for r in shot_runs if r.get("subject")])))
    groups: dict[tuple[tuple[int, int], ...], list[str]] = {}
    for sub in subs_order:
        sp = tuple(dict.fromkeys(sorted((int(r["first"]), int(r["last"])) for r in shot_runs if str(r.get("subject") or "") == sub)))
        if sp:
            groups.setdefault(sp, []).append(sub)
    if groups:
        return "; ".join(f"{E(', '.join(g_subs))}: {', '.join(fmt_clock_span(lo, hi, rate) for lo, hi in sp)}" for sp, g_subs in groups.items())
    return f"{E(', '.join(subs_order))}: {fmt_clock_span(s0, s1, rate)}"


def gap_check_block(rec: Records, rate: Fraction) -> list[str]:
    """Per shot, the count of frame-subjects where the original still shows, by kind, worst first, in two lists (`remake` and `capture`),
    plus the four other run kinds (`carried_not_known`, `original_her_track_empty`, `left_out_by_a_look`, `seam`) and any `failures`,
    read from the table's gap check (`assembly.stack.coverage_json`), never recomputed."""
    cov = rec.gap_check or {}
    by_shot = cov.get("by_shot") or {}
    remake, capture = list(by_shot.get("remake") or []), list(by_shot.get("capture") or [])
    all_runs = list(cov.get("runs") or [])
    failures = list(cov.get("failures") or [])
    cov_rel = ((rec.T.get("assembly") or {}).get("stack") or {}).get("coverage_json") or ""
    sha = str(cov.get("table_sha256") or "")[:12]
    out = ["<h2>Where the original still shows: the gap check</h2>",
           f'<p class="mut">Read from <code>{E(os.path.basename(str(cov_rel)))}</code> (<code>table_sha256</code> <code>{E(sha)}</code>), worst shots first, never recomputed.</p>',
           '<div class="card">']
    if failures:
        out.append(f'<p class="d"><b>FAILED GAP CHECK ({len(failures)} failure(s)):</b></p><ul>')
        for fl in failures:
            out.append(f"<li><code>{E(str(fl))}</code></li>")
        out.append("</ul>")
    remake_words = ", ".join(f"<code>{E(k)}</code> ({E(KIND_WORDS[k])})" for k in REMAKE_GAP_KINDS)
    out.append(f"<h3>Remake ({len(remake)} shot(s), {sum(int(x.get('frames', 0)) for x in remake)} frame-subject(s)): a laid piece fails, or no piece where tracked</h3>")
    out.append(f'<p class="mut">Kinds: {remake_words}.</p>')
    if remake:
        out.append("<ul>")
        for x in remake:
            s0, s1 = int(x["shot"][0]), int(x["shot"][1])
            bk = x.get("by_kind") or {}
            kinds = ", ".join(f"{k}: {int(bk.get(k, 0))}" for k in REMAKE_GAP_KINDS if int(bk.get(k, 0)))
            subs = ", ".join(str(s) for s in (x.get("subjects") or []))
            r_runs = [r for r in all_runs if r.get("kind") in REMAKE_GAP_KINDS and r.get("shot") and [int(r["shot"][0]), int(r["shot"][1])] == [s0, s1]]
            spans = fmt_shot_spans(x, r_runs, s0, s1, rate)
            out.append(f"<li><b>{s0}-{s1}</b> ({E(subs)}; gap(s): {spans}): <b>{int(x['frames'])} frame-subject(s)</b> &mdash; {E(kinds)}</li>")
        out.append("</ul>")
    else:
        out.append('<p class="mut">None counted (gap check FAILED above).</p>' if failures else '<p class="mut">None.</p>')
    out.append(f"<h3>Capture ({len(capture)} shot(s), {sum(int(x.get('frames', 0)) for x in capture)} frame-subject(s)): the original: no piece made for these frames (no capture here, or a capture that holds no track of her)</h3>")
    out.append(f'<p class="mut">Kind: <code>original_tracking_not_known</code> ({E(KIND_WORDS["original_tracking_not_known"])}).</p>')
    if capture:
        out.append("<ul>")
        for x in capture:
            s0, s1 = int(x["shot"][0]), int(x["shot"][1])
            bk = x.get("by_kind") or {}
            kinds = ", ".join(f"{k}: {int(bk.get(k, 0))}" for k in CAPTURE_GAP_KINDS if int(bk.get(k, 0)))
            subs = ", ".join(str(s) for s in (x.get("subjects") or []))
            c_runs = [r for r in all_runs if r.get("kind") in CAPTURE_GAP_KINDS and r.get("shot") and [int(r["shot"][0]), int(r["shot"][1])] == [s0, s1]]
            spans = fmt_shot_spans(x, c_runs, s0, s1, rate)
            out.append(f"<li><b>{s0}-{s1}</b> ({E(subs)}; gap(s): {spans}): <b>{int(x['frames'])} frame-subject(s)</b> &mdash; {E(kinds)}</li>")
        out.append("</ul>")
    else:
        out.append('<p class="mut">None counted (gap check FAILED above).</p>' if failures else '<p class="mut">None.</p>')
    out.append("<h3>Other runs from the gap check</h3><ul>")
    for k in OTHER_RUN_KINDS:
        k_runs = [r for r in all_runs if r.get("kind") == k]
        n_fs = sum(int(r["last"]) - int(r["first"]) + 1 for r in k_runs)
        count_str = f"{len(k_runs)} seam(s)" if k == "seam" else f"{n_fs} frame-subject(s)"
        where = (" on " + ", ".join(f"{int(r['first']) if int(r['first']) == int(r['last']) else str(int(r['first'])) + '-' + str(int(r['last']))} ({E(str(r.get('subject') or ''))})" for r in k_runs)) if k_runs else ""
        out.append(f"<li><code>{E(k)}</code> ({E(KIND_WORDS[k])}): <b>{count_str}</b>{where}</li>")
    out.append("</ul></div>")
    return out


#: The buttons that settle a piece for the owner: a keep, or that it is not the one (the table's own words). Any
#: other button ("not sure") leaves it to review.
SETTLED = tuple(delivery_table.KEEP) + tuple(delivery_table.NOT_THE_ONE)


class Rows(frozenset):
    """The rows the owner's lines on a render may stand under, and the render (`piece`) they are of."""
    piece: str | None = None


class Records:
    """The table, the ledger, the review list and the owner's lines, read once, with the day of each piece."""

    def __init__(self, table: str, ledger: str, verdicts: str, items: str, tables: list[str] | None = None):
        self.table_path, self.ledger_path, self.items_path = table, ledger, items
        self.T = json.load(open(table))
        self.L = json.load(open(ledger))
        named = re.search(r"from the ledger `([^`]+)`", str(self.T.get("made") or ""))
        if not named:
            raise SystemExit(f"--table {table}: the table names no ledger it was made from ({str(self.T.get('made'))[:200]})")
        made_from = in_repo(named.group(1), table)
        if made_from.resolve() != Path(ledger).resolve():               # by its path: a ledger of the same name elsewhere is another ledger
            raise SystemExit(f"--ledger {ledger}: the table says it was made from another ledger, {named.group(1)}")
        self.gap_check, self.gap_check_path = read_gap_check(self.T, table)
        self.by_name = {r["name"]: r for r in self.L["renders"].values()}
        self.lines = read_verdicts(verdicts)
        self.items_in_order = json.load(open(items))["items"]
        self.items = {i["id"]: i for i in self.items_in_order}
        missing = sorted({r["name"] for r in self.T["rows"]} - set(self.by_name))
        if missing:
            raise SystemExit(f"the table names render(s) its ledger does not have: {', '.join(missing)}: not the ledger it was made from")
        # the joined items whose own table is handed in (`--tables`), by each piece they lay: a KEEP there counts on each (the
        # lead's ruling, 2026-10-14, as `delivery_table.last_choice` counts it)
        self.joined_lays: dict[str, list[str]] = {}
        for it in self.items_in_order:
            if "_joined_" in it["id"]:
                t = next((Path(f) / f"t_{it['id'].removesuffix('_stacked')}.json" for f in tables or []
                          if (Path(f) / f"t_{it['id'].removesuffix('_stacked')}.json").is_file()), None)
                for r in (json.load(open(t))["rows"] if t else []):
                    self.joined_lays.setdefault(r["name"], []).append(it["id"])
        days = sorted({self.day(r["name"]) for r in self.T["rows"]}, reverse=True)
        self.day_token = {d: DAY_TOKENS[min(k, len(DAY_TOKENS) - 1)] for k, d in enumerate(days)}

    def item_of(self, name: str) -> dict | None:
        """The newest item of a render on the list: its id is the render's name, or its `name` is (the 10th's items)."""
        return next((i for i in self.items_in_order if i["id"] == name), None) or next((i for i in self.items_in_order if i.get("name") == name), None)

    def rows_of(self, name: str) -> Rows:
        """Every row the owner's lines on a render may stand under: its name and the id of every item of it."""
        out = Rows({name} | {i["id"] for i in self.items_in_order if i["id"] == name or i.get("name") == name})
        out.piece = name
        return out

    def owner_lines(self, rows, every=False) -> list[dict]:
        """The owner's lines on these rows, by time; for a piece, with the KEEPS on every joined item that lays it, marked
        (a keep counts on each piece a joined clip lays: the lead's ruling, 2026-10-14). With `every`, all their lines on
        those joined items, clicks and words, for what they have ANSWERED (a sixth reader, 2026-10-14: a "wrong" and words
        on an older joined clip of the same pieces vanished from both pages)."""
        own = [v for r in rows for v in self.lines.get(r, [])]
        piece = getattr(rows, "piece", None)
        if piece:
            own += [dict(v, through_the_joined_item=j) for j in self.joined_lays.get(piece, []) for v in self.lines.get(j, [])
                    if every or v.get("what") == "choice" and v.get("chosen") in delivery_table.KEEP]
        return sorted(own, key=lambda v: v["at"])

    def last_choice(self, rows, every=False) -> dict | None:
        return next((v for v in reversed(self.owner_lines(rows, every)) if v.get("what") == "choice"), None)

    def answered(self, rows, today: str) -> str:
        """Where an entry goes on to-review, by the owner's lines on it (the lead's rulings of 2026-10-14): `waiting` with no line
        of theirs; `gone` when they settled it, or kept it and wrote after, or wrote with no click, before `today`; else `done`."""
        lines = self.owner_lines(rows, every=True)
        if not lines:
            return "waiting"
        last_day = owners(lines[-1]["at"], "%Y-%m-%d")
        # on the same lines the page shows (a seventh reader, 2026-10-14: a not sure on a joined clip after a keep of the piece's own
        # was filed by the keep, and left the page the next day)
        c = self.last_choice(rows, every=True)
        closed = (c is None or c["chosen"] in delivery_table.KEEP
                  or c["chosen"] in delivery_table.NOT_THE_ONE and not c.get("through_the_joined_item"))
        return "gone" if closed and last_day != today else "done"

    def words_after(self, rows) -> list[dict]:
        """The owner's words after their last click (all their words where there is no click): what rules over it."""
        c = self.last_choice(rows)
        return [v for v in self.owner_lines(rows) if v.get("words") and (c is None or v["at"] > c["at"])]

    def words_with(self, rows, every=False) -> list[dict]:
        """The owner's words since the click before their last: the words that went with the last click, typed before
        it or after it (a reader, 2026-10-14: words typed two seconds before a click were hidden)."""
        lines = self.owner_lines(rows, every)
        choices = [v for v in lines if v.get("what") == "choice"]
        before = choices[-2]["at"] if len(choices) > 1 else None
        return [v for v in lines if v.get("words") and (before is None or v["at"] > before)]

    def settled(self, rows, by=SETTLED) -> bool:
        """Settled by a click in `by` with no words of the owner's after it: their words rule over the button
        (the owner's rule, 2026-10-12), so a keep followed by words is to review again, with the words."""
        c = self.last_choice(rows)
        return bool(c and c["chosen"] in by and not self.words_after(rows))

    def day(self, name: str) -> str:
        return owners(self.by_name[name]["written"], "%Y-%m-%d")

    def day_badge(self, name: str) -> str:
        d = self.day(name)
        return f'<span class="d {self.day_token[d]}">{owners(self.by_name[name]["written"])}</span>'

    def last_click(self, rows, newest: str | None = None) -> str:
        """The owner's last button, on the piece or on any joined clip that lays it, and the words that went with it: the words
        rule over the button. Each line says its frame where it names one, and, given the `newest` of several builds of a joined
        clip, the earlier build a line was said on (an eighth reader, 2026-10-14: 1849Z's words read as said of 1855Z)."""
        c = self.last_choice(rows, every=True)
        on = lambda v: (f", at frame {v['frame']}" if v.get("frame") is not None else "") + (                                # noqa: E731
            f", on the earlier build {v['row']}" if newest and v.get("row") and v["row"] != newest else "")
        said = (f"your last click: {c['chosen']} ({owners(c['at'], '%m/%d %H:%M')}{on(c)})"
                + ((f", on the whole joined clip {c['through_the_joined_item']}, which counts on each piece it lays" if c["chosen"] in delivery_table.KEEP
                    else f", on the whole joined clip {c['through_the_joined_item']}, which names no piece") if c.get("through_the_joined_item") else "")
                if c else "no click of yours")
        when = lambda v: "since" if c is None or v["at"] > c["at"] else "before it"        # noqa: E731
        return said + "".join(f'; {how_said(v)} {when(v)} ({owners(v["at"], "%m/%d %H:%M")}{on(v)})'
                              + (f" on the whole joined clip {v['through_the_joined_item']}" if v.get("through_the_joined_item") else "")
                              + f': "{v["words"]}"' for v in self.words_with(rows, every=True))

    def left_by_ledger(self, name: str) -> set[int]:
        lo = (self.by_name.get(name) or {}).get("leaves_as_the_original") or {}
        return {x for a, b in lo.get("source_frames") or [] for x in range(a, b + 1)}

    def subjects(self) -> list[str]:
        seen: list[str] = []
        for rd in self.T["reading"]:
            if rd["subject"] not in seen:
                seen.append(rd["subject"])
        return seen


BASE_CSS = """body{font:16px/1.5 system-ui,sans-serif;margin:0 auto;padding:16px;background:var(--bg);color:var(--fg)}
a{color:var(--link)} h2{margin-top:1.6em;border-bottom:1px solid var(--line);padding-bottom:.2em} .mut,.why{color:var(--mut)}
.e,.card{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0}
.e h3,.card h3{margin:.2em 0;font-size:1.05em} .e h3 span,.card h3 span{font-weight:normal;font-size:.9em;color:var(--mut)}
ul{margin:.2em 0 .4em 1.2em;padding:0} .d{font-weight:700} ol li{margin:.35em 0}
"""


def page(title: str, extra_css: str, body: str, width: int) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{E(title)}</title><style>\n{theme_css()}{BASE_CSS}body{{max-width:{width}px}}\n{extra_css}</style></head><body>\n{body}\n</body></html>\n")


# ---------------------------------------------------------------- to-review

def shots_of_table(T: dict) -> list[dict]:
    """The table's record shots, each with the subjects in it (present or unsure in its reading) and its rows."""
    out: dict[tuple[int, int], dict] = {}
    for rd in T["reading"]:
        one = out.setdefault((rd["first"], rd["last"]), {"first": rd["first"], "last": rd["last"], "subjects": [], "unsure": []})
        if rd.get("present") in ("yes", "unsure"):
            one["subjects"].append(rd["subject"])
        if rd.get("present") == "unsure":
            one["unsure"].append(rd["subject"])
    for one in out.values():
        one["rows"] = [r for r in T["rows"] if r["first"] <= one["last"] and r["last"] >= one["first"]]
    return sorted(out.values(), key=lambda x: x["first"])


def joined_item(rec: "Records", first: int, last: int, tables: list[str]) -> tuple[dict | None, str]:
    """(the newest joined clip on the list of exactly these frames, and whether it is of THIS table) as (item, why).
    A clip is this table's when the table its build wrote (`<tables>/t_<its name>.json`, bench/shot_review_clips.py)
    lays on these frames the same pieces as this one: a clip built from an earlier table is not shown as this one's."""
    it = next((i for i in rec.items_in_order if "_joined_" in i["id"] and str(i.get("first")) == str(first)
               and str(i.get("frames")) == str(last - first + 1)), None)
    if it is None:
        return None, "no joined clip of this shot is on the list"
    built = next((Path(t) / f"t_{it['id'].removesuffix('_stacked')}.json" for t in tables if (Path(t) / f"t_{it['id'].removesuffix('_stacked')}.json").is_file()), None)
    if built is None:
        return None, f"the newest joined clip of this shot ({it['id']}) has no table under --tables to say what it lays"
    lays = lambda T: sorted((r["name"], r["subject"], max(r["first"], first), min(r["last"], last)) for r in T["rows"] if r["first"] <= last and r["last"] >= first)
    theirs, ours = lays(json.load(open(built))), lays(rec.T)
    if theirs != ours:
        return None, (f"the newest joined clip of this shot ({it['id']}) lays {', '.join(n for n, *_ in theirs) or 'nothing'}, and this table lays "
                      f"{', '.join(n for n, *_ in ours) or 'nothing'}: it is of an earlier table, and a joined clip of this one is to be built")
    return it, ""


def kin_of(rec: "Records", it: dict, first: int, last: int, tables: list[str]) -> list[str]:
    """The joined items of exactly these frames whose own table lays the same pieces on them as this one: the clip and every
    earlier build of the same thing, whose lines it inherits (the lead's ruling of 2026-10-14, on a seventh reader's 1222-1305)."""
    lays = lambda T: sorted((r["name"], r["subject"], max(r["first"], first), min(r["last"], last)) for r in T["rows"] if r["first"] <= last and r["last"] >= first)  # noqa: E731
    ours, out = lays(rec.T), []
    for i in rec.items_in_order:
        if "_joined_" in i["id"] and str(i.get("first")) == str(first) and str(i.get("frames")) == str(last - first + 1):
            built = next((Path(t) / f"t_{i['id'].removesuffix('_stacked')}.json" for t in tables if (Path(t) / f"t_{i['id'].removesuffix('_stacked')}.json").is_file()), None)
            if built is not None and lays(json.load(open(built))) == ours:
                out.append(i["id"])
    return out if it["id"] in out else [it["id"], *out]


def replaced(rec: "Records", previous: dict | None, name: str, subject: str, first: int, last: int) -> list[dict] | None:
    """What the piece replaces: the rows of the same subject the previous table laid on its frames (the same piece
    among them, where it stood there before); None when no previous table was given."""
    if previous is None:
        return None
    return [r for r in previous["rows"] if r["subject"] == subject and r["first"] <= last and r["last"] >= first]


def piece_line(rec: "Records", r: dict, first: int, last: int, link, previous: dict | None) -> str:
    span = range(max(r["first"], first), min(r["last"], last) + 1)
    if set(span) <= rec.left_by_ledger(r["name"]):
        return (f'<li><b>{E(r["subject"])}</b>, {span[0]}-{span[-1]}: the original (inside {E(r["name"])}, whose load emptied its part on these frames) '
                f'&middot; {E(rec.last_click(rec.rows_of(r["name"])))}</li>')
    it = rec.item_of(r["name"]) or {}
    was = replaced(rec, previous, r["name"], r["subject"], span[0], span[-1])
    named = lambda n: (link(rec.item_of(n)["id"], n) if rec.item_of(n) else E(n)) + (f" (rendered {owners(rec.by_name[n]['written'])})" if n in rec.by_name else "")
    said = ("not known here: no previous table was given" if was is None else "the original: the previous table laid no piece of this subject here"
            if not was else ", ".join(("nothing: this same piece stood here in the previous table" if x["name"] == r["name"] else named(x["name"]))
                                      + f" on {max(x['first'], span[0])}-{min(x['last'], span[-1])}" for x in was))
    return (f'<li>{rec.day_badge(r["name"])} <b>{E(r["subject"])}</b>, {span[0]}-{span[-1]}: '
            + (link(it["id"], r["name"]) if it else E(r["name"])) + f' &middot; {E(rec.last_click(rec.rows_of(r["name"])))}'
            + (f'<div class="why">the card\'s look line (a session\'s words): {E(it["look"])}</div>' if it.get("look") else "")
            + f'<div class="why">it replaces, in the previous table: {said}</div></li>')


def since(rec: "Records", previous: str) -> list[str]:
    """The pieces added and the pieces that left against the previous table, and the frames each subject gained or lost."""
    P = json.load(open(previous))

    def span(TT):
        shots = list(TT.get("reading") or []) + list(TT.get("not_in_the_shot") or [])
        rr = shots or TT["rows"]
        return (min(r["first"] for r in rr), max(r["last"] for r in rr)) if rr else None
    # only the frames both tables cover are compared: against a table of other frames (a span's, or the whole video's),
    # what lies outside one of them neither came nor went (a reader, 2026-10-14)
    ours, theirs = span(rec.T), span(P)
    lo, hi = (max(ours[0], theirs[0]), min(ours[1], theirs[1])) if ours and theirs else (1, 0)
    inside = lambda TT: [r for r in TT["rows"] if r["first"] <= hi and r["last"] >= lo]
    new_rows, old_rows = {(r["name"], r["subject"]) for r in inside(rec.T)}, {(r["name"], r["subject"]) for r in inside(P)}
    cover = lambda TT, sub: {x for r in inside(TT) if r["subject"] == sub for x in range(max(r["first"], lo), min(r["last"], hi) + 1)}
    out = [f"<p class='mut'>Against {E(os.path.basename(previous))}, on the frames both tables cover"
           + (f": {lo}-{hi} (this one covers {ours[0]}-{ours[1]}, that one {theirs[0]}-{theirs[1]})." if lo <= hi else ": none, so nothing is compared.")
           + "</p>", "<ul>"]
    out += [f"<li>added: {E(n)} ({E(sub)})</li>" for n, sub in sorted(new_rows - old_rows)] or ["<li>no piece added</li>"]
    out += [f"<li>left: {E(n)} ({E(sub)})</li>" for n, sub in sorted(old_rows - new_rows)] or ["<li>no piece left</li>"]
    for sub in rec.subjects():
        now, then = cover(rec.T, sub), cover(P, sub)
        closed, opened = runs(now - then), runs(then - now)
        if closed:
            out.append(f"<li>{E(sub)}: a piece now on " + ", ".join(f"{p}-{q}" for p, q in closed) + "</li>")
        if opened:
            out.append(f"<li>{E(sub)}: no piece now on " + ", ".join(f"{p}-{q}" for p, q in opened) + "</li>")
    return out + ["</ul>"]


def held_and_set_aside(rec: "Records") -> list[str]:
    """The lead's holds (the table's `held`: a session's word, the reason whole) and the owner's keeps this table does
    not lay (a render passed over on a shot whose last choice is a keep, with the table's reason), as page lines."""
    out = ["<h2>Held back by the lead, and your keeps this table does not lay</h2>"]
    held = rec.T.get("held") or []
    out += [f'<div class="e"><h3>{E(h["load"])} <span>{E(h["subject"])} &middot; {h["first"]}-{h["last"]}</span></h3><p class="why">HELD BACK BY THE LEAD, '
            f'a session\'s word and not yours: {E(h["reason"])}</p></div>' for h in held]
    kept = []
    for rd in rec.T["reading"]:
        for p in rd.get("passed_over") or []:
            c = p.get("the_owners_last_choice") or {}
            if c.get("said") in delivery_table.KEEP:
                after = [w for w in p.get("the_owners_words") or [] if str(w.get("at")) > str(c["at"])]
                kept.append(f'<li><b>{E(rd["subject"])}</b>, {rd["first"]}-{rd["last"]}: your keep of {E(p["render"])} ({E(c["said"])}, '
                            f'{owners(c["at"], "%m/%d %H:%M")}) is not laid here. The table\'s reason: {E(p["why"])}'
                            + "".join(f'. Your words after the keep ({owners(w["at"], "%m/%d %H:%M")}'
                                      + (f', from {E(str(w["from"]))}' if w.get("from") else ", where they came from is not said")
                                      + f'): "{E(w["said"])}"' for w in after) + "</li>")
    out += ([f"<ul>{''.join(kept)}</ul>"] if kept else []) + ([] if held or kept else ["<p class='mut'>None.</p>"])
    return out

def start_here(path: str, drawn: dict[str, str], shots: dict[str, list[str]], out_dir: str) -> list[str]:
    """START HERE, IN THIS ORDER: the lead's list (the owner, 2026-10-15: the most important first), each entry linking what the page
    shows below, or a clip that is no item, with its one line of why and optional `reader_notes` in a closed fold headed as a reader's,
    not the owner's. An entry the page does not hold, a clip that is not there, an entry with no why, or empty `reader_notes` stops the
    page: the one kind of red."""
    import re
    got = json.load(open(path))
    if not isinstance(got, list) or not got:
        raise SystemExit(f"--start-here {path}: a JSON list of {{\"item\", \"why\"}}, in order, and not empty")
    lines = []
    def clip_link(k: int, clip: str) -> str:
        if not Path(clip).is_file():
            raise SystemExit(f"--start-here {path}, entry {k}: the clip {clip} is not there")
        return f'<a href="{E(os.path.relpath(clip, out_dir))}">{E(Path(clip).name)}</a>'
    def reader_notes_fold(k: int, e: dict) -> str:
        if "reader_notes" not in e:
            return ""
        rn = e["reader_notes"]
        by_str = " ".join(str(e.get("reader_by") or "").split())
        if isinstance(rn, str):
            s = " ".join(rn.split())
            if not s:
                raise SystemExit(f"--start-here {path}, entry {k}: reader_notes is a non-empty string or list of non-empty strings")
            note_lines = [s]
        elif isinstance(rn, list):
            if not rn or any(not isinstance(x, str) or not " ".join(x.split()) for x in rn):
                raise SystemExit(f"--start-here {path}, entry {k}: reader_notes is a non-empty string or list of non-empty strings")
            note_lines = [" ".join(x.split()) for x in rn]
        elif isinstance(rn, dict):
            by_str = " ".join(str(rn.get("by") or by_str).split())
            raw_list = rn.get("notes") if "notes" in rn else rn.get("lines")
            if isinstance(raw_list, str) and " ".join(raw_list.split()):
                note_lines = [" ".join(raw_list.split())]
            elif isinstance(raw_list, list) and raw_list and all(isinstance(x, str) and " ".join(x.split()) for x in raw_list):
                note_lines = [" ".join(x.split()) for x in raw_list]
            else:
                raise SystemExit(f"--start-here {path}, entry {k}: reader_notes is a non-empty string or list of non-empty strings")
        else:
            raise SystemExit(f"--start-here {path}, entry {k}: reader_notes is a non-empty string or list of non-empty strings")
        who_said = f" ({E(by_str)})" if by_str else ""
        items_html = "".join(f"<li>{E(ln)}</li>" for ln in note_lines)
        return (f'<details class="why"><summary>A reader\'s notes{who_said} — a session\'s reading, not the owner\'s (not yours)</summary>'
                f'<ul>{items_html}</ul></details>')
    for k, e in enumerate(got, 1):
        what, why = str(e.get("item") or "").strip(), " ".join(str(e.get("why") or "").split())
        url = str(e.get("link") or "").strip()
        if (not what and not url) or not why:
            raise SystemExit(f"--start-here {path}, entry {k}: each entry names an item or a link and says in one line what deciding it changes")
        fold = reader_notes_fold(k, e)
        if url:
            # a question whose buttons are elsewhere (the Open List): the entry goes there, its clips beneath it as plain links
            # (the owner, 2026-10-15: a clip's own link "just gives me the rendered video", and the buttons were not on it)
            if what or not re.match(r"https?://", url) or not str(e.get("label") or "").strip():
                raise SystemExit(f"--start-here {path}, entry {k}: a link entry is an http(s) URL with a label, and names no item beside it")
            clips = [clip_link(k, c) for c in e.get("clips") or []]
            lines.append(f'<li><a href="{E(url)}">{E(" ".join(str(e["label"]).split()))}</a>: {E(why)}'
                         + (f"<br><span class='mut'>the clips: {', '.join(clips)}</span>" if clips else "") + f"{fold}</li>")
            continue
        if re.fullmatch(r"\d+-\d+", what):
            if what not in shots:
                raise SystemExit(f"--start-here {path}, entry {k}: the shot {what} is not on this page (its shots: {', '.join(shots) or 'none'})")
            ins = shots[what]
            shown = f"the shot {E(what)}: " + (", ".join(drawn[i] for i in ins) if ins else "its pieces, below")
        elif what in drawn:
            shown = drawn[what]
        elif "/" in what:
            shown = clip_link(k, what) + " (a clip, not on the review list)"
        else:
            raise SystemExit(f"--start-here {path}, entry {k}: {what!r} is no item this page shows, no shot of it and no clip")
        lines.append(f"<li>{shown}: {E(why)}{fold}</li>")
    return ["<h2>Start here, in this order</h2>", "<p class='mut'>The lead's order: what deciding each changes most. Each is also "
            "where it always is, below.</p>", f"<ol>{''.join(lines)}</ol>"]


def laid_from(T: dict) -> str:
    """What the table says it laid from (the owner's order of 2026-10-16): the list of what is laid and its sha256, read
    from the table's own `made` line, or that the table was made by the older rule."""
    m = re.search(r"ONLY the list of what is laid, `([^`]+)` \(sha256 ([0-9a-f]+), (\d+) lines", str(T.get("made") or ""))
    if m:
        return f" What is laid is the list {E(m.group(1))} (sha256 {E(m.group(2))}, {E(m.group(3))} lines), and nothing else."
    return " What is laid was decided by the table's older rule (keeps, a session's preferences, the newest render), not by one list."


def to_review(a: argparse.Namespace) -> str:
    rec = Records(a.table, a.ledger, a.verdicts, a.items, a.tables)
    rate, out_dir = rate_of(a.source), os.path.dirname(os.path.abspath(a.out))
    previous = json.load(open(a.previous_table)) if a.previous_table else None
    drawn: dict[str, str] = {}                           # each item the page links, to its link: what --start-here may name

    this_shot: list[str] = []                            # the items the shot being drawn links

    def link(i, label=None):
        drawn.setdefault(i, f'<a href="{E(a.page_url + i)}">{E(i)}</a>')
        this_shot.append(i)
        return f'<a href="{E(a.page_url + i)}">{E(label or i)}</a>'
    shots_drawn: dict[str, list[str]] = {}               # each shot the page shows, FIRST-LAST, to the items its entry links
    stem = Path(a.source).stem
    body = [f"<h1>{E(stem)}: to review</h1>",
            f'<p class="mut">Made {datetime.datetime.now(OWNERS_CLOCK):%Y-%m-%d %H:%M} US Central from the table {E(os.path.basename(a.table))} and its '
            f"ledger.{laid_from(rec.T)} Each piece is marked with the day its file was written: "
            + ", ".join(f'<span class="d {t}">{datetime.date.fromisoformat(d):%m/%d}</span>' for d, t in rec.day_token.items())
            + ". Shown: what this table lays and waits on you, then, folded, what you answered: your lines of today, and a not sure or a click on "
            "a joined clip that names no piece until you say more. The joined clips' tables were read from: "
            + (", ".join(E(str(t)) for t in a.tables) if a.tables else "no folder (no joined clip can be this table's)") + ".</p>",
            f'<p>Decisions waiting on you are on <a href="{E(a.open_list)}">the Open List</a>.'
            + (f' Beside this file: <a href="{E(a.provenance)}">{E(a.provenance)}</a>, every stretch and every gap.' if a.provenance else "") + "</p>"]
    # WAITING ON YOU, AND DONE: each entry by the owner's last click on it (the lead's page of 2026-10-14, at the owner's ask of
    # which items were still theirs): no click is waiting on them; a click of theirs, or a piece they settled today, is done
    today = a.today or datetime.datetime.now(OWNERS_CLOCK).strftime("%Y-%m-%d")
    waiting = {"joined": [], "apart": [], "single": [], "joined_one": []}
    done = {"joined": [], "apart": [], "single": [], "joined_one": []}
    entries_before, last_shot = 0, None

    def shot_drawn(sh=None):
        """The shot just drawn, when its iteration added an entry, is kept with the items it links; then the next begins."""
        nonlocal entries_before, last_shot
        now = sum(map(len, waiting.values())) + sum(map(len, done.values()))
        if last_shot is not None and now > entries_before:
            shots_drawn[f"{last_shot['first']}-{last_shot['last']}"] = list(dict.fromkeys(this_shot))
        entries_before, last_shot = now, sh
        this_shot.clear()
    for sh in shots_of_table(rec.T):
        first, last = sh["first"], sh["last"]
        when = f"{clock(first, rate)} to {clock(last + 1, rate)} of the video"
        rows = sorted(sh["rows"], key=lambda r: (r["subject"], r["first"]))
        if not rows:
            continue
        lines = lambda rr: "".join(piece_line(rec, r, first, last, link, previous) for r in rr)
        # the heading IS the way in (the owner, 2026-10-14, could not find the links): each piece it shows, by its item, named
        # a piece is named once, however many rows its restore spans cut it into (2026-10-15: 2823b's three rows read as three pieces)
        heads = lambda rr: ", ".join(link(rec.item_of(n)["id"], n) if rec.item_of(n) else  # noqa: E731
                                     f'{E(n)} (no item of it on the review list yet)' for n in dict.fromkeys(r["name"] for r in rr))
        who = ", ".join(x + (" (the record is unsure)" if x in sh["unsure"] else "") for x in sh["subjects"])
        state = {r["name"]: rec.answered(rec.rows_of(r["name"]), today) for r in rows}
        shot_drawn(sh)
        # a shot with a joined clip of this table on the list is shown by it, whatever its count of subjects (2026-10-15: the joined
        # clip of the one-subject 2823-3166 was on the list and the page showed its piece's rows, "no item of it on the review list yet")
        it, why = joined_item(rec, first, last, a.tables or [])
        if len(sh["subjects"]) >= 2 or it is not None:
            sect = "joined" if len(sh["subjects"]) >= 2 else "joined_one"
            if it is not None:
                # the joined clip is filed by the owner's lines on it: none, waiting; a keep (with words after or not) leaves the
                # page after the day of their last line, and only when no piece of it has a later line of theirs; any other
                # click ("wrong" names no piece) stays under Done
                kin = set(kin_of(rec, it, first, last, a.tables or []))
                own = rec.owner_lines(kin)
                c = rec.last_choice(kin)
                kept = bool(c and c["chosen"] in delivery_table.KEEP)
                # a joined clip nobody watched whose every piece the owner kept is not waiting on them (the lead's ruling of 2026-10-14):
                # under Done on the day of their last line, as "every piece kept; the joined clip not watched", and gone after it
                each = [rec.last_choice(rec.rows_of(r["name"]), every=True) for r in rows]
                if not own and all(x and x["chosen"] in delivery_table.KEEP for x in each):
                    if any(v != "gone" for v in state.values()):
                        done[sect].append(f'<div class="e"><h3>{link(it["id"], f"The shot {first}-{last}")} <span>{E(who)} &middot; {when} &middot; '
                                              f'every piece kept; the joined clip not watched</span></h3><ul>{lines(rows)}</ul></div>')
                    continue
                # a keep, or words with no click, leave the page after the day of the last line; any other click stays (a seventh reader)
                gone = own and (kept or c is None) and owners(own[-1]["at"], "%Y-%m-%d") != today and all(v == "gone" for v in state.values())
                if not gone:
                    ask = ('<p class="why">Your click on the whole clip names no piece: which piece is it? Say it on that piece\'s own line below.</p>'
                           if c and not kept else "")
                    older = sorted(kin - {it["id"]})
                    (done if own else waiting)[sect].append(
                        f'<div class="e"><h3>{link(it["id"], f"The shot {first}-{last}")} <span>{E(who)} &middot; {when} &middot; the joined clip: '
                        f'{E(rec.last_click(kin, newest=it["id"]))}</span></h3>'
                        + (f'<p class="why">With your lines on the earlier build(s) of the same pieces on the same frames: {E(", ".join(older))}</p>' if older else "")
                        + f'{ask}<ul>{lines(rows)}</ul></div>')
                continue
            for to, kind in ((waiting, "waiting"), (done, "done")):
                rr = [r for r in rows if state[r["name"]] == kind]
                if rr:
                    to["apart"].append(f'<div class="e"><h3>The shot {first}-{last}: {heads(rr)} <span>{E(who)} &middot; {when}</span></h3>'
                                       f'<p class="why">{E(why)}. Each piece meanwhile, on its own:</p><ul>{lines(rr)}</ul></div>')
        else:
            # one entry a piece, its rows (cut by restore spans) together under it
            for name in dict.fromkeys(r["name"] for r in rows):
                rr = [r for r in rows if r["name"] == name]
                if state[name] != "gone":
                    (waiting if state[name] == "waiting" else done)["single"].append(
                        f'<div class="e"><h3>{E(rr[0]["subject"])}, {max(rr[0]["first"], first)}-{min(rr[-1]["last"], last)}: {heads(rr)} <span>{when}</span></h3>'
                        f'<ul>{lines(rr)}</ul></div>')

    shot_drawn()

    def sections(got: dict, tag: str) -> list[str]:
        out = [f"<{tag}>Shots with two or more of them, each as its joined clip</{tag}>", *(got["joined"] or ["<p class='mut'>None.</p>"])]
        if got["apart"]:
            out += [f"<{tag}>Shots with two or more of them with no joined clip of this table</{tag}>", *got["apart"]]
        if got["joined_one"]:
            out += [f"<{tag}>Shots of one of them, as its joined clip</{tag}>", *got["joined_one"]]
        return out + [f"<{tag}>Pieces in a shot of one of them</{tag}>", *(got["single"] or ["<p class='mut'>None.</p>"])]
    n_wait, n_done = (sum(map(len, g.values())) for g in (waiting, done))
    if a.start_here:
        body += start_here(a.start_here, drawn, shots_drawn, os.path.dirname(os.path.abspath(a.out)))
    body += [
            *gap_check_block(rec, rate)]
    if previous is not None:
        body += ["<h2>Since the last page</h2>", *since(rec, a.previous_table)]
    body += [f"<h2>Waiting on you: {n_wait} with no click or words of yours yet</h2>",
             "<p class='mut'>Watch these on playback, top to bottom. Each is something this table lays that you have not answered: a shot with a "
             "joined clip by the joined clip, whatever its pieces' own lines say (each piece's line says them).</p>",
             *sections(waiting, "h3"),
             f"<details><summary><b>Done: {n_done} you have answered</b>, the last click and your words with each</summary>",
             f"<p class='mut'>Each has a line of yours. A keep or not-the-one, a keep with your words after it, or words with no click stay here on "
             f"the day of your last line on it ({datetime.date.fromisoformat(today):%m/%d}, your clock) and leave the page after it; your words reach "
             "the sessions through the table and the item. A not sure, and a click on a joined clip that is not a keep (it names no piece), stay "
             "here until you say more.</p>",
             *sections(done, "h3"), "</details>"]
    body += held_and_set_aside(rec)
    return page(f"To review: {stem}", "", "\n".join(body), 900)


# ---------------------------------------------------------------- provenance

def left_out_of(rec: Records, name: str) -> list[dict]:
    """The job's own `left_out` list, beside its build record; empty where the record names no build file."""
    built = ((rec.by_name.get(name, {}).get("records") or [{}])[0]).get("built_file")
    job = in_repo(built, rec.ledger_path).parent / "job.render.json" if built else None
    if not job or not job.is_file():
        return []
    return json.loads(job.read_text()).get("job", {}).get("left_out") or []


def why_left(rec: Records, name: str, sub: str, a: int, b: int) -> str:
    """Why a piece leaves these frames as the original: a look's word in its job (`left_out`) where one covers them,
    else what the ledger read the frames from (the capture's manifest: the frames on which it emptied the part)."""
    for lo in left_out_of(rec, name):
        if lo.get("subject") in (sub, None) and lo["first"] <= b and a <= lo["first"] + lo["frames"] - 1:
            return "a session's word in the load's job file (its `left_out`, a look's): " + " ".join(str(lo.get("why", "")).split())
    said = ((rec.by_name.get(name) or {}).get("leaves_as_the_original") or {}).get("read_from")
    return f"its load emptied the part here, by the ledger ({said})" if said else ""


def stand_why(s: dict) -> str:
    """Why a piece stands, in the table's words; the lead's --prefer said as a session's word, never as the table's rule."""
    if s.get("the_leads_prefer"):
        return f"THE LEAD'S CHOICE (--prefer), a session's word: {s['the_leads_prefer']}"
    return s.get("why", "")


def stretches_of(rec: Records, nothing: dict, up_to: int) -> list[dict]:
    """Every run, a subject at a time inside one shot of the table's reading: a piece, left as the original inside a
    piece, or nothing there."""
    rows_of = {s: [r for r in rec.T["rows"] if r["subject"] == s] for s in rec.subjects()}
    out = []
    for rd in rec.T["reading"]:
        sub, a0, b0 = rd["subject"], rd["first"], min(rd["last"], up_to)
        if a0 > up_to or rd.get("present") == "no":
            continue
        state = {}
        for r in rows_of[sub]:
            for x in range(max(r["first"], a0), min(r["last"], b0) + 1):
                state[x] = ("left", r["name"]) if x in nothing.get(r["name"], ()) else ("piece", r["name"])
        for kind in ("piece", "left", "none"):
            groups: dict[str, list[int]] = {}
            for x in range(a0, b0 + 1):
                k, n = state.get(x, ("none", ""))
                if k == kind:
                    groups.setdefault(n, []).append(x)
            for n, fr in groups.items():
                for a, b in runs(fr):
                    if kind == "piece":
                        why = stand_why(next((s for s in rd.get("stands") or [] if s["render"] == n), {}))
                    elif kind == "left":
                        why = why_left(rec, n, sub, a, b) or "no record read here says why"
                    else:
                        said = next((s for s in rec.T["sources"] + rec.T["gaps"] if s["subject"] == sub and s["first"] <= a <= s["last"]), {})
                        why = said.get("why", "")
                    out.append({"a": a, "b": b, "sub": sub, "kind": kind, "name": n, "why": why, "unsure": rd.get("present") == "unsure"})
    order = rec.subjects()
    return sorted(out, key=lambda s: (s["a"], order.index(s["sub"])))


def changes(rec: Records, watched: str, up_to: int, rate: Fraction) -> list[tuple]:
    """(first, last, subject, in the clip watched, in the next, why) where the piece on a frame differs between tables."""
    PT = json.load(open(watched))

    def state_of(TT):
        return {(r["subject"], x): r["name"] for r in TT["rows"] for x in range(r["first"], min(r["last"], up_to) + 1)}
    old, new = state_of(PT), state_of(rec.T)
    why_gone = {p["render"]: p["why"] for rd in rec.T["reading"] for p in rd.get("passed_over") or []}
    standing = {r["name"] for r in rec.T["rows"]}
    rows = []
    for sub in rec.subjects():
        diff: dict[tuple, list[int]] = {}
        for x in range(0, up_to + 1):
            a, b = old.get((sub, x)), new.get((sub, x))
            if a != b:
                diff.setdefault((a, b), []).append(x)
        for (a, b), fr in diff.items():
            for f0, f1 in runs(fr):
                why = []
                if a and a not in standing:
                    why.append(f"{a}: {why_gone.get(a, 'no line in the reading')}")
                if b:
                    why.append(f"{b}: " + stand_why(next((s for rd in rec.T["reading"] for s in rd.get("stands") or [] if s["render"] == b), {})))
                if a and b is None and a in standing:
                    why.append(f"{a} still stands on its other frames; these are no longer its row's")
                rows.append((f0, f1, sub, a or "the original", b or "the original", " | ".join(why)))
    return sorted(rows)



PROVENANCE_CSS = """.scroll{overflow-x:auto;border:1px solid var(--line);background:var(--card);flex:1 1 auto;min-width:0}
.tlwrap{display:flex} .lanelabels{flex:0 0 auto;font-size:12px;font-weight:600;white-space:nowrap}
.lanelabels div{padding-right:8px;box-sizing:border-box}
.tick{stroke:var(--line);stroke-width:1} .tl{font:11px system-ui;fill:var(--mut)}
rect.gap{fill:none;stroke:var(--gap);stroke-width:2} rect.left{fill:var(--left)} rect.bare{fill:var(--bare)}
.lanes{display:flex;gap:18px;flex-wrap:wrap;margin:.5em 0} .k{display:inline-block;width:22px;height:12px;vertical-align:middle;margin-right:5px}
.k.gap{border:2px solid var(--gap)} .k.left{background:var(--left)} .k.bare{background:var(--bare)}
table{border-collapse:collapse;width:100%;font-size:14px} td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
.tw{overflow-x:auto} .card{border-left:6px solid var(--line)} .gapr td:nth-child(4){color:var(--gap);font-weight:600} .leftr td:nth-child(4){color:var(--mut);font-weight:600}
dl{display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:2px 12px;margin:.4em 0} dt{color:var(--mut)} dd{margin:0;overflow-wrap:anywhere}
pre{white-space:pre-wrap;font-size:13px} :target{outline:3px solid var(--gap);outline-offset:2px}
"""
LANE, TOP = 26, 30          # px a lane and above the first: the svg is drawn at its own height, so the labels beside it line up


def provenance(a: argparse.Namespace) -> str:
    rec = Records(a.table, a.ledger, a.verdicts, a.items, a.tables)
    rate, up_to, day = rate_of(a.source), a.up_to, a.day_folder
    nothing: dict[str, set[int]] = {}
    nothing = {n: rec.left_by_ledger(n) for n in rec.by_name if rec.left_by_ledger(n)}
    left_from = "the ledger"
    if a.record:
        # the record speaks only for the pieces it assembled: a piece of this table it never laid takes the ledger's word (an eighth
        # reader, 2026-10-14: a 10/13 record taken with the 10/14 table said "nothing" of pieces it never saw)
        record = json.load(open(a.record))
        files = {str(r.get("piece", "")) for r in record.get("table") or []}
        held = {n for n, g in rec.by_name.items() if any(f.endswith(str(g["file"])) for f in files)}
        for n in held:
            nothing.pop(n, None)
        for f in record["flags"]:
            if f["rule"] == "piece_changes_nothing" and f["run"] in held:
                nothing.setdefault(f["run"], set()).update(x for p, q in f["source_frames"] for x in range(p, q + 1))
        laid = {r["name"] for r in rec.T["rows"]}
        left_from = (f"the assembler's record {os.path.basename(a.record)} for the {len(held & laid)} piece(s) of this table it assembled, and the ledger for "
                     f"the {len(laid - held)} it did not")
    stopped = rec.L.get("stopped")
    subjects = rec.subjects()
    stretches = stretches_of(rec, nothing, up_to)
    covered = {x for r in rec.T["rows"] for x in range(r["first"], r["last"] + 1)}
    bare = runs(x for x in range(0, up_to + 1) if x not in covered)
    tok = lambda name: rec.day_token[rec.day(name)]
    out_dir = os.path.dirname(os.path.abspath(a.out))           # every link is relative to where the page is written
    rel = lambda path: os.path.relpath(path if os.path.isabs(path) else os.path.join(a.share, path), out_dir)
    stem = Path(a.source).stem
    body = [f"<h1>{E(stem)}: what every stretch is made of, from which day, and where nothing is made</h1>",
            f'<p class="mut">Written {datetime.datetime.now(OWNERS_CLOCK):%Y-%m-%d %H:%M} US Central from the table <b>{E(os.path.basename(a.table))}</b> '
            f"and its ledger;{laid_from(rec.T)[:-1].replace(' What', ' what', 1)}; your clicks as they stood when this was written; the frames a piece leaves as the original from {E(left_from)}. "
            "A local file: nothing here is hosted or tracked. It cannot see the picture: a reason is what a record says.</p>",
            f"<p><b>{len(rec.T['rows'])} pieces: " + ", ".join(
                f'<span class="d {t}">{sum(1 for r in rec.T["rows"] if rec.day(r["name"]) == d)} rendered {datetime.date.fromisoformat(d):%m/%d}</span>'
                for d, t in rec.day_token.items()) + f".</b> Frames 0-{up_to} are shown.</p>",
            *gap_check_block(rec, rate)]
    if a.watched_table:
        ch = changes(rec, a.watched_table, up_to, rate)
        body += ["<h2>What changes in the next clip, against the one you watched, and why</h2>",
                 f"<p class='mut'>The clip you watched was built from {E(os.path.basename(a.watched_table))}; everything else here is from "
                 f"{E(os.path.basename(a.table))}.</p>",
                 "<div class='tw'><table><thead><tr><th>frames</th><th>time</th><th>who</th><th>in the clip you watched</th><th>in the next one</th>"
                 "<th>why (the record's words)</th></tr></thead><tbody>"
                 + "".join(f"<tr><td>{p}-{q}</td><td>{clock(p, rate)}-{clock(q + 1, rate)}</td><td>{E(s)}</td><td>{E(o)}</td><td>{E(n)}</td><td>{E(w)}</td></tr>"
                           for p, q, s, o, n, w in ch) + "</tbody></table></div>"]
    body += held_and_set_aside(rec)
    body += ["<h2>Loads stopped before a render</h2>"]
    if stopped is None:
        body += ["<p class='mut'>This ledger was made before it read the stopped loads: none is shown, and that is not none.</p>"]
    elif not stopped:
        body += ["<p class='mut'>The ledger holds no note of a load stopped before a render.</p>"]
    for x in stopped or []:
        fr = x.get("frames")
        now = [f'{r["name"]} on {max(r["first"], fr[0])}-{min(r["last"], fr[1])}' for r in rec.T["rows"]
               if fr and r["subject"] == x.get("subject") and r["first"] <= fr[1] and r["last"] >= fr[0]]
        body.append(f'<div class="card"><h3>{E(x["load"])} <span>{E(str(x.get("subject")))} &middot; {f"{fr[0]}-{fr[1]}" if fr else "frames not in its job file"}'
                    f'</span></h3><p class="why">What of {E(str(x.get("subject")))} stands on those frames now: {E(", ".join(now)) if now else "nothing"}. '
                    f'The note, whole, a session\'s words ({E(x["note"])}, written {E(central(x["note_written"]))}):</p><pre>{E(x["note_says"])}</pre></div>')
    lanes = len(subjects) + 1
    height = TOP + LANE * lanes + 8
    svg = [f'<svg viewBox="0 0 {up_to + 1} {height}" preserveAspectRatio="none" style="width:{max(1200, (up_to + 1) * 5 // 4)}px;height:{height}px;display:block" '
           f'role="img" aria-label="timeline of frames 0 to {up_to}">']
    for x in range(0, up_to + 1, 100):
        svg.append(f'<line x1="{x}" y1="{TOP - 6}" x2="{x}" y2="{TOP + LANE * lanes}" class="tick"/><text x="{x + 3}" y="12" class="tl">{x}</text>'
                   f'<text x="{x + 3}" y="24" class="tl">{clock(x, rate)}</text>')
    for i, sub in enumerate(subjects):
        y = TOP + i * LANE
        for s in (s for s in stretches if s["sub"] == sub):
            w = s["b"] - s["a"] + 1
            if s["kind"] == "piece":
                svg.append(f'<a href="#p-{E(s["name"])}"><rect x="{s["a"]}" y="{y + 3}" width="{w}" height="{LANE - 6}" class="{tok(s["name"])}">'
                           f'<title>{E(s["name"])} {s["a"]}-{s["b"]} rendered {owners(rec.by_name[s["name"]]["written"])}</title></rect></a>')
            else:
                k = "left" if s["kind"] == "left" else "gap"
                said = "left as the original inside " + s["name"] if k == "left" else "no piece"
                svg.append(f'<a href="#g-{E(sub)}-{s["a"]}"><rect x="{s["a"]}" y="{y + 3}" width="{w}" height="{LANE - 6}" class="{k}">'
                           f'<title>{E(sub)} {s["a"]}-{s["b"]}: {E(said)}</title></rect></a>')
    y = TOP + len(subjects) * LANE
    for p, q in bare:
        svg.append(f'<rect x="{p}" y="{y + 3}" width="{q - p + 1}" height="{LANE - 6}" class="bare"><title>{p}-{q}: inside no piece</title></rect>')
    svg.append("</svg>")
    labels = (f'<div class="lanelabels" style="padding-top:{TOP}px">'
              + "".join(f'<div style="height:{LANE}px;line-height:{LANE}px">{E(x)}</div>' for x in [*subjects, "inside no piece"]) + "</div>")
    body += ["<h2>The timeline: what stands now</h2>",
             '<div class="lanes">' + "".join(f'<span><i class="k {t}"></i>a piece rendered {datetime.date.fromisoformat(d):%m/%d}</span>' for d, t in rec.day_token.items())
             + '<span><i class="k gap"></i>in the shot by the record (or the record unsure) and nothing is made: the original</span>'
               '<span><i class="k left"></i>inside a piece, left as the original by its load</span><span><i class="k bare"></i>bottom lane: frames inside no piece at all</span></div>',
             "<p class='mut'>The numbers are source frames with the video's time under them. Scroll sideways; click a block for its card or its row. "
             "An empty stretch in a lane means that person is not in the shot.</p>",
             f'<div class="tlwrap">{labels}<div class="scroll">{"".join(svg)}</div></div>',
             "<h2>Frames inside no piece, sorted</h2>",
             f"<p>{sum(q - p + 1 for p, q in bare)} of the first {up_to + 1} frames, in {len(bare)} runs: "
             + "; ".join(f"<b>{p}-{q}</b> ({clock(p, rate)})" if p != q else f"<b>{p}</b> ({clock(p, rate)})" for p, q in bare) + ".</p>"]
    tr = []
    for s in stretches:
        t = f"{clock(s['a'], rate)}-{clock(s['b'] + 1, rate)}"
        if s["kind"] == "piece":
            n = s["name"]
            tr.append(f'<tr><td>{s["a"]}-{s["b"]}</td><td>{t}</td><td>{E(s["sub"])}{" (the record is unsure they are in the shot)" if s["unsure"] else ""}</td>'
                      f'<td><a href="#p-{E(n)}">{E(n)}</a></td><td>{rec.day_badge(n)}</td>'
                      f'<td>{E(rec.last_click(rec.rows_of(n)))}</td><td>{E(s["why"])}</td></tr>')
        else:
            what = f'the original inside <a href="#p-{E(s["name"])}">{E(s["name"])}</a>' if s["kind"] == "left" else "the original: no piece"
            what += " (the record is unsure they are in the shot)" if s["unsure"] else ""
            tr.append(f'<tr class="{"leftr" if s["kind"] == "left" else "gapr"}" id="g-{E(s["sub"])}-{s["a"]}"><td>{s["a"]}-{s["b"]}</td><td>{t}</td>'
                      f'<td>{E(s["sub"])}</td><td>{what}</td><td></td><td></td><td>{E(s["why"])}</td></tr>')
    body += ["<h2>Every stretch, a person at a time, sorted by frame</h2>",
             "<div class='tw'><table><thead><tr><th>frames</th><th>time</th><th>who</th><th>what is there</th><th>rendered</th><th>your last click</th>"
             "<th>why it stands, or why nothing is there (the record's words)</th></tr></thead><tbody>" + "".join(tr) + "</tbody></table></div>",
             "<h2>Each piece and where it came from</h2>"]
    for r in sorted(rec.T["rows"], key=lambda r: (r["first"], r["subject"])):
        n = r["name"]
        g = rec.by_name[n]
        gr, form = g.get("graph") or {}, g.get("form") or {}
        made = (g.get("records") or [{}])[0]
        ref = form.get("reference") or {}
        folder = next(iter(glob.glob(os.path.join(day, f"*_{n}"))), None)
        text_file = os.path.join(folder, "text.txt") if folder else None
        text = open(text_file).read() if text_file and os.path.exists(text_file) else ""
        it = rec.item_of(n) or {}
        stacked = it.get("path") and str(in_repo(it["path"], rec.items_path).resolve())
        unchanged = runs(x for x in nothing.get(n, ()) if r["first"] <= x <= r["last"])
        facts = [("rendered", f"{owners(g['written'], '%Y-%m-%d %H:%M')} US Central ({g['written']}) by the session {g.get('session_folder')}"),
                 ("frames", f"{r['first']}-{r['last']} of the source ({clock(r['first'], rate)}-{clock(r['last'] + 1, rate)}); the load is {gr.get('frames')} frames from {gr.get('first')}"),
                 ("replaces", form.get("replaces", "")),
                 ("left as the original inside it", ", ".join(f"{p}-{q}" for p, q in unchanged) or "nothing"),
                 ("window sampled", f"{made.get('window', gr.get('window'))} frames; the context it carries {made.get('context', gr.get('context'))}"),
                 ("rendered on", "zoomed in on what is replaced" if gr.get("render_region") else "the whole frame"),
                 ("seed", str((made.get("load") or {}).get("seed", gr.get("seed")))),
                 ("the still", ", ".join(gr.get("stills") or [])),
                 ("the reference video", f"{os.path.basename(str((gr.get('files') or {}).get('motion_video')))}: " + ("; ".join(ref.get("steps") or []) or str(ref.get("says", "")))),
                 ("cut to the head / face points drawn", f"{(ref.get('has') or {}).get('cut_to_the_head')} / {(ref.get('has') or {}).get('face_points')}"),
                 ("the working copy it read", str(gr.get("of"))),
                 ("built at the commit", f"{made.get('commit')} ({central(made.get('built_at'))}); its record: {g.get('making')}"),
                 ("why it stands", stand_why(next((s for rd in rec.T["reading"] for s in rd.get("stands") or [] if s["render"] == n), {}))),
                 ("on the review page", rec.last_click(rec.rows_of(n)))]
        said = "".join(f"<li>{owners(v['at'], '%m/%d %H:%M')}, {E(how_said(v))}"
                       + (f", at frame {v['frame']}" if v.get("frame") is not None else "")
                       + (f", on the whole joined clip {E(v['through_the_joined_item'])}" if v.get("through_the_joined_item") else "")
                       + f": {E(v['words'])}</li>" for v in rec.owner_lines(rec.rows_of(n), every=True) if v.get("words"))
        links = []
        if folder:
            # each file the folder holds, never the folder: the review server serves files, not folders (2026-10-15: 22 of the
            # page's links did not load through it); the files bench/day_pieces.py writes, in the order a reader wants them
            held = [f for f in ("how_to_queue.txt", "inputs.txt", "text.txt", "graph_as_queued.json", "queue_this.json", "NO_QUEUED_GRAPH.txt")
                    if (Path(folder) / f).is_file()]
            held += sorted(x.name for x in Path(folder).iterdir() if x.is_file() and x.name not in held) if Path(folder).is_dir() else []
            where = os.path.relpath(folder, out_dir)
            links.append(("its workflow folder, " + E(where) + "/: " + ", ".join(f'<a href="{E(where)}/{E(f)}">{E(f)}</a>' for f in held))
                         if held else f"its workflow folder, {E(where)}/, holds no file")
        else:
            links.append("no folder of it in the day's folder (bench/day_pieces.py makes one for a piece that stands when it is run)")
        links.append(f'<a href="{E(rel(g["file"]))}">the render itself</a>')
        if stacked and os.path.exists(stacked):
            links.append(f'<a href="{E(rel(stacked))}">its stacked clip</a> (this load alone)')
        else:
            links.append("no stacked clip of it: " + (f"its item's file {it['path']} is not there" if it.get("path") else "no item on the list names a file of it"))
        body.append(f'<section class="card {tok(n)}" id="p-{E(n)}"><h3>{E(n)} <span>{E(r["subject"])} &middot; {r["first"]}-{r["last"]} &middot; rendered '
                    f'{owners(g["written"])}</span></h3><dl>' + "".join(f"<dt>{E(k)}</dt><dd>{E(str(v))}</dd>" for k, v in facts) + "</dl>"
                    + (f"<p><b>The verdicts record's words on it, whole</b>, each with where it came from:</p><ul>{said}</ul>" if said else "") + "<p>" + " &middot; ".join(links) + "</p>"
                    + (f"<details><summary>its text as queued</summary><pre>{E(text)}</pre></details>" if text else "") + "</section>")
    return page(f"Provenance: {stem}", PROVENANCE_CSS, "\n".join(body), 1200)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    for name, helped in (("to-review", "what the current table lays that the owner has not chosen on"), ("provenance", "what every stretch is made of and from which day")):
        s = sub.add_parser(name, help=helped)
        s.add_argument("--table", required=True, help="the delivery table's JSON (bench/delivery_table.py --json)")
        s.add_argument("--ledger", required=True, help="the render ledger's JSON the table was made from")
        s.add_argument("--verdicts", required=True, help="the verdicts record (data/<stem>/verdicts/verdicts.jsonl)")
        s.add_argument("--items", required=True, help="the review list (data/<stem>/review/items.json)")
        s.add_argument("--source", required=True, help="the source video: its frame rate gives the times, its stem the title")
        s.add_argument("--out", required=True, help="the page to write")
        s.add_argument("--share", required=True, help="the server's output folder, which the ledger's render paths are relative to")
        s.add_argument("--tables", action="append", default=[], help="a folder bench/shot_review_clips.py wrote joined clips' tables into: a "
                       "joined clip is shown as this table's only when its own table lays the same pieces on its shot, and a keep on it counts on each "
                       "piece it lays; repeatable")
        if name == "to-review":
            s.add_argument("--page-url", required=True, help="the review page's address up to the item id: its scheme, host, port and the # before an id")
            s.add_argument("--open-list", required=True, help="the Open List's link, where decisions waiting on the owner are")
            s.add_argument("--previous-table", help="the table the last page was made from: what changed since, and what each piece replaces")
            s.add_argument("--provenance", help="the provenance page's file name beside this one, linked from the top")
            s.add_argument("--today", help="the owner's day (YYYY-MM-DD, their clock) whose settled pieces are listed under Done; today when left out")
            s.add_argument("--start-here", metavar="FILE", help="the lead's list, in order, drawn first: a JSON list of {\"item\": an item the "
                           "page shows, a shot FIRST-LAST or a clip's path, \"why\": one line of what deciding it changes}, or of {\"link\": "
                           "an http(s) URL where a question's buttons are, \"label\", \"why\", \"clips\": [paths, shown beneath it]}, each with optional \"reader_notes\"")
        else:
            s.add_argument("--record", help="the assembler's record of the clip (.mp4.check.json); without it the ledger says what a piece leaves")
            s.add_argument("--day-folder", required=True, help="the day's folder: bench/day_pieces.py's piece folders are read from it")
            s.add_argument("--up-to", required=True, type=int, help="the last frame shown")
            s.add_argument("--watched-table", help="the table of the clip the owner watched: what changes from it, and why")
    a = p.parse_args(argv)
    doc = to_review(a) if a.mode == "to-review" else provenance(a)
    out = Path(a.out)
    was = out.exists()
    out.write_text(doc)
    print(f"{'replaced' if was else 'wrote'} {out}: {len(doc)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
