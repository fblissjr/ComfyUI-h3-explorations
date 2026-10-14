#!/usr/bin/env python3
"""A shot's joined review clip: the shot with every piece that stands on it laid, stacked for the eye, one review item a shot, each piece with the day it was rendered.

    <python> bench/shot_review_clips.py build --ledger LEDGER.json --shots SHOTS.json --plan PLAN.json --source SOURCE.mp4 \\
        --clips <share>/Video/<stem>/day_<date>/shots --tables data/<date>_<stem>_delivery_tables --stack-work SCRATCH \\
        [--stamp T] [--also-capture FOLDER]... [--prefer ...]... [--hold ...]... [--dry] FIRST-LAST [FIRST-LAST ...]
    <python> bench/shot_review_clips.py items --items data/<stem>/review/items.json --clips FOLDER --tables FOLDER --stem STEM \\
        --stamp T --by NAME [--note FIRST-LAST=words]... [--dry] FIRST-LAST [FIRST-LAST ...]
    <python> bench/shot_review_clips.py by-day --table T.json --ledger LEDGER.json --day YYYY-MM-DD --out OUT.md

**NOT MADE** (2026-10-14): four blind readers of the per-shot clip tool and two of the pages each found real faults; the
lead ruled that neither tool takes a new feature until it passes one more read. What the last read found and is not
mended is in docs/checks.md's row for its check ("NOT HELD") and in the lead's notes. Read its output against the
table and the ledger it was made from.

**What it buys.** The owner's rule of 2026-10-13: a shot with two of our subjects in it is judged with every piece
laid, never as one load's clip with the other person the original, and each piece says the day it was rendered.
That evening it was a session script (`internal/claude/2026-10-13_mrohmy_scripts/per_shot_clips.py`) and the table
by day another (`internal/claude/2026-10-13_mrtower_scripts/pieces_by_day.py`). This is both, from files.

**`build`**, a span at a time, each span a whole shot or run of shots of the shots record: the delivery table on
that span alone (`bench/delivery_table.py --span`, which says which render stands and why), then the two commands
that table writes into its own JSON (`assembly`): the assembler and the stack with `--first`. The stack is then
read back by the stack tool's own reading (`stack_delivery.read_back`, what its `check` prints), kept as data
(`check_stack_<name>.json`) and printed whole. Nothing is parsed out of a print. Every file is named
`<source stem>_<first>_<last>_joined_<T>`: the clip, its `_STACKED.mp4` and the assembler's record in `--clips`;
the table (`t_<name>.txt`, `.json`), its reading, the stack's reading and each step's output (`<step>_<name>.out`)
in `--tables`. A name any of those already holds, or a span given twice, is refused before anything runs: a kept
output is not replaced. The first failing step stops the run with its last line. Every path is made absolute
first: the steps run from the repo's root. `--dry` writes the table and prints the two commands, and runs none;
its stamp is then taken, and a real build takes another.

THE ASSEMBLER ALSO WRITES INTO THE CAPTURES IT READS: its record of the clip, `delivery__<name>.json`, into each
capture folder, and `runs/<run>/changed__<name>.csv`, which the capture's own tool lists beside the real
deliveries' (`bench/capture_masked_run.py`), and it writes them with no check of its own. So a build whose name a
capture already holds a record of is refused once its table is made (the captures are the table's), and a throwaway
build leaves its records there for a person to remove.

A one-shot table says, for a subject the shots record has and no row of that table lays, that they are the
original there. The assembler takes that line when the shots record lists the subject and keeps it in its record
as the table's word alone (`shots.said_of_subjects_no_row_has`); the evening's script cut those lines out first.

**`items`** puts one item a span at the top of the review list, the stacked clip linked into the list's `media`
folder. It is handed the build by its stamp and reads that build's named files. Its words are made from the span's
table JSON and the assembler's record: every piece that stands, its frames, the day and hour it was rendered on the
owner's clock (`piece_gaps.OWNERS_CLOCK`), the owner's last choice on it or that they made none, the frames inside
it that its load left as the original or that what it leaves is NOT KNOWN; a piece that stands by the lead's
`--prefer`, said as the lead's choice with its reason whole, and a keep of the owner's it set aside; a render the
lead held back (`--hold`) with its reason whole; a subject the shots record is unsure of (with any piece of theirs
that stands), one it says is not in the shot (the table's `not_in_the_shot`), and one with nothing standing. A span
of several shots begins each sentence with its shot, in frame order. `--note` adds the words of the session that puts
the item on the list (`--by`) to a span's item, whole, after the pieces, said as that session's; one a span. The
owner's own words go into the verdicts record, never through `--note`. A media link of the clip's name that points elsewhere is refused. The
buttons are the table's own (`delivery_table.RENDER_ITEM_BUTTONS`). Refused: a span not built, an id already on the
list, a clip whose assembler's record fails, and a stack whose reading has a hard fault (a frame count that is not
the span's, sound that is not the clip's, or no sound at all). How well each stack row fits its frames is printed, not judged. A
copy of the list, named with the time of the write, goes into `--tables` before the one write.

**`by-day`** writes the table of what stands by the day it was rendered: a row a piece in frame order with why it
stands in the table's own words (the lead's `--prefer` said as the lead's choice), then the renders of `--day` in
the ledger that this table does not lay, with why. The ledger must be the one the table says it was made from.

**What it cannot see.** Whether a piece is good, or whether the owner would pick it: the table's rule says why it
stands and no more. A `source` line for a subject no row lays is checked against no picture. The day is the
render file's `written` time in the ledger, which is when the file was written, not when it was queued.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delivery_table  # noqa: E402
import stack_delivery  # noqa: E402
from piece_gaps import OWNERS_CLOCK  # noqa: E402


def span_of(text: str) -> tuple[int, int]:
    try:
        a, b = (int(v) for v in text.split("-"))
    except ValueError:
        raise SystemExit(f"{text!r} is not FIRST-LAST") from None
    if a > b:
        raise SystemExit(f"{text}: the first frame is after the last")
    return a, b


def clip_name(stem: str, a: int, b: int, stamp: str) -> str:
    return f"{stem}_{a:04d}_{b:04d}_joined_{stamp}"


def owners_clock(utc: str) -> datetime.datetime:
    """A stamp of the records on the owner's clock. Every stamp is UTC (a stamp with no zone is read as UTC, never as
    the machine's own clock)."""
    t = datetime.datetime.fromisoformat(utc.replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)).astimezone(OWNERS_CLOCK)


def files_of(name: str, clips: Path, tables: Path) -> dict[str, Path]:
    """Every file `build` writes for one span, by its role."""
    return {"clip": clips / f"{name}.mp4", "stacked": clips / f"{name}_STACKED.mp4", "record": clips / f"{name}.mp4.check.json",
            "table": tables / f"t_{name}.txt", "json": tables / f"t_{name}.json", "reading": tables / f"reading_{name}.md",
            "stack_reading": tables / f"check_stack_{name}.json",
            **{f"{step}_out": tables / f"{step}_{name}.out" for step in ("table", "build", "stack", "check_stack")}}


def table_command(a: argparse.Namespace, span: tuple[int, int], f: dict[str, Path], work: Path) -> list[str]:
    cmd = [sys.executable, str(HERE / "delivery_table.py"), "--ledger", a.ledger, "--shots", a.shots, "--plan", a.plan, "--span", f"{span[0]}-{span[1]}",
           "--assemble-args", shlex.join(["--source", a.source, "--size", "source", "--out", str(f["clip"])]), "--stack-work", str(work),
           "--table", str(f["table"]), "--reading", str(f["reading"]), "--json", str(f["json"])]
    for x in a.also_capture:
        cmd += ["--also-capture", x]
    for x in a.prefer:
        cmd += ["--prefer", x]
    for x in a.hold:
        cmd += ["--hold", x]
    return cmd


def last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1][:300] if lines else "(no output)"


def stack_faults(got: dict, span: tuple[int, int]) -> list[str]:
    """The hard faults of a stack's reading: a frame count that is not the span's, sound that is not the clip's."""
    want = span[1] - span[0] + 1
    out = [f"the stacked file has {got['frames']} frames and the clip {got['clip_frames']}, where the span is {want}"] \
        if not got["frames"] == got["clip_frames"] == want else []
    sound = got["sound_packets"]
    if not sound["stacked"] and not sound["clip"]:
        return out + ["neither the stacked file nor the clip carries sound"]
    return out + ([] if sound["the_same"] else ["the stacked file's sound is NOT the clip's"])


def names_the_owner(by: str) -> bool:
    """Whether `by` holds the repo's git user name or a word of it (three letters or more), read when asked and never written
    down (an eighth reader, 2026-10-14: the owner's own name passed the guard)."""
    name = subprocess.run(["git", "-C", str(HERE.parent), "config", "user.name"], capture_output=True, text=True).stdout.strip().lower()
    words = [w for w in re.split(r"\W+", name) if len(w) >= 3]
    return bool(name) and (name in by.lower() or any(re.search(rf"\b{re.escape(w)}\b", by.lower()) for w in words))


def joined_ledger(path: str) -> None:
    """Refuse a ledger made without --joined-tables: it reads none of the owner's lines on a joined clip, so an item or by-day
    would say "not judged" of pieces the owner judged on one (a seventh reader; an eighth found items and by-day unguarded)."""
    argv = (json.loads(Path(path).read_text()).get("command") or {}).get("argv") or []
    if "--joined-tables" not in argv:
        raise SystemExit(f"refused, nothing run: the ledger {path} was made without --joined-tables, so it reads none of the owner's lines on a "
                         "joined clip; make it again with the folder of the joined clips' tables (bench/render_ledger.py --joined-tables)")


def build(a: argparse.Namespace) -> int:
    spans = [span_of(s) for s in a.spans]
    if len(set(spans)) != len(spans):
        raise SystemExit("refused, nothing run: a span is given twice")
    for key in ("ledger", "shots", "plan", "source", "clips", "tables", "stack_work"):
        setattr(a, key, str(Path(getattr(a, key)).resolve()))
    a.also_capture = [str(Path(x).resolve()) for x in a.also_capture]
    clips, tables = Path(a.clips), Path(a.tables)
    stem = Path(a.source).stem
    named = {span: clip_name(stem, *span, a.stamp) for span in spans}
    taken = [str(p) for span in spans for p in files_of(named[span], clips, tables).values() if p.exists()]
    if taken:
        raise SystemExit("refused, nothing run: a kept output is not replaced; these are there already:\n  " + "\n  ".join(taken))
    # a ledger made without the joined items' tables reads no click on a joined clip, and the item would say "not judged" of pieces the
    # owner judged on one (a seventh reader, 2026-10-14: the 1854Z and 1929Z ledgers)
    joined_ledger(a.ledger)
    clips.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for span in spans:
        name, t0 = named[span], time.time()
        f = files_of(name, clips, tables)
        work = Path(a.stack_work) / f"stack_work_{name}"
        r = subprocess.run(table_command(a, span, f, work), capture_output=True, text=True, cwd=HERE.parent)
        f["table_out"].write_text(r.stdout + r.stderr)
        if r.returncode:
            print(f"{span[0]}-{span[1]}: THE TABLE exit {r.returncode}: {last_line(r.stdout + r.stderr)} (all of it: {f['table_out']})")
            return 1
        made = json.loads(f["json"].read_text())["assembly"]
        # the assembler writes its record into every capture it reads, with no check of its own: an earlier build of this name is not written over
        earlier = [str(p) for c in made["captures"] for base in (HERE.parent / c, Path(c))
                   for p in [base / f"delivery__{name}.json", *base.glob(f"runs/*/changed__{name}.csv")] if p.is_file()]
        if earlier:
            print(f"{span[0]}-{span[1]}: REFUSED after the table: a capture holds an earlier build's record of this name ({', '.join(sorted(set(earlier)))}); "
                  "take another --stamp")
            return 1
        steps = [("build", made["command"]), ("stack", made["stack"]["build"])]
        if a.dry:
            print(f"{span[0]}-{span[1]}: {name}: the table is written ({f['table']}); dry, so neither was run, nor the stack's reading:")
            for step, line in steps:
                print(f"  {step}: {line}")
            continue
        for step, line in steps:
            s = subprocess.run(shlex.split(line), capture_output=True, text=True, cwd=HERE.parent)
            f[f"{step}_out"].write_text(s.stdout + s.stderr)
            if s.returncode:
                print(f"{span[0]}-{span[1]}: {step} exit {s.returncode}: {last_line(s.stdout + s.stderr)} (all of it: {f[f'{step}_out']})")
                return 1
        got = stack_delivery.read_back(a.source, str(f["clip"]), str(f["stacked"]), span[0])
        f["stack_reading"].write_text(json.dumps(got, indent=1) + "\n")
        lines = stack_delivery.check_lines(got, span[0])
        f["check_stack_out"].write_text("\n".join(lines) + "\n")
        rec = json.loads(f["record"].read_text())
        print(f"{span[0]}-{span[1]}: {name}: wall {time.time() - t0:.0f} s; the assembler: {rec['verdict_line']}; failures {len(rec['failures'])}; "
              f"frames {rec['frames_expected']}; the stack read back:")
        for line in lines:
            print("    " + line)
        faults = stack_faults(got, span)
        if faults:
            print(f"{span[0]}-{span[1]}: THE STACK HAS A HARD FAULT: {'; '.join(faults)}. No item is made for it.")
            return 1
    return 0


#: How many renders passed over are named in an item when nothing of a subject stands, and how much of each one's
#: reason. A display cut, reasoned: the item is read on the owner's page; the table's reading beside the clip has every
#: one whole, and the item says how many there are. A HOLD of the lead's is never cut: it is a session's word.
PASSED_OVER_NAMED, REASON_CHARS = 3, 110


def cut(words: str) -> str:
    return words if len(words) <= REASON_CHARS else words[:REASON_CHARS] + "..."


#: Words the owner sent up to this many seconds before a click go WITH that click (the lead's ruling of 2026-10-14, on an eighth
#: reader: the review page saves the words a few seconds before the click). A starting value from three cases: 0294d, 1222b and
#: 0415d, 2 to 3 s before.
WITH_THE_CLICK_S = 10


def words_after(choice: dict | None, entry: dict) -> list[str]:
    """The owner's words on a render after their last choice (the table's `the_owners_words`), each with its day and hour on
    their clock; with no choice, all their words on it. Their words rule over the button (the owner's rule of 2026-10-12): a
    click they wrote after is said with them, and words with no click are said (a sixth reader, 2026-10-14: third_fp_1419b's
    "i dont like beard", with no click, read as "not judged by you yet")."""
    def with_click(w: dict) -> bool:          # the same second as the click, or up to WITH_THE_CLICK_S before it, goes with it
        return (owners_clock(choice["at"]) - owners_clock(w["at"])).total_seconds() <= WITH_THE_CLICK_S
    return [f"\"{' / '.join(w['said'].splitlines())}\" ({owners_clock(w['at']):%m/%d %H:%M} Central" + (f", at frame {w['frame']}" if w.get("frame") is not None else "") + ")"
            for w in entry.get("the_owners_words") or [] if choice is None or with_click(w)]


def passed_said(e: dict, keep: tuple) -> str:
    """The renders passed over on a subject's shot, not held and with no click or words of the owner's (each of those has its
    own sentence, its reason whole): how many, and the newest few with their reasons cut."""
    passed = [p for p in e["passed_over"] if not p.get("held") and not p.get("the_owners_last_choice") and not p.get("the_owners_words")]
    return (f" ({len(passed)} other render(s) passed over, the newest {min(len(passed), PASSED_OVER_NAMED)}: "
            + "; ".join(p["render"] + ", " + cut(p["why"]) for p in passed[:PASSED_OVER_NAMED]) + ")") if passed else ""


def item_words(table: dict, keep: tuple = delivery_table.KEEP) -> tuple[int, list[str]]:
    """(pieces, one sentence a piece or a subject) for a span's item, from the span's table JSON alone: shot by shot,
    inside a shot subject by subject in the table's order, and a subject's sentences by the frame each is about
    (what is unsure of them first, what was passed over or held last). Where the span holds more than one shot,
    every sentence begins with its shot. Why each piece stands is the table's
    rule: a piece the lead put first (`--prefer`) says so, with the lead's reason whole, and so does a keep of the
    owner's that it set aside; a render the lead held back (`--hold`) is named with its reason whole."""
    entries = [("reading", e) for e in table["reading"]] + [("absent", n) for n in table.get("not_in_the_shot") or []]
    shots = {(e["first"], e["last"]) for _, e in entries}
    said, pieces, seen = [], 0, set()
    rank = {}
    for _, e in entries:
        rank.setdefault(e["subject"], len(rank))

    def say(e: dict, words: str, at: float | None = None) -> None:
        key = (e["first"], rank[e["subject"]], e["first"] if at is None else at, len(said))
        said.append((key, (f"on {e['first']}-{e['last']}, " if len(shots) > 1 else "") + words))
    for kind, e in entries:
        who = e["subject"]
        if kind == "absent":
            say(e, f"{who}: not in this shot by the shots record ({e['why']})")
            continue
        if e["present"] == "unsure":
            say(e, f"{who}: the shots record is unsure they are in this shot (yours to answer on the who page)"
                + ("" if e["stands"] else "; the original throughout"
                   + (f"; present, nothing to make under this plan, by the table: {'; '.join(e['nothing_to_make'])}" if e.get("nothing_to_make") else "")
                   + passed_said(e, keep)), e["first"] - 0.5)
        for s in e["stands"]:
            left = s.get("leaves_as_the_original")
            own = {x for x, y in s["frames"] for x in range(x, y + 1)}
            nothing_shows = bool(left) and own <= {x for a, b in left for x in range(a, b + 1)}
            if s["render"] not in seen and not nothing_shows:      # a piece that leaves the original on all its frames lays nothing
                pieces += 1
                seen.add(s["render"])
            c = s.get("the_owners_last_choice")
            day = owners_clock(s["written"])
            after = words_after(c, s)
            choice = (f"your keep of {owners_clock(c['at']):%m/%d}" if c and c["said"] in keep
                      else f"your last choice on it: \"{c['said']}\"" if c else "not judged by you yet")
            choice += (f", AND YOUR WORDS AFTER IT: {'; '.join(after)}" if c else f", BUT YOUR WORDS ON IT: {'; '.join(after)}") if after else ""
            led = s.get("the_leads_prefer")
            say(e, f"{who}: {s['render']} on {', '.join(f'{x}-{y}' for x, y in s['frames'])}, rendered {day:%m/%d} ({day:%H:%M} Central); {choice}"
                + (f"; IT STANDS BY THE LEAD'S CHOICE (--prefer), a session's word and not yours: {led}" if led else "")
                + ("; what it leaves as the original inside it is NOT KNOWN (its ledger entry does not say)" if left is None
                   else f"; inside it the piece leaves the original on {', '.join(f'{x}-{y}' for x, y in left)} (its capture had no part of them there)"
                   + ("; that is all its frames, so nothing of it shows and it is not counted as laid" if nothing_shows else "")
                   if left else ""), min(x for x, _ in s["frames"]))
        for p in e["passed_over"]:
            c = p.get("the_owners_last_choice")
            if p.get("held"):
                after = words_after(c, p)
                say(e, f"{who}: {p['render']} is HELD BACK by the lead (--hold), a session's word and not yours: {p['held']}"
                    + (f"; your last choice on it: \"{c['said']}\" ({owners_clock(c['at']):%m/%d})" if c else "")
                    + (f", and your words: {'; '.join(after)}" if after else ""), e["last"] + 1)
            elif c and c["said"] in keep:
                after = words_after(c, p)
                say(e, f"{who}: YOUR KEEP {p['render']} (of {owners_clock(c['at']):%m/%d}" + (f", and your words after it: {'; '.join(after)}" if after else "")
                    + f") is not laid here: {p['why']}", e["last"] + 1)
            elif c or p.get("the_owners_words"):
                # the owner's other clicks and words on a render passed over are said with it, its reason whole (a seventh reader, 2026-10-14:
                # 48 such renders on the 1929Z table were only counted, and the cut reason chopped the owner's quoted click)
                after = words_after(c, p)
                say(e, f"{who}: {p['render']} is not laid here ({p['why']}); " + (f"your last choice on it: \"{c['said']}\" ({owners_clock(c['at']):%m/%d})" if c
                                                                                  else "no click of yours on it")
                    + (f", and your words: {'; '.join(after)}" if after else ""), e["last"] + 1)
        if not e["stands"] and e["present"] != "unsure":
            nothing = e.get("nothing_to_make")
            say(e, f"{who}: the original on the whole shot: " + (f"present, nothing to make under this plan, by the table: {'; '.join(nothing)}" if nothing
                                                                 else "no piece of theirs stands here") + passed_said(e, keep))
        elif e["stands"]:
            for g in e.get("left_as_the_source") or []:
                say(e, f"{who}: the original on {g[0]}-{g[1]}, where no piece of theirs is laid", g[0])
    return pieces, [w for _, w in sorted(said, key=lambda x: x[0])]


def make_item(name: str, span: tuple[int, int], table: dict, record: dict, link: str, stamp: str, by: str, note: str | None) -> dict:
    a, b = span
    num, den = (int(v) for v in record["video"]["rate"].split("/"))
    fps = num / den
    pieces, said = item_words(table)
    n_shots = len({(e["first"], e["last"]) for e in list(table["reading"]) + list(table.get("not_in_the_shot") or [])})
    what = f"the shot {a}-{b}" if n_shots <= 1 else f"the {n_shots} shots of {a}-{b}"
    return {"id": f"{name}_stacked",
            "title": f"{name}_stacked: {what} with every piece that stands laid, stacked: the original on top, what each render was given "
                     "in the middle, the joined clip below, the frame number on every frame",
            "file": Path(link).name, "path": link, "first": str(a), "frames": str(b - a + 1), "fps": f"{fps:.3f}",
            "seconds": f"about {a / fps:.1f} s to {(b + 1) / fps:.1f} s of the video",
            "state": "open", "state_words": "not watched by anyone on playback; yours to watch. A joined clip: every piece that stands on these frames is laid, "
                                            "so nobody in it is the original where a piece stands.",
            "what": f"{what[0].upper() + what[1:]}, {b - a + 1} frames, from the build of {stamp}, put on this list by {by}; nothing was rendered for it. "
                    + (f"{pieces} piece(s) laid, each with the day it was rendered: " if pieces else "No piece is laid" + (": " if said else "."))
                    + (". ".join(said) + "." if said else "") + (f" Words of {by}, who put this item on the list: {note}" if note else ""),
            "think": "", "look": "", "options": list(delivery_table.RENDER_ITEM_BUTTONS),
            "prompt": "No single text: this clip is laid from the pieces named above, each rendered with its own text. Each piece's own item on this page shows its text.",
            "prompt_note": f"The table this clip was built from: t_{name}.txt beside the day's tables, with its reading.", "fixed": "Nothing was fixed in this clip."}


def items(a: argparse.Namespace) -> int:
    if re.search(r"owner|fblissjr", a.by, re.I) or names_the_owner(a.by):   # the note is said as the --by session's words: never the owner's
        raise SystemExit(f"--by {a.by!r}: the session that puts the items on the list; its note is said as its words, never the owner's")
    spans = [span_of(s) for s in a.spans]
    if len(set(spans)) != len(spans):
        raise SystemExit("a span is given twice")
    notes = {}
    for x in a.note:
        span, sep, words = x.partition("=")
        if not sep or not words.strip() or span_of(span) not in spans:
            raise SystemExit(f"--note {x!r}: not FIRST-LAST=words for one of the spans given")
        if span_of(span) in notes:
            raise SystemExit(f"--note: two for {span}: one a span, with all its words")
        notes[span_of(span)] = words.strip()
    path, clips, tables = Path(a.items).resolve(), Path(a.clips).resolve(), Path(a.tables).resolve()
    raw = path.read_text()
    listed = json.loads(raw)
    ids = {x["id"] for x in listed["items"]}
    media = path.parent / "media"
    new, new_tables = [], {}
    for span in spans:
        name = clip_name(a.stem, *span, a.stamp)
        f = files_of(name, clips, tables)
        missing = [str(f[k]) for k in ("stacked", "record", "json", "stack_reading") if not f[k].is_file()]
        if missing:
            raise SystemExit(f"{span[0]}-{span[1]}: not built: {', '.join(missing)} not there")
        if f"{name}_stacked" in ids:
            raise SystemExit(f"{span[0]}-{span[1]}: the list already has an item {name}_stacked")
        made_table = json.loads(f["json"].read_text())
        named_ledger = re.search(r"from the ledger `([^`]+)`", str(made_table.get("made") or ""))
        if named_ledger:
            ledger_path = Path(named_ledger.group(1)) if Path(named_ledger.group(1)).is_absolute() else HERE.parent / named_ledger.group(1)
            if ledger_path.is_file():
                joined_ledger(str(ledger_path))
        shots = [(e["first"], e["last"]) for e in made_table["reading"] + list(made_table.get("not_in_the_shot") or [])]
        if not shots or (min(x for x, _ in shots), max(y for _, y in shots)) != span:
            raise SystemExit(f"{span[0]}-{span[1]}: its table {f['json'].name} is of "
                             + (f"{min(x for x, _ in shots)}-{max(y for _, y in shots)}" if shots else "no shot") + ", not of this span")
        if any("step" not in s for e in made_table["reading"] for s in e["stands"]):
            raise SystemExit(f"{span[0]}-{span[1]}: its table was made before the table said why each piece stands in fields of its own "
                             "(bench/delivery_table.py, 2026-10-14): its words would hide the lead's --prefer. Build it again with another stamp")
        record = json.loads(f["record"].read_text())
        if record["verdict"] != "passes":
            raise SystemExit(f"{span[0]}-{span[1]}: the assembler's record says {record['verdict_line']!r}: no item for a clip that fails")
        got = json.loads(f["stack_reading"].read_text())
        faults = stack_faults(got, span)
        if faults:
            raise SystemExit(f"{span[0]}-{span[1]}: the stack's reading has a hard fault: {'; '.join(faults)}: no item for it")
        print(f"{span[0]}-{span[1]}: the stack read back: " + " | ".join(stack_delivery.check_lines(got, span[0])[2:]))
        link = os.path.relpath(media / f["stacked"].name, HERE.parent)
        new.append((f["stacked"], media / f["stacked"].name, make_item(name, span, made_table, record, link, a.stamp, a.by, notes.get(span))))
        new_tables[f"{name}_stacked"] = made_table
    for _, _, item in new:
        print(f"{item['id']}: {item['what']}")
    # AN EARLIER JOINED ITEM OF THE SAME SPAN is retired by the new one (the lead, 2026-10-14: three open items of 1222-1305 stood on
    # the page at once, each a build of the same frames). Only an open one: an item in any other state is somebody's, left as it is.
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    retire = []
    for _, _, item in new:
        prefix = f"{a.stem}_{int(item['first']):04d}_{int(item['first']) + int(item['frames']) - 1:04d}_joined_"
        a0, b0 = int(item["first"]), int(item["first"]) + int(item["frames"]) - 1
        for old in listed["items"]:
            if old["id"].startswith(prefix) and old["id"].endswith("_stacked"):
                stamp = old["id"][len(prefix):-len("_stacked")]
                if stamp > a.stamp:                 # an older build never retires a newer one (a sixth reader, 2026-10-14)
                    raise SystemExit(f"{a0}-{b0}: the list holds a LATER build of the same frames, {old['id']} (its stamp {stamp}, this one's {a.stamp}): "
                                     "nothing written")
                if old.get("state") != "open":
                    print(f"leaves {old['id']} as it is: its state is {old.get('state')!r}, not open")
                    continue
                # retired only into a build that lays the same pieces on the same frames, which inherits its lines on the owner's
                # page (the lead's ruling of 2026-10-14, on a seventh reader: a kept item was retired into one of other pieces)
                old_json = files_of(old["id"].removesuffix("_stacked"), clips, tables)["json"]
                lays = lambda T: sorted((r["name"], r["subject"], max(r["first"], a0), min(r["last"], b0)) for r in T.get("rows") or []  # noqa: E731
                                        if r["first"] <= b0 and r["last"] >= a0)
                if not old_json.is_file():
                    print(f"leaves {old['id']} open: its table {old_json.name} is not there to say what it lays, so its lines are not known to pass to the new one")
                elif lays(json.loads(old_json.read_text())) != lays(new_tables[item["id"]]):
                    print(f"leaves {old['id']} open: it lays other pieces than {item['id']}, so the owner's lines on it stay its own")
                else:
                    retire.append((old, item["id"]))
                    print(f"retires {old['id']}: superseded by {item['id']}, a later build of the same pieces on the same frames, which takes its lines")
            elif ("_joined_" in old["id"] and old["id"].endswith("_stacked") and old.get("state") == "open" and old.get("first") and old.get("frames")
                  and int(old["first"]) <= b0 and int(old["first"]) + int(old["frames"]) - 1 >= a0):
                print(f"an open item of other frames overlaps it and stays open: {old['id']} ({old['first']}-{int(old['first']) + int(old['frames']) - 1}); "
                      "on the frames they share it may lay other pieces, and the page shows both")
    if a.dry:
        print(f"dry: nothing written; {len(new)} item(s) would go to the top of {path}" + (f", and {len(retire)} retired" if retire else ""))
        return 0
    for target, link, _ in new:
        if os.path.lexists(link) and Path(os.path.realpath(link)) != target.resolve():
            raise SystemExit(f"{link} is a link to {os.path.realpath(link)}, not to this build's {target}: nothing written")
    base = f"items.before_{a.by}_{a.stamp}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}"
    copy, k = tables / f"{base}.json", 1
    while copy.exists():                          # two writes in one second: each keeps its own copy
        k += 1
        copy = tables / f"{base}_{k}.json"
    copy.write_text(raw)                          # the list as it was read: what this run changes is against that
    for target, link, _ in new:
        if not os.path.lexists(link):
            os.symlink(target, link)
    for old, by_id in retire:
        old.update(state="superseded", superseded_by=by_id,
                   state_words=f"superseded, kept for the record: an earlier build of the same frames; {by_id} is the one to watch "
                               f"({a.by}, {now}, by bench/shot_review_clips.py items)")
    listed["items"][0:0] = [x for _, _, x in new]
    listed["updated"], listed["by"] = now, a.by
    out = json.dumps(listed, indent=1) + ("\n" if raw.endswith("\n") else "")
    tmp = path.with_name(path.name + f".tmp_{a.by}")
    tmp.write_text(out)
    if path.read_text() != raw:                   # another session wrote the list since it was read: its change is not written over
        tmp.unlink()
        raise SystemExit(f"{path} changed since this run read it: nothing written to it (the copy {copy} holds the list as read); run it again")
    os.replace(tmp, path)
    back = json.loads(path.read_text())
    print(f"added: the list's first items are {[x['id'] for x in back['items'][:len(new)]]}; {len(back['items'])} items; the copy before: {copy}")
    return 0


def by_day_text(table: dict, ledger: dict, day: str, table_name: str, ledger_name: str) -> str:
    """What stands in a table, a row a piece by the day it was rendered, with the table's own words on why; and the
    renders of `day` in the ledger that this table does not lay."""
    pieces: dict[tuple[str, str], dict] = {}
    for r in table["reading"]:
        # an owner's keep of a piece that does not stand on this shot: said beside what stands over it, never hidden under the lead's word
        set_aside = [f"{p['render']} ({p['the_owners_last_choice']['said']!r}, {owners_clock(p['the_owners_last_choice']['at']):%m-%d %H:%M} Central"
                     + "".join(f"; your words after it: {w}" for w in words_after(p["the_owners_last_choice"], p)) + ")"
                     for p in r.get("passed_over") or [] if (p.get("the_owners_last_choice") or {}).get("said") in delivery_table.KEEP]
        for s in r.get("stands") or []:
            one = pieces.setdefault((s["render"], r["subject"]), {"frames": [], "written": s["written"], "why": [], "led": None, "owner": None, "set_aside": [],
                                                                 "left": set()})
            one["frames"] += s["frames"]
            one["left"] |= {x for a, b in s.get("leaves_as_the_original") or [] for x in range(a, b + 1)}
            if s["why"] not in one["why"]:
                one["why"].append(s["why"])
            one["led"] = one["led"] or s.get("the_leads_prefer")            # the lead's word on any shot of it is said
            c = s.get("the_owners_last_choice")
            # the owner's NEWEST click on any shot of it, with their words after it; with no click on any shot, their words
            if c and (not one["owner"] or str(c["at"]) > str(one["owner"]["at"])):
                one["owner"], one["owner_after"] = c, words_after(c, s)
            elif not one["owner"]:
                one["owner_after"] = (one.get("owner_after") or []) + [w for w in words_after(None, s) if w not in (one.get("owner_after") or [])]
            one["set_aside"] += [x for x in set_aside if x not in one["set_aside"]]

    def why_of(p: dict) -> str:
        others = [w for w in p["why"] if not w.startswith("step 0")]       # the table's reasons on the shots the lead's word did not decide
        words = (f"THE LEAD'S CHOICE (--prefer), a session's word: {p['led']}" + (f"; on its other shots, the table's: {'; '.join(others)}" if others else "")
                 if p["led"] else "; ".join(p["why"]))
        if p["set_aside"]:
            words += f". THE OWNER'S OTHER KEEP(S) on these frames, not laid by this table: {', '.join(p['set_aside'])}"
        return words

    def owner_of(p: dict) -> str:
        o = p["owner"]
        if not o:
            return "no click" + "".join(f"; your words on it: {w}" for w in p.get("owner_after") or []) if p.get("owner_after") else "none"
        return f"{o['said']!r}, {owners_clock(o['at']):%m-%d %H:%M}" + "".join(f"; your words after it: {w}" for w in p.get("owner_after") or [])
    def shows_nothing(p: dict) -> bool:      # a piece that leaves the original on all its frames lays nothing (as the item says; a seventh reader)
        return bool(p["left"]) and {x for a, b in p["frames"] for x in range(a, b + 1)} <= p["left"]
    cell = lambda x: str(x).replace("|", "\\|")                                          # noqa: E731
    rows = sorted(((min(a for a, _ in p["frames"]), ",".join(f"{x}-{y}" for x, y in sorted(p["frames"])), sub,
                    name + (" (NOTHING OF IT SHOWS: it leaves the original on all its frames)" if shows_nothing(p) else ""), owners_clock(p["written"]),
                    cell(owner_of(p)), cell(why_of(p)))
                   for (name, sub), p in pieces.items()))
    blank = sum(1 for p in pieces.values() if shows_nothing(p))
    stands = {name for name, _ in pieces}
    shots = list(table["reading"]) + list(table.get("not_in_the_shot") or [])
    span = [min(e["first"] for e in shots), max(e["last"] for e in shots)] if shots else None
    lines = [f"# What stands in {table_name}, by the day each piece was rendered", "",
             f"Read from the table's own reading and the ledger `{ledger_name}` by `bench/shot_review_clips.py by-day`; the day is the render file's "
             f"`written` time on the owner's clock (US Central). `{day}` is the day asked about. Why each piece stands is the table's own words.", "",
             "| frames | subject | piece | rendered (Central) | the owner's last choice on it (Central) | why it stands |", "|---|---|---|---|---|---|"]
    lines += [f"| {fr} | {sub} | {name} | {t:%m-%d %H:%M} | {o} | {w} |" for _, fr, sub, name, t, o, w in rows]
    on_day = sum(1 for x in rows if f"{x[4]:%Y-%m-%d}" == day)
    lines += ["", f"{len(rows)} pieces stand: {on_day} rendered on {day}, {len(rows) - on_day} on other days"
              + (f"; {blank} of them show nothing (each leaves the original on all its frames)" if blank else "") + ".", "",
              "## Renders this table does not lay that carry your clicks or words", "",
              "| frames | subject | piece | your last choice on it (Central) | your words | why this table does not lay it |", "|---|---|---|---|---|---|",
              *[f"| {r['first']}-{r['last']} | {r['subject']} | {p['render']} | "
                + (f"{p['the_owners_last_choice']['said']!r}, {owners_clock(p['the_owners_last_choice']['at']):%m-%d %H:%M}" if p.get("the_owners_last_choice") else "none")
                + f" | {cell('; '.join(words_after(p.get('the_owners_last_choice'), p)) or 'none')} | {cell(p['why'])} |"
                for r in table["reading"] for p in r.get("passed_over") or [] if p.get("the_owners_last_choice") or p.get("the_owners_words")],
              "",
              f"## Renders of {day} in the ledger that this table does not lay", "",
              f"This table covers {span[0]}-{span[1]}; a render on other frames is in no shot of it, and may stand in another table." if span else "This table has no shot.",
              "", "| frames | piece | rendered (Central) | why this table does not lay it |", "|---|---|---|---|"]
    passed: dict[str, list[str]] = {}
    for r in table["reading"]:
        for s in r.get("passed_over") or []:
            passed.setdefault(s["render"], []).append(f"on {r['first']}-{r['last']}: {s['why']}")
    absent = table.get("not_in_the_shot") or []
    that_day = sorted((r for r in ledger["renders"].values() if f"{owners_clock(r['written']):%Y-%m-%d}" == day), key=lambda r: r["written"])
    for r in that_day:
        if r["name"] in stands:
            continue
        g = r.get("graph") or {}
        frames = f"{g['first']}-{g['first'] + g['frames'] - 1}" if g.get("first") is not None and g.get("frames") else "not in its record"
        span_r = [g["first"], g["first"] + g["frames"] - 1] if g.get("first") is not None and g.get("frames") else None
        out_of_shot = next((n for n in absent if span_r and n["subject"] in (r.get("subjects") or [r.get("subject")]) and n["first"] <= span_r[1] and n["last"] >= span_r[0]), None)
        why = "; ".join(passed.get(r["name"]) or []) or (f"its subject {out_of_shot['subject']} is not in the shot {out_of_shot['first']}-{out_of_shot['last']} by the shots record"
                                       if out_of_shot else "on no shot of this table")
        lines.append(f"| {frames} | {r['name']} | {owners_clock(r['written']):%m-%d %H:%M} | {why} |")
    lines += ["", f"{len(that_day)} renders of {day} in the ledger; {sum(1 for r in that_day if r['name'] in stands)} of them stand in this table."]
    return "\n".join(lines) + "\n"


def by_day(a: argparse.Namespace) -> int:
    try:
        datetime.date.fromisoformat(a.day)
    except ValueError:
        raise SystemExit(f"--day {a.day!r} is not YYYY-MM-DD") from None
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refused: {out} is there already; a kept output is not replaced")
    table = json.loads(Path(a.table).read_text())
    named = re.search(r"from the ledger `([^`]+)`", str(table.get("made") or ""))
    if not named:
        raise SystemExit(f"--table {a.table}: the table names no ledger it was made from ({str(table.get('made'))[:200]})")
    made_from = Path(named.group(1)) if Path(named.group(1)).is_absolute() else HERE.parent / named.group(1)
    if made_from.resolve() != Path(a.ledger).resolve():
        raise SystemExit(f"--ledger {a.ledger}: the table says it was made from another ledger, {named.group(1)}")
    joined_ledger(a.ledger)
    text = by_day_text(table, json.loads(Path(a.ledger).read_text()), a.day, Path(a.table).name, Path(a.ledger).name)
    out.write_text(text)
    print(text, end="")
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="the table, the clip, its stack and the stack's check, a span at a time")
    b.add_argument("--ledger", required=True, help="the render ledger's JSON (bench/render_ledger.py --json)")
    b.add_argument("--shots", required=True, help="the video's shots record")
    b.add_argument("--plan", required=True, help="the video's plan")
    b.add_argument("--source", required=True, help="the source video; its stem begins every name")
    b.add_argument("--clips", required=True, help="the folder the clip, its stack and the assembler's record go into")
    b.add_argument("--tables", required=True, help="the folder the table, its reading, its JSON and each step's output go into")
    b.add_argument("--stack-work", required=True, help="a scratch folder for the stack's frames (never the share)")
    b.add_argument("--stamp", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ"),
                   help="the time in every name (default: now, UTC)")
    b.add_argument("--also-capture", action="append", default=[], metavar="FOLDER", help="handed to the table as given; repeatable")
    b.add_argument("--prefer", action="append", default=[], metavar="FIRST-LAST:SUBJECT=LOAD:reason", help="handed to the table as given; repeatable")
    b.add_argument("--hold", action="append", default=[], metavar="FIRST-LAST:SUBJECT=LOAD:reason", help="handed to the table as given; repeatable")
    b.add_argument("--dry", action="store_true", help="write the table and print the two commands; run neither (the stamp is then taken)")
    b.add_argument("spans", nargs="+", metavar="FIRST-LAST", help="a whole shot, or a run of whole shots, of the shots record")
    i = sub.add_parser("items", help="one review item a built span, at the top of the review list")
    i.add_argument("--items", required=True, help="the review list (data/<stem>/review/items.json); the clips are linked into its media folder")
    i.add_argument("--clips", required=True, help="the folder `build` wrote the clips into")
    i.add_argument("--tables", required=True, help="the folder `build` wrote the tables into; the copy of the list before the write goes here")
    i.add_argument("--stem", required=True, help="the source video's stem, as in the clips' names")
    i.add_argument("--stamp", required=True, help="the time in the clips' names")
    i.add_argument("--by", required=True, help="the session that adds the items, written into them")
    i.add_argument("--note", action="append", default=[], metavar="FIRST-LAST=words", help="the words of the session putting the item on the list (--by) for a span's item, added whole and said as theirs; repeatable")
    i.add_argument("--dry", action="store_true", help="print the items; write nothing")
    i.add_argument("spans", nargs="+", metavar="FIRST-LAST")
    d = sub.add_parser("by-day", help="what stands in a table by the day each piece was rendered, and the day's renders that do not stand")
    d.add_argument("--table", required=True, help="the delivery table's JSON")
    d.add_argument("--ledger", required=True, help="the render ledger's JSON the table was made from")
    d.add_argument("--day", required=True, help="the day asked about, YYYY-MM-DD, on the owner's clock")
    d.add_argument("--out", required=True, help="the .md to write")
    a = p.parse_args(argv)
    return {"build": build, "items": items, "by-day": by_day}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
