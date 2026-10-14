#!/usr/bin/env python3
"""A shot's joined review clip: the shot with every piece that stands on it laid, stacked for the eye, one review item a shot, each piece with the day it was rendered.

    <python> bench/shot_review_clips.py build --ledger LEDGER.json --shots SHOTS.json --plan PLAN.json --source SOURCE.mp4 \\
        --clips <share>/Video/<stem>/day_<date>/shots --tables data/<date>_<stem>_delivery_tables --stack-work SCRATCH \\
        [--stamp T] [--also-capture FOLDER]... [--prefer ...]... [--hold ...]... [--dry] FIRST-LAST [FIRST-LAST ...]
    <python> bench/shot_review_clips.py items --items data/<stem>/review/items.json --clips FOLDER --tables FOLDER --stamp T \\
        --by NAME [--note FIRST-LAST=words]... [--dry] FIRST-LAST [FIRST-LAST ...]
    <python> bench/shot_review_clips.py by-day --table T.json --ledger LEDGER.json --day YYYY-MM-DD --out OUT.md

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
deliveries' (`bench/capture_masked_run.py`). A throwaway build leaves them there, and a person removes them.

A one-shot table says, for a subject the shots record has and no row of that table lays, that they are the
original there. The assembler takes that line when the shots record lists the subject and keeps it in its record
as the table's word alone (`shots.said_of_subjects_no_row_has`); the evening's script cut those lines out first.

**`items`** puts one item a span at the top of the review list, the stacked clip linked into the list's `media`
folder. It is handed the build by its stamp and reads that build's named files. Its words are made from the span's
table JSON and the assembler's record: every piece that stands, its frames, the day and hour it was rendered on the
owner's clock (`piece_gaps.OWNERS_CLOCK`), the owner's last choice on it or that they made none, the frames inside
it that its load left as the original or that what it leaves is NOT KNOWN; a subject the shots record is unsure of
(with any piece of theirs that stands), one it says is not in the shot (the table's `not_in_the_shot`), and one
with nothing standing. `--note` adds a person's words to a span's item, whole, after the pieces; one a span. The
buttons are the table's own (`delivery_table.RENDER_ITEM_BUTTONS`). Refused: a span not built, an id already on the
list, a clip whose assembler's record fails, and a stack whose reading has a hard fault (a frame count that is not
the span's, or sound that is not the clip's). How well each stack row fits its frames is printed, not judged. A
copy of the list, named with the time of the write, goes into `--tables` before the one write.

**`by-day`** writes the table of what stands by the day it was rendered: a row a piece in frame order with why it
stands (the step of the table's rule), then the renders of `--day` in the ledger that do not stand, with why.

**What it cannot see.** Whether a piece is good, or whether the owner would pick it: the table's rule says why it
stands and no more. A `source` line for a subject no row lays is checked against no picture. The day is the
render file's `written` time in the ledger, which is when the file was written, not when it was queued.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delivery_table  # noqa: E402
import stack_delivery  # noqa: E402
from piece_gaps import OWNERS_CLOCK  # noqa: E402

STEP_WORDS = {"step 0": "the lead's --prefer (a session's word)", "step 1": "the owner's keep",
              "step 2": "the newest render in the frozen form; the owner has made no keep on it"}


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
    return datetime.datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(OWNERS_CLOCK)


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
    return out + ([] if got["sound_packets"]["the_same"] else ["the stacked file's sound is NOT the clip's"])


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


#: How many renders passed over are named in an item, and how much of each one's reason. A display cut, reasoned: the
#: item is read on the owner's page; the table's reading beside the clip has every one whole, and the item says how many.
PASSED_OVER_NAMED, REASON_CHARS = 3, 110


def item_words(table: dict, keep: tuple = delivery_table.KEEP) -> tuple[int, list[str]]:
    """(pieces, one sentence a piece or a subject) for a span's item, from the span's table JSON alone: the subjects
    of its reading (present or unsure), then those its `not_in_the_shot` says are not in the shot."""
    said, pieces = [], 0
    for e in table["reading"]:
        who = e["subject"]
        if e["present"] == "unsure":
            said.append(f"{who}: the shots record is unsure they are in this shot (yours to answer on the who page)"
                        + ("" if e["stands"] else "; the original throughout"))
        for s in e["stands"]:
            pieces += 1
            c = s.get("the_owners_last_choice")
            day = owners_clock(s["written"])
            choice = (f"your keep of {owners_clock(c['at']):%m/%d}" if c and c["said"] in keep
                      else f"your last choice on it: \"{c['said']}\"" if c else "not judged by you yet")
            left = s.get("leaves_as_the_original")
            said.append(f"{who}: {s['render']} on {', '.join(f'{x}-{y}' for x, y in s['frames'])}, rendered {day:%m/%d} ({day:%H:%M} Central); {choice}"
                        + ("; what it leaves as the original inside it is NOT KNOWN (its ledger entry does not say)" if left is None
                           else f"; inside it the piece leaves the original on {', '.join(f'{x}-{y}' for x, y in left)} (its capture had no part of them there)"
                           if left else ""))
        if not e["stands"] and e["present"] != "unsure":
            passed = e["passed_over"]
            said.append(f"{who}: the original on the whole shot: no piece of theirs stands here"
                        + (f" ({len(passed)} render(s) passed over, the newest {min(len(passed), PASSED_OVER_NAMED)}: "
                           + "; ".join(p["render"] + ", " + (p["why"] if len(p["why"]) <= REASON_CHARS else p["why"][:REASON_CHARS] + "...")
                                       for p in passed[:PASSED_OVER_NAMED]) + ")" if passed else ""))
        for g in (e.get("left_as_the_source") or []) if e["stands"] else []:
            said.append(f"{who}: the original on {g[0]}-{g[1]}, where no piece of theirs is laid")
    for n in table.get("not_in_the_shot") or []:
        said.append(f"{n['subject']}: not in this shot by the shots record ({n['why']})")
    return pieces, said


def make_item(name: str, span: tuple[int, int], table: dict, record: dict, link: str, stamp: str, by: str, note: str | None) -> dict:
    a, b = span
    num, den = (int(v) for v in record["video"]["rate"].split("/"))
    fps = num / den
    pieces, said = item_words(table)
    return {"id": f"{name}_stacked",
            "title": f"{name}_stacked: the shot {a}-{b} with every piece that stands on it laid, stacked: the original on top, what each render was given "
                     "in the middle, the joined shot below, the frame number on every frame",
            "file": Path(link).name, "path": link, "first": str(a), "frames": str(b - a + 1), "fps": f"{fps:.3f}",
            "seconds": f"about {a / fps:.1f} s to {(b + 1) / fps:.1f} s of the video",
            "state": "open", "state_words": "not watched by anyone on playback; yours to watch. A joined clip of one shot: every piece that stands on it is laid, "
                                            "so nobody in it is the original where a piece stands.",
            "what": f"The shot {a}-{b}, {b - a + 1} frames, built by {by} from the table of {stamp}; nothing was rendered for it. "
                    f"{pieces} piece(s) laid, each with the day it was rendered: " + ". ".join(said) + "." + (f" {note}" if note else ""),
            "think": "", "look": "", "options": list(delivery_table.RENDER_ITEM_BUTTONS),
            "prompt": "No single text: this clip is laid from the pieces named above, each rendered with its own text. Each piece's own item on this page shows its text.",
            "prompt_note": f"The table this clip was built from: t_{name}.txt beside the day's tables, with its reading.", "fixed": "Nothing was fixed in this clip."}


def items(a: argparse.Namespace) -> int:
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
    new = []
    for span in spans:
        name = clip_name(a.stem, *span, a.stamp)
        f = files_of(name, clips, tables)
        missing = [str(f[k]) for k in ("stacked", "record", "json", "stack_reading") if not f[k].is_file()]
        if missing:
            raise SystemExit(f"{span[0]}-{span[1]}: not built: {', '.join(missing)} not there")
        if f"{name}_stacked" in ids:
            raise SystemExit(f"{span[0]}-{span[1]}: the list already has an item {name}_stacked")
        record = json.loads(f["record"].read_text())
        if record["verdict"] != "passes":
            raise SystemExit(f"{span[0]}-{span[1]}: the assembler's record says {record['verdict_line']!r}: no item for a clip that fails")
        got = json.loads(f["stack_reading"].read_text())
        faults = stack_faults(got, span)
        if faults:
            raise SystemExit(f"{span[0]}-{span[1]}: the stack's reading has a hard fault: {'; '.join(faults)}: no item for it")
        print(f"{span[0]}-{span[1]}: the stack read back: " + " | ".join(stack_delivery.check_lines(got, span[0])[2:]))
        link = os.path.relpath(media / f["stacked"].name, HERE.parent)
        new.append((f["stacked"], media / f["stacked"].name, make_item(name, span, json.loads(f["json"].read_text()), record, link, a.stamp, a.by, notes.get(span))))
    for _, _, item in new:
        print(f"{item['id']}: {item['what']}")
    if a.dry:
        print(f"dry: nothing written; {len(new)} item(s) would go to the top of {path}")
        return 0
    base = f"items.before_{a.by}_{a.stamp}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}"
    copy, k = tables / f"{base}.json", 1
    while copy.exists():                          # two writes in one second: each keeps its own copy
        k += 1
        copy = tables / f"{base}_{k}.json"
    shutil.copy(path, copy)
    for target, link, _ in new:
        if not os.path.lexists(link):
            os.symlink(target, link)
    listed["items"][0:0] = [x for _, _, x in new]
    listed["updated"], listed["by"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), a.by
    out = json.dumps(listed, indent=1) + ("\n" if raw.endswith("\n") else "")
    tmp = path.with_name(path.name + f".tmp_{a.by}")
    tmp.write_text(out)
    os.replace(tmp, path)
    back = json.loads(path.read_text())
    print(f"added: the list's first items are {[x['id'] for x in back['items'][:len(new)]]}; {len(back['items'])} items; the copy before: {copy}")
    return 0


def why_words(why: str) -> str:
    return next((words for step, words in STEP_WORDS.items() if why.startswith(step)), why)


def by_day_text(table: dict, ledger: dict, day: str, table_name: str) -> str:
    """What stands in a table, a row a piece by the day it was rendered, and the renders of `day` that do not stand."""
    rows = []
    for r in table["reading"]:
        for s in r.get("stands") or []:
            rows.append((s["frames"][0][0], ",".join(f"{x}-{y}" for x, y in s["frames"]), r["subject"], s["render"], owners_clock(s["written"]), why_words(s["why"])))
    rows.sort()
    stands = {x[3] for x in rows}
    lines = [f"# What stands in {table_name}, by the day each piece was rendered", "",
             f"Read from the table's own reading and the ledger by `bench/shot_review_clips.py by-day`; the day is the render file's `written` time on the "
             f"owner's clock (US Central). `{day}` is the day asked about.", "",
             "| frames | subject | piece | rendered (Central) | why it stands |", "|---|---|---|---|---|"]
    lines += [f"| {fr} | {sub} | {name} | {t:%m-%d %H:%M} | {w} |" for _, fr, sub, name, t, w in rows]
    on_day = sum(1 for x in rows if f"{x[4]:%Y-%m-%d}" == day)
    lines += ["", f"{len(rows)} pieces stand: {on_day} rendered on {day}, {len(rows) - on_day} on other days.", "",
              f"## Renders of {day} in the ledger that do not stand", "",
              "| frames | piece | rendered (Central) | why it does not stand |", "|---|---|---|---|"]
    passed = {}
    for r in table["reading"]:
        for s in r.get("passed_over") or []:
            passed.setdefault(s["render"], s["why"])
    that_day = sorted((r for r in ledger["renders"].values() if f"{owners_clock(r['written']):%Y-%m-%d}" == day), key=lambda r: r["written"])
    for r in that_day:
        if r["name"] in stands:
            continue
        g = r.get("graph") or {}
        frames = f"{g['first']}-{g['first'] + g['frames'] - 1}" if g.get("first") is not None and g.get("frames") else "not in its record"
        lines.append(f"| {frames} | {r['name']} | {owners_clock(r['written']):%m-%d %H:%M} | {passed.get(r['name'], 'on no shot of this table')} |")
    lines += ["", f"{len(that_day)} renders of {day} in the ledger; {sum(1 for r in that_day if r['name'] in stands)} of them stand."]
    return "\n".join(lines) + "\n"


def by_day(a: argparse.Namespace) -> int:
    try:
        datetime.date.fromisoformat(a.day)
    except ValueError:
        raise SystemExit(f"--day {a.day!r} is not YYYY-MM-DD") from None
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refused: {out} is there already; a kept output is not replaced")
    text = by_day_text(json.loads(Path(a.table).read_text()), json.loads(Path(a.ledger).read_text()), a.day, Path(a.table).name)
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
    b.add_argument("--dry", action="store_true", help="write the table and print the three commands; run none of them")
    b.add_argument("spans", nargs="+", metavar="FIRST-LAST", help="a whole shot, or a run of whole shots, of the shots record")
    i = sub.add_parser("items", help="one review item a built span, at the top of the review list")
    i.add_argument("--items", required=True, help="the review list (data/<stem>/review/items.json); the clips are linked into its media folder")
    i.add_argument("--clips", required=True, help="the folder `build` wrote the clips into")
    i.add_argument("--tables", required=True, help="the folder `build` wrote the tables into; the copy of the list before the write goes here")
    i.add_argument("--stem", required=True, help="the source video's stem, as in the clips' names")
    i.add_argument("--stamp", required=True, help="the time in the clips' names")
    i.add_argument("--by", required=True, help="the session that adds the items, written into them")
    i.add_argument("--note", action="append", default=[], metavar="FIRST-LAST=words", help="a person's words for a span's item, added whole; repeatable")
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
