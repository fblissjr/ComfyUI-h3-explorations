#!/usr/bin/env python3
"""The delivery table, generated from the render ledger by one stated rule: which render stands on each shot, and why.

    <python> bench/delivery_table.py --ledger LEDGER.json --shots SHOTS.json --plan PLAN.json \\
        [--prefer "FIRST-LAST:SUBJECT=LOAD:the reason"]... --table OUT.txt --reading OUT.md [--json OUT.json]

**What it buys.** The table `bench/assemble_delivery.py` builds a clip from was typed by hand at the end of a day,
from memory of which render was the good one. This writes it from files: the ledger (`bench/render_ledger.py`:
every render, how it was made, the owner's lines on it), the shots file and the plan. Beside the table it writes
a reading that says, for every shot and subject, which render stands and why in one line, and every render it
passed over with why. It assembles nothing.

**The rule for "stands", in this order. Each step is a fact from the files, never a judgement of a picture.**

0. `--prefer`: a render the lead names for a shot and a subject, with a reason that is printed on its line. It
   goes first among the renders that MAY stand; it cannot make one stand that step 3 rules out, or one the
   owner's last choice says is not the one.
1. The owner's keep (`KEEP`: the last choice the owner made on that render is one of those words; a line said
   of a joined clip is not a choice on a render) on a render whose file agrees with its build record.
2. Else the newest render whose file agrees with its build record, whose last choice by the owner does not say
   it is not the one (`NOT_THE_ONE`; `not sure` bars nothing), and whose form is the frozen one for that
   subject: the plan's region for the subject, and a reference whose own records say it is the body model cut
   to the head with the face points drawn. The plain body model cut to the head (no points, no marks) is the
   form's stated exception for a face under the drawing line; no field this tool reads says why a load took
   it, so such a render stands here and its line says the reason was not read.
3. Never: a render with no build record, one whose queue record names no file, one whose file disagrees with
   its record, a whole-person render of any subject, a render of two subjects in one pass (not laid by this
   tool). The owner's rule of 2026-10-13: a piece whose making cannot be said is rendered again.
4. Nothing left: a `source` line for the shot saying no render in the frozen form exists yet, and the reading
   names the newest render that was passed over, which is not laid.

On one shot several renders can stand on different frames. They are taken in the order above, and each later
one is laid only on frames no earlier one took. Frames of a shot that no render takes are a `source` line when
they are the whole shot (the assembler takes a `source` line only over a whole shot) and a comment in the table
otherwise.

**`restore=`** follows the practice of the table of 2026-10-12, as its header states it: a subject's row gives
back every OTHER subject the shots file has in that shot (`yes` or `unsure`): `restore=<other>:whole` on a face
row, which has no business inside anybody else, and `restore=<other>` on any other row. The assembler needs a
capture that holds each named subject's track to act on one.

**The names** are the plan's labels. The plan's `aliases` are printed in the table's header as the `--alias`
flags a capture with an older name needs.

**The assembler's command** is printed whole, in the reading and on the terminal (`assembly`): the table, the
span, every capture the standing pieces' build records name, any handed in with `--also-capture`, an `--alias`
only for an older name a listed capture uses, and `--assemble-args` as given (`--source`, `--out`, `--size`:
what only the caller knows). Beside it, for each `restore=`, on how many of its row's frames a listed capture
holds that subject's track.

**What it cannot see.** Whether a render is good. Whether the render the rule lets stand is the one the owner
would pick: step 2 lets the newest frozen render stand where the owner has made no choice on it or was not sure,
and the reading prints their last choice beside it. Why a load took the plain cut. Whether a restore's subject is
tracked on those frames. A render made after the ledger was: the reading says which ledger it read.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER_SCHEMA = "h3_render_ledger/2"
#: The owner's choices that are a keep, as the review page's buttons word them.
KEEP = ("keep this one",)
#: The owner's choices that say a render is NOT the one: a render whose last choice is one of these does not stand
#: by step 2 or by `--prefer` (the lead's ruling, 2026-10-13). The words are the buttons the review page has shown
#: under a render (`data/<video>/review/items.json`, each item's `options`, as they stood that day); `not sure` is
#: not among them and bars nothing. A button worded otherwise bars nothing until it is added here.
NOT_THE_ONE = ("wrong", "wrong, and I say where", "not needed", "worse, and I say where")
#: The page's word for a choice withdrawn: the owner then has no choice on the render.
TAKEN_BACK = "taken back"
#: The ledger's states of a making (`render_ledger.MAKING`), by what each means here.
AGREES = "recorded, the file agrees"
NO_SOURCE_YET = "no render in the frozen form yet"


def _shot_coverage():
    import importlib.util
    spec = importlib.util.spec_from_file_location("shot_coverage", HERE / "shot_coverage.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def last_choice(render: dict) -> dict | None:
    """The owner's last CHOICE on a render (a button, not typed words), or None. A line said of a joined clip is
    of that clip and is not a choice on the render."""
    choices = [o for o in render.get("owner") or [] if o.get("what") == "choice" and not o.get("on_the_joined_clip")]
    last = max(choices, key=lambda o: str(o.get("at"))) if choices else None
    return None if last is None or last.get("said") == TAKEN_BACK else last


def never_stands(render: dict) -> str | None:
    """Why a render never stands (step 3), or None."""
    making = render["making"]
    if making.startswith("NO BUILD RECORD"):
        return "no build record: its making cannot be said"
    if "NAMES NO FILE" in making:
        return "its queue record names no file and no outcome: its making cannot be said from the record"
    if making != AGREES:
        said = (render["records"][-1]["disagrees"] if render.get("records") else None) or ["(the ledger gives no line)"]
        return "its file disagrees with its build record: " + "; ".join(said)
    if (render["form"].get("region")) == "whole":
        return "a whole-person render"
    if len(render.get("subjects") or []) > 1:
        return "a render of two subjects in one pass: not laid by this tool"
    return None


def form_of(render: dict, region_wanted: str) -> tuple[str | None, str]:
    """(`frozen`, `frozen, by its exception`, or None, with the words): is a render in the frozen form for its subject."""
    region, has = render["form"].get("region"), (render["form"].get("reference") or {}).get("has")
    if region != region_wanted:
        return None, f"it replaces {region or 'what no file says'}, where the plan's region for this subject is {region_wanted}"
    if not has or not has.get("cut_to_the_head"):
        return None, "its reference is not the body model cut to the head: " + (render["form"]["reference"].get("says") or "none")
    if has.get("lid_marks"):
        return None, "its reference has marks drawn on the lids, the form before the frozen one"
    if has.get("face_points"):
        return "frozen", "the frozen form: the plan's region, the body model cut to the head with the face points drawn"
    return "frozen, by its exception", ("the plain body model cut to the head, the form's exception for a face under the drawing line; WHY this load took "
                                        "it is in no field read here (the load's note or its capture's looks)")


def runs(frames: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def parse_prefer(text: str) -> dict:
    """`FIRST-LAST:SUBJECT=LOAD:the reason`: on that shot, for that subject, the load to go first, and why."""
    try:
        where, rest = text.split("=", 1)
        span, subject = where.split(":", 1)
        load, reason = rest.split(":", 1)
        first, last = (int(v) for v in span.split("-"))
        if not (subject and load and reason.strip()):
            raise ValueError
    except ValueError:
        raise SystemExit(f"--prefer {text!r} is not `FIRST-LAST:SUBJECT=LOAD:the reason` (a shot of the shots file, a subject, a load's name, "
                         "and the reason in words: it is printed)") from None
    return {"first": first, "last": last, "subject": subject, "load": load, "reason": reason.strip()}


def decide(ledger: dict, shots: dict, plan: dict, prefers: list[dict] | None = None) -> dict:
    """The table as data: `rows` (pieces laid), `sources` (whole shots left as the source), `gaps` (frames inside a
    shot that has a piece and that no piece takes) and `reading` (one entry a shot and a subject)."""
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise SystemExit(f"the ledger's schema is {ledger.get('schema')!r}, not {LEDGER_SCHEMA!r}: make it again with bench/render_ledger.py")
    prefers = list(prefers or [])
    renders = list(ledger["renders"].values())
    rows, sources, gaps, reading = [], [], [], []
    for p in prefers:
        if not any(s["first"] == p["first"] and s["last"] == p["last"] for s in shots["shots"]):
            raise SystemExit(f"--prefer names frames {p['first']}-{p['last']}, which is not a shot of the shots file")
        if not any(r["name"] == p["load"] for r in renders):
            raise SystemExit(f"--prefer names the load {p['load']!r}, which the ledger has no render of")
    for s in shots["shots"]:
        present = {label: s["who"][label]["present"] for label in shots["subjects"]}
        for label in shots["subjects"]:
            if present[label] == "no":
                continue
            wanted = plan["subjects"][label]["region"]
            on = [r for r in renders if label in (r.get("subjects") or []) and r.get("span") and r["span"][0] <= s["last"] and r["span"][1] >= s["first"]]
            preferred = [p for p in prefers if (p["first"], p["last"], p["subject"]) == (s["first"], s["last"], label)]
            may, passed = [], []
            for r in on:
                never = never_stands(r)
                choice = last_choice(r)
                kept = bool(choice and choice["said"] in KEEP)
                barred = (f"the owner's last choice on it was \"{choice['said']}\" ({str(choice['at'])[:16]}, row {choice.get('row')}): not the one"
                          if choice and choice["said"] in NOT_THE_ONE else None)
                frozen, form_words = form_of(r, wanted)
                named = next((p for p in preferred if p["load"] == r["name"]), None)
                if never:
                    if named:
                        raise SystemExit(f"--prefer names {r['name']} for {s['first']}-{s['last']}, {label}, and it never stands: {never}")
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"], "why": "never stands: " + never})
                elif named and barred:
                    raise SystemExit(f"--prefer names {r['name']} for {s['first']}-{s['last']}, {label}, and {barred}. The owner's choice is "
                                     "over the lead's preference: it stands when the owner changes it")
                elif named:
                    may.append((0, r["written"], r, f"step 0: named by --prefer: {named['reason']}", choice))
                elif kept:
                    may.append((1, choice["at"], r, f"step 1: the owner's keep of {str(choice['at'])[:16]} (row {choice.get('row')}); its file agrees "
                                                     f"with its build record", choice))
                elif frozen and barred:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": f"its file agrees with its build record and it is in the frozen form, and {barred}"})
                elif frozen:
                    may.append((2, r["written"], r, f"step 2: {form_words}; its file agrees with its build record", choice))
                else:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": "its file agrees with its build record, and it has no keep and is not in the frozen form: " + form_words})
            for p in preferred:
                if not any(m[2]["name"] == p["load"] for m in may):
                    raise SystemExit(f"--prefer names {p['load']} for {s['first']}-{s['last']}, {label}, and no render of that load covers those frames for that subject")
            may.sort(key=lambda m: str(m[1]), reverse=True)                    # the newest first,
            may.sort(key=lambda m: m[0])                                       # within the rule's steps in their order
            free = set(range(s["first"], s["last"] + 1))
            stands = []
            for step, _when, r, why, choice in may:
                take = sorted(free & set(range(r["span"][0], r["span"][1] + 1)))
                if not take:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": f"may stand ({why.split(':')[0]}) and every frame of it on this shot is taken by a render before it in the rule's order"})
                    continue
                free -= set(take)
                others = [o for o in shots["subjects"] if o != label and present[o] != "no"]
                region = r["form"].get("region")
                restores = [f"{o}:whole" if region == "face" else o for o in others]
                for a, b in runs(take):
                    rows.append({"first": a, "last": b, "file": r["file"], "name": r["name"], "piece_first": r["span"][0], "subject": label,
                                 "region": region, "restore": restores})
                stands.append({"render": r["name"], "file": r["file"], "written": r["written"], "frames": runs(take), "why": why,
                               "the_owners_last_choice": None if not choice else {"said": choice["said"], "at": choice["at"], "row": choice.get("row")}})
            left = runs(sorted(free))
            entry = {"first": s["first"], "last": s["last"], "subject": label, "present": present[label], "plan_region": wanted,
                     "stands": stands, "passed_over": sorted(passed, key=lambda x: x["written"], reverse=True), "left_as_the_source": left}
            if not stands:
                newest = entry["passed_over"][0]["render"] if entry["passed_over"] else None
                why = (NO_SOURCE_YET if present[label] == "yes" else f"the shots file is unsure this subject is in the shot; {NO_SOURCE_YET}") \
                    + (f" (the newest render passed over: {newest})" if newest else " (no render of this subject covers these frames)")
                sources.append({"first": s["first"], "last": s["last"], "subject": label, "why": why})
                entry["source_line"] = why
            elif left:
                gaps += [{"first": a, "last": b, "subject": label, "why": f"inside the shot {s['first']}-{s['last']}, which has a piece: no render that may stand covers these frames"}
                         for a, b in left]
            reading.append(entry)
    # rows of one render on touching runs with the same restores are one row; face rows after the others on shared frames
    rows.sort(key=lambda r: (r["subject"], r["file"], r["first"]))
    merged: list[dict] = []
    for r in rows:
        if merged and (merged[-1]["file"], merged[-1]["subject"], merged[-1]["restore"]) == (r["file"], r["subject"], r["restore"]) and merged[-1]["last"] + 1 == r["first"]:
            merged[-1]["last"] = r["last"]
        else:
            merged.append(dict(r))
    merged.sort(key=lambda r: (r["first"], r["region"] == "face", r["subject"]))
    return {"rows": merged, "sources": sources, "gaps": gaps, "reading": reading}


def frames(a, b) -> str:
    return f"{a}-{b}"


def tracked_in(capture: Path, labels: set[str], first: int, last: int) -> set[int] | None:
    """The source frames of `first`..`last` on which a capture holds a non-empty track of a subject known by any of
    `labels`, or None when the capture has no such subject or no manifest."""
    import numpy as np
    manifest = capture / "manifest.json"
    if not manifest.is_file():
        return None
    m = json.loads(manifest.read_text())
    entry = next((s for s in m.get("subjects", []) if s["label"] in labels), None)
    if entry is None or not entry.get("sightings"):
        return None
    saved = capture / "subjects" / entry["label"] / f"masks__{entry['sightings'][0]['by']}.npz"
    if not saved.is_file():
        return None
    on = np.load(saved)["track"].reshape(m["frames"], -1).any(axis=1)
    return {m["first_frame"] + int(f) for f in np.nonzero(on)[0] if first <= m["first_frame"] + int(f) <= last}


def assembly(out: dict, ledger: dict, plan: dict, repo: Path, table: str, span: tuple[int, int], args: str = "", also: tuple[str, ...] = ()) -> dict:
    """The assembler's command for this table, whole, and what each `restore=` has to go on.

    The captures are the standing pieces' own (each piece's build record names its capture), in the table's row order, then any handed in
    (`also`: a preview's capture of the span, for its shot tables). `--alias` is printed for a name of the plan's
    `aliases` that a listed capture labels a subject by, and for no other. `args` is what only the caller knows
    (`--source`, `--out`, `--size`), taken as given; without a `--source` and an `--out` in it the command is said
    to be not whole. For each `restore=` on a row: on how many of the row's frames a listed capture holds that
    subject's track. Figures; a restore with nothing to go on is the assembler's to flag when it builds."""
    rel = lambda p: str(Path(p).relative_to(repo)) if Path(p).is_absolute() and Path(p).is_relative_to(repo) else str(p)
    captures: list[str] = []
    for r in out["rows"]:
        records = ledger["renders"][r["file"]].get("records") or []
        capture = records[-1].get("capture") if records else None
        if capture and rel(capture) not in captures:
            captures.append(rel(capture))
    captures += [rel(c) for c in also if rel(c) not in captures]
    aliases = {k: v for k, v in (plan.get("aliases") or {}).items() if not k.startswith("_")}
    names = lambda label: {label} | {old for old, new in aliases.items() if new == label}
    used = set()
    real = lambda c: Path(c) if Path(c).is_absolute() else repo / c
    for c in captures:
        manifest = real(c) / "manifest.json"
        if manifest.is_file():
            used |= {s["label"] for s in json.loads(manifest.read_text()).get("subjects", [])} & set(aliases)
    restores = []
    for r in out["rows"]:
        for x in r["restore"]:
            other = x.split(":")[0]
            held, where = set(), []
            for c in captures:
                got = tracked_in(real(c), names(other), r["first"], r["last"])
                if got:
                    held |= got
                    where.append(c)
            restores.append({"row": f"{frames(r['first'], r['last'])} {r['name']} ({r['subject']})", "restore": x, "frames": r["last"] - r["first"] + 1,
                             "frames_with_a_track": len(held), "in_the_captures": where})
    whole = "--source " in f" {args} " and "--out " in f" {args} "
    parts = ["bench/assemble_delivery.py", f"--table {table}", f"--span {span[0]}-{span[1]}"] + ([args] if args else []) \
        + [f"--capture {c}" for c in captures] + [f"--alias {old}={aliases[old]}" for old in sorted(used)]
    return {"command": " \\\n    ".join(parts), "whole": whole, "captures": captures, "captures_handed_in": [rel(c) for c in also],
            "aliases_used_by_a_capture": sorted(used), "restores": restores}


def table_text(out: dict, share: str, made: str, plan: dict, name: str) -> str:
    """The table as the assembler reads it (`assemble_delivery.read_table`, `read_intents`)."""
    aliases = {k: v for k, v in (plan.get("aliases") or {}).items() if not k.startswith("_")}
    lines = [f"# {name}: GENERATED by bench/delivery_table.py, never typed. {made}",
             "# The names are the plan's labels. " + ("A capture that labels a subject by an older name is read with: "
                                                      + " ".join(f"--alias {old}={new}" for old, new in aliases.items()) + " (the plan's aliases)." if aliases else "The plan has no aliases."),
             "# WHICH RENDER STANDS is the rule in that tool's docstring (the lead's --prefer; the owner's keep on a render whose file agrees with its build",
             "# record; else the newest such render in the frozen form; never one with no record, a disagreeing file or a whole person). The reading beside",
             "# this file says why each piece stands and what was passed over.",
             "# restore= is the practice of the table of 2026-10-12: every other subject the shots file has in the shot is given back, `:whole` on a face row."]
    for r in out["rows"]:
        lines.append(f"{frames(r['first'], r['last'])} {Path(share) / r['file']} {r['piece_first']} subject={r['subject']}"
                     + "".join(f" restore={x}" for x in r["restore"]))
    if out["gaps"]:
        lines.append("# Left as the source INSIDE a shot that has a piece (a `source` line covers a whole shot, so these are said here):")
        lines += [f"#   {frames(g['first'], g['last'])} {g['subject']}: no render that may stand covers these frames" for g in out["gaps"]]
    for s in out["sources"]:
        lines.append(f"source {frames(s['first'], s['last'])} subject={s['subject']} {s['why']}")
    return "\n".join(lines) + "\n"


def reading_text(out: dict, made: str, table: str, built: dict | None = None) -> str:
    """The reading: the assembler's command, then a section a shot and a subject, what stands and why, what was
    passed over and why."""
    lines = ["# Which render stands on each shot, and why: the reading beside a generated delivery table", "", made, "",
             f"The table: `{table}`. Nothing was assembled. The rule is in `bench/delivery_table.py`'s docstring; every `why` below is one of its steps, "
             "a fact from the files, and none says a render is good.", ""]
    if built is not None:
        lines += ["## The command that builds a clip from this table", "", "```", "<the ComfyUI venv's python> " + built["command"], "```", ""]
        if not built["whole"]:
            lines += ["NOT WHOLE: no `--source` and `--out` were handed to this tool (`--assemble-args`), so the command lacks them.", ""]
        lines += ["The captures are the standing pieces' own, from each piece's build record"
                  + (f", then those handed in: {', '.join(built['captures_handed_in'])}" if built["captures_handed_in"] else "; none was handed in")
                  + ". " + ("`--alias` is given for the older names a listed capture uses: " + ", ".join(built["aliases_used_by_a_capture"]) + "."
                            if built["aliases_used_by_a_capture"] else "No listed capture labels a subject by an older name, so there is no `--alias`."), "",
                  "What each `restore=` has to go on: the frames of its row on which a listed capture holds that subject's track. A figure, not a "
                  "failure; the assembler flags a restore with no mask when it builds.", ""]
        lines += [f"- `{x['row']}` `restore={x['restore']}`: a track on {x['frames_with_a_track']} of the row's {x['frames']} frames"
                  + (f" (in {', '.join(x['in_the_captures'])})" if x["in_the_captures"] else ": NO listed capture holds that subject's track on these frames")
                  for x in built["restores"]]
        lines += ["" if built["restores"] else "No row has a `restore=`.", ""]
    for e in out["reading"]:
        lines += [f"## {frames(e['first'], e['last'])}, {e['subject']} (the plan: {e['plan_region']}"
                  + ("" if e["present"] == "yes" else f"; the shots file says present: {e['present']}") + ")", ""]
        for st in e["stands"]:
            choice = st["the_owners_last_choice"]
            lines.append(f"- STANDS on {', '.join(frames(a, b) for a, b in st['frames'])}: **`{st['render']}`** (written {st['written']}). {st['why']}. "
                         + (f"The owner's last choice on it: \"{choice['said']}\", {str(choice['at'])[:16]}." if choice else "The owner has made no choice on it."))
        if not e["stands"]:
            lines.append(f"- NOTHING STANDS. The table's line: `source {frames(e['first'], e['last'])} subject={e['subject']} {e['source_line']}`")
        elif e["left_as_the_source"]:
            lines.append(f"- left as the source inside this shot: {', '.join(frames(a, b) for a, b in e['left_as_the_source'])} (no render that may stand covers them)")
        for p in e["passed_over"]:
            lines.append(f"- passed over: `{p['render']}` (written {p['written']}): {p['why']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ledger", required=True, help="the render ledger's JSON (bench/render_ledger.py --json)")
    p.add_argument("--shots", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--prefer", action="append", default=[], metavar="FIRST-LAST:SUBJECT=LOAD:reason",
                   help="on that shot, for that subject, the load to go first among those that may stand; the reason is printed; repeatable")
    p.add_argument("--assemble-args", default="", metavar='"--source S --out O --size source"',
                   help="what only the caller knows of the build, printed into the assembler's command as given")
    p.add_argument("--also-capture", action="append", default=[], metavar="FOLDER",
                   help="a capture to hand the assembler beside the standing pieces' own (a preview's capture of the span); repeatable")
    p.add_argument("--table", required=True, help="where the table is written")
    p.add_argument("--reading", required=True, help="where the reading is written")
    p.add_argument("--json", help="the same as data")
    a = p.parse_args()
    coverage = _shot_coverage()
    shots = coverage.load_shots(a.shots)
    plan = coverage.load_plan(a.plan, shots["subjects"])
    ledger = json.load(open(a.ledger))
    out = decide(ledger, shots, plan, [parse_prefer(x) for x in a.prefer])
    commit = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    made = (f"Made {now} at commit {commit} from the ledger `{a.ledger}` ({str(ledger.get('made') or 'which does not say when it was made').split(' from')[0]}), "
            f"`{a.shots}` and `{a.plan}`" + ("; with " + "; ".join(f"--prefer {x!r}" for x in a.prefer) if a.prefer else "; no --prefer") + ".")
    built = assembly(out, ledger, plan, HERE.parent, a.table, tuple(shots["span"]), a.assemble_args, tuple(a.also_capture))
    Path(a.table).write_text(table_text(out, ledger["share"], made, plan, Path(a.table).stem))
    Path(a.reading).write_text(reading_text(out, made, a.table, built))
    if a.json:
        Path(a.json).write_text(json.dumps({"made": made, **out, "assembly": built}, indent=1) + "\n")
    print("the assembler's command" + ("" if built["whole"] else " (NOT WHOLE: no --source and --out in --assemble-args)") + ":\n  " + built["command"])
    laid = sum(r["last"] - r["first"] + 1 for r in out["rows"])
    print(f"{len(out['rows'])} piece row(s) over {laid} frame-rows, {len(out['sources'])} `source` line(s), {len(out['gaps'])} stretch(es) left as the "
          f"source inside a shot that has a piece")
    print("wrote", a.table, "and", a.reading, *(["and", a.json] if a.json else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
