#!/usr/bin/env python3
"""Where a masked video stands: one row a shot and a subject, with what the plan wants there, what the delivery
table lays there, and what else has been rendered on those frames.

    <python> bench/shot_coverage.py --shots SHOTS.json --plan PLAN.json --table TABLE.txt [--renders RENDERS.json] \\
        [--verdicts VERDICTS.jsonl] [--json OUT.json]

**What it buys.** On 2026-10-11 nobody could say, per shot and per subject, what the owner had ruled, what
render existed and what was missing: the delivery table was the nearest thing and it is typed by hand, and
the day's renders went to one 140-frame shot while five stretches had no render at all. This reads the
three files that already hold those facts and prints them side by side, so "finish the video" is a list.

**What it reads, each a named file handed in; nothing is found by a key.**

- `--shots`: the cuts of the video in source frames, and for every shot and every subject whether that
  subject is in it (`yes`, `no`, `unsure`) with who said so and from what. Schema `h3_shots/1`
  (`SHOTS_SCHEMA`); `data/<stem>/shots.json` is where a video's lives. A shot a subject is `unsure` on is a
  row of its own, since nothing can be planned on it.
- `--plan`: for each subject what is replaced by default (`face`, `face and hair`, `whole`) with who ruled it,
  how a delivery piece is known to be that subject's (`load_prefixes`), the table's own names for the
  subjects where they differ (`aliases`, as the delivery tool's `--alias`), and dated notes on a stretch
  (`departures`). Schema `h3_plan/1` (`PLAN_SCHEMA`).
- `--table`: the delivery table `bench/assemble_delivery.py` builds from; its two readers are imported from
  there, so the format has one definition.
- `--renders` (optional): the inventory of renders of this video read from each render's own saved graph
  (`data/<stem>/poc/renders.json` today, written by that page's builder). With it, each piece's region is
  read from its graph (`whole subject` or `the wired parts`), and the renders on a shot that the table does
  not use are listed, so a session sees that eleven renders of one shot exist before making a twelfth.
- `--verdicts` (optional): the owner's lines on renders, as `bench/page_answers_to_verdicts.py` copies them
  from the review page into `data/<stem>/verdicts/verdicts.jsonl`; each is printed under the render it is
  about, oldest first, unreworded, so the table carries the owner's eye beside the files' facts.

**What it prints, a row a shot and a subject present or unsure:** the frames, the subject, what the plan
wants, what the table lays there (a piece and its region, or the `source` line's own words, or nothing), and
one of the states in `STATES`, each a plain fact about the files and not a judgement of any render. Then
frames by state for each subject, and the pieces the table lays that no plan row claims. A render in the
table is "as planned" when its graph's region is the kind the plan names (parts for `face` and `face and
hair`; the graph does not say which parts, so face against face and hair is NOT read here and the row says
so); "not as planned" when the kinds differ. A piece covering part of a shot says which frames.

**What it cannot see.** Whether a render is any good (the owner's eye on playback; the records under
`bench/results/`). Whether the parts a render replaced are the face or the face and hair (not in the saved
graph's summary). Whether a shot's cuts are right: the shots file carries the tracker's cuts and whoever
looked; a `note` on a shot is a person's doubt, printed beside it.

The frame numbers are the copy's, as the whole folder's are (`clip_folder.py`, "One clock").
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS_SCHEMA, PLAN_SCHEMA = "h3_shots/1", "h3_plan/1"
PRESENT = ("yes", "no", "unsure")
REGIONS = ("face", "face and hair", "whole")
#: What a saved graph's `replace` says, and which plan regions it is the kind of.
GRAPH_KIND = {"whole subject": ("whole",), "the wired parts": ("face", "face and hair")}
STATES = (
    "as planned",            # a table piece of this subject covers the shot and its graph's region is the plan's kind
    "not as planned",        # a piece covers it and its graph's region is the other kind
    "region not readable",   # a piece covers it and no renders inventory names it, so its region is not read
    "partly",                # a piece of this subject covers some frames of the shot and not the rest
    "source, said why",      # a `source` line leaves the subject as the original here, with words
    "source, no line",       # the subject is here and the table says nothing about them on these frames
    "presence unsure",       # the shots file cannot say the subject is here, so nothing is planned
)


def _delivery():
    """`read_table` and `read_intents` from the delivery tool, imported from its file so the format has one home."""
    spec = importlib.util.spec_from_file_location("assemble_delivery", HERE / "assemble_delivery.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_table, mod.read_intents


def load_shots(path) -> dict:
    d = json.load(open(path))
    if d.get("schema") != SHOTS_SCHEMA:
        raise SystemExit(f"{path}: schema is {d.get('schema')!r}, not {SHOTS_SCHEMA!r}")
    shots = d.get("shots") or []
    last = None
    for s in shots:
        if s["first"] > s["last"] or (last is not None and s["first"] != last + 1):
            raise SystemExit(f"{path}: shot {s['first']}-{s['last']} does not follow {last} with no gap and no overlap")
        last = s["last"]
        for label in d["subjects"]:
            who = s["who"].get(label)
            if not who or who.get("present") not in PRESENT:
                raise SystemExit(f"{path}: shot {s['first']}-{s['last']}: `{label}` needs `present` in {PRESENT}")
    lo, hi = d["span"]
    if shots and (shots[0]["first"] != lo or shots[-1]["last"] != hi):
        raise SystemExit(f"{path}: the shots cover {shots[0]['first']}-{shots[-1]['last']}, the span says {lo}-{hi}")
    return d


def load_plan(path, subjects) -> dict:
    d = json.load(open(path))
    if d.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"{path}: schema is {d.get('schema')!r}, not {PLAN_SCHEMA!r}")
    for label, p in d["subjects"].items():
        if label not in subjects:
            raise SystemExit(f"{path}: `{label}` is not a subject of the shots file ({', '.join(subjects)})")
        if p.get("region") not in REGIONS:
            raise SystemExit(f"{path}: `{label}`: region must be one of {REGIONS}")
    for label in subjects:
        if label not in d["subjects"]:
            raise SystemExit(f"{path}: no plan for `{label}`")
    return d


def alias(label: str, plan: dict) -> str:
    """A table's name for a subject (version a's `singer`) read through the plan's `aliases` to the plan's label."""
    return {k: v for k, v in plan.get("aliases", {}).items() if not k.startswith("_")}.get(label, label)


def subject_of(row: dict, plan: dict) -> tuple[str | None, str]:
    """Whose a table piece is: `subject=` on the row, else the load name's prefix from the plan's `load_prefixes`."""
    if row.get("subject"):
        return alias(row["subject"], plan), "the row's subject="
    name = Path(row["piece"]).name
    for label, prefixes in plan.get("load_prefixes", {}).items():
        for p in prefixes:
            if name.startswith(p):
                return label, f"the load name's prefix `{p}`"
    return None, "nothing names the subject"


def render_of(piece: str, renders: list[dict]) -> dict | None:
    for r in renders:
        if piece.endswith(r["file"]):
            return r
    return None


def overlap(a, b, c, d) -> tuple[int, int] | None:
    lo, hi = max(a, c), min(b, d)
    return (lo, hi) if lo <= hi else None


def owner_said(render_file: str | None, verdicts: list[dict]) -> list[str]:
    """The owner's lines on a render from the verdicts record, oldest first, as `at: what they said`."""
    if not render_file:
        return []
    out = []
    for v in sorted((v for v in verdicts if v.get("render") and render_file.endswith(v["render"])), key=lambda v: v["at"]):
        said = v["chosen"] if v.get("what") == "choice" else str(v.get("words") or "").strip().replace("\n", " / ")
        where = f" at frame {v['frame']}" if v.get("frame") not in (None, "") else ""
        out.append(f"{v['at'][:16]}{where}: {said}")
    return out


def coverage(shots: dict, plan: dict, rows: list[dict], intents: list[dict], renders: list[dict], verdicts: list[dict] | None = None) -> dict:
    verdicts = verdicts or []
    span = shots["span"]
    for row in rows:
        if row["first"] < span[0] or row["last"] > span[1]:
            raise SystemExit(f"table row {row['first']}-{row['last']} `{row['piece']}` is outside the shots file's span {span}")
    pieces = []
    for row in rows:
        label, how = subject_of(row, plan)
        r = render_of(row["piece"], renders)
        pieces.append(dict(row, label=label, label_how=how, render=r,
                           region=(r or {}).get("as_the_graph_says", {}).get("replace")))
    out = {"video": shots["video"], "span": span, "source_frames": shots.get("source_frames"), "rows": [],
           "unclaimed_pieces": [dict(first=p["first"], last=p["last"], piece=p["piece"], why=p["label_how"])
                                for p in pieces if p["label"] is None]}
    for s in shots["shots"]:
        for label in shots["subjects"]:
            who = s["who"][label]
            if who["present"] == "no":
                continue
            wanted = plan["subjects"][label]
            notes = [d for d in plan.get("departures", []) if d["subject"] == label
                     and overlap(s["first"], s["last"], d["first"], d["last"])]
            row = {"first": s["first"], "last": s["last"], "frames": s["last"] - s["first"] + 1, "subject": label,
                   "present": who["present"], "present_how": who.get("how", ""), "present_by": who.get("by", ""),
                   "wanted": wanted["region"], "wanted_by": wanted.get("by", ""),
                   "notes": [f"{d['first']}-{d['last']}: {d['note']} ({d.get('by', '')})" for d in notes],
                   "shot_note": s.get("note", ""), "laid": [], "left": [], "others": []}
            mine = [p for p in pieces if p["label"] == label and overlap(s["first"], s["last"], p["first"], p["last"])]
            for p in mine:
                lo, hi = overlap(s["first"], s["last"], p["first"], p["last"]) or (0, -1)
                row["laid"].append({"frames": [lo, hi], "piece": p["piece"], "region": p["region"],
                                    "named_by": p["label_how"],
                                    "crosses_a_cut": p["first"] < s["first"] or p["last"] > s["last"],
                                    "owner_said": owner_said(p["piece"], list(verdicts))})
            for it in intents:
                both = overlap(s["first"], s["last"], it["first"], it["last"])
                if alias(it["subject"], plan) == label and both:
                    lo, hi = both
                    row["left"].append({"frames": [lo, hi], "why": it["why"]})
            for r in renders:
                if not overlap(s["first"], s["last"], r["first"], r["last"]):
                    continue
                rl, _ = subject_of({"piece": r["file"]}, plan)
                if rl != label:
                    continue
                row["others"].append({"name": r["name"], "frames": [r["first"], r["last"]],
                                      "region": r.get("as_the_graph_says", {}).get("replace"),
                                      "motion": bool(r.get("motion_wired")),
                                      "in_table": any(p["render"] is r for p in mine),
                                      "owner_said": owner_said(r["file"], list(verdicts))})
            row["state"] = state_of(row)
            out["rows"].append(row)
    out["frames_by_state"] = totals(out["rows"])
    return out


def state_of(row: dict) -> str:
    if row["present"] == "unsure":
        return "presence unsure"
    covered = set()
    for p in row["laid"]:
        covered.update(range(p["frames"][0], p["frames"][1] + 1))
    whole = set(range(row["first"], row["last"] + 1))
    if row["laid"] and covered != whole:
        return "partly"
    if row["laid"]:
        kinds = {p["region"] for p in row["laid"]}
        if None in kinds:
            return "region not readable"
        return "as planned" if all(row["wanted"] in GRAPH_KIND.get(k, ()) for k in kinds) else "not as planned"
    return "source, said why" if row["left"] else "source, no line"


def totals(rows: list[dict]) -> dict:
    t = defaultdict(lambda: defaultdict(int))
    for r in rows:
        t[r["subject"]][r["state"]] += r["frames"]
    return {s: dict(v) for s, v in t.items()}


def frames(a, b) -> str:
    return f"{a}-{b}"


def print_report(out: dict) -> None:
    print(f"{out['video']}: the shots file covers {frames(*out['span'])}"
          + (f" of {out['source_frames']} frames in the source; nothing past {out['span'][1]} has a shot, a plan or a render here"
             if out.get("source_frames") else ""))
    print("A row a shot and a subject in it (or unsure). The state is about the files, not about any render's quality;"
          " `as planned` reads the graph's kind only (parts or whole), never face against face and hair.")
    print()
    for r in out["rows"]:
        print(f"{frames(r['first'], r['last'])}  {r['frames']:4d} f  {r['subject']:<7} wanted: {r['wanted']:<14} STATE: {r['state']}")
        if r["present"] == "unsure" or r["present_how"]:
            print(f"    here: {r['present']}; {r['present_how']} ({r['present_by']})")
        for p in r["laid"]:
            cut = "; crosses a cut" if p["crosses_a_cut"] else ""
            print(f"    laid {frames(*p['frames'])}: {Path(p['piece']).name}  region: {p['region'] or 'not read (no renders entry)'}"
                  f"  subject by {p['named_by']}{cut}")
            for said in p.get("owner_said", []):
                print(f"        the owner, {said}")
        for it in r["left"]:
            print(f"    source {frames(*it['frames'])}: {it['why']}")
        for n in r["notes"]:
            print(f"    note {n}")
        if r["shot_note"]:
            print(f"    shot note: {r['shot_note']}")
        others = [o for o in r["others"] if not o["in_table"]]
        if others:
            print(f"    other renders of this subject here, not in the table: "
                  + "; ".join(f"{o['name']} {frames(*o['frames'])} {o['region']}{' +motion' if o['motion'] else ''}" for o in others))
            for o in others:
                for said in o.get("owner_said", []):
                    print(f"        the owner on {o['name']}, {said}")
    print()
    print("Frames by state, a subject at a time (a subject's shots where they are present or unsure):")
    for label, t in out["frames_by_state"].items():
        print(f"  {label}: " + ", ".join(f"{k} {v}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))
    if out["unclaimed_pieces"]:
        print()
        print("Pieces in the table whose subject nothing names (not counted above):")
        for p in out["unclaimed_pieces"]:
            print(f"  {frames(p['first'], p['last'])} {Path(p['piece']).name}: {p['why']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--renders", help="the renders inventory read from saved graphs; without it no region is read")
    ap.add_argument("--verdicts", help="the owner's lines on renders (bench/page_answers_to_verdicts.py's record); printed under each render")
    ap.add_argument("--json", help="write the same rows as JSON here")
    a = ap.parse_args()
    read_table, read_intents = _delivery()
    shots = load_shots(a.shots)
    plan = load_plan(a.plan, shots["subjects"])
    renders = []
    if a.renders:
        d = json.load(open(a.renders))
        renders = d["renders"] if isinstance(d, dict) else d
    verdicts = [json.loads(t) for t in Path(a.verdicts).read_text().splitlines() if t.strip()] if a.verdicts else []
    out = coverage(shots, plan, read_table(a.table), read_intents(a.table), renders, verdicts)
    out["read_from"] = {"shots": a.shots, "plan": a.plan, "table": a.table, "renders": a.renders, "verdicts": a.verdicts}
    print_report(out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
