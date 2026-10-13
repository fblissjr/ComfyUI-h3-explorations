#!/usr/bin/env python3
"""Where a masked video has somebody to replace and how much of them is there: the shots with none of the subjects, and for each shot and subject present the frames their part is on, read from a preview's own masks.

    <python> bench/shot_parts.py --shots data/<stem>/shots.json --plan data/<stem>/plan.json --table TABLE.txt \\
        --preview JOB.json=TABLES_DIR [--preview ...] [--ledger LEDGER.json] [--verdicts VERDICTS.jsonl] \\
        [--answers data/<stem>/poc/answers.jsonl] --json OUT.json --md OUT.md

**What it buys.** The owner asked (2026-10-13) for two lists of the whole video, regenerated and never typed: the
shots where none of the subjects is, where nothing is ever rendered; and the shots where a subject is present and
nothing of theirs can be replaced (turned away, hidden, too small). The first was counted by hand from the shots
record that day. The second was known one shot at a time, after a load had been built on it: a preview saves the
part to replace on every frame, and nothing read it across a video. A session taking shots to build from needs
both in one table, with what stands there already.

**What it reads, each a named file handed in; nothing is found by a key.** `--shots`, `--plan`, `--table`,
`--ledger` (or the older `--renders`) and `--verdicts` as `bench/shot_coverage.py` reads them (its readers and its `coverage()` are imported,
so "what stands there" has one definition). `--preview JOB.json=TABLES_DIR` as `bench/who_page.py` takes a
preview: the job file and the folder its trackers saved into. Where two previews handed in cover a shot and track
the subject, THE LAST LISTED is read and the row names it. `--answers` is the page's answers file, for the
owner's typed words on a shot.

**List 1, nobody.** The shots on which every subject is `no` in the shots record, with their frames. A subject
who is `unsure` on a shot is under a heading of their own and in neither list.

**List 2, a row a shot and a subject present or unsure.**

- the frames; the plan's region, its parts and its keep-outs for the subject (printed, so nobody looks them up);
- the preview read, and whether its tracker was SEEDED (a person's answer said who) or PICKED (the largest
  person on a frame, which may be somebody else: every figure after it is then somebody else's); the frames the
  track is on, since no part on a frame the tracker lost her is another fact than a face turned away;
- THE PART as the preview saved it (`<preview>_<subject>_parts_*.mkv`, what a load would carry): the frames it is
  there, the frames at or over the floor, the frames under it and the frames it is not there, as counts and as
  runs of source frames both ways; its size in pixels on the frames it is there, least, median and most; and
  how many times the floor the median is. Where the plan gives the subject more than one part, EACH PART APART
  as well, from the preview's class map (`sapiens2_parts.PARTS`): hair that shows with no face reads as that;
- the floor: `capture_masked_run.PART_LEAST_PX`, a STARTING VALUE by its own comment, printed beside the
  figures and decided by nobody here. A part that is there and small is a shot for the zoomed load, and reads so
  in the figures;
- one window or more: the window the builder would render the shot in (`masked_job_graph.window_for`: for a shot
  one window holds, the shortest length of the model's video grid that holds it), and whether the shot is longer
  than the planner's longest (`masked_job.chain_lengths`);
- what stands there: the coverage tool's own state for the row, the pieces the table lays and the other
  renders on those frames, each with the owner's lines on it;
- the owner's typed words on the shot, every line, whole; the shot's own note; and the plan's dated notes on any
  stretch of it (`departures`, each with who wrote it: where a person's or a session's reading of a stretch lives,
  such as "no face on these frames, no load", which no mask can say for a shot no preview was made of).

**No verdict.** "The part is on no frame of the preview" is a fact and has a heading. Anything of degree has
none: the other rows are sorted by the frames the part is there, fewest first, and the reading is left to
whoever takes a shot from the table. A row with no preview handed in says so and carries what stands there alone.

**What it cannot see.** A part model's miss reads as "not there". A wrong pick reads as somebody else's part.
The tracker's own cuts inside a shot of the record. Sizes are at the preview's canvas, before any zoom. Whether
any render is good. A preview that covers part of a shot is read on the frames it covers, and the row says how many.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
import capture_masked_run as cap  # noqa: E402
import masked_job  # noqa: E402
import masked_job_graph  # noqa: E402
import shot_coverage  # noqa: E402
import who_page  # noqa: E402
from _lib.frames import probe, stream  # noqa: E402

#: Frames of a class map read at a time. Reasoned: a few dozen megabytes at the lane's canvas, whatever a preview's length.
CHUNK = 64
CANNOT_SEE = ("a part model's miss reads as 'not there'; a PICKED tracker may hold somebody else, and every figure of the row is then theirs; the tracker's own "
              "cuts inside a shot; sizes are at the preview's canvas, before any zoom; whether any render is good")


def runs(flags, first: int) -> list[list[int]]:
    """The runs of true in `flags`, as [first, last] source frames; a gap of one frame ends a run."""
    out: list[list[int]] = []
    for n, on in enumerate(flags):
        if on:
            if out and out[-1][1] == first + n - 1:
                out[-1][1] = first + n
            else:
                out.append([first + n, first + n])
    return out


def figures(px: np.ndarray, first: int, floor: int) -> dict:
    """What a part's size on each frame says: where it is there, where it is at or over the floor, and how large."""
    there = px > 0
    out = {"frames": int(len(px)), "frames_there": int(there.sum()), "frames_at_or_over_the_floor": int((px >= floor).sum()),
           "frames_under_the_floor": int((there & (px < floor)).sum()), "frames_not_there": int((~there).sum()),
           "runs_there": runs(there, first), "runs_not_there": runs(~there, first), "runs_at_or_over_the_floor": runs(px >= floor, first),
           "px": None, "median_over_the_floor": None}
    if there.any():
        out["px"] = {"least": int(px[there].min()), "median": int(np.median(px[there])), "most": int(px[there].max())}
        out["median_over_the_floor"] = round(out["px"]["median"] / floor, 2) if floor else None
    return out


def previews(specs: list[str]) -> list[dict]:
    """Each preview handed in: its job's name, frames and subjects, and for each subject the table and the three mask
    videos its trackers saved, by the names `bench/shot_to_load.py preview` gives them. A file that is not there is None."""
    out = []
    for spec in specs:
        job_path, _, folder = spec.partition("=")
        if not folder:
            raise SystemExit(f"--preview {spec!r}: write JOB.json=TABLES_DIR")
        job = json.load(open(job_path))["job"]
        subjects = {}
        for s in job["subjects"]:
            stem = Path(folder) / f"{job['name']}_{s['label']}"
            there = lambda p: p if p.is_file() else None                                                                   # noqa: E731
            subjects[s["label"]] = {"table": there(Path(f"{stem}_shots_00001_shots.json")), "seeds": s.get("seeds"),
                                    **{kind: there(Path(f"{stem}_{kind}_00001.mkv")) for kind in ("track", "parts", "classes")}}
        out.append({"name": job["name"], "spec": spec, "first": int(job["first"]), "frames": int(job["frames"]),
                    "last": int(job["first"]) + int(job["frames"]) - 1, "subjects": subjects})
    return out


def measured(pv: dict, label: str, parts: list[str], names: tuple, menu: dict) -> dict:
    """Per frame of a preview, a subject's track, saved part and each plan part in pixels, at the saved videos' own size."""
    files = pv["subjects"][label]
    out = {"size": None, "track_px": None, "part_px": None, "apart_px": {}, "missing": [k for k in ("track", "parts", "classes", "table") if not files[k]]}

    def counted(path) -> np.ndarray:
        w, h, _ = probe(str(path))
        out["size"] = [w, h]
        px = np.zeros(pv["frames"], np.int64)
        for n, frame in enumerate(stream(str(path), (w, h), 0, pv["frames"])):
            px[n] = int((frame > 127).sum())
        return px
    if files["track"]:
        out["track_px"] = counted(files["track"])
    if files["parts"]:
        out["part_px"] = counted(files["parts"])
    if files["classes"] and len(parts) > 1:
        w, h, _ = probe(str(files["classes"]))
        wanted = {p: [names.index(c) for c in menu[p]] for p in parts}
        out["apart_px"] = {p: np.zeros(pv["frames"], np.int64) for p in parts}
        for at in range(0, pv["frames"], CHUNK):
            classes = cap.read_classes(str(files["classes"]), (w, h), 0, at, min(CHUNK, pv["frames"] - at))
            for p, indices in wanted.items():
                out["apart_px"][p][at:at + len(classes)] = np.isin(classes, indices).reshape(len(classes), -1).sum(axis=1)
    table = json.loads(files["table"].read_text()) if files["table"] else None
    out["seeded"] = None if table is None else bool(table.get("pick_seed")) or any("seeded" in str(s["subject"].get("state")) for s in table["shots"])
    out["cuts"] = [] if table is None else [[pv["first"] + int(s["first_frame"]), pv["first"] + int(s["last_frame"]), str(s["subject"].get("state"))] for s in table["shots"]]
    return out


def lists(shots: dict, plan: dict, covered: dict, made: list[dict], answers: list[dict], measure=measured) -> dict:
    """The two lists. `covered` is `shot_coverage.coverage()`'s output; `measure` is handed in so a check can count it."""
    sp = cap._pack("sapiens2_parts")
    floor, windows = int(cap.PART_LEAST_PX), masked_job.chain_lengths()
    nobody = [{"first": s["first"], "last": s["last"], "frames": s["last"] - s["first"] + 1} for s in shots["shots"]
              if all(w["present"] == "no" for w in s["who"].values())]
    unsure = [{"first": s["first"], "last": s["last"], "frames": s["last"] - s["first"] + 1, "subject": label, "how": w.get("how", ""), "by": w.get("by", "")}
              for s in shots["shots"] for label, w in s["who"].items() if w["present"] == "unsure"]
    stands = {(r["first"], r["last"], r["subject"]): r for r in covered["rows"]}
    cache: dict = {}
    rows = []
    for s in shots["shots"]:
        for label in shots["subjects"]:
            if s["who"][label]["present"] == "no":
                continue
            wanted, there = plan["subjects"][label], stands[(s["first"], s["last"], label)]
            n = s["last"] - s["first"] + 1
            row = {"first": s["first"], "last": s["last"], "frames": n, "subject": label, "present": s["who"][label]["present"],
                   "region": wanted["region"], "parts": list(wanted.get("parts") or []), "keep_outs": list(wanted.get("keep_outs") or []),
                   "window": {"the_builder_would_take": masked_job_graph.window_for(n, windows), "the_planners_lengths": list(windows),
                              "one_window": n <= max(windows), "frames_over_the_longest": max(n - max(windows), 0)},
                   "stands": {"state": there["state"], "laid": there["laid"], "left_as_the_source": there["left"],
                              "other_renders": [o for o in there["others"] if not o["in_table"]]},
                   "the_owners_words": who_page.owner_words(answers, s["first"], s["last"]), "shot_note": s.get("note", ""),
                   "the_plans_notes": there["notes"],
                   "preview": None, "tracker": None, "part": None, "parts_apart": {}}
            mine = [pv for pv in made if label in pv["subjects"] and not (pv["last"] < s["first"] or pv["first"] > s["last"])]
            if not mine:
                row["no_preview"] = ("no preview handed in covers this shot and tracks this subject" if not any(
                    not (pv["last"] < s["first"] or pv["first"] > s["last"]) for pv in made) else "a preview handed in covers this shot and does not track this subject")
                rows.append(row)
                continue
            pv = mine[-1]
            key = (pv["spec"], label)
            if key not in cache:
                cache[key] = measure(pv, label, row["parts"], sp.CLASS_NAMES, sp.PARTS)
            m = cache[key]
            lo, hi = max(s["first"], pv["first"]), min(s["last"], pv["last"])
            cut = slice(lo - pv["first"], hi - pv["first"] + 1)
            row["preview"] = {"name": pv["name"], "handed_in_as": pv["spec"], "frames": [pv["first"], pv["last"]], "covers": [lo, hi], "frames_covered": hi - lo + 1,
                              "others_handed_in": [p["name"] for p in mine[:-1]], "size": m["size"], "files_not_there": m["missing"]}
            if m["track_px"] is not None:
                on = m["track_px"][cut] > 0
                row["tracker"] = {"seeded": m["seeded"], "frames_on": int(on.sum()), "frames_off": int((~on).sum()), "runs_off": runs(~on, lo),
                                  "cuts_in_the_shot": [c for c in m["cuts"] if not (c[1] < lo or c[0] > hi)]}
            if m["part_px"] is not None:
                row["part"] = {"is": "the part as the preview saved it, what a load would carry", "floor_px": floor,
                               "floor_is": "capture_masked_run.PART_LEAST_PX, a starting value by its own comment", **figures(m["part_px"][cut], lo, floor)}
            row["parts_apart"] = {p: {"is": f"the classes of `{p}` in the preview's class map: " + ", ".join(sp.PARTS[p]), **figures(px[cut], lo, floor)}
                                  for p, px in m["apart_px"].items()}
            rows.append(row)
    return {"video": shots["video"], "span": shots["span"], "nobody": nobody, "unsure": unsure, "rows": rows, "cannot_see": CANNOT_SEE,
            "floor_px": floor, "the_planners_lengths": list(windows)}


def spans(rows: list[list[int]], most: int = 8) -> str:
    text = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in rows[:most])
    return (text or "none") + (f" and {len(rows) - most} more run(s) (all in the .json)" if len(rows) > most else "")


def said(f: dict) -> str:
    px = f["px"]
    return (f"there on {f['frames_there']} of {f['frames']} frames (at or over the floor {f['frames_at_or_over_the_floor']}, under it {f['frames_under_the_floor']}), "
            f"not there on {f['frames_not_there']}" + (f"; px least {px['least']}, median {px['median']}, most {px['most']}; the median is {f['median_over_the_floor']} "
                                                        f"times the floor" if px else "")
            + f"\n      there: {spans(f['runs_there'])}\n      not there: {spans(f['runs_not_there'])}")


def row_text(r: dict) -> list[str]:
    w = r["window"]
    out = [f"### {r['first']}-{r['last']}, {r['subject']}" + (" (UNSURE in the shots record)" if r["present"] == "unsure" else ""),
           f"- {r['frames']} frames; the plan: {r['region']} ({', '.join(r['parts']) or 'no parts named'}); keep-outs: {', '.join(r['keep_outs']) or 'none'}",
           (f"- one window: the builder would take {w['the_builder_would_take']} frames" if w["one_window"] else
            f"- MORE THAN ONE WINDOW: {w['frames_over_the_longest']} frames over the longest ({max(w['the_planners_lengths'])})")]
    if r["preview"] is None:
        out.append(f"- NO PREVIEW READ: {r['no_preview']}")
    else:
        p, t = r["preview"], r["tracker"]
        out.append(f"- preview {p['name']} ({p['frames'][0]}-{p['frames'][1]}), read on {p['frames_covered']} of the shot's {r['frames']} frames"
                   + (f"; also handed in and not read: {', '.join(p['others_handed_in'])}" if p["others_handed_in"] else "")
                   + (f"; FILES NOT THERE: {', '.join(p['files_not_there'])}" if p["files_not_there"] else ""))
        if t:
            out.append(f"- the tracker: {'SEEDED from a person' + chr(39) + 's answer' if t['seeded'] else 'PICKED, which may be somebody else' if t['seeded'] is False else 'its table was not there'}; "
                       f"on {t['frames_on']} frames, off {t['frames_off']}" + (f" ({spans(t['runs_off'])})" if t["frames_off"] else "") + f"; its own cuts in the shot: {len(t['cuts_in_the_shot'])}")
        if r["part"]:
            out.append(f"- THE PART as saved: {said(r['part'])}")
        for name, f in r["parts_apart"].items():
            out.append(f"- `{name}` apart: {said(f)}")
    st = r["stands"]
    out.append(f"- what stands there: {st['state']}" + "".join(
        f"\n      laid {x['frames'][0]}-{x['frames'][1]}: {x['piece']} (region: {x.get('region_exact') or x['region'] or 'not read'})" + "".join(f"\n          the owner: {line}" for line in x["owner_said"]) for x in st["laid"])
        + "".join(f"\n      left as the source {x['frames'][0]}-{x['frames'][1]}: {x['why']}" for x in st["left_as_the_source"])
        + "".join(f"\n      another render, not in the table: {x['name']} ({x['frames'][0]}-{x['frames'][1]})" + "".join(f"\n          the owner: {line}" for line in x["owner_said"])
                  for x in st["other_renders"]))
    for line in r["the_owners_words"]:
        out.append(f"- the owner's words, frame {line['frame']} ({line['at']}): {line['words']}")
    if r["shot_note"]:
        out.append(f"- the shot's note: {r['shot_note']}")
    for note in r["the_plans_notes"]:
        out.append(f"- the plan's note on a stretch of it: {note}")
    return out + [""]


def as_md(out: dict) -> str:
    rows = out["rows"]
    none = [r for r in rows if r["part"] and not r["part"]["frames_there"]]
    some = sorted((r for r in rows if r["part"] and r["part"]["frames_there"]), key=lambda r: (r["part"]["frames_there"], r["first"]))
    unread = [r for r in rows if not r["part"]]
    lines = [f"# {out['video']}: where there is somebody to replace, and how much of them", "",
             f"Made {out['made_at']} by bench/shot_parts.py at {out['tool_commit']}. Regenerated, never typed. Read from:"]
    lines += [f"- {k}: {v}" for k, v in out["read_from"].items() if k != "previews"] + [f"- {len(out['read_from']['previews'])} preview(s), each named on the row that read it", ""]
    lines += [f"The floor is {out['floor_px']} px (capture_masked_run.PART_LEAST_PX), a starting value: it is printed beside the figures and decides nothing here.",
              f"CANNOT SEE: {out['cannot_see']}.", "",
              f"## List 1. Nobody: {len(out['nobody'])} shot(s), {sum(s['frames'] for s in out['nobody'])} frames, of {len(set((r['first'], r['last']) for r in rows)) + len(out['nobody'])} shots", "",
              "Every subject is `no` in the shots record. Nothing is rendered there.", ""]
    lines += [f"- {s['first']}-{s['last']} ({s['frames']} frames)" for s in out["nobody"]] + [""]
    lines += [f"### Unsure, in neither list: {len(out['unsure'])}", ""] + [f"- {u['first']}-{u['last']}, {u['subject']}: {u['how']} ({u['by']})" for u in out["unsure"]] + [""]
    lines += [f"## List 2. A subject is there: {len(rows)} row(s), a shot and a subject each", "",
              f"### The part is on NO frame of the preview: {len(none)} row(s)", ""]
    for r in none:
        lines += row_text(r)
    lines += [f"### The part is on some frame: {len(some)} row(s), fewest frames first, no line drawn", ""]
    for r in some:
        lines += row_text(r)
    lines += [f"### No part read: {len(unread)} row(s) (no preview handed in, or its part file is not there)", ""]
    for r in unread:
        lines += row_text(r)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--table", required=True, help="the delivery table, as bench/shot_coverage.py reads it")
    ap.add_argument("--ledger", help="the render ledger (bench/render_ledger.py --json), as the coverage tool reads it: the one inventory, made new with every table")
    ap.add_argument("--renders", help="the older renders inventory, as the coverage tool reads it; not with --ledger")
    ap.add_argument("--verdicts", help="the owner's lines on renders, as the coverage tool reads them")
    ap.add_argument("--answers", help="the page's answers file, for the owner's typed words on a shot")
    ap.add_argument("--preview", action="append", default=[], metavar="JOB.json=TABLES_DIR", help="a preview and the folder its trackers saved into; repeatable, "
                    "and where two cover a shot and track the subject the LAST listed is read")
    ap.add_argument("--json", required=True, help="the table a session takes its shots from")
    ap.add_argument("--md", required=True, help="the same, to be read")
    a = ap.parse_args()
    read_table, read_intents = shot_coverage._delivery()
    shots = shot_coverage.load_shots(a.shots)
    plan = shot_coverage.load_plan(a.plan, shots["subjects"])
    renders = shot_coverage.read_renders(a.ledger, a.renders)
    verdicts = [json.loads(t) for t in Path(a.verdicts).read_text().splitlines() if t.strip()] if a.verdicts else []
    answers = [json.loads(t) for t in Path(a.answers).read_text().splitlines() if t.strip()] if a.answers else []
    covered = shot_coverage.coverage(shots, plan, read_table(a.table), read_intents(a.table), renders, verdicts)
    out = lists(shots, plan, covered, previews(a.preview), answers)
    out["made_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["tool_commit"] = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "not read"
    out["read_from"] = {"shots": a.shots, "plan": a.plan, "table": a.table, "ledger": a.ledger, "renders": a.renders, "verdicts": a.verdicts, "answers": a.answers, "previews": a.preview}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.md).write_text(as_md(out))
    rows = out["rows"]
    print(f"nobody: {len(out['nobody'])} shot(s), {sum(s['frames'] for s in out['nobody'])} frames; unsure: {len(out['unsure'])}; rows: {len(rows)}; "
          f"the part on no frame: {sum(1 for r in rows if r['part'] and not r['part']['frames_there'])}; on some: {sum(1 for r in rows if r['part'] and r['part']['frames_there'])}; "
          f"no part read: {sum(1 for r in rows if not r['part'])}\nwritten: {a.json}\nwritten: {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
