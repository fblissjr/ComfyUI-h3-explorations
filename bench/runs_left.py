#!/usr/bin/env python3
"""The runs left to make past a frame, longest first, re-cut from a delivery table: one command the card runs after each rebuild.

    <python> bench/runs_left.py --table data/<day>_<video>_delivery_tables/t_<...>.json --plan data/<video>/plan.json
                                [--after 2049] [--least 480] [--json OUT.json]

**What it is for.** The lead (2026-10-15): the queue past 2049 was a session's table (mrdrill's of 2026-10-14, cut against
one table's stamp, its driver not kept), stale the moment a piece landed. This makes it again from whichever table it is
handed, so the card reads what is left after each rebuild. Data only: it proposes; the card and the lead decide. It writes
no job.

**What it reads.** The shots and who is in each from the table's own `reading` (a subject marked present "yes", on the
shots whose frames run past `--after`, cut to the frames after it); a subject the record is `unsure` of is listed and not
run. Each shot's preview: the newest finished preview of each shot (`others_present.newest_previews`) that covers it, the
shot's own `shot_c_<first>` first, else the last listed. The runs: `propose_runs` itself on the preview's saved part
video, the part's largest piece a frame (the chain's `pieces=largest`), cut by the first `--least`, joined across
`propose_runs.BRIDGE` empty frames, as long as the shortest load or more, each with the node's own zoom on its frames.
What is laid: the table's `stands` of the same subject over the run's frames, KEPT where the owner's last choice on that
render is "keep this one", STANDING WITHOUT A KEEP otherwise. The table is the one handed in: it never picks the newest.
The restore spans inside each run: the plan's `subjects.<subject>.restore`, cut to the run (a job's head prints the plan's
spans for the subject everywhere). What enters the face mid-run: on each frame, the px of `ENTERING`'s classes inside the
box of the part's largest piece, read from the preview's saved class map, listed where at least `ENTERING_PX`.

**What it prints.** The runs with frames no piece lays, the most frames left first: the run, the frames left as spans,
the zoom of the whole run, the frames under the floor mid-run, the restore spans inside it, what enters the face's box and
where, and what the table lays on the rest. Then the runs the
table lays whole, and the shots it could not run (no preview covers them, or the preview saved no part of that subject),
each said, never dropped.

**What it cannot see.** The zoom is of the whole run: a load cut to the frames left boxes them again. The node boxes the
load's region (part plus margin), so near `ZOOM_SMALLEST` a zoom here can be a refusal at plan. The capture's floor is its
own (`capture_masked_run.starting_floor`, divided by the zoom squared since 8d1c86d2) and its grade can empty a frame this
keeps. Whether the part is the right person. A piece rendered after the table's stamp. What enters is read in the part's box,
not on the face: a hand beside the face inside its box counts, and the part model calls eye makeup and a telephone Eyeglass
(internal/claude/2026-10-14_mrdrill_scripts/eyeglass_off_the_sunglasses.jpg). The picture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import propose_runs  # noqa: E402

#: The frame the queue starts after. Reasoned: 0-2049 is the stretch the card filled through 2026-10-14 by its own lists
#: (`bench/piece_gaps.py`); past it is this queue's (the lead's brief of 2026-10-15).
AFTER = 2049
#: The floor the runs are cut by, px of the canvas. Inherited: the floor the lead approved runs at on 604-851's far stretches
#: (mrcircle's cut of 2026-10-14) and mrdrill's queue of that day was cut at; a starting value, not the capture's own floor.
LEAST = 480
#: What is listed as entering the face, by the part model's classes (`sapiens2_parts.CLASS_NAMES`). Reasoned: the card's case of
#: 2026-10-15, her sunglasses and her hands from 3060 on 2823-3097 (the lead's relay); a place to look, not a finding.
ENTERING = {"Eyeglass": ("Eyeglass",), "hands": ("Left_Hand", "Right_Hand")}
#: The px of one of those inside the part's box before a frame is listed. A STARTING VALUE, reasoned: a quarter of one 16 px cell;
#: under it is a graze at the box's edge.
ENTERING_PX = 64


def laid_on(reading: list[dict], subject: str, first: int, last: int) -> list[dict]:
    """The table's pieces of the subject on first-last: each render's frames there, kept or standing without a keep, and the frames
    inside them its load left as the original (`leaves_as_the_original`: laid, and the source's all the same)."""
    out = []
    for e in reading:
        if e["subject"] != subject:
            continue
        for s in e.get("stands") or []:
            kept = ((s.get("the_owners_last_choice") or {}).get("said") == "keep this one")
            for a, b in s["frames"]:
                x, y = max(a, first), min(b, last)
                if x <= y:
                    leaves = [f for c, d in s.get("leaves_as_the_original") or [] for f in range(max(c, x), min(d, y) + 1)]
                    out.append({"render": s["render"], "frames": [x, y], "kept": kept, "leaves_as_the_original": leaves})
    return sorted(out, key=lambda p: p["frames"])


def spans(frames) -> str:
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in propose_runs.spans_of(sorted(frames))) or "none"


def shots_after(reading: list[dict], after: int) -> tuple[list[dict], list[dict], list[dict]]:
    """(each shot and subject present "yes" past `after`, cut to the frames after it, with any `nothing_to_make` look; the same for
    `unsure`; and for any `present` but yes, unsure and no, said with it), by frame."""
    seen, run, unsure, other = set(), [], [], []
    for e in sorted(reading, key=lambda e: (e["first"], e["subject"])):
        key = (e["first"], e["last"], e["subject"], e.get("present"))
        if e["last"] <= after or key in seen:
            continue
        seen.add(key)
        row = {"shot": [e["first"], e["last"]], "first": max(e["first"], after + 1), "last": e["last"], "subject": e["subject"],
               "nothing_to_make": e.get("nothing_to_make")}
        if e.get("present") == "yes":
            run.append(row)
        elif e.get("present") == "unsure":
            unsure.append(row)
        elif e.get("present") != "no":
            other.append({**row, "present": e.get("present")})
    return run, unsure, other


def previews_for(shot: list[int], previews: list[dict]) -> list[dict]:
    """The previews that cover the shot, in the order they are tried: its own `shot_c_<first>` first, then the others, the last listed
    first."""
    covering = [p for p in previews if p["first"] <= shot[0] and shot[1] <= p["last"]][::-1]
    own = [p for p in covering if p["name"].rstrip("abcdefghijklmnopqrstuvwxyz") == f"shot_c_{shot[0]:04d}"]
    return own + [p for p in covering if p not in own]


def restore_in(restores: list[dict], first: int, last: int) -> list[dict]:
    """The plan's restore spans of a subject cut to first-last."""
    return [{"classes": r["classes"], "frames": [max(r["first"], first), min(r["last"], last)]} for r in restores
            if r["first"] <= last and r["last"] >= first]


def entering_in(entering, first: int, a: int, b: int) -> dict | str:
    """Per name, the spans of frames a-b where it has `ENTERING_PX` or more inside the part's box, with the most px; or why none."""
    if isinstance(entering, str):
        return entering
    out = {}
    for name, px in entering.items():
        on = [f for f in range(a, b + 1) if px[f - first] >= ENTERING_PX]
        if on:
            out[name] = {"frames": on, "most_px": int(max(px[f - first] for f in on))}
    return out


def queue(table: dict, previews: list[dict], part_of, least: int, after: int, bridge: int, least_frames: int, size, zoom_box,
          plan: dict | None = None) -> dict:
    """The runs past `after` from the table and the previews. `part_of(preview, subject, first, last)` gives {"px", "boxes"} a frame of
    the part's largest piece and "entering" ({name: px a frame inside the part's box}, or why it is not known), or a string saying
    why there is no part."""
    reading = table["reading"]
    todo, unsure, other = shots_after(reading, after)
    runs, not_run, no_run = [], [], []
    for s in todo:
        tried, got, p = [], None, None
        for p in previews_for(s["shot"], previews):
            got = part_of(p, s["subject"], s["first"], s["last"])
            if not isinstance(got, str):
                break
            tried.append(f"{p['name']}: {got}")
        if got is None or isinstance(got, str):
            not_run.append({**s, "why": "; ".join(tried) or "no finished preview covers the shot"})
            continue
        px, boxes = got["px"], got["boxes"]
        restores = (((plan or {}).get("subjects") or {}).get(s["subject"]) or {}).get("restore") or []
        on = tuple(p.get("canvas") or size)
        proposal = propose_runs.propose(np.asarray(px), np.asarray(boxes), s["first"], [least], bridge, least_frames, on, zoom_box)
        if not proposal["runs"]:
            no_run.append({**s, "preview": p["name"], "px_median": int(np.median(px)), "px_most": int(np.max(px)),
                           "frames_clearing": int((np.asarray(px) >= least).sum()), "frames": len(px)})
        for r in proposal["runs"]:
            a, b = r["frames"]
            laid = laid_on(reading, s["subject"], a, b)
            covered = {f for x in laid for f in range(x["frames"][0], x["frames"][1] + 1) if f not in x["leaves_as_the_original"]}
            left = [f for f in range(a, b + 1) if f not in covered]
            runs.append({"shot": s["shot"], "subject": s["subject"], "preview": p["name"], "run": [a, b], "count": b - a + 1,
                         "left": left, "zoom": r["zoom"], "mid_run_under_the_floor": r["mid_run_under_the_floor"], "laid": laid,
                         "px_median": r["px_median"], "px_least": r["px_least"], "restore": restore_in(restores, a, b),
                         "entering": entering_in(got.get("entering", "not read"), s["first"], a, b)})
    runs.sort(key=lambda r: (-len(r["left"]), r["run"][0], r["subject"]))
    return {"table_made": table.get("made"), "after": after, "least": least, "bridge": bridge, "least_frames": least_frames,
            "to_make": [r for r in runs if r["left"]], "laid_whole": [r for r in runs if not r["left"]], "not_run": not_run, "no_run": no_run,
            "unsure": unsure, "present_otherwise": other}


def zoom_said(z: dict) -> str:
    return f"{z['zooms']}x ({z['set_by']})" if z.get("zooms") else f"not zoomed ({z.get('why')})"


def laid_said(laid: list[dict]) -> str:
    return "; ".join(f"{'KEPT' if x['kept'] else 'STANDING WITHOUT A KEEP'} {x['render']} {x['frames'][0]}-{x['frames'][1]}"
                     + (f" (leaves {spans(x['leaves_as_the_original'])} as the original)" if x["leaves_as_the_original"] else "") for x in laid) or "nothing"


def left_said(left: list[int], least_frames: int) -> str:
    return ", ".join(f"{a}-{b}" + (f" ({b - a + 1}, under the shortest load)" if b - a + 1 < least_frames else "")
                     for a, b in propose_runs.spans_of(left)) + f" ({len(left)})"


def entering_said(e) -> str:
    if isinstance(e, str):
        return e
    return "; ".join(f"{k} {spans(v['frames'])} (most {v['most_px']} px)" for k, v in e.items()) or "nothing"


def print_queue(q: dict, table_path: str) -> None:
    print(f"runs past {q['after']} from {table_path} ({str(q['table_made']).split(' from ')[0]}), the part's largest piece cut at {q['least']} px, joined across "
          f"{q['bridge']} empty frame(s), {q['least_frames']} frames or more:")
    print(f"TO MAKE ({len(q['to_make'])} runs, the most frames left first):")
    for i, r in enumerate(q["to_make"], 1):
        print(f"  {i:>3}. {r['subject']:<7} run {r['run'][0]}-{r['run'][1]} ({r['count']}) of shot {r['shot'][0]}-{r['shot'][1]} on {r['preview']}: "
              f"LEFT {left_said(r['left'], q['least_frames'])}; zoom of the run {zoom_said(r['zoom'])}; px median {r['px_median']}, least {r['px_least']}"
              + (f"; under the floor mid-run {spans(r['mid_run_under_the_floor'])}" if r["mid_run_under_the_floor"] else "")
              + "; restore spans inside: " + (", ".join(f"{'+'.join(x['classes'])} {x['frames'][0]}-{x['frames'][1]}" for x in r["restore"]) or "none")
              + f"; entering the face's box: {entering_said(r['entering'])}"
              + f"; laid: {laid_said(r['laid'])}")
    print(f"LAID WHOLE by the table ({len(q['laid_whole'])} runs):")
    for r in q["laid_whole"]:
        print(f"       {r['subject']:<7} run {r['run'][0]}-{r['run'][1]} ({r['count']}): {laid_said(r['laid'])}")
    print(f"NOT RUN ({len(q['not_run'])} shot-subjects):")
    for s in q["not_run"]:
        print(f"       {s['subject']:<7} {s['first']}-{s['last']} of shot {s['shot'][0]}-{s['shot'][1]}: {s['why']}" + (f" ({s['preview']})" if s.get("preview") else ""))
    print(f"NO RUN at {q['least']} px ({len(q['no_run'])} shot-subjects):")
    for s in q["no_run"]:
        print(f"       {s['subject']:<7} {s['first']}-{s['last']} of shot {s['shot'][0]}-{s['shot'][1]} on {s['preview']}: {s['frames_clearing']} of "
              f"{s['frames']} frames clear it, px median {s['px_median']}, most {s['px_most']}"
              + (f"; NOTHING TO MAKE by a look: {' | '.join(s['nothing_to_make'])}" if s.get("nothing_to_make") else ""))
    for s in q["present_otherwise"]:
        print(f"       {s['subject']:<7} {s['first']}-{s['last']}: the record says present {s['present']!r}, not run")
    if q["unsure"]:
        print("THE RECORD IS UNSURE, not run: " + "; ".join(f"{s['subject']} {s['first']}-{s['last']}" for s in q["unsure"]))
    print(f"entering the face's box: {', '.join(f'{k} ({chr(43).join(v)})' for k, v in ENTERING.items())} at {ENTERING_PX} px or more "
          "inside the box of the part's largest piece (a starting value)")
    print("CANNOT SEE: the zoom is the whole run's; near ZOOM_SMALLEST a zoom can be a refusal at plan; the capture's own floor and grade; "
          "whether the part is the right person; a piece rendered after the table's stamp; what enters is read in the part's box, not on "
          "the face, and Eyeglass is also eye makeup or a telephone; the picture")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--table", required=True, help="a delivery table's .json (bench/delivery_table.py --json), the newest")
    ap.add_argument("--plan", required=True, help="the video's plan (data/<video>/plan.json): its restore spans")
    ap.add_argument("--after", type=int, default=AFTER, help=f"the frame the queue starts after ({AFTER})")
    ap.add_argument("--least", type=int, default=LEAST, help=f"the floor the runs are cut by, px of the canvas ({LEAST}, a starting value)")
    ap.add_argument("--data", default=str(HERE.parent / "data"), help="where the previews' job folders are")
    ap.add_argument("--share", default=os.environ.get("H3_OUTPUT_DIR"), help="the server's output folder the previews wrote to (default: $H3_OUTPUT_DIR)")
    ap.add_argument("--json", help="also write the queue as JSON here (a new file)")
    a = ap.parse_args(argv)
    if a.json and Path(a.json).exists():
        raise SystemExit(f"refused: {a.json} is there; a kept output is not replaced")
    if not a.share:
        ap.error("--share is required when H3_OUTPUT_DIR is not set in the environment")
    import capture_masked_run as cap
    import masked_job
    import others_present
    from _lib.frames import read_mask
    previews = []
    for spec in others_present.newest_previews(Path(a.data), Path(a.share)):
        folder = Path(spec.split("=", 1)[0]).parent
        job = json.loads((folder / "job.json").read_text())["job"]
        previews.append({"name": job["name"], "folder": folder, "first": int(job["first"]), "last": int(job["first"]) + int(job["frames"]) - 1,
                         "canvas": job["canvas"], "files": json.loads((folder / "preview.queued.json").read_text()).get("files") or []})
    size = None                                                     # each preview's own canvas (`canvas`), read with its part
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    wanted = {k: [names.index(c) for c in v] for k, v in ENTERING.items()}

    def part_of(p, subject, first, last):
        found = [f for f in p["files"] if f"_{subject}_parts_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            return f"the preview saved {len(found)} part video(s) of {subject} where one is meant"
        size = tuple(p["canvas"])
        mask = read_mask(str(Path(a.share) / found[0]), size, p["first"], first, last - first + 1)[0]
        px, boxes = propose_runs.part_frames(mask, True)
        maps = [f for f in p["files"] if f"_{subject}_classes_" in Path(f).name and f.endswith(".mkv")]
        if len(maps) != 1:
            return {"px": px, "boxes": boxes, "entering": f"not known: the preview saved {len(maps)} class map(s) of {subject}"}
        classes = cap.read_classes(str(Path(a.share) / maps[0]), size, p["first"], first, last - first + 1)
        entering = {k: np.zeros(len(px), int) for k in ENTERING}
        for f, (x0, y0, x1, y1) in enumerate(boxes):
            if x0 < 0:
                continue
            cut = classes[f, y0:y1, x0:x1]
            for k, ids in wanted.items():
                entering[k][f] = int(np.isin(cut, ids).sum())
        return {"px": px, "boxes": boxes, "entering": entering}

    table = json.loads(Path(a.table).read_text())
    q = queue(table, previews, part_of, a.least, a.after, propose_runs.BRIDGE, int(masked_job.shortest_load()[1]), size,
              propose_runs.node_zoom().zoom_box, json.loads(Path(a.plan).read_text()))
    print_queue(q, a.table)
    if a.json:
        Path(a.json).write_text(json.dumps(q, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
