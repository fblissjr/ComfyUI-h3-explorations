#!/usr/bin/env python3
"""A subject's look, declared once in the plan, made into the region a load regenerates; and the plan-time check
that reads the node's own plan back against it.

    python bench/look_rooms.py room JOB_FILE --plan data/<video>/plan.json --by SESSION [--dry]
    python bench/look_rooms.py check JOB_FILE

**What it is for.** A load's region was its subject's part grown by a fixed margin, less the keep-outs, and nothing
said what of the original must change, what must stay and what is new. Three faults of one family came of it on
2026-10-14: her long hair handed back at the shoulders by a keep-out, his hairline flickering where the region's
edge ran through his curls, his beard left by a region that missed it. The design read and its measurements are
internal/claude/2026-10-14_mrpre_look_spec_read.md; the lead ruled it that day.

**The plan's `look`** (on a subject of `data/<video>/plan.json`, with `look_by`): `change`, a list of rooms the
model must regenerate though they are not the subject's part, each `{"room": KIND, ...}` with its own settings and
`by`. A room is joined to the subject's part, so the region is grown from both: a cell holding the subject's own
mask is never given back to a keep-out (`video_mask.window`), and the composite keeps the render where the subject's
mask was. One kind is built:

- `hairline`: for a subject whose hair is KEPT and whose region is the face. His hair over his face and a band of
  his hair at the hairline, each held over the frames round it, so the region's edge sits in his curls and holds
  still instead of running through the hairline the class map flips from frame to frame. A frame's room: the pixels
  of his track that are Hair on some frame within `hold_frames` either side and Face_Neck on some frame within it
  (the fringe the class map flips), and the Hair within `band_px` of Face_Neck on any frame within it (the band),
  from his own class map. Its settings and their provenance: `h3_config.LOOK_HAIRLINE`.

A kind it does not know is refused by name; `add` (a new person's hair from their still) and a `change` room
carried from large frames (his beard) are designed in the read and not built.

**What `room` writes.** The subject's part with the rooms joined, on the preview's frames (frame 0 the preview's
first), as `<preview>_<subject>_part_with_look_<load>.mkv` in `bench/shot_to_load.py`'s parts folder, the rooms
alone beside it (`..._look_rooms_<load>.mkv`, what `check` holds the region to) and a record. The part joined is
the load's own (`parts_file` on its subject, as the job step held it) where the job names one, else the preview's
saved part. The job file is then given `parts_file` (the chain hands the capture that file as the part,
`load_chain.mask_specs`) and `look_rooms` (the record), after a copy of it is kept beside it. A load whose plan has
been built is refused: its job file is held byte-equal from then on.

**What `check` reads.** The node's own planned region, each window's `<load>_window_N_planned_region.npz` as the
plan step wrote it, its token cells laid back through the window's zoom box onto the load's frames: never the
capture's region, which reads a zoomed run on the canvas's 16 px grid (on third_fp_4479b it covered 1.5 to 1.7
times the node's own). RED, naming the frames: a pixel of a room outside the node's region. Data, never red: the
region's cells; for a hairline room, how many px of his hair switch between regenerated and kept from one frame to
the next, kept cells three or four of whose neighbours are regenerated (notches), and how far above his hairline the
region reaches; the px of every other subject's track inside the region.

WHAT IT CANNOT SEE. Whether the model draws what the region allows: on third_fp_4479b it drew a beard inside the
region. Whether a room is the right size: `band_px` was measured on one shot (h3_config). What a regenerated
band of curls looks like on playback, which is the owner's to judge. The part model's own errors.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))

import h3_config  # noqa: E402
import load_chain  # noqa: E402
from load_chain import Refused  # noqa: E402

KINDS = ("hairline",)
#: Where a load's own part is written: `bench/shot_to_load.py`'s `PARTS_AT`, read as text so this tool does not
#: import the job step. `check_look_rooms.py` holds the two equal.
PARTS_AT = "Video/{video}/parts/{preview}"
#: core's FRAME_PER_TOKEN (comfy/ldm/minimax/model.py): the frames under each latent step, cyclic. Inherited; read
#: here as a copy because core is not importable without the server's environment. `check_look_rooms.py` holds it.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cap():
    import capture_masked_run
    return capture_masked_run


def class_index(name: str) -> int:
    names = json.loads((REPO / "vendor" / "meta_sapiens2" / "seg_classes29.json").read_text())["names"]
    return names.index(name)


def square_grow(mask: np.ndarray, px: int) -> np.ndarray:
    """[n, h, w] or [h, w] of bool grown by `px` every way, as a square: the node's own grow is a max-pool."""
    import cv2
    if px <= 0:
        return mask.copy()
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    if mask.ndim == 2:
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    return np.stack([cv2.dilate(m.astype(np.uint8), k) > 0 for m in mask])


def hairline_room(classes: np.ndarray, track: np.ndarray, hold: int, band: int) -> np.ndarray:
    """His hair over his face and at his hairline, held over `hold` frames either side, on his track: [n, h, w]."""
    hair, face = classes == class_index("Hair"), classes == class_index("Face_Neck")
    near = hair & square_grow(face, band)
    n = len(classes)
    room = np.zeros_like(hair)
    for f in range(n):
        w = slice(max(f - hold, 0), f + hold + 1)
        room[f] = ((hair[w].any(0) & face[w].any(0)) | near[w].any(0)) & track[f]
    return room


def rooms_of(look: dict, classes: np.ndarray, track: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Every `change` room of a subject's look joined, and what each was made with."""
    room, made = np.zeros(track.shape, bool), []
    for r in look.get("change") or []:
        kind = r.get("room")
        if kind not in KINDS:
            raise Refused(f"the look names a room {kind!r}: the kinds built are {', '.join(KINDS)}")
        # one kind is built; a second is a branch here
        s = {**h3_config.LOOK_HAIRLINE, **{k: r[k] for k in ("hold_frames", "band_px") if k in r}}
        got = hairline_room(classes, track, int(s["hold_frames"]), int(s["band_px"]))
        room |= got
        px = got.reshape(len(got), -1).sum(1)
        made.append({"room": kind, "settings": s, "by": r.get("by"), "px_a_frame": {"median": float(np.median(px)), "most": int(px.max())}})
    if not made:
        raise Refused("the look names no `change` room: there is nothing to join to the part")
    return room, made


def look_of(plan: dict, subject: str) -> dict:
    p = plan["subjects"].get(subject)
    if not p or not p.get("look"):
        raise Refused(f"the plan gives {subject!r} no `look`: declare it once on the subject, with `look_by`")
    if not str(p.get("look_by") or "").strip():
        raise Refused(f"the plan's `look` for {subject!r} has no `look_by`: say who declared it and on what")
    return p["look"]


def step_room(job_file: Path, plan_file: Path, by: str, dry: bool, output: Path | None = None) -> list[str]:
    ctx = load_chain.read_job(job_file, need_text=False)
    if (ctx["job_folder"] / f"{ctx['name']}.plan.built.json").exists():
        raise Refused(f"the plan of {ctx['name']} is built: its job file is held as it was from then on. A look is joined before the plan")
    output = output or h3_config.output_dir()
    files = load_chain.preview_files(ctx, output)[ctx["subject"]]
    if files["classes"] is None:
        raise Refused(f"the preview {ctx['preview']['name']} saved no class map of {ctx['subject']!r}: a room is read from it")
    plan = json.loads(plan_file.read_text())
    look = look_of(plan, ctx["subject"])
    c = cap()
    w, h, n = c.probe(str(files["classes"]))
    part_from = Path(ctx["subject_keys"]["parts_file"]) if ctx["subject_keys"].get("parts_file") else files["parts"]
    if part_from is None or not Path(part_from).is_file():
        raise Refused(f"the part of {ctx['subject']!r} the look is joined to, {part_from}, is not there")
    pw, ph, pn = c.probe(str(part_from))
    if (pw, ph) != (w, h) or pn < n:
        raise Refused(f"the part {Path(part_from).name} is {pw}x{ph} with {pn} frames and the class map {w}x{h} with {n}: not of one preview")
    classes = c.read_classes(str(files["classes"]), (w, h), 0, 0, n)
    track = c.read_mask(str(files["track"]), (w, h), 0, 0, n)[0]
    part = c.read_mask(str(part_from), (w, h), 0, 0, n)[0]
    room, made = rooms_of(look, classes, track)
    joined = part | room
    added = (joined & ~part).reshape(n, -1).sum(1)
    folder = output / PARTS_AT.format(video=load_chain.video_of(ctx, REPO / "data"), preview=ctx["preview"]["name"])
    stem = f"{ctx['preview']['name']}_{ctx['subject']}"
    part_file, rooms_file = folder / f"{stem}_part_with_look_{ctx['name']}.mkv", folder / f"{stem}_look_rooms_{ctx['name']}.mkv"
    record = {"what": "a subject's part with its look's rooms joined, by bench/look_rooms.py", "load": ctx["name"], "subject": ctx["subject"],
              "preview": ctx["preview"]["name"], "its_first_frame_is_source_frame": ctx["preview"]["first"], "frames": int(n), "size": [int(w), int(h)],
              "part_joined_to": str(part_from), "class_map": str(files["classes"]), "track": str(files["track"]),
              "look_by": plan["subjects"][ctx["subject"]]["look_by"], "rooms": made, "file": str(part_file), "rooms_file": str(rooms_file),
              "px_the_rooms_add_to_the_part_a_frame": {"median": float(np.median(added)), "most": int(added.max())},
              "by": by, "made_at": utc_now()}
    lines = [f"look of {ctx['subject']} on {ctx['name']}: " + "; ".join(f"{m['room']} {m['settings']} ({m['px_a_frame']['median']:.0f} px a frame at the median)" for m in made),
             f"  the rooms add {record['px_the_rooms_add_to_the_part_a_frame']['median']:.0f} px a frame to the part at the median, "
             f"{record['px_the_rooms_add_to_the_part_a_frame']['most']} at most; joined to {Path(part_from).name}"]
    if dry:
        return lines + [f"  WOULD WRITE {part_file} and {rooms_file}, and give the job file `parts_file` and `look_rooms`"]
    for f in (part_file, rooms_file):
        if f.exists():
            raise Refused(f"{f} is there: a look made again is a new load name, or the file is moved aside by a person")
    folder.mkdir(parents=True, exist_ok=True)
    c.write_mask_video(part_file, joined)
    c.write_mask_video(rooms_file, room)
    record["frame_hash"] = load_chain.frame_hash(str(part_file))
    part_file.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    keep = job_file.with_name(job_file.name + f".before_look_rooms_{record['made_at'].replace(':', '')}")
    shutil.copy2(job_file, keep)
    d = json.loads(job_file.read_text())
    mine = next((s for s in d["job"].get("subjects", []) if s["label"] == ctx["subject"]), None)
    if mine is None:
        raise Refused(f"{job_file} names no subject {ctx['subject']!r} in `job.subjects`")
    mine["parts_file"], mine["parts_by"] = str(part_file), f"bench/look_rooms.py, the plan's look ({record['look_by']})"
    mine["look_rooms"] = {k: record[k] for k in ("file", "rooms_file", "rooms", "part_joined_to", "frame_hash", "by", "made_at")}
    job_file.write_text(json.dumps(d, indent=1) + "\n")
    return lines + [f"  wrote {part_file} and {rooms_file}; {job_file.name} now names it as the part (the file before: {keep.name})"]


# ---------------------------------------------------------------- the check, on the node's own plan

def steps_of(frames: int) -> list[int]:
    out, k = [], 0
    while len(out) < frames:
        out += [k] * FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)]
        k += 1
    return out[:frames]


def laid_back(tokens: np.ndarray, zoom: dict | None, size: tuple[int, int]) -> np.ndarray:
    """One latent step's token cells [ch, cw] as the region on the load's frames [h, w]: through the zoom box when
    the window was rendered zoomed (its cells are then of the box enlarged to the canvas)."""
    from PIL import Image
    w, h = size
    if not zoom:
        return np.array(Image.fromarray(tokens.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    x0, y0, x1, y1 = zoom["box"]
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = np.array(Image.fromarray(tokens.astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)) > 127
    return out


def node_region(windows: Path, name: str, frames: int, size: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    """The region each frame of the load was planned with, [frames, h, w], from the plan's own files; and each window's zoom."""
    plan = json.loads((windows / f"{name}_plan.json").read_text())
    w, h = size
    out, zooms, seen = np.zeros((frames, h, w), bool), [], np.zeros(frames, bool)
    for win in plan["windows"]:
        path = windows / f"{name}_window_{win['number']}_planned_region.npz"
        if not path.is_file():
            raise Refused(f"the plan names window {win['number']} and {path.name} is not there")
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            tokens = z["tokens"].astype(bool)
        st = steps_of(int(meta["frames"]))
        if max(st) >= len(tokens):
            raise Refused(f"{path.name}: {meta['frames']} frames take {max(st) + 1} latent steps and the file holds {len(tokens)}")
        zooms.append({"window": win["number"], "zoom": meta.get("zoom")})
        for f in range(int(win["first_written_frame"]), int(win["last_written_frame"]) + 1):
            if f < frames:
                out[f], seen[f] = laid_back(tokens[st[f - int(meta["first_frame"])]], meta.get("zoom"), size), True
    if not seen.all():
        raise Refused(f"the plan's windows write {int(seen.sum())} of the load's {frames} frames")
    return out, zooms


def figures_of(region: np.ndarray, room: np.ndarray, hair: np.ndarray, face: np.ndarray, first: int, cell_px: float) -> dict:
    """What the check prints, from the region and the rooms on the load's frames (frame 0 the load's first)."""
    n = len(region)
    outside = (room & ~region).reshape(n, -1).sum(1)
    switches = [int(((region[f] ^ region[f - 1]) & hair[f]).sum()) for f in range(1, n)]
    reach = []
    for f in range(n):
        cols = np.nonzero(face[f].any(0) & hair[f].any(0) & region[f].any(0))[0][::4]
        g = [int(np.nonzero(face[f][:, x])[0].min() - np.nonzero(region[f][:, x])[0].min()) for x in cols]
        if g:
            reach.append(float(np.median(g)))
    return {"room_px_outside_the_region": {"frames": [first + int(i) for i in np.nonzero(outside)[0]], "px": [int(v) for v in outside[outside > 0]]},
            "region_px_a_frame_median": float(np.median(region.reshape(n, -1).sum(1))), "cell_px_on_the_frame": cell_px,
            "hair_px_switching_from_the_frame_before": {"frames": sum(1 for v in switches if v), "px": int(sum(switches)),
                                                        "most": max(switches, default=0),
                                                        "as_cells": round(sum(switches) / (cell_px * cell_px), 1)},
            "region_reaches_above_the_hairline_px": ([min(reach), float(np.median(reach)), max(reach)] if reach else None)}


def notches(region_cells: np.ndarray) -> int:
    """Kept cells with three or four of their four neighbours regenerated, over every step."""
    nb = np.zeros(region_cells.shape, int)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb += np.roll(np.roll(region_cells, dy, -2), dx, -1)
    return int((~region_cells & (nb >= 3)).sum())


def step_check(job_file: Path, output: Path | None = None) -> tuple[int, list[str]]:
    ctx = load_chain.read_job(job_file, need_text=False)
    output = output or h3_config.output_dir()
    told = ctx["subject_keys"].get("look_rooms")
    if not told:
        raise Refused(f"{job_file} carries no `look_rooms` on {ctx['subject']!r}: run `look_rooms.py room` before the plan")
    c = cap()
    rooms_file = Path(told["rooms_file"])
    w, h, _ = c.probe(str(rooms_file))
    first, frames, at = ctx["job"]["first"], ctx["job"]["frames"], ctx["preview"]["first"]
    region, zooms = node_region(load_chain.plan_windows(ctx, output), ctx["name"], frames, (w, h))
    room = c.read_mask(str(rooms_file), (w, h), at, first, frames)[0]
    files = load_chain.preview_files(ctx, output)
    classes = c.read_classes(str(files[ctx["subject"]]["classes"]), (w, h), at, first, frames)
    hair, face = classes == class_index("Hair"), classes == class_index("Face_Neck")
    scale = next((z["zoom"]["scale"] for z in zooms if z["zoom"]), 1.0)
    with np.load(load_chain.plan_windows(ctx, output) / f"{ctx['name']}_window_{zooms[0]['window']}_planned_region.npz") as z:
        cells = z["tokens"].astype(bool)
    cell_px = h / cells.shape[1] / float(scale)
    fig = figures_of(region, room, hair, face, first, cell_px)
    fig["notch_cells_over_the_steps_of_window_1"] = notches(cells)
    fig["other_subjects_px_in_the_region_most"] = {o: int((c.read_mask(str(files[o]["track"]), (w, h), at, first, frames)[0] & region).reshape(frames, -1).sum(1).max())
                                                   for o in ctx["others_named"]}
    out = ctx["job_folder"] / f"{ctx['name']}.look_check.json"
    out.write_text(json.dumps({"load": ctx["name"], "made_at": utc_now(), "zooms": zooms, **fig}, indent=1) + "\n")
    bad = fig["room_px_outside_the_region"]["frames"]
    said = ", ".join("window %s: %s" % (z["window"], "zoomed %sx" % z["zoom"]["scale"] if z["zoom"] else "whole frame") for z in zooms)
    lines = [f"look check of {ctx['name']}, on the node's own plan ({said}):",
             ("RED: a room the look regenerates lies outside the node's region on source frames " + ", ".join(map(str, bad)))
             if bad else "every pixel of the look's rooms is inside the node's region on every frame",
             f"  data: region {fig['region_px_a_frame_median']:.0f} px a frame at the median, cells of {cell_px:g} px on the frame",
             f"  data: his hair switching between regenerated and kept: {fig['hair_px_switching_from_the_frame_before']}",
             f"  data: the region reaches above his hairline, px (least, median, most of frames): {fig['region_reaches_above_the_hairline_px']}",
             f"  data: kept cells with 3 or 4 regenerated neighbours, over window 1's steps: {fig['notch_cells_over_the_steps_of_window_1']}",
             f"  data: other subjects' px inside the region, most on a frame: {fig['other_subjects_px_in_the_region_most']}",
             f"  written {out}. It cannot see what the model draws in the region, nor how it looks on playback."]
    return (1 if bad else 0), lines


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = a.add_subparsers(dest="step", required=True)
    r = sub.add_parser("room", help="join the plan's look to the load's part, before the plan")
    r.add_argument("job_file", type=Path)
    r.add_argument("--plan", type=Path, required=True, help="data/<video>/plan.json")
    r.add_argument("--by", required=True, help="who is running it")
    r.add_argument("--dry", action="store_true", help="say what would be written and write nothing")
    k = sub.add_parser("check", help="the node's own planned region against the look's rooms, after the plan")
    k.add_argument("job_file", type=Path)
    args = a.parse_args()
    try:
        if args.step == "room":
            print("\n".join(step_room(args.job_file, args.plan, args.by, args.dry)))
            return 0
        code, lines = step_check(args.job_file)
        print("\n".join(lines))
        return code
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
