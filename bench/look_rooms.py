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
`by`. A room is joined to the part the node is handed, so the region is grown from both, and a cell holding that
mask is never given back to a keep-out (`video_mask.window`). One kind is built:

- `hairline`: for a subject whose hair is KEPT and whose region is the face. His hair over his face and a band of
  his hair at the hairline, each held over the frames round it, so the region's edge sits in his curls and holds
  still instead of running through the hairline the class map flips from frame to frame. A frame's room: the pixels
  of his track that are Hair on some frame within `hold_frames` either side and Face_Neck on some frame within it
  (the fringe the class map flips), and the Hair within `band_px` of Face_Neck on any frame within it (the band),
  from his own class map; never a pixel of another subject's track the load names. Its settings and their
  provenance: `h3_config.LOOK_HAIRLINE`. The hold reaches as far as his head moves in it: on a fast turn the room is
  as wide as the turn (the record prints the room's px a frame).

A kind it does not know is refused by name; `add` (a new person's hair from their still) and a `change` room
carried from large frames (his beard) are designed in the read and not built.

**When `room` runs, and what it writes.** After the chain's `first` (the load's capture is made and verified) and
before its plan. It joins the rooms to the part the CAPTURE made for the load, `parts_held` where the capture has it
(its fills graded, `pieces` cut, its floor applied), the same file the builder would otherwise load
(`masked_job.capture_files`). Joining before the capture was the first form (5f832636) and was wrong: the capture
grades a part by the classes it is made of, and on third_fp_1222's preview the joined part (his face and the band of
hair) fell under its bar on 3 frames, which it would have emptied, leaving his own face (a blind reader,
2026-10-14). The joined part, on the capture's frames, is written as `<load>_part_with_look.mkv` beside the
capture's own masks' parts folder (`bench/shot_to_load.py`'s `PARTS_AT`), the rooms alone beside it
(`<load>_look_rooms.mkv`, what `check` holds the region to), and a record. The LOAD is given `parts_mask` and
`parts_first` (the builder wires that file as the part, as is) and `look_rooms` (the record), after a copy of the job
file is kept beside it. Refused: a load whose plan has been built (its job file is held byte-equal from then on), a
capture not finished and verified, a file that is there.

**What `check` reads.** The node's own planned region: each window the plan lists, its `region_file` laid onto the
load's frames as the node lays its render (`video_mask.pixel_alpha` a latent step, `video_mask.lay_back` through the
window's zoom box), a frame the plan leaves as the source (`frames_left_as_source`) holding no region. Never the
capture's region, which reads a zoomed run on the canvas's 16 px grid (on third_fp_4479b it covered 1.5 to 1.7 times
the node's own). The load's frames are its own (`first`, `frames` on the load, else the job's). RED, naming the
frames: a pixel of a room outside the node's region. Data, never red: the region's px; how many px of his hair switch
between regenerated and kept from one frame to the next; kept cells three or four of whose four neighbours are
regenerated (notches), over every written latent step; how far above his hairline the region reaches; the px of
each other subject the load names inside the region. Refused: a plan that continues a stored window, a window whose
zoom is for another canvas, a gap in the frames the windows write.

WHAT IT CANNOT SEE. Whether the model draws what the region allows: on third_fp_4479b it drew a beard inside the
region. Whether a room is the right size: `band_px` was measured on one shot (h3_config). What a regenerated
band of curls looks like on playback, which is the owner's to judge. The part model's own errors. A cut inside a
load: the hold reaches across it (a load holds one shot unless its job names `two_shots`).
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


def hairline_room(classes: np.ndarray, track: np.ndarray, hold: int, band: int, others: np.ndarray | None = None) -> np.ndarray:
    """His hair over his face and at his hairline, held over `hold` frames either side, on his track and off every
    other subject's: [n, h, w]."""
    hair, face = classes == class_index("Hair"), classes == class_index("Face_Neck")
    near = hair & square_grow(face, band)
    n = len(classes)
    room = np.zeros_like(hair)
    for f in range(n):
        w = slice(max(f - hold, 0), f + hold + 1)
        room[f] = ((hair[w].any(0) & face[w].any(0)) | near[w].any(0)) & track[f]
    if others is not None:
        room &= ~others
    return room


def rooms_of(look: dict, classes: np.ndarray, track: np.ndarray, others: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Every `change` room of a subject's look joined, and what each was made with."""
    room, made = np.zeros(track.shape, bool), []
    for r in look.get("change") or []:
        kind = r.get("room")
        if kind not in KINDS:
            raise Refused(f"the look names a room {kind!r}: the kinds built are {', '.join(KINDS)}")
        # one kind is built; a second is a branch here
        s = {**h3_config.LOOK_HAIRLINE, **{k: r[k] for k in ("hold_frames", "band_px") if k in r}}
        got = hairline_room(classes, track, int(s["hold_frames"]), int(s["band_px"]), others)
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


def load_span(ctx: dict) -> tuple[int, int]:
    """The load's own first frame and length: the load's keys, else the job's."""
    return int(ctx["load"].get("first", ctx["job"]["first"])), int(ctx["load"].get("frames", ctx["job"]["frames"]))


def capture_of(ctx: dict) -> dict:
    """The load's capture, finished and verified, and its files for the load's subject and the others it names."""
    folder = ctx["capture"]
    status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").is_file() else {}
    checked = json.loads((folder / "verify.json").read_text()) if (folder / "verify.json").is_file() else None
    if status.get("state") != "done" or checked is None or checked.get("failed") or checked.get("not_made"):
        raise Refused(f"the capture {folder.name} is not finished and verified: the room is joined to the part the capture made, after the chain's `first`")
    manifest = json.loads((folder / "manifest.json").read_text())
    sightings = {s["label"]: [x["by"] for x in s["sightings"]] for s in manifest["subjects"]}
    subject = ctx["subject"]
    by = ctx["load"].get("by") or (sightings.get(subject) or [None])[0]
    if not by or len(sightings.get(subject, [])) > 1 and not ctx["load"].get("by"):
        raise Refused(f"the capture {folder.name} holds {sightings.get(subject)} for {subject!r}: say which with the load's `by`")
    sub = folder / "subjects" / subject
    part = next((p for p in (sub / f"parts_held__{by}.mkv", sub / f"parts__{by}.mkv") if p.is_file()), None)
    if part is None or not (sub / f"classes__{by}.npz").is_file():
        raise Refused(f"the capture {folder.name} holds no part or no class map of {subject!r} by {by!r}")
    others = [o for o in ctx["others_named"] if (folder / "subjects" / o).is_dir()]
    return {"folder": folder, "first": int(manifest["first_frame"]), "frames": int(manifest["frames"]), "size": tuple(manifest["size"]),
            "by": by, "part": part, "classes": sub / f"classes__{by}.npz", "masks": sub / f"masks__{by}.npz",
            "others": {o: next((folder / "subjects" / o).glob("masks__*.npz")) for o in others}}


def tracks_of(masks: Path, w: int) -> np.ndarray:
    with np.load(masks) as z:
        return np.unpackbits(z["track"], axis=-1)[..., :w].astype(bool)


def step_room(job_file: Path, plan_file: Path, by: str, dry: bool, output: Path | None = None) -> list[str]:
    ctx = load_chain.read_job(job_file, need_text=False)
    if (ctx["job_folder"] / f"{ctx['name']}.plan.built.json").exists():
        raise Refused(f"the plan of {ctx['name']} is built: its job file is held as it was from then on. A look is joined before the plan")
    if ctx["load"].get("parts_mask"):
        raise Refused(f"the load {ctx['name']} already names a parts_mask ({ctx['load']['parts_mask']}): a look is joined once, to the capture's part")
    output = output or h3_config.output_dir()
    look = look_of(json.loads(plan_file.read_text()), ctx["subject"])
    look_by = json.loads(plan_file.read_text())["subjects"][ctx["subject"]]["look_by"]
    k = capture_of(ctx)
    c = cap()
    w, h = k["size"]
    n = k["frames"]
    with np.load(k["classes"]) as z:
        classes = z["classes"]
    track = tracks_of(k["masks"], w)
    others = np.zeros((n, h, w), bool)
    for path in k["others"].values():
        others |= tracks_of(path, w)
    part = c.read_mask(str(k["part"]), (w, h), 0, 0, n)[0]
    room, made = rooms_of(look, classes, track, others)
    joined = part | room
    added = (joined & ~part).reshape(n, -1).sum(1)
    folder = output / PARTS_AT.format(video=load_chain.video_of(ctx, REPO / "data"), preview=ctx["preview"]["name"])
    part_file, rooms_file = folder / f"{ctx['name']}_part_with_look.mkv", folder / f"{ctx['name']}_look_rooms.mkv"
    record = {"what": "a load's part, as its capture made it, with its subject's look's rooms joined, by bench/look_rooms.py", "load": ctx["name"],
              "subject": ctx["subject"], "capture": str(k["folder"]), "its_first_frame_is_source_frame": k["first"], "frames": n, "size": [w, h],
              "part_joined_to": str(k["part"]), "class_map": str(k["classes"]), "others_kept_out_of_the_rooms": sorted(k["others"]),
              "look_by": look_by, "rooms": made, "file": str(part_file), "rooms_file": str(rooms_file),
              "px_the_rooms_add_to_the_part_a_frame": {"median": float(np.median(added)), "most": int(added.max())},
              "by": by, "made_at": utc_now()}
    lines = [f"look of {ctx['subject']} on {ctx['name']}: " + "; ".join(f"{m['room']} {m['settings']} ({m['px_a_frame']['median']:.0f} px a frame at the median, "
                                                                         f"{m['px_a_frame']['most']} at most)" for m in made),
             f"  the rooms add {record['px_the_rooms_add_to_the_part_a_frame']['median']:.0f} px a frame to the capture's part at the median, "
             f"{record['px_the_rooms_add_to_the_part_a_frame']['most']} at most; joined to {k['part'].name}; kept off {sorted(k['others']) or 'nobody'}"]
    if dry:
        return lines + [f"  WOULD WRITE {part_file} and {rooms_file}, and give the load `parts_mask`, `parts_first` and `look_rooms`"]
    for f in (part_file, rooms_file):
        if f.exists():
            raise Refused(f"{f} is there: a look made again is a new load name, or the file is moved aside by a person")
    folder.mkdir(parents=True, exist_ok=True)
    c.write_mask_video(part_file, joined)
    c.write_mask_video(rooms_file, room)
    part_file.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    keep = job_file.with_name(job_file.name + f".before_look_rooms_{record['made_at'].replace(':', '')}")
    shutil.copy2(job_file, keep)
    d = json.loads(job_file.read_text())
    one = d["loads"][0]
    one["parts_mask"], one["parts_first"] = str(part_file), k["first"]
    one["look_rooms"] = {key: record[key] for key in ("file", "rooms_file", "rooms", "part_joined_to", "capture", "its_first_frame_is_source_frame", "by", "made_at")}
    job_file.write_text(json.dumps(d, indent=1) + "\n")
    return lines + [f"  wrote {part_file} and {rooms_file}; the load names it as its parts_mask from source frame {k['first']} "
                    f"(the job file before: {keep.name})"]


# ---------------------------------------------------------------- the check, on the node's own plan

def node_region(windows: Path, name: str, frames: int, size: tuple[int, int]) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """The region each frame of the load was planned with, [frames, h, w] of bool, laid as the node lays its render; each
    window's zoom; and each window's token cells over the steps it writes (for the notch count)."""
    import torch
    vm = cap()._pack("video_mask")
    plan = json.loads((windows / f"{name}_plan.json").read_text())
    if plan.get("continue_from"):
        raise Refused(f"the plan of {name} continues a stored window ({plan['continue_from']}): a look's load is planned fresh")
    w, h = size
    out, seen, zooms, cells, end = np.zeros((frames, h, w), bool), np.zeros(frames, bool), [], [], 0
    for row in plan["windows"]:
        if not row.get("region_file"):
            raise Refused(f"window {row['number']} is not planned: {row.get('why', 'no region file')}")
        path = windows / row["region_file"]
        if not path.is_file():
            raise Refused(f"the plan names {row['region_file']} and it is not beside the plan")
        if int(row["first_written_frame"]) != end:
            raise Refused(f"window {row['number']} writes from frame {row['first_written_frame']} of the load and the window before ended on {end}")
        got = vm.load_window_region(str(path))
        zoom, tokens = got["zoom"], got["tokens"]
        rh, rw = int(got["mask"].shape[1]), int(got["mask"].shape[2])
        if zoom and [int(v) for v in zoom.get("canvas", [])] != [w, h]:
            raise Refused(f"{row['region_file']}: the zoom is for a canvas of {zoom.get('canvas')} and the load's masks are {w}x{h}")
        if not zoom and (rw, rh) != (w, h):
            raise Refused(f"{row['region_file']}: the window was planned at {rw}x{rh} and the load's masks are {w}x{h}")
        alpha = vm.pixel_alpha(tokens, rh, rw, 0) > 0.5                       # [window frames, rh, rw], a step's cells on each of its frames
        left = set(int(x) for x in row.get("frames_left_as_source", []))
        lo, hi = int(row["first_written_frame"]), int(row["last_written_frame"])
        idx = [f - int(row["first_frame"]) for f in range(lo, hi + 1)]
        on = alpha[idx].to(torch.float32)
        if zoom:
            on = vm.lay_back(torch.ones(*on.shape, 3), on, torch.zeros(len(idx), h, w, 3), zoom)[..., 0] > 0.5
        else:
            on = on > 0.5
        for j, f in enumerate(range(lo, hi + 1)):
            if f < frames:
                out[f], seen[f] = (on[j].numpy() if f not in left else False), True
        zooms.append({"window": row["number"], "zoom": zoom, "frames_left_as_source": sorted(left)})
        runs = vm.run_lengths(int(tokens.shape[0]))
        starts = np.cumsum([0] + list(runs))
        cells.append(tokens.numpy().astype(bool)[[k for k in range(len(runs)) if starts[k] <= hi - int(row["first_frame"]) and starts[k + 1] > lo - int(row["first_frame"])]])
        end = hi + 1
    if not seen.all():
        raise Refused(f"the plan's windows write {int(seen.sum())} of the load's {frames} frames")
    return out, zooms, cells


def figures_of(region: np.ndarray, room: np.ndarray, hair: np.ndarray, face: np.ndarray, first: int) -> dict:
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
            "region_px_a_frame_median": float(np.median(region.reshape(n, -1).sum(1))),
            "hair_px_switching_from_the_frame_before": {"frames": sum(1 for v in switches if v), "px": int(sum(switches)), "most": max(switches, default=0)},
            "region_reaches_above_the_hairline_px": ([min(reach), float(np.median(reach)), max(reach)] if reach else None)}


def notches(region_cells: np.ndarray) -> int:
    """Kept cells with three or four of their four neighbours regenerated, over every step [steps, ch, cw]; a cell at
    the edge has only the neighbours inside the grid."""
    p = np.pad(region_cells, ((0, 0), (1, 1), (1, 1)))
    nb = p[:, :-2, 1:-1].astype(int) + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
    return int((~region_cells & (nb >= 3)).sum())


def step_check(job_file: Path, output: Path | None = None) -> tuple[int, list[str]]:
    ctx = load_chain.read_job(job_file, need_text=False)
    output = output or h3_config.output_dir()
    told = ctx["load"].get("look_rooms")
    if not told:
        raise Refused(f"{job_file}: the load carries no `look_rooms`: run `look_rooms.py room` before the plan")
    c = cap()
    rooms_file = Path(told["rooms_file"])
    w, h, _ = c.probe(str(rooms_file))
    at = int(told["its_first_frame_is_source_frame"])
    first, frames = load_span(ctx)
    region, zooms, cells = node_region(load_chain.plan_windows(ctx, output), ctx["name"], frames, (w, h))
    room = c.read_mask(str(rooms_file), (w, h), at, first, frames)[0]
    k = capture_of(ctx)
    with np.load(k["classes"]) as z:
        classes = z["classes"][first - k["first"]:first - k["first"] + frames]
    hair, face = classes == class_index("Hair"), classes == class_index("Face_Neck")
    fig = figures_of(region, room, hair, face, first)
    fig["notch_cells_over_the_written_steps"] = sum(notches(x) for x in cells)
    fig["other_subjects_px_in_the_region_most"] = {o: int((tracks_of(p, w)[first - k["first"]:first - k["first"] + frames] & region).reshape(frames, -1).sum(1).max())
                                                   for o, p in k["others"].items()}
    out = ctx["job_folder"] / f"{ctx['name']}.look_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"load": ctx["name"], "made_at": utc_now(), "zooms": zooms, **fig}, indent=1) + "\n")
    bad = fig["room_px_outside_the_region"]["frames"]
    said = ", ".join("window %s: %s" % (z["window"], "zoomed %sx" % z["zoom"]["scale"] if z["zoom"] else "whole frame") for z in zooms)
    lines = [f"look check of {ctx['name']}, source frames {first}-{first + frames - 1}, on the node's own plan ({said}):",
             ("RED: a room the look regenerates lies outside the node's region on source frames " + ", ".join(map(str, bad)))
             if bad else "every pixel of the look's rooms is inside the node's region on every frame",
             f"  data: region {fig['region_px_a_frame_median']:.0f} px a frame at the median",
             f"  data: his hair switching between regenerated and kept: {fig['hair_px_switching_from_the_frame_before']}",
             f"  data: the region reaches above his hairline, px (least, median, most of frames): {fig['region_reaches_above_the_hairline_px']}",
             f"  data: kept cells with 3 or 4 regenerated neighbours, over the written steps: {fig['notch_cells_over_the_written_steps']}",
             f"  data: other subjects' px inside the region, most on a frame: {fig['other_subjects_px_in_the_region_most']}",
             f"  written {out}. It cannot see what the model draws in the region, nor how it looks on playback."]
    return (1 if bad else 0), lines


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = a.add_subparsers(dest="step", required=True)
    r = sub.add_parser("room", help="join the plan's look to the part the load's capture made, after `first` and before the plan")
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
