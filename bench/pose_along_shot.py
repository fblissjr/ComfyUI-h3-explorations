#!/usr/bin/env python3
"""Run the pose model on every frame of a shot and keep its points: what the eye marks and the reader are made from.

    python bench/pose_along_shot.py run --video SOURCE.mov --first 1575 --count 155 --canvas 1024x768 \\
        --box-mask TRACK.mkv --box-mask-first 1575 --model-folder <the store>/<h3_config.POSE_MODEL> \\
        --out data/<job>/lead_1575_points.npz --by <session> [--names NAMES.json]
    python bench/pose_along_shot.py read data/<job>/lead_1575_points.npz [--table]

`run` needs the card: whoever holds it runs it. It reads the frames it is told to, by number; fits a source to
the canvas the way the lane's loader does (a clip already the canvas's size is left as it is); gives the model
one box a frame, the tracked person's from a named mask video or the whole frame; and writes the points, the
scores, the picture's size, the points' NAMES and the source frame of row 0. A record goes beside it
(`.json`). It derives no figure: those have one definition, `bench/_lib/face_points.py`.

HELD TO THE TRACK. With a mask video, each frame is given to the model with every pixel outside that frame's mask
(grown by `HOLD_GROW_PX`) set to `HOLD_GREY`, so the person the model reads is the tracked one and not another
inside the same box. The box alone did not do that: on a close group of women (the lead on 3167-3213,
2026-10-14) the box round the lead's track held a second woman's face above her head, and on 11 of 47 frames
every face point went to that face; the reference step printed PASSED and drew her. The record then says, a frame
at a time, how much of the face (`face_points.face_ids`) lies on the mask and whether its middle does
(`face_points.face_middle_on_mask`), and lists the frames whose face's middle is off: a reference is not made from
those (`bench/head_reference.py`). A frame with an empty mask is held to the last mask before it, as its box is.

THE NAMES. Every tool finds a point by its name, so a points file carries them, and its record says where
they came from. The checkpoint in the store does not name its points (its config calls them `LABEL_0` on).
Meta's code does, and that list is kept in `vendor/meta_sapiens2/pose_keypoints308.json` with Meta's licence beside
it (`vendor/meta_sapiens2/README.md`). It is used for a model folder whose config has that many points and whose own
left-right pairs (`flip_pairs`) are the list's. That says the folder is the model the list is for; it does
not prove the list's order, which `bench/check_meta_sapiens2.py` holds to Meta's file, and what else the order
rests on is in `bench/results/2026-10-11_sapiens2_pose_first_frames.md`. A folder that names its own points is read from its
config, `--names` (a json list in the checkpoint's order) is used as given, and a folder that fits none of
these is refused.

A render is read the same way. Its frame 0 is not source frame 0: `--source-first` says which source frame the
clip's `--first` frame is, and that is what the file records.

`read` needs no card. It says what a shot's points can and cannot carry before anything is drawn from them or
read by them: the face's height and the gap between the lash lines in the picture's pixels, beside
`h3_config.POSE_READING["lid_gap_px_to_draw"]`; how many frames sit under, over and between the two lines kept
for the eye opening, with what those lines are; and the model's own scores with the reminder that a score does
not say lids were seen. It prints figures with what they were measured on and against what, like a tape
measure; it gives no verdict, and what a shot is good for is the reader's to say. `--table` writes one row a frame beside the points (`.csv`).

WHAT IT CANNOT KNOW
- Whether the mask is the right person. It takes the mask it is given and holds the points to it, so where the
  track itself lies on another subject the hold reads that subject's face, confidently, and every frame reads
  as on the track (a preview's track of the lead lay on the second subject on 1003-1009, 2026-10-13).
- What the grey costs the points of the tracked face, and which reading is right. On the first run (the lead's
  3167-3213, 2026-10-14; the 36 frames whose face both readings put on her track) the held points moved a median
  of 2.2 px and at most 11 px from the unheld ones, and the face read about 12% taller (109 px against 96 at the
  median), most of it from the chin's point. Measured and not explained; which chin is right is not known.
- A render read held (the land step's pass, `bench/load_chain.py`) is held to the SOURCE's track of its subject:
  where a rendered face has moved off that track the grey takes part of it. A render's points read before
  2026-10-14 were read unheld, and its figures are not comparable with those read after.
- Whether the points are right. The model places lid points on the back of a head and on a face a few pixels
  tall, and scores them well.
- A frame with no mask keeps the box of the frame before it (the whole frame, if there is none before). The
  record counts them; a shot where that is many frames is one where the tracker lost the person.
- Points the model scores low are not places. On a close-up its pupils and the hands and feet out of frame
  moved by hundreds of pixels when the picture it was given moved by half a pixel. No figure in
  `bench/_lib/face_points.py` uses them.

THE FIT. A source that is not the canvas's shape is fitted through `bench/_lib/frames.py::FIT`, the one place
the tracked readers keep it: cut to the canvas's shape at the centre, then scaled, as the graphs' loader does.
It matters to half a pixel. On this tool's first run with the real model (2026-10-11, one close-up) `FIT` was
the other order, and its points sat a median of half a pixel, up and down, from those of the day's one-off,
which had cut first; with the fit corrected every point was identical to the one-off's. The record beside a
points file says how its frames were fitted.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402
from _lib import face_points as fp  # noqa: E402
from _lib.frames import FIT, probe, read_mask, stream  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402


#: What the two lines of `h3_config.POSE_READING` for the eye opening are, and the one for the lid gap: said beside
#: every count against them, in the same words as `bench/read_render_against_original.py`.
LINES_NOTE = ("On the one close-up those two lines were read off, a person looking at every frame agreed with the figure outside them and could not tell narrowed eyes from open ones between them; they are a starting value and not this clip's")
GAP_NOTE = "a starting value that lies between the median of the one shot whose marks carried and those of shots whose lash lines are under a pixel apart"


#: The mask is grown by this many pixels of the canvas before the picture outside it is greyed. Reasoned: a tracker's
#: mask edge lies a few pixels inside or outside the person's; the face's own edge needs its few pixels of
#: background, and in a close group another person's face lies further off than this. A starting value.
HOLD_GROW_PX = 8
#: What the pixels outside the grown mask are set to: a flat mid-grey, which carries no face. Reasoned.
HOLD_GREY = 128


#: Meta's names for the pose model's points, kept with Meta's licence beside them: `vendor/meta_sapiens2/README.md`.
TRACKED_NAMES = REPO / "vendor" / "meta_sapiens2" / "pose_keypoints308.json"


def load_model(folder: str, names: list[str] | None = None):
    """The pose model from a folder: (predict, names). `predict(picture [h, w, 3], box [x, y, w, h])` gives points
    [n, 2] in the picture's pixels and scores [n]. `names` given by hand are used as given; else the folder's own.
    Imports torch here and nowhere else."""
    import torch
    from transformers import Sapiens2ForPoseEstimation
    from transformers.models.sapiens2.image_processing_sapiens2 import Sapiens2ImageProcessor
    proc = Sapiens2ImageProcessor.from_pretrained(folder, local_files_only=True)
    model = Sapiens2ForPoseEstimation.from_pretrained(folder, local_files_only=True).eval().to("cuda")
    names = list(names) if names else point_names(folder, model.config)

    def predict(picture: np.ndarray, box: list[float]):
        inputs = proc.preprocess(images=[np.ascontiguousarray(picture)], boxes=[[box]], return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=inputs["pixel_values"].to("cuda"))
        out.heatmaps = out.heatmaps.float().cpu()
        got = proc.post_process_pose_estimation(out, boxes=[[box]])[0][0]
        return got["keypoints"].numpy(), got["scores"].numpy()

    return predict, names


def tracked_names(labels: dict, flip_pairs: list | None) -> list[str] | None:
    """Meta's list for the pose model, when a model's config has that many points and its own left-right pairs are
    the list's; else None. The pairs are the one thing a checkpoint's config says about which point is which."""
    if not TRACKED_NAMES.exists():
        return None
    table = json.loads(TRACKED_NAMES.read_text())
    names = [str(n) for n in table["names"]]
    at = {n: i for i, n in enumerate(names)}
    listed = {tuple(sorted((at[a], at[b]))) for a, b in table["swap"].items()}
    own = {tuple(sorted(int(i) for i in pair)) for pair in flip_pairs or []}
    return names if own and own == listed and len(labels) == len(names) else None


def names_and_source(folder: str, config=None) -> tuple[list[str], str]:
    """The points' names and where they came from: the model's own config, a names file in its folder, or Meta's
    tracked list. A folder that fits none is refused: a points file with made-up names would be read by index in
    all but word."""
    own = json.loads((Path(folder) / "config.json").read_text())
    labels = getattr(config, "id2label", None) or own.get("id2label") or {}
    names = [labels[k] for k in sorted(labels, key=int)]
    if names and not all(n.startswith("LABEL_") for n in names):
        return names, "the model's own config"
    for extra in ("keypoint_names.json", "names.json"):
        f = Path(folder) / extra
        if f.exists():
            return list(json.loads(f.read_text())), f"{extra} in the model's folder"
    listed = tracked_names(labels, own.get("flip_pairs"))
    if listed:
        return listed, f"{TRACKED_NAMES.relative_to(REPO)}: the model's own left-right pairs are that list's"
    raise SystemExit(f"{folder} does not name its points (its config's labels are placeholders and it holds no names file), and its "
                     f"point count and left-right pairs are not those of {TRACKED_NAMES.relative_to(REPO)}: give --names, a json list "
                     "in the checkpoint's order")


def point_names(folder: str, config=None) -> list[str]:
    """The points' names alone: `names_and_source`."""
    return names_and_source(folder, config)[0]


def boxes_from(mask: np.ndarray, covered: np.ndarray, size: tuple[int, int]) -> tuple[list[list[float]], list[int]]:
    """One box a frame [x, y, w, h] round a mask's pixels, and the frames that had none and kept the box before."""
    w, h = size
    out, held, box = [], [], [0.0, 0.0, float(w), float(h)]
    for f in range(len(mask)):
        ys, xs = np.nonzero(mask[f] & bool(covered[f]))
        if len(ys):
            box = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        else:
            held.append(f)
        out.append(list(box))
    return out, held


def held_masks(mask: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(grown, as given, borrowed): the mask each frame is held to, its own or, where it is empty or not covered, the
    last one before it that was not (the whole frame where there is none); that mask grown by `HOLD_GROW_PX`; and
    the frames that took an earlier frame's."""
    import cv2
    k = 2 * HOLD_GROW_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown, given, borrowed, last = np.zeros_like(mask), np.zeros_like(mask), [], np.ones(mask.shape[1:], bool)
    for f in range(len(mask)):
        if covered[f] and mask[f].any():
            last = mask[f]
        else:
            borrowed.append(f)
        given[f] = last
        grown[f] = cv2.dilate(last.astype(np.uint8), kernel).astype(bool)
    return grown, given, borrowed


def face_on_track(points: list[np.ndarray], names: list[str], mask: np.ndarray, source_first: int) -> dict | None:
    """A frame at a time, the share of the face's points on the mask the frame was held to (not grown) and whether
    the face's middle is on it; None when the points carry no face (a model that does not name them)."""
    index = {n: i for i, n in enumerate(names)}
    try:
        fp.face_ids(index)
    except KeyError:
        return None
    share = [round(fp.face_on_mask(p, index, m), 3) for p, m in zip(points, mask)]
    middle = [fp.face_middle_on_mask(p, index, m) for p, m in zip(points, mask)]
    return {"what": "a frame at a time: the share of the face's points (bench/_lib/face_points.py::face_ids) inside the tracked mask, and whether "
                    "the face's middle (the median of those points) is a pixel of it; the mask the frame was held to (its own, or an earlier frame's where its own is "
                    "empty), not grown",
            "share_a_frame": share, "middle_on_a_frame": middle,
            "frames_whose_face_middle_is_off_the_mask": [source_first + f for f, ok in enumerate(middle) if not ok]}


def run(video: str, first: int, count: int, canvas: tuple[int, int], out: Path, by: str, predict, names: list[str], *,
        source_first: int | None = None, box_mask: str = "", box_mask_first: int | None = None, model: str = "", names_from: str = "", say=print) -> dict:
    """Read the frames, give each to `predict` in its box, write the points and the record. Returns the record."""
    w, h = canvas
    vw, vh, have = probe(video)
    if first < 0 or first + count > have:
        raise SystemExit(f"{video} has {have} frames; frames {first} to {first + count - 1} were asked for")
    fitted = (vw, vh) != (w, h)
    source_first = first if source_first is None else source_first
    if box_mask:
        at = source_first if box_mask_first is None else box_mask_first
        mask, covered = read_mask(box_mask, (w, h), at, source_first, count)
        boxes, held = boxes_from(mask, covered, (w, h))
        hold, given, borrowed = held_masks(mask, covered)
    else:
        boxes, held, hold, given, borrowed = [[0.0, 0.0, float(w), float(h)]] * count, [], None, None, []
    points, scores = [], []
    for f, picture in enumerate(stream(video, (w, h), first, count, vf=FIT.format(w=w, h=h) if fitted else "", pix="rgb24")):
        if hold is not None:
            picture = np.where(hold[f][..., None], picture, np.uint8(HOLD_GREY))
        p, s = predict(picture, boxes[f])
        points.append(np.asarray(p, np.float32)); scores.append(np.asarray(s, np.float32))
    if len(points) != count:
        raise SystemExit(f"{video} gave {len(points)} frames from frame {first}; {count} were asked for")
    if points[0].shape[0] != len(names):
        raise SystemExit(f"the model gave {points[0].shape[0]} points a frame and {len(names)} names")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keypoints=np.stack(points), scores=np.stack(scores), size=np.array([w, h]), names=np.array(names),
                        first_frame=np.array(source_first))
    record = {
        "points": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model or h3_config.POSE_MODEL, "names_from": names_from or "given by the caller",
        "made_from": {"video": str(video), "its_frames": [first, first + count - 1], "which_are_source_frames": [source_first, source_first + count - 1],
                      "fitted_to_the_canvas": ("as the loader fits a source: cut to the canvas's shape at the centre, then scaled" if fitted
                                               else "no: it is the canvas's size"),
                      "the_filter_it_was_read_through": FIT.format(w=w, h=h) if fitted else "",
                      "box": ({"mask_video": str(box_mask), "its_first_frame_is_source_frame": source_first if box_mask_first is None else box_mask_first}
                              if box_mask else "the whole frame")},
        "size": [w, h], "frames": count,
        "frames_with_no_mask_that_kept_the_box_before": [source_first + f for f in held],
        "held_to_the_mask": ({"how": f"every pixel outside the frame's mask grown by {HOLD_GROW_PX} px set to grey {HOLD_GREY} before the model read it",
                              "frames_held_to_an_earlier_frames_mask": [source_first + f for f in borrowed]} if hold is not None else None),
        "face_on_the_mask": face_on_track(points, names, given, source_first) if box_mask else None,
        "not_known": ["whether the mask is the right person: it is taken as given and the points are held to it, so a track that lies on another "
                      "subject gives that subject's face, read as on the track",
                      "what the grey costs the tracked face's own points: on the first run the face read about 12% taller held than unheld, most of it the "
                      "chin's point; which is right is not known",
                      "whether the points are right: the model places and scores lid points on the back of a head and on a face a few pixels tall"],
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"wrote {out} ({count} frames, {len(names)} points a frame, row 0 is source frame {source_first}) and {out.with_suffix('.json').name}")
    if held:
        say(f"{len(held)} of {count} frames had no mask and kept the box of the frame before")
    on = record["face_on_the_mask"]
    if on:
        say("the face on the tracked mask, a frame at a time (share of its points; * where its middle is OFF the mask):")
        line = []
        for f, (v, ok) in enumerate(zip(on["share_a_frame"], on["middle_on_a_frame"])):
            line.append(f"{source_first + f}:{v:.2f}{'' if ok else '*'}")
            if len(line) == 10 or f == count - 1:
                say("  " + " ".join(line))
                line = []
        off = on["frames_whose_face_middle_is_off_the_mask"]
        say(f"THE FACE'S MIDDLE IS OFF THE TRACKED MASK on {len(off)} of {count} frames: " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans(off))
            if off else f"the face's middle is on the tracked mask on all {count} frames")
    try:
        read(out, say=say)
    except KeyError as exc:                              # the file is written; only the closing summary needs the face's points
        say(f"no summary of the face: these points have no point named {exc}")
    return record


def read(points_path: Path | str, *, names: list[str] | None = None, table: bool = False, say=print) -> dict:
    """What a shot's points can carry: the face's height, the lid gap in pixels against the line, the scores."""
    points = fp.load(points_path, names)
    rows = fp.table(points)
    col = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    line = h3_config.POSE_READING["lid_gap_px_to_draw"]
    face, gap = col("face_height_px"), col("lid_gap_px")
    under = int((gap < line).sum())
    said = {"frames": len(rows), "first_source_frame": points["first"],
            "face_height_px": {"least": round(float(face.min()), 1), "median": round(float(np.median(face)), 1), "most": round(float(face.max()), 1)},
            "lid_gap_px": {"least": round(float(gap.min()), 2), "median": round(float(np.median(gap)), 2), "most": round(float(gap.max()), 2)},
            "lid_gap_px_to_draw": line, "that_line_is": GAP_NOTE, "frames_under_it": under, "median_under_it": bool(np.median(gap) < line),
            "eyes": {"least": round(float(col("eyes").min()), 4), "median": round(float(np.median(col("eyes"))), 4), "most": round(float(col("eyes").max()), 4)},
            "score_lids_median": round(float(np.median(col("score_lids"))), 3), "score_lips_median": round(float(np.median(col("score_lips"))), 3)}
    band, e = h3_config.POSE_READING, col("eyes")
    shut, wide = int((e < band["eyes_shut_under"]).sum()), int((e > band["eyes_open_over"]).sum())
    said["eyes_against_the_starting_lines"] = {"under_the_shut_line": shut, "over_the_open_line": wide,
                                               "between_the_two_lines": len(rows) - shut - wide,
                                               "the_lines": [band["eyes_shut_under"], band["eyes_open_over"]],
                                               "which_are": LINES_NOTE}
    where = "" if points["first"] is None else f", source frames {points['first']} to {points['first'] + len(rows) - 1}"
    say(f"{len(rows)} frames{where}: face {said['face_height_px']['median']:.0f} px tall at the median ({said['face_height_px']['least']:.0f} to {said['face_height_px']['most']:.0f})")
    say(f"lash lines {said['lid_gap_px']['median']:.2f} px apart at the median ({said['lid_gap_px']['least']:.2f} to {said['lid_gap_px']['most']:.2f}); "
        f"{under} of {len(rows)} frames under {line} px, and the shot's median is {'under' if said['median_under_it'] else 'over'} it. That line is "
        f"{GAP_NOTE}.")
    say(f"eye opening: {shut} of {len(rows)} frames under the shut line ({band['eyes_shut_under']}), {wide} over the open line ({band['eyes_open_over']}), "
        f"{len(rows) - shut - wide} between them. {LINES_NOTE}.")
    say(f"eye opening {said['eyes']['least']:.4f} to {said['eyes']['most']:.4f}; lid score {said['score_lids_median']:.2f} and lip score "
        f"{said['score_lips_median']:.2f} at the median. A score does not say lids were seen; the lip gap is not a measure of a mouth.")
    if table:
        dst = Path(points_path).with_name(Path(points_path).stem.removesuffix("_points") + ".csv")
        with open(dst, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(fp.COLUMNS))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: ("" if r[k] is None else (r[k] if isinstance(r[k], int) else round(float(r[k]), 5))) for k in fp.COLUMNS})
        say(f"wrote {dst}")
        said["table"] = str(dst)
    return said


def spans(frames: list[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [(a, b) for a, b in out]


def size_arg(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = a.add_subparsers(dest="what", required=True)
    r = sub.add_parser("run", help="run the model along a shot (needs the card)")
    r.add_argument("--video", required=True); r.add_argument("--first", type=int, required=True, help="the first frame to read, by its number in the file")
    r.add_argument("--count", type=int, required=True); r.add_argument("--canvas", type=size_arg, required=True, help="1024x768")
    r.add_argument("--source-first", type=int, help="the source frame that the file's --first frame is; --first itself when left out (a source)")
    r.add_argument("--box-mask", default="", help="a mask video of the tracked person at the canvas's size; the whole frame when left out")
    r.add_argument("--box-mask-first", type=int, help="the source frame of the mask video's frame 0; the shot's first when left out")
    r.add_argument("--model-folder", required=True, help=f"where the model is: the store's folder named {h3_config.POSE_MODEL}")
    r.add_argument("--names", help="a json list of the points' names in the checkpoint's order, for a model that neither names them nor is the one vendor/meta_sapiens2/ lists")
    r.add_argument("--out", required=True, help="the points file to write (.npz); its record goes beside it")
    r.add_argument("--by", required=True, help="who is running it")
    d = sub.add_parser("read", help="say what a points file can carry (no card)")
    d.add_argument("points"); d.add_argument("--names"); d.add_argument("--table", action="store_true", help="also write one row a frame (.csv)")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    if v.what == "read":
        read(v.points, names=names, table=v.table)
        return 0
    if Path(v.model_folder).name != h3_config.POSE_MODEL:
        print(f"note: the folder is named {Path(v.model_folder).name}, and h3_config.POSE_MODEL is {h3_config.POSE_MODEL}")
    names, names_from = (names, f"--names {v.names}") if names else names_and_source(v.model_folder)      # refused here, before the model is loaded
    predict, names = load_model(v.model_folder, names)
    run(v.video, v.first, v.count, v.canvas, Path(v.out), v.by, predict, names, source_first=v.source_first, box_mask=v.box_mask,
        box_mask_first=v.box_mask_first, model=Path(v.model_folder).name, names_from=names_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
