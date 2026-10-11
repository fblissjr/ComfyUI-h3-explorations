#!/usr/bin/env python3
"""Run the pose model on every frame of a shot and keep its points: what the eye marks and the reader are made from.

    python bench/pose_along_shot.py run --video SOURCE.mov --first 1575 --count 155 --canvas 1024x768 \\
        --box-mask TRACK.mkv --box-mask-first 1575 --model-folder <the store>/<h3_config.POSE_MODEL> \\
        --names NAMES.json --out data/<job>/lead_1575_points.npz --by <session>
    python bench/pose_along_shot.py read data/<job>/lead_1575_points.npz [--table]

`run` needs the card: whoever holds it runs it. It reads the frames it is told to, by number; fits a source to
the canvas the way the lane's loader does (a clip already the canvas's size is left as it is); gives the model
one box a frame, the tracked person's from a named mask video or the whole frame; and writes the points, the
scores, the picture's size, the points' NAMES and the source frame of row 0. A record goes beside it
(`.json`). It derives no figure: those have one definition, `bench/_lib/face_points.py`.

THE NAMES. Every tool finds a point by its name, so a points file carries them. The checkpoint in the store
does not name its points (its config calls them `LABEL_0` on), so they are given with `--names`: a json list
in the checkpoint's order. The list used so far was read out of Meta's own keypoint config for this model and
is an untracked file; where it came from and what its order rests on is in
`bench/results/2026-10-11_sapiens2_pose_first_frames.md`. A model folder that does name its points needs no
`--names`, and one that does not is refused without it.

A render is read the same way. Its frame 0 is not source frame 0: `--source-first` says which source frame the
clip's `--first` frame is, and that is what the file records.

`read` needs no card. It says what a shot's points can and cannot carry before anything is drawn from them or
read by them: the face's height and the gap between the lash lines in the picture's pixels, beside
`h3_config.POSE_READING["lid_gap_px_to_draw"]`; how many frames sit under, over and between the two lines the
eye opening was found readable at; and the model's own scores with the reminder that a score does not say
lids were seen. It prints figures with what they were measured on and against what; it gives no verdict, and
what a shot is good for is the reader's to say. `--table` writes one row a frame beside the points (`.csv`).

WHAT IT CANNOT KNOW
- Whether the box holds the right person. It takes the mask it is given.
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


def point_names(folder: str, config=None) -> list[str]:
    """The points' names, from the model's own files. A config whose labels are placeholders is refused: a points file
    with made-up names would be read by index in all but word."""
    labels = getattr(config, "id2label", None) or json.loads((Path(folder) / "config.json").read_text()).get("id2label", {})
    names = [labels[k] for k in sorted(labels, key=int)]
    if names and not all(n.startswith("LABEL_") for n in names):
        return names
    for extra in ("keypoint_names.json", "names.json"):
        f = Path(folder) / extra
        if f.exists():
            return list(json.loads(f.read_text()))
    raise SystemExit(f"{folder} does not name its points (its config's labels are placeholders and it holds no names file): give --names, "
                     "a json list in the checkpoint's order")


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


def run(video: str, first: int, count: int, canvas: tuple[int, int], out: Path, by: str, predict, names: list[str], *,
        source_first: int | None = None, box_mask: str = "", box_mask_first: int | None = None, model: str = "", say=print) -> dict:
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
    else:
        boxes, held = [[0.0, 0.0, float(w), float(h)]] * count, []
    points, scores = [], []
    for f, picture in enumerate(stream(video, (w, h), first, count, vf=FIT.format(w=w, h=h) if fitted else "", pix="rgb24")):
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
        "model": model or h3_config.POSE_MODEL,
        "made_from": {"video": str(video), "its_frames": [first, first + count - 1], "which_are_source_frames": [source_first, source_first + count - 1],
                      "fitted_to_the_canvas": ("as the loader fits a source: cut to the canvas's shape at the centre, then scaled" if fitted
                                               else "no: it is the canvas's size"),
                      "the_filter_it_was_read_through": FIT.format(w=w, h=h) if fitted else "",
                      "box": ({"mask_video": str(box_mask), "its_first_frame_is_source_frame": source_first if box_mask_first is None else box_mask_first}
                              if box_mask else "the whole frame")},
        "size": [w, h], "frames": count,
        "frames_with_no_mask_that_kept_the_box_before": [source_first + f for f in held],
        "not_known": ["whether the box holds the right person: the mask is taken as given",
                      "whether the points are right: the model places and scores lid points on the back of a head and on a face a few pixels tall"],
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"wrote {out} ({count} frames, {len(names)} points a frame, row 0 is source frame {source_first}) and {out.with_suffix('.json').name}")
    if held:
        say(f"{len(held)} of {count} frames had no mask and kept the box of the frame before")
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
            "lid_gap_px_to_draw": line, "frames_under_it": under, "median_under_it": bool(np.median(gap) < line),
            "eyes": {"least": round(float(col("eyes").min()), 4), "median": round(float(np.median(col("eyes"))), 4), "most": round(float(col("eyes").max()), 4)},
            "score_lids_median": round(float(np.median(col("score_lids"))), 3), "score_lips_median": round(float(np.median(col("score_lips"))), 3)}
    band, e = h3_config.POSE_READING, col("eyes")
    shut, wide = int((e < band["eyes_shut_under"]).sum()), int((e > band["eyes_open_over"]).sum())
    said["eyes_against_the_starting_lines"] = {"under_the_shut_line": shut, "over_the_open_line": wide,
                                               "between_them_where_the_figure_was_not_readable": len(rows) - shut - wide,
                                               "the_lines": [band["eyes_shut_under"], band["eyes_open_over"]],
                                               "which_are": "read off one close-up of one video: a starting value, not this clip's own"}
    where = "" if points["first"] is None else f", source frames {points['first']} to {points['first'] + len(rows) - 1}"
    say(f"{len(rows)} frames{where}: face {said['face_height_px']['median']:.0f} px tall at the median ({said['face_height_px']['least']:.0f} to {said['face_height_px']['most']:.0f})")
    say(f"lash lines {said['lid_gap_px']['median']:.2f} px apart at the median ({said['lid_gap_px']['least']:.2f} to {said['lid_gap_px']['most']:.2f}); "
        f"{under} of {len(rows)} frames under the {line} px a mark can be drawn from or a render read by"
        + (": THE SHOT SITS UNDER THE LINE" if said["median_under_it"] else ""))
    say(f"eye opening: {shut} of {len(rows)} frames under the shut line ({band['eyes_shut_under']}), {wide} over the open line ({band['eyes_open_over']}), "
        f"{len(rows) - shut - wide} between them, where the figure was not readable. The two lines are one close-up's: a starting value.")
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
    r.add_argument("--names", help="a json list of the points' names, when the model's folder does not name them")
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
    predict, names = load_model(v.model_folder, names)
    run(v.video, v.first, v.count, v.canvas, Path(v.out), v.by, predict, names, source_first=v.source_first, box_mask=v.box_mask,
        box_mask_first=v.box_mask_first, model=Path(v.model_folder).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
