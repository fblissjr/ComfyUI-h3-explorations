#!/usr/bin/env python3
"""A keep-out of every person in a shot who is not one of the preview's subjects: the strangers.

    <python> bench/strangers.py run --preview data/<day>_<preview>_job --model-folder <the store>/<h3_config.SAPIENS2["segmentation"]> --by SESSION
    <python> bench/strangers.py read <the strangers file's .json>

**What it is for.** A preview tracks the shot's subjects and nothing else, and its class map is the subject's
alone (the part model reads her with everyone else blanked). So nothing of a load knew where the other people in
a close group were. On the lead's 3167-3213 (2026-10-14) the plan's region lay on another woman's chin and face
and its margin on chins, mouths and eyes of others; on 3240-3276 a blind reader saw strangers in the margin.
Read before building in internal/claude/2026-10-14_mrdrill_stranger_read.md; ruled by the lead (form V3).

**What `run` writes.** The part model on the WHOLE frame, a frame at a time, on the preview's own frames of its
own source, fitted to its canvas as the lane's loader fits it: every pixel it calls human. Less each of the
preview's subjects' tracks (the files its record lists) grown by `h3_config.MASKED_STRANGERS["grow_px"]`, since the
part model reaches past the tracker's edge on a subject's own body. What is left is the strangers, written as a mask
video on the preview's frames (frame 0 the preview's first) to `bench/shot_to_load.py`'s `KEEP_OUTS_AT` as
`<preview>_strangers.mkv`, with a record beside it. One file a preview, for every load made from it. A file that is
there is not replaced: a preview made again is a new name and a new file. A frame where a subject's track is empty
(the tracker lost them) is held to that track's last frame with one, as the pose pass holds its box, and the record
lists those frames: an empty track would make the subject a stranger to their own load. A track not of the canvas's
size is refused. The job step takes the file only with its record, held to the preview it is loading from
(`held_to_its_record`).

**How a load uses it.** A plan's `keep_outs` naming `strangers` makes the job step (`bench/shot_to_load.py job`)
look for this file, name it among the load's `others` (the capture's keep-outs: a cell a stranger touches is the
source's unless the subject's own mask has a pixel in it), and hold the load's part to its own subject's track grown
by the same px (the part model's pixels outside her track are no part of her: on 3167 her part lay on another
woman's face on 45 of 47 frames).

**What it cannot see.** A person the part model does not find (cut off at the frame's edge, behind hair) stays
regenerable. Whether a stranger left as the source reads cleanly at the region's edge: that is the render's. The
grow is a starting value from two shots.

`run` needs the card: whoever holds it runs it, in a gap. `read` needs none: it prints the record."""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "workflows"))

import h3_config  # noqa: E402
from _lib.frames import FIT, probe, read_mask, stream, write_mask_video  # noqa: E402

#: The preview's output key, as `bench/shot_to_load.py` writes it: the video's name is its second part.
OUTPUT_PARTS = 3


class Refused(Exception):
    pass


def grown(masks: np.ndarray, px: int) -> np.ndarray:
    """Each frame's mask grown by `px` (a disc), [frames, h, w] of bool."""
    if px <= 0:
        return masks.copy()
    import cv2
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return np.stack([cv2.dilate(m.astype(np.uint8), disc).astype(bool) for m in masks])


def held_track(track: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """A track with each empty frame held to the last frame before it that has one (the first after it, at the start),
    as the pose pass holds its box; and the indices of the frames held. A tracker that lost its subject on a frame
    would otherwise make the subject a stranger there, and hold the subject's part to nothing."""
    out, held = track.copy(), []
    has = track.reshape(len(track), -1).any(1)
    if not has.any():
        return out, list(range(len(track)))
    last = int(np.argmax(has))
    for i in range(len(track)):
        if has[i]:
            last = i
        else:
            out[i] = track[last]
            held.append(i)
    return out, held


def strangers_of(people: np.ndarray, tracks: list[np.ndarray], px: int) -> np.ndarray:
    """The people found on each frame less every subject's track grown by `px`."""
    out = people.copy()
    for t in tracks:
        out &= ~grown(t, px)
    return out


def preview_of(folder: Path) -> tuple[dict, list[str]]:
    """A preview's job block and the files its record lists; refused when it did not end in success."""
    for name in ("job.json", "preview.queued.json"):
        if not (folder / name).is_file():
            raise Refused(f"the preview's job folder {folder} holds no {name}")
    ran = json.loads((folder / "preview.queued.json").read_text())
    if ran.get("status") != "success":
        raise Refused(f"the preview in {folder} did not end in success (its record says {ran.get('status')!r})")
    return json.loads((folder / "job.json").read_text())["job"], list(ran.get("files") or [])


def file_of(pjob: dict, output: Path) -> Path:
    """Where a preview's strangers file goes: `shot_to_load.KEEP_OUTS_AT`, named by the preview."""
    from shot_to_load import KEEP_OUTS_AT
    parts = Path(pjob["output"]).parts
    if len(parts) != OUTPUT_PARTS or parts[0] != "Video":
        raise Refused(f"the preview's output is {pjob['output']!r}, not Video/<video>/<session>: the video's name is read from it")
    return output / KEEP_OUTS_AT.format(video=parts[1], preview=pjob["name"]) / f"{pjob['name']}_strangers.mkv"


def tracks_of(pjob: dict, files: list[str], output: Path) -> dict[str, Path]:
    """Each subject's track as the preview's record lists it: one each, or refused."""
    out = {}
    for s in pjob["subjects"]:
        label = s["label"]
        found = [f for f in files if f"_{label}_track_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {pjob['name']} lists {len(found)} track videos for {label!r} where one is meant")
        out[label] = output / found[0]
    return out


def held_to_its_record(path: Path, preview: str, first: int, frames: int, tracks: dict[str, str]) -> list[str]:
    """Why a strangers file is not the one this preview's load can take, from the record beside it: an empty list
    when the record says it was made of this preview, on its frames, with the grow `h3_config` has now, taking out
    the very tracks the preview lists. The file is not made again here (that needs the card); its record is held."""
    rec = path.with_suffix(".json")
    if not rec.is_file():
        return [f"{rec.name} is not there beside it: a strangers file is taken only with the record bench/strangers.py writes"]
    r = json.loads(rec.read_text())
    out = []
    for key, want in (("preview", preview), ("first_source_frame", first), ("frames", frames), ("grow_px", int(h3_config.MASKED_STRANGERS["grow_px"])),
                      ("subjects_taken_out", tracks), ("file", str(path))):
        if r.get(key) != want:
            out.append(f"its record's {key} is {r.get(key)!r}, and this load's is {want!r}")
    return out

def seg_model(folder: str):
    """The part model on the card: a function from one RGB picture [h, w, 3] to where it finds a person, [h, w] of bool."""
    import torch
    from transformers import Sapiens2ForSemanticSegmentation
    from transformers.models.sapiens2.image_processing_sapiens2 import Sapiens2ImageProcessor
    proc = Sapiens2ImageProcessor.from_pretrained(folder, local_files_only=True)
    net = Sapiens2ForSemanticSegmentation.from_pretrained(folder, local_files_only=True).eval().to("cuda")

    def predict(pic: np.ndarray) -> np.ndarray:
        h, w = pic.shape[:2]
        inputs = proc(images=[np.ascontiguousarray(pic)], return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(pixel_values=inputs["pixel_values"].to("cuda")).logits.float()
        return (torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0] > 0).cpu().numpy()
    return predict


def run(preview: Path, output: Path, by: str, predict, model: str, say=print) -> dict:
    """Write a preview's strangers file and its record; the record."""
    pjob, files = preview_of(preview)
    out = file_of(pjob, output)
    if out.exists():
        raise Refused(f"{out} is there: one strangers file a preview, never replaced. A preview made again has a new name")
    tracks = tracks_of(pjob, files, output)
    (w, h), first, n = pjob["canvas"], int(pjob["first"]), int(pjob["frames"])
    for label, t in tracks.items():
        tw, th, tn = probe(str(t))
        if (tw, th) != (w, h) or tn < n:
            raise Refused(f"{label}'s track {t.name} is {tw}x{th} with {tn} frames, and the preview's canvas is {w}x{h} on {n} frames: nothing written")
    sw, sh, _ = probe(pjob["source"])
    px = int(h3_config.MASKED_STRANGERS["grow_px"])
    people = np.zeros((n, h, w), bool)
    got = 0
    for i, pic in enumerate(stream(pjob["source"], (w, h), first, n, vf=FIT.format(w=w, h=h), pix="rgb24")):
        people[i] = predict(pic)
        got += 1
    if got != n:
        raise Refused(f"{pjob['source']} gave {got} frames from {first} where the preview has {n}: nothing written")
    held, empty = {}, {}
    for label, t in tracks.items():
        held[label], gone = held_track(read_mask(str(t), (w, h), first, first, n)[0])
        empty[label] = [first + i for i in gone]
    strangers = strangers_of(people, list(held.values()), px)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mask_video(out, strangers)
    a = strangers.reshape(n, -1).sum(1)
    record = {"what": "every person the part model finds on the whole frame, less each of the preview's subjects' tracks grown; bench/strangers.py",
              "preview": pjob["name"], "preview_folder": str(preview), "source": pjob["source"], "source_size": [sw, sh],
              "source_fitted": "cropped to the canvas's shape and scaled, as the lane's loader fits it (_lib.frames.FIT)" if (sw, sh) != (w, h) else "the canvas's size",
              "first_source_frame": first, "frames": n, "canvas": [w, h], "model": model,
              "grow_px": px, "grow_is": "h3_config.MASKED_STRANGERS, a starting value measured on two shots",
              "subjects_taken_out": {label: str(t) for label, t in tracks.items()},
              "frames_a_track_is_empty_and_held": {k: v for k, v in empty.items() if v},
              "empty_tracks_are": "held to the last frame before with one (the first after, at the start), as the pose pass holds its box",
              "figures": {"people_px_a_frame_median": int(np.median(people.reshape(n, -1).sum(1))), "strangers_px_a_frame_median": int(np.median(a)),
                          "strangers_px_most": int(a.max()), "frames_with_a_stranger": int((a > 0).sum())},
              "cannot_see": ["a person the part model does not find stays regenerable",
                             "a person in the shot the preview did not track (a subject the shots record is unsure of) is a stranger here",
                             f"a subject's own pixels more than {px} px off their track (an arm the tracker cut off) are a stranger here",
                             "on a frame whose track is held, where the subject moved since",
                             "how a stranger left as the source reads at the region's edge",
                             "frames_with_a_stranger counts a frame with a single pixel"],
              "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "file": str(out)}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"wrote {out}: {n} frames from {first}; strangers on {record['figures']['frames_with_a_stranger']} of them, "
        f"{record['figures']['strangers_px_a_frame_median']} px a frame at the median and {record['figures']['strangers_px_most']} at most; "
        f"the subjects' tracks ({', '.join(tracks)}) grown {px} px taken out")
    return record


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = a.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run", help="the card: write a preview's strangers file")
    r.add_argument("--preview", required=True, help="the preview's job folder, data/<day>_<preview>_job")
    r.add_argument("--model-folder", required=True, help=f"the store's folder named {h3_config.SAPIENS2['segmentation']}")
    r.add_argument("--by", required=True, help="who is running it")
    r.add_argument("--output", default=None, help="the server's output folder; H3_OUTPUT_DIR when left out")
    d = sub.add_parser("read", help="print a strangers file's record")
    d.add_argument("record")
    v = a.parse_args()
    try:
        if v.step == "read":
            print(json.dumps(json.loads(Path(v.record).read_text()), indent=1))
            return 0
        import os
        output = v.output or os.environ.get("H3_OUTPUT_DIR")
        if not output:
            raise Refused("where the server's output folder is was not said: set H3_OUTPUT_DIR or give --output")
        if Path(v.model_folder).name != h3_config.SAPIENS2["segmentation"]:
            print(f"note: the folder is named {Path(v.model_folder).name}, and h3_config.SAPIENS2['segmentation'] is {h3_config.SAPIENS2['segmentation']}")
        run(Path(v.preview), Path(output), v.by, seg_model(v.model_folder), Path(v.model_folder).name)
        return 0
    except Refused as e:
        print(f"REFUSED: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
