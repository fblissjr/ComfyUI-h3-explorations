"""A clip's shots as a table a person can review once: who was found in each, who was taken, and why.

The Subject Track (`subject_track.py`) follows one person through a clip with
cuts and already knows, per shot, everything a reviewer needs: where the shot
starts and ends, the people the detector found on the frame it judged, which
of them it took as the subject or why it took nobody. Until now that lived in
a paragraph of report text and an unnumbered tile. This module turns it into
one table, in three forms that say the same thing:

- a dict, written as JSON (`build`, `as_json`): what another node reads;
- text, a markdown table with one row per shot (`as_text`): what a person
  reads;
- numbers on the preview tile's outlines (`label_points`), so a row's
  "person 2" is a person one can point at.

**What it buys** (the owner's pick, 2026-10-05, card `use-shot-table`): the
masked lane scales to many clips when a person reads one generated table per
clip and corrects the rows that are wrong, where today the only way to see a
wrong shot is to watch a render. It applies wherever the Subject Track runs.
It is accepted when a reviewer can say from the table and the tile alone
which shots are right, and when "shot 4: person 2" means the same person in
the table, on the tile and in a correction typed into the tracker.

**The one rule both share: how people are numbered** (`person_order`). On
the frame a shot is shown on, person K is the K-th detection **from left to
right by the centre column of its mask**, ties going to the higher one,
counted from 1. Left to right is what a person reads off a picture, and it
does not depend on the order the detector happened to return. A correction
resolves "person K" through the same function, so the tile, the table and
the correction cannot disagree. The shot number is the one the tracker's
report and tile already print: shots in order, from 1.

**What a row holds.** The shot's first and last frame; the frame it is shown
on; each person there with a number, a box, the share of the frame the mask
covers and the detector's score; the subject's state (the tracker's own
words: picked, taken, taken as the only person, absent), which person that
is, the frame it was taken on, the similarity and a sentence of why; whether
the subject is on screen and for how many of the shot's frames, counted off
the mask; for a shot left empty, the person who came closest, which is who a
correction would most likely name; `corrected`, what a correction typed
into the tracker said about the shot ("person 2", "none"), empty when there
was none (the tracker's `Shot.corrected`, read when it exists); and
`caption`, an empty slot for the per-shot caption a later node writes.

**What it does not hold**: a similarity for every person. The tracker keeps
only each shot's best, and this module reads its results without changing
how it follows anyone.

`MiniMaxH3SaveShotTable` writes the three forms beside a render, for a review
graph. It is an output node, so wiring it makes core run the tracker on that
queue; a render graph that relies on the Masked Source's kept mask leaves it
out, as it leaves the tracker's preview unwired.

The table's functions are pure, over tensors and the tracker's dataclasses
read by attribute: nothing here imports the tracker or a model.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io, ui
from PIL import Image

#: Bumped when a field changes meaning or leaves; a reader checks it.
TABLE_VERSION = 1

#: The numbering rule in words, carried in every table so a file read alone
#: says how its person numbers were assigned.
NUMBERING = ("person K is the K-th detection on the shot's shown frame, left to right by the "
             "centre column of its mask, ties to the higher one, counted from 1")

#: What a shot's `track_score` holds, carried in every table.
TRACK_SCORE_IS = ("the tracker's own score that its object is on the frame, per frame of the shot, as the tracker's "
                  "number: over 0 it takes the object as present. null where no track was made or it could not be read")

#: The file endings `MiniMaxH3SaveShotTable` writes after `<prefix>_NNNNN`.
SUFFIX_JSON, SUFFIX_TEXT, SUFFIX_SHEET = "_shots.json", "_shots.md", "_shots.png"


# ------------------------------------------------------------------ numbering

def _box(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """(x, y, width, height) of a [H, W] mask's covered pixels, or None when it is empty."""
    on = mask > 0.5
    rows, cols = on.any(dim=1), on.any(dim=0)
    if not bool(rows.any()):
        return None
    ys, xs = torch.nonzero(rows).flatten(), torch.nonzero(cols).flatten()
    x, y = int(xs[0]), int(ys[0])
    return x, y, int(xs[-1]) - x + 1, int(ys[-1]) - y + 1


def _centre(mask: torch.Tensor) -> tuple[float, float]:
    """(column, row) of the centre of a mask's covered pixels; an empty mask sorts last."""
    on = (mask > 0.5).to(torch.float32)
    total = float(on.sum())
    if total <= 0:
        return float("inf"), float("inf")
    cols = torch.arange(on.shape[1], dtype=torch.float32)
    rows = torch.arange(on.shape[0], dtype=torch.float32)
    return float((on.sum(dim=0) * cols).sum() / total), float((on.sum(dim=1) * rows).sum() / total)


def person_order(masks: torch.Tensor) -> list[int]:
    """Detection indices in the order people are numbered: left to right, ties to the higher one.

    `masks` is [N, H, W]. `person_order(masks)[K - 1]` is the detection that
    is person K, and the list is empty when nothing was detected.
    """
    centres = [_centre(masks[i]) for i in range(int(masks.shape[0]))]
    return sorted(range(len(centres)), key=lambda i: (round(centres[i][0], 3), round(centres[i][1], 3), i))


def person_number(masks: torch.Tensor, index: int | None) -> int | None:
    """The number of detection `index` among `masks`, or None when there is no such detection."""
    if index is None:
        return None
    order = person_order(masks)
    return order.index(int(index)) + 1 if int(index) in order else None


def detection_of(masks: torch.Tensor, person: int) -> int | None:
    """The detection index of person number `person`, or None when the frame has no such person."""
    order = person_order(masks)
    return order[person - 1] if 1 <= int(person) <= len(order) else None


def label_points(masks: torch.Tensor) -> list[tuple[int, float, float]]:
    """(person number, column, row) for each detection: where its number is drawn, in the frame's pixels.

    The point is the top of the mask's box at its centre column, so a number
    sits on the head of the outline it names.
    """
    points = []
    for number, index in enumerate(person_order(masks), 1):
        box = _box(masks[index])
        if box is None:
            continue
        points.append((number, _centre(masks[index])[0], float(box[1])))
    return points


# ------------------------------------------------------------------ the table

def _why(shot, state: str, match: float, pick: str, phrase: str, named_frame: bool) -> str:
    corrected = str(getattr(shot, "corrected", "") or "")
    if corrected:
        # A person's word outranks every automatic reason: the similarity the
        # tracker saw is no longer why the shot is as it is.
        where = f" on frame {shot.seed}" if shot.seed is not None else ""
        return f"corrected by hand: {corrected}{where}"
    if shot.picked:
        how = "the frame named" if named_frame else "chosen automatically"
        return f"the {pick} `{phrase}` on frame {shot.seed}, {how}"
    if shot.seed is not None and getattr(shot, "lone", False):
        return f"the only person on frame {shot.seed}, taken whatever it scores ({shot.best:.2f})"
    if shot.seed is not None:
        return f"similarity {shot.best:.2f} on frame {shot.seed}, at or above the line {match:.2f}"
    if shot.best >= 0:
        return f"best similarity {shot.best:.2f} on frame {shot.shown}, under the line {match:.2f}"
    if not getattr(shot, "seen", 0):
        return "no detection in this shot"
    return f"up to {shot.seen} detection(s), none that could be compared"


def _runs(flags: list[bool], first: int) -> list[list[int]]:
    """The runs of True in `flags` as [first frame, last frame], numbered from `first`."""
    runs, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([first + start, first + i - 1])
            start = None
    return runs


def build(found, detect: Callable[[int], tuple[torch.Tensor, list[float]]], mask: torch.Tensor, *,
          state: Callable[[object], str], phrase: str, pick: str, named_frame: bool, named_value: bool,
          cuts: list[int], left_out: dict[int, int] | None = None, on_others: list[int] | None = None,
          track_scores: dict | None = None) -> dict:
    """The table for one clip, from the tracker's result.

    `found` is the tracker's `Followed`, `detect` its detector callable (the
    frames it already looked at are cached there, so this detects nothing
    again), `mask` the assembled [frames, H, W] mask and `state` the tracker's
    own word for a shot. The rest is what the user set, for the header.

    Three things a tracker may hand in, each None when it has none (added
    2026-10-10, none changes a field that was there): `left_out`, for a
    tracker given the people others hold, {frame looked at: detections left
    out as theirs}; `on_others`, the frames where this track's own mask lies
    mostly on those people; `track_scores`, {frame: the tracker's own score
    there or None}, with its reasons under "trouble".
    """
    theirs = set(on_others or [])
    frames, height, width = int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])
    present = (mask > 0.5).flatten(1).any(dim=1)
    rows = []
    for number, shot in enumerate(found.shots, 1):
        masks, scores = detect(shot.shown)
        people = []
        for person, index in enumerate(person_order(masks), 1):
            box = _box(masks[index])
            people.append({
                "person": person,
                "box": list(box) if box else None,
                "share_of_frame": round(float((masks[index] > 0.5).sum()) / float(height * width), 4),
                "detector_score": round(float(scores[index]), 3) if index < len(scores) else None,
            })
        word = state(shot)
        taken = shot.seed is not None
        candidate = person_number(masks, shot.index)
        with_subject = int(present[shot.start:shot.end].sum())
        # inside a shot the subject is in, the frames the mask leaves empty: a track let go and not found again,
        # or a subject out of view. Empty for a shot with no subject at all, which `on_screen` already says.
        without = _runs((~present[shot.start:shot.end]).tolist(), int(shot.start)) if with_subject else []
        rows.append({
            "shot": number,
            "first_frame": int(shot.start),
            "last_frame": int(shot.end) - 1,
            "frames": int(shot.end) - int(shot.start),
            "shown_frame": int(shot.shown),
            "people": people,
            "subject": {
                "state": word,
                "person": candidate if taken else None,
                "seed_frame": int(shot.seed) if taken else None,
                "similarity": None if shot.picked or shot.best < 0 else round(float(shot.best), 3),
                "why": _why(shot, word, float(found.match), pick, phrase, named_frame),
            },
            "closest_person": None if taken else candidate,
            # mrorange's corrections set this on the tracker's `Shot`; a
            # tracker without them has no such field and the row says so.
            "corrected": str(getattr(shot, "corrected", "") or ""),
            "on_screen": with_subject > 0,
            "frames_with_subject": with_subject,
            "frames_without_subject": without,
            # each time the tracker seeded its track again inside the shot; a tracker without that has no such field
            "regained": [{"lost_from_frame": int(a), "seed_frame": int(f), "detections": int(n),
                          "similarity": round(float(score), 3), "next_person": None if second < 0 else round(float(second), 3)}
                         for a, f, n, score, second in getattr(shot, "regained", [])],
            "probes_after_a_loss": [{"frame": int(f), "detections": int(n), "best": round(float(best), 3),
                                     "next_person": None if second < 0 else round(float(second), 3)}
                                    for f, n, best, second in getattr(shot, "probes", [])],
            "gallery_frames": [int(f) for f in getattr(shot, "gallery", [])],
            "caption": "",
            # with the `others` input wired: detections left out as another tracker's person, on the shown frame and
            # on every frame looked at in the shot; and the frames this track's own mask lies mostly on such a person
            "others_left_out": None if left_out is None else {
                "on_shown_frame": int(left_out.get(int(shot.shown), 0)),
                "on_frames_looked_at": int(sum(v for f, v in left_out.items() if shot.start <= f < shot.end))},
            "frames_on_others": None if on_others is None else _runs(
                [f in theirs for f in range(int(shot.start), int(shot.end))], int(shot.start)),
            # one value per frame of the shot, null where the tracker made no track or its score could not be read
            "track_score": None if track_scores is None else [
                None if track_scores.get(f) is None else round(float(track_scores[f]), 2)
                for f in range(int(shot.start), int(shot.end))],
        })
    return {
        "table": "h3 shot table",
        "version": TABLE_VERSION,
        "frames": frames,
        "size": [width, height],
        "subject_phrase": phrase,
        "pick": pick,
        "pick_frame": None if found.pick_frame is None else int(found.pick_frame),
        "pick_frame_named": bool(named_frame),
        "match": round(float(found.match), 3),
        "match_named": bool(named_value),
        "cuts": [int(c) for c in cuts],
        "numbering": NUMBERING,
        "track_score_is": TRACK_SCORE_IS,
        "track_score_trouble": [] if track_scores is None else [str(x) for x in track_scores.get("trouble", [])],
        "shots": rows,
        # the subject as the picked shot's own track showed them, for a later run to recognise them by
        # (`gallery_of`): the frames, and per frame one signature per place compared, null where there was none
        "gallery": {"frames": [int(f) for f in getattr(found, "gallery_frames", [])],
                    "signatures": [[None if v is None else [round(float(x), 5) for x in v.flatten().tolist()] for v in views]
                                   for views in getattr(found, "gallery", [])],
                    "handed_in": int(getattr(found, "gallery_given", 0)),
                    "pick_probes": [{"frame": int(f), "detections": int(n), "best": round(float(best), 3),
                                     "next_person": None if second < 0 else round(float(second), 3)}
                                    for f, n, best, second in getattr(found, "pick_probes", [])]},
    }


def gallery_of(table: dict) -> list[tuple]:
    """The gallery a shot table carries, as the tracker takes it: one tuple of signatures per frame. Empty when it has none."""
    out = []
    for views in (table.get("gallery") or {}).get("signatures") or []:
        out.append(tuple(None if v is None else torch.tensor(v, dtype=torch.float32) for v in views))
    return out


def gallery_from(source: str) -> list[tuple]:
    """The gallery of an earlier run, from its shot table: the table's JSON itself, or the path of a `_shots.json`.

    A path that is not absolute is looked for under ComfyUI's output folder, where the Save Shot Table node and
    the song node write it. A table with no gallery, one written before the tracker kept one or by a run that
    picked nobody, is refused by name: handing in nothing would silently be today's pick rule.
    """
    text = str(source or "").strip()
    where = "the shot table wired in"
    if not text.startswith("{"):
        path = text if os.path.isabs(text) else os.path.join(folder_paths.get_output_directory(), text)
        if not os.path.isfile(path):
            raise ValueError(f"subject_from: no shot table at {path}")
        where = path
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    try:
        table = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"subject_from: {where} is not a shot table's JSON ({error})") from None
    out = gallery_of(table) if isinstance(table, dict) else []
    if not out:
        raise ValueError(f"subject_from: {where} carries no gallery of the subject. It was written before the Subject "
                         "Track kept one, or by a run that picked nobody: run that Subject Track again")
    return out


def as_json(table: dict) -> str:
    return json.dumps(table, indent=1)


def as_text(table: dict) -> str:
    """One markdown row per shot. Reads as plain text too."""
    lines = [
        f"{len(table['shots'])} shot(s) in {table['frames']} frames; subject: the {table['pick']} "
        f"`{table['subject_phrase']}`" + ("" if table["pick_frame"] is None else f", picked on frame {table['pick_frame']}"),
        "People are numbered left to right on each shot's shown frame.",
        "",
        "| shot | frames | shown | people | subject | on screen | why | caption |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table["shots"]:
        subject = row["subject"]
        if subject["person"] is not None:
            who = f"person {subject['person']}, {subject['state']}"
        elif row["closest_person"] is not None:
            who = f"{subject['state']} (closest: person {row['closest_person']})"
        else:
            who = subject["state"]
        if row["corrected"]:
            who += f" (corrected: {row['corrected']})"
        seen = (f"{row['frames_with_subject']} of {row['frames']} frames" if row["on_screen"] else "no")
        gaps = row.get("frames_without_subject") or []
        if gaps:
            seen += "; none on " + ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in gaps[:6]) + (
                f", and {len(gaps) - 6} more runs" if len(gaps) > 6 else "")
        lines.append(
            f"| {row['shot']} | {row['first_frame']}-{row['last_frame']} | {row['shown_frame']} | "
            f"{len(row['people'])} | {who} | {seen} | {subject['why']} | {row['caption']} |")
    return "\n".join(lines)


# ------------------------------------------------------------------- the node

def sheet(tiles: torch.Tensor):
    """The preview tiles, [shots, h, w, 3], stacked into one picture top to bottom."""
    tall = torch.cat([t for t in tiles[..., :3].to(torch.float32).cpu()], dim=0)
    return Image.fromarray((tall.clamp(0, 1) * 255).round().to(torch.uint8).numpy().astype(np.uint8))


def write(table_json: str, tiles: torch.Tensor | None, folder: str, stem: str) -> list[str]:
    """Write `<stem>_shots.json`, `.md` and, with tiles, `.png` into `folder`. Returns the file names."""
    table = json.loads(table_json)
    if table.get("table") != "h3 shot table":
        raise ValueError("`shot_table` is not a shot table: wire the Subject Track's `shot_table` output here")
    if table.get("version") != TABLE_VERSION:
        raise ValueError(f"this shot table is version {table.get('version')!r} and this node writes version "
                         f"{TABLE_VERSION}: run the Subject Track again")
    os.makedirs(folder, exist_ok=True)
    names = [stem + SUFFIX_JSON, stem + SUFFIX_TEXT]
    with open(os.path.join(folder, names[0]), "w", encoding="utf-8") as fh:
        fh.write(as_json(table) + "\n")
    with open(os.path.join(folder, names[1]), "w", encoding="utf-8") as fh:
        fh.write(as_text(table) + "\n")
    if tiles is not None:
        names.append(stem + SUFFIX_SHEET)
        sheet(tiles).save(os.path.join(folder, names[2]))
    return names


def write_beside(source: dict | None, folder: str, stem: str) -> list[str]:
    """Write the table a Masked Source's bundle carries next to a render named `stem`; the names, or [].

    The song node calls this after it writes its video. The bundle's table
    came with the mask, tracked or kept (`video_mask.py`), so a render that
    spared the tracker still gets its table. No tiles: a kept mask keeps the
    table, not the tracker's preview.
    """
    text = (source or {}).get("shot_table") or ""
    return write(text, None, folder, stem) if text else []


class MiniMaxH3SaveShotTable(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SaveShotTable",
            display_name="MiniMax H3 Save Shot Table",
            category="model/latent/minimax",
            description=(
                "Writes the Subject Track's shot table beside a render: the table as JSON and as text, "
                "and the numbered preview as one picture. For a review graph: wiring it makes the "
                "tracker run on that queue, so a render graph that keeps its mask leaves it out."),
            is_output_node=True,
            inputs=[
                io.String.Input("shot_table", force_input=True,
                                tooltip="The Subject Track's `shot_table` output."),
                io.Image.Input("preview", optional=True,
                               tooltip="The Subject Track's `preview` output. Its tiles are saved as one "
                                       "picture, each outline carrying the person number the table uses."),
                io.String.Input("filename_prefix", default="Video/h3_shots",
                                tooltip="Where the files go under the output folder, as a save node's prefix. "
                                        "Give it the render's prefix to put the table beside the render."),
            ],
            outputs=[io.String.Output(display_name="text", tooltip="The table as text, one row per shot.")],
        )

    @classmethod
    def execute(cls, shot_table, filename_prefix="Video/h3_shots", preview=None) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        names = write(shot_table, preview, folder, f"{filename}_{counter:05}")
        text = as_text(json.loads(shot_table))
        shown = ui.PreviewText(text).as_dict()
        if preview is not None:
            shown["images"] = [{"filename": names[2], "subfolder": subfolder, "type": "output"}]
        return io.NodeOutput(text, ui=shown)
