#!/usr/bin/env python3
"""A keep-out of every person in a shot who is not one of the preview's subjects: the strangers.

    <python> bench/strangers.py people --preview data/<day>_<preview>_job --by SESSION
    <python> bench/strangers.py run --preview data/<day>_<preview>_job --model-folder <the store>/<h3_config.SAPIENS2["segmentation"]> --by SESSION
    <python> bench/strangers.py read <the strangers file's .json>

**What it is for.** A preview tracks the shot's subjects and nothing else, and its class map is the subject's
alone (the part model reads her with everyone else blanked). So nothing of a load knew where the other people in
a close group were. On the lead's 3167-3213 (2026-10-14) the plan's region lay on another woman's chin and face
and its margin on chins, mouths and eyes of others; on 3240-3276 a blind reader saw strangers in the margin.
Read before building in internal/claude/2026-10-14_mrdrill_stranger_read.md; ruled by the lead (form V3).

**What `run` writes.** Beside the strangers file, the part model's classes on the whole frame,
`<preview>_strangers_classes.npz` (`whole_frame_classes` in the record), which the capture reads for her unclaimed hair
(`capture_masked_run.whole_frame_hair`, 2026-10-22). The part model on the WHOLE frame, a frame at a time, on the preview's own frames of its
own source, fitted to its canvas as the lane's loader fits it: every pixel it calls human, and every pixel of another
person the detector found on that frame where the part model called nobody (2026-10-27: on 4647k whole-frame Sapiens2
missed the dark-haired bystander on 4686-4693 while SAM 3.1 found 2,000-3,500 px of her). Less each of the
preview's subjects' tracks (the files its record lists) grown by `h3_config.MASKED_STRANGERS["grow_px"]`, since the
part model reaches past the tracker's edge on a subject's own body, and with them each subject's held things and parts
as the preview saved them (`OWN_FILES`; the lead's rule of 2026-10-14: nothing of a subject's own is ever a stranger; on
1156b her track left out her black hat as a held thing, and the hat was a stranger). Of what is left, a piece that
touches a subject's own, grown, and holds no body core of its own (the part model's `CORE_CLASSES`: the face, hair,
torso, arms, legs and clothing, and not a hand, a hat or a prop) is that subject's, and is listed in the record with
its frame, size and box (`handed_back`; the lead's ruling (A) of 2026-10-14: her hat lifted in her hand at 1164 and her
mitt's fingertips lay outside her track and her held things; the core, not the face alone, on an outside reviewer's
point the same day, so another person's arm across her stays a stranger). Her own hair is the one core that can be hers
past the grow: a piece whose only core is Hair, joined hair pixel by hair pixel to the Hair inside her own, is hers (the
lead's ruling of 2026-10-14: her flying hair just outside her track was a stranger on 1097c and 1156c). And a piece
that touches a subject is a stranger only where it lies on ANOTHER PERSON THE DETECTOR FOUND (the owner's
rule of 2026-10-14, restated 2026-10-23: what a subject wears or holds is theirs, and a piece touching her with no body of
its own is hers). The people are SAM 3.1's on each frame, asked as the preview's own tracker asks, saved by `people`
beside the strangers file (`PEOPLE_ARE`); a detection more than `h3_config.MASKED_STRANGERS["subject_share"]` on a
subject's track is that subject, and every other is another person, held `hold_frames` either side as a track is
(`held_others`: a person the detector drops for a moment is still there). One file serves every subject's load, so a
subject's own detection is another person to every other subject: a piece is handed back only where, for EACH subject it
touches, it lies on no person but that subject (a piece of B's arm on B's detection touching A stays out of A's load). A
touching piece with a core of its own and no pixel on another person is handed back, `by` "on no other person the detector found" (shot_c_0014e, where she is alone: its
record's `kept_though_touching`, made before this rule, is mostly her own Upper_Clothing, and her hair edge with a few
face pixels, which flickered on second_hfp_0014g at 48 and 66-69). Each piece that
touches a subject and is kept is listed with the core classes that kept it and the people it lies on
(`kept_though_touching`). What is left is the strangers, written as a mask
video on the preview's frames (frame 0 the preview's first) to `bench/shot_to_load.py`'s `KEEP_OUTS_AT` as
`<preview>_strangers.mkv`, with a record beside it. One file a preview, for every load made from it. A file that is
there is not replaced: a preview made again is a new name and a new file. A frame where a subject's track is empty
(the tracker lost them) is held to the nearest frame with one when it is at most `hold_frames` away (`held_track`),
and left empty beyond, where a subject has left; the record lists both. A track empty on every frame, or not of the
canvas's size or length, is refused. The record holds a hash of the file's frames: the job step takes it only with a
record of the preview it is loading from that hashes its frames (`held_to_its_record`), and writes that hash into the
job file, where the chain holds the capture's copy to it (`load_chain.mask_specs`).

**How a load uses it.** A plan's `keep_outs` naming `strangers` makes the job step (`bench/shot_to_load.py job`)
look for this file, name it among the load's `others` (the capture's keep-outs: a cell a stranger touches is the
source's unless the subject's own mask has a pixel in it), and hold the load's part to its own subject's track grown
by the same px (the part model's pixels outside her track are no part of her: on 3167 her part lay on another
woman's face on 45 of 47 frames).

**What it cannot see.** A person the part model does not find (cut off at the frame's edge, behind hair) stays
regenerable. A person the detector does not find for longer than `hold_frames` either side of a frame: a piece of theirs
that touches a subject is handed back to her there and regenerated (the record lists every such piece with its core classes, so a look can find one). Whether a
stranger left as the source reads cleanly at the region's edge: that is the render's. The grow is a starting value from
two shots.

`people` and `run` need the card, and run under the card's lock as the chain runs them: `bench/shot_to_load.py`'s job
step runs `run` (and so `people`) where its load is owed a strangers file, only on an empty queue
(`load_chain.empty_queue`), and whoever holds the card runs them by hand only in a gap. Each loads its model in a process
of its own beside ComfyUI, which keeps its models on the card after a render, so each first asks the server to unload
them (`free_server`: `POST /free`, `load_chain.free_card`, at `--host`, the chain's `load_chain.HOST` by default) and says
so; a server that is not up is said and passed over. `people` exits before `run` loads the part model. `read` needs
none: it prints the record."""
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
#: What of each subject's own is taken out of the people found, beside the track: the files a preview saves under these keys
#: (`_<label>_<key>_`), when its record lists them. The lead's rule of 2026-10-14, on mrcircle's 1156b: nothing of a subject's
#: own is ever a stranger. Her track leaves out her black hat as a held thing, and the hat was a stranger at 1164.
OWN_FILES = ("held_things", "parts")
#: Said in the record, and held by the job step: a file made before this form is refused, and a fresh preview makes its own (`MADE_AGAIN`). A piece of what is left that
#: touches a subject's own, grown, and holds no body core of its own (`CORE_CLASSES`) is that subject's (`handed_back`; the
#: lead's ruling (A) of 2026-10-14: on 1156b the tracker left out her hat lifted in her hand at 1164 and her mitt's
#: fingertips, and her held-things file held none of them).
OWN_FORM = ("each subject's track, held things and parts, grown, from the part model's people and every other person the detector found where "
            "the part model found nobody; and a piece touching them with no body core of its own, or whose only core is hair meeting theirs, "
            "or that lies on no other person the detector found")
#: What a strangers file of another making takes (`held_to_its_record`): nothing remakes one in place (one file a preview, never
#: replaced), so a fresh preview, whose job step makes its own (`bench/shot_to_load.py::run_strangers`).
MADE_AGAIN = ("A file is never replaced: make the preview again, `bench/shot_to_load.py preview ... --remake` (the next letter), and its "
              "job step makes that preview's strangers file")
#: Why a piece was handed back to a subject, as the record says it (`handed_back`'s `by`).
NO_CORE, BY_HAIR, NO_OTHER = "no body core of its own", "its hair meets hers", "on no other person the detector found"
#: What the people file holds (`people` writes it, `run` reads it), carried in its record.
PEOPLE_ARE = ("every person SAM 3.1 found on each of the preview's frames, asked as the preview's own tracker asks (its graph's checkpoint, "
              "corrections, phrase, threshold and most people), on the preview's own frames of its own source fitted to its canvas as the lane's "
              "loader fits it: `masks` [detections, h, ceil(w / 8)] of uint8, each a detection bit-packed along the width (np.unpackbits, then the "
              "first w columns); `frame` the preview frame of each (0 the preview's first, `first`); `person` its number on that frame "
              "(shot_table.person_order: left to right by the centre column of its mask, ties to the higher one, counted from 1); `score` the "
              "detector's")
#: The tracker's node in a preview's graph, whose inputs say how it asked for people, and the nodes that made its SAM 3.1.
TRACKER_NODE, CHECKPOINT_NODE, CORRECTIONS_NODE = "MiniMaxH3SubjectTrack", "CheckpointLoaderSimple", "MiniMaxH3SAM31Corrections"
#: The class whose piece is the subject's own when it meets the subject's own hair (the lead's ruling of 2026-10-14).
HAIR_CLASS = "Hair"
#: The part model's classes a stranger is known by (`vendor/meta_sapiens2/seg_classes29.json`): a piece holding any of them
#: stays a stranger, touching a subject or not. The lead's list of 2026-10-14, on an outside reviewer's point: the face,
#: hair, torso, the arms and the clothing (on 3167 the other woman's chin and face lay against the lead's; a crowd's arm
#: across her stays a stranger). Added by mrdrill, reasoned: the legs, and the lips, teeth and tongue, as a body's own.
#: Not a core: a hand, a foot, a shoe or sock, Apparel (a hat, gloves, a belt) or Eyeglass, which a subject holds or wears
#: past her track (her hat and her mitt's fingertips on 1156b).
CORE_CLASSES = ("Face_Neck", "Hair", "Torso", "Left_Upper_Arm", "Left_Lower_Arm", "Right_Upper_Arm", "Right_Lower_Arm",
                "Upper_Clothing", "Lower_Clothing", "Left_Upper_Leg", "Left_Lower_Leg", "Right_Upper_Leg", "Right_Lower_Leg",
                "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")


class Refused(Exception):
    pass


def grown(masks: np.ndarray, px: int) -> np.ndarray:
    """Each frame's mask grown by `px` (a disc), [frames, h, w] of bool."""
    if px <= 0:
        return masks.copy()
    import cv2
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return np.stack([cv2.dilate(m.astype(np.uint8), disc).astype(bool) for m in masks])


def held_track(track: np.ndarray, reach: int | None = None) -> tuple[np.ndarray, list[int], list[int]]:
    """A track with each empty frame held to the nearest frame that has one, before it (or, at the start, after it),
    when that frame is at most `reach` frames away (`h3_config.MASKED_STRANGERS["hold_frames"]`); and the indices of
    the frames held, and of the frames left empty. A tracker that drops its subject for a moment would otherwise make
    them a stranger there and hold their part to nothing; a track empty for longer is a subject who left. Refused by
    the callers: a track empty on every frame."""
    reach = int(h3_config.MASKED_STRANGERS["hold_frames"]) if reach is None else reach
    out, held, empty = track.copy(), [], []
    has = track.reshape(len(track), -1).any(1)
    on = np.nonzero(has)[0]
    for i in np.nonzero(~has)[0]:
        before, after = on[on < i], on[on > i]
        src = before[-1] if len(before) else (after[0] if len(after) else None)
        if src is not None and abs(int(src) - int(i)) <= reach:
            out[i] = track[src]
            held.append(int(i))
        else:
            empty.append(int(i))
    return out, held, empty


def strangers_of(people: np.ndarray, tracks: list[np.ndarray], px: int) -> np.ndarray:
    """The people found on each frame less every subject's own (each mask in `tracks`: a track, held things, parts) grown by `px`."""
    out = people.copy()
    for t in tracks:
        out &= ~grown(t, px)
    return out


def handed_back(strangers: np.ndarray, own: dict[str, np.ndarray], core: np.ndarray, first: int, hair: np.ndarray | None = None,
                classes: np.ndarray | None = None, names: tuple | None = None, own_raw: dict | None = None,
                others: list | None = None) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Each frame's pieces of `strangers` (8-connected) that touch a subject's own grown (`own`: a label's mask, already
    grown) and hold no pixel of `core` (a body core, `CORE_CLASSES`): taken out, and listed with the subject, the frame,
    the size, the box and why (`by`). With `hair` (the whole frame's Hair class) and `own_raw` (each subject's own, not grown): a piece
    whose only core is Hair, joined by Hair pixels to the Hair inside a subject's own (through the grown band, not by touching
    it: a stranger's hair lying in the band is no hair of hers), is that subject's too (the lead's ruling of 2026-10-14: the lead's own flying hair
    just outside her track was a stranger on 1097c and 1156c). With `others` (a frame each: the people the detector found, as
    `others_of` and `held_others` give them, (number, mask, the frame found on, whose); a bare (number, mask) is another
    person on this frame): a piece with a core of its own is that subject's too where, for EVERY subject it touches, it lies
    on no person but that subject (the owner's rule, 2026-10-23: only another person is a stranger; one file serves every
    subject's load, so another subject's own detection is another person here). Also returned: each piece that
    touches a subject and is kept as a stranger, with the core classes that kept it (from `classes`, by `names`) and the people it
    lies on (`on_people`, their numbers on the frame, and `on_people_px`; `on_people_held`, [frame, number, px] of those held to
    it from another frame; `on_subjects`, a subject whose own detection it is), so a look sees why. A piece handed back on no
    other person says how much of it lies on her own detection (`on_her_detection_px`)."""
    from scipy import ndimage
    out, given, kept = strangers.copy(), [], []
    ring = np.ones((3, 3), bool)
    near = {k: ndimage.binary_dilation(v, structure=np.ones((1, 3, 3), bool)) for k, v in own.items()}
    for i in range(len(strangers)):
        parts, n = ndimage.label(strangers[i], structure=ring)
        for j in range(1, n + 1):
            piece = parts == j
            touches = [k for k, m in near.items() if (piece & m[i]).any()]
            if not touches:
                continue
            ys, xs = np.nonzero(piece)
            where = {"frame": first + i, "px": int(piece.sum()), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
            held = piece & core[i]
            by_hair = []
            if held.any() and hair is not None and own_raw is not None and not (held & ~hair[i]).any():
                # her own hair runs on past her track: the piece's hair joined, hair pixel by hair pixel through her grown band, to the
                # hair inside her own
                for k in touches:
                    joined, _ = ndimage.label(hair[i] & (piece | own[k][i]), structure=ring)
                    theirs = set(np.unique(joined[piece & hair[i]])) - {0}
                    if theirs & (set(np.unique(joined[own_raw[k][i] & hair[i]])) - {0}):
                        by_hair.append(k)
            if held.any() and classes is not None and names is not None:
                counts = np.bincount(classes[i][held], minlength=len(names))
                where["core"] = {names[c]: int(v) for c, v in enumerate(counts) if v}
            # (number, px on it, the frame it was found on, whose) for each person the piece lies on, there or held to it (`held_others`)
            lies = [] if others is None else [(o[0], int((piece & o[1]).sum()), o[2] if len(o) > 2 else i, o[3] if len(o) > 3 else None)
                                              for o in others[i]]
            lies = [x for x in lies if x[1]]
            # ONE FILE SERVES EVERY SUBJECT'S LOAD: another person to ANY subject the piece touches keeps it, a subject's own detection
            # being another person to every other subject (a piece of B's arm on B's detection touching A stays out of A's load)
            on = [x for x in lies if any(x[3] != k for k in touches)]
            if held.any() and not by_hair and (others is None or on):
                # a core of its own, and (where the detector's people are given) on another person: a stranger
                said = {} if others is None else {"on_people": [p for p, _, f, _ in on if f == i], "on_people_px": [v for _, v, f, _ in on if f == i],
                                                  **({"on_people_held": [[first + f, p, v] for p, v, f, _ in on if f != i]} if any(f != i for _, _, f, _ in on) else {}),
                                                  **({"on_subjects": sorted({w for *_, w in on if w})} if any(w for *_, w in on) else {})}
                kept.append({"touches": touches, **where, **said})
                continue
            who = by_hair or touches
            out[i] &= ~piece
            mine = sum(v for _, v, f, w in lies if f == i and w == who[0])
            given.append({"subject": who[0], **({"touches_also": who[1:]} if who[1:] else {}), **where,
                          "by": BY_HAIR if by_hair else (NO_OTHER if held.any() else NO_CORE),
                          **({"on_her_detection_px": mine} if others is not None and held.any() and not by_hair else {})})
    return out, given, kept


def said_back(given: list[dict]) -> str:
    """The pieces handed back, said for a look: by subject, the frames and the largest."""
    if not given:
        return "no piece was handed back to a subject"
    by = {}
    for g in given:
        by.setdefault(g["subject"], []).append(g)
    def why(v: list[dict]) -> str:
        counts = {r: sum(1 for g in v if g.get("by", NO_CORE) == r) for r in (NO_CORE, BY_HAIR, NO_OTHER)}
        return ", ".join(f"{c} {r}" for r, c in counts.items() if c)
    return "; ".join(f"{k}: {len(v)} piece(s) handed back on {len({g['frame'] for g in v})} frame(s) "
                     f"({', '.join(str(f) for f in sorted({g['frame'] for g in v})[:12])}{' ...' if len({g['frame'] for g in v}) > 12 else ''}), "
                     f"the largest {max(g['px'] for g in v)} px at {max(v, key=lambda g: g['px'])['frame']} in {max(v, key=lambda g: g['px'])['box']} "
                     f"({why(v)})"
                     for k, v in by.items())


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


def people_file_of(pjob: dict, output: Path) -> Path:
    """Where a preview's people file goes (`people_pass`): beside its strangers file."""
    return file_of(pjob, output).with_name(f"{pjob['name']}_people.npz")


def tracker_asks(folder: Path) -> dict:
    """How the preview's tracker asked SAM 3.1 for people, read from the graph its job folder holds (`preview.json`, the graph
    `bench/masked_job.py` queued): the checkpoint, the corrections wired to its segmenter (None for the stock model), the phrase,
    the threshold and the most people. Refused when the graph has no tracker, or trackers that ask differently."""
    graph = folder / "preview.json"
    if not graph.is_file():
        raise Refused(f"the preview's job folder {folder} holds no preview.json: how its tracker asked for people is read from the graph it queued")
    nodes = {k: v for k, v in json.loads(graph.read_text()).items() if isinstance(v, dict) and "class_type" in v}
    asked = set()
    for t in (v["inputs"] for v in nodes.values() if v["class_type"] == TRACKER_NODE):
        seg = t.get("segmenter")
        src = nodes.get(str(seg[0])) if isinstance(seg, list) and seg else None
        corr = None
        if src is not None and src["class_type"] == CORRECTIONS_NODE:
            corr = {k: v for k, v in src["inputs"].items() if not isinstance(v, list)}
            seg = src["inputs"].get("segmenter")
            src = nodes.get(str(seg[0])) if isinstance(seg, list) and seg else None
        if src is None or src["class_type"] != CHECKPOINT_NODE or any(isinstance(t.get(k), list) for k in ("subject_phrase", "detection_threshold", "max_people")):
            raise Refused(f"the tracker in {graph} takes its SAM 3.1 or its phrase from somewhere this tool does not read (not {CHECKPOINT_NODE}, "
                          f"through {CORRECTIONS_NODE} or not, with the phrase, threshold and most people typed in)")
        asked.add(json.dumps({"checkpoint": src["inputs"]["ckpt_name"], "corrections": corr, "phrase": t["subject_phrase"],
                              "threshold": float(t["detection_threshold"]), "max_people": int(t["max_people"])}, sort_keys=True))
    if len(asked) != 1:
        raise Refused(f"the graph {graph} has {'no tracker' if not asked else f'{len(asked)} trackers that ask for people differently'}: "
                      "the people file is asked for as the preview's tracker asked")
    return json.loads(asked.pop())


def sam_people(asks: dict):
    """SAM 3.1 as the preview's tracker asked it (`tracker_asks`): a function from one RGB picture [h, w, 3] of uint8 to the people
    on it, ([k, h, w] of bool in shot_table's person order, their scores), with `device` said on it. On the card when one is visible.
    Core is imported here and only here, so `run`, which loads the part model, never boots it (`people` runs in its own process)."""
    import importlib
    import types
    from _lib import bootstrap
    bootstrap(cpu=None)
    import comfy.model_management
    import torch
    from _lib.sam3_core import load_core
    from comfy_extras.nodes_sam3 import SAM3_Detect   # core's node, as the tracker calls it (`subject_track._sam_callables`)
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    track, table = importlib.import_module("_h3pack.subject_track"), importlib.import_module("_h3pack.shot_table")
    model, clip = load_core(asks["checkpoint"])
    if asks["corrections"] is not None:
        c = asks["corrections"]
        model, clip = importlib.import_module("_h3pack.sam31_corrections").corrected(
            model, clip, image_range=bool(c.get("correct_image_range", True)), text_activation=bool(c.get("correct_text_activation", True)),
            memory_selection=bool(c.get("memory_selection", True)))
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(track.counted(asks["phrase"], asks["max_people"])))

    def detect(pic: np.ndarray):
        with torch.inference_mode():
            got = SAM3_Detect.execute(model, torch.from_numpy(pic.astype(np.float32) / 255.0)[None], conditioning=cond,
                                      threshold=float(asks["threshold"]), individual_masks=True)
        masks, boxes = getattr(got, "args", got)[:2]
        masks = masks.to(torch.float32).cpu()
        scores = [float(b.get("score", 0.0)) for b in (boxes[0] if boxes else [])][:int(masks.shape[0])]
        order = table.person_order(masks)
        return (masks[order] > 0.5).numpy(), [scores[k] for k in order]
    detect.device = str(comfy.model_management.get_torch_device())
    return detect


def free_server(host: str, say=print) -> str:
    """Ask ComfyUI at `host` to unload its models and free the card (`load_chain.free_card`: `POST /free` with
    {"unload_models": true, "free_memory": true}, as the chain frees it before its pose pass) before this process puts a model
    on it, and say what was done. The server keeps its models on the card after a render (2026-10-23: it held them before a
    strangers pass), and `people` and `run` each load theirs in a process of their own beside it. A server that is not up, or
    that refuses, is said and passed over (the CPU, a check)."""
    import load_chain
    try:
        said = f"the card freed before the model loads (POST http://{host}/free, unload_models and free_memory): {load_chain.free_card(host)}"
    except (OSError, ValueError, KeyError) as e:       # urllib's URLError and HTTPError are OSErrors
        said = f"the server at {host} did not free the card ({type(e).__name__}: {e}): carrying on, as on the CPU or in a check"
    say(said)
    return said


def people_pass(preview: Path, output: Path, by: str, detect, asks: dict, say=print) -> dict:
    """Write a preview's people file (`PEOPLE_ARE`) and its record; the record. `detect` is `sam_people`'s function, or a stand-in."""
    pjob, _ = preview_of(preview)
    out = people_file_of(pjob, output)
    if out.exists():
        raise Refused(f"{out} is there: one people file a preview, never replaced. A preview made again has a new name")
    (w, h), first, n = pjob["canvas"], int(pjob["first"]), int(pjob["frames"])
    masks, frame, person, score, counts = [], [], [], [], []
    for i, pic in enumerate(stream(pjob["source"], (w, h), first, n, vf=FIT.format(w=w, h=h), pix="rgb24")):
        found, scores = detect(pic)
        if found.shape[1:] != (h, w) or len(scores) != len(found):
            raise Refused(f"the detector gave {len(found)} masks of {found.shape[1:]} and {len(scores)} scores on frame {first + i}, where the canvas "
                          f"is {w}x{h}: nothing written")
        masks += [np.packbits(m, axis=-1) for m in found]
        frame += [i] * len(found)
        person += list(range(1, len(found) + 1))
        score += [float(s) for s in scores]
        counts.append(len(found))
    if len(counts) != n:
        raise Refused(f"{pjob['source']} gave {len(counts)} frames from {first} where the preview has {n}: nothing written")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, masks=np.array(masks, np.uint8).reshape(len(masks), h, (w + 7) // 8), frame=np.array(frame, np.int32),
                        person=np.array(person, np.int32), score=np.array(score, np.float32), first=first)
    import hashlib
    record = {"what": PEOPLE_ARE, "asked_as": asks, "device": getattr(detect, "device", "a stand-in"), "preview": pjob["name"],
              "preview_folder": str(preview), "source": pjob["source"], "first_source_frame": first, "frames": n, "canvas": [w, h],
              "people_a_frame": counts, "duplicates_are": ("not joined: the tracker joins a person the detector returned twice before it numbers "
                                                          "(subject_track.one_detection_each), so a number here can differ from the shot table's there"),
              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "file": str(out),
              "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"wrote {out}: {n} frames from {first}, {sum(counts)} detections, at most {max(counts, default=0)} on a frame, on {record['device']}")
    return record


def people_on(pjob: dict, output: Path) -> tuple[list[list[tuple[int, np.ndarray]]], dict]:
    """A preview's people file as `people` wrote it: a frame each, the (person number, mask) of every detection; and its record.
    Refused when it is not there, is of another preview or other frames, or changed since its record was written."""
    import hashlib
    path = people_file_of(pjob, output)
    rec = path.with_suffix(".json")
    if not path.is_file() or not rec.is_file():
        raise Refused(f"{path} or its record is not there: `bench/strangers.py people --preview <the preview's job folder>` makes it, on the "
                      "card (`run` from the command line makes it first). Nothing written")
    r = json.loads(rec.read_text())
    (w, h), first, n = pjob["canvas"], int(pjob["first"]), int(pjob["frames"])
    for key, want in (("preview", pjob["name"]), ("first_source_frame", first), ("frames", n), ("canvas", [w, h])):
        if r.get(key) != want:
            raise Refused(f"the people file's record says {key} {r.get(key)!r}, and this preview's is {want!r}: nothing written")
    if hashlib.sha256(path.read_bytes()).hexdigest() != r.get("sha256"):
        raise Refused(f"{path.name} is not the file its record hashed: changed since it was made. Nothing written")
    got = np.load(path)
    found = np.unpackbits(got["masks"], axis=-1)[..., :w].astype(bool) if len(got["masks"]) else np.zeros((0, h, w), bool)
    out = [[] for _ in range(n)]
    for m, f, p in zip(found, got["frame"], got["person"]):
        out[int(f)].append((int(p), m))
    return out, r


def others_of(found: list, tracks: dict[str, np.ndarray], share: float) -> tuple[list[list[tuple[int, np.ndarray, int, str | None]]], dict[str, list[int]]]:
    """Each frame's people, each as (number, mask, the preview frame it was found on, whose it is): a subject's label for a
    detection with more than `share` of its pixels on that subject's track (`h3_config.MASKED_STRANGERS["subject_share"]`), None
    for another person; and, a subject each, the preview frames where their track holds them and no detection is theirs (the
    detector missed them, or gave them one mask with someone else). A subject's own detection is another person to every OTHER
    subject (the reviewer's point, 2026-10-23: B's arm past B's track, touching A, is kept out of A's load)."""
    out, unseen = [], {k: [] for k in tracks}
    for i, people_here in enumerate(found):
        here, mine = [], set()
        for p, m in people_here:
            area = int(m.sum())
            whose = [k for k, t in tracks.items() if area and (m & t[i]).sum() > share * area]
            mine |= set(whose)
            here.append((p, m, i, whose[0] if whose else None))
        out.append(here)
        for k, t in tracks.items():
            if t[i].any() and k not in mine:
                unseen[k].append(i)
    return out, unseen


def held_others(others: list, reach: int | None = None) -> list[list[tuple[int, np.ndarray, int, str | None]]]:
    """Each frame's people (`others_of`) with those of every frame within `reach` either side
    (`h3_config.MASKED_STRANGERS["hold_frames"]`, as `held_track` holds a track), each as `others_of` gives them (number, mask,
    the preview frame it was found on, whose): a person the detector drops for a moment is still another person there. On shot_c_4912g, SAM 3.1 run on
    the CPU (2026-10-23, its builder's replay) missed the dancer in blue passing in front of the third subject on 4936-4937,
    blurred, and found her on 4935 and 4938; held, her body there stays a stranger. The price: a piece where another person stood up to `reach` frames before or after stays
    a stranger too, as every touching piece with a core did before 2026-10-23."""
    reach = int(h3_config.MASKED_STRANGERS["hold_frames"]) if reach is None else reach
    n = len(others)
    return [[o for j in range(max(0, i - reach), min(n, i + reach + 1)) for o in others[j]] for i in range(n)]


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


def own_of(pjob: dict, files: list[str], output: Path) -> dict[str, list[Path]]:
    """Each subject's own files beside the track (`OWN_FILES`), as the preview's record lists them: none or one of each key,
    or refused. A preview made before a key was saved lists none of it."""
    out = {}
    for s in pjob["subjects"]:
        label, got = s["label"], []
        for key in OWN_FILES:
            found = [f for f in files if f"_{label}_{key}_" in Path(f).name and f.endswith(".mkv")]
            if len(found) > 1:
                raise Refused(f"the preview {pjob['name']} lists {len(found)} {key} videos for {label!r} where one is meant")
            got += [output / f for f in found]
        out[label] = got
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
                      ("hold_frames", int(h3_config.MASKED_STRANGERS["hold_frames"])), ("subject_share", float(h3_config.MASKED_STRANGERS["subject_share"])),
                      ("subjects_taken_out", tracks), ("file", str(path))):
        if r.get(key) != want:
            out.append(f"its record's {key} is {r.get(key)!r}, and this load's is {want!r}")
    if r.get("taken_out") != OWN_FORM:
        out.append(f"its record does not say it took out {OWN_FORM}: made before this form (2026-10-14, when a subject's own hat could be a "
                   f"stranger; 2026-10-23, when her own clothing past her track could be; 2026-10-27, when a bystander SAM found where Sapiens "
                   f"found nobody was left out). {MADE_AGAIN}")
    if not r.get("frame_hash"):
        out.append(f"its record holds no hash of its frames (made before 2026-10-14's second mend). {MADE_AGAIN}")
    elif not out:
        import load_chain
        if load_chain.frame_hash(str(path)) != r["frame_hash"]:
            out.append("its frames are not the ones its record hashed: changed since it was made")
    return out

def seg_model(folder: str):
    """The part model on the card: a function from one RGB picture [h, w, 3] to its class a pixel, [h, w] (0 is Background)."""
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
        return torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].to(torch.uint8).cpu().numpy()
    return predict


def run(preview: Path, output: Path, by: str, predict, model: str, say=print) -> dict:
    """Write a preview's strangers file and its record; the record."""
    pjob, files = preview_of(preview)
    out = file_of(pjob, output)
    if out.exists():
        raise Refused(f"{out} is there: one strangers file a preview, never replaced. A preview made again has a new name")
    tracks = tracks_of(pjob, files, output)
    (w, h), first, n = pjob["canvas"], int(pjob["first"]), int(pjob["frames"])
    held, gone, empty = {}, {}, {}
    for label, t in tracks.items():
        tw, th, tn = probe(str(t))
        if (tw, th) != (w, h) or tn < n:
            raise Refused(f"{label}'s track {t.name} is {tw}x{th} with {tn} frames, and the preview's canvas is {w}x{h} on {n} frames: nothing written")
        mask = read_mask(str(t), (w, h), first, first, n)[0]
        if not mask.any():
            raise Refused(f"{label}'s track {t.name} is empty on every frame: the tracker never held them, and they would be a stranger throughout. "
                          "Nothing written")
        held[label], h_i, e_i = held_track(mask)
        gone[label], empty[label] = [first + i for i in h_i], [first + i for i in e_i]
    # THE DETECTOR'S PEOPLE, read before the part model runs (the owner's rule, 2026-10-23: a piece touching a subject is a stranger
    # only on another person the detector found): refused here, with nothing run, where `people` has not made them
    found, people_rec = people_on(pjob, output)
    share = float(h3_config.MASKED_STRANGERS["subject_share"])
    others, undetected = others_of(found, held, share)
    del found
    own = own_of(pjob, files, output)
    owned = {label: held[label].copy() for label in held}
    for label, paths in own.items():
        for f in paths:
            fw, fh, fn = probe(str(f))
            if (fw, fh) != (w, h) or fn < n:
                raise Refused(f"{label}'s {f.name} is {fw}x{fh} with {fn} frames, and the preview's canvas is {w}x{h} on {n} frames: nothing written")
            owned[label] |= read_mask(str(f), (w, h), first, first, n)[0]
    sw, sh, _ = probe(pjob["source"])
    px = int(h3_config.MASKED_STRANGERS["grow_px"])
    import capture_masked_run as cap
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    core_ids = [names.index(c) for c in CORE_CLASSES]
    people, core, classes = np.zeros((n, h, w), bool), np.zeros((n, h, w), bool), np.zeros((n, h, w), np.uint8)
    got = 0
    for i, pic in enumerate(stream(pjob["source"], (w, h), first, n, vf=FIT.format(w=w, h=h), pix="rgb24")):
        c = predict(pic)
        det = np.zeros((h, w), bool)
        for _, m, _, who in others[i]:
            if who is None:
                det |= m & (c == 0)
        people[i], core[i], classes[i] = (c > 0) | det, np.isin(c, core_ids) | det, c
        got += 1
    if got != n:
        raise Refused(f"{pjob['source']} gave {got} frames from {first} where the preview has {n}: nothing written")
    strangers = strangers_of(people, list(owned.values()), px)
    hair_id = names.index(HAIR_CLASS)
    strangers, given, kept_touching = handed_back(strangers, {k: grown(v, px) for k, v in owned.items()}, core, first, hair=classes == hair_id,
                                                  classes=classes, names=names, own_raw=owned, others=held_others(others))
    moved = [g for g in given if g["by"] == NO_OTHER]
    by_frame = {}
    for g in moved:
        by_frame.setdefault(str(g["frame"]), []).append({k: g[k] for k in ("subject", "px", "box", "core", "on_her_detection_px")})
    # A DETECTION HOLDING HER AND SOMEONE ELSE hands that someone's pieces back with hers (the blind reader, 2026-10-23): what was handed
    # back on her own detection, apart from what lay on no one's, so a look can take the largest of each
    split = {name: [g for g in moved if (g["on_her_detection_px"] > 0) == on] for name, on in (("on_her_own_detection", True), ("on_no_detection", False))}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mask_video(out, strangers)
    # THE WHOLE FRAME'S CLASSES, kept (2026-10-22): the part model has read every person here, and the capture's her-hair-in-hole pass
    # looks for her hair that no one's mask holds near her (`capture_masked_run.whole_frame_hair`), which her own class map, of her crop
    # alone, cannot see where the preview did not read her
    whole = out.with_name(out.stem + "_classes.npz")
    np.savez_compressed(whole, classes=classes, first=first)
    import load_chain
    a = strangers.reshape(n, -1).sum(1)
    record = {"what": ("every person the part model or the detector finds on the whole frame, less each of the preview's subjects' own grown, a piece touching a "
                       "subject only where it lies on another person the detector found; bench/strangers.py"),
              "taken_out": OWN_FORM, "subjects_own_taken_out": {label: [str(f) for f in v] for label, v in own.items()},
              "core_classes": list(CORE_CLASSES), "hair_class": HAIR_CLASS, "handed_back": given, "handed_back_said": said_back(given),
              "kept_though_touching": kept_touching,
              "people_file": str(people_file_of(pjob, output)), "people_sha256": people_rec["sha256"], "people_asked_as": people_rec["asked_as"],
              "people_device": people_rec["device"], "subject_share": share,
              "subject_share_is": ("h3_config.MASKED_STRANGERS, inherited from subject_track.ON_OTHERS (measured 2026-10-10: one person held by two "
                                   "trackers lay 1.000 on the other's mask, two people never more than 0.039)"),
              "other_people_a_frame": [sum(1 for o in here if o[3] is None) for here in others],
              "frames_a_subject_has_no_detection": {k: [first + i for i in v] for k, v in undetected.items() if v},
              "handed_back_on_no_other_person": {"pieces": len(moved), "px": sum(g["px"] for g in moved), "by_frame": by_frame,
                                                 **{name: {"pieces": len(v), "px": sum(g["px"] for g in v), "largest": max((g["px"] for g in v), default=0),
                                                           "frames": sorted({g["frame"] for g in v})} for name, v in split.items()}},
              "whole_frame_classes": str(whole),
              "whole_frame_classes_are": "the part model's class a pixel on the whole frame (`classes` [frames, h, w], uint8, "
                                         "sapiens2_parts.CLASS_NAMES), frame 0 `first`, the preview's first frame; read by "
                                         "capture_masked_run.whole_frame_hair",
              "preview": pjob["name"], "preview_folder": str(preview), "source": pjob["source"], "source_size": [sw, sh],
              "source_fitted": "cropped to the canvas's shape and scaled, as the lane's loader fits it (_lib.frames.FIT)" if (sw, sh) != (w, h) else "the canvas's size",
              "first_source_frame": first, "frames": n, "canvas": [w, h], "model": model,
              "grow_px": px, "grow_is": "h3_config.MASKED_STRANGERS, a starting value measured on two shots",
              "hold_frames": int(h3_config.MASKED_STRANGERS["hold_frames"]),
              "subjects_taken_out": {label: str(t) for label, t in tracks.items()},
              "frames_a_track_is_empty_and_held": {k: v for k, v in gone.items() if v},
              "frames_a_track_is_empty_and_left_empty": {k: v for k, v in empty.items() if v},
              "empty_tracks_are": (f"held to the nearest frame with one, before (after, at the start), when it is at most "
                                   f"{h3_config.MASKED_STRANGERS['hold_frames']} frames away; left empty beyond, where the subject's own body is a stranger"),
              "figures": {"people_px_a_frame_median": int(np.median(people.reshape(n, -1).sum(1))), "strangers_px_a_frame_median": int(np.median(a)),
                          "strangers_px_most": int(a.max()), "frames_with_a_stranger": int((a > 0).sum())},
              "cannot_see": ["a person the part model does not find stays regenerable",
                             f"a person the detector does not find within {h3_config.MASKED_STRANGERS['hold_frames']} frames either side of a frame "
                             "(cut off at the edge, behind someone, too small, blurred): a piece of theirs touching a subject is handed back to her "
                             "there and regenerated (`handed_back_on_no_other_person` lists each by frame, "
                             "with the core classes it held)",
                             "one detection holding a subject and someone else together, mostly on her track: that someone's pieces touching her are "
                             "handed back with her own (counted, with their frames and the largest, under `handed_back_on_no_other_person`'s "
                             "`on_her_own_detection`, beside her own clothing her detection takes in; a look tells which); one mostly off her track "
                             "holding her: her own pieces on it stay strangers (`frames_a_subject_has_no_detection` lists where no detection is hers)",
                             "a piece of one subject touching another subject stays a stranger for both, since the one file serves both loads",
                             "a person in the shot the preview did not track (a subject the shots record is unsure of) is a stranger here",
                             f"a subject's own pixels more than {px} px off their track, held things and parts that touch no part of her grown own "
                             "(an arm the tracker cut off, across a gap) are a stranger here",
                             "a preview made before held things were saved: its subjects' held things are strangers here",
                             "another person's hand, hat or prop alone touching a subject, with no arm, face or clothing of theirs in the piece, is "
                             "handed back to her and regenerated (`handed_back` lists every such piece)",
                             "a piece of the subject's own body core past her track and grown (her arm beyond the grow) that lies on another person's "
                             f"detection, on its frame or within {h3_config.MASKED_STRANGERS['hold_frames']} frames of it, stays a stranger, but her hair: a piece whose only core is Hair that meets the Hair inside her grown own is hers",
                             "a stranger's hair touching hers, with no other core of theirs in the piece, is handed back to her and regenerated",
                             f"a piece of her own with a core, where another person stood within {h3_config.MASKED_STRANGERS['hold_frames']} frames "
                             "of it (`on_people_held`), stays a stranger",
                             "on a frame whose track is held, where the subject moved since",
                             "on a load whose job says another subject's track is not them on named frames (`--track-not`), the pixels under that "
                             "track there are taken out here as that subject's, and are no keep-out of the load: neither a stranger nor an other",
                             "how a stranger left as the source reads at the region's edge",
                             "frames_with_a_stranger counts a frame with a single pixel"],
              "frame_hash": load_chain.frame_hash(str(out)), "canvas_of_the_file": [w, h],
              "made_on_the_copy": "the preview's own source; a load renders from the video's present copy, which bench/shot_to_load.py holds to the "
                                  "same frame clock (`working_copy`)",
              "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "file": str(out)}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"wrote {out}: {n} frames from {first}; strangers on {record['figures']['frames_with_a_stranger']} of them, "
        f"{record['figures']['strangers_px_a_frame_median']} px a frame at the median and {record['figures']['strangers_px_most']} at most; "
        f"the subjects' tracks ({', '.join(tracks)}) and their own held things and parts ({sum(map(len, own.values()))} files) grown {px} px taken out; "
        f"handed back to them, touching their own: {said_back(given)}; {len(moved)} of those on no other person the detector found, "
        f"{sum(len(o) for o in others)} other people found over the frames")
    return record


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = a.add_subparsers(dest="step", required=True)
    p = sub.add_parser("people", help="the card: write a preview's people file, every person SAM 3.1 finds a frame, as its tracker asks")
    p.add_argument("--preview", required=True, help="the preview's job folder, data/<day>_<preview>_job")
    p.add_argument("--by", required=True, help="who is running it")
    p.add_argument("--output", default=None, help="the server's output folder; H3_OUTPUT_DIR when left out")
    import load_chain
    p.add_argument("--host", default=load_chain.HOST, help="the server asked to free the card first (`POST /free`), as the chain reaches it")
    r = sub.add_parser("run", help="the card: write a preview's strangers file (its people file first, in a process of its own, where it is missing)")
    r.add_argument("--preview", required=True, help="the preview's job folder, data/<day>_<preview>_job")
    r.add_argument("--model-folder", required=True, help=f"the store's folder named {h3_config.SAPIENS2['segmentation']}")
    r.add_argument("--by", required=True, help="who is running it")
    r.add_argument("--output", default=None, help="the server's output folder; H3_OUTPUT_DIR when left out")
    r.add_argument("--host", default=load_chain.HOST, help="the server asked to free the card first (`POST /free`), as the chain reaches it")
    d = sub.add_parser("read", help="print a strangers file's record")
    d.add_argument("record")
    v = a.parse_args(argv)
    try:
        if v.step == "read":
            print(json.dumps(json.loads(Path(v.record).read_text()), indent=1))
            return 0
        import os
        output = v.output or os.environ.get("H3_OUTPUT_DIR")
        if not output:
            raise Refused("where the server's output folder is was not said: set H3_OUTPUT_DIR or give --output")
        if v.step == "people":
            asks = tracker_asks(Path(v.preview))
            free_server(v.host)                         # before core touches the card
            detect = sam_people(asks)
            print(f"SAM 3.1 on {detect.device}, asked as the preview's tracker asked: {asks}")
            people_pass(Path(v.preview), Path(output), v.by, detect, asks)
            return 0
        pjob, _ = preview_of(Path(v.preview))
        if not file_of(pjob, Path(output)).exists() and not people_file_of(pjob, Path(output)).exists():
            # core is booted in a process of its own: this one loads the part model, and `capture_masked_run._pack` keeps core on its CPU path
            print(f"{people_file_of(pjob, Path(output))} is not there: the people pass first, in a process of its own")
            ran = subprocess.run([sys.executable, str(Path(__file__).resolve()), "people", "--preview", str(v.preview), "--by", v.by, "--output", str(output),
                                  "--host", v.host])
            if ran.returncode:
                raise Refused(f"the people pass ended with {ran.returncode}: no strangers file is written without the detector's people")
        free_server(v.host)                             # the people pass has exited; before the part model touches the card
        if Path(v.model_folder).name != h3_config.SAPIENS2["segmentation"]:
            print(f"note: the folder is named {Path(v.model_folder).name}, and h3_config.SAPIENS2['segmentation'] is {h3_config.SAPIENS2['segmentation']}")
        run(Path(v.preview), Path(output), v.by, seg_model(v.model_folder), Path(v.model_folder).name)
        return 0
    except Refused as e:
        print(f"REFUSED: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
