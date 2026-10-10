"""The graphs of a masked job, as functions that return them.

A masked job is one clip with any number of subjects. It is run as a
no-sampling preview (who is where), then one load per subject per shot, then
a patch where a finished pass has a fault (the masking board,
`guide-order-of-operations-masked-job`). Until this file every masked session
wrote its own script for that, each patching a graph lifted out of an earlier
render, and a fix made in one script never reached the next.

Nothing here reads a file, a server or a clock: a function takes the job as
plain data and returns an API graph. `bench/masked_job.py` is the tool that
reads the job file, refuses what must be refused, writes the graphs and
queues them; `bench/check_masked_job.py` holds both. A value that is the
job's and not a node's default is a `MASKED_JOB_*` constant in `h3_config.py`,
with what earned it written beside it.

**The job block.** The job file is the loads file
`bench/capture_masked_run.py --loads` takes, with one more top-level key,
`job`, that the capture does not read:

    {"job": {"name": "kitchen",                  a new name for every run
             "source": "/abs/clip_24p.mov",      the every-frame 24 fps copy
             "first": 604, "frames": 447,        the span, in the copy's frames
             "canvas": [1024, 768],
             "output": "Video/<session>",        a prefix under ComfyUI's output folder
             "subjects": [{"label": "a", "pick": "largest", "pick_at": 664, ...}, ...]},
     "defaults": {...}, "loads": [...]}

Every frame number in it is a frame of the source copy. A subject's `pick`
is always given: which person a tracker starts from is the one choice no
default can make. `pick_at` names the source frame the pick is made on;
without it the tracker chooses. `phrase`, `max_people` and `corrections`
are the tracker's inputs of those names; `parts` is the list of the part
node's ticks the subject's passes will take, and without it the subject is
taken whole. `pose: false` leaves a subject's body pose and mesh out of the
preview, for a subject whose passes will use no motion video.

**The order of `subjects` is the order the trackers run in**, and it matters:
each tracker after the first is given every earlier tracker's mask on
`others`, so it cannot take a person an earlier one holds. Put the subject
most likely to be confused with another first.

Only the preview is built so far. The loads and the patch follow.
"""
from __future__ import annotations

import re

from h3_config import (
    FPS, SEGMENTER, SUBJECT_TRACK, SAPIENS2,
    MASKED_JOB_LOADER, MASKED_JOB_FORCE_RATE, MASKED_JOB_SEEK_BACK, MASKED_JOB_SAM_CORRECTIONS,
    MASKED_JOB_PARTS, MASKED_JOB_BODY_MODEL, MASKED_JOB_BOXES, MASKED_JOB_BODY_POSE, MASKED_JOB_MESH,
    MASKED_JOB_LOSSLESS, MASKED_JOB_PICTURE,
)

#: Ids the shipped masked row gives the same nodes (`build_workflows.build_api`, the `freeze_song_source`
#: branch), kept so a job's graph reads beside a shipped one. 101 and 200 are free in that row.
LOADER, SAM, SAM_CORRECTED, SAPIENS, BODY_MODEL = "28", "100", "101", "107", "200"
#: A subject's nodes sit in a block of their own: the first subject's from this id, each later one's a block
#: further on. Inside a block the tracker, the part node and the shot table end as they do in the shipped row
#: (105, 108, 109).
SUBJECT_BLOCK_FIRST, SUBJECT_BLOCK = 1000, 100
#: The part node's ticks: every boolean of `SUBJECT_PARTS` but its fill.
PART_TICKS = tuple(k for k, v in MASKED_JOB_PARTS.items() if isinstance(v, bool) and k != "hold_missing")
#: The outputs a job's graphs read, by the name each node gives them. `bench/node_id_manifest.json` has every
#: node's outputs in order, and `bench/check_masked_job.py` holds this table to it.
SLOT = {
    "MiniMaxH3SAM31Corrections": {"segmenter": 0, "segmenter_clip": 1, "report": 2},
    "MiniMaxH3SubjectTrack": {"mask": 0, "preview": 1, "report": 2, "shot_table": 3},
    "MiniMaxH3SubjectParts": {"parts": 0, "report": 3, "held": 4, "classes": 5},
    "MiniMaxH3SubjectBoxes": {"boxes": 0, "report": 1},
    "MiniMaxH3BodyPose": {"pose_data": 0, "report": 2},
    "MiniMaxH3BodyMeshVideo": {"frames": 0},
}
#: What a job or a subject may be named: the name goes into file names and into the capture's `--mask` spec,
#: where a comma, a colon or an equals sign is syntax.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*\Z")


def seek(frame: int) -> float:
    """The loader's `start_time`, in seconds, that makes `frame` of a 24 fps file the first frame it yields.

    The one place a job turns a frame number into a loader's time. A time on the frame itself depends on
    rounding inside ffmpeg; a time a little before it cannot land one off (`h3_config.MASKED_JOB_SEEK_BACK`)."""
    if frame < 0:
        raise ValueError(f"a loader cannot start at frame {frame}")
    return round(max(frame - MASKED_JOB_SEEK_BACK, 0.0) / FPS, 6)


def loader(path: str, first: int, frames: int, canvas: tuple[int, int], at: int = 0) -> dict:
    """A loader node on `frames` frames of a file from source frame `first`, cropped and scaled to the canvas.

    `at` is the source frame the file's own frame 0 is on: 0 for the source copy, a capture's first frame for
    one of its mask videos."""
    if first < at:
        raise ValueError(f"{path}: frame {first} is before the file's first frame, {at}")
    if frames < 1:
        raise ValueError(f"{path}: a load of {frames} frames")
    width, height = canvas
    return {"class_type": MASKED_JOB_LOADER,
            "inputs": {"video": path, "force_rate": MASKED_JOB_FORCE_RATE, "custom_width": width,
                       "custom_height": height, "frame_load_cap": frames, "start_time": seek(first - at),
                       "format": "AnimateDiff"}}


def check_job(job: dict) -> None:
    """Refuse a job block a graph cannot be built from, in words that name the key."""
    for key in ("name", "source", "first", "frames", "canvas", "output", "subjects"):
        if key not in job:
            raise ValueError(f"the job block lacks `{key}`")
    if not NAME.match(str(job["name"])):
        raise ValueError(f"job name {job['name']!r}: letters, digits and underscores only")
    if not str(job["source"]).startswith("/"):
        raise ValueError(f"source {job['source']!r}: give the file's whole path; the loader reads it as written")
    if len(job["canvas"]) != 2:
        raise ValueError(f"canvas {job['canvas']!r}: a width and a height")
    if job["first"] < 0 or job["frames"] < 1:
        raise ValueError(f"the span starts at {job['first']} and holds {job['frames']} frames")
    if not job["subjects"]:
        raise ValueError("the job names no subject")
    seen = set()
    for subject in job["subjects"]:
        label = str(subject.get("label", ""))
        if not NAME.match(label):
            raise ValueError(f"subject label {label!r}: letters, digits and underscores only")
        if label in seen:
            raise ValueError(f"two subjects are labelled {label!r}")
        seen.add(label)
        if not subject.get("pick"):
            raise ValueError(f"subject {label!r} has no `pick`: which person a tracker starts from is never a default")
        at = subject.get("pick_at")
        if at is not None and not job["first"] <= at < job["first"] + job["frames"]:
            raise ValueError(f"subject {label!r}: pick_at {at} is outside the span "
                             f"{job['first']}..{job['first'] + job['frames'] - 1}")
        unknown = set(subject.get("parts") or ()) - set(PART_TICKS)
        if unknown:
            raise ValueError(f"subject {label!r}: unknown parts {sorted(unknown)}; the part node ticks {PART_TICKS}")


def prefixes(job: dict, label: str) -> dict[str, str]:
    """Where a preview saves each thing it can make for one subject, as save-node prefixes under the output folder.

    Every subject has its own, so a second tracker's table cannot overwrite the first's. `track`, `parts`,
    `classes`, `shots` and `pose` are the capture's own words for a `--mask` spec. `held_things` is the part
    node's `held` output, the mask of what the subject holds: it is not a part with its doubted frames filled
    (the capture makes that one itself, `parts_held__<by>.mkv`), and it never goes on the capture's `held=`."""
    root = f"{job['output']}/{job['name']}/{job['name']}_{label}"
    return {key: f"{root}_{key}" for key in ("track", "parts", "classes", "held_things", "shots", "mesh")} | {"pose": root}


def _out(g: dict, nid: str, name: str) -> list:
    """A link to a node's output, by the output's name."""
    return [nid, SLOT[g[nid]["class_type"]][name]]


def _shown(g: dict, nid: int, source: list) -> None:
    """A report on a text preview, so it runs and its words reach the server's history."""
    g[str(nid)] = {"class_type": "PreviewAny", "inputs": {"source": source}}


def _tracker_inputs(job: dict, subject: dict) -> dict:
    inputs = dict(SUBJECT_TRACK, pick=subject["pick"])
    for theirs, ours in (("subject_phrase", "phrase"), ("max_people", "max_people"), ("corrections", "corrections")):
        if ours in subject:
            inputs[theirs] = subject[ours]
    if subject.get("pick_at") is not None:
        # a DynamicCombo: its member is written dotted in the API form, as the song node's `extent` is
        inputs["pick_on"] = "a frame I name"
        inputs["pick_on.pick_frame"] = subject["pick_at"] - job["first"]
    return inputs


def _save(g: dict, nid: int, images: list, prefix: str, how: dict) -> None:
    g[str(nid)] = {"class_type": "VHS_VideoCombine",
                   "inputs": {"images": images, "frame_rate": FPS, "filename_prefix": prefix, **how}}


def _save_mask(g: dict, nid: int, mask: list, prefix: str) -> None:
    g[str(nid)] = {"class_type": "MaskToImage", "inputs": {"mask": mask}}
    _save(g, nid + 1, [str(nid), 0], prefix, MASKED_JOB_LOSSLESS)


def preview(job: dict) -> dict:
    """The first graph of a job: who is where, with nothing sampled and no model of the render loaded.

    One loader on the span; the SAM loader and the pack's corrections of it; then for each subject, in the
    job's order, a Subject Track told who the earlier ones hold, its shot table saved under the subject's own
    prefix, the part node on its mask, the subject's box on each frame, a body pose from our own nodes with
    its table written, and the mesh drawn at the canvas. Every mask and class map is saved lossless, as the
    capture reads them (`prefixes`)."""
    check_job(job)
    canvas = tuple(job["canvas"])
    frames = [LOADER, 0]
    g = {LOADER: loader(job["source"], job["first"], job["frames"], canvas),
         SAM: {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SEGMENTER}},
         SAM_CORRECTED: {"class_type": "MiniMaxH3SAM31Corrections",
                         "inputs": {"segmenter": [SAM, 0], "segmenter_clip": [SAM, 1], **MASKED_JOB_SAM_CORRECTIONS}},
         SAPIENS: {"class_type": "MiniMaxH3Sapiens2Loader", "inputs": dict(SAPIENS2)},
         BODY_MODEL: {"class_type": "MiniMaxH3BodyModelLoader", "inputs": {"model_file": MASKED_JOB_BODY_MODEL}}}
    sam = {"segmenter": _out(g, SAM_CORRECTED, "segmenter"), "segmenter_clip": _out(g, SAM_CORRECTED, "segmenter_clip")}
    _shown(g, int(SAM_CORRECTED) + 1, _out(g, SAM_CORRECTED, "report"))
    held_by_earlier = earlier = None          # every earlier tracker's mask, joined; the last tracker's own
    for n, subject in enumerate(job["subjects"]):
        block = SUBJECT_BLOCK_FIRST + n * SUBJECT_BLOCK
        at = lambda k: str(block + k)         # noqa: E731
        save = prefixes(job, subject["label"])
        if n == 1:
            held_by_earlier = earlier
        elif n > 1:
            # one MASK a frame is what `others` takes: the join so far, and the tracker before this one
            g[at(3)] = {"class_type": "MaskComposite",
                        "inputs": {"destination": held_by_earlier, "source": earlier, "x": 0, "y": 0, "operation": "add"}}
            held_by_earlier = [at(3), 0]
        g[at(5)] = {"class_type": "MiniMaxH3SubjectTrack",
                    "inputs": {"frames": frames, **sam, **_tracker_inputs(job, subject),
                               **({"others": held_by_earlier} if held_by_earlier else {})}}
        earlier = track = _out(g, at(5), "mask")
        _shown(g, block + 6, _out(g, at(5), "report"))
        g[at(9)] = {"class_type": "MiniMaxH3SaveShotTable",
                    "inputs": {"shot_table": _out(g, at(5), "shot_table"), "preview": _out(g, at(5), "preview"),
                               "filename_prefix": save["shots"]}}
        ticks = set(subject.get("parts") or ())
        g[at(8)] = {"class_type": "MiniMaxH3SubjectParts",
                    "inputs": {"sapiens2": [SAPIENS, 0], "frames": frames, "subject_mask": track,
                               **MASKED_JOB_PARTS, **({k: k in ticks for k in PART_TICKS} if ticks else {})}}
        _shown(g, block + 7, _out(g, at(8), "report"))
        _save_mask(g, block + 20, track, save["track"])
        for k, output, name in ((22, "parts", "parts"), (24, "classes", "classes"), (26, "held", "held_things")):
            _save_mask(g, block + k, _out(g, at(8), output), save[name])
        if not subject.get("pose", True):
            continue
        g[at(40)] = {"class_type": "MiniMaxH3SubjectBoxes", "inputs": {"mask": track, **MASKED_JOB_BOXES}}
        _shown(g, block + 41, _out(g, at(40), "report"))
        g[at(42)] = {"class_type": "MiniMaxH3BodyPose",
                     "inputs": {"body_model": [BODY_MODEL, 0], "frames": frames, "boxes": _out(g, at(40), "boxes"),
                                **MASKED_JOB_BODY_POSE, "subject": subject["label"],
                                "first_source_frame": job["first"], "table_prefix": save["pose"]}}
        _shown(g, block + 43, _out(g, at(42), "report"))
        g[at(44)] = {"class_type": "MiniMaxH3BodyMeshVideo",
                     "inputs": {"pose_data": _out(g, at(42), "pose_data"), **MASKED_JOB_MESH,
                                "width": canvas[0], "height": canvas[1]}}
        _save(g, block + 45, _out(g, at(44), "frames"), save["mesh"], MASKED_JOB_PICTURE)
    return g
