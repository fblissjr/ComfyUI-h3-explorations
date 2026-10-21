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
without it the tracker chooses. `phrase`, `max_people`, `corrections` and
`seeds` are the tracker's inputs of those names (`seeds`: which person to
follow in a shot, as a box on one frame of the load, in the node's own
form; `bench/cast_file.py seeds` writes it from a person's answers); `match` is a number, the likeness
at or above which a person in another shot is taken as the subject, and
without it the tracker sets its own line (`TRACKER_MATCH_AT` has why a job
names one); `parts` is the list of the part
node's ticks the subject's passes will take, and without it the subject is
taken whole. `pose: false` leaves a subject's body pose and mesh out of the
preview, for a subject whose passes will use no motion video.

**The order of `subjects` is the order the trackers run in**, and it matters:
each tracker after the first is given every earlier tracker's mask on
`others`, so it cannot take a person an earlier one holds. Put the subject
most likely to be confused with another first.

**Keys of the job block and of a subject that the graphs do not read and `bench/load_chain.py` does** (2026-10-13):
`sam_memory_selection` (2026-10-17), read by `preview` alone: absent, or `{"on": true, ...}`, the trackers run with SAM
3.1's memory selection (`MASKED_JOB_SAM_CORRECTIONS`); `{"on": false, "why": a person's words, "said_by": who}` turns
it off for this preview, and an off with no reason is refused (`check_job`). The graph always sets the input, so the
job's graph records which ran. `preview`, the job folder of the preview a load's masks come from (`data/<day>_<preview>_job`), from whose own job
the source frame the masks begin on and the subjects' order are read; `who_is_who`, a who-is-who file beside the
job file to read in place of making one; and on a subject, the options of its mask in a capture, passed only
when set: `pieces`, `least`, `hold`, `drop`, `grade` (`bench/capture_masked_run.py` has what each does).
`reference`, written by `bench/shot_to_load.py`: the form of the load's head reference (`drawn`, or `plain` with
a person's why), who said so, the record and the points it was made from, and the face's figures along the load.

**`two_shots`, the one key by which a load may hold a cut** (2026-10-13; `two_shots_of`, `check_load`): two
neighbouring shots of the source in ONE window, on purpose. `{"cut": the source frame the second shot begins on,
"shots": [[first, last], [first, last]], "why": a person's words, "said_by": who}`, with whatever else its writer
keeps beside them (`bench/two_shot_load.py` writes it, from two ordinary loads' job files). It is not a default
and nothing sets it for anybody: a job without it is held to one shot a load, as ever
(`bench/masked_job.py::ONE_SHOT`). A job with it is held to ONE load of ONE subject that holds the named cut,
lies inside the two shots, runs as one window, and is neither zoomed nor a continuation; its text has to
describe each shot under its own header (`TWO_SHOT_HEADERS`).

**A load** is one entry of the file's `loads`, with the file's `defaults` under it: one subject on one
stretch of frames, as a rule one shot. It is written in the capture's words, since the capture reads the same
entry as a plan (`first`, `frames`, `subject`, `others`, `margin`, `edge`, `carried`, `text`, `audio`,
`still`), and the builder reads a few more that the capture carries without using: `seed`, `window`,
`context`, `feather`, `motion` (`mesh` or `none`; without it a whole-subject pass of a subject
with a pose pass takes the mesh and any other load takes none), `continue_from`, `context_noise`,
`audio_first` (the source frame an audio file starts on), `parts_mask` (a mask video to take as the part, in
place of the capture's own), `motion_edge` (the motion video's short edge), `motion_file_2` with `motion_first_2`
and `motion_edge_2` (a second motion video beside the first, shown to the model as the next video reference: for a
load whose text gives each video a subject of its own), `subjects` (a load of TWO subjects in one pass, below), `hair_band` (for a part pass: a share of the part's
height left open under it for the still's hair, cut out of one keep-out, which `hair_band_cuts` names when it is
not the clothing; `bench/masked_job.py::hair_band`, `h3_config.MASKED_JOB_HAIR_BAND`), and `render_region`: `part` renders the load zoomed in on what is
replaced and lays it back (`RENDER_REGIONS`; for a part that is small in the frame; such a load takes a motion
video only as a `motion_file` of its own, which the node shows as it is and does not cut to the box), where `frame`, the
default, renders the whole frame. A default that was measured on one clip is a starting value for
another: every one of them can be set in a load or in the file's `defaults`, and the build record says
which were (`SETTINGS`). `carried` says what a pass takes: `track` is the whole subject, `held` the
part as the capture filled and graded it. One job file makes a whole-person pass or a part pass of the same
subject by changing that key alone.

`load` makes one graph for a load, from the shipped masked graph (`h3_config.MASKED_JOB_BASE_GRAPH`) and the
files of a checked capture: nothing is tracked again. With `plan` it is the same graph with the song node on
`preview`, which samples nothing and writes what each window will be given. The patch follows.
"""
from __future__ import annotations

import copy
import re

from h3_config import (
    FPS, MAX_LENGTH, SEGMENTER, SUBJECT_TRACK, SAPIENS2, MASKED_SOURCE, MASKED_PARTS_SOURCE, MASKED_LOAD_COMPOSITE,
    MASKED_JOB_CONTEXT, MASKED_JOB_CONTEXT_NOISE, MASKED_JOB_SOL_SINK, MASKED_JOB_MOTION, MASKED_JOB_HAIR_BAND,
    MASKED_JOB_CRF, MASKED_JOB_EDGE, MASKED_JOB_STILL_VIEW,
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
    "MiniMaxH3SubjectTrack": {"mask": 0, "preview": 1, "report": 2, "shot_table": 3, "head_mask": 4},
    "MiniMaxH3SubjectParts": {"parts": 0, "report": 3, "held": 4, "classes": 5},
    "MiniMaxH3SubjectBoxes": {"boxes": 0, "report": 1},
    "MiniMaxH3BodyPose": {"pose_data": 0, "report": 2},
    "MiniMaxH3BodyMeshVideo": {"frames": 0},
}
#: The Subject Track's choice for "match at the value I give", whose member holds the value. A job names a
#: value because the tracker's automatic line comes with a rule of its own: a tracker that finds its subject
#: alone on its pick frame takes any lone person in a later shot whatever the likeness, and a tracker that
#: runs after others sees its pick frame with their people set aside, so it can be "alone" beside somebody.
#: Naming a value turns that rule off. The values a job would name are the node's own two floors,
#: `subject_track.MATCH_FLOOR` and, for a tracker whose pick frame shows it alone, `subject_track.PLAIN_FLOOR`.
#: (2026-10-10, since e9de8cc3: with `others` wired the node switches that rule off itself, `subject_track.py::follow`,
#: `alone`, so "alone beside somebody" can no longer happen to a tracker that runs after others. A job names a
#: value for its FIRST tracker, the only one that still has the rule; naming one for a later tracker is harmless.)
TRACKER_MATCH_AT = "at a value"
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
    memory_selection_of(job)
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
        match = subject.get("match")
        if match is not None and (isinstance(match, bool) or not isinstance(match, (int, float)) or not 0 < match <= 1):
            raise ValueError(f"subject {label!r}: match {match!r}; a likeness above 0 and up to 1, or leave it out "
                             f"for the tracker's own line")
        if "track_head" in subject and not isinstance(subject["track_head"], bool):
            raise ValueError(f"subject {label!r}: track_head {subject['track_head']!r}; a boolean")
        track_head_of(job, subject)
        unknown = set(subject.get("parts") or ()) - set(PART_TICKS)
        if unknown:
            raise ValueError(f"subject {label!r}: unknown parts {sorted(unknown)}; the part node ticks {PART_TICKS}")


def memory_selection_of(job: dict) -> bool:
    """Whether a job's trackers run with SAM 3.1's memory selection: the chain's default, or off with a person's reason."""
    said = job.get("sam_memory_selection")
    if said is None:
        return bool(MASKED_JOB_SAM_CORRECTIONS["memory_selection"])
    if not isinstance(said, dict) or not isinstance(said.get("on"), bool):
        raise ValueError(f"sam_memory_selection {said!r}: {{\"on\": true}}, or {{\"on\": false, \"why\": ..., \"said_by\": ...}}")
    if not said["on"] and not (str(said.get("why") or "").strip() and str(said.get("said_by") or "").strip()):
        raise ValueError("sam_memory_selection is off with no `why` or no `said_by`: SAM 3.1's memory selection is the "
                         "chain's default, and it is turned off only on a person's word, with the reason")
    return said["on"]


def track_head_of(job: dict, subject: dict) -> bool:
    """Whether a subject's tracker runs SAM 3.1 head propagation (`track_head`): the chain's default, or off with a reason."""
    said = job.get("sam_head_track")
    if said is not None:
        if not isinstance(said, dict) or not isinstance(said.get("on"), bool):
            raise ValueError(f"sam_head_track {said!r}: {{\"on\": true}}, or {{\"on\": false, \"why\": ..., \"said_by\": ...}}")
        if not said["on"] and not (str(said.get("why") or "").strip() and str(said.get("said_by") or "").strip()):
            raise ValueError("sam_head_track is off with no `why` or no `said_by`: the SAM 3.1 head track is the "
                             "chain's default, and it is turned off only on a person's word, with the reason")
    if "track_head" in subject:
        return bool(subject["track_head"])
    if said is not None:
        return said["on"]
    return bool(SUBJECT_TRACK["track_head"])


def prefixes(job: dict, label: str) -> dict[str, str]:
    """Where a preview saves each thing it can make for one subject, as save-node prefixes under the output folder.

    Every subject has its own, so a second tracker's table cannot overwrite the first's. `track`, `head_track`,
    `parts`, `classes`, `shots` and `pose` are the capture's own words for a `--mask` spec. `held_things` is the part
    node's `held` output, the mask of what the subject holds: it is not a part with its doubted frames filled
    (the capture makes that one itself, `parts_held__<by>.mkv`), and it never goes on the capture's `held=`."""
    root = f"{job['output']}/{job['name']}/{job['name']}_{label}"
    return {key: f"{root}_{key}" for key in ("track", "head_track", "parts", "classes", "held_things", "shots", "mesh")} | {"pose": root}


def _out(g: dict, nid: str, name: str) -> list:
    """A link to a node's output, by the output's name."""
    return [nid, SLOT[g[nid]["class_type"]][name]]


def _shown(g: dict, nid: int, source: list) -> None:
    """A report on a text preview, so it runs and its words reach the server's history."""
    g[str(nid)] = {"class_type": "PreviewAny", "inputs": {"source": source}}


def _tracker_inputs(job: dict, subject: dict) -> dict:
    inputs = dict(SUBJECT_TRACK, pick=subject["pick"], track_head=track_head_of(job, subject))
    # `seeds` (2026-10-13): which person a tracker follows in a shot, as a box on one frame of the load, in the node's
    # own form (`bench/cast_file.py seeds` writes it from a person's answers). Until then a job could not carry one.
    for theirs, ours in (("subject_phrase", "phrase"), ("max_people", "max_people"), ("corrections", "corrections"), ("seeds", "seeds")):
        if ours in subject:
            inputs[theirs] = subject[ours]
    members = {}
    if subject.get("pick_at") is not None:
        # a DynamicCombo: its member is written dotted in the API form, as the song node's `extent` is, and
        # after its parent
        inputs["pick_on"] = "a frame I name"
        members["pick_on.pick_frame"] = subject["pick_at"] - job["first"]
    if subject.get("match") is not None:
        inputs["match"] = TRACKER_MATCH_AT
        members["match.match_threshold"] = float(subject["match"])
    return inputs | members


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
                         "inputs": {"segmenter": [SAM, 0], "segmenter_clip": [SAM, 1], **MASKED_JOB_SAM_CORRECTIONS,
                                    "memory_selection": memory_selection_of(job)}},
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
        if track_head_of(job, subject):
            _save_mask(g, block + 28, _out(g, at(5), "head_mask"), save["head_track"])
        for k, output, name in ((22, "parts", "parts"), (24, "classes", "classes"), (26, "held", "held_things")):
            _save_mask(g, block + k, _out(g, at(8), output), save[name])
        if not subject.get("pose", True):
            continue
        g[at(40)] = {"class_type": "MiniMaxH3SubjectBoxes", "inputs": {"mask": track, **MASKED_JOB_BOXES}}
        _shown(g, block + 41, _out(g, at(40), "report"))
        g[at(42)] = {"class_type": "MiniMaxH3BodyPose",
                     "inputs": {"body_model": [BODY_MODEL, 0], "frames": frames, "boxes": _out(g, at(40), "boxes"),
                                "subject_mask": track, **MASKED_JOB_BODY_POSE, "subject": subject["label"],
                                "first_source_frame": job["first"], "table_prefix": save["pose"]}}
        _shown(g, block + 43, _out(g, at(42), "report"))
        g[at(44)] = {"class_type": "MiniMaxH3BodyMeshVideo",
                     "inputs": {"pose_data": _out(g, at(42), "pose_data"), **MASKED_JOB_MESH,
                                "width": canvas[0], "height": canvas[1]}}
        _save(g, block + 45, _out(g, at(44), "frames"), save["mesh"], MASKED_JOB_PICTURE)
    return g


# ---- a load: one subject on one stretch, from a checked capture ----------------

#: What a load's `motion` may say, and the Masked Source's choice for each.
MOTIONS = {"none": MASKED_JOB_MOTION["none"], "mesh": MASKED_JOB_MOTION["wired"]}
#: The settings of a load a job file may set in place of the builder's default; a build record says, for
#: each, the value used and whether the job set it.
SETTINGS = ("carried", "margin", "feather", "edge", "window", "context", "context_noise", "motion", "motion_edge",
            "motion_edge_2", "hair_band", "render_region", "seed", "composite")
#: What a load's `composite` may say: the Masked Source's own two (`video_mask.COMPOSITE_REGION`, `COMPOSITE_CHANGED`;
#: `bench/check_masked_job.py` holds the two names together). Absent, the node's default.
COMPOSITES = ("whole region", "only what changed")
#: What a load's `render_region` may say, and the Masked Source's choice for each (`video_mask.RENDER_REGIONS`).
#: `frame` is the node's own default and sets nothing; `part` renders the load zoomed in on what is replaced.
#: A zoomed load must be one shot (the node refuses a cut inside the loaded frames, and the tool refuses it
#: first, from the capture's cuts). It takes a motion video only as a file the load names (`motion_file`): the
#: node hands a zoomed load its motion video as it is, not cut to the box (`video_mask.zoomed_source`), so the
#: preview's own body model, which is of the whole frame, is refused here and a video framed on the subject (the
#: body model cut to the head, `bench/head_reference.py`) is what a zoomed load names. Until 2026-10-13 a zoomed
#: load took no motion video at all.
RENDER_REGIONS = {"frame": None, "part": "zoomed in on what is replaced"}
#: What a load's `carried` may say (the capture's words): the whole tracked subject, or its part.
CARRIED = ("track", "held")
#: What a subject of a two-subject load may say is replaced, in the plan's words (`bench/shot_coverage.py::REGIONS`),
#: and what the load carries of that subject for each.
PAIR_REGIONS = {"face": "held", "face and hair": "held", "whole": "track"}
#: The keys one subject of a two-subject load may hold.
PAIR_KEYS = ("subject", "region", "parts_mask", "parts_first", "still", "motion_file", "motion_first", "motion_edge", "others",
             "hair_band", "hair_band_cuts")
#: The keys of a load that a two-subject load does not set itself, since each is read off its `subjects`.
PAIR_READS = ("subject", "others", "carried", "parts_mask", "parts_first", "still", "motion", "motion_file", "motion_first", "motion_edge",
              "motion_file_2", "motion_first_2", "motion_edge_2", "hair_band", "hair_band_cuts", "also")


def one_subject_view(one: dict) -> dict:
    """A load of two subjects as the one-subject load every reader of a job file takes, with the second beside it.

    **A two-subject load** replaces two people in one pass: `subjects` is a list of two, in the order the model
    is given them, each `{"subject", "region"}` with what else that subject brings: its part (`parts_mask`,
    `parts_first`; the capture's own part when left out), its `still`, its reference video (`motion_file`,
    `motion_first`, `motion_edge`), the keep-outs it needs (`others`), and for the first a `hair_band`. The first
    subject's still is `<Picture 1>` and its reference `<Video 1>`; the second's are `<Picture 2>` and
    `<Video 2>`: the order the song node emits them, stills in the chain's order and then the two wired videos
    (`reference_order.py`).

    What the load is, to everything that reads one subject a load (the capture, its gate, this builder's own
    rules): the FIRST subject's load, with the second carried in `also` and its reference in `motion_file_2`.
    So put first the subject whose region is larger. The region is the union of the two parts and the tracked
    mask the union of the two tracks (`load`); the keep-outs are both subjects' together, and neither subject is
    ever one of them. There is ONE margin, the load's `margin`: the node grows its region by one number.

    **What tells the model which still is whose: the text, and nothing else.** A reference still is tokens and
    rows the model attends to from everywhere; nothing ties one to a place in the picture. The text's two
    subject definitions, where each person is, and each reference video being that person's own head are all
    there is. A load without `subjects` is returned as it is."""
    if "subjects" not in one:
        return one
    name, pair = one.get("name", "?"), one["subjects"]
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(s, dict) for s in pair):
        raise ValueError(f"load {name!r}: `subjects` is a list of two, each a subject's own keys; a load of one subject names it with `subject`")
    set_twice = [k for k in PAIR_READS if k in one]
    if set_twice:
        raise ValueError(f"load {name!r} has `subjects` and also sets {', '.join(set_twice)}: in a two-subject load each of those is "
                         f"read off the list")
    for n, s in enumerate(pair, 1):
        unknown = [k for k in s if k not in PAIR_KEYS]
        if unknown or "subject" not in s or s.get("region") not in PAIR_REGIONS:
            raise ValueError(f"load {name!r}: subject {n} of `subjects` needs `subject` and a `region` of {tuple(PAIR_REGIONS)}"
                             + (f"; it also holds {', '.join(unknown)}, which a subject does not take ({', '.join(PAIR_KEYS)})" if unknown else ""))
    first, second = pair
    labels = [first["subject"], second["subject"]]
    if labels[0] == labels[1]:
        raise ValueError(f"load {name!r}: both of its `subjects` are {labels[0]!r}")
    kept = [x for s in pair for x in (s.get("others") or ())]
    if set(kept) & set(labels):
        raise ValueError(f"load {name!r}: {sorted(set(kept) & set(labels))} is one of its two subjects and is listed among `others`: "
                         f"neither subject of a two-subject load is kept out of it")
    if "motion_file" in second and "motion_file" not in first:
        raise ValueError(f"load {name!r}: its second subject names a `motion_file` and its first does not: <Video 1> is the first subject's")
    for key in ("hair_band", "hair_band_cuts"):
        if key in second:
            raise ValueError(f"load {name!r}: `{key}` is the first subject's: the band is cut under the first subject's part. Put the "
                             f"subject that needs it first")
    out = {k: v for k, v in one.items()}
    out.update(subject=labels[0], others=list(dict.fromkeys(kept)), carried=PAIR_REGIONS[first["region"]],
               motion="mesh" if "motion_file" in first else "none")
    for key in ("parts_mask", "parts_first", "still", "motion_file", "motion_first", "motion_edge", "hair_band", "hair_band_cuts"):
        if key in first:
            out[key] = first[key]
    for key in ("motion_file", "motion_first", "motion_edge"):
        if key in second:
            out[f"{key}_2"] = second[key]
    out["also"] = {"subject": labels[1], "region": second["region"], "carried": PAIR_REGIONS[second["region"]],
                   **{k: second[k] for k in ("parts_mask", "parts_first", "still") if k in second}}
    return out


def stills_of(g: dict) -> list[tuple[str, str]]:
    """The stills of a load's graph in the order the model is given them, as (the append node, its image loader):
    the chain the song node's `references` ends, read back to its start."""
    chain, at = [], g[_the(g, SONG)]["inputs"].get("references")
    while isinstance(at, list) and g.get(str(at[0]), {}).get("class_type") == STILL:
        node = g[str(at[0])]["inputs"]
        chain.append((str(at[0]), str(node["image"][0])))
        at = node.get("references")
    return chain[::-1]
SONG, SOURCE, STILL = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource", "MiniMaxH3AppendRefImage"
#: Ids a load adds to the shipped graph, where it uses none: the mask and motion loaders, the audio.
MASK_FILES, AUDIO_FILE, AUDIO_CUT = 150, "145", "146"
#: How much shorter than its load, in frames, a load's audio is cut. The song node writes as many frames as
#: its audio is long, rounded up (`loop_plan.frames_kept`), so audio a single sample past the load's last
#: frame writes one frame more: the first frame of the next shot, or a held one. **Measured** 2026-10-10: of
#: twelve planned loads given the loader's own audio for exactly their frames, one planned a frame past its
#: end. Half a frame short rounds up to the load's own length whatever the sample rate and however a cut
#: rounds; **reasoned**, and held to the planner's own arithmetic by `bench/check_masked_job.py` for every
#: length up to the longest window at two sample rates.
#: Dated note, 2026-10-10, later the same day: the sentence above about a single sample is true of the song
#: node as the server ran it when this was written. Since e79cc219 the planner itself takes a track within
#: `loop_plan.TRACK_SLACK_SECONDS` past a whole frame count as that count, so on a server that serves it one
#: sample long writes nothing more and this cut is a second guard: it still matters for audio that is longer
#: by more than that slack, which an audio file given whole would be. Once that commit is served everywhere a
#: job runs, the cut can be the load's exact length and the trim node after the loader can go.
AUDIO_SHORT_BY = 0.5


def _the(g: dict, class_type: str) -> str:
    """The id of the one node of a class; a shipped graph that has none or two is not the graph this was written for."""
    found = [nid for nid, n in g.items() if n["class_type"] == class_type]
    if len(found) != 1:
        raise ValueError(f"the base graph holds {len(found)} `{class_type}` nodes; a load is built on exactly one")
    return found[0]


def _reached(g: dict, roots: list[str]) -> dict:
    """The nodes a run of `roots` executes: the roots and everything they are linked to."""
    keep, todo = set(), list(roots)
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep.add(nid)
        todo += [v[0] for v in g[nid]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in g]
    return {nid: n for nid, n in g.items() if nid in keep}


def motion_of(job: dict, one: dict) -> str:
    """A load's motion video: what it says, else the mesh for a whole-subject pass of a subject the preview
    made a mesh of, else none. A part pass takes none unless it asks: a face pass with a body mesh beside it
    has not been rendered."""
    if "motion" in one:
        return one["motion"]
    subject = next((s for s in job["subjects"] if s["label"] == one.get("subject")), {})
    return "mesh" if one.get("carried", "track") == "track" and subject.get("pose", True) else "none"


#: The video grid a window's frame count is on: `17n + 5`. Inherited: `h3_rules.FRAME_FACTOR` and
#: `FRAME_REMAINDER`, core's `align_frame_count`; `bench/check_masked_job.py` holds these to `h3_rules.snap_length`.
FRAME_FACTOR, FRAME_REMAINDER = 17, 5


def snapped(frames: int) -> int:
    """`frames` rounded up to the next length the video grid has (`h3_rules.snap_length`, for a count over a frame)."""
    frames = max(int(frames), FRAME_REMAINDER)
    return frames + (FRAME_REMAINDER - frames) % FRAME_FACTOR


def both_clocks(windows: tuple[int, ...]) -> tuple[int, ...]:
    """Every window length on both clocks (17n + 5 and whole on the 25 Hz audio clock: `h3_rules.on_both_clocks`),
    worked out from `loop_plan.CHAIN_LENGTHS` (`windows`) alone since that module needs ComfyUI."""
    if len(windows) < 2:
        return tuple(windows)
    step = windows[1] - windows[0]
    return tuple(c for c in range(windows[0] % step, max(windows) + 1, step) if c > 0)


def one_window(frames: int, window: int, windows: tuple[int, ...]) -> bool:
    """Whether a load of `frames` in a window of `window` is ONE window that stands alone: the window holds it whole
    and is on both clocks (`both_clocks`, `h3_rules.on_both_clocks`: the owner, 2026-10-21)."""
    return 0 < frames <= window and window in both_clocks(windows)


def window_for(frames: int, windows: tuple[int, ...], continues: bool = False) -> int:
    """The window a load is rendered in: for a load one window can hold, the shortest length on both clocks
    (`both_clocks`, `h3_rules.on_both_clocks`) that holds it whole (the owner, 2026-10-21), padded with real source
    frames before and after the shot (`window_pad`) that are tossed after the decode; for a longer one, the planner's
    longest, and it runs in several.

    A load that `continues` another window (`continue_from`) meets it at a seam, so its window is a chained length:
    the shortest of `windows` that holds it, or the longest."""
    choices = sorted(windows) if continues else both_clocks(windows)
    return next((w for w in choices if w >= frames), max(windows))


def window_pad(first: int, frames: int, window: int, source_frames: int | None = None,
               continues: bool = False, floor: int = 0) -> tuple[int, int]:
    """(before, after) real source frames of untouched context around a one-window load shorter than its window.

    Split evenly before and after (`extra // 2` before, the rest after), shifted into `after` when `first` is near
    the start of the clip (`floor`, 0 or an external audio file's `audio_first`) and into `before` when `first + frames`
    is near `source_frames` (the clip's end)."""
    if continues or frames >= window:
        return 0, 0
    extra = int(window) - int(frames)
    avail_before = max(0, int(first) - int(floor))
    before = min(avail_before, extra // 2)
    after = extra - before
    if source_frames is not None and int(first) + int(frames) + after > int(source_frames):
        after = max(0, int(source_frames) - (int(first) + int(frames)))
        before = min(avail_before, extra - after)
    return before, after


def keeps_frames(one: dict, frames: int, window: int) -> bool:
    """Whether a load has a window that is shown frames kept from a window before it, and so may take `context_noise`:
    one that continues another run (`continue_from`), or one longer than its window, which the song node runs as
    chained windows, each later one keeping the last frames of the one before (2026-10-15: until then only a
    continuation could set it, though the node shows an inner window its kept frames the same way)."""
    return bool(one.get("continue_from")) or frames > window


def contexts_for(window: int, windows: tuple[int, ...]) -> tuple[int, ...]:
    """The contexts the planner takes in a window: the lengths on the windows' own grid that are shorter than it.

    Worked out from `loop_plan.CHAIN_LENGTHS` alone, since that module needs ComfyUI: a window and a context
    are on both clocks when they sit the same step apart as the windows do. `bench/check_masked_job.py` holds
    this to `loop_plan.check_window_settings` for every window."""
    step = windows[1] - windows[0]
    return tuple(c for c in range(windows[0] % step, window, step) if c > 0)


#: The shot headers a two-shot load's text carries, and the one it must not: the vendor's form for a text of
#: several shots (`docs/prompting.md`, the `[Shot N]` form). Held by presence, never by count: a text's other
#: sections name a shot as well (`(appears in [Shot 1])`).
TWO_SHOT_HEADERS, THIRD_SHOT_HEADER = ("[Shot 1]", "[Shot 2]"), "[Shot 3]"


def two_shots_of(job: dict) -> dict | None:
    """A job block's `two_shots`, held to its form; None for a job without the key.

    The key is the one way a load may hold a cut, so what it says is exact: `cut`, the source frame the second
    shot begins on; `shots`, the two shots as `[first, last]`, the first ending on the frame before the cut and
    the second beginning on it; `why` and `said_by`, a person's words and who they are. Two shots in one window
    are a choice somebody made for one load, never a default, and a choice with no reason or no name is refused."""
    two = job.get("two_shots")
    if two is None:
        return None
    form = ('`two_shots` is {"cut": the source frame the second shot begins on, "shots": [[first, last], [first, last]], '
            '"why": a person\'s words, "said_by": who}')
    if not isinstance(two, dict):
        raise ValueError(f"the job block's `two_shots` is {two!r}: {form}")
    cut, shots = two.get("cut"), two.get("shots")
    whole = lambda n: type(n) is int and n >= 0   # noqa: E731
    if not (whole(cut) and isinstance(shots, list) and len(shots) == 2
            and all(isinstance(x, list) and len(x) == 2 and all(whole(n) for n in x) and x[0] <= x[1] for x in shots)):
        raise ValueError(f"the job block's `two_shots` has cut {cut!r} and shots {shots!r}: {form}")
    if shots[0][1] + 1 != cut or shots[1][0] != cut:
        raise ValueError(f"the job block's `two_shots` names the cut on {cut} and the shots {shots[0][0]}-{shots[0][1]} and "
                         f"{shots[1][0]}-{shots[1][1]}: two NEIGHBOURING shots, the first ending on the frame before the cut "
                         f"and the second beginning on it")
    for key, what in (("why", "why these two shots share a window"), ("said_by", "who said so")):
        if not isinstance(two.get(key), str) or not two[key].strip():
            raise ValueError(f"the job block's `two_shots` does not say {what} (`{key}`): two shots in one window are a "
                             f"person's choice for one load, never a default")
    return two


def check_load(job: dict, one: dict, windows: tuple[int, ...]) -> None:
    """Refuse a load a graph cannot be built from, in words that name the key."""
    name = str(one.get("name", ""))
    if not NAME.match(name):
        raise ValueError(f"load name {name!r}: letters, digits and underscores only")
    labels = [s["label"] for s in job["subjects"]]
    if one.get("subject") not in labels:
        raise ValueError(f"load {name!r}: its subject {one.get('subject')!r} is not one of the job's, {labels}")
    if one["subject"] in (one.get("others") or ()):
        raise ValueError(f"load {name!r}: its subject is also among its `others`")
    first, frames = one["first"], one["frames"]
    if frames < 1 or first < job["first"] or first + frames > job["first"] + job["frames"]:
        raise ValueError(f"load {name!r}: frames {first}..{first + frames - 1} are not inside the job's span "
                         f"{job['first']}..{job['first'] + job['frames'] - 1}")
    if one.get("carried", "track") not in CARRIED:
        raise ValueError(f"load {name!r}: carried {one['carried']!r}; one of {CARRIED}")
    if motion_of(job, one) not in MOTIONS:
        raise ValueError(f"load {name!r}: motion {one['motion']!r}; one of {tuple(MOTIONS)}")
    if one.get("edge", "tokens") not in MASKED_JOB_EDGE:
        raise ValueError(f"load {name!r}: edge {one['edge']!r}; one of {tuple(MASKED_JOB_EDGE)}")
    also = one.get("also")
    if also is not None:
        if also.get("subject") not in labels or also["subject"] == one["subject"] or also["subject"] in (one.get("others") or ()):
            raise ValueError(f"load {name!r}: its second subject {also.get('subject')!r} must be one of the job's {labels}, not its "
                             f"first subject and not among its `others`")
        if also.get("carried") not in CARRIED:
            raise ValueError(f"load {name!r}: its second subject's carried {also.get('carried')!r}; one of {CARRIED}")
        if one.get("render_region", "frame") != "frame":
            raise ValueError(f"load {name!r} is of two subjects and is rendered zoomed in: one box round one part is a one-subject load")
    if "hair_band" in one:
        band, cuts = one["hair_band"], one.get("hair_band_cuts", MASKED_JOB_HAIR_BAND["cuts"])
        if isinstance(band, bool) or not isinstance(band, (int, float)) or not 0 < band <= MASKED_JOB_HAIR_BAND["most"]:
            raise ValueError(f"load {name!r}: hair_band {band!r}; a share of the part's height, over 0 and up to "
                             f"{MASKED_JOB_HAIR_BAND['most']:g}. A load with no band leaves the key out")
        if one.get("carried", "track") != "held":
            raise ValueError(f"load {name!r} asks for a hair band and takes the whole subject: the band is read off "
                             f"a part (`carried: held`)")
        if cuts not in (one.get("others") or ()):
            raise ValueError(f"load {name!r}: the hair band is cut out of the keep-out {cuts!r}, which is not among "
                             f"the load's `others` {list(one.get('others') or ())}; name one of them with `hair_band_cuts`")
    elif "hair_band_cuts" in one:
        raise ValueError(f"load {name!r} sets `hair_band_cuts` and asks for no `hair_band`")
    if one.get("motion_file_2") and motion_of(job, one) == "none":
        raise ValueError(f"load {name!r} names a second motion video (`motion_file_2`) and takes no motion video: a second "
                         f"comes after a first. Give the load `motion: mesh`, or take the key out")
    for key in ("motion_first_2", "motion_edge_2"):
        if key in one and not one.get("motion_file_2"):
            raise ValueError(f"load {name!r} sets `{key}` and names no `motion_file_2`")
    region = one.get("render_region", "frame")
    if region not in RENDER_REGIONS:
        raise ValueError(f"load {name!r}: render_region {region!r}; one of {tuple(RENDER_REGIONS)}")
    if region == "part" and one.get("carried", "track") != "held":
        raise ValueError(f"load {name!r}: render_region `part` on a load that takes the whole subject; a zoomed "
                         f"load takes a part (`carried: held`)")
    if "largest_zoom" in one:
        raise ValueError(f"load {name!r} sets largest_zoom: no longer a key; the zoom's limit is the node's own constant")
    if region != "frame" and motion_of(job, one) != "none" and not one.get("motion_file"):
        raise ValueError(f"load {name!r} is rendered zoomed in and takes the preview's own motion video "
                         f"(`{motion_of(job, one)}`), which is of the whole frame: the node shows a zoomed load its "
                         f"motion video as it is, not cut to the box. Name a video framed on the subject with "
                         f"`motion_file` (the body model cut to the head), or give the load `motion: none`.")
    if region != "frame" and one.get("continue_from"):
        raise ValueError(f"load {name!r} is rendered zoomed in and continues another run (`continue_from`): the node "
                         f"refuses a zoomed load that is not a run of its own")
    if "held_tail" in one:
        raise ValueError(f"load {name!r} sets held_tail: no longer a key. A load shorter than its window keeps its "
                         f"region open over the held frames, and that is the node's only behaviour since 2026-10-10")
    if "window" in one and one["window"] not in windows and not one_window(frames, one["window"], windows):
        raise ValueError(f"load {name!r}: window {one['window']} is not a length the planner has, {windows}, nor one on both clocks "
                         f"{both_clocks(windows)} that holds this load's {frames} frames as one window")
    window, context = one.get("window", window_for(frames, windows, bool(one.get("continue_from")))), one.get("context", MASKED_JOB_CONTEXT)
    if job.get("source_frames") is not None:
        total_src = int(job["source_frames"])
        if first + frames > total_src:
            raise ValueError(f"load {name!r}: frames {first}..{first + frames - 1} reach past the source's {total_src} frames")
        floor = one.get("audio_first", job["first"]) if one.get("audio") else 0
        before, after = window_pad(first, frames, window, total_src, bool(one.get("continue_from")), floor=floor)
        if not one.get("continue_from") and frames < window and frames + before + after < window:
            raise ValueError(f"load {name!r}: a {window}-frame window around frames {first}..{first + frames - 1} needs {window - frames} "
                             f"context frame(s) from the source, and only {before + after} exist within {floor}..{total_src - 1}")
    if one.get("continue_from") and window not in windows:
        raise ValueError(f"load {name!r} continues another run (`continue_from`) in a window of {window}: a window that meets "
                         f"another at a seam is one of the planner's chained lengths, {windows}")
    # a context is what a window keeps of the one before it: a load that is one window off the chained lengths has
    # none to keep and is held to none (the planner asks none of it either, `loop_plan.check_window_settings`)
    if window in windows and context not in contexts_for(window, windows):
        raise ValueError(f"load {name!r}: context {context} in a window of {window}; the planner takes one of "
                         f"{contexts_for(window, windows)}")
    margin, feather = one.get("margin", MASKED_SOURCE["grow_pixels"]), one.get("feather", MASKED_SOURCE["feather_pixels"])
    if feather > margin:
        raise ValueError(f"load {name!r}: a feather of {feather} px is wider than its margin of {margin} px, so "
                         f"the blend would reach past the region that was regenerated")
    if one.get("context_noise") is not None and not keeps_frames(one, frames, window):
        raise ValueError(f"load {name!r}: context_noise on a load of one window that continues nothing (no continue_from): "
                         f"no window of it keeps frames")
    two = two_shots_of(job)
    if two is not None:
        # two neighbouring shots in one window, on the job's word: the load holds the one named cut, and is the
        # plainest kind of load in everything else, so that the cut is the only new thing in it
        cut, (shot_1, shot_2), last = two["cut"], two["shots"], first + frames - 1
        if not first < cut <= last:
            raise ValueError(f"load {name!r} is source frames {first}-{last} and the job's `two_shots` names the cut on {cut}: "
                             f"a load of a two-shot job holds its cut. A load of one shot is a job without the key")
        if first < shot_1[0] or last > shot_2[1]:
            raise ValueError(f"load {name!r} is source frames {first}-{last} and the job's `two_shots` names the shots "
                             f"{shot_1[0]}-{shot_1[1]} and {shot_2[0]}-{shot_2[1]}: the load reaches outside them")
        if also is not None:
            raise ValueError(f"load {name!r} is of two subjects in a job of two shots (`two_shots`): not built. One subject")
        if region != "frame":
            raise ValueError(f"load {name!r} is rendered zoomed in in a job of two shots (`two_shots`): a zoomed load takes one "
                             f"box for the whole load and the node refuses a cut inside it")
        if one.get("continue_from"):
            raise ValueError(f"load {name!r} continues another run (`continue_from`) in a job of two shots (`two_shots`): a "
                             f"two-shot load is a run of its own")
        if not one_window(frames, window, windows):
            raise ValueError(f"load {name!r} is {frames} frames in a window of {window}: a two-shot load is ONE window on both clocks "
                             f"{both_clocks(windows)} that holds both shots whole. Two shots too long for one "
                             f"window are two loads")


def run_prefix(job: dict, one: dict, plan: bool) -> str:
    """Where a load's run saves, as the song node's prefix: a folder of its own, and another for its plan."""
    folder = one["name"] + ("_plan" if plan else "")
    return f"{job['output']}/{job['name']}/{folder}/{one['name']}"


def load(base: dict, job: dict, one: dict, files: dict, text: str, *, windows: tuple[int, ...],
         plan: bool = False) -> dict:
    """One load's graph, from the shipped masked graph: the render, or with `plan` what the node says it will do.

    `files` is what the tool found for this load, each a `(path, the source frame its frame 0 is on)`: `track`,
    the subject's mask from the capture; `parts`, the part to take, or None for the whole subject; `others`,
    the masks of the people kept out of the margin; `motion`, the saved motion video, or None; `motion_2`, a second
    one the load names, or absent. `text` is the
    load's own text: the builder writes none.

    What it changes in the base, and nothing else: the loader (the source by path, on the load's frames); the
    tracker and the part node, which go, their masks read from files; the Masked Source's region, margin,
    edge, others, motion video and held tail; the song node's text, canvas, window, context, seed, extent,
    prefix and audio; the still and the text encoder's view of it; Sol's sink. The model, the sampler, the
    steps and the shift stay the row's."""
    check_job(job)
    check_load(job, one, windows)
    canvas = tuple(job["canvas"])
    first, frames = one["first"], one["frames"]
    if one.get("audio") and first < one.get("audio_first", job["first"]):
        raise ValueError(f"load {one['name']!r} starts before its audio file does")
    window = one.get("window", window_for(frames, windows, bool(one.get("continue_from"))))
    floor = one.get("audio_first", job["first"]) if one.get("audio") else 0
    before, after = window_pad(first, frames, window, job.get("source_frames"), bool(one.get("continue_from")), floor=floor)
    win_first, win_frames = first - before, frames + before + after
    g = copy.deepcopy(base)
    song, source = _the(g, SONG), _the(g, SOURCE)
    frames_from = g[source]["inputs"]["frames"][0]
    g[frames_from] = loader(job["source"], win_first, win_frames, canvas)
    if before or after:
        g[source].setdefault("_meta", {}).update(pad_before=before, pad_after=after)
    counter = iter(range(MASK_FILES, MASK_FILES + 1000, 10))

    def from_file(path: str, at: int, as_mask: bool = True) -> list:
        nid = next(counter)
        g[str(nid)] = loader(path, first, frames, canvas, at=at)
        if not as_mask:
            return [str(nid), 0]
        g[str(nid + 1)] = {"class_type": "ImageToMask", "inputs": {"image": [str(nid), 0], "channel": "red"}}
        return [str(nid + 1), 0]

    src = g[source]["inputs"]
    for gone in ("segmenter", "segmenter_clip", "shot_table", "parts", "others", "keep", "motion_video", "motion_video_2",
                 "motion_short_edge_2"):
        src.pop(gone, None)
    part = one.get("carried", "track") == "held"
    if part and not files.get("parts"):
        raise ValueError(f"load {one['name']!r} takes a part (`carried` is `held`) and no part mask was found for it")
    src.update(frames=[frames_from, 0], mask=from_file(*files["track"]),
               replace=(MASKED_PARTS_SOURCE if part else MASKED_SOURCE)["replace"],
               grow_pixels=one.get("margin", MASKED_SOURCE["grow_pixels"]),
               feather_pixels=one.get("feather", MASKED_SOURCE["feather_pixels"]))
    # the composite (2026-10-16, the owner on the mom's holes: "only what changed" restores the source wherever the render
    # did not change the picture enough, inside the region too): h3_config.MASKED_LOAD_COMPOSITE unless the load names one
    composite = one.get("composite", MASKED_LOAD_COMPOSITE)       # every load: the owner's decision of 2026-10-16
    if composite not in COMPOSITES:
        raise ValueError(f"load {one['name']!r}: composite {composite!r} is not one of 'whole region', 'only what changed'")
    src["composite"] = composite
    if part:
        src["parts"] = from_file(*files["parts"])
    also, theirs = one.get("also"), files.get("also")
    if also is not None:
        # a second subject in the same pass: the tracked mask is the union of the two tracks and the region the
        # union of the two parts, a whole subject's track standing for its part; one region, one margin
        if not theirs or not theirs.get("track"):
            raise ValueError(f"load {one['name']!r}: no track was found for its second subject {also['subject']!r}")
        second_part = also["carried"] == "held"
        if second_part and not theirs.get("parts"):
            raise ValueError(f"load {one['name']!r}: its second subject {also['subject']!r} takes a part and no part mask was found for it")

        def union(a: list, b: list) -> list:
            nid = str(next(counter))
            g[nid] = {"class_type": "MaskComposite", "inputs": {"destination": a, "source": b, "x": 0, "y": 0, "operation": "add"}}
            return [nid, 0]

        track_1, track_2 = src["mask"], from_file(*theirs["track"])
        src["mask"] = union(track_1, track_2)
        if part or second_part:
            src["parts"] = union(src["parts"] if part else track_1, from_file(*theirs["parts"]) if second_part else track_2)
            src["replace"] = MASKED_PARTS_SOURCE["replace"]
    if "edge" in one:
        src["edge"] = MASKED_JOB_EDGE[one["edge"]]
    joined = None
    for path, at in files.get("others") or ():
        mask = from_file(path, at)
        if joined is None:
            joined = mask
        else:
            nid = str(next(counter))
            g[nid] = {"class_type": "MaskComposite",
                      "inputs": {"destination": joined, "source": mask, "x": 0, "y": 0, "operation": "add"}}
            joined = [nid, 0]
    if joined is not None:
        src["others"] = joined
    motion = motion_of(job, one)
    src.update(motion_reference=MOTIONS[motion], motion_vae=motion != "none")
    if motion != "none":
        if not files.get("motion"):
            raise ValueError(f"load {one['name']!r} asks for motion `{motion}` and no motion video was found for it")
        # at the canvas's short edge (`h3_config.MASKED_JOB_MOTION` has why), and to the video model as well
        # as the text encoder: the encoder reads one picture of it a second
        src.update(motion_video=from_file(*files["motion"], as_mask=False),
                   motion_short_edge=one.get("motion_edge", min(canvas)))
        if files.get("motion_2"):
            # a second video beside the first, the next video reference; at the canvas's short edge unless the load says
            src.update(motion_video_2=from_file(*files["motion_2"], as_mask=False),
                       motion_short_edge_2=one.get("motion_edge_2", min(canvas)))
    for theirs in ("render_region", "largest_zoom"):
        src.pop(theirs, None)
    if RENDER_REGIONS[one.get("render_region", "frame")]:
        src["render_region"] = RENDER_REGIONS[one["render_region"]]
    src.pop("held_tail", None)

    still = _the(g, "LoadImage")
    if not plan and not one.get("still"):
        raise ValueError(f"load {one['name']!r} names no `still`: a render needs its reference picture")
    if one.get("still"):
        g[still]["inputs"]["image"] = one["still"]
    view = g[_the(g, STILL)]["inputs"]
    for key in [k for k in view if k.startswith("qwen_view")]:
        view.pop(key)
    view["qwen_view"] = MASKED_JOB_STILL_VIEW
    if also is not None:
        # the second subject's still, after the first in the chain: <Picture 2>, shown as the first is shown
        if not plan and not also.get("still"):
            raise ValueError(f"load {one['name']!r}: its second subject {also['subject']!r} names no `still`: a render needs both pictures")
        first_still = _the(g, STILL)
        load_2, append_2 = str(next(counter)), str(next(counter))
        g[load_2] = {"class_type": "LoadImage", "inputs": {**g[still]["inputs"], "image": also.get("still") or g[still]["inputs"]["image"]}}
        g[append_2] = {"class_type": STILL, "inputs": {**view, "image": [load_2, 0], "references": [first_still, 0]}}
        g[song]["inputs"]["references"] = [append_2, 0]
    sink = [nid for nid, n in g.items() if "sink_conditioning" in n["inputs"]]
    if len(sink) != 1:
        raise ValueError(f"the base graph holds {len(sink)} nodes with a `sink_conditioning`; a load sets exactly one")
    g[sink[0]]["inputs"]["sink_conditioning"] = MASKED_JOB_SOL_SINK

    run = g[song]["inputs"]
    for key in [k for k in run if k.startswith("extent")] + ["continue_from", "context_noise"]:
        run.pop(key, None)
    run.update(prompt=text, width=canvas[0], height=canvas[1], window_frames=window,
               context_frames=one.get("context", MASKED_JOB_CONTEXT), filename_prefix=run_prefix(job, one, plan),
               crf=MASKED_JOB_CRF, timeline="", preview=bool(plan), keep_windows=True)
    # the kept-mask store and stored-window reuse are gone from the nodes (2026-10-10); a shipped graph built
    # before that still names their inputs, and a load must not carry them over
    run.pop("reuse_windows", None)
    src.pop("reuse_mask", None)
    # the model hears the source's own track, cut to the window (`AUDIO_SHORT_BY` has why it is cut again
    # after the loader has cut it)
    heard = round((win_frames - AUDIO_SHORT_BY) / FPS, 6)
    g[AUDIO_CUT] = {"class_type": "TrimAudioDuration",
                    "inputs": {"audio": [frames_from, 2], "start_index": 0.0, "duration": heard}}
    run["audio"] = [AUDIO_CUT, 0]
    if "seed" in one:
        run["seed"] = one["seed"]
    # a DynamicCombo: the member after its parent
    run["extent"] = "first_seconds"
    run["extent.seconds"] = win_frames / FPS
    if one.get("continue_from"):
        run["continue_from"] = one["continue_from"]
        run["context_noise"] = one.get("context_noise", MASKED_JOB_CONTEXT_NOISE)
    elif one.get("context_noise") is not None:
        # a load of chained windows that names it: its inner windows are shown their kept frames at that noise. By
        # default they are shown them clean, the node's own `CONTEXT_NOISE`: the lead's ruling of 2026-10-15, per
        # load until a render of one says more (`h3_config.MASKED_JOB_CONTEXT_NOISE`)
        run["context_noise"] = one["context_noise"]
    # what the composite kept, saved into each window's region file for the review clips (2026-10-15): only when the load asks
    if one.get("save_kept"):
        run["save_kept"] = True
    if one.get("audio"):
        # a file in place of the source's track, cut the same way: left to run past the window's last frame it
        # makes the song node write frames to fill it
        g.pop(AUDIO_CUT)
        g[AUDIO_FILE] = {"class_type": "VHS_LoadAudio",
                         "inputs": {"audio_file": one["audio"],
                                    "seek_seconds": round((win_first - one.get("audio_first", job["first"])) / FPS, 6),
                                    "duration": heard}}
        run["audio"] = [AUDIO_FILE, 0]
    roots = [nid for nid, n in g.items() if n["class_type"] == "PreviewAny" and n["inputs"]["source"][0] == song]
    return _reached(g, [song] + roots)
