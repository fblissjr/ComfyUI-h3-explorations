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
are the tracker's inputs of those names; `match` is a number, the likeness
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

**A load** is one entry of the file's `loads`, with the file's `defaults` under it: one subject on one
stretch of frames, as a rule one shot. It is written in the capture's words, since the capture reads the same
entry as a plan (`first`, `frames`, `subject`, `others`, `margin`, `edge`, `carried`, `text`, `audio`,
`still`), and the builder reads a few more that the capture carries without using: `seed`, `window`,
`context`, `feather`, `motion` (`mesh`, `mesh zoomed` or `none`; without it a whole-subject pass of a subject
with a pose pass takes the mesh and any other load takes none), `continue_from`, `context_noise`,
`audio_first` (the source frame an audio file starts on), `parts_mask` (a mask video to take as the part, in
place of the capture's own), `motion_edge` (the motion video's short edge), `held_tail` (`open` or `plate`,
for a load shorter than its window), and `render_region` with `largest_zoom`: `subject` or `part` renders
the load zoomed in on the tracked subject or on what is replaced and lays it back (`RENDER_REGIONS`; for a
subject or a part that is small in the frame), where `frame`, the default, renders the whole frame. A default that was measured on one clip is a starting value for
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
    FPS, SEGMENTER, SUBJECT_TRACK, SAPIENS2, MASKED_SOURCE, MASKED_PARTS_SOURCE,
    MASKED_JOB_CONTEXT, MASKED_JOB_CONTEXT_NOISE, MASKED_JOB_SOL_SINK, MASKED_JOB_MOTION, MASKED_JOB_PART_BOX,
    MASKED_JOB_HELD_TAIL, MASKED_JOB_CRF, MASKED_JOB_EDGE, MASKED_JOB_STILL_VIEW,
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


# ---- a load: one subject on one stretch, from a checked capture ----------------

#: What a load's `motion` may say, and the Masked Source's choice for each.
MOTIONS = {"none": MASKED_JOB_MOTION["none"], "mesh": MASKED_JOB_MOTION["wired"],
           "mesh zoomed": MASKED_JOB_MOTION["zoomed"]}
#: The settings of a load a job file may set in place of the builder's default; a build record says, for
#: each, the value used and whether the job set it.
SETTINGS = ("carried", "margin", "feather", "edge", "window", "context", "context_noise", "motion", "motion_edge",
            "held_tail", "render_region", "largest_zoom", "seed")
#: What a load's `render_region` may say, and the Masked Source's choice for each (`video_mask.RENDER_REGIONS`).
#: `frame` is the node's own default and sets nothing. A zoomed load must be one shot: the node refuses a
#: cut inside the loaded frames, and the tool refuses it first, from the capture's cuts.
RENDER_REGIONS = {"frame": None, "subject": "zoomed in on the tracked subject", "part": "zoomed in on what is replaced"}
#: What a load's `held_tail` may say.
TAILS = ("open", "plate")
#: What a load's `carried` may say (the capture's words): the whole tracked subject, or its part.
CARRIED = ("track", "held")
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


def window_for(frames: int, windows: tuple[int, ...]) -> int:
    """The window a load is rendered in: the shortest the planner has that holds it whole, else the longest.

    A load capped at its shot is one window whenever a window can hold it; a shot shorter than the shortest
    window is still one window, its last frame held (`h3_config.MASKED_JOB_HELD_TAIL`). `windows` is
    `loop_plan.CHAIN_LENGTHS`, handed in because that module needs ComfyUI."""
    return next((w for w in sorted(windows) if w >= frames), max(windows))


def contexts_for(window: int, windows: tuple[int, ...]) -> tuple[int, ...]:
    """The contexts the planner takes in a window: the lengths on the windows' own grid that are shorter than it.

    Worked out from `loop_plan.CHAIN_LENGTHS` alone, since that module needs ComfyUI: a window and a context
    are on both clocks when they sit the same step apart as the windows do. `bench/check_masked_job.py` holds
    this to `loop_plan.check_window_settings` for every window."""
    step = windows[1] - windows[0]
    return tuple(c for c in range(windows[0] % step, window, step) if c > 0)


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
    region = one.get("render_region", "frame")
    if region not in RENDER_REGIONS:
        raise ValueError(f"load {name!r}: render_region {region!r}; one of {tuple(RENDER_REGIONS)}")
    if region == "part" and one.get("carried", "track") != "held":
        raise ValueError(f"load {name!r}: render_region `part` on a load that takes the whole subject; zoom in on "
                         f"the `subject`, or take a part (`carried: held`)")
    if "largest_zoom" in one and region == "frame":
        raise ValueError(f"load {name!r}: largest_zoom with no zoomed render_region")
    if one.get("held_tail", "open") not in TAILS:
        raise ValueError(f"load {name!r}: held_tail {one['held_tail']!r}; one of {TAILS}")
    if "window" in one and one["window"] not in windows:
        raise ValueError(f"load {name!r}: window {one['window']} is not a length the planner has, {windows}")
    window, context = one.get("window", window_for(frames, windows)), one.get("context", MASKED_JOB_CONTEXT)
    if context not in contexts_for(window, windows):
        raise ValueError(f"load {name!r}: context {context} in a window of {window}; the planner takes one of "
                         f"{contexts_for(window, windows)}")
    margin, feather = one.get("margin", MASKED_SOURCE["grow_pixels"]), one.get("feather", MASKED_SOURCE["feather_pixels"])
    if feather > margin:
        raise ValueError(f"load {name!r}: a feather of {feather} px is wider than its margin of {margin} px, so "
                         f"the blend would reach past the region that was regenerated")
    if one.get("context_noise") is not None and not one.get("continue_from"):
        raise ValueError(f"load {name!r}: context_noise with no continue_from; a load of its own keeps no frames")


def run_prefix(job: dict, one: dict, plan: bool) -> str:
    """Where a load's run saves, as the song node's prefix: a folder of its own, and another for its plan."""
    folder = one["name"] + ("_plan" if plan else "")
    return f"{job['output']}/{job['name']}/{folder}/{one['name']}"


def load(base: dict, job: dict, one: dict, files: dict, text: str, *, windows: tuple[int, ...],
         plan: bool = False) -> dict:
    """One load's graph, from the shipped masked graph: the render, or with `plan` what the node says it will do.

    `files` is what the tool found for this load, each a `(path, the source frame its frame 0 is on)`: `track`,
    the subject's mask from the capture; `parts`, the part to take, or None for the whole subject; `others`,
    the masks of the people kept out of the margin; `motion`, the saved motion video, or None. `text` is the
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
    g = copy.deepcopy(base)
    song, source = _the(g, SONG), _the(g, SOURCE)
    frames_from = g[source]["inputs"]["frames"][0]
    g[frames_from] = loader(job["source"], first, frames, canvas)
    counter = iter(range(MASK_FILES, MASK_FILES + 1000, 10))

    def from_file(path: str, at: int, as_mask: bool = True) -> list:
        nid = next(counter)
        g[str(nid)] = loader(path, first, frames, canvas, at=at)
        if not as_mask:
            return [str(nid), 0]
        g[str(nid + 1)] = {"class_type": "ImageToMask", "inputs": {"image": [str(nid), 0], "channel": "red"}}
        return [str(nid + 1), 0]

    src = g[source]["inputs"]
    for gone in ("segmenter", "segmenter_clip", "shot_table", "parts", "others", "keep", "motion_video"):
        src.pop(gone, None)
    part = one.get("carried", "track") == "held"
    if part and not files.get("parts"):
        raise ValueError(f"load {one['name']!r} takes a part (`carried` is `held`) and no part mask was found for it")
    src.update(frames=[frames_from, 0], mask=from_file(*files["track"]),
               replace=(MASKED_PARTS_SOURCE if part else MASKED_SOURCE)["replace"],
               grow_pixels=one.get("margin", MASKED_SOURCE["grow_pixels"]),
               feather_pixels=one.get("feather", MASKED_SOURCE["feather_pixels"]))
    if part:
        src["parts"] = from_file(*files["parts"])
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
    for theirs in ("motion_box", "motion_blur_share", "motion_blur_pixels", "motion_grey"):
        src.pop(theirs, None)
    src.update(motion_reference=MOTIONS[motion], motion_vae=motion != "none")
    if motion != "none":
        if not files.get("motion"):
            raise ValueError(f"load {one['name']!r} asks for motion `{motion}` and no motion video was found for it")
        # at the canvas's short edge (`h3_config.MASKED_JOB_MOTION` has why), and to the video model as well
        # as the text encoder: the encoder reads one picture of it a second
        src.update(motion_video=from_file(*files["motion"], as_mask=False),
                   motion_short_edge=one.get("motion_edge", min(canvas)))
        if part and motion == "mesh zoomed":
            src["motion_box"] = MASKED_JOB_PART_BOX
    window = one.get("window", window_for(frames, windows))
    for theirs in ("render_region", "largest_zoom"):
        src.pop(theirs, None)
    if RENDER_REGIONS[one.get("render_region", "frame")]:
        src["render_region"] = RENDER_REGIONS[one["render_region"]]
        if "largest_zoom" in one:
            src["largest_zoom"] = float(one["largest_zoom"])
    src.pop("held_tail", None)
    if frames < window and one.get("held_tail", "open") == "open":
        src["held_tail"] = MASKED_JOB_HELD_TAIL

    still = _the(g, "LoadImage")
    if not plan and not one.get("still"):
        raise ValueError(f"load {one['name']!r} names no `still`: a render needs its reference picture")
    if one.get("still"):
        g[still]["inputs"]["image"] = one["still"]
    view = g[_the(g, STILL)]["inputs"]
    for key in [k for k in view if k.startswith("qwen_view")]:
        view.pop(key)
    view["qwen_view"] = MASKED_JOB_STILL_VIEW
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
    # the model hears the source's own track, cut to the load (`AUDIO_SHORT_BY` has why it is cut again
    # after the loader has cut it)
    heard = round((frames - AUDIO_SHORT_BY) / FPS, 6)
    g[AUDIO_CUT] = {"class_type": "TrimAudioDuration",
                    "inputs": {"audio": [frames_from, 2], "start_index": 0.0, "duration": heard}}
    run["audio"] = [AUDIO_CUT, 0]
    if "seed" in one:
        run["seed"] = one["seed"]
    # a DynamicCombo: the member after its parent
    run["extent"] = "first_seconds"
    run["extent.seconds"] = frames / FPS
    if one.get("continue_from"):
        run["continue_from"] = one["continue_from"]
        run["context_noise"] = one.get("context_noise", MASKED_JOB_CONTEXT_NOISE)
    if one.get("audio"):
        # a file in place of the source's track, cut the same way: left to run past the load's last frame it
        # makes the song node write frames to fill it
        g.pop(AUDIO_CUT)
        g[AUDIO_FILE] = {"class_type": "VHS_LoadAudio",
                         "inputs": {"audio_file": one["audio"],
                                    "seek_seconds": round((first - one.get("audio_first", job["first"])) / FPS, 6),
                                    "duration": heard}}
        if first < one.get("audio_first", job["first"]):
            raise ValueError(f"load {one['name']!r} starts before its audio file does")
        run["audio"] = [AUDIO_FILE, 0]
    roots = [nid for nid, n in g.items() if n["class_type"] == "PreviewAny" and n["inputs"]["source"][0] == song]
    return _reached(g, [song] + roots)
