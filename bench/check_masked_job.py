#!/usr/bin/env python3
"""Check the masked job builder: the graphs it writes and what it refuses.

`bench/masked_job.py` turns a job file into the graphs of a masked job, and
`workflows/masked_job_graph.py` is where those graphs are made. Every masked
session before 2026-10-10 did this with a script of its own that nothing
checked: an option that landed on the wrong input, or a second tracker not
told who the first holds, was found by a render.

Claims, i.e. what breaks if a case is deleted:

  slots are the outputs they name    a link is written by an output's name
                        (`masked_job_graph.SLOT`); the table is held to
                        `bench/node_id_manifest.json`, the committed baseline a
                        schema change cannot rewrite. Without it a node that
                        gains an output mid-list sends a class map where a
                        mask was meant
  the nodes' schemas accept it    every node of this pack in the graph, graded
                        by the generator's own validator against the nodes'
                        `define_schema`: names, required inputs, types, combo
                        options, ranges, links between them. Nodes of other
                        packs are graded by the server at `queue`
  the job's defaults are the nodes'    a `MASKED_JOB_*` dict that says it
                        equals a node's defaults does
  every job value lands on the input it names    the source, the span, the
                        canvas, each subject's pick, phrase, corrections and
                        parts, the pose table's labels
  a later tracker is told who the earlier ones hold    the masks reaching a
                        tracker's `others` are exactly the earlier trackers'
  a preview samples nothing    no sampler, scheduler, song node or model of
                        the render in a preview graph
  no two saves share a name    a second subject's table cannot overwrite the
                        first's
  the capture command reads what the preview saves    the command `build`
                        prints names, for each subject, the files the graph
                        saves and the frame they start on, and never the mask
                        of what a subject holds as a part (`held=`)
  the seek lands on the frame    `masked_job_graph.seek`, read back from a
                        made clip whose every frame says its own number, with
                        the loader's own ffmpeg arguments
  the refusals         a job folder or an output folder that exists; a
                        correction with no tracker version or another one; a
                        job with no pick; a stage queued twice; node code not
                        committed, with and without an override; an input the
                        server does not serve
  a load lands on the inputs it names    a whole-subject load and a part load,
                        each traced from the Masked Source and the song node
                        back to the file and the frame it should read: the
                        masks from the capture's first frame, the mesh from the
                        job's, the audio cut to the load, the region open over
                        a held tail and not set when the load fills its window,
                        the kept frames noised only on a continuation
  a plan is the render with nothing sampled    the two graphs of a load differ
                        in the song node's `preview` and its folder, no more
  a load leaves the row's model alone    every node of the shipped graph a
                        load keeps and does not name is as the shipped file
                        has it, and Sol differs in its sink only
  a load the graph cannot be built from    each named refusal of `check_load`
                        and `load`
  loads from a capture    `build plan` and `build render` on a made capture
                        folder: the files found are the capture's, the part is
                        the capture's filled one, the mesh is the preview's;
                        and what is refused there
                        the build record says which settings the job file set
                        and which were the builder's defaults, and what the
                        capture measured on this clip for the load, with a
                        figure the capture lacks said to be absent
  the gate before a render    (and, in the same case, what `collect` reads out
                        of a finished run's history: its files, its words,
                        its time, its error)
                        `verify` again, the capture's verdict, and the
                        node's own plan read for that load, each refused apart

**What it does not show**, each found by breaking the thing on purpose and
seeing the case stay green:

- that the seek's step back is needed. A time on the frame itself also lands
  on the asked frame of the made clip, so the seek case proves the answer is
  right and says nothing for `MASKED_JOB_SEEK_BACK`, which is inherited;
- a DynamicCombo member written while its parent is on another option
  (`pick_on.pick_frame` with `pick_on` automatic). The generator's validator
  leaves a member of an option that is not selected ungraded, as its own
  comment says. The builder never writes one, and the values case holds that.

And by construction: nodes of other packs (the video loader and saver, core's
mask nodes) are graded by the running server at `queue`, never here; and
nothing here runs a node, so what a tracker does with `others` is
`bench/check_subject_track.py`'s to show.

No card, no server. The schema cases import this pack on core's CPU path; the
seek case runs ffmpeg on a clip of a few hundred small frames.

    <python> bench/check_masked_job.py
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import REPO, case, finish, skip  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config as cfg  # noqa: E402
import masked_job_graph as graphs  # noqa: E402
import masked_job as tool  # noqa: E402

MANIFEST = HERE / "node_id_manifest.json"
#: A job of three subjects, so the join of earlier trackers is exercised: one named frame and parts, one
#: correction, one with its pick and no pose pass.
JOB = {"name": "check", "source": "/clips/check_24p.mov", "first": 604, "frames": 447, "canvas": [1024, 768],
       "output": "Video/check",
       "subjects": [{"label": "a", "pick": "largest", "pick_at": 664, "parts": ["hair", "hands"], "phrase": "singer"},
                    {"label": "b", "pick": "most central", "corrections": "shot 3: person 2", "max_people": 4},
                    {"label": "c", "pick": "largest", "pose": False}]}
#: What stands where a node of another pack is linked, when only this pack's nodes are graded.
SERVER_GRADES = "<a link to a node the server grades>"
POSED = [s for s in JOB["subjects"] if s.get("pose", True)]
_declared: dict | None = None


def _nodes_of(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return sorted(((nid, n["inputs"]) for nid, n in graph.items() if n["class_type"] == class_type),
                  key=lambda item: int(item[0]))


def _pack():
    """This pack's node classes, by node id, through the schema check's own import."""
    import check_schema_defaults as S
    return {node.define_schema().node_id: node for node in S._nodes()}


def declared() -> dict:
    """What `/object_info` would serve for this pack's nodes, from their `define_schema`, with no server."""
    global _declared
    if _declared is None:
        _declared = {}
        for node_id, node in _pack().items():
            info = node.define_schema().get_v1_info(node)
            _declared[node_id] = {"input": info.input, "output": info.output}
    return _declared


def ours_only(graph: dict, oi: dict) -> dict:
    """The graph's nodes that are this pack's, a link to anybody else's node replaced by a placeholder."""
    return {nid: {"class_type": n["class_type"],
                  "inputs": {k: SERVER_GRADES if isinstance(v, list) and graph[v[0]]["class_type"] not in oi else v
                             for k, v in n["inputs"].items()}}
            for nid, n in graph.items() if n["class_type"] in oi}


def slots_are_the_outputs_they_name():
    manifest = {v["node_id"]: v["outputs"] for v in json.loads(MANIFEST.read_text()).values()}
    used = {n["class_type"] for n in graphs.preview(JOB).values()}
    wrong = []
    for class_type, slots in graphs.SLOT.items():
        assert class_type in manifest, f"{class_type} is in SLOT and not in the manifest"
        assert class_type in used, f"{class_type} is in SLOT and in no graph this check builds"
        wrong += [f"{class_type}.{name} is slot {manifest[class_type].index(name) if name in manifest[class_type] else None}, "
                  f"the table says {slot}" for name, slot in slots.items()
                  if name not in manifest[class_type] or manifest[class_type].index(name) != slot]
    assert not wrong, "; ".join(wrong)
    return f"{sum(len(s) for s in graphs.SLOT.values())} outputs of {len(graphs.SLOT)} nodes"


def the_nodes_schemas_accept_it():
    import build_workflows
    oi = declared()
    graph = ours_only(graphs.preview(JOB), oi)
    assert len(graph) >= 3 + 3 * len(JOB["subjects"]) + 3 * len(POSED), f"only {len(graph)} of this pack's nodes were found to grade"
    errors = build_workflows.validate_api(graph, oi, "preview")
    assert not errors, "\n      ".join(errors)
    return f"{len(graph)} nodes of {len({n['class_type'] for n in graph.values()})} classes"


def the_jobs_defaults_are_the_nodes():
    import check_schema_defaults as S
    pack = _pack()
    wrong = []
    for node_id, ours in (("MiniMaxH3SAM31Corrections", cfg.MASKED_JOB_SAM_CORRECTIONS),
                          ("MiniMaxH3BodyPose", cfg.MASKED_JOB_BODY_POSE),
                          ("MiniMaxH3SubjectBoxes", {"smallest_mask": cfg.MASKED_JOB_BOXES["smallest_mask"]})):
        theirs = S._schema_defaults(pack[node_id])
        wrong += [f"{node_id}.{k}: the job has {v!r}, the node {theirs.get(k)!r}" for k, v in ours.items()
                  if k not in theirs or theirs[k] != v]
    assert not wrong, "; ".join(wrong)


def every_job_value_lands_on_the_input_it_names():
    g = graphs.preview(JOB)
    loaders = _nodes_of(g, cfg.MASKED_JOB_LOADER)
    assert len(loaders) == 1, f"{len(loaders)} loaders"
    load = loaders[0][1]
    assert load["video"] == JOB["source"] and load["frame_load_cap"] == JOB["frames"], load
    assert [load["custom_width"], load["custom_height"]] == JOB["canvas"], load
    assert load["start_time"] == graphs.seek(JOB["first"]) and load["force_rate"] == cfg.MASKED_JOB_FORCE_RATE, load
    trackers, parts = _nodes_of(g, "MiniMaxH3SubjectTrack"), _nodes_of(g, "MiniMaxH3SubjectParts")
    poses, meshes = _nodes_of(g, "MiniMaxH3BodyPose"), _nodes_of(g, "MiniMaxH3BodyMeshVideo")
    assert len(trackers) == len(parts) == len(JOB["subjects"]), "a tracker or a part node a subject is missing"
    assert len(poses) == len(meshes) == len(POSED) and 0 < len(POSED) < len(JOB["subjects"]), \
        f"{len(poses)} pose passes and {len(meshes)} meshes for {len(POSED)} subjects that ask for one"
    for (_, pose), (_, mesh), subject in zip(poses, meshes, POSED):
        who = subject["label"]
        assert pose["subject"] == who and pose["first_source_frame"] == JOB["first"], f"{who}: the pose table's labels"
        assert pose["table_prefix"] == graphs.prefixes(JOB, who)["pose"], f"{who}: table_prefix"
        assert [mesh["width"], mesh["height"]] == JOB["canvas"], f"{who}: the mesh is not drawn at the canvas"
        boxes = g[pose["boxes"][0]]["inputs"]["mask"]
        assert g[boxes[0]]["inputs"]["pick"] == subject["pick"] and boxes[0] in dict(trackers), f"{who}: the pose is read from another subject's mask"
    for subject, (_, track), (_, part) in zip(JOB["subjects"], trackers, parts):
        who = subject["label"]
        assert track["pick"] == subject["pick"], f"{who}: pick {track['pick']!r}"
        assert track["subject_phrase"] == subject.get("phrase", cfg.SUBJECT_TRACK["subject_phrase"]), f"{who}: phrase"
        assert track["corrections"] == subject.get("corrections", ""), f"{who}: corrections {track['corrections']!r}"
        assert track["max_people"] == subject.get("max_people", cfg.SUBJECT_TRACK["max_people"]), f"{who}: max_people"
        if "pick_at" in subject:
            # the tracker counts from the first frame it is given, the job from the source's first
            assert track["pick_on.pick_frame"] + JOB["first"] == subject["pick_at"], \
                f"{who}: pick_frame {track['pick_on.pick_frame']}"
            assert track["pick_on"] != cfg.SUBJECT_TRACK["pick_on"], f"{who}: a frame is named and pick_on is automatic"
        else:
            assert track["pick_on"] == cfg.SUBJECT_TRACK["pick_on"] and "pick_on.pick_frame" not in track, f"{who}: pick_on"
        ticked = {k for k in graphs.PART_TICKS if part[k]}
        want = set(subject["parts"]) if subject.get("parts") else {k for k in graphs.PART_TICKS if cfg.SUBJECT_PARTS[k]}
        assert ticked == want, f"{who}: parts {sorted(ticked)}"
        assert part["hold_missing"] is False, f"{who}: the part node fills doubted frames before the capture grades them"
    return f"{len(JOB['subjects'])} subjects, {len(POSED)} with a pose pass"


def _masks_reaching(g: dict, link: list) -> set[str]:
    """The trackers whose masks reach a link, through any joins."""
    node = g[link[0]]
    if node["class_type"] == "MiniMaxH3SubjectTrack":
        return {link[0]}
    assert node["class_type"] == "MaskComposite" and node["inputs"]["operation"] == "add", \
        f"`others` is reached through a {node['class_type']}"
    return _masks_reaching(g, node["inputs"]["destination"]) | _masks_reaching(g, node["inputs"]["source"])


def a_later_tracker_is_told_who_the_earlier_ones_hold():
    g = graphs.preview(JOB)
    trackers = _nodes_of(g, "MiniMaxH3SubjectTrack")
    for n, (nid, inputs) in enumerate(trackers):
        earlier = {t for t, _ in trackers[:n]}
        told = _masks_reaching(g, inputs["others"]) if "others" in inputs else set()
        assert told == earlier, f"tracker {n} ({nid}) is told of {sorted(told)}, the earlier ones are {sorted(earlier)}"
    unused = [nid for nid, n in g.items() if n["class_type"] == "MaskComposite"
              and not any([nid, 0] in list(m["inputs"].values()) for m in g.values())]
    assert not unused, f"a join nothing reads: {unused}"
    return f"{len(trackers)} trackers"


def a_preview_samples_nothing():
    words = ("sampler", "scheduler", "guider", "song", "unet", "vae", "sigma", "noise", "encoder")
    found = sorted({n["class_type"] for n in graphs.preview(JOB).values()
                    if any(w in n["class_type"].lower() for w in words)})
    assert not found, f"a preview holds {found}"
    # the words are worth something only if the render's own graph trips them
    shipped = json.loads((REPO / "workflows" / "h3_video_to_video_masked_upper_song_ref2va_motion_api.json").read_text())
    tripped = {w for w in words if any(w in n["class_type"].lower() for n in shipped.values())}
    assert len(tripped) >= 5, f"the shipped masked graph trips only {sorted(tripped)}: the word list no longer describes a render"


def no_two_saves_share_a_name():
    g = graphs.preview(JOB)
    names = [v for n in g.values() for k, v in n["inputs"].items() if k in ("filename_prefix", "table_prefix")]
    assert len(names) == 5 * len(JOB["subjects"]) + 2 * len(POSED), f"{len(names)} saves for {len(JOB['subjects'])} subjects"
    twice = sorted({n for n in names if names.count(n) > 1})
    assert not twice, f"saved twice: {twice}"
    root = f"{JOB['output']}/{JOB['name']}/"
    assert all(n.startswith(root) for n in names), f"a save outside {root}"


def the_capture_command_reads_what_the_preview_saves():
    g = graphs.preview(JOB)
    saved = {v for n in g.values() for k, v in n["inputs"].items() if k in ("filename_prefix", "table_prefix")}
    lines = tool.capture_command(JOB, "2026-01-01").split(" \\\n")
    assert f"--first {JOB['first']} --frames {JOB['frames']}" in lines[0] and JOB["source"] in lines[0], lines[0]
    assert len(lines) == 1 + len(JOB["subjects"]), f"{len(lines) - 1} --mask lines for {len(JOB['subjects'])} subjects"
    for subject, line in zip(JOB["subjects"], lines[1:]):
        who, spec = subject["label"], line.strip().split(" ", 1)[1]
        assert spec.startswith(f"{who}:"), f"{who}: the line is {spec[:30]!r}"
        given = dict(pair.split("=", 1) for pair in spec.split(":", 1)[1].split(","))
        want = {"track", "parts", "classes", "shots", "by", "at"} | ({"pose"} if subject.get("pose", True) else set())
        assert set(given) == want, f"{who}: the capture is given {sorted(given)}"
        assert given["at"] == str(JOB["first"]), f"{who}: at={given['at']}"
        for key in want - {"by", "at"}:
            prefix = graphs.prefixes(JOB, who)[key]
            assert prefix in saved and given[key].startswith(f"<{prefix}"), f"{who}: {key} is {given[key]}, the preview saves {prefix}"
    return "a subject's held things are left out: they are not a part"


def _first_frame_at(clip: Path, start_time: float) -> int:
    """The number of the first frame the video loader yields from `start_time`, by its own ffmpeg arguments
    (`ComfyUI-VideoHelperSuite`, `load_video_nodes.ffmpeg_frame_generator`): an output seek, behind an input
    seek four seconds short when the time is past four seconds."""
    before, after = ([], [])
    if start_time > 4:
        before, after = ["-ss", str(start_time - 4)], ["-ss", "4"]
    elif start_time > 0:
        after = ["-ss", str(start_time)]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-an", *before, "-i", str(clip), *after, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    assert len(raw) == 16 * 16, f"no frame from {start_time}"
    return round(sum(raw) / len(raw))


def the_seek_lands_on_the_frame():
    if not shutil.which("ffmpeg"):
        skip("no ffmpeg on the path")
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "numbered.mkv"
        # every frame's grey level is its own number, made in grey so that no range conversion moves a level
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=16x16:r=24:d=10",
                        "-vf", "format=gray,geq=lum='N'", "-c:v", "ffv1", str(clip)], check=True)
        asked = (0, 1, 2, 23, 95, 96, 97, 98, 150, 239)
        got = {n: _first_frame_at(clip, graphs.seek(n)) for n in asked}
        assert all(n == f for n, f in got.items()), f"asked and got: {got}"
        # and the reader can be one off: a time on the frame before lands on the frame before
        assert _first_frame_at(clip, graphs.seek(150) - 1 / cfg.FPS) == 149, "the reader cannot tell one frame from the next"
    return f"{len(asked)} frames, both sides of the loader's four-second rule"


# ---- a load -------------------------------------------------------------------

WINDOWS = tool.chain_lengths()
BASE = json.loads((REPO / "workflows" / cfg.MASKED_JOB_BASE_GRAPH).read_text())
#: Two subjects with a mesh and one without; the capture starts later than the job, so a mask's clock and the
#: mesh's are different numbers.
LOAD_JOB = {"name": "check", "source": "/clips/check_24p.mov", "first": 604, "frames": 447, "canvas": [1024, 768],
            "output": "Video/check", "texts": {"w": "words.txt"},
            "subjects": [{"label": "a", "pick": "largest", "parts": ["hair"], "pose": False},
                         {"label": "b", "pick": "largest"}, {"label": "c", "pick": "largest"}]}
CAPTURE_FIRST = 650
WHOLE = {"name": "b_0700", "first": 700, "frames": 100, "subject": "b", "others": ["a", "c"], "margin": 48, "feather": 4,
         "edge": "cells", "still": "b.png", "seed": 11, "audio": "/audio/voice.wav", "audio_first": 620, "text": "w"}
PART = {"name": "a_0700", "first": 700, "frames": WINDOWS[0], "subject": "a", "others": ["b"], "margin": 16,
        "carried": "held", "still": "a.png", "continue_from": "before.safetensors", "text": "w"}
FILES_WHOLE = {"track": ("/capture/b/track.mkv", CAPTURE_FIRST), "parts": None, "motion": ("/output/b_mesh.mp4", 604),
               "others": [("/capture/a/track.mkv", CAPTURE_FIRST), ("/capture/c/track.mkv", CAPTURE_FIRST)]}
FILES_PART = {"track": ("/capture/a/track.mkv", CAPTURE_FIRST), "parts": ("/capture/a/parts_held.mkv", CAPTURE_FIRST),
              "motion": None, "others": [("/capture/b/track.mkv", CAPTURE_FIRST)]}


def _a_load(one=WHOLE, files=FILES_WHOLE, plan=False, job=LOAD_JOB) -> dict:
    return graphs.load(BASE, job, one, files, "the words", windows=WINDOWS, plan=plan)


def _files_behind(g: dict, link: list) -> set[tuple[str, float, int]]:
    """The video files a link is read from, through any mask conversion or join: (path, start time, frames)."""
    node = g[link[0]]
    if node["class_type"] == cfg.MASKED_JOB_LOADER:
        return {(node["inputs"]["video"], node["inputs"]["start_time"], node["inputs"]["frame_load_cap"])}
    links = [v for v in node["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in g]
    assert node["class_type"] in ("ImageToMask", "MaskComposite") and links, f"a mask is read through a {node['class_type']}"
    return set().union(*(_files_behind(g, v) for v in links))


def _read(one: dict, files) -> set:
    """What `_files_behind` should find for a load's file: the load's own frames, on the file's own clock."""
    several = files if isinstance(files, list) else [files]
    return {(path, graphs.seek(one["first"] - at), one["frames"]) for path, at in several}


def a_load_lands_on_the_inputs_it_names():
    g = _a_load()
    src, run = g[graphs._the(g, graphs.SOURCE)]["inputs"], g[graphs._the(g, graphs.SONG)]["inputs"]
    assert _files_behind(g, src["frames"]) == {(LOAD_JOB["source"], graphs.seek(WHOLE["first"]), WHOLE["frames"])}, "the source's frames"
    assert _files_behind(g, src["mask"]) == _read(WHOLE, FILES_WHOLE["track"]), f"the subject's mask: {_files_behind(g, src['mask'])}"
    assert _files_behind(g, src["others"]) == _read(WHOLE, FILES_WHOLE["others"]), f"the others: {_files_behind(g, src['others'])}"
    assert _files_behind(g, src["motion_video"]) == _read(WHOLE, FILES_WHOLE["motion"]), f"the motion video: {_files_behind(g, src['motion_video'])}"
    assert g[src["motion_video"][0]]["class_type"] == cfg.MASKED_JOB_LOADER, "the motion video is read as a mask"
    assert src["replace"] == cfg.MASKED_SOURCE["replace"] and "parts" not in src, "a whole-subject load takes a part"
    assert (src["grow_pixels"], src["feather_pixels"], src["edge"]) == (48, 4, cfg.MASKED_JOB_EDGE["cells"]), "margin, feather or edge"
    assert src["motion_reference"] == cfg.MASKED_JOB_MOTION["wired"] and src["motion_vae"] is True, "the mesh is not wired to the video model"
    assert src["motion_short_edge"] == min(LOAD_JOB["canvas"]), f"the motion video goes in at {src['motion_short_edge']}"
    assert run["window_frames"] == WINDOWS[0] > WHOLE["frames"] and src.get("held_tail") == cfg.MASKED_JOB_HELD_TAIL, \
        "a load shorter than its window does not keep its region open over the held frames"
    assert (run["prompt"], run["seed"], run["preview"]) == ("the words", 11, False), "text, seed or preview"
    assert (run["width"], run["height"], run["context_frames"]) == (*LOAD_JOB["canvas"], cfg.MASKED_JOB_CONTEXT), "canvas or context"
    assert run["extent"] == "first_seconds" and abs(run["extent.seconds"] * cfg.FPS - WHOLE["frames"]) < 1e-6, "the extent is not the load's frames"
    assert run["filename_prefix"] == graphs.run_prefix(LOAD_JOB, WHOLE, False) and run["timeline"] == "", "prefix or timeline"
    assert "continue_from" not in run and "context_noise" not in run, "a load of its own is given kept frames"
    audio = g[run["audio"][0]]
    assert audio["class_type"] == "VHS_LoadAudio" and audio["inputs"]["audio_file"] == WHOLE["audio"], "the audio file"
    assert abs(audio["inputs"]["seek_seconds"] * cfg.FPS - (WHOLE["first"] - WHOLE["audio_first"])) < 1e-3, "the audio does not start on the load's first frame"
    assert abs(audio["inputs"]["duration"] * cfg.FPS - WHOLE["frames"]) < 1e-3, "the audio is not cut to the load"
    assert g[graphs._the(g, "LoadImage")]["inputs"]["image"] == WHOLE["still"], "the still"
    view = {k: v for k, v in g[graphs._the(g, graphs.STILL)]["inputs"].items() if k.startswith("qwen_view")}
    assert view == {"qwen_view": cfg.MASKED_JOB_STILL_VIEW}, f"the text encoder's view of the still: {view}"
    sinks = [n["inputs"]["sink_conditioning"] for n in g.values() if "sink_conditioning" in n["inputs"]]
    assert sinks == [cfg.MASKED_JOB_SOL_SINK], f"Sol's sink: {sinks}"

    g = _a_load(PART, FILES_PART)
    src, run = g[graphs._the(g, graphs.SOURCE)]["inputs"], g[graphs._the(g, graphs.SONG)]["inputs"]
    assert src["replace"] == cfg.MASKED_PARTS_SOURCE["replace"], "a part load replaces the whole subject"
    assert _files_behind(g, src["parts"]) == _read(PART, FILES_PART["parts"]), f"the part: {_files_behind(g, src['parts'])}"
    assert _files_behind(g, src["others"]) == _read(PART, FILES_PART["others"]), "the part load's others"
    assert src["motion_reference"] == cfg.MASKED_JOB_MOTION["none"] and src["motion_vae"] is False and "motion_video" not in src, \
        "a part pass that asked for no motion video has one"
    assert run["window_frames"] == PART["frames"] and "held_tail" not in src, "a load that fills its window is given a held tail"
    assert run["continue_from"] == PART["continue_from"] and run["context_noise"] == cfg.MASKED_JOB_CONTEXT_NOISE, "a continuation's kept frames"
    assert run["audio"] == [src["frames"][0], 2], "with no audio file the model hears something other than the source's own track"
    assert run["seed"] == BASE[graphs._the(BASE, graphs.SONG)]["inputs"]["seed"], "a load with no seed does not take the row's"
    plate = _a_load({**WHOLE, "held_tail": "plate", "motion_edge": 512})
    assert "held_tail" not in plate[graphs._the(plate, graphs.SOURCE)]["inputs"], "a load that asks for the plate is given an open tail"
    assert plate[graphs._the(plate, graphs.SOURCE)]["inputs"]["motion_short_edge"] == 512, "a load's own motion_edge is not used"
    quieter = _a_load({**PART, "context_noise": 0.5}, FILES_PART)
    assert quieter[graphs._the(quieter, graphs.SONG)]["inputs"]["context_noise"] == 0.5, "a load's own context_noise is not used"
    zoomed = _a_load({**PART, "motion": "mesh zoomed"}, {**FILES_PART, "motion": ("/output/a_mesh.mp4", 604)})
    src = zoomed[graphs._the(zoomed, graphs.SOURCE)]["inputs"]
    assert src["motion_reference"] == cfg.MASKED_JOB_MOTION["zoomed"] and src["motion_box"] == cfg.MASKED_JOB_PART_BOX, \
        "a zoomed mesh on a part pass is not framed on the part"
    return "a whole-subject load and a part load"


def a_plan_is_the_render_with_nothing_sampled():
    for one, files in ((WHOLE, FILES_WHOLE), (PART, FILES_PART)):
        plan, render = _a_load(one, files, plan=True), _a_load(one, files)
        song = graphs._the(render, graphs.SONG)
        assert plan[song]["inputs"]["preview"] is True and render[song]["inputs"]["preview"] is False, "the song node's preview"
        assert plan[song]["inputs"]["filename_prefix"] != render[song]["inputs"]["filename_prefix"], "a plan saves into the render's folder"
        for g in (plan, render):
            g[song]["inputs"].pop("preview"), g[song]["inputs"].pop("filename_prefix")
        assert plan == render, f"{one['name']}: the plan is another graph than the render"


def a_load_leaves_the_rows_model_alone():
    g = _a_load()
    named = {cfg.MASKED_JOB_LOADER, graphs.SOURCE, graphs.SONG, graphs.STILL, "LoadImage"}
    kept = [nid for nid, n in g.items() if nid in BASE and n["class_type"] not in named and "sink_conditioning" not in n["inputs"]]
    moved = [f"{nid} ({g[nid]['class_type']})" for nid in kept if g[nid] != BASE[nid]]
    assert not moved, f"changed from the shipped graph: {moved}"
    classes = {g[nid]["class_type"] for nid in kept}
    assert {"UNETLoader", "BasicScheduler", "KSamplerSelect", "VAELoader"} <= classes, f"only {sorted(classes)} were compared"
    still = graphs._the(g, graphs.STILL)
    differs = {k for k in set(g[still]["inputs"]) | set(BASE[still]["inputs"]) if g[still]["inputs"].get(k) != BASE[still]["inputs"].get(k)}
    assert differs and all(k.startswith("qwen_view") for k in differs), f"the still's node differs from the shipped graph in {sorted(differs)}"
    sol = next(nid for nid, n in g.items() if "sink_conditioning" in n["inputs"])
    differs = {k for k in g[sol]["inputs"] if g[sol]["inputs"][k] != BASE[sol]["inputs"].get(k)}
    assert differs <= {"sink_conditioning"}, f"Sol differs from the shipped graph in {sorted(differs)}"
    gone = {n["class_type"] for nid, n in BASE.items() if nid not in g}
    assert {"MiniMaxH3SubjectTrack", "MiniMaxH3SubjectParts"} <= gone, f"a load still tracks: left out are {sorted(gone)}"
    return f"{len(kept)} nodes as shipped"


def the_nodes_schemas_accept_a_load():
    import build_workflows
    oi = declared()
    for label, g in (("whole", _a_load()), ("whole plan", _a_load(plan=True)), ("part", _a_load(PART, FILES_PART)),
                     ("part, zoomed mesh", _a_load({**PART, "motion": "mesh zoomed"}, {**FILES_PART, "motion": ("/output/a.mp4", 604)}))):
        ours = ours_only(g, oi)
        assert {graphs.SOURCE, graphs.SONG, "MiniMaxH3AppendRefImage"} <= {n["class_type"] for n in ours.values()}, \
            f"{label}: the nodes a load changes were not found to grade, only {sorted({n['class_type'] for n in ours.values()})}"
        errors = build_workflows.validate_api(ours, oi, label)
        assert not errors, "\n      ".join(errors)


def _unbuildable(words: str, one: dict, files: dict = FILES_WHOLE, plan: bool = False) -> None:
    try:
        _a_load(one, files, plan=plan)
    except ValueError as e:
        assert all(w in str(e) for w in words.split("|")), f"refused without saying {words!r}: {e}"
        return
    raise AssertionError(f"built a load that should be refused for {words!r}")


def a_load_the_graph_cannot_be_built_from():
    _unbuildable("feather|margin", {**WHOLE, "feather": 49})
    _unbuildable("subject|'z'", {**WHOLE, "subject": "z"})
    _unbuildable("others", {**WHOLE, "others": ["b"]})
    _unbuildable("span", {**WHOLE, "first": 600})
    _unbuildable("span", {**WHOLE, "frames": 400})
    _unbuildable("carried", {**WHOLE, "carried": "all"})
    _unbuildable("motion", {**WHOLE, "motion": "skeleton"})
    _unbuildable("edge", {**WHOLE, "edge": "pixels"})
    _unbuildable("window", {**WHOLE, "window": 100})
    _unbuildable("held_tail", {**WHOLE, "held_tail": "shut"})
    _unbuildable("context", {**WHOLE, "context": 40})
    _unbuildable("context", {**WHOLE, "context": WINDOWS[0]})               # a context as long as its window
    _unbuildable("context_noise|continue_from", {**WHOLE, "context_noise": 0.5})
    _unbuildable("name", {**WHOLE, "name": "b 0700"})
    _unbuildable("part", {**WHOLE, "carried": "held"})                      # a part pass with no part mask
    _unbuildable("motion", WHOLE, {**FILES_WHOLE, "motion": None})          # a mesh by default and none found
    _unbuildable("still", {k: v for k, v in WHOLE.items() if k != "still"})
    _unbuildable("audio", {**WHOLE, "audio_first": 701})
    assert _a_load({k: v for k, v in WHOLE.items() if k != "still"}, plan=True), "a plan needs no still"
    assert [graphs.window_for(n, WINDOWS) for n in (1, WINDOWS[0], WINDOWS[0] + 1, WINDOWS[-1], WINDOWS[-1] + 1)] == \
        [WINDOWS[0], WINDOWS[0], WINDOWS[1], WINDOWS[-1], WINDOWS[-1]], "the window is not the shortest that holds the load"


def _a_capture(tmp: Path) -> Path:
    """A capture folder as `bench/capture_masked_run.py files` and `verify` leave one, with empty mask files."""
    capture = tmp / "data" / "2026-01-01_check"
    for label, files in (("a", ("track__p.mkv", "parts__p.mkv", "parts_held__p.mkv")), ("b", ("track__p.mkv",)), ("c", ("track__p.mkv",))):
        (capture / "subjects" / label).mkdir(parents=True)
        for name in files:
            (capture / "subjects" / label / name).write_bytes(b"")
    (capture / "manifest.json").write_text(json.dumps({
        "name": "check", "first_frame": CAPTURE_FIRST, "frames": 300, "runs": [],
        "subjects": [{"label": label, "sightings": [{"by": "p"}]} for label in "abc"]}))
    # what the capture measured of `b`: a shot table with one cut inside the whole-subject load, and a row a frame
    # whose share of the frame is its own frame number, so a middle, a least and a most are known
    (capture / "subjects" / "b" / "shots__p.json").write_text(json.dumps(
        {"shots": [{"first_frame": 0}, {"first_frame": 90}, {"first_frame": 250}]}))
    (capture / "subjects" / "b" / "per_frame.json").write_text(json.dumps({"columns": [], "rows": [
        {"source_frame": n, "seen_by": "p", "track_share": n / 1000, "parts_share": None,
         "track_box": None if n == 705 else [10, 20, 110, 20 + 160]} for n in range(CAPTURE_FIRST, CAPTURE_FIRST + 300)]}))
    (capture / "status.json").write_text(json.dumps({"state": "done"}))
    (capture / "verify.json").write_text(json.dumps({"failed": 0, "not_made": 0}))
    return capture


def _loads_file(tmp: Path, capture: Path, loads=(WHOLE, PART), **job) -> Path:
    from PIL import Image
    (tmp / "words.txt").write_text("the words\n")
    (tmp / "input").mkdir(exist_ok=True)
    # one still smaller than the size it is shown at, one larger
    Image.new("RGB", (512, 640)).save(tmp / "input" / "b.png")
    Image.new("RGB", (4096, 5000)).save(tmp / "input" / "a.png")
    path = tmp / "loads.json"
    path.write_text(json.dumps({"job": {**LOAD_JOB, "capture": str(capture), **job}, "defaults": {}, "loads": list(loads)}))
    return path


def loads_from_a_capture():
    import importlib
    planner = importlib.import_module(f"{REPO.name}.loop_plan")
    assert WINDOWS == planner.CHAIN_LENGTHS, "the windows read from the source are not the planner's"

    def planner_takes(window: int, context: int) -> bool:
        try:
            planner.check_window_settings(window, context)
        except ValueError:
            return False
        return True
    for window in WINDOWS:
        theirs = tuple(c for c in range(1, WINDOWS[-1] + 60) if planner_takes(window, c))
        assert graphs.contexts_for(window, WINDOWS) == theirs and cfg.MASKED_JOB_CONTEXT in theirs, \
            f"in a window of {window} the builder takes contexts {graphs.contexts_for(window, WINDOWS)}, the planner {theirs}"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture = _a_capture(tmp)
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input")
        (tmp / "output").mkdir()
        # no mesh saved yet: the whole-subject load takes one by default and is refused, and nothing is written
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture), "plan", [], **where), "'b_0700'", "mesh")
        folder = tmp / "data" / "2026-01-01_check_job"
        assert not folder.exists(), "a refused build left a folder behind"
        # the preview's record names the mesh it saved
        folder.mkdir()
        mesh = tmp / "output" / "Video/check/check" / "check_b_mesh_00001.mp4"
        mesh.parent.mkdir(parents=True)
        mesh.write_bytes(b"")
        (folder / "preview.queued.json").write_text(json.dumps(
            {"files": ["Video/check/check/check_b_mesh_00001.mp4", "Video/check/check/check_b_track_00001.mkv"]}))
        written = tool.build_loads(_loads_file(tmp, capture), "plan", [], **where)
        assert [p.name for p in written] == ["b_0700.plan.json", "a_0700.plan.json"], [p.name for p in written]
        found = {"track": (str(capture / "subjects/b/track__p.mkv"), CAPTURE_FIRST), "parts": None, "motion": (str(mesh), LOAD_JOB["first"]),
                 "others": [(str(capture / "subjects/a/track__p.mkv"), CAPTURE_FIRST), (str(capture / "subjects/c/track__p.mkv"), CAPTURE_FIRST)]}
        job = json.loads((folder / "job.plan.json").read_text())["job"]
        assert json.loads(written[0].read_text()) == graphs.load(BASE, job, WHOLE, found, "the words", windows=WINDOWS, plan=True), \
            "the plan written is not the graph of the capture's own files"
        part = json.loads(written[1].with_suffix(".built.json").read_text())
        assert part["files"]["parts"][0].endswith("parts_held__p.mkv"), f"a part load takes {part['files']['parts']}, not the capture's filled part"
        assert part["files"]["motion"] is None and any("no motion video" in line for line in part["shown"]), "the part load's motion"
        assert any("held" in line and "open" in line for line in json.loads(written[0].with_suffix(".built.json").read_text())["shown"]), \
            "what a short load is shown does not say its held tail"
        assert (folder / "b_0700.plan.txt").read_text().strip() == "the words", "the text was not kept beside the graph"
        whole = json.loads(written[0].with_suffix(".built.json").read_text())["shown"]
        shown_at = BASE[graphs._the(BASE, graphs.STILL)]["inputs"]["size_policy.dit_short_edge"]
        assert all("that same copy" in line for line in (whole[1], part["shown"][1])), f"the text encoder's view is not said: {whole[1]}"
        assert any("b.png is 512x640" in line and f"ENLARGED by {shown_at / 512:.2f}" in line for line in whole), \
            f"a still smaller than the size it is shown at is not said to be enlarged: {whole[1]}"
        assert any("a.png is 4096x5000" in line and "reduced by" in line for line in part["shown"]), f"a larger still: {part['shown'][1]}"
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**PART, "name": "a_other", "still": "absent.png"},)),
                                          "render", [], **where), "absent.png", "H3_INPUT_DIR")
        # which settings the job set, and what the capture measured on this clip for the load
        record = json.loads(written[0].with_suffix(".built.json").read_text())
        by_job = {k for k, v in record["settings"].items() if v["set_by"] == "the job file"}
        assert by_job == set(WHOLE) & set(graphs.SETTINGS) and set(record["settings"]) == set(graphs.SETTINGS), \
            f"said to be set by the job file: {sorted(by_job)}"
        assert record["settings"]["context"] == {"value": cfg.MASKED_JOB_CONTEXT, "set_by": "the builder's default"}, record["settings"]["context"]
        assert record["settings"]["margin"]["value"] == WHOLE["margin"], record["settings"]["margin"]
        seen = record["what_this_clips_preview_measured"]
        last = WHOLE["first"] + WHOLE["frames"] - 1
        assert seen["cuts_on_source_frames"] == [CAPTURE_FIRST + 90, CAPTURE_FIRST + 250], seen["cuts_on_source_frames"]
        assert seen["cuts_inside_this_load"] == [CAPTURE_FIRST + 90], f"the cuts inside {WHOLE['first']}..{last}: {seen['cuts_inside_this_load']}"
        assert seen["subject"]["frames_read"] == WHOLE["frames"] and seen["subject"]["frames_with_no_mask"] == 1, seen["subject"]
        assert seen["subject"]["share_of_the_frame"] == {"least": WHOLE["first"] / 1000, "middle": (WHOLE["first"] + WHOLE["frames"] // 2) / 1000,
                                                         "most": last / 1000}, seen["subject"]["share_of_the_frame"]
        assert seen["subject"]["box_height_px_middle"] == 160 and "not in the capture" in seen["part"], seen
        assert seen["load_against_its_window"] == {"frames": WHOLE["frames"], "window": WINDOWS[0], "held_past_the_load": WINDOWS[0] - WHOLE["frames"]}
        absent = part["what_this_clips_preview_measured"]
        assert "not in the capture" in absent["subject"] and "not in the capture" in absent["cuts_on_source_frames"], \
            f"a figure the capture does not hold is not said to be absent: {absent}"
        # a subject's still stands for its loads; a load's own wins; a stretch left out on purpose is carried
        no_still = {k: v for k, v in WHOLE.items() if k != "still"}
        subjects = [dict(s, still="a.png") if s["label"] == "b" else s for s in LOAD_JOB["subjects"]]
        gaps = [{"subject": "c", "first": 800, "frames": 40, "why": "small in an inset; waits for the zoomed render"}]
        both = tool.build_loads(_loads_file(tmp, capture, loads=({**no_still, "name": "b_subjects_still"}, {**WHOLE, "name": "b_own_still"}),
                                            subjects=subjects, left_out=gaps), "plan", [], **where)
        for path, still in zip(both, ("a.png", "b.png")):
            graph, record = json.loads(path.read_text()), json.loads(path.with_suffix(".built.json").read_text())
            assert graph[graphs._the(graph, "LoadImage")]["inputs"]["image"] == still, f"{path.name} loads {graph[graphs._the(graph, 'LoadImage')]['inputs']['image']}"
            assert record["left_out_on_purpose"] == gaps, record["left_out_on_purpose"]
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**WHOLE, "name": "b_gap"},), left_out=[{"subject": "c", "first": 800, "frames": 40}]),
                                          "plan", [], **where), "left_out", "why")
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**WHOLE, "name": "b_gap"},), left_out=[{**gaps[0], "subject": "z"}]),
                                          "plan", [], **where), "'z'")
        # the same load again; then the render, which is another stage and another folder
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture), "plan", ["b_0700"], **where), "b_0700.plan.json", "new `name`")
        rendered = tool.build_loads(_loads_file(tmp, capture), "render", ["b_0700"], **where)
        built = json.loads(rendered[0].with_suffix(".built.json").read_text())
        assert built["output_folder"].endswith("check/b_0700") and built["capture"] == str(capture), built["output_folder"]
        Path(built["output_folder"]).mkdir(parents=True)
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**WHOLE, "motion": "none"},)), "render", [],
                                          **{**where, "date": "2026-01-02"}), "output folder")
        # what a capture must be, and what a load must have
        later = {**where, "date": "2026-01-03"}
        one = {**PART, "name": "a_new"}
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(one,)), "plan", ["nobody"], **later), "nobody")
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**one, "text": "x"},)), "plan", [], **later), "text")
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**one, "first": 610},)), "plan", [], **later), "holds 650")
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**one, "by": "q"},)), "plan", [], **later), "seen by")
        _refused(lambda: tool.build_loads(_loads_file(tmp, tmp / "data" / "nowhere", loads=(one,)), "plan", [], **later), "not a capture")
        (capture / "subjects/a/parts_held__p.mkv").unlink()
        made = tool.build_loads(_loads_file(tmp, capture, loads=(one,)), "plan", [], **later)
        assert json.loads(made[0].with_suffix(".built.json").read_text())["files"]["parts"][0].endswith("parts__p.mkv"), \
            "with nothing doubted the part as saved is not taken"
        (capture / "subjects/a/parts__p.mkv").unlink()
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**one, "name": "a_newer"},)), "plan", [], **later), "parts")
        whole = {**one, "name": "a_newest", "carried": "track", "motion": "none"}
        (capture / "verify.json").write_text(json.dumps({"failed": 1, "not_made": 0}))
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(whole,)), "plan", [], **later), "verify")
        (capture / "verify.json").unlink()
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(whole,)), "plan", [], **later), "verify")
        (capture / "status.json").write_text(json.dumps({"state": "running"}))
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(whole,)), "plan", [], **later), "not finished")
    return "two loads planned and one rendered from a made capture"


def the_gate_before_a_render():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture, folder = _a_capture(tmp), tmp / "data" / "2026-01-01_check_job"
        folder.mkdir()
        built = {"capture": str(capture), "load": {"name": "b_0700"}}
        ok = lambda c: 0                       # noqa: E731
        _refused(lambda: tool.gate(built, folder, lambda c: 4), "verify", "exited 4")
        _refused(lambda: tool.gate(built, folder, ok), "flags.json")
        (capture / "flags.json").write_text(json.dumps({"verdict": "blocked", "blocking": ["f003"]}))
        _refused(lambda: tool.gate(built, folder, ok), "blocked", "f003", "outcome")
        (capture / "flags.json").write_text(json.dumps({"verdict": "clear", "blocking": [], "overridden": ["f003"]}))
        _refused(lambda: tool.gate(built, folder, ok), "plan", "never queued")
        (folder / "b_0700.plan.queued.json").write_text("{}")
        _refused(lambda: tool.gate(built, folder, ok), "no run named 'b_0700'")
        manifest = json.loads((capture / "manifest.json").read_text())
        manifest["runs"] = [{"name": "b_0700", "read_from": "worked out from the saved masks: a plan, nothing rendered"}]
        (capture / "manifest.json").write_text(json.dumps(manifest))
        _refused(lambda: tool.gate(built, folder, ok), "worked", "node's own plan")
        manifest["runs"] = [{"name": "b_0700", "read_from": "the region the node's preview planned for each window", "plan": "b_0700_plan.json"}]
        (capture / "manifest.json").write_text(json.dumps(manifest))
        found = tool.gate(built, folder, ok)
        assert found == {"verify": 0, "verdict": "clear", "overridden": ["f003"], "plan_read": "b_0700_plan.json"}, found
        _refused(lambda: tool.stem_of("render"), "--load")
        # what a finished run left in the server's history, as the record keeps it
        read = tool.read_history({"status": {"status_str": "success", "messages": [
            ["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 63000}]]},
            "outputs": {"7": {"gifs": [{"filename": "b_mesh_00001.mp4", "subfolder": "Video/check/check", "type": "output"}],
                              "text": ["said by seven"]},
                        "3": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}})
        assert read == {"status": "success", "seconds": 62, "files": ["Video/check/check/b_mesh_00001.mp4", "a.png"],
                        "said": {"7": ["said by seven"]}}, read
        failed = tool.read_history({"status": {"status_str": "error", "messages": [
            ["execution_error", {"node_type": "X", "exception_type": "E", "exception_message": "why"}]]}, "outputs": {}})
        assert failed["status"] == "error" and failed["error"]["exception_message"] == "why" and failed["files"] == [], failed
        assert (tool.stem_of("preview"), tool.stem_of("render", "b_0700")) == ("preview", "b_0700.render")


def _job_file(tmp: Path, job: dict) -> Path:
    path = tmp / "job.json"
    path.write_text(json.dumps({"job": job, "defaults": {}, "loads": []}))
    return path


def _refused(fn, *words: str) -> None:
    try:
        fn()
    except tool.Refused as e:
        lacking = [w for w in words if w not in str(e)]
        assert not lacking, f"refused without saying {lacking}: {e}"
        return
    raise AssertionError("was not refused")


def _without_corrections() -> dict:
    job = copy.deepcopy(JOB)
    for subject in job["subjects"]:
        subject.pop("corrections", None)
    return job


def build_refuses_what_it_must():
    version = tool.tracker_mask_version()
    assert version == _pack()["MiniMaxH3SubjectTrack"].MASK_VERSION, "the version read from the source is not the node's"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output")
        (tmp / "output").mkdir()
        # a correction with no version, with another version, and then with today's
        _refused(lambda: tool.build(_job_file(tmp, JOB), "preview", **where), "'b'", "corrections_at", str(version))
        job = copy.deepcopy(JOB)
        job["subjects"][1]["corrections_at"] = version - 1
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **where), str(version - 1), str(version))
        assert not (tmp / "data").exists(), "a refused build left a folder behind"
        job["subjects"][1]["corrections_at"] = version
        folder = tool.build(_job_file(tmp, job), "preview", **where)
        assert json.loads((folder / "preview.json").read_text()) == graphs.preview(job), "the graph written is not the graph built"
        built = json.loads((folder / "preview.built.json").read_text())
        assert built["tracker_mask_version"] == version and set(built["expects"]) == {"a", "b", "c"}, built
        assert set(built["expects"]["c"]) == set(tool.SAVED_AS) - {"mesh", "pose"} and set(built["expects"]["a"]) == set(tool.SAVED_AS), \
            "what the preview must leave on disk does not follow who has a pose pass"
        # every save of the graph is one of the things looked for afterwards, and the other way round
        saved = {v for n in graphs.preview(job).values() for k, v in n["inputs"].items() if k in ("filename_prefix", "table_prefix")}
        looked_for = {p[:-len(tool.SAVED_AS[k])] for things in built["expects"].values() for k, p in things.items()}
        assert looked_for == saved, f"saved and not looked for: {sorted(saved - looked_for)}; looked for and not saved: {sorted(looked_for - saved)}"
        out = tmp / "output"
        for name in ("check_a_track_00001.mkv", "check_a_track_00001.png", "check_a_shots_00001_shots.json", "check_a_pose_table.json"):
            (out / "Video/check/check").mkdir(parents=True, exist_ok=True)
            (out / "Video/check/check" / name).write_bytes(b"")
        found, missing = tool.found_on_disk({"a": built["expects"]["a"]}, out)
        assert found["a"]["track"] == ["Video/check/check/check_a_track_00001.mkv"] and found["a"]["shots"] and found["a"]["pose"], found
        assert sorted(m.split(" ")[1] for m in missing) == ["classes", "held_things", "mesh", "parts"], f"said to be missing: {missing}"
        import shutil as _sh
        _sh.rmtree(out / "Video")
        assert built["builder_not_committed"] == tool.builder_not_committed(), "the record does not say what state the builder was in"
        assert json.loads((folder / "job.json").read_text())["job"] == job, "the job file was not kept as given"
        # the same name again: the job folder is there
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **where), "job folder", "new `name`")
        # another day, the same name: the output folder would be reused
        Path(built["output_folder"]).mkdir(parents=True)
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **{**where, "date": "2026-01-02"}), "output folder")
        # a subject with no pick; a job with no block
        job2 = dict(_without_corrections(), name="check2")
        del job2["subjects"][2]["pick"]
        _refused(lambda: tool.build(_job_file(tmp, job2), "preview", **where), "'c'", "pick")
        (tmp / "bare.json").write_text(json.dumps({"defaults": {}, "loads": []}))
        _refused(lambda: tool.build(tmp / "bare.json", "preview", **where), "`job` block")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=check", "-c", "user.email=check@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def queue_refuses_what_it_must():
    oi = declared()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # a repository of its own, so "not committed" is something this case makes and unmakes
        repo = tmp / "repo"
        for path in ("subject_track.py", "bench/tool.py", "workflows/h3_config.py", "workflows/other.py"):
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text("MASK = 1\n")
        _git(repo, "init", "-q")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "a")
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output")
        (tmp / "output").mkdir()
        job = _without_corrections()
        folder = tool.build(_job_file(tmp, job), "preview", **where)
        # only this pack's nodes are in the graph the server is asked about here: the rest need a server
        (folder / "preview.json").write_text(json.dumps(ours_only(graphs.preview(job), oi)))
        graph, record = tool.before_queue(folder, "preview", oi, repo=repo)
        assert graph and record["uncommitted_node_code"] == [] and record["override"] == "", f"a clean tree gave {record}"
        # an input the server does not serve
        lacking = copy.deepcopy(oi)
        del lacking["MiniMaxH3SubjectTrack"]["input"]["optional"]["others"]
        _refused(lambda: tool.before_queue(folder, "preview", lacking, repo=repo), "MiniMaxH3SubjectTrack", "'others'")
        # node code not committed: a node file and the config count, a bench tool and another workflows file do not
        for path in ("bench/tool.py", "workflows/other.py"):
            (repo / path).write_text("MASK = 2\n")
        assert tool.uncommitted_node_code(repo) == [], f"called node code: {tool.uncommitted_node_code(repo)}"
        for path in ("subject_track.py", "workflows/h3_config.py"):
            (repo / path).write_text("MASK = 2\n")
        assert tool.builder_not_committed(repo) == ["workflows/h3_config.py"], \
            f"a changed config is not named as the builder's: {tool.builder_not_committed(repo)}"
        assert sorted(tool.uncommitted_node_code(repo)) == ["subject_track.py", "workflows/h3_config.py"], \
            f"a changed node file and the config give {tool.uncommitted_node_code(repo)}"
        _refused(lambda: tool.before_queue(folder, "preview", oi, repo=repo), "subject_track.py", "--override")
        _refused(lambda: tool.before_queue(folder, "preview", oi, repo=repo, override="because"), "WHO: WHY")
        _, record = tool.before_queue(folder, "preview", oi, repo=repo, override="check: this case")
        assert record["override"] == "check: this case" and len(record["uncommitted_node_code"]) == 2, \
            f"an override was not recorded with what it overrode: {record}"
        # the output folder appeared since the build; then: queued before
        made = Path(json.loads((folder / "preview.built.json").read_text())["output_folder"])
        assert not made.exists(), "the case's own output folder is already there"
        made.mkdir(parents=True)
        _refused(lambda: tool.before_queue(folder, "preview", oi, repo=repo, override="check: this case"), "output folder")
        made.rmdir()
        (folder / "preview.queued.json").write_text("{}")
        _refused(lambda: tool.before_queue(folder, "preview", oi, repo=repo, override="check: this case"), "queued before")
        _refused(lambda: tool.before_queue(tmp / "nowhere", "preview", oi, repo=repo), "build preview")


def main() -> int:
    for fn in (slots_are_the_outputs_they_name, the_nodes_schemas_accept_it, the_jobs_defaults_are_the_nodes,
               every_job_value_lands_on_the_input_it_names, a_later_tracker_is_told_who_the_earlier_ones_hold,
               a_preview_samples_nothing, no_two_saves_share_a_name, the_capture_command_reads_what_the_preview_saves,
               the_seek_lands_on_the_frame,
               build_refuses_what_it_must, queue_refuses_what_it_must,
               a_load_lands_on_the_inputs_it_names, a_plan_is_the_render_with_nothing_sampled,
               a_load_leaves_the_rows_model_alone, the_nodes_schemas_accept_a_load,
               a_load_the_graph_cannot_be_built_from, loads_from_a_capture, the_gate_before_a_render):
        case(fn.__name__.replace("_", " "), fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
