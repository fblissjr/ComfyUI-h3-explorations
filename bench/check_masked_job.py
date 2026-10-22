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
  a load of two subjects    `subjects`, a list of two: read as its first subject's load with the second beside
                        it (`one_subject_view`); the tracked mask is the union of the two tracks and the region
                        the union of the two parts, a whole subject's track standing for its part; the
                        keep-outs are both subjects' and never either subject; the two stills are chained in
                        the list's order and the two references sit on the node's two inputs, so the static
                        label plan gives <Picture 1>, <Picture 2>, <Video 1>, <Video 2>; one margin; the
                        nodes' schemas accept it; from a capture the record says which picture and which video
                        went to which subject and the shown lines say what the text names and that the text
                        alone tells the model which still is whose; the capture tool reads the same file as
                        its first subject's load; three subjects, one label twice, a subject among the
                        keep-outs, a key set both ways, a key a subject does not take, a second reference
                        with no first, a band on the second subject and a zoomed two-subject load are each
                        refused
  a hair band under a part    a load with `hair_band`, built from a capture whose masks are real videos: the
                        keep-out it names is wired from a file of the build's own, empty inside the band
                        (under the part's box, as deep as the share asked, a little wider) and the capture's
                        own everywhere else, on the load's frames; the band's edge does not move on a frame
                        whose part mask shrinks, where the part's own box does; the capture's file is not
                        touched; a load without the key wires the capture's; the record and the shown line
                        carry the band's figures and say no render had tested it; a band of 0, a band on a
                        whole-subject load, a keep-out the load does not have and a `hair_band_cuts` with
                        no band are each refused
  a job is built on its video's analysis folder    no folder for the file the job renders from, a file that
                        changed since the folder was made and a folder that is another video's are each
                        refused with the command that makes one, for a preview and for loads; a build
                        record names the folder and carries its fingerprint
  a load lands on the inputs it names    a whole-subject load and a part load,
                        each traced from the Masked Source and the song node
                        back to the file and the frame it should read: the
                        masks from the capture's first frame, the mesh from the
                        job's, the audio cut to the load,
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
  a text is of its frames    a text says the frames it was written from and
                        who wrote it, covers its load to the frame, holds no
                        cut of any subject's shot table and stays inside the
                        capture; one text for loads of different shots names
                        every load outside its shot; the same words under
                        two labels; the look, held by the frames and the text
                        hash it showed and by the capture's own clock; a plan
                        refused without one and a render refused at queue
                        when its text changed after the build
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
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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
                    {"label": "b", "pick": "most central", "corrections": "shot 3: person 2", "max_people": 4, "match": 0.9},
                    {"label": "c", "pick": "largest", "pose": False, "seeds": "frame 30: 100, 100, 50, 80"}]}
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
        # a subject's seeds reach its own tracker and no other's
        assert track.get("seeds", "") == subject.get("seeds", ""), f"{who}: seeds {track.get('seeds')!r}"
        if "pick_at" in subject:
            # the tracker counts from the first frame it is given, the job from the source's first
            assert track["pick_on.pick_frame"] + JOB["first"] == subject["pick_at"], \
                f"{who}: pick_frame {track['pick_on.pick_frame']}"
            assert track["pick_on"] != cfg.SUBJECT_TRACK["pick_on"], f"{who}: a frame is named and pick_on is automatic"
        else:
            assert track["pick_on"] == cfg.SUBJECT_TRACK["pick_on"] and "pick_on.pick_frame" not in track, f"{who}: pick_on"
        if "match" in subject:
            assert track["match"] == graphs.TRACKER_MATCH_AT and track["match.match_threshold"] == subject["match"], \
                f"{who}: match {track['match']!r} at {track.get('match.match_threshold')}"
            assert list(track).index("match.match_threshold") > list(track).index("match"), f"{who}: the value comes before its choice"
        else:
            assert track["match"] == cfg.SUBJECT_TRACK["match"] and "match.match_threshold" not in track, f"{who}: match {track['match']!r}"
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
    assert len(names) == 6 * len(JOB["subjects"]) + 2 * len(POSED), f"{len(names)} saves for {len(JOB['subjects'])} subjects"
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
        want = {"track", "head_track", "parts", "classes", "shots", "by", "at"} | ({"pose"} if subject.get("pose", True) else set())
        assert set(given) == want, f"{who}: the capture is given {sorted(given)}"
        assert given["at"] == str(JOB["first"]), f"{who}: at={given['at']}"
        for key in want - {"by", "at"}:
            prefix = graphs.prefixes(JOB, who)[key]
            assert prefix in saved and given[key].startswith(f"<{prefix}"), f"{who}: {key} is {given[key]}, the preview saves {prefix}"
    # By default track_head is True and saved to _head_track; turning it off sets track_head=False on 1005 and omits 1028/1029 and head_track=
    assert g["1005"]["inputs"].get("track_head") is True and g["1105"]["inputs"].get("track_head") is True, g["1005"]["inputs"]
    assert g["1028"]["inputs"]["mask"] == ["1005", graphs.SLOT["MiniMaxH3SubjectTrack"]["head_mask"]], g["1028"]
    assert g["1029"]["inputs"]["filename_prefix"] == graphs.prefixes(JOB, "a")["head_track"], g["1029"]
    job_false = copy.deepcopy(JOB)
    job_false["subjects"][0]["track_head"] = False
    g_false = graphs.preview(job_false)
    assert g_false["1005"]["inputs"].get("track_head") is False and g_false["1105"]["inputs"].get("track_head") is True, g_false["1005"]["inputs"]
    assert "1028" not in g_false and "1029" not in g_false and "1128" in g_false, sorted(g_false)
    cmd_false = tool.capture_command(job_false, "2026-01-01")
    assert "check_a_head_track" not in cmd_false and "head_track=<Video/check/check/check_b_head_track_*.mkv>" in cmd_false, cmd_false
    job_all_off = {**copy.deepcopy(JOB), "sam_head_track": {"on": False, "why": "comparison", "said_by": "owner"}}
    g_all_off = graphs.preview(job_all_off)
    assert g_all_off["1005"]["inputs"]["track_head"] is False and "1028" not in g_all_off and "1128" not in g_all_off
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
            "output": "Video/check", "texts": {"w": {"file": "words.txt", "frames": [690, 899], "by": "a check"}},
            "subjects": [{"label": "a", "pick": "largest", "parts": ["hair"], "pose": False},
                         {"label": "b", "pick": "largest"}, {"label": "c", "pick": "largest"}]}
CAPTURE_FIRST = 650
WHOLE = {"name": "b_0700", "first": 700, "frames": 100, "subject": "b", "others": ["a", "c"], "margin": 48, "feather": 4,
         "edge": "cells", "still": "b.png", "seed": 11, "audio": "/audio/voice.wav", "audio_first": 620, "text": "w"}
#: The window a load of WHOLE's frames is given: the shortest length on the video grid that holds it.
SHORT_WINDOW = graphs.window_for(WHOLE["frames"], WINDOWS)
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
    pad_before, pad_after = graphs.window_pad(WHOLE["first"], WHOLE["frames"], SHORT_WINDOW, LOAD_JOB.get("source_frames"),
                                              bool(WHOLE.get("continue_from")), floor=WHOLE.get("audio_first", LOAD_JOB["first"]))
    src_node = g[graphs._the(g, graphs.SOURCE)]
    src, run = src_node["inputs"], g[graphs._the(g, graphs.SONG)]["inputs"]
    assert _files_behind(g, src["frames"]) == {(LOAD_JOB["source"], graphs.seek(WHOLE["first"] - pad_before), SHORT_WINDOW)}, "the source's frames"
    assert src_node.get("_meta") == {"pad_before": pad_before, "pad_after": pad_after}, f"the source node's _meta: {src_node.get('_meta')}"
    assert _files_behind(g, src["mask"]) == _read(WHOLE, FILES_WHOLE["track"]), f"the subject's mask: {_files_behind(g, src['mask'])}"
    assert _files_behind(g, src["others"]) == _read(WHOLE, FILES_WHOLE["others"]), f"the others: {_files_behind(g, src['others'])}"
    assert _files_behind(g, src["motion_video"]) == _read(WHOLE, FILES_WHOLE["motion"]), f"the motion video: {_files_behind(g, src['motion_video'])}"
    assert g[src["motion_video"][0]]["class_type"] == cfg.MASKED_JOB_LOADER, "the motion video is read as a mask"
    assert src["replace"] == cfg.MASKED_SOURCE["replace"] and "parts" not in src, "a whole-subject load takes a part"
    assert (src["grow_pixels"], src["feather_pixels"], src["edge"]) == (48, 4, cfg.MASKED_JOB_EDGE["cells"]), "margin, feather or edge"
    # the composite (2026-10-16): the node's default unless the load names one; a name the node has not is refused
    # every load is built with h3_config.MASKED_LOAD_COMPOSITE unless it names another (the owner, 2026-10-16)
    assert src["composite"] == cfg.MASKED_LOAD_COMPOSITE == "whole region", f"a load naming no composite got {src.get('composite')!r}"
    old_way = graphs._the(_a_load(one={**WHOLE, "composite": "only what changed"}), graphs.SOURCE)
    assert _a_load(one={**WHOLE, "composite": "only what changed"})[old_way]["inputs"]["composite"] == "only what changed", "a load's own composite is not kept"
    named = graphs._the(_a_load(one={**WHOLE, "composite": "whole region"}), graphs.SOURCE)
    assert _a_load(one={**WHOLE, "composite": "whole region"})[named]["inputs"]["composite"] == "whole region", "a load's composite does not reach the node"
    _unbuildable("is not one of 'whole region', 'only what changed'", {**WHOLE, "composite": "everything"})
    # the builder's names are the node's own: a rename on either side goes red here (mrgan's read, 2026-10-16)
    vm = (REPO / "video_mask.py").read_text()
    assert all(f'= "{x}"' in vm for x in graphs.COMPOSITES) and "COMPOSITE_REGION = \"whole region\"" in vm, "the composite names are not the node's"
    assert "composite" in graphs.SETTINGS, "a build record would not say a job set the composite"
    assert src["motion_reference"] == cfg.MASKED_JOB_MOTION["wired"] and src["motion_vae"] is True, "the mesh is not wired to the video model"
    assert src["motion_short_edge"] == min(LOAD_JOB["canvas"]), f"the motion video goes in at {src['motion_short_edge']}"
    assert run["window_frames"] == SHORT_WINDOW >= WHOLE["frames"] and SHORT_WINDOW in graphs.both_clocks(WINDOWS) and "held_tail" not in src, \
        "a load is in another window than the shortest length on both clocks that holds it, or still sets an input the node no longer has"
    shorter = _a_load({**WHOLE, "frames": 70})
    assert shorter[graphs._the(shorter, graphs.SONG)]["inputs"]["window_frames"] == 90 < WINDOWS[0], "a 70-frame load is not in a 90-frame window"
    # At the clip's end (`source_frames=825`), a 25-frame shot 799..823 in a 39-frame window shifts context before (13 before, 1 after: 786..824)
    at_end = _a_load({**WHOLE, "first": 799, "frames": 25}, job={**LOAD_JOB, "source_frames": 825})
    end_src = at_end[graphs._the(at_end, graphs.SOURCE)]
    assert end_src.get("_meta", {}).get("pad_before") == 13 and end_src.get("_meta", {}).get("pad_after") == 1, end_src.get("_meta")
    assert _files_behind(at_end, end_src["inputs"]["frames"]) == {(LOAD_JOB["source"], graphs.seek(786), 39)}, "end-of-clip context shift"
    # And a load whose window needs more context frames than exist in the clip is refused
    _unbuildable("context frame(s) from the source, and only", {**WHOLE, "first": 700, "frames": 25, "audio_first": 700}, job={**LOAD_JOB, "source_frames": 730})
    assert (run["prompt"], run["seed"], run["preview"]) == ("the words", 11, False), "text, seed or preview"
    assert (run["width"], run["height"], run["context_frames"]) == (*LOAD_JOB["canvas"], cfg.MASKED_JOB_CONTEXT), "canvas or context"
    assert run["extent"] == "first_seconds" and abs(run["extent.seconds"] * cfg.FPS - SHORT_WINDOW) < 1e-6, "the extent is not the window's frames"
    assert run["filename_prefix"] == graphs.run_prefix(LOAD_JOB, WHOLE, False) and run["timeline"] == "", "prefix or timeline"
    assert "continue_from" not in run and "context_noise" not in run, "a load of its own is given kept frames"
    assert "save_kept" not in run, "a load that did not ask saves what the composite kept"
    asked = _a_load({**WHOLE, "save_kept": True}, FILES_WHOLE)
    assert asked[graphs._the(asked, graphs.SONG)]["inputs"].get("save_kept") is True, "a load that asked does not save what the composite kept"
    audio = g[run["audio"][0]]
    assert audio["class_type"] == "VHS_LoadAudio" and audio["inputs"]["audio_file"] == WHOLE["audio"], "the audio file"
    assert abs(audio["inputs"]["seek_seconds"] * cfg.FPS - (WHOLE["first"] - pad_before - WHOLE["audio_first"])) < 1e-3, "the audio does not start on the window's first frame"
    short = SHORT_WINDOW - audio["inputs"]["duration"] * cfg.FPS
    assert 0 < short < 1, f"the audio file is cut {short:.3f} frames short of the window: over the window it writes a frame more, a frame under it loses one"
    assert not any(n["class_type"] == "TrimAudioDuration" for n in g.values()), "a load with an audio file also cuts the source's track"
    assert g[graphs._the(g, "LoadImage")]["inputs"]["image"] == WHOLE["still"], "the still"
    view = {k: v for k, v in g[graphs._the(g, graphs.STILL)]["inputs"].items() if k.startswith("qwen_view")}
    assert view == {"qwen_view": cfg.MASKED_JOB_STILL_VIEW}, f"the text encoder's view of the still: {view}"
    reused = sorted(k for n in g.values() for k in n["inputs"] if k.startswith("reuse"))
    assert not reused, f"a load carries inputs of stores the nodes no longer have: {reused}"
    sinks = [n["inputs"]["sink_conditioning"] for n in g.values() if "sink_conditioning" in n["inputs"]]
    assert sinks == [cfg.MASKED_JOB_SOL_SINK], f"Sol's sink: {sinks}"

    g = _a_load(PART, FILES_PART)
    src, run = g[graphs._the(g, graphs.SOURCE)]["inputs"], g[graphs._the(g, graphs.SONG)]["inputs"]
    assert src["replace"] == cfg.MASKED_PARTS_SOURCE["replace"], "a part load replaces the whole subject"
    assert _files_behind(g, src["parts"]) == _read(PART, FILES_PART["parts"]), f"the part: {_files_behind(g, src['parts'])}"
    assert _files_behind(g, src["others"]) == _read(PART, FILES_PART["others"]), "the part load's others"
    assert src["motion_reference"] == cfg.MASKED_JOB_MOTION["none"] and src["motion_vae"] is False and "motion_video" not in src, \
        "a part pass that asked for no motion video has one"
    assert run["window_frames"] == PART["frames"], "a load that fills a window is given another"
    assert run["continue_from"] == PART["continue_from"] and run["context_noise"] == cfg.MASKED_JOB_CONTEXT_NOISE, "a continuation's kept frames"
    cut = g[run["audio"][0]]
    assert cut["class_type"] == "TrimAudioDuration" and cut["inputs"]["audio"] == [src["frames"][0], 2] and cut["inputs"]["start_index"] == 0, \
        "with no audio file the model hears something other than the source's own track, cut from its start"
    short = PART["frames"] - cut["inputs"]["duration"] * cfg.FPS
    assert 0 < short < 1, f"the source's track is cut {short:.3f} frames short of the load"
    assert run["seed"] == BASE[graphs._the(BASE, graphs.SONG)]["inputs"]["seed"], "a load with no seed does not take the row's"
    plate = _a_load({**WHOLE, "motion_edge": 512})
    assert plate[graphs._the(plate, graphs.SOURCE)]["inputs"]["motion_short_edge"] == 512, "a load's own motion_edge is not used"
    # a second motion video (2026-10-12): its own file on its own clock, the node's second input, at its own short edge
    assert "motion_video_2" not in src and "motion_short_edge_2" not in src, "a load that names one motion video is given a second"
    face = ("/output/b_face.mp4", 690)
    two = _a_load({**WHOLE, "motion_file_2": face[0], "motion_first_2": face[1]}, {**FILES_WHOLE, "motion_2": face})
    both = two[graphs._the(two, graphs.SOURCE)]["inputs"]
    assert _files_behind(two, both["motion_video_2"]) == _read(WHOLE, face), f"the second motion video: {_files_behind(two, both['motion_video_2'])}"
    assert _files_behind(two, both["motion_video"]) == _read(WHOLE, FILES_WHOLE["motion"]), "a second motion video moved the first"
    assert two[both["motion_video_2"][0]]["class_type"] == cfg.MASKED_JOB_LOADER, "the second motion video is read as a mask"
    assert both["motion_short_edge_2"] == min(LOAD_JOB["canvas"]) and both["motion_vae"] is True, f"the second goes in at {both['motion_short_edge_2']}"
    small = _a_load({**WHOLE, "motion_file_2": face[0], "motion_edge_2": 384}, {**FILES_WHOLE, "motion_2": face})
    assert small[graphs._the(small, graphs.SOURCE)]["inputs"]["motion_short_edge_2"] == 384, "a load's own motion_edge_2 is not used"
    stale = copy.deepcopy(BASE)
    stale[graphs._the(stale, graphs.SOURCE)]["inputs"].update(motion_video_2=["1", 0], motion_short_edge_2=96)
    clean = graphs.load(stale, LOAD_JOB, WHOLE, FILES_WHOLE, "the words", windows=WINDOWS)
    assert "motion_video_2" not in clean[graphs._the(clean, graphs.SOURCE)]["inputs"], "a base graph's second motion video reached a load that names none"
    own_run = {k: v for k, v in PART.items() if k != "continue_from"}
    close = _a_load({**own_run, "render_region": "part"}, FILES_PART)
    src = close[graphs._the(close, graphs.SOURCE)]["inputs"]
    assert src["render_region"] == graphs.RENDER_REGIONS["part"] and "largest_zoom" not in src, "a zoomed part load"
    assert "render_region" not in g[graphs._the(g, graphs.SOURCE)]["inputs"], "a load that asks for no zoom is given one"
    assert src["motion_reference"] == graphs.MOTIONS["none"] and "motion_video" not in src, "a zoomed load that names no motion video was given one"
    # A ZOOMED LOAD THAT NAMES ITS MOTION VIDEO KEEPS IT (2026-10-13; refused until then): the head reference of
    # the form on a small face. The graph carries the zoom and the wired video together, the video read as a
    # picture from the frame the load says, and the same load on the whole frame differs in `render_region` alone.
    head = ("/output/a_head_reference.mp4", 700)
    named = {**own_run, "motion": "mesh", "motion_file": head[0], "motion_first": head[1]}
    both_ways = [_a_load({**named, **zoom}, {**FILES_PART, "motion": head}) for zoom in ({"render_region": "part"}, {})]
    zoomed, whole = (b[graphs._the(b, graphs.SOURCE)]["inputs"] for b in both_ways)
    assert zoomed["render_region"] == graphs.RENDER_REGIONS["part"] and zoomed["motion_reference"] == graphs.MOTIONS["mesh"], \
        f"a zoomed load that names a motion video: {zoomed.get('render_region')!r}, {zoomed.get('motion_reference')!r}"
    reader = both_ways[0][zoomed["motion_video"][0]]
    assert reader["class_type"] == cfg.MASKED_JOB_LOADER and reader["inputs"]["video"] == head[0], "the zoomed load's motion video is not the file it names"
    assert zoomed["motion_vae"] is True and zoomed["motion_short_edge"] == min(LOAD_JOB["canvas"]), "the zoomed load's motion video goes in as any other does"
    assert {k for k in set(zoomed) | set(whole) if zoomed.get(k) != whole.get(k)} == {"render_region"}, \
        f"the zoomed load and the same load on the whole frame differ in more than render_region: {sorted(k for k in set(zoomed) | set(whole) if zoomed.get(k) != whole.get(k))}"
    quieter = _a_load({**PART, "context_noise": 0.5}, FILES_PART)
    assert quieter[graphs._the(quieter, graphs.SONG)]["inputs"]["context_noise"] == 0.5, "a load's own context_noise is not used"
    _unbuildable("motion", {**PART, "motion": "mesh zoomed"}, {**FILES_PART, "motion": ("/output/a_mesh.mp4", 604)})
    _unbuildable("motion_file_2|after a first", {**PART, "motion": "none", "motion_file_2": face[0]}, FILES_PART)
    _unbuildable("motion_edge_2|motion_file_2", {**WHOLE, "motion_edge_2": 384})
    _unbuildable("motion_first_2|motion_file_2", {**WHOLE, "motion_first_2": 700})
    return "a whole-subject load, a part load, and a load with a second motion video"


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
                     ("part, rendered zoomed in", _a_load({**{k: v for k, v in PART.items() if k != "continue_from"}, "render_region": "part"}, FILES_PART))):
        ours = ours_only(g, oi)
        assert {graphs.SOURCE, graphs.SONG, "MiniMaxH3AppendRefImage"} <= {n["class_type"] for n in ours.values()}, \
            f"{label}: the nodes a load changes were not found to grade, only {sorted({n['class_type'] for n in ours.values()})}"
        errors = build_workflows.validate_api(ours, oi, label)
        assert not errors, "\n      ".join(errors)


def a_load_says_what_its_context_frames_show():
    """What the real frames a short load is padded with show (`context_shows`, 2026-10-22): a load that names `face covered`
    sets the Masked Source's input to it and changes nothing else of the graph; one that names nothing, or `as filmed`, sets
    nothing (the node's own default); another value is refused; the node's schema takes it; the builder's two names are the
    node's own; and the build record and the lines shown say which form the load carries."""
    import build_workflows
    import masked_job as tool_
    plain, covered, filmed = (_a_load(one) for one in (WHOLE, {**WHOLE, "context_shows": "face covered"},
                                                        {**WHOLE, "context_shows": "as filmed"}))
    src = {k: g[graphs._the(g, graphs.SOURCE)]["inputs"] for k, g in (("plain", plain), ("covered", covered), ("filmed", filmed))}
    assert src["covered"].get("context_shows") == "face covered", f"a load naming `face covered` set {src['covered'].get('context_shows')!r}"
    assert "context_shows" not in src["plain"] and "context_shows" not in src["filmed"], "a load at the node's default sets the input"
    assert {k for k in set(src["covered"]) | set(src["plain"]) if src["covered"].get(k) != src["plain"].get(k)} == {"context_shows"}, \
        "a load naming `face covered` differs from the same load in more than context_shows"
    stale = copy.deepcopy(BASE)
    stale[graphs._the(stale, graphs.SOURCE)]["inputs"]["context_shows"] = "face covered"
    clean = graphs.load(stale, LOAD_JOB, WHOLE, FILES_WHOLE, "the words", windows=WINDOWS)
    assert "context_shows" not in clean[graphs._the(clean, graphs.SOURCE)]["inputs"], "a base graph's context_shows reached a load that names none"
    _unbuildable("context_shows", {**WHOLE, "context_shows": "covered"})
    vm = (REPO / "video_mask.py").read_text()
    assert graphs.CONTEXT_SHOWS == ("as filmed", "face covered") and all(f'= "{x}"' in vm for x in graphs.CONTEXT_SHOWS) \
        and "CONTEXT_SHOWS = (CONTEXT_AS_FILMED, CONTEXT_FACE_COVERED)" in vm, "the builder's context names are not the node's"
    oi = declared()
    errors = build_workflows.validate_api(ours_only(covered, oi), oi, "face covered")
    assert not errors, "\n      ".join(errors)
    assert "context_shows" in graphs.SETTINGS, "a build record would not say a job set what the context frames show"
    said, said_plain = tool_.settings_of({**WHOLE, "context_shows": "face covered"}, covered)["context_shows"], tool_.settings_of(WHOLE, plain)["context_shows"]
    assert said == {"value": "face covered", "set_by": "the job file"} and said_plain["set_by"] == "the builder's default" \
        and "as filmed" in said_plain["value"], f"the build record says {said} and {said_plain}"
    return "a load naming `face covered` sets the node's input and nothing else; the default sets nothing; another value is refused"


def a_load_of_chained_windows_takes_context_noise():
    """A load longer than its window runs as chained windows, each later one keeping the last frames of the one before, and may say
    how noised they are shown (2026-10-15; until then only a continuation could). Named, the graph carries it; not named, the graph
    carries none and the node shows them clean; and the build record says which, never "keeps no frames" of a load that keeps them."""
    import masked_job as tool_
    long = {**WHOLE, "name": "b_0650", "first": CAPTURE_FIRST, "frames": 400}
    plain = _a_load(long)
    run = plain[graphs._the(plain, graphs.SONG)]["inputs"]
    assert run["window_frames"] < long["frames"], f"the fixture is one window of {run['window_frames']}, not chained windows"
    assert "context_noise" not in run and "continue_from" not in run, "a load of chained windows that named no noise is given one"
    said = tool_.settings_of(long, plain)["context_noise"]
    assert "keeps no frames" not in str(said["value"]) and said["set_by"] == "the builder's default", f"the build record says {said}"
    noised = _a_load({**long, "context_noise": 0.8})
    run = noised[graphs._the(noised, graphs.SONG)]["inputs"]
    assert run["context_noise"] == 0.8 and "continue_from" not in run, "a load of chained windows does not carry its own context_noise"
    said = tool_.settings_of({**long, "context_noise": 0.8}, noised)["context_noise"]
    assert said == {"value": 0.8, "set_by": "the job file"}, f"the build record says {said}"
    assert {k for k in set(run) | set(plain[graphs._the(plain, graphs.SONG)]["inputs"])
            if run.get(k) != plain[graphs._the(plain, graphs.SONG)]["inputs"].get(k)} == {"context_noise"}, "the noise changed more of the song node"
    said = tool_.settings_of(WHOLE, _a_load())["context_noise"]
    assert said["value"] == "none: the load keeps no frames", f"a one-window load's build record says {said}"
    # one window that continues nothing keeps no frames: refused, in `a_load_the_graph_cannot_be_built_from`; and a load exactly as long as
    # its window is one window (THE CASE THAT GOES RED when the test is `frames >= window`, the cold reader's mutant of 2026-10-15)
    whole_window = {**WHOLE, "frames": SHORT_WINDOW}
    assert graphs.window_for(whole_window["frames"], WINDOWS) == whole_window["frames"], "the fixture is not a load exactly one window long"
    _unbuildable("context_noise|no window of it keeps frames", {**whole_window, "context_noise": 0.5})
    # per load, never every load's: a job file's `defaults` naming it is refused (the lead's ruling of 2026-10-15)
    try:
        tool_.loads_of({"defaults": {"context_noise": 0.8}, "loads": [long]})
    except tool_.Refused as e:
        assert "defaults" in str(e) and "per load" in str(e), e
    else:
        raise AssertionError("a job file's defaults gave every load a context_noise")
    assert tool_.loads_of({"loads": [{**long, "context_noise": 0.8}]})[0]["context_noise"] == 0.8, "a load's own context_noise was not read"
    return "a load of 400 frames carries 0.8 when it names it and nothing when it does not; the build record says which"


def _unbuildable(words: str, one: dict, files: dict = FILES_WHOLE, plan: bool = False, job: dict = LOAD_JOB) -> None:
    try:
        _a_load(one, files, plan=plan, job=job)
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
    _unbuildable("held_tail|no longer a key", {**WHOLE, "held_tail": "open"})
    _unbuildable("render_region", {**WHOLE, "render_region": "face"})
    _unbuildable("render_region|whole subject", {**WHOLE, "render_region": "part"})
    _unbuildable("render_region", {**WHOLE, "render_region": "subject"})
    own_run = {k: v for k, v in PART.items() if k != "continue_from"}
    _unbuildable("largest_zoom|no longer a key", {**own_run, "render_region": "part", "largest_zoom": 2}, FILES_PART)
    # a zoomed load takes a motion video only as a file it names: the preview's own body model is of the whole frame
    _unbuildable("zoomed in|whole frame|motion_file", {**own_run, "render_region": "part", "motion": "mesh"}, {**FILES_PART, "motion": ("/output/a.mp4", 604)})
    _unbuildable("zoomed in|continue_from", {**PART, "render_region": "part"}, FILES_PART)
    # a context is held to a window on the planner's list; a load that is one window off the list keeps none and is held to none
    _unbuildable("context", {**WHOLE, "window": WINDOWS[0], "context": 40})
    _unbuildable("context", {**WHOLE, "window": WINDOWS[0], "context": WINDOWS[0]})     # a context as long as its window
    assert _a_load({**WHOLE, "frames": 70, "context": 40}), "a load of one window off the planner's list was held to a context"
    # THE WINDOW A LOAD MAY NAME: a length on both clocks that holds it whole (`both_clocks`, 2026-10-21);
    # a load that continues another window takes one of the planner's (`WINDOWS`)
    _unbuildable("window", {**WHOLE, "window": SHORT_WINDOW + 1})                       # off the video grid
    _unbuildable("window", {**WHOLE, "window": 107})                                    # on the video grid, off the audio clock
    _unbuildable("window", {**WHOLE, "window": 90})                                     # on both clocks and does not hold the load
    _unbuildable("window", {**WHOLE, "window": graphs.MAX_LENGTH})                      # at MAX_LENGTH (362), off the audio clock
    assert _a_load({**WHOLE, "window": WINDOWS[-1]}), "a lone window at the longest length on both clocks was refused"
    _unbuildable("seam", {**WHOLE, "frames": 70, "continue_from": "before.safetensors", "window": 90})   # a seam off the chain
    assert _a_load({**WHOLE, "window": WINDOWS[0]}) and _a_load({**WHOLE, "window": WINDOWS[1]}), "a window on both clocks that holds the load was refused"
    _unbuildable("context_noise|one window|no window of it keeps frames", {**WHOLE, "context_noise": 0.5})
    _unbuildable("name", {**WHOLE, "name": "b 0700"})
    _unbuildable("part", {**WHOLE, "carried": "held"})                      # a part pass with no part mask
    _unbuildable("motion", WHOLE, {**FILES_WHOLE, "motion": None})          # a mesh by default and none found
    _unbuildable("still", {k: v for k, v in WHOLE.items() if k != "still"})
    _unbuildable("audio", {**WHOLE, "audio_first": 701})
    assert _a_load({k: v for k, v in WHOLE.items() if k != "still"}, plan=True), "a plan needs no still"
    # THE WINDOW THE BUILDER GIVES: the shortest length on both clocks (`h3_rules.on_both_clocks`) that holds a load one window
    # can hold; the planner's longest for a longer load; for a load that continues another window, the shortest chained length
    import h3_rules
    assert all(graphs.snapped(n) == h3_rules.snap_length(n) for n in range(2, WINDOWS[-1] + 40)), "the builder's snap is not h3_rules.snap_length"
    assert (graphs.FRAME_FACTOR, graphs.FRAME_REMAINDER) == (h3_rules.FRAME_FACTOR, h3_rules.FRAME_REMAINDER)
    assert graphs.both_clocks(WINDOWS) == tuple(n for n in range(1, graphs.MAX_LENGTH + 1) if h3_rules.on_both_clocks(n)) == (39, 90, *WINDOWS)
    top = WINDOWS[-1]
    assert [graphs.window_for(n, WINDOWS) for n in (12, 22, 23, 24, 39, 40, 90, 91, 114, WINDOWS[0], WINDOWS[0] + 1, top, top + 1, graphs.MAX_LENGTH, 2 * top)] == \
        [39, 39, 39, 39, 39, 90, 90, WINDOWS[0], WINDOWS[0], WINDOWS[0], WINDOWS[1], top, top, top, top], "the window is not the shortest length on both clocks that holds the load"
    assert all(graphs.window_for(n, WINDOWS) in graphs.both_clocks(WINDOWS) and graphs.window_for(n, WINDOWS) >= n for n in range(5, top + 1)), "a load one window holds was given a window off both clocks"
    assert [graphs.window_for(n, WINDOWS, continues=True) for n in (24, WINDOWS[0] + 1, WINDOWS[-1], WINDOWS[-1] + 1)] == \
        [WINDOWS[0], WINDOWS[1], WINDOWS[-1], WINDOWS[-1]], "a load that continues another window was given a length off the chain"


def _a_capture(tmp: Path) -> Path:
    """A capture folder as `bench/capture_masked_run.py files` and `verify` leave one, with empty mask files."""
    capture = tmp / "data" / "2026-01-01_check"
    for label, files in (("a", ("track__p.mkv", "parts__p.mkv", "parts_held__p.mkv")), ("b", ("track__p.mkv", "parts__p.mkv")), ("c", ("track__p.mkv",))):
        (capture / "subjects" / label).mkdir(parents=True)
        for name in files:
            (capture / "subjects" / label / name).write_bytes(b"")
    (capture / "manifest.json").write_text(json.dumps({
        "name": "check", "first_frame": CAPTURE_FIRST, "frames": 300, "runs": [],
        "subjects": [{"label": label, "sightings": [{"by": "p", "shots": True}]} for label in "abc"]}))
    # what the capture measured of `b`: a shot table with a cut before the two loads and one after them (source
    # frames 690 and 900: the loads are inside one shot, as a load has to be for a text to cover it), and a row a frame
    # whose share of the frame is its own frame number, so a middle, a least and a most are known
    (capture / "subjects" / "b" / "shots__p.json").write_text(json.dumps(
        {"shots": [{"first_frame": 0}, {"first_frame": 40}, {"first_frame": 250}]}))
    (capture / "subjects" / "b" / "per_frame.json").write_text(json.dumps({"columns": [], "rows": [
        {"source_frame": n, "seen_by": "p", "track_share": n / 1000, "parts_share": None,
         "track_box": None if n == 705 else [10, 20, 110, 20 + 160]} for n in range(CAPTURE_FIRST, CAPTURE_FIRST + 300)]}))
    (capture / "status.json").write_text(json.dumps({"state": "done", "written": "2026-01-01T10:00:00"}))
    (capture / "verify.json").write_text(json.dumps({"failed": 0, "not_made": 0}))
    return capture


PAIR = {"name": "pair_0700", "first": 700, "frames": 100, "margin": 64, "feather": 8, "edge": "cells", "text": "w", "seed": 11,
        "subjects": [{"subject": "a", "region": "face and hair", "still": "a.png", "motion_file": "/output/a_ref.mp4", "motion_first": 700,
                      "motion_edge": 768, "others": ["c"]},
                     {"subject": "b", "region": "face", "parts_mask": "/masks/b_face.mkv", "parts_first": 700, "still": "b.png",
                      "motion_file": "/output/b_ref.mp4", "motion_first": 690, "others": []}]}
FILES_PAIR = {"track": ("/capture/a/track.mkv", CAPTURE_FIRST), "parts": ("/capture/a/parts_held.mkv", CAPTURE_FIRST),
              "motion": ("/output/a_ref.mp4", 700), "motion_2": ("/output/b_ref.mp4", 690), "others": [("/capture/c/track.mkv", CAPTURE_FIRST)],
              "also": {"track": ("/capture/b/track.mkv", CAPTURE_FIRST), "parts": ("/masks/b_face.mkv", 700)}}


def a_load_of_two_subjects():
    import importlib
    import build_workflows
    one = graphs.one_subject_view(PAIR)
    assert (one["subject"], one["others"], one["carried"], one["motion"]) == ("a", ["c"], "held", "mesh"), "the first subject's view"
    assert one["also"] == {"subject": "b", "region": "face", "carried": "held", "parts_mask": "/masks/b_face.mkv", "parts_first": 700, "still": "b.png"}
    assert (one["motion_file"], one["motion_first"], one["motion_edge"]) == ("/output/a_ref.mp4", 700, 768)
    assert (one["motion_file_2"], one["motion_first_2"]) == ("/output/b_ref.mp4", 690) and "motion_edge_2" not in one and one["subjects"] == PAIR["subjects"]
    assert graphs.one_subject_view(WHOLE) is WHOLE, "a load of one subject is changed by the view"
    g = _a_load(one, FILES_PAIR)
    src, run = g[graphs._the(g, graphs.SOURCE)]["inputs"], g[graphs._the(g, graphs.SONG)]["inputs"]
    both_tracks = _read(one, FILES_PAIR["track"]) | _read(one, FILES_PAIR["also"]["track"])
    assert g[src["mask"][0]]["class_type"] == "MaskComposite" and g[src["mask"][0]]["inputs"]["operation"] == "add" and _files_behind(g, src["mask"]) == both_tracks, \
        f"the tracked mask is not the union of the two tracks: {_files_behind(g, src['mask'])}"
    assert _files_behind(g, src["parts"]) == _read(one, FILES_PAIR["parts"]) | _read(one, FILES_PAIR["also"]["parts"]), \
        f"the region is not the union of the two parts: {_files_behind(g, src['parts'])}"
    assert src["replace"] == cfg.MASKED_PARTS_SOURCE["replace"] and src["grow_pixels"] == 64, "the region's kind or its one margin"
    assert _files_behind(g, src["others"]) == _read(one, FILES_PAIR["others"]), f"the keep-outs: {_files_behind(g, src['others'])}"
    assert _files_behind(g, src["motion_video"]) == _read(one, FILES_PAIR["motion"]) and _files_behind(g, src["motion_video_2"]) == _read(one, FILES_PAIR["motion_2"]), \
        "the two references are not on the node's two inputs, each from its own file and clock"
    stills = graphs.stills_of(g)
    assert [g[load]["inputs"]["image"] for _, load in stills] == ["a.png", "b.png"], f"the stills, in order: {[g[load]['inputs']['image'] for _, load in stills]}"
    (first, _), (second, _) = stills
    assert run["references"] == [second, 0] and g[second]["inputs"]["references"] == [first, 0] and "references" not in g[first]["inputs"], "the chain of stills"
    same = {k: v for k, v in g[first]["inputs"].items() if k != "image"}
    assert {k: v for k, v in g[second]["inputs"].items() if k not in ("image", "references")} == same, "the second still is not shown as the first is"
    order = importlib.import_module(f"{REPO.name}.reference_order")
    assert order.assign_labels(order.plan_for(run, g)) == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Video 2>"], \
        f"the labels the model is given: {order.assign_labels(order.plan_for(run, g))}"
    oi = declared()
    errors = build_workflows.validate_api(ours_only(g, oi), oi, "two subjects")
    assert not errors, "\n      ".join(errors)
    assert len(graphs.stills_of(_a_load())) == 1, "a load of one subject has more than its own still"
    # a whole subject beside a part: its track stands for its part; two whole subjects: no parts at all
    mixed = graphs.one_subject_view({**PAIR, "subjects": [PAIR["subjects"][0], {"subject": "b", "region": "whole", "still": "b.png"}]})
    gm = _a_load(mixed, {**FILES_PAIR, "motion_2": None, "also": {"track": FILES_PAIR["also"]["track"], "parts": None}})
    sm = gm[graphs._the(gm, graphs.SOURCE)]["inputs"]
    assert _files_behind(gm, sm["parts"]) == _read(mixed, FILES_PAIR["parts"]) | _read(mixed, FILES_PAIR["also"]["track"]) and "motion_video_2" not in sm
    whole = graphs.one_subject_view({**PAIR, "subjects": [{"subject": "a", "region": "whole", "still": "a.png"}, {"subject": "b", "region": "whole", "still": "b.png"}]})
    gw = _a_load(whole, {"track": FILES_PAIR["track"], "parts": None, "motion": None, "others": [], "also": {"track": FILES_PAIR["also"]["track"], "parts": None}})
    sw_ = gw[graphs._the(gw, graphs.SOURCE)]["inputs"]
    assert "parts" not in sw_ and sw_["replace"] == cfg.MASKED_SOURCE["replace"] and _files_behind(gw, sw_["mask"]) == both_tracks, "two whole subjects"
    first, second = PAIR["subjects"]
    for words, bad in (("list of two", {**PAIR, "subjects": [first, second, second]}), ("list of two", {**PAIR, "subjects": [first]}),
                       ("both of its", {**PAIR, "subjects": [first, {**second, "subject": "a"}]}),
                       ("is listed among `others`", {**PAIR, "subjects": [{**first, "others": ["b"]}, second]}),
                       ("also sets|subject", {**PAIR, "subject": "a"}),
                       ("also sets|motion_file_2", {**PAIR, "motion_file_2": "/x.mp4"}),
                       ("does not take|margin", {**PAIR, "subjects": [{**first, "margin": 16}, second]}),
                       ("`region`", {**PAIR, "subjects": [{**first, "region": "head"}, second]}),
                       ("<Video 1> is the first", {**PAIR, "subjects": [{k: v for k, v in first.items() if not k.startswith("motion")}, second]}),
                       ("first subject's", {**PAIR, "subjects": [first, {**second, "hair_band": 0.5}]})):
        try:
            graphs.one_subject_view(bad)
        except ValueError as e:
            assert all(w in str(e) for w in words.split("|")), f"refused without saying {words!r}: {e}"
        else:
            raise AssertionError(f"a two-subject load that should be refused for {words!r} was read")
    _unbuildable("zoomed in", {**one, "render_region": "part"}, FILES_PAIR)
    _unbuildable("second subject", {**one, "also": {**one["also"], "subject": "z"}}, FILES_PAIR)
    _unbuildable("second subject|no part mask", one, {**FILES_PAIR, "also": {"track": FILES_PAIR["also"]["track"], "parts": None}})
    _unbuildable("still", {**one, "also": {k: v for k, v in one["also"].items() if k != "still"}}, FILES_PAIR)
    # from a capture: the files of both, the record, what is shown; and the capture tool reads the same file
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture = _a_capture(tmp)
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"},
                     look=lambda capture, one, told: {"by": "nobody: this case is about something else"})
        (tmp / "output").mkdir()
        refs = [tmp / "a_ref.mp4", tmp / "b_ref.mp4"]
        for ref in refs:
            ref.write_bytes(b"")
        load = {**PAIR, "subjects": [{**first, "motion_file": str(refs[0])},
                                     {"subject": "b", "region": "face and hair", "still": "b.png", "motion_file": str(refs[1]), "motion_first": 690}]}
        job_file = _loads_file(tmp, capture, loads=(load,))
        written = tool.build_loads(job_file, "plan", [], **where)
        built = json.loads(written[0].with_suffix(".built.json").read_text())
        assert built["files"]["also"] == {"track": [str(capture / "subjects/b/track__p.mkv"), CAPTURE_FIRST],
                                          "parts": [str(capture / "subjects/b/parts__p.mkv"), CAPTURE_FIRST]}, built["files"]["also"]
        assert built["files"]["parts"][0].endswith("subjects/a/parts_held__p.mkv") and built["files"]["others"] == [[str(capture / "subjects/c/track__p.mkv"), CAPTURE_FIRST]]
        who = built["subjects"]
        assert [(w["subject"], w["region"], w["is"], w["picture"], w["video"], w["still"]) for w in who] == \
            [("a", "face and hair", "<Subject 1>", "<Picture 1>", "<Video 1>", "a.png"), ("b", "face and hair", "<Subject 2>", "<Picture 2>", "<Video 2>", "b.png")], who
        assert who[0]["reference"] == str(refs[0]) and who[1]["reference"] == str(refs[1]) and who[1]["part"].endswith("subjects/b/parts__p.mkv")
        said = "\n".join(built["shown"])
        for words in ("TWO SUBJECTS IN ONE PASS", "a (face and hair) is <Subject 1>, its still <Picture 1>, its reference <Video 1>",
                      "b (face and hair) is <Subject 2>, its still <Picture 2>, its reference <Video 2>", "<Subject 2>: NO", "<Video 1>: NO",
                      "TEXT ALONE", "swap or a blend", "one margin for both subjects, 64 px", "the still b.png is 512x640"):
            assert words in said, f"what is shown of a two-subject load lacks {words!r}:\n{said}"
        import capture_masked_run as cap
        (name, plan), = cap.loads_from_file(str(job_file))
        assert name == "pair_0700" and plan["subject"] == "a" and plan["others"] == "c" and plan["carried"] == "held" and plan["at"] == "700", plan
        assert json.loads(plan["subjects"]) == load["subjects"] and json.loads(plan["also"])["subject"] == "b", "the capture tool does not carry the two subjects"
        single = tool.build_loads(_loads_file(tmp, capture, loads=({**{k: v for k, v in PART.items() if k != "continue_from"}, "name": "a_single"},)), "plan", [], **where)
        assert json.loads(single[0].with_suffix(".built.json").read_text())["subjects"] is None, "a load of one subject is recorded as two"
    return "the union of two tracks and of two parts, both subjects' keep-outs, two stills and two references in the list's order, one margin"


def a_hair_band_under_a_part():
    from _lib.frames import read_mask, write_mask_video
    w, h, n = 128, 96, 300                                    # the capture's masks, small; its frames are source frames 650 on
    rule = cfg.MASKED_JOB_HAIR_BAND
    part = np.zeros((n, h, w), bool)
    part[:, 20:50, 40:80] = True                              # a head and hair: 40 wide, 30 tall, its bottom at row 50
    shrunk = 705 - CAPTURE_FIRST                              # on ONE frame of the load the part model loses the lower half
    part[shrunk] = False
    part[shrunk, 20:35, 40:80] = True
    keep = np.zeros((n, h, w), bool)
    keep[:, 50:, :] = True                                    # clothing: everything under the jaw
    keep[:, 5:10, 100:120] = True                             # and a piece nowhere near the band
    keep[:, 25:45, 36:40] = True                              # and a strip down the hair's edge BESIDE the head (2026-10-14: the second subject's 604-948)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture = _a_capture(tmp)
        write_mask_video(capture / "subjects/a/parts_held__p.mkv", part)
        write_mask_video(capture / "subjects/b/track__p.mkv", keep)
        untouched = (capture / "subjects/b/track__p.mkv").read_bytes()
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"},
                     look=lambda capture, one, told: {"by": "nobody: this case is about something else"})
        (tmp / "output").mkdir()
        base = {k: v for k, v in PART.items() if k != "continue_from"}
        banded = {**base, "name": "a_band", "hair_band": 0.5, "hair_band_cuts": "b"}
        plain = {**base, "name": "a_plain"}
        # a dry build: everything a build holds, and nothing written (the chain's `plan --dry` is this)
        would = tool.build_loads(_loads_file(tmp, capture, loads=(banded, plain)), "plan", [], **where, dry=True)
        assert len(would) == 2 and not would[0].parent.exists(), f"a dry build wrote: {sorted(p.name for p in would[0].parent.glob('*'))}"
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_dry", "hair_band_cuts": "c"},)), "plan", [], **where, dry=True), "not among")
        written = tool.build_loads(_loads_file(tmp, capture, loads=(banded, plain)), "plan", [], **where)
        assert written == would, (written, would)
        built, other = (json.loads(p.with_suffix(".built.json").read_text()) for p in written)
        made = Path(built["files"]["others"][0][0])
        assert made.name == "a_band.plan.b_less_hair_band.mkv" and made.parent == written[0].parent and built["files"]["others"][0][1] == banded["first"], built["files"]["others"]
        graph = json.loads(written[0].read_text())
        assert _files_behind(graph, graph[graphs._the(graph, graphs.SOURCE)]["inputs"]["others"]) == {(str(made), graphs.seek(0), banded["frames"])}, \
            "the Masked Source's keep-out is not the build's own file, read from its first frame"
        assert other["files"]["others"] == [[str(capture / "subjects/b/track__p.mkv"), CAPTURE_FIRST]] and other["hair_band"] is None, \
            "a load that asks for no band does not wire the capture's own keep-out"
        assert (capture / "subjects/b/track__p.mkv").read_bytes() == untouched, "the capture's own keep-out was written over"
        got, covered = read_mask(str(made), (w, h), banded["first"], banded["first"], banded["frames"])
        assert covered.all(), "the cut keep-out has fewer frames than the load"
        at = banded["first"] - CAPTURE_FIRST
        side = int(banded["margin"])                                                 # the box and the load's margin each side (2026-10-14)
        assert rule["wider"] == "margin", rule
        x0, x1 = max(40 - side, 0), min(80 + side, w)
        assert rule["starts"] == "top", rule
        y0, y1 = 20, 50 + 15                                                          # from the box's top to half its height under its bottom
        want = keep[at:at + banded["frames"]].copy()
        want[:, y0:y1, x0:x1] = False
        assert x0 < 40 and x1 > 80, "the band is no wider than the part's box: nothing is left for hair that falls outward"
        assert not got[:, y0:y1, x0:x1].any(), "the keep-out is not empty inside the band"
        assert np.array_equal(got, want), f"outside the band the keep-out is not the capture's own: {int((got != want).sum())} pixels differ"
        j = shrunk - at
        assert 0 < j < banded["frames"] - 1 and np.array_equal(got[j], got[j - 1]) and np.array_equal(got[j], got[j + 1]), \
            "the band's edge moved on the frame whose part mask shrank"
        assert part[shrunk].any(1).nonzero()[0][-1] + 1 == 35, "control failed: the part's own box did not move on that frame"
        fig = built["hair_band"]
        assert fig["share_of_the_parts_height"] == 0.5 and fig["cut_out_of"] == "b" and fig["part_height_px_median"] == 30.0, fig
        assert fig["band_px_median"] == {"deep": float(y1 - y0), "wide": float(x1 - x0)} and fig["the_box_is_the_median_over_frames"] == 2 * rule["smooth_frames"] + 1
        assert fig["frames_with_a_band"] == banded["frames"] and fig["frames_written"] == banded["frames"] \
            and fig["keep_out_px_taken_out_a_frame"]["median"] == float(keep[at][y0:y1, x0:x1].sum()) and fig["starts_at_the_part_boxs"] == "top", fig
        assert not got[:, 25:45, 36:40].any(), "the strip beside the head is still in the keep-out"
        assert "not when the band was built" in fig["tested_on_a_render"] and fig["made_from"]["keep_out"].endswith("subjects/b/track__p.mkv")
        assert any(line.startswith("hair band:") and f"{y1 - y0} px deep" in line and "from the part box's top" in line and "not when the band was built" in line
                   for line in built["shown"]), built["shown"]
        assert built["settings"]["hair_band"] == {"value": 0.5, "set_by": "the job file"} and other["settings"]["hair_band"]["value"].startswith("none")
        rendered = tool.build_loads(_loads_file(tmp, capture, loads=(banded, plain)), "render", ["a_band"], **where)
        again = Path(json.loads(rendered[0].with_suffix(".built.json").read_text())["files"]["others"][0][0])
        assert again.name == "a_band.render.b_less_hair_band.mkv" and np.array_equal(read_mask(str(again), (w, h), 700, 700, banded["frames"])[0], got), \
            "the render's cut keep-out is not the plan's"
        # GEMINI'S D13 A1 (the lead's ruling of 2026-10-16): the on-plan rebuild carves the keep-out on the node's grid AFTER the plan cut its band,
        # and the render cuts its band from the carved track. Here the piece nowhere near the band is carved away, as the grid carve would
        carved = keep.copy()
        carved[:, 5:10, 100:120] = False
        write_mask_video(capture / "subjects/b/track__p.mkv", carved)
        later = tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_later"},)), "render", ["a_later"], **where)
        theirs = read_mask(str(Path(json.loads(later[0].with_suffix(".built.json").read_text())["files"]["others"][0][0])), (w, h), 700, 700, banded["frames"])[0]
        assert not np.array_equal(theirs, read_mask(str(made), (w, h), 700, 700, banded["frames"])[0]), "control failed: the carve changed no band file"
        tool.band_again(_loads_file(tmp, capture, loads=(banded, plain)), "a_band", made, output=tmp / "output", repo=tmp)
        # THE CASE THAT GOES RED when the plan's band file is not made again from the carved track: the render's then does not decode equal
        assert np.array_equal(read_mask(str(made), (w, h), 700, 700, banded["frames"])[0], theirs), "the plan's band file made again is not the render's"
        assert not list(made.parent.glob("*.again.mkv")), "the band made again left its working file"
        write_mask_video(capture / "subjects/b/track__p.mkv", keep)
        # from the box's BOTTOM, as built on 2026-10-12: the strip beside the head stays, and only the rows under the part are cut
        old = dict(rule)
        cfg.MASKED_JOB_HAIR_BAND["starts"] = "bottom"
        try:
            under = tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_under"},)), "plan", [], **where)
        finally:
            cfg.MASKED_JOB_HAIR_BAND.clear()
            cfg.MASKED_JOB_HAIR_BAND.update(old)
        cut_under = Path(json.loads(under[0].with_suffix(".built.json").read_text())["files"]["others"][0][0])
        low = read_mask(str(cut_under), (w, h), banded["first"], banded["first"], banded["frames"])[0]
        assert low[:, 25:45, 36:40].all() and not low[:, 50:65, x0:x1].any(), "from the bottom the band reaches beside the head, or does not cut under it"
        # a share of the box's width, as built on 2026-10-12, is still a number the rule takes: a tenth each side
        cfg.MASKED_JOB_HAIR_BAND["wider"] = 0.1
        try:
            tenth = tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_tenth"},)), "plan", [], **where)
        finally:
            cfg.MASKED_JOB_HAIR_BAND.clear()
            cfg.MASKED_JOB_HAIR_BAND.update(old)
        narrow = read_mask(str(Path(json.loads(tenth[0].with_suffix(".built.json").read_text())["files"]["others"][0][0])), (w, h), banded["first"], banded["first"], banded["frames"])[0]
        # THE CASE THAT GOES RED when the band is narrower than the margin: a keep-out column just outside the box's tenth, inside the margin
        assert narrow[:, 50:65, 36 - side // 2].all() and not got[:, 50:65, 36 - side // 2].any(), "the band the margin's width does not reach past a tenth"
        cfg.MASKED_JOB_HAIR_BAND["starts"] = "middle"
        try:
            _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_mid"},)), "plan", [], **where), "starts", "top")
        finally:
            cfg.MASKED_JOB_HAIR_BAND.clear()
            cfg.MASKED_JOB_HAIR_BAND.update(old)
        for words, change in (("hair_band|over 0", {"hair_band": 0}), ("hair_band|over 0", {"hair_band": True}), ("hair_band|up to", {"hair_band": rule["most"] + 1}),
                              ("not among", {"hair_band_cuts": "c"}), ("whole subject", {"carried": "track"})):
            _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_no", **change},)), "plan", [], **where), *words.split("|"))
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**plain, "name": "a_no", "hair_band_cuts": "b"},)), "plan", [], **where), "hair_band_cuts", "no `hair_band`")
        write_mask_video(capture / "subjects/a/parts_held__p.mkv", np.zeros((n, h, w), bool))
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**banded, "name": "a_none"},)), "plan", [], **where), "empty on every frame")
        assert not list(written[0].parent.glob("a_no*")) and not list(written[0].parent.glob("a_none*")) and not list(written[0].parent.glob("a_mid*")), \
            "a refused load left a file"
    return ("the keep-out is cut from the part box's top to under it, a strip beside the head included, and nowhere else; from the bottom the strip stays; "
            "its edge holds on a frame whose part shrinks, and the capture's own file is untouched")


#: A job that names two shots, and a load on both. The made capture's shots are 650-689, 690-899 and 900-949 (`_a_capture`).
TWO_SHOTS = {"cut": 900, "shots": [[690, 899], [900, 949]], "why": "one face across the cut", "said_by": "a check"}
TWO_JOB = {**LOAD_JOB, "two_shots": TWO_SHOTS, "texts": {"w2": {"file": "two.txt", "frames": [850, 939], "by": "a check"}}}
TWO = {"name": "b_two_0850", "first": 850, "frames": 90, "subject": "b", "others": ["a", "c"], "margin": 48, "feather": 4, "edge": "cells",
       "still": "b.png", "seed": 11, "motion": "none", "text": "w2"}


def a_load_of_two_shots():
    """The one way a load may hold a cut: a job that names `two_shots`. Not a default: every hold is on the key."""
    # the key's form: the cut, two neighbouring shots that meet on it, why and who
    assert graphs.two_shots_of(LOAD_JOB) is None and graphs.two_shots_of(TWO_JOB) == TWO_SHOTS
    for bad, words in (("yes", "`two_shots` is"), ({**TWO_SHOTS, "cut": "900"}, "cut '900'"), ({**TWO_SHOTS, "shots": [[690, 899]]}, "shots [[690, 899]]"),
                       ({**TWO_SHOTS, "shots": [[690, 899], [901, 949]]}, "NEIGHBOURING"), ({**TWO_SHOTS, "cut": 899}, "NEIGHBOURING"),
                       ({**TWO_SHOTS, "shots": [[690, 899], [900, 899]]}, "shots [[690, 899], [900, 899]]"), ({**TWO_SHOTS, "shots": [[690, 899.0], [900, 949]]}, "shots"),
                       ({**TWO_SHOTS, "why": " "}, "`why`"), ({k: v for k, v in TWO_SHOTS.items() if k != "said_by"}, "`said_by`")):
        try:
            graphs.two_shots_of({**LOAD_JOB, "two_shots": bad})
            raise AssertionError(f"`two_shots` {bad!r} was taken")
        except ValueError as e:
            assert words in str(e), f"{bad!r} refused without saying {words!r}: {e}"
    # a load of such a job: it builds, as one window on the video grid that holds both shots' frames
    files = {**FILES_WHOLE, "motion": None}
    g = _a_load(TWO, files, job=TWO_JOB)
    run, src = g[graphs._the(g, graphs.SONG)]["inputs"], g[graphs._the(g, graphs.SOURCE)]["inputs"]
    assert (run["window_frames"], run["extent.seconds"]) == (90, 90 / cfg.FPS) and g[src["frames"][0]]["inputs"]["frame_load_cap"] == 90, run
    assert "shot_table" not in src, "a two-shot load wires the node a shot table: the node finds a load's cuts itself, and `plan-cuts` reads what it found"
    # and is held: it holds the cut, inside the two shots, one subject, not zoomed, not continued, one window
    two = lambda **more: {**TWO, **more}   # noqa: E731
    _unbuildable("holds its cut|900", two(first=700), files, job=TWO_JOB)
    _unbuildable("holds its cut", two(first=900, frames=39), files, job=TWO_JOB)
    _unbuildable("reaches outside|690-899", two(first=680, frames=226), files, job=TWO_JOB)
    _unbuildable("zoomed|`two_shots`", two(carried="held", render_region="part"), {**FILES_PART, "track": FILES_WHOLE["track"], "others": FILES_WHOLE["others"]}, job=TWO_JOB)
    _unbuildable("continues another run|`two_shots`", two(continue_from="before.safetensors"), files, job=TWO_JOB)
    pair = {k: v for k, v in PAIR.items()} | {"name": "pair_two", "first": 850, "frames": 90, "text": "w2"}
    _unbuildable("two subjects|`two_shots`", graphs.one_subject_view(pair), FILES_PAIR, job=TWO_JOB)
    long_job = {**TWO_JOB, "two_shots": {**TWO_SHOTS, "shots": [[604, 899], [900, 1050]]}}
    _unbuildable("ONE window|two loads", two(first=650, frames=400), files, job=long_job)
    assert graphs.one_window(WINDOWS[-1] - 1, WINDOWS[-1], WINDOWS) is True, "a lone window at the longest length on both clocks was refused"
    assert graphs.one_window(WINDOWS[-1] + 1, graphs.MAX_LENGTH, WINDOWS) is False, "a window off the audio clock was taken as one window"
    # through a whole build: the text across the named cut is taken, the record and the printed lines say what the load is
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture = _a_capture(tmp)
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"},
                     look=lambda capture, one, told: {"by": "nobody: this case is about something else"})
        (tmp / "output").mkdir()
        (tmp / "two.txt").write_text("[Shot 1] She stands in the hall. [Shot 2] The shot cuts to a closer view of her.\n")
        path = _loads_file(tmp, capture, loads=(TWO,), two_shots=TWO_SHOTS, texts=TWO_JOB["texts"])
        built = json.loads(tool.build_loads(path, "plan", [], **where)[0].with_suffix(".built.json").read_text())
        assert built["two_shots"] == TWO_SHOTS and built["what_this_clips_preview_measured"]["cuts_inside_this_load"] == [900], built["two_shots"]
        said = [line for line in built["shown"] if "TWO SHOTS IN ONE WINDOW" in line]
        assert len(said) == 1 and all(w in said[0] for w in ("a check's word", "one face across the cut", "690-899 and 900-949", "the cut on 900", "frame 50 of the load", "Not a default")), said
        assert any("[Shot 1]: yes, [Shot 2]: yes" in line for line in built["shown"]), built["shown"]
        # the same job without the key: a load that holds a cut, refused as it always was, and nothing written
        other = tmp / "other"
        other.mkdir()
        (other / "two.txt").write_text((tmp / "two.txt").read_text())
        (other / "loads.json").write_text(json.dumps({"job": {**LOAD_JOB, "name": "no_key", "capture": str(capture), "texts": TWO_JOB["texts"]}, "defaults": {}, "loads": [TWO]}))
        _refused(lambda: tool.build_loads(other / "loads.json", "plan", [], **where), "cut(s) on 900", "a text is of one shot", "`two_shots`")
        assert not (tmp / "data" / "2026-01-01_no_key_job").exists(), "a refused build left a folder behind"
        # an ordinary load is shown no such line
        assert not tool.two_shot_lines(LOAD_JOB, WHOLE, "the words")
    return "the key's form; one window, one subject, the cut inside; built with the record saying so; refused without the key"


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
    # the audio a load is given writes exactly the load's frames by the planner's own count, at any length up
    # to the longest window, for two sample rates and for a cut that rounds a sample either way
    over = [(frames, rate, nudge) for frames in range(1, WINDOWS[-1] + 1) for rate in (44100, 48000) for nudge in (-1, 0, 1)
            if planner.frames_kept(10 ** 6, (round(round((frames - graphs.AUDIO_SHORT_BY) / cfg.FPS, 6) * rate) + nudge) / rate) != frames]
    assert not over, f"audio cut for a load writes another number of frames than the load has, e.g. {over[:3]}"
    # the other side: audio a quarter of a frame past a load does write a frame more, at every length, so
    # the cut is doing something. (Until e79cc219 one sample past was enough; the planner now forgives a
    # track within its own slack of a whole frame count, and this control moved outside that slack.)
    past = [frames for frames in range(1, WINDOWS[-1] + 1) if planner.frames_kept(10 ** 6, (frames + 0.25) / cfg.FPS) != frames + 1]
    assert not past, f"audio a quarter of a frame past a load no longer writes a frame more, e.g. at {past[:3]}: the cut may not be needed"

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
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"},
                     look=lambda capture, one, told: {"by": "nobody: this case is about something else"})
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
        # a load that names a second motion video: found by its path, on the clock the load gives, shown and recorded
        face = tmp / "b_face.mp4"
        face.write_bytes(b"")
        second = {**WHOLE, "name": "b_two", "motion_file_2": str(face), "motion_first_2": 690, "motion_edge_2": 384}
        two = tool.build_loads(_loads_file(tmp, capture, loads=(second,)), "plan", [], **where)
        built = json.loads(two[0].with_suffix(".built.json").read_text())
        assert built["files"]["motion_2"] == [str(face), 690], f"the second motion video's file: {built['files'].get('motion_2')}"
        assert any("second motion video" in line and "384" in line and "DOES NOT NAME <Video 2>" in line for line in built["shown"]), \
            "what is shown of a load does not say its second motion video, or that its text leaves it unnamed"
        # a zoomed part load that names its motion video (2026-10-13; refused until then): found by its path, and what
        # is shown says the video is not cut to the box. The same load with no motion video says the zoom and no more.
        head = tmp / "a_head_reference.mp4"
        head.write_bytes(b"")
        close = {**{k: v for k, v in PART.items() if k != "continue_from"}, "name": "a_close", "render_region": "part"}
        named = {**close, "name": "a_close_ref", "motion": "mesh", "motion_file": str(head), "motion_first": 700}
        plain, with_ref = (json.loads(p.with_suffix(".built.json").read_text())
                           for p in tool.build_loads(_loads_file(tmp, capture, loads=(close, named)), "plan", [], **where))
        assert with_ref["files"]["motion"] == [str(head), 700], f"a zoomed load's motion video is {with_ref['files']['motion']}, not the file it names"
        assert with_ref["settings"]["render_region"]["value"] == graphs.RENDER_REGIONS["part"] \
            and with_ref["settings"]["motion"]["value"] == graphs.MOTIONS["mesh"], with_ref["settings"]
        assert any(line.startswith("rendered `zoomed in") and "NOT cut to that box" in line for line in with_ref["shown"]), \
            f"what is shown of a zoomed load with a motion video does not say the video is not cut: {with_ref['shown']}"
        assert any(line.startswith("rendered `zoomed in") for line in plain["shown"]) and not any("NOT cut" in line for line in plain["shown"]), \
            f"what is shown of a zoomed load with no motion video: {plain['shown']}"
        assert built["settings"]["motion_edge_2"] == {"value": 384, "set_by": "the job file"}, built["settings"].get("motion_edge_2")
        assert json.loads(written[0].with_suffix(".built.json").read_text())["settings"]["motion_edge_2"]["value"] == "no second motion video"
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**second, "name": "b_gone", "motion_file_2": str(tmp / "nothing.mp4")},)),
                                          "plan", [], **where), "motion_file_2", "is not a file")
        assert any("padded with" in line and "as untouched context, tossed after the decode" in line for line in json.loads(written[0].with_suffix(".built.json").read_text())["shown"]), \
            "what a short load is shown does not say its context frames tossed after decoding"
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
        assert seen["cuts_on_source_frames"] == [CAPTURE_FIRST + 40, CAPTURE_FIRST + 250], seen["cuts_on_source_frames"]
        assert seen["cuts_inside_this_load"] == [], f"the cuts inside {WHOLE['first']}..{last}: {seen['cuts_inside_this_load']}"
        assert seen["subject"]["frames_read"] == WHOLE["frames"] and seen["subject"]["frames_with_no_mask"] == 1, seen["subject"]
        assert seen["subject"]["share_of_the_frame"] == {"least": WHOLE["first"] / 1000, "middle": (WHOLE["first"] + WHOLE["frames"] // 2) / 1000,
                                                         "most": last / 1000}, seen["subject"]["share_of_the_frame"]
        assert seen["subject"]["box_height_px_middle"] == 160 and "not in the capture" in seen["part"], seen
        pad_before, pad_after = graphs.window_pad(WHOLE["first"], WHOLE["frames"], SHORT_WINDOW, LOAD_JOB.get("source_frames"),
                                                  bool(WHOLE.get("continue_from")), floor=WHOLE.get("audio_first", LOAD_JOB["first"]))
        assert seen["load_against_its_window"] == {"frames": WHOLE["frames"], "window": SHORT_WINDOW, "context_before": pad_before, "context_after": pad_after, "tossed_after_decode": SHORT_WINDOW - WHOLE["frames"], "held_past_the_load": 0}, seen["load_against_its_window"]
        absent = part["what_this_clips_preview_measured"]
        assert "not in the capture" in absent["subject"] and "not in the capture" in absent["cuts_on_source_frames"], \
            f"a figure the capture does not hold is not said to be absent: {absent}"
        # a load shorter than the song node takes is refused at build, by the node's own minimum
        import build_workflows
        least_seconds, least_frames = tool.shortest_load()
        too_short = {**PART, "name": "a_short", "frames": least_frames - 1}
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(too_short,)), "plan", [], **where), "a_short", f"{least_frames} frames")

        def node_refuses(frames: int) -> bool:
            graph = ours_only(_a_load({**PART, "frames": frames}, FILES_PART, plan=True), declared())
            graph[graphs._the(graph, graphs.SONG)]["inputs"]["extent.seconds"] = frames / cfg.FPS
            return any("extent.seconds" in e and "below min" in e for e in build_workflows.validate_api(graph, declared(), "short"))
        assert node_refuses(least_frames - 1) and not node_refuses(least_frames), \
            f"the node's own schema does not draw the line at {least_frames} frames ({least_seconds} s), where the tool reads it"
        # a zoomed load is one shot. The builder had a refusal of its own for a zoomed load that holds a cut; it was
        # taken out on 2026-10-10 as unreachable, because NO load that holds a cut is built now: its text would
        # span the cut. This case shows it: a zoomed load across the cut on 690, with a text of exactly its own
        # frames, is refused by the text rule, naming the cut. The day the text rule is loosened this goes red and
        # the hole is seen (the node's own refusal of a cut in a zoomed load is behind it either way).
        across = {**WHOLE, "name": "b_zoomed", "first": 660, "carried": "held", "motion": "none", "render_region": "part"}
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=(across,), texts={"w": {"file": "words.txt", "frames": [660, 759], "by": "a check"}}),
                                          "plan", [], **where), "b_zoomed", "cut(s) on", str(CAPTURE_FIRST + 40), "one shot")
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
        _refused(lambda: tool.build_loads(_loads_file(tmp, capture, loads=({**one, "first": 610},), texts={"w": {"file": "words.txt", "frames": [610, 899], "by": "a check"}}),
                                          "plan", [], **later), "holds 650")
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


def a_text_is_of_its_frames():
    """The made capture holds source frames 650-949 and cuts on 690 and 900, so its shots are 650-689, 690-899
    and 900-949."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        capture = _a_capture(tmp)
        (tmp / "t.txt").write_text("  the words\nof one shot \n")
        (tmp / "copy.txt").write_text("the words\nof one shot\n")
        (tmp / "other.txt").write_text("other words\n")
        entry = {"file": "t.txt", "frames": [690, 899], "by": "a check"}
        job = {**LOAD_JOB, "texts": {"w": entry}}
        told = tool.text_of(job, WHOLE, tmp)
        assert told == {"label": "w", "file": str(tmp / "t.txt"), "frames": [690, 899], "by": "a check", "text": "the words\nof one shot",
                        "sha256": hashlib.sha256(b"the words\nof one shot").hexdigest()}, told
        # a text says its frames and its writer, and is a file beside the job file; the refusal says the rule
        for bad, words in (("t.txt", ("bare file name",)), ({"file": "t.txt", "by": "a check"}, ("frames",)),
                           ({**entry, "frames": [899, 690]}, ("frames",)), ({**entry, "frames": [690.0, 899]}, ("frames",)),
                           ({**entry, "frames": [True, 899]}, ("frames",)), ({**entry, "frames": [690]}, ("frames",)),
                           ({**entry, "by": " "}, ("who wrote it",)), ({"file": "t.txt", "frames": [690, 899]}, ("who wrote it",)),
                           ({**entry, "file": str(tmp / "t.txt")}, ("beside the job file",)), ({**entry, "file": "sub/t.txt"}, ("beside the job file",)),
                           ({**entry, "file": "none.txt"}, ("missing or empty",))):
            _refused(lambda: tool.text_of({**LOAD_JOB, "texts": {"w": bad}}, WHOLE, tmp), "'b_0700'", "`w`", "true of one window only", *words)
        _refused(lambda: tool.text_of(job, {**WHOLE, "text": "x"}, tmp), "no entry `x`")
        _refused(lambda: tool.text_of(job, {k: v for k, v in WHOLE.items() if k != "text"}, tmp), "no text")
        # a text covers a load when the load's frames lie inside the text's: to the frame at both ends
        last = WHOLE["first"] + WHOLE["frames"] - 1
        for frames, covered in (([WHOLE["first"], last], True), ([WHOLE["first"] + 1, last], False), ([WHOLE["first"], last - 1], False)):
            fn = lambda: tool.text_of({**LOAD_JOB, "texts": {"w": {**entry, "frames": frames}}}, WHOLE, tmp)  # noqa: E731
            if covered:
                assert fn()["frames"] == frames
            else:
                _refused(fn, "'b_0700'", f"{WHOLE['first']}-{last}", f"{frames[0]}-{frames[1]}", "does not cover")

        def texts(frames, loads=(WHOLE,), **more):
            return tool.texts_for({**LOAD_JOB, "texts": {"w": {**entry, "frames": frames}, **more}}, list(loads), tmp, capture)
        # a text is of one shot: exactly a shot passes, a frame more at either end holds a cut
        assert texts([690, 899])["b_0700"]["sha256"] == told["sha256"]
        _refused(lambda: texts([689, 899]), "'b_0700'", "cut(s) on 690", "one shot", "one load a shot", "not a law of the model")
        _refused(lambda: texts([690, 900]), "cut(s) on 900")
        tiny = {**WHOLE, "name": "b_tiny", "first": 690, "frames": 1}
        assert texts([690, 690], loads=(tiny,)) and texts([650, 689], loads=({**tiny, "first": 689},)), "a text of one frame, a load on a cut frame"
        _refused(lambda: texts([689, 690], loads=(tiny,)), "cut(s) on 690")
        # outside the frames the capture saw nobody knows where the cuts are
        _refused(lambda: texts([640, 689], loads=({**tiny, "first": 650},)), "holds 650-949")
        _refused(lambda: texts([900, 950], loads=({**tiny, "first": 900},)), "holds 650-949")
        # one text for loads of different shots, tonight's fault: every load outside the text's shot is named, the
        # load inside it is not, and that one builds alone
        p, q, r = ({**WHOLE, "name": name, "first": first, "frames": 30} for name, first in (("b_p", 650), ("b_q", 700), ("b_r", 900)))
        try:
            texts([690, 899], loads=(p, q, r))
            raise AssertionError("one text for three shots was not refused")
        except tool.Refused as e:
            assert "2 thing(s)" in str(e) and "'b_p'" in str(e) and "'b_r'" in str(e) and "'b_q'" not in str(e), str(e)
            assert str(e).count(tool.TEXT_RULE) == 1, "the rule is not said once"
        assert list(texts([690, 899], loads=(q,))) == ["b_q"]
        _refused(lambda: tool.texts_for({**LOAD_JOB, "texts": {"w": "t.txt"}}, [p, q], tmp, capture), "bare file name", "'b_p', 'b_q'", "true of one window only")
        # the same words under two labels, whatever the files are called; other words are another text; a
        # label no chosen load takes and that is not a text yet stops nothing
        twin = {"file": "copy.txt", "frames": [650, 689], "by": "a check"}
        _refused(lambda: texts([690, 899], loads=(q,), v=twin), "same words", "`v` (copy.txt, frames 650-689)", "`w` (t.txt, frames 690-899)")
        assert texts([690, 899], loads=(q,), v={**twin, "file": "other.txt"}) and texts([690, 899], loads=(q,), v="other.txt")
        # every subject's table counts: a cut only another subject's table holds is a cut
        (capture / "subjects" / "c" / "shots__p.json").write_text(json.dumps({"shots": [{"first_frame": 0}, {"first_frame": 100}]}))
        assert tool.cuts_of(capture) == ([690, 750, 900], CAPTURE_FIRST, CAPTURE_FIRST + 299), tool.cuts_of(capture)
        _refused(lambda: texts([690, 899]), "cut(s) on 750")
        (capture / "subjects" / "c" / "shots__p.json").unlink()
        (capture / "subjects" / "b" / "shots__p.json").rename(tmp / "kept.json")
        _refused(lambda: texts([690, 899]), "no shot table")
        (tmp / "kept.json").rename(capture / "subjects" / "b" / "shots__p.json")
        _refused(lambda: tool.texts_for(job, [WHOLE], tmp, tmp / "nowhere"), "not a capture")

        # ---- the look: held by what was shown, and by the capture's own clock
        row = {"load": "b_0700", "frames": [WHOLE["first"], last], "text_frames": [690, 899], "text_sha256": told["sha256"], "strip": "b_0700.mp4"}

        def looks(*lines):
            (capture / "looks.json").write_text(json.dumps({"looks": list(lines)}))
        look = lambda: tool.looked_at(capture, WHOLE, told)  # noqa: E731
        _refused(look, "nobody has looked", "'b_0700'", "video", "looked")
        looks({"by": "p", "at": "2026-01-01T10:00:01", "loads": "all", "note": "a line of the form before the text rule"})
        _refused(look, "nobody has looked")
        looks({"by": "p", "at": "2026-01-01T10:00:01", "shown": [{**row, "load": "a_0700"}]})
        _refused(look, "nobody has looked")
        for at in ("2026-01-01T09:59:59", "2026-01-01T10:00:00"):                 # before the capture, and the same second
            looks({"by": "p", "at": at, "shown": [row]})
            _refused(look, "not later than the capture", "made again")
        looks({"by": "p", "at": "2026-01-01T10:00:01", "note": "every frame", "shown": [row]})
        assert look() == {"by": "p", "at": "2026-01-01T10:00:01", "note": "every frame", "strip": "b_0700.mp4"}, look()
        for key, other, words in (("frames", [700, 798], "as source frames [700, 798]"), ("text_frames", [690, 898], "written from [690, 898]"),
                                  ("text_sha256", "0" * 64, "other words")):
            looks({"by": "p", "at": "2026-01-01T10:00:01", "shown": [{**row, key: other}]})
            _refused(look, words, "something else than this build")
        # an earlier look that does match is not hidden by a later one that does not, and the reverse
        looks({"by": "p", "at": "2026-01-01T10:00:01", "shown": [row]}, {"by": "q", "at": "2026-01-01T10:00:09", "shown": [{**row, "text_sha256": "0" * 64}]})
        assert look()["by"] == "p"
        # one clock: a time with a zone, or no time, is refused and never compared
        for at, words in (("2026-01-01T15:00:01+00:00", "zone"), ("yesterday", "not a time"), (None, "not a time")):
            looks({"by": "p", "at": at, "shown": [row]})
            _refused(look, words)
        # the capture written again a minute after the look
        looks({"by": "p", "at": "2026-01-01T10:00:01", "shown": [row]})
        (capture / "status.json").write_text(json.dumps({"state": "done", "written": "2026-01-01T10:01:01"}))
        _refused(look, "not later than the capture")
        (capture / "status.json").write_text(json.dumps({"state": "done"}))
        _refused(look, "status.json", "not a time")
        (capture / "status.json").write_text(json.dumps({"state": "done", "written": "2026-01-01T10:00:00"}))

        # ---- `build plan` and `queue render` on it
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output", inputs=tmp / "input",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"})
        (tmp / "output").mkdir()
        one, two = {**WHOLE, "motion": "none"}, {**WHOLE, "name": "b_0800", "first": 800, "frames": 60, "motion": "none"}
        made = lambda stage: tool.build_loads(_loads_file(tmp, capture, loads=(one, two)), stage, [], **where)  # noqa: E731
        # the look is shown and never refused (the owner, 2026-10-12, row c22): a plan on a capture nobody looked
        # at is built; its record says so and why, with the strip's path when one is drawn and the command when not
        (capture / "looks.json").unlink()
        (capture / "looks").mkdir(exist_ok=True)
        (capture / "looks" / "b_0700.mp4").write_bytes(b"a strip")
        unseen = [json.loads(p.with_suffix(".built.json").read_text()) for p in made("plan")]
        assert [r["load"]["name"] for r in unseen] == ["b_0700", "b_0800"], [r["load"]["name"] for r in unseen]
        for r in unseen:
            assert r["look"]["recorded"] is False and "nobody has looked" in r["look"]["why"], r["look"]
            line = next(i for i, l in enumerate(r["shown"]) if l.startswith("NO LOOK IS RECORDED FOR THIS BUILD"))
            assert "nobody has looked" in r["shown"][line] and "to record one:" in r["shown"][line + 1], r["shown"][line:]
        drawn, undrawn = unseen[0]["look"], unseen[1]["look"]
        # the strip's time is UTC and says so, never a local time with no zone
        import re
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", drawn["strip_drawn"]), drawn["strip_drawn"]
        took = datetime.datetime.fromtimestamp((capture / "looks" / "b_0700.mp4").stat().st_mtime, datetime.timezone.utc)
        assert drawn["strip_drawn"] == took.strftime("%Y-%m-%dT%H:%M:%SZ"), (drawn["strip_drawn"], took)
        assert drawn["strip"] == str(capture / "looks" / "b_0700.mp4") and drawn["strip_drawn"] and "looked" in drawn["how"], drawn
        assert undrawn["strip"] is None and undrawn["strip_drawn"] is None and "video" in undrawn["how"], undrawn
        after = [r["shown"][next(i for i, l in enumerate(r["shown"]) if l.startswith("NO LOOK")) + 1] for r in unseen]
        assert "the strip is " in after[0] and "no strip is drawn" in after[1], after
        assert unseen[0]["shown"][-1] == "  | the words", "the look's lines go before the text, which a build prints last"
        shutil.rmtree(tmp / "data" / "2026-01-01_check_job")
        (capture / "looks" / "b_0700.mp4").unlink()
        words = hashlib.sha256(b"the words").hexdigest()
        rows = [{"load": "b_0700", "frames": [700, 799], "text_frames": [690, 899], "text_sha256": words},
                {"load": "b_0800", "frames": [800, 859], "text_frames": [690, 899], "text_sha256": words}]
        looks({"by": "p", "at": "2026-01-01T10:00:01", "note": "both", "shown": rows})
        plans = made("plan")
        record = json.loads(plans[0].with_suffix(".built.json").read_text())
        assert record["look"]["by"] == "p" and record["text"]["frames"] == [690, 899] and record["text"]["by"] == "a check" \
            and record["text"]["text"] == "the words" and record["text"]["sha256"] == words, record
        said = record["shown"]
        at = next(i for i, line in enumerate(said) if line.startswith("text `w` (words.txt), written from source frames 690-899 by a check; this load is 700-799"))
        assert said[at + 1:] == ["  | the words"], f"the build does not print the text in full: {said[at:]}"
        assert not any("characters" in line for line in said), "the build still prints a text's length"
        # a render is built without a look; at queue the look is shown, never refused, on the text it was built with
        (capture / "looks.json").unlink()
        renders = made("render")
        built, graph = json.loads(renders[0].with_suffix(".built.json").read_text()), json.loads(renders[0].read_text())
        assert built["look"] is None
        unseen = tool.text_and_look(built, graph)
        assert unseen["recorded"] is False and "nobody has looked" in unseen["why"] and unseen["strip"] is None, unseen
        assert tool.look_lines(unseen)[0].startswith("NO LOOK IS RECORDED FOR THIS BUILD"), tool.look_lines(unseen)
        assert tool.look_lines({"by": "p", "at": "t"}) == ["looked at by p at t"] and tool.look_lines(None) == []
        looks({"by": "p", "at": "2026-01-01T10:00:01", "note": "both", "shown": rows})
        assert tool.text_and_look(built, graph)["by"] == "p"
        (tmp / "words.txt").write_text("the words, and a sentence more\n")
        _refused(lambda: tool.text_and_look(built, graph), "changed after it was built", "build the load again")
        (tmp / "words.txt").write_text("the words\n")
        graph[graphs._the(graph, graphs.SONG)]["inputs"]["prompt"] = "edited in the graph"
        _refused(lambda: tool.text_and_look(built, graph), "changed after it was built")
        _refused(lambda: tool.text_and_look({k: v for k, v in built.items() if k != "text"} | {"text_file": "words.txt"}, graph),
                 "built before", "true of one window only")
    return "the form, the cover, one shot, the same words twice, the look by what was shown, and both where a build and a queue meet them"


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
        # D12-1 & D12-5: masked_job.gate refuses when frame_table.jsonl is missing or its input_sha256 is stale (and checks frame_table__<also>.jsonl for a pair)
        _refused(lambda: tool.gate(built, folder, ok), "frame_table.jsonl", "table")
        def _sha(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        def _hdr(subj: str, manifest_sha: str | None = None) -> str:
            return json.dumps({
                "frame_table_schema": 1,
                "subject": subj,
                "input_sha256": {
                    "manifest": {"file": str(capture / "manifest.json"), "sha256": manifest_sha or _sha(capture / "manifest.json")},
                    "flags": {"file": str(capture / "flags.json"), "sha256": _sha(capture / "flags.json")},
                    "part_grades": None,
                    "job": None,
                    "shots": None,
                },
            }) + "\n"
        (capture / "frame_table.jsonl").write_text(_hdr("b", manifest_sha="0" * 64))
        _refused(lambda: tool.gate(built, folder, ok), "frame_table.jsonl", "manifest", "table")
        (capture / "frame_table.jsonl").write_text(_hdr("b"))
        found = tool.gate(built, folder, ok)
        assert found == {"verify": 0, "verdict": "clear", "overridden": ["f003"], "plan_read": "b_0700_plan.json"}, found
        built_pair = {"capture": str(capture), "load": {"name": "b_0700", "also": {"subject": "a"}}}
        _refused(lambda: tool.gate(built_pair, folder, ok), "frame_table__a.jsonl", "table")
        (capture / "frame_table__a.jsonl").write_text(_hdr("a"))
        assert tool.gate(built_pair, folder, ok)["plan_read"] == "b_0700_plan.json"
        _refused(lambda: tool.stem_of("render"), "--load")
        line = tool.plan_line({"load": {"name": "b_0700", "first": 700, "subject": "b", "others": ["a", "c"]}, "windows_folder": "/out/b_0700_windows"})
        assert line == "--run b_0700:preview=/out/b_0700_windows,subject=b,others=a+c,at=700", line
        assert tool.plan_line({"load": {"name": "x", "first": 5, "subject": "a"}, "windows_folder": "/w"}).endswith("subject=a,at=5")
        # what a finished run left in the server's history, as the record keeps it
        read = tool.read_history({"status": {"status_str": "success", "messages": [
            ["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 63000}]]},
            "outputs": {"7": {"gifs": [{"filename": "b_mesh_00001.mp4", "subfolder": "Video/check/check", "type": "output"}],
                              "text": ["said by seven"]},
                        "3": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}})
        assert read == {"status": "success", "seconds": 62, "files": ["Video/check/check/b_mesh_00001.mp4", "a.png"],
                        "said": {"7": ["said by seven"]}, "cached_nodes": None}, read
        # THE CASE THAT GOES RED when the record drops the server's list of nodes served from its cache (2026-10-16)
        hit = tool.read_history({"status": {"status_str": "success", "messages": [
            ["execution_start", {"timestamp": 1000}], ["execution_cached", {"nodes": ["12", "3", "40"], "timestamp": 1001}]]}, "outputs": {}})
        assert hit["cached_nodes"] == ["3", "12", "40"], hit
        none = tool.read_history({"status": {"status_str": "success", "messages": [["execution_cached", {"nodes": []}]]}, "outputs": {}})
        assert none["cached_nodes"] == [], none
        failed = tool.read_history({"status": {"status_str": "error", "messages": [
            ["execution_error", {"node_type": "X", "exception_type": "E", "exception_message": "why"}]]}, "outputs": {}})
        assert failed["status"] == "error" and failed["error"]["exception_message"] == "why" and failed["files"] == [], failed
        assert (tool.stem_of("preview"), tool.stem_of("render", "b_0700")) == ("preview", "b_0700.render")
        # what a load's node wrote itself, which no history lists: a plan that saved its plan file was reported as "0 files"
        with tempfile.TemporaryDirectory() as made:
            windows = Path(made) / "b_0700_windows"
            built = {"load": {"name": "b_0700"}, "windows_folder": str(windows)}
            assert tool.windows_files({"output_folder": "/a/preview"}) is None and tool.windows_files(built) == []
            assert tool.plan_file_missing("plan", built, []) == [str(windows / "b_0700_plan.json")] and tool.plan_file_missing("preview", {}, None) == []
            windows.mkdir()
            for name in ("b_0700_window_1_planned_region.npz", "b_0700_plan.json"):
                (windows / name).write_bytes(b"")
            wrote = tool.windows_files(built)
            assert wrote == ["b_0700_plan.json", "b_0700_window_1_planned_region.npz"], wrote
            assert tool.plan_file_missing("plan", built, wrote) == [] and tool.plan_file_missing("render", built, []) == []


def _job_file(tmp: Path, job: dict) -> Path:
    path = tmp / "job.json"
    path.write_text(json.dumps({"job": job, "defaults": {}, "loads": []}))
    return path


def an_uncarved_keep_out_refused():
    """The lead's ruling of 2026-10-16: a keep-out that is not one of our people is handed to the node only from a capture that
    carved it out of the subject's own track (its sighting's `carved_out_of_own_track`); another person is handed on as before."""
    with tempfile.TemporaryDirectory() as t:
        capture = _a_capture(Path(t))
        (capture / "subjects" / "k").mkdir()
        (capture / "subjects" / "k" / "track__p.mkv").write_bytes(b"")
        m = json.loads((capture / "manifest.json").read_text())
        m["subjects"].append({"label": "k", "sightings": [{"by": "p", "shots": False}]})
        (capture / "manifest.json").write_text(json.dumps(m))
        one = {"name": "a_0700", "subject": "a", "first": CAPTURE_FIRST + 10, "frames": 50, "others": ["c", "k"], "motion": "none"}
        # THE CASES THAT GO RED when a keep-out made before the carve is handed to the node, or the words lead to a re-capture in place
        _refused(lambda: tool.capture_files({}, one, capture, lambda label: None), "the keep-out 'k'", "made before the carve; re-capture")
        _refused(lambda: tool.capture_files({}, one, capture, lambda label: None, "render"), "the keep-out 'k'",
                 "planned before the keep-out carve: make the load again under a new name")
        m["subjects"][-1]["sightings"][0]["carved_out_of_own_track"] = {"px": 0, "cells": 0, "on_the_node_grid_from": "no plan yet: the canvas carve alone"}
        (capture / "manifest.json").write_text(json.dumps(m))
        got = tool.capture_files({}, one, capture, lambda label: None)
        assert [Path(x).parent.name for x, _ in got["others"]] == ["c", "k"], got["others"]
        # a render on a capture whose keep-out was not carved on the node's grid (no plan when it was made) is refused (the read's finding 2)
        _refused(lambda: tool.capture_files({}, one, capture, lambda label: None, "render"), "not carved on the node's own grid", "make the load again")
        m["subjects"][-1]["sightings"][0]["carved_out_of_own_track"]["on_the_node_grid_from"] = ["/plans/p_plan.json"]
        (capture / "manifest.json").write_text(json.dumps(m))
        assert [Path(x).parent.name for x, _ in tool.capture_files({}, one, capture, lambda label: None, "render")["others"]] == ["c", "k"]
    return ("a keep-out from a capture made before the carve refused: re-capture before a plan, make the load again after; a render on a keep-out "
            "not carved on the node's grid refused; carved on it, and another person, handed on")


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
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"})
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
        assert set(built["expects"]["c"]) == set(tool.SAVED_AS) - {"mesh", "pose"} \
            and set(built["expects"]["a"]) == set(tool.SAVED_AS), \
            "what the preview must leave on disk does not follow who has a pose pass"
        assert "head_track" not in tool.expected_files({**job, "subjects": [{**job["subjects"][0], "track_head": False}]})["a"]
        _refused(lambda: tool.build(_job_file(tmp, {**job, "sam_head_track": {"on": False}}), "preview", **where),
                 "sam_head_track", "why")
        # every save of the graph is one of the things looked for afterwards, and the other way round
        saved = {v for n in graphs.preview(job).values() for k, v in n["inputs"].items() if k in ("filename_prefix", "table_prefix")}
        looked_for = {p[:-len(tool.SAVED_AS[k])] for things in built["expects"].values() for k, p in things.items()}
        assert looked_for == saved, f"saved and not looked for: {sorted(saved - looked_for)}; looked for and not saved: {sorted(looked_for - saved)}"
        out = tmp / "output"
        for name in ("check_a_track_00001.mkv", "check_a_track_00001.png", "check_a_head_track_00001.mkv", "check_a_shots_00001_shots.json", "check_a_pose_table.json"):
            (out / "Video/check/check").mkdir(parents=True, exist_ok=True)
            (out / "Video/check/check" / name).write_bytes(b"")
        found, missing = tool.found_on_disk({"a": built["expects"]["a"]}, out)
        assert found["a"]["track"] == ["Video/check/check/check_a_track_00001.mkv"] and found["a"]["shots"] and found["a"]["pose"], found
        assert sorted(m.split(" ")[1] for m in missing) == ["classes", "held_things", "mesh", "parts"], f"said to be missing: {missing}"
        # the capture command for a run that saved everything names the files themselves; a run that saved two
        # files of one kind, or none, gets no command
        for label in "abc":
            for key, pattern in built["expects"][label].items():
                (out / pattern.replace("*", "00001")).write_bytes(b"")
        whole, nothing_missing = tool.found_on_disk(built["expects"], out)
        assert not nothing_missing, nothing_missing
        command = tool.capture_from(job, whole, out, "2026-01-01")
        assert "<" not in command and command.count("--mask") == 3 and f"track={out}/Video/check/check/check_b_track_00001.mkv" in command, command
        assert "pose=" not in command.split("--mask c:")[1] and f",at={job['first']}" in command and "held" not in command, command
        _refused(lambda: tool.capture_from(job, {**whole, "b": {**whole["b"], "track": []}}, out, "2026-01-01"), "b", "track")
        # a subject no tracker ever had: no pick, or a pick and no frame of them on any shot
        for label, table in (("a", {"pick_frame": 60, "shots": [{"frames_with_subject": 0}, {"frames_with_subject": 12}]}),
                             ("b", {"pick_frame": None, "shots": [{"frames_with_subject": 0}] * 3}),
                             ("c", {"pick_frame": 4, "shots": [{"frames_with_subject": 0}] * 2})):
            (out / whole[label]["shots"][0]).write_text(json.dumps(table))
        lost = tool.subjects_not_found(whole, out)
        assert [line.split(":")[0] for line in lost] == ["b", "c"] and "no pick" in lost[0] and "frame 4" in lost[1], lost
        _refused(lambda: tool.capture_from(job, {**whole, "a": {**whole["a"], "parts": ["x", "y"]}}, out, "2026-01-01"), "2 files")
        import shutil as _sh
        _sh.rmtree(out / "Video")
        assert built["builder_not_committed"] == tool.builder_not_committed(), "the record does not say what state the builder was in"
        assert json.loads((folder / "job.json").read_text())["job"] == job, "the job file was not kept as given"
        # the same name again: the job folder is there
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **where), "job folder", "new `name`")
        # another day, the same name: the output folder would be reused
        Path(built["output_folder"]).mkdir(parents=True)
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **{**where, "date": "2026-01-02"}), "output folder")
        # a subject with no pick; a job with no block; a non-bool track_head
        for wrong_th in ("yes", 1):
            job_th = dict(_without_corrections(), name="check_th")
            job_th["subjects"][0]["track_head"] = wrong_th
            _refused(lambda: tool.build(_job_file(tmp, job_th), "preview", **where), "'a'", "track_head")
        for wrong in (0, 1.5, "0.9", True):
            job3 = dict(_without_corrections(), name="check3")
            job3["subjects"][0]["match"] = wrong
            _refused(lambda: tool.build(_job_file(tmp, job3), "preview", **where), "'a'", "match")
        job2 = dict(_without_corrections(), name="check2")
        del job2["subjects"][2]["pick"]
        _refused(lambda: tool.build(_job_file(tmp, job2), "preview", **where), "'c'", "pick")
        (tmp / "bare.json").write_text(json.dumps({"defaults": {}, "loads": []}))
        _refused(lambda: tool.build(tmp / "bare.json", "preview", **where), "`job` block")


def a_job_is_built_on_its_videos_analysis_folder():
    import clip_folder as cf
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        video, copy = tmp / "input" / "fun.mkv", tmp / "input" / "fun_24p.mkv"
        video.parent.mkdir()
        video.write_bytes(b"the input video")
        copy.write_bytes(b"the copy a job renders from")

        def about(path: Path, frames: int, rate: float = 24.0) -> dict:
            return {"name": path.name, "bytes": path.stat().st_size, "frames": frames, "rate": rate, "size": [32, 16],
                    "head_sha256": cf.head_hash(path)}
        job = dict(_without_corrections(), source=str(copy))
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output")
        (tmp / "output").mkdir()
        # no folder: refused, with the command that makes one, and nothing written
        _refused(lambda: tool.build(_job_file(tmp, job), "preview", **where), "not been analysed", "clip_analysis.py init", "fun_24p.mkv")
        assert not list((tmp / "data").glob("*_job")), "a refused build left a job folder behind"
        # a non-24 fps source with no 24 fps copy in derived: refused naming the file's rate and clip_analysis.py init --copy
        cf.write_manifest(tmp / "data" / "fun", {"source": about(video, 30, rate=23.976024), "derived": None})
        _refused(lambda: tool.build(_job_file(tmp, dict(job, name="check_rate_no_copy", source=str(video))), "preview", **where),
                 "fun.mkv", "23.976024", "24 fps", "clip_analysis.py init fun.mkv --copy")
        made = cf.write_manifest(tmp / "data" / "fun", {"source": about(video, 30, rate=23.976024), "derived": about(copy, 24, rate=24.0)})
        # a non-24 fps source when derived is 24.0 fps: refused naming the file's rate and the 24 fps derived copy
        _refused(lambda: tool.build(_job_file(tmp, dict(job, name="check_rate_has_copy", source=str(video))), "preview", **where),
                 "fun.mkv", "23.976024", "24 fps", "fun_24p.mkv")
        folder = tool.build(_job_file(tmp, job), "preview", **where)
        assert json.loads((folder / "preview.built.json").read_text())["analysis"] == {"folder": "fun", "fingerprint": made["fingerprint"]}, \
            "the build record does not say which analysis it was built on"
        # the file changed since the folder was made; then: a folder that is another video's
        copy.write_bytes(b"the copy, made again and not the same")
        _refused(lambda: tool.build(_job_file(tmp, dict(job, name="check_b")), "preview", **where), "not the file", "clip_analysis.py init")
        _refused(lambda: tool.build(_job_file(tmp, dict(job, name="check_c", source=str(tmp / "input" / "sun_24p.mkv"))), "preview", **where),
                 "sun_24p.mkv", "not been analysed")
        # the loads are held to it too, before any capture is read
        _refused(lambda: tool.build_loads(_job_file(tmp, dict(job, name="check_d")), "plan", [], inputs=tmp / "input", **where), "not the file")
        # the copy is replaced by another: a job that renders from the one before still builds, on the same folder, held to
        # ITS OWN bytes; a copy the manifest was never told of does not (the state of 2026-10-13, when one name was kept)
        copy.write_bytes(b"the copy a job renders from")
        newer = tmp / "input" / "fun_whole.mkv"
        newer.write_bytes(b"the copy that replaced it")
        kept = cf.retire(cf.read_manifest(tmp / "data" / "fun"), newer.name, "the whole video on one clock", "2026-01-02T00:00:00Z")
        kept = cf.write_manifest(tmp / "data" / "fun", dict(kept, derived=about(newer, 30)))
        old = tool.build(_job_file(tmp, dict(job, name="check_e")), "preview", **where)
        assert json.loads((old / "preview.built.json").read_text())["analysis"] == {"folder": "fun", "fingerprint": kept["fingerprint"]}
        tool.build(_job_file(tmp, dict(job, name="check_f", source=str(newer))), "preview", **where)
        copy.write_bytes(b"the earlier copy, changed since")
        _refused(lambda: tool.build(_job_file(tmp, dict(job, name="check_g")), "preview", **where), "not the file")


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
        where = dict(date="2026-01-01", data=tmp / "data", output=tmp / "output",
                     analysis=lambda job, data: {"folder": "check", "fingerprint": "none: this case is about something else"})
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
               build_refuses_what_it_must, queue_refuses_what_it_must, a_job_is_built_on_its_videos_analysis_folder,
               a_load_lands_on_the_inputs_it_names, a_plan_is_the_render_with_nothing_sampled,
               a_load_leaves_the_rows_model_alone, the_nodes_schemas_accept_a_load,
               a_load_the_graph_cannot_be_built_from, a_load_says_what_its_context_frames_show, a_load_of_chained_windows_takes_context_noise, loads_from_a_capture, a_load_of_two_subjects, a_hair_band_under_a_part,
               a_text_is_of_its_frames, a_load_of_two_shots,
               the_gate_before_a_render, an_uncarved_keep_out_refused):
        case(fn.__name__.replace("_", " "), fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
