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
        assert built["tracker_mask_version"] == version and set(built["saves"]) == {"a", "b", "c"}, built
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
               build_refuses_what_it_must, queue_refuses_what_it_must):
        case(fn.__name__.replace("_", " "), fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
