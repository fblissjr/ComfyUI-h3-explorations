#!/usr/bin/env python3
"""From a shot to a load's job file (`bench/shot_to_load.py`) on a made video: what each step reads, writes and refuses.

A shots record of three shots, a plan, a cast file, an analysis folder and a preview's job folder are made in a
temporary folder; the machine is handed in (a queue that is empty or not, and a `run` that records each command
in place of running it). Every job file the tool writes is handed to the
builder's own `check_job`, and a load's to `masked_job.loads_of`, so a file the builder would refuse fails here.

1. **The frame a tracker picks on is a field.** `frames_on` first, then `frames_boxed`; a frame named only in
   the prose of `how` is NOT read, and the subject falls back to the shot's middle with the note saying so.
2. **The order** is the plan's unless the shot carries a person's `order`, and an `order` that is not the
   shot's subjects is refused. A subject the record does not say is in the shot gets no tracker.
3. **A subject with a box in the cast file inside the shot is seeded** from it and carries no `pick_at`; a box
   outside the shot, or another subject's, seeds nothing.
4. **The plan must say what from**: a subject with no `still` or no `parts` is refused with the keys.
5. **The preview step** writes `shot_c_<first>`, refuses one that is there, and queues only when told, on an
   empty queue.
5b. **The first real run's fault** (2026-10-13). With the REAL `World.run`, which keeps each command's output, and
   a stand-in for the builder that refuses a job folder that exists as the builder does: the preview builds and
   queues, its output is kept beside the job file, and the job folder holds only what the builder made. A job
   folder or an output folder that is there, and a file that is not the one the video's folder was made from, are
   refused before anything is written, in a dry run as well. A seed whose cast line has an empty field is printed
   with no hole and the empty field said to be the file's.
6. **A load's name is made**: `<subject>_<form>_<first>`, refused when a job file, a capture or a job folder of
   any day has it, and the next free letter with `--remake`.
7b. **A job renders from the video's present copy**, whatever copy its preview or the shots record was made on:
   an earlier copy on the same frame clock is said in the job's note with why it was replaced; a copy that begins
   on a later frame, and a file the video never had, are refused with nothing written.
7. **A load's job file stands alone**: `defaults` are `h3_config`'s, the margin by the plan's region; `pick`
   copied and no `pick_at`; the reference named by subject and span, not by the load's name; a text entry with
   no writer, which the chain refuses; a cut with no words refused and with them written into `left_out`; a
   `hair_band` from the plan; and the refusals (a subject not in the shot, a preview of other frames, no pose
   pass for the subject, a whole-person region).
8. **The reference step** (and `--existing`, for a load made again on the same frames: nothing run, the record
   held to the job's frames) runs the pose pass in the subject's tracked box on the LOAD's frames with the mask
   read from the preview's first frame, then the reference tool; writes the drawn form into the job file by
   default and the plain cut only with words; refuses without the store, on a queue that is not empty, with
   `--plain ""`, and where the reference's folder is there.
9. **The figures** are the same rows at the canvas and at the reference's scale, each with its frames under
   the drawing line.

No server, no card, no video decoded.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_shot_to_load.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
sys.path.insert(0, str(HERE.parent))
from _lib import case, finish  # noqa: E402

import cast_file  # noqa: E402
import clip_folder  # noqa: E402
import h3_config as cfg  # noqa: E402
import load_chain as lc  # noqa: E402
import masked_job  # noqa: E402
import masked_job_graph as graphs  # noqa: E402
import shot_to_load as stl  # noqa: E402


class Machine:
    """The machine a step is handed: an empty queue unless told, and a `run` that records."""
    def __init__(self, root: Path):
        self.host, self.dry, self.python, self.pose_model = "nowhere:0", False, "PY", "pose-model"
        self.output, self.inputs, self.data = root / "output", root / "input", root / "data"
        self.busy, self.ran, self.on = (0, 0), [], {}

    def queue(self):
        return self.busy

    def run(self, ctx, label, argv, stop=True):
        self.ran.append((label, [str(x) for x in argv]))
        if label in self.on:
            self.on[label]()
        return 0

    def said(self, label: str) -> list[str]:
        return next(argv for name, argv in self.ran if name == label)

    def labels(self) -> list[str]:
        return [name for name, _ in self.ran]


def refused(fn, *words: str) -> str:
    try:
        fn()
    except lc.Refused as e:
        assert all(w in str(e) for w in words), f"refused without saying {words}: {e}"
        return str(e)
    raise AssertionError(f"not refused; expected {words}")


BY = {"present": "yes", "how": "a person, on a picture", "by": "a person"}
NO = {"present": "no", "how": "a person, on a picture", "by": "a person"}


def shots_record() -> dict:
    """Three shots. 100-159: a (marked on 130), b (boxed on 120), c (a frame only in the prose). 160-199: a alone.
    200-259: nobody."""
    return {"schema": "h3_shots/1", "video": "v.mp4", "source_frames": 260, "copy": "copy.mov", "span": [0, 259], "subjects": ["a", "b", "c"],
            "shots": [{"first": 0, "last": 99, "who": {"a": dict(NO), "b": dict(NO), "c": dict(NO)}},
                      {"first": 100, "last": 159, "who": {"a": dict(BY, frames_on=[130, 140], frames_boxed=[]),
                                                          "b": dict(BY, frames_on=[], frames_boxed=[120]),
                                                          "c": dict(BY, how="frame 111: 2 is c")}},
                      {"first": 160, "last": 199, "who": {"a": dict(BY, frames_on=[999]), "b": dict(NO), "c": dict(NO)}},
                      {"first": 200, "last": 259, "who": {"a": dict(NO), "b": dict(NO), "c": dict(NO)}}]}


def plan_record() -> dict:
    return {"schema": "h3_plan/1", "video": "v.mp4", "subjects": {
        "b": {"region": "face and hair", "by": "a person", "still": "b.png", "parts": ["hair", "face_and_neck"], "hair_band": {"value": 0.6, "from": "carried, untested"}},
        "a": {"region": "face", "by": "a person", "still": "a.png", "parts": ["face_and_neck"]},
        "c": {"region": "whole", "by": "a person", "still": "c.png", "parts": ["face_and_neck"]}}}


class Made:
    def __init__(self, root: Path):
        self.root, self.m = root, Machine(root)
        for d in ("input", "output", "data/v", "jobs"):
            (root / d).mkdir(parents=True)
        for name in ("copy.mov", "copy_first.mov", "copy_part.mov"):
            (root / "input" / name).write_bytes(b"")
        # each file's identity as it is on disk (empty), so the builder's own hold on the file passes as it would on a real one
        about = lambda name, frames: {"name": name, "bytes": 0, "frames": frames, "rate": 24.0, "size": [64, 48],   # noqa: E731
                                      "head_sha256": clip_folder.head_hash(root / "input" / name) if (root / "input" / name).is_file() else "made"}
        # the video's present copy, a copy it had before on the same frame clock, and one that begins on a later frame
        clip_folder.write_manifest(root / "data/v", {"source": about("v.mp4", 260), "derived": about("copy.mov", 260), "copies_before": [
            dict(about("copy_first.mov", 200), first_source_frame=0, replaced_by="copy.mov", replaced_at="2026-01-01", why="its sound was made again"),
            dict(about("copy_part.mov", 160), first_source_frame=100, replaced_by="copy.mov", replaced_at="2026-01-01", why="one clock")]})
        self.shots_file, self.plan_file, self.jobs = root / "shots.json", root / "plan.json", root / "jobs"
        self.shots, self.plan = shots_record(), plan_record()
        self.write()

    def write(self) -> None:
        self.shots_file.write_text(json.dumps(self.shots))
        self.plan_file.write_text(json.dumps(self.plan))

    def read(self):
        self.write()
        return stl._load(str(self.shots_file), str(self.plan_file))

    def preview(self, first: int = 100, **more) -> dict:
        shots, plan = self.read()
        return stl.preview_job(shots, plan, stl.shot_of(shots, first), "v", "/in/copy.mov", "the session", now="now", **more)

    def preview_ran(self, job: dict, day: str = "2026-01-02", pose_less: str = "") -> Path:
        """The preview's job folder and the files its record lists, as a finished preview leaves them."""
        name = job["job"]["name"]
        if pose_less:
            next(s for s in job["job"]["subjects"] if s["label"] == pose_less)["pose"] = False
        folder = self.root / "data" / f"{day}_{name}_job"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(json.dumps(job))
        out, listed = self.root / "output" / job["job"]["output"] / name, []
        out.mkdir(parents=True, exist_ok=True)
        for s in job["job"]["subjects"]:
            for kind, suffix in (("track", "mkv"), ("parts", "mkv"), ("mesh", "mp4")):
                (out / f"{name}_{s['label']}_{kind}_00001.{suffix}").write_bytes(b"")
                listed.append(f"{job['job']['output']}/{name}/{name}_{s['label']}_{kind}_00001.{suffix}")
            (out / f"{name}_{s['label']}_shots_00001_shots.json").write_text("{}")
        (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": listed}))
        return folder

    def load(self, subject: str = "a", shot: int = 100, name: str = "a_fp_0100", **more) -> dict:
        shots, plan = self.read()
        pjob = self.preview(shot)
        folder = self.preview_ran(pjob)
        return stl.load_job(shots, plan, stl.shot_of(shots, shot), subject, pjob["job"], str(folder), "v", self.m.output, "the session", "2026-01-03", name, now="now", **more)


def args(**more) -> argparse.Namespace:
    given = dict(by="the session", day="2026-01-03", cast=None, queue=False, first=None, frames=None, why=None, remake=False, plain=None, model_store="",
                 existing=False)
    given.update(more)
    return argparse.Namespace(**given)


def the_pick_frame_is_a_field() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        job = made.preview()["job"]
        graphs.check_job(job)
        at = {s["label"]: s["pick_at"] for s in job["subjects"]}
        assert at["a"] == 130, f"a was marked on 130 and 140 and is picked on {at['a']}"
        assert at["b"] == 120, f"b has a box on 120 and is picked on {at['b']}"
        # the control: c's `how` names frame 111 in a sentence and no field does. A pattern over the sentence would give 111.
        assert at["c"] == (100 + 159) // 2 != 111, f"c's frame is named only in prose and is picked on {at['c']}"
        note = job["note"]
        assert "c: the largest person on frame 129, THE SHOT'S MIDDLE" in note and "A fallback" in note, note
        assert "a: the largest person on frame 130, a frame the record says they were marked on (frames_on)" in note, note
        assert "b: the largest person on frame 120, a frame a box stands on in the cast file (frames_boxed)" in note, note
        # a field's frame outside the shot is not a frame of the shot: the middle, and said
        solo = made.preview(160)["job"]
        assert solo["subjects"][0]["pick_at"] == (160 + 199) // 2 and "THE SHOT'S MIDDLE" in solo["note"], solo["subjects"]
        assert (job["name"], job["first"], job["frames"], job["canvas"]) == ("shot_c_0100", 100, 60, list(cfg.MASKED_JOB_CANVAS)), job
        assert all(s["pick"] == "largest" and "pose" not in s for s in job["subjects"]), "every subject of a shot's preview gets a pose pass"
    return "a tracker picks on a frame the record names as a field (marked, then boxed); a frame named only in prose is not read and the middle is said to be a fallback"


def the_order_and_who_is_in_the_shot() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        order = lambda job: [s["label"] for s in job["job"]["subjects"]]
        first = made.preview()
        assert order(first) == ["b", "a", "c"], f"the plan's order is b, a, c: {order(first)}"
        assert "the plan's subject order, a starting value" in first["job"]["note"]
        made.shots["shots"][1]["order"] = ["c", "a", "b"]
        again = made.preview()
        assert order(again) == ["c", "a", "b"] and "the shot's own `order`" in again["job"]["note"], order(again)
        made.shots["shots"][1]["order"] = ["c", "a"]
        refused(made.preview, "`order`", "not the subjects")
        del made.shots["shots"][1]["order"]
        assert order(made.preview(160)) == ["a"], "a subject the record does not say is in the shot got a tracker"
        refused(lambda: made.preview(200), "nobody of ours", "nothing to preview")
        refused(lambda: made.preview(101), "no shot of the shots record begins on frame 101")
        made.shots["shots"][1]["who"]["b"]["present"] = "unsure"
        assert order(made.preview()) == ["a", "c"], "a subject the record is unsure of got a tracker"
    return "the trackers run in the plan's order unless the shot carries a person's; only subjects the record says are in the shot are tracked"


def a_box_in_the_cast_file_seeds() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        w, h = cfg.MASKED_JOB_CANVAS
        lines = [cast_file.entry("a", 125, cast_file.HERE, by="a person", box=[10, 20, 30, 40], size=[w // 2, h // 2], table="t.json", at="t1"),
                 cast_file.entry("b", 170, cast_file.HERE, by="a person", box=[10, 20, 30, 40], size=[w, h], table="t.json", at="t2"),
                 cast_file.entry("c", 110, cast_file.NONE_OUTLINED, by="a person", at="t3")]
        subjects = {s["label"]: s for s in made.preview(cast_lines=lines)["job"]["subjects"]}
        # a's box on source frame 125 is frame 25 of the load, at twice the size it was read at
        assert subjects["a"].get("seeds") == "frame 25: 20, 40, 60, 80" and "pick_at" not in subjects["a"], subjects["a"]
        # b's box is in another shot; c's answer says only that the outlined people are not c
        assert "seeds" not in subjects["b"] and subjects["b"]["pick_at"] == 120, subjects["b"]
        assert "seeds" not in subjects["c"] and "pick_at" in subjects["c"], subjects["c"]
        job = made.preview(cast_lines=lines)["job"]
        graphs.check_job(job)
        assert "a: SEEDED from the cast file" in job["note"]
        # a cast line whose own words have an empty field (a page answer with no time on it): printed with no hole, and said
        holed = [cast_file.entry("a", 125, cast_file.HERE, by="a person, on the page at  (lines 3-4)", box=[10, 20, 30, 40], size=[w, h], table="t.json", at="t1")]
        note = made.preview(cast_lines=holed)["job"]["note"]
        assert "at  (" not in note and "a person, on the page at (lines 3-4)" in note and "a field of that line is EMPTY in the cast file" in note, note
        assert "EMPTY in the cast file" not in job["note"], "a cast line with no empty field was said to have one"
        assert all("seeds" not in s for s in made.preview()["job"]["subjects"]), "a preview with no cast file was seeded"
    return "a subject with a box in the cast file inside the shot is seeded from it, scaled to the canvas, and carries no pick_at; a box in another shot seeds nothing"


def the_plan_says_what_from() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        for key in ("still", "parts"):
            kept = made.plan["subjects"]["a"].pop(key)
            refused(made.preview, "'a'", "`still`", "`parts`", "the plan file")
            made.plan["subjects"]["a"][key] = kept
        made.preview()
    return "a subject whose plan lacks its still or its part ticks is refused with the keys to add"


def the_preview_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs))
        stl.step_preview(a, made.m)
        out = made.jobs / "shot_c_0100.json"
        job = json.loads(out.read_text())
        graphs.check_job(job["job"])
        assert job["job"]["source"] == str(made.root / "input" / "copy.mov") and job["job"]["output"] == "Video/v/the session", job["job"]
        assert made.m.ran == [], f"a preview ran with no --queue: {made.m.labels()}"
        refused(lambda: stl.step_preview(a, made.m), "exists")
        out.unlink()
        made.m.busy = (1, 0)
        a.queue = True
        refused(lambda: stl.step_preview(a, made.m), "the queue is not empty")
        out.unlink()
        made.m.busy = (0, 0)
        stl.step_preview(a, made.m)
        assert made.m.labels() == ["build preview", "queue preview"], made.m.labels()
        assert made.m.said("build preview")[1:] == ["bench/masked_job.py", "build", "preview", str(out), "--date", "2026-01-03"], made.m.said("build preview")
        assert made.m.said("queue preview")[1:5] == ["bench/masked_job.py", "queue", str(made.root / "data/2026-01-03_shot_c_0100_job"), "preview"]
    return "the preview's job file is written as shot_c_<first>, once; it is queued only when told, on an empty queue"


def the_preview_step_against_a_builder_that_refuses_a_folder_that_is_there() -> str:
    """The fault of the first real run (2026-10-13): the tool kept a command's output under the job folder, which made
    the folder, and the preview's builder refuses a job folder that exists. The machine here is the REAL `World.run`
    (it keeps the output, as on a real run), and the command it runs is a stand-in for the builder that refuses as the
    builder does. With the output kept under the job folder again, this case is red."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        ran = []

        class Real(Machine):
            def run(self, ctx, label, argv, stop=True):
                return lc.World.run(self, ctx, label, argv, stop)

        def builder(argv, stdout=None, **more):
            """`masked_job.py build preview JOB --date DAY` and `queue FOLDER preview`, as far as their folders go."""
            ran.append(argv[2])
            if argv[2] == "build":
                job = json.loads(Path(argv[4]).read_text())["job"]
                folder = made.root / "data" / f"{argv[6]}_{job['name']}_job"
                if folder.exists():
                    stdout.write(f"REFUSED: the job folder {folder} exists. A run never reuses a name: give the job a new `name`.\n")
                    return subprocess.CompletedProcess(argv, 5)
                folder.mkdir(parents=True)
                (folder / "job.json").write_text(json.dumps({"job": job}))
                stdout.write("built\n")
            else:
                assert Path(argv[3]).is_dir(), f"the queue was handed a job folder that is not there: {argv[3]}"
                stdout.write("queued\n")
            return subprocess.CompletedProcess(argv, 0)

        m = Real(made.root)
        kept, lc.subprocess.run = lc.subprocess.run, builder
        try:
            a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), queue=True)
            stl.step_preview(a, m)
            folder = made.root / "data" / "2026-01-03_shot_c_0100_job"
            assert ran == ["build", "queue"] and (folder / "job.json").is_file(), ran
            # the output is kept beside the job file, and the job folder holds only what the builder put there
            assert sorted(p.name for p in (made.jobs / "chain").iterdir()) == ["shot_c_0100.build_preview.out", "shot_c_0100.queue_preview.out"]
            assert sorted(p.name for p in folder.iterdir()) == ["job.json"], sorted(p.name for p in folder.iterdir())
            # held as the builder holds, BEFORE anything is written: a job folder or an output folder that is there, by name;
            # and a dry run stops on the same refusals
            (made.jobs / "shot_c_0100.json").unlink()
            for dry in (False, True):
                m.dry = dry
                refused(lambda: stl.step_preview(a, m), "the job folder", "shot_c_0100_job exists", "never reuses a name")
            assert not (made.jobs / "shot_c_0100.json").exists() and ran == ["build", "queue"], "a refused preview wrote its job file or ran a command"
            m.dry = False
            shutil.rmtree(folder)
            (m.output / "Video/v/the session/shot_c_0100").mkdir(parents=True)
            refused(lambda: stl.step_preview(a, m), "the output folder", "exists")
            # a file that is not the one the video's folder was made from is the builder's refusal too
            shutil.rmtree(m.output / "Video")
            (made.root / "input" / "copy.mov").write_bytes(b"changed since")
            refused(lambda: stl.step_preview(a, m), "copy.mov is not the file this folder was made from")
        finally:
            lc.subprocess.run = kept
    return ("with the real run keeping each command's output, a builder that refuses a job folder that is there builds and queues the preview; the output is "
            "beside the job file; a folder that is there, and a file that is not the video's, are refused before anything is written, in a dry run too")


def a_loads_name_is_made() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        folder = made.preview_ran(made.preview())
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        stl.step_job(a, made.m)
        assert (made.jobs / "a_fp_0100.json").is_file(), sorted(p.name for p in made.jobs.iterdir())
        refused(lambda: stl.step_job(a, made.m), "a load named a_fp_0100 exists", "--remake")
        a.remake = True
        stl.step_job(a, made.m)
        assert (made.jobs / "a_fp_0100b.json").is_file()
        # a capture of another day holds the next letter: the name after it is taken
        (made.root / "data" / "2025-12-30_a_fp_0100c").mkdir()
        stl.step_job(a, made.m)
        assert (made.jobs / "a_fp_0100d.json").is_file() and not (made.jobs / "a_fp_0100c.json").exists(), sorted(p.name for p in made.jobs.iterdir())
        # the name carries the load's own first frame and the region's form
        b = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="b", preview=str(folder), first=104, frames=20, why="a look cut it")
        stl.step_job(b, made.m)
        assert (made.jobs / "b_hfp_0104.json").is_file(), sorted(p.name for p in made.jobs.iterdir())
        c = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="c", preview=str(folder))
        refused(lambda: stl.step_job(c, made.m), "'whole'", "part loads")
        (folder / "preview.queued.json").write_text(json.dumps({"status": "error"}))
        refused(lambda: stl.step_job(args(**{**vars(a), "remake": False, "shot": 100, "first": 101, "frames": 5, "why": "w"}), made.m), "did not end in success")
        # the job file names its preview from the tree it is in, and the chain reads it
        d = json.loads((made.jobs / "a_fp_0100.json").read_text())
        assert d["job"]["preview"] == str(folder), d["job"]["preview"]
    return "a load's name is its subject, its form and its first frame; one a job file, a capture or a job folder has is refused, and a remake takes the next free letter"


def a_loads_job_file_stands_alone() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        d = made.load()
        job, load = d["job"], d["loads"][0]
        graphs.check_job(job)
        assert d["defaults"] == {**cfg.MASKED_LOAD_DEFAULTS, "margin": cfg.MASKED_LOAD_MARGIN["face"], "seed": cfg.SEED}, d["defaults"]
        view = masked_job.loads_of(d)[0]
        for key, value in d["defaults"].items():
            assert view[key] == value, f"the builder's view of the load has {key}={view.get(key)!r}, not the default {value!r}"
        assert [(s["label"], s["pick"]) for s in job["subjects"]] == [("b", "largest"), ("a", "largest"), ("c", "largest")], job["subjects"]
        assert all("pick_at" not in s and "seeds" not in s for s in job["subjects"]), "a load carries a pick_at or seeds: it tracks nobody"
        assert next(s for s in job["subjects"] if s["label"] == "a")["still"] == "a.png"
        want = made.m.output / "Video/v/references/a_0100_0159/a_head_reference_0100_0159.mp4"
        assert (load["motion_file"], load["motion_first"], load["others"], load["text"]) == (str(want), 100, ["b", "c"], "a_fp_0100"), load
        assert "a_fp_0100" not in str(Path(load["motion_file"]).parent), "a reference is filed under a load's name"
        assert job["capture"] == "data/2026-01-03_a_fp_0100" and job["left_out"] == [] and "hair_band" not in load, (job["capture"], job["left_out"])
        assert job["texts"] == {"a_fp_0100": {"file": "a_fp_0100.txt", "frames": [100, 159], "by": ""}}, job["texts"]
        assert job["reference"]["form"] == "drawn" and job["reference"]["made"] is None, job["reference"]
        # the chain refuses it until the text is a file with a writer, and a tool before the text reads it
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(d))
        refused(lambda: lc.read_job(made.jobs / "a_fp_0100.json", made.root), "is not written", "a person's to write")
        (made.jobs / "a_fp_0100.txt").write_text("words\n")
        refused(lambda: lc.read_job(made.jobs / "a_fp_0100.json", made.root), "texts.a_fp_0100.by is empty")
        assert lc.read_job(made.jobs / "a_fp_0100.json", made.root, need_text=False)["preview"]["first"] == 100
        # a load cut by a look: its words are owed, its frames are the reference's, the frames cut are in left_out
        refused(lambda: made.load(first=104, frames=40), "say what a look cut", "--why")
        cut = made.load(first=104, frames=40, why="a hand is over the face before 104 and after 143", name="a_fp_0104")
        assert cut["job"]["left_out"] == [{"subject": "a", "first": 100, "frames": 4, "why": "a hand is over the face before 104 and after 143"},
                                          {"subject": "a", "first": 144, "frames": 16, "why": "a hand is over the face before 104 and after 143"}], cut["job"]["left_out"]
        assert cut["loads"][0]["motion_file"].endswith("references/a_0104_0143/a_head_reference_0104_0143.mp4") and cut["loads"][0]["motion_first"] == 104, cut["loads"][0]
        assert (cut["job"]["first"], cut["job"]["frames"]) == (104, 40) and "CUT BY A LOOK" in cut["job"]["note"]
        refused(lambda: made.load(first=150, frames=20, why="w"), "not inside the shot")
        # the margin and the band are the plan's region's and the plan's
        band = made.load("b", name="b_hfp_0100")
        assert band["defaults"]["margin"] == cfg.MASKED_LOAD_MARGIN["face and hair"] != cfg.MASKED_LOAD_MARGIN["face"] and band["loads"][0]["hair_band"] == 0.6, band
        masked_job.loads_of(band)
        refused(lambda: made.load("c", name="c_x_0100"), "'whole'", "part loads")
        refused(lambda: made.load("b", shot=160, name="b_hfp_0160"), "does not say 'b' is in 160-199")
        # a preview of other frames, and a preview with no pose pass for the subject
        shots, plan = made.read()
        other = made.preview(160)["job"]
        refused(lambda: stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", other, "p", "v", made.m.output, "s", "2026-01-03", "n"), "not of the shot 100-159")
        bare = made.preview()["job"]
        next(s for s in bare["subjects"] if s["label"] == "a")["pose"] = False
        refused(lambda: stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", bare, "p", "v", made.m.output, "s", "2026-01-03", "n"), "no pose pass for 'a'")
    return ("a load's job file carries its settings in defaults from h3_config, the margin by the plan's region, pick with no pick_at, a reference named by "
            "subject and span, a text with no writer that the chain refuses, and a cut's words in left_out")


def a_job_renders_from_the_present_copy() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        present = str(made.root / "input" / "copy.mov")
        # a preview made on a copy the video had before, on the same frame clock: the load renders from the PRESENT copy and says so
        old = made.preview()
        old["job"]["source"] = "/elsewhere/copy_first.mov"
        folder = made.preview_ran(old)
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        stl.step_job(a, made.m)
        job = json.loads((made.jobs / "a_fp_0100.json").read_text())["job"]
        assert job["source"] == present != old["job"]["source"], f"a load took its preview's copy: {job['source']}"
        assert "ITS PREVIEW WAS MADE ON AN EARLIER COPY: copy_first.mov, which copy.mov replaced (its sound was made again)" in job["note"], job["note"]
        graphs.check_job(job)
        # a preview made on the present copy says nothing of it
        now = made.preview()
        now["job"]["source"] = present
        made.preview_ran(now)
        a.remake = True
        stl.step_job(a, made.m)
        assert "EARLIER COPY" not in json.loads((made.jobs / "a_fp_0100b.json").read_text())["job"]["note"]
        # a preview made on a copy that begins on a later frame: its frame numbers are another clock, and nothing is written
        part = made.preview()
        part["job"]["source"] = "/elsewhere/copy_part.mov"
        made.preview_ran(part)
        before = sorted(p.name for p in made.jobs.iterdir())
        refused(lambda: stl.step_job(a, made.m), "copy_part.mov begins on source frame 100", "another clock")
        assert sorted(p.name for p in made.jobs.iterdir()) == before
        # a copy the manifest has never held is not this video's
        part["job"]["source"] = "/elsewhere/stranger.mov"
        made.preview_ran(part)
        try:
            stl.step_job(a, made.m)
        except clip_folder.Refused as e:
            assert "stranger.mov" in str(e), str(e)
        else:
            raise AssertionError("a preview made on a file the video never had was taken")
        # the shots record naming an earlier copy: the preview is written for the present one, and says so
        made.shots["copy"] = "copy_first.mov"
        made.write()
        p = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=160, folder=str(made.jobs))
        stl.step_preview(p, made.m)
        pv = json.loads((made.jobs / "shot_c_0160.json").read_text())["job"]
        assert pv["source"] == present and "THE SHOTS RECORD NAMES AN EARLIER COPY: copy_first.mov" in pv["note"], pv
        made.shots["copy"] = "copy_part.mov"
        made.write()
        refused(lambda: stl.step_preview(args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs)), made.m), "another clock")
    return ("a load whose preview was made on an earlier copy renders from the video's present copy and its note says which copy and why it was replaced; "
            "a copy on another frame clock is refused and nothing is written; so is a file the video never had")


def the_reference_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        d = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104")
        d["job"]["preview"] = str(Path(d["job"]["preview"]))
        job_file = made.jobs / "a_fp_0104.json"
        job_file.write_text(json.dumps(d))
        folder = made.m.output / "Video/v/references/a_0104_0143"
        drawn = folder / "a_head_reference_0104_0143.mp4"
        figures = lambda points, scale: {"frames": 40, "the_drawing_line_px": 1.6, "measured_on": "made", "cannot_see": "made",
                                         "at_the_canvas": {"face_height_px": {}, "lid_gap_px": {}, "frames_with_the_gap_under_the_line": 30},
                                         "in_the_reference": {"enlarged": scale, "face_height_px": {}, "lid_gap_px": {}, "frames_with_the_gap_under_the_line": 2}}
        record = {"scale": [2.5, 2.5], "files": {"plain_cut": "a_head_reference_0104_0143_plain_cut.mp4"}, "frames": [104, 143], "passed": True,
                  "made_at": "2026-01-03T09:00:00Z", "made_by": "the first session"}
        made.m.on["head reference"] = lambda: drawn.with_suffix(".json").write_text(json.dumps(record))
        made.m.on["pose pass"] = lambda: (folder / "a_0104_0143_points.npz").write_bytes(b"")
        kept = lc.REPO
        import os
        try:
            a = args()
            a.job_file = str(job_file)
            os.environ.pop(lc.MODEL_STORE, None)
            refused(lambda: stl.step_reference(a, made.m, figures), "the pose model's store", lc.MODEL_STORE)
            a.model_store = "/store"
            a.plain = "  "
            refused(lambda: stl.step_reference(a, made.m, figures), "--plain takes the reason in words")
            a.plain = None
            made.m.busy = (0, 1)
            refused(lambda: stl.step_reference(a, made.m, figures), "the queue is not empty")
            made.m.busy = (0, 0)
            assert made.m.ran == [] and not folder.exists(), "something ran, or the folder was made, before the refusals were through"
            stl.step_reference(a, made.m, figures)
        finally:
            lc.REPO = kept
        assert made.m.labels() == ["pose pass", "head reference"], made.m.labels()
        pose, ref = made.m.said("pose pass"), made.m.said("head reference")
        opt = lambda argv, key: argv[argv.index(key) + 1]
        track = made.m.output / "Video/v/the session/shot_c_0100/shot_c_0100_a_track_00001.mkv"
        mesh = made.m.output / "Video/v/the session/shot_c_0100/shot_c_0100_a_mesh_00001.mp4"
        # the LOAD's frames, with the mask and the body model read from the PREVIEW's first frame
        assert (opt(pose, "--first"), opt(pose, "--count"), opt(pose, "--box-mask"), opt(pose, "--box-mask-first")) == ("104", "40", str(track), "100"), pose
        assert opt(pose, "--model-folder") == "/store/pose-model" and opt(pose, "--canvas") == "1024x768" and opt(pose, "--video") == "/in/copy.mov", pose
        assert (opt(ref, "--first"), opt(ref, "--count"), opt(ref, "--mesh"), opt(ref, "--mesh-first"), opt(ref, "--out")) == ("104", "40", str(mesh), "100", str(drawn)), ref
        assert opt(ref, "--points") == opt(pose, "--out") == str(folder / "a_0104_0143_points.npz")
        after = json.loads(job_file.read_text())
        block = after["job"]["reference"]
        assert (block["form"], block["said_by"], block["record"]) == ("drawn", "the session", str(drawn.with_suffix(".json"))) and block["made"].endswith("Z"), block
        assert block["figures"]["in_the_reference"]["enlarged"] == 2.5 and after["loads"][0]["motion_file"] == str(drawn), block
        # the figures decide nothing: thirty frames under the line at the canvas, and the form is the drawn one
        assert block["figures"]["at_the_canvas"]["frames_with_the_gap_under_the_line"] == 30 and block["form"] == "drawn"
        # made once: a reference for this subject and span is there
        refused(lambda: stl.step_reference(a, made.m, figures), "exists", "a new job", "--existing")
        # a load made again on the same frames reads the reference that is there: nothing runs, and the block says whose it was
        again = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104b")
        again_file = made.jobs / "a_fp_0104b.json"
        again_file.write_text(json.dumps(again))
        e = args(existing=True)
        e.job_file = str(again_file)
        ran = len(made.m.ran)
        stl.step_reference(e, made.m, figures)
        took = json.loads(again_file.read_text())["job"]["reference"]
        assert len(made.m.ran) == ran and took["made"] == "2026-01-03T09:00:00Z" and "the first session" in took["taken_as_it_was"] and took["form"] == "drawn", took
        drawn.with_suffix(".json").write_text(json.dumps({**record, "frames": [100, 143]}))
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "frames [100, 143]", "the load is 104-143")
        drawn.with_suffix(".json").write_text(json.dumps({**record, "passed": False}))
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "passed=False")
        other = made.load(name="a_fp_0100")
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(other))
        e.job_file = str(made.jobs / "a_fp_0100.json")
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "not both there")
        # the plain cut, on a person's word, kept
        made2 = Made(Path(tmp) / "second")
        d2 = made2.load()
        job_file2 = made2.jobs / "a_fp_0100.json"
        job_file2.write_text(json.dumps(d2))
        drawn2 = made2.m.output / "Video/v/references/a_0100_0159/a_head_reference_0100_0159.mp4"
        made2.m.on["head reference"] = lambda: drawn2.with_suffix(".json").write_text(json.dumps({"scale": [1.0, 1.0], "files": {"plain_cut": "the_plain_one.mp4"}}))
        b = args(model_store="/store", plain="the face is a few pixels tall and the lines are a smear on the sheet")
        b.job_file = str(job_file2)
        stl.step_reference(b, made2.m, figures)
        after2 = json.loads(job_file2.read_text())
        assert after2["loads"][0]["motion_file"] == str(drawn2.parent / "the_plain_one.mp4"), after2["loads"][0]["motion_file"]
        assert (after2["job"]["reference"]["form"], after2["job"]["reference"]["why"], after2["job"]["reference"]["said_by"]) == (
            "plain", "the face is a few pixels tall and the lines are a smear on the sheet", "the session"), after2["job"]["reference"]
        # the builder still reads both files
        graphs.check_job(after["job"]), graphs.check_job(after2["job"])
    return ("the reference is made on the load's own frames with the preview's mask and body model read from the preview's first frame; the drawn form is "
            "written by default and the plain cut only with words; refused without the store, on a busy queue and where the folder is there")


def the_figures() -> str:
    from _lib import face_points as fp
    kept = fp.load, fp.table
    line = cfg.POSE_READING["lid_gap_px_to_draw"]
    rows = [{"face_height_px": 40.0, "lid_gap_px": line * 0.25}, {"face_height_px": 50.0, "lid_gap_px": line * 0.5}, {"face_height_px": 60.0, "lid_gap_px": line * 2}]
    try:
        fp.load, fp.table = (lambda path: {}), (lambda points: rows)
        got = stl.figures_of(Path("made.npz"), 3.0)
    finally:
        fp.load, fp.table = kept
    c, r = got["at_the_canvas"], got["in_the_reference"]
    assert got["frames"] == 3 and got["the_drawing_line_px"] == line, got
    assert c["face_height_px"] == {"least": 40.0, "median": 50.0, "most": 60.0} and c["frames_with_the_gap_under_the_line"] == 2, c
    # at three times the size the half-line gap clears the line and the quarter-line gap does not
    assert r["enlarged"] == 3.0 and r["face_height_px"] == {"least": 120.0, "median": 150.0, "most": 180.0} and r["frames_with_the_gap_under_the_line"] == 1, r
    assert "form" not in got and "verdict" not in json.dumps(got), "the figures carry a reading"
    return "the face's height and the lash-line gap are the same rows at the canvas and at the reference's scale, each with its frames under the line, and no reading"


case("the frame a tracker picks on is a field of the shots record", the_pick_frame_is_a_field)
case("the trackers' order, and who is in the shot", the_order_and_who_is_in_the_shot)
case("a box in the cast file seeds its tracker", a_box_in_the_cast_file_seeds)
case("the plan says what a subject's face is taken from", the_plan_says_what_from)
case("preview: the job file, once, queued only when told", the_preview_step)
case("preview against a builder that refuses a folder that is there: the first real run's fault", the_preview_step_against_a_builder_that_refuses_a_folder_that_is_there)
case("a load's name is made, and refused when taken", a_loads_name_is_made)
case("a load's job file stands alone", a_loads_job_file_stands_alone)
case("a job renders from the video's present copy", a_job_renders_from_the_present_copy)
case("reference: on the load's frames, drawn by default", the_reference_step)
case("the figures, at the canvas and in the reference", the_figures)

if __name__ == "__main__":
    sys.exit(finish())
