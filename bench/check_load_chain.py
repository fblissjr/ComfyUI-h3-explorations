#!/usr/bin/env python3
"""The load chain (`bench/load_chain.py`) on a made job: what each step reads, runs and refuses, with no server.

A preview's job folder, the files it saved, a load's job file, a capture and a job folder are made in a temporary
folder. The machine is handed in: a queue that is empty or not, a card line, a history, the folders, and a
`run` that records each command in place of running it (and writes the files the command would, where a later
part of the step reads them). Each case holds one step to what the session scripts it replaces did, and to the
rulings that changed them.

1. **The job file and its preview.** One load; `capture` with its day; `preview`, the preview's job folder,
   whose record must be of a finished run; the subject among the preview's; the job's frames inside the
   preview's; a text file. Each lack refused in words.
2. **A mask's `at` is the PREVIEW's first frame.** For a load that begins after its preview does, every mask's
   `at`, `shots_at` and `pose_at` is the preview's first frame and the load's own first frame is in no mask (the
   fault of 2026-10-13, which this tool exists to make untypable). Only the options the job sets on a subject are
   passed, with no `pieces` unless it says; no pose table for a subject whose job says `pose: false`, and a
   refusal when one is owed and missing; the other subjects in the preview's order.
3. **first**: refused when the capture or its `_first` copy exists; who is who made again from the preview's
   tables, each with the frame it begins on, or the file the job names and no command; the capture with the load
   as a plan; the strip under `<capture>_first`.
4. **plan**: refused on a queue that is not empty, with no capture, or with a plan graph already built; and, in a
   dry run as well and before any command, when the builder's own dry build refuses the load (the first real
   load, 2026-10-13: `plan --dry` exited 0 and the plan was then refused on a key the job file held all along).
   That the dry build holds what a build holds and writes nothing is `check_masked_job.py`'s, on the real builder.
5. **on-plan**: refused with no plan file, on a queue that is not empty, or when `_first` exists; the masks the plan
   graph reads under the capture are hashed and recorded (a video outside the capture is not); the capture is
   copied to `_first` and the copy is the same files; the capture is built in place with `--overwrite` on the
   node's plan. A copy that differs is said (the control for the comparison).
6. **The look and the gate**: the last look of this load, on the capture on the node's plan, later than the capture
   as times and not by its maker; a time with an offset against one without is refused; the gate clear. Its
   times are given in UTC.
7. **render**: refused without the two files of words, on a queue that is not empty, with a go or a render graph
   already there, with a job file that is not the one its plan was built from. The go is written beside the job
   file, names whoever looked, and the job file is untouched. A render graph that differs from the plan's in
   more than the preview and the prefix, uncommitted node code, a band file that does not decode equal and a
   mask that no longer decodes equal to what the plan read each stop it before anything is queued; the band
   rule passes a band file that decodes equal.
8. **land**: refused without the pose model's store, on a queue that is not empty, with no window, or when the
   clip exists; the pose pass takes the capture's own track video and first frame, and reads the one window's own
   video. A load of SEVERAL windows: the pose pass reads the joined file; refused when it is not there, when it
   does not hold the load's frame count, or when a window is missing; a window's mask review and zoomed picture
   are not counted as windows.
9. **page**: the item's buttons are the delivery table's own; refused without a person's words, for an item that
   is there, or with no clip; the list is copied into the job folder first.

No server, no card, no video decoded.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_load_chain.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table  # noqa: E402
import load_chain as lc  # noqa: E402


class Machine:
    """The machine a step is handed: an empty queue unless told, and a `run` that records."""
    def __init__(self, root: Path):
        self.host, self.dry, self.python, self.pose_model = "nowhere:0", False, "PY", "pose-model"
        self.output, self.inputs, self.data = root / "output", root / "input", root / "data"
        self.busy, self.ran, self.on, self.voiced, self.voice_frames, self.frames = (0, 0), [], {}, True, (0, 199), {}
        self.builder_says = ""           # what the builder's dry build refuses with; "" when it would build

    def builder_refuses(self, ctx, stage):
        return self.builder_says

    def queue(self):
        return self.busy

    def card(self):
        return "a card line."

    def costs(self, prompt_id):
        return [f"success True ({prompt_id})"]

    def voice(self, source_name):
        if not self.voiced:
            raise lc.Refused("the manifest names no voice tables")
        return {"per_frame": self.data / "voice.csv", "spans": self.data / "spans.json", "frames": list(self.voice_frames)}

    def frames_of(self, path):
        return self.frames[Path(path).name]

    def run(self, ctx, label, argv, stop=True):
        self.ran.append((label, [str(x) for x in argv]))
        if label in self.on:
            self.on[label]()
        return 0

    def said(self, label: str) -> list[str]:
        return next(argv for name, argv in self.ran if name == label)

    def labels(self) -> list[str]:
        return [name for name, _ in self.ran]


class Made:
    """A preview that ran on frames 100 to 159 for b then a, and a load of a on 110 to 129 made from it."""
    def __init__(self, root: Path, **job_more):
        self.root, self.m = root, Machine(root)
        pv = root / "output" / "Video/v/s/pv"
        pv.mkdir(parents=True)
        listed = []
        for label, kinds in (("b", ("track",)), ("a", ("track", "parts", "classes"))):
            for kind in kinds:
                (pv / f"pv_{label}_{kind}_00001.mkv").write_bytes(b"")
                listed.append(f"Video/v/s/pv/pv_{label}_{kind}_00001.mkv")
            (pv / f"pv_{label}_shots_00001_shots.json").write_text("{}")
        (pv / "pv_a_pose_table.json").write_text("{}")
        self.pv = root / "data" / "2026-01-02_pv_job"
        self.pv.mkdir(parents=True)
        (self.pv / "job.json").write_text(json.dumps({"job": {"name": "pv", "first": 100, "frames": 60, "output": "Video/v/s",
                                                              "subjects": [{"label": "b"}, {"label": "a"}]}}))
        self.listed = listed
        (self.pv / "preview.queued.json").write_text(json.dumps({"status": "success", "files": listed}))
        (root / "input").mkdir()
        (root / "input" / "a.png").write_bytes(b"")
        self.inputs = root / "data" / "in"
        self.inputs.mkdir()
        (self.inputs / "ld.txt").write_text("the words of the load\n")
        self.job = {"name": "ld", "source": "/in/copy.mov", "first": 110, "frames": 20, "canvas": [64, 48], "output": "Video/v/s",
                    "capture": "data/2026-01-03_ld", "preview": "data/2026-01-02_pv_job",
                    "subjects": [{"label": "b"}, {"label": "a", "parts": ["face_and_neck"], "still": "a.png"}],
                    "texts": {"ld": {"file": "ld.txt", "frames": [110, 129], "by": "a writer, from these frames"}}}
        self.job.update(job_more)
        self.load = {"name": "ld", "first": 110, "frames": 20, "subject": "a", "others": ["b"], "text": "ld", "motion_file": "/out/ref.mp4"}
        self.job_file = self.inputs / "ld.json"
        self.write()

    def write(self, loads=None) -> None:
        self.job_file.write_text(json.dumps({"job": self.job, "defaults": {}, "loads": [self.load] if loads is None else loads}, indent=1))

    def ctx(self) -> dict:
        return lc.read_job(self.job_file, self.root)

    def capture(self, *, on_plan: bool = True, written: str = "2026-01-03T10:00:00", look_at: str | None = "2026-01-03T10:30:00",
                look_by: str = "a reader", loads=("ld",), verdict: str = "clear") -> Path:
        """A capture folder as the capture tool leaves one, with a look recorded on it unless told otherwise."""
        cap = self.root / "data" / "2026-01-03_ld"
        (cap / "subjects" / "a").mkdir(parents=True, exist_ok=True)
        (cap / "subjects" / "a" / "track__pv.mkv").write_bytes(b"track")
        (cap / "manifest.json").write_text(json.dumps({"first_frame": 110, "frames": 20, "runs": [{"carried_from_the_node": on_plan}],
                                                       "subjects": [{"label": "a", "sightings": [{"by": "pv"}]}]}))
        (cap / "status.json").write_text(json.dumps({"written": written, "made_by": "the maker"}))
        (cap / "flags.json").write_text(json.dumps({"verdict": verdict, "flags": [{}, {}]}))
        if look_at is not None:
            (cap / "looks.json").write_text(json.dumps({"looks": [{"by": "an earlier reader", "at": "2026-01-03T09:00:00", "loads": ["ld"]},
                                                                  {"by": look_by, "at": look_at, "loads": list(loads)}]}))
        return cap

    def planned(self, extra_video: str = "") -> Path:
        """A job folder after the plan: the job file as it was planned from, the plan graph and its queue record."""
        folder = self.root / "data" / "2026-01-03_ld_job"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.plan.json").write_bytes(self.job_file.read_bytes())
        cap = str(self.root / "data" / "2026-01-03_ld")
        graph = {"1": {"class_type": "L", "inputs": {"video": f"{cap}/subjects/a/track__pv.mkv"}},
                 "2": {"class_type": "L", "inputs": {"video": extra_video or "/in/copy.mov"}},
                 "9": {"class_type": "Song", "inputs": {"preview": True, "filename_prefix": "ld_plan", "seed": 1}}}
        (folder / "ld.plan.json").write_text(json.dumps(graph))
        (folder / "ld.plan.queued.json").write_text(json.dumps({"prompt_id": "plan-prompt"}))
        return folder


def refused(fn, *words: str) -> str:
    try:
        fn()
    except lc.Refused as e:
        assert all(w in str(e) for w in words), f"refused without saying {words}: {e}"
        return str(e)
    raise AssertionError(f"not refused; expected {words}")


def args(**more) -> argparse.Namespace:
    given = dict(by="the session", grounds=None, show=None, model_store="", spec=None)
    given.update(more)
    return argparse.Namespace(**given)


def the_job_file_and_its_preview() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        ctx = made.ctx()
        assert (ctx["name"], ctx["subject"], ctx["day"], ctx["capture"].name, ctx["job_folder"].name) == ("ld", "a", "2026-01-03", "2026-01-03_ld", "2026-01-03_ld_job"), ctx
        assert (ctx["preview"]["name"], ctx["preview"]["first"], ctx["preview"]["order"]) == ("pv", 100, ["b", "a"]) and ctx["text"].name == "ld.txt"
        made.write(loads=[made.load, dict(made.load, name="two")])
        refused(made.ctx, "2 loads", "one load a job file")
        made.write()
        for key, words in (("capture", ("names no `capture`",)), ("preview", ("names no `preview`", "never typed")), ("texts", ("names no file",))):
            kept = made.job.pop(key)
            made.write()
            refused(made.ctx, *words)
            made.job[key] = kept
        for change, words in (({"capture": "data/ld"}, ("does not begin with a day",)), ({"first": 90}, ("are not inside the preview pv's",)),
                              ({"frames": 80}, ("are not inside the preview",))):
            kept = {k: made.job[k] for k in change}
            made.job.update(change)
            made.write()
            refused(made.ctx, *words)
            made.job.update(kept)
        # the text is a person's: a file that is not written, or one nobody is said to have written, is refused, except
        # for a tool that runs before the text exists
        made.job["texts"]["ld"]["by"] = ""
        made.write()
        refused(made.ctx, "texts.ld.by is empty")
        made.job["texts"]["ld"]["by"] = "a writer, from these frames"
        made.write()
        (made.inputs / "ld.txt").rename(made.inputs / "aside.txt")
        refused(made.ctx, "is not written", "a person's to write")
        assert lc.read_job(made.job_file, made.root, need_text=False)["name"] == "ld"
        (made.inputs / "aside.txt").rename(made.inputs / "ld.txt")
        made.load["subject"] = "z"
        made.write()
        refused(made.ctx, "is not a subject of the preview pv")
        made.load["subject"] = "a"
        made.write()
        (made.pv / "preview.queued.json").write_text(json.dumps({"status": "error", "files": []}))
        refused(made.ctx, "did not end in success")
    return "one load, a capture with its day, a preview whose run finished, a subject and frames of that preview, a text file: each lack is refused in words"


def a_masks_at_is_the_previews_first_frame() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        ctx = made.ctx()
        files = lc.preview_files(ctx, made.m.output)
        others, mine = lc.mask_specs(ctx, files)
        assert others.startswith("b:track=") and others.endswith(",shots_at=100,by=pv,at=100") and "parts=" not in others, others
        for part in ("a:track=", "parts=", "classes=", "shots_at=100", "pose_at=100", "by=pv", ",at=100"):
            assert part in mine, f"{part!r} is not in the subject's mask: {mine}"
        # THE FAULT OF 2026-10-13: the load begins on 110, its preview on 100, and 110 is in no mask
        assert "110" not in others + mine, f"the load's own first frame is in a mask: {mine}"
        assert not any(f"{key}=" in mine for key in lc.MASK_OPTIONS), f"an option the job does not set was passed: {mine}"
        made.job["subjects"][1].update(pieces="largest", least=1024)
        made.write()
        mine = lc.mask_specs(made.ctx(), files)[1]
        assert mine.endswith(",by=pv,at=100,pieces=largest,least=1024"), mine
        made.job["subjects"][1]["pose"] = False
        made.write()
        assert "pose" not in lc.mask_specs(made.ctx(), files)[1], "a pose table was given to a subject whose job says pose: false"
        del made.job["subjects"][1]["pose"]
        made.write()
        (made.m.output / "Video/v/s/pv/pv_a_pose_table.json").unlink()
        refused(lambda: lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output)), "does not say `pose: false`", "wrote no pose table")
        (made.m.output / "Video/v/s/pv/pv_a_shots_00001_shots.json").unlink()
        refused(lambda: lc.preview_files(made.ctx(), made.m.output), "0 shot tables for 'a'")
        (made.pv / "preview.queued.json").write_text(json.dumps({"status": "success", "files": made.listed + ["Video/v/s/pv/pv_a_track_00002.mkv"]}))
        refused(lambda: lc.preview_files(made.ctx(), made.m.output), "lists 2 track videos for 'a'")
    return ("a load that begins after its preview takes every mask from the preview's first frame and its own first frame is in none; only the "
            "options the job sets are passed; a pose table only where the job does not say pose: false")


def the_first_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        lc.step_first(made.ctx(), m, args())
        assert m.labels() == ["who is who", "files first", "verify first", "gate first", "strip first"], m.labels()
        who = m.said("who is who")
        assert who[:3] == ["PY", "bench/who_is_who_across_shots.py", "corrections"] and who[who.index("--first-frame") + 1] == "110" and who[who.index("--frames") + 1] == "20", who
        tables = [who[i + 1] for i, x in enumerate(who) if x == "--table"]
        assert [t.split("=")[0] for t in tables] == ["b", "a"] and all(t.endswith("_shots.json@100") for t in tables), tables
        files = m.said("files first")
        assert files[files.index("--folder") + 1].endswith("data/2026-01-03_ld") and files[files.index("--first") + 1] == "110" and "--overwrite" not in files
        assert files[files.index("--loads") + 1] == str(made.job_file) and files[files.index("--made-by") + 1] == "the session" and files.count("--mask") == 2
        assert files[files.index("--voice") + 1].endswith("voice.csv") and m.said("gate first")[m.said("gate first").index("--voice-spans") + 1].endswith("spans.json")
        strip = m.said("strip first")
        assert strip[strip.index("--strip-folder") + 1] == "2026-01-03_ld_first" and strip[strip.index("--motion") + 1] == "/out/ref.mp4" \
            and strip[strip.index("--still") + 1] == f"a={m.inputs / 'a.png'}", strip
        gate = m.said("gate first")
        assert gate[gate.index("--who-is-who") + 1].endswith("2026-01-03_ld_job/ld.who_is_who.json") and gate[gate.index("--text") + 1].endswith("ld.txt")
        # a who-is-who file left in the job folder by an earlier run is NOT picked up: it is made again
        left = made.root / "data/2026-01-03_ld_job/ld.who_is_who.json"
        left.parent.mkdir(parents=True, exist_ok=True)
        left.write_text("{}")
        m.ran.clear()
        lc.step_first(made.ctx(), m, args())
        assert m.labels()[0] == "who is who", "a who-is-who file was picked up because it was there"
        # a who-is-who file the job names is read and none is made; one named and missing is refused
        (made.inputs / "who.json").write_text("{}")
        made.job["who_is_who"] = "who.json"
        made.write()
        m.ran.clear()
        lc.step_first(made.ctx(), m, args())
        assert "who is who" not in m.labels() and m.said("gate first")[m.said("gate first").index("--who-is-who") + 1] == str(made.inputs / "who.json")
        made.job["who_is_who"] = "gone.json"
        made.write()
        refused(lambda: lc.step_first(made.ctx(), m, args()), "names the who-is-who file", "not a file")
        del made.job["who_is_who"]
        made.write()
        m.voiced = False
        refused(lambda: lc.step_first(made.ctx(), m, args()), "no voice tables")
        m.voiced = True
        # the tables cover a stretch of the video: a load with a frame outside it is refused, at either end, and one that just fits is not
        ran = len(m.ran)
        for covered in ((0, 128), (111, 199)):
            m.voice_frames = covered
            refused(lambda: lc.step_first(made.ctx(), m, args()), f"cover source frames {covered[0]}-{covered[1]}", "the load is 110-129")
        assert len(m.ran) == ran, "something ran for a load the voice tables do not cover"
        m.voice_frames = (110, 129)
        assert lc.voice_for(made.ctx(), m)["frames"] == [110, 129]
        m.voice_frames = (0, 199)
        for folder in ("2026-01-03_ld", "2026-01-03_ld_first"):
            (made.root / "data" / folder).mkdir()
            refused(lambda: lc.step_first(made.ctx(), m, args()), "exists", "once")
            (made.root / "data" / folder).rmdir()
    return ("who is who is made again from the preview's tables, each with the frame it begins on, or read from the file the job names; the "
            "capture takes the job file as its plan; the strip is drawn under the capture's _first name; an existing capture is refused")


def the_plan_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "the plan is built from the first capture")
        made.capture(on_plan=False)
        m.busy = (1, 0)
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "the queue is not empty (1 running, 0 pending)")
        m.busy = (0, 0)
        # what the builder refuses stops the step before a command is run, and a dry run the same
        m.builder_says = "load 'ld': the hair band is cut out of the keep-out 'clothes', which is not among the load's `others`"
        for dry in (True, False):
            m.dry = dry
            refused(lambda: lc.step_plan(made.ctx(), m, args()), "the builder refuses this load's plan", "nothing was built or queued", "'clothes'")
        assert m.labels() == [], f"a refused plan ran {m.labels()}"
        m.builder_says, m.dry = "", False
        lc.step_plan(made.ctx(), m, args())
        build, queue = m.said("build plan"), m.said("queue plan")
        assert build[1:4] == ["bench/masked_job.py", "build", "plan"] and build[build.index("--date") + 1] == "2026-01-03", build
        assert queue[2] == "queue" and queue[3].endswith("2026-01-03_ld_job") and queue[4:7] == ["plan", "--load", "ld"] and "--wait" in queue, queue
        made.planned()
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "already built", "never a second one")
    return ("a plan is refused with no first capture, on a queue that is not empty, when one is built, and when the builder's dry build refuses it "
            "(a dry run as well, with no command run); it is built with the capture's day and queued")


def the_on_plan_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        cap = made.capture(on_plan=False)
        folder = made.planned()
        refused(lambda: lc.step_on_plan(made.ctx(), m, args()), "no plan file")
        windows = m.output / "Video/v/s/ld/ld_plan/ld_windows"
        windows.mkdir(parents=True)
        (windows / "ld_plan.json").write_text("{}")
        (folder / "ld.who_is_who.json").write_text("{}")
        m.busy = (0, 2)
        refused(lambda: lc.step_on_plan(made.ctx(), m, args()), "the queue is not empty", "may be reading its masks")
        m.busy = (0, 0)
        lc.step_on_plan(made.ctx(), m, args(), hashed=lambda path: "hash of " + Path(path).name)
        masks = json.loads((folder / "ld.plan.masks.json").read_text())["masks"]
        assert masks == {str(cap / "subjects/a/track__pv.mkv"): "hash of track__pv.mkv"}, f"what the plan read, and not the source copy outside the capture: {masks}"
        first = cap.with_name("2026-01-03_ld_first")
        assert lc.same_tree(cap, first) == [] and (first / "looks.json").is_file(), "the first capture was not copied whole"
        files = m.said("files on plan")
        assert "--overwrite" in files and "--loads" not in files, files
        assert files[files.index("--run") + 1] == f"ld:preview={windows},subject=a,others=b,at=110", files[files.index("--run") + 1]
        assert "--strip-folder" not in m.said("strip") and m.labels() == ["files on plan", "verify", "gate", "strip"], m.labels()
        refused(lambda: lc.step_on_plan(made.ctx(), m, args()), "exists", "already copied")
        # the control for the comparison of the copy: a file that differs and a file that is missing are each said
        (first / "status.json").write_text("{}")
        assert lc.same_tree(cap, first) == ["status.json differs"], lc.same_tree(cap, first)
        (first / "status.json").unlink()
        assert "different files" in lc.same_tree(cap, first)[0]
    return ("the masks the plan read are hashed and recorded, the first capture is copied whole and the copy held to it, and the capture is built "
            "in place on the node's plan; a queue that is not empty, a missing plan and a second run are refused")


def the_look_and_the_gate() -> str:
    assert lc.later("2026-01-03T10:30:00", "2026-01-03T10:00:00") and not lc.later("2026-01-03T09:00:00", "2026-01-03T10:00:00")
    # as times and not as text: an hour later by the clock, earlier as a string
    assert lc.later("2026-01-03T10:30:00+00:00", "2026-01-03T12:00:00+05:00"), "two times with offsets were compared as text"
    refused(lambda: lc.later("2026-01-03T10:30:00+00:00", "2026-01-03T10:00:00"), "cannot be set in one zone")
    assert lc.in_utc("2026-01-03T12:00:00+05:00") == "2026-01-03T07:00:00Z"
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        made.capture()
        look = lc.the_look(made.ctx())
        assert (look["by"], look["capture_made_by"], look["flags"], look["overridden"]) == ("a reader", "the maker", 2, []) and look["at_utc"].endswith("Z"), look
        (made.root / "data/2026-01-03_ld/outcomes.json").write_text(json.dumps({"outcomes": [{"flag": "f003", "overridden": True}, {"flag": "f004"}]}))
        assert lc.the_look(made.ctx())["overridden"] == ["f003"]
        for change, words in ((dict(look_at=None), ("nobody has recorded a look",)), (dict(loads=("another",)), ("not of this load",)),
                              (dict(on_plan=False), ("not the capture on the node's plan",)), (dict(look_at="2026-01-03T09:59:59"), ("not later than the capture",)),
                              (dict(look_by="the maker"), ("who made the capture",)), (dict(verdict="blocked"), ("'blocked', not clear",)),
                              (dict(look_at="2026-01-03T10:30:00+00:00"), ("cannot be set in one zone",))):
            for name in ("looks.json",):
                (made.root / "data/2026-01-03_ld" / name).unlink(missing_ok=True)
            made.capture(**change)
            refused(lambda: lc.the_look(made.ctx()), *words)
    return ("the last look must be of this load, on the capture on the node's plan, later than it as times and not by its maker, and the gate "
            "clear; a time with an offset against one without is refused")


def a_render_against_its_plan() -> str:
    plan = {"1": {"inputs": {"video": "/cap/m.mkv"}}, "9": {"inputs": {"preview": True, "filename_prefix": "p", "seed": 1}}}
    render = {"1": {"inputs": {"video": "/cap/m.mkv"}}, "9": {"inputs": {"preview": False, "filename_prefix": "r", "seed": 1}}}
    clean = {"uncommitted_node_code": []}
    assert lc.render_against_plan(plan, render, clean) == []
    other = json.loads(json.dumps(render))
    other["9"]["inputs"]["seed"] = 2
    assert "differs from the plan graph in ['filename_prefix', 'preview', 'seed']" in lc.render_against_plan(plan, other, clean)[0]
    assert "node code is not committed" in lc.render_against_plan(plan, render, {"uncommitted_node_code": ["video_mask.py"]})[0]
    assert "do not have the same nodes" in lc.render_against_plan(plan, {**render, "10": {"inputs": {}}}, clean)[0]
    same = json.loads(json.dumps(plan))
    assert "may differ in ['filename_prefix', 'preview'] and no more" in lc.render_against_plan(plan, same, clean)[0], "a render graph with the preview still on passed"
    # the band rule: the band's own keep-out file may differ in name when the two decode equal, and not otherwise
    band_plan = {**plan, "5": {"inputs": {"video": "/j/ld.plan.clothes_less_hair_band.mkv"}}}
    band_render = {**render, "5": {"inputs": {"video": "/j/ld.render.clothes_less_hair_band.mkv"}}}
    said = lc.render_against_plan(band_plan, band_render, clean, hashed=lambda path: "one hash")
    assert len(said) == 1 and said[0].startswith("note: THE BAND FILE") and said[0].endswith(": the same."), said
    said = lc.render_against_plan(band_plan, band_render, clean, hashed=lambda path: path)
    assert len(said) == 1 and said[0].endswith(": NOT the same.") and not said[0].startswith("note:"), said
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        cap = made.capture()
        folder = made.planned()
        ctx = made.ctx()
        graph = lc.graph_of(folder / "ld.plan.json")
        track = str(cap / "subjects/a/track__pv.mkv")
        assert lc.mask_videos(graph, cap) == [track]
        assert "the on-plan step" in lc.masks_as_planned(ctx, graph, hashed=lambda p: "h")[0]
        (folder / "ld.plan.masks.json").write_text(json.dumps({"masks": {track: "h"}}))
        assert lc.masks_as_planned(ctx, graph, hashed=lambda p: "h") == []
        assert "no longer decodes equal to what the plan read" in lc.masks_as_planned(ctx, graph, hashed=lambda p: "another")[0]
        more = {**graph, "3": {"class_type": "L", "inputs": {"video": str(cap / "subjects/a/parts.mkv")}}}
        assert "which the plan graph did not" in lc.masks_as_planned(ctx, more, hashed=lambda p: "h")[0]
        # the job file is the one the plan was built from, byte for byte
        lc.job_as_planned(ctx)
        made.job["note"] = "a go written into the job file after the plan"
        made.write()
        refused(lambda: lc.job_as_planned(made.ctx()), "is not the file its plan was built from", "not edited after its plan is built")
    return ("a render graph is its plan graph with the preview and the prefix changed and no more; the band file may differ when it decodes equal; "
            "a mask that no longer decodes equal to what the plan read, and a job file edited since its plan, are each said")


def the_render_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        cap = made.capture()
        folder = made.planned()
        track = str(cap / "subjects/a/track__pv.mkv")
        (folder / "ld.plan.masks.json").write_text(json.dumps({"masks": {track: "h"}}))
        for name, words in (("grounds.txt", "the lead relayed the owner's go at a time by its clock"), ("show.txt", "her face on these frames")):
            (made.root / name).write_text(words + "\n")
        a = args(grounds=str(made.root / "grounds.txt"), show=str(made.root / "show.txt"))
        refused(lambda: lc.step_render(made.ctx(), m, args(grounds=str(made.root / "grounds.txt"))), "--grounds and --show")
        m.busy = (1, 1)
        refused(lambda: lc.step_render(made.ctx(), m, a), "the queue is not empty")
        m.busy = (0, 0)
        before = made.job_file.read_bytes()

        def built(extra: dict | None = None, node_code=()):
            """What `build render` leaves: the graph, its build record and the tool's printed output."""
            def write():
                graph = json.loads((folder / "ld.plan.json").read_text())
                graph["9"]["inputs"].update(preview=False, filename_prefix="ld", **(extra or {}))
                (folder / "ld.render.json").write_text(json.dumps(graph))
                (folder / "ld.render.built.json").write_text(json.dumps({"uncommitted_node_code": list(node_code)}))
                (folder / "chain").mkdir(exist_ok=True)
                (folder / "chain" / "ld.build_render.out").write_text("what build printed\n")
            return write

        m.on["build render"] = built()
        lc.step_render(made.ctx(), m, a, hashed=lambda p: "h")
        assert m.labels() == ["build render", "queue render"], m.labels()
        go = (made.inputs / "ld.render.go.md").read_text()
        for words in ("Written by the session BEFORE the render was queued", "the lead relayed the owner's go", "a reader, 2026-01-03T", "by the maker",
                      "prompt plan-prompt", "gate clear, 2 flags, no override", "her face on these frames", "a card line."):
            assert words in go, f"the go lacks {words!r}:\n{go}"
        assert "five hours" not in go and "holds the card" not in go and "The lead's look" not in go, "the go carries a sentence of the scripts that a file does not hold"
        assert made.job_file.read_bytes() == before, "the job file was edited by the render step"
        record = json.loads((folder / "ld.render.chain.json").read_text())
        assert record["go_file"].endswith("ld.render.go.md") and record["look"]["by"] == "a reader" and (folder / "ld.render.build_printed.txt").read_text() == "what build printed\n"
        refused(lambda: lc.step_render(made.ctx(), m, a), "already built or already has a go")
        # each of these stops it AFTER the build and BEFORE anything is queued
        for change, hashed, words in ((dict(extra={"seed": 5}), "h", "differs from the plan graph in"), (dict(node_code=("video_mask.py",)), "h", "node code is not committed"),
                                      (dict(), "another", "no longer decodes equal to what the plan read")):
            for path in (folder / "ld.render.json", made.inputs / "ld.render.go.md"):
                path.unlink()
            m.ran.clear()
            m.on["build render"] = built(**change)
            refused(lambda: lc.step_render(made.ctx(), m, a, hashed=lambda p, h=hashed: h), "not the plan graph with the preview off", words)
            assert m.labels() == ["build render"], f"something was queued after a refusal: {m.labels()}"
        # a job file edited after its plan is refused before anything is written
        for path in (folder / "ld.render.json", made.inputs / "ld.render.go.md"):
            path.unlink()
        made.job["note"] = "edited"
        made.write()
        m.ran.clear()
        refused(lambda: lc.step_render(made.ctx(), m, a), "is not the file its plan was built from")
        assert m.ran == [] and not (made.inputs / "ld.render.go.md").exists()
    return ("the go is written beside the job file and names whoever looked, in UTC; the job file is untouched; a graph that differs in more, node "
            "code not committed and a mask that changed each stop it after the build and before the queue; a job file edited since its plan before anything")


def the_land_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        cap = made.capture()
        folder = made.planned()
        (folder / "ld.render.queued.json").write_text(json.dumps({"prompt_id": "render-prompt"}))
        kept, lc.video_of = lc.video_of, lambda ctx, data: "v"
        env = os.environ.pop(lc.MODEL_STORE, None)
        try:
            refused(lambda: lc.step_land(made.ctx(), m, args()), lc.MODEL_STORE, "--model-store")
            store = made.root / "store"
            (store / "pose-model").mkdir(parents=True)
            a = args(model_store=str(store))
            m.busy = (1, 0)
            refused(lambda: lc.step_land(made.ctx(), m, a), "no pose pass beside a render")
            m.busy = (0, 0)
            refused(lambda: lc.step_land(made.ctx(), m, a), "no window file")
            windows = m.output / "Video/v/s/ld/ld/ld_windows"
            windows.mkdir(parents=True)
            (windows / "ld_window_1.mp4").write_bytes(b"")
            # a mask review and a zoomed picture beside a window are not windows
            (windows / "ld_window_1_with_mask.mp4").write_bytes(b"")
            (windows / "ld_window_1_zoomed.mp4").write_bytes(b"")
            # A LOAD OF TWO WINDOWS: each window's own file holds only the frames that window wrote, so the pose pass
            # reads the joined file, which is held to the load's frame count first
            (windows / "ld_window_2.mp4").write_bytes(b"")
            joined = m.output / "Video/v/s/ld/ld/ld_00001.mp4"
            refused(lambda: lc.step_land(made.ctx(), m, a), "2 windows", "the joined file", "is not there")
            joined.write_bytes(b"")
            m.frames = {"ld_00001.mp4": 19}
            refused(lambda: lc.step_land(made.ctx(), m, a), "the load is 20 frames in 2 windows", "holds 19")
            assert "pose pass" not in m.labels(), "a pose pass ran on a joined file that is not the load's frames"
            (windows / "ld_window_3.mp4").write_bytes(b"")
            (windows / "ld_window_2.mp4").unlink()
            refused(lambda: lc.step_land(made.ctx(), m, a), "windows [1, 3]", "did not finish")
            (windows / "ld_window_3.mp4").unlink()
            (windows / "ld_window_2.mp4").write_bytes(b"")
            m.frames = {"ld_00001.mp4": 20}
            assert lc.rendered_video(made.ctx(), m, joined.parent) == joined
            (windows / "ld_window_2.mp4").unlink()
            # one window: its own video, as ever, and the joined file's frames are not asked for
            m.frames = {}
            assert lc.rendered_video(made.ctx(), m, joined.parent) == windows / "ld_window_1.mp4"
            lc.step_land(made.ctx(), m, a)
            pose, stack = m.said("pose pass"), m.said("stack")
            assert pose[pose.index("--video") + 1] == str(windows / "ld_window_1.mp4"), pose
            assert pose[pose.index("--box-mask") + 1] == str(cap / "subjects/a/track__pv.mkv") and pose[pose.index("--box-mask-first") + 1] == "110", pose
            assert pose[pose.index("--model-folder") + 1] == str(store / "pose-model") and pose[pose.index("--count") + 1] == "20" \
                and pose[pose.index("--source-first") + 1] == "110" and pose[pose.index("--canvas") + 1] == "64x48" and pose[pose.index("--by") + 1] == "the session", pose
            clip = m.output / "Video/v/review_clips/ld_0110_0129_stacked.mp4"
            assert stack[stack.index("--out") + 1] == str(clip) and stack[stack.index("--render") + 1].endswith("Video/v/s/ld/ld/ld_00001.mp4"), stack
            os.environ[lc.MODEL_STORE] = str(store)
            m.ran.clear()
            lc.step_land(made.ctx(), m, args())
            assert m.said("pose pass")[m.said("pose pass").index("--model-folder") + 1] == str(store / "pose-model"), "the environment's store was not read"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"")
            refused(lambda: lc.step_land(made.ctx(), m, a), "this load is landed")
        finally:
            lc.video_of = kept
            os.environ.pop(lc.MODEL_STORE, None)
            if env is not None:
                os.environ[lc.MODEL_STORE] = env
    return ("landing is refused without the pose model's store, on a queue that is not empty, with no window, with a second one, and when the clip "
            "exists; the pose pass takes the capture's own track video and first frame, and the store from the flag or the environment")


def the_page_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        made.capture()
        kept_video, kept_repo = lc.video_of, lc.REPO
        lc.video_of, lc.REPO = (lambda ctx, data: "v"), made.root
        try:
            items = made.root / "data/v/review/items.json"
            (made.root / "data/v/review/media").mkdir(parents=True)
            items.write_text(json.dumps({"updated": "before", "by": "somebody", "items": [{"id": "an_older_item", "options": ["good to go"]}]}, indent=1))
            words = {"title_rest": "the lead, face only, one shot", "state_words": "not judged by anyone", "what": "one shot", "look": "her eyes on 115",
                     "prompt_note": "written from these frames"}
            spec = made.root / "spec.json"
            spec.write_text(json.dumps(words))
            a = args(spec=str(spec))
            refused(lambda: lc.step_page(made.ctx(), m, args()), "--spec is a file")
            refused(lambda: lc.step_page(made.ctx(), m, a), "is not there: the load is not landed")
            clip = m.output / "Video/v/review_clips/ld_0110_0129_stacked.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"")
            for lack in ("look", "title_rest"):
                spec.write_text(json.dumps({k: v for k, v in words.items() if k != lack}))
                refused(lambda: lc.step_page(made.ctx(), m, a), f"lack {lack}", "a person's to write")
            spec.write_text(json.dumps({**words, "title_rest": "ld: again"}))
            refused(lambda: lc.step_page(made.ctx(), m, a), "begins with the load's name")
            spec.write_text(json.dumps(words))
            lc.step_page(made.ctx(), m, a)
            listed = json.loads(items.read_text())
            item = listed["items"][0]
            assert [x["id"] for x in listed["items"]] == ["ld", "an_older_item"] and listed["by"] == "the session" and listed["updated"].endswith("Z"), listed
            assert item["options"] == list(delivery_table.RENDER_ITEM_BUTTONS) and item["options"][0] in delivery_table.KEEP \
                and item["options"][1] in delivery_table.NOT_THE_ONE, f"the page's buttons are not the words the delivery table reads: {item['options']}"
            assert (item["title"], item["first"], item["frames"], item["think"], item["prompt"]) == ("ld: the lead, face only, one shot", "110", "20", "", "the words of the load"), item
            assert os.path.islink(made.root / "data/v/review/media" / clip.name) and item["path"] == f"data/v/review/media/{clip.name}"
            before = json.loads((made.root / "data/2026-01-03_ld_job/items.before_the session_ld.json").read_text())
            assert [x["id"] for x in before["items"]] == ["an_older_item"], "the list was not copied before it was written"
            refused(lambda: lc.step_page(made.ctx(), m, a), "already has an item ld")
        finally:
            lc.video_of, lc.REPO = kept_video, kept_repo
    return ("a render's item goes to the top of the list with the delivery table's own buttons, the list copied into the job folder first; it is "
            "refused without a person's words, with no clip, and for an item that is there")


case("the job file and the preview it names", the_job_file_and_its_preview)
case("a mask's at is the preview's first frame", a_masks_at_is_the_previews_first_frame)
case("first: who is who, the first capture, its strip under _first", the_first_step)
case("plan: built and queued on an empty queue, once", the_plan_step)
case("on-plan: the masks hashed, the capture copied, then built in place", the_on_plan_step)
case("the look and the gate a render rests on", the_look_and_the_gate)
case("a render graph against its plan graph, its masks and its job file", a_render_against_its_plan)
case("render: the go beside the job file, and what stops it", the_render_step)
case("land: the pose pass and the stacked clip", the_land_step)
case("page: the item, with the table's buttons", the_page_step)

if __name__ == "__main__":
    sys.exit(finish())
