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
2b. **A load's keep-outs** (`job.keep_outs`, 2026-10-13: the second subject's loads carried none and a render drew
   her apron again). Each is a `--mask` of its own on the preview's first frame, at `first` and at `on-plan`, and
   one of the run's others on the plan. Refused: a keep-out the job names that the load's `others` leaves out; one
   the load keeps out with no file named; a named file that is not there.
3. **first**: refused when the capture or its `_first` copy exists; who is who made again from the preview's
   tables, each with the frame it begins on, or the file the job names and no command; the capture with the load
   as a plan; the strip under `<capture>_first`. With verify failed and the gate blocked every command still
   runs, and the step's last line says so in capitals and its exit status is the capture tool's own for the
   first of them (the first real load, 2026-10-13: both were not 0 and the step exited 0).
4. **plan**: refused on a queue that is not empty, with no capture, with a plan graph already built, or on a first
   capture whose gate is blocked or that has no flags file; and, in a
   dry run as well and before any command, when the builder's own dry build refuses the load (the first real
   load, 2026-10-13: `plan --dry` exited 0 and the plan was then refused on a key the job file held all along).
   That the dry build holds what a build holds and writes nothing is `check_masked_job.py`'s, on the real builder.
4b. **A kept output is not replaced** (the real `World.run`): a command run again under one name leaves the earlier
   output beside the new one, each whole. And `STEP --help` is that step's own paragraph and usage line.
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
import re
import shutil
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
        self.exits = {}                  # a command's exit status by its label; 0 when not named

    def builder_refuses(self, ctx, stage):
        return self.builder_says

    def queue(self):
        return self.busy

    def card(self):
        return "a card line."

    def free(self):
        self.ran.append(("free the card", []))
        return "the card after it: free."

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
        return self.exits.get(label, 0)

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
                look_by: str = "a reader", loads=("ld",), verdict: str = "clear", blocking=(), flags=None) -> Path:
        """A capture folder as the capture tool leaves one, with a look recorded on it unless told otherwise."""
        cap = self.root / "data" / "2026-01-03_ld"
        (cap / "subjects" / "a").mkdir(parents=True, exist_ok=True)
        (cap / "subjects" / "a" / "track__pv.mkv").write_bytes(b"track")
        (cap / "manifest.json").write_text(json.dumps({"first_frame": 110, "frames": 20, "runs": [{"carried_from_the_node": on_plan}],
                                                       "subjects": [{"label": "a", "sightings": [{"by": "pv"}]}]}))
        (cap / "status.json").write_text(json.dumps({"written": written, "made_by": "the maker"}))
        (cap / "flags.json").write_text(json.dumps({"verdict": verdict, "blocking": list(blocking), "flags": [{}, {}] if flags is None else flags}))
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
    given = dict(by="the session", grounds=None, show=None, model_store="", spec=None, not_first=None)
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
        # a load the job renders on the whole frame: its capture is told so, and its starting floor is not held to a zoom (2026-10-14)
        made.job["whole_frame"] = {"why": "A REASON", "said_by": "x", "at": "t"}
        made.write()
        assert lc.mask_specs(made.ctx(), files)[1].endswith(",pieces=largest,least=1024,zoom=off"), lc.mask_specs(made.ctx(), files)[1]
        del made.job["whole_frame"]
        made.write()
        assert "zoom=" not in lc.mask_specs(made.ctx(), files)[1]
        # A LOAD'S OWN PART (`parts_file` on its subject): the capture is handed that file as the part, and no other subject's mask changes;
        # a file that is not there is refused; a job that names none is handed the preview's saved part, as ever
        usual = lc.mask_specs(made.ctx(), files)
        own = Path(tmp) / "pv_a_part_hair.mkv"
        made.job["subjects"][1]["parts_file"] = str(own)
        made.write()
        refused(lambda: lc.mask_specs(made.ctx(), files), "names a part of the load's own for 'a'", "is not there")
        own.write_bytes(b"")
        others_, mine = lc.mask_specs(made.ctx(), files)
        # and it names the preview's saved part it was made from, which the capture's verify holds it to (2026-10-14)
        assert f",parts={own},parts_from={files['a']['parts']}," in mine and others_ == usual[0] \
            and mine == usual[1].replace(f"parts={files['a']['parts']}", f"parts={own},parts_from={files['a']['parts']}"), mine
        assert f"parts={files['a']['parts']}" in usual[1] and str(own) not in usual[1], usual[1]
        del made.job["subjects"][1]["parts_file"]
        made.write()
        assert lc.mask_specs(made.ctx(), files) == usual, "a job with no part of its own was handed another part"
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


def a_loads_keep_outs() -> str:
    """The second subject's loads through the chain carried no clothing, hands or held-thing keep-out, and the first
    render drew her apron again (2026-10-13). The job names each with its file; the chain hands each to the capture."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        keeps = made.m.output / "Video/v/keep_outs/pv"
        keeps.mkdir(parents=True)
        kept = [{"name": "held", "track": str(made.m.output / "Video/v/s/pv/pv_a_held_things_00001.mkv")},
                {"name": "hands", "track": str(keeps / "pv_a_hands.mkv"), "classes": ["Left_Hand"]},
                {"name": "clothes", "track": str(keeps / "pv_a_clothes.mkv"), "classes": ["Apparel"]}]
        for k in kept:
            Path(k["track"]).write_bytes(b"")
        made.job["keep_outs"] = {"of": "a", "by": "the plan (a person)", "preview": "pv", "first": 100, "made": kept}
        made.load["others"] = ["b", "held", "hands", "clothes"]
        made.write()
        ctx = made.ctx()
        specs = lc.mask_specs(ctx, lc.preview_files(ctx, made.m.output))
        assert specs[2:] == [f"{k['name']}:track={k['track']},by=pv,at=100" for k in kept], specs[2:]
        assert lc.others_of(ctx) == ["b", "held", "hands", "clothes"], lc.others_of(ctx)
        # the first capture is handed five masks, and the capture on the plan names all four as the run's others
        lc.step_first(ctx, made.m, args())
        files = made.m.said("files first")
        assert files.count("--mask") == 5 and f"clothes:track={kept[2]['track']},by=pv,at=100" in files, files
        made.capture(on_plan=False)
        made.planned()
        windows = lc.plan_windows(ctx, made.m.output)
        windows.mkdir(parents=True)
        (windows / "ld_plan.json").write_text("{}")
        (made.root / "data/2026-01-03_ld_job/ld.who_is_who.json").write_text("{}")
        made.m.ran.clear()
        lc.step_on_plan(made.ctx(), made.m, args(), hashed=lambda path: "h")
        on_plan = made.m.said("files on plan")
        assert on_plan[on_plan.index("--run") + 1] == f"ld:preview={windows},subject=a,others=b+held+hands+clothes,at=110", on_plan[on_plan.index("--run") + 1]
        assert on_plan.count("--mask") == 5, on_plan
        # A LOAD WRITTEN WITHOUT A KEEP-OUT ITS JOB NAMES is refused: it would be drawn again in silence
        made.load["others"] = ["b", "held", "hands"]
        made.write()
        refused(made.ctx, "the job names the keep-out 'clothes' and the load's `others` does not", "drawn again")
        # one the load keeps out with no file named cannot be handed to the capture
        made.load["others"] = ["b", "held", "hands", "clothes"]
        made.job["keep_outs"]["made"] = kept[:2]
        made.write()
        refused(made.ctx, "the load keeps out ['clothes']", "`job.keep_outs`")
        made.job["keep_outs"]["made"] = kept
        made.write()
        Path(kept[1]["track"]).unlink()
        refused(lambda: lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output)), "the keep-out 'hands' is named at", "not a file")
        # THE STRANGERS FILE reaches the capture only as the frames the job step held to its record (2026-10-14, a second reader)
        Path(kept[1]["track"]).write_bytes(b"")
        strangers = {"name": "strangers", "track": str(keeps / "pv_strangers.mkv")}
        Path(strangers["track"]).write_bytes(b"")
        made.job["keep_outs"]["made"] = kept + [strangers]
        made.load["others"] = ["b", "held", "hands", "clothes", "strangers"]
        made.write()
        refused(lambda: lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output), hashed=lambda p: "h"), "with no hash of its frames", "made again")
        strangers["frame_hash"] = "h"
        made.write()
        refused(lambda: lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output), hashed=lambda p: "other"), "not the file the job step held", "changed since")
        assert lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output), hashed=lambda p: "h")[-1] == f"strangers:track={strangers['track']},by=pv,at=100"
    return ("a load's keep-outs are each a mask of the capture on the preview's frames and one of the run's others, at first and on the plan; a keep-out the job "
            "names and the load does not keep out, one kept out with no file, and a file that is not there are refused; the strangers file reaches the "
            "capture only with the hash the job step held it to")



def a_banded_keep_out_on_the_plan() -> str:
    """second_hfp_0484 (2026-10-14): the capture on the plan was handed the clothes keep-out whole while the plan graph
    read it with the hair band cut out, and the gate's region over another subject counted the band's own cells
    (0.403 of the whole mask at most; 0.000 of the cut one)."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        keeps = made.m.output / "Video/v/keep_outs/pv"
        keeps.mkdir(parents=True)
        kept = [{"name": "clothes", "track": str(keeps / "pv_a_clothes.mkv"), "classes": ["Apparel"]}]
        Path(kept[0]["track"]).write_bytes(b"")
        made.job["keep_outs"] = {"of": "a", "by": "the plan (a person)", "preview": "pv", "first": 100, "made": kept}
        made.load["others"] = ["b", "clothes"]
        made.load["hair_band"] = 0.6
        made.write()
        ctx = made.ctx()
        # the first capture, before any plan, is handed the keep-out whole, from the preview's first frame
        lc.step_first(ctx, made.m, args())
        assert f"clothes:track={kept[0]['track']},by=pv,at=100" in made.m.said("files first"), made.m.said("files first")
        made.capture(on_plan=False)
        made.planned()
        windows = lc.plan_windows(ctx, made.m.output)
        windows.mkdir(parents=True)
        (windows / "ld_plan.json").write_text("{}")
        (made.root / "data/2026-01-03_ld_job/ld.who_is_who.json").write_text("{}")
        made.m.ran.clear()
        refused(lambda: lc.step_on_plan(made.ctx(), made.m, args(), hashed=lambda path: "h"), "has a hair band", "the plan step writes it")
        band = made.root / "data/2026-01-03_ld_job/ld.plan.clothes_less_hair_band.mkv"
        band.write_bytes(b"")
        lc.step_on_plan(made.ctx(), made.m, args(), hashed=lambda path: "h")
        on_plan = made.m.said("files on plan")
        assert f"clothes:track={band},by=pv,at=110" in on_plan and kept[0]["track"] not in " ".join(map(str, on_plan)), on_plan
        # a load with no band is handed its keep-out whole on the plan as well
        del made.load["hair_band"]
        made.write()
        assert lc.plan_band_file(made.ctx()) is None
        assert lc.mask_specs(made.ctx(), lc.preview_files(made.ctx(), made.m.output), None)[-1] == f"clothes:track={kept[0]['track']},by=pv,at=100"
        # the band file holds the load's frames only: a load shorter than its capture is refused
        made.load["hair_band"] = 0.6
        made.load["frames"] = 19
        made.write()
        refused(lambda: lc.plan_band_file(made.ctx()), "holds the load's frames only")
    return ("on the plan, the keep-out a load's hair band cuts reaches the capture as the plan's own cut file from the load's first frame; at first, "
            "and with no band, it is handed whole; a band file that is not there, and a load shorter than its capture, are refused")


def a_track_that_is_not_its_person() -> str:
    """2026-10-13 (second_hfp_0998b): the lead's preview track lay on the second subject on 1003-1009 after the lead had
    left. The job names a copy with those frames emptied (`shot_to_load.py job --track-not`); every step reads it."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        copy = made.m.output / "Video/v/tracks/pv/pv_b_track_not_115-120.mkv"
        made.job["track_not"] = {"by": "a person", "preview": "pv", "made": [{"subject": "b", "frames": [[115, 120]], "track": str(copy)}]}
        made.write()
        refused(lambda: lc.preview_files(made.ctx(), made.m.output), "track is not them on 115-120", "is not there")
        copy.parent.mkdir(parents=True)
        copy.write_bytes(b"")
        files = lc.preview_files(made.ctx(), made.m.output)
        # THE CASE THAT GOES RED when the chain hands the capture the preview's own track of b
        assert files["b"]["track"] == copy and files["b"]["track_is_not_them_on"] == [[115, 120]] and files["a"]["track"] != copy, files
        assert any(spec.startswith(f"b:track={copy},") for spec in lc.mask_specs(made.ctx(), files)), lc.mask_specs(made.ctx(), files)
        for who in ("a", "z"):
            made.job["track_not"]["made"][0]["subject"] = who
            made.write()
            refused(made.ctx, f"`track_not` names {who!r}", "not another subject of the preview")
    return "a track a person said is not its person on named frames reaches every mask the capture is handed as its copy; a missing copy, the load's own subject and a stranger are refused"

def the_first_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        ended = lc.step_first(made.ctx(), m, args())
        assert m.labels() == ["who is who", "files first", "verify first", "gate first", "strip first"], m.labels()
        assert ended == (0, "") and lc.last_line("first", "ld", ended, "NOW") == (0, "first of ld ended NOW"), ended
        who = m.said("who is who")
        assert who[:3] == ["PY", "bench/who_is_who_across_shots.py", "corrections"] and who[who.index("--first-frame") + 1] == "110" and who[who.index("--frames") + 1] == "20", who
        tables = [who[i + 1] for i, x in enumerate(who) if x == "--table"]
        assert [t.split("=")[0] for t in tables] == ["b", "a"] and all(t.endswith("_shots.json@100") for t in tables), tables
        files = m.said("files first")
        assert files[files.index("--folder") + 1].endswith("data/2026-01-03_ld") and files[files.index("--first") + 1] == "110" and "--overwrite" not in files
        assert files[files.index("--loads") + 1] == str(made.job_file) and files[files.index("--made-by") + 1] == "the session" and files.count("--mask") == 2
        assert files[files.index("--voice") + 1].endswith("voice.csv") and m.said("gate first")[m.said("gate first").index("--voice-spans") + 1].endswith("spans.json")
        strip = m.said("strip first")
        # the references and stills are read from the load's graph, never named from the job file (2026-10-15)
        assert strip[strip.index("--strip-folder") + 1] == "2026-01-03_ld_first" and strip[strip.index("--given") + 1].endswith("2026-01-03_ld_job") \
            and "--one-a-second" in strip and "--motion" not in strip and "--still" not in strip, strip
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
        # verify failed and the gate blocked (the first real load, 2026-10-13): every command still runs, and the step's
        # end says so in capitals and leaves the capture tool's own exit status, never 0
        source = (lc.HERE / "capture_masked_run.py").read_text()
        theirs = {name: int(re.search(rf"^{name} = (\d+)$", source, re.M).group(1)) for name in ("VERIFY_FAILED", "GATE_BLOCKED")}
        assert theirs == {"VERIFY_FAILED": lc.VERIFY_FAILED, "GATE_BLOCKED": lc.GATE_BLOCKED}, theirs
        m.ran.clear()
        m.exits = {"verify first": lc.VERIFY_FAILED, "gate first": lc.GATE_BLOCKED}
        m.on["gate first"] = lambda: made.capture(on_plan=False, verdict="blocked", blocking=["f001", "f002"])
        ended = lc.step_first(made.ctx(), m, args())
        assert m.labels()[-3:] == ["verify first", "gate first", "strip first"], m.labels()
        code, line = lc.last_line("first", "ld", ended, "NOW")
        assert code == lc.VERIFY_FAILED and "ENDED NOW AND IS NOT GOOD" in line and "VERIFY FAILED (exit 4)" in line \
            and "THE GATE IS BLOCKED on f001, f002" in line and "EXIT 4" in line and "`plan` refuses" in line, (code, line)
        assert lc.ended_with(made.root / "data/2026-01-03_ld", 0, lc.GATE_BLOCKED, 0) == (lc.GATE_BLOCKED, "THE GATE IS BLOCKED on f001, f002")
        assert lc.ended_with(made.root / "nowhere", 0, 1, 2) == (1, "THE GATE DID NOT RUN TO ITS END (exit 1); THE STRIP WAS NOT DRAWN (exit 2)")
        shutil.rmtree(made.root / "data/2026-01-03_ld")
        m.exits, m.on = {}, {}
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


def a_kept_output_is_not_replaced_and_a_step_has_its_own_help() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        class Real(Machine):                 # the real `run`, which keeps each command's output, on a machine with no server
            def run(self, ctx, label, argv, stop=True):
                return lc.World.run(self, ctx, label, argv, stop)

        world = Real(root)
        ctx = {"logs": root / "chain", "name": "pv", "job_folder": root / "job"}
        for word in ("one", "two", "three"):
            world.run(ctx, "build preview", [sys.executable, "-c", f"print('{word}')"])
        kept = sorted(p.name for p in (root / "chain").iterdir())
        assert len(kept) == 3 and "pv.build_preview.out" in kept and all(".before_" in k for k in kept if k != "pv.build_preview.out"), kept
        assert (root / "chain" / "pv.build_preview.out").read_text() == "three\n"
        assert sorted(p.read_text() for p in (root / "chain").glob("*.before_*.out")) == ["one\n", "two\n"], kept
        assert not (root / "job").exists(), "a step that names where its output is kept made the job folder"
    for step in lc.STEPS:
        said = lc.step_help(step)
        assert f"bench/load_chain.py {step} " in said and f"- `{step}`:" in said, f"{step}: {said[:200]}"
        assert not any(f"- `{other}`:" in said for other in lc.STEPS if other != step), f"{step}'s help holds another step's words"
    assert "Refused when the queue is not empty" in lc.step_help("plan") and "pose pass" in lc.step_help("land") and "pose pass" not in lc.step_help("plan")
    return ("a command run again under one name keeps the earlier output beside the new one, each whole; a step's own help is its own paragraph and "
            "usage line and no other step's")


def the_plan_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "the plan is built from the first capture")
        made.capture(on_plan=False)
        m.busy = (1, 0)
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "the queue is not empty (1 running, 0 pending)")
        m.busy = (0, 0)
        # a first capture whose gate is blocked is not planned on; a person's override makes the gate clear, and then it is
        made.capture(on_plan=False, verdict="blocked", blocking=["f001", "f002"])
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "is 'blocked', on f001, f002", "a plan is not built on a first capture whose gate is blocked",
                "outcome", "overridden --by WHO --note WHY")
        (made.root / "data/2026-01-03_ld/flags.json").unlink()
        refused(lambda: lc.step_plan(made.ctx(), m, args()), "holds no flags.json")
        made.capture(on_plan=False)
        assert m.labels() == [], f"a refused plan ran {m.labels()}"
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
    return ("a plan is refused with no first capture, on a queue that is not empty, when one is built, on a first capture whose gate is blocked or has "
            "no flags file, and when the builder's dry build refuses it (a dry run as well, with no command run); it is built with the capture's day and queued")


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


def a_render_on_the_gate_alone() -> str:
    """The owner's go-fast word: an ordinary load renders on the gate alone. The maker's record that no look is owed
    stands in a look's place only while the gate is clear and no forcing rule stands, and it is never a look."""
    quiet = [{"id": "f001", "rule": "hand_not_refined", "level": "iffy", "figures": {"frames": 9}},
             {"id": "f002", "rule": "part_not_visible", "level": "iffy", "figures": {"frames": 2}},                                    # two frames: not over two
             {"id": "f003", "rule": "region_mostly_not_the_subject", "level": "iffy", "figures": {"median_share_not_subject": 0.5}},    # a half: not over a half
             {"id": "f004", "rule": "segment_inside_region", "level": "iffy", "segment": "a.Torso"},                                   # the load's own subject
             {"id": "f005", "rule": "segment_inside_region", "level": "iffy", "segment": "b.Hair"},                                    # another's, not clothing or hands
             {"id": "f006", "rule": "segment_inside_region", "level": "iffy", "segment": "a.Left_Hand"}]                               # a hand, and the load's own subject's
    forcing = {"part_in_pieces": ({"id": "f9", "rule": "part_in_pieces"}, "f9 part_in_pieces"),
               "part_changes_size": ({"id": "f9", "rule": "part_changes_size"}, "f9 part_changes_size"),
               "part_off_the_subject": ({"id": "f9", "rule": "part_off_the_subject"}, "f9 part_off_the_subject"),
               "part_moves_on_the_subject": ({"id": "f9", "rule": "part_moves_on_the_subject"}, "f9 part_moves_on_the_subject"),
               "three frames": ({"id": "f9", "rule": "part_not_visible", "figures": {"frames": 3}}, "on more than two frames"),
               "no figure": ({"id": "f9", "rule": "part_not_visible"}, "held as standing"),
               "over a half": ({"id": "f9", "rule": "region_mostly_not_the_subject", "figures": {"median_share_not_subject": 0.51}}, "over a half"),
               "another's clothing": ({"id": "f9", "rule": "segment_inside_region", "segment": "b.Upper_Clothing"}, "another subject's clothing or hands"),
               "another's hand": ({"id": "f9", "rule": "segment_inside_region", "segment": "b.Left_Hand"}, "another subject's clothing or hands")}
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        cap = made.capture(look_at=None, flags=quiet)
        words = made.root / "not_first.txt"
        words.write_text("the fourth oval inset;\n it follows lead_fp_1920c, whose plan and render were read blind")
        a = args(not_first=str(words))
        # no look and no record: refused as ever, and told of the other way
        refused(lambda: lc.the_look(made.ctx()), "nobody has recorded a look", "load_chain.py no-look")
        refused(lambda: lc.step_no_look(made.ctx(), m, args()), "--not-first", "not the first of its kind")
        assert not (cap / lc.NO_LOOK).exists()
        # EVERY FORCING RULE REFUSES THE RECORD, with the flag named; the quiet flags beside it do not
        for name, (flag, said) in forcing.items():
            made.capture(look_at=None, flags=quiet + [flag])
            refused(lambda: lc.step_no_look(made.ctx(), m, a), "A FORCING FLAG STANDS", said, "is owed")
            assert not (cap / lc.NO_LOOK).exists(), f"{name}: a record that no look is owed was written over a forcing flag"
        made.capture(look_at=None, flags=quiet, verdict="blocked")
        refused(lambda: lc.step_no_look(made.ctx(), m, a), "'blocked', not clear")
        made.capture(look_at=None, flags=quiet, on_plan=False)
        refused(lambda: lc.step_no_look(made.ctx(), m, a), "not the capture on the node's plan")
        # A CLEAN LOAD: the record is written, it is not a look, and the render rests on the gate alone
        made.capture(look_at=None, flags=quiet)
        m.dry = True
        lc.step_no_look(made.ctx(), m, a)
        assert not (cap / lc.NO_LOOK).exists(), "a dry run wrote the record"
        m.dry = False
        lc.step_no_look(made.ctx(), m, a)
        kept = json.loads((cap / lc.NO_LOOK).read_text())
        rec = kept["records"][0]
        assert "NOT looks" in kept["what"] and not (cap / "looks.json").exists(), "the record was written as a look"
        assert (rec["load"], rec["by"], rec["gate"], rec["rests_on"]) == ("ld", "the session", "clear", lc.FORCING_WORD) and len(rec["flags_read"]) == 6 \
            and rec["not_the_first_of_its_kind"] == "the fourth oval inset; it follows lead_fp_1920c, whose plan and render were read blind" \
            and rec["of_the_list_and_not_read"] == list(lc.FORCING_NOT_READ) and "f002 part_not_visible (iffy)" in rec["flags_read"], rec
        look = lc.the_look(made.ctx())
        assert look["gate_alone"] is True and "rendered on the gate alone" in look["by"] and look["capture_made_by"] == "the maker" and look["flags"] == 6 \
            and look["rests_on"] == lc.FORCING_WORD, look
        text = lc.go_text(made.ctx(), "the session", "NOW", "the lead's relay", "the face", look, "p1", "a card line.")
        assert "**NO LOOK: RENDERED ON THE GATE ALONE.**" in text and "**The look**" not in text and "the fourth oval inset" in text, text
        # THE RECORD NEVER SATISFIES A LOAD THAT NEEDS A LOOK: a forcing flag on the capture as it stands now refuses the render with the record there
        (cap / "flags.json").write_text(json.dumps({"verdict": "clear", "blocking": [], "flags": quiet + [forcing["part_in_pieces"][0]]}))
        refused(lambda: lc.the_look(made.ctx()), "A FORCING FLAG STANDS", "f9 part_in_pieces", "does not hold against it", "nobody has recorded a look")
        (cap / "flags.json").write_text(json.dumps({"verdict": "blocked", "blocking": ["f001"], "flags": quiet}))
        refused(lambda: lc.the_look(made.ctx()), "'blocked', not clear")
        (cap / "flags.json").write_text(json.dumps({"verdict": "clear", "blocking": [], "flags": quiet}))
        # a head reference whose record does not say it passed is a forcing rule; one that says so is not
        record = made.root / "reference.json"
        record.write_text(json.dumps({"passed": False}))
        made.job["reference"] = {"record": str(record)}
        made.write()
        refused(lambda: lc.the_look(made.ctx()), "A FORCING FLAG STANDS", "the head reference's record does not say its read-backs passed")
        record.write_text(json.dumps({"passed": True}))
        assert lc.the_look(made.ctx())["gate_alone"] is True
        # THE RULE FOR A LOOK IS AS IT WAS: the maker's own look is no look. With the record the render rests on the gate
        # alone and says so; with a forcing flag it is refused, the maker's look named
        (cap / "looks.json").write_text(json.dumps({"looks": [{"by": "the maker", "at": "2026-01-03T10:30:00", "loads": ["ld"]}]}))
        assert lc.the_look(made.ctx())["gate_alone"] is True
        (cap / "flags.json").write_text(json.dumps({"verdict": "clear", "blocking": [], "flags": [forcing["three frames"][0]]}))
        refused(lambda: lc.the_look(made.ctx()), "A FORCING FLAG STANDS", "who made the capture")
        # somebody else's look is a look, and is what the render rests on, record or no record
        (cap / "looks.json").write_text(json.dumps({"looks": [{"by": "a reader", "at": "2026-01-03T10:30:00", "loads": ["ld"]}]}))
        assert lc.the_look(made.ctx())["by"] == "a reader" and "gate_alone" not in lc.the_look(made.ctx())
        # a record of an earlier capture, and a record of another load, are not this load's
        (cap / "looks.json").unlink()
        (cap / "flags.json").write_text(json.dumps({"verdict": "clear", "blocking": [], "flags": quiet}))
        (cap / "status.json").write_text(json.dumps({"written": "2099-01-01T00:00:00", "made_by": "the maker"}))
        refused(lambda: lc.the_look(made.ctx()), "is not later than the capture", "of an earlier capture")
        (cap / "status.json").write_text(json.dumps({"written": "2026-01-03T10:00:00", "made_by": "the maker"}))
        (cap / lc.NO_LOOK).write_text(json.dumps({"records": [{**rec, "load": "another"}]}))
        refused(lambda: lc.the_look(made.ctx()), "nobody has recorded a look", "load_chain.py no-look")
        assert "no-look" in lc.STEPS and "FORCING" in lc.step_help("no-look") and "gate alone" in lc.step_help("render").lower()
    return ("with no look, a render rests on the gate alone only by the maker's record that none is owed: refused, the flag named, for each of the nine "
            "forcing shapes, a blocked gate and a capture not on the plan; written for a clean load, as a record that is not a look; held again at the "
            "render to the flags as they stand, so it never serves a load that needs a look; the maker's own look is still no look; somebody else's look "
            "is what a render rests on when it is there")


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
            m.ran.clear()
            assert lc.step_land(made.ctx(), m, a) is None, "a land whose pose pass and stack ran did not end clean"
            # THE CARD IS FREED BEFORE THE POSE PASS, which runs beside the server in its own process
            assert m.labels() == ["free the card", "pose pass", "stack"], m.labels()
            pose, stack = m.said("pose pass"), m.said("stack")
            assert stack[stack.index("--render-first") + 1] == "110", f"the stack is not told the render's first source frame: {stack}"
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
            # A POSE PASS THAT DID NOT FINISH IS A LAND THAT DID NOT HAPPEN. The stack is still made, so the render can be
            # seen, and the step ends on the pass: its own exit status and the reason in capitals in its last line
            m.ran.clear()
            m.exits = {"pose pass": 1}
            ended = lc.step_land(made.ctx(), m, a)
            assert ended is not None and ended[0] == lc.POSE_PASS_FAILED and lc.POSE_PASS_FAILED not in (0, 5, lc.VERIFY_FAILED, lc.GATE_BLOCKED), ended
            assert "THE POSE PASS DID NOT FINISH (exit 1)" in ended[1] and "THIS LOAD IS NOT LANDED" in ended[1] and m.labels() == ["free the card", "pose pass", "stack"], (ended, m.labels())
            code, line = lc.last_line("land", "ld", ended, "NOW")
            assert code == lc.POSE_PASS_FAILED and "land of ld ENDED NOW AND IS NOT GOOD" in line and f"EXIT {lc.POSE_PASS_FAILED}" in line and "`render` refuses" not in line, line
            assert lc.last_line("land", "ld", None, "NOW") == (0, "land of ld ended NOW")
            # the stacked clip is there and the points are not: `land` again runs the pose pass alone, and is not refused as landed
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"")
            m.ran.clear()
            assert lc.step_land(made.ctx(), m, a)[0] == lc.POSE_PASS_FAILED and m.labels() == ["free the card", "pose pass"], m.labels()
            m.ran.clear()
            m.exits = {}
            assert lc.step_land(made.ctx(), m, a) is None and m.labels() == ["free the card", "pose pass"], "a land run again for its pose pass made the stack again, or did not end clean"
            # landed is the render's points written
            (cap / "ld_render_points.npz").write_bytes(b"")
            refused(lambda: lc.step_land(made.ctx(), m, a), "the render's points exist", "this load is landed")
        finally:
            lc.video_of = kept
            os.environ.pop(lc.MODEL_STORE, None)
            if env is not None:
                os.environ[lc.MODEL_STORE] = env
    return ("landing is refused without the pose model's store, on a queue that is not empty, with no window, with a second one, and when the render's "
            "points exist; the card is freed before the pose pass; the pass takes the capture's own track video and first frame, and the store from the "
            "flag or the environment; a pass that does not finish ends the step on its own exit status with the reason in capitals, the stack still made; "
            "run again with the clip there, the step does the pass alone")


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
case("a load's keep-outs reach its capture", a_loads_keep_outs)
case("a banded keep-out reaches the capture on the plan as the plan cut it", a_banded_keep_out_on_the_plan)
case("a track that is not its person on named frames", a_track_that_is_not_its_person)
case("first: who is who, the first capture, its strip under _first", the_first_step)
case("a kept output is not replaced, and a step has its own help", a_kept_output_is_not_replaced_and_a_step_has_its_own_help)
case("plan: built and queued on an empty queue, once", the_plan_step)
case("on-plan: the masks hashed, the capture copied, then built in place", the_on_plan_step)
case("the look and the gate a render rests on", the_look_and_the_gate)
case("a render on the gate alone", a_render_on_the_gate_alone)
case("a render graph against its plan graph, its masks and its job file", a_render_against_its_plan)
def what_the_gate_is_told() -> str:
    """The lead's rulings of 2026-10-14: the gate is told which keep-outs are the subject's own, what gives them back (the
    job's restore spans holding all of a keep-out's classes) and which frames the job empties on purpose, with its reasons."""
    ctx = {"subject": "second", "load": {"first": 100, "frames": 60}, "subject_keys": {"drop": "110-112+130", "drop_by": "A LOOK'S\n WORD"},
           "job": {"keep_outs": {"of": "second", "made": [{"name": "hands", "classes": ["Left_Hand", "Right_Hand"]}, {"name": "held", "track": "h"},
                                                          {"name": "strangers", "track": "s"}]},
                   "restore": [{"classes": ["Left_Hand", "Right_Hand"], "first": 14, "last": 120, "by": "THE LEAD"},
                               {"classes": ["Left_Hand"], "first": 140, "last": 150, "by": "ONE HAND"},
                               {"classes": ["Eyeglass", "Left_Hand", "Right_Hand"], "first": 300, "last": 310, "by": "ELSEWHERE"}],
                   "left_out": [{"subject": "second", "first": 90, "frames": 10, "why": "CUT BY A LOOK"}]}}
    told = lc.told_to_the_gate(ctx)
    pairs = list(zip(told[::2], told[1::2]))
    # THE CASE THAT GOES RED when strangers is told as her own, a restore of one hand gives back both, or a drop goes untold
    assert pairs == [("--own-keep-out", "hands=Left_Hand+Right_Hand"), ("--own-keep-out", "held="),
                     ("--given-back", "hands:100-120=the plan's restore of Left_Hand+Right_Hand on 14-120 gives it back at assembly (THE LEAD)"),
                     ("--named-empty", "second:110-112=the job's drop= (A LOOK'S WORD)"), ("--named-empty", "second:130-130=the job's drop= (A LOOK'S WORD)"),
                     ("--named-empty", "second:90-99=the job's left_out (CUT BY A LOOK)")], pairs
    other = dict(ctx, job=dict(ctx["job"], keep_outs={**ctx["job"]["keep_outs"], "of": "lead"}), subject_keys={})
    assert [x for x in lc.told_to_the_gate(other) if x.startswith("--")] == ["--named-empty"], "another subject's keep-outs told as hers, or a drop with no reason"
    # the job's reference points go to the gate, for the chin the held keep-out's split reads her clothing below (the lead, 2026-10-14)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix="_points.npz") as pts:
        told = lc.told_to_the_gate(dict(ctx, job=dict(ctx["job"], reference={"points": pts.name})))
        assert told[-2:] == ["--chin-points", pts.name], told
    assert "--chin-points" not in lc.told_to_the_gate(dict(ctx, job=dict(ctx["job"], reference={"points": "/nowhere/x_points.npz"}))), "a missing points file was told"
    return ("her own keep-outs with their classes, not strangers; a restore span gives back only a keep-out whose classes it holds, on the load's frames; the "
            "job's drop= with its reason and its left_out told as named; nothing of another subject's keep-outs; the reference's points as the chin")


def a_load_of_two_subjects() -> str:
    """A PAIR (the lead's ruling of 2026-10-15: a two-subject load runs through the chain like any load). Read as the builder reads it
    (`one_subject_view`): both subjects' parts, class maps and pose tables to the capture, each with its own job keys; neither subject
    among the others; the plan's run told the second subject through the job file; a pose pass for each subject at the land."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        m = made.m
        pv = m.output / "Video/v/s/pv"
        for kind in ("parts", "classes"):
            (pv / f"pv_b_{kind}_00001.mkv").write_bytes(b"")
            made.listed.append(f"Video/v/s/pv/pv_b_{kind}_00001.mkv")
        (pv / "pv_b_pose_table.json").write_text("{}")
        (made.pv / "preview.queued.json").write_text(json.dumps({"status": "success", "files": made.listed}))
        made.job["subjects"][0].update(pieces="largest", drop="112", drop_by="B'S DROP")
        pair = {"name": "ld", "first": 110, "frames": 20, "text": "ld",
                "subjects": [{"subject": "a", "region": "face and hair", "still": "a.png", "motion_file": "/out/ref.mp4", "others": []},
                             {"subject": "b", "region": "face", "motion_file": "/out/ref_b.mp4", "others": []}]}
        made.write([pair])
        ctx = made.ctx()
        assert (ctx["subject"], ctx["also"], ctx["also_keys"].get("pieces")) == ("a", "b", "largest"), (ctx["subject"], ctx["also"], ctx["also_keys"])
        files = lc.preview_files(ctx, m.output)
        b, a = lc.mask_specs(ctx, files)
        # THE CASE THAT GOES RED when the second subject is handed as a track alone, or takes the first's keys
        for part in ("b:track=", "parts=", "classes=", "pose_at=100", ",at=100,pieces=largest"):
            assert part in b, f"{part!r} is not in the second subject's mask: {b}"
        assert a.startswith("a:track=") and "parts=" in a and "pieces=" not in a, a
        assert lc.others_of(ctx) == [], f"a subject of the pair is kept out of the load: {lc.others_of(ctx)}"
        assert ["--named-empty", "b:112-112=the job's drop= (B'S DROP)"] == lc.told_to_the_gate(ctx)[-2:], lc.told_to_the_gate(ctx)
        # A PAIR'S KEEP-OUT IS GIVEN BACK BY ITS OWN SUBJECT'S RESTORE ONLY: a's hands span does not give back b's hands
        hands = {"of": ["a", "b"], "made": [{"name": "hands", "classes": ["Left_Hand"], "of": "b"}, {"name": "strangers", "track": "s"}]}
        spans = [{"classes": ["Left_Hand"], "first": 100, "last": 200, "by": "A'S", "subject": "a"},
                 {"classes": ["Left_Hand"], "first": 115, "last": 120, "by": "B'S", "subject": "b"}]
        told = lc.told_to_the_gate(dict(ctx, job=dict(ctx["job"], keep_outs=hands, restore=spans)))
        assert [x for x in told if x.startswith("hands:")] == ["hands:115-120=the plan's restore of Left_Hand on 115-120 gives it back at assembly (B'S)"], told
        refused(lambda: made.write([dict(pair, subjects=[pair["subjects"][0], dict(pair["subjects"][1], subject="c")])]) or made.ctx(),
                "the load's subject 'c' is not a subject of the preview")
        refused(lambda: made.write([dict(pair, subjects=[pair["subjects"][0], dict(pair["subjects"][1], others=["a"])])]) or made.ctx(),
                "is one of its two subjects and is listed among `others`")
        made.write([pair])
        # THE FORCING RULES read both subjects: the second's own hands inside the region are the load's, another's are not; and both
        # references' records are held to their read-backs
        fl = Path(tmp) / "fl"
        fl.mkdir()
        (fl / "flags.json").write_text(json.dumps({"flags": [{"id": "f1", "rule": "segment_inside_region", "segment": "b.Left_Hand"},
                                                             {"id": "f2", "rule": "segment_inside_region", "segment": "c.Left_Hand"}]}))
        ctx = made.ctx()
        assert [x.split()[0] for x in lc.forcing_flags(fl, ctx["load"], ctx["job"])] == ["f2"], lc.forcing_flags(fl, ctx["load"], ctx["job"])
        bad = Path(tmp) / "ref_b.json"
        bad.write_text(json.dumps({"passed": False}))
        assert any(str(bad) in x for x in lc.forcing_flags(fl, ctx["load"], dict(ctx["job"], reference_2={"record": str(bad)}))), "the second reference's record was not read"
        # a track that is not its person is never one of the pair's own
        made.job["track_not"] = {"made": [{"subject": "b", "track": "/t.mkv", "frames": [[110, 112]]}]}
        made.write([pair])
        refused(lambda: made.ctx(), "`track_not` names 'b'")
        del made.job["track_not"]
        made.write([pair])
        # the plan's run: its words name the first subject and none of the pair as another; the second comes from the job file
        cap = made.capture(on_plan=False)
        made.planned()
        windows = m.output / "Video/v/s/ld/ld_plan/ld_windows"
        windows.mkdir(parents=True)
        (windows / "ld_plan.json").write_text("{}")
        (made.root / "data" / "2026-01-03_ld_job" / "ld.who_is_who.json").write_text("{}")
        lc.step_on_plan(made.ctx(), m, args(), hashed=lambda path: "h")
        said = m.said("files on plan")
        assert said[said.index("--run") + 1] == f"ld:preview={windows},subject=a,at=110" and said[said.index("--loads") + 1] == str(made.job_file), said
        # the land: a pose pass for each subject, each on its own track, each to its own points; landed when both are written
        (cap / "subjects" / "b").mkdir(parents=True)
        (cap / "subjects" / "b" / "track__pv.mkv").write_bytes(b"track")
        manifest = json.loads((cap / "manifest.json").read_text())
        manifest["subjects"].append({"label": "b", "sightings": [{"by": "pv"}]})
        (cap / "manifest.json").write_text(json.dumps(manifest))
        (made.root / "data" / "2026-01-03_ld_job" / "ld.render.queued.json").write_text(json.dumps({"prompt_id": "r"}))
        (m.output / "Video/v/s/ld/ld/ld_windows").mkdir(parents=True)
        (m.output / "Video/v/s/ld/ld/ld_windows/ld_window_1.mp4").write_bytes(b"")
        store = made.root / "store"
        (store / "pose-model").mkdir(parents=True)
        kept, lc.video_of = lc.video_of, lambda ctx, data: "v"
        try:
            m.ran.clear()
            m.exits = {"pose pass of b": 1}
            ended = lc.step_land(made.ctx(), m, args(model_store=str(store)))
            assert m.labels() == ["free the card", "pose pass of a", "pose pass of b", "stack"], m.labels()
            assert ended[0] == lc.POSE_PASS_FAILED and "THE POSE PASS OF b DID NOT FINISH" in ended[1], ended
            pose_a, pose_b = m.said("pose pass of a"), m.said("pose pass of b")
            assert pose_b[pose_b.index("--box-mask") + 1] == str(cap / "subjects/b/track__pv.mkv") and pose_b[pose_b.index("--out") + 1] == str(cap / "ld_b_render_points.npz"), pose_b
            assert pose_a[pose_a.index("--out") + 1] == str(cap / "ld_render_points.npz"), pose_a
            # a's points written, b's not, the clip made: the land is run again for b's pass alone, and is not refused as landed
            (cap / "ld_render_points.npz").write_bytes(b"")
            clip = m.output / "Video/v/review_clips/ld_0110_0129_stacked.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"")
            m.ran.clear()
            m.exits = {}
            assert lc.step_land(made.ctx(), m, args(model_store=str(store))) is None and m.labels() == ["free the card", "pose pass of b"], m.labels()
            (cap / "ld_b_render_points.npz").write_bytes(b"")
            refused(lambda: lc.step_land(made.ctx(), m, args(model_store=str(store))), "this load is landed")
        finally:
            lc.video_of = kept
    return ("a pair is read as the builder reads it: both subjects' parts, class maps and pose tables to the capture, each with its own keys and drop; "
            "neither subject kept out; a subject the preview did not track, or one of the pair kept out, refused; the plan's run told the second "
            "subject through the job file; a pose pass for each subject, run again for the one not written, landed when both are")


case("render: the go beside the job file, and what stops it", the_render_step)
case("land: the pose pass and the stacked clip", the_land_step)
case("page: the item, with the table's buttons", the_page_step)
case("the gate is told what is the subject's own", what_the_gate_is_told)
case("a load of two subjects", a_load_of_two_subjects)

if __name__ == "__main__":
    sys.exit(finish())
