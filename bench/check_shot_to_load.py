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
2b. **A preview made to ask** (`--ask SUBJECT:why`): a subject the record is `unsure` of is tracked all the same,
   picked as an unseeded subject is, the note and the job's `made_to_ask` saying it is tracked to ask and why; a
   shot where everybody is unsure is refused without it, the refusal naming the flag. Asking about a subject who
   is `yes` or `no`, or with no reason, is refused, and NO LOAD is built on a subject the record is unsure of.
3. **A subject with a box in the cast file inside the shot is seeded** from it and carries no `pick_at`; a box
   outside the shot, or another subject's, seeds nothing. THE SEEDED ARE TRACKED FIRST, whatever the plan's order,
   and the note says the seed decides and the job's `pick` does not; a person's `order` on the shot is still over
   that.
3b. **A shot the tracker cuts takes a seed a cut** (`--seed-cuts-of JOB.json=TABLE.json`). With no cuts handed in a
   shot holding three boxes gets ONE seed, the latest in the cast file, and the note names the boxes not used and
   the flag (2026-10-13, 6105-6394: six boxes, one passed, a frame in a fade, the subject held on a small part of
   the shot). With the tracker's cuts read from an earlier preview's table, each cut that holds a box gets its
   own, the note says which table, and the job carries the cuts. A job file of other frames, a table of another
   length or with a gap in its cuts, a file that is not there and the flag without `--cast` are refused. After the
   queue the cuts the new preview's tracker found are held to the table's: other cuts are a refusal that says the
   preview is made and not to be built on.
4. **The plan must say what from**: a subject with no `still` or no `parts` is refused with the keys.
5. **The preview step** writes `shot_c_<first>`, refuses one that is there, and queues only when told, on an
   empty queue. MADE AGAIN (`--remake`) it takes the next free letter, `shot_c_<first>b`, past a job file here
   and a job folder of any day; its tables are named by its own name in its own folder, and nothing of the
   earlier preview is touched.
5b. **The first real run's fault** (2026-10-13). With the REAL `World.run`, which keeps each command's output, and
   a stand-in for the builder that refuses a job folder that exists as the builder does: the preview builds and
   queues, its output is kept beside the job file, and the job folder holds only what the builder made. A job
   folder or an output folder that is there, and a file that is not the one the video's folder was made from, are
   refused before anything is written, in a dry run as well. A seed whose cast line has an empty field is printed
   with no hole and the empty field said to be the file's.
6. **A load's name is made**: `<subject>_<form>_<first>`, refused when a job file, a capture or a job folder of
   any day has it, and the next free letter with `--remake`.
6c. **A load of some of its subject's parts** (`job --parts NAME ... --parts-by`; 2026-10-13: where her head is
   bowed and no face shows, her hair stays the new hair, and no job could say a load's part is the hair alone). The
   job names the part's file, its classes from the part model's own table, the class map it is cut from and the
   preview's saved part it is held to; the load's subject carries `parts_file`; the name is by the parts (`hp`). A
   load that names no parts has none of it and reads as before. Refused: a part the plan does not name, a part
   twice, every part of the plan, parts with no `--parts-by`. The step cuts the file before the job file is
   written, holds it to the saved part (a pixel outside it is refused and nothing is kept), holds a file that is
   there to a fresh cut, and a dry run cuts nothing. The cut itself, on a made class map and part: only the
   named part's pixels, every one inside the saved part.
7b. **A job renders from the video's present copy**, whatever copy its preview or the shots record was made on:
   an earlier copy on the same frame clock is said in the job's note with why it was replaced; a copy that begins
   on a later frame, and a file the video never had, are refused with nothing written.
7. **A load's job file stands alone**: `defaults` are `h3_config`'s, the margin by the plan's region; `pick`
   copied and no `pick_at`; the reference named by subject and span, not by the load's name; a text entry with
   no writer, which the chain refuses; a cut with no words refused and with them written into `left_out`; a
   plan's `hair_band` refused (7c); the plan's `pieces` copied onto the load's subject with who chose it, where the
   chain reads it, and onto no other subject, refused when it is not a value the capture takes or names nobody; `--zoomed`
   with its words makes the load `render_region: part` and keeps who said it, and `--least` with `--least-why` sets the
   subject's floor with its reason, each refused without its words, neither set when not asked, and the load the builder's
   `check_load` takes; and the refusals (a subject not in the shot, a preview of other frames, no pose
   pass for the subject, a whole-person region).
7c. **A load the builder would refuse is not written** (the first real load's fault, 2026-10-13). A `hair_band` in
   the plan is refused at `job`, naming the keep-out the preview does not track; a load under the song node's
   shortest is refused in the builder's words; `--dry` refuses both and writes nothing; and the load that is
   written passes `masked_job_graph.check_load`, which no case here held a written load to before.
7e. **A subject's keep-outs** (2026-10-13: the second subject's loads carried none and a render drew her apron
   again). A plan's `keep_outs` (of `held`, `hands`, `clothes`, with `keep_outs_by`) are in the load's `others`
   and in `job.keep_outs` with each file and its classes; a load written without one the plan names is red, here
   and at the chain's `read_job`. The job step cuts the hands and the clothing from the preview's class map
   before it writes the job file and takes the held things as the preview saved them; a dry run cuts nothing; a
   file that is there is held to a fresh cut and refused when it differs; a preview with no class map of the
   subject is refused. The cut itself is held on a class map written as the part node writes one: the classes'
   pixels on every frame, and the torso is not clothing.
7d. **The text step** says who wrote a load's text: the writer's words into `texts.<label>.by` with when and by
   which session, and nothing else of the job file changed; the chain's `read_job` and the builder's `text_entry`
   then take it; refused with no text file, with no words, and once the load's plan is built; `--dry` writes nothing.
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
import subprocess
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
        self.busy, self.ran, self.on, self.kept_in = (0, 0), [], {}, {}

    def queue(self):
        return self.busy

    def free(self):
        self.ran.append(("free the card", []))
        return "the card after it: free."

    def run(self, ctx, label, argv, stop=True):
        self.ran.append((label, [str(x) for x in argv]))
        self.kept_in[label] = ctx.get("logs")          # where the step says its output is kept; None is the job folder
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
            for kind, suffix in (("track", "mkv"), ("parts", "mkv"), ("classes", "mkv"), ("held_things", "mkv"), ("mesh", "mp4")):
                (out / f"{name}_{s['label']}_{kind}_00001.{suffix}").write_bytes(b"")
                listed.append(f"{job['job']['output']}/{name}/{name}_{s['label']}_{kind}_00001.{suffix}")
            (out / f"{name}_{s['label']}_shots_00001_shots.json").write_text("{}")
        (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": listed}))
        return folder

    def load(self, subject: str = "a", shot: int = 100, name: str = "a_fp_0100", **more) -> dict:
        shots, plan = self.read()
        pjob = self.preview(shot)
        folder = self.preview_ran(pjob)
        saved = json.loads((folder / "preview.queued.json").read_text())["files"]
        return stl.load_job(shots, plan, stl.shot_of(shots, shot), subject, pjob["job"], str(folder), "v", self.m.output, "the session", "2026-01-03", name, now="now",
                            preview_files=saved, **more)


def args(**more) -> argparse.Namespace:
    given = dict(by="the session", day="2026-01-03", cast=None, queue=False, first=None, frames=None, why=None, remake=False, plain=None, model_store="",
                 existing=False, writer=None, why_before=None, why_after=None, zoomed=None, least=None, least_why=None, ask=[], seed_cuts_of=None, parts=None, parts_by=None)
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


def a_preview_made_to_ask() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        who = made.shots["shots"][3]["who"]                                # 200-259, nobody
        for k in who:
            who[k] = {"present": "unsure", "how": "no answer of the owner's is on a frame of this shot", "by": "a session"}
        refused(lambda: made.preview(200), "nobody of ours", "`unsure`", '--ask "SUBJECT:why"')
        refused(lambda: made.preview(0), "nobody of ours")                 # nobody, and nobody unsure: the flag is not offered
        assert "--ask" not in refused(lambda: made.preview(0), "nobody of ours"), "a shot the owner said nobody is in was offered an asking preview"
        job = made.preview(200, ask={"a": "the owner has not answered on this cut"})
        one = job["job"]["subjects"]
        assert [s["label"] for s in one] == ["a"] and one[0]["pick_at"] == 229 and "seeds" not in one[0], one
        assert "TRACKED TO ASK, NOT BECAUSE THE RECORD SAYS THEY ARE HERE" in job["job"]["note"] and "the owner has not answered on this cut" in job["job"]["note"], job["job"]["note"]
        graphs.check_job(job["job"])
        refused(lambda: made.preview(100, ask={"a": "why"}), "--ask a", "'yes'")
        refused(lambda: made.preview(160, ask={"b": "why"}), "--ask b", "'no'")
        refused(lambda: made.preview(200, ask={"a": "  "}), "--ask a", "with the reason")
        made.shots["shots"][1]["who"]["b"]["present"] = "unsure"           # a shot with others in it: the asked one is tracked beside them
        assert sorted(s["label"] for s in made.preview(100, ask={"b": "is the second person b"})["job"]["subjects"]) == ["a", "b", "c"]
        made.shots["shots"][1]["who"]["b"]["present"] = "yes"
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=200, folder=str(made.jobs), ask=["a: the owner has not answered on this cut"])
        stl.step_preview(a, made.m)
        kept = json.loads((made.jobs / "shot_c_0200.json").read_text())["job"]
        graphs.check_job(kept)
        assert kept["made_to_ask"]["subjects"] == {"a": "the owner has not answered on this cut"} and kept["made_to_ask"]["by"] == "the session", kept.get("made_to_ask")
        a.ask = ["a"]
        (made.jobs / "shot_c_0200.json").unlink()
        refused(lambda: stl.step_preview(a, made.m), "is not `SUBJECT:why`")
        # no load on it: the record still does not say the subject is in the shot
        pjob = made.preview(200, ask={"a": "asked"})
        folder = made.preview_ran(pjob)
        shots, plan = made.read()
        saved = json.loads((folder / "preview.queued.json").read_text())["files"]
        refused(lambda: stl.load_job(shots, plan, stl.shot_of(shots, 200), "a", pjob["job"], str(folder), "v", made.m.output, "the session", "2026-01-03", "a_fp_0200",
                                     now="now", preview_files=saved), "does not say 'a' is in 200-259", "unsure")
    return "a subject the record is unsure of is tracked only when the preview is made to ask, with the reason; no load is built on it"


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
        # THE SEEDED FIRST, whatever the plan's order (b, a, c): a guess by the largest person takes whoever is left. The
        # first real preview ran the unseeded subject first and it took the seeded one's person (2026-10-13, shot 604)
        assert [s["label"] for s in job["subjects"]] == ["a", "b", "c"], [s["label"] for s in job["subjects"]]
        assert "the seeded before the others" in job["note"] and "THE SEED DECIDES who is tracked" in job["note"] \
            and "`pick: largest` stands in the job file only because the builder asks" in job["note"], job["note"]
        # two seeded, one not: the two in the plan's order, then the other; a person's `order` on the shot is still over it
        both = lines + [cast_file.entry("b", 130, cast_file.HERE, by="a person", box=[100, 20, 30, 40], size=[w, h], table="t.json", at="t4")]
        assert [s["label"] for s in made.preview(cast_lines=both)["job"]["subjects"]] == ["b", "a", "c"]
        made.shots["shots"][1]["order"] = ["c", "b", "a"]
        by_hand = made.preview(cast_lines=lines)["job"]
        assert [s["label"] for s in by_hand["subjects"]] == ["c", "b", "a"] and "the shot's own `order`" in by_hand["note"], by_hand["note"]
        del made.shots["shots"][1]["order"]
        # a cast line whose own words have an empty field (a page answer with no time on it): printed with no hole, and said
        holed = [cast_file.entry("a", 125, cast_file.HERE, by="a person, on the page at  (lines 3-4)", box=[10, 20, 30, 40], size=[w, h], table="t.json", at="t1")]
        note = made.preview(cast_lines=holed)["job"]["note"]
        assert "at  (" not in note and "a person, on the page at (lines 3-4)" in note and "a field of that line is EMPTY in the cast file" in note, note
        assert "EMPTY in the cast file" not in job["note"], "a cast line with no empty field was said to have one"
        assert all("seeds" not in s for s in made.preview()["job"]["subjects"]), "a preview with no cast file was seeded"
    return "a subject with a box in the cast file inside the shot is seeded from it, scaled to the canvas, and carries no pick_at; a box in another shot seeds nothing"


def a_shot_the_tracker_cuts_takes_a_seed_a_cut() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        w, h = cfg.MASKED_JOB_CANVAS
        box = lambda frame, x, at: cast_file.entry("a", frame, cast_file.HERE, by="a person", box=[x, 20, 30, 40], size=[w, h], table="t.json", at=at)   # noqa: E731
        lines = [box(105, 10, "t1"), box(150, 30, "t2"), box(125, 20, "t3")]        # in the file's order: 125 is the latest
        a_of = lambda job: next(x for x in job["job"]["subjects"] if x["label"] == "a")   # noqa: E731
        # AS IT WAS: one seed a record shot, the latest in the file, whatever the tracker's cuts
        one = made.preview(cast_lines=lines)
        assert a_of(one)["seeds"] == "frame 25: 20, 20, 30, 40", a_of(one)["seeds"]
        assert "SEEDED ON SOURCE FRAME(S) 125." in one["job"]["note"] and "THE OWNER'S BOX(ES) ON 105, 150 ARE NOT USED" in one["job"]["note"] \
            and "--seed-cuts-of" in one["job"]["note"] and "the order they were clicked in" in one["job"]["note"], one["job"]["note"]
        # the tracker's three cuts, each holding one box: three seeds
        three = {"read_from": "T.json", "the_preview": "shot_c_0100", "cuts": [[0, 20], [20, 40], [40, 60]]}
        job = made.preview(cast_lines=lines, seed_cuts=three)
        assert a_of(job)["seeds"] == "frame 5: 10, 20, 30, 40\nframe 25: 20, 20, 30, 40\nframe 50: 30, 20, 30, 40", a_of(job)["seeds"]
        graphs.check_job(job["job"])
        note = job["job"]["note"]
        assert "THE TRACKER'S CUTS OF THIS SHOT, 3 of them, WERE READ FROM T.json" in note and "SEEDED ON SOURCE FRAME(S) 105, 125, 150, one in each of 3 of the tracker's 3 cuts" in note \
            and "ARE NOT USED" not in note, note
        # two boxes in one cut: the cut takes one, the other is named as not used, and the flag is not offered again
        two = made.preview(cast_lines=lines, seed_cuts={"read_from": "T.json", "the_preview": "p", "cuts": [[0, 40], [40, 60]]})
        assert a_of(two)["seeds"] == "frame 25: 20, 20, 30, 40\nframe 50: 30, 20, 30, 40" and "BOX(ES) ON 105 ARE NOT USED" in two["job"]["note"] \
            and "If the tracker cuts this shot" not in two["job"]["note"], two["job"]["note"]
        # a cut with no box gets none, and a subject with no box is picked as ever
        gap = made.preview(cast_lines=lines, seed_cuts={"read_from": "T.json", "the_preview": "p", "cuts": [[0, 3], [3, 40], [40, 60]]})
        assert a_of(gap)["seeds"].count("frame") == 2 and "one in each of 2 of the tracker's 3 cuts" in gap["job"]["note"], gap["job"]["note"]
        assert "seeds" not in next(x for x in job["job"]["subjects"] if x["label"] == "b")
        # THE TABLE AND ITS JOB FILE, as handed in
        shot = stl.shot_of(made.shots, 100)
        jf, tf = made.root / "earlier_job.json", made.root / "earlier_table.json"
        table = lambda frames, cuts: tf.write_text(json.dumps({"frames": frames, "shots": [{"first_frame": x, "last_frame": y - 1} for x, y in cuts]}))   # noqa: E731
        jf.write_text(json.dumps({"job": {"name": "shot_c_0100", "first": 100, "frames": 60}}))
        table(60, three["cuts"])
        pair = f"{jf}={tf}"
        got = stl.cuts_of_table(pair, shot)
        assert got["cuts"] == three["cuts"] and got["read_from"] == str(tf) and got["the_preview"] == "shot_c_0100" and got["its_job_file"] == str(jf), got
        table(50, [[0, 20], [20, 50]])
        refused(lambda: stl.cuts_of_table(pair, shot), "holds 50 frames", "not a table of this shot's frames")
        table(60, [[0, 20], [30, 60]])
        refused(lambda: stl.cuts_of_table(pair, shot), "do not lie end to end")
        table(60, three["cuts"])
        jf.write_text(json.dumps({"job": {"name": "shot_c_0160", "first": 160, "frames": 60}}))
        refused(lambda: stl.cuts_of_table(pair, shot), "is a preview of 160-219, not of the shot 100-159")
        jf.write_text(json.dumps({"job": {"name": "shot_c_0100", "first": 100, "frames": 60}}))
        refused(lambda: stl.cuts_of_table(str(jf), shot), "is not `JOB.json=TABLE.json`")
        refused(lambda: stl.cuts_of_table(f"{jf}={made.root / 'none.json'}", shot), "is not there")
        # THE STEP: the job file carries the cuts; after the queue the tracker's cuts are held to them
        cast = made.root / "cast.jsonl"
        cast.write_text("".join(json.dumps(x) + "\n" for x in lines))
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), cast=str(cast), seed_cuts_of=pair, queue=True)
        saved = made.m.output / "Video/v/the session/shot_c_0100"

        def tracker(cuts):
            def run():
                saved.mkdir(parents=True)
                (saved / "shot_c_0100_a_shots_00001_shots.json").write_text(json.dumps({"frames": 60, "shots": [{"first_frame": x, "last_frame": y - 1} for x, y in cuts]}))
            return run
        made.m.on["queue preview"] = tracker(three["cuts"])
        stl.step_preview(a, made.m)
        kept = json.loads((made.jobs / "shot_c_0100.json").read_text())["job"]
        graphs.check_job(kept)
        assert kept["seed_cuts"]["cuts"] == three["cuts"] and kept["seed_cuts"]["read_from"] == str(tf), kept.get("seed_cuts")
        assert next(x for x in kept["subjects"] if x["label"] == "a")["seeds"].count("frame") == 3
        shutil.rmtree(saved)
        (made.jobs / "shot_c_0100.json").unlink()
        made.m.on["queue preview"] = tracker([[0, 30], [30, 60]])
        refused(lambda: stl.step_preview(a, made.m), "IS MADE AND IS NOT TO BE BUILT ON", "cut the shot into 2 cut(s)", "which are not the same cuts")
        shutil.rmtree(saved)
        (made.jobs / "shot_c_0100.json").unlink()
        a.cast = None
        refused(lambda: stl.step_preview(a, made.m), "--seed-cuts-of goes with --cast")
    return "a shot the tracker cuts gets a seed in each cut that holds a box, from cuts read out of a named table of the same frames; one seed without them, the rest named"


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
        # A PREVIEW MADE AGAIN takes the next free letter and leaves the earlier one as it is (2026-10-13: forty-two previews
        # were to be made again from the owner's answers, each under a name whose job folder was there)
        (made.root / "data/2026-01-03_shot_c_0100_job").mkdir()
        refused(lambda: stl.step_preview(a, made.m), "shot_c_0100.json exists", "--remake", "leaves the earlier one as it is")
        out.unlink()                                             # the job file gone and its job folder there: the builder's own refusal
        refused(lambda: stl.step_preview(a, made.m), "the job folder", "exists", "--remake")
        made.m.ran.clear()
        a.remake = True
        stl.step_preview(a, made.m)
        again = json.loads((made.jobs / "shot_c_0100b.json").read_text())["job"]
        graphs.check_job(again)
        assert again["name"] == "shot_c_0100b" and "THIS IS A PREVIEW MADE AGAIN: shot_c_0100b, after shot_c_0100" in again["note"], again["note"]
        assert made.m.said("queue preview")[3] == str(made.root / "data/2026-01-03_shot_c_0100b_job"), made.m.said("queue preview")
        # its tables are its own name's, in its own folder: what the who page and the cast step are handed as a pair
        assert masked_job.expected_files(again)["a"]["shots"].startswith("Video/v/the session/shot_c_0100b/shot_c_0100b_a_shots_"), masked_job.expected_files(again)["a"]
        (made.root / "data/2025-12-30_shot_c_0100c_job").mkdir()      # a job folder of another day holds the next letter
        stl.step_preview(a, made.m)
        assert (made.jobs / "shot_c_0100d.json").is_file() and not (made.jobs / "shot_c_0100c.json").exists(), sorted(p.name for p in made.jobs.iterdir())
        assert sorted(p.name for p in (made.root / "data").glob("*shot_c_0100*")) == ["2025-12-30_shot_c_0100c_job", "2026-01-03_shot_c_0100_job"], "an earlier folder was touched"
    return ("the preview's job file is written as shot_c_<first>, once; it is queued only when told, on an empty queue; made again it takes the next free "
            "letter, with the earlier preview's files as they were")


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
                stdout.write("a line of the builder's that is longer than the screen shows: " + "x" * 400 + " THE END OF THE LINE\n")
                stdout.write("built\n")
            else:
                assert Path(argv[3]).is_dir(), f"the queue was handed a job folder that is not there: {argv[3]}"
                stdout.write("queued\n")
            return subprocess.CompletedProcess(argv, 0)

        m = Real(made.root)
        kept, lc.subprocess.run = lc.subprocess.run, builder
        try:
            a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), queue=True)
            import contextlib
            import io
            screen = io.StringIO()
            with contextlib.redirect_stdout(screen):
                stl.step_preview(a, m)
            print(screen.getvalue(), end="")
            # a line longer than the screen shows is cut there and SAYS so; the kept file has it whole
            shown = next(line for line in screen.getvalue().splitlines() if "longer than the screen shows" in line)
            assert "THE END OF THE LINE" not in shown and "[CUT HERE:" in shown and "whole in the kept file]" in shown, shown
            assert "THE END OF THE LINE" in (made.jobs / "chain" / "shot_c_0100.build_preview.out").read_text()
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
        # the name carries the load's own first frame and the region's form (b's band is taken out: a band is refused, below)
        del made.plan["subjects"]["b"]["hair_band"]
        made.write()
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
        # each end of a cut with its own reason (2026-10-13: a load remade shorter carried one sentence on both entries)
        two = made.load(first=104, frames=40, why_before="frames before 104 are another load's", why_after="a hand is over the face after 143", name="a_fp_0104")
        assert [(x["first"], x["frames"], x["why"]) for x in two["job"]["left_out"]] == [(100, 4, "frames before 104 are another load's"),
                                                                                     (144, 16, "a hand is over the face after 143")], two["job"]["left_out"]
        assert "100-103: frames before 104 are another load's 144-159: a hand is over the face after 143" in two["job"]["note"], two["job"]["note"]
        mixed = made.load(first=104, frames=40, why="one reason", why_after="its own", name="a_fp_0104")
        assert [x["why"] for x in mixed["job"]["left_out"]] == ["one reason", "its own"], mixed["job"]["left_out"]
        refused(lambda: made.load(first=104, frames=40, why_after="only the end is said", name="a_fp_0104"), "say what a look cut before it", "--why-before")
        refused(lambda: made.load(first=100, frames=40, why_before="nothing is cut there", why_after="w", name="a_fp_0100"), "--why-before is given and no frame before the load is cut")
        masked_job.left_out(two["job"])
        # the plan's `pieces` goes onto the load's subject, with who chose it, and onto nobody else; with none in the plan the job has none
        assert all("pieces" not in s for s in d["job"]["subjects"]) and "pieces=" not in job["note"], d["job"]["subjects"]
        made.plan["subjects"]["a"].update(pieces="largest", pieces_by="a person, 2026-01-01: her arm carried specks of the part")
        pieced = made.load()
        mine, others = next(s for s in pieced["job"]["subjects"] if s["label"] == "a"), [s for s in pieced["job"]["subjects"] if s["label"] != "a"]
        assert (mine["pieces"], mine["pieces_by"]) == ("largest", "the plan (a person, 2026-01-01: her arm carried specks of the part)"), mine
        assert all("pieces" not in s for s in others) and "THE CAPTURE READS a's PART WITH pieces=largest, from the plan" in pieced["job"]["note"], pieced["job"]["note"]
        graphs.check_job(pieced["job"])
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(pieced))
        ctx = lc.read_job(made.jobs / "a_fp_0100.json", made.root, need_text=False)
        assert ctx["subject_keys"]["pieces"] == "largest", ctx["subject_keys"]
        made.plan["subjects"]["a"]["pieces"] = "smallest"
        refused(made.load, "`pieces` is one of", "'smallest'")
        made.plan["subjects"]["a"].update(pieces="largest", pieces_by=" ")
        refused(made.load, "`pieces_by` says who chose it")
        del made.plan["subjects"]["a"]["pieces"], made.plan["subjects"]["a"]["pieces_by"]
        # TWO CHOICES ABOUT ONE LOAD, each with its words and neither with a default: rendered zoomed in, and the capture's floor
        assert "render_region" not in load and "zoomed" not in job and all("least" not in s for s in job["subjects"]), "a load is zoomed or floored with nobody saying so"
        z = made.load(zoomed="her face is  40 px tall on the whole frame", least=600, least_why="the part is drawn four times its whole-frame size")
        zl, zs = z["loads"][0], next(s for s in z["job"]["subjects"] if s["label"] == "a")
        assert zl["render_region"] == "part" and z["job"]["zoomed"] == {"why": "her face is 40 px tall on the whole frame", "said_by": "the session", "at": "now"}, z["job"].get("zoomed")
        assert (zs["least"], zs["least_by"]) == (600, "the session, now: the part is drawn four times its whole-frame size"), zs
        assert all("least" not in s for s in z["job"]["subjects"] if s["label"] != "a"), "another subject's floor was set"
        assert "RENDERED ZOOMED IN on what is replaced (`render_region: part`), on the session's word: her face is 40 px tall" in z["job"]["note"] \
            and "THE CAPTURE'S FLOOR FOR a's PART IS SET FOR THIS LOAD: least=600 px" in z["job"]["note"], z["job"]["note"]
        graphs.check_job(z["job"])
        for one in masked_job.loads_of(z):                        # the builder takes a zoomed load written this way: a part, with a motion file of its own
            graphs.check_load(z["job"], one, masked_job.chain_lengths())
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(z))
        ctx = lc.read_job(made.jobs / "a_fp_0100.json", made.root, need_text=False)
        assert ctx["subject_keys"]["least"] == 600 and ctx["load"]["render_region"] == "part", ctx["subject_keys"]
        refused(lambda: made.load(zoomed="  "), "--zoomed takes the reason in words")
        refused(lambda: made.load(least=600), "--least and --least-why go together")
        refused(lambda: made.load(least_why="words and no number"), "--least and --least-why go together")
        refused(lambda: made.load(least=-1, least_why="w"), "0 (nothing emptied by size) or more")
        assert made.load(least=0, least_why="nothing is emptied by size on this load")["job"]["subjects"][1]["least"] == 0
        # the margin is the plan's region's. A band the plan asks is REFUSED: it is cut out of a keep-out the preview does not track
        refused(lambda: made.load("b", name="b_hfp_0100"), "the plan asks a hair band for 'b' (0.6)", f"{cfg.MASKED_JOB_HAIR_BAND['cuts']!r}",
                "tracks b, a, c", "would refuse this load at its plan", "take `hair_band` out")
        del made.plan["subjects"]["b"]["hair_band"]
        plain = made.load("b", name="b_hfp_0100")
        assert plain["defaults"]["margin"] == cfg.MASKED_LOAD_MARGIN["face and hair"] != cfg.MASKED_LOAD_MARGIN["face"] and "hair_band" not in plain["loads"][0], plain
        for one in masked_job.loads_of(plain):
            graphs.check_load(plain["job"], one, masked_job.chain_lengths())
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


def a_load_the_builder_would_refuse_is_not_written() -> str:
    """The first real load's fault (2026-10-13): `job` wrote a load with the plan's hair band, `job --dry` and the
    chain's `plan --dry` each exited 0, and the builder refused it at its plan."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        folder = made.preview_ran(made.preview())
        b = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="b", preview=str(folder))
        for dry in (True, False):
            made.m.dry = dry
            refused(lambda: stl.step_job(b, made.m), "the plan asks a hair band", "Nothing is written")
        assert not list(made.jobs.iterdir()), f"a refused load was written: {sorted(p.name for p in made.jobs.iterdir())}"
        # with the band out of the plan the same load is written, and a dry run of it writes nothing and says what it held
        del made.plan["subjects"]["b"]["hair_band"]
        made.write()
        made.m.dry = True
        stl.step_job(b, made.m)
        assert not list(made.jobs.iterdir()), "a dry run wrote a job file"
        # anything else the builder refuses on the job file alone stops it too, dry or not: a load under the song node's shortest
        _, least = masked_job.shortest_load()
        short = args(**{**vars(b), "first": 100, "frames": least - 1, "why": "a look cut it"})
        for dry in (True, False):
            made.m.dry = dry
            refused(lambda: stl.step_job(short, made.m), "the builder would refuse this load, so nothing is written", f"{least - 1} frames")
        assert not list(made.jobs.iterdir()), f"a refused load was written: {sorted(p.name for p in made.jobs.iterdir())}"
        made.m.dry = False
        stl.step_job(b, made.m)
        d = json.loads((made.jobs / "b_hfp_0100.json").read_text())
        for one in masked_job.loads_of(d):
            graphs.check_load(d["job"], one, masked_job.chain_lengths())
    return ("a plan's hair band is refused at `job` with the keep-out the preview lacks, and a load under the shortest is refused as the builder would; "
            "a dry run refuses the same and writes nothing; the load that is written passes the builder's check of a load")


def a_subjects_keep_outs() -> str:
    """Every load of the second subject through the tool carried no clothing, hands or held-thing keep-out, and the
    first render drew her apron as the still's top (2026-10-13). The plan names them; `job` makes and names them."""
    import numpy as np
    from _lib.frames import read_mask
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        plain = made.load()
        assert "keep_outs" not in plain["job"] and plain["loads"][0]["others"] == ["b", "c"], "a load has keep-outs nobody named"
        made.plan["subjects"]["a"].update(keep_outs=["held", "hands", "clothes"], keep_outs_by="a person, 2026-01-01: her apron was drawn again")
        d = made.load()
        job, load = d["job"], d["loads"][0]
        # THE CASE THAT GOES RED when a load of a subject whose plan names a keep-out is written without it
        assert load["others"] == ["b", "c", "held", "hands", "clothes"], f"the plan names three keep-outs for a and the load keeps out {load['others']}"
        kept = {k["name"]: k for k in job["keep_outs"]["made"]}
        assert list(kept) == ["held", "hands", "clothes"] and job["keep_outs"]["of"] == "a" and job["keep_outs"]["first"] == 100, job["keep_outs"]
        out = made.m.output
        assert kept["held"]["track"] == str(out / "Video/v/the session/shot_c_0100/shot_c_0100_a_held_things_00001.mkv") and "classes" not in kept["held"], kept["held"]
        assert kept["hands"]["track"] == str(out / "Video/v/keep_outs/shot_c_0100/shot_c_0100_a_hands.mkv") and kept["hands"]["classes"] == list(stl.KEEP_OUTS["hands"]) \
            and kept["hands"]["cut_from"] == str(out / "Video/v/the session/shot_c_0100/shot_c_0100_a_classes_00001.mkv"), kept["hands"]
        assert kept["clothes"]["classes"] == ["Apparel", "Upper_Clothing"] and "Torso" not in kept["clothes"]["classes"], kept["clothes"]
        assert "KEPT AS THE SOURCE'S ON THIS LOAD, from the plan (a person, 2026-01-01: her apron was drawn again)" in job["note"] \
            and "`clothes` (Apparel + Upper_Clothing of the class map)" in job["note"], job["note"]
        graphs.check_job(job)
        for one in masked_job.loads_of(d):
            graphs.check_load(job, one, masked_job.chain_lengths())
        # the chain reads the load as written, and refuses the same file with a keep-out taken out of `others`
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(d))
        assert [n for n, _ in lc.read_job(made.jobs / "a_fp_0100.json", made.root, need_text=False)["keep_outs"]] == ["held", "hands", "clothes"]
        d["loads"][0]["others"] = ["b", "c", "held", "hands"]
        (made.jobs / "a_fp_0100.json").write_text(json.dumps(d))
        refused(lambda: lc.read_job(made.jobs / "a_fp_0100.json", made.root, need_text=False), "the job names the keep-out 'clothes'", "drawn again")
        (made.jobs / "a_fp_0100.json").unlink()
        # what the plan may say
        for wrong, words in (({"keep_outs": ["held", "apron"]}, "a list of"), ({"keep_outs": []}, "a list of"), ({"keep_outs": ["held", "held"]}, "each once"),
                             ({"keep_outs_by": " "}, "`keep_outs_by` says who chose them")):
            was = dict(made.plan["subjects"]["a"])
            made.plan["subjects"]["a"].update(wrong)
            refused(made.load, words)
            made.plan["subjects"]["a"] = was
        # THE JOB STEP makes the cut ones before it writes the job file; a dry run makes nothing and says what it would
        folder = made.preview_ran(made.preview())
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        cuts = []

        def cut(class_map, classes, to):
            cuts.append((Path(class_map).name, tuple(classes), Path(to).name))
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"cut")
            return {"frames": 60, "size": [64, 48], "frames_with_the_mask": 60, "px_most": 9}
        made.m.dry = True
        stl.step_job(a, made.m, cut=cut)
        assert cuts == [] and not (out / "Video/v/keep_outs").exists() and not list(made.jobs.iterdir()), "a dry run cut a keep-out or wrote a job file"
        made.m.dry = False
        stl.step_job(a, made.m, cut=cut)
        assert cuts == [("shot_c_0100_a_classes_00001.mkv", stl.KEEP_OUTS["hands"], "shot_c_0100_a_hands.mkv"),
                        ("shot_c_0100_a_classes_00001.mkv", stl.KEEP_OUTS["clothes"], "shot_c_0100_a_clothes.mkv")], cuts
        record = json.loads((out / "Video/v/keep_outs/shot_c_0100/shot_c_0100_a_hands.json").read_text())
        assert record["classes"] == list(stl.KEEP_OUTS["hands"]) and record["first_source_frame"] == 100 and record["of"] == "a", record
        assert json.loads((made.jobs / "a_fp_0100.json").read_text())["loads"][0]["others"][-3:] == ["held", "hands", "clothes"]
        # a second load of the same subject on the same preview: the files are there, cut again and held equal; not equal is refused
        a.remake = True
        stl.step_job(a, made.m, cut=cut, hashed=lambda path: "the same")
        assert len(cuts) == 4 and all(c[2].endswith(".cut_again.mkv") for c in cuts[2:]) and not list((out / "Video/v/keep_outs/shot_c_0100").glob("*cut_again*")), cuts
        assert (made.jobs / "a_fp_0100b.json").is_file()
        refused(lambda: stl.step_job(a, made.m, cut=cut, hashed=lambda path: path), "is there and is NOT what its classes cut", "moved aside by a person")
        assert not (made.jobs / "a_fp_0100c.json").exists(), "a job file was written over a keep-out of another making"
        # a preview that saved no class map of the subject cannot give them
        (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": [
            f for f in json.loads((folder / "preview.queued.json").read_text())["files"] if "_a_classes_" not in f]}))
        refused(lambda: stl.step_job(a, made.m, cut=cut), "saved 0 `classes` video(s) of 'a'", "the preview is made again")
        # THE CUT ITSELF, on a class map written as the part node writes one (the grey level is the class): the pixels of the classes, on every frame
        import capture_masked_run as cap
        names = cap._pack("sapiens2_parts").CLASS_NAMES
        cmap = np.zeros((3, 48, 64), np.uint8)
        cmap[:, 4:20, 4:20] = names.index("Upper_Clothing")
        cmap[:, 20:30, 4:20] = names.index("Torso")
        cmap[1, 30:40, 30:50] = names.index("Apparel")
        cmap[:, 40:46, 50:60] = names.index("Left_Hand")
        video = Path(tmp) / "classes.mkv"
        enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                              str(video)], input=cmap.tobytes())
        assert enc.returncode == 0 and (cap.read_classes(str(video), (64, 48), 0, 0, 3) == cmap).all(), "the class map did not come back as written"
        got = stl.cut_keep_out(video, list(stl.KEEP_OUTS["clothes"]), Path(tmp) / "made" / "clothes.mkv")
        back = read_mask(str(Path(tmp) / "made" / "clothes.mkv"), (64, 48), 0, 0, 3)[0]
        want = np.isin(cmap, [names.index("Apparel"), names.index("Upper_Clothing")])
        assert (back == want).all() and not back[:, 20:30, 4:20].any() and got == {"frames": 3, "size": [64, 48], "frames_with_the_mask": 3, "px_most": 16 * 16 + 10 * 20}, got
        hands = stl.cut_keep_out(video, list(stl.KEEP_OUTS["hands"]), Path(tmp) / "made" / "hands.mkv")
        assert (read_mask(str(Path(tmp) / "made" / "hands.mkv"), (64, 48), 0, 0, 3)[0] == (cmap == names.index("Left_Hand"))).all() and hands["px_most"] == 60, hands
    return ("a plan's keep-outs are named in the load's others and written with their files into the job file; the job step cuts the hands and the clothing from "
            "the preview's class map and takes the held things as saved; a file that is there is held to a fresh cut; the torso is not clothing")


def a_load_of_some_of_its_parts() -> str:
    import numpy as np
    from _lib.frames import read_mask
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        out = made.m.output
        why = "the lead, on the owner's answer: her hair stays the new hair where no face shows"
        del made.plan["subjects"]["b"]["hair_band"]              # a band is another case's: it needs a keep-out this fixture's preview does not track
        plain = made.load(subject="b", name="b_hfp_0100")
        assert "part" not in plain["job"] and not any("parts_file" in x for x in plain["job"]["subjects"]) and "ALONE" not in plain["job"]["note"], "a load has a part of its own nobody named"
        d = made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why)
        job = d["job"]
        part, one = job["part"], job["part"]["made"][0]
        assert (part["of"], part["parts"], part["the_plans_parts"], part["by"], part["first"]) == ("b", ["hair"], ["hair", "face_and_neck"], why, 100), part
        assert one["classes"] == ["Hair"] and one["track"] == str(out / "Video/v/parts/shot_c_0100/shot_c_0100_b_part_hair.mkv") \
            and one["cut_from"] == str(out / "Video/v/the session/shot_c_0100/shot_c_0100_b_classes_00001.mkv") \
            and one["inside"] == str(out / "Video/v/the session/shot_c_0100/shot_c_0100_b_parts_00001.mkv"), one
        mine = next(x for x in job["subjects"] if x["label"] == "b")
        assert mine["parts_file"] == one["track"] and mine["parts_of_this_load"] == ["hair"] and mine["parts_by"] == why and mine["parts"] == ["hair", "face_and_neck"], mine
        assert not any("parts_file" in x for x in job["subjects"] if x["label"] != "b")
        assert "THIS LOAD'S PART IS HAIR ALONE, not the plan's hair + face_and_neck" in job["note"] and "the chain hands the capture that file" in job["note"], job["note"]
        graphs.check_job(job)
        for load in masked_job.loads_of(d):
            graphs.check_load(job, load, masked_job.chain_lengths())
        # the face alone is the other subset, by the part model's own classes for it
        face = made.load(subject="b", name="b_fp_0100", parts=["face_and_neck"], parts_by=why)["job"]["part"]["made"][0]
        assert face["classes"] == list(stl.part_classes()["face_and_neck"]) and "Face_Neck" in face["classes"] and face["track"].endswith("shot_c_0100_b_part_face_and_neck.mkv"), face
        for wrong, words in (({"parts": ["nose"], "parts_by": why}, "some of the plan's"), ({"parts": ["hair", "hair"], "parts_by": why}, "each once"),
                             ({"parts": ["hair"], "parts_by": " "}, "--parts-by says who chose them"), ({"parts": None, "parts_by": why}, "some of the plan's"),
                             ({"parts": ["face_and_neck", "hair"], "parts_by": why}, "that is the load as it is"), ({"parts": ["upper_clothing"], "parts_by": why}, "some of the plan's")):
            refused(lambda w=wrong: made.load(subject="b", name="b_hp_0100", **w), words)
        # THE STEP: named by its parts; a dry run cuts nothing; the cut is made and held before the job file is written
        folder = made.preview_ran(made.preview())
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="b", preview=str(folder), parts=["hair"], parts_by=why)
        cuts, held = [], []

        def cut(class_map, classes, to):
            cuts.append((Path(class_map).name, tuple(classes), Path(to).name))
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"cut")
            return {"frames": 60, "size": [64, 48], "frames_with_the_mask": 60, "px_most": 9}

        def inside(px):
            def look(cut_file, whole):
                held.append((Path(cut_file).name, Path(whole).name))
                return px
            return look
        made.m.dry = True
        stl.step_job(a, made.m, cut=cut, inside=inside(0))
        assert cuts == [] and held == [] and not (out / "Video/v/parts").exists() and not list(made.jobs.iterdir()), "a dry run cut a part or wrote a job file"
        made.m.dry = False
        refused(lambda: stl.step_job(a, made.m, cut=cut, inside=inside(7)), "has 7 px that are NOT in the preview's saved part", "Nothing is kept")
        assert not (out / "Video/v/parts/shot_c_0100/shot_c_0100_b_part_hair.mkv").exists() and not list(made.jobs.iterdir()), "a part with pixels outside the saved part was kept"
        cuts.clear(), held.clear()
        stl.step_job(a, made.m, cut=cut, inside=inside(0))
        assert cuts == [("shot_c_0100_b_classes_00001.mkv", ("Hair",), "shot_c_0100_b_part_hair.mkv")] and held == [("shot_c_0100_b_part_hair.mkv", "shot_c_0100_b_parts_00001.mkv")], (cuts, held)
        record = json.loads((out / "Video/v/parts/shot_c_0100/shot_c_0100_b_part_hair.json").read_text())
        assert record["parts"] == ["hair"] and record["classes"] == ["Hair"] and record["of"] == "b" and record["px_outside_the_saved_part"] == 0 and record["first_source_frame"] == 100, record
        kept = json.loads((made.jobs / "b_hp_0100.json").read_text())
        assert kept["job"]["part"]["parts"] == ["hair"] and next(x for x in kept["job"]["subjects"] if x["label"] == "b")["parts_file"].endswith("_b_part_hair.mkv")
        assert not (made.jobs / "b_hfp_0100.json").exists(), "a load of the hair alone was named as face and hair"
        # a second load on the same preview with the same parts: the file is there, cut again and held equal; not equal is refused
        a.remake = True
        stl.step_job(a, made.m, cut=cut, inside=inside(0), hashed=lambda path: "the same")
        assert cuts[-1][2].endswith(".cut_again.mkv") and (made.jobs / "b_hp_0100b.json").is_file() and not list((out / "Video/v/parts/shot_c_0100").glob("*cut_again*")), cuts
        refused(lambda: stl.step_job(a, made.m, cut=cut, inside=inside(0), hashed=lambda path: path), "is there and is NOT what its classes cut", "a part of another making")
        assert not (made.jobs / "b_hp_0100c.json").exists()
        # a load of the same subject with no --parts is as it ever was, on the same preview
        a.parts, a.parts_by, a.remake = None, None, False
        stl.step_job(a, made.m, cut=cut, inside=inside(0))
        usual = json.loads((made.jobs / "b_hfp_0100.json").read_text())
        assert "part" not in usual["job"] and not any("parts_file" in x for x in usual["job"]["subjects"]), "a load with no --parts carries a part of its own"
        # THE CUT AND THE HOLD THEMSELVES, on a class map and a part written as the nodes write them
        import capture_masked_run as cap
        names = cap._pack("sapiens2_parts").CLASS_NAMES
        cmap = np.zeros((3, 48, 64), np.uint8)
        cmap[:, 4:20, 4:30] = names.index("Hair")
        cmap[:, 20:34, 8:26] = names.index("Face_Neck")
        cmap[1, 36:44, 40:60] = names.index("Upper_Clothing")
        video, whole, hair = (Path(tmp) / n for n in ("classes.mkv", "parts.mkv", "hair.mkv"))
        enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                              str(video)], input=cmap.tobytes())
        assert enc.returncode == 0 and (cap.read_classes(str(video), (64, 48), 0, 0, 3) == cmap).all(), "the class map did not come back as written"
        table = stl.part_classes()
        both = np.isin(cmap, [names.index(c) for k in ("hair", "face_and_neck") for c in table[k]])
        cap.write_mask_video(whole, both)                                  # the preview's saved part: the plan's two parts, as the part node saves them
        got = stl.cut_keep_out(video, list(table["hair"]), hair)
        back = read_mask(str(hair), (64, 48), 0, 0, 3)[0]
        assert (back == (cmap == names.index("Hair"))).all() and not back[:, 20:34, 8:26].any() and got["px_most"] == 16 * 26, got
        assert stl.part_inside(hair, whole) == 0, "the hair cut from the class map is not inside the part of hair and face"
        # a cut of a class the saved part does not hold is no part of it: every one of its pixels is outside
        clothes = Path(tmp) / "clothes.mkv"
        stl.cut_keep_out(video, ["Upper_Clothing"], clothes)
        assert stl.part_inside(clothes, whole) == 8 * 20, stl.part_inside(clothes, whole)
    return ("a load may name some of its subject's plan parts: the job names the file, its classes and what it is held to, the subject carries `parts_file`, the "
            "name says the parts; the step cuts it, holds it inside the preview's saved part and to a fresh cut; a load that names none is as it was")


def the_text_step() -> str:
    """The text's writer had to be set in the job file by hand after the chain refused it (the first real load, twice)."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        job_file = made.jobs / "a_fp_0100.json"
        job_file.write_text(json.dumps(made.load(), indent=1))
        a = args(job_file=str(job_file), writer="a writer, from frames 100 to 159   looked at every fifth")
        refused(lambda: stl.step_text(a, made.m), "is not written, or is empty")
        (made.jobs / "a_fp_0100.txt").write_text("the words of the load\n")
        refused(lambda: stl.step_text(args(job_file=str(job_file), writer="  "), made.m), "--writer takes a person's words")
        refused(lambda: lc.read_job(job_file, made.root), "texts.a_fp_0100.by is empty", "bench/shot_to_load.py text")
        before = job_file.read_bytes()
        made.m.dry = True
        stl.step_text(a, made.m)
        assert job_file.read_bytes() == before, "a dry run wrote the job file"
        made.m.dry = False
        stl.step_text(a, made.m)
        d = json.loads(job_file.read_text())
        by = d["job"]["texts"]["a_fp_0100"]["by"]
        assert by.startswith("a writer, from frames 100 to 159 looked at every fifth; entered 20") and by.endswith("by the session (bench/shot_to_load.py text)"), by
        was = json.loads(before)
        was["job"]["texts"]["a_fp_0100"]["by"] = by
        assert d == was, "the text step changed more of the job file than the writer"
        assert lc.read_job(job_file, made.root)["text"] == made.jobs / "a_fp_0100.txt"
        masked_job.text_entry(d["job"], "a_fp_0100", made.jobs)
        # once the plan is built the job file is the plan's, and is not edited
        folder = made.root / "data" / "2026-01-03_a_fp_0100_job"
        folder.mkdir()
        (folder / "job.plan.json").write_bytes(job_file.read_bytes())
        kept = job_file.read_bytes()
        refused(lambda: stl.step_text(args(job_file=str(job_file), writer="another"), made.m), "the plan of a_fp_0100 is built", "not edited after")
        assert job_file.read_bytes() == kept
    return ("the text's writer is said with a flag and goes into the job file with when and by which session, nothing else changed; the chain then reads the "
            "load; refused with no text, with no words, and once the plan is built; a dry run writes nothing")


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
                  "made_at": "2026-01-03T09:00:00Z", "made_by": "the first session",
                  "face_on_the_track": {"known": True, "frames_whose_face_middle_is_off_the_track": []}}
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
            # the step's output is kept beside the job file: the load's job folder is the chain's to make, at `first`
            assert made.m.kept_in["pose pass"] == made.m.kept_in["head reference"] == made.jobs / "chain", made.m.kept_in
        finally:
            lc.REPO = kept
        # the card is freed before the pose pass, which runs in its own process beside the server
        assert made.m.labels() == ["free the card", "pose pass", "head reference"], made.m.labels()
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
        # one made before its points were held to the track says nothing of it, or not known: made again
        for track in (None, {"known": False, "why": "says nothing of a track", "frames_whose_face_middle_is_off_the_track": []}):
            drawn.with_suffix(".json").write_text(json.dumps({**{k: v for k, v in record.items() if k != "face_on_the_track"},
                                                              **({"face_on_the_track": track} if track else {})}))
            refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "held to a's track", "made again")
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
case("a preview made to ask who is on a shot the record is unsure of", a_preview_made_to_ask)
case("a box in the cast file seeds its tracker", a_box_in_the_cast_file_seeds)
case("a shot the tracker cuts takes a seed a cut", a_shot_the_tracker_cuts_takes_a_seed_a_cut)
case("the plan says what a subject's face is taken from", the_plan_says_what_from)
case("preview: the job file, once, queued only when told", the_preview_step)
case("preview against a builder that refuses a folder that is there: the first real run's fault", the_preview_step_against_a_builder_that_refuses_a_folder_that_is_there)
case("a load's name is made, and refused when taken", a_loads_name_is_made)
case("a load's job file stands alone", a_loads_job_file_stands_alone)
case("a load the builder would refuse is not written, in a dry run as well", a_load_the_builder_would_refuse_is_not_written)
case("a subject's keep-outs: named by the plan, made and named by the job step", a_subjects_keep_outs)
case("a load of some of its subject's parts", a_load_of_some_of_its_parts)
case("text: who wrote it, said with a flag", the_text_step)
case("a job renders from the video's present copy", a_job_renders_from_the_present_copy)
case("reference: on the load's frames, drawn by default", the_reference_step)
case("the figures, at the canvas and in the reference", the_figures)

if __name__ == "__main__":
    sys.exit(finish())
