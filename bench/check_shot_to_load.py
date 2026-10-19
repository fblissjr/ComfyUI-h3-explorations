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
   chain reads it, and onto no other subject, refused when it is not a value the capture takes or names nobody, and
   refused on a load whose part holds a class of hair or face_and_neck on any frame (a part of the mouth alone too), by
   the plan's parts, `--parts` or a `--parts-on` range (`h3_config.NO_PIECES_ON`, the owner, 2026-10-16; seen red with
   hair taken out of it and with the ranges not read); `--zoomed`
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
import shot_to_load as stl
import strangers as strangers_tool  # noqa: E402


class Machine:
    """The machine a step is handed: an empty queue unless told, and a `run` that records."""
    def __init__(self, root: Path):
        self.host, self.dry, self.python, self.pose_model = "nowhere:0", False, "PY", "pose-model"
        self.output, self.inputs, self.data = root / "output", root / "input", root / "data"
        self.busy, self.ran, self.on, self.kept_in = (0, 0), [], {}, {}

    def queue(self):
        return self.busy

    def voice(self, source_name: str) -> dict:
        """The video's voice tables, as `load_chain`'s machine finds them; a check's own, written here."""
        spans = self.data / "voice_spans.json"
        spans.parent.mkdir(parents=True, exist_ok=True)
        spans.write_text(json.dumps({"voiced_spans_inclusive": [[120, 130]]}))
        return {"per_frame": str(self.data / "voice_per_frame.csv"), "spans": str(spans), "frames": [0, 999]}

    def free(self):
        self.ran.append(("free the card", []))
        return "the card after it: free."

    def run(self, ctx, label, argv, stop=True):
        # what load_chain.Machine.run reads of the context, read here the same way: a step that hands it less fails here as it
        # failed on the card (the job step's strangers pass, 2026-10-14: no `name`)
        _ = (Path(ctx["logs"]) if ctx.get("logs") else ctx["job_folder"] / "chain") / f"{ctx['name']}.{label.replace(' ', '_')}.out"
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
                 existing=False, writer=None, why_before=None, why_after=None, zoomed=None, least=None, least_why=None, ask=[], seed_cuts_of=None, parts=None, parts_by=None, parts_on=[], whole_frame=None, mouth=None, iris=None, form_why="",
                 place_and_action=None, by_hand=None, plan=None,
                 track_not=[], track_not_by=None, drop=None, drop_by=None, leave_word=[], leave_by=None, carry_words_from=None, no_carry_words=None,
                 head_tube=None, depth=None)
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
        # THE CASE THAT GOES RED when the job step's flags do not reach the load's making (`--no-face-on` first went in by hand)
        given, real = {}, stl.load_job
        try:
            stl.load_job = lambda *x, **k: given.update(k) or (_ for _ in ()).throw(stl.Refused("STOPPED HERE"))
            refused(lambda: stl.step_job(args(**{**vars(a), "no_face_on": ["120-130=A WORD"], "no_mouth_on": ["121=B"], "seam": "2344,2548",
                                                 "hold_part": ["3098-3099=C"]}), made.m), "STOPPED HERE")
        finally:
            stl.load_job = real
        assert given["no_face_on"] == [(120, 130, "A WORD")] and given["no_mouth_on"] == [(121, 121, "B")] and given["seam"] == [2344, 2548] \
            and given["hold_part"] == [(3098, 3099, "C")], given
        stl.step_job(a, made.m)
        assert (made.jobs / "a_fp_0100.json").is_file(), sorted(p.name for p in made.jobs.iterdir())
        refused(lambda: stl.step_job(a, made.m), "a load named a_fp_0100 exists", "--remake")
        a.remake = True
        refused(lambda: stl.step_job(a, made.m), "a_fp_0100", "--carry-words-from", "--no-carry-words")
        a.no_carry_words = "fresh remake for name test"
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
        # NO PIECES ON A FACE OR HAIR (the owner, 2026-10-16): a load whose part holds either, on any frame, is refused, whoever wrote the key
        refused(made.load, "`pieces=largest` for 'a' on a load whose part holds face_and_neck", "Take `pieces` and `pieces_by` off")
        band = made.plan["subjects"]["b"].pop("hair_band")
        made.plan["subjects"]["b"].update(pieces="largest", pieces_by="a person")
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by="a person: her hair alone"), "on a load whose part holds hair:")
        del made.plan["subjects"]["b"]["pieces"], made.plan["subjects"]["b"]["pieces_by"]
        made.plan["subjects"]["b"]["hair_band"] = band
        a_parts = made.plan["subjects"]["a"]["parts"]
        made.plan["subjects"]["a"]["parts"] = ["upper_clothing", "face_and_neck"]
        refused(lambda: made.load(name="a_up_0100", parts=["upper_clothing"], parts_by="a person", parts_on=[(110, 115, ["face_and_neck"])]),
                "on a load whose part holds face_and_neck + upper_clothing:")
        made.plan["subjects"]["a"]["parts"] = ["mouth"]          # a face by its classes, though not named face_and_neck
        refused(made.load, "on a load whose part holds mouth:")
        made.plan["subjects"]["a"]["parts"] = ["upper_clothing"]
        pieced = made.load()
        made.plan["subjects"]["a"]["parts"] = a_parts
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
        # ZOOMED IN BY DEFAULT (the owner's word of 2026-10-14), the node deciding at plan; the whole frame on a word, the load's or the plan's
        assert load["render_region"] == "part" and job["zoomed"]["said_by"] == "the chain's default" and "whole_frame" not in job, (load, job.get("zoomed"))
        assert "RENDERED ZOOMED IN" in job["note"] and "the chain's default" in job["note"], job["note"]
        whole = made.load(whole_frame="the shot is a crowd and her face fills a third of the frame")
        assert "render_region" not in whole["loads"][0] and whole["job"]["whole_frame"]["why"].startswith("the shot is a crowd") and "zoomed" not in whole["job"], whole["job"]
        assert "RENDERED ON THE WHOLE FRAME, not zoomed in as the chain's default, on the session's word" in whole["job"]["note"], whole["job"]["note"]
        refused(lambda: made.load(whole_frame=" "), "--whole-frame takes the reason in words")
        refused(lambda: made.load(whole_frame="x", zoomed="y"), "one load is rendered one way")
        made.plan["subjects"]["a"].update(zoom=False, zoom_by="the owner: this subject's loads stay on the whole frame")
        planned_whole = made.load()
        assert "render_region" not in planned_whole["loads"][0] and planned_whole["job"]["whole_frame"]["said_by"] == "the plan", planned_whole["job"]
        made.plan["subjects"]["a"]["zoom_by"] = " "
        refused(made.load, "`zoom: false` for 'a' with no `zoom_by`")
        del made.plan["subjects"]["a"]["zoom"], made.plan["subjects"]["a"]["zoom_by"]
        # THE CAPTURE'S FLOOR is a choice with its words and no default
        assert all("least" not in s for s in job["subjects"]), "a load is floored with nobody saying so"
        z = made.load(zoomed="her face is  40 px tall on the whole frame", least=600, least_why="the part is drawn four times its whole-frame size")
        zl, zs = z["loads"][0], next(s for s in z["job"]["subjects"] if s["label"] == "a")
        assert zl["render_region"] == "part" and z["job"]["zoomed"] == {"why": "her face is 40 px tall on the whole frame", "said_by": "the session", "at": "now"}, z["job"].get("zoomed")
        assert (zs["least"], zs["least_by"]) == (600, "the session, now: the part is drawn four times its whole-frame size"), zs
        assert all("least" not in s for s in z["job"]["subjects"] if s["label"] != "a"), "another subject's floor was set"
        assert "RENDERED ZOOMED IN on what is replaced (`render_region: part`) where the node's box allows, on the session's word: her face is 40 px tall" in z["job"]["note"] \
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
        a.no_carry_words = "remake on same preview"
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



def her_clothing_as_one_stay_keep_out() -> str:
    """The lead's ruling of 2026-10-14 (K, build path (a)) on mrpre's apron read: the part model calls the second subject's white
    apron Background, so the clothes keep-out missed it. A plan look's stay item makes ONE keep-out in place of `clothes`: her
    track's Upper_Clothing, Apparel and Background, less the hair band's box (the builder's own), held over 9 frames."""
    import numpy as np
    import capture_masked_run as cap
    import masked_job
    from _lib.frames import read_mask, write_mask_video
    STAY = {"what": "her clothing", "as": "one keep-out", "replaces_keep_out": "clothes", "classes_on_her_track": ["Upper_Clothing", "Apparel", "Background"],
            "hold_frames": 4}
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        made.plan["subjects"]["a"].update(keep_outs=["held", "hands", "clothes"], keep_outs_by="a person: her apron", hair_band={"value": 0.6, "by": "a person"},
                                          look={"stay": [STAY]}, look_by="THE LEAD'S RULING K")
        # RETIRED (26f208c0): THE CASE THAT GOES RED when a stay keep-out drawn on her own track is still made
        refused(made.load, "drawn on the subject's own track", "26f208c0")
        # HER CLOTHING REDRAWN WITH HER IS NAMED (load_text.clothing_of): a subject whose stay was removed and whose clothing the plan does
        # not name is refused at the job step, before anything is made
        kept_look = made.plan["subjects"]["a"].pop("look")
        made.plan["subjects"]["a"]["look"] = {"stay_removed_by": "the owner chose to redraw her clothing"}
        # THE CASE THAT GOES RED when such a job is written
        refused(made.load, "her clothing is redrawn with her and named in the text; give the plan's `clothing`")
        made.plan["subjects"]["a"]["clothing"] = [{"first": 0, "last": 99, "words": "AN APRON", "by": "a person, from frame 50"}]
        # THE CASE THAT GOES RED when the job takes another shot's words for this load's frames
        refused(made.load, "names no words for frames", "never another shot's")
        made.plan["subjects"]["a"]["clothing"] = [{"first": 0, "last": 10000, "words": "AN APRON", "by": "a person, from frame 100"}]
        assert made.load()["loads"][0]["subject"] == "a", "a subject whose clothing the plan names for these frames was refused"
        # A LOAD CUT INSIDE ITS SHOT (the clothing read): the shot 100-159 straddles entries split at 130, the load 130-149 lies in one
        made.plan["subjects"]["a"]["clothing"] = [{"first": 100, "last": 129, "words": "AN APRON", "by": "a person, from frame 110"},
                                                  {"first": 130, "last": 159, "words": "A DARK TOP", "by": "a person, from frame 140"}]
        # THE CASE THAT GOES RED when the job step reads the shot's frames in place of the load's
        assert made.load(first=130, frames=20, why="a look cut it")["loads"][0]["first"] == 130, "a load whose own frames one entry holds was refused"
        del made.plan["subjects"]["a"]["clothing"]
        made.plan["subjects"]["a"]["look"] = kept_look
        folder = made.preview_ran(made.preview())
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        refused(lambda: stl.step_job(a, made.m, cut=lambda *_a: {}, stay_cut=lambda *_a: {}, hashed=lambda path: "H"), "26f208c0")
        assert not list(made.jobs.glob("*.json")), "a job was written with a stay keep-out"
    # THE KEEP-OUT ITSELF against the definition, written out here apart from the tool (mrpre's arm K): 6 preview frames from 100, the load 101-104
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    W_, H_, N_ = 64, 48, 6
    track = np.zeros((N_, H_, W_), bool)
    track[:, 10:40, 10:40] = True
    cls = np.zeros((N_, H_, W_), np.uint8)                       # Background on her track is her apron
    cls[:, 10:20, 10:40] = names.index("Upper_Clothing")
    cls[:, 30:40, 10:40] = names.index("Torso")                  # not clothing
    cls[3, 20:30, 10:40] = names.index("Torso")                  # one frame the apron reads otherwise: the hold keeps it
    part = np.zeros((N_, H_, W_), bool)
    part[:, 2:9, 20:30] = True                                   # the head above her: its band runs down into her blouse
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        write_mask_video(t / "track.mkv", track)
        write_mask_video(t / "part.mkv", part)
        enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W_}x{H_}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                              "-level", "3", str(t / "classes.mkv")], input=cls.tobytes())
        assert enc.returncode == 0
        stay = {"classes_on_her_track": STAY["classes_on_her_track"], "hold_frames": 1, "track": str(t / "track.mkv"), "class_map": str(t / "classes.mkv"),
                "preview_first": 100, "preview_frames": N_, "first": 101, "frames": 4,
                "less_the_hair_band": {"share": 0.6, "side_px": 2.0, "part": str(t / "part.mkv"), "part_first_source_frame": 100}}
        got = stl.cut_stay(stay, t / "out" / "clothing.mkv")
        back = read_mask(str(t / "out" / "clothing.mkv"), (W_, H_), 100, 100, N_)[0]
        a = track[1:5] & np.isin(cls[1:5], [names.index(c) for c in STAY["classes_on_her_track"]])
        bands, _ = masked_job.band_boxes(str(t / "part.mkv"), 100, 101, 4, (W_, H_), 0.6, 2.0)
        for f in range(4):
            x0, y0, x1, y1 = bands[f]
            a[f, y0:y1, x0:x1] = False
        held = np.stack([a[max(f - 1, 0):f + 2].sum(0) * 2 > len(a[max(f - 1, 0):f + 2]) for f in range(4)])
        want = np.zeros((N_, H_, W_), bool)
        want[1:5] = held
        assert (back == want).all(), "the stay keep-out is not her track's clothing less the band, held over frames, on the load's frames only"
        assert back[2, 20:30, 10:40].any() and not back[:, 30:40].any() and not back[0].any() and not back[5].any(), "the apron, the torso or the frames outside"
        assert bands[0, 0] >= 0 and not back[1, bands[0, 1]:bands[0, 3], bands[0, 0]:bands[0, 2]].any() and got["px_the_band_took_out"] > 0, got
        assert got["px_the_hold_changed"] > 0, got
    return ("a plan look's stay item is refused, retired (26f208c0: it is drawn on her own track); the cut itself, kept for the record, is her track's "
            "clothing classes, Background included, less the band, held over frames, on the load's frames only")


def who_else_is_in_the_shot() -> str:
    """The owner's word of 2026-10-14: "are there people in the shot who are not ours" belongs in the job file. The job step
    reads it from the preview's own tables and the owner's answers (bench/others_present.py), prints it among the defaults,
    and keeps the strangers out by it: where someone else is there or the frames shown cannot say, and on no load where nobody
    is, whatever the plan's key; the plan's key only where it is not known."""
    def load_with(made, people, plan_keep=None, not_people=None, no_strangers=None, no_mouth_on=None, eyes_shut_on=None, eyes_shut_form=None,
                  seam=None, seam_why=None, no_face_on=None, hold_part=None, context_noise=None, context_noise_why=None, mouth_on=None, parts_on=None):
        if plan_keep is not None:
            made.plan["subjects"]["a"].update(keep_outs=plan_keep, keep_outs_by="a person: the plan's own")
        if not_people is not None:
            made.plan["not_people"] = not_people
        shots, plan = made.read()
        pjob = made.preview(100)
        folder = made.preview_ran(pjob)
        out = made.m.output / pjob["job"]["output"] / pjob["job"]["name"]
        for label, pick in (("a", 1), ("b", 2), ("c", 3)):
            (out / f"{pjob['job']['name']}_{label}_shots_00001_shots.json").write_text(json.dumps({"people_are": "every detection", "shots": [
                {"first_frame": 0, "last_frame": 59, "shown_frame": 20, "subject": {"state": "seeded", "person": pick},
                 "people": [{"person": n, "box": box, "share_of_frame": share, "held_by_another_tracker": False} for n, box, share in people]}]}))
        saved = json.loads((folder / "preview.queued.json").read_text())["files"]
        return stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(folder), "v", made.m.output, "the session", "2026-01-03", "a_fp_0100",
                            now="now", preview_files=saved, cast=[], cast_from="cast.jsonl", no_strangers=no_strangers, no_mouth_on=no_mouth_on,
                            eyes_shut_on=eyes_shut_on, eyes_shut_form=eyes_shut_form, seam=seam, seam_why=seam_why, no_face_on=no_face_on, hold_part=hold_part,
                            context_noise=context_noise, context_noise_why=context_noise_why, mouth_on=mouth_on,
                            **({"parts_on": parts_on, "parts_by": "a person"} if parts_on else {}))
    ours = [(1, [0, 0, 10, 10], 0.1), (2, [20, 0, 10, 10], 0.1), (3, [40, 0, 10, 10], 0.1)]
    with tempfile.TemporaryDirectory() as tmp:
        d = load_with(Made(Path(tmp)), ours + [(4, [55, 30, 5, 5], 0.01)])
        job = d["job"]
        lines = {x["what"]: x for x in job["defaults"]}
        # THE CASE THAT GOES RED when someone else in the shot is not in the job file, or the strangers are not kept out by it
        assert job["others_present"]["others_present"] == "yes" and job["others_present"]["frames_with_others"] == [120], job["others_present"]
        assert [k["name"] for k in job["keep_outs"]["made"]] == ["strangers"] and "strangers" in d["loads"][0]["others"], job.get("keep_outs")
        assert lines["other people in the shot"]["carries"].startswith("yes: someone not ours on 1 of the 1 frame(s)") and \
            lines["the strangers keep-out"]["from"].startswith("the data: who else is in the shot, yes"), lines
        assert job["others_present"]["owners_answers"] == "read from cast.jsonl"
        # NO STRANGERS KEEP-OUT on a person's word (--no-strangers, the lead's ruling of 2026-10-15): the word names its contact sheet,
        # and here, where the record has seen someone else on 120, that frame and why it does not matter
        sheet = Path(tmp) / "sheet.jpg"
        sheet.write_bytes(b"a sheet")
        refused(lambda: load_with(Made(Path(tmp) / "w1"), ours + [(4, [55, 30, 5, 5], 0.01)], no_strangers="the lead looked: nobody else"),
                "name its file")
        refused(lambda: load_with(Made(Path(tmp) / "w2"), ours + [(4, [55, 30, 5, 5], 0.01)], no_strangers=f"the lead looked at {sheet}: nobody else"),
                "the record has seen someone else (frame(s) 120)")
        off = load_with(Made(Path(tmp) / "w3"), ours + [(4, [55, 30, 5, 5], 0.01)],
                        no_strangers=f"THE LEAD looked at every frame on {sheet}: nobody else; the box seen on 120. is a lamp")
        rows_ = {x["what"]: x for x in off["job"]["defaults"]}
        # THE CASE THAT GOES RED when the strangers stay kept out on a person's word, or the defaults do not say it was turned off and why
        assert "keep_outs" not in off["job"] and "strangers" not in off["loads"][0]["others"], off["job"].get("keep_outs")
        assert rows_["the strangers keep-out"]["carries"].startswith("not kept out, turned off on a person's word; the part still held") \
            and rows_["the strangers keep-out"]["from"].startswith("TURNED OFF on a person's word") and "THE LEAD looked" in rows_["the strangers keep-out"]["from"], \
            rows_["the strangers keep-out"]
        # THE CASE THAT GOES RED when turning the strangers off also lets her part spill past her own track (the lead, 2026-10-15)
        assert off["job"]["part_on_track"]["why"].startswith("the strangers keep-out is turned off on a person's word") \
            and "though the strangers keep-out is turned off" in off["job"]["note"], off["job"].get("part_on_track")
        # a plan with another keep-out beside the strangers: its reason is the plan's, without the data's strangers clause
        both = load_with(Made(Path(tmp) / "w4"), ours + [(4, [55, 30, 5, 5], 0.01)], plan_keep=["held", "strangers"],
                         no_strangers=f"THE LEAD looked at {sheet}; 120 is a lamp")
        assert [k["name"] for k in both["job"]["keep_outs"]["made"]] == ["held"] and "strangers" not in both["job"]["keep_outs"]["by"], both["job"]["keep_outs"]["by"]
        # NO MOUTH ON A SPAN on a person's word (--no-mouth-on, the lead's word of 2026-10-15 on 1010h): the word names its sheet, the span
        # lies inside the load, and the reference step passes it to head_reference
        refused(lambda: load_with(Made(Path(tmp) / "m1"), ours, no_mouth_on=[(110, 115, "THE LEAD looked: her hand over her mouth")]), "name its file")
        refused(lambda: load_with(Made(Path(tmp) / "m2"), ours, no_mouth_on=[(90, 115, f"THE LEAD looked at {sheet}")]), "the load is 100-159")
        refused(lambda: load_with(Made(Path(tmp) / "m4"), ours, no_mouth_on=[(150, 170, f"THE LEAD looked at {sheet}")]), "the load is 100-159")
        hid = load_with(Made(Path(tmp) / "m3"), ours, no_mouth_on=[(110, 115, f"THE LEAD looked at {sheet}: her hand over her mouth")])
        mouth_row = {x["what"]: x for x in hid["job"]["defaults"]}["the mouth"]
        # THE CASE THAT GOES RED when a person's no-mouth span is not kept in the job, not said among the defaults, or not passed on
        assert hid["job"]["reference"]["mouth"]["no_mouth_on"][0]["first"] == 110 and "NONE on 110-115" in mouth_row["carries"] \
            and "her hand over her mouth" in mouth_row["from"], (hid["job"]["reference"]["mouth"], mouth_row)
        flags = stl.drawn_form(hid["job"], args(), {"classes": Path("/c.mkv")}, 100)
        assert flags[flags.index("--no-mouth-on") + 1].startswith("110-115=a person's word on a look (the session, now): THE LEAD looked"), flags
        # NO LINE OF THE FACE ON A SPAN on a person's word (--no-face-on, the lead's ruling of 2026-10-15 on the hall shot): the same terms
        refused(lambda: load_with(Made(Path(tmp) / "n1"), ours, no_face_on=[(110, 115, "THE LEAD looked: his back to the camera")]), "name its file")
        refused(lambda: load_with(Made(Path(tmp) / "n2"), ours, no_face_on=[(150, 170, f"THE LEAD looked at {sheet}")]), "--no-face-on 150-170", "the load is 100-159")
        away = load_with(Made(Path(tmp) / "n3"), ours, no_face_on=[(120, 130, f"THE LEAD looked at {sheet}: his back to the camera")])
        face_row = {x["what"]: x for x in away["job"]["defaults"]}["the face lines"]
        # THE CASE THAT GOES RED when a person's no-face span is not kept in the job, not said among the defaults, or not passed on
        assert away["job"]["reference"]["no_face_on"][0]["first"] == 120 and "NONE on 120-130" in face_row["carries"] \
            and "his back to the camera" in face_row["from"], (away["job"]["reference"], face_row)
        refused(lambda: load_with(Made(Path(tmp) / "n4"), ours, no_face_on=[(120, 130, f"THE LEAD looked at {sheet}")],
                                  eyes_shut_on=[(110, 125, f"THE LEAD looked at {sheet}")], eyes_shut_form="none"), "both name 120-125")
        flags = stl.drawn_form(away["job"], args(), {"classes": Path("/c.mkv")}, 100)
        assert flags[flags.index("--no-face-on") + 1].startswith("120-130=a person's word on a look (the session, now): THE LEAD looked"), flags
        # A MOUTH ON A SPAN on a person's word (--mouth-on, the lead's ruling of 2026-10-15 on lead_fp_2823c): the same terms, and never
        # against a span the job leaves with no mouth or no face
        refused(lambda: load_with(Made(Path(tmp) / "o1"), ours, mouth_on=[(110, 115, "THE LEAD looked: her mouth open")]), "name its file")
        refused(lambda: load_with(Made(Path(tmp) / "o2"), ours, mouth_on=[(150, 170, f"THE LEAD looked at {sheet}")]), "--mouth-on 150-170", "the load is 100-159")
        refused(lambda: load_with(Made(Path(tmp) / "o3"), ours, mouth_on=[(110, 115, f"THE LEAD looked at {sheet}")],
                                  no_face_on=[(114, 120, f"THE LEAD looked at {sheet}")]), "110-115 against --no-face-on 114-120", "draws none")
        refused(lambda: load_with(Made(Path(tmp) / "o4"), ours, mouth_on=[(110, 115, f"THE LEAD looked at {sheet}")],
                                  no_mouth_on=[(100, 110, f"THE LEAD looked at {sheet}")]), "against --no-mouth-on 100-110")
        # a mouth form other than the check's has no gate to override (the cold reader of 2026-10-15)
        drawn_mouth = Made(Path(tmp) / "o7")
        drawn_mouth.plan["subjects"]["a"]["reference"] = {"mouth": "drawn", "why": "a person's word"}
        refused(lambda: load_with(drawn_mouth, ours, mouth_on=[(110, 115, f"THE LEAD looked at {sheet}")]), "this load's mouth is 'drawn'", "no gate to override")
        sang = load_with(Made(Path(tmp) / "o5"), ours, mouth_on=[(110, 115, f"THE LEAD looked at {sheet}: her mouth open, singing")])
        mouth_row = {x["what"]: x for x in sang["job"]["defaults"]}["the mouth"]
        # THE CASE THAT GOES RED when a person's mouth span is not kept in the job, not said among the defaults, or not passed on
        assert sang["job"]["reference"]["mouth"]["mouth_on"][0]["first"] == 110 and "DRAWN on 110-115" in mouth_row["carries"] \
            and "her mouth open, singing" in mouth_row["from"], (sang["job"]["reference"]["mouth"], mouth_row)
        flags = stl.drawn_form(sang["job"], args(), {"classes": Path("/c.mkv")}, 100)
        assert flags[flags.index("--mouth-on") + 1].startswith("110-115=a person's word on a look (the session, now): THE LEAD looked"), flags
        assert "the face lines" not in {x["what"] for x in hid["job"]["defaults"]} and "--no-face-on" not in stl.drawn_form(hid["job"], args(), {"classes": Path("/c.mkv")}, 100)
        # THE PART HELD ON A PERSON'S WORD (--hold-part, the lead's ruling of 2026-10-15): the same terms; on the subject as the capture's word=
        refused(lambda: load_with(Made(Path(tmp) / "h1"), ours, hold_part=[(110, 115, "THE LEAD looked: her face is there")]), "name its file")
        refused(lambda: load_with(Made(Path(tmp) / "h2"), ours, hold_part=[(150, 170, f"THE LEAD looked at {sheet}")]), "--hold-part 150-170", "the load is 100-159")
        refused(lambda: load_with(Made(Path(tmp) / "h4"), ours, hold_part=[(110, 115, f"A looked at {sheet}"), (115, 118, f"B looked at {sheet}")]),
                "--hold-part 115-118 overlaps")
        kept_on = load_with(Made(Path(tmp) / "h3"), ours, hold_part=[(110, 111, f"THE LEAD looked at {sheet}"), (120, 120, f"G2 looked at {sheet}")])
        mine = next(x for x in kept_on["job"]["subjects"] if x["label"] == "a")
        held_row = {x["what"]: x for x in kept_on["job"]["defaults"]}["the part held on a person's word"]
        # THE CASE THAT GOES RED when the word is not on the subject in the capture's form, not said among the defaults, or not handed on
        assert mine["word"] == "110-111+120-120", mine
        assert "110-111: the session, now: THE LEAD looked" in mine["word_by"] and "on 110-111, 120" in held_row["carries"], (mine, held_row)
        assert "word" not in next(x for x in away["job"]["subjects"] if x["label"] == "a") \
            and "the part held on a person's word" not in {x["what"] for x in away["job"]["defaults"]}
        assert stl.no_mouth_on_arg("1011=X") == (1011, 1011, "X")
        refused(lambda: stl.no_mouth_on_arg("1011-1018"), "with the why")
        refused(lambda: stl.no_mouth_on_arg("1011-1017-1030=X"), "is not FIRST-LAST=WHY")
        # THE SEAMS (the owner's words of 2026-10-15: one load a shot, the seam at a frame chosen, and "you can do shorter windows too just
        # make sure the frame multiples we documented ... are right to avoid seams"): every plan of the planner's chained lengths
        reach = stl.seam_window(406, 90)
        # the planner's longest chained window, derived from the seam rule and never typed (`check_length_ceiling`)
        import masked_job
        L = max(masked_job.chain_lengths())
        # THE CASE THAT GOES RED when the seams offered are not the song node's planner's, or a one-window load is said to have one
        assert {k: v[0] for k, v in reach.items()} == {(141, 192, 243, 294, L, 396): 141, (192, 294, 396): 192, (243, 396): 243, (294,): 294, (L,): L} \
            and reach[(L,)][1] == [L, 192] and reach[(243, 396)][1] == [243, 243, 141], reach
        # each later seam is the one before it plus the length of the window between them less the context it shares
        assert {k: v[0] for k, v in stl.seam_window(573, 90).items()}[(294, 498)] == 294
        # one window holds a load up to the ceiling (2026-10-15), so a load at the ceiling has no seam
        assert stl.seam_window(stl.h3_config.MAX_LENGTH, 90) == {} and stl.seam_window(60, 90) == {}
        assert stl.seam_of(stl.h3_config.MAX_LENGTH, 100, None, None, "s", "n") == (None, None)
        refused(lambda: load_with(Made(Path(tmp) / "s1"), ours, seam=[110], seam_why="A CALM STRETCH"), "one window holds this load's 60 frames")
        refused(lambda: load_with(Made(Path(tmp) / "s2"), ours, seam=[110]), "go together")
        window, row = stl.seam_of(406, 604, [898], "A CALM STRETCH", "the session", "now")
        # THE CASE THAT GOES RED when a chosen seam does not set the window that puts it there, or the defaults do not say it and its kept frames
        assert window == 294 and row["carries"] == ("windows of [294, 243]: window 2 writes from 898, sampled from the frames 808-897 before it; "
                                                    "the last window writes 112 new frame(s) of the load") \
            and row["from"] == "the session, now: A CALM STRETCH", (window, row)
        # three windows: every seam named, in any order, each with the frames it is sampled from
        window, row = stl.seam_of(573, 2050, [2548, 2344], "TWO CALM STRETCHES", "the session", "now")
        assert window == 294 and row["what"] == "the seam" and row["carries"] == ("windows of [294, 294, 192]: window 2 writes from 2344, sampled from the "
                                                                                  "frames 2254-2343 before it; window 3 writes from 2548, sampled from the frames "
                                                                                  "2458-2547 before it; the last window writes 75 new frame(s) of the load"), (window, row)
        window, row = stl.seam_of(406, 604, None, None, "the session", "now")
        assert window is None and "window 2 writes from 949" in row["carries"] and row["from"].startswith("the planner"), row
        # a seam no plan puts there, and some of one plan's seams but not all, are refused with the plans there are
        refused(lambda: stl.seam_of(406, 604, [900], "X", "s", "n"), f"seams on 949 (windows of {L}); or 898 (windows of 294) only")
        # a plan the last window's floor refuses is not offered (406's 243, 192 and 141 plans each write 10 new frames last)
        try:
            stl.seam_of(406, 604, [900], "X", "s", "n")
        except stl.Refused as e:
            assert "1000" not in str(e), e
        refused(lambda: stl.seams_arg("\u00b2"), "comma-separated")
        refused(lambda: stl.seam_of(573, 2050, [2344], "X", "s", "n"), "2344, 2548 (windows of 294)", "name every seam of one plan")
        assert stl.seams_arg("2344,2548") == [2344, 2548] and stl.seams_arg(None) is None and stl.seams_arg("898") == [898]
        refused(lambda: stl.seams_arg("898-900"), "comma-separated")
        refused(lambda: stl.seams_arg("898,,949"), "comma-separated")
        refused(lambda: stl.seams_arg(""), "comma-separated")
        refused(lambda: stl.seam_of(406, 604, [898, 898], "X", "s", "n"), "named twice")
        refused(lambda: stl.seam_of(406, 604, [604], "X", "s", "n"), "604 is not inside the load 604-1009 after its first frame")
        # THE CASE THAT GOES RED when a plan whose last window writes almost nothing is taken as a seam (at 397 frames, 243's last window
        # writes one new frame); and the defaults say how many the last window writes
        refused(lambda: stl.seam_of(397, 0, [243, 396], "X", "s", "n"), "last window writes 1 new frame(s)", "less than one latent step (17")
        assert stl.seam_of(397, 0, [L], "X", "s", "n")[0] == L, "a plan whose last window writes a step or more was refused"
        assert stl.seam_of(60, 100, None, None, "s", "n") == (None, None)
        # the job step puts the chosen window on the load and the seam among the defaults (the fixture's shot is short, so seam_of's answer
        # for a long load stands in for it here)
        real = stl.seam_of
        asked = []
        stl.seam_of = lambda *x, **k: asked.append(x) or (294, {"what": "the seam", "carries": "SEAM ROW", "from": "X"})
        try:
            seamed = load_with(Made(Path(tmp) / "s3"), ours)
        finally:
            stl.seam_of = real
        # THE CASE THAT GOES RED when the window a seam needs is not put on the load, or its row is not among the defaults
        assert seamed["loads"][0]["window"] == 294 and {x["what"]: x for x in seamed["job"]["defaults"]}["the seam"]["carries"] == "SEAM ROW", seamed["loads"][0]
        # THE CASE THAT GOES RED when the job step hands seam_of another load length or first frame than the load's own
        assert asked[0][:2] == (seamed["loads"][0]["frames"], seamed["loads"][0]["first"]), (asked, seamed["loads"][0]["first"], seamed["loads"][0]["frames"])
        # THE KEPT FRAMES' NOISE on a person's word (--context-noise with --context-noise-why, the lead's ruling of 2026-10-15: per load, not a
        # default, for a load longer than one window)
        row = stl.context_noise_of(406, 0.8, "THE TURN AT 945", "the session", "now")
        # THE CASE THAT GOES RED when a load of chained windows is not given its row, with the value and whose word it is
        assert row["what"] == "the kept frames' noise" and row["carries"].startswith("0.8: each later window") \
            and row["from"] == "the session, now: THE TURN AT 945", row
        assert stl.context_noise_of(406, None, None, "s", "n") is None
        refused(lambda: stl.context_noise_of(stl.h3_config.MAX_LENGTH, 0.8, "X", "s", "n"), "one window holds this load's", "no window of it keeps frames")
        refused(lambda: stl.context_noise_of(406, 0.8, None, "s", "n"), "go together")
        refused(lambda: stl.context_noise_of(406, None, "X", "s", "n"), "go together")
        refused(lambda: stl.context_noise_of(406, 0.0, "X", "s", "n"), "above 0")
        refused(lambda: stl.context_noise_of(406, 1.5, "X", "s", "n"), "at most 1")
        refused(lambda: load_with(Made(Path(tmp) / "cn1"), ours, context_noise=0.8, context_noise_why="X"), "one window holds this load's 60 frames")
        assert "context_noise" not in seamed["loads"][0] and "the kept frames' noise" not in {x["what"] for x in seamed["job"]["defaults"]}, \
            "a load that named no noise is given one"
        real = stl.context_noise_of
        asked = []
        stl.context_noise_of = lambda *x, **k: asked.append(x) or {"what": "the kept frames' noise", "carries": "NOISE ROW", "from": "X"}
        try:
            noised = load_with(Made(Path(tmp) / "cn2"), ours, context_noise=0.8, context_noise_why="THE TURN")
        finally:
            stl.context_noise_of = real
        # THE CASE THAT GOES RED when the job step does not put the noise on the load, its row among the defaults, or hands the check
        # another length than the load's
        assert noised["loads"][0]["context_noise"] == 0.8 and {x["what"]: x for x in noised["job"]["defaults"]}["the kept frames' noise"]["carries"] == "NOISE ROW", \
            noised["loads"][0]
        assert asked[0][:3] == (noised["loads"][0]["frames"], 0.8, "THE TURN"), asked
        # HER EYES SHUT ON A SPAN on a person's word (--eyes-shut-on with --eyes-shut-form, the lead's ruling of 2026-10-15 on 2823b)
        look = f"G3 LOOKED at every frame, {sheet}"
        refused(lambda: load_with(Made(Path(tmp) / "e1"), ours, eyes_shut_on=[(110, 115, look)]), "go together")
        refused(lambda: load_with(Made(Path(tmp) / "e2"), ours, eyes_shut_on=[(110, 115, look)], eyes_shut_form="wide"), "go together")
        refused(lambda: load_with(Made(Path(tmp) / "e3"), ours, eyes_shut_on=[(110, 115, "G3 looked")], eyes_shut_form="lids"), "name its")
        refused(lambda: load_with(Made(Path(tmp) / "e4"), ours, eyes_shut_on=[(150, 170, look)], eyes_shut_form="lids"), "the load is 100-159")
        eyes = load_with(Made(Path(tmp) / "e5"), ours, eyes_shut_on=[(110, 115, look)], eyes_shut_form="lids")
        eye_row = {x["what"]: x for x in eyes["job"]["defaults"]}["the eye"]
        # THE CASE THAT GOES RED when a person's shut-eye span is not kept in the job, said among the defaults, or passed on with its form
        assert eyes["job"]["reference"]["iris"]["eyes_shut"]["form"] == "lids" and "SHUT on 110-115 (drawn 'lids')" in eye_row["carries"] \
            and "G3 LOOKED" in eye_row["from"], (eyes["job"]["reference"]["iris"], eye_row)
        flags = stl.drawn_form(eyes["job"], args(), {"classes": Path("/c.mkv")}, 100)
        assert flags[flags.index("--eyes-shut-on") + 1].startswith("110-115=a person's word on a look (the session, now): G3 LOOKED") \
            and flags[flags.index("--eyes-shut-form") + 1] == "lids", flags
    with tempfile.TemporaryDirectory() as tmp:
        # nobody else: no strangers keep-out, though the plan names one; the plan's other keep-outs stand
        d = load_with(Made(Path(tmp)), ours, plan_keep=["held", "strangers"])
        assert d["job"]["others_present"]["others_present"] == "no" and [k["name"] for k in d["job"]["keep_outs"]["made"]] == ["held"], d["job"]["keep_outs"]
        assert "strangers" not in d["loads"][0]["others"] and d["job"]["keep_outs"]["by"].startswith("the plan (a person: the plan's own"), d["job"]["keep_outs"]
    with tempfile.TemporaryDirectory() as tmp:
        # the video's not_people: a lone small other in a shot it names counts as no one, said with its box
        d = load_with(Made(Path(tmp)), ours + [(4, [55, 30, 5, 5], 0.01)], not_people=[{"what": "THE BOX", "shots": [[100, 159]], "by": "the owner"}])
        got = d["job"]["others_present"]
        assert got["others_present"] == "no" and got["not_people"][0]["box"] == [55, 30, 60, 35] and got["not_people"][0]["owners_words"] == "THE BOX", got
        assert "keep_outs" not in d["job"], "the strangers kept out where the plan's not_people leave nobody"
    return ("who else is in the shot is in the job file and its defaults, from the preview's tables; the strangers kept out where someone is, and not where "
            "nobody is whatever the plan's key; the video's not_people counted as no one, with its box")


def a_plan_that_keeps_the_strangers_out() -> str:
    """The lead's 3167 (2026-10-14): the region lay on other women's chins and faces, and her own part lay outside her
    track on another woman's face. The plan's `strangers` keeps every other person out and holds her part to her track."""
    import numpy as np
    from _lib.frames import read_mask, write_mask_video
    import strangers as st
    import h3_config
    px = int(h3_config.MASKED_STRANGERS["grow_px"])
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        made.plan["subjects"]["a"].update(keep_outs=["hands", "strangers"], keep_outs_by="a person, 2026-01-01: other women's faces in the margin")
        d = made.load()
        job, load, mine = d["job"], d["loads"][0], next(s for s in d["job"]["subjects"] if s["label"] == "a")
        out = made.m.output
        want = out / "Video/v/keep_outs/shot_c_0100/shot_c_0100_strangers.mkv"
        kept = {k["name"]: k for k in job["keep_outs"]["made"]}
        assert load["others"] == ["b", "c", "hands", "strangers"] and kept["strangers"]["track"] == str(want), (load["others"], job["keep_outs"])
        assert want == st.file_of(made.preview(100)["job"], out), "the job step and bench/strangers.py name the file differently"
        assert set(kept["strangers"]["tracks"]) == {"a", "b", "c"} and kept["strangers"]["frames"] == 60, kept["strangers"]
        held = out / f"Video/v/parts/shot_c_0100/shot_c_0100_a_part_saved_on_track_{px}px.mkv"
        # THE CASE THAT GOES RED when a load that keeps the strangers out carries the part as the preview saved it
        assert mine["parts_file"] == str(held) and job["part_on_track"]["grow_px"] == px and job["part_on_track"]["part"].endswith("shot_c_0100_a_parts_00001.mkv"), mine
        assert f"THE PART IS HELD TO a's OWN TRACK grown {px} px" in job["note"], job["note"]
        graphs.check_job(job)
        for one in masked_job.loads_of(d):
            graphs.check_load(job, one, masked_job.chain_lengths())
        # a load of some of its parts holds THAT part to the track
        was_parts = made.plan["subjects"]["a"]["parts"]
        made.plan["subjects"]["a"]["parts"] = ["hair", "face_and_neck"]
        some = made.load(name="a_hp_0100", parts=["hair"], parts_by="a person")
        assert some["job"]["part_on_track"]["part"] == some["job"]["part"]["made"][0]["track"], some["job"]["part_on_track"]
        assert next(s for s in some["job"]["subjects"] if s["label"] == "a")["parts_file"].endswith(f"_part_hair_on_track_{px}px.mkv")
        made.plan["subjects"]["a"]["parts"] = was_parts
        # THE JOB STEP: refused, with NOTHING written (no keep-out cut, no held part), until the file is there with its record
        folder = made.preview_ran(made.preview())
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        holds, cuts = [], []

        def hold(part, track, grow, to, first=0):
            holds.append((Path(part).name, Path(track).name, grow, Path(to).name))
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"held")
            return {"frames": 60, "size": [64, 48], "px_taken_off_a_frame_median": 3, "px_taken_off_most": 9, "frames_with_any_taken_off": 7,
                    "frames_the_track_is_empty_and_held": []}

        def cut(class_map, classes, to, **on):
            cuts.append(Path(to).name)
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"cut")
            return {"frames": 60, "size": [64, 48], "frames_with_the_mask": 60, "px_most": 9}
        # the track the part is held to: empty on every frame is refused before anything is written
        canvas = tuple(made.preview(100)["job"]["canvas"])
        a_track = Path(kept["strangers"]["tracks"]["a"])
        write_mask_video(a_track, np.zeros((60, canvas[1], canvas[0]), bool))
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), "empty on every frame", "Nothing is written")
        on = np.zeros((60, canvas[1], canvas[0]), bool)
        on[:, 10:40, 10:40] = True
        write_mask_video(a_track, on)
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), "shot_c_0100_strangers.mkv is not there", "bench/strangers.py run --preview", "Nothing is written")
        assert not holds and not cuts and not held.exists() and not list(made.jobs.glob("*.json")), "something was written for a load the job step refused"
        # with the model store known, the job step runs the strangers pass its load needs, on an empty queue (the lead's order of 2026-10-14)
        ran_before = len(made.m.ran)
        a.model_store = "/store"
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), "shot_c_0100_strangers.mkv is not there")     # the stand-in pass makes no file
        passes = [argv for label, argv in made.m.ran[ran_before:] if label == "strangers pass"]
        assert len(passes) == 1 and passes[0][1:4] == ["bench/strangers.py", "run", "--preview"] and "/store/facebook_sapiens2-seg-1b" in passes[0], passes
        made.m.busy = (1, 0)
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), "the queue is not empty", "no strangers pass beside a render")
        made.m.busy, made.m.dry = (0, 0), True
        before_dry = len([1 for label, _ in made.m.ran if label == "strangers pass"])
        stl.step_job(a, made.m, hold=hold, cut=cut)
        assert len([1 for label, _ in made.m.ran if label == "strangers pass"]) == before_dry, "a dry run ran the strangers pass"
        made.m.dry, a.model_store = False, ""
        want.parent.mkdir(parents=True, exist_ok=True)
        write_mask_video(want, ~on)
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), "is not there beside it")
        tracks = kept["strangers"]["tracks"]
        record = {"preview": "shot_c_0100", "first_source_frame": 100, "frames": 60, "grow_px": px, "hold_frames": int(h3_config.MASKED_STRANGERS["hold_frames"]),
                  "subjects_taken_out": tracks, "file": str(want), "frame_hash": lc.frame_hash(str(want)), "taken_out": strangers_tool.OWN_FORM,
                  "handed_back_said": "a: 1 piece(s) handed back on 1 frame(s) (130), the largest 40 px at 130 in [1, 2, 3, 4]"}
        for wrong, words in (({"preview": "shot_c_0099"}, "record's preview"), ({"grow_px": px + 1}, "record's grow_px"), ({"frames": 59}, "record's frames"),
                             ({"file": str(want) + ".elsewhere"}, "record's file"), ({"frame_hash": None}, "holds no hash of its frames"),
                             ({"frame_hash": "0" * 32}, "not the ones its record hashed"),
                             ({"subjects_taken_out": {"a": tracks["a"]}}, "record's subjects_taken_out"),
                             ({"taken_out": None}, "made before 2026-10-14's mend")):
            want.with_suffix(".json").write_text(json.dumps({**record, **wrong}))
            refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut), words)
        assert not holds and not cuts, "something was written for a strangers file of another making"
        want.with_suffix(".json").write_text(json.dumps(record))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            stl.step_job(a, made.m, hold=hold, cut=cut)
        # the pieces the strangers pass handed back to a subject are printed for the load, so a look sees them (the lead, 2026-10-14)
        assert "a: 1 piece(s) handed back on 1 frame(s) (130)" in printed.getvalue(), printed.getvalue()[-2000:]
        assert holds and holds[-1][1:] == ("shot_c_0100_a_track_00001.mkv", px, held.name) and cuts == ["shot_c_0100_a_hands.mkv"], (holds, cuts)
        # the job carries the hash its strangers file was held to: the chain holds the capture's copy to it
        written = json.loads((made.jobs / "a_fp_0100.json").read_text())
        assert next(k for k in written["job"]["keep_outs"]["made"] if k["name"] == "strangers")["frame_hash"] == record["frame_hash"], written["job"]["keep_outs"]
        assert json.loads(held.with_suffix(".json").read_text())["grow_px"] == px
        # a held part that is there is held to a fresh making; one of another making is refused
        a.remake = True
        a.no_carry_words = "remake held part check"
        stl.step_job(a, made.m, hold=hold, cut=cut, hashed=lambda path: "the same")
        refused(lambda: stl.step_job(a, made.m, hold=hold, cut=cut, hashed=lambda path: path if "_on_track_" in path else "same"), "is NOT the part held to the track",
                "moved aside by a person")
        # THE HOLD ITSELF, on masks: beyond the grow is taken off, within it stays, and a frame whose track is empty is held to the frame before
        part, track = np.zeros((3, 48, 64), bool), np.zeros((3, 48, 64), bool)
        track[:2, 10:30, 10:30] = True
        write_mask_video(Path(tmp) / "short.mkv", track[:2])
        write_mask_video(Path(tmp) / "p3.mkv", part)
        refused(lambda: stl.hold_to_track(Path(tmp) / "p3.mkv", Path(tmp) / "short.mkv", px, Path(tmp) / "x.mkv"), "they are not of one preview")
        write_mask_video(Path(tmp) / "small.mkv", np.ones((3, 24, 32), bool))
        refused(lambda: stl.hold_to_track(Path(tmp) / "p3.mkv", Path(tmp) / "small.mkv", px, Path(tmp) / "x.mkv"), "they are not of one preview")
        part[:, 12:20, 12:20] = True
        part[:, 10:12, 30 + px - 1] = True                   # exactly the grow off the track's edge: kept
        part[:, 14:16, 30 + px] = True                       # a pixel further: taken off
        part[:, 40:44, 50:60] = True                         # far off it: taken off
        for m, f in ((part, "p.mkv"), (track, "t.mkv")):
            write_mask_video(Path(tmp) / f, m)
        got = stl.hold_to_track(Path(tmp) / "p.mkv", Path(tmp) / "t.mkv", px, Path(tmp) / "held.mkv", first=500)
        back = read_mask(str(Path(tmp) / "held.mkv"), (64, 48), 0, 0, 3)[0]
        assert back[:, 12:20, 12:20].all() and back[:, 10:12, 30 + px - 1].all() and not back[:, 14:16, 30 + px].any() and not back[:, 40:44, 50:60].any(), got
        assert got["frames_the_track_is_empty_and_held"] == [502] and got["px_taken_off_most"] == 42, got
    return ("a plan's `strangers` names the preview's strangers file among the load's others, at the path bench/strangers.py writes, and holds the load's "
            "part (the saved one, or the load's own) to its subject's track grown h3_config.MASKED_STRANGERS px through `parts_file`; the job step writes nothing "
            "until the file is there with a record of this preview, its frames, its grow and its tracks; a held part of another making is refused; the hold "
            "keeps what lies within the grow, takes off what lies beyond, and holds a frame whose track is empty to the frame before")

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
        # THE CASE THAT GOES RED when a load of some of the plan's parts says it renders the plan's whole region (second_hp_5461b's header)
        assert f"ONE LOAD: b, hair alone (of the plan's {made.plan['subjects']['b']['region']}, --parts)," in job["note"], job["note"][:300]
        # NO REFERENCE VIDEO (--motion none, the lead's ruling of 2026-10-15 for 5461): a hair load only, with its reason; the load names
        # no reference file, the note and the defaults say so, and the reference step makes nothing
        bare = made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, motion="none", motion_why="HER FACE IS HIDDEN")
        # THE CASE THAT GOES RED when a hair-alone load says it carries a voice its text never reads, or reads a bad voice entry
        assert {x["what"]: x for x in bare["job"]["defaults"]}["the voice"]["carries"].startswith("not read: a load of the hair alone"), bare["job"]["defaults"]
        made.plan["subjects"]["b"]["voice"] = [{"voice": "hums"}]           # a bad entry its text never reads does not stop a hair-alone load
        made.load(subject="b", name="b_hp_0100v", parts=["hair"], parts_by=why, motion="none", motion_why="HER FACE IS HIDDEN")
        made.plan["subjects"]["b"].pop("voice")
        lone = bare["loads"][0]
        # THE CASE THAT GOES RED when a no-reference load still names a reference file, or says nothing of it
        assert lone.get("motion") == "none" and "motion_file" not in lone and "motion_first" not in lone, lone
        assert "NO REFERENCE VIDEO (motion none)" in bare["job"]["note"] and "HER FACE IS HIDDEN" in bare["job"]["note"], bare["job"]["note"][-400:]
        rows_ = {x["what"]: x["carries"] for x in bare["job"]["defaults"]}
        # THE CASE THAT GOES RED when the job still speaks of a drawn reference it does not have
        assert rows_["the reference video"].startswith("none") and not {"the eye", "the mouth", "the reference held to the subject's track"} & set(rows_) \
            and "reference's record" not in rows_["the text"] and "frozen form" not in bare["job"]["note"] \
            and bare["job"]["reference"]["form"] == "none" and "mouth" not in bare["job"]["reference"], (rows_, bare["job"]["reference"])
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, parts_on=[(130, 139, ["face_and_neck"])],
                                  motion="none", motion_why="X"), "on every frame")
        refused(lambda: made.load(subject="b", name="b_hfp_0100", motion="none", motion_why="X"), "for a load of her hair alone")
        # --no-face-on acts on a drawn reference, and a hair-alone load draws no face on any frame already
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, motion="none", motion_why="X",
                                  no_face_on=[(110, 115, f"looked at {made.plan_file}")]), "--no-face-on acts on the drawn reference")
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, no_face_on=[(110, 115, f"looked at {made.plan_file}")]),
                "--no-face-on on a load of her hair alone")
        # --mouth-on never where the load draws no face: a load of her hair alone, or a hair-alone span of --parts-on (the outside reviewer's
        # C7b of 2026-10-15: it passed the job step and failed at the reference)
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, mouth_on=[(110, 115, f"looked at {made.plan_file}")]),
                "--mouth-on on a load of her hair alone")
        refused(lambda: made.load(subject="b", name="b_hfp_0100", parts_by=why, parts_on=[(130, 139, ["hair"])],
                                  mouth_on=[(128, 132, f"looked at {made.plan_file}")]), "--mouth-on 128-132 against the hair-alone span 130-139")
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, motion="none", motion_why=" "), "say why with --motion-why")
        # no reference video, so no mouth to leave out of it (--no-mouth-on)
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, motion="none", motion_why="X",
                                  no_mouth_on=[(110, 115, "X")]), "this load has none (--motion none)")
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, motion="none", motion_why="X",
                                  eyes_shut_on=[(110, 115, "X")], eyes_shut_form="none"), "this load has none (--motion none)")
        refused(lambda: made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_by=why, eyes_shut_on=[(110, 115, "X")], eyes_shut_form="none"),
                "her hair alone")
        jf = Path(tempfile.mkdtemp(prefix="motion_none_")) / "b_hp_0100.json"
        jf.write_text(json.dumps(bare))
        a_ = args()
        a_.job_file = str(jf)
        import contextlib
        import io
        out_ = io.StringIO()
        with contextlib.redirect_stdout(out_):
            stl.step_reference(a_, made.m)
        assert "NO REFERENCE VIDEO (motion none), so nothing is made here" in out_.getvalue() and made.m.ran == [], (out_.getvalue(), made.m.ran)
        a_.plain = "a word"
        refused(lambda: stl.step_reference(a_, made.m), "no reference to cut")
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
        # A PART THAT CHANGES INSIDE ONE LOAD (2026-10-14, the second subject's 1010-1031): the plan's parts, and the hair alone on named frames
        on = made.load(subject="b", name="b_hfp_0100", parts_on=[(110, 120, ["hair"])], parts_by=why)["job"]
        po, pm = on["part"], on["part"]["made"][0]
        assert po["parts"] == ["hair", "face_and_neck"] and po["parts_on"] == [{"first": 110, "last": 120, "parts": ["hair"]}], po
        assert pm["classes_on"] == [{"first": 110, "last": 120, "classes": ["Hair"]}] and pm["track"].endswith("shot_c_0100_b_part_hair_face_and_neck_110-120_hair.mkv"), pm
        assert next(x for x in on["subjects"] if x["label"] == "b")["parts_on"] == po["parts_on"] and "ON 110-120 IT IS HAIR ALONE" in on["note"] \
            and "THIS LOAD'S PART IS THE PLAN'S" in on["note"], on["note"]
        both_on = made.load(subject="b", name="b_hp_0100", parts=["hair"], parts_on=[(130, 139, ["face_and_neck"])], parts_by=why)["job"]["part"]
        assert both_on["parts"] == ["hair"] and both_on["parts_on"][0]["parts"] == ["face_and_neck"], both_on
        for wrong, words in (({"parts_on": [(90, 120, ["hair"])]}, "not inside the load's frames"), ({"parts_on": [(110, 120, ["nose"])]}, "some of the plan's"),
                             ({"parts_on": [(110, 120, ["hair"]), (115, 125, ["hair"])]}, "overlaps"),
                             ({"parts_on": [(110, 120, ["face_and_neck", "hair"])]}, "is the load's own part on its other frames"),
                             ({"parts_on": [(110, 120, ["hair"])], "parts_by": ""}, "--parts-by says who chose them")):
            refused(lambda w=wrong: made.load(subject="b", name="b_hfp_0100", **{"parts_by": why, **w}), words)
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
        a.no_carry_words = "remake part cut check"
        stl.step_job(a, made.m, cut=cut, inside=inside(0), hashed=lambda path: "the same")
        assert cuts[-1][2].endswith(".cut_again.mkv") and (made.jobs / "b_hp_0100b.json").is_file() and not list((out / "Video/v/parts/shot_c_0100").glob("*cut_again*")), cuts
        refused(lambda: stl.step_job(a, made.m, cut=cut, inside=inside(0), hashed=lambda path: path), "is there and is NOT what its classes cut", "a part of another making")
        assert not (made.jobs / "b_hp_0100c.json").exists()
        # the step with --parts-on hands the cut each range's classes and the class map's first source frame, and the record keeps them
        seen_on = []

        def cut_on(class_map, classes, to, classes_on=None, first=None):
            seen_on.append((tuple(classes), classes_on, first))
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"cut")
            return {"frames": 60, "size": [64, 48], "frames_with_the_mask": 60, "px_most": 9}
        b2 = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="b", preview=str(folder), parts_by=why,
                  parts_on=["110-120=hair"])
        stl.step_job(b2, made.m, cut=cut_on, inside=inside(0))
        assert seen_on == [(tuple(c for k in ("hair", "face_and_neck") for c in stl.part_classes()[k]), [{"first": 110, "last": 120, "classes": ["Hair"]}], 100)], seen_on
        rec_on = json.loads((out / "Video/v/parts/shot_c_0100/shot_c_0100_b_part_hair_face_and_neck_110-120_hair.json").read_text())
        assert rec_on["parts_on"] == [{"first": 110, "last": 120, "parts": ["hair"]}] and rec_on["classes_on"][0]["classes"] == ["Hair"], rec_on
        assert json.loads((made.jobs / "b_hfp_0100.json").read_text())["job"]["part"]["parts_on"][0]["first"] == 110
        (made.jobs / "b_hfp_0100.json").unlink()
        refused(lambda: stl.step_job(args(**{**vars(b2), "parts_on": ["110=hair"]}), made.m, cut=cut_on, inside=inside(0)), "not FIRST-LAST=NAME")
        # a load of the same subject with no --parts is as it ever was, on the same preview
        a.parts, a.parts_by, a.remake, a.no_carry_words = None, None, False, None
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
        # a frame inside a range carries that range's classes: frame 1 of a class map whose frame 0 is source frame 100 is source frame 101
        changing = Path(tmp) / "changing.mkv"
        stl.cut_keep_out(video, [c for k in ("hair", "face_and_neck") for c in table[k]], changing, classes_on=[{"first": 101, "last": 101, "classes": list(table["hair"])}], first=100)
        got_on = read_mask(str(changing), (64, 48), 0, 0, 3)[0]
        only_hair = np.isin(cmap, [names.index(c) for c in table["hair"]])
        assert (got_on[0] == both[0]).all() and (got_on[2] == both[2]).all() and (got_on[1] == only_hair[1]).all() and not (got_on[1] == both[1]).all(), \
            "a frame in a range is not cut by the range's classes alone, or a frame outside it is"
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


def a_track_that_is_not_its_person() -> str:
    """2026-10-13: in the preview of 604-1009 the lead's track came back on 1003-1009 lying almost wholly inside the second
    subject's, after the lead had left the picture; a load of the second subject there was blocked, and taking the lead
    out of its others failed the capture's verify. A person's word empties that track on those frames for the load."""
    import numpy as np
    from _lib.frames import read_mask, write_mask_video
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        out = made.m.output
        why = "the lead has left the picture; her track lies on the second subject"
        plain = made.load()
        assert "track_not" not in plain["job"] and "TRACK IS NOT" not in plain["job"]["note"], "a load says a track is not its person when nobody did"
        d = made.load(track_not=[("b", 110, 120, why)], track_not_by="a person, on the tiles")
        t = d["job"]["track_not"]
        one = t["made"][0]
        # THE CASE THAT GOES RED when a person's word on another subject's track is not carried into the job
        assert t["by"] == "a person, on the tiles" and one["subject"] == "b" and one["frames"] == [[110, 120]] and one["first"] == 100, t
        assert one["track"] == str(out / "Video/v/tracks/shot_c_0100/shot_c_0100_b_track_not_110-120.mkv") \
            and one["cut_from"] == str(out / "Video/v/the session/shot_c_0100/shot_c_0100_b_track_00001.mkv"), one
        assert "B'S TRACK IS NOT B ON 110-120, on a person, on the tiles" in d["job"]["note"] and "b stays an other of the load" in d["job"]["note"], d["job"]["note"]
        assert d["loads"][0]["others"] == ["b", "c"], "b was taken out of the load's others"
        graphs.check_job(d["job"])
        two = made.load(track_not=[("b", 130, 139, "x"), ("b", 110, 120, why), ("c", 100, 105, "y")], track_not_by="p")["job"]["track_not"]["made"]
        assert [x["subject"] for x in two] == ["b", "c"] and two[0]["frames"] == [[110, 120], [130, 139]] and two[0]["track"].endswith("_b_track_not_110-120_130-139.mkv"), two
        for wrong, words in (({"track_not": [("a", 110, 120, why)]}, "is the load's own subject"), ({"track_not": [("z", 110, 120, why)]}, "is not a subject of the preview"),
                             ({"track_not": [("b", 90, 120, why)]}, "not inside the load's frames"),
                             ({"track_not": [("b", 110, 120, why), ("b", 115, 125, why)]}, "overlaps"),
                             ({"track_not": [("b", 110, 120, why)], "track_not_by": " "}, "--track-not-by")):
            refused(lambda w=wrong: made.load(**{"track_not_by": "p", **w}), words)
        for text in ("b:110-120", "b:110=why", ":110-120=why", "b:120-110=why", "b:110-120= "):
            refused(lambda x=text: stl.track_not_arg(x), "SUBJECT:FIRST-LAST=WHY")
        assert stl.track_not_arg("b:110-120=her track  is on   the second") == ("b", 110, 120, "her track is on the second")
        # the copy itself: the named source frames emptied and no other
        track = np.zeros((6, 48, 64), bool)
        track[:, 10:20, 10:20] = True
        write_mask_video(Path(tmp) / "t.mkv", track)
        got = stl.cut_track(Path(tmp) / "t.mkv", [[102, 103]], 100, Path(tmp) / "c" / "t.mkv")
        back = read_mask(str(Path(tmp) / "c" / "t.mkv"), (64, 48), 0, 0, 6)[0]
        assert not back[2:4].any() and (back[[0, 1, 4, 5]] == track[[0, 1, 4, 5]]).all() and got["px_emptied"] == 200, got
        # the figure: where one subject's track lies inside the load subject's, by runs of frames, and no word on it
        a_t, b_t = np.zeros((6, 48, 64), bool), np.zeros((6, 48, 64), bool)
        a_t[:, 5:40, 5:40] = True
        b_t[:, 30:45, 30:60] = True
        b_t[3:5] = False
        b_t[3:5, 10:20, 10:20] = True                        # on 103-104 b's track lies wholly inside a's
        folder = made.preview_ran(made.preview())
        files = json.loads((folder / "preview.queued.json").read_text())["files"]
        for label, m in (("a", a_t), ("b", b_t)):
            write_mask_video(out / next(f for f in files if f"_{label}_track_" in f), m)
        lines = stl.track_over_track(made.preview()["job"], files, out, "a", 100, 105)
        assert any("b's track lies more than 0.5 inside a's on 2 frame(s), 103-104 (most 1.00)" in x and "--track-not b:FIRST-LAST=WHY" in x for x in lines), lines
        # THE STEP makes the copy before the job file and prints the figure; a dry run makes nothing
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder),
                 track_not=[f"b:110-120={why}"], track_not_by="a person, on the tiles")
        copies, said = [], []

        def copy(track, frames, first, to):
            copies.append((Path(track).name, frames, first, Path(to).name))
            Path(to).parent.mkdir(parents=True, exist_ok=True)
            Path(to).write_bytes(b"copy")
            return {"frames": 60, "px_emptied": 5}
        over = lambda *x: said.append(x) or ["track over track: a figure"]          # noqa: E731
        made.m.dry = True
        stl.step_job(a, made.m, track_cut=copy, over=over)
        assert not copies and not list(made.jobs.iterdir()) and said, "a dry run copied a track or wrote a job file"
        made.m.dry = False
        stl.step_job(a, made.m, track_cut=copy, over=over, hashed=lambda path: f"HASH OF {Path(path).name}")
        assert copies == [("shot_c_0100_b_track_00001.mkv", [[110, 120]], 100, "shot_c_0100_b_track_not_110-120.mkv")], copies
        kept = json.loads((made.jobs / "a_fp_0100.json").read_text())["job"]["track_not"]["made"][0]
        # THE CASE THAT GOES RED when the job keeps no hash of the copy's frames for the chain to hold it to (the cold reader of 2026-10-15)
        assert kept["frames"] == [[110, 120]] and kept["hash"] == "HASH OF shot_c_0100_b_track_not_110-120.mkv" \
            and kept["cut_from_hash"] == "HASH OF shot_c_0100_b_track_00001.mkv", kept
        a.remake = True
        a.no_carry_words = "remake track_not check"
        refused(lambda: stl.step_job(a, made.m, track_cut=copy, over=over, hashed=lambda path: path), "is there and is NOT the copy", "moved aside by a person")
    return ("a person's word that another subject's track is not them on named frames goes into the job with who said it, why and the copy every step "
            "reads, that subject staying an other; the copy empties those frames and no other; the load's own subject, a stranger to the preview, frames "
            "outside the load, overlaps and no --track-not-by are refused; the figure names the frames where one track lies inside the other's")

def the_text_step() -> str:
    """The text's writer had to be set in the job file by hand after the chain refused it (the first real load, twice)."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        job_file = made.jobs / "a_fp_0100.json"
        job_file.write_text(json.dumps(made.load(), indent=1))
        a = args(job_file=str(job_file), writer="a writer, from frames 100 to 159   looked at every fifth", by_hand="the owner's own words for this shot")
        refused(lambda: stl.step_text(a, made.m), "is not written, or is empty")
        (made.jobs / "a_fp_0100.txt").write_text("the words of the load\n")
        refused(lambda: stl.step_text(args(job_file=str(job_file), writer="  ", by_hand="w"), made.m), "--writer takes a person's words")
        refused(lambda: stl.step_text(args(job_file=str(job_file)), made.m), "give --plan")
        # THE CASE THAT GOES RED when a job hands in a place-and-action sentence and is not refused (load_text.NO_SCENE, 2026-10-15)
        refused(lambda: stl.step_text(args(job_file=str(job_file), place_and_action="sits at a table"), made.m), "takes no place-and-action sentence")
        refused(lambda: stl.step_text(args(job_file=str(job_file), writer="w", by_hand=" "), made.m), "--by-hand takes the reason in words")
        refused(lambda: lc.read_job(job_file, made.root), "texts.a_fp_0100.by is empty", "bench/shot_to_load.py text")
        before = job_file.read_bytes()
        made.m.dry = True
        stl.step_text(a, made.m)
        assert job_file.read_bytes() == before, "a dry run wrote the job file"
        made.m.dry = False
        stl.step_text(a, made.m)
        d = json.loads(job_file.read_text())
        by = d["job"]["texts"]["a_fp_0100"]["by"]
        assert by.startswith("a writer, from frames 100 to 159 looked at every fifth, BY HAND: the owner's own words for this shot; entered 20") \
            and by.endswith("by the session (bench/shot_to_load.py text)"), by
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
        refused(lambda: stl.step_text(args(job_file=str(job_file), writer="another", by_hand="w"), made.m), "the plan of a_fp_0100 is built", "not edited after")
        assert job_file.read_bytes() == kept
    # GENERATED (2026-10-14): from the reference's record, the plan's words and the voice table, around the person's one sentence
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        made.plan["text"] = {"film": "THE FILM LINE."}
        made.plan["subjects"]["a"]["text"] = {"who": "the girl", "looks": "brown eyes", "her": "her", "she": "she", "sings": True}
        d = made.load()
        job_file = made.jobs / "a_fp_0100.json"
        job_file.write_text(json.dumps(d, indent=1))
        a = args(job_file=str(job_file), plan=str(made.plan_file))
        refused(lambda: stl.step_text(a, made.m), "is made after it")
        ref = Path(d["loads"][0]["motion_file"]).with_suffix(".json")
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(json.dumps({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}))
        refused(lambda: stl.step_text(args(job_file=str(job_file)), made.m), "give --plan")
        refused(lambda: stl.step_text(args(job_file=str(job_file), plan=str(made.plan_file), writer="w"), made.m), "a generated text is the generator's")
        stl.step_text(a, made.m)
        words = (made.jobs / "a_fp_0100.txt").read_text()
        t = json.loads(job_file.read_text())["job"]["texts"]["a_fp_0100"]
        # THE CASE THAT GOES RED when the text step does not write the generated text, or says it as a person's
        assert "<Subject 1>, with the face of <Picture 1>, performs <Subject 2>." in words and "the white of the eye is white and the iris black" in words \
            and "the mouth is outlined" in words and "(S1) sings" in words and "THE FILM LINE." in words, words
        assert t["by"].startswith("GENERATED by bench/load_text.py") and "with no place-and-action sentence" in t["by"] and t["generated_from"]["drawn"]["iris"] == "contrast", t
        assert lc.read_job(job_file, made.root)["text"] == made.jobs / "a_fp_0100.txt"
        (made.jobs / "a_fp_0100.txt").write_text("somebody's own words\n")
        refused(lambda: stl.step_text(a, made.m), "holds another text", "written once")
    return ("the text is made from the reference's record, the plan's words and the voice table with no place-and-action sentence (one is refused), "
            "written beside the job file and said as generated; a text written whole by a person only with its reason; refused with no text, with no "
            "words, with neither or both ways, before the reference, over another text, and once the plan is built; a dry run writes nothing")


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
        a.no_carry_words = "remake on present copy"
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
        # THE CHAIN'S DEFAULTS (2026-10-14): the eye in contrast, the mouth by the lip check against the preview's class map of the subject
        classes = made.m.output / "Video/v/the session/shot_c_0100/shot_c_0100_a_classes_00001.mkv"
        assert (opt(ref, "--iris"), opt(ref, "--mouth"), opt(ref, "--classes"), opt(ref, "--classes-first")) == ("contrast", "auto", str(classes), "100"), ref
        after = json.loads(job_file.read_text())
        block = after["job"]["reference"]
        assert (block["form"], block["said_by"], block["record"]) == ("drawn", "the session", str(drawn.with_suffix(".json"))) and block["made"].endswith("Z"), block
        assert block["figures"]["in_the_reference"]["enlarged"] == 2.5 and after["loads"][0]["motion_file"] == str(drawn), block
        assert block["mouth"]["said_by"] == block["iris"]["said_by"] == "the chain's default" and "drawn" in block, block
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
        # a job with a person's no-mouth span takes a reference as it was only where that reference drew no mouth there
        drawn.with_suffix(".json").write_text(json.dumps(record))    # the folder's record of today's form again: the job takes the folder (`reference_folder`)
        sheet = Path(tmp) / "the_sheet.jpg"
        sheet.write_bytes(b"a sheet")
        hid = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104c", no_mouth_on=[(110, 115, f"THE LEAD looked at {sheet}")])
        hid_file = made.jobs / "a_fp_0104c.json"
        hid_file.write_text(json.dumps(hid))
        e.job_file = str(hid_file)
        mouthed = {**record, "drawn": {"mouth": {"form": "auto", "no_mouth_on": {"the gate": [[104, 109]]}}}}
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        # THE CASE THAT GOES RED when --existing takes a drawing with a mouth on frames a person's word says have none
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "draws a mouth on 6 frame(s)", "first 110, last 115")
        mouthed["drawn"]["mouth"]["no_mouth_on"]["no mouth there on the job's word"] = [[110, 115]]
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        stl.step_reference(e, made.m, figures)
        # a job with a person's mouth span takes a reference as it was only where that reference was asked for the same span
        sang = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104m", mouth_on=[(130, 133, f"THE LEAD looked at {sheet}")])
        sang_file = made.jobs / "a_fp_0104m.json"
        sang_file.write_text(json.dumps(sang))
        e.job_file = str(sang_file)
        # THE CASE THAT GOES RED when --existing takes a drawing never asked for the mouth a person's word names
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "on no frames and the job names [(130, 133)]")
        mouthed["drawn"]["mouth"]["mouth_on"] = {"on_whose_word": [{"first": 130, "last": 133, "why": "x"}]}
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        stl.step_reference(e, made.m, figures)
        del mouthed["drawn"]["mouth"]["mouth_on"]
        e.job_file = str(hid_file)
        # a job with a person's no-face span takes a reference as it was only where that reference left those frames bare
        away = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104f", no_face_on=[(120, 125, f"THE LEAD looked at {sheet}")])
        away_file = made.jobs / "a_fp_0104f.json"
        away_file.write_text(json.dumps(away))
        e.job_file = str(away_file)
        # THE CASE THAT GOES RED when --existing takes a drawing with face lines on frames a person's word says show no face
        mouthed["drawn"]["frames_left_bare"] = [120, 121, 122]
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "draws face lines on 3 frame(s)", "first 123, last 125")
        # the plain cut has no line of the face on any frame: taken as it was, whatever its drawing left bare (on a copy of the job, as the
        # step points the job at the plain cut)
        plain_file = made.jobs / "a_fp_0104f_plain.json"
        plain_file.write_text(away_file.read_text())
        e.plain, e.job_file = "A PERSON'S WORD", str(plain_file)
        stl.step_reference(e, made.m, figures)
        e.plain, e.job_file = None, str(away_file)
        mouthed["drawn"]["frames_left_bare"] = list(range(120, 126))
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        # the cues are printed under the figures, as the step's own lines
        import contextlib
        import io
        cues = {"ears_reversed": [{"first": 120, "last": 122, "least": 21.0, "most": 56.0}], "ears_reversed_is": "EARS",
                "ears_not_read_on": [], "face_score_under_the_floor": [{"first": 130, "last": 130, "least": 0.1, "most": 0.1}],
                "face_score_is": "SCORES", "cannot_see": "BLIND"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stl.step_reference(e, made.m, lambda points, scale: {**figures(points, scale), "where_a_face_may_not_show": cues})
        # THE CASE THAT GOES RED when the cues are kept and not printed, or printed in other words than the step's own
        assert "WHERE A FACE MAY NOT SHOW" in out.getvalue() and "\n".join(stl.face_cue_lines(cues)) in out.getvalue(), out.getvalue()[-800:]
        assert stl.face_cue_lines(cues)[0] == "  ears reversed, EARS: 120-122 (21.0..56.0)" and stl.face_cue_lines(cues)[2] == "  SCORES: 130 (0.1..0.1)"
        assert stl.face_cue_lines({"not_read": "NO NAMES"}) == ["  not read: NO NAMES"]
        assert json.loads(away_file.read_text())["job"]["reference"]["no_face_on"][0]["first"] == 120, "the job lost its no-face span when its reference was read in"
        # and with her eyes named shut: taken only where the reference drew them shut, in the job's form
        shut_job = made.load(first=104, frames=40, why="a look cut it", name="a_fp_0104d", eyes_shut_on=[(110, 115, f"G3 LOOKED at {sheet}")],
                             eyes_shut_form="none")
        shut_file = made.jobs / "a_fp_0104d.json"
        shut_file.write_text(json.dumps(shut_job))
        e.job_file = str(shut_file)
        # THE CASE THAT GOES RED when --existing takes a drawing whose eyes are open on frames a person's word says are shut
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "does not draw her eyes shut")
        mouthed["drawn"]["eyes_shut"] = {"form": "lids", "frames": [[110, 115]]}
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "does not draw her eyes shut")
        mouthed["drawn"]["eyes_shut"] = {"form": "none", "frames": [[110, 115]], "on_whose_word": [{"first": 110, "last": 115, "why": "x"}],
                                         "drawn_shut_on": [[110, 113]]}
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        # THE CASE THAT GOES RED when --existing takes a drawing that shut only some of the frames it meant to
        refused(lambda: stl.step_reference(e, made.m, figures), "--existing", "does not draw her eyes shut")
        mouthed["drawn"]["eyes_shut"]["drawn_shut_on"] = [[110, 115]]
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        stl.step_reference(e, made.m, figures)
        # a frame with no face is not shut by head_reference: its own frames are compared, not the spans'
        mouthed["drawn"]["eyes_shut"].update(frames=[[110, 112]], drawn_shut_on=[[110, 112]])
        drawn.with_suffix(".json").write_text(json.dumps(mouthed))
        stl.step_reference(e, made.m, figures)
        e.job_file = str(again_file)
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


def the_defaults_this_load_carries() -> str:
    """The owner's word of 2026-10-14: make these our defaults and stop forgetting the tools we built. Every job step prints, and the
    job file keeps, the defaults the load carries, each with where it comes from; the plan may say otherwise with its reason."""
    import io
    from contextlib import redirect_stdout
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        d = made.load()
        lines = {x["what"]: x for x in d["job"]["defaults"]}
        assert list(lines) == ["the reference held to the subject's track", "the eye", "the mouth", "depth", "zoom", "other people in the shot", "SAM 3.1 memory selection",
                               "the strangers keep-out", "the part's pieces", "the hair band", "restore spans", "the text", "what the composite kept", "the voice"], list(lines)
        assert lines["depth"]["carries"].startswith("off:") and lines["depth"]["from"].startswith("the chain's default"), lines["depth"]
        # SAM 3.1'S MEMORY SELECTION (the owner's word of 2026-10-17): on in every preview by default, said among a load's defaults from
        # the preview its masks come from; off only with a person's reason, which the preview's graph and the load's defaults both carry
        memory = lines["SAM 3.1 memory selection"]
        assert memory["carries"].startswith("on:") and memory["from"].startswith("the chain's default"), memory
        assert graphs.preview(made.preview()["job"])[graphs.SAM_CORRECTED]["inputs"]["memory_selection"] is True, "a preview's graph does not set memory selection on"
        off = made.preview(no_memory_selection="the owner asked for ComfyUI's own memory on this shot")["job"]
        assert off["sam_memory_selection"]["on"] is False and "MEMORY SELECTION IS OFF" in off["note"], off["sam_memory_selection"]
        # THE CASE THAT GOES RED when a preview turned off still tracks with memory selection
        assert graphs.preview(off)[graphs.SAM_CORRECTED]["inputs"]["memory_selection"] is False, "a preview turned off still runs memory selection"
        carries, said = stl.memory_selection_carried(off)
        assert carries.startswith("OFF") and "the owner asked for ComfyUI's own memory" in said, (carries, said)
        refused(lambda: made.preview(no_memory_selection="  "), "--no-memory-selection takes the reason")
        try:
            graphs.check_job(dict(off, sam_memory_selection={"on": False}))
        except ValueError as e:
            assert "only on a person's word" in str(e), e
        else:
            raise AssertionError("a preview turned off with no reason was not refused by the builder")
        assert stl.memory_selection_carried({k: v for k, v in off.items() if k != "sam_memory_selection"})[0].startswith("not recorded")
        assert lines["the voice"]["carries"].startswith("the voice table's") and lines["the voice"]["from"] == "the chain", lines["the voice"]
        # THE PLAN'S VOICE (the owner's word of 2026-10-15 on 5675b): said among the defaults, with who said it; two voices in a load refused
        made.plan["subjects"]["a"]["voice"] = [{"first": 100, "last": 159, "voice": "talks, not heard", "by": "THE OWNER, 2026-10-15"}]
        heard = {x["what"]: x for x in made.load()["job"]["defaults"]}["the voice"]
        # THE CASE THAT GOES RED when the plan's voice is not what the job says it carries
        assert heard["carries"].startswith("talks, not heard, on 100-159 (the plan's") and heard["from"] == "THE OWNER, 2026-10-15", heard
        made.plan["subjects"]["a"]["voice"].append({"first": 150, "last": 170, "voice": "sings", "by": "X"})
        refused(lambda: made.load(), "One load is one voice")
        made.plan["subjects"]["a"].pop("voice")
        assert all(x.get("save_kept") is True for x in d["loads"]) and lines["what the composite kept"]["carries"].startswith("saved"), \
            "a load does not save what its composite kept, the chain's default"
        assert lines["the eye"]["carries"].startswith("white sclera and black iris") and lines["the eye"]["from"].startswith("the chain's default"), lines["the eye"]
        assert lines["the mouth"]["carries"].startswith("drawn when the lip lines lie") and lines["zoom"]["from"].startswith("the chain's default"), lines
        made.plan["subjects"]["a"]["reference"] = {"mouth": "left out", "iris": "between the lids", "why": "the owner, on this subject's stills"}
        d = made.load()
        lines = {x["what"]: x for x in d["job"]["defaults"]}
        assert d["job"]["reference"]["mouth"] == {"form": "left out", "said_by": "the plan", "why": "the owner, on this subject's stills"}, d["job"]["reference"]
        assert lines["the mouth"]["carries"] == "left out" and lines["the eye"]["from"] == "the plan: the owner, on this subject's stills", lines
        made.plan["subjects"]["a"]["reference"] = {"mouth": "open"}
        refused(made.load, "says no `why`")
        made.plan["subjects"]["a"]["reference"] = {"mouth": "open", "why": "w"}
        refused(made.load, "`reference.mouth`", "one of")
        del made.plan["subjects"]["a"]["reference"]
        # the plan's restore spans on the load's frames go into the job, for the gate (load_chain.told_to_the_gate); none elsewhere
        made.plan["subjects"]["a"].update(restore=[{"classes": ["Left_Hand", "Right_Hand"], "first": 110, "last": 120},
                                                    {"classes": ["Eyeglass"], "first": 5000, "last": 5010}], restore_by="THE LEAD, ON HER HANDS")
        d = made.load()
        assert d["job"]["restore"] == [{"classes": ["Left_Hand", "Right_Hand"], "first": 110, "last": 120, "by": "THE LEAD, ON HER HANDS"}], d["job"].get("restore")
        del made.plan["subjects"]["a"]["restore"], made.plan["subjects"]["a"]["restore_by"]
        assert "restore" not in made.load()["job"]
        # printed by the job step, one line each with where it comes from
        folder = made.preview_ran(made.preview())
        out = io.StringIO()
        made.m.dry = True
        with redirect_stdout(out):
            stl.step_job(args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder)), made.m)
        made.m.dry = False
        text = out.getvalue()
        assert "THE DEFAULTS THIS LOAD CARRIES:" in text and "  the eye: white sclera and black iris inside the lids  [the chain's default" in text, text[-1500:]
        # the reference step's own word stands over the job's, with its reason, and not without one
        job = {"reference": {"mouth": {"form": "auto"}, "iris": {"form": "contrast"}}}
        files = {"classes": Path("/c.mkv")}
        assert stl.drawn_form(job, args(), files, 100) == ["--iris", "contrast", "--mouth", "auto", "--classes", "/c.mkv", "--classes-first", 100]
        assert stl.drawn_form(job, args(mouth="drawn", form_why="her mouth reads on the stills"), files, 100) == \
            ["--iris", "contrast", "--mouth", "drawn", "--mouth-why", "her mouth reads on the stills"]
        refused(lambda: stl.drawn_form(job, args(iris="whole"), files, 100), "say why with --form-why")
        # the hole option (one test, 5114e): passed on with the reason, and not without one
        assert stl.drawn_form(job, args(class_mouth_needs_hole=True, form_why="ONE TEST"), files, 100)[-2:] == ["--class-mouth-needs-hole", "ONE TEST"]
        refused(lambda: stl.drawn_form(job, args(class_mouth_needs_hole=True), files, 100), "say why with --form-why")
        assert "--class-mouth-needs-hole" not in stl.drawn_form(job, args(), files, 100), "the hole option is on by default"
        refused(lambda: stl.drawn_form(job, args(class_mouth_needs_hole=True, mouth="drawn", form_why="X"), files, 100), "acts on the mouth by the check")
        # M12 & M17: --darken-head passed on with reason, off by default; under closed_lip_line, silent windows leave the mouth out while voiced windows draw it
        assert stl.drawn_form(job, args(darken_head=True, form_why="DARKER HEAD"), files, 100)[-2:] == ["--darken-head", "DARKER HEAD"]
        refused(lambda: stl.drawn_form(job, args(darken_head=True), files, 100), "say why with --form-why")
        assert "--darken-head" not in stl.drawn_form(job, args(), files, 100), "--darken-head is on by default"
        rf_dark = stl.reference_form({"reference": {"closed_lip_line": True, "darken_head": True, "why": "testb plan"}}, "a")
        assert rf_dark["darken_head"] == {"said_by": "the plan", "why": "testb plan"}, rf_dark
        eye_dark = next(d for d in stl.defaults_carried({}, {}, [{"label": "a"}], "a", None, rf_dark, {"said_by": "plan", "why": "z"}, None) if d["what"] == "the eye")
        assert "DARKENED HEAD MODEL" in eye_dark["carries"] and "darken_head: the plan: testb plan" in eye_dark["from"], eye_dark
        # M17: under closed_lip_line, a silent window leaves the mouth out ("left out") and passes --closed-lip-line so eye/brow/nose marks clip to the head
        made.plan["subjects"]["a"]["reference"] = {"closed_lip_line": True, "darken_head": True, "why": "M17 voice window rule"}
        made.plan["subjects"]["a"]["voice"] = [{"first": 100, "last": 159, "voice": "silent", "by": "THE OWNER"}]
        d_silent = made.load(name="a_fp_silent")
        assert d_silent["job"]["reference"]["mouth"]["form"] == "left out" and "silent window" in d_silent["job"]["reference"]["mouth"]["why"], d_silent["job"]["reference"]["mouth"]
        df_silent = stl.drawn_form(d_silent["job"], args(), files, 100)
        assert df_silent[:4] == ["--iris", "contrast", "--mouth", "left out"] and "--closed-lip-line" in df_silent and "--darken-head" in df_silent, df_silent
        del made.plan["subjects"]["a"]["reference"]
        d_silent_cli = made.load(name="a_fp_silent_cli")
        df_silent_cli = stl.drawn_form(d_silent_cli["job"], args(closed_lip_line=True, form_why="CLI M17"), files, 100)
        assert df_silent_cli[:4] == ["--iris", "contrast", "--mouth", "left out"] and "--closed-lip-line" in df_silent_cli, df_silent_cli
        made.plan["subjects"]["a"]["reference"] = {"closed_lip_line": True, "darken_head": True, "why": "M17 voice window rule"}
        made.plan["subjects"]["a"]["voice"] = [{"first": 100, "last": 159, "voice": "talks, not heard", "by": "THE OWNER"}]
        d_voiced = made.load(name="a_fp_voiced")
        assert d_voiced["job"]["reference"]["mouth"]["form"] == "auto", d_voiced["job"]["reference"]["mouth"]
        del made.plan["subjects"]["a"]["reference"], made.plan["subjects"]["a"]["voice"]
        refused(lambda: stl.drawn_form(job, args(), {"classes": None}, 100), "the preview saved none")
        assert stl.drawn_form({}, args(), files, 100) == [], "a job written before the defaults is drawn as the tool's own"
        # a load whose part is hair alone on some frames (`--parts-on`, 1010d's 1025-1031): no mouth there, said with the job's own words
        hair = {**job, "part": {"by": "THE OWNER'S ANSWER", "parts_on": [{"first": 1025, "last": 1031, "parts": ["hair"]}],
                                "made": [{"classes_on": [{"first": 1025, "last": 1031, "classes": ["Hair"]},
                                                         {"first": 1040, "last": 1041, "classes": ["Face_Neck", "Upper_Lip", "Lower_Lip"]}]}]}}
        got = stl.drawn_form(hair, args(), files, 100)
        # THE CASE THAT GOES RED when the reference is not told the load's lipless frames, or is told a range that holds the lips
        assert got[got.index("--no-mouth-on"):got.index("--no-mouth-on") + 2] == ["--no-mouth-on", "1025-1031=the load's part is hair alone there (THE OWNER'S ANSWER)"] \
            and got.count("--no-mouth-on") == 1, got
        # ... and no line of the face there (second_hp_5461b, 2026-10-15); the span that holds her face is drawn
        assert got[got.index("--no-face-on"):got.index("--no-face-on") + 2] == ["--no-face-on", "1025-1031=the load's part is hair alone there (THE OWNER'S ANSWER)"] \
            and got.count("--no-face-on") == 1, got
        # a load of the hair alone on every frame (`--parts hair`): no line of the face on any of its frames
        whole = {**job, "first": 5461, "frames": 37, "part": {"by": "THE LEAD'S RULING", "parts": ["hair"], "made": [{"classes": ["Hair"]}]}}
        got = stl.drawn_form(whole, args(), files, 100)
        # THE CASE THAT GOES RED when a hair load's reference draws a face (5461's eyes, brows and nose on her crown)
        assert got[-2:] == ["--no-face-on", "5461-5497=the load's part is hair alone on every frame (THE LEAD'S RULING)"], got
        assert "--no-face-on" not in stl.drawn_form({**whole, "part": {**whole["part"], "made": [{"classes": ["Hair", "Face_Neck"]}]}}, args(), files, 100)
        # the plan's restore span of her own glasses (2823's 3063-3097): the reference told it, with the plan's words; her hands' span not
        glasses = {**job, "restore": [{"classes": ["Eyeglass"], "first": 3063, "last": 3097, "by": "THE PLAN'S WORD"},
                                      {"classes": ["Left_Hand", "Right_Hand"], "first": 3060, "last": 3097, "by": "THE PLAN'S WORD"}]}
        got = stl.drawn_form(glasses, args(), files, 100)
        # THE CASE THAT GOES RED when a span of her own glasses is not passed on, or a span of her hands is (895 and 897: another face)
        assert got[-2:] == ["--over-the-eyes", "3063-3097=the plan gives back her Eyeglass there (THE PLAN'S WORD)"] and got.count("--over-the-eyes") == 1, got
    return ("the job prints and keeps the defaults it carries, each with where it comes from (the eye in contrast, the mouth by the lip check, zoomed where "
            "the node allows); the plan's reference form stands over them with its reason, and not without one; the reference step's word with its reason over both")

def the_figures() -> str:
    from _lib import face_points as fp
    kept = fp.load, fp.table
    line = cfg.POSE_READING["lid_gap_px_to_draw"]
    rows = [{"face_height_px": 40.0, "lid_gap_px": line * 0.25}, {"face_height_px": 50.0, "lid_gap_px": line * 0.5}, {"face_height_px": 60.0, "lid_gap_px": line * 2}]
    try:
        fp.load, fp.table = (lambda path: {}), (lambda points: rows)
        got = stl.figures_of(Path("made.npz"), 3.0, cues=lambda path: {"read from": str(path)})
    finally:
        fp.load, fp.table = kept
    c, r = got["at_the_canvas"], got["in_the_reference"]
    assert got["frames"] == 3 and got["the_drawing_line_px"] == line, got
    assert c["face_height_px"] == {"least": 40.0, "median": 50.0, "most": 60.0} and c["frames_with_the_gap_under_the_line"] == 2, c
    # at three times the size the half-line gap clears the line and the quarter-line gap does not
    assert r["enlarged"] == 3.0 and r["face_height_px"] == {"least": 120.0, "median": 150.0, "most": 180.0} and r["frames_with_the_gap_under_the_line"] == 1, r
    assert "form" not in got and "verdict" not in json.dumps(got), "the figures carry a reading"
    assert got["where_a_face_may_not_show"] == {"read from": "made.npz"}, got
    return "the face's height and the lash-line gap are the same rows at the canvas and at the reference's scale, each with its frames under the line, and no reading"


def a_load_of_two_subjects() -> str:
    """`pair` (the lead's ruling of 2026-10-15): two chain-made loads of the same frames as one load of two subjects, the larger region first,
    each subject's own keys, keep-outs and reference carried, neither subject kept out of the other, the whole frame, one margin."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        m = Machine(root)
        pv = root / "data" / "2026-01-02_pv_job"
        pv.mkdir(parents=True)
        (pv / "job.json").write_text(json.dumps({"job": {"name": "pv", "first": 100, "frames": 60, "output": "Video/v/s",
                                                        "subjects": [{"label": "b"}, {"label": "a"}]}}))
        (pv / "preview.queued.json").write_text(json.dumps({"status": "success", "files": []}))
        inputs = root / "in"
        inputs.mkdir()
        plan = inputs / "plan.json"
        plan.write_text(json.dumps({"subjects": {"a": {"region": "face and hair"}, "b": {"region": "face"}}}))

        def job(subject: str, other: str, margin: int, keep: list, **load_more) -> Path:
            ref = root / f"ref_{subject}.mkv"
            ref.write_bytes(b"")
            path = inputs / f"{subject}_ld.json"
            subjects = [{"label": subject, "still": f"{subject}.png", "pieces": "largest"}, {"label": other}]
            path.write_text(json.dumps({"job": {"name": f"{subject}_ld", "source": "/in/copy.mov", "first": 110, "frames": 20, "canvas": [64, 48],
                                                "output": "Video/v/s", "capture": f"data/2026-01-03_{subject}_ld", "preview": "data/2026-01-02_pv_job",
                                                "subjects": subjects, "texts": {f"{subject}_ld": {"file": "x.txt", "frames": [110, 129], "by": ""}},
                                                "keep_outs": {"of": subject, "made": keep}, "reference": {"points": f"{subject}_points.npz"},
                                                "restore": [{"classes": ["Left_Hand"], "first": 0, "last": 999, "by": f"{subject.upper()}'S"}],
                                                "left_out": [{"subject": subject, "first": 100, "frames": 10, "why": "W"}],
                                                "defaults": [{"what": "zoom", "carries": "z", "from": "f"}, {"what": "the eye", "carries": "e", "from": "f"}]},
                                        "defaults": {"margin": margin, "seed": 7, "carried": "held", "motion": "mesh", "motion_edge": 768},
                                        "loads": [{"name": f"{subject}_ld", "first": 110, "frames": 20, "subject": subject, "others": [other] + [k["name"] for k in keep],
                                                   "text": f"{subject}_ld", "motion_file": str(ref), "motion_first": 110, "render_region": "part", **load_more}]}))
            return path
        strangers = {"name": "strangers", "track": "/s.mkv", "frame_hash": "h"}
        ja = job("a", "b", 16, [{"name": "held", "track": "/held.mkv", "classes": []}, strangers])
        jb = job("b", "a", 64, [strangers])        # the larger margin the second's: the pair takes the larger, not the first's
        a_ = lambda first, second, **more: args(**{"first": str(first), "second": str(second), "plan": str(plan), "folder": str(inputs),  # noqa: E731
                                                   "name": None, "day": "2026-01-03", "remake": False, "by": "the session", **more})
        kept, stl.load_chain.video_of = stl.load_chain.video_of, lambda ctx, data: "v"
        try:
            # THE CASES THAT GO RED: the smaller region first, two loads of other frames, a setting that differs, a reference not made
            refused(lambda: stl.step_pair(a_(jb, ja), m), "b (face) is given first", "the larger goes first")
            d = json.loads(jb.read_text())
            d["loads"][0]["first"] = 111
            jb2 = inputs / "b2.json"
            jb2.write_text(json.dumps(d))
            refused(lambda: stl.step_pair(a_(ja, jb2), m), "differ in their frames")
            d = json.loads(jb.read_text())
            d["defaults"]["seed"] = 8
            jb2.write_text(json.dumps(d))
            refused(lambda: stl.step_pair(a_(ja, jb2), m), "settings differ in seed")
            d = json.loads(jb.read_text())
            d["loads"][0]["motion_file"] = str(root / "nowhere.mkv")
            jb2.write_text(json.dumps(d))
            refused(lambda: stl.step_pair(a_(ja, jb2), m), "its head reference is not made")
            # what one load carries once: a window (--seam) on one of them, a band or a held keep-out on the second
            for change, words in ((lambda d: d["loads"][0].update(window=294), "differ in their load's window"),
                                  (lambda d: d["loads"][0].update(context_noise=0.8), "differ in their load's context_noise"),
                                  (lambda d: d["loads"][0].update(hair_band=0.2), "the second load has `hair_band`"),
                                  (lambda d: d["job"]["keep_outs"]["made"].append({"name": "held", "track": "/b_held.mkv"}) or d["loads"][0]["others"].append("held"),
                                   "a held keep-out of its own")):
                d = json.loads(jb.read_text())
                change(d)
                jb2.write_text(json.dumps(d))
                refused(lambda: stl.step_pair(a_(ja, jb2), m), words)
            # each subject's own `hands`, in its own file: the second's is renamed, in its own others as well
            d = json.loads(jb.read_text())
            d["job"]["keep_outs"]["made"].append({"name": "hands", "track": "/b_hands.mkv", "classes": ["Left_Hand"]})
            d["loads"][0]["others"].append("hands")
            jb2.write_text(json.dumps(d))
            d = json.loads(ja.read_text())
            d["job"]["keep_outs"]["made"].append({"name": "hands", "track": "/a_hands.mkv", "classes": ["Left_Hand"]})
            d["loads"][0]["others"].append("hands")
            ja2 = inputs / "a2.json"
            ja2.write_text(json.dumps(d))
            stl.step_pair(a_(ja2, jb2, name="twin"), m)
            twin = json.loads((inputs / "twin.json").read_text())
            # the kept frames' noise both loads carry is the pair's (the cold reader of 2026-10-15: the pair dropped it and kept its row)
            for src, dst in ((ja, inputs / "a3.json"), (jb, inputs / "b3.json")):
                d = json.loads(src.read_text())
                d["loads"][0]["context_noise"] = 0.8
                dst.write_text(json.dumps(d))
            stl.step_pair(a_(inputs / "a3.json", inputs / "b3.json", name="noised"), m)
            noised = json.loads((inputs / "noised.json").read_text())
            # THE CASE THAT GOES RED when the pair does not carry the noise both its loads name
            assert noised["loads"][0].get("context_noise") == 0.8, noised["loads"][0]
            stl.step_pair(a_(ja, jb), m)
        finally:
            stl.load_chain.video_of = kept
        out = json.loads((inputs / "a_b_0110.json").read_text())
        job_, load = out["job"], out["loads"][0]
        first, second = load["subjects"]
        # THE CASE THAT GOES RED when the pair's name inside its file is not the one it is written under (2026-10-15: the last keep-out's)
        assert job_["name"] == load["name"] == load["text"] == "a_b_0110" and list(job_["texts"]) == ["a_b_0110"] \
            and job_["texts"]["a_b_0110"]["file"] == "a_b_0110.txt" and job_["capture"] == "data/2026-01-03_a_b_0110", (job_["name"], load["name"], job_["texts"])
        assert twin["job"]["name"] == twin["loads"][0]["name"] == "twin", twin["job"]["name"]
        # THE CASE THAT GOES RED when a subject is given the other's reference video (a crossed pair) or the first's twice
        assert (first["motion_file"], second["motion_file"]) == (str(root / "ref_a.mkv"), str(root / "ref_b.mkv")) \
            and first["motion_first"] == second["motion_first"] == 110, (first, second)
        assert job_["keep_outs"]["of"] == ["a", "b"] and set(job_["keep_outs"]["by"]) == {"a", "b"}, job_["keep_outs"]
        tk = {k["name"]: k for k in twin["job"]["keep_outs"]["made"]}
        assert (tk["hands"]["track"], tk["hands"]["of"], tk["hands_b"]["track"], tk["hands_b"]["of"]) == ("/a_hands.mkv", "a", "/b_hands.mkv", "b"), tk
        assert twin["loads"][0]["subjects"][1]["others"] == ["strangers", "hands_b"] and "hands" in twin["loads"][0]["subjects"][0]["others"], twin["loads"][0]
        assert (first["subject"], first["region"], first["still"], second["subject"], second["region"], second["still"]) == ("a", "face and hair", "a.png", "b", "face", "b.png"), load
        assert first["others"] == ["held", "strangers"] and second["others"] == ["strangers"], "a subject of the pair is kept out of the other"
        assert first["motion_edge"] == second["motion_edge"] == 768 and "render_region" not in load and "subject" not in load, load
        assert out["defaults"] == {"margin": 64, "seed": 7}, f"one margin, the larger; nothing the builder reads off the list: {out['defaults']}"
        assert [k["name"] for k in job_["keep_outs"]["made"]] == ["held", "strangers"] and job_["keep_outs"]["made"][0]["of"] == "a" \
            and "of" not in job_["keep_outs"]["made"][1], job_["keep_outs"]
        assert [r["subject"] for r in job_["restore"]] == ["a", "b"] and [x["subject"] for x in job_["left_out"]] == ["a", "b"], job_
        assert job_["whole_frame"]["why"] == stl.PAIR_WHOLE_FRAME_WHY and job_["reference"]["points"] == "a_points.npz" and job_["reference_2"]["points"] == "b_points.npz"
        assert [x["what"] for x in job_["defaults"]] == ["a: the eye", "b: the eye", "the pair"], "a subject's zoom row carried into a load on the whole frame"
        assert [s["label"] for s in job_["subjects"]] == ["a", "b"] and job_["subjects"][1].get("pieces") == "largest", job_["subjects"]
        # the chain reads it: the first subject's, with the second beside it
        ctx = stl.load_chain.read_job(inputs / "a_b_0110.json", root, need_text=False)
        assert (ctx["subject"], ctx["also"]) == ("a", "b"), (ctx["subject"], ctx["also"])
        # and its text is a person's: the generated text is one subject's
        refused(lambda: stl.step_text(args(job_file=str(inputs / "a_b_0110.json"), writer=None, place_and_action=None, plan=str(plan), by_hand=None, by="s"), m),
                "a load of two subjects takes its text written whole")
    return ("two loads of the same frames make one load of two subjects, the larger region first, each with its own still, reference, keys and "
            "keep-outs, neither kept out of the other, one margin, the whole frame, no zoom row; the smaller region first, other frames, a setting "
            "that differs and a reference not made are refused; the chain reads it; its text is written whole")
def where_a_face_may_not_show() -> str:
    """The two cues printed under a reference's figures (the lead's ruling of 2026-10-15): ears reversed, and the face points' scores under
    the floor, as runs of source frames with their figures and what each cannot see; data, nothing decided."""
    import tempfile
    import numpy as np
    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_eyebrow_0", "upper_lip_0"]
    # frames 200-205: facing the camera (right ear left of the left), turned away on 202-203, hidden on 205
    kp = np.zeros((6, len(names), 2), np.float32)
    sc = np.full((6, len(names)), 0.9, np.float32)
    for t in range(6):
        kp[t, names.index("left_shoulder"), 0], kp[t, names.index("right_shoulder"), 0] = 200, 100
        kp[t, names.index("right_ear"), 0], kp[t, names.index("left_ear"), 0] = (170, 130) if t in (2, 3) else (130, 170)
    kp[3, names.index("right_ear"), 0] = 180
    sc[5, [names.index(n) for n in ("left_eye", "right_eye", "left_eyebrow_0", "upper_lip_0")]] = 0.1
    sc[4, [names.index(n) for n in ("left_eye", "right_eye", "left_eyebrow_0", "upper_lip_0")]] = stl.FACE_SCORE_FLOOR       # at the floor: not under it
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p_points.npz"
        np.savez(path, keypoints=kp, scores=sc, names=np.array(names), first_frame=np.int64(200), size=np.array([64, 64]))
        got = stl.face_cues(path)
    # THE CASE THAT GOES RED when the ears' sign is read the other way, a run is not of source frames, or the floor takes a frame at it
    assert got["ears_reversed"] == [{"first": 202, "last": 203, "least": 40.0, "most": 50.0}], got["ears_reversed"]
    assert got["face_score_under_the_floor"] == [{"first": 205, "last": 205, "least": 0.1, "most": 0.1}], got["face_score_under_the_floor"]
    assert "profile" in got["cannot_see"] and "5461" in got["cannot_see"], got["cannot_see"]
    # an ear the pose model barely saw: that frame's ears are not read, and it is said
    sc[3, names.index("left_ear")] = stl.POINT_SCORE_FLOOR - 0.01
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p_points.npz"
        np.savez(path, keypoints=kp, scores=sc, names=np.array(names), first_frame=np.int64(200), size=np.array([64, 64]))
        got = stl.face_cues(path)
        assert [(r["first"], r["last"]) for r in got["ears_reversed"]] == [(202, 202)] and [(r["first"], r["last"]) for r in got["ears_not_read_on"]] == [(203, 203)], got
        # a points file without what the cues read says so, and does not stop the step
        np.savez(path, keypoints=kp, scores=sc, names=np.array(names))
        assert stl.face_cues(path) == {"not_read": "the points file holds no first_frame"}, stl.face_cues(path)
        np.savez(path, keypoints=kp, scores=sc, names=np.array(names[:3]), first_frame=np.int64(200))
        assert "left_ear, right_ear, left_shoulder, right_shoulder" in stl.face_cues(path)["not_read"]
        keep = [n for n in names if not any(w in n for w in ("eye", "brow", "lip"))]
        np.savez(path, keypoints=kp[:, [names.index(n) for n in keep]], scores=sc[:, [names.index(n) for n in keep]], names=np.array(keep),
                 first_frame=np.int64(200))
        assert stl.face_cues(path) == {"not_read": "the points file names no eye, brow or lip point"}, stl.face_cues(path)
    return ("ears reversed on a head turned away and not on one facing the camera, scaled by the shoulders; the face points' scores under the "
            "floor and not at it; each as runs of source frames with their least and most, and what the cues cannot see")


def a_span_folder_a_render_read() -> str:
    """The lead's ruling of 2026-10-16: never move or write into a reference folder any render read. A load's reference goes to
    `<subject>_<first>_<last>` while it is not there, or is there of today's form and read by no render; else to `_b`, `_c`, and the
    job says why; `--existing` keeps its refusal words and names the folder a load made again takes."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        data = made.root / "data"
        plain = made.load()
        base = Path(plain["loads"][0]["motion_file"]).parent
        # NO FOLDER YET: the plain name, as ever, and the job says nothing of a new folder
        assert base.name == "a_0100_0159" and "reference_folder" not in plain["job"], (base, plain["job"].get("reference_folder"))
        # A FOLDER A RENDER READ: a queue record names a file in it (its record of today's form, so only the reading moves it on)
        base.mkdir(parents=True)
        (base / "a_head_reference_0100_0159.json").write_text(json.dumps({"frames": [100, 159], "passed": True, "face_on_the_track": {"known": True}}))
        (base / "a_head_reference_0100_0159.mp4").write_bytes(b"the frames a laid piece read")
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
        assert Path(made.load(data=data)["loads"][0]["motion_file"]).parent == base, "a folder of today's form read by no render was not taken as it is"
        (data / "2026-01-02_old_job").mkdir(parents=True)
        (data / "2026-01-02_old_job" / "old.render.queued.json").write_text(json.dumps({"digests": {str(base / "a_head_reference_0100_0159.mp4"): "x"}}))
        b = made.load(data=data)
        # THE CASE THAT GOES RED when a load is pointed at a folder a render read
        assert Path(b["loads"][0]["motion_file"]).parent.name == "a_0100_0159_b", b["loads"][0]["motion_file"]
        assert b["job"]["reference_folder"] == {"folder": "a_0100_0159_b", "why": "the span's folder a_0100_0159 was read by old; a new one"}, b["job"]["reference_folder"]
        assert "ITS REFERENCE GOES TO a_0100_0159_b, A NEW FOLDER: the span's folder a_0100_0159 was read by old" in b["job"]["note"], b["job"]["note"]
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()} == before, "the folder a render read was touched"
        # A RENDER WHOSE QUEUE RECORD NAMES NO INPUT (before 2026-10-15) and whose queued graph does: read all the same
        (data / "2026-01-02_old_job" / "old.render.queued.json").write_text(json.dumps({"prompt_id": "p"}))
        (data / "2026-01-02_old_job" / "old.render.json").write_text(json.dumps({"7": {"inputs": {"video": str(base / "a_head_reference_0100_0159.mp4")}}}))
        # THE CASE THAT GOES RED when only the queue record is read
        assert Path(made.load(data=data)["loads"][0]["motion_file"]).parent.name == "a_0100_0159_b", "a folder named only by a queued graph was taken"
        # _b THERE TOO, NOT OF TODAY'S FORM: _c, each folder's why said
        (base.with_name("a_0100_0159_b")).mkdir()
        c = made.load(data=data)
        assert Path(c["loads"][0]["motion_file"]).parent.name == "a_0100_0159_c" and "the span's folder a_0100_0159_b is not of today's form" in c["job"]["reference_folder"]["why"], c["job"]["reference_folder"]
        # --existing on the folder a render read, its record no longer of today's form: its words, and the folder a load made again takes
        job_file = made.jobs / "a_fp_0100.json"
        job_file.write_text(json.dumps(plain))
        (base / "a_0100_0159_points.npz").write_bytes(b"")
        (base / "a_head_reference_0100_0159.json").write_text(json.dumps({"frames": [100, 159], "passed": True}))
        e = args(existing=True)
        e.job_file = str(job_file)
        made.m.data = data
        refused(lambda: stl.step_reference(e, made.m, lambda *a_, **k_: {}), "does not say its face was held to a's track",
                "(a load made again takes the folder a_0100_0159_c: `shot_to_load.py job --remake`)")
    return "no folder: the plain name; a folder a render read: _b, untouched, the job saying why; _b there too and not of today's form: _c; --existing names the next"


def carry_words_from_refuses_uncarried_word_key() -> str:
    """D20 (1) & D20b (7): any word-bearing key in the old job that the tool does not know how to carry is refused, naming the key."""
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        base = made.load()
        for where, mut, key_named in (
            ("top", lambda d: d.update(extra_top="x"), "extra_top"),
            ("defaults", lambda d: d["defaults"].update(extra_def=1), "defaults.extra_def"),
            ("loads_len", lambda d: d["loads"].append(dict(d["loads"][0])), "expected 1 load"),
            ("load_subject", lambda d: d["loads"][0].update(subject="b"), "old load's subject is 'b', not 'a'"),
            ("subject", lambda d: d["job"]["subjects"][1].update(mystery_word="a person's unknown setting"), "subjects.a.mystery_word"),
            ("least_pair", lambda d: d["job"]["subjects"][1].update(least=300), "least and least_by must both be present"),
            ("word_pair", lambda d: d["job"]["subjects"][1].update(word="110-112"), "word and word_by must both be present"),
            ("drop_pair", lambda d: d["job"]["subjects"][1].update(drop="110-112"), "drop and drop_by must both be present"),
            ("other subject", lambda d: d["job"]["subjects"][0].update(least=300, least_by="someone"), "subjects.b.least"),
            ("job", lambda d: d["job"].update(who_is_who="a person's note"), "job.who_is_who"),
            ("keep_outs", lambda d: d["job"].update(keep_outs={"of": "a", "by": "x", "preview": "p", "first": 100, "made": [], "extra_ko": 1}),
             "job.keep_outs.extra_ko"),
            ("by_hand_text", lambda d: d["job"]["texts"]["a_fp_0100"].update(by="mrgan, BY HAND: custom"), "job.texts.a_fp_0100.by"),
            ("reference", lambda d: d["job"]["reference"].update(custom_line="on 110-120"), "job.reference.custom_line"),
            ("reference_form", lambda d: d["job"]["reference"].update(form="plain"), "job.reference.form"),
            ("mouth", lambda d: d["job"]["reference"]["mouth"].update(said_by="mrgan"), "reference.mouth"),
            ("iris", lambda d: d["job"]["reference"]["iris"].update(said_by="mrgan"), "reference.iris"),
            ("job_defaults_row", lambda d: d["job"]["defaults"].append({"what": "mystery_default", "carries": "x", "from": "y"}), "mystery_default"),
            ("load", lambda d: d["loads"][0].update(continue_from="earlier"), "loads[0].continue_from"),
        ):
            bad = json.loads(json.dumps(base))
            mut(bad)
            bad_file = Path(tmp) / f"bad_{where.replace(' ', '_')}.json"
            bad_file.write_text(json.dumps(bad))
            refused(lambda bf=bad_file: made.load(name="a_fp_0100b", carry_words_from=bf), key_named)
    return "any word-bearing key in the old job that the tool does not know how to carry is refused, naming the key"


def carry_words_from_equals_old_on_every_word() -> str:
    """D20 (1, 3) & D20b (1, 2, 3, 5, 7): --carry-words-from copies every word-bearing setting of the old job for the frames the new
    job covers, keeps the original author and time (never restamped as the carrying session's), attaches a `carried` record with the
    old job's name, file and sha256, records `carried_words_from` on the job, prints overrides as OVERRIDDEN with both values, and
    handles --drop / --drop-by on the CLI and in load_job."""
    import contextlib
    import hashlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        sheet = Path(tmp) / "sheet_100.jpg"
        sheet.write_bytes(b"jpg")
        made.plan["subjects"]["a"]["parts"] = ["hair", "face_and_neck"]
        shots, plan = made.read()
        pjob = made.preview(100)
        pfolder = made.preview_ran(pjob)
        saved = json.loads((pfolder / "preview.queued.json").read_text())["files"]
        old = stl.load_job(
            shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
            "mrgan", "2026-01-02", "a_hfp_0105",
            first=105,
            frames=45,
            now="2026-10-16T00:10:00Z",
            preview_files=saved,
            why_before=f"100-104 cut before: head off frame ({sheet})",
            why_after=f"150-159 cut after: back turned ({sheet})",
            least=480,
            least_why=f"floor 480 on small face ({sheet})",
            hold_part=[(110, 112, f"mrgan looked at {sheet}: face shows in profile"), (125, 125, f"mrgan looked at {sheet}: face shows; blurred")],
            drop="115-118+140-142",
            drop_by=f"mrgan looked at {sheet}: hair only on 115-118 and 140-142",
            no_face_on=[(115, 118, f"mrgan looked at {sheet}: only her hair shows, no face"), (140, 142, f"mrgan looked at {sheet}: turned away")],
            no_mouth_on=[(130, 132, f"mrgan looked at {sheet}: hand over mouth")],
            mouth_on=[(135, 136, f"mrgan looked at {sheet}: mouth open singing")],
            eyes_shut_on=[(120, 122, f"mrgan looked at {sheet}: eyes shut")],
            eyes_shut_form="lids",
            parts_on=[(115, 118, ["hair"])],
            parts_by=f"mrgan looked at {sheet}: hair alone on 115-118",
            zoomed=f"mrgan looked at {sheet}: close crop",
            track_not=[("b", 128, 129, f"mrgan looked at {sheet}: b's track lies on a")],
            track_not_by=f"mrgan, on {sheet}",
            no_strangers=f"mrgan looked at {sheet}: nobody else in frame",
        )
        old_file = Path(tmp) / "a_hfp_0105.json"
        old_file.write_text(json.dumps(old))
        old_sha = hashlib.sha256(old_file.read_bytes()).hexdigest()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            carried = stl.load_job(
                shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                "mrtaco", "2026-01-03", "a_hfp_0105b", now="2026-10-16T22:00:00Z", preview_files=saved,
                carry_words_from=old_file,
            )
        printed = buf.getvalue()
        old_s = next(s for s in old["job"]["subjects"] if s["label"] == "a")
        new_s = next(s for s in carried["job"]["subjects"] if s["label"] == "a")
        # Fix 2: job["carried_words_from"] records old job name, file, sha256, by, at
        cwf_rec = carried["job"].get("carried_words_from") or {}
        assert cwf_rec.get("from") == "a_hfp_0105" and cwf_rec.get("file") == str(old_file) and cwf_rec.get("sha256") == old_sha \
            and cwf_rec.get("by") == "mrtaco" and cwf_rec.get("at") == "2026-10-16T22:00:00Z", f"carried_words_from mismatch: {cwf_rec}"
        assert old_sha[:16] in printed and str(old_file) in printed, printed
        # Fix 1: original author ("mrgan") and time ("2026-10-16T00:10:00Z") preserved, never restamped with "mrtaco, 2026-10-16T22:00:00Z"
        assert new_s.get("least") == old_s["least"] == 480, f"carried job did not carry least=480: {new_s.get('least')}"
        assert new_s.get("least_by", "").startswith("mrgan, 2026-10-16T00:10:00Z:") and "mrtaco" not in new_s.get("least_by", "") \
            and new_s.get("least_by", "").endswith("(carried from a_hfp_0105)") and new_s.get("least_carried", {}).get("sha256") == old_sha, new_s
        assert new_s.get("word") == old_s["word"] == "110-112+125-125", f"carried job did not carry word: {new_s.get('word')}"
        assert new_s.get("word_by", "").startswith("110-112: mrgan, 2026-10-16T00:10:00Z:") and "mrtaco" not in new_s.get("word_by", "") \
            and new_s.get("word_by", "").count("(carried from a_hfp_0105)") == 2 and new_s.get("word_carried", {}).get("sha256") == old_sha, new_s
        assert new_s.get("drop") == old_s["drop"] == "115-118+140-142", f"carried job did not carry drop: {new_s.get('drop')}"
        assert new_s.get("drop_by", "").startswith("mrgan, 2026-10-16T00:10:00Z:") and "mrtaco" not in new_s.get("drop_by", "") \
            and new_s.get("drop_by", "").endswith("(carried from a_hfp_0105)") and new_s.get("drop_carried", {}).get("sha256") == old_sha, new_s
        nf_spans = carried["job"]["reference"].get("no_face_on") or []
        assert [(x["first"], x["last"]) for x in nf_spans] == [(115, 118), (140, 142)], f"carried no_face_on mismatch: {nf_spans}"
        for x in nf_spans:
            assert x["said_by"] == "mrgan" and x["at"] == "2026-10-16T00:10:00Z" and "mrgan, 2026-10-16T00:10:00Z" not in x["why"] \
                and x["why"].endswith("(carried from a_hfp_0105)") and x.get("carried", {}).get("sha256") == old_sha, x
        nm_spans = carried["job"]["reference"]["mouth"].get("no_mouth_on") or []
        m_spans = carried["job"]["reference"]["mouth"].get("mouth_on") or []
        assert [(x["first"], x["last"]) for x in nm_spans] == [(130, 132)], f"carried no_mouth_on mismatch: {nm_spans}"
        assert [(x["first"], x["last"]) for x in m_spans] == [(135, 136)], f"carried mouth_on mismatch: {m_spans}"
        for x in nm_spans + m_spans:
            assert x["said_by"] == "mrgan" and x["at"] == "2026-10-16T00:10:00Z" and "mrgan, 2026-10-16T00:10:00Z" not in x["why"] \
                and x["why"].endswith("(carried from a_hfp_0105)") and x.get("carried", {}).get("sha256") == old_sha, x
        es_block = carried["job"]["reference"]["iris"].get("eyes_shut") or {}
        assert es_block.get("form") == "lids", f"carried eyes_shut form mismatch: {es_block}"
        assert [(x["first"], x["last"]) for x in (es_block.get("spans") or [])] == [(120, 122)], f"carried eyes_shut spans mismatch: {es_block}"
        for x in es_block.get("spans") or []:
            assert x["said_by"] == "mrgan" and x["at"] == "2026-10-16T00:10:00Z" and "mrgan, 2026-10-16T00:10:00Z" not in x["why"] \
                and x["why"].endswith("(carried from a_hfp_0105)") and x.get("carried", {}).get("sha256") == old_sha, x
        assert [(x["first"], x["frames"]) for x in carried["job"]["left_out"]] == [(100, 5), (150, 10)], f"carried left_out mismatch: {carried['job']['left_out']}"
        assert all(x["why"].endswith("(carried from a_hfp_0105)") and x.get("carried", {}).get("sha256") == old_sha for x in carried["job"]["left_out"]), carried["job"]["left_out"]
        part_block = carried["job"].get("part") or {}
        assert part_block.get("parts_on") == [{"first": 115, "last": 118, "parts": ["hair"]}], f"carried parts_on mismatch: {part_block}"
        assert part_block.get("by", "").endswith("(carried from a_hfp_0105)") and part_block.get("carried", {}).get("sha256") == old_sha, part_block
        zoom_block = carried["job"].get("zoomed") or {}
        assert zoom_block.get("said_by") == "mrgan" and zoom_block.get("at") == "2026-10-16T00:10:00Z" \
            and "mrgan, 2026-10-16T00:10:00Z" not in zoom_block.get("why", "") and zoom_block.get("why", "").endswith("(carried from a_hfp_0105)") \
            and zoom_block.get("carried", {}).get("sha256") == old_sha, f"carried zoomed mismatch: {zoom_block}"
        tn_block = carried["job"].get("track_not") or {}
        tn_made = tn_block.get("made") or [{}]
        assert tn_made[0].get("frames") == [[128, 129]], f"carried track_not frames mismatch: {tn_block}"
        assert (tn_made[0].get("why") or [""])[0].endswith("(carried from a_hfp_0105)") \
            and tn_block.get("carried", {}).get("sha256") == old_sha, f"carried track_not why/carried mismatch: {tn_block}"
        st_row = {x["what"]: x for x in carried["job"]["defaults"]}["the strangers keep-out"]
        assert st_row["from"].startswith("TURNED OFF on a person's word (mrgan, 2026-10-16T00:10:00Z):") \
            and "mrtaco" not in st_row["from"] and st_row["from"].endswith("(carried from a_hfp_0105)"), st_row
        assert "WORDS CARRIED FROM a_hfp_0105:" in printed, printed
        graphs.check_job(carried["job"])
        # Chained carry (a_hfp_0105 -> a_hfp_0105b -> a_hfp_0105c) accepts the `carried` metadata and preserves original author "mrgan"
        carried_file = Path(tmp) / "a_hfp_0105b.json"
        carried_file.write_text(json.dumps(carried))
        carried_c = stl.load_job(
            shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
            "mrpen", "2026-01-04", "a_hfp_0105c", now="2026-10-16T23:00:00Z", preview_files=saved,
            carry_words_from=carried_file,
        )
        assert carried_c["job"]["reference"]["no_face_on"][0]["said_by"] == "mrgan" and carried_c["job"]["zoomed"]["said_by"] == "mrgan"
        # Carry whole_frame, motion=none, seam, and context_noise (Fix 1 & Fix 7)
        old_wf = stl.load_job(
            shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
            "mrgan", "2026-01-02", "a_hp_0100", now="2026-10-16T00:10:00Z", preview_files=saved,
            whole_frame="wide group shot", parts=["hair"], parts_by="mrgan: hair only", motion="none", motion_why="bowed head",
        )
        # Add a multi-window seam and context_noise to test carrying seam and context_noise with original author/time
        old_wf["loads"][0]["window"] = 294
        old_wf["loads"][0]["context_noise"] = 0.8
        old_wf["job"]["defaults"].append({"what": "the seam", "carries": "windows of [294, 243]: window 2 writes from 120", "from": "mrgan, 2026-10-16T00:10:00Z: calm at 120"})
        old_wf["job"]["defaults"].append({"what": "the kept frames' noise", "carries": "0.8: each later window", "from": "mrgan, 2026-10-16T00:10:00Z: turn at 120"})
        wf_file = Path(tmp) / "a_hp_0100.json"
        wf_file.write_text(json.dumps(old_wf))
        real_seam, real_cn = stl.seam_of, stl.context_noise_of
        stl.seam_of = lambda n, a_, sm, sw, by_, now_: (294, {"what": "the seam", "carries": "windows of [294, 243]: window 2 writes from 120", "from": f"{by_}, {now_}: {sw}"}) if sm else (None, None)
        stl.context_noise_of = lambda n, cn, cw_, by_, now_: {"what": "the kept frames' noise", "carries": f"{cn:g}", "from": f"{by_}, {now_}: {cw_}"} if cn is not None else None
        try:
            carried_wf = stl.load_job(
                shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                "mrtaco", "2026-01-03", "a_hp_0100b", now="2026-10-16T22:00:00Z", preview_files=saved,
                carry_words_from=wf_file,
            )
        finally:
            stl.seam_of, stl.context_noise_of = real_seam, real_cn
        wf_block = carried_wf["job"].get("whole_frame") or {}
        assert wf_block.get("said_by") == "mrgan" and wf_block.get("at") == "2026-10-16T00:10:00Z" \
            and wf_block.get("why", "").endswith("(carried from a_hp_0100)"), f"carried whole_frame did not preserve original said_by/at/tag: {wf_block}"
        assert carried_wf["loads"][0].get("motion") == "none" and carried_wf["job"]["reference"]["why"].startswith("mrgan, 2026-10-16T00:10:00Z:") \
            and "mrtaco" not in carried_wf["job"]["reference"]["why"] and carried_wf["job"]["reference"]["why"].endswith("(carried from a_hp_0100)"), carried_wf["job"]["reference"]
        wf_defs = {x["what"]: x for x in carried_wf["job"]["defaults"]}
        ref_def = wf_defs.get("the reference video") or {}
        assert ref_def.get("from", "").startswith("mrgan, 2026-10-16T00:10:00Z:") \
            and "mrtaco" not in ref_def.get("from", "") and ref_def.get("from", "").endswith("(carried from a_hp_0100)"), f"reference video default not carried: {ref_def}"
        seam_def = wf_defs.get("the seam") or {}
        assert carried_wf["loads"][0].get("window") == 294 and seam_def.get("from", "").startswith("mrgan, 2026-10-16T00:10:00Z:") \
            and "mrtaco" not in seam_def.get("from", "") and seam_def.get("from", "").endswith("(carried from a_hp_0100)"), f"seam not carried: {seam_def}"
        cn_def = wf_defs.get("the kept frames' noise") or {}
        assert carried_wf["loads"][0].get("context_noise") == 0.8 and cn_def.get("from", "").startswith("mrgan, 2026-10-16T00:10:00Z:") \
            and "mrtaco" not in cn_def.get("from", "") and cn_def.get("from", "").endswith("(carried from a_hp_0100)"), f"context_noise not carried: {cn_def}"
        # Fix 3: Overriding a carried word prints OVERRIDDEN with both values, and does not list the old value under WORDS CARRIED
        buf_ov = io.StringIO()
        with contextlib.redirect_stdout(buf_ov):
            overridden = stl.load_job(
                shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                "mrtaco", "2026-01-03", "a_hfp_0105_ov", now="2026-10-16T22:00:00Z", preview_files=saved,
                carry_words_from=old_file, least=300, least_why=f"new floor on {sheet}",
            )
        ov_out = buf_ov.getvalue()
        ov_s = next(s for s in overridden["job"]["subjects"] if s["label"] == "a")
        carried_section = ov_out.split("WORDS OVERRIDDEN")[0] if "WORDS OVERRIDDEN" in ov_out else ov_out
        assert ov_s["least"] == 300 and "least=480" not in carried_section and "OVERRIDDEN" in ov_out and "300" in ov_out and "480" in ov_out, ov_out
        # Fix 5: load_job drop pairing and range refusals, and CLI --drop / --drop-by
        refused(lambda: made.load(name="a_fp_drop1", drop="110-115"), "drop and drop_by go together")
        refused(lambda: made.load(name="a_fp_drop2", drop_by="someone"), "drop and drop_by go together")
        refused(lambda: made.load(name="a_fp_drop3", drop="90-95", drop_by="someone"), "drop 90-95 is not inside the load's frames 100-159")
        refused(lambda: made.load(name="a_fp_drop4", drop="120-115", drop_by="someone"), "drop 120-115 is not inside the load's frames 100-159")
        cli_a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(pfolder),
                     drop="110-112", drop_by="mrtaco: hidden")
        stl.step_job(cli_a, made.m)
        cli_d = json.loads((made.jobs / "a_fp_0100.json").read_text())
        assert next(s for s in cli_d["job"]["subjects"] if s["label"] == "a")["drop"] == "110-112"
    return ("every word-bearing key of the old job is carried onto the new job for the same frames, keeping its original author and time "
            "with a `carried` record and `job.carried_words_from`, printing overrides with both values, and checking `--drop`/`--drop-by`")


def carry_words_from_refuses_span_outside_new_frames() -> str:
    """D20 (1, 3) & D20b (4, 7): any carried span outside or straddling the new job's frames is refused unless named in --leave-word
    with --leave-by; --leave-word drops named spans (including overruled left_out on a whole-shot remake) and records them; --why*
    lists the old left_out it replaces."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        sheet = Path(tmp) / "sheet_100.jpg"
        sheet.write_bytes(b"jpg")
        made.plan["subjects"]["a"]["parts"] = ["hair", "face_and_neck"]
        shots, plan = made.read()
        pjob = made.preview(100)
        pfolder = made.preview_ran(pjob)
        saved = json.loads((pfolder / "preview.queued.json").read_text())["files"]
        old = stl.load_job(
            shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
            "mrgan", "2026-01-02", "a_hfp_0105",
            first=105,
            frames=50,
            now="2026-10-16T00:10:00Z",
            preview_files=saved,
            why_before=f"100-104 cut ({sheet})",
            why_after=f"155-159 cut ({sheet})",
            no_face_on=[(106, 108, f"mrgan looked at {sheet}: no face")],
            hold_part=[(150, 152, f"mrgan looked at {sheet}: face shows")],
            drop="151-153",
            drop_by=f"mrgan looked at {sheet}: drop tail",
            parts_on=[(106, 108, ["hair"])],
            parts_by=f"mrgan looked at {sheet}: hair only on 106-108",
        )
        old_file = Path(tmp) / "a_hfp_0105.json"
        old_file.write_text(json.dumps(old))
        # All outside spans (no_face_on, hold_part, drop, parts_on) are named in the refusal
        refused(
            lambda: stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                                 "mrtaco", "2026-01-03", "a_hfp_0110", first=110, frames=35, why="cut both ends",
                                 preview_files=saved, carry_words_from=old_file),
            "no_face_on 106-108",
            "hold_part (word) 150-152",
            "drop 151-153",
            "parts_on 106-108",
            "outside the new job's frames 110-144",
        )
        # Expanding to whole shot without --leave-word refuses the old left_out spans
        refused(
            lambda: stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                                 "mrtaco", "2026-01-03", "a_hfp_0100", first=100, frames=60,
                                 preview_files=saved, carry_words_from=old_file),
            "left_out 100-104",
            "left_out 155-159",
        )
        # A span straddling the new edge (106-115 when new load is 110-144) is refused, never clipped
        straddle = json.loads(json.dumps(old))
        straddle["job"]["reference"]["no_face_on"] = [{"first": 106, "last": 115, "why": f"mrgan looked at {sheet}", "said_by": "mrgan", "at": "now"}]
        straddle_file = Path(tmp) / "straddle.json"
        straddle_file.write_text(json.dumps(straddle))
        refused(
            lambda: stl.load_job(shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                                 "mrtaco", "2026-01-03", "a_hfp_0110s", first=110, frames=35, why="cut both ends",
                                 preview_files=saved, carry_words_from=straddle_file,
                                 leave_word=["hold_part:150-152", "drop:151-153", "parts_on:106-108"], leave_by="the owner, 2026-10-16: narrowed"),
            "no_face_on 106-115",
            "outside the new job's frames 110-144",
        )
        # Fix 4: --leave-word with --leave-by drops outside spans on a narrower load, records left_words, and prints WORDS LEFT OUT
        # Also test that --why lists the old left_out spans it replaces under OVERRIDDEN
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            narrow = stl.load_job(
                shots, plan, stl.shot_of(shots, 100), "a", pjob["job"], str(pfolder), "v", made.m.output,
                "mrtaco", "2026-01-03", "a_hfp_0110_ok", first=110, frames=35, why="cut both ends wider",
                preview_files=saved, carry_words_from=old_file,
                leave_word=["no_face_on:106-108", "hold_part:150-152", "drop:151-153", "parts_on:106-108"],
                leave_by="the owner, 2026-10-16: narrowed to 110-144",
            )
        narrow_out = buf.getvalue()
        assert "no_face_on" not in narrow["job"]["reference"]
        assert "word" not in next(s for s in narrow["job"]["subjects"] if s["label"] == "a")
        assert "drop" not in next(s for s in narrow["job"]["subjects"] if s["label"] == "a")
        assert "part" not in narrow["job"]
        left_words = narrow["job"]["carried_words_from"].get("left_words") or []
        assert [(w["kind"], w["first"], w["last"]) for w in left_words] == [
            ("hold_part", 150, 152), ("drop", 151, 153), ("parts_on", 106, 108), ("no_face_on", 106, 108)
        ], left_words
        assert "WORDS LEFT OUT FROM a_hfp_0105:" in narrow_out and "left_out 100-104: OVERRIDDEN" in narrow_out and "left_out 155-159: OVERRIDDEN" in narrow_out, narrow_out
        # Fix 4: The dad's overruled left_out on a whole-shot remake (--leave-word left_out:155-159 --leave-by "the owner, 2026-10-16: overruled")
        dad_old = made.load(
            name="a_hfp_0100_dad", first=100, frames=55, why_after=f"155-159 cut: arm covers face ({sheet})",
            least=400, least_why=f"mrgan looked at {sheet}: floor 400",
        )
        dad_file = Path(tmp) / "a_hfp_0100_dad.json"
        dad_file.write_text(json.dumps(dad_old))
        buf_dad = io.StringIO()
        with contextlib.redirect_stdout(buf_dad):
            dad_whole = made.load(
                name="a_hfp_0100_dad_b", carry_words_from=dad_file,
                leave_word=["left_out:155-159"], leave_by="the owner, 2026-10-16: overruled, keep whole shot",
            )
        assert (dad_whole["loads"][0]["first"], dad_whole["loads"][0]["frames"]) == (100, 60), dad_whole["loads"][0]
        assert dad_whole["job"]["left_out"] == [], dad_whole["job"]["left_out"]
        assert next(s for s in dad_whole["job"]["subjects"] if s["label"] == "a")["least"] == 400
        assert [(w["kind"], w["first"], w["last"]) for w in dad_whole["job"]["carried_words_from"]["left_words"]] == [("left_out", 155, 159)]
        assert "WORDS LEFT OUT FROM a_hfp_0100_dad:" in buf_dad.getvalue() and "left_out 155-159" in buf_dad.getvalue()
        # Refusals for --leave-word: missing --leave-by, no --carry-words-from, or naming a span not in old job
        refused(lambda: made.load(name="a_hfp_lw1", carry_words_from=dad_file, leave_word=["left_out:155-159"]), "--leave-word and --leave-by go together")
        refused(lambda: made.load(name="a_hfp_lw2", leave_word=["left_out:155-159"], leave_by="someone"), "--leave-word goes with --carry-words-from")
        refused(lambda: made.load(name="a_hfp_lw3", carry_words_from=dad_file, leave_word=["no_face_on:110-120"], leave_by="someone"),
                "has no such span")
    return "carried spans outside or straddling the new frames are refused unless named in --leave-word with --leave-by, and replaced left_out spans are listed"


def carry_words_from_retired_keys_not_carried() -> str:
    """D20 (2, 3) & D20b (6): keys the rules of 2026-10-16 retired (pieces/pieces_by, job.restore, job.keep_outs.stay) and keys rebuilt from
    the plan/preview (loads[0].hair_band, loads[0].render_region, loads[0].others, job.keep_outs.by) are NOT carried from the old job and are said on stdout."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        sheet = Path(tmp) / "sheet_100.jpg"
        sheet.write_bytes(b"jpg")
        old = made.load(name="a_fp_0100", least=480, least_why=f"mrgan looked at {sheet}: floor 480")
        old_s = next(s for s in old["job"]["subjects"] if s["label"] == "a")
        old_s["pieces"] = "largest"
        old_s["pieces_by"] = "the plan (mrtwo, 2026-10-14: old rule)"
        old["job"]["restore"] = [{"classes": ["Apparel"], "touching": "head", "first": 0, "last": 999, "by": "mrlol, 2026-10-15: old rule"}]
        old["loads"][0]["hair_band"] = 0.6
        old["job"]["keep_outs"] = {"of": "a", "by": "the plan (old keep_outs_by)", "preview": "shot_c_0100", "first": 100,
                                   "made": [{"name": "clothing", "stay": {"what": "apron"}}]}
        old_file = Path(tmp) / "a_fp_0100.json"
        old_file.write_text(json.dumps(old))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            carried = made.load(name="a_fp_0100b", carry_words_from=old_file)
        printed = buf.getvalue()
        new_s = next(s for s in carried["job"]["subjects"] if s["label"] == "a")
        assert "pieces" not in new_s and "pieces_by" not in new_s, f"retired pieces/pieces_by was carried: {new_s}"
        assert "restore" not in carried["job"], f"retired job.restore was carried: {carried['job'].get('restore')}"
        assert "hair_band" not in carried["loads"][0], f"old hair_band was carried without the plan asking: {carried['loads'][0]}"
        for key_label in ("pieces/pieces_by", "job.restore", "job.keep_outs.stay", "loads[0].hair_band", "loads[0].render_region", "loads[0].others", "job.keep_outs.by"):
            assert key_label in printed and "not carried" in printed, f"{key_label!r} not said as not carried on stdout: {printed!r}"
    return "retired keys and plan/preview-rebuilt keys (hair_band, render_region, others, keep_outs.by) are not carried from the old job and are said on stdout"


def remake_requires_carry_words_or_no_carry_words() -> str:
    """D20c (C15 follow-up): `shot_to_load.py job --remake` when an earlier job exists for the same shot and subject is refused unless
    `--carry-words-from <old>` or `--no-carry-words "<why>"` is given; `--no-carry-words` records the reason in `job.no_carry_words` and
    prints `NO WORDS CARRIED: ...`, and the two flags cannot both be given."""
    import contextlib
    import io
    import os
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        folder = made.preview_ran(made.preview(100))
        a = args(shots=str(made.shots_file), plan=str(made.plan_file), shot=100, folder=str(made.jobs), subject="a", preview=str(folder))
        stl.step_job(a, made.m)
        first_job = made.jobs / "a_fp_0100.json"
        assert first_job.is_file(), f"initial job not written: {first_job}"
        # 1. --remake without --carry-words-from or --no-carry-words is refused, naming the existing job and both flags
        a_remake = args(**{**vars(a), "remake": True})
        refused(lambda: stl.step_job(a_remake, made.m), "--remake on a in 100-159", "a_fp_0100", "--carry-words-from", "--no-carry-words")
        assert not (made.jobs / "a_fp_0100b.json").exists(), "refused remake wrote a job file"
        # CLI subprocess refusal check (no Python traceback, exit code != 0)
        bench_dir = str(Path(__file__).resolve().parent)
        cli_res = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {bench_dir!r}); from pathlib import Path; "
             f"import load_chain as lc, shot_to_load as stl; lc.REPO = stl.REPO = Path({str(made.root)!r}); sys.exit(stl.main())",
             "job", "--shots", str(made.shots_file), "--plan", str(made.plan_file), "--shot", "100",
             "--subject", "a", "--preview", str(folder), "--folder", str(made.jobs),
             "--by", "the session", "--day", "2026-01-03", "--remake"],
            capture_output=True, text=True,
            env={**os.environ, "H3_OUTPUT_DIR": str(made.m.output), "H3_INPUT_DIR": str(made.root / "input")},
        )
        cli_out = cli_res.stdout + cli_res.stderr
        assert cli_res.returncode != 0 and "Traceback" not in cli_out and "a_fp_0100" in cli_out \
            and "--carry-words-from" in cli_out and "--no-carry-words" in cli_out, f"CLI refusal failed: {cli_out!r}"
        # 2. Empty/whitespace --no-carry-words is refused in step_job and load_job
        refused(lambda: stl.step_job(args(**{**vars(a_remake), "no_carry_words": "   "}), made.m),
                "--no-carry-words takes a person's words")
        refused(lambda: made.load(name="a_fp_ncw_empty", no_carry_words="  "),
                "--no-carry-words takes a person's words")
        # 3. Giving both --carry-words-from and --no-carry-words is refused in step_job and load_job
        refused(lambda: stl.step_job(args(**{**vars(a_remake), "carry_words_from": str(first_job), "no_carry_words": "both given"}), made.m),
                "--carry-words-from and --no-carry-words cannot both be given")
        refused(lambda: made.load(name="a_fp_both", carry_words_from=first_job, no_carry_words="both given"),
                "--carry-words-from and --no-carry-words cannot both be given")
        # 4. --remake with --no-carry-words succeeds, prints NO WORDS CARRIED, and records job.no_carry_words
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stl.step_job(args(**{**vars(a_remake), "no_carry_words": "fresh look on re-seeded preview"}), made.m)
        printed = buf.getvalue()
        second_job = made.jobs / "a_fp_0100b.json"
        assert second_job.is_file(), f"remake with --no-carry-words did not write {second_job}"
        d_b = json.loads(second_job.read_text())
        ncw_rec = d_b["job"].get("no_carry_words") or {}
        assert ncw_rec.get("why") == "fresh look on re-seeded preview" and ncw_rec.get("by") == "the session" and ncw_rec.get("at"), f"no_carry_words record mismatch: {ncw_rec}"
        assert "NO WORDS CARRIED: fresh look on re-seeded preview (the session," in printed, f"NO WORDS CARRIED not printed: {printed!r}"
        graphs.check_job(d_b["job"])
        # 5. --remake with --carry-words-from on a job that itself had no_carry_words succeeds and writes a_fp_0100c.json
        stl.step_job(args(**{**vars(a_remake), "carry_words_from": str(second_job)}), made.m)
        third_job = made.jobs / "a_fp_0100c.json"
        assert third_job.is_file(), f"remake with --carry-words-from did not write {third_job}"
        d_c = json.loads(third_job.read_text())
        cwf_c = d_c["job"].get("carried_words_from") or {}
        assert cwf_c.get("from") == "a_fp_0100b" and "no_carry_words" not in d_c["job"], f"carried job c mismatch: {d_c['job']}"
    return ("--remake when an earlier job exists on the same shot and subject is refused unless --carry-words-from or --no-carry-words is given; "
            "--no-carry-words records job.no_carry_words and prints NO WORDS CARRIED, and the two flags are mutually exclusive")


def carry_words_from_refuses_drop_over_face_frame() -> str:
    """C136-C139: each carried drop and no_face_on is printed with who said it and when; carrying a drop or no_face_on
    over a frame where the old capture sees her face is refused (listing those frames for a look); and narrowing/overriding
    to true back_of_head frames succeeds."""
    import contextlib
    import io
    import numpy as np
    import head_tube_parts as htp
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        sheet = Path(tmp) / "sheet_100.jpg"
        sheet.write_bytes(b"jpg")
        # Build an old job with drop=105-110 and no_face_on=105-110
        old = made.load(
            name="a_fp_0100",
            drop="105-110",
            drop_by=f"mrgan looked at {sheet}: hair only; entered 2026-10-16T18:00:00Z by the session",
            no_face_on=[(105, 110, f"mrgan looked at {sheet}: hair only")],
        )
        old_file = made.jobs / "a_fp_0100.json"
        old_file.write_text(json.dumps(old))

        # Create a synthetic capture for a_fp_0100 at made.root / "data" / "2026-01-03_a_fp_0100"
        # Frames 105-108 (rel 5-8): back_of_head (Hair only in tube, 0 px Face_Neck).
        # Frames 109-110 (rel 9-10): profile/facing face showing (Face_Neck = 576 px >= least_px=480).
        cap_dir = made.root / "data" / "2026-01-03_a_fp_0100"
        subj_dir = cap_dir / "subjects" / "a"
        subj_dir.mkdir(parents=True, exist_ok=True)
        n_frames, h, w = 60, 48, 64
        track = np.zeros((n_frames, h, w), dtype=bool)
        track[:, 6:44, 14:50] = True
        parts = np.zeros((n_frames, h, w), dtype=bool)
        parts[:, 16:40, 20:44] = True  # 24x24 = 576 px on normal frames
        parts[5:9] = False  # frames 105-108 have 0 Face_Neck px (back_of_head)
        face_id = htp.NAME_TO_ID["Face_Neck"]
        hair_id = htp.NAME_TO_ID["Hair"]
        cmap = np.zeros((n_frames, h, w), dtype=np.uint8)
        cmap[:, 8:16, 20:44] = hair_id
        cmap[parts] = face_id
        cmap[5:9, 8:40, 20:44] = hair_id  # 100% Hair on 105-108 (back_of_head)

        np.savez_compressed(
            subj_dir / "masks__shot_c_0100.npz",
            track=np.packbits(track, axis=-1),
            parts=np.packbits(parts, axis=-1),
            covered=np.ones(n_frames, dtype=bool),
            shape=np.array([n_frames, h, w]),
        )
        np.savez_compressed(subj_dir / "classes__shot_c_0100.npz", classes=cmap)
        (subj_dir / "part_grades.json").write_text(
            json.dumps({"least_px": 480, "part_is_made_of": ["Face_Neck"]})
        )
        (cap_dir / "manifest.json").write_text(
            json.dumps({"name": "2026-01-03_a_fp_0100", "first_frame": 100, "frames": n_frames, "size": [w, h]})
        )

        # 1. Carrying old_file directly is REFUSED because frames 109-110 show her face (576 px >= 480)
        refused(
            lambda: made.load(name="a_fp_0100b", carry_words_from=old_file),
            "refused to carry drop / no_face_on silently onto frame(s) 109-110",
            "drop 105-110 (the session, now) has face on 109-110",
            "no_face_on 105-110 (the session, now) has face on 109-110",
        )

        # 2. Narrowing drop and no_face_on to the true back_of_head frames (105-108) with a fresh look succeeds,
        #    and carrying that narrowed job prints who said drop and no_face_on and when.
        narrowed = made.load(
            name="a_fp_0100b",
            carry_words_from=old_file,
            drop="105-108",
            drop_by=f"owner looked at {sheet}: 105-108 back of head, 109-110 profile face",
            no_face_on=[(105, 108, f"owner looked at {sheet}: 105-108 back of head")],
        )
        narrowed_file = made.jobs / "a_fp_0100b.json"
        narrowed_file.write_text(json.dumps(narrowed))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            carried_clean = made.load(name="a_fp_0100c", carry_words_from=narrowed_file)
        printed = buf.getvalue()
        assert "drop=105-108 (the session, now):" in printed and "no_face_on 105-108 (the session, now):" in printed, (
            f"expected carried drop and no_face_on to print author and time: {printed!r}"
        )
        subj_c = next(s for s in carried_clean["job"]["subjects"] if s["label"] == "a")
        assert subj_c["drop"] == "105-108", subj_c["drop"]
    return "carried drop and no_face_on print who said them and when, refuse frames where the capture sees her face, and pass on back_of_head frames"


def depth_in_the_job_step() -> str:
    """`shot_to_load.py job --depth`: finds the frames where two tracks (after `track_not`) or a keep-out overlap, runs
    `bench/measure_moge2_depth.py` on just those frames into the capture folder, records `overlap_frames` and `file` in
    `job.depth`, and names the default in `THE DEFAULTS THIS LOAD CARRIES`; off by default."""
    import capture_masked_run as cap
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        refused(lambda: made.load(depth="  "), "--depth takes the reason")
        folder = made.preview_ran(made.preview())
        n_frames, h, w = 60, 48, 64
        t_a = np.zeros((n_frames, h, w), dtype=bool)
        t_a[:, 8:40, 8:40] = True
        t_b = np.zeros((n_frames, h, w), dtype=bool)
        t_b[5:8, 16:32, 16:32] = True       # overlaps a on source frames 105-107
        t_b[10:12, 16:32, 16:32] = True     # overlaps a on source frames 110-111 (emptied by track_not below)
        t_c = np.zeros((n_frames, h, w), dtype=bool)
        t_held = np.zeros((n_frames, h, w), dtype=bool)
        t_held[20, 16:24, 16:24] = True     # keep-out held overlaps a on source frame 120
        made.plan["subjects"]["a"]["keep_outs"] = ["held"]
        made.plan["subjects"]["a"]["keep_outs_by"] = "the session, on sheet.jpg"
        made.write()
        kept_read = cap.read_mask

        def fake_read_mask(path, *_a):
            name = Path(path).name
            if "_a_track_" in name:
                return t_a.copy(), np.ones(n_frames, dtype=bool)
            if "_b_track_" in name:
                return t_b.copy(), np.ones(n_frames, dtype=bool)
            if "_c_track_" in name:
                return t_c.copy(), np.ones(n_frames, dtype=bool)
            if "_a_held_things_" in name:
                return t_held.copy(), np.ones(n_frames, dtype=bool)
            return np.zeros((n_frames, h, w), dtype=bool), np.ones(n_frames, dtype=bool)

        cap.read_mask = fake_read_mask
        try:
            cap_dir = made.root / "data" / "2026-01-03_a_fp_0100"

            def write_depth():
                cap_dir.mkdir(parents=True, exist_ok=True)
                (cap_dir / "depth.npz").write_bytes(b"npz")

            def copy_track(track, frames, first, to):
                Path(to).parent.mkdir(parents=True, exist_ok=True)
                Path(to).write_bytes(b"copy")
                return {"frames": n_frames, "px_emptied": 5}

            made.m.on["moge2 depth"] = write_depth
            stl.step_job(
                args(
                    shots=str(made.shots_file),
                    plan=str(made.plan_file),
                    shot=100,
                    folder=str(made.jobs),
                    subject="a",
                    preview=str(folder),
                    track_not=["b:110-111=b is a on 110-111"],
                    track_not_by="the session",
                    depth="dad crosses in front on 105-107",
                ),
                made.m,
                track_cut=copy_track,
                over=lambda *_x: ["track over track: a figure"],
                hashed=lambda path: f"HASH OF {Path(path).name}",
            )
            assert "moge2 depth" in made.m.labels(), made.m.labels()
            argv = made.m.said("moge2 depth")
            assert argv[1].endswith("bench/measure_moge2_depth.py"), argv
            assert argv[argv.index("--frames") + 1] == "105,106,107,120", argv
            assert Path(argv[argv.index("--out-dir") + 1]) == cap_dir, argv
            written = json.loads((made.jobs / "a_fp_0100.json").read_text())
            dp = written["job"]["depth"]
            assert dp["on"] is True and dp["overlap_frames"] == [105, 106, 107, 120] and dp["file"] == "data/2026-01-03_a_fp_0100/depth.npz", dp
            d_row = next(x for x in written["job"]["defaults"] if x["what"] == "depth")
            assert "4 overlap frame(s) (105-107, 120)" in d_row["carries"] and "dad crosses in front" in d_row["from"], d_row

            # Carry words from a job with depth carries depth_meta
            carried = made.load(name="a_fp_0100b", carry_words_from=made.jobs / "a_fp_0100.json")
            assert carried["job"]["depth"]["on"] is True and "(carried from a_fp_0100)" in carried["job"]["depth"]["why"], carried["job"]["depth"]

            # With no overlap frames in the load, measure_moge2_depth.py is not called
            t_b[:] = False
            t_held[:] = False
            made.m.ran.clear()
            stl.step_job(
                args(
                    shots=str(made.shots_file),
                    plan=str(made.plan_file),
                    shot=100,
                    folder=str(made.jobs),
                    subject="a",
                    preview=str(folder),
                    remake=True,
                    no_carry_words="testing empty overlap",
                    depth="no crossing on this span",
                ),
                made.m,
                over=lambda *_x: ["track over track: a figure"],
            )
            assert "moge2 depth" not in made.m.labels(), made.m.labels()
            written_b = json.loads((made.jobs / "a_fp_0100b.json").read_text())
            assert written_b["job"]["depth"]["overlap_frames"] == [] and written_b["job"]["depth"]["file"] is None, written_b["job"]["depth"]
            d_row_b = next(x for x in written_b["job"]["defaults"] if x["what"] == "depth")
            assert "0 overlap frames in the load" in d_row_b["carries"], d_row_b
        finally:
            cap.read_mask = kept_read
    return "job --depth finds track and keep-out overlap frames (after track_not), runs measure_moge2_depth.py into the capture folder, records job.depth and defaults, and skips the model when 0 frames overlap"


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
case("a plan that keeps the strangers out", a_plan_that_keeps_the_strangers_out)
case("who else is in the shot", who_else_is_in_the_shot)
case("her clothing as one stay keep-out, retired", her_clothing_as_one_stay_keep_out)
case("a span folder a render read", a_span_folder_a_render_read)
case("a track that is not its person on named frames", a_track_that_is_not_its_person)
case("text: who wrote it, said with a flag", the_text_step)
case("a job renders from the video's present copy", a_job_renders_from_the_present_copy)
case("reference: on the load's frames, drawn by default", the_reference_step)
case("the defaults this load carries", the_defaults_this_load_carries)
case("the figures, at the canvas and in the reference", the_figures)
case("pair: two loads of the same frames as one of two subjects", a_load_of_two_subjects)
case("where a face may not show: two cues, printed as data", where_a_face_may_not_show)
case("carry-words-from: a word-bearing key left uncarried is refused", carry_words_from_refuses_uncarried_word_key)
case("carry-words-from: a carried job equals the old one on every word for the same frames", carry_words_from_equals_old_on_every_word)
case("carry-words-from: a span outside the new frames is said and refused", carry_words_from_refuses_span_outside_new_frames)
case("carry-words-from: a retired key is not carried and is said", carry_words_from_retired_keys_not_carried)
case("carry-words-from: a remake on an existing shot and subject requires --carry-words-from or --no-carry-words", remake_requires_carry_words_or_no_carry_words)
case("carry-words-from: a carried drop or no_face_on over a face frame is refused", carry_words_from_refuses_drop_over_face_frame)
case("depth: overlap frames found, measured into the capture folder, and recorded in defaults", depth_in_the_job_step)

if __name__ == "__main__":
    sys.exit(finish())

