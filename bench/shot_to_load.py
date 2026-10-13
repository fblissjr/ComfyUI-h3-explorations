#!/usr/bin/env python3
"""From a shot to a load's job file: the preview of the shot, the load's job, its head reference. Then it stops for the text.

    <python> bench/shot_to_load.py preview --shots SHOTS.json --plan PLAN.json --shot FIRST --folder INPUTS --by SESSION [--cast CAST.jsonl] [--queue]
        (somebody looks at the trackers' tiles; a wrong pick is a person's line in the shots record, and the preview is made again)
    <python> bench/shot_to_load.py job --shots SHOTS.json --plan PLAN.json --shot FIRST --subject LABEL --preview PREVIEW_JOB_FOLDER \\
        --folder INPUTS --by SESSION [--first F --frames N --why "what a look cut"] [--remake]
    <python> bench/shot_to_load.py reference JOB.json --by SESSION [--plain "why"] [--model-store FOLDER] [--existing]
        (a person writes the load's text and says who wrote it; then `bench/load_chain.py first JOB.json`)

**What it buys.** The step before a load's chain was hand-work for every shot: a preview job typed from the shots
record, a pose pass and a head reference run with each other's outputs for arguments, and a job file copied from
the last one with its settings typed again. The video has some sixty shots still to make. This writes the job
files and runs the two tools, from the video's own records, and leaves to a person only what is a person's.

**What is read, and from where.**

- The shots record (`data/<video>/shots.json`): the shot's frames and who is in it. For a subject with no line in
  the cast file, the frame they are picked on is a frame the record names AS A FIELD (`frames_on`, then
  `frames_boxed`; `bench/page_answers_to_cast.py --shots` writes both), never a sentence of `how`; the shot's
  middle is the last fallback and the job's note says it was one. A shot's `order` (a person's list, written
  after a tile shows a wrong pick) is the order its trackers run in; else the plan's subject order, which is a
  starting value and nothing measured.
- The plan (`data/<video>/plan.json`): for each subject the `region` (what is replaced), and, new with this tool,
  the `still` its face is taken from, the `parts` the part model ticks for it, and optionally `hair_band`. A
  subject whose plan lacks `still` or `parts` is refused with the keys to add.
- The cast file (`--cast`): where it holds an answer inside the shot for a subject, that tracker is seeded from
  it (`bench/cast_file.py::seeds`; a "none of the outlined people" answer leaves the shot unseeded).
- The preview's own job folder (`--preview`, handed in by name): its subjects, its files, the frame they begin on.
- The video's analysis folder (`clip_folder.py`), found by the file a record was made on: a job is written to
  render from the video's PRESENT copy, whatever copy the shots record or the preview names, and a record made
  on a copy with another frame clock is refused (`working_copy`).
- `workflows/h3_config.py`: the canvas, and what goes into a job file's `defaults` (`MASKED_LOAD_DEFAULTS`,
  `MASKED_LOAD_MARGIN` by the plan's region, `SEED`), so the job file stands alone.

**The steps.**

- `preview`: ONE preview a shot, every subject the record says is in it, a pose pass for each. Each tracker picks
  the largest person on its frame, which is a guess: the step stops there for the tiles to be looked at.
  `--queue` builds and queues it on an empty queue; without it the two commands are printed.
- `job`: one subject's load on the shot, or on the frames a look left of it (`--first`, `--frames`, with `--why`;
  the frames cut go into `left_out` with those words). The name is made here: `<subject>_<form>_<first>`, a
  letter after it for a remake (`--remake`), and one that exists is refused. The job file names its preview
  (`job.preview`), its capture, its reference by where it WILL be, and a text file that is not written yet.
  A subject's `pick` is the preview's; it carries no `pick_at`, since a load tracks nobody.
- `reference`: on the LOAD's own frames, so the box is round the head on those frames: the pose pass in the
  subject's tracked box, then `bench/head_reference.py`, which writes the drawn form and the plain cut both. It
  goes to `Video/<video>/references/<subject>_<first>_<last>/`: a reference belongs to a shot, a subject and a
  span, never to a load's name. THE DRAWN FORM IS THE DEFAULT and nothing here decides otherwise: the face's
  height and the lash-line gap are printed at the canvas and as they come out in the reference at its scale,
  with the frames under the drawing line, and written into the job file. The plain cut is taken on a person's
  word only, `--plain "why"`, and the form, the why and who said it go into the job file. A load made again on
  the same frames reads the reference that is there: `--existing` runs nothing, holds the record beside it to
  the job's subject and frames, and writes the same block from it.

**It stops for the text.** The job file's text entry names a file and no writer. `bench/load_chain.py` refuses the
load until the file is there and `texts.<label>.by` says who wrote it.

**What it cannot see.** Whether a tracker took the right person (the tiles). Whether a face is too small or too
turned for the drawn lines (the sheet beside the reference, and the figures). What the frames a look should cut
are. Whether the plan's order suits a shot.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))

import load_chain  # noqa: E402
from load_chain import Refused  # noqa: E402

#: Where a job's renders go under the server's output folder. One place: today a session's folder under the
#: video's, until the Open List's row z21 is answered.
OUTPUT_KEY = "Video/{video}/{session}"
#: Where a head reference goes: by the video, the subject and the span it was made on.
REFERENCES = "Video/{video}/references/{subject}_{first:04d}_{last:04d}"
#: A load's form in its name, by the plan's region: the part, then `p` for the face points of the frozen form.
FORM = {"face": "fp", "face and hair": "hfp"}
#: What a preview's tracker picks with. A guess, which is why the preview step stops for the tiles.
PICK = "largest"
PLAN_KEYS = "`still` (the picture its face is taken from, under the server's input folder) and `parts` (the part model's ticks for it)"


def _load(shots_file: str, plan_file: str) -> tuple[dict, dict]:
    import shot_coverage
    shots = shot_coverage.load_shots(shots_file)
    return shots, shot_coverage.load_plan(plan_file, shots["subjects"])


def working_copy(name: str, world) -> tuple[str, Path, str]:
    """(the video's analysis folder's name, the file jobs render from now, words when `name` is not that file).

    `name` is the file a record was made on: the shots record's copy, a preview's source. The video is found by
    it, as its present copy or one it had before (`clip_folder.py`, "The copies a video has had"), and a job is
    written to render from the PRESENT copy whatever the record was made on: a load must not inherit a copy that
    was replaced (2026-10-13: a copy's sound was out of step with its picture, and a load written from a preview
    made on it would have loaded it). Refused when `name` is on another frame clock than the present copy: its
    frame numbers are not the video's."""
    import clip_folder
    folder = clip_folder.folder_of_file(name, world.data)
    manifest = clip_folder.read_manifest(folder)
    present = manifest.get("derived")
    if not present:
        raise Refused(f"{folder.name}'s manifest names no copy that jobs render from")
    if not clip_folder.same_clock(manifest, name):
        was = clip_folder.copy_named(manifest, name)
        raise Refused(f"{name} begins on source frame {was['first_source_frame']} and the video's working copy {present['name']} on frame 0: its frame "
                      "numbers are another clock, and nothing made on it is carried to a job here")
    said = ""
    if name not in (present["name"], manifest["source"]["name"]):
        was = clip_folder.copy_named(manifest, name)
        said = (f" MADE ON AN EARLIER COPY: {name}, which {present['name']} replaced ({was['why']}); both hold the source's frames on one clock, "
                f"and this job renders from {present['name']}.")
    return folder.name, clip_folder.find_source(present["name"], world.inputs), said


def shot_of(shots: dict, first: int) -> dict:
    shot = next((s for s in shots["shots"] if s["first"] == first), None)
    if shot is None:
        raise Refused(f"no shot of the shots record begins on frame {first}")
    return shot


def planned(plan: dict, label: str) -> dict:
    p = plan["subjects"][label]
    if not p.get("still") or not p.get("parts"):
        raise Refused(f"the plan says what is replaced of {label!r} ({p.get('region')}) and not what from: add {PLAN_KEYS} to the subject in the plan file")
    return p


def order_of(shot: dict, plan: dict, present: list[str]) -> tuple[list[str], str]:
    """The order a shot's trackers run in, and where it is from: the shot's own `order`, a person's, else the plan's."""
    if shot.get("order"):
        unknown = [x for x in shot["order"] if x not in present]
        if unknown or sorted(shot["order"]) != sorted(present):
            raise Refused(f"the shot's `order` {shot['order']} is not the subjects the record says are in it ({present})")
        return list(shot["order"]), "the shot's own `order` in the shots record, a person's"
    return [k for k in plan["subjects"] if k in present], "the plan's subject order, a starting value: nothing says who is larger on this shot"


def pick_of(shot: dict, label: str) -> tuple[int, str]:
    """The frame a subject's tracker picks on, and where it is from: a frame the record names as a field, else the middle."""
    who = shot["who"][label]
    for key, words in (("frames_on", "a frame the record says they were marked on"), ("frames_boxed", "a frame a box stands on in the cast file")):
        inside = [f for f in who.get(key) or [] if shot["first"] <= f <= shot["last"]]
        if inside:
            return inside[0], f"{words} ({key})"
    return (shot["first"] + shot["last"]) // 2, "THE SHOT'S MIDDLE: the record names no frame for them as a field. A fallback, not a frame anybody marked"


def seed_of(cast_lines: list[dict] | None, label: str, shot: dict, canvas) -> dict | None:
    """The tracker's seeds for a subject on a shot from the cast file, or None when it holds no box there."""
    if not cast_lines:
        return None
    import cast_file
    try:
        got = cast_file.seeds(cast_lines, label, shot["first"], canvas, [(0, shot["last"] - shot["first"] + 1)], cast_file.LEAVE)
    except ValueError as e:
        raise Refused(str(e)) from None
    return got if got["text"] else None


def preview_job(shots: dict, plan: dict, shot: dict, video: str, source: str, by: str, cast_lines=None, now: str = "") -> dict:
    """The job file of one shot's preview: every subject the record says is in it, a pose pass for each."""
    import h3_config
    present = [k for k in shots["subjects"] if shot["who"][k]["present"] == "yes"]
    if not present:
        raise Refused(f"the shots record says nobody of ours is in {shot['first']}-{shot['last']}: there is nothing to preview")
    order, order_from = order_of(shot, plan, present)
    canvas = list(h3_config.MASKED_JOB_CANVAS)
    subjects, said = [], []
    for label in order:
        p = planned(plan, label)
        one = {"label": label, "pick": PICK, "parts": list(p["parts"]), "still": p["still"]}
        seed = seed_of(cast_lines, label, shot, canvas)
        if seed is not None:
            one["seeds"] = seed["text"]
            said.append(f"{label}: SEEDED from the cast file ({'; '.join(r['why'] for r in seed['shots'] if r['seed'])}), so the order does not matter for them")
        else:
            one["pick_at"], where = pick_of(shot, label)
            said.append(f"{label}: the {PICK} person on frame {one['pick_at']}, {where}")
        subjects.append(one)
    frames = shot["last"] - shot["first"] + 1
    name = f"shot_c_{shot['first']:04d}"
    note = (f"Written by bench/shot_to_load.py for {by}, {now}. THE PREVIEW OF ONE SHOT, source frames {shot['first']} to {shot['last']} ({frames} frames): "
            f"every subject the shots record says is in it, a pose pass for each, nothing sampled. THE TRACKERS RUN IN THIS ORDER: {', '.join(order)} "
            f"({order_from}). THE PICKS: {'; '.join(said)}. A pick by the largest person is a guess: the tiles are looked at before anything is built on this "
            f"preview, and a wrong pick is corrected in the shots record (the shot's `order`, or a frame for the subject) and the preview made again.")
    return {"job": {"name": name, "note": note, "source": source, "first": shot["first"], "frames": frames, "canvas": canvas,
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects, "left_out": [], "texts": {}}, "defaults": {}, "loads": []}


def free_name(base: str, taken, remake: bool) -> str:
    """A load's name: the base, or with `remake` the base and the first letter from b that nothing has. `taken(name)`
    says whether a name is in use."""
    if not remake:
        if taken(base):
            raise Refused(f"a load named {base} exists: a load made again takes a new name (--remake gives it the next letter)")
        return base
    for letter in "bcdefghijklmnopqrstuvwxyz":
        if not taken(base + letter):
            return base + letter
    raise Refused(f"every letter after {base} is taken")


def load_job(shots: dict, plan: dict, shot: dict, subject: str, preview: dict, preview_folder: str, video: str, output: Path, by: str, day: str,
             name: str, first: int | None = None, frames: int | None = None, why: str = "", now: str = "", source: str | None = None) -> dict:
    """The job file of one subject's load on a shot, from the shot's preview. `preview` is the preview's own job
    block; `source` is the file the load renders from, the video's present copy (the preview's own when left out)."""
    import h3_config
    if shot["who"][subject]["present"] != "yes":
        raise Refused(f"the shots record does not say {subject!r} is in {shot['first']}-{shot['last']} (present: {shot['who'][subject]['present']})")
    theirs = {s["label"]: s for s in preview["subjects"]}
    if subject not in theirs:
        raise Refused(f"{subject!r} is not a subject of the preview {preview['name']} ({', '.join(theirs)})")
    if theirs[subject].get("pose", True) is False:
        raise Refused(f"the preview {preview['name']} made no pose pass for {subject!r}, so there is no body model to cut a reference from")
    if (preview["first"], preview["first"] + preview["frames"] - 1) != (shot["first"], shot["last"]):
        raise Refused(f"the preview {preview['name']} is of {preview['first']}-{preview['first'] + preview['frames'] - 1}, not of the shot {shot['first']}-{shot['last']}")
    p = planned(plan, subject)
    region = p["region"]
    if region not in h3_config.MASKED_LOAD_MARGIN:
        raise Refused(f"the plan's region for {subject!r} is {region!r}: this tool makes part loads ({', '.join(h3_config.MASKED_LOAD_MARGIN)})")
    a = shot["first"] if first is None else first
    n = shot["last"] - a + 1 if frames is None else frames
    b = a + n - 1
    if not (shot["first"] <= a <= b <= shot["last"]):
        raise Refused(f"frames {a}-{b} are not inside the shot {shot['first']}-{shot['last']}")
    cut = [(shot["first"], a - 1)] if a > shot["first"] else []
    cut += [(b + 1, shot["last"])] if b < shot["last"] else []
    if cut and not why.strip():
        raise Refused(f"the load is {a}-{b} of the shot {shot['first']}-{shot['last']}: say what a look cut and why (--why); it goes into the job's `left_out`")
    reference = output / REFERENCES.format(video=video, subject=subject, first=a, last=b) / f"{subject}_head_reference_{a:04d}_{b:04d}.mp4"
    load = {"name": name, "first": a, "frames": n, "subject": subject, "others": [k for k in theirs if k != subject], "text": name,
            "motion_file": str(reference), "motion_first": a}
    band = p.get("hair_band")
    if band is not None:
        load["hair_band"] = band["value"] if isinstance(band, dict) else band
    subjects = [{k: v for k, v in s.items() if k in ("label", "phrase", "pick", "parts", "still", "pose")} for s in preview["subjects"]]
    note = (f"Written by bench/shot_to_load.py for {by}, {now}. ONE LOAD: {subject}, {region}, source frames {a} to {b} ({n} frames) of the shot "
            f"{shot['first']}-{shot['last']}, its masks from the preview {preview['name']} (job.preview), in the frozen form: the body model cut to the head "
            f"with the face points drawn, made on these frames by this tool's `reference` step. Its settings are in `defaults`, from workflows/h3_config.py. "
            f"THE TEXT IS NOT WRITTEN: {name}.txt beside this file is a person's, and texts.{name}.by says who wrote it."
            + (f" CUT BY A LOOK: {why.strip()}" if cut else ""))
    return {"job": {"name": name, "note": note, "source": source or preview["source"], "first": a, "frames": n, "canvas": list(preview["canvas"]),
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects,
                    "left_out": [{"subject": subject, "first": x, "frames": y - x + 1, "why": why.strip()} for x, y in cut],
                    "capture": f"data/{day}_{name}", "preview": preview_folder,
                    "texts": {name: {"file": f"{name}.txt", "frames": [a, b], "by": ""}},
                    "reference": {"form": "drawn", "why": "the form for a face, head or hair replacement (AGENTS.md, Settled)", "made": None}},
            "defaults": {**h3_config.MASKED_LOAD_DEFAULTS, "margin": h3_config.MASKED_LOAD_MARGIN[region], "seed": h3_config.SEED},
            "loads": [load]}


def figures_of(points: Path, scale: float) -> dict:
    """The face's height and the lash-line gap along a load, at the canvas and as they come out in the reference at
    its scale, each with the frames under the drawing line. Figures; nothing is decided from them here."""
    import h3_config
    import numpy as np
    from _lib import face_points as fp
    rows = fp.table(fp.load(points))
    line = h3_config.POSE_READING["lid_gap_px_to_draw"]
    face, gap = np.array([r["face_height_px"] for r in rows], float), np.array([r["lid_gap_px"] for r in rows], float)
    three = lambda v, digits: {"least": round(float(v.min()), digits), "median": round(float(np.median(v)), digits), "most": round(float(v.max()), digits)}
    at = lambda k: {"face_height_px": three(face * k, 1), "lid_gap_px": three(gap * k, 2), "frames_with_the_gap_under_the_line": int((gap * k < line).sum())}
    return {"frames": len(rows), "the_drawing_line_px": line, "measured_on": "the pose model's points on the original's frames, in the subject's tracked box",
            "at_the_canvas": at(1.0), "in_the_reference": {"enlarged": round(float(scale), 3), **at(float(scale))},
            "cannot_see": "whether the points are on the face: the model places and scores lid points on the back of a head and on a face a few pixels tall"}


def reference_block(form: str, why: str, by: str, now: str, record: Path, points: Path, figures: dict) -> dict:
    return {"form": form, "why": why, "said_by": by, "made": now, "record": str(record), "points": str(points), "figures": figures}


# ---------------------------------------------------------------- the steps

def step_preview(a, world) -> None:
    shots, plan = _load(a.shots, a.plan)
    shot = shot_of(shots, a.shot)
    video, source, earlier = working_copy(shots["copy"], world)
    cast = None
    if a.cast:
        import cast_file
        cast = cast_file.read(a.cast)
    job = preview_job(shots, plan, shot, video, str(source), a.by, cast, load_chain.utc_now())
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "THE SHOTS RECORD NAMES AN EARLIER COPY")
    out = Path(a.folder) / f"{job['job']['name']}.json"
    if out.exists():
        raise Refused(f"{out} exists: a preview made again takes a shot's record changed first, and its job file moved aside by a person")
    print(f"{job['job']['name']}: source frames {shot['first']}-{shot['last']}; " + "; ".join(
        f"{s['label']} " + ("seeded from the cast" if "seeds" in s else f"picked on {s['pick_at']}") for s in job["job"]["subjects"]))
    print(job["job"]["note"])
    if world.dry:
        return
    Path(a.folder).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(job, indent=1))
    print("wrote", out)
    build = [world.python, "bench/masked_job.py", "build", "preview", out, "--date", a.day]
    queue = [world.python, "bench/masked_job.py", "queue", world.data / f"{a.day}_{job['job']['name']}_job", "preview", "--host", world.host, "--wait"]
    if not a.queue:
        print("NOT QUEUED (no --queue). To run it, on an empty queue:\n  " + " ".join(map(str, build)) + "\n  " + " ".join(map(str, queue)))
        return
    load_chain.empty_queue(world, "a preview is queued on an empty queue")
    ctx = {"job_folder": world.data / f"{a.day}_{job['job']['name']}_job", "name": job["job"]["name"]}
    world.run(ctx, "build preview", build)
    world.run(ctx, "queue preview", queue)
    print("THE TILES ARE TO BE LOOKED AT before a load is built on this preview: a pick by the largest person is a guess.")


def step_job(a, world) -> None:
    shots, plan = _load(a.shots, a.plan)
    shot = shot_of(shots, a.shot)
    pfolder = Path(a.preview)
    for name in ("job.json", "preview.queued.json"):
        if not (pfolder / name).is_file():
            raise Refused(f"the preview's job folder {pfolder} holds no {name}")
    if json.loads((pfolder / "preview.queued.json").read_text()).get("status") != "success":
        raise Refused(f"the preview in {pfolder} did not end in success")
    preview = json.loads((pfolder / "job.json").read_text())["job"]
    video, source, earlier = working_copy(Path(preview["source"]).name, world)
    region = plan["subjects"][a.subject]["region"]
    if region not in FORM:
        raise Refused(f"the plan's region for {a.subject!r} is {region!r}: this tool names and makes part loads ({', '.join(FORM)})")
    first = shot["first"] if a.first is None else a.first
    folder = Path(a.folder)
    # a name is taken by a job file of any day in this folder, and by a capture or a job folder of any day
    taken = lambda n: (folder / f"{n}.json").exists() or any(world.data.glob(f"????-??-??_{n}")) or any(world.data.glob(f"????-??-??_{n}_job"))
    name = free_name(f"{a.subject}_{FORM[region]}_{first:04d}", taken, a.remake)
    rel = str(pfolder.resolve().relative_to(REPO.resolve())) if pfolder.resolve().is_relative_to(REPO.resolve()) else str(pfolder)
    job = load_job(shots, plan, shot, a.subject, preview, rel, video, world.output, a.by, a.day, name, a.first, a.frames, a.why or "", load_chain.utc_now(),
                   source=str(source))
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "ITS PREVIEW WAS MADE ON AN EARLIER COPY")
    out = folder / f"{name}.json"
    print(f"{name}: {a.subject}, {region}, source frames {job['loads'][0]['first']}-{job['loads'][0]['first'] + job['loads'][0]['frames'] - 1}; "
          f"its reference will be {job['loads'][0]['motion_file']}")
    if not world.dry:
        folder.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(job, indent=1))
        print("wrote", out)
    print(f"NEXT: bench/shot_to_load.py reference {out} --by {a.by}\nTHEN IT STOPS FOR THE TEXT: write {folder / (name + '.txt')} and say who wrote it in texts.{name}.by")


def step_reference(a, world, figures=figures_of) -> None:
    ctx = load_chain.read_job(Path(a.job_file), need_text=False)
    job, load, subject = ctx["job"], ctx["load"], ctx["subject"]
    if a.plain is not None and not a.plain.strip():
        raise Refused("--plain takes the reason in words: the plain cut is taken on a person's word, and the word is kept")
    first, last = load["first"], load["first"] + load["frames"] - 1
    drawn = Path(load["motion_file"])
    folder = drawn.parent
    points = folder / f"{subject}_{first:04d}_{last:04d}_points.npz"
    form, why = ("plain", a.plain.strip()) if a.plain is not None else ("drawn", "the form for a face, head or hair replacement (AGENTS.md, Settled)")
    if a.existing:
        record = drawn.with_suffix(".json")
        if not record.is_file() or not points.is_file():
            raise Refused(f"--existing: {record} and {points.name} beside it are not both there; without --existing the reference is made")
        made = json.loads(record.read_text())
        if made.get("frames") != [first, last] or made.get("passed") is not True:
            raise Refused(f"--existing: {record} is of frames {made.get('frames')} with passed={made.get('passed')!r}, and the load is {first}-{last}")
        written(a, ctx, made, record, points, drawn, form, why, figures, taken=True)
        return
    if folder.exists():
        raise Refused(f"{folder} exists: a reference for {subject} on {first}-{last} was made there, or a run stopped part-way. A load made again on the "
                      "same frames reads it (--existing); a load cut by a look is a new job, whose reference is made on its own frames; a folder a run "
                      "left half-made is read and moved aside by a person")
    store = a.model_store or os.environ.get(load_chain.MODEL_STORE)
    if not store:
        raise Refused(f"where the pose model's store is on this machine is not said: set {load_chain.MODEL_STORE} or give --model-store")
    files = load_chain.preview_files(ctx, world.output)[subject]
    mesh = next((world.output / f for f in ctx["preview"]["files"] if f"_{subject}_mesh_" in Path(f).name and f.endswith(".mp4")), None)
    if mesh is None:
        raise Refused(f"the preview {ctx['preview']['name']} saved no body model video for {subject!r}")
    load_chain.empty_queue(world, "no pose pass beside a render")
    canvas = f"{job['canvas'][0]}x{job['canvas'][1]}"
    if not world.dry:
        folder.mkdir(parents=True)
    world.run(ctx, "pose pass", [world.python, "bench/pose_along_shot.py", "run", "--video", job["source"], "--first", first, "--count", load["frames"], "--canvas", canvas,
                                 "--box-mask", files["track"], "--box-mask-first", ctx["preview"]["first"], "--model-folder", Path(store) / world.pose_model,
                                 "--out", points, "--by", a.by])
    world.run(ctx, "head reference", [world.python, "bench/head_reference.py", "--mesh", mesh, "--mesh-first", ctx["preview"]["first"], "--points", points,
                                      "--source", job["source"], "--first", first, "--count", load["frames"], "--out", drawn, "--by", a.by])
    if world.dry:
        return
    record = drawn.with_suffix(".json")
    written(a, ctx, json.loads(record.read_text()), record, points, drawn, form, why, figures)


def written(a, ctx: dict, made: dict, record: Path, points: Path, drawn: Path, form: str, why: str, figures, taken: bool = False) -> None:
    """Write a reference into its load's job file, with the figures, and print them. `taken`: the reference was there."""
    load = ctx["load"]
    got = figures(points, float(made["scale"][0]))
    d = json.loads(Path(a.job_file).read_text())
    d["job"]["reference"] = reference_block(form, why, a.by, made["made_at"] if taken else load_chain.utc_now(), record, points, got)
    if taken:
        d["job"]["reference"]["taken_as_it_was"] = f"made by {made.get('made_by')}; read into this job by {a.by} at {load_chain.utc_now()}, nothing run"
    # the plain cut's name is the reference tool's own, read from its record
    d["loads"][0]["motion_file"] = str(drawn.parent / made["files"]["plain_cut"] if form == "plain" else drawn)
    Path(a.job_file).write_text(json.dumps(d, indent=1))
    c, r = got["at_the_canvas"], got["in_the_reference"]
    print(f"\nTHE FIGURES, on {got['frames']} frames ({got['measured_on']}); the drawing line is {got['the_drawing_line_px']} px. Nothing here decides the form.")
    print(f"  at the canvas:    face {c['face_height_px']} px tall; lash lines {c['lid_gap_px']} px apart; under the line on {c['frames_with_the_gap_under_the_line']} frames")
    print(f"  in the reference (enlarged {r['enlarged']}): face {r['face_height_px']}; lash lines {r['lid_gap_px']}; under the line on {r['frames_with_the_gap_under_the_line']} frames")
    print(f"  cannot see: {got['cannot_see']}")
    print(f"THE FORM: {form}" + (f", on {a.by}'s word: {why}" if form == "plain" else " (the default; the plain cut is taken only with --plain \"why\")")
          + f". The job file reads {d['loads'][0]['motion_file']}. Open the sheet beside it.")
    print(f"IT STOPS HERE FOR THE TEXT: {ctx['text']} is a person's, and texts.{load['text']}.by says who wrote it. Then: bench/load_chain.py first {a.job_file} --by ...")


STEP = {"preview": step_preview, "job": step_job, "reference": step_reference}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    today = datetime.date.today().isoformat()
    for name in ("preview", "job"):
        s = sub.add_parser(name)
        s.add_argument("--shots", required=True)
        s.add_argument("--plan", required=True)
        s.add_argument("--shot", type=int, required=True, metavar="FIRST", help="the shot, by the source frame it begins on")
        s.add_argument("--folder", required=True, help="where the job file is written")
        s.add_argument("--day", default=today, help="the day in the names of the job's folders; today when left out")
    sub.choices["preview"].add_argument("--cast", help="the cast file: a subject with an answer inside the shot is seeded from it")
    sub.choices["preview"].add_argument("--queue", action="store_true", help="build and queue the preview, on an empty queue")
    j = sub.choices["job"]
    j.add_argument("--subject", required=True)
    j.add_argument("--preview", required=True, help="the job folder of the shot's preview")
    j.add_argument("--first", type=int, help="the load's first frame, when a look cut the shot's")
    j.add_argument("--frames", type=int, help="the load's frame count, when a look cut the shot's")
    j.add_argument("--why", help="what a look cut and why; into the job's left_out")
    j.add_argument("--remake", action="store_true", help="the next free letter after the name, for a load made again")
    r = sub.add_parser("reference")
    r.add_argument("job_file")
    r.add_argument("--plain", metavar="WHY", help="take the plain cut, on a person's word; the reason is kept in the job file")
    r.add_argument("--existing", action="store_true", help="a load made again on the same frames: read the reference that is there, run nothing")
    r.add_argument("--model-store", default="", help=f"the folder holding the pose model's folder; over the environment's {load_chain.MODEL_STORE}")
    for s in sub.choices.values():
        s.add_argument("--by", required=True, help="the session")
        s.add_argument("--host", default=load_chain.HOST)
        s.add_argument("--dry", action="store_true", help="print and write nothing, run nothing")
    a = p.parse_args()
    import clip_folder
    try:
        STEP[a.step](a, load_chain.World(a.host, a.dry))
    except (Refused, clip_folder.Refused) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
