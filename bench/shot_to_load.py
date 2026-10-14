#!/usr/bin/env python3
"""From a shot to a load's job file: the preview of the shot, the load's job, its head reference. Then it stops for the text.

    <python> bench/shot_to_load.py preview --shots SHOTS.json --plan PLAN.json --shot FIRST --folder INPUTS --by SESSION [--cast CAST.jsonl] [--queue] [--remake]
        [--seed-cuts-of EARLIER_PREVIEW_JOB.json=ITS_SHOTS_TABLE.json] [--ask "SUBJECT:why the owner is asked about them"]
        (somebody looks at the trackers' tiles; a wrong pick is a person's line in the shots record or a box in the cast
        file, and the preview is made again with --remake, under the next letter)
    <python> bench/shot_to_load.py job --shots SHOTS.json --plan PLAN.json --shot FIRST --subject LABEL --preview PREVIEW_JOB_FOLDER \\
        --folder INPUTS --by SESSION [--first F --frames N --why "what a look cut" | --why-before ".." --why-after ".."] [--remake]
        [--zoomed "why this load is rendered zoomed in"] [--least PX --least-why "why the floor is set there"]
        [--parts NAME ... --parts-by "who chose these parts for this load, and on what"]
    <python> bench/shot_to_load.py reference JOB.json --by SESSION [--plain "why"] [--model-store FOLDER] [--existing]
        (a person writes the load's text, a file beside the job file)
    <python> bench/shot_to_load.py text JOB.json --by SESSION --writer "who wrote it, and from which frames"
        (then `bench/load_chain.py first JOB.json`)

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
  subject whose plan lacks `still` or `parts` is refused with the keys to add. Optionally `pieces` (with
  `pieces_by`: who chose it, on what): how the capture reads the subject's part, copied by `job` onto the
  subject in the load's job file, where the chain passes it to the capture. This tool sets it for nobody.
  Optionally `keep_outs` (with `keep_outs_by`): which of `held`, `hands` and `clothes` that subject's loads keep
  as the source's (`KEEP_OUTS`). `job` makes each from the preview's own files, names it in the load's `others`
  and writes where its file is into the job file (`job.keep_outs`), from which the chain hands it to the capture.
- The cast file (`--cast`): where it holds an answer inside the shot for a subject, that tracker is seeded from
  it (`bench/cast_file.py::seeds`; a "none of the outlined people" answer leaves the shot unseeded). A seeded
  subject's tracker runs before an unseeded one's, whatever the plan's order: the unseeded one's guess then
  takes only who is left.
- The preview's own job folder (`--preview`, handed in by name): its subjects, its files, the frame they begin on.
- The video's analysis folder (`clip_folder.py`), found by the file a record was made on: a job is written to
  render from the video's PRESENT copy, whatever copy the shots record or the preview names, and a record made
  on a copy with another frame clock is refused (`working_copy`).
- `workflows/h3_config.py`: the canvas, and what goes into a job file's `defaults` (`MASKED_LOAD_DEFAULTS`,
  `MASKED_LOAD_MARGIN` by the plan's region, `SEED`), so the job file stands alone.

**The steps.**

- `preview`: ONE preview a shot, every subject the record says is in it, a pose pass for each. Each tracker picks
  the largest person on its frame, which is a guess: the step stops there for the tiles to be looked at.
  `--queue` builds and queues it on an empty queue; without it the two commands are printed. Before anything
  is written the job is held to the builder's refusals that write nothing (`held_as_the_builder_holds`), and
  `--dry` stops there and says so: a dry run does NOT build the graph or queue, and the first real preview
  found what no dry run could (2026-10-13: the tool's own log folder made the job folder the builder then
  refused; the commands' output is kept beside the job file now). The name is `shot_c_<first>`, and one that a
  job file here or a job folder of any day has is refused. A PREVIEW MADE AGAIN (the picks corrected, or the
  owner's answers on the who page now boxes in the cast file) takes `--remake`: the next free letter after the
  name, `shot_c_<first>b`, as a load made again does. Nothing of the earlier preview is moved, replaced or read;
  the load's `job` step is handed the preview it is built on by its folder, so which one stands is said there.
- `job`: one subject's load on the shot, or on the frames a look left of it (`--first`, `--frames`, with `--why`;
  the frames cut go into `left_out` with those words). A load cut at both ends for two reasons says each:
  `--why-before` for the frames before it and `--why-after` for those after, each over `--why` for its side
  (2026-10-13: a load remade shorter carried one sentence on both entries, true of one). The name is made here: `<subject>_<form>_<first>`, a
  letter after it for a remake (`--remake`), and one that exists is refused. The job file names its preview
  (`job.preview`), its capture, its reference by where it WILL be, and a text file that is not written yet.
  A subject's `pick` is the preview's; it carries no `pick_at`, since a load tracks nobody. TWO CHOICES ABOUT ONE
  LOAD, each a person's, each with its words, neither with a default (the lead's ruling, 2026-10-13: they change
  how a load is rendered, not what the owner ruled is replaced). `--zoomed "why"` renders the load zoomed in on
  what is replaced (the load's `render_region: part`; one box for the load, the node's own), for a part too
  small on the whole frame; the words and who said them are kept in the job file (`job.zoomed`) and its note.
  `--least PX --least-why "why"` sets the capture's floor for the subject's part on this load (`least` on the
  subject, with `least_by`), which a zoomed load usually needs: the floor is counted in whole-frame pixels, where
  a part rendered zoomed in is smaller than it is drawn. Before the file is
  written, and in a `--dry` run as well, the load is held to the builder's refusals that need neither its
  capture nor its text (`load_held_as_the_builder_holds`). A plan that asks a `hair_band` is refused here: the
  band is cut out of a keep-out no preview of this tool tracks, and until 2026-10-13 the load was written and
  its own builder refused it at the plan.
- `reference`: on the LOAD's own frames, so the box is round the head on those frames: the pose pass in the
  subject's tracked box, then `bench/head_reference.py`, which writes the drawn form and the plain cut both. It
  goes to `Video/<video>/references/<subject>_<first>_<last>/`: a reference belongs to a shot, a subject and a
  span, never to a load's name. THE DRAWN FORM IS THE DEFAULT and nothing here decides otherwise: the face's
  height and the lash-line gap are printed at the canvas and as they come out in the reference at its scale,
  with the frames under the drawing line, and written into the job file. The plain cut is taken on a person's
  word only, `--plain "why"`, and the form, the why and who said it go into the job file. A load made again on
  the same frames reads the reference that is there: `--existing` runs nothing, holds the record beside it to
  the job's subject and frames, and writes the same block from it. A reference whose record does not say its face
  was held to the subject's track is refused there (2026-10-14: the points of the lead's 3167 were read in her
  track's box alone, drew another woman's face on eleven frames and passed; `bench/pose_along_shot.py`).

**It stops for the text.** The job file's text entry names a file and no writer. `bench/load_chain.py` refuses the
load until the file is there and `texts.<label>.by` says who wrote it.

- `text`: says who wrote it. The text file must be there with words in it; `--writer` is a person's words (who,
  and from which frames), and they go into `texts.<label>.by` with when they were entered and by which session.
  Until 2026-10-13 that key was set by opening the job file and editing it. Words that were there are printed
  as replaced. Refused once the load's plan is built: the job file is then the one the plan was built from, and
  the chain's render holds it to that copy byte for byte.

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
#: Where a load's own part goes when it is a subset of its subject's parts: by the video, the preview it was cut from,
#: the subject and the parts, never by a load's name, so two loads of one preview with the same parts share one file.
PARTS_AT = "Video/{video}/parts/{preview}"
#: A load's name where it carries a subset of the plan's parts: a letter a part, in the plan's order, then `p`.
#: Reasoned: `hfp` and `fp` are the two forms' names already, and a load of the hair alone is neither.
PART_LETTER = {"hair": "h", "face_and_neck": "f", "upper_clothing": "u", "lower_clothing": "l", "hands": "n", "mouth": "m"}
#: Where a subject's keep-outs go: by the video, the preview they were cut from and the subject, never by a load's name.
KEEP_OUTS_AT = "Video/{video}/keep_outs/{preview}"
#: THE KEEP-OUTS A PLAN MAY NAME FOR A SUBJECT: things on that person a part load must leave as the source's. Each is
#: a mask video on the preview's own frames, a subject of the load's capture and one of the load's `others`. `held` is
#: the preview's own held-things file as saved; the other two are classes of the preview's class map of that subject,
#: not clipped to the track and not cleaned to a largest piece.
#: INHERITED from the session cuts of 2026-10-12 (the hands and arms after a hand at her face was drawn again; the
#: clothing after a blouse strap in the margin was) and MEASURED 2026-10-13: cut again from that day's class map of a
#: shot of 457 frames, each is that day's own mask on every frame with no pixel differing; with `Torso` added to the
#: clothing it is equal on 29 of the 457, so the torso was left out on purpose. Every load of the second subject
#: through this tool carried none of them until that day, and the first render drew her apron as the still's top.
KEEP_OUTS = {"held": None,
             "hands": ("Left_Hand", "Right_Hand", "Left_Lower_Arm", "Left_Upper_Arm", "Right_Lower_Arm", "Right_Upper_Arm"),
             "clothes": ("Apparel", "Upper_Clothing")}
#: What a plan may say of how the capture reads a subject's part (`bench/capture_masked_run.py`, `pieces=` on a `--mask`).
PIECES = ("largest",)
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


def held_as_the_builder_holds(job: dict, day: str, world) -> tuple[Path, Path]:
    """A preview's job held to every refusal of the builder that writes nothing, before anything is written or
    queued: the job block, the video's analysis folder and the file's identity, and the two folders the build
    makes, which must not exist (`bench/masked_job.py::build`: "A run never reuses a name"). Returns those two
    folders. What it does not do is build the graph: that is the builder's, and only a real run shows it."""
    import masked_job
    import masked_job_graph
    try:
        masked_job_graph.check_job(job)
        masked_job.analysed(job, world.data)
    except (ValueError, masked_job.Refused) as e:
        raise Refused(str(e)) from None
    folder, saves_to = masked_job.job_folder(job, day, world.data), masked_job.output_folder(job, world.output)
    for path, what in ((folder, "job folder"), (saves_to, "output folder")):
        if path.exists():
            raise Refused(f"the {what} {path} exists. A run never reuses a name: a preview made again takes --remake, which gives it the next free letter")
    return folder, saves_to


def load_held_as_the_builder_holds(given: dict, world) -> None:
    """A load's job file held, before it is written, to every refusal of the builder that needs neither the load's
    capture nor its text: the job block, the video's analysis folder and the file's identity, each load's keys and
    settings (`masked_job_graph.check_load`, with the planner's windows), the shortest load, and `left_out`. A dry
    run is held the same. What it does not do: read a capture, a text or a mask, which are not there yet; the
    chain's `plan --dry` holds the load to those."""
    import masked_job
    import masked_job_graph
    try:
        job = given["job"]
        masked_job_graph.check_job(job)
        masked_job.analysed(job, world.data)
        _, least = masked_job.shortest_load()
        for one in masked_job.loads_of(given):
            if one["frames"] < least:
                raise ValueError(f"load {one['name']!r} is {one['frames']} frames and the song node takes no load under {least}")
            masked_job_graph.check_load(job, one, masked_job.chain_lengths())
        masked_job.left_out(job)
    except (ValueError, masked_job.Refused) as e:
        raise Refused(f"the builder would refuse this load, so nothing is written: {e}") from None


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


def order_of(shot: dict, plan: dict, present: list[str], seeded=()) -> tuple[list[str], str]:
    """The order a shot's trackers run in, and where it is from: the shot's own `order`, a person's; else the
    seeded subjects before the others, each group in the plan's order.

    A seed is a person's word for who the subject is; a pick by the largest person is a guess that takes whoever an
    earlier tracker has not. On the first real preview (2026-10-13, shot 604) the plan's order ran the unseeded
    subject first, and its guess took the person the other subject was then seeded on."""
    if shot.get("order"):
        unknown = [x for x in shot["order"] if x not in present]
        if unknown or sorted(shot["order"]) != sorted(present):
            raise Refused(f"the shot's `order` {shot['order']} is not the subjects the record says are in it ({present})")
        return list(shot["order"]), "the shot's own `order` in the shots record, a person's"
    by_plan = [k for k in plan["subjects"] if k in present]
    first, rest = [k for k in by_plan if k in seeded], [k for k in by_plan if k not in seeded]
    if first and rest:
        return first + rest, ("the seeded before the others, since a seed is a person's word and a pick by the largest person takes whoever an earlier "
                              "tracker has not; within each, the plan's subject order, a starting value")
    return by_plan, "the plan's subject order, a starting value: nothing says who is larger on this shot"


def pick_of(shot: dict, label: str) -> tuple[int, str]:
    """The frame a subject's tracker picks on, and where it is from: a frame the record names as a field, else the middle."""
    who = shot["who"][label]
    for key, words in (("frames_on", "a frame the record says they were marked on"), ("frames_boxed", "a frame a box stands on in the cast file")):
        inside = [f for f in who.get(key) or [] if shot["first"] <= f <= shot["last"]]
        if inside:
            return inside[0], f"{words} ({key})"
    return (shot["first"] + shot["last"]) // 2, "THE SHOT'S MIDDLE: the record names no frame for them as a field. A fallback, not a frame anybody marked"


def cuts_of_table(pair: str, shot: dict) -> dict:
    """The tracker's own cuts of a shot, read from a table an earlier preview of the SAME frames saved: `JOB.json=TABLE.json`.

    The tracker takes one seed a cut OF ITS OWN, and a preview's seeds were chosen one a RECORD shot. On the first
    shot the tracker cut into many (2026-10-13, 6105-6394: twenty-five cuts, six boxes of the owner's, each in
    another long cut) one box was passed, the latest in the cast file and a frame inside a fade to black; every
    other cut was matched against that look and the subject was held on a small part of the shot the unseeded
    preview had held somebody on. With the cuts handed in, each cut that holds an answer gets its own seed. A named
    file, handed in, and the job says it was read; the job file beside it is how the table's frames are known."""
    job_file, _, table_file = pair.partition("=")
    if not job_file.strip() or not table_file.strip():
        raise Refused(f"--seed-cuts-of {pair!r} is not `JOB.json=TABLE.json`: the earlier preview's job file, and the shots table it saved for a subject")
    for f in (job_file, table_file):
        if not Path(f).is_file():
            raise Refused(f"--seed-cuts-of: {f} is not there")
    job = json.loads(Path(job_file).read_text())
    job = job.get("job", job)
    frames = shot["last"] - shot["first"] + 1
    if (job.get("first"), job.get("frames")) != (shot["first"], frames):
        raise Refused(f"--seed-cuts-of: {job_file} is a preview of {job.get('first')}-{(job.get('first') or 0) + (job.get('frames') or 0) - 1}, not of the shot "
                      f"{shot['first']}-{shot['last']}: the tracker's cuts are read from a preview of the same frames")
    table = json.loads(Path(table_file).read_text())
    cuts = [[int(c["first_frame"]), int(c["last_frame"]) + 1] for c in table.get("shots") or []]
    if table.get("frames") != frames or not cuts or cuts[0][0] != 0 or cuts[-1][1] != frames or any(a[1] != b[0] for a, b in zip(cuts, cuts[1:])):
        raise Refused(f"--seed-cuts-of: {table_file} holds {table.get('frames')} frames in {len(cuts)} cut(s) that do not lie end to end over the shot's {frames}: "
                      "not a table of this shot's frames")
    return {"read_from": table_file, "the_preview": job.get("name"), "its_job_file": job_file, "cuts": cuts}


def seed_of(cast_lines: list[dict] | None, label: str, shot: dict, canvas, cuts=None) -> dict | None:
    """The tracker's seeds for a subject on a shot from the cast file, or None when it holds no box there. `cuts` is
    the tracker's own cuts of the shot, (first frame, one past the last) in the shot's frames, when they are known
    (`cuts_of_table`); without them the shot is taken as one cut and gets one seed."""
    if not cast_lines:
        return None
    import cast_file
    try:
        got = cast_file.seeds(cast_lines, label, shot["first"], canvas, [tuple(c) for c in cuts] if cuts else [(0, shot["last"] - shot["first"] + 1)], cast_file.LEAVE)
    except ValueError as e:
        raise Refused(str(e)) from None
    return got if got["text"] else None


def preview_job(shots: dict, plan: dict, shot: dict, video: str, source: str, by: str, cast_lines=None, now: str = "", name: str = "",
                ask: dict | None = None, seed_cuts: dict | None = None) -> dict:
    """The job file of one shot's preview: every subject the record says is in it, a pose pass for each. `name` is
    the preview's, `shot_c_<first>` when left out; a preview made again is handed its lettered name.

    `ask` is {subject: why}: a subject the record is UNSURE of on this shot, tracked all the same because the preview
    is MADE TO ASK THE OWNER who the people on it are (2026-10-13: a shot cut out of another had no answer of the
    owner's on any frame, all three subjects unsure, and the preview its who-page picture is cut from was refused
    for having nobody to track). An unsure subject is planned by nothing, and it is asked about: the tracker picks
    as it does with no seed, the job says the preview is made to ask, and no load is built on it
    (`load_job` refuses a subject the record does not say is in the shot). A subject who is `yes` needs no asking
    and one who is `no` is the owner's answer: both are refused here."""
    import h3_config
    ask = ask or {}
    for label, why in ask.items():
        said_ = (shot["who"].get(label) or {}).get("present")
        if said_ != "unsure" or not str(why).strip():
            raise Refused(f"--ask {label}: a preview is made to ask about a subject the shots record is `unsure` of on {shot['first']}-{shot['last']}, with the reason; "
                          f"the record says {said_!r} of {label!r} there")
    present = [k for k in shots["subjects"] if shot["who"][k]["present"] == "yes" or k in ask]
    if not present:
        raise Refused(f"the shots record says nobody of ours is in {shot['first']}-{shot['last']}: there is nothing to preview"
                      + (". A subject the record is `unsure` of there may be tracked to ASK the owner who is on it: --ask \"SUBJECT:why\""
                         if any(w["present"] == "unsure" for w in shot["who"].values()) else ""))
    canvas = list(h3_config.MASKED_JOB_CANVAS)
    seeds = {label: seed_of(cast_lines, label, shot, canvas, (seed_cuts or {}).get("cuts")) for label in present}
    order, order_from = order_of(shot, plan, present, [k for k, v in seeds.items() if v is not None])
    subjects, said = [], []
    for label in order:
        p = planned(plan, label)
        one = {"label": label, "pick": PICK, "parts": list(p["parts"]), "still": p["still"]}
        seed = seeds[label]
        if seed is not None:
            one["seeds"] = seed["text"]
            # the cast line's own words, with no hole where one of its fields is empty: a line entered from a page answer
            # with no time on it reads "on the page at  (" in the file, and that is said here as the file's
            why = "; ".join(r["why"] for r in seed["shots"] if r["seed"])
            holed = " ".join(why.split()) != why
            used = [r["from_frame"] for r in seed["shots"] if r["seed"]]
            unused = sorted(u["frame"] for u in seed["unused"] if u["answer"] == "here")
            said.append(f"{label}: SEEDED from the cast file ({' '.join(why.split())}"
                        + ("; a field of that line is EMPTY in the cast file, as it reads there" if holed else "")
                        + f"): THE SEED DECIDES who is tracked; `pick: {PICK}` stands in the job file only because the builder asks one of every "
                          f"subject, and decides nothing here. SEEDED ON SOURCE FRAME(S) {', '.join(map(str, used))}"
                        + (f", one in each of {len(used)} of the tracker's {len(seed['shots'])} cuts" if seed_cuts else "")
                        + (f". THE OWNER'S BOX(ES) ON {', '.join(map(str, unused))} ARE NOT USED: a cut takes one seed, and of several in a cut the one latest "
                           "in the cast file is taken, which is the order they were clicked in and no judgement of the picture"
                           + ("" if seed_cuts else ". If the tracker cuts this shot, each of its cuts can take its own: --seed-cuts-of, with the table this preview saves")
                           if unused else ""))
        else:
            one["pick_at"], where = pick_of(shot, label)
            said.append(f"{label}: the {PICK} person on frame {one['pick_at']}, {where}"
                        + (f". TRACKED TO ASK, NOT BECAUSE THE RECORD SAYS THEY ARE HERE: the record is `unsure` of {label} on this shot ({' '.join(str(ask[label]).split())})"
                           if label in ask else ""))
        subjects.append(one)
    frames = shot["last"] - shot["first"] + 1
    base = preview_name(shot)
    name = name or base
    note = (f"Written by bench/shot_to_load.py for {by}, {now}. THE PREVIEW OF ONE SHOT, source frames {shot['first']} to {shot['last']} ({frames} frames): "
            f"every subject the shots record says is in it, a pose pass for each, nothing sampled. THE TRACKERS RUN IN THIS ORDER: {', '.join(order)} "
            f"({order_from}). " + (f"THE TRACKER'S CUTS OF THIS SHOT, {len(seed_cuts['cuts'])} of them, WERE READ FROM {seed_cuts['read_from']}, the table the preview "
                                    f"{seed_cuts['the_preview']} saved of these frames, and each seed was chosen in one of them. " if seed_cuts else "")
            + f"THE PICKS: {'; '.join(said)}. A pick by the largest person is a guess: the tiles are looked at before anything is built on this "
            f"preview, and a wrong pick is corrected in the shots record (the shot's `order`, or a frame for the subject) and the preview made again."
            + (f" THIS IS A PREVIEW MADE AGAIN: {name}, after {base} or a letter before it, which stand as they were made and are not read here." if name != base else ""))
    return {"job": {"name": name, "note": note, "source": source, "first": shot["first"], "frames": frames, "canvas": canvas,
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects, "left_out": [], "texts": {}}, "defaults": {}, "loads": []}


def keep_outs_of(p: dict, subject: str, preview: dict, people: list[str], video: str, output: Path, preview_files) -> tuple[dict | None, str]:
    """A load's keep-outs as the job file holds them, and the note's sentence: nothing when the plan names none.

    Each is {"name", "track": the mask video, on the preview's frames} and for a cut one its `classes`, the class
    map it is cut from and where the cut is written. `preview_files` is what the preview's record says it saved,
    under the output folder. Refused: a name that is not a keep-out, one that is a person of the preview, no
    `keep_outs_by`, and a preview that saved no class map or no held-things file of the subject."""
    names = p.get("keep_outs")
    if names is None:
        return None, ""
    by = " ".join(str(p.get("keep_outs_by") or "").split())
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(k not in KEEP_OUTS for k in names) or not by:
        raise Refused(f"the plan's `keep_outs` for {subject!r} is {names!r} with `keep_outs_by` {p.get('keep_outs_by')!r}: a list of {tuple(KEEP_OUTS)}, "
                      "each once, and `keep_outs_by` says who chose them and on what")
    clash = [k for k in names if k in people]
    if clash:
        raise Refused(f"the keep-out name(s) {clash} are subjects of the preview {preview['name']}: a keep-out and a person cannot share a label in one capture")
    if preview_files is None:
        raise Refused(f"the plan names keep-outs for {subject!r} and the preview's saved files were not handed in: they are cut from the preview's own class map")

    def saved(kind: str) -> Path:
        found = [f for f in preview_files if f"_{subject}_{kind}_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {preview['name']} saved {len(found)} `{kind}` video(s) of {subject!r} where one is meant, and the plan names keep-outs "
                          f"for them ({', '.join(names)}): they are made from that preview's own files, so the preview is made again with them, or the plan's "
                          "`keep_outs` is taken out")
        return output / found[0]
    folder = output / KEEP_OUTS_AT.format(video=video, preview=preview["name"])
    made = []
    for k in names:
        if KEEP_OUTS[k] is None:
            made.append({"name": k, "track": str(saved("held_things")), "what": "the preview's own held-things file, as saved"})
        else:
            made.append({"name": k, "track": str(folder / f"{preview['name']}_{subject}_{k}.mkv"), "classes": list(KEEP_OUTS[k]),
                         "cut_from": str(saved("classes")), "what": "those classes of the preview's class map of the subject, not clipped to the track"})
    said = (f" KEPT AS THE SOURCE'S ON THIS LOAD, from the plan ({by}): " + "; ".join(
        f"`{k['name']}` ({'the preview held-things file' if 'classes' not in k else ' + '.join(k['classes']) + ' of the class map'})" for k in made)
        + ". Each is one of the load's `others` and a subject of its capture, on the preview's frames.")
    return {"of": subject, "by": f"the plan ({by})", "preview": preview["name"], "first": preview["first"], "made": made}, said


def part_classes() -> dict:
    """The classes of each part's name, the part model's own table (`sapiens2_parts.PARTS`): one copy, read and not written here."""
    import capture_masked_run as cap
    return cap._pack("sapiens2_parts").PARTS


def part_of(p: dict, subject: str, parts, parts_by, preview: dict, video: str, output: Path, preview_files, classes_of=part_classes) -> tuple[dict | None, str]:
    """A load's own part where it is a SUBSET of its subject's plan parts, as the job file holds it, and the note's
    sentence: nothing when the load names none.

    The owner's answer of 2026-10-13 on the second subject: where her head is bowed and no face shows, her hair
    stays the new hair. So one shot is face and hair on some frames and hair alone on others, and a load's part was
    the preview's saved part, every part the plan names. Measured that day on the preview of that shot: her
    face-and-neck classes were on every frame of the stretch with no face, about as many pixels as her hair, so the
    saved part would not do; and the plan's parts cut from the class map were the saved part pixel for pixel, so a
    cut of some of them is a part of it. The part is cut from the preview's class map of the subject exactly as a
    keep-out is, and the chain hands the capture that file as the load's part. Refused: a part the plan does not
    name for the subject, a part named twice, every part of the plan (that is the load as it is), no `--parts-by`,
    and a preview that saved no class map or no part of the subject."""
    if not parts and not str(parts_by or "").strip():
        return None, ""
    by = " ".join(str(parts_by or "").split())
    names = list(parts or [])
    if not names or len(set(names)) != len(names) or any(k not in p["parts"] for k in names) or not by:
        raise Refused(f"--parts {names!r} with --parts-by {parts_by!r}: the parts are some of the plan's for {subject!r} ({', '.join(p['parts'])}), each once, "
                      "and --parts-by says who chose them for this load and on what")
    if set(names) == set(p["parts"]):
        raise Refused(f"--parts names every part the plan has for {subject!r} ({', '.join(p['parts'])}): that is the load as it is; leave --parts out")
    names = [k for k in p["parts"] if k in names]                 # the plan's order
    if preview_files is None:
        raise Refused(f"--parts is given for {subject!r} and the preview's saved files were not handed in: the part is cut from the preview's own class map")

    def saved(kind: str) -> Path:
        found = [f for f in preview_files if f"_{subject}_{kind}_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {preview['name']} saved {len(found)} `{kind}` video(s) of {subject!r} where one is meant: a load's own part is cut from "
                          "that preview's class map and held to its saved part")
        return output / found[0]
    table = classes_of()
    classes = [c for k in names for c in table[k]]
    file = output / PARTS_AT.format(video=video, preview=preview["name"]) / f"{preview['name']}_{subject}_part_{'_'.join(names)}.mkv"
    block = {"of": subject, "by": by, "preview": preview["name"], "first": preview["first"], "parts": names, "the_plans_parts": list(p["parts"]),
             "made": [{"name": "part", "track": str(file), "classes": classes, "cut_from": str(saved("classes")), "inside": str(saved("parts")),
                       "what": "those classes of the preview's class map of the subject; every pixel of it is in the preview's saved part"}]}
    said = (f" THIS LOAD'S PART IS {' + '.join(names).upper()} ALONE, not the plan's {' + '.join(p['parts'])}, on {by}. It is cut from the preview's class map "
            f"({' + '.join(classes)}) as a keep-out is, to {file}, and the chain hands the capture that file as the load's part: what the builder loads is what "
            "the capture read. The load's region, its margin and its reference are the plan's, unchanged.")
    return block, said


def part_inside(cut: Path, whole: Path) -> int:
    """The pixels of a cut part that are NOT in the preview's saved part, over every frame: none, for a part of it."""
    import capture_masked_run as cap
    w, h, n = cap.probe(str(whole))
    a, b = cap.read_mask(str(cut), (w, h), 0, 0, n)[0], cap.read_mask(str(whole), (w, h), 0, 0, n)[0]
    return int((a & ~b).sum())


def cut_keep_out(class_map: Path, classes: list[str], out: Path) -> dict:
    """Write one keep-out: the pixels of a saved class map that carry any of `classes`, as a mask video on the class
    map's own frames. Figures of what was written."""
    import numpy as np
    import capture_masked_run as cap
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    w, h, n = cap.probe(str(class_map))
    mask = np.isin(cap.read_classes(str(class_map), (w, h), 0, 0, n), [names.index(c) for c in classes])
    out.parent.mkdir(parents=True, exist_ok=True)
    cap.write_mask_video(out, mask)
    px = mask.reshape(n, -1).sum(1)
    return {"frames": int(n), "size": [int(w), int(h)], "frames_with_the_mask": int((px > 0).sum()), "px_most": int(px.max()) if n else 0}


def make_part(part: dict | None, world, cut=cut_keep_out, hashed=None, inside=part_inside) -> list[str]:
    """Make the part a load's job file names as its own, a subset of its subject's parts, and say what was done.

    As a keep-out is made: cut from the preview's class map; a file that is there is not trusted because it is
    there, a cut made again beside it must decode equal. AND HELD TO THE PREVIEW'S SAVED PART: a pixel of the cut
    that is not in the saved part is no part of it, and the load is refused with the count."""
    hashed = hashed or load_chain.frame_hash
    lines = []
    for k in (part or {}).get("made", []):
        file = Path(k["track"])
        if world.dry:
            lines.append(f"the load's own part ({' + '.join(part['parts'])}): WOULD BE CUT from {k['cut_from']} ({' + '.join(k['classes'])}) to {file}"
                         + (" (a file is there: held to a fresh cut)" if file.exists() else "") + f", and held to the preview's saved part {k['inside']}")
            continue
        if file.exists():
            again = file.with_name(file.stem + ".cut_again.mkv")
            cut(Path(k["cut_from"]), k["classes"], again)
            same = hashed(str(again)) == hashed(str(file))
            again.unlink()
            if not same:
                raise Refused(f"{file} is there and is NOT what its classes cut from {k['cut_from']} now: a part of another making. It is read and moved aside by a person")
            lines.append(f"the load's own part ({' + '.join(part['parts'])}): {file} was there, and a cut made again now decodes equal to it on every frame")
            continue
        got = cut(Path(k["cut_from"]), k["classes"], file)
        outside = inside(file, Path(k["inside"]))
        if outside:
            file.unlink()
            raise Refused(f"the cut of {' + '.join(part['parts'])} from {k['cut_from']} has {outside} px that are NOT in the preview's saved part {k['inside']}: it is no "
                          "part of what the preview saved, so the class map and the part are not of one making. Nothing is kept")
        file.with_suffix(".json").write_text(json.dumps({"what": "a load's own part, some of its subject's parts, cut by bench/shot_to_load.py from a preview's class map",
                                                        "of": part["of"], "parts": part["parts"], "the_plans_parts": part["the_plans_parts"], "by": part["by"],
                                                        "classes": k["classes"], "cut_from": k["cut_from"], "held_to": k["inside"], "px_outside_the_saved_part": 0,
                                                        "preview": part["preview"], "first_source_frame": part["first"], "made_at": load_chain.utc_now(), **got}, indent=1) + "\n")
        lines.append(f"the load's own part ({' + '.join(part['parts'])}): cut {' + '.join(k['classes'])} from {k['cut_from']} to {file}: {got['frames']} frames, on the "
                     f"mask on {got['frames_with_the_mask']}, up to {got['px_most']} px; every pixel of it is in the preview's saved part")
    return lines


def make_keep_outs(kept: dict | None, world, cut=cut_keep_out, hashed=None) -> list[str]:
    """Make the keep-outs a load's job file names that are cut from a class map, and say what was done of each.

    One file a preview, a subject and a keep-out, so a second load of the same subject on the same preview finds
    it there. A file that is there is not trusted because it is there: the cut is made again beside it and the two
    are held to each other frame by frame (`load_chain.frame_hash`); equal, the one there stands; not equal, refused.
    The held-things file is the preview's own and is only looked for."""
    hashed = hashed or load_chain.frame_hash
    lines = []
    for k in (kept or {}).get("made", []):
        track = Path(k["track"])
        if "classes" not in k:
            if not track.is_file() and not world.dry:
                raise Refused(f"the keep-out `{k['name']}` is the preview's held-things file {track}, which is not there")
            lines.append(f"keep-out `{k['name']}`: {track} ({k['what']})")
            continue
        if world.dry:
            lines.append(f"keep-out `{k['name']}`: WOULD BE CUT from {k['cut_from']} ({' + '.join(k['classes'])}) to {track}" + (" (a file is there: held to a fresh cut)" if track.exists() else ""))
            continue
        if track.exists():
            again = track.with_name(track.stem + ".cut_again.mkv")
            cut(Path(k["cut_from"]), k["classes"], again)
            same = hashed(str(again)) == hashed(str(track))
            again.unlink()
            if not same:
                raise Refused(f"{track} is there and is NOT what its classes cut from {k['cut_from']} now: a keep-out of another making. It is read and moved aside by a person")
            lines.append(f"keep-out `{k['name']}`: {track} was there, and a cut made again now decodes equal to it on every frame")
            continue
        got = cut(Path(k["cut_from"]), k["classes"], track)
        track.with_suffix(".json").write_text(json.dumps({"what": "a keep-out cut by bench/shot_to_load.py from a preview's class map", "name": k["name"], "of": kept["of"],
                                                         "classes": k["classes"], "cut_from": k["cut_from"], "preview": kept["preview"],
                                                         "first_source_frame": kept["first"], "made_at": load_chain.utc_now(), **got}, indent=1) + "\n")
        lines.append(f"keep-out `{k['name']}`: cut {' + '.join(k['classes'])} from {k['cut_from']} to {track}: {got['frames']} frames, on the mask on "
                     f"{got['frames_with_the_mask']}, up to {got['px_most']} px")
    return lines


def preview_name(shot: dict) -> str:
    return f"shot_c_{shot['first']:04d}"


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
             name: str, first: int | None = None, frames: int | None = None, why: str = "", now: str = "", source: str | None = None,
             why_before: str = "", why_after: str = "", zoomed: str | None = None, least: int | None = None, least_why: str | None = None,
             preview_files: list[str] | None = None, parts: list[str] | None = None, parts_by: str | None = None, classes_of=part_classes) -> dict:
    """The job file of one subject's load on a shot, from the shot's preview. `preview` is the preview's own job
    block; `source` is the file the load renders from, the video's present copy (the preview's own when left out).
    `parts` with `parts_by` makes the load's part some of its subject's plan parts (`part_of`)."""
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
    # each side of a cut with its own words, or the one sentence given for both
    sides = (("before", "--why-before", why_before, (shot["first"], a - 1), a > shot["first"]), ("after", "--why-after", why_after, (b + 1, shot["last"]), b < shot["last"]))
    cut = []
    for side, flag, own, frames_cut, is_cut in sides:
        said = (own or why).strip()
        if own.strip() and not is_cut:
            raise Refused(f"{flag} is given and no frame {side} the load is cut: the load is {a}-{b} of the shot {shot['first']}-{shot['last']}")
        if is_cut and not said:
            raise Refused(f"the load is {a}-{b} of the shot {shot['first']}-{shot['last']}: say what a look cut {side} it and why (--why for both ends, or {flag}); "
                          "it goes into the job's `left_out`")
        if is_cut:
            cut.append((*frames_cut, said))
    # one sentence when both ends were cut for one reason, else each end's frames with its own
    cut_said = cut[0][2] if len({said for _x, _y, said in cut}) == 1 else " ".join(f"{x}-{y}: {said}" for x, y, said in cut)
    reference = output / REFERENCES.format(video=video, subject=subject, first=a, last=b) / f"{subject}_head_reference_{a:04d}_{b:04d}.mp4"
    load = {"name": name, "first": a, "frames": n, "subject": subject, "others": [k for k in theirs if k != subject], "text": name,
            "motion_file": str(reference), "motion_first": a}
    part, part_said = part_of(p, subject, parts, parts_by, preview, video, output, preview_files, classes_of)
    kept, kept_said = keep_outs_of(p, subject, preview, list(theirs), video, output, preview_files)
    load["others"] += [k["name"] for k in (kept or {}).get("made", [])]
    band = p.get("hair_band")
    if band is not None:
        # The band is cut out of ONE keep-out, which must be among the load's `others`, and a preview made by this tool
        # tracks the shot's subjects and nothing else. The first real load (2026-10-13, the second subject's 949-1009)
        # was written with the plan's band and refused by its own builder two steps later, at its plan.
        cuts = h3_config.MASKED_JOB_HAIR_BAND["cuts"]
        if cuts not in load["others"]:
            raise Refused(f"the plan asks a hair band for {subject!r} ({band['value'] if isinstance(band, dict) else band}), and a band is cut out of the keep-out "
                          f"{cuts!r} (h3_config.MASKED_JOB_HAIR_BAND). The preview {preview['name']} tracks {', '.join(theirs)} and no {cuts!r}, so the builder "
                          f"would refuse this load at its plan. Nothing is written. This tool's previews track a shot's subjects only, so no load made here can "
                          f"carry a band: take `hair_band` out of {subject!r} in the plan file and the load is made without one, or write by hand a load on a "
                          f"preview that tracks the keep-out")
        load["hair_band"] = band["value"] if isinstance(band, dict) else band
    subjects = [{k: v for k, v in s.items() if k in ("label", "phrase", "pick", "parts", "still", "pose")} for s in preview["subjects"]]
    # The capture's `pieces` for the load's subject, when the plan says one: a key on the subject in the job file, where
    # the chain reads it. The tool sets it for nobody; every load before the chain ran with `pieces=largest` typed into
    # a script, and the first loads through the chain had none (2026-10-13: specks of the part on an arm put a margin there).
    pieces = ""
    if p.get("pieces") is not None:
        if p["pieces"] not in PIECES or not str(p.get("pieces_by") or "").strip():
            raise Refused(f"the plan's `pieces` for {subject!r} is {p['pieces']!r} with `pieces_by` {p.get('pieces_by')!r}: `pieces` is one of {PIECES}, "
                          "and `pieces_by` says who chose it and on what")
        mine = next(s for s in subjects if s["label"] == subject)
        mine["pieces"], mine["pieces_by"] = p["pieces"], f"the plan ({p['pieces_by']})"
        pieces = (f" THE CAPTURE READS {subject}'s PART WITH pieces={p['pieces']}, from the plan ({' '.join(str(p['pieces_by']).split())}): each frame's largest "
                  "piece of the part is loaded and its other pieces are not; the capture's `files` step prints what that takes away, a frame at a time.")
    # The load's own part, where it names a subset: a key on the load's subject, where the chain reads it for the capture's `parts=`.
    if part:
        mine = next(s for s in subjects if s["label"] == subject)
        mine["parts_file"], mine["parts_of_this_load"], mine["parts_by"] = part["made"][0]["track"], list(part["parts"]), part["by"]
    # Two choices about this one load, each a person's with its words, neither with a default.
    chosen, zoom_block = "", None
    if zoomed is not None:
        if not zoomed.strip():
            raise Refused("--zoomed takes the reason in words: a load is rendered zoomed in on a person's word, after a look at its part's size, and the word is kept")
        load["render_region"] = "part"
        zoom_block = {"why": " ".join(zoomed.split()), "said_by": by, "at": now}
        chosen += (f" RENDERED ZOOMED IN on what is replaced (`render_region: part`), on {by}'s word: {zoom_block['why']}. One box for the whole load, the node's "
                   "own; the capture reads the node's plan back onto the frame, and the part's sizes and floor it prints are still of the whole frame.")
    if (least is None) != (not str(least_why or "").strip()):
        raise Refused("--least and --least-why go together: the capture's floor for a part is a person's number, and the reason is kept beside it")
    if least is not None:
        if int(least) < 0:
            raise Refused(f"--least {least}: a part's least size is 0 (nothing emptied by size) or more, in px of the whole frame")
        mine = next(s for s in subjects if s["label"] == subject)
        mine["least"], mine["least_by"] = int(least), f"{by}, {now}: {' '.join(least_why.split())}"
        chosen += (f" THE CAPTURE'S FLOOR FOR {subject}'s PART IS SET FOR THIS LOAD: least={int(least)} px of the whole frame, on {by}'s word: "
                   f"{' '.join(least_why.split())}.")
    note = (f"Written by bench/shot_to_load.py for {by}, {now}. ONE LOAD: {subject}, {region}, source frames {a} to {b} ({n} frames) of the shot "
            f"{shot['first']}-{shot['last']}, its masks from the preview {preview['name']} (job.preview), in the frozen form: the body model cut to the head "
            f"with the face points drawn, made on these frames by this tool's `reference` step. Its settings are in `defaults`, from workflows/h3_config.py. "
            f"THE TEXT IS NOT WRITTEN: {name}.txt beside this file is a person's, and texts.{name}.by says who wrote it."
            + pieces + part_said + kept_said + chosen + (f" CUT BY A LOOK: {cut_said}" if cut else ""))
    return {"job": {"name": name, "note": note, "source": source or preview["source"], "first": a, "frames": n, "canvas": list(preview["canvas"]),
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects,
                    "left_out": [{"subject": subject, "first": x, "frames": y - x + 1, "why": said} for x, y, said in cut],
                    "capture": f"data/{day}_{name}", "preview": preview_folder,
                    "texts": {name: {"file": f"{name}.txt", "frames": [a, b], "by": ""}},
                    **({"keep_outs": kept} if kept else {}), **({"part": part} if part else {}), **({"zoomed": zoom_block} if zoom_block else {}),
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
    name = preview_name(shot)
    if a.remake:
        # a name is taken by a job file in this folder and by a job folder of any day, as a load's is
        taken = lambda n: (Path(a.folder) / f"{n}.json").exists() or any(world.data.glob(f"????-??-??_{n}_job"))   # noqa: E731
        name = free_name(name, taken, True)
    ask = {}
    for text in a.ask:
        label, _, why = text.partition(":")
        if not label.strip() or not why.strip():
            raise Refused(f"--ask {text!r} is not `SUBJECT:why`")
        ask[label.strip()] = " ".join(why.split())
    seed_cuts = cuts_of_table(a.seed_cuts_of, shot) if a.seed_cuts_of else None
    if seed_cuts and not cast:
        raise Refused("--seed-cuts-of goes with --cast: the cuts say where a seed from the cast file may go")
    job = preview_job(shots, plan, shot, video, str(source), a.by, cast, load_chain.utc_now(), name=name, ask=ask, seed_cuts=seed_cuts)
    if seed_cuts:
        job["job"]["seed_cuts"] = seed_cuts
    if ask:
        job["job"]["made_to_ask"] = {"what": "this preview tracks a subject the shots record is unsure of, to ask the owner who is on the shot; nothing is built on it "
                                              "until they have answered", "subjects": ask, "by": a.by}
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "THE SHOTS RECORD NAMES AN EARLIER COPY")
    out = Path(a.folder) / f"{job['job']['name']}.json"
    if out.exists():
        raise Refused(f"{out} exists: a preview of this shot was made under that name. A preview made again (a pick corrected in the shots record, "
                      "or a box now in the cast file) takes --remake, which gives it the next free letter and leaves the earlier one as it is")
    print(f"{job['job']['name']}: source frames {shot['first']}-{shot['last']}; " + "; ".join(
        f"{s['label']} " + ("seeded from the cast (its `pick` decides nothing)" if "seeds" in s else f"picked on {s['pick_at']}") for s in job["job"]["subjects"]))
    print(job["job"]["note"])
    folder, _ = held_as_the_builder_holds(job["job"], a.day, world)
    if world.dry:
        print("DRY: nothing written, nothing run. The job was held to the builder's refusals that write nothing (the job block, the video's analysis "
              "folder and the file's identity, the two folders the build makes). NOT exercised: the build of the graph itself, and the queue.")
        return
    Path(a.folder).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(job, indent=1))
    print("wrote", out)
    build = [world.python, "bench/masked_job.py", "build", "preview", out, "--date", a.day]
    queue = [world.python, "bench/masked_job.py", "queue", folder, "preview", "--host", world.host, "--wait"]
    if not a.queue:
        print("NOT QUEUED (no --queue). To run it, on an empty queue:\n  " + " ".join(map(str, build)) + "\n  " + " ".join(map(str, queue)))
        return
    load_chain.empty_queue(world, "a preview is queued on an empty queue")
    # the commands' output is kept beside the job file: the build makes the job folder and refuses one that is there
    ctx = {"job_folder": folder, "name": job["job"]["name"], "logs": Path(a.folder) / "chain"}
    world.run(ctx, "build preview", build)
    world.run(ctx, "queue preview", queue)
    if seed_cuts:
        # the seeds were chosen on cuts read from an earlier table: hold the cuts this preview's tracker found to them
        for s in job["job"]["subjects"]:
            saved = world.output / job["job"]["output"] / job["job"]["name"] / f"{job['job']['name']}_{s['label']}_shots_00001_shots.json"
            if not saved.is_file():
                print(f"THE CUTS WERE NOT COMPARED for {s['label']}: no table at {saved}")
                continue
            found = [[int(c["first_frame"]), int(c["last_frame"]) + 1] for c in json.loads(saved.read_text())["shots"]]
            if found != seed_cuts["cuts"]:
                raise Refused(f"THE PREVIEW {job['job']['name']} IS MADE AND IS NOT TO BE BUILT ON: its tracker cut the shot into {len(found)} cut(s) and the seeds "
                              f"were chosen on the {len(seed_cuts['cuts'])} of {seed_cuts['read_from']}, which are not the same cuts ({s['label']}'s table, {saved})")
            print(f"{s['label']}: the cuts this preview's tracker found are the {len(found)} the seeds were chosen on")
    print("THE TILES ARE TO BE LOOKED AT before a load is built on this preview: a pick by the largest person is a guess.")


def step_job(a, world, cut=cut_keep_out, hashed=None, inside=part_inside, classes_of=part_classes) -> None:
    shots, plan = _load(a.shots, a.plan)
    shot = shot_of(shots, a.shot)
    pfolder = Path(a.preview)
    for name in ("job.json", "preview.queued.json"):
        if not (pfolder / name).is_file():
            raise Refused(f"the preview's job folder {pfolder} holds no {name}")
    ran = json.loads((pfolder / "preview.queued.json").read_text())
    if ran.get("status") != "success":
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
    # a load of some of the plan's parts is named by them, a letter a part: the hair alone is `hp`, neither `hfp` nor `fp`
    theirs = plan["subjects"][a.subject].get("parts") or []
    form = FORM[region] if not a.parts else "".join(PART_LETTER.get(k, "x") for k in theirs if k in a.parts) + "p"
    name = free_name(f"{a.subject}_{form}_{first:04d}", taken, a.remake)
    rel = str(pfolder.resolve().relative_to(REPO.resolve())) if pfolder.resolve().is_relative_to(REPO.resolve()) else str(pfolder)
    job = load_job(shots, plan, shot, a.subject, preview, rel, video, world.output, a.by, a.day, name, a.first, a.frames, a.why or "", load_chain.utc_now(),
                   source=str(source), why_before=a.why_before or "", why_after=a.why_after or "", zoomed=a.zoomed, least=a.least, least_why=a.least_why,
                   preview_files=ran.get("files") or [], parts=a.parts, parts_by=a.parts_by, classes_of=classes_of)
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "ITS PREVIEW WAS MADE ON AN EARLIER COPY")
    out = folder / f"{name}.json"
    print(f"{name}: {a.subject}, {region}, source frames {job['loads'][0]['first']}-{job['loads'][0]['first'] + job['loads'][0]['frames'] - 1}; "
          f"its reference will be {job['loads'][0]['motion_file']}")
    print(job["job"]["note"])                # the load in words, as the preview step prints its own: a dry run shows every choice made for it
    load_held_as_the_builder_holds(job, world)
    # the keep-outs the plan names are made before the job file that points at them is written; a dry run says what would be
    for line in make_part(job["job"].get("part"), world, cut, hashed, inside) + make_keep_outs(job["job"].get("keep_outs"), world, cut, hashed):
        print(line)
    if world.dry:
        print("DRY: nothing written. The load was held to the builder's refusals that need neither its capture nor its text (the job block, the file's "
              "identity, the load's keys and settings, the shortest load, left_out). NOT exercised: anything read from a capture, a text or a mask.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(job, indent=1))
        print("wrote", out)
    print(f"NEXT: bench/shot_to_load.py reference {out} --by {a.by}\nTHEN IT STOPS FOR THE TEXT: write {folder / (name + '.txt')}, then say who wrote it: "
          f"bench/shot_to_load.py text {out} --by {a.by} --writer \"who, and from which frames\"")


def step_reference(a, world, figures=figures_of) -> None:
    ctx = load_chain.read_job(Path(a.job_file), need_text=False)
    # this step's output is kept beside the job file, as the preview's is: the load's job folder is the chain's to make,
    # at `first`, and until 2026-10-13 a reference made it, before anything of the load was built
    ctx["logs"] = Path(a.job_file).parent / "chain"
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
        if not (made.get("face_on_the_track") or {}).get("known"):
            # a reference made before 2026-10-14 read its points in the track's box alone, and on a close group drew
            # another person's face and passed (the lead's 3167): whether its face is on the subject is not known
            raise Refused(f"--existing: {record} does not say its face was held to {subject}'s track (made before the points were, or on the whole "
                          "frame): the reference is made again, in a new folder")
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
    load_chain.free_for_a_pose_pass(world)
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
    print(f"IT STOPS HERE FOR THE TEXT: {ctx['text']} is a person's. When it is written, say who wrote it: bench/shot_to_load.py text {a.job_file} --by {a.by} "
          f"--writer \"who, and from which frames\". Then: bench/load_chain.py first {a.job_file} --by ...")


def step_text(a, world) -> None:
    ctx = load_chain.read_job(Path(a.job_file), need_text=False)
    label, text = ctx["load"]["text"], ctx["text"]
    if not str(a.writer or "").strip():
        raise Refused("--writer takes a person's words: who wrote the text, and from which frames")
    if not text.is_file() or not text.read_text().strip():
        raise Refused(f"the load's text {text} is not written, or is empty: it is written first, and then its writer is said")
    planned = world.data / f"{ctx['day']}_{ctx['job']['name']}_job" / "job.plan.json"
    if planned.exists():
        raise Refused(f"the plan of {ctx['name']} is built ({planned}): its job file is the one the plan was built from and is not edited after. "
                      "A text or a writer that changed is a load made again")
    d = json.loads(Path(a.job_file).read_text())
    was = str(d["job"]["texts"][label].get("by") or "").strip()
    said = f"{' '.join(a.writer.split())}; entered {load_chain.utc_now()} by {a.by} (bench/shot_to_load.py text)"
    print(f"texts.{label}.by: {said}" + (f"\n  REPLACES: {was}" if was else ""))
    if world.dry:
        print("DRY: nothing written.")
        return
    d["job"]["texts"][label]["by"] = said
    Path(a.job_file).write_text(json.dumps(d, indent=1))
    print(f"wrote {a.job_file}. NEXT: bench/load_chain.py first {a.job_file} --by {a.by}")


STEP = {"preview": step_preview, "job": step_job, "reference": step_reference, "text": step_text}


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
    sub.choices["preview"].add_argument("--ask", action="append", default=[], metavar="SUBJECT:why",
                                        help="track a subject the record is `unsure` of on this shot, because the preview is made to ask the owner who is on it; "
                                             "repeatable. No load is built on it until they have answered")
    sub.choices["preview"].add_argument("--seed-cuts-of", metavar="JOB.json=TABLE.json",
                                        help="the tracker's own cuts of this shot, read from the shots table an earlier preview of the same frames saved (with "
                                             "that preview's job file): each cut that holds an answer in the cast file gets its own seed. Without it a shot "
                                             "gets one seed")
    sub.choices["preview"].add_argument("--queue", action="store_true", help="build and queue the preview, on an empty queue")
    sub.choices["preview"].add_argument("--remake", action="store_true", help="the next free letter after the name, for a preview made again; the earlier one stands")
    j = sub.choices["job"]
    j.add_argument("--subject", required=True)
    j.add_argument("--preview", required=True, help="the job folder of the shot's preview")
    j.add_argument("--first", type=int, help="the load's first frame, when a look cut the shot's")
    j.add_argument("--frames", type=int, help="the load's frame count, when a look cut the shot's")
    j.add_argument("--why", help="what a look cut and why; into the job's left_out, for both ends of the load unless one has its own")
    j.add_argument("--why-before", help="what was cut BEFORE the load and why, when it is not the reason for the frames after")
    j.add_argument("--why-after", help="what was cut AFTER the load and why, when it is not the reason for the frames before")
    j.add_argument("--remake", action="store_true", help="the next free letter after the name, for a load made again")
    j.add_argument("--parts", nargs="+", metavar="NAME", help="this load's part is these of its subject's plan parts and no other (hair, face_and_neck, ...); "
                   "cut from the preview's class map as a keep-out is, and handed to the capture as the load's part")
    j.add_argument("--parts-by", metavar="WHO", help="who chose these parts for this load, and on what; kept in the job file")
    j.add_argument("--zoomed", metavar="WHY", help="render this load zoomed in on what is replaced, on a person's word; the reason is kept in the job file")
    j.add_argument("--least", type=int, metavar="PX", help="the capture's floor for the subject's part on this load, in px of the whole frame; with --least-why")
    j.add_argument("--least-why", metavar="WHY", help="why the floor is set there; kept beside it in the job file")
    r = sub.add_parser("reference")
    r.add_argument("job_file")
    r.add_argument("--plain", metavar="WHY", help="take the plain cut, on a person's word; the reason is kept in the job file")
    r.add_argument("--existing", action="store_true", help="a load made again on the same frames: read the reference that is there, run nothing")
    r.add_argument("--model-store", default="", help=f"the folder holding the pose model's folder; over the environment's {load_chain.MODEL_STORE}")
    t = sub.add_parser("text")
    t.add_argument("job_file")
    t.add_argument("--writer", required=True, metavar="WORDS", help="who wrote the load's text, and from which frames; into texts.<label>.by")
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
