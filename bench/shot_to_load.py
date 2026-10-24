#!/usr/bin/env python3
"""From a shot to a load's job file: the preview of the shot, the load's job, its head reference. Then it stops for the text.

    <python> bench/shot_to_load.py preview --shots SHOTS.json --plan PLAN.json --shot FIRST --folder INPUTS --source VIDEO --by SESSION [--cast CAST.jsonl] [--queue] [--remake]
        [--for LABEL ...: the replacement(s) it is made for, required where anything is laid on the shot]
        [--seed-cuts-of EARLIER_PREVIEW_JOB.json=ITS_SHOTS_TABLE.json] [--ask "SUBJECT:why the owner is asked about them"]
        [--no-memory-selection "a person's reason for tracking with ComfyUI's last-six-frames memory"]
        (somebody looks at the trackers' tiles; a wrong pick is a person's line in the shots record or a box in the cast
        file, and the preview is made again with --remake, under the next letter)
    <python> bench/shot_to_load.py job --shots SHOTS.json --plan PLAN.json --shot FIRST --subject LABEL --preview PREVIEW_JOB_FOLDER \\
        --folder INPUTS --by SESSION [--first F --frames N --why "what a look cut" | --why-before ".." --why-after ".."] [--remake]
        [--zoomed "why this load is rendered zoomed in"] [--least PX --least-why "why the floor is set there"]
        [--parts NAME ... --parts-by "who chose these parts for this load, and on what"]
        [--no-face-on FIRST-LAST="who looked, at which sheet (its file), and that the face does not show there"]
        [--seam FRAME[,FRAME...] --seam-why "a person's reason": every seam of one plan of the chained window lengths]
        [--hold-part FIRST-LAST[@FRAME]="who looked, at which sheet (its file), and that her part belongs there"]
            (with @FRAME: held from that frame's graded part, placed where it was and not moved)
        [--seed N --seed-why "who asked for this seed and why": the song node's seed for this load in place of h3_config.SEED]
    <python> bench/shot_to_load.py reference JOB.json --by SESSION [--plain "why"] [--model-store FOLDER] [--existing]
        (it prints, under the figures, where a face may not show by two cues of the pose model's points, as data for the sheet)
        (a person writes the shot's place and action, one sentence)
    <python> bench/shot_to_load.py text JOB.json --by SESSION --plan PLAN.json
        (or, the exception: --by-hand "why" for a text a person wrote whole beside the job file)
        (then `bench/load_chain.py first JOB.json`)
    <python> bench/shot_to_load.py pair FIRST_JOB.json SECOND_JOB.json --plan PLAN.json --folder INPUTS --by SESSION [--name NAME] [--remake]
        (two chain-made loads of the same frames as ONE load of two subjects, the larger region first; its text is written whole,
        `text --by-hand`; then `bench/load_chain.py first`, and every step of the chain as for any load)

**What it buys.** The step before a load's chain was hand-work for every shot: a preview job typed from the shots
record, a pose pass and a head reference run with each other's outputs for arguments, and a job file copied from
the last one with its settings typed again. The video has some sixty shots still to make. This writes the job
files and runs the two tools, from the video's own records, and leaves to a person only what is a person's.

**The defaults a load carries** (the owner's word of 2026-10-14: make these our defaults and stop forgetting the tools we
built): zoomed in where the node's box allows (`--whole-frame "why"`, or the plan's `zoom: false` with `zoom_by`, for the
whole frame); the reference's eye in the contrast form and its mouth by the lip check (`REFERENCE_DEFAULT`; the plan's
`reference: {mouth, iris, why}`, or the reference step's `--mouth`/`--iris` with `--form-why`); the reference cut to the head's box,
or to the face's or the grey head's with the plan's `reference: {crop: "face" | "cranium", why}` (`head_reference.CROP_FORMS`); and from the plan, the
strangers keep-out, `pieces`, the hair band and the restore spans; and from the preview its masks come from, SAM 3.1's
memory selection in the trackers (on since 2026-10-17; the preview step's `--no-memory-selection "why"` turns it off,
kept in the preview's job file as `sam_memory_selection`). The job step prints them as THE DEFAULTS THIS LOAD
CARRIES, a line each with where it comes from, and keeps the same in the job file (`job.defaults`).

**What is read, and from where.**

- The shots record (`data/<video>/shots.json`): the shot's frames and who is in it. For a subject with no line in
  the cast file, the frame they are picked on is a frame the record names AS A FIELD (`frames_on`, then
  `frames_boxed`; `bench/page_answers_to_cast.py --shots` writes both), never a sentence of `how`; the shot's
  middle is the last fallback and the job's note says it was one. A shot's `order` (a person's list, written
  after a tile shows a wrong pick) is the order its trackers run in; else the plan's subject order, which is a
  starting value and nothing measured.
- The plan (`data/<video>/plan.json`): for each subject the `region` (what is replaced), and, new with this tool,
  the `still` its face is taken from, the `parts` the part model ticks for it, and optionally `hair_band`. A
  subject whose plan lacks `still` or `parts` is refused with the keys to add. Optionally `context` with `context_by`:
  what the real frames a short load is padded with show, `as filmed` (the default) or `face covered`, copied onto each
  load as `context_shows` (`context_of`). Optionally `head_marks`, a person's marks on the subject's head and hair by span
  (`head_marks_of`): copied onto the subject in the job file, cut to the load's frames, and handed to the capture. Optionally
  `source_facial_hair: true` with `source_facial_hair_by`, on a face-only subject whose source actor shows facial hair to replace
  (`source_facial_hair_of`): copied onto the subject as `facial_hair: keep`, so the capture's head tube gives back the piece of the
  part joined to the face that it would cut at the jaw. A plan look's stay item made as one
  keep-out (`look.stay[]` with `as: "one keep-out"`, `replaces_keep_out`, `classes_on_her_track`, `hold_frames`, and
  `look_by`) is made by the job step in place of the keep-out it replaces (`stay_of`, `cut_stay`). Optionally `pieces` (with
  `pieces_by`: who chose it, on what): how the capture reads the subject's part, copied by `job` onto the
  subject in the load's job file, where the chain passes it to the capture. This tool sets it for nobody.
  Optionally `keep_outs` (with `keep_outs_by`): which of `held`, `hands`, `clothes` and `strangers` that subject's
  loads keep as the source's (`KEEP_OUTS`); `strangers` also holds the load's part to its subject's track
  (`part_on_track`). `job` makes each from the preview's own files, names it in the load's `others`
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
  `--track-not SUBJECT:FIRST-LAST=WHY` with `--track-not-by`: another subject's preview track is not that person
  there, and every step reads a copy with those frames emptied; the step prints where one track lies more than
  `TRACK_INSIDE` inside the load's subject's, a figure for a person to read;
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
  its own builder refused it at the plan. THE SEED is `h3_config.SEED` unless a person asks for another:
  `--seed N --seed-why "who asked and why"` writes it as the file's `defaults.seed` with `defaults.seed_by`
  (the words, the session and the time) beside it, and says it among THE DEFAULTS THIS LOAD CARRIES; the song
  node reads it (`workflows/masked_job_graph.py`, `run["seed"]`). A seed is per load, so `--carry-words-from`
  does not carry it, and `--seed` with no words, words with no `--seed`, and a seed that is not a whole number
  the song node takes are refused. THE ZOOM TO A TARGET HEAD SIZE is off unless a person asks for it on one load
  (2026-10-22, to test it): `--zoom-up-to N --zoom-up-to-why "who asked and why"` writes `defaults.zoom_up_to`
  with `defaults.zoom_up_to_by` the same way, and the Masked Source may then enlarge the load past its own 2, up
  to N (`video_mask.ZOOM_MOST_ASKED` at most), until the head is `video_mask.ZOOM_HEAD_TARGET` px tall. Refused on
  a load rendered on the whole frame; not carried by `--carry-words-from`. THE MARGIN is `h3_config.MASKED_LOAD_MARGIN`
  by the plan's region unless a person asks for another on one load (2026-10-23, to test a larger mask round her
  hair): `--margin N --margin-why "who asked and why"` writes `defaults.margin` as N with `defaults.margin_by` the
  same way, and every reader of the load's margin takes it from there (the capture's region and her hair's hole,
  the plan and the render's `grow_pixels`). Refused: a margin that is not a whole number from 1 to `MARGIN_MOST`,
  no words, words with no `--margin`, and one under the feather (the builder's refusal); not carried by
  `--carry-words-from`.
- `reference`: on the LOAD's own frames, so the box is round the head on those frames: the pose pass in the
  subject's tracked box, then `bench/head_reference.py`, which writes the drawn form and the plain cut both. It
  goes to `Video/<video>/references/<subject>_<first>_<last>/`: a reference belongs to a shot, a subject and a
  span, never to a load's name. A folder of that name a render read, or one not of today's form, is never moved or
  written into: the job names the next of `_b`, `_c` and says why (`reference_folder`, the lead's ruling of 2026-10-16). THE DRAWN FORM IS THE DEFAULT and nothing here decides otherwise: the face's
  height and the lash-line gap are printed at the canvas and as they come out in the reference at its scale,
  with the frames under the drawing line, and written into the job file. The plain cut is taken on a person's
  word only, `--plain "why"`, and the form, the why and who said it go into the job file. A load made again on
  the same frames reads the reference that is there: `--existing` runs nothing, holds the record beside it to
  the job's subject and frames, and writes the same block from it. A reference whose record does not say its face
  was held to the subject's track is refused there (2026-10-14: the points of the lead's 3167 were read in her
  track's box alone, drew another woman's face on eleven frames and passed; `bench/pose_along_shot.py`).

**It stops for the text.** The job file's text entry names a file and no writer. `bench/load_chain.py` refuses the
load until the file is there and `texts.<label>.by` says who wrote it.

- `text`: writes the load's text, generated from the reference's record, the plan and the voice table (`--plan`;
  no place-and-action sentence since 2026-10-15, `load_text.NO_SCENE`), or takes one a person wrote whole
  (`--by-hand WHY`, the file there with words in it, `--writer` a person's words: who, and from which frames).
  The test-only `--say-not-used`, which wrote a generated text in the old form with both "not used" clauses for the A/B of
  2026-10-21, is retired since 2026-10-22 (`bench/load_text.py`); job files of that day keep its reason in
  `texts.<label>.generated_from.not_used_said`, which nothing reads.
  What made it goes into `texts.<label>.by` with when it was entered and by which session.
  Until 2026-10-13 that key was set by opening the job file and editing it. Words that were there are printed
  as replaced. Refused once the load's plan is built: the job file is then the one the plan was built from, and
  the chain's render holds it to that copy byte for byte.

**What it cannot see.** Whether a tracker took the right person (the tiles). Whether a face is too small or too
turned for the drawn lines (the sheet beside the reference, and the figures). What the frames a look should cut
are. Whether the plan's order suits a shot.
"""
from __future__ import annotations

import argparse
import copy
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

import h3_config  # noqa: E402
import load_chain  # noqa: E402
from load_chain import Refused  # noqa: E402

#: Where a job's renders go under the server's output folder. One place: today a session's folder under the
#: video's, until the Open List's row z21 is answered.
OUTPUT_KEY = "Video/{video}/{session}"
#: Where a head reference goes: by the video, the subject and the span it was made on.
REFERENCES = "Video/{video}/references/{subject}_{first:04d}_{last:04d}"
#: The letters a span's reference folder takes after its plain name, when that one was read by a render or is not of today's form
#: (`reference_folder`; the lead's ruling of 2026-10-16: never move or write into a reference folder any render read).
REFERENCE_SUFFIXES = tuple(f"_{c}" for c in "bcdefghijklmnopqrstuvwxyz")


def reference_misfits(record: Path, first: int, last: int) -> str | None:
    """Why a reference's record is not of today's form for these frames, or None: the checks `--existing` holds first (its frames
    and its pass, and its face held to the subject's track)."""
    if not record.is_file():
        return f"{record.name} is not there"
    made = json.loads(record.read_text())
    if made.get("frames") != [first, last] or made.get("passed") is not True:
        return f"{record.name} is of frames {made.get('frames')} with passed={made.get('passed')!r}"
    if not (made.get("face_on_the_track") or {}).get("known"):
        return f"{record.name} does not say its face was held to the subject's track"
    return None


def loads_that_read(folder: Path, data: Path | None) -> list[str]:
    """The loads that read a folder: a queued graph (`<data>/*_job/<stem>.queued.json` with its `<stem>.json`) of which the queue
    record or the graph itself names a file in it. The graph is read too: a queue record before 2026-10-15 named no input."""
    if data is None:
        return []
    key = str(folder).rstrip("/") + "/"
    out = set()
    for q in Path(data).glob("*_job/*.queued.json"):
        graph = q.with_name(q.name[:-len(".queued.json")] + ".json")
        if key in q.read_text(errors="replace") or (graph.is_file() and key in graph.read_text(errors="replace")):
            out.add(q.name.split(".")[0])
    return sorted(out)


def reference_folder(output: Path, video: str, subject: str, first: int, last: int, data: Path | None) -> tuple[Path, str]:
    """The folder a load's reference goes to, and why it is not the span's plain one ("" when it is): `<subject>_<first>_<last>`
    while it is not there, or is there of today's form and read by no render (the reference step takes it with `--existing`);
    else the next of `REFERENCE_SUFFIXES` by the same rule. A folder a render read is never moved or written into: the ledger, the
    sidecars and the review pages find it by path (the lead's ruling of 2026-10-16)."""
    base = output / REFERENCES.format(video=video, subject=subject, first=first, last=last)
    said = []
    for suffix in ("",) + REFERENCE_SUFFIXES:
        folder = base.with_name(base.name + suffix)
        if not folder.exists():
            return folder, "; ".join(said)
        readers = loads_that_read(folder, data)
        misfit = reference_misfits(folder / f"{subject}_head_reference_{first:04d}_{last:04d}.json", first, last)
        if not readers and not misfit:
            return folder, "; ".join(said)
        said.append(f"the span's folder {folder.name} was read by {', '.join(readers)}" + (f" and {misfit}" if misfit else "") if readers
                    else f"the span's folder {folder.name} is not of today's form ({misfit})")
    raise Refused(f"every reference folder of {subject} on {first}-{last} up to {base.name}{REFERENCE_SUFFIXES[-1]} is taken: " + "; ".join(said))


#: A load's form in its name, by the plan's region: the part, then `p` for the face points of the frozen form.
FORM = {"face": "fp", "face and hair": "hfp"}
def renders(region: str, parts: list[str] | None) -> str:
    """What a load renders, in words: the plan's region, or with `--parts` those parts of it (mrarc's dry run of second_hp_5461b,
    2026-10-15: a hair load's header said "face and hair")."""
    return f"{' and '.join(parts)} alone (of the plan's {region}, --parts)" if parts else region


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
#: Where a track copied with frames emptied goes (`--track-not`): by the video, the preview and the subject and frames.
TRACKS_AT = "Video/{video}/tracks/{preview}"
#: The share of one subject's track inside another's above which the job step names the frames (`track_over_track`).
#: Reasoned: more than half of it lies on the other person, which is how the lead's track lay on the second subject on
#: 1003-1009 (a card's count, 2026-10-13). A line for what is printed, not a verdict: a person says `--track-not`.
TRACK_INSIDE = 0.5
#: THE KEEP-OUTS A PLAN MAY NAME FOR A SUBJECT: things a part load must leave as the source's. Each is a mask video on
#: the preview's own frames, a subject of the load's capture and one of the load's `others`. `held` is the preview's
#: own held-things file as saved; `hands` and `clothes` are classes of the preview's class map of that subject, not
#: clipped to the track and not cleaned to a largest piece; `strangers` is the preview's file below.
#: INHERITED from the session cuts of 2026-10-12 (the hands and arms after a hand at her face was drawn again; the
#: clothing after a blouse strap in the margin was) and MEASURED 2026-10-13: cut again from that day's class map of a
#: shot of 457 frames, each is that day's own mask on every frame with no pixel differing; with `Torso` added to the
#: clothing it is equal on 29 of the 457, so the torso was left out on purpose. Every load of the second subject
#: through this tool carried none of them until that day, and the first render drew her apron as the still's top.
#: `strangers` (2026-10-14, ruled by the lead): every person the part model finds on the whole frame who is not one
#: of the preview's subjects, a piece touching a subject only where it lies on another person the detector found (the
#: owner, 2026-10-23), the file `bench/strangers.py run` writes for the preview; looked for, as `held` is.
KEEP_OUTS = {"held": None,
             "strangers": None,
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


def _check_analysed(job: dict, world, default_source: Path | None = None) -> None:
    import clip_folder
    import masked_job
    name = Path(job["source"]).name
    try:
        clip_folder.folder_of_file(name, world.data)
    except clip_folder.Refused:
        if default_source is not None and Path(job["source"]).is_file():
            masked_job.analysed({**job, "source": str(default_source)}, world.data)
            return
    masked_job.analysed(job, world.data)


def held_as_the_builder_holds(job: dict, day: str, world, default_source: Path | None = None) -> tuple[Path, Path]:
    """A preview's job held to every refusal of the builder that writes nothing, before anything is written or
    queued: the job block, the video's analysis folder and the file's identity, and the two folders the build
    makes, which must not exist (`bench/masked_job.py::build`: "A run never reuses a name"). Returns those two
    folders. What it does not do is build the graph: that is the builder's, and only a real run shows it."""
    import masked_job
    import masked_job_graph
    try:
        masked_job_graph.check_job(job)
        _check_analysed(job, world, default_source=default_source)
    except (ValueError, masked_job.Refused) as e:
        raise Refused(str(e)) from None
    folder, saves_to = masked_job.job_folder(job, day, world.data), masked_job.output_folder(job, world.output)
    for path, what in ((folder, "job folder"), (saves_to, "output folder")):
        if path.exists():
            raise Refused(f"the {what} {path} exists. A run never reuses a name: a preview made again takes --remake, which gives it the next free letter")
    return folder, saves_to


def load_held_as_the_builder_holds(given: dict, world, default_source: Path | None = None) -> None:
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
        _check_analysed(job, world, default_source=default_source)
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
    file, handed in, and the job says it was read; the job file beside it is how the table's frames are known.
    Since 2026-10-23 every box is a seed on its frame whatever the cuts (`cast_file.seeds`), so the cuts change only
    how the note counts them and the check after the queue that the new preview's tracker cut the shot the same way."""
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
                ask: dict | None = None, seed_cuts: dict | None = None, no_memory_selection: str | None = None,
                no_head_track: str | None = None, head_per_stretch: str | None = None) -> dict:
    """The job file of one shot's preview: every subject the record says is in it, a pose pass for each. `name` is
    the preview's, `shot_c_<first>` when left out; a preview made again is handed its lettered name.

    `ask` is {subject: why}: a subject the record is UNSURE of on this shot, tracked all the same because the preview
    is MADE TO ASK THE OWNER who the people on it are (2026-10-13: a shot cut out of another had no answer of the
    owner's on any frame, all three subjects unsure, and the preview its who-page picture is cut from was refused
    for having nobody to track). An unsure subject is planned by nothing, and it is asked about: the tracker picks
    as it does with no seed, the job says the preview is made to ask, and no load is built on it
    (`load_job` refuses a subject the record does not say is in the shot). A subject who is `yes` needs no asking
    and one who is `no` is the owner's answer: both are refused here.

    `no_memory_selection` is a person's reason for tracking with ComfyUI's last-six-frames memory in place of SAM 3.1's
    memory selection, the chain's default (`h3_config.MASKED_JOB_SAM_CORRECTIONS`): kept in the job block as
    `sam_memory_selection`, which the graph reads (`masked_job_graph.memory_selection_of`) and a load's defaults name.

    `no_head_track` is a person's reason for turning off SAM 3.1 head propagation (`track_head`), the chain's default
    (`h3_config.SUBJECT_TRACK["track_head"]`): kept in the job block as `sam_head_track` and on each subject's
    `track_head`."""
    import h3_config
    ask = ask or {}
    if no_memory_selection is not None and not " ".join(str(no_memory_selection).split()):
        raise Refused("--no-memory-selection takes the reason in words: memory selection is the chain's default, and the word is kept")
    if no_head_track is not None and not " ".join(str(no_head_track).split()):
        raise Refused("--no-head-track takes the reason in words: the SAM 3.1 head track is the chain's default, and the word is kept")
    if head_per_stretch is not None and not " ".join(str(head_per_stretch).split()):
        raise Refused("--head-per-stretch takes the reason in words: the head track is main's span head track by default (the owner, "
                      "2026-10-23), and a preview that tracks the head per stretch says why")
    if head_per_stretch is not None and ";" in str(head_per_stretch):
        raise Refused("--head-per-stretch reason cannot contain ';': the tracker splits seeds lines on ';'")
    # which head track the trackers run (`subject_track.py::parse_head_mode`): main's span head track, the default, or per
    # stretch on a person's word; the line rides in every subject's seeds text, which the node already reads
    head_mode = ({"mode": "per stretch", "why": " ".join(str(head_per_stretch).split()), "said_by": by, "at": now}
                 if head_per_stretch is not None else
                 {"mode": "span", "said_by": "the chain's default", "why": "main's span head track (the owner, 2026-10-23)"})
    memory = ({"on": False, "why": " ".join(str(no_memory_selection).split()), "said_by": by, "at": now} if no_memory_selection is not None else
              {"on": bool(h3_config.MASKED_JOB_SAM_CORRECTIONS["memory_selection"]), "said_by": "the chain's default",
               "why": "h3_config.MASKED_JOB_SAM_CORRECTIONS (the owner's word, 2026-10-17)"})
    head_track = ({"on": False, "why": " ".join(str(no_head_track).split()), "said_by": by, "at": now} if no_head_track is not None else
                  {"on": bool(h3_config.SUBJECT_TRACK["track_head"]), "said_by": "the chain's default",
                   "why": "h3_config.SUBJECT_TRACK['track_head'] (the owner's word, 2026-10-19)"})
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
        one = {"label": label, "pick": PICK, "parts": list(p["parts"]), "still": p["still"], "track_head": head_track["on"]}
        seed = seeds[label]
        if seed is not None:
            # how many other subjects the job holds, so this tracker refuses a box prompt where it knows none of them on a
            # frame (`subject_track.py::parse_job_others`; the first tracker of a job is given no earlier tracker's mask)
            others_here = len(present) - 1
            one["seeds"] = seed["text"] + (f"\njob: {others_here} other subject{'s' if others_here != 1 else ''}" if others_here else "")
            # the cast line's own words, with no hole where one of its fields is empty: a line entered from a page answer
            # with no time on it reads "on the page at  (" in the file, and that is said here as the file's
            rows = [r for r in seed["shots"] if r["seeds"]]
            why = "; ".join(dict.fromkeys(x["why"] for r in rows for x in r["seeds"]))
            holed = " ".join(why.split()) != why
            used = [x["from_frame"] for r in rows for x in r["seeds"]]
            said.append(f"{label}: SEEDED from the cast file ({' '.join(why.split())}"
                        + ("; a field of that line is EMPTY in the cast file, as it reads there" if holed else "")
                        + f"): THE SEED DECIDES who is tracked; `pick: {PICK}` stands in the job file only because the builder asks one of every "
                          f"subject, and decides nothing here. SEEDED ON SOURCE FRAME(S) {', '.join(map(str, used))}"
                        + (f", in {len(rows)} of the tracker's {len(seed['shots'])} cuts" if seed_cuts else "")
                        + (": EVERY BOX OF THE OWNER'S INSIDE THE SHOT is a seed on its frame, and the tracker keeps, frame by frame, the track its "
                           "own score holds best (subject_track.py::merge_tracks; what each box took, how, and the frames it was kept on are in "
                           "this preview's shot table, `seeds`)" if len(used) > 1 else ""))
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
            + (f" THIS IS A PREVIEW MADE AGAIN: {name}, after {base} or a letter before it, which stand as they were made and are not read here." if name != base else "")
            + (f" SAM 3.1'S MEMORY SELECTION IS OFF in this preview's trackers, on {by}'s word: {memory['why']}." if not memory["on"] else "")
            + (f" THE SAM 3.1 HEAD TRACK IS OFF in this preview's trackers, on {by}'s word: {head_track['why']}." if not head_track["on"] else "")
            + (f" THE HEAD IS TRACKED PER STRETCH in this preview's trackers (`--head-per-stretch`), on {by}'s word: {head_mode['why']}."
               if head_mode["mode"] == "per stretch" else ""))
    if head_mode["mode"] == "per stretch":
        line = f"head: per stretch: {head_mode['why']}"
        for one in subjects:
            one["seeds"] = (one["seeds"] + "\n" if one.get("seeds") else "") + line
    return {"job": {"name": name, "note": note, "source": source, "first": shot["first"], "frames": frames, "canvas": canvas,
                    **({"source_frames": int(shots["source_frames"])} if shots.get("source_frames") is not None else {}),
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects, "left_out": [], "texts": {},
                    "sam_memory_selection": memory, "sam_head_track": head_track, "sam_head_mode": head_mode}, "defaults": {}, "loads": []}


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
        if k == "strangers":
            import strangers
            tracks = {s["label"]: str(strangers.tracks_of(preview, preview_files, output)[s["label"]]) for s in preview["subjects"]}
            made.append({"name": k, "track": str(folder / f"{preview['name']}_strangers.mkv"), "made_by": "bench/strangers.py run",
                         "frames": preview["frames"], "tracks": tracks,
                         "what": ("every person the part model finds on the whole frame, less the preview's subjects' tracks grown, a piece "
                                  "touching a subject only where it lies on another person the detector found (bench/strangers.py)")})
        elif KEEP_OUTS[k] is None:
            made.append({"name": k, "track": str(saved("held_things")), "what": "the preview's own held-things file, as saved"})
        else:
            made.append({"name": k, "track": str(folder / f"{preview['name']}_{subject}_{k}.mkv"), "classes": list(KEEP_OUTS[k]),
                         "cut_from": str(saved("classes")), "what": "those classes of the preview's class map of the subject, not clipped to the track"})
    said = (f" KEPT AS THE SOURCE'S ON THIS LOAD, from the plan ({by}): " + "; ".join(
        f"`{k['name']}` ({k['what'] if 'classes' not in k else ' + '.join(k['classes']) + ' of the class map'})" for k in made)
        + ". Each is one of the load's `others` and a subject of its capture, on the preview's frames.")
    return {"of": subject, "by": f"the plan ({by})", "preview": preview["name"], "first": preview["first"], "made": made}, said


def part_classes() -> dict:
    """The classes of each part's name, the part model's own table (`sapiens2_parts.PARTS`): one copy, read and not written here."""
    import capture_masked_run as cap
    return cap._pack("sapiens2_parts").PARTS


def seam_window(frames: int, context: int) -> dict[tuple[int, ...], tuple[int, list[int]]]:
    """For a load longer than one window: the load-relative frames where each later window of a chained plan writes its first new frame
    (its seams), with the window length that plans them and the windows' lengths (`loop_plan.plan_windows`, the song node's own planner),
    an entry for each chained length that plans more than one window (two lengths with the same seams would share one). Empty when one
    window holds the load. Every later window shares `context` frames with the one before it, so the first seam is the first window's
    length and each later seam is the one before it plus the length of the window between them less the context. Since 2026-10-15 (the owner, about
    20:15Z: "you can do shorter windows too just make sure the frame multiples we documented ... are right to avoid seams"; the lead's
    reading in docs/wiki/decisions.md) a plan of three windows or more is one of them: every window is a chained length, so the step grid
    holds from the load's first frame."""
    import capture_masked_run as cap
    lp = cap._pack("loop_plan")
    if frames <= max(lp.CHAIN_LENGTHS):                 # one window holds a load up to the longest chained length (masked_job_graph.window_for)
        return {}
    out = {}
    for w in lp.CHAIN_LENGTHS:
        try:
            plan = lp.plan_windows(frames, w, context)
        except ValueError:
            continue
        lengths = plan[0][1] if len(plan) == 1 else []
        if len(lengths) < 2:
            continue
        seams = [lengths[0]]
        for n in lengths[1:-1]:
            seams.append(seams[-1] + n - context)
        out[tuple(seams)] = (w, lengths)
    return out


def seam_of(n: int, a: int, seam: list[int] | None, seam_why: str | None, by: str, now: str) -> tuple[int | None, dict | None]:
    """THE SEAMS OF A LOAD LONGER THAN ONE WINDOW (the owner's word of 2026-10-15: one load over the whole shot, and "make sure the seam is
    at a non-critical place"): the song node chains windows inside the load, each sampled from the last `context` frames of the one before,
    and each later window's first new frame is a seam. `--seam A[,B...] --seam-why WHY` names every seam, and picks the window length
    whose plan puts them there (`seam_window`); the planner's own otherwise. (the load's `window` to set, or None; the defaults row, or
    None for a load one window holds)."""
    import h3_config
    import masked_job
    context = int(h3_config.MASKED_JOB_CONTEXT)                   # the load's context: the job step sets none, so the chain's
    if (seam is None) != (not str(seam_why or "").strip()):
        raise Refused("--seam and --seam-why go together: the frames the later windows begin writing on, and a person's reason")
    longest = max(masked_job.chain_lengths())
    one = longest                                                 # the planner's window; what one window holds
    reach = seam_window(n, context)
    said = lambda seams: ", ".join(str(a + k) for k in seams)     # noqa: E731
    tail = lambda seams: n - seams[-1]                                # noqa: E731  the new frames the last window writes
    # A NAMED PLAN'S LAST WINDOW writes at least one latent step of the load's own frames. Reasoned: under it a whole window renders for
    # a seam on the load's last frames (at 397 frames, 243's last window writes 1). On loads of 363-700 frames it refuses 208 of 1690
    # plans; the context (90) would refuse 1075, plans like 573's [294, 294, 192], whose last window writes 75
    import capture_masked_run as cap
    least_tail = int(cap._pack("loop_plan").STEP_CYCLE)
    window = None
    if seam is not None:
        if n <= one:
            raise Refused(f"--seam {','.join(map(str, seam))}: one window holds this load's {n} frames, so it has no seam")
        if len(set(seam)) != len(seam):
            raise Refused(f"--seam {','.join(map(str, seam))}: a seam is named twice")
        outside = [x for x in seam if not a < x < a + n]
        if outside:
            raise Refused(f"--seam {','.join(map(str, seam))}: {', '.join(map(str, outside))} is not inside the load {a}-{a + n - 1} after its first frame")
        named = tuple(sorted(s - a for s in seam))
        if named in reach and tail(named) < least_tail:
            raise Refused(f"--seam {','.join(map(str, seam))}: that plan's last window writes {tail(named)} new frame(s) of the load, less than one "
                          f"latent step ({least_tail}, the song node's STEP_CYCLE): a whole window rendered for a seam on the load's last frames. "
                          "Name another plan")
        if named not in reach:
            raise Refused(f"--seam {','.join(map(str, seam))}: the chained windows put this load's seams on "
                          + "; or ".join(f"{said(k)} (windows of {w})" for k, (w, _) in sorted(reach.items(), key=lambda x: -x[1][0])
                                         if tail(k) >= least_tail)
                          + " only: name every seam of one plan (each later window writing from there, the frames before it kept)")
        window = reach[named][0]
    if n <= one:
        return window, None
    w = window if window is not None else longest
    seams = next((k for k, (ww, _) in reach.items() if ww == w), None)
    lengths = reach[seams][1] if seams is not None else None
    carries = (f"windows of {lengths}: " + "; ".join(f"window {i + 2} writes from {a + s}, sampled from the frames {a + s - context}-{a + s - 1} before it"
                                                    for i, s in enumerate(seams))
               + f"; the last window writes {tail(seams)} new frame(s) of the load" if seams is not None else f"windows of {w} frames: the planner's")
    return window, {"what": "the seam", "carries": carries,
                    "from": f"{by}, {now}: {' '.join(str(seam_why).split())}" if seam is not None else "the planner (window_for: the longest window)"}


def context_noise_of(n: int, noise: float | None, why: str | None, by: str, now: str) -> dict | None:
    """THE NOISE A LOAD'S LATER WINDOWS ARE SHOWN THEIR KEPT FRAMES AT (the lead's ruling of 2026-10-15, per load): `--context-noise X
    --context-noise-why WHY`, for a load longer than one window. By default its later windows see the frames they keep from the window
    before clean (the song node's `audio_freeze.CONTEXT_NOISE`), and on one stretch of the kitchen a continuation shown them clean did not
    follow its motion video through a turn where one at 0.8 did (`docs/wiki/window_context.md`, "Results, dated", 2026-10-10, the records
    `data/2026-10-10_fun_pose/kitchen_cn_*`). Per load, not a default,
    until a render of a load of several windows says more: its cost is a seam that may show. (the defaults row, or None when not given)."""
    import h3_config
    import masked_job
    if (noise is None) != (not str(why or "").strip()):
        raise Refused("--context-noise and --context-noise-why go together: the noise, and a person's reason")
    if noise is None:
        return None
    if not 0.0 < float(noise) <= 1.0:
        raise Refused(f"--context-noise {noise}: above 0 (the kept frames shown clean, the node's own) and at most 1 (redrawn whole)")
    if n <= max(masked_job.chain_lengths()):
        raise Refused(f"--context-noise {noise}: one window holds this load's {n} frames, so no window of it keeps frames")
    return {"what": "the kept frames' noise", "carries": f"{float(noise):g}: each later window is shown the frames it keeps from the one before "
            "at that noise and redraws them by that share, so it follows its own motion video more and the window before less; the "
            "cost, a first new frame that may differ from the last frame written before it",
            "from": f"{by}, {now}: {' '.join(str(why).split())}"}


#: Room kept under the song node's greatest seed for the windows of one load: it samples window `i` at `seed + i`
#: (`audio_freeze_song.py`, `prepare_noise(..., int(seed) + i)`), and torch refuses a seed over 2**64 - 1 (the ComfyUI venv's
#: `torch.manual_seed(2**64)` raised ValueError, 2026-10-22), so a seed at the node's greatest would fail on window 1, mid-render.
#: **Reasoned**: no load has this many windows.
SEED_ROOM = 1024
#: The greatest seed a job takes. **Inherited** from the song node's `seed` input (`audio_freeze_song.py`: `max=0xffffffffffffffff`),
#: less `SEED_ROOM`.
SEED_MAX = 0xffffffffffffffff - SEED_ROOM


def job_seed_of(seed: int | str | None, seed_why: str | None, by: str, now: str) -> tuple[int | None, dict | None]:
    """THE SEED OF ONE LOAD (`--seed N --seed-why WHY`): (the seed, its record for `defaults.seed_by`), or (None, None) when neither
    is given. A seed is per load and a person's: the job's own is `h3_config.SEED` and nothing but this changes it, so the record says
    who asked and why, and `--carry-words-from` never carries it (`carry_words_of`). Refused: a seed that is not a whole number from 0 to
    `SEED_MAX`, a `--seed` with no words, and words with no `--seed`."""
    if seed is None:
        if seed_why is not None:
            raise Refused("--seed-why goes with --seed: there is no seed to give a reason for")
        return None, None
    given = seed.strip() if isinstance(seed, str) else seed
    digits = isinstance(given, str) and given.isascii() and given.isdigit() and len(given) <= len(str(SEED_MAX))
    whole = digits or (isinstance(given, int) and not isinstance(given, bool))
    if not whole or not 0 <= int(given) <= SEED_MAX:
        raise Refused(f"--seed {seed!r}: a seed is a whole number from 0 to {SEED_MAX} (the song node's greatest, less room for each window's `seed + i`)")
    words = " ".join(str(seed_why or "").split())
    if not words:
        raise Refused("--seed takes --seed-why: who asked for this seed and why (the job's own is h3_config.SEED; a seed is per load and a person's word)")
    return int(given), {"why": words, "by": by, "at": now}


#: The most a load may ask the Masked Source to enlarge it (`--zoom-up-to`): `video_mask.ZOOM_MOST_ASKED`, said again
#: here because this tool does not import the pack (torch); `bench/check_shot_to_load.py` holds the two equal. And the
#: node's own, `video_mask.ZOOM_LARGEST`, which a load asks past.
ZOOM_UP_TO_MOST, ZOOM_NODES_OWN = 4.0, 2.0


def zoom_up_to_of(up_to: float | str | None, up_to_why: str | None, by: str, now: str) -> tuple[float | None, dict | None]:
    """THE ZOOM A LOAD ASKS FOR (`--zoom-up-to N --zoom-up-to-why WHY`, 2026-10-22): (the most the node may enlarge the load, its
    record for `defaults.zoom_up_to_by`), or (None, None) when neither is given. Off by default (the owner's word: the zoom to a
    target head size lands switched off and is tested on a load first), so the record says who asked and why. Refused: a number
    not above the node's own 2 or past `ZOOM_UP_TO_MOST`, a `--zoom-up-to` with no words, and words with no `--zoom-up-to`."""
    if up_to is None:
        if up_to_why is not None:
            raise Refused("--zoom-up-to-why goes with --zoom-up-to: there is no zoom to give a reason for")
        return None, None
    try:
        value = float(up_to)
    except (TypeError, ValueError):
        raise Refused(f"--zoom-up-to {up_to!r}: a number, the most the node may enlarge the load") from None
    if not ZOOM_NODES_OWN < value <= ZOOM_UP_TO_MOST:
        raise Refused(f"--zoom-up-to {value:g}: above the node's own {ZOOM_NODES_OWN:g} and at most {ZOOM_UP_TO_MOST:g}")
    words = " ".join(str(up_to_why or "").split())
    if not words:
        raise Refused("--zoom-up-to takes --zoom-up-to-why: who asked for the larger zoom and why (it is off by default, for a test)")
    return value, {"why": words, "by": by, "at": now}


#: The most a load may ask the Masked Source to grow its region by (`--margin`): the node's `grow_pixels` takes no more
#: (`video_mask.GROW_PIXELS_MAX`). **Inherited** from the node, said again here because this tool does not import the pack (torch);
#: `bench/check_shot_to_load.py` holds the two equal.
MARGIN_MOST = 512


def margin_of(margin: int | str | None, margin_why: str | None, by: str, now: str) -> tuple[int | None, dict | None]:
    """THE MARGIN OF ONE LOAD (`--margin N --margin-why WHY`, 2026-10-23, the owner's go for a larger mask round the second subject's
    hair on 1093): (the px the node grows this load's region by, its record for `defaults.margin_by`), or (None, None) when neither is
    given, and the load takes `h3_config.MASKED_LOAD_MARGIN` by the plan's region. Per load and a person's, as the seed and the zoom are:
    `--carry-words-from` never carries it (`carry_words_of`). Refused: a margin that is not a whole number from 1 to `MARGIN_MOST`, a
    `--margin` with no words, and words with no `--margin`. A margin under the load's feather is the builder's to refuse
    (`masked_job_graph.load`), in its words, before the file is written (`load_held_as_the_builder_holds`)."""
    if margin is None:
        if margin_why is not None:
            raise Refused("--margin-why goes with --margin: there is no margin to give a reason for")
        return None, None
    given = margin.strip() if isinstance(margin, str) else margin
    digits = isinstance(given, str) and given.isascii() and given.isdigit() and len(given) <= len(str(MARGIN_MOST))
    whole = digits or (isinstance(given, int) and not isinstance(given, bool))
    if not whole or not 1 <= int(given) <= MARGIN_MOST:
        raise Refused(f"--margin {margin!r}: a margin is a whole number of px from 1 to {MARGIN_MOST} (the Masked Source's greatest grow_pixels)")
    words = " ".join(str(margin_why or "").split())
    if not words:
        raise Refused("--margin takes --margin-why: who asked for this margin and why (the load's own is h3_config.MASKED_LOAD_MARGIN by its "
                      "region; a margin is per load and a person's word)")
    return int(given), {"why": words, "by": by, "at": now}


def seams_arg(text: str | int | None) -> list[int] | None:
    """`--seam A[,B...]` as source frames; None when not given."""
    if text is None:
        return None
    parts = str(text).split(",")
    if not all(x.strip().isdecimal() for x in parts):
        raise Refused(f"--seam {text!r}: source frames, comma-separated (--seam 898 or --seam 2344,2548)")
    return [int(x) for x in parts]


def no_mouth_on_arg(text: str, flag: str = "--no-mouth-on") -> tuple[int, int, str]:
    """`--no-mouth-on FIRST-LAST=WHY` (or `FRAME=WHY`) as (first, last, why); `flag` names it in a refusal."""
    rng, sep, why = text.partition("=")
    try:
        if len(rng.split("-")) > 2:
            raise ValueError
        x, y = (int(v) for v in (rng.split("-") * 2)[:2])
    except ValueError:
        raise Refused(f"{flag} {text!r} is not FIRST-LAST=WHY") from None
    if not sep or not why.strip() or x > y:
        raise Refused(f"{flag} {text!r} is not FIRST-LAST=WHY, with the why")
    return x, y, " ".join(why.split())


def hold_part_arg(text: str) -> tuple:
    """`--hold-part FIRST-LAST=WHY` (or `FRAME=WHY`) as (first, last, why), as `no_mouth_on_arg` reads it; `FIRST-LAST@FRAME=WHY` as
    (first, last, why, frame): the frame the hold is taken from, its graded part placed where it was (`capture_masked_run.HELD_IN_PLACE`)."""
    rng, sep, why = text.partition("=")
    span, at, src = rng.partition("@")
    if not at:
        return no_mouth_on_arg(text, "--hold-part")
    if not src.strip().isdecimal():
        raise Refused(f"--hold-part {text!r} is not FIRST-LAST@FRAME=WHY: the frame after @ is the source frame the part is held from")
    return (*no_mouth_on_arg(f"{span}{sep}{why}", "--hold-part"), int(src))


def hold_spans_said(word: str) -> str:
    """A subject's `word` as the job's note and its defaults say it: `110-111, 120-120`, a span held in place as `5039-5039 held in
    place from 5038`."""
    return word.replace("+", ", ").replace("@", " held in place from ")


def parts_on_arg(text: str) -> tuple[int, int, list[str]]:
    """`FIRST-LAST=NAME[+NAME...]` of `--parts-on`: source frames, and the parts the load's part is on them."""
    span, sep, names = text.partition("=")
    try:
        a, b = (int(v) for v in span.split("-"))
    except ValueError:
        raise Refused(f"--parts-on {text!r} is not FIRST-LAST=NAME[+NAME]") from None
    if not sep or a > b or not names.strip():
        raise Refused(f"--parts-on {text!r} is not FIRST-LAST=NAME[+NAME]")
    return a, b, [k for k in names.split("+") if k]


def part_of(p: dict, subject: str, parts, parts_by, preview: dict, video: str, output: Path, preview_files, classes_of=part_classes,
            parts_on: list[tuple[int, int, list[str]]] | None = None) -> tuple[dict | None, str]:
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
    and a preview that saved no class map or no part of the subject.

    A PART THAT CHANGES INSIDE ONE LOAD (`parts_on`, `--parts-on FIRST-LAST=NAME[+NAME]`, 2026-10-14): on those
    source frames the part is those parts, and on the load's other frames it is `--parts` or the plan's. For the
    second subject on 1010-1031: face and hair to 1024, her hair alone on 1025-1031 where her head is bowed into her
    hand, one load and one reference (a load of 1025-1031 alone was under the builder's twelve frames, and a load
    ending on 1024 put her own hair back inside the shot). The file is cut a frame at a time, each frame with its own
    classes, and held to the saved part as a whole. Refused besides: a range's parts not some of the plan's, named
    twice, the same as the load's own elsewhere, or ranges that overlap; whether each lies inside the load is the
    job's to hold (`load_job`)."""
    parts_on = list(parts_on or [])
    if not parts and not parts_on and not str(parts_by or "").strip():
        return None, ""
    by = " ".join(str(parts_by or "").split())
    if not parts and not parts_on:
        raise Refused(f"--parts-by {parts_by!r} with no --parts and no --parts-on: the parts are some of the plan's for {subject!r} ({', '.join(p['parts'])})")
    names = list(parts or p["parts"])
    if not names or len(set(names)) != len(names) or any(k not in p["parts"] for k in names) or not by:
        raise Refused(f"--parts {names!r} with --parts-by {parts_by!r}: the parts are some of the plan's for {subject!r} ({', '.join(p['parts'])}), each once, "
                      "and --parts-by says who chose them for this load and on what")
    if parts and set(names) == set(p["parts"]):
        raise Refused(f"--parts names every part the plan has for {subject!r} ({', '.join(p['parts'])}): that is the load as it is; leave --parts out")
    names = [k for k in p["parts"] if k in names]                 # the plan's order
    ons = []
    for a, b, these in sorted(parts_on):
        if not these or len(set(these)) != len(these) or any(k not in p["parts"] for k in these):
            raise Refused(f"--parts-on {a}-{b}={'+'.join(these)}: the parts are some of the plan's for {subject!r} ({', '.join(p['parts'])}), each once")
        these = [k for k in p["parts"] if k in these]
        if these == names:
            raise Refused(f"--parts-on {a}-{b}={'+'.join(these)} is the load's own part on its other frames: leave that range out")
        if ons and a <= ons[-1]["last"]:
            raise Refused(f"--parts-on {a}-{b} overlaps {ons[-1]['first']}-{ons[-1]['last']}: one part a frame")
        ons.append({"first": a, "last": b, "parts": these})
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
    on_words = "".join(f"_{x['first']}-{x['last']}_{'_'.join(x['parts'])}" for x in ons)
    file = output / PARTS_AT.format(video=video, preview=preview["name"]) / f"{preview['name']}_{subject}_part_{'_'.join(names)}{on_words}.mkv"
    block = {"of": subject, "by": by, "preview": preview["name"], "first": preview["first"], "parts": names, "the_plans_parts": list(p["parts"]),
             **({"parts_on": ons} if ons else {}),
             "made": [{"name": "part", "track": str(file), "classes": classes, "cut_from": str(saved("classes")), "inside": str(saved("parts")),
                       **({"classes_on": [{"first": x["first"], "last": x["last"], "classes": [c for k in x["parts"] for c in table[k]]} for x in ons]} if ons else {}),
                       "what": "those classes of the preview's class map of the subject" + (", a frame at a time where a range says other parts" if ons else "")
                               + "; every pixel of it is in the preview's saved part"}]}
    own = (f"THIS LOAD'S PART IS {' + '.join(names).upper()} ALONE, not the plan's {' + '.join(p['parts'])}" if names != list(p["parts"])
           else f"THIS LOAD'S PART IS THE PLAN'S, {' + '.join(names).upper()}")
    on_said = "".join(f"; ON {x['first']}-{x['last']} IT IS {' + '.join(x['parts']).upper()} ALONE" for x in ons)
    said = (f" {own}{on_said}, on {by}. It is cut from the preview's class map ({' + '.join(classes)}" + (", a frame at a time" if ons else "")
            + f") as a keep-out is, to {file}, and the chain hands the capture that file as the load's part: what the builder loads is what "
            "the capture read. The load's region, its margin and its reference are the plan's, unchanged.")
    return block, said


def track_not_arg(text: str) -> tuple[str, int, int, str]:
    """`SUBJECT:FIRST-LAST=WHY` of `--track-not`."""
    head, sep, why = text.partition("=")
    label, _, span = head.partition(":")
    try:
        a, b = (int(v) for v in span.split("-"))
    except ValueError:
        raise Refused(f"--track-not {text!r} is not SUBJECT:FIRST-LAST=WHY") from None
    if not sep or not label.strip() or not why.strip() or a > b:
        raise Refused(f"--track-not {text!r} is not SUBJECT:FIRST-LAST=WHY, with the reason in words")
    return label.strip(), a, b, " ".join(why.split())


def track_not_of(subject: str, preview: dict, words: list[tuple[str, int, int, str]], by: str | None, load_span: tuple[int, int], video: str,
                 output: Path, preview_files) -> tuple[dict | None, str]:
    """A person's word that another subject's preview track is not that person on named frames, as the job holds it:
    one copy of that subject's track a subject, with those frames emptied, which every step of the load reads for that
    subject (`load_chain.preview_files`).

    2026-10-13: in the preview of 604-1009 the lead's track came back on 1003-1009 lying almost wholly inside the
    second subject's, though the lead had left the picture; a load of the second subject there was blocked for a
    region over another subject, and taking the lead out of its others failed the capture's verify (mrpen's notes on
    second_hfp_0998b). With the lead kept as an other and her track empty there, the capture reads an other's empty
    track in a taken shot as data (222a9e3b) and nothing of hers lies on the second subject. Refused: the load's own
    subject (a load is cut where its own person is not: --first, --frames), a subject the preview did not track,
    frames outside the load, no reason, no `--track-not-by`."""
    if not words:
        return None, ""
    by = " ".join(str(by or "").split())
    if not by:
        raise Refused("--track-not is a person's word: say who says it, and on what, with --track-not-by")
    order = [s["label"] for s in preview["subjects"]]
    if preview_files is None:
        raise Refused("--track-not is given and the preview's saved files were not handed in: the copy is made from the preview's own track")
    made: dict[str, dict] = {}
    for label, a, b, why in words:
        if label == subject:
            raise Refused(f"--track-not {label}:{a}-{b}: {label} is the load's own subject; a load is cut where its own person is not (--first, --frames)")
        if label not in order:
            raise Refused(f"--track-not {label}:{a}-{b}: {label!r} is not a subject of the preview {preview['name']} ({', '.join(order)})")
        if not (load_span[0] <= a <= b <= load_span[1]):
            raise Refused(f"--track-not {label}:{a}-{b} is not inside the load's frames {load_span[0]}-{load_span[1]}")
        one = made.setdefault(label, {"subject": label, "frames": [], "why": []})
        if any(x <= b and a <= y for x, y in one["frames"]):
            raise Refused(f"--track-not {label}:{a}-{b} overlaps another range of {label}'s")
        one["frames"].append([a, b])
        one["why"].append(f"{a}-{b}: {why}")
    for label, one in made.items():
        found = [f for f in preview_files if f"_{label}_track_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {preview['name']} saved {len(found)} track video(s) of {label!r} where one is meant")
        one["frames"].sort()
        spans = "_".join(f"{a}-{b}" for a, b in one["frames"])
        one.update(cut_from=str(output / found[0]), first=preview["first"],
                   track=str(output / TRACKS_AT.format(video=video, preview=preview["name"]) / f"{preview['name']}_{label}_track_not_{spans}.mkv"))
    block = {"by": by, "preview": preview["name"], "made": list(made.values())}
    said = "".join(f" {x['subject'].upper()}'S TRACK IS NOT {x['subject'].upper()} ON {', '.join(f'{a}-{b}' for a, b in x['frames'])}, on {by}: "
                   + "; ".join(x["why"]) + f". Every step reads the copy {x['track']}, those frames emptied; {x['subject']} stays an other of the load."
                   for x in made.values())
    return block, said


def cut_track(track: Path, frames: list[list[int]], first: int, out: Path) -> dict:
    """Write a track's copy with the named source frames emptied (its frame 0 is source frame `first`). Figures."""
    import capture_masked_run as cap
    w, h, n = cap.probe(str(track))
    mask = cap.read_mask(str(track), (w, h), 0, 0, n)[0]
    gone = 0
    for a, b in frames:
        at = slice(max(a - first, 0), max(min(b - first + 1, n), 0))
        gone += int(mask[at].sum())
        mask[at] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    cap.write_mask_video(out, mask)
    return {"frames": int(n), "px_emptied": gone}


def make_track_not(block: dict | None, world, cut=cut_track, hashed=None) -> list[str]:
    """Make the track copies a job names, as a keep-out is made: a copy that is there is held to a fresh one, frame by frame."""
    hashed = hashed or load_chain.frame_hash
    lines = []
    for t in (block or {}).get("made", []):
        out = Path(t["track"])
        if world.dry:
            lines.append(f"{t['subject']}'s track, not them on {', '.join(f'{a}-{b}' for a, b in t['frames'])}: WOULD BE COPIED from {t['cut_from']} to {out}"
                         + (" (a file is there: held to a fresh copy)" if out.exists() else ""))
            continue
        if out.exists():
            again = out.with_name(out.stem + ".cut_again.mkv")
            cut(Path(t["cut_from"]), t["frames"], t["first"], again)
            same = hashed(str(again)) == hashed(str(out))
            again.unlink()
            if not same:
                raise Refused(f"{out} is there and is NOT the copy of {t['cut_from']} with those frames emptied: of another making. It is read and moved aside by a person")
            t["hash"] = hashed(str(out))           # held by the chain at every capture, as a keep-out's frames are (`load_chain.mask_specs`)
            t["cut_from_hash"] = hashed(t["cut_from"])  # and the track it was cut from: a preview rewritten under the same name is caught
            lines.append(f"{t['subject']}'s track, not them on {', '.join(f'{a}-{b}' for a, b in t['frames'])}: {out} was there, and a copy made again decodes equal")
            continue
        got = cut(Path(t["cut_from"]), t["frames"], t["first"], out)
        t["hash"], t["cut_from_hash"] = hashed(str(out)), hashed(t["cut_from"])
        lines.append(f"{t['subject']}'s track, not them on {', '.join(f'{a}-{b}' for a, b in t['frames'])}: copied to {out} with {got['px_emptied']} px emptied")
    return lines


def track_over_track(preview: dict, preview_files, output: Path, subject: str, first: int, last: int) -> list[str]:
    """A figure a frame, printed and decided on by nobody here: on the load's frames, the share of each other subject's
    preview track that lies inside the load's subject's track, as runs of frames over `TRACK_INSIDE`, with the most."""
    import capture_masked_run as cap
    tracks = {s["label"]: next((output / f for f in preview_files or [] if f"_{s['label']}_track_" in Path(f).name and f.endswith(".mkv")), None)
              for s in preview["subjects"]}
    mine = tracks.get(subject)
    if mine is None or not mine.is_file():
        return [f"track over track: not read (the preview's track of {subject} is not there)"]
    try:
        w, h, n = cap.probe(str(mine))
    except Exception:                      # an empty or broken file: said, not raised; the capture reads it later and fails there
        return [f"track over track: not read ({mine.name} does not decode)"]
    count = last - first + 1
    me = cap.read_mask(str(mine), (w, h), preview["first"], first, count)[0]
    lines = []
    for label, path in tracks.items():
        if label == subject or path is None or not path.is_file():
            continue
        other = cap.read_mask(str(path), (w, h), preview["first"], first, count)[0]
        px = other.reshape(count, -1).sum(1)
        inside = (other & me).reshape(count, -1).sum(1) / px.clip(min=1)
        over = [first + i for i in range(count) if px[i] and inside[i] > TRACK_INSIDE]
        if over:
            lines.append(f"track over track: {label}'s track lies more than {TRACK_INSIDE:g} inside {subject}'s on {len(over)} frame(s), "
                         + ", ".join(f"{a}-{b}" for a, b in cap.frame_spans(over, join=1)) + f" (most {float(inside.max()):.2f}): a figure. Where that track "
                         f"is not {label}, a person says so: --track-not {label}:FIRST-LAST=WHY")
    return lines or [f"track over track: no other subject's track lies more than {TRACK_INSIDE:g} inside {subject}'s on any frame of the load"]


def part_inside(cut: Path, whole: Path) -> int:
    """The pixels of a cut part that are NOT in the preview's saved part, over every frame: none, for a part of it."""
    import capture_masked_run as cap
    w, h, n = cap.probe(str(whole))
    a, b = cap.read_mask(str(cut), (w, h), 0, 0, n)[0], cap.read_mask(str(whole), (w, h), 0, 0, n)[0]
    return int((a & ~b).sum())


def cut_keep_out(class_map: Path, classes: list[str], out: Path, classes_on: list[dict] | None = None, first: int = 0) -> dict:
    """Write one keep-out: the pixels of a saved class map that carry any of `classes`, as a mask video on the class
    map's own frames. With `classes_on` ({first, last, classes} in source frames; the class map's frame 0 is source
    frame `first`) a frame in a range carries that range's classes instead. Figures of what was written."""
    import numpy as np
    import capture_masked_run as cap
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    w, h, n = cap.probe(str(class_map))
    read = cap.read_classes(str(class_map), (w, h), 0, 0, n)
    mask = np.isin(read, [names.index(c) for c in classes])
    for x in classes_on or []:
        at = slice(max(x["first"] - first, 0), max(min(x["last"] - first + 1, n), 0))
        mask[at] = np.isin(read[at], [names.index(c) for c in x["classes"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    cap.write_mask_video(out, mask)
    px = mask.reshape(n, -1).sum(1)
    return {"frames": int(n), "size": [int(w), int(h)], "frames_with_the_mask": int((px > 0).sum()), "px_most": int(px.max()) if n else 0}


def part_on_track(p: dict, subject: str, part: dict | None, preview: dict, video: str, output: Path, preview_files) -> tuple[dict | None, str]:
    """Where the plan names the `strangers` keep-out for a subject, the load's part held to its subject's own preview
    track grown by `h3_config.MASKED_STRANGERS["grow_px"]`, as the job file holds it, and the note's sentence; nothing
    otherwise. The part held is the load's own (`part_of`) where it has one, else the preview's saved part.

    Why (2026-10-14, ruled by the lead as V3): on the lead's 3167-3213 her part lay outside her own track on 45 of
    47 frames, on another woman's face, and a cell holding her part is never given back to a keep-out, so the
    strangers keep-out could not take it back. The part model's pixels outside her track are no part of her. What
    the hold keeps is so by its making (the part inside the grown track); what the grow was measured on is in
    `h3_config.MASKED_STRANGERS`. A frame where her track is empty is held to the nearest frame with one within
    `hold_frames` (`strangers.held_track`), so a tracker's moment's loss does not empty her part; a track empty on
    every frame, or not of the preview's canvas and length, is refused before anything of the load is written."""
    import h3_config
    off = p.get("strangers_turned_off")
    if "strangers" not in (p.get("keep_outs") or []) and not off:
        return None, ""

    def saved(kind: str) -> Path:
        found = [f for f in preview_files or [] if f"_{subject}_{kind}_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {preview['name']} saved {len(found)} `{kind}` video(s) of {subject!r} where one is meant: the strangers keep-out holds "
                          "the load's part to its subject's track")
        return output / found[0]
    source = Path(part["made"][0]["track"]) if part else saved("parts")
    px = int(h3_config.MASKED_STRANGERS["grow_px"])
    stem = source.stem if part else f"{preview['name']}_{subject}_part_saved"
    file = output / PARTS_AT.format(video=video, preview=preview["name"]) / f"{stem}_on_track_{px}px.mkv"
    block = {"of": subject, "preview": preview["name"], "first": preview["first"], "frames": preview["frames"], "canvas": list(preview["canvas"]),
             "part": str(source), "track": str(saved("track")), "grow_px": px,
             "file": str(file), "why": ("the strangers keep-out is turned off on a person's word, and the part is held to the subject's own track all "
                                        "the same (the lead, 2026-10-15)" if off else "the plan names the strangers keep-out") +
             ": the part model's pixels outside the subject's own track are no part of them"}
    said = (f" THE PART IS HELD TO {subject}'s OWN TRACK grown {px} px (h3_config.MASKED_STRANGERS), "
            + ("though the strangers keep-out is turned off on a person's word" if off else "since the plan keeps the strangers out")
            + f": written to {file}, and the chain hands the capture that file as the load's part.")
    return block, said


def track_refusals(track: Path, canvas, frames: int) -> list[str]:
    """Why a subject's preview track cannot hold a part: not there, not of the canvas, shorter than the preview, or empty on
    every frame. Read before anything of the load is written."""
    import capture_masked_run as cap
    if not track.is_file():
        return [f"the track {track} is not there"]
    tw, th, tn = cap.probe(str(track))
    if [tw, th] != list(canvas) or tn < frames:
        return [f"the track {track.name} is {tw}x{th} with {tn} frames, and the preview's canvas is {canvas[0]}x{canvas[1]} on {frames} frames"]
    if not cap.read_mask(str(track), (tw, th), 0, 0, frames)[0].any():
        return [f"the track {track.name} is empty on every frame: the tracker never held the subject, and their part would be held to nothing"]
    return []


def hold_to_track(part: Path, track: Path, px: int, out: Path, first: int = 0) -> dict:
    """Write a part held to a track grown by `px`, a frame at a time, on the part's own frames (frame 0 is source frame
    `first`); figures of what was written, its frames in source frames."""
    import numpy as np
    import capture_masked_run as cap
    import strangers
    w, h, n = cap.probe(str(part))
    tw, th, tn = cap.probe(str(track))
    if (tw, th) != (w, h) or tn < n:
        raise Refused(f"the track {track.name} is {tw}x{th} with {tn} frames and the part {part.name} {w}x{h} with {n}: they are not of one preview")
    a = cap.read_mask(str(part), (w, h), 0, 0, n)[0]
    kept, gone, empty = strangers.held_track(cap.read_mask(str(track), (w, h), 0, 0, n)[0])
    if len(empty) == n:
        raise Refused(f"the track {track.name} is empty on every frame: the part would be held to nothing")
    held = a & strangers.grown(kept, px)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap.write_mask_video(out, held)
    off = (a & ~held).reshape(n, -1).sum(1)
    return {"frames": int(n), "size": [int(w), int(h)], "px_taken_off_a_frame_median": int(np.median(off)) if n else 0, "px_taken_off_most": int(off.max()) if n else 0,
            "frames_with_any_taken_off": int((off > 0).sum()), "frames_the_track_is_empty_and_held": [first + int(i) for i in gone],
            "frames_the_track_is_empty_and_left_empty": [first + int(i) for i in empty]}


def make_part_on_track(block: dict | None, world, hold=hold_to_track, hashed=None) -> list[str]:
    """Make the held part a job file names, and say what was done. A file that is there is held to a fresh making."""
    if not block:
        return []
    hashed = hashed or load_chain.frame_hash
    file = Path(block["file"])
    if world.dry:
        return [f"the part held to the track: WOULD BE WRITTEN from {block['part']} and {block['track']} grown {block['grow_px']} px to {file}"
                + (" (a file is there: held to a fresh making)" if file.exists() else "")]
    if file.exists():
        again = file.with_name(file.stem + ".made_again.mkv")
        hold(Path(block["part"]), Path(block["track"]), block["grow_px"], again, first=block["first"])
        same = hashed(str(again)) == hashed(str(file))
        again.unlink()
        if not same:
            raise Refused(f"{file} is there and is NOT the part held to the track made again now: a file of another making. It is read and moved aside by a person")
        return [f"the part held to the track: {file} was there, and made again now it decodes equal on every frame"]
    got = hold(Path(block["part"]), Path(block["track"]), block["grow_px"], file, first=block["first"])
    file.with_suffix(".json").write_text(json.dumps({"what": "a load's part held to its subject's preview track grown, by bench/shot_to_load.py", **block,
                                                     "made_at": load_chain.utc_now(), **got}, indent=1) + "\n")
    return [f"the part held to the track: {file}: {got['frames']} frames; taken off the part {got['px_taken_off_a_frame_median']} px a frame at the median, "
            f"{got['px_taken_off_most']} at most, on {got['frames_with_any_taken_off']} frames"]

def make_part(part: dict | None, world, cut=cut_keep_out, hashed=None, inside=part_inside) -> list[str]:
    """Make the part a load's job file names as its own, a subset of its subject's parts, and say what was done.

    As a keep-out is made: cut from the preview's class map; a file that is there is not trusted because it is
    there, a cut made again beside it must decode equal. AND HELD TO THE PREVIEW'S SAVED PART: a pixel of the cut
    that is not in the saved part is no part of it, and the load is refused with the count."""
    hashed = hashed or load_chain.frame_hash
    lines = []
    for k in (part or {}).get("made", []):
        file = Path(k["track"])
        on = {"classes_on": k["classes_on"], "first": part["first"]} if k.get("classes_on") else {}
        if world.dry:
            lines.append(f"the load's own part ({' + '.join(part['parts'])}): WOULD BE CUT from {k['cut_from']} ({' + '.join(k['classes'])}) to {file}"
                         + (" (a file is there: held to a fresh cut)" if file.exists() else "") + f", and held to the preview's saved part {k['inside']}")
            continue
        if file.exists():
            again = file.with_name(file.stem + ".cut_again.mkv")
            cut(Path(k["cut_from"]), k["classes"], again, **on)
            same = hashed(str(again)) == hashed(str(file))
            again.unlink()
            if not same:
                raise Refused(f"{file} is there and is NOT what its classes cut from {k['cut_from']} now: a part of another making. It is read and moved aside by a person")
            lines.append(f"the load's own part ({' + '.join(part['parts'])}): {file} was there, and a cut made again now decodes equal to it on every frame")
            continue
        got = cut(Path(k["cut_from"]), k["classes"], file, **on)
        outside = inside(file, Path(k["inside"]))
        if outside:
            file.unlink()
            raise Refused(f"the cut of {' + '.join(part['parts'])} from {k['cut_from']} has {outside} px that are NOT in the preview's saved part {k['inside']}: it is no "
                          "part of what the preview saved, so the class map and the part are not of one making. Nothing is kept")
        file.with_suffix(".json").write_text(json.dumps({"what": "a load's own part, some of its subject's parts, cut by bench/shot_to_load.py from a preview's class map",
                                                        "of": part["of"], "parts": part["parts"], "the_plans_parts": part["the_plans_parts"], "by": part["by"],
                                                        **({"parts_on": part["parts_on"], "classes_on": k["classes_on"]} if k.get("classes_on") else {}),
                                                        "classes": k["classes"], "cut_from": k["cut_from"], "held_to": k["inside"], "px_outside_the_saved_part": 0,
                                                        "preview": part["preview"], "first_source_frame": part["first"], "made_at": load_chain.utc_now(), **got}, indent=1) + "\n")
        lines.append(f"the load's own part ({' + '.join(part['parts'])}): cut {' + '.join(k['classes'])} from {k['cut_from']} to {file}: {got['frames']} frames, on the "
                     f"mask on {got['frames_with_the_mask']}, up to {got['px_most']} px; every pixel of it is in the preview's saved part")
    return lines


def looked_for(kept: dict | None, world) -> list[str]:
    """The keep-outs a job names that are looked for and not cut (`held`, `strangers`), held before anything of the load
    is written: each is there, and a strangers file to its record (`strangers.held_to_its_record`). Refused otherwise,
    with nothing written; a dry run says what it would refuse."""
    import strangers
    lines = []
    for k in (kept or {}).get("made", []):
        if "classes" in k or "stay" in k:          # made by this step (make_keep_outs, make_stay), not looked for
            continue
        track = Path(k["track"])
        why = [] if track.is_file() else [f"{track} is not there" + (f": `{k['made_by']} --preview <the preview's job folder>` makes it, on the card"
                                                                         if k.get("made_by") else "")]
        if not why and k["name"] == "strangers":
            why = strangers.held_to_its_record(track, kept["preview"], kept["first"], k["frames"], k["tracks"])
            if not why:                         # the chain holds the capture's copy to the same frames (load_chain.mask_specs)
                r = json.loads(track.with_suffix(".json").read_text())
                k["frame_hash"] = r["frame_hash"]
                lines.append(f"keep-out `strangers`, on the preview's frames: {r.get('handed_back_said', 'its record lists no piece handed back')} (bench/strangers.py `handed_back`: "
                             "a piece touching a subject's own with no body core of its own, whose only core is her hair meeting hers, or on no "
                             "other person the detector found; a look sees each in the record)")
        if why and not world.dry:
            raise Refused(f"the keep-out `{k['name']}`: {'; '.join(why)}. Nothing is written")
        lines += [f"keep-out `{k['name']}`: WOULD BE REFUSED: {'; '.join(why)}"] if why else []
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
        if "stay" in k:
            continue
        if "classes" not in k:
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


def others_of_load(preview: dict, output: Path, shot: dict, cast: list[dict] | None, cast_from: str, not_people: list[dict] | None = None) -> dict:
    """Who else is in the load's shot (`others_present.shot_others`), from its preview's tables in the folder its job wrote to."""
    import others_present
    folder = output / preview["output"] / preview["name"]
    tables = {}
    for s in preview["subjects"]:
        found = sorted(folder.glob(f"{preview['name']}_{s['label']}_shots_*_shots.json"))
        if found:
            tables[s["label"]] = (found[-1].name, json.loads(found[-1].read_text()))
    pv = {"name": preview["name"], "first": int(preview["first"]), "last": int(preview["first"]) + int(preview["frames"]) - 1, "tables": tables}
    row = others_present.shot_others({"first": int(shot["first"]), "last": int(shot["last"]), "note": shot.get("note")}, pv if tables else None, cast or [],
                                     None, not_people)
    if not tables:
        row = {"frames": [int(shot["first"]), int(shot["last"])], "others_present": "not known", "why": f"the preview {preview['name']} saved no shot table"}
    out = others_present.summary(row)
    out["said"] = others_present.said(row)
    out["owners_answers"] = f"read from {cast_from}" if cast is not None else "not handed in (--cast): each tracker's pick is taken as ours"
    return out


def frames_in_words(words: str) -> set[int]:
    """The frame numbers a person's words name: a number standing alone (not inside a path or a name), or a range "A-B" or "A to B"."""
    import re
    out: set[int] = set()
    for m in re.finditer(r"(?<![\w/.])(\d+)(?:\s*(?:-|\u2013|to)\s*(\d+))?(?![\w/])", words):
        a = int(m.group(1)); b = int(m.group(2) or a)
        out.update(range(a, b + 1) if b >= a and b - a < 10000 else [a])
    return out


def strangers_by_the_data(p: dict, others: dict) -> dict:
    """The plan's subject with the strangers keep-out as the data says: kept out where someone else is in the shot or the
    frames shown cannot say, and on no load where nobody is; the plan's key only where the data is not known."""
    if others["others_present"] not in ("yes", "cannot say", "no"):
        return dict(p, strangers_from=f"the plan, since who else is in the shot is not known ({others['said']})")
    names = [k for k in p.get("keep_outs") or [] if k != "strangers"] + (["strangers"] if others["others_present"] != "no" else [])
    by = " ".join(str(p.get("keep_outs_by") or "").split())
    strangers_by = f"the data: who else is in the shot, {others['said']}"
    out = dict(p, keep_outs=names or None, strangers_from=strangers_by)
    out.pop("keep_outs_by", None)
    if names:
        out["keep_outs_by"] = "; ".join(x for x in (by if [k for k in names if k != "strangers"] else "",
                                                    f"strangers: {strangers_by}" if "strangers" in names else "") if x)
    return out


#: The name a look's stay keep-out is written under in a load's `others` (`stay_of`). Not the name of the keep-out it replaces,
#: so the builder's hair band, which cuts `clothes`, does not cut it again: its box is already out of it.
STAY_NAME = "clothing"


def stay_of(p: dict, subject: str, kept: dict | None, load: dict, preview: dict, on_track: dict | None, part: dict | None, video: str, output: Path,
            preview_files, name: str, region: str) -> tuple[dict | None, str]:
    """A plan look's STAY item made as ONE keep-out (`look.stay[]` with `as: "one keep-out"`), in place of the keep-out it
    `replaces_keep_out`, and the note's sentence. Her track's pixels whose class is one of `classes_on_her_track`, less the
    hair band's box a frame (`masked_job.band_boxes`, the builder's own; the load's band is then not asked of the builder),
    held over frames: in on a frame when in on most of the frames within `hold_frames` either side, the window cut at the
    load's ends. Written on the preview's frames, empty outside the load's; one file a load (the band is the load's).
    The lead's ruling of 2026-10-14 (K, build path (a)), on mrpre's measured apron read: the part model calls the second
    subject's white apron Background, so the clothes keep-out missed it and the held rule gave its strap back cell by cell."""
    items = [x for x in ((p.get("look") or {}).get("stay") or []) if x.get("as") == "one keep-out"]
    if not items:
        return kept, ""
    # RETIRED (the owner chose to redraw her clothing with her: docs/wiki/decisions.md, 26f208c0): a stay keep-out is drawn from her
    # own track's classes, so it puts the original back inside her region, which nothing may (the owner, 2026-10-16)
    raise Refused(f"the plan's look for {subject!r} has a stay item made as one keep-out ({', '.join(items[0].get('classes_on_her_track') or [])} on her own "
                  "track): it is drawn on the subject's own track, and nothing puts the original back inside her region. The owner chose to redraw her "
                  "clothing with her and the stay is retired (docs/wiki/decisions.md, 26f208c0): take `look.stay` out of the plan")
    import h3_config
    import capture_masked_run as cap
    if len(items) > 1:
        raise Refused(f"the plan's look for {subject!r} has {len(items)} stay items made as one keep-out; one a subject is built")
    x = items[0]
    by = " ".join(str(p.get("look_by") or x.get("by") or "").split())
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    gone = x.get("replaces_keep_out")
    classes, hold = x.get("classes_on_her_track") or [], x.get("hold_frames")
    if not by:
        raise Refused(f"the plan's look for {subject!r} names no `look_by`: who ruled it, and on what")
    if not kept or gone not in [k["name"] for k in kept["made"]]:
        raise Refused(f"the plan's stay item for {subject!r} replaces the keep-out {gone!r}, which the plan's `keep_outs` do not name for this load")
    if not classes or any(c not in names for c in classes) or not isinstance(hold, int) or hold < 0:
        raise Refused(f"the plan's stay item for {subject!r}: `classes_on_her_track` {classes!r} are the part model's classes and `hold_frames` {hold!r} a "
                      "whole number of frames either side, 0 or more")

    def saved(kind: str) -> Path:
        found = [f for f in preview_files or [] if f"_{subject}_{kind}_" in Path(f).name and f.endswith(".mkv")]
        if len(found) != 1:
            raise Refused(f"the preview {preview['name']} saved {len(found)} `{kind}` video(s) of {subject!r} where one is meant: the stay keep-out is made from it")
        return output / found[0]
    band = None
    rule = h3_config.MASKED_JOB_HAIR_BAND
    if "hair_band" in load and load.get("hair_band_cuts", rule["cuts"]) == gone:
        margin = load.get("margin", h3_config.MASKED_LOAD_MARGIN[region])
        band = {"share": float(load.pop("hair_band")), "side_px": float(margin) if rule["wider"] == "margin" else None,
                "part": str(Path(on_track["file"]) if on_track else Path(part["made"][0]["track"]) if part else saved("parts")),
                "part_first_source_frame": preview["first"]}
    file = output / KEEP_OUTS_AT.format(video=video, preview=preview["name"]) / f"{name}_{subject}_{STAY_NAME}.mkv"
    entry = {"name": STAY_NAME, "track": str(file), "replaces": gone, "by": by,
             "stay": {"what": x.get("what", ""), "classes_on_her_track": list(classes), "hold_frames": hold, "track": str(saved("track")),
                      "class_map": str(saved("classes")), "preview_first": preview["first"], "preview_frames": preview["frames"],
                      "first": load["first"], "frames": load["frames"], "less_the_hair_band": band},
             "what": f"her clothing as one stay keep-out: {' + '.join(classes)} on her track, less the hair band's box, held over {2 * hold + 1} frames (the plan's look)"}
    kept = dict(kept, made=[entry if k["name"] == gone else k for k in kept["made"]])
    load["others"] = [STAY_NAME if o == gone else o for o in load["others"]]
    said = (f" HER CLOTHING IS ONE STAY KEEP-OUT, `{STAY_NAME}`, in place of `{gone}`, from the plan's look ({by}): {' + '.join(classes)} of her class map on her "
            f"own track, held over {2 * hold + 1} frames (in on a frame when in on most of them)"
            + (f", less the hair band's box a frame ({band['share']:g} of the part's height; the builder's own box, masked_job.band_boxes), so the load asks the "
               f"builder for no band: its box is already out" if band else "") + ".")
    return kept, said


def cut_stay(stay: dict, out: Path) -> dict:
    """Write a stay keep-out (`stay_of`) as a mask video on the preview's frames, and its figures."""
    import numpy as np
    import capture_masked_run as cap
    import masked_job
    from _lib.frames import read_mask, write_mask_video
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    pf, first, n = stay["preview_first"], stay["first"], stay["frames"]
    w, h, _ = cap.probe(stay["class_map"])
    cls = cap.read_classes(stay["class_map"], (w, h), pf, first, n)
    on = read_mask(stay["track"], (w, h), pf, first, n)[0] & np.isin(cls, [names.index(c) for c in stay["classes_on_her_track"]])
    band = stay.get("less_the_hair_band")
    taken = 0
    if band:
        bands, _ = masked_job.band_boxes(band["part"], band["part_first_source_frame"], first, n, (w, h), band["share"], band["side_px"])
        for f in range(n):
            if bands[f, 0] >= 0:
                x0, y0, x1, y1 = bands[f]
                taken += int(on[f, y0:y1, x0:x1].sum())
                on[f, y0:y1, x0:x1] = False
    k = stay["hold_frames"]
    held = np.zeros_like(on)
    for f in range(n):
        window = on[max(f - k, 0):f + k + 1]
        held[f] = window.sum(0) * 2 > len(window)
    whole = np.zeros((stay["preview_frames"], h, w), bool)
    whole[first - pf:first - pf + n] = held
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mask_video(out, whole)
    a = held.reshape(n, -1).sum(1)
    return {"frames": n, "size": [w, h], "px_a_frame_median": int(np.median(a)), "px_most": int(a.max()),
            "px_the_hold_changed": int((held ^ on).sum()), "px_the_band_took_out": taken}


#: Why a stay keep-out empty on every frame is dropped from its load (the lead's ruling of 2026-10-14 on second_hfp_5461b, a close
#: shot whose hair band's box held all of her clothing): an empty mask keeps nothing out, and the capture reads it as a subject never found.
STAY_EMPTY = "the hair band's box takes all {px} px of her clothing on this shot; her clothing inside the margin is regenerated"


def make_stay(kept: dict | None, world, cut=cut_stay, hashed=None, job: dict | None = None) -> list[str]:
    """Make a load's stay keep-out: written, or, when a file of the name is there, made again beside it and held equal to it
    frame by frame (`load_chain.frame_hash`), refused when not. Its hash goes into the job, as the strangers file's does.
    One that is EMPTY ON EVERY FRAME is dropped from the job (`job`: its keep-outs and every load's `others`), kept under
    `keep_outs.dropped` with why (`STAY_EMPTY`), and said in the job's note."""
    hashed = hashed or load_chain.frame_hash
    lines = []
    for k in list((kept or {}).get("made", [])):
        if "stay" not in k:
            continue
        track = Path(k["track"])
        if world.dry:
            lines.append(f"keep-out `{k['name']}`: WOULD BE MADE, {k['what']}, to {track}" + (" (a file is there: held to one made again)" if track.exists() else ""))
            continue
        if track.exists():
            again = track.with_name(track.stem + ".made_again.mkv")
            got = cut(k["stay"], again)
            same = hashed(str(again)) == hashed(str(track))
            again.unlink()
            if not same:
                raise Refused(f"{track} is there and is NOT the stay keep-out its plan makes now: a keep-out of another making. It is read and moved aside by a person")
            lines.append(f"keep-out `{k['name']}`: {track} was there, and one made again now decodes equal to it on every frame")
        else:
            got = cut(k["stay"], track)
            track.with_suffix(".json").write_text(json.dumps({"what": "a look's stay keep-out made by bench/shot_to_load.py", "name": k["name"], "replaces": k["replaces"],
                                                             "by": k["by"], **k["stay"], "made_at": load_chain.utc_now(), **got}, indent=1) + "\n")
            lines.append(f"keep-out `{k['name']}`: made to {track}: {got['frames']} frames, {got['px_a_frame_median']} px a frame at the median and {got['px_most']} at "
                         f"most; the hold changed {got['px_the_hold_changed']} px, the band took out {got['px_the_band_took_out']} (figures of the mask, not of a render)")
        k["frame_hash"] = hashed(str(track))
        if got["px_most"] == 0:
            why = STAY_EMPTY.format(px=got["px_the_band_took_out"]) if got.get("px_the_band_took_out") else "it is empty on every frame of the load"
            kept["made"] = [x for x in kept["made"] if x is not k]
            kept.setdefault("dropped", []).append({"name": k["name"], "track": str(track), "replaces": k.get("replaces"), "why": why,
                                                   "px_the_band_took_out": got.get("px_the_band_took_out", 0)})
            for load in (job or {}).get("loads", []):
                load["others"] = [o for o in load.get("others", []) if o != k["name"]]
            if job is not None:
                job["job"]["note"] = job["job"].get("note", "") + f" THE STAY KEEP-OUT `{k['name']}` IS DROPPED: {why}."
            lines.append(f"keep-out `{k['name']}`: EMPTY ON EVERY FRAME, DROPPED from the load: {why}")
    return lines


def preview_name(shot: dict) -> str:
    return f"shot_c_{shot['first']:04d}"


def name_taken(name: str, folder: str | Path, world) -> bool:
    """Whether a preview, load or pair name is already in use: in `folder`, in any session's `*_jobs` folder under `world.data`,
    or as a capture or job folder of any day."""
    return ((Path(folder) / f"{name}.json").exists()
            or any(world.data.glob(f"*_jobs/{name}.json"))
            or any(world.data.glob(f"????-??-??_{name}"))
            or any(world.data.glob(f"????-??-??_{name}_job")))


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


def _parse_word_by(word: str, word_by: str, old_name: str, subject: str) -> list[tuple[int, int, str, int | None]]:
    """Parse `subjects.<subject>.word` ('A-B+C-D@S') and `word_by` ('A-B: reason; C-D@S: reason') without splitting on `;` inside
    reasons; each span with the frame it is held from in place (`@S`), or None."""
    spans = [s.strip() for s in str(word).split("+") if s.strip()]
    if not spans:
        raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.word is empty")
    text = " ".join(str(word_by or "").split())
    starts: list[tuple[int, int, str]] = []
    pos = 0
    for i, sp in enumerate(spans):
        prefix = f"{sp}: " if i == 0 else f"; {sp}: "
        idx = (0 if text.startswith(prefix) else -1) if i == 0 else text.find(prefix, pos)
        if idx < 0:
            raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.word_by has no reason for span {sp}")
        val_start = idx + len(prefix)
        starts.append((idx, val_start, sp))
        pos = val_start
    out: list[tuple[int, int, str]] = []
    for i, (_idx, val_start, sp) in enumerate(starts):
        val_end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        why = text[val_start:val_end].strip()
        if not why:
            raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.word_by has an empty reason for span {sp}")
        held, at, src = sp.partition("@")
        try:
            x, y = (int(v) for v in (held.split("-") * 2)[:2])
            g = int(src) if at else None
        except ValueError:
            raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.word span {sp!r} is not FIRST-LAST or FIRST-LAST@FRAME") from None
        out.append((x, y, why, g))
    return out


def _author_and_time_from_by(by_str: str, fallback_rec: dict | None = None) -> tuple[str, str]:
    """Extract ``(said_by, at)`` from a ``'<author>, <timestamp>: <reason>'`` string or carried record."""
    head, sep, _ = str(by_str or "").partition(": ")
    if sep and ", " in head:
        who, _, when = head.partition(", ")
        if who.strip() and when.strip():
            return who.strip(), when.strip()
    if isinstance(fallback_rec, dict):
        return str(fallback_rec.get("by") or ""), str(fallback_rec.get("at") or "")
    return "", ""


def _resolve_old_capture_dir(
    cw: dict | None,
    carry_words_from: dict | Path | str | None,
    data: Path | None,
    subject: str,
    capture_folder: Path | str | None = None,
) -> Path | None:
    """Locate the capture directory for the job being carried from, when present on disk."""
    candidates: list[Path] = []
    if capture_folder is not None:
        candidates.append(Path(capture_folder))
    old_cap = str((cw or {}).get("old_capture") or "").strip()
    cap_name = Path(old_cap).name if old_cap else ""
    if cap_name:
        if data is not None:
            candidates.append(Path(data) / cap_name)
        if isinstance(carry_words_from, (str, Path)):
            p = Path(carry_words_from).resolve()
            candidates.append(p.parent / cap_name)
            candidates.append(p.parent.parent / cap_name)
            candidates.append(p.parent.parent / "data" / cap_name)
        if old_cap:
            candidates.append(REPO / old_cap)
    for cand in candidates:
        if (cand / "subjects" / subject).is_dir():
            return cand
    return None


def capture_face_frames(
    cap_dir: Path,
    subject: str,
    cand_frames: set[int],
    points_fallback: str | Path | None = None,
) -> dict[int, dict]:
    """Return ``{source_frame: details}`` for candidate frames in ``cap_dir`` where the capture sees her face."""
    if not cand_frames:
        return {}
    subj_dir = cap_dir / "subjects" / subject
    masks_files = sorted(subj_dir.glob("masks__*.npz"))
    if not masks_files:
        return {}
    import numpy as np
    import capture_masked_run as cap
    import head_tube_parts as htp
    from _lib import face_points as fp

    mz = np.load(masks_files[0])
    if "parts" not in mz.files or "track" not in mz.files:
        return {}
    classes_files = sorted(subj_dir.glob("classes__*.npz"))
    cmap_full = np.load(classes_files[0])["classes"] if classes_files else None

    man_file = cap_dir / "manifest.json"
    man = json.loads(man_file.read_text()) if man_file.is_file() else {}
    cap_first = int(man.get("first_frame", man.get("first_source_frame", min(cand_frames))))
    canvas = man.get("size") or man.get("canvas") or [0, 0]
    width = int(canvas[0]) if int(canvas[0]) > 0 else (int(cmap_full.shape[2]) if cmap_full is not None else 0)
    points_path: str | Path | None = None
    for s_entry in man.get("subjects") or []:
        if s_entry.get("label") == subject:
            for st in s_entry.get("sightings") or []:
                pts_in = (
                    (st.get("parts_dropped_off_her_head") or {}).get("points")
                    or (st.get("inputs") or {}).get("points")
                )
                if pts_in and Path(pts_in).is_file():
                    points_path = pts_in
                    break
    if points_path is None and points_fallback and Path(points_fallback).is_file():
        points_path = points_fallback

    parts_raw = np.unpackbits(mz["parts"], axis=-1)
    track_raw = np.unpackbits(mz["track"], axis=-1)
    if width > 0:
        parts_raw = parts_raw[..., :width]
        track_raw = track_raw[..., :width]
    parts_bool = parts_raw.astype(bool)
    track_bool = track_raw.astype(bool)
    num_frames, height, w_actual = parts_bool.shape

    pg_file = subj_dir / "part_grades.json"
    pg = json.loads(pg_file.read_text()) if pg_file.is_file() else {}
    least_px = int(pg.get("least_px") or cap.starting_floor(parts_bool, (w_actual, height), False)[0])
    part_names = [n for n in (pg.get("part_is_made_of") or ["Face_Neck"]) if n != "Eyeglass"] or ["Face_Neck"]

    valid_k = sorted(f - cap_first for f in cand_frames if 0 <= f - cap_first < num_frames)
    if not valid_k:
        return {}
    clusters: list[list[int]] = []
    for k in valid_k:
        if not clusters or k - clusters[-1][-1] > 6:
            clusters.append([k])
        else:
            clusters[-1].append(k)

    found: dict[int, dict] = {}
    for cluster in clusters:
        k0 = max(0, cluster[0] - 3)
        k1 = min(num_frames, cluster[-1] + 4)

        ht_res = htp.head_tube_for_capture(
            track_bool[k0:k1],
            parts_bool[k0:k1],
            cmap_full[k0:k1] if cmap_full is not None else None,
            points_path=points_path,
            first_frame=cap_first + k0,
            head_parts=part_names,
        )
        prov = ht_res["provenance"]
        pose_info = ht_res["pose_info"]

        pose_slice, kp_idx = (
            htp._load_pose_slice_for_capture(points_path, cap_first + k0, k1 - k0)
            if points_path is not None
            else ([None] * (k1 - k0), None)
        )
        idx_map = htp.POSE_KEYPOINT_INDEX if kp_idx is None else kp_idx

        for k in cluster:
            rel = k - k0
            sf = cap_first + k
            raw_px = int(parts_bool[k].sum())
            sm_px = int(prov["part_px"][rel])
            state = prov["frame_states"][rel] if rel < len(prov["frame_states"]) else "facing"
            fp_in_part = int(pose_info["face_points_in_part"][rel]) if pose_info is not None else 0
            fp_on_fn = int(pose_info["face_points_on_face_neck"][rel]) if pose_info is not None else 0
            pts_t = pose_slice[rel]
            if pts_t is not None and len(pts_t) > 0:
                union_mask = track_bool[k] | parts_bool[k] | ht_res["part_mask"][rel]
                mid_on_mask = bool(fp.face_middle_on_mask(pts_t[:, :2], idx_map, union_mask))
            else:
                mid_on_mask = True

            sees_face = (
                mid_on_mask
                and state != "back_of_head"
                and (
                    state in ("facing", "side_with_eye")
                    or raw_px >= least_px
                    or sm_px >= least_px
                    or fp_on_fn >= 2
                )
            )
            if sees_face:
                found[sf] = {
                    "state": state,
                    "raw_part_px": raw_px,
                    "part_px": sm_px,
                    "least_px": least_px,
                    "face_points_in_part": fp_in_part,
                    "face_points_on_face_neck": fp_on_fn,
                }
    return found


def carry_words_of(carry_words_from: dict | Path | str, shot: dict, subject: str, plan_parts: list[str],
                   first: int | None, frames: int | None, why: str = "", why_before: str = "", why_after: str = "",
                   by: str = "", now: str = "", leave_word: list[str] | None = None, leave_by: str | None = None) -> dict:
    """Read every person's word from an earlier load's job file for the frames the new load covers, tag each `(carried from <old load>)`,
    keep each word's original author and time with a `carried` record, refuse any word-bearing key the tool does not know how to carry
    or any carried span outside the new frames (unless dropped via `--leave-word` + `--leave-by`), and report retired/rebuilt keys."""
    import hashlib
    import h3_config
    import re
    if isinstance(carry_words_from, dict):
        old = carry_words_from
        old_file = "<dict>"
        old_sha256 = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    else:
        p = Path(carry_words_from)
        if not p.is_file():
            raise Refused(f"--carry-words-from {carry_words_from}: no such file")
        raw_bytes = p.read_bytes()
        old_file = str(p)
        old_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        old = json.loads(raw_bytes.decode())
    if not isinstance(old, dict):
        raise Refused("--carry-words-from: old job is not a JSON object")
    old_job = old.get("job") or {}
    old_loads = old.get("loads") or []
    old_name = str((old_loads[0].get("name") if old_loads and isinstance(old_loads[0], dict) else None) or old_job.get("name") or "old_job").strip()
    leave_word = list(leave_word or [])
    leave_by_str = " ".join(str(leave_by or "").split())
    if bool(leave_word) != bool(leave_by_str):
        raise Refused("--leave-word and --leave-by go together: the carried span to leave out, and who said so and why")
    leave_alias = {"word": "hold_part", "eyes_shut": "eyes_shut_on"}
    leave_kinds = {"left_out", "drop", "hold_part", "parts_on", "track_not", "no_face_on", "no_mouth_on", "mouth_on", "eyes_shut_on", "seam"}
    leave_map: dict[tuple[str, int, int], bool] = {}
    for raw_lw in leave_word:
        kind_part, sep, span_part = str(raw_lw).partition(":")
        if ":" in span_part:
            _, _, span_part = span_part.partition(":")
        k_norm = leave_alias.get(kind_part.strip(), kind_part.strip())
        if not sep or k_norm not in leave_kinds:
            raise Refused(f"--leave-word {raw_lw!r}: expected KIND:FIRST-LAST with KIND in {sorted(leave_kinds)}")
        try:
            lx_, ly_ = (int(v) for v in (span_part.strip().split("-") * 2)[:2])
        except ValueError:
            raise Refused(f"--leave-word {raw_lw!r}: span {span_part!r} is not FIRST-LAST") from None
        if lx_ > ly_:
            raise Refused(f"--leave-word {raw_lw!r}: {lx_}-{ly_} has first > last")
        leave_map[(k_norm, lx_, ly_)] = False
    for k in old:
        if k not in ("job", "defaults", "loads"):
            raise Refused(f"--carry-words-from {old_name}: word-bearing key {k!r} at top of old job cannot be carried")
    base_defaults = set(h3_config.MASKED_LOAD_DEFAULTS) | {"margin", "margin_by", "seed", "seed_by", "zoom_up_to", "zoom_up_to_by"}
    for k in (old.get("defaults") or {}):
        if k not in base_defaults:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'defaults.{k}' in old job cannot be carried")
    if len(old_loads) != 1 or not isinstance(old_loads[0], dict):
        raise Refused(f"--carry-words-from {old_name}: expected 1 load in old job, found {len(old_loads)}")
    old_load = old_loads[0] if (old_loads and isinstance(old_loads[0], dict)) else {}
    if old_load.get("subject") != subject:
        raise Refused(f"--carry-words-from {old_name}: old load's subject is {old_load.get('subject')!r}, not {subject!r}")
    load_known = {"name", "first", "frames", "subject", "others", "text", "motion_file", "motion_first", "render_region", "save_kept",
                  "hair_band", "motion", "window", "context_noise", "context_shows"}
    for k in old_load:
        if k not in load_known:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'loads[0].{k}' in old job cannot be carried")
    job_known = {"name", "note", "source", "source_frames", "shot", "first", "frames", "canvas", "output", "subjects", "capture", "preview", "reference_folder",
                 "texts", "keep_outs", "part_on_track", "others_present", "defaults", "restore",
                 "left_out", "zoomed", "whole_frame", "head_tube", "depth", "sam_head_track", "part", "track_not", "reference", "carried_words_from", "no_carry_words"}
    for k in old_job:
        if k not in job_known:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.{k}' in old job cannot be carried")
    if first is None and frames is None and old_job.get("left_out"):
        kept_lo = [e for e in (old_job.get("left_out") or [])
                   if isinstance(e, dict) and ("left_out", int(e.get("first", 0)), int(e.get("first", 0)) + int(e.get("frames", 0)) - 1) not in leave_map]
        if kept_lo:
            first = int(old_job["first"]) if any(int(e["first"]) == shot["first"] for e in kept_lo) else shot["first"]
            last_f = (int(old_job["first"]) + int(old_job["frames"]) - 1) if any(int(e["first"]) + int(e["frames"]) - 1 == shot["last"] for e in kept_lo) else shot["last"]
            frames = last_f - first + 1
    a = shot["first"] if first is None else first
    n = (shot["last"] - a + 1) if frames is None else frames
    b = a + n - 1
    tag = f"(carried from {old_name})"
    carried_rec = {"from": old_name, "file": old_file, "sha256": old_sha256, "by": by, "at": now}
    outside_spans: list[str] = []
    left_words: list[dict] = []
    left_said: list[str] = []
    overridden_said: list[str] = []

    def tagged(text: str) -> str:
        t = " ".join(str(text).split())
        return t if t.endswith(tag) else f"{t} {tag}"

    def is_left(kind_key: str, x: int, y: int, was_str: str) -> bool:
        if (kind_key, x, y) in leave_map:
            leave_map[(kind_key, x, y)] = True
            left_words.append({"kind": kind_key, "first": x, "last": y, "was": was_str, "by": leave_by_str, "at": now})
            left_said.append(f"{kind_key} {x}-{y}: {was_str} -> LEFT OUT ({leave_by_str})")
            return True
        return False

    def check_span(kind: str, x: int, y: int) -> None:
        if not (a <= x <= y <= b):
            outside_spans.append(f"{kind} {x}-{y}")

    carried_said: list[str] = []
    retired_said: list[str] = []
    out: dict = {
        "first": first, "frames": frames, "why_before": "", "why_after": "", "zoomed": None, "whole_frame": None,
        "zoomed_meta": None, "whole_frame_meta": None, "head_tube": None, "head_tube_meta": None,
        "depth": None, "depth_meta": None, "no_head_track": None, "no_head_track_meta": None,
        "no_closed_lip_line": None, "no_closed_lip_line_meta": None,
        "least": None, "least_why": None, "hold_part": [], "drop": None, "drop_by": None, "drop_meta": None,
        "parts": None, "parts_by": None, "parts_on": [], "track_not": [], "track_not_by": None,
        "motion": None, "motion_why": None, "no_strangers": None, "no_strangers_meta": None,
        "no_face_on": [], "no_face_on_meta": [], "no_mouth_on": [], "no_mouth_on_meta": [],
        "mouth_on": [], "mouth_on_meta": [], "eyes_shut_on": [], "eyes_shut_on_meta": [], "eyes_shut_form": None,
        "seam": None, "seam_why": None, "context_noise": None, "context_noise_why": None,
        "old_name": old_name, "old_file": old_file, "old_sha256": old_sha256, "carried_rec": carried_rec,
        "old_capture": old_job.get("capture"), "old_points": (old_job.get("reference") or {}).get("points"),
        "carried_said": carried_said, "retired_said": retired_said,
        "left_words": left_words, "left_said": left_said, "overridden_said": overridden_said,
    }

    old_seed = (old.get("defaults") or {}).get("seed")
    if "seed_by" in (old.get("defaults") or {}) or (old_seed is not None and old_seed != h3_config.SEED):
        said_by = (old.get("defaults") or {}).get("seed_by")
        retired_said.append(f"defaults.seed: not carried (was {old_seed!r}"
                            + (f", {said_by.get('by')}, {said_by.get('at')}: {said_by.get('why')}" if isinstance(said_by, dict) else "")
                            + f"; a seed is per load, so this load takes h3_config.SEED ({h3_config.SEED}) unless it is given --seed with --seed-why)")
    if "margin_by" in (old.get("defaults") or {}):
        said_by = (old.get("defaults") or {}).get("margin_by")
        retired_said.append(f"defaults.margin: not carried (was {(old.get('defaults') or {}).get('margin')!r}"
                            + (f", {said_by.get('by')}, {said_by.get('at')}: {said_by.get('why')}" if isinstance(said_by, dict) else "")
                            + "; a margin is per load, so this load takes h3_config.MASKED_LOAD_MARGIN by its region unless it is given "
                              "--margin with --margin-why)")
    if "zoom_up_to" in (old.get("defaults") or {}):
        retired_said.append(f"defaults.zoom_up_to: not carried (was {(old.get('defaults') or {}).get('zoom_up_to')!r}; a larger zoom is asked "
                            "for one load at a time, with --zoom-up-to and --zoom-up-to-why)")
    if old_job.get("restore"):
        retired_said.append("job.restore: retired (2026-10-16: plan restore gone from the plan; not carried)")
    if "hair_band" in old_load:
        retired_said.append(f"loads[0].hair_band: not carried (was {old_load.get('hair_band')!r}; rebuilt from the plan's hair_band)")
    if "context_shows" in old_load:
        retired_said.append(f"loads[0].context_shows: not carried (was {old_load.get('context_shows')!r}; rebuilt from the plan's context)")
    if "render_region" in old_load:
        retired_said.append(f"loads[0].render_region: not carried (was {old_load.get('render_region')!r}; rebuilt from zoom/whole_frame)")
    if "others" in old_load:
        retired_said.append(f"loads[0].others: not carried (was {old_load.get('others')!r}; rebuilt from the preview's subjects and the plan's keep_outs)")
    ko = old_job.get("keep_outs")
    if isinstance(ko, dict):
        for k in ko:
            if k not in ("of", "by", "preview", "first", "made"):
                raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.keep_outs.{k}' in old job cannot be carried")
        if ko.get("by"):
            retired_said.append(f"job.keep_outs.by: not carried (was {ko.get('by')!r}; rebuilt from the plan's keep_outs_by and others_present)")
        if any("stay" in item for item in ko.get("made") or [] if isinstance(item, dict)):
            retired_said.append("job.keep_outs.stay: retired (26f208c0, 2026-10-16: stay keep-out retired; not carried)")
    for tk, tv in (old_job.get("texts") or {}).items():
        if isinstance(tv, dict):
            for k in tv:
                if k not in ("file", "frames", "by", "generated_from"):
                    raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.texts.{tk}.{k}' in old job cannot be carried")
            if "BY HAND:" in str(tv.get("by") or ""):
                raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.texts.{tk}.by' (written by hand) cannot be carried by job")

    old_mine = None
    for s in old_job.get("subjects") or []:
        lbl = s.get("label")
        if lbl == subject:
            old_mine = s
        else:
            for k in s:
                if k not in ("label", "phrase", "pick", "parts", "still", "pose", "track_head"):
                    raise Refused(f"--carry-words-from {old_name}: word-bearing key 'subjects.{lbl}.{k}' on another subject cannot be carried")
    if old_mine is None:
        raise Refused(f"--carry-words-from {old_name}: subject {subject!r} is not in old job's subjects")
    subj_known = {"label", "phrase", "pick", "parts", "still", "pose", "track_head", "parts_file", "parts_held_to_track_px",
                  "pieces", "pieces_by", "least", "least_by", "least_carried", "word", "word_by", "word_carried",
                  "drop", "drop_by", "drop_carried", "parts_of_this_load", "parts_by", "parts_on",
                  "head_hidden", "hair_hers", "hair_not_hers", "head_marks_by",     # read again from the plan's `head_marks`, never carried
                  "facial_hair", "facial_hair_by"}                                   # read again from the plan's `source_facial_hair`, never carried
    for k in old_mine:
        if k not in subj_known:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'subjects.{subject}.{k}' in old job cannot be carried")
    if "pieces" in old_mine or "pieces_by" in old_mine:
        retired_said.append(
            f"subjects.{subject}.pieces/pieces_by: retired (e6993b5e, 2026-10-16: refused on a face or hair; not carried: was {old_mine.get('pieces')!r})"
        )
    if ("least" in old_mine) != ("least_by" in old_mine):
        raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.least and least_by must both be present")
    if "least" in old_mine:
        out["least"] = int(old_mine["least"])
        out["least_why"] = tagged(old_mine["least_by"])
        carried_said.append(f"least={out['least']}: {out['least_why']}")
    if ("word" in old_mine) != ("word_by" in old_mine):
        raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.word and word_by must both be present")
    if "word" in old_mine:
        for x, y, w, g in _parse_word_by(old_mine["word"], old_mine["word_by"], old_name, subject):
            if is_left("hold_part", x, y, w):
                continue
            check_span("hold_part (word)", x, y)
            if g is not None:
                check_span(f"hold_part (word) {x}-{y}, held in place from", g, g)
            tw = tagged(w)
            out["hold_part"].append((x, y, tw) if g is None else (x, y, tw, g))
            carried_said.append(f"hold_part {x}-{y}" + (f"@{g}" if g is not None else "") + f": {tw}")
    if ("drop" in old_mine) != ("drop_by" in old_mine):
        raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.drop and drop_by must both be present")
    if "drop" in old_mine:
        dspans = [s.strip() for s in str(old_mine["drop"]).split("+") if s.strip()]
        if not dspans or not str(old_mine.get("drop_by") or "").strip():
            raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.drop or drop_by is empty")
        d_who, d_at = _author_and_time_from_by(
            str(old_mine["drop_by"]),
            old_mine.get("drop_carried") or old_job.get("carried_words_from"),
        )
        kept_dspans = []
        kept_dpairs: list[tuple[int, int]] = []
        for sp in dspans:
            try:
                dx, dy = (int(v) for v in (sp.split("-") * 2)[:2])
            except ValueError:
                raise Refused(f"--carry-words-from {old_name}: subjects.{subject}.drop span {sp!r} is not FIRST-LAST") from None
            if is_left("drop", dx, dy, str(old_mine["drop_by"])):
                continue
            check_span("drop", dx, dy)
            kept_dspans.append(f"{dx}-{dy}")
            kept_dpairs.append((dx, dy))
        if kept_dspans:
            out["drop"] = "+".join(kept_dspans)
            out["drop_by"] = tagged(old_mine["drop_by"])
            out["drop_meta"] = {"said_by": d_who, "at": d_at, "spans": kept_dpairs}
            who_when = f" ({d_who}, {d_at})" if (d_who or d_at) else ""
            carried_said.append(f"drop={out['drop']}{who_when}: {out['drop_by']}")
    pblock = old_job.get("part")
    if pblock is not None or any(k in old_mine for k in ("parts_of_this_load", "parts_by", "parts_on")):
        pblock = pblock or {}
        for k in pblock:
            if k not in ("of", "by", "preview", "first", "parts", "the_plans_parts", "parts_on", "made", "carried"):
                raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.part.{k}' in old job cannot be carried")
        raw_pby = str(old_mine.get("parts_by") or pblock.get("by") or "").strip()
        if not raw_pby:
            raise Refused(f"--carry-words-from {old_name}: subject's part in old job has no parts_by")
        sub_parts = old_mine.get("parts_of_this_load") or pblock.get("parts")
        if sub_parts and list(sub_parts) != list(plan_parts):
            out["parts"] = list(sub_parts)
        for item in (old_mine.get("parts_on") or pblock.get("parts_on") or []):
            if not isinstance(item, dict) or set(item) - {"first", "last", "parts"}:
                raise Refused(f"--carry-words-from {old_name}: invalid parts_on entry {item!r} in old job")
            px, py, pnames = int(item["first"]), int(item["last"]), list(item["parts"])
            if is_left("parts_on", px, py, f"{pnames} ({raw_pby})"):
                continue
            check_span("parts_on", px, py)
            out["parts_on"].append((px, py, pnames))
        if out["parts"] or out["parts_on"]:
            out["parts_by"] = tagged(raw_pby)
            carried_said.append(f"parts={out['parts'] or plan_parts} parts_on={out['parts_on']}: {out['parts_by']}")

    for entry in old_job.get("left_out") or []:
        if not isinstance(entry, dict) or set(entry) - {"subject", "first", "frames", "why", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in left_out {entry!r} cannot be carried")
        if entry.get("subject") != subject or not str(entry.get("why") or "").strip():
            raise Refused(f"--carry-words-from {old_name}: left_out entry {entry!r} is not for {subject!r} or has no why")
        lx = int(entry["first"])
        ly = lx + int(entry["frames"]) - 1
        lw = tagged(entry["why"])
        if is_left("left_out", lx, ly, str(entry["why"])):
            continue
        if a > shot["first"] and (lx, ly) == (shot["first"], a - 1):
            if (why_before or why).strip():
                overridden_said.append(f"left_out {lx}-{ly}: OVERRIDDEN: new {(why_before or why).strip()!r} replaces carried {lw!r}")
            else:
                out["why_before"] = lw
                carried_said.append(f"left_out {lx}-{ly}: {lw}")
        elif b < shot["last"] and (lx, ly) == (b + 1, shot["last"]):
            if (why_after or why).strip():
                overridden_said.append(f"left_out {lx}-{ly}: OVERRIDDEN: new {(why_after or why).strip()!r} replaces carried {lw!r}")
            else:
                out["why_after"] = lw
                carried_said.append(f"left_out {lx}-{ly}: {lw}")
        elif (a > shot["first"] and (why_before or why).strip()) if lx < int(old_job.get("first", a)) else (b < shot["last"] and (why_after or why).strip()):
            new_cut_w = (why_before or why).strip() if lx < int(old_job.get("first", a)) else (why_after or why).strip()
            overridden_said.append(f"left_out {lx}-{ly}: OVERRIDDEN: new {new_cut_w!r} replaces carried {lw!r}")
        else:
            outside_spans.append(f"left_out {lx}-{ly}")

    if old_job.get("zoomed") is not None:
        oz = old_job["zoomed"]
        if not isinstance(oz, dict) or set(oz) - {"why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.zoomed {oz!r} cannot be carried")
        if oz.get("said_by") != "the chain's default":
            z_str = str(oz.get("why") or "")
            out["zoomed"] = tagged(z_str)
            out["zoomed_meta"] = {"said_by": oz.get("said_by"), "at": oz.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"zoomed ({oz.get('said_by')}, {oz.get('at')}): {out['zoomed']}")

    if old_job.get("whole_frame") is not None:
        ow = old_job["whole_frame"]
        if not isinstance(ow, dict) or set(ow) - {"why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.whole_frame {ow!r} cannot be carried")
        if ow.get("said_by") != "the plan":
            w_str = str(ow.get("why") or "")
            out["whole_frame"] = tagged(w_str)
            out["whole_frame_meta"] = {"said_by": ow.get("said_by"), "at": ow.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"whole_frame ({ow.get('said_by')}, {ow.get('at')}): {out['whole_frame']}")

    if old_job.get("head_tube") is not None:
        oht = old_job["head_tube"]
        if not isinstance(oht, dict) or set(oht) - {"on", "why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.head_tube {oht!r} cannot be carried")
        if oht.get("on") and oht.get("said_by") != "the plan":
            ht_str = str(oht.get("why") or "")
            out["head_tube"] = tagged(ht_str)
            out["head_tube_meta"] = {"said_by": oht.get("said_by"), "at": oht.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"head_tube ({oht.get('said_by')}, {oht.get('at')}): {out['head_tube']}")

    if old_job.get("depth") is not None:
        odp = old_job["depth"]
        if not isinstance(odp, dict) or set(odp) - {"on", "why", "said_by", "at", "carried", "overlap_frames", "file"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.depth {odp!r} cannot be carried")
        if odp.get("on") and odp.get("said_by") != "the plan":
            dp_str = str(odp.get("why") or "")
            out["depth"] = tagged(dp_str)
            out["depth_meta"] = {"said_by": odp.get("said_by"), "at": odp.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"depth ({odp.get('said_by')}, {odp.get('at')}): {out['depth']}")

    if old_job.get("sam_head_track") is not None:
        osht = old_job["sam_head_track"]
        if not isinstance(osht, dict) or set(osht) - {"on", "why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.sam_head_track {osht!r} cannot be carried")
        if not osht.get("on", True) and osht.get("said_by") != "the chain's default":
            sht_str = str(osht.get("why") or "")
            out["no_head_track"] = tagged(sht_str)
            out["no_head_track_meta"] = {"said_by": osht.get("said_by"), "at": osht.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"no_head_track ({osht.get('said_by')}, {osht.get('at')}): {out['no_head_track']}")

    if old_job.get("track_not") is not None:
        otn = old_job["track_not"]
        if not isinstance(otn, dict) or set(otn) - {"by", "preview", "made", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in job.track_not {otn!r} cannot be carried")
        out["track_not_by"] = tagged(otn.get("by") or "")
        for item in otn.get("made") or []:
            lbl = item.get("subject")
            for sp_pair, wline in zip(item.get("frames") or [], item.get("why") or []):
                tx, ty = int(sp_pair[0]), int(sp_pair[1])
                w = wline.split(": ", 1)[1] if ": " in wline else str(wline)
                if is_left("track_not", tx, ty, f"{lbl}:{w}"):
                    continue
                check_span(f"track_not {lbl}", tx, ty)
                tw = tagged(w)
                out["track_not"].append((lbl, tx, ty, tw))
                carried_said.append(f"track_not {lbl}:{tx}-{ty}: {tw}")

    old_ref = old_job.get("reference") or {}
    ref_known = {"form", "why", "made", "points", "record", "said_by", "taken_as_it_was", "figures", "drawn", "mouth", "iris", "no_face_on",
                 "darken_head", "head_track", "crop", "carried"}
    for k in old_ref:
        if k not in ref_known:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.reference.{k}' in old job cannot be carried")
    if old_ref.get("darken_head"):
        old_opt = old_ref["darken_head"]
        if not isinstance(old_opt, dict) or set(old_opt) - {"said_by", "why"} or old_opt.get("said_by") not in (None, "the plan"):
            raise Refused(f"--carry-words-from {old_name}: word-bearing setting 'reference.darken_head' ({old_opt!r}) cannot be carried by job")
    if old_ref.get("crop") is not None:
        # the plan's word, read again from the plan by the new job (`reference_form`), never carried
        ocr = old_ref["crop"]
        if not isinstance(ocr, dict) or set(ocr) - {"form", "said_by", "why"} or ocr.get("said_by") != "the plan":
            raise Refused(f"--carry-words-from {old_name}: word-bearing setting 'reference.crop' ({ocr!r}) cannot be carried by job")
    if old_ref.get("head_track") is not None:
        oht_ref = old_ref["head_track"]
        if not isinstance(oht_ref, dict) or set(oht_ref) - {"on", "why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.head_track {oht_ref!r} cannot be carried")
        if not oht_ref.get("on", True) and oht_ref.get("said_by") not in (None, "the chain's default", "the plan") and out["no_head_track"] is None:
            ht_r_str = str(oht_ref.get("why") or "")
            out["no_head_track"] = tagged(ht_r_str)
            out["no_head_track_meta"] = {"said_by": oht_ref.get("said_by"), "at": oht_ref.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"no_head_track ({oht_ref.get('said_by')}, {oht_ref.get('at')}): {out['no_head_track']}")
    if old_ref.get("form") == "none" or old_load.get("motion") == "none":
        out["motion"] = "none"
        out["motion_why"] = tagged(old_ref.get("why") or "")
        carried_said.append(f"motion=none: {out['motion_why']}")
    elif old_ref.get("form") not in (None, "drawn"):
        raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.reference.form' ({old_ref.get('form')!r}) in old job cannot be carried by job")

    for s in old_ref.get("no_face_on") or []:
        if not isinstance(s, dict) or set(s) - {"first", "last", "why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.no_face_on {s!r} cannot be carried")
        fx, fy = int(s["first"]), int(s["last"])
        f_str = str(s.get("why") or "")
        if is_left("no_face_on", fx, fy, f"{s.get('said_by')}, {s.get('at')}: {f_str}"):
            continue
        check_span("no_face_on", fx, fy)
        tf = tagged(f_str)
        out["no_face_on"].append((fx, fy, tf))
        out["no_face_on_meta"].append({"said_by": s.get("said_by"), "at": s.get("at"), "carried": dict(carried_rec)})
        carried_said.append(f"no_face_on {fx}-{fy} ({s.get('said_by')}, {s.get('at')}): {tf}")

    old_mouth = old_ref.get("mouth") or {}
    for k in old_mouth:
        if k not in ("form", "said_by", "why", "no_mouth_on", "mouth_on", "closed_lip_line", "open_mouth", "closed_mouth"):   # open_mouth, closed_mouth: the plan's, made again from it
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.reference.mouth.{k}' in old job cannot be carried")
    if old_mouth.get("said_by") not in (None, "the chain's default", "the plan") and not str(old_mouth.get("why") or "").startswith("silent window "):
        raise Refused(f"--carry-words-from {old_name}: word-bearing setting 'reference.mouth' (said_by {old_mouth.get('said_by')!r}) cannot be carried by job")
    if old_mouth.get("closed_lip_line") is not None:
        ocl = old_mouth["closed_lip_line"]
        if not isinstance(ocl, dict) or set(ocl) - {"on", "why", "said_by", "at", "carried"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.mouth.closed_lip_line {ocl!r} cannot be carried")
        if not ocl.get("on", True) and ocl.get("said_by") not in (None, "the chain's default", "the plan"):
            cl_str = str(ocl.get("why") or "")
            out["no_closed_lip_line"] = tagged(cl_str)
            out["no_closed_lip_line_meta"] = {"said_by": ocl.get("said_by"), "at": ocl.get("at"), "carried": dict(carried_rec)}
            carried_said.append(f"no_closed_lip_line ({ocl.get('said_by')}, {ocl.get('at')}): {out['no_closed_lip_line']}")
    for field, label in (("no_mouth_on", "no_mouth_on"), ("mouth_on", "mouth_on")):
        for s in old_mouth.get(field) or []:
            if not isinstance(s, dict) or set(s) - {"first", "last", "why", "said_by", "at", "carried"}:
                raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.mouth.{field} {s!r} cannot be carried")
            mx, my = int(s["first"]), int(s["last"])
            m_str = str(s.get("why") or "")
            if is_left(label, mx, my, f"{s.get('said_by')}, {s.get('at')}: {m_str}"):
                continue
            check_span(label, mx, my)
            tm = tagged(m_str)
            out[field].append((mx, my, tm))
            out[f"{field}_meta"].append({"said_by": s.get("said_by"), "at": s.get("at"), "carried": dict(carried_rec)})
            carried_said.append(f"{label} {mx}-{my} ({s.get('said_by')}, {s.get('at')}): {tm}")

    old_iris = old_ref.get("iris") or {}
    for k in old_iris:
        if k not in ("form", "said_by", "why", "eyes_shut"):
            raise Refused(f"--carry-words-from {old_name}: word-bearing key 'job.reference.iris.{k}' in old job cannot be carried")
    if old_iris.get("said_by") not in (None, "the chain's default", "the plan"):
        raise Refused(f"--carry-words-from {old_name}: word-bearing setting 'reference.iris' (said_by {old_iris.get('said_by')!r}) cannot be carried by job")
    if old_iris.get("eyes_shut"):
        es = old_iris["eyes_shut"]
        if not isinstance(es, dict) or set(es) - {"form", "spans"}:
            raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.iris.eyes_shut {es!r} cannot be carried")
        for s in es.get("spans") or []:
            if not isinstance(s, dict) or set(s) - {"first", "last", "why", "said_by", "at", "carried"}:
                raise Refused(f"--carry-words-from {old_name}: word-bearing key in reference.iris.eyes_shut.spans {s!r} cannot be carried")
            ex, ey = int(s["first"]), int(s["last"])
            e_str = str(s.get("why") or "")
            if is_left("eyes_shut_on", ex, ey, f"{s.get('said_by')}, {s.get('at')}: {e_str}"):
                continue
            check_span("eyes_shut_on", ex, ey)
            te = tagged(e_str)
            out["eyes_shut_on"].append((ex, ey, te))
            out["eyes_shut_on_meta"].append({"said_by": s.get("said_by"), "at": s.get("at"), "carried": dict(carried_rec)})
            carried_said.append(f"eyes_shut_on ({es.get('form')}) {ex}-{ey} ({s.get('said_by')}, {s.get('at')}): {te}")
        if out["eyes_shut_on"]:
            out["eyes_shut_form"] = es.get("form")

    defs_known = {"the reference video", "the reference held to the subject's track", "the eye", "the mouth",
                  "the part held on a person's word", "the face lines", "the head tube", "depth", "zoom", "other people in the shot",
                  "the strangers keep-out", "the part's pieces", "the hair band", "restore spans", "the text",
                  "what the composite kept", "the voice", "the seam", "the kept frames' noise",
                  "SAM 3.1 memory selection", "SAM 3.1 head track",     # read again from the new job's own preview, never carried
                  "the context frames",                                 # read again from the plan's `context`, never carried
                  "the reference's crop",                               # read again from the plan's `reference.crop`, never carried
                  "the seed",                                           # a seed is per load: only this load's own `--seed`, never carried
                  "the most the node may enlarge this zoomed load",     # per load too: only this load's own `--zoom-up-to`, never carried
                  "the margin"}                                         # per load too: only this load's own `--margin`, never carried
    defs = {}
    for row in old_job.get("defaults") or []:
        if not isinstance(row, dict) or row.get("what") not in defs_known:
            raise Refused(f"--carry-words-from {old_name}: word-bearing default {row!r} in old job cannot be carried")
        defs[row["what"]] = row
    st_row = defs.get("the strangers keep-out")
    if st_row and str(st_row.get("from") or "").startswith("TURNED OFF on a person's word"):
        fr = str(st_row["from"])
        if fr.startswith("TURNED OFF on a person's word (") and "): " in fr:
            who_at, _, w = fr[len("TURNED OFF on a person's word ("):].partition("): ")
            orig_by, _, orig_at = who_at.partition(", ")
            ns_val = tagged(w)
            out["no_strangers_meta"] = {"by": orig_by.strip() or who_at, "at": orig_at.strip(), "who_at": who_at, "carried": dict(carried_rec)}
        else:
            ns_val = tagged(fr)
            out["no_strangers_meta"] = {"by": "", "at": "", "who_at": "", "carried": dict(carried_rec)}
        out["no_strangers"] = ns_val
        carried_said.append(f"no_strangers: {tagged(fr)}")
    sm_row = defs.get("the seam")
    if "window" in old_load or (sm_row and not str(sm_row.get("from") or "").startswith("the planner")):
        seams = [int(x) for x in re.findall(r"writes from (\d+)", (sm_row or {}).get("carries", ""))]
        if not seams or not sm_row or str(sm_row.get("from") or "").startswith("the planner"):
            raise Refused(f"--carry-words-from {old_name}: loads[0].window={old_load.get('window')!r} has no person's seam words in job.defaults")
        seams = [sf for sf in seams if not is_left("seam", sf, sf, str(sm_row["from"]))]
        for sf in seams:
            if not (a < sf <= b):
                outside_spans.append(f"seam {sf}-{sf}")
        out["seam"] = seams
        if out["seam"]:
            out["seam_why"] = tagged(sm_row["from"])
            carried_said.append(f"seam={','.join(map(str, out['seam']))}: {out['seam_why']}")
        else:
            out["seam"] = None
    cn_row = defs.get("the kept frames' noise")
    if "context_noise" in old_load or cn_row:
        if "context_noise" not in old_load or not cn_row or not str(cn_row.get("from") or "").strip():
            raise Refused(f"--carry-words-from {old_name}: context_noise in old job is missing its defaults reason")
        out["context_noise"] = float(old_load["context_noise"])
        out["context_noise_why"] = tagged(cn_row["from"])
        carried_said.append(f"context_noise={out['context_noise']:g}: {out['context_noise_why']}")
    unmatched = [f"{k}:{x}-{y}" for (k, x, y), matched in leave_map.items() if not matched]
    if unmatched:
        raise Refused(f"--carry-words-from {old_name}: --leave-word {', '.join(unmatched)} has no such span in {old_name}")
    if outside_spans:
        raise Refused(f"--carry-words-from {old_name}: carried span(s) {', '.join(outside_spans)} lie outside the new job's frames {a}-{b}")
    return out


def load_job(shots: dict, plan: dict, shot: dict, subject: str, preview: dict, preview_folder: str, video: str, output: Path, by: str, day: str,
             name: str, first: int | None = None, frames: int | None = None, why: str = "", now: str = "", source: str | None = None,
             why_before: str = "", why_after: str = "", zoomed: str | None = None, least: int | None = None, least_why: str | None = None,
             whole_frame: str | None = None,
             preview_files: list[str] | None = None, parts: list[str] | None = None, parts_by: str | None = None, classes_of=part_classes,
             parts_on: list[tuple[int, int, list[str]]] | None = None, track_not: list[tuple[str, int, int, str]] | None = None,
             track_not_by: str | None = None, cast: list[dict] | None = None, cast_from: str = "", motion: str | None = None,
             motion_why: str | None = None, no_strangers: str | None = None, no_mouth_on: list[tuple[int, int, str]] | None = None,
             eyes_shut_on: list[tuple[int, int, str]] | None = None, eyes_shut_form: str | None = None,
             no_face_on: list[tuple[int, int, str]] | None = None, hold_part: list[tuple[int, int, str]] | None = None, seam: list[int] | None = None,
             seam_why: str | None = None, context_noise: float | None = None, context_noise_why: str | None = None,
             mouth_on: list[tuple[int, int, str]] | None = None, data: Path | None = None,
              drop: str | None = None, drop_by: str | None = None, carry_words_from: dict | Path | str | None = None,
              leave_word: list[str] | None = None, leave_by: str | None = None, no_carry_words: str | None = None,
              head_tube: str | None = None, depth: str | None = None, no_head_track: str | None = None,
              no_closed_lip_line: str | None = None, capture_folder: Path | str | None = None,
              seed: int | str | None = None, seed_why: str | None = None,
              zoom_up_to: float | str | None = None, zoom_up_to_why: str | None = None,
              margin: int | str | None = None, margin_why: str | None = None) -> dict:
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
    cw = None
    wb_from_cw = wa_from_cw = zoomed_from_cw = whole_from_cw = head_tube_from_cw = depth_from_cw = no_head_track_from_cw = no_cl_from_cw = least_from_cw = False
    parts_from_cw = track_not_from_cw = motion_from_cw = ns_from_cw = es_from_cw = False
    seam_from_cw = cn_from_cw = drop_from_cw = False
    n_cw_hp = 0
    n_cw_nf = 0
    if carry_words_from is not None and no_carry_words is not None:
        raise Refused("--carry-words-from and --no-carry-words cannot both be given")
    job_seed, seed_by = job_seed_of(seed, seed_why, by, now)
    up_to, up_to_by = zoom_up_to_of(zoom_up_to, zoom_up_to_why, by, now)
    own_margin, margin_by = margin_of(margin, margin_why, by, now)
    ncw_why = ""
    if no_carry_words is not None:
        ncw_why = " ".join(str(no_carry_words).split())
        if not ncw_why:
            raise Refused("--no-carry-words takes a person's words for why the earlier job's words are not carried")
    if carry_words_from is None and (leave_word or str(leave_by or "").strip()):
        raise Refused("--leave-word goes with --carry-words-from: there is no carried job to leave a word out of")
    if carry_words_from is not None:
        cw = carry_words_of(carry_words_from, shot, subject, list(p.get("parts") or []), first, frames, why, why_before, why_after,
                            by=by, now=now, leave_word=leave_word, leave_by=leave_by)
        def _ov(prefix: str, new_desc: str, old_desc: str) -> None:
            cw["carried_said"][:] = [x for x in cw["carried_said"] if not x.startswith(prefix)]
            cw["overridden_said"].append(f"{prefix.rstrip(': =')}: OVERRIDDEN: new {new_desc} replaces carried {old_desc}")
        first = cw["first"] if first is None else first
        frames = cw["frames"] if frames is None else frames
        wb_from_cw = not bool((why_before or why).strip()) and bool(cw["why_before"])
        wa_from_cw = not bool((why_after or why).strip()) and bool(cw["why_after"])
        why_before = why_before or cw["why_before"]
        why_after = why_after or cw["why_after"]
        if zoomed is None and whole_frame is None and cw["zoomed"] is not None:
            zoomed, zoomed_from_cw = cw["zoomed"], True
        elif cw["zoomed"] is not None:
            _ov("zoomed ", f"zoomed={zoomed!r} whole_frame={whole_frame!r}", repr(cw["zoomed"]))
        if whole_frame is None and zoomed is None and cw["whole_frame"] is not None:
            whole_frame, whole_from_cw = cw["whole_frame"], True
        elif cw["whole_frame"] is not None:
            _ov("whole_frame ", f"whole_frame={whole_frame!r} zoomed={zoomed!r}", repr(cw["whole_frame"]))
        if head_tube is None and cw["head_tube"] is not None:
            head_tube, head_tube_from_cw = cw["head_tube"], True
        elif cw["head_tube"] is not None:
            _ov("head_tube ", f"head_tube={head_tube!r}", repr(cw["head_tube"]))
        if depth is None and cw["depth"] is not None:
            depth, depth_from_cw = cw["depth"], True
        elif cw["depth"] is not None:
            _ov("depth ", f"depth={depth!r}", repr(cw["depth"]))
        if no_head_track is None and cw["no_head_track"] is not None:
            no_head_track, no_head_track_from_cw = cw["no_head_track"], True
        elif cw["no_head_track"] is not None:
            _ov("no_head_track ", f"no_head_track={no_head_track!r}", repr(cw["no_head_track"]))
        if no_closed_lip_line is None and cw["no_closed_lip_line"] is not None:
            no_closed_lip_line, no_cl_from_cw = cw["no_closed_lip_line"], True
        elif cw["no_closed_lip_line"] is not None:
            _ov("no_closed_lip_line ", f"no_closed_lip_line={no_closed_lip_line!r}", repr(cw["no_closed_lip_line"]))
        if least is None and cw["least"] is not None:
            least, least_why, least_from_cw = cw["least"], cw["least_why"], True
        elif cw["least"] is not None:
            _ov("least=", f"least={least} ({least_why})", f"least={cw['least']} ({cw['least_why']})")
        if parts is None and not parts_on and cw["parts_by"] is not None:
            parts, parts_on, parts_by, parts_from_cw = cw["parts"], list(cw["parts_on"]), cw["parts_by"], True
        elif cw["parts_by"] is not None:
            _ov("parts=", f"parts={parts} parts_on={parts_on} ({parts_by})", f"parts={cw['parts']} parts_on={cw['parts_on']} ({cw['parts_by']})")
        if not track_not and cw["track_not"]:
            track_not, track_not_by, track_not_from_cw = list(cw["track_not"]), cw["track_not_by"], True
        elif cw["track_not"]:
            _ov("track_not ", f"{track_not} ({track_not_by})", f"{cw['track_not']} ({cw['track_not_by']})")
        if motion is None and cw["motion"] is not None:
            motion, motion_why, motion_from_cw = cw["motion"], cw["motion_why"], True
        elif cw["motion"] is not None:
            _ov("motion=", f"motion={motion} ({motion_why})", f"motion={cw['motion']} ({cw['motion_why']})")
        if no_strangers is None and cw["no_strangers"] is not None:
            no_strangers, ns_from_cw = cw["no_strangers"], True
        elif cw["no_strangers"] is not None:
            _ov("no_strangers:", repr(no_strangers), repr(cw["no_strangers"]))
        fresh_nf = list(no_face_on or [])
        if fresh_nf and cw["no_face_on"]:
            kept_cw_nf = []
            kept_cw_nf_meta = []
            for (cx, cy, csaid), cmeta in zip(cw["no_face_on"], cw["no_face_on_meta"]):
                if any(nx <= cy and cx <= ny for nx, ny, _ in fresh_nf):
                    _ov(f"no_face_on {cx}-{cy}", f"{fresh_nf}", f"{cx}-{cy} ({cmeta.get('said_by')}, {cmeta.get('at')}): {csaid}")
                else:
                    kept_cw_nf.append((cx, cy, csaid))
                    kept_cw_nf_meta.append(cmeta)
            cw["no_face_on"] = kept_cw_nf
            cw["no_face_on_meta"] = kept_cw_nf_meta
        n_cw_nf = len(cw["no_face_on"])
        no_face_on = list(cw["no_face_on"]) + fresh_nf
        no_mouth_on = list(cw["no_mouth_on"]) + list(no_mouth_on or [])
        mouth_on = list(cw["mouth_on"]) + list(mouth_on or [])
        if not eyes_shut_on and cw["eyes_shut_on"]:
            eyes_shut_on, eyes_shut_form, es_from_cw = list(cw["eyes_shut_on"]), cw["eyes_shut_form"], True
        elif cw["eyes_shut_on"]:
            _ov("eyes_shut_on ", f"{eyes_shut_on} (form={eyes_shut_form})", f"{cw['eyes_shut_on']} (form={cw['eyes_shut_form']})")
        n_cw_hp = len(cw["hold_part"])
        fresh_hp = list(hold_part or [])
        hold_part = list(cw["hold_part"]) + fresh_hp
        if seam is None and cw["seam"] is not None:
            seam, seam_why, seam_from_cw = cw["seam"], cw["seam_why"], True
        elif cw["seam"] is not None:
            _ov("seam=", f"seam={seam} ({seam_why})", f"seam={cw['seam']} ({cw['seam_why']})")
        if context_noise is None and cw["context_noise"] is not None:
            context_noise, context_noise_why, cn_from_cw = cw["context_noise"], cw["context_noise_why"], True
        elif cw["context_noise"] is not None:
            _ov("context_noise=", f"context_noise={context_noise} ({context_noise_why})", f"context_noise={cw['context_noise']} ({cw['context_noise_why']})")
        if drop is None and cw["drop"] is not None:
            drop, drop_by, drop_from_cw = cw["drop"], cw["drop_by"], True
        elif cw["drop"] is not None:
            _ov("drop=", f"drop={drop} ({drop_by})", f"drop={cw['drop']} ({cw['drop_by']})")
        # Refuse to carry any frame of drop or no_face_on where the capture sees her face (unless overridden by a fresh word on this job).
        fresh_hp_frames = {f for hx, hy, _ in fresh_hp for f in range(hx, hy + 1)}
        carried_drop_spans = list((cw.get("drop_meta") or {}).get("spans") or []) if drop_from_cw else []
        carried_nf_spans = [
            (fx, fy, cw["no_face_on_meta"][idx])
            for idx, (fx, fy, _) in enumerate(no_face_on[:n_cw_nf])
        ]
        cand_face_check = (
            {f for dx, dy in carried_drop_spans for f in range(dx, dy + 1)}
            | {f for fx, fy, _ in carried_nf_spans for f in range(fx, fy + 1)}
        ) - fresh_hp_frames
        if cand_face_check:
            cap_dir = _resolve_old_capture_dir(cw, carry_words_from, data, subject, capture_folder=capture_folder)
            if cap_dir is not None:
                import capture_masked_run as cap
                face_seen = capture_face_frames(cap_dir, subject, cand_face_check, points_fallback=cw.get("old_points"))
                if face_seen:
                    bad_clauses: list[str] = []
                    if carried_drop_spans:
                        d_meta = cw.get("drop_meta") or {}
                        d_who_at = f"{d_meta.get('said_by')}, {d_meta.get('at')}".strip(", ")
                        for dx, dy in carried_drop_spans:
                            hit = sorted(f for f in range(dx, dy + 1) if f in face_seen and f not in fresh_hp_frames)
                            if hit:
                                sp_str = ", ".join(f"{u}-{v}" for u, v in cap.frame_spans(hit, join=1))
                                bad_clauses.append(
                                    f"drop {dx}-{dy}" + (f" ({d_who_at})" if d_who_at else "") + f" has face on {sp_str}"
                                )
                    for fx, fy, nf_meta in carried_nf_spans:
                        nf_who_at = f"{nf_meta.get('said_by')}, {nf_meta.get('at')}".strip(", ")
                        hit = sorted(f for f in range(fx, fy + 1) if f in face_seen and f not in fresh_hp_frames)
                        if hit:
                            sp_str = ", ".join(f"{u}-{v}" for u, v in cap.frame_spans(hit, join=1))
                            bad_clauses.append(
                                f"no_face_on {fx}-{fy}" + (f" ({nf_who_at})" if nf_who_at else "") + f" has face on {sp_str}"
                            )
                    if bad_clauses:
                        all_hit = sorted(face_seen)
                        all_sp = ", ".join(f"{u}-{v}" for u, v in cap.frame_spans(all_hit, join=1))
                        raise Refused(
                            f"--carry-words-from {cw['old_name']}: refused to carry drop / no_face_on silently onto frame(s) {all_sp} "
                            f"where the capture ({cap_dir.name}) sees {subject}'s face ({'; '.join(bad_clauses)}). "
                            "Look at those frames and narrow or leave out the carried span (--leave-word KIND:FIRST-LAST --leave-by WHY, "
                            "or pass a fresh --drop / --no-face-on / --hold-part)"
                        )
    # WHO ELSE IS IN THE SHOT (the owner's word of 2026-10-14: it belongs in the job file), from the preview's own tables and the
    # owner's answers (bench/others_present.py); the strangers keep-out follows it and not the plan's key, where it can say
    others = others_of_load(preview, output, shot, cast, cast_from, plan.get("not_people"))
    plans = p
    p = dict(strangers_by_the_data(p, others), others_present=others)
    # NO STRANGERS KEEP-OUT ON THIS LOAD, on a person's word (`--no-strangers WHY`, the lead's ruling of 2026-10-15): the keep-out marks
    # her own neck, dress and hair beyond a narrow track as strangers on a close-up (2823b: 263 pieces kept though touching her,
    # 7,657,313 px), and ruling (A) of 2026-10-14 keeps another person's arm or face against her a stranger, so the tool cannot tell them
    # apart; a look can. WHY names the look and the contact sheet it was made on; where the record has seen someone else, those frames too
    if no_strangers is not None:
        said_ns = " ".join(str(no_strangers).split())
        sheets = [w.strip(".,;:()\"'") for w in said_ns.split() if Path(w.strip(".,;:()\"'")).is_file()]
        if not sheets:
            raise Refused("--no-strangers takes a person's word on a look: who looked, at the contact sheet of the load's frames (name its file: the "
                          "tool checks only that a file of that name is there), and that nobody but our subjects is in the picture")
        named = frames_in_words(said_ns)
        seen = [int(f) for f in (others.get("frames_with_others") or [])] if others.get("others_present") == "yes" else []
        if [f for f in seen if f not in named]:
            raise Refused(f"--no-strangers on a shot where the record has seen someone else (frame(s) {', '.join(map(str, seen))}): name each of those "
                          "frames in WHY and why it does not matter")
        left = [k for k in p.get("keep_outs") or [] if k != "strangers"]
        ns_meta = cw["no_strangers_meta"] if (cw and ns_from_cw) else None
        ns_who = ns_meta["who_at"] if (ns_meta and ns_meta.get("who_at")) else f"{by}, {now}"
        p = dict(p, keep_outs=left or None,
                 strangers_turned_off={"why": said_ns, "file_named": sheets[0],
                                       "by": (ns_meta["by"] or by) if ns_meta else by,
                                       "at": (ns_meta["at"] or now) if ns_meta else now,
                                       **({"carried": dict(ns_meta["carried"])} if ns_meta else {})},
                 strangers_from=f"TURNED OFF on a person's word ({ns_who}): {said_ns}")
        p.pop("keep_outs_by", None)
        if left:                                        # the plan's own reason for the keep-outs left, without the data's strangers clause
            p["keep_outs_by"] = " ".join(str(plans.get("keep_outs_by") or "").split())
    region = p["region"]
    if region not in h3_config.MASKED_LOAD_MARGIN:
        raise Refused(f"the plan's region for {subject!r} is {region!r}: this tool makes part loads ({', '.join(h3_config.MASKED_LOAD_MARGIN)})")
    a = shot["first"] if first is None else first
    n = shot["last"] - a + 1 if frames is None else frames
    b = a + n - 1
    if not (shot["first"] <= a <= b <= shot["last"]):
        raise Refused(f"frames {a}-{b} are not inside the shot {shot['first']}-{shot['last']}")
    # HER CLOTHING NAMED FOR THESE FRAMES WHERE IT IS REDRAWN WITH HER (`load_text.clothing_of`, per shot, 1f143f59; 26f208c0): refused
    # here, before a reference or a capture is made for a text that would not name it, or would name another shot's
    import load_text
    try:
        load_text.clothing_of(p, subject, a, b)
    except load_text.Refused as e:
        raise Refused(str(e)) from None
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
    ref_folder, ref_why = reference_folder(output, video, subject, a, b, data)
    reference = ref_folder / f"{subject}_head_reference_{a:04d}_{b:04d}.mp4"
    load = {"name": name, "first": a, "frames": n, "subject": subject, "others": [k for k in theirs if k != subject], "text": name,
            "motion_file": str(reference), "motion_first": a}
    # NO REFERENCE VIDEO (`--motion none`): the lead's ruling of 2026-10-15, one test on second 5461-5497, a load of her hair alone whose
    # face is hidden, where the body model guesses her head upright while she is bowed: the source's own picture carries her bow. Only a
    # load of the hair alone, until a load of another kind is ruled
    if motion is not None:
        if motion != "none":
            raise Refused(f"--motion {motion!r}: the one form taken here is `none`")
        if sorted(parts or []) != ["hair"] or any(sorted(these) != ["hair"] for _x, _y, these in parts_on or []):
            raise Refused("--motion none is for a load of her hair alone on every frame (--parts hair, no --parts-on of anything else), "
                          "the lead's ruling of 2026-10-15: no other kind is ruled")
        if not str(motion_why or "").strip():
            raise Refused("--motion none is a person's word: say why with --motion-why")
        load = {k: v for k, v in load.items() if k not in ("motion_file", "motion_first")} | {"motion": "none"}
    for x, y, _ in parts_on or []:
        if not (a <= x <= y <= b):
            raise Refused(f"--parts-on {x}-{y} is not inside the load's frames {a}-{b}")
    part, part_said = part_of(p, subject, parts, parts_by, preview, video, output, preview_files, classes_of, parts_on)
    if part and cw and parts_from_cw:
        part["carried"] = dict(cw["carried_rec"])
    kept, kept_said = keep_outs_of(p, subject, preview, list(theirs), video, output, preview_files)
    on_track, on_track_said = part_on_track(p, subject, part, preview, video, output, preview_files)
    gone, gone_said = track_not_of(subject, preview, track_not or [], track_not_by, (a, b), video, output, preview_files)
    if gone and cw and track_not_from_cw:
        gone["carried"] = dict(cw["carried_rec"])
    if on_track and gone:
        gone_said += (" WITH THE STRANGERS KEPT OUT AS WELL: the strangers file takes out each subject's PREVIEW track, so on the frames named above "
                      "the pixels under that track are neither a stranger nor an other of this load, and are regenerable where the region reaches them.")
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
    # WHAT THE CONTEXT FRAMES SHOW (`context_of`, the plan's word): written into the load only off the node's default
    context = context_of(p, subject)
    if context["said_by"] == "the plan" and context["value"] != "as filmed":
        load["context_shows"] = context["value"]
    # A LOOK'S STAY ITEM made as one keep-out in place of one the plan names (the plan's `look.stay`; the lead's ruling of
    # 2026-10-14 on mrpre's apron read): her clothing held over frames, the band's box already out of it
    kept, stay_said = stay_of(p, subject, kept, load, preview, on_track, part, video, output, preview_files, name, region)
    kept_said += stay_said
    ref_form = reference_form(p, subject)
    # NO MOUTH ON A SPAN, on a person's word (`--no-mouth-on FIRST-LAST=WHY`, the lead's word of 2026-10-15 on 1010h): where her mouth
    # is hidden behind her hand the class map still labels lips, and a few px of teeth or tongue, at her fingers' edge (1010h's
    # 1011-1017: 1-15 px a frame on a face about 38 px tall), so neither its gate nor --class-mouth-needs-hole can drop them. WHY names
    # the look and the sheet it was made on (a file of that name, which is all the tool checks); each span lies inside the load
    if no_mouth_on:
        if motion == "none":
            raise Refused("--no-mouth-on acts on the drawn reference, and this load has none (--motion none)")
        spans = []
        for idx, (x, y, said) in enumerate(no_mouth_on):
            said = " ".join(str(said).split())
            if not (a <= x <= y <= b):
                raise Refused(f"--no-mouth-on {x}-{y}: the load is {a}-{b}")
            if not any(Path(w.strip(".,;:()\"'")).is_file() for w in said.split()):
                raise Refused(f"--no-mouth-on {x}-{y} takes a person's word on a look: who looked, at which sheet of those frames (name its file: the "
                              "tool checks only that a file of that name is there), and that her mouth is hidden there")
            meta = cw["no_mouth_on_meta"][idx] if (cw and idx < len(cw["no_mouth_on_meta"])) else None
            spans.append({"first": x, "last": y, "why": said,
                          "said_by": meta["said_by"] if meta else by, "at": meta["at"] if meta else now,
                          **({"carried": dict(meta["carried"])} if meta else {})})
        ref_form["mouth"] = dict(ref_form["mouth"], no_mouth_on=spans)
    # NO LINE OF THE FACE ON A SPAN, on a person's word (`--no-face-on FIRST-LAST=WHY`, the lead's ruling of 2026-10-15 on the hall shot's
    # references, which drew eyes and brows on the backs of two heads, and on 4310's 4395-4398): the plain cut body model alone there, as on
    # a load of her hair alone. Nothing finds those frames on its own (`face_cues`, printed at the reference step, points at some and misses
    # others). The same terms as --no-mouth-on: the look and its file named, each span inside the load
    if no_face_on:
        if motion == "none":
            raise Refused("--no-face-on acts on the drawn reference, and this load has none (--motion none)")
        spans = []
        for idx, (x, y, said) in enumerate(no_face_on):
            said = " ".join(str(said).split())
            if not (a <= x <= y <= b):
                raise Refused(f"--no-face-on {x}-{y}: the load is {a}-{b}")
            if not any(Path(w.strip(".,;:()\"'")).is_file() for w in said.split()):
                raise Refused(f"--no-face-on {x}-{y} takes a person's word on a look: who looked, at which sheet of those frames (name its file: the "
                              "tool checks only that a file of that name is there), and that the face does not show there")
            meta = cw["no_face_on_meta"][idx] if (cw and idx < len(cw["no_face_on_meta"])) else None
            spans.append({"first": x, "last": y, "why": said,
                          "said_by": meta["said_by"] if meta else by, "at": meta["at"] if meta else now,
                          **({"carried": dict(meta["carried"])} if meta else {})})
        if sorted(parts or []) == ["hair"]:
            raise Refused("--no-face-on on a load of her hair alone: no line of the face is drawn on any of its frames already")
        ref_form["no_face_on"] = spans
    # A MOUTH ON A SPAN, on a person's word (`--mouth-on FIRST-LAST=WHY`, the lead's ruling of 2026-10-15 on lead_fp_2823c, whose reference
    # drew no mouth on 3098-3099 and 3101-3104 while she sings with it wide open): where the reference's gate drops the mouth, the pose model's
    # lip rings are drawn with a dark inside (head_reference.MOUTH_ON). The same terms as --no-mouth-on: the look and its file named, each
    # span inside the load; never on a frame the job leaves with no mouth or no face
    if mouth_on:
        if motion == "none":
            raise Refused("--mouth-on acts on the drawn reference, and this load has none (--motion none)")
        spans = []
        for idx, (x, y, said) in enumerate(mouth_on):
            said = " ".join(str(said).split())
            if not (a <= x <= y <= b):
                raise Refused(f"--mouth-on {x}-{y}: the load is {a}-{b}")
            if not any(Path(w.strip(".,;:()\"'")).is_file() for w in said.split()):
                raise Refused(f"--mouth-on {x}-{y} takes a person's word on a look: who looked, at which sheet of those frames (name its file: the "
                              "tool checks only that a file of that name is there), and that her mouth is open there")
            meta = cw["mouth_on_meta"][idx] if (cw and idx < len(cw["mouth_on_meta"])) else None
            spans.append({"first": x, "last": y, "why": said,
                          "said_by": meta["said_by"] if meta else by, "at": meta["at"] if meta else now,
                          **({"carried": dict(meta["carried"])} if meta else {})})
        # a mouth form other than the check's (`auto`) has no gate to override (the cold reader of 2026-10-15)
        if ref_form["mouth"].get("form") != "auto":
            raise Refused(f"--mouth-on overrides the reference's frame-at-a-time mouth gate, and this load's mouth is {ref_form['mouth'].get('form')!r} "
                          f"({ref_form['mouth'].get('said_by')}): no gate to override")
        # a load of her hair alone draws no face, and a --parts-on span of hair alone none there (as --no-face-on and --eyes-shut-on are held)
        if sorted(parts or []) == ["hair"]:
            raise Refused("--mouth-on on a load of her hair alone: no line of the face is drawn on any of its frames")
        hair = [(x, y) for x, y, these in parts_on or [] if sorted(these) == ["hair"]]
        on_hair = [f"{x}-{y} against the hair-alone span {c}-{d}" for x, y, _ in mouth_on for c, d in hair if x <= d and c <= y]
        if on_hair:
            raise Refused(f"--mouth-on {'; '.join(on_hair)} (--parts-on): no line of the face is drawn there")
        closed = [(s_["first"], s_["last"], what) for what, ss in (("--no-mouth-on", ref_form["mouth"].get("no_mouth_on") or []),
                                                                   ("--no-face-on", ref_form.get("no_face_on") or [])) for s_ in ss]
        clash = [f"{x}-{y} against {what} {c}-{d}" for x, y, _ in mouth_on for c, d, what in closed if x <= d and c <= y]
        if clash:
            raise Refused(f"--mouth-on {'; '.join(clash)}: one says her mouth is open there, the other draws none")
        ref_form["mouth"] = dict(ref_form["mouth"], mouth_on=spans)
    # HER EYES SHUT ON A SPAN, on a person's word (`--eyes-shut-on FIRST-LAST=WHY` with `--eyes-shut-form`, the lead's ruling of
    # 2026-10-15 on 2823b, whose eyes never closed in the render): the pose model's lid points stop at a slit on a shut eye, and nothing
    # else measured tells it (draw_face_points.EYES_SHUT_FORMS). The same terms as --no-mouth-on: the look and its file named, in the load
    import draw_face_points
    if eyes_shut_on or eyes_shut_form is not None:
        if motion == "none":
            raise Refused("--eyes-shut-on acts on the drawn reference, and this load has none (--motion none)")
        if not eyes_shut_on or eyes_shut_form not in draw_face_points.EYES_SHUT_FORMS:
            raise Refused(f"--eyes-shut-on and --eyes-shut-form go together, the form one of {draw_face_points.EYES_SHUT_FORMS}")
        if sorted(parts or []) == ["hair"]:
            raise Refused("--eyes-shut-on on a load of her hair alone: no line of the face is drawn on it (--no-face-on), so there are no eyes to shut")
        spans = []
        for idx, (x, y, said) in enumerate(eyes_shut_on):
            said = " ".join(str(said).split())
            if not (a <= x <= y <= b):
                raise Refused(f"--eyes-shut-on {x}-{y}: the load is {a}-{b}")
            if not any(Path(w.strip(".,;:()\"'")).is_file() for w in said.split()):
                raise Refused(f"--eyes-shut-on {x}-{y} takes a person's word on a look: who looked, at which table or sheet of those frames (name its "
                              "file: the tool checks only that a file of that name is there), and that her eyes are shut there")
            meta = cw["eyes_shut_on_meta"][idx] if (cw and es_from_cw and idx < len(cw["eyes_shut_on_meta"])) else None
            spans.append({"first": x, "last": y, "why": said,
                          "said_by": meta["said_by"] if meta else by, "at": meta["at"] if meta else now,
                          **({"carried": dict(meta["carried"])} if meta else {})})
        faceless = {f for x in ref_form.get("no_face_on") or [] for f in range(x["first"], x["last"] + 1)}
        both = sorted(f for x in spans for f in range(x["first"], x["last"] + 1) if f in faceless)
        if both:
            raise Refused(f"--eyes-shut-on and --no-face-on both name {both[0]}-{both[-1]}: no line of the face is drawn there, so there are no eyes "
                          "to shut; say one of them on those frames")
        ref_form["iris"] = dict(ref_form["iris"], eyes_shut={"form": eyes_shut_form, "spans": spans})
    subjects = [{k: v for k, v in s.items() if k in ("label", "phrase", "pick", "parts", "still", "pose")} for s in preview["subjects"]]
    # The capture's `pieces` for the load's subject, when the plan says one: a key on the subject in the job file, where
    # the chain reads it. The tool sets it for nobody; every load before the chain ran with `pieces=largest` typed into
    # a script, and the first loads through the chain had none (2026-10-13: specks of the part on an arm put a margin there).
    pieces = ""
    if p.get("pieces") is not None:
        if p["pieces"] not in PIECES or not str(p.get("pieces_by") or "").strip():
            raise Refused(f"the plan's `pieces` for {subject!r} is {p['pieces']!r} with `pieces_by` {p.get('pieces_by')!r}: `pieces` is one of {PIECES}, "
                          "and `pieces_by` says who chose it and on what")
        # NO PIECES ON A FACE OR HAIR (`h3_config.NO_PIECES_ON`): the load's parts on any frame, by their classes, so a part of
        # the mouth alone is a face too; the capture refuses the same on the class map, for a job file written by hand
        held = sorted(set(part["parts"] if part else p["parts"]) | {k for x in (part or {}).get("parts_on") or [] for k in x["parts"]})
        table = classes_of()
        if {c for k in held for c in table[k]} & {c for k in h3_config.NO_PIECES_ON for c in table[k]}:
            raise Refused(f"the plan's `pieces={p['pieces']}` for {subject!r} on a load whose part holds {' + '.join(held)}: a face or hair is rightly "
                          "in more than one piece (split by hair, a hand or an arm), and the largest-piece rule drops the others, which leaves the original "
                          "showing inside the new face (the owner, 2026-10-16, docs/wiki/decisions.md, commit e6993b5e). Take `pieces` and `pieces_by` off "
                          f"the plan's {subject!r}: no load of a face or of hair is made with it")
        mine = next(s for s in subjects if s["label"] == subject)
        mine["pieces"], mine["pieces_by"] = p["pieces"], f"the plan ({p['pieces_by']})"
        pieces = (f" THE CAPTURE READS {subject}'s PART WITH pieces={p['pieces']}, from the plan ({' '.join(str(p['pieces_by']).split())}): each frame's largest "
                  "piece of the part is loaded and its other pieces are not; the capture's `files` step prints what that takes away, a frame at a time.")
    # HER HEAD AND HAIR, A PERSON'S MARKS (`head_marks_of`, the plan's `head_marks`): the spans on the load's frames, a key a mark on the
    # subject, where the chain passes each to the capture (`load_chain.MASK_OPTIONS`, `capture_masked_run.her_hair_in_hole`)
    marks = head_marks_of(p, subject, a, b)
    if marks:
        mine = next(s for s in subjects if s["label"] == subject)
        mine.update(marks)
        pieces += (f" A PERSON'S MARKS ON {subject}'s HEAD AND HAIR, from the plan's head_marks: "
                   + "; ".join(f"{k}={v}" for k, v in marks.items() if k != "head_marks_by") + ".")
    # THE SOURCE'S FACIAL HAIR (`source_facial_hair_of`, the plan's word): a key on the subject, which the chain passes to the capture
    # (`load_chain.MASK_OPTIONS`), where the head tube gives back the face's piece it cut below the jaw (`head_tube_parts.face_piece_kept_whole`)
    facial = source_facial_hair_of(p, subject, sorted(set(part["parts"] if part else p["parts"])
                                                       | {k for x in (part or {}).get("parts_on") or [] for k in x["parts"]}))
    if facial:
        mine = next(s for s in subjects if s["label"] == subject)
        mine.update(facial)
        pieces += (f" THE SOURCE SHOWS FACIAL HAIR ON {subject}, from the plan ({facial['facial_hair_by'].removeprefix('the plan: ')}): the "
                   "capture's head tube gives back the piece of the part joined to the face that it would cut at the jaw (facial_hair=keep).")
    # The load's own part, where it names a subset: a key on the load's subject, where the chain reads it for the capture's `parts=`.
    if part:
        mine = next(s for s in subjects if s["label"] == subject)
        mine["parts_file"], mine["parts_of_this_load"], mine["parts_by"] = part["made"][0]["track"], list(part["parts"]), part["by"]
        if part.get("parts_on"):
            mine["parts_on"] = part["parts_on"]
    if on_track:
        mine = next(s for s in subjects if s["label"] == subject)
        mine["parts_file"], mine["parts_held_to_track_px"] = on_track["file"], on_track["grow_px"]
    # Two choices about this one load, each a person's with its words, neither with a default.
    # ZOOMED IN BY DEFAULT (the owner's word of 2026-10-14: every load made with our tools, not a session's choice per load): a part
    # load is rendered `render_region: part`, and the node's own box decides at plan (`video_mask.zoom_box`: under ZOOM_SMALLEST it
    # renders the whole frame itself, and says so). The whole frame is a person's word with its reason: on the load (--whole-frame)
    # or on the subject in the plan (`zoom: false` with `zoom_by`).
    chosen, zoom_block, whole_block = "", None, None
    if zoomed is not None and whole_frame is not None:
        raise Refused("--zoomed and --whole-frame together: one load is rendered one way")
    for flag, words in (("--zoomed", zoomed), ("--whole-frame", whole_frame)):
        if words is not None and not words.strip():
            raise Refused(f"{flag} takes the reason in words, and the word is kept")
    if p.get("zoom", True) is not False and not str(p.get("zoom_by") or "").strip() and "zoom" in p:
        raise Refused(f"the plan's `zoom` for {subject!r} is {p['zoom']!r}: `false` with `zoom_by` (who said so, and why), or leave it out")
    if p.get("zoom") is False and not str(p.get("zoom_by") or "").strip():
        raise Refused(f"the plan says `zoom: false` for {subject!r} with no `zoom_by`: who said so, and why")
    if whole_frame is not None or (p.get("zoom") is False and zoomed is None):
        whole_block = ({"why": " ".join(whole_frame.split()), "said_by": cw["whole_frame_meta"]["said_by"], "at": cw["whole_frame_meta"]["at"],
                        "carried": dict(cw["whole_frame_meta"]["carried"])} if (cw and whole_from_cw and cw.get("whole_frame_meta")) else
                       {"why": " ".join(whole_frame.split()), "said_by": by, "at": now} if whole_frame is not None
                       else {"why": " ".join(str(p["zoom_by"]).split()), "said_by": "the plan", "at": now})
        chosen += (f" RENDERED ON THE WHOLE FRAME, not zoomed in as the chain's default, on {whole_block['said_by']}'s word: {whole_block['why']}.")
    else:
        load["render_region"] = "part"
        zoom_block = ({"why": " ".join(zoomed.split()), "said_by": cw["zoomed_meta"]["said_by"], "at": cw["zoomed_meta"]["at"],
                       "carried": dict(cw["zoomed_meta"]["carried"])} if (cw and zoomed_from_cw and cw.get("zoomed_meta")) else
                      {"why": " ".join(zoomed.split()), "said_by": by, "at": now} if zoomed is not None
                      else {"why": "the chain's default (2026-10-14): rendered zoomed in whenever the node's own box allows; the node decides at plan and "
                                   "renders on the whole frame under its smallest zoom", "said_by": "the chain's default", "at": now})
        chosen += (f" RENDERED ZOOMED IN on what is replaced (`render_region: part`) where the node's box allows, on {zoom_block['said_by']}'s word: "
                   f"{zoom_block['why']}. One box for the whole load, the node's own; the capture reads the node's plan back onto the frame, and the part's "
                   "sizes and floor it prints are still of the whole frame.")
    if (least is None) != (not str(least_why or "").strip()):
        raise Refused("--least and --least-why go together: the capture's floor for a part is a person's number, and the reason is kept beside it")
    if least is not None:
        if int(least) < 0:
            raise Refused(f"--least {least}: a part's least size is 0 (nothing emptied by size) or more, in px of the whole frame")
        mine = next(s for s in subjects if s["label"] == subject)
        l_said = " ".join(least_why.split())
        mine["least"], mine["least_by"] = int(least), (l_said if (cw and least_from_cw) else f"{by}, {now}: {l_said}")
        if cw and least_from_cw:
            mine["least_carried"] = dict(cw["carried_rec"])
        chosen += (f" THE CAPTURE'S FLOOR FOR {subject}'s PART IS SET FOR THIS LOAD: least={int(least)} px of the whole frame, on {by}'s word: "
                   f"{l_said}.")
    # THE PART HELD ON A PERSON'S WORD (`--hold-part FIRST-LAST=WHY`, the lead's ruling of 2026-10-15 on 2823b's 3098-3104 and 5935's
    # 6008 and 6020): frames the capture's grade leaves empty there take the nearest graded part, moved and cut to the track
    # (capture_masked_run.word_shapes), at any distance. The same terms as --no-mouth-on: the look and its file named, each span inside the load. The load is one
    # shot, so a span inside it holds no cut of the record. HELD IN PLACE (`FIRST-LAST@FRAME=WHY`, 2026-10-23, third_fp_5011d's 5039): the
    # person names the frame it is held from, inside the load and outside the span, and the capture places that frame's graded part where
    # it was, cut to the track, refusing it where that frame has none (capture_masked_run.HELD_IN_PLACE); written as `word=FIRST-LAST@FRAME`
    if hold_part:
        spans = []
        for x, y, said, *named in hold_part:
            said = " ".join(str(said).split())
            src = int(named[0]) if named and named[0] is not None else None
            if not (a <= x <= y <= b):
                raise Refused(f"--hold-part {x}-{y}: the load is {a}-{b}")
            if src is not None and not (a <= src <= b):
                raise Refused(f"--hold-part {x}-{y}@{src}: the frame it is held from is {src}, and the load is {a}-{b}")
            if src is not None and x <= src <= y:
                raise Refused(f"--hold-part {x}-{y}@{src}: the frame it is held from lies inside the span it holds; name a frame outside it "
                              "whose part stands")
            if not any(Path(w.strip(".,;:()\"'")).is_file() for w in said.split()):
                raise Refused(f"--hold-part {x}-{y} takes a person's word on a look: who looked, at which sheet of those frames (name its file: the "
                              "tool checks only that a file of that name is there), and that her part belongs there")
            if any(x <= y_ and x_ <= y for x_, y_, _, _ in spans):
                raise Refused(f"--hold-part {x}-{y} overlaps a span already named: name each frame once")
            spans.append((x, y, said, src))
        mine = next(s for s in subjects if s["label"] == subject)
        word_of = [f"{x}-{y}" + (f"@{src}" if src is not None else "") for x, y, _, src in spans]
        mine["word"] = "+".join(word_of)
        mine["word_by"] = "; ".join(f"{w}: {said if idx < n_cw_hp else f'{by}, {now}: {said}'}" for idx, (w, (_, _, said, _)) in enumerate(zip(word_of, spans)))
        if cw and n_cw_hp:
            mine["word_carried"] = dict(cw["carried_rec"])
        chosen += (f" {subject}'s PART IS HELD ON A PERSON'S WORD on {hold_spans_said(mine['word'])}: frames the capture's grade leaves empty there "
                   f"take the nearest graded part, moved and cut to the track" + (
                       "; a span held in place from a frame takes that frame's graded part where it was, unmoved and cut to the track, and "
                       "the capture refuses it where that frame has no graded part" if any(src is not None for *_, src in spans) else "")
                   + f" ({mine['word_by']}).")
    if (drop is None) != (not str(drop_by or "").strip()):
        raise Refused("drop and drop_by go together: the frames where the capture empties the part, and a person's reason")
    if drop is not None:
        dspans = []
        for raw_s in str(drop).split("+"):
            raw_s = raw_s.strip()
            if not raw_s:
                continue
            try:
                dx, dy = (int(v) for v in (raw_s.split("-") * 2)[:2])
            except ValueError:
                raise Refused(f"drop {raw_s!r} is not FIRST-LAST") from None
            if dx > dy or not (a <= dx <= dy <= b):
                raise Refused(f"drop {dx}-{dy} is not inside the load's frames {a}-{b}")
            dspans.append(f"{dx}-{dy}")
        if not dspans:
            raise Refused("drop names no frames")
        mine = next(s for s in subjects if s["label"] == subject)
        d_said = " ".join(str(drop_by).split())
        mine["drop"] = "+".join(dspans)
        mine["drop_by"] = d_said if (cw and drop_from_cw) else f"{by}, {now}: {d_said}"
        if cw and drop_from_cw:
            mine["drop_carried"] = dict(cw["carried_rec"])
        chosen += f" DROP ON THIS LOAD ({mine['drop']}), on {mine['drop_by']}: the capture empties the part there."
    head_tube_block = None
    if head_tube is not None and not str(head_tube).strip():
        raise Refused("--head-tube takes the reason in words, and the word is kept")
    plan_ht = p.get("head_tube") if "head_tube" in p else (p.get("capture") or {}).get("head_tube")
    plan_ht_by = p.get("head_tube_by") or (p.get("capture") or {}).get("head_tube_by")
    if isinstance(plan_ht, dict):
        plan_ht_on = bool(plan_ht.get("on", True))
        plan_ht_why = " ".join(str(plan_ht.get("why") or plan_ht.get("by") or plan_ht_by or f"subjects.{subject}.head_tube").split())
    elif isinstance(plan_ht, str):
        plan_ht_on = bool(plan_ht.strip())
        plan_ht_why = " ".join(plan_ht.split())
    elif plan_ht is True:
        plan_ht_on = True
        plan_ht_why = " ".join(str(plan_ht_by or f"subjects.{subject}.head_tube").split())
    elif plan_ht in (None, False):
        plan_ht_on = False
        plan_ht_why = ""
    else:
        raise Refused(f"the plan's `head_tube` for {subject!r} is {plan_ht!r}: true/false or {{on, why}}")
    if head_tube is not None or plan_ht_on:
        head_tube_block = (
            {"on": True, "why": " ".join(str(head_tube).split()),
             "said_by": cw["head_tube_meta"]["said_by"], "at": cw["head_tube_meta"]["at"],
             "carried": dict(cw["head_tube_meta"]["carried"])}
            if (cw and head_tube_from_cw and cw.get("head_tube_meta")) else
            {"on": True, "why": " ".join(str(head_tube).split()), "said_by": by, "at": now}
            if head_tube is not None else
            {"on": True, "why": plan_ht_why, "said_by": "the plan", "at": now}
        )
        chosen += (f" HEAD TUBE ENABLED FOR THIS LOAD (`--head-tube`), on {head_tube_block['said_by']}'s word: "
                   f"{head_tube_block['why']}.")
    depth_block = None
    if depth is not None and not str(depth).strip():
        raise Refused("--depth takes the reason in words, and the word is kept")
    plan_dp = p.get("depth") if "depth" in p else (p.get("capture") or {}).get("depth")
    plan_dp_by = p.get("depth_by") or (p.get("capture") or {}).get("depth_by")
    if isinstance(plan_dp, dict):
        plan_dp_on = bool(plan_dp.get("on", True))
        plan_dp_why = " ".join(str(plan_dp.get("why") or plan_dp.get("by") or plan_dp_by or f"subjects.{subject}.depth").split())
    elif isinstance(plan_dp, str):
        plan_dp_on = bool(plan_dp.strip())
        plan_dp_why = " ".join(plan_dp.split())
    elif plan_dp is True:
        plan_dp_on = True
        plan_dp_why = " ".join(str(plan_dp_by or f"subjects.{subject}.depth").split())
    elif plan_dp in (None, False):
        plan_dp_on = False
        plan_dp_why = ""
    else:
        raise Refused(f"the plan's `depth` for {subject!r} is {plan_dp!r}: true/false or {{on, why}}")
    if depth is not None or plan_dp_on:
        depth_block = (
            {"on": True, "why": " ".join(str(depth).split()),
             "said_by": cw["depth_meta"]["said_by"], "at": cw["depth_meta"]["at"],
             "carried": dict(cw["depth_meta"]["carried"])}
            if (cw and depth_from_cw and cw.get("depth_meta")) else
            {"on": True, "why": " ".join(str(depth).split()), "said_by": by, "at": now}
            if depth is not None else
            {"on": True, "why": plan_dp_why, "said_by": "the plan", "at": now}
        )
        chosen += (f" DEPTH ENABLED FOR THIS LOAD (`--depth`), on {depth_block['said_by']}'s word: "
                   f"{depth_block['why']}.")
    no_head_track_block = None
    if no_head_track is not None and not " ".join(str(no_head_track).split()):
        raise Refused("--no-head-track takes the reason in words: the SAM 3.1 head track is the chain's default, and the word is kept")
    if no_head_track is not None:
        mine = next(s for s in subjects if s["label"] == subject)
        mine["track_head"] = False
        no_head_track_block = (
            {"on": False, "why": " ".join(str(no_head_track).split()),
             "said_by": cw["no_head_track_meta"]["said_by"], "at": cw["no_head_track_meta"]["at"],
             "carried": dict(cw["no_head_track_meta"]["carried"])}
            if (cw and no_head_track_from_cw and cw.get("no_head_track_meta")) else
            {"on": False, "why": " ".join(str(no_head_track).split()), "said_by": by, "at": now}
        )
        chosen += (f" THE SAM 3.1 HEAD TRACK IS OFF FOR THIS LOAD (`--no-head-track`), on {no_head_track_block['said_by']}'s word: "
                   f"{no_head_track_block['why']}.")
    if no_closed_lip_line is not None and not " ".join(str(no_closed_lip_line).split()):
        raise Refused("--no-closed-lip-line takes the reason in words: --closed-lip-line is the chain's default, and the word is kept")
    if no_closed_lip_line is not None:
        if motion == "none":
            raise Refused("--no-closed-lip-line acts on the drawn reference, and this load has none (--motion none)")
        ncl_meta = cw["no_closed_lip_line_meta"] if (cw and no_cl_from_cw and cw.get("no_closed_lip_line_meta")) else None
        cl_off_block = {
            "on": False,
            "why": " ".join(str(no_closed_lip_line).split()),
            "said_by": ncl_meta["said_by"] if ncl_meta else by,
            "at": ncl_meta["at"] if ncl_meta else now,
            **({"carried": dict(ncl_meta["carried"])} if ncl_meta else {}),
        }
        ref_form["mouth"] = dict(ref_form["mouth"], closed_lip_line=cl_off_block)
        chosen += (f" CLOSED-LIP LINE OFF FOR THIS LOAD (`--no-closed-lip-line`), on {cl_off_block['said_by']}'s word: "
                   f"{cl_off_block['why']}.")
    # what the composite kept, saved for the review clips (the song node's `save_kept`): a chain default, the job's switch (2026-10-15)
    load["save_kept"] = True
    # THE VOICE (the owner's word of 2026-10-15 on 5675b: the plan says per span whether a subject sings, talks unheard or is silent; the
    # text step reads it, `load_text.voice_of`, in place of the voice table's guess; and under `closed_lip_line`, M17: a silent window
    # draws no mouth at all so the grey head's own lips stay untouched)
    import load_text
    hair_alone = sorted(parts or []) == ["hair"] or motion == "none"
    try:
        voice = None if hair_alone else load_text.voice_of(p, a, b)
    except load_text.Refused as exc:
        raise Refused(str(exc)) from None
    cl_cfg = ref_form["mouth"].get("closed_lip_line")
    cl_on = cl_cfg.get("on", True) if isinstance(cl_cfg, dict) else bool(cl_cfg)
    if cl_on and ref_form["mouth"].get("said_by") == "the chain's default" and voice and voice.get("voice") == "silent":
        ref_form["mouth"] = {
            **ref_form["mouth"],
            "form": "left out",
            "said_by": voice["by"],
            "why": f"silent window on {voice['first']}-{voice['last']} ({voice['by']}): no mouth drawn",
        }
    carried = defaults_carried(p, load, subjects, subject, kept, ref_form, zoom_block, whole_block,
                               head_tube_block=head_tube_block, depth_block=depth_block, preview=preview,
                               no_head_track_block=no_head_track_block)
    if motion == "none":
        # no reference: the rows of the drawn reference say nothing of this load, and the text is made from no record
        carried = [x for x in carried if x["what"] not in ("the reference held to the subject's track", "the eye", "the mouth", "the reference's crop")]
        for x in carried:
            if x["what"] == "the text":
                x["carries"] = ("generated by bench/load_text.py from the plan's words and the voice table, with no reference video (no <Video 1>, "
                                "no <Subject 2>), with no place-and-action sentence (load_text.NO_SCENE)")
        m_from = " ".join(str(motion_why).split()) if (cw and motion_from_cw) else f"{by}, {now}: {' '.join(str(motion_why).split())}"
        carried.append({"what": "the reference video", "carries": "none: the source's own picture carries her movement",
                        "from": m_from})
    carried.append({"what": "the voice", "carries": ("not read: a load of the hair alone, whose text says nothing of a voice" if hair_alone else
                                                     f"{voice['voice']}, on {voice['first']}-{voice['last']} (the plan's, in place of the voice table)"
                                                     if voice else "the voice table's: the sung line where it hears the song in the load and the "
                                                                   "plan's text says the subject sings")
                    + ("" if hair_alone else "; read by the generated text (bench/load_text.py), not by a text written by hand"),
                    "from": "the chain" if hair_alone or not voice else voice["by"]})
    # THE SEAM OF A LOAD LONGER THAN ONE WINDOW (`seam_of`)
    window, seam_row = seam_of(n, a, seam, seam_why, by, now)
    if window is not None:
        load["window"] = window
    if seam_row is not None:
        if cw and seam_from_cw:
            seam_row["from"] = " ".join(str(seam_why).split())
        carried.append(seam_row)
    noise_row = context_noise_of(n, context_noise, context_noise_why, by, now)
    if noise_row is not None:
        load["context_noise"] = float(context_noise)
        if cw and cn_from_cw:
            noise_row["from"] = " ".join(str(context_noise_why).split())
        carried.append(noise_row)
    if up_to_by is not None:
        if load.get("render_region") != "part":
            raise Refused(f"--zoom-up-to on a load rendered on the whole frame: it acts only on a zoomed load")
        carried.append({"what": "the most the node may enlarge this zoomed load", "carries": f"{up_to:g} (the node's own is {ZOOM_NODES_OWN:g}): "
                        "past it a small head is zoomed until it is the node's target height", "from": f"{by}, {now}: {up_to_by['why']}"})
    if seed_by is not None:
        carried.append({"what": "the seed", "carries": f"{job_seed} (h3_config.SEED is {h3_config.SEED}): the song node samples window i at seed + i",
                        "from": f"{by}, {now}: {seed_by['why']}"})
    region_margin = h3_config.MASKED_LOAD_MARGIN[region]
    if margin_by is not None:
        carried.append({"what": "the margin", "carries": f"{own_margin} px (h3_config.MASKED_LOAD_MARGIN for {region!r} is {region_margin}): the Masked "
                        "Source grows the load's region by it, in px of the picture it renders (the zoomed one on a zoomed load)",
                        "from": f"{by}, {now}: {margin_by['why']}"})
    note = (f"Written by bench/shot_to_load.py for {by}, {now}. ONE LOAD: {subject}, {renders(region, parts)}, source frames {a} to {b} ({n} frames) of the shot "
            f"{shot['first']}-{shot['last']}, its masks from the preview {preview['name']} (job.preview), "
            + ("with no reference video (motion none, below). " if motion == "none" else
               "in the frozen form: the body model cut to the head with the face points drawn, made on these frames by this tool's `reference` step. ")
            + f"Its settings are in `defaults`, from workflows/h3_config.py. "
            f"THE TEXT IS NOT WRITTEN: {name}.txt beside this file is a person's, and texts.{name}.by says who wrote it."
            + pieces + part_said + kept_said + on_track_said + gone_said + chosen
            + (f" THE SEED IS {job_seed} (h3_config.SEED is {h3_config.SEED}), on {by}'s word ({now}): {seed_by['why']}." if seed_by else "")
            + (f" ZOOMED UP TO {up_to:g} (the node's own is {ZOOM_NODES_OWN:g}), on {by}'s word ({now}): {up_to_by['why']}." if up_to_by else "")
            + (f" THE MARGIN IS {own_margin} px (h3_config.MASKED_LOAD_MARGIN for {region!r} is {region_margin}), on {by}'s word ({now}): "
               f"{margin_by['why']}." if margin_by else "")
            + (f" CUT BY A LOOK: {cut_said}" if cut else "")
            + (f" ITS REFERENCE GOES TO {ref_folder.name}, A NEW FOLDER: {ref_why}; a new one." if ref_why else "")
            + (f" NO REFERENCE VIDEO (motion none): the reference step makes nothing for this load and the text names no video: "
               f"{' '.join(str(motion_why).split())} ({by}, {now})." if motion == "none" else ""))
    if cw is not None:
        print(f"WORDS CARRIED FROM {cw['old_name']}:")
        print(f"  source: {cw['old_file']} (sha256 {cw['old_sha256']})")
        for line in cw["carried_said"] or ["none"]:
            print(f"  {line}")
        if cw["overridden_said"]:
            print(f"WORDS OVERRIDDEN FROM {cw['old_name']}:")
            for line in cw["overridden_said"]:
                print(f"  {line}")
        if cw["left_said"]:
            print(f"WORDS LEFT OUT FROM {cw['old_name']}:")
            for line in cw["left_said"]:
                print(f"  {line}")
        if cw["retired_said"]:
            print(f"RETIRED KEYS NOT CARRIED FROM {cw['old_name']}:")
            for line in cw["retired_said"]:
                print(f"  {line}")
    if ncw_why:
        print(f"NO WORDS CARRIED: {ncw_why} ({by}, {now})")
    src_frames = shots.get("source_frames") if shots.get("source_frames") is not None else preview.get("source_frames")
    return {"job": {"name": name, "note": note, "source": source or preview["source"], "first": a, "frames": n, "canvas": list(preview["canvas"]),
                    **({"source_frames": int(src_frames)} if src_frames is not None else {}),
                    # the shot's own frames from the shots record, which a job's span can be narrower than: where a short load's
                    # padding leaves THIS, it crosses a cut and is held (`masked_job_graph.shot_span`, `held_context`)
                    "shot": [int(shot["first"]), int(shot["last"])],
                    "output": OUTPUT_KEY.format(video=video, session=by), "subjects": subjects,
                    "left_out": [{"subject": subject, "first": x, "frames": y - x + 1, "why": said,
                                  **({"carried": dict(cw["carried_rec"])} if cw and ((x == shot["first"] and wb_from_cw) or (y == shot["last"] and wa_from_cw)) else {})}
                                 for x, y, said in cut],
                    "capture": f"data/{day}_{name}", "preview": preview_folder,
                    **({"reference_folder": {"folder": ref_folder.name, "why": f"{ref_why}; a new one"}} if ref_why else {}),
                    "texts": {name: {"file": f"{name}.txt", "frames": [a, b], "by": ""}},
                    **({"keep_outs": kept} if kept else {}),
                    # the plan's restore spans on the load's frames, for the gate (`load_chain.told_to_the_gate`): what is given back at
                    # assembly is no region over the subject's own things that nothing answers
                    **({"restore": restore} if (restore := [dict(r, by=p.get("restore_by")) for r in p.get("restore") or []
                                                            if r["first"] <= b and r["last"] >= a]) else {}),
                    **({"part": part} if part else {}), **({"part_on_track": on_track} if on_track else {}), **({"zoomed": zoom_block} if zoom_block else {}),
                    **({"whole_frame": whole_block} if whole_block else {}),
                    **({"head_tube": head_tube_block} if head_tube_block else {}),
                    **({"depth": depth_block} if depth_block else {}),
                    **({"sam_head_track": no_head_track_block} if no_head_track_block else {}),
                    **({"track_not": gone} if gone else {}),
                    **({"carried_words_from": {**cw["carried_rec"], **({"left_words": cw["left_words"]} if cw["left_words"] else {})}} if cw else {}),
                    **({"no_carry_words": {"why": ncw_why, "by": by, "at": now}} if ncw_why else {}),
                    "reference": ({"form": "none", "why": (" ".join(str(motion_why).split()) if (cw and motion_from_cw) else f"{by}, {now}: {' '.join(str(motion_why).split())}"),
                                   "made": None, **({"carried": dict(cw["carried_rec"])} if (cw and motion_from_cw) else {})} if motion == "none" else
                                  {"form": "drawn", "why": "the form for a face, head or hair replacement (AGENTS.md, Settled)", "made": None,
                                   "mouth": ref_form["mouth"], "iris": ref_form["iris"],
                                   **({"darken_head": ref_form["darken_head"]} if ref_form.get("darken_head") else {}),
                                   **({"crop": ref_form["crop"]} if ref_form.get("crop") else {}),
                                   **({"head_track": no_head_track_block or ref_form["head_track"]} if (no_head_track_block or ref_form.get("head_track")) else {}),
                                   **({"no_face_on": ref_form["no_face_on"]} if ref_form.get("no_face_on") else {})}),
                    "others_present": others, "defaults": carried},
            "defaults": {**h3_config.MASKED_LOAD_DEFAULTS, "margin": region_margin if own_margin is None else own_margin,
                         **({"margin_by": margin_by} if margin_by else {}),
                         "seed": h3_config.SEED if job_seed is None else job_seed, **({"seed_by": seed_by} if seed_by else {}),
                         **({"zoom_up_to": up_to, "zoom_up_to_by": up_to_by} if up_to_by else {})},
            "loads": [load]}


#: WHERE A FACE MAY NOT SHOW, two cues read off the pose model's points and printed under the reference's figures for whoever looks at its
#: sheet (the lead's ruling of 2026-10-15, after the hall shot's references drew eyes on the backs of two heads). Data, never a default:
#: on every head reference on the share that day the ears' cue also fires on faces that show (in profile, 690 and 698 on the sheet;
#: three-quarter, 5118), the scores' cue also reads a hand over a face, and neither sees second_5461_5497's head (the pose model read it
#: as facing the camera). Measured by `face_cues` itself:
#: internal/claude/2026-10-15_mrlol_scripts/face_cues_all.txt (face_cues_all.py), the frames in turned_away_frames/sheet.jpg.
#: EARS REVERSED: the right ear right of the left ear in the picture, as on a head turned away. Reasoned: the sign alone; its size is
#: printed, never judged. Read only where both ears and both shoulders score at least `POINT_SCORE_FLOOR`.
EARS_REVERSED_ABOVE = 0.0
#: FACE POINTS' MEAN SCORE under this. A starting value, measured on lead_1419_1465: hidden behind the dad on 1419-1422 at 0.14-0.20, her
#: face showing at 0.39 on 1423; with her back turned on 1443-1465 it reads 0.32-0.80, above it, so this cue misses a head turned away.
#: It also reads a hand over the face (second_1025_1038, every frame).
FACE_SCORE_FLOOR = 0.3
#: The least score of an ear or shoulder point for the ears' cue to be read on a frame. Reasoned, the face's floor reused: points the
#: pose model barely saw collapse together (lead_5935_6020's shoulders 1-12 px apart at scores 0.03-0.42 made the ratio 347 before it).
POINT_SCORE_FLOOR = 0.3
#: The shoulders' width under this is taken as this, so no frame divides by zero. Reasoned: such a frame's points are collapsed, and
#: `POINT_SCORE_FLOOR` leaves nearly all of them unread.
LEAST_SHOULDERS_PX = 1.0
FACE_CUES_NEED = ("left_ear", "right_ear", "left_shoulder", "right_shoulder")


def face_cues(points: Path) -> dict:
    """Where a face may not show along a load, by `EARS_REVERSED_ABOVE` and `FACE_SCORE_FLOOR`, as runs of source frames with the figures;
    and what each cannot see. Nothing is decided from them: a face's lines are left out on a person's word (`job --no-face-on`). A points
    file without what the cues read says so in `not_read`, and the reference step goes on."""
    import numpy as np
    d = np.load(points, allow_pickle=True)
    lacking = [k for k in ("names", "keypoints", "scores", "first_frame") if k not in d.files]
    names = [str(n) for n in d["names"]] if "names" in d.files else []
    lacking += [n for n in FACE_CUES_NEED if names and n not in names]
    if lacking:
        return {"not_read": f"the points file holds no {', '.join(lacking)}"}
    kp, sc, first = d["keypoints"], d["scores"], int(d["first_frame"])
    at = names.index
    face = [i for i, n in enumerate(names) if any(w in n.lower() for w in ("eye", "brow", "lip"))]
    if not face:
        return {"not_read": "the points file names no eye, brow or lip point"}
    shoulders = np.maximum(np.abs(kp[:, at("right_shoulder"), 0] - kp[:, at("left_shoulder"), 0]), LEAST_SHOULDERS_PX)
    ears = (kp[:, at("right_ear"), 0] - kp[:, at("left_ear"), 0]) / shoulders
    seen = (sc[:, [at(n) for n in FACE_CUES_NEED]] >= POINT_SCORE_FLOOR).all(1)
    score = sc[:, face].mean(1)

    def runs(on: np.ndarray, value: np.ndarray, k: float = 1.0) -> list[dict]:
        out = []
        for t in np.flatnonzero(on):
            if out and out[-1]["last"] == first + t - 1:
                out[-1]["last"] = first + int(t)
            else:
                out.append({"first": first + int(t), "last": first + int(t)})
        for r in out:
            v = value[r["first"] - first:r["last"] - first + 1] * k
            r.update(least=round(float(v.min()), 2 if k == 1.0 else 1), most=round(float(v.max()), 2 if k == 1.0 else 1))
        return out
    return {"ears_reversed": runs(seen & (ears > EARS_REVERSED_ABOVE), ears, 100.0),
            "ears_reversed_is": f"the right ear's x less the left ear's over the shoulders' width, x100 (EARS_REVERSED_ABOVE), read where the ears "
                                f"and shoulders score at least {POINT_SCORE_FLOOR} (POINT_SCORE_FLOOR)",
            "ears_not_read_on": runs(~seen, np.zeros(len(seen))),
            "face_score_under_the_floor": runs(score < FACE_SCORE_FLOOR, score),
            "face_score_is": f"the mean of the face points' scores, under {FACE_SCORE_FLOOR} (FACE_SCORE_FLOOR)",
            "cannot_see": "a face in profile or three-quarter reads ears reversed as a head turned away does, and a hand over the face reads as a "
                          "face not seen; a head turned away that the pose model reads as facing the camera (second_5461_5497's) reads neither; "
                          "the ears are not read where the pose model barely saw them or the shoulders, which is every frame of a close-up "
                          "whose shoulders are out of the picture (lead_2823_3097)"}


def face_cue_lines(cues: dict) -> list[str]:
    """`face_cues` as the lines the reference step prints under its figures."""
    if "not_read" in cues:
        return [f"  not read: {cues['not_read']}"]
    spans = lambda rs, figures=True: ", ".join((f"{r['first']}-{r['last']}" if r["first"] != r["last"] else str(r["first"]))  # noqa: E731
                                               + (f" ({r['least']}..{r['most']})" if figures else "") for r in rs) or "none"
    return [f"  ears reversed, {cues['ears_reversed_is']}: {spans(cues['ears_reversed'])}",
            f"  the ears not read on: {spans(cues['ears_not_read_on'], False)}",
            f"  {cues['face_score_is']}: {spans(cues['face_score_under_the_floor'])}",
            f"  cannot see: {cues['cannot_see']}"]


def figures_of(points: Path, scale: float, cues=face_cues) -> dict:
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
            "where_a_face_may_not_show": cues(points),
            "at_the_canvas": at(1.0), "in_the_reference": {"enlarged": round(float(scale), 3), **at(float(scale))},
            "cannot_see": "whether the points are on the face: the model places and scores lid points on the back of a head and on a face a few pixels tall"}


def reference_block(form: str, why: str, by: str, now: str, record: Path, points: Path, figures: dict) -> dict:
    return {"form": form, "why": why, "said_by": by, "made": now, "record": str(record), "points": str(points), "figures": figures}


#: THE REFERENCE'S FORM BY DEFAULT (the owner's word of 2026-10-14: "make these our defaults and stop forgetting the tools we
#: built"; 2026-10-19 on 0278f and 1340d: closed_lip_line on by default, darken_head off): the mouth by the lip check
#: (`head_reference.mouth_check`) with closed lips drawn as one parting line, the mouth fitted above the chin, silent windows
#: leaving the mouth out, and every mark clipped to the grey head's silhouette (`head_reference --closed-lip-line`), and the
#: eye in the contrast form (white sclera, black iris, 264c0840). A subject's plan may say otherwise (`reference: {mouth, iris,
#: closed_lip_line, why}`), and a load at its job (`--no-closed-lip-line WHY`) or reference step (`--no-closed-lip-line
#: --form-why WHY`), each with its reason. An open mouth under the closed-lip line is drawn as one white outline round the inner-lip
#: ring filled dark unless the subject's plan says `reference: {open_mouth, why}` (`draw_face_points.OPEN_MOUTH_FORMS`; the owner,
#: 2026-10-22, on lead_fp_1156d: the lead's outlined, on the class map's lips and to the face's size, the other subjects as they are),
#: carried to `head_reference --open-mouth`. A mouth the closed-lip line calls closed draws nothing, the body model's own lips, unless the
#: subject's plan says `reference: {closed_mouth: "outlined", why}` (`draw_face_points.CLOSED_MOUTH_FORMS`; the owner, 2026-10-23, on
#: second_hfp_5114i: the lips' outline and the line where they meet, placed as the outlined open mouth is), carried to `head_reference
#: --closed-mouth`. The reference is cut to the head's box (`head_reference.CROP_FORMS`, `head`) unless the
#: subject's plan says `reference: {crop: "face", why}` (2026-10-22: the face larger in the picture, after 0278j/k on the head box) or
#: `reference: {crop: "cranium", why}` (the same day: the body model's grey head, not the real hair), carried to `head_reference --crop
#: face|cranium --crop-why`.
REFERENCE_DEFAULT = {"mouth": "auto", "iris": "contrast", "closed_lip_line": True}


def head_marks_of(p: dict, subject: str, a: int, b: int) -> dict:
    """A PERSON'S MARKS on a subject's head and hair (2026-10-22), from the plan's `subjects.<s>.head_marks`: a list of
    {first, last, mark, by} in source frames, `mark` one of `capture_masked_run.HEAD_MARK_KEYS` (`hidden`: her head is not in view
    where SAM lost it; `hers` / `not hers`: the unclaimed hair the capture asked about is hers or someone else's; `hers` on a lost
    head also holds her nearest whole head). The owner marks the frames on the page the capture's refusal names; the mark is
    written here, the one home of a person's span words on a subject, as `restore`, `voice` and `clothing` are. Returns the subject's
    job keys for the spans on the load's frames `a`-`b`, cut to them (`head_hidden`, `hair_hers`, `hair_not_hers` as
    FIRST-LAST+FIRST-LAST, and `head_marks_by`), or {}; refused: a mark not of the list, a span backwards, one with no `by`, and two
    marks on one frame."""
    import capture_masked_run as cap
    out: dict[str, list[str]] = {}
    said, seen = [], {}
    for x in p.get("head_marks") or []:
        if x.get("mark") not in cap.HEAD_MARK_KEYS:
            raise Refused(f"the plan's head_marks for {subject!r} has mark {x.get('mark')!r}: one of {tuple(cap.HEAD_MARK_KEYS)}")
        lo, hi = int(x["first"]), int(x["last"])
        if lo > hi:
            raise Refused(f"the plan's head_marks for {subject!r} has {lo}-{hi}, which runs backwards")
        if not str(x.get("by") or "").strip():
            raise Refused(f"the plan's head_marks for {subject!r} on {lo}-{hi} says no `by`: who marked it, and where")
        lo, hi = max(lo, a), min(hi, b)
        if lo > hi:
            continue
        for f in range(lo, hi + 1):
            if f in seen and seen[f] != x["mark"]:
                raise Refused(f"the plan's head_marks for {subject!r} mark {f} both {seen[f]!r} and {x['mark']!r}")
            seen[f] = x["mark"]
        out.setdefault(cap.HEAD_MARK_KEYS[x["mark"]], []).append(f"{lo}-{hi}")
        said.append(f"{x['mark']} {lo}-{hi} ({' '.join(str(x['by']).split())})")
    return {**{k: "+".join(v) for k, v in out.items()}, "head_marks_by": "the plan: " + "; ".join(said)} if out else {}


def context_of(p: dict, subject: str) -> dict:
    """WHAT THE CONTEXT FRAMES SHOW for this subject's loads (the owner's ask of 2026-10-22: one system for more than one
    scenario), from the plan's `subjects.<s>.context` with `context_by`: `as filmed` (the default, and the key left out) or
    `face covered` (`masked_job_graph.CONTEXT_SHOWS`; the node's `video_mask.CONTEXT_SHOWS` has the rule). The real frames a
    short load is padded with show the original subject unmasked under the first, and the subject's region held open under
    the second, for a subject whose original feature must not be in the window (the beard rule, docs/masked_v2v/AGENTS.md
    §2). Returns {"value", "said_by", "why"}; refused: a value that is neither, or one with no `context_by`."""
    import masked_job_graph
    if "context" not in p:
        if str(p.get("context_by") or "").strip():
            raise Refused(f"the plan has `context_by` for {subject!r} and no `context`: add `context`, `as filmed` or `face covered`, or take both out")
        return {"value": masked_job_graph.CONTEXT_SHOWS[0], "said_by": "the chain's default",
                "why": "the real frames a short load is padded with are untouched context (2026-10-21)"}
    if p["context"] not in masked_job_graph.CONTEXT_SHOWS:
        raise Refused(f"the plan's `context` for {subject!r} is {p['context']!r}: it is `as filmed` or `face covered`")
    if not str(p.get("context_by") or "").strip():
        raise Refused(f"the plan says `context: {p['context']}` for {subject!r} with no `context_by`: who said so, and why")
    return {"value": p["context"], "said_by": "the plan", "why": " ".join(str(p["context_by"]).split())}


def source_facial_hair_of(p: dict, subject: str, held: list[str]) -> dict:
    """THE SOURCE'S FACIAL HAIR (2026-10-23), from the plan's `subjects.<s>.source_facial_hair` with `source_facial_hair_by`: the
    source actor shows facial hair the replacement does not have, so the capture's head tube gives back the piece of the part joined
    to the face that it cuts at the jaw (`head_tube_parts.face_piece_kept_whole`; the beard rule, docs/masked_v2v/AGENTS.md §2: on
    third_fp_5011b the tube cut 986-1607 px a frame of the goatee the preview's part held). `held` is the load's parts. Returns the
    subject's job keys (`facial_hair: keep`, `facial_hair_by`) or {}; refused: a value that is not true or false, a value with no
    `_by` or a `_by` with no value, and `true` on a load whose parts are not face_and_neck without hair (built and measured on a
    face-only part; a part of hair takes the whole head tube already, and a `--motion none` load is one of hair alone)."""
    said, by = p.get("source_facial_hair"), " ".join(str(p.get("source_facial_hair_by") or "").split())
    if said is None:
        if by:
            raise Refused(f"the plan has `source_facial_hair_by` for {subject!r} and no `source_facial_hair`: add `source_facial_hair: true` "
                          "or take both out")
        return {}
    if not isinstance(said, bool):
        raise Refused(f"the plan's `source_facial_hair` for {subject!r} is {said!r}: `true` or `false`, with `source_facial_hair_by`")
    if not by:
        raise Refused(f"the plan says `source_facial_hair: {str(said).lower()}` for {subject!r} with no `source_facial_hair_by`: who said so, and where")
    if not said:
        return {}
    if "hair" in held or "face_and_neck" not in held:
        raise Refused(f"the plan says `source_facial_hair: true` for {subject!r} on a load whose part is {' + '.join(held)}: it is built for a "
                      "face-only part (face_and_neck without hair); a part of hair takes the whole head tube, the beard with it")
    return {"facial_hair": "keep", "facial_hair_by": f"the plan: {by}"}


def reference_form(p: dict, subject: str) -> dict:
    """The mouth and the eye a subject's references are drawn with, and whose word each is."""
    import draw_face_points
    import head_reference
    said = p.get("reference") or {}
    if said and not str(said.get("why") or "").strip():
        raise Refused(f"the plan's `reference` for {subject!r} says no `why`: who said so, and why")
    out = {}
    for key, choices in (("mouth", head_reference.MOUTH_FORMS), ("iris", draw_face_points.IRIS_FORMS)):
        if key in said:
            if said[key] not in choices:
                raise Refused(f"the plan's `reference.{key}` for {subject!r} is {said[key]!r}: one of {choices}")
            out[key] = {"form": said[key], "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
        else:
            out[key] = {"form": REFERENCE_DEFAULT[key], "said_by": "the chain's default", "why": "REFERENCE_DEFAULT (2026-10-14)"}
    if "closed_lip_line" in said:
        if not isinstance(said["closed_lip_line"], bool):
            raise Refused(f"the plan's `reference.closed_lip_line` for {subject!r} is {said['closed_lip_line']!r}: `true` or `false` with `why`, or leave it out")
        if said["closed_lip_line"] is True and out["mouth"]["form"] not in ("auto", "left out"):
            raise Refused(f"the plan's `reference.closed_lip_line` for {subject!r} acts on `mouth: 'auto'` (or `'left out'`); got {out['mouth']['form']!r}")
        out["mouth"]["closed_lip_line"] = {"on": said["closed_lip_line"], "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
    elif "mouth" not in said and REFERENCE_DEFAULT.get("closed_lip_line"):
        out["mouth"]["closed_lip_line"] = {"on": True, "said_by": "the chain's default", "why": "REFERENCE_DEFAULT (2026-10-19)"}
    if "darken_head" in said:
        if said["darken_head"] is not True:
            raise Refused(f"the plan's `reference.darken_head` for {subject!r} is {said['darken_head']!r}: `true` with `why`, or leave it out")
        out["darken_head"] = {"said_by": "the plan", "why": " ".join(str(said["why"]).split())}
    if "open_mouth" in said:
        if said["open_mouth"] not in draw_face_points.OPEN_MOUTH_FORMS:
            raise Refused(f"the plan's `reference.open_mouth` for {subject!r} is {said['open_mouth']!r}: one of {draw_face_points.OPEN_MOUTH_FORMS}")
        if not (out["mouth"].get("closed_lip_line") or {}).get("on"):
            raise Refused(f"the plan's `reference.open_mouth` for {subject!r} is a form of the closed-lip line's drawing, which is off for them")
        out["mouth"]["open_mouth"] = {"form": said["open_mouth"], "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
    if "closed_mouth" in said:
        if said["closed_mouth"] not in draw_face_points.CLOSED_MOUTH_FORMS:
            raise Refused(f"the plan's `reference.closed_mouth` for {subject!r} is {said['closed_mouth']!r}: one of {draw_face_points.CLOSED_MOUTH_FORMS}")
        if not (out["mouth"].get("closed_lip_line") or {}).get("on"):
            raise Refused(f"the plan's `reference.closed_mouth` for {subject!r} is a form of the closed-lip line's drawing, which is off for them")
        out["mouth"]["closed_mouth"] = {"form": said["closed_mouth"], "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
    if "crop" in said:
        if said["crop"] not in head_reference.CROP_FORMS:
            raise Refused(f"the plan's `reference.crop` for {subject!r} is {said['crop']!r}: one of {head_reference.CROP_FORMS}")
        if said["crop"] == "head" and (said.get("head_track") is False or isinstance(said.get("head_track"), dict)):
            raise Refused(f"the plan's `reference.crop` for {subject!r} is 'head', which reads the head track, and its `reference.head_track` turns it off")
        out["crop"] = {"form": said["crop"], "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
    if "head_track" in said:
        ht_val = said["head_track"]
        if isinstance(ht_val, dict):
            if ht_val.get("on") is not False or not str(ht_val.get("why") or said.get("why") or "").strip():
                raise Refused(f"the plan's `reference.head_track` for {subject!r} is {ht_val!r}: `false` or `{{on: false, why}}` with `why`, or leave it out")
            out["head_track"] = {"on": False, "said_by": "the plan", "why": " ".join(str(ht_val.get("why") or said.get("why")).split())}
        elif ht_val is False:
            out["head_track"] = {"on": False, "said_by": "the plan", "why": " ".join(str(said["why"]).split())}
        else:
            raise Refused(f"the plan's `reference.head_track` for {subject!r} is {ht_val!r}: `false` with `why`, or leave it out")
    return out


def memory_selection_carried(preview: dict | None) -> tuple[str, str]:
    """What the trackers of the preview a load's masks come from ran with, and whose word: the line of THE DEFAULTS."""
    said = (preview or {}).get("sam_memory_selection")
    if preview is None:
        return "not read (no preview handed in)", "shot_to_load.py::defaults_carried"
    if said is None:
        return ("not recorded: the preview was made before its job file recorded it (2026-10-17), so its trackers ran with ComfyUI's "
                "last-six-frames memory unless its graph says otherwise", f"the preview {preview.get('name')}'s job file")
    if said.get("on"):
        return ("on: the preview's trackers kept as memory only past frames where the person was seen (sam31_corrections.py, "
                "Meta's rule)", f"{said.get('said_by')}: {said.get('why')}")
    return ("OFF in the preview this load's masks come from, turned off on a person's word: ComfyUI's last-six-frames memory",
            f"{said.get('said_by')}, {said.get('at')}: {said.get('why')}")


def head_track_carried(preview: dict | None, load_head_track: dict | None = None) -> tuple[str, str]:
    """What the SAM 3.1 head track is for this load, from its own `--no-head-track` or the preview its masks come from: the line of THE DEFAULTS."""
    if load_head_track is not None and not load_head_track.get("on", True):
        return ("OFF for this load, turned off on a person's word: the capture does not read the preview's head track",
                f"{load_head_track.get('said_by')}, {load_head_track.get('at')}: {load_head_track.get('why')}")
    if preview is None:
        return "not read (no preview handed in)", "shot_to_load.py::defaults_carried"
    said = (preview or {}).get("sam_head_track")
    if said is None:
        return ("on when the preview saved a head track (`_head_track_*.mkv`), else not recorded (preview made before 2026-10-19)",
                f"the preview {preview.get('name')}'s job file")
    if said.get("on"):
        return ("on: the preview's trackers propagated the subject's head with SAM 3.1 (`track_head=True`) and the capture reads it "
                "as the head tube's seed mask (`head_tube_parts.py::head_tube_for_capture`)",
                f"{said.get('said_by')}: {said.get('why')}")
    return ("OFF in the preview this load's masks come from, turned off on a person's word",
            f"{said.get('said_by')}, {said.get('at')}: {said.get('why')}")


def defaults_carried(p: dict, load: dict, subjects: list, subject: str, kept: dict | None, ref_form: dict, zoom_block, whole_block,
                     head_tube_block: dict | None = None, depth_block: dict | None = None, preview: dict | None = None,
                     no_head_track_block: dict | None = None) -> list[dict]:
    """THE DEFAULTS THIS LOAD CARRIES (the owner's word of 2026-10-14), one line each, with where each comes from: printed by the
    job step and kept in the job file (`job.defaults`). `preview` is the job block of the preview the load's masks come from."""
    import h3_config
    mine = next(s for s in subjects if s["label"] == subject)
    names = [k["name"] for k in (kept or {}).get("made", [])]
    context = context_of(p, subject)
    band = load.get("hair_band")
    mouth, iris = ref_form["mouth"], ref_form["iris"]
    cl_block = mouth.get("closed_lip_line")
    cl_on = cl_block.get("on", True) if isinstance(cl_block, dict) else bool(cl_block)
    cl_off = isinstance(cl_block, dict) and not cl_block.get("on", True)
    cl_from = (
        f"; closed_lip_line: {cl_block['said_by']}" + (f", {cl_block['at']}" if cl_block.get("at") else "") + f": {cl_block['why']}"
        if isinstance(cl_block, dict) else ""
    )
    lines = [
        ("the reference held to the subject's track", "the pose pass reads the face inside the subject's track (pose_along_shot HOLD_GROW_PX); "
         "a short run of frames off it takes its neighbours' points (head_reference FILL_GAP)", "the chain"),
        ("the eye", {"contrast": "white sclera and black iris inside the lids", "between the lids": "the iris ring between the lids",
                     "filled": "a dark iris disc inside the lids", "whole": "the whole iris ring"}[iris["form"]]
         + ("; DARKENED HEAD MODEL (0.67x luma)" if ref_form.get("darken_head") else "")
         + "".join(f"; SHUT on {s['first']}-{s['last']} (drawn {iris['eyes_shut']['form']!r}), on a person's word" for s in (iris.get("eyes_shut") or {}).get("spans") or []),
         f"{iris['said_by']}: {iris['why']}"
         + (f"; darken_head: {ref_form['darken_head']['said_by']}: {ref_form['darken_head']['why']}" if ref_form.get("darken_head") else "")
         + "".join(f"; {s['first']}-{s['last']}: {s['said_by']}, {s['at']}: {s['why']}"
                   for s in (iris.get("eyes_shut") or {}).get("spans") or [])),
        ("the mouth", {"auto": "drawn when the lip lines lie on the class map's mouth (head_reference.MOUTH_CHECK), else left out: decided and printed at "
                               "the reference step", "drawn": "drawn", "left out": "left out"}[mouth["form"]]
         + "".join(f"; NONE on {s['first']}-{s['last']}, turned off on a person's word" for s in mouth.get("no_mouth_on") or [])
         + "".join(f"; DRAWN on {s['first']}-{s['last']} where the gate drops it, the pose model's lip rings, on a person's word" for s in mouth.get("mouth_on") or [])
         + ("; CLOSED LIPS as one parting line where no teeth or tongue show (head_reference --closed-lip-line)" if cl_on else "")
         + ("; closed-lip line OFF, turned off on a person's word" if cl_off else "")
         + (f"; OPEN MOUTHS {mouth['open_mouth']['form'].upper()} (head_reference --open-mouth, draw_face_points.OPEN_MOUTH_FORMS)"
            if cl_on and mouth.get("open_mouth") else "")
         + (f"; CLOSED MOUTHS {mouth['closed_mouth']['form'].upper()} (head_reference --closed-mouth, draw_face_points.CLOSED_MOUTH_FORMS)"
            if cl_on and mouth.get("closed_mouth") else ""),
         f"{mouth['said_by']}: {mouth['why']}" + "".join(f"; {s['first']}-{s['last']}: {s['said_by']}, {s['at']}: {s['why']}"
                                                       for s in (mouth.get("no_mouth_on") or []) + (mouth.get("mouth_on") or []))
         + cl_from
         + (f"; open_mouth: {mouth['open_mouth']['said_by']}: {mouth['open_mouth']['why']}" if cl_on and mouth.get("open_mouth") else "")
         + (f"; closed_mouth: {mouth['closed_mouth']['said_by']}: {mouth['closed_mouth']['why']}" if cl_on and mouth.get("closed_mouth") else "")),
        *([("the reference's crop", {"face": "the face: one box round the faces alone with crop_reference_to_head.HEAD_ROOM, the head track "
                                                   "not read, the crown may be cut (head_reference --crop face)",
                                          "head": "the head: one box round the head track with room above the crown, and the faces "
                                                  "(head_reference --crop head)",
                                          "cranium": "the grey head: one box round the body model's head above mid-face and the faces with "
                                                     "crop_reference_to_head.CRANIUM_ROOM, CRANIUM_NECK below the chin, the head track not read "
                                                     "(head_reference --crop cranium)"}[ref_form["crop"]["form"]],
            f"{ref_form['crop']['said_by']}: {ref_form['crop']['why']}")] if ref_form.get("crop") else []),
        *([("the part held on a person's word", "on " + hold_spans_said(mine["word"]) + ": frames the capture's grade leaves empty there take "
            "the nearest graded part, moved and cut to the track (capture_masked_run.word_shapes)" + (
                "; a span held in place from a frame takes that frame's graded part where it was, unmoved and cut to the track, refused "
                "where that frame has no graded part (capture_masked_run.HELD_IN_PLACE)" if "@" in mine["word"] else ""), mine["word_by"])]
          if mine.get("word") else []),
        *([("the face lines", "NONE on " + ", ".join(f"{s['first']}-{s['last']}" for s in ref_form["no_face_on"]) + ", turned off on a person's word "
            "(the plain cut body model alone there); the capture drops the part's face classes there and keeps the rest, her hair "
            "(a face-only part is emptied there; capture_masked_run.grade_holds)", "; ".join(f"{s['first']}-{s['last']}: {s['said_by']}, {s['at']}: {s['why']}" for s in ref_form["no_face_on"]))]
          if ref_form.get("no_face_on") else []),
        *([("the head tube", "SAM 3.1 whole-head tube bounded by the person track + 3-frame smoothed Sapiens2-Seg on one steady shot box (head_tube_parts --head-tube)",
            f"shot_to_load.py::defaults_carried (opt-in switch, off by default; {head_tube_block['said_by']}: {head_tube_block['why']})")]
          if head_tube_block else []),
        ("depth",
         "on: MoGe-2 depth (bench/measure_moge2_depth.py) measured on frames where two tracks or a keep-out overlap into the capture folder and passed as --depth to the files step"
         if depth_block else "off: shared pixels are not settled by depth unless --depth is given",
         f"shot_to_load.py::defaults_carried (opt-in switch, off by default; {depth_block['said_by']}: {depth_block['why']})"
         if depth_block else "the chain's default (off by default)"),
        ("zoom", "zoomed in where the node's box allows (the node decides at plan)" if zoom_block else "the whole frame",
         f"{(zoom_block or whole_block)['said_by']}: {(zoom_block or whole_block)['why']}"),
        ("the context frames", {"as filmed": "as filmed: the real frames a short load is padded with are untouched, the original subject in "
                                             "them unmasked",
                                "face covered": "face covered: the subject's region held open on the real frames a short load is padded with "
                                                "(the shot's first frame's mask before it, its last frame's after; video_mask.pad_context); "
                                                "nothing on a load that fills its window"}[context["value"]],
         f"{context['said_by']}: {context['why']}"),
        ("other people in the shot", (p.get("others_present") or {}).get("said", "not read"), "bench/others_present.py, from the preview's tables and the owner's answers"),
        ("SAM 3.1 memory selection", *memory_selection_carried(preview)),
        ("SAM 3.1 head track", *head_track_carried(preview, no_head_track_block)),
        ("the strangers keep-out", "kept out" if "strangers" in names else
         "not kept out, turned off on a person's word; the part still held to the subject's own track" if p.get("strangers_turned_off") else
         "not kept out", p.get("strangers_from") or ("the plan" + (f": {p.get('keep_outs_by')}" if p.get("keep_outs") else ""))),
        ("the part's pieces", f"pieces={mine['pieces']}" if mine.get("pieces") else "every piece", "the plan" + (f": {p.get('pieces_by')}" if p.get("pieces") else "")),
        ("the hair band", f"{band:g} of the part's height under it, from the box's top, the margin's width each side"
         + f" (h3_config.MASKED_JOB_HAIR_BAND: starts {h3_config.MASKED_JOB_HAIR_BAND['starts']}, wider {h3_config.MASKED_JOB_HAIR_BAND['wider']})" if band is not None else "none",
         "the plan" if band is not None else "the plan names none"),
        ("restore spans", "; ".join(f"{'+'.join(x['classes'])} on {x['first']}-{x['last']}" for x in p.get("restore") or []) or "none",
         "the plan (bench/delivery_table.py writes them onto the rows)" + (f": {p.get('restore_by')}" if p.get("restore") else "")),
        ("the text", "generated by bench/load_text.py from the reference's record, the plan's words and the voice table, with no "
         "place-and-action sentence (load_text.NO_SCENE, 2026-10-15)", "the chain's default (2026-10-14); `text --by-hand \"why\"` for a text a person writes whole"),
        ("what the composite kept", "saved into each window's region file (the song node's save_kept), for the review clips to draw"
         if load.get("save_kept") else "not saved", "the chain's default (the lead, 2026-10-15)"),
    ]
    return [{"what": a, "carries": b, "from": c} for a, b, c in lines]


def _raw_source_names_and_paths(shots: dict, plan_file: str, default_source: Path, world) -> tuple[set[str], set[Path]]:
    """Return the file names and resolved paths that identify the raw unswapped source video."""
    plan_dir = Path(plan_file).resolve().parent
    raw_names: set[str] = {default_source.name, "source.mp4"}
    raw_paths: set[Path] = {default_source.resolve(), (plan_dir / "source.mp4").resolve()}
    for k in ("source", "video", "copy"):
        val = shots.get(k)
        if val:
            raw_names.add(Path(val).name)
            for cand in (Path(val), plan_dir / val, world.inputs / Path(val).name):
                try:
                    raw_paths.add(cand.resolve())
                except OSError:
                    pass
    return raw_names, raw_paths


def laid_on_frames(plan_file: str, video: str, world, first: int, last: int) -> tuple[Path | None, list[dict]]:
    """The list of what is laid (`<plan's folder>/laid.jsonl`, else `data/<video>/laid.jsonl`) and its lines that lay a piece
    on any of `[first, last]`, of any label, by each line's own spans (`frames`, else `first`-`last`)."""
    plan_dir = Path(plan_file).resolve().parent
    laid_path = next((p for p in (plan_dir / "laid.jsonl", world.data / video / "laid.jsonl") if p.is_file()), None)
    if laid_path is None:
        return None, []
    rows: list[dict] = []
    for line in laid_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        rf, rl = int(row.get("first", -1)), int(row.get("last", -2))
        row_spans = row.get("frames") or [[rf, rl]]
        if (rf <= last and rl >= first) and any(int(u) <= last and int(v) >= first for u, v in row_spans):
            rows.append(row)
    return laid_path, rows


def check_pass2_source(
    shots: dict,
    plan_file: str,
    video: str,
    world,
    first: int,
    last: int,
    other_subjects: list[str],
    given_source: str,
    default_source: Path,
    allow_raw_source: str | None = None,
) -> str:
    """Refuse segmenting or rendering from the raw unswapped source video when another replacement is already laid on
    `[first, last]` in `laid.jsonl` (unless `--allow-raw-source WHY` is given). `other_subjects` are the labels whose laid
    pieces count: at `job` every label but the load's, at `preview` every label but the ones it is made for (`--for`), so
    a preview made to redo a laid piece does not count that piece (the owner, 2026-10-21: every replacement is a pass,
    made on the cut that holds every OTHER replacement laid there; 5011's remake had needed the override). Nothing here
    names a subject: the labels are the shots record's and the list's."""
    if allow_raw_source is not None and not " ".join(str(allow_raw_source).split()):
        raise Refused("--allow-raw-source takes the reason in words")
    laid_path, laid_rows = laid_on_frames(plan_file, video, world, first, last)
    if laid_path is None or not other_subjects:
        return (
            f" RAW SOURCE ALLOWED ON PASS 2 ({' '.join(str(allow_raw_source).split())})."
            if allow_raw_source
            else ""
        )
    other_set = set(other_subjects)
    laid_hits = [row for row in laid_rows if row.get("subject") in other_set]
    if not laid_hits:
        return (
            f" RAW SOURCE ALLOWED ON PASS 2 ({' '.join(str(allow_raw_source).split())})."
            if allow_raw_source
            else ""
        )
    given_p = Path(given_source)
    try:
        given_res = given_p.resolve()
    except OSError:
        given_res = given_p
    raw_names, raw_paths = _raw_source_names_and_paths(shots, plan_file, default_source, world)
    is_raw = given_p.name in raw_names or given_res in raw_paths
    if is_raw and not allow_raw_source:
        laid_said = ", ".join(
            f"{r['subject']} ({r.get('load', '?')} on {r.get('first')}-{r.get('last')})"
            for r in laid_hits
        )
        raise Refused(
            f"on {first}-{last}, {laid_said} is already laid in {laid_path}: "
            f"Pass 2 must segment and render from the post-Pass-1 video containing Pass 1's output, "
            f"not the raw unswapped source video ({given_source}); pass the post-Pass-1 video via --source "
            f"(or --allow-raw-source \"<why>\" if raw source is intentional)"
        )
    if is_raw and allow_raw_source:
        return f" RAW SOURCE ALLOWED ON PASS 2 ({' '.join(str(allow_raw_source).split())})."
    import re
    m_cut = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}):?(\d{2})Z", given_p.name)
    if m_cut and not allow_raw_source:
        cut_hm = f"{m_cut.group(1)}:{m_cut.group(2)}"
        newer = [r for r in laid_hits if str(r.get("added") or "")[:16] > cut_hm]
        if newer:
            newer_said = ", ".join(
                f"{r['subject']} ({r.get('load', '?')} on {r.get('first')}-{r.get('last')}, laid {r.get('added')})"
                for r in newer
            )
            raise Refused(
                f"on {first}-{last}, {newer_said} in {laid_path} is newer than the cut {given_p.name} ({cut_hm}Z): "
                f"Pass 2 must segment and render from a cut built after those laid rows "
                f"(or --allow-raw-source \"<why>\" if an older cut is intentional)"
            )
    return ""


# ---------------------------------------------------------------- the steps

def step_preview(a, world) -> None:
    src_arg = getattr(a, "source", None)
    if not src_arg or not str(src_arg).strip():
        raise Refused(
            "preview requires --source <video> explicitly (never defaults to shots.json's raw source); "
            "on a multi-subject shot where an earlier subject is already laid, pass the post-Pass-1 video"
        )
    if not Path(src_arg).is_file():
        raise Refused(f"--source {src_arg}: no such file")
    shots, plan = _load(a.shots, a.plan)
    shot = shot_of(shots, a.shot)
    video, default_source, earlier = working_copy(shots["copy"], world)
    raw_names, _ = _raw_source_names_and_paths(shots, a.plan, default_source, world)
    src_p = Path(src_arg)
    if src_p.name not in raw_names:
        source = src_p.resolve()
        earlier = ""
    else:
        source = default_source
    cast = None
    if a.cast:
        if not Path(a.cast).is_file():
            raise Refused(f"--cast {a.cast}: no such file")
        import cast_file
        cast = cast_file.read(a.cast)
    name = preview_name(shot)
    if a.remake:
        # a name is taken by a job file in this folder or any *_jobs folder, and by a capture or job folder of any day
        taken = lambda n: name_taken(n, a.folder, world)   # noqa: E731
        name = free_name(name, taken, True)
    ask = {}
    for text in a.ask:
        label, _, why = text.partition(":")
        if not label.strip() or not why.strip():
            raise Refused(f"--ask {text!r} is not `SUBJECT:why`")
        ask[label.strip()] = " ".join(why.split())
    present_on_shot = [k for k in shots["subjects"] if (shot["who"].get(k) or {}).get("present") == "yes" or k in ask]
    # THE REPLACEMENT(S) THIS PREVIEW IS MADE FOR (`--for LABEL`, the owner, 2026-10-21): its source is held to the laid pieces
    # of every OTHER label on these frames, never to its own. Required where anything is laid on the shot; no silent default
    made_for = list(dict.fromkeys(x.strip() for x in getattr(a, "made_for", None) or [] if x.strip()))
    wrong = [x for x in made_for if x not in present_on_shot]
    if wrong:
        raise Refused(f"--for {', '.join(wrong)}: a preview is made for a replacement the shots record says is on {shot['first']}-{shot['last']} "
                      f"({', '.join(present_on_shot) or 'nobody'})")
    laid_path, laid_here = laid_on_frames(a.plan, video, world, int(shot["first"]), int(shot["last"]))
    if laid_here and not made_for:
        raise Refused(f"on {shot['first']}-{shot['last']}, " + ", ".join(f"{r.get('load', '?')} ({r.get('subject')})" for r in laid_here)
                      + f" {'is' if len(laid_here) == 1 else 'are'} laid in {laid_path}: say which replacement this preview is made for, "
                      "--for LABEL (once for each), so that its source is held to the laid pieces of the others alone (check_pass2_source). "
                      "Nothing is written")
    raw_note = check_pass2_source(
        shots,
        a.plan,
        video,
        world,
        int(shot["first"]),
        int(shot["last"]),
        [k for k in shots["subjects"] if k not in made_for],
        str(src_arg),
        default_source,
        allow_raw_source=getattr(a, "allow_raw_source", None),
    )
    seed_cuts = cuts_of_table(a.seed_cuts_of, shot) if a.seed_cuts_of else None
    if seed_cuts and not cast:
        raise Refused("--seed-cuts-of goes with --cast: the cuts say where a seed from the cast file may go")
    job = preview_job(shots, plan, shot, video, str(source), a.by, cast, load_chain.utc_now(), name=name, ask=ask, seed_cuts=seed_cuts,
                      no_memory_selection=getattr(a, "no_memory_selection", None),
                      no_head_track=getattr(a, "no_head_track", None), head_per_stretch=getattr(a, "head_per_stretch", None))
    if seed_cuts:
        job["job"]["seed_cuts"] = seed_cuts
    if made_for:
        job["job"]["made_for"] = {"subjects": made_for, "by": a.by,
                                  "what": "the replacement(s) this preview is made for (--for): its source was held to the laid pieces of the others"}
        raw_note = (f" MADE FOR {', '.join(made_for)} (--for): the source is held to the laid pieces of every other label on these frames "
                    "(check_pass2_source)." + raw_note)
    if ask:
        job["job"]["made_to_ask"] = {"what": "this preview tracks a subject the shots record is unsure of, to ask the owner who is on the shot; nothing is built on it "
                                              "until they have answered", "subjects": ask, "by": a.by}
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "THE SHOTS RECORD NAMES AN EARLIER COPY") + raw_note
    out = Path(a.folder) / f"{job['job']['name']}.json"
    if out.exists():
        raise Refused(f"{out} exists: a preview of this shot was made under that name. A preview made again (a pick corrected in the shots record, "
                      "or a box now in the cast file) takes --remake, which gives it the next free letter and leaves the earlier one as it is")
    print(f"{job['job']['name']}: source frames {shot['first']}-{shot['last']}; " + "; ".join(
        f"{s['label']} " + ("seeded from the cast (its `pick` decides nothing)" if "frame" in str(s.get("seeds") or "") else f"picked on {s['pick_at']}")
        for s in job["job"]["subjects"]))
    ht_cfg = job["job"]["sam_head_track"]
    print(f"SAM 3.1 head track: {'on' if ht_cfg['on'] else 'OFF'} [{ht_cfg['said_by']}: {ht_cfg['why']}]")
    print(job["job"]["note"])
    folder, _ = held_as_the_builder_holds(job["job"], a.day, world, default_source=default_source)
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


def run_strangers(kept: dict | None, a, world, name: str) -> None:
    """The strangers pass (bench/strangers.py: its people pass, SAM 3.1 on every frame, then the part model on the whole frame, each on
    the card in a process of its own) where the job keeps the strangers out and its preview's file is not there: run by this step
    when the model store is known, on an empty queue (the lead's order of 2026-10-14: the chain runs it where someone else is in the
    shot), with the card freed right before it (`world.free()`, as the land step frees it for its pose pass) and the chain's host
    handed on; else left to `looked_for`, which refuses with the command."""
    owed = [k for k in (kept or {}).get("made", []) if k["name"] == "strangers" and not Path(k["track"]).is_file()]
    store = getattr(a, "model_store", "") or os.environ.get(load_chain.MODEL_STORE)
    if not owed or world.dry or not store:
        return
    import h3_config
    load_chain.empty_queue(world, "no strangers pass beside a render")
    # THE CARD FREED FIRST, as the land step frees it for its pose pass (`load_chain.free_for_a_pose_pass`): the pass loads SAM 3.1 and the
    # part model in processes of their own, and the server keeps a render's models on the card (`strangers.py` frees it again itself, for a
    # pass run by hand)
    print("\n--- the card is freed before the strangers pass (POST /free: unload_models, free_memory)", flush=True)
    print(f"    {world.free()}\n    THE SERVER'S MODELS ARE UNLOADED: the next thing queued starts cold", flush=True)
    # the context the machine's run reads: where its output is kept and the name it is kept under (its `name`; a context without
    # one failed with a traceback on second_hfp_0014d's job, 2026-10-14). The preview's folder whole: the command runs from the repo
    world.run({"logs": Path(a.folder).resolve() / "chain", "name": name}, "strangers pass",
              [world.python, "bench/strangers.py", "run", "--preview", str(Path(a.preview).resolve()), "--model-folder",
               str(Path(store) / h3_config.SAPIENS2["segmentation"]), "--by", a.by, "--host", world.host])   # the server it frees the card on


def overlap_frames_of(preview: dict, preview_files: list[str] | None, output: Path, subject: str, first: int, last: int,
                      track_not: dict | None = None, kept: dict | None = None) -> list[int]:
    """Source frames on `first..last` where two subjects' tracks or a keep-out and `subject`'s track overlap.

    Reads each preview subject's track (or its `track_not` copy when `track_not` names one for that subject)
    and each keep-out mask in `kept['made']` on `first..last`. Matches the frames `capture_masked_run.py`
    requires in `--depth` (`req_frames`): frames where more than one subject track is on, or where a keep-out
    mask overlaps `subject`'s track."""
    import numpy as np
    import capture_masked_run as cap
    count = last - first + 1
    if count <= 0:
        return []
    w, h = int(preview["canvas"][0]), int(preview["canvas"][1])
    pf = int(preview["first"])
    tn_by_sub = {t["subject"]: t for t in (track_not or {}).get("made") or []}
    subj_tracks: dict[str, np.ndarray] = {}
    for s in preview.get("subjects") or []:
        lbl = s["label"]
        t_entry = tn_by_sub.get(lbl)
        if t_entry is not None and Path(t_entry["track"]).is_file():
            t_path = Path(t_entry["track"])
            tr = cap.read_mask(str(t_path), (w, h), int(t_entry["first"]), first, count)[0].copy()
        else:
            t_path = next((output / f for f in (preview_files or []) if f"_{lbl}_track_" in Path(f).name and f.endswith(".mkv")), None)
            if t_path is None or not t_path.is_file():
                raise Refused(f"--depth: the preview {preview.get('name')}'s track for {lbl!r} is not there to read overlap frames")
            tr = cap.read_mask(str(t_path), (w, h), pf, first, count)[0].copy()
        if t_entry is not None:
            for xa, xb in t_entry.get("frames") or []:
                lo = max(int(xa) - first, 0)
                hi = max(min(int(xb) - first + 1, count), 0)
                tr[lo:hi] = False
        subj_tracks[lbl] = tr
    keep_tracks: list[np.ndarray] = []
    kf = int((kept or {}).get("first", pf))
    for k in (kept or {}).get("made") or []:
        k_path = Path(k["track"])
        if k_path.is_file():
            keep_tracks.append(cap.read_mask(str(k_path), (w, h), kf, first, count)[0])
        elif "classes" in k and Path(k["cut_from"]).is_file():
            names = cap._pack("sapiens2_parts").CLASS_NAMES
            cls = cap.read_classes(str(k["cut_from"]), (w, h), kf, first, count)
            keep_tracks.append(np.isin(cls, [names.index(c) for c in k["classes"]]))
        else:
            raise Refused(f"--depth: the keep-out `{k['name']}` ({k_path}) is not there to read overlap frames")
    ov = np.zeros(count, dtype=bool)
    if len(subj_tracks) > 1:
        sh = np.sum(list(subj_tracks.values()), axis=0) > 1
        ov |= sh.any(axis=(1, 2))
    if keep_tracks and subject in subj_tracks:
        mine = subj_tracks[subject]
        for kt in keep_tracks:
            ov |= (kt & mine).any(axis=(1, 2))
    return [first + int(i) for i in np.nonzero(ov)[0]]


def make_depth(job: dict, preview: dict, preview_files: list[str] | None, a, world, name: str,
               overlap=overlap_frames_of) -> list[str]:
    """When `job.depth.on` is set, find the frames where two tracks or a keep-out overlap on the load's span and run
    `bench/measure_moge2_depth.py` on just those frames into the capture folder (`<capture>/depth.npz`)."""
    depth_cfg = job["job"].get("depth")
    if not (depth_cfg or {}).get("on"):
        return []
    import capture_masked_run as cap
    first = int(job["loads"][0]["first"])
    last = first + int(job["loads"][0]["frames"]) - 1
    ov_frames = overlap(preview, preview_files, world.output, a.subject, first, last,
                        job["job"].get("track_not"), job["job"].get("keep_outs"))
    depth_cfg["overlap_frames"] = ov_frames
    if not ov_frames:
        depth_cfg["file"] = None
        for d_row in job["job"].get("defaults") or []:
            if d_row.get("what") == "depth":
                d_row["carries"] = "on: 0 overlap frames in the load; nothing measured"
        return [f"depth: no frames where two tracks or a keep-out overlap on {first}-{last}; nothing to measure"]
    cap_dir = world.data.parent / job["job"]["capture"]
    depth_file = cap_dir / "depth.npz"
    depth_cfg["file"] = f"{job['job']['capture']}/depth.npz"
    spans_s = ", ".join(f"{u}-{v}" if u != v else str(u) for u, v in cap.frame_spans(ov_frames, join=1))
    for d_row in job["job"].get("defaults") or []:
        if d_row.get("what") == "depth":
            d_row["carries"] = (
                f"on: MoGe-2 depth (bench/measure_moge2_depth.py) measured on {len(ov_frames)} overlap frame(s)"
                f" ({spans_s}) into {depth_cfg['file']} and passed as --depth to the files step"
            )
    cmd = [
        world.python, "bench/measure_moge2_depth.py",
        "--source", str(job["job"]["source"]),
        "--frames", ",".join(map(str, ov_frames)),
        "--size", f"{job['job']['canvas'][0]}x{job['job']['canvas'][1]}",
        "--out-dir", str(cap_dir),
    ]
    if world.dry:
        return [f"depth: WOULD MEASURE MoGe-2 depth on {len(ov_frames)} overlap frame(s) ({spans_s}) to {depth_file}"]
    world.run({"logs": Path(a.folder).resolve() / "chain", "name": name}, "moge2 depth", cmd)
    if not depth_file.is_file():
        raise Refused(f"--depth: bench/measure_moge2_depth.py did not write {depth_file}")
    return [f"depth: measured MoGe-2 depth on {len(ov_frames)} overlap frame(s) ({spans_s}) to {depth_file}"]


def step_job(a, world, cut=cut_keep_out, hashed=None, inside=part_inside, classes_of=part_classes, hold=hold_to_track, track_cut=cut_track,
             over=track_over_track, stay_cut=cut_stay, overlap=overlap_frames_of) -> None:
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
    video, default_source, _ = working_copy(shots["copy"], world)
    raw_names, _ = _raw_source_names_and_paths(shots, a.plan, default_source, world)
    prev_src = Path(preview["source"])
    if prev_src.is_file() and prev_src.name not in raw_names:
        source = prev_src.resolve()
        earlier = ""
    else:
        video, source, earlier = working_copy(prev_src.name, world)
    region = plan["subjects"][a.subject]["region"]
    if region not in FORM:
        raise Refused(f"the plan's region for {a.subject!r} is {region!r}: this tool names and makes part loads ({', '.join(FORM)})")
    theirs = plan["subjects"][a.subject].get("parts") or []
    cwf = getattr(a, "carry_words_from", None)
    ncw = getattr(a, "no_carry_words", None)
    lw = getattr(a, "leave_word", None) or []
    lby = getattr(a, "leave_by", None)
    eff_first, eff_parts = a.first, a.parts
    if cwf is not None and (eff_first is None or eff_parts is None):
        cw_peek = carry_words_of(cwf, shot, a.subject, list(theirs), a.first, a.frames, a.why or "", a.why_before or "", a.why_after or "",
                                 by=a.by, now=load_chain.utc_now(), leave_word=lw, leave_by=lby)
        if eff_first is None:
            eff_first = cw_peek["first"]
        if eff_parts is None:
            eff_parts = cw_peek["parts"]
    first = shot["first"] if eff_first is None else eff_first
    folder = Path(a.folder)
    # a name is taken by a job file in this folder or any *_jobs folder, and by a capture or a job folder of any day
    taken = lambda n: name_taken(n, folder, world)
    # a load of some of the plan's parts is named by them, a letter a part: the hair alone is `hp`, neither `hfp` nor `fp`
    form = FORM[region] if not eff_parts else "".join(PART_LETTER.get(k, "x") for k in theirs if k in eff_parts) + "p"
    base_stem = f"{a.subject}_{form}_{first:04d}"
    earlier_stems = [s for s in (base_stem, *(base_stem + c for c in "bcdefghijklmnopqrstuvwxyz")) if taken(s)]
    for jf in sorted(folder.glob(f"{a.subject}_*.json")):
        if jf.stem in earlier_stems:
            continue
        try:
            jd = json.loads(jf.read_text()).get("job") or {}
        except Exception:
            continue
        jf_first = jd.get("first")
        if isinstance(jf_first, int) and shot["first"] <= jf_first <= shot["last"]:
            earlier_stems.append(jf.stem)
    if a.remake and earlier_stems and cwf is None and ncw is None:
        raise Refused(f"--remake on {a.subject} in {shot['first']}-{shot['last']} where {', '.join(earlier_stems)} already exist(s) "
                      "takes --carry-words-from <old_job.json> or --no-carry-words \"<why>\"")
    name = free_name(base_stem, taken, a.remake)
    rel = str(pfolder.resolve().relative_to(REPO.resolve())) if pfolder.resolve().is_relative_to(REPO.resolve()) else str(pfolder)
    cast_rows = None
    if getattr(a, "cast", None):
        if not Path(a.cast).is_file():
            raise Refused(f"--cast {a.cast}: no such file")
        cast_rows = [json.loads(x) for x in Path(a.cast).read_text().splitlines() if x.strip()]
    job = load_job(shots, plan, shot, a.subject, preview, rel, video, world.output, a.by, a.day, name, a.first, a.frames, a.why or "", load_chain.utc_now(),
                   source=str(source), why_before=a.why_before or "", why_after=a.why_after or "", zoomed=a.zoomed, least=a.least, least_why=a.least_why,
                   whole_frame=a.whole_frame,
                   preview_files=ran.get("files") or [], parts=a.parts, parts_by=a.parts_by, classes_of=classes_of,
                   parts_on=[parts_on_arg(x) for x in a.parts_on], track_not=[track_not_arg(x) for x in a.track_not], track_not_by=a.track_not_by,
                   cast=cast_rows, cast_from=getattr(a, "cast", None) or "",
                   motion=getattr(a, "motion", None), motion_why=getattr(a, "motion_why", None), no_strangers=getattr(a, "no_strangers", None),
                   no_mouth_on=[no_mouth_on_arg(x) for x in getattr(a, "no_mouth_on", None) or []],
                   mouth_on=[no_mouth_on_arg(x, "--mouth-on") for x in getattr(a, "mouth_on", None) or []], data=world.data,
                   eyes_shut_on=[no_mouth_on_arg(x, "--eyes-shut-on") for x in getattr(a, "eyes_shut_on", None) or []],
                   no_face_on=[no_mouth_on_arg(x, "--no-face-on") for x in getattr(a, "no_face_on", None) or []],
                   hold_part=[hold_part_arg(x) for x in getattr(a, "hold_part", None) or []],
                   eyes_shut_form=getattr(a, "eyes_shut_form", None), seam=seams_arg(getattr(a, "seam", None)), seam_why=getattr(a, "seam_why", None),
                   context_noise=getattr(a, "context_noise", None), context_noise_why=getattr(a, "context_noise_why", None),
                   drop=getattr(a, "drop", None), drop_by=getattr(a, "drop_by", None), carry_words_from=cwf,
                   leave_word=lw, leave_by=lby, no_carry_words=ncw,
                   head_tube=getattr(a, "head_tube", None), depth=getattr(a, "depth", None),
                   no_head_track=getattr(a, "no_head_track", None),
                   no_closed_lip_line=getattr(a, "no_closed_lip_line", None),
                   seed=getattr(a, "seed", None), seed_why=getattr(a, "seed_why", None),
                   zoom_up_to=getattr(a, "zoom_up_to", None), zoom_up_to_why=getattr(a, "zoom_up_to_why", None),
                   margin=getattr(a, "margin", None), margin_why=getattr(a, "margin_why", None))
    load_first = int(job["loads"][0]["first"])
    load_last = load_first + int(job["loads"][0]["frames"]) - 1
    other_subjects = list(dict.fromkeys(
        [k for k in shots["subjects"] if k != a.subject and (shot["who"].get(k) or {}).get("present") == "yes"]
        + [s["label"] for s in (preview.get("subjects") or []) if s.get("label") and s["label"] != a.subject]
    ))
    raw_note = check_pass2_source(
        shots,
        a.plan,
        video,
        world,
        load_first,
        load_last,
        other_subjects,
        str(prev_src),
        default_source,
        allow_raw_source=getattr(a, "allow_raw_source", None),
    )
    job["job"]["note"] += earlier.replace("MADE ON AN EARLIER COPY", "ITS PREVIEW WAS MADE ON AN EARLIER COPY") + raw_note
    out = folder / f"{name}.json"
    print(f"{name}: {a.subject}, {renders(region, eff_parts)}, source frames {job['loads'][0]['first']}-{job['loads'][0]['first'] + job['loads'][0]['frames'] - 1}; "
          + (f"its reference will be {job['loads'][0]['motion_file']}" if job["loads"][0].get("motion_file") else "NO REFERENCE VIDEO (motion none)"))
    print(job["job"]["note"])                # the load in words, as the preview step prints its own: a dry run shows every choice made for it
    print("THE DEFAULTS THIS LOAD CARRIES:")
    for line in job["job"]["defaults"]:
        print(f"  {line['what']}: {line['carries']}  [{line['from']}]")
    load_held_as_the_builder_holds(job, world, default_source=default_source)
    # the keep-outs the plan names are made before the job file that points at them is written; a dry run says what would be
    # what is looked for is held first, and the track a part is held to, so a refusal writes nothing; the held part last: it is
    # made from the load's own part
    run_strangers(job["job"].get("keep_outs"), a, world, name)
    on_track = job["job"].get("part_on_track")
    why = track_refusals(Path(on_track["track"]), on_track["canvas"], on_track["frames"]) if on_track else []
    if why and not world.dry:
        raise Refused(f"the part cannot be held to its track: {'; '.join(why)}. Nothing is written")
    for line in ([f"the part held to the track: WOULD BE REFUSED: {'; '.join(why)}"] if why else []) + (looked_for(job["job"].get("keep_outs"), world) + make_part(job["job"].get("part"), world, cut, hashed, inside)
                 + make_keep_outs(job["job"].get("keep_outs"), world, cut, hashed)
                 + make_track_not(job["job"].get("track_not"), world, track_cut, hashed)
                 + make_part_on_track(job["job"].get("part_on_track"), world, hold, hashed)
                 + make_stay(job["job"].get("keep_outs"), world, stay_cut, hashed, job)
                 + make_depth(job, preview, ran.get("files") or [], a, world, name, overlap=overlap)):
        print(line)
    # where one subject's preview track lies on another's, as a figure, before anything is built on it
    for line in over(preview, ran.get("files") or [], world.output, a.subject, job["loads"][0]["first"], job["loads"][0]["first"] + job["loads"][0]["frames"] - 1):
        print(line)
    if world.dry:
        print("DRY: nothing written. The load was held to the builder's refusals that need neither its capture nor its text (the job block, the file's "
              "identity, the load's keys and settings, the shortest load, left_out). NOT exercised: anything read from a capture, a text or a mask.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(job, indent=1))
        print("wrote", out)
    print(f"NEXT: bench/shot_to_load.py reference {out} --by {a.by}\nTHEN THE TEXT, generated from the reference's record and the plan: "
          f"bench/shot_to_load.py text {out} --by {a.by} --plan PLAN.json (or a text written whole: --by-hand \"why\" --writer \"who, from which frames\")")


def step_reference(a, world, figures=figures_of) -> None:
    ctx = load_chain.read_job(Path(a.job_file), need_text=False)
    # this step's output is kept beside the job file, as the preview's is: the load's job folder is the chain's to make,
    # at `first`, and until 2026-10-13 a reference made it, before anything of the load was built
    ctx["logs"] = Path(a.job_file).parent / "chain"
    job, load, subject = ctx["job"], ctx["load"], ctx["subject"]
    if a.plain is not None and not a.plain.strip():
        raise Refused("--plain takes the reason in words: the plain cut is taken on a person's word, and the word is kept")
    nht = getattr(a, "no_head_track", None)
    if nht is not None and not " ".join(str(nht).split()):
        raise Refused("--no-head-track takes the reason in words: the SAM 3.1 head track is the chain's default, and the word is kept")
    first, last = load["first"], load["first"] + load["frames"] - 1
    if load.get("motion") == "none":
        if a.plain is not None:
            raise Refused("--plain on a load with no reference video (motion none): there is no reference to cut")
        if nht is not None:
            raise Refused("--no-head-track on a load with no reference video (motion none): there is no reference to cut")
        print(f"{ctx['name']}: NO REFERENCE VIDEO (motion none), so nothing is made here: the job's note says why. NEXT: the text step")
        return
    drawn = Path(load["motion_file"])
    folder = drawn.parent
    points = folder / f"{subject}_{first:04d}_{last:04d}_points.npz"
    form, why = ("plain", a.plain.strip()) if a.plain is not None else ("drawn", "the form for a face, head or hair replacement (AGENTS.md, Settled)")
    def again(words: str) -> Refused:
        # `--existing` keeps its refusal words, and names the folder a load made again on these frames takes (`reference_folder`)
        nxt, _why = reference_folder(world.output, load_chain.video_of(ctx, world.data), subject, first, last, world.data)
        return Refused(f"{words} (a load made again takes the folder {nxt.name}: `shot_to_load.py job --remake`)")
    if a.existing:
        record = drawn.with_suffix(".json")
        if not record.is_file() or not points.is_file():
            raise again(f"--existing: {record} and {points.name} beside it are not both there; without --existing the reference is made")
        made = json.loads(record.read_text())
        if made.get("frames") != [first, last] or made.get("passed") is not True:
            raise again(f"--existing: {record} is of frames {made.get('frames')} with passed={made.get('passed')!r}, and the load is {first}-{last}")
        if not (made.get("face_on_the_track") or {}).get("known"):
            # a reference made before 2026-10-14 read its points in the track's box alone, and on a close group drew
            # another person's face and passed (the lead's 3167): whether its face is on the subject is not known
            raise again(f"--existing: {record} does not say its face was held to {subject}'s track (made before the points were, or on the whole "
                          "frame): the reference is made again, in a new folder")
        # the job's spans of her eyes shut on a person's word: a reference taken as it was must have drawn them shut, in that form
        eyes = ((job.get("reference") or {}).get("iris") or {}).get("eyes_shut") or {}
        if eyes.get("spans"):
            drew = (made.get("drawn") or {}).get("eyes_shut") or {}
            # held to the spans the job names (head_reference skips a frame with no face, so its own frames are compared, not the spans')
            asked = sorted((w["first"], w["last"]) for w in drew.get("on_whose_word") or [])
            meant = sorted({f for x, y in drew.get("frames") or [] for f in range(x, y + 1)})
            done = sorted({f for x, y in drew.get("drawn_shut_on") or [] for f in range(x, y + 1)})
            if drew.get("form") != eyes["form"] or asked != sorted((s["first"], s["last"]) for s in eyes["spans"]) or done != meant:
                raise again(f"--existing: {record} does not draw her eyes shut ({eyes['form']!r}) on every frame the job names: the reference is "
                              "made again, in a new folder")
        # the job's spans of no face on a person's word: a reference taken as it was must have left those frames bare
        bare = set((made.get("drawn") or {}).get("frames_left_bare") or [])
        missed = [f for s in (job.get("reference") or {}).get("no_face_on") or [] for f in range(s["first"], s["last"] + 1) if f not in bare] \
            if form != "plain" else []          # the plain cut has no line of the face on any frame
        if missed:
            raise again(f"--existing: {record} draws face lines on {len(missed)} frame(s) the job says show no face on a person's word "
                          f"(first {missed[0]}, last {missed[-1]}): the reference is made again, in a new folder")
        # the job's spans of no mouth on a person's word: a reference taken as it was must have drawn none there
        said = ((job.get("reference") or {}).get("mouth") or {}).get("no_mouth_on") or []
        mouth = (made.get("drawn") or {}).get("mouth") or {}
        if said and mouth and mouth.get("form") != "left out":
            bare = {f for spans in (mouth.get("no_mouth_on") or {}).values() for x, y in spans for f in range(x, y + 1)}
            missed = [f for s in said for f in range(s["first"], s["last"] + 1) if f not in bare]
            if missed:
                raise again(f"--existing: {record} draws a mouth on {len(missed)} frame(s) the job says have none on a person's word "
                              f"(first {missed[0]}, last {missed[-1]}): the reference is made again, in a new folder")
        # the job's spans of a mouth on a person's word: a reference taken as it was must have been asked for them
        wanted = sorted((s["first"], s["last"]) for s in ((job.get("reference") or {}).get("mouth") or {}).get("mouth_on") or [])
        asked = sorted((w["first"], w["last"]) for w in (mouth.get("mouth_on") or {}).get("on_whose_word") or [])
        if wanted != asked:
            raise again(f"--existing: {record} was drawn with the mouth on a person's word on {asked or 'no frames'} and the job names "
                          f"{wanted or 'none'}: the reference is made again, in a new folder")
        ref_said = job.get("reference") or {}
        md = made.get("drawn") or {}
        cl_cfg = (ref_said.get("mouth") or {}).get("closed_lip_line")
        want_cl = (
            False if getattr(a, "no_closed_lip_line", False) else
            True if getattr(a, "closed_lip_line", False) else
            bool(cl_cfg.get("on", True)) if isinstance(cl_cfg, dict) else bool(cl_cfg)
        )
        if form != "plain":
            if want_cl and not (mouth.get("closed_lip_line") or {}).get("on"):
                raise again(f"--existing: {record} was not drawn with --closed-lip-line: the reference is made again, in a new folder")
            if (getattr(a, "no_closed_lip_line", False) or (isinstance(cl_cfg, dict) and not cl_cfg.get("on", True))) and (mouth.get("closed_lip_line") or {}).get("on"):
                raise again(f"--existing: {record} was drawn with --closed-lip-line and this load turns it off: the reference is made again, in a new folder")
            import draw_face_points
            want_om = ((ref_said.get("mouth") or {}).get("open_mouth") or {}).get("form") or draw_face_points.OPEN_MOUTH_FORMS[0]
            drew_om = (mouth.get("closed_lip_line") or {}).get("open_mouth") or draw_face_points.OPEN_MOUTH_FORMS[0]
            if want_cl and (mouth.get("closed_lip_line") or {}).get("on") and want_om != drew_om:
                raise again(f"--existing: {record} drew an open mouth {drew_om!r} and the job asks for {want_om!r} (the plan's reference.open_mouth): "
                            "the reference is made again, in a new folder")
            # a closed mouth's form, the plan's (`reference.closed_mouth`): a record made before it, or without it, drew the default
            want_cm = ((ref_said.get("mouth") or {}).get("closed_mouth") or {}).get("form") or draw_face_points.CLOSED_MOUTH_FORMS[0]
            drew_cm = (((mouth.get("closed_lip_line") or {}).get("closed_mouth") or {}).get("form")
                       or draw_face_points.CLOSED_MOUTH_FORMS[0])
            if want_cl and (mouth.get("closed_lip_line") or {}).get("on") and want_cm != drew_cm:
                raise again(f"--existing: {record} drew a closed mouth {drew_cm!r} and the job asks for {want_cm!r} (the plan's reference.closed_mouth): "
                            "the reference is made again, in a new folder")
        if (getattr(a, "darken_head", False) or ref_said.get("darken_head")) and not (md.get("darken_head") or {}).get("on"):
            raise again(f"--existing: {record} was not drawn with --darken-head: the reference is made again, in a new folder")
        ht_cfg = ref_said.get("head_track") if "head_track" in ref_said else job.get("sam_head_track")
        used_ht = bool((made.get("inputs") or {}).get("head_track"))
        # the box's form (the plan's `reference.crop`). A record made before the record kept it: a box that read a head track is the
        # head's; one that read none is the faces' alone, which is the face's form, and for a job of the head's box the head track's
        # own refusal below decides it (a preview may have saved no head track to read)
        want_crop = (ref_said.get("crop") or {}).get("form") or "head"
        drew_crop = (made.get("crop") or {}).get("form") or ("head" if used_ht or want_crop == "head" else "face")
        if want_crop != drew_crop:
            raise again(f"--existing: {record} was cut to the {drew_crop}'s box and the job asks for the {want_crop}'s (the plan's reference.crop): "
                        "the reference is made again, in a new folder")
        want_no_ht = bool(nht is not None or (isinstance(ht_cfg, dict) and not ht_cfg.get("on", True)) or want_crop != "head")
        if want_no_ht and used_ht:
            raise again(f"--existing: {record} was cut with a head track and this load turns it off (--no-head-track): the reference is made again, in a new folder")
        preview_has_ht = any(f"_{subject}_head_track_" in Path(f).name and f.endswith(".mkv") for f in (ctx.get("preview") or {}).get("files") or [])
        if not want_no_ht and isinstance(made.get("inputs"), dict) and not used_ht and (preview_has_ht or "head_track" in made["inputs"]):
            raise again(f"--existing: {record} was not cut with a head track: the reference is made again, in a new folder")
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
    flags = drawn_form(job, a, files, ctx["preview"]["first"])
    world.run(ctx, "head reference", [world.python, "bench/head_reference.py", "--mesh", mesh, "--mesh-first", ctx["preview"]["first"], "--points", points,
                                      "--source", job["source"], "--first", first, "--count", load["frames"], "--out", drawn, "--by", a.by, *flags])
    if world.dry:
        return
    record = drawn.with_suffix(".json")
    written(a, ctx, json.loads(record.read_text()), record, points, drawn, form, why, figures)


def drawn_form(job: dict, a, files: dict, preview_first: int) -> list:
    """The head reference tool's flags for the mouth and the eye: the job's (the plan's or the chain's default, `reference_form`),
    and `--no-mouth-on` for the load's frames whose own part holds no lips (the job's `part.parts_on`), and `--over-the-eyes` for the
    plan's restore spans of a class over her eyes on the load's frames (the job's `restore`, `head_reference.OVER_THE_EYES`),
    or the reference step's own `--mouth`/`--iris` with `--form-why`. A job written before 2026-10-14 says neither: its form is the
    tool's own as before (the mouth left out, the iris between the lids)."""
    said = job.get("reference") or {}
    mouth = a.mouth or (said.get("mouth") or {}).get("form")
    mouth_why = a.form_why if a.mouth else ((said.get("mouth") or {}).get("why") or "")
    iris = a.iris or (said.get("iris") or {}).get("form")
    nht = getattr(a, "no_head_track", None)
    if nht is not None and not " ".join(str(nht).split()):
        raise Refused("--no-head-track takes the reason in words: the SAM 3.1 head track is the chain's default, and the word is kept")
    ht_cfg = said.get("head_track") if "head_track" in said else job.get("sam_head_track")
    if isinstance(ht_cfg, dict) and not ht_cfg.get("on", True) and not " ".join(str(ht_cfg.get("why") or "").split()):
        raise Refused("reference.head_track is off with no `why`: the SAM 3.1 head track is the chain's default, and the word is kept")
    want_no_ht = bool(nht is not None or (isinstance(ht_cfg, dict) and not ht_cfg.get("on", True)))
    if getattr(a, "closed_lip_line", False) and getattr(a, "no_closed_lip_line", False):
        raise Refused("--closed-lip-line and --no-closed-lip-line cannot both be given")
    if (a.mouth or a.iris or getattr(a, "class_mouth_needs_hole", False) or getattr(a, "closed_lip_line", False)
            or getattr(a, "no_closed_lip_line", False) or getattr(a, "darken_head", False)) and not str(a.form_why or "").strip():
        raise Refused("--mouth and --iris at the reference step are a person's word: say why with --form-why")
    cl_cfg = (said.get("mouth") or {}).get("closed_lip_line")
    cl_cfg_on = cl_cfg.get("on", True) if isinstance(cl_cfg, dict) else bool(cl_cfg)
    if getattr(a, "no_closed_lip_line", False):
        cl = ""
        if not a.mouth and (said.get("mouth") or {}).get("form") == "left out" and str((said.get("mouth") or {}).get("why") or "").startswith("silent window"):
            mouth = "auto"
            mouth_why = "REFERENCE_DEFAULT (2026-10-14)"
    elif getattr(a, "closed_lip_line", False):
        cl = " ".join(a.form_why.split())
    elif cl_cfg_on and isinstance(cl_cfg, dict):
        if cl_cfg.get("said_by") == "the chain's default" and mouth not in ("auto", "left out"):
            cl = ""
        else:
            cl = f"{cl_cfg['said_by']}: {cl_cfg['why']}"
    else:
        cl = ""
    if cl and not a.mouth and (said.get("mouth") or {}).get("said_by") in (None, "the chain's default") and mouth == "auto":
        for row in job.get("defaults") or []:
            if isinstance(row, dict) and row.get("what") == "the voice" and str(row.get("carries") or "").startswith("silent, on "):
                mouth = "left out"
                mouth_why = f"silent window ({row.get('from')}): no mouth drawn"
                break
    out = []
    if iris:
        out += ["--iris", iris]
    if mouth == "auto":
        if files.get("classes") is None:
            raise Refused("the mouth is drawn by the lip check, which reads the subject's class map, and the preview saved none")
        out += ["--mouth", "auto", "--classes", str(files["classes"]), "--classes-first", preview_first]
    elif mouth:
        out += ["--mouth", mouth, "--mouth-why", mouth_why]
    # the load's frames whose part holds no lips (`--parts-on`, hair alone): no mouth drawn there (the lead, 2026-10-14, on 1010d's chest)
    part = job.get("part") or {}
    for on in next(iter(part.get("made") or []), {}).get("classes_on") or []:
        if not {"Upper_Lip", "Lower_Lip"} & set(on["classes"]):
            out += ["--no-mouth-on", f"{on['first']}-{on['last']}=the load's part is {'+'.join(next((x['parts'] for x in part.get('parts_on') or [] if x['first'] == on['first']), on['classes']))} "
                                     f"alone there ({' '.join(str(part.get('by') or 'who said so is not in the job').split())})"]
    # the job's spans of her eyes shut on a person's word (`job --eyes-shut-on`, with its form)
    shut = (said.get("iris") or {}).get("eyes_shut") or {}
    for s in shut.get("spans") or []:
        out += ["--eyes-shut-on", f"{s['first']}-{s['last']}=a person's word on a look ({s['said_by']}, {s['at']}): {s['why']}"]
    if shut.get("spans"):
        out += ["--eyes-shut-form", shut["form"]]
    # the job's spans of no mouth on a person's word (`job --no-mouth-on`, a look where her mouth is hidden)
    for s in (said.get("mouth") or {}).get("no_mouth_on") or []:
        out += ["--no-mouth-on", f"{s['first']}-{s['last']}=a person's word on a look ({s['said_by']}, {s['at']}): {s['why']}"]
    # the job's spans of a mouth on a person's word (`job --mouth-on`, a look where her mouth is open and the gate drops it)
    for s in (said.get("mouth") or {}).get("mouth_on") or []:
        out += ["--mouth-on", f"{s['first']}-{s['last']}=a person's word on a look ({s['said_by']}, {s['at']}): {s['why']}"]
    # the load's frames whose part holds no face (`--parts hair`, or a `--parts-on` span of hair alone): no line of the face drawn there
    # (the lead's ruling of 2026-10-15 on second_hp_5461b: eyes, brows and a nose drawn as on a plain face while hers was hidden)
    made = next(iter(part.get("made") or []), {})
    by = " ".join(str(part.get("by") or "who said so is not in the job").split())
    if made.get("classes") and "Face_Neck" not in made["classes"] and job.get("first") is not None:
        out += ["--no-face-on", f"{job['first']}-{int(job['first']) + int(job['frames']) - 1}=the load's part is {'+'.join(part.get('parts') or made['classes'])} "
                                f"alone on every frame ({by})"]
    for on in made.get("classes_on") or []:
        if "Face_Neck" not in on["classes"]:
            named = next((x["parts"] for x in part.get("parts_on") or [] if x["first"] == on["first"]), on["classes"])
            out += ["--no-face-on", f"{on['first']}-{on['last']}=the load's part is {'+'.join(named)} alone there ({by})"]
    # the job's spans of no face on a person's word (`job --no-face-on`: a head turned away, hidden behind somebody)
    for s in said.get("no_face_on") or []:
        out += ["--no-face-on", f"{s['first']}-{s['last']}=a person's word on a look ({s['said_by']}, {s['at']}): {s['why']}"]
    # the plan's restore spans of a class over her eyes (her own glasses, head_reference.OVER_THE_EYES): the face counts as on her track
    # there by the lower face (the lead, 2026-10-15, on 2823's white sunglasses, off her track from 3073)
    import head_reference
    for r in job.get("restore") or []:
        over = [c for c in r["classes"] if c in head_reference.OVER_THE_EYES]
        if over:
            out += ["--over-the-eyes", f"{r['first']}-{r['last']}=the plan gives back her {'+'.join(over)} there "
                                       f"({' '.join(str(r.get('by') or 'who said so is not in the job').split())[:200]})"]
    if getattr(a, "class_mouth_needs_hole", False):
        if mouth != "auto":
            raise Refused(f"--class-mouth-needs-hole acts on the mouth by the check (auto); this load's mouth is {mouth!r}")
        out += ["--class-mouth-needs-hole", " ".join(a.form_why.split())]
    if cl:
        if mouth not in ("auto", "left out"):
            raise Refused(f"--closed-lip-line acts on the mouth by the check (auto) or left out on a silent window; this load's mouth is {mouth!r}")
        out += ["--closed-lip-line", cl]
        # an open mouth's form, the plan's (`reference.open_mouth`); a job without it draws the default, so its command is as before
        om = (said.get("mouth") or {}).get("open_mouth")
        if om:
            out += ["--open-mouth", om["form"]]
        # a closed mouth's form, the plan's (`reference.closed_mouth`); a job without it draws the default, so its command is as before
        cm = (said.get("mouth") or {}).get("closed_mouth")
        if cm:
            out += ["--closed-mouth", cm["form"]]
    dh = (" ".join(a.form_why.split()) if getattr(a, "darken_head", False) else
          f"{said['darken_head']['said_by']}: {said['darken_head']['why']}" if said.get("darken_head") else "")
    if dh:
        out += ["--darken-head", dh]
    if want_no_ht:
        out += ["--no-head-track"]
    # the box's form, the plan's (`reference.crop`); a job without it is cut to the head's box, so its command is as before
    cr = said.get("crop")
    if cr and cr["form"] != "head":
        out += ["--crop", cr["form"], "--crop-why", f"{cr['said_by']}: {cr['why']}"]
    return out


def written(a, ctx: dict, made: dict, record: Path, points: Path, drawn: Path, form: str, why: str, figures, taken: bool = False) -> None:
    """Write a reference into its load's job file, with the figures, and print them. `taken`: the reference was there."""
    load = ctx["load"]
    got = figures(points, float(made["scale"][0]))
    d = json.loads(Path(a.job_file).read_text())
    was = d["job"].get("reference") or {}
    now_ts = load_chain.utc_now()
    d["job"]["reference"] = reference_block(form, why, a.by, made["made_at"] if taken else now_ts, record, points, got)
    # the mouth and the eye as asked, and as drawn (the record's own words: the check's figures, the iris form)
    md = made.get("drawn") or {}
    d["job"]["reference"].update({k: was[k] for k in ("mouth", "iris", "no_face_on", "darken_head", "head_track", "crop") if k in was},
                                 drawn={"mouth": md.get("mouth"), "iris": md.get("iris"),
                                        "eyes_shut": md.get("eyes_shut"),
                                        "parts_left_out": md.get("parts_left_out"),
                                        "frames_left_bare": md.get("frames_left_bare"),
                                        **({k: md[k] for k in ("eye_rings", "darken_head") if md.get(k) is not None})})
    nht = getattr(a, "no_head_track", None)
    if nht is not None:
        d["job"]["reference"]["head_track"] = {
            "on": False,
            "why": " ".join(str(nht).split()),
            "said_by": a.by,
            "at": now_ts,
        }
    elif "head_track" not in d["job"]["reference"] and isinstance(d["job"].get("sam_head_track"), dict) and not d["job"]["sam_head_track"].get("on", True):
        d["job"]["reference"]["head_track"] = dict(d["job"]["sam_head_track"])
    if taken:
        d["job"]["reference"]["taken_as_it_was"] = f"made by {made.get('made_by')}; read into this job by {a.by} at {now_ts}, nothing run"
    # the plain cut's name is the reference tool's own, read from its record
    d["loads"][0]["motion_file"] = str(drawn.parent / made["files"]["plain_cut"] if form == "plain" else drawn)
    Path(a.job_file).write_text(json.dumps(d, indent=1))
    c, r = got["at_the_canvas"], got["in_the_reference"]
    print(f"\nTHE FIGURES, on {got['frames']} frames ({got['measured_on']}); the drawing line is {got['the_drawing_line_px']} px. Nothing here decides the form.")
    print(f"  at the canvas:    face {c['face_height_px']} px tall; lash lines {c['lid_gap_px']} px apart; under the line on {c['frames_with_the_gap_under_the_line']} frames")
    print(f"  in the reference (enlarged {r['enlarged']}): face {r['face_height_px']}; lash lines {r['lid_gap_px']}; under the line on {r['frames_with_the_gap_under_the_line']} frames")
    print(f"  cannot see: {got['cannot_see']}")
    cues = got.get("where_a_face_may_not_show")
    if cues is not None:
        print("WHERE A FACE MAY NOT SHOW, for the sheet (data; nothing here decides, and a face's lines are left out only on a person's word, job --no-face-on):")
        print("\n".join(face_cue_lines(cues)))
    print(f"THE FORM: {form}" + (f", on {a.by}'s word: {why}" if form == "plain" else " (the default; the plain cut is taken only with --plain \"why\")")
          + f". The job file reads {d['loads'][0]['motion_file']}. Open the sheet beside it.")
    print(f"NEXT, THE TEXT, generated from this record and the plan: bench/shot_to_load.py text {a.job_file} --by {a.by} --plan PLAN.json (or a text "
          f"written whole at {ctx['text']}: --by-hand \"why\" --writer \"who, from which frames\"). Then: bench/load_chain.py first {a.job_file} --by ...")


def step_text(a, world) -> None:
    ctx = load_chain.read_job(Path(a.job_file), world.data.parent, need_text=False)
    label, text = ctx["load"]["text"], ctx["text"]
    if a.by_hand is not None and not str(a.writer or "").strip():
        raise Refused("--writer takes a person's words: who wrote the text, and from which frames")
    if a.by_hand is None and a.writer is not None:
        raise Refused("--writer says who wrote a text written whole (--by-hand); a generated text is the generator's, and names no writer")
    # GENERATED BY DEFAULT (the owner's word of 2026-10-14: the text says exactly what the reference shows, and the least else):
    # every line from the reference's record, the plan and the voice table (`bench/load_text.py`), with no place-and-action sentence
    # since 2026-10-15 (`load_text.NO_SCENE`). A text written whole by a person is the exception, with its reason (`--by-hand`).
    import load_text
    generated = None
    if a.place_and_action is not None:
        raise Refused(load_text.NO_SCENE)
    if a.by_hand is None and ctx.get("also"):
        raise Refused("a load of two subjects takes its text written whole (--by-hand): bench/load_text.py writes one subject's")
    if a.by_hand is None:
        if not a.plan:
            raise Refused("the generated text reads the plan's words for the subject and the video: give --plan")
        voice = world.voice(Path(ctx["job"]["source"]).name)
        try:
            generated, made_from = load_text.for_job(Path(a.job_file), "", Path(a.plan), Path(voice["spans"]))
        except load_text.Refused as e:
            raise Refused(str(e)) from None
        if text.is_file() and text.read_text().strip() and text.read_text() != generated:
            raise Refused(f"{text} holds another text: a text is written once; a load made again is a new text")
        print(generated, end="")
    elif not str(a.by_hand).strip():
        raise Refused("--by-hand takes the reason in words: a text written whole by a person is the exception")
    elif not text.is_file() or not text.read_text().strip():
        raise Refused(f"the load's text {text} is not written, or is empty: it is written first, and then its writer is said")
    planned = world.data / f"{ctx['day']}_{ctx['job']['name']}_job" / "job.plan.json"
    if planned.exists():
        raise Refused(f"the plan of {ctx['name']} is built ({planned}): its job file is the one the plan was built from and is not edited after. "
                      "A text or a writer that changed is a load made again")
    d = json.loads(Path(a.job_file).read_text())
    was = str(d["job"]["texts"][label].get("by") or "").strip()
    said = (f"GENERATED by bench/load_text.py from "
            + ("no reference video (motion none), the plan" if ctx["load"].get("motion") == "none" else "the reference's record, the plan")
            + " and the voice table, with no place-and-action sentence (load_text.NO_SCENE)" if generated is not None else
            f"{' '.join(a.writer.split())}, BY HAND: {' '.join(a.by_hand.split())}") + f"; entered {load_chain.utc_now()} by {a.by} (bench/shot_to_load.py text)"
    print(f"texts.{label}.by: {said}" + (f"\n  REPLACES: {was}" if was else ""))
    if world.dry:
        print("DRY: nothing written.")
        return
    if generated is not None:
        text.write_text(generated)
        d["job"]["texts"][label]["generated_from"] = made_from
    d["job"]["texts"][label]["by"] = said
    Path(a.job_file).write_text(json.dumps(d, indent=1))
    print(f"wrote {a.job_file}. NEXT: bench/load_chain.py first {a.job_file} --by {a.by}")


#: A pair's subjects in the order the builder takes them: the larger region first (`masked_job_graph.one_subject_view`: "put first the
#: subject whose region is larger"). Reasoned from the builder's rule. A whole person is not here: the delivery table lays no whole-person
#: render, so a pair of one is refused.
PAIR_RANK = ("face and hair", "face")
#: Why a pair is rendered on the whole frame (docs/wiki/masked_v2v.md: a two-subject load is not rendered zoomed in).
PAIR_WHOLE_FRAME_WHY = "a load of two subjects is rendered on the whole frame: the zoom is to one subject's part, and a pair has two"
#: What the two jobs' file defaults may differ in: the margin, where the pair takes the larger (one margin a load: the node grows its
#: region by one number), with the `margin_by` of the subject whose margin it took (the words behind a `--margin`; none where that
#: subject's is the region's own); and `seed_by`, the words behind a `--seed` (who, when), which differ between two loads of one seed made
#: at two times: the seed itself must be the same, and the pair carries the first subject's `seed_by` with each subject's own row among its
#: defaults. Any other setting that differs is refused, since the pair would carry one of them silently.
PAIR_DEFAULTS_MAY_DIFFER = ("margin", "margin_by", "seed_by")
#: A subject's rows of THE DEFAULTS its own load carried that a pair does not: it renders on the whole frame, and its text is written whole
#: (the voice is read by a generated text only). The pair's own row says each.
PAIR_NOT_CARRIED = ("zoom", "the text", "the voice")


def pair_job(one: dict, two: dict, regions: dict, name: str, day: str, output_key: str, by: str, now: str) -> dict:
    """The job file of a load of two subjects, from two chain-made one-subject loads' own (`step_pair`): nothing tracked or posed again."""
    s1, s2, l1, l2, j1, j2 = one["subject"], two["subject"], one["load"], two["load"], one["job"], two["job"]
    first, frames = l1["first"], l1["frames"]
    keys = lambda ctx: next((s for s in ctx["job"]["subjects"] if s["label"] == ctx["subject"]), {})       # noqa: E731
    entries = []
    import masked_job_graph as graphs
    for ctx, other in ((one, s2), (two, s1)):
        # a subject's own: its load's, over its file's defaults (a key the builder reads off the list is never a default of a pair)
        l, k = {**{x: v for x, v in ctx["defaults"].items() if x in graphs.PAIR_KEYS}, **ctx["load"]}, keys(ctx)
        entry = {"subject": ctx["subject"], "region": regions[ctx["subject"]], **({"still": k["still"]} if k.get("still") else {})}
        entry.update({key: l[key] for key in ("motion_file", "motion_first", "motion_edge") if key in l})
        entry["others"] = [o for o in l.get("others") or [] if o != other]
        if ctx is one:
            entry.update({key: l[key] for key in ("hair_band", "hair_band_cuts") if key in l})
        entries.append(entry)
    # THE KEEP-OUTS, both subjects' as one list, each saying whose own things it is (for the gate's given-back: a restore of the same
    # subject's). A file both jobs name is one keep-out (the strangers, made from the one preview). A name both jobs give to their own
    # things in different files (each subject's `hands`) is the second's under `<name>_<subject>`, in its own `others` too.
    made, seen, renamed = [], {}, {}
    for ctx in (one, two):
        for k in (ctx["job"].get("keep_outs") or {}).get("made", []):
            kept = k["name"]                  # never `name`, the pair's own (on 2026-10-15 the last keep-out's name became the job's)
            if kept in seen and seen[kept]["track"] == k["track"]:
                continue
            if kept in seen:
                if kept == "strangers":
                    raise Refused(f"both jobs name a strangers keep-out, in different files ({seen[kept]['track']} and {k['track']}): one preview has one")
                renamed[(ctx["subject"], kept)] = kept = f"{kept}_{ctx['subject']}"
            seen[kept] = k
            made.append({**copy.deepcopy(k), "name": kept, **({"of": ctx["subject"]} if kept != "strangers" else {})})
    entries[1]["others"] = [renamed.get((s2, o), o) for o in entries[1]["others"]]
    margin = max(one["defaults"].get("margin", 0), two["defaults"].get("margin", 0))
    # the words behind the margin the pair takes: the subject's whose margin it is (the first's where both are that margin), never the
    # other's, whose margin the pair does not carry
    margin_by = next((ctx["defaults"]["margin_by"] for ctx in (one, two) if ctx["defaults"].get("margin") == margin and "margin_by" in ctx["defaults"]), None)
    subjects = [copy.deepcopy(keys(one)), copy.deepcopy(keys(two))] + [copy.deepcopy(s) for s in j1["subjects"] if s["label"] not in (s1, s2)]
    note = (f"Written by bench/shot_to_load.py pair for {by}, {now}. A LOAD OF TWO SUBJECTS, made from two chain-made loads of the same frames: "
            f"{one['name']} ({s1}, <Picture 1> and <Video 1>) and {two['name']} ({s2}, <Picture 2> and <Video 2>), {s1} first as its region is "
            f"the larger ({regions[s1]} against {regions[s2]}). Their masks, parts, keep-outs, stills and head references are their own; nothing "
            f"was tracked or posed again. Neither subject is kept out of the other. One margin, the larger: {margin} px. RENDERED ON THE WHOLE "
            f"FRAME: {PAIR_WHOLE_FRAME_WHY}. THE TEXT IS NOT WRITTEN: {name}.txt beside this file is a person's, and texts.{name}.by says who wrote it. "
            "The delivery table does not lay a render of two subjects in one pass (bench/delivery_table.py).")
    defaults = [{**x, "what": f"{ctx['subject']}: {x['what']}"} for ctx in (one, two) for x in ctx["job"].get("defaults") or []
                if x["what"] not in PAIR_NOT_CARRIED]
    defaults.append({"what": "the pair", "carries": f"{s1} first (<Picture 1>, <Video 1>), {s2} second (<Picture 2>, <Video 2>); margin {margin} px, "
                     "the larger; rendered on the whole frame; neither subject kept out of the other; the text written whole, by a person, so no "
                     "voice table is read", "from": "bench/shot_to_load.py pair (PAIR_RANK, PAIR_WHOLE_FRAME_WHY, PAIR_NOT_CARRIED)"})
    load = {"name": name, "first": first, "frames": frames, "text": name, "subjects": entries,
            **({"window": l1["window"]} if l1.get("window") is not None else {}),
            **({"context_noise": l1["context_noise"]} if l1.get("context_noise") is not None else {}),
            # the node holds one mask, the union of the two: where either subject's plan covers the face on the context frames, the
            # pair covers both (more regenerated on frames tossed after the decode costs no laid pixel; video_mask.CONTEXT_SHOWS)
            **({"context_shows": "face covered"} if "face covered" in (l1.get("context_shows"), l2.get("context_shows")) else {})}
    if l1.get("save_kept") or l2.get("save_kept"):
        load["save_kept"] = True
    job = {"name": name, "note": note, "source": j1["source"], "first": first, "frames": frames, "canvas": list(j1["canvas"]), "output": output_key,
           "subjects": subjects, "left_out": list(j1.get("left_out") or []) + list(j2.get("left_out") or []),
           "capture": f"data/{day}_{name}", "preview": j1["preview"],
           "texts": {name: {"file": f"{name}.txt", "frames": [first, first + frames - 1], "by": ""}},
           "keep_outs": {"of": [s1, s2], "made": made, "by": {ctx["subject"]: (ctx["job"].get("keep_outs") or {}).get("by") for ctx in (one, two)},
                         **({"renamed": [{"subject": w, "was": n, "is": m} for (w, n), m in renamed.items()]} if renamed else {})},
           **({"track_not": copy.deepcopy(j1["track_not"])} if j1.get("track_not") is not None else {}),
           # the shot both loads are of, from the shots record (`masked_job_graph.shot_span`)
           **({"shot": list(j1["shot"])} if j1.get("shot") is not None else {}),
           # read against the job file's folder (`load_chain.who_is_who_file`), and the pair's may be another: written whole
           **({"who_is_who": str((Path(one["job_file"]).parent / j1["who_is_who"]).resolve())} if j1.get("who_is_who") else {}),
           "restore": [{**r, "subject": ctx["subject"]} for ctx in (one, two) for r in ctx["job"].get("restore") or []],
           "whole_frame": {"why": PAIR_WHOLE_FRAME_WHY, "said_by": "bench/shot_to_load.py pair", "at": now},
           "reference": copy.deepcopy(j1.get("reference") or {}), "reference_2": copy.deepcopy(j2.get("reference") or {}),
           "pair": {"said_by": by, "at": now, "loads": [{"subject": ctx["subject"], "name": ctx["name"], "job_file": str(ctx["job_file"])} for ctx in (one, two)]},
           "defaults": defaults}
    # `carried` and `motion` are worked out for each subject from its region and reference (`one_subject_view`), and a subject's own keys
    # went into its entry: none is a default of the pair
    return {"job": job, "defaults": {**{k: v for k, v in one["defaults"].items() if k not in graphs.PAIR_READS and k != "margin_by"}, "margin": margin,
                                     **({"margin_by": margin_by} if margin_by is not None else {})}, "loads": [load]}


def step_pair(a, world) -> None:
    """A LOAD OF TWO SUBJECTS (the lead's ruling of 2026-10-15): two chain-made one-subject loads of the same frames, as one load the chain
    runs like any other (`bench/load_chain.py` reads it through the builder's `one_subject_view`)."""
    one, two = (load_chain.read_job(Path(f), world.data.parent, need_text=False) for f in (a.first, a.second))
    for ctx in (one, two):
        ctx["defaults"] = json.loads(ctx["raw"]).get("defaults") or {}
        if ctx.get("also"):
            raise Refused(f"{ctx['job_file']} is a load of two subjects already")
        if not ctx["load"].get("motion_file") or not Path(ctx["load"]["motion_file"]).is_file():
            raise Refused(f"{ctx['job_file']}: its head reference is not made (the reference step): a pair takes each subject's own")
    if one["subject"] == two["subject"]:
        raise Refused(f"both loads are of {one['subject']!r}")
    for what, x, y in (("source", one["job"]["source"], two["job"]["source"]), ("preview", one["preview"]["folder"], two["preview"]["folder"]),
                       ("canvas", one["job"]["canvas"], two["job"]["canvas"]),
                       ("frames", (one["load"]["first"], one["load"]["frames"]), (two["load"]["first"], two["load"]["frames"]))):
        if x != y:
            raise Refused(f"the two loads differ in their {what} ({x} and {y}): a pair is two subjects of one load's frames, from one preview")
    # what one load carries once, and both jobs carry: the window (a person's --seam), the kept frames' noise (--context-noise), whose track
    # is not whose (`track_not`), who is who
    for what, x, y in (("load's window (--seam)", one["load"].get("window"), two["load"].get("window")),
                       ("load's context_noise (--context-noise)", one["load"].get("context_noise"), two["load"].get("context_noise")),
                       ("track_not", one["job"].get("track_not"), two["job"].get("track_not")),
                       ("who_is_who", one["job"].get("who_is_who"), two["job"].get("who_is_who"))):
        if x != y:
            raise Refused(f"the two loads differ in their {what} ({x} and {y}): a load carries one, so the two are made with the same")
    differ = sorted(k for k in set(one["defaults"]) | set(two["defaults"]) if k not in PAIR_DEFAULTS_MAY_DIFFER and one["defaults"].get(k) != two["defaults"].get(k))
    if differ:
        raise Refused(f"the two loads' settings differ in {', '.join(differ)} ({', '.join(f'{k}: {one['defaults'].get(k)!r} and {two['defaults'].get(k)!r}' for k in differ)}): "
                      "a load carries one of each")
    plan = json.loads(Path(a.plan).read_text())
    regions = {ctx["subject"]: (plan.get("subjects") or {}).get(ctx["subject"], {}).get("region") for ctx in (one, two)}
    for who, region in regions.items():
        if region not in PAIR_RANK:
            raise Refused(f"the plan's region for {who} is {region!r}; a pair takes one of {PAIR_RANK} (a whole person is not laid by the "
                          "delivery table)")
    if PAIR_RANK.index(regions[one["subject"]]) > PAIR_RANK.index(regions[two["subject"]]):
        raise Refused(f"{one['subject']} ({regions[one['subject']]}) is given first and {two['subject']} ({regions[two['subject']]}) has the larger region: "
                      "the larger goes first (masked_job_graph.one_subject_view)")
    for key in ("hair_band", "hair_band_cuts"):
        if key in two["load"]:
            raise Refused(f"the second load has `{key}`: a band is cut under the first subject's part (one_subject_view); make the pair without it")
    held = [k["name"] for k in (two["job"].get("keep_outs") or {}).get("made", []) if k["name"] == "held"]
    if held:
        raise Refused("the second load has a held keep-out of its own: the capture's held split reads the first subject's chin (--chin-points), "
                      "so it would be split on the wrong person")
    import masked_job_graph as graphs
    folder, video = Path(a.folder), load_chain.video_of(one, world.data)
    taken = lambda n: name_taken(n, folder, world)   # noqa: E731
    name = free_name(a.name or f"{one['subject']}_{two['subject']}_{one['load']['first']:04d}", taken, a.remake)
    if not graphs.NAME.match(name):
        raise Refused(f"the name {name!r}: letters, digits and underscores only")
    now = load_chain.utc_now()
    d = pair_job(one, two, regions, name, a.day, OUTPUT_KEY.format(video=video, session=a.by), a.by, now)
    try:
        graphs.one_subject_view({**d["defaults"], **d["loads"][0]})
    except ValueError as e:
        raise Refused(f"the builder refuses the pair: {e}") from None
    print(f"THE PAIR {name}: {one['subject']} first (<Picture 1>, <Video 1>), {two['subject']} second (<Picture 2>, <Video 2>), source frames "
          f"{one['load']['first']}-{one['load']['first'] + one['load']['frames'] - 1}; keep-outs {', '.join(k['name'] for k in d['job']['keep_outs']['made']) or 'none'}")
    print("THE DEFAULTS THIS LOAD CARRIES:")
    for x in d["job"]["defaults"]:
        print(f"  {x['what']}: {x['carries']} ({x['from']})")
    out = folder / f"{name}.json"
    if world.dry:
        print(f"DRY: {out} not written.")
        return
    out.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {out}. NEXT: the text, written whole beside it as {name}.txt (a pair's text is a person's until bench/load_text.py writes "
          f"two-subject texts), then bench/shot_to_load.py text {out} --by-hand WHY --writer WHO --by {a.by}; then bench/load_chain.py first {out}")


STEP = {"preview": step_preview, "job": step_job, "reference": step_reference, "text": step_text, "pair": step_pair}


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
    sub.choices["preview"].add_argument("--source", metavar="VIDEO",
                                        help="required: the source video for this preview (never defaults to shots.json's raw source); "
                                             "on a multi-subject shot where an earlier subject is already laid, pass the post-Pass-1 video")
    sub.choices["preview"].add_argument("--allow-raw-source", metavar="WHY",
                                        help="override the Pass-2 raw-source refusal when another subject is already laid on this shot, with a recorded reason")
    sub.choices["preview"].add_argument("--for", dest="made_for", action="append", default=[], metavar="LABEL",
                                        help="the replacement this preview is made for, a label of the shots record on this shot; repeatable. Required "
                                             "where anything is laid on the shot: the source is then held to the laid pieces of the others alone")
    sub.choices["preview"].add_argument("--cast", help="the cast file: a subject with an answer inside the shot is seeded from it")
    sub.choices["preview"].add_argument("--ask", action="append", default=[], metavar="SUBJECT:why",
                                        help="track a subject the record is `unsure` of on this shot, because the preview is made to ask the owner who is on it; "
                                             "repeatable. No load is built on it until they have answered")
    sub.choices["preview"].add_argument("--seed-cuts-of", metavar="JOB.json=TABLE.json",
                                        help="the tracker's own cuts of this shot, read from the shots table an earlier preview of the same frames saved (with "
                                             "that preview's job file): each cut that holds an answer in the cast file gets its own seed. Without it a shot "
                                             "gets one seed")
    sub.choices["preview"].add_argument("--no-memory-selection", metavar="WHY",
                                        help="track with ComfyUI's last-six-frames memory, not SAM 3.1's memory selection (the chain's default); the "
                                             "reason is kept in the preview's job file and named in each load's defaults")
    sub.choices["preview"].add_argument("--head-per-stretch", metavar="WHY",
                                        help="track each subject's head per stretch of its person track (the head seeded inside each "
                                             "stretch, main's span head where a stretch gives none) in place of main's span head track, "
                                             "the default; the reason is kept in the preview's job file and its trackers' shot tables")
    sub.choices["preview"].add_argument("--no-head-track", metavar="WHY",
                                        help="do not propagate the subject's head with SAM 3.1 (`track_head=False`) in this preview's trackers "
                                             "(the chain's default is `track_head=True`); the reason is kept in the preview's job file and named "
                                             "in each load's defaults")
    sub.choices["preview"].add_argument("--queue", action="store_true", help="build and queue the preview, on an empty queue")
    sub.choices["preview"].add_argument("--remake", action="store_true", help="the next free letter after the name, for a preview made again; the earlier one stands")
    j = sub.choices["job"]
    j.add_argument("--subject", required=True)
    j.add_argument("--preview", required=True, help="the job folder of the shot's preview")
    j.add_argument("--allow-raw-source", metavar="WHY",
                   help="override the Pass-2 raw-source refusal when another subject is already laid on this shot, with a recorded reason")
    j.add_argument("--first", type=int, help="the load's first frame, when a look cut the shot's")
    j.add_argument("--frames", type=int, help="the load's frame count, when a look cut the shot's")
    j.add_argument("--why", help="what a look cut and why; into the job's left_out, for both ends of the load unless one has its own")
    j.add_argument("--why-before", help="what was cut BEFORE the load and why, when it is not the reason for the frames after")
    j.add_argument("--why-after", help="what was cut AFTER the load and why, when it is not the reason for the frames before")
    j.add_argument("--remake", action="store_true", help="the next free letter after the name, for a load made again")
    j.add_argument("--parts", nargs="+", metavar="NAME", help="this load's part is these of its subject's plan parts and no other (hair, face_and_neck, ...); "
                   "cut from the preview's class map as a keep-out is, and handed to the capture as the load's part")
    j.add_argument("--parts-by", metavar="WHO", help="who chose these parts for this load, and on what; kept in the job file")
    j.add_argument("--track-not", action="append", default=[], metavar="SUBJECT:FIRST-LAST=WHY",
                   help="another subject's preview track is not that person on these source frames (it lies on somebody else): every step reads a "
                        "copy with those frames emptied; with --track-not-by; repeatable")
    j.add_argument("--track-not-by", metavar="WHO", help="who says the track is not that person there, and on what look; kept in the job file")
    j.add_argument("--parts-on", action="append", default=[], metavar="FIRST-LAST=NAME[+NAME]",
                   help="on these source frames the load's part is these of the plan's parts, and elsewhere --parts or the plan's (a part that changes "
                        "inside one load); with --parts-by; repeatable")
    j.add_argument("--zoomed", metavar="WHY", help="render this load zoomed in on what is replaced, with a person's reason kept in the job file "
                   "(the chain's default since 2026-10-14 without it)")
    j.add_argument("--no-mouth-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="no mouth drawn on those source frames of the reference, on a person's word: who looked, at which sheet of those "
                        "frames (its file named; only that a file of that name is there is checked), and that her mouth is hidden there; repeatable")
    j.add_argument("--mouth-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="a mouth drawn on those source frames of the reference where its gate drops it (the pose model's lip rings, their inside "
                        "dark), on a person's word: who looked, at which sheet of those frames (its file named; only that a file of that name is "
                        "there is checked), and that her mouth is open there; refused against --no-mouth-on and --no-face-on; repeatable")
    j.add_argument("--no-face-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="no line of the face drawn on those source frames of the reference (the plain cut body model alone), on a person's word: "
                        "who looked, at which sheet of those frames (its file named; only that a file of that name is there is checked), and that "
                        "the face does not show there (a head turned away, hidden behind somebody). The capture's grade then drops the part's face "
                        "classes on those frames and keeps the rest of it, her hair (a face-only part is emptied there; "
                        "capture_masked_run.grade_holds); repeatable")
    j.add_argument("--hold-part", action="append", default=[], metavar="FIRST-LAST[@FRAME]=WHY",
                   help="hold the subject's part on those source frames where the capture's grade leaves it empty, from the nearest graded part, "
                        "on a person's word: who looked, at which sheet of those frames (its file named; only that a file of that name is there is "
                        "checked), and that her part belongs there. With @FRAME (a frame of the load outside the span) the part is held from that "
                        "frame's graded part, placed where it was, not moved, and cut to the track; the capture refuses it where that frame has no "
                        "graded part; repeatable")
    j.add_argument("--seam", metavar="FRAME[,FRAME...]", help="for a load longer than one window: the source frames its later windows write "
                   "from, every seam of one plan of the chained window lengths (refused otherwise, naming the plans there are); with --seam-why")
    j.add_argument("--seam-why", metavar="WHY", help="a person's reason for --seam, kept in the job")
    j.add_argument("--context-noise", type=float, metavar="X", help="for a load longer than one window: the noise its later windows are "
                   "shown the frames they keep at (above 0, at most 1; without it they see them clean, the song node's own). Per load, on a "
                   "person's word, not a default (the lead, 2026-10-15): 0.8 followed a turn on one stretch where clean did not "
                   "(h3_config.MASKED_JOB_CONTEXT_NOISE), and its cost is a seam that may show; a default once a render of a load of several "
                   "windows says it holds. With --context-noise-why")
    j.add_argument("--context-noise-why", metavar="WHY", help="a person's reason for --context-noise, kept in the job")
    j.add_argument("--eyes-shut-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="her eyes shut on those source frames of the reference, on a person's word: who looked, at which table or sheet (its file "
                        "named), and that her eyes are shut there; with --eyes-shut-form; repeatable")
    import draw_face_points
    j.add_argument("--eyes-shut-form", choices=draw_face_points.EYES_SHUT_FORMS,
                   help="what the reference draws for her eyes on the --eyes-shut-on frames: none (no eye, the body model's own), lids (each "
                        "shut as one line), lowered (each iris a black disc inside the lids, no white), or closed (each eye painted over in the skin "
                        "grey round it, one lid line); no default")
    j.add_argument("--no-strangers", metavar="WHY", help="no strangers keep-out on this load, on a person's word: who looked, at the contact "
                   "sheet of the load's frames (its file named; only that a file of that name is there is checked), and that nobody but our "
                   "subjects is in the picture. The part stays held to the subject's own track")
    j.add_argument("--motion", choices=("none",), help="no reference video for this load (a load of her hair alone only, the lead's ruling of "
                                                     "2026-10-15); with --motion-why")
    j.add_argument("--motion-why", help="why the load takes no reference video, a person's words")
    j.add_argument("--whole-frame", metavar="WHY", help="render this load on the whole frame, not zoomed in as the chain's default; the reason is kept in the job file")
    j.add_argument("--model-store", default="", help=f"the folder holding the part model's folder, over the environment's {load_chain.MODEL_STORE}: with it, "
                   "a strangers pass the job needs and its preview lacks is run here, on an empty queue")
    j.add_argument("--cast", help="the video's cast file (data/<stem>/cast/cast.jsonl): the owner's answers on which numbered person is ours, read for who "
                                  "else is in the shot (bench/others_present.py); each tracker's pick is taken without it, and the job says so")
    j.add_argument("--least", type=int, metavar="PX", help="the capture's floor for the subject's part on this load, in px of the whole frame; with --least-why")
    j.add_argument("--least-why", metavar="WHY", help="why the floor is set there; kept beside it in the job file")
    j.add_argument("--drop", metavar="FIRST-LAST[+FIRST-LAST...]", help="source frame spans where the capture empties the subject's part on this load; with --drop-by")
    j.add_argument("--drop-by", metavar="WHY", help="who says the part is dropped on those frames and why; kept in the job file")
    j.add_argument("--carry-words-from", metavar="OLD_JOB.json", help="carry every person's word from an earlier load's job file for the frames this load covers, "
                   "tagged (carried from <old load>); refuses any word-bearing key it does not know how to carry or any carried span outside this load's frames. "
                   "A seed is per load and is never carried: the old job's `defaults.seed` is named under RETIRED KEYS NOT CARRIED, and this load takes "
                   "h3_config.SEED unless it is given --seed")
    j.add_argument("--leave-word", action="append", default=[], metavar="KIND:FIRST-LAST",
                   help="with --carry-words-from and --leave-by: leave out a carried span (left_out, drop, hold_part, parts_on, track_not, no_face_on, "
                        "no_mouth_on, mouth_on, eyes_shut_on, seam) on purpose, recording the drop in job.carried_words_from.left_words; repeatable")
    j.add_argument("--leave-by", metavar="WHO_WHY", help="who says the --leave-word span(s) are left out and why; kept in job.carried_words_from.left_words")
    j.add_argument("--no-carry-words", metavar="WHY", help="Why a --remake does not carry words from an earlier job on the same shot and subject.")
    j.add_argument("--seed", metavar="N", help="the song node's seed for this load, in place of h3_config.SEED (the job's own): a whole number from 0 to "
                   f"{SEED_MAX}; with --seed-why. The job file gets it as `defaults.seed` with `defaults.seed_by` (the words, who and when), and THE "
                   "DEFAULTS THIS LOAD CARRIES say it. A seed is per load and a person's word: --carry-words-from does not carry it")
    j.add_argument("--seed-why", metavar="WHY", help="who asked for the seed and why; kept in the job file beside the seed. Goes with --seed")
    j.add_argument("--zoom-up-to", metavar="N", help="let the Masked Source enlarge this zoomed load past its own 2, up to N (at most "
                   f"{ZOOM_UP_TO_MOST:g}), until a small head is the node's target height (video_mask.ZOOM_HEAD_TARGET); off by default, for a test "
                   "(2026-10-22). With --zoom-up-to-why; written as defaults.zoom_up_to with defaults.zoom_up_to_by, never carried")
    j.add_argument("--zoom-up-to-why", metavar="WHY", help="who asked for the larger zoom and why; kept in the job file beside it. Goes with --zoom-up-to")
    j.add_argument("--margin", metavar="N", help="the px the Masked Source grows this load's region by, in place of h3_config.MASKED_LOAD_MARGIN by "
                   f"the plan's region: a whole number from 1 to {MARGIN_MOST}, in px of the picture the node renders (the zoomed one on a zoomed "
                   "load); with --margin-why. Written as defaults.margin with defaults.margin_by (the words, who and when), and THE DEFAULTS THIS "
                   "LOAD CARRIES say it. A margin is per load and a person's word: --carry-words-from does not carry it")
    j.add_argument("--margin-why", metavar="WHY", help="who asked for the margin and why; kept in the job file beside it. Goes with --margin")
    j.add_argument("--head-tube", nargs="?", const="opt-in --head-tube", default=None, metavar="WHY",
                   help="enable the person-track-bounded SAM 3.1 whole-head tube + 3-frame smoothed Sapiens2-Seg head pass in capture (off by default); "
                        "optional reason kept in the job file")
    j.add_argument("--depth", nargs="?", const="opt-in --depth", default=None, metavar="WHY",
                   help="measure MoGe-2 depth (bench/measure_moge2_depth.py) on frames where two tracks or a keep-out overlap into the capture folder "
                        "and pass --depth to the files step (off by default); optional reason kept in the job file")
    j.add_argument("--no-head-track", metavar="WHY",
                   help="do not pass the preview's SAM 3.1 head track (`head_track=`) to the capture on this load; the reason is kept in the job file "
                        "and named in THE DEFAULTS THIS LOAD CARRIES")
    j.add_argument("--no-closed-lip-line", metavar="WHY",
                   help="turn off --closed-lip-line for this load (the chain's default since 2026-10-19); the reason is kept in the job file "
                        "and named in THE DEFAULTS THIS LOAD CARRIES")
    r = sub.add_parser("reference")
    r.add_argument("job_file")
    r.add_argument("--plain", metavar="WHY", help="take the plain cut, on a person's word; the reason is kept in the job file")
    r.add_argument("--existing", action="store_true", help="a load made again on the same frames: read the reference that is there, run nothing")
    r.add_argument("--model-store", default="", help=f"the folder holding the pose model's folder; over the environment's {load_chain.MODEL_STORE}")
    r.add_argument("--mouth", choices=("auto", "drawn", "left out"), help="the mouth, over the job's (the plan's or the chain's default), with --form-why")
    r.add_argument("--iris", choices=("between the lids", "whole", "filled", "contrast"), help="the eye, over the job's, with --form-why")
    r.add_argument("--class-mouth-needs-hole", action="store_true", help="an option for one test (5114e): no class-map mouth on a frame "
                   "with no teeth or tongue labelled; with --form-why")
    r.add_argument("--closed-lip-line", action="store_true", help="draw a closed class-map mouth (no teeth or tongue inside it) as one open parting "
                   "line from corner to corner, fit the mouth above the chin, and clip every mark to the grey head's silhouette (the chain's default "
                   "since 2026-10-19); with --form-why when overriding at the reference step")
    r.add_argument("--no-closed-lip-line", action="store_true", help="turn off --closed-lip-line (the chain's default since 2026-10-19) at the "
                   "reference step; with --form-why")
    r.add_argument("--darken-head", action="store_true", help="an option, off unless asked: darken the grey body model by scale (HEAD_DARKEN_SCALE) so white lines "
                   "and eyes read with higher contrast; with --form-why")
    r.add_argument("--form-why", default="", help="why this load's mouth or eye is drawn otherwise than its job says")
    r.add_argument("--no-head-track", metavar="WHY", help="pass --no-head-track to head_reference.py (use the full body matte instead of "
                   "head_track for the crop box); the reason is kept in the job file")
    t = sub.add_parser("text")
    t.add_argument("job_file")
    t.add_argument("--writer", metavar="WORDS", help="with --by-hand: who wrote the load's text, and from which frames; into texts.<label>.by")
    t.add_argument("--place-and-action", metavar="SENTENCE", help="refused since 2026-10-15 (load_text.NO_SCENE): a generated text describes no "
                   "frozen picture; without --by-hand the text is made from the reference's record, the plan and the voice table")
    t.add_argument("--plan", help="the video's plan file, for its words for the subject and the video")
    t.add_argument("--by-hand", metavar="WHY", help="take the text a person wrote whole beside the job file, with the reason")
    pr = sub.add_parser("pair")
    pr.add_argument("first", help="the job file of the subject with the larger region: <Picture 1> and <Video 1>")
    pr.add_argument("second", help="the job file of the other subject, of the same frames and preview: <Picture 2> and <Video 2>")
    pr.add_argument("--plan", required=True, help="the video's plan file, for each subject's region")
    pr.add_argument("--folder", required=True, help="where the pair's job file is written")
    pr.add_argument("--name", help="the pair's name; <first>_<second>_<first frame> when left out")
    pr.add_argument("--day", default=today, help="the day in the names of the pair's folders; today when left out")
    pr.add_argument("--remake", action="store_true", help="the next free letter after the name, for a pair made again")
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
