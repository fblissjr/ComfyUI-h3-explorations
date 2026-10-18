#!/usr/bin/env python3
"""One load of a masked job, a step at a time, read from its job file: first, plan, on-plan, table, render, land, page.

    <python> bench/load_chain.py first   JOB.json --by SESSION
    <python> bench/load_chain.py plan    JOB.json --by SESSION
    <python> bench/load_chain.py on-plan JOB.json --by SESSION
    <python> bench/load_chain.py band-again JOB.json --by SESSION
    <python> bench/load_chain.py table   JOB.json --by SESSION
        (a reader who did not make the capture looks at the plan's strip: `capture_masked_run.py looked`;
         or, on a clear gate with no forcing flag, the maker records that no look is owed:)
    <python> bench/load_chain.py no-look JOB.json --by SESSION --not-first WORDS.txt
    <python> bench/load_chain.py render  JOB.json --by SESSION --grounds GROUNDS.txt --show SHOW.txt
    <python> bench/load_chain.py land    JOB.json --by SESSION [--model-store FOLDER]
    <python> bench/load_chain.py page    JOB.json --by SESSION --spec WORDS.json

**What it buys.** Until 2026-10-13 a load went from its preview to the page through five session scripts copied
from session to session with the session's name changed, taking thirteen arguments typed for each load and
holding the day's date, one video's paths and one machine's folders in their text. One typed argument was wrong
on two loads of that day: the source frame a preview's mask files begin on was given as the load's own first
frame, and the masks sat four and eight frames off the picture in captures that verified and gated clear. This
is those scripts once, tracked, with a check. It takes the job file and reads the rest.

**The principle** (the lead's ruling, 2026-10-13). The job file is the one record a load is made again from, so
what a load depends on is a key in it or a named file it points at; a fact of the video lives with the video; a
fact of the machine is an environment key; nothing is found by a name left over from an earlier run.

**What is read, and from where.**

- The job file (`workflows/masked_job_graph.py` has its keys): the load, its subject, its frames, its motion
  video, its text; the copy, the span, the canvas, the output key; `capture`, the capture's folder, whose name
  begins with the day; `preview`, THE JOB FOLDER OF THE PREVIEW the load's masks come from (new with this tool).
  One load a job file; more is refused.
- The preview's own job folder: the files it saved (`preview.queued.json`), the subjects in its order and THE
  SOURCE FRAME ITS FILES BEGIN ON (its job's `first`), which is the `at`, `shots_at` and `pose_at` of every mask.
  Never typed.
- The load's keep-outs (`job.keep_outs`: things on the subject kept as the source's, each a mask video on the
  preview's frames, written by `bench/shot_to_load.py job` from the plan): each is handed to the capture as a
  `--mask` of its own and is one of the run's others. Refused: one the job names and the load's `others` does not,
  one the load keeps out with no file named, a file that is not there.
- A subject's options for the capture (`pieces`, `least`, `hold`, `drop`, `grade`) are keys on the subject in the
  job file, and only what the job says is passed. KNOWN GAP: the loads made before this tool ran with
  `pieces=largest` typed into the script, and their job files do not say so; nothing is backfilled.
- The video's voice tables from its analysis folder's manifest (`clip_analysis.py voice`); refused when absent, and
  when the per-frame table does not cover every frame of the load (the frames it covers are read from the table).
- The pose model's store from `H3_MODEL_STORE` (the folder holding one named `h3_config.POSE_MODEL`), or
  `--model-store`; refused without. `docs/comfy_notes.md` lists the key.
- The words only a person has: `--grounds` and `--show` for a render (who relayed the go and on what it rests;
  what the render is to show), `--spec` for the page (its `title_rest`, `state_words`, `what`, `look`,
  `prompt_note`). Refused without.

**The steps, and what each refuses.**

- `first`: who is who from the preview's shot tables (made again every time, or the file the job names as
  `who_is_who`); the first capture AT `job.capture` with the load as a plan worked out from the masks; verify;
  the gate; the strip, drawn under `<capture>_first` on the share from the start. Refused when the capture or
  its `_first` copy exists. Verify, the gate and the strip each run whatever the one before said, and THE STEP'S
  LAST LINE AND ITS EXIT STATUS SAY HOW THEY ENDED: 0 when verify passed, the gate is clear and the strip was
  drawn; else the first of the three that was not 0 (a failed verify is the capture tool's 4, a blocked gate
  its 3), with the words in capitals. `on-plan` ends the same way.
- `plan`: the node's plan of the load, built and queued. Refused when the queue is not empty, when there is no
  first capture, when a plan graph of this load is already built, when the first capture's gate is not clear (a
  blocked flag is cleared by the load made again or by a person's override of it, by name, in the capture), and
  when the builder's own dry build refuses the load (a capture that did not pass verify is one of its refusals).
- `on-plan`: the capture again, on the node's plan. With the queue empty: the frame-by-frame hash of every mask
  video the plan graph reads is recorded; the first capture is COPIED to `<capture>_first` and the copy checked
  file for file before anything is overwritten; then the capture is built in place on the plan, verified, gated,
  its strip drawn, and its per-frame table written (`bench/frame_table.py`, for both subjects on a two-subject
  load). Nothing is renamed or moved; when the frame table succeeds, its path is recorded in `<load>.plan.built.json`.
  On a load with a hair band, after verify and the gate, the plan's band file is written again from the carved track
  (`band_again`; its own bytes kept under `<job folder>_first/`), so the render's decodes equal.
- `band-again`: the same, for a banded load whose on-plan ran before that (Gemini's D13 A1, the lead's ruling of
  2026-10-16). Refused on a load with no band, one whose on-plan has not run (no record hashed after the rebuild),
  one whose band was already made again, and once a render graph or render go exists.
- `table`: builds or rebuilds the capture's per-frame table (`bench/frame_table.py`, and for the second subject on
  a two-subject load) in place on `<capture>`, and records its path and sha256 in `<load>.plan.built.json`. Holds the
  job file to `job.plan.json` (`job_as_planned`), and refuses when the capture is not there, when a render graph or
  render go already exists, or when `frame_table.py` fails.
- `no-look`: the load's MAKER records that no look is owed, where the owner's go-fast word allows a render on the
  gate alone (`FORCING_WORD`): the capture on the node's plan, its gate clear, and NONE of the forcing rules
  standing on its flags or its head reference (`FORCING_FLAGS`, `FORCING_FIGURES`, another subject's clothing or
  hands inside the region, the reference's read-backs), with the maker's own words that the load is not the first
  of its kind (`--not-first`, a file). It writes `no_look_owed.json` into the capture: a record of its own kind,
  NOT a look and never counted as one, holding whose word it rests on, the gate's verdict, the flags read and
  what of the list no flag says. REFUSED, with the flag named, when a forcing rule stands: a look by somebody
  who did not make the capture is then owed as ever.
- `render`: the queue empty; the last look on this load, on the capture on the node's plan, later than the
  capture (compared as times; refused when the two cannot be set in one zone) and not by the capture's maker,
  OR, where there is no such look, the maker's record that none is owed, later than the capture and held again
  to the forcing rules as the capture stands (the go file and the render's record then say NO LOOK, RENDERED ON
  THE GATE ALONE); the gate clear; the per-frame table (`frame_table.jsonl`, and `frame_table__<also>.jsonl` on a
  pair) present and its recorded `input_sha256` matching the files on disk; the job file byte-equal to the one
  the plan was built from (`job.plan.json`); the go written
  to `<load>.render.go.md` BESIDE the job file (the job file is not edited after its plan is built); the render
  graph built, differing from the plan graph in the song node's `preview` and `filename_prefix` only (and, for
  a hair-band load, in the band's own keep-out file when the two decode equal), with no uncommitted node code;
  every mask video the render graph reads decoding equal to what the plan read. Then it is queued.
- `land`: the queue empty (no pose pass beside a render); the pose pass on the one window's own video, or for a
  load of several windows on the joined file, held to the load's frame count first; the render's cost lines from
  the server's history; the card freed first (`POST /free`: the pass runs in its own process and the server keeps
  a render's models resident; said, since the next thing queued then starts cold); the pose pass in the subject's
  tracked box; the stacked clip. A POSE PASS THAT DOES NOT FINISH IS A LAND THAT DID NOT HAPPEN: the clip is still
  made, the step ends with `POSE_PASS_FAILED` and the reason in capitals, and run again with the clip there it does
  the pass alone. Landed is the render's points written.
- `page`: the render's item at the top of the video's review list, its buttons `delivery_table.RENDER_ITEM_BUTTONS`
  (the words the generated delivery table reads), a copy of the list kept in the job folder first.

Every command a step runs is printed before it runs and its output kept under the job folder (`chain/`); an
output there from an earlier run of the same command is kept beside the new one, never replaced.
`load_chain.py STEP --help` prints that step's own words from this page.
`--dry` prints the commands and runs none; it still refuses what the step refuses on the files alone. For `plan`
that includes everything the builder refuses: its build is run with `dry`, which writes nothing.

**What it cannot see.** Whether the masks are on the right person (the look is for that). Whether the preview
it is pointed at is the right one for the shot. Whether a mask file that still decodes equal was made from the
right frames to begin with: it holds the capture to the preview by the preview's own first frame, and no more.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

STEPS = ("first", "plan", "on-plan", "band-again", "table", "no-look", "render", "land", "page")
HOST = "127.0.0.1:8188"
#: The environment key for the folder that holds the pose model's own folder (`h3_config.POSE_MODEL`): a fact of
#: the machine, so never a path in a tracked file. Named 2026-10-13 with the lead.
MODEL_STORE = "H3_MODEL_STORE"
#: A subject's keys in the job file that are options of its `--mask` in a capture; only what the job says is passed.
MASK_OPTIONS = ("pieces", "least", "hold", "drop", "grade", "word")
#: Default canvas-pixel margin for `capture_masked_run.off_head` (`points_margin=`). **Inherited**: one H3 latent cell on the
#: canvas (16 px, matching `capture_masked_run.CELL`), named apart from the load's region `margin` so widening a load's region does
#: not widen the off-head head box or chain reach (D22 item 6.2).
OFF_HEAD_MARGIN = 16
#: Where one video's review clips and review list live. Today's practice, in this one place (the Open List's row
#: z21 may change the folders; then here). `{video}` is the analysis folder's name.
REVIEW_CLIPS = "Video/{video}/review_clips"            # under the server's output folder
REVIEW_ITEMS = "data/{video}/review/items.json"        # under the repo
REVIEW_MEDIA = "data/{video}/review/media"             # links to the clips, where the page reads them
#: The song node's inputs a render graph may differ from its plan graph in.
RENDER_DIFFERS_IN = ["filename_prefix", "preview"]
BAND_SUFFIXES = (".plan.clothes_less_hair_band.mkv", ".render.clothes_less_hair_band.mkv")
#: What the capture tool exits with for a verify that failed and for a gate that is blocked
#: (`bench/capture_masked_run.py`, the same two names); `bench/check_load_chain.py` holds them to that file.
VERIFY_FAILED, GATE_BLOCKED = 4, 3
#: `land`'s exit status when its pose pass did not finish. Reasoned: not 5 (a refusal: nothing was run), nor the
#: capture's two; a land with no render points is a land that did not happen, whatever else it made.
POSE_PASS_FAILED = 6
#: Seconds waited after `POST /free` before the card is read: the server frees on its next turn of its own loop.
#: Reasoned, a starting value; the figure printed is what the server says then, whatever it is.
FREE_SETTLES_S = 3
#: How much of a line of a command's output is shown on the screen; the kept file has it whole.
TAIL_CHARS = 300
#: A window's own video in a load's windows folder: `<name>_window_<N>.mp4`, and not its mask review or its zoomed picture.
WINDOW_VIDEO = r"_window_(\d+)\.mp4$"


class Refused(Exception):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the job file, and what it points at

def read_job(job_file: Path, repo: Path = REPO, need_text: bool = True) -> dict:
    """Everything a step reads of a load, from its job file and the preview's job folder it names.

    Refused: more than one load; no `capture` or one whose folder name does not begin with a day; no `preview`,
    or a preview whose record is not of a finished run; a subject or keep-out the preview did not track; and, unless
    `need_text` is off (a tool that runs before the text is written), a text that is not a file or that names no
    writer: the text is a person's."""
    raw = Path(job_file).read_bytes()
    d = json.loads(raw)
    job, loads = d["job"], d.get("loads") or []
    if len(loads) != 1:
        raise Refused(f"{job_file} holds {len(loads)} loads: this tool takes one load a job file")
    load = loads[0]
    # A LOAD OF TWO SUBJECTS (`subjects`, a pair made by `bench/shot_to_load.py pair`; the lead's ruling of 2026-10-15, so a pair runs
    # through the chain like any load): read as the builder reads it (`masked_job_graph.one_subject_view`): the first subject is the
    # load's, the second is `also`, `others` the keep-outs of both and neither subject. Both subjects are read for: their parts, class
    # maps and pose tables to the capture; their keep-outs, drops and restores to the gate; the forcing rules and both references'
    # records; a pose pass for each at the land. The held keep-out's split reads the first subject's chin (`--chin-points`), so
    # `pair` makes no load whose second subject has a held keep-out of its own.
    also = None
    if "subjects" in load:
        sys.path.insert(0, str(repo / "workflows"))
        import masked_job_graph
        try:
            load = masked_job_graph.one_subject_view({**(d.get("defaults") or {}), **load})
        except ValueError as e:
            raise Refused(f"{job_file}: {e}") from None
        also = load["also"]["subject"]
    if not job.get("capture"):
        raise Refused(f"{job_file}: the job names no `capture`, the folder its capture is made in")
    capture = Path(job["capture"]) if Path(job["capture"]).is_absolute() else repo / job["capture"]
    day = capture.name[:10]
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        raise Refused(f"{job_file}: the capture folder {capture.name!r} does not begin with a day (YYYY-MM-DD); the job folder's day is read from it") from None
    if not job.get("preview"):
        raise Refused(f"{job_file}: the job names no `preview`: the job folder of the preview its masks come from "
                      "(data/<day>_<preview>_job). The source frame the masks begin on is read from that preview's own job, never typed")
    pfolder = Path(job["preview"]) if Path(job["preview"]).is_absolute() else repo / job["preview"]
    for name in ("job.json", "preview.queued.json"):
        if not (pfolder / name).is_file():
            raise Refused(f"the preview's job folder {pfolder} holds no {name}")
    pjob = json.loads((pfolder / "job.json").read_text())["job"]
    ran = json.loads((pfolder / "preview.queued.json").read_text())
    if ran.get("status") != "success":
        raise Refused(f"the preview {pjob['name']} did not end in success (its record says {ran.get('status')!r})")
    order = [s["label"] for s in pjob["subjects"]]
    subject = load["subject"]
    for who in (subject, also):
        if who is not None and who not in order:
            raise Refused(f"the load's subject {who!r} is not a subject of the preview {pjob['name']} ({', '.join(order)})")
    mine = next((s for s in job.get("subjects", []) if s["label"] == subject), {})
    also_keys = next((s for s in job.get("subjects", []) if s["label"] == also), {}) if also else None
    span = (pjob["first"], pjob["first"] + pjob["frames"] - 1)
    if not (span[0] <= job["first"] and job["first"] + job["frames"] - 1 <= span[1]):
        raise Refused(f"the job's frames {job['first']}-{job['first'] + job['frames'] - 1} are not inside the preview {pjob['name']}'s, {span[0]}-{span[1]}")
    texts = job.get("texts") or {}
    told = texts.get(load.get("text") or "")
    if not told or not told.get("file"):
        raise Refused(f"{job_file}: the load's text label {load.get('text')!r} names no file in the job's `texts`")
    if need_text and not (Path(job_file).parent / told["file"]).is_file():
        raise Refused(f"the load's text {Path(job_file).parent / told['file']} is not written: it is a person's to write, from the load's own frames")
    if need_text and not str(told.get("by") or "").strip():
        raise Refused(f"{job_file}: texts.{load.get('text')}.by is empty: say who wrote the text, and from which frames "
                      f"(bench/shot_to_load.py text {job_file} --by SESSION --plan PLAN.json, or --by-hand \"why\" --writer \"who, from which frames\")")
    # THE LOAD'S KEEP-OUTS (`job.keep_outs`, written by `bench/shot_to_load.py job` from the plan): mask videos on the
    # preview's frames, each a subject of the capture and one of the load's `others`. Held both ways: a keep-out the
    # job names and the load does not keep out is drawn again in silence; one the load keeps out with no file named
    # cannot be handed to the capture.
    kept = [(k["name"], Path(k["track"])) for k in (job.get("keep_outs") or {}).get("made", [])]
    kept_hash = {k["name"]: k.get("frame_hash") for k in (job.get("keep_outs") or {}).get("made", [])}
    people_named = [o for o in load.get("others") or [] if o in order]
    for name, _track in kept:
        if name not in (load.get("others") or []):
            raise Refused(f"{job_file}: the job names the keep-out {name!r} and the load's `others` does not: it would be drawn again")
        if name in order:
            raise Refused(f"{job_file}: the keep-out {name!r} has the label of a subject of the preview {pjob['name']}")
    # A TRACK THAT IS NOT ITS PERSON ON NAMED FRAMES (`job.track_not`, a person's word written by `shot_to_load.py job
    # --track-not`): another subject's preview track, copied with those frames emptied, is the track every step reads
    # for that subject (`preview_files`). Never the load's own subject: a load is cut where its own person is not.
    for t in (job.get("track_not") or {}).get("made", []):
        if t["subject"] not in order or t["subject"] in (subject, also):
            raise Refused(f"{job_file}: `track_not` names {t['subject']!r}, which is not another subject of the preview {pjob['name']} ({', '.join(order)})")
    stray = [o for o in load.get("others") or [] if o not in order and o not in [n for n, _ in kept]]
    if stray:
        raise Refused(f"{job_file}: the load keeps out {stray}, which is neither a subject of the preview {pjob['name']} ({', '.join(order)}) nor a keep-out "
                      "the job names with its file (`job.keep_outs`): the capture cannot be handed it")
    return {"job_file": Path(job_file), "raw": raw, "job": job, "load": load, "name": load["name"], "subject": subject, "subject_keys": mine,
            "also": also, "also_keys": also_keys,
            "keep_outs": kept, "keep_out_hashes": kept_hash, "others_named": people_named,
            "capture": capture, "day": day, "job_folder": repo / "data" / f"{day}_{job['name']}_job", "repo": repo,
            "preview": {"folder": pfolder, "name": pjob["name"], "first": pjob["first"], "frames": pjob["frames"], "order": order,
                        "files": ran.get("files") or [], "output": pjob["output"]},
            "text": Path(job_file).parent / told["file"]}


def preview_files(ctx: dict, output: Path) -> dict:
    """Each preview subject's files, from the names the preview's record lists: {label: {track, parts, classes, shots, pose}}.

    The track, part and class-map videos are the ones the record lists (so a second numbered save is not
    mistaken for the first); the shot table and the pose table are written by nodes beside them and are found by
    the same prefix. One track a subject, or refused."""
    out = {}
    for label in ctx["preview"]["order"]:
        tracks = [f for f in ctx["preview"]["files"] if f"_{label}_track_" in Path(f).name and f.endswith(".mkv")]
        if len(tracks) != 1:
            raise Refused(f"the preview {ctx['preview']['name']} lists {len(tracks)} track videos for {label!r} where one is meant: {tracks or 'none'}")
        track = output / tracks[0]
        prefix = track.name.split(f"_{label}_track_")[0]
        listed = lambda kind: next((output / f for f in ctx["preview"]["files"] if Path(f).name.startswith(f"{prefix}_{label}_{kind}_") and f.endswith(".mkv")), None)
        shots = sorted(track.parent.glob(f"{prefix}_{label}_shots_*_shots.json"))
        if len(shots) != 1:
            raise Refused(f"{len(shots)} shot tables for {label!r} beside {track.name} where one is meant")
        pose = track.parent / f"{prefix}_{label}_pose_table.json"
        out[label] = {"track": track, "parts": listed("parts"), "classes": listed("classes"), "shots": shots[0], "pose": pose if pose.is_file() else None}
    # a subject's track a person said is not that person on named frames: the copy with those frames emptied (`job.track_not`)
    for t in ((ctx.get("job") or {}).get("track_not") or {}).get("made", []):
        if t["subject"] in out:
            if not Path(t["track"]).is_file():
                raise Refused(f"the job says {t['subject']}'s track is not them on {', '.join(f'{a}-{b}' for a, b in t['frames'])} and its copy "
                              f"{t['track']} is not there: `shot_to_load.py job` makes it")
            # the copy is written by the job step, today; the capture holds it to the preview's own track it was made from
            # (`track_from=`, as a load's own part is held by `parts_from=`): 2026-10-15, second_hfp_0604f's verify failed on its date
            if t.get("cut_from") and Path(t["cut_from"]).resolve() != Path(out[t["subject"]]["track"]).resolve():
                raise Refused(f"the job's copy of {t['subject']}'s track was cut from {t['cut_from']}, and the preview's track of {t['subject']} is "
                              f"{out[t['subject']]['track']}: the job is made again")
            out[t["subject"]].setdefault("track_from", out[t["subject"]]["track"])   # the preview's own, even were a subject named twice
            out[t["subject"]]["track_hash"] = t.get("hash")
            out[t["subject"]]["track_from_hash"] = t.get("cut_from_hash")
            out[t["subject"]]["track"] = Path(t["track"])
            out[t["subject"]]["track_is_not_them_on"] = t["frames"]
    return out


def mask_specs(ctx: dict, files: dict, band: Path | None = None, hashed=None) -> list[str]:
    """The `--mask` of every preview subject for a capture of this load: the load's subject with its part, class map,
    shot table and (unless its job says `pose: false`) its pose table; every other with its track and shot table.

    `band`: on the plan, the keep-out the load's hair band cuts is handed as the plan's own cut file
    (`plan_band_file`), so the capture reads the mask the builder rendered with. Until 2026-10-14 it was handed
    whole, and the gate's `region_over_another_subject` counted the band's own cells: on second_hfp_0484 the
    region covered 0.403 of the whole clothes mask at most, every covered cell inside the band, and 0.000 of the
    cut one.

    `at`, `shots_at` and `pose_at` are THE PREVIEW'S first frame: the source frame its files' frame 0 is on. The
    scripts this replaces passed the capture's own first frame, which is the same number only when a load begins
    where its preview does."""
    at, by = ctx["preview"]["first"], ctx["preview"]["name"]
    out = []
    for label in ctx["preview"]["order"]:
        f = files[label]
        # a track copied on a person's word (`track_not`) carries a hash of its frames from the job step: the capture is handed those frames
        # and no others, as a keep-out is (the cold reader of 2026-10-15: held only to its source's time, the copy itself went unchecked)
        if f.get("track_from"):
            if not f.get("track_hash"):
                raise Refused(f"the job names a copy of {label}'s track with no hash of its frames (written before 2026-10-15's mend): the job is made again")
            if (hashed or frame_hash)(str(f["track"])) != f["track_hash"]:
                raise Refused(f"the copy of {label}'s track at {f['track']} is not the file the job step held to its record: its frames changed since")
            if f.get("track_from_hash") and (hashed or frame_hash)(str(f["track_from"])) != f["track_from_hash"]:
                raise Refused(f"the preview's track of {label} at {f['track_from']} is not the one the job step cut its copy from: its frames changed "
                              "since (the preview made again): the job is made again")
        if label not in (ctx["subject"], ctx.get("also")):
            other_cls = f",classes={f['classes']}" if (ctx["job"].get("depth") or {}).get("on") and f.get("classes") is not None else ""
            out.append(f"{label}:track={f['track']}" + (f",track_from={f['track_from']}" if f.get("track_from") else "")
                       + other_cls + f",shots={f['shots']},shots_at={at},by={by},at={at}")
            continue
        keys = ctx["subject_keys"] if label == ctx["subject"] else (ctx.get("also_keys") or {})   # each subject of a pair, its own
        for kind in ("parts", "classes"):
            if f[kind] is None:
                raise Refused(f"the preview {by} saved no {kind} video for the load's subject {label!r}")
        # A LOAD'S OWN PART (`parts_file` on its subject in the job file, written by `shot_to_load.py job --parts`): some of
        # the subject's parts, cut from the preview's class map on the preview's own frames. The capture is handed it in
        # place of the preview's saved part, so what the builder loads is what the capture read.
        parts, made_from = f["parts"], ""
        if keys.get("parts_file"):
            parts = Path(keys["parts_file"])
            if not parts.is_file():
                raise Refused(f"the job names a part of the load's own for {label!r}, {parts}, and it is not there: `shot_to_load.py job --parts` makes it")
            # made today from the preview's own files: the capture's verify holds it to the preview's saved part it came from
            made_from = f",parts_from={f['parts']}"
        spec = f"{label}:track={f['track']}" + (f",track_from={f['track_from']}" if f.get("track_from") else "") + f",parts={parts}{made_from},classes={f['classes']},shots={f['shots']},shots_at={at}"
        if keys.get("pose", True) is not False:
            if f["pose"] is None:
                raise Refused(f"the job does not say `pose: false` for {label!r} and the preview {by} wrote no pose table for them")
            spec += f",pose={f['pose']},pose_at={at}"
        spec += f",by={by},at={at}"
        for key in MASK_OPTIONS:              # only what the job says: no default of `pieces=largest`
            if key in keys:
                spec += f",{key}={keys[key]}"
        # THE FLOOR HOLDS, NEVER WHERE NO FACE SHOWS (build (B), the lead's ruling of 2026-10-16): the job's --no-face-on spans, a person's look,
        # so the capture holds no part there (`capture_masked_run.grade_holds`, `noface`)
        faceless = [f"{s['first']}-{s['last']}" for s in ((ctx.get("job") or {}).get("reference") or {}).get("no_face_on") or []] if label == ctx["subject"] else []
        if faceless:
            spec += ",noface=" + "+".join(faceless)
        # WHAT LIES OFF HER HEAD (`capture_masked_run.off_head`): the reference step's face points, and `off_head_margin` (default `OFF_HEAD_MARGIN`)
        ref_cfg = (ctx.get("job") or {}).get("reference") or {}
        points = ref_cfg.get("points") if label == ctx["subject"] else None
        if points:
            if not Path(points).is_file():
                raise Refused(f"the job's reference names its face points at {points}, which is not a file: run the reference step")
            pts_margin = int(keys.get("off_head_margin", ref_cfg.get("off_head_margin", OFF_HEAD_MARGIN)))
            spec += f",points={points},points_margin={pts_margin}"
        if (ctx.get("job") or {}).get("whole_frame"):     # the capture's starting floor is held to the zoom; a whole-frame load is not zoomed
            spec += ",zoom=off"
        out.append(spec)
    cuts = band_cuts(ctx) if band is not None else None
    for name, track in ctx.get("keep_outs") or []:
        if not track.is_file():
            raise Refused(f"the keep-out {name!r} is named at {track}, which is not a file: `bench/shot_to_load.py job` makes it when it writes the job file")
        # a keep-out the job step held to its record carries a hash of its frames: the capture is handed those frames and no others.
        # The strangers file always does (bench/strangers.py): one with none was written into a job before the hash was kept.
        want = (ctx.get("keep_out_hashes") or {}).get(name)
        if name == "strangers" and not want:
            raise Refused("the job names the strangers keep-out with no hash of its frames (written before 2026-10-14's second mend): the job is made again")
        if want and (hashed or frame_hash)(str(track)) != want:
            raise Refused(f"the keep-out {name!r} at {track} is not the file the job step held to its record: its frames changed since")
        if name == cuts:                       # the band file's frame 0 is the load's first frame
            out.append(f"{name}:track={band},by={by},at={ctx['load']['first']}")
            continue
        out.append(f"{name}:track={track},by={by},at={at}")
    return out


def band_cuts(ctx: dict) -> str | None:
    """The keep-out the load's hair band is cut out of; None for a load with no band."""
    if "hair_band" not in ctx["load"]:
        return None
    import h3_config
    return ctx["load"].get("hair_band_cuts", h3_config.MASKED_JOB_HAIR_BAND["cuts"])


def plan_band_file(ctx: dict) -> Path | None:
    """The plan's cut keep-out (`masked_job.hair_band` writes it when the plan is built); None for a load with no band.
    Refused when the load has a band and the file is not there, or when the load is not the capture's whole span:
    the file holds the load's frames only, and the capture's frames outside them would read as no keep-out."""
    cuts = band_cuts(ctx)
    if cuts is None:
        return None
    path = ctx["job_folder"] / f"{ctx['name']}.plan.{cuts}_less_hair_band.mkv"
    if not path.is_file():
        raise Refused(f"the load has a hair band and the plan's cut keep-out {path} is not there: the plan step writes it")
    load, job = ctx["load"], ctx["job"]
    if (int(load["first"]), int(load["frames"])) != (int(job["first"]), int(job["frames"])):
        raise Refused(f"the load is on {load['first']}+{load['frames']} and its capture on {job['first']}+{job['frames']}: the band file "
                      "holds the load's frames only")
    return path


def others_of(ctx: dict) -> list[str]:
    """Everybody and everything the capture's run of the load is told is not the subject: the preview's other
    subjects, then the load's keep-outs."""
    return [label for label in ctx["preview"]["order"] if label not in (ctx["subject"], ctx.get("also"))] + [name for name, _ in ctx.get("keep_outs") or []]


def video_of(ctx: dict, data: Path) -> str:
    """The analysis folder's name of the job's video: the name its review clips and its review list are filed under."""
    import clip_folder
    return clip_folder.folder_of_file(Path(ctx["job"]["source"]).name, data).name


def plan_windows(ctx: dict, output: Path) -> Path:
    n = ctx["name"]
    return output / ctx["job"]["output"] / n / f"{n}_plan" / f"{n}_windows"


def render_folder(ctx: dict, output: Path) -> Path:
    return output / ctx["job"]["output"] / ctx["name"] / ctx["name"]


# ---------------------------------------------------------------- what a step refuses, as functions of files

def as_time(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text)


def later(a: str, b: str) -> bool:
    """Is time `a` later than time `b`, as times. Both with an offset, or both without (two times one tool wrote on
    one machine's clock): comparable. One with and one without: refused, since they cannot be set in one zone."""
    ta, tb = as_time(a), as_time(b)
    if (ta.tzinfo is None) != (tb.tzinfo is None):
        raise Refused(f"the times {a!r} and {b!r} cannot be set in one zone: one says its offset and the other does not")
    return ta > tb


def in_utc(text: str) -> str:
    """A record's time in UTC. One written without an offset is this machine's local time, as the capture tool writes it."""
    t = as_time(text)
    return (t.astimezone(datetime.timezone.utc) if t.tzinfo else t.astimezone().astimezone(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- a render on the gate alone

#: WHOSE WORD THE GATE-ALONE PATH RESTS ON. Relayed by the day's leads and set down in their handoffs; not typed by the
#: owner into this file. The standing rule beside it is one look a load by a reader who did not build it.
FORCING_WORD = ("the owner's go-fast word of 2026-10-13 (18:56Z, as the lead's handoff of that day has it): the gate before every render; a look at a "
                "plan's strip by somebody who did not make it only when the gate does not read clear, a forcing flag stands, or the load is the "
                "first of its kind")
#: THE FORCING RULES, in this one place. A load whose capture holds one is owed a look by somebody who did not make it,
#: whatever the gate says. Inherited, word for word in what they name, from that list. Any flag of these four rules:
FORCING_FLAGS = ("part_in_pieces", "part_changes_size", "part_off_the_subject", "part_moves_on_the_subject")
#: A flag of these two, when its own figure is over the list's number: (the figure in the flag, the number, the list's
#: words). The numbers are the list's ("on more than two frames", "over a half"), not this tool's.
FORCING_FIGURES = {"part_not_visible": ("frames", 2, "on more than two frames"),
                   "region_mostly_not_the_subject": ("median_share_not_subject", 0.5, "over a half")}
#: What of the list no flag says today, printed in every no-look record so nobody reads its silence as a pass.
FORCING_NOT_READ = ("ANOTHER SUBJECT'S HELD THINGS inside the region have no flag of their own (their clothing and hands do: `segment_inside_region`)",
                    "a head reference whose HEAD LEAVES ITS BOX is read only as the reference record's `passed`, which holds its three read-backs")
NO_LOOK = "no_look_owed.json"


def forcing_flags(cap: Path, load: dict, job: dict) -> list[str]:
    """Each forcing rule that stands on a capture's flags and a job's reference, in words that name it; none when
    none does. What the list names and no flag says is `FORCING_NOT_READ`."""
    import shot_to_load                 # the keep-outs' classes: what "clothing" and "hands" are, in their one place
    classes = set(shot_to_load.KEEP_OUTS["hands"]) | set(shot_to_load.KEEP_OUTS["clothes"])
    mine = {load.get("subject"), load.get("second_subject"), (load.get("also") or {}).get("subject")} - {None}    # a pair's second is `also`
    said = []
    for f in json.loads((cap / "flags.json").read_text())["flags"]:
        rule, figures = f.get("rule"), f.get("figures") or {}
        if rule in FORCING_FLAGS:
            said.append(f"{f.get('id')} {rule}")
        elif rule in FORCING_FIGURES:
            key, over, words = FORCING_FIGURES[rule]
            if figures.get(key) is None:
                said.append(f"{f['id']} {rule} (its figure `{key}` is not in the flag, so it is held as standing)")
            elif figures[key] > over:
                said.append(f"{f['id']} {rule} ({key} {figures[key]}: {words})")
        elif rule == "segment_inside_region" and "." in str(f.get("segment", "")):
            whose, cls = f["segment"].split(".", 1)
            if whose not in mine and cls in classes:
                said.append(f"{f['id']} segment_inside_region ({f['segment']}: another subject's clothing or hands inside the region)")
    for key in ("reference", "reference_2"):          # a pair's second subject's reference is `reference_2` (`shot_to_load.py pair`)
        ref = job.get(key) or {}
        if ref.get("record"):
            record = Path(ref["record"])
            if not record.is_file() or json.loads(record.read_text()).get("passed") is not True:
                said.append(f"the head reference's record does not say its read-backs passed ({record})")
    return said


def capture_on_the_plan(ctx: dict) -> tuple[dict, dict]:
    """The capture's status and flags, held to what any go rests on: the capture on the node's plan, its gate clear."""
    cap = ctx["capture"]
    for name in ("manifest.json", "status.json", "flags.json"):
        if not (cap / name).is_file():
            raise Refused(f"{cap.name} holds no {name}")
    manifest = json.loads((cap / "manifest.json").read_text())
    if not manifest["runs"] or not manifest["runs"][0].get("carried_from_the_node"):
        raise Refused(f"{cap.name} is not the capture on the node's plan: run the on-plan step first")
    flags = json.loads((cap / "flags.json").read_text())
    if flags.get("verdict") != "clear":
        raise Refused(f"the gate on {cap.name} is {flags.get('verdict')!r}, not clear")
    return json.loads((cap / "status.json").read_text()), flags


def step_no_look(ctx: dict, world, a) -> None:
    """The load's maker records that NO LOOK IS OWED: a record of its own kind in the capture, never a look."""
    said = " ".join(Path(a.not_first).read_text().split()) if a.not_first and Path(a.not_first).is_file() else ""
    if not said:
        raise Refused("--not-first is a file with words in it: the maker's statement that this load is not the first of its kind, and which earlier load of "
                      "the kind it follows (a first of a kind is owed a look whatever its flags)")
    cap = ctx["capture"]
    status, flags = capture_on_the_plan(ctx)
    standing = forcing_flags(cap, ctx["load"], ctx["job"])
    if standing:
        raise Refused(f"A FORCING FLAG STANDS on {cap.name}: {'; '.join(standing)}. A look by somebody who did not make the capture is owed "
                      "(`capture_masked_run.py looked`); no record that none is owed is written")
    entry = {"load": ctx["name"], "at": datetime.datetime.now().isoformat(timespec="seconds"), "by": a.by, "rests_on": FORCING_WORD, "gate": flags.get("verdict"),
             "capture_written": status.get("written"), "capture_made_by": status.get("made_by"),
             "flags_read": [f"{f.get('id')} {f.get('rule')} ({f.get('level')})" for f in flags["flags"]],
             "forcing_rules_held_to": {"any flag of": list(FORCING_FLAGS), "a flag over its figure": {k: v[2] for k, v in FORCING_FIGURES.items()},
                                       "another subject's": "clothing or hands inside the region (segment_inside_region)",
                                       "the head reference": "its record's `passed`"},
             "of_the_list_and_not_read": list(FORCING_NOT_READ), "not_the_first_of_its_kind": said}
    print(f"\n--- no look is owed on {ctx['name']}: the gate is clear, {len(flags['flags'])} flag(s) read and none is a forcing one; recorded for {a.by} in "
          f"{cap.name}/{NO_LOOK}\n    NOT READ BY THIS: " + "; ".join(FORCING_NOT_READ), flush=True)
    if world.dry:
        return
    kept = cap / NO_LOOK
    records = json.loads(kept.read_text())["records"] if kept.is_file() else []
    records.append(entry)
    kept.write_text(json.dumps({"what": "the maker's records that no look is owed on a load: NOT looks, and never counted as one; appended and never rewritten",
                                "records": records}, indent=1) + "\n")


def gate_alone(ctx: dict, why_no_look: str) -> dict:
    """What a render rests on when no look does: the maker's record that none is owed, held again to the capture as it is now."""
    cap = ctx["capture"]
    kept = cap / NO_LOOK
    mine = [r for r in json.loads(kept.read_text())["records"] if r.get("load") == ctx["name"]] if kept.is_file() else []
    if not mine:
        raise Refused(f"{why_no_look}. Or, where the gate is clear and no forcing flag stands, the load's maker records that no look is owed "
                      "(`load_chain.py no-look`): a render on the gate alone")
    status, flags = capture_on_the_plan(ctx)
    standing = forcing_flags(cap, ctx["load"], ctx["job"])
    if standing:
        raise Refused(f"A FORCING FLAG STANDS on {cap.name}: {'; '.join(standing)}. The record that no look is owed does not hold against it: a look by "
                      f"somebody who did not make the capture is owed ({why_no_look})")
    record = mine[-1]
    if not later(record["at"], status["written"]):
        raise Refused(f"the record that no look is owed ({in_utc(record['at'])}) is not later than the capture ({in_utc(status['written'])}): it is of an earlier capture")
    outcomes = cap / "outcomes.json"
    overridden = [o["flag"] for o in json.loads(outcomes.read_text())["outcomes"] if o.get("overridden")] if outcomes.is_file() else []
    return {"gate_alone": True, "by": f"NOBODY: rendered on the gate alone; {record['by']} recorded that no look is owed", "at_utc": in_utc(record["at"]),
            "capture_written_utc": in_utc(status["written"]), "capture_made_by": status.get("made_by"), "flags": len(flags["flags"]), "overridden": overridden,
            "rests_on": record["rests_on"], "not_the_first_of_its_kind": record["not_the_first_of_its_kind"]}


def the_look(ctx: dict) -> dict:
    """The look a render rests on: the last one recorded on the capture, which must be of this load, on the capture
    on the node's plan, later than the capture and not by the capture's maker. Returns what the go file says of it.

    Where no such look is there, the render may rest on the gate alone (`gate_alone`): the maker's record that no
    look is owed, good only while the gate is clear and no forcing flag stands. A record of that kind is never a
    look: a load with a forcing flag is refused with one, and the rule that a look is somebody else's is as it was."""
    try:
        return a_look(ctx)
    except Refused as no_look:
        return gate_alone(ctx, str(no_look))


def a_look(ctx: dict) -> dict:
    cap = ctx["capture"]
    for name in ("looks.json", "manifest.json", "status.json", "flags.json"):
        if not (cap / name).is_file():
            raise Refused(f"{cap.name} holds no {name}" + (": nobody has recorded a look of it (`capture_masked_run.py looked`)" if name == "looks.json" else ""))
    looks = json.loads((cap / "looks.json").read_text())["looks"]
    if not looks:
        raise Refused(f"{cap.name}/looks.json holds no look")
    look = looks[-1]
    if ctx["name"] not in look.get("loads", []):
        raise Refused(f"the last look on {cap.name} is of {look.get('loads')}, not of this load")
    manifest = json.loads((cap / "manifest.json").read_text())
    if not manifest["runs"] or not manifest["runs"][0].get("carried_from_the_node"):
        raise Refused(f"{cap.name} is not the capture on the node's plan: run the on-plan step first")
    status = json.loads((cap / "status.json").read_text())
    if not later(look["at"], status["written"]):
        raise Refused(f"the last look ({in_utc(look['at'])}) is not later than the capture ({in_utc(status['written'])}): it is a look of an earlier capture")
    if look["by"] == status.get("made_by"):
        raise Refused(f"the last look is by {look['by']}, who made the capture: a look is somebody else's")
    flags = json.loads((cap / "flags.json").read_text())
    if flags.get("verdict") != "clear":
        raise Refused(f"the gate on {cap.name} is {flags.get('verdict')!r}, not clear")
    outcomes = cap / "outcomes.json"
    overridden = [o["flag"] for o in json.loads(outcomes.read_text())["outcomes"] if o.get("overridden")] if outcomes.is_file() else []
    return {"by": look["by"], "at_utc": in_utc(look["at"]), "capture_written_utc": in_utc(status["written"]), "capture_made_by": status.get("made_by"),
            "flags": len(flags["flags"]), "overridden": overridden}


def job_as_planned(ctx: dict) -> None:
    """The job file must be the one the plan was built from: the builder keeps that copy as `job.plan.json`."""
    kept = ctx["job_folder"] / "job.plan.json"
    if not kept.is_file():
        raise Refused(f"{kept} is not there: no plan of this load was built")
    if kept.read_bytes() != ctx["raw"]:
        raise Refused(f"{ctx['job_file']} is not the file its plan was built from (it differs from {kept}): a job file is not edited after its "
                      "plan is built. Build the plan again from this file, or put the file back")


def graph_of(path: Path) -> dict:
    g = json.loads(path.read_text())
    return g.get("prompt", g)


def frame_hash(path: str) -> str:
    """One hash of a video's decoded frames, every frame: two files with the same pictures give the same hash."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return hashlib.md5("\n".join(line.split()[-1] for line in out.splitlines() if line and not line.startswith("#")).encode()).hexdigest()


def mask_videos(graph: dict, capture: Path) -> list[str]:
    """The videos under the capture's folder a graph loads: its masks."""
    root = str(capture) + os.sep
    return sorted({n["inputs"]["video"] for n in graph.values() if isinstance(n.get("inputs", {}).get("video"), str) and n["inputs"]["video"].startswith(root)})


def render_against_plan(plan: dict, render: dict, built: dict, hashed=frame_hash) -> list[str]:
    """Why a render graph is not its plan graph with the preview off; an empty list when it is. Also returned in it,
    marked `note:`, the hair-band line when the band rule was used."""
    if set(plan) != set(render):
        return [f"the two graphs do not have the same nodes: {sorted(set(plan) ^ set(render))}"]
    diffs = sorted((i, k) for i in plan for k in set(plan[i]["inputs"]) | set(render[i]["inputs"]) if plan[i]["inputs"].get(k) != render[i]["inputs"].get(k))
    band = [(i, k) for i, k in diffs if k == "video" and str(plan[i]["inputs"][k]).endswith(BAND_SUFFIXES[0]) and str(render[i]["inputs"][k]).endswith(BAND_SUFFIXES[1])]
    rest = sorted(k for i, k in diffs if (i, k) not in band)
    said = []
    if rest != RENDER_DIFFERS_IN:
        said.append(f"the render graph differs from the plan graph in {rest}, where it may differ in {RENDER_DIFFERS_IN} and no more")
    if built.get("uncommitted_node_code"):
        said.append(f"node code is not committed: {built['uncommitted_node_code']}")
    if len(band) > 1:
        said.append("more than one band file differs")
    for i, k in band:
        a, b = hashed(plan[i]["inputs"][k]), hashed(render[i]["inputs"][k])
        line = f"THE BAND FILE: the render graph reads {render[i]['inputs'][k]} where the plan graph read {plan[i]['inputs'][k]}; every frame's hash, plan {a}, render {b}"
        said.append(("note: " + line + ": the same.") if a == b else (line + ": NOT the same."))
    return said


#: The mark of an on-plan record hashed after its rebuild (route C, 2026-10-16); a record without it was hashed before the rebuild carved
#: the keep-outs, so a mask that no longer decodes equal there is a load planned before the keep-out carve.
HASHED_AFTER_THE_REBUILD = "after the on-plan rebuild"
PLANNED_BEFORE_THE_CARVE = "planned before the keep-out carve: make the load again under a new name"


def masks_as_planned(ctx: dict, render: dict, hashed=frame_hash) -> list[str]:
    """Why a mask video the render graph reads is not what the plan read; an empty list when every one decodes equal
    to the hash the on-plan step recorded before it built the capture again in place."""
    record = ctx["job_folder"] / f"{ctx['name']}.plan.masks.json"
    if not record.is_file():
        return [f"{record.name} is not there: the on-plan step, which records what the plan read, has not run"]
    held = json.loads(record.read_text())
    planned, after = held["masks"], held.get("hashed") == HASHED_AFTER_THE_REBUILD
    said = []
    for path in mask_videos(render, ctx["capture"]):
        if path not in planned:
            said.append(f"the render graph reads {path}, which the plan graph did not")
        elif hashed(path) != planned[path]:
            said.append(f"{path} no longer decodes equal to what the plan read" if after else
                        f"{path} no longer decodes equal to what the plan read, which was hashed before the on-plan rebuild carved its keep-outs: "
                        + PLANNED_BEFORE_THE_CARVE)
    return said


def region_as_predicted(ctx: dict, render_folder: Path) -> list[str]:
    """Why the render's own window regions (`<name>_windows/<name>_window_N_region.npz`, written by the node) are not the region the
    on-plan step predicted on the carved files (`<capture>/runs/<name>/predicted/`, route C), unit by unit; an empty list when every
    window's tokens are the same. A difference names the window and its source frames."""
    import numpy as np
    name, cap = ctx["name"], ctx["capture"]
    folder = cap / "runs" / name / "predicted"
    plans = sorted(folder.glob("*_plan.json"))
    if len(plans) != 1:
        return [f"{folder} holds {len(plans)} plan file(s): the on-plan step made no prediction of the region for this load. {PLANNED_BEFORE_THE_CARVE}"]
    plan = json.loads(plans[0].read_text())
    import capture_masked_run
    run_lengths = capture_masked_run._pack("video_mask").run_lengths
    said = []
    for row in plan.get("windows", []):
        if not row.get("region_file"):
            continue
        mine, theirs = folder / row["region_file"], render_folder / f"{name}_windows" / f"{name}_window_{row['number']}_region.npz"
        if not theirs.is_file():
            said.append(f"window {row['number']}: the render wrote no {theirs.name}")
            continue
        with np.load(mine) as a, np.load(theirs) as b:
            p, r = a["tokens"], b["tokens"]
        if p.shape != r.shape:
            said.append(f"window {row['number']}: the render's region is {list(r.shape)} and the prediction's {list(p.shape)}")
            continue
        steps = [k for k in range(p.shape[0]) if not (p[k] == r[k]).all()]
        if steps:
            lengths = run_lengths(p.shape[0])
            first = ctx["load"]["first"] + int(row["first_frame"])
            frames = sorted({first + sum(lengths[:k]) + i for k in steps for i in range(lengths[k])})
            units = int(sum(int((p[k] != r[k]).sum()) for k in steps))
            said.append(f"window {row['number']}: the render's region differs from the prediction by {units} unit(s) on {len(steps)} latent step(s), "
                        f"source frames {frames[0]}-{frames[-1]}")
    return said


def same_tree(a: Path, b: Path) -> list[str]:
    """Why two folders are not the same files with the same bytes; an empty list when they are."""
    names = lambda root: sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    na, nb = names(a), names(b)
    if na != nb:
        return [f"the two hold different files: {sorted(set(na) ^ set(nb))[:6]}"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return [f"{n} differs" for n in na if digest(a / n) != digest(b / n)]


def go_text(ctx: dict, by: str, now: str, grounds: str, show: str, look: dict, plan_prompt: str, card: str) -> str:
    """The go for a render, as the file beside the job file holds it: who gave it, on what, what was looked at and
    by whom, what the render is to show, and the card before it was queued. Times in UTC."""
    return (f"# The go for the render of {ctx['name']}\n\n"
            f"Written by {by} BEFORE the render was queued, {now} by date -u. The job file ({ctx['job_file'].name}) is as its plan was built from it.\n\n"
            f"**The go:** {' '.join(grounds.split())}\n\n"
            + (f"**NO LOOK: RENDERED ON THE GATE ALONE.** {look['by']}, {look['at_utc']}; not the first of its kind by the maker's statement: "
               f"{look['not_the_first_of_its_kind']} It rests on {look['rests_on']}. The capture was written {look['capture_written_utc']} by {look['capture_made_by']}.\n\n"
               if look.get("gate_alone") else
               f"**The look** on this capture on the node's plan: {look['by']}, {look['at_utc']}. The capture was written {look['capture_written_utc']} "
               f"by {look['capture_made_by']}.\n\n") +
            f"**The plan:** prompt {plan_prompt}, nothing sampled. **The capture on the plan:** gate clear, {look['flags']} flags, "
            + (f"overridden by a person: {', '.join(look['overridden'])}" if look["overridden"] else "no override") + ".\n\n"
            f"**What the render is to show:** {' '.join(show.split())}\n\n"
            f"**The card** before the render was queued, {now}: {card}\n")


def page_item(ctx: dict, spec: dict, clip: str, first: int, last: int, link: str, buttons: tuple) -> dict:
    """A render's item for the review list, from the words a person wrote (`spec`) and the load's own figures."""
    missing = [k for k in ("title_rest", "state_words", "what", "look", "prompt_note") if not str(spec.get(k) or "").strip()]
    if missing:
        raise Refused(f"the page's words lack {', '.join(missing)}: they are a person's to write (--spec)")
    name = ctx["name"]
    if spec["title_rest"].startswith(name):
        raise Refused("`title_rest` begins with the load's name, which the title already starts with")
    return {"id": name, "title": f"{name}: {spec['title_rest']}", "file": clip, "path": link, "first": str(first), "frames": str(last - first + 1), "fps": "24",
            "seconds": f"about {first / 24:.1f} s to {(last + 1) / 24:.1f} s of the clip", "state": "open", "state_words": spec["state_words"],
            "what": spec["what"], "think": "", "look": spec["look"], "options": list(buttons), "prompt": ctx["text"].read_text().strip(),
            "prompt_note": spec["prompt_note"], "fixed": "Nothing was fixed in this clip."}


# ---------------------------------------------------------------- the server and the machine

def queue_state(host: str) -> tuple[int, int]:
    with urllib.request.urlopen(f"http://{host}/queue", timeout=5) as r:
        q = json.loads(r.read())
    return len(q["queue_running"]), len(q["queue_pending"])


def card_line(host: str) -> str:
    with urllib.request.urlopen(f"http://{host}/system_stats", timeout=5) as r:
        v = json.loads(r.read())["devices"][0]
    stats = ", ".join(f"{k} {round(v[k] / 2 ** 20)}" for k in ("vram_total", "vram_free", "torch_vram_total", "torch_vram_free"))
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return f"from the server's /system_stats (MiB): {stats}; nvidia-smi (pid, used): {smi or 'nothing'}; queue empty; HEAD is {head}."


def free_card(host: str) -> str:
    """Ask the server to unload its models and free the card (`POST /free`, as `docs/comfy_notes.md` has it), and
    say what the card then holds. A pose pass runs in its own process beside the server; with the server's models
    resident after a render it has no room (2026-10-13: a land's pose pass died of it, and the step went on)."""
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(f"http://{host}/free", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    time.sleep(FREE_SETTLES_S)
    with urllib.request.urlopen(f"http://{host}/system_stats", timeout=5) as r:
        v = json.loads(r.read())["devices"][0]
    return f"the card after it, by the server's /system_stats (MiB): vram_free {round(v['vram_free'] / 2 ** 20)} of {round(v['vram_total'] / 2 ** 20)}"


def cost_lines(host: str, prompt_id: str) -> list[str]:
    with urllib.request.urlopen(f"http://{host}/history/{prompt_id}", timeout=10) as r:
        entries = list(json.loads(r.read()).values())
    if not entries:
        return [f"the server's history has no prompt {prompt_id}"]
    seen: list[str] = [f"{entries[0]['status']['status_str']} {entries[0]['status']['completed']}"]
    for o in entries[0].get("outputs", {}).values():
        for t in o.get("text", []):
            seen += [line for line in t.splitlines() if ("costs:" in line or "references in all" in line) and line not in seen]
    return seen


class World:
    """What a step asks of the machine: the queue, the card, the history, the folders, and running a command. A check
    hands in its own."""
    def __init__(self, host: str = HOST, dry: bool = False):
        import h3_config
        import masked_job
        self.host, self.dry = host, dry
        self.output, self.inputs, self.data = h3_config.output_dir(), masked_job.input_dir(), REPO / "data"
        self.pose_model = h3_config.POSE_MODEL
        self.python = sys.executable

    def queue(self) -> tuple[int, int]:
        return queue_state(self.host)

    def card(self) -> str:
        return card_line(self.host)

    def costs(self, prompt_id: str) -> list[str]:
        return cost_lines(self.host, prompt_id)

    def free(self) -> str:
        return free_card(self.host)

    def voice(self, source_name: str) -> dict:
        import clip_analysis
        import clip_folder
        try:
            return clip_analysis.voice_tables(source_name, self.data)
        except clip_folder.Refused as e:
            raise Refused(str(e)) from None

    def builder_refuses(self, ctx: dict, stage: str) -> str:
        """Why the builder would refuse this load's `stage`, in its own words, or "" when it would build: its real
        build run with `dry`, which holds everything a build holds and writes nothing."""
        import masked_job
        try:
            masked_job.build_loads(ctx["job_file"], stage, [ctx["name"]], date=ctx["day"], data=self.data, output=self.output, inputs=self.inputs, dry=True)
        except masked_job.Refused as e:
            return str(e)
        return ""

    def frames_of(self, path: Path) -> int:
        """A video's frame count, by ffprobe's count of its packets."""
        import clip_folder
        try:
            return clip_folder.probe(Path(path))["frames"]
        except clip_folder.Refused as e:
            raise Refused(str(e)) from None

    def run(self, ctx: dict, label: str, argv: list[str], stop: bool = True) -> int:
        """Run one command of a step: printed first, its output kept, its last lines shown.

        The output is kept under the job folder (`chain/`), unless the step names another place (`ctx["logs"]`); an
        output that is there from an earlier run is kept beside it as `<name>.<command>.before_<its time>.out`. A
        step whose command MAKES the job folder must name one: the preview's builder refuses a job folder that
        exists, and until 2026-10-13 keeping its output made the folder first, so the first real preview of a shot
        was refused by the tool that ran it."""
        line = " ".join(str(a) for a in argv)
        print(f"\n--- {label}\n    {line}", flush=True)
        if self.dry:
            return 0
        logs = Path(ctx["logs"]) if ctx.get("logs") else ctx["job_folder"] / "chain"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{ctx['name']}.{label.replace(' ', '_')}.out"
        if log.exists():
            # A command run again under one name (a preview made again after a wrong pick) keeps what the run before
            # it printed, under that file's own time. Until 2026-10-13 the new output replaced it, and the output of
            # the run that had gone wrong was the one lost.
            stamp = datetime.datetime.fromtimestamp(log.stat().st_mtime, datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            was, n = log.with_name(f"{log.stem}.before_{stamp}.out"), 1
            while was.exists():
                n += 1
                was = log.with_name(f"{log.stem}.before_{stamp}_{n}.out")
            log.rename(was)
            print(f"    an earlier output of this command is kept as {was.name}", flush=True)
        with open(log, "w") as f:
            code = subprocess.run([str(a) for a in argv], stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode
        tail = log.read_text().splitlines()[-4:]
        kept = log.relative_to(REPO) if log.is_relative_to(REPO) else log
        # a long line is cut on the screen and says so, with where it is whole
        shown = [t if len(t) <= TAIL_CHARS else t[:TAIL_CHARS] + f" [CUT HERE: {len(t)} characters, whole in the kept file]" for t in tail]
        print(f"    exit {code} at {utc_now()}; kept in {kept}\n" + "\n".join("    | " + t for t in shown), flush=True)
        if code and stop:
            raise Refused(f"{label} did not exit 0 (exit {code}); the step stops here")
        return code


def voice_for(ctx: dict, world) -> dict:
    """The video's voice tables, held to the load: refused when the per-frame table does not cover every frame of
    the load (the tables of a video are made for a stretch of it, and a load past that stretch has none)."""
    voice = world.voice(Path(ctx["job"]["source"]).name)
    first, last = ctx["load"]["first"], ctx["load"]["first"] + ctx["load"]["frames"] - 1
    a, b = voice["frames"]
    if not (a <= first and last <= b):
        raise Refused(f"the video's voice tables cover source frames {a}-{b} and the load is {first}-{last}: make the tables for these frames and name "
                      "them (`bench/clip_analysis.py voice`); a capture reads the voice on every frame of its load")
    return voice


def free_for_a_pose_pass(world) -> None:
    """Free the card before a pose pass and say so: the pass loads its own model in its own process, and the
    server keeps a render's models resident. What it costs is said with it."""
    print("\n--- the card is freed before the pose pass (POST /free: unload_models, free_memory)", flush=True)
    if world.dry:
        return
    print(f"    {world.free()}\n    THE SERVER'S MODELS ARE UNLOADED: the next thing queued starts cold, and its wall time is the load of the models "
          "as well as its own work", flush=True)


def empty_queue(world, why: str) -> None:
    running, pending = world.queue()
    if (running, pending) != (0, 0):
        raise Refused(f"the queue is not empty ({running} running, {pending} pending): {why}")


# ---------------------------------------------------------------- the steps

def tool(world, name: str) -> list[str]:
    return [world.python, f"bench/{name}"]


def capture_args(ctx: dict, world, files: dict, band: Path | None = None) -> list[str]:
    job = ctx["job"]
    out = ["--folder", ctx["capture"], "--source", job["source"], "--first", job["first"], "--frames", job["frames"]]
    if (job.get("head_tube") or {}).get("on") or (ctx.get("subject_keys") or {}).get("head_tube"):
        out.append("--head-tube")
    depth_cfg = job.get("depth") or {}
    if depth_cfg.get("on") and depth_cfg.get("overlap_frames") != []:
        out += ["--depth", ctx["capture"] / "depth.npz"]
    for spec in mask_specs(ctx, files, band):
        out += ["--mask", spec]
    return out


def ended_with(capture: Path, verify: int, gate: int, strip: int) -> tuple[int, str]:
    """How a capture step ended: the exit status it leaves and the words of its last line; 0 and no words when
    verify passed, the gate is clear and the strip was drawn.

    Each of the three commands runs whatever the one before it said, since a person needs the gate's flags and the
    strip to see WHY a capture is not good. Until 2026-10-13 the step then exited 0 with nothing said after the
    strip's lines: on the first real load its verify had exited 4 and its gate 3, on a load nobody should have
    planned. (The session scripts this tool replaced did the same: they printed `verify exit N` and `gate exit N`
    in passing, drew the strip and exited 0. None of their kept outputs of that day holds a gate that was not clear.)"""
    said = []
    if verify:
        said.append(f"VERIFY FAILED (exit {verify}): the capture does not read what the nodes wrote" if verify == VERIFY_FAILED
                    else f"VERIFY DID NOT RUN TO ITS END (exit {verify})")
    if gate == GATE_BLOCKED:
        flags = capture / "flags.json"
        on = json.loads(flags.read_text()).get("blocking") if flags.is_file() else None
        said.append("THE GATE IS BLOCKED" + (f" on {', '.join(on)}" if on else ""))
    elif gate:
        said.append(f"THE GATE DID NOT RUN TO ITS END (exit {gate})")
    if strip:
        said.append(f"THE STRIP WAS NOT DRAWN (exit {strip})")
    return next((code for code in (verify, gate, strip) if code), 0), "; ".join(said)


def gate_clear(ctx: dict, what: str) -> None:
    """Refuse a capture whose gate is not clear. A blocked flag is cleared by making the load again so that it no
    longer stands, or by a person ruling on it by name, which the capture tool records and the gate then reads."""
    cap = ctx["capture"]
    if not (cap / "flags.json").is_file():
        raise Refused(f"{cap.name} holds no flags.json: its gate did not run to its end, so nothing says it is clear")
    flags = json.loads((cap / "flags.json").read_text())
    if flags.get("verdict") != "clear":
        raise Refused(f"the gate on {cap.name} is {flags.get('verdict')!r}, on {', '.join(flags.get('blocking') or []) or 'nothing it names'}: {what}. "
                      f"Make the load again so the flag no longer stands, or a person rules on each flag by name: bench/capture_masked_run.py outcome "
                      f"{cap} <flag> overridden --by WHO --note WHY (the gate then reads clear and lists who overrode what)")


def told_to_the_gate(ctx: dict) -> list[str]:
    """What the gate is told of the load's job (`capture_masked_run.py preflight`; the lead's rulings of 2026-10-14):
    - `--own-keep-out NAME=CLASSES`: each keep-out of the job made of the load's subject's own things (the job's `keep_outs`
      `of` its subject; strangers are other people), with the classes it is cut from, which a restore span would name;
    - `--given-back NAME:FIRST-LAST=WHY`: each span of the job's `restore` (the plan's, written by the job step) whose
      classes hold all of an own keep-out's, on the load's frames;
    - `--named-empty SUBJECT:FIRST-LAST=WHY`: the frames the job empties on purpose: its subject's `drop` with `drop_by`,
      and its `left_out` with each reason;
    - `--chin-points POINTS.npz`: the job's reference points, whose chin the held keep-out's split reads her clothing below."""
    job, out = ctx["job"], []
    kept = job.get("keep_outs") or {}
    own = {}
    of = kept.get("of")
    if of == ctx["subject"] or (isinstance(of, list) and ctx["subject"] in of):    # a pair's keep-outs are of both its subjects
        own = {k["name"]: list(k.get("classes") or []) for k in kept.get("made") or [] if k["name"] != "strangers"}
    # a pair's keep-out says whose own things it is, and a pair's restore span whose it is (`shot_to_load.py pair`): one gives back the other
    # only when they are the same subject's
    whose = {k["name"]: k.get("of") for k in kept.get("made") or []}
    out += [x for n, c in own.items() for x in ("--own-keep-out", f"{n}={'+'.join(c)}")]
    lo, hi = ctx["load"]["first"], ctx["load"]["first"] + ctx["load"]["frames"] - 1
    for r in job.get("restore") or []:
        for n, c in own.items():
            if whose.get(n) and r.get("subject") and whose[n] != r["subject"]:
                continue
            if c and set(c) <= set(r["classes"]) and r["first"] <= hi and r["last"] >= lo:
                out += ["--given-back", f"{n}:{max(r['first'], lo)}-{min(r['last'], hi)}=the plan's restore of {'+'.join(r['classes'])} "
                                        f"on {r['first']}-{r['last']} gives it back at assembly ({' '.join(str(r.get('by') or 'who said so is not in the job').split())})"]
    for who, keys in ((ctx["subject"], ctx["subject_keys"]), (ctx.get("also"), ctx.get("also_keys") or {})):    # each subject of a pair
        if keys.get("drop") and str(keys.get("drop_by") or "").strip():
            for span in str(keys["drop"]).split("+"):
                a, b = (span.split("-") * 2)[:2]
                out += ["--named-empty", f"{who}:{a}-{b}=the job's drop= ({' '.join(str(keys['drop_by']).split())})"]
    for x in job.get("left_out") or []:
        if x.get("why"):
            out += ["--named-empty", f"{x['subject']}:{x['first']}-{x['first'] + x['frames'] - 1}=the job's left_out ({' '.join(str(x['why']).split())})"]
    points = (job.get("reference") or {}).get("points")
    if points and Path(points).is_file():
        out += ["--chin-points", str(points)]
    return out


def gate_and_strip(ctx: dict, world, by: str, who_is_who: Path, voice: dict, strip_folder: str = "") -> tuple[int, str]:
    cap, job = ctx["capture"], ctx["job"]
    verify = world.run(ctx, f"verify{strip_folder and ' first'}", tool(world, "capture_masked_run.py") + ["verify", cap, "--by", by], stop=False)
    gate = world.run(ctx, f"gate{strip_folder and ' first'}", tool(world, "capture_masked_run.py") + ["preflight", cap, "--text", ctx["text"], "--voice-spans", voice["spans"],
                                                                                                     "--who-is-who", who_is_who, "--gate", *told_to_the_gate(ctx)], stop=False)
    # the references and stills the strip shows are read from the load's own graph (2026-10-15: they were the job file's, and a load of
    # two subjects was shown one of each)
    strip = tool(world, "capture_masked_run.py") + ["video", cap, "--source", job["source"], "--job", ctx["job_file"], "--load", ctx["name"],
                                                    "--given", ctx["job_folder"], "--one-a-second"]
    if strip_folder:
        strip += ["--strip-folder", strip_folder]
    return ended_with(cap, verify, gate, world.run(ctx, f"strip{strip_folder and ' first'}", strip, stop=False))


def who_is_who_file(ctx: dict, world, files: dict) -> Path:
    """The who-is-who file of the load: the file the job names, or made again now from the preview's shot tables.
    Never one picked up because it is there."""
    named = ctx["job"].get("who_is_who")
    if named:
        path = Path(named) if Path(named).is_absolute() else ctx["job_file"].parent / named
        if not path.is_file():
            raise Refused(f"the job names the who-is-who file {named!r}, which is not a file")
        return path
    out = ctx["job_folder"] / f"{ctx['name']}.who_is_who.json"
    argv = tool(world, "who_is_who_across_shots.py") + ["corrections"]
    for label in ctx["preview"]["order"]:
        argv += ["--table", f"{label}={files[label]['shots']}@{ctx['preview']['first']}"]
    argv += ["--first-frame", ctx["job"]["first"], "--frames", ctx["job"]["frames"], "--out", out]
    if not world.dry:
        ctx["job_folder"].mkdir(parents=True, exist_ok=True)
    world.run(ctx, "who is who", argv)
    return out


def step_first(ctx: dict, world, a) -> tuple[int, str]:
    cap = ctx["capture"]
    depth_only = (
        bool((ctx["job"].get("depth") or {}).get("on"))
        and cap.is_dir()
        and (cap / "depth.npz").is_file()
        and {p.name for p in cap.iterdir()} <= {"depth.npz", "record.json", "sheet.jpg"}
    )
    for path in (() if depth_only else (cap,)) + (cap.with_name(cap.name + "_first"),):
        if path.exists():
            raise Refused(f"{path} exists: the first step makes the first capture, once")
    files, voice = preview_files(ctx, world.output), voice_for(ctx, world)
    who = who_is_who_file(ctx, world, files)
    world.run(ctx, "files first", tool(world, "capture_masked_run.py") + ["files"] + capture_args(ctx, world, files)
              + ["--loads", ctx["job_file"], "--voice", voice["per_frame"], "--made-by", a.by])
    return gate_and_strip(ctx, world, a.by, who, voice, strip_folder=cap.name + "_first")


def step_plan(ctx: dict, world, a) -> None:
    if not (ctx["capture"] / "manifest.json").is_file():
        raise Refused(f"{ctx['capture']} is not there: the plan is built from the first capture")
    if (ctx["job_folder"] / f"{ctx['name']}.plan.json").exists():
        raise Refused(f"a plan graph of {ctx['name']} is already built in {ctx['job_folder'].name}: never a second one under a name")
    gate_clear(ctx, "a plan is not built on a first capture whose gate is blocked")
    empty_queue(world, "a plan is queued on an empty queue")
    # The builder's own refusals, before anything is run and in a dry run as well. On the first real load (2026-10-13)
    # `plan --dry` exited 0 and the plan itself was then refused by its builder, on a key the job file had held all along.
    why = world.builder_refuses(ctx, "plan")
    if why:
        raise Refused(f"the builder refuses this load's plan, and nothing was built or queued: {why}")
    world.run(ctx, "build plan", tool(world, "masked_job.py") + ["build", "plan", ctx["job_file"], "--date", ctx["day"]])
    world.run(ctx, "queue plan", tool(world, "masked_job.py") + ["queue", ctx["job_folder"], "plan", "--load", ctx["name"], "--host", world.host, "--wait"])


def step_on_plan(ctx: dict, world, a, hashed=frame_hash) -> tuple[int, str]:
    cap, first = ctx["capture"], ctx["capture"].with_name(ctx["capture"].name + "_first")
    windows = plan_windows(ctx, world.output)
    if not (windows / f"{ctx['name']}_plan.json").is_file():
        raise Refused(f"no plan file in {windows}: the plan step has not run to its end")
    if not (cap / "manifest.json").is_file():
        raise Refused(f"{cap} is not there: there is no first capture to build again on the plan")
    if first.exists():
        raise Refused(f"{first} exists: the first capture was already copied, so this step has run")
    plan_graph = ctx["job_folder"] / f"{ctx['name']}.plan.json"
    if not plan_graph.is_file():
        raise Refused(f"{plan_graph} is not there")
    empty_queue(world, "the capture is built again in place, and a queued graph may be reading its masks")
    files, voice = preview_files(ctx, world.output), voice_for(ctx, world)
    band = plan_band_file(ctx)
    who = who_is_who_file(ctx, world, files) if ctx["job"].get("who_is_who") else ctx["job_folder"] / f"{ctx['name']}.who_is_who.json"
    if not who.is_file() and not world.dry:
        raise Refused(f"{who} is not there: the first step, which makes it, has not run for this job folder")
    if not world.dry:
        shutil.copytree(cap, first, symlinks=True)
        differs = same_tree(cap, first)
        if differs:
            raise Refused(f"the copy {first.name} is not the capture it was copied from ({'; '.join(differs[:4])}): nothing was overwritten")
        print(f"--- copied {cap.name} to {first.name}; the copy holds the same files with the same bytes", flush=True)
    run = f"{ctx['name']}:preview={windows},subject={ctx['subject']}" + (f",others={'+'.join(others_of(ctx))}" if others_of(ctx) else "") + f",at={ctx['load']['first']}"
    world.run(ctx, "files on plan", tool(world, "capture_masked_run.py") + ["files"] + capture_args(ctx, world, files, band)
              + ["--run", run] + (["--loads", ctx["job_file"]] if ctx.get("also") else [])     # a pair's second subject is read off its job file
              + ["--voice", voice["per_frame"], "--made-by", a.by, "--overwrite"])
    # WHAT THE RENDER WILL READ, hashed AFTER the rebuild (route C, the lead's ruling of 2026-10-16): the rebuild carves the keep-outs on
    # the node's own grid and remakes the plan's region on those files (`capture_masked_run.predict_regions`), so the look, this hash and
    # the render's masks are one set of files
    masks = {path: hashed(path) for path in mask_videos(graph_of(plan_graph), cap)}
    print(f"\n--- what the render will read: {len(masks)} mask video(s) under {cap.name}, each frame hashed after the rebuild", flush=True)
    if not world.dry:
        (ctx["job_folder"] / f"{ctx['name']}.plan.masks.json").write_text(json.dumps(
            {"what": "the frame-by-frame hash of every mask video the render graph reads, taken AFTER the capture was built again on its plan "
                     "(the keep-outs carved on the node's own grid, the plan's region remade on these files for the look); the render step refuses a "
                     "mask that no longer decodes equal", "hashed": HASHED_AFTER_THE_REBUILD, "at": utc_now(), "by": a.by, "masks": masks}, indent=1) + "\n")
    ended = gate_and_strip(ctx, world, a.by, who, voice)
    if band is not None:
        band_again(ctx, world, band, plan_graph, hashed)
    run_frame_table(ctx, world, stop=False)
    return ended


def band_again(ctx: dict, world, band: Path, plan_graph: Path, hashed=frame_hash) -> None:
    """THE PLAN'S BAND FILE MADE AGAIN FROM THE CARVED TRACK (Gemini's D13 A1, the lead's ruling of 2026-10-16). The rebuild hands the
    capture the plan's cut keep-out and carves it on the node's grid; the render cuts its band from that carved track, so the plan's file,
    cut before the carve, would not decode equal to the render's and the render step refused (`render_against_plan`). After verify and
    the gate, the builder writes the plan's band file again from the capture as the render reads it (`masked_job.py band`), and the record
    of what the render will read holds its hash. The plan's own bytes are kept first under `<job folder>_first/`, where the reader of the
    plan finds them (`what_h3_got.first_copy`)."""
    kept = ctx["job_folder"].with_name(ctx["job_folder"].name + "_first") / band.name
    if kept.exists():
        raise Refused(f"{kept} exists: the plan's band file was already kept, so this step has run")
    if not world.dry:
        kept.parent.mkdir(exist_ok=True)
        shutil.copy2(band, kept)
    world.run(ctx, "band again", tool(world, "masked_job.py") + ["band", ctx["job_file"], "--load", ctx["name"], "--to", band])
    if world.dry:
        return
    named = next((v for n in graph_of(plan_graph).values() for v in [n.get("inputs", {}).get("video")]
                  if isinstance(v, str) and v.endswith(BAND_SUFFIXES[0])), str(band))
    record = ctx["job_folder"] / f"{ctx['name']}.plan.masks.json"
    held = json.loads(record.read_text())
    held["masks"][named] = hashed(named)
    held["band_made_again"] = {"file": named, "kept": str(kept), "at": utc_now(),
                               "why": "made again from the capture's carved track as the render reads it (masked_job.py band), so the render's band file decodes equal"}
    record.write_text(json.dumps(held, indent=1) + "\n")
    print(f"--- the plan's band file {band.name} made again from the carved track; the plan's own bytes kept at {kept}", flush=True)


def run_frame_table(ctx: dict, world, *, stop: bool) -> None:
    """Run `bench/frame_table.py` on the capture (and for the second subject on a two-subject load), recording its
    path and sha256 in `<load>.plan.built.json` only when the command succeeds."""
    cap = ctx["capture"]
    table_code = world.run(ctx, "frame table", tool(world, "frame_table.py") + [cap, "--job", ctx["job_file"]], stop=stop)
    also_code = 0
    if ctx.get("also"):
        also_code = world.run(
            ctx,
            f"frame table of {ctx['also']}",
            tool(world, "frame_table.py") + [cap, "--job", ctx["job_file"], "--subject", ctx["also"]],
            stop=stop,
        )
    if table_code or also_code:
        print(f"--- WARNING: frame_table.py exited non-zero (primary={table_code}, also={also_code}); "
              f"fix the cause and run `bench/load_chain.py table {ctx['job_file']} --by <session>` before `render`", flush=True)
        return
    plan_built = ctx["job_folder"] / f"{ctx['name']}.plan.built.json"
    if not world.dry and plan_built.is_file():
        built = json.loads(plan_built.read_text())
        ft_primary = cap / "frame_table.jsonl"
        built["frame_table"] = str(ft_primary)
        if ft_primary.is_file():
            built["frame_table_sha256"] = hashlib.sha256(ft_primary.read_bytes()).hexdigest()
        if ctx.get("also"):
            ft_also = cap / f"frame_table__{ctx['also']}.jsonl"
            built["frame_table_also"] = str(ft_also)
            if ft_also.is_file():
                built["frame_table_also_sha256"] = hashlib.sha256(ft_also.read_bytes()).hexdigest()
        plan_built.write_text(json.dumps(built, indent=1) + "\n")


def step_band_again(ctx: dict, world, a, hashed=frame_hash) -> None:
    """The plan's band file made again from the carved track, for a banded load whose on-plan ran before `band_again` was part of it."""
    name, folder = ctx["name"], ctx["job_folder"]
    band = plan_band_file(ctx)
    if band is None:
        raise Refused(f"{name} has no hair band: there is no band file to make again")
    go = ctx["job_file"].parent / f"{name}.render.go.md"
    if (folder / f"{name}.render.json").exists() or go.exists():
        raise Refused(f"a render of {name} is already built or already has a go ({go.name}): the band file is not made again after a render begins")
    record = folder / f"{name}.plan.masks.json"
    held = json.loads(record.read_text()) if record.is_file() else {}
    if held.get("hashed") != HASHED_AFTER_THE_REBUILD:
        raise Refused(f"{record.name} {'is not there' if not held else 'was not hashed after the on-plan rebuild'}: the band is made again only on a "
                      "capture rebuilt on its plan (`on-plan`)")
    if held.get("band_made_again"):
        raise Refused(f"the band file of {name} was already made again ({held['band_made_again'].get('at')}): {record.name}")
    band_again(ctx, world, band, folder / f"{name}.plan.json", hashed)


def step_table(ctx: dict, world, a) -> None:
    cap, name, folder = ctx["capture"], ctx["name"], ctx["job_folder"]
    if not (cap / "manifest.json").is_file():
        raise Refused(f"{cap} is not there: there is no capture to build a frame table on")
    go = ctx["job_file"].parent / f"{name}.render.go.md"
    if (folder / f"{name}.render.json").exists() or go.exists():
        raise Refused(f"a render of {name} is already built or already has a go ({go.name}): the frame table is not rebuilt after a render begins")
    job_as_planned(ctx)
    run_frame_table(ctx, world, stop=True)


def check_frame_table_fresh(ctx: dict, by: str) -> dict[str, str]:
    """Require `frame_table.jsonl` (and `frame_table__<also>.jsonl` for a pair whose second subject is in the capture)
    to exist and have fresh `input_sha256` against the files on disk before `render` proceeds."""
    import frame_table
    cap = ctx["capture"]
    load_for_verify = dict(ctx["load"])
    if ctx.get("also") and "also" not in load_for_verify:
        load_for_verify["also"] = {"subject": ctx["also"]}
    try:
        shas = frame_table.verify_table_fresh(cap, load_for_verify, job_file=ctx["job_file"], by=by)
    except frame_table.Refused as e:
        raise Refused(str(e)) from e
    plan_built = ctx["job_folder"] / f"{ctx['name']}.plan.built.json"
    if plan_built.is_file():
        built = json.loads(plan_built.read_text())
        if built.get("frame_table_sha256") and shas.get(ctx["subject"]) != built["frame_table_sha256"]:
            raise Refused(
                f"{cap.name}/frame_table.jsonl does not match frame_table_sha256 in {plan_built.name}: "
                f"run `bench/load_chain.py table {ctx['job_file']} --by {by}`"
            )
        if ctx.get("also") and built.get("frame_table_also_sha256") and shas.get(ctx["also"]) != built["frame_table_also_sha256"]:
            raise Refused(
                f"{cap.name}/frame_table__{ctx['also']}.jsonl does not match frame_table_also_sha256 in {plan_built.name}: "
                f"run `bench/load_chain.py table {ctx['job_file']} --by {by}`"
            )
    return shas


def step_render(ctx: dict, world, a, hashed=frame_hash) -> None:
    grounds, show = (Path(p).read_text().strip() if p and Path(p).is_file() else "" for p in (a.grounds, a.show))
    if not grounds or not show:
        raise Refused("--grounds and --show are each a file with words in it: who relayed the go and on what it rests, and what the render is to show")
    name, folder = ctx["name"], ctx["job_folder"]
    go = ctx["job_file"].parent / f"{name}.render.go.md"
    if (folder / f"{name}.render.json").exists() or go.exists():
        raise Refused(f"a render of {name} is already built or already has a go ({go.name}): never a second one under a name")
    empty_queue(world, "a render is queued on an empty queue")
    job_as_planned(ctx)
    look = the_look(ctx)
    table_shas = check_frame_table_fresh(ctx, a.by)
    planned = folder / f"{name}.plan.queued.json"
    if not planned.is_file():
        raise Refused(f"{planned} is not there: the plan was not queued through the job tool")
    now = utc_now()
    text = go_text(ctx, a.by, now, grounds, show, look, json.loads(planned.read_text()).get("prompt_id", "not recorded"), world.card())
    print(f"\n--- the go, {now}: " + ("NO LOOK, RENDERED ON THE GATE ALONE: " + look["by"] if look.get("gate_alone") else f"the look is {look['by']}'s") +
          f" of {look['at_utc']}, on a capture written {look['capture_written_utc']} by {look['capture_made_by']}", flush=True)
    if not world.dry:
        go.write_text(text)
    world.run(ctx, "build render", tool(world, "masked_job.py") + ["build", "render", ctx["job_file"], "--date", ctx["day"]])
    if world.dry:
        print("--- (dry: the render graph is not built, so it is not held against the plan graph)")
        return
    shutil.copyfile(folder / "chain" / f"{name}.build_render.out", folder / f"{name}.render.build_printed.txt")
    render = graph_of(folder / f"{name}.render.json")
    table_path = str(ctx["capture"] / "frame_table.jsonl")
    render_built_file = folder / f"{name}.render.built.json"
    render_built = json.loads(render_built_file.read_text())
    said = render_against_plan(graph_of(folder / f"{name}.plan.json"), render, render_built, hashed)
    notes, problems = [s[len("note: "):] for s in said if s.startswith("note: ")], [s for s in said if not s.startswith("note: ")]
    problems += masks_as_planned(ctx, render, hashed)
    record = {"what": "what the render step of bench/load_chain.py held before it queued", "at": now, "by": a.by, "go_file": str(go), "look": look,
              "frame_table": table_path, "frame_table_sha256": table_shas.get(ctx["subject"]),
              **({"frame_table_also": str(ctx["capture"] / f"frame_table__{ctx['also']}.jsonl"), "frame_table_also_sha256": table_shas.get(ctx["also"])} if ctx.get("also") else {}),
              "job_file_is_the_plans": True, "render_against_plan": problems or "differs in the song node's preview and filename_prefix only", "band": notes}
    (folder / f"{name}.render.chain.json").write_text(json.dumps(record, indent=1) + "\n")
    if notes:
        with open(go, "a") as f:
            f.write("\n" + "\n".join(notes) + "\n")
    if problems:
        raise Refused("the render graph is not the plan graph with the preview off: " + "; ".join(problems))
    world.run(ctx, "queue render", tool(world, "masked_job.py") + ["queue", folder, "render", "--load", name, "--host", world.host, "--wait"])


def rendered_video(ctx: dict, world, r: Path) -> Path:
    """The file the render's pose pass reads, whose frame 0 is the load's first frame and which holds every frame
    of the load: the one window's own video for a load of one window, as ever; the joined file for a load of
    several. Each window's own file holds only the frames that window wrote (a render of 447 frames on the share,
    2026-10-10: 345 in the first window's file, 102 in the second's, 447 in the joined file), so no one window's
    file is the load; the joined file is the load's frames once each, and it is held to the load's frame count
    before anything reads it."""
    import re
    name, load = ctx["name"], ctx["load"]
    folder = r / f"{name}_windows"
    numbers = sorted(int(m.group(1)) for p in folder.glob(f"{name}_window_*.mp4") if (m := re.search(WINDOW_VIDEO, p.name[len(name):])))
    if not numbers:
        raise Refused(f"no window file {folder / (name + '_window_1.mp4')}")
    if numbers != list(range(1, len(numbers) + 1)):
        raise Refused(f"the windows folder holds windows {numbers}: one is missing, so the render did not finish")
    if len(numbers) == 1:
        return folder / f"{name}_window_1.mp4"
    joined = r / f"{name}_00001.mp4"
    if not joined.is_file():
        raise Refused(f"the load has {len(numbers)} windows and the joined file {joined} is not there")
    if not world.dry:
        held = world.frames_of(joined)
        if held != load["frames"]:
            raise Refused(f"the load is {load['frames']} frames in {len(numbers)} windows and the joined file {joined.name} holds {held}: its frames "
                          "are not the load's, and a pose pass on it would be read against the wrong source frames")
    print(f"\n--- {len(numbers)} windows: the pose pass reads the joined file {joined.name}, the load's {load['frames']} frames once each", flush=True)
    return joined


def step_land(ctx: dict, world, a) -> tuple[int, str] | None:
    name, cap, load, job = ctx["name"], ctx["capture"], ctx["load"], ctx["job"]
    store = a.model_store or os.environ.get(MODEL_STORE)
    if not store:
        raise Refused(f"where the pose model's store is on this machine is not said: set {MODEL_STORE} (the folder holding one named {world.pose_model}) or give --model-store")
    model = Path(store) / world.pose_model
    if not model.is_dir() and not world.dry:
        raise Refused(f"{model} is not a folder")
    empty_queue(world, "no pose pass beside a render")
    r = render_folder(ctx, world.output)
    window = rendered_video(ctx, world, r)
    first, last = load["first"], load["first"] + load["frames"] - 1
    clip = world.output / REVIEW_CLIPS.format(video=video_of(ctx, world.data)) / f"{name}_{first:04d}_{last:04d}_stacked.mp4"
    # each subject's points: the load's own beside the capture as ever, and a pair's second subject's under its label
    subjects = [ctx["subject"]] + ([ctx["also"]] if ctx.get("also") else [])
    points = {who: cap / (f"{name}_render_points.npz" if who == ctx["subject"] else f"{name}_{who}_render_points.npz") for who in subjects}
    # LANDED is the render's points written, every subject's. A stacked clip with points missing is a land whose pose pass did not
    # finish: the step is run again and does the missing passes alone (until 2026-10-13 it was refused as landed, and the
    # pass had to be run by hand).
    if all(path.exists() for path in points.values()):
        raise Refused(f"the render's points exist ({', '.join(path.name for path in points.values())}): this load is landed")
    pose_alone = clip.exists()
    queued = ctx["job_folder"] / f"{name}.render.queued.json"
    if not queued.is_file():
        raise Refused(f"{queued} is not there: the render was not queued through the job tool")
    print("\n--- the render, by the server's history:\n" + "\n".join("    " + line for line in world.costs(json.loads(queued.read_text())["prompt_id"])), flush=True)
    # THE REGION AS PREDICTED (route C, the lead's ruling of 2026-10-16): the render's own window regions are the region the on-plan step
    # made on the carved files, unit by unit; a difference is a hard fault and the load is not landed
    differs = region_as_predicted(ctx, r)
    if differs:
        raise Refused("the render's region is not the region its look was made on: " + "; ".join(differs))
    manifest = json.loads((cap / "manifest.json").read_text())
    free_for_a_pose_pass(world)
    # the stack is made whether or not the pose pass finishes (a person needs the clip to see the render), and the step
    # then ENDS ON THE PASS: until 2026-10-13 a pass that died left the step's status 0 and its last line plain
    failed = None
    for who in subjects:
        if points[who].exists():
            continue
        seen = next(s for s in manifest["subjects"] if s["label"] == who)["sightings"][0]
        box = cap / "subjects" / who / f"track__{seen['by']}.mkv"      # the capture's own copy, whose frame 0 is the capture's first frame
        pose = world.run(ctx, "pose pass" + (f" of {who}" if len(subjects) > 1 else ""),
                         tool(world, "pose_along_shot.py") + ["run", "--video", window, "--first", 0, "--count", load["frames"],
                                                              "--canvas", f"{job['canvas'][0]}x{job['canvas'][1]}", "--source-first", first,
                                                              "--box-mask", box, "--box-mask-first", manifest["first_frame"],
                                                              "--model-folder", model, "--out", points[who], "--by", a.by], stop=False)
        if pose and failed is None:
            of = f" OF {who}" if len(subjects) > 1 else ""
            failed = (POSE_PASS_FAILED, f"THE POSE PASS{of} DID NOT FINISH (exit {pose}): THE RENDER'S POINTS ARE NOT WRITTEN AND THIS LOAD IS NOT LANDED. "
                      f"Its output is kept in the job folder's chain/. Run `land` again: with the stacked clip there it runs the pose pass alone")
    if pose_alone:
        print(f"\n--- the stacked clip is there from a land whose pose pass did not finish ({clip.name}): only the pose pass was run", flush=True)
        return failed
    # the render's frame 0 is the load's first frame, and it is SAID: the capture was made on the plan, before there was a
    # render, so no run of it names the file, and the stack then took the capture's first frame and printed NOT SAID
    stack = tool(world, "capture_masked_run.py") + ["video", cap, "--source", job["source"], "--render", r / f"{name}_00001.mp4", "--render-first", first,
                                                    "--given", ctx["job_folder"], "--given-load", name, "--one-a-second"]
    world.run(ctx, "stack", stack + ["--out", clip])
    return failed


def step_page(ctx: dict, world, a) -> None:
    import delivery_table
    if not a.spec or not Path(a.spec).is_file():
        raise Refused("--spec is a file with the page's words: title_rest, state_words, what, look, prompt_note")
    spec = json.loads(Path(a.spec).read_text())
    name, load = ctx["name"], ctx["load"]
    video = video_of(ctx, world.data)
    first, last = load["first"], load["first"] + load["frames"] - 1
    clip = world.output / REVIEW_CLIPS.format(video=video) / f"{name}_{first:04d}_{last:04d}_stacked.mp4"
    if not clip.is_file():
        raise Refused(f"{clip} is not there: the load is not landed")
    items_file = REPO / REVIEW_ITEMS.format(video=video)
    listed = json.loads(items_file.read_text())
    if name in [x["id"] for x in listed["items"]]:
        raise Refused(f"the review list already has an item {name}")
    link = Path(REVIEW_MEDIA.format(video=video)) / clip.name
    item = page_item(ctx, spec, clip.name, first, last, str(link), delivery_table.RENDER_ITEM_BUTTONS)
    print(f"\n--- the page: {name} goes to the top of {items_file.relative_to(REPO)}, its buttons {list(delivery_table.RENDER_ITEM_BUTTONS)}", flush=True)
    if world.dry:
        return
    ctx["job_folder"].mkdir(parents=True, exist_ok=True)
    shutil.copy(items_file, ctx["job_folder"] / f"items.before_{a.by}_{name}.json")
    if not os.path.lexists(REPO / link):
        os.symlink(clip, REPO / link)
    listed["items"].insert(0, item)
    listed["updated"], listed["by"] = utc_now(), a.by
    items_file.write_text(json.dumps(listed, indent=1))
    back = json.loads(items_file.read_text())
    print(f"    added: the list's first items are {[x['id'] for x in back['items'][:3]]}; {len(back['items'])} items; updated {back['updated']}")


def last_line(step: str, name: str, ended, now: str) -> tuple[int, str]:
    """A step's last line and its exit status, from what the step returned: nothing for a step that only runs or
    refuses, (status, words) for one that ends on a capture's verify, gate and strip."""
    code, words = ended or (0, "")
    if not code:
        return 0, f"{step} of {name} ended {now}"
    if step == "land":
        return code, f"{step} of {name} ENDED {now} AND IS NOT GOOD: {words}. EXIT {code}."
    return code, (f"{step} of {name} ENDED {now} AND IS NOT GOOD: {words}. EXIT {code}. The capture and its strip are there to be read. "
                  + ("`plan` refuses this load as it stands." if step == "first" else "`render` refuses this load as it stands."))


def step_help(step: str) -> str:
    """What one step does and refuses, and the line that runs it, cut from this file's own words above: `STEP --help`
    printed the tool's one help for every step until 2026-10-13."""
    doc = __doc__ or ""
    usage = [line.strip() for line in doc.splitlines() if line.strip().startswith(f"<python> bench/load_chain.py {step} ")]
    said, taking = [], False
    for line in doc.splitlines():
        if line.startswith(f"- `{step}`:"):
            taking = True
        elif taking and (line.startswith("- `") or not line.strip()):
            break
        if taking:
            said.append(line)
    return "\n".join(usage + [""] + said + ["", "Every step takes --by SESSION, and --dry: print the commands and run none, refusing what the step "
                                                  "refuses on the files alone. `load_chain.py --help` has every flag."])


STEP = {"first": step_first, "plan": step_plan, "on-plan": step_on_plan, "band-again": step_band_again, "table": step_table, "no-look": step_no_look, "render": step_render, "land": step_land, "page": step_page}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=STEPS)
    p.add_argument("job_file", type=Path)
    p.add_argument("--by", required=True, help="the session that runs the step: who makes a capture, gives a go, lands a load")
    p.add_argument("--host", default=HOST)
    p.add_argument("--dry", action="store_true", help="print the commands and run none; the step still refuses what it refuses on the files alone")
    p.add_argument("--grounds", help="render: a file saying who relayed the go, at what time by whose clock, and on what it rests")
    p.add_argument("--show", help="render: a file saying what the render is to show")
    p.add_argument("--not-first", help="no-look: a file with the maker's statement that the load is not the first of its kind, and which earlier load of the kind it follows")
    p.add_argument("--model-store", default="", help=f"land: the folder holding the pose model's folder; over the environment's {MODEL_STORE}")
    p.add_argument("--spec", help="page: a JSON file with the page's words: title_rest, state_words, what, look, prompt_note")
    if len(sys.argv) > 2 and sys.argv[1] in STEPS and ("-h" in sys.argv[2:] or "--help" in sys.argv[2:]):
        print(step_help(sys.argv[1]))
        return 0
    a = p.parse_args()
    try:
        ctx = read_job(a.job_file)
        print(f"{a.step}: {ctx['name']} ({ctx['subject']}, source frames {ctx['load']['first']}-{ctx['load']['first'] + ctx['load']['frames'] - 1}), by {a.by}, "
              f"{utc_now()}; capture {ctx['capture'].name}; masks from the preview {ctx['preview']['name']}, whose files begin on source frame {ctx['preview']['first']}")
        ended = STEP[a.step](ctx, World(a.host, a.dry), a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 5
    code, line = last_line(a.step, ctx["name"], ended, utc_now())
    print("\n" + line)
    return code


if __name__ == "__main__":
    sys.exit(main())
