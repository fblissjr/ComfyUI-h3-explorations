#!/usr/bin/env python3
"""One load of a masked job, a step at a time, read from its job file: first, plan, on-plan, render, land, page.

    <python> bench/load_chain.py first   JOB.json --by SESSION
    <python> bench/load_chain.py plan    JOB.json --by SESSION
    <python> bench/load_chain.py on-plan JOB.json --by SESSION
        (a reader who did not make the capture looks at the plan's strip: `capture_masked_run.py looked`)
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
  and its strip drawn. Nothing is renamed or moved and no record is rewritten.
- `render`: the queue empty; the last look on this load, on the capture on the node's plan, later than the
  capture (compared as times; refused when the two cannot be set in one zone) and not by the capture's maker;
  the gate clear; the job file byte-equal to the one the plan was built from (`job.plan.json`); the go written
  to `<load>.render.go.md` BESIDE the job file (the job file is not edited after its plan is built); the render
  graph built, differing from the plan graph in the song node's `preview` and `filename_prefix` only (and, for
  a hair-band load, in the band's own keep-out file when the two decode equal), with no uncommitted node code;
  every mask video the render graph reads decoding equal to what the plan read. Then it is queued.
- `land`: the queue empty (no pose pass beside a render); the pose pass on the one window's own video, or for a
  load of several windows on the joined file, held to the load's frame count first; the render's cost lines from
  the server's history; the pose pass in the subject's tracked box; the stacked clip.
- `page`: the render's item at the top of the video's review list, its buttons `delivery_table.RENDER_ITEM_BUTTONS`
  (the words the generated delivery table reads), a copy of the list kept in the job folder first.

Every command a step runs is printed before it runs and its output kept under the job folder (`chain/`).
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
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

STEPS = ("first", "plan", "on-plan", "render", "land", "page")
HOST = "127.0.0.1:8188"
#: The environment key for the folder that holds the pose model's own folder (`h3_config.POSE_MODEL`): a fact of
#: the machine, so never a path in a tracked file. Named 2026-10-13 with the lead.
MODEL_STORE = "H3_MODEL_STORE"
#: A subject's keys in the job file that are options of its `--mask` in a capture; only what the job says is passed.
MASK_OPTIONS = ("pieces", "least", "hold", "drop", "grade")
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
    if subject not in order:
        raise Refused(f"the load's subject {subject!r} is not a subject of the preview {pjob['name']} ({', '.join(order)})")
    mine = next((s for s in job.get("subjects", []) if s["label"] == subject), {})
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
        raise Refused(f"{job_file}: texts.{load.get('text')}.by is empty: say who wrote the text, and from which frames")
    return {"job_file": Path(job_file), "raw": raw, "job": job, "load": load, "name": load["name"], "subject": subject, "subject_keys": mine,
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
    return out


def mask_specs(ctx: dict, files: dict) -> list[str]:
    """The `--mask` of every preview subject for a capture of this load: the load's subject with its part, class map,
    shot table and (unless its job says `pose: false`) its pose table; every other with its track and shot table.

    `at`, `shots_at` and `pose_at` are THE PREVIEW'S first frame: the source frame its files' frame 0 is on. The
    scripts this replaces passed the capture's own first frame, which is the same number only when a load begins
    where its preview does."""
    at, by = ctx["preview"]["first"], ctx["preview"]["name"]
    out = []
    for label in ctx["preview"]["order"]:
        f = files[label]
        if label != ctx["subject"]:
            out.append(f"{label}:track={f['track']},shots={f['shots']},shots_at={at},by={by},at={at}")
            continue
        for kind in ("parts", "classes"):
            if f[kind] is None:
                raise Refused(f"the preview {by} saved no {kind} video for the load's subject {label!r}")
        spec = f"{label}:track={f['track']},parts={f['parts']},classes={f['classes']},shots={f['shots']},shots_at={at}"
        if ctx["subject_keys"].get("pose", True) is not False:
            if f["pose"] is None:
                raise Refused(f"the job does not say `pose: false` for {label!r} and the preview {by} wrote no pose table for them")
            spec += f",pose={f['pose']},pose_at={at}"
        spec += f",by={by},at={at}"
        for key in MASK_OPTIONS:              # only what the job says: no default of `pieces=largest`
            if key in ctx["subject_keys"]:
                spec += f",{key}={ctx['subject_keys'][key]}"
        out.append(spec)
    return out


def others_of(ctx: dict) -> list[str]:
    return [label for label in ctx["preview"]["order"] if label != ctx["subject"]]


def still_of(ctx: dict, inputs: Path) -> Path:
    still = ctx["load"].get("still") or ctx["subject_keys"].get("still")
    if not still:
        raise Refused(f"neither the load nor its subject {ctx['subject']!r} names a `still`")
    path = Path(still) if Path(still).is_absolute() else inputs / still
    if not path.is_file():
        raise Refused(f"the still {still!r} is not a file under {inputs}")
    return path


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


def the_look(ctx: dict) -> dict:
    """The look a render rests on: the last one recorded on the capture, which must be of this load, on the capture
    on the node's plan, later than the capture and not by the capture's maker. Returns what the go file says of it."""
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
        raise Refused(f"the last look ({look['at']}) is not later than the capture ({status['written']}): it is a look of an earlier capture")
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


def masks_as_planned(ctx: dict, render: dict, hashed=frame_hash) -> list[str]:
    """Why a mask video the render graph reads is not what the plan read; an empty list when every one decodes equal
    to the hash the on-plan step recorded before it built the capture again in place."""
    record = ctx["job_folder"] / f"{ctx['name']}.plan.masks.json"
    if not record.is_file():
        return [f"{record.name} is not there: the on-plan step, which records what the plan read, has not run"]
    planned = json.loads(record.read_text())["masks"]
    said = []
    for path in mask_videos(render, ctx["capture"]):
        if path not in planned:
            said.append(f"the render graph reads {path}, which the plan graph did not")
        elif hashed(path) != planned[path]:
            said.append(f"{path} no longer decodes equal to what the plan read")
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
            f"**The look** on this capture on the node's plan: {look['by']}, {look['at_utc']}. The capture was written {look['capture_written_utc']} "
            f"by {look['capture_made_by']}.\n\n"
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

        The output is kept under the job folder (`chain/`), unless the step names another place (`ctx["logs"]`). A
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


def empty_queue(world, why: str) -> None:
    running, pending = world.queue()
    if (running, pending) != (0, 0):
        raise Refused(f"the queue is not empty ({running} running, {pending} pending): {why}")


# ---------------------------------------------------------------- the steps

def tool(world, name: str) -> list[str]:
    return [world.python, f"bench/{name}"]


def capture_args(ctx: dict, world, files: dict) -> list[str]:
    job = ctx["job"]
    out = ["--folder", ctx["capture"], "--source", job["source"], "--first", job["first"], "--frames", job["frames"]]
    for spec in mask_specs(ctx, files):
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


def gate_and_strip(ctx: dict, world, by: str, who_is_who: Path, voice: dict, strip_folder: str = "") -> tuple[int, str]:
    cap, job = ctx["capture"], ctx["job"]
    verify = world.run(ctx, f"verify{strip_folder and ' first'}", tool(world, "capture_masked_run.py") + ["verify", cap, "--by", by], stop=False)
    gate = world.run(ctx, f"gate{strip_folder and ' first'}", tool(world, "capture_masked_run.py") + ["preflight", cap, "--text", ctx["text"], "--voice-spans", voice["spans"],
                                                                                                     "--who-is-who", who_is_who, "--gate"], stop=False)
    strip = tool(world, "capture_masked_run.py") + ["video", cap, "--source", job["source"], "--job", ctx["job_file"], "--load", ctx["name"]]
    if ctx["load"].get("motion_file"):
        strip += ["--motion", ctx["load"]["motion_file"]]
    strip += ["--still", f"{ctx['subject']}={still_of(ctx, world.inputs)}"]
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
    for path in (cap, cap.with_name(cap.name + "_first")):
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
    who = who_is_who_file(ctx, world, files) if ctx["job"].get("who_is_who") else ctx["job_folder"] / f"{ctx['name']}.who_is_who.json"
    if not who.is_file() and not world.dry:
        raise Refused(f"{who} is not there: the first step, which makes it, has not run for this job folder")
    masks = {path: hashed(path) for path in mask_videos(graph_of(plan_graph), cap)}
    print(f"\n--- what the plan read: {len(masks)} mask video(s) under {cap.name}, each frame hashed", flush=True)
    if not world.dry:
        (ctx["job_folder"] / f"{ctx['name']}.plan.masks.json").write_text(json.dumps(
            {"what": "the frame-by-frame hash of every mask video the plan graph reads, taken before the capture was built again in place; "
                     "the render step refuses a mask that no longer decodes equal", "at": utc_now(), "by": a.by, "masks": masks}, indent=1) + "\n")
        shutil.copytree(cap, first, symlinks=True)
        differs = same_tree(cap, first)
        if differs:
            raise Refused(f"the copy {first.name} is not the capture it was copied from ({'; '.join(differs[:4])}): nothing was overwritten")
        print(f"--- copied {cap.name} to {first.name}; the copy holds the same files with the same bytes", flush=True)
    run = f"{ctx['name']}:preview={windows},subject={ctx['subject']}" + (f",others={'+'.join(others_of(ctx))}" if others_of(ctx) else "") + f",at={ctx['load']['first']}"
    world.run(ctx, "files on plan", tool(world, "capture_masked_run.py") + ["files"] + capture_args(ctx, world, files)
              + ["--run", run, "--voice", voice["per_frame"], "--made-by", a.by, "--overwrite"])
    return gate_and_strip(ctx, world, a.by, who, voice)


def step_render(ctx: dict, world, a, hashed=frame_hash) -> None:
    grounds, show = (Path(p).read_text().strip() if p and Path(p).is_file() else "" for p in (a.grounds, a.show))
    if not grounds or not show:
        raise Refused("--grounds and --show are each a file with words in it: who relayed the go and on what it rests, and what the render is to show")
    name, folder = ctx["name"], ctx["job_folder"]
    go = ctx["job_file"].parent / f"{name}.render.go.md"
    if (folder / f"{name}.render.json").exists() or go.exists():
        raise Refused(f"a render of {name} is already built or already has a go ({go.name}): never a second one under a name")
    empty_queue(world, "a render is queued on an empty queue")
    look = the_look(ctx)
    job_as_planned(ctx)
    planned = folder / f"{name}.plan.queued.json"
    if not planned.is_file():
        raise Refused(f"{planned} is not there: the plan was not queued through the job tool")
    now = utc_now()
    text = go_text(ctx, a.by, now, grounds, show, look, json.loads(planned.read_text()).get("prompt_id", "not recorded"), world.card())
    print(f"\n--- the go, {now}: the look is {look['by']}'s of {look['at_utc']}, on a capture written {look['capture_written_utc']} by {look['capture_made_by']}", flush=True)
    if not world.dry:
        go.write_text(text)
    world.run(ctx, "build render", tool(world, "masked_job.py") + ["build", "render", ctx["job_file"], "--date", ctx["day"]])
    if world.dry:
        print("--- (dry: the render graph is not built, so it is not held against the plan graph)")
        return
    shutil.copyfile(folder / "chain" / f"{name}.build_render.out", folder / f"{name}.render.build_printed.txt")
    render = graph_of(folder / f"{name}.render.json")
    said = render_against_plan(graph_of(folder / f"{name}.plan.json"), render, json.loads((folder / f"{name}.render.built.json").read_text()), hashed)
    notes, problems = [s[len("note: "):] for s in said if s.startswith("note: ")], [s for s in said if not s.startswith("note: ")]
    problems += masks_as_planned(ctx, render, hashed)
    record = {"what": "what the render step of bench/load_chain.py held before it queued", "at": now, "by": a.by, "go_file": str(go), "look": look,
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


def step_land(ctx: dict, world, a) -> None:
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
    points = cap / f"{name}_render_points.npz"
    if clip.exists() or points.exists():
        raise Refused(f"the stacked clip or the render's points exist ({clip.name}, {points.name}): this load is landed")
    queued = ctx["job_folder"] / f"{name}.render.queued.json"
    if not queued.is_file():
        raise Refused(f"{queued} is not there: the render was not queued through the job tool")
    print("\n--- the render, by the server's history:\n" + "\n".join("    " + line for line in world.costs(json.loads(queued.read_text())["prompt_id"])), flush=True)
    manifest = json.loads((cap / "manifest.json").read_text())
    seen = next(s for s in manifest["subjects"] if s["label"] == ctx["subject"])["sightings"][0]
    box = cap / "subjects" / ctx["subject"] / f"track__{seen['by']}.mkv"      # the capture's own copy, whose frame 0 is the capture's first frame
    world.run(ctx, "pose pass", tool(world, "pose_along_shot.py") + ["run", "--video", window, "--first", 0, "--count", load["frames"],
                                                                     "--canvas", f"{job['canvas'][0]}x{job['canvas'][1]}", "--source-first", first,
                                                                     "--box-mask", box, "--box-mask-first", manifest["first_frame"],
                                                                     "--model-folder", model, "--out", points, "--by", a.by], stop=False)
    stack = tool(world, "capture_masked_run.py") + ["video", cap, "--source", job["source"], "--render", r / f"{name}_00001.mp4"]
    if load.get("motion_file"):
        stack += ["--motion", load["motion_file"]]
    world.run(ctx, "stack", stack + ["--still", f"{ctx['subject']}={still_of(ctx, world.inputs)}", "--out", clip])


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
    return code, (f"{step} of {name} ENDED {now} AND IS NOT GOOD: {words}. EXIT {code}. The capture and its strip are there to be read. "
                  + ("`plan` refuses this load as it stands." if step == "first" else "`render` refuses this load as it stands."))


STEP = {"first": step_first, "plan": step_plan, "on-plan": step_on_plan, "render": step_render, "land": step_land, "page": step_page}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=STEPS)
    p.add_argument("job_file", type=Path)
    p.add_argument("--by", required=True, help="the session that runs the step: who makes a capture, gives a go, lands a load")
    p.add_argument("--host", default=HOST)
    p.add_argument("--dry", action="store_true", help="print the commands and run none; the step still refuses what it refuses on the files alone")
    p.add_argument("--grounds", help="render: a file saying who relayed the go, at what time by whose clock, and on what it rests")
    p.add_argument("--show", help="render: a file saying what the render is to show")
    p.add_argument("--model-store", default="", help=f"land: the folder holding the pose model's folder; over the environment's {MODEL_STORE}")
    p.add_argument("--spec", help="page: a JSON file with the page's words: title_rest, state_words, what, look, prompt_note")
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
