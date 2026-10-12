#!/usr/bin/env python3
"""Build the graphs of a masked job from its job file, and queue them.

    <python> bench/masked_job.py build preview JOB.json          # writes data/<date>_<name>_job/, touches no server
    <python> bench/masked_job.py queue data/<date>_<name>_job preview [--wait]
    <python> bench/masked_job.py collect data/<date>_<name>_job preview      # a run queued without --wait: record what it saved
    <python> bench/masked_job.py build plan JOB.json [--load NAME ...]      # what the node says each load will be given
    <python> bench/masked_job.py queue data/<date>_<name>_job plan --load NAME [--wait]
    <python> bench/masked_job.py build render JOB.json [--load NAME ...]
    <python> bench/masked_job.py queue data/<date>_<name>_job render --load NAME [--wait]   # behind verify and the gate

**What it buys.** Every masked session until 2026-10-10 wrote its own script to turn "this clip, these people,
these shots" into graphs, each one patching a graph lifted out of an earlier render, and a fix made in one script
never reached the next. This is that script once, tracked, with a check (`bench/check_masked_job.py`). The graphs themselves are `workflows/masked_job_graph.py`'s; this file reads the
job, refuses what must be refused, writes what a run was made from, and queues it. The order a job goes in is the
masking board's `guide-order-of-operations-masked-job`.

**The job file** is the loads file `bench/capture_masked_run.py --loads` takes, with a `job` block beside
`defaults` and `loads` that the capture does not read (`masked_job_graph`'s docstring has its keys).

**build** makes a folder, `data/<date>_<name>_job/` (untracked, beside the capture `data/<date>_<name>` the
previews are read into), and writes there: `job.json`, the job file as given; the graph, `<stage>.json`; and
`<stage>.built.json`, which says which commit built it (and which of the builder's own files were not as that
commit has them), the tracker's mask version, and where the graph will save each file. It reads no server and uses no card. `preview` is the first graph of every job, who is where,
nothing sampled (step 1 of the order); it ends by printing the capture command the preview's files go into.

**plan** and **render** make one graph a load (`<load>.<stage>.json`), from the shipped masked graph and the
files of the capture the job block names (`capture`, a folder of `bench/capture_masked_run.py files`): the
subject's mask, the part as the capture filled and graded it, the other subjects' masks, and the mesh the
preview saved. Nothing is tracked again. A plan is the render's graph with the song node on `preview`: it
samples nothing and writes, under its own folder, what each window will be given, which is what the gate
reads (step 3).

**A load's text is data the job serves, tied to the frames it was written from** (the owner, 2026-10-10: a text
kept in a graph or in code is forgotten there and then used for a window it was not written for). `text` in a
load is a label, and the job block's `texts` maps a label to `{"file", "frames": [first, last], "by"}`: a file
beside the job file, the source frames the words were written from, and who wrote them. A text serves the loads
inside its frames and no other, and it is of one shot: its frames hold no cut of the capture. So one text for
loads of different shots cannot be built, and neither can a load that holds a cut: one load a shot, each with
a text of its own frames. That second part is an edge accepted, not a law of the model (`ONE_SHOT`): a load over
several shots with a text that describes each shot under its own header is not built yet. The graph carries the text
because the node needs it; it is a product of the job and never where a text is kept or edited.

`build` prints, for each load, what each model will be shown (step 6): the frames, the window and what is held past
the load's end; the still's own size against the size the video model is shown it at, with "enlarged by" said when
it is; the region; the motion video and which model gets how much of it; the audio; the seed; and the
text in full, with the frames it was written from and who wrote it.

**A default measured on one clip is a starting value for another** (the owner, 2026-10-10: a rule that depends on
the picture takes its value from this clip, measured before anything renders). So every such setting can be set
in a load or in the file's `defaults` (`masked_job_graph.SETTINGS`), and a load's build record says two things
beside its graph: `settings`, each value used and whether the job file or the builder's default set it; and
`what_this_clips_preview_measured`, gathered from the capture and measured by nothing here: the cuts, the
subject's share of the frame and its box over the load's frames, the part's share, the load against its window.
A figure the capture does not hold is said to be absent, in words.

Two keys of the job block for a job of several subjects: a subject's `still` is the still of every load of that
subject unless a load names its own; and `left_out`, a list of `{subject, first, frames, why}`, names the
stretches left as the source on purpose. Those make no graph and are written into every load's record
(`left_out_on_purpose`), so a shot with no load reads as a decision and not as something the builder lost.

**queue** is the only step that uses the card, so it runs on the lead's word. It validates the graph against
what the running server serves (`build_workflows.validate_api`), posts it, and writes `<stage>.queued.json`;
with `--wait` it stays until the server is done and writes what was saved. **collect** does that last part for
a run queued without `--wait`, from the server's history, and posts nothing: the plan and render stages find
the mesh a preview saved in that record, so a preview is collected before a load is built. For a preview it
also exits 1, naming them, when a file the graph was to save is not on disk, and when a subject was not found
in the load at all (its shot table has no pick, or no frame of the subject on any shot): which trackers come
first and where each is picked decide who can be found, and a subject an earlier tracker took is otherwise
only an "absent" on every row of a table nobody opens.

**What it refuses**, each with exit status `REFUSED`:

- a job whose video has no analysis folder, or one that is another file's or stale: every stage is built on
  `data/<stem>/` of the input video (`clip_folder.py`), found from the name of the file the job renders from
  and held to the identity recorded there; the refusal prints the `bench/clip_analysis.py init` command. A
  build record carries the folder's name and fingerprint (`analysis`);

- a job folder that exists, or an output folder that exists: a new name for every run (the order's step 1 has
  what a reused preview folder nearly did on 2026-10-10);
- a `corrections` line whose `corrections_at` is not the tracker's mask version today. A correction names a
  person by the number on a tile, and the numbers are that version's;
- a load whose capture is not finished, did not pass `verify`, lacks a file the load needs or does not reach
  the load's frames; a load shorter than the song node's shortest extent (`shortest_load`); a feather wider than
  its margin; a render with no still, or a still
  the server's input folder does not hold (`H3_INPUT_DIR`);
- a text that does not say the frames it was written from or who wrote it (a bare file name, the form until
  2026-10-10, included); a load whose frames do not lie inside its text's; a text whose frames hold a cut of the
  capture's shot tables or reach outside the frames the capture saw; the same words served under two labels.
  Every chosen load is tested before anything is written and the refusal lists them all (`texts_for`);
- at `queue` of a render, a text file that no longer reads as the text in the built graph. `--override` does not
  let it through: it answers uncommitted node code and nothing else;
- at `queue`: a graph that was queued before; an input the server does not serve; and node code in the tree
  that is not committed, unless `--override "WHO: WHY"` is given, which is written into the record. The server
  serves a commit, and a graph built beside uncommitted node code can describe a server nobody is running;
- at `queue` of a render, the gate (steps 4 and 5): `verify` is run on the capture again and must exit 0; the
  capture's `flags.json` must say `clear`; and the capture must hold a run of the load's name whose region
  was read from the plan this tool's `plan` stage queued, not worked out from the masks. A blocking flag is
  answered in the capture (`outcome <capture> <flag> overridden --by WHO --note WHY`), never here: there is no
  override of the gate in this tool.

What the builder sets so that a flag does not fire: a load's audio is cut to the load. (A load shorter than its
window keeps its region open over the held frames by the node's own rule since 2026-10-10; the builder set it
until then.)

**What is shown and never refused: the look.** Until 2026-10-12 `build plan` and `queue` of a render refused a
load nobody had looked at; the owner ruled that the look is shown and never refused (Open List row c22,
15:51:48Z). The record's `look` now holds the look that matches this build (`looked_at`: later than the capture's
own `written`, its `shown` row for the load with this build's frames, text frames and text hash; the strip and
the record are `bench/capture_masked_run.py video` and `looked`), or says that none is recorded and why, with
the strip's path when one is drawn and the command when none is (`look_or_shown`); the same lines are printed
with the build's and before a render is queued (`look_lines`). The capture is written again when it reads the
plans, so a job is looked at twice; a session's look is still asked for by the lane's rules, and nothing here
refuses on it.

**What refuses nothing.** Every refusal above stands in `build` and `queue` and nowhere else. A built graph
posted to the server by any other road (a session's own script, the canvas) meets none of them: no look, no
gate, no test of its text. And the same words under two labels are caught inside one job file only: two job
files that each serve one file's words to a shot of their own are not seen. Both are known gaps, said here so
that nobody reads the list above as a guarantee about a render.

Not here yet: the one-window patch on a finished render; a named variant of a built and gated load that changes
one stated thing (the masking board, `build-per-video-analysis-folder`), without which a small test is easier
to post by hand than to build.

Run it from ComfyUI's environment, with `H3_OUTPUT_DIR` set as every bench tool that reads renders needs it,
and `H3_INPUT_DIR` when the server's input folder is not the stock one.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import COMFY, REPO  # noqa: E402
from _lib.frames import FPS as MASK_FPS, probe, stream  # noqa: E402

sys.path.insert(0, str(REPO))
import clip_folder  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402
import masked_job_graph as graphs  # noqa: E402

#: The exit status of a refusal, apart from the capture's own (`GATE_BLOCKED`, `VERIFY_FAILED`) and from a crash.
REFUSED = 5
STAGES = ("preview", "plan", "render")
#: What the capture's manifest says of a run whose region it read from the song node's own plan
#: (`bench/capture_masked_run.py`, `read_planned_regions`): the key it sets and a plan worked out from masks lacks.
READ_FROM_THE_NODE = "plan"
#: Where node code is not: a change under one of these is not loaded by the server. `workflows/h3_config.py` is
#: the exception: node files load it by path at run time (`sol_observe.py`, `pdd_lora.py`).
NOT_NODE_CODE = ("bench/", "docs/", "internal/", "changelog.d/", "data/", "workflows/")
NODE_CODE_ALL_THE_SAME = ("workflows/h3_config.py",)
#: The files a graph is made by. A record that names a commit says so only when these are in it.
BUILDER = ("workflows/masked_job_graph.py", "workflows/h3_config.py", "bench/masked_job.py")
HOST = "127.0.0.1:8188"


class Refused(Exception):
    """Something the tool will not do, said in words the caller can act on."""


def tracker_mask_version(repo: Path = REPO) -> int:
    """`MiniMaxH3SubjectTrack.MASK_VERSION`, read from the source and not imported: the node file needs ComfyUI."""
    for node in ast.walk(ast.parse((repo / "subject_track.py").read_text())):
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3SubjectTrack":
            for line in node.body:
                if isinstance(line, ast.Assign) and any(getattr(t, "id", "") == "MASK_VERSION" for t in line.targets):
                    return int(ast.literal_eval(line.value))
    raise Refused("subject_track.py no longer sets MASK_VERSION on MiniMaxH3SubjectTrack; the corrections rule "
                  "has nothing to read")


def check_corrections(job: dict, version: int) -> None:
    """A correction typed by number is refused unless it says which tracker version its numbers were read at."""
    for subject in job["subjects"]:
        if not str(subject.get("corrections", "")).strip():
            continue
        at = subject.get("corrections_at")
        if at != version:
            raise Refused(
                f"subject {subject['label']!r} has corrections "
                + ("and no `corrections_at`" if at is None else f"read at the tracker's mask version {at}")
                + f"; the tracker is at {version}. A correction names a person by the number on a tile: read the "
                  f"numbers off a tile this version made, then set corrections_at to {version}.")


def analysed(job: dict, data: Path) -> dict:
    """The analysis folder of the video a job renders from, checked to be that file's: its name and fingerprint.

    A job is built on an input video's analysis folder (`clip_folder.py`; made by `bench/clip_analysis.py`)
    or it is not built: the folder is found from the name of the file the job renders from, and the file is
    held to the identity the folder recorded, so a folder that is another file's, or older than the file, is
    as good as none."""
    try:
        folder = clip_folder.folder_of_file(Path(job["source"]).name, data)
        manifest = clip_folder.read_manifest(folder)
        clip_folder.verify(manifest, Path(job["source"]))
    except (clip_folder.Refused, OSError) as e:
        raise Refused(f"{e}\n  (the job renders from {job['source']}; the command is `bench/clip_analysis.py init "
                      f"<the input video's file name> --copy {Path(job['source']).name}`)") from e
    return {"folder": folder.name, "fingerprint": manifest["fingerprint"]}


def job_folder(job: dict, date: str, data: Path) -> Path:
    return data / f"{date}_{job['name']}_job"


def output_folder(job: dict, output: Path) -> Path:
    return output / job["output"] / job["name"]


def head(repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def uncommitted_node_code(repo: Path = REPO) -> list[str]:
    """Tracked Python the server loads that differs from HEAD."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain",
                          "--untracked-files=no", "--", "*.py"],
                         capture_output=True, text=True, check=True).stdout
    paths = [line[3:].strip() for line in out.splitlines() if line.strip()]
    return [p for p in paths if p in NODE_CODE_ALL_THE_SAME or not p.startswith(NOT_NODE_CODE)]


def builder_not_committed(repo: Path = REPO) -> list[str]:
    """The builder's own files that are not as HEAD has them, untracked ones included."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--", *BUILDER],
                         capture_output=True, text=True, check=True).stdout
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def chain_lengths(repo: Path = REPO) -> tuple[int, ...]:
    """`loop_plan.CHAIN_LENGTHS`, read from the source and not imported: the module needs ComfyUI."""
    for line in ast.parse((repo / "loop_plan.py").read_text()).body:
        if isinstance(line, ast.Assign) and any(getattr(t, "id", "") == "CHAIN_LENGTHS" for t in line.targets):
            return tuple(ast.literal_eval(line.value))
    raise Refused("loop_plan.py no longer sets CHAIN_LENGTHS; a load's window has nothing to be chosen from")


def shortest_load(repo: Path = REPO) -> tuple[float, int]:
    """The shortest extent the song node takes, in seconds, and the fewest frames a load may therefore be.

    Read from the node's own schema in the source (the `seconds` member of its `extent`), not imported and not
    copied: the node file needs ComfyUI. A load shorter than this is refused by the server at `queue`; it is
    refused here first, at `build`, with the number."""
    for node in ast.walk(ast.parse((repo / "audio_freeze_song.py").read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Input" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "seconds"):
            for keyword in node.keywords:
                if keyword.arg == "min":
                    seconds = float(ast.literal_eval(keyword.value))
                    return seconds, int(-(-seconds * h3_config.FPS // 1))
    raise Refused("audio_freeze_song.py no longer declares a `seconds` input with a minimum; the shortest load has "
                  "nothing to be read from")


def loads_of(given: dict) -> list[dict]:
    """A job file's loads with its defaults under each, named as `bench/capture_masked_run.py` names them."""
    out, seen = [], set()
    for n, load in enumerate(given.get("loads", [])):
        try:
            # a load of two subjects is read as its first subject's, with the second beside it (`one_subject_view`)
            one = graphs.one_subject_view({**given.get("defaults", {}), **load})
        except ValueError as e:
            raise Refused(str(e)) from e
        lacking = [k for k in ("first", "frames", "subject") if k not in one]
        if lacking:
            raise Refused(f"load {n} lacks {', '.join(lacking)}")
        one.setdefault("name", f"{one['subject']}_{int(one['first']):04d}")
        if one["name"] in seen:
            raise Refused(f"two loads are named {one['name']!r}; give one a `name`")
        seen.add(one["name"])
        if isinstance(one.get("others"), str):
            one["others"] = [x for x in one["others"].split("+") if x]
        out.append(one)
    return out


def left_out(job: dict) -> list[dict]:
    """The stretches a job leaves as the source ON PURPOSE (`left_out` in the job block): a subject, its frames
    and why. They make no graph; they are carried into every record of the job, so that a stretch with no load
    reads as a decision somebody wrote down and not as one the builder lost."""
    out = []
    for n, entry in enumerate(job.get("left_out") or []):
        lacking = [k for k in ("subject", "first", "frames", "why") if not entry.get(k) and entry.get(k) != 0]
        if lacking:
            raise Refused(f"left_out entry {n} lacks {', '.join(lacking)}: a stretch left out on purpose says whose, "
                          f"which frames and why")
        if entry["subject"] not in [s["label"] for s in job["subjects"]]:
            raise Refused(f"left_out entry {n} names {entry['subject']!r}, who is not one of the job's subjects")
        out.append(dict(entry))
    return out


#: Said with every refusal of a text, so that the person meeting it learns the rule and not only the key (the
#: owner, 2026-10-10; AGENTS.md "Operative rules": a text is data served by what plans the shots).
TEXT_RULE = ('A text needs the frames it was written from and who wrote it, because a text is true of one window '
             'only: written from one stretch of one shot, it is wrong for another. In the job block: "texts": '
             '{"LABEL": {"file": "a file name beside the job file", "frames": [first, last], "by": "who wrote it"}}, '
             'source frames, both included.')


#: Said once when a text is refused for holding a cut. The way out, and what the refusal is not (the lead,
#: 2026-10-10): the owner ruled out one shot's text served to another shot, not long loads.
ONE_SHOT = ("The way out: one load a shot, each with a text written from its own frames. This is an edge accepted "
            "and not a law of the model: the vendor's format has a text for several shots, each under its own shot "
            "header, and a load over several shots with such a text is not built yet (its headers would have to be "
            "held against the capture's cuts).")


def text_entry(job: dict, label: str, beside: Path) -> dict:
    """One entry of a job's `texts`, as data: the words, the source frames they were written from, who wrote
    them, and a hash of exactly the string that goes into a graph. The builder writes no text and guesses none."""
    named = (job.get("texts") or {}).get(label)
    if named is None:
        raise Refused(f"the job block's `texts` has no entry `{label}`")
    if not isinstance(named, dict):
        raise Refused(f"text `{label}` is given as {named!r}, a bare file name, with no frames and no writer")
    file, frames, by = named.get("file"), named.get("frames"), named.get("by")
    lacking = []
    if not (isinstance(frames, list) and len(frames) == 2 and all(type(n) is int for n in frames) and 0 <= frames[0] <= frames[1]):
        lacking.append(f"the source frames it was written from (`frames` is {frames!r}; it is [first, last])")
    if not isinstance(by, str) or not by.strip():
        lacking.append("who wrote it (`by`)")
    if lacking:
        raise Refused(f"text `{label}` does not say " + " or ".join(lacking))
    if not isinstance(file, str) or not file or Path(file).name != file:
        raise Refused(f"text `{label}`: its `file` is {file!r}; a text is a file beside the job file, named with no folder")
    path = beside / file
    if not path.is_file() or not path.read_text().strip():
        raise Refused(f"text `{label}`: its file {path} is missing or empty")
    text = path.read_text().strip()
    return {"label": label, "file": str(path), "frames": frames, "by": by.strip(), "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def text_of(job: dict, one: dict, beside: Path) -> dict:
    """A load's text (`text_entry`), refused when the load's frames do not lie inside the frames the text was
    written from: a text serves the loads inside its own frames and no other."""
    name, label = one.get("name"), one.get("text")
    if not label:
        raise Refused(f"load {name!r} has no text: give it a `text` label. {TEXT_RULE}")
    try:
        told = text_entry(job, label, beside)
    except Refused as e:
        raise Refused(f"load {name!r}: {e}. {TEXT_RULE}") from e
    first, last = one["first"], one["first"] + one["frames"] - 1
    if not told["frames"][0] <= first <= last <= told["frames"][1]:
        raise Refused(f"load {name!r} is source frames {first}-{last} and its text `{label}` was written from "
                      f"{told['frames'][0]}-{told['frames'][1]}: the text does not cover the load")
    return told


def cuts_of(capture: Path) -> tuple[list[int], int, int]:
    """Every cut a capture's shot tables hold, on source frames, and the first and last source frame the capture
    saw. Each subject's tracker writes a table; a cut that any of them holds is a cut, so tables that disagree
    are read on the strict side."""
    if not (capture / "manifest.json").is_file():
        raise Refused(f"{capture} is not a capture folder (no manifest.json); the job block's `capture` names it")
    manifest = json.loads((capture / "manifest.json").read_text())
    cuts, tables = set(), 0
    for subject in manifest["subjects"]:
        for sighting in subject["sightings"]:
            table = capture / "subjects" / subject["label"] / f"shots__{sighting['by']}.json"
            if table.is_file():
                at = sighting.get("shots_first_source_frame", manifest["first_frame"])
                cuts |= {at + shot["first_frame"] for shot in json.loads(table.read_text()).get("shots", [])[1:]}
                tables += 1
    if not tables:
        raise Refused(f"the capture {capture.name} holds no shot table: where its shots are cut is not known, so no "
                      f"text can be held to one shot")
    return sorted(cuts), manifest["first_frame"], manifest["first_frame"] + manifest["frames"] - 1


def texts_for(job: dict, chosen: list[dict], beside: Path, capture: Path) -> dict:
    """The text of every chosen load, by load name, each held to the rule (`TEXT_RULE`): it says its frames and
    its writer; it covers its load; it is of one shot, so its frames hold no cut of the capture and stay inside
    the frames the capture saw; and no two labels serve the same words. One text across different shots, the
    fault this is for, needs no rule of its own: a text of one shot covers no load outside that shot. Every
    load is tested before anything is said, and the refusal lists them all."""
    cuts, lo, hi = cuts_of(capture)
    told, refused, entries, bad, cut = {}, [], {}, {}, False
    for label in sorted({one["text"] for one in chosen if one.get("text")}):
        try:
            entries[label] = text_entry(job, label, beside)
        except Refused as e:
            bad[label] = str(e)
    for label, why in bad.items():
        refused.append(f"{why}; taken by load(s) " + ", ".join(repr(one["name"]) for one in chosen if one.get("text") == label))
    for one in chosen:
        if one.get("text") in bad:
            continue
        try:
            mine = text_of(job, one, beside)
        except Refused as e:
            refused.append(str(e).removesuffix(f" {TEXT_RULE}"))
            continue
        a, b = mine["frames"]
        inside = [c for c in cuts if a < c <= b]
        if a < lo or b > hi:
            refused.append(f"load {one['name']!r}: its text `{mine['label']}` was written from source frames {a}-{b}, and the "
                           f"capture {capture.name} holds {lo}-{hi}: where the shots are cut outside those is not known")
        elif inside:
            cut = True
            refused.append(f"load {one['name']!r}: its text `{mine['label']}` was written from source frames {a}-{b}, which "
                           f"hold the cut(s) on {', '.join(map(str, inside))}: a text is of one shot")
        else:
            told[one["name"]] = mine
    same = {}
    for label in sorted(job.get("texts") or {}):
        try:
            entry = entries.get(label) or text_entry(job, label, beside)
        except Refused:
            continue                                   # a label no chosen load takes, and not a text yet
        same.setdefault(entry["sha256"], []).append(entry)
    for group in same.values():
        if len(group) > 1:
            refused.append("the same words are served under " + " and ".join(
                f"`{e['label']}` ({Path(e['file']).name}, frames {e['frames'][0]}-{e['frames'][1]})" for e in group)
                + ": one text for several stretches is the fault this rule is for, whatever the labels and the files "
                  "are called. A text that serves another stretch is written for it, from its frames")
    if refused:
        raise Refused(f"{len(refused)} thing(s) about the texts of this job:\n  - " + "\n  - ".join(refused) + f"\n{TEXT_RULE}"
                      + (f"\n{ONE_SHOT}" if cut else ""))
    return told


def text_shown(one: dict, told: dict) -> list[str]:
    """A load's text as a person reads it before it is used: where it was written from, by whom, every word."""
    return [f"text `{told['label']}` ({Path(told['file']).name}), written from source frames {told['frames'][0]}-"
            f"{told['frames'][1]} by {told['by']}; this load is {one['first']}-{one['first'] + one['frames'] - 1}. "
            f"The whole text:"] + [f"  | {line}" for line in told["text"].splitlines()]


def _when(value, what: str) -> datetime.datetime:
    """A time as `bench/capture_masked_run.py` writes it: local, with no zone. Anything else is refused and not
    guessed at, so that two clocks cannot make a look read hours newer or older than it is."""
    try:
        at = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise Refused(f"{what} is {value!r}, which is not a time") from None
    if at.tzinfo is not None:
        raise Refused(f"{what} is {value!r}, a time with a zone: the capture tool writes local time with none, and "
                      f"the look and the capture are compared on that one clock")
    return at


def looked_at(capture: Path, one: dict, told: dict) -> dict:
    """The look a load is built on, or a refusal. A passing check says no rule failed, never that the masks are
    on the right person or that the text is of these frames (the owner, 2026-10-10), so somebody who did not
    make the capture goes through each load's strip, the whole text beside its frames, before a plan or a
    render is built on it. `bench/capture_masked_run.py video --job --load` draws the strip and writes what it
    showed; `looked` copies that into a line of the capture's `looks.json`. A line counts for a load when it is
    later than the capture's own `written` (what was made again is looked at again) and its `shown` row for the
    load has the frames, the text's frames and the text's hash this build has. Held by what was shown and not
    by a file's time: a text copied in with an old time, or a label pointed at another file, is another text."""
    name, frames = one["name"], [one["first"], one["first"] + one["frames"] - 1]
    status = capture / "status.json"
    written = _when(json.loads(status.read_text()).get("written") if status.is_file() else None,
                    f"`written` in the status.json of the capture {capture.name}")
    kept = capture / "looks.json"
    looks = json.loads(kept.read_text()).get("looks", []) if kept.is_file() else []
    mine = [(look, row) for look in looks for row in look.get("shown") or [] if row.get("load") == name]
    how = (f" To look: `bench/capture_masked_run.py video {capture} --source CLIP --job JOB.json --load {name}` draws the "
           f"load's strip with its text; somebody who did not make the capture goes through it; `bench/capture_masked_run.py "
           f"looked {capture} --by WHO --loads {name} --note WHAT_WAS_SEEN` records it.")
    if not mine:
        raise Refused(f"nobody has looked at load {name!r} on the capture {capture.name}." + how)
    since = [(look, row) for look, row in mine if _when(look.get("at"), f"a look's `at` in {kept}") > written]
    if not since:
        look = mine[-1][0]
        raise Refused(f"the last look at load {name!r} ({look.get('by')}, {look.get('at')}) is not later than the capture, "
                      f"written {written.isoformat()}: what was made again is looked at again." + how)
    want = {"frames": frames, "text_frames": told["frames"], "text_sha256": told["sha256"]}
    for look, row in reversed(since):
        if all(row.get(key) == value for key, value in want.items()):
            return {"by": look.get("by"), "at": look.get("at"), "note": look.get("note"), "strip": row.get("strip")}
    look, row = since[-1]
    said = {"frames": f"as source frames {row.get('frames')}, and it is {frames}",
            "text_frames": f"with a text written from {row.get('text_frames')}, and its text is written from {told['frames']}",
            "text_sha256": "with other words in its text than it has now"}
    raise Refused(f"load {name!r} was looked at ({look.get('by')}, {look.get('at')}) "
                  + "; ".join(said[key] for key, value in want.items() if row.get(key) != value)
                  + ": that look is of something else than this build." + how)


def look_or_shown(capture: Path, one: dict, told: dict, look=looked_at) -> dict:
    """The look a load is built on, as `looked_at` finds it, or a record that none is recorded for this build:
    shown and never refused (the owner, 2026-10-12, Open List row c22). The record says why in `looked_at`'s own
    words, names the strip when one is drawn (where the record beside the capture, `looks/<load>.shown.json`, says
    it is: on the share since 2026-10-12, beside the capture before; with when), and gives the command that
    records a look on it, or draws one when none is."""
    try:
        return look(capture, one, told)
    except Refused as e:
        record = capture / "looks" / f"{one['name']}.shown.json"
        named = json.loads(record.read_text()).get("strip") if record.is_file() else None
        strip = Path(named) if named and Path(named).is_absolute() else capture / "looks" / (named or f"{one['name']}.mp4")
        drawn = strip.is_file()
        return {"recorded": False, "why": str(e).split(" To look: ")[0], "strip": str(strip) if drawn else None,
                "strip_drawn": (datetime.datetime.fromtimestamp(strip.stat().st_mtime).isoformat(timespec="seconds")
                                if drawn else None),
                "how": (f"bench/capture_masked_run.py looked {capture} --by WHO --loads {one['name']} --note WHAT_WAS_SEEN"
                        if drawn else
                        f"bench/capture_masked_run.py video {capture} --source CLIP --job JOB.json --load {one['name']}, "
                        f"then `looked`")}


def look_lines(look) -> list[str]:
    """What `build` prints and `queue` prints about a load's look: one line when a look matches this build, two
    when none is recorded (why; the strip's path and when it was drawn, or the command that draws one)."""
    if not isinstance(look, dict):
        return []
    if look.get("recorded", True):
        return [f"looked at by {look.get('by')} at {look.get('at')}" + (f", strip {look['strip']}" if look.get("strip") else "")]
    strip = f"the strip is {look['strip']} (drawn {look['strip_drawn']})" if look.get("strip") else "no strip is drawn"
    return [f"NO LOOK IS RECORDED FOR THIS BUILD: {look['why']}", f"  {strip}; to record one: {look['how']}"]


def text_and_look(built: dict, graph: dict) -> dict:
    """Before a render is queued: the text in the built graph is still the text its file serves (refused when
    not), and the look the load was built on, or that none is recorded (`look_or_shown`; never a refusal)."""
    name, told = built["load"]["name"], built.get("text")
    if not isinstance(told, dict):
        raise Refused(f"load {name!r} was built before a text carried its frames and its writer: build it again. {TEXT_RULE}")
    path = Path(told["file"])
    in_graph = graph[graphs._the(graph, graphs.SONG)]["inputs"]["prompt"]
    if in_graph != told["text"] or not path.is_file() or path.read_text().strip() != in_graph:
        raise Refused(f"the text of load {name!r} changed after it was built: {path} no longer reads as the text in the "
                      f"built graph. A graph is a product of the job and never where a text is kept: build the load again "
                      f"under a new `name`, and it is looked at with the text it now has.")
    return look_or_shown(Path(built["capture"]), built["load"], told)


def capture_files(job: dict, one: dict, capture: Path, mesh_of) -> dict:
    """The files a load is built from, each with the source frame its frame 0 is on; refuses a capture that
    was not finished or checked, and a file that is not there. `mesh_of(label)` finds a subject's saved mesh."""
    if not (capture / "manifest.json").is_file():
        raise Refused(f"{capture} is not a capture folder (no manifest.json); the job block's `capture` names it")
    manifest = json.loads((capture / "manifest.json").read_text())
    status = json.loads((capture / "status.json").read_text()) if (capture / "status.json").is_file() else {}
    if status.get("state") != "done":
        raise Refused(f"the capture {capture.name} was not finished (its status.json says {status.get('state')!r})")
    checked = json.loads((capture / "verify.json").read_text()) if (capture / "verify.json").is_file() else None
    if checked is None or checked.get("failed") or checked.get("not_made"):
        raise Refused(f"the capture {capture.name} has not passed `verify`"
                      + ("" if checked else ": run bench/capture_masked_run.py verify on it first"))
    at, held = manifest["first_frame"], manifest["first_frame"] + manifest["frames"]
    if one["first"] < at or one["first"] + one["frames"] > held:
        raise Refused(f"load {one['name']!r} is on frames {one['first']}..{one['first'] + one['frames'] - 1} and the "
                      f"capture {capture.name} holds {at}..{held - 1}")
    sightings = {s["label"]: [x["by"] for x in s["sightings"]] for s in manifest["subjects"]}

    def of(label: str, kinds: tuple[str, ...]) -> tuple[str, int]:
        if label not in sightings:
            raise Refused(f"load {one['name']!r}: the capture {capture.name} has no subject {label!r}")
        by = one.get("by") or (sightings[label][0] if len(sightings[label]) == 1 else None)
        if by not in sightings[label]:
            raise Refused(f"load {one['name']!r}: {label!r} was seen by {sightings[label]} in the capture; say "
                          f"which with `by`")
        for kind in kinds:
            path = capture / "subjects" / label / f"{kind}__{by}.mkv"
            if path.is_file():
                return str(path), at
        raise Refused(f"load {one['name']!r}: the capture holds no {' or '.join(kinds)} mask of {label!r} by {by!r}")

    files = {"track": of(one["subject"], ("track",)), "parts": None, "motion": None,
             "others": [of(label, ("track",)) for label in one.get("others") or ()]}
    if one.get("carried", "track") == "held":
        if one.get("parts_mask"):
            if not Path(one["parts_mask"]).is_file():
                raise Refused(f"load {one['name']!r}: its parts_mask {one['parts_mask']} is not a file")
            files["parts"] = (str(one["parts_mask"]), one.get("parts_first", at))
        else:
            # the part as the capture filled and graded it; the part as saved only when nothing was doubted
            files["parts"] = of(one["subject"], ("parts_held", "parts"))
    also = one.get("also")
    if also:
        # the second subject of a two-subject load: its track, and its part where it takes one
        files["also"] = {"track": of(also["subject"], ("track",)), "parts": None}
        if also["carried"] == "held":
            if also.get("parts_mask"):
                if not Path(also["parts_mask"]).is_file():
                    raise Refused(f"load {one['name']!r}: its second subject's parts_mask {also['parts_mask']} is not a file")
                files["also"]["parts"] = (str(also["parts_mask"]), also.get("parts_first", at))
            else:
                files["also"]["parts"] = of(also["subject"], ("parts_held", "parts"))
    motion = graphs.motion_of(job, one)
    if motion != "none":
        path = one.get("motion_file") or mesh_of(one["subject"])
        if not path or not Path(path).is_file():
            raise Refused(f"load {one['name']!r} takes motion `{motion}` and no saved mesh of {one['subject']!r} was "
                          f"found: queue the preview with a pose pass for that subject and `collect` it, name a "
                          f"`motion_file`, or give the load `motion: none`")
        files["motion"] = (str(path), one.get("motion_first", job["first"]))
        if one.get("motion_file_2"):
            # a second motion video is always named: nothing the preview saves is one
            if not Path(one["motion_file_2"]).is_file():
                raise Refused(f"load {one['name']!r}: its motion_file_2 {one['motion_file_2']} is not a file")
            files["motion_2"] = (str(one["motion_file_2"]), one.get("motion_first_2", job["first"]))
    return files


def hair_band(one: dict, files: dict, target: Path) -> tuple[dict, dict | None]:
    """A load's files with a band cut out of one keep-out, and what the band is; (files, None) for a load that asks for none.

    **What it is for.** A part pass of a head and hair regenerates the part and a margin round it, less what the
    keep-outs give back (`video_mask.window`: a cell a keep-out touches is the source's unless the subject's own
    mask has a pixel in it). The part ends at the original's jaw. A still whose hair falls further is drawn into
    the margin below, and there every cell is the clothing keep-out's to give back; that mask's pieces change
    frame to frame, so the region's lower edge moved and the hair was drawn where it reached and cut where it did
    not. A band under the part, taken out of that keep-out, leaves the margin open there with an edge that holds
    still.

    **The band, a frame.** The part mask's box on that frame, each side the median of the boxes
    `h3_config.MASKED_JOB_HAIR_BAND["smooth_frames"]` frames before and after it (frames with no part are left out
    of the median, and a frame with none near it has no band); from the box's bottom DOWN by `hair_band` times the
    box's height; as wide as the box and `wider` of its width more on each side; cut at the picture's edges.

    **What it does not do.** It adds nothing to the region: the margin still bounds what is regenerated, and hair
    longer than the margin needs a larger margin. Only the one keep-out is cut; the others give their cells back
    as before, inside the band too. It reads masks and decides nothing about a picture.

    `target` is the mask video the cut keep-out will be written to, by the returned band's `write`, which the
    caller runs once every load of the build has been tested: the build writes nothing until then. Its frame 0 is
    the load's first frame. Nothing here is kept between builds: a plan and a render each make their own."""
    if "hair_band" not in one:
        return files, None
    rule = h3_config.MASKED_JOB_HAIR_BAND
    share, cuts = float(one["hair_band"]), one.get("hair_band_cuts", rule["cuts"])
    where = list(one.get("others") or ()).index(cuts)
    (part, part_at), (keep, keep_at) = files["parts"], files["others"][where]
    first, frames, reach = one["first"], one["frames"], int(rule["smooth_frames"])
    w, h, _ = probe(keep)
    if first < part_at or first < keep_at:
        raise Refused(f"load {one['name']!r}: the hair band reads the part from source frame {part_at} and the keep-out "
                      f"{cuts!r} from {keep_at}, and the load begins on {first}")
    boxes = np.full((frames, 4), np.nan)
    for f, frame in enumerate(stream(part, (w, h), first - part_at, frames, vf=f"scale={w}:{h}:flags=neighbor")):
        on = frame > 127
        rows, cols = np.nonzero(on.any(1))[0], np.nonzero(on.any(0))[0]
        if len(rows):
            boxes[f] = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    seen = ~np.isnan(boxes[:, 0])
    if not seen.any():
        raise Refused(f"load {one['name']!r}: the part mask {part} is empty on every frame of the load: there is no head "
                      f"to put a hair band under")
    bands = np.full((frames, 4), -1, int)
    for f in range(frames):
        near = boxes[max(f - reach, 0):f + reach + 1]
        near = near[~np.isnan(near[:, 0])]
        if len(near):
            x0, y0, x1, y1 = np.median(near, axis=0)
            bands[f] = (max(int(np.floor(x0 - rule["wider"] * (x1 - x0))), 0), min(int(round(y1)), h),
                        min(int(np.ceil(x1 + rule["wider"] * (x1 - x0))), w), min(int(np.ceil(y1 + share * (y1 - y0))), h))
    has = bands[:, 0] >= 0
    figures = {"share_of_the_parts_height": share, "cut_out_of": cuts, "file": str(target), "its_first_frame_is_source_frame": first,
               "made_from": {"part": part, "its_first_frame_is_source_frame": part_at, "keep_out": keep, "keep_outs_first_frame_is_source_frame": keep_at},
               "mask_size": [w, h],
               "part_height_px_median": float(np.median(boxes[seen, 3] - boxes[seen, 1])),
               "band_px_median": {"deep": float(np.median(bands[has, 3] - bands[has, 1])), "wide": float(np.median(bands[has, 2] - bands[has, 0]))},
               "the_box_is_the_median_over_frames": 2 * reach + 1, "wider_each_side_by_share_of_width": rule["wider"],
               "frames_with_a_band": int(has.sum()), "frames_with_no_part_within_reach": [first + int(f) for f in np.nonzero(~has)[0]][:40],
               "frames_with_no_part_of_their_own": int((~seen).sum()),
               "what_it_does_not_do": "it adds nothing to the region: the margin still bounds what is regenerated, and the other keep-outs give their cells back inside the band too",
               "tested_on_a_render": "not when the band was built into this tool (2026-10-12): the first load that carries one is read by a session that did not build it"}

    def write() -> dict:
        """Write the cut keep-out: the keep-out's own frames of the load, with the band emptied. Returns what was taken out."""
        enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                                "-r", str(MASK_FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", str(target)], stdin=subprocess.PIPE)
        taken, wrote = [], 0
        for f, frame in enumerate(stream(keep, (w, h), first - keep_at, frames, vf=f"scale={w}:{h}:flags=neighbor")):
            on = frame > 127
            if has[f]:
                x0, y0, x1, y1 = bands[f]
                taken.append(int(on[y0:y1, x0:x1].sum()))
                on[y0:y1, x0:x1] = False
            enc.stdin.write((on.astype(np.uint8) * 255).tobytes())
            wrote += 1
        for _ in range(frames - wrote):                             # a keep-out that ends before the load does is empty from there, as the loader reads it
            enc.stdin.write(np.zeros((h, w), np.uint8).tobytes())
        enc.stdin.close()
        if enc.wait():
            raise Refused(f"ffmpeg could not write {target}")
        return {"keep_out_px_taken_out_a_frame": {"median": float(np.median(taken)) if taken else 0.0, "most": max(taken, default=0)},
                "keep_out_frames_read": wrote, "frames_written": frames}

    cut = dict(files, others=[(str(target), first) if i == where else entry for i, entry in enumerate(files["others"])])
    return cut, {"figures": figures, "write": write}


def band_lines(band: dict | None) -> list[str]:
    """What is shown of a load's hair band, one line."""
    if band is None:
        return []
    fig = band["figures"]
    return [f"hair band: the keep-out `{fig['cut_out_of']}` is cut by a band under the part, {fig['band_px_median']['deep']:.0f} px deep "
            f"({fig['share_of_the_parts_height']:g} of the part's height of {fig['part_height_px_median']:.0f} px) and "
            f"{fig['band_px_median']['wide']:.0f} px wide at the median, on {fig['frames_with_a_band']} of the load's frames, its box the median over "
            f"{fig['the_box_is_the_median_over_frames']} frames; the mask's own pixels ({fig['mask_size'][0]}x{fig['mask_size'][1]}); "
            f"written to {Path(fig['file']).name}. {fig['tested_on_a_render']}"]


def preview_mesh(folder: Path, job: dict, output: Path):
    """A finder for the mesh a job's preview saved for a subject, from the record `collect` wrote.

    The mesh is the file in that record whose name starts with this job's own prefix for the subject. That is
    one file and no other run's: a job's name is never reused, so nothing else was ever saved under the
    prefix, and the record lists only what this job's preview saved."""
    record = folder / "preview.queued.json"
    saved = json.loads(record.read_text()).get("files", []) if record.is_file() else []

    def find(label: str) -> str | None:
        stem = Path(graphs.prefixes(job, label)["mesh"]).name
        hits = [f for f in saved if Path(f).name.startswith(stem) and f.endswith(".mp4")]
        return str(output / hits[0].lstrip("/")) if hits else None
    return find


def input_dir() -> Path:
    """Where the server reads stills from: `H3_INPUT_DIR`, since this box starts ComfyUI with
    `--input-directory` on a share and nothing in the HTTP API reports it back; else the stock folder."""
    import os
    return Path(os.environ.get("H3_INPUT_DIR") or COMFY / "input")


def still_seen(name: str, shown_at: int, upscale: bool, inputs: Path) -> str:
    """A still's own size against the size the video model is shown it at, said plainly."""
    path = inputs / name
    if not path.is_file():
        raise Refused(f"the still {name!r} is not under {inputs}: name a file the server can load, or set "
                      f"H3_INPUT_DIR to the server's input folder")
    from PIL import Image
    with Image.open(path) as picture:
        width, height = picture.size
    short = min(width, height)
    if short < shown_at and upscale:
        how = f"ENLARGED by {shown_at / short:.2f} to a short edge of {shown_at} (enlarging adds no detail)"
    elif short > shown_at:
        how = f"reduced by {short / shown_at:.2f} to a short edge of {shown_at}"
    else:
        how = f"at its own size (the ceiling is {shown_at})"
    return f"the still {name} is {width}x{height}: the video model is shown it {how}"


def pair_of(one: dict, files: dict) -> list[dict] | None:
    """Which picture and which video went to which subject of a two-subject load, with each one's files; None for one subject."""
    also = one.get("also")
    if not also:
        return None
    first, second = one["subjects"]
    return [{"subject": one["subject"], "region": first["region"], "is": "<Subject 1>", "still": one.get("still"), "picture": "<Picture 1>",
             "reference": files["motion"][0] if files.get("motion") else None, "video": "<Video 1>" if files.get("motion") else None,
             "track": files["track"][0], "part": files["parts"][0] if files.get("parts") else None},
            {"subject": also["subject"], "region": second["region"], "is": "<Subject 2>", "still": also.get("still"), "picture": "<Picture 2>",
             "reference": files["motion_2"][0] if files.get("motion_2") else None, "video": "<Video 2>" if files.get("motion_2") else None,
             "track": files["also"]["track"][0], "part": files["also"]["parts"][0] if files["also"].get("parts") else None}]


def pair_lines(one: dict, files: dict, src: dict, text: str, ref: dict, inputs: Path) -> list[str]:
    """What is shown of a two-subject load beyond a one-subject load's lines; nothing for one subject."""
    pair = pair_of(one, files)
    if pair is None:
        return []
    lines = ["TWO SUBJECTS IN ONE PASS: " + "; ".join(
        f"{p['subject']} ({p['region']}) is {p['is']}, its still {p['picture']}" + (f", its reference {p['video']}" if p["video"] else ", no reference video")
        for p in pair)]
    second = pair[1]
    lines.append((still_seen(second["still"], ref["size_policy.dit_short_edge"], ref["size_policy.allow_upscale"], inputs)
                  if second["still"] else f"no still named for {second['subject']}: a plan samples nothing") + f" (the second subject's, {second['picture']})")
    wanted = [label for p in pair for label in (p["is"], p["picture"], p["video"]) if label]
    lines.append("the text names " + ", ".join(f"{label}: {'yes' if label in text else 'NO'}" for label in wanted))
    lines.append("which still is whose is told to the model by the TEXT ALONE (its two subject definitions and where each person is), and by each "
                 "reference being that person's own head: nothing ties a still to a place in the picture. A swap or a blend of the two faces is what "
                 "this load's first render is read for")
    lines.append(f"one margin for both subjects, {src['grow_pixels']} px (the node grows its region by one number), and the keep-outs are both "
                 f"subjects' together; the region is the union of the two parts")
    return lines


def shown(job: dict, one: dict, files: dict, graph: dict, inputs: Path) -> list[str]:
    """What each model will be shown for a load, one line a thing (the order's step 6), read from the graph."""
    src = graph[graphs._the(graph, graphs.SOURCE)]["inputs"]
    run = graph[graphs._the(graph, graphs.SONG)]["inputs"]
    ref = graph[graphs.stills_of(graph)[0][0]]["inputs"]              # the first still's; a second is shown as the first is
    lines = [f"frames {one['first']}..{one['first'] + one['frames'] - 1} of the copy at {h3_config.FPS:g} fps, canvas "
             f"{run['width']}x{run['height']}, window {run['window_frames']}, context {run['context_frames']}"
             + (f", its last frame held for {run['window_frames'] - one['frames']} frames with the region open"
                if one["frames"] < run["window_frames"] else ""),
             (still_seen(one["still"], ref["size_policy.dit_short_edge"], ref["size_policy.allow_upscale"], inputs)
              if one.get("still") else "no still named: a plan samples nothing")
             + "; the text encoder is shown "
             + (f"its own copy at a short edge of {ref['qwen_view.qwen_short_edge']}" if "qwen_view.qwen_short_edge" in ref
                else f"that same copy (`{ref.get('qwen_view')}`)"),
             f"region: {src['replace']}, margin {src['grow_pixels']} px, feather {src['feather_pixels']} px, edge "
             f"{src.get('edge', 'the node default')}, kept out of the margin: {', '.join(one.get('others') or ()) or 'nobody'}"]
    if files["motion"]:
        lines.append(f"motion video {files['motion'][0]} ({src['motion_reference']}) at a short edge of "
                     f"{src['motion_short_edge']}: every frame to the video model, one picture a second to the text encoder")
        if files.get("motion_2"):
            named = "<Video 2>" in str(run.get("prompt", ""))
            lines.append(f"second motion video {files['motion_2'][0]} at a short edge of {src['motion_short_edge_2']}, the next video "
                         f"reference after the first: the text has to say what each one is, and this load's text "
                         + ("names <Video 2>" if named else "DOES NOT NAME <Video 2>"))
    else:
        lines.append("no motion video")
    lines += pair_lines(one, files, src, str(run.get("prompt", "")), ref, inputs)
    lines.append(f"audio: {one['audio']} cut to the load" if one.get("audio") else "audio: the source's own, as the loader cuts it")
    lines.append(f"seed {run['seed']}"
                 + (f"; continues {run['continue_from']} with its kept frames at noise {run['context_noise']}"
                    if "continue_from" in run else ""))
    return lines


def settings_of(one: dict, graph: dict) -> dict:
    """Each setting of a load a job may set (`masked_job_graph.SETTINGS`): the value the graph carries and who
    set it. A default measured on another clip is a starting value; this is where a reader sees which stood."""
    src = graph[graphs._the(graph, graphs.SOURCE)]["inputs"]
    run = graph[graphs._the(graph, graphs.SONG)]["inputs"]
    used = {"carried": src["replace"], "margin": src["grow_pixels"], "feather": src["feather_pixels"],
            "edge": src.get("edge", "the node's own"), "window": run["window_frames"], "context": run["context_frames"],
            "context_noise": run.get("context_noise", "none: the load keeps no frames"),
            "motion": src["motion_reference"], "motion_edge": src["motion_short_edge"] if "motion_video" in src else "no motion video",
            "motion_edge_2": src["motion_short_edge_2"] if "motion_video_2" in src else "no second motion video",
            "hair_band": one.get("hair_band", "none: the keep-outs as the capture has them"),
            "render_region": src.get("render_region", "the node's own: the whole frame"),
            "seed": run["seed"]}
    return {key: {"value": used[key], "set_by": "the job file" if key in one else "the builder's default"}
            for key in graphs.SETTINGS}


def _middle(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def measured(job: dict, one: dict, capture: Path, graph: dict) -> dict:
    """What this clip's preview measured for a load, gathered from what the capture already holds: nothing is
    measured here. A rule that depends on the picture is set from these and not from a figure of another
    clip; a figure the capture does not hold says so, so that nobody reads its absence as a zero."""
    manifest = json.loads((capture / "manifest.json").read_text())
    first, last = one["first"], one["first"] + one["frames"] - 1
    by = one.get("by") or next(s["sightings"][0]["by"] for s in manifest["subjects"] if s["label"] == one["subject"])
    sighting = next(x for s in manifest["subjects"] if s["label"] == one["subject"] for x in s["sightings"] if x["by"] == by)
    folder = capture / "subjects" / one["subject"]
    out = {"from": str(capture), "gathered_not_measured_here": True}
    table = folder / f"shots__{by}.json"
    if table.is_file():
        at = sighting.get("shots_first_source_frame", manifest["first_frame"])
        cuts = [at + shot["first_frame"] for shot in json.loads(table.read_text()).get("shots", [])][1:]
        out["cuts_on_source_frames"] = cuts
        out["cuts_inside_this_load"] = [c for c in cuts if first < c <= last]
    else:
        out["cuts_on_source_frames"] = f"not in the capture: no shot table of {one['subject']!r} by {by!r}"
    rows = folder / "per_frame.json"
    if rows.is_file():
        table = json.loads(rows.read_text())
        mine = [r for r in table["rows"] if first <= r["source_frame"] <= last and r.get("seen_by", by) == by]
        boxes = [r["track_box"] for r in mine if r.get("track_box")]
        cell = manifest.get("cell_px")
        height = _middle([b[3] - b[1] for b in boxes])
        out["subject"] = {"frames_read": len(mine), "frames_with_no_mask": len(mine) - len(boxes),
                          "share_of_the_frame": {"least": min((r["track_share"] for r in mine), default=None),
                                                 "middle": _middle([r["track_share"] for r in mine]),
                                                 "most": max((r["track_share"] for r in mine), default=None)},
                          "box_height_px_middle": height,
                          "box_height_in_latent_cells_middle": round(height / cell, 1) if height and cell else None}
        parts = [r["parts_share"] for r in mine if r.get("parts_share") is not None]
        out["part"] = ({"share_of_the_frame_middle": _middle(parts),
                        "share_of_the_subject_middle": _middle([r.get("parts_of_track") for r in mine]),
                        "height": "not in the capture: its per-frame table has the part's share and no box of the part"}
                       if parts else "not in the capture: no part was saved for this subject")
    else:
        out["subject"] = "not in the capture: no per-frame table"
    window = graph[graphs._the(graph, graphs.SONG)]["inputs"]["window_frames"]
    out["load_against_its_window"] = {"frames": one["frames"], "window": window, "held_past_the_load": max(window - one["frames"], 0)}
    return out


def build_loads(job_file: Path, stage: str, names: list[str], *, date: str, data: Path, output: Path,
                inputs: Path, repo: Path = REPO, analysis=analysed, look=looked_at) -> list[Path]:
    """Write the plan or render graph of each chosen load into the job folder; returns the graph files. Every
    chosen load's text is held to its frames first (`texts_for`), and a plan records the look it is built on,
    or that none is recorded (`look`, through `look_or_shown`: shown, never refused)."""
    given = json.loads(job_file.read_text())
    if "job" not in given:
        raise Refused(f"{job_file} has no `job` block")
    job = given["job"]
    try:
        graphs.check_job(job)
    except ValueError as e:
        raise Refused(f"{job_file}: {e}") from e
    made_from = analysis(job, data)
    every = loads_of(given)
    stills = {s["label"]: s["still"] for s in job.get("subjects", []) if s.get("still")}
    for one in every:
        # a subject's still, given once in the job block, is every load's of that subject unless a load names its own
        if "still" not in one and one["subject"] in stills:
            one["still"] = stills[one["subject"]]
        if one.get("also") and "still" not in one["also"] and one["also"]["subject"] in stills:
            one["also"]["still"] = stills[one["also"]["subject"]]
    chosen = [one for one in every if not names or one["name"] in names]
    unknown = set(names) - {one["name"] for one in every}
    if unknown or not chosen:
        raise Refused(f"{job_file} has no load named {sorted(unknown)}" if unknown else f"{job_file} lists no loads")
    if not job.get("capture"):
        raise Refused("the job block names no `capture`: a load is built from a checked capture's files")
    capture = Path(job["capture"]) if Path(job["capture"]).is_absolute() else repo / job["capture"]
    folder = job_folder(job, date, data)
    on_purpose = left_out(job)
    base = json.loads((repo / "workflows" / h3_config.MASKED_JOB_BASE_GRAPH).read_text())
    windows, written, todo = chain_lengths(repo), [], []
    least_seconds, least_frames = shortest_load(repo)
    texts = texts_for(job, chosen, job_file.parent, capture)
    for one in chosen:
        if one["frames"] < least_frames:
            raise Refused(f"load {one.get('name')!r} is {one['frames']} frames: the song node takes no extent under "
                          f"{least_seconds:g} s, so a load is {least_frames} frames or more. Leave the stretch as the "
                          f"source (`left_out`, with why) or join it to a neighbouring load.")
        try:
            graphs.check_job(job)
            graphs.check_load(job, one, windows)
            told = texts[one["name"]]
            text = told["text"]
            files = capture_files(job, one, capture, preview_mesh(folder, job, output))
            files, band = hair_band(one, files, folder / f"{one['name']}.{stage}.{one.get('hair_band_cuts', h3_config.MASKED_JOB_HAIR_BAND['cuts'])}_less_hair_band.mkv")
            graph = graphs.load(base, job, one, files, text, windows=windows, plan=stage == "plan")
            says, text_lines = shown(job, one, files, graph, inputs) + band_lines(band), text_shown(one, told)
            seen = measured(job, one, capture, graph)
        except ValueError as e:
            raise Refused(f"{job_file}: {e}") from e
        # (A zoomed load that holds a cut was refused here until 2026-10-10. No load that holds a cut is built
        # now: its text would have to span the cut, and `texts_for` refuses that first. The node's own refusal
        # of a cut inside a zoomed load is still behind it, and the check holds a case that shows this.)
        saves_to = (output / graphs.run_prefix(job, one, stage == "plan")).parent
        graph_file = folder / f"{one['name']}.{stage}.json"
        for path, what in ((graph_file, "graph"), (saves_to, "output folder")):
            if path.exists():
                raise Refused(f"the {what} {path} exists. A run never reuses a name: give the load a new `name`.")
        todo.append((one, graph, graph_file, saves_to, files, told, says, text_lines, seen, band))
    looks = {one["name"]: look_or_shown(capture, one, texts[one["name"]], look)
             for one in (chosen if stage == "plan" else ())}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"job.{stage}.json").write_bytes(job_file.read_bytes())
    for one, graph, graph_file, saves_to, files, told, says, text_lines, seen, band in todo:
        graph_file.write_text(json.dumps(graph, indent=1) + "\n")
        cut = None if band is None else {**band["figures"], **band["write"]()}
        (folder / f"{one['name']}.{stage}.txt").write_text(told["text"] + "\n")
        record = {"stage": stage, "load": one, "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                  "commit": head(repo), "builder_not_committed": builder_not_committed(repo), "analysis": made_from,
                  "uncommitted_node_code": uncommitted_node_code(repo), "job_file": str(job_file),
                  "capture": str(capture), "files": files, "text": told, "look": looks.get(one["name"]),
                  "output_folder": str(saves_to),
                  "windows_folder": str(output / (graphs.run_prefix(job, one, stage == "plan") + "_windows")),
                  "shown": says + look_lines(looks.get(one["name"])) + text_lines, "settings": settings_of(one, graph),
                  "hair_band": cut, "subjects": pair_of(one, files),
                  "left_out_on_purpose": on_purpose,
                  "what_this_clips_preview_measured": seen}
        graph_file.with_suffix(".built.json").write_text(json.dumps(record, indent=1) + "\n")
        written.append(graph_file)
    return written


def plan_line(built: dict) -> str:
    """The `--run` spec the capture reads a load's own plan with, once the plan has run.

    `at=` is the load's first source frame: a node's plan counts from its load's frame 0, and without it the
    capture would place the load on the capture's first frame."""
    one = built["load"]
    return (f"--run {one['name']}:preview={built['windows_folder']},subject={one['subject']}"
            + (f",others={'+'.join(one['others'])}" if one.get("others") else "") + f",at={one['first']}")


def run_verify(capture: Path) -> int:
    """`bench/capture_masked_run.py verify` on a capture: its exit status."""
    return subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "capture_masked_run.py"),
                           "verify", str(capture)], capture_output=True, text=True).returncode


def gate(built: dict, folder: Path, verify=run_verify) -> dict:
    """The last thing before a render is queued: the capture still reads what the nodes wrote, its gate is
    clear, and what the gate read for this load is the node's own plan. Returns what it found, for the record."""
    capture, name = Path(built["capture"]), built["load"]["name"]
    status = verify(capture)
    if status != 0:
        raise Refused(f"`verify` on the capture {capture.name} exited {status}: it no longer reads what the nodes wrote")
    if not (capture / "flags.json").is_file():
        raise Refused(f"the capture {capture.name} has no flags.json: run `preflight {capture} --gate` on the node's plan")
    flags = json.loads((capture / "flags.json").read_text())
    if flags.get("verdict") != "clear":
        raise Refused(f"the gate of {capture.name} is {flags.get('verdict')!r}, on {', '.join(flags.get('blocking') or []) or 'nothing it names'}. "
                      f"Fix what a flag names, or answer it in the capture: outcome {capture} <flag> overridden --by WHO --note WHY")
    planned = folder / f"{name}.plan.queued.json"
    if not planned.is_file():
        raise Refused(f"the plan of load {name!r} was never queued: `queue {folder} plan --load {name} --wait`, then the "
                      f"capture reads it (--run {name}:preview=<its windows folder>,subject=..) and the gate is run on that")
    runs = {r["name"]: r for r in json.loads((capture / "manifest.json").read_text()).get("runs", [])}
    if name not in runs or not runs[name].get(READ_FROM_THE_NODE):
        raise Refused(f"the capture {capture.name} "
                      + (f"worked the region of {name!r} out from the masks" if name in runs else f"holds no run named {name!r}")
                      + f": the gate has not read the node's own plan for this load (--run {name}:preview=...)")
    return {"verify": status, "verdict": flags["verdict"], "overridden": flags.get("overridden") or [],
            "plan_read": runs[name][READ_FROM_THE_NODE]}


def build(job_file: Path, stage: str, *, date: str, data: Path, output: Path, repo: Path = REPO,
          analysis=analysed) -> Path:
    """Write a stage's graph and its record into a new job folder; returns the folder."""
    given = json.loads(job_file.read_text())
    if "job" not in given:
        raise Refused(f"{job_file} has no `job` block")
    job = given["job"]
    try:
        graphs.check_job(job)
        graph = {"preview": graphs.preview}[stage](job)
    except ValueError as e:
        raise Refused(f"{job_file}: {e}") from e
    version = tracker_mask_version(repo)
    check_corrections(job, version)
    made_from = analysis(job, data)
    folder, saves_to = job_folder(job, date, data), output_folder(job, output)
    for path, what in ((folder, "job folder"), (saves_to, "output folder")):
        if path.exists():
            raise Refused(f"the {what} {path} exists. A run never reuses a name: give the job a new `name`.")
    folder.mkdir(parents=True)
    (folder / "job.json").write_bytes(job_file.read_bytes())
    (folder / f"{stage}.json").write_text(json.dumps(graph, indent=1) + "\n")
    record = {"stage": stage, "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "commit": head(repo), "builder_not_committed": builder_not_committed(repo), "analysis": made_from,
              "uncommitted_node_code": uncommitted_node_code(repo),
              "tracker_mask_version": version, "job_file": str(job_file), "output_folder": str(saves_to),
              "output": str(output), "expects": expected_files(job)}
    (folder / f"{stage}.built.json").write_text(json.dumps(record, indent=1) + "\n")
    return folder


#: The file each thing a preview saves ends up as, after its prefix: a save node numbers its file, the shot
#: table and the pose table name theirs. A history lists only some of these, so they are looked for on disk.
SAVED_AS = {"track": "_*.mkv", "parts": "_*.mkv", "classes": "_*.mkv", "held_things": "_*.mkv", "shots": "_*_shots.json",
            "mesh": "_*.mp4", "pose": "_pose_table.json"}


def expected_files(job: dict) -> dict:
    """What a job's preview must leave on disk, as patterns under the output folder: per subject, per thing."""
    out = {}
    for subject in job["subjects"]:
        prefix = graphs.prefixes(job, subject["label"])
        wanted = [k for k in SAVED_AS if subject.get("pose", True) or k not in ("mesh", "pose")]
        out[subject["label"]] = {k: prefix[k] + SAVED_AS[k] for k in wanted}
    return out


def found_on_disk(expects: dict, output: Path) -> tuple[dict, list[str]]:
    """The files that are there for each thing a run was to save, and the things with none."""
    found, missing = {}, []
    for label, things in expects.items():
        found[label] = {}
        for thing, pattern in things.items():
            found[label][thing] = sorted(str(p.relative_to(output)) for p in output.glob(pattern))
            if not found[label][thing]:
                missing.append(f"{label}: {thing} ({pattern})")
    return found, missing


def subjects_not_found(found: dict, output: Path) -> list[str]:
    """The subjects a preview's trackers never had: a shot table with no pick, or with no frame of the subject
    on any shot. Everything after a preview assumes each subject was found somewhere (the who-is-who tool
    needs a look of them, the capture a mask), so this is said by name and is not a success."""
    out = []
    for label, things in found.items():
        for name in things.get("shots") or []:
            table = json.loads((output / name).read_text())
            shots = table.get("shots") or []
            if table.get("pick_frame") is None or not any(shot.get("frames_with_subject") for shot in shots):
                out.append(f"{label}: " + ("no pick was made" if table.get("pick_frame") is None else
                                           f"picked on frame {table['pick_frame']}") + f" and the subject is on no "
                           f"frame of the load's {len(shots)} shot(s). An earlier tracker may hold this person on the "
                           f"frame they were to be picked on: put this subject earlier in `subjects`, or pick them on "
                           f"a shot where they are the only person.")
    return out


def capture_from(job: dict, found: dict, output: Path, date: str) -> str:
    """The capture command for a preview that has run, with the files it really saved: nothing left to fill in.

    One file of each kind for each subject, or the command is not written: two tracks under one prefix are
    two runs in one folder, which is what a new name for every run is there to prevent."""
    lines = [f"bench/capture_masked_run.py files --name {job['name']} --date {date} --source {job['source']} "
             f"--first {job['first']} --frames {job['frames']}"]
    for subject in job["subjects"]:
        mine, spec = found.get(subject["label"]) or {}, []
        for key in ("track", "parts", "classes", "shots", "pose"):
            if key == "pose" and not subject.get("pose", True):
                continue
            if len(mine.get(key) or []) != 1:
                raise Refused(f"{subject['label']}: {len(mine.get(key) or [])} files for `{key}` where one is meant: "
                              f"{mine.get(key) or 'none'}")
            spec.append(f"{key}={output / mine[key][0]}")
        lines.append(f"    --mask {subject['label']}:" + ",".join(spec) + f",by={job['name']},at={job['first']}")
    return " \\\n".join(lines)


def capture_command(job: dict, date: str) -> str:
    """The capture the preview's files go into, with a name left for each file the server numbers.

    The paths are under the server's output folder. What a subject holds (`held_things`) is not in it: the
    capture takes that mask as a subject of its own when a job needs it, never as a part."""
    lines = [f"bench/capture_masked_run.py files --name {job['name']} --date {date} --source {job['source']} "
             f"--first {job['first']} --frames {job['frames']}"]
    for subject in job["subjects"]:
        p = graphs.prefixes(job, subject["label"])
        pose = f"pose=<{p['pose']}_pose_table.json>," if subject.get("pose", True) else ""
        lines.append(f"    --mask {subject['label']}:track=<{p['track']}_*.mkv>,parts=<{p['parts']}_*.mkv>,"
                     f"classes=<{p['classes']}_*.mkv>,shots=<{p['shots']}*.json>,{pose}by={job['name']},"
                     f"at={job['first']}")
    return " \\\n".join(lines)


def call(host: str, path: str, data: dict | None = None) -> dict:
    request = urllib.request.Request(f"http://{host}{path}", data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise Refused(f"the server answered {e.code} to {path}: {e.read().decode()[:4000]}") from e
    except urllib.error.URLError as e:
        raise Refused(f"no server at {host}: {e.reason}") from e


def stem_of(stage: str, load: str = "") -> str:
    """The name a stage's files carry in the job folder: `preview`, or `<load>.<stage>`."""
    if stage == "preview":
        return stage
    if not load:
        raise Refused(f"`{stage}` is queued one load at a time: say which with --load")
    return f"{load}.{stage}"


def before_queue(folder: Path, stage: str, served: dict, *, load: str = "", override: str = "", repo: Path = REPO,
                 verify=run_verify) -> tuple[dict, dict]:
    """Every refusal of `queue`, with nothing posted; returns the graph and the start of its record."""
    stem = stem_of(stage, load)
    built_file, graph_file = folder / f"{stem}.built.json", folder / f"{stem}.json"
    if not built_file.exists() or not graph_file.exists():
        raise Refused(f"{folder} holds no built `{stem}`; run `build {stage}` first")
    if (folder / f"{stem}.queued.json").exists():
        raise Refused(f"`{stem}` of {folder.name} was queued before. A run never reuses a name: build it "
                      f"again under a new `name`.")
    built, graph = json.loads(built_file.read_text()), json.loads(graph_file.read_text())
    if Path(built["output_folder"]).exists():
        raise Refused(f"the output folder {built['output_folder']} exists; something has written there since the build")
    dirty = uncommitted_node_code(repo)
    if dirty and ":" not in override:
        raise Refused("node code in the tree is not committed: " + ", ".join(dirty) + ". The server serves a "
                      'commit. Commit it, or say who queues beside it and why: --override "WHO: WHY".')
    import build_workflows
    errors = build_workflows.validate_api(graph, served, f"{folder.name}/{stage}")
    if errors:
        raise Refused("the running server does not accept this graph:\n  " + "\n  ".join(errors))
    record = {"stage": stage, "commit": head(repo), "uncommitted_node_code": dirty, "override": override}
    if stage == "render":
        record["look"] = text_and_look(built, graph)
        record["gate"] = gate(built, folder, verify)
    return graph, record


def queue(folder: Path, stage: str, *, host: str, load: str = "", override: str = "", wait: bool = False) -> int:
    graph, record = before_queue(folder, stage, call(host, "/object_info"), load=load, override=override)
    for line in look_lines(record.get("look")):
        print(line)
    answer = call(host, "/prompt", {"prompt": graph, "client_id": "masked_job"})
    record |= {"prompt_id": answer.get("prompt_id"), "host": host,
               "queued_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    queued = folder / f"{stem_of(stage, load)}.queued.json"
    queued.write_text(json.dumps(record, indent=1) + "\n")
    print(f"queued {record['prompt_id']}  ({queued})")
    if not wait or not record["prompt_id"]:
        return 0
    while record["prompt_id"] not in call(host, f"/history/{record['prompt_id']}"):
        time.sleep(3)
    return collect(folder, stage, host=host, load=load)


def read_history(entry: dict) -> dict:
    """What a finished run left in the server's history, as the record keeps it: its status, how long the
    server says it took, every file it saved under the output folder, and what each text preview said."""
    status = entry.get("status", {})
    out = {"status": status.get("status_str"), "files": [], "said": {}}
    times = [detail.get("timestamp") for kind, detail in status.get("messages", [])
             if kind in ("execution_start", "execution_success", "execution_error", "execution_interrupted")]
    if len(times) > 1 and all(isinstance(t, (int, float)) for t in times):
        out["seconds"] = round((times[-1] - times[0]) / 1000)
    for kind, detail in status.get("messages", []):
        if kind in ("execution_error", "execution_interrupted"):
            out["error"] = {k: str(detail.get(k))[:3000] for k in ("node_type", "exception_type", "exception_message")}
    for nid, saved in entry.get("outputs", {}).items():
        for value in saved.values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict) and "filename" in item:
                    out["files"].append(f"{item.get('subfolder', '')}/{item['filename']}".lstrip("/"))
                elif isinstance(item, str):
                    out["said"].setdefault(nid, []).append(item)
    out["files"].sort()
    return out


def collect(folder: Path, stage: str, *, host: str, load: str = "") -> int:
    """Fill a queued run's record from the server's history: for a run queued without `--wait`, or by a peer.

    It posts nothing. A run that is not in the history yet is refused, not recorded as empty: a record with
    no files would read as a run that saved none. For a preview it then looks on disk for every file the
    graph was built to save (a history does not list a shot table's json or a pose table) and exits 1 when
    one is not there, whatever the server's status says."""
    queued = folder / f"{stem_of(stage, load)}.queued.json"
    if not queued.is_file():
        raise Refused(f"{queued.name} is not in {folder}: nothing was queued under that name")
    record = json.loads(queued.read_text())
    history = call(host, f"/history/{record['prompt_id']}")
    if record["prompt_id"] not in history:
        raise Refused(f"the server's history does not hold {record['prompt_id']}: the run has not finished, or the "
                      f"server was restarted since and no longer remembers it")
    record |= read_history(history[record["prompt_id"]])
    built = json.loads((folder / f"{stem_of(stage, load)}.built.json").read_text())
    if stage == "preview" and not built.get("expects") and (folder / "job.json").is_file():
        # a preview built before its record listed what must be on disk: the job file says the same thing,
        # and the output folder's root is the saved folder less the job's own prefix and name
        job = json.loads((folder / "job.json").read_text())["job"]
        built["expects"] = expected_files(job)
        built["output"] = str(Path(built["output_folder"]).parents[len(Path(job["output"]).parts)])
    if built.get("expects"):
        record["found"], record["missing"] = found_on_disk(built["expects"], Path(built["output"]))
        record["subjects_not_found"] = subjects_not_found(record["found"], Path(built["output"]))
    queued.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['status']}" + (f" in {record['seconds']} s" if "seconds" in record else "")
          + f"; {len(record['files'])} files; the record is {queued}")
    if "error" in record:
        print(json.dumps(record["error"], indent=1))
    if record.get("subjects_not_found"):
        print("A SUBJECT WAS NOT FOUND IN THIS LOAD:\n  " + "\n  ".join(record["subjects_not_found"]))
    if record.get("missing"):
        print("NOT ON DISK, though the graph was built to save them:\n  " + "\n  ".join(record["missing"]))
    elif record.get("found") and (folder / "job.json").is_file():
        job = json.loads((folder / "job.json").read_text())["job"]
        print("the capture, with the files this run saved (give `--folder PATH` in place of --name and --date to "
              "say where it goes; add --voice, and --who-is-who and --not-in at preflight):\n"
              + capture_from(job, record["found"], Path(built["output"]), built["built_at"][:10]))
    return 0 if record["status"] == "success" and not record.get("missing") and not record.get("subjects_not_found") else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write a stage's graph into a new job folder; no server, no card")
    b.add_argument("stage", choices=STAGES)
    b.add_argument("job_file", type=Path)
    b.add_argument("--date", default=datetime.date.today().isoformat(), help="the date in the job folder's name")
    b.add_argument("--load", action="append", default=[], metavar="NAME", help="plan or render: only this load; repeatable")
    q = sub.add_parser("queue", help="post a built stage to the server: the one step that uses the card")
    q.add_argument("folder", type=Path)
    q.add_argument("stage", choices=STAGES)
    q.add_argument("--load", default="", metavar="NAME", help="plan or render: the load to queue")
    q.add_argument("--host", default=HOST)
    q.add_argument("--wait", action="store_true", help="stay until the server is done and record what it saved")
    q.add_argument("--override", default="", metavar='"WHO: WHY"',
                   help="queue beside uncommitted node code; written into the record")
    c = sub.add_parser("collect", help="record what a queued run saved, from the server's history; posts nothing")
    c.add_argument("folder", type=Path)
    c.add_argument("stage", choices=STAGES)
    c.add_argument("--load", default="", metavar="NAME", help="plan or render: the load")
    c.add_argument("--host", default=HOST)
    a = p.parse_args()
    try:
        if a.command == "build" and a.stage != "preview":
            for graph_file in build_loads(a.job_file, a.stage, a.load, date=a.date, data=REPO / "data",
                                          output=h3_config.output_dir(), inputs=input_dir()):
                built = json.loads(graph_file.with_suffix(".built.json").read_text())
                print(f"{graph_file}\n  built at {built['commit']}; it will save under {built['output_folder']}")
                for line in built["shown"]:
                    print(f"    {line}")
                from_job = [k for k, v in built["settings"].items() if v["set_by"] == "the job file"]
                print(f"    set by the job file: {', '.join(from_job) or 'nothing'}; the rest are the builder's defaults")
                m = built["what_this_clips_preview_measured"]
                if isinstance(m.get("subject"), dict):
                    print(f"    this clip, from the capture: the subject is {m['subject']['share_of_the_frame']['middle']} of the frame "
                          f"(middle of {m['subject']['frames_read']} frames), {m['subject']['frames_with_no_mask']} frames with no "
                          f"mask; cuts inside the load: {m.get('cuts_inside_this_load')}")
                if a.stage == "plan":
                    print("  when it has run, the capture reads it: " + plan_line(built))
            return 0
        if a.command == "build":
            folder = build(a.job_file, a.stage, date=a.date, data=REPO / "data", output=h3_config.output_dir())
            built = json.loads((folder / f"{a.stage}.built.json").read_text())
            print(f"{folder / (a.stage + '.json')}\n  built at {built['commit']}, tracker mask version "
                  f"{built['tracker_mask_version']}; it will save under {built['output_folder']}")
            if built["builder_not_committed"]:
                print("  the builder itself is not as that commit has it: " + ", ".join(built["builder_not_committed"]))
            if built["uncommitted_node_code"]:
                print("  node code not committed (queue will ask for an override): "
                      + ", ".join(built["uncommitted_node_code"]))
            subjects = json.loads(a.job_file.read_text())["job"]["subjects"]
            print("  trackers, in this order: " + "; ".join(
                f"{s['label']} ({s['pick']}" + (f" on source frame {s['pick_at']}" if s.get("pick_at") is not None else ", the tracker's own frame")
                + (f", match at {s['match']} set by the job" if s.get("match") is not None else ", match by the tracker's own line") + ")"
                for s in subjects))
            for s in subjects:
                if str(s.get("corrections", "")).strip():
                    print(f"  {s['label']}: corrections typed, read at the tracker's mask version {s['corrections_at']}:\n      "
                          + str(s["corrections"]).strip().replace("\n", "\n      "))
            print("when it has run, the capture:\n" + capture_command(json.loads(a.job_file.read_text())["job"], a.date))
            return 0
        if a.command == "collect":
            return collect(a.folder, a.stage, host=a.host, load=a.load)
        return queue(a.folder, a.stage, host=a.host, load=a.load, override=a.override, wait=a.wait)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    sys.exit(main())
