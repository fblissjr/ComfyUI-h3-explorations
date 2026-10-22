#!/usr/bin/env python3
"""A rolling history of every render of a video, one row each: how it was made and what the owner said of it.

    <python> bench/render_history.py --video fun --data <the tree's data/> --jsonl OUT.jsonl --csv OUT.csv \\
        [--share OUTPUT_ROOT] [--ledger OLD_LEDGER.json ...] [--carry PREVIOUS.jsonl] [--loads REGEX] [--source COPY.mov[=RATE] ...]

`--data` is the untracked `data/` of the tree whose records are read; the paths the job files hold (`data/<day>_<load>`) are taken from its parent.

**What it buys.** On 2026-10-22 an inventory of the loads of two shots (415-437 and 4912-4946) took a read-only investigator a long run
of tool calls (its transcript is the count), because each fact lives in a different file: the prompt in the queued graph, the padding in a prose line of the
build printout, the commit in the queue record, the reference's drawing form in a record on the share, the verdict in
one of two answers files, whether a load was laid in a table, and for the three renders older than the job tool
only a ledger made while their records still existed. This joins them once, a row a render, and says for each field
which file it came from and, where nothing says, that it cannot be seen. The record is the JSONL; the CSV is made
from it. Both are made whole on every run and never edited.

**Beside the tools near it.** It is a new tool that imports `bench/render_ledger.py`, not a flag on it, because the
ledger's JSON is schema 2 and `delivery_table.py`, `shot_coverage.py`, `shot_review_clips.py` and `review_pages.py` read
it; a field added there is read by all of them, and the ledger answers "which render may stand on which frames", not
"how was each one made". What the ledger already does is used as it is: `walk` and `NUMBERED` (which files are renders),
`saved_graph` and `asked` (the graph ComfyUI saved beside a file), `load_records` and `records_of` (the job tool's
records, tied to a file as the ledger ties them), `copies_of` (the video's copies and their rates), `reference_form`,
`part_classes`, `region_of` and `owner_lines` (the owner's lines by a render's row, file or page item). What it adds is
what no row of the ledger carries: the exact prompt, the padding and its kind, the seed, steps and sampler, the zoom, the
reference's drawing form and the span words, the preview's and the reference's commit, the load's status when it never
rendered, and whether and when it was laid. `bench/what_h3_got.py` reads a load's inputs, digests and zoom for one load
at a time and refuses without its job folder; the prompt here is read the way it reads it (the song node's `prompt` in
the queued graph), and `check_render_history.py` holds the two to the same text.
`bench/render_inventory.py` is another thing (benchmark rows of `run_graph_arms.py`, by model file).

**Rolling, without a key.** Each run reads the files as they are and writes the whole record. A job folder is deleted from `data/` now and then (the
renders of 2026-10-12 lost theirs), and the row of a load whose records are gone would shrink to what its picture says. Two named files keep what a
run would lose, and nothing finds them by a name: `--ledger` (an earlier render ledger, `bench/render_ledger.py --json`, whose record of a render fills the
fields that record held) and `--carry` (an earlier run's JSONL: a load whose job folder is gone takes its earlier row back, with the owner's lines and the
laid facts as they are now, and a load nothing of is left is carried as it was; each such row says `carried_from`).

**One row a render, three kinds.**

- `chain load`: a job folder `data/<day>_<load>_job/` (a preview's folder, with `preview.json`, is not a load). Its
  files tie it to a render by `rl.records_of`. A load that never rendered has a row with no output file and the step it
  reached; a load whose output file was never written says so.
- `job file only`: a `data/*/<load>.json` that is a job file (`job` and `loads` keys, `job.name` the file's stem) with no job folder: written and
  not run.
- `share render, no job record`: a numbered video under the share's `Video/` whose saved graph loads a copy of this
  video and whose load no job folder holds. Its fields come from the graph and the picture beside it. A record kept by an earlier ledger
  handed in with `--ledger` (the job folder gone from `data/`) fills the fields that record held, named as such.
  A re-laid piece (`recomposite_piece.py`: a composite setting changed on an existing render) is not an H3 render and is
  counted apart.

**Where each field is read, first source that has it wins, and the row's `sources` says which.**

| field | sources, in order |
|---|---|
| load, subject, span, frames given | the build record's `load`; the job file's `loads[0]` and `job`; the saved graph's loader; a handed-in ledger's record; the plan's `load_prefixes` (subject) |
| shot | `shots.json`: the shot holding the load's first frame |
| time | the queue record's `queued_at`; the picture's `CreationTime` (no zone: read as this machine's local time); the output's mtime |
| status | the queue record's `status` and `files`; else the furthest chain step with a file (`STEP_FILES`) and the last line of its log; a `<load>.STOPPED*` note, whole |
| window, padding | `load_against_its_window` in the build record (`context_before`, `context_after`, `held_past_the_load`); else the build printout's line; else the graph's loader against the load's frames |
| seed, steps, sampler, context, text | the queued graph `<load>.render.json`, else the picture's saved graph: the song node, `BasicScheduler`, `KSamplerSelect` |
| zoom | `<stem>_zoomed.json` beside the render; else the window's `_region.npz` `meta.zoom` |
| reference and its drawing form | the job's `reference` (form, iris, mouth, span words) and the reference's own record beside its video (`drawn`, `passed`, `writer_stamp`) |
| part, margin, keep-outs, span words | the job file; the capture's manifest for the part's classes; the build record's `settings` and `load.others`; the graph's `grow_pixels` |
| commits | the queue record (render); the preview's `preview.queued.json` (preview); the reference record's `writer_stamp` |
| laid | `laid.jsonl` and `laid.removed.jsonl` by the render's load name or file |
| the owner's lines | `verdicts.jsonl` and `owner_answers.jsonl` by the render's row (the load name), file or page item |

**What it cannot see**, in every row (`cannot_see` lists the fields a row has no value for and where they were looked for):
whether a render is good; which source frames a file holds when its count differs from its record; the per-frame form of
each eye (the reference's record has spans, not a form a frame: it is a function of the points and `draw_face_points.py`);
the owner's words said of a joined clip rather than of the render (`render_ledger.on_a_joined_item`); words the owner
said only in a lead's terminal; how a window's frames past the load were filled for a render older than 2026-10-21 with no
record of it (the graph asks for fewer frames than the window); a record deleted from `data/` that no handed-in ledger kept.
The mp4 itself carries no prompt or graph (checked 2026-10-22: `ffprobe -show_format` on a sample of numbered videos, with a picture and without, gave the
container's brand tags and nothing else, and `exiftool` on one render gave container fields only): the picture beside it (`<stem>.png`, its
`prompt` text chunk, the API graph, and `CreationTime`) does, and a numbered video with neither is left out and counted.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import render_ledger as rl  # noqa: E402
import what_h3_got as w  # noqa: E402

SCHEMA = "h3_render_history/1"
#: The kinds of row. **Reasoned**: one per way a render's record can stand, a fact about which files exist for the load, not a grade.
KINDS = ("chain load", "job file only", "share render, no job record")
#: A job file written by `bench/shot_to_load.py job` is `data/<folder>/<load>.json`; a load's name begins with one of the plan's
#: `load_prefixes` and holds a first frame. This pattern is the fallback when there is no plan. **Inherited** from the lane's load names
#: (`second_hfp_0415j`, `lead_fp_1097b__irisfill`).
JOB_FILE_NAME = re.compile(r"^(?:lead|second|third|face|street|mom)_[A-Za-z0-9_]*\d{3,4}[A-Za-z0-9_]*$")
#: Files a load's folder holds, by the load's name before the first dot. **Inherited** from `bench/load_chain.py` (the names it writes).
LOAD_FILE = re.compile(r"^(?P<load>[^.]+)\.(?:plan|render|who_is_who)\b")
#: The chain's steps in order (`bench/load_chain.py::STEPS`) and the file that says a step began: the first match in the order
#: names the furthest step reached. **Inherited**: each file is one `bench/load_chain.py` writes (`chain/<load>.<label>.out`, the `.plan.*` and `.render.*` records).
STEP_FILES = (("first", ("chain/{l}.files_first.out", "chain/{l}.gate_first.out", "{l}.who_is_who.json")),
              ("plan", ("{l}.plan.json", "{l}.plan.queued.json")),
              ("on-plan", ("{l}.plan.masks.json", "chain/{l}.files_on_plan.out")),
              ("table", ("chain/{l}.frame_table.out",)),
              ("render", ("{l}.render.json", "{l}.render.built.json", "{l}.render.queued.json")))
#: What padding a load had. **Inherited** from `bench/masked_job.py`'s `load_against_its_window` (real context frames, or `held_past_the_load`, the last frame
#: repeated) with the two this tool adds for a render that has no such key: `none` (loader frames equal the window) and `not recorded` (the window is longer than
#: the loader's frames and no record says how the rest was filled).
PADDING_KINDS = ("real source frames", "held copies of the last frame", "none", "not recorded")
#: The fields a row can lack, and where the tool looks for each (the names `sources` uses). A field in `AFTER_RENDER` is not
#: asked of a load that never rendered. **Reasoned**: the list is what the join reads; a field not here is not reported missing.
LOOKED_IN = {
    "subject": ("built.json", "job file", "graph", "ledger", "plan load_prefixes"),
    "span_first": ("built.json", "job file", "graph", "ledger"),
    "shot_first": ("shots.json",),
    "source_video": ("job file", "graph"),
    "window_length": ("built.json", "graph", "ledger"),
    "padding_before": ("built.json load_against_its_window", "build printout", "graph loader against the load"),
    "seed": ("built.json", "graph", "ledger"),
    "steps": ("graph",),
    "sampler": ("graph",),
    "prompt": ("graph render.json", "graph in the picture beside the mp4"),
    "reference_video": ("built.json", "job file", "graph"),
    "reference_form": ("job file reference", "the reference's record beside its video"),
    "still": ("built.json", "job file", "graph"),
    "part": ("job file", "capture manifest", "graph"),
    "margin": ("built.json settings", "graph"),
    "keep_outs": ("built.json load.others", "graph"),
    "span_words": ("job file",),
    "commit": ("queued.json", "built.json", "ledger"),
    "preview_commit": ("the preview's preview.queued.json",),
    "gate": ("queued.json", "ledger"),
    "rendered_at": ("queued.json", "picture CreationTime", "output mtime", "ledger"),
    "zoom": ("<stem>_zoomed.json", "windows region.npz"),
    "windows": ("queued.json windows_files", "the windows folder"),
    "output_frames": ("ffprobe",)}
AFTER_RENDER = {"rendered_at", "zoom", "windows", "output_frames", "padding_before", "gate", "commit", "window_length", "seed", "steps", "sampler"}
#: Columns of the CSV, in order. Each is a scalar or a short rendering of a list; the JSONL has all of it.
CSV_COLUMNS = ("load", "kind", "subject", "shot_first", "shot_last", "span_first", "span_last", "frames_given", "rendered_at", "rendered_at_from",
               "status", "stopped_after", "output_file", "output_frames", "session", "source_video", "window_length", "padding_before", "padding_after",
               "padding_kind", "windows", "context_frames", "seed", "steps", "sampler", "scheduler", "render_region", "zoom_scale", "zoom_box",
               "reference_video", "iris", "mouth", "closed_lip_line", "eyes_shut_by_word", "eyes_shut_by_rule", "still", "parts", "region", "margin",
               "feather", "keep_outs", "span_words", "preview", "preview_commit", "commit", "reference_commit", "gate", "laid", "owner_choices",
               "owner_words", "prompt_sha256", "prompt", "cannot_see")


def read_json(path: Path | None):
    """A JSON file as data, or None when there is none or it does not read."""
    try:
        return json.loads(Path(path).read_text()) if path is not None and Path(path).is_file() else None
    except (OSError, ValueError):
        return None


def jsonl(path: Path | None) -> list[tuple[int, dict]]:
    """A JSONL file as (line number, object), or none when the file is not there."""
    if path is None or not Path(path).is_file():
        return []
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append((n, json.loads(line)))
            except ValueError:
                pass
    return out


def utc_of(text: str | None) -> str | None:
    """A time as UTC to the second; one with no zone is read as this machine's local time."""
    if not text:
        return None
    try:
        t = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def shown(path: str | Path | None, *roots: Path) -> str | None:
    """A path relative to the first root it is under, else as given: the record names nothing of this machine's home."""
    if path is None:
        return None
    p = Path(str(path))
    for root in roots:
        try:
            return str(p.relative_to(root))
        except ValueError:
            continue
    return str(p)


def spans(text) -> list[list[int]]:
    """`311-319+349-360` (a job file's way of writing frames) as [[311, 319], [349, 360]]."""
    out = []
    for part in str(text or "").split("+"):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", part)
        if m:
            out.append([int(m.group(1)), int(m.group(2))])
    return out


def card_spans(items) -> list[list[int]]:
    """[{"first": a, "last": b}, ...] as [[a, b], ...]."""
    return [[int(x["first"]), int(x["last"])] for x in items or [] if isinstance(x, dict) and "first" in x and "last" in x]


def span_words(job: dict | None) -> dict | None:
    """The span words a job file carries, as frames: what `bench/shot_to_load.py job` was told with `--drop`, `--hold-part`
    (`word`), `--parts-on`, `--track-not`, `--no-face-on`, `--no-mouth-on`, `--mouth-on`, `--eyes-shut-on` and `--eyes-shut-form`, and the
    frames a look cut off the load. None when there is no job file; `{}` when it carries none. The words' reasons stay in the job file."""
    if not job:
        return None
    out: dict = {}
    for s in job.get("subjects") or []:
        for key, name in (("drop", "drop"), ("word", "hold_part"), ("hold", "hold"), ("least", "least")):
            if s.get(key) not in (None, "", []):
                out[name] = spans(s[key]) if key != "least" else s[key]
        if s.get("parts_on"):
            out["parts_on"] = [[x["first"], x["last"], x.get("parts")] for x in s["parts_on"] if isinstance(x, dict)]
    ref = job.get("reference") or {}
    if card_spans(ref.get("no_face_on")):
        out["no_face_on"] = card_spans(ref["no_face_on"])
    mouth = ref.get("mouth") if isinstance(ref.get("mouth"), dict) else {}
    for key in ("no_mouth_on", "mouth_on"):
        if card_spans(mouth.get(key)):
            out[key] = card_spans(mouth[key])
    iris = ref.get("iris") if isinstance(ref.get("iris"), dict) else {}
    shut = iris.get("eyes_shut")
    if isinstance(shut, dict):
        out["eyes_shut"] = {"form": shut.get("form"), "spans": card_spans(shut.get("spans"))}
    tn = job.get("track_not")
    if isinstance(tn, dict):
        out["track_not"] = [{"subject": m.get("subject"), "frames": m.get("frames")} for m in tn.get("made") or [] if isinstance(m, dict)]
    if job.get("left_out"):
        out["left_out"] = [{"first": x.get("first"), "frames": x.get("frames")} for x in job["left_out"] if isinstance(x, dict)]
    return out


def song_node(graph: dict | None) -> dict:
    """The song node's inputs: where the seed, window, context and the prompt are (`what_h3_got.SONG`)."""
    for node in (graph or {}).values():
        if str(node.get("class_type", "")).startswith(w.SONG):
            return node["inputs"]
    return {}


def graph_facts(graph: dict | None, sources: dict) -> dict:
    """What a saved or queued graph says, flat: seed, steps, sampler, window, context, the prompt, the still, the source
    loader's start and frames, the Masked Source's settings, and the files wired in. Empty when there is no graph."""
    if not graph:
        return {}
    ask = rl.asked(graph, sources)
    song = song_node(graph)
    out = {"ask": ask, "prompt": song.get("prompt") if isinstance(song.get("prompt"), str) else None}
    for node in graph.values():
        ins, cls = node.get("inputs") or {}, node.get("class_type")
        if cls == "BasicScheduler":
            out.update(steps=ins.get("steps"), scheduler=ins.get("scheduler"))
        elif cls == "KSamplerSelect":
            out["sampler"] = ins.get("sampler_name")
        elif cls in ("KSampler", "SamplerCustom") and "steps" in ins:
            out.update(steps=ins.get("steps"), sampler=ins.get("sampler_name"), scheduler=ins.get("scheduler"))
        elif cls == w.SOURCE:
            out["masked_source"] = {k: v for k, v in ins.items() if not isinstance(v, list)}
    masked = next((n for n in graph.values() if n["class_type"] == w.SOURCE), None)
    src = rl._upstream_video(graph, masked["inputs"].get("frames")) if masked else None
    if src is not None:
        rate = sources.get(Path(src["inputs"]["video"]).name, 24.0)
        out["loader_first"], out["loader_frames"] = rl._loader_span(src, rate)
    return out


def masked_others(graph: dict | None) -> list[str]:
    """The keep-outs a graph wires into the Masked Source's `others`, by the folder under `subjects/` each file is in (`held`, `hands`), else the file's name."""
    node = next((n for n in (graph or {}).values() if n.get("class_type") == w.SOURCE), None)
    link = (node or {}).get("inputs", {}).get("others")
    out = []
    for nid in w.loaders_above(graph, link) if link else []:
        video = graph[nid]["inputs"].get("video")
        if isinstance(video, str):
            p = Path(video)
            out.append(p.parent.name if p.parent.parent.name == "subjects" else p.stem.rsplit(".", 1)[-1])
    return out


def png_time(mp4: Path) -> str | None:
    """The picture's `CreationTime` text chunk (ComfyUI writes it when it saves the picture), or None."""
    png = mp4.with_suffix(".png")
    if not png.is_file():
        return None
    try:
        from PIL import Image
        return Image.open(png).info.get("CreationTime")
    except (OSError, ValueError):
        return None


def windows_listing(mp4: Path) -> list[str]:
    """The window videos in the folder beside a render (`<stem>_windows`), by name, apart from the files made beside them."""
    folder = mp4.with_name(re.sub(r"_\d{5}_?$", "", mp4.stem) + "_windows")
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.glob("*_window_[0-9]*.mp4") if not any(x in p.name for x in ("_with_mask", "_zoomed")))


def zoom_of(mp4: Path | None, graph_region: str | None) -> dict | None:
    """The zoom a render ran at: the node's `<stem>_zoomed.json` beside it, else the window's region record; None when it ran on the whole frame
    or neither is there."""
    if mp4 is None:
        return None
    rec = read_json(mp4.with_name(mp4.stem.rstrip("_") + "_zoomed.json"))
    if rec and rec.get("box"):
        return {"box": rec["box"], "scale": rec.get("zoom"), "set_by": rec.get("set_by"), "render_region": rec.get("render_region"),
                "from": "the node's zoomed record beside the render"}
    folder = mp4.with_name(re.sub(r"_\d{5}_?$", "", mp4.stem) + "_windows")
    for npz in sorted(folder.glob("*_region.npz"))[:1] if folder.is_dir() else []:
        try:
            import numpy as np
            with np.load(npz) as z:
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        except (OSError, ValueError):
            continue
        zm = meta.get("zoom")
        if zm and zm.get("box"):
            return {"box": zm["box"], "scale": zm.get("scale"), "set_by": zm.get("set_by"), "render_region": zm.get("render_region"),
                    "from": f"the window's region record {npz.name}"}
    return None


def reference_of(motion_file: str | None, job_ref: dict | None, load0: dict, share: Path) -> dict | None:
    """The reference video's drawing form: what the job said (form, iris, mouth, span words) with the reference's own record, as the record
    names it. The record has spans, not a form a frame: the per-frame form is not here."""
    if not motion_file and not job_ref:
        return None
    job_ref = job_ref or {}
    record_path = Path(job_ref["record"]) if job_ref.get("record") else Path(motion_file).with_suffix(".json") if motion_file else None
    rec = read_json(record_path) or {}
    drawn = rec.get("drawn") or {}
    iris_j = job_ref.get("iris") if isinstance(job_ref.get("iris"), dict) else {}
    mouth_j = job_ref.get("mouth") if isinstance(job_ref.get("mouth"), dict) else {}
    mouth_r = drawn.get("mouth") if isinstance(drawn.get("mouth"), dict) else {}
    shut = drawn.get("eyes_shut") if isinstance(drawn.get("eyes_shut"), dict) else {}
    rule = ((drawn.get("eye_rings") or {}).get("closed_by_the_rule") or {}) if isinstance(drawn.get("eye_rings"), dict) else {}
    clip = mouth_r.get("closed_lip_line") if isinstance(mouth_r.get("closed_lip_line"), dict) else mouth_j.get("closed_lip_line")
    stamp = rec.get("writer_stamp") or {}
    steps = rl.reference_form(motion_file) if motion_file else {}
    return {"form": job_ref.get("form"), "mesh": load0.get("motion"),
            "iris": drawn.get("iris") if isinstance(drawn.get("iris"), str) else iris_j.get("form"), "iris_said_by": iris_j.get("said_by"),
            "mouth": mouth_j.get("form"), "mouth_drawn_as": mouth_r.get("drawn_as"),
            "closed_lip_line": clip.get("on") if isinstance(clip, dict) else None,
            "eyes_shut_by_word": {"form": shut.get("form"), "frames": shut.get("frames")} if shut else
            ({"form": (iris_j.get("eyes_shut") or {}).get("form"), "frames": card_spans((iris_j.get("eyes_shut") or {}).get("spans"))}
             if isinstance(iris_j.get("eyes_shut"), dict) else None),
            "eyes_shut_by_rule": rule.get("frames") if rule.get("on") else None,
            "frames_left_bare": drawn.get("frames_left_bare"),
            "passed": rec.get("passed"), "made_by": rec.get("made_by"), "made_at": rec.get("made_at"),
            "commit": stamp.get("git_commit"), "dirty": stamp.get("git_dirty"),
            "record": shown(record_path, share, REPO) if rec else None, "steps": steps.get("says")}


def padding_of(built: dict | None, printed: str, facts: dict, frames: int | None, first: int | None, window: int | None) -> tuple[dict, str] | None:
    """(padding, the source it came from), the first of: the build record's `load_against_its_window`; the build printout's line; the graph's
    source loader against the load's frames. `kind` is real source frames, held copies of the last frame, none, or not recorded."""
    lw = ((built or {}).get("what_this_clips_preview_measured") or {}).get("load_against_its_window")
    if isinstance(lw, dict) and "held_past_the_load" in lw:
        before, after, held = lw.get("context_before") or 0, lw.get("context_after") or 0, lw.get("held_past_the_load") or 0
        kind = "real source frames" if before or after else "held copies of the last frame" if held else "none"
        return {"before": before, "after": after if not held else held, "held": held, "kind": kind}, "built.json what_this_clips_preview_measured.load_against_its_window"
    m = re.search(r"padded with (\d+) real frame\(s\) before and (\d+) after", printed)
    if m:
        return {"before": int(m[1]), "after": int(m[2]), "held": 0, "kind": "real source frames"}, "build printout"
    m = re.search(r"its last frame held for (\d+) frames", printed)
    if m:
        return {"before": 0, "after": int(m[1]), "held": int(m[1]), "kind": "held copies of the last frame"}, "build printout"
    lf, lfirst = facts.get("loader_frames"), facts.get("loader_first")
    if lf is not None and frames is not None and first is not None and lfirst is not None:
        before = first - lfirst
        after = lf - frames - before
        if before > 0 or after > 0:
            return {"before": max(before, 0), "after": max(after, 0), "held": 0, "kind": "real source frames"}, "graph: the source loader's start and frames against the load's"
        if window and window > frames:
            return {"before": 0, "after": window - frames, "held": None, "kind": "not recorded"}, \
                "graph: the window is longer than the loader's frames; how the rest was filled is in no record"
        return {"before": 0, "after": 0, "held": 0, "kind": "none"}, "graph: the source loader's frames are the load's own, and the window is not longer than them"
    return None


def same_line_once(lines: list[dict]) -> list[dict]:
    """The owner's lines with a line the verdicts file holds twice taken once: `verdicts.jsonl` carries some of the owner's lines under two
    `source` labels (`answers.jsonl:N` and `owner_answers.jsonl:N`, the answers file's old and new name), the same line at the same second. The
    two labels are kept in `from`, joined."""
    out: dict[tuple, dict] = {}
    for line in lines:
        key = (line.get("at"), line.get("what"), str(line.get("said")), line.get("frame"), line.get("row"))
        if key in out:
            if line["from"] not in out[key]["from"].split(" = "):
                out[key]["from"] += f" = {line['from']}"
        else:
            out[key] = dict(line)
    return list(out.values())


#: A viewing clip of one load: `<load>_<first>_<last>_stacked.mp4` (`bench/stack_delivery.py`). A line of the owner's whose `clip` is one is said of that load's
#: render. A joined cut (`fun_0000_6394_joined_..._VIEW.mp4`) or a `_compare` clip is not: it shows more than one render, and its lines stay off a render.
STACKED_CLIP = re.compile(r"^(?P<load>.+?)_\d{4}_\d{4}_stacked\.mp4$")


def laid_index(laid_path: Path | None) -> tuple[list[dict], list[dict]]:
    """(the lines of `laid.jsonl`, the lines taken out of it, `laid.removed.jsonl`'s `removed` objects with why and when)."""
    now = [d for _, d in jsonl(laid_path)]
    gone = []
    if laid_path is not None:
        for _, d in jsonl(rl_removed(laid_path)):
            x = d.get("removed")
            if isinstance(x, dict):
                gone.append({**x, "_by": d.get("by"), "_at": d.get("at"), "_by_session": d.get("by_session")})
    return now, gone


def rl_removed(laid_path: Path) -> Path:
    """Where `bench/laid.py` keeps the lines it took out (`laid.removed.jsonl` beside the list)."""
    return Path(laid_path).with_name(Path(laid_path).stem + ".removed.jsonl")


class History:
    """One video's renders: the files it reads, and the row of each."""

    def __init__(self, video: str, data: Path, share: Path, sources: dict[str, float], repo: Path = REPO, shots: dict | None = None,
                 plan: dict | None = None, verdicts: list | None = None, answers: list | None = None, items: list | None = None,
                 laid: Path | None = None, ledgers: list[tuple[str, dict]] | None = None, frames_of=rl.count_frames):
        self.video, self.data, self.share, self.sources, self.repo = video, Path(data), Path(share), sources, Path(repo)
        self.shots = (shots or {}).get("shots") or []
        self.prefixes = {k: v for k, v in ((plan or {}).get("load_prefixes") or {}).items() if not k.startswith("_")}
        self.verdicts, self.answers = list(verdicts or []), list(answers or [])
        self.item_of = {i["id"]: i["render"] for i in items or [] if i.get("render")}
        self.laid_now, self.laid_gone = laid_index(laid)
        self.frames_of = frames_of
        self.cuts = self.cut_sidecars()
        self.by_file, self.every = rl.load_records(self.data, self.repo)
        self.record_of_name = {}
        for rec in self.every:
            name = Path(rec["queued_file"]).name.removesuffix(".render.queued.json")
            self.record_of_name[name] = rec
        self.stopped = {x["load"]: x for x in rl.stopped_loads(self.data, self.repo, sources)}
        self.old = {}                  # a render's file -> the record an earlier ledger kept of it, later ledgers over earlier
        for label, led in ledgers or []:
            for rel, r in (led.get("renders") or {}).items():
                if r.get("records"):
                    self.old[rel] = {"ledger": label, "record": r["records"][-1], "render": r}

    def cut_sidecars(self) -> list[dict]:
        """The whole-video cuts the delivery assembler made (`Video/<video>/day_*/<video>_*_joined_*_from_laid.mp4.json`), each with the pieces it laid:
        {cut, made_at, pieces: [(name, first, last)]}. A cut is the record of what WAS laid when it was made; `laid.jsonl` holds only what is laid now."""
        out = []
        for path in sorted((self.share / "Video" / self.video).glob("day_*/*_from_laid.mp4.json")):
            d = read_json(path)
            if isinstance(d, dict) and d.get("pieces"):
                out.append({"cut": path.name.removesuffix(".json"), "made_at": d.get("made_at"),
                            "pieces": [(p.get("name"), p.get("first"), p.get("last")) for p in d["pieces"] if isinstance(p, dict)]})
        return out

    def said_of_its_clip(self, name: str) -> list[dict]:
        """The owner's lines whose `clip` is the stacked viewing clip of this load, in `render_ledger.owner_lines`' shape, from the verdicts file and the answers file."""
        out = []
        pool = [(f"verdicts ({v.get('source')})", v) for v in self.verdicts] + [(f"answers.jsonl:{n}", a) for n, a in self.answers if a.get("who") == "owner"]
        for label, v in pool:
            m = STACKED_CLIP.match(Path(str(v.get("clip") or "")).name)
            if m and m["load"] == name:
                said = v.get("chosen") if v.get("what") == "choice" else v.get("words")
                if said:
                    out.append({"at": v.get("at"), "frame": v.get("frame"), "said": said, "what": v.get("what"), "from": f"{label}, by the clip it names", "row": v.get("row")})
        return sorted(out, key=lambda o: str(o["at"]))

    # ---- discovery -------------------------------------------------------------------------------------------------

    def chain_loads(self) -> dict[str, Path]:
        """{load name: its job folder} from `data/*_job` (a preview's folder is not a load); the newest folder of a name when two hold it."""
        out: dict[str, Path] = {}
        for folder in sorted(self.data.glob("*_job")):
            if (folder / "preview.json").is_file() or (folder / "preview.queued.json").is_file() or "_shot_c_" in folder.name:
                continue                      # a preview's folder (`data/<day>_shot_c_<first><letter>_job`; the older ones have no preview.json) is no load
            names = {m["load"] for f in folder.iterdir() if (m := LOAD_FILE.match(f.name)) and m["load"] != "job"}      # `job.render.json` is the job file's copy, not a load
            names |= {f.name.split(".")[0] for f in (folder / "chain").glob("*.out")} if (folder / "chain").is_dir() else set()
            for name in names:
                out[name] = folder
        return out

    def job_files(self, taken: set[tuple[str, str | None]]) -> dict[tuple[str, str | None], Path]:
        """{(load name, its capture folder): the newest job file} for every `data/*/<load>.json` that is a job file and whose (name, capture) is not in `taken`
        (the loads that have a job folder). A name can be used twice: a job file written and never built frees its name, and the next load takes it
        (`second_hfp_4912b` of 2026-10-20 and of 2026-10-21 are two loads); the capture folder the job names tells them apart."""
        found: dict[str, list[Path]] = defaultdict(list)
        starts = tuple(x for prefixes in self.prefixes.values() for x in prefixes)
        # a load's name begins with a prefix of the plan's (`load_prefixes`) and holds a first frame; with no plan the pattern `JOB_FILE_NAME` stands
        named = (lambda stem: stem.startswith(starts) and re.search(r"\d{3,4}", stem) is not None) if starts else (lambda stem: JOB_FILE_NAME.match(stem) is not None)
        for p in self.data.glob("*/*.json"):
            if named(p.stem) and not p.name.startswith("job."):
                found[p.stem].append(p)
        out: dict[tuple[str, str | None], Path] = {}
        for name, paths in found.items():
            for p in sorted(paths, key=lambda q: q.stat().st_mtime, reverse=True):
                d = read_json(p)
                if isinstance(d, dict) and "job" in d and "loads" in d and (d["job"] or {}).get("name") == name:
                    key = (name, (d["job"] or {}).get("capture"))
                    if key not in taken and key not in out:
                        out[key] = p
        return out

    def job_of(self, name: str, folder: Path | None, built: dict | None) -> tuple[dict | None, str | None]:
        """(the job file's object, where it was read), the first of the folder's own copy, the one the build record names, then `data/*/<load>.json`:
        of those the one whose `job.capture` is the load's own capture folder (the job folder's name without `_job`), else the newest."""
        tries = []
        if folder is not None:
            tries += [folder / "job.render.json", folder / "job.plan.json"]
        if built and built.get("job_file"):
            p = Path(built["job_file"])
            tries.append(p if p.is_absolute() else self.repo / p)
        for p in tries:
            d = read_json(p)
            if isinstance(d, dict) and "job" in d and "loads" in d:
                return d, shown(p, self.repo)
        by_name = sorted(self.data.glob(f"*/{name}.json"), key=lambda q: q.stat().st_mtime, reverse=True)
        mine = shown(folder, self.repo).removesuffix("_job") if folder is not None else None
        found = [(p, read_json(p)) for p in by_name]
        found = [(p, d) for p, d in found if isinstance(d, dict) and "job" in d and "loads" in d]
        for p, d in found:
            if mine and (d["job"] or {}).get("capture") == mine:
                return d, shown(p, self.repo)
        return (found[0][1], shown(found[0][0], self.repo)) if found else (None, None)

    def text_files(self, name: str, folder: Path | None, jobd: dict | None, job_from: str | None) -> list[tuple[str, Path]]:
        """The text files a load may have, in the order they are trusted: the render's, the plan's, the one beside the job file its `texts` names."""
        out = []
        if folder is not None:
            out += [(f"{name}.render.txt", folder / f"{name}.render.txt"), (f"{name}.plan.txt", folder / f"{name}.plan.txt")]
        file = (((jobd or {}).get("job") or {}).get("texts") or {}).get(name, {}).get("file")
        if file and job_from:
            beside = (self.repo / job_from).parent / file
            out.append((shown(beside, self.repo), beside))
        return out

    def new_row(self, name: str, kind: str) -> dict:
        return {"schema": SCHEMA, "video": self.video, "load": name, "kind": kind, "subject": None, "session": None,
                "shot_first": None, "shot_last": None, "span_first": None, "span_last": None, "frames_given": None,
                "rendered_at": None, "rendered_at_from": None, "recorded_at": None, "render_seconds": None, "png_creation_time": None,
                "status": None, "stopped_after": None, "last_log_line": None, "stopped_note": None,
                "gate": None, "gate_overridden": None, "looked_at_by": None,
                "output_file": None, "output_files": [], "output_mtime": None, "output_frames": None,
                "source_video": None, "window_length": None, "loader_frames": None, "padding_before": None, "padding_after": None,
                "padding_kind": None, "windows": None, "window_lengths": None, "context_frames": None, "context_noise": None,
                "seed": None, "steps": None, "sampler": None, "scheduler": None, "render_region": None, "zoom": None,
                "reference_video": None, "reference_folder": None, "reference_form": None, "still": None,
                "part": None, "margin": None, "grow_by": None, "feather": None, "edge": None, "composite": None, "keep_outs": None,
                "span_words": None, "carried_words_from": None, "hair_band": None,
                "preview": None, "preview_commit": None, "preview_queued_at": None,
                "prompt": None, "prompt_sha256": None, "png_graph_prompt_matches": None,
                "commit": None, "plan_commit": None, "plan_queued_at": None, "built_at": None, "uncommitted_node_code": None,
                "laid": None, "owner": [], "records": {}, "sources": {}, "cannot_see": [], "carried_from": None}

    def row(self, name: str, kind: str, folder: Path | None = None, rel: str | None = None, job_only: Path | None = None, shared: bool = False) -> dict:
        r = self.new_row(name, kind)
        src = r["sources"]

        def put(field: str, value, source: str) -> None:
            if value is not None and r[field] is None:
                r[field], src[field] = value, source

        built = read_json(folder / f"{name}.render.built.json") if folder else None
        built_name = f"{name}.render.built.json"
        if built is None and folder is not None and (folder / f"{name}.plan.built.json").is_file():
            built, built_name = read_json(folder / f"{name}.plan.built.json"), f"{name}.plan.built.json"     # a load that stopped at the plan step: its plan's build record
        queued = read_json(folder / f"{name}.render.queued.json") if folder else None
        qgraph = read_json(folder / f"{name}.render.json") if folder else None
        qgraph_name = f"{name}.render.json"
        if qgraph is None and folder is not None and (folder / f"{name}.plan.json").is_file():
            qgraph, qgraph_name = read_json(folder / f"{name}.plan.json"), f"{name}.plan.json"           # the plan step's graph, queued; the render's never was
        printed = (folder / f"{name}.render.build_printed.txt").read_text() if folder and (folder / f"{name}.render.build_printed.txt").is_file() else ""
        plan_q = read_json(folder / f"{name}.plan.queued.json") if folder else None
        if job_only is not None:
            jobd, job_from = read_json(job_only), shown(job_only, self.repo)
        else:
            jobd, job_from = self.job_of(name, folder, built) if folder else (None, None)
        job = (jobd or {}).get("job") or {}
        load0 = next((x for x in (jobd or {}).get("loads") or [] if x.get("name") == name), ((jobd or {}).get("loads") or [{}])[0]) if jobd else {}
        dflt = (jobd or {}).get("defaults") if isinstance((jobd or {}).get("defaults"), dict) else {}     # the settings the job names (seed, margin, feather, edge, carried)
        old = self.old.get(rel) if rel else None
        orec = old["record"] if old else None
        olabel = f"ledger {old['ledger']}" if old else None
        mp4 = self.share / rel if rel else None
        png_g = rl.saved_graph(mp4) if mp4 is not None and mp4.is_file() else None
        graph, graph_from = (qgraph, qgraph_name) if qgraph else (png_g, "the picture beside the mp4") if png_g else (None, None)
        facts = graph_facts(graph, self.sources)
        ask = facts.get("ask") or {}
        loadb = (built or {}).get("load") or {}
        settings = (built or {}).get("settings") or {}
        sv = lambda key: (settings.get(key) or {}).get("value")
        r["records"] = {"job_folder": shown(folder, self.repo), "job_file": job_from,
                        "built": f"{shown(folder, self.repo)}/{built_name}" if built is not None else None,
                        "queued": f"{shown(folder, self.repo)}/{name}.render.queued.json" if queued is not None else None,
                        "graph": f"{shown(folder, self.repo)}/{qgraph_name}" if qgraph else None,
                        "picture": shown(mp4.with_suffix(".png"), self.share) if png_g else None,
                        "capture": shown((built or {}).get("capture") or job.get("capture"), self.repo), "ledger": olabel}

        # who, what span
        put("subject", loadb.get("subject"), f"{built_name} load.subject")
        put("subject", load0.get("subject"), "job file loads[0]")
        put("subject", ((job.get("subjects") or [{}])[0]).get("label"), "job file subjects[0].label")
        put("subject", (orec or {}).get("load", {}).get("subject"), olabel or "")
        capture, label = rl._capture_of((ask.get("files") or {}).get("mask"))
        put("subject", label, "the capture folder the graph's mask sits in")
        if r["subject"] is None:
            for who, prefixes in self.prefixes.items():
                if any(name.startswith(p) for p in prefixes):
                    put("subject", who, "the load name's prefix, by the plan")
        first = loadb.get("first") if loadb.get("first") is not None else load0.get("first") if load0.get("first") is not None else \
            job.get("first") if job.get("first") is not None else (orec or {}).get("load", {}).get("first") if orec else ask.get("first")
        frames = loadb.get("frames") if loadb.get("frames") is not None else load0.get("frames") if load0.get("frames") is not None else \
            job.get("frames") if job.get("frames") is not None else (orec or {}).get("load", {}).get("frames") if orec else ask.get("frames")
        first_from = f"{built_name} load" if loadb.get("first") is not None else "job file" if (load0.get("first") is not None or job.get("first") is not None) \
            else olabel if orec and (orec.get("load") or {}).get("first") is not None else "the graph's source loader"
        if first is not None and frames:
            r["span_first"], r["span_last"], r["frames_given"] = int(first), int(first) + int(frames) - 1, int(frames)
            src["span_first"] = src["span_last"] = src["frames_given"] = first_from
            shot = next((s for s in self.shots if s["first"] <= first <= s["last"]), None)
            if shot:
                r["shot_first"], r["shot_last"] = shot["first"], shot["last"]
                src["shot_first"] = src["shot_last"] = "shots.json"
        # where it was made
        by = re.match(r"Written by bench/shot_to_load\.py for (\S+?),", str(job.get("note") or ""))
        if by:
            put("session", by[1], "job file job.note (\"Written by bench/shot_to_load.py for <session>\")")
        if rel:
            parts = Path(rel).parts
            r["session"] = parts[2] if len(parts) > 2 and parts[1] == self.video else parts[1] if len(parts) > 1 else None
            r["output_file"] = rel
            src["session"] = src["output_file"] = "the file's place under the share"
        if queued:
            r["output_files"] = [f for f in queued.get("files") or [] if rl.NUMBERED.search(f)]
        if rel and rel not in r["output_files"]:
            r["output_files"].append(rel)
        # the run: time, outcome
        qat = utc_of((queued or {}).get("queued_at"))
        if qat:
            put("rendered_at", qat, "queued.json queued_at")
            r["rendered_at_from"] = "the queue record's queued_at"
        elif orec and orec.get("queued_at"):
            put("rendered_at", utc_of(orec["queued_at"]), f"{olabel} (queued_at)")
            r["rendered_at_from"] = f"an earlier ledger's record of the queue ({old['ledger']})"
        if mp4 is not None and mp4.is_file():
            st = mp4.stat()
            r["output_mtime"] = rl.utc(st.st_mtime)
            r["png_creation_time"] = png_time(mp4)
            if r["rendered_at"] is None:
                ct = utc_of(r["png_creation_time"])
                put("rendered_at", ct or r["output_mtime"], "the picture's CreationTime (no zone, read as local time)" if ct else "the output file's mtime")
                r["rendered_at_from"] = "the picture's CreationTime, written without a zone and read as this machine's local time" if ct else "the output file's mtime"
            r["output_frames"] = self.frames_of(mp4)
            src["output_frames"] = "ffprobe packets"
        put("render_seconds", (queued or {}).get("seconds"), "queued.json seconds")
        put("gate", ((queued or {}).get("gate") or {}).get("verdict") or ((orec or {}).get("gate") or {}).get("verdict"), "queued.json gate" if queued else olabel or "")
        if queued is not None and (queued.get("gate") or {}).get("overridden"):
            r["gate_overridden"] = queued["gate"]["overridden"]
        put("looked_at_by", ((queued or {}).get("look") or {}).get("by") or (orec or {}).get("looked_at_by"), "queued.json look.by" if queued else olabel or "")
        put("commit", (queued or {}).get("commit"), "queued.json commit")
        if built is not None and built_name.endswith(".render.built.json"):
            put("commit", built.get("commit"), f"{built_name} commit")
        if plan_q:
            r["plan_commit"], r["plan_queued_at"] = plan_q.get("commit"), utc_of(plan_q.get("queued_at"))
            src["plan_commit"] = src["plan_queued_at"] = f"{name}.plan.queued.json"
        put("commit", (orec or {}).get("commit"), olabel or "")
        put("built_at", utc_of((built or {}).get("built_at") or (orec or {}).get("built_at")), f"{built_name} built_at" if built else olabel or "")
        put("uncommitted_node_code", (queued or {}).get("uncommitted_node_code") if queued else (orec or {}).get("uncommitted_node_code") if orec else None,
            "queued.json" if queued else olabel or "")
        # status, said as the files say it
        self.status(r, name, folder, job_only, built, queued, plan_q, rel, mp4, orec, old)
        if r["gate"] is None and folder is not None and (folder / "chain").is_dir():
            for log in sorted((folder / "chain").glob(f"{name}.gate*.out"), key=lambda q: q.stat().st_mtime, reverse=True):
                said = [x.strip() for x in log.read_text().splitlines() if x.strip().startswith("gate:")]
                if said:
                    r["gate"], src["gate"] = said[-1].removeprefix("gate:").strip(), f"chain/{log.name} (its last `gate:` line)"
                    break
        # the load's inputs and settings
        put("source_video", Path(job["source"]).name if job.get("source") else None, "job file job.source")
        put("source_video", ask.get("of"), "graph source loader")
        window = sv("window") if sv("window") is not None else ask.get("window") if ask.get("window") is not None else (orec or {}).get("window")
        put("window_length", window, f"{built_name} settings.window" if sv("window") is not None else "graph song node window_frames" if ask.get("window") is not None else olabel or "")
        put("loader_frames", facts.get("loader_frames"), "graph source loader frame_load_cap")
        pad = padding_of(built, printed, facts, r["frames_given"], r["span_first"], r["window_length"])
        if pad:
            r["padding_before"], r["padding_after"], r["padding_kind"] = pad[0]["before"], pad[0]["after"], pad[0]["kind"]
            src["padding_before"] = src["padding_after"] = src["padding_kind"] = pad[1].replace("built.json", built_name)
        count = None
        for f in ((queued or {}).get("windows_files") or []):
            if re.search(r"_window_\d+\.mp4$", f):
                count = (count or 0) + 1
        if count is not None:
            r["windows"], src["windows"] = count, "queued.json windows_files"
        elif mp4 is not None and mp4.is_file():
            listing = windows_listing(mp4)
            if listing:
                r["windows"], src["windows"] = len(listing), "the windows folder's listing"
        said = "\n".join(line for lines in ((queued or {}).get("said") or {}).values() for line in (lines if isinstance(lines, list) else [str(lines)]))
        chained = re.search(r"(\d+) windows \[([\d, ]+)\] with (\d+)-frame context", said)
        if chained:
            r["window_lengths"], src["window_lengths"] = [int(x) for x in chained[2].split(",")], "queued.json said: the node's own line \"N windows [..] with M-frame context\""
        put("context_frames", ask.get("context") if ask.get("context") is not None else (orec or {}).get("context"), "graph song node context_frames" if ask.get("context") is not None else olabel or "")
        put("context_noise", ask.get("context_noise"), "graph song node context_noise")
        seed = sv("seed") if sv("seed") is not None else loadb.get("seed") if loadb.get("seed") is not None else ask.get("seed") if ask.get("seed") is not None else \
            (orec or {}).get("load", {}).get("seed") if orec else None
        put("seed", seed, f"{built_name} settings.seed" if sv("seed") is not None else f"{built_name} load.seed" if loadb.get("seed") is not None else
            "graph song node seed" if ask.get("seed") is not None else olabel or "")
        put("seed", dflt.get("seed"), "job file defaults.seed (the settings the job names)")
        for key in ("steps", "sampler", "scheduler"):
            put(key, facts.get(key), f"graph ({'BasicScheduler' if key != 'sampler' else 'KSamplerSelect'})")
        region = sv("render_region") or loadb.get("render_region") or ask.get("render_region")
        put("render_region", region, f"{built_name} settings.render_region" if sv("render_region") else "graph Masked Source render_region")
        z = zoom_of(mp4, region) if mp4 is not None and mp4.is_file() else None
        if z:
            r["zoom"], src["zoom"] = z, z["from"]
        elif r["status"] == "rendered" and (region is None or "whole frame" in str(region)):
            # a render that ran on the whole frame has no box: said as that, not as unseen (the ledger reads a graph with no `render_region` the same way)
            r["zoom"] = {"box": None, "scale": 1.0, "set_by": None, "render_region": region,
                         "from": "render_region says the whole frame" if region else "the graph names no render_region: read as the whole frame, as render_ledger does"}
            src["zoom"] = r["zoom"]["from"]
        motion = loadb.get("motion_file") or load0.get("motion_file") or (orec or {}).get("load", {}).get("motion_file") if (loadb or load0 or orec) else None
        motion = motion or (ask.get("files") or {}).get("motion_video")
        if motion and not str(motion).startswith("made in the graph"):
            put("reference_video", shown(motion, self.share, self.repo), f"{built_name} load.motion_file" if loadb.get("motion_file") else "job file loads[0]" if load0.get("motion_file") else "graph")
            put("reference_folder", Path(motion).parent.name, "the reference file's folder")
            ref = reference_of(motion, job.get("reference"), {**load0, **loadb}, self.share)
            if ref:
                r["reference_form"], src["reference_form"] = ref, "job file reference + the reference's own record"
        elif motion:
            r["reference_form"], src["reference_form"] = {"form": None, "steps": motion}, "graph (the reference is made inside the graph)"
        mine = next((x for x in job.get("subjects") or [] if x.get("label") == r["subject"]), {})
        put("still", loadb.get("still") or load0.get("still") or ", ".join(ask.get("stills") or []) or mine.get("still") or None,
            f"{built_name} load.still" if loadb.get("still") else "job file loads[0]" if load0.get("still") else "graph LoadImage" if ask.get("stills") else "job file subjects[].still")
        # the part, region, keep-outs
        carried = sv("carried") or loadb.get("carried") or load0.get("carried") or ask.get("replace")
        cap = Path(r["records"]["capture"]) if r["records"]["capture"] else capture
        if cap is not None and not cap.is_absolute():
            cap = self.repo / cap
        subject = r["subject"]
        classes = rl.part_classes(cap, subject) if carried in ("held", "the wired parts") and cap is not None else None
        names = ((job.get("subjects") or [{}])[0].get("parts") if job.get("subjects") else None)
        if carried or names:
            region_of = rl.region_of(carried, classes)
            r["part"], src["part"] = {"parts": names, "carried": carried, "classes": classes, "region": region_of}, \
                f"job file subjects[].parts + {built_name} settings.carried + the capture's manifest"
            kept = ((old or {}).get("render") or {}).get("form") or {}
            if region_of in (None, "a part") and kept.get("region"):
                r["part"]["region"], r["part"]["region_from"] = kept["region"], f"{olabel}: its form, read from the capture while it existed"
        put("margin", sv("margin") if sv("margin") is not None else ask.get("margin"), f"{built_name} settings.margin" if sv("margin") is not None else "graph grow_pixels")
        put("margin", dflt.get("margin"), "job file defaults.margin (the settings the job names)")
        ms = facts.get("masked_source") or {}
        put("feather", sv("feather") if sv("feather") is not None else ms.get("feather_pixels"), f"{built_name} settings.feather" if sv("feather") is not None else "graph feather_pixels")
        put("feather", dflt.get("feather"), "job file defaults.feather (the settings the job names)")
        put("grow_by", ms.get("grow_by"), "graph Masked Source grow_by (the margin is grown by this rule)")
        put("edge", sv("edge") or ms.get("edge"), f"{built_name} settings.edge" if sv("edge") else "graph edge")
        put("composite", sv("composite") or ms.get("composite"), f"{built_name} settings.composite" if sv("composite") else "graph composite")
        if loadb.get("others") is not None or load0.get("others") is not None:
            r["keep_outs"], src["keep_outs"] = list(loadb.get("others") if loadb.get("others") is not None else load0.get("others")), f"{built_name} load.others" if loadb.get("others") is not None else "job file loads[0].others"
        elif masked_others(graph):
            r["keep_outs"], src["keep_outs"] = masked_others(graph), "graph: every loader above the Masked Source's `others` (the folder under subjects/, else the file)"
        hb = (built or {}).get("hair_band")
        if isinstance(hb, dict):
            r["hair_band"], src["hair_band"] = {"file": Path(str(hb.get("file"))).name, "band_px_median": hb.get("band_px_median"), "cut_out_of": hb.get("cut_out_of"),
                                                "frames_with_a_band": hb.get("frames_with_a_band")}, f"{built_name} hair_band"
        r["span_words"] = span_words(job) if job else None
        if r["span_words"] is not None:
            src["span_words"] = "job file"
        cw = job.get("carried_words_from")
        if isinstance(cw, dict):
            r["carried_words_from"], src["carried_words_from"] = cw.get("from"), "job file job.carried_words_from"
        # the preview's commit
        pv = job.get("preview")
        if pv:
            pfolder = Path(pv) if Path(pv).is_absolute() else self.repo / pv
            r["preview"], src["preview"] = shown(pfolder, self.repo), "job file job.preview"
            pq = read_json(pfolder / "preview.queued.json")
            if pq:
                r["preview_commit"], r["preview_queued_at"] = pq.get("commit"), utc_of(pq.get("queued_at"))
                src["preview_commit"] = src["preview_queued_at"] = "the preview's preview.queued.json"
        # the exact prompt
        prompt = facts.get("prompt")
        prompt_from = f"{graph_from} {w.SONG} prompt" if prompt is not None else None
        if prompt is None and ask.get("text"):
            prompt, prompt_from = ask["text"], f"{graph_from} (the longest text input: no song node)"
        for label, path in self.text_files(name, folder, jobd, job_from):
            if prompt is None and path.is_file():
                prompt, prompt_from = path.read_text(), f"{label} (a text file: it never reached a queued graph, or the graph is gone)"
        if prompt is not None:
            r["prompt"], r["prompt_sha256"] = prompt, hashlib.sha256(prompt.encode()).hexdigest()
            src["prompt"] = src["prompt_sha256"] = prompt_from
            if qgraph and png_g and facts.get("prompt") is not None and graph_from == f"{name}.render.json":
                r["png_graph_prompt_matches"] = song_node(png_g).get("prompt") == prompt
        # laid, and the owner. A job file written and never built that shares its name with a load that was built gets neither: the lines are the built load's
        mycap = job.get("capture")
        others = sorted({shown(q, self.repo) for q in self.data.glob(f"*/{name}.json")
                         if (read_json(q) or {}).get("job", {}).get("name") == name and (read_json(q) or {}).get("job", {}).get("capture") != mycap})
        r["records"]["other_job_files_of_this_name"] = others
        keys = {name}
        mine = lambda d: d.get("load") in keys or (rel and d.get("file") == rel)
        now = [] if shared else [d for d in self.laid_now if mine(d)]
        gone = [] if shared else [d for d in self.laid_gone if mine(d)]
        r["laid"] = {"now": [{"frames": d.get("frames"), "owner": d.get("owner"), "added": d.get("added"), "added_by": d.get("added_by"), "by": d.get("by")} for d in now],
                     "removed": [{"frames": d.get("frames"), "at": d.get("_at"), "by": d.get("_by"), "by_session": d.get("_by_session")} for d in gone]}
        r["laid"]["in_cuts"] = [{"cut": c["cut"], "made_at": c["made_at"], "frames": [[a, b] for n, a, b in c["pieces"] if n == name]}
                                for c in self.cuts if any(n == name for n, _, _ in c["pieces"])]
        src["laid"] = "laid.jsonl + laid.removed.jsonl + the joined cuts' sidecars on the share"
        ids = {name, *(f"r__{p}__{name}" for p in (r["session"], Path(rel).parts[1] if rel and len(Path(rel).parts) > 1 else None) if p)}
        r["owner"] = [] if shared else same_line_once(rl.owner_lines(rel or "", ids, self.verdicts, self.answers, self.item_of) + self.said_of_its_clip(name))
        src["owner"] = "verdicts.jsonl + owner_answers.jsonl by the row (the load name), the file or a page item"
        # what no source gave
        for field, looked in LOOKED_IN.items():
            if field in AFTER_RENDER and r["status"] != "rendered":
                continue
            if r[field] is None:
                r["cannot_see"].append({"field": field, "looked_in": list(looked)})
        return r

    def status(self, r, name, folder, job_only, built, queued, plan_q, rel, mp4, orec, old) -> None:
        """Say how far the load got, as the files say it, and the last line of the furthest chain log. No reason is guessed: a note a card session wrote is quoted."""
        src = r["sources"]
        for label, when in (("queued.json queued_at", (queued or {}).get("queued_at")), (f"{name}.plan.queued.json queued_at", (plan_q or {}).get("queued_at")),
                            ("the build record's built_at", (built or {}).get("built_at")), ("an earlier ledger's record", (orec or {}).get("queued_at"))):
            if utc_of(when):
                r["recorded_at"], src["recorded_at"] = utc_of(when), label
                break
        if r["recorded_at"] is None and folder is not None:
            logs = [p for p in (folder / "chain").glob(f"{name}.*.out")] if (folder / "chain").is_dir() else []
            if logs:
                r["recorded_at"], src["recorded_at"] = rl.utc(max(p.stat().st_mtime for p in logs)), "the newest chain log's mtime"
        if job_only is not None and r["recorded_at"] is None:
            r["recorded_at"], src["recorded_at"] = rl.utc(job_only.stat().st_mtime), "the job file's mtime"
        note = self.stopped.get(name)
        if note:
            r["stopped_note"], src["stopped_note"] = {"file": note["note"], "written": note["note_written"], "says": note["note_says"]}, "the note a card session wrote beside the job file"
        if queued is not None and queued.get("status") == "success" and rel and mp4 is not None and mp4.is_file():
            r["status"], src["status"] = "rendered", "queued.json status + the file on the share"
        elif queued is not None and queued.get("status") == "success":
            r["status"], src["status"] = "rendered, output file not on the share", "queued.json status; the file it names is gone"
        elif queued is not None and queued.get("status"):
            r["status"], src["status"] = f"queued, ended {queued['status']}", "queued.json status"
        elif orec is not None and rel and mp4 is not None and mp4.is_file():
            r["status"], src["status"] = "rendered", f"ledger {old['ledger']} (the record's status {orec.get('status')}) + the file on the share"
        elif rel and mp4 is not None and mp4.is_file():
            r["status"], src["status"] = "rendered", "the file on the share with a saved graph beside it; no record of the run"
        elif job_only is not None and folder is None:
            r["status"], src["status"] = "job file written, no chain step ran", "a job file with no job folder"
        elif folder is not None:
            reached = None
            for step, names in STEP_FILES:
                if any((folder / n.format(l=name)).is_file() for n in names):
                    reached = (step, [n.format(l=name) for n in names if (folder / n.format(l=name)).is_file()])
            if reached:
                r["stopped_after"], src["stopped_after"] = reached[0], "the furthest chain step with a file"
                r["status"] = f"stopped before render, at the {reached[0]} step" if reached[0] != "render" else "render built, not queued"
                src["status"] = f"the furthest chain step with a file: {', '.join(reached[1])}"
                logs = sorted((folder / "chain").glob(f"{name}.*.out"), key=lambda p: p.stat().st_mtime) if (folder / "chain").is_dir() else []
                if logs:
                    lines = [x.strip() for x in logs[-1].read_text().splitlines() if x.strip()]
                    r["last_log_line"] = f"{logs[-1].name}: {lines[-1][:300]}" if lines else None
                    src["last_log_line"] = f"chain/{logs[-1].name}, its last line"
                    for log in logs:
                        if "gate" in log.name:
                            gl = [x.strip() for x in log.read_text().splitlines() if x.strip()]
                            if gl and gl[-1].startswith("gate:"):
                                r["last_log_line"], src["last_log_line"] = f"{log.name}: {gl[-1][:300]}", f"chain/{log.name}, its last `gate:` line"
            else:
                r["status"], src["status"] = "job folder with no step file", "no chain file in the folder"

    # ---- all rows --------------------------------------------------------------------------------------------------

    def rows(self, share_files: list[Path], loads_re: str | None = None) -> tuple[list[dict], dict]:
        """(the rows, what was counted and left out)."""
        left_out: dict[str, int] = defaultdict(int)
        walked: dict[str, dict] = {}
        for path in share_files:
            rel = str(path.relative_to(self.share))
            recs, _ = rl.records_of(rel, self.share, self.by_file, self.every)
            if loads_re and not recs and not re.search(loads_re, re.sub(r"_\d{5}_?$", "", path.stem)):
                left_out["numbered videos not selected by --loads (their pictures were not read)"] += 1
                continue
            graph = rl.saved_graph(path)
            relaid = rl.saved_relaid(path)
            if relaid is not None:
                left_out["re-laid pieces (a composite setting changed on an existing render, not an H3 render)"] += 1
                continue
            if graph is None and not recs and rel not in self.old:
                left_out["numbered videos with no saved graph and no record (cannot be tied to a video)"] += 1
                continue
            ask = rl.asked(graph, self.sources) if graph is not None else None
            if ask is not None and ask.get("of") not in self.sources and not recs and rel not in self.old:
                left_out["renders of other videos, or numbered videos whose graph is no render"] += 1
                continue
            walked[rel] = {"path": path, "recs": recs}
        chain = self.chain_loads()
        tied: dict[str, list[str]] = defaultdict(list)
        for rel, info in walked.items():
            for rec in info["recs"]:
                name = Path(rec["queued_file"]).name.removesuffix(".render.queued.json")
                if rel not in tied[name]:
                    tied[name].append(rel)
        rows, used = [], set()
        for name, folder in sorted(chain.items()):
            if loads_re and not re.search(loads_re, name):
                continue
            files = tied.get(name, [])
            used.update(files)
            if not self.in_this_video(name, folder, files):
                left_out["loads of other videos (by the job file's source)"] += 1
                continue
            for rel in files or [None]:
                rows.append(self.row(name, "chain load", folder=folder, rel=rel))
        taken = set()
        for name, folder in chain.items():
            jobd, _ = self.job_of(name, folder, read_json(folder / f"{name}.render.built.json") or read_json(folder / f"{name}.plan.built.json"))
            taken.add((name, ((jobd or {}).get("job") or {}).get("capture")))
        for (name, _), path in sorted(self.job_files(taken).items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if loads_re and not re.search(loads_re, name):
                continue
            d = read_json(path) or {}
            if Path(str((d.get("job") or {}).get("source") or "")).name not in self.sources:
                left_out["loads of other videos (by the job file's source)"] += 1
                continue
            rows.append(self.row(name, "job file only", job_only=path, shared=name in chain))
        for rel, info in sorted(walked.items()):
            if rel in used:
                continue
            name = re.sub(r"_\d{5}_?$", "", info["path"].stem)
            if loads_re and not re.search(loads_re, name):
                continue
            rows.append(self.row(name, "share render, no job record", rel=rel))
        rows.sort(key=lambda x: (x["rendered_at"] or x["recorded_at"] or "", x["load"], x["output_file"] or ""))
        return rows, dict(left_out)

    def in_this_video(self, name: str, folder: Path, files: list[str]) -> bool:
        """A load belongs to this video by its job file's source, else by the copy its render's graph loads; with neither it is kept."""
        built = read_json(folder / f"{name}.render.built.json")
        jobd, _ = self.job_of(name, folder, built)
        source = (jobd or {}).get("job", {}).get("source")
        if source:
            return Path(str(source)).name in self.sources
        for rel in files:
            graph = rl.saved_graph(self.share / rel)
            ask = rl.asked(graph, self.sources) if graph else None
            if ask and ask.get("of"):
                return ask["of"] in self.sources
        return True


def csv_row(r: dict) -> dict:
    """A row's scalars and short renderings, in `CSV_COLUMNS`."""
    ref = r["reference_form"] or {}
    part = r["part"] or {}
    z = r["zoom"] or {}
    lay = r["laid"] or {}
    owner = r["owner"]
    return {"load": r["load"], "kind": r["kind"], "subject": r["subject"], "shot_first": r["shot_first"], "shot_last": r["shot_last"],
            "span_first": r["span_first"], "span_last": r["span_last"], "frames_given": r["frames_given"], "rendered_at": r["rendered_at"],
            "rendered_at_from": r["rendered_at_from"], "status": r["status"], "stopped_after": r["stopped_after"], "output_file": r["output_file"],
            "output_frames": r["output_frames"], "session": r["session"], "source_video": r["source_video"], "window_length": r["window_length"],
            "padding_before": r["padding_before"], "padding_after": r["padding_after"], "padding_kind": r["padding_kind"], "windows": r["windows"],
            "context_frames": r["context_frames"], "seed": r["seed"], "steps": r["steps"], "sampler": r["sampler"], "scheduler": r["scheduler"],
            "render_region": r["render_region"], "zoom_scale": None if z.get("scale") is None else round(z["scale"], 3),
            "zoom_box": json.dumps(z.get("box")) if z.get("box") else None, "reference_video": r["reference_video"], "iris": ref.get("iris"),
            "mouth": ref.get("mouth"), "closed_lip_line": ref.get("closed_lip_line"), "eyes_shut_by_word": json.dumps(ref["eyes_shut_by_word"]) if ref.get("eyes_shut_by_word") else None,
            "eyes_shut_by_rule": json.dumps(ref["eyes_shut_by_rule"]) if ref.get("eyes_shut_by_rule") else None, "still": r["still"],
            "parts": "+".join(part.get("parts") or []) or None, "region": part.get("region"), "margin": r["margin"], "feather": r["feather"],
            "keep_outs": "+".join(r["keep_outs"]) if r["keep_outs"] else None, "span_words": json.dumps(r["span_words"]) if r["span_words"] else None,
            "preview": r["preview"], "preview_commit": r["preview_commit"], "commit": r["commit"], "reference_commit": ref.get("commit"), "gate": r["gate"],
            "laid": json.dumps([x["frames"] for x in lay.get("now", [])]) if lay.get("now") else None,
            "owner_choices": " | ".join(f"{str(o['at'])[:16]} {o['said']}" for o in owner if o.get("what") == "choice") or None,
            "owner_words": " | ".join(f"{str(o['at'])[:16]} {str(o['said']).strip()}" for o in owner if o.get("what") != "choice") or None,
            "prompt_sha256": (r["prompt_sha256"] or "")[:16] or None, "prompt": r["prompt"],
            "cannot_see": "; ".join(x["field"] for x in r["cannot_see"]) or None}


def coverage(rows: list[dict]) -> dict:
    """How many rows each source filled each field of, and the fields most often missing: counts of rows, no grade."""
    filled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for field, source in r["sources"].items():
            filled[field][source] += 1
    missing = defaultdict(int)
    for r in rows:
        for x in r["cannot_see"]:
            missing[x["field"]] += 1
    return {"rows": len(rows), "by_kind": {k: sum(r["kind"] == k for r in rows) for k in KINDS},
            "by_status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows if r["status"]})},
            "by_padding_kind": {k: sum(r["padding_kind"] == k for r in rows) for k in PADDING_KINDS},
            "fields_by_source": {f: dict(sorted(s.items(), key=lambda kv: -kv[1])) for f, s in sorted(filled.items())},
            "most_often_missing": dict(sorted(missing.items(), key=lambda kv: -kv[1]))}


def carry(rows: list[dict], previous: list[dict], label: str) -> tuple[list[dict], int]:
    """The rows with an earlier history's rows added where this run has lost what they knew: a load whose job folder is gone from `data/` (the render
    is on the share, so this run has only its graph) takes its earlier row back, with the owner's lines and the laid facts as they are now; a load this run
    finds nothing of (its file and records both gone) is carried as it was. Each carried row says `carried_from`. A row this run reads whole (a chain
    load) is never replaced. The earlier history is a named file the caller hands in, never one found by a key."""
    key = lambda r: (r["load"], r["output_file"] or (r["records"] or {}).get("job_file"))
    now = {key(r): i for i, r in enumerate(rows)}
    out, n = list(rows), 0
    for p in previous:
        k = key(p)
        label_of = p.get("carried_from") or label
        if k not in now:
            out.append({**p, "carried_from": label_of})
            n += 1
        elif out[now[k]]["kind"] != "chain load" and p["kind"] == "chain load":
            cur = out[now[k]]
            out[now[k]] = {**p, "owner": cur["owner"], "laid": cur["laid"], "carried_from": label_of}
            n += 1
    out.sort(key=lambda x: (x["rendered_at"] or x["recorded_at"] or "", x["load"], x["output_file"] or ""))
    return out, n


def write(rows: list[dict], jsonl_path: Path, csv_path: Path) -> None:
    """The JSONL (the record) and the CSV made from it, whole."""
    Path(jsonl_path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    from_jsonl = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines()]
    with open(csv_path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        out.writeheader()
        out.writerows(csv_row(r) for r in from_jsonl)


def build(a: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    """The rows for the video the arguments name; (rows, left out, the sources the run read)."""
    data = Path(a.data) if a.data else REPO / "data"
    vdir = data / a.video
    if not vdir.is_dir():
        raise SystemExit(f"{vdir} is not a folder: the video's records are under data/<video>/ (--video, --data)")
    raw = a.share or os.environ.get("H3_OUTPUT_DIR")
    if not raw or not Path(raw).is_dir():
        raise SystemExit("the share is not a folder: give --share or set H3_OUTPUT_DIR to the server's output root")
    share = Path(raw)
    sources = {}
    if (vdir / "manifest.json").is_file() and not a.no_manifest:
        sources, _ = rl.copies_of(str(vdir))
    sources.update({s.partition("=")[0]: float(s.partition("=")[2] or 24) for s in a.source})
    if not sources:
        raise SystemExit("no copy of the video is named: its manifest is data/<video>/manifest.json, or give --source FILE[=RATE]")
    ledgers = []
    for path in a.ledger:
        d = read_json(Path(path))
        if not isinstance(d, dict) or not str(d.get("schema", "")).startswith("h3_render_ledger/"):
            raise SystemExit(f"--ledger {path}: not a render ledger's JSON (bench/render_ledger.py --json)")
        ledgers.append((Path(path).name, d))
    items = (read_json(vdir / "review" / "items.json") or {}).get("items") or []
    hist = History(a.video, data, share, sources, repo=data.parent, shots=read_json(vdir / "shots.json"), plan=read_json(vdir / "plan.json"),
                   verdicts=[v for _, v in jsonl(vdir / "verdicts" / "verdicts.jsonl")], answers=jsonl(vdir / "owner_answers.jsonl"),
                   items=items, laid=vdir / "laid.jsonl", ledgers=ledgers)
    rows, left_out = hist.rows(rl.walk(share), a.loads)
    carried = 0
    if a.carry:
        previous = [x for _, x in jsonl(Path(a.carry))]
        if not previous or any(x.get("schema") != SCHEMA for x in previous):
            raise SystemExit(f"--carry {a.carry}: not a render history's JSONL ({SCHEMA}, one row a line)")
        rows, carried = carry(rows, previous, Path(a.carry).name)
    read = {"share": str(share), "data": str(data), "copies": sources, "ledgers": [n for n, _ in ledgers],
            "carried": {"from": Path(a.carry).name, "rows": carried} if a.carry else None,
            "files": [str(p.relative_to(data)) for p in (vdir / "shots.json", vdir / "plan.json", vdir / "laid.jsonl", vdir / "owner_answers.jsonl",
                                                         vdir / "verdicts" / "verdicts.jsonl", vdir / "review" / "items.json") if p.is_file()]}
    return rows, left_out, read


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, help="the video's stem: its records are under data/<video>/")
    p.add_argument("--data", help="the records folder (default: this repo's data/)")
    p.add_argument("--share", help="the server's output root (default: H3_OUTPUT_DIR); its Video/ folder is walked")
    p.add_argument("--source", action="append", default=[], metavar="FILE[=RATE]", help="a copy of the video a render may load, by file name; repeatable")
    p.add_argument("--no-manifest", action="store_true", help="do not read the copies from data/<video>/manifest.json (with --source)")
    p.add_argument("--ledger", action="append", default=[], metavar="LEDGER.json",
                   help="an earlier render ledger (bench/render_ledger.py --json) whose record of a render is used where the job folder is gone; later ones win; repeatable")
    p.add_argument("--loads", metavar="REGEX", help="only the loads whose name matches (re.search)")
    p.add_argument("--carry", metavar="HISTORY.jsonl",
                   help="an earlier run's JSONL: a load whose job folder is gone from data/ takes its earlier row back (owner and laid as now), and a load nothing of is left is carried as it was")
    p.add_argument("--jsonl", required=True, help="the record: one object a render")
    p.add_argument("--csv", required=True, help="the same rows as a spreadsheet's")
    p.add_argument("--meta", help="a small JSON of what was read, what was left out and the coverage (default: beside the JSONL)")
    a = p.parse_args()
    rows, left_out, read = build(a)
    write(rows, Path(a.jsonl), Path(a.csv))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"schema": SCHEMA, "made": rl.utc(datetime.datetime.now().timestamp()), "tool_commit": commit, "video": a.video,
            "argv": sys.argv[1:], "read": read, "left_out": left_out, "coverage": coverage(rows)}
    Path(a.meta or str(Path(a.jsonl).with_suffix(".meta.json"))).write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    c = meta["coverage"]
    print(f"{c['rows']} rows: " + ", ".join(f"{k}: {n}" for k, n in c["by_kind"].items()) + "; " + ", ".join(f"{k}: {n}" for k, n in c["by_status"].items()))
    print("left out: " + (", ".join(f"{k}: {n}" for k, n in left_out.items()) or "nothing"))
    print("most often missing: " + (", ".join(f"{k} {n}" for k, n in list(c["most_often_missing"].items())[:8]) or "nothing"))
    print("wrote", a.jsonl, "and", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
