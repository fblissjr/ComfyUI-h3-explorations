#!/usr/bin/env python3
"""Every render of a video that is on the share, a shot and a subject at a time, with how each was made.

    <python> bench/render_ledger.py --shots SHOTS.json --plan PLAN.json --share OUTPUT_ROOT --records DATA_ROOT \\
        --source COPY.mov[=RATE] [--source ...] [--table TABLE.txt] [--verdicts VERDICTS.jsonl] [--answers ANSWERS.jsonl] \\
        [--items ITEMS.json] [--since YYYY-MM-DD] --json OUT.json --md OUT.md [--summary RECORD.md]

**What it buys.** By 2026-10-13 a video had some hundreds of clips in a dozen session folders and nobody, the
owner included, could say which renders existed for a shot, which of them could be made again, or which the
delivery stood on. A piece whose making cannot be said, or whose file and record disagree on frames, is not
one a delivery can be laid from; this prints that as a column for every render, so it is read and not recalled.

**Beside the two tools near it.** `bench/shot_coverage.py` says what a video still needs against its plan and one
delivery table; `bench/renders_page.py` is a page the owner plays clips from, made from the review page's
inventory (`renders.json`). Neither walks the share or reads a build record, so neither can say how a render
was made or whether its file is the one its record describes. This does, and plays nothing.

**What it reads, each a named file or folder handed in; nothing is found by a key.**

- `--share`: the server's output root. Every numbered video under its `Video/` is a candidate (`walk`); its
  frames are counted by ffprobe, and the graph ComfyUI saved in the picture file beside it says what was asked
  for: the copy the frames came from and where, the Masked Source's settings, the mask, part and motion files,
  the still, the seed, the text. A candidate is of this video when that copy is one of `--source` (by file name,
  with the copy's rate when it is not 24: a loader's start is in the copy's seconds).
- `--records`: the folder the job tool writes under (`data/`). `*_job/<load>.render.queued.json` names the files
  a queued render wrote; `<load>.render.built.json` beside it is the build record (the load, its inputs by
  path, the settings, the commit). A render is tied to a record by the record naming its file, and where a queue
  record names none, by its build record's output folder and load name, said on the render's line (`records_of`).
- `--shots`, `--plan`: the cuts and who is in each shot, and the table's names for the subjects
  (`bench/shot_coverage.py` has both schemas and its loaders are used).
- `--table`: a delivery table (`bench/assemble_delivery.py`'s reader); a render it lays says on which frames.
- `--verdicts`, `--answers`, `--items`: the owner's lines. A line is tied to a render by the render's file
  (`render`) or by its row being the render's load name or its page id; `--items` adds the page's own map from
  an item to a render's file. A line of `--answers` that `--verdicts` already copied is printed once.
- `--spots`: the owner's rows that were said of a JOINED clip at stated frames (a file: the clip's name, the
  table it was built from, each row's frames). Such a line is printed under every render that table laid on
  those frames, as said of the joined clip there and never as a verdict on the render alone.
- `--inputs`: the server's input root, so a record's still can be looked for.
- The loads stopped before a render (`stopped`): the note a card session writes beside a load's job file under
  `--records`, `<load>.STOPPED*.txt` or `.md`, read whole, of a job whose source is one of `--source`
  (`stopped_loads`). The owner's provenance page takes them from here, never from a typed list (2026-10-14).
- `--copies-of`: the video's analysis folder. The copies a render may load are then read from its manifest (the
  source and every copy the video has or has had, each at its own rate; `clip_folder.py`), so a copy made since
  the last run is not a name somebody has to remember to type. A copy on another frame clock is left out and
  named in the record: a render from it cannot be placed by its seconds.

The JSON keeps the command it was made by (`command`: the arguments, and one line a shell takes as it stands),
so the table made from it (`bench/delivery_table.py`) prints it and a second run types nothing. The tracked
summary does not: the command holds this machine's paths.

**What it prints: a section a shot and a subject, every render on those frames, oldest first, each with**
its file and frames (the record's, the file's by ffprobe, the saved graph's); ITS MAKING (`MAKING`: a build
record whose figures the file agrees with; one it disagrees with, and how; a build record for its load whose
queue record names no file and no outcome, as the job tool wrote them before 2026-10-11, tied by the output
folder and the load's name; or no record), with the record's inputs still on disk; its form (what is replaced, read from the capture's own part classes or the graph; the
reference video, read from the records its tools wrote beside it, back to the body model; what the text says
the video is); who made it, by the folder it is in, and when; where the table lays it; and the owner's lines.
Then what could not be placed: numbered videos with no saved graph, records whose render is gone, the owner's
lines tied to no render file.

**What `agrees` means, and all it means.** The queued run ended `success` and names this file; the file holds
the frames the load says (a load shorter than its window writes the load's frames); the graph saved beside
the file asks for the record's first frame and count; the file was not written before the record was queued.
Each is a fact about files. A load that continued a stored window writes fewer frames than it loads, and the
record does not say which: that is a disagreement here, by design.

**What it cannot see.** Whether a render is any good, or better than another: nothing here ranks. Whether a
record's inputs are the bytes that ran (a path that still exists may have been rewritten). Which source frames
a file holds when its count differs from its record. A render whose picture file is gone cannot be placed and
is listed apart. Who made a render beyond the folder it sits in.

The frame numbers are the copy's (`clip_folder.py`, "One clock").
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: 2 (2026-10-13): a render's form also carries `region` and its reference `has`, as fields a tool can read
#: without parsing a sentence; an owner's line says whether it is a choice (a button) or words; and the ledger
#: says the share it walked (`bench/delivery_table.py` reads all of these). Since 2026-10-13, late, a form also
#: carries `part_named` (the parts a load's job named as its own, or null): a field added, the schema's number kept,
#: since every reader of 2 reads a ledger with it. Since 2026-10-14 a render also carries `defaults_carried` (its job's
#: `job.defaults`, or null for a job written before them), the same way. Since 2026-10-15 a form also carries `reference_none`
#: (a job made with no reference video on a person's word, `shot_to_load.py job --motion none`, with its why; or null).
SCHEMA = "h3_render_ledger/2"
#: A reference with none of these steps in its records, for a form's `has`.
NO_STEPS = {"cut_to_the_head": False, "face_points": False, "lid_marks": False}
#: A numbered output of the server, and the files beside a render that are not renders themselves.
NUMBERED = re.compile(r"_\d{5}_?\.mp4$")
NOT_A_RENDER = ("_with_mask", "-audio", "_zoomed", "_motion_video_")
SAMPLERS = ("MiniMaxH3AudioFreezeSong", "SamplerCustom", "KSampler")
#: The states of a render's making. Each is a fact about files, read as `agreement` says; none judges a render.
MAKING = ("recorded, the file agrees", "recorded, THE FILE DISAGREES",
          "a build record for its load, WHOSE QUEUE RECORD NAMES NO FILE AND NO OUTCOME", "NO BUILD RECORD")
#: Nodes a link is followed through to the file behind it: they change a picture's kind, not what it is of.
PASSES_THROUGH = ("ImageToMask", "MaskComposite")
#: The part model's classes that make a part a face's, and the one that makes it hair's.
FACE_CLASS, HAIR_CLASS = "Face_Neck", "Hair"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def utc(stamp: float) -> str:
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_utc(text: str | None) -> str | None:
    """A record's time as UTC. The job tool writes its times with their offset; one with none is left as it is."""
    if not text:
        return None
    try:
        t = datetime.datetime.fromisoformat(text)
    except ValueError:
        return text
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t.tzinfo else text


def walk(share: Path, since: str | None = None) -> list[Path]:
    """Every numbered video under the share's `Video/` that could be a render, oldest path first."""
    floor = datetime.datetime.fromisoformat(since).replace(tzinfo=datetime.timezone.utc).timestamp() if since else None
    out = []
    for path in sorted((share / "Video").rglob("*.mp4")):
        if not NUMBERED.search(path.name) or any(word in path.name for word in NOT_A_RENDER) or any(p.endswith("_windows") for p in path.parts):
            continue
        if floor is None or path.stat().st_mtime >= floor:
            out.append(path)
    return out


def count_frames(path: Path) -> int | None:
    """A video's frames by ffprobe (its packets, counted), or None when it does not read."""
    got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    text = got.stdout.strip().split("\n")[0].strip(",")
    return int(text) if got.returncode == 0 and text.isdigit() else None


def saved_graph(mp4: Path) -> dict | None:
    """The graph ComfyUI saved in the picture beside a video, or None when there is none to read."""
    png = mp4.with_suffix(".png")
    if not png.is_file():
        return None
    from PIL import Image
    try:
        return json.loads(Image.open(png).info.get("prompt") or "")
    except (ValueError, OSError):
        return None


def saved_relaid(mp4: Path) -> dict | None:
    """The `relaid` block in the picture beside a re-laid piece (`recomposite_piece.py`), or in `<stem>.recomposite.json` beside it; None for a render."""
    png = mp4.with_suffix(".png")
    if png.is_file():
        from PIL import Image
        try:
            raw = Image.open(png).info.get("relaid")
            if raw:
                got = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(got, dict) and got.get("from"):
                    return got
        except (ValueError, OSError):
            pass
    rec = mp4.with_name(f"{mp4.stem}.recomposite.json")
    if rec.is_file():
        try:
            d = json.loads(rec.read_text())
            if isinstance(d.get("relaid"), dict) and d["relaid"].get("from"):
                return d["relaid"]
        except (ValueError, OSError):
            pass
    return None


def _upstream_video(graph: dict, ref, depth: int = 0, through=None) -> dict | None:
    """The video loader a link leads back to. With `through`, only through those node classes (a mask conversion,
    a composite): a link that comes from any other node was made in the run and is no file. Without it, through
    whatever sits between, which is how the copy a render's frames came from is found."""
    node = graph.get(str(ref[0])) if isinstance(ref, list) and ref else None
    if node is None or depth > 8:
        return None
    if isinstance(node["inputs"].get("video"), str):
        return node
    if through is not None and node["class_type"] not in through:
        return None
    for v in node["inputs"].values():
        if isinstance(v, list) and len(v) == 2:
            got = _upstream_video(graph, v, depth + 1, through)
            if got:
                return got
    return None


def _wired_file(graph: dict, ref) -> str | None:
    """What a Masked Source input is wired to: a file's path, or `made in the graph by <class>` when a node of the
    run made it (an older graph tracked its subject and posed its body model inside the render's own graph)."""
    if not isinstance(ref, list):
        return None
    node = _upstream_video(graph, ref, through=PASSES_THROUGH)
    if node is not None:
        return node["inputs"]["video"]
    direct = graph.get(str(ref[0]))
    return f"made in the graph by {direct['class_type']}" if direct else None


def _loader_span(node: dict, rate: float) -> tuple[int | None, int | None]:
    """The first frame of the copy and the count a loader was asked for. The job graphs set a start a fraction of a
    frame early so the loader lands on the frame; a hand-made graph's exact start comes to the same frame."""
    i = node["inputs"]
    if "start_time" in i:
        return math.ceil(float(i["start_time"]) * rate - 1e-3), i.get("frame_load_cap")
    if "skip_first_frames" in i:
        return int(i["skip_first_frames"]), i.get("frame_load_cap")
    return None, i.get("frame_load_cap")


def asked(graph: dict, sources: dict[str, float]) -> dict:
    """What a saved graph asked for: {"of": the copy's file name} always; the rest when the copy is one of `sources`."""
    if not any(n["class_type"].startswith(SAMPLERS) for n in graph.values()):
        return {"of": None, "why": "no sampler in the saved graph: not a render"}
    masked = next((n for n in graph.values() if n["class_type"] == "MiniMaxH3MaskedSource"), None)
    src = _upstream_video(graph, masked["inputs"].get("frames")) if masked else None
    if src is None:                    # not a masked render: any loader of one of the copies still places it
        src = next((n for n in graph.values() if isinstance(n["inputs"].get("video"), str) and Path(n["inputs"]["video"]).name in sources), None)
    if src is None:
        return {"of": None, "why": "no video is loaded in the saved graph"}
    copy = Path(src["inputs"]["video"]).name
    if copy not in sources:
        return {"of": copy}
    first, frames = _loader_span(src, sources[copy])
    m = masked["inputs"] if masked else {}
    files = {key: _wired_file(graph, m.get(key)) for key in ("mask", "parts", "others", "keep", "motion_video", "motion_video_2")}
    song = next((n["inputs"] for n in graph.values() if n["class_type"].startswith("MiniMaxH3AudioFreezeSong")), {})
    texts = [v for n in graph.values() for k, v in n["inputs"].items() if k in ("prompt", "text") and isinstance(v, str) and len(v) > 80]
    return {"of": copy, "first": first, "frames": frames, "masked": bool(masked),
            "replace": m.get("replace"), "motion_reference": m.get("motion_reference"), "render_region": m.get("render_region"),
            "margin": m.get("grow_pixels"), "files": files,
            "stills": sorted({v for n in graph.values() for k, v in n["inputs"].items() if k == "image" and isinstance(v, str) and v}),
            "seed": song.get("seed"), "window": song.get("window_frames"), "context": song.get("context_frames"),
            # the noise a later window is shown its kept frames at; None where the graph names none, the song node's own (2026-10-15)
            "context_noise": song.get("context_noise") if isinstance(song.get("context_noise"), (int, float)) else None,
            "continue_from": song.get("continue_from") if isinstance(song.get("continue_from"), str) and song.get("continue_from") else None,
            "text": max(texts, key=len) if texts else None}


def load_records(records: Path, repo: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    """The job tool's records of queued renders: {a file's path under the share: [record, ...]} and every record.

    A record is the pair `<load>.render.queued.json` (what ran: the files it wrote, its status, its gate and look)
    and `<load>.render.built.json` (what was built: the load, its inputs, its settings, the commit)."""
    by_file, every = defaultdict(list), []
    for queued in sorted(records.glob("*_job/*.render.queued.json")):
        built = queued.with_name(queued.name.replace(".queued.json", ".built.json"))
        try:
            q = json.loads(queued.read_text())
            b = json.loads(built.read_text()) if built.is_file() else None
        except ValueError:
            continue
        for key in ("gate", "look"):                 # an older record holds these as text
            if isinstance(q.get(key), str):
                try:
                    q[key] = json.loads(q[key])
                except ValueError:
                    q[key] = {"said": q[key]}
        where = lambda p: str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p)
        rec = {"queued_file": where(queued), "built_file": where(built) if b is not None else None, "queued": q, "built": b,
               "files": [f for f in q.get("files") or [] if NUMBERED.search(f) and not any(word in f for word in NOT_A_RENDER)],
               "folder": (b or {}).get("output_folder"), "load_name": ((b or {}).get("load") or {}).get("name")}
        every.append(rec)
        for f in rec["files"]:
            by_file[f].append(rec)
    return by_file, every


def stopped_loads(records: Path, repo: Path, sources: dict) -> list[dict]:
    """The loads stopped before a render, from the note the card writes beside a load's job file: `<load>.STOPPED*`
    (`.txt` or `.md`) with `<load>.json` beside it, whose job's source is one of this video's copies. The note is
    read whole and nothing in it is parsed; the load's subject and frames are its job file's. A note with no job
    file beside it (a copy in the load's capture folder, `STOPPED.txt`) is not read: the one beside the job is.
    Until 2026-10-14 the owner's page took the stopped loads from a list a session typed."""
    where = lambda p: str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p)
    out = []
    for note in sorted(records.glob("*/*.STOPPED*")):
        load = note.name.split(".STOPPED")[0]
        job_file = note.with_name(f"{load}.json")
        if note.suffix not in (".txt", ".md") or not job_file.is_file():
            continue
        try:
            d = json.loads(job_file.read_text())
        except ValueError:
            continue
        if Path(str((d.get("job") or {}).get("source") or "")).name not in sources:
            continue
        one = next((x for x in d.get("loads") or [] if x.get("name") == load), (d.get("loads") or [{}])[0])
        span = [one["first"], one["first"] + one["frames"] - 1] if one.get("first") is not None and one.get("frames") else None
        out.append({"load": load, "subject": one.get("subject"), "frames": span, "job_file": where(job_file), "note": where(note),
                    "note_written": utc(note.stat().st_mtime), "note_says": note.read_text().strip()})
    return sorted(out, key=lambda x: (x["frames"] or [10 ** 9])[0])


def records_of(rel: str, share: Path, by_file: dict, every: list[dict]) -> tuple[list[dict], bool]:
    """The records of a render's file, and whether they NAME it. A queue record from before 2026-10-11 names no file
    and no outcome; its build record names the output folder and the load, and a numbered file of that load's name
    in that folder is taken as its render, said as that."""
    named = by_file.get(rel, [])
    if named:
        return named, True
    path = share / rel
    loose = [rec for rec in every if not rec["files"] and rec["folder"] and rec["load_name"] and Path(rec["folder"]) == path.parent
             and re.sub(r"_\d{5}_?$", "", path.stem) == rec["load_name"]]
    return loose, False


def reference_form(path: str | None, depth: int = 0) -> dict:
    """What a reference (motion) video is, read from the records its tools wrote beside it, back to the body model.

    `bench/head_reference.py`, `crop_reference_to_head.py`, `draw_face_points.py` and `draw_lid_marks.py` each
    write `<video>.json` beside the video with what they made it from. This follows `made_from` (or `inputs`)
    back and says each step in the tool's own fields; a video with no record beside it ends the chain and is
    said as that. Nothing is read from a file's name."""
    if not path:
        return {"steps": [], "has": None, "says": "no reference video"}
    if path.startswith("made in the graph by "):
        return {"steps": [], "has": dict(NO_STEPS), "says": f"{path}: not a file, so no record of it but the saved graph"}
    video = Path(path)
    record = video.with_suffix(".json")
    if not video.is_file():
        return {"steps": [], "has": dict(NO_STEPS), "says": f"the reference video is not on disk ({video.name})"}
    if not record.is_file() or depth > 4:
        beside = "a saved graph beside it (a node's own output: a preview's body model, as it came)" if video.with_suffix(".png").is_file() else "no record beside it"
        return {"steps": [], "has": dict(NO_STEPS), "says": f"{video.name}: {beside}", "ends_at": str(video)}
    try:
        r = json.loads(record.read_text())
    except ValueError:
        return {"steps": [], "has": dict(NO_STEPS), "says": f"{video.name}: its record does not read", "ends_at": str(video)}
    steps, has = [], dict(NO_STEPS)
    drawn = r.get("drawn") or (r if "parts_left_out" in r and "line_width_px" in r else None)
    if "box" in r:
        has["cut_to_the_head"] = True
        scale = r.get("scale")
        steps.append(f"cut to one box round the head ({r['box']}" + (f", enlarged {scale[0]:g} times" if isinstance(scale, list) and scale else "") + ")")
    if drawn is not None:
        has["face_points"] = True
        steps.append("the face points drawn as lines" + (f" (left out: {', '.join(drawn['parts_left_out'])})" if drawn.get("parts_left_out") else "")
                     + ("" if not drawn.get("a_dot_at_every_point") else ", a dot at every point"))
    if "lid_gap_px_median" in r or "a_mark_on_the_canvas_px" in r:
        has["lid_marks"] = True
        steps.append("marks drawn on the lids" + (f" (grown {r['grown']:g} times)" if r.get("grown") else ""))
    made = r.get("made_from") or r.get("inputs") or {}
    before = reference_form(made.get("body_model_video") or made.get("reference_video"), depth + 1) if isinstance(made, dict) else {"steps": [], "says": ""}
    chain = before["steps"] + steps
    has = {k: bool(v or (before.get("has") or {}).get(k)) for k, v in has.items()}      # a step of an earlier record of the chain counts
    return {"steps": chain, "has": has, "record": str(record), "made_by": r.get("made_by"), "made_at": r.get("made_at"), "frames": r.get("frames"),
            "passed": r.get("passed"), "ends_at": before.get("ends_at"),
            "says": "the body model" + ("".join(", then " + s for s in chain) if chain else ", as it came")
                    + f" [from the records beside {video.name}" + (f"; the chain ends at {Path(before['ends_at']).name}" if before.get("ends_at") else "") + "]"}


def part_classes(capture: Path | None, label: str | None) -> list[str] | None:
    """The classes a capture's manifest says a subject's part is made of, or None when it does not say."""
    if capture is None or not label or not (capture / "manifest.json").is_file():
        return None
    try:
        manifest = json.loads((capture / "manifest.json").read_text())
    except ValueError:
        return None
    for s in manifest.get("subjects", []):
        if s["label"] == label:
            return next((seen.get("part_is_made_of") for seen in s.get("sightings", []) if seen.get("part_is_made_of")), None)
    return None


def left_by_the_load(capture: Path | None, label: str | None, span: list[int] | None, run: str | None = None) -> dict | None:
    """The frames inside a render's own span that its load LEFT AS THE ORIGINAL: the node was given no mask for the
    subject there, so it had nothing to replace. None when there is no capture, no such record or no span. In no
    ledger until 2026-10-13, late: they were known only from the assembler's flags after a clip was built, and the
    owner saw one on playback as "he turns back".

    Where the capture holds the run's own mask (`runs/<run>/region.npz`), the frames and their causes are
    `frame_coverage.emptied`'s, the one home of the rule: every frame the node carried nothing on, whether a person's
    `drop=`, the floor, the grade or an empty saved part emptied it (`by_cause`). Until 2026-10-16 this read only the
    manifest's `parts_emptied_on_source_frames`, which is `drop=` alone, and missed the floor's and the grade's frames
    (the kitchen, and 1419d's 1421 and 1449-1459, `internal/claude/2026-10-15_flip_survey/README.md`, "Gaps"). A
    capture with no run mask falls back to that list, said in `read_from`."""
    if capture is None or not label or not span or not (capture / "manifest.json").is_file():
        return None
    fell_back = None
    if run and not (capture / "runs" / run / "region.npz").is_file():
        fell_back = f"the capture holds no runs/{run}/region.npz"
    if run and fell_back is None:
        fc = _module("frame_coverage")
        try:
            e = fc.emptied(fc.Capture(capture), run, label, span)
        except (fc.Failure, ValueError, KeyError) as x:
            e, fell_back = None, f"the run's mask could not be read ({x})"
        if e is not None:
            inside = fc.runs(e["frames"])
            by_cause: dict[str, list[int]] = {}
            for f, cause in e["frames"].items():
                by_cause.setdefault(cause, []).append(f)
            return {"source_frames": inside, "frames": len(e["frames"]),
                    "by_cause": {c: fc.runs(fs) for c, fs in sorted(by_cause.items())},
                    "not_read": fc.runs(e["not_read"]),
                    "read_from": f"the run's carried mask ({e['read_from']}): the frames on which the node was given no mask for this subject, "
                                 "with the cause from the capture's manifest and part_grades.json (bench/frame_coverage.py::emptied)"}
    try:
        manifest = json.loads((capture / "manifest.json").read_text())
    except ValueError:
        return None
    for s in manifest.get("subjects", []):
        if s["label"] == label:
            runs = next((seen.get("parts_emptied_on_source_frames") for seen in s.get("sightings", []) if seen.get("parts_emptied_on_source_frames") is not None), None)
            if runs is None:
                return None
            inside = [[max(a, span[0]), min(b, span[1])] for a, b in runs if a <= span[1] and b >= span[0]]
            return {"source_frames": inside, "frames": sum(b - a + 1 for a, b in inside),
                    "read_from": "the capture's manifest: the frames on which it emptied this subject's part (parts_emptied_on_source_frames)"
                                 + (f"; a person's drop= alone, not the floor's or the grade's, since {fell_back}" if fell_back else "")}
    return None


def part_named(job_file: str | None, repo: Path, label: str | None) -> dict | None:
    """The parts a load's JOB named as its own, some of its subject's plan parts (`shot_to_load.py job --parts`), with
    whose word, read from the job file its build record names; None when the job names none or cannot be read.

    Why the ledger carries it (2026-10-13): a render of the hair alone has a capture whose part is made of the hair
    class only, so its region reads `a part` and not the plan's `face and hair`, and a table that holds a render to
    the plan's region would pass it over and lay the source where the owner asked for the new hair. That the part
    was NAMED for the load, and by whom, is in the job file and nowhere in the capture."""
    if not job_file or not label:
        return None
    path = Path(job_file) if Path(job_file).is_absolute() else repo / job_file
    try:
        job = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for s in (job.get("job") or {}).get("subjects") or []:
        if s.get("label") == label and s.get("parts_of_this_load"):
            return {"parts": list(s["parts_of_this_load"]), "by": s.get("parts_by"), "read_from": str(job_file)}
    return None


def reference_none(job_file: str | None, repo: Path) -> dict | None:
    """A load made with no reference video on a person's word (`shot_to_load.py job --motion none`, the lead's ruling of 2026-10-15 on
    second_hp_5461, her hair alone while her face is hidden): the job's `reference` says form `none` with its why. None when the job
    says otherwise or cannot be read. The reference itself is read as always (`reference_form`): this says only why there is none."""
    if not job_file:
        return None
    path = Path(job_file) if Path(job_file).is_absolute() else repo / job_file
    try:
        said = (json.loads(path.read_text()).get("job") or {}).get("reference") or {}
    except (OSError, ValueError):
        return None
    if said.get("form") != "none":
        return None
    return {"why": said.get("why"), "read_from": str(job_file)}


def defaults_of(job_file: str | None, repo: Path) -> list[dict] | None:
    """THE DEFAULTS THE LOAD CARRIED (`job.defaults`, written by `shot_to_load.py job` since 2026-10-14: a line each, with where it
    came from), read from the job file its build record names; None for a job written before then, or one that cannot be read."""
    if not job_file:
        return None
    path = Path(job_file) if Path(job_file).is_absolute() else repo / job_file
    try:
        return (json.loads(path.read_text()).get("job") or {}).get("defaults")
    except (OSError, ValueError):
        return None


def region_of(carried: str | None, classes: list[str] | None) -> str | None:
    """What a render replaces as one of the plan's regions (`whole`, `face`, `face and hair`), `a part` when the
    part's classes are not a face's or were not read, or None when the files do not say what was carried."""
    if carried in ("track", "whole subject"):
        return "whole"
    if carried in ("held", "the wired parts"):
        if classes is not None and FACE_CLASS in classes:
            return "face and hair" if HAIR_CLASS in classes else "face"
        return "a part"
    return None


def region_form(carried: str | None, classes: list[str] | None, read_from: str) -> str:
    """What a render replaces, in the plan's words where the files say it."""
    region = region_of(carried, classes)
    if region == "whole":
        return f"whole person [{read_from}]"
    if region == "a part":
        return (f"a part; which classes is not in what was read [{read_from}]" if classes is None
                else f"a part made of {', '.join(classes)} [{read_from}]")
    if region is not None:
        return region + f" (the capture's part is made of {', '.join(classes or [])}) [{read_from}]"
    return f"not known ({carried!r}) [{read_from}]"


def text_says(text: str | None) -> str | None:
    """What a load's text says each reference video is: the definitions that name a `<Video N>`, whole."""
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if "<Video " in line and line.strip().startswith("<Subject")]
    return " / ".join(lines) if lines else "the text names no <Video N> in a subject's definition"


def _capture_of(path: str | None) -> tuple[Path | None, str | None]:
    """The capture folder and subject a mask file sits in, when it is `<capture>/subjects/<label>/<file>`."""
    p = Path(path) if path else None
    if p is not None and p.parent.parent.name == "subjects":
        return p.parent.parent.parent, p.parent.name
    return None, None


def agreement(file_frames: int | None, written: float, graph: dict | None, rec: dict) -> dict:
    """A render's file against its record, fact by fact; `disagrees` lists the ones that do not hold."""
    q, b = rec["queued"], rec["built"]
    load = (b or {}).get("load") or {}
    said = []
    if b is None:
        said.append("the queue record has no build record beside it")
    if q.get("status") != "success" and (q.get("status") is not None or q.get("files")):
        said.append(f"the queued run's status is {q.get('status')!r}")
    if load and file_frames is not None and file_frames != load.get("frames"):
        said.append(f"the record's load is {load.get('frames')} frames from {load.get('first')}; the file holds {file_frames}"
                    + ("; the load continues a stored window (`continue_from`), whose kept frames are loaded and not written, and the "
                       "record does not say which source frames the file holds" if load.get("continue_from") else ""))
    if file_frames is None:
        said.append("the file's frames could not be counted")
    # a one-window load shorter than its window loads real source frames before and after it as context, tossed after the
    # decode (2026-10-21, 79831be9): its graph asks for the padded span, which the build record keeps as
    # `load_against_its_window` (`bench/masked_job.py`, `context_before` / `context_after`)
    pad = ((b or {}).get("what_this_clips_preview_measured") or {}).get("load_against_its_window") or {}
    before, after = int(pad.get("context_before") or 0), int(pad.get("context_after") or 0)
    span = ((load["first"] - before, load["frames"] + before + after) if (before or after) and None not in (load.get("first"), load.get("frames"))
            else (load.get("first"), load.get("frames")))
    if graph and load and graph.get("of") and (graph.get("first"), graph.get("frames")) != span:
        said.append(f"the graph saved beside the file asks for {graph.get('frames')} frames from {graph.get('first')}; the record's load "
                    f"is {load.get('frames')} from {load.get('first')}"
                    + (f", with {before} context frame(s) before and {after} after" if before or after else ""))
    queued_at = q.get("queued_at")
    if queued_at:
        try:
            t = datetime.datetime.fromisoformat(queued_at)
            if t.tzinfo and written < t.timestamp():
                said.append(f"the file was written {utc(written)}, before the record was queued ({as_utc(queued_at)}): it is not that run's")
        except ValueError:
            pass
    return {"making": MAKING[1] if said else MAKING[0], "disagrees": said}


def inputs_on_disk(rec: dict, repo: Path, inputs: Path | None = None) -> dict:
    """The record's inputs by path that are still files, and the two folders it names. A path that exists may have
    been rewritten since: this counts presence, nothing more."""
    b = rec["built"] or {}
    paths: list[str] = []

    def take(v):
        if isinstance(v, str):
            paths.append(v)
        elif isinstance(v, list):
            if v and isinstance(v[0], str) and (len(v) == 1 or isinstance(v[1], int)):
                paths.append(v[0])
            else:
                for x in v:
                    take(x)
        elif isinstance(v, dict):
            for x in v.values():
                take(x)
    take(b.get("files") or {})
    real = lambda p: Path(p) if Path(p).is_absolute() else repo / p
    missing = [p for p in paths if not real(p).is_file()]
    folder = lambda key: None if not b.get(key) else real(b[key]).is_dir()
    still = (b.get("load") or {}).get("still")
    # a still is named under the server's input folder; without that folder handed in, nothing is said of it
    still_there = None if not still or inputs is None else (Path(still) if Path(still).is_absolute() else inputs / still).is_file()
    return {"named": len(paths), "missing": missing, "capture_folder": folder("capture"), "windows_folder": folder("windows_folder"),
            "still": still, "still_is_a_file": still_there}


def on_a_joined_clip(rel: str, spots: list[dict], verdicts: list[dict]) -> list[dict]:
    """The owner's lines that were said of a JOINED clip at stated frames, under each render that clip's table laid
    on those frames: never a verdict on the render alone, and printed as what it is.

    `spots` are files handed in (`--spots`): {"clip": the joined clip's name, "table": the table it was built from,
    "rows": that table read, "spots": [{"row", "first", "last"}]}. A row of the verdicts file that is a spot is tied
    to every piece the table lays on frames the spot covers."""
    out = []
    for given in spots:
        for spot in given["spots"]:
            laid = [t for t in given["rows"] if str(t["piece"]).endswith(rel) and t["first"] <= spot["last"] and t["last"] >= spot["first"]]
            if not laid:
                continue
            for v in verdicts:
                said = v.get("chosen") if v.get("what") == "choice" else v.get("words")
                if v.get("row") == spot["row"] and said:
                    out.append({"at": v.get("at"), "frame": v.get("frame"), "said": said, "what": v.get("what"), "from": f"verdicts ({v.get('source')})", "row": v.get("row"),
                                "on_the_joined_clip": given["clip"], "at_frames": [spot["first"], spot["last"]], "through_the_table": given["table"]})
    return out


def on_a_joined_item(rel: str, joined: list[dict], verdicts: list[dict]) -> list[dict]:
    """The owner's lines on a JOINED ITEM of the review page (`bench/shot_review_clips.py items`: a whole clip of a span,
    every piece that stood laid), under each render the item's own table laid, marked `the_whole_clip`: its KEEPS only. The
    lead's rulings of 2026-10-14: the owner judges the final cut whole, so a KEEP there counts as a keep on each piece it lays
    (`delivery_table.last_choice`); a "wrong", a "not sure" or words there are the owner's word on the SHOT and are no piece's
    own (a reader saw "See hair right above his eye", said of the third subject, under the lead's piece): they are kept
    whole under the clip in the ledger's `joined_clips` (`lines_on_joined_clips`).

    `joined` is [{"clip": the item's clip name, "row": the item's id, "table": its table's file, "rows": that table's rows}]."""
    out = []
    for given in joined:
        laid = [t for t in given["rows"] if str(t.get("file", "")).endswith(rel) or t.get("name") == Path(rel).stem.rsplit("_", 1)[0]]
        for t in laid:
            for v in verdicts:
                said = v.get("chosen") if v.get("what") == "choice" else v.get("words")
                if v.get("row") == given["row"] and said and v.get("what") == "choice" and v.get("chosen") in keeps():
                    out.append({"at": v.get("at"), "frame": v.get("frame"), "said": said, "what": v.get("what"), "from": f"verdicts ({v.get('source')})",
                                "row": v.get("row"), "on_the_joined_clip": given["clip"], "the_whole_clip": True, "at_frames": [t["first"], t["last"]],
                                "through_the_table": given["table"]})
    return out


def keeps() -> tuple:
    """The buttons that are a keep: the delivery table's (`delivery_table.KEEP`), one copy, read when needed."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import delivery_table
    return tuple(delivery_table.KEEP)


def lines_on_joined_clips(joined: list[dict], verdicts: list[dict]) -> list[dict]:
    """Each joined item's own lines whole, clicks and words, oldest first, with the pieces its table laid: the owner's word on
    the shot (a keep among them is also under each piece, `on_a_joined_item`)."""
    out = []
    for given in joined:
        mine = [{"at": v.get("at"), "frame": v.get("frame"), "said": v.get("chosen") if v.get("what") == "choice" else v.get("words"), "what": v.get("what"),
                 "from": f"verdicts ({v.get('source')})"} for v in verdicts if v.get("row") == given["row"]]
        mine = sorted((m for m in mine if m["said"]), key=lambda m: str(m["at"]))
        if mine:
            out.append({"clip": given["clip"], "row": given["row"], "table": given["table"],
                        "lays": [{"name": t.get("name"), "subject": t.get("subject"), "frames": [t["first"], t["last"]]} for t in given["rows"]], "lines": mine})
    return out


def joined_items(folders: list[str], verdicts: list[dict]) -> list[dict]:
    """Every joined item the verdicts name whose table is in one of the folders handed in (`--joined-tables`, the
    folder `bench/shot_review_clips.py build` writes `t_<clip>.json` into); found by the item's own id, never by frames."""
    out = []
    for row in sorted({str(v.get("row")) for v in verdicts if str(v.get("row", "")).endswith("_stacked") and "_joined_" in str(v.get("row"))}):
        clip = row.removesuffix("_stacked")
        table = next((Path(f) / f"t_{clip}.json" for f in folders if (Path(f) / f"t_{clip}.json").is_file()), None)
        if table is not None:
            out.append({"clip": clip, "row": row, "table": str(table), "rows": json.loads(table.read_text())["rows"]})
    return out

def owner_lines(rel: str, ids: set[str], verdicts: list[dict], answers: list[tuple[int, dict]], item_of: dict[str, str]) -> list[dict]:
    """The owner's lines on a render, oldest first: by the render's file, by its ids as a row, or by a page item
    that names the file. An answers line the verdicts file already copied is taken once, from the verdicts file."""
    copied = {v.get("source") for v in verdicts}
    mine = lambda row, render: (render and rel.endswith(render)) or row in ids or item_of.get(row) == rel
    out = [{"at": v.get("at"), "frame": v.get("frame"), "said": v.get("chosen") if v.get("what") == "choice" else v.get("words"),
            "what": v.get("what"), "from": f"verdicts ({v.get('source')})", "row": v.get("row")} for v in verdicts if mine(v.get("row"), v.get("render"))]
    out += [{"at": a.get("at"), "frame": a.get("frame"), "said": a.get("chosen") if a.get("what") == "choice" else a.get("words"),
             "what": a.get("what"), "from": f"answers.jsonl:{n}", "row": a.get("row")} for n, a in answers
            if a.get("who") == "owner" and f"answers.jsonl:{n}" not in copied and mine(a.get("row"), None)]
    return sorted((o for o in out if o["said"]), key=lambda o: str(o["at"]))


def ledger(shots: dict, plan: dict, share: Path, records: Path, repo: Path, sources: dict[str, float], table_rows: list[dict],
           verdicts: list[dict], answers: list[tuple[int, dict]], items: list[dict], since: str | None = None,
           frames_of=count_frames, alias=None, spots: list[dict] | None = None, inputs: Path | None = None, joined: list[dict] | None = None) -> dict:
    """The whole ledger as data. `frames_of` counts a video's frames (ffprobe; a check hands in its own)."""
    alias = alias or (lambda label: label)
    by_file, every = load_records(records, repo)
    item_of = {i["id"]: i["render"] for i in items if i.get("render")}
    video_folder = Path(shots.get("video") or "").stem          # `Video/<video>/<session>/...` and `Video/<session>/...` both occur
    renders, no_graph, other, not_renders = [], [], defaultdict(int), defaultdict(int)
    tied: set[int] = set()
    for path in walk(share, since):
        rel = str(path.relative_to(share))
        graph = saved_graph(path)
        relaid = saved_relaid(path)
        if relaid is not None:
            from_p = Path(str(relaid.get("from") or ""))
            from_rel = str(from_p.relative_to(share)) if from_p.is_absolute() and from_p.is_relative_to(share) else str(relaid.get("from") or "")
            relaid = {**relaid, "from": from_rel}
        recs, named = records_of(rel, share, by_file, every)
        if not recs and relaid is not None and relaid.get("from"):
            recs, named = records_of(str(relaid["from"]), share, by_file, every)
        tied.update(id(rec) for rec in recs)
        if graph is None and not recs:
            no_graph.append(rel)
            continue
        ask = asked(graph, sources) if graph is not None else None
        if ask is not None and ask.get("of") not in sources and not recs:
            (other if ask.get("of") else not_renders)[ask.get("of") or ask.get("why")] += 1
            continue
        stat = path.stat()
        n = frames_of(path)
        parts = [x for x in Path(rel).parts[1:-1] if x != video_folder]
        row = {"file": rel, "name": re.sub(r"_\d{5}_?$", "", path.stem), "session_folder": parts[0] if parts else None,
               "written": utc(stat.st_mtime), "file_frames": n,
               "graph": {k: v for k, v in (ask or {}).items() if k != "text"} or None,
               "relaid": relaid, "mask_picture": None, "records": []}
        beside = path.with_name(path.stem + "_with_mask.mp4")
        if beside.is_file():
            row["mask_picture"] = {"file": str(beside.relative_to(share)), "frames": frames_of(beside)}
        for rec in recs:
            b, q = rec["built"] or {}, rec["queued"]
            load = b.get("load") or {}
            row["records"].append({"built_file": rec["built_file"], "queued_file": rec["queued_file"], "commit": b.get("commit") or q.get("commit"),
                                   "built_at": as_utc(b.get("built_at")), "queued_at": as_utc(q.get("queued_at")), "status": q.get("status"),
                                   "load": {k: load.get(k) for k in ("name", "first", "frames", "subject", "carried", "motion", "motion_file", "still",
                                                                     "seed", "continue_from", "render_region") if k in load},
                                   "window": ((b.get("settings") or {}).get("window") or {}).get("value"),
                                   "context": ((b.get("settings") or {}).get("context") or {}).get("value"),
                                   "context_noise": ((b.get("settings") or {}).get("context_noise") or {}).get("value"),
                                   "second_subject": [s.get("subject") for s in b.get("subjects") or []] or None,
                                   "job_file": b.get("job_file"), "capture": b.get("capture"), "text_file": (b.get("text") or {}).get("file") or b.get("text_file"),
                                   "names_the_file": named,
                                   "gate": q.get("gate"), "looked_at_by": (q.get("look") or {}).get("by"),
                                   "uncommitted_node_code": b.get("uncommitted_node_code") or q.get("uncommitted_node_code") or [],
                                   "inputs_on_disk": inputs_on_disk(rec, repo, inputs), "analysis": b.get("analysis"),
                                   **agreement(n, stat.st_mtime, ask, rec)})
        rec0 = row["records"][-1] if row["records"] else None       # the last record that names the file is the run that wrote it
        row["making"] = MAKING[3] if not rec0 else MAKING[2] if not named else rec0["making"]
        load = rec0["load"] if rec0 else {}
        files = (ask or {}).get("files") or {}
        capture, label = _capture_of(files.get("mask"))
        if rec0 and rec0.get("capture"):
            capture = Path(rec0["capture"]) if Path(rec0["capture"]).is_absolute() else repo / rec0["capture"]
        subject = load.get("subject") or label
        subject, subject_from = (alias(subject), "the record's load" if load.get("subject") else "the capture folder its mask file sits in") if subject else (None, None)
        if subject is None:
            for who, prefixes in (plan.get("load_prefixes") or {}).items():
                if not who.startswith("_") and any(row["name"].startswith(p) for p in prefixes):
                    subject, subject_from = who, "the load name's prefix, by the plan"
        row["subject"], row["subject_read_from"] = subject, subject_from
        row["subjects"] = [s for s in ([subject] + [alias(x) for x in (rec0 or {}).get("second_subject") or []]) if s]
        text = None
        if rec0 and rec0.get("text_file"):
            t = Path(rec0["text_file"]) if Path(rec0["text_file"]).is_absolute() else repo / rec0["text_file"]
            text = t.read_text() if t.is_file() else None
        text = text or (ask or {}).get("text")
        carried = load.get("carried") or (ask or {}).get("replace")
        classes = part_classes(capture, load.get("subject") or label) if carried in ("held", "the wired parts") else None
        row["form"] = {
            "region": region_of(carried, classes),
            "replaces": region_form(carried, classes,
                                    "the build record and its capture" if rec0 else "the saved graph" + (" and the capture its mask file sits in" if capture else "")),
            "part_named": part_named((rec0 or {}).get("job_file"), repo, load.get("subject") or label),
            "rendered": load.get("render_region") or (ask or {}).get("render_region") or "the whole frame",
            "reference": reference_form(load.get("motion_file") or files.get("motion_video")),
            "reference_none": reference_none((rec0 or {}).get("job_file"), repo),
            "text_says_the_video_is": text_says(text),
            "still": load.get("still") or ", ".join((ask or {}).get("stills") or []) or None,
            "seed": load.get("seed") if load.get("seed") is not None else (ask or {}).get("seed")}
        row["defaults_carried"] = defaults_of((rec0 or {}).get("job_file"), repo)
        first = load.get("first") if load.get("first") is not None else (ask or {}).get("first")
        count = load.get("frames") if load.get("frames") is not None else (ask or {}).get("frames")
        row["span"] = [first, first + count - 1] if first is not None and count else None
        row["span_read_from"] = "the record's load" if load.get("first") is not None else "the saved graph's loader"
        row["leaves_as_the_original"] = left_by_the_load(capture, load.get("subject") or label, row["span"], load.get("name") or row["name"]) \
            if carried in ("held", "the wired parts") else None
        row["laid_in_the_table"] = [[t["first"], t["last"]] for t in table_rows if str(t["piece"]).endswith(rel)]
        row["owner"] = sorted(owner_lines(rel, {row["name"], f"r__{row['session_folder']}__{row['name']}", f"r__{Path(rel).parts[1]}__{row['name']}"},
                                          verdicts, answers, item_of) + on_a_joined_clip(rel, spots or [], verdicts)
                            + on_a_joined_item(rel, joined or [], verdicts), key=lambda o: str(o["at"]))
        renders.append(row)
    gone = [{"built_file": rec["built_file"], "queued_file": rec["queued_file"], "files": rec["files"], "status": rec["queued"].get("status"),
             "output_folder": rec["folder"],
             "load": {k: ((rec["built"] or {}).get("load") or {}).get(k) for k in ("name", "first", "frames", "subject")}}
            for rec in every if id(rec) not in tied]
    sections = []
    for s in shots["shots"]:
        on = [r for r in renders if r["span"] and r["span"][0] <= s["last"] and r["span"][1] >= s["first"]]
        labels = [label for label in shots["subjects"] if s["who"][label]["present"] != "no"]
        labels += sorted({x for r in on for x in (r["subjects"] or ["subject not known"])} - set(labels))
        for label in labels:
            mine = [r for r in on if label in (r["subjects"] or ["subject not known"])]
            present = s["who"][label]["present"] if label in s["who"] else None
            if mine or present in ("yes", "unsure"):
                sections.append({"first": s["first"], "last": s["last"], "subject": label, "present": present,
                                 "renders": [r["file"] for r in sorted(mine, key=lambda r: r["written"])]})
    lo, hi = shots["span"]
    outside = [r["file"] for r in renders if not r["span"] or r["span"][1] < lo or r["span"][0] > hi]
    rows_tied = {o["row"] for r in renders for o in r["owner"]}
    titles = {i["id"]: i.get("title") for i in items}
    loose = [{"row": row, "why": ("a stacked or joined viewing copy; which render it shows is in the item's title" if row in titles
                                  else "a row with no render's file, no page item and no stated frames on a joined clip"),
              "title": titles.get(row) or next((v.get("title") for v in verdicts if v.get("row") == row and v.get("title")), None)}
             for row in sorted({v.get("row") for v in verdicts if v.get("row") not in rows_tied} - {None})]
    return {"schema": SCHEMA, "video": shots.get("video"), "share": str(share), "sources": sources, "since": since,
            "counts": {"renders": len(renders), **{m: sum(r["making"] == m for r in renders) for m in MAKING},
                       "laid_in_the_table": sum(bool(r["laid_in_the_table"]) for r in renders)},
            "sections": sections, "renders": {r["file"]: r for r in renders},
            "stopped": stopped_loads(records, repo, sources),
            "joined_clips": lines_on_joined_clips(joined or [], verdicts),
            "not_placed": {"renders_outside_the_shots_span": outside,
                           "numbered_videos_with_no_saved_graph_and_no_record": no_graph,
                           "renders_of_other_videos_by_the_copy_loaded": dict(other),
                           "numbered_videos_whose_graph_is_no_render_of_a_video": dict(not_renders),
                           "records_whose_render_is_not_on_the_share": gone,
                           "verdict_rows_tied_to_no_render_file": loose},
            "cannot_see": "whether a render is good or better than another; whether a record's inputs are the bytes that ran; which source "
                          "frames a file holds when its count differs from its record; a render whose picture file is gone"}


def frames(a, b) -> str:
    return f"{a}-{b}"


def render_lines(r: dict) -> list[str]:
    """One render as lines of a page."""
    g, rec, relaid = r["graph"] or {}, (r["records"][-1] if r["records"] else None), r.get("relaid")
    lines = [f"- **`{r['name']}`**  `{r['file']}`"]
    span = f"source frames {frames(*r['span'])} ({r['span_read_from']})" if r["span"] else "its source frames are not known"
    lines.append(f"  - frames: {span}; the file holds {r['file_frames']} by ffprobe"
                 + (f"; the saved graph asks for {g.get('frames')} from {g.get('first')} of {g.get('of')}, window {g.get('window')}, context {g.get('context')}" if g.get("of") else "; no saved graph beside it")
                 + (f"; continues {Path(g['continue_from']).name}" if g.get("continue_from") else ""))
    if relaid:
        changed = ", ".join(f"{k}={v}" for k, v in (relaid.get("changed") or {}).items()) or "none"
        lines.append(f"  - RE-LAID from `{relaid.get('from')}` (sha256 `{str(relaid.get('sha256') or '')[:12]}`, changed: {changed}, by {relaid.get('by')} at {relaid.get('at')}): not a new render")
    if rec:
        on = rec["inputs_on_disk"]
        lines.append(f"  - ITS MAKING: {r['making']}" + (f" (re-laid from `{relaid.get('from')}`)" if relaid else "" if rec["names_the_file"] else " (tied by the build record's output folder and the load's name)")
                     + f". Record `{rec['built_file'] or rec['queued_file']}`: commit {rec['commit']}, queued {rec['queued_at']}, "
                     f"status {rec['status']}, window {rec['window']}, context {rec['context']}"
                     + (f", gate {rec['gate'].get('verdict')}" + (f" (overridden: {rec['gate'].get('overridden')})" if rec["gate"].get("overridden") else "") if rec.get("gate") else "")
                     + (f", looked at by {rec['looked_at_by']}" if rec.get("looked_at_by") else ", no look recorded")
                     + (f", UNCOMMITTED NODE CODE at the build: {rec['uncommitted_node_code']}" if rec["uncommitted_node_code"] else ""))
        for d in rec["disagrees"]:
            lines.append(f"    - {'DISAGREES' if rec['names_the_file'] else 'AND'}: {d}")
        lines.append(f"    - its inputs by path: {on['named'] - len(on['missing'])} of {on['named']} still files"
                     + (f" (gone: {', '.join(Path(m).name for m in on['missing'])})" if on["missing"] else "")
                     + f"; capture folder {'there' if on['capture_folder'] else 'NOT there' if on['capture_folder'] is False else 'not named'}"
                     + f"; windows folder {'there' if on['windows_folder'] else 'NOT there' if on['windows_folder'] is False else 'not named'}"
                     + ("" if not on.get("still") else f"; the still {on['still']} " + ("is a file" if on["still_is_a_file"] else "is NOT a file" if on["still_is_a_file"] is False
                                                                                         else "was not looked for (no --inputs)"))
                     + (f"; the analysis folder's fingerprint at the build {str(rec['analysis'].get('fingerprint'))[:12]}" if (rec.get("analysis") or {}).get("fingerprint") else "")
                     + (f"; {len(r['records'])} records name this file, the last is shown" if len(r["records"]) > 1 else ""))
    else:
        lines.append(f"  - ITS MAKING: {r['making']}"
                     + (f" (re-laid from `{relaid.get('from')}`). What is known of it is the PNG beside it, which carries the render's graph with its composite settings changed and `relaid`"
                        if relaid else ". What is known of it is the graph saved beside it: what was asked for, not what ran or from which commit"))
    mask = r["mask_picture"]
    lines.append("  - mask picture: " + (f"`{Path(mask['file']).name}`, {mask['frames']} frames" if mask else "none beside it"))
    f = r["form"]
    lines.append(f"  - form: replaces {f['replaces']}; rendered {f['rendered']}; reference video: {f['reference']['says']}"
                 + (f"; still {f['still']}" if f["still"] else "") + (f"; seed {f['seed']}" if f["seed"] is not None else ""))
    if f["text_says_the_video_is"]:
        lines.append(f"    - the text: {f['text_says_the_video_is']}")
    carried = r.get("defaults_carried")
    lines.append("  - the defaults it carried: " + ("not said: its job was written before `job.defaults` (2026-10-14)" if carried is None else
                                                    "; ".join(f"{d['what']}: {d['carries']} [{d['from']}]" for d in carried)))
    lines.append(f"  - made: in the folder of `{r['session_folder']}` (the folder, not a signature); file written {r['written']}"
                 + (f"; subject read from {r['subject_read_from']}" if r["subject_read_from"] else "; SUBJECT NOT KNOWN from any file"))
    lines.append("  - in the delivery table: " + (", ".join(frames(a, b) for a, b in r["laid_in_the_table"]) if r["laid_in_the_table"] else "not laid"))
    for o in r["owner"]:
        where = (f", ON THE JOINED CLIP {o['on_the_joined_clip']} at {frames(*o['at_frames'])}, through the table `{o['through_the_table']}` "
                 "(said of the joined clip there, not of this render alone)" if o.get("on_the_joined_clip") else "")
        lines.append(f"  - the owner, {str(o['at'])[:16]}" + (f" at frame {o['frame']}" if o.get("frame") not in (None, "") else "") + where
                     + f": {str(o['said']).strip().replace(chr(10), ' / ')}  [{o['from']}, row {o['row']}]")
    if not r["owner"]:
        lines.append("  - the owner: no line tied to this render")
    return lines


def page(out: dict, made: str, table: str | None) -> str:
    """The whole ledger as a page: what it is, the counts, a section a shot and a subject, what was not placed."""
    c = out["counts"]
    lines = [f"# Every render of {out['video']} on the share, a shot and a subject at a time", "", made, "",
             "Read from files by `bench/render_ledger.py`. Nothing here says a render is good, or better than another; "
             "`ITS MAKING` is one of four facts about files (`MAKING`), and the tool's docstring says what `agrees` means and all it means. "
             "A render that covers several shots, or two subjects, is told in full once and pointed at from its other sections.", "",
             f"What it cannot see: {out['cannot_see']}.", "",
             "| renders of this video | " + " | ".join(MAKING) + " | laid in the table |", "|" + "---|" * (len(MAKING) + 2),
             f"| {c['renders']} | " + " | ".join(str(c[m]) for m in MAKING) + f" | {c['laid_in_the_table']} |", "",
             f"The delivery table read: `{table}`." if table else "No delivery table was given.", ""]
    told: dict[str, str] = {}
    for s in out["sections"]:
        lines += [f"## {frames(s['first'], s['last'])}, {s['subject']}" + ("" if s["present"] == "yes" else
                                                                           f" (the shots file says present: {s['present']})" if s["present"] else " (not a subject of the shots file)"), ""]
        here = f"{frames(s['first'], s['last'])}, {s['subject']}"
        if not s["renders"]:
            lines += ["No render covers these frames for this subject.", ""]
        for f in s["renders"]:
            r = out["renders"][f]
            if f in told:
                lines.append(f"- **`{r['name']}`**  `{f}`: {r['making']}; told in full under \"{told[f]}\"")
            else:
                told[f] = here
                lines += render_lines(r)
        lines.append("")
    stopped = out.get("stopped") or []
    lines += ["## Loads stopped before a render", "",
              f"{len(stopped)} load(s), each from the note its card session wrote beside its job file, read whole (`stopped_loads`).", ""]
    for x in stopped:
        lines += [f"- **`{x['load']}`**, {x['subject']}, " + (frames(*x["frames"]) if x["frames"] else "frames not in its job file")
                  + f": `{x['note']}` (written {x['note_written']})", ""] + ["  > " + t for t in x["note_says"].splitlines()] + [""]
    n = out["not_placed"]
    lines += ["## Not placed", "",
              f"- renders of this video outside the shots file's span: {len(n['renders_outside_the_shots_span'])}"]
    lines += [f"  - `{f}`" for f in n["renders_outside_the_shots_span"]]
    for f in n["renders_outside_the_shots_span"]:
        lines += ["  " + x for x in render_lines(out["renders"][f])[1:]]
    lines.append(f"- build records with no render on the share (no file they name, and none of their load's name in their output folder): "
                 f"{len(n['records_whose_render_is_not_on_the_share'])}")
    lines += [f"  - `{g['built_file'] or g['queued_file']}`: status {g['status']}, load {g['load'].get('name')} "
              f"({g['load'].get('frames')} frames from {g['load'].get('first')}), files {g['files'] or 'none named'}" for g in n["records_whose_render_is_not_on_the_share"]]
    lines.append(f"- numbered videos under the share with no saved graph beside them and no record: {len(n['numbered_videos_with_no_saved_graph_and_no_record'])} "
                 "(they cannot be tied to a video; listed in the JSON)")
    for key, words in (("renders_of_other_videos_by_the_copy_loaded", "renders of other videos, by the copy their graph loads"),
                       ("numbered_videos_whose_graph_is_no_render_of_a_video", "numbered videos whose saved graph is no render of a video, by why")):
        lines.append(f"- {words}: " + (", ".join(f"{k}: {v}" for k, v in sorted(n[key].items())) or "none"))
    lines.append(f"- rows of the verdicts file tied to no render file: {len(n['verdict_rows_tied_to_no_render_file'])}")
    lines += [f"  - `{x['row']}`: {x['why']}" + (f": {x['title']}" if x["title"] else "") for x in n["verdict_rows_tied_to_no_render_file"]]
    return "\n".join(lines) + "\n"


def summary(out: dict, made: str) -> str:
    """The counts a shot and a subject, with no path of a private folder and none of the owner's words: what a
    tracked record can hold. The page beside the JSON is the ledger; this says where it is and what it counted.
    `made` is printed as given, so the caller hands in a line with no path of the machine (`main` does)."""
    lines = ["| frames | subject | renders | " + " | ".join(MAKING) + " | laid in the table | with a line of the owner's |", "|" + "---|" * (len(MAKING) + 5)]
    for s in out["sections"]:
        rs = [out["renders"][f] for f in s["renders"]]
        lines.append(f"| {frames(s['first'], s['last'])} | {s['subject']} | {len(rs)} | " + " | ".join(str(sum(r["making"] == m for r in rs)) for m in MAKING)
                     + f" | {sum(bool(r['laid_in_the_table']) for r in rs)} | {sum(bool(r['owner']) for r in rs)} |")
    return made + "\n\n" + "\n".join(lines) + "\n"


def copies_of(folder: str) -> tuple[dict, list[str]]:
    """({file name: frame rate} of a video's source and every copy it has or has had on the present copy's frame
    clock, the names of its copies on another clock), from its analysis folder's manifest. A render's first frame
    is worked out from the copy's seconds as a frame of the video, which holds only on that clock."""
    sys.path.insert(0, str(HERE.parent))
    import clip_folder
    manifest = clip_folder.read_manifest(Path(folder))
    held = {manifest["source"]["name"]: float(manifest["source"]["rate"])}
    other = []
    for c in clip_folder.copies(manifest):
        if clip_folder.same_clock(manifest, c["name"]):
            held[c["name"]] = float(c["rate"])
        else:
            other.append(c["name"])
    return held, other


def command_of(argv: list[str], python: str | None = None) -> dict:
    """This tool's own command as the ledger keeps it: the arguments, and one line a shell takes as it stands, run
    from the repo's root."""
    import shlex
    return {"argv": list(argv), "line": " ".join(shlex.quote(w) for w in [python or sys.executable, "bench/render_ledger.py", *argv]),
            "run_from": "the repo's root"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shots", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--share", required=True, help="the server's output root; its Video/ folder is walked")
    p.add_argument("--records", required=True, help="the folder the job tool writes under: <records>/*_job/<load>.render.*.json")
    p.add_argument("--source", action="append", default=[], metavar="FILE[=RATE]",
                   help="a copy of the video a render may load, by file name, with its rate when it is not 24; repeatable")
    p.add_argument("--copies-of", metavar="FOLDER", help="the video's analysis folder: its source and every copy it has or has had on the present "
                                                           "copy's frame clock are read from its manifest, each at its own rate; --source adds to them")
    p.add_argument("--table", help="a delivery table")
    p.add_argument("--verdicts")
    p.add_argument("--answers")
    p.add_argument("--items", help="the review page's items: an item's id to a render's file")
    p.add_argument("--spots", action="append", default=[], metavar="SPOTS.json",
                   help='the owner\'s rows that were said of a JOINED clip at stated frames: {"clip", "table", "spots": [{"row", "first", "last"}]}; '
                        "each is printed under the renders that clip's table laid there; repeatable")
    p.add_argument("--joined-tables", action="append", default=[], metavar="FOLDER",
                   help="a folder bench/shot_review_clips.py wrote joined clips' tables into: the owner's lines on a joined item are printed under each "
                        "render its table laid, and a keep there counts on each (the lead's ruling, 2026-10-14); repeatable")
    p.add_argument("--inputs", help="the server's input root, so a record's still can be looked for")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="leave out files written before this day (UTC); said in the record")
    p.add_argument("--json", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--summary", help="also write the counts a shot and a subject, with no private path and no owner's words")
    a = p.parse_args()
    coverage = _module("shot_coverage")
    shots = coverage.load_shots(a.shots)
    plan = coverage.load_plan(a.plan, shots["subjects"])
    rows = _module("assemble_delivery").read_table(a.table) if a.table else []
    lines = lambda path: [(n, json.loads(line)) for n, line in enumerate(open(path), 1) if line.strip()] if path else []
    sources, other_clock = ({}, []) if not a.copies_of else copies_of(a.copies_of)
    sources.update({s.partition("=")[0]: float(s.partition("=")[2] or 24) for s in a.source})
    if not sources:
        raise SystemExit("no copy of the video is named: give --copies-of the video's analysis folder, or --source")
    repo = HERE.parent
    read_table = _module("assemble_delivery").read_table
    spots = []
    for path in a.spots:
        given = json.load(open(path))
        spots.append({**given, "rows": read_table(given["table"]), "file": path})
    out = ledger(shots, plan, Path(a.share), Path(a.records), repo, sources, rows, [v for _, v in lines(a.verdicts)], lines(a.answers),
                 (json.load(open(a.items)).get("items") or []) if a.items else [], a.since, alias=lambda label: coverage.alias(label, plan),
                 spots=spots, inputs=Path(a.inputs) if a.inputs else None, joined=joined_items(a.joined_tables, [v for _, v in lines(a.verdicts)]))
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    made = (f"Made {utc(datetime.datetime.now().timestamp())} at commit {commit} from: the share `{a.share}`" + (f" (files since {a.since})" if a.since else " (every file)")
            + f"; the records under `{a.records}`; `{a.shots}`; `{a.plan}`; the copies {', '.join(sources)}"
            + (f" (from `{a.copies_of}`'s manifest" + (f"; LEFT OUT, on another frame clock: {', '.join(other_clock)}" if other_clock else "") + ")" if a.copies_of else "")
            + "".join(f"; `{x}`" for x in (a.verdicts, a.answers, a.items, *a.spots) if x)
            + "".join(f"; the joined items' tables in `{x}`" for x in a.joined_tables) + ".")
    out["made"] = made
    # its own command, so the table made from it can print it and a second run types nothing (2026-10-13)
    out["command"] = command_of(sys.argv[1:])
    Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(a.md).write_text(page(out, made, a.table))
    if a.summary:
        # what a tracked record can hold: no path of the machine (the share's root, a records folder given by its
        # whole path), so the inputs are named by what they are and the page beside the JSON has where they were
        Path(a.summary).write_text(summary(out, f"Made {made.split(' ')[1]} at commit {commit} by `bench/render_ledger.py` from the share's output "
                                                f"root, the job tool's records, the video's shots and plan files, the copies {', '.join(sources)}"
                                                + (", a delivery table" if a.table else "") + (", the owner's lines" if a.verdicts or a.answers else "")
                                                + f". Where each was read from is on the page beside `{Path(a.json).name}`, which is not tracked."))
    c = out["counts"]
    print(f"{c['renders']} renders of this video: " + ", ".join(f"{m}: {c[m]}" for m in MAKING) + f"; laid in the table: {c['laid_in_the_table']}")
    print(f"sections (a shot and a subject): {len(out['sections'])}; not placed: " + ", ".join(
        f"{k}: {len(v) if isinstance(v, list) else sum(v.values())}" for k, v in out["not_placed"].items()))
    print("wrote", a.json, "and", a.md, *(["and", a.summary] if a.summary else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
