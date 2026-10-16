#!/usr/bin/env python3
"""A sidecar beside every video we make: `<video file name>.json`, holding as much as is known of how the video was made.

    <python> bench/sidecar.py read VIDEO.mp4            # print it; exit 1 when the video is not the one it was written for
    (from code) sidecar.write(video, command=sys.argv, inputs={role: path}, extra={...}, pieces=[...] | windows=[...])

**Why** (the owner, 2026-10-16, relayed by the lead): what is watched must be traceable to how it was made, by
files, not by a session's memory. A path can move and a file can be remade under its name, so the sidecar keeps
the facts themselves: the sha256 of every input as it is when the video is written, the command as run, the commit
and the files that were not committed, and, for each piece of the video, every window H3 rendered: its source
frames, the text it was given IN FULL, the graph that made it IN FULL, the reference video and the still with the
bytes they were read from and every step made to them on the way to H3.

**Its name** is the video's whole file name with `.json` added (`lead_fp_0438c_00001.mp4.json` beside
`lead_fp_0438c_00001.mp4`; the lead, mrtony, 2026-10-16, docs/wiki/decisions.md), named by `sidecar_path` and nowhere
else: `<stem>.json` is already many tools' own record beside a video (a reference's, a zoomed picture's), which a
sidecar never reads or writes over. A sidecar written under the old name (`<stem>.json`, before 2026-10-16T19Z) is
not migrated: `read` does not find it, and the video's next writer writes the new one.

**A sidecar** (`schema` `h3_sidecar/1`): `video`, `video_sha256`, `made_at` (UTC), `made_by` (the caller's word, else
the `H3_SESSION` environment key, else "not said"), `command` (one line, quoted as a shell takes it), `cwd`,
`environment` (every `H3_` key it ran with: stills, inputs and the output folder are read through them), `git`
(`commit`, `dirty`: the paths `git status` lists), `tool` (the script, resolved, and its sha256, or why there is
none), `inputs` (role, path resolved, sha256, size; a missing file says so), `extra` (the caller's), `earlier` (where
it was written over a sidecar of the same bytes, as the assembler's `--check-only` does: that one's command, maker and
time), `checks` (each check of the video made since, appended by `add_check`, the assembler's `--check-only`: its
command, time, verdict, check record's sha256 and whether it was checked against the build's own table; a check never
replaces the build's command, inputs, extra or pieces), and `pieces` (a joined clip: one a piece laid) or `windows` (a
single render). `read` refuses a sidecar whose `video_sha256` is not the video's bytes now, and one that is not of this
schema; finding none, it names an old-name sidecar of this schema beside the video, which it does not read.

**A piece's windows are read, never parsed again here.** The load's graph and what it was handed are read by
`bench/what_h3_got.py::read` (the reader of the full prompt read), with digests on: every input it lists is written
whole (`inputs`), each held to the queue's digest, and its hard failures as `read_failures`. The stills' folder is the
queue record's `input_dir`; else, labeled "not recorded at the queue", `H3_INPUT_DIR` at assembly or
`masked_job.input_dir()`'s stock folder, and a still not found there is written "folder not known", never MISSING.
Where the reader refuses a load, the text is the graph's song node's `prompt`, said so. The windows by its `windows_of`; a reference
video's steps by `bench/render_ledger.py::reference_form`; the bytes a reference was read from as `bench/laid.py`'s
`refs` finds them (the queue record's digest, then the `references/<name>_SUPERSEDED_*` copy with those bytes). A
piece is tied to its job folder by the render ledger's record of it (its queue file), named by the table that laid
it; nothing is found by a name. The ledger's own word on that record (`making`, `disagrees`) is carried, and the job's
graph is held to the PNG VHS saved beside the render (ComfyUI's `is_changed` set aside), except where the graph was
itself taken from that PNG, which is said. A joined clip's pieces are
the rows of the table's .txt the clip lays (the assembler's segments: a `--span` leaves the others out), each tied to
the table JSON's row of the same frames and file; a .txt the JSON does not match is refused as the pieces' source and
said. A load is one graph: every window of it was made by that graph and given its one text, so each window carries
both whole.

**Where a value cannot be read, the entry says why** and the rest is still written: no record of the piece, a ledger
named and not found, a job folder gone, a graph the reader refuses, no window recorded, a piece whose reading fails
(its entry is the error, the other pieces are written). A piece with no job folder takes its graph from the
PNG VHS saved beside the render (its `prompt` text chunk) and says so; with neither it says no graph is recoverable.

**What it cannot see.** Whether the bytes an input has now are the bytes a step read, except where a record kept a
digest (the queue's `digests`). The UI workflow of a graph queued by API: there is none (VHS's PNG holds `prompt`
only). What a node computed and did not save.

**Named gaps, not built** (the blind reads of 2026-10-16, internal/claude/2026-10-16_mrtony/review_sidecar.md and
review2_sidecar.md):
- Only the assembler, the review clips' build (through the assembler) and the stack view write sidecars: renders,
  references, patches and capture strips get none.
- (10) A dirty tree is recorded as paths only: no hash of each dirty file, and the modules this imports (delivery_table,
  what_h3_got, `_lib`) are not hashed, only the calling script. A load's `commit` is the queue record's, HEAD at queue
  time, not a commit the node code is known to match. The queue record's `override`, the build record's commit and
  `cached_nodes` are not kept.
- (11) Only the `H3_` environment keys are kept; the ffmpeg that wrote the bytes (its path and `-version`) is not.
- (13) A window's `first`/`last` include its context frames; the region meta's `trim` and the node's
  `first_written_frame` are not carried, so one delivered frame can map to two windows.
- (14) One text per load is assumed and not checked: a graph that sets a `timeline` or wires `lists` (the node then
  fills the prompt per window) is not refused or flagged, and the node's `said` lines are not kept.
- (15) Digests are cached by (path, size, mtime_ns) for the run (`what_h3_got.sha256`, one cache with the reader's), and `read` uses that cache: a file rewritten at the
  same size inside one timestamp tick, by the same process, is trusted at its old digest.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

SCHEMA = "h3_sidecar/1"
#: the environment key a session's name is read from when the caller does not say it
SESSION_KEY = "H3_SESSION"


def sha256_of(path) -> str:
    """A file's sha256, by the reader's own digest (`bench/what_h3_got.py::sha256`), read once a run: one cache for both."""
    import what_h3_got
    return what_h3_got.sha256(Path(path))


def sidecar_path(video) -> Path:
    """The ONE place a sidecar's name is made: the video's whole file name with `.json` added."""
    video = Path(video)
    return video.with_name(video.name + ".json")


def file_row(role: str, path) -> dict:
    p = Path(path).resolve()
    if not p.is_file():
        return {"role": role, "path": str(p), "sha256": None, "size": None, "missing": "not a file when the sidecar was written"}
    return {"role": role, "path": str(p), "sha256": sha256_of(p), "size": p.stat().st_size}


def git_state(repo: Path = REPO) -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)  # noqa: E731
    head = run("rev-parse", "HEAD")
    if head.returncode:
        return {"commit": None, "dirty": None, "why": head.stderr.strip() or "not a git checkout"}
    dirty = [line[3:] for line in run("status", "--porcelain").stdout.splitlines() if line.strip()]
    return {"commit": head.stdout.strip(), "dirty": dirty}


def png_graph(render: Path) -> dict | None:
    """The graph VHS saved beside a render as `<stem>.png` (its `prompt`, and `workflow` where the UI queued it)."""
    png = render.with_suffix(".png")
    if not png.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(png) as im:
            info = {k: v for k, v in im.info.items() if k in ("prompt", "workflow") and isinstance(v, str)}
    except OSError as e:
        return {"from": str(png), "why": f"the PNG does not read: {e}"}
    out: dict = {"from": str(png), "from_is": "the PNG VHS saved beside the render (its text chunks), not the job's record"}
    for k, v in info.items():
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = None
            out[f"{k}_why"] = "its text chunk is not JSON"
        out[f"{k}_sha256"] = hashlib.sha256(v.encode()).hexdigest()
    if "workflow" not in info:
        out["workflow_why"] = "no UI workflow in the PNG: the graph was queued by API, which sends the prompt alone"
    return out


def seeds_of(graph: dict) -> dict:
    return {nid: {k: v for k, v in n.get("inputs", {}).items() if "seed" in k and not isinstance(v, list)}
            for nid, n in graph.items() if isinstance(n, dict) and any("seed" in k for k in n.get("inputs", {}))}


def read_at(path: Path, kept: str | None, now: str | None = None) -> dict:
    """The bytes an input was read from, as bench/laid.py's refs finds them: the file now against the queue's digest,
    then a `<folder>_SUPERSEDED_*` copy of the same name with those bytes. `now` is the file's digest where the
    reader already made it."""
    if not path.is_file():
        return {"state": "MISSING at that path", "sha256_now": None, "sha256_at_queue": kept, "read_at": None}
    now = now or sha256_of(path)
    if not kept:
        return {"state": "no digest kept at the queue: the bytes now are not known to be the bytes read", "sha256_now": now,
                "sha256_at_queue": None, "read_at": None}
    if now == kept:
        return {"state": "same bytes as queued", "sha256_now": now, "sha256_at_queue": kept, "read_at": str(path)}
    found = next((str(g) for g in sorted(path.parent.parent.glob(f"{path.parent.name}_SUPERSEDED_*/{path.name}")) if sha256_of(g) == kept), None)
    return {"state": "CHANGED since queued" + (f"; the bytes it read are at {found}" if found else "; the bytes it read are not found beside it"),
            "sha256_now": now, "sha256_at_queue": kept, "read_at": found}


#: the fields of a reference record kept for each step of its chain (the tools' own words)
STEP_KEYS = ("what", "made_by", "made_at", "frames", "its_first_frame_is_source_frame", "size", "box", "box_is", "box_set_by",
             "room", "room_set_by", "scale", "face_height_px", "a_point_of_the_canvas_lands_at")


def reference_chain(path: str) -> list[dict]:
    """Each record from the reference video back to the body model, with the fields of STEP_KEYS and what was drawn."""
    out, video = [], Path(path)
    for _ in range(6):
        rec_path = video.with_suffix(".json")
        if not rec_path.is_file():
            out.append({"video": str(video), "record": None, "why": "no record beside it: the chain ends here"})
            break
        try:
            r = json.loads(rec_path.read_text())
        except ValueError:
            out.append({"video": str(video), "record": str(rec_path), "why": "its record does not read"})
            break
        ins = r.get("inputs") if isinstance(r.get("inputs"), dict) else {}
        # the drawing as its record holds it, whole (its own field names: the mouth's `drawn`, `eyes_shut`, the rings), and
        # the face points it was drawn from
        out.append({"video": str(video), "record": str(rec_path), **{k: r[k] for k in STEP_KEYS if k in r}, "drawn": r.get("drawn"),
                    **({"points": {"file": ins["points"], "first_row_is_source_frame": ins.get("their_first_row_is_source_frame")}}
                       if ins.get("points") else {})})
        made = r.get("made_from") or r.get("inputs") or {}
        nxt = made.get("body_model_video") or made.get("reference_video") if isinstance(made, dict) else None
        if not nxt:
            break
        video = Path(nxt)
    return out


def reference_video(row: dict) -> dict:
    """A `<Video k>` input as what_h3_got read it, with the bytes it was read from and every step to H3."""
    import render_ledger
    out = {"tag": row.get("tag"), "path": row.get("file"), "node": row.get("node")}
    if not row.get("file"):
        return {**out, "why": row.get("note") or "the graph names no file"}
    out.update(read_at(Path(row["file"]), row.get("sha256_at_queue"), row.get("sha256")))
    form = render_ledger.reference_form(row["file"])
    out["steps"] = form.get("steps")
    out["steps_say"] = form.get("says")
    out["chain"] = reference_chain(row["file"])
    out["loader"] = row.get("loader")
    out["file"] = {"size": row.get("size"), "frames": row.get("file_frames"), "pix_fmt": row.get("pix_fmt")}
    out["fit"] = row.get("fit") or {"why": row.get("fit_not_given") or "what_h3_got gives no fit for it"}
    out["encoder_frames"] = row.get("encoder_frames") or {"why": "no window known, so which frames the text encoder is shown is not said"}
    out["node_said"] = row.get("node_said")
    return out


def folder_said(entry: dict, row: dict, known: bool) -> dict:
    """A file named relative to the stills' folder, not found there, where no queue record names that folder: "folder not
    known", never MISSING (the server may have read it from a folder nobody recorded)."""
    if not known and str(entry.get("state") or "").startswith("MISSING") and row.get("named") and not Path(row["named"]).is_absolute():
        entry = {**entry, "state": f"folder not known: {row['named']} is not in {Path(row['file']).parent}, and the queue record names no input folder"}
    return entry


def still(row: dict) -> dict:
    out = {"tag": row.get("tag"), "path": row.get("file"), "node": row.get("node")}
    if not row.get("file"):
        return {**out, "why": row.get("note") or "the graph names no file"}
    out.update(read_at(Path(row["file"]), row.get("sha256_at_queue"), row.get("sha256")))
    out.update(size=row.get("size"), append=row.get("append"), node_said=row.get("node_said"))
    return out


def load_record(job_folder, load: str, stage: str = "render", render: Path | None = None, output: Path | None = None) -> dict:
    """One load's windows, graph, references, stills, zoom and seeds, read through what_h3_got; every gap said."""
    import what_h3_got as wh
    job_folder = Path(job_folder)
    out: dict = {"job_folder": str(job_folder), "load": load, "stage": stage}
    graph_file = job_folder / f"{load}.{stage}.json"
    queued_file = job_folder / f"{load}.{stage}.queued.json"
    queued = json.loads(queued_file.read_text()) if queued_file.is_file() else None
    graph: dict | None = None
    if graph_file.is_file():
        raw = graph_file.read_text()
        graph = {"from": str(graph_file), "from_is": "the job folder's graph, as queued" if queued else "the job folder's graph, BUILT AND NOT QUEUED",
                 "prompt": json.loads(raw), "prompt_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                 "prompt_id": (queued or {}).get("prompt_id"), "queued_at": (queued or {}).get("queued_at"),
                 "workflow_why": "no UI workflow: the graph was queued by API, which sends the prompt alone"}
    elif render is not None:
        graph = png_graph(render)
    if graph is None:
        graph = {"why": f"no graph is recoverable: {graph_file} is not there" + (f" and no PNG beside {render.name}" if render else "")}
    out["graph"] = {k: v for k, v in graph.items() if k not in ("prompt", "workflow")}
    out["seeds"] = seeds_of(graph["prompt"]) if isinstance(graph.get("prompt"), dict) else None
    out["commit"] = {"commit": (queued or {}).get("commit"), "uncommitted_node_code": (queued or {}).get("uncommitted_node_code")} if queued \
        else {"why": "no queue record"}
    if (queued or {}).get("input_dir"):
        out["input_dir"] = {"path": str(queued["input_dir"]), "from": "the queue record's input_dir"}
    elif os.environ.get("H3_INPUT_DIR"):
        out["input_dir"] = {"path": os.environ["H3_INPUT_DIR"], "from": "not recorded at the queue: H3_INPUT_DIR at assembly"}
    else:
        import masked_job
        out["input_dir"] = {"path": str(masked_job.input_dir()), "from": "not recorded at the queue: the stock folder (masked_job.input_dir())"}
    known = not out["input_dir"]["from"].startswith("not recorded")
    try:
        if output is None and os.environ.get("H3_OUTPUT_DIR"):
            output = Path(os.environ["H3_OUTPUT_DIR"])
        got = wh.read(job_folder, load, stage, digests=True, output=output, inputs=Path(out["input_dir"]["path"]))
    except SystemExit as e:
        got, out["why"] = None, f"bench/what_h3_got.py does not read it: {e}"
    except Exception as e:  # noqa: BLE001 - a reader that crashes on an old record is said, never fatal to the sidecar
        got, out["why"] = None, f"bench/what_h3_got.py failed on it: {type(e).__name__}: {e}"
    text = dict((got or {}).get("text") or {})
    if got:
        text["from"] = "bench/what_h3_got.py (the song node's prompt)"
    elif isinstance(graph.get("prompt"), dict):
        songs = [n for n in graph["prompt"].values() if isinstance(n, dict) and n.get("class_type") == wh.SONG]
        if len(songs) == 1 and isinstance(songs[0].get("inputs", {}).get("prompt"), str):
            said = songs[0]["inputs"]["prompt"]
            text = {"text": said, "sha256": hashlib.sha256(said.encode()).hexdigest(),
                    "from": f"the graph's song node, as {graph.get('from')} holds it (bench/what_h3_got.py does not read the load)"}
    windows = wh.windows_of(got, queued) if got else []
    source = next((r for r in (got or {}).get("inputs", []) if r["role"] == "the source"), {})
    source_first = (source.get("loader") or {}).get("first_frame")
    cap = (source.get("loader") or {}).get("frame_load_cap")
    load_last = source_first + cap - 1 if source_first is not None and cap else None
    cont = next((r for r in (got or {}).get("inputs", []) if str(r["role"]).startswith("continued from")), None)
    asked = (got or {}).get("windows") or {}
    refs = [reference_video(r) for r in (got or {}).get("inputs", []) if str(r.get("tag") or "").startswith("<Video")]
    common = {"text": text.get("text"), "text_sha256": text.get("sha256"), "text_from": text.get("from"),
              "graph": graph.get("prompt"), "graph_sha256": graph.get("prompt_sha256"), "graph_from": graph.get("from"),
              "prompt_id": graph.get("prompt_id"), "continue_from": None if cont is None else {
                  "path": cont.get("file"), **read_at(Path(cont["file"]), cont.get("sha256_at_queue"))} if cont.get("file") else {"why": cont.get("note")},
              "window_frames": asked.get("window_frames"), "context_frames": asked.get("context_frames"), "context_noise": asked.get("context_noise")}
    if text.get("text") is None:
        common["text_why"] = out.get("why") or "the graph holds no prompt"
    if graph.get("prompt") is None:
        common["graph_why"] = graph.get("why") or graph.get("prompt_why") or "no graph"
    out["windows"] = []
    for k, (a, n) in enumerate(windows, 1):
        w = {"index": k, "first": None if source_first is None else source_first + a, "last": None if source_first is None else source_first + a + n - 1,
             "last_in_the_load": None if load_last is None or source_first is None else min(load_last, source_first + a + n - 1),
             "first_on_the_load": a, "frames": n, **common,
             "encoder_frames": {r["tag"]: (r["encoder_frames"][k - 1] if isinstance(r["encoder_frames"], list) and len(r["encoder_frames"]) >= k else None) for r in refs}}
        if load_last is not None and source_first is not None and source_first + a + n - 1 > load_last:
            w["past_the_load"] = (f"the window runs {source_first + a + n - 1 - load_last} frame(s) past the load's last frame ({load_last}): "
                                  "frames the loader does not hand on, so not source frames of this load")
        if source_first is None:
            w["first_why"] = "the source loader's first frame is not read, so the window's source frames are not said"
        out["windows"].append(w)
    if not out["windows"]:
        out["windows"].append({"index": None, "why": "no window is recorded for this load (no window region meta and no node line \"[k] window from\")"
                               + (f"; {out['why']}" if out.get("why") else ""), **common})
    out["references"] = refs or [{"why": "the graph wires no reference video" if got else out.get("why")}]
    out["stills"] = [folder_said(still(r), r, known) for r in (got or {}).get("inputs", []) if str(r.get("tag") or "").startswith("<Picture")] or \
        [{"why": "the graph appends no still" if got else out.get("why")}]
    # every input the reader lists, whole, each held to the queue's digest (the source as the load cut it, masks, parts,
    # keep-outs, the sound with its trim, the references and stills)
    out["inputs"] = [{**r, **(folder_said(read_at(Path(r["file"]), r.get("sha256_at_queue"), r.get("sha256")), r, known) if r.get("file") else {})}
                     for r in (got or {}).get("inputs", [])] or [{"why": out.get("why") or "the reader lists no input"}]
    if not known:
        out["read_failures_note"] = (f"the stills' folder is {out['input_dir']['from']}: a file the reader says is not there may be in the folder the "
                                     "server read, which no record names")
    out["zoom"] = (got or {}).get("zoom") or {"why": out.get("why")}
    out["replaces"] = (got or {}).get("replaces")
    out["read_failures"] = (got or {}).get("failures")
    return out


def graphs_agree(job_graph: dict, png: dict | None) -> dict:
    """The job folder's graph against the `prompt` VHS saved in the PNG beside the render, ComfyUI's `is_changed` key on a
    node set aside (the server adds it to what it saves)."""
    if not png or not isinstance(png.get("prompt"), dict):
        return {"agrees": None, "why": (png or {}).get("why") or "no PNG beside the render, or its prompt does not read"}
    strip = lambda g: {k: {a: b for a, b in v.items() if a != "is_changed"} if isinstance(v, dict) else v for k, v in g.items()}  # noqa: E731
    a, b = strip(job_graph), strip(png["prompt"])
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"agrees": not differ, "png": png.get("from"), "nodes_that_differ": differ, "set_aside": "is_changed"}


def found(rel, bases) -> Path | None:
    """`rel` as given when absolute, else under the first of `bases` that holds it."""
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    return next((Path(b) / p for b in bases if (Path(b) / p).exists()), None)


def piece_of(row: dict, ledger: dict | None, share: Path, ledger_why: str | None = None, bases=(REPO,)) -> dict:
    """One laid piece: its file, and its load read through the render ledger's record of it (the queue file names
    the job folder). `row` is a delivery table JSON row (`file` relative to the share, `name`, `first`, `last`).
    `ledger_why` is why there is no ledger, where the table named one; `bases` are where a relative path is looked for."""
    rel = row.get("file")
    path = share / rel if rel else None
    out: dict = {"name": row.get("name"), "subject": row.get("subject"), "first": row.get("first"), "last": row.get("last"),
           "piece_first": row.get("piece_first"), "run": row.get("run"), **(file_row("the piece", path) if path else {"why": "the row names no file"})}
    zoomed = path.with_name(path.stem + "_zoomed.json") if path else None
    if zoomed is not None and zoomed.is_file():
        out["zoomed"] = {"record": file_row("the zoomed record", zoomed), "box": None, "zoom": None}
        try:
            z = json.loads(zoomed.read_text())
            out["zoomed"].update(box=z.get("box"), zoom=z.get("zoom"), canvas=z.get("canvas"))
        except ValueError:
            out["zoomed"]["why"] = "its record does not read"
        pic = path.with_name(path.stem + "_zoomed.mp4")
        out["zoomed"]["picture"] = file_row("the zoomed picture", pic)
    elif path is not None:
        out["zoomed"] = {"why": f"no {path.stem}_zoomed.json beside the piece: it was not laid from a zoomed picture (the load's own "
                                "render_region is under load.zoom)"}
    r = (ledger or {}).get("renders", {}).get(rel) if ledger else None
    rec = next((c for c in reversed((r or {}).get("records") or []) if c.get("queued_file")), None)
    if rec is None:
        why = (ledger_why or "no ledger was named") if ledger is None else "the ledger has no queue record of this render"
        png = png_graph(path) if path else None
        out["load"] = {"why": why, "graph": png or {"why": "no graph is recoverable: no queue record and no PNG beside the render"}}
        out["windows"] = [{"index": None, "why": f"{why}: its windows and text are not read",
                           "graph": (png or {}).get("prompt"), "graph_from": (png or {}).get("from")}]
        return out
    queued = found(rec["queued_file"], bases)
    load = (rec.get("load") or {}).get("name") or row.get("name")
    says = {"render": (r or {}).get("making"), "record": rec.get("making"), "disagrees": rec.get("disagrees")}
    if queued is None:
        out["load"] = {"why": f"the ledger's queue file {rec['queued_file']} is not found under {', '.join(str(b) for b in bases)}", "ledger_says": says}
        out["windows"] = [{"index": None, "why": out["load"]["why"]}]
        return out
    rd = load_record(queued.parent, load, "render", render=path, output=share)
    rd["ledger_says"] = says
    if path is not None:
        g = rd.get("windows", [{}])[0].get("graph")
        if "PNG" in str((rd.get("graph") or {}).get("from_is")):
            rd["png"] = {"agrees": None, "why": "the graph is the PNG's own (the job folder's graph is gone): not held to itself"}
        else:
            rd["png"] = graphs_agree(g, png_graph(path)) if isinstance(g, dict) else {"agrees": None, "why": "no job graph to hold the PNG to"}
    out["windows"] = rd.pop("windows")
    out["load"] = rd
    return out


def pieces_of_table(table_json, share: Path | None = None, table_rows: list | None = None, laid: list | None = None) -> tuple[list[dict], dict]:
    """The pieces a delivery table JSON lays, and the files it names (its ledger, the list of what is laid).

    With `table_rows` (every row of the .txt the assembler read, as (first, last, piece path)), each JSON row is tied to
    the .txt row of the same frames and file, and a row on either side with no match REFUSES the JSON as the pieces'
    source (SystemExit naming them). With `laid` (the rows the clip lays), only those are read. Each piece is read on its
    own: one that fails is an entry saying why, and the rest are written."""
    import re
    table_json = Path(table_json).resolve()
    t = json.loads(table_json.read_text())
    rows = t.get("rows", [])
    if table_rows is not None:
        key = lambda r: next((k for k in table_rows if k[0] == r.get("first") and k[1] == r.get("last")  # noqa: E731
                              and str(k[2]).endswith("/" + str(r.get("file")))), None)
        tied = [(r, key(r)) for r in rows]
        stray = [f"{r.get('first')}-{r.get('last')} {r.get('file')}" for r, k in tied if k is None]
        unheld = [f"{a}-{b} {piece}" for a, b, piece in table_rows if not any(k == (a, b, piece) for _, k in tied)]
        if stray or unheld:
            raise SystemExit(f"the table's JSON {table_json.name} is not the .txt the clip was laid from: "
                             + "; ".join(x for x in (f"rows of the .txt it does not hold: {', '.join(unheld)}" if unheld else "",
                                                     f"rows it holds that the .txt does not: {', '.join(stray)}" if stray else "") if x))
        rows = [{**r, "txt_row": list(k)} for r, k in tied if laid is None or tuple(k) in {tuple(x) for x in laid}]
    bases = [REPO, Path.cwd(), *table_json.parents]
    made = str(t.get("made") or "")
    named = {}
    m = re.search(r"from the ledger `([^`]+)`", made)
    ledger, ledger_why = None, None
    if m:
        at = found(m.group(1), bases)
        if at is None:
            ledger_why = f"the ledger the table names is not found: {m.group(1)}, looked for under {', '.join(str(b) for b in bases[:3])} and the table's folders"
        else:
            named["the ledger the table was made from"] = str(at)
            ledger = json.loads(at.read_text())
    m = re.search(r"list of what is laid, `([^`]+)`", made)
    if m:
        named["the list of what is laid"] = str(found(m.group(1), bases) or m.group(1))
    if share is None:
        share = Path((ledger or {}).get("share") or os.environ.get("H3_OUTPUT_DIR") or "")
    pieces = []
    for r in rows:
        try:
            pieces.append({**piece_of(r, ledger, share, ledger_why, bases), **({"txt_row": r["txt_row"]} if "txt_row" in r else {})})
        except (Exception, SystemExit) as e:  # noqa: BLE001 - one piece's failure (a reader that stops too) is its entry, never the sidecar's loss
            pieces.append({"name": r.get("name"), "first": r.get("first"), "last": r.get("last"), "file": r.get("file"),
                           "why": f"reading the piece {r.get('name')} failed: {type(e).__name__}: {e}"})
    return pieces, named


def write(video, command: list[str] | None = None, inputs: dict | None = None, extra: dict | None = None, made_by: str | None = None,
          pieces: list[dict] | None = None, windows: list[dict] | None = None, tool=None) -> Path:
    """Write the video's sidecar (`sidecar_path`) beside it. Refuses to write over a file of that name that is not a
    sidecar; over a sidecar of the same bytes, it keeps that one's command, maker and time as `earlier`."""
    video = Path(video)
    if not video.is_file():
        raise SystemExit(f"sidecar: {video} is not a file")
    side = sidecar_path(video)
    if side.exists():
        try:
            old = json.loads(side.read_text())
        except ValueError:
            old = None
        if not isinstance(old, dict) or old.get("schema") != SCHEMA:
            raise SystemExit(f"sidecar: {side} is there and is not a sidecar ({SCHEMA}); it is never written over")
    else:
        old = None
    command = list(command if command is not None else [sys.executable, *sys.argv])
    tool = Path(tool) if tool else next((Path(c) for c in command if str(c).endswith(".py")), None)
    tool = tool.resolve() if tool else None
    out = {"schema": SCHEMA, "video": str(video.resolve()), "video_sha256": sha256_of(video), "video_size": video.stat().st_size,
           "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "made_by": made_by or os.environ.get(SESSION_KEY) or "not said (no --by and no H3_SESSION)",
           "command": " ".join(shlex.quote(str(c)) for c in command), "cwd": os.getcwd(),
           # stills, inputs and the output folder are read through these keys (docs/comfy_notes.md): every H3_ key the command ran with
           "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith("H3_")}, "git": git_state(),
           "tool": ({"path": str(tool), "sha256": sha256_of(tool)} if tool.is_file() else
                    {"path": str(tool), "sha256": None, "why": f"not a file at {tool} (the command's script, taken relative to the cwd {os.getcwd()})"})
           if tool else {"why": "no script in the command"},
           "inputs": [file_row(role, p) for role, p in (inputs or {}).items()], "extra": extra or {}}
    if old and old.get("video_sha256") == out["video_sha256"]:
        out["earlier"] = {k: old.get(k) for k in ("made_at", "made_by", "command", "video_sha256")}
    if pieces is not None:
        out["pieces"] = pieces
    if windows is not None:
        out["windows"] = windows
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=str) + "\n")     # a Path a caller hands in is written as its text
    tmp.replace(side)
    return side


def add_check(video, command: list[str], verdict, verdict_line, check_record, table, made_by: str | None = None) -> Path:
    """Append one check of a built video to its sidecar's `checks`, keeping the build's record whole. `table` is the table
    the check ran against, said as the build's own or not. A video with no sidecar of its bytes gets one that says how it
    was made is not known, and holds the check."""
    video = Path(video)
    try:
        s = read(video)
    except SystemExit as e:
        s = None
        why = str(e)
    if s is None:
        write(video, [], {}, {}, made_by=made_by)
        s = json.loads(sidecar_path(video).read_text())
        s["made_how"] = f"not known: no sidecar of these bytes was there when the check ran ({why}); this one holds checks only"
    built_table = next((x.get("path") for x in s.get("inputs") or [] if x.get("role") == "the table"), None)
    s.setdefault("checks", []).append({
        "command": " ".join(shlex.quote(str(c)) for c in command), "cwd": os.getcwd(),
        "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "made_by": made_by or os.environ.get(SESSION_KEY) or "not said (no --by and no H3_SESSION)",
        "verdict": verdict, "verdict_line": verdict_line, "check_record": file_row("the check record", check_record),
        "table": str(Path(table).resolve()), "table_is_the_builds": None if built_table is None else str(Path(table).resolve()) == built_table})
    side = sidecar_path(video)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(s, indent=1, default=str) + "\n")
    tmp.replace(side)
    return side


def read(video) -> dict:
    """The sidecar of a video; refused when there is none, when it is not of this schema, or when its video_sha256 is
    not the video's bytes now (the video was rewritten after its sidecar)."""
    video = Path(video)
    side = sidecar_path(video)
    if not side.is_file():
        old = video.with_suffix(".json")
        try:
            was = json.loads(old.read_text()).get("schema") if old.is_file() else None
        except (ValueError, AttributeError):
            was = None
        raise SystemExit(f"sidecar: {video.name} has no sidecar ({side.name})" + (f"; an old-name sidecar: {old}, not read" if was == SCHEMA else ""))
    try:
        s = json.loads(side.read_text())
    except ValueError as e:
        raise SystemExit(f"sidecar: {side} does not read: {e}")
    if s.get("schema") != SCHEMA:
        raise SystemExit(f"sidecar: {side} is schema {s.get('schema')!r}, not {SCHEMA!r}")
    now = sha256_of(video)
    if s.get("video_sha256") != now:
        raise SystemExit(f"sidecar: {video.name} is sha256 {now[:16]} now and its sidecar was written for {str(s.get('video_sha256'))[:16]}: "
                         "the video changed after its sidecar")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="print a video's sidecar in short; exit 1 when it does not hold")
    r.add_argument("video")
    a = ap.parse_args()
    try:
        s = read(a.video)
    except SystemExit as e:
        print(f"FAIL {e}")
        return 1
    print(f"{s['video']}: sha256 {s['video_sha256'][:16]}, made {s['made_at']} by {s['made_by']} at {s['git'].get('commit')}"
          f" ({len(s['git'].get('dirty') or [])} files not committed)")
    print(f"command: {s['command']}")
    for x in s["inputs"]:
        print(f"  {x['role']}: {x['path']} {str(x.get('sha256'))[:16]}" + (f" ({x['missing']})" if x.get("missing") else ""))
    for p in s.get("pieces") or []:
        ws = p.get("windows") or []
        print(f"  piece {p.get('name')} {p.get('first')}-{p.get('last')}: {len(ws)} window(s)"
              + "".join(f"; [{w.get('index')}] {w.get('first')}-{w.get('last')} text {str(w.get('text_sha256'))[:12]} graph {str(w.get('graph_sha256'))[:12]}"
                        + (f" ({w['why']})" if w.get("why") else "") for w in ws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
