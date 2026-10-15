#!/usr/bin/env python3
"""What H3 was handed for one load, read from the graph that was queued: every file, its digest, what the graph does to it.

    H3_INPUT_DIR=... H3_OUTPUT_DIR=... <python> bench/what_h3_got.py --job-folder data/<day>_<load>_job --load <load> \\
        [--stage render|plan] [--json OUT.json]

**What it buys.** Until 2026-10-15 a review strip showed the reference video and the still that its command line
named, and the land step filled those from the JOB file (`load_chain.py`, the stack step), never from the graph
that ran. On the loads read that day the paths matched, but nothing held them to it, and a load of two subjects
was handed one reference and one still of its two. This is the one reading of what a load was given. The strip,
its sidecar and the joined review clips read their inputs here, so a panel shows a file because the graph
named it.

**What is read from where.**
- The graph: `<load>.<stage>.json` in the job folder, ComfyUI's API form, the graph that is queued.
- Whether and when it was queued: `<load>.<stage>.queued.json` (`queued_at`, `files`, the node's own `said`).
  With no such file the graph is "built, not queued": the files are what WILL be sent, as of now.
- Each input: every file loader above each linked input of the Masked Source (`frames` is the source, `mask`,
  `parts`, `others`, `motion_video` is `<Video 1>`, `motion_video_2` is `<Video 2>`), above the song node's
  `audio`, and each still of the song node's `references` chain in the order the model gets them, `<Picture k>`
  (`workflows/masked_job_graph.py::stills_of`, the builder's own reading). The `<Video k>` / `<Picture k>` tags
  follow the job builder's rule (`bench/masked_job.py::pair_of`); the node's own words for each tag, where it
  said any, are kept beside it (`node_said`).
- For a video loader: its start, its frame cap, the source frame its frame 0 is on, and what the loader does to
  the file's frames (VideoHelperSuite cuts the frames to the target's shape at the centre when the shapes
  differ, then scales: `load_video_nodes.py`, the ffmpeg path's `vfilters`). The rule is restated here, as the
  file's own size and the crop box it gives, so a panel can be drawn as the loader drew it.
- A reference video's own record (`<stem>.json` beside it: what it is, the source frame its frame 0 is on) when
  there is one.
- The zoom: the Masked Source's `render_region`, and after the render the node's `<stem>_zoomed.json` beside the
  rendered file and each window's `_region.npz` meta.
- The text: the song node's `prompt`, as the graph holds it, with its sha256.

**Hard failures, the only red** (exit 1): the graph names a file that is not there; a queued load names a file
written after `queued_at`, so what H3 read cannot be shown from it; a digest recorded at queue time
(`digests`, path to sha256) that is not the file's now. Everything else is printed as data.

**What it cannot see.** The tensors inside the node: the masked plate the model sampled from (noise inside the
region, the start blur), the frames of a reference the text encoder was shown (the node says how many, not
which), anything the node computed. A file written after the queue with the same bytes reads as changed: the
time is all a record without digests carries. A loader other than VideoHelperSuite's ffmpeg-path loader and
ComfyUI's `LoadImage` is listed by its class with no file.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(REPO / "workflows"))

FPS = 24
VIDEO_LOADER, IMAGE_LOADER = "VHS_LoadVideoFFmpegPath", "LoadImage"
SONG, SOURCE = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource"
#: The Masked Source's linked inputs that carry a picture, and the name each is shown under. **Inherited** from the
#: node's own inputs (`video_mask.py`, `MiniMaxH3MaskedSource.define_schema`); one not listed is shown by its own name.
ROLES = {"frames": "the source", "mask": "the subject's mask", "parts": "the parts kept", "others": "the keep-outs",
         "motion_video": "<Video 1>", "motion_video_2": "<Video 2>"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def frame_of(seconds: float, rate: float = FPS) -> int:
    """A loader's start time as a frame: the job graphs set it a fraction of a frame early so the loader lands on
    the frame (`bench/stack_delivery.py::frame_of`, the same rule)."""
    return math.ceil(seconds * rate - 1e-3)


def video_size(path: Path) -> list[int] | None:
    got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return [int(x) for x in got.split(",")[:2]] if got else None


def vhs_fit(size: list[int], custom_w: int, custom_h: int, ratio: int = 8) -> dict:
    """What VideoHelperSuite's ffmpeg-path loader does to frames of `size` at a custom width and height, restated
    from `load_video_nodes.py::target_size` and the `vfilters` beside it: the target rounded to `ratio`, and when
    its shape differs from the file's by a pixel or more, a cut at the centre to the target's shape before the
    scale. `crop` is [x, y, w, h] in the file's pixels."""
    w, h = size
    if custom_w == 0 and custom_h == 0:
        tw, th = w, h
    elif custom_h == 0:
        tw, th = custom_w, h * custom_w / w
    elif custom_w == 0:
        tw, th = w * custom_h / h, custom_h
    else:
        tw, th = custom_w, custom_h
    tw, th = int(tw / ratio + 0.5) * ratio, int(th / ratio + 0.5) * ratio
    if custom_w == 0 and custom_h == 0:
        return {"to": [tw, th], "crop": None, "words": f"loaded at its own {w}x{h}"}
    # The loader's test (`abs(w * ar - h) >= 1`) compares a width times the target's shape with a height, so it
    # passes for any frame and the crop filter is always added; ffmpeg's crop then cuts nothing when the shapes
    # match. What is restated is the filter's own expression: the cut it makes, or none
    ar = tw / th
    cw, ch = (w, w / ar) if ar > w / h else (h * ar, h)
    cw, ch = min(w, round(cw)), min(h, round(ch))
    crop = None if (cw, ch) == (w, h) else [(w - cw) // 2, (h - ch) // 2, cw, ch]
    words = (f"cut at the centre to {crop[2]}x{crop[3]} of its {w}x{h}, then scaled to {tw}x{th}" if crop
             else f"scaled from {w}x{h} to {tw}x{th}" if [w, h] != [tw, th] else f"loaded at its own {w}x{h}")
    return {"to": [tw, th], "crop": crop, "words": words}


def loaders_above(g: dict, link, seen: set | None = None) -> list[str]:
    """Every file loader a linked input reads from, nearest first: the walk goes up through masks and composites."""
    seen = set() if seen is None else seen
    if not (isinstance(link, list) and len(link) == 2 and str(link[0]) in g) or str(link[0]) in seen:
        return []
    nid = str(link[0])
    seen.add(nid)
    node = g[nid]
    if node["class_type"] in (VIDEO_LOADER, IMAGE_LOADER):
        return [nid]
    out = []
    for v in node["inputs"].values():
        out += loaders_above(g, v, seen)
    return out


def node_said(queued: dict | None) -> dict[str, list[str]]:
    """The node's own lines that name a tag (`<Picture 1> costs: ...`), by tag, as it said them."""
    out: dict[str, list[str]] = {}
    for lines in ((queued or {}).get("said") or {}).values():
        for line in "\n".join(lines).splitlines():
            for tag in re.findall(r"<(?:Picture|Video) \d+>", line):
                out.setdefault(tag, [])
                if line not in out[tag]:
                    out[tag].append(line)
    return out


def one_file(g: dict, nid: str, role: str, tag: str | None, inputs: Path, queued_at: datetime.datetime | None,
             recorded: dict, digests: bool) -> dict:
    node = g[nid]
    ins = node["inputs"]
    named = ins.get("video") if node["class_type"] == VIDEO_LOADER else ins.get("image")
    path = Path(named) if Path(named).is_absolute() else inputs / named
    row = {"role": role, "tag": tag, "node": nid, "class": node["class_type"], "named": named, "file": str(path),
           "exists": path.is_file()}
    if not row["exists"]:
        return row
    st = path.stat()
    written = datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc)
    row.update(bytes=st.st_size, written=written.isoformat(timespec="seconds"),
               written_after_queue=None if queued_at is None else written > queued_at)
    if digests:
        row["sha256"] = sha256(path)
    if str(path) in recorded:
        row["sha256_at_queue"] = recorded[str(path)]
    if node["class_type"] == VIDEO_LOADER:
        size = video_size(path)
        start = float(ins.get("start_time", 0.0))
        row["loader"] = {"start_time": start, "first_frame": frame_of(start), "frame_load_cap": int(ins.get("frame_load_cap", 0)),
                         "custom": [int(ins.get("custom_width", 0)), int(ins.get("custom_height", 0))]}
        row["size"] = size
        if size and role != "the sound":
            row["fit"] = vhs_fit(size, *row["loader"]["custom"])
        own = path.with_suffix(".json")
        if own.is_file() and role.startswith("<Video"):
            try:
                rec = json.loads(own.read_text())
            except ValueError:
                rec = {}
            row["record"] = {k: rec[k] for k in ("what", "its_first_frame_is_source_frame", "frames", "size", "made_by", "made_at") if k in rec}
            row["record"]["file"] = str(own)
    else:
        try:
            from PIL import Image
            with Image.open(path) as im:
                row["size"] = [im.width, im.height]
        except OSError:
            row["size"] = None
    return row


def read(job_folder: Path, load: str, stage: str = "render", *, inputs: Path | None = None, output: Path | None = None,
         digests: bool = True) -> dict:
    """Everything a load was handed, as one dict; `failures` holds the hard failures, empty when there are none."""
    import masked_job_graph
    job_folder = Path(job_folder)
    graph_file = job_folder / f"{load}.{stage}.json"
    if not graph_file.is_file():
        raise SystemExit(f"no graph {graph_file.name} in {job_folder}: the load is not built at stage {stage!r}")
    g = json.loads(graph_file.read_text())
    queued_file = job_folder / f"{load}.{stage}.queued.json"
    queued = json.loads(queued_file.read_text()) if queued_file.is_file() else None
    queued_at = datetime.datetime.fromisoformat(queued["queued_at"]) if queued and queued.get("queued_at") else None
    if queued_at is not None and queued_at.tzinfo is None:
        raise SystemExit(f"{queued_file.name}: queued_at {queued['queued_at']!r} names no time zone; it cannot be held to a file's time")
    recorded = (queued or {}).get("digests") or {}
    if inputs is None:
        import masked_job
        inputs = masked_job.input_dir()
    out = {"load": load, "stage": stage, "graph": str(graph_file), "graph_sha256": sha256(graph_file),
           "queued": None if queued is None else {k: queued.get(k) for k in ("prompt_id", "queued_at", "status", "commit", "files")},
           "state": "queued" if queued else "built, not queued", "inputs": [], "failures": []}
    said = node_said(queued)
    sources = [nid for nid, n in g.items() if n["class_type"] == SOURCE]
    songs = [nid for nid, n in g.items() if n["class_type"] == SONG]
    if len(sources) != 1 or len(songs) != 1:
        raise SystemExit(f"{graph_file.name} holds {len(sources)} Masked Source and {len(songs)} song nodes; a load has one of each")
    src, song = g[sources[0]]["inputs"], g[songs[0]]["inputs"]
    take = lambda nid, role, tag: out["inputs"].append(one_file(g, nid, role, tag, inputs, queued_at, recorded, digests))
    for key, link in src.items():
        for nid in loaders_above(g, link):
            role = ROLES.get(key, key)
            take(nid, role, role if role.startswith("<") else None)
    for nid in loaders_above(g, song.get("audio")):
        take(nid, "the sound", None)
    for k, (append, loader) in enumerate(masked_job_graph.stills_of(g), 1):
        take(loader, f"<Picture {k}>", f"<Picture {k}>")
        out["inputs"][-1]["append"] = {"node": append, **{x: v for x, v in g[append]["inputs"].items() if not isinstance(v, list)}}
    for row in out["inputs"]:
        if row["tag"] in said:
            row["node_said"] = said[row["tag"]]
    out["text"] = {"node": songs[0], "text": song.get("prompt", ""), "sha256": hashlib.sha256(str(song.get("prompt", "")).encode()).hexdigest()}
    out["replaces"] = {k: src.get(k) for k in ("replace", "part_phrases", "paint_out") if k in src}
    out["zoom"] = {"render_region": src.get("render_region"), "piece": None, "windows": []}
    files = (queued or {}).get("files") or []
    if files:
        if output is None:
            import h3_config
            output = h3_config.output_dir()
        piece = output / files[0]
        out["render"] = str(piece)
        zoomed = piece.with_name(piece.stem + "_zoomed.json")
        if zoomed.is_file():
            out["zoom"]["piece"] = {**json.loads(zoomed.read_text()), "record": str(zoomed)}
        windows = piece.with_name(piece.stem.rsplit("_", 1)[0] + "_windows")
        for npz in sorted(windows.glob("*_region.npz")) if windows.is_dir() else []:
            import numpy as np
            with np.load(npz) as z:
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            zm = npz.with_name(npz.name.replace("_region.npz", "_zoomed.mp4"))
            out["zoom"]["windows"].append({"region": str(npz), "zoom": meta.get("zoom"), "first_frame": meta.get("first_frame"),
                                           "frames": meta.get("frames"), "zoomed": str(zm) if zm.is_file() else None})
    for row in out["inputs"]:
        where = f"{row['role']} (node {row['node']}, {row['named']})"
        if not row["exists"]:
            out["failures"].append(f"{where}: the graph names it and it is not there")
        elif row.get("written_after_queue"):
            out["failures"].append(f"{where}: written {row['written']}, after the load was queued at {queued_at.isoformat() if queued_at else ''}: "
                                   "what H3 read cannot be shown from this file")
        if row.get("sha256_at_queue") and row.get("sha256") and row["sha256_at_queue"] != row["sha256"]:
            out["failures"].append(f"{where}: its sha256 is not the one recorded at queue time")
    return out


def said_lines(got: dict) -> list[str]:
    """The reading as lines a person reads: one an input, then the text, the zoom and the failures."""
    lines = [f"{got['load']} ({got['stage']}): {got['state']}" + (f" at {got['queued']['queued_at']}" if got["queued"] else "")]
    for r in got["inputs"]:
        if not r["exists"]:
            lines.append(f"  {r['role']}: node {r['node']} names {r['named']}: NOT THERE")
            continue
        how = r.get("fit", {}).get("words") or (f"{r['size'][0]}x{r['size'][1]}" if r.get("size") else "")
        at = r.get("loader")
        lines.append(f"  {r['role']}: {r['file']}" + (f"  sha256 {r['sha256'][:16]}" if r.get("sha256") else "")
                     + (f"  from its frame {at['first_frame']}, {at['frame_load_cap']} frames" if at else "") + (f"; {how}" if how else ""))
        if r.get("append"):
            lines.append(f"      appended as {r['tag']}: " + ", ".join(f"{k} {v}" for k, v in r["append"].items() if k != "node"))
        if r.get("record", {}).get("what"):
            lines.append(f"      its record: {r['record']['what']}; its frame 0 is source frame {r['record'].get('its_first_frame_is_source_frame')}")
        for line in r.get("node_said", []):
            lines.append(f"      the node: {line.strip()}")
    z = got["zoom"]
    if z.get("piece"):
        p = z["piece"]
        lines.append(f"  zoom: {p['zoom']:.2f}x on the box {p['box']} of the {p['source_size'][0]}x{p['source_size'][1]} frames ({p.get('set_by')})")
    else:
        lines.append(f"  zoom: render_region {z['render_region']!r}" + ("; no zoomed record beside the render" if got.get("render") else ""))
    lines.append(f"  text: {len(got['text']['text'])} characters, sha256 {got['text']['sha256'][:16]}, from node {got['text']['node']}")
    lines += [f"FAILED: {x}" for x in got["failures"]]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--job-folder", required=True, help="data/<day>_<load>_job")
    p.add_argument("--load", required=True)
    p.add_argument("--stage", default="render", choices=("render", "plan"))
    p.add_argument("--json", help="write the whole reading here")
    a = p.parse_args()
    got = read(Path(a.job_folder), a.load, a.stage)
    print("\n".join(said_lines(got)))
    if a.json:
        Path(a.json).write_text(json.dumps(got, indent=1) + "\n")
        print("wrote", a.json)
    sys.exit(1 if got["failures"] else 0)


if __name__ == "__main__":
    main()
