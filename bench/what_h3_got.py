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
- The latent a load continues from (the song node's `continue_from`): a file like the others.
- The input folder a still is named under: the one the queue step recorded (`input_dir`), else `H3_INPUT_DIR`;
  with neither it refuses, since the stock folder is not where this box's server reads stills from.

**Hard failures, the only red** (exit 1): the graph names a file that is not there; a queued load names a file
written more than a second after `queued_at` (the record keeps whole seconds, taken after the post), so which
version H3 read is not known from it; a digest recorded at queue time
(`digests`, path to sha256) that is not the file's now. Everything else is printed as data.

**A mask video the plan read, rebuilt in place** (`--stage plan`). `load_chain`'s on-plan step builds the capture again
in place right after the plan has run: the same frames, new bytes and a new time, so by the two rules above every mask
the plan read failed (from 631a9206, when the digests landed, until 2026-10-15). Before it builds, that step copies the
capture whole to `<capture>_first`. So a file whose bytes are not the ones recorded at queue time is held to its frames
when the copy beside its capture has exactly those bytes, which makes the copy what the plan was queued with: decoding
equal to the copy frame by frame, it reads "rebuilt in place, frames as planned"; not equal, it fails and names the
first frame that differs. The step's own record of the plan's frames (`<load>.plan.masks.json`, which the render step
holds the masks to) is not read here: nothing ties it to the copy. With no copy, or a copy whose bytes are not the
queue's, the two rules above stand. A file's time is not read on this path: the bytes recorded at the queue are the
anchor. What it cannot see: a mask changed and changed back between the queue and the copy. It decodes every such mask
and its copy, which takes seconds a mask on a long capture. A file outside a capture that was copied (stills, the
source, a reference made elsewhere) keeps the whole-file rule.

- Which frames of a wired reference the text encoder is shown, per window: the node cuts the window's reference
  (the wired video from the window's first frame, its last frame repeated past its end: `video_mask.wired_motion`)
  to 17k+5 frames and samples every twelfth (`reference_conditioning.py::_prepare_reference_video` and the
  `sample_indices` line, `FPS // 2`), restated here as `encoder_frames`. The windows are the rendered ones' region
  meta, else the node's own "[k] window from S s: N frames" lines; before either there are none to say.

**What it cannot see.** What the composite kept of the render (the node's `changed_alpha`), anything else the node
computed. The masked plate is not saved either; with `paint_out` off and the start from noise it is the fitted
frames the loader hands on, noised inside the region, and can be drawn from them. Whether a mask or keep-out the loader hands on uncut was drawn on the source
as the loader cuts it: the graph names a file, not how it was made (the lane's tools fit the source with
`_lib.frames.FIT`, which restates the loader's cut); the reading lists which loaders cut and which do not. A file written after the queue with the same bytes reads as
changed, and one written between the queue and the run is what H3 read and still fails: the time is all a record
without digests carries. A loader other than VideoHelperSuite's video and audio loaders and ComfyUI's `LoadImage`
is listed by its class with no file. The loader's rounding is read from its own format table; a `vae` wired to it,
or a format a plugin added, sets a rounding this does not restate, and the cut is then not given. A graph that
appends a reference video (`MiniMaxH3AppendRefVideo`) shifts the node's `<Video k>` labels and is refused.
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
sys.path.insert(0, str(BENCH))
from _lib import COMFY  # noqa: E402

VIDEO_LOADER, IMAGE_LOADER, AUDIO_LOADER = "VHS_LoadVideoFFmpegPath", "LoadImage", "VHS_LoadAudio"
#: Each loader this reads, and the input that names its file. **Inherited** from the loaders' own inputs.
FILE_KEY = {VIDEO_LOADER: "video", IMAGE_LOADER: "image", AUDIO_LOADER: "audio_file"}
#: VideoHelperSuite's loader, whose format table and crop line are read and restated here.
#: Found under ComfyUI's own custom_nodes (`_lib.COMFY`), so a worktree of this pack outside it reads the same file.
VHS_LOADER_FILE = COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite" / "videohelpersuite" / "load_video_nodes.py"
SONG, SOURCE = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource"
#: The Masked Source's linked inputs that carry a picture, and the name each is shown under. **Inherited** from the
#: node's own inputs (`video_mask.py`, `MiniMaxH3MaskedSource.define_schema`); one not listed is shown by its own name.
ROLES = {"frames": "the source", "mask": "the subject's mask", "parts": "the parts kept", "others": "the keep-outs",
         "motion_video": "<Video 1>", "motion_video_2": "<Video 2>", "keep": "what is kept the original (`keep`)"}
#: The Masked Source's lazy inputs and which `replace` asks for each (`video_mask.py`, `check_lazy_status` and the
#: LAZY_FOR_* names). **Inherited**; a linked lazy input `replace` does not ask for is listed and said to be unread.
LAZY = {"mask": None, "shot_table": None, "segmenter": ("head and hair",), "segmenter_clip": ("head and hair",),
        "parts": ("the wired parts",)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def frame_of(seconds: float, rate: float) -> int:
    """A loader's start time as a frame: the job graphs set it a fraction of a frame early so the loader lands on
    the frame (`bench/stack_delivery.py::frame_of`, the same rule)."""
    return math.ceil(seconds * rate - 1e-3)


def video_probe(path: Path) -> dict | None:
    """The file's first video stream: its size, pixel format and frame rate."""
    got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,pix_fmt,r_frame_rate,nb_frames",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    streams = (json.loads(got or "{}").get("streams") or [])
    if not streams:
        return None
    s = streams[0]
    num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
    return {"size": [int(s["width"]), int(s["height"])], "pix_fmt": s.get("pix_fmt"), "rate": float(num) / float(den or 1),
            "frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None}


def vhs_formats() -> dict:
    """VideoHelperSuite's own `VHSLoadFormats`, read from its file as a literal, never restated."""
    import ast
    tree = ast.parse(VHS_LOADER_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VHSLoadFormats" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no VHSLoadFormats in {VHS_LOADER_FILE}: the loader this restates has changed")


#: ffmpeg's crop on a format whose colour is held at half size rounds its offsets and its size down to the
#: colour's step (`vf_crop.c`, `hsub`/`vsub`). **Inherited**; the step by pixel format, for the formats seen here.
CHROMA_STEP = {"yuv420p": (2, 2), "yuvj420p": (2, 2), "yuv422p": (2, 1), "yuvj422p": (2, 1), "yuv411p": (4, 1),
               "yuv420p10le": (2, 2), "yuv422p10le": (2, 1), "nv12": (2, 2)}


def vhs_fit(size: list[int], custom_w: int, custom_h: int, ratio: int = 8, pix_fmt: str | None = None) -> dict:
    """What VideoHelperSuite's ffmpeg-path loader does to frames of `size` at a custom width and height, restated
    from `load_video_nodes.py::target_size` and the `vfilters` beside it: the target rounded to `ratio`, and when
    its shape differs from the file's by a pixel or more, a cut at the centre to the target's shape before the
    scale. `crop` is [x, y, w, h] in the file's pixels, rounded down to the colour's step as ffmpeg's crop does on
    a `pix_fmt` that holds colour at half size."""
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
    cw, ch = min(w, int(cw)), min(h, int(ch))            # ffmpeg takes the expression's value as a whole number, cut
    sx, sy = CHROMA_STEP.get(pix_fmt or "", (1, 1))
    cw, ch = cw - cw % sx, ch - ch % sy
    x, y = (w - cw) // 2, (h - ch) // 2
    crop = None if (cw, ch) == (w, h) else [x - x % sx, y - y % sy, cw, ch]
    words = (f"cut at the centre to {crop[2]}x{crop[3]} of its {w}x{h}, then scaled to {tw}x{th}" if crop
             else f"scaled from {w}x{h} to {tw}x{th}" if [w, h] != [tw, th] else f"loaded at its own {w}x{h}")
    return {"to": [tw, th], "crop": crop, "words": words}


def digests_of(g: dict, inputs: Path) -> dict[str, str]:
    """Every file a graph's loaders name, and the latent a load continues from, by its path as `read` resolves it,
    to its sha256: for the queue step to keep beside the graph (`digests` in `<stem>.queued.json`), so a later
    reading is held to the bytes, not a time. A record from before that has none. A named file that is not there
    is left out: `read` fails on it by name."""
    out = {}
    for node in g.values():
        named = node["inputs"].get(FILE_KEY.get(node["class_type"], "")) if node["class_type"] in FILE_KEY else \
            node["inputs"].get("continue_from") if node["class_type"] == SONG else None
        if isinstance(named, str) and named.strip():
            path = Path(named) if Path(named).is_absolute() else inputs / named
            if path.is_file() and str(path) not in out:
                out[str(path)] = sha256(path)
    return out


def loaders_above(g: dict, link, seen: set | None = None) -> list[str]:
    """Every file loader a linked input reads from, nearest first: the walk goes up through masks and composites."""
    seen = set() if seen is None else seen
    if not (isinstance(link, list) and len(link) == 2 and str(link[0]) in g) or str(link[0]) in seen:
        return []
    nid = str(link[0])
    seen.add(nid)
    node = g[nid]
    linked = [v for v in node["inputs"].values() if isinstance(v, list) and len(v) == 2 and str(v[0]) in g]
    if node["class_type"] in FILE_KEY or not linked:
        return [nid]                 # a loader, or a node that reads nothing from the graph: where a picture begins
    out = []
    for v in node["inputs"].values():
        out += loaders_above(g, v, seen)
    return out


def node_said(queued: dict | None, text: str = "") -> dict[str, list[str]]:
    """The node's own lines that name a tag (`<Picture 1> costs: ...`), by tag, as it said them. A preview that
    echoes the prompt names the tags too: a line that is a line of the text is the text's, not the node's."""
    out: dict[str, list[str]] = {}
    of_text = {x.strip() for x in str(text).splitlines() if x.strip()}
    for lines in ((queued or {}).get("said") or {}).values():
        for line in "\n".join(lines).splitlines():
            if line.strip() in of_text:
                continue
            for tag in re.findall(r"<(?:Picture|Video) \d+>", line):
                out.setdefault(tag, [])
                if line not in out[tag]:
                    out[tag].append(line)
    return out


def one_file(g: dict, nid: str, role: str, tag: str | None, inputs: Path, queued_at: datetime.datetime | None,
             recorded: dict, digests: bool, named: str | None = None) -> dict:
    """One input's row. `named` is the file when it is not a loader's own input (the song node's `continue_from`)."""
    node = g[nid]
    ins = node["inputs"]
    named = named if named is not None else ins.get(FILE_KEY[node["class_type"]]) if node["class_type"] in FILE_KEY else None
    if not isinstance(named, str):
        return {"role": role, "tag": tag, "node": nid, "class": node["class_type"], "named": None, "file": None, "exists": None,
                "note": "a node this reading does not know as a loader: listed by its class, with no file"}
    path = Path(named) if Path(named).is_absolute() else inputs / named
    row = {"role": role, "tag": tag, "node": nid, "class": node["class_type"], "named": named, "file": str(path),
           "exists": path.is_file()}
    if not row["exists"]:
        return row
    st = path.stat()
    written = datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc)
    row.update(bytes=st.st_size, written=written.isoformat(timespec="seconds"),
               written_after_queue=None if queued_at is None else written > queued_at + datetime.timedelta(seconds=1))
    if digests:
        row["sha256"] = sha256(path)
    if str(path) in recorded:
        row["sha256_at_queue"] = recorded[str(path)]
    if node["class_type"] == VIDEO_LOADER and named is ins.get("video"):
        probe = video_probe(path) or {}
        start, forced = float(ins.get("start_time", 0.0)), float(ins.get("force_rate", 0.0) or 0.0)
        rate = forced or probe.get("rate") or 0.0
        fmt = vhs_formats().get(str(ins.get("format", "None")))
        row["loader"] = {"start_time": start, "rate": rate, "first_frame": frame_of(start, rate) if rate else None,
                         "frame_load_cap": int(ins.get("frame_load_cap", 0)), "format": ins.get("format"),
                         "custom": [int(ins.get("custom_width", 0)), int(ins.get("custom_height", 0))]}
        row["size"], row["pix_fmt"], row["file_frames"] = probe.get("size"), probe.get("pix_fmt"), probe.get("frames")
        if isinstance(ins.get("vae"), list):
            row["fit_not_given"] = "a vae is wired to the loader: its rounding is the vae's, not restated here"
        elif fmt is None:
            row["fit_not_given"] = f"format {ins.get('format')!r} is not in the loader's own table: a plugin's, not restated here"
        elif row["size"] and role != "the sound":
            row["fit"] = vhs_fit(row["size"], *row["loader"]["custom"], ratio=(fmt.get("dim") or (1,))[0], pix_fmt=row["pix_fmt"])
            if fmt.get("frames"):
                row["fit"]["frames_rule"] = f"the format keeps {fmt['frames'][0]}k+{fmt['frames'][1]} frames"
        own = path.with_suffix(".json")
        if own.is_file() and role.startswith("<Video"):
            try:
                rec = json.loads(own.read_text())
            except ValueError:
                rec = {}
            row["record"] = {k: rec[k] for k in ("what", "its_first_frame_is_source_frame", "frames", "size", "made_by", "made_at", "box", "box_is",
                                                 "scale", "a_point_of_the_canvas_lands_at", "drawn", "face_height_px") if k in rec}
            ins_ = rec.get("inputs") or {}
            if ins_.get("points"):
                row["record"]["points"] = {"file": ins_["points"], "first_row_is_source_frame": ins_.get("their_first_row_is_source_frame")}
            row["record"]["file"] = str(own)
    elif node["class_type"] == AUDIO_LOADER:
        row["loader"] = {k: ins.get(k) for k in ("seek_seconds", "duration")}
    elif node["class_type"] == IMAGE_LOADER:
        try:
            from PIL import Image
            with Image.open(path) as im:
                row["size"] = [im.width, im.height]
        except OSError:
            row["size"] = None
    return row


def frame_md5s(path: str) -> list[str]:
    """Each decoded frame's md5, in order (`ffmpeg -f framemd5`, the reading `load_chain.frame_hash` hashes whole)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return [line.split()[-1] for line in out.splitlines() if line and not line.startswith("#")]


def first_copy(path: Path) -> Path | None:
    """The same file in the copy of its capture `load_chain`'s on-plan step kept before building it again
    (`<capture>_first`, the nearest such folder above the file), when there is one."""
    for folder in path.parents:
        if not folder.name:
            break
        copy = folder.with_name(folder.name + "_first") / path.relative_to(folder)
        if copy.is_file():
            return copy
    return None


def frames_as_planned(path: str, copy: Path) -> dict:
    """A mask video held to the frames of the copy the on-plan step kept, frame by frame: {"equal": bool, "how": words}.
    The record's one hash of every frame (`<load>.plan.masks.json`) is not a shortcut here: nothing ties it to the copy,
    and a record and a file that agree with each other and not with the copy would read clean (a cold reader's case,
    2026-10-15)."""
    a, b = frame_md5s(str(copy)), frame_md5s(path)
    if a == b:
        return {"equal": True, "how": f"every one of its {len(b)} frames decodes equal to the copy's"}
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None:
        return {"equal": False, "how": f"it holds {len(b)} frames and the copy {len(a)}, the first {min(len(a), len(b))} the same"}
    return {"equal": False, "how": f"its frame {k} (counted from 0 in the file) is not the copy's"}


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
        import os
        told = (queued or {}).get("input_dir") or os.environ.get("H3_INPUT_DIR")
        if not told:
            raise SystemExit("which folder the server read stills from is not known: the queued record names no input_dir and "
                             "H3_INPUT_DIR is not set. Set it to the folder the server was started with (--input-directory)")
        inputs = Path(told)
    out = {"load": load, "stage": stage, "graph": str(graph_file), "graph_sha256": sha256(graph_file),
           "queued": None if queued is None else {k: queued.get(k) for k in ("prompt_id", "queued_at", "status", "commit", "files")},
           "state": "queued" if queued else "built, not queued", "inputs": [], "failures": []}
    sources = [nid for nid, n in g.items() if n["class_type"] == SOURCE]
    songs = [nid for nid, n in g.items() if n["class_type"] == SONG]
    if len(sources) != 1 or len(songs) != 1:
        raise SystemExit(f"{graph_file.name} holds {len(sources)} Masked Source and {len(songs)} song nodes; a load has one of each")
    appended = [nid for nid, n in g.items() if n["class_type"] == "MiniMaxH3AppendRefVideo"]
    if appended:
        raise SystemExit(f"{graph_file.name} appends a reference video (node {', '.join(appended)}): the node's <Video k> labels then "
                         "count it, and this reading labels by the Masked Source's inputs alone. Not read")
    src, song = g[sources[0]]["inputs"], g[songs[0]]["inputs"]
    said = node_said(queued, song.get("prompt", ""))
    take = lambda nid, role, tag, named=None: out["inputs"].append(one_file(g, nid, role, tag, inputs, queued_at, recorded, digests, named))
    for key, link in src.items():
        for nid in loaders_above(g, link):
            role = ROLES.get(key, key)
            take(nid, role, role if role.startswith("<") else None)
            if key in LAZY and LAZY[key] is not None and src.get("replace") not in LAZY[key]:
                out["inputs"][-1]["unread"] = (f"the node does not ask for `{key}` when replace is {src.get('replace')!r} "
                                               "(`check_lazy_status`): wired, and not read")
    trims = [g[str(song["audio"][0])]["inputs"]] if isinstance(song.get("audio"), list) and \
        g.get(str(song["audio"][0]), {}).get("class_type") == "TrimAudioDuration" else []
    for nid in loaders_above(g, song.get("audio")):
        take(nid, "the sound", None)
        if trims:
            out["inputs"][-1]["trim"] = {k: v for k, v in trims[0].items() if not isinstance(v, list)}
    if isinstance(song.get("continue_from"), str) and song["continue_from"].strip():
        take(songs[0], "continued from (an earlier window's latent)", None, song["continue_from"])
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
    # the canvas each video loader hands on, and which of them cut: a mask that lands on the same canvas uncut was
    # drawn there by whoever made it, which the graph does not say (`what it cannot see`)
    fits = [r for r in out["inputs"] if r.get("fit")]
    out["canvases"] = {"sizes": sorted({tuple(r["fit"]["to"]) for r in fits}),
                       "cut": [{"role": r["role"], "node": r["node"], "crop": r["fit"]["crop"], "of": r["size"]} for r in fits if r["fit"]["crop"]],
                       "uncut": [r["node"] for r in fits if not r["fit"]["crop"]]}
    windows = windows_of(out, queued)
    source_first = next((r.get("loader", {}).get("first_frame") for r in out["inputs"] if r["role"] == "the source"), None)
    for row in out["inputs"]:
        if row.get("tag", "") and str(row["tag"]).startswith("<Video") and row.get("exists") and windows:
            cap = row["loader"]["frame_load_cap"]
            have = min(x for x in (cap or None, row.get("file_frames")) if x) if (cap or row.get("file_frames")) else None
            if have:
                row["reference_frames"] = have
                row["encoder_frames"] = [encoder_frames(a, n, have, source_first) for a, n in windows]
                # the node's own count of what it showed, an answer from the other side: "[k] <Video j> costs: ... shown N pictures"
                for k, e in enumerate(row["encoder_frames"], 1):
                    for line in row.get("node_said", []):
                        m = re.match(rf"\[{k}\] {re.escape(row['tag'])} costs: the text encoder is shown (\d+) pictures", line.strip())
                        if m:
                            e["node_says"] = int(m.group(1))
    for row in out["inputs"]:
        where = f"{row['role']} (node {row['node']}, {row['named']})"
        copy = first_copy(Path(row["file"])) if stage == "plan" and row.get("exists") and row.get("sha256_at_queue") \
            and row.get("sha256") and row["sha256"] != row["sha256_at_queue"] else None
        if copy is not None and sha256(copy) == row["sha256_at_queue"]:
            got = frames_as_planned(row["file"], copy)
            row["frames_as_planned"] = {"copy": str(copy), **got}
            if got["equal"]:
                row["rebuilt"] = (f"rebuilt in place after the plan ran ({row['written']}), frames as planned: {got['how']}; the copy "
                                  f"{copy} has the bytes recorded at queue time")
            else:
                out["failures"].append(f"{where}: rebuilt in place after the plan ran, and its frames are not what the plan read: "
                                       f"{got['how']} ({copy}, the bytes recorded at queue time)")
            continue
        if row["exists"] is False:
            out["failures"].append(f"{where}: the graph names it and it is not there")
        elif row.get("written_after_queue"):
            out["failures"].append(f"{where}: written {row['written']}, after the load was queued at {queued_at.isoformat() if queued_at else ''}: "
                                   "what H3 read cannot be shown from this file")
        if row.get("sha256_at_queue") and row.get("sha256") and row["sha256_at_queue"] != row["sha256"]:
            out["failures"].append(f"{where}: its sha256 is not the one recorded at queue time")
    return out


#: The text encoder's sampling of a reference video. **Inherited** from `reference_conditioning.py`: the frame
#: count cut down until it is 17k+5 (`_prepare_reference_video`), then every `FPS // 2`th frame at 24 fps.
ENCODER_STEP, FRAMES_RULE = 12, (17, 5)


def encoder_frames(window_first: int, window_frames: int, reference_frames: int, source_first: int | None) -> dict:
    """Which frames of a wired reference the text encoder is shown in one window. `window_first` is the window's
    first frame on the load's clock, `reference_frames` how many frames the loader hands on (the cap, or the file's
    own count when shorter). A window frame past the reference's end is its last frame repeated (`held`)."""
    n = window_frames
    while n >= 5 and n % FRAMES_RULE[0] != FRAMES_RULE[1]:
        n -= 1
    out = []
    for i in range(0, n, ENCODER_STEP):
        at = window_first + i
        out.append({"window_frame": i, "reference_frame": min(at, reference_frames - 1), "held": at >= reference_frames,
                    "source_frame": None if source_first is None else source_first + at})
    return {"window_first": window_first, "window_frames": window_frames, "kept_frames": n, "sampled": out}


def windows_of(got: dict, queued: dict | None) -> list[tuple[int, int]]:
    """Each window as (its first frame on the load's clock, its frame count): the rendered windows' region meta,
    else the node's own lines; none before either."""
    if got["zoom"]["windows"]:
        return [(int(w["first_frame"]), int(w["frames"])) for w in got["zoom"]["windows"] if w.get("frames")]
    found = []
    for lines in ((queued or {}).get("said") or {}).values():
        for m in re.finditer(r"\[(\d+)\] window from ([\d.]+)s: (\d+) frames", "\n".join(lines)):
            found.append((round(float(m.group(2)) * 24), int(m.group(3))))
    return sorted(set(found))


def said_lines(got: dict) -> list[str]:
    """The reading as lines a person reads: one an input, then the text, the zoom and the failures."""
    lines = [f"{got['load']} ({got['stage']}): {got['state']}" + (f" at {got['queued']['queued_at']}" if got["queued"] else "")]
    for r in got["inputs"]:
        if r["exists"] is None:
            lines.append(f"  {r['role']}: node {r['node']}, {r['class']}: {r['note']}")
            continue
        if not r["exists"]:
            lines.append(f"  {r['role']}: node {r['node']} names {r['named']}: NOT THERE")
            continue
        how = r.get("fit", {}).get("words") or r.get("fit_not_given") or (f"{r['size'][0]}x{r['size'][1]}" if r.get("size") else "")
        at = r.get("loader") or {}
        span = (f"  from its frame {at['first_frame']}, {at['frame_load_cap']} frames" if "frame_load_cap" in at else
                f"  from {at.get('seek_seconds')} s for {at.get('duration')} s" if "seek_seconds" in at else "")
        lines.append(f"  {r['role']}: {r['file']}" + (f"  sha256 {r['sha256'][:16]}" if r.get("sha256") else "") + span + (f"; {how}" if how else ""))
        if r.get("trim"):
            lines.append("      trimmed: " + ", ".join(f"{k} {v}" for k, v in r["trim"].items()))
        if r.get("unread"):
            lines.append(f"      {r['unread']}")
        if r.get("rebuilt"):
            lines.append(f"      {r['rebuilt']}")
        if r.get("append"):
            lines.append(f"      appended as {r['tag']}: " + ", ".join(f"{k} {v}" for k, v in r["append"].items() if k != "node"))
        if r.get("record", {}).get("what"):
            lines.append(f"      its record: {r['record']['what']}; its frame 0 is source frame {r['record'].get('its_first_frame_is_source_frame')}")
        for line in r.get("node_said", []):
            lines.append(f"      the node: {line.strip()}")
        for k, e in enumerate(r.get("encoder_frames", []), 1):
            held = [x for x in e["sampled"] if x["held"]]
            lines.append(f"      window {k}: the text encoder is shown {len(e['sampled'])} of its frames: reference frames "
                         + ", ".join(str(x["reference_frame"]) + ("*" if x["held"] else "") for x in e["sampled"])
                         + (f"  (* past the reference's {r['reference_frames']} frames: its last frame repeated, {len(held)} of them)" if held else "")
                         + (f"; the node says {e['node_says']}" if "node_says" in e else "; the node's count not found"))
    z = got["zoom"]
    if z.get("piece"):
        p = z["piece"]
        lines.append(f"  zoom: {p['zoom']:.2f}x on the box {p['box']} of the {p['source_size'][0]}x{p['source_size'][1]} frames ({p.get('set_by')})")
    else:
        lines.append(f"  zoom: render_region {z['render_region']!r}" + ("; no zoomed record beside the render" if got.get("render") else ""))
    c = got.get("canvases") or {}
    if c.get("sizes"):
        lines.append("  canvas: every video loader hands on " + " and ".join(f"{a}x{b}" for a, b in c["sizes"])
                     + ("" if len(c["sizes"]) == 1 else ": NOT ONE SIZE") + "; cut by the loader: "
                     + (", ".join(f"{x['role']} (node {x['node']}, {x['crop'][2]}x{x['crop'][3]} at ({x['crop'][0]}, {x['crop'][1]}) of its "
                                  f"{x['of'][0]}x{x['of'][1]})" for x in c["cut"]) or "none")
                     + f"; {len(c['uncut'])} loaded uncut (whether those were drawn on the frames as the loader cuts the source is not in the graph)")
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
