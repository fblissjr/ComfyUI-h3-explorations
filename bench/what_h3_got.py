#!/usr/bin/env python3
"""What H3 was handed for one load, read from the graph that was queued: every file, its digest, what the graph does to it.

    H3_INPUT_DIR=... H3_OUTPUT_DIR=... <python> bench/what_h3_got.py --job-folder data/<day>_<load>_job --load <load> \\
        [--stage render|plan] [--capture data/<day>_<load>] [--json OUT.json]
    H3_INPUT_DIR=... H3_OUTPUT_DIR=... <python> bench/what_h3_got.py data/<day>_<load>_job/<load>.<stage>[.queued].json \\
        [--capture data/<day>_<load>] [--json OUT.json]

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
- When the graph's masks come from a capture folder (`data/<day>_<load>`, inferred from the mask paths or job file,
  or passed via `--capture`): a read-only **Capture & Input Provenance Join** (`capture_audit`) that opens the
  capture's `manifest.json`, `verify.json`, `flags.json`, `runs/<load>/region.npz`, `depth.npz`, and per-subject
  `head_tube__<by>.json` and `hair_in_hole__<by>.json`, cross-checks every graph input (`the source`, `mask`,
  `parts_held`, `others` / `strangers`, `<Video 1>`, `<Picture 1>`) against `manifest.json` and `verify.json`, and
  prints a Capture & Input Provenance Table with each file's producer tool, `sha256` (16-hex prefix), `writer_commit`
  and git age (`stale_at_build`, `code_updated_since_capture`, or `current`), agreement status, active-step `0 px`
  and speck frames, unresolved `lost_asked_on_source_frames` / `hair_undecided_on_source_frames`, and any overridden
  blocking flags in `flags.json`.

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
#: Found under ComfyUI's own custom_nodes (`_lib.COMFY` or the active ComfyUI venv's root when run from an external worktree).
_COMFY_ROOT = next(
    (c for c in (COMFY, Path(sys.prefix).resolve().parent, *(p / "ComfyUI" for p in REPO.parents)) if (c / "comfy").is_dir()),
    COMFY,
)
VHS_LOADER_FILE = _COMFY_ROOT / "custom_nodes" / "ComfyUI-VideoHelperSuite" / "videohelpersuite" / "load_video_nodes.py"
SONG, SOURCE = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource"
#: The Masked Source's linked inputs that carry a picture, and the name each is shown under. **Inherited** from the
#: node's own inputs (`video_mask.py`, `MiniMaxH3MaskedSource.define_schema`); one not listed is shown by its own name.
ROLES = {"frames": "the source", "mask": "the subject's mask", "parts": "the parts kept", "others": "the keep-outs",
         "motion_video": "<Video 1>", "motion_video_2": "<Video 2>", "keep": "what is kept the original (`keep`)",
         "encode_frames": "the conditioning frames (`encode_frames`)"}
#: The Masked Source's lazy inputs and which `replace` asks for each (`video_mask.py`, `check_lazy_status` and the
#: LAZY_FOR_* names). **Inherited**; a linked lazy input `replace` does not ask for is listed and said to be unread.
LAZY = {"mask": None, "shot_table": None, "segmenter": ("head and hair",), "segmenter_clip": ("head and hair",),
        "parts": ("the wired parts",)}


#: (resolved path, size, mtime_ns) -> sha256: a file is read once a run for its digest (the lead, 2026-10-16: a sidecar of the
#: whole video read the source once a piece, 229 s and 382 s on two runs of 45 pieces). Cannot see: a file written again by
#: this same process at the same size inside one timestamp tick keeps its old digest (sidecar.py's named gap 15).
_DIGESTS: dict = {}


def sha256(path: Path) -> str:
    """A file's sha256, read once a run (`_DIGESTS`); `bench/sidecar.py`'s digest is this one."""
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        _DIGESTS[key] = h.hexdigest()
    return _DIGESTS[key]


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


def carved_on_plan(job_folder: Path, load: str, path: str) -> bool:
    """Whether the on-plan step's record (`<load>.plan.masks.json`) was hashed after its rebuild (`load_chain.HASHED_AFTER_THE_REBUILD`,
    route C) and names this mask: then a plan's mask that differs from what the plan read is the grid carve, not a fault."""
    import load_chain
    record = Path(job_folder) / f"{load}.plan.masks.json"
    if not record.is_file():
        return False
    held = json.loads(record.read_text())
    return held.get("hashed") == load_chain.HASHED_AFTER_THE_REBUILD and str(path) in (held.get("masks") or {})


def read(job_folder: Path, load: str, stage: str = "render", *, inputs: Path | None = None, output: Path | None = None,
         digests: bool = True, capture: Path | None = None) -> dict:
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
    # the windows as the song node is asked for them: a window that keeps frames of the one before shows them at `context_noise`, the
    # node's own default when the graph names none (2026-10-15: a load of chained windows may name one)
    out["windows"] = {k: song.get(k) for k in ("window_frames", "context_frames", "context_noise") if not isinstance(song.get(k), list)}
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
            elif carved_on_plan(job_folder, load, row["file"]):
                # ROUTE C (the lead's ruling of 2026-10-16): the on-plan rebuild carves the keep-outs on the node's own grid, after the plan
                # ran, and remakes the plan's region on the carved files; its record was hashed after the rebuild
                row["rebuilt"] = (f"grid-carved after the plan, route C; the look was made on the predicted region: {got['how']} ({copy}, the "
                                  f"bytes recorded at queue time; the on-plan record {load}.plan.masks.json hashed it after the rebuild)")
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
    out["capture_audit"] = audit_capture(out, job_folder, load, stage, inputs, output, capture)
    return out


#: Producer code files in the repo for each preview / capture stage. **Inherited** from `bench/inspect_mask_steps.py`.
PRODUCER_FILES: dict[str, list[str]] = {
    "track": ["subject_track.py", "sam31_corrections.py", "shot_table.py"],
    "head_track": ["subject_track.py", "sam31_corrections.py"],
    "parts": ["sapiens2_parts.py", "head_tube_parts.py", "bench/head_tube_parts.py"],
    "classes": ["sapiens2_parts.py"],
    "pose": ["sapiens2_Pose.py", "bench/pose_along_shot.py"],
    "points": ["bench/pose_along_shot.py", "bench/head_reference.py", "bench/draw_face_points.py"],
    "strangers": ["bench/strangers.py", "sapiens2_parts.py"],
    "head_reference": ["bench/head_reference.py", "bench/draw_face_points.py"],
    "depth": ["bench/measure_moge2_depth.py"],
    "head_tube": ["bench/head_tube_parts.py", "head_tube_parts.py"],
    "hair_in_hole": ["bench/head_tube_parts.py", "head_tube_parts.py", "bench/capture_masked_run.py"],
    "capture": ["bench/capture_masked_run.py"],
}

#: H3 VideoVAE temporal token cadence (`comfy.ldm.minimax.model.FRAME_PER_TOKEN` and `video_mask.run_lengths`).
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _iso_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
        return d.timestamp()
    except ValueError:
        return None


def _git_commit_of(rel_paths: list[str], before_epoch: float | None = None) -> dict | None:
    existing = [p for p in rel_paths if (REPO / p).is_file()]
    if not existing:
        return None
    cmd = ["git", "log", "-n", "1", "--format=%h|%ct|%cs"]
    if before_epoch is not None:
        cmd.append(f"--before={int(before_epoch)}")
    cmd.extend(["--", *existing])
    try:
        out = subprocess.check_output(cmd, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out or "|" not in out:
        return None
    h, ct, cs = out.split("|", 2)
    return {"commit": h, "epoch": float(ct), "date": cs}


def _read_stamp(path: Path | None) -> dict | None:
    """Read `code_stamp` or `commit`/`writer_commit` from a `.json` or `.npz` artifact when present."""
    if path is None or not path.is_file():
        return None
    try:
        if path.suffix == ".json":
            d = json.loads(path.read_text())
            if isinstance(d, dict):
                if isinstance(d.get("code_stamp"), dict):
                    return d["code_stamp"]
                if d.get("writer_commit") or d.get("commit"):
                    return {"writer_commit": d.get("writer_commit") or d.get("commit")}
        elif path.suffix == ".npz":
            import numpy as np
            with np.load(path) as z:
                if "code_stamp" in z.files:
                    raw = z["code_stamp"]
                    s = str(raw.item() if getattr(raw, "ndim", 1) == 0 else raw)
                    return json.loads(s)
    except Exception:
        return None
    return None


def _producer_status(group: str, written_epoch: float | None, build_epoch: float, explicit_commit: str | None = None) -> tuple[str, str]:
    """Return `(writer_commit, code_status)` for a producer group given when the artifact was written and when the capture built."""
    files = PRODUCER_FILES.get(group, [])
    at_write = _git_commit_of(files, before_epoch=(written_epoch + 60) if written_epoch is not None else (build_epoch + 60))
    at_build = _git_commit_of(files, before_epoch=build_epoch + 60)
    at_head = _git_commit_of(files, before_epoch=None)
    writer_commit = explicit_commit or (at_write["commit"] if at_write else "unknown")
    if written_epoch is not None and at_build and at_build["epoch"] > written_epoch + 120 and at_build["commit"] != writer_commit:
        return writer_commit, f"stale_at_build ({files[0]} @ {at_build['commit']} {at_build['date']} > {writer_commit})"
    if at_head and at_head["epoch"] > max(written_epoch or 0.0, build_epoch) + 120 and at_head["commit"] != writer_commit:
        return writer_commit, f"code_updated_since_capture ({files[0]} @ {at_head['commit']} {at_head['date']} > {writer_commit})"
    return writer_commit, "current"


def _find_capture_dir(got: dict, job_folder: Path, load: str, stage: str, capture: Path | None) -> Path | None:
    if capture is not None:
        cap = Path(capture)
        if not (cap / "manifest.json").is_file():
            raise SystemExit(f"no manifest.json in {cap}: --capture must name a capture folder")
        return cap
    for row in got.get("inputs", []):
        if row.get("role") in ("the subject's mask", "the parts kept", "the keep-outs") and row.get("file"):
            p = Path(row["file"])
            for parent in p.parents:
                if (parent / "manifest.json").is_file():
                    return parent
                if parent.name.endswith("_first"):
                    sib = parent.with_name(parent.name[:-6])
                    if (sib / "manifest.json").is_file():
                        return sib
    for jf in (job_folder / f"job.{stage}.json", job_folder / "job.json"):
        if jf.is_file():
            try:
                jd = json.loads(jf.read_text())
                c_str = (jd.get("job") or {}).get("capture") or jd.get("capture")
                if c_str:
                    c_p = Path(c_str) if Path(c_str).is_absolute() else (REPO / c_str)
                    if (c_p / "manifest.json").is_file():
                        return c_p
            except Exception:
                pass
    return None


def _spans_text(spans: list) -> str:
    if not spans:
        return "none"
    parts = []
    for sp in spans:
        if isinstance(sp, (list, tuple)) and len(sp) == 2:
            parts.append(f"{sp[0]}-{sp[1]}" if sp[0] != sp[1] else str(sp[0]))
        else:
            parts.append(str(sp))
    return ", ".join(parts)


def audit_capture(got: dict, job_folder: Path, load: str, stage: str, inputs_dir: Path,
                  output_dir: Path | None, capture: Path | None = None) -> dict | None:
    """Read-only join of the queued graph's inputs against the capture folder's `manifest.json`, `verify.json`, `flags.json`,
    `runs/<load>/region.npz`, `depth.npz`, and per-subject `head_tube__<by>.json` and `hair_in_hole__<by>.json`."""
    cap_dir = _find_capture_dir(got, job_folder, load, stage, capture)
    if cap_dir is None:
        return None
    manifest_path = cap_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_path = cap_dir / "verify.json"
    verify = json.loads(verify_path.read_text()) if verify_path.is_file() else None
    flags_path = cap_dir / "flags.json"
    flags = json.loads(flags_path.read_text()) if flags_path.is_file() else None

    first = int(manifest.get("first_frame", 0))
    n = int(manifest.get("frames", 0))
    size = manifest.get("size") or [1024, 768]
    w_px = int(size[0])
    runs = manifest.get("runs") or []
    run_entry = next((r for r in runs if r.get("name") == load), runs[0] if runs else {})
    subj = run_entry.get("subject", "")
    subj_entry = next((s for s in (manifest.get("subjects") or []) if s.get("label") == subj), {})
    seen = (subj_entry.get("sightings") or [{}])[0]
    by = seen.get("by", "")

    # Capture build epoch and writer commit
    cap_mtime = manifest_path.stat().st_mtime
    in_epochs = [_iso_epoch(v.get("written")) for v in (seen.get("inputs") or {}).values() if isinstance(v, dict)]
    in_epochs = [e for e in in_epochs if e is not None]
    build_epoch = max(in_epochs + [cap_mtime]) if in_epochs else cap_mtime
    man_stamp = _read_stamp(manifest_path)
    queued_commit = (got.get("queued") or {}).get("commit")
    cap_commit, cap_code_status = _producer_status(
        "capture", build_epoch, build_epoch,
        explicit_commit=(man_stamp or {}).get("writer_commit") or queued_commit,
    )

    # Preview job commit & queued_at when available under data/
    data_root = cap_dir.resolve().parent
    preview_commit: str | None = None
    preview_queued_at: str | None = None
    if by:
        for pq in sorted(data_root.glob(f"*_{by}_job/preview.queued.json")):
            try:
                pq_d = json.loads(pq.read_text())
                preview_commit = pq_d.get("commit") or preview_commit
                preview_queued_at = pq_d.get("completed_at") or pq_d.get("queued_at") or preview_queued_at
            except Exception:
                pass

    # Output share root for resolving preview .npz / strangers files
    if output_dir is None:
        import os
        out_env = os.environ.get("H3_OUTPUT_DIR")
        output_dir = Path(out_env) if out_env else (REPO / "output")
    clip_name = Path(manifest.get("clip") or "").name
    video_proj = clip_name.split("_")[0] if "_" in clip_name else "fun"
    vid_out = output_dir / "Video" / video_proj if (output_dir / "Video" / video_proj).is_dir() else (output_dir / "Video")

    # Cross-check graph inputs against manifest.json & verify.json
    mismatches: list[str] = []
    table: list[dict] = []

    by_role: dict[str, list[dict]] = {}
    for r in got.get("inputs", []):
        by_role.setdefault(r["role"], []).append(r)

    # 1. Source video
    src_rows = by_role.get("the source", [])
    if src_rows:
        sr = src_rows[0]
        g_clip = Path(sr.get("named") or sr.get("file") or "").name
        g_first = (sr.get("loader") or {}).get("first_frame")
        g_cap = (sr.get("loader") or {}).get("frame_load_cap")
        win_f = (got.get("windows") or {}).get("window_frames")
        src_ok = (g_clip == clip_name and g_first == first and (g_cap == n or (win_f and g_cap == win_f and n <= g_cap)))
        if not src_ok:
            mismatches.append(
                f"the source: graph loads {g_clip!r} (first_frame={g_first}, frames={g_cap}) != "
                f"manifest clip={clip_name!r} (first_frame={first}, frames={n})"
            )
        cap_note = f"{n}f (loader cap {g_cap}f)" if g_cap != n else f"{n}f"
        table.append({
            "role": "the source",
            "producer": "source plate",
            "path": sr.get("file") or sr.get("named"),
            "sha256": (sr.get("sha256") or "")[:16] or None,
            "writer_commit": "-",
            "code_status": "source",
            "agreement": f"AGREES ({clip_name}, {first}..{first + n - 1}, {cap_note})" if src_ok else f"MISMATCH ({g_clip} {g_first}+{g_cap}f vs {clip_name} {first}+{n}f)",
        })

    # 2. Subject mask (`track__<by>.mkv`)
    subj_dir = cap_dir / "subjects" / subj if subj else cap_dir
    want_mask = subj_dir / f"track__{by}.mkv" if by else None
    for mr in by_role.get("the subject's mask", []):
        mp = Path(mr["file"]) if mr.get("file") else None
        mask_ok = bool(mp and want_mask and (mp.resolve() == want_mask.resolve() or (first_copy(want_mask) and mp.resolve() == first_copy(want_mask).resolve())))
        if not mask_ok and want_mask:
            mismatches.append(f"the subject's mask: graph loads {mr.get('file')!r} != capture {str(want_mask)!r}")
        table.append({
            "role": f"the subject's mask ({subj})",
            "producer": "bench/capture_masked_run.py",
            "path": mr.get("file"),
            "sha256": (mr.get("sha256") or "")[:16] or None,
            "writer_commit": cap_commit,
            "code_status": cap_code_status,
            "agreement": f"AGREES (subjects/{subj}/track__{by}.mkv)" if mask_ok else f"MISMATCH (expected subjects/{subj}/track__{by}.mkv)",
        })

    # 3. Parts kept (`parts_held__<by>.mkv`)
    want_parts = subj_dir / f"parts_held__{by}.mkv" if (by and seen.get("parts")) else None
    for pr in by_role.get("the parts kept", []):
        if pr.get("file") is None:
            continue
        pp = Path(pr["file"])
        parts_ok = bool(want_parts and (pp.resolve() == want_parts.resolve() or (first_copy(want_parts) and pp.resolve() == first_copy(want_parts).resolve())))
        if not parts_ok and want_parts:
            mismatches.append(f"the parts kept: graph loads {pr.get('file')!r} != capture {str(want_parts)!r}")
        st_stamp = _read_stamp(subj_dir / f"head_tube__{by}.json") or man_stamp
        p_commit, p_status = _producer_status(
            "hair_in_hole", build_epoch, build_epoch,
            explicit_commit=(st_stamp or {}).get("writer_commit") or cap_commit,
        )
        table.append({
            "role": f"the parts kept ({subj})",
            "producer": "bench/capture_masked_run.py + head_tube_parts.py",
            "path": pr.get("file"),
            "sha256": (pr.get("sha256") or "")[:16] or None,
            "writer_commit": p_commit,
            "code_status": p_status,
            "agreement": f"AGREES (subjects/{subj}/parts_held__{by}.mkv)" if parts_ok else f"MISMATCH (expected {want_parts})",
        })

    # 4. Keep-outs (`others` / `strangers`)
    expected_others: dict[str, Path] = {}
    for o_lbl in (run_entry.get("others") or []):
        o_entry = next((s for s in (manifest.get("subjects") or []) if s.get("label") == o_lbl), None)
        o_by = ((o_entry or {}).get("sightings") or [{}])[0].get("by", by)
        expected_others[o_lbl] = cap_dir / "subjects" / o_lbl / f"track__{o_by}.mkv"
    seen_other_labels: set[str] = set()
    for kr in by_role.get("the keep-outs", []):
        kp = Path(kr["file"]) if kr.get("file") else None
        matched_lbl = None
        for o_lbl, o_path in expected_others.items():
            if kp and (kp.resolve() == o_path.resolve() or (first_copy(o_path) and kp.resolve() == first_copy(o_path).resolve())):
                matched_lbl = o_lbl
                seen_other_labels.add(o_lbl)
                break
        if matched_lbl is None:
            mismatches.append(
                f"the keep-outs: graph loads {kr.get('file')!r}, which is not in manifest runs[].others "
                f"({list(expected_others.keys())})"
            )
        table.append({
            "role": f"the keep-outs ({matched_lbl or 'UNMATCHED'})",
            "producer": "bench/capture_masked_run.py",
            "path": kr.get("file"),
            "sha256": (kr.get("sha256") or "")[:16] or None,
            "writer_commit": cap_commit,
            "code_status": cap_code_status,
            "agreement": f"AGREES (subjects/{matched_lbl}/{kp.name})" if matched_lbl and kp else "MISMATCH (not in manifest runs[].others)",
        })
    for o_lbl, o_path in expected_others.items():
        if o_lbl not in seen_other_labels:
            mismatches.append(f"the keep-outs: manifest runs[].others names {o_lbl!r} ({o_path}), but the graph does not wire it")
            table.append({
                "role": f"the keep-outs ({o_lbl})",
                "producer": "bench/capture_masked_run.py",
                "path": str(o_path),
                "sha256": sha256(o_path)[:16] if o_path.is_file() else None,
                "writer_commit": cap_commit,
                "code_status": cap_code_status,
                "agreement": "MISSING FROM GRAPH (named in manifest runs[].others)",
            })

    # 5. <Video 1> (head reference video) and <Picture 1> (face reference still)
    for vr in by_role.get("<Video 1>", []):
        rec = vr.get("record") or {}
        rec_first = rec.get("its_first_frame_is_source_frame")
        rec_frames = rec.get("frames")
        frames_ok = (
            rec_frames in (None, n)
            or (isinstance(rec_frames, list) and len(rec_frames) == 2 and rec_frames == [first, first + n - 1])
        )
        v_ok = (rec_first in (None, first)) and frames_ok
        if not v_ok:
            mismatches.append(f"<Video 1>: record first_frame={rec_first}, frames={rec_frames} != manifest ({first}, {n})")
        v_ep = _iso_epoch(rec.get("made_at") or vr.get("written"))
        v_stamp = _read_stamp(Path(rec["file"])) if rec.get("file") else None
        v_commit, v_status = _producer_status(
            "head_reference", v_ep, build_epoch,
            explicit_commit=(v_stamp or {}).get("writer_commit") or (cap_commit if v_ep and abs(v_ep - build_epoch) < 7200 else None),
        )
        f_desc = f"{rec_frames[0]}..{rec_frames[1]} ({n}f)" if (isinstance(rec_frames, list) and len(rec_frames) == 2) else f"{rec_frames}f"
        table.append({
            "role": "<Video 1> (head ref)",
            "producer": rec.get("made_by") or "bench/head_reference.py",
            "path": vr.get("file"),
            "sha256": (vr.get("sha256") or "")[:16] or None,
            "writer_commit": v_commit,
            "code_status": v_status,
            "agreement": f"AGREES (frame 0 = {rec_first}, {f_desc})" if v_ok else f"MISMATCH (frame 0 = {rec_first}, {rec_frames})",
        })
    for pr in by_role.get("<Picture 1>", []):
        table.append({
            "role": "<Picture 1> (still)",
            "producer": "face reference crop",
            "path": pr.get("file"),
            "sha256": (pr.get("sha256") or "")[:16] or None,
            "writer_commit": "-",
            "code_status": "still",
            "agreement": "AGREES" if pr.get("exists") else "MISSING",
        })

    # 6. Upstream preview inputs recorded in `manifest.json` (`seen["inputs"]` and `strangers`)
    inputs_meta = seen.get("inputs") or {}
    for k_in in ("track", "head_track", "parts", "classes", "pose", "points"):
        v_in = inputs_meta.get(k_in)
        if not isinstance(v_in, dict):
            continue
        f_name = str(v_in.get("file") or "")
        w_str = v_in.get("written")
        w_ep = _iso_epoch(w_str)
        disk_p: Path | None = None
        if f_name:
            if Path(f_name).is_file():
                disk_p = Path(f_name)
            elif vid_out.is_dir():
                cands = list(vid_out.glob(f"*/{by}/{f_name}")) + list(vid_out.glob(f"*/{f_name}"))
                if cands:
                    disk_p = cands[0]
        in_stamp = (v_in.get("code_stamp") if isinstance(v_in.get("code_stamp"), dict) else None) or _read_stamp(disk_p)
        exp_c = (in_stamp or {}).get("writer_commit") or (preview_commit if k_in != "points" else None)
        c_hash, c_status = _producer_status(k_in, w_ep, build_epoch, explicit_commit=exp_c)
        prod_label = (PRODUCER_FILES.get(k_in) or [k_in])[0]
        table.append({
            "role": f"preview:{k_in} ({by})",
            "producer": prod_label,
            "path": str(disk_p) if disk_p else f_name,
            "sha256": sha256(disk_p)[:16] if (disk_p and disk_p.is_file()) else None,
            "writer_commit": c_hash,
            "code_status": c_status,
            "agreement": f"written {w_str}" if w_str else "recorded in manifest",
        })

    str_entry = next((s for s in (manifest.get("subjects") or []) if s.get("label") == "strangers"), None)
    if str_entry and str_entry.get("sightings"):
        s_seen = str_entry["sightings"][0]
        s_by = s_seen.get("by", by)
        s_in = ((s_seen.get("inputs") or {}).get("track")) or {}
        sf_name = str(s_in.get("file") or s_seen.get("track") or "")
        sw_str = s_in.get("written")
        sw_ep = _iso_epoch(sw_str)
        s_disk: Path | None = None
        if sf_name and vid_out.is_dir():
            cands = list(vid_out.glob(f"keep_outs/{s_by}/{sf_name}")) + list(vid_out.glob(f"keep_outs/{sf_name}"))
            if cands:
                s_disk = cands[0]
        s_stamp = (s_in.get("code_stamp") if isinstance(s_in.get("code_stamp"), dict) else None) or (
            _read_stamp(s_disk.with_suffix(".json")) if s_disk else None
        )
        sc_hash, sc_status = _producer_status("strangers", sw_ep, build_epoch, explicit_commit=(s_stamp or {}).get("writer_commit"))
        table.append({
            "role": f"raw keep-out:strangers ({s_by})",
            "producer": "bench/strangers.py",
            "path": str(s_disk) if s_disk else sf_name,
            "sha256": sha256(s_disk)[:16] if (s_disk and s_disk.is_file()) else None,
            "writer_commit": sc_hash,
            "code_status": sc_status,
            "agreement": f"written {sw_str}; carved {((s_seen.get('carved_out_of_own_track') or {}).get('px', 0))} px" if sw_str else "recorded in manifest",
        })

    # 7. Capture artifacts: depth.npz, head_tube__<by>.json, hair_in_hole__<by>.json, runs/<load>/region.npz
    owners = manifest.get("owners") or {}
    contested_px = int(owners.get("contested_px", 0) or 0)
    depth_npz = cap_dir / "depth.npz"
    if depth_npz.is_file() or owners.get("depth") or contested_px > 0:
        d_ep = depth_npz.stat().st_mtime if depth_npz.is_file() else build_epoch
        d_stamp = _read_stamp(depth_npz)
        dc_hash, dc_status = _producer_status("depth", d_ep, build_epoch, explicit_commit=(d_stamp or {}).get("writer_commit"))
        table.append({
            "role": "capture:depth.npz",
            "producer": "bench/measure_moge2_depth.py",
            "path": str(depth_npz) if depth_npz.is_file() else "(not present)",
            "sha256": sha256(depth_npz)[:16] if depth_npz.is_file() else None,
            "writer_commit": dc_hash if depth_npz.is_file() else "-",
            "code_status": dc_status if depth_npz.is_file() else "missing",
            "agreement": f"present (contested_px={contested_px})" if depth_npz.is_file() else f"NOT RUN (contested_px={contested_px})",
        })

    ht_json_path = subj_dir / f"head_tube__{by}.json" if by else None
    ht_npz_path = subj_dir / f"head_tube__{by}.npz" if by else None
    head_tube_info: dict | None = None
    if ht_json_path and ht_json_path.is_file():
        ht_d = json.loads(ht_json_path.read_text())
        ht_ep = ht_json_path.stat().st_mtime
        ht_stamp = _read_stamp(ht_json_path) or _read_stamp(ht_npz_path)
        ht_commit, ht_status = _producer_status("head_tube", ht_ep, build_epoch, explicit_commit=(ht_stamp or {}).get("writer_commit") or cap_commit)
        bridged = ht_d.get("bridged_gaps") or []
        held_ends = ht_d.get("held_ends") or []
        lost_iv = ht_d.get("lost_in_view") or []
        head_tube_info = {
            "file": str(ht_json_path),
            "sha256": sha256(ht_json_path)[:16],
            "bridged_gaps": bridged,
            "held_ends": held_ends,
            "lost_in_view": lost_iv,
        }
        table.append({
            "role": f"capture:head_tube__{by}",
            "producer": "bench/head_tube_parts.py::head_tube",
            "path": str(ht_npz_path if (ht_npz_path and ht_npz_path.is_file()) else ht_json_path),
            "sha256": sha256(ht_npz_path if (ht_npz_path and ht_npz_path.is_file()) else ht_json_path)[:16],
            "writer_commit": ht_commit,
            "code_status": ht_status,
            "agreement": f"bridged={len(bridged)}, held_ends={len(held_ends)}, lost_in_view={len(lost_iv)}",
        })

    hih_json_path = subj_dir / f"hair_in_hole__{by}.json" if by else None
    hair_in_hole_info: dict | None = None
    if hih_json_path and hih_json_path.is_file():
        hih_d = json.loads(hih_json_path.read_text())
        hih_ep = hih_json_path.stat().st_mtime
        hih_stamp = _read_stamp(hih_json_path)
        hih_commit, hih_status = _producer_status("hair_in_hole", hih_ep, build_epoch, explicit_commit=(hih_stamp or {}).get("writer_commit") or cap_commit)
        asked_on = hih_d.get("lost_asked_on_source_frames") or []
        undecided_on = hih_d.get("hair_undecided_on_source_frames") or []
        marks = hih_d.get("marks") or {}
        hair_in_hole_info = {
            "file": str(hih_json_path),
            "sha256": sha256(hih_json_path)[:16],
            "lost_asked_on_source_frames": asked_on,
            "hair_undecided_on_source_frames": undecided_on,
            "marks": marks,
        }
        table.append({
            "role": f"capture:hair_in_hole__{by}",
            "producer": "bench/head_tube_parts.py::her_hair_in_hole",
            "path": str(hih_json_path),
            "sha256": sha256(hih_json_path)[:16],
            "writer_commit": hih_commit,
            "code_status": hih_status,
            "agreement": f"asked_on={_spans_text(asked_on)}; hair_undecided={_spans_text(undecided_on)}",
        })

    # 8. Active-step 0 px and speck frames from `runs/<load>/region.npz` + `manifest.json`
    reg_npz_path = cap_dir / "runs" / (run_entry.get("name") or load) / "region.npz"
    zero_px_frames: list[int] = []
    for sp in (run_entry.get("left_as_the_source_with_no_mask") or []):
        if isinstance(sp, (list, tuple)) and len(sp) == 2:
            for sf in range(int(sp[0]), int(sp[1]) + 1):
                if sf not in zero_px_frames:
                    zero_px_frames.append(sf)
    speck_frames: list[dict] = []
    part_floor_px = int((seen.get("part_sizes") or {}).get("floor_px", 64))
    if reg_npz_path.is_file() and n > 0:
        import numpy as np
        with np.load(reg_npz_path) as rz:
            r_arr = (rz["region"] > 0).astype(bool)
            reg_cells = r_arr.reshape(len(r_arr), -1).sum(axis=-1)
            c_raw = rz["carried"]
            if c_raw.dtype == np.uint8 and c_raw.ndim == 3 and c_raw.shape[-1] == (w_px + 7) // 8:
                c_bool = np.unpackbits(c_raw, axis=-1)[..., :w_px].astype(bool)
            else:
                c_bool = (c_raw > 0).astype(bool)
            carried_px = c_bool.reshape(len(c_bool), -1).sum(axis=-1)
        # Partition the load's frames into (1, 4, 4, 4, 4) latent steps per window
        win_cfg = got.get("windows") or {}
        wf_cap = int(win_cfg.get("window_frames") or max(n, 1))
        ctx_f = int(win_cfg.get("context_frames") or 0)
        step_groups: list[list[int]] = []
        w_start = 0
        while w_start < n:
            w_end = min(n, w_start + wf_cap)
            pos = w_start
            s_idx = 0
            while pos < w_end:
                step_len = FRAME_PER_TOKEN[s_idx % len(FRAME_PER_TOKEN)]
                step_groups.append(list(range(pos, min(w_end, pos + step_len))))
                pos += step_len
                s_idx += 1
            if w_end >= n:
                break
            stride = max(1, wf_cap - ctx_f)
            w_start += stride
        idx_to_peers: dict[int, list[int]] = {}
        for grp in step_groups:
            for idx in grp:
                idx_to_peers[idx] = grp
        for i in range(min(n, len(carried_px))):
            sf = first + i
            peers = idx_to_peers.get(i, [i])
            step_active = any(int(reg_cells[p]) > 0 for p in peers if p < len(reg_cells))
            step_max_c = max((int(carried_px[p]) for p in peers if p < len(carried_px)), default=0)
            c_px = int(carried_px[i])
            if step_active and c_px == 0 and sf not in zero_px_frames:
                zero_px_frames.append(sf)
            elif step_active and 0 < c_px < 0.25 * step_max_c and c_px < max(64, part_floor_px):
                speck_frames.append({
                    "source_frame": sf,
                    "carried_px": c_px,
                    "step_max_carried_px": step_max_c,
                    "floor_px": part_floor_px,
                })
        zero_px_frames.sort()
        r_stamp = _read_stamp(reg_npz_path) or man_stamp
        r_commit, r_status = _producer_status("capture", reg_npz_path.stat().st_mtime, build_epoch, explicit_commit=(r_stamp or {}).get("writer_commit") or cap_commit)
        speck_desc = ", ".join(f"{x['source_frame']} ({x['carried_px']}px/{x['step_max_carried_px']}px)" for x in speck_frames) or "none"
        table.append({
            "role": f"capture:runs/{run_entry.get('name') or load}/region.npz",
            "producer": "bench/capture_masked_run.py",
            "path": str(reg_npz_path),
            "sha256": sha256(reg_npz_path)[:16],
            "writer_commit": r_commit,
            "code_status": r_status,
            "agreement": f"active-step 0px={zero_px_frames or 'none'}; speck={speck_desc}",
        })

    # 9. Verify & Flags summary
    verify_summary: dict | None = None
    if verify is not None:
        v_subj = ((verify.get("subjects") or {}).get(subj)) or {}
        v_runs = verify.get("runs") or []
        v_run = next((r for r in v_runs if r.get("name") == load), v_runs[0] if v_runs else {})
        no_mask_stmts = (
            verify.get("no_mask_statements")
            or (v_run.get("no_mask_statements") if isinstance(v_run, dict) else None)
            or v_subj.get("no_mask_statements")
            or []
        )
        v_failed = verify.get("failed")
        verify_summary = {
            "verdict": verify.get("verdict"),
            "failed": v_failed,
            "no_mask_statements": no_mask_stmts,
        }
        if v_failed not in (0, [], None) or verify.get("verdict") not in ("reads what the nodes wrote", "PASS"):
            mismatches.append(f"verify.json verdict is {verify.get('verdict')!r} (failed={v_failed!r})")

    flags_summary: dict | None = None
    if flags is not None:
        all_flags = flags.get("flags") or []
        ov_ids = set(flags.get("overridden") or []) if isinstance(flags.get("overridden"), list) else set()
        blk_ids = set(flags.get("blocking") or []) if isinstance(flags.get("blocking"), list) else set()
        overridden = []
        unresolved = []
        for f in all_flags:
            fid = f.get("id")
            fkind = f.get("rule") or f.get("kind")
            ov_info = f.get("override") if isinstance(f.get("override"), dict) else None
            is_ov = bool(ov_info or f.get("overridden") or (fid and fid in ov_ids))
            f_spans = f.get("source_frames") if f.get("source_frames") is not None else f.get("frames")
            if is_ov:
                overridden.append({
                    "id": fid,
                    "kind": fkind,
                    "subject": f.get("subject"),
                    "source_frames": f_spans,
                    "override_by": (ov_info or {}).get("by"),
                    "override_reason": (ov_info or {}).get("why") or (ov_info or {}).get("reason") or f.get("override_reason"),
                })
            elif f.get("blocking") or (fid and fid in blk_ids):
                unresolved.append({
                    "id": fid,
                    "kind": fkind,
                    "subject": f.get("subject"),
                    "source_frames": f_spans,
                })
        b_raw = flags.get("blocking", flags.get("blocking_count", len(unresolved)))
        flags_summary = {
            "blocking_count": len(b_raw) if isinstance(b_raw, list) else (b_raw if isinstance(b_raw, int) else len(unresolved)),
            "overridden_count": len(overridden),
            "overridden": overridden,
            "unresolved_blocking": unresolved,
        }

    return {
        "capture": str(cap_dir),
        "subject": subj,
        "preview": by,
        "preview_commit": preview_commit,
        "preview_queued_at": preview_queued_at,
        "capture_commit": cap_commit,
        "first_frame": first,
        "frames": n,
        "agrees": len(mismatches) == 0,
        "mismatches": mismatches,
        "active_step_zero_px_frames": zero_px_frames,
        "active_step_speck_frames": speck_frames,
        "contested_px": contested_px,
        "depth_used": bool(depth_npz.is_file() or owners.get("depth")),
        "head_tube": head_tube_info,
        "hair_in_hole": hair_in_hole_info,
        "verify": verify_summary,
        "flags": flags_summary,
        "table": table,
    }


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
    """The reading as lines a person reads: one an input, then the text, the zoom, the capture provenance table, and the failures."""
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
            nh = ((r["record"].get("drawn") or {}).get("no_head_on")) or {}
            nh_sp = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in nh.get("frames") or [])
            nh_txt = f"; no_head_on ({nh.get('form', 'hold')}) on {nh_sp}" if nh_sp else ""
            lines.append(f"      its record: {r['record']['what']}; its frame 0 is source frame {r['record'].get('its_first_frame_is_source_frame')}{nh_txt}")
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
    w = got.get("windows") or {}
    if w:
        lines.append(f"  windows: {w.get('window_frames')} frames each, a later one keeping {w.get('context_frames')} of the one before"
                     + (f", shown at noise {w['context_noise']:g}" if w.get("context_noise") is not None else ", shown at the song node's own noise (none named)"))
    lines.append(f"  text: {len(got['text']['text'])} characters, sha256 {got['text']['sha256'][:16]}, from node {got['text']['node']}")
    ca = got.get("capture_audit")
    if ca:
        ver = (ca.get("verify") or {}).get("verdict") or "none"
        agree_str = "AGREES" if ca["agrees"] else f"MISMATCH ({len(ca['mismatches'])})"
        lines.append(
            f"  capture audit: {ca['capture']} (subject={ca['subject']}, preview={ca['preview']}"
            + (f" @ {ca['preview_commit']}" if ca.get("preview_commit") else "")
            + f", capture @ {ca.get('capture_commit')}, verify={ver}, graph<->manifest={agree_str})"
        )
        lines.append("    Capture & Input Provenance Table:")
        lines.append(f"      {'Role / Stage':<34} | {'Producer':<36} | {'sha256':<16} | {'Writer @ Status':<42} | Agreement / Notes")
        lines.append("      " + "-" * 165)
        for row in ca.get("table", []):
            sh = row.get("sha256") or "-"
            wc = f"{row.get('writer_commit') or '-'} ({row.get('code_status') or '-'})"
            p_short = Path(str(row.get("path") or "")).name if row.get("path") else "-"
            lines.append(
                f"      {str(row.get('role') or ''):<34} | {str(row.get('producer') or ''):<36} | {sh:<16} | {wc:<42} | {p_short}: {row.get('agreement') or ''}"
            )
        z_f = ca.get("active_step_zero_px_frames") or []
        sp_f = ca.get("active_step_speck_frames") or []
        sp_txt = ", ".join(f"{x['source_frame']} ({x['carried_px']}px/{x['step_max_carried_px']}px)" for x in sp_f) or "none"
        lines.append(
            f"    active-step mask audit: 0 px frames={z_f or 'none'}; speck frames={sp_txt}; "
            f"contested_px={ca.get('contested_px', 0)} (depth_used={ca.get('depth_used')})"
        )
        hih = ca.get("hair_in_hole")
        if hih:
            m_counts = {k: len(v) for k, v in (hih.get("marks") or {}).items() if v}
            lines.append(
                f"    hair_in_hole ({Path(hih['file']).name}): asked_on={_spans_text(hih.get('lost_asked_on_source_frames') or [])}; "
                f"hair_undecided={_spans_text(hih.get('hair_undecided_on_source_frames') or [])}; marks={m_counts or 'none'}"
            )
        fl = ca.get("flags")
        if fl:
            ov = fl.get("overridden") or []
            ov_txt = "; ".join(f"{x['id']} {x['kind']} on {_spans_text(x.get('source_frames') or [])}" for x in ov) or "none"
            lines.append(
                f"    flags.json: blocking={fl.get('blocking_count', 0)}, overridden={fl.get('overridden_count', 0)} ({ov_txt})"
            )
        v_info = ca.get("verify") or {}
        if v_info.get("no_mask_statements"):
            nms = ", ".join(
                f"{x.get('source_frames') or x.get('frames')} ({x.get('by') or '-'}): {x.get('why') or x.get('reason')}"
                for x in v_info["no_mask_statements"]
            )
            lines.append(f"    verify no_mask_statements: {nms}")
        for m in ca.get("mismatches") or []:
            lines.append(f"    CAPTURE MISMATCH: {m}")
    lines += [f"FAILED: {x}" for x in got["failures"]]
    return lines


def _parse_graph_arg(target: str, load: str | None, stage: str) -> tuple[Path, str, str]:
    """Resolve either a `--job-folder <dir> --load <load>` pair or a direct path to `<load>.<stage>[.queued].json`."""
    p = Path(target)
    if p.suffix == ".json" and not p.is_dir():
        name = p.name
        if name.endswith(".queued.json"):
            name = name[: -len(".queued.json")]
        elif name.endswith(".json"):
            name = name[: -len(".json")]
        stem, _, st = name.rpartition(".")
        if stem and st in ("render", "plan"):
            return p.parent, (load or stem), st
        raise SystemExit(f"cannot infer load and stage from {p}: expected <load>.<render|plan>[.queued].json")
    if not load:
        raise SystemExit("--load is required when --job-folder names a directory")
    return p, load, stage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("graph_or_queued", nargs="?", help="optional path to data/<day>_<load>_job/<load>.<stage>[.queued].json")
    p.add_argument("--job-folder", help="data/<day>_<load>_job (or path to <load>.<stage>[.queued].json)")
    p.add_argument("--load", help="load name (inferred when a <load>.<stage>[.queued].json path is given)")
    p.add_argument("--stage", default="render", choices=("render", "plan"))
    p.add_argument("--capture", help="optional capture folder data/<day>_<load> (inferred from mask paths when omitted)")
    p.add_argument("--json", help="write the whole reading here")
    a = p.parse_args()
    target = a.job_folder or a.graph_or_queued
    if not target:
        p.error("either a path to <load>.<stage>[.queued].json or --job-folder is required")
    job_folder, load, stage = _parse_graph_arg(target, a.load, a.stage)
    got = read(job_folder, load, stage, capture=Path(a.capture) if a.capture else None)
    print("\n".join(said_lines(got)))
    if a.json:
        Path(a.json).write_text(json.dumps(got, indent=1) + "\n")
        print("wrote", a.json)
    sys.exit(1 if got["failures"] else 0)


if __name__ == "__main__":
    main()
