#!/usr/bin/env python3
"""Lay a saved window of a masked render again, with the song node's own composite, and say what a change moves.

    <python> bench/recomposite_window.py --window R_windows/R_window_2.safetensors --source SRC24.mov --source-first 14 \\
        --set cuts=none --out data/<date>_<name>
    <python> bench/recomposite_window.py --window R_windows/R_window_2.safetensors --source SRC24.mov --source-first 14 \\
        --capture data/<capture> --run R --shots R_00001_shots.json --context 90 --as-rendered cuts=none --out data/<date>_<name>

**What it buys.** A change to the masked lane's composite (what the render is laid with over its source) can be
tried on a render that already exists, with no sampling: a window's stored latent is decoded once, and
`video_mask.lay_window`, the function the song node itself calls, lays it twice, as it was rendered and with one
setting changed. What comes out is a row a frame: how much of the frame each laying keeps of the render, how many
pixels differ between the two, whether the changed one is the source bit for bit, and how far the first is from
the window's own video file (the proof that this is the render and not a look-alike). First use, 2026-10-10: the
frames a latent step lays across a cut (`video_mask.cut_gate`).

**What a window needs.** Three things, and the tool says which it has:
- the stored latent, `<name>_window_N.safetensors`: the sampled latent, before the decode and the composite;
- the window's region, `<name>_window_N_region.npz` beside it (`video_mask.save_window_region`): the fitted mask,
  the token region and the settings the composite ran with. Renders from before 2026-10-10 have none. For those
  the region is REBUILT from a capture of the run (`--capture`, `--run`): the mask is the one read back from the
  render's own review, a tinted and compressed picture, the token region is made from it by the node's own
  `token_mask`, and the count of cells where that differs from the region the review shows is printed. The
  settings come from the render's graph in the capture, the cuts from its shot table (`--shots`), and the
  window's first frame from its stored `next_start` and `--context`. A rebuilt region is refused for a run whose
  graph wires `keep` or `others`: those take tokens out that no mask video here holds;
- the source's own frames, read from the clip by number and fitted as the loader fits them (`_lib/frames.py`).
  A load that was capped inside the window (one load per shot) needs `--source-frames`, the cap: the node holds
  the last frame from there, and so must this.

**A window rendered with `context_noise` above 0** (the song node, 2026-10-10) holds on its context's latent steps
a blend of what was copied from the window before and what was sampled: decoded here, those frames are a redrawn
context, not the previous window's frames. They are trimmed from the window's video, so the rows marked written
are unaffected.

**The decode is a model.** It is the video VAE: no sampling, but card time, or minutes on the CPU. `--device cpu`
(the default) opens no CUDA context beside a render; `--device cuda` is for a gap in the queue. The decode is kept
under `--out` as `decoded_<window's name>.npy` (half floats) and read from there the next time. `--decode-only`
does that for one window or several and stops, so the card is held for the decodes alone and the laying, which is
CPU work, is run afterwards.

**`--set` and `--as-rendered`.** Each is `key=value` over the window's own settings: `cuts=none` (no gate),
`cuts=12,40`, `composite=whole region`, `feather_pixels=4`, `change_threshold=0.08`. `--as-rendered` changes the
first laying and `--set` the second, each from the window's settings and not from the other: a render made before
the gate was laid with `--as-rendered cuts=none`, and is then set beside the window's settings as they stand.

**What it writes**, under `--out` (untracked `data/`): `<window name>.csv`, a row a frame of the window, and
`<window name>.json`, the inputs, the settings of each laying and the summary printed.

Each row also says which latent step holds the frame and whether a cut of the source splits that step, and,
inside what the first laying keeps of the render, how much fine detail the decode and the source hold and how far
each is from its frame before. Those four columns are what a frame of a split step is set beside its neighbours
by; they are figures, and say nothing of which frame looks right. Three more say what the latent step lends
the frame: the cells of its region that the frame's own mask, grown by the margin, does not reach (all of it on a
frame across a cut; a rim where the subject moves fast inside a step), the pixels of the render the first laying
keeps there, and how far the laid frame is from the source there. `--box x0,x1,y0,y1` adds, for a place in the
picture (a prop, a hand), how far the decode is from the source there and how far the laid frame is, so a thing
that came out as the source's can be put down to the sampler or to the composite. The same pair is always
given on the subject's own mask (`mask_decode_off_source`, `mask_laid_off_source`, `mask_share_kept`).

It queues nothing and looks at nothing: whether a frame is better is a look at the clip.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap  # noqa: E402
from _lib.frames import FIT, stream  # noqa: E402

FPS = 24            # the lane's rate; inherited (`comfy_extras/nodes_minimax_h3.py::FPS`, asserted in `main`)
CELL = 16           # pixels a latent cell; inherited from the video VAE's spatial ratio
#: A pixel differs between two layings when a channel is this far apart, in 8-bit levels. Reasoned: under one level
#: nothing a file could hold has changed.
LEVEL = 1.0


def _pack(name: str):
    """A root module of the pack, loaded as the checks load it (a stand-in package, so its relative imports work)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    return importlib.import_module(f"_h3pack.{name}")


def change(settings: dict, edits: list[str]) -> dict:
    """`settings` with each `key=value` applied. `cuts` takes `none` or frame numbers; the rest take their own type."""
    out = dict(settings)
    for edit in edits:
        key, _, value = edit.partition("=")
        if key == "cuts":
            out[key] = [] if value in ("none", "") else [int(x) for x in value.split(",")]
        elif key == "feather_pixels":
            out[key] = int(value)
        elif key == "change_threshold":
            out[key] = float(value)
        elif key == "composite":
            out[key] = value
        else:
            raise SystemExit(f"--set {key}: not a setting the composite reads; one of cuts, composite, feather_pixels, change_threshold")
    return out


def window_meta(path: str) -> dict:
    """A stored window's own record: the latent's shape, its trim, where the next window starts, what its video holds."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        shape = tuple(f.get_slice("stream_0").get_shape())
    return {"shape": shape, "trim": int(meta["trim"]), "next_start": float(meta["next_start"]),
            "written": int(meta["written"]) if "written" in meta else None}


def rebuilt_region(vm, capture: Path, run_name: str, shots: str, first: int, frames: int, latent: tuple) -> tuple[dict, dict]:
    """A window's region for a render that saved none, from a capture of its run, and how good the rebuild is.

    The mask is the capture's `carried` (read back from the render's review); the tokens are the node's own
    `token_mask` of it grown by the run's margin; `cells_differing` counts, over the window, the latent cells
    where that region and the region the review shows disagree, of `cells_in_region`.
    """
    import torch
    manifest = json.loads((capture / "manifest.json").read_text())
    run = next((r for r in manifest["runs"] if r["name"] == run_name), None)
    if run is None:
        raise SystemExit(f"no run {run_name!r} in {capture}; it has {[r['name'] for r in manifest['runs']]}")
    at = run["first_source_frame"] - manifest["first_frame"]        # the load's first frame, on the capture's clock
    if at < 0:
        raise SystemExit(f"the capture starts at source frame {manifest['first_frame']}, after the run's load ({run['first_source_frame']})")
    if run.get("others") or run.get("keep"):
        raise SystemExit(f"run {run_name} wires keep or others: its token region cannot be rebuilt from its mask alone")
    graph = json.loads((capture / "runs" / run_name / "graph.json").read_text())
    graph = graph.get("prompt", graph)
    node = next((n["inputs"] for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3MaskedSource"), None)
    if node is None:
        raise SystemExit(f"the graph saved for {run_name} has no MiniMaxH3MaskedSource")
    width = manifest["size"][0]
    saved = np.load(capture / "runs" / run_name / "region.npz")
    carried = np.unpackbits(saved["carried"], axis=-1)[..., :width]
    whole = torch.from_numpy(carried[at:])              # bytes of 0 or 1: the whole load, for the margin's steadied area
    mask = whole[first:first + frames].to(torch.float32)
    short = frames - int(mask.shape[0])
    if short > 0:       # the capture stops before the window does: nothing masked there, as the node holds a short source
        mask = torch.cat([mask, torch.zeros((short,) + tuple(mask.shape[1:]))], dim=0)
    record = {"grow_pixels": int(node["grow_pixels"]), "grow_by": node.get("grow_by", vm.GROW_FIXED),
              "feather_pixels": int(node["feather_pixels"]), "subject_area": vm.area_share(whole)}
    margin = vm.source_margins(record, first, frames, int(mask.shape[1]) * int(mask.shape[2]))
    edge = node.get("edge", vm.EDGE_TOKENS)
    tokens = vm.token_mask(vm.grow(mask, margin), latent[2], latent[3], latent[4], edge == vm.EDGE_TOKENS)
    seen = torch.from_numpy(saved["region"][at:][first:first + frames])
    made = vm.pixel_alpha(tokens, latent[3], latent[4], 0)[:int(seen.shape[0])] > 0.5
    cuts = json.loads(Path(shots).read_text())["cuts"] if shots else []
    settings = {"composite": node.get("composite", vm.COMPOSITE_REGION), "feather_pixels": int(node["feather_pixels"]),
                "change_threshold": float(node.get("change_threshold", vm.CHANGE_THRESHOLD)), "cuts": [int(c) for c in cuts],
                "grow_pixels": int(node["grow_pixels"]), "grow_by": record["grow_by"], "replace": node.get("replace"), "edge": edge}
    region = {"mask": mask, "tokens": tokens, "margin": margin, "first_frame": first, "source": settings}
    how = {"region": "rebuilt from the capture's read-back of the review", "capture": capture.name, "run": run_name,
           "cells_differing": int((made != seen).sum()), "cells_in_region": int(seen.sum()),
           "frames_compared": int(seen.shape[0])}
    return region, how


def source_frames(clip: str, first: int, frames: int, size: tuple[int, int], have: int | None = None):
    """`frames` frames of the clip from frame `first`, fitted to the canvas as the loader fits them, [F, H, W, 3] in
    0..1, and how many were held. `have` is how many the load had from there on (its loader's cap less the
    window's first frame): past them the last one is repeated, as the node holds a source that runs out
    (`video_mask.window_frames`). A load capped at its shot's end must be read with it, or the frames past the
    shot are the next shot's and every figure on the window's last step is of another picture."""
    import torch
    w, h = size
    want = frames if have is None else max(min(frames, have), 1)
    got = [f.copy() for f in stream(clip, size, first, want, vf=FIT.format(w=w, h=h), pix="rgb24")]
    if not got:
        raise SystemExit(f"{clip} has no frame {first}")
    held = frames - len(got)
    got += [got[-1]] * held
    return torch.from_numpy(np.stack(got)).to(torch.float32) / 255.0, held


_VAES: dict = {}        # the loaded VAE by file name: several windows decoded in one go load it once


def decode_window(path: str, vae_name: str, device: str):
    """The window's stored video latent through the video VAE, [F, H, W, 3] in 0..1, as the song node decodes it."""
    bootstrap(cpu=device == "cpu")
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths
    if vae_name not in _VAES:
        _VAES[vae_name] = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name)))
    vae = _VAES[vae_name]
    latent = comfy.utils.load_torch_file(path)["stream_0"]
    # no autograd, or it holds every activation. `no_grad` and not the executor's `inference_mode`: off the
    # server the VAE's weights are cast inside this call, and a parameter cannot be made from an inference
    # tensor (seen on the first trial, 2026-10-10)
    with torch.no_grad():
        images = vae.decode(latent)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images.float().cpu()


def kept_decode(window: Path, out: Path, vae_name: str, device: str, decode=decode_window):
    """A window's decode and where it came from: read from `out` when it was kept there, else decoded and kept.

    Kept as half floats, finer than anything a video file holds, and read back from the file either way, so
    both layings and every later run are made from the same numbers."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    kept = out / f"decoded_{window.stem}.npy"
    if kept.is_file():
        how = f"read from {kept.name}"
    else:
        np.save(kept, decode(str(window), vae_name, device).numpy().astype(np.float16))
        how = f"decoded on {device} and kept as {kept.name}"
    return torch.from_numpy(np.load(kept)).to(torch.float32), how


def written_frames(video: str, size: tuple[int, int], count: int):
    """The window's own video file, [n, H, W, 3] in 0..1, or None when it is not there."""
    import torch
    if not Path(video).is_file():
        return None
    got = [f.copy() for f in stream(video, size, 0, count, vf=f"scale={size[0]}:{size[1]}", pix="rgb24")]
    return torch.from_numpy(np.stack(got)).to(torch.float32) / 255.0 if got else None


def detail(frames, where):
    """How much fine detail each frame holds inside `where`: the mean difference of its grey from the grey blurred
    over three pixels, in 8-bit levels, [F]; NaN where `where` is empty. A soft or smeared frame reads low."""
    import torch
    import torch.nn.functional as F
    grey = frames.mean(dim=-1)
    blur = F.avg_pool2d(F.pad(grey.unsqueeze(1), (1, 1, 1, 1), mode="replicate"), 3, stride=1)[:, 0]
    fine = (grey - blur).abs() * 255.0
    count = where.flatten(1).sum(dim=1)
    return torch.where(count > 0, (fine * where).flatten(1).sum(dim=1) / count.clamp(min=1), torch.full_like(count, float("nan"), dtype=torch.float32))


def moved(frames, where):
    """How far each frame is from the one before inside `where` on both: the mean grey difference in 8-bit levels,
    [F]; NaN for the first frame and where the two frames share no pixel of `where`."""
    import torch
    grey = frames.mean(dim=-1)
    both = (where[1:] & where[:-1])
    count = both.flatten(1).sum(dim=1)
    step = ((grey[1:] - grey[:-1]).abs() * 255.0 * both).flatten(1).sum(dim=1) / count.clamp(min=1)
    step = torch.where(count > 0, step, torch.full_like(step, float("nan")))
    return torch.cat([torch.full((1,), float("nan")), step])


def lent_cells(vm, region: dict, latent: tuple):
    """[F, h, w] of bool: the cells of each frame's region that the frame's own mask, grown by the margin, does not
    reach. A latent step's region is the maximum over its frames, so these are what the step lends a frame from
    its other frames: everything, on a frame across a cut from the subject; a rim, where the subject moves fast
    inside a step. Made with the node's own `token_mask`, a frame at a time, so the edge is the render's.
    `keep` and `others` are not in a saved region's mask, so a cell they took out of the region is never lent."""
    import torch
    mask, margin = region["mask"], region["margin"]
    whole = region["source"].get("edge", vm.EDGE_TOKENS) == vm.EDGE_TOKENS
    grown = vm.grow(mask, margin)
    own = torch.cat([vm.token_mask(grown[f:f + 1], 1, latent[3], latent[4], whole) for f in range(int(mask.shape[0]))], dim=0)
    there = vm.pixel_alpha(region["tokens"], latent[3], latent[4], 0) > 0.5
    return there & ~(own > 0.5)


def step_rows(vm, latent_t: int, cuts: list[int], first: int) -> list[dict]:
    """For each frame of a window: its latent step, the step's length, and whether a cut falls inside the step."""
    out, at = [], 0
    inside = {int(c) - int(first) for c in cuts or []}
    for k, n in enumerate(vm.run_lengths(latent_t)):
        split = any(at < c < at + n for c in inside)
        out += [{"step": k, "step_frames": n, "step_split_by_a_cut": int(split)}] * n
        at += n
    return out


def rows_of(first_source: int, trim: int, written: int | None, pixels, a, a_alpha, b, b_alpha, file, steps=None,
            images=None, box=None, lent=None, mask=None) -> list[dict]:
    """A row a frame of the window: what each laying keeps, what differs between them, and the first against the file.

    With `steps` (`step_rows`) each row says which latent step holds the frame and whether a cut splits it. With
    `images` (the decode) it also carries, inside what the first laying keeps of the render, the fine detail of
    the decode and of the source there (`detail`) and how far each is from its frame before (`moved`): what a
    frame of a split step has to be set beside its neighbours by. With `box` (x0, x1, y0, y1 in canvas pixels,
    the far edges not counted) it carries how far the decode and the first laying each are from the source inside
    the box, in levels, and the share of the box the laying keeps of the render: the same place before the
    composite and after it, which is what tells the sampler's doing from the composite's. With `lent`
    (`lent_cells`) it carries how many cells the frame's step lends it, how many pixels the first laying keeps
    of the render inside them, and how far the laid frame is from the source there, in levels. With `mask`
    (the window's own fitted mask) it carries the same pair as the box, on the subject: how far the decode and
    the laid frame each are from the source under the mask, and the share of the mask the laying keeps."""
    import torch
    out = []
    lent_px = None
    if lent is not None:
        cell = int(a.shape[1]) // int(lent.shape[1])
        lent_px = lent.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)[:, :a.shape[1], :a.shape[2]]
    kept = a_alpha > 0.5
    fine = None if images is None else (detail(images, kept), detail(pixels, kept), moved(images, kept), moved(pixels, kept))
    for f in range(int(a.shape[0])):
        in_file = f >= trim and (written is None or f - trim < written)
        diff = ((a[f] - b[f]).abs() * 255.0 > LEVEL).any(dim=-1)
        row = {"frame": f, "source_frame": first_source + f, "written": int(in_file),
               "kept_px_as_rendered": int((a_alpha[f] > 0.5).sum()), "kept_px_changed": int((b_alpha[f] > 0.5).sum()),
               "px_differing": int(diff.sum()),
               "changed_is_source": int(bool((b[f] == pixels[f]).all())),
               "as_rendered_is_source": int(bool((a[f] == pixels[f]).all())),
               "as_rendered_off_file": None}
        if file is not None and in_file and f - trim < int(file.shape[0]):
            row["as_rendered_off_file"] = round(float((a[f] - file[f - trim]).abs().mean()) * 255.0, 3)
        if steps is not None:
            row.update(steps[f])
        if mask is not None and images is not None:
            on = mask[f] > 0.5
            if bool(on.any()):
                row["mask_px"] = int(on.sum())
                row["mask_decode_off_source"] = round(float((images[f] - pixels[f]).abs().mean(dim=-1)[on].mean()) * 255.0, 3)
                row["mask_laid_off_source"] = round(float((a[f] - pixels[f]).abs().mean(dim=-1)[on].mean()) * 255.0, 3)
                row["mask_share_kept"] = round(float((a_alpha[f] > 0.5)[on].float().mean()), 4)
            else:
                row.update(mask_px=0, mask_decode_off_source=None, mask_laid_off_source=None, mask_share_kept=None)
        if lent_px is not None:
            here = lent_px[f]
            count = int(here.sum())
            row["lent_cells"] = int(lent[f].sum())
            row["lent_px_kept"] = int(((a_alpha[f] > 0.5) & here).sum())
            row["lent_off_source"] = (round(float((a[f] - pixels[f]).abs().mean(dim=-1)[here].mean()) * 255.0, 3)
                                      if count else None)
        if box is not None and images is not None:
            x0, x1, y0, y1 = box
            there = (slice(y0, y1), slice(x0, x1))
            row["box_decode_off_source"] = round(float((images[f][there] - pixels[f][there]).abs().mean()) * 255.0, 3)
            row["box_laid_off_source"] = round(float((a[f][there] - pixels[f][there]).abs().mean()) * 255.0, 3)
            row["box_share_kept"] = round(float((a_alpha[f][there] > 0.5).float().mean()), 4)
        if fine is not None:
            for name, series in zip(("detail_render", "detail_source", "moved_render", "moved_source"), fine):
                value = float(series[f])
                row[name] = None if value != value else round(value, 3)
        out.append(row)
    return out


def spans(frames: list[int]) -> str:
    """Frame numbers as ranges: `3, 7-9`."""
    out, run = [], []
    for f in sorted(frames) + [None]:
        if run and (f is None or f != run[-1] + 1):
            out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
            run = []
        if f is not None:
            run.append(f)
    return ", ".join(out) or "none"


def summary(rows: list[dict]) -> dict:
    moved = [r for r in rows if r["px_differing"]]
    off = [r["as_rendered_off_file"] for r in rows if r["as_rendered_off_file"] is not None]
    return {"frames": len(rows), "frames_written": sum(r["written"] for r in rows),
            "source_frames_differing": [r["source_frame"] for r in moved],
            "source_frames_differing_and_written": [r["source_frame"] for r in moved if r["written"]],
            "of_those_now_the_source": [r["source_frame"] for r in moved if r["changed_is_source"]],
            "of_those_not_the_source": [r["source_frame"] for r in moved if not r["changed_is_source"]],
            "px_differing_most": max((r["px_differing"] for r in rows), default=0),
            "as_rendered_off_file_levels": ({"frames": len(off), "median": round(float(np.median(off)), 3), "most": round(max(off), 3)}
                                            if off else None)}


def run(a: argparse.Namespace, decode=decode_window) -> dict:
    bootstrap(cpu=a.device == "cpu")
    import torch
    vm = _pack("video_mask")
    from comfy_extras.nodes_minimax_h3 import FPS as core_fps
    if int(core_fps) != FPS:
        raise SystemExit(f"core renders at {core_fps} frames a second and this tool counts in {FPS}")
    window = Path(a.window)
    stored = window_meta(str(window))
    latent = stored["shape"]
    frames = sum(vm.run_lengths(latent[2]))
    size = (latent[4] * CELL, latent[3] * CELL)
    saved = Path(a.region) if a.region else window.with_name(window.stem + "_region.npz")
    if saved.is_file():
        region, how = vm.load_window_region(str(saved)), {"region": "the file the render saved", "file": saved.name}
    elif a.capture and a.run:
        if a.first_frame is not None:
            first = a.first_frame
        elif a.context is not None:
            first = int(round(stored["next_start"] * FPS)) - (frames - a.context)
        else:
            raise SystemExit("a rebuilt region needs the window's first frame in its load: --first-frame, or --context "
                             "(the run's context_frames, from which the stored next_start gives it)")
        region, how = rebuilt_region(vm, Path(a.capture), a.run, a.shots, first, frames, latent)
    else:
        raise SystemExit(f"{saved.name} is not beside the window (a render from before regions were saved): give "
                         "--capture and --run to rebuild it from a capture, with --shots for its cuts")
    if tuple(region["mask"].shape) != (frames, size[1], size[0]) or tuple(region["tokens"].shape) != tuple(latent[2:]):
        raise SystemExit(f"the region is {tuple(region['mask'].shape)} with tokens {tuple(region['tokens'].shape)}; the "
                         f"window is {frames} frames at {size[0]}x{size[1]} with a latent of {tuple(latent[2:])}")
    first = int(region["first_frame"])
    pixels, held = source_frames(a.source, a.source_first + first, frames, size,
                                 None if a.source_frames is None else a.source_frames - first)

    out = Path(a.out)
    images, how["decode"] = kept_decode(window, out, a.vae, a.device, decode)
    if tuple(images.shape[:3]) != tuple(pixels.shape[:3]):
        raise SystemExit(f"the decode is {tuple(images.shape)}; the source's frames are {tuple(pixels.shape)}")

    as_rendered = change(region["source"], a.as_rendered or [])
    changed = change(region["source"], a.set or [])
    laid_a, alpha_a, lines_a = vm.lay_window(images, pixels, region["tokens"], region["mask"], as_rendered, region["margin"], first)
    laid_b, alpha_b, lines_b = vm.lay_window(images, pixels, region["tokens"], region["mask"], changed, region["margin"], first)
    file = written_frames(str(window.with_suffix(".mp4")), size, stored["written"] or frames)
    rows = rows_of(a.source_first + first, stored["trim"], stored["written"], pixels, laid_a, alpha_a, laid_b, alpha_b, file,
                   step_rows(vm, latent[2], region["source"].get("cuts"), first), images,
                   [int(x) for x in a.box.split(",")] if a.box else None, lent_cells(vm, region, latent), region["mask"])
    with open(out / f"{window.stem}.csv", "w", newline="") as fh:
        table = csv.DictWriter(fh, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    record = {"window": window.name, "source": Path(a.source).name, "source_first": a.source_first,
              "window_first_frame": first, "frames": frames, "source_frames_held": held, "inputs": how,
              "as_rendered": as_rendered, "changed": changed, "lines_as_rendered": lines_a, "lines_changed": lines_b,
              "summary": summary(rows)}
    (out / f"{window.stem}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def report(record: dict) -> None:
    s, how = record["summary"], record["inputs"]
    print(f"{record['window']}: {record['frames']} frames from source frame {record['source_first'] + record['window_first_frame']}, "
          f"{s['frames_written']} of them in its video; region: {how['region']}; {how['decode']}")
    if "cells_differing" in how:
        print(f"  LAID FROM A READ-BACK MASK, not the render's own: the token region rebuilt from it differs from the "
              f"region the review shows on {how['cells_differing']} cells of {how['cells_in_region']} in the region, "
              f"over {how['frames_compared']} frames")
    differ = [k for k in record["changed"] if record["changed"][k] != record["as_rendered"][k]]
    print("  changed: " + (", ".join(f"{k} {record['as_rendered'][k]} -> {record['changed'][k]}" for k in differ) or "nothing"))
    for name, lines in (("as rendered", record["lines_as_rendered"]), ("changed", record["lines_changed"])):
        for line in lines:
            print(f"  {name}: {line}")
    print(f"  source frames that differ between the two: {spans(s['source_frames_differing'])}"
          f" (in the window's video: {spans(s['source_frames_differing_and_written'])})")
    print(f"  of those, the source bit for bit after the change: {spans(s['of_those_now_the_source'])}; "
          f"not the source: {spans(s['of_those_not_the_source'])}")
    off = s["as_rendered_off_file_levels"]
    print("  the first laying against the window's own video: "
          + (f"median {off['median']} levels a frame, at most {off['most']}, over {off['frames']} frames" if off
             else "no video beside the window to compare"))


def main() -> None:
    p = argparse.ArgumentParser(description="Lay a saved window of a masked render again and say what a change moves.")
    p.add_argument("--window", required=True, nargs="+", help="a stored window, <name>_window_N.safetensors; several with --decode-only")
    p.add_argument("--decode-only", action="store_true",
                   help="decode each window and keep the decode under --out, then stop: the part that needs a card, on its own")
    p.add_argument("--source", help="the clip the run loaded")
    p.add_argument("--source-first", type=int, help="the clip's frame the run's load started on")
    p.add_argument("--source-frames", type=int, help="how many frames the run's load had (its loader's cap), when it "
                                                     "ends inside the window: the last is held from there, as the node holds it")
    p.add_argument("--region", help="the window's saved region; default: beside the window")
    p.add_argument("--capture", help="a capture folder of the run, to rebuild a region the render did not save")
    p.add_argument("--run", help="the run's name in that capture")
    p.add_argument("--shots", help="the render's shot table (_shots.json), for a rebuilt region's cuts")
    p.add_argument("--context", type=int, help="the run's context_frames, for a rebuilt region's first frame")
    p.add_argument("--first-frame", type=int, help="the window's first frame in its load, in place of --context")
    p.add_argument("--as-rendered", action="append", metavar="KEY=VALUE", help="a setting of the first laying")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="a setting of the second laying")
    p.add_argument("--box", metavar="X0,X1,Y0,Y1", help="a box in canvas pixels to read before and after the composite")
    p.add_argument("--vae", help="the video VAE's file name; default: the shipped one")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", required=True, help="a folder under data/ for the decode, the table and the record")
    a = p.parse_args()
    if a.vae is None:
        sys.path.insert(0, str(REPO / "workflows"))
        import h3_config
        a.vae = h3_config.MODELS["video_vae"]
    if a.decode_only:
        import time
        for window in a.window:
            began = time.perf_counter()
            images, how = kept_decode(Path(window), Path(a.out), a.vae, a.device)
            print(f"{Path(window).name}: {tuple(images.shape)}, {how}, {time.perf_counter() - began:.1f} s")
        return
    if len(a.window) != 1 or a.source is None or a.source_first is None:
        raise SystemExit("laying a window takes one --window, with --source and --source-first")
    a.window = a.window[0]
    report(run(a))


if __name__ == "__main__":
    main()
