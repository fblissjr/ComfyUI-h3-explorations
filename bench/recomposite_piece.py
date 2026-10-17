#!/usr/bin/env python3
"""Lay a whole masked render again from its saved windows with one composite setting changed, as a NEW piece.

    <python> bench/recomposite_piece.py --render R_00001.mp4 --set "composite=whole region" --name second_hfp_0278c_rc \\
        --out-dir <output>/Video/fun/<session> --by WHO [--device cuda]
    <python> bench/recomposite_piece.py --render R_00001.mp4 --save-own-matte --by WHO [--device cuda]

**What it is for** (the owner's "fix that" of 2026-10-16 on the mom's 0278c, whose neck comes and goes at 287 under
"only what changed"; the lead's order). A render keeps each window's sampled latent and its region; laid again with the
node's own `video_mask.lay_window` (and `lay_back` for a zoomed load) and a setting changed, it is a new piece with no
sampling. `bench/recomposite_window.py` does this for one window to read; this writes the piece.

**`--save-own-matte`** (`C2` item 3, 2026-10-17): for a laid piece from before the matte was saved, decodes each
window's stored latent with the render's own VAE and runs `lay_window` (and `lay_back` when zoomed) under the render's
own `composite` from its graph, checks the control that the reconstructed composite reproduces the rendered `.mp4`
within `OWN_MATTE_TOLERANCE` levels (both inside the matte and outside the region, recording mean, p99 and worst pixel
differences in each), and atomically writes `matte` and `kept` (and `piece` when absent) into each window's
`*_window_*_region.npz` with `<stem>.matte_reconstructed.json` beside the render.

**What it writes**, a folder shaped as a render's: `<out-dir>/<name>/<name>_00001.mp4` (the render's own sound copied
over), `<name>_00001_zoomed.mp4` and `_zoomed.json` when the load was zoomed, and `<name>_windows/` with each window's
video and region file, the region holding the node's matte and which piece its frames went into (`piece`), so the
assembler lays the new piece by its matte (`assemble_delivery.Matte`). Beside the piece, `<name>_00001.recomposite.json`:
the render it came from by path and sha256, the setting changed and the settings before, the command, who and when; and
the piece's sidecar says it is a re-composite of that render. The render's PNG graph is NOT copied: it says the old
composite, and the piece is laid by its matte.

**Read from the render, never typed** (the re-composite read, the lead's ruling of 2026-10-16): the source clip, its first
frame and the loader's frame cap from the render's own graph (the PNG beside it: the Masked Source's `frames` loader), past
which the node held the last loaded frame and so does this; and the VAE and encoder `crf` the render ran with (the song
node's `vae` and `crf`). `--source`, `--source-first`, `--vae` and `--crf`, when given, must agree with the graph. Before
anything is decoded, the render's own frames outside its region are set against the source read at its first frame and one
frame either side: a neighbour that fits better is refused, with the figures.

**What it refuses**: a render with no graph beside it; a window whose latent or region file is missing, or whose region says
another run's piece (`piece.stem`, the folder is shared by every run of a prefix); an output folder that exists; a run that
continued another (`continue_from`) and whose region files do not say where their frames went in the piece; written frames
that do not join from the piece's frame 0 with no gap or overlap; `--device cpu` when the render's VAE is int8 (which decodes
on CUDA only). A render from before the piece record carries no stem in its region files, and only the join holds it to its
own windows.

**The decode is a model**: the video VAE (`--device cuda` by default, since the int8 video VAE decodes on CUDA only).
Everything is checked before anything is written; the work goes into a fresh `<out-dir>/<name>.part/`, cleared at the
start and renamed to `<name>/` at the end, so nothing an earlier run made is reused by a key and a run that failed
leaves no folder that blocks its rerun.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recomposite_window as rw  # noqa: E402
from _lib import bootstrap  # noqa: E402
from _lib.frames import FIT, FPS  # noqa: E402

#: The latent cell's size in canvas pixels: the window's canvas is its latent's height and width times this (`recomposite_window`).
CELL = rw.CELL
#: The most a window's reconstructed own-composite picture may sit from the rendered `.mp4` (mean absolute difference in
#: 8-bit levels, inside the matte and outside the region) for `--save-own-matte` to accept the reconstruction.
#: **Measured** on 0278c and 0484c on 2026-10-16 (`review2_recomposite_piece.md`): 0.70–1.45 levels inside the matte and
#: 0.56–1.20 outside the region across H.264 `-crf 12` encoding.
OWN_MATTE_TOLERANCE = 2.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_rate(clip: str) -> float:
    """The source clip's frame rate (`r_frame_rate` of `v:0`) as a float, or `FPS` when unreadable."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1) if num and float(den or 1) else float(FPS)


def windows_of(render: Path) -> list[tuple[int, Path, Path]]:
    """Each window of a render: (number, its stored latent, its region file), in window order. Refused where one lacks either."""
    prefix = render.stem.rsplit("_", 1)[0]
    folder = render.parent / f"{prefix}_windows"
    found = sorted((int(m.group(1)), p) for p in folder.glob(f"{prefix}_window_*.safetensors")
                   for m in [re.fullmatch(rf"{re.escape(prefix)}_window_(\d+)\.safetensors", p.name)] if m)
    if not found:
        raise SystemExit(f"{folder} holds no stored window latent ({prefix}_window_N.safetensors): there is nothing to lay again")
    out = []
    for n, latent in found:
        region = latent.with_name(latent.stem + "_region.npz")
        if not region.is_file():
            raise SystemExit(f"{region.name} is not beside {latent.name}: the composite that window ran with is not known")
        out.append((n, latent, region))
    return out


def continued_from(render: Path) -> str:
    """The song node's `continue_from` in the graph VHS saved beside the render, '' when none or no graph."""
    png = render.with_suffix(".png")
    if not png.is_file():
        return ""
    from PIL import Image
    with Image.open(png) as im:
        graph = json.loads(im.info.get("prompt") or "{}")
    return next((str(n["inputs"].get("continue_from") or "") for n in graph.values()
                 if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3AudioFreezeSong"), "")


def loader_frames(clip: str, first: int, frames: int, size, have: int | None = None):
    """`frames` frames of the clip from frame `first` AS THE RENDER'S LOADER HANDED THEM to the node (VHS: 16 bits, `rgba64le`, divided by
    65535; the loader's fit), [F, H, W, 3] in 0..1, and how many were held: past `have` (the loader's cap less the window's first frame) the
    last is repeated, as the node holds a source that runs out. Not `recomposite_window.source_frames`, which reads 8 bits: ffmpeg's 8-bit
    read of the same frames is +1.58 / +1.60 levels darker (0278c's source, measured 2026-10-16), and a re-laid piece would be that much darker than
    the render it replaces (the re-composite read, question 8)."""
    import torch
    w, h = (int(v) for v in size)
    want = frames if have is None else max(min(frames, have), 1)
    chain = ([f"select=gte(n\\,{int(first)})"] if first else []) + [FIT.format(w=w, h=h)]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0", "-vf", ",".join(chain), "-fps_mode", "passthrough",
                          "-frames:v", str(int(want)), "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"], capture_output=True).stdout
    got = np.frombuffer(raw, np.dtype("<u2")).reshape(-1, h, w, 4)[..., :3]
    if not len(got):
        raise SystemExit(f"{clip} has no frame {first}")
    held = frames - len(got)
    got = np.concatenate([got, np.repeat(got[-1:], held, axis=0)]) if held else got
    return torch.from_numpy((got / 65535.0).astype(np.float32)), held


def graph_of(render: Path) -> dict:
    """The graph VHS saved beside the render (`<stem>.png`, its `prompt`); refused when there is none."""
    png = render.with_suffix(".png")
    if not png.is_file():
        raise SystemExit(f"{png.name} is not beside the render: its source, first frame, loader cap and VAE are read from its graph, never typed")
    from PIL import Image
    with Image.open(png) as im:
        return json.loads(im.info.get("prompt") or "{}")


def source_of(graph: dict) -> dict:
    """The render's source as its graph loaded it: the clip, the source frame its frame 0 is on, the loader's frame cap, the size."""
    ms = next((n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3MaskedSource"), None)
    if ms is None or not isinstance(ms["inputs"].get("frames"), list):
        raise SystemExit("the render's graph holds no Masked Source with a frames loader: the source is not known")
    loader = graph[str(ms["inputs"]["frames"][0])]["inputs"]
    video = str(loader["video"])
    rate = clip_rate(video)
    return {"video": video, "first": int(round(float(loader.get("start_time") or 0.0) * rate)),
            "cap": int(loader.get("frame_load_cap") or 0) or None, "size": [int(loader.get("custom_width") or 0), int(loader.get("custom_height") or 0)],
            "composite": str(ms["inputs"].get("composite") or "")}


def vae_of(graph: dict) -> str:
    """The VAE file the render decoded with: the song node's `vae` input's loader."""
    song_node = next((n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3AudioFreezeSong"), None)
    if song_node is None or not isinstance(song_node["inputs"].get("vae"), list):
        raise SystemExit("the render's graph holds no song node with a VAE: the VAE it decoded with is not known")
    return str(graph[str(song_node["inputs"]["vae"][0])]["inputs"]["vae_name"])


def crf_of(graph: dict) -> int:
    """The encoder `crf` the render wrote its video with: the song node's `crf` input."""
    song_node = next((n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3AudioFreezeSong"), None)
    if song_node is None or song_node.get("inputs", {}).get("crf") is None:
        raise SystemExit("the render's graph holds no song node with a crf: the encoder quality it wrote with is not known")
    return int(song_node["inputs"]["crf"])


def offset_fit(vm, render: Path, source: dict, region: dict, said: dict | None, size, frames: int, trim: int) -> dict:
    """The render's own frames outside its region against the source read at their frames and one frame either side, in levels: the frame
    offset that fits best should be 0. Read on the first window's written frames: piece frame q is the window's frame trim + q - its piece
    first (0 for a render from before the piece record). Outside: where the window's region (laid back through the box when zoomed) is 0."""
    import torch
    zoom, first = region.get("zoom"), int(region["first_frame"])
    p0 = int(said["first"]) if said else 0
    weight = vm.pixel_alpha(region["tokens"], size[1], size[0], 0)
    if zoom is not None:
        blank = torch.zeros(frames, size[1], size[0], 3)
        weight = vm.lay_back(torch.ones_like(blank), weight, blank, zoom)[..., 0]
    outside = weight == 0
    count = min(3, frames - trim - 1)
    got = rw.written_frames(str(render), tuple(size), p0 + count + 1)
    if got is None or int(got.shape[0]) < p0 + count + 1 or count < 1:
        return {"read": False, "why": "the render holds too few frames to read an offset"}
    out = {}
    for shift in (-1, 0, 1):
        at = source["first"] + first + trim + 1 + shift
        if at < 0:
            continue
        src, _ = loader_frames(source["video"], at, count, tuple(size))
        diff = (got[p0 + 1:p0 + 1 + count] - src).abs().mean(-1) * 255.0
        where = outside[trim + 1:trim + 1 + count]
        out[shift] = round(float(diff[where].mean()), 3) if bool(where.any()) else None
    return {"read": True, "levels_by_frame_offset": out}


def _check_render(vm, render: Path, a: argparse.Namespace, decode):
    """Verify the render, its graph, its VAE, its crf, its windows, and its source frame offset before any decode or write."""
    if not render.is_file():
        raise SystemExit(f"{render} is not a file")
    graph = graph_of(render)
    source = source_of(graph)
    graph_vae = vae_of(graph)
    graph_crf = crf_of(graph)
    for given, said, what in ((a.source, source["video"], "--source"), (a.source_first, source["first"], "--source-first"),
                              (a.vae, graph_vae, "--vae"), (a.crf, graph_crf, "--crf")):
        if given is not None and str(given) != str(said):
            raise SystemExit(f"{what} {given} is not the render's own ({said}, read from its graph)")
    if a.device == "cpu" and "int8" in graph_vae.lower():
        raise SystemExit(f"the render's VAE {graph_vae} is int8, which decodes on CUDA only: pass --device cuda")
    windows = windows_of(render)
    checked = []
    for n, latent, region_path in windows:
        with np.load(region_path) as z:
            said = json.loads(str(z["meta"])).get("piece")
        if said and said.get("stem") != render.stem:
            raise SystemExit(f"{region_path.name} is {said.get('stem')}'s window, not {render.stem}'s: another run of the prefix wrote over it")
        checked.append((n, latent, region_path, said))
    first_region = vm.load_window_region(str(windows[0][2]))
    if getattr(a, "set", None):
        rw.change(first_region["source"], a.set)
    vae = graph_vae
    import folder_paths
    vae_path = folder_paths.get_full_path("vae", vae)
    if vae_path is None and decode is None:
        raise SystemExit(f"the VAE {vae} (the render's own, read from its graph) is not in the models' vae folder")
    shape0 = rw.window_meta(str(windows[0][1]))["shape"]
    fit = offset_fit(vm, render, source, first_region, checked[0][3], (shape0[4] * CELL, shape0[3] * CELL), sum(vm.run_lengths(shape0[2])),
                     int(first_region["trim"]))
    if fit["read"]:
        levels = {k: v for k, v in fit["levels_by_frame_offset"].items() if v is not None}
        if 0 not in levels or any(v < levels[0] for k, v in levels.items() if k != 0):
            raise SystemExit(f"the render's frames outside its region fit the source better one frame off: {levels} (levels, by frame offset); "
                             "the source or its first frame is not the render's")
    return source, vae, vae_path, graph_crf, checked, fit


def save_own_matte(a: argparse.Namespace, decode=None) -> dict:
    """Reconstruct and save each window's matte (`matte`, `kept`, and `piece`) from its stored latent and region file,
    verified against the rendered `.mp4` under the render's own composite (`C2` item 3)."""
    bootstrap(cpu=True if a.device == "cpu" else None)
    import torch
    vm = rw._pack("video_mask")
    song = rw._pack("audio_freeze_song")
    render = Path(a.render)
    if a.set:
        raise SystemExit("--save-own-matte reconstructs the render's own composite and takes no --set")
    source, vae, vae_path, crf, checked, fit = _check_render(vm, render, a, decode)
    head = None
    planned, rows = [], []
    with tempfile.TemporaryDirectory(prefix="recomposite_own_matte_") as tmp:
        work = Path(tmp)
        for n, latent, region_path, said in checked:
            stored = rw.window_meta(str(latent))
            shape = stored["shape"]
            frames = sum(vm.run_lengths(shape[2]))
            size = (shape[4] * CELL, shape[3] * CELL)
            region = vm.load_window_region(str(region_path))
            first, trim, zoom = int(region["first_frame"]), int(region["trim"]), region.get("zoom")
            have = None if source["cap"] is None else source["cap"] - first
            written = int(said["written"]) if said else stored["written"]
            if written is None:
                raise SystemExit(f"{latent.name} does not say how many frames its window wrote")
            if said:
                piece_first = int(said["first"])
            else:
                if head is None:
                    if continued_from(render):
                        raise SystemExit(f"{render.name} continued {continued_from(render)} and its region files do not say where their frames went in "
                                         "the piece (a render from before the piece record): its first window's context is not known")
                    head = 0
                piece_first = first + trim - head
            pixels, held = loader_frames(source["video"], source["first"] + first, frames,
                                         size if zoom is None else tuple(int(v) for v in zoom["source_size"]), have)
            if zoom is not None:
                pixels = vm.crop_frames(pixels, zoom["box"], *size)
            images, decoded = rw.kept_decode(latent, work, vae, a.device, decode or rw.decode_window)
            if tuple(images.shape[:3]) != tuple(pixels.shape[:3]):
                raise SystemExit(f"{latent.name}: the decode is {tuple(images.shape)} and the source's frames are {tuple(pixels.shape)}")
            own_settings = dict(region["source"])
            if source.get("composite"):
                own_settings["composite"] = source["composite"]
            laid, alpha, lines = vm.lay_window(images, pixels, region["tokens"], region["mask"], own_settings, region["margin"], first)
            if zoom is not None:
                whole, _ = loader_frames(source["video"], source["first"] + first, frames, size, have)
                piece_frames, matte = vm.lay_back(images, alpha, whole, zoom, weight=True)
            else:
                whole = pixels
                piece_frames, matte = laid, vm.matte_of(alpha)
            mine = piece_frames[trim:trim + written]
            off_region = (matte[trim:trim + written] == 0)
            outside_levels = float(((mine - whole[trim:trim + written]).abs().amax(-1) * 255.0)[off_region].max()) if bool(off_region.any()) else None
            planned.append((piece_first, n, latent, region_path, region, own_settings, zoom, first, trim, written, alpha, matte, mine,
                            outside_levels, held, decoded, lines))
        at = 0
        for piece_first, _n, _l, _rp, _reg, _os, _z, _f, _t, written, _a, _m, _mine, _ol, _h, _d, _ln in sorted(planned, key=lambda p: p[0]):
            if piece_first != at:
                raise SystemExit(f"the windows' written frames do not join: one starts at piece frame {piece_first}, the frames before end at {at}")
            at += int(written)
        joined = torch.cat([p[12] for p in sorted(planned, key=lambda p: p[0])])
        control_mp4 = work / "control.mp4"
        song._write_frames_mp4(str(control_mp4), joined, crf)
        size0 = (int(joined.shape[2]), int(joined.shape[1]))
        reconstructed = rw.written_frames(str(control_mp4), size0, at)
        rendered = rw.written_frames(str(render), size0, at)
    if rendered is None or int(rendered.shape[0]) != at or reconstructed is None or int(reconstructed.shape[0]) != at:
        raise SystemExit(f"{render.name} does not hold the {at} frames its windows wrote")
    to_write = []
    for (piece_first, n, latent, region_path, region, own_settings, zoom, first, trim, written, alpha, matte, _mine,
         outside_levels, held, decoded, lines) in sorted(planned, key=lambda p: p[0]):
        got_w = reconstructed[piece_first:piece_first + written]
        ref = rendered[piece_first:piece_first + written]
        diff = (got_w - ref).abs().mean(-1) * 255.0
        w_matte = matte[trim:trim + written]
        inside = w_matte > 0
        outside = w_matte == 0
        inside_vals = diff[inside].numpy() if bool(inside.any()) else np.zeros(1, np.float32)
        outside_vals = diff[outside].numpy() if bool(outside.any()) else np.zeros(1, np.float32)
        inside_mean = float(inside_vals.mean())
        inside_p99 = float(np.percentile(inside_vals, 99))
        inside_worst = float(inside_vals.max())
        outside_mean = float(outside_vals.mean())
        outside_p99 = float(np.percentile(outside_vals, 99))
        outside_worst = float(outside_vals.max())
        whole_mean = float(diff.mean())
        if inside_mean > OWN_MATTE_TOLERANCE or outside_mean > OWN_MATTE_TOLERANCE or whole_mean > OWN_MATTE_TOLERANCE:
            raise SystemExit(f"{render.name} window {n}: own-composite control disagrees with the rendered file "
                             f"(inside matte mean {inside_mean:.3f} levels, p99 {inside_p99:.3f}, worst {inside_worst:.3f}; "
                             f"outside mean {outside_mean:.3f}, p99 {outside_p99:.3f}, worst {outside_worst:.3f}; "
                             f"overall {whole_mean:.3f}; limit {OWN_MATTE_TOLERANCE}): the region file is not modified")
        before_sha = sha256(region_path)
        to_write.append((n, latent, region_path, before_sha, region, own_settings, zoom, first, trim, written, piece_first,
                         alpha, matte, outside_levels, held, decoded, lines,
                         {"inside_matte_mean_levels": round(inside_mean, 3), "inside_matte_p99_levels": round(inside_p99, 3),
                          "inside_matte_worst_levels": round(inside_worst, 3), "inside_matte_max_levels": round(inside_worst, 3),
                          "outside_matte_mean_levels": round(outside_mean, 3), "outside_matte_p99_levels": round(outside_p99, 3),
                          "outside_matte_worst_levels": round(outside_worst, 3), "outside_matte_max_levels": round(outside_worst, 3),
                          "overall_mean_levels": round(whole_mean, 3)}))
    for (n, latent, region_path, before_sha, region, own_settings, zoom, first, trim, written, piece_first,
         alpha, matte, outside_levels, held, decoded, lines, control) in to_write:
        vm.save_window_region(str(region_path), region["mask"], region["tokens"],
                              {**own_settings, "zoom": zoom}, region["margin"], first, trim,
                              kept=alpha, matte=matte,
                              piece={"stem": render.stem, "first": piece_first, "written": int(written)})
        rows.append({"window": n, "latent": latent.name, "latent_sha256": sha256(latent),
                     "region": region_path.name, "region_sha256_before": before_sha, "region_sha256_after": sha256(region_path),
                     "piece_first": piece_first, "written": int(written), "decode": decoded, "source_frames_held": int(held),
                     "outside_the_region_before_encoding_levels_most": None if outside_levels is None else round(outside_levels, 3),
                     "composite": own_settings.get("composite"), "lines": lines, "control": control})
    command = [sys.executable, *sys.argv] if a.command is None else a.command
    record = {"what": "a render's own matte and kept mask reconstructed from its stored window latents and regions, verified against the render itself",
              "render": str(render), "render_sha256": sha256(render), "composite": source.get("composite"), "crf": crf,
              "windows": rows, "source": source, "source_offset_fit": fit,
              "vae": {"file": vae, "path": str(vae_path) if vae_path else None,
                      "sha256": sha256(Path(vae_path)) if vae_path and Path(vae_path).is_file() else None},
              "frames": at, "command": " ".join(shlex.quote(str(x)) for x in command), "by": a.by,
              "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    rec_path = render.with_name(f"{render.stem}.matte_reconstructed.json")
    rec_part = rec_path.with_name(rec_path.name + ".part")
    rec_part.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(rec_part, rec_path)
    return record


def run(a: argparse.Namespace, decode=None) -> dict:
    if getattr(a, "save_own_matte", False):
        return save_own_matte(a, decode=decode)
    bootstrap(cpu=True if a.device == "cpu" else None)
    import torch
    vm = rw._pack("video_mask")
    song = rw._pack("audio_freeze_song")
    render = Path(a.render)
    if not a.name or not a.out_dir:
        raise SystemExit("laying a new piece requires --name and --out-dir (or pass --save-own-matte)")
    folder = Path(a.out_dir) / a.name
    if folder.exists():
        raise SystemExit(f"{folder} exists: a re-composite is written into a new folder, never over one")
    # EVERYTHING CHECKED BEFORE ANYTHING IS WRITTEN (the re-composite read)
    source, vae, vae_path, crf, checked, fit = _check_render(vm, render, a, decode)
    new_stem = f"{a.name}_00001"
    head = None
    pieces, zoomed_pieces, rows, zoom_note = [], [], [], None
    part = folder.with_name(folder.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    work = part / "work"
    new_windows = part / f"{a.name}_windows"
    new_windows.mkdir(parents=True)
    for n, latent, region_path, said in checked:
        stored = rw.window_meta(str(latent))
        shape = stored["shape"]
        frames = sum(vm.run_lengths(shape[2]))
        size = (shape[4] * CELL, shape[3] * CELL)
        region = vm.load_window_region(str(region_path))
        changed = rw.change(region["source"], a.set or [])
        first, trim, zoom = int(region["first_frame"]), int(region["trim"]), region.get("zoom")
        have = None if source["cap"] is None else source["cap"] - first          # past the loader's cap the node held the last frame
        written = int(said["written"]) if said else stored["written"]
        if written is None:
            raise SystemExit(f"{latent.name} does not say how many frames its window wrote")
        if said:
            piece_first = int(said["first"])
        else:
            if head is None:
                if continued_from(render):
                    raise SystemExit(f"{render.name} continued {continued_from(render)} and its region files do not say where their frames went in "
                                     "the piece (a render from before the piece record): its first window's context is not known")
                head = 0
            piece_first = first + trim - head
        pixels, held = loader_frames(source["video"], source["first"] + first, frames,
                                     size if zoom is None else tuple(int(v) for v in zoom["source_size"]), have)
        if zoom is not None:
            pixels = vm.crop_frames(pixels, zoom["box"], *size)
        images, decoded = rw.kept_decode(latent, work, vae, a.device, decode or rw.decode_window)
        if tuple(images.shape[:3]) != tuple(pixels.shape[:3]):
            raise SystemExit(f"{latent.name}: the decode is {tuple(images.shape)} and the source's frames are {tuple(pixels.shape)}")
        laid, alpha, lines = vm.lay_window(images, pixels, region["tokens"], region["mask"], changed, region["margin"], first)
        if zoom is not None:
            whole, _ = loader_frames(source["video"], source["first"] + first, frames, size, have)
            piece_frames, matte = vm.lay_back(images, alpha, whole, zoom, weight=True)
            zoomed_pieces.append((piece_first, laid[trim:trim + written]))
            zoom_note = zoom
        else:
            whole = pixels
            piece_frames, matte = laid, vm.matte_of(alpha)
        mine = piece_frames[trim:trim + written]
        # the laid frames outside the region before any encode: the source itself (the re-composite read), as data and for the check
        off_region = (matte[trim:trim + written] == 0)
        outside_levels = float(((mine - whole[trim:trim + written]).abs().amax(-1) * 255.0)[off_region].max()) if bool(off_region.any()) else None
        pieces.append((piece_first, mine))
        song._write_frames_mp4(str(new_windows / f"{a.name}_window_{n}.mp4"), mine, crf)
        vm.save_window_region(str(new_windows / f"{a.name}_window_{n}_region.npz"), region["mask"], region["tokens"],
                              {**changed, "zoom": zoom}, region["margin"], first, trim, kept=alpha, matte=matte,
                              piece={"stem": new_stem, "first": piece_first, "written": int(mine.shape[0])})
        rows.append({"window": n, "latent": latent.name, "latent_sha256": sha256(latent), "region": region_path.name, "region_sha256": sha256(region_path),
                     "piece_first": piece_first, "written": int(mine.shape[0]), "decode": decoded, "source_frames_held": int(held),
                     "outside_the_region_before_encoding_levels_most": None if outside_levels is None else round(outside_levels, 3),
                     "lines": lines, "composite_before": region["source"].get("composite"), "composite_now": changed.get("composite")})
    at = 0
    for piece_first, mine in sorted(pieces, key=lambda p: p[0]):
        if piece_first != at:
            raise SystemExit(f"the windows' written frames do not join: one starts at piece frame {piece_first}, the frames before end at {at}")
        at += int(mine.shape[0])
    joined = torch.cat([m for _f, m in sorted(pieces, key=lambda p: p[0])])
    silent = part / f"{new_stem}.silent.mp4"
    song._write_frames_mp4(str(silent), joined, crf)
    piece = part / f"{new_stem}.mp4"
    # the render's own sound, copied over (a render with none gives a piece with none)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(render), "-map", "0:v", "-map", "1:a?", "-c", "copy", str(piece)],
                   check=True)
    silent.unlink()
    if zoom_note is not None:
        song._write_frames_mp4(str(part / f"{new_stem}_zoomed.mp4"), torch.cat([m for _f, m in sorted(zoomed_pieces, key=lambda p: p[0])]), crf)
        beside = render.with_name(render.stem + "_zoomed.json")
        note = json.loads(beside.read_text()) if beside.is_file() else {}
        note.update({"run": new_stem, "piece": f"{new_stem}.mp4", "zoomed": f"{new_stem}_zoomed.mp4", "frames": int(joined.shape[0])})
        (part / f"{new_stem}_zoomed.json").write_text(json.dumps(note, indent=1) + "\n")
    command = [sys.executable, *sys.argv] if a.command is None else a.command
    record = {"what": "a re-composite: the render's saved windows laid again by the node's own lay_window, one setting changed; no sampling",
              "render": str(render), "render_sha256": sha256(render), "set": list(a.set or []), "crf": crf, "windows": rows,
              "source": source, "source_offset_fit": fit, "vae": {"file": vae, "path": str(vae_path) if vae_path else None,
                                                                     "sha256": sha256(Path(vae_path)) if vae_path and Path(vae_path).is_file() else None},
              "frames": int(joined.shape[0]),
              "command": " ".join(shlex.quote(str(x)) for x in command), "by": a.by,
              "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (part / f"{new_stem}.recomposite.json").write_text(json.dumps(record, indent=1) + "\n")
    import sidecar
    sidecar.write(piece, command=[str(x) for x in command], inputs={"render": str(render), "source": source["video"]},
                  extra={"recomposite_of": {"render": str(render), "render_sha256": record["render_sha256"], "set": record["set"],
                                            "record": f"{new_stem}.recomposite.json"}}, made_by=a.by)
    shutil.rmtree(work, ignore_errors=True)
    part.rename(folder)
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--render", required=True, help="the render's piece, R_NNNNN.mp4, with R_windows/ beside it")
    p.add_argument("--source", help="the file the render's loader read: read from its graph; given, it must agree")
    p.add_argument("--source-first", type=int, help="the source frame the render's frame 0 is on: read from its graph; given, it must agree")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a composite setting changed, as recomposite_window takes it")
    p.add_argument("--save-own-matte", action="store_true",
                   help="reconstruct and save each window's matte and kept mask into its _region.npz under the render's own composite, verified against the render")
    p.add_argument("--name", help="the new piece's name (its folder, and <name>_00001.mp4)")
    p.add_argument("--out-dir", help="the folder the new piece's folder is made in")
    p.add_argument("--by", required=True, help="who made it")
    p.add_argument("--vae", help="the video VAE's file name: read from its graph; given, it must agree")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--crf", type=int, help="the video encoder's crf: read from the render's graph; given, it must agree")
    a = p.parse_args(argv)
    a.command = None
    try:
        record = run(a)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if a.save_own_matte:
        out = {k: record[k] for k in ("render", "render_sha256", "composite", "crf", "frames")}
        out["windows"] = [{"window": w["window"], "control": w["control"]} for w in record["windows"]]
    else:
        out = {k: record[k] for k in ("render", "render_sha256", "set", "crf", "frames")}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
