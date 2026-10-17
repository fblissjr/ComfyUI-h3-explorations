#!/usr/bin/env python3
"""Check bench/recomposite_piece.py: a render's saved windows laid again as a new piece, one setting changed, or its own matte saved.

Two renders made for the purpose, each with its decode handed in so no model runs: a plain one window, and a zoomed one of
two windows (the second with its context trimmed), continuing an earlier run, with its windows' piece records, its loader
capped short of its last two frames, and its sound. Cases (each seen red with its code taken out):
  laid by its matte        the new piece's windows hold the node's matte and its stem, the assembler lays it by them, and its
                           companion .png and _with_mask.mp4 are read back by render_ledger, stack_delivery and capture_masked_run
  the render it came from  the record names the render, its latents, its regions and its VAE by sha256, its crf (including a crf
                           wired from a PrimitiveInt node), the setting changed and the settings before, and source_of reads
                           start_time at the clip's own rate; the sidecar says it
  outside the region       before any encode the laid frames outside the region are the source itself (as data in the record,
                           0 unzoomed and sub-half-level zoomed), the encoded plate matches the render's 16-bit source read,
                           and inside it H3's picture whole ("whole region")
  two windows, zoomed      the windows join by their piece records (a continued run's), the trimmed context is not written, the
                           frames past the loader's cap hold its last frame as the node did, the matte is the canvas weight lay_back
                           laid with, the zoomed picture, note, .png and _with_mask.mp4 are written, and the render's sound is copied
  own matte reconstructed  --save-own-matte reconstructs matte, kept and piece into _region.npz under the render's own inputs read
                           from its graph (a graph with no keep-outs reproduces its piece, and the same render reconstructed with
                           today's carve is refused) when the rendered .mp4 agrees within OWN_MATTE_TOLERANCE (recording mean, p99
                           and worst pixel differences inside the matte and outside the region), and refuses without touching
                           _region.npz when they disagree
  what it refuses          an output folder that exists; a render with no graph; a window with no region file; a window of another
                           run; a --source, --source-first, --vae or --crf that is not the render's; an unresolvable crf link;
                           --device cpu with an int8 VAE; a continued run with no piece record; a render whose frames fit the source
                           better one frame off; gapped windows; a CLI refusal exits non-zero with no Python traceback; and a failed
                           run or stale .part decode never blocks or leaks into the rerun

    <comfy venv python> bench/check_recomposite_piece.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import bootstrap, case, finish, needs, refused  # noqa: E402
from _lib.frames import FIT  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))
bootstrap(cpu=True)

import cv2  # noqa: E402
import folder_paths  # noqa: E402
import torch  # noqa: E402
from PIL import Image, PngImagePlugin  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import assemble_delivery as ad  # noqa: E402
import capture_masked_run as cmr  # noqa: E402
import recomposite_piece as rp  # noqa: E402
import recomposite_window as rw  # noqa: E402
import render_ledger as rl  # noqa: E402
import stack_delivery as sd  # noqa: E402

vm = rw._pack("video_mask")
song = rw._pack("audio_freeze_song")
WORK = Path(tempfile.mkdtemp(prefix="check_recomposite_piece_"))
(WORK / "the_render_vae.safetensors").write_bytes(b"h3-test-vae")
folder_paths.add_model_folder_path("vae", str(WORK))
LATENT = (1, 24, 7, 6, 8)                           # seven latent steps on a canvas of six by eight cells: room outside the region
W, H = LATENT[4] * rw.CELL, LATENT[3] * rw.CELL
FRAMES = sum(vm.run_lengths(LATENT[2]))             # a window's frames
LEAD = 3                                            # the load starts on the clip's frame 3
MARGIN, FEATHER = 8, 2
HEAD, CONTEXT, CAP = 4, 6, 36                       # the two-window render: an earlier run's context, the second window's, the loader's cap
BOX = [18, 14, 110, 82]                             # the zoomed render's box on the canvas: 92x68, 1.391x (non-integer ratio so sub-0.5/255 weights exist)
ZOOM = {"box": BOX, "source_size": [W, H], "canvas": [W, H], "scale": W / (BOX[2] - BOX[0]), "render_region": vm.RENDER_ZOOM_REPLACED,
        "set_by": vm.SET_BY_LARGEST}


def lossless(path: Path, frames: np.ndarray, pix: str = "gbrp", rate: str = "24") -> None:
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", rate,
                            "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", pix, str(path)], stdin=subprocess.PIPE)
    assert enc.stdin is not None, "ffmpeg stdin pipe missing"
    enc.stdin.write(frames.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0, f"ffmpeg could not write {path}"


def raw16_frames(path: Path, count: int) -> torch.Tensor:
    """Independent 16-bit rgba64le read for building test fixtures (does not call rp.loader_frames)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", FIT.format(w=W, h=H), "-fps_mode", "passthrough",
                          "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"], capture_output=True, check=True).stdout
    got = np.frombuffer(raw, np.dtype("<u2")).reshape(-1, H, W, 4)[..., :3]
    return torch.from_numpy((got / 65535.0).astype(np.float32))


def held_slice(frames: torch.Tensor, start: int, count: int, have: int | None = None) -> torch.Tensor:
    want = count if have is None else max(min(count, have), 1)
    got = frames[start:start + want].clone()
    held = count - int(got.shape[0])
    return torch.cat([got, got[-1:].repeat(held, 1, 1, 1)], dim=0) if held else got


def soft_clip(n: int, seed: int) -> np.ndarray:
    """A soft picture, as footage is to the codec: a coarse random field a frame, scaled up and blurred."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(60, 180, size=(n, H // 16, W // 16, 3))
    up = np.stack([np.stack([np.kron(f[..., c], np.ones((16, 16))) for c in range(3)], -1) for f in coarse])
    return np.stack([cv2.GaussianBlur(f, (0, 0), 6) for f in up]).round().astype(np.uint8)


# the clip in yuv420p, as footage is: an 8-bit and a 16-bit read of it differ, as they do on the video (the re-composite read, question 8)
lossless(WORK / "clip.mkv", soft_clip(LEAD + HEAD + 40, 20261016), pix="yuv420p")
LOADED = raw16_frames(WORK / "clip.mkv", LEAD + HEAD + 40)                             # independent 16-bit fixture read
CLIP = (LOADED.numpy() * 255.0)


def graph_png(render: Path, start: int, cap: int, continued: str = "", composite: str = "", vae: str = "the_render_vae.safetensors",
              crf: int | list = 12, keep_outs: bool = False, feather_pixels: int = FEATHER) -> None:
    ms_inputs: dict = {"frames": ["28", 0], "feather_pixels": feather_pixels, "change_threshold": vm.CHANGE_THRESHOLD,
                       "grow_pixels": MARGIN, "grow_by": vm.GROW_FIXED, "replace": vm.REPLACE_WHOLE, "edge": vm.EDGE_TOKENS}
    if composite:
        ms_inputs["composite"] = composite
    if keep_outs:
        ms_inputs["others"] = ["210", 0]
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", json.dumps({
        "28": {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": str(WORK / "clip.mkv"), "start_time": start / rw.FPS, "frame_load_cap": cap,
                                                                    "custom_width": W, "custom_height": H}},
        "104": {"class_type": "MiniMaxH3MaskedSource", "inputs": ms_inputs},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "9": {"class_type": "MiniMaxH3AudioFreezeSong", "inputs": {"vae": ["3", 0], "source": ["104", 0], "continue_from": continued, "crf": crf}}}))
    Image.new("RGB", (4, 4)).save(render.with_suffix(".png"), pnginfo=info)


def settings() -> dict:
    return {"composite": vm.COMPOSITE_CHANGED, "feather_pixels": FEATHER, "change_threshold": vm.CHANGE_THRESHOLD, "cuts": None,
            "grow_pixels": MARGIN, "grow_by": vm.GROW_FIXED, "replace": vm.REPLACE_WHOLE, "edge": vm.EDGE_TOKENS}


def window_file(folder: Path, name: str, n: int, trim: int, written: int) -> Path:
    path = folder / f"{name}_window_{n}.safetensors"
    save_file({"stream_0": torch.zeros(LATENT), "stream_1": torch.zeros(1, 2, 2, 4)}, str(path),
              metadata={"key": "k", "trim": str(trim), "next_start": repr(1.0), "nested": "True", "written": str(written)})
    return path


def make_plain(name: str, region: bool = True, start: int = LEAD, continued: str = "", vae: str = "the_render_vae.safetensors") -> dict:
    """One window laid "only what changed", as the song node leaves it, with the graph beside it."""
    root = WORK / name
    folder = root / "r_windows"
    folder.mkdir(parents=True)
    pixels = LOADED[LEAD:LEAD + FRAMES].clone()
    mask = torch.zeros(FRAMES, H, W)
    mask[:, 48:64, 16:40] = 1.0
    tokens = vm.token_mask(vm.grow(mask, MARGIN), *LATENT[2:])
    decode = (pixels + 4.0 / 255.0).clamp(0, 1)     # H3's picture: a few levels off the source everywhere, under the change line,
    decode[:, 50:62, 20:36] = 0.95                  # and the new subject inside the region
    laid, _a, _ = vm.lay_window(decode, pixels, tokens, mask, settings(), MARGIN, 0)
    song._write_frames_mp4(str(root / "r_00001.mp4"), laid, 12)
    graph_png(root / "r_00001.mp4", start, 0, continued=continued, composite=vm.COMPOSITE_CHANGED, vae=vae)
    window_file(folder, "r", 1, 0, FRAMES)
    (folder / "r_window_1.mp4").write_bytes(b"")
    if region:
        vm.save_window_region(str(folder / "r_window_1_region.npz"), mask, tokens, settings(), MARGIN, 0, 0)
    return {"render": root / "r_00001.mp4", "pixels": pixels, "decode": {"r_window_1": decode}, "tokens": tokens}


def make_zoomed(name: str, gap: int = 0) -> dict:
    """Two windows of a zoomed load continuing an earlier run: window 1 holds HEAD frames of its context, window 2 CONTEXT; the loader is
    capped at CAP, past which the node held the last frame; each window's region says where its frames went in the piece."""
    root = WORK / name
    folder = root / "z_windows"
    folder.mkdir(parents=True)
    pieces, zoomed_pieces, decodes = [], [], {}
    z_settings = {**settings(), "feather_pixels": 8}
    for n, (first, trim) in enumerate(((0, HEAD), (FRAMES - CONTEXT, CONTEXT)), 1):
        written = FRAMES - trim
        whole = held_slice(LOADED, LEAD + first, FRAMES, CAP - first)
        zpix = vm.crop_frames(whole, BOX, W, H)
        mask = torch.zeros(FRAMES, H, W)
        mask[:, 42:54, 58:70] = 1.0
        tokens = vm.token_mask(vm.grow(mask, MARGIN), *LATENT[2:])
        decode = (zpix + 4.0 / 255.0).clamp(0, 1)
        decode[:, 24:72, 24:104] = 0.95
        laid, alpha, _ = vm.lay_window(decode, zpix, tokens, mask, z_settings, MARGIN, first)
        piece = vm.lay_back(decode, alpha, whole, ZOOM)
        p0 = first + trim - HEAD + (gap if n == 2 else 0)
        pieces.append(piece[trim:trim + written])
        zoomed_pieces.append(laid[trim:trim + written])
        window_file(folder, "z", n, trim, written)
        (folder / f"z_window_{n}.mp4").write_bytes(b"")
        vm.save_window_region(str(folder / f"z_window_{n}_region.npz"), mask, tokens, {**z_settings, "zoom": ZOOM}, MARGIN, first, trim,
                              piece={"stem": "z_00001", "first": p0, "written": written})
        decodes[f"z_window_{n}"] = decode
    joined = torch.cat(pieces)
    silent = root / "silent.mp4"
    song._write_frames_mp4(str(silent), joined, 12)                 # the node's own writer, as the render's video is
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100", "-shortest",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(root / "z_00001.mp4")], check=True)
    silent.unlink()
    lossless(root / "z_00001_zoomed.mp4", (torch.cat(zoomed_pieces).numpy() * 255).round().astype(np.uint8))
    (root / "z_00001_zoomed.json").write_text(json.dumps({"file": "h3 zoomed piece", "version": 2, "run": "z_00001", "piece": "z_00001.mp4",
                                                          "zoomed": "z_00001_zoomed.mp4", "frames": int(joined.shape[0]), "canvas": [W, H],
                                                          "source_size": [W, H], "box": BOX, "fixed": True, "zoom": ZOOM["scale"]}))
    graph_png(root / "z_00001.mp4", LEAD, CAP, continued="an_earlier_run_window_3.safetensors", composite=vm.COMPOSITE_CHANGED, feather_pixels=8)
    return {"render": root / "z_00001.mp4", "decode": decodes, "frames": int(joined.shape[0])}


def args(made: dict, name: str, **more) -> argparse.Namespace:
    base = dict(render=str(made["render"]), source=None, source_first=None, set=["composite=" + vm.COMPOSITE_REGION],
                save_own_matte=False, allow_rule_change=False, name=name, out_dir=str(WORK / "out"), by="a check",
                vae=None, device="cpu", crf=None, command=["recomposite_piece.py", "--check"])
    return argparse.Namespace(**{**base, **more})


def handed(made: dict, fail: bool = False):
    def decode(path, _vae, _device):
        if fail:
            raise RuntimeError("a decode that fails, as a card out of memory does")
        return made["decode"][Path(path).stem].clone()
    return decode


def frames_of(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MADE = make_plain("render")
NEW = WORK / "out" / "r_rc" / "r_rc_00001.mp4"
ZMADE = make_zoomed("zoomed")
ZNEW = WORK / "out" / "z_rc" / "z_rc_00001.mp4"
_CACHE: dict[str, dict] = {}


def must_run(*a, **kw) -> dict:
    try:
        return rp.run(*a, **kw)
    except SystemExit as exc:
        raise AssertionError(f"unexpected refusal: {exc}") from exc


def plain_rec() -> dict:
    if "r_rc" not in _CACHE:
        shutil.rmtree(WORK / "out" / "r_rc", ignore_errors=True)
        _CACHE["r_rc"] = must_run(args(MADE, "r_rc"), decode=handed(MADE))
    return _CACHE["r_rc"]


def zoomed_rec() -> dict:
    if "z_rc" not in _CACHE:
        shutil.rmtree(WORK / "out" / "z_rc", ignore_errors=True)
        _CACHE["z_rc"] = must_run(args(ZMADE, "z_rc"), decode=handed(ZMADE))
    return _CACHE["z_rc"]


def laid_by_its_matte() -> str:
    plain_rec()
    said = ad.laid_by_of(str(NEW))
    matte = ad.Matte(str(NEW))
    # THE CASE THAT GOES RED when the new piece's windows hold no matte of its own stem, and the assembler rebuilds what it lays
    assert said["rule"] == "matte" and not matte.gaps and all(matte.at(k) is not None for k in range(FRAMES)), (
        f"expected matte rule with no gaps on all {FRAMES} frames: {(said, matte.gaps)}"
    )
    reg = vm.load_window_region(str(WORK / "out" / "r_rc" / "r_rc_windows" / "r_rc_window_1_region.npz"))
    assert reg.get("kept") is not None and reg.get("matte") is not None, f"saved window region must hold kept and matte: {list(reg)}"
    with np.load(WORK / "out" / "r_rc" / "r_rc_windows" / "r_rc_window_1_region.npz") as z:
        meta = json.loads(str(z["meta"]))
    assert meta["piece"] == {"stem": "r_rc_00001", "first": 0, "written": FRAMES} and meta["source"]["composite"] == vm.COMPOSITE_REGION, (
        f"saved window region meta must record new stem r_rc_00001 and composite now: {meta}"
    )
    # THE CASE THAT GOES RED when the re-laid piece does not write its companion .png and _with_mask.mp4 for render_ledger, stack_delivery and capture_masked_run
    assert NEW.with_suffix(".png").is_file() and NEW.with_name("r_rc_00001_with_mask.mp4").is_file(), (
        "re-laid piece must write r_rc_00001.png and r_rc_00001_with_mask.mp4 beside r_rc_00001.mp4"
    )
    assert (WORK / "out" / "r_rc" / "r_rc_windows" / "r_rc_window_1_with_mask.mp4").is_file(), (
        "re-laid piece must write r_rc_window_1_with_mask.mp4 in r_rc_windows/"
    )
    ledger_asked = rl.asked(rl.saved_graph(NEW), {"clip.mkv": 24.0})
    assert (ledger_asked is not None and ledger_asked["of"] == "clip.mkv"
            and ledger_asked["first"] == LEAD and ledger_asked["masked"] is True and ledger_asked["margin"] == MARGIN), (
        f"render_ledger.asked on re-laid .png must see source clip.mkv, first={LEAD}, masked=True: {ledger_asked}"
    )
    # THE CASE THAT GOES RED when the re-laid piece's .png carries no relaid block in its extra info
    with Image.open(NEW.with_suffix(".png")) as im:
        png_relaid_raw = im.info.get("relaid")
    assert png_relaid_raw is not None, "re-laid .png must carry relaid block in its PNG info"
    png_relaid = json.loads(png_relaid_raw)
    assert (png_relaid.get("from") == str(MADE["render"]) and png_relaid.get("sha256") == sha(MADE["render"])
            and png_relaid.get("by") == "a check" and png_relaid.get("changed") == {"composite": vm.COMPOSITE_REGION}
            and rl.saved_relaid(NEW) == png_relaid), (
        f"re-laid .png and render_ledger.saved_relaid must carry from, sha256, by, at and changed: {png_relaid}"
    )
    assert sd.graph_reads(NEW, 24.0) == sd.graph_reads(MADE["render"], 24.0), (
        f"stack_delivery.graph_reads on re-laid piece must match render's source and frame: {sd.graph_reads(NEW, 24.0)}"
    )
    assert cmr.source_settings(cmr.graph_of(str(NEW)))["composite"] == vm.COMPOSITE_REGION, (
        "capture_masked_run.source_settings on re-laid piece must see updated composite"
    )
    saved_reg = cmr.read_saved_regions(cmr.window_region_files(str(NEW)), LEAD, FRAMES, (W, H), None, FRAMES)
    rev_reg = cmr.read_region(str(NEW), str(WORK / "clip.mkv"), LEAD, FRAMES, (W, H), None)
    agree = cmr.readers_agreement(saved_reg[:3], rev_reg[:3])
    assert agree["frames_compared"] == FRAMES and agree["cells_differing"] == 0 and bool(rev_reg[1].any()), (
        f"capture_masked_run.read_region on re-laid _with_mask.mp4 must agree with saved region on all {FRAMES} frames: {agree}"
    )
    assert rp.source_of(rp.graph_of(NEW))["composite"] == vm.COMPOSITE_REGION and rp.crf_of(rp.graph_of(NEW)) == 12, (
        "recomposite_piece.source_of and crf_of on re-laid .png must read updated composite and crf=12"
    )
    return f"laid by its matte on all {FRAMES} frames; region holds kept, matte and new stem; .png (with relaid) and _with_mask.mp4 read back by ledger, stack and capture"


def the_render_it_came_from() -> str:
    rec = plain_rec()
    # THE CASE THAT GOES RED when the record does not name the render, its latents, its regions, its crf, its relaid block or its VAE file on disk by sha256
    assert (rec["render_sha256"] == sha(MADE["render"]) and rec["set"] == ["composite=" + vm.COMPOSITE_REGION] and rec["crf"] == 12
            and rec["png"] == "r_rc_00001.png" and rec["with_mask"] == "r_rc_00001_with_mask.mp4"
            and rec.get("relaid") == {"from": str(MADE["render"]), "sha256": sha(MADE["render"]),
                                      "by": "a check", "at": rec["at"], "changed": {"composite": vm.COMPOSITE_REGION}}), (
        f"record must name render_sha256, set, relaid, crf=12, png and with_mask: {rec}"
    )
    w = rec["windows"][0]
    folder = MADE["render"].parent / "r_windows"
    assert w["latent_sha256"] == sha(folder / "r_window_1.safetensors") and w["region_sha256"] == sha(folder / "r_window_1_region.npz"), (
        f"window row must hold latent_sha256 and region_sha256: {w}"
    )
    assert rec["vae"]["file"] == "the_render_vae.safetensors" and rec["vae"]["sha256"] == sha(WORK / "the_render_vae.safetensors"), (
        f"record vae must hold file and sha256: {rec['vae']}"
    )
    assert w["composite_before"] == vm.COMPOSITE_CHANGED and w["composite_now"] == vm.COMPOSITE_REGION, (
        f"window row must hold composite_before and composite_now: {w}"
    )
    side_path = NEW.parent / (NEW.name + ".json")
    assert side_path.is_file(), f"sidecar {side_path} not written"
    side = json.loads(side_path.read_text())
    assert side.get("extra", {}).get("recomposite_of", {}).get("render_sha256") == rec["render_sha256"], (
        f"sidecar extra.recomposite_of must hold render_sha256: {list(side)}"
    )
    # THE CASE THAT GOES RED when crf_of does not resolve a crf wired from a constant node via h3_config.resolve_widget
    try:
        prim_crf = rp.crf_of({"9": {"class_type": "MiniMaxH3AudioFreezeSong", "inputs": {"crf": ["99", 0]}},
                              "99": {"class_type": "PrimitiveInt", "inputs": {"value": 14}}})
    except SystemExit as exc:
        raise AssertionError(f"crf_of must resolve a crf wired from PrimitiveInt via h3_config.resolve_widget: {exc}") from exc
    assert prim_crf == 14, f"crf_of must resolve a crf wired from PrimitiveInt to 14: {prim_crf}"
    # THE CASE THAT GOES RED when source_of multiplies start_time by a hardcoded 24 instead of the clip's own frame rate
    ntsc = WORK / "ntsc.mkv"
    lossless(ntsc, np.zeros((2, H, W, 3), np.uint8), rate="24000/1001")
    ntsc_src = rp.source_of({
        "28": {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": str(ntsc), "start_time": 1000 * 1001 / 24000,
                                                                    "frame_load_cap": 0, "custom_width": W, "custom_height": H}},
        "104": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"frames": ["28", 0]}}})
    assert ntsc_src["first"] == 1000, f"source_of must compute first frame from clip's own frame rate: {ntsc_src}"
    return "the record names the render, its latent, its region, its crf (literal or PrimitiveInt) and the render's VAE by sha256; source_of uses the clip's rate; the sidecar names the render"


def outside_the_region() -> str:
    rec, zrec = plain_rec(), zoomed_rec()
    # THE CASE THAT GOES RED when the laid frames outside the region are not the source before any encode, or zoomed outside_levels is hardcoded 0.0
    unzoomed = rec["windows"][0]["outside_the_region_before_encoding_levels_most"]
    zoomed = [w["outside_the_region_before_encoding_levels_most"] for w in zrec["windows"]]
    assert unzoomed == 0.0 and all(z is not None and 0.0 < z <= 0.5 for z in zoomed), (
        f"outside_the_region_before_encoding_levels_most must be 0.0 unzoomed and (0, 0.5] zoomed: {(unzoomed, zoomed)}"
    )
    got = frames_of(NEW)
    song._write_frames_mp4(str(WORK / "src_ref.mp4"), MADE["pixels"], 12)
    song._write_frames_mp4(str(WORK / "dec_ref.mp4"), MADE["decode"]["r_window_1"], 12)
    src, dec = frames_of(WORK / "src_ref.mp4"), frames_of(WORK / "dec_ref.mp4")
    region = vm.pixel_alpha(MADE["tokens"], H, W, 0).numpy() > 0.5
    grown = np.ones((2 * (FEATHER + 16) + 1,) * 2, np.uint8)
    outside = ~np.stack([cv2.dilate(r.astype(np.uint8), grown).astype(bool) for r in region])
    # THE CASE THAT GOES RED when the unzoomed source is read at 8 bits and the re-laid piece is darker than the render it replaces
    plate_signed = float((got - frames_of(MADE["render"]))[outside].mean())
    assert abs(plate_signed) < 0.6, f"the new unzoomed piece's plate is {plate_signed:+.2f} levels from the render's own outside the region"
    encoded = float(np.abs(got - src)[outside].mean())         # data: two encodes of frames that differ only inside the region
    band = region & ~np.isin(np.arange(W), range(20, 36))[None, None, :]
    near_dec, near_src = float(np.abs(got - dec)[band].mean()), float(np.abs(got - src)[band].mean())
    # "whole region": H3's picture inside the region, its barely-changed pixels too, where "only what changed" put the source back
    assert near_dec + 1.0 < near_src, f"whole region composite must follow H3 decode inside region: near_dec={near_dec}, near_src={near_src}"
    return (f"before encoding, {unzoomed} levels outside the region unzoomed and {max(zoomed)} at most zoomed; plate {plate_signed:+.2f} "
            f"from render; after encoding {encoded:.2f} (data); inside it {near_dec:.2f} from H3's picture against {near_src:.2f} from the source")


def two_windows_zoomed() -> str:
    zrec = zoomed_rec()
    # THE CASE THAT GOES RED when offset_fit drops + trim on a window with trimmed context
    assert zrec["source_offset_fit"]["levels_by_frame_offset"][0] < 5.0, (
        f"offset_fit at offset 0 must be < 5.0 levels on trimmed window: {zrec['source_offset_fit']}"
    )
    got = frames_of(ZNEW)
    # THE CASE THAT GOES RED when the windows do not join by their piece records, or the trimmed context is written
    assert int(got.shape[0]) == ZMADE["frames"] == 2 * FRAMES - HEAD - CONTEXT, (
        f"zoomed piece frame count must equal 2*FRAMES - HEAD - CONTEXT: {(got.shape, ZMADE['frames'])}"
    )
    regions = [vm.load_window_region(str(WORK / "out" / "z_rc" / "z_rc_windows" / f"z_rc_window_{n}_region.npz")) for n in (1, 2)]
    with np.load(WORK / "out" / "z_rc" / "z_rc_windows" / "z_rc_window_2_region.npz") as z:
        assert json.loads(str(z["meta"]))["piece"] == {"stem": "z_rc_00001", "first": FRAMES - HEAD, "written": FRAMES - CONTEXT}, (
            "window 2 region meta must record stem z_rc_00001, first FRAMES-HEAD, written FRAMES-CONTEXT"
        )
    # a corner outside the box: the source's own frame there (the context trimmed, the windows joined where their records say)
    corner = (slice(None), slice(0, 10), slice(0, 14))
    want = [min(LEAD + HEAD + q, LEAD + CAP - 1) for q in range(ZMADE["frames"])]   # past the loader's cap, its last frame held
    ref = np.stack([CLIP[f] for f in want]).astype(np.float32)
    # THE SOURCE AS THE LOADER READ IT (question 8): the new piece's plate against the render's own, signed, outside the box; an 8-bit read
    # of the source is a level and more darker than the 16-bit frames the node was handed
    plate = frames_of(ZMADE["render"])
    signed = float((got - plate)[corner].mean())
    # THE CASE THAT GOES RED when the source is read at 8 bits and the re-laid piece is darker than the render it replaces
    assert abs(signed) < 0.6, f"the new piece's plate is {signed:+.2f} levels from the render's own outside the box"
    per = np.abs(got - ref)[corner].mean(axis=(1, 2, 3))
    # THE CASE THAT GOES RED when the frames past the loader's cap are the clip's next frames, not its last held, or a window lands off its frames
    assert float(per.max()) < 3.0, f"corner frames past loader cap must hold last frame within 3.0 levels: {[round(float(x), 2) for x in per]}"
    m1 = regions[0].get("matte")
    assert m1 is not None, f"zoomed window 1 region must hold matte: {list(regions[0])}"
    inside = np.zeros((H, W), bool)
    inside[BOX[1]:BOX[3], BOX[0]:BOX[2]] = True
    # THE CASE THAT GOES RED when a zoomed window's matte is its weight in the zoomed space, not the canvas weight lay_back laid with
    assert float(m1[:, ~inside].max()) == 0.0 and float(m1[:, inside].max()) > 0.5, (
        f"zoomed window matte must be 0 outside box and >0.5 inside box: {(float(m1[:, ~inside].max()), float(m1[:, inside].max()))}"
    )
    z_note = WORK / "out" / "z_rc" / "z_rc_00001_zoomed.json"
    assert (WORK / "out" / "z_rc" / "z_rc_00001_zoomed.mp4").is_file() and z_note.is_file(), f"zoomed video and {z_note} must be written"
    note = json.loads(z_note.read_text())
    assert note["run"] == "z_rc_00001" and note["box"] == BOX, f"zoomed note must hold run z_rc_00001 and box {BOX}: {note}"

    def audio(path: Path) -> str:
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True).stdout
    # THE CASE THAT GOES RED when the render's sound is not copied
    assert audio(ZNEW) and audio(ZNEW) == audio(ZMADE["render"]), (
        f"zoomed piece audio must match render audio: {(audio(ZNEW), audio(ZMADE['render']))}"
    )
    # THE CASE THAT GOES RED when zoomed companions (.png and _with_mask.mp4) are not written with sound and readable by ledger, stack and capture
    z_mask_mp4 = ZNEW.with_name("z_rc_00001_with_mask.mp4")
    assert (ZNEW.with_suffix(".png").is_file() and z_mask_mp4.is_file()
            and all((WORK / "out" / "z_rc" / "z_rc_windows" / f"z_rc_window_{n}_with_mask.mp4").is_file() for n in (1, 2))), (
        "zoomed re-laid piece must write z_rc_00001.png, z_rc_00001_with_mask.mp4 and window _with_mask.mp4 files"
    )
    assert audio(z_mask_mp4) == audio(ZMADE["render"]), "zoomed _with_mask.mp4 must carry the render's audio"
    assert cmr.probe(str(z_mask_mp4)) == (W, 2 * H, ZMADE["frames"]), (
        f"zoomed _with_mask.mp4 must be {W}x{2 * H} across {ZMADE['frames']} frames: {cmr.probe(str(z_mask_mp4))}"
    )
    assert len(cmr.window_region_files(str(ZNEW))) == 2, "capture_masked_run.window_region_files must find both zoomed re-laid region files"
    assert (rl.asked(rl.saved_graph(ZNEW), {"clip.mkv": 24.0})["of"] == "clip.mkv"
            and cmr.source_settings(cmr.graph_of(str(ZNEW)))["composite"] == vm.COMPOSITE_REGION
            and sd.graph_reads(ZNEW, 24.0) == sd.graph_reads(ZMADE["render"], 24.0)), (
        "render_ledger, capture_masked_run and stack_delivery must read zoomed re-laid .png"
    )
    held = zrec["windows"][1]["source_frames_held"]
    assert held == FRAMES - (CAP - (FRAMES - CONTEXT)), f"window 2 source_frames_held must equal frames past loader cap: {zrec['windows'][1]}"
    return (f"{ZMADE['frames']} frames joined by the records, the context trimmed, {held} frames held past the cap, the matte the canvas weight, "
            "the zoomed files, .png, _with_mask.mp4 and the sound")


def own_matte_reconstructed() -> str:
    import contextlib
    import io
    own = make_plain("own_plain")
    zown = make_zoomed("own_zoomed")
    # Before reconstruction, neither render's windows hold a saved matte
    assert ad.laid_by_of(str(own["render"]))["rule"] != "matte" and ad.laid_by_of(str(zown["render"]))["rule"] != "matte", (
        "fixture renders must not hold a saved matte before --save-own-matte"
    )
    own_reg_path = own["render"].parent / "r_windows" / "r_window_1_region.npz"
    own_reg_before = own_reg_path.read_bytes()
    # THE CASE THAT GOES RED when --save-own-matte changes a piece's laid_by rule without --allow-rule-change (C61 item 2b)
    refused(lambda: rp.run(args(own, "", save_own_matte=True, allow_rule_change=False, set=[]), decode=handed(own)),
            "changes laid_by from 'changed' to 'matte'", "--allow-rule-change")
    assert own_reg_path.read_bytes() == own_reg_before and not (own["render"].parent / "r_00001.matte_reconstructed.json").exists(), (
        "refused rule change must leave _region.npz untouched and write no .matte_reconstructed.json"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        must_run(args(own, "", save_own_matte=True, allow_rule_change=True, set=[]), decode=handed(own))
    assert "r_00001.mp4: laid_by changed -> matte" in buf.getvalue(), (
        f"--save-own-matte must print laid_by before -> after: {buf.getvalue()!r}"
    )
    zrec = must_run(args(zown, "", save_own_matte=True, allow_rule_change=True, set=[]), decode=handed(zown))
    # THE CASE THAT GOES RED when --save-own-matte does not write matte, kept and piece so assemble_delivery.Matte lays the render by them
    for m, n_frames in ((own, FRAMES), (zown, zown["frames"])):
        said = ad.laid_by_of(str(m["render"]))
        matte = ad.Matte(str(m["render"]))
        assert said["rule"] == "matte" and not matte.gaps and all(matte.at(k) is not None for k in range(n_frames)), (
            f"after --save-own-matte, render must be laid by matte with no gaps: {(said, matte.gaps)}"
        )
    reg1 = vm.load_window_region(str(own_reg_path))
    assert reg1.get("kept") is not None and reg1.get("matte") is not None, (
        f"after --save-own-matte, region file must hold kept and matte: {list(reg1)}"
    )
    rec_file = own["render"].parent / "r_00001.matte_reconstructed.json"
    assert rec_file.is_file(), f"reconstructed matte record {rec_file} not written"
    saved = json.loads(rec_file.read_text())
    w0 = saved["windows"][0]
    c0 = w0["control"]
    assert (saved["render_sha256"] == sha(own["render"]) and saved["vae"]["sha256"] == sha(WORK / "the_render_vae.safetensors")
            and saved.get("laid_by_before") == "changed" and saved.get("laid_by_after") == "matte"
            and saved["crf"] == 12 and w0["region_sha256_before"] != w0["region_sha256_after"]
            and 0.0 < c0["inside_matte_mean_levels"] <= rp.OWN_MATTE_TOLERANCE
            and 0.0 < c0["outside_matte_mean_levels"] <= rp.OWN_MATTE_TOLERANCE
            and c0["inside_matte_mean_levels"] < c0["inside_matte_p99_levels"] <= c0["inside_matte_worst_levels"]
            and c0["outside_matte_mean_levels"] < c0["outside_matte_p99_levels"] <= c0["outside_matte_worst_levels"]), (
        f"reconstructed record must hold sha256s, laid_by_before/after, crf=12, and 0 < mean < p99 <= worst inside/outside: {saved}"
    )
    # Re-running --save-own-matte when laid_by is already 'matte' -> 'matte' succeeds without --allow-rule-change
    buf_same = io.StringIO()
    with contextlib.redirect_stdout(buf_same):
        same_rec = must_run(args(own, "", save_own_matte=True, allow_rule_change=False, set=[]), decode=handed(own))
    assert "r_00001.mp4: laid_by matte -> matte" in buf_same.getvalue() and same_rec["laid_by_before"] == "matte" and same_rec["laid_by_after"] == "matte", (
        f"re-running --save-own-matte when already matte must succeed without --allow-rule-change: {buf_same.getvalue()!r}"
    )
    assert len(zrec["windows"]) == 2 and all(
        0.0 < w["control"]["inside_matte_mean_levels"] <= rp.OWN_MATTE_TOLERANCE
        and w["control"]["inside_matte_mean_levels"] < w["control"]["inside_matte_p99_levels"] <= w["control"]["inside_matte_worst_levels"]
        and w["control"]["outside_matte_mean_levels"] < w["control"]["outside_matte_p99_levels"] <= w["control"]["outside_matte_worst_levels"]
        for w in zrec["windows"]
    ), "zoomed reconstructed record must hold 2 windows with 0 < mean < p99 <= worst inside/outside"
    # THE CASE THAT GOES RED when --save-own-matte uses today's keep-outs carve instead of the render's own graph inputs (C37):
    # a render made without keep-outs reproduces its piece when its graph has no keep-outs input, and is refused when reconstructed with today's carve
    no_ko = make_plain("own_no_keep_outs")
    no_ko_mask = torch.zeros(FRAMES, H, W)
    no_ko_mask[:, 48:64, 16:40] = 1.0
    no_ko_decode = (no_ko["pixels"] + 4.0 / 255.0).clamp(0, 1)
    no_ko_decode[:, 48:64, 16:48] = 0.95                            # subject reaches into the grown margin (x=32..48)
    no_ko["decode"]["r_window_1"] = no_ko_decode
    no_ko_laid, _a, _ = vm.lay_window(no_ko_decode, no_ko["pixels"], no_ko["tokens"], no_ko_mask, settings(), MARGIN, 0)
    song._write_frames_mp4(str(no_ko["render"]), no_ko_laid, 12)
    carved_tokens = no_ko["tokens"].clone()
    carved_tokens[:, :, 2:4] = 0.0                                  # today's keep-outs carve out columns 2:4 (x=32..64)
    no_ko_reg = no_ko["render"].parent / "r_windows" / "r_window_1_region.npz"
    vm.save_window_region(str(no_ko_reg), no_ko_mask, carved_tokens, {**settings(), "composite": vm.COMPOSITE_REGION}, MARGIN, 0, 0)
    graph_png(no_ko["render"], LEAD, 0, composite=vm.COMPOSITE_CHANGED, keep_outs=True)
    carved_bytes = no_ko_reg.read_bytes()
    refused(lambda: rp.run(args(no_ko, "", save_own_matte=True, allow_rule_change=True, set=[]), decode=handed(no_ko)),
            "own-composite control disagrees", "not modified")
    assert no_ko_reg.read_bytes() == carved_bytes and not (no_ko["render"].parent / "r_00001.matte_reconstructed.json").exists(), (
        "refused carve must leave _region.npz untouched and write no .matte_reconstructed.json"
    )
    graph_png(no_ko["render"], LEAD, 0, composite=vm.COMPOSITE_CHANGED, keep_outs=False)
    no_ko_rec = must_run(args(no_ko, "", save_own_matte=True, allow_rule_change=True, set=[]), decode=handed(no_ko))
    reloaded = vm.load_window_region(str(no_ko_reg))
    assert (no_ko_rec["keep_outs"] is False
            and 0.0 < no_ko_rec["windows"][0]["control"]["inside_matte_mean_levels"] <= rp.OWN_MATTE_TOLERANCE
            and torch.equal(reloaded["tokens"], no_ko["tokens"])
            and reloaded["source"]["composite"] == vm.COMPOSITE_CHANGED), (
        "saved region tokens must be uncarved and record must match graph composite and keep_outs=False"
    )
    # THE CASE THAT GOES RED when --save-own-matte writes a matte whose own-composite control disagrees with the rendered .mp4
    bad = make_plain("own_bad")
    bad_reg = bad["render"].parent / "r_windows" / "r_window_1_region.npz"
    before_bytes = bad_reg.read_bytes()
    wrong_decode = {"r_window_1": bad["pixels"].clone()}            # differs inside the matte from the 0.95 subject in bad["render"]
    refused(lambda: rp.run(args(bad, "", save_own_matte=True, allow_rule_change=True, set=[]), decode=lambda _p, _v, _d: wrong_decode["r_window_1"]),
            "own-composite control disagrees", "not modified")
    assert bad_reg.read_bytes() == before_bytes and not (bad["render"].parent / "r_00001.matte_reconstructed.json").exists(), (
        "disagreeing control must leave _region.npz untouched and write no .matte_reconstructed.json"
    )
    return ("plain and zoomed renders gain matte, kept and piece with laid_by before -> after printed and recorded, "
            "a rule change without --allow-rule-change is refused, a graph with no keep-outs reproduces its piece and "
            "the same render with today's carve is refused, and a disagreeing control refuses without touching _region.npz")


def what_it_refuses() -> str:
    plain_rec()
    # Direct self-checks on bench/_lib/__init__.py::refused
    def must_reject(fn) -> None:
        try:
            fn()
        except AssertionError:
            return
        raise AssertionError("refused() accepted a non-refusal or crash")

    def _raise(exc: BaseException) -> None:
        raise exc

    assert refused(lambda: _raise(ValueError("bad value")), "bad value") == "bad value", "refused() must return matching refusal words"
    for bad_fn in (
        lambda: refused(lambda: None, "x"),
        lambda: refused(lambda: _raise(SystemExit()), "None"),
        lambda: refused(lambda: _raise(SystemExit(0)), "0"),
        lambda: refused(lambda: _raise(SystemExit(1)), "1"),
        lambda: refused(lambda: _raise(KeyError("boom")), "boom"),
        lambda: refused(lambda: _raise(SystemExit("Traceback (most recent call last):\nboom")), "boom"),
        lambda: refused(lambda: _raise(SystemExit("other words")), "missing"),
        lambda: refused(subprocess.CompletedProcess([], 0, "ok", ""), "ok"),
        lambda: refused(subprocess.CompletedProcess([], 1, "", "")),
        lambda: refused(subprocess.CompletedProcess([], 1, "", "Traceback (most recent call last):\nboom"), "boom"),
        lambda: refused(subprocess.CompletedProcess([], 1, "", "other words"), "missing"),
    ):
        must_reject(bad_fn)

    refused(lambda: rp.run(args(MADE, "r_rc"), decode=handed(MADE)), "exists")
    bare = make_plain("bare", region=False)
    refused(lambda: rp.run(args(bare, "bare_rc"), decode=handed(bare)), "r_window_1_region.npz is not beside")
    nograph = make_plain("nograph")
    nograph["render"].with_suffix(".png").unlink()
    refused(lambda: rp.run(args(nograph, "nograph_rc"), decode=handed(nograph)), "is not beside the render", "never typed")
    other = make_plain("other")
    folder = other["render"].parent / "r_windows"
    vm.save_window_region(str(folder / "r_window_1_region.npz"), torch.zeros(FRAMES, H, W), other["tokens"], settings(), MARGIN, 0, 0,
                          piece={"stem": "r_00002", "first": 0, "written": FRAMES})
    # THE CASE THAT GOES RED when another run's window, of the same prefix, is laid under this render's sha
    refused(lambda: rp.run(args(other, "other_rc"), decode=handed(other)), "is r_00002's window, not r_00001's")
    # THE CASES THAT GO RED when --source, --source-first, --vae or --crf disagrees with the render's graph, or crf is a link
    refused(lambda: rp.run(args(MADE, "src_rc", source=str(WORK / "other.mkv")), decode=handed(MADE)), "--source", "is not the render's own")
    refused(lambda: rp.run(args(MADE, "typed_rc", source_first=LEAD + 1), decode=handed(MADE)), "--source-first", "is not the render's own")
    refused(lambda: rp.run(args(MADE, "vae_rc", vae="wrong_vae.safetensors"), decode=handed(MADE)), "--vae", "is not the render's own")
    refused(lambda: rp.run(args(MADE, "crf_rc", crf=18), decode=handed(MADE)), "--crf", "is not the render's own")
    linked_crf = make_plain("linked_crf")
    graph_png(linked_crf["render"], LEAD, 0, composite=vm.COMPOSITE_CHANGED, crf=["99", 0])
    refused(lambda: rp.run(args(linked_crf, "linked_crf_rc"), decode=handed(linked_crf)), "literal crf")
    # THE CASE THAT GOES RED when --device cpu is used with an int8 VAE (which decodes on CUDA only), or --device does not default to cuda
    int8_render = make_plain("int8_vae", vae="h3_video_vae_bf16_int8.safetensors")
    refused(lambda: rp.run(args(int8_render, "int8_rc", device="cpu"), decode=handed(int8_render)), "int8", "CUDA only")
    refused([sys.executable, str(Path(rp.__file__)), "--render", str(int8_render["render"]), "--name", "int8_cli_rc", "--out-dir", str(WORK / "out"),
             "--by", "a check"], "h3_video_vae_bf16_int8.safetensors", "not in the models' vae folder")
    # THE CASE THAT GOES RED when a continued run has no piece record in its region file
    cont = make_plain("cont_nopiece", continued="earlier_run_window_1.safetensors")
    refused(lambda: rp.run(args(cont, "cont_rc"), decode=handed(cont)), "continued earlier_run_window_1.safetensors", "first window's context is not known")
    off = make_plain("off", start=LEAD + 1)
    # THE CASE THAT GOES RED when a render is read against its source one frame off
    refused(lambda: rp.run(args(off, "off_rc"), decode=handed(off)), "fit the source better one frame off")
    gapped = make_zoomed("gapped", gap=2)
    # THE CASE THAT GOES RED when windows whose records leave a gap are joined into a piece
    refused(lambda: rp.run(args(gapped, "gapped_rc"), decode=handed(gapped)), "do not join")
    # THE CASE THAT GOES RED when the CLI prints a raw Python traceback on a refusal
    refused([sys.executable, str(Path(rp.__file__)), "--render", str(MADE["render"]), "--name", "cli_rc", "--out-dir", str(WORK / "out"),
             "--by", "a check", "--vae", "wrong_vae.safetensors"], "--vae", "is not the render's own")
    refused([sys.executable, str(Path(rp.__file__)), "--render", str(MADE["render"]), "--name", "cli_val_rc", "--out-dir", str(WORK / "out"),
             "--by", "a check", "--set", "feather_pixels=notanint"], "notanint")
    # THE CASE THAT GOES RED when a failed run leaves a folder that blocks the rerun, or a stale decode in .part/work is reused
    stale_work = WORK / "out" / "again_rc.part" / "work"
    stale_work.mkdir(parents=True)
    np.save(stale_work / "decoded_r_window_1.npy", np.zeros((FRAMES, H, W, 3), np.float16))
    try:
        rp.run(args(MADE, "again_rc"), decode=handed(MADE, fail=True))
    except RuntimeError:
        pass
    assert not (WORK / "out" / "again_rc").exists(), "a failed run left the folder that blocks the rerun"
    stale_work.mkdir(parents=True, exist_ok=True)
    np.save(stale_work / "decoded_r_window_1.npy", np.zeros((FRAMES, H, W, 3), np.float16))
    again_rec = must_run(args(MADE, "again_rc"), decode=handed(MADE))
    assert "decoded on " in again_rec["windows"][0]["decode"], f"stale .part decode was reused: {again_rec['windows'][0]}"
    assert (WORK / "out" / "again_rc" / "again_rc_00001.mp4").is_file() and not (WORK / "out" / "again_rc.part").exists(), (
        "rerun must write again_rc_00001.mp4 and remove again_rc.part"
    )
    return ("an existing folder, no graph, no region, another run's window, disagreed --source/--source-first/--vae/--crf, int8 VAE on cpu, "
            "a continued run with no piece record, a source one frame off, gapped windows, and a CLI refusal without traceback, each refused; stale .part never reused")


case("laid by its matte", laid_by_its_matte)
case("the render it came from", the_render_it_came_from)
case("the source outside the region", outside_the_region)
case("two windows, zoomed, continued, capped, with sound", two_windows_zoomed)
case("own matte reconstructed", own_matte_reconstructed)
case("what it refuses", what_it_refuses)

if __name__ == "__main__":
    sys.exit(finish())
