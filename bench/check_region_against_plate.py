#!/usr/bin/env python3
"""`bench/region_against_plate.py` against a clip and pieces whose answers are known by construction.

The tool's figures are used to choose a post step and to refute a guess about why a render does not sit in its
plate. A measure that cannot be wrong measures nothing, so this builds a textured clip, makes pieces from it with
the lane's own writer, changes one thing in each by a known amount, and reads the tool's answer.

1. **Grain is read at its size, and is not read as detail.** White noise of a known standard deviation, new on
   every frame, added to the piece's luma (the tool's own `--control-noise`): the finest band's grain comes back
   as the root of the piece's own squared plus (0.872 x SD) squared, and its detail does not move.
2. **A softer region is read as softer, by the amount the trial blurs predict.** Two pieces, the region of one
   blurred by a known sigma: its fine-over-mid ratio is lower, and the unblurred piece's own trial row for that
   sigma lands on it in the mid-over-coarse ratio (the finest band of a blurred picture holds little but the
   rounding of the file's 8 bits). The pieces are written without loss for this: an encoder puts fine detail of its
   own back on a blurred picture, which the trial row, blurring what was decoded, does not have.
3. **Tone and cast are read where they were put.** A region lifted by a known amount reads that much brighter
   than what stood there; one with blue added reads the shift in its chroma offset and the other does not.
4. **The region is the region.** Every figure above is taken inside the rectangle that was changed: the pixels the
   tool counts as `new` are a share of the rectangle's own, and the plate band lies outside it.

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_region_against_plate.py

Imports the pack's writer, so ComfyUI core must be importable; no card and no server.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)        # the writer is imported for its ffmpeg call; nothing here uses a device
import torch  # noqa: E402

import assemble_delivery as assembler  # noqa: E402
import region_against_plate as tool  # noqa: E402

W, H = 256, 192
SW, SH = 400, 304
FRAMES = 24
RECT = (60, 40, 190, 150)        # x0, y0, x1, y1 of the region every piece changes
LIFT = 0.16                      # of full scale: far over the assembler's change threshold on any picture
BLUR = 1.0                       # sigma, pixels, of the softened piece: one of the tool's own trial blurs
NOISE = 3.0                      # levels: the control noise
CRF = 0                          # the pieces are written without loss, so that what is read is the change that was
                                 # made and not the encoder's own fine detail on top of it: at crf 10 on this picture
                                 # a region blurred before the encode read a ratio 29% above its trial row
                                 # (measured 2026-10-10); on soft footage the two agreed within 3%
# A picture that barely drifts, with texture at several scales: nearly still, so every pixel counts for grain;
# textured, so there is detail to soften. The colours and line are named because ffmpeg's `gradients` draws them at
# random otherwise; the noise is laid on yuv planes (on the filter's own rgb it swamps the picture) and has no
# temporal flag, so it is the same on every frame and reads as detail.
PICTURE = ("gradients=size={w}x{h}:rate=24000/1001:speed=0.001:nb_colors=4:c0=0x80563a:c1=0x3a5f8c:c2=0xb4a05a:c3=0x4a6a3c"
           ":x0=30:y0=20:x1=370:y1=280:seed=3,format=yuv444p,noise=alls=22:all_seed=11,gblur=sigma=0.7")


def _song():
    """The pack's song node as a module of a stand-in package; the repo root stays off `sys.path` (its `nodes.py`
    would shadow core's)."""
    pkg = sys.modules.get("_h3pack")
    if pkg is None:
        pkg = types.ModuleType("_h3pack")
        pkg.__path__ = [str(REPO)]
        sys.modules["_h3pack"] = pkg
    return importlib.import_module("_h3pack.audio_freeze_song")


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, **kw)
    assert proc.returncode == 0, f"{' '.join(str(c) for c in cmd[:6])} ...: {proc.stderr.decode(errors='replace')[-300:]}"
    return proc.stdout


def loaded(source: Path) -> torch.Tensor:
    raw = run([assembler.FFMPEG, "-v", "error", "-an", "-i", str(source), "-pix_fmt", "rgba64le", "-vf", assembler.loader_fit(W, H),
               "-fps_mode", "passthrough", "-f", "rawvideo", "-"])
    a = np.frombuffer(raw, np.dtype(np.uint16).newbyteorder("<")).reshape(FRAMES, H, W, 4)[..., :3]
    return torch.from_numpy((a / 65535.0).astype(np.float32))


with tempfile.TemporaryDirectory() as _tmp:
    TMP = Path(_tmp)
    SOURCE = TMP / "source.mp4"
    run([assembler.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", PICTURE.format(w=SW, h=SH), "-frames:v", str(FRAMES),
         "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264", "-crf", "8",
         "-pix_fmt", "yuv420p", *assembler.BT709_TAGS, str(SOURCE)])
    WHOLE = loaded(SOURCE)
    assert float(WHOLE.std()) > 0.03, "the test picture came out flat"
    song = _song()
    x0, y0, x1, y1 = RECT

    def piece(name: str, change) -> str:
        frames = WHOLE.clone()
        frames[:, y0:y1, x0:x1] = change(frames[:, y0:y1, x0:x1]).clamp(0, 1)
        path = str(TMP / f"{name}.mp4")
        assert song._write_frames_mp4(path, frames, CRF) == FRAMES
        return path

    def soft(region: torch.Tensor) -> torch.Tensor:
        a = region.numpy()
        return torch.from_numpy(np.stack([cv2.GaussianBlur(f, (0, 0), BLUR) for f in a]))

    def bluer(region: torch.Tensor) -> torch.Tensor:
        out = region + LIFT
        out[..., 2] += 0.10
        return out

    PLAIN = piece("plain", lambda r: r + LIFT)
    SOFT = piece("soft", lambda r: soft(r) + LIFT)
    BLUE = piece("blue", bluer)

    def measured(path: str, noise: float = 0.0) -> dict:
        return tool.measure(argparse.Namespace(source=str(SOURCE), piece=path, first=0, frames=None, step=1, control_noise=noise))["sets"]

    got: dict = {}

    def grain_at_its_size() -> str:
        got["plain"], noisy = measured(PLAIN), measured(PLAIN, NOISE)
        own = got["plain"]["new"]["grain_sd_by_band"][0]
        want = float(np.hypot(own, 0.872 * NOISE))
        read = noisy["new"]["grain_sd_by_band"][0]
        assert abs(read - want) < 0.08 * want, f"noise of {NOISE} levels should read {want:.2f} in the finest band and reads {read:.2f}"
        before, after = got["plain"]["new"]["detail_sd_by_band"][0], noisy["new"]["detail_sd_by_band"][0]
        assert before > 1.0, f"the test picture has no fine detail to speak of ({before})"
        assert abs(after - before) < 0.05 * before, f"the picture's own detail moved from {before} to {after} when only noise was added"
        return f"{NOISE:g} levels of noise read {read:.2f} (expected {want:.2f}); detail {before:.2f} then {after:.2f}"

    def softer_is_softer() -> str:
        got["soft"] = measured(SOFT)
        fine, mid = "detail_ratio_fine_over_mid", "detail_ratio_mid_over_coarse"
        sharp, softer = got["plain"]["new"], got["soft"]["new"]
        trial = got["plain"][f"new, blurred {BLUR}"]
        assert softer[fine] < 0.5 * sharp[fine] and softer[mid] < 0.8 * sharp[mid], \
            f"a region blurred by sigma {BLUR} reads {softer[fine]} and {softer[mid]} against {sharp[fine]} and {sharp[mid]} unblurred"
        # the trial row is judged on mid over coarse: after a blur of one pixel the finest band holds little but the
        # rounding of the file's 8 bits, which the trial, blurring floats, does not have
        assert abs(trial[mid] - softer[mid]) < 0.10 * softer[mid], f"the trial blur predicts {trial[mid]} mid over coarse; the piece blurred that much reads {softer[mid]}"
        return (f"fine over mid {sharp[fine]} unblurred, {softer[fine]} blurred by {BLUR}; mid over coarse {sharp[mid]}, {softer[mid]}, "
                f"and the trial row said {trial[mid]}")

    def tone_where_it_was_put() -> str:
        got["blue"] = measured(BLUE)
        new, old = got["plain"]["new"], got["plain"]["old"]
        lift = new["luma_percentiles_1_5_25_50_75_95_99"][3] - old["luma_percentiles_1_5_25_50_75_95_99"][3]
        want = LIFT * 219                                  # full scale is 219 levels of luma in a tv-range file
        assert abs(lift - want) < 6, f"a region lifted by {want:.0f} levels reads {lift} brighter at its median"
        u_plain = np.mean([v[0] for v in new["chroma_offset_u_v_by_luma"].values() if v[0]])
        u_blue = np.mean([v[0] for v in got["blue"]["new"]["chroma_offset_u_v_by_luma"].values() if v[0]])
        assert u_blue - u_plain > 5, f"blue added to the region moved its U offset from {u_plain:.1f} to {u_blue:.1f}"
        plate = abs(np.mean([v[0] for v in got["blue"]["plate"]["chroma_offset_u_v_by_luma"].values() if v[0]])
                    - np.mean([v[0] for v in got["plain"]["plate"]["chroma_offset_u_v_by_luma"].values() if v[0]]))
        assert plate < 1.0, f"the plate's own cast moved by {plate:.1f} when only the region was tinted"
        return f"median luma +{lift} (put: {want:.0f}); U offset {u_plain:.1f} to {u_blue:.1f} with blue added, the plate's unmoved"

    def the_region_is_the_region() -> str:
        area = (x1 - x0) * (y1 - y0) * (FRAMES - 1)
        inner = (x1 - x0 - 2 * tool.INSET) * (y1 - y0 - 2 * tool.INSET) * (FRAMES - 1)
        px = got["plain"]["new"]["pixels"]
        assert 0.85 * inner <= px <= area, f"`new` holds {px} px; the rectangle inset by {tool.INSET} is {inner} and whole is {area}"
        band = got["plain"]["plate"]["pixels"]
        key = "luma_percentiles_1_5_25_50_75_95_99"
        assert band > 0.5 * px and got["plain"]["plate"][key][3] < got["plain"]["new"][key][3] - 15, "the plate band is not outside the lifted rectangle"
        return f"`new` is {px} px of the rectangle's {area}; the plate band {band} px and not lifted"

    case("grain is read at its size, and not as detail", grain_at_its_size)
    case("a softer region is read as softer", softer_is_softer)
    case("tone and cast are read where they were put", tone_where_it_was_put)
    case("the region is the region", the_region_is_the_region)
sys.exit(finish())
