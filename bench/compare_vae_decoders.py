#!/usr/bin/env python3
"""Decode one saved latent with several video VAE files: time, peak VRAM, pixels.

Written 2026-09-26 for two questions: what the INT8 ConvRot video VAE costs
and saves at the owner's length (its 2026-08-10 measurement was at 124 frames;
CHANGELOG, "Decode 12.8s -> 9.9s"), and what `h3_config.DRAFT_VAE` saves
(`docs/open_experiments.md` #31). A decode of a fixed latent is a legitimate
numerical comparison, unlike a render: nothing upstream of the decoder moves,
so every pixel difference is the decoder's.

Each arm runs in a fresh process, so its peak VRAM is its own (weights plus
activations) and no arm inherits another's cache. Each child decodes twice and
reports both times: the first pays the weight transfer and kernel warm-up, the
second is the steady state. The children parse the server's `--fast` flags
(default `fp16_accumulation`, as `start.sh` launches it), because core's H3
VAE takes a different convolution path under it.

Controls, and the reading depends on them:
  fp16 twice    must be bit-identical, or every delta below includes
                nondeterminism.
  taeh3         the tiny autoencoder, an approximation by design. It must sit
                well below INT8 in PSNR, or the metric is not measuring decoder
                fidelity.

Needs the GPU with no server holding it (`POST /free` first) and the ComfyUI
venv. The latent is a `SaveLatent` file of the VIDEO half, as
`h3_text_to_video_flashgen_draft` writes it.

    python bench/compare_vae_decoders.py LATENT.latent --out bench/results/<date>_<what>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO / "workflows"))

from h3_config import DRAFT_VAE, MODELS  # noqa: E402

INT8_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
#: (label, folder, file). fp16 runs twice: the second run is the determinism
#: control and never a result.
ARMS = [
    ("fp16", "vae", MODELS["video_vae"]),
    ("fp16_repeat", "vae", MODELS["video_vae"]),
    ("int8", "vae", INT8_VAE),
    ("taeh3", "vae_approx", DRAFT_VAE),
]


def _child(args) -> int:
    sys.path.insert(0, str(COMFY))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0]] + (["--fast", *args.fast] if args.fast else [])
    import numpy as np
    import safetensors.torch
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise(args.folder, args.file)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))
    latent = safetensors.torch.load_file(args.latent)["latent_tensor"].float()

    torch.cuda.reset_peak_memory_stats()
    seconds = []
    for _ in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the executor runs every node under inference_mode; without it autograd
        # holds every activation and the decode runs out of memory
        with torch.inference_mode():
            images = vae.decode(latent)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated()
    frames = images.reshape(-1, *images.shape[-3:])
    u8 = (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    np.save(args.save, u8)
    print(json.dumps(dict(
        file=args.file, vae_dtype=str(getattr(vae, "vae_dtype", None)),
        latent_shape=list(latent.shape), frames_shape=list(u8.shape),
        decode_s_first=round(seconds[0], 2), decode_s_steady=round(seconds[1], 2),
        peak_mib=round(peak / 2**20))))
    return 0


def _psnr(a, b) -> float:
    import numpy as np
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _compare(ref, arm) -> dict:
    import numpy as np
    if ref.shape != arm.shape:
        return dict(error=f"shape {list(arm.shape)} against {list(ref.shape)}")
    per_frame = [_psnr(ref[i], arm[i]) for i in range(len(ref))]
    finite = [p for p in per_frame if p != float("inf")]

    def motion(x):
        return float(np.mean(np.abs(np.diff(x.astype(np.int16), axis=0))))
    return dict(
        bit_identical=bool(np.array_equal(ref, arm)),
        psnr_db=round(_psnr(ref, arm), 2),
        psnr_frame_min_db=round(min(finite), 2) if finite else None,
        worst_frame=int(np.argmin(per_frame)) if finite else None,
        max_abs=int(np.max(np.abs(ref.astype(np.int16) - arm.astype(np.int16)))),
        # frame-to-frame change of the arm over the reference's; flicker reads
        # above 1, smoothing below (the 2026-08-10 INT8 record's temporal metric)
        motion_ratio=round(motion(arm) / motion(ref), 4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("latent", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fast", nargs="*", default=["fp16_accumulation"])
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--folder", help=argparse.SUPPRESS)
    ap.add_argument("--file", help=argparse.SUPPRESS)
    ap.add_argument("--save", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(args)

    import numpy as np
    rows, frames = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, folder, fname in ARMS:
            save = str(Path(tmp) / f"{label}.npy")
            cmd = [sys.executable, __file__, str(args.latent.resolve()), "--child",
                   "--folder", folder, "--file", fname, "--save", save,
                   "--fast", *args.fast]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(COMFY))
            if r.returncode != 0:
                print(r.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"arm {label} failed")
            rows[label] = json.loads(r.stdout.strip().splitlines()[-1])
            frames[label] = np.load(save, mmap_mode="r")
            print(label, rows[label], flush=True)
        ref = frames["fp16"]
        for label in rows:
            if label != "fp16":
                rows[label]["against_fp16"] = _compare(ref, frames[label])
        controls = dict(
            determinism=rows["fp16_repeat"]["against_fp16"]["bit_identical"],
            taeh3_below_int8=bool(rows["taeh3"]["against_fp16"]["psnr_db"]
                                  < rows["int8"]["against_fp16"]["psnr_db"]))
    record = dict(
        question="what the INT8 and taeh3 video decoders cost and save against fp16, on one saved latent",
        latent=args.latent.name, fast=args.fast, arms=rows, controls=controls)
    text = json.dumps(record, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if all(controls.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
