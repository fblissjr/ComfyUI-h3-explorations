#!/usr/bin/env python3
"""What does promoting the H3 video VAE's encoder to fp32 actually change?

Run it with the ComfyUI venv python (`docs/wiki/ops/comfy_notes.md`). Loads the real
video VAE; no server is needed. It runs on the card when there is one. On the CPU,
`--frames 1 --size 128x128` (the 2026-08-21 input) is known to work; the clip default
has not been tried there.

**Why this exists rather than a pair of clips.** Encoder precision is a
numerical knob, and `CLAUDE.md` records that a rendered clip cannot A/B one:
the trajectory diverges completely from any perturbation, on any sampler, so
the changed arm's output is a DIFFERENT SAMPLE rather than a degraded version
of the same one. `workflows/h3_probe_ref_vae_encoder_{fp16,fp32}_api.json`
price the knob and prove it runs; they cannot say which is better and are
labelled so. This file is the comparison that is controlled by construction --
same input tensor, same module, one dtype changed, measured at the call.

**It does not say fp32 is better either.** Nothing here has a reference to be
right against: the fp16 checkpoint is the only artifact on disk, so a delta is
a delta. What it answers is whether the node does anything at all, and how big
"anything" is next to a coarser dtype.

Three arms, and the two that are not the measurement are what make it readable:

  determinism   fp16 against fp16, re-encoded. Must be BIT-IDENTICAL. If it is
                not, every delta below is contaminated by nondeterminism and
                none of them mean anything.
  fp32          the measurement.
  bf16          the direction control. bf16 has fewer mantissa bits than fp16,
                so its delta from the fp16 baseline must be LARGER than fp32's.
                If it is not, this script is not measuring precision -- it is
                measuring cast noise, and the ordering would be arbitrary.

**A clip, not a frame (2026-11-18).** Until that date the input was one random
frame, and open experiment #33 closed "no change" for `--fp16-accumulation` on
it. A real 141-frame clip showed the switch moving the encode
(`bench/results/2026-11-18_vae_fp16_accumulation.md`): `CausalConv3d.forward` in
core's `comfy/ldm/minimax/vae.py` sends a one-frame input through its plain path
in every temporally padded conv, so kitchen's kernel never ran there. The default
is now FRAMES frames at SIZE, random unless `--clip` names a video, whose first
frames are scaled to the size. Random pixels are not picture content; pass
`--clip` when the size of a delta matters, not only whether there is one.

Exit 0 when both controls hold, 1 when either does not, whatever the deltas
said.

    <comfy-venv>/bin/python bench/grade_vae_encoder_precision.py --out data/results/<date>_vae_encoder_precision.json
    <comfy-venv>/bin/python bench/grade_vae_encoder_precision.py --fp16-accumulation \
        --clip <video> --out data/results/<date>_vae_encoder_fp16acc.json

`--out` may be left off only for the 2026-08-21 input (`--frames 1 --size
128x128`, no `--clip`), which rewrites that record's data file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parents[1]))         # ComfyUI root, first
sys.path.insert(1, str(_REPO))

# One fixed pixel block, so every arm encodes the same bytes.
SEED = 0
# 672x384 is #33's size (bench/results/2026-09-27_vae_encoder_fp16acc.md): big
# enough for kitchen's launch rules to matter, small enough that every arm fits
# on the card in one process. 22 = 17 + 5, the shortest length that holds a whole
# 17-frame causal clip after the leading one (h3_rules.py::snap_length).
SIZE = (672, 384)
FRAMES = 22
# The 2026-08-21 record's input, the only one allowed to write its data file.
INPUT_2026_08_21 = {"size": (128, 128), "frames": 1, "clip": None}


def _dtype(module):
    for p in module.parameters():
        return p.dtype
    return None


def _fresh_vae(vae_name: str):
    import comfy.sd
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path("vae", vae_name)
    if path is None:
        return None, None
    # Reloaded from disk per arm rather than deep-copied. The node casts the
    # module in place and the loader caches it, so sharing one object between
    # arms would let arm N-1's cast leak into arm N -- the baseline sharing
    # mutable state with the thing it measures, which CLAUDE.md rules out.
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(path)), path


def _encode(vae_name: str, encoder: str | None, pixels):
    from vae_precision import MiniMaxH3VAEPrecision
    vae, _ = _fresh_vae(vae_name)
    if vae is None:
        return None, None, None
    if encoder is not None:
        out = MiniMaxH3VAEPrecision.execute(vae, encoder=encoder,
                                            decoder="unchanged")
        vae = out[0] if isinstance(out, (tuple, list)) else out.result[0]
    enc_dtype = _dtype(vae.first_stage_model.encoder)
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    # Under inference mode, as the server runs every node. Without it autograd
    # kept every activation of the encode: 22 frames at 672x384 ran the card out
    # of memory on 2026-11-18, and so did one frame at 1344x768.
    with torch.inference_mode():
        lat = vae.encode(pixels).detach().cpu()
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    # Each arm loads its own VAE; free it before the next, `out` included (it
    # holds the cast copy), so a later arm starts from an empty card.
    import comfy.model_management as mm
    del vae
    if encoder is not None:
        del out
    mm.unload_all_models()
    mm.soft_empty_cache()
    mem = {"peak_mib": round(peak) if peak is not None else None,
           "left_after_free_mib": round(torch.cuda.memory_allocated() / 2**20) if cuda else None}
    return lat, enc_dtype, mem


def _delta(a, b):
    a, b = a.detach(), b.detach()
    d = (a.float() - b.float()).abs()
    ref = a.float().abs().mean()
    return {"max": float(d.max()), "mean": float(d.mean()),
            "relative_mean": float(d.mean() / ref) if ref > 0 else None,
            "bit_identical": bool(__import__("torch").equal(a.float(), b.float()))}


def _clip_frames(path: Path, frames: int, size: tuple[int, int]):
    """The first `frames` frames of a video, scaled to `size`, as [F, H, W, 3] in 0..1."""
    import subprocess
    import numpy as np
    import torch
    w, h = size
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(frames),
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    got = len(raw) // (w * h * 3)
    if got < frames:
        raise SystemExit(f"{path.name} gave {got} frames; --frames asks for {frames}")
    arr = np.frombuffer(raw, np.uint8).reshape(frames, h, w, 3)
    return torch.from_numpy(arr.copy()).float() / 255.0


def _kitchen_version():
    try:
        import importlib.metadata as m
        return m.version("comfy_kitchen")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    # Another VAE file's own encoder as an extra arm, at its load dtype, against
    # the shipped file's baseline: whether swapping the FILE moves the latent.
    # Added 2026-09-26 for the INT8 ConvRot file, whose decoder is quantized.
    ap.add_argument("--against", action="append", default=[], metavar="VAE_NAME")
    # Experiment #33 (docs/wiki/ops/open_experiments.md): the fp16 encode with
    # `--fast fp16_accumulation`'s switch on, as start.sh launched the server
    # until 2026-09-28 (it passes no --fast since).
    # kitchen's fp16_conv3d only runs on CUDA with the switch on, so this arm
    # needs the card; comparing it across kitchen builds is the experiment.
    ap.add_argument("--size", default=None, metavar="WxH",
                    help="input size; default SIZE, #33's 672x384 "
                         "(bench/results/2026-09-27_vae_encoder_fp16acc.md). A tiny input may never "
                         "reach the kitchen kernel's launch rules")
    ap.add_argument("--frames", type=int, default=FRAMES,
                    help="frames to encode: 1, or a 17n+5 length (default FRAMES). One frame skips "
                         "kitchen's kernel in every temporally padded conv")
    ap.add_argument("--clip", type=Path, default=None,
                    help="a video whose first --frames frames, scaled to --size, are the input; "
                         "random pixels when left off")
    ap.add_argument("--fp16-accumulation", action="store_true",
                    help="add the fp16-accumulate encode as an arm (#33); needs CUDA")
    ap.add_argument("--out", type=Path,
                    help="record path; required with --against so the 2026-08-21 record is not overwritten")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x")) if args.size else SIZE
    if args.frames != 1 and (args.frames < 5 or (args.frames - 5) % 17):
        ap.error(f"--frames {args.frames}: the video VAE takes 1 or 17n+5 (h3_rules.py::snap_length)")
    is_0821 = {"size": size, "frames": args.frames, "clip": args.clip} == INPUT_2026_08_21
    if args.out is None and not (is_0821 and not args.against and not args.fp16_accumulation):
        ap.error("--out is needed unless this is the 2026-08-21 input (--frames 1 --size 128x128, "
                 "no --clip, --against or --fp16-accumulation), so that record's data is not overwritten")
    try:
        import torch
        import comfy.sd  # noqa: F401
        import folder_paths  # noqa: F401
    except ImportError as exc:
        print(f"ComfyUI is not importable from here ({exc}); run this with the "
              f"ComfyUI venv python (see docs/wiki/ops/comfy_notes.md)")
        return 2

    sys.path.insert(0, str(_REPO / "workflows"))
    import h3_config
    vae_name = h3_config.MODELS["video_vae"]

    if args.clip is not None:
        pixels = _clip_frames(args.clip, args.frames, size)
    else:
        torch.manual_seed(SEED)
        pixels = torch.rand(args.frames, size[1], size[0], 3)

    base, base_dtype, base_mem = _encode(vae_name, None, pixels)
    if base is None:
        print(f"{vae_name} is not on disk; nothing to grade")
        return 2
    print(f"video VAE: {vae_name}")
    print(f"input: {args.frames} frame(s) at {size[0]}x{size[1]}, "
          f"{args.clip.name if args.clip else 'random pixels, seed ' + str(SEED)}")
    print(f"baseline encoder dtype: {base_dtype}   latent {tuple(base.shape)}\n")

    rows, lats = {}, {}
    for label, enc in (("determinism", None), ("fp32", "fp32"), ("bf16", "bf16")):
        lat, dt, mem = _encode(vae_name, enc, pixels)
        lats[label] = lat
        rows[label] = {"encoder_dtype": str(dt), **_delta(base, lat), "memory": mem}
        r = rows[label]
        rel = "n/a" if r["relative_mean"] is None else f"{r['relative_mean']:.2e}"
        print(f"  {label:<12} encoder {str(dt):<15} max {r['max']:.6f}  "
              f"mean {r['mean']:.8f}  rel {rel}  "
              f"{'identical' if r['bit_identical'] else ''}")

    if args.fp16_accumulation:
        if not torch.cuda.is_available():
            print("--fp16-accumulation needs CUDA: kitchen's fp16_conv3d never runs on the CPU")
            return 2
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            lat, dt, mem = _encode(vae_name, None, pixels)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        rows["fp16_accumulation"] = r = {"encoder_dtype": str(dt), **_delta(base, lat), "memory": mem}
        fp32_lat = lats["fp32"]  # the fp32 arm above; same input, switch off
        r["from_fp32"] = _delta(fp32_lat, lat)
        r["baseline_from_fp32"] = _delta(fp32_lat, base)
        print(f"  fp16_accumulation  encoder {str(dt):<15} max {r['max']:.6f}  mean {r['mean']:.8f}  "
              f"{'identical' if r['bit_identical'] else ''}\n"
              f"    from fp32: accumulate mean {r['from_fp32']['mean']:.8f} against plain fp16 "
              f"{r['baseline_from_fp32']['mean']:.8f}")

    for other in args.against:
        lat, dt, _ = _encode(other, None, pixels)
        if lat is None:
            print(f"{other} is not on disk")
            return 2
        rows[f"file:{other}"] = r = {"encoder_dtype": str(dt), **_delta(base, lat)}
        rel = "n/a" if r["relative_mean"] is None else f"{r['relative_mean']:.2e}"
        print(f"  file {other}\n               encoder {str(dt):<15} max {r['max']:.6f}  "
              f"mean {r['mean']:.8f}  rel {rel}  "
              f"{'identical' if r['bit_identical'] else ''}")

    det_ok = rows["determinism"]["bit_identical"]
    order_ok = rows["bf16"]["mean"] > rows["fp32"]["mean"]
    print("\n--- controls ---")
    print(f"  determinism  fp16 re-encoded is bit-identical: "
          f"{'yes' if det_ok else 'NO -- every delta above is contaminated'}")
    print(f"  direction    bf16 delta exceeds fp32 delta: "
          f"{'yes' if order_ok else 'NO -- this is not measuring precision'}")

    record = {
        "question": "does promoting the H3 video VAE encoder change the latent?",
        "vae": vae_name,
        "input": {"frames": args.frames, "size": list(size),
                  "clip": args.clip.name if args.clip else None,
                  "seed": None if args.clip else SEED},
        "baseline_memory": base_mem,
        "device": str(__import__("comfy.model_management").model_management.get_torch_device()),
        "comfy_kitchen": _kitchen_version(),
        "baseline_encoder_dtype": str(base_dtype),
        "arms": rows,
        "controls": {"determinism_holds": det_ok, "direction_holds": order_ok},
        "reading": ("a delta, not a verdict: the fp16 checkpoint is the only "
                    "artifact on disk, so neither arm has a reference to be "
                    "right against"),
    }
    out = args.out or _REPO / "data" / "results" / "2026-08-21_vae_encoder_precision.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    shown = out.resolve()
    if shown.is_relative_to(_REPO):
        shown = shown.relative_to(_REPO)
    print(f"\nwrote {shown}")
    return 0 if (det_ok and order_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
