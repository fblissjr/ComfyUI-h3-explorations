#!/usr/bin/env python3
"""The control for the audio-freeze node: the sibling pack's song node, bit for bit.

`docs/h3_audio_freeze.md` section 4 step 1 names the control: fed the same
track at the same start, `audio_freeze.py::MiniMaxH3FreezeAudio` and the
installed pack's `MiniMaxH3SongMaskedAVContext`
(`custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_song_audio_context.py`,
a sibling under ComfyUI, not this repo) must write the same audio latent and
the same nested mask. Two independent implementations of one contract, on
the real audio VAE on CPU, agreeing to the bit, is the cheapest proof that
neither slices off the grid. Where they differ, the record says by how much
and where, and the diff is the finding.

**An audit, not a gate.** The pack is a third-party install that may be
absent, so this exits 2 (nothing graded) when it cannot be imported, and it
never blocks a commit. It also writes a dated record under `bench/results/`
because the comparison depends on the pack's revision, the VAE file and the
track.

The pack's node is called with `context_length=0` and no source, so only its
audio path runs; its `_require_h3_mask_support` probes core for the mask path
and installs nothing on this build (measured 2026-08-30).

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/audit_audio_freeze_control.py \\
        --audio <path to the track> [--start 0.0] [--length 345] [--out bench/results/...json]

The track is a path argument on purpose: the server's input directory is
outside this repo and is not written into it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(COMFY / "custom_nodes"))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy.nested_tensor  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_minimax_h3 import temporal_shape  # noqa: E402

import audio_freeze as af  # noqa: E402
from h3_config import CANVAS, LONG_LENGTH, MODELS  # noqa: E402

PACK = "ComfyUI-H3-Motion-Context-MultiRef"


def _load_track(path: Path):
    # Core's own decoder (PyAV), the one LoadAudio uses; torchaudio.load
    # wants torchcodec, which this venv does not carry.
    from comfy_extras.nodes_audio import load
    wave, rate = load(str(path))
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    return {"waveform": wave, "sample_rate": int(rate)}


def _pack_node():
    try:
        mod = importlib.import_module(f"{PACK}.h3_song_audio_context")
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return mod.MiniMaxH3SongMaskedAVContext(), None


def _pack_revision() -> str | None:
    head = COMFY / "custom_nodes" / PACK / ".git" / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref: "):
        p = COMFY / "custom_nodes" / PACK / ".git" / ref[5:]
        return p.read_text().strip()[:12] if p.exists() else None
    return ref[:12]


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_ours": list(a.shape), "shape_pack": list(b.shape), "equal": False}
    d = (a - b).abs()
    return {"shape": list(a.shape), "equal": bool(torch.equal(a, b)),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "first_differing_index": (int(d.flatten().nonzero()[0]) if d.max() > 0 else None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="the track to slice")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--length", type=int, default=LONG_LENGTH)
    ap.add_argument("--out", default=str(HERE / "results" / f"{date.today().isoformat()}_audio_freeze_control.json"))
    args = ap.parse_args()

    pack, why = _pack_node()
    if pack is None:
        print(f"  skip  the sibling pack {PACK} did not import: {why}")
        return 2

    vae_path = COMFY / "models" / "vae" / MODELS["audio_vae"]
    if not vae_path.exists():
        print(f"  skip  {MODELS['audio_vae']} not found")
        return 2
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_path)))

    frame_count, latent_t, audio_t = temporal_shape(args.length)
    video = torch.zeros(1, 24, latent_t, CANVAS["height"] // 16, CANVAS["width"] // 16)
    audio = torch.zeros(1, 32, 2, audio_t)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    track = _load_track(Path(args.audio))

    ours = af.MiniMaxH3FreezeAudio.execute(latent, audio_vae, track, args.start, 0.0)
    ours = getattr(ours, "args", ours)
    ours_latent, ours_clip, ours_report = ours[0], ours[1], ours[2]

    theirs_latent, _trim, theirs_clip = pack.prepare(
        latent, audio_vae, track, clip_start_seconds=args.start, context_length=0)

    _ov, oa = ours_latent["samples"].unbind()
    _tv, ta = theirs_latent["samples"].unbind()
    ovm, oam = ours_latent["noise_mask"].unbind()
    tvm, tam = theirs_latent["noise_mask"].unbind()

    record = {
        "date": date.today().isoformat(),
        "what": "audio-freeze node against the sibling pack's song node, same track, same start, real audio VAE on CPU",
        "inputs": {"audio": Path(args.audio).name, "start_seconds": args.start,
                   "length": args.length, "frame_count": frame_count,
                   "latent_t": latent_t, "audio_t": audio_t,
                   "canvas": [CANVAS["width"], CANVAS["height"]],
                   "audio_vae": MODELS["audio_vae"], "pack_revision": _pack_revision()},
        "ours_report": ours_report,
        "audio_latent": _diff(oa, ta),
        "video_mask": _diff(ovm, tvm),
        "audio_mask": _diff(oam, tam),
        "clip_audio": _diff(ours_clip["waveform"], theirs_clip["waveform"]),
        "clip_rates": [ours_clip["sample_rate"], theirs_clip["sample_rate"]],
        "caveats": [
            "the pack's node was run with context_length=0 and no source, so only its audio path is compared",
            "a difference in clip_audio is a difference in slicing or resampling, not in the encoder; read it first",
            "the video stream is zeros on both sides by construction and is not compared",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    all_equal = all(record[k].get("equal") for k in ("audio_latent", "video_mask", "audio_mask", "clip_audio"))
    for k in ("audio_latent", "video_mask", "audio_mask", "clip_audio"):
        r = record[k]
        print(f"  {k:<13} {'equal' if r.get('equal') else 'DIFFER'}  "
              + (f"max_abs {r['max_abs_diff']:.3g}" if "max_abs_diff" in r else f"{r}"))
    print(f"  record: {out.relative_to(REPO)}")
    print("  ok    the two implementations agree to the bit" if all_equal else
          "  note  they differ; the record says where. Read clip_audio first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
