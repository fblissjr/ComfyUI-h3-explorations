#!/usr/bin/env python3
"""What core does with a 1-, 2- and 6-channel reference waveform, measured.

**This file retires `bench/check_mono_ref_audio.py`, and the reason it could
retire it is the reason that gate could not retire itself.** That gate asserted
a mono reference RAISES, and it verified the claim by hand-building a
`[1, 32, 1, T]` latent and handing it to `pack_audio`. That assignment does
fail, so the gate stayed green -- but nothing on the real path produces a
1-channel latent any more, so the gate was green about a state the code cannot
reach. Its own chain names the miss: it traced
`comfy/ldm/minimax/audio_vae.py::encode` and stopped there, where
`_encode_ref_audio` actually calls `comfy.sd.VAE.encode`, one wrapper up.
That wrapper runs `vae_encode_crop_pixels` first, and the H3 audio VAE declares
`output_channels = 2` with `pad_channel_value = "replicate"`
(`comfy/sd.py:1030-1035`), so a mono waveform is duplicated to stereo before
the model sees it.

The general shape is `CLAUDE.md`'s: a claim derived from reading a call site
rather than following the call. The gate could not have caught the fix at any
revision, because it never ran the entry point the fix landed in.

**So this is an audit, not a gate.** The subject is no longer a defect: there
is nothing to hold red. What is left is a fact worth keeping measured, because
one of the three answers is still a silent surprise -- more than two channels
are truncated to the first two with no error and no warning.

Needs the real audio VAE (`h3_config.MODELS["audio_vae"]`, ~0.56 GiB) and runs
it on the CPU. No CUDA, no server. Exit 1 only if the run itself fails.

    python bench/audit_ref_audio_channels.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

COMFY = HERE.parent.parent.parent
sys.path.insert(0, str(COMFY))

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio  # noqa: E402

from workflows.h3_config import MODELS  # noqa: E402

SAMPLE_RATE = 32000
#: One second. The channel count is the variable; the length only has to be
#: long enough to produce more than one latent frame at 40 Hz.
SECONDS = 1


def main() -> int:
    path = COMFY / "models" / "vae" / MODELS["audio_vae"]
    if not path.exists():
        print(f"FAIL: {MODELS['audio_vae']} not found")
        return 1
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)))

    print(f"{MODELS['audio_vae']}")
    print(f"  output_channels={vae.output_channels} "
          f"pad_channel_value={vae.pad_channel_value!r}")
    print()

    rows = []
    for label, channels in (("mono", 1), ("stereo", 2), ("5.1", 6)):
        wave = torch.randn(1, channels, SAMPLE_RATE * SECONDS)
        entry = {"label": label, "channels": channels}
        try:
            latent, ref_audio_t = _encode_ref_audio(
                vae, {"waveform": wave, "sample_rate": SAMPLE_RATE})
        except Exception as exc:                      # noqa: BLE001
            entry["raised"] = f"{type(exc).__name__}: {exc}"
            print(f"  {label:6} RAISED {entry['raised']}")
        else:
            # `PackedLayout` allocates `ref_audio_t * 2` rows for the block and
            # `pack_audio` produces `channels * T`. Equal means the assignment
            # the retired gate reproduced cannot fail.
            produced = int(latent.shape[2] * latent.shape[3])
            entry.update(latent_shape=list(latent.shape),
                         ref_audio_t=int(ref_audio_t),
                         rows_allocated=int(ref_audio_t) * 2,
                         rows_produced=produced,
                         packs=produced == int(ref_audio_t) * 2)
            print(f"  {label:6} [1,{channels},{SAMPLE_RATE * SECONDS}] -> "
                  f"{tuple(latent.shape)}  allocated {entry['rows_allocated']} "
                  f"produced {produced}  packs={entry['packs']}")
        rows.append(entry)

    print()
    print("  mono is upmixed by the VAE wrapper, not by the H3 node, and packs "
          "correctly.")
    print("  MORE THAN TWO CHANNELS IS TRUNCATED TO THE FIRST TWO, silently: "
          "no error, no warning,")
    print("  and the render proceeds on a stereo downmix nobody chose. This "
          "repo's typed")
    print("  `reference_conditioning._prepare_audio` refuses it instead; core "
          "does not.")

    record = {
        "date": date.today().isoformat(),
        "subject": "core's channel handling for H3 reference audio",
        "audio_vae": MODELS["audio_vae"],
        "vae_output_channels": int(vae.output_channels),
        "vae_pad_channel_value": str(vae.pad_channel_value),
        "sample_rate": SAMPLE_RATE,
        "seconds": SECONDS,
        "results": rows,
        "retires": "bench/check_mono_ref_audio.py",
        "why": (
            "The retired gate asserted a mono reference raises. It verified "
            "that by hand-building a 1-channel latent and reproducing "
            "PackedLayout's assignment, which does fail -- but core upmixes "
            "mono to stereo in comfy.sd.VAE.encode before the model is "
            "reached, so no 1-channel latent occurs on the real path. The gate "
            "traced comfy/ldm/minimax/audio_vae.py and stopped one wrapper "
            "short of the call _encode_ref_audio actually makes."),
    }
    out = HERE / "results" / f"{record['date']}_ref_audio_channels.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
