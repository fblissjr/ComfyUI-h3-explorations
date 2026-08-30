#!/usr/bin/env python3
"""What the generic VAE crop does to a reference waveform on the H3 audio path.

Subject: `comfy/sd.py`'s `vae_encode_crop_pixels` runs on every `VAE.encode`
call, and the MiniMax H3 audio branch does not set `crop_input = False`
(`comfy/sd.py:1030-1046` sets `output_channels`, `pad_channel_value`,
`downscale_ratio = 800` and never touches the flag defaulted `True` at `:515`).
`_encode_ref_audio` presents `[1, samples, channels]`, so `dims` is the SAMPLE
axis and the generic crop narrows it to a multiple of `spacial_compression_encode()`
-- which for this VAE is the audio `downscale_ratio`, 800 -- taking
`(samples % 800) // 2` off the FRONT and the rest off the back.

Two things are measured, separately, because they fail differently:

1. **Which samples are dropped**, by handing the crop a ramp and reading the
   first surviving value. Exact, needs no model.
2. **Whether the drop changes `ref_audio_t`**, by running the real audio VAE
   with the flag as shipped and with `crop_input = False`, which is the whole
   of Comfy-Org/ComfyUI#15972. The second arm is the control: if the row count
   is identical the crop is cosmetic, and if it differs the shipped path emits
   a shorter audio block than the waveform contains.

Needs the real audio VAE (`h3_config.MODELS["audio_vae"]`, ~0.56 GiB), runs it
on the CPU. No CUDA, no server. Exit 1 only if the run itself fails.

    python bench/audit_ref_audio_crop.py
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

# The audio VAE is 0.56 GiB and this measurement is arithmetic, so it runs on the
# CPU rather than contending with whatever holds the card. Two steps, and the
# first is the one that is easy to miss: `comfy.cli_args` parses an EMPTY argv
# unless `args_parsing` is enabled, so appending `--cpu` alone is silently
# ignored and the run lands on the GPU anyway.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio  # noqa: E402

from workflows.h3_config import FPS, MODELS  # noqa: E402

SAMPLE_RATE = 32000

#: What this repo's own trim produces for the shipped default target. Derived,
#: not typed: `_prepare_audio` caps a soundtrack at `round(duration * sr)` with
#: `duration = frame_count / FPS` (`reference_conditioning.py:563,224`), and the
#: `length` widget defaults to 124 (`reference_conditioning.py:960`). This is
#: the case that decides whether the crop reaches a shipped graph at all.
SHIPPED_FRAME_COUNT = 124
SHIPPED_TRIM_SAMPLES = int(round(SHIPPED_FRAME_COUNT / FPS * SAMPLE_RATE))

#: Lengths in samples. One aligned to 800 as the null case, then the PR's own
#: case, then two chosen so the remainder is odd and even -- the front/back
#: split is `//2`, so an odd remainder loses one more sample off the back.
CASES = [
    ("aligned", 800 * 80),
    ("pr_15972", 437_333),
    ("rem_odd", 800 * 80 + 401),
    ("rem_even", 800 * 80 + 400),
    ("shipped_trim", SHIPPED_TRIM_SAMPLES),
]


def main() -> int:
    path = COMFY / "models" / "vae" / MODELS["audio_vae"]
    if not path.exists():
        print(f"FAIL: {MODELS['audio_vae']} not found")
        return 1
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)))

    ratio = vae.spacial_compression_encode()
    print(f"device={comfy.model_management.get_torch_device()}")
    print(f"{MODELS['audio_vae']}")
    print(f"  crop_input={vae.crop_input} "
          f"spacial_compression_encode()={ratio} "
          f"downscale_ratio={vae.downscale_ratio}")
    print()

    rows = []
    for label, samples in CASES:
        rem = samples % ratio
        # 1. which samples survive, read off a ramp
        ramp = torch.arange(samples, dtype=torch.float32).reshape(1, samples, 1)
        ramp = ramp.repeat(1, 1, 2)
        vae.crop_input = True
        cropped = vae.vae_encode_crop_pixels(ramp)
        first = int(cropped[0, 0, 0].item())
        last = int(cropped[0, -1, 0].item())
        dropped_front = first
        dropped_back = (samples - 1) - last

        # 2. what the row count does, shipped against the PR's one-line fix
        wave = torch.randn(1, 2, samples)
        audio = {"waveform": wave, "sample_rate": SAMPLE_RATE}
        vae.crop_input = True
        _, t_shipped = _encode_ref_audio(vae, audio)
        vae.crop_input = False
        _, t_fixed = _encode_ref_audio(vae, audio)
        vae.crop_input = True

        entry = dict(label=label, samples=samples, remainder=rem,
                     kept=int(cropped.shape[1]),
                     dropped_front=dropped_front, dropped_back=dropped_back,
                     ref_audio_t_shipped=int(t_shipped),
                     ref_audio_t_crop_disabled=int(t_fixed),
                     rows_lost=(int(t_fixed) - int(t_shipped)) * 2)
        rows.append(entry)
        print(f"  {label:9} {samples:>8} samples  rem {rem:>3}  "
              f"front -{dropped_front:<3} back -{dropped_back:<3}  "
              f"ref_audio_t {t_shipped} -> {t_fixed}  "
              f"rows lost {entry['rows_lost']}")

    # The paired claim: the same crop is a no-op on the video reference, so a
    # soundtrack drifts against its own frames. Asserted from reading first,
    # then measured here -- `_prepare_reference_video` hands `vae.encode` a
    # `[T, H, W, C]` batch, so T is dim 0 and `dims = shape[1:-1]` never sees
    # it. H and W ARE narrowed to a multiple of 16; shipped graphs escape that
    # by snapping the canvas to 32, which is a property of `_resize`, not of
    # the crop.
    print()
    video = comfy.sd.VAE.__new__(comfy.sd.VAE)
    video.crop_input = True
    video.downscale_ratio = (lambda a: a, 16, 16)   # the H3 video VAE's shape contract
    video.output_channels = 3
    video.pad_channel_value = None
    video_rows = []
    for label, (t, h, w) in (("canvas_32_aligned", (SHIPPED_FRAME_COUNT, 768, 1344)),
                             ("canvas_unaligned", (SHIPPED_FRAME_COUNT, 100, 100))):
        got = tuple(video.vae_encode_crop_pixels(torch.zeros(t, h, w, 3)).shape)
        video_rows.append(dict(label=label, given=[t, h, w], got=list(got),
                               frames_kept=got[0] == t))
        print(f"  video {label:18} [T={t},{h},{w}] -> {got}  frames kept={got[0] == t}")
    print(f"  the time axis is dim 0 on video, so the crop never reaches it; "
          f"H/W are narrowed to a multiple of "
          f"{video.spacial_compression_encode()}")

    print()
    ms = ratio / SAMPLE_RATE * 1000
    print(f"  one latent step is {ratio} samples = {ms:.2f} ms at {SAMPLE_RATE} Hz")
    print(f"  shipped_trim is this repo's own cap for {SHIPPED_FRAME_COUNT} frames "
          f"at {FPS} fps: {SHIPPED_TRIM_SAMPLES} samples")
    out = HERE / "results" / f"{date.today().isoformat()}_ref_audio_crop.json"
    out.write_text(json.dumps(
        dict(vae=MODELS["audio_vae"], sample_rate=SAMPLE_RATE,
             spacial_compression_encode=int(ratio),
             crop_input_default=True,
             shipped_frame_count=SHIPPED_FRAME_COUNT,
             shipped_trim_samples=SHIPPED_TRIM_SAMPLES,
             cases=rows, video_time_axis=video_rows), indent=2) + "\n")
    print(f"  wrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
