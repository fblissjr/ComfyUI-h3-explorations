"""One resampler for every waveform this pack hands to H3's audio VAE.

Core dropped torchaudio from its requirements in Comfy-Org/ComfyUI#16457 and
resamples H3 reference audio with `comfy.audio.resample`, a port of
torchaudio's bandlimited sinc at the same defaults. This pack calls that
function, so it resamples exactly as core does and needs no dependency core
no longer installs. On a core older than #16457 there is no `comfy.audio`,
and torchaudio, which that core still required, is the fallback.

The two are interchangeable, not merely close. Their outputs were
`torch.equal` on CPU and CUDA, in fp32, fp16 and bf16, at every source rate
tried (measured: `bench/results/2026-09-25_upstream_survey_checks.md`,
check 1).
"""

from __future__ import annotations

import torch

try:
    from comfy.audio import resample as _resample
    RESAMPLER = "comfy.audio.resample (torchaudio's sinc at its defaults, as core resamples)"
except ImportError:  # core before Comfy-Org/ComfyUI#16457
    import torchaudio

    _resample = torchaudio.functional.resample
    RESAMPLER = "torchaudio sinc at its defaults (this core predates comfy.audio)"


def resample(waveform: torch.Tensor, rate: int, new_rate: int) -> torch.Tensor:
    """`waveform` resampled along its last axis from `rate` to `new_rate` Hz."""
    if int(rate) == int(new_rate):
        return waveform
    return _resample(waveform, int(rate), int(new_rate))
