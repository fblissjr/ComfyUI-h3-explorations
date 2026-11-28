#!/usr/bin/env python3
"""Offline unit tests for standalone `h3_audio_vae.py` (`MiniMaxH3DecodeAudio`)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
for p in (str(REPO), str(COMFY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import comfy.nested_tensor  # noqa: E402
from h3_audio_vae import MiniMaxH3DecodeAudio  # noqa: E402


class _FakeAudioVAE:
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000
    downscale_ratio = 800

    def __init__(self, *, oom_above_t: int | None = None, gain: float = 0.85):
        self.oom_above_t = oom_above_t
        self.gain = gain

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        t_steps = int(z.shape[-1])
        if self.oom_above_t is not None and t_steps > self.oom_above_t:
            raise RuntimeError("CUDA out of memory")
        b = int(z.shape[0])
        n = t_steps * 800
        x = torch.linspace(0.0, 20.0 * math.pi, n, dtype=torch.float32)
        return (self.gain * torch.sin(x)).view(1, n, 1).expand(b, n, 2).contiguous()


def main() -> int:
    v_lat = torch.zeros(1, 24, 5, 8, 8)
    a_lat = torch.randn(1, 32, 2, 10)
    nested = comfy.nested_tensor.NestedTensor((v_lat, a_lat))

    # 1. Unity gain on loud audio (std ~ 0.60 > 0.20, zero core std*5 crush) + NestedTensor unbind
    out = MiniMaxH3DecodeAudio.execute(_FakeAudioVAE(gain=0.85), {"samples": nested}, level="raw")
    aud, rep = out[0], out[1]
    wav = aud["waveform"]
    assert aud["sample_rate"] == 32000 and tuple(wav.shape) == (1, 2, 8000)
    peak = float(wav.abs().max().item())
    assert abs(peak - 0.85) < 1e-4, f"expected unity peak 0.85, got {peak}"
    print(f"ok    unity_gain_nested: {rep}")

    # 2. 1D chunked OOM fallback produces exact shape [1, 2, T*800] for both long (>400) and short (<=400) latents
    long_a = torch.randn(1, 32, 2, 500)
    out_oom = MiniMaxH3DecodeAudio.execute(_FakeAudioVAE(oom_above_t=400, gain=0.70), {"samples": long_a}, level="raw")
    wav_oom = out_oom[0]["waveform"]
    assert tuple(wav_oom.shape) == (1, 2, 500 * 800)
    assert abs(float(wav_oom.abs().max().item()) - 0.70) < 1e-3
    short_oom_a = torch.randn(1, 32, 2, 200)
    out_oom_s = MiniMaxH3DecodeAudio.execute(_FakeAudioVAE(oom_above_t=150, gain=0.70), {"samples": short_oom_a}, level="raw")
    assert tuple(out_oom_s[0]["waveform"].shape) == (1, 2, 200 * 800)
    print(f"ok    oom_1d_chunked: long={tuple(wav_oom.shape)}, short={tuple(out_oom_s[0]['waveform'].shape)}")

    # 3. clip_guard scales >1.0 peaks cleanly while leaving <=0.999 untouched
    out_hot = MiniMaxH3DecodeAudio.execute(_FakeAudioVAE(gain=1.40), {"samples": a_lat}, level="clip_guard")
    hot_peak = float(out_hot[0]["waveform"].abs().max().item())
    assert abs(hot_peak - 0.999) < 1e-4, f"expected 0.999 ceiling, got {hot_peak}"
    out_ok = MiniMaxH3DecodeAudio.execute(_FakeAudioVAE(gain=0.60), {"samples": a_lat}, level="clip_guard")
    assert abs(float(out_ok[0]["waveform"].abs().max().item()) - 0.60) < 1e-4
    print(f"ok    clip_guard: hot->{hot_peak:.4f}, safe=0.6000")

    # 4. Schema contract
    schema = MiniMaxH3DecodeAudio.define_schema()
    assert schema.node_id == "MiniMaxH3DecodeAudio" and len(schema.inputs) == 3
    print(f"ok    schema: {schema.node_id} ({len(schema.inputs)} inputs)")

    print("\nall 4 case(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
