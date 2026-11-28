"""Standalone unity-gain MiniMax H3 audio VAE decoder (`MiniMaxH3DecodeAudio`).

Decodes H3 stereo audio latents (`[B, C, 2, T]` or `NestedTensor(video, audio)`)
without ComfyUI core `VAEDecodeAudio`'s `std * 5.0` loudness crush (which attenuates
mastered music above -14 dBFS by up to -6 dB and causes seam pumping across chained
windows) and without core's broken 2D spatial image tiling OOM fallback.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)


def _clip_guard(audio: torch.Tensor, ceiling: float = 0.999) -> tuple[torch.Tensor, list[str]]:
    peak = float(audio.abs().max().item()) if audio.numel() else 0.0
    if peak <= ceiling:
        return audio, []
    scale = ceiling / peak
    return audio * scale, [f"clip_guard scaled peak {peak:.4f} -> {ceiling:.3f}"]


def decode_audio_1d_chunked(
    audio_vae,
    latent: torch.Tensor,
    *,
    chunk_t: int = 400,
    overlap_t: int = 20,
    hop: int = 800,
) -> torch.Tensor:
    """Decode [B, C, 2, T] audio latents in 1D overlap-add chunks (avoids core 2D image tiling)."""
    b, _, _, total_t = latent.shape
    if total_t <= chunk_t:
        return audio_vae.decode(latent).movedim(-1, 1).to(torch.float32)
    step = max(1, chunk_t - overlap_t)
    out = torch.zeros((b, 2, total_t * hop), dtype=torch.float32, device="cpu")
    weight = torch.zeros((1, 1, total_t * hop), dtype=torch.float32, device="cpu")
    pos = 0
    while pos < total_t:
        end = min(total_t, pos + chunk_t)
        piece = audio_vae.decode(latent[..., pos:end]).movedim(-1, 1).to(device="cpu", dtype=torch.float32)
        n_samps = int(piece.shape[-1])
        w = torch.ones((1, 1, n_samps), dtype=torch.float32)
        fade = min(overlap_t * hop, n_samps // 2)
        if pos > 0 and fade > 0:
            w[..., :fade] = torch.linspace(0.0, 1.0, fade, dtype=torch.float32)
        if end < total_t and fade > 0:
            w[..., -fade:] = torch.linspace(1.0, 0.0, fade, dtype=torch.float32)
        s0 = pos * hop
        s1 = min(out.shape[-1], s0 + n_samps)
        take = s1 - s0
        out[..., s0:s1] += piece[..., :take] * w[..., :take]
        weight[..., s0:s1] += w[..., :take]
        if end >= total_t:
            break
        pos += step
    return out / weight.clamp(min=1e-6)


class MiniMaxH3DecodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DecodeAudio",
            display_name="MiniMax H3 Decode Audio",
            category="model/latent/minimax",
            description=(
                "Decode MiniMax H3 audio latents (plain 4D or NestedTensor AV) at unity gain "
                "without ComfyUI core VAEDecodeAudio's std*5.0 loudness crush or 2D spatial "
                "OOM tiling fallback."
            ),
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Latent.Input("samples"),
                io.Combo.Input("level", options=["raw", "clip_guard"], default="raw"),
            ],
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, audio_vae, samples, level: str = "raw") -> io.NodeOutput:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[-1]
        if not torch.is_tensor(latent) or latent.ndim != 4 or int(latent.shape[2]) != 2:
            raise ValueError(
                f"MiniMaxH3DecodeAudio expects stereo [B, C, 2, T] audio latent, got {tuple(getattr(latent, 'shape', ()))}"
            )
        vae_rate = int(
            samples.get("sample_rate")
            if isinstance(samples, Mapping) and samples.get("sample_rate") is not None
            else getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000))
        )
        hop = int(getattr(audio_vae, "downscale_ratio", 800) or 800)
        try:
            audio = audio_vae.decode(latent).movedim(-1, 1).to(torch.float32)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            fallback_chunk = min(400, max(32, int(latent.shape[-1]) // 2))
            audio = decode_audio_1d_chunked(audio_vae, latent, chunk_t=fallback_chunk, hop=hop)

        fixes: list[str] = []
        if level == "clip_guard":
            audio, fixes = _clip_guard(audio)
        elif level != "raw":
            raise ValueError(f"MiniMaxH3DecodeAudio: unsupported level {level!r}")

        peak = float(audio.abs().max().item()) if audio.numel() else 0.0
        rms = math.sqrt(float((audio * audio).mean().item()) + 1e-12) if audio.numel() else 0.0
        rms_db = 20.0 * math.log10(max(rms, 1e-9))
        report = (
            f"decoded {int(audio.shape[-1])} samples ({audio.shape[-1] / vae_rate:.3f}s @ {vae_rate} Hz), "
            f"peak={peak:.4f}, rms={rms_db:.2f} dBFS"
            + ("; " + "; ".join(fixes) if fixes else "")
        )
        logger.info("[h3] MiniMaxH3DecodeAudio: %s", report)
        return io.NodeOutput({"waveform": audio, "sample_rate": vae_rate}, report)
