"""Freeze a known audio track into an H3 AV latent so the video follows it.

`docs/h3_audio_freeze.md` owns the lane; this file is the node.

**What it does.** Takes the joint video+audio latent the conditioner emits,
slices one window of a song onto H3's audio latent grid, encodes it with the
audio VAE, writes it into the target audio rows, and attaches a nested
`noise_mask` (video ones, audio zeros). Core does the rest: the sampler
unbinds the nested mask per stream (`comfy/samplers.py::CFGGuider.sample`),
`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint` re-injects the clean
audio every step through the schedule carry, and the DiT labels a mask-zero
row with the conditioning timestep (`comfy/ldm/minimax/model.py`, the
`audio_denoise_mask` block). Nothing here patches core.

**The frozen latent is a control signal, not the deliverable.** The audio VAE
round trip is lossy, so the node also returns the exact waveform slice it
encoded, at the VAE's rate, for the muxer. Mux that; do not decode the audio
stream. A consequence worth knowing: the track you freeze need not be the
track you ship (a stem for the latent, the mix for the file).

**The grid.** H3's audio latent runs at `AUDIO_LATENT_FPS` steps per second,
each one hop of `spacial_compression_encode()` samples at the VAE's rate
(`comfy_extras/nodes_minimax_h3.py::temporal_shape`; `comfy/sd.py`, the H3
audio VAE block). The start time snaps to that grid and the slice is exactly
`audio_t * hop` samples, so core's generic input crop
(`comfy/sd.py::vae_encode_crop_pixels`, which our
`reference_conditioning._encode_ref_audio_aligned` pads around) is a no-op
here by construction: the check asserts the encoder returned exactly
`audio_t` steps rather than trusting it.

**Mask value.** `audio_mask` 0.0 is the hard freeze. A value above zero puts
the audio rows at `1 - m * sigma` instead of the clean timestep, which lets
the model own them a little (idea 2 in the lane doc). Core quantises masks
to a 1/256 grid and DROPS a stream's mask when every value is within a
thousandth of 1.0 (`MiniMaxH3._denoise_mask_values`), so 1.0 here means
"no freeze at all", and values just under it do not survive.

**Trap this node guards.** Stock `SetLatentNoiseMask` reshapes one tensor
and stores it flat; on an AV latent the sampler then pads a ones mask for the
audio stream and the track regenerates silently. A flat mask on the input is
refused here with that sentence rather than overwritten.

Not in the release, not in sglang, not in DiffSynth's inference path: all
three treat this as inpainting, and the fl2va checkpoint never saw a clean
audio row (sglang's fl2va task admits image keyframes only). The lane doc
carries the argument for why it is still close to the trained regime.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import torch
from comfy_api.latest import io

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS

logger = logging.getLogger(__name__)


def _av_streams(samples):
    """Return (video, audio) from an H3 AV latent, or raise naming what came in."""
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "MiniMaxH3FreezeAudio expects the joint video+audio latent from the H3 "
            "conditioner (a nested tensor), got a plain tensor of shape "
            f"{tuple(getattr(samples, 'shape', ()))}")
    streams = list(samples.unbind())
    if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
        raise ValueError(
            "MiniMaxH3FreezeAudio expects video [B,24,T,H,W] and audio [B,32,2,T] "
            f"streams, got {[tuple(s.shape) for s in streams]}")
    if streams[0].shape[0] != 1:
        raise ValueError("MiniMaxH3FreezeAudio: batch size 1 only")
    return streams[0], streams[1]


def _stereo(audio, field: str = "audio") -> tuple[torch.Tensor, int]:
    """First batch item as [1, 2, L] float32, mono duplicated, more than two refused."""
    if not isinstance(audio, Mapping):
        raise ValueError(f"{field} must be a Comfy AUDIO value")
    waveform = audio.get("waveform")
    rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor):
        raise ValueError(f"{field}.waveform must be a tensor of shape [batch, channels, samples]")
    if waveform.ndim != 3:
        raise ValueError(f"{field}.waveform must have shape [batch, channels, samples]")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"{field}.sample_rate must be a positive number")
    waveform = waveform[:1].to(torch.float32)
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif channels != 2:
        # Silently taking two of more channels can destroy the mix; the same
        # refusal reference_conditioning._prepare_audio makes.
        raise ValueError(
            f"{field} must be mono or stereo; got {channels} channels. "
            "Downmix before this node.")
    return waveform, int(rate)


def audio_grid(audio_vae) -> tuple[int, int]:
    """(sample rate, samples per latent step) as the loaded audio VAE declares them."""
    rate = int(getattr(audio_vae, "audio_sample_rate", 32000))  # core's own fallback
    hop = int(audio_vae.spacial_compression_encode())
    nominal = rate / AUDIO_LATENT_FPS
    if abs(hop - nominal) > 1e-6:
        raise ValueError(
            f"audio VAE declares {hop} samples per latent step at {rate} Hz, which is "
            f"not the {AUDIO_LATENT_FPS:g} steps/s grid H3's latent runs on")
    return rate, hop


def slice_window(waveform: torch.Tensor, rate: int, audio_vae, start_seconds: float,
                 audio_t: int) -> tuple[torch.Tensor, int, int, int]:
    """Cut exactly `audio_t` latent steps of samples from `start_seconds`, on the grid.

    Returns (slice [1,2,audio_t*hop] at the VAE rate, start step, VAE rate,
    steps of trailing silence that were padded because the song ran out).
    The start snaps to the nearest latent step; the slice length is exact by
    construction, so the VAE's generic input crop cannot move anything.
    """
    vae_rate, hop = audio_grid(audio_vae)
    if rate != vae_rate:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, rate, vae_rate)
    start_step = int(round(float(start_seconds) * AUDIO_LATENT_FPS))
    if start_step < 0:
        raise ValueError(f"start_seconds {start_seconds} is before the song starts")
    want = int(audio_t) * hop
    start = start_step * hop
    total = int(waveform.shape[-1])
    if start >= total:
        raise ValueError(
            f"start_seconds {start_seconds} ({start_step} latent steps) is past the "
            f"end of the audio ({total / vae_rate:.3f}s)")
    piece = waveform[..., start:start + want]
    short = want - int(piece.shape[-1])
    padded_steps = 0
    if short > 0:
        piece = torch.nn.functional.pad(piece, (0, short))
        padded_steps = math.ceil(short / hop)
    return piece.contiguous(), start_step, vae_rate, padded_steps


def freeze_masks(video: torch.Tensor, audio: torch.Tensor, audio_mask: float,
                 existing=None):
    """The nested mask: video ones (or the incoming video mask), audio at `audio_mask`.

    A flat incoming mask is refused: it is what stock SetLatentNoiseMask
    writes, and the sampler would pad a ones mask for the audio stream
    (`comfy/samplers.py::CFGGuider.sample`), un-freezing the track silently.
    """
    video_mask = torch.ones((1, 1) + tuple(video.shape[2:]), dtype=torch.float32)
    if existing is not None:
        if not getattr(existing, "is_nested", False):
            raise ValueError(
                "the incoming latent carries a flat noise_mask, which is what stock "
                "SetLatentNoiseMask writes; on an H3 AV latent the sampler pads a ones "
                "mask for the audio stream and the track regenerates. Remove it, or "
                "feed this node a nested (video, audio) mask.")
        parts = list(existing.unbind())
        if parts and tuple(parts[0].shape[-3:]) == tuple(video.shape[2:]):
            video_mask = parts[0].to(torch.float32).reshape((1, 1) + tuple(video.shape[2:]))
        else:
            raise ValueError(
                "the incoming nested noise_mask's video part does not match the "
                f"video latent: mask {tuple(parts[0].shape) if parts else ()} against "
                f"latent {tuple(video.shape)}")
    audio_part = torch.full((1, 1) + tuple(audio.shape[2:]), float(audio_mask),
                            dtype=torch.float32)
    return comfy.nested_tensor.NestedTensor((video_mask, audio_part))


class MiniMaxH3FreezeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FreezeAudio",
            display_name="MiniMax H3 Freeze Audio",
            category="model/latent/minimax",
            description=(
                "Write a window of a known audio track into the H3 AV latent's audio "
                "rows and freeze them, so the video is generated against that track. "
                "Sits between the conditioner (or preflight) and the sampler. Mux the "
                "clip_audio output, not a decoded audio stream: the latent is a "
                "control signal and the VAE round trip is lossy. docs/h3_audio_freeze.md."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="The joint video+audio latent from the H3 conditioner."),
                io.Vae.Input("audio_vae", tooltip="The H3 audio VAE."),
                io.Audio.Input("audio", tooltip="The track. Mono is duplicated to stereo; more than two channels is refused."),
                io.Float.Input("start_seconds", default=0.0, min=0.0, max=36000.0, step=0.025,
                               tooltip=("Where in the track this window starts. Snaps to the "
                                        "audio latent grid (one step per 1/40 s). For a loop, "
                                        "advance it by the window minus the context each pass.")),
                io.Float.Input("audio_mask", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip=("Sampler mask value for the audio rows. 0.0 freezes the "
                                        "track (rows at the clean timestep every step). Above zero "
                                        "lets the model own the rows a little (1 - m * sigma). "
                                        "1.0 is no freeze; values within 0.001 of 1.0 are dropped "
                                        "by core and behave as 1.0.")),
            ],
            outputs=[
                io.Latent.Output(display_name="latent",
                                 tooltip="The latent with the track written in and a nested noise_mask attached."),
                io.Audio.Output(display_name="clip_audio",
                                tooltip="Exactly the window that was encoded, at the VAE's rate. Wire this to the muxer."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, audio_vae, audio, start_seconds, audio_mask) -> io.NodeOutput:
        video, target_audio = _av_streams(latent["samples"])
        audio_t = int(target_audio.shape[-1])
        waveform, rate = _stereo(audio)
        piece, start_step, vae_rate, padded_steps = slice_window(
            waveform, rate, audio_vae, start_seconds, audio_t)

        # Core's own call shape: VAE.encode takes [B, samples, channels].
        z = audio_vae.encode(piece.movedim(1, -1))
        if tuple(z.shape[:3]) != tuple(target_audio.shape[:3]) or int(z.shape[-1]) != audio_t:
            raise ValueError(
                f"audio VAE returned {tuple(z.shape)} for a {audio_t}-step window; the "
                f"target audio latent is {tuple(target_audio.shape)}. The encoder did not "
                "land on the grid this node cut, which is a contract error, not something "
                "to pad over.")
        z = z.to(device=target_audio.device, dtype=target_audio.dtype)

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = freeze_masks(video, z, audio_mask, latent.get("noise_mask"))

        clip_audio = {"waveform": piece, "sample_rate": vae_rate}
        report = (
            f"froze {audio_t} audio latent steps ({audio_t / AUDIO_LATENT_FPS:.3f}s) from "
            f"step {start_step} ({start_step / AUDIO_LATENT_FPS:.3f}s) of the track; "
            f"audio_mask {float(audio_mask):g}; "
            + (f"{padded_steps} trailing steps are silence (the track ran out); "
               if padded_steps else "")
            + f"video mask {'kept from input' if latent.get('noise_mask') is not None else 'all ones'}"
        )
        logger.info("[h3] MiniMaxH3FreezeAudio: %s", report)
        return io.NodeOutput(out, clip_audio, report)
