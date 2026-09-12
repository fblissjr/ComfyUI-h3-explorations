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

**What it fixes, and says so.** The vendor's path takes float PCM in
[-1, 1] at the VAE's rate, stereo, with no gain. This node resamples with
the vendor's resampler, duplicates mono, downmixes more than two channels
with ffmpeg's default coefficients under the standard channel order for
that count (a ComfyUI AUDIO tensor carries no layout, so the order is
assumed and reported), removes a DC offset, and pulls a clipped window back
under full scale (`level`, default `clip_guard`, which changes a well-formed
file by nothing). Every transform is named in the report.

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
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS, video_latent_t

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


# ffmpeg's default channel orders and stereo downmix, the ones sglang's
# `-ac 2` applies from the file's own layout. A ComfyUI AUDIO tensor carries
# no layout, so the order is assumed by channel count and the report says so.
# Coefficients: front pair straight, centre and rears at 1/sqrt(2), LFE
# dropped, then scaled so the largest gain sum is one (ffmpeg's rematrix
# normalisation). 3 = FL FR FC, 4 = FL FR BL BR, 5 = FL FR FC BL BR,
# 6 = FL FR FC LFE BL BR, 8 = FL FR FC LFE BL BR SL SR.
_SIDE = 0.7071067811865476
_DOWNMIX = {
    3: ("FL FR FC", [(1.0, 0.0, _SIDE)], [(0.0, 1.0, _SIDE)]),
    4: ("FL FR BL BR", [(1.0, 0.0, _SIDE, 0.0)], [(0.0, 1.0, 0.0, _SIDE)]),
    5: ("FL FR FC BL BR", [(1.0, 0.0, _SIDE, _SIDE, 0.0)], [(0.0, 1.0, _SIDE, 0.0, _SIDE)]),
    6: ("FL FR FC LFE BL BR", [(1.0, 0.0, _SIDE, 0.0, _SIDE, 0.0)], [(0.0, 1.0, _SIDE, 0.0, 0.0, _SIDE)]),
    8: ("FL FR FC LFE BL BR SL SR",
        [(1.0, 0.0, _SIDE, 0.0, _SIDE, 0.0, _SIDE, 0.0)],
        [(0.0, 1.0, _SIDE, 0.0, 0.0, _SIDE, 0.0, _SIDE)]),
}


def _stereo(audio, field: str = "audio") -> tuple[torch.Tensor, int, list[str]]:
    """First batch item as [1, 2, L] float32, plus the transforms applied, in words.

    Mono is duplicated (what ffmpeg does for sglang). More than two channels
    is downmixed with ffmpeg's default coefficients under the standard order
    for that count, and the assumed order is reported; a count with no
    standard layout is averaged into both channels.
    """
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
    notes: list[str] = []
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
        notes.append("mono duplicated to both channels")
    elif channels == 2:
        pass
    elif channels in _DOWNMIX:
        order, left, right = _DOWNMIX[channels]
        lw = torch.tensor(left[0], dtype=torch.float32)
        rw = torch.tensor(right[0], dtype=torch.float32)
        norm = max(float(lw.sum()), float(rw.sum()))
        x = waveform[0]  # [C, L]
        mixed = torch.stack([(lw[:, None] * x).sum(0), (rw[:, None] * x).sum(0)]) / norm
        waveform = mixed.unsqueeze(0)
        notes.append(f"{channels} channels downmixed to stereo with ffmpeg's default "
                     f"coefficients, assuming the order {order}")
    else:
        mono = waveform.mean(dim=1, keepdim=True)
        waveform = mono.repeat(1, 2, 1)
        notes.append(f"{channels} channels have no standard layout; averaged into both channels")
    return waveform, int(rate), notes


def condition_level(piece: torch.Tensor, level: str) -> tuple[torch.Tensor, list[str]]:
    """Bring the window into the VAE's input contract, float PCM in [-1, 1], and say what changed.

    `clip_guard`: remove a DC offset above a hundredth of full scale, and
    scale down only when the peak exceeds one. `peak`: also scale so the
    peak sits just under full scale, up or down. `none`: touch nothing.
    The vendor's path applies no gain at all, so `clip_guard` changes a
    well-formed file by nothing.
    """
    notes: list[str] = []
    if level == "none":
        return piece, notes
    dc = float(piece.mean())
    if abs(dc) > 0.01:
        piece = piece - dc
        notes.append(f"DC offset {dc:+.4f} removed")
    peak = float(piece.abs().max()) if piece.numel() else 0.0
    target = 0.999
    if level == "peak" and peak > 1e-6:
        piece = piece * (target / peak)
        notes.append(f"peak {peak:.3f} scaled to {target}")
    elif peak > 1.0:
        piece = piece * (target / peak)
        notes.append(f"peak {peak:.3f} exceeded full scale; scaled to {target}")
    return piece, notes


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
                io.Combo.Input("level", options=["clip_guard", "peak", "none"], default="clip_guard",
                               tooltip=("What to do about the window's level before encoding. clip_guard "
                                        "(default): remove a DC offset and scale down only if the peak "
                                        "exceeds full scale, so a well-formed file is untouched, as in the "
                                        "vendor's path. peak: also scale the peak to just under full scale. "
                                        "none: encode as given. The report says what happened.")),
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
    def execute(cls, latent, audio_vae, audio, start_seconds, audio_mask, level="clip_guard") -> io.NodeOutput:
        video, target_audio = _av_streams(latent["samples"])
        audio_t = int(target_audio.shape[-1])
        in_channels = int(audio["waveform"].shape[1]) if isinstance(audio, Mapping) and isinstance(audio.get("waveform"), torch.Tensor) else -1
        waveform, rate, fixes = _stereo(audio)
        piece, start_step, vae_rate, padded_steps = slice_window(
            waveform, rate, audio_vae, start_seconds, audio_t)
        piece, level_fixes = condition_level(piece, level)
        fixes += level_fixes
        # What the VAE is about to see, stated so a wrong input cannot pass
        # quietly: the vendor's path takes float PCM in [-1, 1] at 32 kHz,
        # stereo, with no gain or normalisation (docs/h3_audio_freeze.md
        # section 9). Anything outside that is reported, not corrected.
        peak = float(piece.abs().max()) if piece.numel() else 0.0
        rms = float(piece.pow(2).mean().sqrt()) if piece.numel() else 0.0
        level_note = ""
        if peak > 1.0:
            level_note = f" PEAK {peak:.3f} still exceeds 1.0 (level={level}): the VAE was trained on [-1, 1];"
        elif peak < 1e-4:
            level_note = " the window is silent;"

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
            + f"video mask {'kept from input' if latent.get('noise_mask') is not None else 'all ones'}; "
            + f"source {rate} Hz {in_channels} ch -> {vae_rate} Hz stereo "
            + ("(resampled, torchaudio sinc at its defaults, the vendor's resampler)" if rate != vae_rate else "(no resample)")
            + ("; " + "; ".join(fixes) if fixes else "")
            + f"; peak {peak:.3f} rms {rms:.4f};" + level_note
        )
        logger.info("[h3] MiniMaxH3FreezeAudio: %s", report)
        return io.NodeOutput(out, clip_audio, report)


# ---------------------------------------------------------------------------
# The loop: one window of a long track, with the previous window's tail frozen
# in as context. The LTX pack's geometry (window, context, stride) on H3's grid.

def pixel_frames(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t)))


def window_geometry(latent_t: int, context_frames: int) -> dict:
    """Frames, context steps and stride for one window, or raise naming the rule broken.

    Three rules, all derived from core's grid and stated in
    docs/h3_audio_freeze.md section 4 step 6:
    - the context is a valid video run (17k + 5 frames) so it maps to whole
      latent steps;
    - it lands on the audio grid (frames * 5/3 whole), so the seam is at an
      audio step: 39, 90, 141, ... frames (39 + 51k);
    - the previous window's tail starts on the run pattern's phase
      ((latent_t - context_steps) % 5 == 0), or the copied latents would
      describe 1-frame and 4-frame steps in the wrong order.
    """
    frames = pixel_frames(latent_t)
    c = int(context_frames)
    if c < 0:
        raise ValueError("context_frames must be non-negative")
    if c == 0:
        return {"frames": frames, "context_frames": 0, "context_steps": 0,
                "stride_frames": frames, "stride_seconds": frames / FPS}
    if c % 17 != 5:
        raise ValueError(f"context_frames {c} is not a video run (17k + 5): 39, 56, 73, ...")
    if (c * 5) % 3 != 0:
        raise ValueError(f"context_frames {c} does not land on the audio grid; use 39 + 51k: 39, 90, 141, ...")
    steps = video_latent_t(c)
    if steps >= latent_t:
        raise ValueError(f"context {c} frames ({steps} latent steps) is not shorter than the window ({latent_t} steps)")
    if (latent_t - steps) % len(FRAME_PER_TOKEN) != 0:
        raise ValueError(
            f"a {c}-frame context ({steps} steps) does not start on the run phase of a "
            f"{latent_t}-step window; the tail would be copied out of phase")
    stride = frames - c
    if (stride * 5) % 3 != 0:
        raise ValueError(f"stride {stride} frames does not land on the audio grid")
    return {"frames": frames, "context_frames": c, "context_steps": steps,
            "stride_frames": stride, "stride_seconds": stride / FPS}


class MiniMaxH3FreezeAudioWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FreezeAudioWindow",
            display_name="MiniMax H3 Freeze Audio Window (loop)",
            category="model/latent/minimax",
            description=(
                "One window of a long track: freezes the track's slice for this window "
                "(start = window_index * (window - context)), and, from the second window on, "
                "copies the previous window's video latent tail into this window's head and "
                "freezes it as context. Chain one per window, each feeding its own sampler; "
                "drop trim_frames from each decoded window after the first and concatenate. "
                "span_audio is the track from the start to the end of this window, for the muxer. "
                "docs/h3_audio_freeze.md section 4 step 6."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="A fresh joint video+audio latent for this window, from the H3 conditioner."),
                io.Vae.Input("audio_vae"),
                io.Audio.Input("audio", tooltip="The whole track."),
                io.Int.Input("window_index", default=0, min=0, max=999,
                             tooltip="0 for the first window. The song start is window_index * stride."),
                io.Int.Input("context_frames", default=39, min=0, max=999,
                             tooltip=("Frames of the previous window kept as frozen context at the head of "
                                      "this one. Must be a video run that lands on the audio grid: 39, 90, 141 "
                                      "(39 + 51k). 0 disables context. Ignored on window 0.")),
                io.Latent.Input("previous", optional=True,
                                tooltip="The previous window's SAMPLED latent (the sampler's output). Required from window 1 on."),
                io.Float.Input("audio_mask", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("level", options=["clip_guard", "peak", "none"], default="clip_guard"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Audio.Output(display_name="clip_audio", tooltip="This window's slice, at the VAE's rate."),
                io.Audio.Output(display_name="span_audio", tooltip="The track from its start to the end of this window, for the muxer of the concatenated video."),
                io.Int.Output(display_name="trim_frames", tooltip="Frames to drop from the head of this window's decode: context_frames after window 0, else 0."),
                io.Float.Output(display_name="start_seconds"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, audio_vae, audio, window_index, context_frames, previous=None,
                audio_mask=0.0, level="clip_guard") -> io.NodeOutput:
        video, target_audio = _av_streams(latent["samples"])
        latent_t = int(video.shape[2])
        audio_t = int(target_audio.shape[-1])
        geo = window_geometry(latent_t, context_frames if window_index > 0 else 0)
        start_seconds = float(window_index) * geo["stride_seconds"]

        waveform, rate, fixes = _stereo(audio)
        piece, start_step, vae_rate, padded_steps = slice_window(
            waveform, rate, audio_vae, start_seconds, audio_t)
        piece, level_fixes = condition_level(piece, level)
        fixes += level_fixes
        z = audio_vae.encode(piece.movedim(1, -1))
        if tuple(z.shape[:3]) != tuple(target_audio.shape[:3]) or int(z.shape[-1]) != audio_t:
            raise ValueError(
                f"audio VAE returned {tuple(z.shape)} for a {audio_t}-step window; the target "
                f"audio latent is {tuple(target_audio.shape)}")
        z = z.to(device=target_audio.device, dtype=target_audio.dtype)

        video = video.clone()
        video_mask = torch.ones((1, 1) + tuple(video.shape[2:]), dtype=torch.float32)
        c = geo["context_steps"]
        if window_index > 0 and c > 0:
            if previous is None:
                raise ValueError(f"window {window_index} needs the previous window's sampled latent on `previous`")
            prev_video, _prev_audio = _av_streams(previous["samples"])
            if tuple(prev_video.shape[1:]) != tuple(video.shape[1:]):
                raise ValueError(
                    f"previous window's video latent {tuple(prev_video.shape)} does not match this "
                    f"window's {tuple(video.shape)}: every window must share canvas and length")
            video[:, :, :c] = prev_video[:, :, latent_t - c:].to(device=video.device, dtype=video.dtype)
            video_mask[:, :, :c] = 0.0
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        audio_part = torch.full((1, 1) + tuple(z.shape[2:]), float(audio_mask), dtype=torch.float32)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_part))

        # the track from 0 to the end of this window, on the same grid, for the muxer
        vae_rate2, hop = audio_grid(audio_vae)
        full = waveform
        if rate != vae_rate2:
            import torchaudio
            full = torchaudio.functional.resample(full, rate, vae_rate2)
        end = (start_step + audio_t) * hop
        span = full[..., :end]
        if span.shape[-1] < end:
            span = torch.nn.functional.pad(span, (0, end - span.shape[-1]))
        span, _ = condition_level(span.contiguous(), level)

        trim = int(geo["context_frames"]) if window_index > 0 else 0
        report = (
            f"window {window_index}: {geo['frames']} frames, context {geo['context_frames']} frames "
            f"({c} latent steps{' copied from the previous window and frozen' if window_index > 0 and c else ''}), "
            f"stride {geo['stride_frames']} frames; track from step {start_step} ({start_seconds:.3f}s), "
            f"{audio_t} steps frozen at mask {float(audio_mask):g}"
            + (f", {padded_steps} trailing steps silent" if padded_steps else "")
            + f"; trim {trim} frames from this window's decode; source {rate} Hz -> {vae_rate} Hz"
            + ("; " + "; ".join(fixes) if fixes else "")
        )
        logger.info("[h3] MiniMaxH3FreezeAudioWindow: %s", report)
        return io.NodeOutput(out, {"waveform": piece, "sample_rate": vae_rate},
                             {"waveform": span, "sample_rate": vae_rate}, trim, start_seconds, report)

