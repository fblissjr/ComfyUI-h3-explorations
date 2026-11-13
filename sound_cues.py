"""Zero-file procedural sound-island conditioning (`MiniMaxH3SoundCues`).

Synthesizes short `32 kHz` float32 transient cue buffers in memory, encodes each
non-overlapping island through the Audio VAE (`_encode_ref_audio`), and attaches
them to `minimax_keyframes` (`cond_audio`) on `positive` conditioning without
writing `.wav` files to disk or chaining `LoadAudio` + `MiniMaxH3GuideLatent`.

Physics enforced automatically (`docs/wiki/research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`):

1. **Exact integer 40 Hz Audio VAE RoPE positions (`snap_to_3f=True`):**
   `PackedLayout` places each audio island at `cond_t = cursor + (5/3) * frame_idx`.
   Snapping `frame_idx` to a multiple of `3` guarantees exact integer 40 Hz token
   alignment (`25 ms` per token; `3` frames at `24 fps` = `5` audio tokens = `125 ms`).

2. **Automatic compound-cue merging (zero 1D RoPE overlap):**
   Two cues closer than the previous cue's duration (for example `54: snap` +
   `57: slam`, `125 ms` apart while `snap` spans `175 ms` = `7` tokens) would
   otherwise assign duplicate 1D temporal RoPE indices `[95, 97)` across two
   separate `cond_audio` islands. `MiniMaxH3SoundCues` merges touching or
   overlapping cues into a single continuous buffer at the leading anchor frame
   with exact sample offsets, then verifies non-overlap with
   `validate_audio_keyframe_spans`.

3. **Continuous ambient rumble floor (`floor_dbfs=-28.0`):**
   Zero-floor `.wav` islands (`-94 dBFS` tails) teach the Audio DiT that the
   acoustic room floor is digital silence, suppressing inter-event ambience by
   `~20 dB`. Mixing a deterministic `-28 dBFS` low-frequency room/street floor
   into every island preserves natural `-33..-35 dBFS` inter-event ambience.

Pair with `MiniMaxH3ReleaseReferences(audio_end_percent=0.33)` on `BasicGuider`
(`1x` 6-pass cost) or `audio_end_percent=0.60` with `MiniMaxH3EditGuider(streams="video_only", last_step=3)`
so synthetic cue waveforms steer video landing timestamps on early steps without
bleeding into the final soundtrack.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import node_helpers
import numpy as np
import torch
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io, ui
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio

try:
    from . import h3_rules as _rules
    from .guide_latent import _validate_av_latent, validate_audio_keyframe_spans
except ImportError:  # run from the repo root by a check
    import h3_rules as _rules
    from guide_latent import _validate_av_latent, validate_audio_keyframe_spans


SAMPLE_RATE = 32000
VIDEO_FPS = 24.0
AUDIO_TOKEN_HZ = 40.0
SAMPLES_PER_AUDIO_TOKEN = SAMPLE_RATE // int(AUDIO_TOKEN_HZ)  # 800 samples = 25 ms

SOUND_CUE_PRESETS: dict[str, float] = {
    "hop": 0.200,        # 8 tokens (200 ms): body thud + gravel/asphalt footfall
    "thud": 0.200,       # alias for hop
    "snap": 0.175,       # 7 tokens (175 ms): crisp 800-4500 Hz taut leather/fabric snap
    "yank": 0.175,       # alias for snap
    "slam": 0.200,       # 8 tokens (200 ms): heavy 48/96 Hz + 80-1400 Hz ground impact
    "impact": 0.200,     # alias for slam
    "yank_slam": 0.300,  # 12 tokens (300 ms): snap at t=0 + slam at +125 ms (+3 frames)
    "step": 0.125,       # 5 tokens (125 ms): lighter claw/footstep tap
    "click": 0.100,      # 4 tokens (100 ms): crisp mechanical click
}

_CUE_LINE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?s?)\s*(?::|\s+)\s*([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*@\s*([+-]?\d+(?:\.\d+)?))?\s*$"
)


@dataclass(frozen=True)
class ParsedSoundCue:
    raw_time: str
    requested_frame: int
    resolved_frame: int
    preset: str
    gain: float
    duration_sec: float


@dataclass(frozen=True)
class MergedSoundIsland:
    anchor_frame: int
    cues: tuple[ParsedSoundCue, ...]
    duration_sec: float

    @property
    def start_token(self) -> float:
        return float(FRAME_RESCALE) * self.anchor_frame

    @property
    def est_tokens(self) -> int:
        return max(1, int(math.ceil(self.duration_sec * AUDIO_TOKEN_HZ - 1e-9)))


def _bandpass_fft(x: np.ndarray, sr: int, f_lo: float, f_hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mask = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float64)
    if len(mask) > 4:
        mask = np.convolve(mask, np.hanning(5) / np.sum(np.hanning(5)), mode="same")
    return np.fft.irfft(spec * mask, n=len(x))


def _ambient_floor(n: int, sr: int, rng: np.random.Generator, target_dbfs: float) -> np.ndarray:
    """Continuous room/street rumble floor so Audio VAE islands never end in -94 dBFS digital zero."""
    if target_dbfs <= -59.5 or n <= 0:
        return np.zeros(n, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / sr
    rumble = 0.55 * np.sin(2.0 * np.pi * 54.0 * t + float(rng.uniform(0.0, 2.0 * np.pi)))
    rumble += 0.30 * np.sin(2.0 * np.pi * 108.0 * t + float(rng.uniform(0.0, 2.0 * np.pi)))
    tex = _bandpass_fft(rng.standard_normal(n), sr, 75.0, 420.0)
    tex /= max(float(np.std(tex)), 1e-8)
    bed = 0.60 * rumble + 0.40 * tex
    bed_rms = max(float(np.sqrt(np.mean(bed ** 2))), 1e-8)
    want_rms = 10.0 ** (float(target_dbfs) / 20.0)
    return bed * (want_rms / bed_rms)


def _render_preset_transient(preset: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    if preset in ("hop", "thud"):
        f0 = 55.0 + 95.0 * np.exp(-t / 0.020)
        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        thump = np.sin(phase) * np.exp(-t / 0.040) * 0.72
        thump += 0.30 * np.sin(2.0 * phase) * np.exp(-t / 0.022)
        scrape = _bandpass_fft(rng.standard_normal(n), sr, 90.0, 900.0)
        scrape /= max(float(np.max(np.abs(scrape))), 1e-8)
        scrape *= np.exp(-t / 0.028) * 0.48
        raw = thump + scrape
    elif preset in ("snap", "yank"):
        snap_noise = _bandpass_fft(rng.standard_normal(n), sr, 800.0, 4500.0)
        snap_noise /= max(float(np.max(np.abs(snap_noise))), 1e-8)
        snap_env = np.exp(-t / 0.011)
        f_whip = 420.0 + 1150.0 * np.exp(-t / 0.009)
        whip = np.sin(2.0 * np.pi * np.cumsum(f_whip) / sr) * np.exp(-t / 0.014) * 0.50
        raw = 0.78 * snap_noise * snap_env + whip
    elif preset in ("slam", "impact"):
        f_slam = 48.0 + 135.0 * np.exp(-t / 0.022)
        p_slam = 2.0 * np.pi * np.cumsum(f_slam) / sr
        slam_body = (np.sin(p_slam) + 0.35 * np.sin(2.0 * p_slam)) * np.exp(-t / 0.050) * 0.80
        slam_grit = _bandpass_fft(rng.standard_normal(n), sr, 80.0, 1400.0)
        slam_grit /= max(float(np.max(np.abs(slam_grit))), 1e-8)
        slam_grit *= np.exp(-t / 0.036) * 0.62
        raw = slam_body + slam_grit
    elif preset == "yank_slam":
        raw = np.zeros(n, dtype=np.float64)
        n_snap = min(n, int(round(0.175 * sr)))
        raw[:n_snap] += _render_preset_transient("snap", n_snap, sr, rng)
        slam_off = int(round((3.0 / VIDEO_FPS) * sr))  # +3 frames = +125 ms
        if slam_off < n:
            n_slam = n - slam_off
            raw[slam_off:] += _render_preset_transient("slam", n_slam, sr, rng)
        return raw
    elif preset == "step":
        f0 = 95.0 + 140.0 * np.exp(-t / 0.012)
        tap = np.sin(2.0 * np.pi * np.cumsum(f0) / sr) * np.exp(-t / 0.020) * 0.52
        grit = _bandpass_fft(rng.standard_normal(n), sr, 180.0, 1600.0)
        grit /= max(float(np.max(np.abs(grit))), 1e-8)
        raw = tap + grit * np.exp(-t / 0.018) * 0.45
    elif preset == "click":
        click = _bandpass_fft(rng.standard_normal(n), sr, 1200.0, 5000.0)
        click /= max(float(np.max(np.abs(click))), 1e-8)
        raw = click * np.exp(-t / 0.007) * 0.72
    else:
        raise ValueError(f"MiniMaxH3SoundCues: unknown preset {preset!r}")

    fade_in = min(48, n)
    if fade_in > 1:
        raw[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    return raw


def parse_sound_cues(
    cues_text: str,
    frame_count: int,
    *,
    snap_to_3f: bool = True,
) -> list[ParsedSoundCue]:
    """Parse multiline cue beat-sheet lines into frame-resolved `ParsedSoundCue` entries."""
    if frame_count < 2:
        raise ValueError(
            f"MiniMaxH3SoundCues requires a multi-frame video latent (got frame_count={frame_count})"
        )
    if not isinstance(cues_text, str) or not cues_text.strip():
        raise ValueError("MiniMaxH3SoundCues: `cues` beat sheet is empty")

    parsed: list[tuple[int, int, ParsedSoundCue]] = []
    for line_no, raw_line in enumerate(cues_text.splitlines(), 1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        m = _CUE_LINE_RE.match(stripped)
        if m is None:
            raise ValueError(
                f"MiniMaxH3SoundCues line {line_no}: could not parse {raw_line!r}; "
                f"expected `<frame_or_seconds>: <preset> [@gain]`, e.g. `24: hop` or `2.25s: slam @1.2`"
            )
        time_tok, preset_raw, gain_raw = m.group(1), m.group(2).lower(), m.group(3)
        if preset_raw not in SOUND_CUE_PRESETS:
            raise ValueError(
                f"MiniMaxH3SoundCues line {line_no}: unknown preset {preset_raw!r}; "
                f"expected one of {sorted(SOUND_CUE_PRESETS)}"
            )
        gain = 1.0 if gain_raw is None else float(gain_raw)
        if not (0.05 <= gain <= 2.5):
            raise ValueError(
                f"MiniMaxH3SoundCues line {line_no}: gain {gain} outside [0.05, 2.5]"
            )

        if time_tok.endswith("s"):
            sec = float(time_tok[:-1])
            if sec < 0.0:
                raise ValueError(
                    f"MiniMaxH3SoundCues line {line_no}: negative seconds {time_tok!r} are not supported"
                )
            req_frame = int(round(sec * VIDEO_FPS))
        else:
            if "." in time_tok:
                raise ValueError(
                    f"MiniMaxH3SoundCues line {line_no}: fractional time {time_tok!r} requires an `s` suffix (e.g. `{time_tok}s`)"
                )
            f_val = int(time_tok)
            req_frame = f_val if f_val >= 0 else frame_count + f_val

        if req_frame < 0 or req_frame >= frame_count:
            raise ValueError(
                f"MiniMaxH3SoundCues line {line_no}: cue frame {req_frame} ({time_tok!r}) "
                f"is outside [0, {frame_count - 1}]"
            )

        if snap_to_3f:
            max_multiple = ((frame_count - 1) // 3) * 3
            resolved = min(max_multiple, max(0, int(round(req_frame / 3.0)) * 3))
        else:
            resolved = req_frame

        cue = ParsedSoundCue(
            raw_time=time_tok,
            requested_frame=req_frame,
            resolved_frame=resolved,
            preset=preset_raw,
            gain=gain,
            duration_sec=SOUND_CUE_PRESETS[preset_raw],
        )
        parsed.append((resolved, line_no, cue))

    if not parsed:
        raise ValueError("MiniMaxH3SoundCues: `cues` contains only comments or whitespace")

    parsed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in parsed]


def merge_sound_cue_islands(cues: list[ParsedSoundCue]) -> list[MergedSoundIsland]:
    """Merge touching or overlapping cues into single compound islands so 1D RoPE never collides."""
    if not cues:
        return []
    islands: list[MergedSoundIsland] = []
    cur_anchor = cues[0].resolved_frame
    cur_cues: list[ParsedSoundCue] = [cues[0]]
    cur_end_sec = cur_anchor / VIDEO_FPS + cues[0].duration_sec

    for cue in cues[1:]:
        cue_start_tok = float(FRAME_RESCALE) * cue.resolved_frame
        cur_dur_sec = cur_end_sec - (cur_anchor / VIDEO_FPS)
        cur_end_tok = float(FRAME_RESCALE) * cur_anchor + math.ceil(cur_dur_sec * AUDIO_TOKEN_HZ - 1e-9)
        if cue_start_tok <= cur_end_tok + 1e-6:
            cur_cues.append(cue)
            cur_end_sec = max(cur_end_sec, cue.resolved_frame / VIDEO_FPS + cue.duration_sec)
        else:
            total_dur = round((cur_end_sec - cur_anchor / VIDEO_FPS) * AUDIO_TOKEN_HZ) / AUDIO_TOKEN_HZ
            islands.append(
                MergedSoundIsland(
                    anchor_frame=cur_anchor,
                    cues=tuple(cur_cues),
                    duration_sec=max(0.050, total_dur),
                )
            )
            cur_anchor = cue.resolved_frame
            cur_cues = [cue]
            cur_end_sec = cur_anchor / VIDEO_FPS + cue.duration_sec

    total_dur = round((cur_end_sec - cur_anchor / VIDEO_FPS) * AUDIO_TOKEN_HZ) / AUDIO_TOKEN_HZ
    islands.append(
        MergedSoundIsland(
            anchor_frame=cur_anchor,
            cues=tuple(cur_cues),
            duration_sec=max(0.050, total_dur),
        )
    )
    return islands


def synthesize_island_waveform(
    island: MergedSoundIsland,
    *,
    floor_dbfs: float = -28.0,
    seed: int = 20261112,
    island_index: int = 0,
) -> torch.Tensor:
    """Synthesize a stereo `[1, 2, samples]` float32 waveform at `32 kHz` for one merged island."""
    n_samples = max(SAMPLES_PER_AUDIO_TOKEN, int(round(island.duration_sec * SAMPLE_RATE)))
    rng = np.random.default_rng(int(seed) + 1009 * int(island_index) + 37 * int(island.anchor_frame))
    mix = np.zeros(n_samples, dtype=np.float64)

    for cue in island.cues:
        offset = int(round(((cue.resolved_frame - island.anchor_frame) / VIDEO_FPS) * SAMPLE_RATE))
        cue_len = max(1, int(round(cue.duration_sec * SAMPLE_RATE)))
        avail = max(0, n_samples - offset)
        if avail <= 0:
            continue
        seg_len = min(cue_len, avail)
        transient = _render_preset_transient(cue.preset, seg_len, SAMPLE_RATE, rng) * float(cue.gain)
        mix[offset : offset + seg_len] += transient

    floor_wave = _ambient_floor(n_samples, SAMPLE_RATE, rng, float(floor_dbfs))
    fade_out = min(int(0.015 * SAMPLE_RATE), n_samples // 4)
    if fade_out > 1:
        # Smooth only the transient tail while keeping the ambient floor continuous to the boundary.
        mix[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    combined = np.tanh((mix + floor_wave) * 1.15) * 0.88
    mono = torch.from_numpy(combined.astype(np.float32))
    return torch.stack([mono, mono], dim=0).unsqueeze(0).contiguous()


class MiniMaxH3SoundCues(io.ComfyNode):
    """Procedural zero-file timeline sound-island conditioning (`minimax_keyframes` `cond_audio`)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SoundCues",
            display_name="MiniMax H3 Sound Cues",
            category="MiniMaxH3",
            description=(
                "Generate procedural short-sound (`cond_audio`) islands in memory from a "
                "frame/second beat sheet, snap to exact integer 40 Hz Audio VAE RoPE tokens, "
                "auto-merge close cues so 1D RoPE positions never collide, and mix a `-28 dBFS` "
                "ambient rumble floor so inter-event room tone is never muted. Pair with "
                "`MiniMaxH3ReleaseReferences(audio_end_percent=0.33)` on `BasicGuider` or "
                "`audio_end_percent=0.60` with `MiniMaxH3EditGuider(streams='video_only', last_step=3)`."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 Audio VAE (`minimax_h3_audio_vae_fp32.safetensors`)."),
                io.String.Input(
                    "cues",
                    multiline=True,
                    default=_rules.SOUND_CUES_DEFAULTS["cues"],
                    tooltip=(
                        "One cue per line: `<frame_or_seconds>: <preset> [@gain]`. "
                        "Presets: `hop`/`thud`, `snap`/`yank`, `slam`/`impact`, `yank_slam`, "
                        "`step`, `click`. Note: `cond_audio` locks landing/impact time, so "
                        "place `hop` on the desired touchdown frame (H3 initiates liftoff "
                        "~6-8 frames earlier and reaches apex ~3-5 frames earlier)."
                    ),
                ),
                io.Boolean.Input(
                    "snap_to_3f",
                    default=_rules.SOUND_CUES_DEFAULTS["snap_to_3f"],
                    optional=True,
                    tooltip=(
                        "Snap cue frames to multiples of 3 so `frame_idx * 5 / 3` lands on "
                        "an exact integer 40 Hz Audio VAE RoPE token."
                    ),
                ),
                io.Float.Input(
                    "floor_dbfs",
                    default=_rules.SOUND_CUES_DEFAULTS["floor_dbfs"],
                    min=-60.0,
                    max=-12.0,
                    step=0.5,
                    optional=True,
                    tooltip=(
                        "Continuous low-frequency ambient floor mixed into each island (dBFS). "
                        "`-28.0` prevents `-94 dBFS` digital-zero tails from muting inter-event ambience; "
                        "`-60.0` disables the floor."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=_rules.SOUND_CUES_DEFAULTS["seed"],
                    min=0,
                    max=0x7FFFFFFF,
                    optional=True,
                    tooltip="Deterministic RNG seed for procedural noise-band synthesis.",
                ),
                io.Conditioning.Input(
                    "edited_positive",
                    optional=True,
                    tooltip=(
                        "Optional second prompt conditioning for prompt-delta Edit Guidance. "
                        "When wired, all synthesized cue islands are encoded once through the "
                        "Audio VAE and attached to both `positive` (slot 0) and `edited_positive` (slot 2)."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Conditioning.Output(display_name="edited_positive"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        latent,
        audio_vae,
        cues: str = _rules.SOUND_CUES_DEFAULTS["cues"],
        snap_to_3f: bool = _rules.SOUND_CUES_DEFAULTS["snap_to_3f"],
        floor_dbfs: float = _rules.SOUND_CUES_DEFAULTS["floor_dbfs"],
        seed: int = _rules.SOUND_CUES_DEFAULTS["seed"],
        edited_positive=None,
    ) -> io.NodeOutput:
        video, audio_tensor = _validate_av_latent(latent)
        if audio_vae is None:
            raise ValueError("MiniMaxH3SoundCues requires `audio_vae`")
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(int(video.shape[2])))
        if frame_count < 2:
            raise ValueError(
                "MiniMaxH3SoundCues requires a multi-frame video latent (single-frame latents carry no timeline audio)"
            )
        total_audio_tokens = int(audio_tensor.shape[-1])

        parsed = parse_sound_cues(cues, frame_count, snap_to_3f=bool(snap_to_3f))
        islands = merge_sound_cue_islands(parsed)

        added_keyframes: list[dict] = []
        report_lines = [
            f"sound cues: {len(parsed)} cue(s) -> {len(islands)} non-overlapping cond_audio island(s) "
            f"(frame_count={frame_count}, audio_tokens={total_audio_tokens}, floor={float(floor_dbfs):.1f} dBFS)"
        ]
        for idx, island in enumerate(islands):
            max_rt = math.floor(total_audio_tokens - FRAME_RESCALE * island.anchor_frame)
            if max_rt < 1:
                raise ValueError(
                    f"MiniMaxH3SoundCues: cue at frame {island.anchor_frame} is past the end of "
                    f"the {total_audio_tokens}-token audio latent"
                )
            waveform = synthesize_island_waveform(
                island,
                floor_dbfs=float(floor_dbfs),
                seed=int(seed),
                island_index=idx,
            )
            audio_latent, audio_rt = _encode_ref_audio(
                audio_vae,
                {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            )
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
                audio_rt = int(audio_latent.shape[-1])
            added_keyframes.append(
                {
                    "resolved_frame_index": island.anchor_frame,
                    "audio_latent": audio_latent,
                }
            )
            t0 = float(FRAME_RESCALE) * island.anchor_frame
            t1 = t0 + float(audio_rt)
            cue_desc = " + ".join(
                f"{c.preset}@f{c.resolved_frame}"
                + (f"(snapped from {c.requested_frame})" if c.resolved_frame != c.requested_frame else "")
                + (f"x{c.gain:g}" if abs(c.gain - 1.0) > 1e-6 else "")
                for c in island.cues
            )
            report_lines.append(
                f"  [{idx + 1}] frame {island.anchor_frame:>3d} ({island.anchor_frame / VIDEO_FPS:5.2f}s) | "
                f"RoPE [{t0:5.1f}, {t1:5.1f}) ({audio_rt:>2d} tok) | {cue_desc}"
            )

        keyframes = list(positive[0][1].get("minimax_keyframes", [])) + added_keyframes
        validate_audio_keyframe_spans(keyframes, caller="MiniMaxH3SoundCues")
        out_positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        out_edited = None
        if edited_positive is not None:
            edited_keyframes = (
                list(edited_positive[0][1].get("minimax_keyframes", [])) + added_keyframes
            )
            validate_audio_keyframe_spans(edited_keyframes, caller="MiniMaxH3SoundCues")
            out_edited = node_helpers.conditioning_set_values(
                edited_positive, {"minimax_keyframes": edited_keyframes}
            )
        return io.NodeOutput(
            out_positive,
            latent,
            out_edited,
            ui=ui.PreviewText("\n".join(report_lines)),
        )
