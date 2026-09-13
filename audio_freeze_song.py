"""Drop a song and a prompt; the windows come from the track.

`docs/h3_audio_freeze.md` section 4 step 6, the other mode of the loop. The
shot-per-window chain (`MiniMaxH3FreezeAudioWindow` groups joined by
`MiniMaxH3JoinWindows`) is a director's cut: the frames are the cut and are
typed per window. This node is the LTX pack's music-video mode on H3: one
prompt (or a few, separated by `---`), one track, and the node plans the
windows from the track's length, runs them one after another inside itself,
writes each window's new frames to a file as it goes, and joins the files with
the full track at the end. Nothing here holds more than one window of frames.

**The plan.** Windows are `window_frames` long with `context_frames` of the
previous window frozen at their head, so each adds `window - context` frames.
The last window is the smallest length on both clocks (141, 192, 243, 294,
345 frames) that reaches the end of the track; if the track runs out inside
it, the freeze pads silence and the report says how much. `max_seconds` caps
the plan for a quick look.

**The prompt.** One block is used for every window (`uniform`). Several
blocks separated by a line of `---` are used in order with the last one
repeating (`cycle`, the default when there are several), or one is drawn per
window from the seed (`random`); a block may start with a line `frames: N` to
set that window's length (on both clocks).

**Sampling** is what SamplerCustomAdvanced does, per window: a BasicGuider on
the model with the window's conditioning, prepared noise at `seed + i`, the
given sampler and sigmas, the nested noise mask from the window node. The
conditioning is this pack's own `MiniMaxH3Conditioning`, called directly.

Untested live on the night it was written (2026-09-12); its planner and
prompt parser are checked on CPU. A new tool's first run is a throwaway.
"""

from __future__ import annotations

import logging
import math
import os
import random
import subprocess

import torch
from comfy_api.latest import io

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic
from comfy_extras.nodes_minimax_h3 import FPS, video_latent_t

from .audio_freeze import (MiniMaxH3EncodeTrack, MiniMaxH3FreezeAudioWindow,
                           _ffmpeg, _stereo, _write_wav, audio_grid)
from .conditioning import MiniMaxH3Conditioning

logger = logging.getLogger(__name__)

# Lengths on both clocks: video runs (17k + 5) whose frame count is a multiple
# of 3, so the audio slice is whole latent steps. 39 + 51k.
CHAIN_LENGTHS = (141, 192, 243, 294, 345)


def plan_windows(total_frames: int, window_frames: int, context_frames: int, rng=None) -> list[int]:
    """Frame counts per window covering `total_frames`, each on both clocks.

    With `rng`, every window but the last draws its length from the lengths
    on both clocks at or below `window_frames` (and longer than the context),
    so the same prompt can be tested under uniform and non-uniform windows.
    """
    if window_frames not in CHAIN_LENGTHS:
        raise ValueError(f"window_frames {window_frames} is not on both clocks; use one of {CHAIN_LENGTHS}")
    if context_frames and (context_frames % 17 != 5 or (context_frames * 5) % 3 != 0 or context_frames >= window_frames):
        raise ValueError(f"context_frames {context_frames} must be 39, 90 or 141 and shorter than the window")
    if total_frames <= 0:
        raise ValueError("the track is empty")
    choices = [n for n in CHAIN_LENGTHS if n <= window_frames and n > context_frames]
    first = rng.choice(choices) if rng is not None else window_frames
    plan = [first]
    covered = first
    while covered < total_frames:
        remaining = total_frames - covered
        fit = [n for n in CHAIN_LENGTHS if n > context_frames and n - context_frames >= remaining]
        if fit:
            n = min(fit)
        elif rng is not None:
            n = rng.choice(choices)
        else:
            n = window_frames
        plan.append(n)
        covered += n - context_frames
    return plan


def parse_prompt_blocks(text: str) -> list[tuple[int | None, str]]:
    """Blocks separated by a `---` line; an optional leading `frames: N` line per block."""
    blocks = []
    for raw in text.replace("\r\n", "\n").split("\n---\n"):
        body = raw.strip("\n")
        if not body.strip():
            continue
        frames = None
        first, _, rest = body.partition("\n")
        if first.strip().lower().startswith("frames:"):
            frames = int(first.split(":", 1)[1].strip())
            body = rest
        blocks.append((frames, body.strip("\n")))
    if not blocks:
        raise ValueError("the prompt is empty")
    return blocks


def _write_frames_mp4(path: str, images: torch.Tensor, crf: int) -> int:
    """[T, H, W, 3] float in [0, 1] to an H.264 mp4 through ffmpeg's rawvideo pipe."""
    t, h, w, _ = images.shape
    cmd = [_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for i in range(0, t, 24):
            chunk = (images[i:i + 24].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy().tobytes()
            proc.stdin.write(chunk)
    finally:
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {err.decode(errors='replace')[-400:]}")
    return int(t)


class MiniMaxH3AudioFreezeSong(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioFreezeSong",
            is_output_node=True,
            display_name="MiniMax H3 Audio Freeze Song (whole track)",
            category="model/latent/minimax",
            description=(
                "Drop a track and a prompt: the node plans windows from the track's length, "
                "runs them in sequence with the previous window's tail frozen as context and "
                "the track's slice frozen in each, writes each window's new frames to a file "
                "as it goes, and joins the files with the full track. One prompt for every "
                "window, or blocks separated by a `---` line. docs/h3_audio_freeze.md."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Vae.Input("audio_vae"),
                io.Audio.Input("audio", tooltip="The whole track."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="One prompt for every window, or blocks separated by a line of ---; a block may start with `frames: N`."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("window_frames", default=345, min=141, max=345, step=51,
                             tooltip="Frames per window, on both clocks: 141, 192, 243, 294 or 345."),
                io.Int.Input("context_frames", default=39, min=0, max=141, step=51,
                             tooltip="Frames of the previous window frozen at the head of the next: 39, 90 or 141."),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=36000.0, step=0.5,
                               tooltip="Cover only this much of the track (0 = all of it), for a quick look."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Float.Input("audio_mask", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("level", options=["clip_guard", "peak", "none"], default="clip_guard"),
                io.String.Input("filename_prefix", default="Video/h3_song"),
                io.Int.Input("crf", default=19, min=0, max=51),
                io.Combo.Input("prompt_mode", options=["cycle", "uniform", "random"], default="cycle",
                               tooltip=("How prompt blocks map to windows. cycle: in order, the last repeating. "
                                        "uniform: the first block for every window. random: one block per window, "
                                        "drawn from the seed. With one block all three are the same.")),
                io.Combo.Input("window_mode", options=["uniform", "random"], default="uniform",
                               tooltip=("uniform: every window is window_frames (the last sized to reach the end). "
                                        "random: each window's length drawn from the seed among the lengths on both "
                                        "clocks at or below window_frames, so one prompt can be tested under "
                                        "uniform and non-uniform windows.")),
            ],
            outputs=[
                io.String.Output(display_name="path"),
                io.String.Output(display_name="report"),
                io.Custom("VHS_FILENAMES").Output(display_name="Filenames"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, audio, sampler, sigmas, prompt, width, height,
                window_frames, context_frames, max_seconds, seed, audio_mask, level,
                filename_prefix, crf, prompt_mode="cycle", window_mode="uniform") -> io.NodeOutput:
        import folder_paths
        waveform, rate, _ = _stereo(audio)
        vae_rate, hop = audio_grid(audio_vae)
        seconds = waveform.shape[-1] / rate
        if max_seconds > 0:
            seconds = min(seconds, float(max_seconds))
        total_frames = int(math.ceil(seconds * FPS))
        rng = random.Random(int(seed))
        plan = plan_windows(total_frames, int(window_frames), int(context_frames),
                            rng=rng if window_mode == "random" else None)
        blocks = parse_prompt_blocks(prompt)
        if prompt_mode == "uniform":
            pick = [0] * len(plan)
        elif prompt_mode == "random":
            pick = [rng.randrange(len(blocks)) for _ in plan]
        else:
            pick = [min(i, len(blocks) - 1) for i in range(len(plan))]
        # a block's own `frames:` wins for its window; the plan's last window still must reach the end
        frames_per_window = [blocks[pick[i]][0] or n for i, n in enumerate(plan)]

        enc = MiniMaxH3EncodeTrack.execute(audio_vae, audio, level)
        enc = getattr(enc, "args", enc)
        track_latent, enc_report = enc[0], enc[1]

        out_dir = folder_paths.get_output_directory()
        full_out, filename, counter, _sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        os.makedirs(full_out, exist_ok=True)
        stem = f"{filename}_{counter:05d}"

        files, reports = [], [enc_report]
        prev = None
        start = 0.0
        for i, frames in enumerate(frames_per_window):
            comfy.model_management.throw_exception_if_processing_interrupted()
            text = blocks[pick[i]][1]
            cond_out = MiniMaxH3Conditioning.execute(clip, vae, text, width, height, frames,
                                                     canvas="explicit")
            cond_out = getattr(cond_out, "args", cond_out)
            cond, latent = cond_out[0], cond_out[1]
            win = MiniMaxH3FreezeAudioWindow.execute(
                latent, audio_vae, audio, start, int(context_frames) if prev is not None else 0,
                previous=prev, audio_mask=audio_mask, level=level, track_latent=track_latent)
            win = getattr(win, "args", win)
            wlatent, _clip_audio, _span, trim, next_start, wreport, _new_audio = win
            reports.append(f"[{i}] {wreport}")

            guider = Guider_Basic(model)
            guider.set_conds(cond)
            latent_image = comfy.sample.fix_empty_latent_channels(model, wlatent["samples"])
            noise = comfy.sample.prepare_noise(latent_image, int(seed) + i)
            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=wlatent.get("noise_mask"),
                                    callback=callback, disable_pbar=False, seed=int(seed) + i)
            samples = samples.to(comfy.model_management.intermediate_device())
            prev = {"samples": samples}

            # the video VAE takes the video stream; core's VAEDecode unbinds the pair the same way
            video_stream = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
            images = vae.decode(video_stream)
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])
            images = images[int(trim):]
            path = os.path.join(full_out, f"{stem}_w{i:02d}.mp4")
            written = _write_frames_mp4(path, images, crf)
            files.append(path)
            reports.append(f"[{i}] wrote {written} frames to {os.path.basename(path)}")
            del images
            comfy.model_management.soft_empty_cache()
            start = float(next_start)

        # join, and mux the whole track cut to the video
        list_path = os.path.join(full_out, stem + "_concat.txt")
        wav_path = os.path.join(full_out, stem + "_track.wav")
        out_path = os.path.join(full_out, stem + ".mp4")
        with open(list_path, "w") as f:
            for p in files:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        _write_wav(wav_path, waveform, rate)
        cmd = [_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
               "-i", wav_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-shortest", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(list_path)
        os.remove(wav_path)
        total = sum(frames_per_window) - int(context_frames) * (len(frames_per_window) - 1)
        report = (f"{len(frames_per_window)} windows {frames_per_window} with {context_frames}-frame context, "
                  f"prompt blocks {[p + 1 for p in pick]} ({prompt_mode}), windows {window_mode}, "
                  f"{total} frames ({total / FPS:.2f}s) over {seconds:.2f}s of track -> {out_path}\n" + "\n".join(reports))
        logger.info("[h3] MiniMaxH3AudioFreezeSong: %s", report.splitlines()[0])
        return io.NodeOutput(out_path, report, (True, [out_path]))
