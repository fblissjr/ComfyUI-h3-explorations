"""Where a frozen-audio loop's files go, and what they carry.

`docs/h3_audio_freeze.md` owns the lane. `MiniMaxH3AudioFreezeSong` writes its
video through here; the layout lives outside the node so the next loop writer
gets the same one. (`MiniMaxH3JoinWindows`, the shot workflows' join node,
shared it until both retired on 2026-09-14.)

**The finished file** is `<prefix>_NNNNN.mp4`, the counter from core's
`folder_paths.get_save_image_path`. Not VHS's `-audio.mp4` spelling: core's
counter parses the digits after the prefix up to the next `_` or `.`, so a lone
`<prefix>_00001-audio.mp4` does not advance it, and with the PNG switched off
the next run would overwrite the last (tested against
`folder_paths.py::get_save_image_path`, 2026-09-14).

**Metadata** lives in the PNG only, the owner's policy since 2026-09-23 and the
one the VHS fork follows (`CLEAN_OUTPUT_ARGS` in its
`videohelpersuite/nodes.py`). The mp4 carries nothing but what it needs to
play: `CLEAN_OUTPUT_ARGS` below copies no metadata from any input and drops
ffmpeg's versioned encoder strings. `write_metadata_png` adds
`<prefix>_NNNNN.png`, the first frame carrying the prompt and workflow, which
is what a drag into the frontend reads and what `bench/diff_clip_graphs.py`
reads.

**Window files** live in `<prefix>_windows/`, one video and one latent per
window slot (`loop_resume.window_paths`), overwritten in place by the next run
of the same graph. The folder name has no digits after the prefix, so core's
counter ignores it.
"""

from __future__ import annotations

import contextlib
import datetime
import io as _bytes_io
import json
import os
import subprocess

from comfy_api.latest import io, ui
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_extras.nodes_minimax_h3 import FPS

from .audio_freeze import _ffmpeg, _write_wav


def window_dir(full_out: str, filename: str) -> str:
    """The working folder for a graph's window files, beside its finished files."""
    return os.path.join(full_out, f"{filename}_windows")


# Output options for every video file this pack writes through ffmpeg,
# inherited from the owner's VHS fork: `-map_metadata -1` copies no metadata
# from any input, `-fflags +bitexact` drops the versioned encoder strings.
CLEAN_OUTPUT_ARGS = ["-map_metadata", "-1", "-fflags", "+bitexact"]
# What a written file says about its colour, and the conversion that makes it
# true. Inherited from VHS's `video_formats/h264-mp4.json`, which converts and
# tags the same way. Left to itself ffmpeg turns piped rgb24 into BT.601 values
# and writes no tag (measured 2026-10-10 on a render's kept pixels against its
# BT.709 source), and a player that takes BT.709 for a picture this size then
# shows a render off in colour beside its own source. A loader reads either
# form back to the same rgb. `bench/check_audio_freeze.py` holds the case.
# `SAY_BT709` ends the filter chain: piped rgb frames carry no primaries or
# transfer, and ffmpeg n9.0.2 then leaves both out of the file whatever the
# output options say (measured 2026-10-10; VHS sets them on its input).
TO_BT709 = "scale=out_color_matrix=bt709"
SAY_BT709 = "setparams=color_primaries=bt709:color_trc=bt709"
BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
# AAC bitrate for the joined track. Inherited from the owner's VHS fork
# (`AAC_BITRATE` in its `videohelpersuite/nodes.py`), which measured it on
# music; the song node's track is music. Owner's choice, 2026-09-23.
AAC_BITRATE = "256k"


def join_and_mux(files: list[str], waveform, rate: int, out_path: str, scratch_dir: str, stem: str,
                 frames: int) -> None:
    """Concatenate `files` without re-encoding and mux the track under them. No metadata.

    `frames` is how many frames the files hold between them, and the output is exactly that
    long: the length is given to ffmpeg as a duration and the track is padded with silence to
    reach it. Until 2026-10-06 the length was left to `-shortest`, which with copied video
    cuts where the audio's last packet ends: a track as long as its video, or shorter, came
    back a frame to four short of what the windows held, and on some pictures dozens long
    (`bench/check_audio_freeze.py` holds the cases). Nothing is cut here any more. A track
    that ends before its windows do is cut at the frames, before they are encoded
    (`loop_plan.frames_kept`), because a copied stream cannot be cut cleanly between its
    reordered frames.

    The windows must share every muxer setting or the stream copy fails. The concat list and
    the track are written to `scratch_dir` and removed whether or not ffmpeg succeeds.
    """
    list_path = os.path.join(scratch_dir, stem + "_concat.txt")
    wav_path = os.path.join(scratch_dir, stem + "_track.wav")
    try:
        with open(list_path, "w") as f:
            for p in files:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        _write_wav(wav_path, waveform, rate)
        cmd = [_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
               "-i", wav_path, "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-b:a", AAC_BITRATE,
               "-t", f"{int(frames) / FPS:.6f}", *CLEAN_OUTPUT_ARGS, out_path]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed joining into {out_path}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
    finally:
        for p in (list_path, wav_path):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)


def write_metadata_png(png_path: str, video_path: str, prompt=None, extra_pnginfo=None) -> str:
    """The video's first frame as a PNG carrying the prompt and workflow chunks, as VHS writes it."""
    proc = subprocess.run([_ffmpeg(), "-v", "error", "-i", video_path, "-frames:v", "1",
                           "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not read the first frame of {video_path}: "
                           f"{proc.stderr.decode(errors='replace')[-400:]}")
    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    for key, value in (extra_pnginfo or {}).items():
        info.add_text(key, json.dumps(value))
    info.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])
    with Image.open(_bytes_io.BytesIO(proc.stdout)) as frame:
        # compress_level 4: inherited from VHS's metadata image
        frame.save(png_path, pnginfo=info, compress_level=4)
    return png_path


def saved_outputs(out_path: str, subfolder: str, png_path: str | None):
    """The VHS_FILENAMES value (the PNG first, the video last, as VHS lists them) and the preview."""
    filenames = (True, ([png_path] if png_path else []) + [out_path])
    preview = ui.PreviewVideo([ui.SavedResult(os.path.basename(out_path), subfolder, io.FolderType.output)])
    return filenames, preview
