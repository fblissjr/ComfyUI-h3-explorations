"""Drop a song and a prompt; the windows come from the track.

`docs/h3_audio_freeze.md` section 4 step 6, the loop: the LTX pack's
music-video mode on H3. One prompt, one track, and the node plans the windows
from the track's length, runs them one after another inside itself, writes
each window's new frames to a file as it goes, and joins the files with the
full track at the end. Nothing here holds more than one window of frames.

**The plan, the timeline and the prompt** are `loop_plan.py`'s: windows at
most `window_frames` long with `context_frames` frozen at each head, an
optional `mm:ss label` timeline the windows line up with, and one prompt or
one `--- label` block per label. `extent` is the whole track, or its first N
seconds for a quick look; if the track runs out inside the last window, the
freeze pads silence and the report says how much.

**Lists.** A `__name__` in the prompt takes a value from a Prompt List chained
into `lists`, through `prompt_lists.fill_windows`, the call every loop node
makes (`prompt_lists.py` holds the owner's rules): once per timeline entry, or
once per window with no timeline. Filling happens during planning, so every
key and encode below sees the filled-in text.

**Preview.** `preview` on plans, fills and reports every window's time range,
entry, filled prompt and whether resume would reuse it, and stops. The model,
encoder, VAEs, sampler, sigmas and references are lazy inputs this node does
not ask for in preview (`check_lazy_status`), so core never runs the loaders
behind them. The shipped graphs show the report on a Preview as Text.

**Everything is encoded before anything samples.** The track once, and each
distinct prompt once. A window's conditioning is a function of its text and
its references alone -- not its seed, start or the previous window -- because
continuity reaches the next window as the previous window's latent tail frozen
in, never through the encoder. So Qwen3-VL and the DiT trade places on the
card once per run rather than once per window, and a song on one prompt
encodes one prompt. With references the frame count joins the key: the
reference compiler is handed it, and a shorter window costs at most one more
encode. With resume, only the windows that render are encoded for, and a run
that reuses every window never loads the encoder.

**References.** `references` takes an Append Ref Image chain and presents it
with every window's prompt through `MiniMaxH3ReferenceConditioning`'s own
execute, so the stills are fitted, VAE-encoded and labelled exactly as on the
reference graphs. The prompt then follows the reference prompt format
(`docs/prompting.md` section 2.2). The fl2va checkpoint takes references
(owner, 2026-09-14).

**A source video** (`source`, a Masked Source node; owner, 2026-10-04) turns
the loop into masked video-to-video on the track's own picture: each window
starts from the source's frames over the same span instead of an empty latent,
only the masked subject regenerates, and after the decode the source's own
pixels go back everywhere else. The mask joins the window node's by a minimum,
so a later window's frozen context stays frozen whole. `video_mask.py` has the
reduction and why the composite is needed; the audio is still the track's,
frozen and muxed from the original. One sampler per window is what lets a mask
through as wired: a two-sampler graph hands its second sampler a still-noisy
latent, which core's inpaint step then treats as the clean plate
(`comfy/samplers.py::KSamplerX0Inpaint`). A peer session reproduced that on
core's own sampler classes with a stub model, and found that restoring the
plate in the pinned rows between the two samplers fixes it; neither has run
on H3. A window in which nothing is masked is not sampled: every row would be
pinned and the pass would return the source. **A motion reference** (the
Masked Source's `motion_reference`, 2026-10-05) is built here per window from
the source's own frames and appended to the reference chain as a video, with
or without the video model's copy, so the model is shown the original's
movement through the channel it was trained to take motion from; the window's
conditioning key then carries the window number, since each window's
reference differs (`video_mask.motion_reference` says what the reference is).

**Sampling** is what SamplerCustomAdvanced does, per window: a BasicGuider on
the model with the window's conditioning, prepared noise at `seed + i`, the
given sampler and sigmas, the nested noise mask from the window node.

**Files and resume.** `loop_output.py` and `loop_resume.py`: window files in
`<prefix>_windows/`, each video with its sampled latent beside it, overwritten
in place by the next run of the graph. A run reuses the stored windows whose
inputs have not changed, in order, and renders from the first that has
(`reuse_windows`); `loop_resume.py` says what a window's key covers and what it
cannot see. The seed holds after each queue so a re-queue can reuse.
With a `source` wired, each window also gets a **mask review**
(`save_mask_review`, 2026-10-06): its render stacked over a view of what was
regenerated (`video_mask.overlay_pieces`), kept beside the window's video as
`..._with_mask.mp4` and joined into `<prefix>_NNNNN_with_mask.mp4` as the
windows are joined, so the pair plays in step with one track. A reused window
stored without one is stacked from its stored video.
A window that does render again, because its seed, the sampler or the
schedule changed, reuses two things an earlier run in this server session
made, under the same switch: its source latent and its conditioning
(`window_keep.py`, 2026-10-06). Those are kept in memory, found by the
objects they were made from, and the report says which windows used them.
`keep_windows` off removes this run's window files after the join. The
finished `<prefix>_NNNNN.mp4` carries no metadata (`loop_output.py` says why);
`save_metadata_png` writes the first frame as a PNG carrying the prompt and
workflow, the only place the graph is kept.

First run: `bench/results/2026-09-12_audio_freeze_song_smoke.jsonl`, one short
window; `bench/results/2026-09-14_audio_freeze_song_stage1_smoke.jsonl`, two
short windows with and without references. Resume arrived on 2026-09-14 after
that; its first run is a throwaway.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
import subprocess

import torch
from comfy_api.latest import io, ui

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic
from comfy_extras.nodes_minimax_h3 import FPS, _empty_av_latent

from .audio_freeze import MiniMaxH3EncodeTrack, MiniMaxH3FreezeAudioWindow, _ffmpeg, _stereo
from .conditioning import MiniMaxH3Conditioning
from . import loop_plan, loop_resume, part_coverage, shot_table
from .prompt_lists import H3PromptLists, fill_windows
from .loop_output import (BT709_TAGS, CLEAN_OUTPUT_ARGS, SAY_BT709, TO_BT709, join_and_mux, saved_outputs,
                          window_dir, write_metadata_png)
from .reference_conditioning import H3References, MiniMaxH3ReferenceConditioning, RuntimeVideoReference, _order_records
from .reference_order import assign_labels
from . import video_mask
from . import window_keep

logger = logging.getLogger(__name__)

#: Reuse of a rendered window stored on disk, as a whole: only while this is True. **False since 2026-10-07 by the
#: owner's decision, and to be turned on by a code change and nothing else**; `reuse_windows` on the node cannot
#: turn it on. A stored window's key (`loop_resume.py`) is built from the queued graph, the track, the text and the
#: seed, and from nothing derived from this pack's code, so a window rendered before a fix is handed back after it
#: when the settings are the same. The kept mask had the same blind spot and hid a fix that evening
#: (`video_mask.MASK_REUSE_ENABLED`). **Before either is turned back on, the key has to change when the code does.**
#: Windows are still written (`keep_windows`), and what a session keeps in memory (`window_keep.py`) is not affected:
#: a restart clears it, and a code change needs a restart.
WINDOW_REUSE_ENABLED = False

#: The inputs a preview does not ask for: every one runs a loader or a model.
LAZY = ("model", "clip", "vae", "audio_vae", "sampler", "sigmas", "references", "source")


def _write_pieces_mp4(path: str, pieces, width: int, height: int, crf: int, under: str | None = None) -> int:
    """Float frames in [0, 1], [n, height, width, 3] a piece at a time, to an H.264 mp4 through ffmpeg's
    rawvideo pipe. Frames written.

    With `under`, the piped frames are stacked below that video's, frame for frame, in one picture of
    twice the height: a stored window's video over a view drawn now.

    The file is BT.709 and says so (`loop_output.TO_BT709`). The piped frames are converted before they
    are stacked: left to the stack, they would take ffmpeg's default matrix under the file's tag.
    """
    cmd = [_ffmpeg(), "-y", "-v", "error"] + (["-i", under] if under else []) + [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{int(width)}x{int(height)}", "-r", str(FPS), "-i", "-"]
    if under:
        cmd += ["-filter_complex", f"[1:v]{TO_BT709},format=yuv420p[drawn];"
                                   f"[0:v][drawn]vstack=inputs=2:shortest=1,{SAY_BT709}[v]",
                "-map", "[v]", "-r", str(FPS)]
    else:
        cmd += ["-vf", f"{TO_BT709},{SAY_BT709}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
            *BT709_TAGS, *CLEAN_OUTPUT_ARGS, path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    written = 0
    try:
        for piece in pieces:
            for i in range(0, int(piece.shape[0]), 24):
                part = piece[i:i + 24]
                proc.stdin.write((part.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy().tobytes())
                written += int(part.shape[0])
    finally:
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {err.decode(errors='replace')[-400:]}")
    return written


def _write_frames_mp4(path: str, images: torch.Tensor, crf: int) -> int:
    """[T, H, W, 3] float in [0, 1] to an H.264 mp4 through ffmpeg's rawvideo pipe."""
    t, h, w, _ = images.shape
    return _write_pieces_mp4(path, [images], w, h, crf)


def _write_review_mp4(path: str, pieces, width: int, height: int, crf: int, under: str | None = None) -> int:
    """A mask review at `path`, whole or not at all: written under a temporary name beside it and renamed
    when ffmpeg has finished.

    A window's review is reused whenever its file exists, and nothing else marks it finished (a window's
    video has its latent, written last). So a file under the final name must never be a partial one: a
    run that died inside this encode would otherwise leave every later run to fail at the review's join,
    after all its windows had sampled.
    """
    part = path[:-len(".mp4")] + ".part.mp4"
    try:
        written = _write_pieces_mp4(part, pieces, width, height, crf, under)
        os.replace(part, path)
        return written
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(part)


def _review_or_reason(make, what: str) -> str | None:
    """Run `make`, which writes a mask review, and return None, or why it failed.

    The review is a picture for the eye and never costs the render: a failure in it becomes a line of
    the report and a warning in the log, and the run goes on to its metadata, its shot table and its
    outputs exactly as it would with the switch off. Core's interrupt is not an `Exception`, so stopping
    a run during a review stops it.
    """
    try:
        make()
    except Exception as exc:  # noqa: BLE001 -- see above: anything the review can raise, the render outlives
        logger.warning("[h3] MiniMaxH3AudioFreezeSong: the mask review of %s failed; the render is unaffected: "
                       "%s: %s", what, type(exc).__name__, exc)
        return f"{type(exc).__name__}: {str(exc)[:300]}"
    return None




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
                "encodes the track and every distinct prompt once, then runs the windows in "
                "sequence with the previous window's tail frozen as context and the track's slice "
                "frozen in each, writes each window's new frames to <prefix>_windows/ as it goes, "
                "and joins the files with the full track, reusing unchanged windows from the last "
                "run. An optional timeline lines the windows up with sections of the track, one "
                "prompt block per label; preview reports the plan without loading a model. "
                "docs/h3_audio_freeze.md."
            ),
            # The order was reset on 2026-09-14, when prompt_mode and window_mode
            # went (owner): shipped graphs carry inputs by name, and a song graph
            # saved from the editor before then needs re-making. Append from here.
            inputs=[
                io.Model.Input("model", lazy=True),
                io.Clip.Input("clip", lazy=True),
                io.Vae.Input("vae", lazy=True, tooltip="Video VAE."),
                io.Vae.Input("audio_vae", lazy=True),
                io.Audio.Input("audio", tooltip="The whole track."),
                io.Sampler.Input("sampler", lazy=True),
                io.Sigmas.Input("sigmas", lazy=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip=("One prompt for every window; with a timeline, optionally one block per "
                                         "label, each opened by a line `--- label`.")),
                io.String.Input("timeline", multiline=True, default="",
                                tooltip=("Optional `mm:ss label` lines, the first at 00:00, e.g. `00:32 chorus`. "
                                         "Windows line up with each time, and lists move on once per line.")),
                io.Boolean.Input("preview", default=False,
                                 tooltip=("Report every window's time range, timeline entry, filled prompt and "
                                          "whether it would be reused, then stop: no model loads, nothing renders.")),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("window_frames", default=345, min=141, max=345, step=51,
                             tooltip=("The longest window, on both clocks: 141, 192, 243, 294 or 345. Windows "
                                      "that end a timeline entry or the track may be shorter.")),
                io.Int.Input("context_frames", default=39, min=39, max=141, step=51,
                             tooltip="Frames of the previous window frozen at the head of the next: 39, 90 or 141."),
                # A DynamicCombo, not a Float whose 0 meant "the whole track":
                # the owner's rule (2026-09-13) is that a number never means a
                # mode. Selection first, then the option's own widget.
                io.DynamicCombo.Input(
                    "extent",
                    options=[
                        io.DynamicCombo.Option("whole", []),
                        io.DynamicCombo.Option("first_seconds", [
                            io.Float.Input("seconds", default=30.0, min=0.5, max=36000.0, step=0.5,
                                           tooltip="Cover only the first N seconds of the track."),
                        ]),
                    ],
                    tooltip=("How much of the track to cover. whole: every window the plan needs "
                             "to reach the end. first_seconds: only the first N seconds, for a "
                             "quick look at the seams before committing to the whole song.")),
                # control_after_generate declared, not left to the frontend: it
                # draws a control widget for any INT named `seed` whether or not
                # the schema asks (`useIntWidget.ts`). Fixed, not the frontend's
                # randomize: a new seed after every queue would re-render every
                # window, which resume (`reuse_windows`) exists to avoid
                # (owner, 2026-09-14).
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=io.ControlAfterGenerate.fixed),
                io.Float.Input("audio_mask", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("level", options=["clip_guard", "peak", "none"], default="clip_guard"),
                io.String.Input("filename_prefix", default="Video/h3_song"),
                io.Int.Input("crf", default=19, min=0, max=51),
                io.Boolean.Input("save_metadata_png", default=True,
                                 tooltip=("Also write <prefix>_NNNNN.png, the first frame carrying the prompt and "
                                          "workflow, beside the video. The video carries neither, so off keeps "
                                          "no record of the graph.")),
                io.Boolean.Input("keep_windows", default=True,
                                 tooltip=("Keep this run's window files (the videos and the latents resume reads) "
                                          "in <prefix>_windows/ after the join. Off removes them, so the next run "
                                          "renders every window.")),
                H3References.Input("references", optional=True, lazy=True,
                                   tooltip=("Reference stills from an Append Ref Image chain, presented with every "
                                            "window's prompt and encoded once per distinct prompt. The prompt then "
                                            "names them as <Picture N>.")),
                io.Boolean.Input("reuse_windows", default=False,
                                 tooltip=("STORED WINDOWS ARE NOT REUSED FOR NOW, whatever this is set to: every "
                                          "window renders, until that is turned back on in the code.\n\n"
                                          "What it still does when on: a window that renders again reuses its "
                                          "source encode and its prompt encode from an earlier run in this "
                                          "session, when nothing they are made from has changed. A restart clears "
                                          "those.\n\nWhat it does when stored windows are enabled: reuse the "
                                          "stored windows whose inputs have not changed, in order, and render from "
                                          "the first that has.")),
                H3PromptLists.Input("lists", optional=True,
                                    tooltip=("Prompt List nodes filling __name__ placeholders in the prompt: one "
                                             "value per timeline entry, or per window with no timeline.")),
                # appended 2026-10-04: masked video-to-video on the track's own picture
                video_mask.H3MaskedSource.Input(
                    "source", optional=True, lazy=True,
                    tooltip=("A Masked Source node: the track's own video and a mask over one subject. Each "
                             "window then keeps the source outside the mask and regenerates the subject, "
                             "from the references and the prompt.")),
                # appended 2026-10-06, at the owner's ask: a render that shows what it regenerated
                io.Boolean.Input("save_mask_review", default=True, optional=True,
                                 tooltip=("Also saves the render stacked over a view of what was regenerated: "
                                          "the mask the source carries, what else regenerates around it, and "
                                          "what was kept from the render. Needs a Masked Source on `source`.")),
                # appended 2026-10-06: a clip too long to load whole, rendered as consecutive runs
                io.String.Input("continue_from", default="", optional=True,
                                tooltip=("Continues another run: the path of that run's last stored window, the "
                                         ".safetensors beside its video in its _windows folder. This run's first "
                                         "window then takes that window's last frames as its context, as each "
                                         "window takes the one before it. This run's track and video must start "
                                         "context_frames frames before its first new frame; those frames are not "
                                         "written. To sample as one long run would, set seed to that run's seed "
                                         "plus the windows it rendered. Empty starts cold.")),
            ],
            outputs=[
                io.String.Output(display_name="path"),
                io.String.Output(display_name="report"),
                io.Custom("VHS_FILENAMES").Output(display_name="Filenames"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, preview=False, **kwargs):
        # None is a connected input core has not run yet; an unconnected
        # optional input is absent. A preview asks for nothing.
        if preview:
            return []
        return [name for name in LAZY if name in kwargs and kwargs[name] is None]

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, audio, sampler, sigmas, prompt, timeline, preview,
                width, height, window_frames, context_frames, extent, seed, audio_mask, level,
                filename_prefix, crf, save_metadata_png=True, keep_windows=True, references=None,
                reuse_windows=False, lists=None, source=None, save_mask_review=True,
                continue_from="") -> io.NodeOutput:
        import folder_paths
        # A DynamicCombo arrives as one nested dict (the selection under its own
        # id, the option's inputs beside it) or, from an API prompt that sets
        # only the selection, as a bare string.
        choice = extent if isinstance(extent, str) else extent["extent"]
        if choice not in ("whole", "first_seconds"):
            raise ValueError(f"unknown extent {choice!r}")
        max_seconds = (float(extent["seconds"]) if choice == "first_seconds" and not isinstance(extent, str)
                       else (30.0 if choice == "first_seconds" else None))
        waveform, rate, _ = _stereo(audio)
        track_seconds = seconds = waveform.shape[-1] / rate
        if max_seconds is not None:
            seconds = min(seconds, max_seconds)
        total_frames = int(math.ceil(seconds * FPS))
        context_frames = int(context_frames)
        plan = loop_plan.plan_song(total_frames, int(window_frames), context_frames, prompt, timeline)
        # filled before anything is keyed or encoded: resume and the encode
        # cache see the text a window renders, never its placeholders
        texts, list_lines = fill_windows(plan.uses, lists)
        windows = loop_plan.place_windows(plan, texts, context_frames)
        entries = plan.entries
        n_windows = len(windows)

        out_dir = folder_paths.get_output_directory()
        full_out, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        work_dir = window_dir(full_out, filename)
        stem = f"{filename}_{counter:05d}"
        graph = getattr(cls.hidden, "prompt", None)
        extra = getattr(cls.hidden, "extra_pnginfo", None)

        # The stored windows that still hold: a prefix whose keys match. With no
        # queued prompt (a direct call) there is no root, nothing is reused and
        # nothing is stored for reuse.
        root = loop_resume.root_key(
            loop_resume.graph_signature(graph, getattr(cls.hidden, "unique_id", None),
                                        skip=loop_resume.SONG_PER_WINDOW),
            loop_resume.track_hash(waveform, rate))
        keys, reused = [], []
        # A run that continues another's: its first window is keyed on the stored window it takes its
        # context from, as window two is keyed on window one, and writes its length less that context.
        # `head` is those frames; the run's track starts on the first of them.
        continue_from = str(continue_from or "").strip()
        continued_key = loop_resume.stored_key(continue_from) if continue_from else None
        head = context_frames if continue_from else 0
        if root is not None:
            for w in windows:
                keys.append(loop_resume.window_key(root, w.number, w.text, w.frames, w.start,
                                                   int(seed) + w.number - 1, keys[-1] if keys else continued_key))
        total = loop_plan.frames_covered([w.frames for w in windows], context_frames)
        # what is written of that: every frame, or as far as the track runs when it ends first;
        # the frames past it come off the last window's tail before it is encoded
        kept_frames = loop_plan.frames_kept(total, track_seconds)
        writes = loop_plan.frames_written([w.frames for w in windows], context_frames, kept_frames, head)
        again = None
        if WINDOW_REUSE_ENABLED and reuse_windows and root is not None:
            for w in windows:
                stored = loop_resume.read_window(work_dir, filename, w.number)
                if stored is None or stored["key"] != keys[w.number - 1]:
                    break
                holds = loop_resume.stored_frames(stored, w.frames)
                if holds != writes[w.number - 1]:
                    # sampled the same way, cut differently: a last window stored whole before
                    # 2026-10-06 under a track that ends inside it. Its key matches for ever, so
                    # it is refused here and renders once more, cut at the frames.
                    again = (w.number, holds)
                    break
                reused.append(stored)
        first = len(reused)

        clock = loop_plan.clock
        lines = [f"{n_windows} windows {[w.frames for w in windows]} with {context_frames}-frame context, "
                 f"{total} frames ({total / FPS:.2f}s) over {seconds:.2f}s of track"
                 + (f"; {kept_frames} are written, the track ends first" if kept_frames < total else "")]
        if continue_from:
            lines.append(f"continues from {os.path.basename(continue_from)}: the first {head} frames of this run's "
                         f"track are that window's context and are not written; {kept_frames - head} frames are")
        # A run that cannot cover what it was asked for says so here, second line of the report,
        # and in the log: it is not refused (`loop_plan.extent_shortfall`). On a masked graph the
        # usual cause is the loader's frame cap left under a raised extent; the two are separate
        # widgets (2026-10-06). `source` is lazy and None in a preview, which then reports the
        # track alone.
        asked_reads = None
        if max_seconds is not None and source is not None:
            with contextlib.suppress(ValueError):   # a timeline the longer extent would refuse
                asked_reads = loop_plan.frames_read(int(math.ceil(max_seconds * FPS)), int(window_frames),
                                                    context_frames, timeline)
        shortfall = loop_plan.extent_shortfall(
            track_seconds, max_seconds, total_frames,
            int(source["frames"].shape[0]) if source is not None else None, asked_reads)
        if shortfall:
            lines.append(shortfall)
            logger.warning("[h3] MiniMaxH3AudioFreezeSong: %s", shortfall)
        for i, (at, label) in enumerate(entries):
            mine = [w for w in windows if w.entry == i]
            lines.append(f"{label} at {clock(at)}: " + (
                f"windows {mine[0].number}-{mine[-1].number}, starting at {clock(mine[0].first_frame / FPS)} "
                f"({mine[0].first_frame / FPS - at:+.2f}s)" if mine else "past what this run covers"))
        for w in windows:
            end = w.first_frame + w.frames - (context_frames if w.number > 1 else 0)
            lines.append(f"[{w.number}] {clock(w.first_frame / FPS)}-{clock(end / FPS)}, {w.frames} frames"
                         + (f", {entries[w.entry][1]}" if w.entry is not None else "")
                         + (", reused" if w.number <= first
                            else f", renders again: its stored file holds {again[1]} frames and this run "
                                 f"writes {writes[w.number - 1]}" if again and again[0] == w.number
                            else ", renders"))
            if preview:
                lines.append("    " + w.text.replace("\n", "\n    "))
        lines += list(list_lines)

        # In a preview every lazy input arrives as None: nothing above this
        # line may read model, clip, vae, audio_vae, sampler, sigmas or
        # references, or a preview raises instead of reporting.
        if preview:
            report = "preview, nothing rendered: " + "\n".join(lines)
            logger.info("[h3] MiniMaxH3AudioFreezeSong: %s", report.splitlines()[0])
            return io.NodeOutput("", report, (True, []), ui=ui.PreviewText(report))

        os.makedirs(work_dir, exist_ok=True)
        if source is not None:
            # refused before any encode: a wrong source would fail at its window, minutes in
            have = int(source["frames"].shape[0])
            beyond = [w.number for w in windows if int(round(w.start * FPS)) >= have]
            if beyond:
                raise ValueError(
                    f"the source video has {have} frames and window {beyond[0]} of {n_windows} starts past "
                    f"its end: load more of it at {FPS} fps (the loader's frame cap), or shorten `extent`")
            grown = video_mask.source_margins(source, 0, have, int(width) * int(height))
            lines.append(f"source video: {have} frames, mask grown {video_mask.margin_note(grown)}"
                         + (f" ({source['grow_by']}, at most {source['grow_pixels']} px)"
                            if torch.is_tensor(grown) else "")
                         + f", blend {source['feather_pixels']} px"
                         + (", subject painted out before the encode" if source.get("paint_out") else "")
                         + ((f", sampling starts {int(source['start_knots'])} knot(s) late: the top "
                             f"{100.0 * float(source['start_top']):.0f}% of the subject from the original blurred by "
                             f"{int(source['start_blur'])} px, the rest of it from nothing")
                            if source.get("start_from", video_mask.START_NOISE) != video_mask.START_NOISE else ""))
            # the part mask's coverage of the tracked subject, when the Masked Source found it in doubt. A
            # render shows it here, above the first window; a preview has no source and cannot.
            if part_coverage.record_line(source):
                lines.append(part_coverage.record_line(source))
        # the mask review needs a source to show; without one the switch does nothing
        review = bool(save_mask_review) and source is not None
        # Every rendering window's conditioning before any window samples; see
        # the module docstring for why the key is the text (and, with
        # references, the frame count) and nothing else.
        track_latent, conds, cond_keys = None, {}, {}
        kept_conds: list[int] = []
        reports = list(lines)
        # Where the render's time goes, by stage, for the report (2026-10-06): this node encodes,
        # samples and decodes inside itself, so nothing outside it can time its stages. Wall clock
        # at stage ends, where a result has already come back from the card. Changes nothing rendered.
        spent: dict[str, float] = {}
        lap: dict[str, float] = {}
        clock = [time.perf_counter()]

        def mark(stage: str) -> None:
            now = time.perf_counter()
            spent[stage] = spent.get(stage, 0.0) + now - clock[0]
            lap[stage] = lap.get(stage, 0.0) + now - clock[0]
            clock[0] = now

        if first < n_windows:
            enc = MiniMaxH3EncodeTrack.execute(audio_vae, audio, level)
            enc = getattr(enc, "args", enc)
            track_latent = enc[0]
            reports.append(enc[1])
            mark("track encode")
            # A motion reference is built per window from the source (`video_mask.motion_reference`):
            # the window's own frames, so its key carries the window number.
            motion = source.get("motion_reference", video_mask.MOTION_NONE) if source is not None else video_mask.MOTION_NONE
            motion_label = None
            for w in windows[first:]:
                with_refs = references is not None or motion != video_mask.MOTION_NONE
                ck = (w.text, w.frames if with_refs else None, w.number if motion != video_mask.MOTION_NONE else None)
                cond_keys[w.number] = ck
                if ck in conds:
                    continue
                comfy.model_management.throw_exception_if_processing_interrupted()
                lap.clear()
                # What an earlier run of this stretch encoded is reused under `reuse_windows`
                # (`window_keep.py`): found by the objects it was made from, so a hit is the
                # same conditioning an encode would give, and the text encoder is not asked for.
                # With the switch off nothing is read from the keep, and what this run encodes
                # replaces what was there, as its windows replace the stored ones.
                cond_kept = window_keep.cond_key(clip, w.text, w.frames, width, height, references, vae, audio_vae,
                                                 source, int(round(w.start * FPS)))
                kept = window_keep.CONDS.get(*cond_kept) if reuse_windows else None
                if kept is not None:
                    conds[ck], label = kept
                    motion_label = label if label is not None else motion_label
                    kept_conds.append(w.number)
                    mark("conditioning")
                    reports.append(f"[{w.number}] conditioning kept from an earlier run (not encoded)")
                    continue
                refs_w = references
                if motion == video_mask.MOTION_WIRED:
                    # a video the user wired beside the source: this window's own frames of it
                    ref_frames = video_mask.wired_motion(source, int(round(w.start * FPS)), w.frames, width, height)
                    refs_w = tuple(references or ()) + (RuntimeVideoReference(
                        frames=ref_frames, loaded_fps=float(FPS), soundtrack=None,
                        use_vae=bool(source.get("motion_vae", False))),)
                    motion_label = assign_labels(_order_records(refs_w))[-1]
                    mark("motion reference")
                elif motion != video_mask.MOTION_NONE:
                    pixels, mask, _held = video_mask.window_frames(
                        source, int(round(w.start * FPS)), w.frames, width, height)
                    # zoomed in, the window's own box per shot, around the tracked subject (`video_mask.window_boxes`)
                    boxes = (video_mask.window_boxes(source, int(round(w.start * FPS)), w.frames, width, height)
                             if motion == video_mask.MOTION_ZOOM else None)
                    # The subject widened by half of `grow_pixels`, whatever `grow_by` is: a margin taken from
                    # the subject's size greys a limb that leaves the part's edge (`video_mask.MOTION_WIDEN`).
                    ref_frames = video_mask.motion_reference(
                        pixels, mask, motion, int(source["motion_short_edge"]), video_mask.motion_widening(source),
                        boxes)
                    if boxes is not None:
                        reports.append(f"[{w.number}] motion reference zoomed in: "
                                       + video_mask.zoom_note(boxes, height, width, int(source["motion_short_edge"])))
                    del pixels, mask, boxes
                    refs_w = tuple(references or ()) + (RuntimeVideoReference(
                        frames=ref_frames, loaded_fps=float(FPS), soundtrack=None,
                        use_vae=bool(source.get("motion_vae", False))),)
                    motion_label = assign_labels(_order_records(refs_w))[-1]
                    mark("motion reference")
                if refs_w is None:
                    out = MiniMaxH3Conditioning.execute(clip, vae, w.text, width, height, w.frames,
                                                        canvas="explicit")
                else:
                    out = MiniMaxH3ReferenceConditioning.execute(clip, refs_w, w.text, width, height,
                                                                 w.frames, vae=vae, audio_vae=audio_vae)
                conds[ck] = getattr(out, "args", out)[0]
                del refs_w
                # with the reference's name in the prompt, which the report line below reads
                window_keep.CONDS.put(*cond_kept, (conds[ck], motion_label if motion != video_mask.MOTION_NONE else None))
                mark("conditioning")
                # per window: the first carries the encoder coming onto the card, the later ones are warm
                reports.append(f"[{w.number}] conditioning seconds: "
                               + ", ".join(f"{name} {took:.1f}" for name, took in lap.items()))
            mark("conditioning")
            if motion != video_mask.MOTION_NONE and first < n_windows:
                reports.append(f"motion reference: {motion} at a {int(source['motion_short_edge'])} short edge, "
                               + ("with the video model's copy" if source.get("motion_vae") else "text encoder only")
                               + f", named {motion_label} in the prompt, built per window from the source")
        reports.append((f"reused windows 1-{first} of {n_windows}" if first else "no stored window reused")
                       + (f"; {len(conds) - len(kept_conds)} conditioning(s) encoded"
                          + (f" and {len(kept_conds)} kept from an earlier run" if kept_conds else "")
                          + f" for {n_windows - first} rendered window(s)"
                          if first < n_windows else "; nothing rendered")
                       + (" with references" if references is not None and first < n_windows else ""))

        files = [s["video"] for s in reused]
        stored_latents = [s["latent"] for s in reused]
        prev = loop_resume.load_window_latent(reused[-1]["latent"]) if reused and first < n_windows else None
        if prev is None and continue_from and first < n_windows:
            # no window of this run is reused: its first takes its context from the run it continues
            prev = loop_resume.load_window_latent(continue_from)
        for w in windows[first:]:
            i = w.number - 1
            comfy.model_management.throw_exception_if_processing_interrupted()
            lap.clear()
            clock[0] = time.perf_counter()
            latent, _count = _empty_av_latent(width, height, w.frames)
            src_pixels = src_tokens = src_mask = kept = None
            if source is not None:
                # this window starts from the source's own frames over its span
                empty_video, empty_audio = latent["samples"].unbind()
                src_pixels, src_encode, src_tokens, src_mask, held = video_mask.window(
                    source, int(round(w.start * FPS)), w.frames, width, height, *empty_video.shape[2:])
                # the margin `window` grew the region by, for the report and the composite below
                margin = video_mask.source_margins(source, int(round(w.start * FPS)), w.frames,
                                                   int(width) * int(height))
                if held:
                    # of the frames the source cannot give, those past the track are sampled on and dropped
                    past = min(int(held), w.frames - (context_frames if i or head else 0) - writes[i])
                    reports.append(f"[{w.number}] the source ends {held} frames before this window does; "
                                   "its last frame is held, unmasked"
                                   + (f" ({past} of them are past the track's end and are not written)" if past else ""))
                # The plate's encode does not depend on the seed or the schedule: under `reuse_windows`
                # an earlier run's is reused (`window_keep.py`). Kept as the VAE returned it, before the
                # late start's multiply below; the window node copies the video before it writes context in.
                latent_kept = window_keep.latent_key(source, vae, int(round(w.start * FPS)), w.frames, width, height)
                z = window_keep.LATENTS.get(*latent_kept) if reuse_windows else None
                if z is not None:
                    reports.append(f"[{w.number}] source latent kept from an earlier run (not encoded)")
                else:
                    z = vae.encode(src_encode)
                    window_keep.LATENTS.put(*latent_kept, z)
                del src_encode
                if tuple(z.shape) != tuple(empty_video.shape):
                    raise ValueError(
                        f"the video VAE returned {tuple(z.shape)} for a {w.frames}-frame window; the "
                        f"window's latent is {tuple(empty_video.shape)}")
                z = z.to(device=empty_video.device, dtype=empty_video.dtype)
                empty = video_mask.start_zero_tokens(source, src_mask, src_tokens, int(round(w.start * FPS)))
                if empty is not None:
                    # H3's latent has no shift and a scale of one (`comfy/latent_formats.py::MiniMaxH3Video`),
                    # so a zero here is a zero for the model: these tokens carry no source into a late start
                    z = z * (1.0 - empty[None, None].to(z))
                latent = {"samples": comfy.nested_tensor.NestedTensor((z, empty_audio))}
                mark("source encode")
            # Always the real value: the window node freezes nothing when
            # `previous` is None and keeps the widget for what the NEXT window
            # takes. Passing 0 for the first window was the zero-as-mode this
            # session removed, and it raised on the first run after (2026-09-13).
            win = MiniMaxH3FreezeAudioWindow.execute(
                latent, audio_vae, audio, w.start, context_frames,
                previous=prev, audio_mask=audio_mask, level=level, track_latent=track_latent)
            win = getattr(win, "args", win)
            wlatent, _clip_audio, _span, trim, next_start, wreport, _new_audio = win
            reports.append(f"[{w.number}] {wreport}")
            if src_tokens is not None:
                # a minimum, so the context the window node froze stays frozen whole
                frozen_video, frozen_audio = wlatent["noise_mask"].unbind()
                wlatent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                    (torch.minimum(frozen_video, src_tokens[None, None].to(frozen_video)), frozen_audio))
                reports.append(f"[{w.number}] source kept outside the mask: "
                               f"{100.0 * float(src_tokens.mean()):.1f}% of the window's video tokens regenerate"
                               + (f"; the region's margin is {video_mask.margin_note(margin)} over the window "
                                  f"({source['grow_by']})" if torch.is_tensor(margin) else ""))

            mark("window setup")
            untouched = src_tokens is not None and not bool(src_tokens.any())
            if untouched:
                # nothing is masked in this window (the subject is off screen): every
                # row would be pinned, so the pass is skipped and the window is the source
                samples = wlatent["samples"]
                reports.append(f"[{w.number}] nothing masked in this window; not sampled, the source is written")
            else:
                guider = Guider_Basic(model)
                guider.set_conds(conds[cond_keys[w.number]])
                latent_image = comfy.sample.fix_empty_latent_channels(model, wlatent["samples"])
                noise = comfy.sample.prepare_noise(latent_image, int(seed) + i)
                x0_output = {}
                run_sigmas = sigmas
                if source is not None and source.get("start_from", video_mask.START_NOISE) != video_mask.START_NOISE:
                    late = int(source["start_knots"])
                    if late >= int(sigmas.shape[-1]) - 1:
                        raise ValueError(
                            f"start_knots {late} leaves no step of a {int(sigmas.shape[-1]) - 1}-step schedule")
                    run_sigmas = sigmas[late:]
                callback = latent_preview.prepare_callback(model, run_sigmas.shape[-1] - 1, x0_output)
                samples = guider.sample(noise, latent_image, sampler, run_sigmas,
                                        denoise_mask=wlatent.get("noise_mask"),
                                        callback=callback, disable_pbar=False, seed=int(seed) + i)
                samples = samples.to(comfy.model_management.intermediate_device())
                mark("sampling")
            prev = {"samples": samples}

            if untouched and src_pixels is not None:
                images = src_pixels.clone()
            else:
                # the video VAE takes the video stream; core's VAEDecode unbinds the pair the same way
                video_stream = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
                images = vae.decode(video_stream)
                if images.ndim == 5:
                    images = images.reshape(-1, *images.shape[-3:])
            mark("decode")
            if src_pixels is not None and src_tokens is not None and not untouched:
                if source.get("composite") == video_mask.COMPOSITE_CHANGED:
                    # the render is kept only where it changed the picture or the old subject stood
                    alpha = video_mask.changed_alpha(images, src_pixels, src_tokens, src_mask,
                                                     source["feather_pixels"], margin // 2,
                                                     source["change_threshold"])
                    whole = float(video_mask.pixel_alpha(src_tokens, height, width, 0).mean())
                    reports.append(f"[{w.number}] composite keeps only what changed: "
                                   f"{100.0 * float((alpha > 0.5).float().mean()) / max(whole, 1e-6):.0f}% of the "
                                   "regenerated pixels, the source restored in the rest")
                else:
                    alpha = video_mask.pixel_alpha(src_tokens, height, width, source["feather_pixels"])
                # a latent step's region covers its whole run of frames; across a cut from the subject
                # that is the next shot's picture, and it stays the source's (`video_mask.cut_gate`)
                first = int(round(w.start * FPS))
                gate = video_mask.cut_gate(src_mask, int(src_tokens.shape[0]), source.get("cuts"), first)
                if not bool(gate.all()):
                    alpha = alpha * gate[:, None, None].to(alpha)
                    across = [first + int(f) for f in (gate < 0.5).nonzero().flatten()]
                    reports.append(f"[{w.number}] {len(across)} frame(s) lie across a cut from the subject inside "
                                   "one latent step and are left as the source: frame(s) "
                                   + ", ".join(str(f) for f in across))
                images = video_mask.composite(images, src_pixels, alpha)
                # under `only what changed` the weight is also what the mask review outlines: it is held
                # through the window's write for that, and freed after the review, where it used to be
                # freed here
                kept = alpha if source.get("composite") == video_mask.COMPOSITE_CHANGED else None
                del alpha
                mark("composite")
            images = images[int(trim):]
            overrun = int(images.shape[0]) - writes[i]
            if w.number == n_windows and overrun > 0:
                # past the track's end: dropped here, at the frames, so the join copies whole windows
                images = images[:writes[i]]
                reports.append(f"[{w.number}] the track ends {overrun} frames before this window "
                               "does; they are not written")
            video_path, latent_path = loop_resume.window_paths(work_dir, filename, w.number)
            # the old latent goes first: a latent on disk must mean its video finished. Its old mask
            # review goes with it: a review on disk must be of the video beside it, and a run with the
            # switch off would otherwise leave the last render's review for a later run to join
            for stale in (latent_path, loop_resume.review_path(work_dir, filename, w.number)):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(stale)
            written = _write_frames_mp4(video_path, images, crf)
            files.append(video_path)
            if root is not None:
                stored_latents.append(loop_resume.save_window(work_dir, filename, w.number, keys[i], samples,
                                                              trim, next_start, written))
            reports.append(f"[{w.number}] wrote {written} frames to {os.path.basename(video_path)}")
            comfy.model_management.soft_empty_cache()
            mark("write")
            if review and src_pixels is not None:
                # The render over what was regenerated, in one picture (`video_mask.overlay_pieces`): built and
                # piped a cycle of frames at a time, so nothing the size of the window is held a second time.
                def window_review():
                    layers = video_mask.window_layers(src_mask, src_tokens, height, width, source["replace"], kept)
                    rows = video_mask.overlay_pieces(src_pixels, layers, int(src_tokens.shape[0]), int(trim),
                                                     f"{100.0 * float(src_tokens.mean()):.0f}% of this window regenerates")
                    _write_review_mp4(loop_resume.review_path(work_dir, filename, w.number),
                                      video_mask.render_over(images, rows), width, 2 * height, crf)
                why = _review_or_reason(window_review, f"window {w.number}")
                if why:
                    reports.append(f"[{w.number}] mask review FAILED; the window's video is unaffected: {why}")
                mark("mask review")
            del images, kept
            reports.append(f"[{w.number}] seconds: " + ", ".join(f"{name} {took:.1f}" for name, took in lap.items()))

        # join, and mux the whole track cut to the video
        out_path = os.path.join(full_out, stem + ".mp4")
        clock[0] = time.perf_counter()
        # a continued run's file starts on its first new frame: the track from there, and that many frames
        heard = waveform[..., int(round(head / FPS * rate)):]
        join_and_mux(files, heard, rate, out_path, work_dir, stem, kept_frames - head)
        mark("join and mux")
        review_files: list[str] = []
        if review:
            # One file beside the render: `<stem>_with_mask.mp4`, the windows' reviews joined as the windows
            # are. A reused window stored without one (before 2026-10-06, or with the switch off) has no
            # render in memory: its stored video is stacked over a view drawn now, with no outline, since
            # what the composite kept was never stored.
            restacked = []
            review_out = os.path.join(full_out, stem + "_with_mask.mp4")

            def joined_review():
                for rw in windows:
                    path = loop_resume.review_path(work_dir, filename, rw.number)
                    if rw.number <= first and not os.path.isfile(path):
                        stored = reused[rw.number - 1]
                        shape = _empty_av_latent(width, height, rw.frames)[0]["samples"].unbind()[0].shape[2:]
                        pixels, _encode, tokens, mask, _held = video_mask.window(
                            source, int(round(rw.start * FPS)), rw.frames, width, height, *shape)
                        del _encode
                        rows = video_mask.overlay_pieces(
                            pixels, video_mask.window_layers(mask, tokens, height, width, source["replace"]),
                            int(tokens.shape[0]),
                            int(stored["trim"]), f"{100.0 * float(tokens.mean()):.0f}% of this window regenerates")
                        # as many frames as the stored video holds, which is what this run writes of the
                        # window or it would not have been reused (`loop_resume.stored_frames`)
                        _write_review_mp4(path, video_mask.first_frames(rows, writes[rw.number - 1]), width, height,
                                          crf, under=stored["video"])
                        del pixels, tokens, mask
                        restacked.append(rw.number)
                    review_files.append(path)
                join_and_mux(review_files, heard, rate, review_out, work_dir, stem + "_with_mask", kept_frames - head)
            # A failure here, or a window's review missing because its own write failed, is a line of the
            # report; the render above is already joined, and its metadata, shot table and outputs follow
            # as they would with the switch off. A half-written joined review is removed. The next run
            # reuses the windows and stacks whichever reviews are missing from their stored videos.
            why = _review_or_reason(joined_review, "the run")
            if why:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(review_out)
            mark("mask review")
            reports.append(f"mask review FAILED; the render is complete and unaffected: {why}" if why else
                           f"mask review written beside the video: {os.path.basename(review_out)}, the render over "
                           "what was regenerated"
                           + (f"; window(s) {restacked} were stacked from their stored video, with no outline of "
                              "what the render kept" if restacked else ""))
        if spent:
            reports.append(f"seconds by stage, {sum(spent.values()):.0f} in all: "
                           + ", ".join(f"{name} {took:.1f}" for name, took in sorted(spent.items(), key=lambda kv: -kv[1])))
        png_path = (write_metadata_png(os.path.join(full_out, stem + ".png"), out_path, graph, extra)
                    if save_metadata_png else None)
        # the tracker's per-shot table, kept with the mask (`video_mask.py`), beside the video
        # under the same number: `<stem>_shots.json` and `<stem>_shots.md`
        table_files = shot_table.write_beside(source, full_out, stem)
        if table_files:
            reports.append("shot table written beside the video: " + ", ".join(table_files))
        if not keep_windows:
            for p in files + stored_latents + review_files:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(p)
            try:
                os.rmdir(work_dir)
            except OSError:
                pass  # windows from a longer earlier run are still there; they are that run's
        report = f"{first} reused -> {out_path}; " + "\n".join(reports)
        logger.info("[h3] MiniMaxH3AudioFreezeSong: %s", report.splitlines()[0])
        filenames, preview_ui = saved_outputs(out_path, subfolder, png_path)
        return io.NodeOutput(out_path, report, filenames, ui=preview_ui)
