#!/usr/bin/env python3
"""The audio-freeze node's contract, and the two ways a graph can silently unfreeze.

`docs/h3/audio_freeze.md` is the lane; `audio_freeze.py` is the node. Three
things here, each a way the freeze could look present and not be:

1. **The slice is on the grid and exact.** `slice_window` must return exactly
   `audio_t * hop` samples at the VAE's rate from a start snapped to the
   audio latent grid, pad only when the track runs out, and refuse a start
   past the end. An off-by-one here is a track that drifts against its own
   frames, which no schema check can see.
2. **The mask survives the sampler.** The nested mask the node writes is
   reshaped per stream by `comfy/samplers.py::CFGGuider.sample` through
   `comfy.utils.reshape_mask`; the audio part must come out with its values
   intact, and a flat mask on the way in must be refused, because a flat
   mask is what stock `SetLatentNoiseMask` writes and the sampler would pad
   the audio stream with ones.
3. **No shipped graph unfreezes by wiring.** Over `graph_paths()`: no graph
   carries `SetLatentNoiseMask` at all; every graph carrying the freeze node
   feeds the sampler from it, muxes its `clip_audio`, takes the preflight's
   latent, and has no audio decoder left to consume.
4. **A window's stored latent, and nothing reused by a key.** `loop_resume.py`:
   a window's sampled latent is written beside its video and reads back
   whole, with its trim, next start and frames written in its header and no
   key; a file that is not a stored window is refused in words; and the
   module holds no key function and the song node no reuse input (until
   2026-10-10 this item held the keys of stored-window reuse, removed under
   the owner's rule that nothing an earlier run made is reused by a key).
5. **The plan lines windows up with the timeline.** `loop_plan.py`: on the
   example song's timeline every entry after the first starts within half a
   `GRID` step of its time, and so does every interior part length in a sweep,
   against a control planner of full windows only, which misses; windows run
   longest first; entries past the covered length drop; a too-short entry, a
   bad timeline line, a label with no block, a block with no label and a cut
   past its window's end are refused; lists get one use per entry, or one per
   window with no timeline; `frames_read` is the furthest frame any placed
   window slices from the track, which is what a source loader has to hold;
   a run that cannot cover what it was asked for gets one line saying so
   (`extent_shortfall`: a track shorter than the extent, or a source whose
   picture ends before its track), and one that can gets none;
   and every input a preview skips is declared lazy on the song node, with a
   copy missing one refused.
6. **The join returns every frame its windows hold.** `loop_output.join_and_mux`
   through ffmpeg on two small windows, noisy and flat, under a track exactly
   as long as the video, a second shorter and a second longer: every frame
   back, evenly spaced, and the track padded or cut to the video. Under
   `-shortest` the first two lost a frame to four (2026-10-06). And
   `loop_plan.frames_kept`: every covered frame, or as far as the track runs,
   which is where a short track is cut, at the frames; `loop_plan.frames_written`,
   what each window's file holds, with a cut that never reaches what the last
   window writes on any window length, context or track length.
   And `bench/join_stretches.py`, which joins several runs' windows over
   consecutive stretches under the clip's own audio: every frame, evenly
   spaced, the audio as long as the video with a tone where the span puts
   it, the reviews joined from their own files, and each refusal (a window
   missing, a stretch given twice or out of order, a span that does not end
   where the frames do) with nothing written.
7. **A written file says what colour it holds, and it is true.** Bars of
   saturated colour through the song node's writer, alone and in the review's
   stacked form: each file is tagged BT.709 in matrix, transfer and primaries
   and tv in range, and read back as tagged it is nearer what was written
   than read back under BT.601. Until 2026-10-10 the
   files held BT.601 values with no tag, so a player that takes BT.709 for a
   picture this size showed a render off in colour beside its own source.

8. **A load's first frame fixes which cuts split a latent step.**
   `loop_plan.split_steps` (2026-10-10). Every window
   length and context the planner allows adds a whole number of step cycles,
   and `step_span` agrees with the frame count the window node takes for a
   latent length, so the grid is one per load. On the cuts and the subject's
   shots of one render of that day, loaded from source frame 14, the frames
   named are the five its delivery's own check found repainted (CHANGELOG
   0.254.0) and no other; **the control**: the same cuts counted from frame
   zero, as if the load's first frame did not matter, name other frames. On
   random cuts, shots and starts the frames named are those
   `video_mask.cut_gate` leaves unlaid, window by window, and a cut with the
   subject on both sides is `shared` and gated nowhere. Every start a cycle
   back is tried once, the asked one included, and the first is never worse
   than the asked one.

The encoder is faked (zeros of the right shape) so this runs with no model,
no CUDA and no server; the real audio VAE is exercised by
`bench/audit_audio_freeze_control.py` against the sibling pack's song node.

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_audio_freeze.py

Imports comfy, so the ComfyUI checkout two directories up must be importable;
the script adds it to `sys.path` itself.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
WORKFLOWS = REPO / "workflows"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WORKFLOWS))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context for a shape check; the sibling checks do the same
import comfy.nested_tensor  # noqa: E402
import comfy.utils  # noqa: E402
import h3_config  # noqa: E402


def _load_audio_freeze():
    """`audio_freeze` as a module of a stand-in package, the way `check_node_ids.py`
    loads `nodes`: its `from .audio_resample import` (0.139.0) has no parent
    package under a bare import, which left this check failing at import from
    2026-09-25 until 2026-10-01."""
    import importlib.util
    import types
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.audio_freeze", REPO / "audio_freeze.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.audio_freeze"] = module
    spec.loader.exec_module(module)
    return module


af = _load_audio_freeze()

FREEZE = "MiniMaxH3FreezeAudio"
WINDOW = "MiniMaxH3FreezeAudioWindow"
STOCK_MASK = "SetLatentNoiseMask"


class FakeAudioVAE:
    """The two things the node reads off a loaded H3 audio VAE, and an encode of the right shape."""
    audio_sample_rate = 32000

    def __init__(self, hop=800, t_override=None):
        self.hop = hop
        self.t_override = t_override

    def spacial_compression_encode(self):
        return self.hop

    def encode(self, samples_last):  # [B, samples, channels]
        t = self.t_override or samples_last.shape[1] // self.hop
        return torch.zeros(1, 32, 2, t)


def _fail(problems, msg):
    problems.append(msg)


def check_slice(problems):
    vae = FakeAudioVAE()
    rate = 44100
    song = torch.randn(1, 2, rate * 10)
    audio_t = 207  # what temporal_shape gives a 124-frame clip
    piece, start_step, vae_rate, padded = af.slice_window(song, rate, vae, 1.0, audio_t)
    if tuple(piece.shape) != (1, 2, audio_t * vae.hop):
        _fail(problems, f"slice shape {tuple(piece.shape)}, expected (1, 2, {audio_t * vae.hop})")
    if (start_step, vae_rate, padded) != (40, 32000, 0):
        _fail(problems, f"slice meta {(start_step, vae_rate, padded)}, expected (40, 32000, 0)")
    # snap: 1.02 s is 40.8 steps -> 41
    _, s2, _, _ = af.slice_window(song, rate, vae, 1.02, audio_t)
    if s2 != 41:
        _fail(problems, f"start 1.02s snapped to step {s2}, expected 41")
    # the track runs out: 10 s of song, start at 9 s, ask for 207 steps (5.175 s)
    piece, _, _, padded = af.slice_window(song, rate, vae, 9.0, audio_t)
    if tuple(piece.shape) != (1, 2, audio_t * vae.hop):
        _fail(problems, "padded slice is not the exact window length")
    want_pad = audio_t - (10 - 9) * 40
    if padded != want_pad:
        _fail(problems, f"padded steps {padded}, expected {want_pad}")
    try:
        af.slice_window(song, rate, vae, 11.0, audio_t)
        _fail(problems, "a start past the end of the track was accepted")
    except ValueError:
        pass
    # mono is duplicated, three channels refused
    w, _, _ = af._stereo({"waveform": torch.zeros(1, 1, 100), "sample_rate": 32000})
    if tuple(w.shape) != (1, 2, 100):
        _fail(problems, f"mono became {tuple(w.shape)}, expected (1, 2, 100)")
    w3, _, notes = af._stereo({"waveform": torch.ones(1, 6, 100), "sample_rate": 32000})
    if tuple(w3.shape) != (1, 2, 100) or not notes or "downmixed" not in notes[0]:
        _fail(problems, f"5.1 input was not downmixed to stereo with a note: {tuple(w3.shape)} {notes}")
    # ffmpeg's normalisation: all-ones input lands at exactly full scale, not above it
    if abs(float(w3.abs().max()) - 1.0) > 1e-6:
        _fail(problems, f"downmix of an all-ones 5.1 input peaks at {float(w3.abs().max()):.4f}, expected 1.0")
    hot = torch.full((1, 2, 800 * 4), 1.5)
    guarded, gn = af.condition_level(hot, "clip_guard")
    if float(guarded.abs().max()) > 1.0 or not gn:
        _fail(problems, "clip_guard did not pull a 1.5 peak under full scale")
    tone = 0.5 * torch.sin(torch.linspace(0, 40 * 3.14159, 800)).expand(1, 2, 800)
    same, sn = af.condition_level(tone, "clip_guard")
    if abs(float(same.abs().max()) - float(tone.abs().max())) > 1e-6 or sn:
        _fail(problems, f"clip_guard changed a well-formed window: {sn}")
    raw, rn = af.condition_level(hot, "none")
    if float(raw.abs().max()) != 1.5 or rn:
        _fail(problems, "level=none touched the window")
    offset = torch.full((1, 2, 800), 0.2) + torch.linspace(-0.1, 0.1, 800)
    fixed, on = af.condition_level(offset, "clip_guard")
    if abs(float(fixed.mean())) > 1e-5 or not on:
        _fail(problems, f"a DC offset was not removed: mean {float(fixed.mean()):.4f} {on}")


def check_masks(problems):
    video = torch.zeros(1, 24, 37, 48, 84)
    audio = torch.zeros(1, 32, 2, 207)
    mask = af.freeze_masks(video, audio, 0.0)
    vm, am = mask.unbind()
    if tuple(vm.shape) != (1, 1, 37, 48, 84) or tuple(am.shape) != (1, 1, 2, 207):
        _fail(problems, f"mask shapes {tuple(vm.shape)} {tuple(am.shape)}")
    if vm.min() != 1.0 or am.max() != 0.0:
        _fail(problems, "video mask is not all ones or audio mask is not all zeros")
    # the sampler's own reshape, per stream, against the latent shapes
    vm2 = comfy.utils.reshape_mask(vm, tuple(video.shape))
    am2 = comfy.utils.reshape_mask(am, tuple(audio.shape))
    if tuple(vm2.shape) != tuple(video.shape) or tuple(am2.shape) != tuple(audio.shape):
        _fail(problems, f"reshape_mask gave {tuple(vm2.shape)} {tuple(am2.shape)}")
    if vm2.min() != 1.0 or am2.max() != 0.0:
        _fail(problems, "mask values did not survive the sampler's reshape")
    # a graded value survives too, and so does a half-and-half audio mask
    am3 = comfy.utils.reshape_mask(af.freeze_masks(video, audio, 0.25).unbind()[1], tuple(audio.shape))
    if not torch.allclose(am3, torch.full_like(am3, 0.25)):
        _fail(problems, "a graded audio mask value did not survive reshape")
    # flat incoming mask refused; nested incoming video mask kept
    try:
        af.freeze_masks(video, audio, 0.0, existing=torch.ones(1, 1, 48, 84))
        _fail(problems, "a flat incoming noise_mask was accepted")
    except ValueError:
        pass
    keep = torch.ones(1, 1, 37, 48, 84)
    keep[:, :, :5] = 0.0
    nested = comfy.nested_tensor.NestedTensor((keep, torch.ones(1, 1, 2, 207)))
    vm4, am4 = af.freeze_masks(video, audio, 0.0, existing=nested).unbind()
    if vm4[:, :, :5].max() != 0.0 or vm4[:, :, 5:].min() != 1.0 or am4.max() != 0.0:
        _fail(problems, "an incoming nested video mask was not kept, or audio was not re-frozen")


def check_window_geometry(problems):
    # 345 frames: 102 latent steps. 39 frames is 12 steps, phase-aligned, on the audio grid.
    geo = af.window_geometry(102, 39)
    if (geo["context_steps"], geo["stride_frames"]) != (12, 306):
        _fail(problems, f"window geometry for 345/39 is {geo}")
    if abs(geo["stride_seconds"] * 40 - round(geo["stride_seconds"] * 40)) > 1e-9:
        _fail(problems, "the stride does not land on the audio grid")
    for bad in (17, 22, 56, 40):
        try:
            af.window_geometry(102, bad)
            _fail(problems, f"context {bad} was accepted")
        except ValueError:
            pass
    # No zero mode (owner rule 2026-09-13): 0 is refused by the geometry, and
    # a first window gets its full-window stride from `no_context_geometry`,
    # reached by leaving `previous` unwired, never by a widget value.
    try:
        af.window_geometry(102, 0)
        _fail(problems, "context 0 was accepted; a number must not mean a mode")
    except ValueError:
        pass
    if af.no_context_geometry(102)["stride_frames"] != 345:
        _fail(problems, "a first window did not get a full-window stride")
    # The loop plan mixes window lengths, and the window node checks the
    # previous window's length against the context it hands on, so every
    # length on both clocks must pass at every context shorter than it.
    import loop_plan as lp
    steps = {af.pixel_frames(t): t for t in range(1, 120)}
    for n in lp.CHAIN_LENGTHS:
        for c in (39, 90, 141):
            if c < n:
                try:
                    af.window_geometry(steps[n], c)
                except ValueError as exc:
                    _fail(problems, f"a {n}-frame window cannot hand on a {c}-frame context: {exc}")
    # window 1 copies the previous tail into the head and freezes it
    video = torch.randn(1, 24, 102, 48, 84)
    audio = torch.randn(1, 32, 2, 575)
    fresh = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros_like(video), torch.zeros_like(audio)))}
    prev = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    song = {"waveform": torch.randn(1, 2, 44100 * 40), "sample_rate": 44100}
    # The first window of a chain: no previous, the widget at its real value.
    # This is the call the song node makes for window 0; until 2026-09-13 it
    # passed 0 here and the first run after the zero mode was removed raised.
    first = getattr(af.MiniMaxH3FreezeAudioWindow.execute(
        fresh, FakeAudioVAE(), song, 0.0, 39, previous=None), "args", None)
    if first[3] != 0 or abs(first[4] - 306 / 24) > 1e-9:
        _fail(problems, f"a first window trimmed {first[3]} or set next_start {first[4]}; expected 0 and the 39-context stride")
    fvm, _fam = first[0]["noise_mask"].unbind()
    if fvm.min() != 1.0:
        _fail(problems, "a first window froze video rows with nothing to copy from")
    # A WINDOW THAT STANDS ALONE (2026-10-13): one no window can follow, because it is off the audio grid or no longer than
    # the context a next one would keep. Nothing is asked of a next window's geometry; it ends where its frames end, its
    # audio is core's rounded count cut exactly, and it freezes no video row. The widget's context is whatever the job gave.
    for frames, steps, context in ((39, 12, 90), (56, 17, 90), (22, 7, 39), (90, 27, 90), (107, 32, 90)):
        audio_t = round(frames / 24 * 40)
        alone = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, steps, 48, 84), torch.zeros(1, 32, 2, audio_t)))}
        try:
            got = getattr(af.MiniMaxH3FreezeAudioWindow.execute(alone, FakeAudioVAE(), song, 2.0, context, previous=None), "args", None)
        except ValueError as e:
            _fail(problems, f"a {frames}-frame window that stands alone (context widget {context}) was refused: {e}")
            continue
        if af.pixel_frames(steps) != frames or got[3] != 0 or abs(got[4] - (2.0 + frames / 24)) > 1e-9 or "no window can follow this one" not in got[5]:
            _fail(problems, f"a {frames}-frame window that stands alone: trim {got[3]}, next start {got[4]} (its own end is {2.0 + frames / 24}), report {got[5][:120]!r}")
        if got[0]["noise_mask"].unbind()[0].min() != 1.0 or int(got[0]["samples"].unbind()[1].shape[-1]) != audio_t:
            _fail(problems, f"a {frames}-frame window that stands alone froze a video row or holds another count of audio steps than core's rounded {audio_t}")
    # the controls: a window that CAN be followed still gives the next start by its stride (a 90-frame first window with a
    # 39-frame context), and a window with something before it is still held to the chained geometry, whatever its length
    chained = getattr(af.MiniMaxH3FreezeAudioWindow.execute(
        {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 27, 48, 84), torch.zeros(1, 32, 2, 150)))}, FakeAudioVAE(), song, 0.0, 39, previous=None), "args", None)
    if abs(chained[4] - 51 / 24) > 1e-9 or "no window can follow" in chained[5]:
        _fail(problems, f"a 90-frame first window with a 39-frame context no longer gives the next window's start by its stride: {chained[4]}")
    try:
        af.MiniMaxH3FreezeAudioWindow.execute(
            {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 17, 48, 84), torch.zeros(1, 32, 2, 93)))}, FakeAudioVAE(), song, 2.0, 39,
            previous={"samples": comfy.nested_tensor.NestedTensor((torch.randn(1, 24, 17, 48, 84), torch.randn(1, 32, 2, 93)))})
        _fail(problems, "a 56-frame window with a window BEFORE it was taken: a chained window off the audio grid")
    except ValueError:
        pass
    out = af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 306 / 24, 39, previous=prev)
    args = getattr(out, "args", out)
    lat, _clip, span, trim, next_start, _rep, new_audio = args
    if new_audio["waveform"].shape[-1] != (575 - 65) * 800:
        _fail(problems, f"new_audio is {new_audio['waveform'].shape[-1]} samples, expected the window minus 65 context steps")
    v2, _ = lat["samples"].unbind()
    if not torch.equal(v2[:, :, :12], video[:, :, 90:]):
        _fail(problems, "window 1 did not copy the previous window's last 12 latent steps into its head")
    if v2[:, :, 12:].abs().max() != 0.0:
        _fail(problems, "window 1 wrote outside the context")
    # `context_noise` (2026-10-10): its default writes the mask the node always wrote, bit for bit; a value
    # lands on the context's steps and nowhere else, changes no latent, and is said in the report
    named = getattr(af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 306 / 24, 39, previous=prev,
                                                          context_noise=af.CONTEXT_NOISE), "args", None)
    for a, b in zip(lat["noise_mask"].unbind(), named[0]["noise_mask"].unbind()):
        if not torch.equal(a, b):
            _fail(problems, "context_noise at its default does not write the mask the node writes with the input unwired")
    soft = getattr(af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 306 / 24, 39, previous=prev,
                                                         context_noise=0.5), "args", None)
    svm, sam = soft[0]["noise_mask"].unbind()
    if not bool((svm[:, :, :12] == 0.5).all()) or svm[:, :, 12:].min() != 1.0 or sam.max() != 0.0:
        _fail(problems, "context_noise 0.5 must put 0.5 on the context's 12 latent steps and leave every other row as it was")
    if not torch.equal(soft[0]["samples"].unbind()[0], v2) or soft[3] != trim or soft[4] != next_start:
        _fail(problems, "context_noise changed the window's latent, its trim or where the next window starts")
    if "shown at noise 0.5" not in soft[5] or "shown at noise" in _rep:
        _fail(problems, f"the report says the context's noise only when it is above 0: {soft[5]!r}")
    unseen = getattr(af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 0.0, 39, previous=None,
                                                           context_noise=0.5), "args", None)
    if unseen[0]["noise_mask"].unbind()[0].min() != 1.0:
        _fail(problems, "context_noise touched a first window, which has no context")
    for bad in (-0.1, 1.5):
        try:
            af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 306 / 24, 39, previous=prev, context_noise=bad)
            _fail(problems, f"context_noise {bad} was accepted")
        except ValueError:
            pass
    if af.CONTEXT_NOISE != 0.0:
        _fail(problems, f"CONTEXT_NOISE is {af.CONTEXT_NOISE}: the default must stay the mask every render so far was made with")
    for name in ("audio_freeze.py", "audio_freeze_song.py"):
        text = (REPO / name).read_text(encoding="utf-8")
        # appended to the end, optional, so a graph saved before it keeps its inputs where they were; an input appended after it
        # (the song node's save_kept, 2026-10-15) is held to the same: optional, never in front of it
        at = text.find('io.Float.Input("context_noise", default=CONTEXT_NOISE, min=0.0, max=1.0, step=0.05, optional=True,')
        later = re.split(r"\bio\.\w+\.Input\(", text[at + 20:text.find("outputs=[", at)])[1:] if at >= 0 else []
        if at < 0 or any("optional=True" not in one for one in later):
            _fail(problems, f"{name}: context_noise must be optional at the shared default, after every older input, and every input "
                            "appended after it optional too")
    if "context_noise=context_noise)" not in (REPO / "audio_freeze_song.py").read_text(encoding="utf-8"):
        _fail(problems, "the song node does not hand its context_noise to the window node")
    if "may not start where the last one ended" not in soft[5]:
        _fail(problems, f"the report does not say what a context shown with noise costs at the seam: {soft[5]!r}")
    vm, am = lat["noise_mask"].unbind()
    if vm[:, :, :12].max() != 0.0 or vm[:, :, 12:].min() != 1.0 or am.max() != 0.0:
        _fail(problems, "window 1's masks are wrong")
    if trim != 39 or abs(next_start - 2 * 306 / 24) > 1e-9:
        _fail(problems, f"window 1 trim/next_start {trim} {next_start}")
    # windows may differ in length: a 141-frame window (42 steps, 235 audio steps) after the 345-frame one
    short = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 42, 48, 84), torch.zeros(1, 32, 2, 235)))}
    out2 = getattr(af.MiniMaxH3FreezeAudioWindow.execute(short, FakeAudioVAE(), song, 306 / 24, 39, previous=prev), "args", None)
    v3, _ = out2[0]["samples"].unbind()
    if not torch.equal(v3[:, :, :12], video[:, :, 90:]) or out2[3] != 39:
        _fail(problems, "a shorter second window did not take the previous window's tail")
    if abs(out2[4] - (306 / 24 + (141 - 39) / 24)) > 1e-9:
        _fail(problems, f"next_start after a 141-frame window is {out2[4]}")
    # 124 frames is a legal render length but not a chain window: its stride is off the audio grid
    try:
        af.window_geometry(37, 39)
        _fail(problems, "a 124-frame chained window was accepted")
    except ValueError:
        pass
    if span["waveform"].shape[-1] != (510 + 575) * 800:
        _fail(problems, f"span audio is {span['waveform'].shape[-1]} samples, expected {(510 + 575) * 800}")
    # no previous: no context, trim 0, whatever the start
    out3 = getattr(af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 306 / 24, 39), "args", None)
    if out3[3] != 0 or out3[0]["noise_mask"].unbind()[0].min() != 1.0:
        _fail(problems, "a window with no previous latent froze video context")


def check_execute(problems):
    video = torch.randn(1, 24, 37, 48, 84)
    audio = torch.randn(1, 32, 2, 207)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    song = {"waveform": torch.randn(1, 2, 44100 * 8), "sample_rate": 44100}
    out = af.MiniMaxH3FreezeAudio.execute(latent, FakeAudioVAE(), song, 0.5, 0.0)
    args = getattr(out, "args", out)
    new_latent, clip, report = args[0], args[1], args[2]
    v2, a2 = new_latent["samples"].unbind()
    if not torch.equal(v2, video):
        _fail(problems, "execute changed the video stream")
    if tuple(a2.shape) != tuple(audio.shape) or a2.abs().max() != 0.0:
        _fail(problems, "execute did not write the encoded (fake, zero) audio into the audio rows")
    if "noise_mask" not in new_latent or not getattr(new_latent["noise_mask"], "is_nested", False):
        _fail(problems, "execute did not attach a nested noise_mask")
    if clip["sample_rate"] != 32000 or tuple(clip["waveform"].shape) != (1, 2, 207 * 800):
        _fail(problems, f"clip_audio is {clip['sample_rate']} Hz {tuple(clip['waveform'].shape)}")
    if "froze 207" not in report or "step 20 " not in report:
        _fail(problems, f"report does not say what happened: {report!r}")
    # the encoder landing off the grid is a contract error, not padded over
    try:
        af.MiniMaxH3FreezeAudio.execute(latent, FakeAudioVAE(t_override=206), song, 0.0, 0.0)
        _fail(problems, "an encoder returning the wrong step count was accepted")
    except ValueError:
        pass
    # a plain-tensor latent is refused with a sentence
    try:
        af.MiniMaxH3FreezeAudio.execute({"samples": video}, FakeAudioVAE(), song, 0.0, 0.0)
        _fail(problems, "a plain (non-AV) latent was accepted")
    except ValueError:
        pass


def check_graphs(problems) -> tuple[int, int]:
    paths = h3_config.graph_paths(WORKFLOWS, "*_api.json")
    frozen = 0
    for p in paths:
        g = json.loads(p.read_text())
        classes = {nid: n.get("class_type") for nid, n in g.items()}
        if STOCK_MASK in classes.values():
            _fail(problems, f"{p.name}: carries {STOCK_MASK}, which flattens an AV noise_mask")
        ids = [nid for nid, ct in classes.items() if ct in (FREEZE, WINDOW)]
        if not ids:
            continue
        frozen += 1
        for fid in ids:
            src = g[fid]["inputs"].get("latent")
            ok = src == ["26", 1] or (isinstance(src, list) and classes.get(src[0]) == "MiniMaxH3Conditioning" and src[1] == 1)
            if not ok:
                _fail(problems, f"{p.name}: node {fid} takes its latent from {src}, not the preflight or a conditioner")
        samplers = [nid for nid, ct in classes.items() if ct == "SamplerCustomAdvanced"]
        for sid in samplers:
            src = g[sid]["inputs"].get("latent_image")
            if not (isinstance(src, list) and src[0] in ids and src[1] == 0):
                _fail(problems, f"{p.name}: sampler {sid} does not take a frozen latent ({src})")
        muxers = [nid for nid, ct in classes.items() if ct == "VHS_VideoCombine"]
        for mid in muxers:
            src = g[mid]["inputs"].get("audio")
            ok = isinstance(src, list) and src[0] in ids and (
                (classes[src[0]] == FREEZE and src[1] == 1) or (classes[src[0]] == WINDOW and src[1] in (2, 6)))
            if not ok:
                _fail(problems, f"{p.name}: muxer {mid} does not take a freeze node's track ({src})")
        if "VAEDecodeAudio" in classes.values():
            _fail(problems, f"{p.name}: an audio decoder is still present on a freeze graph")
        windows = [nid for nid in ids if classes[nid] == WINDOW]
        for wid in windows:
            w = g[wid]["inputs"]
            if w.get("previous") and not (isinstance(w.get("start_seconds"), list) or w.get("start_seconds")):
                _fail(problems, f"{p.name}: window node {wid} has a previous latent but starts at 0")
    return len(paths), frozen


def check_resume(problems):
    """A window's stored latent: written beside its video, read back whole by the two readers a person names
    (`continue_from`, the re-lay tool), and refused when the file is not one. Nothing is keyed and nothing is
    reused (2026-10-10): the module holds no key function, and the song node has no reuse input."""
    import os
    import tempfile
    import loop_resume as lr

    with tempfile.TemporaryDirectory() as d:
        video, audio = torch.randn(1, 24, 3, 4, 6), torch.randn(1, 32, 2, 5)
        samples = comfy.nested_tensor.NestedTensor((video, audio))
        path = lr.save_window(d, "song", 1, samples, 39, 5.875, 97)
        if path != lr.window_paths(d, "song", 1)[1] or not os.path.isfile(path):
            _fail(problems, f"store: a window's latent was written to {path}, not beside its video")
        back = lr.load_window_latent(path)["samples"]
        if not getattr(back, "is_nested", False) or not all(torch.equal(a, b) for a, b in zip(back.unbind(), (video, audio))):
            _fail(problems, "store: a stored latent did not round-trip")
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        if (meta.get("trim"), meta.get("next_start"), meta.get("written")) != ("39", "5.875", "97") or "key" in meta:
            _fail(problems, f"store: a stored window's header is {meta}; it holds the trim, the next start and the frames written, and no key")
        other = os.path.join(d, "other.safetensors")            # a latent some other node saved: no trim in it
        comfy.utils.save_torch_file({"samples": torch.zeros(1, 4, 3, 2, 2)}, other, metadata={"note": "not a window"})
        for bad in (os.path.join(d, "absent.safetensors"), __file__, other):
            for reader in (lr.load_window_latent, lr.check_window_latent):
                try:
                    reader(bad)
                    _fail(problems, f"store: {os.path.basename(bad)} was taken for a stored window by {reader.__name__}")
                except ValueError as exc:
                    if "is not a stored window's latent" not in str(exc):
                        _fail(problems, f"store: the refusal of {os.path.basename(bad)} does not say what it is not: {exc}")
        lr.check_window_latent(path)            # and the real one passes, by its header alone
    gone = [name for name in ("graph_signature", "root_key", "window_key", "track_hash", "stored_key", "read_window",
                              "stored_frames", "SONG_PER_WINDOW") if hasattr(lr, name)]
    song_source = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    # a wrong `continue_from` is refused where the run starts: ahead of the preview's return and of every encode
    at = song_source.find("loop_resume.check_window_latent(continue_from)")
    if at < 0 or at > song_source.find("        if preview:\n            if source is not None:"):
        _fail(problems, "store: the song node does not check the file `continue_from` names before its preview returns")
    if gone or '"reuse_windows"' in song_source or "window_keep" in song_source:
        _fail(problems, f"store: nothing an earlier run made is reused by a key, and {gone or 'the song node'} still keys or reuses one")


#: The example song's sections (`build_workflows._SONG_FLICKER_TIMELINE`) and
#: its length in frames, rounded up from ffprobe's duration of
#: `just-a-flicker.mp3` (read 2026-09-14): a fixture, not a claim about the file.
FLICKER_TIMELINE = "00:00 intro\n00:10 verse\n00:32 chorus\n00:53 verse\n01:14 chorus\n01:35 bridge\n01:57 chorus\n02:18 outro"
FLICKER_FRAMES = 3908


def _refused(problems, label, call, needle=None):
    try:
        call()
        _fail(problems, f"{label} was accepted")
    except ValueError as exc:
        if needle is not None and needle not in str(exc):
            _fail(problems, f"{label}: the refusal did not name {needle!r} ({exc})")


def lazy_problems(source: str) -> list[str]:
    """Names in the song node's LAZY that its schema does not declare lazy=True, and LAZY itself missing."""
    import ast
    tree = ast.parse(source)
    lazy = next((ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "LAZY" for t in n.targets)), None)
    if not lazy:
        return ["no LAZY tuple"]
    declared = {c.args[0].value for c in ast.walk(tree)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "Input"
                and c.args and isinstance(c.args[0], ast.Constant)
                and any(k.arg == "lazy" and isinstance(k.value, ast.Constant) and k.value.value is True
                        for k in c.keywords)}
    return [f"{name} is in LAZY but not declared lazy" for name in lazy if name not in declared]


def check_join(problems):
    """`loop_output.join_and_mux` returns every frame its files hold, in order, whatever the track's length.

    The cases that lost frames under `-shortest` (2026-10-06): two windows under a track exactly as long
    as the video and under a shorter one, on noisy frames and on flat ones, which x264 packs differently.
    Reading a red against the old command: "exactly as long" fails because of where ffmpeg stops a copied
    stream, which is the bug; "a second shorter" fails by construction, since the old join cut the video
    to its track on purpose and the new one is handed frames already cut, so that case states the contract.
    A `loop_output.py` from before the change is a TypeError at the call, not a FAIL line.

    The windows are written by the song node's own writer and the files are read back by the ffmpeg
    the node uses (`audio_freeze._ffmpeg`), as packets, so the case runs wherever the node does and a
    changed encoder flag is tested as changed. The last pair of lengths ends on a one-frame file, which
    a context of 90 or 141 can plan.
    """
    import importlib
    import re
    import subprocess
    import tempfile
    assert "_h3pack" in sys.modules
    lo = importlib.import_module("_h3pack.loop_output")
    song = importlib.import_module("_h3pack.audio_freeze_song")
    ffmpeg, fps = lo._ffmpeg(), int(lo.FPS)

    def packets(path: str, stream: str) -> tuple[list[int], float]:
        """(every packet's timestamp, the stream's end in seconds) of one stream, copied to ffmpeg's framecrc."""
        text = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", f"0:{stream}:0", "-c", "copy", "-f", "framecrc", "-"],
                              capture_output=True, text=True).stdout
        base = re.search(r"#tb 0: (\d+)/(\d+)", text)
        rows = [[int(x) for x in line.split(",")[:4]] for line in text.splitlines() if line and not line.startswith("#")]
        if base is None or not rows:
            return [], 0.0
        end = max(pts + duration for _stream, _dts, pts, duration in rows) * int(base.group(1)) / int(base.group(2))
        return sorted(pts for _stream, _dts, pts, _duration in rows), end

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        # (22, 17): the sizes the loss was first seen at, with a 5-frame trim; (22, 1): the shortest last file
        for lengths in ((22, 17), (22, 1)):
            total = sum(lengths)
            for kind in ("noisy", "flat"):
                files = []
                for i, n in enumerate(lengths):
                    path = str(Path(tmp) / f"{kind}_{i}.mp4")
                    frames = torch.rand(n, 96, 160, 3) if kind == "noisy" else torch.full((n, 96, 160, 3), 0.5)
                    if song._write_frames_mp4(path, frames, 19) != n:
                        _fail(problems, f"join: the writer did not report the {n} frames it was given")
                    files.append(path)
                for label, seconds in (("exactly as long as the video", total / fps), ("a second shorter", total / fps - 1.0),
                                       ("a second longer", total / fps + 1.0)):
                    out = str(Path(tmp) / "joined.mp4")
                    samples = max(int(round(44100 * seconds)), 1)
                    lo.join_and_mux(files, torch.zeros(1, 2, samples), 44100, out, tmp, "joined", total)
                    stamps, _end = packets(out, "v")
                    steps = {b - a for a, b in zip(stamps, stamps[1:])}
                    if len(stamps) != total or len(steps) != 1:
                        _fail(problems, f"join: {kind} windows of {lengths} under a track {label} came back with "
                                        f"{len(stamps)} frames of {total}" + ("" if len(steps) == 1 else ", not evenly spaced"))
                    _stamps, audio = packets(out, "a")
                    if abs(audio - total / fps) > 0.05:
                        _fail(problems, f"join: the track under {kind} windows of {lengths} ends at {audio:.3f}s "
                                        f"for a {total / fps:.3f}s video")


def check_writer_colour(problems):
    """The song node's writer: the tag on a file, and whether the values under it match.

    The control is the pair of readings: on these bars the wrong matrix is several times further from what
    was written, so a file tagged one way and converted the other fails, and so does a file with no tag.
    """
    import importlib
    import subprocess
    import tempfile
    lo = importlib.import_module("_h3pack.loop_output")
    song = importlib.import_module("_h3pack.audio_freeze_song")
    ffmpeg = lo._ffmpeg()
    width, height, count = 320, 96, 6
    bars = torch.tensor([[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0.8, 0.45, 0.35]])
    frames = bars.repeat_interleave(width // len(bars), dim=0)[None, None].expand(count, height, width, 3).contiguous()

    def tag(path: str) -> str:
        """The four things a file says about its colour, as ffprobe orders them: range, matrix, transfer, primaries."""
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=color_range,color_space,color_transfer,color_primaries",
                               "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()

    def off(path: str, matrix: str | None, crop: str | None = None) -> float:
        """Mean distance, in levels, between what was written and the file read back under `matrix` (None: its tag)."""
        steps = ([crop] if crop else []) + ([f"scale=in_color_matrix={matrix}"] if matrix else [])
        raw = subprocess.run([ffmpeg, "-v", "error", "-i", path, *(["-vf", ",".join(steps)] if steps else []),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        if len(raw) != count * height * width * 3:
            return float("inf")
        back = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(count, height, width, 3).float()
        return float((back - frames * 255.0).abs().mean())

    with tempfile.TemporaryDirectory() as tmp:
        plain = str(Path(tmp) / "plain.mp4")
        stacked = str(Path(tmp) / "stacked_with_mask.mp4")
        song._write_frames_mp4(plain, frames, 19)
        # a review is one picture of twice the height: the render over the view of what was regenerated
        song._write_review_mp4(stacked, iter([torch.cat([frames, frames], dim=1)]), width, 2 * height, 19)
        for label, path, crop in (("a render", plain, None),
                                  ("a review, the render's half", stacked, f"crop={width}:{height}:0:0"),
                                  ("a review, the drawn half", stacked, f"crop={width}:{height}:0:{height}")):
            if tag(path) != "tv,bt709,bt709,bt709":
                _fail(problems, f"writer colour: {label} is tagged {tag(path) or 'nothing'} "
                                f"(range, matrix, transfer, primaries), not tv,bt709,bt709,bt709")
            tagged, other = off(path, None, crop), off(path, "bt601", crop)
            if not tagged * 2 < other:
                _fail(problems, f"writer colour: {label} read back as tagged is {tagged:.2f} levels from what was "
                                f"written and {other:.2f} under BT.601; its values are not what its tag says")


def check_join_stretches(problems):
    """`bench/join_stretches.py`: several runs' windows, one file, the clip's own audio over the span.

    Two stand-in runs of two windows each, written by the song node's writer, and a clip whose audio
    is silent but for a tone that begins half a second into the span: the joined file holds every
    frame evenly spaced, its audio is as long as its video, and the tone is where the span puts it,
    which is what a wrong seek would move. Then each way a hand-run goes wrong is refused with nothing
    written: a window missing, a stretch given twice, the stretches out of order, a span that does not
    end where the frames do. The reviews join the same way from their own files.
    """
    import importlib
    import importlib.util
    import math
    import os
    import re
    import subprocess
    import tempfile
    lo = importlib.import_module("_h3pack.loop_output")
    song = importlib.import_module("_h3pack.audio_freeze_song")
    spec = importlib.util.spec_from_file_location("_h3_join_stretches", HERE / "join_stretches.py")
    js = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = js
    spec.loader.exec_module(js)
    ffmpeg, fps, rate = lo._ffmpeg(), int(lo.FPS), 44100
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        lengths = {"a": (22, 17), "b": (12, 9)}
        for run, counts in lengths.items():
            os.makedirs(os.path.join(tmp, f"{run}_windows"))
            for i, n in enumerate(counts, start=1):
                for suffix, height in (("", 96), ("_with_mask", 192)):
                    song._write_frames_mp4(os.path.join(tmp, f"{run}_windows", f"{run}_00001_window_{i}{suffix}.mp4"),
                                           torch.rand(n, height, 160, 3), 19)
            open(os.path.join(tmp, f"{run}_windows", f"{run}_00001_window_1.safetensors"), "wb").close()   # a latent beside them
        start, total = 3.0, sum(sum(c) for c in lengths.values())
        split = start + sum(lengths["a"]) / fps
        end = start + total / fps
        clip = os.path.join(tmp, "clip.wav")
        wave = torch.zeros(1, 2, int(rate * (end + 2.0)))
        tone_at = int(round((start + 0.5) * rate))
        wave[..., tone_at:tone_at + rate // 4] = 0.5 * torch.sin(torch.arange(rate // 4) * 2 * math.pi * 440 / rate)
        lo._write_wav(clip, wave, rate)
        a, b = os.path.join(tmp, "a_windows"), os.path.join(tmp, "b_windows")

        def packets(path, stream):
            text = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", f"0:{stream}:0", "-c", "copy", "-f", "framecrc", "-"],
                                  capture_output=True, text=True).stdout
            return sorted(int(line.split(",")[2]) for line in text.splitlines() if line and not line.startswith("#"))
        for review in (False, True):
            out = os.path.join(tmp, "joined_review.mp4" if review else "joined.mp4")
            try:
                wrote = js.join(clip, start, end, [(a, start), (b, split)], out, review=review)
            except Exception as exc:  # noqa: BLE001 -- a refusal of the right inputs is the finding
                _fail(problems, f"join stretches: two runs that tile their span were not joined ({type(exc).__name__}: {exc})")
                return
            stamps = packets(out, "v")
            steps = {y - x for x, y in zip(stamps, stamps[1:])}
            if wrote != total or len(stamps) != total or len(steps) != 1:
                _fail(problems, f"join stretches: {total} frames in four {'reviews' if review else 'windows'} came back as "
                                f"{len(stamps)}" + ("" if len(steps) == 1 else ", not evenly spaced"))
            pcm = subprocess.run([ffmpeg, "-v", "error", "-i", out, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                                 capture_output=True).stdout
            heard = torch.frombuffer(bytearray(pcm), dtype=torch.float32)
            loud = (heard.abs() > 0.1).nonzero()
            if abs(heard.numel() / rate - total / fps) > 0.05:
                _fail(problems, f"join stretches: the audio is {heard.numel() / rate:.3f}s under a {total / fps:.3f}s video")
            size = re.search(r", (\d+)x(\d+)[, ]", subprocess.run([ffmpeg, "-hide_banner", "-i", out], capture_output=True, text=True).stderr)
            if size is None or int(size.group(2)) != (192 if review else 96):
                _fail(problems, f"join stretches: asked for the {'reviews' if review else 'windows'}, the joined picture is "
                                f"{size.group(0).strip(', ') if size else 'unreadable'}")
            if not loud.numel() or abs(int(loud[0]) / rate - 0.5) > 0.03:
                _fail(problems, "join stretches: the clip's audio is not where the span puts it: a tone half a second into "
                                f"the span is heard at {int(loud[0]) / rate:.3f}s" if loud.numel() else
                                "join stretches: the clip's audio over the span is silent in the joined file")
        before = Path(tmp, "joined.mp4").read_bytes()
        try:
            js.join(clip, start, end, [(a, start), (b, split)], os.path.join(tmp, "joined.mp4"))
            _fail(problems, "join stretches: an existing file was overwritten without being asked")
        except ValueError as exc:
            if "exists" not in str(exc):
                _fail(problems, f"join stretches: an existing file was refused for another reason: {str(exc)[:120]}")
        if Path(tmp, "joined.mp4").read_bytes() != before:
            _fail(problems, "join stretches: an existing file was changed without being asked")
        left = sorted(os.listdir(tmp))
        # a join that comes out with the wrong count is refused after the fact, and takes its file with it
        honest = js.count_frames
        js.count_frames = lambda tool, path: honest(tool, path) + (1 if path.endswith(".part.mp4") else 0)
        try:
            js.join(clip, start, end, [(a, start), (b, split)], os.path.join(tmp, "miscounted.mp4"))
            _fail(problems, "join stretches: a joined file with the wrong frame count was kept")
        except ValueError:
            pass
        finally:
            js.count_frames = honest
        if sorted(os.listdir(tmp)) != left:
            _fail(problems, f"join stretches: a join refused after the fact left {sorted(set(os.listdir(tmp)) - set(left))} behind")
        refusals = (("a stretch given twice", [(a, start), (a, split)], end, "twice"),
                    ("the stretches out of order", [(b, split), (a, start)], end, "out of order"),
                    ("a stretch left out", [(b, split)], end, "missing"),
                    ("a span that ends past the frames", [(a, start), (b, split)], end + 1.0, "span"))
        for label, stretches, to, word in refusals:
            out = os.path.join(tmp, "refused.mp4")
            try:
                js.join(clip, start, to, stretches, out)
                _fail(problems, f"join stretches: {label} was joined, not refused")
            except ValueError as exc:
                if word and word not in str(exc):
                    _fail(problems, f"join stretches: {label} was refused without saying so: {str(exc)[:120]}")
            if os.path.exists(out) or sorted(os.listdir(tmp)) != left:
                _fail(problems, f"join stretches: {label} was refused but left a file behind")
        # What it cannot catch, and does not claim to: stretches swapped WITH their start times swapped to
        # match. The files do not say where in the clip they belong; the starts given are taken as true.
        os.remove(os.path.join(b, "b_00001_window_1.mp4"))
        try:
            js.join(clip, start, end, [(a, start), (b, split)], os.path.join(tmp, "refused.mp4"))
            _fail(problems, "join stretches: a run with a window missing was joined, not refused")
        except ValueError as exc:
            if "without a gap" not in str(exc):
                _fail(problems, f"join stretches: a missing window was refused without naming the gap: {str(exc)[:120]}")


def check_continue(problems, song_source):
    """A run that continues another run's last window (`continue_from`): the arithmetic of the stretches.

    Held here, on the planner and the store: a continued run writes each window's length less the
    context, its first included; two stretches laid as the rule says (the second's track starting
    `context` frames before the first one's end) write exactly the frames of one run over both,
    with no frame twice and none missing, for a spread of stretch lengths; a run without the input
    writes what it wrote before; and the first window takes its context from the stored window named.
    NOT held, and it cannot be without a render: that the seam is invisible.
    """
    import os
    import tempfile
    import loop_plan as lp
    import loop_resume as lr
    ctx = 39
    if lp.frames_written([345, 345, 141], ctx, 753, ctx) != [306, 306, 102] or lp.frames_written([141], ctx, 141, ctx) != [102] \
            or lp.frames_written([345, 345, 141], ctx, 720, ctx) != [306, 306, 69]:
        _fail(problems, "continue: a continued run does not write each window's length less the context")
    if lp.frames_written([345, 345, 141], ctx, 753, 0) != lp.frames_written([345, 345, 141], ctx, 753):
        _fail(problems, "continue: a run that continues nothing no longer writes what it wrote")
    for lengths, head in (([141], 141), ([345, 141], 345), ([141], -1)):
        try:
            lp.frames_written(lengths, ctx, lp.frames_covered(lengths, ctx), head)
        except ValueError:
            continue
        _fail(problems, f"continue: a head of {head} frames on windows {lengths} was not refused")
    # two stretches against one run: stretch one cold over `first` frames, stretch two continued, its
    # track starting at frame `covered_one - ctx` of the span and running to the span's end
    for window in (141, 345):
        for first in range(window, 1200, 37):
            one = [n for _e, each in lp.plan_windows(first, window, ctx, ()) for n in each]
            covered_one = lp.frames_covered(one, ctx)
            wrote_one = lp.frames_written(one, ctx, covered_one)
            for more in range(window, 1200, 53):
                two = [n for _e, each in lp.plan_windows(more, window, ctx, ()) for n in each]
                covered_two = lp.frames_covered(two, ctx)          # from the second track's own frame zero, context included
                wrote_two = lp.frames_written(two, ctx, covered_two, ctx)
                starts_at = covered_one - ctx                       # where the second run's loader starts, in span frames
                first_new = starts_at + ctx                         # the first frame it writes
                if first_new != sum(wrote_one) or sum(wrote_two) != covered_two - ctx or min(wrote_two) < 1 \
                        or wrote_two[1:] != lp.frames_written(two, ctx, covered_two)[1:]:
                    _fail(problems, f"continue: stretches {one} then {two}: the second writes from span frame {first_new}, "
                                    f"the first ends at {sum(wrote_one)}; it writes {sum(wrote_two)} of {covered_two - ctx}")
                    break
    # the node's side, from its source (the stored window it continues is read by `check_resume`'s cases)
    for text, what in (("prev = loop_resume.load_window_latent(continue_from) if continue_from else None", "seed its first window from the continued latent"),
                       ("heard = waveform[..., int(round(head / FPS * rate)):]", "start its file's audio on its first new frame"),
                       ("head = context_frames if continue_from else 0", "count the context as not written")):
        if text not in song_source:
            _fail(problems, f"continue: the song node does not {what}")


def check_song_plan(problems):
    import math
    import loop_plan as lp

    grid, half, ctx = lp.GRID, lp.GRID // 2, 39
    entries = lp.parse_timeline(FLICKER_TIMELINE)
    plan = lp.plan_windows(FLICKER_FRAMES, 345, ctx, entries)
    if [i for i, _l in plan] != list(range(len(entries))):
        _fail(problems, f"plan: the example song kept entries {[i for i, _l in plan]}")
    covered = 0
    for i, lengths in plan:
        want = int(round(entries[i][0] * lp.FPS))
        if i and abs(covered - want) > half:
            _fail(problems, f"plan: {entries[i][1]} at {entries[i][0]}s starts at frame {covered}, not within "
                            f"{half} of {want}")
        if any(n not in lp.CHAIN_LENGTHS for n in lengths) or lengths != sorted(lengths, reverse=True):
            _fail(problems, f"plan: {entries[i][1]} got windows {lengths}")
        covered += sum(n - (0 if covered == 0 and j == 0 else ctx) for j, n in enumerate(lengths))
    if not FLICKER_FRAMES <= covered < FLICKER_FRAMES + grid:
        _fail(problems, f"plan: the example song's windows cover {covered} frames of {FLICKER_FRAMES}")

    # every interior part length lands within half a step; a planner of full
    # windows only is the control that shows the bound can fail
    def adds(lengths):
        return sum(n - ctx for n in lengths)

    worst = max(abs(adds(lp.segment_lengths(t, False, False, 345, ctx)) - t) for t in range(102, 2500))
    naive = max(abs(adds([345] * max(1, round(t / (345 - ctx)))) - t) for t in range(102, 2500))
    if worst > half:
        _fail(problems, f"plan: an interior part landed {worst} frames off its target")
    if naive <= half:
        _fail(problems, "plan: the full-windows control also landed within half a step; the sweep proves nothing")

    whole = lp.plan_windows(3000, 345, ctx)
    whole_frames = sum(n - (0 if j == 0 else ctx) for j, n in enumerate(whole[0][1]))
    if len(whole) != 1 or whole[0][1][0] != 345 or not 3000 <= whole_frames < 3000 + grid:
        _fail(problems, f"plan: with no timeline a 3000-frame track got {whole}")
    short = lp.plan_windows(math.ceil(30 * lp.FPS), 345, ctx, entries)
    if [i for i, _l in short] != [0, 1]:
        _fail(problems, f"plan: a 30-second extent kept entries {[i for i, _l in short]}, not the first two")
    _refused(problems, "plan: an entry too short for one window",
             lambda: lp.plan_windows(2000, 345, ctx, lp.parse_timeline("00:00 a\n00:10 b\n00:12 c")), "'b'")
    for label, text in (("a timeline not starting at 00:00", "00:05 a"),
                        ("a timeline going backwards", "00:00 a\n00:00 b"),
                        ("a timeline line with no label", "00:00"),
                        ("a timeline line that is not mm:ss", "0:5 a")):
        _refused(problems, f"timeline: {label}", lambda text=text: lp.parse_timeline(text))

    if lp.parse_prompt_blocks("plain prompt") != {None: "plain prompt"}:
        _fail(problems, "blocks: a prompt with no --- line was not one block")
    two = lp.parse_prompt_blocks("--- verse\nV\n--- chorus\nC")
    if two != {"verse": "V", "chorus": "C"}:
        _fail(problems, f"blocks: labelled blocks read as {two}")
    for label, text in (("text before the first label", "intro\n--- verse\nV"),
                        ("a --- line with no label", "---\nV"),
                        ("two blocks with one label", "--- a\nX\n--- a\nY"),
                        ("an empty block", "--- a\n\n--- b\nY")):
        _refused(problems, f"blocks: {label}", lambda text=text: lp.parse_prompt_blocks(text))
    labelled = lp.parse_prompt_blocks("--- intro\nI\n--- verse\nV")
    _refused(problems, "blocks: labelled blocks with no timeline", lambda: lp.texts_for_entries(labelled, []))
    _refused(problems, "blocks: a timeline label with no block",
             lambda: lp.texts_for_entries(labelled, lp.parse_timeline("00:00 intro\n00:10 chorus")), "chorus")
    _refused(problems, "blocks: a block matching no timeline label",
             lambda: lp.texts_for_entries(labelled, lp.parse_timeline("00:00 intro")), "verse")

    song = lp.plan_song(FLICKER_FRAMES, 345, ctx, "one prompt", FLICKER_TIMELINE)
    if len(song.uses) != len(entries):
        _fail(problems, f"lists: {len(song.uses)} uses for {len(entries)} timeline entries")
    windows = lp.place_windows(song, [f"text {i}" for i in range(len(entries))], ctx)
    for w in windows:
        if w.text != f"text {w.entry}":
            _fail(problems, f"lists: window {w.number} of entry {w.entry} took {w.text!r}")
    for a, b in zip(windows, windows[1:]):
        if abs(b.start - (a.start + (a.frames - ctx) / lp.FPS)) > 1e-9 or \
                b.first_frame != a.first_frame + a.frames - (ctx if a.number > 1 else 0):
            _fail(problems, f"lists: window {b.number} starts at {b.start}s, frame {b.first_frame}")
            break
    plain = lp.plan_song(3000, 345, ctx, "one prompt", "")
    if len(plain.uses) != len(plain.segments[0][1]):
        _fail(problems, f"lists: with no timeline {len(plain.uses)} uses for {len(plain.segments[0][1])} windows")
    # Mid-shot times, the only kind the house writes since 2026-09-18 (a shot
    # header carries none). This is the one live control that the guard fires.
    late_text = "[Shot 1] A wide shot of the room. At 00:13.000, she turns and walks out."
    cut = lp.plan_song(600, 345, ctx, late_text, "")
    _refused(problems, "cuts: a mid-shot time past a window's end",
             lambda: lp.place_windows(cut, [late_text] * len(cut.uses), ctx), "window 2")
    early_text = "[Shot 1] A wide shot of the room. At 00:04.500, she turns and walks out."
    early = lp.plan_song(600, 345, ctx, early_text, "")
    lp.place_windows(early, [early_text] * len(early.uses), ctx)
    headers_only = "[Shot 1] A wide shot of the room. [Shot 2] The shot cuts to her hands."
    lp.place_windows(lp.plan_song(600, 345, ctx, headers_only, ""), [headers_only] * len(cut.uses), ctx)

    # What a source loader has to hold: `frames_read` against the furthest frame any placed
    # window slices from the track (the first frame and count the song node hands
    # `video_mask.window`), with and without a timeline, on whole and ragged lengths.
    for total, timeline in ((math.ceil(30 * lp.FPS), ""), (345, ""), (346, ""), (3000, ""), (141, ""),
                            (FLICKER_FRAMES, FLICKER_TIMELINE)):
        placed = lp.plan_song(total, 345, ctx, "one prompt", timeline)
        furthest = max(int(round(w.start * lp.FPS)) + w.frames
                       for w in lp.place_windows(placed, ["one prompt"] * len(placed.uses), ctx))
        reads = lp.frames_read(total, 345, ctx, timeline)
        if reads != furthest or reads < total:
            _fail(problems, f"frames read: a {total}-frame track{' with a timeline' if timeline else ''} is said to "
                            f"read {reads} frames and its windows slice up to frame {furthest}")
    if lp.frames_covered([345], ctx) != 345 or lp.frames_covered([], ctx) != 0:
        _fail(problems, "frames read: one window does not cover its own length, or none covers something")

    # The line a run prints when it cannot cover what it was asked for (`extent_shortfall`).
    quiet = (lp.extent_shortfall(31.375, 30.0, 720, 753, 753),         # the shipped masked graph: nothing to say
             lp.extent_shortfall(200.0, None, 4800),                   # the whole track, no source
             lp.extent_shortfall(30.0 - 0.5 / lp.FPS, 30.0, 720, 720), # under a frame short
             lp.extent_shortfall(10.0, 10.0, 240, 240, 243))           # a few frames short of the plan, not of the track
    if any(quiet):
        _fail(problems, f"shortfall: a run that covers what it was asked for printed {[q for q in quiet if q]}")
    raised = lp.extent_shortfall(31.375, 40.0, 753, 753, lp.frames_read(math.ceil(40 * lp.FPS), 345, ctx))
    if not raised or "753 frames" not in raised or "frame_load_cap" not in raised \
            or str(lp.frames_read(math.ceil(40 * lp.FPS), 345, ctx)) not in raised or "40s" not in raised:
        _fail(problems, f"shortfall: an extent raised past the loader's cap printed {raised!r}")
    plain = lp.extent_shortfall(14.4, 30.0, 346)
    if not plain or "frame_load_cap" in plain or "14.40s" not in plain:
        _fail(problems, f"shortfall: a short track with no source printed {plain!r}")
    early = lp.extent_shortfall(30.0, 30.0, 720, 600, 753)
    if not early or "600 frames" not in early or "last 120" not in early or "shorter than asked" in early:
        _fail(problems, f"shortfall: a source that ends before its track printed {early!r}")
    # What is written of the frames a plan covers: all of them, or as far as the track runs.
    for covered, track, want in ((753, 44.375, 753), (753, 31.375, 753), (753, 753 / lp.FPS - 1e-4, 753),
                                 (753, 30.0, 720), (753, 30.01, 721), (345, 14.0, 336), (345, 14.375, 345)):
        if lp.frames_kept(covered, track) != want:
            _fail(problems, f"frames kept: {covered} frames under a {track}s track keeps "
                            f"{lp.frames_kept(covered, track)}, not {want}")
    # A track cut to a load's length ends a fraction of a sample off a whole frame, and that fraction is not a
    # frame (`track_frames`, 2026-10-10: a 33-frame load whose track came one sample long wrote 34). The loads
    # are the ones a job cut that day, at the rate their track came in; a real part of a frame still counts.
    import math
    for frames, rate in ((33, 44100), (41, 44100), (22, 44100), (33, 48000), (47, 32000), (113, 22050)):
        exact = frames / lp.FPS * rate
        for samples in (math.floor(exact), math.ceil(exact), math.ceil(exact) + 1):
            if lp.track_frames(samples / rate) != frames or lp.frames_kept(141, samples / rate) != frames:
                _fail(problems, f"track frames: a {frames}-frame load whose track is {samples} samples at {rate} Hz counts as "
                                f"{lp.track_frames(samples / rate)} frames and keeps {lp.frames_kept(141, samples / rate)}")
    for seconds, want in ((33 / lp.FPS, 33), (33 / lp.FPS + 0.002, 34), (34 / lp.FPS, 34), (33.5 / lp.FPS, 34),
                          (0.0, 0), (0.0004, 1), (1 / lp.FPS, 1)):
        if lp.track_frames(seconds) != want:
            _fail(problems, f"track frames: a track of {seconds!r}s runs {lp.track_frames(seconds)} frames, not {want}")
    # What each window's file holds (`frames_written`), and that the cut the song node makes on a
    # short track's last window, AFTER its head trim, never reaches what that window writes: a
    # planner property, held for every window length, three contexts and a spread of track lengths
    # with and without a timeline. (mrop's review of the cut, 2026-10-06.)
    for window in lp.CHAIN_LENGTHS:
        for context in (39, 90, 141):
            if context >= window:
                continue
            for spread in ("", "00:00 a\n00:20 b", "00:00 a\n00:12 b\n00:40 c"):
                marks = lp.parse_timeline(spread)
                for track in range(1, 2200):
                    try:
                        parts = lp.plan_windows(track, window, context, marks)
                    except ValueError:
                        continue            # an entry too short for one window: refused, not planned
                    lengths = [n for _entry, each in parts for n in each]
                    covered = lp.frames_covered(lengths, context)
                    kept = lp.frames_kept(covered, track / lp.FPS)      # the most a run can cut: its track is never shorter
                    try:
                        writes = lp.frames_written(lengths, context, kept)
                    except ValueError as exc:
                        _fail(problems, f"frames written: a {track}-frame track at windows of {window}: {exc}")
                        break
                    whole = [n - (context if i else 0) for i, n in enumerate(lengths)]
                    if sum(writes) != kept or min(writes) < 1 or writes[:-1] != whole[:-1] or writes[-1] > whole[-1]:
                        _fail(problems, f"frames written: a {track}-frame track at windows of {window} with {context} of "
                                        f"context plans {lengths} and writes {writes}, which is not {kept} frames cut "
                                        "from the last window's tail alone")
                        break
    if lp.frames_written([345, 345, 141], ctx, 753) != [345, 306, 102] or lp.frames_written([345, 345, 141], ctx, 720) != [345, 306, 69] \
            or lp.frames_written([], ctx, 0) != []:
        _fail(problems, "frames written: the shipped thirty seconds does not write what its windows hold")
    for lengths, kept in (([345, 141], 345 + 141 - ctx - (141 - ctx)), ([345], 0), ([345], 346)):
        try:
            lp.frames_written(lengths, ctx, kept)
        except ValueError:
            continue
        _fail(problems, f"frames written: windows {lengths} keeping {kept} frames was not refused")
    song_source = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "join_and_mux(files, heard, rate, out_path, work_dir, stem, kept_frames - head)" not in song_source \
            or "images = images[:writes[i]]" not in song_source \
            or "writes = loop_plan.frames_written([w.frames for w in windows], context_frames, kept_frames, head)" not in song_source:
        _fail(problems, "frames kept: the song node no longer cuts the last window at the frames and joins that many")
    check_continue(problems, song_source)
    if "trim, next_start, written))" not in song_source:
        _fail(problems, "frames written: the song node no longer stores a window's frame count with its latent")
    if "loop_plan.extent_shortfall(" not in song_source or "logger.warning(\"[h3] MiniMaxH3AudioFreezeSong: %s\", shortfall)" not in song_source:
        _fail(problems, "shortfall: the song node no longer prints and logs the line")

    source = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    for p in lazy_problems(source):
        _fail(problems, f"preview: {p}")
    anchor = 'io.Model.Input("model", lazy=True)'
    if source.count(anchor) != 1:
        _fail(problems, f"preview: the control lost its anchor {anchor!r}")
    elif not lazy_problems(source.replace(anchor, 'io.Model.Input("model")')):
        _fail(problems, "preview: a song node whose model input is not lazy still passed")
    # a preview asks for the source (its plan, 2026-10-10) and the reference chain (what each reference will cost,
    # 2026-10-12): never for a model or a model's loader
    if 'PREVIEW_ASKS = ("source", "references")' not in source \
            or "return [name for name in PREVIEW_ASKS if name in kwargs and kwargs[name] is None]" not in source:
        _fail(problems, "preview: what a preview asks for is no longer the source and the reference chain; a model behind it would load in a preview")


def check_step_grid(problems):
    import importlib
    import random

    import loop_plan as lp
    vm = importlib.import_module("_h3pack.video_mask")

    # A WINDOW OFF THE CHAINED LENGTHS (2026-10-13): taken for a track it holds whole, as ONE window, on the video grid and no
    # longer than the longest chained length; refused for anything else, and the chained lengths and their plans are as they were
    for total, window in ((24, 39), (39, 39), (12, 22), (100, 107), (150, 158), (300, 311)):
        if lp.plan_windows(total, window, 90) != [(0, [window])] or not lp.stands_alone(window, total):
            _fail(problems, f"window: a {total}-frame track in a {window}-frame window is not planned as that one window")
    for total, window, why in ((40, 39, "the track is longer than the window"), (24, 40, "off the video grid"), (24, 38, "off the video grid"),
                               (400, 362, "past the longest chained length"), (24, 5, "shorter than the track")):
        if not _raises(lp.plan_windows, total, window, 90):
            _fail(problems, f"window: a {total}-frame track was planned in a {window}-frame window ({why})")
    if not _raises(lp.check_window_settings, 39, 90) or not _raises(lp.check_window_settings, 107, 39):
        _fail(problems, "window: check_window_settings took a length off the chained list with no track to hold it to")
    if not _raises(lp.plan_windows, 24, 39, 90, ((0.0, "a"), (0.5, "b"))):
        _fail(problems, "window: a window that stands alone was planned over two timeline entries")
    # the chained lengths are the seam rule's under the ceiling (h3_rules.chain_lengths), never a typed list (2026-10-15)
    import h3_rules
    on_both = [n for n in range(1, h3_rules.MAX_LENGTH + 1) if n % 17 == 5 and n % 3 == 0 and n >= h3_rules.SHORTEST_CHAINED]
    if list(lp.CHAIN_LENGTHS) != on_both or max(lp.CHAIN_LENGTHS) > h3_rules.MAX_LENGTH:
        _fail(problems, f"window: the chained lengths are not every length on both clocks up to the ceiling: {lp.CHAIN_LENGTHS}")
    top = max(lp.CHAIN_LENGTHS)
    for total, window, want in ((24, 141, [141]), (150, top, [192]), (400, top, None), (top, top, [top])):
        got = [n for _e, each in lp.plan_windows(total, window, 90) for n in each]
        if (want is not None and got != want) or any(n not in lp.CHAIN_LENGTHS for n in got):
            _fail(problems, f"window: a chained length's plan moved: {total} frames at {window} gives {got}")
    if not _raises(lp.check_window_settings, 141, 141) or _raises(lp.check_window_settings, 141, 90):
        _fail(problems, "window: a chained window's context rule moved")

    # one grid a load: what any window adds is whole cycles, and the step edges are the window node's own
    contexts = [c for c in range(1, max(lp.CHAIN_LENGTHS)) if not _raises(lp.check_window_settings, max(lp.CHAIN_LENGTHS), c)]
    if not contexts:
        _fail(problems, "step grid: no context passes check_window_settings; the sweep below checked nothing")
    for n in lp.CHAIN_LENGTHS:
        for c in contexts:
            if c < n and (n - c) % lp.STEP_CYCLE:
                _fail(problems, f"step grid: a {n}-frame window with {c} of context adds {n - c} frames, not a whole "
                                f"number of {lp.STEP_CYCLE}-frame cycles; the next window's steps fall elsewhere")
    edges = [af.pixel_frames(t) for t in range(4 * len(af.FRAME_PER_TOKEN) + 1)]
    for a, b in zip(edges, edges[1:]):
        if any(lp.step_span(f) != (a, b - a) for f in range(a, b)):
            _fail(problems, f"step grid: step_span disagrees with pixel_frames on the step over frames {a} to {b - 1}")

    # the measured case: cuts and the shots with the subject, from the render's own shot table, in source frames
    first, frames = 14, 600
    cuts = [first + c for c in (154, 172, 264, 280, 401, 424, 470, 502, 590)]
    present = [(first + a, first + b) for a, b in ((0, 153), (264, 279), (401, 423), (470, 501))]
    found = [294, 414, 438, 482, 483]          # measured: the delivery's check, 2026-10-10
    split = lp.split_steps(cuts, first, present, frames)
    named = sorted(f for s in split for f in s["across"])
    if named != found or any(s["shared"] for s in split):
        _fail(problems, f"step grid: the measured render's repainted frames are {found}; split_steps named {named}")
    blind = sorted(f for s in lp.split_steps(cuts, 0, present, frames) for f in s["across"])
    if blind == found:
        _fail(problems, "step grid control: counted from frame zero the same cuts still name the measured frames, so "
                        "the case does not show that the load's first frame matters")

    # the node's own gate as the second implementation, window by window
    rng = random.Random(20261010)
    for _trial in range(200):
        first = rng.randrange(0, 60)
        length = rng.choice(lp.CHAIN_LENGTHS)
        start = first + lp.STEP_CYCLE * rng.randrange(0, 3)      # a window of the load: whole cycles in
        bounds = sorted(rng.sample(range(start + 1, start + length), rng.randrange(1, 9)))
        shots = list(zip([start] + bounds, [b - 1 for b in bounds] + [start + length - 1]))
        on = [s for s in shots if rng.random() < 0.5]
        mask = torch.zeros(length, 2, 2)
        for a, b in on:
            mask[a - start:b - start + 1] = 1.0
        latent_t = next(t for t in range(1, length + 1) if af.pixel_frames(t) == length)
        gate = vm.cut_gate(mask, latent_t, bounds, start)
        want = sorted(start + int(f) for f in (gate < 0.5).nonzero().flatten())
        split = lp.split_steps(bounds, first, on, start + length - first)
        got = sorted(f for s in split for f in s["across"] if f >= start)
        if got != want:
            _fail(problems, f"step grid: load from {first}, window from {start} of {length}, cuts {bounds}, subject on {on}: "
                            f"split_steps names {got}, the node's cut_gate leaves {want} unlaid")
            break
        if any(float(gate[f - start]) < 0.5 for s in split for f in s["shared"]):
            _fail(problems, "step grid: a frame split_steps calls shared (the subject on both sides) is one the gate left unlaid")
            break


def _raises(call, *args) -> bool:
    try:
        call(*args)
    except ValueError:
        return True
    return False


def main() -> int:
    problems: list[str] = []
    check_slice(problems)
    check_masks(problems)
    check_execute(problems)
    check_window_geometry(problems)
    check_resume(problems)
    check_song_plan(problems)
    check_step_grid(problems)
    check_join(problems)
    check_join_stretches(problems)
    check_writer_colour(problems)
    n, frozen = check_graphs(problems)
    print(f"  {n} api graphs walked, {frozen} carry {FREEZE}, none carry {STOCK_MASK}"
          if not any(STOCK_MASK in p for p in problems) else
          f"  {n} api graphs walked, {frozen} carry {FREEZE}")
    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for pr in problems:
            print(f"    - {pr}")
        return 1
    print("  ok    slice on the grid and exact; nested mask survives the sampler's "
          "reshape and a flat one is refused; every freeze graph is wired end to end; "
          "a stored latent round-trips and nothing is keyed or reused; the loop plan lines up with its timeline and its refusals and controls bite; "
          "a load's first frame names the cuts that split a latent step, as the node's gate does; "
          "the join returns every frame its windows hold; a written file is BT.709 and says so")
    return 0


if __name__ == "__main__":
    sys.exit(main())
