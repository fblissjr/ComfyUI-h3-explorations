#!/usr/bin/env python3
"""The audio-freeze node's contract, and the two ways a graph can silently unfreeze.

`docs/h3_audio_freeze.md` is the lane; `audio_freeze.py` is the node. Three
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
4. **Resume reuses only what its key covers.** `loop_resume.py`: the graph
   signature moves with a checkpoint, a sampler, the crf or the canvas and
   not with the song node's per-window inputs; renumbering nodes moves
   nothing; a missing prompt or a dangling link gives no key at all; a
   window's key depends on the previous window's; and a stored window reads
   back with its key, trim, next start and latent, and not at all once its
   video is gone.
5. **The plan lines windows up with the timeline.** `loop_plan.py`: on the
   example song's timeline every entry after the first starts within half a
   `GRID` step of its time, and so does every interior part length in a sweep,
   against a control planner of full windows only, which misses; windows run
   longest first; entries past the covered length drop; a too-short entry, a
   bad timeline line, a label with no block, a block with no label and a cut
   past its window's end are refused; lists get one use per entry, or one per
   window with no timeline; and every input a preview skips is declared lazy
   on the song node, with a copy missing one refused.

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
import audio_freeze as af  # noqa: E402

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
    import copy
    import os
    import tempfile
    import loop_resume as lr

    base = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "74": {"class_type": "MiniMaxH3AudioFreezeSong",
               "inputs": {"model": ["1", 0], "sampler": ["7", 0], "prompt": "a dancer", "crf": 19,
                          "seed": 5, "filename_prefix": "Video/x", "extent": "whole", "width": 1344,
                          "reuse_windows": True}},
    }

    def sig(g):
        return lr.graph_signature(g, "74", skip=lr.SONG_PER_WINDOW)

    s0 = sig(base)
    if s0 is None:
        _fail(problems, "resume: a complete prompt gave no graph signature")
        return

    def setter(nid, name, value):
        return lambda g: g[nid]["inputs"].__setitem__(name, value)

    for label, edit, same in (
            ("a checkpoint change", setter("1", "unet_name", "b.safetensors"), False),
            ("a sampler change", setter("7", "sampler_name", "er_sde"), False),
            ("a crf change", setter("74", "crf", 23), False),
            ("a canvas change", setter("74", "width", 1152), False),
            ("a prompt edit", setter("74", "prompt", "a dancer, closer"), True),
            ("a seed change", setter("74", "seed", 6), True),
            ("a filename change", setter("74", "filename_prefix", "Video/y"), True),
            ("an extent change", setter("74", "extent", "first_seconds"), True),
            ("a timeline edit", setter("74", "timeline", "00:00 intro"), True),
            ("the preview switch", setter("74", "preview", True), True),
            ("the reuse switch", setter("74", "reuse_windows", False), True)):
        g = copy.deepcopy(base)
        edit(g)
        if (sig(g) == s0) != same:
            _fail(problems, f"resume: {label} {'moved' if same else 'did not move'} the graph signature")
    renumbered = {"9": copy.deepcopy(base["1"]), "7": copy.deepcopy(base["7"]), "74": copy.deepcopy(base["74"])}
    renumbered["74"]["inputs"]["model"] = ["9", 0]
    if sig(renumbered) != s0:
        _fail(problems, "resume: renumbering the graph moved its signature")
    dangling = copy.deepcopy(base)
    dangling["74"]["inputs"]["model"] = ["42", 0]
    if lr.graph_signature(None, "74") is not None or lr.graph_signature(base, "99") is not None \
            or sig(dangling) is not None:
        _fail(problems, "resume: a missing prompt, node or link still gave a key")

    root = lr.root_key(s0, lr.track_hash(torch.zeros(1, 2, 100), 44100))
    if lr.root_key(s0, lr.track_hash(torch.ones(1, 2, 100), 44100)) == root:
        _fail(problems, "resume: the root ignores the track's samples")
    k1 = lr.window_key(root, 1, "t", 141, 0.0, 5, None)
    if lr.window_key(root, 2, "t", 141, 4.25, 6, k1) == lr.window_key(root, 2, "t", 141, 4.25, 6, "other"):
        _fail(problems, "resume: a window's key ignores the previous window's")

    with tempfile.TemporaryDirectory() as d:
        video, audio = torch.randn(1, 24, 3, 4, 6), torch.randn(1, 32, 2, 5)
        samples = comfy.nested_tensor.NestedTensor((video, audio))
        video_path, _latent = lr.window_paths(d, "song", 1)
        if lr.read_window(d, "song", 1) is not None:
            _fail(problems, "resume: an empty store read as a window")
        open(video_path, "wb").close()
        lr.save_window(d, "song", 1, k1, samples, 39, 5.875)
        got = lr.read_window(d, "song", 1)
        if got is None or (got["key"], got["trim"], got["next_start"]) != (k1, 39, 5.875):
            _fail(problems, f"resume: a stored window read back as {got}")
        else:
            back = lr.load_window_latent(got["latent"])["samples"]
            if not getattr(back, "is_nested", False) or not all(
                    torch.equal(a, b) for a, b in zip(back.unbind(), (video, audio))):
                _fail(problems, "resume: a stored latent did not round-trip")
        os.remove(video_path)
        if lr.read_window(d, "song", 1) is not None:
            _fail(problems, "resume: a latent whose video is gone read as a finished window")


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
        if any(n not in lp.CHAIN_LENGTHS or n > 345 for n in lengths) or lengths != sorted(lengths, reverse=True):
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

    source = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    for p in lazy_problems(source):
        _fail(problems, f"preview: {p}")
    anchor = 'io.Model.Input("model", lazy=True)'
    if source.count(anchor) != 1:
        _fail(problems, f"preview: the control lost its anchor {anchor!r}")
    elif not lazy_problems(source.replace(anchor, 'io.Model.Input("model")')):
        _fail(problems, "preview: a song node whose model input is not lazy still passed")


def main() -> int:
    problems: list[str] = []
    check_slice(problems)
    check_masks(problems)
    check_execute(problems)
    check_window_geometry(problems)
    check_resume(problems)
    check_song_plan(problems)
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
          "resume keys; the loop plan lines up with its timeline and its refusals and controls bite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
