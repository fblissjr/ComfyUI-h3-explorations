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
    if af.window_geometry(102, 0)["stride_frames"] != 345:
        _fail(problems, "context 0 did not give a full-window stride")
    # window 1 copies the previous tail into the head and freezes it
    video = torch.randn(1, 24, 102, 48, 84)
    audio = torch.randn(1, 32, 2, 575)
    fresh = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros_like(video), torch.zeros_like(audio)))}
    prev = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    song = {"waveform": torch.randn(1, 2, 44100 * 40), "sample_rate": 44100}
    out = af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 1, 39, previous=prev)
    args = getattr(out, "args", out)
    lat, _clip, span, trim, start, _rep = args
    v2, _ = lat["samples"].unbind()
    if not torch.equal(v2[:, :, :12], video[:, :, 90:]):
        _fail(problems, "window 1 did not copy the previous window's last 12 latent steps into its head")
    if v2[:, :, 12:].abs().max() != 0.0:
        _fail(problems, "window 1 wrote outside the context")
    vm, am = lat["noise_mask"].unbind()
    if vm[:, :, :12].max() != 0.0 or vm[:, :, 12:].min() != 1.0 or am.max() != 0.0:
        _fail(problems, "window 1's masks are wrong")
    if trim != 39 or abs(start - 306 / 24) > 1e-9:
        _fail(problems, f"window 1 trim/start {trim} {start}")
    if span["waveform"].shape[-1] != (510 + 575) * 800:
        _fail(problems, f"span audio is {span['waveform'].shape[-1]} samples, expected {(510 + 575) * 800}")
    try:
        af.MiniMaxH3FreezeAudioWindow.execute(fresh, FakeAudioVAE(), song, 1, 39)
        _fail(problems, "window 1 without a previous latent was accepted")
    except ValueError:
        pass


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
            if g[fid]["inputs"].get("latent") != ["26", 1]:
                _fail(problems, f"{p.name}: node {fid} takes its latent from {g[fid]['inputs'].get('latent')}, not the preflight")
        samplers = [nid for nid, ct in classes.items() if ct == "SamplerCustomAdvanced"]
        for sid in samplers:
            src = g[sid]["inputs"].get("latent_image")
            if not (isinstance(src, list) and src[0] in ids and src[1] == 0):
                _fail(problems, f"{p.name}: sampler {sid} does not take a frozen latent ({src})")
        muxers = [nid for nid, ct in classes.items() if ct == "VHS_VideoCombine"]
        for mid in muxers:
            src = g[mid]["inputs"].get("audio")
            ok = isinstance(src, list) and src[0] in ids and (
                (classes[src[0]] == FREEZE and src[1] == 1) or (classes[src[0]] == WINDOW and src[1] == 2))
            if not ok:
                _fail(problems, f"{p.name}: muxer {mid} does not take a freeze node's track ({src})")
        if "VAEDecodeAudio" in classes.values():
            _fail(problems, f"{p.name}: an audio decoder is still present on a freeze graph")
        windows = [nid for nid in ids if classes[nid] == WINDOW]
        for wid in windows:
            w = g[wid]["inputs"]
            if w.get("window_index", 0) > 0 and not w.get("previous"):
                _fail(problems, f"{p.name}: window node {wid} is not window 0 and has no previous latent")
    return len(paths), frozen


def main() -> int:
    problems: list[str] = []
    check_slice(problems)
    check_masks(problems)
    check_execute(problems)
    check_window_geometry(problems)
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
          "reshape and a flat one is refused; every freeze graph is wired end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
