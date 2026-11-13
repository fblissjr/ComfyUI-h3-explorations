#!/usr/bin/env python3
"""Unit and graph checks for `MiniMaxH3SoundCues` and `MiniMaxH3AppendSpriteTrajectory`.

Claims, i.e. what breaks if a case is deleted:
  sound cues merge + execute  `MiniMaxH3SoundCues` parses frame and `s` beat-sheet
                              lines, snaps cues to `3f` integer 40 Hz Audio VAE
                              RoPE boundaries, merges adjacent cues whose buffers
                              overlap (`54: snap` + `57: slam` -> one 13-token
                              island at `f54`), mixes the `-28 dBFS` rumble floor,
                              and appends `audio_latent` keyframes to `positive`
  sound cues refusals         unknown cue presets, malformed lines, negative or
                              overflowing frame indices, 1-frame AV latents, and
                              pre-existing overlapping `minimax_keyframes` raise
                              `ValueError` naming the bad span
  sprite trajectory execute   `MiniMaxH3AppendSpriteTrajectory` extracts subject
                              alpha via deterministic border-seeded GrabCut or an
                              explicit `mask`, decouples horizontal travel `(x, y)`
                              from `face=right|left|auto` orientation polarity,
                              applies parabolic `hop` lift + shadow attenuation,
                              and appends `RuntimeVideoReference(allow_upscale=False)`
  static consumers + graph    `reference_order`, `wired_labels`, and
                              `preflight_graph` resolve `MiniMaxH3AppendSpriteTrajectory`
                              as `<Video N>` without upscaling `384x224`, and
                              `workflows/h3_image_ref_dpad_sound_pika_api.json`
                              wires `end_percent=1.0`, `video_end_percent=0.33`,
                              `audio_end_percent=0.60`, and `MiniMaxH3EditGuider`

No GPU, no model weights, no server.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Keep `REPO.parent` behind ComfyUI's root and do not insert `REPO` itself ahead
# of ComfyUI, or the pack's `nodes.py` shadows ComfyUI's `import nodes` inside
# `comfy_extras.nodes_minimax_h3` (same rule as `bench/check_schema_defaults.py`).
sys.path.insert(0, str(HERE))
from _lib import COMFY, bootstrap  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO.parent))
bootstrap(cpu=True)
if str(COMFY) != sys.path[0]:
    sys.path.insert(0, str(COMFY))

import comfy.nested_tensor  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_PKG = REPO.name
rc_mod = importlib.import_module(f"{_PKG}.reference_conditioning")
ro_mod = importlib.import_module(f"{_PKG}.reference_order")
sc_mod = importlib.import_module(f"{_PKG}.sound_cues")
sp_mod = importlib.import_module(f"{_PKG}.sprite_trajectory")
from check_ref_prompt_labels import wired_labels  # noqa: E402
from preflight_graph import _reference_media, reference_video_report  # noqa: E402


class _FakeAudioVAE:
    """Stub Audio VAE mapping `[B, N, 2]` (movedim from 32 kHz `[B, 2, N]`) -> `[B, 32, 2, ceil(N / 800)]`."""

    def encode(self, wav_bnc: torch.Tensor) -> torch.Tensor:
        assert wav_bnc.ndim == 3 and int(wav_bnc.shape[-1]) == 2, f"unexpected shape {tuple(wav_bnc.shape)}"
        n = int(wav_bnc.shape[1])
        rt = max(1, math.ceil(n / 800.0))
        return torch.ones(int(wav_bnc.shape[0]), 32, 2, rt, dtype=torch.float32)


def _av_latent_124f() -> dict:
    # 124 frames = 37 video latent steps (2 + 7 * 5 -> 5 + 7*17 = 124), 207 audio tokens.
    vid = torch.zeros(1, 24, 37, 48, 84, dtype=torch.float32)
    aud = torch.zeros(1, 32, 2, 207, dtype=torch.float32)
    return {"samples": comfy.nested_tensor.NestedTensor((vid, aud))}


def _av_latent_1f() -> dict:
    vid = torch.zeros(1, 24, 1, 48, 48, dtype=torch.float32)
    aud = torch.zeros(1, 32, 2, 2, dtype=torch.float32)
    return {"samples": comfy.nested_tensor.NestedTensor((vid, aud))}


def _synthetic_subject_image(h: int = 160, w: int = 220) -> torch.Tensor:
    """Asymmetric subject (bright head on right side + tall candle on top) on dark backdrop."""
    img = np.full((1, h, w, 3), 0.10, dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    shell = ((xx - 105) / 42.0) ** 2 + ((yy - 96) / 28.0) ** 2 <= 1.0
    img[0, shell] = (0.58, 0.56, 0.52)
    # Distinct bright orange-gold head protruding to the RIGHT (+x) so flip_x is measurable
    head = ((xx - 156) / 14.0) ** 2 + ((yy - 92) / 11.0) ** 2 <= 1.0
    img[0, head] = (0.95, 0.72, 0.20)
    # Upright white candle + bright flame above shell center
    img[0, 42:68, 101:109] = (0.96, 0.96, 0.94)
    img[0, 32:42, 102:108] = (1.00, 0.86, 0.22)
    return torch.from_numpy(img)


def test_sound_cues_merge_and_execute() -> None:
    pos = [[torch.zeros(1, 12, 64), {"pooled_output": None}]]
    lat = _av_latent_124f()
    vae = _FakeAudioVAE()

    # Default beat sheet has `54: snap` (7 tok: [90, 97)) + `57: slam` (8 tok: [95, 103)),
    # which MUST auto-merge into one 13-token island [90, 103) at frame 54!
    out = sc_mod.MiniMaxH3SoundCues.execute(pos, lat, vae)
    out_pos, out_lat, out_ed = out.args
    assert out_lat is lat and out_ed is None
    kfs = out_pos[0][1]["minimax_keyframes"]
    assert len(kfs) == 3, f"expected 3 merged islands, got {len(kfs)}"
    anchors = [kf["resolved_frame_index"] for kf in kfs]
    rts = [int(kf["audio_latent"].shape[-1]) for kf in kfs]
    assert anchors == [24, 54, 96], f"unexpected anchors {anchors}"
    assert rts == [8, 13, 8], f"expected [8, 13, 8] tokens after 54+57 merge, got {rts}"

    # Optional `edited_positive` attaches the exact same encoded islands to slot 2
    ed_in = [[torch.zeros(1, 14, 64), {"pooled_output": None}]]
    out2 = sc_mod.MiniMaxH3SoundCues.execute(pos, lat, vae, cues="54: hop", edited_positive=ed_in)
    out_pos2, _, out_ed2 = out2.args
    assert out_ed2 is not None
    assert len(out_ed2[0][1]["minimax_keyframes"]) == 1
    assert out_ed2[0][1]["minimax_keyframes"][0]["audio_latent"] is out_pos2[0][1]["minimax_keyframes"][0]["audio_latent"]

    # Seconds syntax + 3f snapping (`0.96s` -> f23 -> snaps to f24; `snap_to_3f=False` keeps f23)
    parsed_snap = sc_mod.parse_sound_cues("0.96s: hop\n2.25s: click @0.8", 124, snap_to_3f=True)
    assert [c.resolved_frame for c in parsed_snap] == [24, 54]
    parsed_nosnap = sc_mod.parse_sound_cues("0.96s: hop", 124, snap_to_3f=False)
    assert [c.resolved_frame for c in parsed_nosnap] == [23]

    # Ambient floor check: tail of island with floor=-28 dBFS is ~-28 dBFS, not -94 dBFS silence
    island = sc_mod.merge_sound_cue_islands(sc_mod.parse_sound_cues("54: hop", 124))[0]
    wav_floor = sc_mod.synthesize_island_waveform(island, floor_dbfs=-28.0, seed=7)
    tail_rms = float(torch.sqrt(torch.mean(wav_floor[0, 0, -1200:] ** 2)))
    tail_dbfs = 20.0 * math.log10(max(tail_rms, 1e-9))
    assert -34.0 <= tail_dbfs <= -22.0, f"expected ~-28 dBFS tail floor, got {tail_dbfs:.2f} dBFS"

    wav_no_floor = sc_mod.synthesize_island_waveform(island, floor_dbfs=-60.0, seed=7)
    tail_nf_rms = float(torch.sqrt(torch.mean(wav_no_floor[0, 0, -400:] ** 2)))
    assert tail_nf_rms < tail_rms * 0.15, "disabling floor (-60 dBFS) should drop tail energy sharply"

    # Refusals
    for bad_cues in ("", "   # comment only\n", "24: explosion", "200: hop", "1.5: hop"):
        try:
            sc_mod.MiniMaxH3SoundCues.execute(pos, lat, vae, cues=bad_cues)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError on bad cues {bad_cues!r}")

    try:
        sc_mod.MiniMaxH3SoundCues.execute(pos, _av_latent_1f(), vae)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on single-frame latent")


def test_sprite_trajectory_orientation_hop_and_mask() -> None:
    img = _synthetic_subject_image()
    img_rec = rc_mod.MiniMaxH3AppendRefImage.execute(
        img,
        {"size_policy": "max", "dit_short_edge": 2048, "allow_upscale": True},
        {"qwen_view": "separate", "qwen_short_edge": 512},
    ).args[0]

    # 1. Default backpedal (`face=right` across 48 -> 92 + `hop 44..54: lift=0.29`)
    out_back = sp_mod.MiniMaxH3AppendSpriteTrajectory.execute(
        img,
        length=124,
        references=img_rec,
    )
    records_back, frames_back = out_back.args
    assert len(records_back) == 2
    assert isinstance(records_back[1], rc_mod.RuntimeVideoReference)
    assert records_back[1].use_vae is True
    assert records_back[1].allow_upscale is False
    assert tuple(frames_back.shape) == (124, 224, 384, 3)

    # Same trajectory with `face=left` on waypoint 92 flips the asymmetric orange head horizontally
    kf_turn = (
        "0: x=0.20, y=0.70, face=right\n"
        "48: x=0.64, y=0.70, face=right\n"
        "92: x=0.26, y=0.70, face=left\n"
        "123: x=0.26, y=0.70, face=left\n"
        "hop 44..54: lift=0.29"
    )
    _, frames_turn = sp_mod.MiniMaxH3AppendSpriteTrajectory.execute(
        img,
        length=124,
        keyframes=kf_turn,
    ).args

    # At frame 75 (mid `-x` retreat), `frames_back[75]` and `frames_turn[75]` sit at the exact same cx,
    # while the bright orange head sits on `+x` of sprite center in `back` and `-x` in `turn`!
    f_back = frames_back[75].numpy()
    f_turn = frames_turn[75].numpy()
    orange_back_xs = np.where((f_back[:, :, 0] > 0.80) & (f_back[:, :, 2] < 0.30))[1]
    orange_turn_xs = np.where((f_turn[:, :, 0] > 0.80) & (f_turn[:, :, 2] < 0.30))[1]
    assert len(orange_back_xs) > 5 and len(orange_turn_xs) > 5
    assert float(orange_back_xs.mean()) > float(orange_turn_xs.mean()) + 15.0, (
        f"expected face=right head x ({orange_back_xs.mean():.1f}) > face=left head x ({orange_turn_xs.mean():.1f})"
    )

    # Hop apex at frame 49 lifts the orange head substantially higher (smaller y) than frame 44
    f44 = frames_back[44].numpy()
    f49 = frames_back[49].numpy()
    y44 = float(np.where((f44[:, :, 0] > 0.80) & (f44[:, :, 2] < 0.30))[0].mean())
    y49 = float(np.where((f49[:, :, 0] > 0.80) & (f49[:, :, 2] < 0.30))[0].mean())
    assert y49 < y44 - 35.0, f"expected hop apex y49 ({y49:.1f}) < liftoff y44 ({y44:.1f}) - 35 px"

    # Explicit mask path + zero-mask refusal
    explicit_mask = torch.zeros(1, 160, 220, dtype=torch.float32)
    explicit_mask[0, 50:125, 65:170] = 1.0
    _, frames_masked = sp_mod.MiniMaxH3AppendSpriteTrajectory.execute(
        img, length=22, keyframes="0: x=0.3, y=0.7\n21: x=0.7, y=0.7", mask=explicit_mask
    ).args
    assert tuple(frames_masked.shape) == (22, 224, 384, 3)

    try:
        sp_mod.MiniMaxH3AppendSpriteTrajectory.execute(
            img, length=22, mask=torch.zeros(1, 160, 220, dtype=torch.float32)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on all-zero mask")


def test_shipped_hybrid_workflow_and_static_walkers() -> None:
    wf_path = REPO / "workflows" / "h3_image_ref_dpad_sound_pika_api.json"
    assert wf_path.is_file(), f"missing {wf_path}"
    g = json.loads(wf_path.read_text())

    assert g["50"]["class_type"] == "MiniMaxH3AppendRefImage"
    assert g["51"]["class_type"] == "MiniMaxH3AppendSpriteTrajectory"
    assert g["51"]["inputs"]["references"] == ["50", 0]
    assert g["51"]["inputs"]["length"] == ["27", 2]

    assert g["70"]["class_type"] == "MiniMaxH3SoundCues"
    assert g["70"]["inputs"]["cues"] == "54: hop"

    assert g["135"]["class_type"] == "MiniMaxH3ReleaseReferences"
    assert abs(float(g["135"]["inputs"]["video_end_percent"]) - 0.33) < 1e-6
    assert abs(float(g["135"]["inputs"]["audio_end_percent"]) - 0.60) < 1e-6
    assert abs(float(g["135"]["inputs"]["end_percent"]) - 1.0) < 1e-6

    assert g["9"]["class_type"] == "MiniMaxH3EditGuider"
    assert g["9"]["inputs"]["streams"] == "video_only"
    assert g["9"]["inputs"]["edited"] == ["70", 0]
    assert g["9"]["inputs"]["original"] == ["26", 0]

    # Static reference chain walkers see `<Picture 1>` + `<Video 1>` cleanly
    cond_ins = g["5"]["inputs"]
    labels = wired_labels(cond_ins, g)
    assert labels == ["<Picture 1>", "<Video 1>"], f"unexpected labels {labels}"

    media, _policies, typed = _reference_media(cond_ins, g)
    assert typed is True
    assert "ref_images.ref_image_0" in media
    assert "ref_videos.ref_video_0" in media
    rep = "\n".join(reference_video_report(media, g, 124))
    assert "384x224" in rep, f"expected 384x224 geometry in preflight video report:\n{rep}"

    # Malformed sprite trajectory link raises ChainError
    broken = json.loads(wf_path.read_text())
    broken["51"]["inputs"]["image"] = ["9999", 0]
    try:
        ro_mod.resolve_chain(broken, "51")
    except ro_mod.ChainError:
        pass
    else:
        raise AssertionError("expected ChainError on dangling sprite image link")


def main() -> int:
    test_sound_cues_merge_and_execute()
    print("  ok    MiniMaxH3SoundCues beat-sheet, 54+57 compound merge, -28 dBFS floor, and refusals")
    test_sprite_trajectory_orientation_hop_and_mask()
    print("  ok    MiniMaxH3AppendSpriteTrajectory GrabCut/mask, face=right vs left, and parabolic hop")
    test_shipped_hybrid_workflow_and_static_walkers()
    print("  ok    h3_image_ref_dpad_sound_pika_api.json wiring + static reference walkers")
    print("\nall ok -- sound cues and sprite trajectory checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
