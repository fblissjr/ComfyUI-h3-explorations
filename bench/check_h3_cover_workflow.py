#!/usr/bin/env python3
"""Offline CPU verification gate for `bench/h3_cover_workflow.py`."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)
sys.path.insert(0, str(REPO))
import h3_prompt_contract as prompt_contract  # noqa: E402

sys.path = [p for p in sys.path if p != str(REPO)]

import h3_cover_workflow as cw  # noqa: E402


def _harmonic_tone(freq_hz: float, dur_s: float, sr: int, amp: float = 0.25) -> torch.Tensor:
    n = int(round(dur_s * sr))
    t = torch.arange(n, dtype=torch.float32) / float(sr)
    y = amp * (0.65 * torch.sin(2.0 * math.pi * freq_hz * t) + 0.25 * torch.sin(4.0 * math.pi * freq_hz * t) + 0.10 * torch.sin(6.0 * math.pi * freq_hz * t))
    return y.unsqueeze(0).repeat(2, 1)


def test_pick_best_ref_vocal_window() -> None:
    sr = 44100
    full = torch.zeros(2, 28 * sr, dtype=torch.float32)
    # 1.0..9.0 s: loud 220 Hz voice cut abruptly mid-note right at edge (high head/tail dBFS penalty)
    full[:, int(1.0 * sr) : int(9.0 * sr)] = _harmonic_tone(220.0, 8.0, sr, amp=0.45)
    # 10.75..17.75 s: clean phrase inside [10.5..18.5 s] bounded by quiet pauses on both ends (~196 Hz)
    full[:, int(10.85 * sr) : int(17.95 * sr)] = _harmonic_tone(196.0, 7.1, sr, amp=0.30)
    # 19.3..26.3 s: clean phrase inside [19.0..27.0 s] bounded by quiet pauses at higher pitch (~262 Hz)
    full[:, int(19.35 * sr) : int(26.35 * sr)] = _harmonic_tone(262.0, 7.0, sr, amp=0.30)

    best_low = cw.pick_best_ref_vocal_window(full, sr=sr, register_bias="low")
    assert best_low["wav_32k"].shape == (2, 256000), f"Unexpected wav shape: {best_low['wav_32k'].shape}"
    assert 9.75 <= best_low["start_s"] <= 11.25, f"Expected quiet-bounded low window near 10.5 s, got {best_low['start_s']}"
    assert best_low["head_dbfs"] < -35.0 and best_low["tail_dbfs"] < -35.0, best_low
    assert 190.0 <= best_low["f0_stats"]["median"] <= 202.0, best_low["f0_stats"]

    best_high = cw.pick_best_ref_vocal_window(full, sr=sr, register_bias="high")
    assert best_high["f0_stats"]["median"] > best_low["f0_stats"]["median"] + 40.0, (best_low, best_high)
    print(f"ok    pick_best_ref_vocal_window: low={best_low['start_s']}s ({best_low['f0_stats']['median']}Hz), high={best_high['start_s']}s ({best_high['f0_stats']['median']}Hz)")


def test_choose_cover_semitones() -> None:
    sr = 24000
    ref_stats = {"p10": 155.0, "p25": 175.0, "median": 233.1, "p75": 245.0, "p90": 280.0}
    tgt = _harmonic_tone(311.1, 8.0, sr, amp=0.25)
    info = cw.choose_cover_semitones(ref_stats, tgt)
    assert info["semitones"] == -5.0, info
    assert info["vocal_formant"] is False, info
    assert info["active_periodic_frac"] > 0.90, info

    # Ceiling guard test: very wide p75 vs high ref_p90 nudges down 1 st without violating p10
    wide_ref = {"p10": 140.0, "p25": 195.0, "median": 220.0, "p75": 240.0, "p90": 250.0}
    n = 8 * sr
    half = n // 2
    wide_tgt = torch.cat([_harmonic_tone(200.0, 4.0, sr)[0], _harmonic_tone(290.0, 4.0, sr)[0]])
    info_ceil = cw.choose_cover_semitones(wide_ref, wide_tgt)
    assert info_ceil["semitones"] <= 0.0, info_ceil

    # Low periodicity warning on unpitched noise
    torch.manual_seed(0)
    noise = 0.15 * torch.randn(2, 6 * sr)
    info_noise = cw.choose_cover_semitones(ref_stats, noise)
    assert info_noise["active_periodic_frac"] < 0.35 and info_noise["warnings"], info_noise
    assert cw.choose_cover_semitones(ref_stats, noise, explicit_semitones=-4.0)["semitones"] == -4.0
    print(f"ok    choose_cover_semitones: auto=-5.0 st (formant={info['vocal_formant']}), ceil={info_ceil['semitones']} st, noise warns={len(info_noise['warnings'])}")


def test_plan_breath_aligned_windows_and_auto_denoise() -> None:
    sr = 24000
    dur_s = 46.0
    y = torch.zeros(int(dur_s * sr), dtype=torch.float32)

    def _put_tone(t0: float, t1: float, hz: float, amp: float = 0.25) -> None:
        i0, i1 = int(t0 * sr), int(t1 * sr)
        y[i0:i1] = _harmonic_tone(hz, t1 - t0, sr, amp=amp)[0, : i1 - i0]

    # Target breath grid aligned around start_s = 2.5..4.0 s:
    _put_tone(4.2, 13.8, 175.0)     # Window 1 low verse; gap [13.8..17.0] silences w1->w2 seam (~14.8..16.25)
    _put_tone(17.2, 19.0, 175.0)    # Window 2 Shot 1 sung; Shot 2 [19.0..27.0] silent ("instrumental") + clean w2->w3 seam
    _put_tone(27.5, 35.5, 250.0)    # Window 3 high chorus; gap [35.5..42.0] silences w3 end (~36.0..37.5)

    plan_auto = cw.plan_breath_aligned_windows(y, n_windows=3, start_s=None)
    assert plan_auto["n_windows"] == 3, plan_auto
    assert plan_auto["seam_max_dbfs"] < -35.0, f"Expected quiet breath seams, got {plan_auto}"
    assert any(w["shot2_mode"] == "instrumental" for w in plan_auto["windows"]), plan_auto["windows"]

    d_scalar = cw.resolve_window_denoises(plan_auto["windows"], "0.29")
    assert d_scalar == [0.29, 0.29, 0.29], d_scalar
    d_auto = cw.resolve_window_denoises(plan_auto["windows"], "auto")
    assert d_auto[0] == 0.26 and d_auto[-1] == 0.29, (plan_auto["windows"], d_auto)
    print(f"ok    plan_breath_aligned_windows: start={plan_auto['start_s']}s, seam_max={plan_auto['seam_max_dbfs']}dBFS, auto_denoise={d_auto}")


def test_prompts_and_graph_contract() -> None:
    windows = [
        {"index": 1, "start_s": 0.0, "end_s": 12.25, "shot1_span_s": [0.0, 6.125], "shot2_span_s": [6.125, 12.25], "shot1_mode": "sung", "shot2_mode": "sung", "register_role": "low"},
        {"index": 2, "start_s": 10.625, "end_s": 22.875, "shot1_span_s": [10.625, 16.75], "shot2_span_s": [16.75, 22.875], "shot1_mode": "sung", "shot2_mode": "instrumental", "register_role": "low"},
        {"index": 3, "start_s": 21.25, "end_s": 33.5, "shot1_span_s": [21.25, 27.375], "shot2_span_s": [27.375, 33.5], "shot1_mode": "sung", "shot2_mode": "sung", "register_role": "high"},
    ]
    with tempfile.TemporaryDirectory() as td:
        lrc = Path(td) / "lyrics.lrc"
        lrc.write_text("[00:01.00] Line one verse\n[00:07.50] Line two verse\n[00:12.00] Verse bridge\n[00:23.50] Chorus hook one\n[00:29.00] Chorus hook two\n", encoding="utf-8")
        paired = cw.parse_lyrics_for_windows(lrc, windows)
    assert paired[1][1] is None, paired

    refs = prompt_contract.describe_references([
        prompt_contract.ImageRef("image_1"),
        prompt_contract.ImageRef("image_2"),
        prompt_contract.AudioRef("audio_1"),
    ])
    prompts = []
    for k, (w, (l1, l2)) in enumerate(zip(windows, paired), 1):
        p = cw.build_window_prompt_skeleton(window_idx=k, win_info=w, n_pictures=2, shot1_lyrics=l1, shot2_lyrics=l2)
        fails, warns = prompt_contract.validate_h3_prompt_mechanics(p, mode="ref2va", manifest=refs, length=294, enforce_min_words=True)
        assert not fails and not warns, f"w{k}: fails={fails}, warns={warns}"
        assert cw.find_unfilled_placeholders([p]) == []
        prompts.append(p)
    assert "<|lyrics_start|>" not in prompts[1].split("[Shot 2]")[1], "Instrumental shot should have no lyric tag"

    skel_only = cw.build_window_prompt_skeleton(window_idx=1, win_info=windows[0], n_pictures=1)
    assert len(cw.find_unfilled_placeholders([skel_only])) == 1

    manifest_split = {
        "job": "unit_test",
        "width": 1024,
        "height": 768,
        "window_frames": 294,
        "context_frames": 39,
        "stride_frames": 255,
        "total_frames": 804,
        "n_windows": 3,
        "ref_register": "split",
        "ref_images_rel": ["cover_jobs/unit/ref_pic_01.png", "cover_jobs/unit/ref_pic_02.png"],
        "ref_vocal_rel": "cover_jobs/unit/ref_vocal_192f.wav",
        "ref_vocal_low_rel": "cover_jobs/unit/ref_vocal_low_192f.wav",
        "ref_vocal_high_rel": "cover_jobs/unit/ref_vocal_high_192f.wav",
        "scaffold_rel": "cover_jobs/unit/scaffold_full_32k.wav",
        "windows": windows,
    }
    g3 = cw.build_cover_graph(manifest_split, prompts, window_denoises=[0.26, 0.26, 0.29], use_frozen_video_cache=True, aud_to_vid_gain=1.0)
    assert "122_fvc" in g3 and g3["122_fvc"]["class_type"] == "MiniMaxH3FrozenVideoCache"
    assert g3["160_w1"]["inputs"]["denoise"] == 0.26 and g3["160_w3"]["inputs"]["denoise"] == 0.29
    assert g3["5_w1"]["inputs"]["encoded_references"][0] == "132_low" and g3["5_w3"]["inputs"]["encoded_references"][0] == "132_high"
    assert "hyb_aud_w2" in g3 and g3["hyb_aud_w2"]["class_type"] == "AudioConcat"
    assert not any(n.get("class_type") == "MiniMaxH3GuideLatent" for n in g3.values())

    # aud_to_vid_gain < 1.0 automatically omits FrozenVideoCache and uses FreezeAudio for 1-window jobs
    m1 = dict(manifest_split, windows=windows[:1], n_windows=1, total_frames=294, ref_register="single")
    g1 = cw.build_cover_graph(m1, prompts[:1], window_denoises=[0.30], use_frozen_video_cache=True, aud_to_vid_gain=0.0)
    assert "122_fvc" not in g1 and "145_w1" in g1 and g1["49_w1"]["class_type"] == "MiniMaxH3FreezeAudio"
    print(f"ok    prompts_and_graph_contract: 3w+FVC+split ({len(g3)} nodes) & 1w+a2v=0 ({len(g1)} nodes) verified")


def main() -> int:
    test_pick_best_ref_vocal_window()
    test_choose_cover_semitones()
    test_plan_breath_aligned_windows_and_auto_denoise()
    test_prompts_and_graph_contract()
    print("\nall 4 case(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
