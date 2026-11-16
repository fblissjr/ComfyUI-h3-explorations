#!/usr/bin/env python3
"""Bucket 3 probe: Timeline-scoped Frame-0 `section_anchor` + `anchor_end_percent=0.33` release.

Validates offline via `--dry-run` (default) or executes in-process on CUDA via `--run`:
  4-Window 2-Section Timeline (447f = 18.625s, windows 192f/ctx 39f -> 4x141f, studio -> stage at 10.125s):
    1. `b3_4win_off`          -- `section_anchor="off"` (unanchored control: drift + authentic stage darks)
    2. `b3_4win_whole_track`  -- `section_anchor="whole_track", anchor_end_percent=0.33` (W1 studio held into stage)
    3. `b3_4win_per_section`  -- `section_anchor="per_section", anchor_end_percent=0.33` (W1->W2 & W3->W4, reset at stage)

  2-Window Release Duration Check (243f = 10.125s, windows 141f/ctx 39f, studio):
    4. `b3_2win_rel033`       -- `section_anchor="per_section", anchor_end_percent=0.33`
    5. `b3_2win_hold100`      -- `section_anchor="per_section", anchor_end_percent=1.00`

Runs in-process via `server_memory_mode()` so no `:8188` restart is needed. All media writes live under
`$H3_OUTPUT_DIR/Video/bucket3_probe_2026-11-16/` (zero media in `data/`).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import tempfile
import time
import types
import wave
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

DEFAULT_OUT_SUBDIR = "bucket3_probe_2026-11-16"
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_CONTEXT = 39
SONG_4WIN_FRAMES = 447  # 18.625s = 141 + 102 + 102 + 102
SONG_4WIN_MAX_WINDOW = 192
SONG_4WIN_TIMELINE = "00:00 studio\n00:10.125 stage"
SONG_2WIN_FRAMES = 243  # 10.125s = 141 + 102
SONG_2WIN_MAX_WINDOW = 141
DEFAULT_SEED = 730451892
CONTEXT_NOISE = 0.6

STAGE_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, low-key dark theater lighting, "
    "a medium shot frames a dancer in her twenties in a dark midnight-blue velvet stage space, a single cool "
    "cyan overhead spotlight isolating her face and shoulders while the surrounding black stage curtains and floor "
    "fall into deep shadow. She wears a grey vest with hair tied back, lips closed, producing no vocal sound, "
    "hitting sharp rhythmic isolations of her shoulders and chin cleanly on every beat as haze drifts through "
    "the narrow cyan beam. [Shot 2] The camera cuts to a closer waist-up angle from the dark wing side as she "
    "pivots on the beat, deep indigo shadows wrapping her silhouette and sharp cyan rim light tracing her hair "
    "and collarbone while she holds still on a final beat, breathing steadily with lips closed.\n\n"
    "overall_soundscape: Hollow leather footsteps on matte black stage marley, soft cloth brush, and short "
    "controlled breaths between rhythmic drum hits under hushed theater room tone.\n\n"
    "non_diegetic_music: N/A"
)


def four_win_specs() -> list[tuple[str, str, float]]:
    return [
        ("b3_4win_off", "off", 0.33),
        ("b3_4win_whole_track", "whole_track", 0.33),
        ("b3_4win_per_section", "per_section", 0.33),
    ]


def two_win_specs() -> list[tuple[str, str, float]]:
    return [
        ("b3_2win_rel033", "per_section", 0.33),
        ("b3_2win_hold100", "per_section", 1.00),
    ]


def _ensure_paths() -> None:
    if str(_REPO_ROOT / "workflows") not in sys.path:
        sys.path.append(str(_REPO_ROOT / "workflows"))
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))


def _import_pack_module(name: str):
    pkg_name = "_h3pack"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_REPO_ROOT)]
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.{name}")


def _prompts() -> tuple[str, str]:
    studio_prompt = (_REPO_ROOT / "prompt_bank" / "t2va_studio_dancer_close.txt").read_text(encoding="utf-8").strip()
    two_sec = f"--- studio\n{studio_prompt}\n\n--- stage\n{STAGE_PROMPT}"
    return studio_prompt, two_sec


def build_song_graph(
    pat,
    *,
    arm: str,
    prompt: str,
    timeline: str,
    window_frames: int,
    section_anchor: str,
    anchor_end_percent: float,
    audio_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    context_frames: int = DEFAULT_CONTEXT,
    context_noise: float = CONTEXT_NOISE,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    g = pat._make_base_graph(
        prompt=prompt,
        width=width,
        height=height,
        length=141,
        seed=seed,
        out_subdir=out_subdir,
        arm=arm,
    )
    sigmas_link = g["10"]["inputs"]["sigmas"]
    model_link = g["9"]["inputs"]["model"]
    sampler_link = g["10"]["inputs"]["sampler"]
    for nid in ("5", "6", "9", "10", "11", "12", "13", "26", "27"):
        g.pop(nid, None)
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
    g["74"] = {
        "class_type": "MiniMaxH3AudioFreezeSong",
        "inputs": {
            "model": model_link,
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "audio": ["48", 0],
            "sampler": sampler_link,
            "sigmas": sigmas_link,
            "prompt": prompt,
            "timeline": timeline,
            "preview": False,
            "width": int(width),
            "height": int(height),
            "window_frames": int(window_frames),
            "context_frames": int(context_frames),
            "extent": "whole",
            "seed": int(seed),
            "audio_mask": 0.0,
            "level": "clip_guard",
            "filename_prefix": f"Video/{out_subdir}/{arm}",
            "crf": 19,
            "save_metadata_png": True,
            "keep_windows": False,
            "context_noise": float(context_noise),
            "context_profile": "seam_taper",
            "section_anchor": str(section_anchor),
            "anchor_end_percent": float(anchor_end_percent),
        },
    }
    return g


def build_all_graphs(pat, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    studio_prompt, two_sec_prompt = _prompts()
    graphs: dict[str, dict[str, Any]] = {}
    if args.only in ("4win", "all"):
        for arm, mode, end_pct in four_win_specs():
            graphs[arm] = build_song_graph(
                pat,
                arm=arm,
                prompt=two_sec_prompt,
                timeline=SONG_4WIN_TIMELINE,
                window_frames=SONG_4WIN_MAX_WINDOW,
                section_anchor=mode,
                anchor_end_percent=end_pct,
                audio_name="h3_probe_20261116_b3_drum_18p625s.wav",
                width=args.width,
                height=args.height,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    if args.only in ("2win", "all"):
        for arm, mode, end_pct in two_win_specs():
            graphs[arm] = build_song_graph(
                pat,
                arm=arm,
                prompt=studio_prompt,
                timeline="",
                window_frames=SONG_2WIN_MAX_WINDOW,
                section_anchor=mode,
                anchor_end_percent=end_pct,
                audio_name="h3_probe_20261116_b3_drum_10p125s.wav",
                width=args.width,
                height=args.height,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    return graphs


def verify_dry_run(args: argparse.Namespace) -> int:
    from _lib import bootstrap

    bootstrap(cpu=True)
    import nodes as _comfy_nodes  # noqa: F401 -- lock sys.modules['nodes'] to ComfyUI/nodes.py
    _ensure_paths()
    import h3_rules
    import probe_anchor_transfer as pat

    lp = _import_pack_module("loop_plan")
    studio_prompt, two_sec_prompt = _prompts()

    plan_4 = lp.plan_song(SONG_4WIN_FRAMES, SONG_4WIN_MAX_WINDOW, DEFAULT_CONTEXT, two_sec_prompt, SONG_4WIN_TIMELINE)
    ws_4 = lp.place_windows(plan_4, ["studio", "stage"], DEFAULT_CONTEXT)
    got_4 = [(w.number, w.entry, w.frames, w.first_frame) for w in ws_4]
    want_4 = [(1, 0, 141, 0), (2, 0, 141, 141), (3, 1, 141, 243), (4, 1, 141, 345)]
    if got_4 != want_4:
        raise AssertionError(f"4-window timeline layout mismatch: got {got_4}, want {want_4}")

    plan_2 = lp.plan_song(SONG_2WIN_FRAMES, SONG_2WIN_MAX_WINDOW, DEFAULT_CONTEXT, studio_prompt, "")
    ws_2 = lp.place_windows(plan_2, [studio_prompt] * len(plan_2.uses), DEFAULT_CONTEXT)
    got_2 = [(w.number, w.entry, w.frames, w.first_frame) for w in ws_2]
    want_2 = [(1, None, 141, 0), (2, None, 141, 141)]
    if got_2 != want_2:
        raise AssertionError(f"2-window layout mismatch: got {got_2}, want {want_2}")

    graphs = build_all_graphs(pat, args)
    chains = set(h3_rules.chain_lengths())
    for arm, g in graphs.items():
        assert g["58"]["class_type"] == "MiniMaxH3SageAttention", f"{arm}: Sage missing"
        assert g["21"]["class_type"] == "MiniMaxH3Sol", f"{arm}: Sol missing"
        s_in = g["74"]["inputs"]
        assert g["74"]["class_type"] == "MiniMaxH3AudioFreezeSong"
        assert s_in["window_frames"] in chains
        assert h3_rules.on_both_clocks(s_in["context_frames"]) and s_in["context_frames"] < s_in["window_frames"]
        assert s_in["context_profile"] == "seam_taper"
        assert s_in["section_anchor"] in ("off", "per_section", "whole_track")
        assert 0.05 <= float(s_in["anchor_end_percent"]) <= 1.0
        serialized = json.dumps(g)
        assert "/home/" not in serialized and "/mnt/" not in serialized, f"{arm}: non-relative path leaked"

    print(
        f"[dry-run OK] {len(graphs)} Bucket 3 graphs validated "
        f"({args.width}x{args.height}, 4win={got_4}, 2win={got_2}, seed={args.seed}); 0 leaks."
    )
    return 0


def _synthesize_audio_dict(pat, duration_s: float) -> dict[str, Any]:
    import numpy as np
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "drum.wav"
        pat.synthesize_drum_wav(wav_path, sample_rate=32000, duration_s=duration_s)
        with wave.open(str(wav_path), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
        stereo = pcm.reshape(-1, ch).T  # [2, S]
    return {"waveform": torch.from_numpy(stereo).unsqueeze(0).contiguous(), "sample_rate": int(sr)}


def _seam_stats(rgb, luma, seam_idx: int) -> dict[str, Any]:
    import numpy as np

    rgb_f = rgb.astype(np.float64)
    frame_luma = luma.mean(axis=(1, 2), dtype=np.float64)
    seam_l1 = float(np.abs(rgb_f[seam_idx + 1] - rgb_f[seam_idx]).mean(dtype=np.float64))
    seam_luma_jump = float(abs(frame_luma[seam_idx + 1] - frame_luma[seam_idx]))
    neigh_indices = list(range(max(0, seam_idx - 6), seam_idx)) + list(
        range(seam_idx + 1, min(rgb.shape[0] - 1, seam_idx + 7))
    )
    local_adj = [float(np.abs(rgb_f[t + 1] - rgb_f[t]).mean(dtype=np.float64)) for t in neigh_indices]
    local_adj_mean = float(np.mean(local_adj))
    return {
        "seam_l1": round(seam_l1, 4),
        "local_adj_l1": round(local_adj_mean, 4),
        "seam_jump_ratio": round(seam_l1 / max(local_adj_mean, 1e-8), 4),
        "seam_luma_jump": round(seam_luma_jump, 4),
    }


def _window_stats(rgb, luma, sl: slice) -> dict[str, Any]:
    import cv2
    import numpy as np

    sub_rgb = rgb[sl]
    sub_luma = luma[sl]
    rgb_mean = sub_rgb.mean(axis=(0, 1, 2), dtype=np.float64)
    diffs = np.abs(np.diff(sub_luma.astype(np.float64), axis=0)).mean(axis=(1, 2), dtype=np.float64)
    sharp = [float(cv2.Laplacian(sub_luma[idx], cv2.CV_32F).var()) for idx in range(0, sub_luma.shape[0], 4)]
    return {
        "luma_mean": round(float(sub_luma.mean(dtype=np.float64)), 3),
        "rgb_mean": [round(float(x), 2) for x in rgb_mean],
        "cool_b_minus_r": round(float(rgb_mean[2] - rgb_mean[0]), 2),
        "motion_l1": round(float(diffs.mean(dtype=np.float64)), 4),
        "sharpness_lap_var": round(float(np.mean(sharp)), 2),
    }


def _measure_4win_arm(pat, rgb, mp4_path: Path) -> dict[str, Any]:
    luma = pat._luma_series(rgb)
    slices = {
        "w1": slice(0, 141),
        "w2": slice(141, 243),
        "w3": slice(243, 345),
        "w4": slice(345, 447),
    }
    win = {k: _window_stats(rgb, luma, sl) for k, sl in slices.items()}
    w1_luma, w2_luma = win["w1"]["luma_mean"], win["w2"]["luma_mean"]
    w3_luma, w4_luma = win["w3"]["luma_mean"], win["w4"]["luma_mean"]
    s12 = _seam_stats(rgb, luma, 140)
    s23 = _seam_stats(rgb, luma, 242)
    s34 = _seam_stats(rgb, luma, 344)
    return {
        "windows": win,
        "w1_luma_mean": w1_luma,
        "w2_luma_mean": w2_luma,
        "w3_luma_mean": w3_luma,
        "w4_luma_mean": w4_luma,
        "w2_vs_w1_luma_pct": round((w2_luma - w1_luma) / max(w1_luma, 1e-8) * 100.0, 2),
        "w4_vs_w3_luma_pct": round((w4_luma - w3_luma) / max(w3_luma, 1e-8) * 100.0, 2),
        "w3_vs_w1_luma_pct": round((w3_luma - w1_luma) / max(w1_luma, 1e-8) * 100.0, 2),
        "seam_12_f140_141": s12,
        "section_23_f242_243": s23,
        "seam_34_f344_345": s34,
        **pat._measure_audio_dbfs(mp4_path),
    }


def _measure_2win_arm(pat, rgb, mp4_path: Path) -> dict[str, Any]:
    luma = pat._luma_series(rgb)
    w1 = _window_stats(rgb, luma, slice(0, 141))
    w2 = _window_stats(rgb, luma, slice(141, 243))
    w1_luma, w2_luma = w1["luma_mean"], w2["luma_mean"]
    s12 = _seam_stats(rgb, luma, 140)
    return {
        "windows": {"w1": w1, "w2": w2},
        "w1_luma_mean": w1_luma,
        "w2_luma_mean": w2_luma,
        "w2_vs_w1_luma_pct": round((w2_luma - w1_luma) / max(w1_luma, 1e-8) * 100.0, 2),
        "seam_l1_f140_141": s12["seam_l1"],
        "local_adj_l1_mean": s12["local_adj_l1"],
        "seam_jump_ratio": s12["seam_jump_ratio"],
        "seam_luma_jump": s12["seam_luma_jump"],
        "w2_motion_l1": w2["motion_l1"],
        "w2_sharpness_lap_var": w2["sharpness_lap_var"],
        **pat._measure_audio_dbfs(mp4_path),
    }


def run_gpu(args: argparse.Namespace) -> int:
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    from _lib import COMFY, bootstrap, card_visible, needs, server_memory_mode

    bootstrap(cpu=False)

    import comfy.cli_args as cli_args

    cli_args.args.mmap_torch_files = True
    cli_args.args.supports_fp8_compute = True
    cli_args.args.cuda_malloc = True
    cli_args.args.reserve_vram = 1.0
    importlib.import_module("cuda_malloc")  # must run before first `import torch`

    needs("a CUDA device", card_visible())
    dynamic_vram = server_memory_mode()
    import nodes as comfy_nodes  # lock sys.modules['nodes'] to ComfyUI/nodes.py before adding _REPO_ROOT

    emp = COMFY / "extra_model_paths.yaml"
    if emp.is_file():
        importlib.import_module("utils.extra_config").load_extra_path_config(str(emp))

    _ensure_paths()
    import folder_paths
    import h3_config

    out_root = h3_config.output_dir()
    in_root = Path(os.environ.get("H3_INPUT_DIR") or str(out_root.parent / "input"))
    folder_paths.set_output_directory(str(out_root))
    folder_paths.set_input_directory(str(in_root))

    out_dir = out_root / "Video" / args.out_subdir
    data_dir = _REPO_ROOT / "data" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_file = data_dir / "probe_summary.json"
    prev_summary = (
        json.loads(summary_file.read_text(encoding="utf-8"))
        if (args.resume and summary_file.is_file())
        else {}
    )
    prev_arms = prev_summary.get("arms") if isinstance(prev_summary.get("arms"), dict) else {}

    asyncio.run(comfy_nodes.init_builtin_extra_nodes())
    lb_mod = _import_pack_module("lora_branch")
    nodes_mod = _import_pack_module("nodes")
    sol_mod = _import_pack_module("sol_attn_h3")
    enc_mod = _import_pack_module("h3_encoder_loader")
    song_mod = _import_pack_module("audio_freeze_song")
    import probe_anchor_transfer as pat

    print(
        f"[init] dynamic_vram={dynamic_vram}, alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}, "
        f"reserve_vram={cli_args.args.reserve_vram}, loading 6-step Pika pipeline in-process...",
        flush=True,
    )
    t_load0 = time.monotonic()
    unet = comfy_nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"
    )[0]
    lora_m = lb_mod.MiniMaxH3LoRABranch.execute(
        unet,
        "h3/pika_h3_flex_fl2va_svd995_bf16.safetensors",
        strength=1.0,
        modules="all",
        blocks="all",
        start_percent=0.0,
        end_percent=1.0,
    ).args[0]
    shift_m = comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3SigmaShift"].execute(
        model=lora_m, shift_video=12.0, shift_audio=3.0
    ).args[0]
    sage_m = nodes_mod.MiniMaxH3SageAttention.execute(
        shift_m, mode="auto", head_chunks=1, patch_token_refiner=False
    ).args[0]
    sol_m = sol_mod.MiniMaxH3Sol.execute(
        sage_m,
        tau=1.0,
        quantizer="rotated",
        dense_blocks="49",
        sink_conditioning="exact_kv_and_all_rows",
        token_routing={"token_routing": "off"},
        start_percent=0.2,
        end_percent=1.0,
        min_tokens=12288,
        verbose=False,
    ).args[0]
    clip_res = enc_mod.MiniMaxH3EncoderLoader.execute(
        "qwen3vl_32b_minimax_h3_bf16_pruned.safetensors", device="default"
    )
    clip = getattr(clip_res, "args", clip_res)[0]
    vae = comfy_nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(
        "minimax_h3_video_vae_int8_convrot.safetensors"
    )[0]
    audio_vae = comfy_nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(
        "minimax_h3_audio_vae_fp32.safetensors"
    )[0]
    sampler = comfy_nodes.NODE_CLASS_MAPPINGS["KSamplerSelect"].execute(sampler_name="lcm").args[0]
    sigmas = comfy_nodes.NODE_CLASS_MAPPINGS["ManualSigmas"].execute(
        sigmas="1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
    ).args[0]
    load_s = round(time.monotonic() - t_load0, 2)
    print(f"[init] Pika stack ready in {load_s:.2f}s", flush=True)

    summary: dict[str, Any] = {
        "out_subdir": args.out_subdir,
        "width": args.width,
        "height": args.height,
        "context_frames": DEFAULT_CONTEXT,
        "context_noise": CONTEXT_NOISE,
        "context_profile": "seam_taper",
        "seed": args.seed,
        "dynamic_vram": dynamic_vram,
        "load_s": load_s,
        "arms": dict(prev_arms) if args.resume else {},
    }
    for k in ("four_win_stack", "two_win_stack"):
        if args.resume and k in prev_summary:
            summary[k] = prev_summary[k]

    def _flush() -> None:
        summary_file.write_text(
            json.dumps(pat._scrub_paths(summary, out_root, in_root), indent=2),
            encoding="utf-8",
        )

    def _find_saved(prefix: str) -> Path | None:
        hits = [
            p
            for p in sorted(out_dir.glob(f"{prefix}_*.mp4"))
            if not (p.stem.endswith("_with_mask") or p.stem.endswith("_zoomed"))
        ]
        return hits[-1] if hits else None

    def _run_song_arm(
        arm: str,
        *,
        audio_dict: dict[str, Any],
        prompt: str,
        timeline: str,
        window_frames: int,
        section_anchor: str,
        anchor_end_percent: float,
    ) -> tuple[Path, list[str], float]:
        if args.resume and arm in prev_arms:
            existing = _find_saved(arm)
            if existing is not None:
                prev_rec = prev_arms[arm]
                if (
                    prev_rec.get("section_anchor") == section_anchor
                    and float(prev_rec.get("anchor_end_percent", -1.0)) == float(anchor_end_percent)
                ):
                    print(f"[resume] {arm} -> reusing {existing.name}", flush=True)
                    return existing, list(prev_rec.get("anchor_report_lines") or ()), float(prev_rec.get("elapsed_s") or 0.0)

        t0 = time.monotonic()
        res = song_mod.MiniMaxH3AudioFreezeSong.execute(
            sol_m,
            clip,
            vae,
            audio_vae,
            audio_dict,
            sampler,
            sigmas,
            prompt,
            timeline,
            False,
            int(args.width),
            int(args.height),
            int(window_frames),
            DEFAULT_CONTEXT,
            "whole",
            int(args.seed),
            0.0,
            "clip_guard",
            f"Video/{args.out_subdir}/{arm}",
            19,
            save_metadata_png=True,
            keep_windows=False,
            context_noise=CONTEXT_NOISE,
            context_profile="seam_taper",
            section_anchor=section_anchor,
            anchor_end_percent=float(anchor_end_percent),
        )
        elapsed_s = round(time.monotonic() - t0, 2)
        out_path_str, report_text, *_ = getattr(res, "args", res)
        mp4 = Path(out_path_str)
        anchor_lines = [
            ln
            for ln in str(report_text).splitlines()
            if ("anchor" in ln.lower() or "seconds by stage" in ln.lower())
        ]
        print(f"[done] {arm} in {elapsed_s:.2f}s -> {mp4.name}", flush=True)
        for al in anchor_lines:
            print(f"       {al}", flush=True)
        return mp4, anchor_lines, elapsed_s

    size = (args.width, args.height)
    studio_prompt, two_sec_prompt = _prompts()

    # ---- Part 1: 4-Window 2-Section Timeline (studio -> stage) ----
    if args.only in ("4win", "all"):
        audio_4win = _synthesize_audio_dict(pat, SONG_4WIN_FRAMES / 24.0)
        stacked_4: list[tuple[str, Any]] = []
        ref_4_mp4: Path | None = None
        for arm, mode, end_pct in four_win_specs():
            mp4, anchor_lines, elapsed_s = _run_song_arm(
                arm,
                audio_dict=audio_4win,
                prompt=two_sec_prompt,
                timeline=SONG_4WIN_TIMELINE,
                window_frames=SONG_4WIN_MAX_WINDOW,
                section_anchor=mode,
                anchor_end_percent=end_pct,
            )
            if ref_4_mp4 is None:
                ref_4_mp4 = mp4
            rgb = pat._load_rgb_frames(mp4, size, SONG_4WIN_FRAMES)
            stacked_4.append((arm.replace("b3_4win_", ""), rgb))
            metrics = _measure_4win_arm(pat, rgb, mp4)
            summary["arms"][arm] = {
                "file": str(mp4),
                "elapsed_s": elapsed_s,
                "section_anchor": mode,
                "anchor_end_percent": float(end_pct),
                "anchor_report_lines": anchor_lines,
                **metrics,
            }
            _flush()

        off_w3 = summary["arms"]["b3_4win_off"]["w3_luma_mean"]
        for arm, _, _ in four_win_specs():
            rec = summary["arms"][arm]
            rec["w3_vs_off_w3_luma_pct"] = round((rec["w3_luma_mean"] - off_w3) / max(off_w3, 1e-8) * 100.0, 2)
        summary["four_win_stack"] = pat._write_grid_video(
            out_dir,
            "bucket3_4win_stacked",
            stacked_4,
            cols=len(stacked_4),
            rows=1,
            audio_source_mp4=ref_4_mp4,
        )
        _flush()

    # ---- Part 2: 2-Window Hold-Duration Check (0.33 vs 1.00 + Bucket 2 unanchored ref) ----
    if args.only in ("2win", "all"):
        audio_2win = _synthesize_audio_dict(pat, SONG_2WIN_FRAMES / 24.0)
        stacked_2: list[tuple[str, Any]] = []
        ref_2_mp4: Path | None = None
        b2_ref = out_root / "Video" / "bucket2_probe_2026-11-16" / "b2_song_taper_cn060_00001.mp4"
        if b2_ref.is_file():
            b2_rgb = pat._load_rgb_frames(b2_ref, size, SONG_2WIN_FRAMES)
            stacked_2.append(("b2_off_ref", b2_rgb))
            ref_2_mp4 = b2_ref
            summary["b2_taper_off_ref"] = {
                "file": str(b2_ref),
                **_measure_2win_arm(pat, b2_rgb, b2_ref),
            }

        for arm, mode, end_pct in two_win_specs():
            mp4, anchor_lines, elapsed_s = _run_song_arm(
                arm,
                audio_dict=audio_2win,
                prompt=studio_prompt,
                timeline="",
                window_frames=SONG_2WIN_MAX_WINDOW,
                section_anchor=mode,
                anchor_end_percent=end_pct,
            )
            if ref_2_mp4 is None:
                ref_2_mp4 = mp4
            rgb = pat._load_rgb_frames(mp4, size, SONG_2WIN_FRAMES)
            stacked_2.append((arm.replace("b3_2win_", ""), rgb))
            metrics = _measure_2win_arm(pat, rgb, mp4)
            summary["arms"][arm] = {
                "file": str(mp4),
                "elapsed_s": elapsed_s,
                "section_anchor": mode,
                "anchor_end_percent": float(end_pct),
                "anchor_report_lines": anchor_lines,
                **metrics,
            }
            _flush()

        summary["two_win_stack"] = pat._write_grid_video(
            out_dir,
            "bucket3_2win_stacked",
            stacked_2,
            cols=len(stacked_2),
            rows=1,
            audio_source_mp4=ref_2_mp4,
        )
        _flush()

    print(
        f"[complete] Wrote $H3_OUTPUT_DIR/Video/{args.out_subdir} and data/{args.out_subdir}/probe_summary.json",
        flush=True,
    )
    return 0


def recompute_metrics(args: argparse.Namespace) -> int:
    """Recompute all metrics in float64 from existing rendered MP4s without touching the GPU."""
    from _lib import bootstrap

    bootstrap(cpu=True)
    import nodes as _comfy_nodes  # noqa: F401 -- lock sys.modules['nodes'] before _ensure_paths
    _ensure_paths()
    import h3_config
    import probe_anchor_transfer as pat

    out_root = h3_config.output_dir()
    in_root = Path(os.environ.get("H3_INPUT_DIR") or str(out_root.parent / "input"))
    out_dir = out_root / "Video" / args.out_subdir
    summary_file = _REPO_ROOT / "data" / args.out_subdir / "probe_summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(f"Missing summary JSON to recompute: {summary_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    arms = summary.get("arms") or {}
    size = (args.width, args.height)

    def _resolve_arm_mp4(prefix: str, rec: dict[str, Any]) -> Path:
        raw = str(rec.get("file") or "")
        if raw.startswith("$H3_OUTPUT_DIR/"):
            cand = out_root / raw[len("$H3_OUTPUT_DIR/") :]
            if cand.is_file():
                return cand
        hits = [
            p
            for p in sorted(out_dir.glob(f"{prefix}_*.mp4"))
            if not (p.stem.endswith("_with_mask") or p.stem.endswith("_zoomed"))
        ]
        if not hits:
            raise FileNotFoundError(f"No MP4 found for {prefix!r} under {out_dir}")
        return hits[-1]

    if args.only in ("4win", "all"):
        for arm, mode, end_pct in four_win_specs():
            if arm not in arms:
                continue
            prev_rec = arms[arm]
            mp4 = _resolve_arm_mp4(arm, prev_rec)
            rgb = pat._load_rgb_frames(mp4, size, SONG_4WIN_FRAMES)
            metrics = _measure_4win_arm(pat, rgb, mp4)
            arms[arm] = {
                "file": str(mp4),
                "elapsed_s": prev_rec.get("elapsed_s"),
                "section_anchor": mode,
                "anchor_end_percent": float(end_pct),
                "anchor_report_lines": prev_rec.get("anchor_report_lines") or [],
                **metrics,
            }
        if "b3_4win_off" in arms:
            off_w3 = arms["b3_4win_off"]["w3_luma_mean"]
            for arm, _, _ in four_win_specs():
                if arm in arms:
                    rec = arms[arm]
                    rec["w3_vs_off_w3_luma_pct"] = round(
                        (rec["w3_luma_mean"] - off_w3) / max(off_w3, 1e-8) * 100.0, 2
                    )

    if args.only in ("2win", "all"):
        b2_ref = out_root / "Video" / "bucket2_probe_2026-11-16" / "b2_song_taper_cn060_00001.mp4"
        if b2_ref.is_file():
            b2_rgb = pat._load_rgb_frames(b2_ref, size, SONG_2WIN_FRAMES)
            summary["b2_taper_off_ref"] = {
                "file": str(b2_ref),
                **_measure_2win_arm(pat, b2_rgb, b2_ref),
            }
        for arm, mode, end_pct in two_win_specs():
            if arm not in arms:
                continue
            prev_rec = arms[arm]
            mp4 = _resolve_arm_mp4(arm, prev_rec)
            rgb = pat._load_rgb_frames(mp4, size, SONG_2WIN_FRAMES)
            metrics = _measure_2win_arm(pat, rgb, mp4)
            arms[arm] = {
                "file": str(mp4),
                "elapsed_s": prev_rec.get("elapsed_s"),
                "section_anchor": mode,
                "anchor_end_percent": float(end_pct),
                "anchor_report_lines": prev_rec.get("anchor_report_lines") or [],
                **metrics,
            }

    summary["arms"] = arms
    summary_file.write_text(
        json.dumps(pat._scrub_paths(summary, out_root, in_root), indent=2),
        encoding="utf-8",
    )
    print(f"[recompute OK] Updated data/{args.out_subdir}/probe_summary.json in float64", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate graphs and timelines offline (default)")
    parser.add_argument("--run", action="store_true", help="Execute Bucket 3 probes in-process on CUDA")
    parser.add_argument("--resume", action="store_true", help="Reuse matching existing renders in --out-subdir")
    parser.add_argument("--recompute-metrics", action="store_true", help="Recompute float64 metrics from saved MP4s offline")
    parser.add_argument("--only", default="all", choices=("4win", "2win", "all"), help="Subset of arms to run")
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.run:
        return run_gpu(args)
    if args.recompute_metrics:
        return recompute_metrics(args)
    return verify_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
