#!/usr/bin/env python3
"""Bucket 4 probe: Mass-conserving `return_lse=True` subset balance + `section_anchor` `0.33` release.

Validates offline via `--dry-run` (default) or executes in-process on CUDA via `--run`:
  2-Window Studio Hold Sweep (243f = 10.125s = 141f + 102f, ctx 39f, seam_taper 0.6,
  section_anchor='per_section', anchor_end_percent=0.33, anchor_blocks='30-49'):
    1. `b4_2win_w4_b30_49`   -- Sage(auto) + Sol, anchor_weight=4.0
    2. `b4_2win_w8_b30_49`   -- Sage(auto) + Sol, anchor_weight=8.0
    3. `b4_2win_kitchen_w4`  -- ModelAttentionBackend('comfy kitchen attention') + Sol,
                                anchor_weight=4.0 (+ `sol_observe raw=0` route counts)

  4-Window 2-Section Timeline Check (447f = 18.625s, studio -> stage at 10.125s):
    4. `b4_4win_per_sec_w4`  -- Sage(auto) + Sol, anchor_weight=4.0

Runs in-process via `server_memory_mode()` so no `:8188` restart is needed. All media writes live under
`$H3_OUTPUT_DIR/Video/bucket4_probe_2026-11-16/` (zero media in `data/`).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

DEFAULT_OUT_SUBDIR = "bucket4_probe_2026-11-16"
DEFAULT_ANCHOR_BLOCKS = "30-49"
DEFAULT_ANCHOR_END = 0.33
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_CONTEXT = 39
DEFAULT_SEED = 730451892
CONTEXT_NOISE = 0.6
SONG_4WIN_FRAMES = 447
SONG_4WIN_MAX_WINDOW = 192
SONG_4WIN_TIMELINE = "00:00 studio\n00:10.125 stage"
SONG_2WIN_FRAMES = 243
SONG_2WIN_MAX_WINDOW = 141


def two_win_specs() -> list[tuple[str, str, float, str]]:
    return [
        ("b4_2win_w4_b30_49", "sage", 4.0, DEFAULT_ANCHOR_BLOCKS),
        ("b4_2win_w8_b30_49", "sage", 8.0, DEFAULT_ANCHOR_BLOCKS),
        ("b4_2win_kitchen_w4", "kitchen", 4.0, DEFAULT_ANCHOR_BLOCKS),
    ]


def four_win_specs() -> list[tuple[str, str, float, str]]:
    return [
        ("b4_4win_per_sec_w4", "sage", 4.0, DEFAULT_ANCHOR_BLOCKS),
    ]


def _make_base_chain(
    pat,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    out_subdir: str,
    arm: str,
    chain: str,
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
    if chain == "kitchen":
        # Replace Sage node "58" with ModelAttentionBackend("comfy kitchen attention")
        g["58"] = {
            "class_type": "ModelAttentionBackend",
            "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"},
        }
    return g


def build_song_graph(
    pat,
    *,
    arm: str,
    chain: str,
    prompt: str,
    timeline: str,
    window_frames: int,
    anchor_weight: float,
    anchor_blocks: str,
    audio_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    g = _make_base_chain(
        pat,
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        out_subdir=out_subdir,
        arm=arm,
        chain=chain,
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
            "context_frames": DEFAULT_CONTEXT,
            "extent": "whole",
            "seed": int(seed),
            "audio_mask": 0.0,
            "level": "clip_guard",
            "filename_prefix": f"Video/{out_subdir}/{arm}",
            "crf": 19,
            "save_metadata_png": True,
            "keep_windows": False,
            "context_noise": CONTEXT_NOISE,
            "context_profile": "seam_taper",
            "section_anchor": "per_section",
            "anchor_end_percent": DEFAULT_ANCHOR_END,
            "anchor_weight": float(anchor_weight),
            "anchor_blocks": str(anchor_blocks),
        },
    }
    return g


def build_all_graphs(pat, pb3, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    studio_prompt, two_sec_prompt = pb3._prompts()
    graphs: dict[str, dict[str, Any]] = {}
    if args.only in ("2win", "all"):
        for arm, chain, weight, blocks in two_win_specs():
            graphs[arm] = build_song_graph(
                pat,
                arm=arm,
                chain=chain,
                prompt=studio_prompt,
                timeline="",
                window_frames=SONG_2WIN_MAX_WINDOW,
                anchor_weight=weight,
                anchor_blocks=blocks,
                audio_name="h3_probe_20261116_b4_drum_10p125s.wav",
                width=args.width,
                height=args.height,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    if args.only in ("4win", "all"):
        for arm, chain, weight, blocks in four_win_specs():
            graphs[arm] = build_song_graph(
                pat,
                arm=arm,
                chain=chain,
                prompt=two_sec_prompt,
                timeline=SONG_4WIN_TIMELINE,
                window_frames=SONG_4WIN_MAX_WINDOW,
                anchor_weight=weight,
                anchor_blocks=blocks,
                audio_name="h3_probe_20261116_b4_drum_18p625s.wav",
                width=args.width,
                height=args.height,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    return graphs


def verify_dry_run(args: argparse.Namespace) -> int:
    from _lib import bootstrap

    bootstrap(cpu=True)
    import nodes as _comfy_nodes  # noqa: F401
    import probe_bucket3_section_anchor as pb3

    pb3._ensure_paths()
    import h3_rules
    import probe_anchor_transfer as pat

    graphs = build_all_graphs(pat, pb3, args)
    chains = set(h3_rules.chain_lengths())
    for arm, g in graphs.items():
        if "kitchen" in arm:
            assert g["58"]["class_type"] == "ModelAttentionBackend", f"{arm}: kitchen backend missing"
            assert g["58"]["inputs"]["attention"] == "comfy kitchen attention"
        else:
            assert g["58"]["class_type"] == "MiniMaxH3SageAttention", f"{arm}: Sage missing"
        assert g["21"]["class_type"] == "MiniMaxH3Sol", f"{arm}: Sol missing"
        s_in = g["74"]["inputs"]
        assert g["74"]["class_type"] == "MiniMaxH3AudioFreezeSong"
        assert s_in["window_frames"] in chains
        assert h3_rules.on_both_clocks(s_in["context_frames"])
        assert s_in["context_profile"] == "seam_taper"
        assert s_in["section_anchor"] == "per_section"
        assert float(s_in["anchor_end_percent"]) == DEFAULT_ANCHOR_END
        assert 0.1 <= float(s_in["anchor_weight"]) <= 16.0
        assert s_in["anchor_blocks"] == DEFAULT_ANCHOR_BLOCKS
        serialized = json.dumps(g)
        assert "/home/" not in serialized and "/mnt/" not in serialized, f"{arm}: path leak"

    print(
        f"[dry-run OK] {len(graphs)} Bucket 4 graphs validated "
        f"({args.width}x{args.height}, anchor_end={DEFAULT_ANCHOR_END}, "
        f"anchor_blocks={DEFAULT_ANCHOR_BLOCKS!r}, seed={args.seed}); 0 leaks."
    )
    return 0


def _summarize_sol_observe_rows(obs_mod, jsonl_path: Path) -> dict[str, Any]:
    rows = obs_mod.read_rows(jsonl_path)
    calls = [r for r in rows if r.get("kind") == "call"]
    dit_calls = [r for r in calls if r.get("block") is not None]
    total_routes = Counter(str(r.get("route")) for r in dit_calls)
    lse_blocks = sorted({int(r["block"]) for r in dit_calls if r.get("route") == "lse_balance"})
    w1_routes = Counter(str(r.get("route")) for r in dit_calls[:300])
    w2_routes = Counter(str(r.get("route")) for r in dit_calls[300:])
    return {
        "n_calls_total": len(calls),
        "n_dit_calls": len(dit_calls),
        "total_dit_routes": dict(sorted(total_routes.items())),
        "window1_dit_routes": dict(sorted(w1_routes.items())),
        "window2_dit_routes": dict(sorted(w2_routes.items())),
        "lse_balance_blocks": lse_blocks,
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
    importlib.import_module("cuda_malloc")

    needs("a CUDA device", card_visible())
    dynamic_vram = server_memory_mode()
    import nodes as comfy_nodes

    emp = COMFY / "extra_model_paths.yaml"
    if emp.is_file():
        importlib.import_module("utils.extra_config").load_extra_path_config(str(emp))

    import probe_bucket3_section_anchor as pb3

    pb3._ensure_paths()
    import folder_paths
    import h3_config
    import probe_anchor_transfer as pat

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
    lb_mod = pb3._import_pack_module("lora_branch")
    nodes_mod = pb3._import_pack_module("nodes")
    sol_mod = pb3._import_pack_module("sol_attn_h3")
    obs_mod = pb3._import_pack_module("sol_observe")
    enc_mod = pb3._import_pack_module("h3_encoder_loader")
    song_mod = pb3._import_pack_module("audio_freeze_song")

    print(
        f"[init] dynamic_vram={dynamic_vram}, alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}, "
        f"loading 6-step Pika Sage+Sol and Kitchen+Sol chains...",
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
    sol_sage_m = sol_mod.MiniMaxH3Sol.execute(
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

    kitchen_m = comfy_nodes.NODE_CLASS_MAPPINGS["ModelAttentionBackend"].execute(
        model=shift_m, attention="comfy kitchen attention"
    ).args[0]
    sol_kitchen_m = sol_mod.MiniMaxH3Sol.execute(
        kitchen_m,
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

    clip = getattr(
        enc_mod.MiniMaxH3EncoderLoader.execute("qwen3vl_32b_minimax_h3_bf16_pruned.safetensors", device="default"),
        "args",
        None,
    )[0]
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
    print(f"[init] Pika chains ready in {load_s:.2f}s", flush=True)

    summary: dict[str, Any] = {
        "out_subdir": args.out_subdir,
        "width": args.width,
        "height": args.height,
        "context_frames": DEFAULT_CONTEXT,
        "context_noise": CONTEXT_NOISE,
        "context_profile": "seam_taper",
        "section_anchor": "per_section",
        "anchor_end_percent": DEFAULT_ANCHOR_END,
        "anchor_blocks": DEFAULT_ANCHOR_BLOCKS,
        "seed": args.seed,
        "dynamic_vram": dynamic_vram,
        "load_s": load_s,
        "arms": dict(prev_arms) if args.resume else {},
    }
    for k in ("two_win_stack", "four_win_stack"):
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

    def _run_arm(
        arm: str,
        *,
        chain: str,
        audio_dict: dict[str, Any],
        prompt: str,
        timeline: str,
        window_frames: int,
        anchor_weight: float,
        anchor_blocks: str,
        observe_sol: bool = False,
    ) -> tuple[Path, list[str], float, dict[str, Any] | None]:
        if args.resume and arm in prev_arms:
            existing = _find_saved(arm)
            if existing is not None:
                prev_rec = prev_arms[arm]
                if (
                    prev_rec.get("chain") == chain
                    and float(prev_rec.get("anchor_weight", -1.0)) == float(anchor_weight)
                    and prev_rec.get("anchor_blocks") == anchor_blocks
                ):
                    print(f"[resume] {arm} -> reusing {existing.name}", flush=True)
                    return (
                        existing,
                        list(prev_rec.get("anchor_report_lines") or ()),
                        float(prev_rec.get("elapsed_s") or 0.0),
                        prev_rec.get("sol_observe_summary"),
                    )

        model_in = sol_kitchen_m if chain == "kitchen" else sol_sage_m
        obs_summary: dict[str, Any] | None = None
        if observe_sol:
            with tempfile.TemporaryDirectory(prefix="b4_sol_obs_") as obs_tmp:
                obs_mod.arm(f"dir={obs_tmp},raw=0")
                try:
                    t0 = time.monotonic()
                    res = song_mod.MiniMaxH3AudioFreezeSong.execute(
                        model_in,
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
                        section_anchor="per_section",
                        anchor_end_percent=DEFAULT_ANCHOR_END,
                        anchor_weight=float(anchor_weight),
                        anchor_blocks=str(anchor_blocks),
                    )
                    elapsed_s = round(time.monotonic() - t0, 2)
                finally:
                    obs_mod.arm(None)
                jfiles = sorted(Path(obs_tmp).glob("sol_observe_*.jsonl"))
                if jfiles:
                    obs_summary = _summarize_sol_observe_rows(obs_mod, jfiles[0])
        else:
            t0 = time.monotonic()
            res = song_mod.MiniMaxH3AudioFreezeSong.execute(
                model_in,
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
                section_anchor="per_section",
                anchor_end_percent=DEFAULT_ANCHOR_END,
                anchor_weight=float(anchor_weight),
                anchor_blocks=str(anchor_blocks),
            )
            elapsed_s = round(time.monotonic() - t0, 2)

        out_path_str, report_text, *_ = getattr(res, "args", res)
        mp4 = Path(out_path_str)
        anchor_lines = [
            ln
            for ln in str(report_text).splitlines()
            if ("anchor" in ln.lower() or "seconds by stage" in ln.lower())
        ]
        print(f"[done] {arm} ({chain}, w={anchor_weight}) in {elapsed_s:.2f}s -> {mp4.name}", flush=True)
        for al in anchor_lines:
            print(f"       {al}", flush=True)
        if obs_summary is not None:
            print(f"       sol_observe: {obs_summary}", flush=True)
        return mp4, anchor_lines, elapsed_s, obs_summary

    size = (args.width, args.height)
    studio_prompt, two_sec_prompt = pb3._prompts()

    # ---- Part 1: 2-Window Studio Hold Sweep ----
    if args.only in ("2win", "all"):
        audio_2win = pb3._synthesize_audio_dict(pat, SONG_2WIN_FRAMES / 24.0)
        stacked_2: list[tuple[str, Any]] = []
        ref_2_mp4: Path | None = None

        b3_ref = out_root / "Video" / "bucket3_probe_2026-11-16" / "b3_2win_rel033_00001.mp4"
        if b3_ref.is_file():
            b3_rgb = pat._load_rgb_frames(b3_ref, size, SONG_2WIN_FRAMES)
            stacked_2.append(("b3_w1", b3_rgb))
            ref_2_mp4 = b3_ref
            summary["b3_2win_rel033_ref"] = {
                "file": str(b3_ref),
                **pb3._measure_2win_arm(pat, b3_rgb, b3_ref),
            }

        for arm, chain, weight, blocks in two_win_specs():
            mp4, anchor_lines, elapsed_s, obs_summary = _run_arm(
                arm,
                chain=chain,
                audio_dict=audio_2win,
                prompt=studio_prompt,
                timeline="",
                window_frames=SONG_2WIN_MAX_WINDOW,
                anchor_weight=weight,
                anchor_blocks=blocks,
                observe_sol=(chain == "kitchen"),
            )
            if ref_2_mp4 is None:
                ref_2_mp4 = mp4
            rgb = pat._load_rgb_frames(mp4, size, SONG_2WIN_FRAMES)
            stacked_2.append((arm.replace("b4_2win_", ""), rgb))
            metrics = pb3._measure_2win_arm(pat, rgb, mp4)
            rec: dict[str, Any] = {
                "file": str(mp4),
                "chain": chain,
                "elapsed_s": elapsed_s,
                "anchor_weight": float(weight),
                "anchor_blocks": blocks,
                "anchor_report_lines": anchor_lines,
                **metrics,
            }
            if obs_summary is not None:
                rec["sol_observe_summary"] = obs_summary
            summary["arms"][arm] = rec
            _flush()

        summary["two_win_stack"] = pat._write_grid_video(
            out_dir,
            "bucket4_2win_stacked",
            stacked_2,
            cols=len(stacked_2),
            rows=1,
            audio_source_mp4=ref_2_mp4,
        )
        _flush()

    # ---- Part 2: 4-Window 2-Section Timeline Check (studio -> stage) ----
    if args.only in ("4win", "all"):
        audio_4win = pb3._synthesize_audio_dict(pat, SONG_4WIN_FRAMES / 24.0)
        stacked_4: list[tuple[str, Any]] = []
        ref_4_mp4: Path | None = None

        b3_4ref = out_root / "Video" / "bucket3_probe_2026-11-16" / "b3_4win_per_section_00001.mp4"
        if b3_4ref.is_file():
            b3_4rgb = pat._load_rgb_frames(b3_4ref, size, SONG_4WIN_FRAMES)
            stacked_4.append(("b3_w1", b3_4rgb))
            ref_4_mp4 = b3_4ref
            summary["b3_4win_per_sec_ref"] = {
                "file": str(b3_4ref),
                **pb3._measure_4win_arm(pat, b3_4rgb, b3_4ref),
            }

        for arm, chain, weight, blocks in four_win_specs():
            mp4, anchor_lines, elapsed_s, _obs = _run_arm(
                arm,
                chain=chain,
                audio_dict=audio_4win,
                prompt=two_sec_prompt,
                timeline=SONG_4WIN_TIMELINE,
                window_frames=SONG_4WIN_MAX_WINDOW,
                anchor_weight=weight,
                anchor_blocks=blocks,
                observe_sol=False,
            )
            if ref_4_mp4 is None:
                ref_4_mp4 = mp4
            rgb = pat._load_rgb_frames(mp4, size, SONG_4WIN_FRAMES)
            stacked_4.append(("b4_w4", rgb))
            metrics = pb3._measure_4win_arm(pat, rgb, mp4)
            summary["arms"][arm] = {
                "file": str(mp4),
                "chain": chain,
                "elapsed_s": elapsed_s,
                "anchor_weight": float(weight),
                "anchor_blocks": blocks,
                "anchor_report_lines": anchor_lines,
                **metrics,
            }
            _flush()

        summary["four_win_stack"] = pat._write_grid_video(
            out_dir,
            "bucket4_4win_stacked",
            stacked_4,
            cols=len(stacked_4),
            rows=1,
            audio_source_mp4=ref_4_mp4,
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
    import nodes as _comfy_nodes  # noqa: F401
    import probe_bucket3_section_anchor as pb3

    pb3._ensure_paths()
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

    if args.only in ("2win", "all"):
        for arm, chain, weight, blocks in two_win_specs():
            if arm not in arms:
                continue
            prev_rec = arms[arm]
            mp4 = _resolve_arm_mp4(arm, prev_rec)
            rgb = pat._load_rgb_frames(mp4, size, SONG_2WIN_FRAMES)
            rec = {
                "file": str(mp4),
                "chain": chain,
                "elapsed_s": prev_rec.get("elapsed_s"),
                "anchor_weight": float(weight),
                "anchor_blocks": blocks,
                "anchor_report_lines": prev_rec.get("anchor_report_lines") or [],
                **pb3._measure_2win_arm(pat, rgb, mp4),
            }
            if "sol_observe_summary" in prev_rec:
                rec["sol_observe_summary"] = prev_rec["sol_observe_summary"]
            arms[arm] = rec

    if args.only in ("4win", "all"):
        for arm, chain, weight, blocks in four_win_specs():
            if arm not in arms:
                continue
            prev_rec = arms[arm]
            mp4 = _resolve_arm_mp4(arm, prev_rec)
            rgb = pat._load_rgb_frames(mp4, size, SONG_4WIN_FRAMES)
            arms[arm] = {
                "file": str(mp4),
                "chain": chain,
                "elapsed_s": prev_rec.get("elapsed_s"),
                "anchor_weight": float(weight),
                "anchor_blocks": blocks,
                "anchor_report_lines": prev_rec.get("anchor_report_lines") or [],
                **pb3._measure_4win_arm(pat, rgb, mp4),
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
    parser.add_argument("--dry-run", action="store_true", help="Validate graphs and knobs offline (default)")
    parser.add_argument("--run", action="store_true", help="Execute Bucket 4 probes in-process on CUDA")
    parser.add_argument("--resume", action="store_true", help="Reuse matching existing renders in --out-subdir")
    parser.add_argument("--recompute-metrics", action="store_true", help="Recompute float64 metrics offline")
    parser.add_argument("--only", default="all", choices=("2win", "4win", "all"), help="Subset of arms to run")
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
