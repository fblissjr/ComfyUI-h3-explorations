#!/usr/bin/env python3
"""Live `:8188` Bucket 2 probe: AudioAttentionGain guard/scoping & Song seam_taper A/B.

Validates offline via `--dry-run` (default) or executes sequentially on `:8188` via `--run`:
  Part A (`AudioAttentionGain` block scoping & double-scaling guard fix, 141f Pika):
    1. `b2_gain_all_fixed`  -- `key_gain=2.0, value_gain=1.5, blocks="all"` with Block-49 guard active
    2. `b2_gain_0_29_mild`  -- `key_gain=1.35, value_gain=1.15, blocks="0-29"` mild rhythm gain

  Part B (2-window `MiniMaxH3AudioFreezeSong` causal seam taper A/B, 243f = 2 x 141f, 39f ctx):
    1. `b2_song_flat_cn060`   -- `context_noise=0.6, context_profile="flat"`
    2. `b2_song_taper_cn060`  -- `context_noise=0.6, context_profile="seam_taper"` (c-2, c-1 pinned clean)

All media writes under `$H3_OUTPUT_DIR/Video/<out-subdir>/` (zero media in `data/`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT / "workflows") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflows"))

from _lib import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

import _paths  # noqa: E402
import h3_config  # noqa: E402
import h3_rules  # noqa: E402
import preflight_graph  # noqa: E402
import probe_anchor_transfer as pat  # noqa: E402
from _lib import comfy_client  # noqa: E402

DEFAULT_OUT_SUBDIR = "bucket2_probe_2026-11-16"
DEFAULT_WIDTH = pat.DEFAULT_WIDTH
DEFAULT_HEIGHT = pat.DEFAULT_HEIGHT
DEFAULT_LENGTH = pat.DEFAULT_LENGTH  # 141f = 5.875s on_both_clocks
DEFAULT_CONTEXT = 39
SONG_FRAMES = 2 * DEFAULT_LENGTH - DEFAULT_CONTEXT  # 243f = 10.125s
DEFAULT_SEED = pat.DEFAULT_SEED

DRUM_7S_NAME = "h3_probe_20261116_b2_drum_7s.wav"
DRUM_10P125S_NAME = "h3_probe_20261116_b2_drum_10p125s.wav"


def part_a_specs() -> list[tuple[str, float, float, str]]:
    return [
        ("b2_gain_all_fixed", 2.0, 1.5, "all"),
        ("b2_gain_0_29_mild", 1.35, 1.15, "0-29"),
    ]


def part_b_specs() -> list[tuple[str, float, str]]:
    return [
        ("b2_song_flat_cn060", 0.6, "flat"),
        ("b2_song_taper_cn060", 0.6, "seam_taper"),
    ]


def build_part_a_graph(
    *,
    arm: str,
    key_gain: float,
    value_gain: float,
    blocks: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    return pat.build_probe1a_graph(
        arm=arm,
        key_gain=key_gain,
        value_gain=value_gain,
        blocks=blocks,
        width=width,
        height=height,
        length=length,
        seed=seed,
        out_subdir=out_subdir,
        audio_name=DRUM_7S_NAME,
    )


def build_part_b_graph(
    *,
    arm: str,
    context_noise: float,
    context_profile: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    context_frames: int = DEFAULT_CONTEXT,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    """Build a 2-window Pika `MiniMaxH3AudioFreezeSong` graph on `t2va_studio_dancer_close.txt`."""
    prompt = pat.PROMPT_DANCER.read_text(encoding="utf-8").strip()
    g = pat._make_base_graph(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=seed,
        out_subdir=out_subdir,
        arm=arm,
    )
    sigmas_link = g["10"]["inputs"]["sigmas"]
    model_link = g["9"]["inputs"]["model"]
    sampler_link = g["10"]["inputs"]["sampler"]
    for nid in ("5", "6", "9", "10", "11", "12", "13", "26", "27"):
        g.pop(nid, None)
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": DRUM_10P125S_NAME}}
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
            "timeline": "",
            "preview": False,
            "width": int(width),
            "height": int(height),
            "window_frames": int(length),
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
            "context_profile": str(context_profile),
        },
    }
    return g


def build_all_graphs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    if args.only in ("a", "all"):
        for arm, kg, vg, blks in part_a_specs():
            graphs[arm] = build_part_a_graph(
                arm=arm,
                key_gain=kg,
                value_gain=vg,
                blocks=blks,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    if args.only in ("b", "all"):
        for arm, cn, prof in part_b_specs():
            graphs[arm] = build_part_b_graph(
                arm=arm,
                context_noise=cn,
                context_profile=prof,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    return graphs


def verify_dry_run(args: argparse.Namespace) -> int:
    graphs = build_all_graphs(args)
    chains = set(h3_rules.chain_lengths())
    for arm, g in graphs.items():
        assert g["58"]["class_type"] == "MiniMaxH3SageAttention", f"{arm}: Sage node missing"
        assert g["21"]["class_type"] == "MiniMaxH3Sol", f"{arm}: Sol node missing"
        if "74" in g:
            s_in = g["74"]["inputs"]
            assert g["74"]["class_type"] == "MiniMaxH3AudioFreezeSong"
            assert s_in["window_frames"] in chains, f"{arm}: window_frames={s_in['window_frames']} off chain_lengths"
            assert h3_rules.on_both_clocks(s_in["context_frames"]) and s_in["context_frames"] < s_in["window_frames"]
            assert s_in["context_profile"] in ("flat", "seam_taper")
        else:
            findings = preflight_graph.grade(g["5"], g, stem=arm)
            fails = [msg for lvl, msg in findings if lvl == "FAIL"]
            if fails:
                raise AssertionError(f"preflight_graph FAIL on {arm}: {fails}")
        serialized = json.dumps(g)
        assert "/home/" not in serialized and "/mnt/" not in serialized, f"{arm}: non-relative path leaked"
    print(
        f"[dry-run OK] {len(graphs)} graphs validated ({args.width}x{args.height}, length={args.length}, "
        f"song_frames={2 * args.length - DEFAULT_CONTEXT}, seed={args.seed}, only={args.only}); "
        "0 preflight FAIL; all use MiniMaxH3SageAttention + MiniMaxH3Sol(49)."
    )
    return 0


def _measure_part_b_arm(
    rgb: np.ndarray,
    mp4_path: Path,
    *,
    seam_idx: int = DEFAULT_LENGTH - 1,
) -> dict[str, Any]:
    """Measure seam discontinuity at `seam_idx -> seam_idx + 1` and W2 vs W1 luma stability."""
    luma = pat._luma_series(rgb)  # [T, H, W]
    frame_luma = luma.mean(axis=(1, 2))
    rgb_f = rgb.astype(np.float32)

    seam_l1 = float(np.abs(rgb_f[seam_idx + 1] - rgb_f[seam_idx]).mean())
    seam_luma_jump = float(abs(frame_luma[seam_idx + 1] - frame_luma[seam_idx]))
    neigh_indices = list(range(max(0, seam_idx - 6), seam_idx)) + list(
        range(seam_idx + 1, min(rgb.shape[0] - 1, seam_idx + 7))
    )
    local_adj = [float(np.abs(rgb_f[t + 1] - rgb_f[t]).mean()) for t in neigh_indices]
    local_adj_mean = float(np.mean(local_adj))

    w1_luma = float(frame_luma[: seam_idx + 1].mean())
    w2_luma = float(frame_luma[seam_idx + 1 :].mean())
    audio_stats = pat._measure_audio_dbfs(mp4_path)

    return {
        "seam_frame_left": int(seam_idx),
        "seam_frame_right": int(seam_idx + 1),
        "seam_l1_f140_141": round(seam_l1, 4),
        "local_adj_l1_mean": round(local_adj_mean, 4),
        "seam_jump_ratio": round(seam_l1 / max(local_adj_mean, 1e-8), 4),
        "seam_luma_jump": round(seam_luma_jump, 4),
        "w1_luma_mean": round(w1_luma, 3),
        "w2_luma_mean": round(w2_luma, 3),
        "w2_vs_w1_luma_pct": round((w2_luma - w1_luma) / max(w1_luma, 1e-8) * 100.0, 2),
        **audio_stats,
    }


def _find_song_mp4(out_dir: Path, entry: dict[str, Any], arm: str) -> Path:
    try:
        return pat._pick_saved_video(entry, out_dir.parent.parent, file_prefix=arm)
    except Exception:
        hits = sorted(out_dir.glob(f"{arm}_*.mp4"))
        if hits:
            return hits[-1]
        raise


def run_live(args: argparse.Namespace) -> int:
    q = comfy_client.get_queue(base_url=args.url)
    if (q.get("queue_running") or q.get("queue_pending")) and not args.resume:
        raise RuntimeError(f"ComfyUI queue is not empty: {q}")

    out_root = h3_config.output_dir()
    in_root = _paths.comfy_input()
    out_dir = out_root / "Video" / args.out_subdir
    prov_dir = out_root / "provenance"
    data_dir = _REPO_ROOT / "data" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_prompts_file = data_dir / "prompts.json"
    saved_summary_file = data_dir / "probe_summary.json"
    prev_prompts = (
        json.loads(saved_prompts_file.read_text(encoding="utf-8"))
        if (args.resume and saved_prompts_file.is_file())
        else {}
    )
    prev_summary = (
        json.loads(saved_summary_file.read_text(encoding="utf-8"))
        if (args.resume and saved_summary_file.is_file())
        else {}
    )
    prev_graphs = prev_prompts.get("graphs") if isinstance(prev_prompts.get("graphs"), dict) else {}
    prev_arms = prev_summary.get("arms") if isinstance(prev_summary.get("arms"), dict) else {}

    queued_graphs: dict[str, Any] = dict(prev_graphs) if args.resume else {}
    arm_files: dict[str, str] = (
        dict(prev_prompts.get("saved_files"))
        if (args.resume and isinstance(prev_prompts.get("saved_files"), dict))
        else {}
    )

    def _flush_prompts() -> None:
        rec = {"out_subdir": args.out_subdir, "saved_files": arm_files, "graphs": queued_graphs}
        saved_prompts_file.write_text(json.dumps(pat._scrub_paths(rec, out_root, in_root), indent=2), encoding="utf-8")

    def _find_existing(prefix: str) -> Path | None:
        audio_hits = sorted(out_dir.glob(f"{prefix}_*-audio.mp4"))
        if audio_hits:
            return audio_hits[-1]
        hits = sorted(out_dir.glob(f"{prefix}_*.mp4"))
        return hits[-1] if hits else None

    def _run_arm(
        arm: str,
        graph: dict[str, Any],
        *,
        is_song: bool = False,
    ) -> tuple[Path, float | None, float, dict[str, Any] | None]:
        queued_graphs[arm] = graph
        if args.resume and arm in prev_graphs:
            cand = _find_existing(arm)
            if cand is not None and json.dumps(prev_graphs[arm], sort_keys=True) == json.dumps(
                pat._scrub_paths(graph, out_root, in_root), sort_keys=True
            ):
                prev_rec = prev_arms.get(arm) if isinstance(prev_arms.get(arm), dict) else {}
                arm_files[arm] = str(cand)
                _flush_prompts()
                print(f"[resume] {arm} -> reusing {cand.name}", flush=True)
                return cand, prev_rec.get("execution_s"), 0.0, prev_rec.get("provenance")

        existing_stamps = set(prov_dir.glob("stamp_*.json")) if prov_dir.is_dir() else set()
        t0 = time.monotonic()
        pid = comfy_client.submit_prompt(graph, base_url=args.url)
        print(f"[queue] {arm} -> {pid}", flush=True)
        entry = comfy_client.wait_for_prompt(pid, base_url=args.url)
        elapsed_s = round(time.monotonic() - t0, 2)
        exec_s = comfy_client.execution_duration_s(entry)
        saved = _find_song_mp4(out_dir, entry, arm) if is_song else pat._pick_saved_video(entry, out_root, file_prefix=arm)
        stamp = None if is_song else pat._collect_provenance_stamp(prov_dir, existing_stamps, arm)
        arm_files[arm] = str(saved)
        _flush_prompts()
        print(f"[done] {arm} in {elapsed_s:.2f}s (gpu {exec_s or 0:.2f}s) -> {saved.name}", flush=True)
        return saved, exec_s, elapsed_s, stamp

    summary: dict[str, Any] = {
        "out_subdir": args.out_subdir,
        "width": args.width,
        "height": args.height,
        "length": args.length,
        "context_frames": DEFAULT_CONTEXT,
        "song_frames": 2 * args.length - DEFAULT_CONTEXT,
        "seed": args.seed,
        "arms": dict(prev_arms) if args.resume else {},
    }
    if args.resume:
        for stack_key in ("part_a_stack", "part_b_stack"):
            if stack_key in prev_summary:
                summary[stack_key] = prev_summary[stack_key]

    size = (args.width, args.height)

    # ---- Part A: Guarded AudioAttentionGain ----
    if args.only in ("a", "all"):
        drum_7s = pat.synthesize_drum_wav(in_root / DRUM_7S_NAME, duration_s=7.0)
        print(f"[fixture] synthesized {drum_7s.name} in $H3_INPUT_DIR", flush=True)
        part_a_videos: list[tuple[str, np.ndarray]] = []
        audio_ref_mp4: Path | None = None

        g1_off = out_root / "Video" / "anchor_probe_2026-11-16" / "p1a_gain_off_00001-audio.mp4"
        if g1_off.is_file():
            part_a_videos.append(("g1_off_ref", pat._load_rgb_frames(g1_off, size, args.length)))
            audio_ref_mp4 = g1_off

        for arm, kg, vg, blks in part_a_specs():
            g = build_part_a_graph(
                arm=arm,
                key_gain=kg,
                value_gain=vg,
                blocks=blks,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
            mp4, exec_s, elapsed_s, stamp = _run_arm(arm, g, is_song=False)
            if audio_ref_mp4 is None:
                audio_ref_mp4 = mp4
            rgb = pat._load_rgb_frames(mp4, size, args.length)
            part_a_videos.append((arm.replace("b2_", ""), rgb))
            metrics = pat._measure_probe1a_arm(rgb, mp4)
            summary["arms"][arm] = {
                "file": str(mp4),
                "execution_s": exec_s,
                "elapsed_s": elapsed_s,
                "key_gain": kg,
                "value_gain": vg,
                "blocks": blks,
                "provenance": stamp,
                **metrics,
            }

        summary["part_a_stack"] = pat._write_grid_video(
            out_dir,
            "bucket2_gain_stacked",
            part_a_videos,
            cols=len(part_a_videos),
            rows=1,
            audio_source_mp4=audio_ref_mp4,
        )

    # ---- Part B: 2-Window Song seam_taper A/B ----
    if args.only in ("b", "all"):
        song_frames = 2 * args.length - DEFAULT_CONTEXT
        drum_10s = pat.synthesize_drum_wav(in_root / DRUM_10P125S_NAME, duration_s=song_frames / 24.0)
        print(f"[fixture] synthesized {drum_10s.name} ({song_frames / 24.0:.3f}s) in $H3_INPUT_DIR", flush=True)
        part_b_videos: list[tuple[str, np.ndarray]] = []
        audio_ref_song: Path | None = None

        for arm, cn, prof in part_b_specs():
            g = build_part_b_graph(
                arm=arm,
                context_noise=cn,
                context_profile=prof,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
            mp4, exec_s, elapsed_s, _ = _run_arm(arm, g, is_song=True)
            if audio_ref_song is None:
                audio_ref_song = mp4
            rgb = pat._load_rgb_frames(mp4, size, song_frames)
            part_b_videos.append((arm.replace("b2_song_", ""), rgb))
            metrics = _measure_part_b_arm(rgb, mp4, seam_idx=args.length - 1)
            summary["arms"][arm] = {
                "file": str(mp4),
                "execution_s": exec_s,
                "elapsed_s": elapsed_s,
                "context_noise": cn,
                "context_profile": prof,
                **metrics,
            }

        summary["part_b_stack"] = pat._write_grid_video(
            out_dir,
            "bucket2_seam_stacked",
            part_b_videos,
            cols=2,
            rows=1,
            audio_source_mp4=audio_ref_song,
        )

    _flush_prompts()
    saved_summary_file.write_text(
        json.dumps(pat._scrub_paths(summary, out_root, in_root), indent=2),
        encoding="utf-8",
    )
    print(
        f"[complete] Wrote $H3_OUTPUT_DIR/Video/{args.out_subdir} and data/{args.out_subdir}/probe_summary.json",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate all graphs offline (default)")
    parser.add_argument("--run", action="store_true", help="Submit graphs sequentially to live :8188 ComfyUI")
    parser.add_argument("--resume", action="store_true", help="Reuse matching existing renders in --out-subdir")
    parser.add_argument("--only", default="all", choices=("a", "b", "all"), help="Which probe subset to run")
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--url", default=comfy_client.DEFAULT_COMFY_URL)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.run:
        return run_live(args)
    return verify_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
