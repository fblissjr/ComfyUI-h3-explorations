#!/usr/bin/env python3
"""CPU-only verification for `bench/index_steering_renders.py`.

Creates synthetic PNGs with embedded ComfyUI API graphs covering:
1. 1D Control Vector (`MiniMaxH3Steer`) with low-`mpc` vector metadata & baked holds
2. `heylook` Edit Guidance (`MiniMaxH3EditGuider`) on `ref2va` with `first_step=1`,
   `ref_end_percent=1.0`, `sudden_midclip` zero-prefix mask, locked soundscape,
   identical `retention_analysis`, and stale baseline phrase retention
3. `fast_1x` short-circuit (`EditGuider` with `scale=1.0, first_step=0`, no mask)
   and 2-subject `MiniMaxH3AppendRefImage` chain
4. `--since` mtime filtering and SQLite views (`v_problem_summary`, `v_renders_flat`, `v_vector_health`)
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from _lib import case, finish  # noqa: E402
import index_steering_renders as idx  # noqa: E402


def _write_png_with_graph(path: Path, graph: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (8, 8), (12, 34, 56))
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", json.dumps(graph))
    im.save(path, pnginfo=info)


def _write_fake_safetensors(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    str_meta = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in meta.items()}
    header = json.dumps({"__metadata__": str_meta}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def _graph_cv(vec_name: str, *, target: str = "look_only", ref2va: bool = True) -> dict:
    g: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3_ref2va.safetensors" if ref2va else "h3_fl2va.safetensors"}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["9", 0], "noise": ["11", 0], "sigmas": ["12", 0]}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "12": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0"}},
        "115": {
            "class_type": "MiniMaxH3Steer",
            "inputs": {
                "model": ["1", 0],
                "vector": vec_name,
                "strength": 1.2,
                "mode": "add_norm_matched",
                "target": target,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "streams": "both",
            },
        },
        "26": {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0]}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["115", 0], "conditioning": ["26", 0]}},
    }
    if ref2va:
        g["15"] = {"class_type": "LoadImage", "inputs": {"image": "subj_a.jpg"}}
        g["50"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0]}}
        g["5"] = {
            "class_type": "MiniMaxH3ReferenceConditioning",
            "inputs": {
                "prompt": "detailed_description: Short stub prompt standing beside a partner.\noverall_soundscape: Quiet room.",
                "references": ["50", 0],
                "width": 1344,
                "height": 768,
                "length": 141,
            },
        }
    else:
        g["5"] = {
            "class_type": "MiniMaxH3Conditioning",
            "inputs": {"prompt": "detailed_description: " + ("word " * 120), "width": 1344, "height": 768, "length": 141},
        }
    return g


def _graph_edit_guider(*, fast_1x: bool = False) -> dict:
    orig_prompt = (
        "subject_definitions: <Picture 1> is woman_a; <Picture 2> is man_b.\n"
        "retention_analysis: partially_preserved - identity retained.\n"
        "detailed_description: Standing in neat casual clothes inside a softly lit room.\n"
        "overall_soundscape: Quiet indoor room tone."
    )
    edit_prompt = (
        "subject_definitions: <Picture 1> is woman_a; <Picture 2> is man_b.\n"
        "retention_analysis: partially_preserved - identity retained.\n"
        "detailed_description: Reaching in neat casual clothes inside a softly lit room.\n"
        "overall_soundscape: Quiet indoor room tone."
    )
    g: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3_ref2va.safetensors"}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["9", 0], "noise": ["11", 0], "sigmas": ["12", 0]}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": 77}},
        "12": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0"}},
        "15": {"class_type": "LoadImage", "inputs": {"image": "subj_a.jpg"}},
        "16": {"class_type": "LoadImage", "inputs": {"image": "subj_b.jpg"}},
        "50": {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0]}},
        "51": {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["16", 0], "references": ["50", 0]}},
        "132": {"class_type": "MiniMaxH3EncodeReferences", "inputs": {"references": ["51", 0], "width": 1344, "height": 768, "length": 141}},
        "101": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": orig_prompt}},
        "102": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": edit_prompt}},
        "5": {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": ["101", 0]}},
        "131": {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": ["102", 0]}},
        "135": {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {"model": ["1", 0], "end_percent": 0.33 if fast_1x else 1.0, "clean_level": 0.999},
        },
        "9": {
            "class_type": "MiniMaxH3EditGuider",
            "inputs": {
                "model": ["135", 0],
                "original": ["5", 0],
                "edited": ["131", 0],
                "scale": 1.0 if fast_1x else 1.5,
                "audio_scale": 1.0,
                "first_step": 0 if fast_1x else 1,
                "last_step": 3,
                "streams": "both" if fast_1x else "independent",
            },
        },
    }
    if not fast_1x:
        g["200"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}}
        g["201"] = {"class_type": "MaskToImage", "inputs": {"mask": ["200", 0]}}
        g["202"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["201", 0], "amount": 70}}
        g["203"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 64, "height": 64}}
        g["204"] = {"class_type": "MaskToImage", "inputs": {"mask": ["203", 0]}}
        g["205"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["204", 0], "amount": 71}}
        g["206"] = {"class_type": "ImageBatch", "inputs": {"image1": ["202", 0], "image2": ["205", 0]}}
        g["207"] = {"class_type": "ImageToMask", "inputs": {"image": ["206", 0]}}
        g["9"]["inputs"]["mask"] = ["207", 0]
    return g


def _test_full_index() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out_root = tmp / "Video"
        vec_root = tmp / "vectors"
        _write_fake_safetensors(
            vec_root / "v2" / "wardrobe_weak_vec__fl2va__diff_span.safetensors",
            {
                "name": "wardrobe_weak_vec",
                "family": "fl2va",
                "points": {"post_refiner": {"raw_norm": 72.0}},
                "quality_report": {"mean_pairwise_cosine": 0.85, "split_half_cosine": 0.96},
                "pairs": {"count": 10, "n_holds": 0},
            },
        )
        _write_fake_safetensors(
            vec_root / "v2" / "wardrobe_weak_vec__ref2va__diff_span.safetensors",
            {
                "name": "wardrobe_weak_vec",
                "family": "ref2va",
                "points": {"post_refiner": {"raw_norm": 48.0}},
                "quality_report": {
                    "mean_pairwise_cosine": 0.42,
                    "split_half_cosine": 0.79,
                    "j_lens_readout": {"b_side_alignment": {"status": "warn", "warning": True}},
                },
                "pairs": {"count": 10, "n_holds": 2},
            },
        )
        _write_png_with_graph(out_root / "cv_run" / "weak_00001.png", _graph_cv("wardrobe_weak_vec"))
        _write_png_with_graph(out_root / "cv_run" / "fl2va_match_00001.png", _graph_cv("wardrobe_weak_vec", target="scene_only", ref2va=False))
        zero_g = _graph_cv("wardrobe_weak_vec", target="scene_only", ref2va=False)
        zero_g["115"]["inputs"]["strength"] = 0.0
        _write_png_with_graph(out_root / "cv_run" / "zero_strength_00001.png", zero_g)
        _write_png_with_graph(out_root / "edit_run" / "locked_00001.png", _graph_edit_guider(fast_1x=False))
        _write_png_with_graph(out_root / "edit_run" / "fast_00001.png", _graph_edit_guider(fast_1x=True))
        vid_only_g = _graph_edit_guider(fast_1x=True)
        vid_only_g["9"]["inputs"]["streams"] = "video_only"
        _write_png_with_graph(out_root / "edit_run" / "vid_only_1_0_00001.png", vid_only_g)

        # Add an old PNG that should be skipped by --since
        old_png = out_root / "cv_run" / "stale_00001.png"
        _write_png_with_graph(old_png, _graph_cv("v2/weak_vec.safetensors", ref2va=False))
        old_ts = time.mktime((2026, 1, 1, 0, 0, 0, 0, 0, 0))
        os.utime(old_png, (old_ts, old_ts))

        db_path = tmp / "audit.sqlite"
        jsonl_path = tmp / "audit.jsonl"
        rc = idx.main([
            "--root", str(out_root),
            "--dirs", "cv_run", "edit_run",
            "--since", "2026-06-01",
            "--vector-roots", str(vec_root),
            "--out-db", str(db_path),
            "--out-jsonl", str(jsonl_path),
        ])
        assert rc == 0, f"expected rc=0, got {rc}"

        lines = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 6, f"expected 6 non-stale renders, got {len(lines)}"
        by_stem = {r["stem"]: r for r in lines}

        fl_rec = by_stem["fl2va_match"]
        assert fl_rec["vectors"][0]["raw_norm"] == 72.0, f"expected fl2va vector join (72.0), got {fl_rec['vectors'][0]}"
        assert fl_rec["vectors"][0]["vector_file"].endswith("__fl2va__diff_span.safetensors")

        z_rec = by_stem["zero_strength"]
        assert z_rec["method"] == "plain" and not z_rec["vectors"], f"strength=0 should be plain, got {z_rec['method']}"

        vo_rec = by_stem["vid_only_1_0"]
        assert vo_rec["method"] == "edit_guidance", f"streams=video_only scale=1.0 must not be fast_1x, got {vo_rec['method']}"

        cv_rec = by_stem["weak"]
        assert cv_rec["method"] == "control_vector"
        assert cv_rec["family"] == "ref2va"
        assert cv_rec["ref_images"] == ["subj_a.jpg"]
        for expected_flag in (
            "ref_pinned_1_0",
            "ref_count_mismatch",
            "short_stub_prompt",
            "low_mpc_vector_inflated",
            "look_only_on_ref2va",
            "baked_holds_present",
            "j_lens_b_side_warning",
        ):
            assert expected_flag in cv_rec["problem_codes"], f"missing {expected_flag} in {cv_rec['problem_codes']}"

        ed_rec = by_stem["locked"]
        assert ed_rec["method"] == "edit_guidance"
        assert ed_rec["ref_images"] == ["subj_a.jpg", "subj_b.jpg"], ed_rec["ref_images"]
        assert ed_rec["temporal_curve"] == "sudden_midclip_70_71", ed_rec["temporal_curve"]
        assert ed_rec["zero_prefix_frames"] == 70
        for expected_flag in (
            "ref_pinned_1_0",
            "first_step_1_layout_lock",
            "sudden_midclip_zero_prefix",
            "soundscape_locked_dry",
            "retention_cancelled_zero",
            "high_prompt_similarity",
            "stale_baseline_phrase_leak",
        ):
            assert expected_flag in ed_rec["problem_codes"], f"missing {expected_flag} in {ed_rec['problem_codes']}"

        f1_rec = by_stem["fast"]
        assert f1_rec["method"] == "fast_1x"
        assert f1_rec["ref_end_percent"] == 0.33
        assert "ref_pinned_1_0" not in f1_rec["problem_codes"]
        assert "first_step_1_layout_lock" not in f1_rec["problem_codes"]

        conn = sqlite3.connect(str(db_path))
        try:
            n_flat = conn.execute("SELECT COUNT(*) FROM v_renders_flat").fetchone()[0]
            assert n_flat == 6
            n_vh = conn.execute("SELECT COUNT(*) FROM v_vector_health").fetchone()[0]
            assert n_vh == 2
            summary_rows = conn.execute("SELECT folder, method, problem_code, n_affected FROM v_problem_summary").fetchall()
            assert len(summary_rows) >= 10, summary_rows
        finally:
            conn.close()


if __name__ == "__main__":
    case("index_steering_renders end-to-end SQLite/JSONL & flags", _test_full_index)
    finish()
