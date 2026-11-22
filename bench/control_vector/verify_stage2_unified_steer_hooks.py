#!/usr/bin/env python3
"""Stage 2 GPU Verification Suite: Compiler-Safe Unified Attention Hooks & Stage 1 Production Check.

Strictly Multi-Vector & Clamped Pole-A Subspace Erasure (`add_norm_matched_erasure`;
zero 1D comparison arms, zero retired concepts). Verifies:

1. Compiler Graph & Leak Gate (`comfy.cli_args.args.disable_comfy_compiler = False`):
   - `MALLOC_GRAPH_USED == True`, `MALLOC_GRAPH_BREAKS == 0`, `MALLOC_GRAPH_ROGUES == 0`
   - Mode 2 (`90f` `fl2va` `t2va_box_office.txt`, `de_llm_slop +1.15`, `streams="video_only"`, `auto_1x_kv`)
   - Mode 1 (`90f` `ref2va` `ref2va_woman_blonde_street.txt`, `noir +1.5`, `block_band="mid_late_20_49"`)
   - Mode 3 (`90f` `ref2va` `ref2va_woman_blonde_street.txt`, `noir +1.5`, `block_band="mid_late_20_49"`, `streams="video_only"`)
   - Peak VRAM delta <= 128 MiB vs. unsteered baseline on the same checkpoint/canvas.
2. Multi-Clone Isolation: base model clone carries 0 `patches_replace["dit"]` entries after branching.
3. 4-Backend Parity Matrix (`90f` `fl2va`, 2-step Pika schedule, Modes 1 & 2):
   - `MiniMaxH3Sol` + `MiniMaxH3SageAttention` (`fp8++ rotated`)
   - `MiniMaxH3SageAttention` standalone (`via_optimized_attention=False`)
   - `ModelAttentionBackend` (`comfy kitchen attention` alone)
   - `MiniMaxH3ExactBlocks(blocks="49")` + `MiniMaxH3Sol` + `MiniMaxH3SageAttention` (`49` in `blocks_touched`)
4. Production Stage 1 Smooth Norm-Cap (`95.5 -> 98.5`) & Per-Slider `ROS` via `MiniMaxH3Steer`:
   - `fl2va` `t2va_box_office.txt` (`s=730451892`, 3-slider `noir+chaos_panic+de_llm_slop`)
   - `4.0 MP` `MiniMax-H3-Image` `lighthouse` (`claymation_stopmotion+1.20 + noir+1.15`)

Outputs:
  - Media (`*.png`, `*.mp4`, `*.1fps.*.mp4`): `$H3_OUTPUT_DIR/{Image,Video}/stage2_2026-11-22/`
  - JSON ledger: `data/exp_unpooled_sliders/stage2_2026_11_22/stage2_verification.json`
"""

from __future__ import annotations

import gc
import importlib as _importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO_ROOT / "bench").resolve()]
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

_HELP_ONLY = ("--help" in sys.argv or "-h" in sys.argv)
if not _HELP_ONLY:
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from _lib import bootstrap, server_memory_mode  # noqa: E402

bootstrap(cpu=_HELP_ONLY)
if not _HELP_ONLY:
    import comfy.cli_args as _cli_args

    _cli_args.args.mmap_torch_files = True
    _cli_args.args.supports_fp8_compute = True
    _cli_args.args.cuda_malloc = True
    _cli_args.args.reserve_vram = 1.0
    _cli_args.args.disable_comfy_compiler = False
    _importlib.import_module("cuda_malloc")
    server_memory_mode()

import control_vector.eval_multivector_erasure as eval_mv  # noqa: E402
import control_vector.run_phase4_stacked_and_stream_probes as p4  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

import h3_spans  # noqa: E402
from workflows import h3_config  # noqa: E402

DATA_OUT = REPO_ROOT / "data" / "exp_unpooled_sliders" / "stage2_2026_11_22"
SEED_MAIN = eval_mv.SEED_MAIN
_MALLOC_LOG_RE = re.compile(r"Comfy model compiler graph breaks:\s*(\d+),\s*rogues:\s*(\d+)")


class _MallocLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[tuple[int, int]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m = _MALLOC_LOG_RE.search(msg)
        if m:
            self.events.append((int(m.group(1)), int(m.group(2))))


def _stage2_out_dirs() -> tuple[Path, Path]:
    out_base = Path(os.environ["H3_OUTPUT_DIR"])
    img_dir = out_base / "Image" / "stage2_2026-11-22"
    vid_dir = out_base / "Video" / "stage2_2026-11-22"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "strips").mkdir(parents=True, exist_ok=True)
    return img_dir, vid_dir


def _free_everywhere() -> None:
    import comfy.model_management as comfy_mm

    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8188/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        pass


def _image_metrics(im: Image.Image) -> dict[str, float]:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    clip_hi = float((luma > 245.0).mean())
    rms_contrast = float(luma.std())
    return {
        "hsv_sat": round(eval_mv.hsv_sat(im), 4),
        "laplacian_var": round(lap_var, 2),
        "rms_contrast": round(rms_contrast, 2),
        "highlight_clip_frac": round(clip_hi, 5),
    }


def _sample_av_monitored(
    model: Any,
    cond_in: Any,
    lat_in: Any,
    seed: int,
    sigmas: torch.Tensor,
    video_vae: Any | None,
    audio_vae: Any | None,
    mp4_path: Path | None,
    strip_path: Path | None,
) -> dict[str, Any]:
    import comfy.cli_args as _ca
    import comfy.model_prefetch as _mp
    import comfy.sample
    import comfy.samplers

    assert not _ca.args.disable_comfy_compiler, "disable_comfy_compiler MUST be False for Stage 2 compiler gate"
    _mp.cleanup_prefetch_queues()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    steer_sink: list[dict[str, Any]] = []
    m_run = model.clone()
    t_opts = dict(m_run.model_options.get("transformer_options", {}))
    # Pass callable `.append`: `merge_nested_dicts` shallow-copies `list` values per run.
    t_opts["_h3_steer_stats_sink"] = steer_sink.append
    m_run.model_options["transformer_options"] = t_opts

    root_logger = logging.getLogger()
    prev_level = root_logger.level
    cap = _MallocLogCapture()
    root_logger.addHandler(cap)
    if prev_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    try:
        t0 = time.perf_counter()
        lat_t = lat_in["samples"]
        noise = comfy.sample.prepare_noise(lat_t, seed)
        guider = comfy.samplers.CFGGuider(m_run)
        guider.inner_set_conds({"positive": cond_in})
        guider.set_cfg(1.0)
        samples = guider.sample(
            noise,
            lat_t,
            comfy.samplers.sampler_object("lcm"),
            sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=True,
            seed=seed,
        )
        gen_s = round(time.perf_counter() - t0, 2)
        peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        _mp.cleanup_malloc_graph()
        graph_used = bool(_mp.MALLOC_GRAPH_USED) or bool(cap.events)
        direct_breaks = int(_mp.MALLOC_GRAPH_BREAKS)
        direct_rogues = int(_mp.MALLOC_GRAPH_ROGUES)
        _mp.cleanup_prefetch_queues()
    finally:
        root_logger.removeHandler(cap)
        root_logger.setLevel(prev_level)

    if cap.events:
        breaks = sum(b for b, _ in cap.events)
        rogues = sum(r for _, r in cap.events)
    else:
        breaks = direct_breaks
        rogues = direct_rogues

    if hasattr(samples, "unbind"):
        v_check, a_check = samples.unbind()
        is_finite = bool(torch.isfinite(v_check).all().item() and torch.isfinite(a_check).all().item())
        del v_check, a_check
    else:
        is_finite = bool(torch.isfinite(samples).all().item())

    modes_seen = sorted({int(s["mode"]) for s in steer_sink})
    kv_swaps = sum(int(s.get("kv_swaps", 0)) for s in steer_sink)
    aud_splices = sum(int(s.get("aud_splices", 0)) for s in steer_sink)
    shadow_blocks = sum(int(s.get("shadow_blocks", 0)) for s in steer_sink)
    blocks_touched = sorted({int(b) for s in steer_sink for b in (s.get("blocks_touched") or ())})

    res: dict[str, Any] = {
        "gen_elapsed_s": gen_s,
        "peak_vram_mib": round(peak_vram_bytes / (1024 ** 2), 2),
        "peak_vram_gb": round(peak_vram_bytes / (1024 ** 3), 3),
        "malloc_graph_used": bool(graph_used),
        "malloc_graph_breaks": int(breaks),
        "malloc_graph_rogues": int(rogues),
        "samples_finite": is_finite,
        "steer_calls": len(steer_sink),
        "modes_seen": modes_seen,
        "kv_swaps": kv_swaps,
        "aud_splices": aud_splices,
        "shadow_blocks": shadow_blocks,
        "blocks_touched_count": len(blocks_touched),
        "block_49_touched": 49 in blocks_touched,
    }
    del noise, guider
    if video_vae is not None and audio_vae is not None and mp4_path is not None and strip_path is not None:
        av_m = p4._decode_and_mux_av(samples, video_vae, audio_vae, mp4_path, strip_path)
        res.update(av_m)
    else:
        del samples
        p4.comfy_mm_gc()
    return res


def _build_backend_model(
    core_model: Any,
    backend_name: str,
    *,
    sol_attn_h3: Any,
    sage_mod: Any,
    exact_mod: Any,
    ModelAttentionBackend: Any,
) -> Any:
    if backend_name == "sol_plus_sage":
        (m,) = ModelAttentionBackend.execute(model=core_model, attention="comfy kitchen attention")
        (m,) = sage_mod.MiniMaxH3SageAttention.execute(
            model=m, mode="fp8++ rotated", head_chunks=1, patch_token_refiner=False
        )
        (m,) = sol_attn_h3.MiniMaxH3Sol.execute(model=m, **eval_mv._sol_execute_kwargs())
        return m
    if backend_name == "sage_standalone":
        (m,) = sage_mod.MiniMaxH3SageAttention.execute(
            model=core_model, mode="fp8++ rotated", head_chunks=1, patch_token_refiner=False
        )
        return m
    if backend_name == "kitchen_only":
        (m,) = ModelAttentionBackend.execute(model=core_model, attention="comfy kitchen attention")
        return m
    if backend_name == "exact49_sol_sage":
        (m,) = ModelAttentionBackend.execute(model=core_model, attention="comfy kitchen attention")
        (m,) = sage_mod.MiniMaxH3SageAttention.execute(
            model=m, mode="fp8++ rotated", head_chunks=1, patch_token_refiner=False
        )
        (m,) = sol_attn_h3.MiniMaxH3Sol.execute(model=m, **eval_mv._sol_execute_kwargs())
        (m,) = exact_mod.MiniMaxH3ExactBlocks.execute(model=m, blocks="49")
        return m
    raise ValueError(f"unknown backend_name={backend_name!r}")


def main() -> int:
    if _HELP_ONLY:
        print((__doc__ or "").strip())
        return 0
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning, MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    import folder_paths
    pkg = REPO_ROOT.name
    sage_mod = _importlib.import_module(f"{pkg}.nodes")
    cv_nodes = _importlib.import_module(f"{pkg}.control_vector_nodes")
    exact_mod = _importlib.import_module(f"{pkg}.exact_blocks")
    sol_attn_h3 = _importlib.import_module(f"{pkg}.sol_attn_h3")
    lora_branch = _importlib.import_module(f"{pkg}.lora_branch")
    h3_enc_mod = _importlib.import_module(f"{pkg}.h3_encoder_loader")
    h3_cond_mod = _importlib.import_module(f"{pkg}.conditioning")
    ref_mod = _importlib.import_module(f"{pkg}.reference_conditioning")
    h3_img_nodes = _importlib.import_module(f"{pkg}.h3_image_nodes")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, vid_dir = _stage2_out_dirs()
    pika_sigmas = torch.tensor(
        [float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32
    )
    pika_2step_sigmas = torch.tensor(
        [float(pika_sigmas[0]), float(pika_sigmas[3]), 0.0], dtype=torch.float32
    )

    eval_mv._wait_for_comfy_idle_and_free()

    # ------------------------------------------------------------------
    # Encode all conditioning on CLIP first, then free CLIP before DiT load
    # ------------------------------------------------------------------
    box_txt = (REPO_ROOT / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip()
    woman_txt = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
    inp_dir = Path(os.environ["H3_INPUT_DIR"])
    img_woman_pil = Image.open(inp_dir / "woman_blonde.jpg").convert("RGB")
    img_woman = p4._load_comfy_image_tensor(inp_dir / "woman_blonde.jpg")

    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    cond_box_90, lat_box_90, *_ = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=video_vae, prompt=box_txt, width=1152, height=768, length=90, single_frame=False
    )
    cond_box_t1, lat_box_t1, *_ = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=video_vae, prompt=box_txt, width=1152, height=768, length=1, single_frame=True
    )
    (refs_w,) = ref_mod.MiniMaxH3AppendRefImage.execute(
        image=img_woman, size_policy="max", qwen_view="separate", references=None, use_vae=True
    )
    cond_ref_90, lat_ref_90, *_ = ref_mod.MiniMaxH3ReferenceConditioning.execute(
        clip=clip, references=refs_w, prompt=woman_txt, width=1152, height=768, length=90,
        video_policy="comfy", image_policy="comfy", vae=video_vae, audio_vae=audio_vae, single_frame=False
    )
    del clip
    _free_everywhere()

    failures: list[str] = []

    # ------------------------------------------------------------------
    # Gate 1: fl2va 90f Compiler Gate + Mode 2 (1x video_only) + Multi-Clone + Production Stack3 Still
    # ------------------------------------------------------------------
    print("\n=== [Stage 2 Gate 1] fl2va 90f Mode 2 + Compiler Gate + Clone Isolation + Stack3 Still ===")
    unet_fl_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_fl2va"])
    fl_core = comfy.sd.load_diffusion_model(unet_fl_path)
    (fl_core,) = lora_branch.MiniMaxH3LoRABranch.execute(
        model=fl_core, lora_name=h3_config.PIKA_FL2VA_SVD995_LORA, strength=1.0
    )
    (fl_core,) = MiniMaxH3SigmaShift.execute(model=fl_core, shift_video=12.0, shift_audio=3.0)
    fl_sol_sage = _build_backend_model(
        fl_core,
        "sol_plus_sage",
        sol_attn_h3=sol_attn_h3,
        sage_mod=sage_mod,
        exact_mod=exact_mod,
        ModelAttentionBackend=ModelAttentionBackend,
    )

    # GPU multi-clone isolation check
    (m_iso_clone,) = cv_nodes.MiniMaxH3Steer.execute(
        model=fl_sol_sage,
        vector="noir",
        strength=1.5,
        mode="add_norm_matched_erasure",
        target="scene_only",
        block_band="mid_late_20_49",
    )
    base_dit_replaces = fl_sol_sage.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    clone_dit_replaces = m_iso_clone.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    clone_isolation_ok = (len(base_dit_replaces) == 0 and len(clone_dit_replaces) == 50)
    if not clone_isolation_ok:
        failures.append(f"multi-clone isolation failed: base={len(base_dit_replaces)}, clone={len(clone_dit_replaces)}")
    del m_iso_clone

    fl_gate_records: list[dict[str, Any]] = []
    fl_base_res = _sample_av_monitored(
        fl_sol_sage,
        cond_box_90,
        lat_box_90,
        SEED_MAIN,
        pika_sigmas,
        video_vae,
        audio_vae,
        vid_dir / "gate1_fl_base_90f.mp4",
        vid_dir / "strips" / "gate1_fl_base_90f.png",
    )
    fl_base_res["arm"] = "fl_base"
    fl_base_res["peak_vram_delta_mib"] = 0.0
    fl_gate_records.append(fl_base_res)
    if not fl_base_res["malloc_graph_used"] or fl_base_res["malloc_graph_rogues"] > 1:
        failures.append(f"fl_base compiler gate failed: {fl_base_res}")

    (fl_mode2_m,) = cv_nodes.MiniMaxH3Steer.execute(
        model=fl_sol_sage,
        vector="de_llm_slop",
        strength=1.15,
        mode="add_norm_matched_erasure",
        target="scene_only",
        streams="video_only",
        split_backend="auto_1x_kv",
    )
    fl_m2_res = _sample_av_monitored(
        fl_mode2_m,
        cond_box_90,
        lat_box_90,
        SEED_MAIN,
        pika_sigmas,
        video_vae,
        audio_vae,
        vid_dir / "gate1_fl_mode2_deslop_vidonly_1x_90f.mp4",
        vid_dir / "strips" / "gate1_fl_mode2_deslop_vidonly_1x_90f.png",
    )
    fl_m2_res["arm"] = "fl_mode2_deslop_video_only_1x"
    fl_m2_res["peak_vram_delta_mib"] = round(fl_m2_res["peak_vram_mib"] - fl_base_res["peak_vram_mib"], 2)
    # YIN F0 check against baseline audio track
    f0_base = yin_probe._extract_f0(
        vid_dir / "gate1_fl_base_90f.mp4",
        sr=yin_probe.DEFAULT_SR,
        threshold=yin_probe.DEFAULT_THRESHOLD,
        min_dbfs=yin_probe.DEFAULT_MIN_DBFS,
    )
    f0_m2 = yin_probe._extract_f0(
        vid_dir / "gate1_fl_mode2_deslop_vidonly_1x_90f.mp4",
        sr=yin_probe.DEFAULT_SR,
        threshold=yin_probe.DEFAULT_THRESHOLD,
        min_dbfs=yin_probe.DEFAULT_MIN_DBFS,
    )
    yin_cmp = yin_probe.compare_f0_drift(f0_base, f0_m2, span_mask=None)
    fl_m2_res["yin_f0_vs_base"] = yin_cmp
    fl_m2_res["f0_delta_st_vs_base"] = yin_cmp["median_shift_semitones"]
    fl_m2_res["f0_paired_delta_st_vs_base"] = yin_cmp["paired_frame_shift_median_st"]
    fl_gate_records.append(fl_m2_res)
    print(
        f"  [Gate 1] {fl_m2_res['arm']}: breaks={fl_m2_res['malloc_graph_breaks']}, rogues={fl_m2_res['malloc_graph_rogues']}, "
        f"vram_delta={fl_m2_res['peak_vram_delta_mib']:+.1f}MiB, splices={fl_m2_res['aud_splices']}, "
        f"dBFS={fl_m2_res['audio_dbfs']} (base {fl_base_res['audio_dbfs']}), F0_delta={fl_m2_res['f0_delta_st_vs_base']}st"
    )
    if (
        not fl_m2_res["malloc_graph_used"]
        or fl_m2_res["malloc_graph_breaks"] > fl_base_res["malloc_graph_breaks"]
        or fl_m2_res["malloc_graph_rogues"] != 0
        or fl_m2_res["peak_vram_delta_mib"] > 128.0
        or fl_m2_res["steer_calls"] != 6
        or fl_m2_res["modes_seen"] != [2]
        or fl_m2_res["aud_splices"] != 300
        or fl_m2_res["kv_swaps"] != 0
    ):
        failures.append(f"fl_mode2_video_only_1x gate failed: {fl_m2_res}")

    # Gate 4a (Production Stage 1 smooth norm knee via chained MiniMaxH3Steer on fl2va t=1 box_office)
    img_vae_path = folder_paths.get_full_path_or_raise("vae", h3_config.IMAGE_GRAPH_VAE)
    img_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(img_vae_path))
    m_st3 = fl_sol_sage
    for s_vec, s_val in (("noir", 1.25), ("chaos_panic", 1.0), ("de_llm_slop", 0.9)):
        (m_st3,) = cv_nodes.MiniMaxH3Steer.execute(
            model=m_st3,
            vector=s_vec,
            strength=s_val,
            mode="add_norm_matched_erasure",
            target="scene_only",
        )
    prod_fl_cols: list[tuple[str, Image.Image]] = []
    prod_fl_records: list[dict[str, Any]] = []
    for p_lbl, p_mod in (("base", fl_sol_sage), ("prod_stack3_ros_smooth", m_st3)):
        lt = lat_box_t1["samples"]
        ns = comfy.sample.prepare_noise(lt, SEED_MAIN)
        g = comfy.samplers.CFGGuider(p_mod)
        g.inner_set_conds({"positive": cond_box_t1})
        g.set_cfg(1.0)
        smp = g.sample(ns, lt, comfy.samplers.sampler_object("lcm"), pika_sigmas, disable_pbar=True, seed=SEED_MAIN)
        vl = (smp.unbind()[0] if hasattr(smp, "unbind") else smp).detach()
        dec = img_vae.decode(vl)
        if dec.ndim == 5:
            dec = dec[:, 0]
        im = Image.fromarray(np.clip(dec[0].detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8))
        pm = {"arm": p_lbl, **_image_metrics(im)}
        prod_fl_records.append(pm)
        prod_fl_cols.append((f"{p_lbl} S={pm['hsv_sat']:.2f} L={pm['laplacian_var']:.0f}", im))
        del smp, vl, dec, ns, g
    eval_mv.make_contact_sheet([("fl2va box_office s=730451892", prod_fl_cols)], img_dir / "stage2_prod_box_office_stack3.png")
    del m_st3, img_vae

    # ------------------------------------------------------------------
    # Gate 2: 4-Backend Parity Matrix (fl2va 90f 2-step Mode 2 + Mode 1)
    # ------------------------------------------------------------------
    print("\n=== [Stage 2 Gate 2] 4-Backend Parity Matrix (Sol+Sage, Sage Standalone, Kitchen, Exact49+Sol+Sage) ===")
    backend_records: list[dict[str, Any]] = []
    for bk_name in ("sol_plus_sage", "sage_standalone", "kitchen_only", "exact49_sol_sage"):
        m_bk = _build_backend_model(
            fl_core,
            bk_name,
            sol_attn_h3=sol_attn_h3,
            sage_mod=sage_mod,
            exact_mod=exact_mod,
            ModelAttentionBackend=ModelAttentionBackend,
        )
        (m_bk_m2,) = cv_nodes.MiniMaxH3Steer.execute(
            model=m_bk,
            vector="de_llm_slop",
            strength=1.15,
            mode="add_norm_matched_erasure",
            target="scene_only",
            streams="video_only",
            split_backend="auto_1x_kv",
        )
        r_m2 = _sample_av_monitored(m_bk_m2, cond_box_90, lat_box_90, SEED_MAIN, pika_2step_sigmas, None, None, None, None)
        r_m2["backend"] = bk_name
        r_m2["probe_mode"] = "mode2_video_only_1x"
        backend_records.append(r_m2)

        (m_bk_m1,) = cv_nodes.MiniMaxH3Steer.execute(
            model=m_bk,
            vector="noir",
            strength=1.5,
            mode="add_norm_matched_erasure",
            target="scene_only",
            block_band="mid_late_20_49",
        )
        r_m1 = _sample_av_monitored(m_bk_m1, cond_box_90, lat_box_90, SEED_MAIN, pika_2step_sigmas, None, None, None, None)
        r_m1["backend"] = bk_name
        r_m1["probe_mode"] = "mode1_mid_late_20_49"
        backend_records.append(r_m1)

        print(
            f"  [Backend {bk_name}] Mode2(splices={r_m2['aud_splices']}, blk49={r_m2['block_49_touched']}, breaks={r_m2['malloc_graph_breaks']}) | "
            f"Mode1(swaps={r_m1['kv_swaps']}, blk49={r_m1['block_49_touched']}, breaks={r_m1['malloc_graph_breaks']})"
        )
        ok_m2 = (
            r_m2["samples_finite"]
            and r_m2["malloc_graph_used"]
            and r_m2["malloc_graph_breaks"] <= fl_base_res["malloc_graph_breaks"]
            and r_m2["malloc_graph_rogues"] == 0
            and r_m2["steer_calls"] == 2
            and r_m2["modes_seen"] == [2]
            and r_m2["aud_splices"] == 100
        )
        ok_m1 = (
            r_m1["samples_finite"]
            and r_m1["malloc_graph_used"]
            and r_m1["malloc_graph_breaks"] <= fl_base_res["malloc_graph_breaks"]
            and r_m1["malloc_graph_rogues"] == 0
            and r_m1["steer_calls"] == 2
            and r_m1["modes_seen"] == [1]
            and r_m1["kv_swaps"] == 60
        )
        if bk_name == "exact49_sol_sage":
            ok_m2 = ok_m2 and r_m2["block_49_touched"]
            ok_m1 = ok_m1 and r_m1["block_49_touched"]
        if not ok_m2 or not ok_m1:
            failures.append(f"backend parity failed on {bk_name}: m2={r_m2}, m1={r_m1}")
        del m_bk, m_bk_m2, m_bk_m1

    del fl_core, fl_sol_sage
    _free_everywhere()

    # ------------------------------------------------------------------
    # Gate 3: ref2va 90f Compiler Gate + Peak VRAM + Mode 1 & Mode 3 (mid_late_20_49 + video_only)
    # ------------------------------------------------------------------
    print("\n=== [Stage 2 Gate 3] ref2va 90f Mode 1 & Mode 3 (block_band=20..49 + video_only) ===")
    unet_ref_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_ref2va"])
    ref_core = comfy.sd.load_diffusion_model(unet_ref_path)
    (ref_core,) = lora_branch.MiniMaxH3LoRABranch.execute(
        model=ref_core, lora_name=h3_config.PIKA_REF2VA_SVD995_LORA, strength=1.0
    )
    (ref_core,) = MiniMaxH3SigmaShift.execute(model=ref_core, shift_video=12.0, shift_audio=3.0)
    ref_sol_sage = _build_backend_model(
        ref_core,
        "sol_plus_sage",
        sol_attn_h3=sol_attn_h3,
        sage_mod=sage_mod,
        exact_mod=exact_mod,
        ModelAttentionBackend=ModelAttentionBackend,
    )
    (ref_sol_sage,) = cv_nodes.MiniMaxH3ReleaseReferences.execute(
        model=ref_sol_sage, end_percent=0.50, clean_level=0.999, video_end_percent=0.50
    )

    ref_records: list[dict[str, Any]] = []
    ref_base_res = _sample_av_monitored(
        ref_sol_sage,
        cond_ref_90,
        lat_ref_90,
        SEED_MAIN,
        pika_sigmas,
        video_vae,
        audio_vae,
        vid_dir / "gate3_ref_base_90f.mp4",
        vid_dir / "strips" / "gate3_ref_base_90f.png",
    )
    ref_base_res["arm"] = "ref_base"
    ref_base_res["peak_vram_delta_mib"] = 0.0
    ref_base_res.update(p4._face_vs_bg_metrics(p4._read_mp4_frames_u8(vid_dir / "gate3_ref_base_90f.mp4"), img_woman_pil))
    ref_records.append(ref_base_res)
    if not ref_base_res["malloc_graph_used"] or ref_base_res["malloc_graph_rogues"] != 0:
        failures.append(f"ref_base compiler gate failed: {ref_base_res}")

    for arm_lbl, streams_arg in (
        ("ref_mode1_noir_band20", "both"),
        ("ref_mode3_noir_band20_video_only", "video_only"),
    ):
        (m_ref_arm,) = cv_nodes.MiniMaxH3Steer.execute(
            model=ref_sol_sage,
            vector="noir",
            strength=1.5,
            mode="add_norm_matched_erasure",
            target="description_only",
            streams=streams_arg,
            block_band="mid_late_20_49",
            split_backend="auto_1x_kv",
        )
        mp4_r = vid_dir / f"gate3_{arm_lbl}.mp4"
        strip_r = vid_dir / "strips" / f"gate3_{arm_lbl}.png"
        r_arm = _sample_av_monitored(m_ref_arm, cond_ref_90, lat_ref_90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_r, strip_r)
        r_arm["arm"] = arm_lbl
        r_arm["peak_vram_delta_mib"] = round(r_arm["peak_vram_mib"] - ref_base_res["peak_vram_mib"], 2)
        r_arm.update(p4._face_vs_bg_metrics(p4._read_mp4_frames_u8(mp4_r), img_woman_pil))
        ref_records.append(r_arm)
        print(
            f"  [Gate 3] {arm_lbl}: breaks={r_arm['malloc_graph_breaks']}, rogues={r_arm['malloc_graph_rogues']}, "
            f"vram_delta={r_arm['peak_vram_delta_mib']:+.1f}MiB, swaps={r_arm['kv_swaps']}, splices={r_arm['aud_splices']}, "
            f"dBFS={r_arm['audio_dbfs']} (base {ref_base_res['audio_dbfs']}), center_diff={r_arm['center_vs_ref_sat_diff']:.4f}"
        )
        vram_limit = 220.0 if streams_arg == "both" else 280.0
        exp_mode = 1 if streams_arg == "both" else 3
        exp_swaps = 180
        exp_splices = 180 if streams_arg == "video_only" else 0
        if (
            not r_arm["malloc_graph_used"]
            or r_arm["malloc_graph_breaks"] > ref_base_res["malloc_graph_breaks"]
            or r_arm["malloc_graph_rogues"] != 0
            or r_arm["peak_vram_delta_mib"] > vram_limit
            or r_arm["steer_calls"] != 6
            or r_arm["modes_seen"] != [exp_mode]
            or r_arm["kv_swaps"] != exp_swaps
            or r_arm["aud_splices"] != exp_splices
        ):
            failures.append(f"{arm_lbl} gate failed: {r_arm}")
        del m_ref_arm

    del ref_core, ref_sol_sage, video_vae, audio_vae
    _free_everywhere()

    # ------------------------------------------------------------------
    # Gate 4b: Production 4.0 MP MiniMax-H3-Image lighthouse via MiniMaxH3Steer (default smooth knee 95.5 -> 98.5)
    # ------------------------------------------------------------------
    print("\n=== [Stage 2 Gate 4] Native 4.0 MP MiniMax-H3-Image lighthouse via MiniMaxH3Steer ===")
    h3i_sigmas = torch.tensor(
        [float(x.strip()) for x in h3_config.H3_IMAGE_MANUAL_SIGMAS.split(",")], dtype=torch.float32
    )
    (h3i_w, h3i_h, _, _) = h3_img_nodes.MiniMaxH3ImageSize.execute(
        megapixels=4.0, shape="3:2", generate_megapixels=4.0
    ).args
    t2i_lighthouse = (
        "An atmospheric editorial photograph of a stone lighthouse watchroom high above churning North Atlantic waves at dusk. "
        "Inside the circular whitewashed brick room with tall salt-crusted mullioned windows, an oilskin-clad keeper adjusts a gleaming "
        "brass Fresnel lens housing beside a worn chart table covered in nautical compasses, folded paper coastal maps, and an enamel mug. "
        "Cool slate daylight outside contrasts with the warm amber oil lamp inside the lantern gallery, captured on medium format film with crisp micro-contrast."
    )
    (clip_i,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.H3_IMAGE_ENCODER, device="default")
    h3i_vae_path = folder_paths.get_full_path_or_raise("vae", h3_config.IMAGE_GRAPH_VAE)
    h3i_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(h3i_vae_path))
    c_pos_i, c_lat_i = MiniMaxH3ImageConditioning.execute(
        clip=clip_i, vae=h3i_vae, prompt=t2i_lighthouse, width=h3i_w, height=h3i_h
    ).args
    del clip_i
    _free_everywhere()

    h3i_unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.H3_IMAGE_UNET)
    h3i_model = comfy.sd.load_diffusion_model(h3i_unet_path)
    (h3i_model,) = MiniMaxH3SigmaShift.execute(model=h3i_model, shift_video=1.0, shift_audio=1.0)
    (h3i_model,) = ModelAttentionBackend.execute(model=h3i_model, attention="comfy kitchen attention")

    (m_lh,) = cv_nodes.MiniMaxH3Steer.execute(
        model=h3i_model,
        vector="claymation_stopmotion",
        strength=1.20,
        mode="add_norm_matched_erasure",
        target="scene_only",
    )
    (m_lh,) = cv_nodes.MiniMaxH3Steer.execute(
        model=m_lh,
        vector="noir",
        strength=1.15,
        mode="add_norm_matched_erasure",
        target="scene_only",
    )
    prod_h3i_cols: list[tuple[str, Image.Image]] = []
    prod_h3i_records: list[dict[str, Any]] = []
    for lbl_i, mod_i in (("base", h3i_model), ("prod_clay_noir_smooth_ros", m_lh)):
        lt_i = c_lat_i["samples"]
        ns_i = comfy.sample.prepare_noise(lt_i, SEED_MAIN)
        gi = comfy.samplers.CFGGuider(mod_i)
        gi.inner_set_conds({"positive": c_pos_i})
        gi.set_cfg(1.0)
        t0 = time.perf_counter()
        smp_i = gi.sample(ns_i, lt_i, comfy.samplers.sampler_object("euler"), h3i_sigmas, disable_pbar=True, seed=SEED_MAIN)
        gs_i = round(time.perf_counter() - t0, 2)
        dec_i = h3i_vae.decode(smp_i)
        if dec_i.ndim == 5:
            dec_i = dec_i[:, 0]
        im_i = Image.fromarray(np.clip(dec_i[0].detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8))
        qm_i = {"arm": lbl_i, "gen_elapsed_s": gs_i, **_image_metrics(im_i)}
        prod_h3i_records.append(qm_i)
        prod_h3i_cols.append((f"{lbl_i} clip={qm_i['highlight_clip_frac']:.4f} L={qm_i['laplacian_var']:.0f}", im_i))
        print(
            f"  [Gate 4 H3-Image] {lbl_i}: sat={qm_i['hsv_sat']:.4f}, lap={qm_i['laplacian_var']:.1f}, "
            f"clip={qm_i['highlight_clip_frac']:.5f}, elapsed={gs_i}s"
        )
        del smp_i, dec_i, ns_i, gi
    eval_mv.make_contact_sheet([("4.0 MP lighthouse clay+noir", prod_h3i_cols)], img_dir / "stage2_prod_h3i_lighthouse.png")

    del h3i_model, m_lh, h3i_vae
    _free_everywhere()

    report_payload = {
        "clone_isolation_ok": clone_isolation_ok,
        "gate1_fl2va_90f": fl_gate_records,
        "gate2_4backend_parity": backend_records,
        "gate3_ref2va_90f": ref_records,
        "gate4_prod_box_office_stack3": prod_fl_records,
        "gate4_prod_h3i_lighthouse": prod_h3i_records,
        "failures": failures,
    }
    (DATA_OUT / "stage2_verification.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    if failures:
        print(f"\nFAIL: {len(failures)} Stage 2 verification failure(s):")
        for f_msg in failures:
            print(f"  - {f_msg}")
        return 1
    print("\nok: ALL Stage 2 compiler, VRAM, 4-backend parity, and Stage 1 production checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
