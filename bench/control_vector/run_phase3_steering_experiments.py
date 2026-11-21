#!/usr/bin/env python3
"""Phase 3 Steering & Elicitation Experiments (GPU Window 1).

Stages (`--stage`):
  - `de_slop` : Stage A — `de_llm_slop` Anti-Slop Master EQ across 4 diverse non-market
                scenes (expanded via `heylook` when reachable, else prompt_bank fallback) +
                synthetic glossy prompts + 1 speaking 90f AV clip (`t2va_box_office`).
  - `exp1`    : Stage B — Surgical Erasure v2: `r=8` (`svd_ac` vs `contested_svd` vs
                `global_contested_svd`), strength ramp `s in [-1, 0, 0.5, 1.0, 1.25, 1.5]`,
                early erasure `[0..0.50]` vs `[0..1]`, and 3-slider simultaneous stack
                (`noir +1.25` + `chaos_panic +1.0` + `de_llm_slop +0.9`).
  - `exp2`    : Stage C — Same-Modality 1:1 Vision-Token Erasure (`pooling="vision_tokens"`,
                `erasure.mu_a` centering) built via `make_control_vector_from_media_pairs`
                on 10 color->greyscale pairs, tested on `woman_blonde` `ref2va` (Gear 1 vs
                Gear 2a/2b) and `fl2va_noodle` `fl2va` (`90f` on `on_both_clocks`).
  - `exp3`    : Stage D — Shadow-Text K/V Block-Band Routing (`h_txt_base` + top-level
                `optimized_attention_override` chaining under `MiniMaxH3Sol`, zero
                `patches_replace["dit"]`) on `woman_blonde` `ref2va` (`paper_origami +1.55`)
                across `[0..49]`, `[0..19]`, `[20..49]`, `[30..49]`, and `LSE` ref balance.
  - `all`     : Runs Stages A-D sequentially and unloads `:8188` cleanly on exit.

Outputs:
  - Media (`*.png`, `*.mp4`, `*.1fps.*.mp4`): `$H3_OUTPUT_DIR/{Image,Video}/phase3_2026-11-21/`
  - Metrics JSON: `data/exp_unpooled_sliders/phase3_2026_11_21/*.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO_ROOT / "bench").resolve()]
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT))

from _lib import bootstrap, server_memory_mode  # noqa: E402

_HELP_ONLY = ("--help" in sys.argv or "-h" in sys.argv)
if not _HELP_ONLY:
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
bootstrap(cpu=_HELP_ONLY)
if not _HELP_ONLY:
    import importlib as _importlib
    import comfy.cli_args as _cli_args

    _cli_args.args.mmap_torch_files = True
    _cli_args.args.supports_fp8_compute = True
    _cli_args.args.cuda_malloc = True
    _cli_args.args.reserve_vram = 1.0
    _importlib.import_module("cuda_malloc")
    server_memory_mode()

import control_vector.eval_multivector_erasure as eval_mv  # noqa: E402
from _lib.frames import OneASecond  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import attention as h3_attn  # noqa: E402
import control_vectors as cv  # noqa: E402
import h3_prompt_expander  # noqa: E402
import h3_spans  # noqa: E402
from workflows import h3_config  # noqa: E402

DATA_OUT = REPO_ROOT / "data" / "exp_unpooled_sliders" / "phase3_2026_11_21"
NORM_TARGET = eval_mv.NORM_TARGET
SEED_MAIN = eval_mv.SEED_MAIN
SEEDS_2 = eval_mv.SEEDS_3[:2]


def _phase3_out_dirs() -> tuple[Path, Path]:
    out_base = Path(os.environ["H3_OUTPUT_DIR"])
    img_dir = out_base / "Image" / "phase3_2026-11-21"
    vid_dir = out_base / "Video" / "phase3_2026-11-21"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "strips").mkdir(parents=True, exist_ok=True)
    return img_dir, vid_dir


def _comfy_bank_dir() -> Path:
    store = os.environ.get("H3_MODEL_STORE", "").strip()
    if store:
        cand = Path(store) / "control_vectors" / "bank"
        if cand.is_dir():
            return cand
    return REPO_ROOT.parent.parent / "models" / "control_vectors" / "bank"


def _free_vram_everywhere() -> None:
    import comfy.model_management as comfy_mm

    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def _write_one_a_second(vid_frames_u8: np.ndarray, stem_path: Path) -> list[dict[str, Any]]:
    n_f, h, w, _ = vid_frames_u8.shape
    writer = OneASecond(stem_path, (int(w), int(h)))
    for fi in range(n_f):
        writer.write(np.ascontiguousarray(vid_frames_u8[fi]).tobytes(), fi)
    code, parts = writer.close()
    if code != 0:
        raise RuntimeError(f"OneASecond failed ({code}) for {stem_path}")
    return parts


def _image_quality_metrics(im: Image.Image) -> dict[str, float]:
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


def _load_comfy_image_tensor(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _encode_prompt_dict(clip: Any, name: str, prompt_text: str) -> dict[str, Any]:
    import control_vector_nodes as cv_nodes

    enc_2d, tags_1d, tok_ids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, prompt_text)
    pieces = [dec_fn([int(tid)]) for tid in tok_ids]
    sp_m = h3_spans.spans(tok_ids, tags_1d.tolist(), dec_fn)
    return {
        "name": name,
        "prompt": prompt_text,
        "enc": enc_2d.detach().cpu().to(torch.float32),
        "tags": tags_1d.detach().cpu().to(torch.int64),
        "tok_ids": list(tok_ids),
        "pieces": pieces,
        "scene_mask": h3_spans.inference_mask(sp_m, "scene_only").detach().cpu(),
        "desc_mask": h3_spans.inference_mask(sp_m, "description_only").detach().cpu(),
        "sound_mask": h3_spans.inference_mask(sp_m, "soundscape_only").detach().cpu(),
    }


def _load_bank_vector_op(stem: str, family: str = "fl2va") -> dict[str, torch.Tensor]:
    import control_vector_nodes as cv_nodes

    vp = cv_nodes.resolve_vector_file_path(stem, preferred_family=family)
    vt, _ = cv.load_vector(vp)
    op = {
        "v1d": vt["vector.post_refiner"],
        "mu_d": vt.get("erasure.mu_d", vt["vector.post_refiner"]),
        "vr": vt.get("erasure.vr", torch.zeros((0, cv.POST_REFINER_DIM))),
        "diag": vt.get("erasure.diag", torch.zeros((0,))),
    }
    if "erasure.mu_a" in vt:
        op["mu_a"] = vt["erasure.mu_a"]
    return op


def _build_global_contested_variant(
    a_spans: list[torch.Tensor],
    b_spans: list[torch.Tensor],
    a_pieces_per_pair: list[list[str]] | None,
    *,
    tau: float = 0.15,
    rank: int = 8,
    ridge_rel: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Exact tau-matched `mu_d` + GLOBAL-centered skiplist/contested `Vr` (weighted by ||d||)."""
    a_pool = torch.stack([ap.double().mean(dim=0) for ap in a_spans], dim=0)
    b_pool = torch.stack([bp.double().mean(dim=0) for bp in b_spans], dim=0)
    v1d = (b_pool - a_pool).mean(dim=0)

    a_keep_rows: list[torch.Tensor] = []
    d_norms: list[torch.Tensor] = []
    d_sum = torch.zeros_like(v1d)
    n_tok = 0.0

    for p_idx, (ap_raw, bp_raw) in enumerate(zip(a_spans, b_spans)):
        ap = ap_raw.double()
        bp = bp_raw.double()
        au = ap / torch.linalg.vector_norm(ap, dim=-1, keepdim=True).clamp(min=1e-12)
        bu = bp / torch.linalg.vector_norm(bp, dim=-1, keepdim=True).clamp(min=1e-12)
        p_mat = torch.softmax((au @ bu.T) / float(tau), dim=-1)
        dp = (p_mat @ bp) - ap
        d_sum = d_sum + dp.sum(dim=0)
        n_tok += float(ap.shape[0])

        keep = torch.ones(ap.shape[0], dtype=torch.bool)
        if a_pieces_per_pair and p_idx < len(a_pieces_per_pair):
            pcs = a_pieces_per_pair[p_idx]
            if len(pcs) == ap.shape[0]:
                for ti, pc in enumerate(pcs):
                    w = pc.strip().lower().strip(".,;:!?\"'()[]{}")
                    if not w or not any(c.isalnum() for c in w) or w in eval_mv.SKIPLIST_WORDS:
                        keep[ti] = False
                if int(keep.sum().item()) < 2:
                    keep = torch.ones(ap.shape[0], dtype=torch.bool)
        a_keep_rows.append(ap[keep])
        d_norms.append(torch.linalg.vector_norm(dp[keep], dim=-1))

    mu_d = d_sum / max(n_tok, 1.0)
    a_cat = torch.cat(a_keep_rows, dim=0)
    dn_cat = torch.cat(d_norms, dim=0)
    w_row = (dn_cat / dn_cat.mean().clamp(min=1e-9)).clamp(0.25, 2.5)
    ac_glob = (a_cat - a_cat.mean(dim=0, keepdim=True)) * w_row.unsqueeze(-1)

    _u, sv, vh = torch.linalg.svd(ac_glob, full_matrices=False)
    r_eff = min(int(rank), int((sv > 1e-9 * sv[0]).sum().item()))
    vr = vh[:r_eff].contiguous()
    lam = float(ridge_rel) * float(sv[0].item() ** 2)
    diag = ((sv[:r_eff] ** 2) / (sv[:r_eff] ** 2 + lam)).clamp(0.0, 1.0).contiguous()
    return {"v1d": v1d.float(), "mu_d": mu_d.float(), "vr": vr.float(), "diag": diag.float()}


def _load_pika_model_by_family(family: str, *, with_lora: bool = True, use_sol: bool = False):
    import comfy.sd
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    import control_vector_nodes as cv_nodes
    sol_attn_h3 = cv_nodes._repo("sol_attn_h3")

    lora_branch = cv_nodes._repo("lora_branch")
    unet_key = "unet_ref2va" if family == "ref2va" else "unet_fl2va"
    lora_name = h3_config.PIKA_REF2VA_SVD995_LORA if family == "ref2va" else h3_config.PIKA_FL2VA_SVD995_LORA

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS[unet_key])
    model = comfy.sd.load_diffusion_model(unet_path)
    if with_lora:
        (model,) = lora_branch.MiniMaxH3LoRABranch.execute(model=model, lora_name=lora_name, strength=1.0)
    (model,) = MiniMaxH3SigmaShift.execute(model=model, shift_video=12.0, shift_audio=3.0)
    (model,) = ModelAttentionBackend.execute(model=model, attention="comfy kitchen attention")
    if use_sol:
        (model,) = sol_attn_h3.MiniMaxH3Sol.execute(model=model, **eval_mv._sol_execute_kwargs())
    return model


def _decode_and_mux_av(
    samples: Any,
    video_vae: Any,
    audio_vae: Any,
    mp4_path: Path,
    strip_path: Path,
    *,
    write_1fps: bool = True,
) -> dict[str, Any]:
    from comfy_extras.nodes_audio import vae_decode_audio

    v_lat, a_lat = samples.unbind()
    v_lat = v_lat.detach().cpu()
    a_lat = a_lat.detach().cpu()
    del samples
    _free_vram_everywhere()

    aud_obj = vae_decode_audio(audio_vae, {"samples": a_lat})
    aud_wav = aud_obj["waveform"].detach().cpu()
    aud_sr = int(aud_obj["sample_rate"])
    del aud_obj, a_lat
    _free_vram_everywhere()

    vid_dec = video_vae.decode(v_lat)
    if vid_dec.ndim == 5:
        vid_dec = vid_dec[0]
    vid_u8 = vid_dec.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    del vid_dec, v_lat
    _free_vram_everywhere()

    metrics = eval_mv.mux_and_measure_av(vid_u8, aud_wav, aud_sr, mp4_path, strip_path, fps=24)
    mid_start = min(18, vid_u8.shape[0] // 4)
    mid_end = max(mid_start + 1, vid_u8.shape[0] - mid_start)
    mid_sats = [eval_mv.hsv_sat(Image.fromarray(vid_u8[fi])) for fi in range(mid_start, mid_end, 4)]
    metrics["sat_mid_mean"] = round(float(np.mean(mid_sats)), 4)
    mid_im = Image.fromarray(vid_u8[vid_u8.shape[0] // 2])
    metrics["mid_laplacian_var"] = _image_quality_metrics(mid_im)["laplacian_var"]
    metrics["mp4"] = mp4_path.name
    metrics["strip"] = strip_path.name
    if write_1fps:
        parts = _write_one_a_second(vid_u8, mp4_path.with_suffix(""))
        metrics["one_fps_files"] = [Path(p["file"]).name for p in parts]
    return metrics


def _sample_av_with_model(
    model: Any,
    cond_in: Any,
    lat_in: Any,
    seed: int,
    sigmas: torch.Tensor,
    video_vae: Any,
    audio_vae: Any,
    mp4_path: Path,
    strip_path: Path,
) -> dict[str, Any]:
    import comfy.sample
    import comfy.samplers

    t0 = time.perf_counter()
    lat_t = lat_in["samples"]
    noise = comfy.sample.prepare_noise(lat_t, seed)
    guider = comfy.samplers.CFGGuider(model)
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
    gen_s = round(time.perf_counter() - t0, 1)
    del noise, guider
    res = _decode_and_mux_av(samples, video_vae, audio_vae, mp4_path, strip_path, write_1fps=True)
    res["gen_elapsed_s"] = gen_s
    return res


def _try_expand_scene(idea: str, fallback_file: str) -> tuple[str, bool]:
    """Expand via `heylook` with fast 12s timeout; fall back cleanly to `prompt_bank/<file>`."""
    fb_txt = (REPO_ROOT / "prompt_bank" / fallback_file).read_text(encoding="utf-8").strip()
    heylook_url = os.environ.get("H3_HEYLOOK_URL", "").strip()
    if not heylook_url:
        return fb_txt, False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                h3_prompt_expander.expand_video_prompt,
                prompt=idea,
                concept="none",
                mode="t2va",
                length=90,
                enabled=True,
                base_url=heylook_url,
                thinking="off",
                temperature=0.6,
                max_tokens=1600,
                use_cache=True,
            )
            out = fut.result(timeout=18.0)
        expanded = str(out.get("prompt", "")).strip()
        if "integrated_multimodal_description:" in expanded and len(expanded) > 120:
            return expanded, True
    except Exception as exc:
        print(f"  [expander] fallback ({type(exc).__name__}: {exc}) -> {fallback_file}", flush=True)
    return fb_txt, False


# ---------------------------------------------------------------------------
# Stage A: de_llm_slop Anti-Slop Master EQ Showcase
# ---------------------------------------------------------------------------
def run_de_slop() -> dict[str, Any]:
    import comfy.sd
    import comfy.utils
    from comfy.patcher_extension import WrappersMP
    import folder_paths
    import control_vector_nodes as cv_nodes
    h3_cond_mod = cv_nodes._repo("conditioning")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, vid_dir = _phase3_out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    scene_specs = [
        ("box_office", "A rainy evening cinema ticket booth with an older cashier handing two paper tickets across a brass dish.", "t2va_box_office.txt"),
        ("sitcom_kitchen", "Two flatmates arguing good-naturedly over burnt toast beside a humming refrigerator in an apartment kitchen.", "t2va_sitcom_kitchen.txt"),
        ("hardware_aisle", "An afternoon hardware store paint aisle where a clerk shakes a dented metal primer can by wooden shelves.", "t2va_hardware_aisle_short.txt"),
        ("freight_elevator", "Two mechanics checking a stamped steel control plate inside a rattling freight elevator with bare cage bulbs.", "t2va_freight_elevator.txt"),
    ]
    resolved_scenes: list[dict[str, Any]] = []
    for key, idea, fb in scene_specs:
        txt, used_hl = _try_expand_scene(idea, fb)
        resolved_scenes.append({"key": key, "prompt": txt, "used_heylook": used_hl, "fallback": fb})
    # Also inject a synthetic "commercial HDR gloss" prompt to measure explicit de-slop cleanup
    glossy_base = (
        "integrated_multimodal_description: [Shot 1] Live-action medium shot inside a neighborhood bike repair shop "
        "watching an apprentice true a steel wheel beside chrome spoke wrenches across the 3.75-second take, "
        "bathed in syrupy teal-orange HDR rim halos, creamy shallow bokeh blur melting distant racks, and glossy plastic skin smoothing. "
        "No dialogue is spoken on locked sticks.\n\n"
        "overall_soundscape: Truing caliper fingertips ping lightly against tensioned steel bicycle spokes.\n\n"
        "non_diegetic_music: N/A"
    )
    resolved_scenes.append({"key": "synthetic_gloss_bike_shop", "prompt": glossy_base, "used_heylook": False, "fallback": "synthetic"})

    _wait = eval_mv._wait_for_comfy_idle_and_free()
    (h3_enc_mod,) = (cv_nodes._repo("h3_encoder_loader"),)
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    prompt_dicts = {s["key"]: _encode_prompt_dict(clip, s["key"], s["prompt"]) for s in resolved_scenes}

    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    cond_box_90, lat_box_90 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip,
        vae=video_vae,
        prompt=prompt_dicts["box_office"]["prompt"],
        width=1152,
        height=768,
        length=90,
        single_frame=False,
    )[:2]
    del clip
    _free_vram_everywhere()

    de_op_fl = _load_bank_vector_op("de_llm_slop", "fl2va")
    de_op_1d = {**de_op_fl, "vr": torch.zeros((0, cv.POST_REFINER_DIM)), "diag": torch.zeros((0,))}

    model, img_vae = eval_mv._load_pika_fl2va_model(with_lora=True)
    arms = [
        ("Base (0.0)", None, 0.0, (0.0, 1.0)),
        ("1D (+1.15)", de_op_1d, 1.15, (0.0, 1.0)),
        ("Erasure (+1.15 [0..1])", de_op_fl, 1.15, (0.0, 1.0)),
        ("Erasure (+1.15 [0..0.50])", de_op_fl, 1.15, (0.0, 0.50)),
    ]
    still_rows: list[dict[str, Any]] = []
    sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for s in resolved_scenes:
        pk = s["key"]
        pr = prompt_dicts[pk]
        for sd in SEEDS_2:
            cols = []
            for lbl, op_u, st_v, ew in arms:
                im = eval_mv._sample_still_pika(
                    model, img_vae, pr, op=op_u, strength=st_v, seed=sd, sigmas=pika_sigmas, erase_window=ew
                )
                qm = _image_quality_metrics(im)
                still_rows.append(
                    {
                        "scene": pk,
                        "used_heylook": s["used_heylook"],
                        "seed": sd,
                        "arm": lbl,
                        **qm,
                    }
                )
                cols.append((f"{lbl} L={qm['laplacian_var']:.0f} S={qm['hsv_sat']:.2f}", im))
            sheet_rows.append((f"{pk[:18]} s={sd}", cols))

    sheet_p = img_dir / "stage_a_de_llm_slop_sheet.png"
    eval_mv.make_contact_sheet(sheet_rows, sheet_p)
    print(f"[Stage A] Wrote {sheet_p.name} ({len(still_rows)} stills).")

    del model, img_vae
    _free_vram_everywhere()

    # Speaking 90f AV check on t2va_box_office.txt
    av_model = _load_pika_model_by_family("fl2va", with_lora=True, use_sol=True)
    av_results: list[dict[str, Any]] = []
    for lbl, mode_s, st_v in [
        ("base", "none", 0.0),
        ("1d_s1p15", "add_norm_matched", 1.15),
        ("erasure_s1p15", "add_norm_matched_erasure", 1.15),
    ]:
        m_run = av_model
        if mode_s != "none":
            m_run = cv_nodes.attach_control_vector(
                av_model,
                "de_llm_slop",
                strength=st_v,
                mode=mode_s,
                mask="scene_only",
            )
        mp4_p = vid_dir / f"stage_a_box_office_90f_{lbl}.mp4"
        strip_p = vid_dir / "strips" / f"stage_a_box_office_90f_{lbl}.png"
        m_av = _sample_av_with_model(
            m_run, cond_box_90, lat_box_90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
        )
        m_av["arm"] = lbl
        av_results.append(m_av)
        print(f"  [Stage A AV] {lbl}: sat={m_av['sat_mean']:.4f}, lap={m_av['mid_laplacian_var']:.1f}, dbfs={m_av['audio_dbfs']}")

    del av_model, video_vae, audio_vae
    _free_vram_everywhere()

    out = {
        "scenes_meta": [{k: v for k, v in s.items() if k != "prompt"} for s in resolved_scenes],
        "stills": still_rows,
        "av_box_office_90f": av_results,
        "sheet": sheet_p.name,
    }
    (DATA_OUT / "stage_a_de_llm_slop.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Stage B (Exp 1): Surgical Erasure v2 (Ramp + r=8 Global Contested SVD + 3-Slider Stack)
# ---------------------------------------------------------------------------
def run_exp1() -> dict[str, Any]:
    import control_vector_nodes as cv_nodes

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, _ = _phase3_out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    prompts = eval_mv._load_nonmarket_prompts()
    # Add box_office cleanly as well
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    prompts["box_office"] = _encode_prompt_dict(
        clip, "box_office", (REPO_ROOT / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip()
    )
    del clip
    _free_vram_everywhere()

    banks = {
        cn: torch.load(eval_mv.BANKS_DIR / f"{cn}.pt", map_location="cpu", weights_only=False)
        for cn in ("noir", "paper_origami", "chaos_panic")
    }
    ops_by_concept: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for cn, b_d in banks.items():
        a_pcs = b_d.get("a_pieces_per_pair") or eval_mv._resolve_bank_pieces(cn, b_d["a_spans_per_pair"])
        ops_by_concept[cn] = {
            "1D": eval_mv.build_erasure_variant(b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], rank=0),
            "svd_ac_r8": eval_mv.build_erasure_variant(b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], rank=8, vr_mode="svd_ac"),
            "contested_svd_r8": eval_mv.build_erasure_variant(
                b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], a_pieces_per_pair=a_pcs, rank=8, vr_mode="contested_svd"
            ),
            "global_contested_svd_r8": _build_global_contested_variant(
                b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], a_pcs, rank=8
            ),
        }

    model, img_vae = eval_mv._load_pika_fl2va_model(with_lora=True)

    # Part 1: Strength ramp + Vr variant comparison (keep r=8!)
    ramp_strengths = [-1.0, 0.0, 0.5, 1.0, 1.25, 1.5]
    ramp_records: list[dict[str, Any]] = []
    ramp_sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    cells = [
        ("noir", "indoor_neutral"),
        ("paper_origami", "sitcom_kitchen"),
        ("chaos_panic", "hardware_aisle"),
    ]
    for cn, pn in cells:
        pr = prompts[pn]
        for vr_lbl, ew in [
            ("1D", (0.0, 1.0)),
            ("svd_ac_r8", (0.0, 1.0)),
            ("contested_svd_r8", (0.0, 1.0)),
            ("global_contested_svd_r8", (0.0, 1.0)),
            ("global_contested_svd_r8_ew50", (0.0, 0.50)),
        ]:
            op_key = "global_contested_svd_r8" if vr_lbl == "global_contested_svd_r8_ew50" else vr_lbl
            op_u = ops_by_concept[cn][op_key]
            cols = []
            for st_v in ramp_strengths:
                im = eval_mv._sample_still_pika(
                    model, img_vae, pr, op=op_u, strength=st_v, seed=SEED_MAIN, sigmas=pika_sigmas, erase_window=ew
                )
                qm = _image_quality_metrics(im)
                ramp_records.append(
                    {"concept": cn, "prompt": pn, "variant": vr_lbl, "strength": st_v, "erase_window": list(ew), **qm}
                )
                cols.append((f"s={st_v:+.2f} S={qm['hsv_sat']:.2f} L={qm['laplacian_var']:.0f}", im))
            ramp_sheet_rows.append((f"{cn[:10]} {vr_lbl[:14]}", cols))

    ramp_sheet_p = img_dir / "stage_b_exp1_ramp_sheet.png"
    eval_mv.make_contact_sheet(ramp_sheet_rows, ramp_sheet_p)

    # Part 2: 3-slider simultaneous stack (noir +1.25 + chaos_panic +1.0 + de_llm_slop +0.9) on box_office & sitcom_kitchen
    de_op = _load_bank_vector_op("de_llm_slop", "fl2va")
    stack_specs = [
        ("Base", []),
        ("Single noir +1.25", [(ops_by_concept["noir"]["svd_ac_r8"], 1.25, True)]),
        (
            "Stack3 1D (n+c+d)",
            [
                (ops_by_concept["noir"]["1D"], 1.25, False),
                (ops_by_concept["chaos_panic"]["1D"], 1.0, False),
                ({**de_op, "vr": torch.zeros((0, 5376)), "diag": torch.zeros((0,))}, 0.9, False),
            ],
        ),
        (
            "Stack3 Erasure (n+c+d)",
            [
                (ops_by_concept["noir"]["svd_ac_r8"], 1.25, True),
                (ops_by_concept["chaos_panic"]["global_contested_svd_r8"], 1.0, True),
                (de_op, 0.9, True),
            ],
        ),
    ]
    from comfy.patcher_extension import WrappersMP
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers

    stack_records: list[dict[str, Any]] = []
    stack_sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for pn in ("box_office", "sitcom_kitchen"):
        pr = prompts[pn]
        enc_3d = pr["enc"].to(torch.bfloat16).unsqueeze(0)
        tags_2d = pr["tags"].unsqueeze(0)
        sm = pr["scene_mask"]
        for sd in SEEDS_2:
            cols = []
            for st_lbl, item_list in stack_specs:
                m_run = model
                if item_list:
                    m_run = model.clone()

                    def _dm_stack(executor, x, timestep, context, transformer_options={}, items=item_list, **kwargs):
                        ctx_use = context
                        if context is not None:
                            ctx_ref = eval_mv._refine_context_if_needed(executor, context, transformer_options)
                            ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                            rows = ref64[sm]
                            tot = torch.zeros_like(rows)
                            for op_i, st_i, er_i in items:
                                tot = tot + eval_mv.apply_erasure_shift_rows(rows, op_i, st_i, erase_active=er_i)
                            out = ref64.clone()
                            out[sm] = rows + tot
                            ctx_use = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
                        return executor(x, timestep, ctx_use, transformer_options, **kwargs)

                    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["exp1_stack"] = [_dm_stack]

                nested = nt.NestedTensor((torch.zeros((1, 24, 1, 768 // 16, 1152 // 16)), torch.zeros((1, 32, 2, 2))))
                noise = comfy.sample.prepare_noise(nested, sd)
                guider = comfy.samplers.CFGGuider(m_run)
                guider.inner_set_conds({"positive": [[enc_3d, {"minimax_token_tags": tags_2d.to(torch.long)}]]})
                guider.set_cfg(1.0)
                samples = guider.sample(
                    noise, nested, comfy.samplers.sampler_object("lcm"), pika_sigmas,
                    denoise_mask=None, callback=None, disable_pbar=True, seed=sd,
                )
                vid_lat = (samples.unbind()[0] if isinstance(samples, nt.NestedTensor) else samples).detach()
                decoded = img_vae.decode(vid_lat)
                if decoded.ndim == 5:
                    decoded = decoded[:, 0]
                im = Image.fromarray(np.clip(decoded[0].detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8))
                qm = _image_quality_metrics(im)
                stack_records.append({"prompt": pn, "seed": sd, "arm": st_lbl, **qm})
                cols.append((f"{st_lbl} S={qm['hsv_sat']:.2f} L={qm['laplacian_var']:.0f}", im))
            stack_sheet_rows.append((f"{pn[:12]} s={sd}", cols))

    stack_sheet_p = img_dir / "stage_b_exp1_stack3_sheet.png"
    eval_mv.make_contact_sheet(stack_sheet_rows, stack_sheet_p)

    del model, img_vae
    _free_vram_everywhere()

    out = {
        "ramp_records": ramp_records,
        "stack3_records": stack_records,
        "sheets": [ramp_sheet_p.name, stack_sheet_p.name],
    }
    (DATA_OUT / "stage_b_surgical_erasure_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Stage B] Saved stage_b_surgical_erasure_v2.json ({len(ramp_records)} ramp + {len(stack_records)} stack renders).")
    return out


# ---------------------------------------------------------------------------
# Stage C (Exp 2): Same-Modality 1:1 Vision-Token Erasure (erasure.mu_a)
# ---------------------------------------------------------------------------
def _make_10_color_bw_pairs(inp_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 10 distinct landscape 512x320 RGB->greyscale contrast pairs from $H3_INPUT_DIR stills."""
    sources = [
        Image.open(inp_dir / "woman_blonde.jpg").convert("RGB"),
        Image.open(inp_dir / "fl2va_noodle_f000.png").convert("RGB"),
        Image.open(inp_dir / "fl2va_noodle_f123.png").convert("RGB"),
    ]
    a_list, b_list = [], []
    w_t, h_t = 512, 320
    # 10 deterministic crops/tonal shifts across the 3 source images
    crops = [
        (0, (0.0, 0.0, 1.0, 1.0), 1.00),
        (0, (0.10, 0.05, 0.90, 0.85), 1.08),
        (0, (0.05, 0.12, 0.95, 0.96), 0.92),
        (1, (0.0, 0.0, 1.0, 1.0), 1.00),
        (1, (0.08, 0.04, 0.92, 0.92), 1.06),
        (1, (0.15, 0.10, 0.98, 0.96), 0.94),
        (2, (0.0, 0.0, 1.0, 1.0), 1.00),
        (2, (0.04, 0.08, 0.88, 0.92), 1.05),
        (2, (0.12, 0.02, 0.96, 0.88), 0.95),
        (0, (0.18, 0.08, 0.82, 0.78), 1.02),
    ]
    for s_idx, (fx0, fy0, fx1, fy1), gain in crops:
        im = sources[s_idx]
        iw, ih = im.size
        crop = im.crop((int(fx0 * iw), int(fy0 * ih), int(fx1 * iw), int(fy1 * ih))).resize(
            (w_t, h_t), Image.Resampling.LANCZOS
        )
        arr = np.clip(np.asarray(crop, dtype=np.float32) * gain, 0.0, 255.0).astype(np.uint8)
        im_a = Image.fromarray(arr)
        im_b = im_a.convert("L").convert("RGB")
        a_list.append(torch.from_numpy(np.asarray(im_a, dtype=np.float32) / 255.0))
        b_list.append(torch.from_numpy(np.asarray(im_b, dtype=np.float32) / 255.0))
    return torch.stack(a_list, dim=0), torch.stack(b_list, dim=0)


def _read_mp4_frames_u8(mp4_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    frames_rgb = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames_rgb, axis=0)


def _portrait_split_sat_metrics(vid_u8: np.ndarray) -> dict[str, float]:
    """Compare saturation in center subject band vs outer borders across sampled frames."""
    n_f, h, w, _ = vid_u8.shape
    cx0, cx1 = int(0.32 * w), int(0.68 * w)
    cy0, cy1 = int(0.15 * h), int(0.85 * h)
    c_sats, o_sats = [], []
    for fi in range(0, n_f, 4):
        hsv = np.asarray(Image.fromarray(vid_u8[fi]).convert("HSV"), dtype=np.float32)[..., 1] / 255.0
        mask_c = np.zeros((h, w), dtype=bool)
        mask_c[cy0:cy1, cx0:cx1] = True
        c_sats.append(float(hsv[mask_c].mean()))
        o_sats.append(float(hsv[~mask_c].mean()))
    return {
        "center_sat_mean": round(float(np.mean(c_sats)), 4),
        "border_sat_mean": round(float(np.mean(o_sats)), 4),
    }


def _wrap_dual_text_and_vision(
    base_model: Any,
    *,
    text_mask: torch.Tensor | None,
    text_op: dict[str, torch.Tensor] | None,
    text_strength: float,
    vision_mask: torch.Tensor | None,
    vis_op: dict[str, torch.Tensor] | None,
    vis_strength: float,
    erase_text: bool = True,
) -> Any:
    """Apply post-refiner steering separately on `text_mask` (`tags==1`) and `vision_mask` (`tags==0` with `mu_a`)."""
    from comfy.patcher_extension import WrappersMP

    m_run = base_model.clone()
    cache_ctx: dict[int, torch.Tensor] = {}

    def _dm_wrap(executor, x, timestep, context, transformer_options={}, **kwargs):
        ctx_use = context
        if context is not None:
            ctx_ref = eval_mv._refine_context_if_needed(executor, context, transformer_options)
            k = id(ctx_ref)
            if k not in cache_ctx:
                ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                out = ref64.clone()
                if text_op is not None and text_mask is not None and bool(text_mask.any()) and abs(text_strength) > 1e-9:
                    rows = out[text_mask]
                    sh = eval_mv.apply_erasure_shift_rows(rows, text_op, text_strength, erase_active=erase_text)
                    out[text_mask] = rows + sh
                if vis_op is not None and vision_mask is not None and bool(vision_mask.any()) and abs(vis_strength) > 1e-9:
                    rows_v = out[vision_mask]
                    spec_v = (
                        vis_op.get("vr"),
                        vis_op.get("diag"),
                        vis_op.get("mu_d"),
                        None,
                        vis_op.get("mu_a"),
                    )
                    sh_v = cv._erasure_row_shift64(rows_v, vis_op["v1d"], vis_strength, spec_v)
                    out[vision_mask] = rows_v + sh_v
                cache_ctx[k] = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
            ctx_use = cache_ctx[k]
        return executor(x, timestep, ctx_use, transformer_options, **kwargs)

    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["phase3_dual_tv"] = [_dm_wrap]
    return m_run


def run_exp2() -> dict[str, Any]:
    import comfy.sd
    import comfy.utils
    import folder_paths
    import control_vector_nodes as cv_nodes
    h3_cond_mod = cv_nodes._repo("conditioning")
    ref_mod = cv_nodes._repo("reference_conditioning")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    _, vid_dir = _phase3_out_dirs()
    inp_dir = Path(os.environ["H3_INPUT_DIR"])
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    woman_txt = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
    noodle_txt = (REPO_ROOT / "prompt_bank" / "t2va_noodle_bar.txt").read_text(encoding="utf-8").strip()

    img_a, img_b = _make_10_color_bw_pairs(inp_dir)
    img_woman = _load_comfy_image_tensor(inp_dir / "woman_blonde.jpg")
    img_n0 = _load_comfy_image_tensor(inp_dir / "fl2va_noodle_f000.png")
    img_n1 = _load_comfy_image_tensor(inp_dir / "fl2va_noodle_f123.png")

    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    # Build ref2va condition (`on_both_clocks`: 90f)
    (refs_w,) = ref_mod.MiniMaxH3AppendRefImage.execute(
        image=img_woman, size_policy="max", qwen_view="separate", references=None, use_vae=True
    )
    cond_ref90, lat_ref90 = ref_mod.MiniMaxH3ReferenceConditioning.execute(
        clip=clip, references=refs_w, prompt=woman_txt, width=1152, height=768, length=90,
        video_policy="comfy", image_policy="comfy", vae=video_vae, audio_vae=audio_vae, single_frame=False
    )[:2]
    tags_ref = cond_ref90[0][1]["minimax_token_tags"].reshape(-1).cpu()
    masks_ref = cond_ref90[0][1]["h3_span_masks"]
    ref_desc_mask = h3_spans.inference_mask(masks_ref, "description_only").reshape(-1).cpu().bool()
    ref_vis_mask = (tags_ref == 0)

    # Build fl2va Noodle Counter condition (`on_both_clocks` + >=124f legal FL2VA span: 141f)
    cond_fl141, lat_fl141 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=video_vae, prompt=noodle_txt, width=1152, height=768, length=141,
        first_frame=img_n0, last_frame=img_n1, canvas="from_keyframe", single_frame=False
    )[:2]
    tags_fl = cond_fl141[0][1]["minimax_token_tags"].reshape(-1).cpu()
    masks_fl = cond_fl141[0][1]["h3_span_masks"]
    fl_scene_mask = h3_spans.inference_mask(masks_fl, "scene_only").reshape(-1).cpu().bool()
    fl_vis_mask = (tags_fl == 0)

    print(
        f"[Exp 2] ref90 tokens={tags_ref.numel()} (vis={int(ref_vis_mask.sum())}, desc={int(ref_desc_mask.sum())}); "
        f"fl141 tokens={tags_fl.numel()} (vis={int(fl_vis_mask.sum())}, scene={int(fl_scene_mask.sum())})"
    )

    # Build vision_tokens erasure vectors into scratch dir (keep production bank clean)
    scratch_vec_dir = vid_dir / "scratch_vectors"
    scratch_vec_dir.mkdir(parents=True, exist_ok=True)
    ref_model = _load_pika_model_by_family("ref2va", with_lora=True, use_sol=True)
    vis_ref_path, vis_ref_rep = cv_nodes.make_control_vector_from_media_pairs(
        image_a=img_a,
        image_b=img_b,
        model=ref_model,
        clip=clip,
        output_name="noir_vis_phase3_ref",
        pooling="vision_tokens",
        short_edge=512,
        scene_prompt=woman_txt,
        output_dir=scratch_vec_dir,
        overwrite=True,
        refiner_device="cuda",
    )
    vt_ref_vis, _ = cv.load_vector(vis_ref_path)
    assert "erasure.mu_a" in vt_ref_vis, f"Expected erasure.mu_a in {vis_ref_path}"
    op_vis_ref = {
        "v1d": vt_ref_vis["vector.post_refiner"],
        "mu_d": vt_ref_vis.get("erasure.mu_d", vt_ref_vis["vector.post_refiner"]),
        "vr": vt_ref_vis["erasure.vr"],
        "diag": vt_ref_vis["erasure.diag"],
        "mu_a": vt_ref_vis["erasure.mu_a"],
    }

    txt_noir_ref = _load_bank_vector_op("noir", "ref2va")
    ref_results: list[dict[str, Any]] = []
    ref_arms = [
        ("gear0_1d_text", False, None, 1.5, 0.0, 1.0, 0.999),
        ("gear1_selective_color_erasure", True, None, 1.5, 0.0, 1.0, 0.999),
        ("gear2a_vis_erasure_rel033", True, op_vis_ref, 1.5, 1.25, 0.33, 0.999),
        ("gear2b_vis_erasure_rel017_clean085", True, op_vis_ref, 1.5, 1.25, 0.17, 0.85),
    ]
    for arm_name, er_t, vis_op_u, st_t, st_v, ep_v, cl_v in ref_arms:
        (m_rel,) = cv_nodes.MiniMaxH3ReleaseReferences.execute(
            model=ref_model, end_percent=ep_v, clean_level=cl_v, video_end_percent=ep_v, guide_end_percent=ep_v
        )
        m_run = _wrap_dual_text_and_vision(
            m_rel,
            text_mask=ref_desc_mask,
            text_op=txt_noir_ref,
            text_strength=st_t,
            vision_mask=ref_vis_mask,
            vis_op=vis_op_u,
            vis_strength=st_v,
            erase_text=er_t,
        )
        mp4_p = vid_dir / f"stage_c_ref2va_woman_{arm_name}.mp4"
        strip_p = vid_dir / "strips" / f"stage_c_ref2va_woman_{arm_name}.png"
        m_res = _sample_av_with_model(
            m_run, cond_ref90, lat_ref90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
        )
        m_res.update(_portrait_split_sat_metrics(_read_mp4_frames_u8(mp4_p)))
        m_res["arm"] = arm_name
        ref_results.append(m_res)
        print(
            f"  [Exp 2 ref2va] {arm_name}: sat={m_res['sat_mean']:.4f}, "
            f"center_sat={m_res['center_sat_mean']:.4f}, border_sat={m_res['border_sat_mean']:.4f}, dbfs={m_res['audio_dbfs']}"
        )

    del ref_model
    _free_vram_everywhere()

    # Now FL2VA Noodle Counter 90f
    fl_model = _load_pika_model_by_family("fl2va", with_lora=True, use_sol=True)
    vis_fl_path, vis_fl_rep = cv_nodes.make_control_vector_from_media_pairs(
        image_a=img_a,
        image_b=img_b,
        model=fl_model,
        clip=clip,
        output_name="noir_vis_phase3_fl",
        pooling="vision_tokens",
        short_edge=512,
        scene_prompt=noodle_txt,
        output_dir=scratch_vec_dir,
        overwrite=True,
        refiner_device="cuda",
    )
    del clip
    _free_vram_everywhere()

    vt_fl_vis, _ = cv.load_vector(vis_fl_path)
    assert "erasure.mu_a" in vt_fl_vis
    op_vis_fl = {
        "v1d": vt_fl_vis["vector.post_refiner"],
        "mu_d": vt_fl_vis.get("erasure.mu_d", vt_fl_vis["vector.post_refiner"]),
        "vr": vt_fl_vis["erasure.vr"],
        "diag": vt_fl_vis["erasure.diag"],
        "mu_a": vt_fl_vis["erasure.mu_a"],
    }
    txt_noir_fl = _load_bank_vector_op("noir", "fl2va")

    fl_results: list[dict[str, Any]] = []
    fl_arms = [
        ("text_only_erasure_rel033", None, 1.55, 0.0, 0.33, 0.999),
        ("dual_text_plus_vis_rel033", op_vis_fl, 1.55, 1.35, 0.33, 0.999),
        ("dual_text_plus_vis_rel017_clean085", op_vis_fl, 1.55, 1.35, 0.17, 0.85),
    ]
    for arm_name, vis_op_u, st_t, st_v, ep_v, cl_v in fl_arms:
        (m_rel,) = cv_nodes.MiniMaxH3ReleaseReferences.execute(
            model=fl_model, end_percent=ep_v, clean_level=cl_v, video_end_percent=ep_v, guide_end_percent=ep_v
        )
        m_run = _wrap_dual_text_and_vision(
            m_rel,
            text_mask=fl_scene_mask,
            text_op=txt_noir_fl,
            text_strength=st_t,
            vision_mask=fl_vis_mask,
            vis_op=vis_op_u,
            vis_strength=st_v,
            erase_text=True,
        )
        mp4_p = vid_dir / f"stage_c_fl2va_noodle_{arm_name}.mp4"
        strip_p = vid_dir / "strips" / f"stage_c_fl2va_noodle_{arm_name}.png"
        m_res = _sample_av_with_model(
            m_run, cond_fl141, lat_fl141, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
        )
        m_res["arm"] = arm_name
        fl_results.append(m_res)
        print(f"  [Exp 2 fl2va] {arm_name}: sat={m_res['sat_mean']:.4f}, mid_sat={m_res['sat_mid_mean']:.4f}, dbfs={m_res['audio_dbfs']}")

    del fl_model, video_vae, audio_vae
    _free_vram_everywhere()

    out = {
        "vis_build_reports": {
            "ref2va": {"path": vis_ref_path.name, "has_mu_a": True, "mpc": vis_ref_rep.get("mean_pairwise_cosine")},
            "fl2va": {"path": vis_fl_path.name, "has_mu_a": True, "mpc": vis_fl_rep.get("mean_pairwise_cosine")},
        },
        "ref2va_woman_90f": ref_results,
        "fl2va_noodle_141f": fl_results,
    }
    (DATA_OUT / "stage_c_vision_erasure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Stage D (Exp 3): Shadow-Text K/V Block-Band Routing (`blocks.0..49`)
# ---------------------------------------------------------------------------
def _attach_shadow_text_kv_band(
    model: Any,
    *,
    text_mask: torch.Tensor,
    text_op: dict[str, torch.Tensor],
    text_strength: float,
    active_blocks: frozenset[int],
    run_sanity_check: bool = False,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Install shadow `h_txt_base` + `optimized_attention_override` under `Sol` (zero `patches_replace`)."""
    import comfy.model_management as comfy_mm
    import comfy.model_prefetch as comfy_mp
    import comfy.quant_ops
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    from comfy.patcher_extension import WrappersMP

    m_run = model.clone()
    dm = m_run.get_model_object("diffusion_model")
    heads = int(dm.blocks[0].attn.heads)
    head_dim = int(dm.blocks[0].attn.head_dim)
    state: dict[str, Any] = {
        "h_txt_base": None,
        "step_kv_by_block": {},
        "ctx_cache": {},
        "sanity_max_block0_diff": None,
        "kv_swaps": 0,
    }

    def _slice_rope(rope_freqs: Any, tx_a: int, tx_b: int) -> Any:
        if rope_freqs is None:
            return None
        if rope_freqs.ndim >= 2 and rope_freqs.shape[1] >= tx_b:
            return rope_freqs[:, tx_a:tx_b]
        return rope_freqs[tx_a:tx_b]

    def _make_pre_hook(b_idx: int):
        def _pre_hook(module, args, kwargs):
            h_b = state.get("h_txt_base")
            if h_b is None:
                return
            x_full, t_emb, mod_segments, rope_freqs = args[0], args[1], args[2], args[3]
            to = kwargs.get("transformer_options") or {}
            layout = to.get("minimax_h3_layout")
            if layout is None:
                return
            tx_a, tx_b, _ = next(s for s in layout.segments if s[2] == "text")
            if b_idx == 0 and run_sanity_check and state["sanity_max_block0_diff"] is None:
                state["sanity_max_block0_diff"] = float((x_full[tx_a:tx_b] - state["ctx_steered_0"]).abs().max().item())

            mod_txt = [(sa - tx_a, sb - tx_a, m_row) for (sa, sb, m_row) in mod_segments if sa >= tx_a and sb <= tx_b]
            rope_b = _slice_rope(rope_freqs, tx_a, tx_b)
            l_txt = tx_b - tx_a

            with comfy_mp.pause_malloc_graph(), torch.no_grad():
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = module.adaln_proj(t_emb)
                h_in = _mod_scale_shift(module.norm1(h_b.clone()), shift_msa, scale_msa, mod_txt)
                q_b, k_b, v_b = module.attn.qkv_proj(h_in).split(heads * head_dim, dim=-1)
                v_b = v_b.view(l_txt, heads, head_dim)
                if rope_b is not None:
                    q_r = q_b.view(1, l_txt, heads, head_dim)
                    k_r = k_b.view(1, l_txt, heads, head_dim)
                    qw = comfy_mm.cast_to(module.attn.q_norm.weight, device=h_in.device)
                    kw = comfy_mm.cast_to(module.attn.k_norm.weight, device=h_in.device)
                    q_r, k_r = comfy.quant_ops.ck.rms_rope_split_half(
                        q_r, k_r, rope_b, qw, kw, epsilon=module.attn.q_norm.eps, rot_dim=rope_b.shape[-3] * 2
                    )
                    q_b, k_b = q_r[0], k_r[0]
                else:
                    q_b = module.attn.q_norm(q_b.view(l_txt, heads, head_dim))
                    k_b = module.attn.k_norm(k_b.view(l_txt, heads, head_dim))

                state["step_kv_by_block"][b_idx] = {
                    "module": module,
                    "tx_a": tx_a,
                    "tx_b": tx_b,
                    "l_txt": l_txt,
                    "mod_txt": mod_txt,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "q_b": q_b.transpose(0, 1).unsqueeze(0).contiguous(),
                    "k_b": k_b.transpose(0, 1).unsqueeze(0).contiguous(),
                    "v_b": v_b.transpose(0, 1).unsqueeze(0).contiguous(),
                }

        return _pre_hook

    hook_handles = [blk.register_forward_pre_hook(_make_pre_hook(idx), with_kwargs=True) for idx, blk in enumerate(dm.blocks)]

    def _make_shadow_override(previous):
        def override(func, q, k, v, n_heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            held = not torch.is_tensor(q)
            options = kwargs.get("transformer_options") or {}
            b_idx = options.get("block_index")
            info = state["step_kv_by_block"].pop(b_idx, None) if b_idx is not None else None

            def _pass_through(q_u, k_u, v_u):
                if not torch.is_tensor(q_u):
                    return previous.container_function(
                        q_u, k_u, v_u, n_heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
                    )
                target = func if previous is None else functools.partial(previous, func)
                return target(
                    q_u, k_u, v_u, n_heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
                )

            if info is not None and skip_reshape and mask is None:
                k_peek = k if torch.is_tensor(k) else k.peek()
                v_peek = v if torch.is_tensor(v) else v.peek()
                tx_a, tx_b, l_txt = info["tx_a"], info["tx_b"], info["l_txt"]
                # 1) Advance shadow unsteered text stream h_txt_base with exact SDPA against [base_txt, steered_non_txt]
                with comfy_mp.pause_malloc_graph(), torch.no_grad():
                    kb_cast = info["k_b"].to(device=k_peek.device, dtype=k_peek.dtype)
                    vb_cast = info["v_b"].to(device=v_peek.device, dtype=v_peek.dtype)
                    k_other = [k_peek[:, :, :tx_a], kb_cast, k_peek[:, :, tx_b:]] if tx_a > 0 else [kb_cast, k_peek[:, :, tx_b:]]
                    v_other = [v_peek[:, :, :tx_a], vb_cast, v_peek[:, :, tx_b:]] if tx_a > 0 else [vb_cast, v_peek[:, :, tx_b:]]
                    qb_cast = info["q_b"].to(device=k_peek.device, dtype=k_peek.dtype)
                    attn_b = F.scaled_dot_product_attention(
                        qb_cast, torch.cat(k_other, dim=2), torch.cat(v_other, dim=2), dropout_p=0.0, is_causal=False
                    )
                    attn_b_2d = attn_b[0].transpose(0, 1).reshape(l_txt, n_heads * head_dim)
                    blk = info["module"]
                    x_mid = _mod_gate(state["h_txt_base"].clone(), info["gate_msa"], blk.attn.out_proj(attn_b_2d), info["mod_txt"])
                    h_mlp_in = _mod_scale_shift(blk.norm2(x_mid.clone()), info["shift_mlp"], info["scale_mlp"], info["mod_txt"])
                    state["h_txt_base"] = _mod_gate(x_mid, info["gate_mlp"], blk.mlp(h_mlp_in), info["mod_txt"])

                # 2) Outside `active_blocks`, replace K/V text rows `[tx_a:tx_b]` seen by all queries with `(k_b, v_b)`
                if int(b_idx) not in active_blocks:
                    state["kv_swaps"] += 1
                    if held:
                        box = type(q)
                        q_t, k_t, v_t = q.take(), k.take(), v.take()
                        with comfy_mp.pause_malloc_graph():
                            k_new = k_t.clone()
                            v_new = v_t.clone()
                            k_new[:, :, tx_a:tx_b] = kb_cast
                            v_new[:, :, tx_a:tx_b] = vb_cast
                        return previous.container_function(
                            box(q_t), box(k_new), box(v_new), n_heads, mask=mask, attn_precision=attn_precision,
                            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
                        )
                    with comfy_mp.pause_malloc_graph():
                        k_new = k_peek.clone()
                        v_new = v_peek.clone()
                        k_new[:, :, tx_a:tx_b] = kb_cast
                        v_new[:, :, tx_a:tx_b] = vb_cast
                    return _pass_through(q, k_new, v_new)

            return _pass_through(q, k, v)

        if getattr(previous, "container_function", None) is not None:
            override.container_function = functools.partial(override, None)
        return override

    def _dm_wrap(executor, x, timestep, context, transformer_options={}, **kwargs):
        if context is None:
            return executor(x, timestep, context, transformer_options, **kwargs)
        ctx_ref = eval_mv._refine_context_if_needed(executor, context, transformer_options)
        ck = id(ctx_ref)
        if ck not in state["ctx_cache"]:
            ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
            out64 = ref64.clone()
            if abs(text_strength) > 1e-9:
                rows = out64[text_mask]
                sh = eval_mv.apply_erasure_shift_rows(rows, text_op, text_strength, erase_active=True)
                out64[text_mask] = rows + sh
            state["ctx_cache"][ck] = (
                ctx_ref[0].detach().cpu().clone(),
                out64.to(dtype=ctx_ref.dtype).cpu().unsqueeze(0),
            )
        base_cpu, steered_cpu = state["ctx_cache"][ck]
        with comfy_mp.pause_malloc_graph():
            steered_3d = steered_cpu.to(device=ctx_ref.device, dtype=ctx_ref.dtype)
            state["h_txt_base"] = base_cpu.to(device=steered_3d.device, dtype=steered_3d.dtype).clone()
            state["ctx_steered_0"] = steered_3d[0]
            state["step_kv_by_block"].clear()

        to_next = dict(transformer_options or {})
        prev_override = to_next.get("optimized_attention_override")
        to_next["optimized_attention_override"] = _make_shadow_override(prev_override)
        try:
            return executor(x, timestep, steered_3d, to_next, **kwargs)
        finally:
            state["h_txt_base"] = None
            state["ctx_steered_0"] = None
            state["step_kv_by_block"].clear()

    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["exp3_shadow_kv"] = [_dm_wrap]
    return m_run, hook_handles, state


def _face_vs_bg_origami_metrics(vid_u8: np.ndarray, ref_woman: Image.Image) -> dict[str, float]:
    """Measure portrait center-crop sharpness/sat & correlation vs `<Picture 1>` + border sharpness."""
    n_f, h, w, _ = vid_u8.shape
    cx0, cx1 = int(0.30 * w), int(0.70 * w)
    cy0, cy1 = int(0.12 * h), int(0.78 * h)
    # Resize ref_woman to center crop height/width for histogram color-sat distance
    ref_center = np.asarray(ref_woman.resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS).convert("HSV"), dtype=np.float32)
    ref_sat = float(ref_center[..., 1].mean() / 255.0)
    c_sats, b_sats, b_laps = [], [], []
    for fi in range(0, n_f, 4):
        im = Image.fromarray(vid_u8[fi])
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
        rgb = np.asarray(im, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F) ** 2
        mask_c = np.zeros((h, w), dtype=bool)
        mask_c[cy0:cy1, cx0:cx1] = True
        c_sats.append(float(hsv[mask_c].mean()))
        b_sats.append(float(hsv[~mask_c].mean()))
        b_laps.append(float(lap[~mask_c].mean()))
    c_sat_m = float(np.mean(c_sats))
    return {
        "center_sat_mean": round(c_sat_m, 4),
        "center_vs_ref_sat_diff": round(abs(c_sat_m - ref_sat), 4),
        "border_sat_mean": round(float(np.mean(b_sats)), 4),
        "border_laplacian_energy": round(float(np.mean(b_laps)), 2),
    }


def run_exp3() -> dict[str, Any]:
    import comfy.cli_args as _ca
    import comfy.sd
    import comfy.utils
    import folder_paths
    import control_vector_nodes as cv_nodes
    ref_mod = cv_nodes._repo("reference_conditioning")

    prev_cc = _ca.args.disable_comfy_compiler
    _ca.args.disable_comfy_compiler = True
    try:
        return _run_exp3_impl(comfy, folder_paths, cv_nodes, ref_mod)
    finally:
        _ca.args.disable_comfy_compiler = prev_cc


def _run_exp3_impl(comfy: Any, folder_paths: Any, cv_nodes: Any, ref_mod: Any) -> dict[str, Any]:

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    _, vid_dir = _phase3_out_dirs()
    inp_dir = Path(os.environ["H3_INPUT_DIR"])
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    woman_txt = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
    img_woman_pil = Image.open(inp_dir / "woman_blonde.jpg").convert("RGB")
    img_woman = _load_comfy_image_tensor(inp_dir / "woman_blonde.jpg")

    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    (refs_w,) = ref_mod.MiniMaxH3AppendRefImage.execute(
        image=img_woman, size_policy="max", qwen_view="separate", references=None, use_vae=True
    )
    cond_ref90, lat_ref90 = ref_mod.MiniMaxH3ReferenceConditioning.execute(
        clip=clip, references=refs_w, prompt=woman_txt, width=1152, height=768, length=90,
        video_policy="comfy", image_policy="comfy", vae=video_vae, audio_vae=audio_vae, single_frame=False
    )[:2]
    del clip
    _free_vram_everywhere()

    masks_ref = cond_ref90[0][1]["h3_span_masks"]
    ref_desc_mask = h3_spans.inference_mask(masks_ref, "description_only").reshape(-1).cpu().bool()
    op_origami = _load_bank_vector_op("paper_origami", "ref2va")

    sol_attn_h3 = cv_nodes._repo("sol_attn_h3")

    ref_model = _load_pika_model_by_family("ref2va", with_lora=True, use_sol=False)
    (ref_model_rel,) = cv_nodes.MiniMaxH3ReleaseReferences.execute(
        model=ref_model, end_percent=0.50, clean_level=0.999, video_end_percent=0.50
    )
    sol_kw_default = eval_mv._sol_execute_kwargs()
    sol_kw_dense30_49 = {**sol_kw_default, "dense_blocks": ",".join(str(i) for i in range(30, 50))}
    (sol_default_model,) = sol_attn_h3.MiniMaxH3Sol.execute(model=ref_model_rel, **sol_kw_default)
    (sol_dense30_49_model,) = sol_attn_h3.MiniMaxH3Sol.execute(model=ref_model_rel, **sol_kw_dense30_49)

    band_arms = [
        ("unsteered_base", 0.0, frozenset(), False),
        ("all_blocks_0_49", 1.55, frozenset(range(50)), False),
        ("early_only_0_19", 1.55, frozenset(range(0, 20)), False),
        ("mid_late_20_49", 1.55, frozenset(range(20, 50)), False),
        ("late_only_30_49", 1.55, frozenset(range(30, 50)), False),
        ("all_plus_lse_ref_balance", 1.55, frozenset(range(50)), True),
    ]
    exp3_results: list[dict[str, Any]] = []
    for arm_name, st_arm, active_blks, use_lse in band_arms:
        if use_lse:
            # Make blocks 30..49 dense under Sol so attach_lse_reference_balance genuinely executes on them.
            m_base = h3_attn.attach_lse_reference_balance(
                sol_dense30_49_model, ref_indices_a=[0], weight=1.65, blocks="30-49"
            )
        else:
            m_base = sol_default_model
        m_run, handles, st = _attach_shadow_text_kv_band(
            m_base,
            text_mask=ref_desc_mask,
            text_op=op_origami,
            text_strength=st_arm,
            active_blocks=active_blks,
            run_sanity_check=True,
        )
        mp4_p = vid_dir / f"stage_d_ref2va_origami_{arm_name}.mp4"
        strip_p = vid_dir / "strips" / f"stage_d_ref2va_origami_{arm_name}.png"
        try:
            m_res = _sample_av_with_model(
                m_run, cond_ref90, lat_ref90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
            )
        finally:
            for hh in handles:
                hh.remove()

        vid_u8 = _read_mp4_frames_u8(mp4_p)
        pm = _face_vs_bg_origami_metrics(vid_u8, img_woman_pil)
        m_res.update(_portrait_split_sat_metrics(vid_u8))
        m_res.update(pm)
        m_res["arm"] = arm_name
        m_res["text_strength"] = st_arm
        m_res["active_blocks_count"] = len(active_blks)
        m_res["kv_swaps"] = st["kv_swaps"]
        m_res["sanity_block0_diff"] = st["sanity_max_block0_diff"]
        m_res["lse_ref_balance"] = use_lse
        exp3_results.append(m_res)
        print(
            f"  [Exp 3] {arm_name}: swaps={st['kv_swaps']}, sat={m_res['sat_mean']:.4f}, "
            f"border_lap={m_res['border_laplacian_energy']}, center_sat_diff={m_res['center_vs_ref_sat_diff']}, dbfs={m_res['audio_dbfs']}"
        )

    del ref_model, ref_model_rel, sol_default_model, sol_dense30_49_model, video_vae, audio_vae
    _free_vram_everywhere()

    out = {"exp3_shadow_text_kv_ref2va_origami_90f": exp3_results}
    (DATA_OUT / "stage_d_shadow_text_kv.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 Steering & Elicitation Experiments (Stages A-D)")
    ap.add_argument("--stage", required=True, choices=("de_slop", "exp1", "exp2", "exp3", "all"))
    args = ap.parse_args()

    try:
        if args.stage in ("de_slop", "all"):
            run_de_slop()
        if args.stage in ("exp1", "all"):
            run_exp1()
        if args.stage in ("exp2", "all"):
            run_exp2()
        if args.stage in ("exp3", "all"):
            run_exp3()
    finally:
        _free_vram_everywhere()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
