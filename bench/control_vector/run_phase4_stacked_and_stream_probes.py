#!/usr/bin/env python3
"""Phase 4 Clamped Stacking, Inverted Block-Band, 1x Q_aud Split, & H3-Image Parity Probes.

Holistic 4-showcase empirical suite (`--stage`):
  - `showcase_a` : Clamped union runtime stacking (`stack3` stills on `box_office` &
                   `sitcom_kitchen` + `90f` speaking 4-slider `1.00x` AV stack on
                   `t2va_box_office.txt`: `Base` vs. `Old Parallel Sum (λ_max=1.86)` vs.
                   `New Clamped Union (λ_max<=1.0)`).
  - `showcase_b` : Inverted shadow-text K/V band (`base` in main stream, tiny steered
                   text shadow `h_txt_steer` patched in-place into `blocks.20..49`) on
                   `90f` `<Picture 1>` `ref2va` `woman_blonde` across `noir +1.5` and
                   `felt_muppet +1.5` (`unsteered_base`, `all_0_49`, `inverted_20_49`).
  - `showcase_c` : `1x` `Q_aud -> K_txt_base` attention split vs. `2x` `streams="video_only"`
                   on `90f` `t2va_box_office.txt` (`de_llm_slop +1.15`), isolating 1-hop
                   (`Q_aud -> K_txt`) vs. 2-hop (`Q_aud -> K_vid`) audio attenuation and
                   scoring vocal pitch drift via `bench/probe_yin_f0_drift.py`.
  - `showcase_d` : `svd_ac_r8 [0..1]` vs. `svd_ac_r8_ew50 [0..0.50]` (`t=1` Pika `fl2va` on
                   `noir`, `paper_origami`, `de_llm_slop` at `s = 1.0, 1.25, 1.5`) PLUS
                   native `4.0 MP` `MiniMax-H3-Image` A/B (`paper_origami` + `de_llm_slop`
                   and `claymation_stopmotion` + `noir`) comparing single-concept native
                   rank-8 erasure vs. 2-slider clamped-union stacking via `MiniMaxH3Steer`.
  - `all`        : Runs Showcases A-D sequentially and unloads `:8188` cleanly on exit.

Outputs:
  - Media (`*.png`, `*.mp4`, `*.1fps.*.mp4`): `$H3_OUTPUT_DIR/{Image,Video}/phase4_2026-11-22/`
  - Metrics JSON: `data/exp_unpooled_sliders/phase4_2026_11_22/*.json`
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib as _importlib
import json
import os
from pathlib import Path
import subprocess
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
    import comfy.cli_args as _cli_args

    _cli_args.args.mmap_torch_files = True
    _cli_args.args.supports_fp8_compute = True
    _cli_args.args.cuda_malloc = True
    _cli_args.args.reserve_vram = 1.0
    _importlib.import_module("cuda_malloc")
    server_memory_mode()

import control_vector.eval_multivector_erasure as eval_mv  # noqa: E402
from _lib.frames import OneASecond  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import control_vectors as cv  # noqa: E402
import h3_spans  # noqa: E402
from workflows import h3_config  # noqa: E402

DATA_OUT = REPO_ROOT / "data" / "exp_unpooled_sliders" / "phase4_2026_11_22"
SEED_MAIN = eval_mv.SEED_MAIN
SEEDS_2 = eval_mv.SEEDS_3[:2]


def _phase4_out_dirs() -> tuple[Path, Path]:
    out_base = Path(os.environ["H3_OUTPUT_DIR"])
    img_dir = out_base / "Image" / "phase4_2026-11-22"
    vid_dir = out_base / "Video" / "phase4_2026-11-22"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "strips").mkdir(parents=True, exist_ok=True)
    return img_dir, vid_dir


def _comfy_bank_dir() -> Path:
    import control_vector_nodes as cv_nodes
    import folder_paths

    cand = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0]) / "bank"
    if cand.is_dir():
        return cand
    store = os.environ.get("H3_MODEL_STORE", "").strip()
    if store:
        return Path(store) / "control_vectors" / "bank"
    return cand


def _free_vram_everywhere() -> None:
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


def _write_one_a_second(vid_frames_u8: np.ndarray, stem_path: Path) -> list[dict[str, Any]]:
    n_f, h, w, _ = vid_frames_u8.shape
    writer = OneASecond(stem_path, (int(w), int(h)))
    for fi in range(n_f):
        writer.write(np.ascontiguousarray(vid_frames_u8[fi]).tobytes(), fi)
    code, parts = writer.close()
    if code != 0:
        raise RuntimeError(f"OneASecond failed ({code}) for {stem_path}")
    return parts


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


def _encode_prompt_dict(clip: Any, name: str, prompt_text: str) -> dict[str, Any]:
    import control_vector_nodes as cv_nodes

    enc_2d, tags_1d, tok_ids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, prompt_text)
    sp_m = h3_spans.spans(tok_ids, tags_1d.tolist(), dec_fn)
    return {
        "name": name,
        "prompt": prompt_text,
        "enc": enc_2d.detach().cpu().to(torch.float32),
        "tags": tags_1d.detach().cpu().to(torch.int64),
        "scene_mask": h3_spans.inference_mask(sp_m, "scene_only").detach().cpu(),
        "desc_mask": h3_spans.inference_mask(sp_m, "description_only").detach().cpu(),
        "sound_mask": h3_spans.inference_mask(sp_m, "soundscape_only").detach().cpu(),
    }


def _load_bank_op(stem: str, family: str = "fl2va") -> dict[str, torch.Tensor]:
    import control_vector_nodes as cv_nodes

    vp = cv_nodes.resolve_vector_file_path(stem, preferred_family=family)
    vt, _ = cv.load_vector(vp)
    return {
        "path": str(vp),
        "v1d": vt["vector.post_refiner"],
        "mu_d": vt.get("erasure.mu_d", vt["vector.post_refiner"]),
        "vr": vt.get("erasure.vr", torch.zeros((0, cv.POST_REFINER_DIM))),
        "diag": vt.get("erasure.diag", torch.zeros((0,))),
    }


def _load_pika_model_by_family(family: str, *, with_lora: bool = True, use_sol: bool = True):
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
) -> dict[str, Any]:
    from comfy_extras.nodes_audio import vae_decode_audio

    v_lat, a_lat = samples.unbind()
    v_lat = v_lat.detach().cpu()
    a_lat = a_lat.detach().cpu()
    del samples
    comfy_mm_gc()

    aud_obj = vae_decode_audio(audio_vae, {"samples": a_lat})
    aud_wav = aud_obj["waveform"].detach().cpu()
    aud_sr = int(aud_obj["sample_rate"])
    del aud_obj, a_lat
    comfy_mm_gc()

    vid_dec = video_vae.decode(v_lat)
    if vid_dec.ndim == 5:
        vid_dec = vid_dec[0]
    vid_u8 = vid_dec.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    del vid_dec, v_lat
    comfy_mm_gc()

    metrics = eval_mv.mux_and_measure_av(vid_u8, aud_wav, aud_sr, mp4_path, strip_path, fps=24)
    mid_start = min(18, vid_u8.shape[0] // 4)
    mid_end = max(mid_start + 1, vid_u8.shape[0] - mid_start)
    mid_sats = [eval_mv.hsv_sat(Image.fromarray(vid_u8[fi])) for fi in range(mid_start, mid_end, 4)]
    metrics["sat_mid_mean"] = round(float(np.mean(mid_sats)), 4)
    mid_im = Image.fromarray(vid_u8[vid_u8.shape[0] // 2])
    metrics["mid_laplacian_var"] = _image_quality_metrics(mid_im)["laplacian_var"]
    metrics["mp4"] = mp4_path.name
    metrics["strip"] = strip_path.name
    parts = _write_one_a_second(vid_u8, mp4_path.with_suffix(""))
    metrics["one_fps_files"] = [Path(p["file"]).name for p in parts]
    return metrics


def comfy_mm_gc() -> None:
    import comfy.model_management as comfy_mm

    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
    gen_s = round(time.perf_counter() - t0, 2)
    peak_gb = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
    del noise, guider
    res = _decode_and_mux_av(samples, video_vae, audio_vae, mp4_path, strip_path)
    res["gen_elapsed_s"] = gen_s
    res["peak_vram_gb"] = peak_gb
    return res


def _mod_scale_shift(x: torch.Tensor, shift_t: torch.Tensor, scale_t: torch.Tensor, mod_segments: list) -> torch.Tensor:
    out = x
    for sa, sb, m_row in mod_segments:
        if sb <= sa:
            continue
        out[sa:sb] = out[sa:sb] * (1.0 + scale_t[m_row]) + shift_t[m_row]
    return out


def _mod_gate(x: torch.Tensor, gate_t: torch.Tensor, upd: torch.Tensor, mod_segments: list) -> torch.Tensor:
    for sa, sb, m_row in mod_segments:
        if sb <= sa:
            continue
        x[sa:sb] = x[sa:sb] + gate_t[m_row] * upd[sa:sb]
    return x


def _slice_rope(rope_freqs: Any, tx_a: int, tx_b: int) -> Any:
    if rope_freqs is None:
        return None
    if rope_freqs.ndim >= 2 and rope_freqs.shape[1] >= tx_b:
        return rope_freqs[:, tx_a:tx_b]
    return rope_freqs[tx_a:tx_b]


def _face_vs_bg_metrics(vid_u8: np.ndarray, ref_woman: Image.Image) -> dict[str, float]:
    n_f, h, w, _ = vid_u8.shape
    cx0, cx1 = int(0.30 * w), int(0.70 * w)
    cy0, cy1 = int(0.12 * h), int(0.78 * h)
    ref_center = np.asarray(
        ref_woman.resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS).convert("HSV"),
        dtype=np.float32,
    )
    ref_sat = float(ref_center[..., 1].mean() / 255.0)
    c_sats, b_sats, b_laps, c_laps = [], [], [], []
    mask_c = np.zeros((h, w), dtype=bool)
    mask_c[cy0:cy1, cx0:cx1] = True
    for fi in range(0, n_f, 4):
        im = Image.fromarray(vid_u8[fi])
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
        rgb = np.asarray(im, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F) ** 2
        c_sats.append(float(hsv[mask_c].mean()))
        b_sats.append(float(hsv[~mask_c].mean()))
        b_laps.append(float(lap[~mask_c].mean()))
        c_laps.append(float(lap[mask_c].mean()))
    c_sat_m = float(np.mean(c_sats))
    return {
        "center_sat_mean": round(c_sat_m, 4),
        "center_vs_ref_sat_diff": round(abs(c_sat_m - ref_sat), 4),
        "border_sat_mean": round(float(np.mean(b_sats)), 4),
        "center_laplacian_energy": round(float(np.mean(c_laps)), 2),
        "border_laplacian_energy": round(float(np.mean(b_laps)), 2),
    }


# ---------------------------------------------------------------------------
# Showcase A: Clamped Union Runtime Stacking (Stills + 90f Speaking 4-Slider AV)
# ---------------------------------------------------------------------------
def run_showcase_a() -> dict[str, Any]:
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    from comfy.patcher_extension import WrappersMP
    import folder_paths
    import control_vector_nodes as cv_nodes

    h3_cond_mod = cv_nodes._repo("conditioning")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, vid_dir = _phase4_out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    eval_mv._wait_for_comfy_idle_and_free()
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")

    prompt_texts = {
        "box_office": (REPO_ROOT / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip(),
        "sitcom_kitchen": (REPO_ROOT / "prompt_bank" / "t2va_sitcom_kitchen.txt").read_text(encoding="utf-8").strip(),
    }
    prompts = {k: _encode_prompt_dict(clip, k, v) for k, v in prompt_texts.items()}

    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    cond_box_90, lat_box_90, *_ = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip,
        vae=video_vae,
        prompt=prompt_texts["box_office"],
        width=1152,
        height=768,
        length=90,
        single_frame=False,
    )
    del clip
    _free_vram_everywhere()

    op_noir = _load_bank_op("noir", "fl2va")
    op_chaos = _load_bank_op("chaos_panic", "fl2va")
    op_deslop = _load_bank_op("de_llm_slop", "fl2va")

    # Verify spectral eigenvalue drop from old parallel sum vs clamped union before rendering
    w_cat = torch.cat(
        [
            torch.sqrt(op["diag"].double().clamp(0.0, 1.0) * min(s_v, 1.0)).unsqueeze(-1) * op["vr"].double()
            for op, s_v in ((op_noir, 1.25), (op_chaos, 1.0), (op_deslop, 0.9))
        ],
        dim=0,
    )
    old_eigs = (torch.linalg.svdvals(w_cat) ** 2).tolist()
    n_over = sum(1 for e in old_eigs if e > 1.0)
    print(f"[Showcase A] Old parallel sum top eigenvalues={ [round(x, 3) for x in old_eigs[:7]] } (n > 1.0: {n_over})")

    # Part 1: t=1 Stills (Base vs Single noir vs Old Parallel Sum vs New Clamped Union)
    model, img_vae = eval_mv._load_pika_fl2va_model(with_lora=True)

    still_arms = ("Base", "Single noir +1.25", "Old Parallel Sum (λ_max=1.86)", "New Clamped Union (λ_max<=1.0)")
    still_records: list[dict[str, Any]] = []
    sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []

    stack_triples = [
        (op_noir, 1.25),
        (op_chaos, 1.0),
        (op_deslop, 0.9),
    ]

    for pn in ("box_office", "sitcom_kitchen"):
        pr = prompts[pn]
        enc_3d = pr["enc"].to(torch.bfloat16).unsqueeze(0)
        tags_2d = pr["tags"].unsqueeze(0)
        sm = pr["scene_mask"]
        for sd in SEEDS_2:
            cols = []
            for arm_lbl in still_arms:
                m_run = model
                if arm_lbl != "Base":
                    m_run = model.clone()

                    def _dm_wrap(executor, x, timestep, context, transformer_options={}, mode_lbl=arm_lbl, **kwargs):
                        ctx_use = context
                        if context is not None:
                            ctx_ref = eval_mv._refine_context_if_needed(executor, context, transformer_options)
                            if mode_lbl == "Single noir +1.25":
                                ctx_use = cv.apply_post_refiner(
                                    ctx_ref,
                                    tags_2d.to(device=ctx_ref.device),
                                    [(op_noir["v1d"], 1.25, "add_norm_matched_erasure")],
                                    mask=sm.unsqueeze(0).to(device=ctx_ref.device),
                                    erasure_specs=[(op_noir["vr"], op_noir["diag"], op_noir["mu_d"])],
                                    warn_if_stacked_norm=False,
                                )
                            elif mode_lbl.startswith("Old Parallel"):
                                ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                                rows = ref64[sm]
                                tot = torch.zeros_like(rows)
                                for op_i, st_i in stack_triples:
                                    tot = tot + cv._erasure_row_shift64(
                                        rows,
                                        op_i["v1d"],
                                        st_i,
                                        (op_i["vr"], op_i["diag"], op_i["mu_d"]),
                                    )
                                out = ref64.clone()
                                out[sm] = rows + tot
                                ctx_use = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
                            else:
                                ctx_use = cv.apply_post_refiner(
                                    ctx_ref,
                                    tags_2d.to(device=ctx_ref.device),
                                    [(op_i["v1d"], st_i, "add_norm_matched_erasure") for op_i, st_i in stack_triples],
                                    mask=sm.unsqueeze(0).to(device=ctx_ref.device),
                                    erasure_specs=[(op_i["vr"], op_i["diag"], op_i["mu_d"]) for op_i, _ in stack_triples],
                                    warn_if_stacked_norm=False,
                                )
                        return executor(x, timestep, ctx_use, transformer_options, **kwargs)

                    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["phase4_still"] = [_dm_wrap]

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
                still_records.append({"prompt": pn, "seed": sd, "arm": arm_lbl, **qm})
                cols.append((f"{arm_lbl[:18]} S={qm['hsv_sat']:.2f} L={qm['laplacian_var']:.0f}", im))
                print(f"  [Showcase A Still] {pn} s={sd} {arm_lbl}: sat={qm['hsv_sat']:.4f}, lap={qm['laplacian_var']:.1f}")
            sheet_rows.append((f"{pn[:12]} s={sd}", cols))

    sheet_p = img_dir / "showcase_a_stack3_stills.png"
    eval_mv.make_contact_sheet(sheet_rows, sheet_p)
    del model, img_vae
    _free_vram_everywhere()

    # Part 2: 90f speaking 4-slider 1.00x AV comparison on t2va_box_office.txt
    av_model = _load_pika_model_by_family("fl2va", with_lora=True, use_sol=True)
    op_reverb = _load_bank_op("reverb_cathedral", "fl2va")
    box_sm = prompts["box_office"]["scene_mask"]
    box_snd = prompts["box_office"]["sound_mask"]
    box_tags = prompts["box_office"]["tags"].unsqueeze(0)

    av_records: list[dict[str, Any]] = []
    for av_lbl in ("base", "old_parallel_4slider", "new_clamped_4slider"):
        if av_lbl == "base":
            m_run = av_model
        elif av_lbl == "old_parallel_4slider":
            m_run = av_model.clone()

            def _dm_old4(executor, x, timestep, context, transformer_options={}, **kwargs):
                ctx_use = context
                if context is not None:
                    ctx_ref = eval_mv._refine_context_if_needed(executor, context, transformer_options)
                    ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                    rows_s = ref64[box_sm]
                    tot_s = torch.zeros_like(rows_s)
                    for op_i, st_i in stack_triples:
                        tot_s = tot_s + cv._erasure_row_shift64(
                            rows_s, op_i["v1d"], st_i, (op_i["vr"], op_i["diag"], op_i["mu_d"])
                        )
                    out = ref64.clone()
                    out[box_sm] = rows_s + tot_s
                    rows_a = ref64[box_snd]
                    out[box_snd] = rows_a + cv._erasure_row_shift64(
                        rows_a, op_reverb["v1d"], 1.25, (op_reverb["vr"], op_reverb["diag"], op_reverb["mu_d"])
                    )
                    ctx_use = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
                return executor(x, timestep, ctx_use, transformer_options, **kwargs)

            m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["phase4_old4"] = [_dm_old4]
        else:
            m_run = av_model
            for st_stem, st_val, tgt_mask in (
                ("noir", 1.25, "scene_only"),
                ("chaos_panic", 1.0, "scene_only"),
                ("de_llm_slop", 0.9, "scene_only"),
                ("reverb_cathedral", 1.25, "soundscape_only"),
            ):
                m_run = cv_nodes.attach_control_vector(
                    m_run,
                    st_stem,
                    strength=st_val,
                    mode="add_norm_matched_erasure",
                    mask=tgt_mask,
                )

        mp4_p = vid_dir / f"showcase_a_4slider_av_{av_lbl}.mp4"
        strip_p = vid_dir / "strips" / f"showcase_a_4slider_av_{av_lbl}.png"
        m_av = _sample_av_with_model(
            m_run, cond_box_90, lat_box_90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
        )
        m_av["arm"] = av_lbl
        av_records.append(m_av)
        print(
            f"  [Showcase A 90f AV] {av_lbl}: sat={m_av['sat_mean']:.4f}, "
            f"lap={m_av['mid_laplacian_var']:.1f}, dbfs={m_av['audio_dbfs']}, "
            f"centroid={m_av['spectral_centroid_hz']}Hz, tail={m_av['tail_sustain_ratio']}"
        )

    del av_model, video_vae, audio_vae
    _free_vram_everywhere()

    out = {
        "parallel_sum_top_eigs": [round(x, 4) for x in old_eigs[:12]],
        "parallel_sum_n_gt_1": int(n_over),
        "stills": still_records,
        "av_4slider_box_office_90f": av_records,
        "sheet": sheet_p.name,
    }
    (DATA_OUT / "showcase_a_clamped_stacking.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Showcase B: Inverted Shadow blocks.20..49 on 90f ref2va woman_blonde
# ---------------------------------------------------------------------------
def _attach_inverted_shadow_kv_band(
    model: Any,
    *,
    tags_1d: torch.Tensor,
    text_mask: torch.Tensor,
    text_op: dict[str, torch.Tensor] | None,
    text_strength: float,
    active_blocks: frozenset[int],
    run_sanity_check: bool = True,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Keep unsteered `h_txt_base` in the main DiT stream (`0` overhead in blocks 0..19) and
    patch text K/V slice `[tx_a:tx_b]` in-place from a tiny steered text shadow in `active_blocks`."""
    import comfy.model_management as comfy_mm
    import comfy.model_prefetch as comfy_mp
    import comfy.quant_ops
    from comfy.patcher_extension import WrappersMP

    m_run = model.clone()
    dm = m_run.get_model_object("diffusion_model")
    heads = int(dm.blocks[0].attn.heads)
    head_dim = int(dm.blocks[0].attn.head_dim)
    state: dict[str, Any] = {
        "h_txt_steer": None,
        "step_kv_by_block": {},
        "ctx_cache": {},
        "sanity_max_block0_diff": None,
        "kv_swaps": 0,
    }

    def _make_pre_hook(b_idx: int):
        def _pre_hook(module, args, kwargs):
            h_s = state.get("h_txt_steer")
            if h_s is None:
                return
            x_full, t_emb, mod_segments, rope_freqs = args[0], args[1], args[2], args[3]
            to = kwargs.get("transformer_options") or {}
            layout = to.get("minimax_h3_layout")
            if layout is None:
                return
            tx_a, tx_b, _ = next(s for s in layout.segments if s[2] == "text")
            if b_idx == 0 and run_sanity_check and state["sanity_max_block0_diff"] is None:
                state["sanity_max_block0_diff"] = float((x_full[tx_a:tx_b] - state["ctx_base_0"]).abs().max().item())

            mod_txt = [(sa - tx_a, sb - tx_a, m_row) for (sa, sb, m_row) in mod_segments if sa >= tx_a and sb <= tx_b]
            rope_s = _slice_rope(rope_freqs, tx_a, tx_b)
            l_txt = tx_b - tx_a

            with comfy_mp.pause_malloc_graph(), torch.no_grad():
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = module.adaln_proj(t_emb)
                h_in = _mod_scale_shift(module.norm1(h_s.clone()), shift_msa, scale_msa, mod_txt)
                q_s, k_s, v_s = module.attn.qkv_proj(h_in).split(heads * head_dim, dim=-1)
                v_s = v_s.view(l_txt, heads, head_dim)
                if rope_s is not None:
                    q_r = q_s.view(1, l_txt, heads, head_dim)
                    k_r = k_s.view(1, l_txt, heads, head_dim)
                    qw = comfy_mm.cast_to(module.attn.q_norm.weight, device=h_in.device)
                    kw = comfy_mm.cast_to(module.attn.k_norm.weight, device=h_in.device)
                    q_r, k_r = comfy.quant_ops.ck.rms_rope_split_half(
                        q_r, k_r, rope_s, qw, kw, epsilon=module.attn.q_norm.eps, rot_dim=rope_s.shape[-3] * 2
                    )
                    q_s, k_s = q_r[0], k_r[0]
                else:
                    q_s = module.attn.q_norm(q_s.view(l_txt, heads, head_dim))
                    k_s = module.attn.k_norm(k_s.view(l_txt, heads, head_dim))

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
                    "q_s": q_s.transpose(0, 1).unsqueeze(0).contiguous(),
                    "k_s": k_s.transpose(0, 1).unsqueeze(0).contiguous(),
                    "v_s": v_s.transpose(0, 1).unsqueeze(0).contiguous(),
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
                with comfy_mp.pause_malloc_graph(), torch.no_grad():
                    ks_cast = info["k_s"].to(device=k_peek.device, dtype=k_peek.dtype)
                    vs_cast = info["v_s"].to(device=v_peek.device, dtype=v_peek.dtype)
                    k_other = [k_peek[:, :, :tx_a], ks_cast, k_peek[:, :, tx_b:]] if tx_a > 0 else [ks_cast, k_peek[:, :, tx_b:]]
                    v_other = [v_peek[:, :, :tx_a], vs_cast, v_peek[:, :, tx_b:]] if tx_a > 0 else [vs_cast, v_peek[:, :, tx_b:]]
                    qs_cast = info["q_s"].to(device=k_peek.device, dtype=k_peek.dtype)
                    attn_s = F.scaled_dot_product_attention(
                        qs_cast, torch.cat(k_other, dim=2), torch.cat(v_other, dim=2), dropout_p=0.0, is_causal=False
                    )
                    del k_other, v_other
                    attn_s_2d = attn_s[0].transpose(0, 1).reshape(l_txt, n_heads * head_dim)
                    blk = info["module"]
                    x_mid = _mod_gate(state["h_txt_steer"].clone(), info["gate_msa"], blk.attn.out_proj(attn_s_2d), info["mod_txt"])
                    h_mlp_in = _mod_scale_shift(blk.norm2(x_mid.clone()), info["shift_mlp"], info["scale_mlp"], info["mod_txt"])
                    state["h_txt_steer"] = _mod_gate(x_mid, info["gate_mlp"], blk.mlp(h_mlp_in), info["mod_txt"])

                if int(b_idx) in active_blocks:
                    state["kv_swaps"] += 1
                    if held:
                        box = type(q)
                        q_t, k_t, v_t = q.take(), k.take(), v.take()
                        with comfy_mp.pause_malloc_graph():
                            k_t[:, :, tx_a:tx_b] = ks_cast
                            v_t[:, :, tx_a:tx_b] = vs_cast
                        return previous.container_function(
                            box(q_t), box(k_t), box(v_t), n_heads, mask=mask, attn_precision=attn_precision,
                            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
                        )
                    with comfy_mp.pause_malloc_graph():
                        k_peek[:, :, tx_a:tx_b] = ks_cast
                        v_peek[:, :, tx_a:tx_b] = vs_cast
                    return _pass_through(q, k_peek, v_peek)

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
            base_cpu = ctx_ref[0].detach().cpu().clone()
            if text_op is not None and abs(text_strength) > 1e-9:
                steered_3d_cpu = cv.apply_post_refiner(
                    base_cpu.unsqueeze(0),
                    tags_1d.unsqueeze(0).cpu(),
                    [(text_op["v1d"], text_strength, "add_norm_matched_erasure")],
                    mask=text_mask.unsqueeze(0).cpu(),
                    erasure_specs=[(text_op["vr"], text_op["diag"], text_op["mu_d"])],
                    warn_if_stacked_norm=False,
                )
                steered_cpu = steered_3d_cpu[0].cpu()
            else:
                steered_cpu = base_cpu.clone()
            state["ctx_cache"][ck] = (base_cpu.unsqueeze(0), steered_cpu)

        base_3d_cpu, steered_cpu = state["ctx_cache"][ck]
        with comfy_mp.pause_malloc_graph():
            base_3d = base_3d_cpu.to(device=ctx_ref.device, dtype=ctx_ref.dtype)
            state["ctx_base_0"] = base_3d[0]
            state["h_txt_steer"] = steered_cpu.to(device=ctx_ref.device, dtype=ctx_ref.dtype).clone()
            state["step_kv_by_block"].clear()

        to_next = dict(transformer_options or {})
        prev_override = to_next.get("optimized_attention_override")
        to_next["optimized_attention_override"] = _make_shadow_override(prev_override)
        try:
            return executor(x, timestep, base_3d, to_next, **kwargs)
        finally:
            state["h_txt_steer"] = None
            state["ctx_base_0"] = None
            state["step_kv_by_block"].clear()

    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["phase4_inv_shadow"] = [_dm_wrap]
    return m_run, hook_handles, state


def run_showcase_b() -> dict[str, Any]:
    import comfy.cli_args as _ca
    import comfy.sd
    import comfy.utils
    import folder_paths
    import control_vector_nodes as cv_nodes

    ref_mod = cv_nodes._repo("reference_conditioning")
    prev_cc = _ca.args.disable_comfy_compiler
    _ca.args.disable_comfy_compiler = True
    try:
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        _, vid_dir = _phase4_out_dirs()
        inp_dir = Path(os.environ["H3_INPUT_DIR"])
        pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

        woman_txt = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
        img_woman_pil = Image.open(inp_dir / "woman_blonde.jpg").convert("RGB")
        img_woman = _load_comfy_image_tensor(inp_dir / "woman_blonde.jpg")

        eval_mv._wait_for_comfy_idle_and_free()
        h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
        (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
        vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
        video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
        avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
        audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

        (refs_w,) = ref_mod.MiniMaxH3AppendRefImage.execute(
            image=img_woman, size_policy="max", qwen_view="separate", references=None, use_vae=True
        )
        cond_ref90, lat_ref90, *_ = ref_mod.MiniMaxH3ReferenceConditioning.execute(
            clip=clip, references=refs_w, prompt=woman_txt, width=1152, height=768, length=90,
            video_policy="comfy", image_policy="comfy", vae=video_vae, audio_vae=audio_vae, single_frame=False
        )
        tags_ref = cond_ref90[0][1]["minimax_token_tags"].reshape(-1).cpu()
        masks_ref = cond_ref90[0][1]["h3_span_masks"]
        ref_desc_mask = h3_spans.inference_mask(masks_ref, "description_only").reshape(-1).cpu().bool()
        del clip
        _free_vram_everywhere()

        op_noir = _load_bank_op("noir", "ref2va")
        op_felt = _load_bank_op("felt_muppet", "ref2va")

        ref_model = _load_pika_model_by_family("ref2va", with_lora=True, use_sol=True)
        (ref_model_rel,) = cv_nodes.MiniMaxH3ReleaseReferences.execute(
            model=ref_model, end_percent=0.50, clean_level=0.999, video_end_percent=0.50
        )

        arms = [
            ("unsteered_base", "none", None, 0.0, frozenset()),
            ("noir_all_0_49", "noir", op_noir, 1.5, frozenset(range(50))),
            ("noir_inverted_20_49", "noir", op_noir, 1.5, frozenset(range(20, 50))),
            ("felt_muppet_all_0_49", "felt_muppet", op_felt, 1.5, frozenset(range(50))),
            ("felt_muppet_inverted_20_49", "felt_muppet", op_felt, 1.5, frozenset(range(20, 50))),
        ]
        results: list[dict[str, Any]] = []
        for arm_lbl, vec_stem, op_u, st_val, active_blks in arms:
            m_run, handles, st = _attach_inverted_shadow_kv_band(
                ref_model_rel,
                tags_1d=tags_ref,
                text_mask=ref_desc_mask,
                text_op=op_u,
                text_strength=st_val,
                active_blocks=active_blks,
                run_sanity_check=True,
            )
            mp4_p = vid_dir / f"showcase_b_inv_shadow_{arm_lbl}.mp4"
            strip_p = vid_dir / "strips" / f"showcase_b_inv_shadow_{arm_lbl}.png"
            try:
                m_res = _sample_av_with_model(
                    m_run, cond_ref90, lat_ref90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
                )
            finally:
                for hh in handles:
                    hh.remove()
            vid_u8 = _read_mp4_frames_u8(mp4_p)
            m_res.update(_face_vs_bg_metrics(vid_u8, img_woman_pil))
            m_res["arm"] = arm_lbl
            m_res["concept"] = vec_stem
            m_res["strength"] = st_val
            m_res["kv_swaps"] = st["kv_swaps"]
            m_res["sanity_block0_diff"] = st["sanity_max_block0_diff"]
            results.append(m_res)
            print(
                f"  [Showcase B] {arm_lbl}: sanity={m_res['sanity_block0_diff']}, swaps={st['kv_swaps']}, "
                f"sat={m_res['sat_mean']:.4f}, center_diff={m_res['center_vs_ref_sat_diff']:.4f}, "
                f"border_lap={m_res['border_laplacian_energy']}, elapsed={m_res['gen_elapsed_s']}s, "
                f"vram={m_res['peak_vram_gb']}GB"
            )

        del ref_model, ref_model_rel, video_vae, audio_vae
        _free_vram_everywhere()

        out = {"showcase_b_inverted_shadow_ref2va_90f": results}
        (DATA_OUT / "showcase_b_inverted_blocks20.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    finally:
        _ca.args.disable_comfy_compiler = prev_cc


# ---------------------------------------------------------------------------
# Showcase C: 1x Q_aud -> K_txt_base Split vs. 2x streams="video_only" + YIN F0
# ---------------------------------------------------------------------------
def _attach_1x_qaud_unsteered_ktxt(
    model: Any,
    *,
    tags_1d: torch.Tensor,
    text_mask: torch.Tensor,
    text_op: dict[str, torch.Tensor],
    text_strength: float,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Run main DiT with steered text (video sees steered text at 1.00x) while splicing ONLY
    `audio` query attention output from exact SDPA against `[K_txt_base, K_non_txt_steered]`."""
    import comfy.model_management as comfy_mm
    import comfy.model_prefetch as comfy_mp
    import comfy.quant_ops
    from comfy.patcher_extension import WrappersMP

    m_run = model.clone()
    dm = m_run.get_model_object("diffusion_model")
    heads = int(dm.blocks[0].attn.heads)
    head_dim = int(dm.blocks[0].attn.head_dim)
    state: dict[str, Any] = {
        "h_txt_base": None,
        "step_kv_by_block": {},
        "ctx_cache": {},
        "aud_splices": 0,
    }

    def _make_pre_hook(b_idx: int):
        def _pre_hook(module, args, kwargs):
            h_b = state.get("h_txt_base")
            if h_b is None:
                return
            t_emb, mod_segments, rope_freqs = args[1], args[2], args[3]
            to = kwargs.get("transformer_options") or {}
            layout = to.get("minimax_h3_layout")
            if layout is None:
                return
            tx_a, tx_b, _ = next(s for s in layout.segments if s[2] == "text")
            au_spans = [(sa, sb) for (sa, sb, kind) in layout.segments if kind == "audio"]
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
                    "au_spans": au_spans,
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

    def _make_override(previous):
        def override(func, q, k, v, n_heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
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
                q_peek = q if torch.is_tensor(q) else q.peek()
                k_peek = k if torch.is_tensor(k) else k.peek()
                v_peek = v if torch.is_tensor(v) else v.peek()
                tx_a, tx_b, l_txt = info["tx_a"], info["tx_b"], info["l_txt"]
                au_spans = info["au_spans"]
                aud_2d = None
                with comfy_mp.pause_malloc_graph(), torch.no_grad():
                    kb_cast = info["k_b"].to(device=k_peek.device, dtype=k_peek.dtype)
                    vb_cast = info["v_b"].to(device=v_peek.device, dtype=v_peek.dtype)
                    k_alt = torch.cat(
                        [k_peek[:, :, :tx_a], kb_cast, k_peek[:, :, tx_b:]] if tx_a > 0 else [kb_cast, k_peek[:, :, tx_b:]],
                        dim=2,
                    )
                    v_alt = torch.cat(
                        [v_peek[:, :, :tx_a], vb_cast, v_peek[:, :, tx_b:]] if tx_a > 0 else [vb_cast, v_peek[:, :, tx_b:]],
                        dim=2,
                    )
                    qb_cast = info["q_b"].to(device=k_peek.device, dtype=k_peek.dtype)
                    attn_b = F.scaled_dot_product_attention(qb_cast, k_alt, v_alt, dropout_p=0.0, is_causal=False)
                    attn_b_2d = attn_b[0].transpose(0, 1).reshape(l_txt, n_heads * head_dim)
                    blk = info["module"]
                    x_mid = _mod_gate(state["h_txt_base"].clone(), info["gate_msa"], blk.attn.out_proj(attn_b_2d), info["mod_txt"])
                    h_mlp_in = _mod_scale_shift(blk.norm2(x_mid.clone()), info["shift_mlp"], info["scale_mlp"], info["mod_txt"])
                    state["h_txt_base"] = _mod_gate(x_mid, info["gate_mlp"], blk.mlp(h_mlp_in), info["mod_txt"])

                    if au_spans:
                        q_aud = torch.cat([q_peek[:, :, a:b] for a, b in au_spans], dim=2)
                        aud_attn = F.scaled_dot_product_attention(q_aud, k_alt, v_alt, dropout_p=0.0, is_causal=False)
                        aud_2d = aud_attn.transpose(1, 2).reshape(1, -1, n_heads * head_dim).contiguous()
                        del q_aud, aud_attn
                    del k_alt, v_alt
                    del q_peek, k_peek, v_peek

                out_main = _pass_through(q, k, v)
                if aud_2d is not None and not skip_output_reshape:
                    state["aud_splices"] += 1
                    off = 0
                    aud_cast = aud_2d.to(dtype=out_main.dtype)
                    for a, b in au_spans:
                        n_seg = b - a
                        out_main[:, a:b] = aud_cast[:, off : off + n_seg]
                        off += n_seg
                return out_main

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
            base_cpu = ctx_ref[0].detach().cpu().clone()
            steered_3d_cpu = cv.apply_post_refiner(
                base_cpu.unsqueeze(0),
                tags_1d.unsqueeze(0).cpu(),
                [(text_op["v1d"], text_strength, "add_norm_matched_erasure")],
                mask=text_mask.unsqueeze(0).cpu(),
                erasure_specs=[(text_op["vr"], text_op["diag"], text_op["mu_d"])],
                warn_if_stacked_norm=False,
            )
            state["ctx_cache"][ck] = (base_cpu, steered_3d_cpu[0].cpu())

        base_cpu, steered_cpu = state["ctx_cache"][ck]
        with comfy_mp.pause_malloc_graph():
            steered_3d = steered_cpu.unsqueeze(0).to(device=ctx_ref.device, dtype=ctx_ref.dtype)
            state["h_txt_base"] = base_cpu.to(device=ctx_ref.device, dtype=ctx_ref.dtype).clone()
            state["step_kv_by_block"].clear()

        to_next = dict(transformer_options or {})
        prev_override = to_next.get("optimized_attention_override")
        to_next["optimized_attention_override"] = _make_override(prev_override)
        try:
            return executor(x, timestep, steered_3d, to_next, **kwargs)
        finally:
            state["h_txt_base"] = None
            state["step_kv_by_block"].clear()

    m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["phase4_qaud_1x"] = [_dm_wrap]
    return m_run, hook_handles, state


def run_showcase_c() -> dict[str, Any]:
    import comfy.cli_args as _ca
    import comfy.sd
    import comfy.utils
    import folder_paths
    import control_vector_nodes as cv_nodes

    h3_cond_mod = cv_nodes._repo("conditioning")
    prev_cc = _ca.args.disable_comfy_compiler
    _ca.args.disable_comfy_compiler = True
    try:
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        _, vid_dir = _phase4_out_dirs()
        pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

        box_txt = (REPO_ROOT / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip()

        eval_mv._wait_for_comfy_idle_and_free()
        h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
        (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
        vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
        video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
        avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
        audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

        cond_box_90, lat_box_90, *_ = h3_cond_mod.MiniMaxH3Conditioning.execute(
            clip=clip, vae=video_vae, prompt=box_txt, width=1152, height=768, length=90, single_frame=False
        )
        pr_box = _encode_prompt_dict(clip, "box_office", box_txt)
        del clip
        _free_vram_everywhere()

        op_deslop = _load_bank_op("de_llm_slop", "fl2va")
        av_model = _load_pika_model_by_family("fl2va", with_lora=True, use_sol=True)

        records: list[dict[str, Any]] = []
        base_f0 = None
        speech_mask = None
        voiced_spans_inclusive: list[list[int]] = []

        for arm_lbl in ("base", "steered_both_1x", "steered_1x_qaud_ktxt_base", "steered_2x_video_only"):
            handles: list[Any] = []
            st: dict[str, Any] = {}
            if arm_lbl == "base":
                m_run = av_model
            elif arm_lbl == "steered_both_1x":
                (m_run,) = cv_nodes.MiniMaxH3Steer.execute(
                    model=av_model,
                    vector="de_llm_slop",
                    strength=1.15,
                    mode="add_norm_matched_erasure",
                    target="scene_only",
                    streams="both",
                    allow_recipe_mismatch=True,
                )
            elif arm_lbl == "steered_2x_video_only":
                (m_run,) = cv_nodes.MiniMaxH3Steer.execute(
                    model=av_model,
                    vector="de_llm_slop",
                    strength=1.15,
                    mode="add_norm_matched_erasure",
                    target="scene_only",
                    streams="video_only",
                    allow_recipe_mismatch=True,
                )
            else:
                m_run, handles, st = _attach_1x_qaud_unsteered_ktxt(
                    av_model,
                    tags_1d=pr_box["tags"],
                    text_mask=pr_box["scene_mask"],
                    text_op=op_deslop,
                    text_strength=1.15,
                )

            mp4_p = vid_dir / f"showcase_c_qaud_{arm_lbl}.mp4"
            strip_p = vid_dir / "strips" / f"showcase_c_qaud_{arm_lbl}.png"
            try:
                m_av = _sample_av_with_model(
                    m_run, cond_box_90, lat_box_90, SEED_MAIN, pika_sigmas, video_vae, audio_vae, mp4_p, strip_p
                )
            finally:
                for hh in handles:
                    hh.remove()

            m_av["arm"] = arm_lbl
            m_av["aud_splices"] = st.get("aud_splices", 0)
            f0_arm = yin_probe._extract_f0(
                mp4_p, sr=yin_probe.DEFAULT_SR, threshold=yin_probe.DEFAULT_THRESHOLD, min_dbfs=yin_probe.DEFAULT_MIN_DBFS
            )
            if arm_lbl == "base":
                base_f0 = f0_arm
                vs_dir = DATA_OUT / "voice_spans_box_office_base"
                subprocess.run(
                    [sys.executable, str(REPO_ROOT / "bench" / "voice_spans.py"), "separate", str(mp4_p), str(vs_dir)],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "bench" / "voice_spans.py"),
                        "spans",
                        str(vs_dir),
                        "--fps",
                        "24",
                        "--frames",
                        "90",
                    ],
                    check=True,
                )
                vs_json = vs_dir / "voice_spans.json"
                vs_doc = json.loads(vs_json.read_text(encoding="utf-8"))
                voiced_spans_inclusive = list(vs_doc.get("voiced_spans_inclusive") or [])
                speech_mask = yin_probe.load_voice_span_mask(
                    vs_json, int(base_f0.numel()), sr=yin_probe.DEFAULT_SR
                )
                m_av["voiced_spans_inclusive"] = voiced_spans_inclusive
                m_av["yin_f0_summary_speech"] = yin_probe.summarize_f0_series(f0_arm, span_mask=speech_mask)
                m_av["yin_f0_summary_all"] = yin_probe.summarize_f0_series(f0_arm, span_mask=None)
            elif base_f0 is not None:
                m_av["yin_f0_vs_base_speech"] = yin_probe.compare_f0_drift(base_f0, f0_arm, span_mask=speech_mask)
                m_av["yin_f0_vs_base_all"] = yin_probe.compare_f0_drift(base_f0, f0_arm, span_mask=None)

            records.append(m_av)
            f0_speech = (m_av.get("yin_f0_vs_base_speech") or {}).get("median_shift_semitones", 0.0)
            f0_all = (m_av.get("yin_f0_vs_base_all") or {}).get("median_shift_semitones", 0.0)
            print(
                f"  [Showcase C] {arm_lbl}: elapsed={m_av['gen_elapsed_s']}s, dbfs={m_av['audio_dbfs']}, "
                f"sat={m_av['sat_mean']:.4f}, lap={m_av['mid_laplacian_var']:.1f}, "
                f"f0_speech={f0_speech}st, f0_all={f0_all}st, splices={m_av['aud_splices']}"
            )

        del av_model, video_vae, audio_vae
        _free_vram_everywhere()

        out = {
            "showcase_c_qaud_1x_vs_2x_box_office_90f": records,
            "voiced_spans_inclusive": voiced_spans_inclusive,
        }
        (DATA_OUT / "showcase_c_qaud_audio_split.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    finally:
        _ca.args.disable_comfy_compiler = prev_cc


# ---------------------------------------------------------------------------
# Showcase D: svd_ac_r8_ew50 + Native 4.0 MP MiniMax-H3-Image Parity
# ---------------------------------------------------------------------------
def run_showcase_d() -> dict[str, Any]:
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning, MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    import control_vector_nodes as cv_nodes

    h3_img_nodes = cv_nodes._repo("h3_image_nodes")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, _ = _phase4_out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    # Part 1: svd_ac_r8 [0..1] vs svd_ac_r8_ew50 [0..0.50] across noir, paper_origami, de_llm_slop
    eval_mv._wait_for_comfy_idle_and_free()
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")

    ew_scenes = {
        "box_office": (REPO_ROOT / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip(),
        "sitcom_kitchen": (REPO_ROOT / "prompt_bank" / "t2va_sitcom_kitchen.txt").read_text(encoding="utf-8").strip(),
        "freight_elevator": (REPO_ROOT / "prompt_bank" / "t2va_freight_elevator.txt").read_text(encoding="utf-8").strip(),
    }
    ew_prompts = {k: _encode_prompt_dict(clip, k, v) for k, v in ew_scenes.items()}
    del clip
    _free_vram_everywhere()

    fl_ops = {cn: _load_bank_op(cn, "fl2va") for cn in ("noir", "paper_origami", "de_llm_slop")}
    model_fl, img_vae = eval_mv._load_pika_fl2va_model(with_lora=True)

    ew_cells = [
        ("noir", "box_office"),
        ("paper_origami", "sitcom_kitchen"),
        ("de_llm_slop", "freight_elevator"),
    ]
    ew_strengths = (1.0, 1.25, 1.5)
    ew_records: list[dict[str, Any]] = []
    ew_sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []

    for cn, pn in ew_cells:
        pr = ew_prompts[pn]
        op_u = fl_ops[cn]
        for sd in SEEDS_2:
            cols = []
            # Base column once per row
            im_b = eval_mv._sample_still_pika(
                model_fl, img_vae, pr, op=None, strength=0.0, seed=sd, sigmas=pika_sigmas
            )
            qm_b = _image_quality_metrics(im_b)
            ew_records.append({"concept": cn, "prompt": pn, "seed": sd, "variant": "base", "strength": 0.0, "erase_window": [0.0, 1.0], **qm_b})
            cols.append((f"base S={qm_b['hsv_sat']:.2f} L={qm_b['laplacian_var']:.0f}", im_b))
            for st_v in ew_strengths:
                for ew_lbl, ew_win in (("r8_full", (0.0, 1.0)), ("r8_ew50", (0.0, 0.50))):
                    im = eval_mv._sample_still_pika(
                        model_fl, img_vae, pr, op=op_u, strength=st_v, seed=sd, sigmas=pika_sigmas, erase_window=ew_win
                    )
                    qm = _image_quality_metrics(im)
                    ew_records.append({
                        "concept": cn, "prompt": pn, "seed": sd, "variant": ew_lbl, "strength": st_v,
                        "erase_window": list(ew_win), **qm
                    })
                    cols.append((f"{ew_lbl} s={st_v:.2g} S={qm['hsv_sat']:.2f} L={qm['laplacian_var']:.0f}", im))
            ew_sheet_rows.append((f"{cn[:9]}/{pn[:8]} s={sd}", cols))

    ew_sheet_p = img_dir / "showcase_d_svd_ac_ew50.png"
    eval_mv.make_contact_sheet(ew_sheet_rows, ew_sheet_p)
    del model_fl, img_vae
    _free_vram_everywhere()

    # Part 2: Native 4.0 MP MiniMax-H3-Image Rank-8 Erasure & 2-Slider Clamped Union Stacking
    bank_dir = _comfy_bank_dir()
    h3i_sigmas = torch.tensor([float(x.strip()) for x in h3_config.H3_IMAGE_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    (h3i_w, h3i_h, _, _) = h3_img_nodes.MiniMaxH3ImageSize.execute(
        megapixels=4.0, shape="3:2", generate_megapixels=4.0
    ).args

    t2i_clockmaker = (REPO_ROOT / "prompt_bank" / "p_h3_image" / "t2i_clockmaker.txt").read_text(encoding="utf-8").strip()
    t2i_lighthouse = (
        "An atmospheric editorial photograph of a stone lighthouse watchroom high above churning North Atlantic waves at dusk. "
        "Inside the circular whitewashed brick room with tall salt-crusted mullioned windows, an oilskin-clad keeper adjusts a gleaming "
        "brass Fresnel lens housing beside a worn chart table covered in nautical compasses, folded paper coastal maps, and an enamel mug. "
        "Cool slate daylight outside contrasts with the warm amber oil lamp inside the lantern gallery, captured on medium format film with crisp micro-contrast."
    )
    h3i_scenes = [("clockmaker", t2i_clockmaker), ("lighthouse", t2i_lighthouse)]

    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.H3_IMAGE_ENCODER, device="default")
    h3i_vae_path = folder_paths.get_full_path_or_raise("vae", h3_config.IMAGE_GRAPH_VAE)
    h3i_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(h3i_vae_path))

    h3i_conds = {}
    for sk, stxt in h3i_scenes:
        c_pos, c_lat = MiniMaxH3ImageConditioning.execute(clip=clip, vae=h3i_vae, prompt=stxt, width=h3i_w, height=h3i_h).args
        h3i_conds[sk] = (c_pos, c_lat)
    del clip
    _free_vram_everywhere()

    h3i_unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.H3_IMAGE_UNET)
    h3i_model = comfy.sd.load_diffusion_model(h3i_unet_path)
    (h3i_model,) = MiniMaxH3SigmaShift.execute(model=h3i_model, shift_video=1.0, shift_audio=1.0)
    (h3i_model,) = ModelAttentionBackend.execute(model=h3i_model, attention="comfy kitchen attention")

    h3i_records: list[dict[str, Any]] = []
    h3i_sheet_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []

    for concept_stem, second_stem, s1, s2 in (
        ("paper_origami", "de_llm_slop", 1.20, 0.90),
        ("claymation_stopmotion", "noir", 1.20, 1.15),
    ):
        h3_file = sorted(bank_dir.glob(f"{concept_stem}__h3_image__*.safetensors"))[0]
        h3_file2 = sorted(bank_dir.glob(f"{second_stem}__h3_image__*.safetensors"))[0]
        for sk, _ in h3i_scenes:
            c_pos, c_lat = h3i_conds[sk]
            cols = []
            for arm_lbl, arm_vecs in (
                ("base", []),
                ("native_r8", [(str(h3_file), 1.25)]),
                (f"stack2_{second_stem}", [(str(h3_file), s1), (str(h3_file2), s2)]),
            ):
                m_run = h3i_model
                attached_vectors: list[str] = []
                erasure_ranks: list[int] = []
                for vec_path, st_v in arm_vecs:
                    (m_run,) = cv_nodes.MiniMaxH3Steer.execute(
                        model=m_run,
                        vector=vec_path,
                        strength=st_v,
                        mode="add_norm_matched_erasure",
                        target="scene_only",
                        start_percent=0.0,
                        end_percent=1.0,
                        streams="both",
                        allow_recipe_mismatch=True,
                    )
                if arm_vecs:
                    specs = m_run.model_options["transformer_options"][cv_nodes.WRAPPER_KEY]
                    assert all(sp.erasure_vr is not None for sp in specs), (
                        f"expected native h3_image erasure_vr on all stacked specs for {concept_stem}/{arm_lbl}, "
                        f"got {[sp.vector_name for sp in specs if sp.erasure_vr is None]}"
                    )
                    attached_vectors = [sp.vector_name for sp in specs]
                    erasure_ranks = [int(sp.erasure_vr.shape[0]) for sp in specs]
                lat_t = c_lat["samples"]
                noise = comfy.sample.prepare_noise(lat_t, SEED_MAIN)
                guider = comfy.samplers.CFGGuider(m_run)
                guider.inner_set_conds({"positive": c_pos})
                guider.set_cfg(1.0)
                t0 = time.perf_counter()
                samples = guider.sample(
                    noise, lat_t, comfy.samplers.sampler_object(h3_config.H3_IMAGE_SAMPLER), h3i_sigmas,
                    denoise_mask=None, callback=None, disable_pbar=True, seed=SEED_MAIN,
                )
                gen_s = round(time.perf_counter() - t0, 2)
                dec = h3i_vae.decode(samples)
                if dec.ndim == 5:
                    dec = dec[:, 0]
                im = Image.fromarray(np.clip(dec[0].detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8))
                png_p = img_dir / f"h3i_{concept_stem}_{sk}_{arm_lbl}.png"
                im.save(png_p)
                qm = _image_quality_metrics(im)
                h3i_records.append({
                    "concept": concept_stem,
                    "second_concept": second_stem if len(arm_vecs) > 1 else None,
                    "scene": sk,
                    "arm": arm_lbl,
                    "attached_vectors": attached_vectors,
                    "erasure_ranks": erasure_ranks,
                    "size": [h3i_w, h3i_h],
                    "gen_elapsed_s": gen_s,
                    "png": png_p.name,
                    **qm,
                })
                cols.append((f"{arm_lbl[:14]} L={qm['laplacian_var']:.0f} S={qm['hsv_sat']:.2f}", im))
                print(
                    f"  [Showcase D h3_image 4MP] {concept_stem} {sk} {arm_lbl} "
                    f"(ranks={erasure_ranks}): sat={qm['hsv_sat']:.4f}, lap={qm['laplacian_var']:.1f}, {gen_s}s"
                )
            h3i_sheet_rows.append((f"{concept_stem[:10]}/{sk[:8]}", cols))

    h3i_sheet_p = img_dir / "showcase_d_h3_image_native_erasure.png"
    eval_mv.make_contact_sheet(h3i_sheet_rows, h3i_sheet_p)

    del h3i_model, h3i_vae
    _free_vram_everywhere()

    out = {
        "ew50_stills": ew_records,
        "ew50_sheet": ew_sheet_p.name,
        "h3_image_4mp_native_erasure_and_stack2": h3i_records,
        "h3_image_sheet": h3i_sheet_p.name,
        "h3_image_size": [h3i_w, h3i_h],
    }
    (DATA_OUT / "showcase_d_ew50_and_h3_image.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 Steering Probes (Showcases A-D)")
    ap.add_argument("--stage", required=True, choices=("showcase_a", "showcase_b", "showcase_c", "showcase_d", "all"))
    args = ap.parse_args()

    try:
        if args.stage in ("showcase_a", "all"):
            run_showcase_a()
        if args.stage in ("showcase_b", "all"):
            run_showcase_b()
        if args.stage in ("showcase_c", "all"):
            run_showcase_c()
        if args.stage in ("showcase_d", "all"):
            run_showcase_d()
    finally:
        _free_vram_everywhere()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
