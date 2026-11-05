#!/usr/bin/env python3
"""
Batch-label MiniMax H3 control vectors using the quantized ExLlamaV3
`Qwen3-VL-32B-Instruct` model (`5.37 bpw` or `4.65 bpw`).

Supports three complementary modes (`--mode`):
  1. `self_report`:
     - Loads `vector.encoder` ([5120]) + `points["encoder"]["median_row_norm"]` from
       a control-vector `.safetensors` file if present, OR computes `vector.encoder`
       ([5120], unit-normalized, `scene_text` pooling excluding row 0, `<d>...</d>`
       dialogue, and `(S1)` speaker tags) directly from the `.toml` contrastive pairs
       using `layers.0..49` of the loaded ExLlamaV3 model.
     - Injects `alpha * median_row_norm * v_encoder` at the output of
       `model.language_model.layers.49` (`model.modules[53]`) across a sweep of
       strengths (e.g. `alpha in [-0.20, 0.00, +0.15, +0.25]`) while properly
       materializing `params["resid_pending"]` before `layers.50..63` + `lm_head`.
     - Collects the steered readout completions and synthesizes a structured JSON
       self-report label.
  2. `blind_pairs`:
     - Strips `[vector]` header metadata (`name`, `quality`, `a_side`, `b_side`) and
       presents only the contrastive `(Prompt A vs. Prompt B)` diffs to the model
       to infer a structured JSON label (`inferred_slug`, `inferred_quality`,
       `a_pole`, `b_pole`, `modality`, `visual_signature`, `acoustic_signature`,
       `confounds`).
  3. `vlm`:
     - Loads the `6 bpw` vision tower (`Model.from_config(config, component="vision")`)
       and evaluates one or more rendered gate-ladder images or contact sheets.
  4. `all` (default for `.toml` / `.safetensors` batch runs):
     - Runs both `self_report` (pure activation-steering at `layers.49`) and
       `blind_pairs` (contrastive span synthesis) and writes a combined JSON + Markdown
       report.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib
import types
from typing import Any

import torch
from safetensors import safe_open

_CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
if not (_CANDIDATE_ROOT / "exllamav3").is_dir():
    _CANDIDATE_ROOT = Path(os.environ.get("EXLLAMAV3_ROOT", Path.home() / "workspace" / "exllamav3"))
REPO_ROOT = _CANDIDATE_ROOT
sys.path.insert(0, str(REPO_ROOT))

from exllamav3 import Cache, CacheLayer_quant, Config, GreedySampler, Model, Tokenizer

DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.DOTALL)
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")


def format_chatml(system_prompt: str, user_prompt: str, assistant_prefix: str = "") -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    )


def get_chatml_stop_ids(tokenizer: Tokenizer) -> set[int]:
    stops: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.single_id(tok_str)
        if tid is not None:
            stops.add(int(tid))
    return stops


def scene_text_mask_from_ids(input_ids: torch.Tensor, tokenizer: Tokenizer) -> torch.BoolTensor:
    """
    Build a boolean mask [L] selecting `scene_text` tokens (excluding row 0,
    `<d>...</d>` dialogue spans, and `(S1)` speaker tags) on Qwen's vocabulary.
    """
    ids = input_ids[0].tolist()
    L = len(ids)
    id_to_piece = tokenizer.get_id_to_piece_list(include_special_tokens=True)
    pieces = [id_to_piece[i] if 0 <= i < len(id_to_piece) else "" for i in ids]
    full_text = "".join(pieces)

    excluded_spans: list[tuple[int, int]] = []
    for m in DIALOGUE_RE.finditer(full_text):
        excluded_spans.append((m.start(), m.end()))
    for m in SPEAKER_RE.finditer(full_text):
        excluded_spans.append((m.start(), m.end()))

    mask = torch.ones(L, dtype=torch.bool)
    if L > 1:
        mask[0] = False  # row0_excluded=True (BOS/attention-sink anomaly)

    pos = 0
    for idx, piece in enumerate(pieces):
        c0, c1 = pos, pos + len(piece)
        pos = c1
        for m0, m1 in excluded_spans:
            if max(c0, m0) < min(c1, m1):
                mask[idx] = False
                break

    if not bool(mask.any()):
        mask[:] = True
        if L > 1:
            mask[0] = False
    return mask


@torch.inference_mode()
def forward_to_layer49(model: Model, layer_49_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run `embed_tokens` + `layers.0..49` (up to `model.modules[layer_49_idx]`) using
    non-cached FlashAttention (`flash_attn_nc`) and return the materialized fp32
    residual stream `[1, L, 5120]`.
    """
    params: dict[str, Any] = {
        "attn_mode": "flash_attn_nc",
        "input_ids": input_ids,
    }
    x: torch.Tensor = input_ids
    for module in model.modules[: layer_49_idx + 1]:
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
    pending = params.pop("resid_pending", None)
    if pending is not None:
        x = x.float() + pending.float()
    return x.float()


@torch.inference_mode()
def compute_encoder_vector_from_toml(
    model: Model,
    tokenizer: Tokenizer,
    toml_path: Path,
) -> tuple[torch.Tensor, float, float, list[dict[str, str]]]:
    """
    Compute unit-normalized `vector.encoder` ([5120], fp32), `median_row_norm`
    (with `row0_excluded=True`), and `raw_norm` directly from a `.toml` pair file
    using `layers.0..49` of the ExLlamaV3 model.
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    layer_49_idx = model.modules.index(layer_49)

    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    pairs = data.get("pair", [])
    if not pairs:
        raise ValueError(f"No [[pair]] entries in {toml_path}")

    a_pooled: list[torch.Tensor] = []
    b_pooled: list[torch.Tensor] = []
    weights: list[float] = []
    median_norms: list[float] = []
    diff_Records: list[dict[str, str]] = []

    for p in pairs:
        a_txt = p.get("a", "").strip()
        b_txt = p.get("b", "").strip()
        if not a_txt or not b_txt:
            continue
        w = float(p.get("weight", 1.0))

        ids_a = tokenizer.encode(a_txt, add_bos=False, add_eos=False, encode_special_tokens=True)
        ids_b = tokenizer.encode(b_txt, add_bos=False, add_eos=False, encode_special_tokens=True)

        h_a = forward_to_layer49(model, layer_49_idx, ids_a)[0]  # [L_a, 5120]
        h_b = forward_to_layer49(model, layer_49_idx, ids_b)[0]  # [L_b, 5120]

        mask_a = scene_text_mask_from_ids(ids_a, tokenizer).to(h_a.device)
        mask_b = scene_text_mask_from_ids(ids_b, tokenizer).to(h_b.device)

        # Row-norm census excluding row 0
        norms_a = torch.linalg.vector_norm(h_a[1:] if h_a.shape[0] > 1 else h_a, dim=-1)
        norms_b = torch.linalg.vector_norm(h_b[1:] if h_b.shape[0] > 1 else h_b, dim=-1)
        median_norms.append(float(norms_a.median().item()))
        median_norms.append(float(norms_b.median().item()))

        a_pooled.append(h_a[mask_a].to(torch.float64).mean(dim=0))
        b_pooled.append(h_b[mask_b].to(torch.float64).mean(dim=0))
        weights.append(w)

        a_span, b_span = extract_diff_span(a_txt, b_txt)
        diff_Records.append({"a_span": a_span, "b_span": b_span})

    a_mat = torch.stack(a_pooled, dim=0)  # [N, 5120]
    b_mat = torch.stack(b_pooled, dim=0)  # [N, 5120]
    w_vec = torch.tensor(weights, dtype=torch.float64, device=a_mat.device)
    w_vec = w_vec / w_vec.sum()

    raw_vec = ((b_mat - a_mat) * w_vec.unsqueeze(-1)).sum(dim=0)
    raw_norm = float(torch.linalg.vector_norm(raw_vec).item())
    unit_vec = (raw_vec / max(raw_norm, 1e-12)).to(dtype=torch.float32).cpu()
    median_row_norm = float(sum(median_norms) / len(median_norms))
    return unit_vec, median_row_norm, raw_norm, diff_Records


def extract_diff_span(a_txt: str, b_txt: str) -> tuple[str, str]:
    """Extract the differing substring between Prompt A and Prompt B."""
    p = 0
    min_len = min(len(a_txt), len(b_txt))
    while p < min_len and a_txt[p] == b_txt[p]:
        p += 1
    s = 0
    while (
        s < (min_len - p)
        and a_txt[len(a_txt) - 1 - s] == b_txt[len(b_txt) - 1 - s]
    ):
        s += 1
    a_mid = a_txt[p : len(a_txt) - s].strip()
    b_mid = b_txt[p : len(b_txt) - s].strip()
    return a_mid, b_mid


def load_encoder_vector_from_safetensors(sf_path: Path) -> tuple[torch.Tensor | None, float | None, dict[str, Any]]:
    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        meta_raw = f.metadata() or {}
        meta = json.loads(meta_raw["control_vector"]) if "control_vector" in meta_raw else {}
        if "vector.encoder" in f.keys():
            vec = f.get_tensor("vector.encoder").to(torch.float32)
            norm = float(torch.linalg.vector_norm(vec.to(torch.float64)).item())
            if norm > 1e-12:
                vec = vec / norm
            med_norm = (
                meta.get("points", {})
                .get("encoder", {})
                .get("median_row_norm", 191.0)
            )
            return vec, float(med_norm), meta
        return None, None, meta


@torch.inference_mode()
def generate_cached(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    embeddings: list[Any] | None = None,
    layer_49_vec: torch.Tensor | None = None,
    layer_49_scale: float = 0.0,
    steer_mode: str = "all_ex_row0",
) -> str:
    """
    Run a single-sequence prefill + greedy decode pass with an optional `layers.49`
    residual-stream steering hook. Using a direct `model.prefill` + `model.forward`
    loop on a dedicated `Cache` avoids `Generator`'s `PageTable` KV cache reuse
    across steering sweeps.
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    orig_forward = layer_49.forward

    if layer_49_vec is not None and abs(layer_49_scale) > 1e-9:
        delta = (
            layer_49_vec.to(device=layer_49.device, dtype=torch.float32) * float(layer_49_scale)
        ).view(1, 1, -1)

        def steered_forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
            x = orig_forward(x, params, out_dtype)
            pending = params.pop("resid_pending", None)
            if pending is not None:
                x = x.float() + pending.float()
            d = delta.to(device=x.device, dtype=x.dtype)
            if x.shape[1] > 1 and steer_mode == "all_ex_row0":
                x = x.clone()
                x[:, 1:, :] = x[:, 1:, :] + d
            else:
                x = x + d
            return x

        layer_49.forward = types.MethodType(steered_forward, layer_49)

    try:
        context_ids = tokenizer.encode(
            prompt_text,
            add_bos=False,
            add_eos=False,
            encode_special_tokens=True,
            embeddings=embeddings,
        )
        sampler = GreedySampler()
        stop_ids = get_chatml_stop_ids(tokenizer)

        prefill_params: dict[str, Any] = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, cache.max_num_tokens),
        }
        if embeddings:
            prefill_params["embeddings"] = embeddings
            if model.caps.get("mrope"):
                freqs, offset = model.g_rope.get_mrope_freqs(
                    context_ids,
                    embeddings,
                    context_ids.shape[-1] + max_new_tokens + 16,
                )
                prefill_params["alt_rope_freqs"] = freqs
                prefill_params["alt_rope_offset"] = offset - context_ids.shape[-1]

        if context_ids.shape[-1] > 1:
            model.prefill(input_ids=context_ids[:, :-1], params=prefill_params)

        alt_rope_freqs = prefill_params.get("alt_rope_freqs")
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            step_params: dict[str, Any] = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": context_ids.shape[-1] - 1,
                "batch_shape": (1, cache.max_num_tokens),
            }
            if alt_rope_freqs is not None:
                step_params["alt_rope_freqs"] = alt_rope_freqs

            logits = model.forward(input_ids=context_ids[:, -1:], params=step_params)
            sample = sampler.forward(logits, tokenizer=tokenizer)
            tid = int(sample.item())
            if tid in stop_ids:
                break
            generated_ids.append(tid)
            context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)

        if not generated_ids:
            return ""
        out_tensor = torch.tensor([generated_ids], dtype=torch.long)
        return tokenizer.decode(out_tensor, decode_special_tokens=False)[0].strip()
    finally:
        layer_49.forward = orig_forward


def run_self_report_probe(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    vec_encoder: torch.Tensor,
    median_row_norm: float,
    alphas: list[float],
) -> dict[str, Any]:
    """
    Steer `layers.49` across `alphas` on two neutral open-ended sensory prompts and
    then ask the unsteered model to synthesize a structured JSON label from the
    steered completions.
    """
    probe_prompt = format_chatml(
        "You describe cinematic shots and soundscapes vividly and concisely in 2 sentences.",
        "Write a 2-sentence description of a short film clip: its visual texture, lighting, camera feel, and soundscape.",
        assistant_prefix="[Shot 1] ",
    )

    sweep_outputs: dict[str, str] = {}
    for alpha in alphas:
        scale = float(alpha) * float(median_row_norm)
        text = generate_cached(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            prompt_text=probe_prompt,
            max_new_tokens=110,
            layer_49_vec=vec_encoder,
            layer_49_scale=scale,
        )
        sweep_outputs[f"{alpha:+.2f}"] = "[Shot 1] " + text

    sweep_block = "\n\n".join(
        f"Steering strength alpha = {k} (scale = {float(k)*median_row_norm:+.1f}):\n{v}"
        for k, v in sweep_outputs.items()
    )
    synth_prompt = format_chatml(
        "You analyze activation-steering readouts from a video-and-audio conditioning model and output only valid JSON.",
        "Below are completions from the exact same neutral prompt when a hidden control vector is injected at layer 49 "
        "across negative, zero, and positive strengths.\n\n"
        f"{sweep_block}\n\n"
        "Compare the positive (+0.15, +0.25) vs baseline (+0.00) and negative (-0.20) completions. "
        "Return a single JSON object with keys:\n"
        '  "self_report_slug": short snake_case label,\n'
        '  "self_report_quality": 1-sentence description of the steered aesthetic/physical/acoustic shift,\n'
        '  "negative_pole": what negative steering emphasizes,\n'
        '  "positive_pole": what positive steering emphasizes,\n'
        '  "dominant_modality": "visual" | "acoustic" | "audiovisual"',
        assistant_prefix="{\n",
    )
    raw_json = "{\n" + generate_cached(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        prompt_text=synth_prompt,
        max_new_tokens=350,
    )
    parsed = try_parse_json(raw_json)
    return {
        "median_row_norm": round(median_row_norm, 3),
        "sweep_completions": sweep_outputs,
        "synthesis": parsed if parsed is not None else raw_json,
    }


def run_blind_pair_synthesis(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    diff_records: list[dict[str, str]],
) -> dict[str, Any] | str:
    lines = []
    for idx, rec in enumerate(diff_records[:10], start=1):
        lines.append(f"Pair {idx:02d}:\n  - Baseline (A): {rec['a_span']}\n  + Steered  (B): {rec['b_span']}")
    diffs_text = "\n".join(lines)

    prompt = format_chatml(
        "You are an expert researcher labelling contrastive control vectors for a joint video-and-audio diffusion model. Output only valid JSON.",
        "Below are the differing clauses extracted from 10 contrastive prompt pairs (Baseline Pole A vs. Steered Pole B). "
        "Without knowing the file name, deduce the exact latent concept being isolated:\n\n"
        f"{diffs_text}\n\n"
        "Return a single JSON object with keys:\n"
        '  "inferred_slug": concise snake_case identifier (2-4 words),\n'
        '  "inferred_quality": concise description of the vector quality,\n'
        '  "a_pole": baseline state in Pole A,\n'
        '  "b_pole": steered state in Pole B,\n'
        '  "modality": "visual" | "acoustic" | "audiovisual" | "typography",\n'
        '  "visual_signature": specific visual/lighting/camera/material traits shifted,\n'
        '  "acoustic_signature": specific foley/room-tone/vocal traits shifted (or "none"),\n'
        '  "confounds_or_notes": any unintended correlations across the pairs (or "clean")',
        assistant_prefix="{\n",
    )
    raw = "{\n" + generate_cached(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=450,
    )
    parsed = try_parse_json(raw)
    return parsed if parsed is not None else raw


def try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-label H3 control vectors with ExLlamaV3 Qwen3-VL-32B-Instruct")
    parser.add_argument(
        "-m",
        "--model_dir",
        type=Path,
        default=Path.home() / "Storage" / "Qwen3-VL-32B-Instruct-exl3-h3-5.37bpw",
        help="Path to compiled ExLlamaV3 model directory",
    )
    parser.add_argument(
        "--toml_dir",
        type=Path,
        default=Path.home() / "ComfyUI" / "custom_nodes" / "ComfyUI-h3-explorations" / "prompt_bank" / "control_vectors",
        help="Directory containing control-vector .toml files",
    )
    parser.add_argument(
        "--vectors_dir",
        type=Path,
        default=Path.home() / "ComfyUI" / "models" / "control_vectors",
        help="Directory containing compiled control-vector .safetensors files",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Optional comma-separated substring filter for vector names (e.g. 'noir,film_16mm,reverb')",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "self_report", "blind_pairs", "vlm"],
        default="all",
        help="Labelling mode to run",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-0.20,0.00,0.15,0.25",
        help="Comma-separated alpha strengths (in median_row_norm units) for layer-49 self-report steering",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help="Image / contact-sheet paths when running `--mode vlm`",
    )
    parser.add_argument(
        "-o",
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent / "h3_control_vector_labels.json",
        help="Output JSON report path",
    )
    parser.add_argument("--cache_tokens", type=int, default=4096)
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    filters = [f.strip() for f in args.filter.split(",") if f.strip()]

    t0 = time.time()
    print(f"Loading ExLlamaV3 config from {args.model_dir} ...")
    config = Config.from_directory(str(args.model_dir))
    tokenizer = Tokenizer.from_config(config)

    vision_model = None
    if args.mode == "vlm" or args.images:
        vision_model = Model.from_config(config, component="vision")
        vision_model.load(progressbar=True)

    model = Model.from_config(config)
    cache = Cache(
        model,
        max_num_tokens=args.cache_tokens,
        layer_type=CacheLayer_quant,
        k_bits=8,
        v_bits=8,
    )
    model.load(progressbar=True)
    print(f"Loaded model + Q8 KV cache in {time.time() - t0:.2f}s")

    if args.mode == "vlm" and args.images:
        from PIL import Image

        assert vision_model is not None
        img_embs = [
            vision_model.get_image_embeddings(tokenizer=tokenizer, image=Image.open(p))
            for p in args.images
        ]
        placeholders = "\n".join(ie.text_alias for ie in img_embs) + "\n"
        vlm_prompt = format_chatml(
            "You analyze control-vector gate-ladder frames and contact sheets for a video diffusion model. Output only valid JSON.",
            placeholders
            + "Compare the progression across the provided image(s)/frames and return a JSON object with keys: "
            '"inferred_slug", "visual_progression", "artifacts_or_saturation", "recommended_strength_notes".',
            assistant_prefix="{\n",
        )
        out = "{\n" + generate_cached(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            prompt_text=vlm_prompt,
            max_new_tokens=300,
            embeddings=img_embs,
        )
        print(out)
        return

    toml_files = sorted(args.toml_dir.glob("*.toml"))
    if filters:
        toml_files = [p for p in toml_files if any(f in p.stem for f in filters)]

    results: dict[str, Any] = {}
    if filters and args.out_json.is_file():
        try:
            results = json.loads(args.out_json.read_text(encoding="utf-8"))
        except Exception:
            results = {}
    for toml_path in toml_files:
        stem = toml_path.stem
        t_vec = time.time()
        toml_data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        vec_hdr = toml_data.get("vector", {})

        # Check if a matching safetensors in vectors_dir already has `vector.encoder`
        sf_matches = sorted(args.vectors_dir.glob(f"{stem}*.safetensors"))
        vec_enc = None
        med_norm = None
        for sf_path in sf_matches:
            vec_enc, med_norm, _ = load_encoder_vector_from_safetensors(sf_path)
            if vec_enc is not None:
                break

        print(f"\n=== [{stem}] (safetensors files: {len(sf_matches)}) ===")
        computed_vec, computed_med_norm, raw_norm, diff_records = compute_encoder_vector_from_toml(
            model=model,
            tokenizer=tokenizer,
            toml_path=toml_path,
        )
        if vec_enc is None or med_norm is None:
            vec_enc = computed_vec
            med_norm = computed_med_norm
            vec_source = "computed_exl3_layer49"
        else:
            vec_source = "safetensors_vector.encoder"

        entry: dict[str, Any] = results.get(stem, {})
        entry.update({
            "toml_file": toml_path.name,
            "safetensors_files": [p.name for p in sf_matches],
            "ground_truth_header": {
                "name": vec_hdr.get("name", stem),
                "quality": vec_hdr.get("quality", ""),
                "a_side": vec_hdr.get("a_side", ""),
                "b_side": vec_hdr.get("b_side", ""),
            },
            "encoder_vector_source": vec_source,
            "layer49_raw_norm": round(raw_norm, 4),
            "layer49_median_row_norm_ex_row0": round(float(med_norm), 4),
        })

        if args.mode in ("all", "blind_pairs"):
            entry["blind_pair_label"] = run_blind_pair_synthesis(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                diff_records=diff_records,
            )
            print(f"  [blind_pairs] -> {json.dumps(entry['blind_pair_label'], ensure_ascii=False)}")

        if args.mode in ("all", "self_report"):
            entry["layer49_self_report"] = run_self_report_probe(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                vec_encoder=vec_enc,
                median_row_norm=float(med_norm),
                alphas=alphas,
            )
            print(f"  [self_report] -> {json.dumps(entry['layer49_self_report']['synthesis'], ensure_ascii=False)}")

        entry["elapsed_sec"] = round(time.time() - t_vec, 2)
        results[stem] = entry

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nSaved batch labels for {len(results)} control vector(s) to {args.out_json}")


if __name__ == "__main__":
    main()
