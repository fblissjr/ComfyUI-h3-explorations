#!/usr/bin/env python3
"""Build the H3 `Model2Vec`-style contextual multi-token & phrase static lexicon.

Inspired by Minish Lab (`Model2Vec` / `tokenlearn` / `POTION`, Tulkens & van Dongen):
encodes curated single-word, multi-token BPE (`origami`, `muppet`, `plasticine`,
`halation`, `pyrotechnics`, `kodachrome`, `anamorphic`, `cyanotype`, `monochrome`,
`reverb`, `16mm`, `35mm`, `chiaroscuro`, `asmr`), and 2-word cinematic/material/acoustic
phrases (`silver gelatin`, `direct flash`, `paper origami`, `cathedral reverb`, etc.)
across contextual cloze templates through `Qwen3-VL-32B` (`layers.49`, `5120`-d) and all
four H3 `TokenRefiner` recipes (`fl2va`, `ref2va`, `fl2va_base`, `h3_image`, `5376`-d).

Applies contrastive baseline subtraction + Minish Lab top-2 common-mode PC removal
so `j_lens.synthesize_from_words` and `MiniMaxH3WordVector` resolve multi-token words
and 2-word phrases in `< 1 ms` on CPU with zero GPU load at render time.

Usage:
    COMFY="$(cd ../.. && pwd)" && set -a && . ./h3.env && set +a
    "$COMFY/.venv/bin/python" bench/control_vector/build_model2vec_lexicon.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO_ROOT / "bench").resolve()]
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT))

from _lib import bootstrap  # noqa: E402

bootstrap(cpu=False)

import folder_paths  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import control_vector_nodes as cv_nodes  # noqa: E402
from bench.control_vector import helpers as _h  # noqa: E402
import j_lens  # noqa: E402
from workflows import h3_config  # noqa: E402


CURATED_LEXICON: tuple[tuple[str, str], ...] = (
    # (term, category: "visual" | "lighting" | "material" | "audio" | "baseline")
    # Optical & Film Stocks
    ("noir", "lighting"),
    ("film noir", "lighting"),
    ("monochrome", "lighting"),
    ("black-and-white", "lighting"),
    ("silver gelatin", "visual"),
    ("chiaroscuro", "lighting"),
    ("16mm", "visual"),
    ("35mm", "visual"),
    ("8mm", "visual"),
    ("super 8", "visual"),
    ("halation", "visual"),
    ("kodachrome", "visual"),
    ("technicolor", "visual"),
    ("cyanotype", "visual"),
    ("sepia", "visual"),
    ("bleach-bypass", "visual"),
    ("anamorphic", "visual"),
    ("fisheye", "visual"),
    ("tilt-shift", "visual"),
    ("telephoto", "visual"),
    ("vhs", "visual"),
    ("crt", "visual"),
    ("cctv", "visual"),
    ("camcorder", "visual"),
    ("infrared", "lighting"),
    ("thermal", "lighting"),
    ("risograph", "visual"),
    ("xerox", "visual"),
    ("grain", "visual"),
    ("film grain", "visual"),
    ("archival", "visual"),
    ("vintage", "visual"),
    # Lighting & Color Grades
    ("direct flash", "lighting"),
    ("xenon flash", "lighting"),
    ("darkroom", "lighting"),
    ("crimson", "lighting"),
    ("magenta", "lighting"),
    ("neon", "lighting"),
    ("cyberpunk", "lighting"),
    ("bioluminescent", "lighting"),
    ("golden hour", "lighting"),
    ("sunlit", "lighting"),
    ("sunlight", "lighting"),
    ("daylight", "lighting"),
    ("overcast", "lighting"),
    ("tungsten", "lighting"),
    ("candlelight", "lighting"),
    ("amber", "lighting"),
    ("warm", "lighting"),
    ("cold", "lighting"),
    ("silhouette", "lighting"),
    ("backlit", "lighting"),
    ("venetian blinds", "lighting"),
    ("strobe", "lighting"),
    ("volumetric fog", "lighting"),
    # Mediums & Materials
    ("origami", "material"),
    ("paper origami", "material"),
    ("folded paper", "material"),
    ("papercraft", "material"),
    ("muppet", "material"),
    ("felt", "material"),
    ("felt puppet", "material"),
    ("plush", "material"),
    ("claymation", "material"),
    ("plasticine", "material"),
    ("stop-motion", "material"),
    ("watercolor", "visual"),
    ("watercolor wash", "visual"),
    ("gouache", "visual"),
    ("impasto", "visual"),
    ("oil painting", "visual"),
    ("ink wash", "visual"),
    ("charcoal", "visual"),
    ("pastel", "visual"),
    ("woodcut", "visual"),
    ("linocut", "visual"),
    ("rotoscope", "visual"),
    ("anime", "visual"),
    ("cel-shaded", "visual"),
    ("8-bit", "visual"),
    ("16-bit", "visual"),
    ("pixel art", "visual"),
    ("porcelain", "material"),
    ("marble", "material"),
    ("bronze", "material"),
    ("verdigris", "material"),
    ("brass", "material"),
    ("steampunk", "material"),
    ("clockwork", "material"),
    ("automaton", "material"),
    ("molten lava", "material"),
    ("basalt", "material"),
    ("obsidian", "material"),
    ("crystalline", "material"),
    ("knitted wool", "material"),
    ("crochet", "material"),
    ("toy brick", "material"),
    ("stained glass", "material"),
    ("chrome", "material"),
    ("latex", "material"),
    ("inflatable", "material"),
    ("balloon", "material"),
    ("skeleton", "material"),
    ("skeletal", "material"),
    ("translucent", "material"),
    # Weather, Physics & Action States
    ("pyrotechnics", "visual"),
    ("explosions", "visual"),
    ("fireworks", "visual"),
    ("sparks", "visual"),
    ("blizzard", "visual"),
    ("snowstorm", "visual"),
    ("frost", "visual"),
    ("downpour", "visual"),
    ("drenched", "visual"),
    ("heatwave", "visual"),
    ("shimmer", "visual"),
    ("gale", "visual"),
    ("windstorm", "visual"),
    ("underwater", "visual"),
    ("submerged", "visual"),
    ("zero-gravity", "visual"),
    ("weightless", "visual"),
    ("chaos", "visual"),
    ("panic", "visual"),
    ("calm", "visual"),
    ("serene", "visual"),
    ("symmetry", "visual"),
    ("symmetrical", "visual"),
    ("miniature", "visual"),
    # Audio, Acoustics & Vocal Delivery
    ("reverb", "audio"),
    ("reverberation", "audio"),
    ("cathedral", "audio"),
    ("cathedral reverb", "audio"),
    ("echo", "audio"),
    ("cavernous", "audio"),
    ("anechoic", "audio"),
    ("muffled", "audio"),
    ("hydrophone", "audio"),
    ("whisper", "audio"),
    ("whispered", "audio"),
    ("asmr", "audio"),
    ("breathless", "audio"),
    ("operatic", "audio"),
    ("bellow", "audio"),
    ("auctioneer", "audio"),
    ("patter", "audio"),
    ("klaxon", "audio"),
    ("siren", "audio"),
    ("vinyl crackle", "audio"),
    ("tape hiss", "audio"),
    ("chiptune", "audio"),
    ("synth", "audio"),
    # Pole-A Baselines & Common Hold Words
    ("live-action", "baseline"),
    ("realistic", "baseline"),
    ("photorealistic", "baseline"),
    ("photograph", "baseline"),
    ("digital", "baseline"),
    ("modern", "baseline"),
    ("clean", "baseline"),
    ("sharp", "baseline"),
    ("contemporary", "baseline"),
    ("ordinary", "baseline"),
    ("standard", "baseline"),
    ("natural", "baseline"),
    ("normal", "baseline"),
    ("color", "baseline"),
    ("colorful", "baseline"),
    ("saturated", "baseline"),
    ("human", "baseline"),
    ("person", "baseline"),
    ("flesh", "baseline"),
    ("skin", "baseline"),
    ("matte", "baseline"),
    ("indoor", "baseline"),
    ("outdoor", "baseline"),
    ("coat", "baseline"),
    ("parka", "baseline"),
    ("jacket", "baseline"),
    ("rain", "baseline"),
    ("snow", "baseline"),
)

TEMPLATES_BY_CAT: dict[str, tuple[tuple[str, str], ...]] = {
    "visual": (
        (
            "integrated_multimodal_description:\n[Shot 1] Cinematic medium shot rendered in {term} aesthetic inside a quiet brick studio with wooden benches.\n\noverall_soundscape:\nSteady studio room tone.\n\nnon_diegetic_music:\nN/A",
            "integrated_multimodal_description:\n[Shot 1] Cinematic medium shot rendered in unmodified default baseline aesthetic inside a quiet brick studio with wooden benches.\n\noverall_soundscape:\nSteady studio room tone.\n\nnon_diegetic_music:\nN/A",
        ),
    ),
    "lighting": (
        (
            "integrated_multimodal_description:\n[Shot 1] Medium portrait illuminated by {term} lighting and palette inside a quiet apartment room.\n\noverall_soundscape:\nSoft room tone.\n\nnon_diegetic_music:\nN/A",
            "integrated_multimodal_description:\n[Shot 1] Medium portrait illuminated by unmodified default baseline lighting and palette inside a quiet apartment room.\n\noverall_soundscape:\nSoft room tone.\n\nnon_diegetic_music:\nN/A",
        ),
    ),
    "material": (
        (
            "integrated_multimodal_description:\n[Shot 1] Close medium shot where the figures, clothing, and props are crafted from {term} surfaces and forms.\n\noverall_soundscape:\nSubtle surface rustle.\n\nnon_diegetic_music:\nN/A",
            "integrated_multimodal_description:\n[Shot 1] Close medium shot where the figures, clothing, and props are crafted from unmodified default baseline surfaces and forms.\n\noverall_soundscape:\nSubtle surface rustle.\n\nnon_diegetic_music:\nN/A",
        ),
    ),
    "audio": (
        (
            "integrated_multimodal_description:\n[Shot 1] Medium shot of a speaker talking in a hall.\n\noverall_soundscape:\nAcoustic space and voice shaped by {term} resonance and timbre.\n\nnon_diegetic_music:\nN/A",
            "integrated_multimodal_description:\n[Shot 1] Medium shot of a speaker talking in a hall.\n\noverall_soundscape:\nAcoustic space and voice shaped by unmodified default baseline resonance and timbre.\n\nnon_diegetic_music:\nN/A",
        ),
    ),
}
TEMPLATES_BY_CAT["baseline"] = TEMPLATES_BY_CAT["visual"]


def _diff_span_slice(ids_a: list[int], ids_b: list[int]) -> tuple[slice, slice]:
    if ids_a == ids_b:
        raise ValueError("identical token sequences in _diff_span_slice")
    n_min = min(len(ids_a), len(ids_b))
    first = 0
    while first < n_min and ids_a[first] == ids_b[first]:
        first += 1
    end_a, end_b = len(ids_a), len(ids_b)
    while end_a > first and end_b > first and ids_a[end_a - 1] == ids_b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    first_safe = max(1, min(first, len(ids_a) - 1, len(ids_b) - 1))
    if end_a <= first_safe:
        end_a = min(len(ids_a), first_safe + 1)
    if end_b <= first_safe:
        end_b = min(len(ids_b), first_safe + 1)
    if end_a <= first_safe or end_b <= first_safe:
        raise ValueError(f"empty diff slice: a={first_safe}:{end_a} (len {len(ids_a)}), b={first_safe}:{end_b} (len {len(ids_b)})")
    return slice(first_safe, end_a), slice(first_safe, end_b)


def _decone_matrix(mat: torch.Tensor, n_pcs: int = 2) -> torch.Tensor:
    """Minish Lab All-but-the-Top common-mode PC removal + unit row normalization."""
    if not torch.isfinite(mat).all():
        bad_rows = torch.nonzero(~torch.isfinite(mat).all(dim=-1)).flatten().tolist()
        raise ValueError(f"_decone_matrix received non-finite rows: {bad_rows}")
    x = mat.to(dtype=torch.float64)
    xc = x - x.mean(dim=0, keepdim=True)
    if n_pcs > 0 and xc.shape[0] > n_pcs + 2:
        _u, _s, vh = torch.linalg.svd(xc, full_matrices=False)
        pcs = vh[:n_pcs]
        xc = xc - (xc @ pcs.T) @ pcs
    nrm = torch.linalg.vector_norm(xc, dim=-1, keepdim=True).clamp(min=1e-12)
    return (xc / nrm).to(dtype=torch.float32).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="Optional output path (defaults to $H3_MODEL_STORE/j_lens/model2vec_h3_lexicon.safetensors)")
    ap.add_argument("--device", default="default")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        store_root = os.environ.get("H3_MODEL_STORE", "").strip()
        base_models = Path(store_root) if store_root else Path(folder_paths.models_dir)
        out_path = (base_models / "j_lens" / "model2vec_h3_lexicon.safetensors").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dedup_terms: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term_raw, cat in CURATED_LEXICON:
        t_clean = term_raw.strip().lower()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            dedup_terms.append((t_clean, cat))

    print(f"[1/3] Encoding {len(dedup_terms)} contextual Model2Vec terms via Qwen3-VL-32B...")
    enc_name = h3_config.MODELS["clip"]
    clip = _h._load_clip(enc_name, device=args.device)

    base_cache: dict[str, tuple[torch.Tensor, list[int]]] = {}
    for cat_templates in TEMPLATES_BY_CAT.values():
        for _pos_t, neg_t in cat_templates:
            if neg_t not in base_cache:
                enc_n, _tags_n, ids_n, _dec_n = cv_nodes.encode_prompt_for_vector(clip, neg_t)
                base_cache[neg_t] = (enc_n.detach().to(device="cpu", dtype=torch.float32), list(ids_n))

    term_encoded: list[list[tuple[torch.Tensor, torch.Tensor, slice, slice]]] = []
    for idx, (term, cat) in enumerate(dedup_terms):
        templates = TEMPLATES_BY_CAT[cat]
        pair_items: list[tuple[torch.Tensor, torch.Tensor, slice, slice]] = []
        for pos_tmpl, neg_tmpl in templates:
            pos_txt = pos_tmpl.format(term=term)
            enc_p, _tags_p, ids_p, _dec_p = cv_nodes.encode_prompt_for_vector(clip, pos_txt)
            enc_n, ids_n = base_cache[neg_tmpl]
            sl_n, sl_p = _diff_span_slice(ids_n, list(ids_p))
            pair_items.append((enc_n, enc_p.detach().to(device="cpu", dtype=torch.float32), sl_n, sl_p))
        term_encoded.append(pair_items)
        if (idx + 1) % 40 == 0 or idx + 1 == len(dedup_terms):
            print(f"  encoded {idx + 1}/{len(dedup_terms)} terms ({time.perf_counter() - t0:.1f}s)")

    del clip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    K = len(dedup_terms)
    enc_mat = torch.zeros((K, 5120), dtype=torch.float32)
    for idx, pair_items in enumerate(term_encoded):
        diffs = []
        for enc_n, enc_p, sl_n, sl_p in pair_items:
            dn = enc_n[sl_n].to(dtype=torch.float64).mean(dim=0)
            dp = enc_p[sl_p].to(dtype=torch.float64).mean(dim=0)
            diffs.append(F.normalize((dp - dn).to(dtype=torch.float32), dim=0))
        enc_mat[idx] = F.normalize(torch.stack(diffs, dim=0).mean(dim=0), dim=0)
    enc_mat = _decone_matrix(enc_mat, n_pcs=2)

    refiner_specs: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("fl2va", h3_config.MODELS["unet_fl2va"], _h._parse_lora_specs([f"{h3_config.PIKA_FL2VA_SVD995_LORA}:1.0"])),
        ("ref2va", h3_config.MODELS["unet_ref2va"], _h._parse_lora_specs([f"{h3_config.PIKA_REF2VA_SVD995_LORA}:1.0"])),
        ("fl2va_base", h3_config.MODELS["unet_fl2va"], []),
        ("h3_image", h3_config.H3_IMAGE_UNET, []),
    ]

    tensors_out: dict[str, torch.Tensor] = {"encoder": enc_mat}
    ref_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/3] Projecting through 4 TokenRefiner recipes on {ref_dev}...")
    for fam_key, dit_file, loras in refiner_specs:
        dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_file))
        runner = cv_nodes.build_refiner_runner(dit_path, device=ref_dev, loras=loras)
        pr_mat = torch.zeros((K, 5376), dtype=torch.float32)
        with torch.no_grad():
            for idx, pair_items in enumerate(term_encoded):
                diffs_pr = []
                for enc_n, enc_p, sl_n, sl_p in pair_items:
                    rn = runner(enc_n.to(device=ref_dev)).detach().to(device="cpu", dtype=torch.float64)[sl_n].mean(dim=0)
                    rp = runner(enc_p.to(device=ref_dev)).detach().to(device="cpu", dtype=torch.float64)[sl_p].mean(dim=0)
                    diffs_pr.append(F.normalize((rp - rn).to(dtype=torch.float32), dim=0))
                pr_mat[idx] = F.normalize(torch.stack(diffs_pr, dim=0).mean(dim=0), dim=0)
        tensors_out[f"post_refiner.{fam_key}"] = _decone_matrix(pr_mat, n_pcs=2)
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  finished post_refiner.{fam_key} ({time.perf_counter() - t0:.1f}s)")

    terms_only = [t for t, _ in dedup_terms]
    cats_only = [c for _, c in dedup_terms]
    meta = {
        "format": "h3_model2vec_lexicon_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_terms": str(K),
        "decone_pcs": "2",
        "terms_json": json.dumps(terms_only),
        "categories_json": json.dumps(cats_only),
    }
    print(f"[3/3] Saving {out_path.name} ({K} terms)...")
    save_file(tensors_out, str(out_path), metadata=meta)
    j_lens._chmod_group_readable(out_path)
    j_lens._LEXICON_CACHE.clear()
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
