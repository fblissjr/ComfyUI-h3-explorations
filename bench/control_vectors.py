#!/usr/bin/env python3
"""Build, inspect, meter, and grade MiniMax H3 control vectors.

What it checks: builds `.safetensors` control vectors from contrastive prompt
pair files (`prompt_bank/control_vectors/*.toml`), inspects saved vector files
and compatibility recipes, meters bank prompts along a control vector without
rendering (E2.8), and grades activation/prediction movement against `flip_null`,
`random_null`, the `" ."` period edit, the prompt-edit baseline, and a
length-matched padding control (E2.1-E2.7).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if __name__ == "__main__":
    import importlib.util as _ilu
    import os as _os

    if _ilu.find_spec("torch") is None:
        _venv_py = REPO_ROOT.parent.parent / ".venv" / "bin" / "python"
        if _venv_py.is_file() and Path(sys.executable).resolve() != _venv_py.resolve():
            _os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

# Ensure REPO_ROOT precedes BENCH_DIR on sys.path so `import control_vectors`
# resolves to the root library module (`REPO_ROOT/control_vectors.py`), while
# keeping BENCH_DIR on sys.path so `bench/_lib/__init__.py` can import `_lib.*`.
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from bench._lib import bootstrap  # noqa: E402

bootstrap()

import folder_paths  # noqa: E402
import torch  # noqa: E402

import control_vector_nodes as cv_nodes  # noqa: E402
import control_vectors as cv  # noqa: E402
import h3_pair_compiler as pair_compiler  # noqa: E402
import h3_spans  # noqa: E402
import j_lens  # noqa: E402
import substrate  # noqa: E402
from workflows import h3_config  # noqa: E402

LOOK_FAMILY_IDS: tuple[str, ...] = (
    "t2va_look_anchor",
    "t2va_look_faded_16mm",
    "t2va_look_noir",
    "t2va_look_golden_hour",
    "t2va_look_bleach_bypass",
    "t2va_look_neon",
    "t2va_look_pastel_ove",
    "t2va_look_sodium_rain",
    "t2va_look_sepia_plate",
)


def _resolve_prompt_text(arg: str) -> tuple[str, str]:
    """Return `(label, prompt_text)` from a prompt-bank id or file path."""
    p = Path(arg)
    if p.is_file():
        return p.stem, p.read_text(encoding="utf-8").strip()
    bank_txt = REPO_ROOT / "prompt_bank" / (
        arg if arg.endswith(".txt") else f"{arg}.txt"
    )
    if bank_txt.is_file():
        return bank_txt.stem, bank_txt.read_text(encoding="utf-8").strip()
    raise ValueError(f"prompt {arg!r} not found as a file or under prompt_bank/")


def _load_clip(encoder_name: str, device: str = "default") -> Any:
    enc_loader_mod = cv_nodes._repo("h3_encoder_loader")

    (clip,) = enc_loader_mod.MiniMaxH3EncoderLoader.execute(
        encoder_name=encoder_name, device=device
    )
    return clip


def _parse_lora_specs(specs: list[str] | None) -> list[dict[str, Any]]:
    if not specs:
        return []
    out: list[dict[str, Any]] = []
    for item in specs:
        if ":" in item:
            fname, s_str = item.rsplit(":", 1)
            strength = float(s_str)
        else:
            fname, strength = item, 1.0
        lpath = cv_nodes.resolve_lora_file_path(fname)
        out.append(
            {
                "file": Path(lpath).name,
                "sha256": cv.safetensors_recipe_digest(lpath),
                "strength": strength,
                "applied": "branch",
            }
        )
    return out


def _load_encoded_references(
    clip: Any,
    vae_name: str,
    ref_images: list[str],
    *,
    width: int = 1152,
    height: int = 768,
    length: int = h3_config.LENGTH,
) -> Any:
    """Build `H3EncodedReferences` from `--ref-images` so E2.4 can compare vectors
    made with and without a ref2va reference prefix."""
    from PIL import Image
    import numpy as np
    import comfy.sd
    import comfy.utils

    ref_cond_mod = cv_nodes._repo("reference_conditioning")
    ref_enc_mod = cv_nodes._repo("reference_encode")

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=sd)
    refs = None
    for img_spec in ref_images:
        img_path = Path(img_spec)
        if not img_path.is_file():
            img_path = Path(folder_paths.get_input_directory()) / img_spec
        if not img_path.is_file():
            raise ValueError(f"reference image {img_spec!r} not found")
        pil = Image.open(img_path).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        (refs,) = ref_cond_mod.MiniMaxH3AppendRefImage.execute(
            image=tensor,
            size_policy={"size_policy": "max", **h3_config.REF_IMAGE_SIZE},
            qwen_view=dict(h3_config.REF_QWEN_VIEW),
            references=refs,
            use_vae=True,
        )
    (encoded_refs,) = ref_enc_mod.MiniMaxH3EncodeReferences.execute(
        clip=clip,
        vae=vae,
        references=refs,
        width=width,
        height=height,
        length=length,
    )
    return encoded_refs


def _resolve_requested_families(
    raw: Any,
    *,
    auto_family: str | None = None,
) -> list[str]:
    """Normalize family selector ('both', 'auto', 'fl2va,ref2va', 'fl2va', 'ref2va') into a list of families."""
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        s = str(raw or "both").strip().lower()
        if s in ("both", "fl2va,ref2va", "ref2va,fl2va", "all"):
            return ["fl2va", "ref2va"]
        if s == "auto":
            return [auto_family] if auto_family in ("fl2va", "ref2va") else ["fl2va", "ref2va"]
        items = [part.strip() for part in s.split(",") if part.strip()]
    out: list[str] = []
    for item in items:
        if item in ("both", "all"):
            for f in ("fl2va", "ref2va"):
                if f not in out:
                    out.append(f)
        elif item == "auto":
            f_auto = auto_family if auto_family in ("fl2va", "ref2va") else "fl2va"
            if f_auto not in out:
                out.append(f_auto)
        elif item in ("fl2va", "ref2va"):
            if item not in out:
                out.append(item)
        else:
            raise ValueError(
                f"unknown family {item!r}; expected 'both', 'fl2va', 'ref2va', or 'fl2va,ref2va'"
            )
    return out or ["fl2va", "ref2va"]


def _decode_image_spec_to_tensor(img_spec: str) -> torch.Tensor:
    import base64
    import io
    from PIL import Image
    import numpy as np

    s = str(img_spec or "").strip()
    if not s:
        raise ValueError("empty image specification")
    if s.startswith("data:image/"):
        _, b64_part = s.split(",", 1)
        raw_bytes = base64.b64decode(b64_part)
        pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    img_path = Path(s)
    if not img_path.is_file():
        img_path = Path(folder_paths.get_input_directory()) / s
    if not img_path.is_file():
        raise ValueError(f"image {s!r} not found")
    pil = Image.open(img_path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_image_tensor_list(spec_str: str | list[str]) -> list[torch.Tensor]:
    if isinstance(spec_str, (list, tuple)):
        specs = [str(s).strip() for s in spec_str if str(s).strip()]
    else:
        s_raw = str(spec_str or "").strip()
        if s_raw.startswith("data:image/"):
            specs = [s_raw]
        else:
            specs = [s.strip() for s in s_raw.split(",") if s.strip()]
    if not specs:
        raise ValueError("at least one image path or uploaded image is required")
    return [_decode_image_spec_to_tensor(img_spec) for img_spec in specs]


def _resolve_vector_for_family(
    spec: str | Path,
    family: str = "fl2va",
    *,
    extra_dir: Path | None = None,
) -> Path:
    """Resolve a vector path, filename, or short stem (e.g. 'handheld_urgency') for `family`."""
    cand = Path(str(spec).strip())
    if cand.is_file():
        return cand
    fam = "ref2va" if "ref2va" in str(family).lower() else "fl2va"
    stem = cand.stem.split("__")[0]
    search_dirs: list[Path] = []
    if extra_dir is not None:
        search_dirs.append(Path(extra_dir))
    models_root = getattr(folder_paths, "models_dir", None)
    if models_root:
        search_dirs.append(Path(models_root) / cv_nodes.CONTROL_VECTORS_FOLDER)
    for d_str in folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER):
        p = Path(d_str)
        if p not in search_dirs:
            search_dirs.append(p)
    for cv_dir in search_dirs:
        if not cv_dir.is_dir():
            continue
        direct = cv_dir / cand
        if direct.is_file():
            return direct
        if not cand.suffix:
            direct_st = cv_dir / f"{cand.name}.safetensors"
            if direct_st.is_file():
                return direct_st
        for pat in (
            f"{stem}__{fam}__diff_span__*.safetensors",
            f"{stem}__{fam}__*.safetensors",
            f"{stem}__*.safetensors",
        ):
            hits = sorted(cv_dir.glob(pat))
            if hits:
                return hits[0]
    return cv_nodes.resolve_vector_file_path(str(spec).strip())


def _apply_post_refiner_holds(
    final_vec: torch.Tensor,
    *,
    family: str,
    hold_specs: list[str] | tuple[str, ...] | None = None,
    hold_words: str | None = None,
    hold_alpha: float = 1.0,
    prompt: str | None = None,
) -> tuple[torch.Tensor, list[str], float, float]:
    """Project out optional `--hold` `.safetensors` vectors and/or `--hold-words` via `j_lens.apply_soft_holds`,
    preserving the vector's pre-hold L2 norm so its native pair-difference magnitude and recommended strength band remain calibrated.
    """
    norm_before = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
    a_val = float(hold_alpha)
    if a_val <= 0.0:
        return final_vec, [], norm_before, norm_before
    hold_tensors: list[torch.Tensor] = []
    hold_labels: list[str] = []
    for h_spec in (hold_specs or ()):
        if not str(h_spec).strip():
            continue
        h_path = _resolve_vector_for_family(str(h_spec).strip(), family=family)
        ht, hname, _ = j_lens._load_vector_from_source(h_path, space="post_refiner")
        hold_tensors.append(ht)
        hold_labels.append(hname)
    hw_str = str(hold_words or "").strip()
    if hw_str and j_lens.bundle_available():
        hw_synth = j_lens.synthesize_from_words(hw_str, family=family, prompt=prompt)
        hold_tensors.append(
            hw_synth["vector.post_refiner"].detach().to(device="cpu", dtype=torch.float32).flatten()
        )
        hold_labels.append(f"words:{hw_str}")
    if not hold_tensors:
        return final_vec, [], norm_before, norm_before
    scrubbed = j_lens.apply_soft_holds(final_vec, hold_tensors, alpha=a_val)
    norm_after_raw = float(torch.linalg.vector_norm(scrubbed.to(dtype=torch.float64)).item())
    if norm_after_raw <= 1e-6 * max(1.0, norm_before):
        raise ValueError("holds completely cancelled the control vector (residual norm is zero)")
    return scrubbed.to(dtype=torch.float32), hold_labels, norm_before, norm_after_raw


def _inject_target_prompt_into_pairset(
    pair_set: cv.PairSet,
    target_prompt_arg: str,
    *,
    target_weight: float = 2.5,
    slot_hint: str = "auto",
) -> cv.PairSet:
    """Inject a user-supplied target scene prompt as `Pair 0` at `weight = target_weight` alongside `Pairs 1..9`."""
    raw_tp = str(target_prompt_arg or "").strip()
    if not raw_tp or not pair_set.pairs:
        return pair_set
    try:
        _, tp_text = _resolve_prompt_text(raw_tp)
    except Exception:
        tp_text = raw_tp
    tp_text = tp_text.strip()
    if not tp_text:
        return pair_set

    a_clause = (pair_set.pairs[0].a_span.split(" | ")[0].strip() or pair_set.a_side.strip())
    b_clause = (pair_set.pairs[0].b_span.split(" | ")[0].strip() or pair_set.b_side.strip())
    if a_clause == b_clause:
        a_clause = pair_set.a_side.split(" | ")[0].strip()
        b_clause = pair_set.b_side.split(" | ")[0].strip()

    marker = "\n\noverall_soundscape:"
    eff_slot = slot_hint if slot_hint != "auto" else pair_compiler.resolve_slot_name("auto", sample_text=b_clause)
    if eff_slot == "soundscape" and "overall_soundscape:" in tp_text:
        tp_a = tp_text.rstrip() + f" {a_clause.rstrip('.')}."
        tp_b = tp_text.rstrip() + f" {b_clause.rstrip('.')}."
    elif marker in tp_text:
        head, tail = tp_text.split(marker, 1)
        tp_a = head.rstrip() + f" The shot is {a_clause.rstrip('.')}." + marker + tail
        tp_b = head.rstrip() + f" The shot is {b_clause.rstrip('.')}." + marker + tail
    else:
        tp_a = tp_text.rstrip() + f" The shot is {a_clause.rstrip('.')}."
        tp_b = tp_text.rstrip() + f" The shot is {b_clause.rstrip('.')}."

    tp_a_span, tp_b_span = cv.differing_span(tp_a, tp_b, frame=pair_set.frame)
    pair0 = cv.PairItem(
        index=0,
        weight=float(target_weight),
        a=tp_a,
        b=tp_b,
        a_span=tp_a_span,
        b_span=tp_b_span,
    )
    tail_pairs = tuple(
        cv.PairItem(
            index=idx,
            weight=p.weight,
            a=p.a,
            b=p.b,
            a_span=p.a_span,
            b_span=p.b_span,
        )
        for idx, p in enumerate(pair_set.pairs[:9], start=1)
    )
    all_pairs = (pair0, *tail_pairs)
    digest_payload = json.dumps(
        {
            "name": pair_set.name,
            "mode": pair_set.mode,
            "pairs": [(p.weight, p.a, p.b) for p in all_pairs],
        },
        sort_keys=True,
    ).encode("utf-8")
    return cv.PairSet(
        path=pair_set.path,
        sha256=hashlib.sha256(digest_payload).hexdigest(),
        name=pair_set.name,
        quality=pair_set.quality,
        a_side=pair_set.a_side,
        b_side=pair_set.b_side,
        mode=pair_set.mode,
        frame=pair_set.frame,
        exclude_shots=pair_set.exclude_shots,
        pairs=all_pairs,
        holds=pair_set.holds,
    )


def _compile_words_or_spec_to_pairset(
    words_arg: str,
    *,
    name: str | None = None,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    slot: str = "auto",
    mode: str = "t2va",
    target_prompt: str = "",
    target_weight: float = 2.5,
) -> cv.PairSet:
    """Compile a `--words` CLI argument or multi-line/semicolon-separated spec into a `PairSet`."""
    import inspect

    raw_text = str(words_arg or "").strip()
    if "\n" not in raw_text and ";" in raw_text:
        raw_text = "\n".join(part.strip() for part in raw_text.split(";") if part.strip())

    has_multi_or_prefix = ("\n" in raw_text) or any(
        raw_text.lower().startswith(p) or f"\n{p}" in raw_text.lower()
        for p in ("soundscape:", "audio:", "foley:", "hold:", "holds:", "name:", "slot:", "mode:")
    )

    tp_resolved = ""
    if str(target_prompt or "").strip():
        try:
            _, tp_resolved = _resolve_prompt_text(str(target_prompt).strip())
        except Exception:
            tp_resolved = str(target_prompt).strip()

    parse_sig = inspect.signature(pair_compiler.parse_short_contrast_spec)
    compile_sig = inspect.signature(pair_compiler.compile_words_to_pairset)

    if has_multi_or_prefix:
        hdr_lines: list[str] = []
        lower_lines = [ln.strip().lower() for ln in raw_text.splitlines() if ln.strip()]
        if name and not any(ln.startswith("name:") for ln in lower_lines):
            hdr_lines.append(f"name: {name}")
        if quality and not any(ln.startswith("quality:") for ln in lower_lines):
            hdr_lines.append(f"quality: {quality}")
        if a_side and not any(ln.startswith("a_side:") for ln in lower_lines):
            hdr_lines.append(f"a_side: {a_side}")
        if b_side and not any(ln.startswith("b_side:") for ln in lower_lines):
            hdr_lines.append(f"b_side: {b_side}")
        if slot and slot != "auto" and not any(ln.startswith("slot:") for ln in lower_lines):
            hdr_lines.append(f"slot: {slot}")
        if mode and not any(ln.startswith("mode:") for ln in lower_lines):
            hdr_lines.append(f"mode: {mode}")
        if tp_resolved and not any(ln.startswith(("target_prompt:", "scene_prompt:")) for ln in lower_lines):
            hdr_lines.append(f"target_prompt: {tp_resolved}")
            hdr_lines.append(f"target_weight: {float(target_weight)}")
        full_spec = ("\n".join(hdr_lines) + "\n" + raw_text).strip() if hdr_lines else raw_text
        spec_out = pair_compiler.parse_short_contrast_spec(full_spec)
        return spec_out["pair_set"]

    compile_kwargs: dict[str, Any] = {
        "name": name,
        "quality": quality,
        "a_side": a_side,
        "b_side": b_side,
        "slot": slot,
        "mode": mode,
    }
    if tp_resolved and "target_prompt" in compile_sig.parameters:
        compile_kwargs["target_prompt"] = tp_resolved
        if "target_weight" in compile_sig.parameters:
            compile_kwargs["target_weight"] = float(target_weight)
    return pair_compiler.compile_words_to_pairset(raw_text, **compile_kwargs)


def cmd_make_media(args: argparse.Namespace) -> int:
    if args.pooling not in cv.VALID_MEDIA_POOLING:
        raise ValueError(
            f"pooling must be one of {cv.VALID_MEDIA_POOLING}, got {args.pooling!r}"
        )
    list_a = _load_image_tensor_list(args.image_a)
    list_b = _load_image_tensor_list(args.image_b)
    if len(list_a) != len(list_b):
        raise ValueError(
            f"--image-a and --image-b must have the same number of images, got {len(list_a)} vs {len(list_b)}"
        )
    # Match geometry per pair to the first image's aspect ratio at short_edge
    matched_a_list: list[torch.Tensor] = []
    matched_b_list: list[torch.Tensor] = []
    target_size = (0, 0)
    for ia, ib in zip(list_a, list_b):
        ma, mb, t_sz = cv_nodes.match_media_pair_geometry(
            ia if not matched_a_list else cv_nodes._repo("reference_conditioning")._view_or_source(ia, ia.shape[2], ia.shape[1], target_size[0], target_size[1]),
            ib,
            short_edge=int(args.short_edge),
        )
        target_size = t_sz
        matched_a_list.append(ma)
        matched_b_list.append(mb)
    batch_a = torch.cat(matched_a_list, dim=0)
    batch_b = torch.cat(matched_b_list, dim=0)

    dit_override = getattr(args, "dit", None) or getattr(args, "unet", None)
    families_raw = getattr(args, "families", None)
    if dit_override:
        dit_names = [str(dit_override)]
    elif families_raw:
        fams = _resolve_requested_families(families_raw, auto_family="ref2va")
        dit_names = [
            h3_config.MODELS["unet_ref2va"] if f == "ref2va" else h3_config.MODELS["unet_fl2va"]
            for f in fams
        ]
    else:
        dit_names = [h3_config.MODELS["unet_ref2va"]]

    encoder_name = args.encoder or h3_config.MODELS["clip"]
    encoder_path = Path(folder_paths.get_full_path_or_raise("text_encoders", encoder_name))
    loras = _parse_lora_specs(args.lora)

    clip = getattr(args, "_clip_override", None) or _load_clip(encoder_name, device=args.device)
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    hold_specs = getattr(args, "hold", None)
    hold_words = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha = float(getattr(args, "hold_alpha", 1.0))

    for dit_name in dit_names:
        dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        model_or_path = getattr(args, "_model_override", None) or dit_path
        out_path, report = cv_nodes.make_control_vector_from_media_pairs(
            image_a=batch_a,
            image_b=batch_b,
            model=model_or_path,
            clip=clip,
            output_name=str(args.output_name or "media_contrast"),
            pooling=str(args.pooling),
            short_edge=int(args.short_edge),
            scene_prompt=str(args.scene_prompt or ""),
            output_dir=out_dir,
            overwrite=bool(args.overwrite),
            made_by="bench/control_vectors.py::make-media/1",
            loras=loras,
            encoder_path_override=encoder_path,
            refiner_device=getattr(args, "refiner_device", "cpu"),
        )
        if hold_specs or hold_words:
            m_tensors, m_meta = cv.load_vector(out_path)
            m_fam = str(m_meta.get("family", "ref2va"))
            scrubbed_vec, hold_labels, n_bef, n_aft = _apply_post_refiner_holds(
                m_tensors["vector.post_refiner"],
                family=m_fam,
                hold_specs=hold_specs,
                hold_words=hold_words,
                hold_alpha=hold_alpha,
                prompt=str(args.scene_prompt or "") or None,
            )
            if hold_labels:
                m_tensors["vector.post_refiner"] = scrubbed_vec
                m_meta.setdefault("provenance", {})["applied_holds"] = {
                    "holds": hold_labels,
                    "alpha": hold_alpha,
                    "norm_before": round(n_bef, 4),
                    "norm_after_raw": round(n_aft, 4),
                }
                if j_lens.bundle_available():
                    try:
                        m_meta.setdefault("quality_report", {})["j_lens_readout"] = j_lens.decode_vector(
                            scrubbed_vec, space="post_refiner", family=m_fam
                        )
                    except Exception:
                        pass
                cv.save_vector(out_path, m_tensors, m_meta, overwrite=True)
                report["metadata"] = m_meta

        qrep = report["metadata"]["quality_report"]
        pts = report["metadata"]["points"]["post_refiner"]
        print(f"=== Media-Contrast Control Vector Saved: {out_path.name} ===")
        print(f"  path            : {out_path}")
        print(f"  qwen_view_size  : {report['qwen_view_size'][0]}x{report['qwen_view_size'][1]} (32-px M-RoPE grid)")
        print(f"  pooling         : {args.pooling}")
        print(f"  raw_norm        : {pts['raw_norm']:.4f}   median_row_norm: {pts['median_row_norm']:.4f}")
        print(f"  pairwise_cosine : {qrep['mean_pairwise_cosine']:.4f}   split_half: {qrep['split_half_cosine']:.4f}")
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    if args.pooling not in cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {cv.VALID_POOLING}, got {args.pooling!r}")

    resolved_pair_path: Path | None = None
    words_arg = str(getattr(args, "words", "") or "").strip()
    tp_arg = str(getattr(args, "target_prompt", "") or "").strip()
    tp_weight = float(getattr(args, "target_weight", 2.5))
    slot_arg = str(getattr(args, "slot", "auto") or "auto")
    if words_arg:
        compile_mode = "ref2va" if getattr(args, "ref_images", None) else "t2va"
        pair_set = _compile_words_or_spec_to_pairset(
            words_arg,
            name=(getattr(args, "output_name", "") or "").strip() or None,
            slot=slot_arg,
            mode=compile_mode,
            target_prompt=tp_arg,
            target_weight=tp_weight,
        )
    else:
        if not getattr(args, "pairs", None):
            raise ValueError("either a pair TOML file or --words is required for 'make'")
        resolved_pair_path = cv_nodes.resolve_pair_file_path(args.pairs)
        pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
        if tp_arg:
            pair_set = _inject_target_prompt_into_pairset(
                pair_set, tp_arg, target_weight=tp_weight, slot_hint=slot_arg
            )

    dit_name = getattr(args, "dit", None) or getattr(args, "unet", None) or (
        h3_config.MODELS["unet_ref2va"]
        if (args.ref_images or pair_set.mode == "ref2va")
        else h3_config.MODELS["unet_fl2va"]
    )
    dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    encoder_name = args.encoder or h3_config.MODELS["clip"]
    encoder_path = Path(folder_paths.get_full_path_or_raise("text_encoders", encoder_name))
    loras = _parse_lora_specs(args.lora)

    family = cv.detect_family(dit_path)
    recipe = cv_nodes.extract_model_recipe(
        None,
        None,
        dit_path_override=dit_path,
        encoder_path_override=encoder_path,
        loras_override=loras,
    )

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()
    ref_cache = getattr(args, "_refiner_cache", None)
    ref_key = (
        str(dit_path),
        str(args.refiner_device),
        tuple((l["file"], float(l["strength"])) for l in loras),
    )
    if isinstance(ref_cache, dict) and ref_key in ref_cache:
        refiner_run = ref_cache[ref_key]
    else:
        refiner_run = cv_nodes.build_refiner_runner(
            dit_path, device=args.refiner_device, loras=loras
        )
        if isinstance(ref_cache, dict):
            ref_cache[ref_key] = refiner_run
    t_ref = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    print(f"=== Control Vector Make: {pair_set.name} ===")
    pairs_label = (
        str(resolved_pair_path.relative_to(REPO_ROOT) if resolved_pair_path.is_relative_to(REPO_ROOT) else resolved_pair_path)
        if resolved_pair_path is not None
        else str(pair_set.path)
    )
    print(f"  pairs_file  : {pairs_label}")
    print(f"  pooling     : {args.pooling}   frame: {pair_set.frame}   mode: {pair_set.mode}")
    print(f"  family      : {family}")
    print(f"  checkpoint  : {recipe['checkpoint']['file']} ({recipe['checkpoint']['sha256'][:12]})")
    if loras:
        lora_summary = ", ".join(
            f"{l['file']}:{l['strength']:g} ({l['sha256'][:12]})" for l in loras
        )
        print(f"  loras       : {lora_summary}")
    print(f"  encoder     : {recipe['encoder']['file']} ({recipe['encoder']['sha256'][:12]})")
    print(f"  load_times  : encoder={t_clip - t0:.2f}s   refiner={t_ref - t_clip:.2f}s")

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
    enc_a_pooled_list: list[torch.Tensor] = []
    enc_b_pooled_list: list[torch.Tensor] = []
    weights_list: list[float] = []
    enc_censuses: list[dict[str, float]] = []
    ref_censuses: list[dict[str, float]] = []
    pair_spans: list[dict[str, Any]] = []

    print(f"\n--- Pair Spans ({len(pair_set.pairs)} pairs) ---")
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_refs
        )
        enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_refs
        )
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        enc_censuses.append(cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(cv.row_norm_census(enc_b, tags_b))
        ref_censuses.append(cv.row_norm_census(ref_a, tags_a))
        ref_censuses.append(cv.row_norm_census(ref_b, tags_b))

        first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
        spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = h3_spans.pooling_mask(
            spans_a,
            args.pooling,
            first_diff=min(first_diff, len(ids_a) - 1),
            end_diff=end_a,
            other_ids=ids_b,
        )
        mask_b = h3_spans.pooling_mask(
            spans_b,
            args.pooling,
            first_diff=min(first_diff, len(ids_b) - 1),
            end_diff=end_b,
            other_ids=ids_a,
        )

        a_pooled_list.append(ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32))
        b_pooled_list.append(ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32))
        if args.include_encoder:
            mask_a_enc = mask_a.clone()
            mask_b_enc = mask_b.clone()
            if mask_a_enc.numel() > 0:
                mask_a_enc[0] = False
            if mask_b_enc.numel() > 0:
                mask_b_enc[0] = False
            if not bool(mask_a_enc.any()):
                mask_a_enc = mask_a
            if not bool(mask_b_enc.any()):
                mask_b_enc = mask_b
            enc_a_pooled_list.append(enc_a.to(dtype=torch.float64)[mask_a_enc].mean(dim=0).to(dtype=torch.float32))
            enc_b_pooled_list.append(enc_b.to(dtype=torch.float64)[mask_b_enc].mean(dim=0).to(dtype=torch.float32))
        weights_list.append(float(item.weight))
        pair_spans.append(
            {
                "index": idx,
                "weight": float(item.weight),
                "first_diff": int(first_diff),
                "end_a": int(end_a),
                "end_b": int(end_b),
                "a_span": item.a_span,
                "b_span": item.b_span,
                "len_a": int(len(ids_a)),
                "len_b": int(len(ids_b)),
                "pooled_rows_a": int(mask_a.sum().item()),
                "pooled_rows_b": int(mask_b.sum().item()),
                "pooled_tokens": int(mask_b.sum().item()),
            }
        )
        pool_info = (
            f"pool=({int(mask_a.sum().item()):2d}t/{int(mask_b.sum().item()):2d}t [{first_diff}:{end_a}/{end_b}]) "
            if args.pooling == "diff_span"
            else ""
        )
        print(
            f"  [{idx:02d}] w={item.weight:.2f} first_diff_tok={first_diff:3d} "
            f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} {pool_info}| "
            f"-{item.a_span[:44]!r} -> +{item.b_span[:44]!r}"
        )

    a_mat = torch.stack(a_pooled_list, dim=0)
    b_mat = torch.stack(b_pooled_list, dim=0)
    w_vec = torch.tensor(weights_list, dtype=torch.float32)

    use_unit_mean_make = bool(getattr(args, "unit_mean", False))
    raw_vec = cv.mean_difference(
        a_mat, b_mat, w_vec, unit_mean=use_unit_mean_make
    )
    tensors_to_save: dict[str, torch.Tensor] = {}

    enc_hold_a_make: list[torch.Tensor] = []
    enc_hold_b_make: list[torch.Tensor] = []
    hold_groups: list[str] = []
    hold_weights: list[float] = []
    if not args.no_holds and pair_set.holds:
        hold_a_list: list[torch.Tensor] = []
        hold_b_list: list[torch.Tensor] = []
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = cv_nodes.encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_refs
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = cv_nodes.encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_refs
            )
            h_ref_a = refiner_run(h_enc_a)
            h_ref_b = refiner_run(h_enc_b)
            h_diff, h_end_a, h_end_b = h3_spans.differing_token_bounds(h_ids_a, h_ids_b)
            h_spans_a = h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a)
            h_spans_b = h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b)
            h_mask_a = h3_spans.pooling_mask(
                h_spans_a,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_a) - 1),
                end_diff=h_end_a,
                other_ids=h_ids_b,
            )
            h_mask_b = h3_spans.pooling_mask(
                h_spans_b,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_b) - 1),
                end_diff=h_end_b,
                other_ids=h_ids_a,
            )
            hold_a_list.append(
                h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
            )
            hold_b_list.append(
                h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
            )
            if args.include_encoder:
                hma_e = h_mask_a.clone()
                hmb_e = h_mask_b.clone()
                if hma_e.numel() > 0:
                    hma_e[0] = False
                if hmb_e.numel() > 0:
                    hmb_e[0] = False
                if not bool(hma_e.any()):
                    hma_e = h_mask_a
                if not bool(hmb_e.any()):
                    hmb_e = h_mask_b
                enc_hold_a_make.append(
                    h_enc_a.to(dtype=torch.float64)[hma_e].mean(dim=0).to(dtype=torch.float32)
                )
                enc_hold_b_make.append(
                    h_enc_b.to(dtype=torch.float64)[hmb_e].mean(dim=0).to(dtype=torch.float32)
                )
            hold_groups.append(h_item.group)
            hold_weights.append(float(h_item.weight))
        basis = cv.hold_basis(
            torch.stack(hold_a_list, dim=0),
            torch.stack(hold_b_list, dim=0),
            hold_groups,
            torch.tensor(hold_weights, dtype=torch.float32),
        )
        final_vec = cv.project_out(raw_vec, basis)
        tensors_to_save["holds.post_refiner"] = basis
    else:
        final_vec = raw_vec

    hold_specs_make = getattr(args, "hold", None)
    hold_words_make = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha_make = float(getattr(args, "hold_alpha", 1.0))
    applied_holds_make: list[str] = []
    if not args.no_holds and (hold_specs_make or hold_words_make):
        final_vec, applied_holds_make, norm_bef_make, norm_aft_make = _apply_post_refiner_holds(
            final_vec,
            family=family,
            hold_specs=hold_specs_make,
            hold_words=hold_words_make,
            hold_alpha=hold_alpha_make,
            prompt=str(getattr(args, "target_prompt", "") or "") or None,
        )

    tensors_to_save["vector.post_refiner"] = final_vec

    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    q_report = cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
    raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
    median_ref_row_norm = float(sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses))
    median_enc_row_norm_ex0 = float(sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses))

    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": median_ref_row_norm,
        }
    }
    if args.include_encoder:
        enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
        enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
        enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec, unit_mean=use_unit_mean_make)
        if enc_hold_a_make:
            enc_basis = cv.hold_basis(
                torch.stack(enc_hold_a_make, dim=0),
                torch.stack(enc_hold_b_make, dim=0),
                hold_groups,
                torch.tensor(hold_weights, dtype=torch.float32),
            )
            enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
        enc_raw_norm = float(torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item())
        if enc_raw_norm <= 1e-12:
            raise ValueError("vector.encoder has zero norm")
        enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(dtype=torch.float32)
        tensors_to_save["vector.encoder"] = enc_vec
        points_meta["encoder"] = {
            "unit": "median_row_norm",
            "raw_norm": enc_raw_norm,
            "median_row_norm": median_enc_row_norm_ex0,
            "row0_excluded": True,
        }

    if encoded_refs is not None:
        ref_hash = hashlib.sha256()
        for sp in getattr(encoded_refs, "spans", ()):
            ref_hash.update(cv._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    stem = (args.output_name or pair_set.name).strip()
    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    metadata: dict[str, Any] = {
        "format": cv.FORMAT_VERSION,
        "name": stem,
        "quality": pair_set.quality,
        "family": family,
        "pairs": {
            "path": rel_pair_path,
            "sha256": pair_set.sha256,
            "n_pairs": len(pair_set.pairs),
            "n_holds": 0 if args.no_holds else (len(pair_set.holds) + len(applied_holds_make)),
            "a_side": pair_set.a_side,
            "b_side": pair_set.b_side,
            "pair_spans": pair_spans,
        },
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if encoded_refs is not None else pair_set.mode,
            "frame": pair_set.frame,
            "prefix": prefix_meta,
            "pooling": args.pooling,
        },
        "points": points_meta,
        "quality_report": q_report,
        "provenance": {
            "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
            "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made_by": "bench/control_vectors.py",
            **(
                {
                    "applied_holds": {
                        "holds": applied_holds_make,
                        "alpha": hold_alpha_make,
                        "norm_before": round(norm_bef_make, 4),
                        "norm_after_raw": round(norm_aft_make, 4),
                    }
                }
                if applied_holds_make
                else {}
            ),
        },
    }

    ch_stats = cv.channel_census(final_vec)
    enc_r0_ratio = sum(c["row0_to_median_ratio"] for c in enc_censuses) / len(enc_censuses)
    ref_r0_ratio = sum(c["row0_to_median_ratio"] for c in ref_censuses) / len(ref_censuses)
    print("\n--- E2.1 Row-Norm & Channel Census ---")
    print(
        f"  encoder_rows : median_ex_row0={median_enc_row_norm_ex0:.4f}  "
        f"row0/median={enc_r0_ratio:.4f}"
    )
    print(
        f"  refiner_rows : median_tag1={median_ref_row_norm:.4f}  "
        f"row0/median={ref_r0_ratio:.4f}"
    )
    print(
        f"  channels     : top1_energy={ch_stats['top1_energy_share']:.4f}  "
        f"top5_energy={ch_stats['top5_energy_share']:.4f}  "
        f"top10_energy={ch_stats['top10_energy_share']:.4f}  "
        f"max_abs/rms={ch_stats['max_abs_over_rms']:.2f}"
    )

    jl_make_readout: dict[str, Any] | None = None
    if j_lens.bundle_available():
        try:
            jl_make_readout = j_lens.decode_vector(final_vec, space="post_refiner", family=family)
            q_report["j_lens_readout"] = jl_make_readout
        except Exception:
            jl_make_readout = None

    print("\n--- Agreement & Norm Statistics ---")
    print(f"  raw_norm             : {raw_norm:.6f}")
    print(f"  mean_pairwise_cosine : {q_report['mean_pairwise_cosine']:.4f}")
    print(f"  split_half_cosine    : {q_report['split_half_cosine']:.4f}")
    print(
        f"  per_pair_cosine      : min={min(q_report['per_pair_cosine']):.4f}  "
        f"max={max(q_report['per_pair_cosine']):.4f}"
    )
    if jl_make_readout is not None:
        print(f"  j_lens (+)           : {jl_make_readout['pos_summary']}")
        print(f"  j_lens (-)           : {jl_make_readout['neg_summary']}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute() and len(out_path.parts) == 1:
            out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
            out_path = out_dir / out_path
    else:
        out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        out_path = out_dir / cv.default_vector_filename(
            stem,
            family,
            pooling=args.pooling,
            prefix=prefix_meta,
            apply_holds=not bool(args.no_holds),
            n_holds=len(pair_set.holds),
            loras=loras,
        )

    t_enc_done = time.perf_counter()
    saved = cv.save_vector(
        out_path,
        tensors_to_save,
        metadata,
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    elapsed = time.perf_counter() - t0
    print(
        f"\n  timings     : encoder_load={t_clip - t0:.2f}s  refiner_load={t_ref - t_clip:.2f}s  "
        f"encode_and_refine={t_enc_done - t_ref:.2f}s  total={elapsed:.2f}s"
    )
    print(f"  saved       : {saved.name} ({saved.stat().st_size} bytes, {elapsed:.2f}s)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    vpath = cv_nodes.resolve_vector_file_path(args.vector)
    tensors, metadata = cv.load_vector(vpath)
    vec = (
        tensors["vector.post_refiner"]
        if "vector.post_refiner" in tensors
        else next(iter(tensors.values()))
    )
    ch_stats = cv.channel_census(vec)

    print(f"=== Control Vector Inspect: {metadata['name']} ===")
    print(f"  file            : {vpath.name}")
    print(f"  format          : {metadata['format']}")
    print(f"  quality         : {metadata['quality']}")
    print(f"  family          : {metadata['family']}")
    print(f"  tensors         : { {k: list(t.shape) for k, t in tensors.items()} }")
    print(f"  pairs           : {json.dumps(metadata['pairs'], sort_keys=True)}")
    print(f"  recipe          : {json.dumps(metadata['recipe'], sort_keys=True)}")
    print(f"  measured_under  : {json.dumps(metadata['measured_under'], sort_keys=True)}")
    print(f"  points          : {json.dumps(metadata['points'], sort_keys=True)}")
    print(f"  quality_report  : {json.dumps(metadata['quality_report'], sort_keys=True)}")
    print(
        f"  channel_census  : top1={ch_stats['top1_energy_share']:.4f} "
        f"top5={ch_stats['top5_energy_share']:.4f} "
        f"top10={ch_stats['top10_energy_share']:.4f} "
        f"max_abs/rms={ch_stats['max_abs_over_rms']:.2f}"
    )

    if j_lens.bundle_available() and "vector.post_refiner" in tensors:
        hold_paths = [cv_nodes.resolve_vector_file_path(h) for h in (getattr(args, "hold", None) or ())]
        hold_words_arg = str(getattr(args, "hold_words", "") or "").strip() or None
        prompt_arg = str(getattr(args, "prompt", "") or "").strip() or None
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))
        jl_rep = j_lens.decode_safetensors(
            vpath,
            holds=hold_paths if hold_paths else None,
            hold_words=hold_words_arg,
            alpha=hold_alpha,
            prompt=prompt_arg,
        )
        pr_jl = jl_rep["post_refiner"]
        if jl_rep.get("holds"):
            print(
                f"  j_lens_holds    : {jl_rep['holds']} (alpha={hold_alpha:.2f}, "
                f"norm {pr_jl['norm_before_holds']:.2f} -> {pr_jl['norm_after_holds']:.2f})"
            )
        print(f"  j_lens (PR +)   : {pr_jl['pos_summary']}")
        print(f"  j_lens (PR -)   : {pr_jl['neg_summary']}")
        if "encoder" in jl_rep:
            enc_jl = jl_rep["encoder"]
            print(f"  j_lens (EN +)   : {enc_jl['pos_summary']}")
            print(f"  j_lens (EN -)   : {enc_jl['neg_summary']}")
            print(
                f"  j_lens cosines  : cos(J_ref v_enc, v_PR)={jl_rep['refiner_forward_cosine']:.4f}  "
                f"cos_J63(Wiener)={jl_rep['wiener_j63_cosine']:.4f}"
            )
        rd_jl = pr_jl.get("readiness") or {}
        if rd_jl:
            print(f"  j_lens readiness: {rd_jl.get('summary', '')}")
        pc_jl = pr_jl.get("prompt_coupling") or {}
        if pc_jl:
            print(f"  prompt coupling : {pc_jl.get('summary', '')}")
            for rec_line in pc_jl.get("recommendations", ()):
                print(f"                    -> {rec_line}")

    check_dit_arg = getattr(args, "check_dit", None) or getattr(args, "check_unet", None)
    if check_dit_arg:
        dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", check_dit_arg))
        fam = cv.detect_family(dit_path)
        enc_name = args.check_encoder or metadata["recipe"]["encoder"]["file"]
        enc_path = (
            Path(folder_paths.get_full_path_or_raise("text_encoders", enc_name))
            if enc_name != "unspecified"
            else None
        )
        runtime_recipe = cv_nodes.extract_model_recipe(
            None,
            None,
            dit_path_override=dit_path,
            encoder_path_override=enc_path,
            loras_override=_parse_lora_specs(args.check_lora),
        )
        warnings = cv.check_vector_compatibility(
            metadata,
            loaded_family=fam,
            loaded_recipe=runtime_recipe,
            allow_recipe_mismatch=args.allow_recipe_mismatch,
        )
        print(f"  compatibility   : ok (warnings={warnings})")

    meter_targets = list(args.meter or [])
    if args.meter_look_family:
        for pid in LOOK_FAMILY_IDS:
            if pid not in meter_targets:
                meter_targets.append(pid)

    if meter_targets:
        enc_name = args.check_encoder or metadata["recipe"]["encoder"]["file"]
        if enc_name == "unspecified":
            enc_name = h3_config.MODELS["clip"]
        clip = _load_clip(enc_name, device=args.device)
        dit_name = check_dit_arg or metadata["recipe"]["checkpoint"]["file"]
        dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        meter_loras = (
            _parse_lora_specs(args.check_lora)
            if args.check_lora
            else list(metadata["recipe"].get("loras", ()))
        )
        refiner_run = cv_nodes.build_refiner_runner(
            dit_path, device=args.refiner_device, loras=meter_loras
        )
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        if pooling in ("from_first_diff", "diff_span"):
            pooling = "scene_text"

        print(f"\n--- Meter Projection along {metadata['name']} (pooling={pooling}) ---")
        rows: list[tuple[str, float, float, float, int, float, float, list[tuple[str, float]]]] = []
        v64_unit = torch.nn.functional.normalize(vec.detach().to(device="cpu", dtype=torch.float64), dim=-1)
        for target in meter_targets:
            label, text = _resolve_prompt_text(target)
            enc_rows, tags, tids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, text)
            ref_rows = refiner_run(enc_rows)
            sp = h3_spans.spans(tids, tags.tolist(), dec_fn)
            mask = h3_spans.pooling_mask(sp, pooling)
            m = cv.meter_score(ref_rows, vec, mask=mask)
            r64_cpu = ref_rows.detach().to(device="cpu", dtype=torch.float64)
            r64_unit = torch.nn.functional.normalize(r64_cpu, dim=-1)
            tok_cos_all = (r64_unit @ v64_unit).tolist()
            m_bool = mask.flatten().to(device="cpu", dtype=torch.bool)
            active_idxs = [i for i, flag in enumerate(m_bool.tolist()) if flag]
            active_cos = [tok_cos_all[i] for i in active_idxs] if active_idxs else tok_cos_all
            max_tcos = max(active_cos) if active_cos else 0.0
            p90_tcos = (
                float(torch.quantile(torch.tensor(active_cos, dtype=torch.float64), 0.90).item())
                if active_cos
                else 0.0
            )
            scored_toks: list[tuple[str, float]] = []
            for idx_t in active_idxs:
                tok_str = dec_fn([int(tids[idx_t])]).strip() if idx_t < len(tids) else ""
                if len(tok_str) >= 2 and any(ch.isalnum() for ch in tok_str):
                    scored_toks.append((tok_str, float(tok_cos_all[idx_t])))
            scored_toks.sort(key=lambda x: x[1], reverse=True)
            rows.append(
                (
                    label,
                    m["projection_units"],
                    m["signed_projection"],
                    m["cosine"],
                    int(mask.sum().item()),
                    max_tcos,
                    p90_tcos,
                    scored_toks[:5],
                )
            )
        anchor_units = rows[0][1] if rows else 0.0
        for label, p_units, s_proj, cos_val, n_tok, max_tc, p90_tc, top_toks in rows:
            delta_u = p_units - anchor_units
            top_str = ", ".join(f"{w} ({c:+.3f})" for w, c in top_toks)
            print(
                f"  {label:<36s} units={p_units:+8.4f} (delta={delta_u:+8.4f})  "
                f"proj={s_proj:+8.4f}  cos={cos_val:+7.4f}  p90_tok={p90_tc:+6.3f}  max_tok={max_tc:+6.3f}  tokens={n_tok:3d}"
            )
            if top_str:
                print(f"      top_receptors: {top_str}")
    return 0


def _prompt_token_count(clip: Any, text: str) -> int:
    clean = cv_nodes._rules.normalize_prompt(text)
    entries = cv_nodes._extract_token_entries(clip.tokenize(clean))
    return len(entries)


def _build_padding_prompt(base_text: str, target_extra_tokens: int, clip: Any) -> str:
    """Append neutral filler clauses to `base_text` so its token length matches
    the edit prompt's token length L (sizing the length-induced RoPE shift)."""
    if target_extra_tokens <= 0:
        return base_text
    base_len = _prompt_token_count(clip, base_text)
    target_len = base_len + target_extra_tokens
    marker = "\n\noverall_soundscape:"
    filler_unit = " The scene continues in steady, unbroken real time across the frame."
    pad = ""
    cur_text = base_text

    def _inject(p_str: str) -> str:
        return (
            base_text.replace(marker, p_str + marker, 1)
            if marker in base_text
            else base_text + p_str
        )

    for _ in range(64):
        if _prompt_token_count(clip, cur_text) >= target_len:
            break
        pad += filler_unit
        cur_text = _inject(pad)

    words = pad.split(" ")
    while len(words) > 1:
        cand_pad = " ".join(words[:-1])
        cand_text = _inject(cand_pad)
        if _prompt_token_count(clip, cand_text) >= target_len:
            words.pop()
            cur_text = cand_text
        else:
            # If trimming one word drops below target_len (e.g. a 2-token word),
            # try adding a single-token word so the length matches target_len.
            for one_tok in (" now", " here", " still", "."):
                exact_cand = _inject(cand_pad + one_tok)
                if _prompt_token_count(clip, exact_cand) == target_len:
                    return exact_cand
            break
    return cur_text


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.to(dtype=torch.float64) - b.to(dtype=torch.float64)).reshape(-1)
    base = b.to(dtype=torch.float64).reshape(-1)
    denom = float(torch.linalg.vector_norm(base).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.linalg.vector_norm(diff).item() / denom)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().to(dtype=torch.float64).reshape(-1)
    b64 = b.detach().to(dtype=torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a64).item())
    nb = float(torch.linalg.vector_norm(b64).item())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(torch.dot(a64, b64).item() / (na * nb))


def _sigma_shifted(model: Any) -> Any:
    """Apply `MiniMaxH3SigmaShift` parameters from `h3_config.SIGMA_SHIFT`."""
    import comfy.model_sampling

    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(
        shift=h3_config.SIGMA_SHIFT["shift_video"],
        audio_shift=h3_config.SIGMA_SHIFT["shift_audio"],
    )
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", sampling)
    to = m.model_options["transformer_options"] = m.model_options.get(
        "transformer_options", {}
    ).copy()
    to["minimax_h3_sigma_shift_video"] = h3_config.SIGMA_SHIFT["shift_video"]
    to["minimax_h3_sigma_shift_audio"] = h3_config.SIGMA_SHIFT["shift_audio"]
    return m


def _one_forward(
    model: Any,
    cond: list[Any],
    noise: Any,
    latent: Any,
    sigmas: torch.Tensor,
    step: int,
    seed: int,
) -> Any:
    """Evaluate the DiT's denoised prediction at schedule index `step` through
    `Guider_Basic` + `KSAMPLER` (`bench/measure_marker_epsilon.py::one_forward`)."""
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    captured: dict[str, Any] = {}

    def _single(model_fn, x, sigs, extra_args, callback=None, disable=None):
        sigma = sigs[0] * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
        out = model_fn(x, sigma, **extra_args)
        captured["out"] = out
        return out

    guider = Guider_Basic(model)
    guider.set_conds(cond)
    result = guider.sample(
        noise,
        latent,
        comfy.samplers.KSAMPLER(_single),
        sigmas[step : step + 2],
        denoise_mask=None,
        callback=None,
        disable_pbar=True,
        seed=seed,
    )
    if "out" not in captured:
        raise RuntimeError("KSAMPLER never evaluated the model during grade forward")
    return result


def _attach_raw_vector(
    model: Any,
    name: str,
    vec: torch.Tensor,
    strength: float,
    *,
    mask: str = "non_vision",
    mode: str = "add",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Clone `model` with an in-memory `(vec, strength, mask, mode, start_percent, end_percent)` attached via
    `control_vector_nodes._make_control_vector_wrapper` on `WrappersMP.DIFFUSION_MODEL`."""
    import comfy.patcher_extension as pe

    m = model.clone()
    spec = cv_nodes._AppliedVectorSpec(
        vector_name=name,
        vector=vec.detach().to(device="cpu", dtype=torch.float32),
        strength=float(strength),
        mask=mask,
        mode=mode,
        start_percent=float(start_percent),
        end_percent=float(end_percent),
        metadata=metadata or {"measured_under": {"prefix": "none"}},
        advisories=(),
    )
    t_opts = m.model_options.setdefault("transformer_options", {})
    t_opts[cv_nodes.WRAPPER_KEY] = (spec,)
    wrappers = m.wrappers.setdefault(pe.WrappersMP.DIFFUSION_MODEL, {})
    wrappers[cv_nodes.WRAPPER_KEY] = [cv_nodes._make_control_vector_wrapper((spec,))]
    return m


def _pred_blocks(pred: Any) -> dict[str, torch.Tensor]:
    """Unpack a DiT prediction into `{'video': Tensor, 'audio': Tensor}` flat float64 CPU vectors."""
    if isinstance(pred, dict) and "video" in pred and "audio" in pred:
        return {
            "video": pred["video"].detach().to("cpu", torch.float64).flatten(),
            "audio": pred["audio"].detach().to("cpu", torch.float64).flatten(),
        }
    parts = pred.unbind() if getattr(pred, "is_nested", False) else (pred,)
    if len(parts) >= 2:
        return {
            "video": parts[0].detach().to("cpu", torch.float64).flatten(),
            "audio": parts[1].detach().to("cpu", torch.float64).flatten(),
        }
    x = parts[0].detach().to("cpu", torch.float64).flatten()
    return {"video": x, "audio": x}


def _block_metrics(
    arm_blocks: dict[str, torch.Tensor],
    base_blocks: dict[str, torch.Tensor],
    edit_blocks: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-block (`video`, `audio`) `relative_l2`, `cosine_with_base`,
    `max_abs_diff`, and (when `edit_blocks` is given) `delta_cosine_with_edit`."""
    out: dict[str, dict[str, float]] = {}
    for bname in ("video", "audio"):
        arm_x = arm_blocks[bname]
        base_x = base_blocks[bname]
        diff = arm_x - base_x
        base_norm = float(torch.linalg.vector_norm(base_x).item())
        rel_l2 = float(torch.linalg.vector_norm(diff).item() / base_norm) if base_norm > 1e-12 else 0.0
        entry: dict[str, float] = {
            "relative_l2": rel_l2,
            "cosine_with_base": _cosine(arm_x, base_x),
            "max_abs_diff": float(diff.abs().max().item()) if diff.numel() > 0 else 0.0,
        }
        if edit_blocks is not None:
            edit_diff = edit_blocks[bname] - base_x
            entry["delta_cosine_with_edit"] = _cosine(diff, edit_diff)
        out[bname] = entry
    return out


def cmd_grade(args: argparse.Namespace) -> int:
    vpath = cv_nodes.resolve_vector_file_path(args.vector)
    tensors, metadata = cv.load_vector(vpath)
    vec = tensors["vector.post_refiner"]
    prompt_label, prompt_text = _resolve_prompt_text(args.prompt)

    enc_name = args.encoder or metadata["recipe"]["encoder"]["file"]
    if enc_name == "unspecified":
        enc_name = h3_config.MODELS["clip"]
    dit_name = getattr(args, "dit", None) or getattr(args, "unet", None) or metadata["recipe"]["checkpoint"]["file"]
    dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    if getattr(args, "no_lora", False):
        grade_loras: list[dict[str, Any]] = []
    elif getattr(args, "lora", None):
        grade_loras = _parse_lora_specs(args.lora)
    else:
        grade_loras = list(metadata["recipe"].get("loras", ()))

    t_grade_start = time.perf_counter()
    clip = _load_clip(enc_name, device=args.device)
    t_clip_loaded = time.perf_counter()
    refiner_run = cv_nodes.build_refiner_runner(
        dit_path, device=args.refiner_device, loras=grade_loras
    )
    t_ref_loaded = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    base_cond, base_enc, base_tags, base_ids, _ = cv_nodes.encode_conditioning_for_vector(
        clip, prompt_text, encoded_references=encoded_refs
    )
    base_ctx = refiner_run(base_enc)
    base_ctx_b = base_ctx.unsqueeze(0)
    base_tags_b = base_tags.unsqueeze(0)

    period_cond, period_enc, period_tags, period_ids, _ = cv_nodes.encode_conditioning_for_vector(
        clip, prompt_text + " .", encoded_references=encoded_refs
    )
    period_ctx = refiner_run(period_enc)

    raw_norm = float(torch.linalg.vector_norm(vec.to(dtype=torch.float64)).item())
    gen_rand = torch.Generator(device="cpu").manual_seed(int(args.seed))
    rand_dir = cv.random_null(raw_norm, int(vec.shape[0]), generator=gen_rand)
    flip_dir: torch.Tensor | None = None
    if args.pairs:
        pair_set = cv.load_pairs(
            cv_nodes.resolve_pair_file_path(args.pairs), bank_dir=cv_nodes.PROMPT_BANK_DIR
        )
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        p_a: list[torch.Tensor] = []
        p_b: list[torch.Tensor] = []
        p_w: list[float] = []
        for item in pair_set.pairs:
            ea, ta, ia, da = cv_nodes.encode_prompt_for_vector(
                clip, item.a, encoded_references=encoded_refs
            )
            eb, tb, ib, db = cv_nodes.encode_prompt_for_vector(
                clip, item.b, encoded_references=encoded_refs
            )
            ra = refiner_run(ea)
            rb = refiner_run(eb)
            fdiff, end_a, end_b = h3_spans.differing_token_bounds(ia, ib)
            ma = h3_spans.pooling_mask(
                h3_spans.spans(ia, ta.tolist(), da),
                pooling,
                first_diff=min(fdiff, len(ia) - 1),
                end_diff=end_a,
            )
            mb = h3_spans.pooling_mask(
                h3_spans.spans(ib, tb.tolist(), db),
                pooling,
                first_diff=min(fdiff, len(ib) - 1),
                end_diff=end_b,
            )
            p_a.append(ra.to(dtype=torch.float64)[ma].mean(dim=0).to(dtype=torch.float32))
            p_b.append(rb.to(dtype=torch.float64)[mb].mean(dim=0).to(dtype=torch.float32))
            p_w.append(float(item.weight))
        gen_flip = torch.Generator(device="cpu").manual_seed(int(args.seed))
        flip_dir = cv.flip_null(
            torch.stack(p_a, dim=0),
            torch.stack(p_b, dim=0),
            weights=torch.tensor(p_w, dtype=torch.float32),
            generator=gen_flip,
        )

    edit_cond: list[Any] | None = None
    edit_ctx: torch.Tensor | None = None
    edit_ids: list[int] | None = None
    edit_tags: torch.Tensor | None = None
    pad_cond: list[Any] | None = None
    pad_ctx: torch.Tensor | None = None
    pad_ids: list[int] | None = None
    pad_tags: torch.Tensor | None = None
    if args.edit_prompt:
        _edit_label, edit_text = _resolve_prompt_text(args.edit_prompt)
        edit_cond, edit_enc, edit_tags, edit_ids, _ = cv_nodes.encode_conditioning_for_vector(
            clip, edit_text, encoded_references=encoded_refs
        )
        edit_ctx = refiner_run(edit_enc)
        extra_tok = len(edit_ids) - len(base_ids)
        pad_text = _build_padding_prompt(prompt_text, extra_tok, clip)
        pad_cond, pad_enc, pad_tags, pad_ids, _ = cv_nodes.encode_conditioning_for_vector(
            clip, pad_text, encoded_references=encoded_refs
        )
        pad_ctx = refiner_run(pad_enc)

    # Free the 25 GB text encoder before any DiT load so they never contend for VRAM.
    t_enc_phase_done = time.perf_counter()
    del clip
    del encoded_refs
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass

    strengths = [float(s.strip()) for s in args.strengths.split(",") if s.strip()]
    n_vis = int((base_tags == 0).sum().item())
    n_txt = int((base_tags != 0).sum().item())

    print(f"=== Control Vector Grade: {metadata['name']} on {prompt_label} ===")
    print(f"  L_base={len(base_ids)} (tag1={n_txt}, tag0_vision={n_vis})  vector_norm={raw_norm:.4f}")
    print(
        f"  encode_times : encoder_load={t_clip_loaded - t_grade_start:.2f}s  "
        f"refiner_load={t_ref_loaded - t_clip_loaded:.2f}s  "
        f"encode_and_refine={t_enc_phase_done - t_ref_loaded:.2f}s"
    )
    if edit_ids is not None and pad_ids is not None:
        print(f"  L_edit={len(edit_ids)} (delta={len(edit_ids) - len(base_ids):+d})  L_pad={len(pad_ids)}")

    print("\n--- Post-Refiner Context Movement (Tag-1 Mean & Sequence Rel-L2) ---")
    base_tag1_mean = base_ctx.to(dtype=torch.float64)[base_tags != 0].mean(dim=0)
    base_tag1_norm = float(torch.linalg.vector_norm(base_tag1_mean).item())

    context_report: dict[str, Any] = {
        "L_base": len(base_ids),
        "n_tag1": n_txt,
        "n_vision": n_vis,
        "vector_raw_norm": raw_norm,
        "arms": {},
    }

    if edit_ctx is not None and edit_tags is not None and edit_ids is not None:
        edit_tag1_mean = edit_ctx.to(dtype=torch.float64)[edit_tags != 0].mean(dim=0)
        edit_delta = edit_tag1_mean - base_tag1_mean
        edit_move = float(torch.linalg.vector_norm(edit_delta).item()) / base_tag1_norm
        ctx_matched_s = float(torch.linalg.vector_norm(edit_delta).item()) / raw_norm
        context_report["L_edit"] = len(edit_ids)
        context_report["arms"]["edit_prompt"] = {
            "tag1_mean_rel_l2": edit_move,
            "cos_with_v": _cosine(edit_delta, vec),
            "context_matched_strength": ctx_matched_s,
        }
        print(
            f"  edit_prompt      : tag1_mean_rel_l2={edit_move:.6f}  "
            f"cos_with_v={_cosine(edit_delta, vec):+.4f}  "
            f"ctx_matched_strength={ctx_matched_s:.4f}"
        )
    if pad_ctx is not None and pad_tags is not None and pad_ids is not None:
        pad_tag1_mean = pad_ctx.to(dtype=torch.float64)[pad_tags != 0].mean(dim=0)
        pad_delta = pad_tag1_mean - base_tag1_mean
        pad_move = float(torch.linalg.vector_norm(pad_delta).item()) / base_tag1_norm
        context_report["L_pad"] = len(pad_ids)
        context_report["arms"]["padding_control"] = {
            "tag1_mean_rel_l2": pad_move,
            "cos_with_v": _cosine(pad_delta, vec),
        }
        print(
            f"  padding_control  : tag1_mean_rel_l2={pad_move:.6f}  "
            f"cos_with_v={_cosine(pad_delta, vec):+.4f}"
        )

    period_tag1_mean = period_ctx.to(dtype=torch.float64)[period_tags != 0].mean(dim=0)
    period_delta = period_tag1_mean - base_tag1_mean
    period_move = float(torch.linalg.vector_norm(period_delta).item()) / base_tag1_norm
    context_report["arms"]["period_null"] = {
        "tag1_mean_rel_l2": period_move,
        "cos_with_v": _cosine(period_delta, vec),
    }
    print(f"  period_null (' .'): tag1_mean_rel_l2={period_move:.6f}  cos_with_v={_cosine(period_delta, vec):+.4f}")

    for s in strengths:
        pushed_masked = cv.apply_post_refiner(
            base_ctx_b, base_tags_b, [(vec, s)], mask_vision=True
        )[0]
        rel_seq = _rel_l2(pushed_masked, base_ctx)
        vis_diff = (
            float((pushed_masked[base_tags == 0] - base_ctx[base_tags == 0]).abs().max().item())
            if n_vis > 0
            else 0.0
        )
        arm_entry: dict[str, Any] = {"seq_rel_l2": rel_seq, "vis_max_diff": vis_diff}
        msg = f"  vector s={s:+6.2f}  : seq_rel_l2={rel_seq:.6f}  vis_max_diff={vis_diff:.2e}"
        if n_vis > 0:
            pushed_unmasked = cv.apply_post_refiner(
                base_ctx_b, base_tags_b, [(vec, s)], mask_vision=False
            )[0]
            unmasked_vis_rel = _rel_l2(
                pushed_unmasked[base_tags == 0], base_ctx[base_tags == 0]
            )
            arm_entry["unmasked_vis_rel_l2"] = unmasked_vis_rel
            msg += f"  unmasked_vis_rel_l2={unmasked_vis_rel:.6f}"
        context_report["arms"][f"vector_s{s:g}"] = arm_entry
        print(msg)

    if flip_dir is not None:
        flip_pushed = cv.apply_post_refiner(
            base_ctx_b, base_tags_b, [(flip_dir, 1.0)], mask_vision=True
        )[0]
        context_report["arms"]["flip_null"] = {
            "seq_rel_l2": _rel_l2(flip_pushed, base_ctx),
            "cos_with_v": _cosine(flip_dir, vec),
        }
        print(
            f"  flip_null s=+1.0 : seq_rel_l2={_rel_l2(flip_pushed, base_ctx):.6f}  "
            f"cos_with_v={_cosine(flip_dir, vec):+.4f}"
        )
    rand_pushed = cv.apply_post_refiner(
        base_ctx_b, base_tags_b, [(rand_dir, 1.0)], mask_vision=True
    )[0]
    context_report["arms"]["random_null"] = {
        "seq_rel_l2": _rel_l2(rand_pushed, base_ctx),
        "cos_with_v": _cosine(rand_dir, vec),
    }
    print(
        f"  rand_null s=+1.0 : seq_rel_l2={_rel_l2(rand_pushed, base_ctx):.6f}  "
        f"cos_with_v={_cosine(rand_dir, vec):+.4f}"
    )

    dit_results: list[dict[str, Any]] = []
    dit_provenance: dict[str, Any] = {
        "dit_file": dit_path.name,
        "unet_file": dit_path.name,
        "loras": grade_loras,
        "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
        "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
        "attention": "sol" if getattr(args, "sol", False) else "stock_dense",
        "timings_s": {
            "encoder_load_s": round(t_clip_loaded - t_grade_start, 3),
            "refiner_load_s": round(t_ref_loaded - t_clip_loaded, 3),
            "encode_and_refine_s": round(t_enc_phase_done - t_ref_loaded, 3),
        },
    }
    if not getattr(args, "no_dit", False):
        use_pika = bool(getattr(args, "pika", False))
        manual_sigmas_str = getattr(args, "manual_sigmas", None) or (
            h3_config.PIKA_MANUAL_SIGMAS if use_pika else None
        )
        probe_steps_str = args.probe_steps
        if use_pika and probe_steps_str == "0,8,15":
            probe_steps_str = "0,2,5"
        probe_steps = [int(x.strip()) for x in probe_steps_str.split(",") if x.strip()]
        dit_runner = getattr(args, "_dit_runner", None)
        if dit_runner is None:
            import comfy.model_management as mm
            import comfy.sample
            import comfy.samplers
            import comfy.sd
            import comfy.utils
            from comfy_extras.nodes_minimax_h3 import _empty_av_latent

            t_dit_load_start = time.perf_counter()
            lora_note = (
                f", loras={[l['file'] for l in grade_loras]}" if grade_loras else ""
            )
            print(
                f"\n[grade] loading DiT {dit_path.name} "
                f"(core={dit_provenance['core_revision']}, attn={dit_provenance['attention']}{lora_note}) "
                f"for denoised-prediction grading..."
            )
            raw_model = comfy.sd.load_diffusion_model(str(dit_path))
            if grade_loras:
                lb = cv_nodes._repo("lora_branch")
                for l_spec in grade_loras:
                    l_path = cv_nodes.resolve_lora_file_path(
                        l_spec["file"], expected_sha256=l_spec.get("sha256")
                    )
                    l_sd = lb.native_keys(
                        comfy.utils.load_torch_file(str(l_path), safe_load=True),
                        raw_model,
                    )
                    l_branches = lb.parse_lora(l_sd, float(l_spec.get("strength", 1.0)))
                    raw_model = lb.attach(raw_model, l_branches)
            base_model = _sigma_shifted(raw_model)
            if getattr(args, "sol", False):
                sol_mod = cv_nodes._repo("sol_attn_h3")
                sol_steps = (
                    h3_config.PIKA_STEPS if use_pika else h3_config.SAMPLING["steps"]
                )
                sol_cfg = h3_config.sol_for_graph(sol_steps)
                (base_model,) = sol_mod._apply_sol(
                    base_model,
                    tau=sol_cfg["tau"],
                    quantizer=sol_cfg["quantizer"],
                    dense_blocks=sol_cfg["dense_blocks"],
                    sink_conditioning=sol_cfg["sink_conditioning"],
                    token_routing=sol_cfg["token_routing"],
                    routing_blocks=sol_cfg.get("routing_blocks", ""),
                    start_percent=sol_cfg["start_percent"],
                    end_percent=sol_cfg["end_percent"],
                    min_tokens=sol_cfg["min_tokens"],
                    verbose=False,
                )
            latent_dict, _frames = _empty_av_latent(args.width, args.height, args.length)
            latent = latent_dict["samples"]
            noise = comfy.sample.prepare_noise(latent, args.noise_seed)
            if manual_sigmas_str:
                sigmas = torch.tensor(
                    [float(s.strip()) for s in manual_sigmas_str.split(",") if s.strip()],
                    dtype=torch.float32,
                )
            else:
                sigmas = comfy.samplers.calculate_sigmas(
                    base_model.get_model_object("model_sampling"),
                    h3_config.SAMPLING["scheduler"],
                    h3_config.SAMPLING["steps"],
                )
            dit_provenance["sigmas"] = [round(float(x), 6) for x in sigmas.tolist()]
            t_dit_loaded = time.perf_counter()
            dit_provenance["timings_s"]["dit_load_s"] = round(t_dit_loaded - t_dit_load_start, 3)
            print(f"[grade] DiT + latent ready in {t_dit_loaded - t_dit_load_start:.2f}s (canvas={args.width}x{args.height}x{args.length})")

            def _default_dit_runner(m_obj: Any, cond_obj: list[Any], step_idx: int) -> tuple[float, Any]:
                return (
                    float(sigmas[step_idx]),
                    _one_forward(m_obj, cond_obj, noise, latent, sigmas, step_idx, args.noise_seed),
                )

            dit_runner = _default_dit_runner
        else:
            base_model = getattr(args, "_base_model")

        forward_times_s: list[float] = []
        raw_dit_runner = dit_runner

        def _timed_dit_runner(m_obj: Any, cond_obj: list[Any], step_idx: int) -> tuple[float, Any]:
            tf0 = time.perf_counter()
            out = raw_dit_runner(m_obj, cond_obj, step_idx)
            forward_times_s.append(time.perf_counter() - tf0)
            return out

        dit_runner = _timed_dit_runner

        print(f"\n--- DiT Denoised-Prediction Movement (steps={probe_steps}) ---")
        for step_idx in probe_steps:
            sigma_val, pred_base_raw = dit_runner(base_model, base_cond, step_idx)
            base_b = _pred_blocks(pred_base_raw)

            # Hard check: strength 0.0 hooked reproduces plain base prediction exactly
            m_zero = _attach_raw_vector(
                base_model, metadata["name"], vec, 0.0, mask="non_vision", metadata=metadata
            )
            _, pred_zero_raw = dit_runner(m_zero, base_cond, step_idx)
            zero_b = _pred_blocks(pred_zero_raw)
            zero_metrics = _block_metrics(zero_b, base_b)
            if zero_metrics["video"]["relative_l2"] != 0.0 or zero_metrics["audio"]["relative_l2"] != 0.0:
                raise ValueError(
                    f"DiT strength=0.0 null check failed at step {step_idx}: "
                    f"video_rel_l2={zero_metrics['video']['relative_l2']} "
                    f"audio_rel_l2={zero_metrics['audio']['relative_l2']}"
                )

            step_arms: dict[str, Any] = {"strength_0_null": zero_metrics}
            edit_b: dict[str, torch.Tensor] | None = None
            if edit_cond is not None:
                _, pred_edit_raw = dit_runner(base_model, edit_cond, step_idx)
                edit_b = _pred_blocks(pred_edit_raw)
                step_arms["edit_prompt"] = _block_metrics(edit_b, base_b)
                em = step_arms["edit_prompt"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'edit_prompt':<22s} "
                    f"video_rel_l2={em['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={em['audio']['relative_l2']:.6f}"
                )

            if pad_cond is not None:
                _, pred_pad_raw = dit_runner(base_model, pad_cond, step_idx)
                pad_b = _pred_blocks(pred_pad_raw)
                step_arms["padding_control"] = _block_metrics(pad_b, base_b, edit_b)
                pm = step_arms["padding_control"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'padding_control':<22s} "
                    f"video_rel_l2={pm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={pm['audio']['relative_l2']:.6f}"
                )

            _, pred_period_raw = dit_runner(base_model, period_cond, step_idx)
            period_b = _pred_blocks(pred_period_raw)
            step_arms["period_null"] = _block_metrics(period_b, base_b, edit_b)
            prm = step_arms["period_null"]
            print(
                f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'period_null':<22s} "
                f"video_rel_l2={prm['video']['relative_l2']:.6f}  "
                f"audio_rel_l2={prm['audio']['relative_l2']:.6f}"
            )

            if flip_dir is not None:
                m_flip = _attach_raw_vector(
                    base_model, "flip_null", flip_dir, 1.0, mask="non_vision", metadata=metadata
                )
                _, pred_flip_raw = dit_runner(m_flip, base_cond, step_idx)
                flip_b = _pred_blocks(pred_flip_raw)
                step_arms["flip_null"] = _block_metrics(flip_b, base_b, edit_b)
                fm = step_arms["flip_null"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'flip_null (s=+1.0)':<22s} "
                    f"video_rel_l2={fm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={fm['audio']['relative_l2']:.6f}"
                )

            m_rand = _attach_raw_vector(
                base_model, "random_null", rand_dir, 1.0, mask="non_vision", metadata=metadata
            )
            _, pred_rand_raw = dit_runner(m_rand, base_cond, step_idx)
            rand_b = _pred_blocks(pred_rand_raw)
            step_arms["random_null"] = _block_metrics(rand_b, base_b, edit_b)
            rm = step_arms["random_null"]
            print(
                f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'random_null (s=+1.0)':<22s} "
                f"video_rel_l2={rm['video']['relative_l2']:.6f}  "
                f"audio_rel_l2={rm['audio']['relative_l2']:.6f}"
            )

            for s in strengths:
                if s == 0.0:
                    step_arms["vector_s0"] = zero_metrics
                    continue
                m_vec = _attach_raw_vector(
                    base_model, metadata["name"], vec, s, mask="non_vision", metadata=metadata
                )
                _, pred_vec_raw = dit_runner(m_vec, base_cond, step_idx)
                vec_b = _pred_blocks(pred_vec_raw)
                vm = _block_metrics(vec_b, base_b, edit_b)
                step_arms[f"vector_s{s:g}"] = vm
                extra_str = ""
                if edit_b is not None:
                    extra_str = (
                        f"  cos_edit(v={vm['video']['delta_cosine_with_edit']:+.4f},"
                        f"a={vm['audio']['delta_cosine_with_edit']:+.4f})"
                    )
                if n_vis > 0 or getattr(args, "compare_unmasked", False):
                    m_unmasked = _attach_raw_vector(
                        base_model, metadata["name"], vec, s, mask="all_rows", metadata=metadata
                    )
                    _, pred_unm_raw = dit_runner(m_unmasked, base_cond, step_idx)
                    unm_b = _pred_blocks(pred_unm_raw)
                    step_arms[f"vector_unmasked_s{s:g}"] = _block_metrics(unm_b, base_b, edit_b)
                    step_arms[f"masked_vs_unmasked_s{s:g}"] = _block_metrics(unm_b, vec_b)
                    extra_str += (
                        f"  unmasked_diff_v={step_arms[f'masked_vs_unmasked_s{s:g}']['video']['relative_l2']:.6f}"
                    )
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {f'vector (s={s:+.2f})':<22s} "
                    f"video_rel_l2={vm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={vm['audio']['relative_l2']:.6f}{extra_str}"
                )

            matched_strength_video: float | None = None
            if edit_b is not None and "vector_s1" in step_arms:
                s1_v = step_arms["vector_s1"]["video"]["relative_l2"]
                ed_v = step_arms["edit_prompt"]["video"]["relative_l2"]
                if s1_v > 1e-12:
                    matched_strength_video = ed_v / s1_v
                    print(
                        f"  [step {step_idx:2d}] matched_strength_video = {matched_strength_video:.4f} "
                        f"(edit_video_rel_l2={ed_v:.6f} / vec_s1_video_rel_l2={s1_v:.6f})"
                    )

            dit_results.append(
                {
                    "step": step_idx,
                    "sigma": sigma_val,
                    "matched_strength_video": matched_strength_video,
                    "arms": step_arms,
                }
            )
        if forward_times_s:
            dit_provenance["timings_s"]["n_forwards"] = len(forward_times_s)
            dit_provenance["timings_s"]["mean_forward_s"] = round(
                sum(forward_times_s) / len(forward_times_s), 3
            )
            dit_provenance["timings_s"]["total_forward_s"] = round(sum(forward_times_s), 3)
            print(
                f"\n  [grade] {len(forward_times_s)} DiT forwards in {sum(forward_times_s):.2f}s "
                f"(mean={sum(forward_times_s) / len(forward_times_s):.2f}s/forward)"
            )

    if getattr(args, "out", None):
        out_record = {
            "vector": metadata["name"],
            "vector_file": vpath.name,
            "prompt": prompt_label,
            "edit_prompt": args.edit_prompt,
            "seed": int(args.seed),
            "provenance": dit_provenance,
            "context_movement": context_report,
            "dit_prediction_movement": dit_results,
        }
        Path(args.out).write_text(json.dumps(out_record, indent=2) + "\n", encoding="utf-8")

    return 0


def _display_path(p: Path) -> str:
    resolved = p.resolve()
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    parts = resolved.parts
    if "ComfyUI-h3-explorations" in parts:
        idx = parts.index("ComfyUI-h3-explorations")
        return str(Path(*parts[idx + 1 :]))
    return p.name


def cmd_from_md(args: argparse.Namespace) -> int:
    md_path = Path(args.md_path)
    if not md_path.is_file():
        raise ValueError(f"markdown file not found: {md_path}")

    out_target: str | Path = (
        Path(args.out) if getattr(args, "out", None) else cv_nodes.PAIR_BANK_DIR
    )
    parsed = cv.parse_markdown_pairs(md_path.read_text(encoding="utf-8"))
    saved_toml = cv.write_markdown_to_toml(
        md_path,
        out_target,
        name=getattr(args, "name", None),
        mode=getattr(args, "mode", None),
        frame=getattr(args, "frame", "scene") or "scene",
        quality=getattr(args, "quality", None),
        a_side=getattr(args, "a_side", None),
        b_side=getattr(args, "b_side", None),
        test_prompt_out=getattr(args, "test_prompt_out", None),
        overwrite=bool(getattr(args, "overwrite", False)),
        bank_dir=cv_nodes.PROMPT_BANK_DIR,
    )
    pair_set = cv.load_pairs(saved_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    test_prompt = parsed["test_scene_prompt"]
    has_test_prompt = bool(test_prompt)

    print(f"=== Control Vector From-MD: {pair_set.name} ===")
    print(f"  md_file           : {_display_path(md_path)}")
    print(f"  saved_toml        : {_display_path(saved_toml)}")
    print(
        f"  name              : {pair_set.name}   mode: {pair_set.mode}   "
        f"frame: {pair_set.frame}   n_pairs: {len(pair_set.pairs)}"
    )
    if has_test_prompt:
        tp_msg = f"yes ({len(test_prompt.split())} words, {len(test_prompt)} chars)"
        if getattr(args, "test_prompt_out", None):
            tp_msg += f" -> {_display_path(Path(args.test_prompt_out))}"
    else:
        tp_msg = "no"
    print(f"  test_scene_prompt : {tp_msg}")
    print(f"  quality           : {pair_set.quality}")
    print(f"  a_side            : {pair_set.a_side}")
    print(f"  b_side            : {pair_set.b_side}")
    print(f"\n--- Pair Spans ({len(pair_set.pairs)} pairs) ---")
    for item in pair_set.pairs:
        sec_idx, sec_name = cv.differing_section_info(item.a, item.b, frame=pair_set.frame)
        wa = len(item.a_span.split())
        wb = len(item.b_span.split())
        print(
            f"  [{item.index:02d}] [sec {sec_idx}:{sec_name}] ({wa:2d}w -> {wb:2d}w) | "
            f"-{item.a_span[:48]!r} -> +{item.b_span[:48]!r}"
        )
    return 0


def _recommended_strength_text(raw_norm: float) -> str:
    if raw_norm <= 1e-6:
        return "1.0"
    s_lo = max(0.25, round(50.0 / raw_norm, 2))
    s_hi = max(s_lo + 0.25, round(85.0 / raw_norm, 2))
    return f"{s_lo:.2f} .. {s_hi:.2f}"


def cmd_build(args: argparse.Namespace) -> int:
    if getattr(args, "build_all", False):
        toml_dir = Path(getattr(args, "toml_dir", None) or cv_nodes.PAIR_BANK_DIR)
        all_tomls = sorted(toml_dir.glob("*.toml"))
        if not all_tomls:
            raise ValueError(f"no .toml pair files found in {toml_dir}")
        encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
        t_batch_0 = time.perf_counter()
        clip_shared = getattr(args, "_clip_override", None) or _load_clip(
            encoder_name, device=args.device
        )
        refiner_cache: dict[Any, Any] = (
            args._refiner_cache
            if isinstance(getattr(args, "_refiner_cache", None), dict)
            else {}
        )
        all_rows: list[dict[str, Any]] = []
        print(
            f"=== Batch Building {len(all_tomls)} Control Vector Banks from {_display_path(toml_dir)} ==="
        )
        for toml_path in all_tomls:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.build_all = False
            sub_args.source = str(toml_path)
            sub_args.name = None
            sub_args.output_name = ""
            sub_args._clip_override = clip_shared
            sub_args._refiner_cache = refiner_cache
            sub_args._summary_rows_out = all_rows
            cmd_build(sub_args)
        if getattr(args, "_summary_rows_out", None) is not None:
            args._summary_rows_out.extend(all_rows)
        t_batch_elapsed = time.perf_counter() - t_batch_0
        print(
            f"\n=== Batch Build Complete: {len(all_rows)} vector file(s) from {len(all_tomls)} TOML bank(s) in {t_batch_elapsed:.2f}s ==="
        )
        return 0

    from_words_str = str(getattr(args, "from_words", "") or "").strip()
    if from_words_str:
        t_w0 = time.perf_counter()
        raw_stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or "").strip()
        if not raw_stem:
            first_w = [w.strip().lower() for part in from_words_str.split(",") for w in part.split() if w.strip()]
            raw_stem = "_".join(first_w[:3]) if first_w else "word_vector"
        stem = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw_stem.lower()).strip("_") or "word_vector"
        if not cv.NAME_RE.match(stem):
            raise ValueError(f"output vector stem must match [a-z0-9_]+, got {stem!r}")

        target_fams = _resolve_requested_families(getattr(args, "families", "both"))
        out_dir = (
            Path(args.out_dir)
            if getattr(args, "out_dir", None)
            else Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        against_w = str(getattr(args, "against_words", "") or "").strip() or None
        hold_w = str(getattr(args, "hold_words", "") or "").strip() or None
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))
        prompt_w = str(getattr(args, "prompt", "") or "").strip() or None

        summary_rows_w: list[dict[str, Any]] = []
        for fam in target_fams:
            synth = j_lens.synthesize_from_words(
                from_words_str,
                against_w,
                family=fam,
                hold_words=hold_w,
                alpha=hold_alpha,
                prompt=prompt_w,
            )
            loras_meta = j_lens._DEFAULT_RECIPES.get(fam, {}).get("loras")
            fname = cv.default_vector_filename(
                stem,
                fam,
                pooling="diff_span",
                loras=loras_meta,
            )
            saved = j_lens.save_synthesized_vector(out_dir / fname, synth, name=stem)
            r_norm = float(synth["readout"]["norm_after_holds"])
            summary_rows_w.append(
                {
                    "file": saved.name,
                    "family": fam,
                    "pooling": "diff_span (j_lens words)",
                    "raw_norm": r_norm,
                    "mpc": 1.0,
                    "split_half": 1.0,
                    "rec_strength": _recommended_strength_text(r_norm),
                    "j_lens": synth["readout"],
                }
            )
        if getattr(args, "_summary_rows_out", None) is not None:
            args._summary_rows_out.extend(summary_rows_w)
        elapsed_w = time.perf_counter() - t_w0
        print(
            f"\n=== Ready to Render in ComfyUI ({len(summary_rows_w)} word-synthesized vector file(s) built in {elapsed_w * 1000.0:.1f} ms on CPU) ==="
        )
        for row in summary_rows_w:
            jl = row.get("j_lens") or {}
            rd = jl.get("readiness") or {}
            print(
                f"  * {row['file']}\n"
                f"      family={row['family']}  pooling={row['pooling']}  "
                f"raw_norm={row['raw_norm']:.2f}  recommended strength: {row['rec_strength']}\n"
                f"      j_lens (+): {jl.get('pos_summary', '')}\n"
                f"      j_lens (-): {jl.get('neg_summary', '')}"
            )
            if rd:
                print(f"      readiness : {rd.get('summary', '')}")
                for rec_line in rd.get("recommendations", ()):
                    print(f"                  -> {rec_line}")
            pc = jl.get("prompt_coupling") or {}
            if pc:
                print(f"      coupling  : {pc.get('summary', '')}")
        return 0

    words_arg = str(getattr(args, "words", "") or "").strip()
    is_placebo = bool(getattr(args, "placebo", False))
    compiled_override: cv.PairSet | None = getattr(args, "_compiled_pair_set", None)

    if (
        not is_placebo
        and not words_arg
        and compiled_override is None
        and (not getattr(args, "source", None) or not str(args.source).strip())
    ):
        raise ValueError("either a source (.md/.toml/stem), --from-words, --words, --placebo, or --all is required for 'build'")

    raw_poolings = [p.strip() for p in str(args.pooling).split(",") if p.strip()]
    if not raw_poolings:
        raise ValueError("--pooling must specify at least one pooling mode")
    poolings: list[str] = []
    for pm in raw_poolings:
        if pm == "all":
            for cand in ("diff_span", "from_first_diff", "scene_text"):
                if cand not in poolings:
                    poolings.append(cand)
        elif pm in cv.VALID_POOLING:
            if pm not in poolings:
                poolings.append(pm)
        else:
            raise ValueError(
                f"pooling must be one of {cv.VALID_POOLING} or 'all', got {pm!r}"
            )

    resolved_pair_path: Path | None = None
    test_prompt_saved: Path | None = None
    stem_override = (getattr(args, "name", None) or getattr(args, "output_name", None) or "").strip() or None
    compile_mode = getattr(args, "mode", None) or ("ref2va" if getattr(args, "ref_images", None) else "t2va")
    slot_arg = str(getattr(args, "slot", "auto") or "auto")
    tp_arg = str(getattr(args, "target_prompt", "") or "").strip()
    tp_weight = float(getattr(args, "target_weight", 2.5))

    if compiled_override is not None:
        pair_set = compiled_override
        if tp_arg:
            pair_set = _inject_target_prompt_into_pairset(
                pair_set, tp_arg, target_weight=tp_weight, slot_hint=slot_arg
            )
    elif is_placebo:
        pair_set = pair_compiler.compile_placebo_pairset(
            slot=slot_arg if slot_arg != "auto" else "opening_framing",
            mode=compile_mode,
        )
    elif words_arg:
        pair_set = _compile_words_or_spec_to_pairset(
            words_arg,
            name=stem_override,
            quality=getattr(args, "quality", None),
            a_side=getattr(args, "a_side", None),
            b_side=getattr(args, "b_side", None),
            slot=slot_arg,
            mode=compile_mode,
            target_prompt=tp_arg,
            target_weight=tp_weight,
        )
        toml_out_arg = getattr(args, "toml_out", None) or (
            cv_nodes.PAIR_BANK_DIR if getattr(args, "save_toml", False) else None
        )
        if toml_out_arg:
            t_out = Path(toml_out_arg)
            if t_out.suffix.lower() != ".toml":
                t_out = t_out / f"{pair_set.name}.toml"
            md_doc_for_toml = _pair_set_to_markdown(None, pair_set)
            resolved_pair_path = cv.write_markdown_to_toml(
                md_doc_for_toml,
                t_out,
                name=t_out.stem,
                mode=pair_set.mode,
                frame=pair_set.frame,
                quality=getattr(args, "quality", None) or pair_set.quality,
                a_side=getattr(args, "a_side", None) or pair_set.a_side,
                b_side=getattr(args, "b_side", None) or pair_set.b_side,
                overwrite=bool(getattr(args, "overwrite", False)),
                bank_dir=cv_nodes.PROMPT_BANK_DIR,
            )
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    else:
        src_arg = str(args.source).strip()
        src_path = Path(src_arg)
        if src_path.suffix.lower() == ".md" or (src_path.is_file() and src_arg.lower().endswith(".md")):
            if not src_path.is_file():
                raise ValueError(f"markdown file not found: {src_path}")
            md_text = src_path.read_text(encoding="utf-8")
            parsed = cv.parse_markdown_pairs(md_text)
            toml_target_path = (
                Path(args.toml_out) if getattr(args, "toml_out", None) else cv_nodes.PAIR_BANK_DIR
            )
            eff_stem = (
                stem_override
                if stem_override
                else (toml_target_path.stem if toml_target_path.suffix.lower() == ".toml" else src_path.stem)
            )
            tp_out = getattr(args, "test_prompt_out", None)
            if tp_out is None and parsed.get("test_scene_prompt"):
                base_dir = toml_target_path if toml_target_path.suffix.lower() != ".toml" else toml_target_path.parent
                tp_out = base_dir / f"{eff_stem}_test_prompt.txt"
            try:
                resolved_pair_path = cv.write_markdown_to_toml(
                    src_path,
                    toml_target_path,
                    name=stem_override,
                    mode=getattr(args, "mode", None),
                    frame=getattr(args, "frame", "scene") or "scene",
                    quality=getattr(args, "quality", None),
                    a_side=getattr(args, "a_side", None),
                    b_side=getattr(args, "b_side", None),
                    test_prompt_out=tp_out,
                    overwrite=bool(getattr(args, "overwrite", False)),
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            except (ValueError, FileExistsError) as exc:
                if "already exists" not in str(exc):
                    raise
                cand_toml = (
                    toml_target_path
                    if toml_target_path.suffix.lower() == ".toml"
                    else toml_target_path / f"{eff_stem}.toml"
                )
                if not cand_toml.is_file():
                    raise
                existing_ps = cv.load_pairs(cand_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                existing_ab = [(p.a, p.b) for p in existing_ps.pairs]
                new_ab = [(p["a"], p["b"]) for p in parsed["pairs"]]
                existing_holds = [(h.group, h.a, h.b) for h in existing_ps.holds]
                new_holds = [(h["group"], h["a"], h["b"]) for h in parsed.get("holds", [])]
                if existing_ab != new_ab or existing_holds != new_holds:
                    raise
                resolved_pair_path = cand_toml
                if tp_out is not None and parsed.get("test_scene_prompt"):
                    Path(tp_out).parent.mkdir(parents=True, exist_ok=True)
                    Path(tp_out).write_text(parsed["test_scene_prompt"] + "\n", encoding="utf-8")
            if tp_out is not None and Path(tp_out).is_file():
                test_prompt_saved = Path(tp_out)
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
        else:
            resolved_pair_path = cv_nodes.resolve_pair_file_path(src_arg)
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
        if tp_arg:
            pair_set = _inject_target_prompt_into_pairset(
                pair_set, tp_arg, target_weight=tp_weight, slot_hint=slot_arg
            )

    stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or pair_set.name).strip()
    if not cv.NAME_RE.match(stem):
        raise ValueError(f"output vector stem must match [a-z0-9_]+, got {stem!r}")

    dit_arg = getattr(args, "dit", None) or getattr(args, "unet", None)
    if dit_arg:
        family_targets = [("custom", str(dit_arg))]
    else:
        auto_fam = "ref2va" if (getattr(args, "ref_images", None) or pair_set.mode == "ref2va") else "fl2va"
        req_fams = _resolve_requested_families(getattr(args, "families", "both"), auto_family=auto_fam)
        family_targets = [
            (f, h3_config.MODELS["unet_ref2va"] if f == "ref2va" else h3_config.MODELS["unet_fl2va"])
            for f in req_fams
        ]

    encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
    encoder_path = Path(folder_paths.get_full_path_or_raise("text_encoders", encoder_name))

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()

    encoded_refs = None
    if getattr(args, "ref_images", None):
        imgs = [s.strip() for s in str(args.ref_images).split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    if encoded_refs is not None:
        ref_hash = hashlib.sha256()
        for sp in getattr(encoded_refs, "spans", ()):
            ref_hash.update(cv._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    print(f"=== Control Vector One-Shot Build: {stem} ===")
    pairs_label = _display_path(resolved_pair_path) if resolved_pair_path is not None else str(pair_set.path)
    print(f"  pairs_toml        : {pairs_label} ({len(pair_set.pairs)} pairs)")
    if test_prompt_saved is not None:
        print(f"  test_prompt_file  : {_display_path(test_prompt_saved)}")
    print(f"  poolings          : {', '.join(poolings)}")
    print(f"  encoder           : {encoder_path.name} (loaded in {t_clip - t0:.2f}s)")

    # Encode all pairs and holds once through Qwen3-VL layers.49
    t_enc_start = time.perf_counter()
    cached_pairs: list[dict[str, Any]] = []
    enc_censuses: list[dict[str, float]] = []
    weights_list: list[float] = []
    print(f"\n--- Encoding {len(pair_set.pairs)} Contrastive Pairs (Single Pass) ---")
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_refs
        )
        enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_refs
        )
        enc_censuses.append(cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(cv.row_norm_census(enc_b, tags_b))
        first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
        spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
        cached_pairs.append(
            {
                "idx": idx,
                "item": item,
                "enc_a": enc_a,
                "tags_a": tags_a,
                "ids_a": ids_a,
                "spans_a": spans_a,
                "enc_b": enc_b,
                "tags_b": tags_b,
                "ids_b": ids_b,
                "spans_b": spans_b,
                "first_diff": first_diff,
                "end_a": end_a,
                "end_b": end_b,
            }
        )
        weights_list.append(float(item.weight))
        print(
            f"  [{idx:02d}] w={item.weight:.2f} diff_toks=[{first_diff}:{end_a}/{end_b}] "
            f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} | "
            f"-{item.a_span[:42]!r} -> +{item.b_span[:42]!r}"
        )

    cached_holds: list[dict[str, Any]] = []
    if not getattr(args, "no_holds", False) and pair_set.holds:
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = cv_nodes.encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_refs
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = cv_nodes.encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_refs
            )
            h_diff, h_end_a, h_end_b = h3_spans.differing_token_bounds(h_ids_a, h_ids_b)
            cached_holds.append(
                {
                    "h_item": h_item,
                    "enc_a": h_enc_a,
                    "ids_a": h_ids_a,
                    "spans_a": h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a),
                    "enc_b": h_enc_b,
                    "ids_b": h_ids_b,
                    "spans_b": h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b),
                    "first_diff": h_diff,
                    "end_a": h_end_a,
                    "end_b": h_end_b,
                }
            )
    t_enc_done = time.perf_counter()
    print(f"  encoded {len(pair_set.pairs)} pairs in {t_enc_done - t_enc_start:.2f}s")

    w_vec = torch.tensor(weights_list, dtype=torch.float32)
    median_enc_row_norm_ex0 = float(
        sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses)
    )
    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    out_dir = (
        Path(args.out_dir)
        if getattr(args, "out_dir", None)
        else Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    build_cache_out: dict[str, Any] | None = getattr(args, "_build_cache_out", None)
    hold_specs_build = getattr(args, "hold", None)
    hold_words_build = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha_build = float(getattr(args, "hold_alpha", 1.0))
    prompt_build = str(getattr(args, "prompt", "") or getattr(args, "target_prompt", "") or "").strip() or None

    for fam_hint, dit_name in family_targets:
        dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        family = cv.detect_family(dit_path)
        if getattr(args, "no_lora", False):
            loras: list[dict[str, Any]] = []
        elif getattr(args, "lora", None):
            loras = _parse_lora_specs(args.lora)
        else:
            default_lora_name = (
                h3_config.PIKA_REF2VA_SVD995_LORA
                if (family == "ref2va" or fam_hint == "ref2va")
                else h3_config.PIKA_FL2VA_SVD995_LORA
            )
            try:
                loras = _parse_lora_specs([f"{default_lora_name}:1.0"])
            except Exception:
                loras = []

        ref_cache = getattr(args, "_refiner_cache", None)
        cache_key = (
            str(dit_path),
            str(encoder_path),
            str(args.refiner_device),
            json.dumps(loras, sort_keys=True),
        )
        if isinstance(ref_cache, dict) and cache_key in ref_cache:
            family, recipe, refiner_run = ref_cache[cache_key]
        else:
            recipe = cv_nodes.extract_model_recipe(
                None,
                None,
                dit_path_override=dit_path,
                encoder_path_override=encoder_path,
                loras_override=loras,
            )
            refiner_run = cv_nodes.build_refiner_runner(
                dit_path, device=args.refiner_device, loras=loras
            )
            if isinstance(ref_cache, dict):
                ref_cache[cache_key] = (family, recipe, refiner_run)

        # Refine all pairs & holds once for this (family, lora) target
        refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        ref_censuses: list[dict[str, float]] = []
        for cp in cached_pairs:
            ra = refiner_run(cp["enc_a"])
            rb = refiner_run(cp["enc_b"])
            refined_pairs.append((ra, rb))
            ref_censuses.append(cv.row_norm_census(ra, cp["tags_a"]))
            ref_censuses.append(cv.row_norm_census(rb, cp["tags_b"]))
        median_ref_row_norm = float(
            sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses)
        )

        refined_holds: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ch in cached_holds:
            refined_holds.append((refiner_run(ch["enc_a"]), refiner_run(ch["enc_b"])))

        for pool_mode in poolings:
            a_pooled_list: list[torch.Tensor] = []
            b_pooled_list: list[torch.Tensor] = []
            enc_a_pooled_list: list[torch.Tensor] = []
            enc_b_pooled_list: list[torch.Tensor] = []
            pair_spans: list[dict[str, Any]] = []

            for cp, (ref_a, ref_b) in zip(cached_pairs, refined_pairs):
                first_diff = cp["first_diff"]
                end_a = cp["end_a"]
                end_b = cp["end_b"]
                ids_a = cp["ids_a"]
                ids_b = cp["ids_b"]
                item = cp["item"]
                mask_a = h3_spans.pooling_mask(
                    cp["spans_a"],
                    pool_mode,
                    first_diff=min(first_diff, len(ids_a) - 1),
                    end_diff=end_a,
                    other_ids=ids_b,
                )
                mask_b = h3_spans.pooling_mask(
                    cp["spans_b"],
                    pool_mode,
                    first_diff=min(first_diff, len(ids_b) - 1),
                    end_diff=end_b,
                    other_ids=ids_a,
                )
                a_pooled_list.append(
                    ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32)
                )
                b_pooled_list.append(
                    ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32)
                )
                if getattr(args, "include_encoder", True):
                    mask_a_enc = mask_a.clone()
                    mask_b_enc = mask_b.clone()
                    if mask_a_enc.numel() > 0:
                        mask_a_enc[0] = False
                    if mask_b_enc.numel() > 0:
                        mask_b_enc[0] = False
                    if not bool(mask_a_enc.any()):
                        mask_a_enc = mask_a
                    if not bool(mask_b_enc.any()):
                        mask_b_enc = mask_b
                    enc_a_pooled_list.append(
                        cp["enc_a"].to(dtype=torch.float64)[mask_a_enc].mean(dim=0).to(dtype=torch.float32)
                    )
                    enc_b_pooled_list.append(
                        cp["enc_b"].to(dtype=torch.float64)[mask_b_enc].mean(dim=0).to(dtype=torch.float32)
                    )
                pair_spans.append(
                    {
                        "index": int(cp["idx"]),
                        "weight": float(item.weight),
                        "first_diff": int(first_diff),
                        "end_a": int(end_a),
                        "end_b": int(end_b),
                        "a_span": item.a_span,
                        "b_span": item.b_span,
                        "len_a": int(len(ids_a)),
                        "len_b": int(len(ids_b)),
                        "pooled_rows_a": int(mask_a.sum().item()),
                        "pooled_rows_b": int(mask_b.sum().item()),
                        "pooled_tokens": int(mask_b.sum().item()),
                    }
                )

            a_mat = torch.stack(a_pooled_list, dim=0)
            b_mat = torch.stack(b_pooled_list, dim=0)
            use_unit_mean = bool(getattr(args, "unit_mean", False))
            raw_vec = cv.mean_difference(a_mat, b_mat, w_vec, unit_mean=use_unit_mean)
            tensors_to_save: dict[str, torch.Tensor] = {}

            hold_a_list: list[torch.Tensor] = []
            hold_b_list: list[torch.Tensor] = []
            enc_hold_a_list: list[torch.Tensor] = []
            enc_hold_b_list: list[torch.Tensor] = []
            hold_groups: list[str] = []
            hold_weights: list[float] = []
            if cached_holds:
                for ch, (h_ref_a, h_ref_b) in zip(cached_holds, refined_holds):
                    h_diff = ch["first_diff"]
                    h_ids_a = ch["ids_a"]
                    h_ids_b = ch["ids_b"]
                    h_mask_a = h3_spans.pooling_mask(
                        ch["spans_a"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_a) - 1),
                        end_diff=ch["end_a"],
                        other_ids=h_ids_b,
                    )
                    h_mask_b = h3_spans.pooling_mask(
                        ch["spans_b"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_b) - 1),
                        end_diff=ch["end_b"],
                        other_ids=h_ids_a,
                    )
                    hold_a_list.append(
                        h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
                    )
                    hold_b_list.append(
                        h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
                    )
                    if getattr(args, "include_encoder", True):
                        hma_e = h_mask_a.clone()
                        hmb_e = h_mask_b.clone()
                        if hma_e.numel() > 0:
                            hma_e[0] = False
                        if hmb_e.numel() > 0:
                            hmb_e[0] = False
                        if not bool(hma_e.any()):
                            hma_e = h_mask_a
                        if not bool(hmb_e.any()):
                            hmb_e = h_mask_b
                        enc_hold_a_list.append(
                            ch["enc_a"].to(dtype=torch.float64)[hma_e].mean(dim=0).to(dtype=torch.float32)
                        )
                        enc_hold_b_list.append(
                            ch["enc_b"].to(dtype=torch.float64)[hmb_e].mean(dim=0).to(dtype=torch.float32)
                        )
                    hold_groups.append(ch["h_item"].group)
                    hold_weights.append(float(ch["h_item"].weight))
                basis = cv.hold_basis(
                    torch.stack(hold_a_list, dim=0),
                    torch.stack(hold_b_list, dim=0),
                    hold_groups,
                    torch.tensor(hold_weights, dtype=torch.float32),
                )
                final_vec = cv.project_out(raw_vec, basis)
                tensors_to_save["holds.post_refiner"] = basis
            else:
                final_vec = raw_vec

            applied_holds_build: list[str] = []
            norm_bef_build = 0.0
            norm_aft_build = 0.0
            if not getattr(args, "no_holds", False) and (hold_specs_build or hold_words_build):
                final_vec, applied_holds_build, norm_bef_build, norm_aft_build = _apply_post_refiner_holds(
                    final_vec,
                    family=family,
                    hold_specs=hold_specs_build,
                    hold_words=hold_words_build,
                    hold_alpha=hold_alpha_build,
                    prompt=prompt_build,
                )

            tensors_to_save["vector.post_refiner"] = final_vec
            gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
            q_report = cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
            raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())

            points_meta: dict[str, Any] = {
                "post_refiner": {
                    "unit": "mean_pair_difference",
                    "raw_norm": raw_norm,
                    "median_row_norm": median_ref_row_norm,
                }
            }
            enc_a_mat: torch.Tensor | None = None
            enc_b_mat: torch.Tensor | None = None
            if getattr(args, "include_encoder", True):
                enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
                enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
                enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec, unit_mean=use_unit_mean)
                if enc_hold_a_list:
                    enc_basis = cv.hold_basis(
                        torch.stack(enc_hold_a_list, dim=0),
                        torch.stack(enc_hold_b_list, dim=0),
                        hold_groups,
                        torch.tensor(hold_weights, dtype=torch.float32),
                    )
                    enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
                enc_raw_norm = float(
                    torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item()
                )
                if enc_raw_norm <= 1e-12:
                    raise ValueError("vector.encoder has zero norm")
                enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(
                    dtype=torch.float32
                )
                tensors_to_save["vector.encoder"] = enc_vec
                points_meta["encoder"] = {
                    "unit": "median_row_norm",
                    "raw_norm": enc_raw_norm,
                    "median_row_norm": median_enc_row_norm_ex0,
                    "row0_excluded": True,
                }

            jl_build_readout: dict[str, Any] | None = None
            if j_lens.bundle_available():
                try:
                    jl_build_readout = j_lens.decode_vector(
                        final_vec, space="post_refiner", family=family
                    )
                    if prompt_build:
                        jl_build_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                            final_vec,
                            prompt_build,
                            readout=jl_build_readout,
                            family=family,
                        )
                    q_report["j_lens_readout"] = jl_build_readout
                except Exception:
                    jl_build_readout = None

            measured_mode = "ref2va" if encoded_refs is not None else pair_set.mode
            metadata: dict[str, Any] = {
                "format": cv.FORMAT_VERSION,
                "name": stem,
                "quality": pair_set.quality,
                "family": family,
                "pairs": {
                    "path": rel_pair_path,
                    "sha256": pair_set.sha256,
                    "n_pairs": len(pair_set.pairs),
                    "n_holds": len(cached_holds) + len(applied_holds_build),
                    "a_side": pair_set.a_side,
                    "b_side": pair_set.b_side,
                    "pair_spans": pair_spans,
                },
                "recipe": recipe,
                "measured_under": {
                    "mode": measured_mode,
                    "frame": pair_set.frame,
                    "prefix": prefix_meta,
                    "pooling": pool_mode,
                },
                "points": points_meta,
                "quality_report": q_report,
                "provenance": {
                    "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
                    "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
                    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "made_by": "bench/control_vectors.py",
                    **(
                        {
                            "applied_holds": {
                                "holds": applied_holds_build,
                                "alpha": hold_alpha_build,
                                "norm_before": round(norm_bef_build, 4),
                                "norm_after_raw": round(norm_aft_build, 4),
                            }
                        }
                        if applied_holds_build
                        else {}
                    ),
                },
            }

            fname = cv.default_vector_filename(
                stem,
                family,
                pooling=pool_mode,
                prefix=prefix_meta,
                apply_holds=not bool(getattr(args, "no_holds", False)),
                n_holds=len(pair_set.holds) + len(applied_holds_build),
                loras=loras,
            )
            out_path = out_dir / fname
            saved = cv.save_vector(
                out_path,
                tensors_to_save,
                metadata,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
            if isinstance(build_cache_out, dict):
                build_cache_out[f"{family}:{pool_mode}"] = {
                    "family": family,
                    "pooling": pool_mode,
                    "stem": stem,
                    "out_path": saved,
                    "a_mat": a_mat.detach().cpu(),
                    "b_mat": b_mat.detach().cpu(),
                    "enc_a_mat": enc_a_mat.detach().cpu() if enc_a_mat is not None else None,
                    "enc_b_mat": enc_b_mat.detach().cpu() if enc_b_mat is not None else None,
                    "weights": [float(w) for w in weights_list],
                    "hold_a_list": [t.detach().cpu() for t in hold_a_list],
                    "hold_b_list": [t.detach().cpu() for t in hold_b_list],
                    "enc_hold_a_list": [t.detach().cpu() for t in enc_hold_a_list],
                    "enc_hold_b_list": [t.detach().cpu() for t in enc_hold_b_list],
                    "hold_groups": list(hold_groups),
                    "hold_weights": list(hold_weights),
                    "median_ref_row_norm": median_ref_row_norm,
                    "median_enc_row_norm_ex0": median_enc_row_norm_ex0,
                    "pair_spans": list(pair_spans),
                    "pairs_ab": [(p.a, p.b) for p in pair_set.pairs],
                    "pair_set": pair_set,
                    "rel_pair_path": rel_pair_path,
                    "recipe": recipe,
                    "loras": loras,
                    "prefix_meta": prefix_meta,
                    "measured_mode": measured_mode,
                    "dit_path": dit_path,
                    "unet_path": dit_path,
                    "encoder_path": encoder_path,
                    "refiner_device": str(args.refiner_device),
                    "seed": int(args.seed),
                }
            rec_band = [
                round(50.0 / max(raw_norm, 1e-6), 2),
                round(85.0 / max(raw_norm, 1e-6), 2),
            ]
            summary_rows.append(
                {
                    "file": saved.name,
                    "family": family,
                    "pooling": pool_mode,
                    "raw_norm": raw_norm,
                    "vector_norm": raw_norm,
                    "mpc": float(q_report["mean_pairwise_cosine"]),
                    "split_half": float(q_report["split_half_cosine"]),
                    "split_half_cosine": float(q_report["split_half_cosine"]),
                    "per_pair_cosine": [round(float(x), 4) for x in q_report["per_pair_cosine"]],
                    "pair_spans": pair_spans,
                    "rec_strength": _recommended_strength_text(raw_norm),
                    "recommended_strength_band": rec_band,
                    "j_lens": jl_build_readout,
                }
            )

    if getattr(args, "_summary_rows_out", None) is not None:
        args._summary_rows_out.extend(summary_rows)
    elapsed = time.perf_counter() - t0
    print(f"\n=== Ready to Render in ComfyUI ({len(summary_rows)} vector file(s) built in {elapsed:.2f}s) ===")
    for row in summary_rows:
        print(
            f"  * {row['file']}\n"
            f"      family={row['family']}  pooling={row['pooling']}  "
            f"raw_norm={row['raw_norm']:.2f}  cos={row['mpc']:.4f} (split_half={row['split_half']:.4f})  "
            f"recommended strength: {row['rec_strength']}"
        )
        jl_row = row.get("j_lens")
        if isinstance(jl_row, dict) and jl_row.get("pos_summary"):
            print(f"      j_lens (+): {jl_row['pos_summary']}")
            print(f"      j_lens (-): {jl_row['neg_summary']}")
    if test_prompt_saved is not None:
        print(f"\n  Test Scene Prompt saved to: {_display_path(test_prompt_saved)}")
    return 0


class ControlVectorUIState:
    def __init__(
        self,
        *,
        toml_dir: Path | None = None,
        vec_dir: Path | None = None,
        default_encoder: str | None = None,
        default_dit: str | None = None,
        default_device: str | None = None,
        default_llm_provider: str | None = None,
        default_llm_model: str | None = None,
    ) -> None:
        self.toml_dir: Path = toml_dir or Path(
            os.environ.get("H3_CV_TOML_DIR", "").strip() or cv_nodes.PAIR_BANK_DIR
        )
        self.vec_dir: Path = vec_dir or Path(
            os.environ.get("H3_CV_VEC_DIR", "").strip()
            or folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0]
        )
        self.default_encoder: str = (
            default_encoder
            or os.environ.get("H3_ENCODER", "").strip()
            or h3_config.MODELS["clip"]
        )
        self.default_dit: str | None = (
            default_dit or os.environ.get("H3_DIT", "").strip() or None
        )
        self.default_device: str = (
            default_device or os.environ.get("H3_DEVICE", "").strip() or "default"
        )
        self.default_llm_provider: str = (
            default_llm_provider
            or os.environ.get("H3_LLM_PROVIDER", "").strip()
            or ("exllamav3" if os.environ.get("H3_EXL3_MODEL_DIR", "").strip() else "transformers")
        )
        self.default_llm_model: str = (
            default_llm_model
            or os.environ.get("H3_LLM_MODEL", "").strip()
            or os.environ.get("H3_EXL3_MODEL_DIR", "").strip()
        )
        self._tokenizer: Any = None
        self._clips: dict[tuple[str, str], Any] = {}
        self._refiner_cache: dict[Any, Any] = {}
        self._llm_cache: dict[tuple[str, str, bool], Any] = {}
        self.last_build_cache: dict[str, Any] = {}
        self.shutdown_requested: bool = False

    def get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            import comfy.text_encoders.minimax as mm_te

            self._tokenizer = mm_te.MiniMaxH3Tokenizer()
        return self._tokenizer

    def get_clip(self, encoder_name: str | None = None, device: str | None = None) -> Any:
        resolved_enc = encoder_name or self.default_encoder or h3_config.MODELS["clip"]
        resolved_dev = device or self.default_device or "default"
        key = (str(resolved_enc), str(resolved_dev))
        if key not in self._clips:
            self._clips[key] = _load_clip(resolved_enc, device=str(resolved_dev))
        return self._clips[key]


def _extract_system_prompt_text() -> str:
    candidates = [
        REPO_ROOT / "docs" / "wiki" / "system_prompts" / "h3_control_vector_system_prompt.md",
        REPO_ROOT / "docs" / "research" / "control_vectors" / "2026-11-05_control_vector_system_prompt.md",
    ]
    open_fence = "````markdown\n"
    for sp_path in candidates:
        if not sp_path.is_file():
            continue
        raw = sp_path.read_text(encoding="utf-8")
        fence_start = raw.find(open_fence)
        if fence_start >= 0:
            body_start = fence_start + len(open_fence)
            fence_end = raw.find("\n````", body_start)
            if fence_end >= 0:
                return raw[body_start:fence_end].strip()
    return ""


SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("transformers", "exllamav3")


def _resolve_exl3_library_root(explicit_root: str | None = None) -> Path | None:
    """Locate the `exllamav3` Python package root from `explicit_root`, `EXLLAMAV3_ROOT`, or sibling directories."""
    for raw in (explicit_root, os.environ.get("EXLLAMAV3_ROOT", "")):
        s = str(raw or "").strip()
        if s and (Path(s).expanduser() / "exllamav3").is_dir():
            return Path(s).expanduser().resolve()
    for parent in [REPO_ROOT.parent, *REPO_ROOT.parents]:
        if (parent / "exllamav3").is_dir():
            return parent
    base = REPO_ROOT.parents[2] if len(REPO_ROOT.parents) >= 3 else REPO_ROOT.parent
    for cand in base.glob("*/*"):
        if cand.name == "exllamav3" and (cand / "exllamav3").is_dir():
            return cand.resolve()
    return None


def _generate_pairs_with_llm(
    *,
    provider: str,
    model_spec: str,
    concept_prompt: str,
    system_prompt: str | None = None,
    load_in_4bit: bool = True,
    max_new_tokens: int = 3200,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    rep_p: float = 1.05,
    seed: int = 42,
    exl3_root: str | None = None,
    llm_cache: dict[tuple[str, str, bool], Any] | None = None,
) -> dict[str, Any]:
    """Generate H3 control-vector Markdown pairs using a user-selected `provider` (`transformers` or `exllamav3`)
    and any user-specified `model_spec` (local directory path or HuggingFace repo ID).
    """
    prov = str(provider or "").strip().lower()
    if prov in ("exl3", "exllama", "exllamav3"):
        prov = "exllamav3"
    elif prov in ("hf", "huggingface", "transformers"):
        prov = "transformers"
    else:
        raise ValueError(
            f"unsupported LLM provider {provider!r}; expected one of {SUPPORTED_LLM_PROVIDERS}"
        )

    mspec = str(model_spec or "").strip()
    if not mspec:
        mspec = (
            os.environ.get("H3_LLM_MODEL", "").strip()
            or os.environ.get("H3_EXL3_MODEL_DIR", "").strip()
        )
    if not mspec:
        raise ValueError(
            "LLM model path or HuggingFace model ID is required (pass --llm-model, set H3_LLM_MODEL, or enter it in the Studio UI)"
        )

    concept = str(concept_prompt or "").strip()
    if not concept:
        raise ValueError("concept description is required to generate pairs")

    sys_txt = (system_prompt or _extract_system_prompt_text()).strip()
    if not sys_txt:
        sys_txt = (
            "You are an expert MiniMax H3 prompt engineer. Author 10 contrasting Prompt A (Baseline) / "
            "Prompt B (Steered) pairs in Markdown using `#### Pair N`, `##### Prompt A (Baseline)`, and "
            "`##### Prompt B (Steered)` headings where only the target concept clause changes."
        )

    user_msg = (
        f"Target Control Vector Concept / Axis:\n{concept}\n\n"
        "Generate the complete Markdown control-vector pack (`### Test Scene Prompt` followed by "
        "`#### Pair 1` through `#### Pair 10` with `##### Prompt A (Baseline)` and `##### Prompt B (Steered)`)."
    )

    t0 = time.perf_counter()
    cache_store = llm_cache if llm_cache is not None else {}
    cache_key = (prov, mspec, bool(load_in_4bit))

    if prov == "exllamav3":
        ex_root = _resolve_exl3_library_root(exl3_root)
        if ex_root is not None and str(ex_root) not in sys.path:
            sys.path.insert(0, str(ex_root))
        try:
            from exllamav3 import Cache, CacheLayer_quant, ComboSampler, Config, Model, Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ExLlamaV3 ('exllamav3') is not importable; set EXLLAMAV3_ROOT to your exllamav3 checkout or select the 'transformers' provider."
            ) from exc

        model_dir = Path(mspec).expanduser()
        if not model_dir.is_dir():
            raise ValueError(f"ExLlamaV3 model directory not found: {mspec!r}")

        if cache_key in cache_store:
            model, cache, tokenizer = cache_store[cache_key]
        else:
            config = Config.from_directory(str(model_dir))
            tokenizer = Tokenizer.from_config(config)
            model = Model.from_config(config)
            cache = Cache(
                model,
                max_num_tokens=8192,
                layer_type=CacheLayer_quant,
                k_bits=8,
                v_bits=8,
            )
            model.load(progressbar=False)
            cache_store[cache_key] = (model, cache, tokenizer)

        chatml = (
            f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        context_ids = tokenizer.encode(
            chatml,
            add_bos=False,
            add_eos=False,
            encode_special_tokens=True,
        )
        sampler = ComboSampler(
            rep_p=float(rep_p),
            temperature=float(temperature),
            min_p=0.0,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        stop_ids: set[int] = set()
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        for tok_s in ("<|im_end|>", "<|endoftext|>"):
            try:
                enc_s = tokenizer.encode(tok_s, add_bos=False, add_eos=False, encode_special_tokens=True)
                if enc_s.numel() == 1:
                    stop_ids.add(int(enc_s.item()))
            except Exception:
                pass

        prefill_params: dict[str, Any] = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, cache.max_num_tokens),
        }
        if context_ids.shape[-1] > 1:
            model.prefill(input_ids=context_ids[:, :-1], params=prefill_params)

        gen_ids: list[int] = []
        for step_idx in range(int(max_new_tokens)):
            step_params: dict[str, Any] = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": context_ids.shape[-1] - 1,
                "batch_shape": (1, cache.max_num_tokens),
            }
            logits = model.forward(input_ids=context_ids[:, -1:], params=step_params)
            rand_u32 = int.from_bytes(
                hashlib.blake2b(f"{seed}:{step_idx}".encode("utf-8"), digest_size=4).digest(),
                "little",
            )
            sample = sampler.forward(
                logits,
                sequence_ids=context_ids,
                rand_u32=rand_u32,
                tokenizer=tokenizer,
            )
            tid = int(sample.item())
            if tid in stop_ids:
                break
            gen_ids.append(tid)
            context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)

        out_tensor = torch.tensor([gen_ids], dtype=torch.long) if gen_ids else torch.zeros((1, 0), dtype=torch.long)
        markdown_out = tokenizer.decode(out_tensor, decode_special_tokens=False)[0].strip() if gen_ids else ""
    else:
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace 'transformers' is not importable in this Python environment."
            ) from exc

        resolved_target = str(Path(mspec).expanduser()) if Path(mspec).expanduser().exists() else mspec
        if cache_key in cache_store:
            hf_model, hf_tokenizer = cache_store[cache_key]
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(resolved_target, trust_remote_code=True)
            load_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            }
            if load_in_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception:
                    pass
            cfg = AutoConfig.from_pretrained(resolved_target, trust_remote_code=True)
            arch_names = " ".join(getattr(cfg, "architectures", None) or ())
            model_cls = AutoModelForCausalLM
            if "Qwen3VL" in arch_names and hasattr(transformers, "Qwen3VLForConditionalGeneration"):
                model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration")
            elif "Qwen2_5_VL" in arch_names and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
                model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")
            hf_model = model_cls.from_pretrained(resolved_target, **load_kwargs)
            hf_model.eval()
            cache_store[cache_key] = (hf_model, hf_tokenizer)

        messages = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_msg},
        ]
        if hasattr(hf_tokenizer, "apply_chat_template"):
            try:
                prompt_str = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_str = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_str = (
                f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        inputs = hf_tokenizer(prompt_str, return_tensors="pt")
        target_dev = next(hf_model.parameters()).device
        inputs = {k: v.to(target_dev) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = hf_model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=max(float(temperature), 1e-4),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(rep_p),
            )
        new_tokens = out_ids[0, inputs["input_ids"].shape[-1] :]
        markdown_out = hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "ok": True,
        "provider": prov,
        "model": mspec,
        "markdown": markdown_out,
        "elapsed_s": elapsed_s,
    }


def _list_ui_toml_files(toml_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not toml_dir.is_dir():
        return out
    for p in sorted(toml_dir.glob("*.toml")):
        try:
            ps = cv.load_pairs(p, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            out.append(
                {
                    "name": ps.name,
                    "stem": p.stem,
                    "file": p.name,
                    "pairs": len(ps.pairs),
                    "n_pairs": len(ps.pairs),
                    "holds": len(ps.holds),
                    "n_holds": len(ps.holds),
                    "mode": ps.mode,
                    "frame": ps.frame,
                    "quality": ps.quality,
                    "a_side": ps.a_side,
                    "b_side": ps.b_side,
                }
            )
        except Exception:
            continue
    return out


def _list_ui_vectors(vec_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not vec_dir.is_dir():
        return out
    for p in sorted(vec_dir.glob("*.safetensors")):
        try:
            _, meta = cv.load_vector(p)
            pr_meta = meta.get("points", {}).get("post_refiner", {})
            raw_norm = float(pr_meta.get("raw_norm", 0.0))
            qr = meta.get("quality_report", {})
            mpc = float(qr.get("mean_pairwise_cosine", 0.0))
            shc = float(qr.get("split_half_cosine", 0.0))
            pairs_meta = meta.get("pairs", {})
            loras_list = [
                str(l.get("file", ""))
                for l in meta.get("recipe", {}).get("loras", ())
                if isinstance(l, dict) and l.get("file")
            ]
            out.append(
                {
                    "file": p.name,
                    "name": str(meta.get("name", p.stem)),
                    "family": str(meta.get("family", "")),
                    "pooling": str(meta.get("measured_under", {}).get("pooling", "scene_text")),
                    "n_pairs": int(pairs_meta.get("n_pairs", 0)),
                    "n_holds": int(pairs_meta.get("n_holds", 0)),
                    "norm": raw_norm,
                    "raw_norm": raw_norm,
                    "mpc": mpc,
                    "mean_cosine": mpc,
                    "mean_pairwise_cosine": mpc,
                    "split_half": shc,
                    "split_half_cosine": shc,
                    "rec_strength": _recommended_strength_text(raw_norm),
                    "loras": loras_list,
                    "j_lens": qr.get("j_lens_readout"),
                }
            )
        except Exception:
            continue
    return out


def _pair_set_to_markdown(toml_path: Path | None, ps: cv.PairSet) -> str:
    tp_txt = ""
    if toml_path is not None:
        tp_cand = toml_path.parent / f"{toml_path.stem}_test_prompt.txt"
        if tp_cand.is_file():
            tp_txt = tp_cand.read_text(encoding="utf-8").strip()
        elif toml_path.is_file():
            raw_lines = toml_path.read_text(encoding="utf-8").splitlines()
            in_tp = False
            tp_lines: list[str] = []
            for line in raw_lines:
                if line.strip() == "# Test Scene Prompt:":
                    in_tp = True
                    continue
                if in_tp:
                    if line.startswith("#"):
                        tp_lines.append(line[1:].lstrip(" "))
                    else:
                        break
            tp_txt = "\n".join(tp_lines).strip()

    chunks: list[str] = []
    if tp_txt:
        chunks.append(f"### Test Scene Prompt\n\n{tp_txt}\n")
    chunks.append(f"### Contrastive Pairs (1–{len(ps.pairs)})\n")
    for idx, item in enumerate(ps.pairs, start=1):
        chunks.append(
            f"#### Pair {idx}\n"
            f"##### Prompt A (Baseline)\n{item.a}\n\n"
            f"##### Prompt B (Steered)\n{item.b}\n"
        )
    if ps.holds:
        chunks.append("### Hold Pairs\n")
        for idx, h_item in enumerate(ps.holds, start=1):
            chunks.append(
                f"#### Hold {idx} ({h_item.group})\n"
                f"##### Prompt A (Baseline)\n{h_item.a}\n\n"
                f"##### Prompt B (Nuisance)\n{h_item.b}\n"
            )
    return "\n".join(chunks)


def _is_markdown_pairs_text(text: str) -> bool:
    return bool(cv._PAIR_HEADING_RE.search(text))


def _preview_markdown_pairs(
    state: ControlVectorUIState,
    text: str,
    *,
    frame: str = "scene",
    name: str = "",
    slot: str = "auto",
    mode: str = "t2va",
) -> dict[str, Any]:
    if frame not in cv.VALID_FRAMES:
        return {"ok": False, "error": f"frame must be one of {cv.VALID_FRAMES}, got {frame!r}"}
    inferred_slot: str | None = None
    try:
        if _is_markdown_pairs_text(text):
            parsed = cv.parse_markdown_pairs(text)
        else:
            spec_hdr = f"slot: {slot}\nmode: {mode}\n" if slot != "auto" or mode != "t2va" else ""
            spec = pair_compiler.parse_short_contrast_spec(spec_hdr + text)
            ps = spec["pair_set"]
            inferred_slot = spec["slot"]
            if not name:
                name = ps.name
            parsed = {
                "mode": ps.mode,
                "test_scene_prompt": "",
                "pairs": [{"a": p.a, "b": p.b} for p in ps.pairs],
                "holds": [],
            }
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    tok = state.get_tokenizer()
    hf_tok = tok.qwen3vl_32b.tokenizer
    decode_fn = lambda ids: hf_tok.decode(list(ids))

    pair_rows: list[dict[str, Any]] = []
    ds_counts: list[float] = []
    ffd_counts: list[float] = []
    scene_counts: list[float] = []
    first_valid_spans: tuple[str, str] | None = None

    for idx, item in enumerate(parsed["pairs"], start=1):
        a_txt = item["a"]
        b_txt = item["b"]
        try:
            a_span, b_span = cv.differing_span(a_txt, b_txt, frame=frame)
            sec_idx, sec_name = cv.differing_section_info(a_txt, b_txt, frame=frame)
            if first_valid_spans is None:
                first_valid_spans = (a_span, b_span)
            c_prefix = len(os.path.commonprefix([a_txt, b_txt]))
            pos = a_txt.find(a_span, max(0, c_prefix - len(a_span)))
            if pos < 0:
                pos = a_txt.find(a_span)
            prefix = a_txt[:pos] if pos >= 0 else ""
            suffix = a_txt[pos + len(a_span) :] if pos >= 0 else ""

            entries_a = tok.tokenize_with_weights(a_txt)["qwen3vl_32b"][0]
            entries_b = tok.tokenize_with_weights(b_txt)["qwen3vl_32b"][0]
            ids_a = [int(e[0]) if not isinstance(e[0], dict) else 0 for e in entries_a]
            ids_b = [int(e[0]) if not isinstance(e[0], dict) else 0 for e in entries_b]
            tags_a = [1] * len(ids_a)
            tags_b = [1] * len(ids_b)
            first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
            spans_a = h3_spans.spans(ids_a, tags_a, decode_fn)
            spans_b = h3_spans.spans(ids_b, tags_b, decode_fn)

            m_ds_a = int(
                h3_spans.pooling_mask(
                    spans_a, "diff_span", first_diff=min(first_diff, len(ids_a) - 1), end_diff=end_a
                ).sum().item()
            )
            m_ds_b = int(
                h3_spans.pooling_mask(
                    spans_b, "diff_span", first_diff=min(first_diff, len(ids_b) - 1), end_diff=end_b
                ).sum().item()
            )
            m_ffd_a = int(
                h3_spans.pooling_mask(
                    spans_a, "from_first_diff", first_diff=min(first_diff, len(ids_a) - 1)
                ).sum().item()
            )
            m_ffd_b = int(
                h3_spans.pooling_mask(
                    spans_b, "from_first_diff", first_diff=min(first_diff, len(ids_b) - 1)
                ).sum().item()
            )
            m_sc_a = int(h3_spans.pooling_mask(spans_a, "scene_text").sum().item())
            m_sc_b = int(h3_spans.pooling_mask(spans_b, "scene_text").sum().item())

            ds_counts.append(0.5 * (m_ds_a + m_ds_b))
            ffd_counts.append(0.5 * (m_ffd_a + m_ffd_b))
            scene_counts.append(0.5 * (m_sc_a + m_sc_b))

            pair_rows.append(
                {
                    "index": idx,
                    "valid": True,
                    "section": f"sec {sec_idx}:{sec_name}",
                    "prefix": prefix,
                    "a_span": a_span,
                    "b_span": b_span,
                    "diff_a": a_span,
                    "diff_b": b_span,
                    "suffix": suffix,
                    "tokens": {
                        "first_diff": int(first_diff),
                        "diff_span": m_ds_b,
                        "scene_text": m_sc_b,
                        "diff_span_a": m_ds_a,
                        "diff_span_b": m_ds_b,
                        "ffd_a": m_ffd_a,
                        "ffd_b": m_ffd_b,
                        "scene_a": m_sc_a,
                        "scene_b": m_sc_b,
                    },
                }
            )
        except ValueError as exc:
            pair_rows.append(
                {
                    "index": idx,
                    "valid": False,
                    "error": str(exc),
                }
            )

    quality = ""
    if first_valid_spans is not None:
        quality = f"{first_valid_spans[1][:72]} vs {first_valid_spans[0][:72]}"

    avg_tokens = None
    if ds_counts:
        avg_tokens = {
            "diff_span": round(sum(ds_counts) / len(ds_counts), 1),
            "from_first_diff": round(sum(ffd_counts) / len(ffd_counts), 1),
            "scene_text": round(sum(scene_counts) / len(scene_counts), 1),
        }

    lint_warnings: list[str] = []
    if all(r.get("valid", False) for r in pair_rows):
        try:
            lint_warnings = cv.lint_pair_set(parsed["pairs"], frame=frame)
        except Exception:
            lint_warnings = []

    return {
        "ok": True,
        "name": name,
        "suggested_name": name,
        "mode": parsed["mode"],
        "frame": frame,
        "n_pairs": len(pair_rows),
        "n_holds": len(parsed.get("holds", [])),
        "inferred_slot": inferred_slot,
        "test_scene_prompt": parsed["test_scene_prompt"],
        "test_prompt": parsed["test_scene_prompt"],
        "quality": quality,
        "avg_tokens": avg_tokens,
        "lint_warnings": lint_warnings,
        "pairs": pair_rows,
    }


def _build_pairset_from_ui_rows(
    rows: list[dict[str, Any]],
    *,
    holds: list[dict[str, Any]] | None = None,
    name: str = "",
    slot: str = "auto",
    mode: str = "t2va",
    frame: str = "scene",
    target_prompt: str = "",
    target_weight: float = 2.5,
) -> cv.PairSet:
    """Build a `cv.PairSet` from structured 10-Row Scene-Diff Table rows (`rows` + optional `holds`)."""
    active_rows = [r for r in rows if bool(r.get("enabled", True)) and str(r.get("b", "")).strip()]
    if not active_rows:
        raise ValueError("at least one enabled row with a non-empty Pole B is required")

    first_b = str(active_rows[0].get("b", "")).strip()
    is_full_scene = (
        frame == "bare"
        or "integrated_multimodal_description:" in first_b
        or "detailed_description:" in first_b
    )
    if is_full_scene:
        pair_items: list[cv.PairItem] = []
        for idx, r in enumerate(active_rows):
            a_txt = str(r.get("a", "")).strip()
            b_txt = str(r.get("b", "")).strip()
            w_val = float(r.get("weight", 1.0))
            a_sp, b_sp = cv.differing_span(a_txt, b_txt, frame=frame, label=f"row[{idx}]")
            pair_items.append(
                cv.PairItem(
                    index=idx,
                    weight=w_val,
                    a=a_txt,
                    b=b_txt,
                    a_span=a_sp,
                    b_span=b_sp,
                )
            )
        hold_items: list[cv.HoldItem] = []
        for h_idx, hr in enumerate(holds or ()):
            ha = str(hr.get("a", "")).strip()
            hb = str(hr.get("b", "")).strip()
            if not ha or not hb:
                continue
            ha_sp, hb_sp = cv.differing_span(ha, hb, frame=frame, label=f"hold[{h_idx}]")
            hold_items.append(
                cv.HoldItem(
                    index=h_idx,
                    group=str(hr.get("group", "nuisance") or "nuisance"),
                    weight=float(hr.get("weight", 1.0)),
                    a=ha,
                    b=hb,
                    a_span=ha_sp,
                    b_span=hb_sp,
                )
            )
        a_side_val = pair_items[0].a_span[:72]
        b_side_val = pair_items[0].b_span[:72]
        resolved_name = (name or pair_compiler._slugify_name(b_side_val)).strip()
        if not cv.NAME_RE.match(resolved_name):
            raise ValueError(f"vector name must match [a-z0-9_]+, got {resolved_name!r}")
        digest = hashlib.sha256(
            json.dumps(
                {
                    "name": resolved_name,
                    "mode": mode,
                    "pairs": [(p.weight, p.a, p.b) for p in pair_items],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        ps = cv.PairSet(
            path=Path(f"<workbench:{resolved_name}>"),
            sha256=digest,
            name=resolved_name,
            quality=f"{b_side_val} vs. {a_side_val}",
            a_side=a_side_val,
            b_side=b_side_val,
            mode=mode,
            frame=frame,
            exclude_shots=(),
            pairs=tuple(pair_items),
            holds=tuple(hold_items),
        )
        if str(target_prompt or "").strip():
            ps = _inject_target_prompt_into_pairset(
                ps, target_prompt, target_weight=float(target_weight), slot_hint=slot
            )
        return ps

    clauses: list[pair_compiler.ContrastClause] = []
    sound_clauses: list[pair_compiler.ContrastClause] = []
    for r in active_rows:
        r_slot = str(r.get("slot") or slot or "auto")
        clauses.append(
            pair_compiler.ContrastClause(
                a=str(r.get("a", "")).strip(),
                b=str(r.get("b", "")).strip(),
                slot=r_slot if r_slot != "auto" else None,
                weight=float(r.get("weight", 1.0)),
            )
        )
        sb = str(r.get("sound_b", "") or "").strip()
        sa = str(r.get("sound_a", "") or "").strip()
        if sb:
            sound_clauses.append(
                pair_compiler.ContrastClause(
                    a=sa,
                    b=sb,
                    slot="soundscape",
                    weight=float(r.get("weight", 1.0)),
                )
            )
    hold_clauses: list[pair_compiler.ContrastClause] = []
    for hr in (holds or ()):
        ha = str(hr.get("a", "") or "").strip()
        hb = str(hr.get("b", "") or "").strip()
        if hb:
            hold_clauses.append(
                pair_compiler.ContrastClause(
                    a=ha,
                    b=hb,
                    slot="auto",
                    group=str(hr.get("group", "nuisance") or "nuisance"),
                    weight=float(hr.get("weight", 1.0)),
                )
            )
    tp_clean = str(target_prompt or "").strip() or None
    return pair_compiler.compile_words_to_pairset(
        clauses,
        name=name or None,
        slot=slot,
        mode=mode,
        n_pairs=min(10, max(1, len(clauses))),
        soundscape_contrasts=sound_clauses or None,
        hold_contrasts=hold_clauses or None,
        target_prompt=tp_clean,
        target_weight=float(target_weight),
    )


def _repool_cached_build(
    state: ControlVectorUIState,
    req: dict[str, Any],
) -> dict[str, Any]:
    """Re-pool the most recent build in `< 1 ms` on pure CPU (or re-encode only dirty edited pairs)."""
    if not state.last_build_cache:
        raise ValueError("no active build cache in memory; build a 10-pair vector first")

    t_start = time.perf_counter()
    weights_req = req.get("weights")
    enabled_req = req.get("enabled")
    hold_indices_req = req.get("hold_indices") or []
    rows_req = req.get("rows")
    unit_mean = bool(req.get("unit_mean", False))
    hold_words = str(req.get("hold_words", "") or "").strip() or None
    hold_alpha = float(req.get("hold_alpha", 1.0))
    hold_vectors = req.get("hold_vectors") or []
    prompt_req = str(req.get("prompt", "") or "").strip() or None

    dirty_reencoded = 0
    built_rows: list[dict[str, Any]] = []

    for cache_key, entry in state.last_build_cache.items():
        family = entry["family"]
        pool_mode = entry["pooling"]
        a_mat: torch.Tensor = entry["a_mat"]
        b_mat: torch.Tensor = entry["b_mat"]
        enc_a_mat: torch.Tensor | None = entry["enc_a_mat"]
        enc_b_mat: torch.Tensor | None = entry["enc_b_mat"]
        n_total = int(a_mat.shape[0])

        # Check if any row's text was edited; if so, re-encode only that dirty row
        if isinstance(rows_req, list) and len(rows_req) == n_total:
            for idx, r_item in enumerate(rows_req):
                if not isinstance(r_item, dict):
                    continue
                new_a = str(r_item.get("a", "")).strip()
                new_b = str(r_item.get("b", "")).strip()
                old_a, old_b = entry["pairs_ab"][idx]
                if new_a and new_b and (new_a != old_a or new_b != old_b):
                    clip = state.get_clip()
                    dit_p = entry.get("dit_path") or entry.get("unet_path")
                    ref_cache_key = (
                        str(dit_p),
                        str(entry["encoder_path"]),
                        str(entry["refiner_device"]),
                        json.dumps(entry["loras"], sort_keys=True),
                    )
                    if ref_cache_key in state._refiner_cache:
                        _, _, refiner_run = state._refiner_cache[ref_cache_key]
                    else:
                        refiner_run = cv_nodes.build_refiner_runner(
                            dit_p, device=entry["refiner_device"], loras=entry["loras"]
                        )
                    enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(clip, new_a)
                    enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(clip, new_b)
                    ref_a = refiner_run(enc_a)
                    ref_b = refiner_run(enc_b)
                    first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
                    spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
                    spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
                    mask_a = h3_spans.pooling_mask(
                        spans_a,
                        pool_mode,
                        first_diff=min(first_diff, len(ids_a) - 1),
                        end_diff=end_a,
                        other_ids=ids_b,
                    )
                    mask_b = h3_spans.pooling_mask(
                        spans_b,
                        pool_mode,
                        first_diff=min(first_diff, len(ids_b) - 1),
                        end_diff=end_b,
                        other_ids=ids_a,
                    )
                    a_mat[idx] = (
                        ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32).cpu()
                    )
                    b_mat[idx] = (
                        ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32).cpu()
                    )
                    if enc_a_mat is not None and enc_b_mat is not None:
                        ma_e = mask_a.clone()
                        mb_e = mask_b.clone()
                        if ma_e.numel() > 0:
                            ma_e[0] = False
                        if mb_e.numel() > 0:
                            mb_e[0] = False
                        if not bool(ma_e.any()):
                            ma_e = mask_a
                        if not bool(mb_e.any()):
                            mb_e = mask_b
                        enc_a_mat[idx] = (
                            enc_a.to(dtype=torch.float64)[ma_e].mean(dim=0).to(dtype=torch.float32).cpu()
                        )
                        enc_b_mat[idx] = (
                            enc_b.to(dtype=torch.float64)[mb_e].mean(dim=0).to(dtype=torch.float32).cpu()
                        )
                    a_sp, b_sp = cv.differing_span(new_a, new_b, frame=entry["pair_set"].frame)
                    entry["pair_spans"][idx] = {
                        "index": idx,
                        "weight": float(r_item.get("weight", entry["weights"][idx])),
                        "first_diff": int(first_diff),
                        "end_a": int(end_a),
                        "end_b": int(end_b),
                        "a_span": a_sp,
                        "b_span": b_sp,
                        "len_a": int(len(ids_a)),
                        "len_b": int(len(ids_b)),
                        "pooled_rows_a": int(mask_a.sum().item()),
                        "pooled_rows_b": int(mask_b.sum().item()),
                        "pooled_tokens": int(mask_b.sum().item()),
                    }
                    entry["pairs_ab"][idx] = (new_a, new_b)
                    dirty_reencoded += 1

        hold_idx_set = {
            int(i) for i in (hold_indices_req or ()) if 0 <= int(i) < n_total
        }
        active_indices_req = req.get("active_indices")
        if isinstance(active_indices_req, list):
            active_set_req = {int(i) for i in active_indices_req if 0 <= int(i) < n_total}
            active_indices = [
                i for i in range(n_total) if i in active_set_req and i not in hold_idx_set
            ]
        else:
            active_indices = [
                i
                for i in range(n_total)
                if (not isinstance(enabled_req, list) or i >= len(enabled_req) or bool(enabled_req[i]))
                and i not in hold_idx_set
            ]
        if not active_indices:
            raise ValueError("at least one pair must remain enabled and not moved to hold")

        eff_weights = [
            float(weights_req[i])
            if (isinstance(weights_req, list) and i < len(weights_req))
            else float(entry["weights"][i])
            for i in range(n_total)
        ]
        entry["weights"] = eff_weights
        for i in range(n_total):
            entry["pair_spans"][i]["weight"] = eff_weights[i]

        a_sub = a_mat[active_indices]
        b_sub = b_mat[active_indices]
        w_sub = torch.tensor([eff_weights[i] for i in active_indices], dtype=torch.float32)

        raw_vec = cv.mean_difference(a_sub, b_sub, w_sub, unit_mean=unit_mean)
        tensors_to_save: dict[str, torch.Tensor] = {}

        hold_a = list(entry["hold_a_list"]) + [a_mat[hi] for hi in sorted(hold_idx_set)]
        hold_b = list(entry["hold_b_list"]) + [b_mat[hi] for hi in sorted(hold_idx_set)]
        h_groups = list(entry["hold_groups"]) + [f"pair_{hi}" for hi in sorted(hold_idx_set)]
        h_weights = list(entry["hold_weights"]) + [1.0 for _ in sorted(hold_idx_set)]

        if hold_a:
            basis = cv.hold_basis(
                torch.stack(hold_a, dim=0),
                torch.stack(hold_b, dim=0),
                h_groups,
                torch.tensor(h_weights, dtype=torch.float32),
            )
            final_vec = cv.project_out(raw_vec, basis)
            tensors_to_save["holds.post_refiner"] = basis
        else:
            final_vec = raw_vec

        applied_holds: list[str] = []
        norm_bef = 0.0
        norm_aft = 0.0
        if hold_vectors or hold_words:
            final_vec, applied_holds, norm_bef, norm_aft = _apply_post_refiner_holds(
                final_vec,
                family=family,
                hold_specs=hold_vectors,
                hold_words=hold_words,
                hold_alpha=hold_alpha,
                prompt=prompt_req,
            )

        tensors_to_save["vector.post_refiner"] = final_vec
        gen = torch.Generator(device="cpu").manual_seed(int(entry["seed"]))
        if len(active_indices) >= 2:
            q_report = cv.quality_report(a_sub, b_sub, w_sub, generator=gen)
        else:
            q_report = {
                "mean_pairwise_cosine": 1.0,
                "split_half_cosine": 1.0,
                "per_pair_cosine": [1.0],
            }

        # Compute per-pair cosine for all n_total rows against final_vec so the UI table shows every row
        all_per_pair_cos: list[float] = []
        for i in range(n_total):
            d_i = b_mat[i] - a_mat[i]
            all_per_pair_cos.append(round(_cosine(d_i, final_vec), 4))

        raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
        points_meta: dict[str, Any] = {
            "post_refiner": {
                "unit": "mean_pair_difference",
                "raw_norm": raw_norm,
                "median_row_norm": entry["median_ref_row_norm"],
            }
        }
        if enc_a_mat is not None and enc_b_mat is not None:
            enc_raw_vec = cv.mean_difference(
                enc_a_mat[active_indices],
                enc_b_mat[active_indices],
                w_sub,
                unit_mean=unit_mean,
            )
            enc_hold_a = list(entry.get("enc_hold_a_list") or []) + [
                enc_a_mat[hi] for hi in sorted(hold_idx_set)
            ]
            enc_hold_b = list(entry.get("enc_hold_b_list") or []) + [
                enc_b_mat[hi] for hi in sorted(hold_idx_set)
            ]
            if enc_hold_a:
                enc_basis = cv.hold_basis(
                    torch.stack(enc_hold_a, dim=0),
                    torch.stack(enc_hold_b, dim=0),
                    h_groups,
                    torch.tensor(h_weights, dtype=torch.float32),
                )
                enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
            enc_raw_norm = float(torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item())
            if enc_raw_norm > 1e-12:
                enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(dtype=torch.float32)
                tensors_to_save["vector.encoder"] = enc_vec
                points_meta["encoder"] = {
                    "unit": "median_row_norm",
                    "raw_norm": enc_raw_norm,
                    "median_row_norm": entry["median_enc_row_norm_ex0"],
                    "row0_excluded": True,
                }

        jl_readout: dict[str, Any] | None = None
        if j_lens.bundle_available():
            try:
                jl_readout = j_lens.decode_vector(final_vec, space="post_refiner", family=family)
                if prompt_req:
                    jl_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                        final_vec,
                        prompt_req,
                        readout=jl_readout,
                        family=family,
                    )
                q_report["j_lens_readout"] = jl_readout
            except Exception:
                jl_readout = None

        pair_set = entry["pair_set"]
        metadata: dict[str, Any] = {
            "format": cv.FORMAT_VERSION,
            "name": entry["stem"],
            "quality": pair_set.quality,
            "family": family,
            "pairs": {
                "path": entry["rel_pair_path"],
                "sha256": pair_set.sha256,
                "n_pairs": len(active_indices),
                "n_holds": len(hold_a) + len(applied_holds),
                "a_side": pair_set.a_side,
                "b_side": pair_set.b_side,
                "pair_spans": [entry["pair_spans"][i] for i in active_indices],
            },
            "recipe": entry["recipe"],
            "measured_under": {
                "mode": entry["measured_mode"],
                "frame": pair_set.frame,
                "prefix": entry["prefix_meta"],
                "pooling": pool_mode,
            },
            "points": points_meta,
            "quality_report": q_report,
            "provenance": {
                "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
                "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "made_by": "bench/control_vectors.py::repool/1",
                **(
                    {
                        "applied_holds": {
                            "holds": applied_holds,
                            "alpha": hold_alpha,
                            "norm_before": round(norm_bef, 4),
                            "norm_after_raw": round(norm_aft, 4),
                        }
                    }
                    if applied_holds
                    else {}
                ),
            },
        }
        saved = cv.save_vector(entry["out_path"], tensors_to_save, metadata, overwrite=True)
        rec_band = [
            round(50.0 / max(raw_norm, 1e-6), 2),
            round(85.0 / max(raw_norm, 1e-6), 2),
        ]
        built_rows.append(
            {
                "file": saved.name,
                "family": family,
                "pooling": pool_mode,
                "raw_norm": raw_norm,
                "vector_norm": raw_norm,
                "mpc": float(q_report["mean_pairwise_cosine"]),
                "split_half": float(q_report["split_half_cosine"]),
                "split_half_cosine": float(q_report["split_half_cosine"]),
                "per_pair_cosine": all_per_pair_cos,
                "active_indices": active_indices,
                "hold_indices": sorted(hold_idx_set),
                "pair_spans": entry["pair_spans"],
                "rec_strength": _recommended_strength_text(raw_norm),
                "recommended_strength_band": rec_band,
                "dirty_reencoded": dirty_reencoded,
                "j_lens": jl_readout,
            }
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return {
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "dirty_reencoded": dirty_reencoded,
        "built": built_rows,
        "vectors": _list_ui_vectors(state.vec_dir),
    }


def _compose_and_save_sculpted_vector(
    steps: list[dict[str, Any]],
    *,
    name: str,
    family: str = "fl2va",
    hold_words: str | None = None,
    hold_alpha: float = 1.0,
    prompt: str | None = None,
    out_dir: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Compose a multi-vector recipe (`add`, `add_norm_matched`, `hold`) + optional `hold_words` (`alpha`)
    and write `.safetensors` + `.md` sidecars for `family`.
    """
    raw_stem = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name or "").strip().lower()).strip("_")
    if not raw_stem:
        raw_stem = "sculpted_vector"
    if not cv.NAME_RE.match(raw_stem):
        raise ValueError(f"sculpted vector stem must match [a-z0-9_]+, got {raw_stem!r}")

    target_dir = out_dir or Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_steps: list[dict[str, Any]] = []
    enc_add_terms: list[torch.Tensor] = []
    enc_hold_terms: list[tuple[torch.Tensor, float]] = []
    base_meta: dict[str, Any] | None = None
    step_labels: list[str] = []

    for item in steps:
        raw_src = item["vector"]
        mode = str(item.get("mode", "add")).strip()
        strength = float(item.get("strength", 1.0))
        vpath = _resolve_vector_for_family(raw_src, family=family, extra_dir=target_dir)
        tensors, meta = cv.load_vector(vpath)
        if base_meta is None and mode in ("add", "add_norm_matched"):
            base_meta = meta
        resolved_steps.append({"vector": vpath, "mode": mode, "strength": strength})
        step_labels.append(f"{mode}:{meta.get('name', vpath.stem)}@{strength:g}")
        if "vector.encoder" in tensors:
            ve = tensors["vector.encoder"].detach().to(device="cpu", dtype=torch.float32).flatten()
            if mode in ("add", "add_norm_matched"):
                enc_add_terms.append(strength * ve)
            elif mode == "hold":
                enc_hold_terms.append((ve, strength))

    if hasattr(j_lens, "save_composed_recipe_vector") and j_lens.bundle_available():
        loras_meta = (
            base_meta.get("recipe", {}).get("loras")
            if base_meta is not None
            else j_lens._DEFAULT_RECIPES.get(family, {}).get("loras")
        )
        fname = cv.default_vector_filename(raw_stem, family, pooling="diff_span", loras=loras_meta)
        out_path = target_dir / fname
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing control vector: {out_path}")
        saved_res = j_lens.save_composed_recipe_vector(
            out_path,
            resolved_steps,
            name=raw_stem,
            family=family,
            hold_words=hold_words,
            alpha=hold_alpha,
            prompt=prompt,
        )
        saved_path = saved_res["path"] if isinstance(saved_res, dict) and "path" in saved_res else Path(saved_res)
        _, saved_meta = cv.load_vector(saved_path)
        r_norm = float(saved_meta["points"]["post_refiner"]["raw_norm"])
        jl_rep = saved_meta.get("quality_report", {}).get("j_lens_readout")
        return {
            "file": saved_path.name,
            "path": saved_path,
            "family": family,
            "raw_norm": r_norm,
            "net_norm": r_norm,
            "rec_strength": _recommended_strength_text(r_norm),
            "j_lens": jl_rep,
        }

    # Compose directly via j_lens.apply_soft_holds + decode_vector
    add_vecs: list[torch.Tensor] = []
    hold_vecs: list[tuple[torch.Tensor, float]] = []
    for st in resolved_steps:
        v_pr, _, _ = j_lens._load_vector_from_source(st["vector"], space="post_refiner")
        m_st = st["mode"]
        s_st = float(st["strength"])
        v64 = v_pr.to(dtype=torch.float64)
        vn = float(torch.linalg.vector_norm(v64).item())
        if m_st == "add":
            add_vecs.append((s_st * v64).to(dtype=torch.float32))
        elif m_st == "add_norm_matched":
            scale_f = (j_lens.NORM_MATCHED_TARGET / vn) if vn > 1e-12 else 0.0
            add_vecs.append((s_st * scale_f * v64).to(dtype=torch.float32))
        elif m_st == "hold":
            hold_vecs.append((v_pr, s_st))
        else:
            raise ValueError(f"unsupported sculpt mode {m_st!r}")
    if not add_vecs:
        raise ValueError("sculpt requires at least one 'add' or 'add_norm_matched' vector")

    cur_vec = torch.stack([v.to(dtype=torch.float64) for v in add_vecs], dim=0).sum(dim=0).to(dtype=torch.float32)
    pre_hold_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
    for hv, h_alpha in hold_vecs:
        cur_vec = j_lens.apply_soft_holds(cur_vec, [hv], alpha=float(h_alpha))

    hw_labels: list[str] = []
    if hold_words and str(hold_words).strip() and float(hold_alpha) > 0.0:
        cur_vec, hw_labels, _, _ = _apply_post_refiner_holds(
            cur_vec,
            family=family,
            hold_words=hold_words,
            hold_alpha=float(hold_alpha),
            prompt=prompt,
        )
    else:
        post_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
        if post_norm <= 1e-6 * max(1.0, pre_hold_norm):
            raise ValueError("holds completely cancelled the sculpted vector")

    raw_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
    tensors_to_save: dict[str, torch.Tensor] = {"vector.post_refiner": cur_vec}
    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": (
                float(base_meta["points"]["post_refiner"]["median_row_norm"])
                if base_meta and "post_refiner" in base_meta.get("points", {})
                else j_lens.DEFAULT_POST_REFINER_MEDIAN_ROW_NORM
            ),
        }
    }
    if enc_add_terms:
        enc_cur = torch.stack([v.to(dtype=torch.float64) for v in enc_add_terms], dim=0).sum(dim=0).to(dtype=torch.float32)
        for h_ve, h_a in enc_hold_terms:
            enc_cur = j_lens.apply_soft_holds(enc_cur, [h_ve], alpha=float(h_a))
        en_norm = float(torch.linalg.vector_norm(enc_cur.to(dtype=torch.float64)).item())
        if en_norm > 1e-12:
            tensors_to_save["vector.encoder"] = (enc_cur.to(dtype=torch.float64) / en_norm).to(dtype=torch.float32)
            points_meta["encoder"] = {
                "unit": "median_row_norm",
                "raw_norm": 1.0,
                "median_row_norm": j_lens.DEFAULT_ENCODER_MEDIAN_ROW_NORM,
                "row0_excluded": True,
            }

    jl_readout: dict[str, Any] | None = None
    if j_lens.bundle_available():
        jl_readout = j_lens.decode_vector(cur_vec, space="post_refiner", family=family)
        if prompt:
            jl_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                cur_vec, prompt, readout=jl_readout, family=family
            )

    recipe = (
        base_meta["recipe"]
        if base_meta and "recipe" in base_meta
        else j_lens._DEFAULT_RECIPES.get(family, j_lens._DEFAULT_RECIPES["fl2va"])
    )
    loras_meta = recipe.get("loras")
    spec_str = "|".join(step_labels + hw_labels)
    metadata: dict[str, Any] = {
        "format": cv.FORMAT_VERSION,
        "name": raw_stem,
        "quality": f"Sculpted recipe ({', '.join(step_labels + hw_labels)})",
        "family": family,
        "pairs": {
            "path": "bench/control_vectors.py::sculpt",
            "sha256": hashlib.sha256(spec_str.encode("utf-8")).hexdigest(),
            "n_pairs": len(add_vecs),
            "n_holds": len(hold_vecs) + len(hw_labels),
            "a_side": "baseline",
            "b_side": ", ".join(step_labels),
        },
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if family == "ref2va" else "t2va",
            "frame": "scene",
            "prefix": "none",
            "pooling": "diff_span",
        },
        "points": points_meta,
        "quality_report": {
            "mean_pairwise_cosine": 1.0,
            "split_half_cosine": 1.0,
            "per_pair_cosine": [1.0 for _ in add_vecs],
            **({"j_lens_readout": jl_readout} if jl_readout else {}),
        },
        "provenance": {
            "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
            "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made_by": "bench/control_vectors.py::sculpt/1",
            "sculpt_steps": step_labels + hw_labels,
        },
    }
    fname = cv.default_vector_filename(
        raw_stem,
        family,
        pooling="diff_span",
        apply_holds=bool(hold_vecs or hw_labels),
        n_holds=len(hold_vecs) + len(hw_labels),
        loras=loras_meta,
    )
    saved = cv.save_vector(target_dir / fname, tensors_to_save, metadata, overwrite=overwrite)
    return {
        "file": saved.name,
        "path": saved,
        "family": family,
        "raw_norm": raw_norm,
        "rec_strength": _recommended_strength_text(raw_norm),
        "j_lens": jl_readout,
    }


def _meter_prompt_tokens(
    state: ControlVectorUIState,
    *,
    vector_arg: str,
    prompt_arg: str,
    pooling: str = "scene_text",
) -> dict[str, Any]:
    """Run exact `5376`-d contextual `Qwen3-VL` + `TokenRefiner` per-token receptor metering (`cv.meter_score`)."""
    v_cand = state.vec_dir / vector_arg
    vpath = v_cand if v_cand.is_file() else cv_nodes.resolve_vector_file_path(vector_arg)
    tensors, metadata = cv.load_vector(vpath)
    vec = (
        tensors["vector.post_refiner"]
        if "vector.post_refiner" in tensors
        else next(iter(tensors.values()))
    )
    try:
        label, text = _resolve_prompt_text(prompt_arg)
    except Exception:
        label, text = "prompt", str(prompt_arg).strip()
    if not text:
        raise ValueError("prompt text must be non-empty")

    enc_name = metadata.get("recipe", {}).get("encoder", {}).get("file") or h3_config.MODELS["clip"]
    if enc_name == "unspecified":
        enc_name = h3_config.MODELS["clip"]
    clip = state.get_clip(enc_name, "default")

    dit_name = metadata.get("recipe", {}).get("checkpoint", {}).get("file") or (
        h3_config.MODELS["unet_ref2va"]
        if metadata.get("family") == "ref2va"
        else h3_config.MODELS["unet_fl2va"]
    )
    dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    enc_path = Path(folder_paths.get_full_path_or_raise("text_encoders", enc_name))
    meter_loras = list(metadata.get("recipe", {}).get("loras", ()))
    ref_cache_key = (
        str(dit_path),
        str(enc_path),
        "cpu",
        json.dumps(meter_loras, sort_keys=True),
    )
    if ref_cache_key in state._refiner_cache:
        _, _, refiner_run = state._refiner_cache[ref_cache_key]
    else:
        recipe = cv_nodes.extract_model_recipe(
            None,
            None,
            dit_path_override=dit_path,
            encoder_path_override=enc_path,
            loras_override=meter_loras,
        )
        refiner_run = cv_nodes.build_refiner_runner(dit_path, device="cpu", loras=meter_loras)
        state._refiner_cache[ref_cache_key] = (metadata.get("family", "fl2va"), recipe, refiner_run)

    t0 = time.perf_counter()
    enc_rows, tags, tids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, text)
    ref_rows = refiner_run(enc_rows)
    sp = h3_spans.spans(tids, tags.tolist(), dec_fn)
    eff_pooling = pooling if pooling in ("scene_text", "non_vision", "all_rows") else "scene_text"
    mask = h3_spans.pooling_mask(sp, eff_pooling)
    m = cv.meter_score(ref_rows, vec, mask=mask)

    v64 = vec.detach().to(dtype=torch.float64).reshape(-1)
    v_norm = float(torch.linalg.vector_norm(v64).item())
    v_unit = v64 / max(v_norm, 1e-12)
    rows64 = ref_rows.detach().to(dtype=torch.float64)

    tokens_out: list[dict[str, Any]] = []
    for t_idx in range(len(tids)):
        piece = dec_fn([int(tids[t_idx])])
        r_t = rows64[t_idx]
        r_norm = float(torch.linalg.vector_norm(r_t).item())
        proj_t = float(torch.dot(r_t, v_unit).item())
        cos_t = proj_t / r_norm if r_norm > 1e-12 else 0.0
        in_mask = bool(mask[t_idx].item())
        tokens_out.append(
            {
                "index": t_idx,
                "text": piece,
                "word": piece,
                "token": piece,
                "in_scene_mask": in_mask,
                "cosine": round(cos_t, 4),
                "projection": round(proj_t, 4),
            }
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "ok": True,
        "vector": vpath.name,
        "prompt_label": label,
        "pooling": eff_pooling,
        "elapsed_ms": elapsed_ms,
        "meter": m,
        "tokens": tokens_out,
    }


def _discover_ui_clips() -> list[dict[str, Any]]:
    """Return available comparison clips under $H3_OUTPUT_DIR/Video/control_vectors/ (or empty list)."""
    out: list[dict[str, Any]] = []
    out_root = os.environ.get("H3_OUTPUT_DIR", "")
    if not out_root:
        return out
    cv_vid_dir = Path(out_root) / "Video" / "control_vectors"
    if not cv_vid_dir.is_dir():
        return out
    for mp4 in sorted(cv_vid_dir.rglob("*.mp4"))[:60]:
        try:
            rel = mp4.relative_to(Path(out_root)).as_posix()
            out.append({"name": mp4.name, "path": rel, "url": f"/media?path={rel}"})
        except Exception:
            continue
    return out


def handle_ui_request(
    state: ControlVectorUIState,
    method: str,
    path: str,
    body: bytes = b"",
) -> tuple[int, str, str]:
    clean_path = path.split("?", 1)[0]
    if method == "GET" and clean_path in ("/", "/index.html"):
        html_path = BENCH_DIR / "control_vector_ui.html"
        return 200, "text/html; charset=utf-8", html_path.read_text(encoding="utf-8")

    if method == "GET" and clean_path == "/api/init":
        payload = {
            "ok": True,
            "toml_dir": _display_path(state.toml_dir),
            "vec_dir": _display_path(state.vec_dir),
            "system_prompt": _extract_system_prompt_text(),
            "toml_files": _list_ui_toml_files(state.toml_dir),
            "vectors": _list_ui_vectors(state.vec_dir),
            "clips": _discover_ui_clips(),
            "j_lens_available": j_lens.bundle_available(),
            "canonical_frames": pair_compiler.load_canonical_frames(),
            "llm_providers": list(SUPPORTED_LLM_PROVIDERS),
            "default_llm_provider": state.default_llm_provider,
            "default_llm_model": state.default_llm_model,
        }
        return 200, "application/json", json.dumps(payload)

    req: dict[str, Any] = {}
    if body:
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": f"invalid JSON: {exc}"})

    if method == "POST" and clean_path == "/api/generate-pairs-llm":
        try:
            prov_req = str(req.get("provider") or state.default_llm_provider or "transformers").strip()
            model_req = str(req.get("model") or state.default_llm_model or "").strip()
            concept_req = str(req.get("concept") or req.get("axis") or "").strip()
            load_4bit = bool(req.get("load_in_4bit", True))
            max_tok = int(req.get("max_new_tokens", 3200))
            temp_req = float(req.get("temperature", 0.7))
            res_llm = _generate_pairs_with_llm(
                provider=prov_req,
                model_spec=model_req,
                concept_prompt=concept_req,
                load_in_4bit=load_4bit,
                max_new_tokens=max_tok,
                temperature=temp_req,
                llm_cache=state._llm_cache,
            )
            return 200, "application/json", json.dumps(res_llm)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/expand-scenes":
        try:
            pole_a = str(req.get("pole_a", "") or "").strip()
            pole_b = str(req.get("pole_b", "") or "").strip()
            sound_a = str(req.get("sound_a", "") or "").strip()
            sound_b = str(req.get("sound_b", "") or "").strip()
            hold_a = str(req.get("hold_a", "") or "").strip()
            hold_b = str(req.get("hold_b", "") or "").strip()
            slot = str(req.get("slot", "auto") or "auto")
            mode = str(req.get("mode", "t2va") or "t2va")
            name = str(req.get("name", "") or "").strip()
            target_prompt = str(req.get("target_prompt", "") or "").strip()
            target_weight = float(req.get("target_weight", 2.5))
            spec_text = str(req.get("text") or req.get("words") or "").strip()

            if not spec_text:
                if not pole_b:
                    return 400, "application/json", json.dumps({"ok": False, "error": "Pole B (target concept) is required"})
                lines = [f"{pole_a} ||| {pole_b}" if pole_a else pole_b]
                if sound_b:
                    lines.append(f"soundscape: {sound_a} ||| {sound_b}" if sound_a else f"soundscape: {sound_b}")
                if hold_b:
                    lines.append(f"hold: {hold_a} ||| {hold_b}" if hold_a else f"hold: {hold_b}")
                spec_text = "\n".join(lines)

            ps = _compile_words_or_spec_to_pairset(
                spec_text,
                name=name or None,
                slot=slot,
                mode=mode,
                target_prompt=target_prompt,
                target_weight=target_weight,
            )
            frames_bank = pair_compiler.load_canonical_frames()["frames"].get(ps.mode, [])
            rows_payload: list[dict[str, Any]] = []
            for idx, p in enumerate(ps.pairs):
                is_tp = bool(target_prompt) and idx == 0
                f_idx = (idx - 1) if bool(target_prompt) else idx
                f_meta = frames_bank[f_idx % len(frames_bank)] if (not is_tp and frames_bank) else {}
                f_id = "target_scene_prompt" if is_tp else f_meta.get("id", f"scene_{idx + 1:02d}")
                f_domain = "target_prompt (Pair 0)" if is_tp else f_meta.get("domain", "canonical")
                rows_payload.append(
                    {
                        "index": idx,
                        "frame_id": f_id,
                        "scene_id": f_id,
                        "scene_title": f"{f_id} ({f_domain})",
                        "domain": f_domain,
                        "weight": float(p.weight),
                        "enabled": True,
                        "a": p.a,
                        "b": p.b,
                        "a_span": p.a_span,
                        "b_span": p.b_span,
                    }
                )
            holds_payload = [
                {
                    "index": h.index,
                    "group": h.group,
                    "weight": float(h.weight),
                    "a": h.a,
                    "b": h.b,
                    "a_span": h.a_span,
                    "b_span": h.b_span,
                }
                for h in ps.holds
            ]
            resolved_slot = pair_compiler.resolve_slot_name(slot, sample_text=ps.b_side)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": ps.name,
                        "slot": resolved_slot,
                        "slot_auto_detected": slot == "auto",
                        "mode": ps.mode,
                        "quality": ps.quality,
                        "a_side": ps.a_side,
                        "b_side": ps.b_side,
                        "pole_a": ps.a_side,
                        "pole_b": ps.b_side,
                        "test_scene_prompt": target_prompt or (ps.pairs[0].a if ps.pairs else ""),
                        "test_prompt": target_prompt or (ps.pairs[0].a if ps.pairs else ""),
                        "rows": rows_payload,
                        "pairs": rows_payload,
                        "holds": holds_payload,
                        "markdown": _pair_set_to_markdown(None, ps),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/preview":
        try:
            text_req = str(req.get("text", "") or "")
            rows_req = req.get("rows")
            frame_req = str(req.get("frame", "scene") or "scene")
            name_req = str(req.get("name", "") or "").strip()
            slot_req = str(req.get("slot", "auto") or "auto")
            mode_req = str(req.get("mode", "t2va") or "t2va")
            if isinstance(rows_req, list) and rows_req and not text_req.strip():
                ps_rows = _build_pairset_from_ui_rows(
                    rows_req,
                    holds=req.get("holds"),
                    name=name_req,
                    slot=slot_req,
                    mode=mode_req,
                    frame=frame_req,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                text_req = _pair_set_to_markdown(None, ps_rows)
                if not name_req:
                    name_req = ps_rows.name
            res = _preview_markdown_pairs(
                state,
                text_req,
                frame=frame_req,
                name=name_req,
                slot=slot_req,
                mode=mode_req,
            )
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/load-toml":
        fname = str(req.get("file", "")).strip()
        if not fname:
            return 400, "application/json", json.dumps({"ok": False, "error": "missing 'file'"})
        try:
            cand = state.toml_dir / (fname if fname.endswith(".toml") else f"{fname}.toml")
            tpath = cand if cand.is_file() else cv_nodes.resolve_pair_file_path(fname)
            ps = cv.load_pairs(tpath, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            md_str = _pair_set_to_markdown(tpath, ps)
            frames_bank = pair_compiler.load_canonical_frames()["frames"].get(ps.mode, [])
            rows_payload = [
                {
                    "index": idx,
                    "frame_id": frames_bank[idx % len(frames_bank)]["id"] if frames_bank else f"pair_{idx + 1:02d}",
                    "scene_id": frames_bank[idx % len(frames_bank)]["id"] if frames_bank else f"pair_{idx + 1:02d}",
                    "scene_title": (
                        f"{frames_bank[idx % len(frames_bank)]['id']} ({frames_bank[idx % len(frames_bank)]['domain']})"
                        if frames_bank
                        else f"pair_{idx + 1:02d}"
                    ),
                    "domain": frames_bank[idx % len(frames_bank)]["domain"] if frames_bank else ps.mode,
                    "weight": float(p.weight),
                    "enabled": True,
                    "a": p.a,
                    "b": p.b,
                    "a_span": p.a_span,
                    "b_span": p.b_span,
                }
                for idx, p in enumerate(ps.pairs)
            ]
            holds_payload = [
                {
                    "index": h.index,
                    "group": h.group,
                    "weight": float(h.weight),
                    "a": h.a,
                    "b": h.b,
                    "a_span": h.a_span,
                    "b_span": h.b_span,
                }
                for h in ps.holds
            ]
            tp_cand = tpath.parent / f"{tpath.stem}_test_prompt.txt"
            tp_txt = tp_cand.read_text(encoding="utf-8").strip() if tp_cand.is_file() else (ps.pairs[0].a if ps.pairs else "")
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "file": tpath.name,
                        "name": ps.name,
                        "frame": ps.frame,
                        "mode": ps.mode,
                        "quality": ps.quality,
                        "a_side": ps.a_side,
                        "b_side": ps.b_side,
                        "n_pairs": len(ps.pairs),
                        "n_holds": len(ps.holds),
                        "test_scene_prompt": tp_txt,
                        "test_prompt": tp_txt,
                        "rows": rows_payload,
                        "pairs": rows_payload,
                        "holds": holds_payload,
                        "markdown": md_str,
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/save-toml":
        text = str(req.get("text", ""))
        name = str(req.get("name", "")).strip()
        frame = str(req.get("frame", "scene") or "scene")
        slot = str(req.get("slot", "auto") or "auto")
        mode = str(req.get("mode", "t2va") or "t2va")
        overwrite = bool(req.get("overwrite", True))
        rows_in = req.get("rows")
        try:
            if isinstance(rows_in, list) and rows_in:
                ps = _build_pairset_from_ui_rows(
                    rows_in,
                    holds=req.get("holds"),
                    name=name,
                    slot=slot,
                    mode=mode,
                    frame=frame,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                if not name:
                    name = ps.name
                md_doc = _pair_set_to_markdown(None, ps)
                tp_out = None
                saved_toml = cv.write_markdown_to_toml(
                    md_doc,
                    state.toml_dir,
                    name=name,
                    mode=ps.mode,
                    frame=frame,
                    quality=ps.quality,
                    a_side=ps.a_side,
                    b_side=ps.b_side,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            elif _is_markdown_pairs_text(text):
                if not name:
                    return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
                parsed = cv.parse_markdown_pairs(text)
                tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
                saved_toml = cv.write_markdown_to_toml(
                    text,
                    state.toml_dir,
                    name=name,
                    frame=frame,
                    test_prompt_out=tp_out,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            else:
                ps = _compile_words_or_spec_to_pairset(
                    text,
                    name=name or None,
                    slot=slot,
                    mode=mode,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                if not name:
                    name = ps.name
                md_doc = _pair_set_to_markdown(None, ps)
                tp_out = None
                saved_toml = cv.write_markdown_to_toml(
                    md_doc,
                    state.toml_dir,
                    name=name,
                    mode=ps.mode,
                    frame=frame,
                    quality=ps.quality,
                    a_side=ps.a_side,
                    b_side=ps.b_side,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            n_saved_pairs = len(ps.pairs) if "ps" in locals() else len(parsed.get("pairs", []))
            disp_toml = _display_path(saved_toml)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": name,
                        "toml": disp_toml,
                        "toml_path": disp_toml,
                        "saved_toml": disp_toml,
                        "pairs": n_saved_pairs,
                        "toml_files": _list_ui_toml_files(state.toml_dir),
                        "test_prompt_file": _display_path(tp_out) if tp_out and tp_out.is_file() else None,
                    }
                ),
            )
        except (ValueError, FileExistsError, SystemExit) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/build":
        text = str(req.get("text", "") or req.get("words", "") or "")
        name = str(req.get("name", "") or req.get("output_name", "") or "").strip()
        frame = str(req.get("frame", "scene") or "scene")
        slot = str(req.get("slot", "auto") or "auto")
        mode = str(req.get("mode", "t2va") or "t2va")
        families = str(req.get("families", "both") or "both")
        pooling = str(req.get("pooling", "diff_span") or "diff_span")
        no_lora = bool(req.get("no_lora", False))
        overwrite = bool(req.get("overwrite", True))
        include_encoder = bool(req.get("include_encoder", True))
        unit_mean = bool(req.get("unit_mean", False))
        save_toml_req = bool(req.get("save_toml", False))
        hold_words = str(req.get("hold_words", "") or "").strip()
        hold_alpha = float(req.get("hold_alpha", 1.0))
        hold_vectors = req.get("hold_vectors") or []
        target_prompt = str(req.get("target_prompt", "") or "").strip()
        target_weight = float(req.get("target_weight", 2.5))
        rows_in = req.get("rows")
        try:
            import contextlib
            import io

            saved_toml: Path | None = None
            compiled_ps: cv.PairSet | None = None
            if isinstance(rows_in, list) and rows_in:
                compiled_ps = _build_pairset_from_ui_rows(
                    rows_in,
                    holds=req.get("holds"),
                    name=name,
                    slot=slot,
                    mode=mode,
                    frame=frame,
                    target_prompt=target_prompt,
                    target_weight=target_weight,
                )
                if not name:
                    name = compiled_ps.name
                if save_toml_req:
                    md_doc = _pair_set_to_markdown(None, compiled_ps)
                    saved_toml = cv.write_markdown_to_toml(
                        md_doc,
                        state.toml_dir,
                        name=name,
                        mode=compiled_ps.mode,
                        frame=frame,
                        quality=compiled_ps.quality,
                        a_side=compiled_ps.a_side,
                        b_side=compiled_ps.b_side,
                        overwrite=overwrite,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )
            elif _is_markdown_pairs_text(text):
                if not name:
                    return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
                parsed = cv.parse_markdown_pairs(text)
                tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
                cand_toml = state.toml_dir / f"{name}.toml"
                reused_existing = False
                if cand_toml.is_file():
                    try:
                        existing_ps = cv.load_pairs(cand_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                        existing_ab = [(p.a, p.b) for p in existing_ps.pairs]
                        new_ab = [(p["a"], p["b"]) for p in parsed["pairs"]]
                        existing_holds = [(h.group, h.a, h.b) for h in existing_ps.holds]
                        new_holds = [(h["group"], h["a"], h["b"]) for h in parsed.get("holds", [])]
                        if (
                            existing_ab == new_ab
                            and existing_holds == new_holds
                            and existing_ps.frame == frame
                        ):
                            saved_toml = cand_toml
                            reused_existing = True
                            if tp_out is not None and parsed.get("test_scene_prompt"):
                                tp_out.parent.mkdir(parents=True, exist_ok=True)
                                tp_out.write_text(parsed["test_scene_prompt"] + "\n", encoding="utf-8")
                    except Exception:
                        reused_existing = False
                if not reused_existing:
                    saved_toml = cv.write_markdown_to_toml(
                        text,
                        state.toml_dir,
                        name=name,
                        frame=frame,
                        test_prompt_out=tp_out,
                        overwrite=True if overwrite else False,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )
            else:
                compiled_ps = _compile_words_or_spec_to_pairset(
                    text,
                    name=name or None,
                    slot=slot,
                    mode=mode,
                    target_prompt=target_prompt,
                    target_weight=target_weight,
                )
                if not name:
                    name = compiled_ps.name
                if save_toml_req:
                    md_doc = _pair_set_to_markdown(None, compiled_ps)
                    saved_toml = cv.write_markdown_to_toml(
                        md_doc,
                        state.toml_dir,
                        name=name,
                        mode=compiled_ps.mode,
                        frame=frame,
                        quality=compiled_ps.quality,
                        a_side=compiled_ps.a_side,
                        b_side=compiled_ps.b_side,
                        overwrite=overwrite,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )

            argv = [
                "build",
                *( [str(saved_toml)] if saved_toml is not None else [] ),
                "--name",
                name,
                "--families",
                families,
                "--pooling",
                pooling,
                "--out-dir",
                str(state.vec_dir),
            ]
            if no_lora:
                argv.append("--no-lora")
            if unit_mean:
                argv.append("--unit-mean")
            if hold_words:
                argv.extend(["--hold-words", hold_words, "--hold-alpha", str(hold_alpha)])
            for hv in hold_vectors:
                if str(hv).strip():
                    argv.extend(["--hold", str(hv).strip()])
            if target_prompt and saved_toml is not None:
                argv.extend(["--target-prompt", target_prompt, "--target-weight", str(target_weight)])
            if include_encoder:
                argv.append("--include-encoder")
            else:
                argv.append("--no-encoder")
            if overwrite:
                argv.append("--overwrite")
            b_args = build_parser().parse_args(argv)
            if compiled_ps is not None and saved_toml is None:
                b_args._compiled_pair_set = compiled_ps
            rows_out: list[dict[str, Any]] = []
            build_cache: dict[str, Any] = {}
            b_args._summary_rows_out = rows_out
            b_args._build_cache_out = build_cache
            b_args._clip_override = state.get_clip(b_args.encoder, b_args.device)
            b_args._refiner_cache = state._refiner_cache
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_build(b_args)
            state.last_build_cache = build_cache
            disp_saved = (
                _display_path(saved_toml)
                if saved_toml is not None
                else str(compiled_ps.path if compiled_ps is not None else "")
            )
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": name,
                        "toml_path": disp_saved,
                        "saved_toml": disp_saved,
                        "built": rows_out,
                        "toml_files": _list_ui_toml_files(state.toml_dir),
                        "vectors": _list_ui_vectors(state.vec_dir),
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except (ValueError, FileExistsError, SystemExit) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/repool":
        try:
            req_name = str(req.get("name", "") or "").strip()
            if not state.last_build_cache and req_name:
                cand_toml = state.toml_dir / (req_name if req_name.endswith(".toml") else f"{req_name}.toml")
                if cand_toml.is_file():
                    import contextlib
                    import io

                    b_args = build_parser().parse_args(
                        [
                            "build",
                            str(cand_toml),
                            "--out-dir",
                            str(state.vec_dir),
                            "--overwrite",
                        ]
                    )
                    rows_out: list[dict[str, Any]] = []
                    build_cache: dict[str, Any] = {}
                    b_args._summary_rows_out = rows_out
                    b_args._build_cache_out = build_cache
                    b_args._clip_override = state.get_clip(b_args.encoder, b_args.device)
                    b_args._refiner_cache = state._refiner_cache
                    with contextlib.redirect_stdout(io.StringIO()):
                        cmd_build(b_args)
                    state.last_build_cache = build_cache
            res = _repool_cached_build(state, req)
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/sculpt":
        try:
            steps = req.get("steps") or req.get("specs")
            if not isinstance(steps, list) or not steps:
                return 400, "application/json", json.dumps({"ok": False, "error": "missing 'steps' recipe list"})
            name = str(req.get("output_name") or req.get("name") or "sculpted_vector").strip()
            target_fams = _resolve_requested_families(req.get("families", "both"))
            hold_words = str(req.get("hold_words", "") or "").strip() or None
            hold_alpha = float(req.get("hold_alpha", 1.0))
            prompt_req = str(req.get("prompt", "") or "").strip() or None
            overwrite_req = bool(req.get("overwrite", True))
            save_req = bool(req.get("save", True))

            if not save_req and j_lens.bundle_available():
                resolved_steps = []
                for st in steps:
                    v_src = state.vec_dir / str(st["vector"])
                    if not v_src.is_file():
                        v_src = cv_nodes.resolve_vector_file_path(str(st["vector"]))
                    resolved_steps.append(
                        {
                            "vector": v_src,
                            "mode": str(st.get("mode", "add")),
                            "strength": float(st.get("strength", 1.0)),
                        }
                    )
                rep = j_lens.decode_recipe(
                    resolved_steps,
                    family=target_fams[0],
                    hold_words=hold_words,
                    alpha=hold_alpha,
                    prompt=prompt_req,
                )
                return (
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "ok": True,
                            "saved": [],
                            "built": [],
                            "readout": rep.get("readout") or rep,
                            "readiness": rep.get("readiness") or (rep.get("readout") or {}).get("readiness"),
                            "prompt_coupling": rep.get("prompt_coupling") or (rep.get("readout") or {}).get("prompt_coupling"),
                        }
                    ),
                )

            saved_rows: list[dict[str, Any]] = []
            for tfam in target_fams:
                r = _compose_and_save_sculpted_vector(
                    steps,
                    name=name,
                    family=tfam,
                    hold_words=hold_words,
                    hold_alpha=hold_alpha,
                    prompt=prompt_req,
                    out_dir=state.vec_dir,
                    overwrite=overwrite_req,
                )
                saved_rows.append(
                    {
                        "file": r["file"],
                        "family": r["family"],
                        "raw_norm": r["raw_norm"],
                        "net_norm": r["raw_norm"],
                        "vector_norm": r["raw_norm"],
                        "rec_strength": r["rec_strength"],
                        "j_lens": r["j_lens"],
                    }
                )
            first_jl = (saved_rows[0].get("j_lens") if saved_rows else None) or {}
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "built": saved_rows,
                        "saved": saved_rows,
                        "readout": first_jl,
                        "readiness": first_jl.get("readiness"),
                        "prompt_coupling": first_jl.get("prompt_coupling"),
                        "vectors": _list_ui_vectors(state.vec_dir),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/prompt-meter":
        try:
            vec_arg = str(req.get("vector", "") or "").strip()
            prompt_arg = str(req.get("prompt", "") or "").strip()
            pooling_arg = str(req.get("pooling", "scene_text") or "scene_text")
            if not vec_arg or not prompt_arg:
                return 400, "application/json", json.dumps({"ok": False, "error": "both 'vector' and 'prompt' are required"})
            res = _meter_prompt_tokens(
                state,
                vector_arg=vec_arg,
                prompt_arg=prompt_arg,
                pooling=pooling_arg,
            )
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path in ("/api/build-media", "/api/make-media"):
        try:
            import contextlib
            import io

            img_a = str(req.get("image_a_data") or req.get("image_a") or "").strip()
            img_b = str(req.get("image_b_data") or req.get("image_b") or "").strip()
            out_name = str(req.get("output_name") or req.get("name") or "media_contrast").strip()
            families = str(req.get("families", "ref2va") or "ref2va")
            pooling = str(req.get("pooling", "scene_text") or "scene_text")
            short_edge = int(req.get("short_edge", 512))
            scene_prompt = str(req.get("scene_prompt") or req.get("prompt") or "").strip()
            hold_words = str(req.get("hold_words", "") or "").strip()
            hold_alpha = float(req.get("hold_alpha", 1.0))
            no_lora = bool(req.get("no_lora", False))
            overwrite = bool(req.get("overwrite", True))
            if not img_a or not img_b:
                return (
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "both Image A (baseline) and Image B (steered) are required"}),
                )
            argv = [
                "make-media",
                "--image-a",
                img_a,
                "--image-b",
                img_b,
                "--output-name",
                out_name,
                "--families",
                families,
                "--pooling",
                pooling,
                "--short-edge",
                str(short_edge),
                "--out-dir",
                str(state.vec_dir),
            ]
            if overwrite:
                argv.append("--overwrite")
            if no_lora:
                argv.append("--no-lora")
            if scene_prompt:
                argv.extend(["--scene-prompt", scene_prompt])
            if hold_words:
                argv.extend(["--hold-words", hold_words, "--hold-alpha", str(hold_alpha)])
            m_args = build_parser().parse_args(argv)
            m_args._clip_override = state.get_clip(m_args.encoder, m_args.device)
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_make_media(m_args)
            vec_list_after = _list_ui_vectors(state.vec_dir)
            built_media = [
                {
                    "file": v["file"],
                    "family": v.get("family", "ref2va"),
                    "grid": f"{short_edge}px (32-px M-RoPE)",
                    "raw_norm": v.get("raw_norm", 0.0),
                    "vector_norm": v.get("raw_norm", 0.0),
                    "rec_strength": v.get("rec_strength", ""),
                    "j_lens": v.get("j_lens"),
                }
                for v in vec_list_after
                if v["file"].startswith(f"{out_name}__")
            ]
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": out_name,
                        "built": built_media,
                        "vectors": vec_list_after,
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path in ("/api/j-lens", "/api/build-words", "/api/preview-words"):
        try:
            if not j_lens.bundle_available():
                return (
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "J-Lens bundle models/j_lens/j_lens_h3.safetensors not found"}),
                )
            from_words = str(req.get("from_words") or req.get("words") or req.get("pos_words") or "").strip()
            prompt_req = str(req.get("prompt", "") or "").strip() or None
            if from_words:
                fam = str(req.get("family", "fl2va") or "fl2va")
                families_req = str(req.get("families", "") or "").strip().lower()
                against_w = str(req.get("against_words", "") or "").strip() or None
                hold_w = str(req.get("hold_words", "") or "").strip() or None
                alpha_w = float(req.get("alpha", req.get("hold_alpha", 1.0)))
                synth = j_lens.synthesize_from_words(
                    from_words,
                    against_w,
                    family=fam if fam in ("fl2va", "ref2va") else "fl2va",
                    hold_words=hold_w,
                    alpha=alpha_w,
                    prompt=prompt_req,
                )
                save_name = str(req.get("save_name") or req.get("name") or "").strip()
                should_save = (
                    clean_path == "/api/build-words"
                    or (clean_path != "/api/preview-words" and (bool(req.get("save", False)) or bool(save_name)))
                )
                if not save_name and should_save:
                    save_name = "_".join(re.findall(r"[a-z0-9]+", from_words.lower())[:4]) or "word_vector"
                saved_file = None
                saved_files: list[str] = []
                built_words: list[dict[str, Any]] = []
                if should_save and save_name:
                    target_fams = (
                        _resolve_requested_families(families_req)
                        if families_req
                        else [synth["family"]]
                    )
                    for tfam in target_fams:
                        s_fam = (
                            synth
                            if tfam == synth["family"]
                            else j_lens.synthesize_from_words(
                                from_words,
                                against_w,
                                family=tfam,
                                hold_words=hold_w,
                                alpha=alpha_w,
                                prompt=prompt_req,
                            )
                        )
                        loras_meta = j_lens._DEFAULT_RECIPES.get(tfam, {}).get("loras")
                        fname = cv.default_vector_filename(save_name, tfam, pooling="diff_span", loras=loras_meta)
                        saved_path = j_lens.save_synthesized_vector(state.vec_dir / fname, s_fam, name=save_name)
                        saved_files.append(saved_path.name)
                        r_fam = s_fam["readout"]
                        r_norm = float(r_fam.get("norm_after_holds", r_fam.get("norm_before_holds", 65.0)))
                        built_words.append(
                            {
                                "file": saved_path.name,
                                "family": tfam,
                                "raw_norm": r_norm,
                                "rec_strength": _recommended_strength_text(r_norm),
                                "readout": r_fam,
                                "readiness": s_fam.get("readiness") or r_fam.get("readiness"),
                                "prompt_coupling": s_fam.get("prompt_coupling") or r_fam.get("prompt_coupling"),
                            }
                        )
                    if saved_files:
                        saved_file = saved_files[0]
                rep_words = {
                    "family": synth["family"],
                    "pos_words_matched": synth["pos_words_matched"],
                    "pos_words_unmatched": synth.get("pos_words_unmatched", []),
                    "pos_matched": synth["pos_words_matched"],
                    "neg_words_matched": synth["neg_words_matched"],
                    "hold_words": synth["hold_words"],
                    "hold_words_matched": synth["hold_words"],
                    "hold_matched": synth["hold_words"],
                    "alpha": synth["alpha"],
                    "name": save_name or None,
                    "save_name": save_name or None,
                    "saved_file": saved_file,
                    "saved_files": saved_files,
                    "built": built_words,
                    "readout": synth["readout"],
                    "readiness": synth.get("readiness") or synth["readout"].get("readiness"),
                    "prompt_coupling": synth.get("prompt_coupling") or synth["readout"].get("prompt_coupling"),
                }
                return (
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "ok": True,
                            **rep_words,
                            "report": rep_words,
                            "vectors": _list_ui_vectors(state.vec_dir) if saved_files else None,
                        }
                    ),
                )
            steps = req.get("steps")
            if isinstance(steps, list) and steps:
                resolved_steps = []
                for st in steps:
                    v_src = state.vec_dir / str(st["vector"])
                    if not v_src.is_file():
                        v_src = cv_nodes.resolve_vector_file_path(str(st["vector"]))
                    resolved_steps.append(
                        {
                            "vector": v_src,
                            "mode": str(st.get("mode", "add")),
                            "strength": float(st.get("strength", 1.0)),
                        }
                    )
                rep = j_lens.decode_recipe(
                    resolved_steps,
                    family=str(req.get("family", "fl2va") or "fl2va"),
                    hold_words=str(req.get("hold_words", "") or "").strip() or None,
                    alpha=float(req.get("alpha", req.get("hold_alpha", 1.0))),
                    prompt=prompt_req,
                )
                rep_out = {
                    **rep,
                    "net_norm": rep.get("norm_after_holds", rep.get("norm_before_holds", 0.0)),
                }
                return 200, "application/json", json.dumps({"ok": True, **rep_out, "report": rep_out})

            vec_name = str(req.get("vector", "") or "").strip()
            if not vec_name:
                return 400, "application/json", json.dumps({"ok": False, "error": "missing 'vector', 'steps', or 'from_words'"})
            v_path = state.vec_dir / vec_name
            if not v_path.is_file():
                v_path = cv_nodes.resolve_vector_file_path(vec_name)
            holds_raw = req.get("holds") or []
            hold_paths = []
            for h_name in holds_raw:
                hp = state.vec_dir / str(h_name)
                if not hp.is_file():
                    hp = cv_nodes.resolve_vector_file_path(str(h_name))
                hold_paths.append(hp)
            rep = j_lens.decode_safetensors(
                v_path,
                holds=hold_paths if hold_paths else None,
                hold_words=str(req.get("hold_words", "") or "").strip() or None,
                alpha=float(req.get("alpha", req.get("hold_alpha", 1.0))),
                prompt=prompt_req,
            )
            pr_ro = rep.get("post_refiner") or rep.get("readout") or {}
            rep_out = {
                **rep,
                "readout": pr_ro,
                "positive_words": pr_ro.get("positive_words", []),
                "negative_words": pr_ro.get("negative_words", []),
                "norm": float(pr_ro.get("norm_after_holds", pr_ro.get("norm_before_holds", 0.0))),
            }
            return 200, "application/json", json.dumps({"ok": True, **rep_out, "report": rep_out})
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/quit":
        state.shutdown_requested = True
        return 200, "application/json", json.dumps({"ok": True})

    return 404, "application/json", json.dumps({"ok": False, "error": f"not found: {clean_path}"})


def cmd_ui(args: argparse.Namespace) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import webbrowser

    toml_dir_override = Path(args.toml_dir) if getattr(args, "toml_dir", None) else None
    vec_dir_override = Path(args.vec_dir) if getattr(args, "vec_dir", None) else None
    state = ControlVectorUIState(
        toml_dir=toml_dir_override,
        vec_dir=vec_dir_override,
        default_encoder=getattr(args, "encoder", None),
        default_dit=getattr(args, "dit", None),
        default_device=getattr(args, "device", None),
        default_llm_provider=getattr(args, "llm_provider", None),
        default_llm_model=getattr(args, "llm_model", None),
    )
    srv_ref: list[ThreadingHTTPServer] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: Any) -> None:
            return

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            code, ctype, payload = handle_ui_request(state, method, self.path, body)
            raw_bytes = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw_bytes)
            if state.shutdown_requested and srv_ref:
                threading.Thread(target=srv_ref[0].shutdown, daemon=True).start()

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    server = ThreadingHTTPServer((str(args.host), int(args.port)), _Handler)
    srv_ref.append(server)
    host, actual_port = server.server_address[:2]
    url = f"http://{host}:{actual_port}/"
    print(f"=== MiniMax H3 Control Vector Studio ===")
    print(f"  url   : {url}")
    print(f"  note  : Click 'Quit Server' in the browser tab or press Ctrl+C here to exit.")
    if not getattr(args, "no_browser", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_sculpt(args: argparse.Namespace) -> int:
    """Compose one or more control vectors (`add`, `add_norm_matched`, `hold`) + optional `--hold-words` (`--hold-alpha`)
    into a Ready-to-Render `.safetensors` + `.md` sidecar on CPU in `< 5 ms`.
    """
    t0 = time.perf_counter()
    vecs = list(args.vectors or ())
    if not vecs:
        raise ValueError("at least one input vector is required for 'sculpt'")
    raw_modes = [m.strip() for m in str(args.modes or "").split(",") if m.strip()]
    raw_strengths = [float(s.strip()) for s in str(args.strengths or "").split(",") if s.strip()]
    invert_flag = bool(getattr(args, "invert", False))

    steps: list[dict[str, Any]] = []
    for idx, v_spec in enumerate(vecs):
        m_val = raw_modes[idx] if idx < len(raw_modes) else (raw_modes[-1] if raw_modes else "add")
        s_val = raw_strengths[idx] if idx < len(raw_strengths) else (raw_strengths[-1] if raw_strengths else 1.0)
        if invert_flag and idx == 0 and s_val > 0:
            s_val = -s_val
        steps.append({"vector": v_spec, "mode": m_val, "strength": s_val})
    for h_spec in (getattr(args, "hold", None) or ()):
        if str(h_spec).strip():
            steps.append(
                {
                    "vector": str(h_spec).strip(),
                    "mode": "hold",
                    "strength": float(getattr(args, "hold_alpha", 1.0)),
                }
            )

    stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or "").strip()
    if not stem:
        first_stem = Path(str(vecs[0])).stem.split("__")[0]
        stem = f"{first_stem}_inverted" if invert_flag else f"{first_stem}_sculpted"

    target_fams = _resolve_requested_families(getattr(args, "families", "both"))
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    hold_words = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha = float(getattr(args, "hold_alpha", 1.0))
    prompt_arg = str(getattr(args, "prompt", "") or "").strip() or None

    saved_rows: list[dict[str, Any]] = []
    for fam in target_fams:
        r = _compose_and_save_sculpted_vector(
            steps,
            name=stem,
            family=fam,
            hold_words=hold_words,
            hold_alpha=hold_alpha,
            prompt=prompt_arg,
            out_dir=out_dir,
            overwrite=bool(getattr(args, "overwrite", True)),
        )
        saved_rows.append(r)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"=== Sculpted {len(saved_rows)} Control Vector File(s) in {elapsed_ms:.1f} ms on CPU ===")
    for row in saved_rows:
        print(
            f"  * {row['file']}\n"
            f"      family={row['family']}  raw_norm={row['raw_norm']:.2f}  "
            f"recommended strength: {row['rec_strength']}"
        )
        jl = row.get("j_lens") or {}
        if jl.get("pos_summary"):
            print(f"      j_lens (+): {jl['pos_summary']}")
            print(f"      j_lens (-): {jl['neg_summary']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pui = sub.add_parser(
        "ui",
        help="Launch the self-contained local browser UI for authoring, previewing, and building control vectors",
    )
    pui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    pui.add_argument("--port", type=int, default=0, help="Port (default: 0 picks an unused ephemeral port)")
    pui.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not auto-open a browser tab")
    pui.add_argument("--toml-dir", dest="toml_dir", default=None, help="Optional directory override for .toml pair packs")
    pui.add_argument("--vec-dir", dest="vec_dir", default=None, help="Optional directory override for .safetensors control vectors")
    pui.add_argument("--encoder", default=None, help="Optional conditioning encoder filename override")
    pui.add_argument("--dit", dest="dit", default=None, help="Optional DiT checkpoint filename override")
    pui.add_argument("--device", default=None, help="Optional conditioning encoder device override")
    pui.add_argument(
        "--llm-provider",
        dest="llm_provider",
        choices=SUPPORTED_LLM_PROVIDERS,
        default=None,
        help="Default generative LLM provider for pair authoring ('transformers' or 'exllamav3')",
    )
    pui.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help="Default generative LLM model directory path or HuggingFace model ID",
    )

    pb = sub.add_parser(
        "build",
        help="One-shot end-to-end build: simple words (--words), Markdown (.md), or TOML (.toml) -> fl2va/ref2va .safetensors vectors",
    )
    pb.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Input LLM Markdown (.md) file, pair TOML (.toml) file, or stem under prompt_bank/control_vectors/ (omit when using --words, --placebo, or --all)",
    )
    pb.add_argument(
        "--words",
        default="",
        help="Simple words, short phrase, or 'pole_a -> pole_b' contrast (supports ';' or newline multi-clause, 'soundscape:', and 'hold:' lines) to compile into 10 canonical H3 scene pairs in memory",
    )
    pb.add_argument(
        "--slot",
        default="auto",
        help="Structural H3 slot for --words ('auto', 'opening_framing', 'subject_behavior', 'interpersonal_dynamics', 'camera_motion', 'speech_delivery', 'soundscape')",
    )
    pb.add_argument(
        "--target-prompt",
        dest="target_prompt",
        default="",
        help="Optional target H3 scene prompt (bank id, file path, or text) to compile as Pair 0 at --target-weight alongside 9 canonical frames",
    )
    pb.add_argument(
        "--target-weight",
        dest="target_weight",
        type=float,
        default=2.5,
        help="Weight for Pair 0 when --target-prompt is passed (default: 2.5)",
    )
    pb.add_argument(
        "--placebo",
        action="store_true",
        help="Build a size-matched benign placebo control vector in the same canonical frames",
    )
    pb.add_argument(
        "--all",
        dest="build_all",
        action="store_true",
        help="Build vectors for all .toml banks in prompt_bank/control_vectors/ in a single encoder load",
    )
    pb.add_argument(
        "--from-words",
        dest="from_words",
        default="",
        help="Synthesize a 5376-d control vector directly from simple positive words in <5 ms on CPU via J-Lens (no .toml or Qwen load required)",
    )
    pb.add_argument(
        "--against-words",
        dest="against_words",
        default="",
        help="Optional negative words when using --from-words (defaults to neutral baseline words)",
    )
    pb.add_argument(
        "--hold",
        action="append",
        default=None,
        help="Optional .safetensors control vector(s) to project out (at --hold-alpha) across any build mode",
    )
    pb.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU (at --hold-alpha) across any build mode",
    )
    pb.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when using --hold or --hold-words (default: 1.0)",
    )
    pb.add_argument(
        "--prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning and @prompt_subject",
    )
    pb.add_argument(
        "--unit-mean",
        dest="unit_mean",
        action="store_true",
        help="Average unit-normalized pair directions (equal directional weight per pair) scaled to mean pair norm",
    )
    pb.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to build vectors for in one pass (default: 'both' builds fl2va and ref2va)",
    )
    pb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode or comma-separated modes (e.g. 'diff_span', 'from_first_diff', 'scene_text', or 'all'; default: 'diff_span')",
    )
    pb.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pb.add_argument("--output-name", dest="output_name", default="", help="Optional vector stem override")
    pb.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override when importing .md or compiling --words")
    pb.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pb.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pb.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pb.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pb.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also write the compiled 10-pair .toml file under prompt_bank/control_vectors/ when using --words")
    pb.add_argument("--toml-out", dest="toml_out", default=None, help="Optional output path/dir for generated .toml")
    pb.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional output path for extracted Test Scene Prompt")
    pb.add_argument("--dit", dest="dit", default=None, help="Explicit DiT diffusion checkpoint filename override")
    pb.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pb.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pb.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override (default: standard Pika H3 Flex LoRA per family)")
    pb.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build for stock 30-step base DiT without Pika LoRAs")
    pb.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pb.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pb.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pb.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", default="default")
    pb.add_argument("--refiner-device", default="cpu")
    pb.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")
    pb.add_argument("--overwrite", action="store_true", help="Overwrite existing .toml and .safetensors files")

    pfm = sub.add_parser(
        "from-md",
        aliases=["import-md"],
        help="Convert an LLM-authored control-vector Markdown file into a validated .toml pair file",
    )
    pfm.add_argument("md_path", help="Input Markdown (.md) file path")
    pfm.add_argument(
        "--out",
        default=None,
        help="Output .toml file path or output directory (defaults to prompt_bank/control_vectors/<stem>.toml)",
    )
    pfm.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pfm.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override (auto-detected if omitted)")
    pfm.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pfm.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pfm.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pfm.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pfm.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional path to write the extracted Test Scene Prompt text")
    pfm.add_argument("--overwrite", action="store_true", help="Overwrite an existing .toml file at the target path")

    pm = sub.add_parser("make", help="Build a control vector from a pair TOML file or --words")
    pm.add_argument("pairs", nargs="?", default=None, help="Pair TOML file path or stem under prompt_bank/control_vectors/ (omit when using --words)")
    pm.add_argument("--words", default="", help="Simple words or 'pole_a -> pole_b' contrast to compile into 10 canonical H3 pairs in memory")
    pm.add_argument("--slot", default="auto", help="Structural H3 slot for --words ('auto' or one of the 6 canonical slots)")
    pm.add_argument("--target-prompt", dest="target_prompt", default="", help="Optional target H3 scene prompt to compile as Pair 0")
    pm.add_argument("--target-weight", dest="target_weight", type=float, default=2.5, help="Weight for Pair 0 when --target-prompt is passed")
    pm.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pm.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pm.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pm.add_argument("--pooling", choices=cv.VALID_POOLING, default="diff_span")
    pm.add_argument("--unit-mean", dest="unit_mean", action="store_true", help="Average unit-normalized pair directions")
    pm.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pm.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pm.add_argument("--lora", action="append", default=None, help="LoRA file[:strength]")
    pm.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pm.add_argument("--ref-role", default="subject", help="Role for first reference image")
    pm.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pm.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pm.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pm.add_argument("--output-name", default="", help="Optional vector stem override")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--device", default="default")
    pm.add_argument("--refiner-device", default="cpu")
    pm.add_argument("--out", default=None, help="Output .safetensors path or filename")
    pm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file at the target path")

    pmm = sub.add_parser(
        "make-media",
        help="Build a control vector from contrasting <Picture 1> image pair(s) with 32-px M-RoPE grid matching",
    )
    pmm.add_argument("--image-a", required=True, help="Baseline/negative pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--image-b", required=True, help="Positive/steered pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--output-name", default="media_contrast", help="Output vector stem (default: 'media_contrast')")
    pmm.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="ref2va",
        help="Which H3 model families to build media vectors for (default: 'ref2va')",
    )
    pmm.add_argument("--pooling", choices=cv.VALID_MEDIA_POOLING, default="scene_text")
    pmm.add_argument("--short-edge", type=int, default=512, help="Target shorter edge in pixels (snapped to 32)")
    pmm.add_argument("--scene-prompt", default="", help="Optional neutral H3 ref2va scene prompt")
    pmm.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pmm.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pmm.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pmm.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pmm.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pmm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pmm.add_argument("--lora", action="append", default=None, help="LoRA file[:strength]")
    pmm.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build without Pika LoRAs")
    pmm.add_argument("--device", default="default")
    pmm.add_argument("--refiner-device", default="cpu")
    pmm.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory")
    pmm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file")

    psculpt = sub.add_parser(
        "sculpt",
        help="Compose multi-vector recipes (add, add_norm_matched, hold) + --hold-words (--hold-alpha) into Ready-to-Render .safetensors files in <5 ms on CPU",
    )
    psculpt.add_argument("vectors", nargs="+", help="One or more .safetensors vector files or stems to compose")
    psculpt.add_argument(
        "--modes",
        default="add",
        help="Comma-separated modes matching 'vectors' ('add', 'add_norm_matched', 'hold'; default: 'add')",
    )
    psculpt.add_argument(
        "--strengths",
        default="1.0",
        help="Comma-separated strengths (or hold alphas in [0.0, 2.0]) matching 'vectors' (default: '1.0')",
    )
    psculpt.add_argument("--invert", action="store_true", help="Invert the primary vector direction (multiply strength by -1.0)")
    psculpt.add_argument("--hold", action="append", default=None, help="Additional .safetensors vector(s) to project out (`mode = 'hold'`)")
    psculpt.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    psculpt.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Attenuation alpha in [0.0, 2.0] for --hold and --hold-words (default: 1.0)",
    )
    psculpt.add_argument("--prompt", default="", help="Optional target H3 scene prompt for prompt-coupling readout")
    psculpt.add_argument("--name", default=None, help="Output vector stem")
    psculpt.add_argument("--output-name", dest="output_name", default="", help="Output vector stem override")
    psculpt.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to write (default: 'both' writes fl2va and ref2va)",
    )
    psculpt.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory")
    psculpt.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing output files (default: True)")

    pi = sub.add_parser("inspect", help="Inspect a control vector file and optionally meter prompts")
    pi.add_argument("vector", help="Path or filename in models/control_vectors/")
    pi.add_argument(
        "--hold",
        action="append",
        default=None,
        help="Optional control vector .safetensors file(s) to project out before J-Lens readout",
    )
    pi.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU before J-Lens readout",
    )
    pi.add_argument(
        "--prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning",
    )
    pi.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when --hold is passed (default: 1.0)",
    )
    pi.add_argument("--check-dit", dest="check_dit", default=None, help="Verify compatibility against a DiT checkpoint")
    pi.add_argument("--check-unet", dest="check_dit", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--check-encoder", default=None, help="Verify compatibility against an encoder checkpoint")
    pi.add_argument("--check-lora", action="append", default=None)
    pi.add_argument("--allow-recipe-mismatch", action="store_true")
    pi.add_argument("--meter", action="append", default=None, help="Prompt-bank id or file path to meter")
    pi.add_argument("--meter-look-family", action="store_true", help="Meter t2va_look_anchor and all 8 look siblings (E2.8)")
    pi.add_argument("--device", default="default")
    pi.add_argument("--refiner-device", default="cpu")

    pg = sub.add_parser("grade", help="Grade activation/prediction movement against nulls and edit baselines")
    pg.add_argument("--vector", required=True, help="Vector .safetensors path or name")
    pg.add_argument("--prompt", required=True, help="Base prompt id or path")
    pg.add_argument("--edit-prompt", default=None, help="Edit baseline prompt id or path")
    pg.add_argument("--pairs", default=None, help="Pair TOML file to build flip_null")
    pg.add_argument("--strengths", default="0,0.5,1.0,2.0,-1.0")
    pg.add_argument("--probe-steps", default="0,8,15", help="Schedule step indices for DiT forward grading")
    pg.add_argument("--no-dit", action="store_true", help="Run post-refiner context movement only without loading the DiT")
    pg.add_argument("--sol", action="store_true", help="Opt in to MiniMaxH3Sol sparse attention during DiT grading (default: stock dense attention)")
    pg.add_argument("--compare-unmasked", action="store_true", help="Also evaluate mask='all_rows' for E2.3")
    pg.add_argument("--width", type=int, default=1152)
    pg.add_argument("--height", type=int, default=768)
    pg.add_argument("--length", type=int, default=h3_config.LENGTH)
    pg.add_argument("--noise-seed", type=int, default=730451892)
    pg.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pg.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pg.add_argument("--encoder", default=None)
    pg.add_argument("--lora", action="append", default=None, help="LoRA file[:strength] (defaults to vector recipe.loras)")
    pg.add_argument("--no-lora", action="store_true", help="Ignore vector recipe.loras during grading")
    pg.add_argument("--manual-sigmas", default=None, help="Comma-separated manual sigmas override")
    pg.add_argument("--pika", action="store_true", help="Use Pika H3 Flex 6-step manual sigmas and default probe steps 0,2,5")
    pg.add_argument("--ref-images", default=None)
    pg.add_argument("--ref-role", default="subject")
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--device", default="default")
    pg.add_argument("--refiner-device", default="cpu")
    pg.add_argument("--out", default=None, help="Optional JSON output path for grade report")

    pw = sub.add_parser(
        "words",
        help="Synthesize Ready-to-Render fl2va & ref2va .safetensors vectors directly from 3-5 words in <2 ms on CPU via J-Lens (with Pre-Render Readiness Predictor)",
    )
    pw.add_argument(
        "pos_words",
        help="Comma- or space-separated positive concept words (e.g. 'blizzard, snowdrift, freezing, snow' or '8-bit, LED, steampunk')",
    )
    pw.add_argument(
        "--against-words",
        dest="against_words",
        default="",
        help="Optional negative words or preset ('@prompt_subject', '@neutral', '@replace_human', '@replace_live_action', '@replace_digital')",
    )
    pw.add_argument(
        "--prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning and @prompt_subject extraction",
    )
    pw.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU",
    )
    pw.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Attenuation alpha in [0.0, 2.0] for --hold-words (default: 1.0)",
    )
    pw.add_argument("--name", default=None, help="Optional output vector stem override")
    pw.add_argument("--output-name", dest="output_name", default="", help="Optional output vector stem override")
    pw.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to synthesize (default: 'both' writes fl2va and ref2va)",
    )
    pw.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")

    pgen = sub.add_parser(
        "generate-pairs",
        help="Generate 10 surgical H3 Prompt A / Prompt B pairs using a local generative LLM ('transformers' or 'exllamav3') and h3_control_vector_system_prompt.md",
    )
    pgen.add_argument(
        "concept",
        help="Target control-vector concept or 'pole_a -> pole_b' description",
    )
    pgen.add_argument(
        "--llm-provider",
        dest="llm_provider",
        choices=SUPPORTED_LLM_PROVIDERS,
        default=os.environ.get("H3_LLM_PROVIDER", "transformers") or "transformers",
        help="Generative LLM provider ('transformers' or 'exllamav3'; default: 'transformers' or $H3_LLM_PROVIDER)",
    )
    pgen.add_argument(
        "--llm-model",
        dest="llm_model",
        default=os.environ.get("H3_LLM_MODEL", "") or os.environ.get("H3_EXL3_MODEL_DIR", ""),
        help="Path to local LLM model directory or HuggingFace model ID (default: $H3_LLM_MODEL or $H3_EXL3_MODEL_DIR)",
    )
    pgen.add_argument(
        "--exl3-root",
        dest="exl3_root",
        default=os.environ.get("EXLLAMAV3_ROOT", ""),
        help="Optional path to exllamav3 checkout when using --llm-provider exllamav3",
    )
    pgen.add_argument(
        "--no-4bit",
        dest="no_4bit",
        action="store_true",
        help="Disable 4-bit BitsAndBytes quantization when loading a transformers model",
    )
    pgen.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=3200)
    pgen.add_argument("--temperature", type=float, default=0.7)
    pgen.add_argument("--name", default="", help="Optional stem name to also write <name>.md / <name>.toml")
    pgen.add_argument("--out-md", dest="out_md", default="", help="Optional output .md file path")
    pgen.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also parse and save to prompt_bank/control_vectors/<name>.toml")
    pgen.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    return p


def cmd_generate_pairs(args: argparse.Namespace) -> int:
    res = _generate_pairs_with_llm(
        provider=args.llm_provider,
        model_spec=args.llm_model,
        concept_prompt=args.concept,
        load_in_4bit=not bool(getattr(args, "no_4bit", False)),
        max_new_tokens=int(getattr(args, "max_new_tokens", 3200)),
        temperature=float(getattr(args, "temperature", 0.7)),
        exl3_root=getattr(args, "exl3_root", None) or None,
    )
    md_text = res["markdown"]
    if getattr(args, "out_md", ""):
        out_p = Path(args.out_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_text + "\n", encoding="utf-8")
        print(f"wrote {out_p} ({res['elapsed_s']}s via {res['provider']})")
    else:
        print(md_text)
    if getattr(args, "save_toml", False) and getattr(args, "name", ""):
        saved = cv.write_markdown_to_toml(
            md_text,
            cv_nodes.PAIR_BANK_DIR,
            name=str(args.name).strip(),
            overwrite=bool(getattr(args, "overwrite", False)),
            bank_dir=cv_nodes.PROMPT_BANK_DIR,
        )
        print(f"wrote {_display_path(saved)}")
    return 0


def cmd_words(args: argparse.Namespace) -> int:
    """Synthesize fl2va + ref2va control vectors directly from plain English words in < 2 ms via J-Lens."""
    args.from_words = args.pos_words
    args.build_all = False
    return cmd_build(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "ui":
            return cmd_ui(args)
        if args.cmd == "generate-pairs":
            return cmd_generate_pairs(args)
        if args.cmd == "words":
            return cmd_words(args)
        if args.cmd == "sculpt":
            return cmd_sculpt(args)
        if args.cmd == "build":
            return cmd_build(args)
        if args.cmd in ("from-md", "import-md"):
            return cmd_from_md(args)
        if args.cmd == "make":
            return cmd_make(args)
        if args.cmd == "make-media":
            return cmd_make_media(args)
        if args.cmd == "inspect":
            return cmd_inspect(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
