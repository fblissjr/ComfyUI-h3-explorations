"""Shared helpers for the MiniMax H3 control vector CLI and UI package."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import folder_paths
import torch

import control_vector_nodes as cv_nodes
import control_vectors as cv
import h3_pair_compiler as pair_compiler
import h3_spans
import j_lens
from workflows import h3_config

PACKAGE_DIR = Path(__file__).resolve().parent
BENCH_DIR = PACKAGE_DIR.parent
REPO_ROOT = BENCH_DIR.parent
WEB_DIR = PACKAGE_DIR / "web"

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


def _resolve_cli_target_prompt(args: Any) -> str:
    """Resolve `--target-prompt` (literal inline text) or `--target-prompt-file` (file or prompt_bank stem)."""
    tp_inline = str(getattr(args, "target_prompt", "") or "").strip()
    tp_file = str(getattr(args, "target_prompt_file", "") or "").strip()
    if tp_inline and tp_file:
        raise ValueError("cannot pass both --target-prompt and --target-prompt-file")
    if tp_file:
        _, resolved = _resolve_prompt_text(tp_file)
        return resolved
    return tp_inline


def _free_idle_comfy_vram(min_free_gib: float = 18.0) -> bool:
    """If local ComfyUI (`H3_COMFY_URL` / `COMFY_URL` / `http://127.0.0.1:8188`) is idle
    (`queue_running == []`) and GPU 0 has less than `min_free_gib` GiB free, request `/free`
    so CLI encoder/LLM loads do not hit CUDA OOM."""
    if not torch.cuda.is_available():
        return False
    try:
        free_b, _ = torch.cuda.mem_get_info(0)
        if free_b >= int(min_free_gib * (1024**3)):
            return False
    except Exception:
        return False
    import urllib.request

    base_url = (
        os.environ.get("H3_COMFY_URL")
        or os.environ.get("COMFY_URL")
        or "http://127.0.0.1:8188"
    ).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/queue", timeout=1.5) as resp:
            q = json.loads(resp.read().decode("utf-8"))
        if q.get("queue_running"):
            return False
        req = urllib.request.Request(
            f"{base_url}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0):
            pass
        return True
    except Exception:
        return False


def _load_clip(encoder_name: str, device: str = "default") -> Any:
    if str(device).lower() != "cpu":
        _free_idle_comfy_vram()
    enc_loader_mod = cv_nodes._repo("h3_encoder_loader")

    (clip,) = enc_loader_mod.MiniMaxH3EncoderLoader.execute(
        encoder_name=encoder_name, device=device
    )
    csm = getattr(clip, "cond_stage_model", None)
    if csm is not None and not hasattr(csm, "memory_estimation_function"):
        csm.memory_estimation_function = lambda _tokens, device=None: 2560 * 1024 * 1024
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
    use_vae: bool = False,
) -> Any:
    """Build `H3EncodedReferences` from `--ref-images` so E2.4 can compare vectors
    made with and without a ref2va reference prefix.

    By default each picture is encoded for the text encoder alone and no VAE is loaded. `build`, `make`,
    `from-prompt` and `bench/control_vector_rows.py` read text rows and never sample, and the video VAE
    beside the 32B encoder ran a 24 GB card out of memory (seen 2026-11-09). `use_vae=True` is for a caller
    that samples: `grade` passes it when its DiT probe runs, because `encode_conditioning_for_vector` hands
    the references' latents to the DiT as `minimax_refs` when they exist. That path has not been run: it
    may meet the same memory limit."""
    from PIL import Image
    import numpy as np
    import comfy.sd
    import comfy.utils

    ref_cond_mod = cv_nodes._repo("reference_conditioning")
    ref_enc_mod = cv_nodes._repo("reference_encode")

    vae = None
    if use_vae:
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
            # The sizing the shipped video reference graphs carry
            # (`workflows/build_workflows.py::_append_image_inputs`): the node's own `max`
            # defaults, and the config's separate text-encoder view.
            size_policy="max",
            qwen_view={"qwen_view": "separate", "qwen_short_edge": h3_config.REF_QWEN_SHORT_EDGE},
            references=refs,
            use_vae=use_vae,
        )
    # The node returns the encoded references and an empty latent; a sampler takes the latent, this does not.
    encoded_refs, _latent = ref_enc_mod.MiniMaxH3EncodeReferences.execute(
        clip=clip,
        vae=vae,
        references=refs,
        width=width,
        height=height,
        length=length,
    )
    return encoded_refs


_FL2VA_FAMILY_ALIASES: frozenset[str] = frozenset(
    {
        "fl2va",
        "t2va",
        "i2va",
        "image",
        "p-h3-image",
        "p_h3_image",
        "t2i",
        "image_t2i",
        "image-t2i",
        "image_edit",
        "image-edit",
    }
)


def _resolve_requested_families(
    raw: Any,
    *,
    auto_family: str | None = None,
) -> list[str]:
    """Normalize family selector ('both', 'fl2va', 'ref2va', 'auto', or aliases like 'image', 't2i', 'image_edit') into a list of families."""
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
        elif item in _FL2VA_FAMILY_ALIASES:
            if "fl2va" not in out:
                out.append("fl2va")
        elif item == "ref2va":
            if "ref2va" not in out:
                out.append("ref2va")
        else:
            raise ValueError(
                f"unknown family {item!r}; expected 'both' (fl2va + ref2va), 'fl2va' (Video T2VA/FL2VA/I2VA + Image T2I/Edit), or 'ref2va' (Video Ref2VA)"
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
    fam = "ref2va" if "ref2va" in str(family).lower() else "fl2va"
    other_fam = "fl2va" if fam == "ref2va" else "ref2va"

    def _maybe_swap_sibling(found: Path) -> Path:
        tag_other = f"__{other_fam}"
        if tag_other not in found.name:
            return found
        s_stem = found.name.split(tag_other, 1)[0]
        pooling_hint = (
            "diff_span"
            if "__diff_span" in found.name
            else ("from_first_diff" if "__from_first_diff" in found.name else None)
        )
        sib = cv_nodes._match_sibling_or_stem_in_dir(
            found.parent,
            s_stem,
            fam,
            preferred_pooling=pooling_hint,
            prefer_lora="__lora_" in found.name,
        )
        if sib is not None and sib.is_file():
            return sib
        return found

    if cand.is_file():
        return _maybe_swap_sibling(cand)
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
            return _maybe_swap_sibling(direct)
        if not cand.suffix:
            direct_st = cv_dir / f"{cand.name}.safetensors"
            if direct_st.is_file():
                return _maybe_swap_sibling(direct_st)
        for pat in (
            f"{stem}__{fam}__diff_span__*.safetensors",
            f"{stem}__{fam}__*.safetensors",
            f"{stem}__*.safetensors",
        ):
            hits = sorted(cv_dir.glob(pat))
            if hits:
                return hits[0]
    return cv_nodes.resolve_vector_file_path(str(spec).strip(), preferred_family=fam)


def _apply_post_refiner_holds(
    final_vec: torch.Tensor,
    *,
    family: str,
    hold_specs: list[str] | tuple[str, ...] | None = None,
    hold_words: str | None = None,
    hold_alpha: float = 1.0,
    prompt: str | None = None,
    extra_basis: torch.Tensor | None = None,
    extra_dir: Path | None = None,
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
        h_path = _resolve_vector_for_family(str(h_spec).strip(), family=family, extra_dir=extra_dir)
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
    if extra_basis is not None and extra_basis.numel() > 0:
        scrubbed = cv.project_out(scrubbed, extra_basis)
    norm_after_raw = float(torch.linalg.vector_norm(scrubbed.to(dtype=torch.float64)).item())
    if norm_after_raw <= 1e-6 * max(1.0, norm_before):
        raise ValueError("holds completely cancelled the control vector (residual norm is zero)")
    return scrubbed.to(dtype=torch.float32), hold_labels, norm_before, norm_after_raw


def _compile_words_or_spec_to_pairset(
    words_arg: str,
    *,
    name: str | None = None,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    mode: str | None = None,
    target_prompt: str = "",
    target_weight: float = 1.0,
    bank_scenes: int = 0,
    **_unused_kwargs: Any,
) -> cv.PairSet:
    """Compile a `--words` CLI argument or multi-line/semicolon-separated spec in-place on `target_prompt`."""
    raw_text = str(words_arg or "").strip()
    tp_resolved = str(target_prompt or "").strip()

    has_meta_header = any(
        raw_text.lower().startswith(p) or f"\n{p}" in raw_text.lower()
        for p in ("name:", "quality:", "a_side:", "b_side:", "mode:", "target_prompt:", "scene_prompt:", "bank_scenes:")
    )
    if has_meta_header and "\n" not in raw_text and ";" in raw_text:
        raw_text = "\n".join(part.strip() for part in raw_text.split(";") if part.strip())
    if has_meta_header:
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
        if mode and not any(ln.startswith("mode:") for ln in lower_lines):
            hdr_lines.append(f"mode: {mode}")
        full_spec = ("\n".join(hdr_lines) + "\n" + raw_text).strip() if hdr_lines else raw_text
        spec_out = pair_compiler.parse_short_contrast_spec(
            full_spec,
            target_prompt=tp_resolved or None,
            target_weight=float(target_weight),
            bank_scenes=int(bank_scenes),
        )
        return spec_out["pair_set"]

    return pair_compiler.compile_words_to_pairset(
        raw_text,
        name=name,
        quality=quality,
        a_side=a_side,
        b_side=b_side,
        mode=mode,
        target_prompt=tp_resolved or None,
        target_weight=float(target_weight),
        bank_scenes=int(bank_scenes),
    )


def _encode_pair_set_through_clip(
    clip: Any,
    pair_set: cv.PairSet,
    *,
    encoded_refs: Any = None,
    no_holds: bool = False,
    log_rows: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, float]], list[float]]:
    """Shared single-pass Qwen3-VL `layers.49` pair and hold encoder for `build` and `make`."""
    cached_pairs: list[dict[str, Any]] = []
    enc_censuses: list[dict[str, float]] = []
    weights_list: list[float] = []
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
        if log_rows:
            print(
                f"  [{idx:02d}] w={item.weight:.2f} diff_toks=[{first_diff}:{end_a}/{end_b}] "
                f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} | "
                f"-{item.a_span[:42]!r} -> +{item.b_span[:42]!r}"
            )

    cached_holds: list[dict[str, Any]] = []
    if not no_holds and pair_set.holds:
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
                    "tags_a": h_tags_a,
                    "ids_a": h_ids_a,
                    "spans_a": h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a),
                    "enc_b": h_enc_b,
                    "tags_b": h_tags_b,
                    "ids_b": h_ids_b,
                    "spans_b": h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b),
                    "first_diff": h_diff,
                    "end_a": h_end_a,
                    "end_b": h_end_b,
                }
            )
    return cached_pairs, cached_holds, enc_censuses, weights_list



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


def _display_path(p: Path) -> str:
    resolved = p.resolve()
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    parts = resolved.parts
    if "ComfyUI-h3-explorations" in parts:
        idx = parts.index("ComfyUI-h3-explorations")
        return str(Path(*parts[idx + 1 :]))
    return p.name



def _recommended_strength_text(raw_norm: float) -> str:
    """The strength to try, for the node's default mode first.

    Until 2026-11-09 this returned only the range for mode `add`, which depends on the vector's stored
    length. The node defaults to `add_norm_matched`, which rescales every vector to one length, so that
    range did not apply to it. Both come from `control_vectors.strength_ranges`.
    """
    (m_lo, m_hi), add = cv.strength_ranges(raw_norm)
    text = f"{m_lo:.2f} .. {m_hi:.2f} (add_norm_matched, the default)"
    if add is None:
        return text
    s_lo = max(0.25, round(add[0], 2))
    s_hi = max(s_lo + 0.25, round(add[1], 2))
    return f"{text}; {s_lo:.2f} .. {s_hi:.2f} (add)"



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
        w_tag = f" (weight={item.weight:g})" if round(float(item.weight), 6) != 1.0 else ""
        chunks.append(
            f"#### Pair {idx}{w_tag}\n"
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

