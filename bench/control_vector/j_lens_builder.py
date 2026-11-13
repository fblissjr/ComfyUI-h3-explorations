r"""Reproducible builder and verifier for `<ComfyUI>/models/j_lens/j_lens_h3.safetensors` (`h3_j_lens_v1`).

Mathematical Architecture of the J-Lens Operator Bundle (`h3_j_lens_v1`)
========================================================================
MiniMax H3 extracts text conditioning from `Qwen3-VL-32B-Instruct` truncated after
`layers.49` (`hidden_size = 5120`), projects it through the DiT's `condition_proj`
(`5120 -> 5376`), and passes it through the 2-block bidirectional `TokenRefiner`
(`5376 -> 5376`) before feeding refined text tokens into the 50-block DiT.

Because ComfyUI's shipped H3 text encoder (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`)
is truncated at `layers.49` (omitting `layers.50..63` and `lm_head`), and because
control vectors operate in `5376`-d post-refiner space after `TokenRefiner`,
`models/j_lens/j_lens_h3.safetensors` precomputes and stores two coupled operator
families:

1. **Encoder (`layers.49`) <-> Post-Refiner (`5376`-d) Transport & Wiener Inverse**
   (computed from the shipped 50-layer ComfyUI encoder + both H3 `TokenRefiner`
   families `fl2va` and `ref2va` over a calibration corpus of `prompt_bank/` prompts,
   restricting expectations to non-sink tokens `[:, 1:, :]`):
   - `Sigma_49_p030` (`[5120, 5120]`, float16): Spectral-filtered fractional
     covariance $\Sigma_{49, \text{filt}}^{0.30}$ of `tok1+` activations at
     `layers.49`, with the top-1 common-mode eigenvalue $\lambda_{\max}$ zeroed
     out and remaining eigenvalues raised to $p = 0.30$.
   - `J_ref_fl2va`, `J_ref_ref2va` (`[5376, 5120]`, float32): Analytical expected
     Jacobians $\bar{J}_{\text{ref}}^{(f)} = \mathbb{E}[\partial \mathbf{r}_{\text{PR}} / \partial \mathbf{h}_{49}]$
     across `condition_proj` and the 2-block `TokenRefiner` (including the
     broadcast attention value path $\beta W_O W_V J_{\text{RMSNorm}_1}$ and
     SwiGLU FFN $J_{\text{FFN}} J_{\text{RMSNorm}_2}$).
   - `J_wiener_fl2va`, `J_wiener_ref2va` (`[5120, 5376]`, float32): Closed-form
     empirical-covariance Wiener inverses ($\gamma = 0.20$ shrinkage,
     $\lambda_{\text{rel}} = 0.03$):
     $$J_{\text{Wiener}}^{(f)} = \Sigma_{49} (\bar{J}_{\text{ref}}^{(f)})^\top
       \left(\bar{J}_{\text{ref}}^{(f)} \Sigma_{49} (\bar{J}_{\text{ref}}^{(f)})^\top + \lambda^2 I_{5376}\right)^{-1}$$

2. **`layers.49` -> `layers.63` -> `lm_head` Lexical Unembedding Operators**
   (preserved from an existing `j_lens_h3.safetensors` bundle via `--base-bundle`
   when refreshing `TokenRefiner` operators against new DiT checkpoints or Pika LoRAs):
   - `J_49_to_63` (`[5120, 5120]`, float32): Expected Jacobian from `layers.49`
     through `layers.50..63` and final `RMSNorm`.
   - `M_union` (`[27407, 5120]`, float16): Pre-multiplied unembedding rows
     $W_{\text{head}}[\text{union\_idx}] \bar{J}_{49 \to 63}$.
   - `cloze_base_union` (`[7, 27407]`, float32): Unsteered baseline logit vectors
     across the 7 syntactic H3 cloze prefixes.
   - `union_idx` (`[27407]`, int32), `valid_idx` (`[24912]`, int32),
     `valid_pos_in_union` (`[24912]`, int32), and `vocab_words_json` (`24,912`
     space-prefixed English words).

See `docs/wiki/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md` for
full derivations and empirical benchmarks.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import control_vector_nodes as cv_nodes
import control_vectors as cv
import j_lens
from workflows import h3_config

from . import helpers as _h
from .helpers import REPO_ROOT, _parse_lora_specs

REQUIRED_BUNDLE_TENSORS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "J_49_to_63": ((5120, 5120), torch.float32),
    "J_ref_fl2va": ((5376, 5120), torch.float32),
    "J_ref_ref2va": ((5376, 5120), torch.float32),
    "J_wiener_fl2va": ((5120, 5376), torch.float32),
    "J_wiener_ref2va": ((5120, 5376), torch.float32),
    "M_union": ((27407, 5120), torch.float16),
    "Sigma_49_p030": ((5120, 5120), torch.float16),
    "cloze_base_union": ((7, 27407), torch.float32),
    "union_idx": ((27407,), torch.int32),
    "valid_idx": ((24912,), torch.int32),
    "valid_pos_in_union": ((24912,), torch.int32),
}

OPTIONAL_BUNDLE_TENSORS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "J_ref_h3_image": ((5376, 5120), torch.float32),
    "J_wiener_h3_image": ((5120, 5376), torch.float32),
}


def verify_j_lens_bundle(bundle_path: str | Path | None = None) -> dict[str, Any]:
    """Verify that `j_lens_h3.safetensors` exists, matches the `h3_j_lens_v1`
    contract, has finite tensors of the exact expected shapes, and round-trips
    through `j_lens.load_j_lens_bundle` and `j_lens.synthesize_from_words`."""
    path = Path(bundle_path).resolve() if bundle_path is not None else j_lens.default_bundle_path()
    if not path.is_file():
        raise FileNotFoundError(f"J-Lens operator bundle not found at {path}")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        if meta.get("format") != "h3_j_lens_v1":
            raise ValueError(
                f"invalid format {meta.get('format')!r} in {path.name}; expected 'h3_j_lens_v1'"
            )
        if "vocab_words_json" not in meta:
            raise ValueError(f"missing 'vocab_words_json' metadata in {path.name}")
        words = json.loads(meta["vocab_words_json"])
        if not isinstance(words, list) or len(words) < 1000:
            raise ValueError(f"invalid vocab_words_json list in {path.name} (len={len(words) if isinstance(words, list) else 'N/A'})")

        keys = set(f.keys())
        missing = sorted(set(REQUIRED_BUNDLE_TENSORS.keys()) - keys)
        if missing:
            raise ValueError(f"{path.name} is missing required tensor(s): {missing}")

        tensor_summary: dict[str, dict[str, Any]] = {}
        expected_specs = dict(REQUIRED_BUNDLE_TENSORS)
        for opt_k, opt_spec in OPTIONAL_BUNDLE_TENSORS.items():
            if opt_k in keys:
                expected_specs[opt_k] = opt_spec
        for k, (exp_shape, exp_dtype) in expected_specs.items():
            t = f.get_tensor(k)
            if k in ("M_union", "cloze_base_union", "union_idx", "valid_idx", "valid_pos_in_union"):
                if t.ndim != len(exp_shape):
                    raise ValueError(f"tensor {k!r} rank mismatch: got {tuple(t.shape)}, expected rank {len(exp_shape)}")
            elif tuple(t.shape) != exp_shape:
                raise ValueError(f"tensor {k!r} shape mismatch: got {tuple(t.shape)}, expected {exp_shape}")
            if t.dtype != exp_dtype:
                raise ValueError(f"tensor {k!r} dtype mismatch: got {t.dtype}, expected {exp_dtype}")
            if t.is_floating_point() and not bool(torch.isfinite(t.float()).all()):
                raise ValueError(f"tensor {k!r} contains non-finite values")
            tensor_summary[k] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
            }

    loaded_bundle = j_lens.load_j_lens_bundle(path)
    synth = j_lens.synthesize_from_words("blizzard, snowdrift, freezing", family="fl2va", bundle=loaded_bundle)
    v_pr = synth["vector.post_refiner"]
    if tuple(v_pr.shape) != (cv.POST_REFINER_DIM,):
        raise ValueError(f"synthesize_from_words returned unexpected shape {tuple(v_pr.shape)}")
    if "J_ref_h3_image" in loaded_bundle:
        synth_img = j_lens.synthesize_from_words("blizzard, snowdrift, freezing", family="h3_image", bundle=loaded_bundle)
        if tuple(synth_img["vector.post_refiner"].shape) != (cv.POST_REFINER_DIM,):
            raise ValueError(f"synthesize_from_words(family='h3_image') returned unexpected shape {tuple(synth_img['vector.post_refiner'].shape)}")

    return {
        "path": str(path),
        "format": meta["format"],
        "vocab_words_count": len(words),
        "tensors": tensor_summary,
        "probe_readout_pos": synth["readout"].get("pos_summary", ""),
    }


def _rmsnorm_jacobian(x_2d: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Analytical per-token RMSNorm Jacobian averaged across `N` token rows `x_2d` (`[N, D]`).

    For each row $\\mathbf{x} \\in \\mathbb{R}^D$:
    $$J_{\\text{RMSNorm}}(\\mathbf{x}) = s(\\mathbf{x}) \\operatorname{diag}(\\mathbf{w})\\left(I_D - \\frac{\\hat{\\mathbf{x}}\\hat{\\mathbf{x}}^\\top}{D}\\right)$$
    where $s(\\mathbf{x}) = (\\lVert\\mathbf{x}\\rVert_2^2 / D + \\varepsilon)^{-1/2}$ and $\\hat{\\mathbf{x}} = s(\\mathbf{x})\\mathbf{x}$.
    """
    x = x_2d.to(dtype=torch.float64)
    w = weight.detach().to(device=x.device, dtype=torch.float64)
    d_dim = x.shape[-1]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    s = torch.rsqrt(var + float(eps))
    x_hat = x * s
    mean_s = s.mean(dim=0)  # [1]
    # E[s(x) * x_hat * x_hat^T / D]
    weighted_x_hat = x_hat * torch.sqrt(s)
    outer_mean = (weighted_x_hat.T @ weighted_x_hat) / float(x.shape[0] * d_dim)
    eye = torch.eye(d_dim, dtype=torch.float64, device=x.device)
    j_mean = w.unsqueeze(1) * (mean_s * eye - outer_mean)
    return j_mean.to(dtype=torch.float32)


def compute_refiner_jacobian_and_wiener(
    h49_sequences: list[torch.Tensor],
    sigma_49: torch.Tensor,
    *,
    dit_path: Path,
    loras: list[dict[str, Any]] | None = None,
    beta_attn: float = 1.0,
    wiener_lambda_rel: float = 0.03,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(J_ref, J_wiener)` for an H3 `TokenRefiner` checkpoint + optional LoRA.

    Implements §2.2 and §3.2 of `docs/wiki/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`:
    - Forward passes each `H_49` sequence (`[1, S, 5120]`, keeping token 0 in attention)
      through `condition_proj` and the 2 `TokenRefiner` blocks while accumulating
      analytical Jacobians over non-sink tokens `[:, 1:, :]`.
    - Computes the empirical-covariance Wiener inverse:
      $$J_{\\text{Wiener}} = \\Sigma_{49} \\bar{J}_{\\text{ref}}^\\top (\\bar{J}_{\\text{ref}} \\Sigma_{49} \\bar{J}_{\\text{ref}}^\\top + \\lambda^2 I_{5376})^{-1}$$
    """
    import comfy.ldm.minimax.model as MM
    import comfy.ops

    sig_defaults = {
        k: v.default
        for k, v in inspect.signature(MM.MiniMaxH3Model.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    ops = comfy.ops.disable_weight_init
    dtype = torch.float32
    proj = ops.Linear(
        sig_defaults["text_dim"],
        sig_defaults["hidden_size"],
        bias=True,
        dtype=dtype,
        device=device,
    )
    ref = MM.TokenRefiner(
        sig_defaults["token_refiner_num_layers"],
        sig_defaults["hidden_size"],
        sig_defaults["num_attention_heads"],
        sig_defaults["attention_head_dim"],
        sig_defaults["ffn_hidden_size"],
        sig_defaults["norm_eps"],
        sig_defaults["qk_norm_eps"],
        sig_defaults["final_norm_eps"],
        dtype=dtype,
        device=device,
        operations=ops,
    )
    sd_p: dict[str, torch.Tensor] = {}
    sd_r: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_path), framework="pt", device="cpu") as fh:
        for k in fh.keys():
            clean_k = k
            for prefix in ("model.diffusion_model.", "diffusion_model."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix) :]
                    break
            if clean_k.startswith("condition_proj."):
                sd_p[clean_k[len("condition_proj.") :]] = fh.get_tensor(k).to(device=device, dtype=dtype)
            elif clean_k.startswith("token_refiner."):
                sd_r[clean_k[len("token_refiner.") :]] = fh.get_tensor(k).to(device=device, dtype=dtype)
    proj.load_state_dict(sd_p, strict=True)
    ref.load_state_dict(sd_r, strict=True)
    proj.eval()
    ref.eval()
    if loras:
        cv_nodes._attach_refiner_lora_branches(ref, loras)

    # Collect intermediate activations at each TokenRefiner stage across all calibration sequences
    x0_tok1_list: list[torch.Tensor] = []
    block_inputs: list[list[torch.Tensor]] = [[] for _ in range(len(ref.blocks))]
    post_attn_inputs: list[list[torch.Tensor]] = [[] for _ in range(len(ref.blocks))]
    final_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for h_seq in h49_sequences:
            h_in = h_seq.to(device=device, dtype=dtype)
            if h_in.ndim == 3:
                h_in = h_in.squeeze(0)
            if h_in.shape[0] <= 1:
                continue
            x = proj(h_in)
            x0_tok1_list.append(x[1:, :].detach().cpu())
            for b_idx, blk in enumerate(ref.blocks):
                block_inputs[b_idx].append(x[1:, :].detach().cpu())
                x_attn = x + blk.attn(blk.norm1(x))
                post_attn_inputs[b_idx].append(x_attn[1:, :].detach().cpu())
                x = x_attn + blk.mlp(blk.norm2(x_attn))
            final_inputs.append(x[1:, :].detach().cpu())

    # Linear condition_proj weight [5376, 5120]
    j_cur = proj.weight.detach().to(device=device, dtype=torch.float32).clone()
    norm_eps = float(sig_defaults["norm_eps"])
    final_norm_eps = float(sig_defaults["final_norm_eps"])
    inner_dim = int(sig_defaults["num_attention_heads"]) * int(sig_defaults["attention_head_dim"])

    def _effective_linear_weight(lin_mod: Any, in_features: int, out_features: int) -> torch.Tensor:
        with torch.no_grad():
            eye_in = torch.eye(in_features, dtype=dtype, device=device)
            zero_in = torch.zeros((1, in_features), dtype=dtype, device=device)
            y0 = lin_mod(zero_in)
            y_eye = lin_mod(eye_in) - y0
            return y_eye.T.contiguous()

    for b_idx, blk in enumerate(ref.blocks):
        x_blk = torch.cat(block_inputs[b_idx], dim=0).to(device=device, dtype=dtype)
        j_norm1 = _rmsnorm_jacobian(x_blk, blk.norm1.weight, norm_eps)
        w_qkv = _effective_linear_weight(blk.attn.qkv_proj, cv.POST_REFINER_DIM, inner_dim * 3)
        w_v = w_qkv[2 * inner_dim : 3 * inner_dim, :]
        w_o = _effective_linear_weight(blk.attn.out_proj, inner_dim, cv.POST_REFINER_DIM)
        j_attn = float(beta_attn) * (w_o @ (w_v @ j_norm1))
        j_after_attn = torch.eye(cv.POST_REFINER_DIM, dtype=torch.float32, device=device) + j_attn

        x_pa = torch.cat(post_attn_inputs[b_idx], dim=0).to(device=device, dtype=dtype)
        j_norm2 = _rmsnorm_jacobian(x_pa, blk.norm2.weight, norm_eps)
        with torch.no_grad():
            y_norm2 = blk.norm2(x_pa)
            # SwiGLU / MLP expected Jacobian via exact finite-Free vector-Jacobian or analytical derivative
            w_fc1 = _effective_linear_weight(blk.mlp.fc1, cv.POST_REFINER_DIM, int(sig_defaults["ffn_hidden_size"]) * 2)
            ffn_h = int(sig_defaults["ffn_hidden_size"])
            w_gate = w_fc1[:ffn_h, :]
            w_up = w_fc1[ffn_h:, :]
            g = y_norm2 @ w_gate.T
            u = y_norm2 @ w_up.T
            sig_g = torch.sigmoid(g)
            silu_g = F.silu(g)
            silu_prime_g = sig_g * (1.0 + g * (1.0 - sig_g))
            mean_gate_factor = (silu_prime_g * u).mean(dim=0)
            mean_up_factor = silu_g.mean(dim=0)
            w_fc2 = _effective_linear_weight(blk.mlp.fc2, ffn_h, cv.POST_REFINER_DIM)
            j_ffn = w_fc2 @ (
                mean_gate_factor.unsqueeze(1) * w_gate + mean_up_factor.unsqueeze(1) * w_up
            )
        j_mlp_path = j_ffn @ j_norm2
        j_block = (
            torch.eye(cv.POST_REFINER_DIM, dtype=torch.float32, device=device) + j_mlp_path
        ) @ j_after_attn
        j_cur = j_block @ j_cur

    x_fin = torch.cat(final_inputs, dim=0).to(device=device, dtype=dtype)
    j_final_norm = _rmsnorm_jacobian(x_fin, ref.final_norm.weight, final_norm_eps)
    j_ref = (j_final_norm @ j_cur).detach().to(device="cpu", dtype=torch.float32)

    # Closed-form Wiener inverse:
    # J_Wiener = Sigma_49 @ J_ref^T @ (J_ref @ Sigma_49 @ J_ref^T + lambda^2 I_5376)^{-1}
    j64 = j_ref.to(dtype=torch.float64)
    s64 = sigma_49.to(device="cpu", dtype=torch.float64)
    gram = j64 @ s64 @ j64.T
    eigvals = torch.linalg.eigvalsh(gram)
    sigma_max_sq = float(eigvals.max().clamp(min=1e-12).item())
    lam_sq = (float(wiener_lambda_rel) ** 2) * sigma_max_sq
    reg_gram = gram + lam_sq * torch.eye(cv.POST_REFINER_DIM, dtype=torch.float64)
    j_wiener = torch.linalg.solve(reg_gram, j64 @ s64).T.contiguous().to(dtype=torch.float32)
    return j_ref, j_wiener


def compute_sigma_49_and_p030(
    h49_sequences: list[torch.Tensor],
    *,
    shrinkage_gamma: float = 0.20,
    power: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shrinkage covariance `Sigma_49` (`float32`) and spectral-filtered
    fractional covariance `Sigma_49_p030` (`float16`) from `H_49[:, 1:, :]` rows."""
    rows = torch.cat(
        [seq[0, 1:, :] if seq.ndim == 3 else seq[1:, :] for seq in h49_sequences if (seq.shape[1] if seq.ndim == 3 else seq.shape[0]) > 1],
        dim=0,
    ).to(dtype=torch.float64)
    centered = rows - rows.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / float(max(1, centered.shape[0] - 1))
    d_dim = cov.shape[0]
    trace_mean = float(torch.trace(cov).item()) / float(d_dim)
    gamma = float(shrinkage_gamma)
    sigma_49 = (1.0 - gamma) * cov + gamma * trace_mean * torch.eye(d_dim, dtype=torch.float64)

    # Spectral-filtered fractional covariance: zero top-1 common-mode eigenvalue, raise rest to p=0.30
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pos_vals = eigvals.clamp(min=0.0)
    if pos_vals.numel() >= 2:
        pos_vals[-1] = 0.0  # Zero top-1 common-mode eigenvalue
    pow_vals = pos_vals.pow(float(power))
    max_pow = float(pow_vals.max().clamp(min=1e-12).item())
    pow_vals = pow_vals / max_pow
    sigma_p030 = (eigvecs @ (pow_vals.unsqueeze(1) * eigvecs.T)).to(dtype=torch.float16)
    return sigma_49.to(dtype=torch.float32), sigma_p030


def cmd_build_j_lens(args: argparse.Namespace) -> int:
    """Verify or rebuild `<ComfyUI>/models/j_lens/j_lens_h3.safetensors` (`h3_j_lens_v1`)."""
    out_path = (
        Path(args.out).resolve()
        if getattr(args, "out", None)
        else j_lens.default_bundle_path()
    )
    if getattr(args, "verify_only", False):
        info = verify_j_lens_bundle(out_path)
        print(f"=== J-Lens Operator Bundle Verified: {info['path']} ===")
        print(f"  format           : {info['format']}")
        print(f"  vocab_words      : {info['vocab_words_count']}")
        print(f"  tensors          : {len(info['tensors'])} ({', '.join(sorted(info['tensors'].keys()))})")
        print(f"  probe readout (+): {info['probe_readout_pos']}")
        return 0

    if out_path.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(
            f"J-Lens bundle already exists at {out_path} (pass --overwrite to replace it, or --verify-only to inspect it)"
        )

    existing_path = (
        Path(args.base_bundle).resolve()
        if getattr(args, "base_bundle", None)
        else (out_path if out_path.is_file() else j_lens.default_bundle_path())
    )
    if not existing_path.is_file():
        raise FileNotFoundError(
            f"base bundle not found at {existing_path}; upper-Qwen tensors (J_49_to_63, M_union, cloze_base_union) "
            "require an existing h3_j_lens_v1 bundle when building from the 50-layer truncated ComfyUI encoder"
        )

    t0 = time.perf_counter()
    with safe_open(str(existing_path), framework="pt", device="cpu") as f:
        base_meta = dict(f.metadata() or {})
        preserved_tensors = {
            k: f.get_tensor(k).clone()
            for k in (
                "J_49_to_63",
                "M_union",
                "cloze_base_union",
                "union_idx",
                "valid_idx",
                "valid_pos_in_union",
            )
        }

    bank_json = REPO_ROOT / "prompt_bank" / "bank.json"
    bank_entries = json.loads(bank_json.read_text(encoding="utf-8")).get("prompts", [])
    max_prompts = int(getattr(args, "num_prompts", 24) or 24)
    selected_prompts: list[str] = []
    for entry in bank_entries:
        rel_name = str(entry.get("file") or (f"{entry['id']}.txt" if entry.get("id") else ""))
        p_file = REPO_ROOT / "prompt_bank" / rel_name
        if rel_name and p_file.is_file():
            selected_prompts.append(p_file.read_text(encoding="utf-8").strip())
        if len(selected_prompts) >= max_prompts:
            break
    if not selected_prompts:
        raise ValueError("no calibration prompts found under prompt_bank/")

    encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
    clip = getattr(args, "_clip_override", None) or _h._load_clip(
        encoder_name, device=getattr(args, "device", "default")
    )

    print(f"=== Building J-Lens Operator Bundle ({len(selected_prompts)} calibration prompts) ===")
    h49_sequences: list[torch.Tensor] = []
    max_tokens = int(getattr(args, "max_tokens", 64) or 64)
    for p_txt in selected_prompts:
        enc_rows, _, _, _ = cv_nodes.encode_prompt_for_vector(clip, p_txt)
        h49_sequences.append(enc_rows[:max_tokens].unsqueeze(0).detach().cpu())

    sigma_49, sigma_p030 = compute_sigma_49_and_p030(h49_sequences)

    fl2va_dit = Path(
        _h.folder_paths.get_full_path_or_raise(
            "diffusion_models", getattr(args, "fl2va_dit", None) or h3_config.MODELS["unet_fl2va"]
        )
    )
    ref2va_dit = Path(
        _h.folder_paths.get_full_path_or_raise(
            "diffusion_models", getattr(args, "ref2va_dit", None) or h3_config.MODELS["unet_ref2va"]
        )
    )
    fl2va_loras: list[dict[str, Any]] = []
    ref2va_loras: list[dict[str, Any]] = []
    if not getattr(args, "no_lora", False):
        try:
            fl2va_loras = _parse_lora_specs([f"{h3_config.PIKA_FL2VA_SVD995_LORA}:1.0"])
        except Exception:
            fl2va_loras = []
        try:
            ref2va_loras = _parse_lora_specs([f"{h3_config.PIKA_REF2VA_SVD995_LORA}:1.0"])
        except Exception:
            ref2va_loras = []

    ref_dev = getattr(args, "refiner_device", "cpu") or "cpu"
    j_ref_fl2va, j_wiener_fl2va = compute_refiner_jacobian_and_wiener(
        h49_sequences, sigma_49, dit_path=fl2va_dit, loras=fl2va_loras, device=ref_dev
    )
    j_ref_ref2va, j_wiener_ref2va = compute_refiner_jacobian_and_wiener(
        h49_sequences, sigma_49, dit_path=ref2va_dit, loras=ref2va_loras, device=ref_dev
    )

    out_tensors = {
        **preserved_tensors,
        "Sigma_49_p030": sigma_p030.contiguous(),
        "J_ref_fl2va": j_ref_fl2va.contiguous(),
        "J_ref_ref2va": j_ref_ref2va.contiguous(),
        "J_wiener_fl2va": j_wiener_fl2va.contiguous(),
        "J_wiener_ref2va": j_wiener_ref2va.contiguous(),
    }
    h3_img_name = getattr(args, "h3_image_dit", None) or h3_config.H3_IMAGE_UNET
    h3_img_full = _h.folder_paths.get_full_path("diffusion_models", h3_img_name)
    if h3_img_full and Path(h3_img_full).is_file():
        j_ref_h3_image, j_wiener_h3_image = compute_refiner_jacobian_and_wiener(
            h49_sequences, sigma_49, dit_path=Path(h3_img_full), loras=[], device=ref_dev
        )
        out_tensors["J_ref_h3_image"] = j_ref_h3_image.contiguous()
        out_tensors["J_wiener_h3_image"] = j_wiener_h3_image.contiguous()
    elif getattr(args, "h3_image_dit", None):
        raise FileNotFoundError(
            f"MiniMax-H3-Image DiT checkpoint {h3_img_name!r} not found under diffusion_models/"
        )
    out_meta = {
        "format": "h3_j_lens_v1",
        "description": base_meta.get(
            "description",
            "Precomputed H3 Jacobian Lens (J_49_to_63, TokenRefiner J_ref/J_wiener, 7-prefix cloze baseline logits, and BPE word map)",
        ),
        "vocab_words_json": base_meta["vocab_words_json"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(out_path), metadata=out_meta)
    j_lens._BUNDLE_CACHE.pop(out_path, None)
    info = verify_j_lens_bundle(out_path)
    elapsed = time.perf_counter() - t0
    print(f"=== Saved & Verified J-Lens Bundle in {elapsed:.2f}s: {info['path']} ===")
    return 0
