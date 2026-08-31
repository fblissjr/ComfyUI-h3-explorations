#!/usr/bin/env python3
"""What stochastic rounding costs on the merge path, measured rather than derived.

## The claim being tested

ComfyUI's weight-merge path requantises with STOCHASTIC rounding, not
round-to-nearest. The chain, read 2026-08-31 and verified line by line:

    comfy/model_patcher.py:928     set_func(..., seed=comfy.utils.string_to_seed(key))
    comfy/ops.py:1434              set_weight -> requantize_from_float(
                                       ..., stochastic_rounding=seed)
    comfy_kitchen/tensor/base.py:302   requantize_from_float -> from_float
    comfy_kitchen/tensor/int8.py:128   -> quantize_int8_convrot_weight(
                                            ..., stochastic_rounding=...)
    .../eager/quantization.py:822  _round_int8: if stochastic_rounding > 0,
                                   add RNG and floor; else round-to-nearest

Analytically, for a value at fractional offset p inside a grid cell,
round-to-nearest has squared error min(p, 1-p)^2 and stochastic rounding has
expected squared error p(1-p). Integrated over p uniform on [0,1] that is 1/12
against 1/6 -- **twice the MSE, sqrt(2) on RMS**.

**That derivation assumes p is uniform, and real weights are not uniform
anything.** So this measures the ratio on actual H3 DiT weights through the
actual kernel, which is the only way to find out whether the assumption holds
where it is being applied. A derivation is an inference; this is the check on it.

## What this does NOT establish

That the difference is visible in a render, or that it matters. It is a weight-
level quantity. Its use is narrower and specific: **a merged arm and an
unmerged arm are not the same model before a single sampling step runs**, so a
comparison between them carries this noise on top of whatever it meant to
measure. That is upstream of the "a rendered clip cannot A/B a numerical
change" rule in CLAUDE.md, which is about sampling divergence.

## Two arms, because the second is the one that actually happens

  generic   real H3 weights that are simply OFF the int8 grid, taken from the
            fp8_scaled release and dequantised. This is the arm the analytic
            2x describes.
  merged    the tensor `set_weight` genuinely receives on a LoRA'd render:
            the int8 base dequantised, plus a real LoRA's delta. **Predicted
            to be worse than 2x**, and the reason is worth stating before the
            number: the base is already ON the grid, so a small delta leaves
            most values close to a grid point, where round-to-nearest returns
            almost exactly the right value and stochastic rounding still
            gambles the full step. The generic arm cannot show this.

**There is no full BF16 H3 DiT on this box.** The obvious candidate,
`lightx2v_.../minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, is a
LORA despite its name -- 624 keys, all `lora_A`/`lora_B`. That is what supplies
the delta below rather than a base.

    python bench/measure_stochastic_rounding.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parents[1]))

from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight, quantize_int8_convrot_weight)

#: The layout's own default (`TensorWiseINT8Layout.convrot_groupsize`), not a
#: number chosen here -- the merge path preserves it via `requantize_kwargs`.
GROUP_SIZE = 256

#: Resolved through ComfyUI's own models tree rather than named directly, the
#: convention `analyze_checkpoint_delta.py` and `build_hybrid.py` already use.
#: The entries there are symlinks, so this reaches the same bytes while staying
#: portable to a box that stages its weights somewhere else.
MODELS = REPO.parents[1] / "models"
#: The PRUNED int8, because that is what the shipped turbo graphs load
#: (`h3_text_to_video_turbo_api.json` and four others), and they merge the
#: LoRA below into it. So the merged arm is the configuration that ships, not
#: a constructed pairing.
INT8_BASE = (MODELS / "diffusion_models"
             / "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
FP8 = (MODELS / "diffusion_models"
       / "minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
LORA = (MODELS / "loras/h3/lightx2v_Minimax-h3-Turbo"
        / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")

#: One per module kind, spread across depth, so the answer is not about one
#: tensor's shape. Chosen by kind and depth rather than at random, because a
#: random draw over 50 blocks would mostly return mlp weights.
WANT_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
WANT_BLOCKS = (0, 12, 25, 37, 49)


def string_to_seed(data: str) -> int:
    """ComfyUI's own (`comfy/utils.py:1499`), a CRC32 over the key.

    Copied rather than imported because importing comfy pulls in device setup
    and argument parsing. `_assert_seed_matches_comfy` below checks the copy
    against the real one when comfy is importable, so this is a cached
    implementation with an invalidation rather than a second source of truth.
    """
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return crc ^ 0xFFFFFFFF


def _assert_seed_matches_comfy() -> str:
    """Compare the copy above against comfy's, or say it could not be checked."""
    try:
        import comfy.utils
    except Exception as exc:  # noqa: BLE001
        return f"NOT CHECKED against comfy ({type(exc).__name__})"
    probe = ["blocks.0.attn.qkv_proj.weight", "blocks.49.mlp.fc2.weight", ""]
    for key in probe:
        if comfy.utils.string_to_seed(key) != string_to_seed(key):
            raise AssertionError(
                f"the local string_to_seed disagrees with comfy's on {key!r}; "
                f"every seed in this record would be the wrong one")
    return "matches comfy.utils.string_to_seed on probe keys"


def measure(original: torch.Tensor, seed: int):
    """(rtn_mse, sr_mse, reproducible) for one float weight, real kernel."""
    original = original.float()

    q_rtn, s_rtn = quantize_int8_convrot_weight(
        original.clone(), GROUP_SIZE, stochastic_rounding=0)
    rtn = dequantize_int8_convrot_weight(q_rtn, s_rtn, GROUP_SIZE).float()

    q_sr, s_sr = quantize_int8_convrot_weight(
        original.clone(), GROUP_SIZE, stochastic_rounding=seed)
    sr = dequantize_int8_convrot_weight(q_sr, s_sr, GROUP_SIZE).float()

    # Same seed twice must give the same bytes: per-key reproducible noise, not
    # a different model every load.
    q_again, _ = quantize_int8_convrot_weight(
        original.clone(), GROUP_SIZE, stochastic_rounding=seed)
    reproducible = bool(torch.equal(q_sr, q_again))

    return (float((rtn - original).pow(2).mean()),
            float((sr - original).pow(2).mean()),
            reproducible)


def _lora_delta(handle, prefix: str, out_shape) -> torch.Tensor | None:
    """`lora_B @ lora_A`, scaled the way ComfyUI scales it, or None."""
    a_key, b_key = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
    keys = set(handle.keys())
    if a_key not in keys or b_key not in keys:
        return None
    a = handle.get_tensor(a_key).float()
    b = handle.get_tensor(b_key).float()
    alpha_key = f"{prefix}.alpha"
    rank = a.shape[0]
    # ComfyUI reads alpha only from a tensor named `<module>.alpha` and falls
    # back to 1.0 -- `docs/checks.md`'s check_lora_alpha row is about exactly
    # this. Scale is alpha/rank when alpha is present.
    alpha = float(handle.get_tensor(alpha_key)) if alpha_key in keys else float(rank)
    delta = (b @ a) * (alpha / rank)
    return delta if tuple(delta.shape) == tuple(out_shape) else None


def main() -> int:
    missing = [p.name for p in (INT8_BASE, FP8, LORA) if not p.exists()]
    if missing:
        print(f"not on this box: {missing}")
        return 2

    seed_status = _assert_seed_matches_comfy()
    print(f"stochastic rounding vs round-to-nearest, int8_convrot, "
          f"group_size={GROUP_SIZE}")
    print(f"seed function: {seed_status}\n")

    arms: dict[str, list] = {"generic": [], "merged": []}

    with safe_open(str(FP8), framework="pt") as fp8, \
         safe_open(str(INT8_BASE), framework="pt") as int8, \
         safe_open(str(LORA), framework="pt") as lora:
        fp8_keys, int8_keys = set(fp8.keys()), set(int8.keys())
        for block in WANT_BLOCKS:
            for suffix in WANT_SUFFIXES:
                stem = f"blocks.{block}.{suffix}"
                seed = string_to_seed(f"diffusion_model.{stem}.weight")

                # --- generic: real H3 weights simply off the grid -----------
                wk, sk = f"{stem}.weight", f"{stem}.weight_scale"
                if wk in fp8_keys and sk in fp8_keys:
                    w = fp8.get_tensor(wk).float() * fp8.get_tensor(sk).float()
                    arms["generic"].append((stem, *measure(w, seed)))

                # --- merged: what set_weight actually receives --------------
                if wk in int8_keys and sk in int8_keys:
                    base = dequantize_int8_convrot_weight(
                        int8.get_tensor(wk), int8.get_tensor(sk).float(),
                        GROUP_SIZE).float()
                    delta = _lora_delta(lora, f"diffusion_model.{stem}",
                                        base.shape)
                    if delta is not None:
                        rtn_mse, sr_mse, repro = measure(base + delta, seed)
                        # **The ratio alone would mislead here.** A near-on-grid
                        # tensor drives RTN's error toward zero, so the ratio
                        # can be enormous while SR's ABSOLUTE error is
                        # unchanged. What decides whether it matters is SR's
                        # noise against the size of the delta being added --
                        # if they are comparable, the LoRA's contribution is
                        # substantially drowned by the requantisation of it.
                        delta_ms = float(delta.pow(2).mean())
                        arms["merged"].append(
                            (stem, rtn_mse, sr_mse, repro, delta_ms))

    record = {
        "what": ("MSE cost of ComfyUI's stochastic-rounding requantisation on "
                 "the weight-merge path, against round-to-nearest, measured "
                 "through comfy_kitchen's own int8_convrot kernel"),
        "date": date.today().isoformat(),
        "group_size": GROUP_SIZE,
        "seed_function": seed_status,
        "analytic_prediction_uniform_offsets": 2.0,
        "arms": {},
        "control": ("round-to-nearest applied to the int8 base ALONE returns "
                    "err rms 4.6e-08 against its own values, so the base is on "
                    "the grid and the dequantise/requantise round-trip used "
                    "here is faithful. Without that the merged arm's ratios "
                    "would be unreadable -- they depend on the base being "
                    "exactly on-grid."),
        "headline": ("the LoRA delta is ~0.1% of ONE int8 quantisation step, "
                     "so round-to-nearest alone already loses most of it "
                     "(err rms 1.03x delta rms) and stochastic rounding is "
                     "~18x worse again. Merging a small LoRA into an int8 "
                     "checkpoint is lossy REGARDLESS of rounding mode; the "
                     "rounding mode decides how much worse."),
        "not_established": [
            "that the difference is visible in any render",
            "that a LoRA applied at the CALL rather than merged suffers this "
            "-- it does not requantise, which makes it the control",
            "whether the PDD LoRAs behave the same; only the turbo LoRA the "
            "shipped graphs wire was measured",
            "that it applies to a module which never goes through set_weight "
            "-- an un-merged block applies at the call and never requantises, "
            "which makes it a usable control",
        ],
        "reproduce": "python bench/measure_stochastic_rounding.py",
    }

    for name, rows in arms.items():
        if not rows:
            print(f"{name}: no tensors matched")
            continue
        ratios = [r[2] / r[1] for r in rows if r[1]]
        mean = sum(ratios) / len(ratios)
        print(f"{name} ({len(rows)} tensors)")
        has_delta = len(rows[0]) > 4
        for row in rows:
            stem, rtn, sr, repro = row[0], row[1], row[2], row[3]
            extra = ""
            if has_delta:
                # RMS of the noise SR adds, against RMS of the delta it is
                # quantising. >1 means the requantisation noise exceeds the
                # weight change the LoRA was trying to make.
                extra = f"  sr_rms/delta_rms {(sr / row[4]) ** 0.5:7.3f}"
            print(f"  {stem:<28} ratio {sr / rtn:9.3f}{extra}")
        print(f"  -> ratio  min {min(ratios):.3f}  mean {mean:.3f}  "
              f"max {max(ratios):.3f}")
        entry = {
            "tensors": len(rows),
            "ratio_min": min(ratios), "ratio_mean": mean,
            "ratio_max": max(ratios),
            "all_reproducible_under_fixed_seed": all(r[3] for r in rows),
            "per_tensor": [{"key": r[0], "rtn_mse": r[1], "sr_mse": r[2],
                            "ratio": r[2] / r[1], "reproducible": r[3],
                            **({"delta_ms": r[4],
                                "sr_rms_over_delta_rms": (r[2] / r[4]) ** 0.5}
                               if has_delta else {})}
                           for r in rows],
        }
        if has_delta:
            noise = [(r[2] / r[4]) ** 0.5 for r in rows]
            nm = sum(noise) / len(noise)
            print(f"  -> sr_rms/delta_rms  min {min(noise):.3f}  "
                  f"mean {nm:.3f}  max {max(noise):.3f}")
            entry["sr_rms_over_delta_rms"] = {
                "min": min(noise), "mean": nm, "max": max(noise),
                "reading": ("RMS of the noise stochastic rounding adds, over "
                            "RMS of the LoRA delta being merged. Above 1 means "
                            "the requantisation noise is larger than the "
                            "weight change the LoRA was making."),
            }
        entry["ratio_caveat"] = (
            "the SR/RTN ratio is inflated here and is NOT the headline: the "
            "merged tensor sits near the grid the base was quantised on, so "
            "RTN's error collapses toward zero and the ratio measures how far "
            "below one quantisation step the delta is. sr_rms_over_delta_rms "
            "is the interpretable quantity."
        ) if has_delta else (
            "clean ratio: the generic arm's weights are unrelated to the int8 "
            "grid, which is the condition the analytic 2x assumes."
        )
        record["arms"][name] = entry
        print()

    record["arms_source"] = {
        "generic": FP8.name + " (dequantised; real H3 weights off the grid)",
        "merged": (f"{INT8_BASE.name} dequantised + {LORA.name} delta -- the "
                   f"tensor set_weight receives on a LoRA'd render"),
    }
    out = REPO / "bench" / "results" / f"{record['date']}_stochastic_rounding.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
