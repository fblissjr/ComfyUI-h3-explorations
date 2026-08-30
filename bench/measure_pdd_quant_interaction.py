"""Does the PDD LoRA's strength change how much QUANTISATION costs?

The premise this exists to test: *scale PDD down in quantisation-sensitive
blocks, so the two errors do not compound.* That premise needs the two errors
to interact. On an `int8_convrot` checkpoint they plausibly do, because
patching a quantised module is not `W_int8 + s*dW` --
`ModelPatcher.patch_weight_to_device` dequantises (`convert_weight`), adds the
patch in float, and hands the result to `set_weight`, which calls
`requantize_from_float(..., scale="recalculate")`. So the LoRA changes the
values being quantised, and can move the per-row amax that sets the step.

Measured per module, against the bf16 release as truth:

  e_shipped   ||deq(int8) - W_ref|| / ||W_ref||
              what a render with NO LoRA carries. The 2026-08-21 number.
  e_patched(s)||deq(requant(deq(int8) + s*dW)) - (W_ref + s*dW)||
                                              / ||W_ref + s*dW||
              what a render at strength `s` carries: the distance from the
              weight that runs to the weight an unquantised run would use.

**Red proof, and it is why `e_vs_unpatched` is here.** A column that is flat
in `s` is only a result if the harness could have seen a slope.
`e_vs_unpatched(s)` compares the same actual weight against the UNPATCHED
target, so it must RISE with `s` by construction -- it is measuring the LoRA
itself. If that column rises while `e_patched` stays flat, the flatness is
measured rather than an insensitivity of the instrument.

CPU only, no server. A sample of modules across depth, not all 200.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_checkpoint_delta import header  # noqa: E402
from analyze_quant_delta import (  # noqa: E402
    Reference, hf_to_comfy, head_dim, marker, stats, weight_in_compute_space)

sys.path.insert(0, str(_HERE.parents[2]))
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    _build_hadamard, _rotate_weight, dequantize_int8_convrot_weight)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def requantize(w: np.ndarray, gs: int) -> np.ndarray:
    """The shipped int8_convrot path, applied to a float weight.

    Mirrors `analyze_quant_delta.format_floor`'s int8 branch, which is the
    rotate / per-output-row amax / round / un-rotate sequence
    `TensorWiseINT8Layout.quantize(convrot=True, per_channel=True)` runs.
    Deterministic rounding: `set_weight` passes a stochastic-rounding seed, so
    this is the expectation of what ships rather than one draw of it.
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    h = _build_hadamard(gs, dtype=torch.float32)
    rot = _rotate_weight(t, h, gs)
    row = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.clamp(torch.round(rot / row), -127, 127).to(torch.int8)
    return dequantize_int8_convrot_weight(q, row, gs).numpy()


def lora_delta(f, prefix: str) -> np.ndarray:
    a = f.get_tensor(prefix + ".lora_A.weight").to(torch.float32)
    b = f.get_tensor(prefix + ".lora_B.weight").to(torch.float32)
    alpha = float(f.get_tensor(prefix + ".alpha").item())
    return ((alpha / a.shape[0]) * (b @ a)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="fl2va")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0,7,16,24,32,40,49")
    ap.add_argument("--strengths", default="0.0,0.25,0.5,1.0")
    args = ap.parse_args()

    blocks = [int(x) for x in args.blocks.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    hdr, off = header(args.base)
    ref = Reference(args.reference)
    hd = head_dim(hdr)

    rows = []
    with safe_open(args.lora, "pt") as f:
        for blk in blocks:
            for kind in KINDS:
                mod = f"blocks.{blk}.{kind}"
                gs = int((marker(args.base, hdr, off, mod) or {})
                         ["convrot_groupsize"])
                w_q = weight_in_compute_space(args.base, hdr, off, mod)
                w_ref = hf_to_comfy(mod + ".weight",
                                    ref.get(mod + ".weight"), hd)
                d = lora_delta(f, f"diffusion_model.{mod}")
                if not (w_q.shape == w_ref.shape == d.shape):
                    raise SystemExit(
                        f"{mod}: int8 {w_q.shape} ref {w_ref.shape} "
                        f"lora {d.shape}")

                row = {
                    "block": blk, "kind": kind, "groupsize": gs,
                    "pdd_rel": float(np.linalg.norm(d.astype(np.float64))
                                     / np.linalg.norm(w_ref.astype(np.float64))),
                    "e_shipped": stats(w_ref, w_q)["rel_delta"],
                    "by_strength": {},
                }
                for s in strengths:
                    actual = requantize(w_q + s * d, gs)
                    target = w_ref + s * d
                    row["by_strength"][f"{s}"] = {
                        "e_patched": stats(target, actual)["rel_delta"],
                        # the red proof: same actual, unpatched target
                        "e_vs_unpatched": stats(w_ref, actual)["rel_delta"],
                    }
                rows.append(row)
                print(f"  {mod:28s} shipped {row['e_shipped']:.6f}  "
                      + "  ".join(
                          f"s={s}:{row['by_strength'][str(s)]['e_patched']:.6f}"
                          for s in strengths), flush=True)

    def col(key: str, s: float) -> list[float]:
        return [r["by_strength"][str(s)][key] for r in rows]

    summary = {
        "e_shipped_mean": float(np.mean([r["e_shipped"] for r in rows])),
        "by_strength": {
            f"{s}": {
                "e_patched_mean": float(np.mean(col("e_patched", s))),
                "e_patched_max": float(np.max(col("e_patched", s))),
                "e_vs_unpatched_mean": float(np.mean(col("e_vs_unpatched", s))),
            } for s in strengths},
    }
    lo, hi = strengths[0], strengths[-1]
    summary["e_patched_change_over_strength_range"] = (
        summary["by_strength"][f"{hi}"]["e_patched_mean"]
        / summary["by_strength"][f"{lo}"]["e_patched_mean"])
    summary["red_proof_e_vs_unpatched_change"] = (
        summary["by_strength"][f"{hi}"]["e_vs_unpatched_mean"]
        / summary["by_strength"][f"{lo}"]["e_vs_unpatched_mean"])

    out = {
        "measured": "2026-08-30",
        "produced_by": "bench/measure_pdd_quant_interaction.py",
        "question": ("does the PDD LoRA's strength change the quantisation "
                     "error a module carries, which is what a per-block "
                     "strength schedule would have to exploit"),
        "checkpoint": args.checkpoint,
        "lora": Path(args.lora).name,
        "base": Path(args.base).name,
        "reference": str(args.reference.name),
        "rounding": ("deterministic; the shipped path uses seeded stochastic "
                     "rounding, so these are its expectation"),
        "is_not": ("an activation or output measurement. Stored-weight "
                   "distance only"),
        "summary": summary,
        "modules": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
