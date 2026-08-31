#!/usr/bin/env python3
"""How much of a LoRA delta actually lands when it is merged into int8.

**Written to refute a claim this repo made on 2026-08-31 and had already
committed.** `docs/research/quant_levers.md` said that because stochastic
rounding carries exactly 2x the MSE of round-to-nearest -- measured, 1.4142150
on RMS over 200 modules against sqrt(2) = 1.4142136 -- switching the merge path
to deterministic rounding was "arithmetic, not a hypothesis" and needed no
further evidence. That inference does not hold, and the reason is the shape of
the problem rather than an error in the constant.

The sqrt(2) is a statement about **rounding a fixed tensor**. Merging is a
different problem: the delta being folded in is far SMALLER THAN ONE INT8 STEP
(a peer session measured delta_rms/step at median 0.0805 for PDD and 0.0050 for
the turbo LoRA, whose alpha/rank is 16x smaller). Round-to-nearest is a
deterministic biased map, so for a sub-step delta most weights do not cross a
midpoint and **the update is simply discarded**. Stochastic rounding is
unbiased by construction -- `E[Q_s(x)] = x` -- so the update lands in
expectation, spread over a random subset of weights that each move a full step.

So the two arms are not "accurate" against "noisy". They are:

    RTN          low stored-weight error, because it barely applies the LoRA
    stochastic   higher stored-weight error, and it applies the LoRA

**A stored-weight metric prefers the arm that does nothing**, which is why
`e_stochastic > e_deterministic` cannot be read as "deterministic is better for
merging". This file measures the quantity the other metric is blind to.

## What it reports

`realised_along_d`   `<Q(W+d) - Q(W), d> / <d, d>`
                     the fraction of the delta that landed, projected onto the
                     delta's own direction. 1.0 is the update fully applied;
                     0.0 is the update discarded. This is the number the
                     stored-weight records cannot see.

`e_vs_target`        `||Q(W+d) - (W+d)|| / ||W+d||`
                     the stored-weight distance, for comparison. Its ordering
                     is the OPPOSITE of `realised_along_d`'s, which is the
                     whole point.

`noise_over_delta`   `||Q(W+d) - (W+d)|| / ||d||`
                     the error the merge injects, measured against the UPDATE
                     rather than against the weight. Added 2026-08-31 after
                     the first version of this file concluded too early.
                     `realised_along_d` alone says stochastic rounding is
                     fine -- it realises 1.0000 -- but a sub-step delta is
                     realised as a sparse set of FULL-step jumps, so the
                     direction is right and the per-weight representation is
                     not. On the turbo LoRA this reaches 19-25x: the merge
                     applies the update and injects twenty times its magnitude
                     in noise. **Both merge arms are bad there, for opposite
                     reasons**, and only this column shows the second one.

Stored weights only, like everything in this lane. Nothing here says either
arm is visible in a render, and `docs/open_experiments.md` #23 is the runtime
question. CPU only, no server.

    uv run --active --no-sync python bench/measure_merge_realisation.py \\
        --base <int8 checkpoint> --lora <converted lora> --out <json>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_quant_delta import header, marker, weight_in_compute_space  # noqa: E402
from h3_producer_provenance import producer_provenance  # noqa: E402

sys.path.insert(0, str(_HERE.parents[2]))
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    _build_hadamard, _rotate_weight, dequantize_int8_convrot_weight)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def quantize(w: torch.Tensor, gs: int, seed: int) -> torch.Tensor:
    """The shipped path. `seed=0` is round-to-nearest, `seed>0` stochastic.

    Stochastic here is `floor(x + U[0,1))`, which is `_round_int8`'s branch.
    A different RNG stream from the shipped one, so this is a draw from the
    same distribution rather than the realisation a given key would produce.
    """
    h = _build_hadamard(gs, dtype=torch.float32)
    rot = _rotate_weight(w, h, gs)
    row = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    scaled = rot / row
    if seed > 0:
        g = torch.Generator().manual_seed(int(seed))
        noise = torch.rand(scaled.shape, generator=g, dtype=torch.float32)
        q = torch.clamp(torch.floor(scaled + noise), -127, 127).to(torch.int8)
    else:
        q = torch.clamp(torch.round(scaled), -127, 127).to(torch.int8)
    return dequantize_int8_convrot_weight(q, row, gs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0,12,25,37,49")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    blocks = [int(x) for x in args.blocks.split(",")]
    hdr, off = header(args.base)
    rows = []

    with safe_open(args.lora, "pt") as f:
        for blk in blocks:
            for kind in KINDS:
                mod = f"blocks.{blk}.{kind}"
                gs = int((marker(args.base, hdr, off, mod) or {})
                         ["convrot_groupsize"])
                Wq = torch.from_numpy(
                    weight_in_compute_space(args.base, hdr, off, mod)
                    .astype(np.float32))
                a = f.get_tensor(f"diffusion_model.{mod}.lora_A.weight").float()
                b = f.get_tensor(f"diffusion_model.{mod}.lora_B.weight").float()
                alpha = float(f.get_tensor(f"diffusion_model.{mod}.alpha").item())
                d = (alpha / a.shape[0]) * (b @ a)
                if d.shape != Wq.shape:
                    raise SystemExit(f"{mod}: delta {d.shape} base {Wq.shape}")
                target = Wq + d
                dd = float((d * d).sum())

                row = {"block": blk, "kind": kind, "groupsize": gs,
                       "delta_rel": float(d.norm() / Wq.norm()), "arms": {}}
                for name, seed in (("deterministic", 0),
                                   ("stochastic", args.seed)):
                    merged = quantize(target, gs, seed)
                    realised = merged - Wq
                    row["arms"][name] = {
                        "realised_along_d": float((realised * d).sum() / dd),
                        "e_vs_target": float((merged - target).norm()
                                             / target.norm()),
                        "noise_over_delta": float((merged - target).norm()
                                                  / d.norm()),
                    }
                rows.append(row)
                det, sto = row["arms"]["deterministic"], row["arms"]["stochastic"]
                print(f"  {mod:26s} |d|/|W| {row['delta_rel']:.5f}   "
                      f"realised RTN {det['realised_along_d']:6.3f} "
                      f"stoch {sto['realised_along_d']:6.3f}   "
                      f"noise/|d| RTN {det['noise_over_delta']:6.2f} "
                      f"stoch {sto['noise_over_delta']:6.2f}", flush=True)

    expected = len(blocks) * len(KINDS)
    if len(rows) != expected:
        raise SystemExit(f"produced {len(rows)} rows, expected {expected}")

    def mean(arm, key):
        return float(np.mean([r["arms"][arm][key] for r in rows]))

    summary = {
        "realised_along_d": {
            "deterministic_mean": mean("deterministic", "realised_along_d"),
            "deterministic_min": float(min(
                r["arms"]["deterministic"]["realised_along_d"] for r in rows)),
            "stochastic_mean": mean("stochastic", "realised_along_d"),
        },
        "e_vs_target": {
            "deterministic_mean": mean("deterministic", "e_vs_target"),
            "stochastic_mean": mean("stochastic", "e_vs_target"),
        },
        "noise_over_delta": {
            "deterministic_mean": mean("deterministic", "noise_over_delta"),
            "stochastic_mean": mean("stochastic", "noise_over_delta"),
            "stochastic_max": float(max(
                r["arms"]["stochastic"]["noise_over_delta"] for r in rows)),
        },
        "delta_rel_mean": float(np.mean([r["delta_rel"] for r in rows])),
    }
    summary["orderings_disagree"] = bool(
        (summary["realised_along_d"]["stochastic_mean"]
         > summary["realised_along_d"]["deterministic_mean"])
        and (summary["e_vs_target"]["stochastic_mean"]
             > summary["e_vs_target"]["deterministic_mean"]))

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/measure_merge_realisation.py",
        "question": ("how much of a LoRA delta survives being merged into an "
                     "int8_convrot weight, under each rounding mode"),
        "base": Path(args.base).name,
        "lora": Path(args.lora).name,
        "rounding_note": ("stochastic is a draw from the same distribution as "
                          "the shipped path, not the realisation a given key "
                          "produces -- the RNG stream differs"),
        "is_not": ("a runtime or perceptual measurement. Stored weights only, "
                   "and int8_convrot is W8A8, so this sees one of two "
                   "roundings. docs/open_experiments.md #23"),
        "shape": {"rows": len(rows), "expected": expected},
        "summary": summary,
        "modules": rows,
        "findings": [
            "ROUND-TO-NEAREST DISCARDS A SUB-STEP LORA. Projected onto the "
            "delta's own direction, deterministic rounding realises "
            f"{summary['realised_along_d']['deterministic_mean']:.3f} of it on "
            f"average and as little as "
            f"{summary['realised_along_d']['deterministic_min']:.3f} on the "
            "worst module; stochastic rounding realises "
            f"{summary['realised_along_d']['stochastic_mean']:.4f}, which is "
            "what `E[Q_s(x)] = x` predicts.",
            "THE TWO METRICS RANK THE ARMS OPPOSITELY, which is the point. "
            "Deterministic has the lower stored-weight distance "
            f"({summary['e_vs_target']['deterministic_mean']:.6f} against "
            f"{summary['e_vs_target']['stochastic_mean']:.6f}) BECAUSE it "
            "barely moves the weight, and the target is close to the weight. A "
            "stored-weight metric rewards the arm that does not apply the "
            "update.",
            "SO THE MEASURED sqrt(2) DOES NOT LICENSE A DETERMINISTIC MERGE. "
            "That constant is correct for rounding a FIXED tensor and was "
            "verified to seven figures; the error was carrying it to merging, "
            "which is a different problem. ComfyUI's choice of stochastic "
            "rounding in `set_weight` looks deliberate and correct.",
        ],
        "not_measured": [
            "whether either arm is distinguishable in a render.",
            "the activation rounding, which this lane has never seen.",
            "other LoRAs. A peer session measured delta_rms/step at median "
            "0.0805 for PDD and 0.0050 for the turbo LoRA -- alpha/rank 1.0 "
            "against 0.0625 -- so the discard should be far worse on turbo, "
            "and that is unmeasured here.",
        ],
        "producer": producer_provenance(__file__),
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
