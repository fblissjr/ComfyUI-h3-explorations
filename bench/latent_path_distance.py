#!/usr/bin/env python3
"""How far saved latents sit from a reference latent, per latent frame. No decode.

Written 2026-09-26 for the PDD schedule tests (`docs/distills/which_distill.md`, "Tests
that would move this section"; predictions P1, P2 and P5 in
`bench/results/2026-09-26_distill_run_predictions.md`). The reference is
usually the base on Euler at 32 steps (`h3_probe_t2v_base_euler32_savelat`),
which shares the distills' starting noise and is the path PDD's heads were
distilled from. The others are PDD at 4, 6, 8, 16 and 32 steps, merged or
exact, at the same seed and prompt.

Per latent frame t, for each latent against the reference:
- `rel_l2[t]`: ||z_t - ref_t|| / ||ref_t||;
- `cos[t]`: the cosine between z_t and ref_t, flattened.
And per latent, with no reference:
- `delta[t]`: ||z_t - z_{t-1}|| / ||z_t||, the latent's own frame-to-frame
  change (0 at t = 0), the latent-space analogue of
  `measure_clip_delta.py`.
Plus, per latent, the Pearson correlation between its `rel_l2` and the
reference's `delta` over t >= 1: whether distance from the reference
concentrates in the frames where the reference moves (P1).

Inputs are `SaveLatent` files of the VIDEO half (`latent_tensor`
[1, 24, T, H, W]). Shapes must match the reference; a mismatch is refused,
because it would mean a different canvas or length, which is a different
scene.

    python bench/latent_path_distance.py REF.latent OTHER.latent [...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch


def load(path: Path) -> torch.Tensor:
    t = safetensors.torch.load_file(str(path))["latent_tensor"].double()
    if t.ndim != 5 or t.shape[0] != 1:
        raise SystemExit(f"{path.name}: expected a video latent [1, C, T, H, W], got {tuple(t.shape)}")
    return t[0]                                    # [C, T, H, W]


def per_frame(z: torch.Tensor) -> torch.Tensor:
    return z.permute(1, 0, 2, 3).reshape(z.shape[1], -1)   # [T, C*H*W]


def delta(z: torch.Tensor) -> list[float]:
    f = per_frame(z)
    out = [0.0]
    for t in range(1, f.shape[0]):
        out.append(float((f[t] - f[t - 1]).norm() / f[t].norm().clamp(min=1e-12)))
    return out


def against(z: torch.Tensor, ref: torch.Tensor) -> dict:
    a, b = per_frame(z), per_frame(ref)
    rel = ((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-12)).tolist()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1).tolist()
    return {"rel_l2": rel, "cos": cos,
            "rel_l2_whole": float((z - ref).norm() / ref.norm()),
            "cos_whole": float(torch.nn.functional.cosine_similarity(z.flatten(), ref.flatten(), dim=0))}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    xa, ya = np.asarray(x), np.asarray(y)
    if xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("others", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    ref = load(args.reference)
    ref_delta = delta(ref)
    out = {"measured_by": "bench/latent_path_distance.py", "reference": args.reference.name,
           "shape": list(ref.shape), "reference_delta": ref_delta, "latents": {}}
    print(f"reference {args.reference.name}  shape {tuple(ref.shape)}")
    print(f"{'latent':<64} {'rel_l2':>8} {'cos':>8} {'r(dist,ref delta)':>18}")
    for p in args.others:
        z = load(p)
        if z.shape != ref.shape:
            raise SystemExit(f"{p.name}: shape {tuple(z.shape)} against the reference's {tuple(ref.shape)}")
        row = against(z, ref)
        row["delta"] = delta(z)
        row["r_dist_vs_ref_delta"] = pearson(row["rel_l2"][1:], ref_delta[1:])
        out["latents"][p.name] = row
        r = row["r_dist_vs_ref_delta"]
        print(f"{p.name:<64} {row['rel_l2_whole']:>8.4f} {row['cos_whole']:>8.4f} "
              f"{'-' if r is None else f'{r:+.3f}':>18}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
