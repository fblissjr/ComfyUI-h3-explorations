#!/usr/bin/env python3
"""How low-rank is FastH3's backbone change, measured on the bf16 sources?

Written 2026-09-29 for the fasth3-adapter direction. The int8 extraction
(`bench/measure_checkpoint_lora_rank.py`, 2026-09-26) failed: the int8 files
share their row scales' grid, so the difference is +-1 code steps and says
little about the true change. The bf16 files are the true sources
(`2026-09-26_fasth3_weights.md`, finding 1), so this measures
`dW = W_fasth3 - W_fl2va` there, per linear, straight from bf16.

Per module it reports:

- `rel_delta`: ||dW|| / ||W_base||;
- `frac_changed`: the share of elements where the two bf16 values differ (a
  delta under bf16's spacing shows up as sparse +-1-ulp noise, which has no
  low-rank structure whatever the training did);
- `energy@r`: the share of ||dW||^2 in the top r singular values;
- `rank@p`: the smallest r holding p of the energy, and it as a share of
  min(shape).

The spectrum comes from the eigenvalues of the Gram matrix on the smaller
side (sigma^2 = eig), in float64 for the eigensolve.

The control (`--control`): an iid Gaussian matrix of the same shape scaled to
the same norm, through the same code. A flat spectrum is what "no structure"
reads as here; a module is low-rank only if it sits well above this.

    python bench/measure_bf16_delta_rank.py [--blocks 0,12,24,37,49] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
STORAGE = Path.home() / "Storage"
# Provenance: paths inherited from 2026-09-26_fasth3_weights.md's inputs.
FASTH3 = STORAGE / "FastVideo-FastH3-Comfy/diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors"
FL2VA = STORAGE / "Comfy-Org_MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
RANKS = (16, 64, 128, 256, 512, 1024)
# Provenance: reasoned. 50/90/99 read as "the bulk", "nearly all", "all but noise".
LEVELS = (0.5, 0.9, 0.99)


def spectrum_energy(d: torch.Tensor) -> torch.Tensor:
    """Squared singular values of `d`, descending, via the smaller Gram matrix."""
    d = d.float()
    g = d.T @ d if d.shape[0] >= d.shape[1] else d @ d.T
    ev = torch.linalg.eigvalsh(g.double()).clamp_min(0)
    return ev.flip(0)


def summarize(e: torch.Tensor) -> dict:
    tot = float(e.sum())
    n = e.numel()
    row = {}
    if tot <= 0:
        return row
    cum = torch.cumsum(e, 0) / tot
    for r in RANKS:
        if r <= n:
            row[f"energy@{r}"] = round(float(cum[r - 1]), 4)
    for p in LEVELS:
        r = int((cum < p).sum().item()) + 1
        row[f"rank@{int(p * 100)}"] = r
        row[f"rank@{int(p * 100)}_share"] = round(r / n, 4)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=FL2VA)
    ap.add_argument("--other", type=Path, default=FASTH3)
    ap.add_argument("--blocks", default="0,6,12,18,24,30,37,43,49")
    ap.add_argument("--control-blocks", default="24", help="blocks that also get the iid-noise control")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    ctrl_blocks = set(args.control_blocks.split(",")) if args.control_blocks else set()

    rows, controls = {}, {}
    g = torch.Generator().manual_seed(0)  # provenance: reasoned, any fixed seed
    with safe_open(str(args.base), "pt") as a, safe_open(str(args.other), "pt") as b:
        for blk in args.blocks.split(","):
            for kind in KINDS:
                key = f"blocks.{blk}.{kind}.weight"
                t0 = time.time()
                wa, wb = a.get_tensor(key), b.get_tensor(key)
                d = wb.float() - wa.float()
                row = {
                    "shape": list(d.shape),
                    "rel_delta": round(float(d.norm() / wa.float().norm()), 6),
                    "frac_changed": round(float((wa != wb).float().mean()), 4),
                }
                row.update(summarize(spectrum_energy(d)))
                rows[key] = row
                line = (f"{key:<32} rel {row['rel_delta']:.5f} chg {row['frac_changed']:.3f}  "
                        f"@64 {row.get('energy@64')} @256 {row.get('energy@256')} @1024 {row.get('energy@1024')}  "
                        f"r90 {row['rank@90_share']:.3f} r99 {row['rank@99_share']:.3f}")
                if blk in ctrl_blocks:
                    z = torch.randn(d.shape, generator=g)
                    z *= d.norm() / z.norm()
                    controls[key] = summarize(spectrum_energy(z))
                    line += f"  | noise @256 {controls[key].get('energy@256')} r90 {controls[key]['rank@90_share']:.3f}"
                print(f"{line}  ({time.time() - t0:.0f}s)", flush=True)
                if args.json:
                    args.json.write_text(json.dumps({"partial": True, "layers": rows, "controls": controls}, indent=1))

    keys = ["rel_delta", "frac_changed"] + [f"energy@{r}" for r in RANKS] + [f"rank@{int(p * 100)}_share" for p in LEVELS]
    by_kind = {}
    for kind in KINDS:
        sub = [r for k, r in rows.items() if f".{kind}." in k]
        by_kind[kind] = {m: round(statistics.median(r[m] for r in sub if m in r), 4) for m in keys}
    med = {m: round(statistics.median(r[m] for r in rows.values() if m in r), 4) for m in keys}
    print("median:", med)
    for kind, m in by_kind.items():
        print(f"  {kind}: {m}")
    if args.json:
        args.json.write_text(json.dumps({
            "measured_by": "bench/measure_bf16_delta_rank.py",
            "base": args.base.name, "other": args.other.name, "blocks": args.blocks,
            "layers": rows, "controls": controls, "median": med, "median_by_kind": by_kind,
        }, indent=1))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
