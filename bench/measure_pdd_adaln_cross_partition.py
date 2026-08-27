#!/usr/bin/env python3
"""How wrong is a PDD file's baked adaln when applied to the OTHER partition?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no CUDA, no render.

The question this answers. On a pruned checkpoint the adaln update is
pre-solved into that checkpoint's 8-column curve basis
(`docs/h3_pdd.md`, "The pruned base has nowhere to put the adaln delta"), and
that basis is PARTITION-SPECIFIC. `convert_pdd_lora.py` refuses a bake whose fit
residual exceeds 1e-3, but a residual cannot detect a wrong basis: fl2va's and
ref2va's tables are SVDs of very similar smooth curves, so they span nearly the
same subspace and differ in their COORDINATES. A cross-partition bake therefore
fits well and is wrong at runtime.

`docs/h3_pdd.md` quotes 0.0205 against 0.0001 for that, and the node's
`PARTITION_TOLERANCE` is placed between a dtype cast and the partition gap on
the strength of it. **Neither number was recorded for the artifacts this repo
actually ships.** This measures them, on the shipped files, so a decision about
whether to allow fl2va PDD on a ref2va base is made against this box's numbers
rather than a figure in a code comment.

What it does NOT need: the published source LoRA. The quantity that matters is
the difference between the modulation a file produces on its own table and on
the other one -- both are computed from the same baked tensors, so ground truth
cancels and only the basis changes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
LORAS = Path.home() / "ComfyUI" / "models" / "loras" / "h3"
DIFFUSION = Path.home() / "ComfyUI" / "models" / "diffusion_models"
PARTITIONS = ("fl2va", "ref2va")


def table(part: str) -> torch.Tensor:
    with safe_open(DIFFUSION / f"minimax_h3_{part}_pruned_int8_convrot.safetensors",
                   framework="pt") as f:
        return f.get_tensor("adaln_t_table").double()


def modulation(sd, block: int, t: torch.Tensor) -> torch.Tensor:
    """What the block's adaln patch adds, over the whole 1025-row time grid."""
    w = sd[f"h3_pdd.adaln_baked.blocks.{block}.diff"].double()
    b = sd[f"h3_pdd.adaln_baked.blocks.{block}.diff_b"].double()
    return t @ w.T + b


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main() -> int:
    tables = {p: table(p) for p in PARTITIONS}
    # A dtype cast is the floor this has to clear to mean anything: the same
    # basis in bf16 against fp64. Without it a small cross-partition number
    # could not be distinguished from rounding.
    cast_floor = {p: rel(tables[p].to(torch.bfloat16).double(), tables[p])
                  for p in PARTITIONS}

    out = {}
    for part in PARTITIONS:
        other = "ref2va" if part == "fl2va" else "fl2va"
        path = LORAS / f"minimax_h3_{part}_pdd_8step_comfy.safetensors"
        sd = load_file(path)
        with safe_open(path, framework="pt") as f:
            meta = dict(f.metadata() or {})
        blocks = sorted(int(k.split(".")[3]) for k in sd
                        if k.startswith("h3_pdd.adaln_baked.") and k.endswith(".diff"))
        errs = {}
        for i in blocks:
            correct = modulation(sd, i, tables[part])
            wrong = modulation(sd, i, tables[other])
            errs[i] = rel(wrong, correct)
        vals = sorted(errs.values())
        out[part] = {
            "file": path.name,
            "declares_pruned_base": meta.get("h3_pdd_pruned_base"),
            "applied_to": f"minimax_h3_{other}_pruned_int8_convrot.safetensors",
            "blocks": len(blocks),
            "relative_error": {
                "min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
                "mean": sum(vals) / len(vals)},
            "per_block": {str(k): v for k, v in sorted(errs.items())},
        }

    record = {
        "question": "how wrong is a PDD file's baked adaln applied to the other "
                    "partition's pruned checkpoint, on the shipped artifacts",
        "method": "the baked patch evaluated over the full 1025-row time grid "
                  "against its own adaln_t_table and against the other "
                  "partition's; ground truth cancels because both sides use the "
                  "same baked tensors. float64 throughout.",
        "path_policy": "logical identifiers only",
        "bf16_cast_floor": cast_floor,
        "partition_tolerance_in_node": 0.015,
        "arms": out,
    }
    stamp = REPO / f"bench/results/{date.today().isoformat()}_pdd_adaln_cross_partition.json"
    stamp.write_text(json.dumps(record, indent=1) + "\n")

    print(f"bf16 cast floor (same basis): "
          f"{', '.join(f'{k} {v:.2e}' for k, v in cast_floor.items())}\n")
    for part, r in out.items():
        e = r["relative_error"]
        print(f"{r['file']}")
        print(f"  declares base : {r['declares_pruned_base']}")
        print(f"  applied to    : {r['applied_to']}")
        print(f"  {r['blocks']} blocks, relative error "
              f"min {e['min']:.4f}  median {e['median']:.4f}  max {e['max']:.4f}")
    print(f"\nwrote {stamp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
