#!/usr/bin/env python3
"""How small an exact overlay of one int8 H3 checkpoint on another would be.

The question (owner, 2026-09-29): ship a FastH3-based checkpoint as something
smaller than a 22 GB re-upload. A target that is the base plus a few changes
can travel as an overlay: the tensors the base lacks, the tensors that differ,
and, for int8 linears that share a shape, only the codes that changed. This
measures each part, on the files, CPU only:

- tensor bytes by category (backbone linears, gates, token refiner,
  adaln/time, final layer, other), and which exist only in the target;
- for every int8 linear both files hold as int8: the fraction of codes that
  differ, the largest code difference, and whether the per-row scales match;
- the sizes of an exact overlay's parts: changed codes as int32 index plus
  int8 delta, and the entropy bound of that ternary diff;
- with `--svd`, the singular-value energy of the target-only gate weights,
  read from a bf16 source, to say whether a low-rank form could stand in.

Measures only. Nothing is written but the record.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/measure_checkpoint_overlay.py \\
        --base <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --target <models>/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors \\
        [--svd <FastH3 bf16 file> --svd-blocks 0,25,49] --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import struct
from pathlib import Path

import torch
from safetensors import safe_open

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def nbytes(entry) -> int:
    n = 1
    for d in entry["shape"]:
        n *= d
    return n * DTYPE_BYTES[entry["dtype"]]


def category(key: str) -> str:
    if "to_gate_compress" in key:
        return "gates"
    if "adaln" in key or "time" in key:
        return "adaln/time"
    if key.startswith("token_refiner"):
        return "token_refiner"
    if "final_layer" in key:
        return "final_layer"
    if key.startswith("blocks.") and any(s in key for s in ("qkv_proj", "out_proj", "fc1", "fc2")):
        return "backbone_linears"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="Size an exact int8 overlay of one H3 checkpoint on another.")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--svd", type=Path, default=None, help="a bf16 source of the target, for the gates' rank")
    ap.add_argument("--svd-blocks", default="0,25,49")
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()

    hb, ht = header(args.base), header(args.target)
    rec = {"base": args.base.name, "target": args.target.name}

    by_cat, only_target = collections.Counter(), collections.Counter()
    dtypes = collections.defaultdict(set)
    for k, v in ht.items():
        c = category(k)
        by_cat[c] += nbytes(v)
        dtypes[c].add(v["dtype"])
        if k not in hb:
            only_target[c] += nbytes(v)
    rec["target_bytes_by_category"] = {c: {"bytes": by_cat[c], "only_in_target_bytes": only_target[c],
                                           "dtypes": sorted(dtypes[c])} for c in by_cat}
    rec["only_in_base"] = sorted(k for k in hb if k not in ht)

    fb, ft = safe_open(str(args.base), "pt"), safe_open(str(args.target), "pt")
    total = changed = maxdiff = 0
    layers, dtype_changed, scale_rows = [], [], 0
    for k in sorted(ht):
        if not k.endswith(".weight") or ht[k]["dtype"] != "I8" or k not in hb:
            continue
        if hb[k]["dtype"] != "I8":
            dtype_changed.append({"key": k, "base_dtype": hb[k]["dtype"], "target_bytes": nbytes(ht[k])})
            continue
        a, c = ft.get_tensor(k), fb.get_tensor(k)
        d = (a.int() - c.int()).abs()
        n, m = a.numel(), int((d != 0).sum())
        total, changed, maxdiff = total + n, changed + m, max(maxdiff, int(d.max()))
        sk = k[: -len(".weight")] + ".weight_scale"
        same_scale = sk in hb and sk in ht and torch.equal(fb.get_tensor(sk), ft.get_tensor(sk))
        scale_rows += ht[sk]["shape"][0] if sk in ht else 0
        layers.append({"key": k, "changed_fraction": m / n, "scales_equal": same_scale})
    p = changed / total if total else 0.0
    entropy = -(1 - p) * math.log2(1 - p) - p * math.log2(p / 2) if 0 < p < 1 else 0.0
    rec["shared_int8_linears"] = {
        "count": len(layers), "codes": total, "changed": changed, "changed_fraction": p,
        "max_abs_code_diff": maxdiff,
        "scales_equal_count": sum(x["scales_equal"] for x in layers),
        "min_layer": min(layers, key=lambda x: x["changed_fraction"]) if layers else None,
        "max_layer": max(layers, key=lambda x: x["changed_fraction"]) if layers else None,
        "overlay_bytes_index_int32_plus_delta_int8": changed * 5,
        "overlay_bytes_entropy_bound": total * entropy / 8,
        "overlay_bytes_row_scales_f32": scale_rows * 4,
    }
    rec["int8_in_target_other_dtype_in_base"] = dtype_changed
    rec["per_layer"] = layers

    if args.svd:
        sv = {}
        with safe_open(str(args.svd), "pt") as f:
            for blk in (int(x) for x in args.svd_blocks.split(",")):
                w = f.get_tensor(f"blocks.{blk}.attn.to_gate_compress.weight").float()
                s = torch.linalg.svdvals(w)
                e = (s ** 2).cumsum(0) / (s ** 2).sum()
                sv[f"blocks.{blk}"] = {"shape": list(w.shape),
                                       **{f"rank_for_{q}": int((e < q).sum()) + 1 for q in (0.9, 0.99, 0.999)}}
        rec["gate_energy_rank"] = sv

    args.record.write_text(json.dumps(rec, indent=1) + "\n")
    s = rec["shared_int8_linears"]
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_layer",)}, indent=1)[:4000])
    print(f"\nchanged codes {s['changed_fraction'] * 100:.3f}%; wrote {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
