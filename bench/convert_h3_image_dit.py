#!/usr/bin/env python3
"""Write the MiniMax-H3-Image release as one bf16 file ComfyUI loads, nothing approximated.

The release ships the DiT as shards plus a separate file of per-step output
heads. This writes them as one `diffusion_models` file: every weight at the
dtype the release stores it in (bf16, with the fp32 islands kept fp32), the
timestep embedder and full-width AdaLN kept, the six output heads stacked the
way core reads them. `bench/h3_image/release.py` says what differs from the
release shards and where core says so. No linear is quantized and no AdaLN is
replaced by a table, so this file is the reference a quantized file is judged
against, not a file sized for the card.

    <comfy venv python> bench/convert_h3_image_dit.py --release <MiniMax-H3-Image folder> --out <file>

`--verify <quantized file>` reads an existing ComfyUI file of the same model
beside the release and reports, without writing anything, whether each
unquantized tensor the two share is the release's bytes and whether the output
heads are. It needs no card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h3_image import release as rel  # noqa: E402


def verify(release: rel.Sharded, heads_file, other_path: Path) -> int:
    """Every tensor the other file holds unquantized, against the release in ComfyUI layout."""
    same, differ, absent = [], [], []
    with safe_open(str(other_path), "pt") as other:
        other_keys = set(other.keys())
        for key in release.keys():
            if key not in other_keys:
                absent.append(key)
                continue
            if other.get_slice(key).get_dtype() in ("I8", "U8"):
                continue                      # a quantized linear: bench/measure_h3_image_weights.py reads those
            theirs = other.get_tensor(key)
            ours = rel.comfy_tensor(key, release, heads_file)
            if theirs.shape != ours.shape:
                differ.append((key, f"shape {tuple(theirs.shape)} against {tuple(ours.shape)}"))
            elif torch.equal(theirs.float(), ours.float()):
                same.append(key)
            else:
                gap = float((theirs.float() - ours.float()).norm() / ours.float().norm())
                differ.append((key, f"relative difference {gap:.3g}"))
            release.close()
    print(f"unquantized tensors equal to the release: {len(same)}")
    print(f"unquantized tensors that differ: {len(differ)}")
    for key, why in differ:
        print(f"  {key}: {why}")
    print(f"release tensors the file does not hold: {len(absent)}")
    for key in absent:
        print(f"  {key}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True, help="the MiniMax-H3-Image release folder")
    ap.add_argument("--out", type=Path, help="the file to write")
    ap.add_argument("--verify", type=Path, help="an existing ComfyUI file of this model to compare with the release")
    args = ap.parse_args()
    if (args.out is None) == (args.verify is None):
        ap.error("give exactly one of --out and --verify")

    release = rel.Sharded(args.release / "transformer")
    heads_file = safe_open(str(args.release / "pdd_fused_heads.safetensors"), "pt")
    if args.verify:
        return verify(release, heads_file, args.verify)

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this tool does not overwrite")
    meta = rel.metadata(json.loads((args.release / "pdd_config.json").read_text()))
    entries = rel.plan(release, heads_file)
    started = time.time()

    def progress(i: int, key: str) -> None:
        if i % 50 == 0:
            print(f"  {i}/{len(entries)} {key} ({time.time() - started:.0f}s)", flush=True)
        release.close()

    size = rel.write_safetensors(args.out, entries, lambda key: rel.comfy_tensor(key, release, heads_file),
                                 meta, progress)
    print(f"wrote {args.out.name}: {len(entries)} tensors, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
