#!/usr/bin/env python3
"""Write an exact overlay of one int8 H3 checkpoint on another, piece by piece.

`checkpoint_overlay.py` holds the format and says what each piece is. This is
its command line: a target and the base it was built on in, one safetensors
overlay out, and a record of what went into it. CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/build_checkpoint_overlay.py \\
        --base <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --target <models>/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors \\
        --out <models>/h3_overlays/fasth3_v2_on_fl2va.safetensors \\
        --record data/results/<date>_<name>_overlay.json

The overlay proves nothing until `bench/check_checkpoint_overlay.py` applies it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint_overlay import build_overlay, read_overlay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Write an exact overlay of TARGET on BASE.")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = build_overlay(str(args.base), str(args.target), str(args.out))
    args.out.chmod(0o644)   # the server runs as another user in the models group
    meta, pieces = read_overlay(str(args.out))
    rec = {"base": meta["base_name"], "base_sha256": meta["base_sha256"],
           "target": meta["target_name"], "target_sha256": meta["target_sha256"],
           "overlay": args.out.name, "overlay_bytes": args.out.stat().st_size,
           "pieces": {p: len(names) for p, names in pieces.items()}, **stats}
    if args.record:
        args.record.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "pieces"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
