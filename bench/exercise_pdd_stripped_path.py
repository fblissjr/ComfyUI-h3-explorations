#!/usr/bin/env python3
"""Exercise `MiniMaxH3PDDLoRA.execute` on a LOADED model with the baked pair.

Not a check: a one-off exercise of the stripped path end to end, which the
sidecar contract's synthetic cases cannot reach. Three cases, CPU only:

    stripped sidecar on the unbaked base   -> refused, naming the base
    stripped sidecar on its own bake       -> loads
    full sidecar on the bake               -> refused as a double apply

ComfyUI's model manager opens a CUDA context at import, so this sets the
`--cpu` flag before importing it and must run with CUDA_VISIBLE_DEVICES=.
The loads are lazy (mmap), so the wall time reported is the node's, not a
checkpoint read. First run 2026-09-05, log in
`bench/results/2026-09-05_bake_node_stripped_path.txt`.

    CUDA_VISIBLE_DEVICES= python bench/exercise_pdd_stripped_path.py \\
        --base <pruned int8 checkpoint> --bake <baked checkpoint> \\
        --stripped h3/<stripped sidecar> --full h3/<full sidecar>
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))
os.chdir(_HERE.parents[2])                      # folder_paths resolves from here

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import pdd_lora as P  # noqa: E402


def case(label, model, lora, expect):
    t = time.perf_counter()
    try:
        P.MiniMaxH3PDDLoRA.execute(model, lora, strength=1.0, steps=8)
        r = "LOADED"
    except Exception as e:  # noqa: BLE001
        r = "REFUSED: " + str(e).replace("\n", " ")[:220]
    print(f"[{label}] expect {expect} -> {r}  ({time.perf_counter() - t:.0f}s)", flush=True)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--bake", required=True)
    ap.add_argument("--stripped", required=True, help="lora name relative to models/loras")
    ap.add_argument("--full", required=True, help="lora name relative to models/loras")
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    print("device", mm.get_torch_device(), flush=True)
    base = comfy.sd.load_diffusion_model(a.base)
    r1 = case("stripped on the base", base, a.stripped, "REFUSED (unbaked base)")
    del base
    gc.collect()
    bake = comfy.sd.load_diffusion_model(a.bake)
    r2 = case("stripped on its bake", bake, a.stripped, "LOADED")
    r3 = case("full on the bake", bake, a.full, "REFUSED (double apply)")
    # Each refusal is matched on the sentence that names ITS decision, not on
    # "REFUSED" alone: a population or probe-completeness refusal would
    # otherwise pass as the double-apply one (interrupted review, 2026-09-05).
    ok = (r1.startswith("REFUSED") and "unbaked base" in r1 and r2 == "LOADED"
          and r3.startswith("REFUSED") and "SECOND time" in r3)
    print("every case as expected" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
