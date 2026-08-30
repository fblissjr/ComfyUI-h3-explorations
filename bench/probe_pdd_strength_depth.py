#!/usr/bin/env python3
"""Where in the DiT does PDD's STRENGTH change the output? One block at a time.

## What this measures, and why it needs `unmerged_blocks` to exist at all

`docs/research/pdd/depth_and_axes.md` §3 is the question: where does running a
block undistilled, half-distilled or over-distilled move the output most.

**A strength change on a MERGED block cannot answer it.** The backbone LoRA is
a weight patch applied at load, so it affects every step; two arms diverge from
step 0 and are different samples rather than a perturbation and its baseline,
which CLAUDE.md's different-sample rule says answers nothing. The fix is the
same one `bench/probe_block_propagation.py` gets for free from Sol's shape:
confine the difference to ONE model evaluation.

`MiniMaxH3PDDLoRA`'s `unmerged_blocks` applies a block's delta at the call
rather than in the weight, and `unmerged_window` gates that on the sigma the
step tracker records. So one block, one step, one strength -- and every earlier
step is bit-identical across arms by construction.

## The baseline is PER BLOCK, and that is not a detail

An un-merged, windowed block carries NO delta outside the window. So an arm at
block N is not comparable to a fully merged render -- it is comparable only to
another arm at block N with the same window. The baseline for block N is
therefore block N un-merged and windowed at `strength` 1.0, and the arms vary
the strength inside the window against it.

Comparing every block's arms to one shared merged baseline would fold "this
block lost its delta for three steps" into the number and read it as a strength
effect. That is the mistake this docstring exists to prevent.

## What it does not establish

* One seed, one prompt, one canvas, one step. The network's response at that
  sigma, not a quality result.
* rel L2 on a latent is not a perceptual quantity, and CLAUDE.md's rule means a
  rendered pair could not make it one.
* The measured step is the last of four, where the 4-evaluation partition's
  final block begins -- the coarsest stretch of the schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WF = HERE.parent / "workflows"

PROBE_STEPS = 4
# 0.66 -> sigma 0.8608 at shift 12, between step 2's 0.9231 and step 3's 0.8.
# The same window `probe_block_propagation.py` uses, for the same reason.
PROBE_WINDOW = "0.66-1.0"
BASELINE_STRENGTH = 1.0


def post(host: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://{host}{path}", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get(host: str, path: str) -> dict:
    with urllib.request.urlopen(f"http://{host}{path}", timeout=60) as r:
        return json.loads(r.read().decode())


def build_arm(base: dict, block: int, strength: float, seed: int,
              length: int, tag: str) -> dict:
    g = json.loads(json.dumps(base))

    def one(cls):
        hits = [k for k, v in g.items() if v.get("class_type") == cls]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one {cls}, found {len(hits)}")
        return hits[0]

    pdd, noise, sampler = one("MiniMaxH3PDDLoRA"), one("RandomNoise"), \
        one("KSamplerSelect")
    g[noise]["inputs"]["noise_seed"] = seed
    g[sampler]["inputs"]["sampler_name"] = "euler"
    for k, v in g.items():
        if v.get("class_type") == "MiniMaxH3Resolution":
            v["inputs"]["length"] = length

    p = g[pdd]["inputs"]
    p["unmerged_blocks"] = str(block)
    p["unmerged_strength"] = float(strength)
    p["unmerged_window"] = PROBE_WINDOW

    # Latent out, no decode: a VAE pass is wall time that only adds another
    # approximation between the model's output and the number.
    sca = one("SamplerCustomAdvanced")
    for cls in ("VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine"):
        for k in [k for k, v in g.items() if v.get("class_type") == cls]:
            del g[k]
    g["9001"] = {"class_type": "SaveLatent",
                 "inputs": {"samples": [sca, 0],
                            "filename_prefix": f"h3_pdd_depth/{tag}"}}
    return g


def run(host: str, graph: dict, timeout: float) -> str:
    pid = post(host, "/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = get(host, f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise SystemExit(f"arm failed: {json.dumps(st)[:400]}")
            return pid
        time.sleep(2)
    raise SystemExit(f"arm timed out after {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--workflow", default="h3_text_to_video_pdd_4step_api.json")
    ap.add_argument("--blocks", default="0,1,2,8,16,24,32,40,45,48,49")
    ap.add_argument("--strengths", default="0.0,0.5,1.5")
    ap.add_argument("--length", type=int, default=345)
    ap.add_argument("--seed", type=int, default=730451892)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = json.loads((WF / args.workflow).read_text())
    blocks = [int(b) for b in args.blocks.split(",")]
    strengths = [float(s) for s in args.strengths.split(",")]

    plan = []
    for b in blocks:
        plan.append((b, BASELINE_STRENGTH, f"b{b:02d}_base"))
        for s in strengths:
            plan.append((b, s, f"b{b:02d}_s{s}"))

    print(f"{len(plan)} arms: {len(blocks)} blocks x "
          f"({len(strengths)} strengths + 1 per-block baseline), "
          f"{args.length} frames, {PROBE_STEPS} steps, window {PROBE_WINDOW}")
    rows = []
    t_start = time.time()
    for i, (b, s, tag) in enumerate(plan, 1):
        g = build_arm(base, b, s, args.seed, args.length, tag)
        t0 = time.time()
        pid = run(args.host, g, args.timeout)
        dt = time.time() - t0
        rows.append({"block": b, "strength": s, "tag": tag,
                     "prompt_id": pid, "seconds": round(dt, 1)})
        print(f"  [{i}/{len(plan)}] {tag:16s} {dt:6.1f}s  {pid[:8]}",
              flush=True)

    out = {
        "measured": time.strftime("%Y-%m-%d"),
        "produced_by": "bench/probe_pdd_strength_depth.py",
        "question": ("where in the DiT does PDD's strength change the output, "
                     "measured one block and one step at a time"),
        "workflow": args.workflow, "length": args.length,
        "steps": PROBE_STEPS, "window": PROBE_WINDOW, "seed": args.seed,
        "baseline": ("PER BLOCK: the same block un-merged and windowed at "
                     "strength 1.0. NOT a shared merged render -- see the "
                     "script docstring."),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "arms": rows,
        "scored": False,
        "note": ("latents only; scoring is a separate pass over "
                 "output/latents/h3_pdd_depth/"),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
