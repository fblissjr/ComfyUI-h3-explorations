#!/usr/bin/env python3
"""Drive the arms for `docs/open_experiments.md` #22 and attribute the captures.

`run_graph_arms.py` submits graphs and times them; it knows nothing about the
capture module, which writes into ONE directory named by `H3_CAPTURE` at server
start. Filenames are keyed by (length, sequence, block, step, render), and every
arm here shares length, sequence, block and step by construction -- that is the
point of the experiment. So two arms differ in the filename only by the render
counter, and attribution would rest on nothing but submission order surviving
eleven renders without a single failure.

This driver removes that dependency: one arm per runner invocation, and the
files that appeared are moved into a per-arm directory before the next arm
runs. An arm that fails leaves its (empty or partial) directory behind and does
not shift anybody else's.

Needs a server started with capture armed, e.g.

    H3_CAPTURE="dir=<capdir>,blocks=0:12:24:36:49,steps=0,cycle=1,final=1"

`--dry-run` prints the eleven commands and moves nothing, which is how you read
the plan without spending a model load.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T2V = "workflows/bench/h3_text_to_video_stamped_api.json"
REF3 = "workflows/h3_probe_capture_ref3_api.json"

PRUNED = "minimax_h3_{v}_pruned_int8_convrot.safetensors"
# The unpruned files are reachable only through the nested folder symlink, so
# ComfyUI lists them under that prefix. Written as ComfyUI reports it rather
# than as the filesystem holds it, because this string is a widget value.
UNPRUNED = "diffusion_models/minimax_h3_{v}_int8_convrot.safetensors"
FP8 = "minimax_h3_{v}_pruned_fp8_scaled.safetensors"

# Ordered to keep checkpoint reloads down while leaving the determinism-floor
# repeat at the end, where it re-loads a checkpoint that has been evicted --
# which is the floor a real arm pays, not the floor of a warm cache.
ARMS = [
    ("t2v_fl2va_pruned",           T2V,  PRUNED.format(v="fl2va")),
    ("ref3_fl2va_pruned",          REF3, PRUNED.format(v="fl2va")),
    ("ref3_fl2va_fp8",             REF3, FP8.format(v="fl2va")),
    ("t2v_fl2va_unpruned",         T2V,  UNPRUNED.format(v="fl2va")),
    ("ref3_fl2va_unpruned",        REF3, UNPRUNED.format(v="fl2va")),
    ("t2v_ref2va_pruned",          T2V,  PRUNED.format(v="ref2va")),
    ("ref3_ref2va_pruned",         REF3, PRUNED.format(v="ref2va")),
    ("ref3_ref2va_fp8",            REF3, FP8.format(v="ref2va")),
    ("t2v_ref2va_unpruned",        T2V,  UNPRUNED.format(v="ref2va")),
    ("ref3_ref2va_unpruned",       REF3, UNPRUNED.format(v="ref2va")),
    ("ref3_ref2va_pruned_repeat",  REF3, PRUNED.format(v="ref2va")),
]

SQUARE = "768x768  1  576 tok/frame  0.33x"


def arm_command(label: str, graph: str, unet: str, canvas: str, length: int,
                seed: int, out: Path) -> list[str]:
    return [
        sys.executable, str(REPO / "bench/run_graph_arms.py"),
        "--arm", f"{label}={graph}",
        "--set", f'{label}:UNETLoader.unet_name="{unet}"',
        "--set", f"{label}:BasicScheduler.steps=1",
        "--set", f'{label}:MiniMaxH3Resolution.shape="square"',
        "--set", f'{label}:MiniMaxH3Resolution.shape.square_resolution="{canvas}"',
        "--set", f"{label}:MiniMaxH3Resolution.length={length}",
        "--runs", "1", "--seed", str(seed),
        "--out", str(out),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True,
                    help="the dir= given to H3_CAPTURE at server start")
    ap.add_argument("--out", required=True, help="timing JSONL")
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--seed", type=int, default=730451892)
    ap.add_argument("--only", help="run one arm by label")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    out = Path(args.out)
    arms = [a for a in ARMS if not args.only or a[0] == args.only]
    if not arms:
        raise SystemExit(f"--only {args.only!r} matches no arm")

    manifest = []
    for i, (label, graph, unet) in enumerate(arms, 1):
        cmd = arm_command(label, graph, unet, SQUARE, args.length, args.seed, out)
        print(f"\n[{i}/{len(arms)}] {label}  ({unet})", flush=True)
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        before = {p.name for p in cap.glob("*.pt")}
        rc = subprocess.run(cmd, cwd=REPO).returncode
        after = {p.name for p in cap.glob("*.pt")}
        new = sorted(after - before)
        dest = cap / label
        dest.mkdir(exist_ok=True)
        for name in new:
            shutil.move(str(cap / name), str(dest / name))
        print(f"    rc={rc}  {len(new)} capture file(s) -> {dest.name}/", flush=True)
        manifest.append({"arm": label, "graph": graph, "unet": unet,
                         "returncode": rc, "captures": new})
        if not new:
            print("    WARNING: no captures. The tap is armed at server start "
                  "and installed by the sage node; a graph without that node, "
                  "or a server started without H3_CAPTURE, writes nothing.",
                  flush=True)

    if not args.dry_run:
        (cap / "arm_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nwrote {cap / 'arm_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
