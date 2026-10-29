#!/usr/bin/env python3
"""Score the Sol dense_blocks panel: each arm's saved latents against the
all-dense arm at the same scene and seed, with each arm's sampler time.

Written 2026-10-02 for `bench/sol_dense_blocks_panel_arms.json` (the design and
the reason are in its `what` and `why`). Arm labels are `dbp_<scene>__<arm>`;
the arm named `dense` (`dense_blocks=0-49`) is the reference, because it is what
Sol approximates on this graph: every block on the kitchen INT8 fallback.

Latents are found through `/history/<prompt_id>` on the server that rendered
them (the row's `prompt_id`; `SaveLatent` reports its files there), so a row
is never matched to a file by a filename counter. The server must still be up.

Per arm, per (scene, seed), against the dense arm:
  video_rel   ||z - dense|| / ||dense|| on the final video latent
  pass1_rel   the same after the PDD stage (sigma 0.8, before the finisher)
  audio_rel   the same on the final audio latent
  frame0_rel  video_rel on latent frame 0 alone (the opening composition)

`pass1_rel` is dominated by the seed's shared noise (the latent at sigma 0.8
is mostly noise), so it orders arms but is not on the final latent's scale.
A clean-video rescaling of it was tried on 2026-10-02 and came out above 1,
so the noise/signal split does not hold literally on this sampler; dropped.
Per cell, `sol_pairwise`: the mean video_rel between every pair of non-dense
arms, so "how far from dense" can be read against "how far from each other".
And per scene, `seed_rel`: the dense arm at seed 0 against the dense arm at
seed 1. That is the distance between two different samples of the same
prompt, so an arm's `video_rel / seed_rel` says how much of "a different
sample" the arm's approximation amounts to.

Speed: `sampler_s` (both SamplerCustomAdvanced nodes) and per-stage seconds
from `per_node_s` (node 10 = PDD stage, node 123 = FlashGen finisher). The
probe was not armed, so timings are valid; the text encode lands in total_s,
not sampler_s.

    python bench/score_sol_dense_blocks_panel.py ROWS.jsonl --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import safetensors.torch
import torch

PDD_NODE, FINISH_NODE = "10", "123"
# SaveLatent node ids in the savelat graph (read from the graph, 2026-10-02).
SAVE_NODES = {"102": "video", "103": "audio", "108": "pass1_video", "109": "pass1_audio"}


def history(host: str, prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{host}/history/{prompt_id}") as r:
        return json.load(r).get(prompt_id, {})


def latent_paths(host: str, prompt_id: str, out_root: Path) -> dict[str, Path]:
    outs = history(host, prompt_id).get("outputs", {})
    paths = {}
    for nid, kind in SAVE_NODES.items():
        files = (outs.get(nid) or {}).get("latents") or []
        if files:
            f = files[0]
            paths[kind] = out_root / f.get("subfolder", "") / f["filename"]
    return paths


def load(p: Path) -> torch.Tensor:
    return safetensors.torch.load_file(str(p))["latent_tensor"].double()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--output-root", type=Path, default=Path(os.environ["H3_OUTPUT_DIR"]) if os.environ.get("H3_OUTPUT_DIR") else None,
                    help="the server's --output-directory (default: $H3_OUTPUT_DIR)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.output_root is None:
        ap.error("--output-root is required when H3_OUTPUT_DIR is not set in the environment")

    rows = [json.loads(l) for l in args.rows.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("warmup") and not r.get("error")]
    cells: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        scene, arm = r["label"].removeprefix("dbp_").split("__")
        if r.get("suspect_cache_hit"):
            print(f"skip {r['label']} seed {r['seed']}: suspect cache hit", file=sys.stderr)
            continue
        pn = r.get("per_node_s") or {}
        cells[(scene, r["seed"])][arm] = {
            "paths": latent_paths(args.host, r["prompt_id"], args.output_root),
            "sampler_s": r.get("sampler_s"), "total_s": r.get("total_s"),
            "pdd_s": pn.get(PDD_NODE), "finish_s": pn.get(FINISH_NODE),
        }

    per_cell = []
    for (scene, seed), arms in sorted(cells.items()):
        if "dense" not in arms:
            continue
        ref = {k: load(p) for k, p in arms["dense"]["paths"].items()}
        for arm, a in arms.items():
            z = {k: load(p) for k, p in a["paths"].items()}
            per_cell.append({
                "scene": scene, "seed": seed, "arm": arm,
                "video_rel": rel(z["video"], ref["video"]),
                "frame0_rel": rel(z["video"][:, :, :1], ref["video"][:, :, :1]),
                "pass1_rel": rel(z["pass1_video"], ref["pass1_video"]) if "pass1_video" in z else None,
                "audio_rel": rel(z["audio"], ref["audio"]) if "audio" in z else None,
                "sampler_s": a["sampler_s"], "pdd_s": a["pdd_s"], "finish_s": a["finish_s"],
                "total_s": a["total_s"],
            })

    sol_pairwise = {}
    for (scene, seed), arms in sorted(cells.items()):
        names = sorted(a for a in arms if a != "dense")
        vids = {a: load(arms[a]["paths"]["video"]) for a in names}
        d = [rel(vids[a], vids[b]) for i, a in enumerate(names) for b in names[i + 1:]]
        if d:
            sol_pairwise[f"{scene}/{seed}"] = {"mean": st.mean(d), "min": min(d), "max": max(d)}

    # Between-sample scale: dense at one seed against dense at the other, per scene.
    seed_rel = {}
    by_scene = defaultdict(dict)
    for (scene, seed), arms in cells.items():
        if "dense" in arms:
            by_scene[scene][seed] = arms["dense"]["paths"]["video"]
    for scene, s in by_scene.items():
        if len(s) == 2:
            a, b = (load(p) for _, p in sorted(s.items()))
            seed_rel[scene] = rel(a, b)
    for c in per_cell:
        c["video_rel_over_seed"] = (c["video_rel"] / seed_rel[c["scene"]]
                                    if c["scene"] in seed_rel else None)

    # Per arm: means, and paired wins against the shipped arm on the same cells.
    arms = sorted({c["arm"] for c in per_cell})
    key = {(c["scene"], c["seed"], c["arm"]): c for c in per_cell}
    summary = {}
    for arm in arms:
        cs = [c for c in per_cell if c["arm"] == arm]
        def m(k):
            v = [c[k] for c in cs if c[k] is not None]
            return st.mean(v) if v else None
        closer = sum(1 for c in cs if (c["scene"], c["seed"], "optC") in key
                     and c["video_rel"] < key[(c["scene"], c["seed"], "optC")]["video_rel"])
        summary[arm] = {"cells": len(cs), "video_rel": m("video_rel"), "pass1_rel": m("pass1_rel"),
                        "frame0_rel": m("frame0_rel"),
                        "audio_rel": m("audio_rel"), "video_rel_over_seed": m("video_rel_over_seed"),
                        "sampler_s": m("sampler_s"), "pdd_s": m("pdd_s"), "finish_s": m("finish_s"),
                        "closer_than_optC": closer}

    print(f"{'arm':<9} {'n':>2} {'video':>7} {'pass1':>7} {'audio':>7} {'/seed':>6} "
          f"{'sampler':>8} {'pdd':>7} {'finish':>7} {'<optC':>6}")
    order = sorted(arms, key=lambda a: summary[a]["video_rel"] or 0)
    for arm in order:
        s = summary[arm]
        f = lambda v, w=7, d=4: f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"
        print(f"{arm:<9} {s['cells']:>2} {f(s['video_rel'])} {f(s['pass1_rel'])} {f(s['audio_rel'])} "
              f"{f(s['video_rel_over_seed'], 6, 3)} {f(s['sampler_s'], 8, 1)} {f(s['pdd_s'], 7, 1)} "
              f"{f(s['finish_s'], 7, 1)} {s['closer_than_optC']:>6}")
    print("frame0_rel:", {a: round(summary[a]["frame0_rel"], 4) for a in order})
    print("sol_pairwise (mean video_rel between non-dense arms, per cell):",
          {k: round(v["mean"], 4) for k, v in sol_pairwise.items()})
    print("seed_rel (dense seed 0 vs seed 1):", {k: round(v, 4) for k, v in seed_rel.items()})

    if args.out:
        args.out.write_text(json.dumps({
            "measured_by": "bench/score_sol_dense_blocks_panel.py", "rows": str(args.rows),
            "reference_arm": "dense", "seed_rel": seed_rel, "sol_pairwise": sol_pairwise, "summary": summary, "cells": per_cell,
        }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
