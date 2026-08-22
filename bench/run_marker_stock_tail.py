#!/usr/bin/env python3
"""Render six more STOCK-arm seeds, to settle the rolloff85 outlier.

Built 2026-08-21 and deliberately NOT run that night. Needs a live server and
the card; everything it needs to know is baked in so tomorrow is one command.

## The question, narrowed

`bench/results/2026-08-21_marker_arm_audio_spectrum.json` showed the stock arm
with sd 886 on `rolloff85_hz` against the routed arm's 119. That reads as an
unstable arm and is not one: **drop the single outlier and the stock sd falls to
137**, which is the routed arm's spread. The entire difference was one row.

Nor is it simply a quiet clip. The renders are matched-seed pairs, the outlier's
own pair is the quietest clip in BOTH arms, and the routed one came back at
1109 Hz -- ordinary for its arm.

So one draw in six went somewhere the other five did not. That is entirely
consistent with seed noise, and equally consistent with a tail the stock arm has
and the routed arm does not. **Six more stock seeds decide it, and what decides
it is a SECOND outlier, not a wider spread.** One more clip near 3400 Hz in
twelve makes it a property of the arm; twelve tight clips make the first one
noise.

Only the stock arm is rendered. The routed arm is not in question -- it was
tight to begin with, and spending the card on it would answer nothing.

## Why the prefix is fresh, which cost a session

The existing clips write `Video/marker_stock`, and that counter already carried
a capability-check render from before the batch. `bench/blind_batch.py`
correctly refused the whole set on 2026-08-21 -- "clips with this prefix predate
the JSONL's first row" -- because counter order could not be trusted. A fresh
prefix per batch is what makes the output mappable afterwards, so this script
sets one rather than inheriting the graph's.

## Seeds

910007-910012, continuing the original 910001-910006 without reusing one. A
repeat seed would return ComfyUI's cached sampler result rather than a render;
`run_graph_arms` guards that with `suspect_cache_hit`, and this avoids it by
construction.

    <comfy-venv>/bin/python bench/run_marker_stock_tail.py            # render
    <comfy-venv>/bin/python bench/run_marker_stock_tail.py --print    # show only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

GRAPH = _REPO / "internal" / "refs" / "marker_arm_stock_api.json"
OUT = _REPO / "bench" / "results" / "2026-08-22_marker_stock_tail.jsonl"
PREFIX = "Video/20260822-marker-stock-tail/marker_stock_tail"
SEED_BASE = 910007
RUNS = 6


def command() -> list[str]:
    return [
        sys.executable, str(_REPO / "bench" / "run_graph_arms.py"),
        "--arm", f"stock={GRAPH}",
        "--set", f"stock:VHS_VideoCombine.filename_prefix={PREFIX}",
        "--runs", str(RUNS),
        "--seed", str(SEED_BASE),
        "--out", str(OUT),
    ]


def main() -> int:
    if not GRAPH.exists():
        print(f"the stock arm graph is not on disk: {GRAPH}")
        print("it lives under internal/, which is gitignored -- if this is a "
              "fresh checkout the graph has to be rebuilt before this runs")
        return 2

    cmd = command()
    print("  " + " \\\n    ".join(cmd) + "\n")
    if "--print" in sys.argv:
        print("--print: nothing rendered.")
        return 0

    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\nrun_graph_arms exited {rc}; nothing further was run")
        return rc

    print(f"\nrendered -> {OUT.relative_to(_REPO)}")
    print("\nnext, and it must include BOTH stock jsonls or the spread is "
          "computed over six seeds again:")
    print(f"  <comfy-venv>/bin/python bench/grade_arm_audio_spectrum.py \\\n"
          f"    --arm stock=<the 12 stock clips> \\\n"
          f"    --arm official_tokens=<the 6 routed clips>")
    print("\nRead it as: a SECOND clip near 3400 Hz makes the outlier a "
          "property of the arm. Twelve tight clips make the first one noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
