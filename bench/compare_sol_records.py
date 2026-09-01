#!/usr/bin/env python3
"""Compare two renders' Sol counts, call for call: the cache-state control.

Routed counts are a function of the attention inputs, and the inputs are a
function of the graph, the seed and the weights. So two renders of one graph
at one seed must produce the same counts whether the process was cold (first
prompt after a restart) or warm (models and node outputs resident), and
whether or not a restart sat between them. If they do not, something in the
cache state reaches the numerics, and that is worth knowing before any
routing figure is quoted.

    python bench/compare_sol_records.py <record-A> <record-B> [--prompt-a ID] [--prompt-b ID]

Each argument is a directory or a jsonl; `--prompt-*` picks one prompt id
when a file holds several. Sol rows are paired by (block, schedule index),
falling back to (block, sigma) when no index matched, and the raw count
tensors are compared bitwise. Exit 0 when every paired call agrees exactly,
1 otherwise, 2 when the records cannot be paired.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _live_sol import sol_observe as _load_observer  # noqa: E402
from sol_observe_report import _load  # noqa: E402


def _record(path, prompt):
    p = Path(path)
    jsonl = p if p.is_file() else sorted(p.glob("sol_observe_*.jsonl"))[-1]
    rows = _load(jsonl)
    calls = [r for r in rows if r.get("kind") == "call" and r["route"] == "sol"]
    ids = sorted(p for p in {r.get("prompt_id") for r in calls} if p)
    if prompt:
        calls = [r for r in calls if r.get("prompt_id") == prompt]
    elif len(ids) > 1:
        sys.exit(f"{jsonl.name} holds {len(ids)} prompt ids; pick one with --prompt-a/--prompt-b: {ids}")
    renders = {r["prompt_id"]: r for r in rows if r.get("kind") == "render"}
    header = next((r for r in rows if r.get("kind") == "header"), {})
    return jsonl, calls, renders, header


def _key(r):
    idx = (r.get("schedule") or {}).get("schedule_index")
    return (r.get("block"), idx if idx is not None else round(float(r.get("sigma") or -1), 6))


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--prompt-a")
    ap.add_argument("--prompt-b")
    args = ap.parse_args()
    obs = _load_observer()

    ja, ca, ra, ha = _record(args.a, args.prompt_a)
    jb, cb, rb, hb = _record(args.b, args.prompt_b)
    for label, j, c, rr, h in (("A", ja, ca, ra, ha), ("B", jb, cb, rb, hb)):
        pid = c[0].get("prompt_id") if c else None
        rd = rr.get(pid) or {}
        print(f"{label}: {j.name}, pid {h.get('pid')}, prompt {pid}, {len(c)} sol rows, "
              f"workflow {rd.get('workflow_file', '(no render row)')}, "
              f"process render index {rd.get('process_render_index', '?')}")
    if not ca or not cb:
        print("nothing to pair")
        return 2
    A = {_key(r): r for r in ca}
    B = {_key(r): r for r in cb}
    common = sorted(set(A) & set(B), key=lambda k: (str(k[1]), k[0]))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    print(f"paired {len(common)} call(s); only in A {len(only_a)}, only in B {len(only_b)}")
    if not common:
        return 2
    exact, differ, worst = 0, [], 0
    for k in common:
        ta = obs.read_raw(ja, A[k])
        tb = obs.read_raw(jb, B[k])
        if ta.shape != tb.shape:
            differ.append((k, f"shape {tuple(ta.shape)} vs {tuple(tb.shape)}"))
            continue
        if bool((ta == tb).all()):
            exact += 1
        else:
            d = (ta - tb).abs()
            worst = max(worst, int(d.max()))
            differ.append((k, f"{int((d > 0).sum())} of {d.numel()} entries differ, max |diff| {int(d.max())}"))
    print(f"bitwise identical: {exact} of {len(common)}")
    for k, why in differ[:12]:
        print(f"  differs at block {k[0]} step {k[1]}: {why}")
    if differ:
        print(f"\nDIFFER: {len(differ)} paired call(s), worst |diff| {worst} block(s); "
              "the cache state (or something else between the runs) reaches the routing")
        return 1
    print("\nevery paired call agrees exactly: routing did not depend on the cache state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
