#!/usr/bin/env python3
"""Check the e2e bench measures the configuration the graphs actually ship.

`check_generator_constants.py` pins the *generator* against upstream constants.
Nothing pinned the *bench*, and on 2026-08-13 that gap cost a day of numbers:
the `mode="fp16 (most accurate)"` flip landed in `h3_config.py` and in every
shipped graph, while `bench_e2e_h3.py` kept a hardcoded `"mode": "auto"`. So
every e2e arm measured between then and 2026-08-14 compared against a sage
baseline nobody ships.

**The direction of that error is the reason this check exists.** `auto`
resolves to `fp8_cuda++`, the FASTEST kernel; the shipped mode is the most
accurate one and costs ~1.58x. A fast baseline makes every competing arm look
*worse*, so the bug produced conservative-looking numbers -- the kind nobody
double-checks. A bug that flatters its own result gets caught; one that
understates does not.

What this pins, and what it does not: it compares the bench's `shipped` arm
against the shipped graphs, node for node. It says nothing about whether those
settings are *right* -- `check_distill_settings.py` and the reasoning in
`h3_config.py` own that. This only says the bench and the graphs agree, which
is the property that silently broke.

Claims, i.e. what breaks if a case is deleted:

  clip_matches       the bench loads a different text encoder from the shipped
                     graph, so every e2e arm pays a different memory/load path
                     and conditions from different quantized weights.
  sage_matches       the bench measures a sage configuration nobody ships, in
                     the direction that understates every arm compared against
                     it. This is the case that would have caught 2026-08-13.
  sol_matches        the bench's `shipped` arm drifts from SOL_RECOMMENDED_CUDA
                     -- so an arm named for the shipped config measures
                     something else, and its deltas are against a baseline
                     that appears nowhere.
  sol_node_matches   the bench builds a different Sol NODE than the graphs
                     wire, which would compare two kernels while reporting a
                     knob. Cheap, and it is the thing that changed today.

Shown red: 2026-08-14, by reverting the bench's sage node to `mode="auto"` --
the exact historical bug. `sage_matches` failed and named the key. Shown red
again 2026-08-23 when the shipped encoder moved to the owner's W4A16 AWQ build
while the bench retained int8_convrot; `clip_matches` named that exact drift.

Needs neither CUDA nor a model nor a server. Runs in about a second.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "workflows"))

# The shipped graphs to compare against. The API form of the plain t2v graph
# omits Sol by policy (Sol ships OFF), so the Sol comparison uses the one
# graph that exists precisely to keep that question answerable.
SAGE_GRAPH = "h3_text_to_video_api.json"
SOL_GRAPH = "h3_probe_sol_on_api.json"

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_bench():
    """Import bench_e2e_h3 without running main()."""
    spec = importlib.util.spec_from_file_location(
        "_bench_e2e", _REPO / "bench" / "bench_e2e_h3.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def node_inputs(graph_file, class_type):
    graph = json.loads((_REPO / "workflows" / graph_file).read_text())
    for node in graph.values():
        if node.get("class_type") == class_type:
            return {k: v for k, v in node["inputs"].items()
                    if not isinstance(v, list)}    # drop wired links
    return None


bench = load_bench()
sage_arm, sol_arm = bench.resolve_arm("shipped")
built = bench.build_prompt(dict(bench.DEFAULTS, length=362), sage=sage_arm,
                           seed=1, sol=sol_arm)


def built_inputs(class_type):
    for node in built.values():
        if node.get("class_type") == class_type:
            return {k: v for k, v in node["inputs"].items()
                    if not isinstance(v, list)}
    return None


print(f"the bench's `shipped` arm vs what workflows/ ships "
      f"(sol backend: {bench.SOL_BACKEND}):\n")

print("text encoder:")
# **Find the encoder loader by looking, not by name.** This named
# `MiniMaxH3AWQEncoderLoader` outright, so when the shipped encoder became the
# ComfyUI-native INT8 build on 2026-08-27 and the graphs correctly moved to
# `CLIPLoader`, the check reported "missing" -- reading a correct migration as
# drift. Which loader is right is decided by the file; this check's subject is
# whether the bench and the shipped graphs AGREE, whichever it is.
ENCODER_LOADERS = ("CLIPLoader", "MiniMaxH3EncoderLoader",
                   "MiniMaxH3AWQEncoderLoader")
_loader = next((c for c in ENCODER_LOADERS if node_inputs(SAGE_GRAPH, c)), None)
want = node_inputs(SAGE_GRAPH, _loader) if _loader else None
got = built_inputs(_loader) if _loader else None
if want is None or got is None:
    check("clip_matches", False,
          f"missing an encoder loader ({_loader or 'none found'}) in "
          f"{'the graph' if want is None else 'the bench graph'}")
else:
    diff = {k: (want.get(k), got.get(k)) for k in set(want) | set(got)
            if want.get(k) != got.get(k)}
    check("clip_matches", not diff,
          f"differs: {diff} (graph, bench)" if diff else f"{len(want)} inputs agree")

print()
print("sage node:")
want = node_inputs(SAGE_GRAPH, "MiniMaxH3SageAttention")
got = built_inputs("MiniMaxH3SageAttention")
if want is None or got is None:
    check("sage_matches", False,
          f"missing MiniMaxH3SageAttention in "
          f"{'the graph' if want is None else 'the bench graph'}")
else:
    diff = {k: (want.get(k), got.get(k)) for k in set(want) | set(got)
            if want.get(k) != got.get(k)}
    check("sage_matches", not diff,
          f"differs: {diff} (graph, bench)" if diff else f"{len(want)} inputs agree")

print("\nsol node:")
sol_class, _ = bench.sol_node()
want_cls = None
graph = json.loads((_REPO / "workflows" / SOL_GRAPH).read_text())
for node in graph.values():
    if str(node.get("class_type", "")).startswith("SolAttn"):
        want_cls = node["class_type"]
        break
check("sol_node_matches", want_cls == sol_class,
      f"graph wires {want_cls!r}, bench builds {sol_class!r}"
      if want_cls != sol_class else f"both {sol_class}")

want = node_inputs(SOL_GRAPH, want_cls) if want_cls else None
got = built_inputs(sol_class)
if want is None or got is None:
    check("sol_matches", False, "one side has no Sol node to compare")
elif want_cls != sol_class:
    check("sol_matches", False, "skipped: different nodes, values not comparable")
else:
    diff = {k: (want.get(k), got.get(k)) for k in set(want) | set(got)
            if want.get(k) != got.get(k)}
    check("sol_matches", not diff,
          f"differs: {diff} (graph, bench)" if diff else f"{len(want)} inputs agree")

print()
if failures:
    print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
    print("The bench and the shipped graphs disagree, so every arm measured "
          "here is\nagainst a configuration nobody runs.")
    raise SystemExit(1)
print("the bench measures what the graphs ship")
