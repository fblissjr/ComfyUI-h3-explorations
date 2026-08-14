#!/usr/bin/env python3
"""Check the installed `comfy_kitchen` still carries the Sol-Attn CUDA kernel.

This is the first check here that covers **a call INTO a dependency we do not
control**, which CLAUDE.md names as the gap no amount of local validation can
close. It does not close it either. It covers exactly one contract -- that
`comfy_kitchen.sol_attn` exists and accepts the arguments our node passes --
and nothing about whether the kernel is correct. `check_solattn_correctness.py`
is the check for that, and `smoke_h3.py` is still the only thing that submits.

Why it exists. `comfy_kitchen.sol_attn` ships only on kijai's fork, branch
`sol_attn`, which is unmerged and publishes no wheel. The build we install
from that branch declares version `0.2.31` -- byte-identical to the PyPI
version ComfyUI pins in `requirements.txt`. So the fork build and the stock
wheel are indistinguishable to `pip list`, and a `pip install -r
requirements.txt --force-reinstall`, a Manager repair, or a fresh venv
silently swaps the stock wheel back in. The node then falls back to dense on
every call. **That failure renders successfully.** It is slower and
numerically different and nothing reports it, which is the same shape as the
`frame_count` break of 2026-08-13.

## Absent is not the same as broken

Sol-Attn is opt-in and shipped OFF (see CLAUDE.md). An install with no
`sol_attn` is the *expected* state for anyone who has not built the fork, so
asserting its presence unconditionally would report red on a correct machine
-- the third-case trap that produced five bugs in one session.

So presence is gated, and the gate is deliberately narrow: it is not "does a
graph use Sol-Attn" but "does a graph wire the node that needs *this
dependency*". `SolAttnPatch` (kijai's Triton pack) also does Sol-Attn and does
not touch `comfy_kitchen` at all, so it must not arm this check. Only
`SolAttnMiniMax` does.

As of writing, no shipped graph wires `SolAttnMiniMax`, so the gated case
SKIPS and this script exits 2 rather than 0 -- following
`check_distill_settings.py`, because a check that silently did not run reads
exactly like a check that passed. Pass `--require` to assert presence anyway;
that is also how the case was shown red.

Claims, i.e. what breaks if a case is deleted:

  installed          nothing is asserted; deleting it only removes the state
                     dump. Kept because every other case's verdict is
                     unreadable without knowing which build answered.
  present            a stock-wheel `comfy_kitchen` gets swapped in under a
                     graph that needs the fork, and every Sol call silently
                     falls back to dense. Renders fine, no error, wrong
                     numbers.
  cuda_backend       an install carrying ONLY the eager reference passes as
                     healthy. The eager path is O(T^2), materialises the full
                     score tensor and refuses past 4 GiB, so it cannot run at
                     H3's real sequence length -- it would raise mid-render,
                     not degrade.
  schema             every key in `SOL_CUDA_DEFAULTS` is an input the node
                     actually declares. The node is upstream's and its inputs
                     are expected to change -- upstream is weighing making
                     `centroid_tail` unconditional, which would remove it. A
                     pinned key that no longer exists is a hard error at
                     execute, and a key upstream RENAMES is worse: the pin
                     silently stops reaching the knob it names while the bench
                     arm keeps printing under the old name. Parsed from the
                     node file with `ast`, so this stays free of ComfyUI.

  signature          our node calls `sol_attn` with `centroid_tail` and the
                     direct CUDA entry with `reuse_qkv_memory`. Both arrived
                     on the branch within days of each other, and the branch
                     rebases. A rename lands as a TypeError inside the
                     override, which `make_override` catches and converts to a
                     silent dense fallback -- so the kwarg going away is
                     invisible at render time by construction.

Shown red: 2026-08-14, against the stock PyPI `comfy-kitchen==0.2.31` still
installed at the time, with `--require`. `present` failed, and `cuda_backend`
and `signature` reported skipped-for-cause rather than passing vacuously.
Re-run after the fork wheel was installed to confirm green for the right
reason. Both transcripts are in the session log.

Needs neither CUDA nor a model nor a server, and imports no ComfyUI. Runs in
about a second.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Where the node file may be. It is upstream's and lives OUTSIDE this repo on
# purpose (see custom_nodes/ComfyUI-SolAttn-cuda/README.md); the reference copy
# under internal/ is the fallback for a checkout that has not installed it.
_NODE_PATHS = (
    _REPO.parent / "ComfyUI-SolAttn-cuda" / "sol_attn_minimax.py",
    _REPO / "internal" / "refs" / "sol_attn_minimax.py",
)


def declared_inputs(path):
    """Input ids the node file declares, via `io.<Type>.Input("name", ...)`.

    Parsed rather than imported: importing it drags in comfy_api and would
    make this check need a ComfyUI on sys.path, which nothing else here does.
    """
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Input"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            found.append(node.args[0].value)
    return found

# The node that needs THIS dependency. `SolAttnPatch` is kijai's Triton pack
# and does not touch comfy_kitchen, so it deliberately does not arm the gate.
CUDA_SOL_NODE = "SolAttnMiniMax"

# What `internal/refs/sol_attn_minimax.py` actually passes. `_run()` calls the
# registry entry for the normal path and reaches into `backends.cuda` directly
# for the reuse path, so the two are checked separately.
REGISTRY_KWARGS = ("tau", "scale", "sink_blocks", "sink_q", "max_blocks",
                   "centroid_tail")
CUDA_KWARGS = REGISTRY_KWARGS + ("reuse_qkv_memory",)

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def skip(name, why):
    print(f"  SKIP {name}   {why}")
    skipped.append(name)


def graphs_wiring(node_id):
    """Shipped graphs that wire `node_id`, in either the API or the UI form."""
    out = []
    for path in sorted((_REPO / "workflows").glob("*.json")):
        try:
            graph = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        # API form: {id: {"class_type": ...}}. UI form: {"nodes": [{"type": ...}]}
        types = set()
        if isinstance(graph, dict):
            for node in graph.get("nodes", []) or []:
                if isinstance(node, dict) and "type" in node:
                    types.add(node["type"])
            for node in graph.values():
                if isinstance(node, dict) and "class_type" in node:
                    types.add(node["class_type"])
        if node_id in types:
            out.append(path.name)
    return out


def missing_params(func, wanted):
    """Which of `wanted` this callable does not accept."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None            # not introspectable; caller decides
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []              # **kwargs swallows anything; nothing to prove
    return [name for name in wanted if name not in params]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require", action="store_true",
                    help="assert sol_attn is present even when no graph wires "
                         "SolAttnMiniMax (used to show this check red)")
args = parser.parse_args()

print("installed comfy_kitchen:")
try:
    ck = importlib.import_module("comfy_kitchen")
except Exception as exc:
    print(f"  SKIP installed   comfy_kitchen is not importable: {exc}")
    print("\ncomfy_kitchen is a hard dependency of ComfyUI core "
          "(comfy/quant_ops.py, comfy/float.py, comfy/ldm/modules/attention.py).")
    print("Nothing here can run. Exit 2 -- undetermined, not passing.")
    raise SystemExit(2)

try:
    version = importlib.metadata.version("comfy-kitchen")
except importlib.metadata.PackageNotFoundError:
    version = "<no dist metadata>"
has_sol = hasattr(ck, "sol_attn")
print(f"  ok   installed   version {version}, "
      f"from {getattr(ck, '__file__', '?')}")
print(f"       sol_attn    {'present' if has_sol else 'ABSENT'}")
if version == "0.2.31" and has_sol:
    print("       note        version is indistinguishable from the stock "
          "PyPI wheel; presence of sol_attn is the only signal.")

armed_by = graphs_wiring(CUDA_SOL_NODE)
print(f"\nsol_attn presence (gate: graphs wiring {CUDA_SOL_NODE}):")
if armed_by:
    check("present", has_sol,
          f"required by {len(armed_by)} graph(s): {', '.join(armed_by[:3])}"
          + (" ..." if len(armed_by) > 3 else ""))
elif args.require:
    check("present", has_sol, "--require given; no graph wires it")
else:
    skip("present", f"no shipped graph wires {CUDA_SOL_NODE}; absent is the "
                    "expected state, not a failure")

print("\nthe CUDA backend carries it, not only the eager reference:")
if not has_sol:
    skip("cuda_backend", "sol_attn absent; nothing to introspect")
else:
    try:
        cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    except Exception as exc:
        cuda = None
        check("cuda_backend", False, f"backends.cuda not importable: {exc}")
    if cuda is not None:
        check("cuda_backend", hasattr(cuda, "sol_attn"),
              "eager-only builds are O(T^2) and refuse past 4 GiB, so they "
              "cannot run at H3 sequence length")

print("\nthe signature still accepts what our node passes:")
if not has_sol:
    skip("signature", "sol_attn absent; nothing to introspect")
else:
    gone = missing_params(ck.sol_attn, REGISTRY_KWARGS)
    if gone is None:
        check("signature", False, "comfy_kitchen.sol_attn is not introspectable")
    else:
        check("signature", not gone,
              f"registry entry missing {gone}" if gone
              else f"registry entry accepts {len(REGISTRY_KWARGS)} kwargs")
    cuda = sys.modules.get("comfy_kitchen.backends.cuda")
    if cuda is None or not hasattr(cuda, "sol_attn"):
        skip("signature_cuda", "backends.cuda.sol_attn unavailable")
    else:
        gone = missing_params(cuda.sol_attn, CUDA_KWARGS)
        if gone is None:
            check("signature_cuda", False, "not introspectable")
        else:
            check("signature_cuda", not gone,
                  f"direct CUDA entry missing {gone}" if gone
                  else f"direct CUDA entry accepts {len(CUDA_KWARGS)} kwargs")

print("\nSOL_CUDA_DEFAULTS pins only knobs the node declares:")
node_file = next((p for p in _NODE_PATHS if p.is_file()), None)
if node_file is None:
    skip("schema", f"node file not found in {[str(p) for p in _NODE_PATHS]}")
else:
    sys.path.insert(0, str(_REPO / "workflows"))
    try:
        from h3_config import SOL_CUDA_DEFAULTS
    except Exception as exc:
        SOL_CUDA_DEFAULTS = None
        check("schema", False, f"cannot import SOL_CUDA_DEFAULTS: {exc}")
    if SOL_CUDA_DEFAULTS is not None:
        declared = declared_inputs(node_file)
        orphan = sorted(set(SOL_CUDA_DEFAULTS) - set(declared))
        # `model` is a wired socket and `tau_profile` is force_input, so
        # neither can carry a widget value -- both are correctly absent.
        widgetless = sorted(set(declared) - set(SOL_CUDA_DEFAULTS))
        check("schema", not orphan,
              f"pins {orphan}, which {node_file.name} does not declare"
              if orphan else
              f"{len(SOL_CUDA_DEFAULTS)} pinned, all declared "
              f"(not pinned, by design: {widgetless})")

print()
if failures:
    print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
    raise SystemExit(1)
if skipped:
    # Exit 2, not 0. A check that did not run must not read as one that passed.
    print(f"INCOMPLETE: {len(skipped)} case(s) skipped: {', '.join(skipped)}")
    raise SystemExit(2)
print("all cases passed")
