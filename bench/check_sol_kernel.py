#!/usr/bin/env python3
"""Check the installed `comfy_kitchen` still carries the Sol-Attn CUDA kernel.

This is the first check here that covers **a call INTO a dependency we do not
control** -- the gap no amount of local validation can close, because both
sides of any assertion we could write here come from our own tree. It does not close it either. It covers exactly one contract -- that
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

## Presence and gating

Sol-Attn is enabled ON by default across all shipped video workflows
(`docs/SOLATTN.md` owns its knobs, and `bench/check_attention_defaults.py` owns
the exempt set -- the single-frame image workflows that used to omit it are
parked as of 2026-08-27 and ship no longer).

So presence is gated by "does a graph wire the node that needs *this
dependency*". `SolAttnPatch` (kijai's Triton pack) also does Sol-Attn and does
not touch `comfy_kitchen` at all, so it must not arm this check. Only
`SolAttnMiniMax` does. Because shipped video graphs wire `SolAttnMiniMax`,
the presence of `comfy_kitchen.sol_attn` is actively verified.

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

  no_triton_graphs   no shipped graph wires the Triton Sol node. Both nodes
                     are legal and both render, so a graph that drifted back
                     would run a different kernel silently while every pinned
                     setting and every number in this repo describes the CUDA
                     one. The generator derives the id from one constant, so
                     this catches a hand-edited graph or a stale regeneration.

  vendored           the file ComfyUI loads is the one this repo tracks. Before
                     2026-08-14 three untracked copies of this node existed on
                     one box and nothing could say which was running; a
                     measurement is meaningless if the code under it is
                     unidentified.
  node_version       the tracked file's sha256 is a version we have named and
                     dated. Upstream publishes this node through conversation
                     rather than a repository, so an unrecorded hash FAILS
                     rather than warns -- it forces the drop to be recorded in
                     vendor/README.md before it can be run.

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

# Where the node file may be. Since 2026-08-14 it is VENDORED into this repo
# (vendor/README.md) and the installed path is a symlink into it, so the
# tracked file and the running file cannot diverge.
_NODE_PATHS = (
    _REPO / "vendor" / "sol_attn_minimax.py",
    _REPO.parent / "ComfyUI-SolAttn-cuda" / "sol_attn_minimax.py",
)

# Where ComfyUI loads the node from. Should be a symlink INTO vendor/, so the
# tracked file and the running file cannot diverge -- see vendor/README.md.
_INSTALLED = _REPO.parent / "ComfyUI-SolAttn-cuda" / "sol_attn_minimax.py"
_VENDORED = _REPO / "vendor" / "sol_attn_minimax.py"

# sha256 -> label. Upstream publishes this file through conversation, not a
# repository, so provenance is by hand and an unrecorded hash is a FAILURE
# rather than a warning: it forces a version to be named and dated before it
# can be run, which is exactly what was missing when three untracked copies
# existed on this box and none was authoritative.
KNOWN_NODE_VERSIONS = {
    "3a5f0051fce61d9da1a0b1aaaf03bc16af654d7be59a929bcde395a058918d73":
        "v1 (2026-08-14 10:48) -- sink_q = whole conditioning range",
    "d856ba83557d18fbe642011e7a101f597cceea75fcf2e9d600ae064d062de526":
        "v2 (2026-08-14 14:19) -- sink_q narrowed to the target audio span",
    "7805cf3706bf9b9123932e66f1dd311c3f005b6a1c188d40cc5a23321debc0dd":
        "v3 (2026-08-22) -- tau/tau_profile folded into a `selection` "
        "DynamicCombo alongside top-k, routed_cap_percent dropped; needs a "
        "kernel with topk_ratio (0.2.31+sol.23d1a66 or later)",
    "1c55a4b51011041a03e62ed73458c9ce280ffd8ca6fc5f353b2806d978504ac1":
        "v3.1 (2026-08-29) -- OURS, not upstream's: the kernel call reads "
        "`sol_attn`'s signature and passes `centroid_tail` and "
        "`reuse_qkv_memory` only where they exist, so one node drives both "
        "kijai's branch build and the merged upstream (#117, dae00a1) which "
        "dropped them. `centroid_tail=False` raises rather than being "
        "silently ignored, because the merged kernel always evaluates the "
        "tail at the centroid and swallowing the request would change the "
        "math without saying so",
}


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

# The Triton node. Shipped graphs migrated off it on 2026-08-14 (`8a12646`)
# and must not drift back: both nodes are valid, both load, both render, and a
# graph wiring the Triton one would run a DIFFERENT KERNEL while every pinned
# setting, every doc and every measurement in this repo describes the CUDA one.
# Nothing else catches it -- check_workflow_schema.py validates against
# /object_info, where SolAttnPatch is a perfectly legal node.
TRITON_SOL_NODE = "SolAttnPatch"

# What `internal/refs/sol_attn_minimax.py` actually passes. `_run()` calls the
# registry entry for the normal path and reaches into `backends.cuda` directly
# for the reuse path, so the two are checked separately.
# **Split into required and optional on 2026-08-29, when the prediction in
# this file's own header came true.** It said upstream was "weighing making
# `centroid_tail` unconditional, which would remove it". Sol-Attn merged as
# Comfy-Org/comfy-kitchen#117 (dae00a1) and that is exactly what happened:
# `centroid_tail`, `reuse_qkv_memory` and `max_blocks` are gone from both
# entries, and `tail`, `block_len` and `coarse_gate` arrived.
#
# REQUIRED is what the node passes on EVERY call, so a build missing any of
# these cannot render at all. OPTIONAL is what it passes only when the kernel
# takes it -- `vendor/sol_attn_minimax.py::_kernel_kwargs` reads the signature
# and adapts, so their absence is a capability difference rather than a defect.
# Reported either way, because "which build is installed" is the first thing
# anyone debugging a Sol number needs and both builds call themselves 0.2.31.
REQUIRED_KWARGS = ("tau", "scale", "sink_blocks", "sink_q", "topk_ratio")
OPTIONAL_KWARGS = ("centroid_tail", "reuse_qkv_memory", "max_blocks")

failures = []
skipped = []

# Graph discovery, shared with every other walker. A bare non-recursive glob
# here stops covering any directory `GRAPH_DIRS` routes to -- demonstrated by
# `workflows/image/` between 2026-08-16 and 2026-08-27 -- while the "no shipped
# graph wires the Triton node" case still prints a confident count. See
# h3_config.GRAPH_DIRS.
sys.path.insert(0, str(_REPO / "workflows"))
from h3_config import graph_paths, SOL_RECOMMENDED_CUDA  # noqa: E402


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
    for path in graph_paths(_REPO / "workflows"):
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
    gone = missing_params(ck.sol_attn, REQUIRED_KWARGS)
    if gone is None:
        check("signature", False, "comfy_kitchen.sol_attn is not introspectable")
    else:
        check("signature", not gone,
              f"registry entry missing {gone}, which the node passes on EVERY "
              f"call -- no render can succeed" if gone
              else f"registry entry accepts all {len(REQUIRED_KWARGS)} required kwargs")
        have = [k for k in OPTIONAL_KWARGS
                if not missing_params(ck.sol_attn, (k,))]
        print(f"        optional present: {have or 'none'}; "
              f"absent: {[k for k in OPTIONAL_KWARGS if k not in have] or 'none'}")

        # The one place an absent optional is NOT merely a capability
        # difference. `centroid_tail=False` is a different computation, and a
        # build without the kwarg evaluates the tail at the centroid
        # unconditionally -- so a config asking for False cannot be honoured.
        # The node raises rather than silently ignoring it; this says so before
        # a render does.
        if "centroid_tail" not in have and not SOL_RECOMMENDED_CUDA.get(
                "centroid_tail", True):
            check("centroid_tail_expressible", False,
                  "h3_config asks for centroid_tail=False and this kernel has "
                  "no such argument -- the merged build always evaluates the "
                  "tail at the query block's centroid. Every Sol call would "
                  "raise.")
        else:
            check("centroid_tail_expressible", True,
                  "the shipped centroid_tail is what this kernel can do")
    cuda = sys.modules.get("comfy_kitchen.backends.cuda")
    if cuda is None or not hasattr(cuda, "sol_attn"):
        skip("signature_cuda", "backends.cuda.sol_attn unavailable")
    else:
        gone = missing_params(cuda.sol_attn, REQUIRED_KWARGS)
        if gone is None:
            check("signature_cuda", False, "not introspectable")
        else:
            check("signature_cuda", not gone,
                  f"direct CUDA entry missing {gone}" if gone
                  else f"direct CUDA entry accepts all required kwargs")

import hashlib

print("\nno shipped graph wires the Triton node:")
triton_graphs = graphs_wiring(TRITON_SOL_NODE)
check("no_triton_graphs", not triton_graphs,
      f"{len(triton_graphs)} graph(s) wire {TRITON_SOL_NODE}: "
      f"{', '.join(triton_graphs[:3])}" + (" ..." if len(triton_graphs) > 3 else "")
      if triton_graphs else
      f"0 of {len(graph_paths(_REPO / 'workflows'))} graphs")

print("\nthe node ComfyUI loads is the one this repo tracks:")
if not _VENDORED.is_file():
    skip("vendored", f"no vendored copy at {_VENDORED}")
    skip("node_version", "nothing to hash")
else:
    digest = hashlib.sha256(_VENDORED.read_bytes()).hexdigest()
    if not _INSTALLED.exists():
        skip("vendored", f"node not installed at {_INSTALLED}")
    else:
        same = _INSTALLED.resolve() == _VENDORED.resolve()
        how = "symlink" if _INSTALLED.is_symlink() else "copy"
        if not same and _INSTALLED.is_file():
            same = hashlib.sha256(_INSTALLED.read_bytes()).hexdigest() == digest
        check("vendored", same,
              f"installed is a {how} of the tracked file"
              if same else
              f"{_INSTALLED} differs from {_VENDORED} -- the running node is "
              f"not the tracked one")
    label = KNOWN_NODE_VERSIONS.get(digest)
    check("node_version", label is not None,
          label if label else
          f"sha256 {digest[:16]}... is not a recorded version; add it to "
          f"KNOWN_NODE_VERSIONS and vendor/README.md before running it")

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
