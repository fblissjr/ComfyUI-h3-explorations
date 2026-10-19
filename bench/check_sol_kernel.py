#!/usr/bin/env python3
"""Check the installed `comfy_kitchen` still carries the Sol-Attn CUDA kernel.

This is the first check here that covers **a call INTO a dependency we do not
control** -- the gap no amount of local validation can close, because both
sides of any assertion we could write here come from our own tree. It does not close it either. It covers exactly one contract -- that
`comfy_kitchen.sol_attn` exists and accepts the arguments our node passes --
and nothing about whether the kernel is correct. `check_solattn_correctness.py`
is the check for that, and `smoke_h3.py` is still the only thing that submits.

Why it exists. `comfy_kitchen.sol_attn` used to ship only on kijai's unmerged
`sol_attn` branch. **It is upstream now** -- Comfy-Org/comfy-kitchen#117
(`dae00a1`) -- and by 0.2.33 the stock PyPI wheel carries the kernel, the
public `sol_attn_chunked` (#150) and token routing (#156), so the original
failure this file was built for, a stock wheel with no `sol_attn` at all,
cannot happen at that version. **This paragraph used to say the merged build
"ALSO declares version 0.2.31"**, and read the whole risk as three builds
sharing one version string.

The risk did not go away, it moved. A local build and the stock wheel still
declare the same `X.Y.Z` and only the local segment separates them, so a
`pip install -r requirements.txt --force-reinstall`, a Manager repair or a
fresh venv still swaps ours out silently. What that swap costs is now a
capability rather than the kernel: on 2026-09-08 exactly this happened, and
every case here passed on the stock wheel, because the six kwargs the node
passes on every call are upstream. The `observable` case is what was missing.

The original failure mode still applies to any version before the merge, and
its shape is why the file is written this way: a silent dense fallback
renders successfully, slower and numerically different, and nothing reports
it -- the same shape as the `frame_count` break of 2026-08-13.

## Presence and gating

Sol-Attn is enabled ON by default across all shipped video workflows
(`docs/attention/sol_attn.md` owns its knobs, and `bench/check_attention_defaults.py` owns
the exempt set -- the single-frame image workflows that used to omit it are
parked as of 2026-08-27 and ship no longer).

So presence is gated by "does a graph wire the node that needs *this
dependency*". `SolAttnPatch` (kijai's Triton pack) also does Sol-Attn and does
not touch `comfy_kitchen` at all, so it must not arm this check. Two nodes do:
`MiniMaxH3SolAttn`, ours, which every shipped graph wires since 2026-08-30,
and `SolAttnMiniMax`, the vendored upstream node those graphs wired before it.
**Either arms the gate**, because a graph left on the old id still renders and
still needs the kernel -- and that is precisely the graph where a silent dense
fallback would go unnoticed.

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
                     actually declares. A pinned key that no longer exists is a
                     hard error at execute, and a RENAMED key is worse: the pin
                     silently stops reaching the knob it names while the bench
                     arm keeps printing under the old name. Parsed from the
                     node file with `ast`, so this stays free of ComfyUI.
                     **This case earned itself on 2026-08-29**, when the merge
                     removed `centroid_tail` -- which this file's own header
                     had predicted. The node is ours now rather than upstream's,
                     so the drift it guards is our own edit rather than a drop
                     landing underneath us; that is a smaller risk, not no
                     risk, and the case costs nothing.

  no_triton_graphs   no shipped graph wires the Triton Sol node. Both nodes
                     are legal and both render, so a graph that drifted back
                     would run a different kernel silently while every pinned
                     setting and every number in this repo describes the CUDA
                     one. The generator derives the id from one constant, so
                     this catches a hand-edited graph or a stale regeneration.

  observable         an armed server reaches the render before anything says
                     the installed kernel cannot be observed. `blk_cnt` is
                     ours (not upstream), the route observer needs it, and
                     `sol_attn_h3.py::_require_kernel` refuses without it at
                     PATCH time -- which is after the model is staged, so the
                     render dies late instead of never starting. The
                     environment gates the assertion because the requirement
                     is real only when H3_SOL_OBSERVE asks for it; unarmed,
                     the case still names the capability, because a reader
                     comparing two Sol numbers needs to know whether the
                     build that produced them could be observed at all.

  signature          `comfy_kitchen.sol_attn` still accepts every kwarg the
                     node passes. **This is the case that cannot be replaced by
                     a runtime error**, because there is no runtime error: a
                     missing kwarg raises a TypeError inside `_run`, and
                     `make_override`'s `override` catches every exception and
                     falls through to dense. The render then succeeds, slower
                     and numerically different, and says nothing.

                     `sol_attn_h3.py::_require_kernel` asserts the same thing
                     at PATCH time, where an exception does reach the user.
                     Both exist deliberately: this one runs without CUDA, a
                     model or a server and tells you before you queue anything;
                     that one catches an install swapped out between this check
                     and the render.

  optional_bindings  every bound function in `comfy_kitchen.backends.cuda._C`
                     whose Python callers in `comfy_kitchen` can pass `None`
                     (every `std::optional` or nullable pointer parameter in
                     `dlpack_bindings.cpp`, including `sage_sdpa_prequantized`
                     and `sage_sdpa`'s `attn_mask`) declares `| None` in its
                     compiled nanobind signature. Earned 2026-10-18 when
                     rebuilding `0.2.38+sol` against `nanobind 2.10.2` compiled
                     bare `nb::arg("attn_mask")` bindings without `accepts_none`
                     (fixed in fork commit `997dd79`), crashing the dense
                     fallback at step 0 with `TypeError: ... Invoked with
                     types: ..., NoneType`. `optional_bindings_refusal` proves
                     the grader goes red on a signature missing `| None`.

  clean_build_gate   `vendor/rebuild_kernel.sh` wipes `$SRC/build` before
                     `uv build --wheel` whenever `nanobind`, `python` or
                     `cuda_arch` differs from `comfy_kitchen_build.json` (or
                     when the record or any of those keys is missing), prints
                     what changed and why it cleaned, and writes all three keys
                     to `comfy_kitchen_build.json`. `clean_build_gate_refusal`
                     proves the grader goes red if a toolchain key stops
                     triggering a clean.

Shown red: 2026-08-14, against the stock PyPI `comfy-kitchen==0.2.31` still
installed at the time, with `--require`. `present` failed, and `cuda_backend`
and `signature` reported skipped-for-cause rather than passing vacuously.
Re-run after the fork wheel was installed to confirm green for the right
reason. Both transcripts are in the session log.

`observable` shown red 2026-09-08, against the stock `comfy-kitchen==0.2.33`
that ComfyUI's pin had just installed over our build, with `H3_SOL_OBSERVE=1`
in the environment: FAIL, exit 1. The same run unarmed passed while printing
that the build does not take `blk_cnt`, which is the state the case is meant
to make legible rather than fail on. `--require-observe` forces the armed
assertion without setting the variable.

Needs neither CUDA nor a model nor a server, and imports no ComfyUI. Runs in
about a second.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.metadata
import inspect
import io
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# The node `SOL_CUDA_DEFAULTS` describes. Ours since 2026-08-30; it is a module
# in this pack, so there is one path and no symlink to chase.
#
# **The schema case went red the moment this still pointed at the vendored
# file**, which is the case doing its job rather than an inconvenience: the
# config had moved to `pooled_tail` and was being graded against a node that
# declares `centroid_tail`. A pin graded against the wrong node file is exactly
# the silent-drift failure this case exists for.
_NODE_PATHS = (_REPO / "sol_attn_h3.py",)
# `vendored_reference` and `node_version` were removed 2026-10-18: the read-only
# reference copy `vendor/sol_attn_minimax.py` was deleted in `2a478404`
# (2026-10-11), leaving both cases permanently skipped (`exit 2`).


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
#
# **Ours since 2026-08-30.** Both this and the vendored `SolAttnMiniMax` need
# `comfy_kitchen.sol_attn`, so either one in a graph should arm the presence
# case -- a graph left on the old node still renders and still needs the
# kernel. Listing both is not belt-and-braces: shipped graphs move to the new
# id in one regeneration, and a graph that did NOT move is exactly the state
# where a silent dense fallback would go unnoticed.
# `MiniMaxH3Sol` since the redesign (2026-09-27); `MiniMaxH3SolAttn` before it.
CUDA_SOL_NODE = "MiniMaxH3Sol"
VENDORED_SOL_NODE = "SolAttnMiniMax"

# The Triton node. Shipped graphs migrated off it on 2026-08-14 (`8a12646`)
# and must not drift back: both nodes are valid, both load, both render, and a
# graph wiring the Triton one would run a DIFFERENT KERNEL while every pinned
# setting, every doc and every measurement in this repo describes the CUDA one.
# Nothing else catches it -- the generator validates against /object_info,
# where SolAttnPatch is a perfectly legal node.
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
REQUIRED_KWARGS = ("tau", "scale", "sink_blocks", "sink_q", "topk_ratio",
                   "tail")
# **Empty since 2026-08-30, and kept as an empty tuple rather than deleted.**
# `sol_attn_h3.py` adapts to nothing: it asserts the signature once at patch
# time and refuses, because there is one supported kernel now. So every kwarg
# it passes is required and this list has nothing to hold. It stays because a
# future capability the node passes conditionally belongs here, and because an
# empty list is a visible statement that nothing is conditional -- deleting it
# would leave a reader unable to tell that from an oversight.
#
# What it used to hold, all removed from the kernel by comfy-kitchen#117:
# `centroid_tail`, `reuse_qkv_memory`, `max_blocks`.
OPTIONAL_KWARGS = ()

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
parser.add_argument("--require-observe", action="store_true",
                    help="assert the kernel takes blk_cnt even with "
                         "H3_SOL_OBSERVE unset (used to show that case red)")
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
# **This used to read `version == "0.2.31"`**, which stopped firing when
# ComfyUI moved its pin to 0.2.33 (core `18ebc2af`) -- the note went quiet
# on exactly the release it describes. The ambiguity was never about one
# number: any version with no local segment is a version the stock wheel
# also declares, so that is what is tested now and no upstream version
# number is copied into this file.
if "+" not in version and has_sol:
    print("       note        version is indistinguishable from the stock "
          "PyPI wheel; presence of sol_attn is the only signal.")

armed_by = sorted(set(graphs_wiring(CUDA_SOL_NODE))
                  | set(graphs_wiring(VENDORED_SOL_NODE)))
print(f"\nsol_attn presence (gate: graphs wiring {CUDA_SOL_NODE} or "
      f"{VENDORED_SOL_NODE}):")
if armed_by:
    check("present", has_sol,
          f"required by {len(armed_by)} graph(s): {', '.join(armed_by[:3])}"
          + (" ..." if len(armed_by) > 3 else ""))
elif args.require:
    check("present", has_sol, "--require given; no graph wires it")
else:
    skip("present", f"no shipped graph wires {CUDA_SOL_NODE} or "
                    f"{VENDORED_SOL_NODE}; absent is the expected state, not "
                    f"a failure")

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

        # **`centroid_tail_expressible` was RETIRED here on 2026-08-30.** It
        # checked that a config asking for `centroid_tail=False` could be
        # honoured by the installed kernel. Neither the kernel nor the node has
        # that argument now, so the case had no subject: it graded a key that
        # `SOL_RECOMMENDED_CUDA` no longer carries, against a signature that no
        # longer has it, and could only ever pass. The `schema` case below is
        # what catches a config pinning a knob the node does not declare.
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

print("\nthe route observer can be served by this build:")
# **This case earned itself on 2026-09-08**, when torch was upgraded, ComfyUI
# moved its pin to `comfy-kitchen==0.2.33` and the stock wheel replaced our
# build. Every case above passed on it: the six kwargs the node passes on
# every call are all upstream now. What the stock wheel does NOT have is
# `blk_cnt`, the count out-parameter the route observer reads, and nothing
# here noticed -- so the check reported a healthy install of a build that
# `sol_attn_h3.py::_require_kernel` refuses the moment H3_SOL_OBSERVE is set.
#
# The environment is the gate, because the requirement is real only when
# something asks for it. Armed, an absent `blk_cnt` is a hard failure: the
# node raises at PATCH time, after the model is staged, so the render dies
# late rather than never starting. Unarmed, a build without it is legal and
# renders correctly, and the case says so while naming the capability -- a
# reader comparing two Sol numbers needs to know which build answered.
observe_env = os.environ.get("H3_SOL_OBSERVE", "")
if not has_sol:
    skip("observable", "sol_attn absent; nothing to introspect")
else:
    gone = missing_params(ck.sol_attn, ("blk_cnt",))
    takes_blk_cnt = gone is not None and not gone
    if observe_env or args.require_observe:
        why = (f"H3_SOL_OBSERVE={observe_env!r}" if observe_env
               else "--require-observe given")
        check("observable", takes_blk_cnt,
              f"{why}, and the installed sol_attn takes blk_cnt"
              if takes_blk_cnt else
              f"{why}, but the installed sol_attn has NO blk_cnt: the route "
              f"cannot be observed and the node refuses at patch time. "
              f"Rebuild the kernel with vendor/rebuild_kernel.sh "
              f"or start the server without H3_SOL_OBSERVE.")
    else:
        check("observable", True,
              "H3_SOL_OBSERVE unset, so nothing here needs the count; the "
              "installed build " + ("does take blk_cnt, so an armed server "
                                    "would work" if takes_blk_cnt else
                                    "does NOT take blk_cnt, so arming the "
                                    "observer would be refused at patch time"))

def _parse_c_signature(doc):
    """Parse `(param_names, {param_name: type_and_default})` from a nanobind docstring's first line."""
    if not doc:
        return [], {}
    line = doc.splitlines()[0]
    lp, rp = line.find("("), line.rfind(")")
    if lp == -1 or rp == -1 or rp < lp:
        return [], {}
    inner = line[lp + 1:rp]
    params, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            params.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur and "".join(cur).strip():
        params.append("".join(cur).strip())
    names, by_name = [], {}
    for p in params:
        if ":" in p:
            k, v = p.split(":", 1)
            k = k.strip()
            names.append(k)
            by_name[k] = v.strip()
    return names, by_name


def _expr_can_be_none(expr, nullable_vars, nullable_funcs):
    if isinstance(expr, ast.Constant) and expr.value is None:
        return True
    if isinstance(expr, ast.Name) and expr.id in nullable_vars:
        return True
    if isinstance(expr, ast.IfExp):
        return (_expr_can_be_none(expr.body, nullable_vars, nullable_funcs)
                or _expr_can_be_none(expr.orelse, nullable_vars, nullable_funcs))
    if (isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in nullable_funcs):
        return True
    return False


def _discover_nullable_c_params(ck_mod, c_sigs):
    """Find every `(func_name, param_name)` where `comfy_kitchen` Python wrappers can pass `None` to `_C`."""
    nullable_c = {}
    pkg_dir = Path(ck_mod.__file__).resolve().parent
    for py in sorted(pkg_dir.rglob("*.py")):
        if any(p in py.parts for p in ("hip", "triton", "eager", "ascend")):
            continue
        try:
            tree = ast.parse(py.read_text())
        except (OSError, SyntaxError):
            continue
        nullable_funcs = set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for ret in ast.walk(fn):
                    if isinstance(ret, ast.Return) and ret.value is not None:
                        if _expr_can_be_none(ret.value, set(), set()):
                            nullable_funcs.add(fn.name)
                        elif (isinstance(ret.value, ast.Tuple)
                              and any(_expr_can_be_none(e, set(), set())
                                      for e in ret.value.elts)):
                            nullable_funcs.add(fn.name)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            nullable = set()
            all_args = fn.args.posonlyargs + fn.args.args
            n_def = len(fn.args.defaults)
            for i, a in enumerate(all_args):
                if i >= len(all_args) - n_def:
                    d = fn.args.defaults[i - (len(all_args) - n_def)]
                    if isinstance(d, ast.Constant) and d.value is None:
                        nullable.add(a.arg)
            for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
                if d is not None and isinstance(d, ast.Constant) and d.value is None:
                    nullable.add(a.arg)
            for _ in range(4):
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.FunctionDef) and sub is not fn:
                        for ret in ast.walk(sub):
                            if (isinstance(ret, ast.Return)
                                    and ret.value is not None
                                    and _expr_can_be_none(ret.value, nullable, nullable_funcs)):
                                nullable_funcs.add(sub.name)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if (isinstance(t, ast.Name)
                                    and _expr_can_be_none(sub.value, nullable, nullable_funcs)):
                                nullable.add(t.id)
                            elif isinstance(t, ast.Tuple):
                                if (isinstance(sub.value, ast.Tuple)
                                        and len(t.elts) == len(sub.value.elts)):
                                    for te, ve in zip(t.elts, sub.value.elts):
                                        if (isinstance(te, ast.Name)
                                                and _expr_can_be_none(ve, nullable, nullable_funcs)):
                                            nullable.add(te.id)
                                elif _expr_can_be_none(sub.value, nullable, nullable_funcs):
                                    for te in t.elts:
                                        if isinstance(te, ast.Name):
                                            nullable.add(te.id)
                    elif (isinstance(sub, ast.AnnAssign)
                          and isinstance(sub.target, ast.Name)
                          and sub.value is not None
                          and _expr_can_be_none(sub.value, nullable, nullable_funcs)):
                        nullable.add(sub.target.id)
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    is_c = ((isinstance(node.func.value, ast.Name)
                             and node.func.value.id == "_C")
                            or (isinstance(node.func.value, ast.Attribute)
                                and node.func.value.attr == "_C"))
                    if not is_c:
                        continue
                    cname = node.func.attr
                    pnames, _ = c_sigs.get(cname, ([], {}))
                    for idx, arg_expr in enumerate(node.args):
                        if (idx < len(pnames)
                                and _expr_can_be_none(arg_expr, nullable, nullable_funcs)):
                            nullable_c.setdefault(cname, set()).add(pnames[idx])
                    for kw in node.keywords:
                        if kw.arg and _expr_can_be_none(kw.value, nullable, nullable_funcs):
                            nullable_c.setdefault(cname, set()).add(kw.arg)
    return {k: sorted(v) for k, v in sorted(nullable_c.items())}


def _grade_optional_c_bindings(c_sigs, required_nullable):
    """Return `(bad, total_params)` for `required_nullable` against parsed `_C` signatures."""
    bad = []
    total = 0
    for fn_name, params in sorted(required_nullable.items()):
        _, by_name = c_sigs.get(fn_name, ([], {}))
        if not by_name:
            bad.append(f"{fn_name} (<missing signature>)")
            continue
        for p in params:
            total += 1
            t = by_name.get(p)
            if t is None:
                bad.append(f"{fn_name}.{p} (<missing param>)")
            elif "| None" not in t and "Optional[" not in t and "= None" not in t:
                bad.append(f"{fn_name}.{p}: {t}")
    return bad, total


print("\nevery optional parameter in comfy_kitchen.backends.cuda._C accepts None:")
cuda_mod = sys.modules.get("comfy_kitchen.backends.cuda")
if cuda_mod is None or not hasattr(cuda_mod, "_C"):
    skip("optional_bindings", "comfy_kitchen.backends.cuda._C unavailable")
else:
    _c_ext = cuda_mod._C
    _c_sigs = {
        name: _parse_c_signature(getattr(getattr(_c_ext, name), "__doc__", None))
        for name in dir(_c_ext)
        if not name.startswith("__")
    }
    _nullable_map = _discover_nullable_c_params(ck, _c_sigs)
    # Explicit floor on the attention and Sol entries our renders hit directly,
    # so a wrapper refactor cannot quietly shrink the checked set below them.
    for _fn, _ps in (
        ("sage_sdpa_prequantized", ("attn_mask",)),
        ("sage_sdpa", ("attn_mask",)),
        ("sol_attn", ("key_bias", "threshold", "block_len", "tau_map")),
        ("sol_attn_core", ("threshold", "block_len")),
        ("sol_producer_chunk", ("block_len",)),
    ):
        _cur = set(_nullable_map.get(_fn, ())) | set(_ps)
        _nullable_map[_fn] = sorted(_cur)
    _bad_opt, _total_opt = _grade_optional_c_bindings(_c_sigs, _nullable_map)
    check(
        "optional_bindings",
        not _bad_opt and _total_opt > 0,
        f"{_total_opt} optional parameter(s) across {len(_nullable_map)} _C "
        f"function(s) accept None (sage_sdpa_prequantized.attn_mask, "
        f"sage_sdpa.attn_mask, sol_attn.* included)"
        if not _bad_opt and _total_opt > 0
        else f"C++ binding(s) in comfy_kitchen.backends.cuda._C reject None "
        f"(nanobind 2.10.x requires .none() or = nb::none() on std::optional "
        f"nb::arg bindings in dlpack_bindings.cpp): {'; '.join(_bad_opt)}",
    )
    # Refusal control: strip `| None` from sage_sdpa_prequantized.attn_mask (the
    # exact signature produced by bare `nb::arg("attn_mask")` before 997dd79)
    # and prove the grader reports it by name.
    _mut_names, _mut_by_name = _c_sigs.get("sage_sdpa_prequantized", ([], {}))
    _mut_sigs = dict(_c_sigs)
    _mut_copy = dict(_mut_by_name)
    _mut_copy["attn_mask"] = "ndarray[device='cuda']"
    _mut_sigs["sage_sdpa_prequantized"] = (_mut_names, _mut_copy)
    _refusal_bad, _ = _grade_optional_c_bindings(_mut_sigs, _nullable_map)
    check(
        "optional_bindings_refusal",
        _refusal_bad == ["sage_sdpa_prequantized.attn_mask: ndarray[device='cuda']"],
        f"stripped '| None' caught as {_refusal_bad[0]}"
        if _refusal_bad == ["sage_sdpa_prequantized.attn_mask: ndarray[device='cuda']"]
        else f"expected ['sage_sdpa_prequantized.attn_mask: ndarray[device=\\'cuda\\']'], got {_refusal_bad}",
    )


def _grade_rebuild_clean_gate(script_text):
    """Verify `vendor/rebuild_kernel.sh` cleans `$SRC/build` on toolchain/arch changes and records all three keys."""
    errors = []
    clean_tag = "<<'PYEOF_CLEAN'\n"
    clean_end = "\nPYEOF_CLEAN"
    build_cmd = "uv build --wheel"
    c_start = script_text.find(clean_tag)
    c_end = script_text.find(clean_end, c_start + len(clean_tag)) if c_start != -1 else -1
    b_pos = script_text.find(build_cmd)
    if c_start == -1 or c_end == -1:
        return ["missing <<'PYEOF_CLEAN' block in vendor/rebuild_kernel.sh"]
    if b_pos == -1 or c_end > b_pos:
        errors.append("PYEOF_CLEAN block does not run before `uv build --wheel`")

    rec_tag = "<<'PYEOF'\n"
    rec_end = "\nPYEOF"
    r_start = script_text.find(rec_tag)
    r_end = script_text.find(rec_end, r_start + len(rec_tag)) if r_start != -1 else -1
    if r_start == -1 or r_end == -1:
        errors.append("missing <<'PYEOF' record writer block in vendor/rebuild_kernel.sh")
    else:
        rec_code = script_text[r_start + len(rec_tag):r_end]
        for key in ("cuda_arch", "python", "nanobind"):
            if f'"{key}":' not in rec_code:
                errors.append(f"comfy_kitchen_build.json writer omits {key!r}")

    clean_code = script_text[c_start + len(clean_tag):c_end]
    try:
        compiled = compile(clean_code, "vendor/rebuild_kernel.sh::PYEOF_CLEAN", "exec")
    except SyntaxError as exc:
        return errors + [f"PYEOF_CLEAN syntax error: {exc}"]

    scenarios = (
        (
            "unchanged",
            {"cuda_arch": "89", "python": "3.14.2", "nanobind": "2.10.2"},
            ("89", "3.14.2", "2.10.2"),
            False,
            ("unchanged", "keeping"),
        ),
        (
            "missing_keys",
            {"cuda_arch": "89"},
            ("89", "3.14.2", "2.10.2"),
            True,
            ("cleaned", "python: <missing> -> 3.14.2", "nanobind: <missing> -> 2.10.2", "stale CMake/ninja"),
        ),
        (
            "nanobind_diff",
            {"cuda_arch": "89", "python": "3.14.2", "nanobind": "2.9.2"},
            ("89", "3.14.2", "2.10.2"),
            True,
            ("cleaned", "nanobind: 2.9.2 -> 2.10.2", "stale CMake/ninja"),
        ),
        (
            "python_diff",
            {"cuda_arch": "89", "python": "3.13.9", "nanobind": "2.10.2"},
            ("89", "3.14.2", "2.10.2"),
            True,
            ("cleaned", "python: 3.13.9 -> 3.14.2", "stale CMake/ninja"),
        ),
        (
            "cuda_arch_diff",
            {"cuda_arch": "80", "python": "3.14.2", "nanobind": "2.10.2"},
            ("89", "3.14.2", "2.10.2"),
            True,
            ("cleaned", "cuda_arch: 80 -> 89", "stale CMake/ninja"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for label, prev_rec, cur_tuple, expect_cleaned, expected_substrings in scenarios:
            case_dir = tmp_path / label
            src_dir = case_dir / "fork"
            build_dir = src_dir / "build"
            build_dir.mkdir(parents=True)
            (build_dir / "stale.o").write_text("stale")
            rec_path = case_dir / "comfy_kitchen_build.json"
            if prev_rec is not None:
                rec_path.write_text(json.dumps(prev_rec))
            buf = io.StringIO()
            saved_argv = sys.argv
            try:
                sys.argv = ["PYEOF_CLEAN", str(rec_path), str(src_dir), *cur_tuple]
                with contextlib.redirect_stdout(buf):
                    exec(compiled, {"__name__": "__main__"})
            except Exception as exc:
                errors.append(f"{label}: raised {type(exc).__name__}: {exc}")
                continue
            finally:
                sys.argv = saved_argv
            out = buf.getvalue().strip()
            was_cleaned = not build_dir.exists()
            if was_cleaned != expect_cleaned:
                errors.append(
                    f"{label}: {'did not remove' if expect_cleaned else 'unexpectedly removed'} build/ (output: {out!r})"
                )
            for sub in expected_substrings:
                if sub not in out:
                    errors.append(f"{label}: output missing {sub!r} (got {out!r})")
    return errors


print("\nvendor/rebuild_kernel.sh cleans build/ on nanobind, python or cuda_arch change:")
_rebuild_sh = _REPO / "vendor" / "rebuild_kernel.sh"
if not _rebuild_sh.is_file():
    check("clean_build_gate", False, "vendor/rebuild_kernel.sh not found")
else:
    _sh_text = _rebuild_sh.read_text()
    _clean_errs = _grade_rebuild_clean_gate(_sh_text)
    check(
        "clean_build_gate",
        not _clean_errs,
        "keeps build/ when cuda_arch, python and nanobind match comfy_kitchen_build.json; "
        "wipes build/ and prints what changed and why on missing keys or any mismatch"
        if not _clean_errs
        else f"vendor/rebuild_kernel.sh clean-build gate failed: {'; '.join(_clean_errs)}",
    )
    # Refusal control: skip nanobind in the comparison predicate and prove the
    # grader catches that a nanobind upgrade no longer cleans build/.
    _mut_sh = _sh_text.replace("if prev.get(k) != v", 'if k != "nanobind" and prev.get(k) != v')
    _mut_errs = _grade_rebuild_clean_gate(_mut_sh)
    _want_prefix = "nanobind_diff: did not remove build/"
    _caught = [e for e in _mut_errs if e.startswith(_want_prefix)]
    check(
        "clean_build_gate_refusal",
        bool(_caught),
        f"omitted nanobind check caught as {_caught[0].split(' (output:')[0]}"
        if _caught
        else f"expected error starting with {_want_prefix!r}, got {_mut_errs}",
    )

print("\nno shipped graph wires the Triton node:")
triton_graphs = graphs_wiring(TRITON_SOL_NODE)
check("no_triton_graphs", not triton_graphs,
      f"{len(triton_graphs)} graph(s) wire {TRITON_SOL_NODE}: "
      f"{', '.join(triton_graphs[:3])}" + (" ..." if len(triton_graphs) > 3 else "")
      if triton_graphs else
      f"0 of {len(graph_paths(_REPO / 'workflows'))} graphs")

print("\nSOL_CUDA_DEFAULTS pins only knobs the node declares:")
node_file = next((p for p in _NODE_PATHS if p.is_file()), None)
if node_file is None:
    skip("schema", f"node file not found: {[p.name for p in _NODE_PATHS]}")
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
