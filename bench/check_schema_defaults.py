#!/usr/bin/env python3
"""Check every node's schema defaults match its `execute` signature defaults.

ComfyUI does **not** inject a schema default for an input the prompt omits.
The `default=` in `io.Schema` populates the widget when a node is dropped in
the UI; the value that actually reaches `execute` for an absent input is the
Python signature default. So the two are independent, and when they drift the
UI and the API path silently disagree.

That is not hypothetical. `MiniMaxH3KeyframeCanvas.length` moved to 124 in the
schema on 2026-08-13 while `execute` kept `length=0`, so a graph opened in the
browser was fixed and an API prompt that omitted the field still emitted 0 and
rendered a 5-frame, 0.208-second clip. The API path is how `bench/*` drives
renders, i.e. the split was on exactly the path the numbers come from.

Claims, i.e. what breaks if a case is deleted:
  defaults agree        every optional input that has a schema default has the
                        same value as its `execute` parameter. This is the
                        whole file
  every node is seen    the walk actually found this pack's nodes. Without it
                        an import failure or a renamed entry point turns the
                        check into a silent pass over an empty list
  required stay required an input with no schema default must not have a
                        signature default either -- that would make a required
                        input quietly optional on the API path

No CUDA, no model. Needs ComfyUI importable for the node modules to load.

    PYTHONPATH=/path/to/ComfyUI python bench/check_schema_defaults.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Order matters, and getting it wrong is not subtle. `custom_nodes/` has to be
# on the path so the package imports by name, but the repo directory itself
# must NOT be: it contains a `nodes.py`, and ComfyUI's own
# `comfy_extras/nodes_minimax_h3` does a bare `import nodes` expecting
# ComfyUI's. Put the ComfyUI root ahead of both so that import resolves there.
sys.path.insert(0, str(REPO.parent))          # custom_nodes/, for the package
sys.path.insert(0, str(REPO.parents[1]))      # ComfyUI root, must win

# Loading schemas must not select the GPU. The node modules import Comfy's
# model-management layer for runtime helpers even though this check never
# executes a node.
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True

_PKG = REPO.name

_SENTINEL = object()


def _schema_defaults(node):
    """{input_id: default} for inputs that declare one."""
    out = {}
    for spec in node.define_schema().inputs:
        default = getattr(spec, "default", _SENTINEL)
        if default is not _SENTINEL:
            out[spec.id] = default
    return out


def _signature_defaults(node):
    """{param: default} for `execute` params that declare one."""
    fn = node.execute
    fn = fn.__func__ if hasattr(fn, "__func__") else fn
    sig = inspect.signature(fn)
    return {name: p.default for name, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty}


def _nodes():
    """Every io.ComfyNode this pack registers, via its own entry point."""
    import importlib

    pkg = importlib.import_module(_PKG)
    ext = None
    for attr in ("comfy_entrypoint", "NODES_LIST"):
        if hasattr(pkg, attr):
            ext = getattr(pkg, attr)
            break
    if ext is None:
        raise AssertionError(
            f"{_PKG} exposes neither comfy_entrypoint nor NODES_LIST; this "
            "check cannot find the nodes and would otherwise pass over none")

    import asyncio

    def _resolve(value):
        return asyncio.run(value) if inspect.iscoroutine(value) else value

    extension = _resolve(ext())
    return list(_resolve(extension.get_node_list()))


def main():
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("schema defaults against execute() signature defaults")

    nodes = _nodes()

    def every_node_is_seen():
        assert nodes, "no nodes found; this check would pass over an empty list"
        print(f"        ({len(nodes)} node(s): "
              f"{', '.join(n.define_schema().node_id for n in nodes)})")

    def defaults_agree():
        bad = []
        for node in nodes:
            nid = node.define_schema().node_id
            sch, sig = _schema_defaults(node), _signature_defaults(node)
            for key, want in sch.items():
                if key not in sig:
                    continue          # positional/required in execute; fine
                if sig[key] != want:
                    bad.append(f"{nid}.{key}: schema={want!r} execute={sig[key]!r}")
        assert not bad, (
            "schema and execute disagree, so the UI and the API path see "
            "different values:\n         " + "\n         ".join(bad))

    def required_stay_required():
        bad = []
        for node in nodes:
            schema = node.define_schema()
            nid = schema.node_id
            sig = _signature_defaults(node)
            for spec in schema.inputs:
                has_default = getattr(spec, "default", _SENTINEL) is not _SENTINEL
                optional = bool(getattr(spec, "optional", False))
                if not has_default and not optional and spec.id in sig:
                    bad.append(f"{nid}.{spec.id} is required in the schema but "
                               f"defaults to {sig[spec.id]!r} in execute")
        assert not bad, "\n         ".join(bad)

    check("every node is seen", every_node_is_seen)
    check("schema and execute defaults agree", defaults_agree)
    check("required inputs have no signature default", required_stay_required)

    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- the UI and the API path see the same defaults")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
