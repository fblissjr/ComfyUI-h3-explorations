#!/usr/bin/env python3
"""Guard the one rule that matters, against a baseline the schema cannot move.

`docs/comfy_notes.md` states it: **never rename a node's `node_id=`, and never
insert an input or output anywhere but the end.** (It lived in `CLAUDE.md` until
`ce41e3f`, 2026-08-17, which moved it there.) Saved graphs store `type` as that
string and match `widgets_values` by index, so a rename or an insertion breaks
every graph built before it -- this repo's `workflows/*.json` and, the part no
check can see, the owner's live graphs outside it.

## Why nothing caught this before

**Every existing guard derives its expectation from the thing it is checking.**
`bench/check_workflow_schema.py` validates the saved graphs against a live
`/object_info`; both come from the schema. `workflows/build_workflows.py`
regenerates all 89 graphs from the schema. So renaming a `node_id` and
regenerating leaves every artifact internally consistent, every fast check
green, and only the owner's external graphs broken -- silently, which is the
exact failure mode `docs/comfy_notes.md` describes.

A control whose input is derived from the thing it is checking cannot fail.
That is the same family as `verify_adjacency` running on the one input where a
Hilbert curve cannot jump; see `docs/checks.md`. The fix is the only thing that
fixes that family: **an independent baseline**, committed, that a schema change
cannot rewrite.

## What the manifest is

`bench/node_id_manifest.json` records, per registered node, the `node_id` and
the ordered input and output names -- the whole of what a saved graph addresses
positionally. It is committed, and updating it is a deliberate act. That is the
point: a diff to this file in a pull request is the review prompt.

**If this check goes red, the default answer is to revert the rename, not to
update the manifest.** Update it only when adding a node, or appending an input
or output at the END, which are the two changes `docs/comfy_notes.md` permits.

## Running it

    python bench/check_node_ids.py            # verify
    python bench/check_node_ids.py --write    # regenerate after a PERMITTED change

Needs ComfyUI importable for `comfy_api`; no server, no GPU, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "bench" / "node_id_manifest.json"


def collect():
    """{class name: {node_id, inputs, outputs}} straight from the schemas.

    Read from the extension's own node list rather than from a saved graph or
    from `/object_info`, because both of those are downstream of the schema and
    would agree with any rename.
    """
    # ComfyUI's root goes on FIRST and this repo's root goes on NOT AT ALL.
    # `docs/comfy_notes.md`: a bare `import nodes` resolves to OURS, and a later
    # `import nodes` inside comfy_extras then finds this pack and dies on a
    # relative import. So load `nodes.py` as a member of a synthetic package
    # instead, which also gives its `from .assert_chain import ...` a parent to
    # resolve against. Going through `__init__.py` would work too and is still
    # the wider surface -- it runs whatever registration the pack grows next.
    # The concrete objection until 2026-08-27 was that it imported
    # `single_frame`, which patched `comfy_extras` at import; that shim is
    # parked (`archive/single_frame.py`) and the pack no longer patches core at all,
    # so what remains is the `import nodes` trap above, which is reason enough.
    import asyncio
    import importlib.util
    import types

    sys.path.insert(0, str(REPO.parent.parent))   # ComfyUI root, for comfy_api
    # Schema collection is a CPU-only operation. On this install importing
    # Comfy's model management otherwise selects the CUDA device even though
    # no node is executed, which makes this nominally GPU-free guard contend
    # with a running render for no reason.
    import comfy.cli_args
    comfy.cli_args.args.cpu = True

    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules["_h3pack"] = pkg
    spec = importlib.util.spec_from_file_location("_h3pack.nodes", REPO / "nodes.py")
    _pack = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.nodes"] = _pack
    spec.loader.exec_module(_pack)

    ext = _pack.H3ExplorationsExtension()
    classes = asyncio.run(ext.get_node_list())

    out = {}
    for cls in classes:
        schema = cls.define_schema()
        out[cls.__name__] = {
            "node_id": schema.node_id,
            "inputs": [getattr(i, "id", getattr(i, "name", "?"))
                       for i in (schema.inputs or [])],
            "outputs": [getattr(o, "id", None) or getattr(o, "display_name", None)
                        or type(o).__name__ for o in (schema.outputs or [])],
        }
    return out


def compare(actual: dict, manifest: dict) -> list[str]:
    """Every way a saved graph can be broken, as separate messages.

    Kept a pure function of two dicts so the red-demonstration harness can feed
    it a mutated mapping without touching the source it guards.
    """
    errs = []
    for name, want in manifest.items():
        got = actual.get(name)
        if got is None:
            errs.append(f"{name}: registered node is GONE. Every saved graph "
                        f"using `{want['node_id']}` breaks.")
            continue
        if got["node_id"] != want["node_id"]:
            errs.append(f"{name}: node_id RENAMED {want['node_id']!r} -> "
                        f"{got['node_id']!r}. Every saved graph with the old "
                        f"`type` breaks, with no error in the UI. Revert it.")
        if got["inputs"][:len(want["inputs"])] != want["inputs"]:
            errs.append(f"{name}: inputs REORDERED or renamed.\n"
                        f"      was {want['inputs']}\n"
                        f"      now {got['inputs']}\n"
                        f"      `widgets_values` is matched by index, so every "
                        f"later value in every existing graph re-points.")
        elif len(got["inputs"]) > len(want["inputs"]):
            errs.append(f"{name}: inputs APPENDED "
                        f"{got['inputs'][len(want['inputs']):]} -- permitted, but "
                        f"the manifest must be updated deliberately (--write).")
        if got["outputs"][:len(want["outputs"])] != want["outputs"]:
            errs.append(f"{name}: outputs REORDERED. Links are integer slots.")
    for name in actual.keys() - manifest.keys():
        errs.append(f"{name}: NEW node not in the manifest -- permitted, but run "
                    f"--write so the baseline is a deliberate record.")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest. Only after a PERMITTED change: "
                         "a new node, or an input/output appended at the end.")
    args = ap.parse_args()

    actual = collect()
    if args.write:
        MANIFEST.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST.relative_to(REPO)} with {len(actual)} nodes")
        return 0

    if not MANIFEST.exists():
        print(f"FAIL  {MANIFEST.relative_to(REPO)} is missing. Run --write once "
              f"to record the current schema as the baseline.")
        return 1

    manifest = json.loads(MANIFEST.read_text())
    errs = compare(actual, manifest)
    print(f"node ids and positional contract, {len(actual)} registered node(s), "
          f"against a committed baseline")
    if not errs:
        print("  ok    every node_id and every input/output position is unchanged")
        return 0
    for e in errs:
        print(f"  FAIL  {e}")
    print(f"\n{len(errs)} failure(s). Default action is to REVERT, not to --write.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
