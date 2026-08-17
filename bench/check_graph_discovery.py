#!/usr/bin/env python3
"""No check may find graphs by globbing. Discovery goes through `graph_paths()`.

CLAUDE.md: "**every check that walks graphs must go through `graph_paths()`** --
a bare non-recursive glob passes green over a subset". That is the failure this
guards: `workflows/*.json` misses `workflows/image/`, so a check that globs
reports success over part of the set and looks identical to one that passed
over all of it.

**The convention currently holds across every graph-walking check.** That is
exactly when it is cheapest to lock in, and exactly when nothing would notice
it breaking -- a new check written next month with a plausible
`WORKFLOWS.glob("*.json")` would pass its own assertions, silently over a
subset, and no existing guard would say a word. Same family as the `node_id`
gap: a convention satisfied by everyone and enforced by nothing.

## Why this parses rather than greps

`bench/check_ref_prompt_labels.py:66` contains the string
`WORKFLOWS.glob("*.json")` **in a comment explaining not to do it**. A regex
over source flags that line and the first fix anyone reaches for is to reword
the comment, which teaches the opposite lesson. So this walks the AST and only
sees real calls; comments and docstrings are invisible to it by construction.

## Exemptions

There are none, and that is deliberate. `bench/check_workflow_schema.py` is the
documented exception to the *convention* -- CLAUDE.md notes it takes paths from
the CLI, "the one place a directory has to be typed" -- but it does not glob, so
it needs no exemption here.

**If you are about to add one, write down why in `EXEMPT` and make it specific
to a mechanism, not to a file.** An allowlist is where "add your file here"
quietly becomes the fix for a red, and this check exists because a convention
without teeth decays.

## Running it

    python bench/check_graph_discovery.py

No imports of the checks, no ComfyUI, no server, no GPU. Parses source only, so
a check with a broken import is still audited.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"

#: {filename: reason}. A reason naming a file rather than a mechanism is not a
#: reason -- see the module docstring.
EXEMPT: dict[str, str] = {
    "check_ref_prompt_labels.py":
        "its subject IS discovery coverage: it enumerates the tree "
        "independently to find graph directories GRAPH_DIRS does not reach. "
        "Routing that through graph_paths() would make it derive its "
        "expectation from the thing it checks, which is the defect it exists "
        "to catch. COST: the rest of that file is unaudited by this check.",
}

#: What counts as enumerating a directory.
ENUMERATORS = {"glob", "rglob", "iterdir"}

#: Receiver names that mean "this is a graph directory". Checked because
#: `pathlib.Path("/proc").iterdir()` in `check_single_frame.py` is process
#: enumeration, and a rule that cannot tell it from graph discovery reports a
#: correct file as broken -- which trains people to ignore this check.
GRAPH_RECEIVERS = ("workflow", "graph_dir", "graph_root")


def enumeration_sites(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, description) for every directory-enumeration call in the tree.

    Only `glob`/`rglob` calls whose pattern literal mentions `.json` count, plus
    every `iterdir`. A `.rglob("*.py")` is not graph discovery and is left
    alone -- `check_provenance_stamp.py` legitimately does that.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in ENUMERATORS:
            continue
        # Only enumerations of something graph-shaped. `ast.unparse` of the
        # receiver is the cheapest readable form of "what is being walked".
        receiver = ast.unparse(func.value).lower()
        if not any(tok in receiver for tok in GRAPH_RECEIVERS):
            continue
        if func.attr == "iterdir":
            out.append((node.lineno, f"{ast.unparse(func.value)}.iterdir()"))
            continue
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if ".json" in arg.value:
                out.append((node.lineno, f'{func.attr}("{arg.value}")'))
        else:
            # A non-literal pattern cannot be judged statically. Report it
            # rather than assume it is safe: an unreadable enumeration is the
            # case this check is least able to reason about, so it should be
            # the case a human looks at.
            out.append((node.lineno, f"{func.attr}(<non-literal pattern>)"))
    return out


def audit(files) -> list[str]:
    """Kept a pure function of a path list so the red harness can feed it
    scratch files without writing anything into `bench/`."""
    errs = []
    for path in sorted(files):
        name = path.name
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            errs.append(f"{name}: will not parse ({exc}); cannot be audited")
            continue
        sites = enumeration_sites(tree)
        if not sites:
            continue
        if name in EXEMPT:
            continue
        for line, what in sites:
            errs.append(
                f"{name}:{line} enumerates with {what} instead of "
                f"`h3_config.graph_paths()`. A bare glob over `workflows/` "
                f"misses `workflows/image/`, so this check would pass green "
                f"over a subset. Route it through graph_paths(), or add an "
                f"EXEMPT entry stating the mechanism that makes it safe.")
    return errs


def main() -> int:
    files = sorted(BENCH.glob("check_*.py"))
    files = [f for f in files if f.name != Path(__file__).name]
    errs = audit(files)

    print(f"graph discovery routes through graph_paths(), "
          f"{len(files)} check(s) audited")
    if EXEMPT:
        for k, v in EXEMPT.items():
            print(f"  note  {k} exempt: {v}")
    if not errs:
        print("  ok    no check enumerates graph files directly")
        return 0
    for e in errs:
        print(f"  FAIL  {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
