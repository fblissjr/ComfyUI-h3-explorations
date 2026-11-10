#!/usr/bin/env python3
"""Check that every `h3_config.NAME` a tracked Python file reads is a name `workflows/h3_config.py` defines.

Why: `bench/control_vector/helpers.py::_load_encoded_references` read two constants the config never had,
from the day it was written (2026-11-04) until 2026-11-09. The code sat on a path no CPU check runs (it
needs the encoder and a reference picture), so every vector build with `--ref-images` failed at that line
and nothing was red.

What it checks: it parses each tracked `.py` file, finds attribute reads on a name bound to the config
module (`import h3_config`, `from workflows import h3_config`, `import workflows.h3_config as X`), and
fails on a read of a name the module does not have.

What it cannot see: a config read through `getattr` with a computed name, a name reached through another
alias, a wrong value under a right name, or a wrong name on any other module.

    <comfy-venv-python> bench/check_config_names.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from workflows import h3_config  # noqa: E402

#: Folders whose Python is not this pack's to keep true. **Reasoned**: `archive/` is retired code kept as
#: written, `vendor/` is other people's.
SKIP_PREFIXES = ("archive/", "vendor/")


def _aliases(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "h3_config" or (a.name.endswith(".h3_config") and a.asname):
                    names.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "h3_config":
                    names.add(a.asname or a.name)
    return names


def missing_in(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    aliases = _aliases(tree)
    return [(node.lineno, node.attr) for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in aliases and not hasattr(h3_config, node.attr)]


def main() -> int:
    files = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.py"], capture_output=True, text=True,
                           check=True).stdout.split()
    files = [f for f in files if not f.startswith(SKIP_PREFIXES)]
    bad: list[str] = []
    readers = 0
    for rel in files:
        try:
            found = missing_in((REPO_ROOT / rel).read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append(f"{rel}: does not parse ({exc.msg}, line {exc.lineno})")
            continue
        readers += 1
        bad += [f"{rel}:{line}: h3_config.{name} is not defined in workflows/h3_config.py" for line, name in found]
    # The check's own control: a made-up name must be caught.
    if missing_in("from workflows import h3_config\nx = h3_config.NO_SUCH_NAME_EVER\n") != [(2, "NO_SUCH_NAME_EVER")]:
        bad.append("the check did not catch a made-up config name")
    print(f"config names: {readers} tracked Python file(s) parsed")
    for line in bad:
        print(f"  FAIL  {line}")
    if bad:
        print(f"\n{len(bad)} problem(s)")
        return 1
    print("  ok    every h3_config name a tracked file reads is defined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
