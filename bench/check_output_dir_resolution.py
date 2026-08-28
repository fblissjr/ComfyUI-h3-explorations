#!/usr/bin/env python3
"""No script derives the render output directory by counting `..`.

## The escaped instance

`bench/grade_pdd_partitions.py` had

    OUT = ... else Path(__file__).resolve().parents[2] / "output"

which from `bench/` is `<repo>/../output`, i.e. `custom_nodes/output` -- not an
output directory on any install. Seven arms rendered at 362 frames and every one
was thrown away undecoded, because the path was only touched after the queue
drained.

Why it is easy to write: two conventions are in use across `bench/`, and they
differ by one.

    Path(__file__).resolve().parents[2]   # from the FILE:  custom_nodes
    HERE.parents[2]                       # HERE = the DIR: ComfyUI root

Reading one while writing the other is the whole bug. `h3_config.output_dir()`
exists so nobody counts levels, and it RAISES on a missing directory so the
failure lands before the GPU time rather than after it.

## Why this is narrow on purpose

It flags exactly one shape: a path expression that counts parents AND names
`output`. It deliberately does NOT police the `sys.path` bootstraps that reach
the ComfyUI root by counting -- those run BEFORE `h3_config` is importable, so
demanding they use it would be demanding a cycle. `CLAUDE.md`: a check that
reports red while the state is correct trains you to ignore red.

`H3_OUTPUT_DIR` is not the fix by itself, because a script can honour it and
still fall back to a wrong stock path when it is unset -- which is precisely
what the escaped instance did.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: A path expression that counts parents and names `output` in the same
#: statement. Matches the escaped instance and the shapes next to it.
DERIVES_OUTPUT = re.compile(
    r"""parents?\s*(\[\s*\d+\s*\]|\.parent)[^\n]{0,80}["']output["']"""
    r"""|["']output["'][^\n]{0,80}parents?\s*(\[\s*\d+\s*\]|\.parent)""")

#: Files allowed to name it themselves, with the reason. `h3_config` IS the
#: resolver; a file that merely documents the rule is not doing it.
ALLOWED = {
    "workflows/h3_config.py": "defines output_dir(); it is the one owner",
    "bench/check_output_dir_resolution.py": "this file quotes the bug",
}


def main() -> int:
    offenders = []
    scanned = 0
    for path in sorted(REPO.glob("bench/*.py")) + sorted(REPO.glob("workflows/*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED:
            continue
        scanned += 1
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if DERIVES_OUTPUT.search(line):
                offenders.append((rel, n, line.strip()[:100]))

    print("Render output directory: resolved in one place, not counted per script")
    if offenders:
        for rel, n, src in offenders:
            print(f"  FAIL  derives_output_dir  {rel}:{n}\n        {src}")
        print(f"\n{len(offenders)} script(s) derive the output directory by counting "
              f"parents. Import `h3_config.output_dir()` instead: it honours "
              f"H3_OUTPUT_DIR, falls back to the stock path, and RAISES when the "
              f"directory is absent -- which is the half that matters, because a "
              f"wrong output path is otherwise only discovered after the render.")
        return 1
    print(f"  ok    derives_output_dir  {scanned} script(s) scanned, none counts "
          f"parents to reach `output`")
    print(f"  ok    allowlist_is_necessary  {len(ALLOWED)} exemption(s), each named")
    print("\nall ok -- one resolver, and it fails before the GPU time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
