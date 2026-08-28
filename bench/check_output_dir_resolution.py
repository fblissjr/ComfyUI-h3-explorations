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


def _fallback_must_earn_its_use() -> list[str]:
    """The stock fallback refuses a directory that holds no renders.

    The escaped instance, 2026-08-28: `<comfy>/output` EXISTS on this box,
    holding ComfyUI's placeholder and nothing else, because the server writes
    to a share via `--output-directory`. The first `output_dir()` raised only on
    ABSENCE, so it returned that directory happily and a peer session's analysis
    reported "no renders" instead of "wrong path". An empty-but-present
    directory reads as success, which is worse than a missing one.

    Driven against the real `output_dir()` rather than a copy of its rule.
    """
    import importlib.util, os, tempfile
    spec = importlib.util.spec_from_file_location(
        "h3cfg", REPO / "workflows" / "h3_config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    out = []
    keep = os.environ.pop("H3_OUTPUT_DIR", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            # Point the FALLBACK at a stock-shaped directory: present, holding
            # only ComfyUI's placeholder. This is the case that matters, and an
            # earlier version of this function tested the NAMED path instead --
            # which cannot fail, because naming a path is what honours it.
            # Neutering the resolver left that version green.
            stock = Path(d) / "output"
            stock.mkdir()
            (stock / "_output_images_will_be_put_here").touch()
            cfg.COMFY_ROOT = Path(d)

            try:
                got = cfg.output_dir()
                out.append(f"  FAIL  empty_stock_refused  fallback returned "
                           f"{got}, which holds no renders. An empty-but-"
                           f"present directory reads as success and produces "
                           f"'no results' instead of 'wrong path'.")
            except SystemExit:
                pass

            # The same fallback WITH a render is accepted.
            (stock / "x.png").touch()
            try:
                cfg.output_dir()
            except SystemExit:
                out.append("  FAIL  render_dir_accepted  a fallback holding a "
                           "render was refused")

            # And an explicitly NAMED path is honoured even when empty, because
            # the caller naming it is the whole point.
            empty = Path(d) / "named"; empty.mkdir()
            os.environ["H3_OUTPUT_DIR"] = str(empty)
            try:
                cfg.output_dir()
            except SystemExit:
                out.append("  FAIL  named_path_honoured  an explicitly named "
                           "H3_OUTPUT_DIR was refused for being empty")
    finally:
        os.environ.pop("H3_OUTPUT_DIR", None)
        if keep is not None:
            os.environ["H3_OUTPUT_DIR"] = keep
    return out


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
    fb = _fallback_must_earn_its_use()
    if fb:
        offenders_msg = "\n".join(fb)
        print("Render output directory: resolved in one place, not counted per script")
        print(offenders_msg)
        return 1
    print(f"  ok    derives_output_dir  {scanned} script(s) scanned, none counts "
          f"parents to reach `output`")
    print("  ok    fallback_earns_its_use  a named path is honoured; a stock "
          "path with no renders is refused")
    print(f"  ok    allowlist_is_necessary  {len(ALLOWED)} exemption(s), each named")
    print("\nall ok -- one resolver, and it fails before the GPU time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
