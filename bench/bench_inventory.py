#!/usr/bin/env python3
"""Say, for every script at the top of `bench/`, what in the repo depends on it.

    <comfy venv python> bench/bench_inventory.py              # the table, markdown
    <comfy venv python> bench/bench_inventory.py --json       # the same rows, for a tool
    <comfy venv python> bench/bench_inventory.py --only unreferenced

Built for the bench reorganization (`docs/projects/reorg/`): before a script is
moved, archived or deleted, this says who would notice. One row per tracked
`bench/*.py`, with:

- `kind`: `check` (`check_*.py`, run by `bench/run_checks.py`, which finds them by
  name), `lib` (imported by another tracked script), or `tool`.
- `imported_by`: tracked `.py` files that `import` it or `from` it.
- `called_by`: tracked code, graph, arm or shell files that name `bench/<name>.py`
  or `<name>.py` (a subprocess call, an arm manifest, a recipe), not counting the
  script itself and the generated indexes.
- `docs`: tracked `.md` files that name it, the generated indexes, `CHANGELOG.md`
  and `changelog.d/` excluded (they record history, not a dependency).
- `last`: the date of the last commit that touched it.
- `status`: `check`, `lib`, `wired` (something calls it), `documented` (only prose
  names it) or `unreferenced` (nothing tracked names it).

## What it cannot see

A name built at run time (`f"bench/{name}.py"`), a call from an untracked file
(`internal/`, a session's scratch script, the owner's shell history) and a
habit of running a tool by hand all read as "not referenced". So
`unreferenced` means "nothing tracked names it", never "safe to delete": the
reading is left to whoever prunes. A mention is a name match, so a script whose
name is a common word can be over-counted.

Needs no ComfyUI and no card; reads `git ls-files` and one `git log` pass.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEXT = {".py", ".md", ".json", ".sh", ".txt", ".toml", ".yml", ".yaml", ".html", ".js"}
# Generated indexes list every script by construction; history files record
# what changed, not what depends on what.
NOT_A_DEPENDENCY = {"INDEX.md", "bench/INDEX.md", "bench/results/INDEX.md", "workflows/INDEX.md", "CHANGELOG.md"}


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (REPO / p).is_file()]


def last_touched(paths: set[str]) -> dict[str, str]:
    """One pass over the log: the newest commit date per path."""
    log = subprocess.run(["git", "log", "--format=@%cs", "--name-only", "--", "bench"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    seen: dict[str, str] = {}
    date = ""
    for line in log.splitlines():
        if line.startswith("@"):
            date = line[1:]
        elif line in paths and line not in seen:
            seen[line] = date
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--only", choices=["check", "lib", "wired", "documented", "unreferenced"],
                    help="print only rows of this status")
    args = ap.parse_args()
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `| head` ends the table quietly

    files = tracked()
    scripts = sorted(p for p in files if re.fullmatch(r"bench/[^/]+\.py", p))
    corpus: dict[str, str] = {}
    for p in files:
        if Path(p).suffix not in TEXT or p in NOT_A_DEPENDENCY or p.startswith("changelog.d/"):
            continue
        try:
            corpus[p] = (REPO / p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    stems = {Path(s).stem: s for s in scripts}
    imported_by: dict[str, set[str]] = defaultdict(set)
    called_by: dict[str, set[str]] = defaultdict(set)
    docs: dict[str, set[str]] = defaultdict(set)
    imp = re.compile(r"^\s*(?:from\s+(?:bench\.)?(\w+)\s+import|import\s+(?:bench\.)?(\w+))", re.M)
    named = re.compile(r"(?<![\w/.-])(?:bench/)?(\w+)\.py\b")
    for path, text in corpus.items():
        if path.endswith(".py"):
            for m in imp.finditer(text):
                stem = m.group(1) or m.group(2)
                if stem in stems and stems[stem] != path:
                    imported_by[stem].add(path)
        bucket = docs if path.endswith(".md") else called_by
        for m in named.finditer(text):
            stem = m.group(1)
            if stem in stems and stems[stem] != path:
                bucket[stem].add(path)

    dates = last_touched(set(scripts))
    rows = []
    for stem, path in stems.items():
        lines = corpus.get(path, "").count("\n")
        kind = "check" if stem.startswith("check_") else ("lib" if imported_by[stem] else "tool")
        status = kind if kind != "tool" else (
            "wired" if called_by[stem] else "documented" if docs[stem] else "unreferenced")
        rows.append({"script": path, "kind": kind, "status": status, "lines": lines,
                     "last": dates.get(path, ""), "imported_by": sorted(imported_by[stem]),
                     "called_by": sorted(called_by[stem]), "docs": sorted(docs[stem])})
    if args.only:
        rows = [r for r in rows if r["status"] == args.only]

    if args.json:
        json.dump(rows, sys.stdout, indent=1)
        print()
        return 0
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    print(f"{len(rows)} script(s): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    print("\n| script | status | lines | last | imported by | called by | docs |\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['script']}` | {r['status']} | {r['lines']} | {r['last']} | "
              f"{len(r['imported_by'])} | {len(r['called_by'])} | {len(r['docs'])} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
