#!/usr/bin/env python3
"""Where each `data/results/...` path cited in the repo's text actually is on this machine.

`data/` is gitignored and local. Older run data in `data/results/` was moved into dated
archive folders (`data/results/2026-08-18_to_2026-10-03/`, `data/results/2026-10-04_to_2026-11-03/`)
without its citations changing, so most of the paths the docs and code comments cite
point where nothing is. The move kept each path below its folder, so
`bench/_lib/results.py` (the lookup the tools read through) finds each one; this tool
runs that lookup for every citation and prints what it found.

Each cited path is one of:

  here      the path exists as written.
  moved     not there, but in exactly one archive folder; printed with `--moved`.
  several   in more than one archive folder; printed always.
  missing   in neither; printed always. It may live on another machine, under
            `bench/results/`, or never have been saved.

It reports and never fails. `bench/check_doc_links.py` treats `data/` citations the same
way, as warnings, because a machine without the run data is not broken.

**What counts as a citation.** `data/results/` followed by a path, in every file git would
take (tracked, plus untracked files no ignore rule covers). Skipped: templates and globs
(a `<`, `*`, `{`, `[`, `$` or `?` right after the path, or `DATE` or `...` inside it) and
output paths given to `--out`. Code that builds the path in pieces
(`REPO / "data" / "results" / name`) is not seen.

`--fix FILE ...` rewrites the moved citations in the files named, and only those, to
where each file is now. Dated records can stay as written; this tool finds their files.

    <comfy-venv>/bin/python bench/resolve_data_citations.py [--moved] [--fix FILE ...]

Needs git and the local `data/results/`; no ComfyUI, no card. Added 2026-11-18
(`bench/results/2026-11-18_text_encoder_int8_vs_bf16_speed.md`, section 3).
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.results import RESULTS, find_results  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".sh", ".txt", ".toml", ".json", ".js", ".html", ".yml", ".yaml"}
CITE = re.compile(r"data/results/([^\s`'\"()\[\]|<>*{}$\\?,;]+)")
TEMPLATE_NEXT = set("<*{[$?")


def corpus() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if Path(f).suffix in TEXT_SUFFIXES and not f.startswith("data/")]


def citations(text: str):
    """(start, end, cited path) for each citation in `text`."""
    for m in CITE.finditer(text):
        cited = m.group(1).rstrip(".:")
        end = m.start(1) + len(cited)
        if m.end() < len(text) and text[m.end()] in TEMPLATE_NEXT:
            continue
        if "DATE" in cited or "..." in cited:
            continue
        if re.search(r"--out[= ]\S*$", text[max(0, m.start() - 40):m.start()]):
            continue
        yield m.start(1), end, cited


def resolve(cited: str, root: Path = RESULTS) -> tuple[str, list[Path]]:
    """here, moved, several or missing, with where `bench/_lib/results.py` found it."""
    found = find_results(cited, root)
    if found == [root / cited]:
        return "here", found
    if len(found) == 1:
        return "moved", found
    return ("several", found) if found else ("missing", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--moved", action="store_true", help="also list each moved citation and where it is now")
    ap.add_argument("--fix", nargs="+", metavar="FILE", default=[],
                    help="rewrite the moved citations in these files (repo-relative) to their new paths")
    args = ap.parse_args()
    if not RESULTS.is_dir():
        print("no data/results/ on this machine; nothing to resolve")
        return 0

    by_state = collections.defaultdict(dict)   # state -> cited -> (files, found)
    for f in corpus():
        try:
            text = (REPO / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for _, _, cited in citations(text):
            state, found = resolve(cited)
            files, _ = by_state[state].setdefault(cited, (set(), found))
            files.add(f)

    for f in args.fix:
        path = REPO / f
        text = path.read_text(encoding="utf-8")
        out, last, n = [], 0, 0
        for start, end, cited in citations(text):
            state, found = resolve(cited)
            if state != "moved":
                continue
            new = found[0].relative_to(RESULTS).as_posix() + ("/" if cited.endswith("/") else "")
            out += [text[last:start], new]
            last, n = end, n + 1
        if n:
            path.write_text("".join(out) + text[last:], encoding="utf-8")
        print(f"fixed {n} citation(s) in {f}")

    print(f"data/results citations: " + ", ".join(
        f"{s} {len(by_state[s])}" for s in ("here", "moved", "several", "missing")))
    for state in (["moved"] if args.moved else []) + ["several", "missing"]:
        for cited, (files, found) in sorted(by_state[state].items()):
            where = ", ".join(p.relative_to(RESULTS).as_posix() for p in found) or "-"
            print(f"  {state:<8} data/results/{cited}\n           now: {where}\n"
                  f"           cited in: {', '.join(sorted(files)[:4])}"
                  + (f" (+{len(files) - 4})" if len(files) > 4 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
