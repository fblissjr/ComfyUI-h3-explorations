#!/usr/bin/env python
"""Report the documents under `docs/` that no link reaches.

## What it walks

Every markdown link in `CLAUDE.md` and in each page under `docs/wiki/` is a
starting point; from there it follows links between documents and prints each
`.md` under `docs/` that nothing reaches. Reachability, not membership: a deep
dive reached only through its parent (`morton.md` through `SOLATTN.md`) is not
a finding, and a document no file links at all is.

## Why it no longer generates the index

Until 2026-09-11 this script built `docs/wiki/index.md` from CLAUDE.md's
routing tables, and `--check` refused a stale copy. The owner moved those
tables into the wiki, so CLAUDE.md routes there instead of carrying them.
`docs/wiki/index.md` is now written by hand and is the only copy of the
routes, so there is nothing left to go stale and nothing to generate.

## A report, not a gate

Some unreached documents are deliberate (dated records inside a research
subtree whose own README is their route), and a gate would have to encode
which. Links naming a file that is gone are `check_doc_links.py`'s job and are
not re-checked here. Exits 0 after reporting, or 2 when a starting file is
missing.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WIKI = REPO / "docs" / "wiki"
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Active onboarding routers and leaf specs should fit in a single agy read_file
# call (46,080 bytes) with margin so older history gets rolled into shards.
ROUTER_MAX_BYTES = 45_000
ACTIVE_ROUTERS = (
    "AGENTS.md",
    "data/AGENTS.md",
    "docs/comfy_notes.md",
    "docs/wiki/handover.md",
    "docs/wiki/working_with_the_owner.md",
    "docs/wiki/index.md",
    "docs/wiki/decisions.md",
    "docs/wiki/masked_v2v.md",
    "docs/wiki/next_steps.md",
    "docs/masked_v2v/AGENTS.md",
    "docs/masked_v2v/PROTOCOL.md",
    "docs/masked_v2v/reference_and_prompt_rules.md",
    "docs/masked_v2v/chain_trace_and_remakes.md",
    "docs/checks.md",
    "docs/prompting.md",
    "docs/evidence.md",
    "docs/roadmap.md",
    "docs/research/INDEX.md",
)


def starting_files() -> list[Path]:
    return [REPO / "CLAUDE.md", *sorted(WIKI.glob("*.md"))]


def docs_on_disk() -> list[str]:
    found = []
    for p in sorted((REPO / "docs").rglob("*.md")):
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("docs/wiki/"):
            continue  # the wiki describes the docs, not itself
        found.append(rel)
    return found


def link_targets(rel: str) -> list[str]:
    """Repo-relative targets of the relative markdown links in one file."""
    path = REPO / rel
    base = Path(rel).parent
    out = []
    for href in LINK.findall(path.read_text(encoding="utf-8", errors="replace")):
        href = href.split("#", 1)[0].strip()
        if not href or "://" in href:
            continue
        target = href if href.startswith("docs/") else (base / href).as_posix()
        try:
            out.append((REPO / target).resolve().relative_to(REPO).as_posix())
        except (ValueError, OSError):
            continue
    return out


def unreachable(on_disk: list[str]) -> list[str]:
    known = set(on_disk)
    reached: set[str] = set()
    frontier: list[str] = []
    for start in starting_files():
        for t in link_targets(start.relative_to(REPO).as_posix()):
            if t in known and t not in reached:
                reached.add(t)
                frontier.append(t)
    while frontier:
        current = frontier.pop()
        if not (REPO / current).exists():
            continue
        for t in link_targets(current):
            if t in known and t not in reached:
                reached.add(t)
                frontier.append(t)
    return [p for p in on_disk if p not in reached]


def collapse(paths: list[str], threshold: int = 3) -> list[str]:
    """Group the list by directory so it stays readable.

    A directory contributing more than `threshold` entries is named once with
    its count: the useful fact there is "this whole subtree is unreachable",
    not each file in it.
    """
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_dir[str(Path(p).parent)].append(p)
    lines: list[str] = []
    for d, members in sorted(by_dir.items()):
        if len(members) > threshold:
            lines.append(f"{d}/ -- {len(members)} file(s), the whole subtree")
        else:
            lines.extend(members)
    return lines


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for required in (REPO / "CLAUDE.md", WIKI / "index.md"):
        if not required.exists():
            print(f"FAIL  {required.relative_to(REPO)} not found")
            return 2
    unrouted = unreachable(docs_on_disk())
    print(f"ok    walked links from CLAUDE.md and {len(starting_files()) - 1} wiki page(s)")
    print(f"      unreachable       {len(unrouted)} doc(s) no link reaches")
    for entry in collapse(unrouted):
        print(f"        {entry}")
    oversized = []
    missing_routers = []
    checked_paths = list(ACTIVE_ROUTERS)
    for shard_dir in ("docs/checks", "docs/prompting", "docs/wiki/decisions"):
        d = REPO / shard_dir
        if d.is_dir():
            for sp in sorted(d.glob("*.md")):
                checked_paths.append(sp.relative_to(REPO).as_posix())
    for rel in checked_paths:
        p = REPO / rel
        if not p.exists():
            missing_routers.append(rel)
        elif p.stat().st_size > ROUTER_MAX_BYTES:
            oversized.append((rel, p.stat().st_size))
    ok_count = len(checked_paths) - len(oversized) - len(missing_routers)
    print(f"      router size check {ok_count}/{len(checked_paths)} <= {ROUTER_MAX_BYTES} B (46080 B read_file limit)")
    for rel in missing_routers:
        print(f"        MISSING {rel} -- update ACTIVE_ROUTERS or restore file")
    for rel, sz in oversized:
        print(f"        OVER {rel} ({sz} B > {ROUTER_MAX_BYTES} B) -- shard older entries")
    return 2 if missing_routers else 0


if __name__ == "__main__":
    raise SystemExit(main())
