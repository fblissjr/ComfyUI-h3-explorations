#!/usr/bin/env python
"""Fail when a skill routes an agent to a path that does not exist.

## Why this one and not a wider link checker

A skill is an ENTRY POINT. When a doc has a stale link a reader shrugs and looks
elsewhere; when a skill's routing is dead the agent that invoked it goes
nowhere, and it goes nowhere silently, having been told it was following the
repo's own instructions.

That is not hypothetical. `h3-experiment` step 4 -- the step that answers "how
is a prompt written" -- routed to `internal/2026-8-20-system-prompts/`, a
gitignored directory that does not exist, while `docs/prompting.md` sat
unreferenced by any skill at all. Found 2026-09-01 by reading the skill, not by
any check: `check_doc_links.py` scans `.claude/` but its grammar is
`path.ext:12` citations and markdown links, and a bare backticked directory is
neither. Widening THAT grammar would false-red across every doc in the repo, so
the narrow rule lives here instead.

## The rule, and why it is decidable

Every repo-relative path a `SKILL.md` names in backticks must exist. That is it.
No judgement about whether the path is the RIGHT one -- only that following it
lands somewhere. Measured when written: 30 paths across the shipped skills, one
missing, no false positives.

**Existence, not git-tracked.** `internal/blind_keys/` is correctly gitignored
-- sealed scoring keys -- and a skill may legitimately route to it, so requiring
tracked-ness would go red on a working entry point.

## Declaring a path absent on purpose

You cannot retract a dead pointer without naming it, and this repo's rule is to
say what a withdrawn claim used to be. So a path may be declared below with a
reason, the same idiom `check_doc_links.py` uses. **A declaration is a claim
that the path SHOULD be absent**, so it is checked in both directions: a
declared path that comes back is a stale declaration and fails too.
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# path -> why it is named despite being absent. Empty since 2026-09-03: the
# one entry (`internal/2026-8-20-system-prompts`, h3-experiment's withdrawn
# routing) left the skill when the history note moved to CHANGELOG.md.
DECLARED_ABSENT: dict[str, str] = {}

ROOTS = ("docs", "bench", "workflows", "internal", "coderef", "vendor_config",
         "vendor_guides", "archive", ".claude")
PATH = re.compile(r"`((?:" + "|".join(re.escape(r) for r in ROOTS)
                  + r")/[^`\s]+)`")


REVIEWED = re.compile(r"^reviewed:\s*([0-9a-f]{7,40})\s*$", re.M)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, text=True,
                          capture_output=True, check=False).stdout.strip()


def review_point(text: str) -> str | None:
    """The commit a skill was last read against its sources, from frontmatter."""
    m = REVIEWED.search(text.split("\n---", 1)[0] if text.startswith("---") else "")
    return m.group(1) if m else None


def moved_since(target: str, reviewed: str) -> str | None:
    """The short hash of the last commit touching `target` if it is not an
    ancestor of `reviewed`, else None. Uncommitted edits are invisible here on
    purpose: a review point is a commit, so only commits can pass it."""
    last = _git("log", "-1", "--format=%h", "--", target)
    if not last:
        return None
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", last, reviewed],
                        cwd=REPO, capture_output=True).returncode == 0
    return None if ok else last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="a route changed since its skill's review point fails "
                         "instead of being reported")
    args = ap.parse_args()
    skills = sorted(glob.glob(str(REPO / ".claude" / "skills" / "*" / "SKILL.md")))
    if not skills:
        print("FAIL  no SKILL.md found; this check scans nothing and a silent "
              "green would read as a pass")
        return 2

    seen, missing, unreviewed, moved = 0, [], [], []
    for path in skills:
        name = Path(path).parent.name
        text = Path(path).read_text(encoding="utf-8")
        reviewed = review_point(text)
        if reviewed is None:
            unreviewed.append(name)
        routes: list[str] = []
        for m in PATH.finditer(text):
            # `path::Symbol` is a pointer whose path half must exist here;
            # the symbol half is check_doc_links.py's `symbols_exist` case.
            target = m.group(1).split("::", 1)[0].rstrip("/,.")
            if "*" in target or "<" in target:
                continue          # a glob or a placeholder, not a route
            seen += 1
            if (REPO / target).exists() or target in DECLARED_ABSENT:
                routes.append(target)
                continue
            missing.append((name, target))
        if reviewed:
            for t in sorted(set(routes)):
                if (c := moved_since(t, reviewed)):
                    moved.append((name, reviewed, t, c))

    stale = [p for p in DECLARED_ABSENT if (REPO / p).exists()]

    for skill, target in missing:
        print(f"  FAIL  {skill} routes to `{target}`, which does not exist. An "
              f"agent following this skill lands nowhere.")
    for p in stale:
        print(f"  FAIL  `{p}` is declared absent but exists; the declaration is "
              f"stale and should be removed")
    for name in unreviewed:
        print(f"  FAIL  {name} has no `reviewed: <commit>` in its frontmatter; "
              f"nothing can say whether its sources moved since it was read")
    tag = "FAIL  " if args.strict else "REVIEW"
    for name, reviewed, t, c in moved:
        print(f"  {tag}  {name} names `{t}`, changed in {c} after its review "
              f"point {reviewed}; re-read the skill against it and bump `reviewed`")

    print(f"\n  {seen} route(s) across {len(skills)} skill(s); "
          f"{len(DECLARED_ABSENT)} declared absent")
    if missing or stale or unreviewed or (moved and args.strict):
        return 1
    print("  ok    every skill route resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
