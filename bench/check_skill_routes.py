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

import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# path -> why it is named despite being absent.
DECLARED_ABSENT = {
    "internal/2026-8-20-system-prompts":
        "named by h3-experiment step 4 only to record that its old routing "
        "was withdrawn 2026-09-01; the live routing is docs/prompting.md",
}

ROOTS = ("docs", "bench", "workflows", "internal", "coderef", "vendor_config",
         "vendor_guides", "archive", ".claude")
PATH = re.compile(r"`((?:" + "|".join(re.escape(r) for r in ROOTS)
                  + r")/[^`\s]+)`")


def main() -> int:
    skills = sorted(glob.glob(str(REPO / ".claude" / "skills" / "*" / "SKILL.md")))
    if not skills:
        print("FAIL  no SKILL.md found; this check scans nothing and a silent "
              "green would read as a pass")
        return 2

    seen, missing = 0, []
    for path in skills:
        name = Path(path).parent.name
        for m in PATH.finditer(Path(path).read_text(encoding="utf-8")):
            target = m.group(1).rstrip("/,.")
            if "*" in target or "<" in target:
                continue          # a glob or a placeholder, not a route
            seen += 1
            if (REPO / target).exists() or target in DECLARED_ABSENT:
                continue
            missing.append((name, target))

    stale = [p for p in DECLARED_ABSENT if (REPO / p).exists()]

    for skill, target in missing:
        print(f"  FAIL  {skill} routes to `{target}`, which does not exist. An "
              f"agent following this skill lands nowhere.")
    for p in stale:
        print(f"  FAIL  `{p}` is declared absent but exists; the declaration is "
              f"stale and should be removed")

    print(f"\n  {seen} route(s) across {len(skills)} skill(s); "
          f"{len(DECLARED_ABSENT)} declared absent")
    if missing or stale:
        return 1
    print("  ok    every skill route resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
