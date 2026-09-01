#!/usr/bin/env python
"""Fail when a doc says "nothing guards X" and something already does.

## The escaped instance, which is why this exists at all

CLAUDE.md's rule is that no new check is earned until a drift instance appears
that the existing gates provably could not have caught. This one is from
2026-09-01 and it is three-fold:

- `bench/check_camera_vocabulary.py` was added 2026-08-28 and grades every
  shipped prompt's camera motion against base_en 4.3.
- `docs/checks.md`'s **Uncontrolled requirements** table -- the standing audit
  whose entire job is listing what nothing watches -- went on saying that
  requirement was guarded by "nothing", while the SAME FILE indexed the checker
  a hundred lines earlier. One file, both halves.
- `docs/prompting.md` said the same thing and carried a same-day *"re-checked"*
  stamp. The re-check had looked in two named files and nowhere else, so it
  confirmed the claim it set out to verify and laundered a stale sentence into
  a fresh-looking one.

**A date records when a claim was written, not whether it is true**, and a
narrow re-check is worse than no re-check.

## What makes this decidable rather than a grep-and-judge

A check that reports red while the state is correct trains you to ignore red,
which is why `check_doc_links.py` and `check_retraction_consumers.py` both
prefer an allowlist to a fuzzy search. This one avoids that by only firing on
an EXACT, machine-readable coincidence: a row whose guard cell says "nothing",
whose subject cites a numbered guide section, and a `bench/check_*.py` whose
source names that same section. No judgement about whether the checker is
*adequate* -- only that the row's "nothing" is literally false.

**So it is deliberately narrow.** It cannot see a requirement described in
prose with no section number, and it says nothing about whether a guarded
requirement is guarded WELL. A green run means "no row claims nothing about
something a checker names", never "the audit is accurate".
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "docs" / "checks.md"
HEADING = "## Uncontrolled requirements"

# "nothing" as a GUARD claim, not the ordinary word. The table writes it
# emphasised or as a sentence of its own.
NOTHING = re.compile(r"\*\*nothing\b|\bnothing\.(?:\s|$)|\*\*none\b", re.I)
# `base_en 4.3`, `ref-en §5.2`, `base_en 4.3's` -- a numbered guide section.
SECTION = re.compile(r"(?:base_en|ref_en|base-en|ref-en)\s*§?\s*(\d+\.\d+)")


def main() -> int:
    # Optional path override. Its purpose is the red proof: this check's green
    # is only meaningful if it can be shown going red, and the state it fires
    # on is a PAST state of the audit file. Point it at a historical copy
    # (`git show <rev>:docs/checks.md`) to reproduce that.
    audit = Path(sys.argv[1]) if len(sys.argv) > 1 else AUDIT
    globals()["AUDIT"] = audit
    if not AUDIT.exists():
        print(f"FAIL  {AUDIT.relative_to(REPO)} not found")
        return 2
    text = AUDIT.read_text(encoding="utf-8")
    if HEADING not in text:
        # The table is the whole input. If it is renamed, this check would pass
        # over an empty scan and read as green, which is the failure mode it
        # exists to prevent -- so say so loudly instead.
        print(f"FAIL  '{HEADING}' not found; the table moved or was renamed. "
              f"This check scans nothing without it and would report a "
              f"misleading green.")
        return 2

    checkers = {}
    for path in sorted(glob.glob(str(REPO / "bench" / "check_*.py"))):
        p = Path(path)
        if p.name == Path(__file__).name:
            continue
        checkers[p.name] = p.read_text(encoding="utf-8", errors="replace")

    rows = [ln for ln in text[text.index(HEADING):].splitlines()
            if ln.startswith("|") and ln.count("|") >= 3]
    scanned = failures = 0
    for row in rows:
        cells = row.split("|")
        subject, guard = cells[1], "|".join(cells[2:])
        if not NOTHING.search(guard):
            continue
        sections = set(SECTION.findall(subject + guard))
        if not sections:
            continue
        scanned += 1
        hits = sorted(name for name, src in checkers.items()
                      if any(sec in src for sec in sorted(sections)))
        if hits:
            failures += 1
            print(f"  FAIL  a row claims NOTHING guards "
                  f"{', '.join(sorted(sections))}, but {', '.join(hits)} "
                  f"names that section")
            print(f"        row: {subject.strip()[:110]}")

    print(f"\n  scanned {scanned} row(s) claiming 'nothing' about a numbered "
          f"guide section, against {len(checkers)} checker(s)")
    if scanned == 0:
        print("  note  no row currently pairs a 'nothing' guard with a "
              "numbered section, so this run proves nothing. That is a "
              "property of the table, not a pass.")
    if failures:
        print(f"  {failures} stale 'enforced by nothing' claim(s)")
        return 1
    print("  ok    no row claims 'nothing' about something a checker names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
