#!/usr/bin/env python3
"""Every check on disk has a row in the index, and every row names a real file.

`docs/checks.md`'s index is the repo's answer to "what does this repo check". A
check that exists and is not listed is invisible to anyone reading that answer,
and a row naming a file that is gone teaches the next reader something false.
Nothing watched either direction until 2026-08-17.

## Why this earned its place

The repo's rule is no new check until a drift instance appears that the existing
gates provably could not have caught. This one has two, hours apart, from
different authors:

- `check_graph_discovery.py` was on disk with no row, found in the morning.
- `check_capture_manifest.py` landed the same afternoon the same way.

Neither was caught, because `check_doc_links.py` verifies that citations *resolve*
and says nothing about whether a file that exists is cited at all. That is the
gap, and it is exactly one grep wide.

## Why this is not the check that was refuted

An earlier attempt tried to detect whether a requirement was *enforced*, by
searching `bench/` for something that looked like it enforced the prose. That was
refuted: enforcement is semantic, the search could not go green on a correct
table, and every version either false-positived or laundered the answer through an
exemption list seeded from the file under test.

This compares two directory listings. There is no judgment in it and no
expectation derived from the thing being checked -- the index is read, the
filesystem is read, and the two sets are differenced.

## Deliberately not asserted

**Rows that are not `check_*.py`.** `preflight_graph.py`, `smoke_h3.py` and
`count_packed_rows.py` are listed on purpose and are not checks; the index says
why. So the disk-to-index direction covers `check_*.py` only, while the
index-to-disk direction covers every `.py` a row names. Asserting a count of rows
would fail the moment a fourth non-check is listed, which is a legitimate change.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKS_MD = REPO / "docs" / "checks.md"

# A row's subject is the first backticked *.py in it. The index's own convention
# is that the subject leads the row, and every row has followed it since the file
# was written.
_ROW_SUBJECT = re.compile(r'^\|\s*`([A-Za-z0-9_./-]+\.py)`')


def index_section(md_text: str) -> list[str]:
    """The index table's data rows, header and separator excluded.

    Takes text rather than reading the file so a harness can feed it a synthetic
    table. A collector that read the real file could not be shown red.
    """
    rows, inside = [], False
    for line in md_text.splitlines():
        if line.startswith("## The index"):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if not inside or not line.startswith("|"):
            continue
        if re.match(r"^\|[\s:|-]+\|\s*$", line) or line.startswith("| check "):
            continue
        # A leading pipe is not enough. The index has five columns, and prose that
        # happens to begin with `|` would otherwise be read as a row with no `.py`
        # subject and reported as an error -- a false red, which this repo rates
        # worse than no check. Caught by this check's own harness (case G2) before
        # it ever ran on the real file.
        if len([c for c in line.strip().strip("|").split("|")]) < 5:
            continue
        rows.append(line)
    return rows


def audit(md_text: str, check_files: list[str]) -> list[str]:
    """Difference the index against a list of check filenames. Pure."""
    rows = index_section(md_text)
    if not rows:
        return ["the index section is empty or unparseable -- refusing to pass on silence"]

    subjects = {m.group(1) for r in rows if (m := _ROW_SUBJECT.match(r))}
    errs = []

    for f in sorted(check_files):
        if os.path.basename(f) not in subjects:
            errs.append(f"{os.path.basename(f)} is on disk with no row in the index")

    for r in rows:
        m = _ROW_SUBJECT.match(r)
        if not m:
            errs.append(f"a row names no .py subject: {r[:60]}...")
            continue
        named = m.group(1)
        candidates = [REPO / named, REPO / "bench" / named]
        if not any(c.exists() for c in candidates):
            errs.append(f"the index names {named}, which is not on disk")

    return errs


def main() -> int:
    md = CHECKS_MD.read_text(encoding="utf-8")
    checks = sorted(glob.glob(str(REPO / "bench" / "check_*.py")))
    # This file is a check and is indexed like any other; no self-exemption.
    errs = audit(md, checks)

    n_rows = len(index_section(md))
    print(f"doc inventory: {len(checks)} check(s) on disk, {n_rows} index row(s)")
    if errs:
        for e in errs:
            print(f"  FAIL  {e}")
        print(f"\n{len(errs)} inventory error(s). The index is the repo's answer to what it")
        print("checks; a check missing from it is invisible and a row naming a gone file lies.")
        return 1
    print("  ok    every check is indexed, and every row names a file that exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
