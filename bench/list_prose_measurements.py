#!/usr/bin/env python3
"""Inventory the measurements that live in prose, so they can be moved out.

`docs/prose_measurements.md` is the rule and the migration plan; this is the
instrument that sizes the backlog and shows closure. It is a REPORT, not a
check: it exits 0 whatever it finds, because a gate over prose that still
carries hundreds of legitimate-looking numbers would be red while the state is
correct, and CLAUDE.md says what that trains a reader to do.

What it catches, deliberately narrowly, in three classes: numbers wearing a
UNIT -- a multiplier (`2.4x`), a percentage (`63.2%`), a size (`25.28 GiB`), a
time (`691 s`), a rate (`12 it/s`), a count with a noun this repo measures
(`6300 rows`, `100k tokens`); SCIENTIFIC notation (`1.2e-5`), which is always
a magnitude; and a bare DECIMAL on a line that also carries a measurement word
(`residual`, `relative`, `cosine`, `error`, ...), added after the PDD lane
found residuals and relative errors written that way. `PATTERN_VERSION` says
which of these a record was taken under; a count from one version is not
comparable to a count from another. Those are the class whose home is a record
rather than a sentence. It does NOT try to find bare counts ("sixteen rows claim
calibration") -- `claim-audit` measured a regex at above 85% false positives on
that class, and a reader extracts those. Identifiers are excluded by
construction: a canvas (`1344x768`), a date, a version (`0.2.31`), a block
index, a token id -- a number that names a thing rather than measures one.

What it skips: fenced and inline code (a constant or a command IS the pointer
the rule asks for), and the dated-record set -- files whose job is to carry
numbers with their conditions. That set is `RECORD_PATTERNS` below; it is the
one judgement in this file, and a path added to it should say why.

    python bench/list_prose_measurements.py              # per-file totals
    python bench/list_prose_measurements.py --file docs/SOLATTN.md   # the lines
    python bench/list_prose_measurements.py --json out.json          # a record
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Prose that is governed: everything the wiki index routes to, plus the two
# files a session reads first.
GOVERNED_ROOTS = ["docs", "README.md", "CLAUDE.md", "VISION.md", ".claude/skills"]

# Dated records. Numbers belong here because the file carries when and under
# what conditions they were taken. Each line says why it is a record.
RECORD_PATTERNS = [
    r"^CHANGELOG\.md$",                       # per-version, past tense
    r"^docs/check_postmortems\.md$",          # frozen run logs, stale by design
    r"^docs/bench_plan\.md$",                 # pre-registered predictions and scores
    r"/\d{4}-\d{2}-\d{2}[_-]",                # a date in the filename is the observation point
    r"^docs/research/.*/brainstorming/",      # session-dated working notes
    r"^docs/wiki/index\.md$",                 # generated; fix the generator
    r"^docs/prompt_bank\.md$",                # generated; frame counts are grid points
    r"^docs/prompt_catalogue\.md$",           # generated from the graphs
    r"/arxiv_\d",                            # a transcribed paper; its numbers are the paper's
    r"^docs/rules_history\.md$",             # CLAUDE.md frozen at the 2026-09-03 cut
]

# Bump when a pattern below changes; written into every --json record.
PATTERN_VERSION = 2

UNIT = (
    r"(?:x|%|s|ms|min|minutes?|seconds?|h|hours?|"
    r"[KMGT]i?B|bytes?|"
    r"fps|it/s|tok/s|"
    r"rows?|tokens?|frames?|steps?|evaluations?|params?|shards?|blocks?|"
    r"seeds?|renders?|clips?|arms?|calls?|epochs?)"
)
# A number, optionally k/M-suffixed, optional space, then a unit at a word
# boundary. `1344x768` does not match: the `x` is followed by a digit.
MEASURE = re.compile(
    rf"(?<![\w.])(\d+(?:[.,]\d+)?[kKM]?)\s?{UNIT}(?![\w/])"
)
SCI = re.compile(r"(?<![\w.])\d+(?:\.\d+)?e[+-]?\d+(?![\w])")
# A bare decimal is a measurement only when the line says what it measures.
MEASURE_WORDS = re.compile(
    r"\b(residual|relative|rel\s?L2|cosine|error|delta|spread|ratio|accuracy|"
    r"psnr|update|floor|drift)\b", re.I)
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?!\d|\.\d|%|x|\w)")
# Things that look like the above but name rather than measure.
IDENTIFIER = re.compile(
    r"\d{4}-\d{2}-\d{2}"          # dates
    r"|\b\d+x\d+\b"               # canvases
    r"|\b\d+\.\d+\.\d+\b"         # versions
    r"|#\d+\b"                    # issue / PR numbers
    r"|\bsm\d+\b"                 # arch names
)
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")


def governed_files() -> list[Path]:
    out: list[Path] = []
    for root in GOVERNED_ROOTS:
        p = REPO / root
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.md")))
    return out


def is_record(rel: str) -> bool:
    return any(re.search(pat, rel) for pat in RECORD_PATTERNS)


def scan_file(path: Path) -> list[dict]:
    hits: list[dict] = []
    in_fence = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_CODE.sub(" ", raw)
        line = LINK_TARGET.sub("]()", line)
        line = IDENTIFIER.sub(" ", line)
        found = [(m.group(0), "unit") for m in MEASURE.finditer(line)]
        found += [(m.group(0), "sci") for m in SCI.finditer(line)]
        if MEASURE_WORDS.search(line):
            taken = {f[0] for f in found}
            found += [(m.group(0), "decimal") for m in DECIMAL.finditer(line)
                      if not any(m.group(0) in s for s in taken)]
        for text, kind in found:
            hits.append({
                "line": lineno,
                "match": text,
                "kind": kind,
                "table": raw.lstrip().startswith("|"),
                "text": raw.strip()[:160],
            })
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="print every hit in this one file")
    ap.add_argument("--json", help="write the full inventory to this path")
    ap.add_argument("--records", action="store_true",
                    help="also list the files skipped as dated records")
    args = ap.parse_args()

    inventory: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for path in governed_files():
        rel = path.relative_to(REPO).as_posix()
        if is_record(rel):
            skipped.append(rel)
            continue
        hits = scan_file(path)
        if hits:
            inventory[rel] = hits

    if args.file:
        rel = Path(args.file).as_posix()
        for h in inventory.get(rel, []):
            flag = (" [table]" if h["table"] else "") + (f" [{h['kind']}]" if h["kind"] != "unit" else "")
            print(f"{rel}:{h['line']}: {h['match']}{flag}\n    {h['text']}")
        if rel not in inventory:
            print(f"{rel}: no measurements found"
                  + (" (skipped as a dated record)" if is_record(rel) else ""))
        return 0

    totals = Counter({k: len(v) for k, v in inventory.items()})
    width = max((len(k) for k in totals), default=10)
    for rel, n in totals.most_common():
        in_tables = sum(h["table"] for h in inventory[rel])
        print(f"{n:5d}  {rel:<{width}}  ({in_tables} in tables)")
    print(f"\n{sum(totals.values())} measurements in prose across "
          f"{len(totals)} governed files; {len(skipped)} files skipped as records")
    if args.records:
        print("\nskipped as dated records:")
        for rel in skipped:
            print(f"  {rel}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "script": "bench/list_prose_measurements.py",
            "pattern_version": PATTERN_VERSION,
            "governed_roots": GOVERNED_ROOTS,
            "record_patterns": RECORD_PATTERNS,
            "total": sum(totals.values()),
            "per_file": dict(totals.most_common()),
            "skipped_records": skipped,
            "hits": inventory,
        }, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
