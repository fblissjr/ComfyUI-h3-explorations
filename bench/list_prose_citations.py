#!/usr/bin/env python
"""List every PROSE citation of a `bench/results/` record, for triage.

A prose citation is a result filename appearing in a comment, a docstring, or
markdown -- as opposed to a path the code opens. The distinction matters
because the two rot differently. A runtime path fails loudly when the file
moves; a prose citation goes stale in place, and the sentence around it keeps
asserting whatever it asserted the day it was written.

This repo cites results in prose deliberately: `workflows/h3_config.py` points
at the record behind a shipped default rather than copying the number into the
comment ("a number copied into this comment is a second copy"). That makes the
citation load-bearing -- it is what keeps the default attributable -- and it
also makes it the thing nobody re-reads. Hence this worksheet.

**This script judges nothing.** It reports where each citation is, what the
surrounding sentence claims, and whether the cited file still exists and where.
Whether a claim is still TRUE is a human call, which is the column left blank.

Usage:

    python bench/list_prose_citations.py                  # write the worksheet
    python bench/list_prose_citations.py --stdout         # print instead

The output is a dated snapshot. Regenerate it after acting on the verdicts;
the diff is the record of what was resolved.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
#: Any dated record name. The `jsonl` alternative must precede `json` -- with
#: the other order the engine matches the shorter one and silently truncates
#: every `.jsonl` name to a `.json` that exists nowhere. That bug produced a
#: wrong count once already.
NAME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9_.\-]+?\.(?:jsonl|json|md|log)")


def locate() -> dict[str, str]:
    """Every result record -> where it lives, relative to `bench/results/`."""
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(RESULTS):
        rel = Path(root).relative_to(RESULTS)
        for f in files:
            if f.endswith(".py"):
                continue
            out[f] = "results/" if rel == Path(".") else f"archive/{rel}/"
    return out


def code_lines(path: Path) -> set[int]:
    """Line numbers holding a string literal that is NOT a docstring.

    Those are the runtime paths; everything else naming a record is prose.
    Parsed rather than pattern-matched, because a filename inside a comment and
    one inside an argparse default look identical to a regex.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    docs = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and ast.get_docstring(n, clean=False)
    }
    return {
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docs
        and NAME.search(n.value)
    }


def claim(lines: list[str], i: int) -> str:
    """The sentence around line `i`, so a verdict needs no second file open."""
    lead = re.compile(r"^\s*(#:|#|//|\*|-|\||>)?\s?")
    take = [lead.sub("", lines[j]).strip() for j in range(max(0, i - 2), min(len(lines), i + 3))]
    return " ".join(t for t in take if t)[:400]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    where = locate()
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=REPO, check=True
    ).stdout.split()

    found: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    for rel in tracked:
        if rel.startswith("bench/results/") or not rel.endswith((".py", ".md")):
            continue
        p = REPO / rel
        try:
            lines = p.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        skip = code_lines(p) if rel.endswith(".py") else set()
        for i, line in enumerate(lines):
            if i + 1 in skip:
                continue
            for name in dict.fromkeys(NAME.findall(line)):
                if name not in where:
                    continue
                found[rel].append((i + 1, name, where[name], claim(lines, i)))

    total = sum(len(v) for v in found.values())
    archived = sum(1 for v in found.values() for r in v if r[2] != "results/")

    def historical(rel: str) -> bool:
        """Past tense by design, so "no longer true" is not a defect there.

        A changelog entry describes what was believed at a version; a closed
        lane's own write-ups describe what that lane found. Both are supposed
        to freeze. Triaging them against today's code would generate work whose
        correct outcome is always "leave it", which is how a worksheet trains
        you to skim.
        """
        return (
            rel == "CHANGELOG.md"
            or rel.startswith("docs/research/qwen3-vl-special-tokens-post-training/")
            or rel.startswith("docs/check_postmortems")
            or "/postmortem" in rel
        )

    body = [
        f"# Prose citations of `bench/results/` records -- {date.today().isoformat()}",
        "",
        "Generated by `bench/list_prose_citations.py`. It judges nothing: every",
        "row is a place where a comment, docstring or doc names a result record.",
        "The **Verdict** column is yours.",
        "",
        "Suggested verdicts: `TRUE` (claim still holds), `STALE` (was true, is",
        "not now -- correct it and say what it used to claim), `IRRELEVANT` (the",
        "record no longer bears on anything; the citation can go), `MOVE` (claim",
        "holds but belongs somewhere else).",
        "",
        "**Before deleting a citation, check what it is holding up.** Several of",
        "these are the only attribution a shipped default has -- `h3_config.py`",
        "points at the record precisely so the number is not copied into the",
        "comment. Removing one of those turns a measured value into an inherited",
        "one, which is the failure `docs/SOLATTN.md` audits.",
        "",
        f"Citations: {total}, across {len(found)} files. "
        f"Pointing at an archived record: {archived}.",
        "",
    ]

    def section(title: str, note: str, keys: list[str]) -> None:
        body.extend([f"# {title}", "", note, ""])
        for rel in keys:
            body.extend([f"## `{rel}`", "", "| line | record | lives in | claim | Verdict |",
                         "|---|---|---|---|---|"])
            for ln, name, loc, text in sorted(found[rel]):
                safe = text.replace("|", "\\|").replace("`", "'")
                body.append(f"| {ln} | `{name}` | `{loc}` | {safe} |  |")
            body.append("")

    live = sorted(r for r in found if not historical(r))
    past = sorted(r for r in found if historical(r))
    section(
        "Operative -- triage these",
        "Prose a reader acts on today: node and config comments, the working "
        "docs, and script docstrings. A stale claim here misleads someone "
        "making a decision, which is the whole cost being paid.",
        live,
    )
    section(
        "Historical -- expected to be frozen",
        "Changelog entries, closed-lane write-ups and postmortems. These "
        "describe what was believed at a point in time and are SUPPOSED to go "
        "out of date; correct one only if it was wrong when written. Listed "
        "for completeness, not for triage.",
        past,
    )

    text = "\n".join(body)
    if args.stdout:
        print(text)
        return 0
    out = args.out or RESULTS / f"{date.today().isoformat()}_prose_citations.md"
    out.write_text(text, encoding="utf-8")
    print(f"{total} citations across {len(found)} files -> {out.relative_to(REPO)}")
    print(f"  pointing at an archived record: {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
