#!/usr/bin/env python
"""Fail when a retracted claim reaches a file nobody signed off on.

A retraction is done when every **consumer** of the claim is enumerated, not
when the claim is corrected. On 2026-08-14 that distinction cost four separate
wrong statements: "sage gets nothing" was retracted at the top of
`docs/SOLATTN.md` and went on producing wrong answers in the Ordering section,
in a bench-plan paragraph, and in `h3_capture.py`'s docstring -- where it would
have produced a *number* from a skewed sample of timesteps. Separately, the
23-point reference-load swing was retracted in `docs/evidence.md` while still
carrying the argument for an entire bench run.

The claims were all corrected. Nobody enumerated who was quoting them.

**Why this is an allowlist and not a grep-and-judge.** "Is this mention
caveated" is not mechanically decidable. `docs/bench_plan.md` contains the
string "zero DiT calls" inside a bullet that begins "RETRACTED 2026-08-14",
which is correct; a checker that flags it trains readers to skim its output,
and CLAUDE.md's standard is that a check reporting red while the state is
correct is worse than no check. Worse, the same string can be two claims:
`attention.py` says "2.7x" about kernel speed against torch flash, which is
correct and unrelated to the retracted 2.7x accuracy figure.

So this asks one decidable question instead: **has this phrase reached a file
that is not on its list.** That is the failure mode that actually occurred --
all four consumers were files that acquired the claim after the retraction.

## What it does not defend

**The retracted thing is sometimes a pairing, not a token.** `docs/bench_plan.md`
read "one 345-frame video reference" where 345 is the *reference* length and
looks like a legal shipped value, concealing that the *target* was 362, which
is not legal. Nothing in that sentence is a matchable phrase. No configuration
of this check finds it, and a second reader did.

It also cannot tell a correct mention from a wrong one inside an allowed file.
Adding a file to a row asserts that someone read that occurrence.

## Where the data lives

`docs/evidence.md`, in a fenced ```retraction-consumers block -- with the
ledger it belongs to, so the enumeration cannot drift from the retraction it
enumerates. h3_config.py's rule applied to prose: nothing gets a second copy.

Claims, i.e. what breaks if a case is deleted:

- `parses_the_ledger`      -- a malformed or missing block fails loudly rather
                              than silently checking nothing. This check's own
                              green must mean something.
- `no_unlisted_consumers`  -- the real gate. A retracted phrase in a file not
                              on its row's list.
- `stale_allowlist`        -- an allowlisted file that no longer contains the
                              phrase. WARNS rather than fails: it means someone
                              cleaned up, which is good, and failing on it would
                              punish the cleanup. But an allowlist that has
                              drifted is no longer an enumeration, so it is
                              surfaced rather than swallowed.

Needs no ComfyUI, no CUDA, no model. Runs in well under a second.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "evidence.md"

# Prose only. Generated graphs are regenerated from build_workflows.py, so a
# phrase reaching one of them is a finding about the generator, which is a
# .py file and is covered.
SUFFIXES = (".md", ".py")

# internal/ is gitignored research and session logs; it records what was
# believed at the time on purpose, and holding it to the current retraction
# state would make every postmortem fail the moment it was superseded.
SKIP_DIRS = {".git", "internal", "__pycache__", "coderef", "vendor", "archive"}


def iter_files():
    for path in sorted(REPO.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if SKIP_DIRS & set(path.relative_to(REPO).parts):
            continue
        # This file quotes every phrase it hunts, in a docstring explaining
        # why each one is here. Scanning itself made it fail on first run --
        # correctly, and uselessly. The enumerator is not a consumer.
        if path.resolve() == Path(__file__).resolve():
            continue
        yield path


def scannable(name, text):
    """The prose of a file, minus any definition block.

    The fenced `retraction-consumers` block in the ledger *is* the enumeration
    -- it necessarily contains every phrase, and counting it as a consumer
    would make the ledger permit itself. Strip it and scan the surrounding
    prose, which is ordinary text and is held to the same rule as anywhere
    else.
    """
    if name == str(LEDGER.relative_to(REPO)):
        return re.sub(r"```retraction-consumers\n.*?```", "", text, flags=re.S)
    return text


def parse_ledger(text):
    """Pull the fenced retraction-consumers block into (phrase, allow, why)."""
    m = re.search(r"```retraction-consumers\n(.*?)```", text, re.S)
    if not m:
        raise SystemExit(
            "FAIL parses_the_ledger: no ```retraction-consumers block in "
            f"{LEDGER.relative_to(REPO)}. This check cannot silently pass "
            "with nothing to check."
        )
    rows, cur = [], None
    for line in m.group(1).splitlines():
        if line.startswith("PHRASE:"):
            cur = {"phrase": line.split(":", 1)[1].strip(), "allow": [], "why": ""}
            rows.append(cur)
        elif line.startswith("ALLOW:"):
            if cur is None:
                raise SystemExit("FAIL parses_the_ledger: ALLOW before PHRASE")
            cur["allow"] = line.split(":", 1)[1].split()
        elif line.startswith("WHY:"):
            if cur is None:
                raise SystemExit("FAIL parses_the_ledger: WHY before PHRASE")
            cur["why"] = line.split(":", 1)[1].strip()
    if not rows:
        raise SystemExit("FAIL parses_the_ledger: block present but empty")
    for row in rows:
        if not row["allow"]:
            raise SystemExit(
                f"FAIL parses_the_ledger: {row['phrase']!r} lists no files. "
                "An empty allowlist would pass trivially."
            )
    return rows


def main():
    rows = parse_ledger(LEDGER.read_text(encoding="utf-8"))
    print(f"  ok    parses_the_ledger   {len(rows)} retracted phrase(s)")

    contents = {}
    for path in iter_files():
        name = str(path.relative_to(REPO))
        try:
            contents[name] = scannable(name, path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue

    unlisted, stale = [], []
    for row in rows:
        found = {name for name, text in contents.items() if row["phrase"] in text}
        allowed = set(row["allow"])
        for name in sorted(found - allowed):
            unlisted.append((row["phrase"], name))
        for name in sorted(allowed - found):
            stale.append((row["phrase"], name))

    for phrase, name in stale:
        print(f"  warn  stale_allowlist    {name} no longer contains "
              f"{phrase!r} -- drop it from that row")
    if not stale:
        print("  ok    stale_allowlist     every allowlisted file still has its phrase")

    if unlisted:
        print()
        for phrase, name in unlisted:
            print(f"  FAIL  no_unlisted_consumers  {phrase!r} appears in {name}, "
                  f"which is not on its row in docs/evidence.md")
        print(
            "\nA retracted claim reached a file nobody signed off on. Either the\n"
            "occurrence is wrong and should be fixed, or it is correct in context\n"
            "and the file belongs on that row -- with a WHY saying who read it.\n"
            "Do not add it just to silence this."
        )
        return 1

    print(f"  ok    no_unlisted_consumers  {len(rows)} phrase(s) confined to "
          "their enumerated files")
    print("\nall ok -- every retracted claim is where somebody said it could be")
    return 0


if __name__ == "__main__":
    sys.exit(main())
