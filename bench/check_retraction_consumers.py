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

## `ALLOW: (none)` -- the phrase must appear nowhere

A claim deleted repo-wide rather than caveated has no consumers left, and its
row cannot name one. Until 2026-08-25 the only ways to express that were to
delete the row, which throws the tripwire away, or to plant a deliberate
mention in the ledger's own prose so the row had something to match --
`docs/evidence.md` does exactly that for `2.7x more accurate` and for
`bypassed_for_capture`, each spelled once and saying so.

`ALLOW: (none)` says it directly: this phrase belongs in no file, and any
occurrence is an unlisted consumer. It is spelled out rather than left as an
empty `ALLOW:` so that a typo or a truncated line still fails the parse
instead of quietly becoming the strictest row in the ledger.

What made this worth building rather than working around a third time: the
`replaced rather than adjusted` row warned from 2026-08-20, and the remedy the
warning itself printed -- drop the file from the row -- would have emptied the
row's only entry and turned the warn into `FAIL parses_the_ledger`. The advice
was unreachable for exactly the case that produces it most often.

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

# Spelled out rather than an empty `ALLOW:`, so a truncated or mistyped line
# still fails the parse instead of silently becoming a must-appear-nowhere row.
NOWHERE = "(none)"


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


def normalise(text):
    """Collapse whitespace and case so line-wrapped prose still matches.

    Added 2026-08-14 after this check passed over a live consumer. `h3_capture.py`
    carried "sage gets nothing" split across a line break -- `"sage gets "` then
    `"nothing"` -- and a raw substring test walked straight past it. Prose is
    wrapped at 79 columns everywhere in this repo, so a multi-word phrase is
    *more* likely than not to straddle a break, which made the naive matcher
    close to useless on exactly the phrases worth tracking.

    Case-folding is the same class of hole: "Zero DiT calls" opening a sentence.
    """
    # Python joins adjacent string literals, so a phrase can be split by a
    # `" ... "` seam as well as a newline -- h3_capture.py hid a live consumer
    # that way, past the whitespace fix. Collapse the seam the way the parser
    # does, then the whitespace, then case.
    text = re.sub(r"[\"']\s*[\"']", "", text)
    return re.sub(r"\s+", " ", text).casefold()


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
            cur = {
                "phrase": line.split(":", 1)[1].strip(),
                "allow": [],
                "nowhere": False,
                "why": "",
            }
            rows.append(cur)
        elif line.startswith("ALLOW:"):
            if cur is None:
                raise SystemExit("FAIL parses_the_ledger: ALLOW before PHRASE")
            names = line.split(":", 1)[1].split()
            cur["nowhere"] = names == [NOWHERE]
            cur["allow"] = [] if cur["nowhere"] else names
        elif line.startswith("WHY:"):
            if cur is None:
                raise SystemExit("FAIL parses_the_ledger: WHY before PHRASE")
            cur["why"] = line.split(":", 1)[1].strip()
    if not rows:
        raise SystemExit("FAIL parses_the_ledger: block present but empty")
    for row in rows:
        if not row["allow"] and not row["nowhere"]:
            raise SystemExit(
                f"FAIL parses_the_ledger: {row['phrase']!r} lists no files. "
                f"Write `ALLOW: {NOWHERE}` if the phrase is meant to appear "
                "nowhere; a bare empty ALLOW is treated as a typo."
            )
    return rows


def main():
    rows = parse_ledger(LEDGER.read_text(encoding="utf-8"))
    nowhere = sum(1 for row in rows if row["nowhere"])
    print(f"  ok    parses_the_ledger   {len(rows)} retracted phrase(s), "
          f"{nowhere} of them must appear nowhere")

    contents = {}
    for path in iter_files():
        name = str(path.relative_to(REPO))
        try:
            contents[name] = normalise(scannable(name, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue

    unlisted, stale = [], []
    for row in rows:
        needle = normalise(row["phrase"])
        found = {name for name, text in contents.items() if needle in text}
        allowed = set(row["allow"])
        for name in sorted(found - allowed):
            unlisted.append((row["phrase"], name))
        for name in sorted(allowed - found):
            stale.append((row["phrase"], name, len(allowed) == 1))

    for phrase, name, only in stale:
        # Telling someone to drop the last file on a row would empty it, and an
        # empty ALLOW fails the parse. That advice was unreachable for the case
        # that produces this warning most often.
        remedy = (
            f"it was the only file on that row, so write `ALLOW: {NOWHERE}` "
            "to keep the tripwire"
            if only else "drop it from that row"
        )
        print(f"  warn  stale_allowlist    {name} no longer contains "
              f"{phrase!r} -- {remedy}")
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
