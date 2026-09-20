#!/usr/bin/env python3
"""Red control for `preflight_graph.py::speaker_id_rules`.

The rule is ref-en 5.4's, STATED: "Assign `(Sx)` once according to the order of
actual vocal events in the target video. Reuse the corresponding ID at every
actual vocal event in `detailed_description`." The same paragraph states the
exemption as a prohibition: a verbal cue inside a "directly reused BGM or
complete soundtrack" that no person physically produces takes `<Audio N>` as
its audible source and you "do not invent an additional `(Sx)`".

Exists because the rule shipped green. `4bd7b429` (0.128.0) fixed four
unattributed vocal events in the two composed reference scenes by hand, after
a full read of the bank; nothing in the repo could have found them, because
`docs/prompting.md` section 11's speaker-id row read `nothing` in its
checked-by column. A check whose only evidence is that it passes today would
be indistinguishable from one that cannot fire, which is the rule in
CLAUDE.md, so the input it was built against is reconstructed from git rather
than described in prose.

Cases, all offline and in seconds -- no GPU, no server, no fixture files:

  kitchen_pre    ref2va_scene_kitchen at 4bd7b429^   -> FAIL (4 unattributed)
  subway_pre     ref2va_scene_subway  at 4bd7b429^   -> FAIL (4 unattributed)
  kitchen_now    the same entry at HEAD              -> clean
  subway_now     the same entry at HEAD              -> clean
  soundtrack     ref2va_soundtrack_fully_copy        -> clean (the exemption)
  base_is_noted  a base-mode entry with the same shape -> note, never FAIL

`base_is_noted` is the case that keeps the rule inside the guide. base-en
states the id and its stability and says nothing about repeating it per vocal
event; only its two examples show one per line, and an example is not a rule
(`docs/prompting.md` section 12.13). A future edit that promotes the base note
to a FAIL turns this case red, which is the point of it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

import preflight_graph as pf  # noqa: E402

# The commit that fixed them by hand. Its parent is the tree the rule was
# built against; both are read through git, so this file carries no copy of
# the prompt text that could drift from either.
FIX = "4bd7b429"
MARK = "no (Sx) earlier in its shot"


def _at(rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _rows(text: str, guide: str) -> list[tuple[str, str]]:
    """Grade just the main field, which is what the rule reads."""
    field = pf.MAIN_FIELD[guide]
    sec = pf.split_sections(text, pf.REF_SECTIONS if guide == "ref"
                            else pf.BASE_SECTIONS)
    return pf.speaker_id_rules(sec.get(field, ""), guide)


def _count(rows, level: str) -> int:
    return sum(1 for lvl, msg in rows if lvl == level and MARK in msg)


def main() -> int:
    cases: list[tuple[str, str, int, int]] = []  # name, verdict, fails, notes
    bad: list[str] = []

    pre = {stem: _at(f"{FIX}^", f"prompt_bank/{stem}.txt")
           for stem in ("ref2va_scene_kitchen", "ref2va_scene_subway")}
    if any(v is None for v in pre.values()):
        print(f"FAIL  cannot read prompt_bank/ at {FIX}^ -- this control "
              f"proves nothing without the tree the rule was built against")
        return 2
    pre_text: dict[str, str] = {k: v for k, v in pre.items() if v is not None}

    for stem, text in pre_text.items():
        n = _count(_rows(text, "ref"), "FAIL")
        cases.append((f"{stem.split('_', 1)[1]}_pre", "FAIL expected", n, 0))
        if n == 0:
            bad.append(f"{stem} at {FIX}^ grades clean; the rule cannot fire "
                       f"on the input it was built against")

    for stem in ("ref2va_scene_kitchen", "ref2va_scene_subway",
                 "ref2va_soundtrack_fully_copy"):
        text = (REPO / "prompt_bank" / f"{stem}.txt").read_text()
        n = _count(_rows(text, "ref"), "FAIL")
        cases.append((stem, "clean expected", n, 0))
        if n:
            bad.append(f"{stem} at HEAD grades {n} FAIL; the shipped bank "
                       f"must be clean under its own rule")

    base = (REPO / "prompt_bank" / "t2va_hardware_aisle_short.txt").read_text()
    rows = _rows(base, "base")
    f, nt = _count(rows, "FAIL"), _count(rows, "note")
    cases.append(("base_is_noted", "note, never FAIL", f, nt))
    if f:
        bad.append("a base-mode prompt graded FAIL; base-en states the id and "
                   "its stability, not its reuse per vocal event (prompting.md "
                   "12.13), so this rule reports there and does not grade")
    if nt == 0:
        bad.append("the base-mode note did not fire, so the report half of "
                   "the rule is inert and its absence would be invisible")

    for name, verdict, f, nt in cases:
        mark = "ok   " if not bad else "     "
        print(f"  {mark} {name:<32} {verdict:<18} {f} FAIL, {nt} note")
    print()
    for b in bad:
        print(f"  FAIL  {b}")
    if bad:
        return 1
    print("  ok    the rule fires on the text it was written against, is "
          "silent on the fixed text and on the guide's soundtrack exemption, "
          "and grades nothing in the base modes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
