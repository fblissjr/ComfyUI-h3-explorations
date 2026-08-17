"""The spine every red harness in this directory runs on.

A red harness proves a check can fail. Before this file, the three that existed
each re-invented module loading, a RED/GREEN reporter and a restore convention,
and the copy-paste diverged: one of the three lost its expected-outcome
comparison entirely and exited 0 whether every case came back red or every case
came back green. The recorded evidence for three rows of `docs/checks.md` was a
program that returned success unconditionally.

## The rule is derived, not authored

A case does not carry an expected verdict. It carries a KIND, and the
expectation follows from the baseline:

    MUTATION   the verdict must DIFFER from the unmutated baseline
    NEAR_MISS  the verdict must MATCH it

That distinction is the whole design. An authored expectation ("case M3 must be
red") is a claim needing its own verification, and every case added is another
one -- the regress that makes a harness suite collapse into tests for tests.
A derived expectation is one rule for every case, forever, and it doubles as the
needle check: a mutation that never reached the subject leaves the verdict
unchanged, which is exactly what MUTATION already asserts.

**An exception is not a difference.** A case that raises is ERRORED, never
"differed" -- otherwise any typo in a mutation reads as proof the check works.

## Exit codes

    0  every case behaved as its kind requires
    1  a case did not, or raised
    2  a fixture was absent, so nothing ran

Exit 2 exists because a silently skipped control reads exactly like a passing
one. `check_distill_settings.py` set that precedent and `docs/checks.md` names
it as the pattern worth copying.

## This file has its own control

Shared infrastructure fails silently across every harness at once, which is the
defect it was built to remove, one level up. `spine_control.py` runs two
fixtures through this module -- one whose mutation is inert, one healthy -- and
requires the first to fail and the second to pass. Neither expectation is
authored per case; both are structural.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]

MUTATION = "mutation"
NEAR_MISS = "near-miss"


def subject(rel: str):
    """Load a bench module by repo-relative path.

    The one copy of the importlib block. Every harness loaded its subject by
    path rather than by name because `bench/` is not a package; seven separate
    copies of these three lines existed across the experimental scripts.
    """
    path = REPO / rel
    if not path.exists():
        raise FileNotFoundError(f"subject not found: {rel}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load subject: {rel}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Harness:
    """Collects cases, judges them against a baseline, reports an exit code.

    `verdict` maps whatever the subject returns onto "is this red". The default
    reads a truthy return (an error list) as red; a subject reporting `ok=True`
    for green passes `verdict=lambda ok: not ok`.
    """

    def __init__(self, subject: str, verdict: Callable[[Any], bool] = bool):
        self.subject_path = subject
        self._verdict = verdict
        self._baseline = None
        self._rows: list[tuple[str, str, str, str]] = []
        self._bad = 0

    def fixture(self, path, why: str) -> "Harness":
        """Declare a file this harness cannot run without.

        Absent -> exit 2 with the reason, rather than a bare traceback from
        whatever tried to open it first.
        """
        p = Path(path)
        if not p.exists():
            print(f"  SKIP  fixture absent: {p}")
            print(f"        {why}")
            print("\ncould not run -- exit 2 so this is not read as a pass")
            raise SystemExit(2)
        return self

    def baseline(self, run) -> "Harness":
        try:
            self._baseline = self._verdict(run())
        except Exception:
            traceback.print_exc()
            print("\nbaseline raised -- nothing can be judged against it")
            raise SystemExit(1)
        state = "RED" if self._baseline else "GREEN"
        print(f"  baseline: {state}  (every judgement below is relative to this)")
        return self

    def case(self, label: str, kind: str, run) -> "Harness":
        if self._baseline is None:
            raise RuntimeError("baseline() must be called before any case()")
        if kind not in (MUTATION, NEAR_MISS):
            raise ValueError(f"unknown kind: {kind}")
        try:
            got = self._verdict(run())
        except Exception as exc:
            self._bad += 1
            self._rows.append(("ERROR", kind, label, f"{type(exc).__name__}: {exc}"))
            return self
        differs = got != self._baseline
        want_differs = kind == MUTATION
        mark = "RED" if got else "GREEN"
        if differs == want_differs:
            self._rows.append((mark, kind, label, ""))
        else:
            self._bad += 1
            why = (
                "mutation left the verdict unchanged -- it never reached the subject"
                if kind == MUTATION
                else "near-miss moved the verdict -- the check fires on input it must ignore"
            )
            self._rows.append((mark, kind, label, why))
        return self

    def report(self) -> int:
        print()
        for mark, kind, label, note in self._rows:
            flag = "  <-- WRONG" if note else ""
            print(f"  {mark:<5} {kind:<9} {label}{flag}")
            if note:
                print(f"        {note}")
        n = len(self._rows)
        print()
        if self._bad:
            print(f"{self._bad} of {n} case(s) did not behave as their kind requires")
            print(f"subject: {self.subject_path}")
            return 1
        print(f"all {n} case(s) behaved as their kind requires")
        print(f"subject: {self.subject_path}")
        return 0


def main(build) -> None:
    """Run a harness builder and exit on its report.

    Harnesses end with `main(build)` so the exit code is never forgotten -- the
    omission that made all three predecessors return success unconditionally.
    """
    sys.exit(build().report())
