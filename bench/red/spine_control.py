"""The control on `harness.py` itself.

A bug in shared harness infrastructure is inherited by every harness at once,
and it looks identical to the defect this directory was built to remove: a
program that reports success no matter what happened. So the spine gets the
same treatment it applies to its subjects.

Two fixtures, run as subprocesses so the real exit code is the evidence rather
than an in-process return value:

    _fixture_inert.py    one mutation that does not mutate  -> spine must exit 1
    _fixture_healthy.py  a real mutation and a real near-miss -> spine must exit 0

Neither expectation is authored per case. Both are structural properties of the
spine's own rule, which is why this is one control rather than a tower of them.

Run it directly. Exit 0 means the spine can go red for the right reason and
green for the right reason.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASES = [
    ("_fixture_inert.py", 1, "an inert mutation must be caught"),
    ("_fixture_healthy.py", 0, "a correct harness must pass"),
]


def run(name: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(HERE / name)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def main() -> int:
    bad = 0
    for name, want, why in CASES:
        code, out = run(name)
        ok = code == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<22} exit={code} want={want}  {why}")
        if not ok:
            for line in out.strip().splitlines():
                print(f"          {line}")
    print()
    if bad:
        print(f"{bad} of {len(CASES)} spine control(s) failed -- harness.py is not trustworthy")
        return 1
    print(f"all {len(CASES)} spine control(s) passed -- harness.py fails and passes for the right reasons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
