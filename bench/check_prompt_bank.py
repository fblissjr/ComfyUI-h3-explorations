#!/usr/bin/env python3
"""The prompt bank's gate, where the sweep can see it.

`bench/build_prompt_bank.py --check` is the gate on `prompt_bank/`: every
prompt grades clean at its own length, no entry has a shape problem that is
not recorded, and `docs/prompting/prompt_bank.md` is the table the bank would write
today. `bench/run_checks.py` runs files named `check_*.py`, and the gate is
not named that way, so a sweep never ran it. On 2026-10-06 it was found red
at a commit whose sweeps had all been green (a bank entry ending in a
newline, with two shape findings). This file is the gate under a name the
sweep runs. It grades nothing itself: one copy of the rules, in the builder.

The exit code is the builder's: 0 clean, 1 a finding or a stale table, 2 an
empty bank, which is nothing graded.

    <python> bench/check_prompt_bank.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
GATE = BENCH / "build_prompt_bank.py"
# the builder prints one `noted` line per entry whose findings are recorded; those are not findings
QUIET = ("  noted ",)


def main() -> int:
    ran = subprocess.run([sys.executable, str(GATE), "--check"], capture_output=True, text=True)
    for line in (ran.stdout + ran.stderr).splitlines():
        if not line.startswith(QUIET):
            print(line)
    if ran.returncode not in (0, 1, 2):
        print(f"FAIL  bench/build_prompt_bank.py --check exited {ran.returncode}: it crashed before grading")
        return 1
    return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
