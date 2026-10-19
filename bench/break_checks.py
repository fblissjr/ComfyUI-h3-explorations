#!/usr/bin/env python3
"""Break the code on purpose, in a copy, and see whether a check notices: with the unbroken copy run FIRST.

Nothing here is required of a check (`docs/method/checks.md`: the owner retired "shown red" on 2026-09-11). It exists
because sessions do it anyway before trusting a new case, and on 2026-10-10 one did it wrong: fourteen breaks were
reported as seen from copies that could not import ComfyUI, so every run was red whatever was broken, and four of
the breaks had never been seen by any case. Three things make a break evidence, and this tool does all three:

- THE CONTROL. Each check named by a break is run first on the unbroken copy, by the same command, in the same
  place. If that is not green nothing else is run.
- A COPY SHAPED LIKE THE REAL THING. The checks find ComfyUI two folders above the pack. The copy is made at
  `<temp>/custom_nodes/<the pack's name>` with every entry of ComfyUI linked beside `custom_nodes`, so a check
  imports what it imports in the tree. The tree itself is never edited: other sessions run checks in it.
- WHY IT WAS RED. A break counts as seen only when the check exits non-zero AND its output has no import error;
  the first failing line is printed so a reader can see it names the behaviour that was broken.

    <python> bench/break_checks.py BREAKS.json [--keep]

BREAKS.json is a list of {"file": a path in the pack, "old": text that occurs exactly once in it, "new": what
replaces it, "check": a check's name without `bench/` or `.py`}. Each break is applied alone and restored.

Prints one line a break: SEEN, NOT SEEN (the check stayed green: no case covers it, or the change does nothing),
NOT APPLIED (the text was not found exactly once), WRONG REASON (red by an import error). Exits 0 only when the
controls are green and every break was applied and seen. A break that is NOT SEEN because it changes nothing is
for a person to say; this tool cannot tell an equivalent change from a missing case.

No model, no card. It copies the pack without `.git`, `coderef`, `internal`, `data`, `docs` and caches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent
#: Left out of the copy: history, other checkouts, private and captured data, prose, caches. A check that reads one
#: of these cannot be run here, and says so by failing its control.
LEFT_OUT = (".git", "coderef", "internal", "data", "docs", "__pycache__", "archive", "assets", "standalone", "*.pyc")
SEEN, NOT_SEEN, NOT_APPLIED, WRONG_REASON = "SEEN", "NOT SEEN", "NOT APPLIED", "WRONG REASON"
IMPORT_ERRORS = ("ModuleNotFoundError", "ImportError")


def shaped_copy(root: Path) -> Path:
    """A copy of the pack at `root/custom_nodes/<name>`, with ComfyUI's own entries linked beside `custom_nodes`."""
    (root / "custom_nodes").mkdir(parents=True)
    for entry in COMFY.iterdir():
        if entry.name != "custom_nodes":
            (root / entry.name).symlink_to(entry)
    for sibling in (COMFY / "custom_nodes").iterdir():       # a check may import another pack's module by path
        if sibling.resolve() != REPO.resolve() and sibling.is_dir():
            (root / "custom_nodes" / sibling.name).symlink_to(sibling)
    copy = root / "custom_nodes" / REPO.name
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(*LEFT_OUT), symlinks=True)
    return copy


def run_check(copy: Path, check: str, python: str) -> tuple[int, list[str]]:
    """A check's exit code in the copy, and the lines of its output that say what failed."""
    done = subprocess.run([python, str(copy / "bench" / f"{check}.py")], capture_output=True, text=True, cwd=copy,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines = [line.strip() for line in (done.stdout + done.stderr).splitlines()]
    return done.returncode, [line for line in lines if line.startswith(("FAIL", "Traceback")) or any(w in line for w in IMPORT_ERRORS)
                             or "Error:" in line]


def judge(code: int, said: list[str]) -> str:
    if code == 0:
        return NOT_SEEN
    return WRONG_REASON if any(w in line for line in said for w in IMPORT_ERRORS) else SEEN


def break_checks(breaks: list[dict], python: str, keep: bool = False) -> tuple[dict[str, bool], list[dict]]:
    """The controls ({check: green}) and one result a break ({"outcome", "said", ...the break}). See the module docstring."""
    root = Path(tempfile.mkdtemp(prefix="h3_break_checks_"))
    try:
        copy = shaped_copy(root)
        controls = {check: run_check(copy, check, python)[0] == 0 for check in sorted({b["check"] for b in breaks})}
        results = []
        if all(controls.values()):
            for b in breaks:
                path = copy / b["file"]
                source = path.read_text() if path.is_file() else ""
                if source.count(b["old"]) != 1:
                    results.append({**b, "outcome": NOT_APPLIED, "said": [f"`old` occurs {source.count(b['old'])} time(s) in {b['file']}"]})
                    continue
                path.write_text(source.replace(b["old"], b["new"]))
                try:
                    code, said = run_check(copy, b["check"], python)
                finally:
                    path.write_text(source)
                results.append({**b, "outcome": judge(code, said), "said": said})
        return controls, results
    finally:
        if keep:
            print(f"the copy is kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    p.add_argument("breaks", help="a JSON list of {file, old, new, check}")
    p.add_argument("--keep", action="store_true", help="leave the copy where it is and say where")
    a = p.parse_args()
    breaks = json.loads(Path(a.breaks).read_text())
    controls, results = break_checks(breaks, sys.executable, keep=a.keep)
    for check, green in controls.items():
        print(f"CONTROL       {check}: {'green' if green else 'RED on the unbroken copy: nothing else was run'}")
    for r in results:
        first = (r["said"][0] if r["said"] else "").replace("\n", " ")
        print(f"{r['outcome']:13s} {r['check']}: {r['new'].strip().splitlines()[0][:60] if r['new'].strip() else '(text removed)'} | {first[:140]}")
    return 0 if controls and all(controls.values()) and results and all(r["outcome"] == SEEN for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
