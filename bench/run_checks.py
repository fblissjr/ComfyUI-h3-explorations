#!/usr/bin/env python3
"""Run every `bench/check_*.py` with the card masked, and say what is red that should not be.

    <comfy venv python> bench/run_checks.py
    <comfy venv python> bench/run_checks.py --only sol --logs /tmp/sweep
    <comfy venv python> bench/run_checks.py --changed

One row per check: its exit code, its wall time, and the line that explains a
result that is not green. Then a verdict against `bench/checks_baseline.json`.

## What the three results mean

    0      green: every case the check has was graded and passed.
    2      nothing graded: the check needs something this run does not have (a
           card, a dataset, the server). Expected when masked, never a failure.
    other  red, including a crash and a timeout.

`docs/checks.md` "Running them" is the standard for those codes. A check that
needs a resource says so through `bench/_lib::needs`.

## The baseline

`bench/checks_baseline.json` names the checks that are red for a stated reason,
one line each. **This script exits 1 only when a check is red and the baseline
does not name it**, so a sweep before a commit has one answer. A baseline entry
that is no longer red is printed as STALE and does not change the exit code;
delete the entry, because a stale one would cover the next real red.

## The targeted sweep

`--changed` runs the checks that the files changed in the working tree can
turn red, and says which and why. `--since <commit>` adds the files changed
since that commit. The rule:

- **A node file changed: every check runs.** A node file is a module at the
  repo root, the shared constants, the generator, a generated graph, a
  vendored config or this harness (`is_node_file`). The owner's rule is the
  full sweep before a schema, default or loader commit, and nothing short
  of reading the diff can say a root module's change is not one.
- **Otherwise:** a changed check itself, every check whose source names a
  changed file (by its module name for a script, by its file name for
  anything else), and `ALWAYS`, the checks that read the whole tree.

It is a way to be quick on a bench tool or a record, never a substitute for
the full sweep where the rule above asks for one; it prints which it was.

## What a sweep is not allowed to do

Every check is started with `CUDA_VISIBLE_DEVICES=` (empty), so nothing here
reaches the card while somebody renders, and with `H3_CHECK_SWEEP=1`
(`bench/_lib::SWEEP_ENV`), which a check that would otherwise queue a render
reads and refuses on. `PYTHONPATH` is passed through untouched and not added
to: a check that needs it and does not bootstrap shows up red, which is the
point.

It does not replace `bench/smoke_h3.py`, which submits to a live server and is
not a `check_*.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

from _lib import SWEEP_ENV  # noqa: E402

BASELINE = BENCH / "checks_baseline.json"

# reasoned: the slowest check in the 2026-10-05 sweep took well under a
# minute, so ten minutes only ever ends a hang.
TIMEOUT_S = 600

GREEN, NOT_GRADED = 0, 2

# reasoned: these read the whole tree (the indexes, every doc's links, every
# text file's paths, every skill's routes, every check's way of finding
# graphs), so a change anywhere can turn one red.
ALWAYS = ("check_doc_inventory", "check_doc_links", "check_no_owner_paths",
          "check_graph_discovery", "check_skill_routes")
# reasoned: beside the root modules, the files whose change is a change to
# what every node or graph does, or to how every check runs.
NODE_PREFIXES = ("workflows/", "vendor_config/", "bench/_lib/", "vendor/")
NODE_FILES = ("bench/run_checks.py", "bench/checks_baseline.json")


def discover(only: str | None) -> list[Path]:
    """Every `check_*.py` directly under `bench/`, by name."""
    paths = sorted(BENCH.glob("check_*.py"))
    if only:
        paths = [p for p in paths if only in p.stem]
    return paths


def changed_files(since: str | None) -> list[str]:
    """Repo-relative paths changed in the working tree (untracked included), and since `since` when given."""
    def git(*args: str) -> list[str]:
        done = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()[-200:]}")
        return [ln for ln in done.stdout.splitlines() if ln.strip()]
    # porcelain v1: two status letters, a space, the path; a rename shows "old -> new"
    paths = {ln[3:].split(" -> ")[-1].strip('"') for ln in git("status", "--porcelain", "--untracked-files=all")}
    if since:
        paths.update(git("diff", "--name-only", f"{since}..HEAD"))
    return sorted(paths)


def is_node_file(path: str) -> bool:
    """A file whose change asks for the full sweep."""
    if "/" not in path:
        return path.endswith(".py")
    return path.startswith(NODE_PREFIXES) or path in NODE_FILES


def select_for(changed: list[str], checks: list[Path]) -> tuple[list[Path], dict[str, str], str | None]:
    """(the checks to run, {check: why}, the node file that asked for everything or None)."""
    for path in changed:
        if is_node_file(path):
            return checks, {}, path
    why: dict[str, str] = {name: "reads the whole tree" for name in ALWAYS}
    # what a check would have to name to depend on a changed file
    names = {(Path(p).stem if p.endswith(".py") else Path(p).name): p for p in changed}
    for check in checks:
        if f"bench/{check.name}" in changed:
            why[check.stem] = "changed"
            continue
        text = check.read_text(errors="replace")
        for name, path in names.items():
            if name and name in text:
                why.setdefault(check.stem, f"names {path}")
                break
    return [c for c in checks if c.stem in why], why, None


def explain(code: int, output: str) -> str:
    """The one line worth showing beside a result that is not green."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return "(no output)"
    if code == NOT_GRADED:
        for ln in reversed(lines):
            if ln.startswith("nothing graded"):
                return ln
    return lines[-1]


def run_one(path: Path, env: dict[str, str]) -> tuple[int | None, float, str]:
    """(exit code or None on a timeout, seconds, combined output)."""
    start = time.monotonic()
    try:
        done = subprocess.run(
            [sys.executable, str(path)], cwd=str(REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", timeout=TIMEOUT_S, stdin=subprocess.DEVNULL,
        )
        code, output = done.returncode, done.stdout
    except subprocess.TimeoutExpired as exc:
        code = None
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else \
            (exc.stdout or b"").decode(errors="replace")
        output += f"\ntimed out after {TIMEOUT_S} s"
    return code, time.monotonic() - start, output


def load_baseline(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    expected = data.get("expected_red", {})
    if not isinstance(expected, dict) or not all(
            isinstance(v, str) and v.strip() for v in expected.values()):
        raise SystemExit(f"{path.name}: `expected_red` must map a check's name "
                         "to a one-line reason")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="run only checks whose name contains this")
    parser.add_argument("--logs", type=Path,
                        help="keep one log per check and a summary.tsv in this directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--changed", action="store_true",
                        help="run the checks the working tree's changed files can turn red; "
                             "every check when a node file is among them")
    parser.add_argument("--since", metavar="COMMIT",
                        help="with --changed: also count the files changed since this commit")
    args = parser.parse_args()
    if args.since and not args.changed:
        parser.error("--since goes with --changed")

    checks = discover(args.only)
    if not checks:
        print("nothing graded: no check_*.py matched")
        return NOT_GRADED
    targeted = False
    if args.changed:
        changed = changed_files(args.since)
        if not changed:
            print("nothing graded: no file is changed" + (f" since {args.since}" if args.since else ""))
            return NOT_GRADED
        checks, why, node_file = select_for(changed, checks)
        if node_file:
            print(f"FULL sweep: {node_file} is a node file ({len(changed)} file(s) changed)\n")
        else:
            targeted = True
            print(f"TARGETED sweep, {len(checks)} check(s) for {len(changed)} changed file(s); "
                  "not the full sweep a schema, default or loader change asks for")
            for check in checks:
                print(f"  {check.stem}: {why[check.stem]}")
            print()
    expected = load_baseline(args.baseline)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[SWEEP_ENV] = "1"

    keep = args.logs
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.TemporaryDirectory()
    log_dir = keep or Path(scratch.name)

    width = max(len(p.stem) for p in checks)
    print(f"{'exit':>4}  {'sec':>5}  {'check':<{width}}  note")
    rows: list[tuple[str, int | None, float]] = []
    red: dict[str, str] = {}
    counts = {"green": 0, "not graded": 0, "red": 0}
    for path in checks:
        code, seconds, output = run_one(path, env)
        (log_dir / f"{path.stem}.log").write_text(output)
        rows.append((path.stem, code, seconds))
        if code == GREEN:
            counts["green"] += 1
            note = ""
        elif code == NOT_GRADED:
            counts["not graded"] += 1
            note = explain(code, output)
        else:
            counts["red"] += 1
            note = explain(code if code is not None else 1, output)
            red[path.stem] = note
        shown = "t/o" if code is None else str(code)
        print(f"{shown:>4}  {seconds:5.1f}  {path.stem:<{width}}  {note[:160]}",
              flush=True)

    if keep:
        (keep / "summary.tsv").write_text("".join(
            f"{'timeout' if code is None else code}\t{name}\t{seconds:.1f}\n"
            for name, code, seconds in rows))

    print(f"\n{len(checks)} check(s): " + ", ".join(
        f"{n} {label}" for label, n in counts.items()))

    ran = {name for name, _, _ in rows}
    unexpected = sorted(set(red) - set(expected))
    known = sorted(set(red) & set(expected))
    stale = sorted(n for n in expected if n in ran and n not in red)
    unknown = sorted(n for n in expected
                     if not (BENCH / f"{n}.py").is_file())
    scope = "of the targeted set " if targeted else ""

    for name in known:
        print(f"expected red  {name}: {expected[name]}")
    for name in stale:
        print(f"STALE         {name} is in {args.baseline.name} and is not red "
              "in this run; delete the entry")
    for name in unknown:
        print(f"STALE         {name} is in {args.baseline.name} and no such "
              "check exists; delete the entry")

    if unexpected:
        print(f"\nFAIL  {len(unexpected)} check(s) red that "
              f"{args.baseline.name} does not name:")
        for name in unexpected:
            print(f"\n--- {name}  (last lines)")
            tail = (log_dir / f"{name}.log").read_text().splitlines()[-15:]
            print("\n".join(f"    {ln[:300]}" for ln in tail))
        if not keep:
            print("\nre-run with --logs DIR to keep the full logs")
        return 1
    print(f"\nok    no check {scope}is red that the baseline does not name"
          + (f"; logs in {keep}" if keep else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
