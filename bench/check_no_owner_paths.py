#!/usr/bin/env python3
"""Reject machine-owner home paths in repository text artifacts.

Model and media locations are runtime inputs. Persist logical names, hashes,
dataset revisions, and repo-relative paths instead of usernames or home roots.

## What it scans, decided 2026-10-05

**The files git would take**: tracked files, plus untracked files no ignore
rule covers, which are the ones the next `git add` can pick up. That is the
repository; a path in it leaves this machine.

**Not the gitignored trees.** `internal/` and `data/` hold session notes, raw
captures and run scripts that are never distributed, and a raw capture's
first line records where it ran. Until this date the walk covered them, and
the check was red on those lines alone, with no tracked file among them, so
it could not say whether the repository was clean. Scrubbing them would mean
rewriting other sessions' dated notes for no reader.

Two ways to look wider. `--include-ignored` walks everything under the root,
as the check did before. A root given on the command line that is not this
repo is walked whole, ignore rules or not, because a staging tree being
prepared for upload is published whatever this repo's `.gitignore` says:

    python bench/check_no_owner_paths.py <staging tree>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "coderef"}
TEXT_SUFFIXES = {
    "", ".css", ".html", ".ini", ".js", ".json", ".jsonl", ".md",
    ".patch", ".py", ".rejected", ".sh", ".toml", ".txt", ".yaml",
    ".yml",
}

# Construct the literals so this checker does not contain an example that
# satisfies its own rule.
USERNAME = rb"[A-Za-z0-9._-]+"
POSIX_HOME = re.compile(b"/" + b"home" + rb"/" + USERNAME + rb"/")
MAC_HOME = re.compile(b"/" + b"Users" + rb"/" + USERNAME + rb"/")
WINDOWS_HOME = re.compile(
    rb"[A-Za-z]:\\" + b"Users" + rb"\\" + USERNAME + rb"\\",
    re.IGNORECASE,
)
MOUNT_SHARE = re.compile(b"/" + b"mnt" + rb"/" + b"hub" + rb"(?:/|\b)")
PATTERNS = (POSIX_HOME, MAC_HOME, WINDOWS_HOME, MOUNT_SHARE)


def git_files(root: Path) -> list[Path] | None:
    """What git would take from `root`: tracked, plus untracked and not ignored.

    None when `root` is not the top of a git work tree or git is absent, and
    the caller walks the directory instead.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            return None
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, check=False)
    except OSError:
        return None
    if listed.returncode != 0:
        return None
    return sorted(root / name.decode("utf-8", "surrogateescape")
                  for name in listed.stdout.split(b"\0") if name)


def text_files(root: Path, candidates: list[Path] | None = None):
    """Text artifacts under `root`: the given files, or the whole tree."""
    for path in (candidates if candidates is not None else sorted(root.rglob("*"))):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path, relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=ROOT,
                        help="tree to scan (default: this repo, the files git would take)")
    parser.add_argument("--include-ignored", action="store_true",
                        help="walk everything under the root, gitignored files too")
    args = parser.parse_args()
    root = args.root.resolve()
    candidates = None if args.include_ignored else git_files(root)
    scope = ("every file under the root" if candidates is None
             else "tracked and untracked-unignored files")
    failures = []
    scanned = 0
    for path, relative in text_files(root, candidates):
        scanned += 1
        data = path.read_bytes()
        for line_number, line in enumerate(data.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                failures.append(f"{relative}:{line_number}")

    if failures:
        print(f"FAIL  machine-owner home path stored in repository text ({scope}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if not scanned:
        # An empty scan must not read as a clean one.
        print("nothing graded: no text artifact found to scan")
        return 2
    print(f"ok    no_owner_paths   {scanned} text artifact(s) scanned ({scope})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
