#!/usr/bin/env python3
"""Reject machine-owner home paths in repository text artifacts.

Model and media locations are runtime inputs. Persist logical names, hashes,
dataset revisions, and repo-relative paths instead of usernames or home roots.
"""

from __future__ import annotations

import argparse
import re
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
PATTERNS = (POSIX_HOME, MAC_HOME, WINDOWS_HOME)


def text_files(root: Path):
    for path in sorted(root.rglob("*")):
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
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    failures = []
    scanned = 0
    for path, relative in text_files(root):
        scanned += 1
        data = path.read_bytes()
        for line_number, line in enumerate(data.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                failures.append(f"{relative}:{line_number}")

    if failures:
        print("FAIL  machine-owner home path stored in repository text:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"ok    no_owner_paths   {scanned} text artifact(s) scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
