#!/usr/bin/env python3
"""Which version of a producer wrote a result. Pure stdlib, either virtualenv.

`canonical/2026-08-24_gate2_readiness.md` accepts a Gate 2A record only from
"one committed harness version". A result file that does not say which
version wrote it cannot satisfy that, and the first Gate 2A tables were demoted
partly because three harness versions had produced them and nothing in the
files told them apart. So the producer's own commit, file hash and dirty state
travel inside every report.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def producer_provenance(script: str | Path) -> dict:
    """Commit, file SHA-256 and whether the file is dirty against that commit."""
    path = Path(script).resolve()
    record: dict = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repository_commit": None,
        "dirty_against_commit": None,
    }
    try:
        record["repository_commit"] = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path.parent), "status", "--porcelain", "--", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        record["dirty_against_commit"] = bool(status)
    except Exception as exc:  # no git, or not a checkout: say so, do not guess
        record["git_error"] = f"{type(exc).__name__}: {exc}"
    return record
