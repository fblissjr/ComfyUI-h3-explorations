"""Code-version stamps and upstream-mtime checks for derived lane files.

Every derived `.json` or `.npz` artifact in the masked video-to-video lane
records the script path, short git HEAD commit, dirty state, file SHA-256,
and UTC timestamp of the code that wrote it (`docs/wiki/handover.md` item 2c).
When a downstream tool inspects or refuses to overwrite an existing derived
artifact, `check_upstream_mtime` reports whether any upstream input on disk is
newer than that downstream artifact.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent

#: 1.0 s mtime tolerance when comparing upstream inputs against a derived file.
#: reasoned: files written back-to-back in the same pipeline step or across
#: coarse filesystem timestamp quantization can land within <1 s of each other;
#: flagging only when upstream mtime exceeds downstream mtime + 1.0 s avoids
#: false stale warnings on same-step outputs while catching any genuine re-run.
MTIME_TOLERANCE_S = 1.0

#: Required keys in a JSON `writer_stamp` dictionary.
#: reasoned: names the script, git commit, dirty state, file content digest,
#: and UTC write timestamp needed to tell whether a derived file came from the
#: current code or an older edit.
STAMP_KEYS = ("script", "git_commit", "git_dirty", "sha256", "stamped_at")

#: Required keys added to a `.npz` archive by `npz_writer_fields`.
#: reasoned: self-identifies the writer script, git HEAD commit, and file
#: SHA-256 inside the archive itself even when no sidecar JSON sits beside it.
NPZ_STAMP_KEYS = ("writer_script", "writer_commit", "writer_sha256")

_FILE_CACHE: dict[tuple[str, str, int], tuple[str, bool, str]] = {}


def _resolve_in_repo(script_path: str | Path) -> tuple[Path, Path, str]:
    p = Path(script_path).resolve()
    try:
        rel = p.relative_to(REPO).as_posix()
        return REPO, p, rel
    except ValueError:
        return REPO, p, p.name


def _git_and_sha256(script_path: str | Path) -> tuple[str, str, bool, str]:
    repo, p, rel = _resolve_in_repo(script_path)
    mtime_ns = p.stat().st_mtime_ns if p.exists() else -1
    cache_key = (str(repo), rel, mtime_ns)
    cached = _FILE_CACHE.get(cache_key)
    if cached is not None:
        head, dirty, sha = cached
        return rel, head, dirty, sha

    sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    head = "unknown"
    dirty = False
    try:
        r_head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r_head.returncode == 0 and r_head.stdout.strip():
            head = r_head.stdout.strip()
        r_status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", rel],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r_status.returncode == 0:
            dirty = bool(r_status.stdout.strip())
    except Exception:  # noqa: BLE001
        pass

    _FILE_CACHE[cache_key] = (head, dirty, sha)
    return rel, head, dirty, sha


def writer_stamp(
    script_path: str | Path,
    extra_scripts: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable code-version stamp for `script_path`.

    When `extra_scripts` is non-empty, each helper script's `git_commit`,
    `git_dirty`, and `sha256` are recorded under `"extra_scripts"` and any
    dirty helper also sets top-level `"git_dirty"` to True.
    """
    rel, head, dirty, sha = _git_and_sha256(script_path)
    stamp: dict[str, Any] = {
        "script": rel,
        "git_commit": head,
        "git_dirty": dirty,
        "sha256": sha,
        "stamped_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if extra_scripts:
        extras: dict[str, dict[str, Any]] = {}
        for extra in extra_scripts:
            e_rel, e_head, e_dirty, e_sha = _git_and_sha256(REPO / extra if not Path(extra).is_absolute() else extra)
            extras[e_rel] = {
                "git_commit": e_head,
                "git_dirty": e_dirty,
                "sha256": e_sha,
            }
            if e_dirty:
                stamp["git_dirty"] = True
        stamp["extra_scripts"] = extras
    return stamp


def npz_writer_fields(script_path: str | Path) -> dict[str, np.ndarray]:
    """Return `writer_script`, `writer_commit`, and `writer_sha256` arrays for `np.savez_compressed`."""
    import numpy as np

    rel, head, _dirty, sha = _git_and_sha256(script_path)
    return {
        "writer_script": np.array(rel),
        "writer_commit": np.array(head),
        "writer_sha256": np.array(sha),
    }


def check_upstream_mtime(
    downstream_path: str | Path,
    upstream_paths: dict[str, str | Path | None],
) -> list[str]:
    """Return human-readable warnings for any upstream file newer than `downstream_path` + 1.0 s."""
    down = Path(downstream_path)
    if not down.exists():
        return []
    down_mtime = down.stat().st_mtime
    stale: list[str] = []
    for label, up in upstream_paths.items():
        if up is None or str(up) == "":
            continue
        up_path = Path(up)
        if not up_path.exists():
            continue
        up_mtime = up_path.stat().st_mtime
        if up_mtime > down_mtime + MTIME_TOLERANCE_S:
            delta = up_mtime - down_mtime
            stale.append(
                f"upstream {label} ({up_path.name}) is {delta:.1f}s newer than {down.name}"
            )
    return stale


def verify_writer_stamp(
    stamp: object,
    *,
    expected_script: str | None = None,
) -> list[str]:
    """Validate a JSON `writer_stamp` dict and return a list of problems (empty when valid)."""
    if not isinstance(stamp, dict):
        return [f"writer_stamp is {type(stamp).__name__}, expected dict"]
    problems: list[str] = []
    for k in STAMP_KEYS:
        if k not in stamp:
            problems.append(f"missing key {k!r}")
    if problems:
        return problems
    if not isinstance(stamp["script"], str) or not stamp["script"]:
        problems.append("script must be a non-empty string")
    elif expected_script is not None and stamp["script"] != expected_script:
        problems.append(f"script is {stamp['script']!r}, expected {expected_script!r}")
    if not isinstance(stamp["git_commit"], str) or not stamp["git_commit"]:
        problems.append("git_commit must be a non-empty string")
    if not isinstance(stamp["git_dirty"], bool):
        problems.append("git_dirty must be a bool")
    if not isinstance(stamp["sha256"], str) or len(stamp["sha256"]) != 64:
        problems.append("sha256 must be a 64-char hex string")
    if not isinstance(stamp["stamped_at"], str) or "T" not in stamp["stamped_at"]:
        problems.append("stamped_at must be an ISO-8601 UTC timestamp")
    return problems
