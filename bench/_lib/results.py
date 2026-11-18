"""Find run data under `data/results/` after it has been archived.

Older run data moves from `data/results/` into dated archive folders named
`<first day>_to_<last day>` (`2026-08-18_to_2026-10-03`, `2026-10-04_to_2026-11-03`),
keeping each entry's path below the folder. Tools and docs still name the old
path. These functions look where the move puts things: the path as written,
then the same path inside each archive folder. Nothing is matched by a bare
file name, so a file in an unrelated study folder is never picked up.

Reads only. A tool writes new data to `data/results/` itself, as before, and
the next archive move breaks no reader.

    from _lib.results import result_path, result_glob
    rows = json.loads(result_path("2026-09-26_followup.jsonl").read_text())

Every function takes `root`, the `data/results/` folder, so a check or a
self-test can point it at a temporary tree. Checked by
`bench/check_data_results_lookup.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"
ARCHIVE_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}")


def archive_folders(root: Path = RESULTS) -> list[Path]:
    """The dated archive folders directly under `root`, oldest first."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and ARCHIVE_NAME.fullmatch(p.name))


def find_results(name: str | Path, root: Path = RESULTS) -> list[Path]:
    """Every place `name`, a path below `root`, exists: as written, else in each archive folder."""
    here = root / name
    if here.exists():
        return [here]
    return [a / name for a in archive_folders(root) if (a / name).exists()]


def result_path(name: str | Path, root: Path = RESULTS) -> Path:
    """`root / name` when it is there, else the one archive folder's copy.

    When nothing holds it, `root / name` comes back unchanged, so the caller's
    own missing-file handling runs as it did before the move. Raises
    ValueError when more than one archive folder holds it.
    """
    found = find_results(name, root)
    if len(found) > 1:
        raise ValueError(f"{name} is in several archive folders: "
                         + ", ".join(p.relative_to(root).as_posix() for p in found))
    return found[0] if found else root / name


def result_glob(pattern: str, root: Path = RESULTS) -> list[Path]:
    """`root.glob(pattern)` and the same glob in each archive folder, sorted by path below its folder.

    One path per relative path: the top level wins, then the oldest archive
    folder. The archive folders themselves are never matches.
    """
    seen: dict[str, Path] = {}
    for base in [root, *archive_folders(root)]:
        if not base.is_dir():
            continue
        for p in base.glob(pattern):
            if base == root and p.parent == root and ARCHIVE_NAME.fullmatch(p.name):
                continue
            seen.setdefault(p.relative_to(base).as_posix(), p)
    return [seen[k] for k in sorted(seen)]
