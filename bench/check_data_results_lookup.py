#!/usr/bin/env python3
"""That `bench/_lib/results.py` finds archived run data where the archive move put it, and nowhere else.

Older run data moves from `data/results/` into dated archive folders
(`<first day>_to_<last day>`), keeping each entry's path below the folder. The
tools that read old inputs and `bench/resolve_data_citations.py` both look
through this lookup, so a wrong answer here is a wrong input there: a tool
that reads a same-named file from an unrelated folder, or a citation reported
as found when it is not.

Every case runs on a temporary `data/results/` built here, so it needs no run
data and passes the same on any machine.

Claims, i.e. what breaks if a case is deleted:

- `here_wins`           -- a path present as written comes back as written,
                           even when an archive folder holds the same path.
- `archive_found`       -- a path absent at the top comes back from the one
                           archive folder that holds it, its subpath kept.
- `no_bare_name_match`  -- a file of the same name in a study folder (not an
                           archive folder) is not found by name; the lookup
                           returns the path as written, so the caller's own
                           missing-file handling runs.
- `several_refused`     -- two archive folders holding the same path is an
                           error naming both, not a silent pick.
- `glob_spans_archives` -- a glob sees the top level and every archive folder,
                           one match per path (the top level wins), sorted,
                           never an archive folder itself and never inside a
                           study folder.
- `resolver_states`     -- `resolve_data_citations.resolve` reports here,
                           moved, several and missing from the same lookup.
- `citation_grammar`    -- the resolver's parser skips templates, globs,
                           placeholders and `--out` paths, and drops a
                           sentence's closing period.

    <python> bench/check_data_results_lookup.py

No ComfyUI, no card, no run data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
from _lib.results import result_glob, result_path  # noqa: E402
import resolve_data_citations as rdc  # noqa: E402

_tmp = tempfile.TemporaryDirectory(prefix="h3_results_lookup_")
ROOT = Path(_tmp.name) / "results"
A1 = ROOT / "2026-01-01_to_2026-01-31"
A2 = ROOT / "2026-02-01_to_2026-02-28"
STUDY = ROOT / "2026-03-01_some_study"
for path in [ROOT / "top.json", ROOT / "both.json", A1 / "both.json", A1 / "old.json",
             A1 / "2026-01-05_study" / "inner.json", A1 / "twice.json", A2 / "twice.json",
             A2 / "later.json", STUDY / "stray.json"]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")


def here_wins():
    assert result_path("top.json", ROOT) == ROOT / "top.json"
    got = result_path("both.json", ROOT)
    assert got == ROOT / "both.json", f"both.json came back as {got}"


def archive_found():
    assert result_path("old.json", ROOT) == A1 / "old.json"
    assert result_path("later.json", ROOT) == A2 / "later.json"
    got = result_path("2026-01-05_study/inner.json", ROOT)
    assert got == A1 / "2026-01-05_study" / "inner.json", f"subpath lost: {got}"


def no_bare_name_match():
    got = result_path("stray.json", ROOT)
    assert got == ROOT / "stray.json", f"found by bare name in a study folder: {got}"
    got = result_path("inner.json", ROOT)
    assert got == ROOT / "inner.json", f"found by bare name below an archive folder: {got}"


def several_refused():
    try:
        got = result_path("twice.json", ROOT)
    except ValueError as exc:
        assert A1.name in str(exc) and A2.name in str(exc), f"error does not name both: {exc}"
        return
    raise AssertionError(f"two archive copies gave {got} instead of an error")


def glob_spans_archives():
    got = [p.relative_to(ROOT).as_posix() for p in result_glob("*.json", ROOT)]
    want = ["top.json", "both.json", f"{A1.name}/old.json", f"{A1.name}/twice.json",
            f"{A2.name}/later.json"]
    want_sorted = sorted(want, key=lambda r: r.split("/")[-1])
    assert got == want_sorted, f"got {got}, want {want_sorted}"
    names = [p.name for p in result_glob("2026-*", ROOT)]
    assert A1.name not in names and A2.name not in names, f"an archive folder matched a glob: {names}"


def resolver_states():
    states = {c: rdc.resolve(c, ROOT)[0] for c in ("top.json", "old.json", "twice.json", "stray.json")}
    want = {"top.json": "here", "old.json": "moved", "twice.json": "several", "stray.json": "missing"}
    assert states == want, f"got {states}"


def citation_grammar():
    text = ("see `data/results/a_2026.json`. and data/results/<date>_x.json, data/results/2026-09-2{6,7}_y, "
            "data/results/*.json, data/results/DATE_name.json, data/results/...json, "
            "--out data/results/new_out.json and (data/results/folder/b.jsonl)")
    got = [c for _, _, c in rdc.citations(text)]
    assert got == ["a_2026.json", "folder/b.jsonl"], f"got {got}"


case("here_wins", here_wins)
case("archive_found", archive_found)
case("no_bare_name_match", no_bare_name_match)
case("several_refused", several_refused)
case("glob_spans_archives", glob_spans_archives)
case("resolver_states", resolver_states)
case("citation_grammar", citation_grammar)
sys.exit(finish())
