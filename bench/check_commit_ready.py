#!/usr/bin/env python3
"""Pre-commit readiness check for explicit commit pathspecs (`C61` item 2d).

Given `<paths...>`, verifies before `git commit -F <msg> -- <paths...>`:
1. `git check-ignore` on every path: refuses any path that is ignored by git.
2. Every path exists on disk or is tracked in git (catches typos in pathspecs).
3. Whether any new or deleted path (or docstring/heading change) needs `bench/build_index.py`:
   refuses if a new indexed file has not been staged with `git add -- <file>`, if any generated
   index (`INDEX.md`, `bench/INDEX.md`, `bench/results/INDEX.md`, `workflows/INDEX.md`) is stale,
   or if an index modified for a new/deleted path is omitted from `<paths...>`.

No git pre-commit hook is installed (hooks are shared across worktrees and owned by path-privacy).
When run with no arguments (as `bench/run_checks.py` invokes `bench/check_*.py`), runs its self-test
suite in a temporary git repository.

    <comfy venv python> bench/check_commit_ready.py <paths...>
    <comfy venv python> bench/check_commit_ready.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish, refused  # noqa: E402

UNINDEXED_TOP_DIRS: frozenset[str] = frozenset({"changelog.d", "docs", "prompt_bank"})
INDEX_FILES: tuple[str, ...] = ("INDEX.md", "bench/INDEX.md", "bench/results/INDEX.md", "workflows/INDEX.md")


def _normalize_rel(path_str: str, repo: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo.resolve())).replace("\\", "/")
        except ValueError:
            raise ValueError(f"{path_str} is outside repository {repo}") from None
    norm = str(Path(path_str)).replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _git_lines(repo: Path, *args: str) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def indexes_needed_for(rel: str) -> list[str]:
    """Return which generated index file(s) list or count `rel` when `rel` is added or deleted."""
    if rel in INDEX_FILES:
        return []
    parts = Path(rel).parts
    if not parts:
        return []
    top = parts[0]
    if top in UNINDEXED_TOP_DIRS:
        return []
    if len(parts) == 1:
        return ["INDEX.md"]
    if top == "bench":
        if len(parts) >= 2 and parts[1] == "results":
            if rel.endswith(".md"):
                return ["bench/results/INDEX.md"]
            return ["bench/results/INDEX.md"]
        return ["INDEX.md", "bench/INDEX.md"]
    if top == "workflows":
        if rel.endswith(".json"):
            return ["INDEX.md", "workflows/INDEX.md"]
        return ["INDEX.md"]
    return ["INDEX.md"]


def check_commit_ready(paths: list[str], repo: Path = REPO) -> list[str]:
    """Return a list of refusal reasons for `paths` in `repo` (empty list if ready to commit)."""
    if not paths:
        return ["no paths given"]
    problems: list[str] = []
    rels: list[str] = []
    for raw in paths:
        try:
            rels.append(_normalize_rel(raw, repo))
        except ValueError as e:
            problems.append(str(e))
    if problems:
        return problems

    tracked_set = set(_git_lines(repo, "ls-files"))
    head_set = set(_git_lines(repo, "ls-tree", "-r", "--name-only", "HEAD"))
    rel_set = set(rels)

    # 1. git check-ignore on each path (and verify path exists on disk or in git)
    for rel in rels:
        ig = subprocess.run(
            ["git", "check-ignore", "-v", "--", rel],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if ig.returncode == 0:
            detail = ig.stdout.strip().splitlines()[0] if ig.stdout.strip() else rel
            problems.append(f"{rel} is gitignored ({detail})")
            continue
        disk_path = repo / rel
        if not disk_path.exists() and rel not in tracked_set and rel not in head_set:
            problems.append(f"{rel} does not exist on disk and is not tracked in git")

    if problems:
        return problems

    # 2. Check whether any new or deleted path needs build_index
    for rel in rels:
        disk_exists = (repo / rel).exists()
        is_new = disk_exists and (rel not in head_set or rel not in tracked_set)
        is_deleted = (not disk_exists) and (rel in tracked_set or rel in head_set)
        if not is_new and not is_deleted:
            continue
        needed = indexes_needed_for(rel)
        if not needed:
            continue
        if is_new and rel not in tracked_set:
            problems.append(
                f"{rel} is a new path that needs bench/build_index.py: run `git add -- {rel}` and `bench/build_index.py` first"
            )
        missing_idx = [idx for idx in needed if idx not in rel_set]
        if missing_idx:
            kind = "new" if is_new else "deleted"
            problems.append(
                f"{rel} ({kind}) affects {', '.join(needed)}: run `bench/build_index.py` and include {', '.join(missing_idx)} in the commit paths"
            )

    # 3. If bench/build_index.py is present in `repo`, verify that the generated indexes match the effective tracked set
    build_index_py = repo / "bench" / "build_index.py"
    if build_index_py.is_file():
        chk = subprocess.run(
            [sys.executable, str(build_index_py), "--check"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if chk.returncode != 0:
            fail_lines = [ln.strip() for ln in chk.stdout.splitlines() if ln.strip().startswith("FAIL")]
            summary = "; ".join(fail_lines) if fail_lines else chk.stdout.strip()
            problems.append(f"indexes are stale (run `bench/build_index.py`): {summary}")

    # TODO(code-map): Once ready to enforce at commit time on `main` (skipping feature worktrees),
    # check whether `rels` touches any of the 44 masked-lane modules in `bench/masked_code_map.py`
    # and run `bench/masked_code_map.py --check-exact` (or warn on narrative stamp drift via `--status`)
    # so `docs/masked_v2v/CODE_MAP_INDEX.md` (`--write`) and narrative section stamps (`--stamp-narrative`)
    # cannot be omitted when landing masked-lane changes on `main`.

    return problems


def commit_ready_self_check() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp)
        (fake / "bench").mkdir(parents=True, exist_ok=True)
        (fake / "docs").mkdir(parents=True, exist_ok=True)
        (fake / "internal").mkdir(parents=True, exist_ok=True)
        (fake / ".gitignore").write_text("internal/\ndata/\n*.pyc\n")
        (fake / "INDEX.md").write_text("# Index\n")
        (fake / "bench" / "INDEX.md").write_text("# Bench Index\n")
        (fake / "bench" / "existing.py").write_text("X = 1\n")
        (fake / "docs" / "note.md").write_text("# Note\n")

        subprocess.run(["git", "init", "-q"], cwd=str(fake), check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(fake), check=True)
        subprocess.run(["git", "config", "user.name", "tester"], cwd=str(fake), check=True)
        subprocess.run(
            ["git", "add", ".gitignore", "INDEX.md", "bench/INDEX.md", "bench/existing.py", "docs/note.md"],
            cwd=str(fake),
            check=True,
        )
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(fake), check=True)

        # 1. Modifying an existing tracked file passes
        (fake / "bench" / "existing.py").write_text("X = 2\n")
        assert check_commit_ready(["bench/existing.py"], repo=fake) == [], (
            f"expected existing tracked edit to pass, got {check_commit_ready(['bench/existing.py'], repo=fake)}"
        )

        # 2. A new doc under docs/ (indexed by hand-written router, not build_index) passes without INDEX.md
        (fake / "docs" / "new_doc.md").write_text("# New\n")
        subprocess.run(["git", "add", "--", "docs/new_doc.md"], cwd=str(fake), check=True)
        assert check_commit_ready(["docs/new_doc.md"], repo=fake) == [], (
            f"expected new docs/ file to not require build_index, got {check_commit_ready(['docs/new_doc.md'], repo=fake)}"
        )

        # 3. Gitignored path is refused naming .gitignore
        (fake / "internal" / "secret.md").write_text("nope\n")
        errs_ig = check_commit_ready(["internal/secret.md"], repo=fake)
        assert any("is gitignored" in e and ".gitignore" in e for e in errs_ig), (
            f"expected gitignored refusal naming .gitignore, got {errs_ig}"
        )

        # 4. Nonexistent path is refused
        errs_missing = check_commit_ready(["bench/ghost.py"], repo=fake)
        assert any("does not exist on disk" in e for e in errs_missing), (
            f"expected nonexistent path refusal, got {errs_missing}"
        )

        # 5. New bench/*.py file not yet git-added or missing INDEX.md / bench/INDEX.md in commit paths is refused
        (fake / "bench" / "brand_new.py").write_text("Y = 1\n")
        errs_new_unadded = check_commit_ready(["bench/brand_new.py"], repo=fake)
        assert any("git add -- bench/brand_new.py" in e for e in errs_new_unadded) and any("bench/INDEX.md" in e for e in errs_new_unadded), (
            f"expected refusal requiring git add and build_index for new bench script, got {errs_new_unadded}"
        )

        subprocess.run(["git", "add", "--", "bench/brand_new.py"], cwd=str(fake), check=True)
        errs_new_no_idx = check_commit_ready(["bench/brand_new.py"], repo=fake)
        assert any("include INDEX.md, bench/INDEX.md" in e for e in errs_new_no_idx), (
            f"expected refusal when INDEX.md and bench/INDEX.md are omitted from paths, got {errs_new_no_idx}"
        )
        assert check_commit_ready(["bench/brand_new.py", "INDEX.md", "bench/INDEX.md"], repo=fake) == [], (
            "expected staged new bench file + both index paths to pass"
        )

        # 6. Deleted tracked bench/*.py file without INDEX.md / bench/INDEX.md in commit paths is refused
        (fake / "bench" / "existing.py").unlink()
        errs_del = check_commit_ready(["bench/existing.py"], repo=fake)
        assert any("bench/existing.py (deleted) affects INDEX.md, bench/INDEX.md" in e for e in errs_del), (
            f"expected deleted bench file without index paths to be refused, got {errs_del}"
        )
        assert check_commit_ready(["bench/existing.py", "INDEX.md", "bench/INDEX.md"], repo=fake) == [], (
            "expected deleted bench file + both index paths to pass"
        )

        # 7. CLI refusal exits non-zero with REFUSED: and no Python traceback
        out = refused([sys.executable, str(Path(__file__).resolve()), "--repo", str(fake), "internal/secret.md"], "REFUSED:", "is gitignored")
        assert "internal/secret.md" in out

    return "gitignored paths, nonexistent paths, and new/deleted indexed paths without git add or build_index are refused; clean commits pass"


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        case("commit_ready_self_check", commit_ready_self_check)
        return finish()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=str(REPO), help="repository root (defaults to REPO)")
    p.add_argument("paths", nargs="+", help="paths to check before committing")
    a = p.parse_args(args_list)
    repo = Path(a.repo).resolve()
    problems = check_commit_ready(a.paths, repo=repo)
    if problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    print(f"ok    {len(a.paths)} path(s) commit-ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
