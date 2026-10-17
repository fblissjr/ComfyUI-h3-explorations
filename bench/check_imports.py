#!/usr/bin/env python3
"""Static AST check over tracked root `*.py` and `bench/*.py` (`C61` item 2a).

Walks the AST of every tracked root `*.py` and `bench/**/*.py` file (excluding `archive/`),
skipping `import` and `from ... import ...` statements nested inside `try`/`except` blocks,
and fails if any import names a folder or module of this repo (or an unknown top-level package)
that does not exist on disk.

    <comfy venv python> bench/check_imports.py
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402

#: External ComfyUI runtime and custom-node modules (plus standalone/ when omitted by break_checks.py)
COMFY_MODULES: frozenset[str] = frozenset({
    "comfy",
    "comfy_aimdo",
    "comfy_api",
    "comfy_execution",
    "comfy_extras",
    "comfy_kitchen",
    "folder_paths",
    "h3_mutant_distill",
    "h3x",
    "latent_preview",
    "node_helpers",
    "nodes",
    "sageattention",
    "server",
    "videohelpersuite",
})

STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__", "builtins"}


def _is_module_or_pkg(parts: list[str], roots: list[Path]) -> bool:
    return any(
        r.joinpath(*parts).with_suffix(".py").is_file()
        or (r.joinpath(*parts).is_dir() and (r.joinpath(*parts) / "__init__.py").is_file())
        for r in roots
    )


def _exists_in_roots(parts: list[str], roots: list[Path]) -> bool:
    if _is_module_or_pkg(parts, roots):
        return True
    if len(parts) == 1 and any(r.joinpath(*parts).is_dir() for r in roots):
        return True
    return False


def _is_namespace_dir_only(parts: list[str], roots: list[Path]) -> bool:
    return not _is_module_or_pkg(parts, roots) and any(r.joinpath(*parts).is_dir() for r in roots)


def _external_known(top: str) -> bool:
    if top in STDLIB_MODULES or top in COMFY_MODULES:
        return True
    try:
        return importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        return False


def tracked_python_files(repo: Path = REPO) -> list[Path]:
    """Return tracked root `*.py` and `bench/**/*.py` files, excluding `archive/`."""
    if (repo / ".git").exists():
        out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=str(repo), text=True).splitlines()
        rels = sorted(out)
    else:
        rels = sorted(
            str(p.relative_to(repo))
            for p in [*repo.glob("*.py"), *(repo / "bench").rglob("*.py")]
            if p.is_file() and "__pycache__" not in p.parts
        )
    files: list[Path] = []
    for rel in rels:
        if rel.startswith("archive/"):
            continue
        p = Path(rel)
        if p.parent == Path(".") or rel.startswith("bench/"):
            full = repo / p
            if full.is_file():
                files.append(full)
    return files


def check_file_imports(path: Path, repo: Path = REPO) -> list[str]:
    """Parse `path`'s AST (skipping `try`/`except` bodies) and return a list of missing imports."""
    roots = [repo, repo / "bench", repo / "workflows", repo / "standalone"]
    rel = path.relative_to(repo) if path.is_relative_to(repo) else path
    tree = ast.parse(path.read_text(), filename=str(rel))
    bad: list[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Try):
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "_h3pack":
                    if len(parts) > 1 and not _exists_in_roots(parts[1:], [repo]):
                        bad.append(f"{rel}:{node.lineno}: import {alias.name} (missing in repo)")
                    continue
                top = parts[0]
                if _exists_in_roots([top], roots):
                    if not _exists_in_roots(parts, roots):
                        bad.append(f"{rel}:{node.lineno}: import {alias.name} (missing in repo)")
                elif not _external_known(top):
                    bad.append(f"{rel}:{node.lineno}: import {alias.name} (missing module or folder {top!r})")
        elif isinstance(node, ast.ImportFrom):
            lvl = node.level or 0
            mod = node.module or ""
            dots = "." * lvl
            if lvl > 0:
                base = path.parent
                for _ in range(lvl - 1):
                    base = base.parent
                target_parts = mod.split(".") if mod else []
                if target_parts and not _exists_in_roots(target_parts, [base]):
                    bad.append(f"{rel}:{node.lineno}: from {dots}{mod} import ... (missing relative module)")
                elif not target_parts:
                    for alias in node.names:
                        if not _exists_in_roots([alias.name], [base]) and not (base / "__init__.py").is_file():
                            bad.append(f"{rel}:{node.lineno}: from {dots} import {alias.name} (missing relative module)")
            else:
                parts = mod.split(".") if mod else []
                if parts and parts[0] == "_h3pack":
                    sub = parts[1:]
                    if sub and not _exists_in_roots(sub, [repo]):
                        bad.append(f"{rel}:{node.lineno}: from {mod} import ... (missing in repo)")
                else:
                    top = parts[0] if parts else ""
                    if _exists_in_roots([top], roots):
                        if not _exists_in_roots(parts, roots):
                            bad.append(f"{rel}:{node.lineno}: from {mod} import ... (missing in repo)")
                        elif _is_namespace_dir_only(parts, roots):
                            for alias in node.names:
                                if not _exists_in_roots([*parts, alias.name], roots):
                                    bad.append(f"{rel}:{node.lineno}: from {mod} import {alias.name} (missing in {mod}/)")
                    elif not _external_known(top):
                        bad.append(f"{rel}:{node.lineno}: from {mod} import ... (missing module or folder {top!r})")
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return bad


def check_repo_imports(repo: Path = REPO, files: list[Path] | None = None) -> list[str]:
    """Check all tracked root `*.py` and `bench/*.py` files in `repo`."""
    target_files = tracked_python_files(repo) if files is None else files
    bad: list[str] = []
    for f in target_files:
        bad.extend(check_file_imports(f, repo=repo))
    return bad


def tracked_root_and_bench_imports_exist() -> str:
    files = tracked_python_files(REPO)
    assert len(files) >= 400, f"expected >= 400 tracked root and bench .py files, got {len(files)}"
    assert not any(f.relative_to(REPO).parts[0] == "archive" for f in files), "archive/ must not be scanned"
    bad = check_repo_imports(REPO, files=files)
    assert not bad, "missing imports in tracked root *.py or bench/*.py:\n  " + "\n  ".join(bad)
    return f"{len(files)} tracked root *.py and bench/*.py files scanned; 0 missing imports"


def catches_missing_repo_folder_and_module_imports() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp)
        for d in (fake / "bench" / "_lib", fake / "workflows", fake / "standalone", fake / "archive"):
            d.mkdir(parents=True, exist_ok=True)
        (fake / "bench" / "_lib" / "__init__.py").write_text("REPO = None\n")
        (fake / "existing_root.py").write_text("X = 1\n")
        (fake / "bench" / "existing_bench.py").write_text("Y = 2\n")
        (fake / "workflows" / "h3_config.py").write_text("Z = 3\n")

        subprocess.run(["git", "init", "-q"], cwd=str(fake), check=True)

        # 1. Valid imports (including an unresolvable import inside try/except and a broken import inside archive/)
        good_root = fake / "good_node.py"
        good_root.write_text(
            "import json\n"
            "import existing_root\n"
            "from . import existing_root as er\n"
            "try:\n"
            "    import meta_sam3\n"
            "    from deleted_pkg import something\n"
            "except ImportError:\n"
            "    pass\n"
        )
        good_bench = fake / "bench" / "good_tool.py"
        good_bench.write_text(
            "import existing_bench\n"
            "import existing_root\n"
            "from _lib import REPO\n"
            "from workflows import h3_config\n"
            "import _h3pack.existing_root as er\n"
        )
        archived = fake / "archive" / "old_sam3.py"
        archived.write_text("import meta_sam3\n")

        subprocess.run(
            ["git", "add", "existing_root.py", "good_node.py", "bench/existing_bench.py", "bench/good_tool.py",
             "bench/_lib/__init__.py", "workflows/h3_config.py", "archive/old_sam3.py"],
            cwd=str(fake),
            check=True,
        )
        tracked = tracked_python_files(fake)
        assert archived not in tracked, "archive/old_sam3.py must be excluded from tracked_python_files"
        assert check_repo_imports(fake, files=tracked) == [], (
            f"expected 0 errors on valid + try/except + archive files, got {check_repo_imports(fake, files=tracked)}"
        )

        # 2. Unprotected imports of missing repo folders/modules must fail
        bad_file = fake / "bench" / "bad_tool.py"
        bad_file.write_text(
            "import meta_sam3\n"
            "from meta_sam3.model import sam\n"
            "import workflows.no_such_mod\n"
            "from workflows import no_such_helper\n"
            "from .no_such_sibling import run\n"
            "import _h3pack.no_such_node\n"
        )
        errs = check_file_imports(bad_file, repo=fake)
        assert len(errs) == 6, f"expected all 6 broken imports to be caught, got {len(errs)}: {errs}"
        for needle in (
            "import meta_sam3",
            "from meta_sam3.model import",
            "import workflows.no_such_mod",
            "from workflows import no_such_helper",
            "from .no_such_sibling import",
            "import _h3pack.no_such_node",
        ):
            assert any(needle in e for e in errs), f"expected {needle!r} in {errs}"

    return "catches missing repo folders/modules, relative imports, namespace imports, and _h3pack targets; skips try/except and archive/"


if __name__ == "__main__":
    case("tracked_root_and_bench_imports_exist", tracked_root_and_bench_imports_exist)
    case("catches_missing_repo_folder_and_module_imports", catches_missing_repo_folder_and_module_imports)
    sys.exit(finish())
