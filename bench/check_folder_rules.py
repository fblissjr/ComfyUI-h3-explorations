#!/usr/bin/env python3
"""Every folder's rules file loads, points at its index, and every doc is on an index.

The repo discloses itself in layers: root `AGENTS.md`, a folder's `AGENTS.md`
(its rules, loaded when a session opens a file there), the folder's index (what
is in it), then the page. `docs/method/agents_and_indexes.md` is the standard.
Each layer is only as good as the pointer to the next one, and on 2026-10-19 the
docs reorganization found three ways a pointer goes quietly missing: a folder
with rules and no `CLAUDE.md`, so Claude Code never loads them; a page with no
row on any index, reachable only by a link someone else happened to write; and
a top-level folder that gained rules without root `AGENTS.md` naming it.

Cases, each a hard failure (a missing file or a missing name, never a degree):

- `claude_loads_agents`: beside every tracked `AGENTS.md` is a `CLAUDE.md` that is
  exactly `@AGENTS.md`, the shim Claude Code reads.
- `agents_names_its_index`: an `AGENTS.md` in a folder that has an index
  (`index.md`, `INDEX.md` or `README.md`) names that index.
- `every_doc_on_an_index`: every tracked `.md` under `docs/` is linked from the
  index of its own folder, or, when its folder has none, of the nearest folder
  above it that has one. `AGENTS.md`, `CLAUDE.md` and the index itself are not
  pages.
- `root_lists_the_folders`: the folders root `AGENTS.md` lists as having their own
  `AGENTS.md` are exactly the top-level folders that have one.

It does not check that a link resolves (`check_doc_links.py`) or that a doc is
reachable from the router (`build_wiki_index.py`), and it cannot tell whether an
index row says something true about its page. Needs no ComfyUI and no card.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402

INDEX_NAMES = ("index.md", "INDEX.md", "README.md")
NOT_PAGES = {"AGENTS.md", "CLAUDE.md", *INDEX_NAMES}
LINK = re.compile(r"\]\(([^)\s#]+)")


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (REPO / p).is_file()]


FILES = tracked()
FILESET = set(FILES)


def index_of(folder: str) -> str | None:
    for name in INDEX_NAMES:
        rel = f"{folder}/{name}" if folder else name
        if rel in FILESET:
            return rel
    return None


def claude_loads_agents() -> str:
    agents = [p for p in FILES if PurePosixPath(p).name == "AGENTS.md"]
    bad = []
    for a in agents:
        shim = str(PurePosixPath(a).with_name("CLAUDE.md"))
        if shim not in FILESET or (REPO / shim).read_text(encoding="utf-8").strip() != "@AGENTS.md":
            bad.append(a)
    assert not bad, "no `CLAUDE.md` holding exactly `@AGENTS.md` beside: " + ", ".join(bad)
    return f"{len(agents)} AGENTS.md, each with its shim"


def agents_names_its_index() -> str:
    bad, n = [], 0
    for a in (p for p in FILES if PurePosixPath(p).name == "AGENTS.md"):
        folder = str(PurePosixPath(a).parent).replace(".", "") if PurePosixPath(a).parent != PurePosixPath(".") else ""
        idx = index_of(folder)
        if idx is None or folder == "":
            continue  # the root's index is generated and named in its own table
        n += 1
        if PurePosixPath(idx).name not in (REPO / a).read_text(encoding="utf-8"):
            bad.append(f"{a} (its index is {idx})")
    assert not bad, "an AGENTS.md that does not name its folder's index: " + "; ".join(bad)
    return f"{n} folder(s) with rules and an index, each named"


def linked_from(index: str) -> set[str]:
    base = PurePosixPath(index).parent
    text = (REPO / index).read_text(encoding="utf-8")
    out = set()
    for target in LINK.findall(text):
        if "://" in target:
            continue
        out.add(str(PurePosixPath(*(base / target).parts)).replace("/./", "/"))
    # normalise `..` segments
    return {str(Path(REPO / p).resolve().relative_to(REPO)) if (REPO / p).exists() else p for p in out}


def every_doc_on_an_index() -> str:
    pages = [p for p in FILES if p.startswith("docs/") and p.endswith(".md") and PurePosixPath(p).name not in NOT_PAGES]
    missing = []
    for page in pages:
        folder = PurePosixPath(page).parent
        while True:
            idx = index_of(str(folder))
            if idx or str(folder) == "docs":
                break
            folder = folder.parent
        if not idx or page not in linked_from(idx):
            missing.append(f"{page} (index: {idx or 'none'})")
    assert not missing, f"{len(missing)} doc(s) on no index: " + "; ".join(missing)
    return f"{len(pages)} doc(s), each linked from its folder's index"


def root_lists_the_folders() -> str:
    root = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Folders with their own `AGENTS\.md`\*\*:(.*?)Read the folder's", root, re.S)
    assert m, "root AGENTS.md has no '**Folders with their own `AGENTS.md`**:' line to read"
    listed = set(re.findall(r"`([\w.-]+)/`", m.group(1)))
    have = {PurePosixPath(p).parts[0] for p in FILES if PurePosixPath(p).name == "AGENTS.md" and len(PurePosixPath(p).parts) == 2}
    extra, missing = sorted(listed - have), sorted(have - listed)
    assert not extra and not missing, (f"root AGENTS.md lists without an AGENTS.md: {extra}; "
                                       f"has one but is not listed: {missing}")
    return f"{len(have)} top-level folder(s) with rules, all listed"


if __name__ == "__main__":
    case("claude_loads_agents", claude_loads_agents)
    case("agents_names_its_index", agents_names_its_index)
    case("every_doc_on_an_index", every_doc_on_an_index)
    case("root_lists_the_folders", root_lists_the_folders)
    sys.exit(finish())
