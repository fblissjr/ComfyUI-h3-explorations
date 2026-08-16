#!/usr/bin/env python
"""Fail when a doc points at a file or line that is not there any more.

Two kinds of pointer rot, both of which happened here:

**Markdown links between docs.** On 2026-08-16 `docs/sol_engine_reference.md`
was renamed to `docs/sol_upstream.md`. `CLAUDE.md:41` went stale instantly, and
it went stale inside an hour of a wholesale CLAUDE.md rewrite whose entire
purpose was removing stale references. Nothing said so. A peer session noticed
by eye.

**Code citations of the form `path:line`.** There were 19 of these across
`docs/` when this check was written. All 19 were accurate that day, which is the
only reason this check is preventive rather than a cleanup: `vendor/` had not
changed since the commit `docs/morton.md` pins its line numbers at. The next
edit to any cited file silently invalidates them, and the failure is invisible
-- a reader follows `sol_attn_route.cu:18` to whatever line 18 says today and
believes it.

Claims, i.e. what breaks if a case is deleted:

- `parses_the_corpus`  -- a run that finds no files, or files but no citations,
                          FAILS. This check must not go green by looking at
                          nothing. CLAUDE.md's rule: a check whose input already
                          satisfies the expected outcome cannot fail.
- `doc_links_resolve`  -- every relative markdown link to a repo file exists.
                          This is the rename case above.
- `citations_resolve`  -- every `path:line` citation resolves to a real file.
- `citations_in_range` -- the cited line or range is inside that file.
- `no_bare_basenames`  -- a citation with no directory part that does not sit
                          at the repo root is refused. `sol_layout.cuh:81` is
                          not a path; it is a hint that happens to be unique
                          today. Cite `coderef/<repo>/path/to/sol_layout.cuh`.
- `declared_absent_still_absent` -- WARNS when a declared-absent path comes
                          back. Same reasoning as `check_retraction_consumers`'s
                          `stale_allowlist`: it means someone restored
                          something, and failing would punish the restore.

## Where the data lives

`docs/checks.md`, in a fenced ```doc-link-absent block -- with the index of
checks it belongs to, so the enumeration cannot drift from the check that reads
it. Same pattern as the `retraction-consumers` block in `docs/evidence.md`, and
the same reason: nothing gets a second copy.

Entries are paths that are cited on purpose and cannot resolve, the deleted
Triton pack being the whole of it today.

## What it does not defend

**It cannot tell whether a cited line still says what the citation claims.**
`morton_perm` moving from line 150 to line 90 while something else lands on 150
passes every case here. Only a human reading both settles that, which is why
`docs/morton.md` pins a commit as well.

**It does not scan `CHANGELOG.md`, deliberately.** A changelog records what was
true when it was written. `CHANGELOG.md:455` correctly names
`docs/sol_engine_reference.md` as the file 0.18.0 added, and a checker that
demanded that be rewritten would be falsifying history, not maintaining it.

**It resolves against the filesystem, not against git.** `workflows/image/` was
untracked but present when this was written, and a `git ls-files` corpus would
have reported every citation into it as missing -- a red on correct, brand-new
work.

**Paths under `coderef/` are advisory.** Those are gitignored symlinks to
sister checkouts. A machine that has not cloned diffusers is not broken, so an
unresolvable `coderef/` citation warns and does not fail.

Takes paths on the command line; with none, it discovers the corpus by walking
the repo rather than globbing a directory somebody remembered. A link checker
that can only see the directories its author anticipated has the same hole that
let `workflows/image/` go ungoverned by two prompt checks until 2026-08-16.

Needs no ComfyUI, no CUDA, no model. Runs in well under a second.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
LEDGER = REPO / "docs" / "checks.md"

SKIP_DIRS = {".git", "internal", "__pycache__", "coderef", "vendor", "archive"}

# A changelog is a historical record; see the docstring.
SKIP_FILES = {"CHANGELOG.md"}

# `path.ext:12` or `path.ext:12-34`, inside backticks.
CITATION = re.compile(
    r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,5}):(\d+)(?:-(\d+))?`"
)
# [text](target) where target is a relative path, not a URL or an anchor.
DOC_LINK = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#)([^)#]+)(?:#[^)]*)?\)")


def iter_corpus(argv):
    """The files to scan: CLI paths, else every .md the repo actually has."""
    if argv:
        for arg in argv:
            path = Path(arg)
            if not path.is_absolute():
                path = REPO / path
            yield path.resolve()
        return
    for path in sorted(REPO.rglob("*.md")):
        rel = path.relative_to(REPO)
        if SKIP_DIRS & set(rel.parts) or rel.name in SKIP_FILES:
            continue
        yield path


def parse_ledger(text):
    """Pull the fenced doc-link-absent block into {path: why}."""
    m = re.search(r"```doc-link-absent\n(.*?)```", text, re.S)
    if not m:
        raise SystemExit(
            "FAIL parses_the_corpus: no ```doc-link-absent block in "
            f"{LEDGER.relative_to(REPO)}. An absent ledger is not an empty "
            "one -- this check cannot know what is deliberately unresolvable."
        )
    absent, cur = {}, None
    for line in m.group(1).splitlines():
        if line.startswith("PATH:"):
            cur = line.split(":", 1)[1].strip()
            absent[cur] = ""
        elif line.startswith("WHY:"):
            if cur is None:
                raise SystemExit("FAIL parses_the_corpus: WHY before PATH")
            absent[cur] = line.split(":", 1)[1].strip()
    for path, why in absent.items():
        if not why:
            raise SystemExit(
                f"FAIL parses_the_corpus: {path!r} is declared absent with no "
                "WHY. An unexplained exemption is how a real break hides."
            )
    return absent


def resolve(cited):
    """Every existing file a citation could mean, most-specific root first.

    `COMFY.parent` is a root so that a citation into ComfyUI's own tree can be
    written `ComfyUI/nodes.py` and mean it. That is not decoration: `nodes.py`
    exists in BOTH roots, and CLAUDE.md's `nodes.py:2245-2250` -- in the
    section warning that `import nodes` resolves to ours -- silently resolved
    to our 194-line file instead of ComfyUI's 2595-line one. Found by
    `ambiguous_roots` on this check's first run.
    """
    hits = []
    for root in (REPO, COMFY, COMFY.parent):
        candidate = root / cited
        if candidate.is_file() and candidate not in hits:
            hits.append(candidate)
    # `coderef/...` needs no special case: it is under REPO and the loop above
    # already found it. An explicit second branch here appended every coderef
    # hit twice and made 12 correct citations look ambiguous against
    # themselves -- caught on this check's second run, by the case that exists
    # to catch exactly this shape of duplicate.
    return hits


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    absent = parse_ledger(LEDGER.read_text(encoding="utf-8"))

    files = [p for p in iter_corpus(argv) if p.is_file()]
    if not files:
        print("FAIL parses_the_corpus: no files to scan")
        return 1

    citations, links = [], []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO) if REPO in path.parents else path
        for n, line in enumerate(text.splitlines(), 1):
            for m in CITATION.finditer(line):
                citations.append((str(rel), n, m.group(1), int(m.group(2)),
                                  int(m.group(3) or m.group(2))))
            for m in DOC_LINK.finditer(line):
                links.append((str(rel), n, m.group(1), path.parent))

    if not citations:
        print(f"FAIL parses_the_corpus: {len(files)} file(s) scanned, zero "
              "`path:line` citations found. Either the corpus is wrong or the "
              "citation format changed; this check will not pass on silence.")
        return 1
    print(f"  ok    parses_the_corpus   {len(files)} file(s), "
          f"{len(citations)} citation(s), {len(links)} doc link(s)")

    fails, warns = [], []

    # --- doc_links_resolve -------------------------------------------------
    bad_links = []
    for src, n, target, base in links:
        if not (base / target).exists() and not (REPO / target).exists():
            bad_links.append((src, n, target))
    for src, n, target in bad_links:
        fails.append(f"  FAIL  doc_links_resolve   {src}:{n} -> {target} does not exist")
    if not bad_links:
        print(f"  ok    doc_links_resolve   all {len(links)} resolve")

    # --- no_bare_basenames / citations_resolve / citations_in_range --------
    bare, unresolved, out_of_range, coderef_missing, absent_returned = [], [], [], [], []
    ambiguous = []
    for src, n, cited, start, end in citations:
        if cited in absent:
            if resolve(cited):
                absent_returned.append((src, n, cited))
            continue
        hits = resolve(cited)
        if "/" not in cited and not hits:
            bare.append((src, n, cited))
            continue
        if not hits:
            if cited.startswith("coderef/"):
                coderef_missing.append((src, n, cited))
            else:
                unresolved.append((src, n, cited))
            continue
        if len(hits) > 1:
            ambiguous.append((src, n, cited, hits))
            continue
        target = hits[0]
        nlines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        if not (1 <= start <= end <= nlines):
            out_of_range.append((src, n, cited, start, end, nlines))

    for src, n, cited in bare:
        fails.append(
            f"  FAIL  no_bare_basenames   {src}:{n} cites `{cited}` with no "
            f"directory part and no file of that name at the repo root. Cite a "
            f"path relative to the repo or to ComfyUI, or declare it absent.")
    if not bare:
        print("  ok    no_bare_basenames   every citation is a path")

    for src, n, cited in unresolved:
        fails.append(f"  FAIL  citations_resolve  {src}:{n} -> {cited} does not exist")
    if not unresolved:
        print(f"  ok    citations_resolve  {len(citations)} citation(s) resolve")

    for src, n, cited, hits in ambiguous:
        where = " and ".join(str(h.relative_to(COMFY.parent)) for h in hits)
        fails.append(
            f"  FAIL  ambiguous_roots    {src}:{n} cites `{cited}`, which "
            f"exists as {where}. Say which: prefix ComfyUI's tree with "
            f"`ComfyUI/`.")
    if not ambiguous:
        print("  ok    ambiguous_roots    no citation resolves in two roots")

    for src, n, cited, start, end, nlines in out_of_range:
        span = f"{start}-{end}" if start != end else str(start)
        fails.append(
            f"  FAIL  citations_in_range {src}:{n} -> {cited}:{span} but that "
            f"file has {nlines} lines")
    if not out_of_range:
        print("  ok    citations_in_range every cited line exists")

    for src, n, cited in absent_returned:
        warns.append(
            f"  warn  declared_absent_still_absent  {cited} is declared absent "
            f"in docs/checks.md but resolves now ({src}:{n}) -- drop the entry")
    if not absent_returned:
        print(f"  ok    declared_absent_still_absent  {len(absent)} declared, "
              "still absent")

    for src, n, cited in coderef_missing:
        warns.append(
            f"  warn  coderef_absent     {src}:{n} -> {cited}; sister checkout "
            "not present. Not a failure -- see this file's docstring")

    for line in warns:
        print(line)
    if fails:
        print()
        for line in fails:
            print(line)
        print(
            f"\n{len(fails)} broken pointer(s). A doc that points at a file or "
            "line\nthat is not there teaches the next reader something false, "
            "and it is\nthe cheapest class of error in this repo to prevent.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
