#!/usr/bin/env python
"""Generate `docs/wiki/index.md` from CLAUDE.md's routing tables and the tree.

## Why this is generated and not written

CLAUDE.md already carries the curation: two tables mapping a document to the
question it answers, written by the people who own those documents. A wiki that
retyped those blurbs would be a second copy of the thing this repo's ownership
rule exists to prevent, and it would go stale the first time a blurb was
edited -- silently, because nothing compares two prose descriptions.

So the router is DERIVED. The blurbs stay in CLAUDE.md, which is loaded into
every session anyway; this file re-shapes them into an entry point that answers
"where do I start" for a reader who is not an agent with CLAUDE.md in context,
and that can carry the sister-checkout map alongside them.

## What it reports rather than asserts

**Unreachable documents.** Not "has no CLAUDE.md row" -- that was the first
version of this report and it named 72 files, nearly all of them dated records
inside a research subtree whose own README is their route. Naming them all is
how a report teaches you to ignore it.

The question that matters is whether a reader can GET to a document, so this
walks the link graph: start from the rows in CLAUDE.md's tables, follow markdown
links between documents, and report only what nothing reaches. `morton.md` and
`sol_upstream.md` are reached through `SOLATTN.md` by design and are correctly
absent from the table; a document nobody links is the actual defect.

It is a REPORT, not a gate. Per CLAUDE.md's no-new-check rule, no drift instance
has escaped here yet. A gate would have to encode which unreachable documents
are deliberate, and it cannot.

**Rows naming a file that is gone** are already `check_doc_links.py`'s job and
are not re-checked here.

## The generated file

`docs/wiki/index.md` is generated. Do not hand-edit it -- rerun this. Anything
that must be written by a person goes in a sibling page under `docs/wiki/`,
which this script never touches.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLAUDE = REPO / "CLAUDE.md"
OUT = REPO / "docs" / "wiki" / "index.md"

# A routing row: `| [`docs/x.md`](docs/x.md) | what it answers |`
#
# **The first cell may name MORE THAN ONE document.** `prompt_catalogue.md` and
# `prompt_audit.md` share a row because they are a generated/hand-written pair
# and routing to one without the other is misleading. Until 2026-09-01 this
# pattern required the cell to hold exactly one link, so that row matched
# nothing and BOTH prompt documents were silently absent from the generated
# router -- not reported as unreachable either, since other documents link
# them. A parser that drops a row it cannot shape is the failure this file's
# own `parsed no routing rows` guard exists to catch, one row at a time.
ROW = re.compile(r"^\|\s*(\[`docs/[^`]+\.md`\]\([^)]+\)(?:\s*\+\s*\[`docs/[^`]+\.md`\]\([^)]+\))*)\s*\|\s*(.+?)\s*\|\s*$")

# Every `docs/...md` path named inside a matched first cell, in order.
ROW_PATHS = re.compile(r"\[`(docs/[^`]+\.md)`\]")

# The two tables that carry routing blurbs, by the heading above them.
SECTIONS = {
    "### Read these before you start": "start here",
    "### Reference, when you touch the thing it covers": "reference",
}


def parse_claude(text: str) -> list[tuple[tuple[str, ...], str, str]]:
    """Return (paths, blurb, section-label) for every routing row, in file order.

    `paths` is a tuple because one row may route to several documents.
    """
    rows: list[tuple[tuple[str, ...], str, str]] = []
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in SECTIONS:
            current = SECTIONS[stripped]
            continue
        if stripped.startswith("### ") and stripped not in SECTIONS:
            current = None
            continue
        if current is None:
            continue
        m = ROW.match(line)
        if m:
            # The wiki's own entry point has a row in CLAUDE.md so a reader can
            # find it. Filtered here so the generated router never lists itself.
            paths = tuple(p for p in ROW_PATHS.findall(m.group(1))
                          if not p.startswith("docs/wiki/"))
            if not paths:
                continue
            rows.append((paths, m.group(2), current))
    return rows


def docs_on_disk() -> list[str]:
    root = REPO / "docs"
    found = []
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("docs/wiki/"):
            continue  # the wiki describes the docs, not itself
        found.append(rel)
    return found


LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def unreachable(routed: set[str], on_disk: list[str]) -> list[str]:
    """Documents nothing links to, starting from the routed set.

    Reachability, not membership. A deep dive reached only through its parent
    (`morton.md` through `SOLATTN.md`) is correctly absent from CLAUDE.md's
    tables and is NOT a finding; a document no file links at all is.
    """
    known = set(on_disk)
    reached: set[str] = set()
    frontier = [p for p in routed if p in known]
    reached.update(frontier)

    while frontier:
        current = frontier.pop()
        path = REPO / current
        if not path.exists():
            continue
        base = Path(current).parent
        for href in LINK.findall(path.read_text(encoding="utf-8", errors="replace")):
            href = href.split("#", 1)[0].strip()
            if not href or "://" in href:
                continue
            target = (base / href).as_posix() if not href.startswith("docs/") else href
            # Normalise `../` segments without touching the filesystem.
            target = Path(target).as_posix()
            try:
                target = (REPO / target).resolve().relative_to(REPO).as_posix()
            except (ValueError, OSError):
                continue
            if target in known and target not in reached:
                reached.add(target)
                frontier.append(target)

    return [p for p in on_disk if p not in reached]


def collapse(paths: list[str], threshold: int = 3) -> list[str]:
    """Group an unreachable list by directory so it stays readable.

    A report nobody reads is worth nothing. A directory contributing more than
    `threshold` entries is named once with its count -- the useful fact there is
    "this whole subtree is unreachable", not each file in it.
    """
    from collections import defaultdict
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_dir[str(Path(p).parent)].append(p)
    lines: list[str] = []
    for d, members in sorted(by_dir.items()):
        if len(members) > threshold:
            lines.append(f"{d}/ -- {len(members)} file(s), the whole subtree")
        else:
            lines.extend(members)
    return lines


def first_sentence(blurb: str, limit: int = 200) -> str:
    """Trim a blurb to its first sentence for the compact routing table.

    Blurbs are markdown: inline code, `**bold**` and links. Leave them intact
    rather than stripping, so the route reads the way its owner wrote it -- but
    only cut where every span is closed.

    **The backtick check alone was not enough**, and the first version shipped
    the proof: two rows in the generated index opened `**` and never closed it,
    because the cut landed mid-emphasis. A markdown viewer then renders the
    rest of the table as bold. So track emphasis and links as well as code, and
    require a sentence end to be followed by whitespace -- `docs/x.md` and
    `0.83.0` both contain a `.` that ends nothing.
    """
    code = False
    bold = 0
    ital = 0
    link = 0
    i = 0
    n = len(blurb)
    last_safe: int | None = None

    while i < n:
        ch = blurb[i]
        if ch == "`":
            code = not code
            i += 1
            continue
        if not code:
            if blurb.startswith("**", i):
                bold ^= 1
                i += 2
                continue
            if ch == "*":
                ital ^= 1
            elif ch == "[":
                link += 1
            elif ch == "]":
                link = max(0, link - 1)
        clear = not code and not bold and not ital and link == 0
        # A sentence end only counts when followed by whitespace or EOS.
        if clear and ch in ".;" and (i + 1 >= n or blurb[i + 1].isspace()):
            if i > 40:
                return blurb[: i + 1]
            last_safe = i + 1
        elif clear and ch == " ":
            last_safe = i
        if i >= limit and clear:
            return blurb[:i].rstrip() + " ..."
        i += 1

    # Nothing to cut on, or every candidate sat inside a span: return whole.
    # Truncating here would be the bug this function exists to avoid.
    del last_safe
    return blurb


def links(paths: tuple[str, ...]) -> str:
    """Render a row's documents as one cell, joined the way CLAUDE.md joins them."""
    return " + ".join(f"[`{p}`](../../{p})" for p in paths)


def render(rows, unrouted) -> str:
    start = [r for r in rows if r[2] == "start here"]
    ref = [r for r in rows if r[2] == "reference"]

    out: list[str] = []
    w = out.append
    w("# The wiki: where to start, and who owns each answer")
    w("")
    w("**Generated by `bench/build_wiki_index.py`. Do not hand-edit — rerun it.**")
    w("Everything below is re-shaped from CLAUDE.md's own routing tables, so a")
    w("blurb edited there appears here on the next run and cannot drift in between.")
    w("Pages written by a person live beside this one and are never touched by")
    w("the generator; they are listed under *Written pages*.")
    w("")
    w("This is a router, not an authority. It states no fact about H3 that is not")
    w("owned somewhere else, and where it disagrees with an owner the owner is right.")
    w("")
    w("## Written pages")
    w("")
    w("| page | what it is for |")
    w("|---|---|")
    w("| [`next_steps.md`](next_steps.md) | **what to do next, and only pointers to "
      "why.** The short form of `docs/roadmap.md`'s current forward plan; start "
      "there when the question is what to work on, and when the two disagree the "
      "roadmap is right |")
    w("| [`references.md`](references.md) | the sister checkouts under `coderef/`: "
      "what each one implements for H3, what has been compared against it, and at "
      "which revision. The map to reach for before proposing a borrow |")
    w("| [`stages.md`](stages.md) | one row per stage of a render: our code, the "
      "document that owns it, the check that guards it, and the implementation to "
      "compare against |")
    w("| [`prompting.md`](prompting.md) | **prompting: where every rule, example "
      "and verdict comes from.** The five sources that claim to govern a prompt "
      "ranked by authority, which document owns which question, at least five "
      "worked examples per mode, and how to grade a draft before rendering it |")
    w("")
    w("## Start here")
    w("")
    w("Read these before starting work, in this order.")
    w("")
    w("| # | file | what it answers |")
    w("|---|---|---|")
    for i, (paths, blurb, _) in enumerate(start, 1):
        w(f"| {i} | {links(paths)} | {blurb} |")
    w("")
    w("## Reference, when you touch the thing it covers")
    w("")
    w("| file | what it answers |")
    w("|---|---|")
    for paths, blurb, _ in ref:
        w(f"| {links(paths)} | {first_sentence(blurb)} |")
    w("")
    w("## Documents nothing reaches")
    w("")
    if not unrouted:
        w("None: every `.md` under `docs/` is reachable by following links from")
        w("CLAUDE.md's routing tables.")
    else:
        w("Present on disk and reached by no link from any routed document.")
        w("**This is a report, not a defect list** -- some of these are deliberate.")
        w("A deep dive reached through its parent does not appear here; that is the")
        w("difference between this list and a membership test.")
        w("")
        for entry in collapse(unrouted):
            w(f"- `{entry}`")
    w("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the generated file is stale")
    args = ap.parse_args()

    if not CLAUDE.exists():
        print(f"FAIL  CLAUDE.md not found at {CLAUDE.relative_to(REPO)}")
        return 2

    rows = parse_claude(CLAUDE.read_text(encoding="utf-8"))
    if not rows:
        # The parse is the whole value of this script. Silence here means the
        # table shape changed, and emitting an empty router would look like a
        # repo with no documents rather than a broken parser.
        print("FAIL  parsed no routing rows from CLAUDE.md -- table shape changed?")
        return 2

    routed = {p for paths, _, _ in rows for p in paths}
    on_disk = docs_on_disk()
    unrouted = unreachable(routed, on_disk)

    missing = sorted(p for p in routed if not (REPO / p).exists())

    text = render(rows, unrouted)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("FAIL  docs/wiki/index.md is stale -- rerun bench/build_wiki_index.py")
            return 1
        print("ok    docs/wiki/index.md is current")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")

    print(f"ok    wrote {OUT.relative_to(REPO)}")
    print(f"ok    routed            {len(rows)} row(s) from CLAUDE.md")
    print(f"      unreachable       {len(unrouted)} doc(s) no link reaches")
    for entry in collapse(unrouted):
        print(f"        {entry}")
    if missing:
        # check_doc_links.py owns this direction; printed here so a run that
        # generates a router pointing at a gone file says so at the time.
        print(f"      rows naming a missing file  {len(missing)} "
              "(check_doc_links.py owns this)")
        for p in missing:
            print(f"        {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
