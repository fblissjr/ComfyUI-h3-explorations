#!/usr/bin/env python3
"""Build the top of `CHANGELOG.md` from one fragment file per entry, and number the entries when it is built.

    <python> bench/build_changelog.py --with changelog.d/<mine>.md     # write; prints the version mine got
    <python> bench/build_changelog.py --check                          # is CHANGELOG.md what the fragments make
    <python> bench/build_changelog.py --next                           # the version a new patch entry would get

**What it buys.** Several sessions commit to this tree at once, and each
used to add its entry to the top of one file and pick the next version
number by reading it. Two failures followed, both seen on 2026-10-05: two
sessions took the same number, and a commit by pathspec of `CHANGELOG.md`
carried a peer's uncommitted entry with it. With a file per entry nobody
edits a shared file by hand, and the number is assigned from the order the
fragments were committed in, so it cannot collide.

**A fragment** is `changelog.d/<any-name>.md`:

    bump: patch

    ### Changed

    - What changed, as an entry was always written.

`bump` is `patch` or `minor`. A major bump is refused: it needs the owner's
word, and then an edit here. The name is yours to choose and only has to be
unique; a session prefix does that.

**The order and the numbers.** Fragments already committed are ordered by
the commit that added each (`git log`, oldest first; two added by one
commit, by name). The first takes the newest version below the marker line
plus its bump, and so on up. Fragments given with `--with` are not
committed yet and go on top, in the order given. A number never changes
once a later fragment exists, because fragments are never deleted or
reordered. Only the fragments in `HEAD` and those named with `--with` are
read when writing: a peer's uncommitted fragment in the shared tree is not
yours to publish.

**What is rewritten.** The entries between the file's preamble and the
marker line. Everything below the marker is the changelog as it stood when
fragments began, kept byte for byte.

**It refuses to write over a hand edit.** Under each generated heading is a
comment naming the fragment that made the entry. An entry above the marker
with no such line was added by hand, and one naming a fragment that is
neither committed nor given with `--with` is somebody else's: in both
cases it stops and names the heading, because rebuilding would delete the
entry. An entry that names your fragment is rebuilt from the file as it is
now, so an entry can be revised before it is committed.

**It refuses to build beside somebody else's uncommitted work** (2026-10-10,
after four commits in one day carried a peer's hunk). Every build writes
`pyproject.toml`'s version, so that file is in every session's pathspec, and
a pathspec commit takes a file whole. So a build stops, writing nothing,
when `pyproject.toml` differs from `HEAD` by anything but its version line:
the other lines are somebody's, and a commit that carries the version would
carry them. It names the lines. With the refusal above (an entry in
`CHANGELOG.md` made from a fragment that is neither committed nor named),
this makes builds serial without anybody announcing one: the second session
waits for the first one's commit and builds again.

**`--check`** (run by `bench/check_changelog.py` in the sweep) passes when
the region is exactly what `HEAD`'s fragments make, with any fragments
that are in the working tree and not yet committed on top of them. So it
is green in the tree of a session about to commit, and red when the file
is stale, hand-edited, or a fragment is malformed.

**The race that remains.** Two sessions build at once, each from `HEAD`
plus its own fragment; the second to commit writes a file without the
first's entry and may print a number the first also printed. The fragments
are both in `HEAD` afterwards, so the next build restores the entry and
the numbers; `--check` is red in between and says to rebuild. A commit
subject that quotes a number can be one off after such a race; the
changelog is the record.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
FRAGMENTS = REPO / "changelog.d"
MARKER = ("<!-- Entries above this line are generated from changelog.d/ by bench/build_changelog.py. "
          "Do not edit them by hand: add a fragment. Entries below are kept as written. -->")
HEADING = re.compile(r"^## (\d+)\.(\d+)\.(\d+)[ \t]*$", re.M)
# under each generated heading: which fragment made the entry. A comment, so it does not show when rendered.
MADE_FROM = re.compile(r"^<!-- (changelog\.d/[^ ]+\.md) -->[ \t]*$", re.M)
BUMPS = ("patch", "minor")
NOT_FRAGMENTS = {"README.md"}


class Refused(Exception):
    """The build cannot go ahead; the message says what to do."""


def bumped(version: tuple[int, int, int], bump: str) -> tuple[int, int, int]:
    major, minor, patch = version
    return (major, minor + 1, 0) if bump == "minor" else (major, minor, patch + 1)


def parse_fragment(text: str, name: str) -> tuple[str, str]:
    """(bump, body) of a fragment's text. The body is what goes under the version heading."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("bump:"):
        raise Refused(f"{name}: the first line must be `bump: patch` or `bump: minor`")
    bump = lines[0].split(":", 1)[1].strip()
    if bump == "major":
        raise Refused(f"{name}: a major bump needs the owner's word; this script does not assign one")
    if bump not in BUMPS:
        raise Refused(f"{name}: bump is {bump!r}; it must be one of {', '.join(BUMPS)}")
    body = "\n".join(lines[1:]).strip("\n")
    if not body.startswith("### "):
        raise Refused(f"{name}: after the bump line and a blank line, the entry must start with a `### ` section")
    if re.search(r"^##? ", body, re.M):
        raise Refused(f"{name}: an entry holds `### ` sections only; a `# ` or `## ` line would read as another entry")
    if MARKER in body:
        raise Refused(f"{name}: an entry may not hold the marker line")
    return bump, body


def split(text: str) -> tuple[str, str, str]:
    """(preamble, the generated region, the marker and everything below it) of the changelog's text."""
    at = text.find(MARKER)
    if at < 0:
        raise Refused(f"{CHANGELOG.name} has no marker line; run with --init once to place it above the newest entry")
    first = HEADING.search(text)
    start = first.start() if first and first.start() < at else at
    return text[:start], text[start:at], text[at:]


def entries(region: str) -> list[tuple[tuple[int, int, int], str | None, str]]:
    """[(version, the fragment it says it was made from or None, body)] of a run of entries, in the order they appear."""
    marks = list(HEADING.finditer(region))
    if region.strip() and (not marks or region[:marks[0].start()].strip()):
        raise Refused("the region above the marker holds text that is not under a `## x.y.z` heading")
    out = []
    for mark, following in zip(marks, [*marks[1:], None]):
        body = region[mark.end():following.start() if following else len(region)].strip("\n")
        made_from = MADE_FROM.match(body)
        source = made_from.group(1) if made_from else None
        if made_from:
            body = body[made_from.end():].strip("\n")
        out.append((tuple(int(g) for g in mark.groups()), source, body))
    return out


def base_version(below: str) -> tuple[int, int, int]:
    first = HEADING.search(below)
    if not first:
        raise Refused("no `## x.y.z` entry below the marker to count from")
    return tuple(int(g) for g in first.groups())


def render(base: tuple[int, int, int], fragments: list[tuple[str, tuple[str, str]]]) -> tuple[str, list[tuple[int, int, int]]]:
    """(the generated region, each fragment's version) for fragments given oldest first as (name, (bump, body))."""
    versions, version = [], base
    for _name, (bump, _body) in fragments:
        version = bumped(version, bump)
        versions.append(version)
    blocks = [f"## {'.'.join(map(str, v))}\n<!-- {name} -->\n\n{body}\n\n"
              for v, (name, (_bump, body)) in zip(versions, fragments)]
    return "".join(reversed(blocks)), versions


def _git(*args: str) -> list[str]:
    done = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed: {done.stderr.strip()[-200:]}")
    return [ln for ln in done.stdout.splitlines() if ln.strip()]


def committed() -> list[tuple[str, str]]:
    """[(repo-relative path, text as HEAD has it)] of the fragments in HEAD, in the order they were committed."""
    rel = FRAGMENTS.name
    in_head = {p for p in _git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
               if p.endswith(".md") and Path(p).name not in NOT_FRAGMENTS}
    if not in_head:
        return []
    order: list[str] = []
    # one blank-separated group of names per commit, oldest commit first; names sorted inside a commit
    group: list[str] = []
    for line in subprocess.run(["git", "log", "--reverse", "--diff-filter=A", "--name-only", "--format=%x00", "--", rel],
                               cwd=str(REPO), capture_output=True, text=True).stdout.splitlines() + ["\x00"]:
        if line == "\x00":
            order += sorted(p for p in group if p in in_head and p not in order)
            group = []
        elif line.strip():
            group.append(line.strip())
    order += sorted(in_head - set(order))          # a fragment the log did not show (a shallow clone)
    out = []
    for path in order:
        done = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=str(REPO), capture_output=True, text=True)
        out.append((path, done.stdout))
    return out


def uncommitted() -> list[str]:
    """Repo-relative paths of fragments in the working tree that HEAD does not have."""
    in_head = set(_git("ls-tree", "-r", "--name-only", "HEAD", "--", FRAGMENTS.name))
    return sorted(str(p.relative_to(REPO)) for p in FRAGMENTS.glob("*.md")
                  if p.name not in NOT_FRAGMENTS and str(p.relative_to(REPO)) not in in_head)


def check(text: str, head: list[tuple[str, str]], loose: dict[str, str]) -> list[str]:
    """What is wrong with the changelog's generated region; empty when nothing is.

    `head` is the committed fragments oldest first as (name, text); `loose`
    the uncommitted ones in the working tree, {name: text}.
    """
    try:
        _preamble, region, below = split(text)
        base = base_version(below)
        want = [parse_fragment(body, name) for name, body in head]
        loose_parsed = {name: parse_fragment(body, name) for name, body in loose.items()}
        have = list(reversed(entries(region)))                       # oldest first
    except Refused as exc:
        return [str(exc)]
    problems = []
    version = base
    for n, (bump, body) in enumerate(want):
        version = bumped(version, bump)
        if n >= len(have):
            problems.append(f"the entry of {head[n][0]} ({'.'.join(map(str, version))}) is missing: rebuild")
            continue
        if have[n] != (version, head[n][0], body):
            problems.append(f"entry {'.'.join(map(str, have[n][0]))} is not what {head[n][0]} makes "
                            f"({'.'.join(map(str, version))}): rebuild, or it was edited by hand")
    left = dict(loose_parsed)
    for got_version, source, got_body in have[len(want):]:
        shown = ".".join(map(str, got_version))
        if source is None:
            problems.append(f"entry {shown} is made by no fragment: it was added by hand; move it into changelog.d/")
            continue
        if source not in left:
            problems.append(f"entry {shown} says it was made from {source}, which is not a fragment here: "
                            "rebuild, or restore the file")
            continue
        bump, body = left.pop(source)
        version = bumped(version, bump)
        if got_body != body:
            problems.append(f"entry {shown} is not what {source} says now: rebuild with --with {source}")
        elif got_version != version:
            problems.append(f"entry {shown} of {source} should be numbered {'.'.join(map(str, version))}: rebuild")
    return problems


def build(text: str, head: list[tuple[str, str]], mine: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """(the changelog's new text, [(fragment name, version)]) from committed fragments and mine on top."""
    preamble, region, below = split(text)
    fragments = [(name, parse_fragment(body, name)) for name, body in [*head, *mine]]
    known = {name for name, _parsed in fragments}
    for version, source, _body in entries(region):
        shown = ".".join(map(str, version))
        if source is None:
            raise Refused(f"entry {shown} above the marker was added by hand: no fragment is named under its heading. "
                          "Rebuilding would delete it; move it into a file in changelog.d/ and build with --with.")
        if source not in known:
            raise Refused(f"entry {shown} above the marker was made from {source}, which is not in HEAD and was not "
                          "given with --with. If it is yours, name it; if it is a peer's, their build is in the tree "
                          "and not committed yet: wait for that commit and build again. Rebuilding now would delete "
                          "their entry. Nothing was written.")
    generated, versions = render(base_version(below), fragments)
    return preamble + generated + below, [(name, ".".join(map(str, v))) for (name, _p), v in zip(fragments, versions)]


PYPROJECT = REPO / "pyproject.toml"
PYPROJECT_VERSION = re.compile(r'^(version\s*=\s*")([0-9]+\.[0-9]+\.[0-9]+)(")', re.M)


def pyproject_foreign(head: str, now: str) -> list[str]:
    """The lines by which `pyproject.toml` differs from `HEAD`'s, its version line apart: `-` for a line HEAD
    has and the tree does not, `+` for the reverse. Empty when the version is the only difference."""
    import difflib
    blank = lambda text: PYPROJECT_VERSION.sub(r"\g<1>x.y.z\g<3>", text).splitlines()     # noqa: E731
    return [line for line in difflib.unified_diff(blank(head), blank(now), lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]


def refuse_beside_foreign_pyproject() -> None:
    """Stop when `pyproject.toml` holds somebody's uncommitted change: this build would put it in my commit."""
    if not PYPROJECT.exists():
        return                  # a scratch tree with no pack
    done = subprocess.run(["git", "show", f"HEAD:{PYPROJECT.name}"], cwd=str(REPO), capture_output=True, text=True)
    if done.returncode != 0:
        return                  # not in HEAD yet: nothing to compare with
    foreign = pyproject_foreign(done.stdout, PYPROJECT.read_text())
    if foreign:
        shown = "; ".join(line.strip() for line in foreign[:4]) + (" ..." if len(foreign) > 4 else "")
        raise Refused(f"{PYPROJECT.name} differs from HEAD by more than its version line ({shown}). That is somebody's "
                      "uncommitted work, and this build writes the version into the same file, so a commit carrying the "
                      "version would carry it too. Wait for that commit (or ask its author) and build again. Nothing was written.")


def sync_pyproject(version: str) -> bool:
    """`pyproject.toml`'s `version` set to the changelog's newest; True when it was changed.

    That field is what ComfyUI's registry and manager read as the pack's version, and it had
    sat at 0.13.0 since f66fa9d1 while the changelog went past 0.213 (owner, 2026-10-06:
    "might need to make sure what mask version vs node versions mean in this repo").
    """
    if not PYPROJECT.exists():
        return False            # a scratch tree (the check's) has a changelog and no pack
    text = PYPROJECT.read_text()
    want = version
    m = PYPROJECT_VERSION.search(text)
    if m is None:
        raise Refused(f"{PYPROJECT.name} has no version line this script recognises")
    if m.group(2) == want:
        return False
    PYPROJECT.write_text(text[:m.start(2)] + want + text[m.end(2):])
    print(f"wrote {PYPROJECT.name}: version {m.group(2)} -> {want}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--with", dest="mine", nargs="+", type=Path, default=[], metavar="FRAGMENT",
                    help="your fragment(s), not committed yet; they go on top in this order")
    ap.add_argument("--check", action="store_true", help="say whether CHANGELOG.md is what the fragments make; write nothing")
    ap.add_argument("--next", action="store_true", help="print the version a new patch fragment would get; write nothing")
    ap.add_argument("--init", action="store_true", help="place the marker line above the newest entry, once")
    args = ap.parse_args()
    text = CHANGELOG.read_text()
    try:
        if args.init:
            if MARKER in text:
                print("the marker is already there")
                return 0
            first = HEADING.search(text)
            if not first:
                raise Refused("no `## x.y.z` entry to place the marker above")
            CHANGELOG.write_text(text[:first.start()] + MARKER + "\n\n" + text[first.start():])
            print(f"marker placed above {first.group(0).strip()}")
            return 0
        head = committed()
        if args.check:
            problems = check(text, head, {p: (REPO / p).read_text() for p in uncommitted()})
            for problem in problems:
                print(f"FAIL  {problem}")
            if not problems:
                print(f"ok    CHANGELOG.md is what {len(head)} committed fragment(s) make")
            return 1 if problems else 0
        if args.next:
            _text, versions = build(text, head, [("(next)", "bump: patch\n\n### Changed\n\n- x")])
            print(versions[-1][1])
            return 0
        mine = []
        for path in args.mine:
            full = path if path.is_absolute() else Path.cwd() / path
            if full.resolve().parent != FRAGMENTS.resolve():
                raise Refused(f"{path} is not in {FRAGMENTS.name}/")
            mine.append((str(full.resolve().relative_to(REPO)), full.read_text()))
        in_head = {name for name, _body in head}
        mine = [(name, body) for name, body in mine if name not in in_head]
        new, versions = build(text, head, mine)
        refuse_beside_foreign_pyproject()
        if new != text:
            CHANGELOG.write_text(new)
        sync_pyproject(versions[-1][1] if versions else ".".join(map(str, base_version(split(new)[2]))))
        for name, version in versions[len(head):]:
            print(f"{version}  {name}")
        print(("wrote" if new != text else "unchanged:") + f" CHANGELOG.md, {len(versions)} generated entr"
              + ("y" if len(versions) == 1 else "ies"))
        return 0
    except Refused as exc:
        print(f"STOP  {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
