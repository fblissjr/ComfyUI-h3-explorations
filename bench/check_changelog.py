#!/usr/bin/env python3
"""`CHANGELOG.md`'s top is what the fragments in `changelog.d/` make, and the builder cannot lose an entry.

    <python> bench/check_changelog.py

**What it guards.** Entries are one file each under `changelog.d/` and
`bench/build_changelog.py` writes them into `CHANGELOG.md` with their
version numbers. Before that, two sessions took one number and a commit
carried a peer's entry (both 2026-10-05). The builder's promises are the
cases here; the last case is the tree itself.

  numbers_follow_the_order       a patch, a minor and a patch on a base
                                 count up as semver says, newest on top,
                                 and everything below the marker is kept
                                 byte for byte.
  a_rebuild_changes_nothing      building twice is building once.
  a_hand_edit_is_not_overwritten an entry above the marker with no fragment
                                 named under its heading stops the build,
                                 even when a fragment says the same words;
                                 so does one made from a fragment that is
                                 neither committed nor named. The control:
                                 an entry that names my fragment is rebuilt
                                 from the file, so it can be revised.
  a_stale_file_is_red            a committed fragment's entry missing from
                                 the file fails the check and says rebuild.
  an_uncommitted_fragment_may_ride_or_not
                                 the check passes with a working-tree
                                 fragment built on top and with it left
                                 out, so a peer's fragment in the shared
                                 tree does not turn anybody's check red;
                                 a wrong number on it fails.
  a_bad_fragment_is_refused      no bump line, a major bump, a body that is
                                 not `### ` sections, a `## ` line inside.
  the_order_is_the_commits       in a scratch repository, through the
                                 command line: a fragment named z committed
                                 before one named a is numbered first; a
                                 peer's uncommitted fragment is not built
                                 in; a hand edit stops the build with the
                                 file left as it was.
  a_foreign_pyproject_change_stops_the_build
                                 in a scratch repository: a build writes
                                 the version into `pyproject.toml` and may
                                 be run again; with one more line changed
                                 in that file by somebody else it stops,
                                 names the line and leaves both files as
                                 they were; and a second session's build
                                 beside a first one's uncommitted entry
                                 stops and says to wait for that commit.
  the_tree                       `build_changelog.py --check` on this
                                 checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import build_changelog as B  # noqa: E402
from _lib import case, finish  # noqa: E402

OLD = "## 0.9.4\n\n### Fixed\n\n- An old entry.\n\n## 0.9.3\n\n### Added\n\n- An older one.\n"
TEXT = "# Changelog\n\nSemantic versioning.\n\n" + B.MARKER + "\n\n" + OLD


def frag(bump: str, line: str) -> str:
    return f"bump: {bump}\n\n### Changed\n\n- {line}\n"


HEAD = [("changelog.d/a.md", frag("patch", "first")), ("changelog.d/b.md", frag("minor", "second")),
        ("changelog.d/c.md", frag("patch", "third"))]


def numbers_follow_the_order():
    new, versions = B.build(TEXT, HEAD, [])
    assert [v for _n, v in versions] == ["0.9.5", "0.10.0", "0.10.1"], versions
    region = new[:new.index(B.MARKER)]
    assert region.index("## 0.10.1") < region.index("## 0.10.0") < region.index("## 0.9.5"), "newest is not on top"
    assert region.index("third") < region.index("second") < region.index("first")
    assert new.endswith(B.MARKER + "\n\n" + OLD), "the text below the marker changed"
    assert new.startswith("# Changelog\n\nSemantic versioning.\n\n## 0.10.1\n<!-- changelog.d/c.md -->\n"), new[:90]


def a_rebuild_changes_nothing():
    once, _v = B.build(TEXT, HEAD, [])
    twice, _v = B.build(once, HEAD, [])
    assert once == twice
    assert B.check(once, HEAD, {}) == [], B.check(once, HEAD, {})


def a_hand_edit_is_not_overwritten():
    built, _v = B.build(TEXT, HEAD[:2], [])
    by_hand = built.replace("## 0.10.0\n", "## 0.10.1\n\n### Added\n\n- typed in by hand\n\n## 0.10.0\n", 1)
    try:
        B.build(by_hand, HEAD[:2], [])
    except B.Refused as exc:
        assert "0.10.1" in str(exc) and "by hand" in str(exc), exc
    else:
        raise AssertionError("an entry no fragment makes was overwritten")
    assert any("made by no fragment" in p for p in B.check(by_hand, HEAD[:2], {})), B.check(by_hand, HEAD[:2], {})
    # naming a fragment with the same words does not make the hand entry safe to drop: it still has no origin
    mine = ("changelog.d/hand.md", "bump: patch\n\n### Added\n\n- typed in by hand\n")
    try:
        B.build(by_hand, HEAD[:2], [mine])
    except B.Refused:
        pass
    else:
        raise AssertionError("a hand entry was overwritten because a fragment happened to say the same")
    # the control: an entry that names my fragment is rebuilt from the file as it is now, so an entry can be revised
    first, _v = B.build(built, HEAD[:2], [mine])
    revised = ("changelog.d/hand.md", "bump: patch\n\n### Added\n\n- said better\n")
    second, versions = B.build(first, HEAD[:2], [revised])
    assert "said better" in second and "typed in by hand" not in second and versions[-1][1] == "0.10.1"
    stale = B.check(first, HEAD[:2], dict([revised]))
    assert len(stale) == 1 and "says now" in stale[0], stale
    # and an entry made from a fragment nobody named here is somebody else's: the build stops
    try:
        B.build(first, HEAD[:2], [])
    except B.Refused as exc:
        assert "changelog.d/hand.md" in str(exc), exc
    else:
        raise AssertionError("a built-in entry of an unnamed, uncommitted fragment was dropped")


def a_stale_file_is_red():
    stale, _v = B.build(TEXT, HEAD[:2], [])              # the third fragment is committed and not in the file
    problems = B.check(stale, HEAD, {})
    assert len(problems) == 1 and "c.md" in problems[0] and "rebuild" in problems[0], problems
    fresh, _v = B.build(stale, HEAD, [])
    assert B.check(fresh, HEAD, {}) == []


def an_uncommitted_fragment_may_ride_or_not():
    mine = ("changelog.d/mine.md", frag("patch", "mine, not committed"))
    peer = ("changelog.d/peer.md", frag("patch", "a peer's, not committed"))
    loose = dict([mine, peer])
    without, _v = B.build(TEXT, HEAD, [])
    with_mine, versions = B.build(TEXT, HEAD, [mine])
    assert versions[-1][1] == "0.10.2"
    assert "a peer's" not in with_mine, "a fragment not named with --with was built in"
    assert B.check(without, HEAD, loose) == [] and B.check(with_mine, HEAD, loose) == []
    wrong = with_mine.replace("## 0.10.2\n", "## 0.10.7\n", 1)
    assert any("should be numbered 0.10.2" in p for p in B.check(wrong, HEAD, loose)), B.check(wrong, HEAD, loose)
    # both on top are numbered in the order they were built
    both, versions = B.build(with_mine, HEAD, [mine, peer])
    assert [v for _n, v in versions[-2:]] == ["0.10.2", "0.10.3"] and B.check(both, HEAD, loose) == []


def a_bad_fragment_is_refused():
    bad = {
        "no bump line": "### Changed\n\n- x\n",
        "a major bump": "bump: major\n\n### Changed\n\n- x\n",
        "an unknown bump": "bump: big\n\n### Changed\n\n- x\n",
        "no section": "bump: patch\n\n- x\n",
        "a version heading inside": "bump: patch\n\n### Changed\n\n- x\n\n## 9.9.9\n\n- y\n",
    }
    for what, text in bad.items():
        try:
            B.parse_fragment(text, "f.md")
        except B.Refused:
            continue
        raise AssertionError(f"a fragment with {what} was accepted")
    assert B.parse_fragment("bump: minor\n\n### Added\n\n- x\n", "f.md") == ("minor", "### Added\n\n- x")


def the_order_is_the_commits():
    """In a scratch repository: two fragments committed z first and then a are numbered z, a; a third, uncommitted, is ignored unless named."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "bench").mkdir()
        (root / "changelog.d").mkdir()
        shutil.copy(B.__file__, root / "bench" / "build_changelog.py")
        (root / "CHANGELOG.md").write_text(TEXT)

        def git(*args):
            done = subprocess.run(["git", "-c", "user.name=check", "-c", "user.email=check@example.invalid",
                                   "-c", "commit.gpgsign=false", *args], cwd=tmp, capture_output=True, text=True)
            assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr.strip()[-200:]}"

        def tool(*args):
            return subprocess.run([sys.executable, "bench/build_changelog.py", *args], cwd=tmp, capture_output=True, text=True)

        git("init", "-q")
        git("add", "CHANGELOG.md", "bench")
        git("commit", "-q", "-m", "base")
        for name, line in (("z-first", "committed first"), ("a-second", "committed second")):
            (root / "changelog.d" / f"{name}.md").write_text(frag("patch", line))
            built = tool("--with", f"changelog.d/{name}.md")
            assert built.returncode == 0, built.stdout + built.stderr
            git("add", "CHANGELOG.md", f"changelog.d/{name}.md")
            git("commit", "-q", "-m", name)
        text = (root / "CHANGELOG.md").read_text()
        assert text.index("## 0.9.6") < text.index("committed second") < text.index("## 0.9.5") < text.index("committed first"), \
            "the fragments are not numbered in the order they were committed"
        assert tool("--check").returncode == 0, tool("--check").stdout
        # a peer's uncommitted fragment in the tree: not built in by a plain build, and the check stays green
        (root / "changelog.d" / "peer.md").write_text(frag("patch", "a peer's"))
        plain = tool()
        assert plain.returncode == 0 and "a peer's" not in (root / "CHANGELOG.md").read_text(), plain.stdout
        assert tool("--check").returncode == 0, tool("--check").stdout
        assert tool("--next").stdout.strip() == "0.9.7", tool("--next").stdout
        # a hand edit above the marker stops the build, with its exit code, and the file is left as it was
        by_hand = text.replace("## 0.9.6\n", "## 0.9.7\n\n### Added\n\n- by hand\n\n## 0.9.6\n", 1)
        (root / "CHANGELOG.md").write_text(by_hand)
        stopped = tool()
        assert stopped.returncode == 3 and "STOP" in stopped.stdout and (root / "CHANGELOG.md").read_text() == by_hand
        assert tool("--check").returncode == 1
        # an uncommitted fragment, built, then revised: the second build takes the revision
        (root / "CHANGELOG.md").write_text(text)
        (root / "changelog.d" / "mine.md").write_text(frag("patch", "first wording"))
        assert tool("--with", "changelog.d/mine.md").returncode == 0
        (root / "changelog.d" / "mine.md").write_text(frag("patch", "second wording"))
        assert tool("--check").returncode == 1, "a fragment revised after it was built did not turn the check red"
        again = tool("--with", "changelog.d/mine.md")
        after = (root / "CHANGELOG.md").read_text()
        assert again.returncode == 0 and "second wording" in after and "first wording" not in after, again.stdout
        assert tool("--check").returncode == 0, tool("--check").stdout


def a_foreign_pyproject_change_stops_the_build():
    """A build never writes the version into a `pyproject.toml` that holds somebody else's uncommitted lines."""
    head = '[project]\nname = "x"\nversion = "0.9.4"\n'
    assert B.pyproject_foreign(head, head.replace("0.9.4", "0.9.5")) == [], "a version line alone was called foreign"
    theirs = head.replace("0.9.4", "0.9.5") + 'dependencies = ["y"]\n'
    assert B.pyproject_foreign(head, theirs) == ['+dependencies = ["y"]'], B.pyproject_foreign(head, theirs)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "bench").mkdir()
        (root / "changelog.d").mkdir()
        shutil.copy(B.__file__, root / "bench" / "build_changelog.py")
        (root / "CHANGELOG.md").write_text(TEXT)
        (root / "pyproject.toml").write_text(head)

        def git(*args):
            done = subprocess.run(["git", "-c", "user.name=check", "-c", "user.email=check@example.invalid",
                                   "-c", "commit.gpgsign=false", *args], cwd=tmp, capture_output=True, text=True)
            assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr.strip()[-200:]}"

        def tool(*args):
            return subprocess.run([sys.executable, "bench/build_changelog.py", *args], cwd=tmp, capture_output=True, text=True)

        git("init", "-q")
        git("add", "CHANGELOG.md", "bench", "pyproject.toml")
        git("commit", "-q", "-m", "base")
        (root / "changelog.d" / "mine.md").write_text(frag("patch", "mine"))
        first = tool("--with", "changelog.d/mine.md")
        assert first.returncode == 0 and 'version = "0.9.5"' in (root / "pyproject.toml").read_text(), first.stdout
        assert tool("--with", "changelog.d/mine.md").returncode == 0, "my own build could not be run again"
        # somebody else's line arrives in pyproject.toml, uncommitted
        git("checkout", "-q", "--", "CHANGELOG.md", "pyproject.toml")
        (root / "pyproject.toml").write_text(head + 'dependencies = ["y"]\n')
        before = ((root / "CHANGELOG.md").read_text(), (root / "pyproject.toml").read_text())
        stopped = tool("--with", "changelog.d/mine.md")
        assert stopped.returncode == 3 and "STOP" in stopped.stdout and "dependencies" in stopped.stdout, stopped.stdout
        assert ((root / "CHANGELOG.md").read_text(), (root / "pyproject.toml").read_text()) == before, "a refused build wrote a file"
        # a first session's build is in the tree, uncommitted; a second session builds beside it
        git("checkout", "-q", "--", "pyproject.toml")
        assert tool("--with", "changelog.d/mine.md").returncode == 0
        (root / "changelog.d" / "second.md").write_text(frag("patch", "a second session's"))
        waits = tool("--with", "changelog.d/second.md")
        assert waits.returncode == 3 and "wait for that commit" in waits.stdout and "mine.md" in waits.stdout, waits.stdout
        assert "a second session's" not in (root / "CHANGELOG.md").read_text()


def the_tree():
    problems = B.check(B.CHANGELOG.read_text(), B.committed(), {p: (B.REPO / p).read_text() for p in B.uncommitted()})
    assert not problems, "; ".join(problems)


def main() -> int:
    for fn in (numbers_follow_the_order, a_rebuild_changes_nothing, a_hand_edit_is_not_overwritten, a_stale_file_is_red,
               an_uncommitted_fragment_may_ride_or_not, a_bad_fragment_is_refused, the_order_is_the_commits,
               a_foreign_pyproject_change_stops_the_build, the_tree):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
