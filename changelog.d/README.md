# One changelog entry a day

**A commit does not touch the changelog.** No fragment, no build, no version
bump: the commit message is the record (owner, 2026-10-10). Until that day
every commit built its own entry, each build wrote `CHANGELOG.md` and
`pyproject.toml`, and with several sessions in one tree those two files
were in everybody's commit: on that one day three commits carried a peer's
entry and every session queued behind the one before it.

**Once, at the end of a working day, one session writes the day's entry**
from the commits:

1. Read the day's commits: `git log --since=<the day's start> --reverse
   --format='%h %s%n%b'`. The entry is a summary of what changed for someone
   using the repo, grouped as Added, Changed and Fixed; it is not the list
   of commits, and it carries no detail a postmortem should hold.
2. Write that as one fragment, `changelog.d/day-<YYYY-MM-DD>.md`, in the
   form below (`bump: minor` when the day added a node, an input or a tool;
   `patch` otherwise).
3. Build once and commit the fragment, `CHANGELOG.md` and `pyproject.toml`
   by pathspec, each of the three steps as its own command.

`CHANGELOG.md`'s newest entries are generated from the files here by
`bench/build_changelog.py`; its docstring is the full account. The fragments
written per commit before 2026-10-10 stay: the entries above the marker are
built from them.

## The fragment and the build

1. Write the fragment:

   ```
   bump: patch

   ### Changed

   - What changed, written as an entry always was.
   ```

   `bump` is `patch` or `minor`. Do not write a version number: the build
   assigns it. Do not write a date.

2. Build, naming your fragment. It prints the version yours got:

   ```bash
   <python> bench/build_changelog.py --with changelog.d/<yours>.md
   ```

3. Commit your fragment, `CHANGELOG.md` and `pyproject.toml` by pathspec
   (`git add -- changelog.d/<yours>.md` first: it is a new file you
   created). Run the build as a command of its own, with nothing piped
   after it and nothing chained behind it, and read what it prints: a
   refusal that is piped or chained does not stop the commit after it.

To revise an entry before it is committed, edit your fragment and run step
2 again. Each generated entry has a comment under its heading naming the
fragment it came from, which is how the build knows an entry is yours to
rewrite.

## When the build says STOP

It refuses to build beside somebody else's uncommitted work, and writes
nothing when it does:

- **an entry in `CHANGELOG.md` made from a fragment that is not yours and not
  committed**: another session has built and not committed yet. Wait for
  that commit, then build again;
- **`pyproject.toml` differs from `HEAD` by more than its version line**: the
  other lines are somebody's, and the version this build writes would put
  the file in your commit with them. Wait for that commit, or ask its author.

Before you commit, read `git diff --numstat -- CHANGELOG.md pyproject.toml`
in a step of its own: `pyproject.toml` is one line out and one in, and
`CHANGELOG.md` has no line removed.

## What not to do

- **Do not edit `CHANGELOG.md` above the marker line.** The build stops
  when it finds an entry there with no fragment named under its heading,
  so nothing is lost, but the commit after yours cannot build until the
  entry is moved into a fragment.
- **Do not delete, rename or reorder fragments.** Entry numbers come from
  the order fragments were committed in; removing one renumbers every
  entry after it.
- **Do not build with a peer's uncommitted fragment.** Only the fragments
  in `HEAD` and the ones you name are read, which is the point: a peer's
  file in the shared tree is theirs to publish.

## When the check is red

`bench/check_changelog.py` runs `build_changelog.py --check` in the sweep.
"Rebuild" means two sessions built at the same moment and the later commit
wrote the file without the earlier one's entry: run step 2 with no
`--with` and commit `CHANGELOG.md`. "Made by no fragment" means a hand
edit: move it into a fragment.
