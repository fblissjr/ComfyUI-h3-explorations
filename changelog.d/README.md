# One file per changelog entry

`CHANGELOG.md`'s newest entries are generated from the files here by
`bench/build_changelog.py`. Its docstring is the full account; this is the
part you need at commit time.

## To add an entry

1. Write `changelog.d/<session>-<a-few-words>.md`:

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

3. Commit your fragment and `CHANGELOG.md` with your other paths, by
   pathspec as always (`git add -- changelog.d/<yours>.md` first: it is a
   new file you created).

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
