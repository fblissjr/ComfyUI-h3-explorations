bump: minor

### Changed

- `bench/build_changelog.py` refuses to build beside somebody else's uncommitted work and writes nothing when it does. It stops when `pyproject.toml` differs from `HEAD` by anything but its version line, naming the lines: every build writes the version into that file, so it is in every session's pathspec, and a commit takes a file whole. Its existing refusal of an entry in `CHANGELOG.md` made from a fragment that is neither committed nor named now says what to do: wait for that commit and build again. Together they make builds serial without anybody announcing one. Four commits on 2026-10-10 had carried a peer's hunk in a shared file; the last was a `dependencies` block in `pyproject.toml` under a commit about something else. `bench/check_changelog.py` gains a case in a scratch repository for both refusals, and `changelog.d/README.md` says what a STOP means.
