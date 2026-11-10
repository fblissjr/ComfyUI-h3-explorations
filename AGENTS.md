# ComfyUI-h3-explorations

Emptied on 2026-11-10 as a trial (`a3996113`). The rule below came back the same day at the owner's word; the file as
it stood before is `git show a3996113^:AGENTS.md`.

## Operative rules

- **The shared tree.** `git status` before editing, and commit promptly with
  `git commit -F <msg> -- <paths>`. Never `git add`, `commit -a`, a bare
  `git commit` or `--amend`: each has swept a peer's work into another
  commit. The one exception is `git add -- <file>` for a new file you
  created. A file holding a peer's uncommitted hunk is theirs to commit.
  Subagents never run git in the shared tree (a worker commits only on its
  own branch in its own worktree). The owner pushes from their own terminal, so
  every commit message is written as already public. A session never runs
  `git push`, to any remote, with or without force.
