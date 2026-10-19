# docs/projects/: work that spans folders and sessions

Loaded when a session opens a file here; `docs/AGENTS.md` and the root `AGENTS.md` still govern.

- **One folder a project.** Its `index.md` opens with the goal and where it stands, then what is done (with commits),
  what is left (in order) and what waits on the owner. `index.md` here lists every project and its state.
- **A project page is a plan, not a log.** A finished step is one line with its commit; the reasoning lives in the
  commit, `docs/state/decisions.md` or a record. A decision the owner must make is a card on the H3 Open List, and the
  project names it.
- **`<project>/local/` is gitignored**: scratch, data a project was built from, anything not to be published. A tracked
  page may name a file there; it does not depend on it.
- **When a project is done**, its index says so on its first line and the folder stays as the record; its standing
  rules have moved to the pages that own them.
