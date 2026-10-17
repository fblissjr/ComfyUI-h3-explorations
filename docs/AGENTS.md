# docs/: what the repo knows, in prose

Loaded when a session opens a file here; the root `AGENTS.md` still governs.

- **`wiki/index.md` is the router**: find the page that owns a fact there before writing one.
- **Prose is a cache.** Where a page and the code disagree, the code is right: correct the page and log what it used to
  claim in `wiki/decisions.md`. A figure in prose points at the command or constant that holds it.
- **`research/` holds dated records** (a day's study, a postmortem): what was true then, not current guidance.
- **`checks.md` is the standard for checks** and their index.
- **Written for the owner**: plain American English, relative paths only, no quotes of the owner's passing remarks
  (`wiki/working_with_the_owner.md`, "Writing").
