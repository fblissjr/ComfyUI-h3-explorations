# docs/: the wiki

Loaded when a session opens a file here; the root `AGENTS.md` still governs.

- **`index.md` is the router**, and each folder has its own `index.md` naming every page in it. Find the page that
  owns a fact there before writing one; a new page gets a row in its folder's index.
- **Short pages that link down.** An overview states the answer and links to the detail; it does not repeat it. A fact
  has one home; everywhere else links to it.
- **Prose is a cache.** Where a page and the code disagree, the code is right: correct the page and log what it used to
  claim in `state/decisions.md`. A figure in prose points at the command or constant that holds it.
- **History is not guidance.** A page carries the current answer; what it used to say goes to `state/decisions.md`,
  and a dated study goes under `research/`, which is what was true then.
- **`method/checks.md` is the standard for checks** and their index.
- **Written for the owner**: plain American English, relative paths only, no quotes of the owner's passing remarks
  (`working/working_with_the_owner.md`, "Writing").
