# Folder rules and indexes: how the repo discloses itself

**In short.** A session finds its way down in four layers: root `AGENTS.md`, a folder's `AGENTS.md`, the folder's
index, then the page or file. `AGENTS.md` says **how to act** in a folder and is loaded automatically; an index says
**what is in it** and is read on purpose. They never duplicate each other: rules do not list files, and an index does
not carry rules. `bench/check_folder_rules.py` fails when a layer's pointer to the next is missing.

## The two files

| | `AGENTS.md` | the index |
|---|---|---|
| answers | what must I do or never do here, and why | what is here, one line each, and where to start |
| read | automatically: Claude Code loads it through the `CLAUDE.md` beside it (`@AGENTS.md`) when it reads a file in the folder; the Antigravity CLI only when a prompt expands from the folder | on purpose, when looking for something |
| length | a few bullets; a rule a session could work around without noticing does not belong (root `AGENTS.md`'s own test) | as long as the folder, one row a file |
| always | names its folder's index | names every file it covers |

## Which folders get an `AGENTS.md`

A folder gets one when it has a rule of its own: a file a tool reads by format, a file that is generated and never
edited, a do-not (never run, never import, never edit by hand), or an owner's rule about its content. Every top-level
folder with such a rule has one, and root `AGENTS.md` lists them by name. A subfolder with no rule of its own has
none and inherits its parent's: an empty rules file is noise a session still has to read. Progressive disclosure comes
from the indexes, not from rules files in every folder.

## Which index, by kind of folder

| folder | index | kept by |
|---|---|---|
| the root, `bench/`, `bench/results/`, `workflows/`, `archive/` | `INDEX.md`, **generated** from each file's own docstring or first heading | `bench/build_index.py`; `--check` says whether it is current |
| a top-level folder of at most 30 tracked files | a section of the root `INDEX.md` | the same generator (`SMALL_DIRECTORY`); a folder past 30 needs its own `INDEX.md` |
| a `docs/` folder | `index.md`, **hand-written**, ordered from the overview down to the detail | whoever adds a page adds its row |
| a self-contained piece (a standalone pack, an example job, an archived feature, a vendored package) | its `README.md` | its author |

Generated where the file can describe itself (code, records, graphs), hand-written where the order is the point (a
reader should meet the overview before the detail).

## Keeping them true

- **A new page or file gets its row** in the same commit: in a `docs/` folder by hand, elsewhere by running
  `bench/build_index.py`.
- **A moved or renamed file is moved with its links**: every link and path mention across the repo is rewritten in the
  same commit, and a quoted heading keeps its text. A `path:line` citation into a page whose text changed is pinned
  to the commit that still has those lines (`` `docs/x.md` lines 10-12 at `<commit>` ``).
- **Links into gitignored files stay** (`internal/`, `data/`, `coderef/`, `bench/results/**/*.json`): they exist on
  the owner's machine, and the link says where the evidence is. A failing `check_doc_links.py` on them in a clone
  without them is not a broken link.
- **Except dated session notes past seven days**: a tracked file does not cite a dated file under `internal/` (session
  logs, mailboxes, reviews, postmortems there) or a file named for a session (`<date>_mr<name>`) more than seven days
  old. Session notes are working state and are cleared; a fact worth keeping moves into a tracked record, and the
  citation points there. A tracked record is named for its subject, never for the session that wrote it.

## The checks

| check | what it fails on |
|---|---|
| `bench/check_folder_rules.py` | an `AGENTS.md` without its `CLAUDE.md` shim or without naming its index; a doc under `docs/` on no index; root `AGENTS.md`'s folder list out of step with the folders that have rules |
| `bench/check_doc_links.py` | a link or `path:line` citation that does not resolve or is out of range |
| `bench/build_wiki_index.py` | a report, not a gate: every doc no link from root `AGENTS.md` or `docs/index.md` reaches |
| `bench/build_index.py --check` | a generated index that is stale or links an untracked file |
