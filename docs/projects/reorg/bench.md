# The bench: what to do with 474 scripts in one folder

**In short.** Don't sort `bench/` into subfolders: a script's path is an identifier that imports, the check runner,
arm manifests, records and docs all depend on. **Prune it instead.** Archive what nothing current uses, one lane at a
time from a list the owner approves, and let the generated `bench/INDEX.md` and the inventory do the organizing. The
first list is the 79 scripts that no current subject page names and that nothing calls or imports.

## Why not subfolders

- **Sibling imports.** A script runs as `python bench/x.py`, so `bench/` is `sys.path[0]` and 115 scripts are
  imported by others by bare name (`from shot_table import ...`, `from _lib import ...`). A script moved into a
  subfolder breaks every one of its importers, or needs a path shim in each.
- **Discovery by name.** `bench/run_checks.py` finds checks by `bench/check_*.py`, and `--changed` maps a changed file
  to the checks whose source names it.
- **Named callers.** Arm manifests, graphs' recipes, `bench/results/` records and every doc name scripts as
  `bench/<name>.py`. A record that names a path is history and is not rewritten, so a moved script leaves records
  pointing at nothing.
- **What a subfolder would buy, the index already gives**: `bench/INDEX.md` groups scripts by filename prefix
  (`check_`, `measure_`, `analyze_`, ...), and `bench/bench_inventory.py` adds the lane.

## The data

`python bench/bench_inventory.py` (`--json` for a tool, `--only <status>` to filter) gives each script's dependents
and a status. Run it before and after every step; these figures are from 2026-10-19:

| status | scripts | what it means |
|---|---|---|
| check | 157 | `check_*.py`, run by `bench/run_checks.py` |
| lib | 115 | imported by another tracked script |
| wired | 108 | named by tracked code, a graph, an arm manifest or a shell file |
| documented | 76 | named only in prose |
| unreferenced | 18 | named by nothing tracked |

The `topics` column says which subject folders' pages name a script. The 79 scripts that are `documented` or
`unreferenced` **and** on no subject page are the first candidates: only dated records, the decisions log or nothing
name them. 29 of them have had no real change since September (the inventory skips the commits in
[`mechanical_commits.txt`](mechanical_commits.txt) when it dates a script).

What the inventory cannot see: a name built at run time, a caller in `internal/` or a session's scratch, and a tool
the owner runs by hand. `unreferenced` means nothing tracked names it, never "unused".

## The steps

1. **A lane's list.** For one lane (closed lanes first: `docs/state/roadmap.md`, "Closed lanes"), list its candidate
   scripts with `bench_inventory.py --json`, each with what it measured and the record it produced. The owner marks
   each keep or archive. The list and the marks live in `local/` until approved, then the approved list goes in this
   page.
2. **Archive what was marked.** For each script: `git mv` it to `archive/bench/` with one line in its docstring saying
   when and why; retire its check, if it has one, to `docs/method/checks.md`'s retired rows; repoint the current pages
   that name it (records keep the path they had); run `bench/build_index.py`; and confirm the inventory, the doc-link
   check and `bench/run_checks.py` show nothing new red. One commit per lane.
3. **Checks of archived tools.** A check whose only subject went to the archive is retired with it, never kept to
   test retired code.
4. **Repeat** lane by lane; then the `documented` scripts a subject page names (are they still the tool that page
   should point at?), then the `lib` scripts imported by only archived ones.

## The first candidate

`bench/analyze_pdd_unmerge_curve.py`: an aggregation over two 2026-08-31 records, of the merged PDD path that has been
the control arm since 2026-09-26. Nothing imports, calls or checks it; `docs/distills/pdd.md` names it in prose for a
figure, and dated records name it. Archiving it means: `git mv` to `archive/bench/`, repoint the one line in `pdd.md`,
rebuild the index. Its stale paths into `bench/results/` were fixed to `data/results/` on 2026-10-19 so the archived
copy says where its inputs are.

## What waits on the owner

- The keep or archive marks on each lane's list.
- Whether a closed lane's checks are retired with its tools (step 3 says yes).
