# docs/method/: how the repo decides something is true

Loaded when a session opens a file here; `docs/AGENTS.md` and the root `AGENTS.md` still govern.

- **`index.md` lists every page here.** Read it first; a new page gets a row.
- **`checks.md` is read by tools**: its index of checks by `bench/check_doc_inventory.py`, its `doc-link-absent` block
  by `bench/check_doc_links.py`. **`evidence.md`'s `retraction-consumers` block** is read by
  `bench/check_retraction_consumers.py`. Keep those blocks' formats.
- **Pre-registrations are written before the result** (`docs/research/2026-08-14_sol_bench_plan.md`): a prediction is never edited after the run;
  the outcome goes beside it.
