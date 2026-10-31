# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-10-31`)**: [`handover.md`](handover.md) ("State when closed", "What not to assume", "Tomorrow, in order", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-10-30` + date router)**: [`decisions.md`](decisions.md).
- **Masked lane operative rules, live backlog, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-10-31`)

Follow [`handover.md`](handover.md) ("Tomorrow, in order"), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), and [`decisions.md`](decisions.md) (`2026-10-30`):
1. **Remake `second_hfp_1419f` cleanly through the chain as `second_hfp_1419g` (`1419–1465`, `second` Pass 3)**:
   - Run `bench/shot_to_load.py job --remake` with `--drop 1424-1453 --no-face-on 1424-1453 --no-draw-face-on 1419-1423`, then `bench/shot_to_load.py reference`, `bench/shot_to_load.py text`, and `bench/load_chain.py first` (do not render the hand-patched `1419f`).
2. **Tighten `valid_owner_word` in [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) before rendering `second_hfp_1419g`**:
   - Accept only an owner JSONL line (`who == "owner"`) that names the load and the flag's rule, record the owner's Option A for `1419–1423` in `data/fun/owner_answers.jsonl`, sign the override `--by <session>` on the owner's word, update the docstring and check wording, verify with `bench/break_checks.py` and an independent blind read, land by pathspec, and then take `second_hfp_1419g` through `render`, 3-pass assembly, `reading.md`, and `page`.
3. **Owner review on `:8765` (`lead_fp_6125c` on `6125–6373`; `second_hfp_4479g` vs `4479h` on `4479–4509`) before `bench/laid.py sync` and the full rebuild**:
   - Watch `6232–6246`, `6346–6352`, and around `6372` on `lead_fp_6125c` (on the main `:8765` page), and click keep on whichever of `second_hfp_4479g` or `second_hfp_4479h` is wanted in the cut.
4. **Finish, test, blind-read, and land `../../h3_worktrees/agy-laid-carve` (`328ca0bc`), then run `bench/laid.py sync --video fun` and `bench/rebuild_delivery.py --video fun --clip --share h264 --share-max-mb 95`**:
   - Wire [`../../data/fun/poc/serve.py`](../../data/fun/poc/serve.py) so a keep click triggers `bench/laid.py sync`, add unit tests in `bench/check_laid.py`, `bench/check_delivery_table.py`, and `bench/check_rebuild_delivery.py`, pass `bench/break_checks.py` and an independent blind read, land on `main`, sync `data/fun/laid.jsonl`, rebuild the whole-video cut (`--clip` required after `24eaa7f1`), and compare the new cut frame-by-frame against `1943Z` (reporting differences against `CHANGE` in [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) and timing `-threads 12` vs `1943Z`'s `threads=34`).
5. **Continue remaining multi-subject Pass-2/3 spans (`1010–1038`, `604–1009`)** on the rebuilt cut.

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
