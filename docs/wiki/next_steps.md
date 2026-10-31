# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-10-31`)**: [`handover.md`](handover.md) ("State when closed", "What not to assume", "Tomorrow, in order", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-10-30` + date router)**: [`decisions.md`](decisions.md).
- **Masked lane operative rules, live backlog, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-10-31`)

Follow [`handover.md`](handover.md) ("Tomorrow, in order"), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) (§6), and [`decisions.md`](decisions.md) (`2026-10-31`):
1. **Run Pass 1 remakes (`1419–1465` `second_hfp_1419g` first, then Multi-Subject Batch A Pass 1 loads) in parallel with worktree code builds (`agy-laid-carve` and `valid_owner_word`)**:
   - **Pass 1 renders do not wait on the two builds**: Under the repo's rule for landing between steps ([`../../AGENTS.md`](../../AGENTS.md)), `../../h3_worktrees/agy-laid-carve` (`328ca0bc`, with `_source_holds_other_subject` load-name matching and `replaces` handling fixed, and `lead_fp_4053c`/`second_hfp_4479g` excluded) only has to land **before the first remake is laid**, and tightening `valid_owner_word` in [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) (`L3951-3999, 6779-6813`: match `frames` against `source_frames` and require `reference_shows_no_figure` in `words` across both `read_owner_word` and `gate_verdict`, without matching flag ID `f001` or a `run` key) only has to land **before signing a hard-flag override**.
   - **Three-pass remake on `1419–1465` (`second` Pass 1 -> `lead` Pass 2 -> `third` Pass 3)**: Build `second_hfp_1419g` cleanly on `shot_c_1419i` (`--drop 1424-1453 --no-face-on 1424-1453 --no-draw-face-on 1419-1423,1454`, do not render hand-patched `1419f`), sign the `reference_shows_no_figure` override on `1419–1423` citing the owner's `2026-10-31` line in `data/fun/owner_answers.jsonl`, render and page Pass 1 on its own (with `lead`/`third` gaps counted above the player), and after Pass 1 is kept and laid, run Pass 2 (`lead_fp_1419e`) and Pass 3 (`third_fp_1419e`, keeping `1419–1435` unless marked `not seen`).
2. **Owner playback review on `:8765` (`1702Z` whole-video stacked cut + `lead_fp_6125c` tail `6367–6372`)**:
   - Inspect the `1702Z` whole-video stacked cut on `:8765` (including the 12-load `zoom_up_to: 2.0` spot-check list in [`decisions.md`](decisions.md)) and the `6367–6372` card for `lead_fp_6125c`. Leave `data/fun/laid.jsonl` row 87 untouched until the owner decides on `6367–6372`, then update row 87 (`added_by`, `replaces`, and `frames`) **once** via `bench/laid.py`.
3. **Continue the `< 2026-10-28` Master Remake Queue in [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) (§6.2)**:
   - Remake only shots where **every kept piece predates `2026-10-28`** (`4647–4725`, `4912–4946`, and `5114–5129` stand unless flagged on playback; `2050–2248` runs now and `1039h`, `3967c`, and `2254c` sit at the end of the queue).

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
