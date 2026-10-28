# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-10-28`)**: [`handover.md`](handover.md) ("Tomorrow, in order", "State when closed", "What not to assume", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-10-28` + date router)**: [`decisions.md`](decisions.md).
- **Masked lane operative rules, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-10-28`)

Follow [`handover.md`](handover.md)'s **"Tomorrow, in order"** (rewritten at the `2026-10-27` close) and the `2026-10-28` corrections in [`decisions.md`](decisions.md):
1. Rebuild the whole-video delivery table and morning composite cut (`bench/rebuild_delivery.py --video fun --clip`) so the cut includes `second_hfp_4647i` (`laid.jsonl` row `70`), `third_fp_5114e` (row `71`), and the `187dc9fd` `restore=` pin fixes, and copy the resulting `.mp4` and `.json` sidecars to `$H3_INPUT_DIR` (`--source`).
2. Unblock and render `second_hfp_4912m` (`4912–4946`, Pass 1 `second`) after an independent non-maker `looked` pass, overriding post-`on-plan` `f002` (`key: 7eb6b893b25b6aff`), and running `flags-read`.
3. Queue Pass 3 `lead` preview on `4647–4725` (`shot_c_4647m`) on the new morning cut behind `4912m`'s render, and after `4912m` is kept and laid, rebuild the cut and render Pass 2 `third` on `4912–4946` (`margin = 16`, face only, no beard).
4. Clear the `lead` single-subject queue by frames owed (`6105–6394` [290f], `2254–2598` [345f laid on `2271–2598`], `3759–3908` [150f], `516–603` [88f], `3595–3671`, `5044–5113`, then `4774, 3476, 3240, 4227` and `1419–1465` `second`).

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
