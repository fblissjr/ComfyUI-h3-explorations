# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-10-29`)**: [`handover.md`](handover.md) ("Tomorrow, in order", "State when closed", "What not to assume", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-10-29` + date router)**: [`decisions.md`](decisions.md).
- **Masked lane operative rules, live backlog, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-10-29`)

Follow [`handover.md`](handover.md)'s **"Tomorrow, in order"** (rewritten at the `2026-10-28` close) and [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) ("Today's queue, as ready on disk"):
1. Rebuild the whole-video delivery table and composite cut (`bench/rebuild_delivery.py --video fun --clip`) with all `77` rows of `data/fun/laid.jsonl` (including the `8` keeps of `2026-10-28`, rows `70–77`) and stage the cut on `:8765`.
2. Build and render `lead_fp_4053d` (`4053–4091`, `39f`) on `shot_c_4053e` as the first live reference under `dc07ccee` (`1.958x`), reading `_sheet.jpg` before `first`.
3. Preview `shot_c_5044f` and build/render `lead_fp_5044b` (`5044–5113`, `70f`) under `dc07ccee` (`2.798x`, `0` edge frames).
4. Execute the remaining single-subject `lead` queue (`4774–4841`, `3476–3523`, `4227–4271`, `3240–3276`, `3595–3671`, `3110–3166`, and `6105–6394` once the 9 `who_is_who` rows are answered) and `4912–4946` (`second_hfp_4912o` paged + CPU neighbor-hair box check).

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
