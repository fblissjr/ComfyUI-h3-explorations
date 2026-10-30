# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-10-30`)**: [`handover.md`](handover.md) ("Next session, in order", "State when closed", "What not to assume", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-10-30` + date router)**: [`decisions.md`](decisions.md).
- **Masked lane operative rules, live backlog, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-10-30`)

Follow [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) ("Today's queue, as ready on disk") and [`decisions.md`](decisions.md) (`2026-10-30`):
1. **Three new keeps laid today in `data/fun/laid.jsonl` (`81` rows, `sha256 2a286a70ec9726ad`)**: **`second_hfp_4912t`** (`4912–4946`, `35f`, `second` Pass 1, row 79), **`second_hfp_0278o`** (`278–293`, `16f`, `second`, row 80), and **`lead_fp_3595c`** (`3595–3671`, `77f`, `lead`, row 81).
2. **Finish active single-subject Wave 1 loads (`zoom_up_to = 4.0`)**:
   - **`4227–4271` (`lead`)**: `lead_fp_4227d` (`4227–4257`, `4.00x`) and `lead_fp_4243` (`4243–4271`, `3.84x`) rendered and staged on `:8765` for owner review.
   - **`5085–5113` (`lead`)**: `lead_fp_5085d` (`29f + 10` held pad = `39f` window, `real_after = 0`, `4.00x` render zoom) rendering/paging via `gemgo`.
   - **`6105–6394` (`lead`)**: `lead_fp_6125` (`6125–6189`, `65f`, `zoom_up_to = 4.0`) and remaining contiguous sub-loads on `shot_c_6105g`, followed by `3110–3166` (`57f` tail of `2823–3166`).
   - **`14–167` (`second`, `second_hfp_0014j`)**: **Hold awaiting owner decision** on closed-mouth drawing form (`second_hfp_0014i` built fresh `second_0014_0167_e` with `closed_mouth: "outlined"` outer lip ring + center line; decide whether to add a line-only `closed_mouth` form in `bench/draw_face_points.py` + `bench/load_text.py` or use `closed_mouth: "sculpted"`).
3. **Complete the whole-video delivery rebuild (`bench/rebuild_delivery.py --video fun --clip`)**:
   - Today's `1514Z` run (`80` rows) stopped after `assemble` and `gap_check` (before `stack`, `pages`, `snapshot`, and `rebuild_fun_*.json`), and row 81 (`lead_fp_3595c`) plus Wave 1 keeps are now laid. Run a full `bench/rebuild_delivery.py --video fun --clip` so `:8765` serves the current whole-video cut and Pass-2/3 previews read the registered composite cut.
4. **Execute Multi-Subject Pass-2 / Pass-3 Queue on the Rebuilt Cut**:
   - **`4912–4946` Pass 2 (`third`, `35f`)**: Now unblocked by kept `second_hfp_4912t` (row 79). Run fresh preview `shot_c_4912j` (`--source <post-4912t_cut> --for third --remake --queue`) and build/render `third_fp_4912...` (`region: "face"`, `parts: ["face_and_neck"]`, `source_facial_hair: true`, `margin = 16`, `zoom_up_to = 4.0`, no beard per `data/fun/owner_answers.jsonl:1456`).
   - Remaining multi-subject Pass-2/3 spans on the rebuilt cut: `1419–1465` (`47f`), `1536–1574` (`39f`), `1010–1038` (`29f`), and `604–1009` (`406f`).

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
