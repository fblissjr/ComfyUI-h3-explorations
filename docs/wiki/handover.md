# Handover: read this before anything else

Rewritten 2026-10-27 at the close of the 2026-10-27 working day by the Gemini lead (`gemlead`). Where this page and a
record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
[`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md),
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), and `data/fun/`.

## State when closed

- **Landed on `main` today (`8f11eed7..1acffe40`, 28 commits on 2026-10-27)**:
  - **Morning tracker, strangers & `head_tube` builds (`426638a6`, `1e91de26`, `6b0ea9a6`, `39e1f29a`/`68d81aff`)**:
    1. `bench/strangers.py::run` (`426638a6`): includes unowned SAM 3.1 person detections (`people.npz`) where whole-frame `Sapiens2` finds nobody.
    2. `bench/capture_masked_run.py` & `bench/head_tube_parts.py` (`1e91de26`): passes `head_tube`'s tube to `grade_holds`, exempts frames inside `head_track` from `doubted`, and syncs `part_grades.json` after `her_hair_in_hole`.
    3. `subject_track.py` & `bench/shot_to_load.py` (`6b0ea9a6`): defaults `head_mode` to `"per stretch"` (`HEAD_PER_STRETCH`) and requires `--people` verification on `bench/cast_file.py box`.
    4. `bench/capture_masked_run.py` (`39e1f29a`/`68d81aff`): excludes depth-preserved foreground keep-outs from `keep_out_inside_own_track`.
  - **Delivery table `loads`-pinned `restore` & `--help` / `frame_table.py` (`187dc9fd`, `bbe13fea`, `c1090081`, `1acffe40`)**:
    1. `bench/delivery_table.py::own_restores` (`L308–355`), `bench/shot_coverage.py::load_plan` (`L202–205`), and `bench/shot_to_load.py::load_job` (`L3496–3500`) support `"loads": ["<load_name>", ...]` on `plan["subjects"][label]["restore"]` entries (`187dc9fd`).
    2. Added `--help` epilogs and module docstrings on `restore=` to `bench/assemble_delivery.py` and `bench/delivery_table.py`, and added `"saved part kept, restored by her_hair_in_hole"` (`1e91de26`) to `bench/frame_table.py::CLOSED_DID_VOCAB` (`1acffe40`).
  - **Code map & doc modularization (`d2c2b850..bda20310`, `09221125`)**: sharded oversized docs into `< 40 KB` routers (`15470307`, `3886db1f`), added `bench/masked_code_map.py` AST CLI (`--help`, `--callers`, `--callees`, `--data-chain`, `--step-chain`, `--status`, `--stamp-narrative`), and documented the ready-to-test `gem/perception-lhf` branch in `../../h3_worktrees/gem-perception-lhf` ([`../masked_v2v/2026-10-27_perception_lhf_branch.md`](../masked_v2v/2026-10-27_perception_lhf_branch.md)).
- **Renders kept and laid today (`data/fun/owner_answers.jsonl` & `data/fun/laid.jsonl`)**:
  - **`second_hfp_4647i` (`4647–4725`)**: kept by the owner (`data/fun/owner_answers.jsonl:1653`, `"keep this one"`, `"Looks good!"`) and laid as row `70` in `data/fun/laid.jsonl`.
  - **`third_fp_5114e` (`5114–5129`)**: kept by the owner (`data/fun/owner_answers.jsonl:1655-1656`, `"keep this one"`, `"Looks good!"`) and laid as row `71` in `data/fun/laid.jsonl`.
  - `bench/laid.py keeps`: `71` laid rows across `53` shots, `0` missing keeps, `0` laid marked wrong.
- **Servers (`:8188` and `:8765`)**: both stopped cleanly at close of day.

## What not to assume

- **Self-accessory `restore=<self>.<Class>` is ONLY for Regime 2 (where H3 omitted an accessory inside the mask), never for Regime 1 (where the render already has it) or Regime 3 (outside `matte`)**:
  - Proven by today's A/B join tests (`bench/assemble_delivery.py` on `3909–4052` and `2823–3109`, documented in [`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md) §1.2–1.3): `lead_fp_3967c` (`3967–4052`), `lead_fp_2823b` (`2823–3097`), and `lead_fp_3098b` (`3098–3109`) already carry clean single white sunglasses (and hands/earrings on `2823b`) inside their rendered `.mp4`s. Applying `restore=lead.Eyeglass` on top of them pastes a second face-shifted copy from the source plate and creates double rims/hooks.
  - `data/fun/plan.json` (`subjects.lead.restore`) now holds only the two `lead` loads where H3 omitted the accessory: `"loads": ["lead_fp_3909b"]` (`3909–3966`, `Eyeglass`) and `"loads": ["lead_fp_0168c"]` (`168–184`, `Apparel@head`).
- **Proven recipe for distant/occluded face+hair loads (`second_hfp_4647i`, documented in [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md) §4.1)**:
  - Pass `--part-least-px second=400:...` on `shot_to_load.py job` (or `--least 400` on `load_chain.py first`) so `grade_holds` does not empty emerging small head/hair masks under `1,024 px`, and pass `--no-draw-face-on` on occluded/profile frames where 3D-pose face points drift off the grey head.

## Tomorrow, in order

1. **Finish `second_hfp_4912m` (`4912–4946`, Pass 2 on `second`)**:
   - All prep (`shot_c_4912i` preview, `reference`, `text`, `first`, `verify`, `plan`, and `on-plan` masks/strips in `data/2026-10-27_second_hfp_4912m` and `data/2026-10-26_gemlead_jobs/second_hfp_4912m.json`) is already completed on disk.
   - Start `:8188` and `:8765`, override `f002` (`region_on_a_frame_with_no_mask`, the known occluded frame `4924` where `second` is hidden behind the dancer in blue), run `bench/load_chain.py table data/2026-10-26_gemlead_jobs/second_hfp_4912m.json --by gemlead` (which now passes with `1acffe40`), read the prompt in full (`bench/what_h3_got.py`), record `looked` + `flags_before_render.py`, and run `render -> land -> page`.
2. **Pass 3 on `4647–4725` (`lead`) and remaining single-subject `lead` spans**:
   - Once `second_hfp_4647i` (`4647–4725`, Pass 2) and `second_hfp_4912m` (`4912–4946`, Pass 2) are laid, rebuild the working composite cut from `data/fun/delivery_table_laid.txt` and run `lead`'s Pass 3 on `4647–4725`, plus the remaining single-subject `lead` spans (`6105–6394`, `3759–3908`, `516–603`, `2254–2598` laid on `2271–2598`, and the `604` gaps).
3. **Optional evaluation of `gem/perception-lhf` (`../../h3_worktrees/gem-perception-lhf`)**:
   - Follow the test checklist in [`../masked_v2v/2026-10-27_perception_lhf_branch.md`](../masked_v2v/2026-10-27_perception_lhf_branch.md) when ready.

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file) and [`decisions.md`](decisions.md) (`2026-10-27`).
  2. [`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md) (`restore=` rules, `2823b`/`3098b`/`3967c` join-test audit, and long-term clip-level restore design).
  3. [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md) (`§4.1`: proven `second_hfp_4647i` recipe) and [`../masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md).
  4. [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md) and `<python> bench/masked_code_map.py --help`.
- **Laid ledger & delivery table**: `data/fun/laid.jsonl` (`71` rows across `53` shots) and `data/fun/delivery_table_laid.txt` (`71` piece rows, with `restore=` only on `lead_fp_0168c` and `lead_fp_3909b`).
