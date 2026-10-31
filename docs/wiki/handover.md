# Handover: read this before anything else

Rewritten `2026-10-31T17:45Z` by the Gemini lead (`gemlead`) for handoff to `gemhand`.
Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
[`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md),
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), and `data/fun/`.

## State when closed

- **Full-video delivery rebuild `2026-10-31T1702Z` completed cleanly (`0` failed proofs)**:
  - `bench/rebuild_delivery.py --video fun --clip --no-pages` completed all 7 steps (`laid_verify, ledger, delivery_table, assemble, gap_check, stack, snapshot`; record at `data/2026-10-31_fun_delivery_tables/rebuild_fun_2026-10-31T1702Z.json`).
  - Assembled cut: `Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T1702Z_from_laid.mp4` (`0` failed proofs in `.check.json`) and stacked companion `Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T1702Z_from_laid_STACKED.mp4` (plus `.1fps.` companions), built from all `87` rows in `data/fun/laid.jsonl` (`sha256 b2894f97f43a24fc`).
  - **`lead_fp_6125c` (`6125–6373`) kept and laid at row 87 (`[[6125, 6366]]`)**: Owner kept `lead_fp_6125c` (`data/fun/owner_answers.jsonl:1737-1738`: `"keep this one"`, `"great!!"`). Because `bench/assemble_delivery.py::ends_of` has no tail exemption in code (`second_hfp_5889e` is likewise laid trimmed to `5889–5917`), laying `6125–6372` failed `ends` on `6367–6372` (`0 px` fade-out frames where the piece equals the source), so row 87 was laid on `[[6125, 6366]]`. Both the new `1702Z` whole-video stacked cut (`fun_cut_2026-10-31T1702Z_stacked`) and a `6367–6372` decision card (`lead_fp_6125c_tail_6367_6372`) are staged at the top of `data/fun/review/items.json` (`321` items) on `:8765`. Leave `laid.jsonl` row 87 untouched until the owner decides on `6367–6372`, then update row 87 (`added_by`, `replaces`, and `frames`) **once** via `bench/laid.py`.
- **Owner's `1419–1423` early-hair ruling recorded at `data/fun/owner_answers.jsonl:1739`**:
  - Recorded dated today (`2026-10-31T17:44:00Z`) with `relayed_by: "gemlead"` and `"frames": [1419, 1423]` (using only existing schema keys, no load name as `row`): *"Keep second's early hair on 1419-1423 even with the reference black there, and override reference_shows_no_figure on those frames, for the three-pass remake of 1419-1465 under any load letter. If it hurts 1454-1465, also show me a version starting at 1454."*
- **Global `second` mouth recipe (`closed_mouth: "sculpted"`, `open_mouth: "inner outline"`) & 11 Pass-1 Previews Completed (`DONE`) on Raw Source**:
  - Updated `data/fun/plan.json` (`subjects.second.reference` and `subjects.second.recipe`; backup at `data/fun/plan.2026-10-31_before_second_sculpted.json`) and documented the per-subject recipes in [`../masked_v2v/subject_and_drawing_recipes.md`](../masked_v2v/subject_and_drawing_recipes.md).
  - Completed (`DONE`) 11 fresh previews on the raw source (`shot_c_1419i`, `shot_c_1010c`, `shot_c_1093h`, `shot_c_1222c`, `shot_c_1306h`, `shot_c_1466c`, `shot_c_4479l`, `shot_c_5011h`, `shot_c_5533f`, `shot_c_5675f`, `shot_c_5768d`), plus 8 single-subject `lead` preview jobs built (`BUILT`) and ready to queue (`shot_c_1156f`, `1945c`, `1975c`, `2050i`, `2623e`, `3315e`, `3524e`, `5245e`).
- **Servers & Parked Worktree**:
  - ComfyUI (`:8188`) and review server (`:8765`, `data/fun/poc/serve.py`) are both running and idle (`0` background rebuild/render processes running).
  - Parked worktree `../../h3_worktrees/agy-laid-carve` (`328ca0bc`) is clean and untouched on disk.

## What not to assume

- **Do NOT render hand-patched `second_hfp_1419f` (`data/2026-10-30_second_hfp_1419f`) or single-pass `1419g` on `shot_c_1419h`**:
  - `1419–1465` is a **3-pass remake (`second` Pass 1 -> `lead` Pass 2 -> `third` Pass 3)** starting from `shot_c_1419i` on the raw source (`lead_fp_1419d` and `third_fp_1419d` are `10-15` Wave-0 pieces with `18f` `emptied_inside`). Build `second_hfp_1419g` cleanly on `shot_c_1419i`.
- **Do NOT remake kept pieces from `>= 2026-10-28` unless flagged by the owner on playback**:
  - The remake scope is strictly shots where **every kept piece predates `2026-10-28`**. Multi-subject chains `4647–4725`, `4912–4946`, and `5114–5129` stand unless flagged on playback.
- **Do NOT hand-edit `data/fun/laid.jsonl` or touch row 87 before the owner decides on `6367–6372`**:
  - Row 87 (`added_by`, `replaces`, and `frames`) will be updated **once** via `bench/laid.py` after the owner's decision on `6367–6372`.
- **Voice spans path & `video_mask.py` zoom constants**:
  - The audio voice spans file is `data/2026-10-13_fun_audio_source_0000_6394/voice_spans.json` (never `data/2026-10-13_fun_voice_spans/`). The zoom constants in [`../../video_mask.py`](../../video_mask.py) (`L459-479`) are `ZOOM_LARGEST = 2.0`, `ZOOM_MOST_ASKED = 4.0`, `ZOOM_HEAD_TARGET = 300`, and `ZOOM_SMALLEST = 1.25`.
- **Precaution on `send_message` across top-level sessions**:
  - Avoid calling `send_message` to another top-level session ID (its role in an earlier UI stream cancel is inferred, not tested; communicate via the handover files on disk).

## Tomorrow, in order

1. **Run Pass 1 remakes (`1419–1465` `second_hfp_1419g` first, then Multi-Subject Batch A Pass 1 loads) in parallel with worktree builds (`agy-laid-carve` and `valid_owner_word`)**:
   - **Builds do not hold up Pass 1 renders**: Under the repo's rule for landing between steps ([`../../AGENTS.md`](../../AGENTS.md)), Pass 1 previews/jobs/renders can run on `:8188` while the two code builds happen in worktrees (`../../h3_worktrees/agy-laid-carve` only has to land **before the first remake is laid**, and `valid_owner_word` only has to land **before signing a hard-flag override**).
   - **Build A (`../../h3_worktrees/agy-laid-carve` at `328ca0bc`, before the first lay)**:
     - Fix `_source_holds_other_subject` (`bench/laid.py:514-522`) to check `p.get("load") == other_r["load"]` (not just `subject`), fix `replaces` handling in `bench/laid.py sync`, and exclude `lead_fp_4053c` and `second_hfp_4479g` from automatic sync restoration. Run `bench/check_laid.py`, `bench/check_delivery_table.py`, `bench/check_rebuild_delivery.py`, `bench/break_checks.py`, and an independent blind read before landing by pathspec.
   - **Build B (`valid_owner_word` in [`../../bench/capture_masked_run.py`](../../bench/capture_masked_run.py) `L3951-3999, 6779-6813`, before overriding `reference_shows_no_figure` on `1419g`)**:
     - Cut a worktree (`../../h3_worktrees/<builder>-owner-word`) and tighten `valid_owner_word` so it accepts only an `owner_answers.jsonl` line (`who == "owner"`) whose `frames` match the flag's `source_frames` and whose `words` name the flag's rule (`reference_shows_no_figure`), covering both `read_owner_word` (`L3985`) and `gate_verdict` (`L3994`). Do **not** match ephemeral flag IDs (`f001`) and do **not** rely on a `run` field (`f001` in `flags.json` has no `run` field). Run `bench/check_capture_masked_run.py`, `bench/break_checks.py`, and an independent blind read before landing.
   - **Pass 1 on `1419–1465` (`second_hfp_1419g` on `shot_c_1419i`)**:
     - Build `second_hfp_1419g` on `shot_c_1419i` (`--drop 1424-1453 --no-face-on 1424-1453 --no-draw-face-on 1419-1423,1454`, `closed_mouth: "sculpted"`, `zoom_up_to: 4.0`, `--depth`), override `reference_shows_no_figure` on `1419–1423` citing `data/fun/owner_answers.jsonl:1739` (`--by <session>`), read `bench/what_h3_got.py` in full, render, land, get an independent no-verdict reading (`<load>.reading.md`), and page Pass 1 **on its own** with `lead` and `third` gaps counted right above the player. (If black reference frames `1419–1423` hurt `1454–1465`, also build and show a version starting at `1454`.) After Pass 1 is kept and laid, run Pass 2 (`lead_fp_1419e`) and Pass 3 (`third_fp_1419e`, keeping `1419–1435` unless marked `not seen`) on each freshly rebuilt cut.
2. **Owner playback review on `:8765` (`fun_cut_2026-10-31T1702Z_stacked` and `lead_fp_6125c_tail_6367_6372`)**:
   - Once the owner decides on `6367–6372` for `lead_fp_6125c`, update `data/fun/laid.jsonl` row 87 (`added_by`, `replaces`, and `frames`) **once** via `bench/laid.py`.
3. **Execute the `< 2026-10-28` Master Remake Queue in [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) (§6.2)**:
   - Run the remaining ready Pass-1 jobs on `shot_c_1010c`, `1093h`, `1222c`, `1306h`, `1466c`, `4479l`, `5011h`, `5533f`, `5675f`, `5768d` (note: `data/2026-10-31_gemlead_jobs/second_hfp_4479i.json` is already built through `text`), then the single-subject `second` (`0415n`, `0484f`, `5889e`), multi-subject `second+third` (`4602`, `4726`, `5461`) and `second+lead` (`604–1009`), and single-subject `lead` remakes (`2050–2248` now; `1039h`, `3967c`, and `2254c` at the end of the queue).

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file), [`next_steps.md`](next_steps.md), and [`decisions.md`](decisions.md) (`2026-10-31`).
  2. [`working_with_the_owner.md`](working_with_the_owner.md) and [`../comfy_notes.md`](../comfy_notes.md).
  3. [`../../data/AGENTS.md`](../../data/AGENTS.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), [`../masked_v2v/subject_and_drawing_recipes.md`](../masked_v2v/subject_and_drawing_recipes.md), and [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md).
- **Detailed session handoff, audit files, & active records**: `internal/gemini/2026-10-31_handoff.md`, `internal/claude/2026-10-31_advisory_for_next_session.md`, `data/2026-10-31_fun_delivery_tables/rebuild_fun_2026-10-31T1702Z.json`, `data/fun/laid.jsonl` (`87` rows, `sha256 b2894f97f43a24fc`), `data/fun/owner_answers.jsonl` (`1739` lines), `data/fun/review/items.json` (`321` items), `data/2026-10-31_gemlead_jobs/`, and `../../h3_worktrees/agy-laid-carve` (`328ca0bc`).
