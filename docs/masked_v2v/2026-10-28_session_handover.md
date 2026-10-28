# 2026-10-28 Session Handover & Tomorrow's Operating Guide

**Session ID:** `ebdd0a24-32bd-4ef6-8f72-73bfe7927103` (`gemlead`)  
**Advisory Peer:** Claude (`mrfizz`, notes in `internal/claude/2026-10-28_mrfizz/note_to_gemlead_1938Z.md`)  
**Base Commit (`main`):** `606ee0ea` / `4eeb54cb` / `e88857cf`  
**Session Postmortem (Forensic Transcript Audit):** `internal/gemini/2026-10-28_session_ebdd0a24_subagent_postmortem.md` (gitignored under `internal/gemini/`)  
**Canonical Render & Remake Backlog:** [fun_render_backlog.md](fun_render_backlog.md)

---

## 1. What Was Shipped & Verified on `2026-10-28`

### 1.1 Code & Tests Landed on `main` (`c8dee328`, `c60c6474`, `e88857cf`)
1. **Rejecting `gem/perception-lhf` (`16b0805c`) & Landing Clean Crown-Join in `bench/head_tube_parts.py` (`c8dee328`, `e88857cf`)**:
   - Audited the `gem/perception-lhf` worktree branch (`16b0805c`) on `4912` (`second_hfp_4912m`) and proved its 3-line gate caused a `+6,138 px` 1-px speck explosion on frame `4926` and a `+1,858 px` white-hat bleed on frame `4946`. Rejected `gem/perception-lhf` without merging.
   - Implemented the clean 2-gate upward crown join directly in [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py) (`c8dee328`, lines `135–148`, `2090–2116`):
     - `_SPECK_MIN_PX = 64`: filters out `< 64 px` stray SAM `head_track` specks before dilation (eliminating the `4926` 1-px speck explosion).
     - `_MIN_CROWN_ABOVE_HY0_PX = 256`: on head-and-hair loads (`wants_hair=True`, e.g. `second`) with a head track, expands `head_track_bool` upward into connected components of `parts_bool & raw_track_bool` (`y <= hy1`) that overlap the main `head_track` body by `>= 64 px` and have `>= 256 px` above `hy0`, without growing `tube_info["tube"]`.
2. **Pre-Render `check_head_tube_parts.py` Gate & Unit Tests (`c60c6474`)**:
   - Added [`bench/check_head_tube_parts.py`](../../bench/check_head_tube_parts.py) and unit tests (`tests/test_head_tube_parts.py`), wired into [`bench/load_chain.py`](../../bench/load_chain.py) `first` so any load where `parts_held.mkv` starves Sapiens `Face_Neck` or `Hair` fails loudly at `first` before GPU render time is spent.
3. **Parking `4912–4946` (`second_hfp_4912m/n/o`) After Isolating Bystander Hair Bleed**:
   - Proved on `second_hfp_4912o` that even with `100%` of `second`'s crown in `parts_held.mkv`, `5,000–52,000 px` of near-black (`L=7`) bystander hair inside the `2.46x` zoom window on `31/35` frames bleeds into `second`'s blond hair via H3 self-attention. Parked `4912–4946` in Tier 4 (`fun_render_backlog.md` Seq `4.2`) until an out-of-subject hair keep-out / luminance suppression mechanism is built.

### 1.2 Renders Landed & Kept by the Owner (`2026-10-28`)
| Span | Load(s) | Subject | Key Settings / Resolution | Verdict (`data/fun/owner_answers.jsonl`) |
|---|---|---|---|---|
| `4647–4725` (`79f`) | `lead_fp_4647d` | `lead` | Pass 3 over `third_fp_4647e` (`10-24`) + `second_hfp_4647i` (`10-27`) on `shot_c_4647m`; `--no-head-track` (since `shot_c_4647m` seeded `head_track` on `4666` back-of-head hair dome), `--drop 4647-4680 --no-face-on 4647-4680`, `--least 1024` (preventing `starting_floor` at `L2736` from measuring undropped `4651–4680` slivers before `drop=` at `L2849`), `--depth` | **KEPT** (`19:27Z`, `"looks good!"`, `laid.jsonl` row 72) |
| `516–558` (`43f`) | `lead_fp_0516f` | `lead` | Sub-load 1 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`3.88x` H3 render zoom** (`[348, 290, 612, 488]`), `5.72x` `<Video 1>` crop (`78.8 px` median face) | **KEPT** (`21:39Z`) |
| `559–578` (`20f`) | `lead_fp_0559` | `lead` | Sub-load 2 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.51x` H3 render zoom** (`[472, 240, 880, 546]`), `3.68x` `<Video 1>` crop (`85.6 px` median face) | **KEPT** (`21:39Z`) |
| `579–599` (`21f`) | `lead_fp_0579` | `lead` | Sub-load 3 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.67x` H3 render zoom** (`[640, 264, 1024, 552]`), `3.89x` `<Video 1>` crop, `--hold-part 597-597` | **KEPT** (`21:39Z`) |

### 1.3 Kept Renders Re-Verified & Closed Without Remake
- **`3967–4052` (`lead_fp_3967c`)**, **`3060–3109` (`lead_fp_3049c`)**, **`168–184` (`lead_fp_0168d`)**: Verified good by the owner (`20:42Z`, `data/fun/owner_answers.jsonl:1662`).
- **`5935–6104` (`lead_fp_5935d`)**: Delivery check flags `d122` (`piece_changes_away_from_its_subject`) and `d123` (`piece_changes_across_a_cut` at `6050`, `6103`) were inspected frame-by-frame across all 12 flagged frames (`5982, 6035, 6036, 6038, 6039, 6048, 6049, 6077, 6080, 6089, 6101, 6102`) by the owner, Gemini, and Claude (`mrfizz` items 13 & 15). `d123` is a foreground dancer lens wipe right at the `40.0` threshold; `d122` is invisible motion-blur feather around her hair and pom-pom. **Stands as kept.**

---

## 2. Active & Remaining Render Queue

See [fun_render_backlog.md](fun_render_backlog.md) for the complete, dependency-ordered table across all phases (approved by the owner on `2026-10-28T21:42Z`):

1. **Phase 0 (Active Unrendered Spans)**:
   - `3759–3908` (`lead_fp_3759c`, `150f` continuous load on `shot_c_3759f`, `--drop 3851-3857` back-of-head spin).
   - `2254–2598` (`lead_fp_2254c`, `345f` on `shot_c_2050h`, `zoom_up_to: 4.0`, 10 interior `not seen` spans dropped via `--drop` and `--no-face-on`).
   - `6105–6394` (`lead_fp_6105e`, `290f` on `shot_c_6105f`, non-lead cuts dropped).
2. **Tier 1 (Kept Pieces with Visible Faults or `emptied_inside` Dropouts)**:
   - `3909–3966` (`lead_fp_3909c`, `58f`, Seq 1.0): Prompt fix (`"<Subject 1> wears white sunglasses."`) on fresh preview `shot_c_3909e` so H3 draws her white sunglasses through the `3915–3921` and `3928–3931` head tosses where Sapiens has `0 px` of `Eyeglass`.
   - Single-subject `emptied_inside` remakes: `294–414` (`lead_fp_0294d`), `4272–4309` (`lead_fp_4272c`), `4053–4126` (`lead_fp_4053e`), `1134–1155` (`lead_fp_1134c`), `438–483` (`lead_fp_0438c`).
   - Multi-subject crossing shots: `604–1009` (`lead_fp_0604j` as Pass 2 over kept `second_hfp_0604h`), `1419–1465` (`lead_fp_1419g` / `second_hfp_1419c`).
3. **Tier 1B (Zero-GPU Data / Assembly Fixes)**:
   - Next whole-video `assemble_delivery.py` build laying all newly kept Phase 0 + Tier 1 pieces.
4. **Tier 2 (`d120` `piece_leaves_some_of_the_source` Edge Dropouts)** & **Tier 3/4 (Pre-Cutoff Upgrades)**.

---

## 3. Hard Operational Safeguards From Today's Postmortem

Full forensic details and transcript citations are in `internal/gemini/2026-10-28_session_ebdd0a24_subagent_postmortem.md`. Every session picking up this handover must enforce these six rules:

1. **Re-Read Disk Records (`git log`, `laid.jsonl`, `owner_answers.jsonl`, `job.json`) After Context Compaction**:
   - After 13 context compactions in one day, relying on compacted summary memory caused the first draft of this handover to conflate rejected worktree `gem/perception-lhf` (`16b0805c`) with `c8dee328` and to misstate `lead_fp_4647d`'s flags. Always run `git log -n 10 --oneline` and inspect `data/fun/laid.jsonl` and `data/2026-10-28_gemlead_jobs/*.json` before writing handover or backlog prose (`RULE[user_global]`).
2. **Check `manifest.json` `zoom.scale` and `<Video 1>` `lash_gap` Before Running `render` on Small Faces**:
   - On `lead_fp_0516e` (`516–599`), the full `84f` travel box (`496x292`) fell below `MIN_USEFUL_ZOOM = 1.25` (`1.08x -> 1.0x`), causing H3 to fall back to `1.0x` whole-frame while `<Video 1>` (`2.165x`) had `83/84` frames under the `1.6 px` lash-gap threshold (white shut-eye ovals).
   - **Rule**: Whenever a subject's canvas face height is `< 35 px`, verify in `data/<date>_<load>/manifest.json` that `zoom.scale >= 1.25` (ideally `>= 2.5x`) and in `pose_pass.out` that `<Video 1>` lash lines are `>= 1.6 px` (`0` frames under `1.6 px`). Split long travel shots at natural movement boundaries and set `zoom_up_to: 4.0` in `data/fun/plan.json` (`shot_overrides`).
3. **Pass `--least` Scaled by Zoom When `drop=` Removes Tiny Entry/Exit Slivers (`starting_floor` Ordering)**:
   - In `bench/capture_masked_run.py`, `starting_floor` (`L2736`) runs before `drop=` (`L2849`). On `lead_fp_4647c`, undropped back-of-head slivers on `4651–4680` caused `starting_floor` to compute `1.0x` (`4096 px`) instead of `2.0x` (`1024 px`), starving `4681–4725`. Whenever `--drop` removes part-sliver entry/exit frames on a zoomed load, pass `--least 1024` (or `4096 / zoom^2`) explicitly.
4. **Preserve Owner Ruling Holds (`--hold-part`) When Rebuilding Without `--carry-words-from`**:
   - When `--carry-words-from` fails on an obsolete sheet path, inspect `job.carried_words_from` or `overridden_by_word_frames` in the prior capture's `part_grades.json` before passing `--no-carry-words`, and re-supply any owner ruling holds (`--hold-part FIRST-LAST=WHY`) explicitly.
5. **Verify Preview Creation Date & Commit Provenance on Disk**:
   - Never assume a preview named `shot_c_<N>e` is post-cutoff from its letter alone. Always check `stat` / `job.json` date (`>= 2026-10-28` for `c8dee328` crown-join in `head_tube_parts.py`, `>= 2026-10-27` for `SAM 2.1` `head_track` and `strangers.py`).
6. **`page_spec.json` Schema Check (`look` Must Be a String) & Visual Contact-Sheet Inspection Before `page`**:
   - `data/fun/poc/index.html:494` calls `(it.look || '').trim()`. Always pass `"look"` in `page_spec.json` as a plain string (`str`), never a JSON list, and always inspect a rendered contact sheet (`ORIG` vs `RENDER`) before running `load_chain.py page`.
