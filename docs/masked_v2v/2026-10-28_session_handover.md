# 2026-10-28 Session Handover & Tomorrow's Operating Guide

**Session ID:** `ebdd0a24-32bd-4ef6-8f72-73bfe7927103` (`gemlead`)  
**Advisory Peer:** Claude (`mrfizz`, notes in `internal/claude/2026-10-28_mrfizz/note_to_gemlead_1938Z.md`)  
**Base Commit (`main`):** `e88857cf` (`Merge branch 'gem/perception-lhf' into main`)  
**Session Postmortem (Forensic Transcript Audit):** `internal/gemini/2026-10-28_session_ebdd0a24_subagent_postmortem.md` (gitignored under `internal/gemini/`)  
**Canonical Render & Remake Backlog:** [fun_render_backlog.md](fun_render_backlog.md)

---

## 1. What Was Shipped & Verified on `2026-10-28`

### 1.1 Code & Architecture Merged to `main` (`e88857cf`, 9 Commits, 251 Tests Passing)
1. **Dynamic Preview Seed Discovery (`bench/find_subject_appearances.py`, `bench/preview_for.py`)**:
   - Replaces static preview seeds with frame-accurate re-entry seed discovery (`--discover-appearances`), preventing missed subject re-entries after `>12` absent frames (`604–1009`, `2050–2622`, `3759–3908`, `516–603`).
2. **Whole-Video Self-Accessory Restore (`bench/assemble_delivery.py`)**:
   - Restores `Eyeglass` and `Apparel@head` from the original film over laid renders across the entire video (`0–6394`), verified by the owner on `fun_0000_6394_joined_2026-10-28T1505Z_from_laid` (`168–184` hat, `3060–3109` sunglasses, `3910–3966` sunglasses, `3967–4052` sunglasses).
3. **Preview-Time Crown-Join into `parts.mkv` (`nodes/video_mask.py`, `bench/shot_to_load.py`)**:
   - Joins top-of-head `head_track` (`SAM 2.1`) crown pixels directly into `parts.mkv` above the highest `Face_Neck` / `Hair` row (`CROWN_JOIN_BAND_FRAC = 0.35`, `CROWN_JOIN_CAP_PX = 32`) whenever `head_track.mkv` is present and `--no-head-track` is not set.
   - Eliminated the `12–16 px` dark crown gap on `4647–4725` (`lead_fp_4647d`, kept by the owner).

### 1.2 Renders Landed & Kept by the Owner (`2026-10-28`)
| Span | Load(s) | Subject | Key Settings / Resolution | Verdict (`data/fun/owner_answers.jsonl`) |
|---|---|---|---|---|
| `4647–4725` (`79f`) | `lead_fp_4647d` | `lead` | Pass 3 over `third_fp_4647e` (`10-24`) + `second_hfp_4647d` (`10-27`); `shot_c_4647m` with crown-join; `--no-head-on 4651-4680` (`hold`) | **KEPT** (`19:27Z`, `"looks good!"`) |
| `4912–4982` (`71f`) | `lead_fp_4912p` | `lead` | `shot_c_4912g` with crown-join; `--eyes-shut-on 4959-4968` (`--eyes-shut-form closed`, fixing `4912o`'s double-iris on `4964–4965`); `--hold-part 4912-4918` | **KEPT** (`20:14Z`) |
| `516–558` (`43f`) | `lead_fp_0516f` | `lead` | Sub-load 1 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`3.88x` H3 render zoom** (`[348, 290, 612, 488]`), `5.72x` `<Video 1>` crop (`78.8 px` median face) | **KEPT** (`21:39Z`) |
| `559–578` (`20f`) | `lead_fp_0559` | `lead` | Sub-load 2 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.51x` H3 render zoom** (`[472, 240, 880, 546]`), `3.68x` `<Video 1>` crop (`85.6 px` median face) | **KEPT** (`21:39Z`) |
| `579–599` (`21f`) | `lead_fp_0579` | `lead` | Sub-load 3 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.67x` H3 render zoom** (`[640, 264, 1024, 552]`), `3.89x` `<Video 1>` crop, `--hold-part 597-597` | **KEPT** (`21:39Z`) |

### 1.3 Kept Renders Re-Verified & Closed Without Remake
- **`3967–4052` (`lead_fp_3967c`)**, **`3060–3109` (`lead_fp_3049c`)**, **`168–184` (`lead_fp_0168d`)**: Verified good by the owner (`20:42Z`).
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

Full forensic details and transcript citations are in `internal/gemini/2026-10-28_session_ebdd0a24_subagent_postmortem.md`. Every session picking up this handover must enforce these five rules:

1. **Check `manifest.json` `zoom.scale` and `<Video 1>` `lash_gap` Before Running `render` on Small Faces**:
   - On `lead_fp_0516e` (`516–599`), the full `84f` travel box (`496x292`) exceeded the default `2.0x` zoom cap, causing H3 to fall back to `1.0x` whole-frame while `<Video 1>` (`2.165x`) had `83/84` frames under the `1.6 px` lash-gap threshold (white shut-eye ovals).
   - **Rule**: Whenever a subject's canvas face height is `< 35 px`, verify in `data/<date>_<load>/manifest.json` that `zoom.scale >= 2.5` and in `pose_pass.out` that `<Video 1>` lash lines are `>= 1.6 px` (`0` frames under `1.6 px`). Split long travel shots at natural movement boundaries and set `zoom_up_to: 4.0` in `data/fun/plan.json` (`shot_overrides`).
2. **Preserve Owner Ruling Holds (`--hold-part`) When Rebuilding Without `--carry-words-from`**:
   - When `--carry-words-from` fails on an obsolete sheet path, inspect `job.carried_words_from` or `overridden_by_word_frames` in the prior capture's `part_grades.json` before passing `--no-carry-words`, and re-supply any owner ruling holds (`--hold-part FIRST-LAST=WHY`) explicitly.
3. **Verify Preview Creation Date & Commit Provenance on Disk**:
   - Never assume a preview named `shot_c_<N>e` is post-cutoff from its letter alone. Always check `stat` / `job.json` date (`>= 2026-10-28` for crown-join into `parts.mkv`, `>= 2026-10-27` for `SAM 2.1` `head_track` and `strangers.py`).
4. **`page_spec.json` Schema Check (`look` Must Be a String)**:
   - `data/fun/poc/index.html:494` calls `(it.look || '').trim()`. Always pass `"look"` in `page_spec.json` as a plain string (`str`), never a JSON list, or `:8765` will throw a `TypeError` during DOM construction.
5. **Visual Contact-Sheet Inspection Before `load_chain.py page`**:
   - Never run `page` immediately after `land` without extracting and viewing a contact sheet comparing `ORIG` vs `RENDER` across representative frames (including any held, shut-eye, or fast-motion frames).
