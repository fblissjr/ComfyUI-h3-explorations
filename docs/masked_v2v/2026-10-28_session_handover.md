# 2026-10-28 Session Handover & Operational Safeguards (Historical Record)

> [!NOTE]
> **Historical snapshot from `2026-10-28` (`ebdd0a24`, updated for factual accuracy on `2026-10-29`).**
> For the current day boundary and active order, read [`docs/wiki/handover.md`](../wiki/handover.md) and [`docs/masked_v2v/fun_render_backlog.md`](fun_render_backlog.md).

**Session ID:** `ebdd0a24-32bd-4ef6-8f72-73bfe7927103` (`gemlead`)  
**Advisory Peer:** Claude (`mrfizz`, notes in `internal/claude/2026-10-28_mrfizz/note_to_gemlead_1938Z.md`)  
**Close Commit (`main`):** `f4ea7284`  
**Session Postmortems:** `internal/gemini/2026-10-29_postmortem_and_day_plan.md` and `internal/claude/2026-10-29_mrfam/postmortem_and_plan.md` (gitignored under `internal/`)  
**Canonical Render & Remake Backlog:** [fun_render_backlog.md](fun_render_backlog.md)

---

## 1. What Was Shipped & Verified on `2026-10-28`

### 1.1 Code & Tests Landed on `main` (`c8dee328`, `c60c6474`, `e88857cf`, `dc07ccee`, `c3b4c64d`..`f4ea7284`)
1. **Rejecting `gem/perception-lhf` (`16b0805c`) & Landing Clean Crown-Join in `bench/head_tube_parts.py` (`c8dee328`, `e88857cf`)**:
   - Audited the `gem/perception-lhf` worktree branch (`16b0805c`) on `4912` (`second_hfp_4912m`) and proved its 3-line gate caused a `+6,138 px` 1-px speck explosion on frame `4926` and a `+1,858 px` white-hat bleed on frame `4946`. Rejected `gem/perception-lhf` without merging.
   - Implemented the clean 2-gate upward crown join directly in [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py) (`c8dee328`, lines `135–148`, `2090–2116`):
     - `_SPECK_MIN_PX = 64`: filters out `< 64 px` stray SAM `head_track` specks before dilation (eliminating the `4926` 1-px speck explosion).
     - `_MIN_CROWN_ABOVE_HY0_PX = 256`: on head-and-hair loads (`wants_hair=True`, e.g. `second`) with a head track, expands `head_track_bool` upward into connected components of `parts_bool & raw_track_bool` (`y <= hy1`) that overlap the main `head_track` body by `>= 64 px` and have `>= 256 px` above `hy0`, without growing `tube_info["tube"]`.
2. **Pre-Render `check_head_tube_parts.py` Gate & Unit Tests (`c60c6474`)**:
   - Added [`bench/check_head_tube_parts.py`](../../bench/check_head_tube_parts.py) and unit tests (`tests/test_head_tube_parts.py`), wired into [`bench/load_chain.py`](../../bench/load_chain.py) `first` so any load where `parts_held.mkv` starves Sapiens `Face_Neck` or `Hair` fails loudly at `first` before GPU render time is spent.
3. **SAM 3.1 `head_track` 4-Corner Containment in `cranium_box()` & Default MoGe-2 Depth (`dc07ccee`, `c3b4c64d`..`f4ea7284`)**:
   - Replaced the ad-hoc `0.30` 2D score floor (`b7875bf1`) in [`bench/crop_reference_to_head.py`](../../bench/crop_reference_to_head.py) and [`bench/head_reference.py`](../../bench/head_reference.py) (`dc07ccee`) with 4-corner SAM 3.1 `head_track` containment grown by `CRANIUM_HEAD_TRACK_GROW = 0.50` times the active median face height (`win[k]`), bounding `grey_head` to `win[k]` so raised 3D arms or flipped torso limbs cannot stretch the reference crop box, and normalized `marks_against_the_body_model` by `span_fh` in [`bench/pose_along_shot.py`](../../bench/pose_along_shot.py).
   - Enabled `"depth": {"on": true}` by default in [`data/fun/plan.json`](../../data/fun/plan.json) and removed hardcoded share paths across `bench/` (`c3b4c64d`..`f4ea7284`).
4. **Status of `4912–4946` (`second_hfp_4912m/n/o`, Active in `fun_render_backlog.md` Seq `1.9`)**:
   - On `second_hfp_4912m` (marked wrong by the owner at `15:49Z`, `owner_answers.jsonl:1657-1658`), `4912n` (seed test), and `4912o` (crown-join test), `1,406–9,321 px` of other people's hair (`strangers` + `third`) inside the `4.0x` zoom box (`[364, 148, 620, 340]`) on `28/35` frames bleeds into `second`'s blond hair. The owner never parked `4912–4946`, and `4912n`/`4912o` were held off the review page by the sessions; `4912–4946` remains active in [`fun_render_backlog.md`](fun_render_backlog.md) Seq `1.9`.

### 1.2 Renders Landed & Kept by the Owner (`2026-10-28`, 8 Keeps in `data/fun/laid.jsonl` Rows 70–77)
| Span | Load(s) | Subject | Key Settings / Resolution | Verdict (`data/fun/owner_answers.jsonl`) |
|---|---|---|---|---|
| `4647–4725` (`79f`) | `lead_fp_4647d` | `lead` | Pass 3 over `third_fp_4647e` (`10-24`) + `second_hfp_4647i` (`10-27`) on `shot_c_4647m`; `--no-head-track`, `--drop 4647-4680 --no-face-on 4647-4680`, `--least 1024`, `--depth` | **KEPT** (`19:27Z`, `"looks good!"`, `laid.jsonl` row 70) |
| `516–558` (`43f`) | `lead_fp_0516f` | `lead` | Sub-load 1 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`3.88x` H3 render zoom** (`[348, 290, 612, 488]`), `5.72x` `<Video 1>` crop (`78.8 px` median face) | **KEPT** (`21:39Z`, `laid.jsonl` row 71) |
| `559–578` (`20f`) | `lead_fp_0559` | `lead` | Sub-load 2 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.51x` H3 render zoom** (`[472, 240, 880, 546]`), `3.68x` `<Video 1>` crop (`85.6 px` median face) | **KEPT** (`21:39Z`, `laid.jsonl` row 72) |
| `579–599` (`21f`) | `lead_fp_0579` | `lead` | Sub-load 3 of 3 on `shot_c_0516f`; `zoom_up_to: 4.0` -> **`2.67x` H3 render zoom** (`[640, 264, 1024, 552]`), `3.89x` `<Video 1>` crop, `--hold-part 597-597` | **KEPT** (`21:39Z`, `laid.jsonl` row 73) |
| `3759–3908` (`150f`) | `lead_fp_3759c` | `lead` | Continuous load on `shot_c_3759f`, `--drop 3851-3857` back-of-head spin | **KEPT** (`23:18Z`, `laid.jsonl` row 74) |
| `2254–2598` (`345f`, laid on `2271–2598`) | `lead_fp_2254c` | `lead` | Built on `shot_c_2050h`, `zoom_up_to: 4.0`, 10 interior `not seen` spans dropped | **KEPT** (`23:19Z`, `laid.jsonl` row 75 / updated to `2271–2598` per 10-25 ruling) |
| `3909–3966` (`58f`) | `lead_fp_3909c` | `lead` | Built on `shot_c_3909e` with `"<Subject 1> wears white sunglasses."` (`replaces: lead_fp_3909b`) | **KEPT** (`23:20Z`, `laid.jsonl` row 76) |
| `294–414` (`121f`) | `lead_fp_0294e` | `lead` | Built on `shot_c_0294b`, `2.592x` cranium reference, `312–319` and `350–357` dropped (`replaces: lead_fp_0294d`) | **KEPT** (`01:24Z`, `laid.jsonl` row 77) |

### 1.3 Kept Renders Re-Verified & Renders Paged for Owner Verdict (`01:10Z`)
- **Kept renders re-verified**: **`3967–4052` (`lead_fp_3967c`)**, **`3060–3109` (`lead_fp_2823b` on `2823–3097` and `lead_fp_3098b` on `3098–3109`)**, and **`168–184` (`lead_fp_0168c` on `156–232`)** were verified good on the join clip by the owner (`20:42Z`, `data/fun/owner_answers.jsonl:1662`). **`5935–6104` (`lead_fp_5935d`)** delivery check flags `d122` and `d123` were inspected frame-by-frame and stand as kept.
- **Rendered, landed, and on the review page (`:8765`) awaiting owner verdict**: `lead_fp_4272c` (`4272–4309`), `lead_fp_1134b` (`1134–1155`), and `lead_fp_0438b` (`438–483`).

---

## 2. Active & Remaining Render Queue

See [fun_render_backlog.md](fun_render_backlog.md) ("Today's queue, as ready on disk") and [`docs/wiki/handover.md`](../wiki/handover.md) ("Tomorrow, in order") for the live dependency-ordered queue. Do not read a queue from this historical handover.

---

## 3. Hard Operational Safeguards From the `2026-10-28` Postmortem

Full forensic details and transcript citations are in `internal/gemini/2026-10-29_postmortem_and_day_plan.md` and `internal/claude/2026-10-29_mrfam/postmortem_and_plan.md`. Every session must enforce these six rules:

1. **Re-Read Disk Records (`git log`, `laid.jsonl`, `owner_answers.jsonl`, `items.json`) After Context Compaction**:
   - After 21 context compactions across `3,760` steps on `2026-10-28`, relying on compacted summary memory caused the first draft of this handover to cite non-existent load IDs and conflate rejected worktree `gem/perception-lhf` (`16b0805c`) with `c8dee328`. Always run `git status --short`, `git log -n 5 --oneline`, `tail -n 5 data/fun/laid.jsonl`, and inspect `data/fun/review/items.json` before writing handover or backlog prose (`RULE[user_global]`).
2. **Check `manifest.json` `zoom.scale` and `<Video 1>` `lash_gap` Before Running `render` on Small Faces**:
   - On `lead_fp_0516e` (`516–599`), the full `84f` travel box (`496x292`) fell below `MIN_USEFUL_ZOOM = 1.25` (`1.08x -> 1.0x`), causing H3 to fall back to `1.0x` whole-frame while `<Video 1>` (`2.165x`) had `83/84` frames under the `1.6 px` lash-gap threshold (white shut-eye ovals).
   - **Rule**: Whenever a subject's canvas face height is `< 35 px`, verify in `data/<date>_<load>/manifest.json` that `zoom.scale >= 1.25` (ideally `>= 2.5x`) and in `pose_pass.out` that `<Video 1>` lash lines are `>= 1.6 px` (`0` frames under `1.6 px`). Split long travel shots at natural movement boundaries and set `zoom_up_to: 4.0` in `data/fun/plan.json` (`shot_overrides`).
3. **Pass `--least` Scaled by Zoom When `drop=` Removes Tiny Entry/Exit Slivers (`starting_floor` Ordering)**:
   - In `bench/capture_masked_run.py`, `starting_floor` (`L2736`) runs before `drop=` (`L2849`). On `lead_fp_4647c`, undropped back-of-head slivers on `4651–4680` caused `starting_floor` to compute `1.0x` (`4096 px`) instead of `2.0x` (`1024 px`), starving `4681–4725`. Whenever `--drop` removes part-sliver entry/exit frames on a zoomed load, pass `--least 1024` (or `4096 / zoom^2`) explicitly.
4. **Preserve Owner Ruling Holds (`--hold-part`) When Rebuilding Without `--carry-words-from`**:
   - When `--carry-words-from` fails on an obsolete sheet path, inspect `job.carried_words_from` or `overridden_by_word_frames` in the prior capture's `part_grades.json` before passing `--no-carry-words`, and re-supply any owner ruling holds (`--hold-part FIRST-LAST=WHY`) explicitly.
5. **Verify Preview Creation Date & Commit Provenance on Disk**:
   - Never assume a preview named `shot_c_<N>e` is post-cutoff from its letter alone. Always check `preview.queued.json` `commit` against the stage writers (`>= 2026-10-28` for `c8dee328` crown-join in `head_tube_parts.py`, `>= 2026-10-27` for `SAM 3.1` `head_track` and `strangers.py`).
6. **`page_spec.json` Schema Check (`look` Must Be a String) & Mandatory Visual/Luma QA Before `page`**:
   - `data/fun/poc/index.html:494` calls `(it.look || '').trim()`. Always pass `"look"` in `page_spec.json` as a plain string (`str`), never a JSON list, and never chain `land && page`: always inspect the rendered `.1fps.mp4` (`read_file`) and luma/B-R table before running `load_chain.py page`.
