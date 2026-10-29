# Decisions and reversals

Written by hand. One dated line per decision the owner made, or per claim in
the prose that was corrected: what changed, what it used to say, where it
lives now, and the commit. Newest first. `CLAUDE.md` and `VISION.md` carry no
history notes; this page is where those go. Another document may also keep a
dated note in place, where a reader would otherwise trust the stale text.

Older history lives elsewhere and is not copied here:

- [`CHANGELOG.md`](../../CHANGELOG.md): every change, by version.
- [`docs/rules_history.md`](../rules_history.md): `CLAUDE.md` as it stood
  before the 2026-09-03 cut, frozen.
- [`docs/roadmap.md`](../roadmap.md): "Closed lanes", and the dated "Owner
  decisions" and forward-plan sections.
- `bench/results/`: the verdict records, each with its conditions.

## 2026-10-29

- **A shut stretch is named with `--eyes-shut-form closed` and a text sentence that says the eyes are shut; for an hour on 2026-10-29 the rules page said `lowered`** (corrected 2026-10-29 after `second_hfp_0278n`, marked wrong by the owner at 21:20Z): `closed` paints the eye with no iris and the text has no sentence for it, so H3 took the still's straight-ahead eyes, which the owner's rule of 2026-10-22 predicts; the kept loads with an eye word used `lids` and `lowered`. The same render's mouth opened because the second subject's outlined closed mouth made `bench/load_text.py::mouth_drawn` write the mouth-follow clause on a silent shot; the same text kept the mouth shut on `second_hfp_5114k`, so the template change is a judgement call, not made. A scan of every job on disk found the `closed` form on no other load, and the mouth pattern only on `5114k` and four unkept 10-22 jobs. The advisory session's first correction said `lowered`, and `gemrender` refuted it the same evening: on `278–293` the downcast rule had already drawn the `lowered` picture on `0278m`, so `lowered` re-renders the rejected letter; the rules page now says `closed` with a by-hand shut-eyes sentence until the template has one. The record is `docs/masked_v2v/2026-10-29_findings_0278n_text_and_frame_table.md`, which also records the `bench/frame_table.py` restored-frames bug `gem4912` found (restored frames reported as emptied), fix in progress on its branch.

- **Leave reader files as the reader wrote them; add a dated note beside a slip rather than editing the reading** (the owner, 2026-10-29, in `gemlead`'s terminal): `<capture>/looks/<load>.reading.md` and other reader files stand as the reader wrote them; if a reading contains a slip (such as a retired `Verdict` line or a literal share path), add a dated note beside the slip rather than editing or rewriting the reader's words. Applies to readings from here on (`docs/masked_v2v/PROTOCOL.md` §5.1).

- **Park `4912–4946` (`second_hfp_4912p`) and focus on the rest of `docs/masked_v2v/fun_render_backlog.md`** (the owner, 2026-10-29, in `gemlead`'s terminal): after `second_hfp_4912p` (the conditioning-only `encode_file` test on `4912–4946`, where `MiniMaxH3MaskedSource` passed the raw `1980s` source to the VAE and the `second_hfp_4912d` composite to Qwen3-VL) still rendered dark brown hair (`median L* = 11.0`, `mean L* = 16.0` on `4912–4920`), shot `4912–4946` (`second`) is parked while the remaining `fun` render backlog is completed (`docs/masked_v2v/fun_render_backlog.md`).

- **Second subject on `278–293` (`second_hfp_0278m`, `16f`) reopened on the owner's word** (the owner, 2026-10-29, on the review page `data/fun/owner_answers.jsonl:1688-1689` and in `gemlead`'s terminal): `second_hfp_0278m` (`2026-10-22`, `laid.jsonl` row 21) is marked `"wrong, and I say where"` (*"Her eyes are open in the render. In the source they are not. Doesn’t match the input on 278-293."*). Why `0278m` rendered open eyes: on `278–293` (`data/2026-10-22_second_hfp_0278m_job/job.render.json:688-781`), `DOWNCAST_RING_OUTSIDE_MIN = 0.65` fired on all 16 frames (`share_of_the_ring_outside_the_lids = 0.69..0.77`) and `EYE_OPEN` propped the lids open (`lids_opened_to = 0.12` on `278..293`) with a dark iris disc, whereas `--eyes-shut-on 278-293 --eyes-shut-form closed` (`bench/draw_face_points.py::sculpted_eye`, calibrated on `278, 286, 291` on `2026-10-22` to paint out the 3D body model's open sculpted eye in `skin_grey` and draw one closed lid line) was never applied to a `0278` render. Remade today as `second_hfp_0278n` on a fresh single-subject preview `shot_c_0278g` with `--eyes-shut-on 278-293 --eyes-shut-form closed`.

- **Opening scene (`second_hfp_0014g`, `14–167`, `154f`) reopened on the owner's word** (the owner, 2026-10-29, on the review page `data/fun/owner_answers.jsonl:1686-1687` and in `gemlead`'s terminal): `second_hfp_0014g` (`2026-10-23`, `laid.jsonl` row 39, made on `shot_c_0014e` from `2026-10-22` with `"the strangers-cut flicker on 48 and 66-69 accepted as a nitpick"`) is marked `"wrong, and I say where"` (*"janky, artifacts around the mask edges throughout 14-167. Made on the 23rd before the crown join, the depth default and the hair-in-hole fixes. Remake on a fresh preview with current tooling."*). Remade today as `second_hfp_0014h` on a fresh single-subject preview `shot_c_0014f` from the raw source with today's chain (automatic MoGe-2 depth on overlap frames, `head_tube_parts.py` crown join `c8dee328`, and `cranium_box` containment `dc07ccee`), queued after what is already built; `lead_fp_0168c` (`168–185`) stays as kept unless the owner marks it too. Per the owner's instruction, the pre-render reader names the mask edge frames (`48, 66–69`, etc.) specifically, and the post-land reader compares the mask edges against `second_hfp_0014g` as well as the source.

- **A changed dependency invalidates what was made from it** (the owner, 2026-10-29, the Claude advisory session's
  terminal, 16:5xZ): if anything in the shot record, the plan, or the job that a mask or a render depends on changes,
  that mask or render is invalidated and made again; nothing made from the old input is carried. This restates the
  lane's prep rule (`docs/masked_v2v/AGENTS.md`, "Prep") with the trigger widened from "code that feeds the stage" to
  every input in the dependency chain: `shots.json`, `plan.json`, `cast.jsonl`, the job's words, and the writers. The
  one-variable exception stands only where nothing in that chain changed. Applied the same day to the 4912 test
  (`second_hfp_4912p`): the shot, plan, preview and capture are unchanged, so the masks stand; the job gains one input
  the masks do not depend on (a conditioning-only video), so the render is new. What enforces it today: writer stamps
  on derived files (`bench/_lib/code_stamp.py`, `bench/inspect_mask_steps.py` block `[0]`) for code changes, and the
  recipe gate at `job` for the plan's settled knobs. What does not: no tool yet refuses a capture whose `shots.json`,
  `cast.jsonl` or `plan.json` words changed after it was made; that comparison is by hand until built.

- **The lane's prose said "No session looks at a render before the owner does" in three places after the owner had
  reversed it** (corrected 2026-10-29 by the Claude advisory session on the owner's ask, in the shared tree with a
  clean `git status`). `docs/masked_v2v/AGENTS.md` ("Judging a render"), `docs/masked_v2v/chain_trace_and_remakes.md`
  (stage 6) and `docs/masked_v2v/PROTOCOL.md` 3.3 each still carried the 2026-10-22 sentence. The owner's word of
  2026-10-28 (the pre-`page` rule above, after `second_hfp_4912m` reached the review page unread) reversed it, and the
  lead's own postmortem of 2026-10-28 named the stale sentence as the license for that page. The three now say: a
  landed render is read before `page` by a reader who did not make it, who writes what it sees with frames, what it
  cannot see and the measured part shift, with no verdict and no hold. `PROTOCOL.md` 5.1 gains the step between
  `land` and `page`, and both copies of the `video-understanding` skill gain "Who reads a landed render, and what the
  reading is" (the owner, 2026-10-29: independent, not blind, so the lead's context holds a reading and not a strip).

## 2026-10-28

- **Replaced the blind `0.30` 2D face confidence floor and `2.5x` scale filter in `cranium_box()` with SAM 3.1 `head_track` 4-corner containment (`CRANIUM_HEAD_TRACK_GROW = 0.50`) + bounding `grey_head` to `win[k]`, and fixed `marks_against_the_body_model()` span normalisation** (the owner, 2026-10-28; commit `dc07ccee`):
  1. **Why `CRANIUM_SCORE_FLOOR = 0.30` and unbounded `grey_head` were replaced**: `b7875bf1`'s `0.30` 2D face confidence floor dropped valid turned-away/profile frames (e.g. `lead_5044_5113` frame `5083` at score `0.14`, leaving `9` edge frames at `3.606x`), while leaving `grey_head` unbounded above mid-face allowed raised 3D arms or flipped torso limbs connected through the neck/shoulder (`lead_0294_0414` frame `317`, `lead_5044_5113` frame `5044`, `lead_2254_2598_c`, `third_5011_5043`, `third_4479_4509`) to stretch the cranium box.
  2. **The new rule (`bench/crop_reference_to_head.py::cranium_box`, `bench/head_reference.py::run`)**: When a `head_track` is read (`head_reference.py` now discovers and passes `head_track` under `--cranium`), each tracked frame's head-track bounding box is grown by `CRANIUM_HEAD_TRACK_GROW = 0.50` times the active sizing median face height (`face`) on every side (`win[k]`): frames with an empty head track (`frames_with_no_head_track`) or whose 2D face box is not 4-corner contained in `win[k]` (`frames_off_the_head_track`) do not size the box, and `grey_head` is bounded to `win[k]`. One active sizing median (`face`) is used everywhere; `CRANIUM_FACE_OUTLIER = 2.00` applies only as a fallback when no `head_track` is present (`0/110` records on disk).
  3. **3D body-model marks normalisation (`bench/head_reference.py::marks_against_the_body_model`)**: `nd` and `eye_span_fh` are now divided by `min(frame_fh, span_fh)` where `span_fh` is the span's median positive face height across non-`faceless` frames, so a blown-up single-frame occlusion face box (`436.8 px` against `49.7 px` on `lead_5044_5113` frame `5044`) cannot divide its own nose-distance error by `9x`.

- **Mandatory pre-`page` measurement and `.1fps.mp4` inspection rule, and `second_hfp_4912n` (`seed 730451893`) one-variable controlled test on `4912–4946`** (the owner, 2026-10-28):
  1. **Pre-`page` QA rule**: Never call `bench/load_chain.py page` immediately after `land`. Every landed render must first be checked by measuring the replaced-part color/luma shift inside `parts_held` (e.g., per-frame luma and `B-R` shift against the `1024x768` `cv2.INTER_AREA` source frames and kept reference loads) and inspecting the landed `.1fps.mp4` stacked review clip (`read_file` per `.gemini/skills/video-understanding/SKILL.md`). Only renders that pass the pre-`page` check are sent to the review server.
  2. **Controlled one-variable seed test on `4912–4946` (`second_hfp_4912m` at `seed 730451892` vs. `second_hfp_4912n` at `seed 730451893`)**: After `second_hfp_4912m` rendered dark brown hair on `4912–4920` and `4926–4940` (frame `04916` flagged on the review page), measuring the node's `tokens` mask (`*_window_1_region.npz`) confirmed that `0 px` of `second`'s raw Sapiens hair inside her track lay outside the DiT `tokens` hole on `4912–4920` in both `4912m` and `4912e` (`1,348–2,522 px` outside `tokens` on `4927–4928` in `4912m` due to `head_tube_parts.py` clipping where SAM `head_track` only segmented her lower face). Because all three dark `4912` renders (`4912e`, `4912g`, `4912m`) used the default `seed 730451892`, `second_hfp_4912n` (`data/2026-10-28_second_hfp_4912n`) was built and rendered at `seed 730451893` (`second_hfp_0278k` precedent) as a strict one-variable test:
     - **Identical masks**: `bench/look_carry.py` verified `track`, `parts`, and `region` equal on `35 of 35` frames against `2026-10-27_second_hfp_4912m`.
     - **Identical prompt text (`sha256`)**: `280e22d11e9c2d2b12f6c0ef12724ce585e38d6419a9de201ce1bff30b8f03b2` (`second_hfp_4912m.txt` == `second_hfp_4912n.txt`).
     - **Identical `<Video 1>` head reference (`sha256`)**: `be911be40960d2c81ae6383a5c00cd514964bed3e3036ec3aec8754f5a10348e` (`second_4912_4946_h/second_head_reference_4912_4946.mp4` == `second_4912_4946_i/second_head_reference_4912_4946.mp4`).
     - **Result (out of range; seed is not the cause)**: Inside `parts_held` on `4912–4920`, `second_hfp_4912n` measured `dluma = -0.7 to +9.0` (`+1.1` on `4916`) and `dBR = -8.7 to -23.0` (`-10.1` on `4916`), and `-5.5 to +6.7` `dluma` on `4926–4940`—matching `4912m` (`dluma = +6.0 to +11.7`, `dBR = -11.6 to -20.7` on `4912–4920`, negative on `4926–4935`) and well outside kept blonde `second_hfp_4647i` (`dluma = +17 to +20`, `dBR = -32 to -37`). Visual inspection of `second_hfp_4912n_4912_4946_stacked.1fps.04912-04946.mp4` confirmed dark brown hair across `4912–4940`. `4912n` was held back from automatic paging on this pre-`page` measurement (the first application of today's pre-`page` QA rule, then paged on explicit owner request for side-by-side inspection).
     - **What is now ruled out on `4912` by measurement, and what the `4647g` → `4647h` → `4647i` progression proves**: (a) sampler seed (`730451892` vs. `730451893`), (b) her raw Sapiens hair outside the DiT `tokens` hole on `4912–4920` (`0 px` in `4912e`, `4912m`, `4912n`, and kept `4647i`), (c) composite `matte` vs. raw window (the raw window itself is dark brown), (d) prompt text, `<Picture 1>` still, and `<Video 1>` reference form (byte-identical recipe to kept `4647i`), (e) head pixel area (`kept` blonde loads span `1.5k–56k px`), and (f) other people's dark hair outside the hole (`196k px` in `4647g`, `4647h`, and `4647i` alike). Conversely, comparing `4647g`, `4647h`, and `4647i` (which share identical masks on `4647–4672`, `IoU = 1.00`, same text, seed, and source cut) proves **within-window 3D bidirectional temporal contamination**: `4647g` left ~29 late frames of her head unmasked and rendered dark even on `4647–4656` (`luma 72.6`, shift `-4 to -7`); `4647h` left ~12 post-occlusion frames (`4684–4695`) unmasked and rendered blonde on `4647–4673` (`luma 95.1`, shift `+11 to +16`) then dark; `4647i` left `0` visible frames unmasked and rendered blonde end-to-end (`luma 98.6`, shift `+19 to +22`). On `4912–4946`, **no render has ever had all visible frames of her head inside the `tokens` hole at the same time**: `4912e` (`2026-10-20`) and `4912g` (`2026-10-25`) covered her crown on `4927–4932` but left `2,549–3,561 px` out on `4926` and `1,723–4,549 px` out on `4943–4946` (`tok[10]=0` in `4912e`, `tok[9]=tok[10]=0` in `4912g`); `4912m` and `4912n` covered `4943–4945` via `hers` + `least=400`, but `head_tube_parts.py` (`2026-10-25`) clipped her crown (`y=302..348`) out of `parts_held` on `4926–4932`, leaving `1,097–3,067 px` of her dark hair outside the `tokens` hole on `4926–4932` and `1,674 px` carved out on `4946`. Thus the next `4912` render is made only after the `head_tube` crown-clipping fix is built in a worktree, blind-audited with `0 px` diff on kept `hfp` controls, and verified to leave `0 px` of her visible hair/face outside the `tokens` hole across all 35 frames.

- **Corrections to 2026-10-26 and 2026-10-27 prose in `docs/wiki/decisions.md`, `docs/wiki/handover.md`, `docs/wiki/next_steps.md`, and `docs/masked_v2v/chain_trace_and_remakes.md`** (verified against `git show 1e91de26`, `data/2026-10-27_second_hfp_4647i/subjects/second/part_grades.json`, `data/2026-10-27_second_hfp_4912m/flags.json`, `data/fun/laid.jsonl`, and `data/fun/owner_answers.jsonl:1285,1375,1456`):
  1. **What `1e91de26` actually landed vs. the 2026-10-26 "retire `hold_to_track`" prose**: The 2026-10-26 entry below (`L110`) and `docs/wiki/next_steps.md` (`2026-10-27`) described retiring `hold_to_track` and `_head_seed_from_person_mask`'s flat slice. An initial morning sprint on `2026-10-27` attempted a 9-stage retirement (`a143e1cf`) that broke the pipeline on `4647` (`Upper_Clothing` added to `Hair`, fabricated `marks.json` schema, control-set regression) and was discarded cleanly (`internal/gemini/2026-10-27_morning_build_postmortem_by_gemini.md`). What actually landed on `main` (`1e91de26`) was surgical (+35/-6 lines in `bench/capture_masked_run.py` and `bench/head_tube.py`): `grade_holds` (`bench/capture_masked_run.py:4678-4695`) takes `head_tube`'s tube directly when `head_tube` ran (`head_tube: true`, bypassing the `least_px` floor branch where `head_tube` kept the mask, with `did: "saved part kept"` or `"saved part kept, fill refused"`), and `her_hair_in_hole` clears `taken_for_hers` inside `head_track` (`"saved part kept, restored by her_hair_in_hole"`). `hold_to_track` and `_head_seed_from_person_mask` remain in `bench/capture_masked_run.py`.
  2. **`second_hfp_4647i` ran at the default `1,024 px` floor, not `--part-least-px` / `least = 400`**: The 2026-10-27 entry below (`L27`), `docs/wiki/handover.md` (`L97`), and `docs/masked_v2v/chain_trace_and_remakes.md` (`§4.1`) claimed `second_hfp_4647i` was saved by `--part-least-px second=400:...` (`shot_to_load.py`'s actual CLI flag is `--least PX --least-why`). In fact, `data/2026-10-27_second_hfp_4647i/subjects/second/part_grades.json` (`L1–15`) records `least_px: 1024`, `least_px_set_by: "the starting value (PART_LEAST_PX)..."`, and `head_tube: true`, and its 7 sub-`1,024 px` emerging frames on `4686–4702` (`4686, 4687, 4691, 4692, 4700, 4701, 4702` at `609–951 px` in `parts_held__shot_c_4647l.mkv`, plus `4673` at `622 px` and `4703–4715` at `332–694 px`; `4682` is `0 px` `emptied`, while `4688` is `1,289 px` and `4698` is `1,260 px`) were retained because `grade_holds` took `head_tube`'s tube (`did: "saved part kept"` or `"saved part kept, fill refused"`). Thus `second_hfp_4912m` (`least_px: 400` in `data/2026-10-27_second_hfp_4912m/subjects/second/part_grades.json:5`) is the first load rendered with `least: 400` (`4946` is retained by `"saved part kept, restored by her_hair_in_hole"` at `4,549 px`, while `4943–4945` are marked `emptied` in `part_grades.json` and carried in `region` by `head_tube`), kept as a single stated variable on `4912m` and not added to `data/fun/plan.json`'s global `recipe`.
  3. **`second_hfp_4912m` gate state (`blocked` at open) and `second_hfp_4912l` retired**: `docs/wiki/handover.md` (`L27`) called `second_hfp_4912m` "unblocked (`flags.json` verdict: ready to render)", and `docs/masked_v2v/chain_trace_and_remakes.md` (`§4.2`) cited `second_hfp_4912l` (which dropped the `4920–4926` `--no-draw-face-on` span and was replaced by `4912m`). On disk, `data/2026-10-27_second_hfp_4912m/flags.json` had `"verdict": "blocked"` on `f002` (`key: 7eb6b893b25b6aff`, `region_on_a_frame_with_no_mask` on hidden frames `4922–4925, 4941–4942`) because `outcomes.json` was written two minutes before `on-plan` and overrode only pre-plan `f002` (`key: fba2b42a97fb68b2`).
  4. **Two owed multi-subject pieces missing from `docs/wiki/handover.md`'s queue**: (a) `third` on `4912–4946` (`35f`, Pass 2 on the composite cut after `second_hfp_4912m` is kept, `margin = 16`, face only, no beard, per `data/fun/owner_answers.jsonl:1456` `gap_4912_third`; `0` loads currently laid on `4912–4946` in `data/fun/laid.jsonl`), and (b) `second` on `1419–1465` (`47f`, unreplaced removal `second_hfp_1419c` in `data/fun/laid.removed.jsonl:4`, owed per `data/fun/owner_answers.jsonl:1285,1375` where `lead_fp_1419f` is already kept and laid on `1419–1465`).

## 2026-10-27

- **Delivery table `restore=` spans pinned to explicit `"loads"`, `3060–3109` and `3967–4052` `Eyeglass`/`Hand` restores retired after A/B join tests, and 3-regime accessory rule documented** (the owner, 2026-10-27; commits `187dc9fd`, `bbe13fea`, `a5bde270`, `c1090081`, `1acffe40`; documented in [`docs/masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md)):
  1. **Root cause of `day_2026-10-27` double sunglasses on `lead_fp_3967c` (`3967–4052`)**: `lead_fp_3967c_00001.mp4` rendered clean white sunglasses in H3 (prompt `<Subject 1> wears white sunglasses`), and the per-shot review clip (`lead_fp_3967c_3967_4052_stacked.mp4`, built at `bench/load_chain.py` `land` time) showed `lead_fp_3967c_00001.mp4` directly without running `bench/assemble_delivery.py`. At whole-video assembly, `bench/delivery_table.py::own_restores` matched `data/fun/plan.json`'s legacy `"first": 3909, "last": 4052` `Eyeglass` entry by frame span alone and emitted `restore=lead.Eyeglass` onto `lead_fp_3967c`'s row (`3970–4035`), causing `bench/assemble_delivery.py::worn_of` to paste the source actress's translucent sunglasses on top of H3's white sunglasses.
  2. **Step 1 (`187dc9fd`)**: `bench/delivery_table.py::own_restores` (`L308–355`), `bench/delivery_table.py::decide` (`L523`), `bench/shot_coverage.py::load_plan` (`L202–205`), and `bench/shot_to_load.py::load_job` (`L3496–3500`) now support `"loads": ["<load_name>", ...]` on `plan["subjects"][label]["restore"]` entries so an exception applies only to the named kept load(s) and never leaks onto a later remake or adjacent load on those frames.
  3. **A/B join-test audit (`3909–4052` and `2823–3109`) & retirement of `3060–3109` restores**: Running `bench/assemble_delivery.py` with and without `restore=` proved that both `lead_fp_3098b` (`3098–3109`, where `"Eyeglass"` is in `part_is_made_of` at `94.8%–100%` inside `matte > 127`, the prompt names white sunglasses, and 27 unmasked context frames `3071–3097` show them) and `lead_fp_2823b` (`2823–3097`, where `"Eyeglass"` is in `part_is_made_of`, the prompt names raising and putting on the white sunglasses, and its reconstructed `"only what changed"` matte preserves both sunglasses and hands across `3060–3097`) already carry clean single sunglasses and hands in their rendered `.mp4`s. Applying `9efd2f2b`'s face-shifted `restore=lead.Eyeglass` on top of them caused double top rims on `3098b` (`3099–3101`, `3104–3108`) and double ear-hooks/top edges on `2823b` (`3068`, `3072`). Dropping `3060–3109` (`Eyeglass` and `Left_Hand+Right_Hand`) from `data/fun/plan.json` leaves only the two `lead` loads where H3 omitted the accessory inside the mask: `lead_fp_3909b` (`3909–3966`, `Eyeglass`) and `lead_fp_0168c` (`168–184`, `Apparel@head` hat). Dangling earrings on `lead_fp_2823b` (`2823–3097`) lie `57%–79%` outside `matte` in untouched source video and require no `restore=`.
  4. **`--help` epilogs and `her_hair_in_hole` `did` allowlist (`1acffe40`)**: Added the 3-regime `restore=` rules to `bench/assemble_delivery.py --help` and `bench/delivery_table.py --help`, and added `"saved part kept, restored by her_hair_in_hole"` (`1e91de26`) to `bench/frame_table.py::CLOSED_DID_VOCAB` (`L74`, `L934`).

- **Kept and laid `second_hfp_4647i` (`4647–4725`, `data/fun/owner_answers.jsonl:1653`, `laid.jsonl` row `70`) and `third_fp_5114e` (`5114–5129`, `data/fun/owner_answers.jsonl:1655-1656`, `laid.jsonl` row `71`), and recorded the proven `--part-least-px` + `--no-draw-face-on` recipe (`fbe9d371`)** (the owner, 2026-10-27):
  1. **`second_hfp_4647i` (`4647–4725`)**: Kept by the owner (`"keep this one"`, `"Looks good!"`) after combining (a) the morning tracker/strangers/head_tube builds (`426638a6`, `1e91de26`, `6b0ea9a6`, `39e1f29a`), (b) `--part-least-px second=400:...` so `grade_holds` (`part_grades.json`) retains emerging small head/hair masks (`696–942 px` at `2.17x` zoom) instead of emptying them under the `1,024 px` floor, and (c) `--no-draw-face-on` on occluded/profile frames (`4674–4695`, `4702–4725`) where 3D-pose face points drift off the grey head while keeping `<Video 1>`'s grey head silhouette and clean face outlines on `4647–4673` and `4696–4701`. Documented in [`docs/masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md) (`§4.1`) and [`docs/masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md) (`§1`).
  2. **`third_fp_5114e` (`5114–5129`)**: Remade at settled `margin = 16` (`regenerating_share = 0.0888`) on a fresh preview (`shot_c_5114e`), eliminating the white reference eye/brow lines from `third_fp_5114d` (`margin = 64`, `0.2008`); kept by the owner (`data/fun/owner_answers.jsonl:1655-1656`) and laid as row `71` in `data/fun/laid.jsonl`.

- **Sharded `docs/checks.md`, `docs/prompting.md`, `docs/evidence.md`, and `docs/roadmap.md` into `< 40 KB` routers, added `docs/research/INDEX.md`, added a `58/58 <= 45,000 B` router size check in `bench/build_wiki_index.py`, and aligned masked-lane docs with `426638a6`..`6b0ea9a6`** (the owner, 2026-10-27, `15470307`):
  1. Sharded `docs/checks.md` (`492.8 KB` → `36.5 KB` router) into `docs/checks/standard_and_gaps.md` (`34.6 KB`), `docs/checks/doc_link_absent_history.md` (`23.1 KB`), and `docs/checks/index_01.md`..`index_15.md` (`< 34.5 KB` each), keeping the `189`-row 4-column summary index and `doc-link-absent` ledger in `docs/checks.md`.
  2. Sharded `docs/prompting.md` (`147.7 KB` → `35.5 KB` router for `§1`–`§5.7`) into 6 leaf files under `docs/prompting/` (`§5.8`–`§16`, `< 28 KB` each), and updated `bench/check_prompt_docs_sync.py` (`read_manual()`, `section_10_slice()`) and `.claude/skills/h3-prompt/SKILL.md`.
  3. Archived historical retracted-claims postmortems and August/September 2026 forward plans from `docs/evidence.md` (`73.4 KB` → `34.2 KB`) and `docs/roadmap.md` (`69.8 KB` → `25.4 KB`) into 4 files under `docs/wiki/archive/` (`< 26 KB` each).
  4. Added [`docs/research/INDEX.md`](../research/INDEX.md) (`10.5 KB`) linked from [`index.md`](index.md) (`0` unreachable docs across `docs/`) and extended `bench/build_wiki_index.py` to check `58/58 <= 45,000 B`.
  5. Updated [`docs/masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md), [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`docs/masked_v2v/code_map_stages.md`](../masked_v2v/code_map_stages.md), and `data/fun/AGENTS.md` to reflect the landed 2026-10-27 morning builds (`426638a6`: `strangers.py` unowned SAM detections where Sapiens finds nobody; `1e91de26`: `grade_holds` takes `head_tube`'s tube when `head_tube` ran; `6b0ea9a6`: `head_mode` defaults to `"per stretch"` `HEAD_PER_STRETCH` in `subject_track.py` and `bench/shot_to_load.py`, and `bench/cast_file.py box` requires `--people` verification).

- **Modularized oversized docs into `< 40 KB` router indexes and leaf specs (`docs/masked_v2v/`, `docs/wiki/archive/`, `docs/wiki/decisions/`), added top router callouts to early-October wiki pages, and fixed 3 broken citations** (the owner, 2026-10-27, `3886db1f`). To ensure every active router and leaf doc fits in a single `agy` `read_file` call (`46,080` bytes) and adheres to progressive disclosure with zero duplication:
  1. Extracted `§7` (`~15 KB`) and `§8`+`§10` (`~31 KB`) from `docs/masked_v2v/AGENTS.md` (`83.2 KB` → `32.3 KB` router) into [`docs/masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md) and [`docs/masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md), removing the two-half `read_file` workaround from root `AGENTS.md` step 4.
  2. Archived the historical `2026-10-04..2026-10-15` chronicle of `docs/wiki/masked_v2v.md` (`161 KB` → `3.8 KB` router) to [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md) and the `2026-09-05..2026-10-20` queues of `docs/wiki/next_steps.md` (`190 KB` → `2.2 KB` router) to [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
  3. Sharded earlier decisions (`2026-09-11..2026-10-23`) out of `docs/wiki/decisions.md` (`491 KB` → `37 KB` active log + router table) into 17 `< 40 KB` dated files under `docs/wiki/decisions/`.
  4. Added top router callouts to [`stages.md`](stages.md), [`state_signals.md`](state_signals.md), [`known_before_a_render.md`](known_before_a_render.md), [`who_is_who.md`](who_is_who.md), [`who_is_who_evidence.md`](who_is_who_evidence.md), [`window_context.md`](window_context.md), and [`window_edges_data.md`](window_edges_data.md), and fixed the 3 broken citations in `bench/check_speaker_id_control.py:2`, `docs/prompt_length_experiment.md:35`, and `docs/research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md:264,304,439`.

- **Consolidated environment setup, ComfyUI startup, `comfy` import rules, `agy` subshell behavior, and worktree placement into `docs/comfy_notes.md`, and retired `internal/gemini/local_env.md`, `internal/gemini/PROTOCOL.md`, and `internal/gemini/wait_for_change.py`** (the owner, 2026-10-27). Previously, untracked `internal/gemini/local_env.md` duplicated `docs/comfy_notes.md` (and carried a bare `main.py` server recipe without `--output-directory` / `--input-directory`, whereas ComfyUI nodes write through `folder_paths.get_output_directory()` set by `<comfy>/start.sh` flags), while `AGENTS.md` and `docs/wiki/working_with_the_owner.md` routed start-of-day and landing clearance through `internal/gemini/PROTOCOL.md`, a two-lead Claude/Gemini mailbox, and the H3 Open List. Removed `internal/gemini/local_env.md`, `internal/gemini/PROTOCOL.md`, and `internal/gemini/wait_for_change.py`; added the repo-relative environment discovery block (reading `H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`, and `PY` from `../../start.sh` and `../../models/sapiens2/` with zero private paths), the `agy` subshell non-persistence rule, `bench/_lib/__init__.py::bootstrap` import rules, and `<comfy>/h3_worktrees/<branch>` placement rules to `docs/comfy_notes.md` and `docs/masked_v2v/PROTOCOL.md` Stage 0; and updated `AGENTS.md`, `docs/wiki/working_with_the_owner.md`, `data/fun/AGENTS.md`, and `data/fun/FINAL_STRETCH_RUNBOOK.md` so new Gemini (`agy`) sessions have a single tracked source of truth.

- **Local files in the repo are the sole source of truth for decisions and state, not the H3 Open List** (the owner,
  2026-10-27). Root `AGENTS.md` ("Where state lives") used to say the one home of "decisions waiting on the owner, their
  answers, and their passing thoughts" was "the H3 Open List (the owner's page with Claude): one card a decision; the
  'Tell Claude anything' box is the owner's inbox". Corrected in `AGENTS.md` to name the local files
  (`docs/wiki/decisions.md`, `docs/wiki/handover.md`, `docs/masked_v2v/`, and `data/<video>/`), with any hosted page
  treated only as a downstream view that may carry decisions to the owner and is never read back as the record.

## 2026-10-26

- **Corrections to 2026-10-25 prose in `docs/wiki/handover.md`, `docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`, and `docs/masked_v2v/AGENTS.md`** (verified against `git show 449b17f1`, `bench/capture_masked_run.py:2558, 2643, 2797`, `data/2026-10-25_second_hfp_4647g/manifest.json`, and `data/2026-10-25_second_hfp_4912g/`):
  1. **`video_mask.py::HELD_EDGE_EMPTY_RUN`**: `docs/wiki/handover.md` claimed `449b17f1` lowered `HELD_EDGE_EMPTY_RUN` from `len(FRAME_PER_TOKEN)` (`5`, `17` frames) to `4`, and `2026-10-25_pass_contamination_and_mask_step_forensics.md` claimed it was `8` (`2` tokens) before `449b17f1`. In fact, no empty-run reach-back cap existed in `video_mask.py::held_edge_source` after `f5e0d66d`; commit `449b17f1` introduced `HELD_EDGE_EMPTY_RUN = max(FRAME_PER_TOKEN)` (`4`, one 4-frame latent step) for the first time.
  2. **`head_tube` default vs. `her_hair_in_hole` input on `4647g`**: `docs/wiki/handover.md` and `2026-10-25_pass_contamination_and_mask_step_forensics.md` claimed `second_hfp_4647g` ran without `--head-tube` because `--head-tube` was off by default. In fact, `bench/capture_masked_run.py:2558` enables `head_tube` by default (`not args.no_head_tube`, `head_tube: true` in `data/2026-10-25_second_hfp_4647g/manifest.json:277`), and `head_tube__shot_c_4647j.json` already regained and bridged `4686-4695` (`907-2,867 px`), while `bench/capture_masked_run.py:2797` handed raw `head_track_arr` (`0 px` on `4674-4695`) instead of `head_tube__<by>.npz["tube"]` to `her_hair_in_hole`. On `4696-4703`, `second`'s person track held `1,416-6,652 px` (`0 px` only on `4697`), but on `4698-4706` that track was person 3, the woman in the black vest, and her own head track on `4698-4702` was `0, 75, 456, 377, 705 px` (item 6 below; corrected 2026-10-26 after a reader's check). The instruction to graft `second`'s tracks from raw preview `shot_c_4647f` across plates and to pass multi-span `--hold-part A-B+C-D` (which `bench/capture_masked_run.py:2643` refuses; multiple gaps use owner `head_marks` in `marks.json`) has been struck.
  3. **`4912g` occlusion & class-filtered `taken_for_hers` reversion**: `2026-10-25_pass_contamination_and_mask_step_forensics.md` and `docs/masked_v2v/AGENTS.md` §10 described `4921-4926` as a false `strangers` self-swallow and `449b17f1`'s class-filtered `taken_for_hers` (`Apparel` + non-subject `face_ids`) as the fix. In fact, on `4922-4925` `second` is occluded behind the woman in blue (`strangers` pixels are the woman in blue), on `4940` (`1,907 px`) only her face is visible because her hair is already behind the hat brim, and on `4912h` class-filtering `taken_for_hers` allowed same-colour bystander hair (`Hair` class) on `4943-4946` (`2,381-3,777 px`) to be claimed as `second`'s hair. `her_hair_in_hole` reverts to subtracting every other person's mask (`~taken[k]`) and settling occlusion gaps via `head_tube` and owner `head_marks`.
  4. **`H3_MODEL_STORE` in `2026-10-25_pass_contamination_and_mask_step_forensics.md` §5.1**: previously listed `<comfy-root>/models` as the fallback; corrected to `$H3_MODEL_STORE` (`<model-store>`, never `<comfy-root>/models`).
  5. **Shot `2050-2622` second half in `docs/wiki/handover.md`**: previously listed a `573`-frame chained load over `2050-2622` with `--drop 2249-2270`; corrected to match the owner's 2026-10-25 ruling below (`lead_fp_2050g` stands on `2050-2248`, and the second half `2271-2598` renders as a single `345`-frame window at `2254-2598` laid on `2271-2598`).
  6. **Disproved Pass-1 DiT background-erasure and SAM-contamination claim on `4647-4725`**: `docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md` (§2.1) and `docs/masked_v2v/AGENTS.md` (§1, lines 49-56) claimed that `third_fp_4647e`'s 48 px regenerating region (`tokens`) erased `second`'s face and white collar in the Pass-1 cut (`fun_0000_6394_joined_2026-10-25T1903Z_from_laid.mp4`) and caused SAM 3.1's `second_track` to collapse to `0 px` on `4697` in `shot_c_4647j`. Measured directly against disk (`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md` §0.1-0.2, §0.4; `docs/masked_v2v/verifying_raw_frames_and_keeps.md` §2, §7-8): `bench/assemble_delivery.py` (`Matte.at`, line 898; `laid_by_of`, line 915, `rule: "matte"`) lays a piece by the node's saved per-frame `matte` (`*_window_*_region.npz["matte"]`, which lays the latent step's region on every frame of the step whose own mask is non-empty, and `0` on a frame with no mask); on `4647-4725`, `third_fp_4647e`'s `matte` was `0` on `second_track` across all 79 frames, and inside `second_track` the cut differed from the raw film by single digits. The `4697` track collapse in `shot_c_4647j` was caused by an unwithdrawn approximate cast box on `4698` (`cast.jsonl` line 186, drawn on `2026-10-23` and withdrawn at line 210 on `2026-10-26`), which was refused on `shot_c_4647h` (`0.42` inside at most, `data/2026-10-24_shot_c_4647h_job/preview.queued.json`; `0.49` was load frame 50, the withdrawn `4697` box) and taken on `shot_c_4647j` (`0.56` inside, `0.16` coverage on person 3), splitting her track and emptying `4697`. Both files are updated in place.

- **A larger margin on the third subject went through again, on `third_fp_5114d` (2026-10-26), and the owner's word on
  what the 2026-10-23 margin was.** The load was built with `--margin 64` (`data/2026-10-26_third_fp_5114d_job/job.render.json`,
  `defaults.margin`, `margin_by` by the Gemini lead at 19:20Z, citing the beard rule of `data/fun/AGENTS.md` and the
  owner's 5114b note on 5129) against the subject's recipe of 16 in `data/fun/plan.json`; the render drew `<Video 1>`'s white
  eye and nose lines on his face, as 5114c at 64 had on 2026-10-23. **The owner's word, 2026-10-26: the beard margin of
  2026-10-23 was a one-off, ad hoc thing**, tried that afternoon and reversed that evening by the 5675 pair (the entry
  below); it was never a recipe, a rule or a setting to carry, and no document presents it as one. Three things let it
  through: (1) `bench/flags_before_render.py::recipe_check_of` marks a knob that differs from the recipe `overridden`
  whenever its `_by` carries any non-empty `why`, and the tool refuses only `unexplained` rows, so a reason the session
  writes itself is authorization; (2) the rule is prose, at lines 55-57 of `data/fun/AGENTS.md`, three lines under the
  beard rule the reason cited; (3) the owner's 5114b note (`data/fun/owner_answers.jsonl` line 1559) carries the session's
  inline answer of 2026-10-23 that the margin is the way, and nothing beside it says that answer was reversed. The beard
  it was meant to cover is covered without it: the jaw pad under `source_facial_hair` (`4cebb0aa`, 2026-10-24, after
  5114b) keeps thousands of beard pixels a frame on 5126-5129 in 5114d's capture where 5114b's kept none
  (`subjects/third/head_tube__<by>.json`, `source_facial_hair.kept_px`). A reader's figure, reproduced from each load's plan record on the share
  (`<session>/<load>/<load>_plan/<load>_windows/<load>_plan.json`, `regenerating_share`): the white lines follow the
  regenerated share of the zoomed canvas, not the margin alone. Clean at 16 (4479e 0.047, 5675c 0.070, 5114b 0.078) and
  at 64 where his head is small in the frame (4479d 0.096, which closes the 2026-10-23 entry's open question); the lines on
  every load at 0.15 and over (5011e, 5461d, 5114c, 5114d at 0.20, 5675f at 0.26). The margin is the knob that moved it. The Gemini lead's `--no-face-on 5114-5129` as a remedy is withdrawn: on a face-only
  subject that word empties the part (`docs/masked_v2v/AGENTS.md` §2); the words that drop the lines and keep the mask are
  `reference --plain WHY` and `job --no-draw-face-on`. What changes: a knob a decision settled lives in the subject's recipe
  in `plan.json`, `job` refuses a command-line value that differs from it in the tool's own words with no reason route, the
  one kept exception (second_hfp_1093k at 128) is written in the plan's `shot_overrides` with the keep it cites, and the gate
  reads the plan; built in a worktree and landed between steps (`cbd39fbe`). Corrected today for what they used
  to claim: `docs/wiki/handover.md` ("pre-render recipe diff gate"), `docs/masked_v2v/PROTOCOL.md` 3.2 and
  `docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md` §6 ("enforces pre-render gates"), which each
  called the recipe diff a gate: it prints a diff and refuses only a difference with no reason. 5114e is the same load at 16.

- **Hand-drawn boxes on pasted screenshots are feedback to guide verified SAM detections, never direct tracker seeds; `/who/` page clicks (`1-174`) and session-verified SAM detection boxes (`197-209`) stand; and what both sides verified on `second_hfp_4647h`** (the owner's word, 2026-10-26). `data/fun/cast/cast.jsonl` has three groups of entries:
  1. **Group 1 (`lines 1-174`)**: the owner's clicks on the `/who/` HTML page — **valid and stand**.
  2. **Group 2 (the owner's annotated screenshots, rough sketches, and confirmed words about who or what segment—`hair + face`, `hair`, etc.—is visible on a frame, after the session reads back what it heard and what it sees)**: recorded as ground truth (`plan.json` `head_marks`, the shot's record, `owner_sheets/`) so the owner's eyes and word are never lost, and used to inform tracker changes, SAM 3.1 prompting/labeling, part/label logic, and Group 3 verification (avoiding manual frame rules wherever the pipeline can see it directly). The rough hand-drawn rectangles from the 2026-10-23 screenshots (`data/fun/owner_sheets/2026-10-23_4647_who_is_who_owner.webp` and `2026-10-23_4912_who_is_who_owner.webp`) that were copied directly into `cast.jsonl` (`lines 175-192` and `194-196`, two already withdrawn at `193` and `210`) are **never included directly as raw box seeds**.
  3. **Group 3 (`lines 197-209`)**: exact SAM person detection boxes (`2292 P6`, `2325 P1`, `2436 P3`, etc. on `2050`) verified by a session against the SAM masks/detections using the owner's Group 2 visual cues — **valid and stand**.
  Sequencing: on `4647-4725` the owner's only page click on `second` is frame 4666 (`line 128`), before the occlusion, so withdrawing Group 2 (`lines 175-192` and `194-196`) tonight with nothing in their place would lose her from 4682 on; the withdrawal of Group 2 (`175-192` and `194-196`) goes with tomorrow's tracker build,
  where re-finding her after an occlusion is the tracker's own job (its head detection per stretch with `--head-per-stretch`
  on by default, and the signature match), replayed on 4647 and 4912 before anything renders (with one click on her outline
  on 4690 as the fallback if the match cannot re-find her). On `second_hfp_4647h` (`4647-4725`): the first half (`4647-4673`)
  replaced her face and hair with blonde hair cleanly (`+36.9` BGR luma increase), proving `2.17x` zoom in the single
  79-frame window swaps her to blonde once unmasked dark hair is removed; she reverted to dark hair after the occlusion
  because (1) on `4682-4685` (`t=11`, `0` regenerated tokens) her person track (`765-1,082 px`) held only her bare forearm
  (`y=297-359`), so crop-Sapiens cropped only her forearm (`y=289-367`, `608-2,123 px` false-positive `Face_Neck` on her
  arm, suppressed by `head_tube` with `0` face points; `p_8px` still held `457-1,860 px` of that arm crop) while her emerging
  head edge on `4684-4685` had no track or mask and the `hidden` mark (`4674-4685`) zeroed the step; (2) on `4686` and
  `4693-4695`, because `--head-per-stretch` was off and `head_track` was `0 px` on `4682-4695`, `head_tube` (`off_tube_pose`)
  sliced `945-1,354 px` off crop-Sapiens's valid head/hair mask (`1,271 px` lost in `parts_held` on `4686`, `945-1,135 px`
  on `4693-4695`), while on `4688, 4691-4693` `parts_held` did hold `726-1,460 px` of `Hair` (class `4` in
  `classes__shot_c_4647k.npz`); and (3) on `4686-4693` (`t=12-13`) the laid matte reached `40-70 px` above her head into the
  dark-haired bystander woman SAM had detected (`2,000-3,500 px` in `shot_c_4647k_people.npz` at scores `0.54-0.81`) while
  `strangers.mkv` held `0 px` there because whole-frame `Sapiens2` missed her in `bench/strangers.py::run`. Tomorrow's
  builds (with no renders on the card, replayed on `4647k`, `4912h` and `1093` first): retire `hold_to_track` in favor of
  her Sapiens face+hair crop mask unioned with her head track (never unioning `Upper_Clothing`/`Torso` from `person_track`
  into a face+hair part, and anchoring the head room to actual head detections rather than arm-only crops) less every other
  person's SAM detection; include SAM's other-person detections in `strangers.py` where whole-frame Sapiens finds nobody;
  and re-find her after an occlusion in the tracker (`--head-per-stretch` default + signature match) while withdrawing the
  screenshot boxes.

## 2026-10-25

- **`video_mask.py::zoomed_source` docstring (`2061-2062`) and tooltip (`2396`) lost to commit `79831be9`; `two_shot_load.py`
  whole-frame forcing is a tool-side choice; and `1730-2049` was rendered at `2.0x` zoom, not `1.0x` whole-frame** (verified
  against `video_mask.py:2048-2146`, `git show 79831be9 -- video_mask.py`, `bench/two_shot_load.py:93-114`, and
  `data/2026-10-13_lead_fp_1730..2013b_job/*.render.json` node `104` on 2026-10-25):
  1. `video_mask.py:2061-2062` (`"a load with a cut inside it is refused: a box held across a cut frames nobody on one side
     of it"`) and `video_mask.py:2396` (`"a load with a cut in it is refused"`) are stale prose: commit `79831be9`
     (2026-10-21) deleted the 5-line `if cuts: raise ValueError(...)` check from `zoomed_source` so padded context frames
     could cross a cut under zoom. Consequently, `bench/two_shot_load.py::WHOLE_FRAME_WHY` (`lines 93-95, 113-114`) is now a
     tool-side choice rather than a node refusal.
  2. In `video_mask.py:938-973` (`cut_gate`, `unlaid_frames`) and `1005-1009` (`lay_window`), `cut_gate` only zeroes a
     sub-segment `(a, b)` where `present[a:b].any()` is `False`, so every frame in `across` already has an empty mask and
     would land in `bare` (`across + bare`) regardless; a live-detected cut inside a window changes only the plan JSON label
     (`frames_across_a_cut` vs `frames_with_no_mask`), not a composite pixel.
  3. **Correction to the `2026-10-25` entry below on `1730-2049`**: that entry claimed the 8 `2026-10-13` pieces
     (`lead_fp_1730..2013b`, rows `R15-R22` of `data/fun/laid.jsonl`) were rendered at `1.0x` whole-frame (because the first
     check looked for node `61` rather than node `104` in the `2026-10-13` graphs). In fact, node `104`
     (`MiniMaxH3MaskedSource`) in all 8 `.render.json` files and `*_00001_zoomed.json` ran with
     `render_region: "zoomed in on what is replaced"` at **`2.00x` (`512x384` crop -> `1024x768`)**, and her phone circle
     jumps across opposite quadrants at every cut (`[512,316,1024,700]` on `1730`, `[512,28,1024,412]` on `1755`,
     `[512,288,1024,672]` on `1802b`, `[0,4,512,388]` on `1849b`, `[0,78,512,462]` on `1898`, `[512,340,1024,724]` on
     `1945b`). Single-shot zoomed loads (`39` and `90` windows) therefore preserve `2.00x` zoom (`4x` area) on each phone
     circle, while `bench/two_shot_load.py` is kept in `bench/` for the niche of two ultra-short cuts (`< 39f` combined).

- **Unified shot, window, and seam rule: no mid-face splices, natural cuts at multi-frame `0 px` off-screen gaps inside
  long shots (`2050`), and single-shot zoomed loads as the default** (the owner,
  2026-10-25 about 16:04Z and 17:02Z). Refines the 2026-10-15 "one shot, one load" rule (`docs/masked_v2v/AGENTS.md`) and
  path 2 of the 2026-10-25 window rule around the physical causes of seams and identity bleed:
  1. **No mid-face splices**: two separate renders of the same subject never meet on consecutive frames where the subject
     is visible (`2623-2711` is remade whole in a `90`-frame window with `seed: 730451892` rather than splicing `2623-2624`
     onto `lead_fp_2625`; `2823-3109` is remade in one `294`-frame window to eliminate the `3097/3098` splice between
     `lead_fp_2823b` and `lead_fp_3098b`).
  2. **A multi-frame `0 px` off-screen interval inside a shot over `345` frames is a natural cut**: on `2050-2622`,
     `lead_fp_2050g` (`2050-2248`, `199f`, kept 2026-10-24) and the second half (`2271-2598`, `328f`, after which she is
     off-screen on `2599-2622`) are separated by `22` frames (`2249-2270`) where her head is turned away (`carried == 0 px`,
     untouched original video), so the two halves never touch (`0` seam). Keeping `lead_fp_2050g` on `2050-2248` and
     rendering `2271-2598` in a single `345`-frame window placed at **`2254-2598`** (starting inside the `2249-2270` `0 px`
     spin gap, ending on her last tracked frame `2598` so post-shatter frames `2599-2615` never enter the window or box, and
     laid on `2271-2598`) is approved by the owner.
  3. **Zoom nuance across close-ups vs distant shots**: `render_region: "part"` (`zoomed in on what is replaced`) enlarges
     small faces up to the target box (`2.0x` default, `4.0x` for `second`), and on close-up shots where the subject's part
     with room already exceeds `1 / ZOOM_SMALLEST` (`1.25x`) of the canvas (`video_mask.py:1963-1965, 2109-2113`),
     `zoom_box` automatically returns `None` and renders at `1.0x` whole-frame without manual override.
  4. **Mask end to end, chop at the lay**: every window runs to its legal `both_clocks` length (`39`, `90`, `141`, `192`,
     `243`, `294`, `345`) with zero unmasked frames of the subject inside the window, and `laid.jsonl` chops what the cut
     does not need.

- **Expanded post-quality-jump remake plan across all 64 laid rows (`R01-R64`) and subject-specific invariants** (the owner,
  2026-10-25 about 15:00Z, 15:56Z, and 17:02Z). Reverses the 2026-10-15 rule that a kept piece is not remade for a better
  form. Full disk audit of all 64 rows in `data/fun/laid.jsonl` (`R01-R26` pre-`10-20` = `26` pieces / `2,032` laid frames;
  `R27-R41` `10-20` to `10-22` = `15` pieces / `860` laid frames; `R42-R64` `10-23` to `10-24` = `23` pieces):
  1. **Why pre-`2026-10-23` pieces (`R01-R41`, except late `10-22` `R39` and `R40`) and 11 faulted `10-23`/`10-24` pieces
     are queued for remake**:
     - `R01, R03-R08, R10-R25` (`10-13` to `10-15`, `23` pieces) used `"only what changed"` composite (`0.05` luma bleed),
       non-clock or song-padded `141` windows, `face` crop, no `--closed-lip-line`, and trimmed shot edges (`0294d`,
       `1061p`, `1097c`, `1802b`, `1898`, `1975b`, `2013b`, `2625`, `1466b`/`1575b`, `2823b`).
     - `R02, R09, R26` (`10-17` to `10-18`, `3` pieces) predated `b270be7e` (mouth smudge retirement), `both_clocks`, and
       `cranium` crop (`R09` also trimmed `1408-1418`).
     - `R27-R38` (`10-20` to early `10-22`, `12` pieces) predated `hold_context` (`R36 second_hfp_4602d`, `R37
       second_hfp_0415n`, and `R38 third_fp_4602c` ran with unmasked real frames across cuts in `pad_before` and
       `pad_after`), `cranium` crop, `4.0x` zoom on `second`, and plain `"blonde hair"` text (`R28 second_hfp_1010g` also
       ran Pass 2 on raw source).
     - Among `R42-R64` (`10-23` to `10-24`), `12` pieces stand cleanly (`R42, R43, R44, R46, R47, R49, R50, R51, R53, R54,
       R60, R61`), while `11` have specific window/seam/mask faults (`R45` unconfirmed smile; `R56, R57` `held_fell_back`
       unmasked tail; `R58` `3097/3098` splice; `R62` `9f` unmasked tail; `R52` `17f` empty; `R63` `12f` empty; `R55` Pass 2
       baked on `10-14` `R05`; `R59` `2f` empty; `R48` `1f` empty; `R64` partial `2050-2248`).
  2. **Subject-specific invariants enforced across every remake (`data/fun/plan.json`, `data/fun/cast/cast.jsonl`)**:
     - **Subject 1 (`lead`)**: Face swap only (`region: "face"`), ALWAYS keeping accessories (`Eyeglass` sunglasses on
       `3063-3109` and `3909-4052`, `Left_Hand+Right_Hand` on `3060-3109`, `Apparel@head` hat on `168-184`), translated to
       the rendered face's coordinates at assembly (`assemble_delivery.py` `WORN_ON_THE_HEAD`).
     - **Subject 2 (`second`, mom)**: Face AND hair swap (`region: "face and hair"`, blonde hair in `fun_mom_edit.jpg`
       replacing the original actress's dark bouffant hairdo; plain `"blonde hair"` with no hair-shortening wording;
       `--zoom-up-to 4`), accounting for small distant appearances and intermittent foreground occlusions (`no_face_on` /
       `head_hidden`).
     - **Subject 3 (`third`, dad)**: Face only (`region: "face"`), clean-shaven (`fun_dad.jpg`, `source_facial_hair: true`
       -> `facial_hair=keep` + `JAW_BEARD_PAD_FRAC = 0.35` downward jaw pad at default `margin: 16`; no beard/stubble words
       in text). In the hallway sequence (`1222-1418`, specifically `1222-1305` and the first half of `1306-1418` around
       frame `1300` where he raises his hand pointing at the lead), `third` is the one mouthing/speaking the sung vocal line
       even though the vocals are female (`mouth_drawn_open` / mouth opening and closing transfer in the prompt).
  3. **Multi-pass execution order**: every multi-subject shot executes in strict pass order (`laid.py remove` old pieces ->
     rebuild cut -> Pass 1 on raw working copy -> rebuild cut -> Pass 2 on cut with `--for` -> rebuild cut -> Pass 3 on cut
     with `--for`), preserving all `cast.jsonl` seed boxes, `who_is_who`, `head_marks`, and `restore=` spans.

- **Padding frames never show the original subject unmasked in a window; the three-path window rule** (the owner,
  2026-10-25 about 00:10Z to 00:20Z; recorded in
  `docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`). On `lead_fp_2050c` (`198f` in a `243f`
  window) and `lead_fp_2050e` (`45f` in a `90f` window), the hole was blanked (`0.04%-0.16%` of pixels unchanged inside
  `parts_held`, `79.8%-85.9%` changed by `> 5 / 255`) and H3 followed `<Video 1>`'s expression marks, yet synthesised the
  original lead singer's face instead of `<Picture 1>`'s. Every kept lead render had zero unmasked frames of the singer in
  its window (the two pieces first named as counterexamples, `3098b` and `4053c`, were withdrawn on the frames once their
  padding showed she had already left the frame and the tracker had drifted onto a bubble, another woman, and backup
  dancers, bug 4 of 2026-10-23); `2050c` and `2050e` had `21` (`2272-2292`) and `45` (`2095-2139`) unmasked frames of the
  singer in their padding. The owner's ruling:
  1. **A shot up to one single-window ceiling (`345` frames on both clocks)** runs as one load to its legal window length
     (`39`, `90`, `141`, `192`, `243`, `294`, or `345`; `362` is text-to-video's ceiling and ends with the sound off its
     last frame by a third of an audio latent), denoised with the subject's tracked mask and `<Video 1>` end to end across
     the entire window, and is chopped to the shot's span at the lay (`laid.jsonl`'s `frames`).
  2. **A shot longer than `345` frames** runs as chained windows inside one load, each re-denoising a `90`-frame overlap
     against the previous window's rendered frames and tossing the overlap, as chained windows already do.
  3. **The padding side follows the footage**: masked real frames after the shot where the footage continues (such as
     `2050`'s mirror spin), and held copies of the shot's last frame with its mask open across a hard cut (preserving the
     2026-10-22 seam rule of `4479e`). Nothing goes in front except a shot at the source clip's end.
  What it reverses and what the prose used to claim: `docs/masked_v2v/AGENTS.md` ("One shot, one load... Padding goes after
  the shot, as filmed when the footage continues and held at a cut"), `docs/masked_v2v/frames_rules.md` (§4, "The padding
  frames are real frames of the clip when the shot goes on past the load... Only the shot's own frames carry its masks"),
  and `docs/h3_resolutions.md` ("Padding goes after the shot... real frames where the shot goes on past the load") each
  claimed that padding inside continuous footage runs unmasked "as filmed" (the 2026-10-21 default kept on 2026-10-22 for
  the mom's lowered gaze beside the shot). Updated in all three documents to state the three-path rule and link the
  research record.

- **The 2026-10-24 handover's prose lost to the files** (Gemini Spark's close-of-day read, checked line by line by the
  Claude lead on 2026-10-25). What it used to say, and what the files hold: `bench/_lib/name_taken.py` (the function is
  `bench/shot_to_load.py::name_taken`); `bench/review_server.py look` (the look items are `bench/review_pages.py` and
  `bench/load_chain.py::look_page_item`); a `nodes/` folder (the node files sit at the repo root); `JAW_BEARD_PAD_FRAC =
  0.25` (`bench/head_tube_parts.py` holds `0.35`, and `0.25` is `video_mask.HELD_EDGE_LEAST`); file names `in_frames`,
  `ref_frames`, `mask_frames`, `mask_review` and `raw` that no node writes (the names are in
  `docs/masked_v2v/verifying_raw_frames_and_keeps.md`); and the spans `5675-5762`, `5461-5525`, `4310-4374` and
  `1306-1384` for pieces whose laid rows read `5675-5715`, `5461-5497` and `4310-4389`, and whose removed row,
  `third_fp_1306g`, read `1306-1418` in the `0045Z` table and in the owner's keep and `wrong` answers. Its line for
  `1306h` said the `2341Z` cut holds the lead and the second on 1306, where `data/fun/laid.jsonl` has the lead alone,
  `lead_fp_1306t` on `1306-1407`. Spark's own read put 1306g's laid span at `1306-1396`, which no record holds. None of
  it touched the cut, the laid rows or the three worktrees. Why `bench/check_doc_links.py` passed: it reads markdown
  links, `path:line` citations and `path::Symbol` references, not a bare path in backticks. The day's changelog
  fragment was also not written at the close (the Claude lead's job, `changelog.d/README.md`); written on 2026-10-25
  from the 16 commits.

## 2026-10-24

- **`meta_perception_models.md` brought up to the chain** (mrfabez, a Claude session advising; docs only). It used to
  say: nothing in the pack reads depth; the keypoint model was tried once by no node; only lid marks had been drawn and
  nothing for a mouth; the preflight rules were not written; MoGe's own field-of-view estimate was never run here. Each is
  overtaken by code the page's dated notes of 2026-10-24 now name (the pose pass, `--depth`, the face drawing and the
  mouth gate, the capture's `flag_pose`, the 2026-10-18 depth record). The reading behind it, Gemini Spark's
  perception-models review of 2026-10-23 checked claim by claim, is
  `docs/research/masking/2026-10-24_mrfabez_perception_models_response.md`. Of its remedies: two name what the day's plan
  already does (the pose-table printout; a beard rule off the class table), one is already on (memory selection), and
  the two model changes (a text-prompted beard; joint multi-object tracking) run against the 2026-10-04 tracker decision
  and the one-object memory correction. The one new item is a reader of the tracker's saved per-frame score along a span,
  tested on the two drift cases before it is anything.
- **Corrected: why 1419's head was still lost after the restart.** The 2026-10-23 entry said "the head seed's size
  test has no fallback". Main's span head track does fall back to the top third (`subject_track.py::_span_head`), but
  on 1419 it seeded from a real head on 1423 (1,436 px) and lost it across the empty run 1426-1430, so the top third
  never ran. The stretch's own size test compared her 1,901 px outline on 1431 with only the nearest headed frame,
  1425, occluded at 366 px, and stopped there (`subject_track.py:1237`). Traced in
  `internal/gemini/2026-10-23_post_restart_findings.md`; bug 22 of the 2026-10-23 bugs file.
- **The 2026-10-23 handover's next-day order, revised after the reviews of the close** (Gemini's end-of-day review and
  the Claude review of it): the name check fixed before any new job; the previews that need no new code (3967 and
  3240 among them) moved ahead of the builds; the beard test preceded by a class print from the preview's own class
  video and a read of the drawings' enlargement (the larger margin may not be the whole cause of the white lines,
  bug 27); `--hold-part` dropped for 1419 (it holds a part unmoved, and she walks); and the chin guard's place made
  explicit, since 5114i waits on it. The owner's order, the landed tools before new tracker code, is unchanged.
- **The same order, by pass** (Gemini Spark's check of the multi-pass chain, at the owner's question; each shot checked
  against `data/fun/laid.jsonl`). The revised order had listed previews with no source. Now each is on its pass's
  source: the third's 5675, 5011, 5461 and 4479 (the second laid on each) and the second's 4647 (third_fp_4653b
  laid) and 1419 (the lead and the third laid) on the newest cut with `--for`, and the third's beard shots join the
  early previews. The third's 5114 waits for second_hfp_5114i's remake and a rebuilt cut: made on the stand-in, it
  would be made twice. Three of the review's shot facts did not hold: on 4647 the laid piece is the third's, not the
  lead's, so the second is pass 2 there, not the third pass 3; nothing is laid on 4774 or 4912. A second Claude
  review added: `check_pass2_source` does not catch an older cut; and if the chin guard does not land, the owner is
  asked whether the third's 5114 goes on the stand-in. Its order for 4647 (the third after the second) is reversed
  here: third_fp_4653b covers 4653-4677 of 4647-4725 (`data/fun/laid.jsonl:25`, `frames`), so the third's next load
  there is the kept subject's and goes first. The 0045Z rebuild's coverage owes the lead, the second and the third on
  4647, so it is a chain of three; the lead and the second follow the third's kept piece and a rebuilt cut, in the
  order the owner says. The per-shot source list moved to the plan's section 3; the handover keeps the rule. Then the
  owner, asked whether the third's later frames and the lead on 4647 are to be replaced: not known yet. So 4647 leaves
  the next day's order except for entering the third's boxes. The lead's question had been answered on 2026-10-21
  (she is replaced there, `data/fun/owner_answers.jsonl:1413`), so only the third's later frames and the order wait.
  Proposed and withdrawn the same night: a second Claude review proposed laying a new third load around
  third_fp_4653b (on 4647-4652 and 4678-4725, with seams at each end of it), citing the lead's two pieces on 1466-1729
  (`data/fun/laid.jsonl:12-13`) as precedent, and the handover and row r24_01 took it in. It breaks the frame rules
  (2026-10-15, one shot is never split; `docs/masked_v2v/AGENTS.md`, zero mid-shot seams), and the owner asked that
  4647 align with them. So his load runs and is laid over the whole shot, replacing third_fp_4653b once a new render
  is kept; 4653b is a split piece standing from before the rule, which the rule says is remade whole, put to the owner
  first. The lead's 1466-1729 pair is another standing split, owed a whole-shot remake the same way (not yet asked).


## Earlier Decisions by Date (`2026-09-11` through `2026-10-23`)

Sharded into `< 40 KB` files under `docs/wiki/decisions/` so every file fits in a single `read_file` call:

| Date Range | File |
|---|---|
| `2026-10-23` | [`decisions/2026-10-23.md`](decisions/2026-10-23.md) |
| `2026-10-22` | [`decisions/2026-10-22.md`](decisions/2026-10-22.md) |
| `2026-10-21` | [`decisions/2026-10-21.md`](decisions/2026-10-21.md) |
| `2026-10-18 through 2026-10-20` | [`decisions/2026-10-18_to_2026-10-20.md`](decisions/2026-10-18_to_2026-10-20.md) |
| `2026-10-16 through 2026-10-17` | [`decisions/2026-10-16_to_2026-10-17.md`](decisions/2026-10-16_to_2026-10-17.md) |
| `2026-10-15` | [`decisions/2026-10-15.md`](decisions/2026-10-15.md) |
| `2026-10-14` | [`decisions/2026-10-14.md`](decisions/2026-10-14.md) |
| `2026-10-13` | [`decisions/2026-10-13.md`](decisions/2026-10-13.md) |
| `2026-10-12` | [`decisions/2026-10-12.md`](decisions/2026-10-12.md) |
| `2026-10-11` | [`decisions/2026-10-11.md`](decisions/2026-10-11.md) |
| `2026-10-10 (Part 1)` | [`decisions/2026-10-10_part1.md`](decisions/2026-10-10_part1.md) |
| `2026-10-10 (Part 2)` | [`decisions/2026-10-10_part2.md`](decisions/2026-10-10_part2.md) |
| `2026-10-05 through 2026-10-09` | [`decisions/2026-10-05_to_2026-10-09.md`](decisions/2026-10-05_to_2026-10-09.md) |
| `2026-10-01 through 2026-10-04` | [`decisions/2026-10-01_to_2026-10-04.md`](decisions/2026-10-01_to_2026-10-04.md) |
| `2026-09-25 through 2026-09-29` | [`decisions/2026-09-25_to_2026-09-29.md`](decisions/2026-09-25_to_2026-09-29.md) |
| `2026-09-17 through 2026-09-23` | [`decisions/2026-09-17_to_2026-09-23.md`](decisions/2026-09-17_to_2026-09-23.md) |
| `2026-09-11 through 2026-09-15` | [`decisions/2026-09-11_to_2026-09-15.md`](decisions/2026-09-11_to_2026-09-15.md) |
