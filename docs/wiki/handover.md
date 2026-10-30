# Handover: read this before anything else

Rewritten 2026-10-29 at `18:22Z` by the Gemini lead (`gemlead`).
Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
[`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md),
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), and `data/fun/`.

## State when closed

- **8 pieces kept and laid today (`2026-10-29`, `data/fun/laid.jsonl` at `78` rows, `sha256 0d7cfe6d2d59bc83`, `0` missing or marked-wrong keeps)**:
  - Fixed `lead_fp_2254c` (`2254–2598`, row 75) in `laid.jsonl` (`first: 2050, last: 2622, frames: [[2254, 2598]]`), rebuilt the whole-video cut `fun_0000_6394_joined_2026-10-29T1450Z_from_laid.mp4`, and staged it on `:8765`.
  - Laid 7 new owner keeps from `:8765`: `lead_fp_0438b` (`438–483`, `data/fun/owner_answers.jsonl:1677-1678`), `lead_fp_1134b` (`1134–1155`, `data/fun/owner_answers.jsonl:1679-1680`), `lead_fp_4272c` (`4272–4309`, `data/fun/owner_answers.jsonl:1681`), `lead_fp_4053d` (`4053–4091`, `data/fun/owner_answers.jsonl:1682-1683`: *"loosk good!"*), `lead_fp_3240f` (`3240–3276`, `data/fun/owner_answers.jsonl:1684-1685`: *"looks good!"*), `lead_fp_4774c` (`4774–4826`, `data/fun/owner_answers.jsonl:1690-1691`: *"looks great!"*), and `lead_fp_3476c` (`3476–3523`, `data/fun/owner_answers.jsonl:1692-1693`: *"looks amazing! you got it at all the transitions!!"*).
- **2 earlier `second` keeps reopened by the owner and removed to `data/fun/laid.removed.jsonl` (`rows 15–16`)**:
  - `second_hfp_0014g` (`14–167`, `154f`, `data/fun/owner_answers.jsonl:1686-1687`: *"janky, artifacts around the mask edges throughout 14-167... Remake on a fresh preview with current tooling."*). Fresh preview graph `data/2026-10-29_shot_c_0014f_job/preview.json` is built on CPU and ready to queue.
  - `second_hfp_0278m` (`278–293`, `16f`, `data/fun/owner_answers.jsonl:1688-1689`: *"Her eyes are open in the render. In the source they are not. Doesn’t match the input on 278-293."*). Fresh preview graph `data/2026-10-29_shot_c_0278g_job/preview.json` is built on CPU and ready to queue; build `second_hfp_0278n` with `--eyes-shut-on 278-293 --eyes-shut-form closed` (`sculpted_eye`).
- **All 7 stale 10-26 previews refreshed on the GPU today (`2026-10-29`)**:
  - `shot_c_5044f` (+ `bench/strangers.py`), `shot_c_4774f` (+ `bench/strangers.py`), `shot_c_3476f` (+ `bench/strangers.py`), `shot_c_4227f` (+ `bench/strangers.py`), `shot_c_3240f` (+ `bench/strangers.py`), `shot_c_3595f` (+ `bench/strangers.py`), and `shot_c_6105g` (`14/14` active sub-shots seeded after resolving all 9 `who_is_who` stretches in `data/fun/shots.json` and `data/fun/cast/cast.jsonl`).
- **`lead_fp_5049b` (`5049–5104`, `56f` of `5044–5113`) built through `reference` + `text` (`PASSED`, `3.879x` cranium crop)**:
  - `data/2026-10-29_gemlead_jobs/lead_fp_5049b.json` and `$H3_OUTPUT_DIR/Video/fun/references/lead_5049_5104_b/lead_head_reference_5049_5104.mp4` (`passed: true`, `0.371 px` read-back) are ready on disk. Next step: `bench/load_chain.py --by <session> first data/2026-10-29_gemlead_jobs/lead_fp_5049b.json`.

## What not to assume

- **Subagents for all video reading (`read_file` on `.mp4` is never called in the lead session)**:
  - Always dispatch an independent subagent (`invoke_subagent` / `send_message`) to read `.1fps.mp4` look strips before `render` (`bench/capture_masked_run.py looked`) and landed review clips (`.1fps.mp4` + `24 fps` `.mp4` + `parts_held` `Y`/`B-R` shift -> `<capture>/looks/<load>.reading.md`) between `land` and `page` (`093759dd`, `1e0d37f5`).
- **Record omitted edge spans via `bench/shots_look.py add`, never `--allow-partial-shot`**:
  - Whenever edge frames of a shot are left out because the replaced part is not visible (e.g., `4827–4841` on `4774`, `5044–5048` and `5105–5113` on `5044`), inspect a zoomed contact sheet of those frames and record `"not seen"` in `data/fun/shots.json` via `bench/shots_look.py add`.
- **Two shots where SAM 3.1 `head_track` locked onto hair and suppressed the face (`4647b` on `2026-10-28`, `4774c` frames `4806–4812` on `2026-10-29`)**:
  - On `4774c` (`4806–4812`), `lead` is full-face singing (`10,275–12,684 px` in Sapiens `Face_Neck`), while SAM's `head_track` held only `245–757 px` of bouncing hair (`off_tube_suppressed = 9,233–11,488 px`), caught by the pre-render reader before `render` and fixed with `--no-head-track` + `--head-tube`. Two candidate code fixes for a render-free build window: (1) seed the head track on a frame with face keypoints, and (2) `bench/head_tube_parts.py` should not suppress a face when the part model's face pixels are many times the head track's.
- **`4912–4946` (`second` + `third`) is owned by session `gem4912` (CPU root-cause analysis)**:
  - Do not build or edit `4912` until `gem4912` is ready for its landing/GPU slot; always check `git status --short` before committing.

- **Added 18:32Z by the Claude advisory session (`mrfam`), on top of the lead's close (`88bd0d15`)**:
  - **The cut on the review page is still the 15:12Z assembly** (`fun_0000_6394_joined_2026-10-29T1450Z_from_laid`, the
    eight keeps of 10-28 and nothing kept today). The 18:14Z run under `data/2026-10-29_fun_delivery_tables/` wrote a
    ledger, table and reading from the current rows but no clip and no rebuild record. The close rebuild with `--clip`
    is owed, and until the two remakes are kept it lays the original mom on 14 to 167 and 278 to 293: the ledger holds
    "kept" and "removed" and nothing for "wrong, but laid until replaced". The owner knows and has not ruled.
  - **The 4912 test lands in this order**: the Claude blind read of `gem/4912-hair` (`21753055`) is green with one ask
    (the zoomed-source case in `check_video_mask` compares content, not shape; `gem4912` commits it on the branch);
    then the landing by pathspec from the merge base on an empty queue; then the server restarted by the recipe in
    `docs/comfy_notes.md`; then `gem4912` runs `second_hfp_4912p` through every gate itself. Neither `gem4912` nor
    this page's reader restarts the server with a load on the card. The owner's rule: if it comes back dark, document
    and move on.
  - **Depth bleeds across an occlusion edge onto the re-emerging subject's first sliver** (4912 frame 4926,
    `internal/gemini/2026-10-29_4912/strip_4926_depth_who_is_who.jpg`): the keep-in-front margin, a fixed constant,
    cannot separate them, and the carve took the occluder's hair into the hole on that step. A general effect and a
    build candidate; untouched for the 4912 test on purpose.
  - **The owner's invalidation rule** (`decisions.md`, 2026-10-29): a change in the shot record, the plan or the job
    that a mask or render depends on invalidates it; nothing made from the old input is carried. Writer stamps enforce
    it for code; nothing yet refuses a capture whose `shots.json`, `cast.jsonl` or `plan.json` words changed after it.
  - **The outgoing lead compacted at least six times** and its last messages looped on a task poll; the owner replaced
    it. Its reconstruction from the `agy` transcripts is the fresh session's subagent's job, as a postmortem record,
    and a cross-check on this page. The Claude day file with the 10-28 postmortem, today's plan, the check table and
    every correction is `internal/claude/2026-10-29_mrfam/postmortem_and_plan.md`.

- **Added 2026-10-30 01:55Z by the Claude advisory session (`mrfam`), the evening's close, on top of the lead's close and the
  18:32Z block** (`gemrender` stopped at 23:48Z with the card idle; `gem4912` drove 4912 on the owner's word until about 01:40Z):
  - **`second_hfp_0278o` is on the review page awaiting the owner's verdict** (sculpted mouth, `closed` drawing, by-hand
    shut-eyes text): the mouth stays closed on all 16 frames; the eyes hold on 278 to 286, lift on 287 and are open on 288 to
    293 while the source stays down over the bowl (the reading's measured frames; the backlog row says 286 to 293). Half the
    `0278m` complaint answered, half not; the open-eyed portrait pulls harder than the reference and the sentence on those frames.
  - **Every other backlog render is on hold until tomorrow on the owner's word to `gemrender` at 23:42Z** (`cee63a5b`):
    `0014h`, `3595c`, `4227c`, `5085`, `6105e`, the 3110 tail. Their CPU builds are on disk in `data/2026-10-29_gemlead_jobs/`.
  - **4912 after nine renders: the hole was a real defect and is fixed; the shot is still brown, and it is now a decision, not
    a bug.** `16ccd81a` (landed, `main`) joins her attached hair within the head band and prints `hair_outside_hole_share` per
    frame; `second_hfp_4912r` with that hole alone changed came back lighter and warmer (hair median L* 6.3 to 11.8, b* +9 to
    +16) and still well under every kept kitchen shot (26 to 39); `second_hfp_4912s`, a one-time test with another reference
    picture on the owner's word and the prompt unchanged, was no better (10.2, +14). What surrounds the hole from 4915, the
    dancer's hair and the man behind her, is what is left. Options for the owner: a split window around the dancer (4927 to
    4940 alone would say whether the first window drives it), a brown-haired version laid as is, or leave it. `4912p`, `4912q`,
    `4912r` are on the page with readings; `4912s`'s reading and page are `gem4912`'s last step. Figures and the code trace:
    `docs/masked_v2v/2026-10-29_findings_0278n_text_and_frame_table.md` section 3, `internal/gemini/2026-10-29_4912/
    root_cause_and_test_plan.md` 6.5 to 6.7, and the Claude day file.
  - **Do not use the picture from `4912s` again, and do not record it**: the owner's word, 2026-10-30 about 00:30Z. The only
    place it is named is that job's own file. `data/fun/plan.json` still names `fun_mom_edit.jpg` for `second`.
  - **The first `lead` or `third` capture after `16ccd81a` shows what the join adds for that subject** in the new column; the
    replay before landing covered the sixteen kept `second` loads only. The join adds hair pixels only, within the head band.
  - **The backlog's 4912 row describes `4912q` wrongly** as the eyes-shut and sculpted-mouth test; `4912q` was the held grey
    head reference test (`second_4912_4946_l`, no black frames). The lead's row to fix.
  - **The day's changelog is written** (`changelog.d/day-2026-10-29.md`, built as 0.300.0) and the decisions page carries the
    4912 line; the server on `:8188` was left for whoever closes `gem4912`'s last step (check `/proc/<pid>/environ` and the
    queue before stopping it by the recipe).

## Next session, in order

1. **Run `first..table`, pre-render subagent look, `render`, `land`, post-land `reading.md`, and `page` on `lead_fp_5049b` (`5049–5104`, `56f`)**:
   - Job and `PASSED` reference (`lead_5049_5104_b`, `3.879x`) are ready at `data/2026-10-29_gemlead_jobs/lead_fp_5049b.json`.
   - Note on `5059` (`3,361 px` back-of-hair) and `5078` (`184 px` hat-woman forehead sliver): both were visually verified on `zoom_5044_5103.jpg` to hold no visible `lead` face, so `--drop 5056-5059+5066-5078` stands.
2. **Queue the fresh previews and run the two reopened `second` remakes (`second_hfp_0014h` on `14–167` and `second_hfp_0278n` on `278–293`)**:
   - Queue `data/2026-10-29_shot_c_0014f_job` and `data/2026-10-29_shot_c_0278g_job` (`bench/masked_job.py queue ... preview --wait`), run `bench/strangers.py run`, and build:
     - `second_hfp_0014h` (`14–167`, `154f`): pre-render reader checks edge frames (`48, 66–69`, etc.) specifically; post-land reader compares mask edges against `second_hfp_0014g` as well as the source.
     - `second_hfp_0278n` (`278–293`, `16f` + post-shot padding): pass `--eyes-shut-on 278-293 --eyes-shut-form closed` so her eyes stay closed matching the source.
3. **Hold the landing/render slot for `gem4912` on `4912–4946` when ready, and continue the single-subject `lead` queue (`lead_fp_4227c` on `shot_c_4227f`, `lead_fp_3595b` on `shot_c_3595f`, and `lead_fp_6105e` on `shot_c_6105g`)**:
   - `shot_c_4227f_strangers.mkv` and `shot_c_3595f_strangers.mkv` are already written on disk; `zoom_4227_4271.jpg` and `check_3595_3671_all.jpg` are inspected (`--drop 4232-4234+4243-4257` for `4227c`; `--drop 3618-3621+3642-3644` for `3595b`).

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file), [`next_steps.md`](next_steps.md), and [`decisions.md`](decisions.md) (`2026-10-29`).
  2. [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), and [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md).
- **Laid ledger & active job files**: `data/fun/laid.jsonl` (`78` rows, `sha256 0d7cfe6d2d59bc83`), `data/fun/laid.removed.jsonl` (`16` rows), `data/2026-10-29_gemlead_jobs/lead_fp_5049b.json`, `data/2026-10-29_shot_c_0014f_job/`, and `data/2026-10-29_shot_c_0278g_job/`.
