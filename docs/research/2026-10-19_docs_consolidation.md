# 2026-10-19: what the docs consolidation cut, and where each piece went

**In short.** On 2026-10-19 `docs/wiki/` was dissolved into topic folders under `docs/` (`901bee1`), and then each
folder's pages were rewritten to lead with the answer, with superseded layers, finished items and duplicates cut.
This record is the list of every cut, folder by folder: what a page used to say and where its content lives now.
The text as it stood before the rewrite is in git at `901bee1`, at the same paths. The claims that were *wrong*
against the code or a newer record, not merely superseded, are also logged in `docs/state/decisions.md` under
2026-10-19.

Each section is one folder's log as its rewriter wrote it: moves, then removed or corrected claims, then what it left
for the lead (most of which the same day's integration commit acted on).

## docs/state/

Line counts, before -> after: next_steps.md 2668 -> 253; roadmap.md 1064 -> 227; open_experiments.md 3248 -> 78;
index.md 12 -> 12; decisions.md 4999 and handover.md 358 untouched. The full prior text of every page is in git at
`901bee1`. Ran the gitignored-link fixer on docs/state (0 repointed;
my rewrites already use `../../`). Checks run: `check_doc_links` on docs/state (only the three `data/results/` links
remain, environmental, plus the pre-existing handover/decisions failures), `build_wiki_index` (ok; `docs/h3/nodes.md`
unreachable, not this folder), `check_retraction_consumers` (ok: the two retracted figures of open experiment #15 kept
in open_experiments #15's closed row), `check_skill_routes` (ok), `check_prompt_docs_sync` (ok).

### Moves

none (no page renamed, merged or deleted).

### Removed or corrected claims

#### next_steps.md (top block, lines 1-23, kept verbatim)

- **The 2026-10-15 queue (items 1-8) and the 2026-10-14 queue**: superseded by `handover.md` "Tomorrow, in order" (the
  top block already says so) and by later decisions: the kitchen's second subject and the lead's kitchen are kept and
  laid (`second_hfp_0604h`, `lead_fp_0606`; decisions 2026-10-18); the dad's 1397-1418 remake and beard work dropped by
  the owner (handover, "Dropped by the owner today"); "one hole a pass" not built (decisions 2026-10-16); the 10-14
  tooling items 1-2 and the zoom floor done 2026-10-14. Kept as open rows: gaze (item 8), the gap print, break_checks,
  the box drawn on the strip.
- **"Her face-only pieces change her hair too" (2026-10-15)**: dropped by the owner, 2026-10-15 ~22:12Z (her hair was
  never an issue). Links: `internal/gemini/2026-10-15_watch_0604_1009_and_5675b.md`,
  `internal/claude/2026-10-15_mrlol_scripts/lead_hair_seam.txt`.
- **"The outside reviewer sees video at one frame a second"**: a working fact, not an open item; `data/AGENTS.md`
  covers playback questions to Gemini. Link: `internal/gemini/2026-10-15_clips_review.md`. (Lead: check it is stated in
  `data/AGENTS.md` or `docs/working/working_with_the_owner.md`.)
- **Strangers keep-out row (extend the hair rule to every core class)**: superseded by decisions 2026-10-16, "No
  keep-out gives back a cell inside the person's own track" (the capture carves the subject's track out of each
  keep-out). Links: `internal/gemini/2026-10-15_watch_strangers_0630d.md`, `data/2026-10-15_strangers_touching_0630d/`,
  `internal/claude/2026-10-15_mrlol_scripts/strangers_on_her.txt`.
- **"A last latent step part held does not damp motion"**: a finding (no planner change), not an open item. Link:
  `internal/claude/2026-10-15_mrlol_scripts/last_step_motion.txt`.
- **"Her hair tips shimmer at the blend's outer edge" (changed_alpha threshold)**: superseded by the one compositing
  law and "whole region" composite (decisions 2026-10-16/17). Link: `internal/gemini/2026-10-15_watch_hair_item4.md`.
- **Profile far side and stray lines drawn**: superseded by the drawing build (marks clipped to the grey head's
  silhouette, a far eye shrinks into the edge; decisions 2026-10-18, `8ac7e928`). Link:
  `data/2026-10-14_lead_fp_1466b/look_blind/reading.md`.
- **The who page numbers a shot on one frame (4647-4725)**: parked by the lead 2026-10-14 (14 frames of a far third
  subject); cut as parked. Uncertain; listed in "Left for the lead".
- **The 2026-10-12 sweep tool row**: built (`bench/sweep_loads.py`, `check_sweep_loads.py`).
- **Three two-subject rows and "three people in one pass"**: the lane is closed (roadmap "Closed lanes", 2026-10-15).
- **"Her hair comes and goes beside her head" (hair band, fa5c58ac, 7504bbc4)**: the band and her keep-outs retired
  2026-10-17; 0604h laid. Links: `internal/claude/2026-10-14_mrtwo_scripts/hair_0604/findings.md`,
  `data/2026-10-12_second_hf_0604_b/subjects/clothes/`, `internal/claude/2026-10-14_mrdrill_hair_read.md`.
- **The composite's `only what changed` decides where hair stops; the 0014d keep-out speck shared across a step**:
  superseded by the "whole region" composite and the retired self keep-outs (decisions 2026-10-16). Links:
  `internal/claude/2026-10-14_mrpre_hair_room_0604d_read.md`, `internal/claude/2026-10-14_mrpre/`.
- **`draw_lid_marks` read-back and `capture_masked_run.py video` stacking**: both fixed 2026-10-12 (rows said so).
- **2026-10-11 block**: the two lid-marks records are findings; "owed one render with marks at three times" and "the
  builder's motion choice and a generated workflow for the lid-marks form" are superseded by the head reference form
  (`data/AGENTS.md`, "Settled for the masked lane"; `workflows/masked_job_graph.py`); "the local page reorganized" done
  (one review page, Start here; decisions 2026-10-16/18); trial 2 of `docs/masked_v2v/perception_models.md` dropped
  (uncertain; see "Left for the lead"). Links: `internal/claude/2026-10-12_mrfern_handoff.md` (kept on the layered-view
  row), `bench/examples/masked_job_lid_marks/`.
- **The lead's four singing loads (2026-10-12)**: per-shot work since kept (street spin `lead_fp_0438e` laid
  2026-10-18); its gaps were per-video facts (1134-1155 left as source, no voice on 1093-1188) belonging in
  `data/fun/AGENTS.md`. Links: `Video/fun/mrhorn/lead_looked_at/lead_part_1141_1154.jpg`, `lead_part_0315_0358.jpg`,
  `data/2026-10-10_fun_audio/voice_spans.json`, `data/fun/shots.json`, `data/2026-10-13_second_cz_1010/READING_mrolaf.md`.
  The "zoomed preview stage is not to be built" ruling (2026-10-13) is a finding in that READING file.
- **Built 2026-10-13, cut as done**: `bench/stack_delivery.py`, `bench/load_chain.py`, `bench/shot_to_load.py`
  ("it has run no real load" was stale), `bench/render_ledger.py` and `bench/delivery_table.py` (the plain-cut field
  kept as an open row). Links: `data/2026-10-11_fun_joined_b/`, `data/2026-10-13_fun_joined_stacked/build_stacked_join.py`.
- **A zoomed load: the capture reads it, the job tool cannot ask**: zoom is now in the chain (`zoom_box`, floor at the
  zoom, `starting_floor`, 2026-10-14; `8ac7e928`). Its unverified notes (fit vs crop at non-canvas sizes) are in git.
- **Done 2026-10-14 (`--parts-on`, the reference held to the subject's track, `--track-not`, the zoomed keep-outs at
  first capture, the zoomed part floor)**: rows cut. Link: `internal/claude/2026-10-14_mrdrill_runs_past_2049.md`.
- **The delivery table orders overlapping keeps by render time**: superseded by `data/fun/laid.jsonl` being the only
  record of what is laid (decisions 2026-10-16).
- **The shots record calls 5935-6394 one shot**: looked at and split at 6105 on the lead's ruling. Link:
  `data/fun/cuts_look_2026-10-13/reading.md`.
- **`claimed_by_more_than_one_track` counts two people**: closed the same day (row said so).
- **"Two-subject load, both stills referenced per the vendor guide"**: built, rendered once, closed with the lane.
- **"For new checks and tools: green or red only for a hard failure"**: now a root `AGENTS.md` rule; the open half is
  the four-reds row, kept.
- **The removals of 2026-10-10 "landed that evening"** and its order note (restart, rebuild, sweep): done.
- **Masked v2v 2026-10-04 to 10-07 block**: version-one pointers, the 10-06 prompt node / review graph / parts graph
  (superseded by `bench/load_text.py` and the job chain), the 10-07 SAM tracker owed tests (superseded by memory
  selection, 10a and the head track, decisions 2026-10-17/18), the `keep` input and `held` output (self keep-outs
  retired), the still to the encoder alone (closed), "the region is too big for a small subject" (the margin is
  handover item 2b), "a corrected shot is searched again", the Subject Track's fourth clip and look-alikes, "the
  scene's clothes on the new subject" (clothing is now named in the text). Kept: the turn, the shadow, the resample,
  the full-size still, the frozen cache, the parked items. Link: `data/2026-10-07_capture_windows/README.md`.
- **Encoder int8 against bf16 (2026-09-27): closed**: `ENCODER_INT8` stays; records under `data/results/2026-09-27_encoder_quant_*.json` (open_experiments #47).
- **"Open after the 2026-09-27 night", decided items**: Sol quantizer `rotated` (0.166.0), save format H.264 crf 19
  (#38), PDD6 an option, #37 and #34 closed, #46 no stripped file, research files deleted: all decided 2026-09-27
  (`bench/results/2026-09-27_*` records). Kept: O1 blacks, the finish-as-default call.
- **Sol redesign "dense tail beats Sol on 45, 48, 49 so `dense_blocks` stays", and its three 2026-10-02 annotations
  (Options A, B under trial)**: superseded by Option C adopted 0.185.0 (decisions 2026-10-02). Sage vs kitchen as
  Sol's fallback: settled 2026-10-01 (kitchen stays).
- **FastH3 (2026-09-29) bullets, the overlay node, PDD8 finished by FastH3 (parked), no general adapter**: findings in
  their records; open remainders moved to #35/#36 rows.
- **"Comparing across 0.166.0" and the RoPE / VAE tile-blending warnings**: kept, moved to "Do not rely on".
- **The three "Idea saved for later, 2026-09-26" sections**: cut to one line. The route-3 build plan (FlashGen first,
  PDD finishing; handoff sigma 0.8889; knots [19, 24, 28, 32]) was built and rendered as the reverse switch
  (`rev_h080`, `bench/results/2026-09-27_reverse_switch.md`); PDD first with a FlashGen finish shipped as
  `h3_text_to_video_pdd8_flashgen_finish`. Links: `data/results/2026-09-26_x0_steps_subway_pdd8.json`. The schedule
  idea's premise is #43.
- **PDMD retired and parked (2026-10-05, 2026-10-01)**: in "Closed lanes".
- **FastH3 before judging it (2026-09-26)**: superseded by #35/#36 and the 2026-09-29 reads.
- **FlashGen before audio refine (2026-09-26)**: owed looks superseded by the 2026-09-27 reviews; cut. Uncertain (see
  "Left for the lead").
- **0.142.0 notes**: `dense_blocks` 45,48,49 superseded (Option C); "the tail default is unscored" superseded by the
  2026-10-02 panel; check_audio_freeze and check_schema_defaults kept in "Checks and hygiene".
- **2026-09-25 throwaway run**: listen, quoted session, step caching ("worth reopening only if 16-step base graphs are
  still rendered"), temporal offset (nothing to do): cut as stale or settled; the two cheap probes kept.
- **2026-09-25 PDD digs**: "can PDD do better" three arms cut (PDD parked); head-half and report-upstream kept;
  check_sol_observe and Kijai symlinks done; INT8 VAE superseded (0.151.0).
- **2026-10-03 sparse**: the output check is closed (kept on the roadmap).
- **2026-10-02 upstream survey**: PR 16681 done in 0.184.10; "not proposed" items cut (sglang integrated mode,
  Spectrum, vllm-omni no-resize stills: `bench/refview2_arms.json` is the test for `allow_upscale`).
- **2026-09-25 upstream survey**: torchaudio and bf16 refusal done 0.139.0; "next kitchen tag" closed as #33; the INT8
  VAE "not proposed" reopened and adopted 0.151.0.
- **2026-09-20 "documented pointers to scripts are unchecked, and six have rotted"**: superseded by
  `bench/check_doc_links.py` (backticked paths, the doc-link-absent block in `docs/method/checks.md`).
- **2026-09-19**: the blind panel (`bench/blind_panel.py`, key under `internal/blind_keys/`) held the profiler and the
  `dense_blocks` contest; `dense_blocks` was decided 2026-10-02 and the output check closed 2026-10-03, so the
  panel's gating role is gone (scoring it is still possible). Pair alignment, judge tie rate, every-fifth-latent mass
  (CLOSED): findings in `bench/results/2026-09-19_*`.
- **2026-09-15**: TaoMate removed; "cafe pair: worth re-judging 8 vs 5 steps" cut with PDD parked
  (`bench/results/2026-09-15_cafe_kids_verdict.json`).
- **2026-09-14**: the two retired Sol-on probes; the frozen-audio loop's stage-by-commit history (`22b046c` ...
  `e07c462`, smoke records `data/results/2026-09-14_audio_freeze_song_*_smoke.jsonl`, postmortem
  `internal/postmortems/2026-09-14_session_api-only-and-loop-simplification.md`); UI/API twins (API-only since
  2026-09-14, plan `internal/2026-09-14_api_only_workflows_plan.md`). Owed renders kept.
- **2026-09-13**: refview2 closed unscored (key lost, `qwen_view` decided 2026-10-03); "what changed on 2026-09-13"
  (history, `internal/log/log_2026-09-13.md`); "before queueing a reference render read the preview" (guidance in
  `docs/h3/references.md`); sage `smooth_k` trade cut: rejected by this stack for materialising a K copy
  (`docs/attention/block49_quant_error.md`). The served sage tag check kept.
- **2026-09-12, 09-11, 09-10**: audio-freeze morning brief (stale); graphs regenerated; Sol core A/B retired
  2026-09-15 (`data/results/2026-09-10_sol_core_ab_*`); all-rows switch done 0.185.0; pair B graded
  (`data/results/2026-09-10_sol_impl_capture_grade.json`); keyframe determinism (#30). Kept: run-row code stamp,
  check_dit_prefix_attention, core's reference-audio crop, check_node_ids.
- **2026-09-05 and evalman/bake lane items**: bake pair, PDD ladder and just-Sol scored 2026-09-05; frontier tables
  done; all-rows one pair superseded by adoption; the armed just-Sol render (Sol policy search closed 2026-10-03);
  the bake hypothesis tested (did not hold); fp16-sage moot; turbo rung closed. Records:
  `data/results/2026-09-05_*_verdict.json`, `data/results/2026-09-08_sol_nosage_2026-09-04_frontier.json`.
- **"Candidates on trial (2026-09-05)" table**: superseded; its step mixes predate `07b903c` and the default chain
  moved 2026-09-15. The graphs the owner renders with are `h3_config.DAILY_GRAPHS`.
- **"Now" section**: the 2026-10-02 dense-blocks reanalysis (superseded by Option C); the dated 2026-09-04 to 09-19
  status layers (reorder off: Morton closed; floor moved; ladder reading; online probe; PDD ladder; reference
  pathway (#26); block 49 token routing (#29); attention-share floor
  (`data/results/2026-09-08_attention_share_bound.json`, `bench/derive_attention_share.py`); token routing stages;
  upstream checks; small fixes). All superseded or recorded.

#### roadmap.md

- **"Current forward plan — 2026-09-04" and "— 2026-09-03"**: replaced by "The attention plan, where it stands" (a
  table of the 09-04 steps with their state). Superseded content: steps 1, 3 (adopted 0.185.0), 4 (Option C), 7
  (token_aug, #29) done; "Not established after 2026-09-04" items judged 2026-09-05 or adopted; the 09-03 "emerging
  architecture" hypothesis (Sage outer steps) superseded 2026-09-15; the PDD bake paragraph (tested, did not hold;
  PDD parked). The decision standard and "what would count" kept.
- **"Owner decisions, 2026-09-05 evening"**: kept as a table with each decision's fate (decisions.md starts at
  2026-09-11, so these have no other home), plus the two 2026-09-04 decisions.
- **"The regime question (2026-08-20)" and its verdicts; "Verdicts, 2026-08-22"**: cut to findings in "What is
  established" (power, speaker attribution, hum). Superseded: the lightx2v regime (turbo lane closed 2026-09-26),
  session 1's LoRA verdicts and the SLA regime set (lane closed). Links: `data/results/2026-08-20_sla_arms.jsonl`
  (kept), `data/results/2026-08-20_session1_verdict.json`, `data/results/2026-08-20_sla_regime_arms.jsonl`,
  `data/results/2026-08-20_ref_transfer_single.jsonl`.
- **"CLOSED 2026-08-16: token ordering as a quality question"**: Morton is in "Closed lanes" (2026-09-27); the full
  reasoning is `docs/attention/morton.md` and
  `internal/postmortems/2026-08-16_session_sol-ordering-and-blind-controls.md`.
- **"Next, in order" (as of 2026-08-20)**: its own status note said only item 4 (`ncu`) was open, and item 4 was done
  2026-09-17 (`bench/results/2026-09-17_sol_stage_profile.md`). Cut.
- **"What would make the error-decomposition line worth continuing"**: all done or refuted; the one forward note
  (compare routing masks, not outputs) kept under "What is not established".
- **"Two decisions that need no card"**: `dense_blocks="0-1"` decided against 2026-09-14 (decisions.md); the block-49
  fold is `docs/attention/block49_quant_error.md`; the NVLabs sm89 comparison kept as a row.
- **"What is established" narrative**: cut to one-line findings with their records; withdrawn 2026-08-20 readings
  (adaln anti-correlation, "the most-rewritten tensor", "adaln is the only difference") logged here, not restated.
- **Dials**: the "PDD Sol recipe is four eyeballed knobs" paragraph cut: its (1) is moot since 2026-09-11, and its
  "evidence against the shipped `0,1,2`" is superseded by Option C. The left rail's git rule replaced by a pointer to
  root `AGENTS.md`; "lightx2v student" labelled as a closed lane. "Last updated: 2026-09-11" now 2026-10-19.
- **"What is not established" table**: rows done 2026-08-17/19 (error decomposition, routed density, reference-heavy
  captures) and the power pair (run) cut; the `ncu` question answered by 2026-09-17.

#### open_experiments.md

- Entries 2, 3, 7, 9, 11, 15, 16 (16a-16g), 18, 20, 21, 22, 23, 25, 30, 31, 33, 34, 35, 37, 38, 39, 46, 47, 48 moved to
  "Closed entries", one line each with how they closed and where the result lives. Corrected: #3 said "blocker: none"
  though the kitchen int8 backend became the default 2026-09-15; #9 said no arm renders without speed-ups, though the
  dense stamped baseline exists since 2026-09-03.
- Open entries cut to question, decision, blocker and link. Their arms, pre-registrations, method notes and the
  prediction tables (e.g. #8's solve for the dense wall-clock share, #17's port file list, #22's design, #23's arm
  table) are in git at `901bee1`.
- The header's "next sweep 2026-09-28" (missed) is now 2026-11-02; the banner about `internal/` plans for #1, #3, #6
  cut (stale).
- The relative links `../bench/...` and `../data/...` (wrong after the move into `docs/state/`) are rewritten as
  repo-relative backticks or `../../`.

#### Gitignored links that left the pages with cut text

(Generated: every `internal/`, `data/`, `output/`, `coderef/`, `contact_sheets/`, `Video/` and results-JSON target
in the old text that is no longer in the new page, with its old line number.)

#### Cut from `docs/state/next_steps.md` (old line: target)

- 117: `internal/gemini/2026-10-15_watch_0604_1009_and_5675b.md`
- 120: `internal/claude/2026-10-15_mrlol_scripts/lead_hair_seam.txt`
- 135: `internal/gemini/2026-10-15_clips_review.md`
- 146: `internal/gemini/2026-10-15_watch_strangers_0630d.md`
- 147: `data/2026-10-15_strangers_touching_0630d/`
- 149: `internal/claude/2026-10-15_mrlol_scripts/strangers_on_her.txt`
- 160: `internal/claude/2026-10-15_mrlol_scripts/last_step_motion.txt`
- 172: `internal/gemini/2026-10-15_watch_hair_item4.md`
- 192: `internal/claude/2026-10-14_mrdrill_scripts/iris_offset_render_zoomed_1097c_1097_1133.csv`
- 228: `data/2026-10-14_lead_fp_1466b/look_blind/reading.md`
- 297: `data/<date>_sweep_<tag>/`
- 349: `internal/claude/2026-10-14_mrtwo_scripts/hair_0604/findings.md`
- 350: `data/2026-10-12_second_hf_0604_b/subjects/clothes/`
- 361: `internal/claude/2026-10-14_mrdrill_hair_read.md`
- 377: `internal/claude/2026-10-14_mrpre_hair_room_0604d_read.md`
- 382: `internal/claude/2026-10-14_mrpre/crops/hair_0604d_bottom_left_604_612.jpg`
- 390: `internal/claude/2026-10-14_mrpre/`
- 536: `Video/fun/mrhorn/lead_looked_at/lead_part_1141_1154.jpg`
- 539: `data/2026-10-10_fun_audio/voice_spans.json`
- 577: `data/2026-10-13_second_cz_1010/READING_mrolaf.md`
- 588: `data/2026-10-11_fun_joined_b/`
- 589: `data/2026-10-13_fun_joined_stacked/build_stacked_join.py`
- 957: `internal/claude/2026-10-14_mrdrill_runs_past_2049.md`
- 985: `data/fun/cuts_look_2026-10-13/reading.md`
- 1094: `data/<video>/shots.json`
- 1128: `data/2026-10-10_cuts_piece5_analysis_input/`
- 1198: `data/2026-10-07_capture_windows/README.md`
- 1291: `data/results/2026-09-27_encoder_quant_dit.json`
- 1292: `data/results/2026-09-27_encoder_quant_floor.json`
- 1295: `data/results/2026-09-27_encoder_quant_refiner.json`
- 1333: `internal/pdd_shims/shim_{fasth3`
- 1475: `data/results/2026-09-26_x0_steps_subway_pdd8.json`
- 1622: `bench/results/2026-09-26_int8_lora_requant.json`
- 1864: `internal/blind_keys/`
- 1910: `bench/results/2026-09-15_cafe_kids_verdict.json`
- 1963: `data/results/2026-09-14_audio_freeze_song_timeline_smoke.jsonl`
- 1972: `internal/postmortems/2026-09-14_session_api-only-and-loop-simplification.md`
- 1982: `data/results/2026-09-14_audio_freeze_song_stage1_smoke.jsonl`
- 1986: `data/results/2026-09-14_audio_freeze_song_resume_smoke.jsonl`
- 1994: `data/results/2026-09-14_audio_freeze_song_lists_smoke.jsonl`
- 2015: `internal/2026-09-14_api_only_workflows_plan.md`
- 2028: `data/results/2026-09-13_refview2_arms.jsonl`
- 2029: `Video/blind/refview2_2026-09-13/`
- 2073: `Video/h3_candidate_t2v_pdd8_baked_audio_freeze_untold_loose_pdd8_00002-audio.mp4`
- 2079: `internal/log/log_2026-09-13.md`
- 2117: `data/results/2026-09-13_reference_settings_three_stills.json`
- 2136: `bench/results/2026-09-12_audio_freeze_step2_verdict.json`
- 2145: `data/results/2026-09-12_audio_freeze_song_smoke.jsonl`
- 2168: `data/results/2026-09-10_sol_core_ab_arms.jsonl`
- 2176: `data/results/2026-09-10_sol_core_ab_outputs.json`
- 2182: `data/results/2026-09-10_sol_impl_capture_grade.json`
- 2199: `bench/results/2026-09-10_sol_core_ab_audio_loudness.json`
- 2234: `data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`
- 2246: `data/results/2026-09-05_pdd_bake_outputs.json`
- 2259: `data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`
- 2266: `data/results/2026-09-04_pdd_ladder_audio_loudness.json`
- 2269: `data/results/2026-09-05_pdd_ladder_2026-09-04_frontier.json`
- 2279: `data/results/2026-09-03_ladder_outputs.json`
- 2289: `data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`
- 2298: `data/results/2026-09-08_sol_nosage_2026-09-04_frontier.json`
- 2310: `data/results/2026-09-04_probe_allrows_vs_shipped_pixels.json`
- 2312: `data/results/2026-09-04_probe_render_vs_unarmed_pixels.json`
- 2318: `data/results/2026-09-04_sol_probe_base16_subway.json`
- 2339: `bench/results/2026-09-05_turbo_rung_*.jsonl`
- 2380: `data/results/2026-09-04_ladder_2026-09-03_frontier.json`
- 2488: `data/results/2026-09-04_ladder_2026-09-03_verdict.json`
- 2511: `data/results/2026-09-04_ladder_audio_loudness.json`
- 2525: `data/results/2026-09-03_probe_replay_base16.txt`
- 2527: `data/results/2026-09-03_sol_probe_base16_standoff.json`
- 2555: `data/results/2026-09-04_ref_pathway_2026-09-03_verdict.json`
- 2567: `data/results/2026-09-08_token_aug_determinism_shape.json`
- 2571: `data/results/2026-09-08_token_aug_selection_structure.json`
- 2588: `data/results/2026-09-08_token_aug_stock_wheel_control.json`
- 2591: `data/results/2026-09-08_token_aug_repro_shapes.json`
- 2599: `data/results/2026-09-08_token_aug_admitted_count.json`
- 2606: `data/results/2026-09-08_attention_share_bound.json`
- 2634: `data/results/2026-09-04_stairwell_dense_retime.jsonl`

#### Cut from `docs/state/roadmap.md` (old line: target)

- 61: `data/results/2026-09-04_ladder_2026-09-03_verdict.json`
- 101: `data/results/2026-09-03_sol_probe_base16_standoff.json`
- 121: `data/results/2026-09-04_sol_probe_subway_allrows_vs_shipped.json`
- 125: `data/results/2026-09-04_probe_allrows_vs_shipped_pixels.json`
- 128: `data/results/2026-09-04_probe_render_vs_unarmed_pixels.json`
- 196: `data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`
- 199: `data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`
- 208: `data/results/2026-09-04_stairwell_dense_retime_pixels.json`
- 361: `data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`
- 685: `data/results/2026-08-20_ref_transfer_single.jsonl`
- 698: `data/results/2026-08-20_session1_verdict.json`
- 712: `data/results/2026-08-20_sla_regime_arms.jsonl`
- 787: `internal/postmortems/2026-08-16_session_sol-ordering-and-blind-controls.md`
- 854: `data/results/2026-08-20_sol_error_per_head_fl2va.json`
- 855: `data/results/2026-08-20_head_magnitudes_fl2va.json`
- 863: `data/results/2026-08-19_sol_error_per_head.json`
- 881: `data/results/2026-09-17_sol_exact_stage_vs_compute.json`
- 924: `data/results/2026-08-19_sol_error_control.json`
- 989: `data/results/2026-08-18_erode_recipe_arms.jsonl`

#### Cut from `docs/state/open_experiments.md` (old line: target)

- 769: `internal/reference_library.md`
- 1003: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/eager/sol_attn.py`
- 1026: `internal/scripts/archive/sol_curve_2026-08-16/probe_routed_density.py:41`
- 1039: `internal/scripts/archive/sol_curve_2026-08-16/probe_routed_density.py:47`
- 1105: `coderef/ComfyUI-SolAttn_triton/__init__.py`
- 1163: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu::mma_bf16`
- 1165: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh::pack_bf2`
- 1185: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:63-65`
- 1201: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:83`
- 1243: `data/results/2026-09-17_sol_exact_stage_vs_compute.json`
- 1332: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:19-21`
- 1334: `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/CMakeLists.txt:135-139`
- 1412: `coderef/LightX2V`
- 1413: `coderef/SLA`
- 1581: `internal/recipes/`
- 1634: `bench/results/<date>_pruning_sensitivity.json`
- 1957: `data/results/2026-09-03_ref_pathway_arms.jsonl`
- 1958: `Video/h3_probe_ref_pathway_*`
- 1979: `data/results/2026-09-04_ref_pathway_encoder_arms_pixels.json`
- 2047: `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py`
- 2073: `bench/results/*_outputs.json`
- 2348: `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py`
- 2405: `data/results/2026-09-26_flashgen_lora_path_s1.jsonl`
- 2491: `data/results/2026-09-27_vae_encoder_fp16acc_before_672.json`
- 2506: `internal/postmortems/2026-09-26_session_distill-weights-and-render-dataset.md`
- 2599: `internal/pdd_shims/shim_fasth3_temb.safetensors`
- 2611: `data/results/2026-09-26_fasth3_lora_rank.json`
- 2838: `internal/postmortems/2026-09-26_session_distill-followup-and-attention-parity.md`
- 3071: `data/results/2026-09-27_encoder_quant_dit.json`


### Left for the lead

- **`handover.md` is 358 lines against the about-80-line rule** (four headings present; "State when closed" alone is
  about 95 lines and repeats the night's results block).
- **decisions.md has no entries before 2026-09-11**; the owner decisions of 2026-09-04 and 2026-09-05 live only in
  `roadmap.md`, "Owner decisions, 2026-09-05 evening". If decisions.md is meant to be the only log, copy them there.
- **Inbound links to quote updates**: `docs/research/pdd/2026-09-25_upstream_pdd_comparison.md` line 145 at `901bee1` and
  `bench/INDEX.md:602` cite roadmap "Owner decisions, 2026-09-05 evening" (heading kept, now a table);
  `workflows/build_workflows.py:4997` comments on "Candidates on trial, 2026-09-05", a next_steps section now cut
  (the comment should point at `h3_config.DAILY_GRAPHS` or the graphs' own notes). Anything citing roadmap "Current
  forward plan — 2026-09-03/04", "The regime question", "Next, in order" or next_steps "Now" / "Open after the
  2026-09-27 night" (a grep found none outside the changelog and archive) should point at roadmap "The attention plan,
  where it stands" or next_steps "Attention, distills and the base model".
- **Root `AGENTS.md` and `CLAUDE.md` still say `docs/wiki/...`** (`docs/wiki/handover.md`, `docs/wiki/next_steps.md`,
  `docs/wiki/decisions.md`, `docs/wiki/working_with_the_owner.md`) though the files are under `docs/state/` and
  `docs/working/` (seen in the root `AGENTS.md` loaded this session; may already be rewritten on disk).
- **`roadmap.md` "Closed lanes", the two-subject row**, says "where two are close, one hole a pass", while decisions
  2026-10-16 says one hole a pass is not built. Left as written (a closed-lane row); a lead's call whether to add "not
  built".
- **The full text of the cut August-September roadmap sections** (the regime question, token ordering, the
  error-decomposition line, the per-head pricing, the hybrid checkpoint measurements) is only in git now. If it should
  stay readable, move it from `901bee1:docs/state/roadmap.md` into a research page, e.g.
  `docs/research/2026-08_sol_speed_quality_history.md`, and link it from roadmap "What is established".
- **Unsure, cut (restore if wrong)**: the 10-13 "who page numbers a shot on one frame" row (parked by the lead);
  "trial 2 of `docs/masked_v2v/perception_models.md`, The trials" and "the record of a video on the share with the
  tool that takes the page's raw answers into it" (10-11 owed items; status unknown); "FlashGen off T2VA: the owner's
  look" and "a 125-frame arm at FlashGen's trained length" (2026-09-26).
- **Unsure, kept**: open_experiments #4, #8, #10 (pre-September; round 2 of #8 never recorded); #28 (re-scoped to the
  unrun before-and-after capture); next_steps' sage chain assert row (sage is off by default) and the four-reds row.
- **Per-video rows that belong in `data/fun/AGENTS.md`** (not on this box): 1134-1155 left as the source, no voice on
  1093-1188 in `voice_spans.json`, the owner's typed words on 1920-6394, the shots record's frame fields.
- **`docs/h3/nodes.md`** is unreachable from the router (`build_wiki_index`), not this folder's.

## docs/method/ and docs/working/

The originals are in git at `901bee1`. The gitignored-link fixer (docs/method/*.md docs/working/*.md) was run before finishing (it repointed 2
links; evidence.md's `../data/results/...` links had already been made `../../data/results/...` by hand).

### Moves

none (no page renamed, merged or deleted). Section headings that other files quote and that no longer exist:

- `docs/working/coderef.md`, "What moved by 2026-09-10" / "-09-11" / "-09-19" / "-09-25" / "-10-02" / "-10-05" /
  "-10-09" -> folded into per-checkout sections of the same page: "sglang", "vllm-omni", "LightX2V", "FastVideo",
  "DiffSynth-Studio", "diffusers", "ComfyUI core", "ComfyUI-UtilsCollection", and the tables. "Masks and edited video
  upstream, beside the masked lane", "The streaming references: TaoMate" and "A distill's reference is its trainer's
  contract" keep their names.
- `docs/method/evidence.md`, "Settled about H3, moved here from `CLAUDE.md` on 2026-09-03" -> "Settled about H3"
  (the name root `AGENTS.md` quotes). "Why this file exists" and "Relationship to sol_attn.md's header" became two
  lead paragraphs of the opening.
- `docs/method/eval_comparison.md`, "Motion regime is part of a perceptual claim, not context for it", "Correction,
  same day: DELTA is the wrong axis for choosing a scene", "Scored the same evening, then corrected within the hour"
  -> "What a scene can show" (with subsections "Frame-to-frame change, and why it is not motion", "Demand, not
  result").
- `docs/working/comfy_notes.md`: "Running things", "The recipe, by hand", "The three arming keys", "Captures are
  transient", "If you started the server, stop it when you are done", "The local review page has its own small
  server, started and stopped by hand", "After a ComfyUI update", "Changing a graph", "Adding a probe or an arm"
  (bold lead-in), "What is cached, and by whom", "Traps that have each bitten more than once" are all kept (several
  became `###` headings). "Between the kill and the launch, confirm the launcher chain is gone too" is now a bullet
  under "The recipe, by hand".
- `docs/working/runbook.md`: no heading changed; "last updated" line dropped.

### Removed or corrected claims

#### docs/method/checks.md

- "The standard" carried three dated layers (shown-red requirement retired 2026-09-11; 2026-10-10 and 2026-10-16
  notes on mutation runs). Replaced by one current statement: showing red is not required; when done, use
  `bench/break_checks.py`, with the three conditions (green control run, failing line names the break, one break per
  copy). The 2026-10-10 "fourteen breaks, four never seen" and 2026-10-16 "night of mutation runs" narratives dropped
  (the findings stay).
- "A metric that ranks two arms": cut to the rule, the three-metric table, the subset finding and the
  second-implementation finding; the narrative of two consecutive wrong commits and the `CLAUDE.md` quote dropped.
  Note: it cites `bench/results/2026-08-31_merge_realisation_{pdd,turbo}.json`, which is not on this box (kept per
  the gitignored-links rule).
- "Running them": "Until 2026-10-05 this section said exactly two needed it (`check_clone_v_wiring.py`,
  `check_correctness.py`); a sweep found `check_reference_encode.py` needed it too" -- superseded by "No check needs
  `PYTHONPATH`". The list of five checks that crashed masked until 2026-10-05 dropped (fixed; the rule stays).
- "What is deliberately not checked": "seven things this repo has decided not to measure" -> "the things" (a count in
  prose, against this file's own rule).
- "Uncontrolled requirements": "Next sweep: 2026-09-28" (passed) -> "The last sweep scheduled here was for
  2026-09-28". Closed rows removed (each requirement is now held by a check whose index row carries the same
  content): `check_model_contents.py` (two rows), `check_model_files.py`, `check_lora_alpha.py`,
  `check_graph_discovery.py`, `check_node_ids.py`, `check_reference_contracts.py`, `check_marker_arms.py`,
  `check_reference_runtime.py` (including "two 39-frame GPU smokes are green", run history),
  `check_conditioning_behaviour.py`, `check_distill_grid.py`, `check_pdd_bank_encoding.py`, `check_pdd_sigmas.py`.
  The split-graph SigmaShift row ("nothing since 2026-09-26; no split graph ships") merged into "What is deliberately
  not checked". The camera-vocabulary row cut to its open half (speaker identity, not mechanizable). The Sol-default
  row cut to its open half (graphs outside `graph_paths()`). The power-state row corrected: it said bench runs
  persist nothing; `bench/run_graph_arms.py` persists the limit per row since 2026-08-18 (per `hardware.md`), only
  `bench/bench_e2e_h3.py` persists nothing. Its "this row asserted ... a peer refuted it" narrative dropped.
- "Gaps": item 2 "~~No runner~~" and item 3 "~~The PYTHONPATH split~~" (closed 2026-10-05) removed; 2 now states the
  remaining part (no case registry). Item 6 "covers `CLAUDE.md` only" -> root `AGENTS.md`.
- Retired checks: wording tightened, nothing removed.
- The index table and the `doc-link-absent` block are byte-identical to before.

#### docs/method/evidence.md

- "Last updated: 2026-09-03; the encoder and reference-sizing rows 2026-09-13" dropped.
- Do not rely: the row "any reference measurement taken before `_ref_prompt()` is fixed" removed; it read "RESTORED
  2026-08-17" (Step 0 refactor guarded by `bench/check_ref_prompt_labels.py`, Step 0b blind re-renders on seeds 42
  and 137 at 5.0/5.0). Retired by its own re-measure.
- Do not rely: the two capture-manifest rows (`bypassed_for_capture`, and `sage_mode`/`models.clip`/`models.loras`)
  shortened to the standing warning (manifests from before 2026-08-17 / 2026-08-26 carry constants) and the fix. Dropped:
  the cite that `docs/method/drift_frontier.md` had been corrected in place, "`:133` writes rank 256" line ref (kept as
  "it writes `rank: 256`"), "the sentinel emitter below" (no such section on the page).
- The 362 paragraph: restated the root `AGENTS.md` settled line; now links to it and keeps only what is unique
  (diffusers' 15.0 s cap, `reference_would_emit()`, the "362 is illegal" withdrawal, what 362 rests on).
- These hold: "All 34 API graphs carry `LONG_LENGTH`, read from their widgets" removed (a count from 2026-08-14 with
  no command; `check_*` and the generator own it now). "Every `bench/check_*.py` passes, and the ones added today were
  each shown red first" removed (superseded: the shown-red rule was retired 2026-09-11, and `bench/checks_baseline.json`
  lists checks red for a stated reason).
- These hold: the marker-rows bullet's "Corrected 2026-09-03, hours after this row was written" layer folded into the
  plain current statement (the prediction-delta form ran and was refuted by its control; the blind-render form has
  not run).
- These hold: "The marker ids are fixed by loading the release's tokenizer directory" (one line pointing at itself)
  merged into the full "marker ids are the release's own" bullet.
- These hold: the PDD bullet's italic 2026-09-25 note folded in; the withdrawn clause "vary the transform at fixed
  partition" is gone from the body (the research record holds it).
- These hold: reference sizing folded to the current state: `qwen_view` is `separate` at 512 by the owner's decision
  of 2026-10-03; the other three defaults at vendor parity. Dropped history: the separate 512 view that shipped from
  2026-08-27 on one render (CHANGELOG 0.82.0); "since 2026-09-13 the node's defaults are vendor parity ... qwen_view=shared";
  `MiniMaxH3ReferenceFit` deprecated 2026-08-28 and deleted in 0.173.0; the flat `qwen_short_edge` whose 0 meant
  shared until 2026-08-31; `MiniMaxH3ReferenceReport` deleted in 0.174.0.
- These hold: the qkv row-order bullet's "A 'correction' to this bullet on 2026-08-29 was itself wrong and is
  withdrawn" layer folded into a plain statement of both release formats (native fused, diffusers split).
- These hold: "Withdrawn 2026-09-03: the node is vendored, symlinked and hash-pinned" plus its 2026-09-27 italic
  replaced by the current fact: the graphs wire `MiniMaxH3Sol`, graded by `bench/check_sol_node_equivalence.py`.
  (`vendor/sol_attn_minimax.py` is no longer on disk.)
- After the consumer block: the paragraph "And it cannot be closed by adding a row ... a spelling that should appear
  nowhere cannot be expressed" removed; superseded by `ALLOW: (none)`, which the block itself defines and which went
  in on 2026-08-25. Replaced by one sentence.
- Environment: the comfy-kitchen build history (`0.2.31+sol.c04ef20`, rebuilt `23d1a66` 2026-08-22, `dae00a1`
  2026-08-29) replaced by a pointer to `sol_attn.md`, "Install the CUDA kernel", which owns the installed build
  (currently `0.2.34+sol.757657f` per that page). Kept: the three-significant-figures warning across a kernel
  rebuild. Removed as stale: "`custom_nodes/ComfyUI-SolAttn-cuda/` created; its `sol_attn_minimax.py` is a symlink into
  `vendor/`" (the pack was deleted 2026-08-16 per `sol_attn.md`; the vendored file is gone); "`coderef/comfy-kitchen-kijai/`
  at `c04ef20` / `23d1a66`" (the revision lives in `coderef.md`); "it was carried as a standing working-tree
  modification until 2026-08-14"; "`nanobind` 2.14.0 added" (comfy_notes says the venv's nanobind is 2.10.2).
- Environment: "weight_quantization and vae_quantization ... fixed 2026-08-17, both now required and asserted, so a
  manifest can omit both and still pass" was self-contradictory; corrected to: `weight_quantization` required and
  asserted, `vae_quantization` deliberately not required (per `capture_manifest_schema.md` and the check's index row).
- Environment: the power-limit paragraphs shortened to a pointer to `hardware.md` plus the one unique fact (changed
  from stock 2026-08-17, after the bench_plan and sol_attn timings).

#### docs/method/drift_frontier.md

- "Where this stands, 2026-08-17 end of session" replaced by "Where this stands": superseded by later rows (F1-F4
  dropped 2026-09-11; `bench/red/` removed in `0a764a33`; `_git_head` collapsed; the hand-edited
  `workflows/h3_probe_capture_ref3_api.json` no longer has any Sol node, checked by reading the graph). Dropped from it:
  "everything left green ... all five harnesses in `bench/red/`"; "Pick up here: F1 unlocks the most"; the
  postmortem-filing instructions (F7 row owns them); the "in flight with the parallel Claude session" list; the
  `bypassed_for_capture` correction paragraph (evidence.md owns it). Kept: the "pattern this session kept producing"
  lesson. Rows untouched.

#### docs/method/config_drift.md, prose_measurements.md, sustainability.md

- config_drift: "What will not fix it" restated sustainability's do-not-do list (a docs-match-code checker) and the
  dates point; now links. "The one-line version" restated `VISION.md` "Truth" / root `AGENTS.md` "Do not trust
  prose"; now links and keeps the decision-versus-state line.
- prose_measurements: references to `CLAUDE.md`'s Guiding Principles, caching rule and correction rule repointed to
  `VISION.md` / root `AGENTS.md`. "The wiki" section described `docs/index.md` as inheriting `CLAUDE.md` rows and being
  "skipped as a generated file"; superseded (the router is hand-written, folder indexes route, code indexes are
  generated by `bench/build_index.py`). The order list's "read these before you start set `docs/index.md` names"
  updated. The shared-checkout paragraph now links to root `AGENTS.md`, "The shared tree".
- sustainability: "last updated: 2026-08-28" dropped. Item 2's ranked backlog table (encoder bounds, a PDD blind
  session, W4 DiT arch probe, SLA router coverage, video VAE default) removed as a second copy of `roadmap.md`'s
  queue (the page itself said "roadmap owns this queue; if it disagrees, it wins"); several rows are on closed lanes
  (SLA 2026-09-27, encoder quantisation 2026-08-27). The VAE lesson kept with its record link. Item 3 "Make the rest of
  `bench/` visible" (106 non-check files indexed by nothing) marked done: `bench/INDEX.md` generated by
  `bench/build_index.py`. Item 5 said `docs/index.md` "is regenerated from tables" (false now); repointed to the
  generated indexes.

#### docs/method/eval_comparison.md

- "Motion regime" plus its two same-day corrections folded into "What a scene can show", answer-first: the three
  findings table (within-clip correlation stands; across-clip delta refuted twice; demand is the candidate). The
  label "motion" is replaced by "delta" throughout, as the correction required. Owner quotations removed (brief rule);
  their content kept as paraphrase. All numbers and both tables kept.
- "A stress scene": the 2026-09-18 correction block at the top folded into the body (render a bank prompt only at its
  declared length; the noodle-bar render was out of distribution). Every record link kept.
- Title "Paired Video Evaluation & Stacking Guide" -> "Comparing renders: the A/B process and the stacking tool", with
  an answer-first opening. `CLAUDE.md` citations repointed to root `AGENTS.md` / `evidence.md`.

#### docs/working/coderef.md

Folded seven "What moved by <date>" sections into per-checkout current state with the last revision read. Superseded
reads dropped from the body (git history holds the full text):

- The 2026-08-28 implementations table's revisions (sglang `803b4fb31c`, LightX2V `5169278f`, DiffSynth `102fe99`,
  diffusers `9f7aee482`) replaced by the 2026-10-09 revisions.
- "No engine implemented PDD as of 2026-08-28" -> sglang implements it since `973fb44471`.
- vllm-omni #7693 (bf16 AdaLN rounding on SM90) -> superseded by #7913 (fp32 everywhere). The 2026-09-25 claim that
  sglang's affine is fp32 -> corrected 2026-10-02: sglang and core round at the same three points; vllm-omni is the
  outlier.
- "ComfyUI's templates loading an INT8 video VAE contradicts an owner decision" -> the owner reopened it 2026-09-26;
  `MODELS["video_vae"]` names it.
- "`bench/check_distill_settings.py` reads those [LightX2V] configs" (2026-09-10, 2026-09-25) -> it has read none
  since the turbo lane closed 2026-09-26.
- FastH3 V2 "ComfyUI cannot load that layout ... not on disk here" -> the Comfy conversion is on disk and two probe
  graphs load it.
- UtilsCollection's reference-video path "does not truncate, snap or apply the canvas rule" -> superseded by
  `ed4716e`; the label-order departure stands.
- comfy-kitchen row's two in-place corrections (2026-09-11, 2026-09-27) and comfy-kitchen-kijai's (2026-09-11) folded.
- FastVideo 2026-10-05 "the contract's tile has one route with no Triton fallback" -> refined 2026-10-09: that is the
  Ref2VA student's tile; V2's has a Triton route.
- pdmd row: the 2026-10-01 read (`03ee66b`) replaced by the 2026-10-05 read (`041b70b`).
- Model-Optimizer `b311c054d` (2026-09-19) -> `5a7bb6764e` (2026-10-09).
- "last updated" header (a 15-entry change list) dropped.
- Dropped as no longer relevant to anything that runs (turbo lane closed 2026-09-26, or titles only): vllm-omni
  #7062's implication for the two 8-step v1.0 768p files; `715b8b874`; the Minimax-H3-Turbo HF v1.1 768p fp8 file
  (2026-09-10); flashinfer commit ids (summarised as "capability 10, 12, SM90+ only"); DiffSynth other-model commits;
  LightX2V `3acec9e8`, `6214d38a`, `8335bb48`, `cb61e625`, XPU/MPS VAE details; sglang `9101e895ea`, `00b21d6415`,
  `516fd1c1b5`, `33e7b4cc4b`, `d677848968`, `aa5551d9b6`, `43b4abd857`, `17b2f35ae7` (the last one's doc-link fix is
  done); vllm-omni `d0b0cfd02`, `06afa310c`, `88a35c092`, `3a82c8588`, `acf662023`, `965ad68f1`, #7449, #7241;
  FastVideo `0cc41a22`, `e1e255935`, `a8688dddd`; UtilsCollection `f1576a6`, `4b81485`, `aad325d`, `4353de3`,
  `749d8d6`; the per-read "Unmoved" and "Not read" lists (their revisions are in the tables).
- The 2026-10-05 note on how the read was done (three read-only subagents, hand-checked claims) and the 2026-10-09
  note (one red link check, corrected) dropped.
- Intro: "`vendor/sol_attn_minimax.py` is a read-only reference node that is not loaded" -> it is no longer in
  `vendor/` (`ls vendor/` shows only `AGENTS.md`, `CLAUDE.md`, `README.md`, `meta_moge2`, `meta_sapiens2`,
  `rebuild_kernel.sh`).

#### docs/working/comfy_notes.md

- Rewritten answer-first with a rules table at the top. "Last updated: 2026-08-16" dropped.
- "Until 2026-09-26 this said the server writes `user/comfyui_<port>.log` ..." (twice, in the restart and traps
  sections) folded to the current fact: it no longer writes that file; send launcher output to a file and read
  `GET /internal/logs/raw`.
- The 2026-09-04 power-loss aside ("the evening run lost nothing to this") dropped.
- "`keyframe_canvas.py` (and `reference_fit.py`, retired 0.173.0)" -> the retired module dropped.
- The "Two partial catches exist" / "Why nothing else can" paragraphs on node ids condensed.
- The 362 paragraph shortened to a pointer to `evidence.md`.
- Added: `bench/_paths.py::comfy_output` reads its own `H3_COMFY_OUTPUT` before the server's `--output-directory`
  (read from the code; two keys name the output folder).

#### docs/working/runbook.md

- (a) "Written procedure" restated comfy_notes' restart recipe and cache facts step by step; now a pointer table.
  (c) the review-server launch, verify and stop sub-bullets now point to comfy_notes. The "last updated" line dropped.
- Two citations "`docs/index.md`, `bench/render_ledger.py` row" and "... `bench/delivery_table.py` row" pointed at a row
  the router no longer has; repointed to `bench/INDEX.md`.

#### docs/working/hardware.md

- "Last updated: 2026-10-03" dropped. The "A correction, kept because it is instructive" paragraph (the field existed;
  inferred from the `required` list) removed; the current state is in the table. "This paragraph used to conclude that
  PCIe therefore carries continuous traffic ... Measured 2026-08-18 and it does not" folded to the current finding.
  "~~Neither has been run.~~" struck text removed. Power section opens with the rule.

#### docs/working/pipeline_telemetry.md, capture_manifest_schema.md, working_with_the_owner.md

- telemetry: "Last updated" dropped; the owner's quoted request paraphrased (no quotes of passing remarks).
- schema: title to sentence case; "`CLAUDE.md` names" -> root `AGENTS.md`.
- owner page: "last updated" line folded into the opening sentence; "Green or red only for a hard failure" now cites
  root AGENTS.md. No preference changed.

#### Untouched

`2026-08-14_sol_bench_plan.md` (pre-registrations; no obviously duplicated note found; the link fixer may have adjusted its
gitignored links), `check_postmortems.md` (history), `rules_history.md` (frozen), `capture_manifest_schema.md`
body.

### Left for the lead

1. **Inbound section links into coderef.md** (rewrite the quoted heading):
   - `docs/attention/sol_upstream.md` line 57 at `901bee1` "What moved by 2026-10-09", under "ComfyUI core" and "comfy-kitchen" ->
     coderef.md "ComfyUI core" (kitchen is now the `comfy-kitchen` row of "The ComfyUI-side references").
   - `docs/h3/audio_freeze.md` line 436 at `901bee1` "What moved by 2026-09-25" -> coderef.md "vllm-omni".
   - `docs/h3/custom_node_gaps.md` line 305 at `901bee1` "What moved by 2026-10-02" -> coderef.md "diffusers".
   - `docs/state/decisions.md`, `CHANGELOG.md`, `changelog.d/mrfetch-upstream-read-1009.md` also name those sections;
     history, probably leave.
2. **`vendor/sol_attn_minimax.py` is gone but still cited**: `docs/attention/morton.md` (lines ~101, 104, 308, 599,
   734, 806, 809) and `docs/attention/sol_attn.md` (~565, 1591, 1658, 1722, 2133) cite it by path and line. Either
   declare it in checks.md's `doc-link-absent` block with the commit it was last present at, or repoint. I did not find
   when it was removed (I ran no git).
3. **INDEX.md is stale** (`check_doc_inventory.py`: "1 line(s) to add, 1 to drop") -- pre-existing before my edits;
   `bench/build_index.py` needs running.
4. **checks.md index rows still say "CLAUDE.md"** (`check_node_ids.py` "the rule CLAUDE.md opens with", and a few
   others). I left the table byte-identical because tools read it; a word change inside a cell is safe for the parser
   if the lead wants it.
5. **evidence.md's "Sage runs 5 of 16 steps under Sol"** and "Both backends are arithmetically equivalent" are
   2026-08-14 statements kept as they were; whether the current Sol node (`MiniMaxH3Sol`) and window still give 5/16 is
   not re-derived here.
6. **sustainability.md says "Retire it when it stops being argued with."** Its five items: 3 is done, 2 now points to
   the roadmap. The owner may want the page retired into `research/`.
7. **Two keys name the output folder**: `H3_OUTPUT_DIR` (`workflows/h3_config.py::output_dir`) and `H3_COMFY_OUTPUT`
   (`bench/_paths.py`). Noted in comfy_notes; a code-side consolidation is a candidate.
8. **drift_frontier.md's header "Next sweep: 2026-09-28"** is in the record part of the page (not the snapshot), so I
   left it. checks.md's equivalent is now "the last sweep scheduled here was for 2026-09-28". Whether that sweep
   happened is not recorded in my folder.
9. **checks.md's Uncontrolled table was swept against the old `CLAUDE.md`**; it now says so, and that root `AGENTS.md`
   has not been swept. A sweep of the current AGENTS.md is a candidate.
10. `docs/research/masking/2026-10-05_mryellow.md (no longer in the tree since 2026-10-11)` is kept as prose in
    coderef.md, as before.
11. **`docs/h3/resolutions.md` disappeared during my run** (the h3/ worker's move; `docs/h3/` now has `geometry.md`
    and `geometry_cost.md`). My pages link it at `docs/method/evidence.md:79,129`, `docs/working/coderef.md:234,488`,
    `docs/working/runbook.md:152,153` (several quoting its sections "The length axis" and "Whole on the audio clock"),
    and frozen `docs/method/rules_history.md` lines 310-311 at `901bee1` also cites `docs/h3/resolutions.md` and
    `docs/h3/geometry_and_nodes.md`. Rewrite with the h3/ worker's move map; these are the only non-environmental
    `check_doc_links` failures in my folders.

## docs/h3/

Ran the gitignored-link fixer on docs/h3 at the end (4 links repointed). Every link into
gitignored trees (coderef/, internal/, data/, output/, bench/results/*.json) in the old pages was checked to survive
in some new page (script diff of citations, old vs new); the only citations dropped are to deleted tracked tools
(`bench/analyze_canvas_geometry.py`, `bench/analyze_morton.py`, `bench/check_h3_awq_encoder.py`,
`bench/check_literal_widgets.py` mention, `bench/check_mono_ref_audio.py`, `bench/check_ref_video_prediction.py`,
`reference_fit.py:181/199-206`). `check_doc_links` on docs/h3 now fails only on comfy/, comfy_extras/, custom_nodes/,
ComfyUI/ and data/ targets (environmental). `build_wiki_index`: 0 unreachable. `check_retraction_consumers`: ok.
`check_skill_routes`: ok (two pre-existing REVIEW notes on prompting pages, not this folder).

Line counts, before -> after: audio_freeze 632 -> 452; comfyui_vendor_gaps 1133 -> 418; conditioning 435 -> 393;
custom_node_gaps 508 -> 277; geometry_and_nodes 395 -> (merged); image_editing 381 -> 59; index 30 -> 34;
input_impacts 451 -> (merged); references 1544 -> 293 (+ reference_sizing 224, reference_prompts 243); resolutions
334 -> (merged); stages 92 -> 89; uses 189 -> 184; new geometry 184, geometry_cost 182, nodes 93. Folder 6134 -> 3135.

### Moves

docs/h3/resolutions.md -> docs/h3/geometry.md (rules, "The length axis", "Whole on the audio clock"); its tables -> docs/h3/geometry_cost.md
docs/h3/input_impacts.md -> docs/h3/geometry_cost.md
docs/h3/geometry_and_nodes.md -> docs/h3/geometry.md (geometry half, keyframe canvas); docs/h3/nodes.md (which nodes, node order); modality tags -> docs/h3/conditioning.md ("Three modality tags, not two")
docs/h3/references.md "Sizing a reference image", "The vendor image path, stage by stage", "A third knob", "Should video and audio have fit nodes too?" -> docs/h3/reference_sizing.md
docs/h3/references.md "A label is a bare ordinal", "Prompt structure", "The relationships..." (incl. "Edit a source video", "Replace a character...", "Continue...", "Transfer motion...", "Follow camera movement...", "Reference a voice", "Reference a music style"), "The shipped arms" -> docs/h3/reference_prompts.md
(references.md keeps "The four reference types", "Encoder-only references", "Encoding references apart from the prompt", "Video references", "Audio references", "Frame rate", "What references cost" incl. "Budget by pixel area", "Labels", "Known limitations, collected".)
audio_freeze.md keeps sections 1-9 numbered as before (code cites section 4, 5, 8). custom_node_gaps keeps §5.1 and section 8. conditioning keeps sections 1 and 1b. comfyui_vendor_gaps keeps gap numbers (gap 16 cited by reference_conditioning.py).

### Removed or corrected claims

- geometry_and_nodes.md: "the default pair is `er_sde` / `simple`" -> `h3_config.SAMPLING` is `euler`/`simple` since 2026-10-05 (custom_node_gaps already said so).
- geometry_and_nodes.md: "wire `MiniMaxH3ReferenceReport`" and "the node is still installed" (`MiniMaxH3KeyframeCanvas`) -> both nodes retired (0.174.0 / 0.173.0); the conditioner's on-node preview carries the reference report's pricing; `resolve_keyframe_geometry` remains as a library.
- geometry_and_nodes.md: node list showed `MiniMaxH3SageAttention` as the dense node and the "use from this repo" list as current -> the default dense kernel is core's Model Attention Backend (`h3_config.DENSE_BACKEND_NODE`, `DEFAULT_DENSE_CHAIN = "kitchen"`, 2026-09-15); the sage node is the alternate chain. nodes.md.
- geometry_and_nodes.md: dated layers ("until 2026-09-14 this said...", "until 2026-09-17 the block...", Sol node's former names, "This said it sits in all eight shipped graphs until 2026-09-14", "Two percentages withdrawn 2026-09-08", "Until 2026-10-15 this paragraph said one grid step below the ceiling") -> dropped; current answer kept.
- geometry_and_nodes.md: "Video lengths on the audio grid ... community finding relayed 2026-10-14 ... not measured" and resolutions.md's 2026-10-17 version -> one statement in geometry.md, "Whole on the audio clock" (the later, rule-shaped version: best practice, seam rule via `h3_rules.chain_lengths`).
- resolutions.md: "362 is the ceiling ... and the shipped `LONG_LENGTH`" and int32 table "362 ... this is the shipped default" -> stated via constants (`h3_rules.MAX_LENGTH`, `h3_config.LONG_LENGTH`, `LONG_WINDOW`).
- resolutions.md / input_impacts.md: the two int32 sections ("The int32 threshold" and "There is no 100,000-token budget") merged; the "safe because build_kernel refuses..." claim scoped to the sage node, since the default dense kernel is now core's backend node.
- input_impacts.md: "311 is the value point", "Skip 328", "1024x768 at 311 is the cheapest aligned combination" -> dropped as Morton-alignment advice; Morton retired 2026-09-27 (the page itself had marked that section history). Kept: the Sol token-floor table and a short summary of the block-geometry findings in geometry_cost.md, pointing at attention/morton.md and git history for the maps.
- input_impacts.md: "Reproducing everything here" commands -> dropped: `bench/analyze_canvas_geometry.py` and `bench/analyze_morton.py` are not in the tree.
- input_impacts.md: "`bench/bench_e2e_h3.py:439` sets [the 60,000 floor]" -> it is `SOL_TOKEN_FLOOR` at line 462; cited by name.
- references.md: dated opening layers (what "the reference pipeline" meant until 2026-08-21, "Written 2026-08-13 ... corrected ...") -> one sentence on sglang as authority and diffusers as portability target.
- references.md: "Since 2026-09-13 a freshly dragged AppendRefImage does ... `qwen_view=shared`" with a 2026-10-03 italic note -> current defaults stated once (vendor parity for the video copy; `qwen_view=separate` at `REF_QWEN_SHORT_EDGE` since 2026-10-03), pointing at `define_schema`.
- references.md: "The four combinations -- PRE-FOLD WIRING ONLY" (ReferenceFit + core node, "10 of the 18 graphs that wire ReferenceFit", node "NO CHANGE" log) -> removed: `MiniMaxH3ReferenceFit` retired in 0.173.0 and no graph wires it.
- references.md: "the text segment lands 75-160 rows above the reference segment ... images cost double is an understatement" vs the later "removed on 2026-08-28 as unsupported" -> kept the measured ladder numbers, dropped the interpretation; pricing pointed at preflight and the conditioner's preview.
- references.md: "Both stages on three real sizes, and what the v2 decision changes" (v1/v2 W4 table, owner decision of 2026-08-25, "6 upscale-on to 28 off and 39 of 40 on video_policy=encoder") -> one history paragraph; AWQ lane closed 2026-09-13.
- references.md: the encoder-regime table (v1 / v2 / INT8) and its two dated corrections ("widest of the three", "the shipped encoder is v2") -> current fact only: on `ENCODER_INT8` core's bounds bind nothing a reference node produces and match the release's counts.
- references.md: "The video-reference arms ship 362 frames as of 2026-08-16. REF_VIDEO_LENGTH was deleted rather than raised" -> `REF_VIDEO_LENGTH` was reintroduced 2026-08-22 (h3_config.py); the page now says the arms render at `h3_config.LONG_LENGTH` and that no length constant is right twice.
- references.md: keyed-keep store of reference encodes (removed 2026-10-10) dated note -> the current behavior only.
- references.md: mono-audio "WITHDRAWN 2026-08-29" narrative (why it was believed, retired check) -> current fact only, with the records.
- references.md: "`MiniMaxH3ReferenceVideoFit` remains useful for reporting" -> retired 0.173.0; dropped.
- comfyui_vendor_gaps.md: the stacked 2026-08-29 and 2026-09-13 corrections at the top (W4 artifact as shipped encoder, `video_policy=encoder` dormant), the "Adjacent checkpoint-format status" AWQ section, the W4 rows of the processor-policy table, the per-gap "Everything built against these gaps" and "Native status versus enforcement" tables -> one summary table with status and enforcement; AWQ facts reduced to one Settled bullet.
- comfyui_vendor_gaps.md: closing paragraph "the current W4 artifact's narrow still snapshot remains the unhandled deployed-path issue that motivates v2" -> removed; superseded by the 2026-09-13 lane closure the same file recorded.
- comfyui_vendor_gaps.md gap 6: "This repo's custom fit nodes report the reached resolution" -> the fit nodes are retired; the conditioner's on-node preview reports it.
- comfyui_vendor_gaps.md Settled: core `421a1c24` "neither core's H3 nodes nor any node in this pack writes [a noise mask]" (read 2026-09-10) -> the freeze and masked nodes write one (audio_freeze.md section 3's 2026-09-25 note).
- comfyui_vendor_gaps.md gap 2's fps table and gap 3/4 bounds table duplicated references.md -> kept in references.md / reference_sizing.md, linked from the gap.
- custom_node_gaps.md: model-branch diagram with `SolAttnMiniMax (vendored)` and `SageChainAssert`; §3 rows for retired nodes (`KeyframeCanvas`, `ReferenceFit`, `ReferenceVideoFit`, `SageChainAssert`, `MarkerArm`, `SolAttnCurve`); "The Sol node every shipped graph wires is ... `vendor/sol_attn_minimax.py` via symlink" -> current chain (core backend or sage node, then `MiniMaxH3Sol`); the vendored-symlink paragraph dropped as it describes the pre-2026-09-27 Sol node. Unsure: whether `vendor/sol_attn_minimax.py` is still loaded anywhere; kept out rather than asserting.
- custom_node_gaps.md §4.1 item 2 (`er_sde` against Euler) -> closed 2026-10-05; one sentence. Item 5 ("proposed correction to references.md's condition-noise row") -> already applied in the vendor-path table ("recipe yes, draw no"); folded. §7 item 5 removed for the same reason.
- custom_node_gaps.md §5.2 "Four live configurations: sage+Sol (the default), sage alone, the SLA router, stock SDPA" -> two dense chains under Sol (kitchen default, sage rotated) plus the stock baseline; the SLA router is retired.
- conditioning.md: the two correction stacks at the top (2026-08-29, 2026-09-13) and the `encoder`-policy history sentences -> one current statement.
- conditioning.md section 2: "Nothing splits them on the shipped path since 2026-09-13 ... `qwen_view=shared` is the append node's default" -> `qwen_view=separate` is the default since 2026-10-03; rewritten.
- audio_freeze.md section 3: "No shipped graph writes a `noise_mask` (verified 2026-08-30)" with a 2026-09-25 "stale" note -> the freeze nodes write one.
- audio_freeze.md section 6: "The trained ceiling does not land on the audio grid; `LONG_LENGTH` does" -> `LONG_LENGTH` is the ceiling since 2026-10-15; `LONG_WINDOW` is the one on both clocks.
- audio_freeze.md section 4: the clips-to-judge table, "built while the owner was away", "first run still owed", the retired shots-per-window graphs, and the 2026-10-04 masked-lane narrative -> one verdict table and a short list of what was built; the masked-lane detail points at docs/masked_v2v/overview.md, keeping the not-a-trained-task, one-sampler, shadow and margin facts and every record link. Owner quotes removed.
- image_editing.md: the full scene, format-ladder, cost and validator narrative -> a short past-tense record of what it was and what it found.
- stages.md, uses.md: "last updated" layers and the quoted owner ask removed; CLAUDE.md references -> root AGENTS.md.

### Left for the lead

Inbound links to repoint (old page -> new):
- AGENTS.md (root): `docs/h3_resolutions.md`, "The length axis" -> `docs/h3/geometry.md`, "The length axis" (heading kept).
- prompt_bank/AGENTS.md: `h3/resolutions.md`, "The length axis" -> `docs/h3/geometry.md`, "The length axis".
- docs/method/evidence.md:79 and docs/state/decisions.md: `resolutions.md`, "Whole on the audio clock" -> `geometry.md`, same heading. evidence.md:129 `resolutions.md` -> `../h3/geometry.md`. evidence.md:149 "references.md is the authority" for sizing -> `../h3/reference_sizing.md`.
- docs/state/open_experiments.md: `references.md`, "Sizing a reference image" -> `reference_sizing.md`, "Sizing a reference image: the video model's copy".
- docs/state/decisions.md: `references.md`, "Edit a source video" -> `reference_prompts.md`, same heading.
- workflows/h3_config.py:1603 "`docs/h3/resolutions.md` lists it (its 4:3 row)" -> `docs/h3/geometry_cost.md`, "The common ratios".
- bench/check_widget_deviations.py:108,264,270,276,354 (`geometry_and_nodes.md`, `resolutions.md`) -> `geometry.md` / `geometry_cost.md` / `nodes.md` (line 354's "~3227 MiB" head-chunk figure lives in h3_config beside SAGE_NODE, not on any page now); bench/build_sidecar_examples.py:71 -> `geometry.md`; bench/check_doc_links.py:30 (history docstring, may stay).
- Other inbound links found by grep: docs/prompting/prompting.md, docs/working/runbook.md, docs/working/coderef.md, docs/masked_v2v/working_rules.md, docs/masked_v2v/window_context.md, docs/method/prose_measurements.md, docs/method/rules_history.md, docs/research/{2026-09-25_continuation_guide_rows,sglang_comparison,technique_transfer}.md, bench/results/2026-09-17_owner_reported_bad_clips.md, changelog.d/*, CHANGELOG.md (history; probably leave).
- docs/method/checks.md doc-link-absent block: the `reference_fit.py` entry's WHY names docs/h3/references.md and custom_node_gaps.md as citing it; neither does now (harmless, but the WHY is stale). The two `custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/...` entries are still cited by audio_freeze.md section 3.

Restatements of the geometry outside this folder, with the link that should replace each:
- docs/state/roadmap.md:158 "every ratio at or above 1.75 costs the same 1008 tokens/frame" -> **wrong** (h3_config.py's CANVAS_TIER note corrected it 2026-08-28: the four named canvases were selected; the band varies, worst +7.3%). Replace with "[`../h3/geometry_cost.md`](../h3/geometry_cost.md)".
- docs/state/roadmap.md:104-105 and :144 (362 ceiling, 17n+5 grid, cost) -> "Canvas and length: [`../h3/geometry.md`](../h3/geometry.md)."
- docs/prompting/prompting.md:264-268 (snap up to 17k+5 at 24 fps, achievable durations) -> keep the prompting consequence, replace the rule with "the frame grid: [`../h3/geometry.md`](../h3/geometry.md), "The length axis"".
- docs/working/runbook.md:152-153 (MAX_LENGTH = 362, 17n+5, audio clock every third step from 39) -> "[`../h3/geometry.md`](../h3/geometry.md), "The length axis" and "Whole on the audio clock"".
- docs/working/comfy_notes.md:280 ("362 frames is the ceiling") -> "the ceiling is `h3_rules.MAX_LENGTH` ([`../h3/geometry.md`](../h3/geometry.md))".
- docs/attention/sol_attn.md:151 ("362 is the longest window and 345 is where diffusers stops") -> "[`../h3/geometry.md`](../h3/geometry.md), "The length axis"".
- docs/research/2026-08-14_sol_bench_plan.md:23 and bench/bench_e2e_h3.py:963 call the int32 crossing "the model's ~100k ceiling" -> it is a sage kernel offset width, not the model's: "[`../h3/geometry_cost.md`](../h3/geometry_cost.md), "The int32 crossing"". (input_impacts.md also named docs/attention/sol_attn.md:271; no longer there.)
- docs/prompting/prompt_bank.md:197 ("every legal value on the 17k+5 grid") is generated; the generator's heading text could link geometry.md.
- docs/masked_v2v/working_rules.md:103,114 and running_a_job.md:80, docs/state/next_steps.md:115, docs/method/checks.md:327, docs/method/evidence.md:72 cite `h3_rules.MAX_LENGTH`/`FRAME_FACTOR` by constant: fine as they are.

Unsure, kept:
- custom_node_gaps.md §7 item 1 (a stale "sage on and Sol absent" comment in build_workflows.py, measured 2026-08-28): not re-verified.
- reference_prompts.md and references.md keep the 2026-08-13/15 reference-cost measurements as dated data; they were taken under `qwen_view=shared`-equivalent sizing and do not price today's `separate` default (said on the page).
- audio_freeze.md section 5 ideas and section 8 vendor read kept nearly whole (cited by code and other docs).

## docs/attention/

Ran the gitignored-link fixer on docs/attention (0 links needed repointing after the rewrite; every
gitignored link was written with `../../` from the start). Every `internal/`, `data/`, `coderef/` and
`bench/results/**/*.json` reference in the original seven pages was checked present in the folder afterwards (95 of
95, by exact string). Checks run: `check_doc_links` (remaining failures are all `data/`, `internal/`,
`bench/results/*.json` or ComfyUI `comfy/` targets), `check_retraction_consumers` (all ok; the six phrases allowlisted
for `sol_attn.md` and the two for `morton.md` are still there, and no new page carries one), `check_skill_routes` (ok),
`build_wiki_index` (0 unreachable).

### Moves

No page was renamed or deleted. Content moved out of the long pages into new pages in this folder. For inbound links
and quoted headings:

- docs/attention/sol_attn.md (sections "Start here" 2-3 "Install the CUDA kernel" / "Verify it", "What the kernel exposes", "The exact branch is all-INT8", "Dense calls take core's container entry", "The node is ours now", "The Triton node") -> docs/attention/sol_kernel.md
- docs/attention/sol_attn.md ("Ordering", "The one silent exception ...") -> docs/attention/sol_wiring.md
- docs/attention/sol_attn.md ("References are pinned exact, and a video reference dominates", "The sink conflates two things", the 2026-08-14 segment breakdown, the segment table) -> docs/attention/sol_sink.md
- docs/attention/sol_attn.md ("`dense_blocks`: useful experimental evidence", "Propagation, measured", "Exact at a block", "The paper's own dense-layer policy is not evidence about H3") -> docs/attention/sol_dense_blocks.md
- docs/attention/sol_attn.md ("Inside Sol: sparsity costs more than the kernel's INT8...", "Which shipped values have evidence, audited 2026-08-28", "Where the exact work actually goes, by depth") -> docs/attention/sol_error.md
- docs/attention/sol_attn.md ("What has actually been measured on it here", "The frontier, at 362 frames", "... at 124 frames", "Where the time goes", "The unit trap", "The sigma window is not a step fraction", "Quality", "Record the commit") -> docs/attention/sol_measured_arms.md
- docs/attention/sol_attn.md ("Live route telemetry", "Where the call's time goes", "A tau per head and a setting per segment", "The tau sweep", "Measurement traps hit while producing this", "Reproducing") -> docs/attention/sol_instruments.md
- docs/attention/sol_attn.md ("What Sana's newer H3 packages offer this card, 2026-09-10" and its subsections) -> docs/attention/sol_upstream.md, section "What we would adopt, and what would decide it"
- docs/attention/sol_attn.md ("The defaults, re-read against the sage-side error records, 2026-09-14", heading kept verbatim) -> docs/attention/block49_quant_error.md
- docs/attention/sol_upstream.md (sections "comfy-kitchen and core, 2026-10-09", "... 2026-10-02", "... 2026-09-25", "comfy-kitchen, 2026-09-22", "comfy-kitchen and core, 2026-09-19", "comfy-kitchen, 2026-09-15: ...", headings kept verbatim) -> docs/attention/sol_upstream_reads.md
- docs/attention/sol_upstream.md ("Comfy-Org/comfy-kitchen, as of 2026-09-11 / 09-10 / 09-08 / 2026-09-04", "2026-09-15, evening: `qk_balance` prepared for upstream") -> docs/attention/sol_upstream.md, section "comfy-kitchen" (current state, consolidated)
- docs/attention/quant_policy.md (status log, "Tier 2 design note") -> docs/attention/quant_policy.md (tier table) and docs/attention/sol_error.md ("The route itself under INT8")

New pages: sol_kernel.md, sol_wiring.md, sol_sink.md, sol_dense_blocks.md, sol_error.md, sol_measured_arms.md,
sol_instruments.md, sol_upstream_reads.md (all in index.md).

### Removed or corrected claims

- **sol_attn.md, `dense_blocks` row and the 2026-09-14 section's note**: said Option A `"39,41,42,49"` and Option B
  were "under trial" to replace `"45,48,49"`. The code ships Option C `"38,39,40,41,42,49"` (`h3_config.SOL_DENSE_TAIL`),
  adopted 2026-10-02 on local probe error and kept by owner decision after the panel. Corrected on sol_attn.md and
  sol_dense_blocks.md.
- **quant_policy.md, attention table**: said "adopted Option A '39,41,42,49' on 2026-10-02" and that the routed steps
  run Sol "with `qk_balance` on". Replaced by Option C (`SOL_DENSE_TAIL`) and `quantizer="rotated"` (default since
  2026-09-27).
- **sol_attn.md, "References are pinned exact"**: called the `2*T*S - S²` column "the v1 formula", withdrawn, "prices
  nothing that runs here". Since 2026-10-02 the shipped `exact_kv_and_all_rows` runs every sink query exact, which is
  exactly that formula, so 2.9% / 8.1% / 50.5% / 31.2% / 57.9% are the current shipped mode's forced-exact shares.
  sol_sink.md now carries it as the `exact_kv_and_all_rows` column; the "do not rely on" row in sol_attn.md now says
  57.9% is wrong only as the price of `exact_kv_and_rows`. Worth a `decisions.md` entry and an edit to evidence.md's
  reference-load row (see below).
- **sol_attn.md, "Quality" / sage accuracy**: said "No kernel has been graded against either [capture]" and "this repo
  has no accuracy figure to quote". Superseded by `data/results/2026-08-18_sage_accuracy_on_capture.json`
  (`bench/grade_sage_on_capture.py`), as evidence.md already records. Corrected in sol_measured_arms.md.
- **sol_attn.md, "Start here"**: the first "Start here: this page, and the two it links to" and the second "Start here"
  (backend choice, install, verify) are replaced by the page table and sol_kernel.md. The backend table (CUDA vs
  Triton, "do not uninstall Triton") is dropped: Triton was deleted 2026-08-16.
- **sol_attn.md, "Read this before quoting any number"**: "Status 2026-08-16 ... the table below was written
  2026-08-14" framing dropped; the do-not-rely rows kept and updated (centroid_tail, reuse_qkv_memory rows now say the
  inputs are gone; the retracted sage-and-Sol row now names the dense kernel under Sol).
- **sol_attn.md, "Know where the gains live"**: the canvas x length token table and "362 is LONG_LENGTH as of
  2026-08-16 ... 345 column is history" removed; the token table duplicates docs/h3/geometry_cost.md ("The Sol-Attn
  token floor"), linked instead.
- **sol_attn.md, "Two shipped configs, not one: PDD arms have their own"** and **"What would replace the eyeballing,
  knob by knob"** including "`end_percent`: there is a rule, and 0.74 is the conservative side of it": removed. Moot
  since 2026-09-11 (`end_percent` 1.0 everywhere; `SOL_PDD_OVERRIDES` and `SOL_END_PERCENT_BY_STEPS` empty). The
  unrun prediction (0.87 vs 0.74 at 8 evaluations) and the derivation that 0.74 sits at PDD's coarsest block boundary
  (`percent_to_sigma(0.75) = 0.8`, index 24 of the 32-point grid) are history for `decisions.md` if wanted.
- **sol_attn.md, "Where this stands, 2026-08-29"**: removed. Its current facts kept (merge-equivalence 7.67e-05 as a
  do-not-rely row; the node history in sol_kernel.md). Dropped: installed-build names (`0.2.31+sol.dae00a1`,
  `0.2.31+sol.24908e1`), the kijai-vs-upstream `diff -rq` and sha256-manifest verification of 2026-08-30 (kijai's
  branch fully merged since), the centroid_tail/reuse_qkv_memory "inert widgets" note (inputs gone).
- **sol_attn.md, "`pooled_tail=False` is SLA"** and **"`centroid_tail` may stop being a toggle"**: removed (SLA lane
  closed 2026-09-27; centroid tail unconditional since kitchen#117). The Sol-Engine no-toggle citation
  (`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:413-424`) survives in sol_upstream.md.
- **sol_attn.md, "Configuration findings"**: removed; the Triton tau guidance (82.3 s) and int8 tooltip moved to
  sol_measured_arms.md; the `dense_blocks` bullet ("NVLabs `0-1` copied 2026-08-26", "ship empty") superseded by the
  2026-10-02 list; the Morton bullet superseded by morton.md.
- **sol_attn.md, "What has no measurement at all"** (audit 2026-08-28: no record for `dense_blocks='0-1'`,
  `sink_conditioning='exact_kv_and_rows'`, `start_percent=0.2`): superseded by the 2026-10-01, 10-02 and 10-05 panels.
  "What would settle the open half" (tau sweep approved 2026-08-28, unrun) superseded by the 2026-10-03 tau sweep
  instrument.
- **sol_attn.md, "The Triton node"**: "Dependency 2 NOT resolved ... `sol_block_probe.py` scaffolded, all stubs",
  "Still unverified ... whether a probe wrapping `optimized_attention_override` sees the CUDA node's DiT calls":
  superseded by the route observer and capture graders. Kept in brief in sol_kernel.md.
- **sol_attn.md, "Reproducing"**: the `centroid_tail` A/B command ("the one experiment with a deadline") and the
  `--sol-backend triton` reproduce command dropped (toggle gone; backend refuses).
- **sol_attn.md, "What is open"**: closed rows dropped (centroid_tail closed unanswered; sink_conditioning at
  reference load de-prioritised; CUDA vs Triton e2e, Triton deleted; block-49 fold row, overtaken by rotation; the
  `dense_blocks` row's "empty is the honest default as of 2026-09-02").
- **sol_attn.md, "The defaults, re-read ... 2026-09-14"**: "`dense_blocks`: keep it empty, and close the adopt `0-1`
  question as no" superseded by the 2026-10-02 list; the four-step fold experiment's open steps overtaken by
  `qk_balance` and `rotate`. Condensed into block49_quant_error.md.
- **sol_attn.md, "What Sana's newer H3 packages offer"**: "Two corrections, carried over from sol_upstream.md, which
  held a version of this table until 2026-09-10" dropped (folded into the table cells); the per-step tau row's "has
  per-block `tau_profile` only" corrected to `tau_table` (tau_profile retired 2026-09-27); the permuted-sink row's
  "has to compose with `morton`" corrected (Morton gone; `rows` now expresses the split, ungraded); token order "ours:
  `morton`, off" corrected to native.
- **sol_attn.md, "Core's `BlockSparseAttention` against ours"**: "Ours as shipped today (qk_balance on, ...)"
  corrected to rotated, kitchen fallback, Option C, all_rows.
- **sol_upstream.md**: the "Last updated" stack and the five `*Corrected 2026-09-10*` notes dropped (current text
  kept); the PR 16239 paragraphs under core's Sol node dropped (closed unmerged 2026-09-16); the four
  "Comfy-Org/comfy-kitchen, as of ..." snapshots folded into one current "comfy-kitchen" section (dropped as
  superseded: "no tag after v0.2.33", "PR 167 open", "PR 156 open", "our `sol-blk-cnt` commits did not cherry-pick",
  PR 171 key_bias rebase hazard, PR 172 reach analysis, PR 146/124 status of 2026-09-04).
- **sol_upstream_reads.md (moved dated reads)**: the 2026-09-25 INT8 VAE paragraph ("the file the owner removed ...
  `bench/check_model_files.py` goes red ... the owner's decision stands") replaced by its 2026-10-02 superseding note
  (the lane reopened 2026-09-26; `MODELS["video_vae"]` names the INT8 file). The 16508 paragraph folded with its two
  dated notes (no `--fast` since 2026-09-28; `MiniMaxH3Sol` takes fp16 since 2026-09-27, so the "every Sol graph would
  run dense" reasoning no longer holds). The 2026-09-25 "by policy it is not built" folded with its 2026-09-27
  reversal (tracks upstream main).
- **morton.md**: cut from 1773 to 140 lines. Removed as superseded or history: the one-canvas evidence box, "Where
  everything is" (vendored line numbers into deleted code), "Should you turn it on?", "What to try next" and "Tests run
  and not run" (lane closed), the six-link assumption chain, Hilbert/serpentine/rotation sweeps, the "`3d` pin selected
  at its best canvas" open question (pin gone), edge and time concentration tables, the related-work survey
  (Reformer, SVG2, Kandinsky, PARO; "Related papers" heading quoted by docs/research/2026-09-19_in_context_reference_and_sinks.md:593
  now has no target section). Added the 2026-09-17 equal-cost result, which contradicts the page's old geometric case
  for `2d_frame`/`hilbert` (both lose to raster on blocks 32 and 49).
- **block49_quant_error.md**: "So what, revised 2026-09-15" status list removed as superseded (it said all levers were
  off by default, the blind comparison waits on the owner, Sol's rotation "served and off", the structural fix "not
  started"); rotation is the default since 2026-09-27. Section 7's "whether blocks 45 and 48 behave like 49 ... no
  capture exists" superseded by the 2026-10-02 capture (folded into section 1). Section 8's "All three levers are
  still off in the default graph" removed.
- **quant_policy.md**: "Today's order (2026-09-15)", "Instrumentation this plan needs" (captures of 45/48 done
  2026-10-02; the audio question closed as take variation) and the dated status log removed; their findings kept in the
  tier table, the open dense-kernel question and sol_error.md.

### Left for the lead

- **Code quoting a moved heading** (heading kept verbatim, file changed):
  - `bench/probe_encoder_rope_kitchen.py:10` cites `docs/attention/sol_upstream.md`, section "comfy-kitchen and core,
    2026-09-19" -> now `docs/attention/sol_upstream_reads.md`.
  - `bench/grade_channel_balance.py:4` cites `docs/attention/sol_attn.md`, "The defaults, re-read against ..." -> now
    `docs/attention/block49_quant_error.md`.
  - `bench/results/2026-09-25_upstream_survey_checks.md:4` and `bench/results/2026-09-19_encoder_rope_kitchen.md:17`
    cite sol_upstream.md dated sections -> sol_upstream_reads.md (records; fix only if records are rewritten).
- **docs/working/coderef.md** (sections "What moved by 2026-09-25" and "... 2026-10-02") says core and kitchen reads
  "live in docs/attention/sol_upstream.md, section ..." -> `docs/attention/sol_upstream_reads.md`.
- **docs/method/evidence.md, retraction-consumers WHY text**: says sol_attn.md's consumers of the retracted 1.16x figure are
  "the do not rely on row and the corrected Configuration-findings bullet"; the bullet is gone, only the table row
  remains. And the "reference-load table: 35.1% / 57.9%" ledger row should note that the 57.9% formula is the price of
  `exact_kv_and_all_rows`, the shipped sink since 2026-10-02 (see sol_sink.md). Add a `decisions.md` entry for that
  correction.
- **docs/state/roadmap.md:139, :489, :972** quote sol_attn.md sections "What Sana's newer H3 packages offer this card"
  (now sol_upstream.md "What we would adopt, and what would decide it"), "What would replace the eyeballing" (removed,
  moot since 2026-09-11) and "The defaults, re-read ..." (now block49_quant_error.md).
- **docs/research/2026-09-19_in_context_reference_and_sinks.md:250, :592-593**: "References are pinned exact" now in
  sol_sink.md; morton.md "Related papers" removed.
- **docs/h3/geometry_and_nodes.md:305** (if it still exists after the h3 merge) cites sol_attn.md "The Triton node" ->
  sol_kernel.md, "The node's history, in brief".
- **docs/h3/input_impacts.md** no longer exists (merged into docs/h3/geometry.md and geometry_cost.md by the h3 agent);
  this folder now links `../h3/geometry.md` and `../h3/geometry_cost.md` ("The Sol-Attn token floor", "Sol block
  geometry by canvas (the closed Morton lane)") for canvas and length against Sol's blocks. The sol_attn.md canvas x
  length token table was dropped in favour of those.
- **Stale code comments in `workflows/h3_config.py`** (prose is a cache; not mine to edit):
  `SOL_RECOMMENDED_CUDA`'s `start_percent` comment says "0.0 on every PDD graph since 2026-10-01 (SOL_PDD_OVERRIDES)",
  but `SOL_PDD_OVERRIDES` is empty since 2026-10-05; the `min_tokens` comment opens "4096 against the node's own
  12288"; `SOL_DENSE_OPTION_A`'s comment says "Adopted 2026-10-02" (it was replaced the same day by B then C);
  `SOL_RECOMMENDED_CUDA`'s `dense_blocks` comment still leads with "Empty again by owner decision on 2026-09-02".
- **Unsure and kept**: sol_upstream_reads.md keeps core PR notes that are not about attention (VAE tile seams,
  torchaudio, text-encoder PRs) because coderef.md delegates core reads to this folder and some are still current
  (next_steps.md carries several). It could move to `docs/research/` or `docs/working/` as an upstream log.
  sol_upstream.md remains 755 lines because it holds per-upstream source citations (coderef pointers) that were all
  kept.
- Viewer verdicts in block49_quant_error.md section 6 keep short quoted phrases from the 2026-09-15 viewers ("way
  better", "super close, maybe faces"); the owner's own remarks were paraphrased. A reviewer may want those paraphrased
  too.

## docs/distills/ and docs/prompting/

Ran the gitignored-link fixer (docs/distills/*.md docs/prompting/*.md) (0 links repointed; the rewritten pages were
written with correct `../../data/...` paths). Every gitignored reference in the original pages (internal/, data/,
coderef/, bench/results/*.json) was checked by grep to survive somewhere in the same folder; none was lost.
Checks run: `check_doc_links` (only environmental failures: data/, internal/, comfy/), `build_wiki_index` (0
unreachable), `check_prompt_docs_sync` (all ok: 12 pins, 17 section-10 examples, 41 scenes with verdicts),
`check_skill_routes` (ok), `check_retraction_consumers` (ok; the retracted ref2v phrase kept in ref2v_distillation.md
inside its withdrawing parenthesis). `pdd_artifact_inventory.py --check` DID NOT RUN (models/loras/h3 not on this
box); only prose outside the generated regions was edited and no commit hash was added.

### Moves

- `docs/distills/pdd.md` -> split: `docs/distills/pdd.md` (overview, contract, traps, bake, measured) and new
  `docs/distills/pdd_node.md` (node, pdd_math and converter internals; checks against the paper, Kijai, core,
  UtilsCollection, T8; what was borrowed). Inbound links to pdd.md sections on the node internals or cross-checks
  ("Where the step index comes from", "Validated against the paper and an independent conversion", "Core is learning
  this", "A third implementation", "What was borrowed") -> `docs/distills/pdd_node.md`.
- Headings renamed or removed in pdd.md that code or docs may quote: "What a backbone bake pins" KEPT (no date suffix
  now: was "What a backbone bake pins, traced 2026-08-31"; code quotes it without the date). "Enforced by nothing" ->
  "Not enforced, and not measured" (cited by `pdd_lora.py:2346` as history; harmless). "The sweep, 1 to 10
  evaluations" -> table under "Schedule: the node emits it" (`workflows/h3_config.py:2681` says "pdd.md's sweep table"; still
  true). Composition table -> "What is measured", "Where the file's run-time footprint is" (`pdd_lora.py:737`).
- prompting.md: no section moved or renumbered; every section the h3-prompt skill names (2, 3.1, 3.3, 4, 5, 5.10, 6,
  8, 9, 10, 12, 14.3, 16, 16.4, "Four layers...", "Where everything else lives") is where it was. §13 lost its
  "Withdrawn 2026-09-01" and "Still open" subheadings (content merged into one list).
- prompt_audit.md: dated "Verdicts added <date>" subsections merged into grouped tables; "The four that matter",
  "Misalignments", "Marker coverage", "House-rule findings", "What to design next", "The cheap gate" -> "Open findings"
  and "What to render next".

### Removed or corrected claims

distills/pdd.md:
- Said the backbone merge into int8 at run time "ships today" (fidelity table row) and framed `unmerged_blocks` and the
  bake as the alternatives. Superseded by `backbone_apply` defaulting to `exact branch` (owner, 2026-09-26, 0.154.0;
  `pdd_lora.py` input, `lora_branch.py`, `docs/attention/quant_policy.md`): the merge is now the control arm and
  `unmerged_blocks` works only under it.
- Said "no baked artifact has been built", "no stripped file is on disk", "What does NOT exist yet is the baked
  checkpoint". Superseded: the fl2va bake and stripped sidecar exist since 2026-09-05 (`MODELS["unet_fl2va_pdd8_baked"]`,
  `PDD_FL2VA_STRIPPED_LORA`) and the masked-lane and song graphs load them (`workflows/INDEX.md`).
- Said core's PR #15908 was open and that our node would break on merge (2026-08-27 read). Merged 2026-08-29
  (`2504e68d`); the arity-transparent patch is the current fact; the open question (keep the head half?) kept.
- Kept a "reported, not verified" block that Kijai may have re-uploaded with delta encoding. Superseded by the
  2026-09-25 confirmation (HF commit `f94b1bcc94`, local copies stale), now stated plainly in pdd_node.md.
- The "Why audio suffers more than video" section's carry mechanism and its scored prediction table. Refuted
  2026-09-25 (`docs/research/pdd/2026-09-25_upstream_pdd_comparison.md` finding 1; `docs/method/evidence.md`); kept
  as one line plus links.
- Partition fidelity at 39 frames ("`[28,2,1,1]` refuted", then "ordering inverted at 362", then "caveat withdrawn").
  Replaced by the current reading: the metric does not separate partitions at 362 frames; quote neither ordering.
- "Our legality rule is wrong in both directions ... 6 REFUSED" and "Rows 5, 6 and 7 read raises". Superseded
  2026-08-29/31: `resolve_emit_steps` accepts envelope partitions 5-7; the sweep table shows the current behaviour.
- The Sol `end_percent` history (0.74 per PDD step count, step-table lookups going stale on a hand edit, 1.0 since
  2026-09-11). Replaced by: PDD graphs take `SOL_PDD_CUDA` = the base recipe, overrides empty; see sol_attn.md.
- "The node fuses the bank at load" (composition section) and "keeping only fused heads saves 31 MiB". Superseded
  2026-08-29: `_FusedHeads` fuses on first use (pdd_lora.py docstring).
- "The 2026-08-28 deprioritisation of the bake" as a constraint: the page itself said it was stale as of 2026-08-31;
  dropped.
- "Do not justify it with the OOM": kept as one sentence (17.5 MiB, record link kept).
- The "fl2va and ref2va time curves differ by 7.8% relative" sentence. Removed: `docs/method/evidence.md` row marks it
  UNSUPPORTED and asks for it to be deleted from the two places that state it (pdd.md was one).
- Converter version 1/2/3 metadata narrative and the 2026-08-29 first-unpruned-build story: duplicated
  pdd_artifacts.md sections 4 and 5; removed from pdd.md.
- The 2026-08-26 renders' md5 table, the container-hash two-mechanism note and the 2026-08-28 SIGMAS-rewiring render
  table: reduced to their current lesson (hash decoded frames; warm-up transient; SIGMAS output is consumed and
  rewiring moved no render) under "Traps"; the full story is in `archive/docs/pdd_handoffs/2026-08-28_handoff.md`.
- The "Distilled motion looks WRONG" section: reduced to its numbers and the owner's refined reading under "What is
  measured".
- Many "Corrected <date>: this said X" layers (patch_heads installs one patch; 16 steps ~2e-3; "Both of us accept any
  step count"; the "two instances, not three" latch note; "4 evaluations were our own extrapolation"; the
  unconsumed-keys row "never true"; "per partition withdrawn"; "+19 MB per block" pricing): only the current statement
  kept. The withdrawn +19 MB figure is replaced by "both arms staged 19995 MB" (pdd_lora.py:1711 cites this).

distills/pdd_artifacts.md:
- "fused at load (time)" for the head bank, three places. Corrected to "fused per block on first use"
  (`pdd_lora.py::_FusedHeads`).
- Glossary said the first backbone bake pair "has not rendered". Removed; graphs load it (workflows/INDEX.md). The
  2026-09-05 changelog entry's "Not rendered" left as that date's record.
- Removed the "**Withdrawn: ... dated the retirement to 2026-08-27**" notes (two) and the "(An earlier version of this
  line said ...)" aside; the git-dated statement stands.

distills/which_distill.md:
- The er_sde mechanism bullet stated as current, with a "Dated note, 2026-10-05" above it. Now: the base samples Euler
  since 2026-10-05 (`h3_config.SAMPLING`), and every base-vs-distill comparison on the page predates it.
- The averaging inference cited "the clone at 1 s on subway" as support, and PDD8 "Bad" said it "clones a guy". Folded
  in the 2026-09-27 base control (not a clone; `bench/results/2026-09-27_clone_base_control.md`).
- The i2v finisher line with its italic correction: now states the daily graph ships; the 2026-09-27 remark kept as
  what was seen.
- Owner quotes paraphrased per the writing rule; verdict ids (ow-fd-08/09/11) kept.
- Paths changed from `../bench/...` (wrong relative) to repo-relative `bench/...`.

distills/ref2v_distillation.md:
- Opening premise "lightx2v has shipped three FL2VA turbo LoRAs and none for ref2v" with an "Overtaken in part"
  banner. Now states that dedicated ref2v distills exist (lightx2v ref2v turbo on disk, PDD Ref2VA) and Fact C is
  retired as a universal.
- "If you run it anyway" kept as guidance for reopening the lightx2v turbo lane (closed 2026-09-26).

prompting/prompting.md:
- Header "last updated" line and the "internal/PROMPTING.md was superseded and is now DELETED" paragraph: folded into
  §14.5 and the opening.
- §3.1: dropped "Until 2026-09-26 this paragraph said the rule was 'NOT yet measured in either direction'" and the
  owner's verbatim remarks (paraphrased); pinned sentences kept verbatim.
- §5.8, §5.9: dropped the "Added 2026-09-01 ... the portable standard was ahead of this file" history.
- §10 intro: dropped "Until 2026-09-01 this section asserted its examples grade clean and nothing checked it".
- §11: dropped "this row said 'nothing' until 2026-09-01" (two rows), the escaped-instance aside, and "Narrowed
  2026-09-20".
- §12.13: dropped "Added 2026-09-20"; its pointer to "the two retracted rules in §13" now points at "Four layers"
  (§13 no longer lists them).
- §13: removed "Withdrawn 2026-09-01" (L2VA has no shipped prompt; no graph carries a marker but `<d>`; every ref2va
  prompt is one shot at 42-68 words; `scene_prompt()` defect, fixed 2026-08-28, function deleted 2026-10-01) and the
  WITHDRAWN blockquote ("Camera-motion vocabulary is enforced by nothing"). The two commands kept.
- §14.1: removed struck row 5 (`internal/PROMPTING.md`) and its paragraph; one line says it was deleted.
- §14.2, §14.3, §14.4, §14.5, §15 intro: history removed ("Removed and restored the same day", `a52999d`; the first
  pass's wrong vendor-turn count; "A first pass here flagged twelve"); findings kept.

prompting/prompt_audit.md:
- Dropped verdicts on scenes the catalogue no longer has: `t2va_dancer_shot_floor`, `t2va_dancer_shot_face`,
  `ref2va_masked_subject_swap`, `ref2va_masked_subject_motion`, `ref2va_masked_person_silent`, and the non-key rows
  "r2v-swap family" (defined a `<Picture N>` ref §2.2 says not to define) and "the composed ref2va scenes not named
  above".
- `R2V_PROMPT` row read **rewrite** ("describes 10 s on a 15.083 s graph"). The banner said `d5be353` acted on it and
  the catalogue text is now a fourteen-second take; row now "rewritten `d5be353`".
- `LONG_T2V_PROMPT` "rewrite ... all still present" section superseded by the 2026-08-28 rewrite; kept as an open
  finding with the defects and the outcome.
- "`REF_SCENE_SHOTS` exists and is unreachable" and design item 5 "Wire it through `scene=`": superseded (the
  2026-10-01 note: `_ref_prompt` takes `scene=`, the scene arms are wired).
- "Nothing has been rendered through the marker arms": superseded (rendered; `2026-08-28_scene_arm_renders.md`); no marker verdict.
- "The cheap gate this audit argues for" (a motion denylist, unbuilt): superseded, `check_camera_vocabulary.py` runs.
- Done items 1 and 2 of "What to design next" removed; the L2VA render finding kept under "Open findings".
- The 2026-09-03 `derived:<graph>` key rename note removed (history; no current key affected).
- Misalignments section replaced by a link to prompting.md §14.3 (duplicate).
- Owner quotes ("fantastic", "both are pretty awesome") paraphrased except "fantastic" kept as the verdict word.

### Left for the lead

- `docs/research/2026-08-28_scene_arm_renders.md` (viewing guide of 2026-08-28) and `docs/research/2026-08-16_prompt_length_experiment.md`
  (pre-registration of 2026-08-16 with its verdict) are dated records: recommend moving both to `docs/research/`
  (e.g. `research/2026-08-28_scene_arm_renders.md`, `research/2026-08-16_prompt_length_experiment.md`) and updating
  links in prompting.md (§9.10, §9.11, §16.1, "Where everything else lives"), prompt_audit.md, `docs/prompting/index.md`.
  I did not edit them.
- `pdd_lora.py` `unmerged_blocks` tooltip is stale against the 200-module three-kind analysis: it says worst-first is
  "49, 15, 20, 11, 7, 10, 18, 23, 14, 17", '-1' recovers 100%, worst 5 recover 15%, worst 25 recover 68%. The newer
  analysis (`bench/analyze_pdd_unmerge_curve.py`, kept in pdd.md) gives `7, 20, 15, 11, 8, 10, 9, 6, 14, 23`, 12.6% /
  59.9% / 76.2% (fc2 unreachable). Code change, not mine. The same tooltip still says "Measured ... 0.00942 to 0.01058"
  framing of merge as the shipped path, now the control.
- `workflows/h3_config.py` PDD comment block (around `PDD_STEPS`) says "the node fuses the heads at load"; the node
  fuses on first use.
- `bench/grade_pdd_partitions.py:319`, `bench/run_audio_carry_arms.py:322`, `bench/analyze_pdd_stream_energy.py:342`
  print "prompt_audit.md verdicts `rewrite`" for the market prompt; the audit row now reads "~~rewrite~~ rewritten
  `d5be353`". Still true of the pre-rewrite text they used; consider rewording.
- `pdd_lora.py:938`, `:1535` and `workflows/build_workflows.py:4377` say "See docs/distills/pdd.md" for the core/two
  implementations note and the file pairing; the core detail is now in `docs/distills/pdd_node.md` (pdd.md keeps the
  open question). `bench/convert_pdd_lora.py:81` cites pdd.md's "worst 1.1e-4" (kept in pdd.md).
- `bench/measure_marker_epsilon.py:45` cites pdd.md for dense SDPA at 2.4x: kept in pdd.md under "Attention".
- `docs/method/evidence.md` row "7.8% relative ... delete the figure from the two places that state it": removed from
  pdd.md; `bench/convert_pdd_lora.py`'s docstring is the other place.
- `docs/research/pdd/README.md` says "`docs/distills/pdd.md` owns the contract and every file here defers to it":
  still true; add `pdd_node.md` there if wanted. Its last line said the README restates numbers pdd.md owns.
- `docs/state/next_steps.md` line 1323 at `901bee1` cites which_distill.md "Rules of thumb, provisional": heading kept (moved up the
  page). `bench/latent_path_distance.py:4` and `workflows/build_workflows.py:5138` cite "Tests that would move this section":
  kept.
- Unsure, kept: pdd.md "Not enforced": the `_comfy` runtime-injection `KeyError` claim (2026-08-28 diff of key
  inventories) is unverified against current code; the h3-prompt skill text is unchanged and its `reviewed:` commit
  will show prompting.md changed (report-only).

## docs/masked_v2v/

Ran the gitignored-link fixer (docs/masked_v2v/*.md) (0 links repointed; every page here stays at the same depth).
`check_doc_links.py docs/masked_v2v/*.md`: only `comfy/`, `comfy_extras/` env failures remain (as at baseline).
`build_wiki_index.py`: 0 unreachable. `check_retraction_consumers.py`, `check_skill_routes.py`: ok.

### Moves

Pages:

- docs/masked_v2v/who_is_who_evidence.md -> docs/masked_v2v/who_is_who.md (merged; page deleted)
- docs/masked_v2v/sam31.md (sections "Where our SAM 3.1 code lives...", "Layer 1...", "What `MiniMaxH3SAM31Corrections` leaves as ComfyUI computes it (`NOT_CORRECTED`)", "Layer 2...") -> docs/masked_v2v/sam31_core.md (new page)
- New pages: pieces.md, text.md, working_rules.md, running_a_job.md, renders_by_form.md, limits_and_directions.md, sam31_core.md

Quoted headings (heading text kept; repoint the quotes):

- docs/masked_v2v/overview.md "The frame every masked text is written in" -> docs/masked_v2v/text.md (quoted by data/AGENTS.md:38, bench/load_text.py:209)
- docs/masked_v2v/overview.md "Say the least first" -> docs/masked_v2v/text.md (now a subheading; decisions.md:2771)
- docs/masked_v2v/overview.md "The text: the fragment" -> docs/masked_v2v/text.md (bench/examples/masked_job_lid_marks/README.md:167; decisions.md)
- docs/masked_v2v/overview.md "One whole text, as an example" -> docs/masked_v2v/text.md
- docs/masked_v2v/overview.md "The rule: data before a render, and the same data after" -> docs/masked_v2v/working_rules.md (data/AGENTS.md "How the lane works"; decisions.md:2779; perception_models.md)
- docs/masked_v2v/overview.md "Reading one window whole: its text, its frames, its mask, its audio" -> docs/masked_v2v/working_rules.md (data/AGENTS.md)
- docs/masked_v2v/overview.md "Rules of the lane" -> docs/masked_v2v/working_rules.md (data/AGENTS.md; decisions.md:1163,1192,1236)
- docs/masked_v2v/overview.md "Running a job: one tool, on the video's analysis folder" -> docs/masked_v2v/running_a_job.md (bench/examples/masked_job_lid_marks/README.md:47; decisions.md:1253)
- docs/masked_v2v/overview.md "A shot, made whole: the recipe the owner kept on 2026-10-15" -> docs/masked_v2v/running_a_job.md (research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md:22)
- docs/masked_v2v/overview.md "Two neighbouring shots in one load: a tool, not a default" -> docs/masked_v2v/running_a_job.md (decisions.md:1258)
- docs/masked_v2v/overview.md "Depth: who is in front" -> docs/masked_v2v/running_a_job.md
- docs/masked_v2v/overview.md "Seeing what the tracker and the masks did" -> docs/masked_v2v/running_a_job.md (decisions.md:2786)
- docs/masked_v2v/overview.md the two-subject bullet of "Running a job" ("the two-subject section"; postmortem 2026-10-15:51) -> docs/masked_v2v/running_a_job.md, cut to a closed-lane bullet
- docs/masked_v2v/overview.md sentence "A two-subject load is not rendered zoomed in" (quoted by bench/shot_to_load.py:3585) -> docs/masked_v2v/pieces.md, "A small subject, rendered zoomed in"
- docs/masked_v2v/overview.md "Who is who: a session reads first, the owner answers, and both are kept" -> docs/masked_v2v/who_is_who.md
- docs/masked_v2v/overview.md "The pieces, in the order a render meets them" (with ### 1-4) -> docs/masked_v2v/pieces.md
- docs/masked_v2v/overview.md "The region" (bold item of piece 2; next_steps.md:1175) -> docs/masked_v2v/pieces.md
- docs/masked_v2v/overview.md "the prompt" (piece 3; next_steps.md:1183) -> docs/masked_v2v/pieces.md
- docs/masked_v2v/overview.md "Nothing is kept between runs by a key" -> docs/masked_v2v/pieces.md (next_steps.md:1207)
- docs/masked_v2v/overview.md "A latent step is several frames, and its grid is fixed for a whole load" -> docs/masked_v2v/pieces.md (bench/two_shot_load.py:54; decisions.md:621; window_edges.md)
- docs/masked_v2v/overview.md "A small subject, rendered zoomed in" -> docs/masked_v2v/pieces.md
- docs/masked_v2v/overview.md "What is known from renders, by form" -> docs/masked_v2v/renders_by_form.md
- docs/masked_v2v/overview.md "The motion video as a body model, with marks drawn on its face" (and its ### "What the reference video was", "How it was made: the nodes and the code", "What it carried on that shot", "Why we think it worked, and what is not shown", "Limits known today", "Reading a render of this form", "What comes next for the video") -> docs/masked_v2v/renders_by_form.md (decisions.md:1077, 1632)
- docs/masked_v2v/overview.md "Known limits" -> docs/masked_v2v/limits_and_directions.md (decisions.md:3083)
- docs/masked_v2v/overview.md "Directions" -> docs/masked_v2v/limits_and_directions.md
- docs/masked_v2v/overview.md "What it is" stays in docs/masked_v2v/overview.md (next_steps.md:1142)
- docs/masked_v2v/overview.md "Where to look" stays in overview.md
- docs/masked_v2v/overview.md "How this page grew" -> removed (history; see below)
- docs/masked_v2v/who_is_who.md "section 5" ("5. Seeding it up front"; quoted by bench/results/2026-10-10_who_is_who_across_shots.md:513) -> docs/masked_v2v/who_is_who.md, "The design further ahead", the bullet "The casting pass (seeding it up front)". The numbered sections are gone.
- docs/masked_v2v/who_is_who_evidence.md all sections -> docs/masked_v2v/who_is_who.md ("The plan: two votes that agree, or a question", "The evidence, and whether it is used", "Cuts: finding them, and making a missed one harmless")
- docs/masked_v2v/window_context.md "Results, dated" stays (h3_config.py:1501, shot_to_load.py:643, next_steps.md:35)
- docs/masked_v2v/state_signals.md "A mouth open with no voice" stays (capture_masked_run.py:5481)
- docs/masked_v2v/perception_models.md "The trials" stays

### Removed or corrected claims

- overview.md: "How this page grew" (reorder of 2026-10-12, additions of 2026-10-13) and the header's dated edit list: removed as change history; the 2026-10-12 record is `bench/results/2026-10-12_masked_v2v_page_held_to_its_clips.md` and decisions.md under that date.
- overview.md: the 2026-10-12 italic note that "the masking board" names a retired board: removed (the Open List is the one place; decisions.md 2026-10-12). Card names kept where cited are gone with the narrative.
- overview.md, "What it is": "two more graphs since 2026-10-06, neither rendered when that was written"; "(Said until 2026-10-07 that no shipped graph ticked them)"; "(Said for a few hours that day that the wording had not been watched)": removed; current graph table only. Added the second probe graph `h3_probe_v2v_masked_upper_song_ref2va_motion_cache_api.json`, which the page did not name.
- overview.md, "What it is": the page implied jobs are built from the shipped graphs; corrected: the chain (`shot_to_load.py`, `load_chain.py`, `masked_job_graph.py`) builds every load's graphs; the shipped graphs are single graphs to open by hand.
- overview.md, "The frame every masked text...": "Moved here on 2026-10-12, word for word" note; step 2's "Corrected 2026-10-15: until then this step asked for where the subject is..."; step 3's "Until 2026-10-15 this step said Nothing about action, expression or voice"; "Until 2026-10-15 this said A voice is measured (a separation of the track says where one is)": removed, current text only (the 2026-10-15 rulings are in decisions.md).
- overview.md, "Reading one window whole": "Not built ... the statement of who is in each window (`docs/state/handover.md`, After the close)" and its 2026-10-12 note that handover has no such section: replaced by "who is in each shot is data/<video>/shots.json, answered on the who page".
- overview.md, "Reading one window whole": the text row pointed at `bench/masked_job.py build` printing a text; now points at `bench/what_h3_got.py` on the queued graph first (root AGENTS rule). The audio row's "an audio: line... a sentence, not yet data: the audio in a load's record is not landed" dropped as unverified-stale; kept the capture's voice table and `shots --loads`.
- overview.md, "Rules of the lane": the bullets on what a text says (not an edit, sings only) and on the head-reference standard restated data/AGENTS.md "Settled for the masked lane": replaced by links; the reasoning and evidence (verdicts lines, 604-948 figures) kept. "Until 2026-10-15 this said the text is a plain caption of the finished shot..." removed.
- overview.md, "Rules of the lane", window length: "Until 2026-10-15 this said a shot was divided between windows only when it has to be... and that a load longer than the planner's longest length ran in chained windows even where MAX_LENGTH would hold it whole"; the "(Two short shots could share one window... no load is made that way today. Later the same day: ...)" layer: removed, current rule only.
- overview.md, "A shot, made whole": "Until 2026-10-15 evening this said the context as long as the planner allows"; "Until 2026-10-15 evening this said never by default" (voice): removed. "a longer context, and context_noise on a load of several windows, are being built (next_steps item 1)": replaced by the built flag `--context-noise X --context-noise-why` (window_context.md, 2026-10-15 entry).
- overview.md, "Running a job": "Added 2026-10-10" layers; "A load that holds a cut cannot be built for now ... 2026-10-13: until that day this bullet ended there": folded into one current sentence (ONE_SHOT unless `two_shots`).
- overview.md, "Running a job": the two-subject-in-one-pass bullet (margin, capture reading of both, the pair command's refusals, the first render's details, "No render had been made of one when this was written"): cut to a closed-lane bullet (roadmap, Closed lanes, 2026-10-15). Kept: the closure, `second_lead_0852` blended, the tooling (`shot_to_load.py pair`, `one_subject_view`), the core reading that nothing binds a reference to a hole, and the next test. Dropped detail (still true of the unused tool): one margin for both, the first subject's region first, which rules read the first subject only, the strip's per-subject colours. The tool's docstring holds them.
- overview.md, "Running a job", hair band: "No render had been made with a band when this was written, and this page says nothing of what one does" and "goes into her face-and-hair loads from here": dropped; data/AGENTS.md says the second subject's hair band was retired as a default 2026-10-17. Kept what the key does and `read_hair_band.py`.
- overview.md, "Running a job": "until 2026-10-12 they refused the load instead" (looks): removed. Added `load_chain.py no-look` (from its docstring).
- overview.md, "Running a job", "Not built" list: the seeds item ("a subject's seeds in a job file") is built (`subject_track.py::parse_seeds`, cast seeds); the cheap-variant item is built as `bench/sweep_loads.py`; the "load over several shots" item is built as `two_shots`. Kept as built; remaining not-built items listed.
- overview.md, "Seeing what the tracker and the masks did": "Written 2026-10-10, after a session had to be told this tooling existed"; "Cut on 2026-10-12 from a paragraph that restated the docstring"; the "Until 2026-10-12 this paragraph ended None of the captures ... carries a field for which person" note: removed. Added that the sidecar (`bench/sidecar.py`) is written beside every video.
- overview.md, piece 1 (Subject Track): the detailed bullets (cuts, people, pick, match, one-person clip, tracking, corrections, regain, subject_from, one detection a person, seeds, who is who after a cut, why not core's tracker) duplicated sam31_core.md "Layer 2": replaced by a short summary linking sam31.md/sam31_core.md/who_is_who.md. Dropped dated layers: "the shipped default is most central since 2026-10-07 ... it was largest for every render before that date"; "before that each axis was scaled alike"; "since 2026-10-07 looked for again ... before that it was tracked once"; the 2026-10-10 note on the removed sum over all subjects (kept in who_is_who.md); "(added 2026-10-10)" tags. `subject_from` and its limit kept in limits_and_directions/sam31_core; `bench/join_stretches.py` mention dropped (still in bench/INDEX).
- overview.md, piece 2: `edge`'s "Until 2026-10-12 this said nothing had rendered with it"; `motion_video`'s 2026-10-11/2026-10-12 italic notes ("Built at the owner's word (Open List row 16) ... reaches the card at the server's next restart"): folded into current statements; Open List row 16 citation dropped.
- overview.md, piece 2: "No kept mask" bullet and piece 3's "Until 2026-10-12 this said the first time; the mask is kept..." : removed (the mask store is gone; pieces.md "Nothing is kept between runs by a key").
- overview.md, piece 2: added the composite's job default `h3_config.MASKED_LOAD_COMPOSITE` ("whole region") and `held_mask` (from pipeline.md and window_context.md).
- overview.md, piece 3: "The head and whole-person roles share the movement sentence and are otherwise as they were" (2026-10-07) and the 2026-10-10 note that "as they were no longer holds": replaced by the current rule. The 2026-10-11 note that the body-model form stays a documented practice and not in the node: kept as "the job chain's text is bench/load_text.py's".
- overview.md, "Nothing is kept between runs by a key": the "A section of its own since 2026-10-12" note and "Until 2026-10-12 this list named video_mask.py too": removed. The table's "what it was keyed on" column cut.
- overview.md, "A latent step...": the step-grid audit row's note "until 2026-10-12 this row named that record's .json, which left the tree..." folded to: tables' data under the untracked data/2026-10-10_recomposite_decodes/. "a function that ranked every start one cycle back was removed on 2026-10-10" dropped. The "What frozen is, and why a window follows its context" paragraph duplicated window_context.md "The mechanism": replaced by a link.
- overview.md, "A small subject, rendered zoomed in": "Built 2026-10-10; when this was written no render had been made with it"; the dated lines of 2026-10-10 (first plan, CUT DOWN to what its one render used, largest_zoom input removed, the cut of a wired video to the box refused) and 2026-10-13 (a zoomed run takes a wired motion video, uncut; Open List row p04): folded into current statements. Corrected by code: the chain renders zoomed by default (bench/shot_to_load.py, "The defaults a load carries"), where the page read as a niche option.
- overview.md, "What is known from renders, by form": added a row for the head-cut face-points form (owner's verdicts 2026-10-12) and the open held tail.
- overview.md, the marks section: "Provisional, written 2026-10-11 ... Two more shots were rendering"; "Tracked since the night of 2026-10-11; until the names landed that night this step said the tool could not run from a fresh checkout"; "Tracked since the evening ...; the kitchen shot's video was drawn before that by the one-off"; "Until the night of 2026-10-11 this paragraph said the pose pass was a one-off script"; "A shorter wording, movements and expressions, was under test ... had not rendered"; "Not rendered when this was written. Rendered later the same evening" (mouth); "Until that day this line read takes no motion video" (zoom): folded into current statements, one-off script paths kept.
- overview.md, "The text: the fragment": the owner's quoted words of 2026-10-12 18:00Z (pattern condition) replaced by a paraphrase pointing at decisions.md (writing rule: no quotes of passing remarks); "SINCE 2026-10-14 THE ONE HOME ..." paragraph rewritten as "Who writes a load's text".
- overview.md, "Directions": "What the owner has left to choose: a clause narrowed to facing, which is prompt text" dropped: superseded by NO_SCENE (2026-10-15, no place-and-action sentence). "Correcting one shot by hand is built ... (Said until 2026-10-05 that this needs a frontend widget)" and "(Said until 2026-10-06 that the nodes were in no shipped graph)" and Known limits' "(This passage said until 2026-10-05 ... and until 2026-10-07 ...)": removed.
- overview.md, "Where to look": "what the owner said of a render" row pointing at two 2026-10-04 records dropped (verdicts are data/<video>/verdicts); "what was measured on which clip" row folded into limits.
- pipeline.md: the owner's quoted sentence in the header removed (writing rule). The "last run" column (fun's dates as of 2026-10-16) removed: video state belongs in data/fun/AGENTS.md. Corrected: `bench/sidecar.py` and `bench/stack_delivery.py` were "uncommitted on 2026-10-16"; both are tracked now (bench/INDEX.md lists them). Header now says it has one job, the data flow; overview.md carries how a job runs.
- sam31.md: header "last updated" list; "(Until 2026-10-18 this line still said No shipped graph wires it yet ...)": removed. Added an "In short" table at the top. Link to "docs/masked_v2v/overview.md" for the pick/corrections repointed to pieces.md and who_is_who.md.
- who_is_who.md: the design's section 6 decision arithmetic (the built total, the evidence total, the impostor level, the take tests, the stop-or-guess switch, the ask row's ordering and its four dated corrections), section 8's two stages, section 9's cost model and counts, section 10's three-way test and its nine falsifying results, section 12's build order, "The limits this design proposes" table (TOGETHER_LEAD, impostor level, ASK_SUGGEST_LEAD, SIGN_CELLS_LEAST, CAST_LOOKS_COMPARED=4, DECISION_LOOKS_LEAST=3, DECISION_LOOKS_AGREE=0.8, NAMES_BOXES_MET=0.3, THING_NEAR=0.5, CAST_SHOTS_TO_CALIBRATE=3, CAST_SEED_LOOKS=4, ASKS_BEFORE_RESEED=12) and the cast's file layout table (log.jsonl, state.json, lock, sheets/, analysis/): removed. These were a design for parts built and removed on 2026-10-10 (decisions.md, "What the owner cut from the tools around the nodes") or never built; the cast file exists in its least form (`bench/cast_file.py`, `cast.jsonl`). The design's direction (appearances, scenes, one confirmation a scene, the casting pass, things) is kept in "The design further ahead". If the guessed constants are wanted again, they are in the parent commit's who_is_who.md.
- who_is_who.md: premortems 1-19 written against that design, with the 2026-10-10 notes of which ones showed on the three-subject run: cut to one table of failures still open. Dropped as having nothing to act through: 7 (the lead's floor set from one clip), 8 (soft film and sharp video, kept as a line on the second clip), 10 (scenes grouped wrong), 11 (a vote that lies given weight), 16 (cast stale against a re-cut source, kept as a cast rule), 18 (the guess switch left on).
- who_is_who.md, section 1 ("all subjects at once held", "two views summed lead by more than either alone"): removed with the totals that night (they measured the tracker's own likeness arranged again); the record keeps them.
- who_is_who.md section 4: "A least form of this is built ... nothing calls it yet": superseded; the cast file is called by `page_answers_to_cast.py` and its seeds feed previews (overview's "Who is who" section, 2026-10-13). "nothing calls it yet" dropped.
- who_is_who.md section 5: "only the first tracker's looks show everybody on the frame" superseded the same night (a tracker now lists everybody and marks the held, TABLE_VERSION 2); kept only the current statement.
- who_is_who_evidence.md section 2: "Vote one: the likeness, as built. Every way of handing a shot's people to the names is totalled ...": superseded by the same page's note (vote one is the tracker's own take). Section 5's two 2026-10-10 paragraphs on the evidence total and its correction: cut to one sentence (it failed in the opposite direction, preferring a clear bystander to a small subject). Section 10's enforcement table and its two update notes: folded into "What is built" and "Failures still open"; rows "a weak row suggests nobody", "nothing is taken without a person (`--unsure take`)" are now held by the tools taking nothing. "My reading of one sentence of the owner's" caveat on who records it: dropped (the condition stands as stated).
- who_is_who_evidence.md: "the Subject Track would need its per-shot seed (built by the tracker's owner, in its sweep when this was written)": the seed is built (`seeds`); kept the remaining need (carrying a foreign frame in front of a shot).
- state_signals.md: "How this page is kept" (dated lines, never rewrite) and "Who holds what" (session ownership of 2026-10-10): removed as history. "Where it is going" (columns become a mark or a sentence by rule) superseded by NO_SCENE 2026-10-15: the rule-written sentence (head-angle bins: <25 faces the camera, 25-65 three-quarters, 65-115 profile, >115 from behind, mrwolf's reasoned edges; the queued render `cont_ctx90_facing_m768_kvrows`) is removed; the bins are recorded here.
- state_signals.md, "Who is in front of whom": "not measured on any clip here", "do not download on a guess", "the owner is downloading the pointmap model": superseded by MoGe-2 depth on overlap frames (2026-10-18 record). Pointmap reading kept.
- state_signals.md, "How a signal reaches the model": "a wired video has no zoom" (superseded later the same day by f7e5de99), "the marked style ... in mrfrog's tree, uncommitted" (committed 86febba3), "the drawing is a one-off script ... in no node", "no tracked record yet ... committed within the hour", and the mrwolf arms ranking for the kitchen continuation ("say nothing new", then corrected): folded into current bullets.
- state_signals.md, "Whether the face shows": "Not built" (superseded the same day: grade_holds is the rule) removed; "Head lift ... not a column of the capture yet" and "A hand at the face: not built" superseded by pose_state.
- window_context.md, "The held tail": the dated stack (NOT BUILT scope; BUILT, not committed; read from code that a shorter window is possible; "possible and outside what the model is documented to make"; the open tail as default; "the plate tail and the input are removed"; the 2026-10-13 shortest-window build): folded into the current answer (held_mask always; plate tail removed 2026-10-10) plus evidence bullets. `held_tail` input no longer exists.
- window_context.md, "Results, dated": "THE QUESTION THE NEXT TWO ARMS ANSWER" and "0.5 is on the card" removed (answered by the next entries). The 2026-10-15 ruling moved to the end so the list reads in date order. Levers table: "on the card 2026-10-10" statuses changed to "tried 2026-10-10"; context_noise status now states the results.
- window_edges.md: "A direction with nothing built" and "until then every item is a want" (items 1, 2, 5 are built); the narrative of the first run's seven findings condensed into the table of what the tool does now (each still held by a made-sound control in check_clip_signals.py).
- known_before_a_render.md: "How this page is kept" (dated lines, rows never rewritten) removed; the 2026-10-10 dated lines folded into the table's last column and a "Found later the same day" table. Rows corrected: the fading short load's remedy was "being built ... the Masked Source's held_tail"; held_tail was removed and the tail is always open. "The flag fired and the lead overrode it" was corrected the same day (no gate, no override then). "A sentence written from one shot drawn in another ... not built": the job file now records a text's frames and refuses one text across shots. The masking board card names (guide-order-of-operations-masked-job, build-render-a-small-subject-zoomed-in, route-second-person-as-face-and-hair-only, build-someone-looked-at-every-frame-before-a-render) dropped with the board.
- perception_models.md: header "last updated" list and the 2026-10-11 italic notes in the short answers, the store table, "Where each was read", "Sapiens2" and "Not run" folded into current statements. Corrected: "Nothing in this pack reads depth today" and "Not run: any depth or pointmap model on a frame of ours" are superseded by `bench/measure_moge2_depth.py` (2026-10-18 record). "A measure, never an input" corrected to the 2026-10-11 ruling (drawn layers may be shown).

### Left for the lead

- Repoint every inbound quote listed in Moves. Code comments: data/AGENTS.md:28 and :38 (overview.md -> working_rules.md and text.md), bench/load_text.py:209 (-> text.md), bench/two_shot_load.py:54 (-> pieces.md), bench/shot_to_load.py:3585 (-> pieces.md, "A small subject, rendered zoomed in"), bench/examples/masked_job_lid_marks/README.md:47 (-> running_a_job.md) and :167 (-> text.md). Docs: docs/state/next_steps.md:1142 (stays overview "What it is"), :1175, :1183, :1207, :1220 (-> pieces.md); docs/state/decisions.md:621, :1077, :1163, :1192, :1236, :1253, :1258, :1632, :2771, :2779, :2786, :3074, :3083 (each per Moves); docs/research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md:22, :51 (-> running_a_job.md). The data/AGENTS.md "Start here" row "the pipeline end to end, and the lane's rules: pipeline.md, overview.md" could name working_rules.md for the rules.
- Every inbound link to docs/masked_v2v/who_is_who_evidence.md -> who_is_who.md (one outside this folder per the grep).
- docs/state/decisions.md should get entries for the claims above that were superseded only here (not yet logged there): known_before_a_render.md's held_tail remedy, pipeline.md's "uncommitted" sidecar/stack_delivery, perception_models.md's "nothing reads depth today", state_signals.md's head-angle sentence rule, and the who-is-who design constants removed.
- docs/working/runbook.md section (b) may duplicate running_a_job.md's chain summary; I left the runbook as the commands' home and linked it.
- Unsure and kept: `bench/load_chain.py` not refusing `no-look` on a `two_shots` job (stated by a reviewer 2026-10-13; not re-read in code); the hair band's current status for subjects other than the second (data/AGENTS.md retires it for the second subject only); perception_models.md is still 562 lines (the core-against-Meta section and the SAM-Audio reading are current reference; a split into a short page plus perception_models_reading.md is a candidate).
- sam31_core.md still contains the original "---" separators and long tables verbatim (file:line citations kept as written).
