# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`roadmap.md`](roadmap.md), the roadmap is right and this
page is stale.

**Starting a day: read [`handover.md`](handover.md) first.** It is what the
last working day left, in the order its lead would take it up, and it opens
with what not to assume.

## To-do (the owner's action list, 2026-09-05)

**Pointers only, one line each, consolidated across the sessions working this
tree.** Each names the record that closes it. The reasoning lives where the
pointer goes; this list is stale the moment the roadmap disagrees with it.
Items from the `mrpink` helper lane first; `evalman`'s render lane appends its
own below the rule.

**The order of 2026-10-19 is `handover.md`'s "Tomorrow, in order" (written 2026-10-18 22:05Z); where the queue
below disagrees, the handover is right.** First: the drawing build's tests read by the owner and its defaults put to
them (the grey head's marks clipped to it, a little darker, the mouth decided per window by voice); then the second
half, worst first, every old shot from a fresh first preview.

**Below the order above: the open items, one line each, with where the reasoning lives.** Nothing here is in order
unless it says so; the handover and the Open List set the day's order. A video's own state (who is in each shot, what
is laid, the gaps, the owner's verdicts) is not here: it is `data/AGENTS.md`, "Where the lane's state lives" (for
`fun`, `data/fun/AGENTS.md`). What was done, decided or retired is in [`decisions.md`](decisions.md), not here.

## Masked video to video: open questions

- **Gaze does not carry.** The render does not follow a sideways glance it is shown (lead_fp_1097b and the zoomed
  1097c, 1119-1133), and the pose model misses a vertical eye roll (1156b at 1158-1160). The owner's allowed tests
  (2026-10-15), one render each on 1097-1133, seed held: (1) `--room 0.3` or a head-following box, no code; then (2)
  the mesh's own eye sockets painted out and the iris pushed into the leading corner. Not to do: a glance in the text, a
  lower still weight, a seeded eye stamp. Why and the figures: [`decisions.md`](decisions.md), 2026-10-14 and
  2026-10-15 (the corrected zoom entry); the outside reviewer's C9 (`internal/gemini/2026-10-15_gaze_ideas_c9.md`) and
  section 11 (`internal/gemini/2026-10-14_relay_gaze_hair_expressions.md`, lines 263-293); the iris offsets
  (`internal/claude/2026-10-14_mrdrill_scripts/iris_offset_render_iris*_1097_1133.csv`,
  `iris_offset_render_zoomed_1097c_1097_1133.csv`); the eye roll sheet
  (`data/2026-10-14_fun_batch_r_inputs/lead_fp_1156b.eyes_1158_1160_1170.jpg`).
- **The pose model's lids miss closing eyes**: lowered eyes drawn open (second_hfp_0278c 290-293, 0415d 437), a laughing
  squint drawn as an open almond (3247, 3261, 3276), shut laughing eyes drawn open on all of 3167-3213. Upstream of the
  drawing; a row, not a job. The figure wanted: `bench/_lib/face_points.py::eye_opening` against the original's, a
  frame at a time. Records: `data/2026-10-14_second_hfp_0278c/eyes/lid_gap_render_vs_original.txt`,
  `data/2026-10-14_second_hfp_0415d/eyes/lid_gap_render_vs_original.txt`,
  `data/2026-10-14_lead_fp_3167/look_blind/reading.md`, `internal/claude/2026-10-14_mrtwo_scripts/eyes/findings.md`.
- **Where the body model's arm crosses its head, the face points are drawn on the arm** (lead_fp_1097c 1097-1104, the
  outside reviewer at one frame a second, not yet seen by a session). Not counted; wanted first a look, then a way to
  find the body model's arm over its head (`internal/gemini/2026-10-15_clips_c2.md`, section 1).
- **A zoomed load has one box**, so a small face that travels across the frame cannot be zoomed (the lead's 168-184);
  the node refuses and renders the whole frame. Wanted: a box that follows the subject; first read on a plan what a
  stepping box does to the reference and the lay-back
  (`data/2026-10-13_lead_fp_0168_job/lead_fp_0168.plan.queued.json`).
- **Two shots in one window** is built (`bench/two_shot_load.py`) and has never rendered. Next: one pair made both
  ways, the cut on a step's edge, for the owner's eye; then a tough shot beside an easy one
  ([`../masked_v2v/running_a_job.md`](../masked_v2v/running_a_job.md), "Two neighbouring shots in one load").
- **Whole-frame video to video (2026-10-09)**: the swimmers clip by text with a men's suit, written and not rendered
  (`internal/claude/2026-10-09_mrship/`, `prompt_swim3_*.txt`, `swim3_copy384_*_api.json`); open with the owner whether the rule the text pairs point at goes into
  `docs/prompting/prompting.md`, and a repair for the one held frame
  ([`../../bench/results/2026-10-09_whole_frame_text_and_copy_size.md`](../../bench/results/2026-10-09_whole_frame_text_and_copy_size.md)).
- **The frozen video cache on a masked window** lives in one probe graph (`h3_config.FROZEN_VIDEO_CACHE`). Owed, in
  order: the owner's eye on the upper-body pair; a several-window clip as a pair; a cached step against a stock one on
  the right kernel; memory during a cached step; then whether a masked graph carries it by default, the owner's call
  ([`../../bench/results/2026-10-07_frozen_cache_masked_upper_window.md`](../../bench/results/2026-10-07_frozen_cache_masked_upper_window.md)).
- **Lower, from the first clips (2026-10-04 to 10-07)**: the original subject's shadow stays in the plate; the source is
  resampled twice on its way to the canvas, and what one resize does to SAM's masks is unmeasured; the still shared
  with the text encoder at full size on the masked graphs; the turn with no prompt text is the owner's choice (the
  generic clause narrowed to facing, or different clothes accepted;
  [`../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`](../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md)).
- **Parked or not being built, by the owner**: two samplers with the plate restored between them (`plate_restore.py`,
  in no graph); protecting the neighbours' pixels, `paint_out` as a default, a ControlNet, the baseline chain for this
  lane.

## Masked video to video: gaps in the tools

Each found by a run, none built. The dated entry in [`decisions.md`](decisions.md) has the finding.

- `bench/capture_masked_run.py::planned_region` models the node's old region rule: it lacks `drop_specks`,
  `close_lone_holes` and `drop_islands`, so its `--plan` figures differ from the node's saved region (no gate reads it;
  the saved region is the truth). Wanted: it calls the node's own `window()` (mrlol's cold reader on 561f385a).
- Nothing prints whether a body model is on its subject: the body model's head against the subject's own part or
  track, a column a frame in the pose rows ([`../../bench/results/2026-10-12_body_model_posed_the_other_person.md`](../../bench/results/2026-10-12_body_model_posed_the_other_person.md)).
- `bench/capture_masked_run.py files --overwrite` leaves a subject's files from the build before when the new build
  gives it fewer inputs, and the gate reads them. Until fixed, build a changed capture in a new folder.
- `bench/recomposite_window.py` writes tables, not pictures. Wanted: a laying as a video, several side by side over a
  box, source frame numbers burned in.
- The review page's server and page check are untracked (`data/<video>/poc/serve.py`, `check_page.py`; how to start
  and stop them is [`../working/runbook.md`](../working/runbook.md)). Wanted: tracked tools with a check each, the
  page's folder handed in by path.
- A pair measure (sharpness in the region, mouth opening against the source, the step at a piece's first frame) is two
  session scripts (`internal/claude/2026-10-13_mrtower_scripts/measure_pair_1920c_1920d.py`). Wanted: one tracked tool over two landed loads and a span, figures only, with a check.
- `bench/page_answers_to_cast.py` takes a line as the owner's because its `who` says so; the server does not know who
  clicked. Wanted: the server stamps where a click came from.
- The check that a render's graph is its plan's graph is a session script handed between card holders. Wanted: a
  tracked tool (or the builder's `queue`) that compares per-stage file inputs by content.
- The plan strip says a load renders zoomed and on which box (`bench/capture_masked_run.py::render_said`) but does not draw
  the box.
- A flag's frame count and its burned span can differ by the frames between its runs (`frame_spans`' `join`); changing
  it changes every standing override's key. The small way: the label says "N of these frames".
- The gaps print in `bench/delivery_table.py` and `bench/frame_coverage.py`, not yet in `bench/shot_parts.py` or
  `bench/shot_to_load.py job`; and the reason a load left frames (grade, floor, keep-out) is in the manifest, not
  printed.
- After a record's cut is mended, `page_answers_to_cast.py --shots` refuses the whole run on one answer whose shot is
  gone. Wanted: such answers counted and listed with the shot that now holds the frame, the run going on.
- `bench/break_checks.py` cannot run `check_masked_job`: its copy has no `.git`, and that check asks for HEAD.
- The voice table's level is the two channels' mean (`bench/voice_spans.py`), which reads low where they differ. Not
  changed on the lead's word, since every text leans on `voiced`; wanted first, a read of which frames change sides.
- Of several boxes in one cut, the seed is the one clicked last (`bench/cast_file.py::seeds`). Figures to print beside
  the boxes, none a rule: area, the frame's brightness, distance from the cut's ends, overlap with the node's detection.
- A mended cut costs a preview of each shot it changes (`shot_to_load.py job` refuses a preview of other frames).
- A who picture does not say how much of its box a person's mask fills (`bench/who_page.py`).
- A picks batch holds one line per record shot however many tracker cuts the owner answered on
  (`bench/picks_against_answers.py::lines_of`).
- A shot longer than `loop_plan.CHAIN_LENGTHS`' last and no longer than `h3_rules.MAX_LENGTH` is cut into two windows.
  No shot of `fun.mp4` is in that case.
- The builder's window choice does not read the cuts (the owner's rule of 2026-10-12: boundaries on a near cut, no
  window starting within about 24 frames of one).
- A capture's times are local with no zone, and `bench/masked_job.py::_when` refuses a zone. Wanted, in one change on a
  quiet card: offsets written, both forms read, a word on captures already on disk.
- The text grader has no donor for a masked load: `bench/grade_prompt_text.py --mode ref2va` fails every masked text on
  `<Video 1>`, and snaps the length. Until a donor or a sockets-from-the-load mode exists, its FAIL on a masked text
  says nothing about the text.
- A field in the job file for why a load took the plain cut, so `bench/delivery_table.py` need not take it by `--prefer`.
- The shots record's `frames_on` and `frames_boxed`: 0-1919 needs a mode that adds them from the cast file; and a marked
  frame is not a pick (about half the typed picks on 1920-6394 took the wrong person). How the owner gives one box a
  shot at the least cost is open (`data/fun/shots.json`).
- The owner's typed words on 1920-6394 are in no shot's note: the dry run of
  `page_answers_to_cast.py --words-to-notes` is the lead's to read first (read on 10-12 into
  `data/fun/who_1920_words_read.md`).
- The layered view of the preprocessing: shared drawing helpers into `bench/_lib` first, then the view in an existing
  tool (the Open List's row t76; the page shape in `internal/claude/2026-10-12_mrfern_handoff.md`, last section).
- The capture's full path for each thing a node uses, source to target, every change with the node and values, per
  window, and a capture that says up front whether the source's shape is the canvas's (the owner, 2026-10-11). The
  handover's facts pass (item 1c) is its first part.
- The four reds of the green-and-red census are cut (`check_correctness`'s bar, `check_sol_chunked`'s cosine and peak,
  `check_channel_balance`'s fragile selection, `draw_lid_marks`' fit verdict), each its own commit
  ([`../../bench/results/2026-10-12_green_red_census.md`](../../bench/results/2026-10-12_green_red_census.md); the
  owner's row z08). Queued, not started.

## Attention, distills and the base model

- **Sol node redesign, test 3**: the token-routing re-grade on the fixed kernel, on the test 2 captures (kept until
  2026-10-31), no render ([`../research/2026-09-27_sol_node_redesign.md`](../research/2026-09-27_sol_node_redesign.md)).
- **Approved and queued 2026-09-27**: [#40](open_experiments.md) (a second seed on the finalists), #43's scene pass,
  #44 (timestamps inside a shot). Then #41's blind pairs.
- **The owner's calls**: whether `h3_text_to_video_pdd8_flashgen_finish` replaces the default t2v PDD graph
  ([`../../bench/results/2026-09-27_finisher_grid.md`](../../bench/results/2026-09-27_finisher_grid.md)); #45's
  reference node, or close #45; #36's blind batch `fasth3_gate_dial_look_anchor`; the per-head tau table, which needs
  a guard margin and a re-fit on the 512 layout before any use
  ([`../../bench/results/2026-10-03_per_head_ref_kill.md`](../../bench/results/2026-10-03_per_head_ref_kill.md)).
- **O1's splotchy blacks** are not the codec's 8 px blocking; the test is a lossless decode of a clip the owner names
  ([`../../bench/results/2026-09-27_o1_lossless.md`](../../bench/results/2026-09-27_o1_lossless.md)).
- **ref2va**: block 49 graded on video and audio query rows on the real kernels over the ref2va and t2v test 2 captures
  (`bench/grade_dense_kernels_on_captures.py` records `rows_video_audio`; needs the card); regenerate
  [`data/results/2026-09-29_partition_delta_map.jsonl`](../../data/results/2026-09-29_partition_delta_map.jsonl), which predates its script (do not read `cos` from it); issue
  16604 is untested on CUDA or ROCm
  ([`../../bench/results/2026-09-29_ref_rows_16604.md`](../../bench/results/2026-09-29_ref_rows_16604.md)).
- **Sparse attention, from 2026-10-03**: every attention timing and routed share of that day is on the 2048-view
  layout, none re-taken on the 512 default; the "Not done" items of CHANGELOG 0.185.20.
- **Frozen-video cache on the refine pass**: fewer refine steps (`h3_config.AUDIO_REFINE` is inherited and untuned; a
  listen); a same-graph repeat after a cache clear for the noise floor; a VRAM store or K/V contents for shorter cached
  steps ([`../../bench/results/2026-09-25_frozen_cache_s1.md`](../../bench/results/2026-09-25_frozen_cache_s1.md)).
- **Two cheap probes from 2026-09-25**: do LoRA'd layers stay on the int8 path at run time
  ([`../research/2026-09-25_temporal_offset_and_adaln_rounding.md`](../research/2026-09-25_temporal_offset_and_adaln_rounding.md),
  last section); and the refine arms' video about 46 dB from their base arms, decode nondeterminism or the masked
  blend, by a same-seed repeat after a cache clear.
- **PDD, the owner's decisions**: whether `MiniMaxH3PDDLoRA` keeps its own head swap
  ([`../research/pdd/2026-09-25_upstream_pdd_comparison.md`](../research/pdd/2026-09-25_upstream_pdd_comparison.md));
  reporting sglang's fc1 half-swap and bf16 plan, and UtilsCollection's `final_layer` patch, upstream
  ([`../../bench/results/2026-09-25_sglang_pdd_fc1_merge.md`](../../bench/results/2026-09-25_sglang_pdd_fc1_merge.md)).
  PDD quality work is parked ([`roadmap.md`](roadmap.md), "Closed lanes").
- **Ideas saved for later (2026-09-26)**, each the owner's: route a distill by what the shot does, or switch distills
  by step ([`../research/2026-09-26_distill_routing.md`](../research/2026-09-26_distill_routing.md),
  [`../distills/which_distill.md`](../distills/which_distill.md)); fit the schedule to how much of the frame changes
  (its premise is [#43](open_experiments.md)). PDD first with a FlashGen finish, and the reverse switch, were built
  and rendered on 2026-09-27.
- **The PDD bake's leftovers**: the bake script's review candidates (the session's internal notes of 2026-09-05 and the
  postmortem under `internal/postmortems/`: a control-before-bake gate, the contract check's
  retyped filenames, the dirty-tree marker, duplicated helpers; closes when each is fixed or dismissed with a reason in `CHANGELOG.md`); the stripped sidecar into the HF staging tree (its
  publication is the owner's call); the ref2va bake after the fl2va pair is judged
  ([`../research/pdd/2026-09-05_bake_plan.md`](../research/pdd/2026-09-05_bake_plan.md)).
- **Upstream to watch**: if core PR 16713 merges, re-run `bench/probe_ref_rows_16604.py`; if 16508 merges, the Sol
  node's bf16 refusal is what users see and `--bf16-unet` its fix; core's own reference-audio nodes crop until
  Comfy-Org/ComfyUI#15972 lands (ours end-pads); two optional decoders (LynnReal's light H3 VAE, vllm-omni's latent
  upscaler) go through `bench/compare_vae_decoders.py` before any default moves, the owner's call whether to look.
- **The M3-to-H3 bridge**: the image lane is next; the fit was stopped unfinished
  ([`../research/m3_h3_bridge.md`](../research/m3_h3_bridge.md);
  `internal/claude/2026-09-14_m3-h3-bridge_independent-take.md`, the fixed script
  `internal/claude/m3-h3-adapter/fit_image_lane.py`, which needs a GPU window agreed with the card holder).
- **The frozen-audio loop**: owed the full-length example render (a throwaway first run), resume on the reference song
  graph, one queue from the editor for the workflow chunk; song graphs saved from the editor need remaking. The
  untold_loose second seed (2203; `data/results/2026-09-13_audio_freeze_untold_seed2.jsonl`) awaits the owner's line in the step 2 verdict file
  ([`../h3/audio_freeze.md`](../h3/audio_freeze.md)).
- **The owner reported two clips bad on 2026-09-17**, open
  ([`../../bench/results/2026-09-17_owner_reported_bad_clips.md`](../../bench/results/2026-09-17_owner_reported_bad_clips.md)).
- **Prompting**: the open items of the 2026-10-01 audit
  ([`../../bench/results/2026-10-01_prompting_docs_audit.md`](../../bench/results/2026-10-01_prompting_docs_audit.md));
  `bench/grade_prompt_text.py` passes a ref2va prompt whose unmarked vocal events lack `(Sx)` reuse (it would have
  caught three prompts).

## Checks and hygiene

- `check_vsa_core_patch.py`'s logic fix, so that "nothing patches core" has a working check again.
- The always-red checks: `check_doc_links` on `docs/h3/audio_freeze.md`'s sister-repo citations, and
  `check_audio_freeze`'s ImportError (it imports `audio_freeze.py` top-level, whose relative import fails). The Sol
  kernel checks are owed a run with the GPU visible.
- `check_schema_defaults.py` and `token_routing` (`'off'` in the schema, `None` in `execute`): align them or exempt the
  input with its reason, the owner's decision.
- `check_node_ids.py`: whether it should tolerate a required-then-optional reorder.
- Two stale Sol readers: `check_provenance_stamp.py` red on `no_missing_knobs` (`token_aug_profile` not recorded by
  `provenance.py`); `analyze_sol_error.py`'s CUDA arm passes `centroid_tail=`. Run each to see.
- `bench/check_dit_prefix_attention.py` cannot run with CUDA masked (its import chain asks for a device); run it with
  the card free, or with ComfyUI's `--cpu`.
- Run rows carry the working tree's commit when written, not the serving process's (`bench/run_graph_arms.py`); until
  rows carry the server's start or its commit, a verdict write-up states them by hand.
- The served sage build is not in `run_graph_arms.py`'s `substrate`, and no check asserts it equals the tagged one.
- The queue waiter matches by substring (it has bitten twice); the dataset builder should raise on an output claimed
  by two renders; the scoring page's pair tab reuses the singles' tags ("same" and "can't tell" on one half).
- The sage chain assert's call-time proof on the no-sage graph skips for headroom; it closes when a just-Sol render's
  log carries the "no sage: probes ... reached no sage kernel" line. The just-Sol frontier row for the tau-raised arm
  has no loudness entry.
- `bench/check_sol_kernel.py` slimmed to its build printout, `observable` and `schema` (agreed with the owner); in the
  same stretch, the comments in code that cite the removed Sol reference file.
- Not examined: `workflows/h3_probe_sol_on_refs_api.json` against the shipped graph it mirrors (retire it or say why it
  stays); the editor side of the API-only workflows (Export (API) diffing clean against four shipped files).
- The 2026-09-03 base-16 capture is kept to 2026-10-31, then due for `bench/recycle_captures.py`, the owner's call.

## Do not rely on

- A render compared across 0.166.0 pins `quantizer=balanced` or re-renders its baseline.
- A same-seed clip from before the 2026-09-15 core pull (the text encoder's RoPE on kitchen) or the 2026-09-22 pull
  (VAE tile blending) is not bit-comparable with one after it.

## Then, in the roadmap's order

The forward plan in [`roadmap.md`](roadmap.md) ("The attention plan, where it stands") owns the sequence, the
decision standard and what would count as finding it. The masked lane's order is the handover's.

## Where the reasoning lives

- What was measured and what was withdrawn:
  [`docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md`](../research/2026-09-03_sol_exact_pquant_and_base_capture.md)
  and [`docs/method/evidence.md`](../method/evidence.md).
- The goal in the owner's words: [`../../VISION.md`](../../VISION.md).
- How a rendered comparison is judged: [`docs/method/eval_comparison.md`](../method/eval_comparison.md).
- What is not measured: [`open_experiments.md`](open_experiments.md). The graphs the owner renders with day to day:
  `h3_config.DAILY_GRAPHS`.

## Updating this page

When an item is done, remove it here (and in the roadmap, in the same commit) and log it in `decisions.md`; a done
item left standing is the drift this repo names most often.
