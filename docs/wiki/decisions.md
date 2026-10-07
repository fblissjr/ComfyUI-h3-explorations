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

## 2026-10-07

- **The Subject Track picks the most central person by default, not the
  largest** (the owner, on a measured difference). On the three windows
  tested, two stretches of the lane's crowd clip and one of a second clip,
  the two rules name different detections; `largest` takes a figure low in
  the frame on the crowd clip and one whose box touches two edges of the
  frame on the second, and `most central` the person meant. With
  `most central`, followed alone, the subject keeps a mask on every frame
  of the stretch that had been called hard, each frame's mask overlapping
  the one before: continuity, which no label confirms. **The evidence is
  not yet in a tracked record**; the tracker lane's record of the central
  figure is owed. *Corrected the same day on a fresh session's reading:
  this entry, the 0.220.0 changelog entry and the comment beside the
  default said "a figure at the frame's edge" on "the lane's crowd clip"
  for all three windows and cited
  `bench/results/2026-10-07_subject_track_under_nudge.md`, every arm of
  which was picked with `largest`.* Changed
  together: the node's default (`subject_track.py`),
  `h3_config.SUBJECT_TRACK` and every generated masked graph. It was
  `largest` since the node was written; an arm recorded before this date
  was picked with `largest`, and reproducing one needs `pick` set to that.
- **Reversed by the owner: the frozen-row cache's masked use is back in the
  tree.** Retired on 2026-10-06 for saving no time; the next morning's
  kernel record found the cached block's attention had run on torch's
  kernel, the same pair with that corrected saved what the design predicted
  (`bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`), and the
  owner, having watched the pair, restored the module, its check, the
  generator's `masked_cache` and the probe graph as they stood before the
  retirement. One change on top: the cached block no longer removes the
  attention override. `docs/wiki/masked_v2v.md`, `docs/wiki/next_steps.md`
  and `archive/frozen_cache_masked/README.md` said "retired" and now say
  this. **Also corrected:** the fixed-pair record said its change walked the
  override chain to the backend under Sol-Attn. On Sol's real override it
  walked nowhere and left the override in place, so what was measured is Sol
  declining the call to the backend below it; the record carries a dated
  note and the tree's code says what it does.
- **Corrected: the upper-body recipe had rendered on the prompt node's
  text.** Written into three places with the new graph (0.215.0): that the
  node's wording for that region "has not been watched" and had rendered
  only "on another clip" (`docs/wiki/masked_v2v.md`,
  `prompt_bank/bank.json`, `docs/prompt_audit.md`), taken from the arms
  record, whose rows predate the arm. It rendered on 2026-10-06 on the same
  window and seed, beside the typed arm; the stacked pair is on the output
  share and the record carries a dated note. The owner then watched the
  pair: both good, not told apart, which confirms the node's wording for
  the graph. Found when the owner asked on the board for samples.

- **Corrected: a typed correction has run on a real clip.**
  `docs/wiki/masked_v2v.md`, "Known limits", said "not yet run on a clip",
  and the masking board's card said the same and held a demonstration for
  the owner's word. It ran on 2026-10-06 on the car window
  (`bench/results/2026-10-06_subject_track_defaults.md`, "The correction").
  Found when the owner asked on the board what became of the card. What is
  still not built is a click or a box for a person the detector never
  found.

- **Corrected: the frozen-row cache's first masked run did not run its
  attention on the kitchen kernel** (`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`).
  What the prose used to say: the 2026-10-06 record, "The call goes to the
  kitchen backend's int8 attention, the kernel the dense stock steps use",
  and from it the archive's README and `next_steps.md`, that the miss is
  the kernel's cost on a rectangle. The kernel timed alone says the
  rectangle costs the square's rate; the cached block's call falls to
  torch's kernel because it drops the override that carries the backend.
  No decision is changed by this entry: the owner retired the masked use on
  that record, and reopening it is theirs. The same call is in the live
  module, so the audio-refine pass's cached steps run torch's kernel too;
  it is left as it is, since changing it changes that pass's numbers.

## 2026-10-06

- **The frozen-row cache's masked use is retired; the node stays for the
  audio-refine pass** (owner, 2026-10-06, relayed by the lane's lead
  session, on the first masked run:
  `bench/results/2026-10-06_frozen_cache_masked_window.md`). On a Sol-Attn
  graph the cache saves no sampling time at that window's live share and
  departs from the stock step on every cached step; on the refine pass it
  is measured to pay (`bench/results/2026-09-25_frozen_cache_s1.md`). The
  owner's terms: cite why, and keep the masked code where it is plainly
  used by nothing, which is `archive/`. **Not done at that commit**, done the
  same evening (the retirement commit after 25103f0e): the masked gate,
  `masked_cache` in the generator and the probe graph left the tree for
  `archive/frozen_cache_masked/`. What was tried is one
  window, one seed, no halo and no refresh, and the record claims no more
  than that. Three pieces of prose are corrected with it. What they used to
  say: `docs/wiki/next_steps.md`
  listed the run as owed and its saving as modelled;
  `frozen_video_cache.LIVE_SHARE_LIMIT`'s comment said the limit "sits under
  break-even" and that "the first masked run's step times replace the
  model"; the generator's note on the probe graph said no graph carries the
  cache "until a masked window has been timed and looked at with it".

- **Prose corrected: the frozen video cache had run on the card, and is in
  one masked graph.** The comment on `h3_config.FROZEN_VIDEO_CACHE` ended
  "Not yet run on the card"; the refine pass ran through it on the card on
  2026-09-25 and again on 2026-10-06
  (`bench/results/2026-09-25_frozen_cache_s1.md`,
  `bench/results/2026-10-06_frozen_cache_stage_split.md`), and the comment
  now points at the later record. `docs/wiki/next_steps.md` said of the
  cache "It is in no masked graph"; the generator's `masked_cache` argument
  now writes one probe graph with it
  (`h3_probe_v2v_masked_song_ref2va_motion_cache_api.json`; retired the
  same day to `archive/frozen_cache_masked/`), and no shipped or daily graph
  carries it. No masked window has run through it.

- **Prose corrected: the frozen-row cache takes a partly masked video.**
  `docs/wiki/next_steps.md` listed "a frozen-row cache for a partly masked
  video (`frozen_video_cache.py::_gate` takes a wholly frozen one only)" as
  owed; the gate now takes any call that freezes some row and regenerates
  another, and the entry says what is owed instead (an arm, a matched pair,
  the owner's eye). `docs/checks.md` said of `check_frozen_video_cache.py`
  "text rows live must beat text cached (`LIVE_KINDS` monkeypatched)"; that
  ordering held by the seed on the check's model and is no longer asserted,
  and the constant is `ALWAYS_LIVE_KINDS`. The row says what the check holds
  now.

- **The masked graphs' prompt is written by a node, and the fl2va PDD8 graph
  ships the generic text** (owner, 2026-10-06: generalise the lane beyond one
  clip, and a prompt that covers different scenarios without the long
  reference-format text being retyped). `MiniMaxH3MaskedPrompt`
  (`masked_prompt.py`) writes it from who the still shows, two choices and
  the Masked Source wired into it; the sentences are constants in
  `masked_prompt_text.py`. What it replaced: both shipped masked graphs held
  a typed text in the song node. The PDD8 graph's was
  `ref2va_masked_subject_swap`, which says one performer, alone, standing
  through a fourteen-second static shot; it now ships
  `ref2va_masked_person_swap`, the text mrhf wrote on 2026-10-04 to claim
  none of that. The motion graph's was `ref2va_masked_subject_motion`, which
  says "the man"; it ships the same text for "the person"
  (`ref2va_masked_person_motion`), a placeholder still being anybody. Against
  what: the generic text for "the man" is what every arm of
  `bench/results/2026-10-05_masked_v2v_motion_arms.md` ran, with and without
  the motion lines; the "person" form was rendered once, one window and one
  seed, without them (mrhf's note beside the texts under `internal/`, which
  is not tracked). The person text with the motion lines has not been
  rendered, and neither have the node's head-and-hair and silent texts.
  *Later the same day:* it rendered once, beside the same graph with "a
  man". The man text reproduced the 2026-10-05 arm exactly, so the node,
  the shipped graph and the day's code change nothing in the output on
  that window and seed; the person text turned him and started the turn
  later. The default stays "a person" and the lane's page says to set
  `subject`: `bench/results/2026-10-06_masked_v2v_person_text.md`.

## 2026-10-05

- **Sol starts at 0.2 on every graph; the 2026-10-01 call that set 0.0 on
  PDD graphs is reversed** (owner, 2026-10-05: "0.0 makes no sense to have
  on. zero dense doesnt make sense"; "no 0.0 default for sol anywhere").
  `h3_config.SOL_PDD_OVERRIDES` is empty again and the `sol_only` candidate's
  0.0 override is gone; every PDD graph rebuilt. Against what: the 2026-10-01
  panel showed only that a text-to-video finish could not be told apart at
  0.0 and 0.2; the ref2va PDD8 arms of 2026-10-05 ran Sol from step zero and
  lost the subject's video reference, and the Sol-off and Sol-at-0.2 arms of
  the same evening are in `bench/results/2026-10-05_masked_v2v_motion_arms.md`.
- **VOID is out, and the code built for it is removed** (owner, 2026-10-05,
  on the board: "Park VOID entirely, lets remove the code we built from it -
  it was built for cogvideox and its ghosting and not worth keeping or
  preserving. Note we tested it and thats it. No current state docs or
  workflows with it please."). Removed: `MiniMaxH3VoidConditioning`, the
  checkpoint converter and their two checks, all last present at commit
  df3bd21b. This reverses the line below from earlier the same day, which
  parked the route and said the node and the converter stay. Kept: the
  record of the test, `bench/results/2026-10-05_void_plate_turn.md`.

- **The masked lane gets a second shipped graph on ref2va for shots that
  need the original's movement** (owner, 2026-10-05, on the masking board:
  "For this you just need my approval? If so yes").
  `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json`: the
  Masked Source's motion reference on, the ref2va base at
  `h3_config.MASKED_MOTION_STEPS`. Against what: eleven arms on the band
  clip's turn (`bench/results/2026-10-05_masked_v2v_motion_arms.md`): fl2va
  ignores an encoder-only video reference of the subject at any step count,
  distilled or not; ref2va carries it at 12 and 16 and loses it at 8 with or
  without its PDD bake. The fl2va PDD8 graph stays the default. The owner
  asked where 12 comes from: the lowest count on a three-point ladder (8,
  12, 16) seen to carry the reference, one seed at 8 and 12.
  Corrected the same evening: the graph's first prompt (the shipped masked
  text with the reference lines grafted on) did not turn him on its own
  render, which mrteal's yaw metric caught; the shipped prompt is now the
  exact text the arms measured, and the threshold ladder gained 10 (no turn).
- **A check may run a coderef file as a one-off numeric reference; nothing
  shipped depends on coderef** (owner, 2026-10-05, on the masking board:
  "yes you can test it just dont write code that relies on it here - just
  one offs"). Raised by mrteal's `bench/check_sam3d_body_vith.py`, which
  runs Meta's backbone file in a child process to compare against. The
  clause is in `AGENTS.md`, "Reference implementations".
- **A clip that starts on a still gets the FlashGen finish: the docs now say
  what the code ships** (owner, 2026-10-05, on the masking board: "Please
  resolve this so it reflects current state code"). `h3_config.DAILY_GRAPHS`
  has carried `h3_i2v_pdd8_flashgen_finish` since 2026-10-03, among "the
  three graphs the owner renders with" (0.185.24). Three passages still said
  the opposite, from the 2026-09-27 finisher review ("looked off",
  ow-fd-09): `docs/h3_distills.md` under "Rules of thumb", two entries
  further down this page, and one line of `next_steps.md` ("i2v stays PDD8
  alone"). Each now carries a dated note; none is deleted, since the review
  did say it.

- **The VOID clean-plate route is parked** (owner, 2026-10-05, on every arm
  rendered that day: "Not a single clip showed any improvement."; their
  reason for stopping, as mryellow relayed it and not in their words: VOID
  is trained on CogVideoX, so it may be a waste of time). Parked, not ruled
  out: a plate under the hole is still wanted, and this tool on this base
  through core's port is set aside. No pass 2, no run of upstream's own
  code, no blended windows. What was found on the way stays:
  `MiniMaxH3VoidConditioning` and `bench/convert_void_checkpoint.py` with
  their checks, and in `bench/results/2026-10-05_void_plate_turn.md` the four
  places core's port differs from upstream's inference, one of which
  (the pixel range the mask is encoded at) reaches core's own template and
  is the owner's to report or not.

- **The generated indexes list tracked files only, and the sweep holds them
  current** (owner, 2026-10-05: "do not index anything that isnt git
  tracked"; 0.190.18). `bench/build_index.py` reads `git ls-files` and
  nothing else. It used to list what was on disk under `bench/`, so an
  untracked script appeared in `bench/INDEX.md`. The changelog's 0.189.8
  entry said nothing gated the generator's `--check`;
  `bench/check_doc_inventory.py` now runs it.
- **A masking note said the VOID passes were only in the upstream repo**
  (`docs/research/masking/2026-10-04_mrhf.md`, "VOID for the plate with
  nobody in it": the Comfy-Org repackage "holds only" the T5 encoder, the VAE
  and RAFT). Wrong, and against the same note's section 5: both passes are
  in the repackage too, and only those load in core, upstream's having
  diffusers key names. mrorange found it reading core's template and the
  file's header; the note carries a dated correction and
  `bench/results/2026-10-05_void_checkpoint_conversion.md` the evidence.

- **The 2026-09-13 blind reference-view session is closed unscored** (owner,
  2026-10-05: "Im not sure how much value it added anyway", after the sealed
  key under `internal/blind_keys/` was found gone from disk). Its question,
  the `qwen_view` default, was settled by eye on 2026-10-03. Nothing else
  depends on the key; a new session seals its own.
- **No history rewrite** (owner, 2026-10-05: "Dont worry about the history
  rewrite, can move on from that"): 563328d9 stays in history and every
  later hash stands.

- **The HF hybrid checkpoint `unet_hybrid_b30` is retired** (owner,
  2026-10-05: "Remove that hybrid model, its long been deprecated", after
  mrteal found its symlink dangling and the download gone). The key leaves
  `h3_config.MODELS`, the loader map and the model-contents baseline; the
  four dangling links (b15, b20, b25, b30) leave `models/diffusion_models/`.
  Our own `unet_hybrid_adaln_all` stays, with its file. No graph used either.

- **`check_no_owner_paths.py` scans the files git would take, not the
  gitignored trees** (mrteal, on mryellow's brief for the owner; 0.189.5).
  It used to walk everything under the repo, `internal/` and `data/`
  included, and was red on raw captures and session notes there with no
  tracked file among the hits. `docs/checks.md` index row has the reasons
  and the two ways to look wider.
- **The `h3-ab-session` skill no longer routes to `internal/blind_keys/`**
  (0.189.5). The directory did not move: `bench/blind_batch.py::KEY_DIR`
  still names it, and the skill now points at that constant. It is absent
  from this box, so the sealed keys of earlier blind sessions are not on
  disk; nothing in the repo records who removed them or when.
- **`check_model_contents.py` used to print "in the baseline but no longer
  named by h3_config" for `unet_hybrid_b30`**, which `h3_config.MODELS`
  names (0.189.5). The file's symlink dangled. It was left open as the
  owner's call and the baseline was not regenerated; the owner retired the
  key the same day (the entry above), and the baseline dropped its
  fingerprint in 0.190.1.
- **`docs/checks.md` "Running them" said exactly two checks need
  `PYTHONPATH`** (`check_clone_v_wiring.py`, `check_correctness.py`; 0.189.4).
  A sweep without the variable found `check_reference_encode.py` needed it
  too, and now none does: each calls `bench/_lib::bootstrap`. Its Gaps item 3
  used to list the split as open. The index rows for the first two used to
  read "CUDA, `PYTHONPATH`"; both returned 0 with "skipping" when masked and
  now exit 2. The `check_reload_invariance.py` row used to say only that it
  renders; it now says when it exits 2 and that a sweep never posts.
- **The at-the-call LoRA node reads Kohya-style keys** (owner, 2026-10-05,
  on the sglang read: "may as well"; 0.189.0). `lora_branch.py::native_keys`
  renames them through core's table. It replaced a refusal that pointed such
  a file at `LoraLoaderModelOnly`. The module docstring's "Formats" paragraph
  used to say every key must sit under `diffusion_model.`.
- **PDMD is retired and removed** (owner, 2026-10-05: "pdmd - lets retire it.
  i didnt see anything from it worth keeping"; 0.188.0). Removed: the four
  graphs, the `PDMD_*` constants, the contract case in
  `check_distill_settings.py` (which now fails a graph loading a PDMD LoRA),
  `bench/measure_pdmd_lora_conversion.py`, `bench/pdmd_vs_flashgen_arms.json`
  and `docs/research/pdmd/`. Retired, not refuted: one blind look. The
  records stay in `bench/results/` (`2026-10-01_*pdmd*`), the code in git, and
  the lane in `docs/roadmap.md` "Closed lanes", whose row used to say the
  graphs "stay buildable on kijai's files" and names what would reopen it no
  longer. `docs/h3_distills.md` used to carry a PDMD row and section; it keeps
  a pointer to the verdict record.
- **The base samples with Euler; `er_sde` is gone from every graph** (owner,
  2026-10-05: "lets change our default sampler for all base / non-distill
  workflows - or rather, any workflow that uses er_sde - to now use euler. no
  er_sde anywhere"; 0.187.0). It replaced `er_sde` / `simple`, the default
  since 2026-08-15. The prompt was vllm-omni keeping Euler its default when it
  added `res_multistep`. Not measured here: the base at its step count on
  Euler. `workflows/h3_config.py::SAMPLING` has the provenance. Prose
  corrected with it: `docs/h3_distills.md` said the base and the distills do
  not share starting noise at one seed, `docs/custom_node_gaps.md` item 2
  listed the sampler as an open gap against the reference engines, and
  `docs/research/h3_dit_implementations.md` said the shipped graphs split
  between `er_sde` and `euler`. Each keeps its text under a dated note.

## 2026-10-04

- **The Subject Track sets itself, and was validated by two other sessions
  before it landed** (owner, 2026-10-04: "OK - if mrblue can validate this
  works, im good", "And same with mrpink when hes done and ready"). mrblue's
  run found the first draft failing on a one-person clip; the rework landed
  as 0.186.45. That evening, on the owner's word ("If you know how to solve
  those, do it. Everyone else is offline"), a third clip broke it again and
  the node now compares a person in two places and asks a lone person for a
  head (0.186.47). It replaced: a pick frame and a match value the user had
  to set, then a score floor that kept a microphone out by a narrow margin.
  [`masked_v2v.md`](masked_v2v.md) has how it works and its limits.
- **Inputs marked advanced stay visible, and that is accepted** (owner,
  2026-10-04, on finding they do not fold away in their frontend: "its
  visible without clicking anything", then "yep thats fine leave it"). The
  marking stays in the schemas and is not relied on to shorten a node.
- **A late start from a softened copy of the original is not adopted for the
  turn.** Eight arms: he turns, and his clothes change, and on one seed his
  hair (`bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`). The code
  the arms ran on is saved beside that record and is not in the tree. What
  is left is the owner's to choose between: a clause about facing, or
  different clothes.

- **Masked video to video, version one, is the two-node design on the
  distilled chain** (owner, 2026-10-04: "Yes I did say the two node design",
  "Lets make that the shipped workflow", "if good, we can lock in and say
  success for v1"). Subject Track picks one person once and follows them
  across cuts; Masked Source says what happens to them, at the margin and
  composite the owner chose; the PDD8 song chain with Sol-Attn renders it.
  Decided against, with the owner's words where there are any: a ControlNet
  ("I dont want a controlnet though"); the baseline chain for this work
  ("i dont think its worth the long render time"); two samplers, parked for
  another day; more work on the first clip ("not worth the effort", its dark
  lines being seams in the source's own backdrop). Open at the time of
  writing: the replaced subject does not follow the original's movement
  unless the prompt names it (owner: "he doesnt turn around at the end").

- **Masked video to video is built on the song node, not on a daily graph or
  a ControlNet** (owner, 2026-10-04: "Not sure i wanna use a controlnet model
  unless we absolutely have to"; "we should look to our frozen audio workflows
  and prompts for reference because the lip sync worked great in them"). The
  owner asked for the distilled daily graphs as the base; the PDD8 song chain
  was used because a mask does not survive two samplers as wired
  (`docs/h3_audio_freeze.md` section 4, with the fix a peer found). SAM 3.1 runs from Meta's original
  checkpoint repacked here (`bench/convert_sam3_checkpoint.py`), at the
  owner's word: "I wanted original weights so we have control over
  conversion".
- **`docs/h3_references.md` said "There is no mask"** of editing a source
  video. Still true of the ref2va edit; a dated note there now points at the
  masked path.
- **A node's tooltip says what the input does, when to change it and what it
  costs; why a default is what it is goes in a comment and here** (owner,
  2026-10-04: "make the tooltip and code comments and everything else
  actually simple to understand. Chosen by the owner means nothing to someone
  using it"; and of the widget names, "I like the widget names as-is because
  its part of the pipeline"). Applied to the reference nodes (0.186.1); the
  input ids stay.
- **The masked source keeps its mask across runs, and nothing is saved or
  loaded by hand** (owner, 2026-10-04, asked whether pre-encoding could speed
  up testing, and shown that the tracker and the part detection cost more
  than a window of sampling after a restart: "save the mask, not the
  latent"). `MiniMaxH3MaskedSource.reuse_mask`, on by default, `mask_store.py`.
  They are kept in the output folder's `masks/` (owner, the same day: "output
  folder in a masks folder like latents is"), not in ComfyUI's user folder,
  where the first commit put them.
- **Masked renders are one pass for now; two samplers are for another day**
  (owner, 2026-10-04: "its more complexity than we need ... just focus on one
  pass"). `MiniMaxH3RestorePlate` stays registered and unused; the latent
  output and the daily-chain graph it would need are not being built.
- **An omitted `qwen_view` or `size_policy` stays a loud failure, with no
  None branch** (owner, 2026-10-04, on the two options put to them). Comments
  in `reference_conditioning.py` and a docstring in
  `bench/check_reference_runtime.py` used to claim core substitutes the
  schema's first option for an API prompt that omits a DynamicCombo. Core
  substitutes nothing and the node fails at execute. The same day the owner
  restated the default: the text encoder's copy is `separate` at a 512 short
  edge "for now", not `shared`.

## 2026-10-03

- **The graphs the owner renders with get their own folder, and the
  workflows are to be organized by purpose** (owner, 2026-10-03: "if we're
  using something regularly ... why keep that in distill experiments?" and
  "can we organize our workflows while we're here? not just throw it all in
  one giant folder"). Landed: `workflows/daily/` with the ref2va, t2v and i2v
  PDD8-then-FlashGen finish graphs (`h3_config.DAILY_GRAPHS`). Chosen and
  not yet landed: shipped picks by task, experiments by topic, the
  instrumented twins and bench graphs apart; prepared on the local branch
  `workflows-by-purpose`. This narrows the 2026-09-27 rule that session-made
  distill graphs go to `distill_experiments`: a graph in daily use is not an
  experiment.
- **Last-bits differences in the conditioning are accepted** (owner,
  2026-10-03: "i dont care if it differs by such a small amount"; on the
  board: "dont care if theyre not bit identical"). That is the difference
  between encoding references and prompt in one pass and keeping the
  references across prompt edits (`bench/results/2026-10-03_reference_split.json`).
  It is a switch, `keep_references`, and on in no generated graph: the
  generator change that turned it on was withdrawn before commit while a 512
  encoder view was about to become the default, where the one pass is already
  a few seconds (CHANGELOG 0.185.19).
- **A still can be an encoder-only reference by its own switch** (owner,
  2026-10-03: "maybe we should at least have a setting in our ref node(s) to
  use vae or not", after one clip rendered that way: "look the same to me,
  cant tell the difference"). `use_vae` on `MiniMaxH3AppendRefImage`, on by
  default; no default moved (`bench/results/2026-10-03_encoder_only_reference.md`).
- **A still's encoder view is `separate` at 512 by default again** (owner,
  2026-10-03: "yeah lets do 512, unless theres a reason not to based on
  everything else we did", then "i agree make it 512 default"; said in peer
  sessions and relayed). The append node's `qwen_view` default, the constant
  (`h3_rules.REF_QWEN_SHORT_EDGE`) and the generated reference graphs move
  together. **It reverses the `qwen_view` part of the 2026-09-13 vendor-parity
  decision below**; `size_policy=max`, the 2048 short edge and `allow_upscale`
  stay at parity, and the vendor still gives one copy to both readers.
  **Evidence, and it is mixed.** Three clips of the PDD8 then FlashGen finish
  graph at the judged seed, not blind: market and backstage at the 512 view
  with the shipped sparse attention ("both looked pretty good to me"), and
  backstage at the 512 view with dense attention, rendered to separate the
  view from sparse attention ("dialogue is good here and everything is
  fine"). The sampler ran about a minute shorter on the sparse pair
  (`bench/results/2026-10-03_sol_output_check.md`, section 5).
  *The level moves with the view:* backstage against the dense clip at the
  shared view is +1.1 LU for the shipped render, +3.8 LU at 512 with sparse
  attention and +4.1 LU at 512 with dense
  (`bench/results/2026-10-03_backstage_qview512_clip.json`), the range the
  owner called loud on 2026-10-02; the owner heard the dense 512 clip and
  passed it. On the market scene the 512 clip's level does not move against
  the shipped clip (`2026-10-03_encoder_only_reference.md`, loudness).
  *A swapped line:* on the backstage clip at 512 with sparse attention a line
  was spoken by the wrong character; at 512 with dense attention it was not.
  The generated graphs run sparse attention, so the pairing that becomes the
  default is the one whose only dialogue clip had the swapped line. One clip,
  not attributed.
  **Not looked at:** more than one reference, a small face in a wide still,
  text or a logo in the still, a base (non-distilled) graph. No render was
  made on the rebuilt graphs. This change was built, held when the level
  result came back, and released when the owner passed the separating clip.
  The remedy for a still that needs the detail, or a scene that comes out
  loud, is `qwen_view=shared` on its append node. The instruments stay on
  `shared`: the `_savelat` and `_x0` twins and the bench graphs (byte-for-byte
  comparisons), and the `h3_probe_refview2_*` scenes (their manifest's
  `parity` arm). A workflow saved before this keeps the view it was saved
  with. What the prose used to claim: `docs/evidence.md` "Reference sizing",
  `docs/h3_references.md`, `docs/custom_node_gaps.md` and
  `docs/h3_conditioning_end_to_end.md` called `shared` the default; each has a
  dated note now.

- **The output check goes ahead of the per-head calibration** (owner,
  2026-10-03: "and output check first"). Every sparse lever in flight is
  lossy, nobody is judging clips, and local error did not predict the
  2026-10-02 verdicts, so a calibrated table had nothing that could accept
  it. The sweep instrument and calibrator are built and wait
  (`bench/results/2026-10-03_tau_sweep.md`); the check's record is
  `bench/results/2026-10-03_sol_output_check.md`.

- **The sparse attention default stays as it is** (owner, 2026-10-03: "leave
  the shipped default as is", and no more renders on the question). On the
  market scene at three seeds the owner passed dense attention at all three,
  sparse in the first stage with a dense finisher at all three, and the
  shipped setting (sparse in both stages) at two; the third has the double
  crate, and it is the seed the 2026-10-02 panel judged. A dense finisher
  costs part of the saving. The owner kept the default; a dense finisher is
  a per-render remedy, not a shipped setting.
  `bench/results/2026-10-03_sol_output_check.md`.

- **No automatic judge for action defects** (owner, 2026-10-03, asked
  whether to build one: "no"). The output check found that latent distance
  from the dense render, cut count and loudness do not predict the owner's
  verdict on the market scene, where the defect is an action done wrong. The
  owner's glance stays the acceptance test for a lossy change; latent
  distance keeps one use, certifying that a render is nearly the dense
  sample. `bench/results/2026-10-03_sol_output_check.md`.

- **Corrected: `docs/SOLATTN.md`'s options table carried two retired
  defaults.** The `dense_blocks` row said `45,48,49` and the
  `sink_conditioning` row said `exact_kv_and_rows`, and that no graph ships
  `exact_kv_and_all_rows`. Both rows now point at
  `sol_attn_h3.py::SOL_DENSE_TAIL` and `SOL_SINK_DEFAULT`. Found in refdude's
  review of the ref2va path, 2026-10-03.

- **Sparse settings go per head and per segment** (owner, 2026-10-03, on the
  lever board: per-head table, "love it. lets do it"; separating text, video,
  audio and reference rows, "a granularity dive ive been wanting"). The same
  day the owner said no more judging sessions for now and asked for
  lower-level work on the kernels and on how the nodes compose with core. The
  kernel half is in the installed kitchen build (`tau_map`,
  `bench/results/2026-10-03_kitchen_tau_map.md`, 0.185.7); the node and the
  calibrated table are not built.

- **Corrected: `docs/sol_upstream.md` named the installed kitchen build by version.** Its 2026-10-02
  section said `vendor/rebuild_kernel.sh --check` reports the build current against core's pin
  "which core moved to `0.2.36`", and pointed at the 2026-10-01 merge record for its contents. The
  build was rebuilt on 2026-10-03 against core's next pin. The section keeps its text with a dated
  note that points at the pin in ComfyUI's `requirements.txt`, the build record beside the venv and
  `bench/check_sol_kernel.py`. The 2026-10-01 entry below and the dated records that name that
  day's build are history and stand.
- **Corrected: `docs/hardware.md` said host RAM was "ample" and stopped there.** Every module is
  configured below the speed it reports (found by a dotfiles session, verified with `udevadm`).
  `bench/hwinfo.py` now prints each slot and flags it; the doc says what it can and cannot move.
  Its "Last updated" header read 2026-08-17 with a last commit of 2026-08-25.
- **Corrected: `bench/profile_sol_stages.py` wrote a fixed `model` string** ("captures from a base
  16-step t2v render") into every record, and `bench/results/2026-10-03_sol_stages_ref2va.json`
  carried it for a ref2va capture. The tool now reads the source render from the capture set's
  `manifest.json`; that record's field is rewritten to what the tool writes now.

- **The Sol node is shown as "MiniMax H3 Sparse Attention"** (owner, 2026-10-03:
  "MiniMax H3 Sparse Attention works"; not core's exact name, which is "Model
  Sparse Attention"). The owner wants the node to be the home for sparsity and
  attention efficiency in general, not Sol's method alone. Display name only:
  `node_id` stays `MiniMaxH3Sol` (`docs/comfy_notes.md`, the `node_id` rule), so
  no graph changes. It used to read "MiniMax H3 Sol-Attn" (0.185.1).

## 2026-10-02

- **Sol's `sink_conditioning` default becomes `exact_kv_and_all_rows`; `dense_blocks`
  stays Option C** (owner decision, 0.185.0). The owner stopped the output-level
  panel after one judged seed and asked for "a good solid general default".
  `all_rows` was the one Sol setting rated fine on the dialogue scene and lost on
  no scene; the dense_blocks choice did not move the verdict. Applied to every
  graph, video references included, against the roadmap's 2026-09-10 plan that
  kept ref2va on `exact_kv_and_rows`. Evidence and cost:
  `bench/results/2026-10-02_sol_dense_blocks_panel.md`. Retired the
  `h3_candidate_t2v_sol_allrows` graph, now the same as its base. The
  `SOL_DENSE_TAIL` comments used to say Test 9A "proved" Option C; they now say it
  was kept by decision.

- **The dense_blocks campaign's report was corrected against its captures**
  (0.184.13, `bench/results/2026-10-02_sol_campaign_reanalysis.md`). It used to
  say Test 9C ran `exact_kv_and_all_rows` (it repeated Test 8's settings), that
  the Test 9 arms held the seed fixed (every run has its own seed), that Test 7
  shows multi-shot robustness (it shares Test 6's settings with a different
  prompt and seed), and that Options A to C and `start_percent=0.2` lowered
  error (on the cells every run measured, they sit within prompt-and-seed
  noise). The Option A, B and C adoptions below rest on local probe error alone.
  `SOL_DENSE_TAIL` is unchanged pending the owner.

- **Sol's `dense_blocks` adopts Option C / Full Ridge Shield (`"38,39,40,41,42,49"`)** (owner decision, 2026-10-02).
  Replaces Option B (`"39,40,41,42,49"`) as active default `SOL_DENSE_TAIL`. Empirical findings
  from Test 8 ($\tau=1.3$) revealed that Block 38 alone breached the 20% error ceiling (22.09% peak call error).
  Test 9A proved that shielding Block 38 alongside Blocks 39–42 and 49 drops whole-network peak call error to
  a record-low 17.73% (Block 43 is now the worst remaining block), while ref_img max error drops below
  20% (19.37%) and routed density remains ultra-sparse at 15.18% (~20.3% relative compute reduction over
  $\tau=1.1$). 166 workflow graphs rebuilt and validated against the live server. (0.184.12).

- **Sol's `dense_blocks` adopts Option B (`"39,40,41,42,49"`)** (owner decision, 2026-10-02).
  Replaces Option A (`"39,41,42,49"`) as active default `SOL_DENSE_TAIL`. Empirical findings
  from Test 5 and Test 6 proved that shielding Block 40 eliminates the last middle routing spike
  (where Block 40 had 19.17% error under Option A), bringing whole-network peak call error down
  from 23.26% to 19.74% (the first time peak error broke below 20%). 166 workflow graphs rebuilt
  and validated against the live server. (0.184.11).

- **Prose corrected by the 2026-10-02 upstream read** (0.184.9). Each keeps a
  dated note in place:
  - `references.md`, "What moved by 2026-09-25", said sglang keeps the AdaLN
    affine in fp32 and core rounds at more points. sglang's bf16 block kernel
    rounds at the same three points as core's `_mod_scale_shift`, and has since
    2026-08-18, so the claim was wrong when written.
  - `research/sglang_h3_pipeline.md` section 10 and the seventh read in
    `research/sglang_comparison.md` said sglang's ComfyUI app runs H3 in
    server mode only. Since `f1e62e3a2e` it also runs the H3 DiT per step under
    a ComfyUI graph.
  - `custom_node_gaps.md` item 3 said diffusers decodes the H3 VAE in fp32
    under autocast. Since `51a454be9` its decoder runs in the pipeline dtype.
  - `sol_upstream.md`'s 2026-09-25 INT8 VAE paragraph and the matching
    sentence in `references.md` still described the file as removed; the
    owner reopened it on 2026-09-26.
  - `research/2026-09-25_step_caching_survey.md` listed Spectrum as literature
    only; sglang ships it for H3, opt-in, since `ae47bcd4da`.
  - Core line citations in `sol_upstream.md` and `h3_capture.py` re-read
    after `2d6b7328` moved H3's embed span.

- **Test 2 confirms middle error peak invariant across two-stage sampler and exposes token routing limits**
  (empirical finding, 2026-10-02). Running `h3_text_to_video_pdd8_flashgen_finish_api.json` (6 steps PDD8 +
  2 steps FlashGen finisher; 119,102 tokens with 1 reference image at 2752x1536 yielding 7,360 VAE reference
  rows; 400 DiT calls captured across two sequential `MiniMaxH3Sol` nodes without overwriting) verified that
  the middle error plateau is invariant across samplers: Block 39 at 25.18% avg relative $L_2$ (vs 25.22% in T1),
  Block 42 at 24.59% (vs 25.67% in T1), Block 40 at 23.93% (vs 23.43% in T1), and quiet tail blocks 46 (7.37%)
  and 47 (6.53%). Furthermore, `token_routing="measured"` succeeded on localized layers (Block 0 dropped to 5.37%,
  Block 32 dropped to 12.99%) but failed catastrophically on diffuse middle layers (Block 40 Head 47 exploded to
  111.06% error, cos 0.6655), proving diffuse middle layers cannot be rescued by token routing and require
  full dense execution (`dense_blocks`). Preflight geometry confirms Triton int32 crossing safety at 99,864 tokens
  via 64-bit index arithmetic and 64-bit CUDA quantizer offsets (`ELEMENT_OFFSET_BITS=64`). Full analysis in
  [`docs/research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md).

- **Sol's `dense_blocks` adopts Option A (`"39,41,42,49"`), with Option B (`"39,40,41,42,49"`)
  named** (owner, 2026-10-02: "change the sol dense detail in whatever builds the workflows and the
  h3 config and sol attn h3 to be the Option A candidate list, but also include the Option B candidate
  list as a named variable, so switching is possible"). Replaces the 2026-09-25 tail default
  `SOL_DENSE_TAIL = "45,48,49"`. `SOL_DENSE_OPTION_A`, `SOL_DENSE_OPTION_B` and
  `SOL_DENSE_HISTORICAL_TAIL` are defined in `sol_attn_h3.py` and `workflows/h3_config.py`.
  `SOL_DENSE_TAIL` defaults to Option A. 166 graphs rebuilt and validated against the live server.
  (0.184.8).

- **Re-evaluating Sol's default `dense_blocks="45,48,49"` under `quantizer="rotated"`** (instrumentation
  finding and experimental trial). The 2026-09-25 default `dense_blocks="45,48,49"` was established
  under unrotated INT8 attention, where channel outlier spikes in $K$-norm destroyed quantization
  accuracy on blocks 45, 48, and 49 (`docs/h3_block49_quant_error.md`). With `quantizer="rotated"`
  (Hadamard rotation) active since 2026-09-27, activation energy is evenly dispersed across all 128
  channels: Test 1's all-block instrumentation run (Ref2VA, 119k tokens, 8 PDD steps, unmasked) proved
  that Block 48 drops to 7.14% avg relative $L_2$ error (lower than almost all middle blocks), and
  blocks 46–47 are 6–7%. However, routing sparsity truncation error reveals a severe error spike in the
  middle semantic integration layers: Block 42 at 25.67% avg relative $L_2$ error (max 27.01%),
  Block 39 at 25.22%, Block 41 at 24.35%, and Block 40 at 23.43% (worst heads losing up to 80%
  magnitude with cosine similarities down to 0.77). Retaining Block 48 in `dense_blocks` is wasteful
  while leaving the 25% middle cluster unprotected. Block 49 remains mandatory as it directly feeds
  `final_layer.video_out`. Top candidate sets under a 5-test matrix: Option A (`"39,41,42,49"`, 4 blocks)
  and Option B (`"39,40,41,42,49"`, 5 blocks). Full analysis and test plan in
  [`docs/research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)
  (0.184.7).

## 2026-10-01

- **PDMD is parked, on kijai's resized files** (owner, 2026-10-01: "just keep kijais and note it.
  nothing else to do with it"). The blind first look preferred FlashGen, and kijai's resize was
  "same" as full rank on every scene (`docs/h3_distills.md`, "PDMD"). `h3_config.PDMD_LORA` and
  `PDMD_2STEP_LORA` name kijai's files. The full-rank conversions, their converter and the four
  kijai-arm probes are retired; the owner deleted the downloaded weights. This reverses 0.182.6's
  choice of full rank as the default (below). `docs/roadmap.md`, "Closed lanes", says what reopens it
  (0.184.6).

- **FlashGen and PDMD keep `start_percent` 0.2** (owner, 2026-10-01: "change it back", scoped to
  the FlashGen/PDMD extension). 0.184.1 had extended the PDD result to them without a measurement;
  0.184.3 reverts it. PDD stays at 0.0.

- **Sol's `start_percent` is 0.0 on FlashGen and PDMD graphs too** (owner, 2026-10-01: "flashgen
  should change i think"; PDMD: "may as well"). This extends the PDD decision below to the distill LoRAs
  applied at the call, through `h3_config.SOL_DISTILL_LORA_OVERRIDES`, kept apart from
  `SOL_PDD_OVERRIDES`. **Not measured on them:** the panel's FlashGen pass starts at sigma 0.8, below
  the window, so no FlashGen or PDMD render from noise has been judged at 0.0. 32 graphs rebuilt,
  `start_percent` the only change. The first PDMD-against-FlashGen renders
  (`bench/results/2026-10-01_pdmd_vs_flashgen.jsonl`) ran before it, at 0.2 on every arm (0.184.1).

- **The at-call LoRA branch stays as it is** (owner, 2026-10-01: neither fix). Fusing swiglu into
  fc2's projection or folding the add into kitchen's int8 matmul would recover part of the branch's
  cost, at the price of exact parity with the published `H3ExactLoRA` and, for the kitchen route, a
  carried kernel (`bench/results/2026-10-01_lora_branch_profile.md`).

- **Sol's `start_percent` is 0.0 on PDD graphs, 0.2 elsewhere** (owner, 2026-10-01, option A of two:
  the distill graphs only, not every Sol graph). Five blind pairs on the t2v finish showed no
  difference and 0.0 is much faster; the base graphs were not tested, so they keep 0.2. Upstreams
  disagree (core 0.2, sglang a fixed dense step count), so the adopt-upstream rule did not decide it.
  `docs/SOLATTN.md` and `h3_config.py` used to say the knob had never been measured (0.184.0).

- **Kitchen int8 stays Sol's dense fallback** (the owner reopened sage fp8++ rotated for it on
  2026-10-01). Measured the same day: kitchen is more accurate on every dense cell, the steps before
  Sol's window included, and sage saves about one percent of the sampler. `next_steps.md` used to
  list the lane as parked pending exactly this (0.183.5).

- **PDMD ships as `h3_text_to_video_pdmd`, unrendered** (owner, 2026-10-01: "Go for it"). It is PDMD
  4-step on its trainer's contract: the full-rank file at the call, Euler on `simple` at 12/3, the
  repo's Sol default. The 2-step file and kijai's two resizes are `distill_experiments/` probes, and
  `check_distill_settings.py` grades all four. The first render is the owner's call (`30b82bb6`,
  0.183.1).

- **PDMD runs our full-rank conversion; kijai's rank-reduced files are the arm** (owner,
  2026-10-01: "Sounds like we both agree"). Kijai's resize keeps a fixed share of each delta and less
  where q/k/v hit his rank cap (`bench/results/2026-10-01_pdmd_{4,2}step_lora_conversion.json`).
  Applied at the call, rank buys little speed (`bench/results/2026-10-01_lora_branch_profile.md`).
  So the default carries the published delta exactly, as `FLASHGEN_R64_LORA` does for FlashGen
  (`bench/convert_pdmd_lora.py`, `12cd81ad`, 0.182.6).

- **h3-mutant-distill gains a second node, `H3KeyframeCanvas`** (owner, 2026-10-01: "I agree with
  your auto resize approaches"), so its i2v examples size the canvas from the first frame instead of
  stretching it. The pack's README no longer reads as one node only; `H3ExactLoRA` still loads every
  adapter. Keyframes are not sized like references: the 2048 short edge is ref2va's alone (0.183.0).

- **comfy-kitchen's build branch moved to upstream main `3f7210f`** (owner: merge what upstream has).
  `h3-frontier` took seven upstream commits by merge (`aade8d5`); the build is
  `0.2.36+sol.aade8d5.up.3f7210f`. Outputs are bit-identical on every layer and render checked, and
  the int8 GEMM's qkv and fc1 are faster. Record: `bench/results/2026-10-01_kitchen_merge_aade8d5.md`.
  The branch was pushed to `origin` and `nas` the same day, at the owner's word, as fast-forwards to
  `aade8d5`. The kitchen's own tests must be run from a copy outside the clone, and its `test_bindings_*` tests are HIP-only checks that poison a CUDA run
  (same record).

- **Audio refine is never a stage of a mutant example** (owner, 2026-10-01: "usually not
  worth the extra cost"). It belongs in a workflow of its own, named `*_audio_refine`.
  None of the nine examples had one; `bench/build_mutant_examples.py` now refuses an
  example carrying `AUDIO_REFINE_CLASSES` under any other stem (0.181.4).

- **`h3_config.py`'s `end_percent` note said "sage still takes the steps before
  `start_percent`"**; since 2026-09-15 the default graphs run those steps on
  `DENSE_BACKEND_NODE` (kitchen int8, as the server log's `[h3-sol]` line says), with
  sage only on the `FLOOR_STEMS` arms. Corrected in place (0.181.2).

- **The i2v PDD8 pick is promoted to `workflows/`, and two recipes the mutant repos did
  not ship become examples there** (owner, 2026-10-01, "ok to both" to the two calls in
  `bench/results/2026-10-01_r2v_i2v_graph_coverage.md`). `h3_first_frame_to_video_pdd`
  left `distill_experiments/` for the root (0.179.2). `h3_i2v_flashgen` and
  `h3_r2v_pdd8_flashgen_finish` join h3-mutant-distill's example workflows (0.180.0), each
  held to its pack graph by `bench/check_mutant_parity.py`. The second call named no
  rows, so this reads it as the unshipped rows with positive evidence. Left out: the
  FastH3 finishers (parked 2026-09-29, and the FastH3 mutant is on hold) and the i2v
  FlashGen finish (rejected 2026-09-27). Say so and they are one recipe each. Nothing was
  pushed or uploaded: the GitHub README and the HF card here carry the new rows, and
  publishing them is the owner's action.

- **The wiki's prompting page is folded into `docs/prompting.md` and removed**
  (owner: resolve any redundancy so there is one source of truth). It was a
  router that also restated rules, and the restatements had drifted. What it
  used to claim that was wrong:
  - the shot-header timestamp rule "is not yet measured in either direction"
    (measured once on 2026-09-18; the 2026-09-26 correction below missed this
    page);
  - turns per shot "open in base format" (§14.3 closed it on 2026-09-01);
  - "sources 1 to 3 are gitignored" (the guides in `vendor_guides/` are
    tracked);
  - the bank's `adapt` column as which prompts "can take another frame count
    without a rewrite" (the bank preamble says it is not a permission);
  - that the portable files' every quotation is checked (the check pins the
    Part One strings and the camera table in the system prompt, and more in the
    HTML, never their prose rules).
  `docs/wiki/index.md` had also described it as ranking five sources with at
  least five worked examples per mode; it ranked four and carried none. Its two
  routing tables moved into the opening of `docs/prompting.md` and into its §11.
- **`docs/prompting.md` corrected where it disagreed with itself or the code.**
  - §5.6 said every line takes a mouth-closing cue and every on-screen
    non-speaker "produces no vocal sound"; §14.3 and §15.3 item 5 had narrowed
    both. The cue is positional and the phrase is for a character who never
    vocalises.
  - §9.10 repeated the withdrawn "42-68 words" and said the two scene arms were
    never rendered. Both gone; the preflight command stands in for the count.
  - §15.2 said nothing is stripped. Core's tokenizer strips nothing; our
    conditioning nodes strip both ends (0.160.0).
  - The owner's ruling of 2026-09-01, that cinematography wording outside base
    §4.3's table is acceptable, lived only in the portable copies and in a code
    comment that cited "section 8". It is now §4, and the camera warnings are
    informational.
  - The no-header-timestamps rule is labelled OWNER, as the layer table defines
    it; §3.1 and §11 said HOUSE.
  - §13's `workflows/*.json` glob skipped `workflows/distill_experiments/`; the
    command now walks `h3_config.graph_paths`.
  - "All five sources" in the header and the index: four are live.
- **Specificity is written down (`docs/prompting.md` §16)** from the owner's
  statement on 2026-10-01: pin what matters where the scene is ambiguous, and
  because only so much can be controlled, pick the few that matter and let the
  model carry the rest. The records behind it were scattered across the
  changelog, `bench/results/` and two research notes; §16 indexes them. It
  records that nothing measures the second half, and that the bank's
  "specificity ladder" is a different experiment.
- **The `h3-prompt` skill carried one rule found nowhere else**, to introduce a
  likeness as `[Name] (played by [Actor] in [Show])`. It came from another
  session's external writer craft (2026-09-11). It is now an OPEN line in
  `docs/prompting.md` §9.11, and the skill names sections and restates no rule.

- **Prompt-carrying nodes have one registry, and prose that said otherwise is
  corrected** (the owner asked for the song-node gap to be redesigned; 0.181.0).
  `workflows/prompts.py` said the catalogue "keys on the same set" of nodes
  (it did not), and `bench/preflight_graph.py` called its list "every node that
  carries a prompt" (true until the song node, which it never knew). Both lists
  are now `h3_config.PROMPT_INPUTS`. The manual's section 3.1 said a malformed shot
  header makes the shot list empty and "the grader is removed rather than reddened",
  and its section 11 said such a header "takes the shot rules inert"; preflight now
  FAILs it (0.181.1), and the same pass made the stamped-header test catch
  every spelling and the retention-line speaker-id test read compound ids.

- **The portable copies' prose is synced to the manual** (0.181.1). What they used
  to say: the HTML labelled the timestamp rule House (the manual says OWNER), gave
  the un-narrowed "every on-screen character not given an explicit 'produces no
  vocal sound'", said "cut timestamps picked by vibe overrun the clip", and carried
  a 2026-09-01 date line. The writer prompt tagged the `N/A` habit warning
  `[guide]` (it is house), gave a turn count that disagreed with the bank, carried an
  undated count of outputs graded clean, and told the writer never to write an
  absence while the bank's own cast-count fixes state one. Both now carry the
  specificity rules, and twelve rule sentences are pinned across the manual and the
  copies (`RULE_PINS` in `bench/check_prompt_docs_sync.py`). The published claude.ai
  copy is still the 2026-09-01 snapshot.

- **`docs/checks.md` said the vendor guide lives in gitignored `internal/`**, in the
  conformance row's needs cell and the camera row's allowlist note. The guides are
  `vendor_guides/`, tracked and hash-pinned, and the camera check parses the one it
  needs. Code comments carried the same claim, and cited `base-en.txt` and
  `ref-en.txt` (not our files) with line numbers off by one or two; all corrected.

- **Prompting-related comments cited `CLAUDE.md` for rules it no longer holds**:
  "cite one before building", the one-implementation trap, the second-reader finding,
  "a default is not a decision", "when something gains an off state", and the
  numbers-in-prose rule. They now point at `docs/rules_history.md`, `docs/checks.md`,
  `docs/evidence.md` and `docs/prose_measurements.md`. Bench scripts that are not
  about prompts still cite it: `grep -rn "CLAUDE.md" bench --include=*.py`.

- **Two docstrings said a bare `workflows/*_api.json` glob misses nothing; it has
  missed `distill_experiments/` since 2026-09-27.** `bench/preflight_graph.py`
  said its single glob "currently misses nothing" because `GRAPH_DIRS` was
  `("",)`, and `bench/check_graph_discovery.py` said its rule held "while
  `GRAPH_DIRS` is `("",)`". `GRAPH_DIRS` gained `distill_experiments` on
  2026-09-27, so the first command skipped every distill graph, the ref2va and i2v
  ones included. Both docstrings are corrected and `preflight_graph.py`'s usage
  line carries both globs. Found while checking that those graphs are kept
  (`bench/results/2026-10-01_r2v_i2v_graph_coverage.md`: they are, and
  `bench/clip_recipe_coverage.py` now asks the question).

## 2026-09-29

- **Two ref2va findings re-checked by a second pass; details corrected**
  (board `ref2va-verify`, mrblue). Both of h3lora's findings mostly hold, and
  their records are left as written; this line is where a reader who trusted a
  detail is pointed. The records are
  `bench/results/2026-09-29_ref2va_partition_delta_verify.md` and
  `2026-09-29_ref2va_block49_verify.md`. What changed:
  - The block 49 record's attention-mass figures were read as made with the
    script's docstring command; they were made at `--queries 256` (the default
    is 384), and the median effective-key figure moves with sample size. Only
    `heads_eff_under_20` supports "peakiness alike".
  - The block 49 record's "the text rows' gates are exactly zero, so nothing
    trained them" was stated for the whole ref2va text span. That span includes
    vision rows tagged as video, whose block 49 modulation is not zero. The
    dropped-row claim is unaffected.
  - The partition-delta record's block-0 audio modulation "standout" is partly a
    small-denominator ratio, and the audio tail's rise starts near block 30, not
    block 45.
  - `2026-09-29_partition_delta_map.jsonl` predates its script: `cos` is above 1
    in about half its rows. Regenerate it before reading `cos`.
  - Stale lines in `next_steps.md` were corrected in the same pass: #36 "waits
    on the owner's go" (it was rendered and read, and its blind batch awaits the
    owner), and PDD8-finished-by-FastH3 "the scoring is what remains" (it was
    scored and parked, two lines up).
  - What the checks now support: ref2va's block 49 is the hardest INT8 cell
    mainly because the grade counts rows the model discards
    (`2026-09-27_sol_redesign_test2.md` did not say why), and "not a time warp"
    holds under a control that can fail (`bench/control_time_warp.py`).
  - Later the same day, on the real kernels (`2026-09-29_grade_video_audio_real_kernels.md`,
    all 56 heads): over the video and audio rows ref2va's block 49 is close to t2v's,
    so "the hardest cell" was mostly the grade. A first run at the grader's default of
    8 heads read the opposite and was withdrawn; the default is a trap for block 49.

- **FastH3 finisher and FastH3 mutant: parked** (owner). No more FastH3-finisher
  tests, and the FastH3 mutant (overlay plus loader, and shipped loader graphs)
  is on hold, because FastH3 still requires its full weights. This records
  what was tried, not a verdict on the method: PDD8 then FastH3 at 12/3 was
  indistinguishable from the FlashGen finish by eye on one t2v and one ref2va
  scene, and at 10/3 it was grainy
  (`bench/results/2026-09-29_blind_sessions_read.md`). Whether FastH3's finish
  adds anything on audio or motion was not measured. Built and kept: the
  finisher graphs, the overlay and its loader. Board: `fasth3-finisher` and
  `overlay-followups`, both parked.
- **FastH3's look travels with its gates, not its backbone change** (#35,
  measured, not judged by eye). `2026-09-27_fasth3_swap.md` put the look in
  "its weights and gates" and could not say which; the gates arm answers:
  fl2va plus FastH3's gate tensors behaves like FastH3, and FastH3 without
  them behaves like fl2va. Two predictions filed on 2026-09-27 (G1, G2) were
  falsified. It also closes the general adapter (`fasth3-adapter` on the board):
  the backbone change is not low-rank in bf16 and does not carry the look.
  Record: `bench/results/2026-09-29_fasth3_gates.md`; rank:
  `bench/results/2026-09-29_fasth3_bf16_rank.md`. One seed, three scenes, and
  the no-gates arm also drops VSA's coarse branch, so gate values against the
  branch's presence is open.
- **Block 49's INT8 problem is specific to unrotated attention** (owner,
  2026-09-29). It is sage's plain fp8++ and stock Sol's per-row K scale that
  meet block 49's four loud K channels badly; comfy-kitchen's `int8_attention`
  already rotates q and k and never had the problem, and the sage fork's
  rotated mode (`qk_rotate`, the sage node's `auto` since 0.129.0) was added
  to address it. `docs/h3_block49_quant_error.md` says so; the router row in
  `index.md` now does too, and the passages of `docs/SOLATTN.md` that read
  block 49 as a general last-block problem carry dated notes: the output head
  "reached directly" (propagation was measured 2026-08-29 and moves least for
  45, 48 and 49), the dense fallback described as sage, and the K-channel fold
  as "the move at block 49". Nothing was removed.
- **TaoMate-H3 is history only** (owner, again 2026-09-29; deprecated
  2026-09-27). `CLAUDE.md`, `docs/h3_quant_policy.md` and
  `docs/h3_block49_quant_error.md` said or implied it was a live reference for
  the tail's precision; each now says otherwise.

- **Eight nodes retired, `llmcompressor` dropped, 33 old bench scripts deleted**
  (owner, 2026-09-29; CHANGELOG 0.173.0 to 0.176.0). Each of
  `MiniMaxH3KeyframeCanvas`, `MiniMaxH3ReferenceFit`, `MiniMaxH3ReferenceVideoFit`,
  `MiniMaxH3MarkerArm`, `SageChainAssert`, `MiniMaxH3ReferenceReport`,
  `MiniMaxH3VSAAttention` and `MiniMaxH3QuantObserve` was in no shipped graph.
  Open experiment #23 is closed, not refuted; the AWQ and GPTQ recipe files that
  needed `llmcompressor` went with the dependency ("llmcompressor is for awq
  stuff - remove that dependency"); the bench scripts went on the owner's "most
  benches that havent been touched since august can probably go", applied as: last
  commit before 2026-09-01, no live reference, and not cited by a September
  record. What prose said before: `docs/custom_node_gaps.md` called
  `MiniMaxH3ReferenceVideoFit` "NOT deprecated ... still live", `SageChainAssert`
  "registered so saved graphs still load", and `MiniMaxH3VSAAttention` a live
  alternative to Sol; `docs/wiki/next_steps.md` told a reader to wire
  `MiniMaxH3ReferenceReport` before a reference render. Each now points at the
  replacement (`MiniMaxH3Conditioning`, `MiniMaxH3AppendRefImage`, core's
  `BlockSparseAttention` in `vsa` mode, the conditioner's preview). Saved graphs
  outside this repo that wire a retired node no longer load. `docs/roadmap.md`
  "Closed lanes" carries #23 and the VSA node.

## 2026-09-28

- **The launcher dropped `--fast fp16_accumulation`, and `comfy.env` dropped
  `NVIDIA_TF32_OVERRIDE=1`** (owner: "Yes make these changes"). Both DiTs
  compute bf16/int8 only, so the flag reached only the fp16 text encoders,
  and its `PRIORITIZE_FP16` made core PR 16508 an fp16 switch for H3. The
  override forced TF32 onto fp32 matmuls, the audio VAE's among them.
  `docs/sol_upstream.md`, `next_steps.md` and #33 used to say this launcher
  runs fp16 accumulation; each carries a dated note. The launcher and its
  reasons live in the dotfiles repo (`comfy/start.sh`, default mode).
  `bench/compare_vae_decoders.py` defaults to no `--fast` to match.

## 2026-09-27

- **PDD6 is a low-motion and close-up option, not the default** (owner:
  "closeups like that are fine for pdd6 - low motion, not a ton of stuff to
  keep track of"; chose "Low-motion option"). PDD8 won all three motion
  scenes: a sword touched on samurai, a missing hand on slapstick, morphing on
  subway (finding ow-fd-11). The graph stays;
  `../h3_distills.md`, "Rules of thumb", says when to use it.
- **The PDD8 finish: full FlashGen from sigma 0.8 on t2v, none on i2v** (the
  owner's review, ow-fd-08 and ow-fd-09).
  - On t2v the finishes were near indistinguishable by eye. Either FlashGen
    finish fixed sign text that PDD8 alone and the base finish garbled.
  - On i2v the finish's brightening "looked off", so PDD8 alone stays.
    *(Not the state of the code since 2026-10-03; see 2026-10-05.)*
  - Warm, it costs no sampling time: its evaluations replace PDD8's.
  - Whether it replaces the default t2v PDD graph is the owner's call.
  - `../../bench/results/2026-09-27_finisher_grid.md`, "The owner's review".
- **Closed, no change: late-only FlashGen as PDD8's finisher (#34).**
  Late-only matched full FlashGen as a finisher on seven scenes, and the full
  finish added no haze for it to remove
  (`../../bench/results/2026-09-27_late_switch.md`). The base finish (#37) has
  its own entry below.
- **Closed: late-only FlashGen alone (#46); no stripped LoRA is built**
  (owner's review, ow-fd-10).
  - Late-only was more natural on the single-figure beach ladder, and more so
    on the unusual rung, where full FlashGen walked in slow motion.
  - It lost coherence where people and objects must hold: a third person on
    subway_chase_short, and an unrecognisable piano on FT1's slapstick.
  - O2 (FlashGen overfit) stays open, not refuted. FT1's "half the haze" does
    not generalise at 124 frames (annotated in its record).
- **Answered for selection: #45, the kitchen's VSA against FastVideo's.**
  - At blocks 0 and 24 the kitchen matches a FastVideo-exact reference to
    within about 1%, so the selection deviations do not explain FastH3's
    over-polish.
  - At block 49 the gap is int8 on the text rows, not selection.
  - Building the reference node, to see whether that is visible, is the
    owner's call.
  - `../../bench/results/2026-09-27_vsa_selection_grade.md`.

- **The save format stays H.264 8-bit crf 19** (owner, #38). H.265 10-bit
  at crf 22 measured about half the dark blocking at half the size
  (`bench/results/2026-09-27_encode_format_ab.md`), and the owner saw no
  difference in the pair: "Dont think I can see a difference". The two
  encodes differ by under one level in the darks. So O1's `dark_block8`
  excess of about +0.05 is below the owner's eye (vd-m12), and O1's
  "splotchy blacks" are not the codec's 8 px blocking; that question is open
  on a clip the owner names.
- **PDD8 then full FlashGen from sigma 0.8 becomes a top-level t2v graph**
  (owner: "then yeah switch it"; 0.167.0), as
  `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json`. The finisher
  grid review found it better than PDD8 alone on three t2v scenes and never
  worse. Full and late-only finishes were indistinguishable, so the full
  one, as trained. Fresh runs sample in the same time as PDD8: its 6 PDD8
  evaluations plus 2 FlashGen ones are 8, like PDD8's own. It is an additional
  graph: `h3_text_to_video_pdd` stays the default until the owner says
  otherwise. i2v stays PDD8 alone, because the finish brightens the frame at
  once. *(Not the state of the code since 2026-10-03; see 2026-10-05.)*
- **The base-model finish (#37) closes with no change** (owner's review,
  vd-v01). It was never preferred, and it morphed a face on courtroom. It is
  not inert: it matched the FlashGen finishes on noodle_bar and partly fixed
  spec_unusual's sign text. PDD8's coarse tail is part of its weakness, and
  FlashGen fixes it better. The measures called it gentle and missed the
  legibility gain. `step_switch_to="base"` stays in the generator for the
  bench.

- **Sol's default quantizer is `rotated`** (owner, 0.166.0). It was
  `balanced`, inherited from `qk_balance=True` (2026-09-15). The owner chose
  it from test 2's grade on the shipped PDD8 graphs: lowest quantization
  error on every Sol-block cell, cheaper than `balanced`, small effect on
  total error. The owner said to switch at the next ComfyUI restart, with the
  reviews done. `bench/results/2026-09-27_sol_redesign_test2.md`.

- **Our conditioning nodes strip the prompt at both ends** (owner: "strip
  leading and trailing whitespace in our conditioning nodes and \n at the end
  (not in the middle - \n has value in the middle)"; "make sure no prompts in
  prompt_bank have the trailing \n"). `h3_rules.normalize_prompt`, in
  `a5ab229f` (0.160.0). Core's tokenizer keeps edge whitespace, so a trailing
  newline was one more token and a different sample at the same seed; the
  generator already stripped bank prompts. The bank check now fails on edge
  whitespace. Core's own H3 nodes still do not strip.
- **Closed: the encoder precision study.** `ENCODER_INT8` stays the default.
  At the int8 DiT, int8's effect sits under a floor that a dose-response
  confirmed (`bench/results/2026-09-27_encoder_quant_floor.json`). Floor-free,
  at the bf16 conditioning path, its image-token error is no worse than random
  and its text-token error is somewhat worse
  (`bench/results/2026-09-27_encoder_quant_refiner.json`).
  `docs/open_experiments.md` #47 closes unbuilt.

- **`qk_balance` is not inert behind the dense tail** (Sol redesign test 1,
  0.163.1). The audit (§4 item 1), the redesign doc, `docs/SOLATTN.md` and
  `docs/h3_block49_quant_error.md` had said or implied that the balance gate
  opens only on blocks 45, 48 and 49, so the shipped dense tail left it
  nothing to do. Renders in three modes and a gate measurement on captures
  refute that. Each of those docs now keeps a dated note.
  `bench/results/2026-09-27_sol_redesign_test1.md`.

- **TaoMate-H3 is deprecated and removed** (owner: "taomate is deprecated
  you can remove everything from it here. we're not gonna pursue it
  anymore."; 0.161.0). Removed: the `MiniMaxH3TaoMateStreamSampler` node and its runtime
  (`taomate_stream_sampler.py`, `taomate_streaming.py`), the two probe graphs
  (`h3_probe_taomate_3step`, `h3_probe_taomate_3step_audio_freeze`), the
  `TAOMATE_*` constants in `workflows/h3_config.py`, the check
  `bench/check_taomate_streaming.py` and the TaoMate cases in
  `check_distill_settings.py` and `check_distill_grid.py`, the converter
  `bench/convert_taomate_lora.py`, the harness
  `bench/verify_taomate_stream.py`, the three arm manifests, and
  `docs/h3_taomate.md`. Deprecated by the owner, not pursued; not a verdict on
  the method. The records stay in `bench/results/` (`2026-09-15_taomate_*`),
  the code in git, and the lane is in `docs/roadmap.md` "Closed lanes". The
  upstream checkouts stay as references (`references.md`, "The streaming
  references: TaoMate").
- **The owner's working encoder is the bf16 pruned file; a better int8 of our
  own is not worth building** (owner: "since the encoder runs once, I agree -
  it may not be worth the effort. Especially if we can run a pruned bf16. I
  care mostly about its vision encoder not losing precision"). Both encoder
  files had been deleted by accident. The int8 file was restored from the
  owner's Comfy-Org download, whose sha256 matches the one the Hub recorded.
  The bf16 pruned file was rebuilt from the release by
  `bench/convert_h3_bf16_encoder.py`, which reads the sharded `text_encoder/`
  directly since `c743d100`. The weight-side headroom record already put the
  shipped int8 at its format's floor
  (`bench/results/2026-08-29_int8_convrot_headroom.json`), so the encoder
  quantisation lane stays closed.
- **Corrected: what the 2026-09-20 entry below says `ENCODER_INT8` was
  measured against.** It was only ever graded against the two W4A16
  artifacts. It has now been measured against the bf16 pruned file on
  shipped graphs and on the violin-maker scene in three modes
  (`bench/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`).
  - The vision tower and embedding table are bf16 in both files and add no
    error.
  - Text tokens stay close.
  - Image tokens carry a heavy tail that the int8 decoder adds, and one
    violin ref2va patch is inflated well past its bf16 norm.
  - bf16 costs seconds per encode.
- **Running: does the DiT respond to that tail** (owner: "The experiment
  would be good"). `bench/measure_encoder_quant_dit.py` (`3acade9e`) runs
  the base DiT at fixed noise, latent and sigma with a null, the treatment,
  an image-tokens-only arm, a norm-matched random control and a one-edit
  scale row. Predictions were filed before any comparison printed. The
  decision it gates is in [`next_steps.md`](next_steps.md).

- **The Sol redesign ships as `MiniMaxH3Sol`, and `MiniMaxH3SolAttn` is
  deleted once the generated graphs move** (owner: "1 delete", "3 your pick on
  name"). The kitchen build branch `h3-frontier` is pushed to the owner's
  GitHub fork and `nas`, never to Comfy-Org ("Never upstream to comfy").
- **Closed: the SLA lane and Morton** (owner: "SLA is closed unless you see
  some reason not to"; "Morton seems dead too unless you see remnants of it").
  Checked first:
  - kitchen's top-k has been unchanged since 2026-09-04, and it lacks SLA's
    learned linear branch;
  - no H3 implementation in core, kitchen or `coderef/` reorders tokens.
  The new Sol node drops top-k, `keep_percent`, `pooled_tail`, `morton`,
  `morton_curve` and the code only they reach. Both are in `docs/roadmap.md`
  "Closed lanes". The redesign's tests use **test renders, not blind panels**
  (owner: "I dont need blind renders just test renders").
- **Approved: redesign `MiniMaxH3SolAttn` as a new node and freeze the old one**
  (owner: "I'm good with this approach"). Retiring an input retires the code
  only it reaches, with its checks and prose ("since its all git tracked
  anyway"). Morton becomes one `reorder` input, with one test as a speed
  lever before removal. The plan, reasons, tests and bug list:
  `docs/research/2026-09-27_sol_node_redesign.md`.
- **Reversed: the kitchen build tracks upstream main, not ComfyUI's pinned
  tag** (owner: "I want to be able to stay on the frontier here... So long as
  we know what exists where and why"). It used to be that untagged main was
  "not built by policy" (`vendor/rebuild_kernel.sh`, since 2026-09-11). That
  kept #207/#208 (kitchen int8 attention; #208 adds an Ada-only cached-Q path
  at head_dim 128 with no mask, which is how H3's dense backend calls it) and
  #192's `fp16_conv3d` depth gate out for a release cycle (the gate turned
  out not to reach the tiled H3 encode:
  `bench/results/2026-09-27_vae_encoder_fp16acc.md`). Now we build
  `h3-frontier`, upstream main plus our Sol commits, moved forward by merge.
  The installed version names both halves (`+sol.<ours>.up.<base>`), and the
  build record lists the base, its distance past the pin and the carried
  commits. `h3-build` (8176242) is the retired tag-based line. A rebase of it
  was refused as a destructive rewrite of a shared clone, which is also why
  the branch moves by merge. 0.158.0.
- **Corrected: `vsa_attention.py` "cannot run", its defect "unreachable".**
  Stock core has built `to_gate_compress` since e308cc73 (#16072), so the old
  refusal passes and the `_publish_layout` leak is reachable. The node is
  parked by an explicit refusal (0.157.1). The remaining stale prose (the
  module docstring, `docs/research/vsa/vsa_node.md`, and
  `check_vsa_core_patch.py`'s "applied from the draft PR" message) is listed
  for correction.
- **Corrected: the VSA prose, following the entry above.**
  `vsa_attention.py`'s docstring, description and refusal said the node cannot
  run, core has no `gate_compress`, PR #15958 is an unmerged draft, the defect
  is unreachable, "nothing has been rendered", the T8 pack is "the only other
  implementation", and keep 10 is "VSA's published 0.90 sparsity". They now
  say it is parked, core builds the gate since e308cc73, it rendered on
  2026-08-30 on a draft-patched core, and FastH3 V2 trains at keep 20
  (`h3_config.FASTH3_CONTRACT_VSA`). `docs/research/vsa/vsa_node.md` said core
  was stock without the gate and the blocker stood until #15958 merged; it now
  says the same as the module. `bench/check_vsa_core_patch.py` said present
  support was "applied from PR #15958 ... DRAFT"; it now says stock core has
  it and that the check cannot tell stock from a local edit.
- **Corrected: `docs/SOLATTN.md` and `docs/wiki/next_steps.md` on Sol against
  core.** The upstream-policy table had the `dense_blocks` cells swapped
  (core's column held our `SOL_DENSE_TAIL`, ours said empty); the option table
  said the node and configs "ship empty as of 2026-09-02" and called
  `qk_balance` off and a declared deviation of the policy graph; the kernel
  table called `sol_attn_chunked` structurally unreachable. Now: core's
  default is empty and ours is the tail; empty was 2026-09-02 to 2026-09-25;
  `qk_balance` is off in the node and shipped on; `MiniMaxH3SolChunked`
  reaches the chunked kernel. Both pages said core's and our Sol outputs
  "differ by no more than the all-routed floor" and that ours as shipped "sits
  closer to exact" than core's defaults. The 2026-09-10 record
  (`bench/results/2026-09-10_sol_impl_capture_grade.json`, the `vs_ours`
  fields) has them above our floor in most cells and in every carried cell,
  and ours at that day's policy losing on block 0 and the per-row mean while
  winning the whole-tensor mean through block 49; that is what they say now.
- **Corrected: Sol code prose.** `sol_chunked_h3.py` and
  `bench/check_sol_chunked.py`'s docstring said the producer's routing
  threshold is stale; it is the K centring mean and the V scale (with
  kitchen's margin) that come from the previous step, and `kcvar` is current.
  `sol_attn_h3.py` said every local change since the fork was listed and
  named sage as the fallback; it now says only the fork-time changes are listed,
  and names `ModelAttentionBackend` (comfy kitchen attention) on the default
  graphs. `workflows/h3_config.py`'s `min_tokens` comment named sage as the
  fallback on every graph, and `SOL_CUDA_DEFAULTS` was called what the node
  "gives you untouched" while it pins `qk_balance` True; comments only, no
  value changed. `docs/sol_upstream.md` cited core's `model.py` a line early
  (:168-171, :623, :754; now :169-172, :624, :755) and now notes the V
  scale's clip margin.
## 2026-09-26

- **Distill research graphs moved to `workflows/distill_experiments/`**
  (owner, 2026-09-27; 0.157.0). 49 graphs, routed by
  `build_workflows._is_distill_experiment`; the everyday distill graphs stay
  at the root. The TaoMate graphs failed the live validation because their
  LoRA files are no longer on disk, so that build was written without
  validation; every other graph validated.
- **FastH3's pruned file does not share fl2va's curve basis** (measured,
  `bench/results/2026-09-26_fasth3_weights.md`). Its `adaln_t_table` and
  every `adaln_proj` differ from fl2va's by several times their own norm, so
  the conditioning moves between checkpoints only as a whole set
  (`bench/build_adaln_swap.py`). `h3_config.MODELS["unet_fasth3_v2"]`'s
  comment said it was "pruned int8 convrot on the fl2va curve basis".
- **The lightx2v turbo and Turbo-SLA LoRAs are retired, and turbo LoRAs are
  a closed lane** (owner: "lightx and turbo stuff irrelevant"; on SLA, "i
  thought we stopped that ages ago"). The distill lanes are FastH3, PDD and,
  undecided, FlashGen. Fourteen graphs went (0.156.0): `h3_text_to_video_turbo`,
  `h3_probe_turbo_768p_owner`, `_sla`, `_sla_dense`, `h3_probe_turbo_home_canvas`,
  `h3_probe_t2v_turbo_lx12_sage`, `h3_probe_split_base_first` and `_last`,
  `h3_probe_ref2v_turbo`, `h3_image_ref_plus_text_to_video_turbo_4step`, and
  the four `h3_probe_ref_turbo768p_*`. What prose claimed and no longer does:
  `h3_config.py`'s shift note carried the five-row lightx2v shift table and
  said the 768p students were "the trap" at 6/3; `docs/SOLATTN.md` counted two
  shipped graphs under the token floor; `docs/open_experiments.md` #20 said the
  two shipped SLA arms addressed its open half; `docs/h3_references.md` named
  `h3_probe_ref2v_turbo` as the one ref2va exception. The SLA probes had been
  generated since 2026-08-20 and never rendered. `check_distill_settings.py`
  now fails any graph loading a turbo LoRA, and its vendor-row grading left
  with the rows. The 2026-09-05 turbo rung closes unscored; its records and
  blind sessions stay. The model files are still on disk; deleting them is
  the owner's call.
- **Bug, fixed in 0.154.8: the exact branch stacked on another model's
  branch.** When one server process rendered Turbo, then PDD, then FlashGen,
  each wrapped the previous model's still-applied forward. The owner caught it
  by eye ("flashgen looks awful blocky"; "it's the code"). My first explanation,
  a regression between 13:05 and 16:07 read off a 16-pixel grid metric, was
  wrong: the owner named a clip it rated blocky as fine. The contaminated
  renders are listed in `bench/results/2026-09-26_followup_contamination.md`.
- **Owner: every LoRA on an int8 checkpoint goes through our exact branch**
  (0.154.0). This covers PDD's backbone and refiner (`MiniMaxH3PDDLoRA`
  `backbone_apply`), and the generator's `lora_branch` now defaults on.
  Before this, the branch was FlashGen's alone, PDD merged everything, and
  `unmerged_blocks` was an off-by-default knob that could not reach
  `mlp.fc2`. The measurement behind it:
  `bench/results/2026-09-26_int8_lora_requant.json`. Merges survive only on
  the two control probes. The policy lives in `docs/h3_quant_policy.md`.
- **Corrected: the audio-refine pass does keep frozen video exact.** A
  one-execution probe measured it
  (`bench/results/2026-09-26_frozen_row_probe.md`). The 2026-09-25 record
  said its refine arms reused pass 1 from cache, and the 46 dB gap was read
  against that; in fact pass 1 re-rendered and differed. Dated notes are in
  that record and in `docs/research/2026-09-26_distill_routing.md`. The open
  question is PDD8's run-to-run reproducibility. FlashGen reproduces bit for
  bit across servers.
- **Draft decodes for scouting are declined** (owner: "i dont think its worth
  it no matter what. thats probably why its always used as a preview node
  only"). `docs/open_experiments.md` #31 is closed as declined, and the draft
  graph `h3_text_to_video_flashgen_draft` is retired (0.153.1). Latent saving,
  the `draft_decode` switch and `bench/decode_draft_keepers.py` stay. The
  owner also set aside previews for now ("dont worry about adding preview
  stuff").
- **Every flagged prompt fixed, the bench prompt included** (owner: "Yes fix
  all"). `t2va_frontier_standoff` is the bench and baseline prompt, so the
  dense baseline graph's text changed (0.151.2). A baseline render before
  0.151.2 is not a same-prompt comparison with one after. The refview2 ablation
  prompts changed too, while their blind session is pending; the clips already
  rendered keep the old text.
- **The prompt bank is fixed for the subway-chase defect class** (owner: "we
  need to fix all our prompts"). 38 of 139 prompts got minimal edits, graded
  unchanged. The composed ref2va prompts, the refview2 ablation scenes (a blind
  session is pending) and anything whose fix changes what it tests are listed
  for the owner in `next_steps.md`, unedited. 0.151.1.
- **Corrected: the no-header-timestamps rule has been measured once.**
  `prompting.md` section 3.1 and the portable standard said it was "not yet
  measured in either direction". `bench/results/2026-09-18_timestamps_diner.md`
  had measured one same-seed pair on the rule's own day, and the owner
  preferred the timestamped take slightly. The owner keeps the house rule
  knowingly (asked 2026-09-26, answered "1"). A blind multi-scene test is the
  open alternative.
- **`t2va_subway_chase` rewritten** (owner, after the three-way distill look:
  "make sure its super clear and specific about what is going on in this
  scene").
  - **The scene now states its cast:** exactly two people, and the suspect
    never speaks.
  - **One physical path**, turnstiles then stairs down then the platform, where
    shot 2 had the agent on stairs and shot 3 looked up an escalator nobody was
    placed on. The owner saw an escalator chase in the wrong direction and a
    cloned pursuer.
  - **Every action has an agent** (`prompting.md` section 15.3, items 6 and 7).
  - It grades 0 FAIL and 0 WARN, as before. Every subway render before this
    entry used the old text, and each row records the prompt hash.
- **The shipped video VAE is the INT8 ConvRot build** (owner: "yes switch"),
  `h3_config.MODELS["video_vae"]`, 0.151.0. It reverses the 2026-08-21
  removal (`bc25d89`), and the closed-lanes row in `docs/roadmap.md` is gone.
  `VIDEO_VAE_FP16` keeps the fp16 name for the comparison tools.
  `MiniMaxH3VAEPrecision` now refuses to cast a quantized half. Prose that
  named the fp16 file as shipped carries dated notes in `next_steps.md` and
  `capture_manifest_schema.md`.
- **A distill runs on its trainer's contract, not a downstream template**
  (owner, approving the postmortem's proposal). The rule and its instance are
  in `references.md`, "A distill's reference is its trainer's contract", and
  CLAUDE.md points there.
- **The owner could not tell the INT8 ConvRot video VAE from fp16** on one
  345-frame latent, decoded both ways and watched sighted
  (`bench/results/2026-09-26_vae_decoders_345f.md`). The lane was reopened
  for measurement the same day. The shipped VAE is still fp16 until the owner
  decides to switch.
- **Corrected: where the server log lives.** `docs/comfy_notes.md` said
  `user/comfyui_<port>.log` always holds the current session. The server
  started 2026-09-26 by `start.sh` wrote only to its stdout pipe, and that
  file stopped at 2026-09-16. The paragraph now names `/internal/logs/raw`,
  core's in-memory buffer, with a dated note in place. The same claim was also
  in the restart recipe ("redirecting the launcher to `/dev/null` loses
  nothing"), now corrected with its own note.
- **Pipeline telemetry exists** (owner: "we should know whats happening
  throughout our pipeline in a structured schema and be able to record it end
  to end"). `H3_TELEMETRY` is `docs/pipeline_telemetry.md`. Two things said in
  session that day were wrong, and are corrected there rather than repeated:
  "Staged" is reserved address space on the card, not data held in RAM; and
  `--mmap-torch-files` does not affect `.safetensors`, which dynamic VRAM
  memory-maps anyway.
- **Corrected: FastH3 V2 was built to ComfyUI's template, not FastVideo's
  contract.**
  - `docs/wiki/references.md` said the FastH3 file was "Not on disk here, and
    no graph of ours loads it". A dated note there now says what does.
  - `docs/research/vsa/fastvideo_vsa_checkpoint.md` quoted Preview v1's 0.9
    sparsity with nothing saying V2 is 0.8.
  - The `h3_config.FASTH3_CORE_VSA` comment said the template "wins here".
    It now names the contract arms, `h3_config.FASTH3_CONTRACT_*` (0.148.0),
    and the record that compares them.
- **FlashGen ships as `h3_text_to_video_flashgen`**, with rank 64 applied at
  the call (owner: "lets take our best stab at making a good workflow ship",
  while unable to look). Chosen by source rather than by eye:
  - the int8 merge keeps little of FlashGen's delta
    (`bench/results/2026-09-26_int8_lora_requant.json`);
  - vllm-omni's native route applies the LoRA at run time.

  The render pair is unjudged (`bench/results/2026-09-26_flashgen_lora_path_s1.md`).
  If the owner prefers the merged take, the loader swaps back. 0.146.0.
- **The shipped refine graphs carry the frozen-video cache** (owner, on the
  unblinded pair: "Cant hear a difference whatsoever"). A practical default in
  a tinkering repo: it roughly halves the refine sampler
  (`bench/results/2026-09-25_frozen_cache_s1.md`). The PDD8 and FlashGen
  refine probes wire it. `h3_probe_t2v_flashgen_4step_audio_refine_uncached`
  replaces the `_cached` probe as its control. 0.144.0.

## 2026-09-25

- **Sol's default `dense_blocks` is `45,48,49`** (owner: "why wouldnt we set
  dense blocks to 45,48,49 by default?", then "do both"). A practical default
  in a tinkering repo, overriding `h3_config`'s earlier condition that the
  shared default stay empty until all 50 blocks are measured and a set
  validated; that condition is kept with a dated note. The kitchen-chain
  render of the tail is unscored. `h3_probe_t2v_no_dense_tail` is the control.
  0.142.0. *(2026-10-02, re-evaluated with real data under `quantizer="rotated"`:
  On an unmasked 50-block Ref2VA capture at sequence length 119,102, Hadamard
  rotation resolved the unrotated K-norm outlier problem, dropping block 48
  to 7.14% avg relative $L_2$ error [max 9.41%] and block 45 to 9.55%. In contrast,
  routing sparsity truncation error heavily concentrates in the middle semantic
  integration cluster: block 42 at 25.67% [max 27.01%], block 39 at 25.22%,
  block 41 at 24.35%, and block 40 at 23.43% [worst heads losing 57%–80% magnitude
  with min cosine down to 0.77]. Keeping block 48 dense while leaving 39–42 sparse
  wastes dense compute. Block 49 remains mandatory as it feeds `final_layer.video_out`
  [13.73% error]. Top candidates under active trial: Option A `"39,41,42,49"` (4 blocks)
  and Option B `"39,40,41,42,49"` (5 blocks). See 2026-10-02 entry and
  [`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)).*
- **`token_routing` reworked into one dropdown** (owner: "that token routing
  field UX is confusing"). `off` replaces "text field" as the default, and the
  list is read only under `custom`. The old default meant off by an empty list,
  which is the no-sentinel rule's shape. 0.142.0.
- **Reference-contract case 5c retired** (owner). It asserted that core's
  forced-CLIP-schedule branch drops `minimax_token_tags`: `comfy/sd.py` read
  only `o[:2]` there, so a graph wiring CLIP hooks would tag every row as
  text. Core `6bfaacc6` (#16400, 2026-09-20) merges `o[2]` on that branch too,
  the case went red, and its docstring said to retire rather than repair it.
  `comfyui_vendor_gaps.md` gap 2b, `custom_node_gaps.md` ("Core holds four")
  and `checks.md` described the gap as live and are corrected.
- **Continuation note corrected: LongMedia's temporal offset is partial.** It
  leaves a keyframe's audio and a reference video's frames unshifted, which
  desyncs both. The note had the first second-hand and missed the second
  (`../research/2026-09-25_temporal_offset_and_adaln_rounding.md`).
- **The FastH3 V2 symlink was repointed** from a relative target, which did not
  resolve inside `models/diffusion_models/`, to the file on the Storage share.
  The owner had placed it; the file is complete.
- **The stale Kijai PDD symlinks were removed** (owner). The four
  `MiniMax-H3-*-Acc-8Step*_comfy.safetensors` links in the ComfyUI checkout's
  `models/loras/h3/` pointed at the pre-2026-08-27 upload, which merged core
  decodes to a doubled head. No graph named them. The files are still in the
  Storage share.
- **Upstream reports are drafted, not posted** (owner: save them internally).
  The sglang PDD builder's fc1 swap and bf16 heads were confirmed by running
  sglang's own tools; the drafts are in the gitignored
  `internal/upstream_reports/`.
- **PDD reopened for research** (owner: "dig into the upstream stuff's PDD
  (not only comfy) to see if we can do better than comfy native AND our
  current"). `docs/roadmap.md` "Closed lanes" had PDD quality work parked
  since 2026-09-05. The reopen covers reading and CPU work; a render in this
  lane still waits on the owner's go. Also reopened for research: continuation
  by guide rows, the tile-seam question and the upstream digs (the upstream
  survey's buckets 3 and 5), framed as evidence for the owner's open decisions.
- **The PDD audio change-of-variable mechanism is refuted, and prose resting on it
  is corrected** (`../research/pdd/2026-09-25_upstream_pdd_comparison.md`; numbers in
  `bench/results/2026-09-25_upstream_pdd_comparison.md`, section 3; commit
  `aa9f2420`). Each has a dated note in place.
  - `audio_under_pdd.md` section 1 and "vary the TRANSFORM" said the block-start
    carry is an audio-only error growing with width. Core's carry equals the
    vendor's two-schedule audio Euler step exactly, so the loss is PDD's own.
  - `h3_pdd.md` "Why audio suffers more than video" derived the same mechanism.
  - `evidence.md`'s PDD bullet ended "vary the transform at fixed partition";
    withdrawn, and a do-not-rely row added.
  - `audio_carry_probe.py`, `bench/measure_pdd_audio_carry.py` and
    `bench/run_audio_carry_arms.py` described their premise as a correction.
    Their measured effects stand as departures from the exact path.
  - `pdd_math.py::fuse_heads` and `pdd_lora.py` said the vendor fuses in bf16;
    diffusers keeps the heads fp32.
  - `pdd_implementations.md` section 4.1 called the stale copies the vendor's;
    they are Kijai's conversions, re-uploaded upstream 2026-08-27T21:25Z. The
    "reported, not verified" notes in `2026-08-28_handoff.md` and
    `comfyui_h3_t2va_trace.md` are now verified.
- **Prose corrected by the 2026-09-25 upstream survey**
  ([`references.md`](references.md), "What moved by 2026-09-25";
  [`../sol_upstream.md`](../sol_upstream.md), "comfy-kitchen and core,
  2026-09-25"; [`../research/sglang_comparison.md`](../research/sglang_comparison.md),
  "Seventh read"). Each has a dated note in place. Commit `d226ec0f`, with
  its checks recorded in
  `bench/results/2026-09-25_upstream_survey_checks.md` in the follow-up.
  - "No engine implements PDD" (`references.md`, `pdd_implementations.md`
    section 1, the `index.md` row): sglang has implemented it since
    `973fb44471` (2026-09-23).
  - `h3_pdd.md` "Core is learning this" read Comfy-Org/ComfyUI#15908 as open.
    It merged 2026-08-29. The question it said would go live on merge, whether
    our node keeps the head half, has been live since then and is undecided.
  - `references.md`'s vllm-omni #7693 bullet named a function #7913 deleted,
    and called the rounding SM90-only; vllm-omni now keeps fp32 on every arch.
  - `h3_audio_freeze.md` section 3, "No shipped graph writes a `noise_mask`":
    stale since the freeze nodes shipped.
  - The rotation survey's E.5, "LightX2V refuses any feature cache on H3" and
    ships Sol and caching "never combined": `8652c6f1` ships DPCache with Sol.
  - `sglang_h3_pipeline.md`, FastH3 "Registered under `registry.py`": its
    config now registers from its own pipeline-config file.
  - `sol_upstream.md`, SubBlock's `sage_fp8` as SM90 only: now SM90 and SM120.

## 2026-09-23

- **Video files carry no metadata; the graph lives in the first-frame PNG**
  (owner's policy, "metadata saved in png files - that's it"; the VHS fork
  follows it). `loop_output.py` stopped embedding a `comment` tag and
  `creation_time`; 0.138.0. Prose that said the mp4 carries the graph was
  corrected in `loop_output.py`, `audio_freeze_song.py` (docstring and the
  `save_metadata_png` tooltip), `bench/diff_clip_graphs.py`,
  `bench/verify_vsa_render.py` and `docs/eval_comparison.md`; dated notes were
  added in place in `docs/h3_pdd.md` and `docs/research/vsa/vsa_node.md`,
  whose two-mechanism findings still describe every older clip.
- **Reference-video graphs load through VHS's ffmpeg loader** (owner's
  request, relayed by the dotfiles session). `h3_config.REF_VIDEO_LOADER`;
  0.137.0. The loaders keep different frames on 25 and 30 fps sources, so
  renders from before and after are not substrate-comparable; the record is
  `bench/results/2026-09-23_vhs_loader_comparison.json`. Prose that named
  `VHS_LoadVideo` as the reason for the choice ("the one that exposes
  `force_rate`": both do) was corrected in `build_workflows.py`,
  `docs/h3_references.md` and `docs/comfyui_vendor_gaps.md`.
- **The larryvrh turbo pack is gone** (owner removed it from the install and
  said to remove the graphs that use it). Three probe graphs, the generator's
  `turbo_pack` path and the `TURBO_PACK_*` constants; 0.137.0. The finding
  that lived beside those constants in `h3_config.py`: the v4 LoRA cannot be
  loaded by `LoraLoaderModelOnly` on our checkpoint, for two independent
  reasons. Its keys are bare where `comfy/lora.py` expects a
  `diffusion_model.` prefix, so nothing loads. And its 51 `adaln_proj` modules
  were trained full-width against a curve-form base whose adaln input is
  8 wide, so no loader can add them as a weight patch. That is why the pack
  shipped its own `silu(t_emb)` grid. The full comment is in git at
  `workflows/h3_config.py` as of 0.136.2. `docs/h3_ref2v_distillation.md`
  now says the ref2va-with-v4 probes were retired with the pack before any
  render of them was judged, so that question is unanswered, not closed.

## 2026-09-20

- **The sage fork's masked CUDA path is wrong, and this pack's refusal to use
  it is now a correctness guard rather than a scoping choice** (found while
  building a sage override for ComfyUI's Qwen-Image 2.1, which is the first
  target either repo has that passes a mask). `sageattn()` routes masked sm89
  calls to the fp8++ kernel's `MaskMode::kGeneral`, added in the fork's v0.5.5;
  that path applies `attn_mask` to the last 128 key columns only and silently
  ignores everything before it. It has never been correct -- not a regression.

  *(Updated later the same day: the kernel was fixed -- `apply_general_mask`
  sat at `apply_causal_mask`'s call sites, which is a diagonal-bound
  optimisation that does not generalise, so the main loop masked nothing. The
  unmasked and causal paths are bit-identical across the fix, so nothing below
  about H3 changes. This pack still declines masked calls, now because Triton
  skips fully masked K blocks and the CUDA kernel does not, making it faster
  and more accurate on masks rather than the only correct one.)*

  *(Corrected 2026-09-22: the reason in the note above named the wrong
  mechanism. The fork's own backlog entry "Route masked sm89 calls to Triton
  rather than fp8++" puts Triton's accuracy edge on PV quantized to fp16
  rather than fp8. It also finds Triton winning on a dense mask, where
  skipping blocks buys little, so the speed gap is not isolated. The
  conclusion, that Triton is the masked path to use and this pack keeps
  declining, is unchanged; `attention.py` now states the reason without
  claiming a mechanism. Also, "last 128 key columns" was a fixed-shape
  reading: the fork's `docs/cuda_mask_kernel_scoping.md` finds the window
  tile-aligned, the last two K blocks, so its width varies with kv_len.)*

  **H3 is unaffected, by construction rather than by luck.** Its single
  attention call site passes `mask=None` as a literal, not a variable
  (`comfy/ldm/minimax/model.py`, `Attention.forward`), so no code path reaches
  the defect. `denoise_mask` and `audio_denoise_mask` are per-row velocity and
  sigma scaling applied to the output; they never touch attention. **Multiple
  references lengthen the packed sequence and do not introduce a mask** --
  checked specifically, because it is the obvious place a mask would appear.
  Nothing rendered by this pack needs re-checking.

  What changed here: the comment in `attention.py` above the mask decline used
  to say "Sage has no mask support on this path", which is now false in detail
  and right in effect, and read as an invitation to relax the guard once sage
  gained masks. It now says why the guard stays. No behaviour change; the
  override already declined masked calls.

  The one correct masked entry point is `sageattn_qk_int8_pv_fp16_triton`. Note
  the naming trap: fp16 *triton* is correct, fp16 *cuda* silently drops masks
  whole, and the mode names do nothing to warn about it. If this pack ever
  grows a masked path, that is the only one to use.

  Lives in the sage fork: `docs/cuda_mask_kernel_scoping.md` (owner),
  `CHANGELOG.md` "Known kernel bugs" (record),
  `tests/repros/repro_fp8_mask_window.py` (the gate, exits non-zero while
  present), `tests/bench/masked_kernel_survey/` (every entry point against a
  causal, a prefix and a suffix mask).


- **The AWQ encoder comparison is about two badly executed artifacts, not about
  the method, and it must not be read as evidence against quantising our own
  encoder** (owner, 2026-09-20, in their words: "we poorly quantized that - it
  should NOT be proof of US quantizing being a bad thing... We control the
  calibration if quantize, and group size, and everything else! We just did it
  super poorly when we did it with AWQ, and then priorities shifted"). Every
  record of the four-encoder holdout stands as measured; what changes is the
  inference drawn from it. The lane closed because a poor first attempt
  coincided with attention moving elsewhere, not because the approach was shown
  to be unpromising, and nothing in `bench/results/` ever established the
  latter. **Calibration data is the untapped area**, and the part our own
  programme did worst; the recipe is where the effort went and not where it was
  lost. Annotated in place at `workflows/h3_config.py::ENCODER_INT8`,
  `docs/evidence.md`, `docs/roadmap.md` "Closed lanes", `docs/rules_history.md`,
  `bench/results/archive/v2_encoder/README.md`,
  `docs/research/archive/awq_quantization_suite/README.md` and its report, and inside
  `bench/results/2026-08-25_four_encoders_holdout_layer50.json` itself.
- **All four arms of that holdout hold the vision tower at BF16**, which the
  record did not say and a reader could not infer from it. Read from the file
  headers: `int8_convrot` and `nvfp4_awq` carry `.comfy_quant` markers on
  language-decoder linears only, and both W4A16 arms carry none at all. So the
  holdout compares decoder treatment and load path, and is **silent on tower
  precision**. A session reading it on 2026-09-20 concluded the opposite and
  had to retract. **The owner's position is that the vision tower needs to be
  BF16, for several reasons** (owner, 2026-09-20); that is a standing position,
  not a measurement, and no record here measures it.
- **The owner primarily runs the BF16 Qwen3-VL encoder**, although
  `h3_config.MODELS["clip"]` ships `ENCODER_INT8` on every graph (owner,
  2026-09-20). The shipped default is not the owner's working configuration, so
  a default here is not an endorsement: `ENCODER_INT8` won a holdout against
  two artifacts the owner calls badly executed.
- **`bench/check_model_contents.py` asserts what is inside every model
  `h3_config` names**, against a committed fingerprint baseline.
  `check_model_files.py` resolves names through `/object_info` and never opens
  a file, so a reship or a rebuild under an unchanged name was invisible to
  every check. `--report` prints the census, including matched-shape controls
  and their absence. The two text encoders are the only files here carrying no
  control, which is why neither can settle a role-versus-shape question about
  the tower.

- **The prompt rules of 0.128.0 are written down, and four of the six turn out
  to be the vendor guides' own** (the owner: "the rules should be what's in
  `vendor_guides/`"). `4bd7b429` fixed six wording classes across the bank and
  touched no rule-stating file, so the rules existed only as edited prompt text.
  Read back against the guides: **reusing `(Sx)` at every vocal event is STATED**
  by ref §5.4, as is the exemption that a verbal cue inside a reused soundtrack
  takes `<Audio N>` and no new id; **a subject's position in the frame is
  STATED** by base §4.1 and ref §5.3; a voice descriptor contradicting the
  identity is a reading of base §4.4's "stable identity" list. **base_en states
  neither reuse nor its absence**, so the same text can be conformant as a base
  prompt and not as a ref2va one (`docs/prompting.md` §12.13). Agentless actions
  and contradicted counts are in neither guide and stay HOUSE, tagged as such in
  §15.3 items 6 and 7.
- **`bench/preflight_graph.py::speaker_id_rules` is the check**, FAIL on ref2va and a
  note on the base modes, because that is where the guides put the rule. Shown
  red on the bank as it stood at `4bd7b429^` and green on the same entries now;
  the whole bank passes `build_prompt_bank.py --check` with it in.
- **Not decided, the owner's call.** 0.128.0 applied ref §5.4's reuse rule to
  base-mode prompts too and five base entries were not swept, so the bank does
  it in some base prompts and not others. The guide does not require it there.
  The note says which entries; sweeping them or leaving base alone is a
  decision, not a fix.
- **`docs/prompt_audit.md` is stale against the current bank text** and now says
  so in place. Its verdicts predate `705063a3` and `4bd7b429`; a green
  `check_prompt_docs_sync.py` does not speak to it, because that check tests
  that a verdict exists and not that it is current.

## 2026-09-19

- **`MiniMaxH3EncoderLoader` takes core's `device` input** (owner: whatever
  device the user puts the encoder on is respected). Same options, same
  `model_options` and optional as core's `CLIPLoader` has it
  (`h3_encoder_loader.DEVICES`); before, the node had no placement input and
  its rebuild factory would have dropped one. The node's description also
  told users to use "the AWQ loader instead" for a W4A16 file; that loader
  was deleted on 2026-09-13, and the sentence is gone.
- **Upstream survey: three pieces of prose had lost.** `docs/sol_upstream.md`
  described kitchen PR 171 (chunked `key_bias`) as an open draft with a rebase
  hazard for `blk_cnt`, core PR 16239 as an open PR that would rewrite core's
  Sol node, and kitchen PR 176 (W4A8 GEMV) as assessed and not carried. 171
  and 16239 were closed unmerged by their author on 2026-09-16; 176 merged
  into `v0.2.35`, which this install has carried since the 0.123.1 rebuild.
  Dated notes now stand beside each, and the new section "comfy-kitchen and
  core, 2026-09-19" holds the state.
- **`docs/open_experiments.md` #28 (the kitchen VAE kernels) said both PRs
  were open and the check waited on a tag.** Both halves merged (kitchen 167
  in `v0.2.34`, core 16187 on 2026-09-15) and have been in this install
  since its 2026-09-15 core pull; the item's 2026-09-19 status says
  so and notes that its before-and-after capture check has no record. No
  decision taken: reopening it is the owner's call.
- **`docs/research/sglang_comparison.md`'s Sol defaults table said our dense
  attention outside the Sol window is Sage.** It has been kitchen's dense
  attention since 2026-09-15 (`workflows/h3_config.py::DEFAULT_DENSE_CHAIN`);
  a dated note under the table says so. The same survey found sglang's new
  RTX 5090 recipe sets memory placement only, so the adopt-upstream rule has
  nothing new to act on (that page's "Sixth read").
- **`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` D.9
  credited DFSAttn with "layer-wise sparsity profiling + bidirectional
  co-clustering".** That mechanism is SVOO's (arXiv 2603.18636, whose v1
  title D.9 also listed separately); DFSAttn is arXiv 2605.23445 (Hilbert
  reordering, hierarchical block scoring, mask caching). Found independently
  by two research subagents (`docs/research/2026-09-19_token_selection_scorers.md`
  section 1, `docs/research/2026-09-19_video_sparse_attention_2026.md` "ID
  check"); corrected in place by the survey's owner in `15db381`.

## 2026-09-18

- **No timestamps on shot headers (the owner).** "The only time you should use
  timestamps in prompts is if you break stuff up mid-shot." Removed from the
  whole prompt bank, the shipped graphs, the prompting guide's examples, the
  portable system prompt and the prompt skill; `bench/preflight_graph.py`
  fails a stamped header. It departs from the vendor guides' stated format, on
  the owner's judgement that fixed cut times are "probably a big cause of
  issues"; one same-seed render with and without them is the first evidence
  either way (`Video/timestamps_test/`).
- **`morton` stays off by default, and the panel did NOT decide it** *(corrected 2026-09-19)*. Five valid on-length
  pairs scored blind, one seed each: plain order better on two, the `3d` reorder on one, two ties; the sixth pair
  (the kitchen scene) differed in chain and `qk_balance` as well as token order and is struck
  (`bench/results/2026-09-18_sol_reorder_panel.md`). A panel that split is not a panel that agrees; the reorder
  stays off because nothing argues for turning it on, and the short-clip panel, still unscored, is now the
  evidence that matters, since the one visible win for the reorder was on a short clip.
- **The noodle bar is retired at full length (the owner).** Its prompt is
  written for 107 frames and was rendered at 345, out of distribution; post
  office (141) likewise. A bank prompt is rendered at the length its bank
  entry declares, and `bench/run_graph_arms.py` now refuses anything else
  without a flag. The 2026-09-17 entry below stands as what was seen and is
  withdrawn as evidence about normal use
  (`bench/results/2026-09-18_off_length_prompts.md`).

## 2026-09-17

- **The owner scored every noodle bar stack: only Sol's `3d` token reorder is
  clean.** Every plain-order arm morphs around the four second mark (both
  dense chains, Sol `rotate`, sage balanced and rotated, all three
  token-routing presets, and the default rendered on 2026-09-15); the reorder
  at tau 1.0 looks best and at tau 1.3 second
  (`bench/results/2026-09-17_sol_options_noodlebar_batch.md`).
  - **What it decides.** Making the reorder work with ComfyUI's memory
    compiler is the next piece of work; the node refuses `morton` while the
    compiler is active (`CHANGELOG.md` 0.122.1).
  - **What it does not decide.** Any default. The scene is a stress scene: a
    discriminating one, and one scene at one seed. A default moves on a panel
    of scenes ([`../eval_comparison.md`](../eval_comparison.md), "A stress
    scene is not a typical scene").
  - **Corrected the same evening.** Several 2026-09-17 stacks had the
    2026-09-15 default on top as their reference; the day's runner scripts had
    passed the prompt with a trailing newline, so those rows are different
    samples. Recorded in both batch records; token routing showed no visible
    benefit and stays off.

- **`SageChainAssert` taken out of every generated workflow and the
  generator** (owner, 2026-09-17; `cfeeaa1`). The node stays registered, so
  saved graphs still load.
  - **Why.** On the default chain it could only confirm that no sage kernel
    ran. Its generation-time flags refused an intended editor swap to our
    sage node on `h3_candidate_t2v_pdd8_baked`.
  - **Lost with it.** The call-time probe on the sage arms, the "nothing
    patched" guard on the baselines, and the "[h3] sol window" log line.
    `bench/check_attention_defaults.py` still grades the wiring.
  - **Corrected.** Three docs still showed our sage node as the default dense
    node after the 2026-09-15 flip, which that change's prose sweep missed:
    - `docs/h3_geometry_and_nodes.md`'s chain block;
    - `docs/wiki/stages.md`'s attention row;
    - `docs/SOLATTN.md`'s Ordering diagram, which also showed the assert.

## 2026-09-15

- **Two bench manifests and one probe graph retired** (the owner deferred the
  call to the sage-fork session, which chose it; `531fbab`). Both would re-run as
  something other than their records: `bench/ladder_arms.json` (rendered
  2026-09-03, verdict `bench/results/2026-09-04_ladder_2026-09-03_verdict.json`)
  refuses on its sage-mode patch now that `h3_text_to_video` has no sage node,
  and `bench/sol_core_ab_arms.json` with `h3_probe_t2v_sol_core` (rendered
  2026-09-10, no verdict record) would set ours on kitchen against core's node
  on sage. The four manifests that pointed at the ladder for what each scene
  tests carry the descriptions inline. **Corrected:** `docs/wiki/next_steps.md`
  said the core-versus-ours A/B was "BUILT not rendered"; its 2026-09-10 records
  say it rendered.
- **Sage out of the default attention chain; Sol's `qk_balance` on** (owner,
  2026-09-15; `28d8ee5`). **Reverses the 2026-09-04 floor decision** ("sage
  always on is the floor", `docs/roadmap.md` forward plan 2026-09-04, quoted
  in `bench/sol_core_ab_arms.json` and `bench/sol_nosage_arms.json`, which stay
  as records of what those runs held). Every video graph that names no mode
  now carries core's `ModelAttentionBackend` at `h3_config.DENSE_BACKEND_NODE`
  under Sol, and `h3_config.SOL_RECOMMENDED_CUDA` carries `qk_balance=True`.
  Taken on capture grades and one render per arm, not a blind verdict: the
  community-chain arms were unscored when it landed
  (`bench/results/2026-09-15_block49_community_chain.md`,
  `bench/results/2026-09-15_ck_int8_attention_block49.json`). What stays on
  sage or stock attention, and why, is
  `bench/check_attention_defaults.py::FLOOR_STEMS`. `h3_probe_t2v_balanced`
  and `h3_probe_t2v_ck_balanced` retired. Prose that said the fallback under
  Sol is sage on every graph (`docs/SOLATTN.md` knob table and two dense-block
  sections, `docs/wiki/next_steps.md` "Now") carries a dated note in place.
  The policy page's table rows for this landed early, in `76b2378`.
- **`docs/research/2026-09-14_block49_quant_error.md` moved to
  `docs/h3_block49_quant_error.md` and revised** (owner, 2026-09-15): it is
  a reference page now, not a dated research note. Its "So what" was
  rewritten with the blast radius (a design property of INT8 attention
  meeting this model, affecting every H3 user on any quantized-attention
  kernel, not a bug in any kernel and not confined to these forks) and a
  numbered status: diagnosed and closed; two levers built, measured, both
  off; open items are the blind visibility check, the factor in Sol's
  quantizer, and sending the finding upstream. Every link was rewritten;
  `bench/build_wiki_index.py` reaches the new path.
- **Kitchen build moved to `0.2.34+sol.2aff3c5`.** ComfyUI's pin went to
  `comfy-kitchen==0.2.34`; `h3-build` was rebased onto the tag by the recipe
  in `vendor/rebuild_kernel.sh`, old tip archived as `archive/h3-build-0.2.33`,
  six `blk_cnt` commits reapplied clean. Brings the H3 VAE kernels (#167/#175),
  persistent RoPE allocations (#162), `compress-mode=size` (#165). Kijai's
  branches carried nothing unmerged on the CUDA side; open PRs assessed and
  not carried: `docs/sol_upstream.md`, 2026-09-15. `docs/SOLATTN.md` "Install
  the CUDA kernel" used to say main had moved past 0.2.33 with no tag.
- **`docs/h3_block49_quant_error.md` gained sections 7
  (the sage-side `qk_balance` kernel fix) and "So what".** No default changed.

## 2026-09-14

- **The generator runs masked, as documented, rather than needing the card**
  (owner, choosing the code fix over a doc line saying it needs the GPU). Prose
  that lost: `docs/checks.md` ("While a render is on the card") said the
  generator takes ComfyUI's CPU path when no device is visible, and the
  generator's docstring said nothing touches the GPU; a masked build raised in
  `resolution.py`'s core import before the CPU switch ran, at `1345791` and
  after. Fixed in the generator (`_core_cpu_when_no_card`) and in
  `bench/check_generator_constants.py`; `docs/checks.md` is true again and the
  docstring now says a visible card gets a CUDA context (CHANGELOG 0.108.1).
- **The frozen-audio loop simplified before extending** (owner, after the
  footguns were listed: "simplifying is the only way to avoid that"; the
  worry was a new place starting where a window ends, not where the song
  changes section). `prompt_mode`, random window lengths, `frames:` lines, the
  implicit wildcard fallback and Fill Prompt Lists' `count` went; the list
  node's `seed` became `shuffle`; the song node gained `timeline` and
  `preview`, planned by `loop_plan.py` (CHANGELOG 0.108.0). Prose that lost,
  and what it claimed: the shot-workflow bullet below said the song node's
  prompt blocks and `frames:` lines do what those workflows did (the timeline
  and `--- label` blocks do now); `prompt_lists.py` said a placeholder with no
  list node reads `name.txt` or `name.json` from the wildcards folder;
  `docs/comfy_notes.md` and `bench/check_node_ids.py` said an input may only
  ever be appended (removal is the owner's call now that shipped graphs are
  API format); `docs/h3_geometry_and_nodes.md` gave `MiniMaxH3ImageToVideo`
  as the t2v/i2v node without saying the pack wires `MiniMaxH3Conditioning`.
- **Shipped workflows are API format only** (owner: "If I dont need a layout,
  node titles, bypassed nodes (we dont even use those), or notes in workflows,
  why dont we just simplify all of them to API json workflows?"). The UI twins,
  `build_ui`, `UIGraph`, `validate_ui`, `cross_check`, the in-graph notes and
  `bench/check_workflow_schema.py` went; `validate_api` gained the value-type
  and DynamicCombo-member checks `validate_ui` had, and member order. The editor
  loads an API file by input name (the frontend's `loadApiJson`, read from the
  source of comfyui_frontend_package 1.52.7, not exercised). Prose that lost,
  and what it claimed: `UIGraph`'s docstring said the generator made no
  widget-to-input conversions (its Resolution and PDD step links were exactly
  that); `bench/smoke_h3.py` said Sol-Attn ships off; `bench/check_graph_discovery.py`
  said it had no exemptions (it has one); `docs/custom_node_gaps.md` item 3
  described a UI/API node mismatch that no longer exists. The deleted notes
  also carried stale claims (the Sol window at 0.2/0.9, a keyframe canvas node
  the graphs do not wire, sage-only twins of the Sol probes); they went with
  the notes. Facts that lived only in the notes moved to their docs, and the
  move corrected doc prose that had lost: `docs/SOLATTN.md` printed `[sol_attn]`
  log lines (the prefix is `[h3-sol]`); `docs/h3_geometry_and_nodes.md` told
  readers to wire `MiniMaxH3KeyframeCanvas` on every i2v and fl2v graph (the
  conditioner owns the canvas), showed a node order through the deleted
  `SolAttnPatch`, and said `MiniMaxH3SigmaShift` sits in every shipped graph;
  `docs/h3_references.md` said the concise swap twin stays shipped (retired
  2026-08-22); `docs/h3_ref2v_distillation.md` said no v4 ref2va graph existed;
  `docs/h3_image_editing.md` pointed at a note in a graph that now lives only
  under `archive/`. Plan: `internal/2026-09-14_api_only_workflows_plan.md`.
- **Two duplicate Sol-on probes retired** (owner: "retire the
  duplicative/redundant workflows"). `h3_probe_sol_on` and `h3_probe_sol_on_i2v`
  matched `h3_text_to_video` and `h3_first_frame_to_video` in everything but
  the output name once Sol went on by default. What prose claimed: the
  generator read `h3_probe_sol_on` "against h3_text_to_video.json, which is
  now sage-only", and `check_bench_matches_shipped.py` said the plain t2v graph
  omits Sol. This session's own note first counted `h3_probe_sol_on_all_refs`
  among the duplicates; its reference video and audio disprove that, and it
  stays.
- **Prompt lists serve every loop, not the song node** (owner: "I should be
  able to use this for any type of looping workflow we build here"; agreed
  the approach the same day). A node that loops inside itself fills through
  `prompt_lists.fill_windows`, takes `lists` and fingerprints the wildcard
  files it may read, and `bench/check_prompt_lists.py` fails one that does
  not. A graph that loops by chaining nodes, or one clip per queue, uses
  `MiniMaxH3FillPromptLists` by index; value N of a list depends only on the
  list and N, so both routes agree. The fallback stays: a placeholder with no
  list node reads a wildcard file of its name, shuffled at seed 0 (the
  option the session recommended over always raising, or a seed widget for
  it, and the owner agreed).
- **No live preview node in generated graphs** (owner: it wastes GPU).
  `ModelPreviewOverrideKJ` with taeh3 left every UI graph and the generator's
  `preview` path. What the prose used to claim: the generator's node note and
  graph comment called it "arguably the largest optimization here" and worth
  more than any kernel knob; `docs/h3_geometry_and_nodes.md` said the same.
  The owner also answered the API-only plan's open questions
  (`internal/2026-09-14_api_only_workflows_plan.md`, top): loop stage 2 goes
  before the conversion.
- **The frozen-audio loop plan, confirmed** (owner, in answers the same day).
  Encoding comes before sampling: the track and each distinct prompt once.
  Reference stills go with every window's prompt, with a workflow for it.
  The song node keeps its window files in a per-graph working folder
  (`<prefix>_windows/`, kept by default) and gains resume from the first
  changed window, with its seed held fixed between queues so a re-queue can
  reuse windows. Prompt lists fill `__name__` placeholders -- not `{name}`,
  which the frontend's dynamic prompts rewrite, and not `[name]` or `<name>`,
  which H3 prompts already use -- one list per node with its own seed held
  fixed, in order or reshuffled on each pass with no repeat before the list
  is used up, advancing only when a window uses the placeholder. Wildcard
  files live in `wildcards/` under ComfyUI's input directory (the
  `--input-directory` override included) and are selectable on the node.
  `docs/wiki/next_steps.md` carries the stages.
- **The shot-per-window workflows retired, unrendered** (owner: never used).
  `workflows/h3_text_to_video_audio_freeze_shots.json` and
  `..._shots_repeat.json` with their API twins, the generator's
  `freeze_shots` knob, and `MiniMaxH3JoinWindows` went. The song node's
  prompt blocks and `frames:` lines do what they did. The two-window seam
  graph (`workflows/h3_text_to_video_audio_freeze_2windows_api.json`) stays.
- **A loop feeds the encoder no per-window image** (owner, after discussion).
  A frame from the previous window would carry drift forward and force an
  encode between windows; the anchor for identity is a fixed reference still.
  `docs/h3_audio_freeze.md` step 5 said a first-frame keyframe was "needed
  before the loop, because the loop anchors each window on a frame"; the loop
  shipped anchored on the previous window's frozen latent tail. Corrected in
  place.
- **References work on the fl2va checkpoint** (owner: done often). Three
  places said or implied otherwise and are corrected:
  `MiniMaxH3ReferenceConditioning`'s description ("Use an H3 reference
  checkpoint"), `docs/h3_geometry_and_nodes.md`'s node table (fl2va "for
  t2v/i2v, ref2va for reference-to-video"), and `docs/h3_audio_freeze.md`
  step 6 ("the fix is the ref2va checkpoint with a per-window reference
  image").
- **The song node's seed has a control widget.** The generator's comment said
  "no control widget: the node's seed input declares none"; the frontend draws
  one by name for any INT called `seed` (`useIntWidget.ts`), so the shipped
  song UI graphs loaded shifted. The node declares it now
  (`audio_freeze_song.py`, the `seed` input).
- **The song node's docstring and graph note said its first run had not
  happened.** `bench/results/2026-09-12_audio_freeze_song_smoke.jsonl` records
  one. The docstring points there now.

## 2026-09-13

- **The AWQ lane's code is deleted** (owner). The lane closed on 2026-08-27
  (`docs/roadmap.md` "Closed lanes"); today `h3_awq_encoder.py` and its
  `MiniMaxH3AWQEncoderLoader`, the `config/` W4 snapshot directories,
  `docs/h3_awq_encoder.md` and the bench tools that served them
  (`build_h3_awq_standalone.py`, `check_h3_awq_encoder.py`,
  `convert_h3_awq_candidate.py`, `capture_h3_encoder_states.py`,
  `measure_qwen_view_under_snapshot.py`, `measure_still_policy_token_cost.py`,
  `build_native_h3_calibration_batch.py`) went with it, as did
  `h3_config.ENCODER_V1` and `ENCODER_V2`. The records under `bench/results/`
  and `docs/research/` stay as history. `h3_config.MODELS["clip"]` is
  `ENCODER_INT8`, and every generated graph loads it through
  `MiniMaxH3EncoderLoader` (`workflows/build_workflows.py`, the loader
  comment). What the prose used to claim: `docs/wiki/stages.md` named the
  AWQ loader as the alternate encoder load; `docs/custom_node_gaps.md` §5.1
  said every graph wires core's `CLIPLoader` and the adapter was live code
  read by preflight and the config; `docs/h3_geometry_and_nodes.md` called it
  "the loader used by every generated graph".
- **The `encoder` option is gone from `MiniMaxH3ReferenceConditioning`**
  (owner). `image_policy` and `video_policy` each offer `comfy` (default) and
  `release` (`reference_geometry.IMAGE_POLICIES`, the conditioner's
  `define_schema`). On the shipped core-loaded encoder `encoder` always
  resolved to `comfy`, since the CLIP carried no contract, and for stills
  `release` and `comfy` produce the same geometry at every legal short edge on
  that encoder (`bench/results/2026-08-29_qwen_view_under_snapshot.json`).
  `video_policy`'s default moved from `encoder` (which ran as `comfy`) to
  `comfy`; every graph was rebuilt. What the prose used to claim:
  `docs/comfyui_vendor_gaps.md` listed `video_policy=encoder` as the "shipped
  hybrid encoder policy" and `docs/h3_references.md` said "the encoder-aware
  hybrid is now the generated default"; both had been marked dormant on
  2026-08-29 and are now marked removed.
- **`MiniMaxH3AppendRefImage` defaults are vendor parity** (owner):
  `size_policy=max`, `dit_short_edge=2048`, `allow_upscale=True`,
  `qwen_view=shared`, which is what sglang, diffusers and DiffSynth do
  (`docs/research/sglang_h3_pipeline.md` "Reference stills"): one prepared
  still feeds both the video VAE and Qwen3-VL. Read the values from the node's
  `define_schema`. Before today `allow_upscale` was off (since 2026-08-28) and
  `qwen_view` was `separate` at 512 (since 2026-08-27, on one observation,
  CHANGELOG 0.82.0). `h3_rules.REF_QWEN_SHORT_EDGE` is now only the value
  pre-filled when a user picks `separate`; `REF_VIDEO_BUDGET` still sets
  `ref_upscale=False` on the video-bearing reference arms, as an arm setting
  for memory. What the prose used to claim: `docs/evidence.md` "Reference
  sizing" called the 512 view the shipped default; `docs/custom_node_gaps.md`
  §4.1 said "three independent implementations agree, and we differ";
  `docs/h3_conditioning_end_to_end.md` said most append nodes set
  `qwen_short_edge=512`.
- **The reference-view ablation is rebuilt** (owner). The three-arm Gate 6
  family (`h3_probe_refview_{a_source,b_qwen2048,c_parity}`,
  `bench/gate6_refview_arms.json`) was priced on 2026-08-25, never rendered,
  and is deleted. The new one is five ref2va scene graphs
  (`h3_config.REFVIEW2_SCENES`, `workflows/h3_probe_refview2_*.json`) built
  at the node defaults, with six arms per scene as widget patches in
  `bench/refview2_arms.json`. Unrendered; nothing is claimed.
- **`docs/h3_input_impacts.md` pointed at `preflight.py:28`** for the int32
  crossing; the line moved, and `preflight.py::_INT32_FUSED` is the pointer now.
  The same section gains the second ceiling, the CUDA v-side `uint32` wrap in
  the sage fork's `csrc/fused/fused.cu`, from the fork's own CHANGELOG.

## 2026-09-14

- **`dense_blocks="0-1"` decided against.** The roadmap's open call since
  2026-08-16. Two instruments that share no code agree the first blocks carry
  the least error on both terms; `dense_blocks` stays `""`. Reasoning in
  `docs/roadmap.md` under the original item and `docs/SOLATTN.md`, "The
  defaults, re-read against the sage-side error records, 2026-09-14".
- **`MiniMaxH3ChannelBalance` added, off by default, an experiment.** Folds a
  per-channel q/k rebalancing into the norm weights of the blocks whose
  K-norm is lopsided (45, 48, 49 on the shipped checkpoint). No workflow
  wires it; no default changed. Why the blocks are lopsided and what the fold
  buys: `docs/h3_block49_quant_error.md`.
- **Corrected: "K offset predicts `smooth_k`'s benefit."** The sage fork's
  real-activation spike printed that inference; graded across ten cells the
  benefit tracks block depth and error magnitude, not the offset, and the
  line is removed there. `smooth_k` stays off.

## 2026-09-12

- **PDD reopened for the audio-freeze lane only** (owner: "worth testing if
  PDD works with this since it's faster iteration"). The lane was parked on
  2026-09-05 for quality work; this is iteration speed, and the documented
  PDD weakness is its audio, which a frozen track takes out of PDD's hands.
  `workflows/h3_candidate_t2v_pdd8_baked_audio_freeze.json` is the graph;
  the parked quality work stays parked.
- **Audio-freeze lane opened** (owner): a known track frozen into the target
  audio rows with a per-stream mask, on the fl2va base first, the LTX pack's
  loop geometry on H3's grid after; reference audio parked until the loop
  runs because it regenerates the track. `docs/h3_audio_freeze.md` owns it.
  Before this, the mask path was known here only as something no shipped
  graph used (2026-08-30) and the looping packs were explicitly unread
  (`custom_node_gaps.md` section 8).

## 2026-09-11

- **The frontier table in `next_steps.md` counted a dense last step.** It said
  the leader runs five sage and eleven Sol of sixteen; since `07b903c` it runs
  four and twelve (reasoned, not rendered). A note above the table says so.
- **`docs/comfy_notes.md` said Sage runs `fp16 (most accurate)`, not `auto`.**
  The config has shipped `auto` since `497b421` (2026-08-18), which scoped the
  fp16 verdict to renders without Sol; the paragraph now says so.
- **The `shown red` column is gone from `docs/checks.md`** (owner). It
  recorded which checks had been shown failing under the retired
  red-before-green rule; git history has the cells.
- **`docs/prompting.md` 5.10 rewritten around the owner's point** (owner): too
  many words in a shot make the speaker rush. It carried a second word-budget
  formula beside §3.4's and blamed the rushed delivery on the Audio VAE; it now
  budgets from speaking time, points at §3.4, and claims no mechanism.
- **`CLAUDE.md` cut to what every session acts on** (owner). Seven "Settled
  about H3" facts, the numeric-input and capture-broadly rules, and the
  `coderef/` search advice moved to `docs/evidence.md`, `VISION.md` and
  [`references.md`](references.md); nothing was withdrawn. "A rendered clip
  cannot A/B a numerical change" stayed, because code and docs cite it as
  `CLAUDE.md`'s.
- **`bench/red/` removed** (owner). The red harnesses and their shared spine
  served only the retired red-before-green rule, and nothing imported or ran
  them; git history has them.
- **The `h3-experiment` skill removed** (owner). Its two steps nothing else
  held are in `docs/comfy_notes.md` "Adding a probe or an arm".
- **Documented commands run the ComfyUI venv's python** (owner). They said
  `uv run --active --no-sync python`; the venv's own interpreter is exactly
  what the server runs and involves no uv project step.
- **Run bench scripts with `python`, not `uv run`** (owner). `docs/eval_comparison.md`
  and the `h3-prompt` skill said `uv run python bench/...`; plain `uv run` in
  this repo builds a repo-local venv and, with `VIRTUAL_ENV` set, can recreate
  the ComfyUI one, as `docs/comfy_notes.md` records.
- **`CLAUDE.md` routes to the wiki instead of carrying its tables** (owner;
  CHANGELOG 0.99.72). The "What is where" tables moved into
  [`index.md`](index.md), which is now written by hand.
  `bench/build_wiki_index.py` used to generate that page from `CLAUDE.md`; it
  now only reports documents no link reaches.
- **History notes leave `CLAUDE.md`** (owner; CHANGELOG 0.99.72). Its rule
  "when you find prose that lost, correct it and say what it used to claim"
  now logs the old claim on this page.
- **"A perceptual claim needs a distribution of seeds judged blind, never a
  pair" is withdrawn** (owner; CHANGELOG 0.99.72) from `VISION.md`,
  `CLAUDE.md`, the `h3-experiment` skill, `docs/open_experiments.md` and
  [`docs/prompting.md`](../prompting.md), under the tinkering-repo rule.
  `docs/eval_comparison.md` still describes the blind process for when one
  is wanted.
- **Where the Sol-Attn kernel comes from.** `CLAUDE.md` said it was
  "installed from comfy-kitchen main". It is built from the owner's fork by
  `vendor/rebuild_kernel.sh`: the tag ComfyUI pins plus the `blk_cnt`
  commits, enforced since `d378479`.
- **The tinkering-repo rule** (owner). `CLAUDE.md` opens with it: not
  research-grade, rigour proportional to the claim, and a default that
  sglang and ComfyUI's own node or comfy-kitchen agree on is adopted without
  waiting on our evals.
- **No check has to be shown red before it is trusted** (owner, `48e1219`).
  `docs/checks.md` "The standard" says what it used to require.
- **Closed lanes moved into the repo** (`48e1219`). They were recorded only
  in agent memory; `docs/roadmap.md` "Closed lanes" is their home.
- **The probe canvas** (`48e1219`). The `h3-experiment` skill said to bench
  canvases cheaper than 16:9 by default; probes run at 1152x768 or 1344x768
  with 345 frames, the owner's rule of 2026-08-30.
- **Sol runs through the last step** (owner, `07b903c`). `end_percent` is 1.0
  on every graph, adopting sglang's and core's default. It used to stop short
  so the last step ran dense; the retired values are in the comments beside
  `workflows/h3_config.py::SOL_END_PERCENT_BY_STEPS` and `SOL_PDD_OVERRIDES`.
