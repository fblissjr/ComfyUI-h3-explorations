# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`../roadmap.md`](../roadmap.md), the roadmap is right and this
page is stale.

## To-do (the owner's action list, 2026-09-05)

**Pointers only, one line each, consolidated across the sessions working this
tree.** Each names the record that closes it. The reasoning lives where the
pointer goes; this list is stale the moment the roadmap disagrees with it.
Items from the `mrpink` helper lane first; `evalman`'s render lane appends its
own below the rule.

**2026-09-12:**

- Audio-freeze lane: [`../h3_audio_freeze.md`](../h3_audio_freeze.md)
  section 4 is the plan, rewritten after the first verdicts. Step 1 built
  and controlled; step 2's mechanism question closed on the owner's word for
  both scenes (dancer follows the beat, speaker lip-syncs;
  `bench/results/2026-09-12_audio_freeze_step2_verdict.json`). Next: PDD8
  as the iteration chain (`bench/audio_freeze_pdd_arms.json`), the two open
  pairs (loose mask at a second seed, transcript or not), first-frame, then
  the loop.

**2026-09-11:**

- Graphs regenerated on the rebuilt venv and validated against a live server;
  `bench/check_attention_defaults.py` passes, with `end_percent` 1.0 on every
  Sol node except `h3_candidate_t2v_pdd8_sol_narrow`'s.
- `bench/check_node_ids.py` fails on `MiniMaxH3SolAttn` since `f980aeb`
  (2026-09-08) because it compares declaration order. The server's
  `input_order` keeps the required inputs in the manifest's order and adds
  `token_aug_blocks` as the only optional input after them. Fix the check to
  compare required then optional, then record the append. *2026-09-12: the
  append is recorded (`bench/node_id_manifest.json`, regenerated with
  `--write` when the freeze node was added) and the check passes against
  it; whether the check should also tolerate a required-then-optional
  reorder is still open.*

**2026-09-10 (upstream survey session):**

- Core vs ours Sol A/B, BUILT not rendered: `bench/sol_core_ab_arms.json`
  (courtroom, disco, hacker; two seeds; pairs A ours vs core, B ours at core's
  values vs core, C ours vs all-rows). Graph `h3_probe_t2v_sol_core` (API only).
  Blocked until the ComfyUI venv rebuild (`docs/comfy_notes.md`, the
  `start.sh` note) is verified and the server restarted.
- All-rows sink switch: planned, gated on pair C (`docs/roadmap.md`, forward
  plan step 3).
- Pair B, GRADED on activations
  (`bench/results/2026-09-10_sol_impl_capture_grade.json`, capture
  `2026-09-10_sol_impl_courtroom`, its repo records beside it): at matched
  knobs core's chunked producer and our node differ by no more than the
  kernel's own all-routed floor in every cell, carried statistics are inert,
  and the rebuilt Q/K/V equal the capture. So pair B's rendered pairs can only
  show sample divergence. The same record has ours as shipped closer to exact
  than core's defaults; pair A's scores say whether that is visible. The text
  segment is the worst segment in every arm, which is the all-rows argument
  again.
- A/B scoring: the owner scores `sol_core_ab_2026-09-10`; then
  `bench/score_session.py`, the loudness record
  (`bench/measure_clip_loudness.py --outputs
  bench/results/2026-09-10_sol_core_ab_outputs.json --baseline-rung ours
  --out bench/results/2026-09-10_sol_core_ab_audio_loudness.json`),
  the frontier table, and the all-rows decision from pair C. Whoever writes
  the verdict up says two things `score_session.py` has no field for: pair
  B was settled on activations before scoring, so its verdicts read as
  sample divergence; and every row's code is "server started 15:55, node code
  `8870f8e`", because the rows stamp the repo HEAD at render time, not the
  code the server loaded.
- Run rows need a code stamp from the server. The rows
  `bench/run_graph_arms.py` writes carry `substrate.git_commit` and
  `git_dirty`, read from the working tree when each row is written, so a
  commit during a batch relabels every later row. In
  `bench/results/2026-09-10_sol_core_ab_arms.jsonl` the stamp moves from
  `3e553f6` to `311cffa` mid-batch, while the server ran `8870f8e`
  throughout. Done when rows also carry the serving process's start time,
  or the commit at its start. Wrong premise if the owner rules that no
  commit is made during a batch. Until then, verdict write-ups state the
  server's start time and loaded code by hand.
- The Base16 capture passed its `keep_until` on 2026-09-10 and the recycler
  lists it recyclable; deleting it is the owner's call.
- `bench/check_dit_prefix_attention.py` is indexed as CPU-only but cannot run
  with CUDA masked: it imports `comfy_extras.nodes_minimax_h3`, whose import
  chain reaches `comfy.model_management`, which asks CUDA for a device at
  import. Run it with the card free, or give it ComfyUI's `--cpu` the way
  `bench/audit_ref_audio_crop.py` does.
- Keyframe-encode determinism: `bench/measure_keyframe_encode_determinism.py`,
  written, not yet run (open experiment 30).
- Reference audio now end-padded to the audio VAE's hop in our path
  (`reference_conditioning.py::_encode_ref_audio_aligned`), closing gap 16 for
  this pack; core's own nodes still crop until Comfy-Org/ComfyUI#15972 lands.
- A doc pointer drifted with today's sglang pull and fails
  `bench/check_doc_links.py`:
  `docs/research/qwen3-vl-special-tokens-post-training/brainstorming/claude-encoder/2026-08-25-serving-stacks-survey.md:53`.

- **Owner decisions 2026-09-05 evening** (`docs/roadmap.md`, the decisions
  block): composition counts as prompt adherence; PDD parked; the turbo
  rung goes with two LoRAs at two seeds (bake session owns it, plus a sage
  floor render at the second seed); token routing stage one goes (survey
  session offline at the time, pick it up from the plan page).
- The bake pair: SCORED and joined 2026-09-05, pairs only (`bench/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`, read its
  `join_notes_2026-09-05`). The baked arm lost to the sage floor on four
  scenes and read same on one, on the same properties the merged arm lost
  on the day before (audio level, lighting, framing, the subway sign
  text); merged against baked read same on two, merged won two (standoff, subway,
  the latter after the owner corrected a mis-clicked verdict) and baked won
  diner on composition. Reading: at
  this seed the PDD8 defects are the schedule's, not the merge's; the
  bake is a hygiene and cost change and stays the PDD8 checkpoint of
  choice for that reason alone. Next lever for PDD quality is the
  partition (the tail-weighted six-evaluation schedule the node already
  accepts, `h3_config.PDD_MANUAL_SIGMAS`), not the weights. Records beside it:
  `bench/results/2026-09-05_pdd_bake_outputs.json`,
  `..._pdd_bake_audio_loudness.json` (absolute levels; the merged twins'
  are in the 2026-09-04 ladder record) and
  `..._pdd_bake_2026-09-05_frontier.json`, where the baked arm's sampler
  time sits beside the merged arm's, as predicted.
- Scoring page rubric (`bench/rubrics/default.json`): the pair tab's per-half
  quick tags reuse the singles' option list, so "same" and "can't tell"
  appear as tags on one half of a pair, where they mean nothing (the owner
  asked what they meant, 2026-09-05). Give the pair tab its own tag list
  (good / not good / off per half) before the next session is built; the
  three built pages are unaffected and the join ignores those two tags on
  a half.
- PDD ladder: SCORED and joined 2026-09-05, pairs only, singles unscored by
  the owner's choice (`bench/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`). Reading, one seed, five scenes: the sage
  16-step floor beat PDD8 under sage alone on every scene and beat PDD8 with
  the narrow Sol window on four with one can't-tell, so the PDD8 loss is
  PDD's own, not Sol's; Sol on two of eight steps read *same* as sage alone
  on all five; the shipped four-step window lost to sage alone on standoff
  (background definition) and subway (scrambled first-frame text,
  artifacts) and read *same* on three. The owner heard the floor louder than
  PDD on every scene compared; `bench/results/2026-09-04_pdd_ladder_audio_loudness.json`
  is the level record. Merge against schedule is what the bake pair decides.
- Frontier table on that verdict: DONE 2026-09-05,
  `bench/results/2026-09-05_pdd_ladder_2026-09-04_frontier.json`, speed against
  the sage floor beside the owner's note per pair (the tool now skips
  warmup rows, which shared their arm's label).
- The chain assert's call-time proof on the no-sage graph: the exercise
  skips for headroom on this card with the DiT staged, so the Sol-over-stock
  state is proved at registration only. Size the exercise to the probes the
  state actually fires (`assert_chain.py::_exercise`); a branch, after the
  card is down. Closes when a just-Sol render's log carries the "no sage:
  probes ... reached no sage kernel" line. Item 7 of the 2026-09-04
  postmortem under `internal/postmortems/`.
- The fp16-sage rung's lost time (`bench/results/2026-09-03_ladder_outputs.json`
  shows it slower than Sol as shipped on every scene): moot since the
  2026-09-04 plan dropped the rung (annotated in the 2026-09-03 postmortem
  under `internal/postmortems/` on 2026-09-05); reopens only if the rung
  returns.

---

Items from the `evalman` render lane:

- Just-Sol session: SCORED and joined 2026-09-05, pairs only (`bench/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`, read its
  `join_notes_2026-09-05`). The sage floor beat just-Sol on four scenes with
  one can't-tell; Sol as shipped beat it on three and read same on two; the
  tau-raised arm lost both its subway pairs. Nearly every win is a
  composition difference (sample divergence), the one quality loss is
  stairwell (lighting, skin, level), so the direction is against running
  Sol without sage and the evidence is weak. The shipped chain (sage
  outer steps, Sol inside) stays the leader. Frontier table: UNBLOCKED and
  written 2026-09-08,
  `bench/results/2026-09-08_sol_nosage_2026-09-04_frontier.json` (the session's
  own outputs record plus the ladder's, which carries the sage floor rows).
  The split now takes the FIRST underscore, so a rung may carry one, with
  `--scene` to declare a scene name that does; `bench/frontier_table.py
  --controls` covers the underscored rung, the declared scene and the
  unchanged plain labels. `bench/measure_clip_loudness.py` took the same fix,
  prophylactically: its 2026-09-04 record never included the tau-raised arm,
  so nothing measured was wrong -- but the frontier table now carries a row
  whose loudness has no entry. Closes when that arm has a loudness row or the
  record says why it is absent.
- One pair, no card time: the all-rows sink mode's armed subway clip
  against the 2026-09-03 subway Sol clip
  (`bench/results/2026-09-04_probe_allrows_vs_shipped_pixels.json` says the
  change is the size of dense-versus-sage;
  `bench/results/2026-09-04_probe_render_vs_unarmed_pixels.json` says an
  armed clip is a valid sample). Closes with a scored one-pair session.
- An armed render of the just-Sol graph under `H3_SOL_PROBE`, so the
  probe's counterfactual is stock attention and the record ranks Sol
  against near-exact attention directly (roadmap 2026-09-04, step 1's next
  footing). Closes with a probe record and its comparison against
  `bench/results/2026-09-04_sol_probe_base16_subway.json`.
- Then the vendor's start-schedule arm, open experiment 27 in
  `../open_experiments.md` (`upman`'s lane), in the step-policy slot: probe
  first, then one blind pair.
- The PDD bake's hypothesis, written 2026-09-05 in the roadmap's bake
  paragraph: the defects named blind on the shipped PDD8 rung are the kind
  a requantised merge produces; the baked checkpoint on the same scenes at
  the same seed is the test. The plan, owner-approved 2026-09-05 and under
  the render lane's review, is
  [`../research/pdd/2026-09-05_bake_plan.md`](../research/pdd/2026-09-05_bake_plan.md);
  closes with the baked pair on disk, the contract check green on it, and
  the merged-versus-baked pair blinded.
- Unscheduled: sage at more steps against dense at sixteen, open
  experiment 6 in `../open_experiments.md`, the base-model question the
  owner raised.

Items from the PDD backbone bake lane (the alibaba-pai PDD LoRA folded into the fl2va int8 checkpoint; the peer session that built it happens to share a name with the larryvrh turbo LoRA, which is a different artifact and not part of this bake):

- **The turbo rung is RENDERED and BLINDED** (owner decision 3 in
  `../roadmap.md`; run records `bench/results/2026-09-05_turbo_rung_{s1,s2,floor_s2}.jsonl`,
  judged JSONLs beside them, every arm present, night of 2026-09-05). The
  owner scores `turbo_rung_s1_2026-09-05` and `turbo_rung_s2_2026-09-05`
  under the output folder's blind directory, 15 pairs each, pairs only.
  Then `bench/score_session.py` on each export, graded against the
  manifest's `predictions` block, and the frontier table per seed
  (`bench/frontier_table.py` with the run's outputs record; build that with
  `bench/build_outputs_record.py` against the server that rendered, which
  is the one up now). Closes with the two verdict records and a frontier
  row per arm.

- The bake script's open review candidates: a control-before-bake gate,
  the contract check's retyped filenames now that `h3_config` names the
  bake and the stripped sidecar, the dirty-tree marker in the bake's
  metadata, and the duplicated helpers. The list with a state per row is
  in the session's internal notes (2026-09-05, bake code review) and
  forward item 4 of the postmortem under `internal/postmortems/`. Closes
  when each row is fixed or dismissed with a reason in `CHANGELOG.md`.
- The stripped sidecar into the HF sidecar staging tree beside the four
  full ones; the checkpoint's publication is the owner's separate call.
  Closes when the bundle notice names it.
- The ref2va bake: `bench/bake_pdd_checkpoint.py` pointed at the other
  partition, after the fl2va pair is judged. Closes with the pair on disk
  and the three verifications green.

## Candidates on trial (2026-09-05): what to render your own prompts on

The owner asked for canonical graphs carrying the lane's current best
guess. Generated by `workflows/build_workflows.py` like every graph here
(never hand-edited; the Sol fields that differ from the recipe are declared
in `bench/check_attention_defaults.py::DEVIATIONS`, which grades each
deviation as real). Open the UI graph, replace the prompt widget, render.
Rows are ordered fastest first by their step mix; the per-step costs that
turn a mix into a time are the medians in
`bench/results/2026-09-03_ladder_outputs.json`. Nothing below is a verdict,
and the last column says why it is on the list and what would make it one.

**The step mixes below predate `07b903c` (2026-09-11)**, which moved
`end_percent` to 1.0 and removed the dense last step. The leader and the
all-rows candidate now run four sage and twelve Sol of sixteen, reasoned from
the Sol window's sigma band at shift 12 rather than re-rendered; the PDD rows
changed too and are not recounted here.

| graph | what it is | step mix (of 16, or of 8 under PDD) | why it is here, and what settles it |
|---|---|---|---|
| `workflows/h3_candidate_t2v_pdd8_sol_narrow.json` | PDD8 with Sol on two of the eight steps | six sage, two Sol | the shipped PDD8 rung lost every scene blind and carried Sol on four steps; this asks whether Sol on the coarse schedule is what lost. Blinded as `pdd_ladder_2026-09-04`; unjudged |
| `workflows/h3_candidate_t2v_pdd8_baked.json` | the shipped PDD8 graph on the baked checkpoint with the stripped sidecar | four sage, four Sol | same work as the shipped PDD8 graph with the merge noise gone: the PDD backbone is quantised once with the weights instead of merged and requantised at load (`docs/research/pdd/2026-09-05_bake_plan.md`). Settled by the merged-versus-baked pair, `pdd_bake_2026-09-05`, rendered under sage alone so Sol is not in it |
| `workflows/h3_probe_t2v_pdd8_sage.json` | PDD8 under sage alone, no Sol | eight sage | the PDD floor under the sage decision; if it reads the same as the sage floor, PDD8 is a near-doubling over it. Blinded as `pdd_ladder_2026-09-04`; unjudged |
| `workflows/h3_candidate_t2v_sol_only.json` | just Sol, from the first step to the last-but-one, no sage node | one stock, fifteen Sol | the fastest base-model arm: the probe records put Sol's disagreement lowest early in the window, so the dense warm-up may be unneeded, and the owner asked what Sol does without sage under it. Blinded as `sol_nosage_2026-09-04`; unjudged |
| `workflows/h3_text_to_video.json` | **the leader**: sage auto plus Sol as shipped | five sage, eleven Sol | the only arm with a blind verdict behind it: same as dense on four scenes of five at one seed, and the fifth scene's slip belonged to sage (`bench/results/2026-09-04_ladder_2026-09-03_frontier.json`, `docs/roadmap.md` step 1) |
| `workflows/h3_candidate_t2v_sol_allrows.json` | the leader's chain with the text rows dense too | five sage, eleven Sol, plus a few hundred dense rows | free on the probe (the text segment's disagreement collapses to the audio floor) and the pixels move as far as dense-versus-sage; one blind pair away, already rendered |

Not on the list: the shipped PDD8 graph (`workflows/h3_text_to_video_pdd.json`),
which lost to the true baseline on every scene of the 2026-09-03 ladder; it
stays the reference the two PDD candidates are judged against. PDD
parameters on every PDD graph are the vendor's contract as
`workflows/h3_config.py` states them (the fl2va eight-step LoRA at its
shipped strength, heads on, the sampler's own step count); the baked
candidate above changes the weights those parameters act on, not the
parameters. The chunked Sol node is not wired into any of these:
it is a memory lever (`sol_chunked_h3.py`), and at the trained canvas and
length every arm above rendered on this card without it.

## Now

**The routing policy is the lane, and sage is the floor.** Two owner
decisions on 2026-09-04, recorded with what they change and the sequence
they set in [`../roadmap.md`](../roadmap.md) (forward plan 2026-09-04): the
next render on the card is the online probe on the subway scene, the one
scene where Sol lost blind, set beside the standoff record; then the PDD
ladder below; then the three policy axes the node already exposes, segment,
block and step, each measured on the probe before one blind pair.

**Two sessions wait on the owner's eyes, and one pair costs no card time.**
`pdd_ladder_2026-09-04` (PDD8 under sage alone and with Sol on two of eight
steps, against the ladder's sage and shipped-PDD8 clips) and, once its
renders land, the just-Sol session (`bench/sol_nosage_arms.json`, against the
ladder's sage and Sol clips). A third, one pair on the subway scene, needs no
render: the all-rows sink mode's armed clip against the 2026-09-03 Sol clip,
since arming the probe does not perturb a render
(`bench/results/2026-09-04_probe_render_vs_unarmed_pixels.json`). After the
just-Sol renders, an ARMED render of the just-Sol graph is the probe's next
footing: with no sage, its counterfactual is stock attention, so the record
measures Sol against near-exact attention directly rather than against sage.

**Read the ladder verdict, as a frontier.**
`bench/results/2026-09-04_ladder_2026-09-03_frontier.json`
(`bench/frontier_table.py`) puts each arm's speed against the sage floor
beside the owner's own words from every pair it sat in; that is the form
every ladder is read in from now on, speed beside what was noticed, never
pass or fail.
The owner scored every stacked pair of session `ladder_2026-09-03` on
2026-09-04 (five bank scenes by five rungs, matched seed, trained canvas;
`bench/results/2026-09-03_ladder_outputs.json` maps arm to clip and holds
the timings). `bench/results/2026-09-04_ladder_2026-09-03_verdict.json`
holds the tally per contest, the owner's note on each pair, and which arm
sat in each slot, joined with the sealed key after the key's rows and labels
were checked against the render record. Two reading rules the record cannot
enforce: within a scene every contest shares one clip per rung, so the
verdicts in a scene are one sample of each arm judged more than once, not
independent votes; and no single was scored, so nothing audible was judged
and the manifest's audio questions (the diner's speech against its sound
bed, the opera's sung line) stand open. The direction the record gives: the
PDD8 rung lost to the dense baseline on every scene, on a named defect each
time, and the owner named it as PDD unprompted on two pairs; on the four
scenes without fast motion, sage alone and Sol as shipped were called the
same as dense or could not be told from it; on the subway chase every rung
lost to dense, narrowly for sage and Sol in the owner's words; the fp16-sage
rung went both ways. One seed anchors; it does not decide. **The PDD8 rung
is confounded, as the owner pointed out on reading the result:** every rung
above dense carries sage, the sol, solfp16 and pdd8 rungs all carry Sol, and
pdd8 also carries the PDD-specific Sol `end_percent`
(`h3_config.sol_for_graph`). No arm renders PDD8 without Sol or without
sage, so a pdd8 loss does not attribute to the step distillation, the
kernels, or their interaction. The audio half is the owner's report in
conversation, recorded with its provenance in the verdict record's
`owner_report_after_join`: good across the board, a bit quieter on the PDD
clips; `bench/results/2026-09-04_ladder_audio_loudness.json`
(`bench/measure_clip_loudness.py`, EBU R128 per clip) is the level beside
that sentence, descriptive at one clip per arm, and its sign summary says
on how many scenes each rung sits below dense. The owner does not want
more seeds; the question is what runs faster at quality the owner cannot
tell from dense, and the ladder's answer so far is sage alone and Sol as
shipped on four scenes of five. The blind page for this session
on the share was edited by hand after it was built (its pair questions
export under `clip_rubric`); `bench/score_session.py` reads that key and
records which it read. Regenerate a page rather than editing it.

**The online Sol-versus-Sage instrument is validated; read it, then widen
it.** `sol_block_probe.py` passed its fixture controls, reproduced the
offline record on every retained Base16 cell
(`bench/results/2026-09-03_probe_replay_base16.txt`), and its first canonical
record holds every invariant
(`bench/results/2026-09-03_sol_probe_base16_standoff.json`, summary as data;
the closing section of
[`../research/2026-09-03_sol_exact_pquant_and_base_capture.md`](../research/2026-09-03_sol_exact_pquant_and_base_capture.md)
says what it ranks and what it cannot say). Next, first on the card under
the 2026-09-04 plan: the subway scene on the same footing, then legal PDD8
through the node's own sigmas, then Ref2VA, each its own population. Arm
with `H3_SOL_PROBE` on a restart you own; timings from an armed server are
void, and the 2026-09-03 probe render's row shows the render itself takes
about three times an unarmed Sol render.

**Render the PDD ladder, then blind it against the ladder's sage clips.**
Built 2026-09-04 at the owner's ask (PDD8 without Sol, and PDD8 with Sol at
more conservative settings) and cut to three rungs under the sage-always
decision: `bench/pdd_ladder_arms.json`, on the five ladder scenes at the
ladder's seed. PDD8 under sage alone is a new probe graph
(`h3_probe_t2v_pdd8_sage`, generated, never edited); the shipped PDD8 graph
renders twice, once as shipped and once with Sol's window narrowed to two of
the eight steps, which is reasoned from `docs/SOLATTN.md`'s sigma-window
arithmetic rather than measured. The manifest names the controlled pairs,
the reading rules, the sage rows to append from the ladder's JSONL, and the
one caveat: the ladder's clips predate the 2026-09-04 ComfyUI pull, so pairs
against them are cross-regime until a rung re-rendered after the pull is
shown pixel-identical to its 2026-09-03 clip. The probe graphs were built
with `--no-validate` on a stopped server; validate against the live
`/object_info` before the first render. Score the singles this time.

**The reference pathway verdict is in; read it before touching a reference
graph's cost.** Scored and joined 2026-09-04
(`bench/results/2026-09-04_ref_pathway_2026-09-03_verdict.json`, one seed,
pairs only): the owner could not tell both pathways from encoder only on
either conditioner, named none of the identity items as differing, and
named one signature on every encoder-only arm, static people in the
background. Open experiment 26 in
[`../open_experiments.md`](../open_experiments.md) carries the reading and
what would close it; it leans to the cheaper arm and is not closed. No
action is queued on it; a second seed is the manifest's `run` line if the
owner wants one.

**Block 49's token-routing instability: the variation has the selection unit's
shape, and the kernel source names a mechanism.** Two records, read in order:
`bench/results/2026-09-08_token_aug_determinism_shape.json` establishes that
the variation is not accumulation order and eliminates the two obvious causes
(centroid fidelity, refuted by our own morton measurement; the high-norm-row
hotspot, refuted by measurement the same day), and
`bench/results/2026-09-08_token_aug_selection_structure.json`
(`bench/probe_token_aug_selection_structure.py`) says the moving rows carry the
granularity of one token-routing centroid, against a scattered control the run
measures on its own rows and a shifted-grid control for the alignment. All five
captured blocks have now been tested at every captured step and 49 is the only
unstable one, at all of them; every budget behaves the same way. Each control
cell carries its own plain arm, so a deterministic block is one where the lever
engaged and stayed stable rather than one where it admitted nothing.
**Its predecessor's "unobservable from outside the kernel" is half wrong and
open experiment 29 carries the correction**: the kernel's source says the
selection boundary is a histogram bin edge, not a rank cut, and that a token
clearing it takes a slot by `atomicAdd` and is dropped if the slot exceeds the
budget -- so the docstring's "the set never depends on scheduling" holds only
while the count clearing the threshold fits. Read entry 29 before proposing a
cause. **It is upstream's defect, measured and not merely argued**: it reproduces on a
plain upstream wheel holding none of our code, with a positive control on each
arm that it is the wheel it claims to be
(`bench/results/2026-09-08_token_aug_stock_wheel_control.json`). A standalone
repro and the smallest input that still moves are in
`bench/repro_token_aug_nondeterminism.py` and
`bench/results/2026-09-08_token_aug_repro_shapes.json`; reproduction is
non-monotone in head count and sequence length, so **a negative on one shape is
not evidence of correctness** and candidates must be tested in fresh processes.
**The cause is measured** (arm 5, which needed no kernel change after all --
the workspace and the count's offset are both reachable from Python): the
admitted token count exceeds the budget by orders of magnitude on a few hundred
centroid groups, the same groups every launch, and every group whose output
moves is one that overflowed
(`bench/results/2026-09-08_token_aug_admitted_count.json`). Above the budget the
kernel's slot assignment is decided by atomic arrival order. Note the first
proposed fix is retracted in the entry: the overshoot is far too large for the
boundary-rounding explanation.

**How much of an H3 render is attention: a floor, derived rather than
profiled.** `bench/derive_attention_share.py` writes
`bench/results/2026-09-08_attention_share_bound.json` from the 2026-09-03
ladder's paired arms. Read it before ranking any attention work, because it
carries two different floors and they answer different questions: the dense
one is what "attention dominates H3" means, and the sage one is the
denominator an Amdahl argument actually needs, since it is the configuration
that ships. The record states its own limits, including that both are floors
and neither bounds the share from above.

**Token routing: stages 1, 2 and 3 all done 2026-09-08.** The kernel is rebased onto v0.2.33 and installed
(`0.2.33+sol.990ae4c`), and `token_aug_blocks` exists on `MiniMaxH3SolAttn`
as a per-block knob that ships off in every graph. Neither touched the card.
Stage 2 reproduced: every aggregate not involving `token_aug` is bit-identical
to kijai's graded build, so the rebase changed no arithmetic. It also found
that the plan's stop condition was unachievable as written, because the
`token_aug` arms are nondeterministic run to run, and that the nondeterminism
sits on exactly the one block where the lever hurts. Both in
[`../research/2026-09-05_token_aug_plan.md`](../research/2026-09-05_token_aug_plan.md)
and its record. What has NOT happened is any render with it on; stage 4 is
the expensive one and needs the owner's go.

**Both upstream-survey checks landed 2026-09-04.** Token routing
(Comfy-Org/comfy-kitchen PR 156) graded on the Base16 cells:
[`../research/2026-09-04_sol_token_aug_grade.md`](../research/2026-09-04_sol_token_aug_grade.md)
owns it; better on four blocks, worse on block 49 at every step, budget
nearly inert, so it is a per-block candidate for the block-policy step and
not the frontier move roadmap step 7 asked about; nothing installed. The
post-pull core regime (the Comfy Compiler's malloc graph, on whenever aimdo
is) was checked on one dense rung
(`bench/results/2026-09-04_stairwell_dense_retime.jsonl`): sampler time
inside the 2026-09-03 rows and the clip pixel-identical to the 2026-09-03
one by the scoring lane's comparison, so the 09-03 clips remain valid blind
references. `docs/sol_upstream.md` holds the kitchen state;
`docs/research/sglang_comparison.md` the rest of the survey.

**Small fixes the ladder exposed, before the next one.** The blind tool's
default output root landed 2026-09-04: `bench/_paths.py::comfy_output` reads
the launcher's `--output-directory` from the live server's command line and
refuses with no server and no `H3_COMFY_OUTPUT`, rather than answering with
the local directory. The fp16-sage rung is slower than Sol as shipped on every scene,
and a per-step timing would say where. Each is a checkable item in the
session postmortem under `internal/postmortems/` dated 2026-09-03. The
audio-spectrum grader's refusal below three clips per arm landed 2026-09-04
(changelog 0.99.40).

## Then, in the roadmap's order

The forward plan in [`../roadmap.md`](../roadmap.md) (section "Current
forward plan — 2026-09-03") owns the sequence, the decision standard, and
what would count as finding it. The PDD bake is a parallel lane there, not a
step behind the Sol work.

## Where the reasoning lives

- What was measured and what was withdrawn:
  [`../research/2026-09-03_sol_exact_pquant_and_base_capture.md`](../research/2026-09-03_sol_exact_pquant_and_base_capture.md)
  and [`../evidence.md`](../evidence.md).
- The goal in the owner's words: [`../../VISION.md`](../../VISION.md).
- How a rendered comparison is judged: [`../eval_comparison.md`](../eval_comparison.md).

## Updating this page

When a step above is done, replace it here and in the roadmap in the same
commit; a done step left standing is the drift this repo names most often.
