# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`../roadmap.md`](../roadmap.md), the roadmap is right and this
page is stale.

## Now

**Read the ladder verdict, then widen the seed count where it separated.**
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
on how many scenes each rung sits below dense. Next, in order: PDD8's own
baseline pair on the same scenes and seed, PDD8 under stock attention and
PDD8 under sage alone, which the generator already knows how to build
(`workflows/build_workflows.py`, the `dense_attn` extra: `True` wires
neither kernel, `"sage"` wires sage with Sol absent), so the existing pdd8
clips become the third arm of a three-rung PDD ladder; then the same scenes
at more seeds (the 2026-09-03 session postmortem's forward item 4). The
blind page for this session
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
says what it ranks and what it cannot say). Next in the roadmap's order: a
second scene on the same footing, then legal PDD8 through the node's own
sigmas, then Ref2VA, each its own population. Arm with `H3_SOL_PROBE` and
`H3_SOL_OBSERVE` on a restart you own; timings from an armed server are
void.

**Export the reference pathway scores, then join them.** All five arms of
`bench/ref_pathway_arms.json` rendered once on 2026-09-03
(`bench/results/2026-09-03_ref_pathway_arms.jsonl`): our reference
conditioner and core's, each with and without the VAEs wired, plus fl2va
under encoder-only stills. Blinded 2026-09-04 as session
`ref_pathway_2026-09-03` with the manifest's four controlled contests. The
owner scored the page that day, but no `scores_ref_pathway_2026-09-03.json`
reached its batch folder on the share, so nothing has been joined and the
key stays sealed. The page keeps its answers in the browser that scored it:
reopen it there, press Export scores, put the file beside the batch's
MANIFEST, then `bench/score_session.py` with the key. Open experiment 26 in
[`../open_experiments.md`](../open_experiments.md) says what would count as
an answer; timing is already in the record.

**Small fixes the ladder exposed, before the next one.** The blind tool's
default output root is the local output directory, not the share; pass
`--output-root` until `bench/_paths.py::comfy_output` reads the launcher's
directory. The audio-spectrum grader warns instead of refusing below three
clips per arm, so its verdicts on a one-seed ladder are void; make it refuse.
The fp16-sage rung is slower than Sol as shipped on every scene, and a
per-step timing would say where. Each is a checkable item in the session
postmortem under `internal/postmortems/` dated 2026-09-03.

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
