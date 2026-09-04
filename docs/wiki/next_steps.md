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
says what it ranks and what it cannot say). Next in the roadmap's order: a
second scene on the same footing, then legal PDD8 through the node's own
sigmas, then Ref2VA, each its own population. Arm with `H3_SOL_PROBE` and
`H3_SOL_OBSERVE` on a restart you own; timings from an armed server are
void.

**Render the PDD ladder, then blind it against the ladder's dense clips.**
Built 2026-09-04 at the owner's ask (PDD8 without Sol, and PDD8 with Sol at
more conservative settings): `bench/pdd_ladder_arms.json`, four PDD8 rungs on
the five ladder scenes at the ladder's seed. PDD8 under sage alone and PDD8
under stock attention are new probe graphs (`h3_probe_t2v_pdd8_sage`,
`h3_probe_t2v_pdd8_dense`, generated, never edited); the shipped PDD8 graph
renders twice, once as shipped and once with Sol's window narrowed to two of
the eight steps, which is reasoned from `docs/SOLATTN.md`'s sigma-window
arithmetic rather than measured. The manifest names the controlled pairs,
the reading rules, the reference rows to append from the ladder's JSONL, and
the one caveat: the ladder's dense clips predate the 2026-09-04 ComfyUI pull,
so pairs against them are cross-regime until a re-rendered dense rung is
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

**Two checks the 2026-09-04 upstream survey earned, owner-approved, sequenced
behind whatever the scoring lane has on the card.** First, grade kijai's
`token_aug` (Comfy-Org/comfy-kitchen PR 156) offline on the retained Base16
capture with `bench/measure_sol_exact_variants.py --capture`, the footing PR
150 was graded on, from a wheel built into a scratch target rather than the
venv; his claimed defect is a brightness pulse, which is a thing a blind pair
can see, and our `blk_cnt` commits conflict on his branch, so the offline
grade comes before any rebase. Second, re-time one ladder rung on the
post-2026-09-04 core: that pull turned on the Comfy Compiler's malloc graph
around the H3 forward (`comfy/model_prefetch.py::malloc_graph_enabled`, on
whenever aimdo is), so every timing after it is a different regime from the
09-03 record and one rung says how different. `docs/sol_upstream.md` holds
the survey's kitchen state; `docs/research/sglang_comparison.md` the rest.

**Small fixes the ladder exposed, before the next one.** The blind tool's
default output root is the local output directory, not the share; pass
`--output-root` until `bench/_paths.py::comfy_output` reads the launcher's
directory. The fp16-sage rung is slower than Sol as shipped on every scene,
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
