# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`../roadmap.md`](../roadmap.md), the roadmap is right and this
page is stale.

## Now

**Score the speedup ladder.** Rendered 2026-09-03: five bank scenes chosen
for their failure surfaces, each from the true baseline (stock attention, no
LoRA) up through sage alone, sage plus Sol as shipped, sage fp16 plus Sol, and
shipped PDD8, matched seed, trained canvas
(`bench/results/2026-09-03_ladder_outputs.json` maps arm to clip and holds
the timings). Blinded as session `ladder_2026-09-03`, every rung against its
scene's dense baseline plus Sol against the fp16 rung. What remains is the
owner's part: the scoring app in that batch folder, then
`bench/score_session.py`, free-text notes per pair. This is the first
perceptual anchor taken from the true baseline, and everything numeric gets
judged against it. One seed per arm, so it anchors; it does not decide.

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

**Render the reference pathway ablation once the card is free.**
`bench/ref_pathway_arms.json`: our reference conditioner and core's, each
with and without the VAEs wired, plus fl2va under encoder-only stills.
Open experiment 26 in [`../open_experiments.md`](../open_experiments.md)
says what would count as an answer. Needs a server restarted onto the
current core and this pack.

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
