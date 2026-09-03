# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`../roadmap.md`](../roadmap.md), the roadmap is right and this
page is stale.

## Now

**Render the speedup ladder and look at it.** `bench/ladder_arms.json`: three
bank scenes chosen for their failure surfaces, each from the true baseline
(stock attention, no LoRA) up through sage alone, sage plus Sol as shipped,
sage fp16 plus Sol, and shipped PDD8, matched seed, trained canvas. Then
`bench/blind_batch.py`, the scoring app, `bench/score_session.py`, and the
owner's free-text notes per pair. This is the first perceptual anchor taken
from the true baseline, and everything numeric gets judged against it.

**Validate the online Sol-versus-Sage instrument.** `sol_block_probe.py` is
built and its fixture controls are green (`bench/check_sol_probe.py
--controls`). Two things remain before its numbers are trusted: the capture
replay (`--replay-capture`, against the exact-branch record) and the first
canonical Base16 record (the shipped t2v graph, Sol on, `dense_blocks`
empty, armed with `H3_SOL_PROBE` and `H3_SOL_OBSERVE`), read with
`--record`. Both need the card and are queued behind the ladder.

**Render the reference pathway ablation once the card is free.**
`bench/ref_pathway_arms.json`: our reference conditioner and core's, each
with and without the VAEs wired, plus fl2va under encoder-only stills.
Open experiment 26 in [`../open_experiments.md`](../open_experiments.md)
says what would count as an answer. Needs a server restarted onto the
current core and this pack.

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
