# 2026-09-26 follow-up batch: contaminated rows

`data/results/2026-09-26_followup.jsonl`, first launch, on 0.154.6/0.154.7.
Before 0.154.8, `lora_branch.install` wrapped whichever forward was applied
(CHANGELOG 0.154.8). The batch ran Turbo, then PDD (exact branch), then
FlashGen in one process, so:

| row | ran as | use |
|---|---|---|
| smoke_turbo_branch | base + Turbo (first branch model; clean) | valid |
| warmup_pdd8 | base + Turbo + PDD | contaminated |
| look_anchor__pdd8, look_noir__pdd8 | base + Turbo + PDD | contaminated |
| warmup_flashgen | base + Turbo + PDD + FlashGen | contaminated |
| look_anchor__flashgen, look_noir__flashgen | base + Turbo + PDD + FlashGen | contaminated |
| warmup_fasth3, look_anchor__fasth3 | FastH3's own checkpoint, a different model object | valid |

The batch was stopped at `look_noir__fasth3`. The contaminated clips and
latents were moved out of the analysis paths. The rerun on 0.154.8 appends
to the same file, with the same labels, after a server restart. Where a label
appears twice, read the later row.

The owner's reads of the valid clips, 2026-09-26: FastH3's look_anchor is
"super high detail like almost way too much causing it to look a bit ai
generated in polish".

**Beyond this batch** (the VAE session, 2026-09-26):
- `h3_probe_t2v_step_switch_flashgen_pdd8` chains FlashGen's branch and PDD
  in one graph, which is exactly this shape. A render of it counts only on
  0.154.8 or later. None had rendered by the time of the fix.
- Before 0.154.0, PDD merged its backbone rather than patching forwards, so
  no earlier graph stacked two branches.
- The 2026-09-26 cross-process FlashGen reproducibility pair ran the same
  load sequence in both processes, a FlashGen-only warmup then the render.
  It stands as "the same sequence reproduces" and says nothing about another
  load order.
