# Sage chain against kitchen chain, same prompt and seed, 2026-09-18

> **2026-09-19, read first: the kitchen-scene pair here was NOT sage against kitchen.** Both clips are the sage chain: `Video/block49_kitchen/default_s730451892_*` is sage `auto` with Sol `qk_balance` off (rendered 2026-09-15 before the default chain moved), against sage `fp8++ balanced` with Sol `qk_balance` on. So the owner's one audio preference ("just more natural") was for balanced sage plus balanced Sol over plain sage, not for a chain, and the mechanism sketched below does not apply to it. The DINER pair is a true chain pair (checked with `bench/diff_clip_graphs.py`) and is unscored. Measured audio differences between the chains on the true pairs are within the floor on every metric: `2026-09-19_audio_sage_vs_kitchen.md`.

Model: MiniMax H3, base 16 steps, 345 frames at the trained canvas, plain token
order, seed 730451892. Kitchen chain (the default): core's attention backend on
kitchen int8 dense for the steps below Sol's window, Sol with `qk_balance`
after. Sage chain: sage `fp8++ balanced` for those dense steps, the same Sol
settings (`workflows/build_workflows.py --chain sage`). The kitchen-chain clips
are the 2026-09-15 plain clips the owner scored in the reorder panel; the
sage-chain clips were rendered today from the prompt bytes embedded in those
clips (identical within each pair, checked from the metadata). Timing rows:
`2026-09-18_sage_chain_panel_arms.jsonl`. Blind stacks and the prompts:
`Video/sage_chain_panel/stacks/`.

Why: everything scored on 2026-09-18 ran on the kitchen chain, and the last
sage-against-kitchen look was the owner's six-scene verdict of 2026-09-17
("very hard to tell the difference").

## The owner's verdicts, blind, keys opened after

| scene | kitchen chain | sage chain | notes |
|---|---|---|---|
| kitchen | the expediter wears a white shirt and black apron; line cook at about 4 s is a woman | the expediter wears a black shirt; line cook at about 4 s is a woman | "not sure i see anything else but my eyes are tired." A wardrobe difference the prompt leaves open; the character it does specify (a woman, soprano) is right on both. No visual defect on either chain. **Audio, listened to afterwards on the singles: the sage-chain clip is better, "just more natural".** |
| diner | | | not yet scored |

## What it says so far

**Audio is the one place the owner preferred a chain here**, and it went to
sage. One scene, one listen, so it is a lead and not a finding. It is a
plausible one: the chains differ only on the dense steps (the first few of
sixteen, below Sol's window), and on those steps the dense kernel computes
EVERY row, the audio rows included, while inside Sol's window the audio rows
are attended exactly on both chains. So whatever the two dense kernels do
differently to the audio tokens happens early, where the coarse structure of
the sound is laid down. Untested. The cheapest test needs no GPU: the six
2026-09-15 pairs (`Video/block49_repro/*_default_*-audio.mp4` against
`*_sagelevers_*-audio.mp4`) were scored by eye only; listening to them, blind,
asks the same question on six scenes.


One scene, no defect on either chain, the visible difference is unscripted
wardrobe: the same kind of take-to-take variation the reorder panel showed. It
agrees with the 2026-09-17 verdict. Note for the reorder panel: on this scene
the `3d` arm rendered the line cook as a man; plain order on BOTH chains renders
her as a woman, so that miss belongs to that take of the reorder, not to a
chain.
