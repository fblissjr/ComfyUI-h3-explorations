# Sage chain against kitchen chain, same prompt and seed, 2026-09-18

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
| kitchen | the expediter wears a white shirt and black apron; line cook at about 4 s is a woman | the expediter wears a black shirt; line cook at about 4 s is a woman | "not sure i see anything else but my eyes are tired." A wardrobe difference the prompt leaves open; the character it does specify (a woman, soprano) is right on both. No defect on either chain |
| diner | | | not yet scored |

## What it says so far

One scene, no defect on either chain, the visible difference is unscripted
wardrobe: the same kind of take-to-take variation the reorder panel showed. It
agrees with the 2026-09-17 verdict. Note for the reorder panel: on this scene
the `3d` arm rendered the line cook as a man; plain order on BOTH chains renders
her as a woman, so that miss belongs to that take of the reorder, not to a
chain.
