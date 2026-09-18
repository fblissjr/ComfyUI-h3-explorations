# Two clips the owner reported as bad on 2026-09-17, and what is known about each

Recorded 2026-09-18 at the owner's request, so the report is in this repo and
not only in a conversation. Both clips were queued from the ComfyUI interface
by the owner on the afternoon of 2026-09-17; both carry their full prompt and
editor workflow in the mp4 `comment` tag and the png, which is where every
setting below was read from. Neither has been re-rendered or A/B'd. What is
observation and what is inference is marked.

## What the owner said

- `Video/h3_probe_t2v_sage_rotate_00002-audio.mp4` (13:52): "looked like shit
  even tho it was 16 steps and no pdd".
- `Video/agent_00001-audio.mp4` (14:14): "even worse"; the next morning, "just
  really grainy / digital artifacted".
- First suspicion: something changed that afternoon in the sage fork, the
  kitchen fork or this pack.

## What was ruled out

Code. Renders on this stack repeat bit for bit, and the default graph, the
graph with Sol `rotate`, and the shipped `h3_probe_t2v_sage_rotate` graph all
rendered identically on 2026-09-15 and on 2026-09-17, the last of them on the
final stack of that day (`2026-09-17_sol_options_noodlebar_batch.md`, closing
checks). So neither clip is bad because of a change in any of the three
repositories. Both differ from every earlier clip in their INPUTS.

## `agent_00001`: a lattice artifact, and a workflow that is off its recipe

Observed (stills at 2, 6 and 10 s, and a native-resolution crop): a regular
lattice or mosaic pattern over the whole frame, clearest on flat sky and on the
tree silhouette, at roughly the pitch of the patch grid. It is a structured
pattern, not film grain and not the morphing seen on the noodle bar.

Settings read from the clip, against what this pack ships:

| setting | in the clip | what the pack ships for that case |
|---|---|---|
| checkpoint | `minimax_h3_fl2va_pruned_int8_convrot_pdd8_baked_s1`, the PDD 8-step BAKED checkpoint | the baked checkpoint belongs to the PDD graphs, at 8 steps with the PDD recipe (`docs/h3_pdd.md`) |
| steps / scheduler | 18, `simple` | 8 for the baked checkpoint; 16 for the base model |
| sigma shift | no `MiniMaxH3SigmaShift` node in the graph | 12.0 video / 3.0 audio on every shipped graph |
| length | 362 frames | 345 is the legal ceiling at this canvas; 362 is past the reference's 15.0 s check (`docs/h3_geometry_and_nodes.md`) |
| seed | 492696341373225 (random) | fixed seeds in every record here |
| attention | sage `fp8++ balanced` + `MiniMaxH3ChannelBalance` fold + Sol with `qk_balance` and `rotate` on, base-model Sol recipe | the PDD graphs carry their own Sol recipe (`h3_config.SOL_PDD_CUDA`) |
| encoder | `qwen3vl_32b_minimax_h3_bf16_pruned` | either |

Inference, NOT tested: a step-distilled checkpoint run for more than twice its
step count on a base schedule with no sigma shift is the first suspect for a
lattice pattern, and the over-length clip is the second; attention is a distant
third, because the same attention chain on the base checkpoint (the next clip)
shows no lattice. The test is one render: the same prompt and seed on
`h3_candidate_t2v_pdd8_baked` as shipped, 345 frames.

## `h3_probe_t2v_sage_rotate_00002`: the shipped graph with a different prompt

It is `workflows/h3_probe_t2v_sage_rotate_api.json` exactly as shipped (base
checkpoint, 16 steps, shift 12/3, 345 frames, seed 730451892, sage
`fp8++ balanced` + the fold node + Sol `qk_balance` and `rotate`), with the
prompt replaced by a BBC natural-history scene (meerkats, golden hour). The
same graph with its own market prompt is `..._00001` from 2026-09-15, which the
owner said "looks great", and which re-rendered bit-identically on 2026-09-17.

Observed in stills at 2, 7 and 12 s: clean, no lattice. So whatever the owner
saw is in motion (morphing, flicker) or in texture at full resolution, and it
comes with the prompt, not with the graph or the stack. Not investigated
further. Fine natural texture (fur, grass, backlit whiskers) is where an
approximate attention shows first, and this chain stacks three balancers of
which one, the weights fold, is measured redundant under sage's balanced mode
and re-rounds q/k in bf16 (`2026-09-17_channel_balance_vs_sage_balanced_b49_s15.json`).
The test is the same prompt and seed on the default chain, plain order against
the `3d` reorder, and once fully dense.

## Status

Open. Two renders would settle the first clip and three the second; none has
been run. Related: every plain-token-order clip of the noodle bar morphs and
only the `3d` reorder is clean (`2026-09-17_sol_options_noodlebar_batch.md`),
which is a different defect from the lattice above.
