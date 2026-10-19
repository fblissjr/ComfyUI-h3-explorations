# H3: the model and its ComfyUI path

Overview pages first, then the detail each links down to.

## Start here

| page | what it is for |
|---|---|
| [`uses.md`](uses.md) | **can we do X?** Everything that can be made here, one row per use: what H3 does for it, the nodes and graph, its state, the evidence |
| [`stages.md`](stages.md) | one row per stage of a render: our code, the page that owns it, the check that guards it, the implementation to compare against |
| [`nodes.md`](nodes.md) | which nodes a render uses, which to skip, and the order on the model branch |

## Inputs and what they cost

| page | what it is for |
|---|---|
| [`geometry.md`](geometry.md) | canvas, frame grid and length: what to type, the legal set and why, `MAX_LENGTH` and "longest is not best", the audio clock, a canvas from a keyframe |
| [`geometry_cost.md`](geometry_cost.md) | the tables: every canvas, every length, the Sol-Attn token floor, the int32 crossing, block geometry by canvas |
| [`references.md`](references.md) | every reference type, what ComfyUI does to it, what it costs, the label rules, the known limits |
| [`reference_sizing.md`](reference_sizing.md) | a still's two copies and their knobs, the vendor's image path stage by stage, the encoder's pixel bounds, the video and audio policies |
| [`reference_prompts.md`](reference_prompts.md) | a label is a bare ordinal; how to prompt each relationship (edit, swap, continue, motion, structure, voice, music); the shipped reference graphs |
| [`conditioning.md`](conditioning.md) | every conditioning input end to end: what the encoder sees, what the VAE does, how it is positioned and tagged, how it conditions the DiT |
| [`audio_freeze.md`](audio_freeze.md) | freezing a known audio track so the video follows it: the mechanism, what it found, the song node, the plan and ideas |
| [`image_editing.md`](image_editing.md) | the single-frame path, parked 2026-08-27: what it was and what it found |

## Against other implementations

| page | what it is for |
|---|---|
| [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | every known divergence of ComfyUI's H3 path from the release, with its cost, native status and what handles or watches it |
| [`custom_node_gaps.md`](custom_node_gaps.md) | what our nodes do end to end, and where that differs from sglang, LightX2V, DiffSynth, diffusers and native ComfyUI |

Studies behind these pages, under [`../research/`](../research/index.md): the t2va trace through ComfyUI, the DiT
across implementations, sglang's pipeline and how it compares, what the release declares, the conditioning nodes.
