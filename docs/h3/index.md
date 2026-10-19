# H3: the model and its ComfyUI path

## Start here

| page | what it is for |
|---|---|
| [`uses.md`](uses.md) | **can we do X?** Everything that can be made here, one row per use: what H3 does for it, the nodes and graph, its state, the evidence |
| [`stages.md`](stages.md) | one row per stage of a render: our code, the page that owns it, the check that guards it, the implementation to compare against |

## Inputs and their costs

| page | what it is for |
|---|---|
| [`resolutions.md`](resolutions.md) | every legal canvas and why the set is what it is; the length axis |
| [`geometry_and_nodes.md`](geometry_and_nodes.md) | the frame grid, the token maths, and which node to use |
| [`input_impacts.md`](input_impacts.md) | what canvas, length and Sol-Attn cost together |
| [`references.md`](references.md) | every reference type, its processing, label rules, the sizing knobs, how to prompt it |
| [`conditioning.md`](conditioning.md) | every conditioning input end to end: what the encoder sees, what the VAE does, how it is positioned, how it conditions the DiT |
| [`audio_freeze.md`](audio_freeze.md) | freezing a known audio track so the video follows it: the mechanism, the routes, the plan |
| [`image_editing.md`](image_editing.md) | the single-frame path, parked 2026-08-27 |

## Against other implementations

| page | what it is for |
|---|---|
| [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | every known divergence of ComfyUI's H3 path from the release, with impact and what enforces it |
| [`custom_node_gaps.md`](custom_node_gaps.md) | what our nodes do end to end, and where that differs from sglang, LightX2V, DiffSynth, diffusers and native ComfyUI |

Studies behind these pages, under [`../research/`](../research/index.md): the t2va trace through ComfyUI, the DiT
across implementations, sglang's pipeline and how it compares, what the release declares, the conditioning nodes.
