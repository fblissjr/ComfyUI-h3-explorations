# Sol's token reorder under ComfyUI's memory compiler: accepted, 2026-09-18

> **2026-09-18, read first:** the noodle bar prompt is written for 107 frames and post office for 141; both were rendered here at 345, past the end of their scripts, where the model improvises. What this record measured stands; what it is evidence of changed. See `2026-09-18_off_length_prompts.md`.

Model: MiniMax H3, text to video with audio, `workflows/h3_text_to_video_api.json`
(the default chain: kitchen int8 dense, Sol with `qk_balance`), full length at
the trained canvas, noodle bar prompt from the bank, seed held. Card, driver,
torch and the kitchen build are in each row of
`2026-09-18_sol_reorder_under_memory_compiler_arms.jsonl` (`substrate`). Core
at `a8686f2b`, memory compiler ON (a normal server start, no
`--disable-comfy-compiler`).

## What changed

Until today the reorder did its work in a pre-hook on DiT block 0, inside the
scope whose allocations core records once and replays for every block, and the
first step failed with "aimdo memory compile error"
(`2026-09-17_sol_options_noodlebar_batch.md`; the node refused `morton` under
the compiler from 0.122.1). The reorder now runs outside the blocks: video rows
permuted on their way into the embedder, `position_ids` rows in the RoPE
wrapper, order restored on the final layer's video output
(`sol_attn_h3.py`, `install_h3_morton`; CHANGELOG 0.123.0).

## The test

Renders repeat bit for bit on this stack, so an infinite PSNR and an equal
audio hash against an existing clip is an identity test. The references are
the two compiler-OFF clips of 2026-09-17 (`2026-09-17_sol_reorder_noodlebar_arms.jsonl`):
`nocompiler_reorder3d_tau10` and `nocompiler_default`. The prompt was passed
with the same bytes those arms used (trailing newline included), read from
that record. Three renders in a row on ONE server, so the third also tests
whether a plan recorded with the reorder off goes stale when it comes on:

| order | arm | toggle | compared to | video | audio | compile error | graph breaks logged |
|---|---|---|---|---|---|---|---|
| 1 | compiler_reorder3d_tau10 | `morton` on, `3d` | nocompiler_reorder3d_tau10 | identical | identical | none | same as arm 2 |
| 2 | compiler_flip_off | `morton` off | nocompiler_default | identical | identical | none | the default's |
| 3 | compiler_flip_on_again | `morton` on, `3d` | nocompiler_reorder3d_tau10 | identical | identical | none | same as arm 2 |

Video by `ffmpeg -lavfi psnr` on the silent mp4 (infinite on every plane),
audio by the md5 of the decoded stream of the `-audio` mp4.
The server log read `Comfy model compiler graph breaks: 3, rogues: 0` after each
of the three renders.

## What it says

- The reorder works with the compiler on, and adds no graph break: the
  compiler's end-of-render line reads the same with the toggle on as off.
- Moving the permutation from the hidden states to the embedder's input, and
  from the RoPE table to `position_ids`, did not move a bit of the output.
- Sampler seconds are in the arms file (`sampler_s`), next to the 2026-09-17
  compiler-off rows for the same arms; the reorder costs nothing measurable
  either way.
- Flipping the toggle between renders on one server needs no restart.

## What it does not say

Nothing about whether the reorder should be a default: that is the panel's job
(`docs/eval_comparison.md`, "A stress scene is not a typical scene"). Nothing
about a render with reference or keyframe conditioning, where the target rows
are a slice of the embedder's input at a non-zero offset: that path is covered
by the CPU check (`bench/check_sol_reorder_equivalence.py`, `with_conditioning`)
and has not been rendered. Nothing about inpainting: under a non-uniform video
denoise mask the reorder now declines.
