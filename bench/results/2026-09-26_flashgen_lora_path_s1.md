# How FlashGen reaches the int8 weight: first render pair, 2026-09-26

**A first look, not a verdict.** One seed and one scene, unjudged by the owner
at the time of writing. The manifest is `bench/flashgen_lora_path_arms.json`,
and the rows are `2026-09-26_flashgen_lora_path_s1.jsonl`. The why is
`bench/results/2026-09-26_int8_lora_requant.json`: a merged FlashGen delta
survives on the int8 checkpoint only under rounding noise several times its
size.

**Arms:** all on the diner scene (`t2va_diner_breakup`), seed 730451892,
1344x768, 345 frames, FlashGen's own sigmas on Euler.
- `r13`: kijai's rank-13 resize, merged by `LoraLoaderModelOnly`. This is the
  shipped FlashGen probe.
- `r64`: the publisher's full rank 64, merged the same way.
- `r64_branch`: rank 64 applied at the call by `MiniMaxH3LoRABranch`.
- `r64_branch_dense`: the same with Sol off on the kitchen backend, as
  vllm-omni's NPU recipe runs FlashGen.

## What happened to the run

- **The first `r64_branch` failed at load.** The node tested for
  `torch.nn.Linear`, and the int8 checkpoint's Linear comes from
  `comfy.ops.mixed_precision_ops`, which is not one. The CPU check had built its
  model on `manual_cast`, whose Linear is one, so it passed. Both are fixed: the
  node duck-types, and the check loads its weights into the mixed-precision
  class. The failed row stays in the JSONL with its error.
- **The server was restarted before the two branch arms**, which then rendered
  back to back. The dense arm reused the patched model from the node cache: the
  branch node logged once, 259 modules applied at the call.

## Checks that need no judgement

**How far apart the takes are** (`ffmpeg psnr`, whole clip):

| pair | PSNR |
|---|---|
| r13 vs r64 | 21.5 dB |
| r64 vs r64_branch | 15.5 dB |
| r13 vs r64_branch | 15.3 dB |
| r64_branch vs r64_branch_dense | 21.2 dB |

The two merged files are the same take with small differences. Applying the
LoRA at the call gives a different take. That is what the requant measurement
predicts: the two merges lose much of the same delta, and the branch keeps it.
Whether the different take is a better one is for the owner's eye.

**Frames, by eye** (four frames per arm, at frames 24, 120, 216 and 312):
- The merged arms are warm and saturated, and render the prompt's
  "BLUE STAR DINER" legibly as a neon sign.
- The branched arms are cooler, with more background detail through the
  window. Their neon text comes out mirrored or garbled.

This is description, not a ranking.

**Audio level** (`volumedetect` on the muxed audio, mean / max dB):

| arm | mean | max |
|---|---|---|
| r13 | -33.0 | -12.8 |
| r64 | -32.5 | -11.0 |
| r64_branch | -32.5 | -11.3 |
| r64_branch_dense | -31.1 | -11.3 |

**Sampler time** per row: `sampler_s` in the JSONL. The branch arm ran first
after a restart, so its row is not a warm comparison with `r64`. Dense
attention costs about what the Sol window saves.

## Second scene: subway, same seed

`subway_r64` and `subway_r64_branch` rendered warm, back to back.
- **PSNR between them:** 13.3 dB. A different take again, with the same shots
  and composition.
- **Audio level:** mean -22.9 dB merged and -19.1 dB at the call; max -6.6 and
  -2.6 dB.
- **What the branch costs:** it adds about 19 s to the four steps on this pair
  (`sampler_s`: 120.6 against 140.0). That is the per-call transfer of the
  rank-64 matrices from RAM plus their matmuls. It is not optimized.

## The shipped graph

`h3_text_to_video_flashgen` (0.146.0) rendered once on diner at the same seed
(row `ship_diner`). Its video decodes identical to `r64_branch`
(`ffmpeg psnr`: inf). It is the same graph under another output prefix, and the
branch path reproduces bit for bit across runs.

## The faster branch (0.147.0)

Rows `fast_r64` and `fast_ship` rendered back to back in one session after the
restart that loaded the change. Their `sampler_s` gives the branch's overhead
over the merged loader.
- **Against `ship_diner`** (the same graph and seed on the old branch):
  `ffmpeg psnr` gives 20.2 dB. The in-place add changed the rounding, and four
  deterministic steps amplified it into a visibly shifted take.
- **Exactness is unchanged.** On the GPU, the in-place path's error against a
  float64 reference was equal to or lower than the old path's (0.147.0 in
  `CHANGELOG.md`).

## Clips

On the output share under `Video/`, each with a `-audio.mp4` companion:
- `h3_probe_t2v_flashgen_4step_r13_00001`
- `h3_probe_t2v_flashgen_r64_4step_r64_00001`
- `h3_probe_t2v_flashgen_r64_4step_branch_r64_branch_00001`
- `h3_probe_t2v_flashgen_r64_4step_branch_dense_r64_branch_dense_00001`
- `h3_probe_t2v_flashgen_r64_4step_subway_r64_00001`
- `h3_probe_t2v_flashgen_r64_4step_branch_subway_r64_branch_00001`
- `text_to_video_flashgen_ship_diner_00001`
- `h3_probe_t2v_flashgen_r64_4step_fast_r64_00001`, `text_to_video_flashgen_fast_ship_00001`

## Next

- **The owner's look**, at `r64` against `r64_branch` above all, since that pair
  isolates the merge.
- **A second seed** before anything is shipped on this. The second scene is above.
- **The same pair for a LightX2V Turbo LoRA**, whose delta the merge loses
  almost entirely.
- **A 125-frame arm**, FlashGen's trained length.
