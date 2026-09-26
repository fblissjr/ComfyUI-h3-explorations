# FastH3 V2 to FastVideo's contract, against ComfyUI's template, 2026-09-26

**A first look, not a verdict.** One seed and one scene, unjudged by the owner.
The manifest is `bench/fasth3_contract_arms.json`, and the rows are
`2026-09-26_fasth3_contract_s1.jsonl`.

**Arms:** all on the diner scene, seed 730451892 (the owner's washed-out FastH3
clip had this seed), 1344x768, 345 frames, shift 10/3, BasicGuider, kitchen
backend.
- `template`: ComfyUI's template. 8 `simple` steps on `res_multistep`, VSA keep
  10% after a 20% dense warm-up.
- `contract`: FastVideo's contract (`h3_config.FASTH3_CONTRACT_*`). The
  release's positions on Euler, VSA keep 20% on every step.
- `contract_attn`: the template's sampling with the contract's attention.
- `contract_sampling`: the contract's sampling with the template's attention.

## Checks that need no judgement

- **Every arm rendered.** No row carries `error` or `suspect_cache_hit`.
- **The attention change decides the take**, not the sampler (`ffmpeg psnr`,
  whole clip):
  - `contract` vs `contract_attn`, which share the attention: 16.8 dB;
  - `template` vs `contract_sampling`, which share the attention: 17.6 dB;
  - across the two groups: 14.5 and 14.7 dB.
- **The contract's attention is also faster:** `sampler_s` 199 against 217 to
  221. VSA from the first step at keep 20% costs less than keep 10% with a
  dense warm-up step.
- **Audio level** (`volumedetect`, mean / max dB):

  | arm | mean | max |
  |---|---|---|
  | template | -27.6 | -8.1 |
  | contract | -29.0 | -9.8 |
  | contract_attn | -26.2 | -9.9 |
  | contract_sampling | -31.1 | -10.2 |

## By eye, not a ranking

Four frames per arm at frames 24, 120, 216 and 312, beside the shipped FlashGen
render at the same seed.

- **`template`:** frames 120 and 216 are the same close-up. The cut to the
  prompt's third shot, the view through the window, does not happen by frame
  216.
- **`contract` and `contract_attn`:** four distinct shots, including the view in
  through the rain-streaked window. The neon reads "BLUE STAR" (mirrored from
  inside, as the geometry puts it), and in `contract_attn`'s last frame
  "BLUE STAR DINER" is legible.
- **`contract_sampling`:** four shots, but the third is an interior wide, and it
  is the darkest arm (`signalstats` YAVG).

None of these looks washed out the way the owner described the 0.141.0 clip.
Whether that clip's look came from the template, the older graph, or the
owner's display, this run cannot say.

## Clips

On the output share under `Video/`, each with a `-audio.mp4` companion:
- `h3_probe_t2v_fasth3_8step_template_00001`
- `h3_probe_t2v_fasth3_8step_contract_contract_00001`
- `h3_probe_t2v_fasth3_8step_contract_attn_contract_attn_00001`
- `h3_probe_t2v_fasth3_8step_contract_sampling_contract_sampling_00001`

## Next

- **The owner's look**, at `contract` against `template` above all.
- **A second seed and scene**, then FastH3 at the contract against PDD8 and the
  shipped FlashGen.
- **Not available here:** the bf16 FastH3 file the Comfy repack also ships. Only
  the int8 build is downloaded.
