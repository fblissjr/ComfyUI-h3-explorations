# Did core's move of the encoder RoPE onto kitchen change its numbers? 2026-09-19

Tool: `bench/probe_encoder_rope_kitchen.py`. Two runs, numbers in their JSON:

- `bench/results/2026-09-19_encoder_rope_kitchen.json`: `--device cuda`, the
  kernel renders use. Run by the lead session on the RTX 4090 in a gap before
  its renders, with the probe's own command; interpreted and committed here.
- `bench/results/2026-09-19_encoder_rope_kitchen_cpu_eager.json`: `--device
  cpu`, kitchen's eager path, a smoke test of the script.

## The question

ComfyUI core `6cff1e97` (PR 16326), pulled into this install on the morning
of 2026-09-15, replaced the Qwen3-VL text encoder's torch RoPE
(`comfy/text_encoders/llama.py::apply_rope`) with a call to
`comfy_kitchen.apply_rope_split_half`. No render record spans the change
(`docs/sol_upstream.md`, section "comfy-kitchen and core, 2026-09-19"). The
probe feeds the old function (copied from `6cff1e97^`) and the installed one
the same bf16 q and k, laid out as the encoder lays them out, with the
32B config's M-RoPE tables over a sequence that includes a vision span, and
grades both against an fp64 reference. A sign-flipped control must fail.

## What it found

- **On CPU (eager): bit-identical.** Old and new agree in every element; the
  control fires.
- **On the 4090 (CUDA): not bit-identical, and equally accurate.** A handful
  of elements per tensor differ, out of every q and k element in the probe
  (the JSON has the counts at both input scales). The largest difference at
  each scale is one bf16 unit in the last place for the largest values at
  that scale (`q_max_abs_vs_old`); the per-element steps were not listed. Against the fp64 reference
  the two implementations' relative errors agree to many digits, the new one
  marginally lower. The control fires on both devices, so the comparison can
  go red.
- **Reading.** The kernel rounds a few elements the other way from the old
  torch sequence of multiply-adds: a rounding-order difference, not an error
  and not a change in accuracy. Nothing here is worth chasing in kitchen.

## What it means for records

The encoder's output is not guaranteed bit-identical across the 2026-09-15
core pull, so a same-seed render from before that pull and one after it may be
different samples, however alike. Any record that pairs clips across that
morning should say so. The lead has already checked this week's reused clips
against that boundary (`internal/COORDINATION.md`, lead's reply of
2026-09-19): the clips reused in standing verdicts re-rendered bit for bit
after the pull. Whether the one-ulp differences grow through the encoder's
layers into a visible conditioning change is not measured here; it would take
a full encode of one prompt on each side, and the equal accuracy gives no
reason to expect a direction.
