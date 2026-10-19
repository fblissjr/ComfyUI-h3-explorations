# Sage fp8++ rotated against kitchen int8 as Sol's dense fallback: wall time (2026-10-01)

The owner reopened the lane parked on 2026-09-27 (`docs/state/next_steps.md`, the Sol node
redesign): our sage attention from the sage fork, in its `fp8++ rotated` mode, in place of
kitchen's `int8_attention` under `MiniMaxH3Sol`. Kitchen is already the more accurate kernel on
every dense-tail cell graded (`2026-09-27_sol_redesign_test2.md`, "The dense tail"), so the sage
case rests on speed, or on the two dense steps before Sol's window, which no capture has graded.
This is the speed half.

**Result: sage is faster by a little over one percent of the sampler on the t2v finish.** The
two settled runs of each arm agree closely, so the gap is real, but it is small.

## How

- **Graphs:** the shipped `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json` (kitchen)
  against the same graph from the generator's own sage chain (`build_workflows.py --chain sage
  --out internal/chain_sage`, validated against the live server; `h3_config.DENSE_CHAINS["sage"]`).
  The only difference is `MiniMaxH3SageAttention` in place of `ModelAttentionBackend`.
- **Scene:** `t2va_police_interrogation`, 1344x768, 345 frames, seed 730451892 held.
- **Runs:** `bench/sage_floor_timing_arms.json`: one discarded kitchen warmup, then kitchen, sage,
  kitchen, sage. Rows: `2026-10-01_sage_floor_timing.jsonl`, each naming kitchen
  `0.2.36+sol.aade8d5.up.3f7210f`. Same prompt throughout, so the text encode is paid once, in the
  warmup. No row carries an error or a cache hit.

## What it covers and does not

- Sage takes every dense call: the two steps before `start_percent` on every block, and blocks
  45, 48 and 49 on every step. The saving is spread over all of them; this run does not split it.
- One scene and one canvas. The dense share of the sampler depends on `start_percent`: if the
  panel in `bench/start_percent_panel_arms.json` moves it to 0.0, the steps before the window go
  and only the dense tail is left, where kitchen is the more accurate kernel.
- The accuracy half on the early steps is `bench/sage_floor_capture_arms.json`, not yet run.
