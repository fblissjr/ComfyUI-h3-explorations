# Reorder panel: plain token order against the `3d` reorder, seven scenes, 2026-09-18

> **2026-09-18, read first:** the noodle bar prompt is written for 107 frames and post office for 141; both were rendered here at 345, past the end of their scripts, where the model improvises. What this record measured stands; what it is evidence of changed. See `2026-09-18_off_length_prompts.md`.

Model: MiniMax H3, text to video with audio, `workflows/h3_text_to_video_api.json`
(default chain: kitchen int8 dense, Sol with `qk_balance`, tau 1.0), full
length at the trained canvas, one seed per scene. Core at `a8686f2b`, memory
compiler on, kitchen `0.2.35+sol.8176242`, h3 pack at 0.123.0. Card, driver and
sampler seconds per render: `2026-09-18_sol_reorder_panel_arms.jsonl`.

## Why

On 2026-09-17 the owner found every plain-order noodle bar arm morphing around
four seconds and the two `3d` reorder arms clean
(`2026-09-17_sol_options_noodlebar_batch.md`). That is one stress scene at one
seed. A default moves on a panel (`docs/eval_comparison.md`, "A stress scene is
not a typical scene, and neither carries a default alone"), and this is it.

## Arms

One thing differs between the two clips of a pair: `morton` off against
`morton` on with curve `3d`. Same prompt bytes (stripped), same seed.

| scene | seed | plain-order clip | reorder clip |
|---|---|---|---|
| diner | 730451892 | `Video/block49_repro/diner_default_*` (2026-09-15) | `Video/reorder_panel/diner_3d_*` |
| kitchen | 730451892 | `Video/block49_kitchen/default_*` (2026-09-15) | `Video/reorder_panel/kitchen_3d_*` |
| market | 20260915 | `Video/block49_repro/market_default_*` (2026-09-15) | `Video/reorder_panel/market_3d_*` |
| hardware_aisle_short | 730451892 | `Video/block49_repro/hardware_aisle_short_default_*` (2026-09-15) | `Video/reorder_panel/hardware_aisle_short_3d_*` |
| post_office | 730451892 | `Video/block49_repro/post_office_default_*` (2026-09-15) | `Video/reorder_panel/post_office_3d_*` |
| cafe_kids | 730451892 | `Video/reorder_panel/cafe_kids_plain_*` | `Video/reorder_panel/cafe_kids_3d_*` |
| crowd_churn_long | 730451892 | `Video/reorder_panel/crowd_churn_long_plain_*` | `Video/reorder_panel/crowd_churn_long_3d_*` |

The five 2026-09-15 plain clips are reused, not re-rendered. That is sound
because renders repeat bit for bit on this stack and the first row of the arms
file proves it for today's build: `diner_plain_k0235` is the diner plain arm
rendered today, identical in video and audio to the 2026-09-15 clip
(`2026-09-18_kitchen_0.2.35_rebuild.md`). Prompt and seed for those five were
read from the metadata embedded in the 2026-09-15 clips, not from the bank, so
the pair shares its bytes by construction. The first five scenes are typical
ones in which the owner saw nothing wrong on 2026-09-17 (the question there is
"does the reorder harm anything"); `cafe_kids` (many figures and props at one
table) and `crowd_churn_long` (a fast handheld take through a crowd) were added
as hard moments for identity and duplication.

All ten renders succeeded; the server logged no compile error.

## Scoring

Blind pairs, top and bottom in random order per scene:
`Video/reorder_panel/stacks/blind_<scene>_plain_vs_3d.mp4`, key beside each as
`key_<scene>.json` (open it after scoring). One question per clip: does
anything morph, duplicate, or lose its identity? Stacks carry no audio; sound
goes back to the singles (`*-audio.mp4`).

Owner's verdicts: not yet scored.

| scene | top | bottom | notes |
|---|---|---|---|
| diner | | | |
| kitchen | | | |
| market | | | |
| hardware_aisle_short | | | |
| post_office | | | |
| cafe_kids | | | |
| crowd_churn_long | | | |
