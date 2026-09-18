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

Owner's verdicts, 2026-09-18, all six on-length pairs (post office is off-length and was skipped):

- plain order better: diner, kitchen, market (slightly)
- `3d` better: cafe_kids, on a prompt with two wording faults that are now fixed; re-rendered as `cafe_kids_v2_*`, not yet scored
- no difference: crowd_churn_long, hardware_aisle_short

**Conclusion.** On no scene did the reorder remove a defect plain order had. Where the owner saw a difference it was the
difference between two takes (staging, which character speaks, a figure the prompt did not ask for, the sex of a character),
and it went to plain order three times and to `3d` once. `morton` stays OFF by default. The capture metrics had said the
reorder lowers Sol's error on most cells and raises it on one block of one scene; the eye does not see the former as a
benefit on scenes rendered at the length their prompts were written for. What is left for the reorder is an untested use:
letting tau rise for speed (`3d` at a higher tau against plain order at tau 1.0, same six scenes).

| scene | plain order | `3d` reorder | notes (scored blind as clip 1 / clip 2; each key opened only after its verdict) |
|---|---|---|---|
| diner | **better, the owner's pick**: "a better scene overall ... more things in it that make sense", a neon sign of the diner, more legible text | `3d`: weaker; less legible text, and a cook walks past OUTSIDE briefly, which the prompt does not ask for (its cook is at the griddle, in the background) | Owner, 2026-09-18, blind as clip 1 / clip 2; key opened after: clip 1 is `3d`, clip 2 is plain (the 2026-09-15 clip). Column order here is plain, then `3d`. A loss for the reorder on a typical scene: subtle, but on text legibility and an unscripted figure, which are the kinds of thing the panel exists to catch |
| kitchen | **the owner's pick**: at about 4 s the line cook is a short-haired woman with earrings, which matches the prompt ("her", "she", a "light, quick soprano") | `3d`: the line cook at about 4 s is a man, and on the audio clip a woman's voice comes out of a man's face ("weird") | Owner, 2026-09-18, blind; key opened after: clip 1 is `3d`, clip 2 is plain (the 2026-09-15 clip). A loss for the reorder on a character the prompt does specify |
| market | **slightly better, the owner's pick**: no coins seen, but they are heard and "you can assume he put them in"; she stacks the oranges onto crates at the end, as the prompt asks | `3d`: good too, but the coins appear from the middle of the crate | Owner, 2026-09-18, blind; key opened after: clip 1 is plain (the 2026-09-15 clip), clip 2 is `3d`. "Both are good"; without the coin moment the owner "may have said equal" (their message names clip 1 for the coins at that point and clip 2 earlier; read as clip 2, which is where they first placed it). The owner puts the coins down to the prompt, which names no one dropping them, the same sentence behind the coins-from-nowhere on the other market seed (`2026-09-15_block49_repro_batch.md`). So: a slight plain-order preference, on a moment the prompt leaves open |
| hardware_aisle_short | no difference seen | no difference seen | Owner, 2026-09-18, blind: "im not sure i can tell a difference with hardware." Key opened after the verdict |
| post_office | | | |
| cafe_kids | plain order: no morph or duplicate reported. Two burgers on each child's plate but the red-haired boy's; the brunette reads slightly red-haired; the red-haired boy (centre) and the soda girl speak the lines the prompt gives them | `3d`: no morph or duplicate reported. One burger each; "just better" framing, the better-looking clip; the red-haired boy sits at the left edge, out of frame, and his line is spoken by the brunette girl, whom the prompt declares silent | Owner, 2026-09-18, scored blind as clip 1 / clip 2, key opened after: clip 1 is plain, clip 2 is `3d`. Owner's call: "if the prompt isn't specific on that, then clip 2 wins." The prompt does not say where anyone sits; it does say the red-haired boy is on-screen and speaks that line, that the brunette makes no vocal sound, and "a plate of hamburgers" (plural) each. **Owner's verdict: clip 2 (`3d`) wins.** Two burgers per child is implausible whatever the grammar allows ("why would each kid be eating two hamburgers"); the plural is loose prompt wording that plain order rendered literally and `3d` rendered sensibly. Caveat kept: `3d` puts the first line on the wrong child. Neither clip shows a morph or a duplicate, so this is a win on the quality of the take, not on the defect the panel asks about. Both prompt fixes made the same day at the owner's word (one hamburger per plate; the red-haired boy seated at the centre facing the camera), and the pair re-rendered on the new text as `cafe_kids_v2_*`, stack `blind_cafe_kids_v2_plain_vs_3d.mp4` |
| crowd_churn_long | no difference seen | no difference seen | Owner, 2026-09-18, blind: "dont think i can tell the difference ... both get grainy in the middle to end because so much is going on but both look and sound equal to me." Key opened after: clip 1 is `3d`, clip 2 is plain. The grain from the middle on is in BOTH arms, so it is not a token-order effect; it is what this chain does on the densest scene in the bank, and a question for a fully dense control, not for this panel |
