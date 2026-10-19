# #34: PDD8 finished by late-only FlashGen, against the full-FlashGen finish (2026-09-27)

`docs/state/open_experiments.md` #34. PDD8 runs to sigma 0.8 (`STEP_SWITCH_REV["h080"]`),
then FlashGen finishes in two steps, either whole (`rev_h080`) or on blocks
34-49 alone (`rev_late_h080`, `MiniMaxH3LoRABranch.blocks="34-49"`).

**The question.** FT1 found FlashGen's late blocks carry its 4-step finish and
its early blocks its haze (`2026-09-26_flashgen_weights_predictions.md`). Would
a late-only finisher lift PDD8's highlights, as the full finish does, with
less haze?

**The runs.**
- Manifest `bench/followup_late_switch_arms.json`; rows in
  `2026-09-27_late_switch.jsonl`.
- Seed 730451892, each scene at its bank prompt's written length: noodle_bar
  107 frames, radio_drama 243, courtroom_verdict and subway_chase 345.
- Graphs `workflows/distill_experiments/h3_probe_t2v_step_switch_pdd8_flashgen_{late_,}h080_savelat_api.json`
  (`e2873162`), on MiniMaxH3SolAttn, kitchen `0.2.35+sol.fc32da2.up.c8c7825`.
- New rows: `rev_late_h080` on all four scenes, and `rev_h080` on the three
  scenes that lacked it.
- Existing rows used as comparisons: `subway_chase__rev_h080` and every
  `<scene>__pdd8` (`2026-09-26_followup.jsonl`, the 0.154.8 rerun). The prompts
  are unchanged since those rows rendered (checked by `prompt_sha256`).
- The courtroom PDD8 reference resolved to the kitchen after-run's latent
  (`_00002`). That run was bit-identical to the original
  (`2026-09-27_kitchen_int8attn_after.md`).

**One property of this run makes the pair tighter than planned.** Within each
scene, `rev_h080` rendered straight after `rev_late_h080`, and ComfyUI served
the identical PDD8 pass from its cache. So both finishers start from the same
PDD8 state at sigma 0.8. The same caching makes the `rev_h080` timings
meaningless; nothing below uses a timing.

## Measures

`bench/analyze_followup.py --group late_switch` (`c0fc155e`), per scene:
`2026-09-27_late_switch_<scene>_{tone,resolution,temporal,divergence}.json`.
Medians over the clip; one clip per arm.

| scene | arm | white | rms_contrast | haze | detail | hf | moved_share | boil | motion_detail |
|---|---|---|---|---|---|---|---|---|---|
| noodle_bar | PDD8 | .882 | .191 | .070 | .0663 | .0225 | .264 | .639 | 2.061 |
| | full finish | .873 | .197 | .067 | .0696 | .0259 | .280 | .617 | 2.267 |
| | late finish | .862 | .195 | .066 | .0684 | .0256 | .283 | .599 | 2.228 |
| radio_drama | PDD8 | .798 | .190 | .049 | .0451 | .0164 | .072 | .250 | 1.256 |
| | full finish | .848 | .203 | .051 | .0483 | .0178 | .084 | .285 | 1.322 |
| | late finish | .842 | .201 | .050 | .0478 | .0177 | .082 | .279 | 1.353 |
| courtroom_verdict | PDD8 | .907 | .220 | .099 | .0446 | .0122 | .124 | .418 | 2.293 |
| | full finish | .930 | .225 | .101 | .0474 | .0138 | .134 | .458 | 2.650 |
| | late finish | .923 | .224 | .101 | .0470 | .0139 | .132 | .437 | 2.585 |
| subway_chase | PDD8 | .918 | .243 | .220 | .0396 | .0073 | .242 | .894 | 2.485 |
| | full finish | .958 | .250 | .224 | .0399 | .0077 | .244 | .892 | 2.739 |
| | late finish | .956 | .249 | .222 | .0403 | .0079 | .244 | .895 | 2.694 |

## Verdict

**#34's rule: take the late-only finisher if it matches `rev_h080`'s
highlight lift with less haze on most scenes; if haze matches, full FlashGen
stays. Haze matches, so full FlashGen stays.**

- **Haze:** late-only and full are within 0.002 on every scene, and both are
  within 0.004 of PDD8 alone. The full finish from 0.8 does not add FlashGen's
  haze, so a late-only finisher has none to remove.
- **Highlights:** both finishers lift PDD8's white on radio_drama, courtroom
  and subway, with the full finish a hair higher each time. On noodle_bar
  neither lifts it (PDD8 is already the brightest there).
- **Detail and motion:** both finishers raise detail, `hf` and motion detail
  over PDD8 on all four scenes, by nearly the same amount.
- **Divergence** from PDD8 is nearly the same for the two finishers (subway
  .439 full, .433 late). That is divergence, not effect size (the batch's
  method note).

**Reading (inference, not measured).** FlashGen alone shows its haze and lifted
blacks when it runs from pure noise (FT1, the distill signatures). Two steps
from sigma 0.8, over a PDD8 composition, do not reproduce it. So the haze is
set at high noise, where FlashGen's early high-rank blocks shape the global
tone. It is not something the late steps add. FT1's "early blocks carry the
haze" stands for FlashGen alone; it does not reach a late finish.

**Limits.** One seed, one clip per arm, four scenes, measured and not yet
judged. The owner's eye on the pairs would settle whether any difference is
visible. The clips are `Video/h3_probe_t2v_step_switch_pdd8_flashgen_{late_,}h080_savelat_<scene>__rev_{late_,}h080_00001-audio.mp4`.

**What changes.** Nothing ships. The reverse switch with the full FlashGen
finish (`rev_h080`) stays the candidate route for PDD8's dim highlights, the
VAE session's `2026-09-27_reverse_switch.md` and the board's
"reverse-interiors" direction. The late-only graph stays in
`distill_experiments/` as the recorded arm.
