# The 2026-09-26 night: what was worth it

Written 2026-09-27 at the owner's request: this session's view of the most
promising and practical findings and techniques of the night. It is a
judgement, not a record. Every claim points at the record that holds it, and
everything rests on one seed (730451892), with the owner's eye still pending
on most of it. The full account is `2026-09-27_followup_summary.md`, the
per-render inventory is `2026-09-27_inventory.md`, and the queryable version
is `2026-09-27_render_dataset/`.

## Findings, best first

1. **The blocky, splotchy blacks are our save setting, not a model.**
   - Decoded losslessly from a PDD8 latent, the darks were clean, and the
     8-bit h264 mp4 added the blocking (`2026-09-27_o1_lossless.md`).
   - One clip, but the fix is one line in the generator and would lift every
     render. The best return on effort of the night.
2. **FlashGen's 4-step ability lives in a small late change.**
   - Its last 16 blocks, a nearly rank-2 update, alone produce a finished
     render. Its large early change mostly adds the hazy, lifted-black look.
     Early-only is black mush.
   - FT1 in `2026-09-26_flashgen_weights_predictions.md`, and
     `2026-09-26_flashgen_weights.json`.
   - A distill may split into a "finish in few steps" part and a
     "look and composition" part. The second is where the owner's
     overfitting theory (O2) would put the adherence trouble.
3. **LoRAs on int8 must be applied exactly, not merged.**
   - The merge lost most of FlashGen's and Turbo's change and noised PDD's
     (`2026-09-26_int8_lora_requant.json`).
   - Now the default everywhere (0.154.0). The owner caught the stacking bug
     in it by eye (`2026-09-26_followup_contamination.md`).
4. **A seed-matched base and distill do not share starting noise.** The base's
   `er_sde` replaces the seeded noise at its first step (`docs/distills/which_distill.md`,
   the working model). That explains much of "same seed, different scene"
   and changes how every base-against-distill pair is read.
5. **The three distills put their change in different places, and each has a
   stable signature.**
   - FlashGen changes early weights and barely touches timing.
   - FastH3's look lives in its attention gates and backbone; its time
     conditioning is near-inert (`2026-09-27_fasth3_swap.md`).
   - Across every look: FlashGen lifts the blacks, PDD8 is flat and dim
     ("naturally so", the owner), and FastH3 is detailed and warm-orange
     (`2026-09-26_distill_signatures.md`, `2026-09-26_followup_looks.md`).
   - Enough to choose a distill per shot.
6. **Two lessons of method.**
   - The subway "clone" was the prompt's own two people
     (`2026-09-27_clone_base_control.md`): render the base before blaming a
     distill.
   - At one seed, latent distance measures divergence, not effect size (the
     method note in `2026-09-26_distill_run_predictions.md`).

## Techniques and recipes, most promising first

1. **PDD first, FlashGen finishing** (the reverse switch, the owner's idea).
   - It keeps PDD8's composition and colour and lifts its dim highlights to
     the others' level (`2026-09-27_reverse_switch.md`).
   - The clearest win of the night. Next: the lamp-lit interiors, where PDD8
     trails most.
2. **FlashGen on its late blocks only** (`MiniMaxH3LoRABranch.blocks="34-49"`).
   - A less hazy FlashGen with deeper blacks and full motion detail (FT1).
   - Open: whether it also follows prompts better.

   *Annotation 2026-09-27, later: the haze claim does not generalise.* In the
   finisher grid's #46 arms at 124 frames, late-only had less haze on three
   scenes and more on two, and less contrast on four
   (`2026-09-27_finisher_grid.md`, Part B). As a finisher after PDD8 it
   matches full FlashGen (#34).
3. **Untested, and the most promising next render: PDD first, finished by
   late-only FlashGen.** If the late blocks carry the finish and the early
   blocks carry the haze, this should lift PDD's highlights without adding
   FlashGen's haze. One render.

   *Annotation 2026-09-27, later: tested, no gain.* The late-only finish
   matches the full one, and neither adds haze over PDD8
   (`2026-09-27_late_switch.md`). Technique 1, with full FlashGen as the
   finisher, stands as the recipe.
4. **PDD6.** Within 5-10% of PDD8's detail at three quarters of the steps
   (`2026-09-27_ladder.md`). A practical default candidate.
5. **Weight map, then block transplant.**
   - Map effective rank and inter-distill cosines, form a hypothesis, and test
     it with a loader's `blocks` input.
   - It went from hypothesis to result in one render round (FT1), and it
     works for any adapter.
6. **The scaffolding that made the night's corrections fast.** Each one turned
   a wrong guess into a correction within hours:
   - predictions registered before rendering;
   - `_savelat` twins;
   - the per-step x0 observer;
   - a base control;
   - lossless decode against the mp4.

## Directionally wrong or dead ends

- **Blending adapter weights.** The contamination was that experiment, and it
  gave "blocky as hell".
- **Lowering PDD's strength.** It loses detail and fixes nothing
  (`2026-09-27_pdd_strength.md`).
- **FastH3 with VSA off.** It loses the detail, which is part of what the
  owner wants from it.
- **Swapping FastH3's time conditioning.** Inert.
- **Testing FlashGen's overfitting by seed spread.** The owner's correction:
  the signature is prompt specificity.
