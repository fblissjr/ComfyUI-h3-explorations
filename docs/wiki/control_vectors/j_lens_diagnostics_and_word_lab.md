# J-Lens: naming a vector in words

last updated: 2026-11-10

[Back to the start page](../control_vectors.md)

J-Lens is a precomputed bundle, `j_lens/j_lens_h3.safetensors` in the models folder, and the code that
reads it, `j_lens.py`. It is a diagnostic. "The plan" is
[`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md);
its section 17 is the reading this page is taken from.

## What it does

- It names a vector in words. A build and `inspect` print the words a vector pushes toward and away from.
  This is the use that has been checked: on 2026-11-09 its words were compared with a second way of
  labelling vectors, from cached rows, over the bank (`bench/control_vector_lexicon.py`; the plan, section
  20). J-Lens named the concepts better overall, the other way caught some it missed, and both are kept.
- It builds a vector from a few words on the CPU in `< 2 ms`, with no encoder pass (the `words` command, and
  the `words` input of the apply node). Verified across 7 Pika renders (`Arms 53, 54, 61..65` in
  [`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)):
  strong for surface/atmosphere restyles (`"8-bit, LED, steampunk"`, `"blizzard, snowdrift, freezing, snow"`,
  `"disco, glitter, dancing"`, `"medieval, knight, armor, castle"`), while single-token lexical bags without
  sentence context are weaker than full-sentence contrastive pairs on structural/physical actions.
- It takes words out of a vector (`hold_words`). Covered in `bench/check_control_vectors.py`, though no
  recorded video arm in `bench/results/` uses `hold_words` directly (cross-vector orthogonalization via
  chaining a 10-pair vector with `mode = "hold"` was verified in `Arm 58`).
- It prints readiness figures beside the words. The thresholds the earlier version of this page gave for
  them had no arm behind them and are not repeated.

## How it works, in outline

The bundle has two halves. One maps a direction in the encoder's space to changes in the likelihood of
single words, using the encoder's own upper layers and output head. The other maps between the encoder's
space and the refined space a vector lives in, for each family, so a vector can be carried back to where
words can be read. The tensors and their shapes are `REQUIRED_BUNDLE_TENSORS` in
`bench/control_vector/j_lens_builder.py`; the method is in
[`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md),
a dated record.

## Checking the bundle

```bash
<comfy-venv-python> bench/control_vectors.py build-j-lens --verify-only
```

## Before rebuilding it: read this

`build-j-lens` without `--verify-only` is half a builder, and using it as it stands can lose the bundle
(the plan, section 17.1, read against the code on 2026-11-09):

- It recomputes only the half that maps between the two spaces. The word half is copied from the bundle
  already on disk (`--base-bundle` or `--out`), and no tracked code can make that half again.
- Since 2026-11-10 `build-j-lens` refuses to overwrite an existing bundle file unless `--overwrite` is passed
  explicitly (or pass `--out <new_path.safetensors> --base-bundle <existing_bundle>` to write a separate file).
- If the Pika LoRA spec fails to parse, it goes on without the LoRA and says nothing.
- The bundle does not record which checkpoints, LoRAs, prompts or constants made it.
- Keep a backup copy of `j_lens_h3.safetensors` before passing `--overwrite`.

## What is decided, and not done

Decided on 2026-11-09: write the real builder for the word half, from the full encoder weights, and check
it against the bundle on disk. It depends only on the base encoder, not on a family or a LoRA. Then the
research folder `docs/research/control_vectors/exl3_qwen3vl/` goes, except its report
([`h3_control_vector_report.md`](../../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md)),
with the one piece
of it the tracked code needs moved into tracked code. The other changes the plan lists for the builder and
for `j_lens.py` are in its section 17.2. None of this has been started.
