# J-Lens: naming a vector in words

last updated: 2026-11-09

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
- It builds a vector from a few words on the CPU, with no encoder pass (the `words` command, and the
  `words` input of the apply node). No session has rendered with such a vector on record here. It treats
  words one at a time, with no sentence around them, so expect less of it than of a vector built from
  pairs.
- It takes words out of a vector (`hold_words`). Also not rendered on record here.
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
  already on disk, and no tracked code can make that half again.
- By default it writes over the bundle it copies from. That file is the only copy of the half that cannot
  be rebuilt.
- If the Pika LoRA spec fails to parse, it goes on without the LoRA and says nothing.
- The bundle does not record which checkpoints, LoRAs, prompts or constants made it.
- Its sample is the opening tokens of the first prompts of the bank in file order, so a rebuild would not
  reproduce the bundle on disk.

Keep a second copy of the bundle before running a rebuild.

## What is decided, and not done

Decided on 2026-11-09: write the real builder for the word half, from the full encoder weights, and check
it against the bundle on disk. It depends only on the base encoder, not on a family or a LoRA. Then the
research folder `docs/research/control_vectors/exl3_qwen3vl/` goes, except its report
([`h3_control_vector_report.md`](../../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md)),
with the one piece
of it the tracked code needs moved into tracked code. The other changes the plan lists for the builder and
for `j_lens.py` are in its section 17.2. None of this has been started.
