# How the Word Readout Works

Last updated: 2026-11-12.

The word readout turns a control vector into two short lists of English words: what the vector pushes toward
and what it pushes away from. In this repo it is called the J-Lens. For what to call it elsewhere, and how it
sits beside control vectors and edit guidance, see
[`which_tool_and_what_to_call_it.md`](which_tool_and_what_to_call_it.md).

This page follows one vector, `film_16mm_1974`, through the readout. The math is in
[`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md)
and is not repeated here.

---

## 1. Try it

From the repo root, with the ComfyUI venv's python. It runs on the CPU and loads no model.

```bash
CUDA_VISIBLE_DEVICES= <comfy-venv-python> bench/control_vectors.py inspect \
  film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors
```

The file name is the one in `<comfy>/models/control_vectors/` when this page was written. The last part is the
recipe's tag and changes when a vector is rebuilt under another recipe, so list the folder if the name is
refused.

Look for two lines in what it prints:

- `j_lens (PR +)`: the words the vector pushes toward. For this vector they are words like "vintage", "dated"
  and "faded".
- `j_lens (PR -)`: the words it pushes away from. Here, words like "cinematic", "lighting" and "sleek".

That is the whole readout. The rest of this page is how those two lines are made.

---

## 2. The four steps

### Step 1: H3 throws away the layers that speak

H3's text encoder is a language model with its top cut off. H3 takes the encoder's state part of the way up
and hands it to the video model. The layers above that point, and the head that turns a state into words, are
not in the shipped encoder file at all. [`../control_vectors.md`](../control_vectors.md) section 2 describes
the two text stages, and section 10 of
[`../../research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md)
has the command that shows which encoder files carry a head.

So there is nothing in the running pipeline that can say what a state means in words.

### Step 2: one matrix stands in for the missing layers

The full encoder does have those layers. Once, ahead of time, it was worked out how a small change at the cut
moves the full encoder's final output, averaged over a set of prompts. The result is one matrix,
`J_49_to_63` in the bundle file `<comfy>/models/j_lens/j_lens_h3.safetensors`.

Multiplying a direction by that matrix gives a fair guess at what the missing layers would have done with it.
It is an average, so it is a guess that holds on the whole and can miss in any one case.

### Step 3: a control vector has to be carried back first

A control vector does not live at the cut. It lives one stage later, after the token refiner, because that is
where `MiniMaxH3Steer` adds it. A second matrix in the bundle carries a vector back from there to the cut.
There is one per family (`J_wiener_fl2va`, `J_wiener_ref2va`), because the two families' refiners are
different.

For `film_16mm_1974` this is the difference between the `(PR +)` lines and the `(EN +)` lines that `inspect`
prints. `PR` reads the vector the render uses, carried back through the refiner. `EN` reads the copy of the
vector that was saved at the cut. The two lists mostly agree, and where they differ the `PR` list is the one
that describes what gets rendered.

### Step 4: score every word, and keep the top few

The readout has a fixed list of whole single-token English words. For each one it asks how much more likely
the vector makes that word at the end of a few fill-in-the-blank prompts, such as a sentence that stops at
"The lighting, color palette, contrast, and film stock look". It compares against two things: the same prompt
with no vector, and the same prompt with the vector reversed. A word is scored by the smaller of those two
gains, so it has to rise against both. The best few are the "toward" list. Doing the same with the vector
reversed gives the "away" list.

The code is `j_lens.py::decode_vector`.

---

## 3. Three uses of the same machinery

- **Read.** What section 1 did. Every build also writes the readout into the `.md` file beside the vector,
  and `j_lens.py::check_b_side_alignment` warns at build time when the words a pair file aimed for did not
  make it into the "toward" list.
- **Scrub.** Take a word's direction out of a vector. This is `hold_words` on `MiniMaxH3Steer`,
  `MiniMaxH3LoadControlVector` and `MiniMaxH3WordVector`, and `--hold-words` on the command line. The code is
  `j_lens.py::apply_soft_holds`.
- **Build.** Run it backward: start from a word list and get a vector. This is the `MiniMaxH3WordVector` node
  wired into `MiniMaxH3Steer`'s `custom_vector`, or `bench/control_vectors.py words`. The code is
  `j_lens.py::synthesize_from_words`.

[`j_lens_diagnostics_and_word_lab.md`](j_lens_diagnostics_and_word_lab.md) has the settings for each.

---

## 4. What it cannot see

- **Only whole single-token words.** A term the tokenizer splits into pieces is not on the list, so the
  readout cannot say it. [`j_lens_diagnostics_and_word_lab.md`](j_lens_diagnostics_and_word_lab.md) section 3
  names some.
- **No order and no relations.** It gives a bag of words. It cannot say who does what to whom.
- **What the encoder would have said, not what the DiT does.** The video model never runs the missing layers.
  The words are evidence about what the video model is handed. Only a render shows what it makes of it.
- **Close neighbors.** Two vectors that overlap read alike. `film_16mm_1974` is the example: with the bundle
  as it stood before 2026-11-12 its "toward" list included "tape", and blind judges took it for the VHS
  vector.
  [`../../../bench/results/2026-11-12_control_vector_readout_blind_score.md`](../../../bench/results/2026-11-12_control_vector_readout_blind_score.md),
  sections 6.4 and 7.3.
- **The words move when the bundle is rebuilt.** The same record, section 8.4. A readout quoted in a document
  is true of the bundle it was read with.
- **Only part of a vector.** A vector built from a vector's own words points at it and is still some way
  from it, and part of every bank vector is a direction the whole bank shares.
  [`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../bench/results/2026-11-12_control_vector_word_share.md),
  sections 3 and 3.1.
