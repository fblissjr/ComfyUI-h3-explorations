# How much of a control vector do its words point at? Experiment 2, CPU half (2026-11-12)

lane: control

Date: 2026-11-12

The CPU half of experiment 2 of
[`docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md)
(section 11). The plan asked what share of a control vector is a sparse mix of word directions. The first cut
answered that as asked, and its own baselines show the answer says nothing about words. A second cut asks a
narrower thing that does separate a vector from the rest.

Both cuts are throwaways. Their scripts have no check and live beside their outputs in the untracked
`data/control_vectors/word_share_2026-11-12/` (`word_share.py` with `word_share.json`, `own_words.py` with
`own_words.json`). The figures below are what those scripts wrote. Nothing was rendered.

## 1. What was read

- **Vectors.** Every `*__fl2va__diff_span__lora_*.safetensors` in `<comfy>/models/control_vectors/`, both
  tensors: `vector.post_refiner` and `vector.encoder`.
- **Bundle.** `<comfy>/models/j_lens/j_lens_h3.safetensors` as rebuilt on 2026-11-12; `word_share.json` records
  its size and modification time.
- **Code.** `j_lens.py` as it stood in the working tree, read and not changed. Repo commit `fc151158`.
- CPU only.

## 2. First cut: a sparse mix of word directions

The split the plan describes, after the J-lens paper: fit each vector with up to a fixed number of word
directions by matching pursuit, and call what is left the remainder. The share is
`1 - |remainder|^2 / |vector|^2`. Two meanings of "word direction" were tried, each with weights held
non-negative (the paper's rule) and with signed weights:

- `word_lab`: the vector the Word Lab makes for one word, `normalize(J_ref Sigma^0.30 mu_w)` with `mu_w` as
  `j_lens.py::_mean_word_direction` builds it, against `vector.post_refiner`.
- `lens_rows`: the raw J-Lens rows, against `vector.encoder`. This is the paper's own split.

The same fit was run on three kinds of direction that carry no designed concept:

- `isotropic`: random directions.
- `shaped`: random directions with the shape the space gives them (`J_ref Sigma^0.30 g` for post-refiner,
  `Sigma^0.30 g` for encoder).
- `two scenes`: the difference of two random prompts' mean scene rows, from
  `data/control_vectors/rows_2026-11-09`.

Median share with sixteen words:

| Words as | Weights | Built vectors | `isotropic` | `shaped` | `two scenes` |
|---|---|---:|---:|---:|---:|
| `word_lab` | non-negative | 0.56 | 0.02 | 0.63 | 0.63 |
| `word_lab` | signed | 0.59 | 0.02 | 0.65 | 0.65 |
| `lens_rows` | non-negative | 0.015 | 0.005 | 0.012 | 0.027 |
| `lens_rows` | signed | 0.17 | 0.04 | 0.07 | 0.16 |

`word_share.json` has the shares at other word counts, the ranges, and each vector's figures.

What the table says:

- **The share does not tell a control vector from a direction with no concept in it.** A random direction of
  the right shape, and the difference of two unrelated scenes, are fitted as well as the built vectors or
  better, by the same number of words.
- **The words the fit picks are not the vector's words.** After the first one or two (for `head_swell`,
  "dome"; for `disco_fever`, "disco" and "dazzling") the picks are unrelated words with large weights of
  both signs. With many thousands of word directions in a space whose energy sits in a few hundred
  dimensions, sixteen of them fit most things.
- **The paper's split, taken literally, is degenerate here.** With non-negative weights on the raw rows the
  share stops growing after a few words, at about the level of a random shaped direction.
- **The fit itself works.** A Word Lab vector made from five words came back almost whole within sixteen
  (`word_share.json`, `control`).

So this number is not the "word-readable share" the plan wanted, and its two parts are not worth a render.

## 3. Second cut: does a vector's own readout point at it?

For each vector: take its J-Lens readout (eight words toward, eight away), have the Word Lab make a vector
from exactly those words (`j_lens.py::synthesize_from_words`), and take the cosine with every built vector.

| | Cosine, median | Range or spread |
|---|---:|---|
| A vector's own words against that vector | 0.57 | 0.39 to 0.69 |
| Another vector's words against that vector | 0.18 | 5th to 95th percentile: -0.03 to 0.42 |

- For every vector, its own words are the best-matching word set. For 39 of 46, it is also the best-matching
  vector for its own words; for the other seven it is second.
- As a share of the vector's energy, the direction its own words name is the cosine squared: about a third
  for a typical vector, between about a sixth and a half across the bank (`own_words.json`, `per_vector`).
- This reproduces, on today's bundle, the figure in
  `docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md` section 6.1, and adds what that
  section did not have: the same words against other vectors.

### 3.1 The built vectors share a direction

- The built vectors are not spread evenly: a typical pair has a clearly positive cosine, and a typical
  vector has a cosine of about 0.55 with the mean of all of them, so close to a third of its energy lies
  along a direction every vector has (`own_words.json`, `built_vectors_cos_with_their_mean`).
- The Word Lab's vectors lean the same way, less strongly.
- With that shared direction taken out of both sides, a vector's own words still point at it (median cosine
  0.46) and another vector's words do not (median about zero, 95th percentile 0.24).

This is the "shared plain side" of
`docs/research/control_vectors/2026-11-09_control_vector_simplification_plan.md` section 20.1, seen in the
vectors themselves: every pair file sets its concept against something plain, so every vector carries
"away from plain".

### 3.2 The eight vectors the words could not name

The eight vectors that no bundle's words named in
[`2026-11-12_control_vector_readout_blind_score.md`](2026-11-12_control_vector_readout_blind_score.md)
(section 9.2) have own-words cosines like the rest of the bank's. With the shared direction removed they sit
a little lower than the others (`own_words.json`). Their words point at them about as well as any vector's
do; what the words do not do is tell them from a neighbor.

## 4. What this cannot see

- **What the DiT does with any part.** Every figure here is geometry on the conditioning side.
- **Whether the cosine is the right measure of "the same steering".** Two vectors with a cosine of 0.57 may
  render alike or not.
- **The old bundle.** These figures are for the rebuilt one only.
- **Anything about `ref2va`.** Only the `fl2va` vectors were read.

## 5. Readings, not findings

- The plan's question was wrong as posed. "A sparse mix of word directions" fits anything; "the direction
  its own readout names" is the thing that belongs to one vector.
- A vector's energy now has three parts that can be named and separated: the direction all bank vectors
  share, the direction its own words name beyond that, and what is left. Going by the medians in
  `own_words.json`, these are roughly a third, a sixth and a half of a typical vector. The three were not
  computed vector by vector here.
- That makes the render half a different experiment from the one planned: four arms per vector (the whole
  vector, the named direction, the remainder, and the shared direction alone), not two parts of a
  matching-pursuit split.
