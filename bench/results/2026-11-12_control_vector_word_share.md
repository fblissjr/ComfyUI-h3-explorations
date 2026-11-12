# How much of a control vector do its words point at? Experiment 2 (2026-11-12)

lane: control

Date: 2026-11-12

The CPU half of experiment 2 of
[`docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md)
(section 11). The plan asked what share of a control vector is a sparse mix of word directions. The first cut
answered that as asked, and its own baselines show the answer says nothing about words. A second cut asks a
narrower thing that does separate a vector from the rest.

Both cuts are throwaways. Their scripts have no check and live beside their outputs in the untracked
`data/control_vectors/word_share_2026-11-12/` (`word_share.py` with `word_share.json`, `own_words.py` with
`own_words.json`). The figures below are what those scripts wrote. Sections 1 to 5 are the CPU half; section 6
is the render half, eight short clips made the same day and waiting for the owner's eye.

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

## 6. The render half

Rendered at the owner's word. Eight clips, one prompt, one seed. No clip has been judged: section 6.4 lists
what to look at, from two stills of each, and is not a reading of what the clips show in motion or sound.

### 6.1 The parts

For a test vector `v` (post-refiner), three unit directions that are at right angles to each other:

- **shared**: the mean of the other bank vectors' unit vectors, with both test vectors and
  `film_16mm_1974_neutral` left out of the mean.
- **named**: the Word Lab vector of `v`'s own eight-and-eight J-Lens words, with its component along
  `shared` removed.
- **rest**: `v` with its components along `shared` and `named` removed.

Share of each vector's energy (`part_vectors.json` in the run folder, written by `make_part_vectors.py`):

| Vector | shared | named | rest |
|---|---:|---:|---:|
| `film_16mm_1974` | 0.16 | 0.30 | 0.54 |
| `head_swell` | 0.19 | 0.15 | 0.66 |

Each part was written as an ordinary control-vector file with the test vector's own metadata, used for the
renders and then removed from `<comfy>/models/control_vectors/`. `make_part_vectors.py` rebuilds them.

### 6.2 What was rendered

- **Tool.** `bench/edit_guidance_arms.py` with `--vector-arm`, on
  `workflows/h3_text_to_video_pika_control_vector_api.json` (the shipped Pika `t2va` control-vector graph,
  nothing else changed). The arm graphs and `exp2_record.json` are in
  `data/control_vectors/word_share_2026-11-12/renders/`.
- **Prompt.** `prompt_bank/t2va_offpath_tortoise.txt`, read in full before queueing: one shot, an elderly man
  walking a tortoise across a zebra crossing past a waiting bus, no speech.
- **Length and seed.** The bank's own frame count for that prompt (`prompt_bank/bank.json`), the graph's
  seed; both are in each arm graph.
- **Arms.** `base` with no vector; for each test vector `whole`, `named` and `rest`; and `shared` once.
  Every steered arm is one `MiniMaxH3Steer` at the graph's shipped strength, in `add_norm_matched` and
  `scene_only`. That mode rescales every vector to one length, so the arms compare directions, not sizes.
- **Server.** ComfyUI as the owner had just restarted it; every arm returned `success`.
- **Repo commit:** `aa3c8df7`.

### 6.3 Where the clips are

Under `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/`:

- `exp2_film_16mm_1974_stacked.mp4`: five rows, top to bottom `base`, whole, named, rest, `shared`.
- `exp2_head_swell_stacked.mp4`: the same five rows for `head_swell`.
- `exp2/`: the eight single clips with their sound. The stacks carry no sound
  (`bench/stack_labeled_clips.py`).

### 6.4 What to look at

From frames 30 and 90 of each clip. A list of where to look, not a judgement.

- **The bus, in the film stack.** In the whole-vector row and in the named row the bus in those two stills is
  an older model with a radiator grille; in the `base`, rest and `shared` rows it is a modern one. Whether
  the picture also carries grain, fade or a colour cast in any row is for playback.
- **The man's head, in the head stack.** In those two stills his head looks the same size in every row,
  the whole-vector row included. If playback agrees, `head_swell` at this strength did not do its own job
  on this prompt, and its named and rest rows cannot be read for which part carries it.
- **The `shared` row.** In its stills the man has a white beard, the tortoise is larger and harnessed, the
  candle flame is large, and there is a flare beside the bus. Some of these also show in the `head_swell`
  rows (the harness, the larger flame).
- **Every steered row against `base`.** The street and the framing differ from `base` in all of them, the
  rest rows included.
- **Sound.** Not looked at here at all. The singles have it.

### 6.5 What the render half cannot see

- **More than one prompt, one seed and one strength.** A different seed alone changes a street.
- **Whether a part's effect at matched length says what it does at its own length.** Inside the whole
  vector the parts are not equally long (section 6.1); here each was given the whole vector's length.
- **`head_swell`'s parts, if its whole vector did nothing.** See section 6.4.

