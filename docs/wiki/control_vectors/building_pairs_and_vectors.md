# Building pairs and vectors

last updated: 2026-11-09

[Back to the start page](../control_vectors.md)

How a vector is made, from the pairs to the file. "The plan" is
[`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).

## Where a vector sits

A prompt reaches the DiT in two steps. The text encoder reads the pictures and then the text, left to
right, and gives one row per token. Inside the DiT a projection and a two-block refiner rework those rows,
looking both ways. A vector is added to the rows the refiner hands on, before the DiT's main blocks
(`control_vectors.py::apply_post_refiner`).

A vector file holds that direction as `vector.post_refiner`, at the length the pairs gave it, and a unit
copy of the same direction before the refiner as `vector.encoder`, which J-Lens reads. The file's metadata
records the pairs it came from, so a vector can be rebuilt from itself.

## Where pairs come from

A pair is two prompts, A and B, that differ in one clause. There are three sources.

- A pair file. A `.toml` under `prompt_bank/control_vectors/` with a `[vector]` table, `[[pair]]` blocks and
  optional `[[hold]]` blocks. The keys the loader takes are `control_vectors.ALLOWED_VECTOR_KEYS`,
  `ALLOWED_PAIR_KEYS` and `ALLOWED_HOLD_KEYS`; anything else is refused. Every file in the bank is a
  text-only prompt in the three-section layout.
- Your own prompt. You give the prompt and one or more `old wording -> new wording` clauses, and the pair
  compiler makes A the prompt as written and B the prompt with the clauses applied
  (`h3_pair_compiler.py::compile_words_to_pairset`). The clause grammar is the one edit guidance uses
  (`h3_prompt_edit.py`): one clause a line, `soundscape:` to aim at the soundscape, `hold:` for a hold.
- Two pictures. The same neutral prompt is encoded on top of picture A and of picture B, sized so both give
  the same number of rows (`control_vector_nodes.py::match_media_pair_geometry`). No session has run this
  path on 2026-11-09; treat it as untested.

What the loader refuses is in `control_vectors.py::differing_span` and `load_pairs`, and is held by
`bench/check_control_vectors.py`. In short: the two sides must differ, in more than whitespace, in one
clause (several only when the file declares `allow_multi_span`), with wording on both sides, and without
changing a section header or a shot header.

## Pooling: which rows are averaged

`control_vectors.VALID_POOLING` lists the choices. The default, `diff_span`, averages only the rows of the
wording that differs. `from_first_diff` averages from the first difference to the end of the text, and
`scene_text` averages every row of scene prose.

A study on 2026-11-09 over cached rows (`bench/control_vector_pooling.py`; the plan, section 20) found those
three give nearly the same direction, and that the rows a pair leaves unchanged move in a different, smaller
and consistent direction. The default stayed `diff_span`.

## Holds: taking a direction out

A hold is a second set of pairs for something you do not want the vector to carry, such as a colour cast
that comes along with a film look. Each hold group is averaged to a direction, the directions are made
orthonormal (`control_vectors.py::hold_basis`, two-pass Gram-Schmidt), and the vector's component along
them is removed (`project_out`).

A hold changes what the vector does, and that is easy to forget. The bank's `film_16mm_1974` file holds
out a warm colour cast, so its vector does not add warmth, while an edit that says "warm" does (the plan,
section 25.1).

## The component the bank's vectors share

The bank's vectors are not independent: unrelated looks point partly the same way (the plan, section 20).
Two causes are on the table and neither is settled: the steered clause is usually longer than the plain
one, and many plain sides share words like "natural". Nothing is subtracted for it. A vector built from
one pair of your own prompt carries more of that component when the new wording is much longer than the
old, which is why the prompt handed to an LLM asks for sides of similar length.

## Building with a reference picture

For a prompt that names `<Picture 1>`, the build can encode the real picture in front of the text
(`--ref-images` on the command line, `encoded_references` on the build node), so both sides of the pair are
read with the picture in view.

The helper behind `--ref-images` had never run until 2026-11-09: it read two config names that did not
exist. It is fixed and has now built one vector on a reference picture (the plan, sections 23.5 and 25). It
encodes the picture for the text encoder only and loads no VAE, because the VAE plays no part in the text
rows.

How much the picture matters for a vector is partly measured: with a picture in front, the description and
soundscape rows stay close to their text-only selves, and the subject, summary and retention rows move
about twice as far (the plan, section 23.5). So a vector built without a picture is a small error on the
rows the default mask reaches, not a different space.

## Families

A build writes one file per family, fl2va and ref2va, and the family is in the file name. The node picks
the file that matches the loaded model, and refuses a mismatch unless told otherwise. On 2026-11-09 the two
families' text paths measured as close to one space, and the image model's text path as fl2va's (the plan,
sections 23.1 and 23.3). The two-file scheme has not been changed on the strength of that; how each DiT
reads the same vector is not measured.

## Strength

A build prints two recommended ranges (`control_vectors.strength_ranges`). The first is for the node's
default mode, which rescales every vector to one length, and is the same for every vector. The second is
for mode `add`, which uses the vector as stored, and depends on the vector.

## Quality figures a build prints

For a file of several pairs the build reports how well the pairs agree with each other and how well two
random halves of them agree (`control_vectors.py`, the quality report in the sidecar). For one pair there
is nothing to compare and it says so. These say whether the pairs point one way; they do not say the
direction is the one you meant.
