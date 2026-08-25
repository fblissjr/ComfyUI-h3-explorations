# Marker evaluation corpus

**Status:** Seed set, not the frozen evaluation corpus. Freezing is the owner's
act and requires coverage this does not yet have.
**Brief:** [`owner_authored_marker_corpus.md`](../../docs/research/qwen3-vl-special-tokens-post-training/canonical/owner_authored_marker_corpus.md).
It is the authority; nothing here restates a rule it owns.

## What is here

`scenes/` holds one JSON scene specification per scene.
[`compile_marker_corpus.py`](../compile_marker_corpus.py) turns each into every
marker arm; `compiled.json` is that output.
[`check_marker_corpus.py`](../check_marker_corpus.py) holds the result to the
assertions the brief names and carries its own violation arm.

```
bench/compile_marker_corpus.py     # scene spec -> arms, deterministic
bench/check_marker_corpus.py       # the brief's assertions, plus --violation-arm
```

## The one design decision worth knowing before you read the code

**Arms are derived, never written.** A scene is serialized to prompt text
exactly once, and every arm is a declared transformation of that single string.
The rejected generator's central defect was that its dialogue "positive" asked
for audible speech while its "contrast" asked for silent reading — two briefs,
not one marker. Here that is not checked for, it is unrepresentable: there is
one piece of prose and no arm may author its own.

**An arm is a triple, not a prompt:** prompt bytes, tokenizer identity, model
transform. That is what lets the mean-initialised-rows arm exist in the same
corpus — it differs from the release-ID arm only in a model-side change. The
compiler names that transform and never applies it. **Nothing in this repo
writes a token row.**

## Coverage, and what is deliberately absent

The compiled document carries a `declared_missing_cells` list with a reason per
cell. Two are missing because the media has not been looked at, and authoring
speaker or scene prose against unheard audio or unseen video would manufacture
exactly the prompt/media mismatch this corpus exists to avoid. One is not a gap
at all: `<|cutoff|>` marks an incomplete vocal event at the video boundary, so a
text-only cutoff row is nonsensical rather than missing.

Media is drawn from pool families untouched by any v2 calibration or holdout
bundle, so an evaluation of the v2 candidate on this corpus is not reading its
own calibration data back.

## `stratum` is not `marker_families`

`marker_families` says which markers occur. `stratum` says what a result from
that row may speak about. They differ because a lyrics run necessarily wraps a
`<d>` block, so every lyrics scene contains the dialogue marker — and reading
such a row as dialogue evidence would pool an interaction into a single-family
estimate, which the brief forbids.
