# Owner-authored H3 marker corpus: construction brief

**Date:** 2026-08-24
**Status:** Accepted authoring direction; corpus design, not training
authorization. Canonical gates still apply.

## Decision

We should write the missing marker evaluation data ourselves. The problem with
the retired synthetic artifact recorded in the
[`special-token-prototype` archive](../archive/rejected/special-token-prototype/README.md)
is not that synthetic prompts
are categorically invalid. Its generator changed the requested scene between
arms, produced extensive duplication, left many rows without a contrast, and
did not cover the marker families and reference roles this project needs.

[`internal/prompts/2026-08-22_office_refs_fr_subs.md`](../../../../internal/prompts/2026-08-22_office_refs_fr_subs.md)
is a strong structural example: bound reference identities, multilingual
dialogue, translated inline captions, explicit shot timing, speaker identity,
silence and mouth-closure controls, retention rules, camera behavior, and a
judgeable sound brief coexist in one valid Ref2VA prompt. Do not reuse its
characters, setting, dialogue, shot sequence, or exact wording. Derive new
scene specifications that preserve those validation properties.

## Two disjoint corpora, in this order

1. **Frozen evaluation corpus.** Author, validate, hash, and freeze the scenes,
   reference assets, prompts, token traces, render settings, and seed sets
   before looking at any training result.
2. **Candidate training/development corpus.** Author separately only after the
   evaluation split is frozen. Keep it disjoint by scene specification, spoken
   and visible strings, reference-media hashes, and prompt hash. Its existence
   does not authorize a trainer or training run.

Do not split after expanding marker arms: every arm derived from one scene must
remain in the same split.

## Author one scene specification, compile every arm

Do not hand-write three semantically drifting prompts. Store one canonical
scene specification containing:

- task and reference roles;
- ordered media and hashes;
- subject definitions and retention intent;
- shots, cuts, keyframe times, actions and camera behavior;
- speaker, language, exact audible words and delivery;
- exact visible caption text and layout intent;
- lyrics, music role, and cutoff intent where applicable;
- explicit silence and mouth-closure requirements;
- canvas, duration, frame count and reference-sizing policy; and
- marker spans as typed semantic fields rather than ad hoc prose fragments.

A deterministic compiler should render the scene into the compared arms:

1. **Release-ID arm:** exact released marker spelling through the patched
   tokenizer, resolving to IDs 151669--151675.
2. **True legacy-BPE arm:** the same serialized prompt text through an actual
   unpatched tokenizer. Never simulate this by adding spaces around marker
   characters.
3. **Stripped-marker arm:** remove only the marker strings. Preserve the text
   inside paired markers and every other character of the scene brief.

The positive and contrast must request the same scene. The rejected generator's
dialogue “positive” requested audible speech while its “contrast” requested
silent reading; that tests two different briefs, not marker representation.

For caption experiments, retain an explicit prose request for subtitle position
and appearance in every arm. The marker-wrapped caption string, the identical
bare string, and the prose-only contribution must be separable so marker effect
is not confused with repeating caption text. The existing owner-reviewed
caption arms demonstrate the required factorial logic; reuse the logic, not the
scene or wording.

## Marker-family grammar

Author each family only where it has a coherent, judgeable target:

- **Dialogue:** `<d>[Language] exact audible words</d>`. Keep speaker, delivery,
  timing, silence, mouth closure, and audiovisual brief constant across arms.
- **Caption:** `<|caption_start|>visible text<|caption_end|>` inline beside the
  corresponding event, with the visible-text request also established in
  prose. Include same-language and translated-subtitle cases.
- **Lyrics:** `<|lyrics_start|>` around a sung run, with one or more valid
  `<d>` lines inside it. Compare sung delivery against the same sung brief, not
  against spoken narration.
- **Cutoff:** `<|cutoff|>` directly after `</d>` for a deliberately incomplete
  vocal event at the video boundary. Keep the incomplete line and end timing
  identical when the marker is stripped or fragmented.

Authorization remains per family. A dialogue result says nothing about caption,
lyrics, or cutoff, even when one authored scene contains more than one family.
Interactions such as dialogue plus translated captions should be their own
declared stratum, not pooled into either single-family estimate.

## Multimodal coverage

Reference-bearing dialogue is the primary population, with text-only retained
as a regression stratum. Author coherent coverage across the roles for which
real, authorized media exists:

- single reference still / identity;
- ordered two- or multi-still Ref2VA;
- first/last/arbitrary keyframes at target-canvas geometry;
- mixed keyframe plus Ref2VA requests;
- standalone reference audio paired with a referenced speaker;
- reference video and video soundtrack only when the media is a genuine input
  reference, never a generated output relabeled as one; and
- text-only controls.

Not every marker belongs in every modality. Do not fill a Cartesian matrix with
nonsensical marker use. Instead, name the missing cells and why they are not
part of that marker's intended workload.

Use multiple new scene archetypes, identities, environments, languages, shot
structures, and marker positions. Variation must come from curated scene
specifications or deterministic, reviewed vocabularies—not an unseeded random
combination generator whose nominal row count hides duplicates.

## Required provenance and controls

Every compiled arm must record:

- scene-spec ID and content hash;
- split and marker family;
- exact prompt bytes and hash;
- tokenizer identity/config hash;
- full token-ID stream and token tags;
- marker spans and IDs;
- ordinary-BPE `<Picture i>` / `<Video i>` / `<Audio i>` label positions;
- vision spans, grids and ordered media hashes;
- an explicit ordinary-text alignment map across arms;
- reference role and both stages of effective geometry;
- render settings and matched seed-set ID; and
- the exact allowed textual diff from the canonical scene.

Assertions must fail when:

- an arm changes prose, media, settings, or target behavior beyond its declared
  marker transformation;
- the legacy arm does not use a genuinely unpatched tokenizer;
- the release IDs differ from 151669--151675;
- a stripped paired marker loses or changes its enclosed text;
- marker grammar or nesting is invalid;
- an ordinary text alignment is claimed where none exists;
- prompt/media hashes overlap the frozen evaluation and training splits; or
- a reference-video row resolves to generated output media.

Run the existing prompt and graph preflight checks rather than inventing a
parallel grammar. A new corpus-specific check is justified only for an escaped
condition those checks cannot observe, such as semantic arm drift from one
scene specification.

## What authoring this corpus does not decide

It does not establish that any token needs post-training, choose a loss, make
AWQ autograd work, or authorize a training run. Its first purpose is to make the
no-training marker evaluation capable of answering the owner's actual
multimodal question. Only a demonstrated per-family deficit can promote the
separate training-design lane.
