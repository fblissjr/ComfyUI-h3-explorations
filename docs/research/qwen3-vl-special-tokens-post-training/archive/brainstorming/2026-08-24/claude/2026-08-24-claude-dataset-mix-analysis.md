# Dataset mix analysis for v2 calibration and marker evaluation

> **SUPERSEDED — do not act on this file.** Its content is consolidated into
> [`2026-08-24-claude-to-codex-DATA-HANDOFF.md`](2026-08-24-claude-to-codex-DATA-HANDOFF.md),
> which is the version to read and act on. This file is kept as the
> correction record.

**Date:** 2026-08-24
**From:** Claude
**To:** Codex
**Status:** Analysis. Answers the six questions blocking the execution plan.
Nothing here authorizes a run.

Every count below was produced by reading the real datasets, not by trusting an
inventory. Where a figure comes from sampling rather than a full pass, it says
so.

> **CORRECTED 2026-08-24, after the owner pointed out H3-IR carries input
> references.** An earlier version of this file declared reference video and
> reference audio absent. That was wrong, and the error was inherited: Gemini's
> inventory read only `images` and `target_ir`, and H3-IR carries five more
> fields that were never opened. The corrected availability is below; the
> superseded verdict is kept in the table so the mistake is legible rather than
> quietly rewritten.

## H3-IR carries the role structure already — it was never read

**SOURCE**, from the raw `train.jsonl` rather than the inventory. Every row
carries `channel`, `videos`, `audio_timeline`, `has_independent_audio`,
`has_video_audio_track`, `media_sha256`, `messages`, `license` and
`redistribution_allowed`. The inventory used two of these.

**MEASURED** over all 1,110 rows:

| `channel` | rows |
|---|---:|
| `image_only_low` | 668 |
| `image_only_high` | 238 |
| `image_audio` | 102 |
| `text_only` | 82 |
| `video_only_single` | 12 |
| `image_video_no_audio` | 8 |

- **20 rows carry genuine input reference video.** 19 MP4s ship in the repo.
  Spot-checked three: 1280x720 @30fps and 720x1280 @24fps, 4--5 s, all with AAC
  tracks, all decoding. These are *inputs*, not H3 outputs.
- **120 rows carry `has_independent_audio`** with a full `[AUDIO_TIMELINE]`
  block — `<Audio 1>` source labels, duration, verbatim and aligned transcript,
  soundscape caption. **18** additionally carry a video audio track.
- **132 rows explicitly declare a Picture as a first or last frame** — real
  keyframe-shaped material, identifiable by pattern rather than heuristic.
- All 1,110 rows are `cc0-1.0` with `redistribution_allowed: true`.
- `media_sha256` is supplied per file by the dataset, so dedup keys need no
  re-derivation.

**The audio nuance, and it is favourable.** No audio files ship —
`media/audio*` is empty and `media_sha256` lists only images and videos. So
those 120 rows have no waveform. For **encoder calibration that does not
matter**: per the native contract, a reference audio contributes `"<Audio j>: "`
to Qwen and no tensor at all. Audio-bearing rows are therefore fully usable for
AWQ calibration on the label alone. They are *not* usable for a full-pipeline
arm, where the DiT's `audio_vae` needs a waveform and `ref_audio_t` advances the
rotary clock.

**`channel` replaces the rejected task classifier.** The first preflight
inferred task type from prose patterns. The dataset states it.

## Superseded headline: two of four roles have no honest source at all

| role | genuine source material | verdict |
|---|---|---|
| reference still, single | H3-IR, canvas-preview | **available and diverse** |
| reference still, ordered multi | H3-IR (1--9/row), canvas-preview (1--5/row) | **available** |
| keyframe (FL2VA) | ~~9 prompts~~ **132 rows declare a Picture as first/last frame** | **available** |
| reference video | ~~none~~ **H3-IR, 20 rows, 19 MP4s** | **available, small** |
| reference audio | ~~none~~ **H3-IR, 120 rows, labels only** | **available for encoder calibration** |
| marker: dialogue | H3-IR 495 rows w/ refs, avatar_500 500 rows | **available** |
| marker: lyrics | 28 pairs, none reference-bearing | **inadequate** |
| marker: caption | nothing | **absent** |
| marker: cutoff | nothing | **absent** |

**That verdict is withdrawn.** It was true only of the sources the inventory had
opened. It remains true that the Malcolmrey clips and avatar_500's videos are
generated **outputs**, and relabelling either as an input reference would be the
defect the first preflight was rejected for — but H3-IR ships 19 genuine input
MP4s, and that is a different thing entirely.

What survives from the original verdict is a scale problem rather than an
absence: 20 video rows and 132 keyframe-shaped rows are small populations. They
are enough to *represent* those roles in a calibration mix; they are not enough
to make either role a large stratum, and a holdout split has to come out of the
same 20.

## Source-by-source, measured

**`StellarVoyager/H3-IR` — 1,110 rows. The calibration backbone.**
Images per row 0--9 (168 single, 269 two, 282 three, tailing to 7 rows of nine).
3,101 image records, **339 distinct dimensions**, 0.09--18.66 MP, median 2.36.
Aspect classes: 1,815 16:9-ish, 626 portrait, 372 square, 155 wide, 133 other
landscape. 1,028 rows carry `subject_definitions:` and `retention_analysis:`.
495 rows carry `<d>` and `<Picture n>` in the same prompt.

This is the only source with real geometric diversity, and it is enough to build
role strata from.

**`marcuskwan/minimax-h3-canvas-preview30-20260811` — 30 requests. New, and
worth taking.** All `task: ref2va`, 76 conditions all `role: reference, type:
image`, 1--5 images per request, every target 1344x768. Ships the actual input
PNGs under `bundle/inputs/`, plus per-request JSON, prompts, and rewrite
receipts. Sampled 24 of 76 images: 15 distinct dimensions, 0.11--3.24 MP, median
1.57, spanning 16:9, wide, square and other landscape.

This is the cleanest Ref2VA material available — genuine inputs with recorded
request structure, not reconstructed. It carries **zero markers of any family**,
so it is calibration material, not marker-evaluation material.

**`oakmindai/minimax_h3_avatar_500` — 500 rows. Derive, do not calibrate.**
Prompts are excellent and correctly formed, with dialogue markers placed inline
in reference-bearing IR. The images are not: **500/500 are exactly 1024x1024**,
median **74% of the frame near-black** (44 of 50 sampled over half empty), every
one a full-body character isolated on a flat backdrop, all synthetic
(Z-Image-Turbo, 8 steps, per its README).

One aspect ratio and a near-empty frame is the wrong activation distribution for
AWQ scale selection, and it collapses every geometry stratum to a single point.
Its videos are pipeline outputs. **Contamination is confirmed, not inferred:**
the reconstruction claimed rows 0--15 fed the first calibration, and hashing the
real dataset against its recorded `prompt_sha256` gives 16/16 match.

**`kaiw7/jav-minimax-h3` — 10,140 rows. Not usable.** JavisBench t2va
generations at 1344x768. Zero markers, zero references, text-only by
construction. Its response payloads record `'model': 'sora-2'`, so its
provenance as H3 output is not established by the metadata.

**Malcolmrey local — 1,552 rows.** Generated outputs. Usable as *prompt* source
only, and only if relabelled honestly.

## 1. Calibration data versus marker-evaluation data

They must be separate populations because they optimize for different things,
and the temptation to merge them is strong since both want marker-bearing
reference prompts.

**Calibration** wants the activation distribution the deployed path produces:
geometric diversity, realistic frame content, honest role mix. Marker presence
is incidental — it should reflect production frequency, not be maximized.

**Marker evaluation** wants matched contrastive pairs per family, where the only
thing that varies is the marker representation. Geometric diversity is a
nuisance variable there and should be *held constant*, not maximized.

Concretely: a good calibration row is a real 3-image Ref2VA request at an
unusual aspect ratio. A good marker-evaluation pair is the same prompt twice with
`<d>[English] …</d>` versus "silently", on identical media. Neither is a good
example of the other.

**They must also be disjoint by media**, not just by prompt. A marker pair whose
image was in calibration measures the artifact on something it was fitted to.

## 2. Honest availability

Stated above, and corrected. All four reference roles have a source; two of them
are small. The binding constraints are now:

- **reference video: 20 rows.** Enough to represent the role, not enough for a
  large stratum, and a holdout has to come from the same 20. H3-IR's videos are
  4--5 s at 24 and 30 fps, so the 24 fps normalization path and the
  truncate-and-snap-to-`17n+5` path both get exercised.
- **reference audio: 120 rows, labels only.** Fully usable for encoder
  calibration, unusable for any full-pipeline arm.
- **keyframes: 132 rows** identifiable by an explicit "is the first/last frame"
  declaration in the IR rather than by heuristic.
- **markers beyond dialogue: still genuinely absent.** Caption, cutoff and
  lyrics have no reference-bearing coverage in any source. This is the one gap
  the corrected reading does not close.

## 3. Proposed role strata

Sized so each stratum is honestly fillable from what exists. Percentages are
proposals, not measurements.

| stratum | source | geometry role |
|---|---|---|
| single-image Ref2VA | H3-IR, canvas-preview | the common case |
| ordered multi-image, 2--3 | H3-IR, canvas-preview | binding across `<Picture i>` |
| ordered multi-image, 4--9 | H3-IR | long-sequence, worst-case tokens |
| wide/tall extremes | H3-IR wide + portrait | exercises the cap and the 3.0625:1 Qwen ceiling |
| small-source | H3-IR sub-0.5 MP | the no-upscale floor under `max` |
| dialogue-marker Ref2VA | H3-IR 495 | markers at production frequency |
| vendor-upscale stress | any of the above with `allow_upscale=True` | the declared stress stratum |

The wide/tall and small-source strata matter more than their share suggests:
they are the only rows that exercise the geometry where the encoder cap binds
differently, and avatar_500 cannot contribute to any of them.

Token distributions should be *reported from the built manifest*, not targeted
in advance — the first preflight's error was picking bucket counts first.

## 4. Deduplication and separation

Keep the first preflight's hash scheme (normalized prompt, ordered media,
combined key); it survived independent checking. Add:

- **cross-source dedup.** H3-IR and canvas-preview may share stock imagery;
  nothing has checked media hashes across sources.
- **calibration/evaluation disjoint by media hash as well as prompt hash**, with
  a rejection manifest giving a reason per excluded row.
- **avatar_500 rows 0--15 excluded from every evaluation arm** by ID, now that
  they are confirmed.
- **near-duplicate detection**, not just exact. The contrastive file collapsed
  660 rows to 119 distinct pairs; exact-hash dedup would have called all 660
  unique.

## 5. Gaps requiring owner-authored material

1. ~~Reference video~~ — **withdrawn**, H3-IR ships 19 input MP4s. Authoring is
   only needed if 20 rows proves too few to both calibrate and hold out.
2. ~~Reference audio~~ — **withdrawn for encoder calibration**; 120 label-only
   rows suffice there. Still needed for any full-pipeline arm, where a waveform
   is required.
3. ~~Keyframe pairs~~ — **withdrawn**, 132 rows declare one explicitly.
4. **Caption, cutoff and lyric markers in reference-bearing prompts** — zero
   coverage across every dataset checked. Under per-family authorization these
   three families are currently unevaluable. **This is the only real gap.**

The `internal/prompts/` system prompts are the right template for (4): they
already combine markers with reference labels in correct IR form and are
owner-authored. avatar_500's prompt structure is a second good model.

If (1)--(3) cannot be sourced, the honest move is to declare those roles out of
scope for v2 and say so in the manifest, rather than substitute.

## 6. What Gemini inventory can be salvaged

**Salvageable:** the raw per-row inventory fields for H3-IR — dimensions,
SHA-256s, prompt hashes, marker counts. I re-derived several against the real
files and they hold.

**Not salvageable:** every video row's semantics (outputs labelled as input
references, with fabricated two-frame counts and `[0.0, 0.5]` timestamps); the
task-type classifier, which is a prose heuristic; the selection logic, which
takes source quotas rather than stratifying; the row-level trace, built on random
frames; and the audit report, whose gates are unconditional.

**Missing entirely, and it matters:** the inventory covers H3-IR and Malcolmrey
only — 2,662 rows. The first calibration also drew 16 rows from avatar_500, so
Phase A never inventoried one of the three sources the run it reconstructed
actually used. Any replacement inventory must enumerate its sources rather than
inherit that list.

## What this analysis does not settle

- No proposed population size. That follows from the feasibility measurement I
  still owe, not from the mix.
- No claim that these strata improve anything. They are chosen to be
  representative and honestly fillable; whether the mix helps is what the
  held-out benchmark is for.
- Canvas-preview geometry is from 24 of 76 sampled images.
- I have not verified that canvas-preview's input PNGs decode cleanly at full
  population, nor checked its licence for redistribution.
