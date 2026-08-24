# Final data plan for v2 calibration, holdout, and marker evaluation

> **SUPERSEDED — do not act on this file.** Its content is consolidated into
> [`2026-08-24-claude-to-codex-DATA-HANDOFF.md`](2026-08-24-claude-to-codex-DATA-HANDOFF.md),
> which is the version to read and act on. This file is kept as the
> correction record.

**Date:** 2026-08-24
**From:** Claude
**To:** Codex
**Status:** Decision-complete on data. Absolute population size is the one open
item and it is gated on the feasibility measurement, not on the mix.

Supersedes the availability verdicts in
[`2026-08-24-claude-dataset-mix-analysis.md`](2026-08-24-claude-dataset-mix-analysis.md),
which declared video and audio absent before the owner pointed out that H3-IR
carries input references and the inventory had never opened those fields.

## The shape: three disjoint populations, not two

| population | purpose | wants |
|---|---|---|
| **calibration** | fits the AWQ scales | the deployed activation distribution: geometric diversity, honest role mix, production-frequency markers |
| **holdout** | BF16-versus-W4 benchmark | the same distribution, never seen during calibration |
| **marker evaluation** | the post-training gate | matched contrastive pairs per family, geometry held *constant* |

Mutually disjoint by **normalized prompt hash and media hash**, not by row index.
A marker pair whose image was in calibration measures the artifact on something
it was fitted to; a holdout row sharing media with a calibration row is not held
out.

## Source allocation, final

| source | role | why |
|---|---|---|
| **StellarVoyager/H3-IR** | calibration + holdout backbone | only source with real geometric diversity *and* full IR prompts; cc0-1.0, redistribution allowed; ships `media_sha256` |
| **marcuskwan/canvas-preview30** | calibration, multi-image stratum | genuine Ref2VA inputs with recorded request structure; 76 images, 1--5 per request |
| **consciousengines/h3-video-edit-showcase** | prep-path fixture only | 5 prepared clips plus 2 pre-prep originals; **no IR prompts**, `license: other` unread |
| **oakmindai/avatar_500** | prompt-structure template only | 500/500 at 1024², median 74% near-black; rows 0--15 confirmed in the first calibration |
| **Malcolmrey local** | excluded | generated outputs |
| **kaiw7/jav-minimax-h3** | excluded | t2va only, zero markers, `model: sora-2` in its own metadata |
| **contrastive_pairs_1k** | marker-eval seed, dialogue only | 119 distinct pairs, not 1,000 |

## Calibration strata

**MEASURED**, fillable counts from H3-IR's 1,028 vision-bearing rows (`channel
!= text_only`). Every row in calibration must be vision-bearing — the sequential
trace admits one modality envelope, and a text-only row cannot join a
vision-traced run.

| stratum | available | role |
|---|---:|---|
| multi-image 2--3 | 551 | the common Ref2VA case |
| dialogue-marker | 505 | markers at production frequency, not maximized |
| wide/tall extreme | 341 | the only rows where the cap and the 3.0625:1 Qwen ceiling bite differently |
| multi-image 4--9 | 297 | long-sequence, worst-case token count |
| single-image | 168 | I2VA-shaped |
| keyframe-declared | 132 | **canvas geometry, not reference geometry** |
| audio-label | 120 | `<Audio j>` presentation with no tensor |
| video-reference | 20 | two-frame blocks, timestamps, the 24fps path |
| small-source <0.5 MP | **7** | the no-upscale floor under `max` |

Two strata need explicit handling:

- **keyframe-declared rows are a different geometry class.** H3-IR states which
  picture is the first or last frame (`"<Picture 3> is the first frame of
  [Shot 1]"`), so they are identifiable by declaration rather than heuristic. A
  keyframe arrives at *canvas* geometry because the DiT places it on the target
  timeline sharing the target grid. Presenting one at reference geometry would
  calibrate on a distribution the deployed path never produces.
- **small-source has 7 rows.** Not splittable. Put all 7 in calibration and
  declare the holdout carries no small-source stratum, rather than pretend.

**Proportions, not counts.** The absolute population follows from the
feasibility measurement. Proposed shape: multi-image ~40%, single ~15%,
keyframe ~12%, wide/tall ~12%, audio-label ~10%, video ~7%, small-source the
remainder. Dialogue markers ride across strata at their natural rate rather than
forming a stratum of their own.

## Holdout

Drawn from the same strata, disjoint by both hashes. The binding constraint is
video: **20 rows total**, so a 14/6 calibration/holdout split leaves six video
rows to benchmark on. That is thin and should be stated in the report rather
than discovered in it.

The five showcase clips can extend the holdout's *media* if prompts are
authored for them, and they are the only material that exercises the reference-
prep path with both ends visible: `branch/{drone,road}/source.mp4` at 56.17s and
14.00s against their 124-frame prepared forms. That is a **fixture for the
truncate-and-snap-to-`17n+5` path**, which the audit found silently lossy, not
calibration data.

Text-only T2VA stays out of calibration entirely and serves as the held-out
regression arm, per your locked decision.

## Marker evaluation, and what must be authored

| family | reference-bearing pairs | status |
|---|---:|---|
| dialogue | 91 | usable seed |
| lyrics | 28, **none reference-bearing** | inadequate |
| caption | **0** | must be authored |
| cutoff | **0** | must be authored |

Under per-family authorization, three of seven tokens are currently
unevaluable. This is the only genuine authoring requirement left — every
reference role now has a source.

Author against `internal/prompts/`'s four external H3 system prompts, which
already combine markers with reference labels in correct IR form and are
owner-authored. avatar_500's prompt structure is a second model: it places a
dialogue marker inline inside reference-bearing IR correctly.

Marker pairs must hold media constant between arms. Their media must come from
outside both calibration and holdout.

## Rules

1. **Role classification comes from `channel`**, not from a prose heuristic. The
   dataset states it; the rejected preflight inferred it.
2. **Dedup keys come from the dataset's own `media_sha256`** plus a normalized
   prompt hash, with cross-source media comparison — H3-IR and canvas-preview
   may share stock imagery and nothing has checked.
3. **Near-duplicate detection, not just exact.** Exact hashing would have called
   all 660 contrastive rows unique when they are 119.
4. **Every excluded row gets a rejection manifest entry with a reason.**
5. **avatar_500 rows 0--15 are excluded from every evaluation arm by ID**, now
   that the 16/16 hash match confirms them in the first calibration.
6. **Both geometry stages recorded per row** — upstream sizing then the encoder
   cap — because the row's actual encoder input is their composition.
7. **No fabricated media semantics.** A generated output is never relabelled an
   input reference.

## Declared out of scope for v2

- **Reference audio as a tensor.** 120 rows give the `<Audio j>` presentation,
  which is all Qwen ever sees. No waveform ships, so any full-pipeline audio arm
  is out of scope and should say so.
- **Small-source holdout.** 7 rows, all to calibration.
- **Caption, cutoff and lyric authorization**, until pairs exist.

## The one open item

Absolute population size. It follows from the VRAM and wall-clock measurement I
owe, run against a candidate population at the locked `max`/no-upscale default —
not from the mix, and not from a number chosen in advance. The first preflight's
error was picking bucket counts before measuring; this plan gives proportions
and fillable strata so the size can be set by what the box can actually do.
