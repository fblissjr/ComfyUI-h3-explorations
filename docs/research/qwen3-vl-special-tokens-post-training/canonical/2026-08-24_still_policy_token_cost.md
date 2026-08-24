# What each still-image policy costs in sequence length

**Status:** Measured decision input for Decision 2
**Observation date:** 2026-08-24
**Scope:** the 1,016 image-bearing rows of the accepted H3-IR candidate pool,
measured through the two installed implementations. No model was loaded;
geometry only.

Decision 2 picks the still-image preprocessing policy the v2 candidate owns. It
has been argued on fidelity. It is also a feasibility decision, because the
policy sets how many visual tokens every calibration row carries, and
`IntermediatesCache.from_dataloader` materialises one hidden state per row
before the first subgraph runs.

**Read the last two sections before acting on the first two.** The cost tables
answer what it costs to feed accepted candidate-pool files at a given cap. That is
not the same question as what cap the candidate should declare, and this file
reached a wrong conclusion once by treating them as one.

Producer: [`measure_still_policy_token_cost.py`](../../../../bench/measure_still_policy_token_cost.py).
Output: [`2026-08-24_still_policy_token_cost.json`](../../../../bench/results/2026-08-24_still_policy_token_cost.json).

The two policies are executed, not reimplemented: `native` calls
`process_qwen2vl_images` with the arguments `Qwen3VL.preprocess_embed` passes,
and `constrained` calls `h3_awq_encoder._source_image_patches`, which is the
current artifact's snapshotted processor. Grid geometry depends only on height
and width, so each of the 339 distinct source dimensions is evaluated once and
mapped back to every row using it.

## The gap is a factor of six, not a detail

**MEASURED.** Merged visual tokens per image, and assembled native-H3 sequence
length per row:

| | median tokens/image | median row | p99 row | max row |
|---|---:|---:|---:|---:|
| constrained, 200,704--301,056 px | 264 | 1,855 | 3,712 | 4,226 |
| native, comfy default to 12,845,056 px | 2,304 | 9,252 | 56,212 | 67,048 |

Mean row length is 6.11x. Across the whole image-bearing population the
activation cache is 18.8 GiB constrained against 115.0 GiB native.

**MEASURED.** Capping rows to fit a budget under full native policy removes most
of the population: 848 of 1,016 rows exceed 4,096 tokens, 556 exceed 8,192, and
212 exceed 16,384.

**INFERENCE.** That last row is the trap. Codex's Decision 2 requires a
per-row token budget and forbids silently resizing into the constrained band, so
an over-budget row must be dropped rather than shrunk. Dropping by token count
selects against large source images — and the most common inventoried dimension,
2048x1152 with 571 records, costs 2,304 tokens per image on its own. A
population filtered to fit would calibrate on geometry the deployed workflow
does not use, which is the same class of defect the v2 lane exists to repair.

## The median saturates well below the native cap

**MEASURED.** Sweeping a declared `max_pixels` through the same real
implementation, over the same population:

| declared max_pixels | median tokens/image | median row | max row | cache GiB |
|---:|---:|---:|---:|---:|
| 301,056 (current artifact) | 264 | 1,855 | 4,226 | 18.80 |
| 602,112 | 576 | 2,742 | 6,951 | 27.63 |
| 1,204,224 | 1,125 | 4,303 | 11,981 | 42.04 |
| 2,408,448 | 2,304 | 5,938 | 22,498 | 66.75 |
| 4,816,896 | 2,304 | 8,165 | 34,544 | 90.79 |
| 12,845,056 (comfy default) | 2,304 | 9,252 | 67,048 | 114.99 |

**MEASURED.** Median tokens per image reaches its native value of 2,304 at
2,408,448 pixels and does not rise above it. Past that point a higher cap
changes only the tail: max row length triples from 22,498 to 67,048 and the
population cache grows by 48 GiB, while the typical reference image is already
receiving exactly the geometry full native policy would give it.

**Superseded inference.** An earlier version of this file read the saturation
point as an argument for declaring an intermediate cap around 2,408,448. That
was wrong, and the next section says why: it treated the cap as if it set
calibration cost, when the cap is a ceiling on the deployed reference envelope.
The two are different quantities. The measurements above stand; that conclusion
does not.

## The cap is a ceiling on references, and the reference envelope is 2048

The generation canvas and the reference image are two different knobs, and this
file's first conclusion confused them. **SOURCE:** the diffusers integration
declares both separately, with defaults `canvas_short_edge` 768,
`canvas_max_pixels` 1,032,192 — exactly 1344x768 — and
`reference_image_short_edge` 2048. Its image reference states that the image
"never binds the generated geometry — it is encoded at a short edge of its own,
2048 for the released checkpoint, whatever canvas the request generates at."
That independently confirms [`h3_references.md`](../../../h3_references.md),
which read the same 2048 short edge with no area cap out of sglang on
2026-08-21, and which already warns that these two knobs are constantly confused
for each other.

DiffSynth-Studio independently carries the same defaults:
`ref_image_short_edge=2048`, `ref_video_short_edge=768`, and
`ref_video_max_pixels=768*1344`. Its model-details page states that Ref2VA
training examples are loaded at native image resolution and passed through the
pipeline that performs that reference-image resize. This corroborates the
role split, but does not expose MiniMax's unpublished original training
population.

**SOURCE, checked at the primary source.** 16,777,216 is what the release
itself declares. `processor/preprocessor_config.json` in the official
MiniMax-H3 checkout at commit `d21241f0a4b3acbb34c97dae47fa417b7065e438` carries
`longest_edge` 16,777,216 and `shortest_edge` 65,536, and is byte-identical to
this repo's `vendor_config/preprocessor_config.json` after JSON normalization;
`video_preprocessor_config.json` matches too. The release also states a default
768-pixel short side for output, which is the canvas knob, and describes a
separate H3-Regenerate-2K path above it.

**UNKNOWN at the primary source, and this corrects an overclaim.** The 2048
reference short edge is **not declared anywhere in the released repository** —
not in `modular_model_index.json`, not in any config, and the release ships no
reference-preparation code to read it from.

**SOURCE, from four implementations.** All four carry the constant, and the
three serving-side ones agree on the behaviour around it:

| implementation | reference image | reference video |
|---|---|---|
| sglang, MiniMax's own serving stack (read 2026-08-21, [`h3_references.md`](../../../h3_references.md)) | 2048 short edge, upscale on, nearest-32, no area cap | canvas rule from the clip's own aspect |
| diffusers | `reference_image_short_edge` 2048 | — |
| DiffSynth-Studio ([`MiniMax-H3.md`](../../../../coderef/DiffSynth-Studio/docs/en/Model_Details/MiniMax-H3.md)) | `ref_image_short_edge` 2048, upscaling allowed, nearest-32, "no area cap applies" | `ref_video_short_edge` 768, `ref_video_max_pixels` 768*1344 |
| installed ComfyUI core | `REF_IMAGE_SHORT_EDGE = 2048`, but scaled by `min(1.0, ...)` so it **caps at** 2048 and never upscales to it | `MAX_PIXELS = 768 * 1344` |

Four implementations carrying the same constant, one of them the vendor's own
serving stack, is strong evidence about how the model is served. It remains
evidence about serving rather than a release declaration, which is the
distinction this section exists to keep.

**Note the fourth row differs in kind.** ComfyUI declares 2048 but clamps its
scale with `min(1.0, ...)`, so `ref_image_size="max"` caps at a 2048 short edge
and never upscales to it; a reference already smaller passes through untouched.
The three serving implementations upscale. That divergence is
[`h3_references.md`](../../../h3_references.md)'s subject and is fixed locally by
`MiniMaxH3ReferenceFit(allow_upscale=True)`, which the shipped graphs leave off.
So ComfyUI corroborates the constant, not the behaviour.

**Note the video row.** DiffSynth's reference *video* short edge is 768 with a
soft area cap of exactly 1,032,192 — the canvas area, the same number diffusers
declares as `canvas_max_pixels`. So reference video is sized to the canvas while
reference image is sized to 2048 with no cap. They are not one policy, and a
calibration population mixing them inherits both.

**INFERENCE, not measurement.** With that caveat, the two candidate caps are
exactly the 2048-short-edge envelope at two aspect-ratio ceilings. The
arithmetic is exact and checkable; the reading of *why* is an inference resting
on the serving convention above, and an earlier version of this file labelled it
MEASURED, which was wrong:

| cap | equals | covers a 2048-short-edge reference to |
|---:|---|---|
| 16,777,216 (release declaration) | 2048<sup>2</sup> x 4 | 4:1, the widest ratio H3 accepts |
| 12,845,056 (comfy function default) | 2048<sup>2</sup> x 3.0625 | 3.0625:1, then it clips |

The 3.0625:1 figure is not derived here. `h3_references.md` already recorded
that the same prepared image feeds Qwen and the visual-condition tokenizer
"until 3.0625:1 — past that the Qwen ceiling shrinks one branch and not the
other". That ceiling is this cap. The coincidence is striking and both figures
are exact, but a coincidence between an sglang constant and a release constant
is not the release stating a rationale, and the release states none.

**MEASURED.** What the current artifact's 301,056 does to a reference sized the
way the release serves it. A 16:9 reference at 2048 short edge is 3648x2048,
7,471,104 pixels, 7,296 merged tokens. Under the current artifact's cap the same
reference is reduced to 301,056 pixels and about 294 merged tokens — roughly 25x
less visual detail reaching the conditioner, for the input whose whole job is
identity fidelity.

**INFERENCE.** So the cap must not be chosen from the calibration cost table
above. Raising a ceiling costs nothing when the inputs are smaller than it;
lowering it destroys reference detail whenever they are larger. The v2 candidate
should declare 16,777,216, matching the release, so that the encoder is never
the binding constraint on a reference — and the size references actually arrive
at stays owned by the reference-preparation knobs, where it belongs.

That recommendation rests on the release declaration alone, which is SOURCE at
the primary repository. It does not depend on the 2048 reading being right. If
the reference short edge later turns out to be something else, 16,777,216 is
still the number the release declares, and it is still a ceiling rather than a
target.

**SOURCE.** Those knobs are separate from this decision and are already
documented: `ref_image_size` and `MiniMaxH3ReferenceFit(allow_upscale=...)`.
The shipped graphs contain both upscale-enabled and no-upscale reference paths;
there is no universal graph default to infer. The accepted v2 primary policy is
`max` with no upstream upscale, with a separately named 2048-short-edge stress
stratum.

## Bounds on this measurement

- Visual token counts are measured. Row sequence lengths are **assembled** —
  prompt and `"<Picture i>: "` label tokens come from the installed
  `MiniMaxH3Tokenizer`, then combined with the measured visual blocks according
  to the presentation in `minimax.py`. They are not captured launcher
  sequences, and a corrected preflight must re-derive them from the instrumented
  path rather than cite these.
- Cache figures cover all 1,016 image-bearing rows. A selected calibration
  population is smaller and its cache scales with the rows actually chosen; do
  not read these as a run budget.
- Nothing here measures VRAM, wall-clock, or whether any given cap runs on the
  RTX 4090. The vision tower attends within each image as one segment, so a
  cap near 12,845,056 puts roughly 50,176 patches in a single attention
  segment; that cost is unmeasured and is not asserted either way.
- Video rows are excluded. Their geometry is owned by the video policy and the
  reference node, not by this decision.
- The source dimensions are read directly from the real files named by the
  pinned H3-IR candidate pool. The measurement no longer consumes any rejected
  preflight artifact.
