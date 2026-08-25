# Calibration geometry is a composition, not a single policy

**Status:** Source-verified constraint on the v2 calibration strategy
**Observation date:** 2026-08-24
**Scope:** installed ComfyUI reference path, `h3_awq_encoder.py`, the current W4
artifact's config snapshot, and the repo's own measured upscale cost. Code read
and arithmetic run; no model loaded.

Decision 2 has been discussed as though the v2 candidate picks *one* still-image
policy. It does not. What geometry a reference actually arrives at is the
composition of two independently-owned stages, and a calibration manifest built
against either one alone describes a distribution the deployed path does not
produce.

## The two stages

**Stage 1, upstream sizing.** Owned by the reference nodes, before anything
reaches the encoder. **SOURCE:** three distinct knobs, two of which are no-ops on
typical input:

| knob | direction | effect |
|---|---|---|
| `ref_image_size="match"` (core default) | down only | to the generation's pixel area, about 1,032,192 |
| `ref_image_size="max"` | down only | to a 2048 short edge — a ceiling, not a target |
| `MiniMaxH3ReferenceFit(allow_upscale=True)` | up only | to a 2048 short edge — the floor, a separate node |

**MEASURED**, through the real code paths, on resolutions the deployed workflow
actually receives:

| source | `match` | `max` | with upscale |
|---|---|---|---|
| 1280x720 | 1280x704, 0.90 MP | 1280x704, 0.90 MP | 3648x2048, 7.47 MP |
| 1920x1080 | 1344x768, 1.03 MP | 1920x1088, 2.09 MP | 3648x2048, 7.47 MP |
| 1024x1024 | 1024x1024, 1.05 MP | 1024x1024, 1.05 MP | 2048x2048, 4.19 MP |
| 768x1344 | 768x1344, 1.03 MP | 768x1344, 1.03 MP | 2048x3584, 7.34 MP |
| 3840x2160 | 1344x768, 1.03 MP | 3648x2048, 7.47 MP | 3648x2048, 7.47 MP |

`match` and `max` are identical below a 2048 short edge, so on most production
input they do the same nothing; `match` discards real pixels above about 1 MP;
only `allow_upscale` ever enlarges.

The same composition worked through stage two as well, for three real sizes
under the v1 cap and the v2 release bounds, with the 2026-08-25 owner
decision on the upscale, is in
[`h3_references.md`](../../../h3_references.md), "Both stages on three real
sizes".

**Stage 2, the encoder cap.** Owned by the artifact.
`h3_awq_encoder.py::_image_processor()` builds a `Qwen2VLImageProcessor` from the
artifact's `processor_config.json` snapshot and
`install_source_processors` binds it to the CLIP instance. The current artifact
declares 200,704--301,056 px; the release declares 65,536--16,777,216.

## Why the composition matters more than either stage

**MEASURED.** Stage 2 is the binding constraint today. Whatever stage 1
produces, the current artifact clamps the conditioner's view to roughly 294
merged tokens. A reference sized generously upstream still reaches layer 50
crushed, and nothing reports it.

**SOURCE.** The two stages also do not clamp the same consumers. The VAE
branch takes stage 1's output; the Qwen branch takes stage 1's output *through*
stage 2. So the same reference can reach the DiT at full requested geometry and
the conditioner at a fraction of it. `docs/h3_references.md` already records a
related split, where past 3.0625:1 the Qwen ceiling shrinks one branch and not
the other.

**MEASURED, and it predates the cap.** `docs/h3_references.md` records that
upscaling one reference cost 6,144 tokens rather than the 3,072 the VAE table
predicts, "because the conditioner's vision blocks read the resized image too".
That measurement was taken on an encoder without the 301,056 cap. Under the
current W4 artifact the conditioner half of that cost would not appear — and
neither would the benefit.

**MEASURED.** `workflows/h3_config.py` records the upstream cost of the floor
knob: `allow_upscale=True` 89.1s against `False` 18.1s at the same seed, 4.9x
wall clock, with a side-by-side finding no identity difference.

**Correction, 2026-08-24.** An earlier version of this file said the shipped
graphs set `False`. There is no universal upstream default: six graphs set
`True`, including `h3_image_ref_plus_text_to_video`, and 28 set `False`. Any
calibration population or benchmark arm built on "the shipped default" is built
on something that does not exist; the policy has to be named per graph.

## Consequences for the v2 calibration strategy

1. **The manifest must record both stages per row, not one policy.** A row's
   geometry is `stage1(source, knobs)` then `stage2(cap)`. Recording only the
   declared cap, or only the source dimensions, leaves the actual encoder input
   underdetermined. This is a requirement on the row-level trace schema.

2. **Calibrate against the intended serving configuration, not today's
   defaults.** If the deployed graphs move from `match` to `max`, or turn the
   floor on, the geometry distribution changes and a manifest built beforehand is
   calibrated for a path that no longer exists. **So the upstream sizing decision
   has to be made before the calibration manifest is built**, not after. This is
   a sequencing constraint the plan did not previously carry.

3. **Declaring the release cap for v2 changes nothing on its own.** With
   `match` as the upstream default, references arrive at about 1 MP regardless
   of whether the encoder would accept 16.8 MP. Raising the ceiling only matters
   in combination with an upstream policy that produces geometry above the old
   cap. The two must be decided together.

4. **The role split survives all of this.** Keyframes arrive at canvas geometry
   because the DiT places them on the target timeline sharing the target grid;
   references get their own slot and their own grid. See
   [`2026-08-24_keyframe_vs_reference_positioning.md`](2026-08-24_keyframe_vs_reference_positioning.md).
   So keyframe rows and reference rows are different geometry strata in the
   manifest, and that is architectural rather than conventional.

5. **Still-image sizing is per reference, not per request.**
   `MiniMaxH3AppendRefImage` stores `size_policy` on each
   `RuntimeImageReference`, and `_compile_reference_records` computes each
   image's resize and latent grid independently. Mixed reference grids are
   structurally supported: each reference receives its own sequential DiT slot
   and its own Qwen vision block. This permits deliberate per-image budgeting,
   such as retaining more real detail for an identity reference than for a
   background or style board. It is also an implicit influence/cost knob because
   a larger reference contributes more packed rows; it must not be described as
   changing sharpness alone.

6. **The typed node exposes `max` one node upstream.**
   `MiniMaxH3ReferenceConditioning` does not duplicate a `ref_image_size`
   widget. Its `MiniMaxH3AppendRefImage` inputs already carry independent
   `size_policy="match"|"max"` values into the compiler. Every currently
   generated typed-reference graph inspected on 2026-08-24 uses `max` at that
   append seam; `MiniMaxH3ReferenceFit.allow_upscale` varies by graph. A scan
   that looks only for `ref_image_size` on the final conditioner therefore
   reports a false absence.

### Open ownership defect in the upstream fit warning

**SOURCE.** `reference_fit.py::qwen_max_pixels()` still introspects Comfy's
native `process_qwen2vl_images` default. It cannot identify the selected CLIP:
`MiniMaxH3ReferenceFit` has no CLIP input, while the selected AWQ budget is bound
later to the CLIP instance. Under an AWQ graph, `keep_towers_matched=True` can
therefore compare against the wrong ceiling and fail to report the VAE/Qwen
split.

Unconditionally replacing that native value with
`h3_awq_encoder.source_image_pixel_bounds()` would create the inverse bug for a
native or different encoder. The selected encoder's effective bound has to be
carried to the point that owns both consumers, or the typed conditioner has to
derive it from its actual `clip` before VAE encoding.

**Resolved on the conditioner, 2026-08-25.** `MiniMaxH3ReferenceConditioning`
now derives it from its actual `clip`: the loader stamps the artifact's
declaration on the CLIP and `image_policy` / `video_policy = encoder` read it
back, so the typed path applies the loaded encoder's ceiling before VAE
encoding whichever loader built the CLIP; enforced by `bench/check_reference_runtime.py::encoder_policy_binds_to_the_loaded_clip` and its red mutations M7/M8. `reference_fit.py::qwen_max_pixels()` is
unchanged and still has no `clip`, so the fit node remains a native-path
reporter and must not be read as the AWQ ceiling.

## Fixing the encoder cap, if that is decided

**SOURCE.** The cap is two integers in the artifact's snapshot:
`image_processor.size.shortest_edge` and `longest_edge` in
`config/qwen3vl_32b_minimax_h3_w4a16_awq/processor_config.json`.

**SOURCE.** It is hash-guarded. `bench/check_h3_awq_encoder.py` hashes
`processor_config.json` against the directory's `sha256.json`, so an in-place
edit goes red. That guard is correct and should not be bypassed: the snapshot
records what the artifact *declares*, and widening it is a serving-policy
decision rather than a transcription fix.

Three ways were considered, in the order they had to be attempted:

1. **COMPLETED: for measurement, change nothing on disk.** The processor is now
   parameterized per CLIP instance, and the benchmark compared both policies on
   the same weights. No repoint or hash change occurred. The held-out result in
   [`2026-08-24_layer50_processor_policy_benchmark.md`](2026-08-24_layer50_processor_policy_benchmark.md)
   does not support widening the deployed artifact's default.
2. **NOT ACCEPTED on the measured result:** update `processor_config.json` and
   `sha256.json` together, with a provenance note recording that the serving
   policy now differs from the calibration-time processor and why. The artifact
   was calibrated at 301,056, and the wider-policy numerical gap is why this step
   remains parked.
3. **For v2**, calibrate at the declared bounds in the first place, so
   declaration and calibration agree and no note is required.

**MEASURED.** The current artifact's AWQ scales were determined from activations
at 200,704--301,056 px. Serving it at release bounds applies those scales to a
distribution far from the one that chose them. On three held-out real images,
the wider-policy BF16-versus-W4 numerical gap increased consistently. The exact
scope and limitations are owned by
[`2026-08-24_layer50_processor_policy_benchmark.md`](2026-08-24_layer50_processor_policy_benchmark.md);
do not expand that result into a render-quality claim.
