# Canonical native-H3 presentation and comparison contract

**Status:** Authoritative implementation contract
**Last updated:** 2026-08-24

## Native presentation

The installed native implementation is the authority. MiniMax H3 conditioning
is not chat-templated. The sequence is assembled as follows:

- **T2VA:** raw prompt text.
- **FL2VA / image inputs:** `"<Picture i>: "`, then a vision-start token, the
  expanded vision block, and a vision-end token. The prompt follows the image
  blocks.
- **Ref2VA:** reference items remain in request order. Each type has an
  independent one-based counter:
  - image: `"<Picture i>: "` plus its vision block;
  - audio: `"<Audio j>: "` only; reference audio does not enter Qwen as a
    tensor; and
  - video: `"<Video k>: "`, followed by two-frame temporal blocks, each
    preceded by `"<T.T seconds>"` using the installed one-decimal formatting.
- An odd number of sampled video frames is repeat-padded with the last frame
  before two-frame block construction.
- The prompt text follows all ordered reference items.
- H3 text positions carry token tag 1. Each entire vision span, including its
  flanking vision-start and vision-end tokens, carries token tag 0.
- H3 returns the unnormalized state after language layer index 49.

Primary implementation:

- [`minimax.py`](../../../../../../comfy/text_encoders/minimax.py)
- [`reference_conditioning.py`](../../../../reference_conditioning.py)
- [`check_reference_contracts.py`](../../../../bench/check_reference_contracts.py)

Any calibration or benchmark builder must compare its token IDs, ordered media
records, vision spans, `grid_thw`, timestamps, and token tags against this path.
Equivalent-looking prose is not sufficient.

## Media-role sizing and Qwen processing are separate stages

Do not collapse the size at which H3 prepares conditioning media with the
later pixel bounds inside Qwen's image/video processor. Three independent
serving implementations converge on different upstream geometry by role:

- **FL2VA keyframes** are placed on the resolved target H3 canvas. MiniMax's
  official README describes H3-Base as the 768p model with a default
  768-pixel short edge; the normal landscape canvas is 1344x768 (1,032,192
  pixels).
- **Ref2VA still-image references** do not inherit that canvas. SGLang,
  Diffusers, and DiffSynth-Studio independently implement a 2048-pixel short
  edge, upscaling included, nearest-32 geometry, and no H3 canvas-area cap.
- **Ref2VA video references** follow the 768-short-edge / 1,032,192-pixel
  canvas rule before duration-aware Qwen sampling and two-frame presentation.

Direct implementation evidence:

- [`reference_encoding.py`](../../../../coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py)
- [`before_encoder.py`](../../../../coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py)
- [`minimax_h3_audio_video.py`](../../../../coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py)
- [`DiffSynth MiniMax-H3 model details`](../../../../coderef/DiffSynth-Studio/docs/en/Model_Details/MiniMax-H3.md)
- [`MiniMax-H3/README.md`](../../../../coderef/MiniMax-H3/README.md)

The DiffSynth training path also hands Ref2VA images to this same role-aware
pipeline at native file resolution and lets its `ref_image_short_edge=2048`
stage prepare them. That is corroborating implementation evidence, not proof
of MiniMax's unpublished original training population.

At Qwen3-VL's patch-size-16 / spatial-merge-2 geometry, a 1344x768 image emits
`(1344 / 32) * (768 / 32) = 1008` merged visual tokens. This is a required
FL2VA/canvas control. It is not evidence that Ref2VA still images should be
reduced to the same area; the three serving implementations explicitly keep
those roles separate.

## Qwen processor policies are separate variables

Do not use one phrase such as “the H3 image/video processor” for all of these:

1. **Installed ComfyUI native still-image policy.** The normal Comfy Qwen image
   path uses its code defaults: 3,136--12,845,056 pixels and bilinear
   `F.interpolate` on float input.
2. **Release-declared still-image policy.** The released
   `preprocessor_config.json` declares 65,536--16,777,216 pixels and is owned by
   that processor configuration rather than by the installed Comfy code
   defaults.
3. **Current AWQ still-image policy.** The AWQ loader installs the artifact's
   snapshotted image processor on that CLIP instance. Its declared bounds are
   200,704--301,056 pixels; the current adapter also crosses a uint8
   round-and-clamp boundary and uses the snapshotted processor's resize path.
4. **Release video policy.** Owned by the released video processor
   configuration and applied clip-wide before native two-frame presentation.
5. **Encoder-artifact video policy.** Owned by the selected encoder's
   snapshotted video configuration. The reference node exposes it as
   `video_policy="encoder"`.

The current release and encoder video configurations may agree today. Their
ownership remains separate so future divergence cannot silently change a
comparison.

The installed-native and release-declared still policies agree on pixel bounds
for inputs between 65,536 and 12,845,056 pixels, but they are not one policy.
Their boundary behavior and processing ownership differ. These bounds apply
after the role-specific preparation above. The source-verified details are
recorded in
[`2026-08-24_calibration_input_seam.md`](2026-08-24_calibration_input_seam.md).

The observed 264--289 merged visual-token range is limited to representative
still images processed under the current AWQ still-image bounds. It is not a
universal H3 image count and says nothing by itself about multi-image or video
sequence totals.

## Mandatory benchmark split

Every BF16-versus-W4 result must identify which of these two questions it
answers.

### A. Weight-only numerical comparison

This comparison isolates weight quantization drift:

- force identical native-H3 token IDs into BF16 and W4;
- force identical reference ordering and timestamp text;
- force identical decoded media and identical visual preprocessing into both;
- verify matching pixel/patch input hashes, `grid_thw`, vision-span offsets,
  total sequence length, and token tags before comparing outputs;
- use the same output tap: raw layer 50, with no final norm or LM head; and
- capture arms sequentially if required by memory, recording enough provenance
  to reject mismatched captures.

For the current W4 artifact, the primary isolated comparison uses the current
artifact's processor policy for **both** BF16 and W4. For a v2 artifact, repeat
the isolation using the v2 artifact's declared processor policy for both arms.
A further common-policy comparison between W4 candidates may be useful, but it
must be named separately.

If token IDs, grids, tags, or output shapes differ, a direct rowwise cosine or
MSE is invalid and the comparison must stop.

### B. Deployed-path comparison

This comparison measures the combined difference a user receives:

- BF16 runs through its release/native processor policy;
- each W4 artifact runs through its own declared deployed processor policy;
- input geometry, token-count, and alignment differences are reported as
  results rather than hidden; and
- text-token or semantic-span alignment is required when sequence lengths
  differ. Flattening unequal or semantically misaligned sequences into one
  cosine is forbidden.

This arm combines preprocessing and weight effects. It must not be described as
pure quantization drift.

## Minimum numerical reporting

For aligned weight-only captures, report at least:

- flattened cosine similarity;
- tokenwise cosine distribution, including minimum and declared percentiles;
- MSE and RMSE in a declared accumulation dtype;
- relative L2;
- activation RMS for both arms;
- results split over all rows, text-tag rows, vision-tag rows, and marker
  positions where present; and
- sequence length, grid, media, tokenizer/config, checkpoint, and code hashes.

These numbers characterize selected cases. They do not by themselves prove
render equivalence, perceptual quality, or harmless calibration shortcuts.
