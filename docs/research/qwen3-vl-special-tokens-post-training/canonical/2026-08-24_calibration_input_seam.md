# What can reach AWQ calibration, and what cannot

**Status:** Source-verified and measured findings on the calibration seam
**Observation date:** 2026-08-24
**Scope:** the `llm-compressor` checkout at commit `8357e9459228be5831ff43f9449cdd7733d3d877`,
its virtualenv (`transformers` 5.15.1, `torch` 2.13.0+cu132,
`llmcompressor` 0.13.1.dev38+g501f432bf), the installed ComfyUI H3 path, and the
current AWQ adapter. No 32B model was loaded and no quantization was launched.

This answers the technical lead's first question in
[`2026-08-24-codex-to-claude.md`](../archive/brainstorming/2026-08-24/codex/2026-08-24-codex-to-claude.md):
whether exact native-H3 text, multi-image, and two-frame video inputs can reach
AWQ calibration without chat wrapping, first-image slicing, fallback
duplication, or processor mismatch. It is evidence about the seam, not an
accepted plan and not a preflight acceptance. The preflight artifact set stands
rejected by [`2026-08-24_awq_v2_preflight_review.md`](2026-08-24_awq_v2_preflight_review.md).

## Answer

| native H3 family | can it reach calibration unmodified | evidence |
|---|---|---|
| I2VA, single image | yes | **MEASURED** Q1a, Q2 |
| FL2VA, two images | yes | **MEASURED** Q2 |
| Ref2VA, ordered multi-image | yes | **MEASURED** Q2 |
| video reference, two-frame blocks | yes, on the image keys | **MEASURED** Q3 |
| audio reference | nothing to feed; it is a text label only | **SOURCE** [`minimax.py`](../../../../../../comfy/text_encoders/minimax.py) |
| T2VA, text-only | **not in the same `oneshot` run as any of the above** | **MEASURED** Q1a, Q1b, Q4 |

Chat wrapping, first-image slicing and fallback duplication were all choices
made by the previous run's script, not requirements of the library. The seam
already accepts preconstructed native multimodal inputs. The binding constraint
is different from any of the four the handoff named: it is that one `oneshot`
call traces the model once, and that trace fixes which modalities exist for the
whole run.

Case identifiers refer to
[`probe_calibration_input_seam.py`](../../../../bench/probe_calibration_input_seam.py);
its output is
[`2026-08-24_calibration_input_seam_probe.json`](../../../../bench/results/2026-08-24_calibration_input_seam_probe.json).
Neither is part of the archived rejected
[`gemini-preflight/`](../archive/rejected/gemini-preflight/README.md)
submission; they live under `bench/` with the repo's other measurements.

## 1. The library already accepts preconstructed native inputs

**SOURCE.** Three independent bypasses exist, in
`coderef/llm-compressor/src/llmcompressor/datasets/utils.py`:

- `coderef/llm-compressor/src/llmcompressor/datasets/utils.py::get_processed_dataset`
  returns a `Dataset` unchanged as soon as it has an `input_ids` column. The
  `TextGenerationDataset` registry, which owns every prompt-formatting path, is
  never reached.
- `coderef/llm-compressor/src/llmcompressor/datasets/utils.py::get_calibration_dataloader`
  returns a `DataLoader` verbatim if one is passed as `dataset`, bypassing
  dataset loading, sampling and collation together. `DatasetArguments.dataset`
  declares `DataLoader` in its type.
- `coderef/llm-compressor/src/llmcompressor/datasets/utils.py::_make_collate_fn`
  returns a callable `data_collator` before it considers any built-in collator.

**INFERENCE, from those three.** `apply_chat_template` in the completed run's
script was the script's own call, on the way in. No library path restores it.
No narrowing patch to `llm-compressor` is required to feed native presentation;
the narrowest correct seam is a preconstructed `DataLoader`, which also removes
the sampler and collator from the trust surface.

**SOURCE, and a correction.** Because a callable collator short-circuits
`_make_collate_fn`, `max_seq_length` reaches neither
`DataCollatorWithTruncation` nor `TextGenerationDataset`, the only two consumers
of it. In the completed run and in any run built the same way it is inert: it
truncates nothing. This refines finding 6 of the preflight review — the
`max_seq_length=2048` against 4,226-token rows is not a truncation risk, it is a
declared parameter with no effect, and the sequence-length policy has to be
enforced by the manifest instead.

## 2. The trace, not the recipe, fixes the modality envelope

**SOURCE.** `coderef/llm-compressor/src/llmcompressor/pipelines/sequential/pipeline.py::SequentialPipeline`
traces subgraphs once, from `next(iter(dataloader))`, and executes every later
batch against that one graph.
`coderef/llm-compressor/src/llmcompressor/pipelines/sequential/helpers.py::populate_concrete_args`
turns every forward parameter absent from that first batch into a concrete
constant at its default, which for `pixel_values` and `image_grid_thw` is
`None`. The traced graph then has no placeholder for them, and
`IntermediatesCache.fetch` passes only keys the graph declares.

**SOURCE.** By default the first batch is not the first manifest row.
`coderef/llm-compressor/src/llmcompressor/datasets/utils.py::LengthAwareSampler`
orders by descending sequence length, so the longest row traces.

**MEASURED (Q1a).** Trace from a vision row, then feed a text-only row: the
graph demands `pixel_values` and `image_grid_thw`, the row has neither, and it
raises `TypeError`. Loud, and therefore safe.

**MEASURED (Q1b).** Trace from a text-only row, then feed a vision row: it
returns normally with `pixel_values` and `image_grid_thw` dropped. Two batches
carrying identical token ids and completely different images produce a
bit-identical subgraph output — a max absolute delta of exactly zero. The vision
tower never ran, the DeepStack features never existed, and the `<|image_pad|>`
positions were calibrated as ordinary text embeddings.

This is the `active_plan.md` immediate stop condition "a calibration library
silently drops media", and Q1b is the configuration that produces it. It
produces no warning. A run in that state would report every manifest row as
processed and would be wrong about all of them.

**MEASURED (Q4).** The obvious bridge does not work. Giving a text-only row an
empty `pixel_values` of shape `[0, patch_dim]` and an empty `[0, 3]` grid raises
inside the vision tower rather than degrading to a text row.

## 3. Every vision-bearing family shares one trace and one key set

**MEASURED (Q2, Q3).** A trace taken from a single-image row accepts, unchanged
and with the media demonstrably reaching the language stack, both a nine-image
row with mixed grids and a row of three two-frame video blocks. Image count,
per-block grid, patch count and sequence length are all placeholders, not
constants.

**SOURCE.** That video result is not a coincidence of the probe's construction.
[`minimax.py`](../../../../../../comfy/text_encoders/minimax.py)'s
`process_video_block` emits `grid_thw = [1, grid_h, grid_w]` and a patch vector
of `3 * temporal_patch_size * patch_size ** 2`, and
[`qwen_vl.py`](../../../../../../comfy/text_encoders/qwen_vl.py)'s
`process_qwen2vl_images` reaches the same patch dimension for a still by
repeating the single frame across the temporal patch. A still image and a
two-frame video block are the same shape of object to the vision tower; the pair
of frames occupies the temporal slot the still fills by repetition. So H3 video
blocks are image-keyed by construction, and transformers labels them modality 1
in `mm_token_type_ids`, matching what the installed native path does.

**Consequence for the architecture.** A calibration population in which every
row carries at least one vision block covers I2VA, FL2VA, ordered multi-image
Ref2VA and video-reference blocks in one run, with no fabricated media and no
patch to `llm-compressor`. Text-only T2VA rows cannot join it. Including them
requires either a second traced graph — a real change to the sequential
pipeline, reviewable but not free — or their declared exclusion. Note that text
is not thereby absent from calibration: on most vision rows of the rejected
manifest's own trace, the text tag carries more tokens than the vision tag. What
is absent is a sequence whose first token is text.

**UNKNOWN.** Whether excluding text-only rows measurably changes W4 drift on
T2VA workloads. That is a question for the held-out benchmark, not for this
document, and it must not be answered by asserting either way here. The text
share above is read off the rejected trace, whose video rows understate their
own vision token count, so it is an upper bound on the text share rather than a
measurement of the population a corrected manifest would produce.

## 4. Two further requirements a hand-built batch must satisfy

**SOURCE.** `transformers` 5.15.1 refuses multimodal input without
`mm_token_type_ids` (text 0, image 1, video 2), raising rather than guessing,
because M-RoPE is computed from it. The processor emits it; a hand-built native
batch must derive it. It is a different quantity from H3's adaLN token tags:
`mm_token_type_ids` marks only the pad positions, while H3 tags the whole vision
span including the flanking vision-start and vision-end tokens. Both are needed,
for different consumers, and neither substitutes for the other.

**SOURCE.** Precomputing `inputs_embeds` and skipping the vision keys entirely
is not a legitimate shortcut, even though `Qwen3VLForConditionalGeneration`
accepts it. `Qwen3VLTextModel` injects DeepStack features into the hidden state
after decoder layer `i` for each `i` below the number of DeepStack mergers, and
the released tower declares three of them — so the injection lands on layers H3
consumes and AWQ quantizes. A calibration path that bypasses the vision tower
would collect statistics for those layers from a distribution the deployed model
never produces.

## 5. Preprocessing policies: the contract's four are five

[`native_h3_contract.md`](native_h3_contract.md) separates four preprocessing
policies and treats "native/release still-image policy" as one. **SOURCE:** the
installed native path and the release declaration are two, and they differ:

| policy | still-image pixel bounds | resize | owner |
|---|---|---|---|
| ComfyUI native code path | 3,136 to 12,845,056 | `F.interpolate` bilinear on float | `process_qwen2vl_images` defaults, which `Qwen3VL.preprocess_embed` does not override |
| release declaration | 65,536 to 16,777,216 | processor's own | `vendor_config/preprocessor_config.json` |
| current AWQ artifact | 200,704 to 301,056 | `Qwen2VLImageProcessor`, `resample: 3` bicubic, after a round-and-clamp to uint8 | the artifact's snapshot, installed by `h3_awq_encoder.py::install_source_processors` |

The two video policies remain separately owned as the contract describes:
release 4,096 to 25,165,824 and the loaded encoder's snapshot, which agree
today (the encoder policy binds to the CLIP's stamped contract since 2026-08-25;
see the contract's policy 5). Native ComfyUI is a third execution policy: it applies
3,136--12,845,056 independently to each two-frame block, while the release and
encoder configurations apply their maximum over the whole sampled clip before
native two-frame presentation. This scope difference is bounded rather than
universal: the measured release resize starts at legal target lengths of 311
frames for 1344x768 input and cannot activate inside the legal range for the
measured 960x544 input. Shipped reference graphs select the encoder policy, not
the stock-native one, except the `release` probe arm.

**INFERENCE.** The first two rows agree for any image between 65,536 and
12,845,056 pixels, which is most production stills, so the divergence is a
boundary effect rather than a routine one. It still has to be named, because
Codex's weight-only arm forces one policy into both models and the deployed-path
arm does not, and "the native policy" currently denotes two different things.

**SOURCE.** The interpolation and the uint8 round-and-clamp in
`h3_awq_encoder.py::_source_image_patches` are a second, separate difference
between the deployed W4 path and the native BF16 path, independent of the pixel
bounds. Any deployed-path comparison inherits all three: bounds, interpolation
kernel, and the uint8 boundary.

**Consequence.** "Derive the v2 processor settings from the declared target
serving policy" is under-determined until the target is named as one of these
rows. Whichever is chosen, it is a v2 serving-policy decision to be preserved in
the candidate's processor configuration, not a restoration of parity.

## 6. Independent confirmations of the preflight rejection

Reached before reading [`2026-08-24_awq_v2_preflight_review.md`](2026-08-24_awq_v2_preflight_review.md)
and consistent with it. Two additions it does not name:

**MEASURED.** The rejected builder's still-image path never repeats the frame
across the temporal patch, so it emits a patch vector of
`3 * patch_size ** 2` where the vision tower requires
`3 * temporal_patch_size * patch_size ** 2`. Its own geometry report records the
two side by side: every still row carries the short vector, every video row the
correct one. The still path could not have completed a forward pass.

**MEASURED.** Every still in that geometry report and every H3-IR row in its
row-level trace falls inside 200,704 to 301,056 pixels — the current artifact's
constrained band, not the release band the v2 lane was asked to target. The
audit report does not state which policy it used.

## What this does not establish

- Nothing about numerical fidelity, drift, or render quality. The probe uses a
  tiny random-init model; it measures the shape of the seam, not the behaviour
  of the 32B weights.
- Nothing about memory or runtime feasibility on the RTX 4090 for any proposed
  population. `IntermediatesCache.from_dataloader` materialises every row before
  the first subgraph runs, and `propagate_error` defaults to true so each
  subgraph is executed twice per row; neither cost has been measured here.
- This seam probe itself establishes nothing about implementation parity.
  Independent follow-up probes have since measured exact M-RoPE agreement and
  float32 vision/DeepStack arithmetic agreement within `2.384e-7`; their scope
  and remaining limits are recorded in
  [`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md).
- No acceptance of any manifest, population, or launch command.
