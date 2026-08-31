# Gate 1: the native-H3 to `llm-compressor` calibration seam

**Date:** 2026-08-24
**Status:** Deliverable for Codex review. Not authority; not a launch request.
**Scope:** Gate 1 of [`active_plan.md`](../../canonical/active_plan.md) only. No
quantization was launched, no candidate directory created, no deployed artifact
touched, and the feasibility pilot has not started.

## What this delivers

Five executables and their results. Each is red/green and each carries controls
that were watched failing, because a gate nobody has seen fail is not evidence.

| what | executable | result |
|---|---|---|
| build the batches `oneshot` will consume, from the installed path | [`build_native_h3_calibration_batch.py`](../../../../../bench/build_native_h3_calibration_batch.py) | the bundle |
| grade that bundle against independent arms, and make it fail | [`check_native_h3_presentation.py`](../../../../../bench/check_native_h3_presentation.py) | [`presentation_parity.json`](../../../../../bench/results/2026-08-24_native_h3_presentation_parity.json) |
| follow the batch through the real library to the traced graph | [`prove_calibration_seam.py`](../../../../../bench/prove_calibration_seam.py) | [`seam_proof.json`](../../../../../bench/results/2026-08-24_calibration_seam_proof.json) |
| map and strictly load the released checkpoint into the calibration model | [`check_calibration_model_mapping.py`](../../../../../bench/check_calibration_model_mapping.py) | [`model_mapping.json`](../../../../../bench/results/2026-08-24_calibration_model_mapping.json) |
| compare the two stacks at the H3 output boundary on released weights | [`compare_transformers_comfy_layer50.py`](../../../../../bench/compare_transformers_comfy_layer50.py) | [`layer50_mixed.json`](../../../../../bench/results/archive/v2_encoder/2026-08-24_crossstack_layer50_mixed.json), [`layer50_controls.json`](../../../../../bench/results/archive/v2_encoder/2026-08-24_crossstack_layer50_controls.json) |
| attribute the divergence that comparison found | [`probe_released_vision_precision.py`](../../../../../bench/probe_released_vision_precision.py) | [`vision_precision.json`](../../../../../bench/results/2026-08-24_released_vision_precision.json) |

Plus one repair to an accepted artifact, described under *Escaped defect* below.

## Revisions

Recorded because a symlink name is not a version.

| moving part | revision |
|---|---|
| this repository | `84d31f2bdcbb8b7e38ce797315cac059c16142e9` at build time |
| installed ComfyUI | `b78cec879b9460d5cb25228a83a942fb78d2cd24` |
| `coderef/llm-compressor` | `8357e9459228be5831ff43f9449cdd7733d3d877` |
| `coderef/sglang` | `30f9ed09d1c84f0fcbeabdb897fb2b027d90af0b` |
| `coderef/diffusers` | `efabd60d61c2b7aabf9f182bee6b5b6058980304` |
| `coderef/DiffSynth-Studio` | `72fb128e07a145323dceb072fe15f902745e18fe` |
| `coderef/MiniMax-H3` | `d21241f0a4b3acbb34c97dae47fa417b7065e438` |
| H3-IR dataset | `460db3256f19dc70d0def2068a22e6e0dca87e8e` |
| ComfyUI virtualenv | python 3.13.9, torch 2.13.0+cu132, transformers 5.15.0 |
| `llm-compressor` virtualenv | python 3.13.9, torch 2.13.0+cu132, transformers 5.15.1, llmcompressor 0.13.1.dev38+g501f432bf |

## The architecture, and the one place it is not object identity

**SOURCE.** The two virtualenvs cannot import each other. `comfy` requires
`comfy_aimdo`, which the pinned quantization environment does not have; the
ComfyUI environment has no `llmcompressor`. Installing either into the other
would perturb the environment the candidate is meant to be built in, so the
seam is a **hashed bundle**: the ComfyUI process writes the batch tensors and a
field-by-field presentation record, and the `llm-compressor` process re-derives
every hash at every hop.

That is a real difference from "the exact object", and it is named here rather
than glossed. What replaces object identity is a chain that is checked at each
link -- bundle file, `DataLoader`, `IntermediatesCache`, traced subgraph inputs
-- plus a control that swaps two rows' batch files behind the manifest's back
and requires the chain to notice.

**Nothing in the builder reimplements presentation.** Labels, ordering,
timestamp formatting, marker ids, vision-block length, image-pad expansion,
attention masks, H3 token tags, M-RoPE inputs and DeepStack placement are all
produced by executing the installed code that owns them, including
`comfy/sd1_clip.py::SDClipModel.process_tokens` unmodified. Patch geometry and
resize come from the release-declared processors configured out of
`vendor_config/`. Upstream role sizing comes from
`comfy_extras/nodes_minimax_h3.py` and `reference_conditioning.py`.

The one substitution is hidden *width*: the vision tower and token embedding run
at released patch, merge, depth and DeepStack geometry with a reduced hidden
size, because every presentation field is geometry. No weight-dependent claim
rests on that model. The weight-dependent claims come from the cross-stack
comparison, which uses the real released BF16 checkpoint on both sides.

## Fixtures

**MEASURED.** Five real H3-IR rows, one per accepted primary role, built from
real decoded media at the accepted v2 role policy: keyframes on the resolved H3
canvas, ordinary stills at `max` with no upstream upscale, reference video under
the release 768-short-edge canvas rule with duration-aware Qwen sampling and
native two-frame presentation.

| role | blocks | grids | sequence | vision positions |
|---|---:|---|---:|---:|
| single-image | 1 | 60x44 | 1,699 | 662 |
| multi-image 2--3 | 2 | 156x104, 52x68 | 5,857 | 4,944 |
| keyframe-only | 1 | 48x84 | 2,007 | 1,010 |
| keyframe-plus-reference | 3 | 96x172, 96x172, 48x84 | 10,358 | 9,270 |
| video-reference | 5 | 48x84 each | 6,189 | 5,050 |

The keyframe-only row's 48x84 grid is the 1,008-merged-token canvas control
[`native_h3_contract.md`](../../canonical/native_h3_contract.md) names, reached
here through the real code rather than asserted.

The mixed row is the one to look at for feasibility: three pictures, two
distinct per-picture geometries, 10,358 tokens and 37,056 patch rows in a single
row. Per-picture roles are not optional for it, exactly as the plan says.

## What the presentation gate establishes

**MEASURED.** Two independent arms, neither of which calls anything the builder
calls beyond the release processor configuration and the pixels:

- A **vendor-shaped arm** following `coderef/sglang`'s own
  `minimax_h3/presentation.py` algorithm reproduces the builder's `input_ids`
  exactly on all five rows, reference video included.
- A **`Qwen3VLProcessor` arm** reproduces `input_ids`, `attention_mask`,
  `mm_token_type_ids`, `pixel_values` and `image_grid_thw` exactly on the rows
  it can express.

**MEASURED, and a real constraint on any future builder.** The installed path
tokenizes each label segment separately, and so does the vendor's. Concatenating
the full string before tokenizing merges the trailing space of an `<Audio j>: `
label into the following prompt word and emits a sequence **one token shorter**.
Every other label is followed by a special token, which breaks the merge anyway,
so the case arises exactly once per row and only when an audio label is the last
ordered item. The `Qwen3VLProcessor` arm is single-shot by construction, so its
sequence fields are excluded on those rows and the exclusion is recorded per row
rather than quietly dropped.

**MEASURED.** The release does not state the seven H3 marker ids anywhere
directly: its `added_tokens_decoder` stops at the last stock Qwen entry, and the
seven appear only in `additional_special_tokens`, whose order then decides the
id each receives. Deriving the expected ids from those two vendored declarations
and comparing against the installed tokenizer puts all seven where the release
puts them. This also makes the check sensitive to a reordering of that list,
which would silently reassign every marker.

**MEASURED.** The release video processor and
`comfy/text_encoders/minimax.py::process_video_block` agree on the two-frame
block to `5.914e-8` maximum absolute difference, same grid, same layout. That is
recorded per block in every bundle.

**MEASURED.** Nine deliberate defects were built for real -- chat framing, first
image only, reference reorder, timestamp shift, dropped temporal repeat, grid
shrink to the current artifact's band, dropped media, flipped token tags, zeroed
`mm_token_type_ids`. Each moves its own named record field, and a row a mutation
cannot structurally reach comes back byte-identical.

## What the seam gate establishes

**MEASURED.** Every batch file and every tensor in it hashes to the record. A
preconstructed `DataLoader` handed to `get_calibration_dataloader` is returned as
the same object, so no sampler or collator intervenes. `next(iter(dataloader))`
-- the object `SequentialPipeline` traces from -- and every later batch still
hash to the record, and so does what `IntermediatesCache.fetch` returns for
subgraph 0's declared inputs.

**MEASURED.** The traced subgraph declares `pixel_values`, `image_grid_thw` and
`mm_token_type_ids`. All five rows run against the graph traced from the first,
and perturbing the pixels moves the subgraph output on every one, so the media
reaches the language stack rather than being dropped.

**MEASURED.** The released tower's three DeepStack mergers inject after decoder
layers 0, 1 and 2 -- inside the 0--49 window H3 consumes and inside the 64 the
candidate quantizes.

**MEASURED, and it closes an open question the contract did not answer.** The
vendor's serving stack labels H3 two-frame blocks `<|video_pad|>`; ComfyUI
splices embeddings and never materialises a pad id at all, so the calibration
batch must choose. Relabelling every block as video and moving its grid to
`video_grid_thw` produces **bit-identical** M-RoPE position ids. At `grid_t = 1`
the choice does not matter, which the source reading predicted and this
measures.

**MEASURED.** Transformers' own `get_rope_index` on each real batch equals the
ComfyUI position ids the builder recorded, on all five rows.

**MEASURED, controls.** A text-only row fed to the vision trace raises, which is
the loud failure this population depends on. Swapping two rows' batch files
behind the manifest is detected. Eight of the nine mutated bundles break the
identity chain; the ninth, flipped token tags, does not and **must not** --
token tags are returned to the DiT for adaLN and are absent from the calibration
batch entirely, so the presentation gate owns that one. Recording it as
"expected not detected" is a specificity claim: a chain that flagged everything
would be flagging the bundle rather than the defect.

## What the checkpoint gate establishes

**MEASURED.** The released 1,058-tensor checkpoint and the calibration model's
state dict are a perfect bijection: zero missing, zero unexpected, zero shape
mismatches, and every stored tensor at the dtype the config declares. Three
non-persistent RoPE buffers are recorded as excluded rather than filtered
silently -- they are recomputed at init and appear in no checkpoint, and
comparing against the wider parameters-plus-buffers set reports them missing
every run, which is a false red.

**MEASURED.** `from_pretrained` reports empty missing, unexpected and mismatched
key lists, builds all 64 decoder layers and three DeepStack mergers, and every
parameter is bfloat16. That call maps safetensors lazily, so the arm also
touches the first and last element of all 1,058 state-dict tensors: none is left
on `meta` and none has non-finite ends.

**MEASURED.** 352 tensors must stay BF16 in the candidate -- 351 vision, of
which 18 are DeepStack mergers, plus the input embedding.

**MEASURED, controls.** A renamed, a reshaped and a recast key are each reported
with the right reason.

## What the output-boundary comparison establishes, and what it cost to trust it

This is the only part using released weights on both sides. Both arms consume
the same bundle; the ComfyUI arm replays the bundle's patch tensors rather than
recomputing them, so no processor difference can enter, and the comparator
refuses metrics unless the presentation hashes agree.

**MEASURED.** On the two-picture row, ComfyUI's layer-50 state against
Transformers at float32:

| position class | flattened cosine | relative L2 |
|---|---:|---:|
| text rows | 0.9999966 | 0.0026 |
| vision rows | 0.9196 | 0.393 |

and at the input to decoder layer 0, before any DeepStack injection, text rows
are **bit-identical** while vision rows already differ by 0.0188. So the whole
divergence originates in the vision features and the language stack amplifies
it, chaotically and per row: on a small two-block row the same comparison lands
at relative L2 0.0020, on a vision-heavy row at 0.337.

**MEASURED, and this is the finding.** Driving the released vision tower alone
in both implementations attributes it completely:

| comparison | relative L2 |
|---|---:|
| ComfyUI float32 params vs Transformers float32 | 0.00138 |
| ComfyUI bfloat16 params vs ComfyUI float32 params | 0.0187 |
| ComfyUI bfloat16 params (deployed) vs Transformers float32 | 0.0188 |
| ComfyUI bfloat16 params (deployed) vs Transformers bfloat16 | 0.0955 |

**SOURCE.** `Qwen35VisionModel.fast_pos_embed_interpolate` calls `ops.Embedding`
with no `out_dtype` and builds its bilinear coefficients at
`self.pos_embed.weight.dtype`. So the position-embedding lookup, the
interpolation weights and their product run at the *stored* dtype even though
every `manual_cast` linear upcasts to the float32 activation. "ComfyUI runs the
vision tower in float32" is true of the linears and false of the position
embedding.

**MEASURED.** A ComfyUI tower built with bfloat16 parameters reproduces the
deployed in-situ result exactly, and all 351 in-situ vision weights are
bit-identical to the released shards, so this is precision in the interpolation
path and not a corrupted load or a memory-management artefact.

**INFERENCE.** No Transformers dtype matches the deployed path on both axes at
once: float32 matches its linears and not its position embedding, bfloat16 the
reverse and worse. Of the two, float32 is closer at the vision boundary
(0.0188 against 0.0955) and much closer at layer 50 (0.337 against 0.825 on the
mixed row). This informs the calibration dtype decision; it does not make it,
and it is not a claim about which produces a better quantization.

**This corrects an overreach in a closed finding, without retracting it.**
[`2026-08-24_transformers_comfy_parity.md`](../../canonical/2026-08-24_transformers_comfy_parity.md)
measured vision arithmetic agreement at `2.384e-7` and told this lane to consume
rather than repeat it. That probe used a small seeded float32 configuration, and
its own limits section says it is not a substitute for validating the full seam.
It was right about the arithmetic: at matched precision the released towers agree
to 0.00138, which is float32 accumulation over 27 real layers, not a disagreement.
What it could not see is that the deployed configuration does not run at matched
precision. The repo's rule that an assumption which has only met one
implementation is not a tested assumption applies to configurations too.

**Controls, and an honest limit.** On the tightly-agreeing row, tapping decoder
layer 48 instead of 49 moves relative L2 from 0.0020 to 0.0506, a 25x separation.
Scaling one layer-0 `down_proj` output by one percent moves it to 0.0023, a 14
percent change -- detectable but not commanding.

On the vision-heavy row the aggregate hides the wrong-layer control entirely:
0.337 becomes 0.347, which no reader would call a signal. **Split by position
class it is unmistakable** -- text rows go from 0.0026 to 0.0852, a 33x
separation, because the precision term lives in the vision rows and leaves the
text rows clean. The one percent weight error stays invisible there in every
split.

So this comparison detects a wrong output depth reliably, provided the metrics
are read by position class; it detects a small weight error only on rows whose
vision share is low; and its aggregate is dominated by the precision term in
proportion to a row's vision share. That is why the reported numbers are split
rather than flattened, and any later use of this instrument has to respect the
bound.

Two controls had to be rebuilt before they could fail, and both failures are
worth recording because both looked green:

- The first weight perturbation edited a parameter in place under
  `device_map="auto"`. Accelerate refills offloaded parameters from its own map
  at call time, so the edit was discarded and the perturbed arm returned a state
  bit-identical to the clean one. It is now an output scale on a bias-free
  projection.
- The first vision-tower perturbation scaled a bfloat16 weight by 0.1 percent.
  bfloat16's relative step is about 0.4 percent, so the value rounded straight
  back and the control reported an exact zero.

## Escaped defect found and repaired

**MEASURED.** The accepted candidate pool's only media check was a count of
images whose geometry PIL could read, and it was skipped entirely for
`video-reference` rows. All 20 video rows had entered the pool on the strength
of a declared filename; 16 of the 19 distinct files they name were absent from
the local snapshot, and nothing said so.

Repaired: `bench/build_h3_calibration_pool.py::media_status` now opens and hashes every
declared file with no exemption by role or media kind, and a row whose media does
not verify receives a rejection reason.
[`check_pool_media_integrity.py`](../../../../../bench/check_pool_media_integrity.py)
holds it red/green against a deleted file, a corrupted file, an undeclared hash
and a mismatched *video* file -- the last being the escaped defect itself.

The 16 absent files were fetched from the pinned revision and all 19 verify
against their declared hashes. The pool was rebuilt: the role partition, the
overlay counts, the component structure and the largest component are unchanged,
so the accepted mix did not move.

**Two corrections for `calibration_data_pool.md`, for Codex to integrate.**

1. Its provenance section records the builder's SHA-256 and the three output
   hashes. All four have changed, and the outputs now carry a
   `media_verification` block per row.
2. It says the accepted split preflight must recompute all **3,101** media
   hashes. That figure counts pictures only -- 141 keyframes plus 2,960 ordinary
   references. The pooled rows declare **3,121** media items across 1,651
   distinct files; the missing 20 are the video references, omitted by exactly
   the exemption this repair removes.

## Two seam facts for Gate 2, measured here as side effects

**MEASURED.** Passing an all-ones `attention_mask` into the Transformers model
sends SDPA to its math backend, which materialises a `heads x seq x seq` tensor
-- 9.13 GiB at float32 and 6,189 tokens, and quadratic from there. Dropping a
mask that masks nothing leaves the attention causal and identical and cut the
forward from 24.1 s to 8.0 s at bfloat16. A calibration batch that carries the
mask will meet this on long rows.

**MEASURED.** Even without the mask, one float32 forward at 6,189 tokens did not
fit on the 4090 alongside offloaded weights, while 1,746 tokens did. The mixed
keyframe-plus-reference fixture is 10,358 tokens. This is a bounded observation
from a comparison harness, not a feasibility result -- Gate 2 owns that -- but it
is the direction Gate 2 should expect.

## What this does not establish

- Nothing about numerical fidelity of any quantized checkpoint, and nothing
  about render quality. No DiT render was produced.
- Nothing about memory, runtime or whether any calibration population fits. The
  two observations above are from a comparison harness with its own memory
  policy, not from the sequential pipeline.
- Nothing about which Transformers dtype the candidate should calibrate in. The
  vision-boundary and layer-50 numbers rank the arms on those axes only.
- The cross-stack comparison rests on **two real rows**, chosen because a
  float32 forward fits. It is fixture-level evidence about the boundary, not a
  population estimate, and its sensitivity to a weight error is modest.
- The seam is a hashed bundle across a process boundary, not one live object.
  The chain is checked at every link and the disconnect control fires, but the
  claim is "the hashes agree at each hop", not "it is the same Python object".
- The reduced-width model used for tracing and for the presentation record
  establishes geometry, not arithmetic. The arithmetic claims come from the
  released-weight arms.

## Open questions for Codex

1. **Calibration dtype.** float32 is closer to the deployed path on both
   measured axes, and one float32 forward at 6,189 tokens did not fit on this
   box. Whether the candidate calibrates in float32 with more aggressive
   offload, or in bfloat16 with the position-embedding gap declared, is a
   decision this lane should not make alone.
2. **`attention_mask` in the emitted batch.** The builder currently emits it.
   Dropping it when it is all ones is free and avoids the SDPA math backend, but
   it changes what the launcher hands `oneshot` and should be an accepted
   decision rather than an optimisation this lane applies.
3. **`calibration_data_pool.md` provenance and the 3,101 figure**, per the
   corrections above.
4. **Whether the pool builder's media verification belongs in `canonical/`** as
   a required property of an accepted pool, given it was absent when the pool was
   accepted.

## Preservation

The deployed checkpoint, its symlink, its processor snapshot and its hash file
are untouched. No candidate directory exists. `llm-compressor` was used only to
construct dataloaders, caches and traces on a reduced-width model; `oneshot` was
never called and no recipe was instantiated. The bundles built for these runs
live outside the repository and are not committed.
