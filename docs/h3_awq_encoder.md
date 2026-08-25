# MiniMax H3 compressed-tensors W4A16 encoder adapter

Last verified against the installed ComfyUI and comfy-kitchen checkouts on
2026-08-25.

`MiniMaxH3AWQEncoderLoader` is a repo-local format adapter for a specific
family of Qwen3-VL-32B MiniMax H3 encoders: full Hugging Face checkpoints
written by `compressed-tensors` as symmetric group-128 W4A16 AWQ. It lets one
such checkpoint remain in its serving-oriented representation while using
ComfyUI's native MiniMax H3 architecture, tokenizer, model patcher and
conditioning path.

This is deliberately narrower than “ComfyUI AWQ support.” Core already loads
the separately converted NVFP4-AWQ H3 artifact. AWQ describes how weights were
calibrated; it does not define one checkpoint namespace, packing, or runtime
operator.

## Responsibility boundary

| behaviour | owner |
|---|---|
| 5120-wide, 50-layer MiniMax H3 text architecture | native ComfyUI |
| Unnormalized hidden state after language layer 50 | native ComfyUI |
| MiniMax tokenizer and the seven H3 tokens at ids 151669–151675 | native ComfyUI; the token fix is merged upstream |
| Vision-span `minimax_token_tags`, including the start/end delimiters | native ComfyUI |
| H3 prompt/reference presentation | native ComfyUI plus this repo's typed conditioning nodes |
| Recognition and validation of this compressed-tensors artifact | this repo's `MiniMaxH3AWQEncoderLoader` |
| Full-HF namespace to native-H3 namespace adaptation | this repo's loader |
| Selection of source layers 0–49 and removal of full-Qwen-only tensors | this repo's loader |
| Packed W4 weight view, scale layout and symmetric zero tensors | this repo's loader |
| AWQ W4A16 tensor layout and CUDA/eager operators | native `comfy-kitchen` |
| FP32-to-BF16 boundary needed to select kitchen's CUDA backend | this repo's loader |
| Still-image preprocessing from this artifact's processor config | this repo's loader |
| Duration-aware reference-video sizing/resizing | this repo's `MiniMaxH3ReferenceConditioning` working with settings exported by the loader module |
| Dynamic VRAM/offload scheduling | native ComfyUI; the loader only supplies a reconstruction callback for its custom representation |

The implementation is [`h3_awq_encoder.py`](../h3_awq_encoder.py). The exact
small source files it accepts are versioned under `config/`, one directory per
artifact generation, of which
[`config/qwen3vl_32b_minimax_h3_w4a16_awq/`](../config/qwen3vl_32b_minimax_h3_w4a16_awq/)
is v1's; see *Artifact generations* below. The multi-gigabyte checkpoints
remain external.

## Why stock `CLIPLoader` cannot load this representation

The stock loader lists every installed text-encoder safetensors file. Listing
a filename is discovery, not proof that core understands its contents.

This artifact carries the original Hugging Face namespace:

```text
model.visual.*
model.language_model.embed_tokens.*
model.language_model.layers.0 ... 63.*
model.language_model.norm.*
lm_head.*
```

Core's current Qwen3-VL detector sees `model.visual.*` first. It classifies a
vision merger output of 2560 as Qwen3-VL-4B and every other value as
Qwen3-VL-8B. The 32B merger output is 5120, so this full-HF namespace is
classified as 8B and core constructs a 4096-wide model. Passing
`clip_type=MINIMAX` does not override that earlier model detection. The public
load boundary consequently fails on a 4096-versus-5120 size mismatch.

That is a real detection/format limitation, but it is **not** a bug in the
hidden states produced by core's supported H3 checkpoints. The native INT8
ConvRot and NVFP4 artifacts have already been converted to the namespace core
expects:

```text
visual.*
model.embed_tokens.*
model.layers.0 ... 49.*
```

They therefore select ComfyUI's native 5120-wide `QWEN3VL_32B` MiniMax model,
whose config already specifies 50 layers, no language head, and no final norm.

Adding `5120 -> QWEN3VL_32B` to the full-HF detection branch would not by
itself make this checkpoint loadable. Core would still need to:

1. distinguish a full general-purpose Qwen3-VL-32B from the MiniMax H3
   50-layer conditioning use;
2. rename the Hugging Face language and vision namespaces;
3. retain only layers 0–49 and omit the final norm and language head; and
4. understand compressed-tensors' `weight_packed`, `weight_scale`, and
   `weight_shape` records.

The local node performs that complete adaptation rather than applying a
shape-only detector workaround.

## Load and adaptation sequence

### 1. Filename-independent selection

The node offers the real `.safetensors` population under
`models/text_encoders`. A generated workflow necessarily stores its selected
filename in the combo widget, but the implementation does not accept a file
because of its basename.

After selection, the loader validates:

- safetensors metadata declaring `scheme=w4a16` and `quantization=awq`;
- an embedded config exactly matching one versioned snapshot, which is also how
  the loader decides which artifact this is;
- the decoder layer count that snapshot declares, at hidden size 5120;
- symmetric, group-128, 4-bit packed weights with BF16 scales; and
- the complete adapted tensor inventory and every expected shape.

This strict artifact contract is intentional. A merely similar AWQ file must
fail loudly instead of partially populating a model under core's general
`strict=False` text-encoder loading policy.

### 2. H3 namespace and depth

The adapter renames:

```text
model.language_model.* -> model.*
model.visual.*          -> visual.*
```

It drops every source language layer past H3's depth — 50–63 in a 64-layer
source — along with `model.language_model.norm.*` and `lm_head.*`. The resulting state dict exactly targets native ComfyUI's H3
model. There are 350 quantized language linears: seven in each of the retained
50 layers.

The node does not implement an alternative layer-50 tap. Truncating the source
state dict makes native Comfy's already-correct 50-layer, no-final-norm model
the execution authority.

### 3. Packed W4 adaptation

The source stores eight signed four-bit values in each `int32` word after the
signed values have been offset into unsigned nibbles. On a little-endian host,
viewing that storage as `int8` produces four consecutive bytes, each already
holding the two nibbles that `TensorCoreAWQW4A16Layout` expects.

The **weight repack** is therefore a zero-copy view. The entire adaptation is
not literally allocation-free:

- source scales are transposed and made contiguous once because kitchen owns
  them as `(K / group_size, N)`;
- the symmetric source omits affine zero tensors, so the loader constructs
  exact-zero tensors required by kitchen's general AWQ ABI; and
- retained vision tensors, the full-shape input embedding, and decoder
  normalization weights retain their source BF16 tensors. The full-Qwen final
  language norm is not among those retained tensors.

No second multi-gigabyte checkpoint is written to disk at load. Producing
the artifact from a candidate directory does write one, once; see *Artifact
generations* below.

### 4. Native Comfy model construction and strict inventory check

After adaptation, the loader calls core's text-encoder constructor with
`CLIPType.MINIMAX` and custom mixed-precision operations. Core supplies the H3
architecture and tokenizer. The loader then compares every provided tensor
and shape against the concrete native module, rejecting missing, unexpected,
or incompatible entries. Symmetric zero tensors are the only synthesized
inventory exception.

The tokenizer check is a compatibility assertion, not a tokenizer patch. It
proves that the selected artifact's 20 declared special tokens, role ids, and
the native tokenizer agree. The seven MiniMax-specific tokens are already
provided by current ComfyUI.

## Artifact generations, and how one is recognized

Since 2026-08-25 this adapter serves more than one calibration of the same
architecture. v1 is the deployed single file. A candidate calibrated on the
release's own geometry arrives from `llm-compressor` as a Hugging Face
*directory*: shards, `config.json`, the release `preprocessor_config.json` and
`video_preprocessor_config.json`, tokenizer files, a recipe and a run record.

**The adapter still accepts exactly one thing: a single `.safetensors` whose
metadata declares `scheme=w4a16`, `quantization=awq` and the artifact's own
`config.json` verbatim.**
[`bench/convert_h3_awq_candidate.py`](../bench/convert_h3_awq_candidate.py)
produces that file from a candidate directory, together with the versioned
config snapshot under `config/`, in one pass. Teaching the loader to read the
directory instead was the alternative and was rejected: the single file keeps
one load path and one contract rather than two, is indifferent to how many
shards the candidate was saved in, gives the artifact a distinctive name in
ComfyUI's combo rather than a generic `model.safetensors` inside a directory,
and keeps the existing full-file digest control meaningful. It costs one more
copy of the weights on disk, and one host-memory-sized consolidation pass.

The conversion is faithful. Decoder layers past H3's depth, the final language
norm and the LM head are all kept, so the artifact stays re-derivable from the
calibration output and comparable to it; the adapter drops them at load, as it
always has.

### Recognition is by content, not by name

`_resolve_snapshot` compares the selected file's embedded config against every
`config.json` under `config/` and takes the one that matches exactly. Nothing
resolves by filename. That is what refuses a v2 file selected while only the v1
snapshot is installed, and — the case worth naming, because it is the tempting
repair when a candidate is refused — what refuses it against a v1 snapshot
somebody widened to "match" it. Both are red controls in
[`bench/check_h3_awq_encoder.py`](../bench/check_h3_awq_encoder.py)
(`snapshot_resolution_is_by_content`), and every installed snapshot is hashed
against its own `sha256.json` on every run.

**A snapshot pins the contract, not the calibration.** Two candidates from
different runs of the same recipe carry the same `config.json` and the same
processor and tokenizer files, so both resolve here and both load — correctly,
because the contract is what the loader validates and it is identical. What
distinguishes them is the artifact digest in `sha256.json` and the copied
`h3_v2_run_record.json`, and those describe the candidate this snapshot was
written from, not every candidate that resolves to it. Measured 2026-08-25 on
two independently produced two-layer candidates: every contract file identical,
run record and weights different. So give a re-run its own snapshot name rather
than `--force`-ing over one a deployed artifact still matches;
`bench/check_h3_awq_encoder.py`'s candidate arm distinguishes the two cases and
says which it found.

Each snapshot's files are byte-for-byte copies of the candidate's own, so those
digests are statements about the artifact as well as about the copies. The
digest of the produced `.safetensors` is not: `safetensors` 0.8.0 does not
write a byte-reproducible header — measured 2026-08-25, two writes of one
identical state dict differ in the header and agree on its length, with and
without sorted keys. Read it as an integrity record of the file that was
deployed, never as a reproducibility claim.

### What a second generation changed in the contract

Each of these was a v1 detail this adapter had only ever met once. The
replacement branches on the observable instead:

| v1's shape | what a release-geometry candidate carries | what the adapter reads |
|---|---|---|
| top-level `dtype: bfloat16` | `dtype: float32`, because the recipe keeps the vision patch embed in FP32 | the **decoder's** dtype from `text_config`, which is bfloat16 in both; the top-level value is recorded and allowed to be either |
| `num_hidden_layers: 64`, pinned as a constant | the same 64, but a reduced-layer smoke artifact carries fewer | the artifact's declared depth, with the H3 depth derived as the smaller of that and native's, and the quantized-linear count asserted exactly against it |
| `processor_config.json`, still settings nested under `image_processor` | `preprocessor_config.json`, the same settings flat | whichever file is present, and the settings from wherever they sit inside it |
| 20 special tokens under `extra_special_tokens` | the same 20, same order, under `additional_special_tokens`; `added_tokens_decoder` stops before the H3 ids | whichever key carries the list, with the list itself and the native tokenizer's realized ids asserted as before |

The release and v1 processor settings construct the *same* slow
`Qwen2VLImageProcessor` apart from `size`: the release file omits `resample`
and the `do_*` flags and the class defaults supply v1's values. Measured on the
installed transformers, 2026-08-25.

### What it changes for a graph, and what it does not

Nothing in a graph. The still-image ceiling a v2 artifact declares is the
release's, far above v1's, and the loader stamps whichever the loaded artifact
declares on the CLIP as `_h3_encoder_contract`. `image_policy=encoder` therefore
prices a reference against the encoder actually loaded, and
`bench/preflight_graph.py` resolves the same contract statically. Swapping the
artifact is a combo change.

Two things a v2 artifact does **not** get:

- **Its FP32 vision patch embed does not survive the load, and losing it costs
  nothing.** ComfyUI builds the H3 model at the decoder's dtype and
  `load_state_dict` casts on copy, which for an arbitrary FP32 tensor is not an
  exact round trip. This one is: the candidate's FP32 patch embed is the
  release's BF16 upcast, kept wide at calibration so the vision input is never
  downcast, so the cast back is exact. Measured on the two-layer candidate,
  2026-08-25: its `visual.patch_embed.proj.{weight,bias}` round-trip through
  BF16 bit for bit, and equal the installed release BF16 encoder's own patch
  embed bit for bit. `bench/check_h3_awq_encoder.py`'s candidate arm asserts the
  round trip on whatever FP32 tensors an artifact carries, so a future recipe
  that computes genuinely new FP32 values there goes red rather than having
  them silently discarded.
- **Reduced-depth artifacts cannot be constructed at all.** Core recognizes the
  H3 encoder by the presence of decoder layer 49 and builds a fixed 50-layer
  model, so a two-layer smoke artifact is detected as a different Qwen3-VL
  entirely. The loader refuses it by name of its declared depth rather than
  handing core a state dict it will misdetect, which is the shape of the
  2026-08-23 escape. Its adaptation is still depth-parametric and can be
  inspected directly, which is how the conversion path is exercised without the
  full candidate.

`config/qwen3vl_32b_minimax_h3_w4a16_awq_v2_smoke/` is that two-layer smoke
candidate's snapshot, committed so the adapter carries a second generation a
check can resolve, bind and refuse against with nothing external present. Its
weights are a disposable fixture; the artifact digest in its `sha256.json`
records the file the converter wrote on the day, and no deployed file matches
it.

## W4A16 execution

ComfyUI's `SDClipModel` presents H3 embeddings to the language model as FP32.
The comfy-kitchen W4A16 CUDA backend accepts FP16 or BF16 activations. Without
a local boundary, FP32 selects kitchen's eager dequantization implementation.

For an adapted W4 linear, the custom operation therefore:

1. remembers the caller's FP32 dtype;
2. casts the activation and optional bias to the BF16 scale dtype;
3. calls `torch.nn.functional.linear` with the kitchen quantized tensor; and
4. casts the result back to FP32 for native H3 residual arithmetic.

The loader logs the selected kitchen backend once. `backend=cuda` proves that
dispatch left the generic eager backend, but it does not prove that every
matrix shape uses fused INT4 MMA internally.

In the comfy-kitchen version installed on 2026-08-23, the effective CUDA route
is selected from flattened activation rows `M`:

| flattened rows | current CUDA route |
|---:|---|
| `M <= 8` | simple CUDA GEMV-style kernel |
| `8 < M <= 256` | fused INT4 × BF16/FP16 MMA; no intermediate BF16 weight matrix |
| `M > 256` | CUDA dequantization to BF16 followed by cuBLAS |

The nearby kitchen docstring says 512, but the executed threshold constant is
256. This is implementation state, not part of the node's permanent contract,
and must be rechecked when comfy-kitchen changes. Image and video references
can produce enough visual tokens to enter the large-`M` route.

This path is kitchen-native AWQ W4A16 execution. The node does not import or
implement Marlin, so documentation should not describe it as a Marlin runtime
kernel.

## Source-config preprocessing

### Still images

The loader binds a processor to this CLIP instance using the selected
artifact's snapshotted `processor_config.json`. For the current artifact that
means:

- patch size 16, temporal patch size 2, merge size 2;
- mean and standard deviation of 0.5 per channel;
- bicubic resampling (`resample=3`); and
- a 200704–301056-pixel still-image budget.

Native Qwen3-VL processing uses the same patch geometry and normalization for
H3, but Comfy's shared helper currently uses its own pixel bounds and bilinear
interpolation. The local behavior is therefore artifact-config-driven rather
than a claim that the native vision architecture is wrong.

This snapshot is much narrower than either stock ComfyUI or the release. It is
the binding Qwen constraint on reference stills under the current W4 loader;
upstream `max` or 2048-short-edge preparation cannot make the encoder retain
geometry that this stage removes. The measured layer-49 policy benchmark
therefore keeps the deployed snapshot unchanged and treats v2 calibration—not
an in-place config edit—as the repair path. This artifact-specific condition
must not be reported as a stock-ComfyUI processor bug.

The bicubic-through-uint8 path also differs numerically from stock ComfyUI's
float/bilinear path. Its independent contribution to layer-49 drift has not
been isolated from the pixel-bound change, so no fidelity ranking between the
resize paths is established.

Only the Qwen encoder view is owned here. Reference image/video geometry used
by the H3 VAE remains the responsibility of the conditioning path and can
intentionally differ.

### The stamped contract, and who reads it

`install_source_processors` stamps the artifact's declaration on the CLIP's
transformer as `_h3_encoder_contract`: still and video bounds and patch
geometry, with the still bounds being the ones that instance was bound with
(measurement override included). `snapshot_contract()` builds it from this
module's snapshot through `_config`, so the standalone build reaches its
embedded configs the same way; `snapshot_contract(directory)` builds it from
any artifact directory carrying the two processor files, which is how a
candidate shipped as an HF directory declares itself with no table row.
`reference_geometry.encoder_contract_from_clip` reads the stamp back, so the
typed conditioner's `image_policy` / `video_policy = encoder` apply the loaded
encoder's declaration and not this module's; a CLIP with no stamp resolves to
the native path. `encoder_contract_from_artifact(name)` is the static twin for
`bench/preflight_graph.py`, which sees a graph rather than a CLIP.
Controlled by `bench/check_reference_runtime.py` (`encoder_policy_binds_to_the_loaded_clip`,
`preflight_resolves_encoder_from_the_loader_node`) and by the standalone arm
of `bench/check_h3_awq_encoder.py`, which requires `install_source_processors`
to be source-identical in the one-file build.

### Reference video

The loader's video patchifier consumes an already fitted two-frame block. It
validates the temporal count and 32-pixel spatial grid, normalizes, and
patchifies without resizing again.

The earlier sizing and resize live in `reference_conditioning.py`:

- `video_policy=comfy` leaves the sampled frames for native core's block
  processing;
- `video_policy=encoder` keeps the repo's native-compatible no-upscale VAE
  view while fitting the separate 2-fps Qwen view with this encoder artifact's
  duration-aware pixel budget and bicubic processor; and
- `video_policy=release` uses the separately owned release snapshot.

The encoder and release video configs currently agree, but their ownership is
kept separate so either source can change without silently redefining the
other.

The adapter does not establish that native block resizing causes temporal
jitter. That would require a measured video comparison; it is not an accepted
reason for this implementation.

## Installed checkpoint comparison

This table describes the three concrete files inspected on 2026-08-23, not
every checkpoint that may share a similar filename.

| representation | loader owner | language linears | token embedding | vision tower | approximate file size | RTX 4090 execution note |
|---|---|---|---|---|---:|---|
| H3 INT8 ConvRot | core `CLIPLoader` | 350 `int8_tensorwise` ConvRot | full-shape BF16 | 351 BF16 tensors | 26 GB | Native core format; transfer/offload cost needs measurement rather than inference from file size alone |
| H3 NVFP4-AWQ | core `CLIPLoader` | 350 `nvfp4` records | full-shape INT8 plus FP32 per-row scale | 351 BF16 tensors | 15 GB | Ada has no native NVFP4 compute, and this specific file also declares `full_precision_matrix_mult=true`; expect dequantized/full-precision matmul |
| Full-HF compressed-tensors W4A16 AWQ | this repo's loader | 350 retained group-128 W4A16 linears | full-shape BF16 | 351 BF16 tensors | 19 GB source file | CUDA backend is available; fused versus dequant-plus-cuBLAS depends on `M` as described above |

Header inspection on 2026-08-24 shows that neither Comfy-native compressed
file prunes vocabulary rows or embedding width: both retain an input lookup of
shape `[151936, 5120]`. The INT8 ConvRot file stores that table in BF16; the
NVFP4 file stores it as INT8 with one FP32 scale per row. Quantizing the latter
is compression, not structural pruning.

Those two files are already H3-only structural subsets on disk. Each contains
decoder layers 0–49, no final language-model norm, and no LM head. After
normalizing quantized records into logical weights, each retains the same H3
set: the input embedding, all first-50-layer decoder tensors, and all 351 BF16
vision/DeepStack tensors. Their physical pruning is therefore limited to
decoder layers 50–63, the final language-model norm, and the separate LM head.

The W4 source file contains 64 language layers plus full-Qwen output tensors;
the adapter discards the 14 unused layers, final norm and language head before
constructing H3. A real smoke logged about 14.97 GB of staged H3-relevant
weights. That is not a peak-VRAM guarantee: activations, attention workspaces,
reference token count, other loaded models and Comfy's offload policy still
matter.

The fact that the W4 artifact retains BF16 embeddings and vision weights is a
property of the artifact's quantization recipe, not code added by the node.
Likewise, the NVFP4 artifact's quantized embedding is not evidence by itself of
visible degradation in MiniMax delimiter or subject tokens. Quality claims
need comparison with the BF16 encoder.

## What the node does not do

`MiniMaxH3AWQEncoderLoader` does **not**:

- add generic compressed-tensors or generic AWQ support to `CLIPLoader`;
- fix native Comfy's H3 hidden-size, layer count, output tap, modality tags, or
  tokenizer—those are already native and correct in the installed checkout;
- process reference audio or change the 32 kHz H3 audio path;
- make every CUDA matmul a fused INT4 operation;
- prove that W4A16 is faster or higher quality than INT8 ConvRot or NVFP4; or
- keep the entire H3 video pipeline resident on a 24 GB GPU at once.

It is a strict bridge between one serving-oriented storage ABI and native
Comfy H3 execution, with artifact-faithful preprocessing added to that CLIP
instance.

## Standalone Hugging Face distribution

The model repository also carries
`comfyui_minimax_h3_awq_loader.py`, a generated single-file form of this node.
It can be downloaded directly into `ComfyUI/custom_nodes`; cloning this full
research repo is not required merely to load the encoder.

The standalone file is not maintained as a second implementation.
[`bench/build_h3_awq_standalone.py`](../bench/build_h3_awq_standalone.py):

1. copies the format, execution, preprocessing, validation and load paths from
   the authoritative [`h3_awq_encoder.py`](../h3_awq_encoder.py);
2. embeds exact text plus SHA-256 provenance for `config.json`,
   `tokenizer_config.json`, `processor_config.json`, and
   `video_preprocessor_config.json`;
3. adds its own V3 `comfy_entrypoint`, so ComfyUI discovers the `.py` file
   directly; and
4. derives four examples from the installed official ComfyUI H3 templates,
   replacing only the text-encoder loader and, for the FL2VA examples, applying
   this repo's declared v1.1 owner recipe.

Rebuild to any staging directory with uv:

```bash
uv run --active --no-sync python bench/build_h3_awq_standalone.py \
  --output-dir /path/to/hf-model-repo
```

The generated examples are text-to-video, image-reference-plus-text,
first-frame-to-video, and an explicit first/last-frame copy with two image
loaders. Their conditioning, scheduler, sampler, AV decode and save nodes
remain native ComfyUI. The T2VA/keyframe copies use the current repo-owned
v1.1 recipe—strength 0.75, six render steps and shift 6/3—and label that as an
owner choice rather than vendor-attested v1.1 settings. The reference example
remains on its separate native ref2va base recipe.

The build also emits `comfyui_minimax_h3_encoder_ab_compare.json`, a visual
review utility rather than a generation graph. It loads two completed clips,
concatenates them left-to-right at matching size, and writes a 24 fps H.264
comparison. That file requires VideoHelperSuite and ComfyUI-KJNodes; audio is
intentionally disconnected so the comparison has one unambiguous clock.

The single file does not include this repo's typed reference builders,
preflight, attention experiments, or duration-aware
`MiniMaxH3ReferenceConditioning video_policy=encoder`. In particular, a
standalone native reference-video graph follows core's reference-video sizing
path; users who want the repo-owned split VAE/Qwen video policy still need the
full research repo.

Do not install the generated file and the full research repo in the same
ComfyUI instance. Both intentionally register `MiniMaxH3AWQEncoderLoader`, and
which duplicate definition wins would depend on custom-node load order.

## Workflows and verification

Every currently generated **generation example** uses
`MiniMaxH3AWQEncoderLoader`. The most direct examples are:

- [`workflows/h3_text_to_video.json`](../workflows/h3_text_to_video.json) for a
  text-only UI graph; and
- [`workflows/h3_image_ref_plus_text_to_video.json`](../workflows/h3_image_ref_plus_text_to_video.json)
  for the BF16 vision tower and source still-image processor.

[`bench/check_h3_awq_encoder.py`](../bench/check_h3_awq_encoder.py) controls the
native/local boundary against the real installed files. It verifies:

- core's native NVFP4-AWQ recognition as a positive control;
- core's continued rejection of the compressed-tensors file, so the local
  adapter is retired if upstream support appears;
- packed-nibble order and CPU numerical execution;
- all 350 retained W4 linears, 50-layer depth, strict tensor inventory,
  processor source and tokenizer ids; and
- deterministic standalone output, exact embedded-config digests, critical
  function parity, direct V3 discovery, and one standalone loader per
  generation example;
- the separate comparison workflow's two VHS loaders, rightward matched-size
  KJNodes concatenation, and 24 fps VHS output; and
- optionally, real CUDA dispatch (`--gpu`) and the external model's full-file
  digest (`--verify-model-hash`).

The full CPU construction runs through the generated standalone module after
those parity controls pass. Set `H3_AWQ_STANDALONE` to the HF-hosted `.py` path
to additionally require that published/local copy to equal a fresh build.

The highest-value remaining comparison is an encoder-only BF16/INT8/NVFP4/W4
benchmark. Quantization fidelity should first be measured on text-only inputs
and on identical preprocessed vision tensors; normal end-to-end processing
should be a separate arm so processor differences are not mistaken for weight
quantization error. Runtime reporting should distinguish kitchen's inner
`gemv`, fused-MMA and dequant-plus-cuBLAS routes instead of recording only the
outer `cuda` backend.
