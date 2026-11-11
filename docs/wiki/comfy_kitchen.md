# comfy-kitchen: what it gives a custom node, how a call reaches a kernel, and what our fork adds

last updated: 2026-11-11

comfy-kitchen is the kernel library under ComfyUI: fused, quantized and sparse versions of the operations a
diffusion model spends its time in, written per backend and picked per call. Every H3 forward pass goes
through it several times a block. Read this page before writing code that runs inside a forward pass (an
attention override, a block patch, a guider, a hook), so the code uses what is already fused and does not
undo it.

This page is a map with pointers. A function's contract is its docstring in
`coderef/comfy-kitchen/comfy_kitchen/__init__.py`; this page says which function to look at and why. It does
not own:

- the Sol knobs, their measurements and what not to rely on: [`docs/SOLATTN.md`](../SOLATTN.md);
- what upstream does and claims, snapshot by snapshot: [`docs/sol_upstream.md`](../sol_upstream.md);
- the fork's branches, the build and the gate: the header of `vendor/rebuild_kernel.sh`, and
  [`docs/comfy_notes.md`](../comfy_notes.md), section "After a ComfyUI update";
- which precision each H3 layer runs at: [`docs/h3_quant_policy.md`](../h3_quant_policy.md).

The source is `coderef/comfy-kitchen`, the owner's fork with `h3-frontier` checked out, which is also the
source of the installed build (`bash vendor/rebuild_kernel.sh --check` names both).

## The short version

1. **Reach kitchen through core's wrappers when one exists.** `optimized_attention`, `comfy.ops.linear_input_act`
   and an `operations.Linear` whose weight is a quantized tensor already call the fused op and still honor
   offload, LoRA patches and the model's stored attention config. A direct `comfy_kitchen.*` call skips all of
   that. Call kitchen directly only for something core has no wrapper for (our Sol node does, for the options
   core does not pass).
2. **A call a fused backend cannot take is not an error.** The registry tries the next backend, and the last
   one is plain torch. A wrong dtype or layout therefore shows up as a slow render, not a traceback. Section
   "Which backend took the call" says how to see it.
3. **Keep what the fused kernels need.** bfloat16 or float16, never a `.float()` round trip; `head_dim` 128 and
   `(B, T, H, 128)` for Sol; a contiguous last dimension. Views are fine, copies are not needed.
4. **Never read a value back on the hot path.** `.item()`, `.cpu()`, `bool(tensor)` and a print of a tensor
   each wait for the GPU. Kitchen launches on torch's current stream and returns at once; a read-back turns
   that into a stall on every block of every step.
5. **Kitchen makes one pass cheaper. It does not remove a pass.** Code that runs the model twice a step pays
   twice. See "Code that runs every forward pass".

## How a call reaches a kernel

1. **The public function** in `coderef/comfy-kitchen/comfy_kitchen/__init__.py` (the names in its `__all__`).
   It builds the keyword arguments and asks for an implementation. Two shapes exist. Most functions call
   `coderef/comfy-kitchen/comfy_kitchen/registry.py::BackendRegistry.get_implementation` themselves
   (`coderef/comfy-kitchen/comfy_kitchen/__init__.py::int8_linear` is one). Some go through a
   `torch.ops.comfy_kitchen.<name>` custom op first (`coderef/comfy-kitchen/comfy_kitchen/__init__.py::sol_attn`
   is one); the op is declared with `torch.library.custom_op`, most of them in the eager backend's files, and
   its body does the same registry lookup. `torch.library.custom_op` is PyTorch's way to present a function
   to its compiler as one opaque operation.
2. **The registry picks a backend for this call**, not once at import.
   `coderef/comfy-kitchen/comfy_kitchen/registry.py::BackendRegistry.get_capable_backend` walks the priority
   list (`BackendRegistry.__init__` holds the default order), skips a backend that is unavailable or does not
   implement the function, and validates the actual arguments against that backend's constraints
   (`coderef/comfy-kitchen/comfy_kitchen/constraints.py::validate_function_call`: dtype, device, shape rules,
   the GPU's compute capability, and a per-function rule such as
   `coderef/comfy-kitchen/comfy_kitchen/constraints.py::sol_attn_common_call_rule`). The first backend that
   passes is used. If none passes it raises `NoCapableBackendError` with each backend's reason.
3. **The backends.** `eager` is the reference in plain torch and implements almost everything
   (`coderef/comfy-kitchen/comfy_kitchen/backends/eager/`). `cuda` is the compiled extension this card runs.
   `triton` covers a subset. `hip` (AMD) and `ascend` are registered as unavailable here.
4. **The CUDA backend** is Python in `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py`: it
   checks and arranges the tensors, hands them to the extension through DLPack
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py::_wrap_for_dlpack`) together with torch's
   current CUDA stream, and returns. The extension's entry points are in
   `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/dlpack_bindings.cpp`; the kernels are the `.cu` files
   under `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/` and
   `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/`. A kernel launch does not wait for the
   GPU: the work is queued behind whatever torch already queued on that stream.
5. **The build.** The extension is compiled for this card's architecture by `vendor/rebuild_kernel.sh`; a
   kernel guarded by a newer architecture in the source is not in our build. The installed version string
   names our tip and the upstream commit it sits on.

### Which backend took the call

- `comfy_kitchen.list_backends()` prints each backend, whether it is available, and why not.
- `comfy_kitchen.registry.get_capable_backend(name, kwargs)` answers for one call's real arguments, and
  raises with the reasons when no fused backend qualifies.
- The logger `comfy_kitchen.dispatch` at DEBUG logs the backend chosen for every call.
- `with comfy_kitchen.use_backend("eager"):` forces the reference, which is how a fused result is compared
  with the plain-torch one (`coderef/comfy-kitchen/tests/` does this throughout).

## What is available

Grouped by what a node author reaches for. The left column is a name in
`coderef/comfy-kitchen/comfy_kitchen/__init__.py`; read its docstring for shapes and dtypes. "Core" is where
ComfyUI itself calls it, which is the best worked example of correct use.

### Attention

| function | what it does | core | kernel source |
|---|---|---|---|
| `sol_attn` | sparse attention: each query block attends a routed set of key blocks exactly and the rest through one pooled term a block. Takes post-RoPE `q, k, v`. | `comfy/ldm/modules/attention.py::attention_comfy_kitchen_sol` | `sage_attention/sol_attn*.cu`, `sol_layout.cuh` |
| `sol_attn_chunked` | the same attention fed the fused qkv projection in chunks, so full `q, k, v` never exist; applies the RMSNorm and RoPE itself and carries key statistics from the previous step. One sequence only. | `comfy_extras/nodes_sparse_attention.py::h3_sparse_attention` | `sage_attention/sol_attn_producer.cu` |
| `int8_attention` | dense attention with `q, k, v` quantized to INT8 inside the call. | `comfy/ldm/modules/attention.py::attention_comfy_kitchen_int8` | `sage_attention/quant_qk_int8.cu`, `quant_v_int8.cu`, `sage_attn_launcher.cu` |
| `prequantize_int8_attention`, `int8_attention_from_prequantized` | the same in two halves, so the floating-point inputs can be freed before the output is allocated. | `comfy/ldm/modules/attention.py::_attention_comfy_kitchen_int8_containers` | as above |
| `na3d`, `na2d`, `window_attention` | attention over a local neighborhood or window, without building the neighbors. | `comfy/ldm/lightricks/vae/na_diffusion_decoder.py`, `comfy/ldm/wan/model_vsr.py` | `ops/na3d.cu` |
| `flash_attention_decode*`, `gated_delta_*`, `deltanet_conv_step*` | single-token decode for the text encoders' language models. Not on the DiT's path. | `comfy/text_encoders/llama.py`, `comfy/text_encoders/qwen35.py` | `ops/flash_decode.cu`, `ops/gated_delta.cu` |

`sol_attn_is_available` and `int8_attention_is_available` answer for a device before a node commits to a
path.

### Normalization and position, fused

| function | what it does | core |
|---|---|---|
| `rms_rope_split_half_` and its family (`rms_rope`, `rms_rope1`, the `_split_half` and in-place `_` forms) | per-head RMSNorm and RoPE in one pass, in place on a view of the qkv buffer. | `comfy/ldm/minimax/model.py::Attention` |
| `apply_rope` and its family | RoPE alone, same forms. | `comfy/text_encoders/llama.py` |
| `adaln`, `rms_adaln` | `norm(x) * (1 + scale) + shift` in one pass. | none on H3's path |
| `rms_gated_residual` | `residual + gate * rms_norm(activation)`. | none on H3's path |
| `group_norm_silu_pad3d`, `group_norm_stats3d`, `fp16_conv3d` | the VAE and upscaler path: per-frame GroupNorm, SiLU and causal padding in one pass; a conv3d with bias and residual in its epilogue. | `comfy/ldm/minimax/latent_upscaler.py`, `comfy/ldm/seedvr/vae.py` |

The in-place forms exist because the fused q and k are views into one qkv allocation. The kernels address a
tensor through its own strides, so a permuted or sliced view is read where it lies.

### Linear layers

| function | what it does |
|---|---|
| `int8_linear` | a linear layer with INT8 weights and the activation quantized inside the call. `input_act` folds the previous activation or norm into the quantizer (`"swiglu"`, `"gelu_tanh"`, `"situ"`, `"rms_norm"`), and `residual` with `residual_scale` folds the block's add into the GEMM's epilogue, so neither intermediate is written out and read back. |
| `int8_linear_modulated`, `int8_linear_rms_modulated` | the same with a per-row modulation scale, and with an RMSNorm, applied in the producer. |
| `fp16_linear` | a linear that accumulates in fp16, with the same optional residual. |
| `scaled_mm_nvfp4`, `scaled_mm_mxfp8`, `convrot_w4a4_linear`, `w4a8_int8_linear`, `scaled_mm_svdquant_w4a4`, `gemv_awq_w4a16` | the matmul for each other weight format. |
| `quantize_*`, `dequantize_*` | weight and activation conversion for each format. |

A node should almost never call these. A quantized weight is a
`coderef/comfy-kitchen/comfy_kitchen/tensor/base.py::QuantizedTensor` with a layout class (one file each
under `coderef/comfy-kitchen/comfy_kitchen/tensor/`), and core's `comfy/ops.py::linear_input_act` asks that
weight for `linear(act(x))`, falling back to the plain sequence when the layout has no fused form. Calling
`linear_input_act(layer, x, "swiglu")` instead of `layer(silu(gate) * up)` is the whole difference between
the fused path and two extra tensors a block. `bench/bench_kitchen_int8_linear.py` times `int8_linear` on
H3's own layers.

### Utilities

`memmove` (an overlapping byte copy on the device, used by core's weight gathering), `set_allocation_context`
(core registers kitchen's own persistent allocations with its memory management through it), the backend
controls (`set_backend_priority`, `disable_backend`, `enable_backend`, `use_backend`, `list_backends`), and a
set of mesh geometry operations that nothing in this pack uses.

## One H3 block, call by call

`comfy/ldm/minimax/model.py`, in the order a block runs them:

1. `Attention.forward` projects to one qkv buffer with a single linear. With an INT8 file the weight's
   layout runs it as `int8_linear` (`coderef/comfy-kitchen/comfy_kitchen/tensor/int8.py`).
2. It applies `rms_rope_split_half_` in place on the q and k views of that buffer.
3. It wraps `q, k, v` in `AttentionTensorContainer` and calls `optimized_attention` with
   `preferred_attention`, the attention the file itself asks for (`comfy/ldm/modules/attention.py::ComfyAttention`
   reads it from the checkpoint). An `optimized_attention_override` in `transformer_options` wins over that
   (`comfy/ldm/modules/attention.py::wrap_attn`). That override is the hook our Sol node uses.
4. `MLP.forward` is `comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`.

H3 packs text, video and audio into one sequence with no batch dimension (`comfy/ldm/minimax/model.py::PackedLayout`),
which is why the attention inputs arrive as one sequence and why two prompts cannot share a pass by batching.

## What the fork adds to upstream

`git -C coderef/comfy-kitchen log --no-merges upstream/main..h3-frontier` is the whole list, and
`git -C coderef/comfy-kitchen diff upstream/main...h3-frontier --stat` is every file it touches. Everything
else in the build is upstream main as of the commit named in the installed version.

It is four Sol options, one binding fix, and ports of two open upstream fixes. Each option is off by default
and the off path is upstream's code.

| option on `sol_attn` | what it does | where | not available on |
|---|---|---|---|
| `blk_cnt` | an optional device buffer the same launch fills with how many key blocks each query block attended exactly. Nothing is read back unless the caller reads it later. | the copy after the launch in the CUDA backend's `sol_attn`; `coderef/comfy-kitchen/comfy_kitchen/backends/eager/sol_attn.py` for the reference | HIP |
| `qk_balance` | rescales q up and k down per head and channel before INT8 quantization, leaving every score unchanged, so a head whose key energy sits in a few channels keeps resolution in the rest. Heads without that concentration are left exactly as they were. | `sage_attention/sol_attn_preprocess.cu`, constants in `sage_attention/sol_layout.cuh` | HIP, `sol_attn_chunked` |
| `rotate` | rotates every q and k row by one fixed orthogonal matrix before INT8 quantization, so no single channel or token sets the row's scale. The transform `int8_attention` already applies. Composes with `qk_balance`. | `sage_attention/sol_layout.cuh`, used by preprocess, the producer and the token stage | HIP |
| `tau_map` | a routing threshold per head and query block in place of the one `tau`. | `sage_attention/sol_attn_preprocess.cu` | HIP, `sol_attn_chunked`, with `topk_ratio` |

`sol_attn_chunked` takes `blk_cnt` and `rotate` as keyword-only arguments. The binding fix marks an optional
mask argument as accepting `None` under the nanobind version the venv has
([`docs/comfy_notes.md`](../comfy_notes.md) has the story).

Upstream status on 2026-11-11: `blk_cnt` is open as Comfy-Org/comfy-kitchen#168 and the binding fix as #256.
The other three have no open pull request. A ComfyUI update that reinstalls the stock wheel removes all of
them, and `sol_attn_h3.py::_require_kernel` then fails the node before sampling, because the installed
`sol_attn` no longer accepts an argument the node passes.

Where the pack uses them: `sol_attn_h3.py` (the `MiniMaxH3Sol` node; `quantizer` selects `qk_balance` and
`rotate`, `make_override` builds the attention override), `sol_observe.py` (reads `blk_cnt`),
`sparse_table.py` (per-head tables that become a `tau_map`), `sol_chunked_h3.py` (the chunked producer).

**A gap in the pack's chunked node, found by the Gemini session on 2026-11-11 and not fixed.**
`sol_chunked_h3.py::make_chunked_forward` passes the Sol node's `tau`, dense blocks and sinks to
`sol_attn_chunked` and nothing else. It never reads the `quantizer` setting, so `rotate` stays off on that
path even when the Sol node says `rotated`, and it ignores `token_routing`, `tau_table` and `rows` without a
word. No shipped graph wires `MiniMaxH3SolChunked`, so no render is affected. Before that node is used again
it should take `rotate` from the setting and refuse the settings the producer cannot honor.

### Two open upstream fixes, carried since 2026-11-11

Both are open pull requests upstream that no maintainer had reviewed on that day. They are carried because
both faults showed on this card, and they are dropped when upstream merges them or fixes the same thing its
own way (`vendor/rebuild_kernel.sh --check` marks a carried commit that upstream now has).
[`bench/results/2026-11-11_kitchen_upstream_sync_and_two_fixes.md`](../../bench/results/2026-11-11_kitchen_upstream_sync_and_two_fixes.md)
has the before and after, what was not taken, and how to go back;
`bench/probe_kitchen_swiglu_and_token_overflow.py` prints the same figures for whatever build is installed.

- **Comfy-Org/comfy-kitchen#124, the CUDA half.** On architectures below sm_90 the row quantizer applied
  `input_act="swiglu"` in fp32, so its INT8 codes differed from quantizing the product computed in the storage
  dtype; upstream's quantizer for sm_90 and up already rounded. The fused path serves a layer only when its
  activated width is a multiple of 256 no wider than `_CONVROT_FUSED_MAX_K`
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py`), so of the INT8 files this pack loads it
  reaches the DiT (`comfy/ldm/minimax/model.py::MLP`, every block), the video VAE's decoder
  (`comfy/ldm/minimax/vae.py::FeedForward`) and the two small P-H3-Image language models, and it does not reach
  the 32B encoder, whose MLP is wider than the limit. With the port the fused and unfused paths agree bit for
  bit. **An INT8 sample or decode on a build with the port does not match one made before it; a conditioning
  from the INT8 32B encoder does**, so a record of INT8 renders is for the build it names.
  Where: `load_input_act` in `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/int8_linear.cu`.
- **Comfy-Org/comfy-kitchen#242.** With `token_aug` on, a query group with more candidates than the budget
  kept whichever arrived first, so two identical calls could differ. Such a group is now redone in one pass in
  a fixed order, which costs time in proportion to how many groups overflow. Where:
  `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu`.

### How Sol is laid out, for reading the source

`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu` opens with the stage list and
owns the workspace plan (`sol_attn_plan`): one caller-allocated buffer, every stage reading and writing
slots of it.

| stage | file | what it does |
|---|---|---|
| preprocess | `sol_attn_preprocess.cu` | quantizes q, k, v to INT8, pools k and v per block, computes the routing threshold. `qk_balance`, `rotate` and `tau_map` act here. |
| vtranspose | `sol_attn_vtranspose.cu` | lays v out transposed for the exact stage. |
| route | `sol_attn_route.cu` | chooses each query block's key blocks and accumulates the pooled tail for the rest. `blk_cnt` is this stage's count. |
| token | `sol_attn_token.cu` | with `token_aug`, picks individual high-scoring tokens outside the routed blocks. |
| exact | `sol_attn_exact.cu` | walks each routed list with INT8 arithmetic, continuing route's running softmax. |
| producer | `sol_attn_producer.cu` | the chunked entry: norm, RoPE and quantization chunk by chunk in place of preprocess. |

## Rotation and RoPE: what is done for you, and what your code must keep

Three different things here are called a rotation, and two are called RoPE. Each is handled for you as long as
your code stays on the ordinary path, and each has a place where custom code has to carry it. The table came
from the Gemini session's review on 2026-11-11 and every row was checked against the source named in it.

| mechanism | done for you | your code must handle it when |
|---|---|---|
| **The q and k rotation inside attention** (a Hadamard rotation before INT8, so one loud channel cannot set a row's scale) | always on in kitchen's dense INT8 attention (`coderef/comfy-kitchen/comfy_kitchen/sage_attention.py::int8_attention`); on in Sol through the node's `quantizer`, whose default is `sol_attn_h3.py::SOL_QUANTIZER_DEFAULT`. A guider, a hook or a LoRA branch never sees it | you call `comfy_kitchen.sol_attn` yourself (pass `rotate`), or write a new attention entry |
| **ConvRot, the rotation stored in INT8 weights** (input columns rotated in groups of 256) | any call through `operations.Linear` or `comfy.ops.linear_input_act`; the activation is rotated to match inside the kernel (`convrot64_quantize_row` in `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/int8_linear.cu`). `lora_branch.py` adds its low-rank term on the unrotated input and output, so it needs nothing | you read INT8 weights off disk. The stored codes are in rotated space, so `codes * scale` is not the weight: dequantize through the `QuantizedTensor`, or compare something the rotation preserves such as row norms (`bench/measure_h3_image_weights.py` does both) |
| **The DiT's RoPE** (split-half, over part of each head's channels) | any node that leaves the packed positions and `Attention.forward` alone | (1) **you change how many text rows there are.** The media rows' first coordinate starts after the text, so a different text length moves every media row: `edit_guidance.py::EditGuider._prepare` shifts the shorter prompt's media rows under `align_positions`, and `control_vector_nodes.py::_drop_visual_cond_from_layout` keeps slices of the original `position_ids` and never recounts from zero. (2) **you replace the attention forward.** Apply the same fused op with the width read from the table, `rot_dim = rope_freqs.shape[-3] * 2`, never a typed number: `frozen_video_cache.py::_qkv_rope` is the pattern. `sol_attn_chunked` takes `rope_freqs` and applies it itself |
| **The text encoder's M-RoPE** (vision tokens get grid coordinates, and the text after them starts where the grid ends) | the ordinary conditioning nodes | you encode in pieces or build pairs. `reference_encode.py::_positions` asks core for the positions of a slice given every vision block before it, and `control_vector_nodes.py::match_media_pair_geometry` holds both sides of a pair to one grid so their positions are identical |

A change to kitchen needs a second look at this table only if it touches `rms_rope.cu`, `sol_layout.cuh` or
`quant_qk_int8.cu`; a change to core needs one if it touches `comfy/ldm/minimax/model.py::Attention` or
`PackedLayout`.

## Code that runs every forward pass

The edit guider is the worked case. `edit_guidance.py::EditGuider.predict_noise` asks core for two
predictions a step, the original prompt and the edited one, and `EditGuider._prepare` marks them so they are
never batched together. That is two full passes of the model, and each pass makes every call in "One H3
block" for every block. Kitchen is already doing its work inside each pass. The guider's own arithmetic, one
blend of two predictions, is a single small operation a step and not worth optimizing.

So the cost of a guider is decided by how many passes it runs, which is why `first_step` and `last_step`
exist ([`edit_guidance.md`](edit_guidance.md)), and no kernel changes that. What kitchen-aware code can do is
avoid making each pass slower. For anything that runs inside a pass:

- **Take the containers.** Core hands an attention override `q, k, v` as single-owner containers when the
  override has a `container_function` (`comfy/ldm/modules/attention.py::wrap_attn`). Taking them lets the
  inputs be freed as soon as they are consumed. `sol_attn_h3.py::make_override` shows the form, including how
  to pass a call it does not want to the previous override.
- **Do not change dtype.** The fused backends take bfloat16 and float16. A `.float()` on the way in sends the
  call to the eager reference without a word, and a cast back on the way out is a second full copy.
- **Do not copy to make something contiguous.** The staging loads need a contiguous last dimension and an
  aligned base (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/dlpack_bindings.cpp::need_staging_layout`),
  which a permuted or sliced view of the qkv buffer already has.
- **Prefer the in-place and `out=` forms** (`rms_rope_split_half_`, `fp16_conv3d` and `group_norm_silu_pad3d`
  with `out`), and `addcmul_` or `lerp_` on an existing tensor for a blend, over building a new tensor a call.
- **Collect on the device, read after the step.** `blk_cnt` is the pattern: the launch fills a device buffer
  and the caller reads it when the sampler is not waiting. A statistic computed with `.item()` inside a block
  costs a stall a block.
- **Carry what the next step can reuse.** `sol_attn_chunked` returns the key mean and v scale for the next
  step's call so they are not measured again; anything else that changes slowly across steps can be carried
  the same way.
- **Decide once, outside the pass.** Check what can be checked when the patch is applied
  (`sol_attn_is_available` for the device, and the kernel's arguments, as `sol_attn_h3.py::_require_kernel`
  does), so a missing capability fails the node before sampling and not part-way through a render.
- **Measure with the queue drained.** A timing without `torch.cuda.synchronize()` on both sides measures the
  launch, not the work. [`docs/pipeline_telemetry.md`](../pipeline_telemetry.md) records what a prompt spent
  per node; [`docs/hardware.md`](../hardware.md) lists what invalidates a timing.

When a pass is slower than expected, the first question is which backend took each call, and the section
above answers it before any profiler is needed.
