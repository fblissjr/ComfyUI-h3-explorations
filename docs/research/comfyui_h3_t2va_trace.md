# ComfyUI's H3 t2va pipeline, traced end to end

Written 2026-08-29 against the ComfyUI checkout at `e7051b03`, the installed
`comfy_kitchen` wheel, and the artifacts on this box. Five scoped readers took
one stage each (the two VAEs; the Qwen3-VL encoder; the sampler and guider; the
model load and quantization ops; the commit history) and reported with line
citations; the DiT itself, the PDD lane, and every number below were traced and
measured here. **Where a reader's claim was surprising it was re-derived by
execution, and those are marked.**

**What this file owns.** What *ComfyUI's own code* does, call by call, from
loading four artifacts to writing pixels and samples, for one `t2va` render at
1344x768x345. It compares nothing to another engine.
[`h3_dit_implementations.md`](h3_dit_implementations.md) owns the five-way
numerical comparison and [`sglang_comparison.md`](sglang_comparison.md) the
runtime one; where this file touches those it points and stops.

**Deliberately out of scope**, per the brief: every node in this pack except
`pdd_lora.py`, Sol-Attn, and sage. The graph traced is stock ComfyUI nodes only.

**Evidence labels**, used inline:

- *read* — the code at the cited path, at the revision above.
- *measured* — computed for this document, from a file on disk or by executing
  the code path in the ComfyUI venv.
- *log* — read out of this box's own `comfyui_8188.log`.
- *inference* — a conclusion, with the mechanism named so it can be refuted.

Core paths are relative to the ComfyUI checkout (`comfy/...`, and
`ComfyUI/nodes.py` where the basename is ambiguous). The
`.venv/.../comfy_kitchen/...` citations are the installed wheel, written
ComfyUI-relative so `bench/check_doc_links.py` can range-check them.

---

## 0. The two scenarios, and the shape of the render

### 0.1 What is loaded

| slot | artifact | on disk |
|---|---|---|
| `UNETLoader` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GiB, 932 tensors |
| `CLIPLoader` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.28 GiB, 1602 tensors |
| `VAELoader` (video) | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB, 562 tensors |
| `VAELoader` (audio) | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB, 917 tensors |
| **scenario 2 only** | `minimax_h3_fl2va_pdd_8step_comfy.safetensors` | 1.044 GiB, 730 tensors |

*measured*, from the safetensors headers.

### 0.2 The geometry this render resolves to

`EmptyMiniMaxH3LatentAV(width=1344, height=768, length=345)` goes through
`comfy_extras/nodes_minimax_h3.py:33-46` (*read*):

```python
def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n

def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS                       # FPS = 24
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)  # 40
```

345 already satisfies `345 % 17 == 5`, so nothing snaps. *measured*:

| quantity | value | where it comes from |
|---|---|---|
| frames | 345 | `align_frame_count(345)` is a fixed point |
| duration | 14.375 s | `345 / 24` |
| video latent | `[1, 24, 102, 48, 84]` | `latent_t = (340//17)*5+2`, `H/16`, `W/16` |
| audio latent | `[1, 32, 2, 575]` | `round(14.375 * 40)` |
| rows per latent frame | 1008 | `(48/2) * (84/2)`, the 1x2x2 patch |
| **video rows** | **102,816** | `102 * 1008` |
| **audio rows** | **1,150** | `575 * 2`, channel-major stereo |
| **text rows** | **500** | the prompt, tokenized (*measured*, §5.2) |
| **packed sequence S** | **104,466** | `500 + 1150 + 102816` |

Everything downstream is a function of those five numbers. **1344x768 is not
resized** — `adapt_canvas` exists in that file but only `MiniMaxH3ReferenceToVideo`
calls it; `MiniMaxH3ImageToVideo` and `EmptyMiniMaxH3LatentAV` take width and
height verbatim (*read*, `comfy_extras/nodes_minimax_h3.py:78-86, 137`).

### 0.3 What the two scenarios differ by

Scenario 2 adds one node between the loader and the sampler and rewires one
edge. Nothing else moves: same checkpoint, same encoder, same prompt, same
canvas, same `euler`, same shift 12/3.

| | scenario 1 | scenario 2 |
|---|---|---|
| DiT weights | as loaded | + 308 patch keys at strength 1.0 |
| `final_layer.{video,audio}_out` | the checkpoint's own | forward-patched per step |
| sigmas | `BasicScheduler` | `MiniMaxH3PDDLoRA`'s `SIGMAS` output |
| evaluations | whatever `steps` says | 8, and only 1/2/4/8/16/32 are legal |

---

## 1. Loading the DiT

### 1.1 `UNETLoader` to `ModelPatcher`, and what never gets asked

`ComfyUI/nodes.py:982-1005` -> `comfy/sd.py:2355` `load_diffusion_model` ->
`comfy/sd.py:2280` `load_diffusion_model_state_dict` (*read*). Two things worth
naming before the detection:

`weight_dtype` on the node offers `default / fp8_e4m3fn / fp8_e4m3fn_fast /
fp8_e5m2` and **no int8 entry, which is correct** — the format is discovered
from the file, not declared at the node. Setting it to anything else would force
`unet_dtype` and would *not* dequantize anything, because the quantized weight
is installed by `_load_quantized_module`, which never reads `unet_dtype`.

`comfy.utils.convert_old_quants` runs first and is a no-op here: it rewrites the
legacy `scale_weight` convention into the modern per-layer `comfy_quant` blob,
and this file already ships modern. Whichever route a file arrives by, the
module loader sees exactly one thing — a `<layer>.comfy_quant` uint8 tensor
holding JSON.

### 1.2 Detection reads eleven keys and hardcodes the patch grid

`comfy/model_detection.py:390-417`, the whole MiniMax H3 branch (*read*). What
each lookup yields for this file (*measured*, against the header):

| config field | key sniffed | shape | value |
|---|---|---|---|
| gate | `video_patch_proj.weight` **and** `audio_patch_proj.weight` | — | branch taken |
| `num_layers` | `blocks.{n}.` scan | — | 50 |
| `token_refiner_num_layers` | `token_refiner.blocks.{n}.` scan | — | 2 |
| `hidden_size` | `video_patch_proj.weight` | `[5376, 96]` | 5376 |
| `latents_dim` | `final_layer.video_out.weight` | `[96, 5376]` | `96 // 4` = 24 |
| `audio_latents_dim` | `final_layer.audio_out.weight` | `[32, 5376]` | 32 |
| `attention_head_dim` | `blocks.0.attn.q_norm.weight` | `[128]` | 128 |
| `num_attention_heads` | `blocks.0.attn.qkv_proj.weight` | `[21504, 5376]` | `21504 // 384` = 56 |
| `ffn_hidden_size` | `blocks.0.mlp.fc1.weight` | `[28672, 5376]` | `28672 // 2` = 14336 |
| `text_dim` | `condition_proj.weight` | `[5376, 5120]` | 5120 |
| `adaln_curve_grid` | `adaln_t_table` | `[1025, 8]` | 1025 |
| `time_embed_dim` | same tensor | | 8 |
| `rope_inv_freq_len` | `rope.inv_freq` | `[16]` | 16 |

**`patch_size` is never detected.** It is hardcoded twice — as the `// 4`
divisor with the comment `# patch 1x2x2`, and as the constructor default
`patch_size=(1, 2, 2)` in `comfy/ldm/minimax/model.py:470`. A checkpoint with a
different patch grid is silently misread as having a different `latents_dim`.

**There is no `in_channels`.** The two input widths are `latents_dim` (24) and
`audio_latents_dim` (32), each with its own patch projection. This is also why
`BASE.inpaint_model()` — `self.unet_config["in_channels"] > 4` — would raise on
H3, and why `MiniMaxH3.get_model` overrides `get_model` outright rather than
going through the base implementation that calls it.

**`adaln_t_table` is what selects the pruned path**, and it is a positive test
on an observable, not a filename:

```python
        table_key = '{}adaln_t_table'.format(key_prefix)
        if table_key in state_dict_keys:
            table = state_dict[table_key].shape  # [grid, k]
            dit_config["adaln_curve_grid"] = table[0]
            dit_config["time_embed_dim"] = table[1]
        else:
            te = state_dict['{}time_embedder.proj_in.weight'.format(key_prefix)]
            ...
```

**Nothing distinguishes fl2va from ref2va.** *measured*: comparing the two
`int8_convrot` headers key by key gives `fl2va only: []`, `ref2va only: []`,
`shape/dtype differ: []`, 932 tensors each. `unet_config` for H3 has exactly one
key (`image_model`), so `BASE.matches` is one string comparison and every
variant hits the same class. The graph is responsible for pairing the right
checkpoint with the right conditioning nodes, and there is no guard. (This is
what the repo's PDD node's `base_video_out` distance check exists to catch, from
the other side — §4.3.)

**Missing keys fail two different ways.** The gate itself and `adaln_t_table`
are guarded by `in state_dict_keys`; `final_layer.video_out.weight`,
`blocks.0.attn.q_norm.weight`, `blocks.0.attn.qkv_proj.weight`,
`blocks.0.mlp.fc1.weight`, `condition_proj.weight` and `rope.inv_freq` are bare
subscripts that raise `KeyError` out of `detect_unet_config` and propagate to
the node. `count_blocks` is the silent one — it counts consecutively from 0 and
stops at the first gap, so a file missing `blocks.7.*` reports `num_layers = 7`
and the other 43 blocks land in `unexpected_keys`.

### 1.3 What the file actually stores

*measured*, dtype census over the 932 tensors:

| dtype | tensors | bytes | what |
|---|---|---|---|
| `I8` | 200 | 17.944 GiB | 50 blocks x {`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`} |
| `BF16` | 220 | 1.488 GiB | 2 refiner blocks (unquantized), 200 norms, `condition_proj`, `final_layer.norm` |
| `F16` | 102 | 0.081 GiB | 51 x {`adaln_proj.linear.weight`, `.bias`} |
| `F32` | 210 | 0.017 GiB | 200 `weight_scale`, `adaln_t_table`, `rope.inv_freq`, both patch projections, both output heads |
| `U8` | 200 | ~0 | the `comfy_quant` JSON sidecars |
| | **932** | **19.53 GiB** | |

Per quantized linear, three tensors. `blocks.0.attn.qkv_proj`:

```
  .weight        I8   [21504, 5376]
  .weight_scale  F32  [21504, 1]        <- per OUTPUT ROW, not per tensor
  .comfy_quant   U8   [72]
     -> {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

The format string says `tensorwise` and the scale is `[N, 1]`. **The name
records the algorithm family, not the scale granularity**; the layout accepts
either and switches on `scale.numel()` (`.venv/lib/python3.13/site-packages/comfy_kitchen/backends/eager/quantization.py:1005-1009`).
There is no `input_scale` and no zero point — the scheme is symmetric and the
activation scale is computed per row at run time.

**The two token-refiner blocks are unquantized BF16**, which is a deliberate
asymmetry: the refiner runs once per sampling run, not once per step (§5.4).

**There is no `__metadata__` on this file.** The quantization description is
per-layer, inside the tensor payload.

### 1.4 The whole-model quant config is one boolean

`comfy/utils.py:1432-1437` (*read*):

```python
def detect_layer_quantization(state_dict, prefix):
    for k in state_dict:
        if k.startswith(prefix) and k.endswith(".comfy_quant"):
            logging.info("Found quantization metadata version 1")
            return {"mixed_ops": True}
    return None
```

One `.comfy_quant` key anywhere flips the whole model onto the mixed-precision
op set; every layer's format is then read independently at load. That is what
lets one file mix int8 blocks, bf16 refiner blocks and fp16 AdaLN with no
top-level declaration.

`comfy/sd.py:2322-2336` then deliberately throws away the observed storage dtype
(*read*):

```python
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None
```

`comfy.utils.weight_dtype` returns the dtype holding the most *elements*, which
here would be `torch.int8` — a meaningless answer for compute. Forcing it to
`None` makes both `unet_dtype` and `unet_manual_cast` fall through to
`supported_inference_dtypes = [bfloat16, float32]` and return **bf16**.
`torch.float16` is excluded, so H3 will not run in fp16.

### 1.5 Op selection, and the fp32 island that is not fp32

`comfy/ops.py:1651-1667` (*read*) — the quant branch is checked **first**, ahead
of `fp8_ops`, `cublas_ops`, `disable_weight_init` and `manual_cast`:

```python
    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        logging.info("Using mixed precision operations")
        disabled = set()
        ...
        return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled)
```

Note the third argument is absent: `full_precision_mm` defaults to `False`.
Hold that — it is the whole difference between the DiT and the text encoder
(§2.4).

Now the part that costs precision, `comfy/ops.py:1297-1303`:

```python
        class Linear(torch.nn.Module, CastWeightBiasOp):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                super().__init__()
                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
```

**The `dtype=` the caller passed is discarded.** And `comfy/ops.py:1146-1147`
loads any layer with no `comfy_quant` blob at that dtype:

```python
    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
```

So on this artifact, every one of these lands as **bf16**:

| module | declared in `model.py` | on disk | loaded as |
|---|---|---|---|
| `final_layer.video_out` | `dtype=torch.float32` | F32 | **bf16** |
| `final_layer.audio_out` | `dtype=torch.float32` | F32 | **bf16** |
| `video_patch_proj` | `dtype=torch.float32` | F32 | **bf16** |
| `audio_patch_proj` | `dtype=torch.float32` | F32 | **bf16** |
| `blocks.N.adaln_proj.linear` | `adaln_dtype=torch.float32` | F16 | **bf16** |

*measured*, by executing the load path against synthetic tensors of the real
shapes and dtypes:

```
final_layer.video_out  declared fp32, disk F32  -> loaded torch.bfloat16 | bias torch.bfloat16
adaln_proj.linear      declared fp32, disk F16  -> loaded torch.bfloat16
attn.qkv_proj          disk I8+scale          -> loaded QuantizedTensor storage torch.int8
                                                  orig_dtype torch.bfloat16 convrot True
```

`comfy/ldm/minimax/model.py:302` calls those two heads "the checkpoint's fp32
island". **On the int8 artifact the island does not survive the load.** The
comment is true of the file and of the bf16 checkpoint (which goes through
`manual_cast`, where the declared dtype *is* honoured); it is false of the
configuration this box renders. Three layers disagree about `adaln_proj`
in particular — the model asks fp32, the file stores F16, the loader produces
bf16, and bf16 has *fewer* mantissa bits than the F16 on disk.

The one thing that stays fp32 is `adaln_t_table`, because it is a
`register_buffer` on `MiniMaxH3Model` and is filled by the parent module's
ordinary `_load_from_state_dict` into the existing fp32 buffer.

**So what — the magnitudes, and the one place it actually bites.** *measured*,
what the bf16 load costs each tensor against its value on disk:

| tensor | on disk | rel error from the bf16 load |
|---|---|---|
| `final_layer.video_out.weight` | F32 | 6.1e-4 |
| `final_layer.audio_out.weight` | F32 | 3.7e-4 |
| `video_patch_proj.weight` | F32 | 8.9e-4 |
| `audio_patch_proj.weight` | F32 | 8.5e-4 |
| `blocks.N.adaln_proj.linear.weight` | **F16** | **1.7e-3** |
| for contrast: any int8 block linear | I8 | **8.8e-3** |

Read that bottom row first. **The island loss is 5x to 20x smaller than the
quantization error the same checkpoint already carries in every block.** So this
is not a hidden precision disaster, and nobody should expect a visible
difference from it alone.

Two things do follow, and they are not about magnitude:

**The AdaLN row is a strict regression, not a rounding choice.** F16 has 11
mantissa bits and bf16 has 8. The file paid for precision the loader discards on
contact — the only such case in the checkpoint. Everything else is fp32 on disk
being rounded once, which is unavoidable at bf16.

**It is a confound for any checkpoint A/B.** A bf16 H3 checkpoint has no
`comfy_quant` blobs, so it goes through `manual_cast` / `disable_weight_init`,
where `operations.Linear` really is `torch.nn.Linear` and `dtype=torch.float32`
*is* honoured. So "bf16 checkpoint against int8 checkpoint" is not a comparison
of the block linears. It changes the block linears **and** the output heads
**and** the patch projections **and** the AdaLN projection, all at once. Anyone
attributing a rendered difference to int8 quantization is attributing it to four
changes. `docs/evidence.md` #22's first-step velocity delta of 5.6-9.4% between
the two checkpoints is measured across all four.

**And one stale justification.** `pdd_lora.py`'s dtype note says it stores fused
heads in fp32 because "`final_layer`'s two output projections are the
checkpoint's fp32 island". On a pruned int8 base that island does not exist, so
the stated reason does not hold there. The *decision* still does — the node's
heads run against an fp32 activation from `mod()` — and at `strength=1.0` the
base head cancels out of `base + strength * (fused - base)` anyway, so nothing
numerical rides on it. The reasoning is what needs updating, not the code.

### 1.6 The int8 weight is not dequantized at load

`comfy/ops.py:1225-1229` (*read*):

```python
        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )
```

int8 bytes go straight to device as `_qdata`. `orig_dtype` records only what the
dequantizer should target if anyone asks. **Nothing on the load path calls
`dequantize()`** — it is reachable only from a LoRA `weight_function`, from
`cast_bias_weight`'s dtype-mismatch branch, or from `convert_weight`.

`QUANT_ALGOS["int8_tensorwise"]` (`comfy/quant_ops.py:262-267`) carries
`"quantize_input": False`, which is what puts int8 on a different forward route
from fp8/nvfp4: the module does not pre-wrap the input, the kernel quantizes it
itself.

### 1.7 What convrot is, and where each half lives

A **regular Hadamard of order 256**, applied group-wise along the input
dimension. `.venv/lib/python3.13/site-packages/comfy_kitchen/tensor/int8_utils.py:11-37` (*read*):

```python
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], ...)
    h = h4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h_normalized = h / (size**0.5)
```

256 is `4^4`, so `H = kron(h4,h4,h4,h4)/16`. This is the *regular* Hadamard (the
4x4 seed's rows sum to 2), not Sylvester — hence the power-of-4 demand. It is
deterministic, cached, and **never stored**: only `convrot_groupsize` is in the
checkpoint.

The rotation is **baked into the weight offline** and **applied to the
activation online**:

```python
def _rotate_weight(weight, h, group_size):
    """Rotate weight matrix offline: W_rot = W @ H_block^T."""
def _rotate_activation(x, h, group_size):
    """Rotate activation online using Optimized Matmul implementation."""
```

`(xH)(WH^T)^T = x H H^T W^T = x W^T` because `H` is orthogonal, so the pair is
exactly identity in infinite precision. Its whole purpose is to spread outlier
channels across each 256-wide group before rounding, so the int8 grid is spent
on a flatter distribution. This is why every H3 int8 layer has an input
dimension divisible by 256 — 5376 = 21x256, 14336 = 56x256, 7168 = 28x256 — and
why the kernel raises rather than falling back if that fails.

`H` is symmetric, so `_rotate_weight` is its own inverse and the same function
un-rotates during `dequantize`.

### 1.8 The matmul, and where the accumulation happens

Dispatch, `comfy/ops.py:1373-1379` (*read*):

```python
                _use_quantized = (
                    getattr(self, 'layout_type', None) is not None and
                    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                    not getattr(self, 'comfy_force_cast_weights', False) and
                    len(self.weight_function) == 0 and len(self.bias_function) == 0
                )
```

Four ways off the int8 path, all visible in one expression. For the DiT all four
hold, so `weight_only_quant = True` and `forward_comfy_cast_weights` takes the
arm at `comfy/ops.py:1341-1355` that passes `dtype=self.weight.dtype` — making
`cast_bias_weight`'s `weight.dtype != dtype` test false, so the weight survives
the cast still quantized. `torch.nn.functional.linear` then routes through
`__torch_function__` to `comfy_kitchen`'s registered op.

The arithmetic, `.venv/lib/python3.13/site-packages/comfy_kitchen/backends/eager/quantization.py:971-1057` (the
eager reference; the CUDA backend fuses the same steps):

```python
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)          # bf16
    x_8, x_scale = quantize_int8_rowwise(x_2d)                   # per-row absmax/127
    result = _int8_matmul_accumulate(x_8, weight.T.contiguous()) # INT32
        chunk = result[i:end_i].float()
        chunk_scales = x_scale[i:end_i].to(torch.float32) * weight_scale
        chunk_scaled = (chunk * chunk_scales).to(out_dtype)      # fp32 scale, one cast to bf16
```

**Accumulation is int32 and exact.** At K=5376 the worst case is
`5376 * 127 * 127 = 8.67e7`, comfortably inside int32. All the error is in the
two roundings; the scale application runs in fp32 and there is exactly one cast
to bf16 at the end. The activation rotation itself runs in bf16.

On this box the real kernel is the CUDA backend
(`.venv/lib/python3.13/site-packages/comfy_kitchen/backends/cuda/__init__.py:1854-2058`), which fuses the rotation,
the row-wise quantization and (where applicable) the SwiGLU into one kernel, then
either a CUTLASS int8 GEMM with a fused fp32-dequant epilogue or a cuBLAS int8
GEMM plus a separate dequant kernel. Both accumulate int32 and scale fp32.

### 1.9 The SwiGLU fusion, which is H3-specific

`comfy/ldm/minimax/model.py:199-206` (*read*):

```python
class MLP(nn.Module):
    def forward(self, x):
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
```

`fc2` does not go through `Linear.forward` at all. `comfy/ops.py:958-991`
pushes the activation *inside* the int8 quantizer kernel so the intermediate is
never written to HBM. At S = 104,466 that intermediate is **2.79 GiB** in bf16
(*measured*: `104466 x 14336 x 2`), which is the whole argument for the fusion.

The eager fallback names the SwiGLU convention out loud
(`comfy/ops.py:947-949`):

```python
def _swiglu_eager(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)
```

SiLU on the **first** half. The release stores `[gate; value]` and ComfyUI reads
it as stored; only diffusers swaps, at conversion time. This is also the swap
`bench/convert_pdd_lora.py` performs on `mlp.fc1`'s `lora_B` row halves.

**The fusion is exclusive to int8 with the tensorwise layout.** The first branch
of `linear_input_act` falls back to eager SwiGLU + ordinary `linear` for every
other weight representation, so a bf16 or fp8 H3 pays a 2.79 GiB materialization
per block per step that the int8 one does not. It also drops out the moment a
LoRA weight function is attached to `fc2` — §13.3.

### 1.10 What this box actually does at load time

*log*, from a real render on 2026-08-29:

```
Requested to load MiniMaxH3
0 models unloaded.
Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 308 patches attached.
  Force pre-loaded 210 weights: 1175 KB.
```

**This is the dynamic vbar path, not the classic lowvram split.**
`CoreModelPatcher.is_dynamic()` returns `True` (`comfy/model_patcher.py:1791`),
so `ModelPatcher.load`'s greedy largest-first residency loop is not what runs.
The 19,995 MB is staged into a pinned host buffer and faulted in per module
through `comfy_aimdo.model_vbar`, driven from the DiT's own block loop by
`comfy.model_prefetch.make_prefetch_queue` (§9.1). The 210 force-preloaded
weights at 1,175 KB are the small always-resident tensors.

The classic path's budget is still worth knowing, because it is what a
non-dynamic patcher would do. `comfy/sampler_helpers.py:165-179` computes
`memory_required` for a doubled batch (a CFG pair this graph never runs) and
`minimum_memory_required` for one. With `memory_usage_factor = 0.114`
(`comfy/supported_models.py:973`, unchanged since the support commit) and the
packed latent's 9,907,136 elements, *measured*:

```
memory_required (batch*2)    =    44.12 GiB
minimum_memory_required      =    22.06 GiB
```

On a 24 GiB card, `lowvram_model_memory = max(0, free - 22.06 GiB)` — under a
gigabyte of resident weights at best. **The estimate is a heuristic, not a
measurement**, and it ignores conditioning length entirely for H3 (there is no
`memory_usage_factor_conds` and no `extra_conds_shapes` override), so a
ref-heavy graph is under-budgeted by construction while a long t2va one is
over-budgeted. Either way the conclusion for this box is the same: the DiT
streams.

---

## 2. Loading the text encoder

### 2.1 The dropdown does not matter

**There is no `minimax_h3` CLIPLoader type.** The string is `minimax`
(`CLIPType.MINIMAX = 35`); `minimax_h3` is a DiT `image_model` string.
And for this file the type is irrelevant — the H3 branch is reached purely by
state-dict shape detection (`comfy/sd.py:1674-1676`, *read*):

```python
    if "visual.deepstack_merger_list.0.norm.weight" in sd and "model.layers.49.self_attn.q_proj.weight" in sd:
        # MiniMax H3 conditioning encoder: Qwen3-VL-32B, truncated to 50 layers
        return TEModel.QWEN3VL_32B
```

Two discriminators: `visual.` **without** the `model.` prefix (the 4B/8B branch
tests the prefixed key), and the presence of layer 49. Selecting
`stable_diffusion` in the dropdown would load it identically. The dispatch at
`comfy/sd.py:1924-1926` is conditioned on nothing but `te_model`, and — unlike
every neighbouring Qwen3-VL branch — applies no `state_dict_prefix_replace`,
because the converted checkpoint already ships in that layout.

### 2.2 A physically 50-layer stack with no final norm

`comfy/text_encoders/llama.py:288-297` (*read*):

```python
@dataclass
class Qwen3VL_32BConfig(Qwen3VL_8BConfig):
    # MiniMax H3 conditioning checkpoint: truncated to the first 50 of 64 layers,
    # consumed as the unnormalized hidden state after layer 50 (no final norm, no lm_head)
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 50
    num_attention_heads: int = 64
    lm_head: bool = False
    final_norm: bool = False
```

`final_norm=False` makes `self.norm = None` (`comfy/text_encoders/llama.py:785-791`),
so the returned tensor is the raw post-layer-50 residual stream. This is not
runtime slicing — the checkpoint genuinely has nothing past layer 49.
`64 heads x 128 = 8192` against `hidden_size 5120` gives a non-square q_proj,
confirmed in the file: `q_proj` is `I8 [8192, 5120]`, `k_proj`/`v_proj` are
`[1024, 5120]` (8 KV heads), `o_proj` is `[5120, 8192]`.

`MiniMaxH3ClipModel` (`comfy/text_encoders/minimax.py:106-111`) accepts `layer`
and `layer_idx` and then **ignores them**, passing literals up:

```python
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=MiniMaxQwen3VL, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)
```

Consequences: `CLIPSetLastLayer` cannot reach it; no padding mask is ever built;
and **`pooled_output` is always `None`**, because `Llama2_.forward` returns a
2-tuple and `len(outputs) >= 3` is never true.

### 2.3 The output tensor

| | |
|---|---|
| shape | `[1, 500, 5120]` for this prompt |
| dtype | `torch.float32` |
| device | CPU (`intermediate_device()`), unless `--gpu-only` |
| `pooled_output` | `None` |
| extras | `minimax_token_tags`: `int64 [500]`, all 1s for t2va |

The fp32 comes from a hardcoded upcast at the embedding
(`comfy/sd1_clip.py:213`, `out_dtype=torch.float32`) which `Llama2_.forward`
never downcasts, and `comfy/rmsnorm.py:7-11` casts the *weight* to `x.dtype`
rather than the reverse. `comfy/sd.py:269-270` makes the compute dtype match:

```python
        #Match torch.float32 hardcode upcast in TE implemention
        self.patcher.set_model_compute_dtype(torch.float32)
```

**So the encoder runs fp32 activations through all 50 layers.** The BF16 that
`llama_detect` pulls off `model.layers.0.input_layernorm.weight` is the
*construction* dtype, not the execution dtype.

The mask is causal-only and dense: `enable_attention_masks=False` means
`attention_mask` arrives `None`, so only the `triu_(1)` causal mask is built —
a `[S, S]` fp32 tensor, ~1 MB at 500 tokens.

Attention resolves inside the encoder's own forward
(`comfy/text_encoders/llama.py:857`, *read*):

```python
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
```

`small_input=True` short-circuits to `attention_pytorch` before the `mask`
argument is consulted. **No `MODEL`-input attention patcher can reach the
encoder** — Sol, sage and the SLA router all take `io.Model.Input` and patch the
DiT. This is the two-stage confusion `CLAUDE.md` warns about, confirmed at the
call site.

### 2.4 The int8 path exists, is available, and is never taken

This is the finding worth carrying out of this document.

*measured*, over the 1602 tensors: **350 are quantized** — 50 layers x 7 linears
(q, k, v, o, gate, up, down), each with `.weight I8`, `.weight_scale F32
[out, 1]`, `.comfy_quant U8[72]` carrying the same
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
**The entire 351-key vision tower is BF16, and so is `model.embed_tokens`.**

Two independent gates disqualify every one of those 350 layers from the int8
GEMM. `comfy/sd1_clip.py:114` (*read*):

```python
                operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)
```

hardcoded for **every** quantized text encoder, and `comfy/ops.py:1151-1152`
only ever raises that flag, never lowers it. And `comfy/sd.py:270`'s
`set_model_compute_dtype(torch.float32)` sets `force_cast_weights = True`
(`comfy/model_patcher.py:743`), stamped onto every module as
`comfy_force_cast_weights` at load (`comfy/model_patcher.py:1016`). Either alone
makes `_use_quantized` False at `comfy/ops.py:1373-1379`.

With `_use_quantized = False` and `weight_only_quant` False,
`forward_comfy_cast_weights` takes the plain branch and
`comfy/ops.py:431-436` runs:

```python
    if weight_has_function or weight.dtype != dtype:
        weight = weight.to(dtype=dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
```

`weight.dtype` is BF16 (`orig_dtype`), `dtype` is `input.dtype` = fp32. They
differ, so **every linear dequantizes its full weight to fp32 on every forward**,
un-rotating the Hadamard as it goes
(`.venv/lib/python3.13/site-packages/comfy_kitchen/tensor/int8.py:159-177` -> `dequantize_int8_convrot_weight`),
and then runs an fp32 `F.linear`. There is no cache.

The reader that found this confirmed it by executing the load and forward with
the real `comfy_quant` blob and spying on both entry points:

```
_full_precision_mm after load: True
weight type: QuantizedTensor    storage dtype: torch.int8
params.convrot: True   gs: 256
int8_linear calls: 0    dequantize calls: 1
```

I re-derived the two gates independently from source (`comfy/sd1_clip.py:114`,
`comfy/sd.py:270`, `comfy/model_patcher.py:743`, `comfy/ops.py:1373-1379`) and
they hold.

| question | answer |
|---|---|
| per-tensor or per-row scale? | per-row, despite the `int8_tensorwise` name |
| where does dequantization happen? | `comfy/ops.py:431-436`, per matmul, per forward, uncached |
| accumulate dtype? | **fp32.** The int8 GEMM never runs |
| what does int8 storage buy? | disk, host RAM, PCIe volume. **No arithmetic speed at all** |

**Is this a bug? No — it is a deliberate policy, and the git history says so.**
`full_precision_mm=True` is the direct continuation of what `25022e0b`
(2025-11-24, "Cleanup and fix issues with text encoder quants") replaced:
`scaled_fp8_ops(fp8_matrix_mult=False, ...)`. Upstream's position is that a text
encoder stores quantized and computes in full precision. It conditions
everything downstream from a single pass, so the trade is defensible.

**What it costs and buys, measured rather than reasoned.** Same layer
(`model.layers.25.mlp.gate_proj`, `[25600, 5120]`), same box, weights already
resident on the GPU:

| | int8_convrot | bf16 |
|---|---|---|
| whole-encoder file | **25.28 GiB** | **47.97 GiB** |
| — LM layers (24.38 G elements either way) | 22.72 GiB | 45.41 GiB |
| — `embed_tokens` + vision tower | 1.45 + 1.11 GiB, BF16 in both | same |
| weight -> fp32, per forward | 0.78 ms | 0.87 ms |
| fp32 GEMM at 500 tokens | 1.87 ms | 1.87 ms |
| host -> device, per layer | **10.87 ms** (125 MiB) | **21.75 ms** (250 MiB) |
| weight error vs the bf16 source | **8.8e-3** | 0 (it *is* the source) |

Three things fall out, and the first two are the opposite of what "never runs
int8 arithmetic" suggests:

**The dequantization is not a tax.** int8 -> fp32 with the Hadamard un-rotation
is *faster* than a bf16 -> fp32 cast (0.78 vs 0.87 ms), because both are
memory-bound on writing the same 524 MiB fp32 result and int8 reads half as many
input bytes. The un-rotation itself costs 0.02 ms over a plain int8 dequant —
and it is genuinely happening (*measured*: a convrot round trip reproduces the
source to 7.9e-3, where skipping the rotation would be O(1) wrong).

**Transfer dominates both by ~5x, and that is where int8 pays.** 10.87 ms of
PCIe against 1.87 ms of GEMM, per layer. The encoder does not fit in 24 GiB in
either format, so both stream; int8 streams half the bytes. **That is the reason
to keep using it over the bf16 file you already have** — not disk space, and not
arithmetic.

**The price is 0.88% relative weight error**, uniform across the quantized
linears and consistent with what `docs/evidence.md` records for the DiT's int8
lane.

*Corrected 2026-08-29, same day, by measurement.* An earlier version of this
section said the fp32 dequantization was "consistent with this repo's own
recorded ~691 s encode". **It is not, and the arithmetic was never done.** A
full streaming pass of the 25.28 GiB model at the 12 GiB/s measured above is
~2 s; all 350 GEMMs at 500 tokens are ~0.7 s; the dequantizations are ~0.3 s.
Those sum to about 3 s against 691 s on record, so **the encode's real
bottleneck is none of the three and remains unexplained.** Anyone optimizing it
should profile first rather than reason from this section.

**If you wanted the int8 GEMM anyway**, two edits are needed and neither is
local: dropping `full_precision_mm=True` at `comfy/sd1_clip.py:114` and dropping
`set_model_compute_dtype(torch.float32)` at `comfy/sd.py:270`. Both are global to
every text encoder, the second changes encoder numerics for every model in the
tree, and on this evidence the win would be at most the 1.87 ms GEMM per layer
against a 10.87 ms transfer — i.e. under 15%, for a numerical change upstream
deliberately avoided. Not worth it.

**Contrast with the DiT**, which is the same quantization scheme in the same
wheel and *does* take the int8 kernel: `pick_operations` passes no
`full_precision_mm` (it defaults `False`, `comfy/ops.py:1290`) and nothing calls
`set_model_compute_dtype` on the diffusion patcher, so all four clauses of
`_use_quantized` hold. The DiT is also the case where the kernel matters — it
runs 50 blocks x 8 steps rather than one pass, and its activations are bf16
rather than fp32, so the int8 path is both reachable and worth reaching.


---

## 3. Loading the two VAEs

Both are keyed off state-dict probes, not metadata (`comfy/sd.py:995`, `:1030`,
*read*):

```python
            elif "decoder.transformer_blocks.0.scale1" in sd and "encoder.down.5.block.0.conv1.weight" in sd:  # MiniMax H3 video VAE
                minimax_ops = comfy.ops.disable_weight_init
                minimax_quant = comfy.utils.detect_layer_quantization(sd, "")
                if minimax_quant is not None:  # int8+convrot quantized decoder
                    minimax_ops = comfy.ops.mixed_precision_ops(minimax_quant, dtype if dtype is not None else torch.float16)
                self.first_stage_model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(operations=minimax_ops)
...
            elif "pre_block.attn.zero_k_bias" in sd:  # MiniMax H3 audio VAE (DAC encoder + BigVGAN decoder)
                self.first_stage_model = comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE()
```

For the fp16 video VAE this box loads, `detect_layer_quantization` returns
`None` and `operations` stays `disable_weight_init`. Note the `operations=`
override, when it does fire, reaches **only the decoder** — `EncoderFCN3D`,
`quant_conv` and `post_quant_conv` are built from the module-global `ops`.

`vae_dtype` (`comfy/model_management.py:1262`) picks from each VAE's declared
`working_dtypes`:

- video: `[torch.float16, torch.float32]` — **bf16 is deliberately excluded**, so
  on a card where `should_use_fp16` is false it silently runs fp32 rather than
  bf16. On this box: **fp16**.
- audio: `[torch.float32]` — always fp32.

The audio VAE's fp32-only status is not conservatism. `SnakeBeta` stores its
alpha and beta in log space and exponentiates per call
(`comfy/ldm/minimax/audio_vae.py:50`), so any `alpha > 11.09` overflows fp16,
and `(beta + 1e-9).reciprocal()` is just as fragile. Note that `--fp16-vae`
short-circuits before `working_dtypes` is consulted, so it *would* force the
audio VAE into that hazard.

---

## 4. Scenario 2 only: what the PDD node installs

`MiniMaxH3PDDLoRA.execute` runs once, at patch time, before any sampling. It
touches four surfaces.

### 4.1 The converted artifact

*measured*, `minimax_h3_fl2va_pdd_8step_comfy.safetensors`, 730 tensors,
1.044 GiB:

| prefix | count | what |
|---|---|---|
| `diffusion_model.*` | 624 | 208 modules x {`lora_A.weight` BF16, `lora_B.weight` BF16, `alpha` F32} |
| `h3_pdd.adaln_baked.*` | 100 | 50 x {`diff` F16 `[96768, 8]`, `diff_b` F32 `[96768]`} |
| `h3_pdd.bank.*` | 4 | video `[32, 96, 5376]` + bias, audio `[32, 32, 5376]` + bias, BF16 |
| `h3_pdd.adaln_table` | 1 | F32 `[1025, 8]` |
| `h3_pdd.base_video_out` | 1 | F32 `[96, 5376]` |

Metadata: `pdd_num_steps 32`, `pdd_block_size 4`, `pdd_nfe 8`,
`lora_rank 64`, `lora_alpha 64.0` (so `alpha/rank` = **1.0**),
`pdd_shift_video 12.0`, `pdd_shift_audio 3.0`, `adaln_baked_blocks 50`,
`h3_pdd_pruned_base minimax_h3_fl2va_pruned_int8_convrot.safetensors`.

208 backbone modules = 52 blocks (50 + 2 refiner) x 4 targets
(`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`).

### 4.2 The three guards that run before anything is installed

**Partition, by distance not by hash** (`pdd_lora.py:996-1029`). `dist =
||live - ref|| / ||ref||` against `PARTITION_TOLERANCE = 0.015`; the two
partitions sit ~0.05 apart and a load-time cast moves it a few thousandths. The
shape check runs first, and its error message names the exact failure it exists
for: an enlarged `[32*96, 5376]` head bank left resident on a cached model by
core's PDD path (§10.3).

**Curve table, against `TABLE_TOLERANCE = 5e-3`** — a bf16 cast of the table is
0.00164 away and the two partitions' tables are 0.01835 apart, so the threshold
sits between them. A mismatch **falls back** to the runtime injection rather
than raising, because the injection is correct on any pruned base.

**Bank encoding, at conversion time.** `bench/convert_pdd_lora.py::assert_bank_verbatim`
refuses a source whose head stack looks delta-encoded. *measured* on the file
this box holds:

```
video: median(rows1..)/row0 = 1.0010   ||row_i - row_0||/||row_0|| in [0.0259, 0.0500]
audio: median(rows1..)/row0 = 0.9999   ||row_i - row_0||/||row_0|| in [0.0127, 0.0456]
```

The published bank is **verbatim heads**. Consecutive heads differ by 2.6-5% of
a head's own norm. This matters in §10.3.

### 4.3 The backbone: a dequantise/patch/requantise round trip

`comfy.lora.load_lora` resolves the 624 `diffusion_model.*` tensors plus the 100
baked `diff`/`diff_b` into **308 patch keys**, and `m.add_patches(loaded, 1.0)`
records them. *log* confirms the count reaching the model:

```
Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 308 patches attached.
```

208 backbone weight keys + 50 `adaln_proj.linear.weight` + 50
`adaln_proj.linear.bias` = 308. The node asserts `len(applied) == len(loaded)`
and raises otherwise, which is the guard against a layout change renaming a
subset.

**What happens to an int8 weight under a patch depends on which loader path it
is on**, and the two are not equivalent:

*Classic resident path* (`comfy/model_patcher.py:899-928`): move to device,
`convert_func` -> `weight.dequantize()`, `calculate_weight` in
`lora_compute_dtype` (**fp16** on this card, per
`comfy/model_management.py:2029-2040`), then `set_func` ->
`requantize_from_float(..., scale="recalculate", stochastic_rounding=seed)`.
The scale is recomputed from the *patched* absmax, and the seed is
`string_to_seed(key)` — deterministic per layer, so the render reproduces.

*Dynamic vbar path* — **what this box runs**. `comfy/model_patcher.py:1929-1935`
installs the patch as `weight_lowvram_function` and leaves `weight_function`
empty:

```python
                        lowvram_patch = LowVramPatch(key, self.patches)
                        lowvram_patch._pin_state = pin_state
                        setattr(m, param_key + "_lowvram_function", lowvram_patch)
```

That distinction is load-bearing. In `resolve_cast_module_with_vbar`
(`comfy/ops.py:308-322`) the requant is gated on `len(fns) == 0`, where `fns` is
`weight_function`:

```python
        if not resident and lowvram_fn is not None:
            x = to_dequant(x, dtype if compute_dtype is None else compute_dtype)
            x = lowvram_fn(x)
            if (want_requant and len(fns) == 0 or update_weight):
                ...
                    y = orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)
            if want_requant and len(fns) == 0:
                x = y
            if update_weight:
                orig.copy_(y)
```

So on this box a patched int8 layer **is** requantized and **does** keep the
int8 GEMM, and `orig.copy_(y)` writes the result back into the staged buffer so
the LoRA is applied once per resident signature rather than once per forward.
`LowVramPatch.__call__` passes `intermediate_dtype=weight.dtype`, so the LoRA
matmul itself runs in **bf16** here — where the classic resident path would run
it in fp16.

**Two consequences worth stating plainly.** The LoRA changes the quantization
grid, not just the weight, because `scale="recalculate"` refits the absmax — so
two strengths are not two points on a smooth curve through one quantization. And
the resident and dynamic paths run the same LoRA at different precisions, decided
by how much VRAM was free, so any LoRA A/B on this model is comparing two things
at once unless headroom is held fixed.

The escape hatch exists because it had to: `79c555ce` (2026-06-28, "Fix int8 mm
being skipped on offloaded lora weights") changed one argument from
`want_requant=want_requant` to `want_requant=True`, and `470ac36a` (2026-06-26)
replaced `QuantizedTensor.from_float(x, s.layout_type, ...)` with
`orig.requantize_from_float(...)` because naming a layout is not describing a
quantization — `from_float` loses `convrot` and `per_channel`, producing a
valid-looking unrotated tensor-wise weight with no error.

### 4.4 The adaln: a weight patch, because the bake fit

On a pruned base the LoRA's 2688-dim adaln update has nowhere to live. The
converter pre-solves it into the checkpoint's own rank-8 curve basis
(affine — the basis plus a constant column, because the pruned form is an SVD of
the *centred* time curve and the mean lives in the bias), refusing at 1e-3 per
block. That turns it into an ordinary `diff`/`diff_b` pair, which
`comfy.lora.load_lora` maps at `comfy/lora.py:72-82` and which takes `strength`
natively. **50 forward patches are therefore not installed**, and the log line
says which of the three adaln forms ran.

### 4.5 The heads: three object patches, and where they sit

```python
m.add_object_patch("diffusion_model.final_layer.forward", _make_final_layer_forward(...))
m.add_object_patch("diffusion_model.forward",              _make_capture_forward(...))
m.add_object_patch("diffusion_model.final_layer.video_out.forward", _make_head_forward(...))
m.add_object_patch("diffusion_model.final_layer.audio_out.forward", _make_head_forward(...))
```

`add_object_patch` sets an instance attribute named `forward`. `nn.Module.__call__`
resolves `self.forward` through the instance dict first, so
`self.video_out(...)` inside the stock `FinalLayer.forward` reaches the patch —
which is why the node can own the output projection without touching the
modulation maths around it.

The `diffusion_model.forward` patch is the only one that sees
`transformer_options`, which is where `sample_sigmas` and the graph's shift live.
It is installed **outside** the head gate, so a `patch_heads=False` or
`strength=0.0` arm still gets the shift guard. It chains onto whatever forward
was already installed rather than replacing it.

`_make_final_layer_forward` binds four positional names and forwards `*args`.
That is not defensive padding: core widened `FinalLayer.forward` to
`(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)` in `2504e68d`
(2026-08-29, "MiniMax-H3: Support PDD LoRA"), and a signature pinned to four
would raise `TypeError` on the first sampling step.

`HEAD_PATCH_KEYS` is checked for a clash before installing, because
`add_object_patch` is last-writer-wins per key and two owners of `video_out` is a
plausible wrong render with nothing said.

---

## 5. Prompt to conditioning

### 5.1 The presentation is the raw prompt

`MiniMaxH3Tokenizer.tokenize_with_weights` (`comfy/text_encoders/minimax.py:148-202`,
*read*) with no images and no refs reduces to a single `add_text(text)`:

```python
        def add_text(s):
            if not s:
                return
            token_batches = self.qwen3vl_32b.tokenize_with_weights(
                s, return_word_ids=False, disable_weights=True,
            )
            if len(token_batches) != 1:
                raise ValueError("MiniMax H3 text segment exceeds the supported prompt length.")
            entries.extend(token_batches[0])
```

No chat template — `MiniMaxH3Tokenizer` subclasses `SD1Tokenizer` directly, not
`Qwen3VLTokenizer` where `llama_template` lives, and it overrides
`tokenize_with_weights` outright. No BOS, no EOS, no vision tokens.

`Qwen3VLSDTokenizer`'s configuration (`comfy/text_encoders/qwen3vl.py:150-151`)
disables every chunking and padding mechanism in `SDTokenizer`:

```python
        super().__init__(tokenizer_path, pad_with_end=False, ..., tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False,
                         max_length=99999999, min_length=1, pad_token=151643, ...)
```

**No chunking, no truncation, no padding, no prompt weighting.** The 77-token
weighted-chunk machinery is present and entirely inert: the batching loop's
`len(t_group) + len(batch) > self.max_length - has_end_token` never fires at
`max_length = 99,999,999`. `disable_weights=True` sets
`parsed_weights = [(text, 1.0)]` (`comfy/sd1_clip.py:585-588`), so `(cat:1.2)`
is literal text — and, because `has_weights` is then always False, there is
exactly **one** encoder forward per encode rather than the two an
unconditional-reference encode would need.

The `raise ValueError` above is dead on this path for the same reason
`max_length` is unreachable.

### 5.2 What this prompt tokenizes to

*measured*, loading the bundled tokenizer with the seven markers registered:

```
len(tokenizer): 151676
prompt tokens: 500
first ids: [396, 47172, 26290, 318, 57597, 11448, 510, 58, 36402, 220, 16, 60]
```

500 tokens, no padding. `S = 500 + 1150 + 102816 = 104,466`.

The seven H3 markers are registered by key only
(`comfy/text_encoders/minimax.py:129-133`), the dict's values being unasserted
documentation. *measured*, all seven resolve to their documented ids and all
seven round-trip as **one** token:

```
<d> 151669  </d> 151670  <|cutoff|> 151671  <|lyrics_start|> 151672
<|lyrics_end|> 151673  <|caption_start|> 151674  <|caption_end|> 151675
"a cat <d> walking" -> [64, 8251, 220, 151669, 11435]
```

This prompt uses none of them, so it is bit-identical before and after
`924743af`. Two ordering facts make the marker path work at all: `e5a38e3f`
(2026-08-18) rerouted `add_text` from a raw HF call to
`Qwen3VLSDTokenizer.tokenize_with_weights`, and `924743af` (2026-08-23) added
the markers to that same tokenizer instance. Either alone does nothing.

### 5.3 The tags ride out on a mutable attribute

`comfy/text_encoders/minimax.py:113-121` is the only reason
`MiniMaxH3ClipModel` exists:

```python
    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out
```

`last_token_tags` is set on the transformer during forward and read afterwards.
It is not per-batch-element and it is stale-readable if a forward is skipped.
For t2va it is `torch.ones(500, dtype=long)`.

### 5.4 `extra_conds` runs the refiner once, and hides the fp32

`comfy/model_base.py:2164-2213` (*read*). The two lines that matter most:

```python
        if cross_attn is not None:
            # run condition_proj + token refiner once per sampling instead of per step
            cross_attn = self.diffusion_model.preprocess_text_embeds(
                cross_attn.to(device=kwargs["device"], dtype=self.get_dtype_inference()))
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
```

**`c_crossattn` is not raw Qwen states.** `condition_proj` (5120 -> 5376) and the
2-block token refiner run here, once per sampling run, and the result is cached
in the cond. That is why those three modules are the unquantized BF16 ones in the
checkpoint — they run 1 time, not 8. `preprocess_text_embeds` is idempotent by
width check, so anything injecting pre-refined 5376-wide states is respected.

Note it is `CONDRegular`, not `CONDCrossAttn`. `CONDCrossAttn.can_concat` pads to
an LCM sequence length to batch prompts of different lengths; `CONDRegular`
requires exact shape equality. **Two prompts of different token length can never
share a batch on H3** — deliberate, because the packed layout is built against a
specific `cross_attn.shape[1]`.

Everything else H3-specific goes into one opaque dict:

```python
        # Everything H3-specific rides in one dict so _apply_model's dtype cast
        # (which would flatten fp32 cond latents and long tags to bf16) skips it.
        payload = {}
```

`BaseModel._apply_model` casts every cond with a `.dtype` to bf16. A dict has
none, so the sweep walks past it. That is the whole reason there is one
`minimax_payload` rather than a dozen typed conds, and **it is a contract nothing
asserts** — anything H3-specific added outside this dict is silently bf16'd.

For t2va the payload is `{"seed": <sampler seed>, "audio_scale": 4.0,
"text_token_tags": <[500] int64>, "layout": <PackedLayout>}`. No keyframes, no
refs, no cond latents.

The layout is built once here:

```python
            payload["layout"] = comfy.ldm.minimax.model.PackedLayout(
                cross_attn.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2,
                latent_shapes[1][-1], keyframes=..., refs=...)
```

`(vs[3] + 1) // 2 * 2` ceilings H and W to even for the 2x2 patch. 48 and 84 are
already even.

---

## 6. The packed layout, for this render

`PackedLayout.__init__` (`comfy/ldm/minimax/model.py:344-462`). *measured*, by
instantiating it with this render's numbers:

```
seq_len 104466
segments [(0, 500, 'text'), (500, 1650, 'audio'), (1650, 104466, 'video')]
position_ids (104466, 3) torch.float64
```

Three segments, in that order. `[text | audio | video]` — for t2va there are no
`cond` or `ref` rows at all, and target audio is always immediately before target
video.

The position grid, *measured* by reading rows out of `layout.position_ids`:

| segment | rows | t | h | w |
|---|---|---|---|---|
| text | 0..499 | `0.0 .. 499.0` | 0 | 0 |
| audio ch0 | 500..1074 | `500.0 .. 1074.0` | 0 | `-5.166010` |
| audio ch1 | 1075..1649 | `500.0 .. 1074.0` | 0 | `36.158105` |
| video | 1650..104465 | 102 distinct, `500.0 .. 1068.333` | 24 values, `3.905 .. 27.087` | 42 values, `-5.166 .. 36.158` |

Five things fall out of that table:

**The video time axis continues from where the text axis ends.** `cursor =
float(text_len)` = 500.0. So changing the prompt length shifts every video and
audio position id. A 400-token prompt and a 500-token one are not the same
geometry.

**One clock for both streams.** Video spans `5/3 * (1,4,4,4,4)` per latent frame
by exclusive cumsum; audio advances 1.0 per latent frame. *measured*: the video
spans sum to `345 * 5/3 = 575.0` and the audio covers 575 frames, so both streams
cover `[500, 1075)` exactly. At 24 fps, `24 * 5/3 = 40 Hz`, so one unit is 1/40 s
in both.

**Audio rows carry `h = 0` and are pinned to the two extremes of the video's w
grid**, one per stereo channel. That is the only thing distinguishing the
channels positionally.

**The spatial axes are area-normalized and go negative on a landscape canvas.**
`_axis_from_sqrt_area` gives `ratio = dim / sqrt(h*w)`, then
`linspace((1-r)/2, (1+r)/2, dim//2, endpoint=False) * 32`. For 48x84,
`sqrt(4032) = 63.5`, so `r_h = 0.756` and `r_w = 1.323`: the h axis occupies
`0.756 * 32 = 24.2` units centred on 16, the w axis `42.3` units centred on 16,
which puts w's first value at `-5.166`. Built in **float64**.

**The layout is cached in the payload and rebuilt only on a signature miss.**
`_forward` re-derives it if `layout.signature != (text_len, latent_t, lat_h,
lat_w, audio_t)`, so a graph that changes canvas mid-run does not silently reuse
a stale grid.

---

## 7. Sampling setup

### 7.1 The latent is flattened before the sampler sees it

`CFGGuider.sample` (`comfy/samplers.py:1275-1290`, *read*):

```python
        if latent_image.is_nested:
            sampler_shapes = [tuple(x.shape) for x in latent_image.unbind()]
            latent_image, latent_shapes = comfy.utils.pack_latents(latent_image.unbind())
            noise, _ = comfy.utils.pack_latents(noise.unbind())
```

`pack_latents` reshapes each stream to `[B, 1, -1]` and concatenates, giving
`[1, 1, 9907136]` (*measured*: 9,870,336 video + 36,800 audio). **Every sampler
runs on that flat vector**; the streams are recovered inside `_apply_model` by
`unpack_latents` using `latent_shapes` from the payload.

### 7.2 Noise, and one reproducibility trap

`comfy/sample.py:22-38` (*read*):

```python
def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    if latent_image.is_nested:
        tensors = latent_image.unbind()
        noises = []
        for t in tensors:
            noises.append(prepare_noise_inner(t, generator, noise_inds))
        noises = comfy.nested_tensor.NestedTensor(noises)
```

CPU only, fp32, then cast to the latent dtype. There is no GPU-RNG path on this
node. **Video noise is drawn first and audio noise continues the same stream**,
so at a fixed seed changing the canvas or the frame count changes the *audio*
noise too. Also worth knowing: `torch.manual_seed` seeds the global default CPU
generator, not a private one.

### 7.3 The sigma schedule

H3 is `ModelType.FLOW_AV` -> `CONST` + `ModelSamplingAV`
(`comfy/model_base.py:144-149`). The 1000-entry table is
`sigmas[i] = 12t/(1+11t)` for `t = (i+1)/1000`
(`comfy/model_sampling.py:298-319` with `time_snr_shift`), giving
`sigma_min = 0.011869` and `sigma_max = 1.0` exactly.

`simple_scheduler` (`comfy/samplers.py:645-652`) is a floored subsample of that
table walking backwards with stride `1000/steps`, plus an appended literal `0.0`
— hence **N steps, N+1 sigmas**. It never interpolates.

*measured*, at 8 steps and shift 12:

```
[1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0]
```

**The final Euler step spans 63.2% of the sigma range.** That is not a property
of the sampler; it is the shift-12 curve, which crushes the schedule into
`[0.8, 1.0]` and leaves the tail to one step. At 4 steps the last step spans 80%.

`BasicScheduler`'s `denoise < 1.0` is **not** a rescale: it builds
`int(steps/denoise)` steps and keeps the last `steps+1` — a tail slice of the
same table.

### 7.4 Scenario 2: the sigmas come from the node instead

`emit_sigmas(shift_v, num_steps, block_w)` returns
`1 - block_bounds(12.0, 32, 4)` = `1 - pdd_time_grid(12, 32)[::4]`. *measured*,
this is **bit-identical** to `calculate_sigmas(model_sampling, "simple", 8)`:

```
simple 8: [1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0]
pdd emit: [1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0]
torch.equal: True    max abs diff: 0.0
```

At 16 steps they differ by 2.1e-3, because `simple` quantizes against its
1000-entry table and `1000 % 16 != 0`; the closed form is the more correct of the
two there.

`resolve_emit_steps` raises if the requested count does not divide 32 — the legal
set is `{1, 2, 4, 8, 16, 32}` — because at any other count no on-grid schedule
exists and the render would complete and merely be wrong.

A custom SIGMAS vector is consumed with **no renormalization anywhere**: no sort,
no clamp, no monotonicity check. The nine places it is touched are the progress
bar (`sigmas.shape[-1] - 1`), the empty check, a device move, the
`transformer_options` stamp, `max_denoise`, `noise_scaling(sigmas[0])`,
`total_steps`, the loop, and `inverse_noise_scaling(sigmas[-1])`.

### 7.5 The guider runs one forward per step

`Guider_Basic` (`comfy_extras/nodes_custom_sampler.py:797-799`) is `CFGGuider`
with `cfg` left at its 1.0 default and no `"negative"` key, so `uncond_` is
`None` twice over. In `_calc_cond_batch` the `None` slot allocates a zeros
buffer and a `1e-37` counts buffer, never runs, and becomes 0; `cfg_function`
computes `0 + (cond_pred - 0) * 1.0`, which is bit-exact. **One DiT forward per
step, batch size 1.** The H3 DiT refuses batch > 1 anyway
(`comfy/ldm/minimax/model.py:583-584`).

### 7.6 `sample_euler` is the reference integrator

`comfy/k_diffusion/sampling.py:189-212`. `KSamplerSelect("euler")` passes empty
`extra_options`, so `s_churn = 0`: the stochastic branch never runs and
`torch.randn_like` is never called. The step is exactly

```
denoised = model(x, sigma_i)
x        = x + (x - denoised)/sigma_i * (sigma_{i+1} - sigma_i)
```

`CONST.calculate_denoised` is `x - v*sigma` and `to_d` is `(x - denoised)/sigma`,
so the two cancel exactly: **the Euler step is stepping on the DiT's raw
velocity output.** Any hook that intercepts `denoised` is looking at a derived
quantity.

The callback fires with the *pre-update* `x`, so `denoised_output` is the x0
estimate at `sigmas[-2]`, not at `sigmas[-1]`. With a terminal 0 the two outputs
are numerically identical (`x + (x-d)/s * (-s) = d`); with a nonzero terminal
sigma they differ, and `inverse_noise_scaling` additionally divides by
`(1 - sigma_terminal)`.

---

## 8. One forward, from the sampler to the DiT

### 8.1 `_apply_model`

`comfy/model_base.py:211-253` (*read*), the six lines that matter:

```python
        xc = self.model_sampling.calculate_input(sigma, x)     # CONST: identity
        dtype = self.get_dtype_inference()                     # bf16
        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()            # sigma * 1000
        if "latent_shapes" in extra_conds:
            xc = utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))   # -> [video, audio]
        model_output = self.diffusion_model(xc, t, context=context, ...)
        ...
        return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)
```

So the DiT receives `x` as a **list** of two tensors, both bf16, and `timestep`
as `sigma * 1000`. `calculate_input` being the identity is the flow-model
difference from EPS models, which divide by `sqrt(sigma^2 + sigma_data^2)`.

### 8.2 The audio carry, and why it is exact

`comfy/ldm/minimax/model.py:553-577` (*read*):

```python
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        if scale != 1.0:
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]

        out = ...WrapperExecutor...execute(x, timestep, context, ...)

        if scale != 1.0:
            # d/d(sigma_v) of the carried variable
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])
```

`audio_scale = shift_v / shift_a = 12/3 = 4.0`, applied on the way in by
`process_latent_in` (`comfy/model_base.py:2158`) and undone on the way out.

**This is a derivation, not a heuristic, and it is exact per forward.** With
`y` the carried audio variable and `z0` the clean audio latent, the design asserts
`y = sigma_v * eps + (1 - sigma_v) * scale * z0`. Multiplying by
`carry = sigma_a/sigma_v` recovers the true audio state
`x_a = sigma_a * eps + (1 - sigma_a) * z0` iff
`sigma_a (1 - sigma_v) scale = sigma_v (1 - sigma_a)`. *measured*, that identity
holds to fp32 rounding across the range:

```
sigma_v   sigma_a    s(1-a)/a      (1-s)*scale     diff
0.999     0.996012   0.00399981    0.00400000     -1.9e-07
0.900     0.692308   0.40000019    0.40000000      1.9e-07
0.500     0.200000   1.99999978    2.00000000     -2.2e-07
0.200     0.058824   3.19999999    3.20000000     -1.3e-08
0.050     0.012987   3.80000000    3.80000000      3.6e-09
```

The output line is the chain rule on the same change of variable. The sampler
needs `dy/d(sigma_v) = eps - scale*z0`; the network returns
`out_a = eps - z0` (after the sign flip); substituting
`z0 = x_a - sigma_a * out_a` gives

```
eps - scale*z0 = (1 - scale) * x_a + (1 + (scale - 1) * sigma_a) * out_a
```

which is the code line verbatim, with `x_a = audio_src * carry`.

**Where it stops being exact is under a PDD fused head**, because the transform's
coefficients are frozen at the step's own sigma while a fused head returns the
block's *mean* velocity. That qualification is
[`pdd/audio_under_pdd.md`](pdd/audio_under_pdd.md)'s to own.

Note the wrapper boundary: the carry is undone **outside** the
`WrappersMP.DIFFUSION_MODEL` executor, so any wrapper sees the stream's own
latent. `93cb5edb` (2026-08-07) renamed `audio_x` to `audio_src` and recomputes
`audio_src * carry` in the output line specifically so a wrapper mutating its
input in place cannot corrupt the correction.

### 8.3 Timesteps and modulation rows

`_forward` (`comfy/ldm/minimax/model.py:579-758`). `dtype = context.dtype` — the
compute dtype is taken from the conditioning tensor, not from the model.

```python
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
```

**`t = 1 - sigma`**, so `t` runs 0 -> 1 as denoising proceeds, and the DiT reads
**one scalar** — per-element sigma variation is discarded, and per-row variation
comes back only through `denoise_mask`.

For t2va the segment timestep map collapses to two distinct values:

```python
        seg_t = {"text": t_v, "video": t_v, "audio": t_a, ...}
        unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments} | ...)
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, ...}
```

Since `shift_v > shift_a`, `sigma_a < sigma_v`, so `t_a > t_v` and
`unique_t = [t_v, t_a]`, `t_row = {t_v: 0, t_a: 1}`. `t_emb` is therefore
`[2, 8]` and `AdalnProj` expands it to `2 * 3 = 6` modulation rows. **Only three
of the six are ever read** (*inference*, from the row arithmetic — a text row at
tag 1 of timestep 0, a video row at tag 0 of timestep 0, an audio row at tag 2 of
timestep 1):

```
mod_segments = [(0, 500, 1), (500, 1650, 5), (1650, 104466, 0)]
```

Three contiguous spans, three integer row indices. That is the whole modulation
plan for a t2va step, and it is what makes ComfyUI's slice-based application
possible (§9.3).

### 8.4 Embedding, and the two fp32 patch projections

```python
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        ...
        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
```

`patchify_video` is `reshape -> einsum("nctrhpwq->nthwcrpq") -> reshape`, giving
row order `(t, h, w)` with w fastest and feature index `4*c + 2*p + q` inside the
96-wide row. `pack_audio` is channel-major: rows `[0, 575)` are channel 0.

The latents are cast to fp32 before the projection — **but the projection weight
is bf16 on this artifact** (§1.5), so the fp32 input is matmul'd against a bf16
weight and the result cast back to bf16. The upcast buys the reshape's exactness,
not the projection's.

The refiner is skipped here because `extra_conds` already ran it:

```python
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states), ...)
```

`context` arrives 5376-wide, so the branch is not taken.

The sequence is then assembled by contiguous slice into one preallocated buffer:

```python
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
```

**1.05 GiB** at this size in bf16 (*measured*: `104466 * 5376 * 2`).

### 8.5 The curve lookup replaces the time embedder

```python
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)
```

Two rows, linearly interpolated out of a 1025-row fp32 table. The
`clamp(max=grid-2)` keeps `t = 1.0` on the last interval rather than reading past
the table. **The table stores the post-SiLU curve**, which is why
`AdalnProj.apply_silu` is `False` on this path — applying SiLU again would square
the nonlinearity. That is the single most dangerous thing to get wrong about a
pruned checkpoint.

The unpruned branch's sinusoid (`comfy/ldm/minimax/model.py:141-152`) puts
**cos before sin** and does not scale `t` by 1000. It does not run here.

### 8.6 RoPE

```python
    def rope_freqs(self, position_ids, device):
        pos = position_ids.to(torch.float32).to(device)          # from float64
        inv = comfy.model_management.cast_to(self.rope.inv_freq, device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)        # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)                # [S, 48]
        return torch.cat((half, half), dim=-1)                   # [S, 96]
```

then `rope_rotation_table` builds `[1, S, 1, 48, 2, 2]` rotation matrices from
the first half (the two halves are identical by construction). *measured*, that
table is 38 MiB in bf16 at this sequence length, rebuilt once per forward.

`rope.inv_freq` is read from the checkpoint, not recomputed. *measured*: the
shipped 16-element fp32 tensor is **bitwise equal** to `10000^(-i/16)` recomputed
in fp32, and off by up to 3.7e-9 if you compute in float64 and cast down — which
is the obvious way to check it and does not match.

Three axes x 16 frequencies = 48, duplicated to 96, so `rot_dim = 96` and
**dims 96..127 of each head pass through unrotated**. Convention is split-half
(`x[i]` pairs with `x[i + rot_dim/2]`), never interleaved.

---

## 9. The block, fifty times

### 9.1 The loop, and the prefetch that makes it possible

```python
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            ...
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
```

`make_prefetch_queue` returns `None` unless
`transformer_options["prefetch_dynamic_vbars"]` is set, which `_apply_model`
sets from `self.current_patcher.is_dynamic()`. On this box it is on (§1.10), so
block `i+1`'s weights are faulted in on a side stream while block `i` computes.
The queue is padded `[None] + blocks + [None]` so the first pop primes and the
trailing pop drains.

`("double_block", i)` in `transformer_options["patches_replace"]["dit"]` is the
per-block replacement hook. Nothing in this trace uses it.

### 9.2 The block

`comfy/ldm/minimax/model.py:286-291`:

```python
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, ...), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
```

`scale` enters as `1 + scale`; **`gate` is applied raw** — no `1 +`, no `tanh`.
Norms are RMSNorm with learnable weight at `eps = 1e-5`.

### 9.3 The modulation, which is where ComfyUI diverges most

`comfy/ldm/minimax/model.py:231-241`:

```python
def _mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b].mul_(1.0 + _mod_row(scale, row, h.dtype)).add_(_mod_row(shift, row, h.dtype))
    return h

def _mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], _mod_row(gate, row, x.dtype))
    return x
```

Three in-place slice ops per call, because the sequence is uniform per segment.
Every reference implementation instead does `scale.index_select(0, adaln_indices)`
over all 104,466 rows for each of six chunks, every block, every step, into new
tensors. **Same arithmetic; roughly 2,400 full-sequence gathers and their
allocations avoided per render.** On a 24 GiB card that is not a
micro-optimization.

`_mod_row` takes either an integer row index or a per-token `LongTensor` — the
generalization `ff6c8a8a` (2026-08-17) added for per-row denoise masks, and the
one `2504e68d` later builds on. For t2va every row is an integer.

The AdaLN projection itself is `Linear(8 -> 96768)` with `apply_silu=False`, then
`view(M*3, 6*5376).chunk(6, -1)`. Chunk order is
`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp` and the row
layout is modality-major (`timestep_index * 3 + tag`, tags 0 video / 1 text /
2 audio).

### 9.4 Attention

`comfy/ldm/minimax/model.py:169-197`:

```python
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            ...
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]; k = k[0]
        v = v.clone()
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        ...
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, ...)
        return self.out_proj(out.squeeze(0))
```

**`split(heads*head_dim, dim=-1)` is a contiguous three-way split**, so ComfyUI
assumes the `[q_all; k_all; v_all]` layout unconditionally. It does not sniff.

**Why is qkv fused at all, and whose choice was it?** The release's. It ships
its DiT in **two** formats (*measured*, over both weight indexes in
`coderef/MiniMax-H3`):

| directory | class | keys | attention layout |
|---|---|---|---|
| `transformer/`, `transformer_ref/` | `MiniMaxH3Transformer3DModel` (diffusers) | 638 | split `to_q` / `to_k` / `to_v` |
| `FL2VA/transformer/`, `Ref2VA/transformer/` | `MiniMaxH3DiTModel` (native) | 535 | **fused `attn.qkv_proj`**, 52 of them |

**The native format's key names are ComfyUI's, exactly** — `blocks.0.attn.qkv_proj.weight`,
`attn.out_proj`, `attn.q_norm`, `mlp.fc1`, `adaln_proj.linear`,
`final_layer.video_out`, `rope.inv_freq`. ComfyUI's H3 implementation is written
against the release's own native checkpoint, and diffusers is the one that
transforms — its converter splits the fused projection to fit the diffusers
`Attention` class. `comfy/model_detection.py:399` reads
`blocks.0.attn.qkv_proj.weight` as a bare subscript, so the diffusers-format
copy cannot load here at all.

The fusion earns its place three ways, all of which matter more here than in a
typical DiT: one GEMM of `[S, 5376] @ [5376, 21504]` instead of three narrower
ones, at a sequence length where arithmetic intensity is the whole game; one
convrot activation rotation shared by all three projections instead of three;
and one set of `weight_scale` rows, one output buffer, one weight fault under
the streaming loader. It costs the layout ambiguity below, and it is why a LoRA
must be fused before it can be applied (§4.3).

**The ambiguity is real and is between the release and the repack.** The native
release shards store the fused rows **per-head interleaved**,
`[q_h|k_h|v_h] x 56`; the Comfy-Org repacks store them **contiguous**,
`[q_all; k_all; v_all]`, which is what the `split` above assumes. Both are
`[21504, 5376]` and `56*3*128` is the row count either way, so a file in the
wrong order loads clean and renders noise. `docs/evidence.md:176-186` records a
`DeepBeepMeep`/WanGP bf16 file that does exactly that.

**Retracted 2026-08-29, the same day it was written.** An earlier version of
this section asserted that "the release ships no fused qkv" and demoted the
interleaved order to a third-party convention, on the strength of grepping
`coderef/MiniMax-H3/transformer/`'s index and finding only `to_q`/`to_k`/`to_v`.
That is one of the release's two formats. The claim was wrong, it was used to
"correct" two other documents that were right, and both have been reverted.

`rms_rope_split_half_` fuses per-head RMSNorm and partial split-half rope into
one in-place kernel writing back into the qkv buffer. The eager reference
(`.venv/lib/python3.13/site-packages/comfy_kitchen/backends/eager/rope.py:67-92`) shows the order and the partial
split exactly:

```python
    x_norm = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=scale, eps=epsilon)
    if rot_dim and rot_dim != x.shape[-1]:
        # partial rotary: rotate the first rot_dim dims, pass the rest through
        rotated = apply_rope_split_half1(x_norm[..., :rot_dim], freqs_cis)
        return torch.cat((rotated, x_norm[..., rot_dim:]), dim=-1)
```

**Norm spans the full 128; rope spans the first 96.** V is neither normed nor
rotated.

`v = v.clone()` looks backwards for a memory optimization and is the opposite.
`q`, `k` and `v` are views into one fused qkv buffer — **4.18 GiB in bf16** at
this sequence length (*measured*: `104466 * 21504 * 2`). While `v` is a view the
whole 3x allocation stays alive through attention. Cloning `v` (1.39 GiB) lets
the 3x buffer be freed as soon as q and k are consumed by the kernel. That is
`62b3c94b` (2026-08-11, "Fix peak memory issue with H3"), one line.

`AttentionTensorContainer` is a single-owner box (`bf4c9a08`): a backend that
prequantizes q/k/v can `take()` and free them. Backends without a
`container_function` get the plain tensors.

`optimized_attention` resolves at **import time** to `attention_pytorch` on this
box, which calls `comfy.ops.scaled_dot_product_attention` — a wrapper that runs
inside an `sdpa_kernel` priority context (`FLASH, CUDNN, EFFICIENT, MATH`) above
128k elements rather than raw `F.sdpa`. `mask=None`, `is_causal=False`: **one
packed document, full bidirectional attention, no mask, no padding.**

ComfyUI passes **no `scale` argument**, so the softmax scale is SDPA's default
`128 ** -0.5` rather than an explicit choice.

### 9.5 What the block costs at this sequence length

*measured*, per block, at S = 104,466:

| buffer | shape | bf16 |
|---|---|---|
| residual stream `h` | `[S, 5376]` | 1.05 GiB |
| fused qkv output | `[S, 21504]` | **4.18 GiB** |
| attention output | `[S, 7168]` | 1.39 GiB |
| `fc1` output | `[S, 28672]` | **5.58 GiB** |
| SwiGLU output | `[S, 14336]` | 2.79 GiB — **never materialized** (§1.9) |
| rope table | `[1, S, 1, 48, 2, 2]` | 38 MiB (plus a 38 MiB fp32 angle buffer) |

and the arithmetic:

```
attention   2 * 2 * S^2 * 128 * 56  = 312.9 TFLOP per block
linears     2 * S * (3*5376*7168 + 7168*5376 + 5376*28672 + 14336*5376) = 80.5 TFLOP per block
per forward (50 blocks)             ~ 19,671 TFLOP
8 evaluations                       ~ 157 PFLOP
```

**Attention is ~4x the linear cost at this length**, and it grows quadratically
while the linears grow linearly. That ratio is the single most useful number for
reasoning about H3 canvas and length choices, and it inverts below roughly
S ~ 26k.

---

## 10. The final layer and the output heads

### 10.1 Stock path

`comfy/ldm/minimax/model.py:306-330`:

```python
    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (self.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype)) + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        n = self.video_out.weight.shape[0] // self.video_out.out_features
        if n == 1:
            return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))
```

The final AdaLN has `modalities=1` and `expand=2`, so its rows are indexed by
timestep alone — hence `video_seg = (1650, 104466, 0)` and
`audio_seg = (500, 1650, 1)` for this render, the bare `t_row` values rather than
`t_row*3 + tag`.

**Only the two target segments are sliced and projected.** Every reference
implementation runs both heads over the whole sequence and selects afterwards.
Same numbers for the rows that survive; here it is 104,466 rows of two projections
avoided per step.

`mod()` casts to fp32 before the head — but the head weight is bf16 on this
artifact (§1.5), so this is an fp32 activation against a bf16 weight.

Then the output:

```python
        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
```

**The negation is half of one decision.** H3 predicts a data-ward velocity
(`x0 = x + sigma*v`); ComfyUI's `CONST.calculate_denoised` is `x - v*sigma`; so
returning `-v` reproduces the vendor rule exactly. Port one half into an engine
using the other and the ODE runs backwards.

### 10.2 Scenario 2: the repo node's head swap

`_make_head_forward` replaces each output linear's `forward`:

```python
    def forward(inp):
        block = tracker.block_v if stream == "video" else tracker.block_a
        w, b = heads.get(block, inp.device, inp.dtype)
        return F.linear(inp, w, b)
```

`inp` is fp32 (from `mod()`), so the fused head is cast to fp32 and the
projection runs in fp32 — a real difference from the stock path's bf16 weight.

The block span is derived from `t_emb` and the sampler's schedule, never from a
call counter. `_StepTracker.observe` maps each of `sample_sigmas` back through
`base_sigma` to a grid index; `_StepTracker._pick` matches the step's `t_emb`
against precomputed boundary embeddings by `cdist`. `_FusedHeads.get` then fuses
the span once and caches per `(block, device, dtype)`:

```python
def fuse_block(stack, shift, num_steps, start, stop):
    steps = pdd_time_grid(shift, num_steps).diff()
    plan = fusion_plan(steps, start, stop)[start:stop]
    return torch.tensordot(plan, stack[start:stop].to(torch.float64), dims=([0],[0])).to(torch.float32)
```

**A dt-weighted mean of the heads the step spans**, computed in float64 from the
published bf16 bank. The two streams use different shifts (12 and 3) so they
weight the same block index differently.

`master = base_w + strength * (fused - base_w)`. At `strength = 1.0` the base
head arithmetically cancels, so the bf16-ness of the checkpoint's own head does
not matter for this arm. At any other strength it does.

*measured*, the trajectory this produces:

```
 i   sigma_v       t_v   sigma_a       t_a   idx_v   idx_a   carry    block
 0  1.000000  0.000000  1.000000  0.000000     0.0     0.0  1.0000  [0,4)
 1  0.988235  0.011765  0.954546  0.045454    12.0    46.5  0.9659  [4,8)
 2  0.972973  0.027027  0.900000  0.100000    27.7   102.4  0.9250  [8,12)
 3  0.952381  0.047619  0.833333  0.166667    48.8   170.7  0.8750  [12,16)
 4  0.923077  0.076923  0.750000  0.250000    78.8   256.0  0.8125  [16,20)
 5  0.878049  0.121951  0.642857  0.357143   124.9   365.7  0.7321  [20,24)
 6  0.800000  0.200000  0.500000  0.500000   204.8   512.0  0.6250  [24,28)
 7  0.631579  0.368421  0.300000  0.700000   377.3   716.8  0.4750  [28,32)
```

`idx_v` and `idx_a` are the fractional `adaln_t_table` rows the two streams read.
Two things stand out: **the audio stream runs far ahead of the video in `t` at
every step** (0.70 against 0.37 at the last evaluation), and **the last
evaluation happens at `sigma_v = 0.63`** — the video is 37% denoised when the
model is called for the last time, and the final Euler step carries the rest.

### 10.3 Core has its own PDD path, and it assumes the opposite encoding

`2504e68d` (2026-08-29) added `_pdd_head` to core. It fires only when
`video_out.weight.shape[0] // out_features > 1`, i.e. when a LoRA has *enlarged*
the head into a `[32*96, 5376]` bank. Our converted file leaves that weight its
original size, so core takes its `n == 1` path and the repo node's two
output-linear patches own the swap. Correct, and also two implementations of one
mechanism in one process.

Core's fusion formula (`comfy/ldm/minimax/model.py:332-342`):

```python
        rows = weight.reshape(n, -1, weight.shape[1])
        first = max(start, 1)
        return nn.functional.linear(h, rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]), ...)
```

`rows[0] + sum(w_k * rows[k])` is correct **only if rows 1.. are deltas from row
0** — the comment says so: "row block 0 is a full head, later blocks are offsets
from it". The published alibaba-pai bank is **verbatim**, measured in §4.2. Fed a
verbatim bank, that formula is wrong by roughly 100%. *measured*, against the
correct `sum(w_k * rows[k])` on this box's bank:

```
block [0,4):  rel diff core-vs-verbatim = 0.7728
block [4,8):  rel diff                  = 1.0006
block [28,32): rel diff                 = 1.0016
```

**This is not a bug in core against its own artifact.** The converter it is
written for is Kijai's, and this repo's own converter docstring records that
`Comfy-Org/ComfyUI#15908` changed its head formula after `bd016b75ff9b` and the
HF repo's `lastModified` sits two minutes after that commit — i.e. the artifact
was re-uploaded as deltas to match. The finding is narrower and still worth
having: **nothing in core checks which encoding the resident bank uses, and the
two encodings have identical shapes, dtypes and key names.** A verbatim-encoded
enlarged head loads clean and renders wrong. Same discriminator the repo's
converter already uses would settle it in one line — `||row_i||/||row_0||` is
~1.0 for verbatim and ~0.02 for deltas, a factor of 35.

A second, smaller edge on the same path: `comfy/model_detection.py:396` computes
`latents_dim = final_layer.video_out.weight.shape[0] // 4`. Core's enlarging
patch is applied by `ModelPatcher.load`, *after* detection, so this is safe
today. A checkpoint shipping the bank baked in would be detected as
`latents_dim = 768`.

### 10.4 Core's PDD against ours, surface by surface

Core's target artifact is Kijai's conversion, and reading it settles what core
assumes. *measured*, `MiniMax-H3-FL2VA-Acc-8Step_pruned_comfy.safetensors`
(1.61 GiB, 578 tensors, all under `diffusion_model.`):

```
final_layer.video_out.lora_down.weight   F32  [3072, 5376]     3072 = 32 x 96
final_layer.video_out.lora_up.weight    BF16  [3072, 3072]
final_layer.video_out.reshape_weight    I64   [2]
blocks.N.adaln_proj.linear.lora_A       BF16  [64, 8]          the curve basis
blocks.N.adaln_proj.linear.diff_b       F32   [96768]
```

and its metadata says the method out loud: `"PDD head banks as pad-and-add
reshape_weight LoRA (strength 1.0 only); adaln projected to curve basis"`, with
`"alpha/rank folded into lora_B"`.

So the head bank arrives as a **rank-3072 LoRA plus a `reshape_weight`** that
pads `[96, 5376]` to `[3072, 5376]` with zeros and adds. Row block 0 lands on
the padded base, blocks 1.. land on zeros. For core's
`rows[0] + sum(w * rows[k])` to be right, block 0 must therefore hold
`head_0 - base` and blocks k>=1 must hold `head_k - head_0`. **Core and Kijai's
artifact are self-consistent**, which is what §10.3's finding turns on: the
assumption is correct for the artifact it was written against, and unchecked for
any other.

| | core (`2504e68d` + Kijai's file) | this pack (`pdd_lora.py`) |
|---|---|---|
| node needed | none, stock `LoraLoaderModelOnly` | a custom node |
| head delivery | enlarges `video_out.weight` to `[3072, 5376]` | forward patch, weight untouched |
| fusion timing | per forward, inside `FinalLayer` | per block, cached on first use |
| bank encoding | deltas from head 0, **assumed** | verbatim, **asserted at conversion** |
| strength | **1.0 only** (its own metadata says so; pad-and-add scales the base too) | any, via `base + s*(fused - base)` |
| adaln on a pruned base | rank-64 LoRA in the 8-dim basis + `diff_b` | dense `diff [96768, 8]` + `diff_b` |
| file size | 1.61 GiB | **1.044 GiB** |
| alpha | folded into `lora_B`, no `.alpha` tensors | explicit `.alpha`, tripled with the fused rank |
| partition guard | none | `base_video_out` relative distance |
| shift guard | none | raises on a mismatch, checked every forward |
| off-grid step count | the user's problem | not expressible: the node emits `SIGMAS` |
| resident-weight side effects | enlarged tensor survives on the cached model | none |

**What core does that we could adopt.** One thing, and it is now free. Since
`2504e68d` widened `FinalLayer.forward` to
`(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)`, our
`_make_final_layer_forward` already *receives* `sample_sigmas` and the resolved
shifts in its `*args`. That makes `_make_capture_forward` removable — with it
the patch on `diffusion_model.forward`, and the `default_shift_v`/`_a` fallback
that exists only because `transformer_options` may carry no shift key at all.
Core resolves the class default before passing `shifts`, so the guard would get
a strictly better input than it reads today. Two conditions: the final-layer
wrapper would have to be installed **unconditionally** rather than under the
head gate, or it re-opens the 2026-08-28 hole where `patch_heads=False` skipped
the shift check; and it would pin the node to core >= `2504e68d`, where today's
`*args` widening is exactly what lets it run on both sides of that commit.

**Where core is wrong**, beyond §10.3's unchecked bank encoding: no partition
guard, on a family whose two members ship identical key sets; no check that the
graph's shift is the one the heads were distilled at, even though `_pdd_head`
computes its `dt` weights from that shift; and the enlarged `video_out.weight`
outliving the graph that installed it, which is the concrete failure this pack's
shape check was written against.

**Where we are wrong.** Two implementations of one mechanism now live in one
process. Neither is reachable through the other — our converted file leaves the
weight its original size, so core takes `n == 1` — but a user with this pack and
a Kijai file has two things that own H3's output heads, and only one of them
refuses to stack.


---

## 11. Back through the sampler

`_forward` returns a two-element list, `forward()` applies the audio velocity
correction, `_apply_model` re-packs with `pack_latents` and calls
`calculate_denoised`, `to_d` divides that back out, and `sample_euler` takes the
step. Then, once, at the end:

```python
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
```

`latent / (1 - sigma)`, an exact no-op at the terminal 0.

`process_latent_out` (`comfy/model_base.py:2161`) divides the audio slice by
`audio_scale = 4.0`, undoing what `process_latent_in` did. `MiniMaxH3Video`
carries `scale_factor = 1.0` and no channel mean/std, so `LatentFormat`'s own
`process_in`/`process_out` are identity — **all latent normalization for H3 lives
inside the two VAEs**.

`SamplerCustomAdvanced` then unpacks to the nested view before applying
`process_latent_out` (the ordering `49a74228` fixed; before it, the audio-scale
inverse ran on the flat buffer).

---

## 12. Decode

### 12.1 Video

`VAEDecode` takes `latent.unbind()[0]`; `VAEDecodeAudio` takes `[-1]`. Both rely
on that ordering, which `MiniMaxH3AV.fix_empty_latent` and
`EmptyMiniMaxH3LatentAV` both establish.

The video VAE is a **3D causal CNN encoder plus a 36-layer ViT3D decoder** — not
a symmetric autoencoder. `vae_ratio = 16` spatial, `vae_ratio_t = 4` temporal,
24 latent channels; the decoder is `Linear(24 -> 2048)`, 36 blocks at 32 heads x
64, then `Linear(2048 -> 3072)` = one 4x16x16 pixel patch per latent token.

`decode` (`comfy/ldm/minimax/vae.py:698-710`):

```python
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * latents_std + latents_mean
```

**The hardcoded `LATENTS_MEAN`/`LATENTS_STD` lists in that file are a fallback,
not what runs.** They are *persistent* buffers and the shipped checkpoint carries
them; `self.first_stage_model.to(self.vae_dtype)` runs at `comfy/sd.py:1076`,
*before* `load_state_dict` at `:1085`, so they land as fp16 either way. The
effective first std is `1.22265625`, not the source's `1.2223774194717407`.
Denormalization therefore runs at fp16.

The decoder's native output space is ImageNet-normalized RGB;
`_finalize_pixels` is the only place fp32 enters:

```python
    def _finalize_pixels(self, part):
        # raw decoder output -> float32 pixels in [0, 1] (the VAE wrapper's process_output is identity)
        part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)
        return part.add_(self.pixel_mean.to(...)).clamp_(0.0, 1.0)
```

`comfy/sd.py:1011-1012` sets `self.process_output = lambda image: image` to match
— `2a68ce33` (2026-08-09) stopped the old `[0,1] -> [-1,1] -> [0,1]` round trip.

**Temporal chunking is what decides frame counts.** Derived at construction and
*measured* on a meta instance: `clip_length=17`, `token_drop=3`,
`frame_pre_padding=3`, `tokens_chunk_size=5`, `token_overlap=2`,
`frame_overlap=5`. Each iteration decodes **7 latent tokens (5 + 2 overlap) into
28 frames**, splits at 20, drops 3 off the front of each half, writes the first
17 and holds the second 5 as a linear cross-fade into the next chunk. There is no
cached conv state — chunk independence comes entirely from that 5-frame blend,
from front-only zero temporal padding, and from `TemporalIsolatedGroupNorm`
computing GroupNorm statistics **per frame**.

The first frame is not special-cased: `frame_pre_padding = 3` is dropped from the
very first chunk too, which is where the causal zero-padding artifacts live.

**Spatial tiling is always on and is not configurable.** `tiling=True` is the
constructor default and `sd.py` never overrides it. *measured*:

```
split_tiles(1344) -> starts [0,176,352,528,704,896,1088], len 256, overlaps [80,80,80,80,64,64]
split_tiles(768)  -> starts [0,160,336,512],              len 256, overlaps [96,80,80]
```

So a 1344x768 decode is **7 x 4 = 28 ViT forwards per temporal chunk**, each on a
16x16x7 latent tile. That is the dominant decode cost and it is invisible from
the graph. `encode_tiled`/`decode_tiled` discard all kwargs and
`handles_tiling = True`, so **`VAEDecodeTiled`'s four inputs are entirely inert
on this VAE** with no user-visible indication.

The output buffer is preallocated on the intermediate device (CPU) at
`intermediate_dtype()` = fp32 — **3.98 GiB** for this render
(`345 x 768 x 1344 x 3 x 4`) — and fp16 GPU chunks
stream into it. The full video never sits on the GPU.

Memory estimate: `chunk_frames = (5+2)*4 = 28`, the decode clamp is
`min(frames, 30)`, so the estimate is **flat in video length** past 30 frames —
correct given the chunking, but it means a length regression in the chunker
would not show up as a memory failure. For this render the *GPU* reservation is
~1.16 GB while the CPU output buffer above is 3.98 GiB, which is the whole point
of the streaming: the reservation is per chunk, the buffer is the whole video.

*One inconsistency worth knowing.* `decode_output_shape` and `sd.py`'s
`upscale_ratio[0]` agree exactly on the `5k+2` grid and diverge off it (at
`T_lat = 3` the model produces 9 frames and the formula claims 5). Only
`memory_used_decode` consumes the latter, so this is a reservation bug, not an
output bug — and `EmptyMiniMaxH3LatentAV` cannot produce an off-grid latent.
`latent_t = 102 = 5*20 + 2` is on the grid.

### 12.2 Audio

`[B, 32, 2, T]` at 40 latent frames per second, 800 samples each, 32 kHz.
**Stereo is a batch axis, not a channel axis** — the mono encoder and decoder are
applied twice with no cross-channel coupling. No mid/side, no decorrelation.

**There is no ISTFT and no mel spectrogram.** The decoder is a time-domain
BigVGAN operating directly on a learned 32 -> 2048 projection; `num_mels=2048`
is an inherited parameter name. `prod(upsample_rates) = 5*5*2*2*2*2*2 = 800`,
exactly the encoder hop. The final `clamp_(-1.0, 1.0)` is a hard clip standing
in for BigVGAN's usual `tanh`, so overdriven output distorts abruptly.

Every activation is `Activation1d` — upsample x2, snake, decimate x2, with
Kaiser-windowed sinc filters. 21 AMPBlocks x 6 plus `activation_post`. That is
where the decoder's time goes.

`latents_mean`/`latents_std` here have **no hardcoded fallback**
(`register_buffer(..., torch.empty(32))`); they come entirely from the
checkpoint, and `strict=False` means an absent pair would run on uninitialized
memory behind a `logging.warning`.

Output length is exactly `T * 800` = `575 * 800` = **460,000 samples** =
14.375 s at 32 kHz — matching the video's 345 frames at 24 fps to the sample.

`vae_decode_audio`'s `std` normalization is a **no-op for this VAE**: BigVGAN
already clamps to `[-1,1]`, so `std * 5 > 1.0` requires an RMS above 0.2, which
a clipped vocoder essentially never reaches. It looks like a loudness normalizer
and is not one.

---

## 13. Scenario 2, as a delta

Everything in §5 through §9 is unchanged. Five things move.

**13.1 308 weight keys are patched at strength 1.0.** 208 backbone modules plus
50 baked adaln `diff`/`diff_b` pairs. On this box's dynamic path each is
dequantized to bf16, patched in bf16, requantized with a **recalculated** scale,
and written back into the staged buffer — so the int8 GEMM survives, but the
quantization grid is refit around the patched weights.

**13.2 The sigmas come from the node.** Bit-identical to
`BasicScheduler(simple, 8)` at shift 12 (*measured*), so this changes nothing
numerically; what it changes is that off-grid stops being expressible.

**13.3 Every patched `mlp.fc2` loses the SwiGLU fusion — or does not, depending
on the loader path.**

*First, the two loader paths, because the rest of this only makes sense against
them.* ComfyUI has two ways to run a model bigger than VRAM:

- **Classic partial load** (`ModelPatcher.load`). Decide **once, at load time**
  which modules stay resident, largest-first until a byte budget runs out. The
  rest live on the offload device and are copied per forward. A LoRA on an
  offloaded module is installed as `m.weight_function = [LowVramPatch(...)]` and
  re-applied on **every** forward.
- **Dynamic vbar** (`CoreModelPatcher`, `comfy_aimdo.model_vbar`). The whole
  model is **staged** into a pinned host buffer and faulted into VRAM per module
  on demand, with the next block prefetched on a side stream (§9.1). A LoRA is
  installed as `m.weight_lowvram_function` — a **different attribute**.

That attribute name is the whole difference. `resolve_cast_module_with_vbar`
gates its requantization on `len(fns) == 0`, where `fns` is `weight_function`:
on the dynamic path that list stays empty, so the requant fires, the weight
comes back as a `QuantizedTensor`, and `orig.copy_(y)` writes it into the staged
buffer so the LoRA is applied once per resident signature rather than once per
forward. On the classic path `weight_function` is non-empty, `cast_bias_weight`
dequantizes, and the weight comes back as a plain float tensor.

*log*, from a real render on this box:

```
Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 308 patches attached.
```

"Staged", not "loaded partially" — **this box is on the dynamic path**, and
therefore keeps both the int8 GEMM and the fusion below under a PDD LoRA. `linear_input_act` bails to eager SwiGLU the moment
`cast_bias_weight` hands back a non-`QuantizedTensor`, which is exactly what a
`weight_function` forces. On the **classic** path a `LowVramPatch` lands in
`weight_function` and the 2.79 GiB intermediate is materialized per block per
step. On the **dynamic vbar** path it lands in `weight_lowvram_function` instead,
`fns` stays empty, the requant fires, and the fusion is kept. This box is on the
dynamic path (*log*), so it keeps it — but the two paths differ by 2.79 GiB x 50
blocks of HBM traffic per step, and which one you get is decided by free VRAM.

**13.4 Both output heads run in fp32 against a fused head.** The stock path's
heads are bf16 on this artifact (§1.5); the node's `_FusedHeads` holds fp32
masters and casts to `inp.dtype`, which `mod()` has already made fp32. At
strength 1.0 the base head cancels, so this arm is strictly the fused head at
fp32.

**13.5 The audio carry stops being exact.** §8.2's transform freezes its
coefficients at the step's own sigma; a fused head returns the block's mean
velocity over `[start, stop)`. The identity still holds pointwise, the input to
it no longer does.

---

## 14. Sharp edges, ranked by how quietly they would fail

Everything here was found by tracing, and none of it is a claim that this box's
renders are wrong today.

**1. The int8 text encoder never runs int8 arithmetic** (§2.4). Two gates,
`full_precision_mm=True` at `comfy/sd1_clip.py:114` and
`set_model_compute_dtype(torch.float32)` at `comfy/sd.py:270`, force a per-forward
dequantization to fp32 and an fp32 GEMM. `int8_linear` is called zero times. The
storage saving is real; the arithmetic saving is zero. **This is the largest
actionable finding in this document** and it is upstream's to fix, not ours —
both edits are global to every text encoder.

**2. The DiT's fp32 island is bf16 on the int8 artifact** (§1.5).
`MixedPrecisionOps.Linear.__init__` discards the declared `dtype`, so the two
output heads, both patch projections and every AdaLN projection load at the
compute dtype. Verified by executing the load. The comment in
`comfy/ldm/minimax/model.py:302` is true of the file and false of this
configuration; `adaln_proj` in particular goes F16-on-disk to bf16-in-memory,
which is a *loss* of mantissa.

**3. Core's PDD head fusion assumes a delta-encoded bank, and nothing checks**
(§10.3). ~100% wrong velocity on a verbatim bank, identical shapes and key names
either way. Not reachable through this repo's node, which leaves the weight its
original size.

**4. qkv layout is assumed, not sniffed** (§9.4). ComfyUI splits contiguously
unconditionally, where the release's own native shards are per-head interleaved.
A `DeepBeepMeep`/WanGP bf16 file in release order is `[21504, 5376]` like
Comfy's, loads clean, and renders noise. `bench/check_model_files.py` guards
model *names*.

**5. The `minimax_payload` dtype-cast exemption is a contract nothing asserts**
(§5.4). Anything H3-specific added outside that dict gets flattened to bf16 by
`_apply_model`'s sweep, including int64 tags and fp32 cond latents.

**6. fl2va and ref2va are byte-identically shaped** (§1.2). No detection, no
guard, no warning. Supplying refs to an fl2va checkpoint produces a working
render, not an error. The repo's PDD node's `base_video_out` distance check is
the only thing on this box that would notice, and only for a PDD arm.

**7. A `pad_to_patch_size` on guide latents was written and lost in a squash.**
`48b10a3e` added `z = comfy.ldm.common_dit.pad_to_patch_size(z, self.patch_size)`
to `_cond_video_rows`; the squashed `e01fb4c5` does not carry it, and only the
*target* stream is padded at `comfy/ldm/minimax/model.py:582`. Not reachable in
t2va; reachable in principle through `MiniMaxH3AddGuide`.

**8. `estimate_memory` budgets for a CFG pair this graph never runs and ignores
conditioning length entirely for H3** (§1.10). 44.12 GiB requested against a
22.06 GiB minimum for this render. Harmless on the dynamic path; on the classic
path it would drive residency to near zero.

**9. The audio VAE's `extra_1d_channel` is unset where both peer stereo VAEs set
it.** In `VAE.decode`'s OOM fallback and in `decode_tiled`, that routes a
`[1,32,2,575]` latent to the **2D image tiler**, which would try to allocate
`[1, 3, 1600, 460000]` fp32. Not reachable today — no shipped graph wires
`VAEDecodeAudioTiled` and the non-tiled reservation is ~200 MB — but it is the
failure mode you meet precisely when things are already going wrong.

**10. Audio noise is not independent of canvas** (§7.2). One generator, drawn
sequentially: change the video shape at a fixed seed and the audio noise changes.

**11. NaN suppression in the video decoder's attention.**
`optimized_attention(...).nan_to_num_(0.0)` at `comfy/ldm/minimax/vae.py:240-241`
turns an fp16 overflow into zeros. If a decoded tile is ever flat or dark, this
is where the evidence was destroyed.

**12. `set_parameters(shift=...)` defaults `audio_shift=None`**, which collapses
`audio_scale` to 1.0. Any shift-override node that forgets the second argument
silently un-scales the audio stream. `MiniMaxH3SigmaShift` passes both.

---

## 15. Numbers, in one place

*measured* unless noted. All for 1344x768, 345 frames, this prompt.

| | |
|---|---|
| frames / duration | 345 / 14.375 s at 24 fps |
| video latent | `[1, 24, 102, 48, 84]`, 9,870,336 elements |
| audio latent | `[1, 32, 2, 575]`, 36,800 elements |
| packed sampler vector | `[1, 1, 9,907,136]` |
| text rows | 500 |
| audio rows | 1,150 |
| video rows | 102,816 |
| **packed sequence S** | **104,466** |
| segments | `[(0,500,text), (500,1650,audio), (1650,104466,video)]` |
| distinct timesteps per step | 2 (`t_v`, `t_a`) |
| AdaLN rows built / read | 6 / 3 |
| residual stream (bf16) | 1.05 GiB |
| fused qkv buffer (bf16) | 4.18 GiB |
| `fc1` output (bf16) | 5.58 GiB |
| SwiGLU intermediate avoided | 2.79 GiB per block per step |
| rope rotation table (bf16) | 38 MiB |
| attention FLOP / block | 312.9 T |
| linear FLOP / block | 80.5 T |
| forward (50 blocks) | ~19,671 TFLOP |
| 8 evaluations | ~157 PFLOP |
| DiT staged to host (*log*) | 19,995 MB |
| PDD patch keys (*log*) | 308 |
| decoded pixels buffer (CPU fp32) | 3.98 GiB |
| video decode GPU reservation | ~1.16 GB |
| audio decode GPU reservation | ~499 MB |
| video decode ViT forwards | 28 per temporal chunk |
| audio samples out | 460,000 at 32 kHz |

### Dtype at every stage

| stage | weights | activations |
|---|---|---|
| tokenizer | — | int64 |
| Qwen3-VL embed | BF16 | **fp32** (hardcoded upcast) |
| Qwen3-VL 50 layers | int8 stored, **dequantized to fp32 per forward** | fp32 |
| conditioning out | — | fp32 on CPU |
| `condition_proj` + refiner | BF16 (unquantized) | bf16 |
| patchify | — | fp32 in, bf16 out |
| `video/audio_patch_proj` | **bf16** (declared fp32) | fp32 in, bf16 out |
| `adaln_t_table` | fp32 buffer | fp32 |
| `adaln_proj.linear` | **bf16** (declared fp32, stored F16) | bf16 |
| block norms | BF16 | bf16 |
| block linears | **int8 + fp32 row scale**, convrot | bf16 in, int32 accum, fp32 scale, bf16 out |
| rope table | fp32 buffer | fp64 positions -> fp32 -> bf16 table |
| attention | — | bf16, SDPA default scale |
| `final_layer.norm` | BF16 | bf16 |
| `final_layer.{video,audio}_out` | **bf16** (declared fp32) | **fp32** activation |
| [PDD] fused heads | fp32 masters from BF16 bank, fused in fp64 | fp32 |
| sampler | — | fp32 |
| video VAE | fp16 | fp16, fp32 at finalize |
| audio VAE | fp32 | fp32 |
