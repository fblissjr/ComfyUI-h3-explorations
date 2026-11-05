# Runbook: Quantizing & Running `Qwen3-VL-32B-Instruct` (EXL3) for MiniMax H3 Control-Vector Labelling

This runbook documents the end-to-end workflow for quantizing `Qwen3-VL-32B-Instruct` with ExLlamaV3 on a single RTX 4090 (24 GB VRAM, 125 GiB System RAM) and running it for **H3 control-vector batch labelling** (both VLM video/contact-sheet critique and `layers.49` $\to$ `layers.50..63` activation-steering self-report).

---

## 1. Architecture & RTX 4090 VRAM Budget

`Qwen3-VL-32B-Instruct` has `64` text decoder layers (`hidden_size = 5120`, `intermediate_size = 25600`, `GQA 64Q / 8KV`, `head_dim = 128`) plus a `27`-block vision tower (`model.visual.*`).

In the MiniMax H3 pipeline:
- **Conditioning tap point (`layers.49`)**: H3 extracts `hidden_states[-15]` — the output of **`model.language_model.layers.49`** (`50`th block, 0-indexed) — before projecting `5120 -> 5376` into the DiT `token_refiner`.
- **Readout tail (`layers.50..63` + `norm` + `lm_head`)**: H3 drops `layers.50..63` during video generation, **but** when `Qwen3-VL-32B-Instruct` acts as an activation-steered self-reporter (injecting a `5120`-dim control vector at `layers.49` and reading out text tokens), `layers.50..63` and `lm_head` translate the steered `layers.49` state into natural language.
- **Residual stream coordinate basis**: EXL3's sign flips (`suh`/`svh`) and 128-wide Hadamard rotations (`had_r_128`) are applied and inverted strictly inside each `LinearEXL3` projection (`exllamav3/modules/quant/exl3.py`). Between `layers.49` and `layers.50`, the residual stream tensor is in the native unrotated `5120`-dim `fp32` basis, so H3 `vector.encoder` (`[5120]`) adds directly without any basis transform.

### Component Breakdown

| Component | Parameter Count | Placement | Storage / VRAM Scaling |
|---|---:|---|---|
| `model.language_model.embed_tokens` | `777.91M` | **System RAM (`CPU`)** (`caps["prefer_cpu"] = True`) | `1.45 GiB` System RAM / **`0.00 GiB` GPU VRAM** |
| **H3 Encoder Trunk (`layers.0..49`)** | `24,379.39M` (`50` blocks) | GPU VRAM | `2.838 GiB` per `1.00 bpw` |
| **H3 Readout Tail (`layers.50..63`)** | `6,826.23M` (`14` blocks) | GPU VRAM | `0.795 GiB` per `1.00 bpw` (`4.77 GiB` at `6 bpw`) |
| `model.language_model.lm_head` | `777.91M` | GPU VRAM | `0.54 GiB` (`-hb 6`) or `0.72 GiB` (`-hb 8`) |
| `model.visual.*` (Vision Tower) | `593.36M` | GPU VRAM (loaded when `load_vision=True`) | `0.41 GiB` (`-vb 6`, default) or `1.11 GiB` (`-vb 16`) |

---

## 2. Files in This Folder

| File | Purpose |
|---|---|
| `h3_readout_5.37bpw.yaml` | **Recipe 1 (`5.37 bpw` decoder, `head_bits: 8`)**: `6 bpw` on all 64 `self_attn` blocks, `6 bpw` on all 14 readout blocks (`layers.50..63` attn + MLP), `5 bpw` on trunk MLPs (`layers.0..49`), `8 bpw` `lm_head`. **`20.65 GiB` total GPU weights** with `-vb 6` (`~2.7 GiB` free VRAM on 24 GB 4090). |
| `h3_multimodal_4.65bpw.yaml` | **Recipe 2 (`4.65 bpw` decoder, `head_bits: 6`)**: `6 bpw` on all 14 readout blocks (`layers.50..63` attn + MLP), `5 bpw` on trunk `self_attn` (`layers.0..49`) and boundary MLPs (`layers.0..2`, `layers.48..49`), `4 bpw` on mid-trunk MLPs (`layers.3..47`), `6 bpw` `lm_head`. **`17.86 GiB` total GPU weights** with `-vb 6` (`18.04 GiB` if overridden with `-hb 8`), leaving **`~5.3–5.5 GiB` free VRAM** for multi-frame video contact sheets and larger KV caches. |
| `build_h3_cal_data.py` | Builds the hybrid `250 × 2048` (`torch.long`) calibration file (`h3_cal_data.safetensors`) combining **200 rows (80%)** standard ExLlamaV3 corpus + **25 rows (10%)** shuffled H3 scene prompts + **25 rows (10%)** round-robin interleaved pairs and chat self-report prompts across all **12/12** `.toml` control vectors. |
| `label_h3_control_vectors.py` | Batch-labels H3 control vectors via `layers.49` activation-steering self-report (`--mode self_report`), blind pair diff synthesis (`--mode blind_pairs`), and VLM image/contact-sheet critique (`--mode vlm`). |
| `build_control_vector_viewer.py` | Reads `prompt_bank/control_vectors/*.toml`, `h3_control_vector_labels.json`, and `h3_control_vector_report.md` to build `control_vector_viewer.html` and optionally serve live reloads via `--serve --port 8788`. |
| `control_vector_viewer.html` | Self-contained interactive HTML explorer for inspecting `.toml` pair spans/pooling masks (`scene_text` vs `from_first_diff`), `layers.49 -> lm_head` lexical log-odds shifts, coupled `Anchor A`/`B` rollouts across $\alpha$, and the `14 × 14` cosine similarity matrix. |

---

## 3. Step-by-Step Conversion Commands

Set your shell variables for the source checkpoint, working directory (holds resumable module checkpoints during conversion), and output directory:

```bash
export IN_DIR="<HOME>/Storage/MiniMaxAI_MiniMax-H3/text_encoder"
export WORK_DIR="<HOME>/Storage/exl3_work_qwen3vl"
```

### (Optional) Step 0: Build `h3_cal_data.safetensors`

If you add new `.toml` control vectors or `.txt` prompts later, regenerate `h3_cal_data.safetensors` from inside your `exllamav3` checkout with:

```bash
.venv/bin/python models/Qwen3-VL-32B-Instruct/build_h3_cal_data.py \
    -i "$IN_DIR" \
    -o models/Qwen3-VL-32B-Instruct/h3_cal_data.safetensors
```

### Step 1A: Convert Recipe 1 (`5.37 bpw` — Maximum Readout & Self-Report Fidelity)

Runs single-pass layer-by-layer Hessian capture + LDLQ trellis quantization (`~52 min` on RTX 4090). Note that we keep the compiled `tokenizer.json` as stock `Qwen3-VL-32B-Instruct` (`--patch-model-dir` is **not** run by default because rows `151669..151675` are untrained in Qwen's `embed_tokens` and dead identical rows in `lm_head`):

```bash
export OUT_DIR="<HOME>/Storage/Qwen3-VL-32B-Instruct-exl3-h3-5.37bpw"

.venv/bin/python convert.py \
    -i "$IN_DIR" \
    -o "$OUT_DIR" \
    -w "$WORK_DIR" \
    -rcp models/Qwen3-VL-32B-Instruct/h3_readout_5.37bpw.yaml \
    -cd models/Qwen3-VL-32B-Instruct/h3_cal_data.safetensors \
    -vb 6
```

#### Measured Post-Conversion Accuracy (`5.37 bpw` vs. `bf16` Reference States)

| Checkpoint Stage | Bitrate (`bpw`) | Rel. Frobenius Error (`rfn`) | Cosine Error (`cos`) | Cosine Similarity | SQNR (`dB`) |
|---|---:|---:|---:|---:|---:|
| **`layers.49`** (H3 conditioning tap point) | `5.20` (`6` attn / `5` MLP) | `0.003769` | `0.000028` | **`99.9972%`** | `42.94 dB` |
| **`layers.50`** (First H3 readout layer) | `6.00` (`6` attn / `6` MLP) | `0.002472` | `0.000010` | **`99.9990%`** | `47.32 dB` |
| **`layers.63`** (Final H3 readout layer) | `6.00` (`6` attn / `6` MLP) | `0.004829` | `0.000014` | **`99.9986%`** | `46.32 dB` |
| **`lm_head`** (Full `151,936`-vocab logits) | `8.00` | `0.002409` | `0.000011` | **`99.9989%`** | **`51.15 dB`** |
| **`model.visual.*`** (`27` vision blocks + `3` deepstack mergers) | `6.00` | `rmse ~0.00020` | — | — | — |

### Step 1B: Convert Recipe 2 (`4.65 bpw` — Extra VRAM Headroom for Multi-Frame VLM Grids)

*(Optional: add `-hb 8` to the `convert.py` command if you want an `8 bpw` `lm_head`; that adds `+0.18 GiB`, bringing total GPU weights from `17.86 GiB` to `18.04 GiB`.)*

```bash
export OUT_DIR="<HOME>/Storage/Qwen3-VL-32B-Instruct-exl3-h3-4.65bpw"

.venv/bin/python convert.py \
    -i "$IN_DIR" \
    -o "$OUT_DIR" \
    -w "$WORK_DIR" \
    -rcp models/Qwen3-VL-32B-Instruct/h3_multimodal_4.65bpw.yaml \
    -cd models/Qwen3-VL-32B-Instruct/h3_cal_data.safetensors \
    -vb 6
```

### Step 2: Resuming an Interrupted Conversion

`convert.py` checkpoints quantized layers into `"$WORK_DIR/qtensors/"` and calibration hidden states into `"$WORK_DIR/ckpt/"` every 120 seconds. If interrupted (`Ctrl+C` or reboot), resume cleanly with:

```bash
.venv/bin/python convert.py -w "$WORK_DIR" -r
```

### Step 3: Post-Conversion Smoke Test (Tokenizer & Module Verification)

Verify that the compiled model loads and that `Tokenizer` natively encodes `<|im_start|>` (`151644`) and `<|im_end|>` (`151645`) while encoding `<d>`, `</d>`, `<|cutoff|>` as trained Qwen BPE tokens (and optionally test `register_minimax_tokens(tok)` for `layers.0..49` `MiniMaxQwenSDTokenizer` parity):

```bash
.venv/bin/python -c '
import os
from exllamav3 import Config, Tokenizer

cfg = Config.from_directory(os.environ["OUT_DIR"])
tok = Tokenizer.from_config(cfg)
sample = "<|im_start|>user\n(<S1>)<d>Hello world.</d><|cutoff|><|im_end|>"
ids = tok.encode(sample, add_bos=False, add_eos=False, encode_special_tokens=True)
print("raw_vocab_size:", tok.raw_vocab_size, "actual_vocab_size:", tok.actual_vocab_size)
print("Token IDs:", ids.tolist())
assert 151644 in ids[0].tolist() and 151645 in ids[0].tolist()
print("Decoded:", repr(tok.decode(ids, decode_special_tokens=True)))
'
```

---

## 4. Implementation Guide for H3 Control-Vector Batch Labelling

### 4.1 Exporting `5120`-Dim `vector.encoder` Tensors
Existing `.safetensors` files in the control-vector directory that were built without `--include-encoder` contain only `vector.post_refiner` (`[5376]`).
- For **VLM visual labelling** (feeding rendered gate-ladder contact sheets or `.1fps.` video strips into `Qwen3-VL-32B-Instruct`), only the rendered media is needed.
- For **activation-steering self-report** (injecting a control vector at `layers.49` inside `Qwen3-VL-32B-Instruct`), either run `bench/control_vectors.py` (`make --include-encoder --overwrite`) or let `label_h3_control_vectors.py` compute `vector.encoder` (`[5120]`, unit-normalized, along with `median_row_norm` with `row0_excluded: True`) directly from the `.toml` contrastive pairs using `layers.0..49` of the loaded ExLlamaV3 model.

### 4.2 Resolving `layers.49` (`model.modules[53]`)
In `exllamav3/architecture/qwen3.py` (`L136-145`), `Qwen3VLModel` inserts 3 `DeepstackEmbed` modules immediately after `layers.0`, `layers.1`, and `layers.2`. With `embed_tokens` at `model.modules[0]`, `layers.49` is at **`model.modules[53]`** (not `49` or `50`). Always look it up by key:

```python
layer_49 = model.find_module("model.language_model.layers.49")
assert layer_49 is not None and layer_49.layer_idx == 49
```

### 4.3 Hooking `layers.49` with `resid_defer` / `resid_pending` Handling
In ExLlamaV3:
- `prefill_ls` in `exllamav3/model/model_ls.py` (`L470-486`, used for prompt tokens `0..L-2`) leaves `params["resid_defer"]` unset, so `TransformerBlock.forward` returns the materialized `x + y` (`fp32`).
- `forward_ls` in `exllamav3/model/model_ls.py` (`L505-512`, used for the last prompt token `L-1` and each decode step) sets `params["resid_defer"] = True`, deferring the MLP output into `params["resid_pending"]` to fuse into `layers.50`'s entry `RMSNorm`.

To steer `layers.49` cleanly across both prefill and decode, force `layers.49` to materialize its residual addition before adding the scaled control vector:

```python
import types
import torch

def attach_layer49_steering(model, v_encoder: torch.Tensor, scale: float):
    """
    v_encoder: [5120] unit vector (torch.float32)
    scale: alpha * median_row_norm (e.g. alpha in {0.10, 0.20, 0.30} * ~191.0)
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    orig_forward = layer_49.forward
    delta = (v_encoder.to(dtype=torch.float32, device=layer_49.device) * float(scale)).view(1, 1, -1)

    def steered_forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        x = orig_forward(x, params, out_dtype)
        pending = params.pop("resid_pending", None)
        if pending is not None:
            x = x.float() + pending.float()
        # Steer tokens (exclude position 0 during prompt prefill if matching row0_excluded=True)
        if x.shape[1] > 1:
            x = x.clone()
            x[:, 1:, :] = x[:, 1:, :] + delta.to(x.device, dtype=x.dtype)
        else:
            x = x + delta.to(x.device, dtype=x.dtype)
        return x

    layer_49.forward = types.MethodType(steered_forward, layer_49)
    return orig_forward
```

### 4.4 Avoiding `Generator` KV Page-Cache Collisions Across Steering Sweeps
`Job.prepare_for_queue` and `Job.prefill` in `exllamav3/generator/job.py` (`L1325-1340`, `L1466-1501`) hash KV cache pages by **input token IDs alone**. If you sweep multiple control vectors or $\alpha$ values on the same self-report prompt via `Generator`:
- Either run a direct `model.prefill()` + `model.forward()` loop with a single-sequence `Cache(model, max_num_tokens=...)` reset (`cache.reset()`) between vectors, **or**
- Include a unique vector tag/index inside the user prompt prefix so `PageTable` never reuses `layers.50..63` KV pages generated under a previous vector.

### 4.5 KV Cache Quantization for Heavy Multimodal Batching
When loading `vision_model = config.load_vision_model()` alongside the text model (`model.load()`), use an 8-bit KV cache (`CacheLayer_quant(k_bits=8, v_bits=8)`) to cut KV cache VRAM in half on the RTX 4090:

```python
from exllamav3 import Cache, CacheLayer_quant

cache = Cache(
    model,
    max_num_tokens=8192,
    layer_type=CacheLayer_quant,
    k_bits=8,
    v_bits=8,
)
```
