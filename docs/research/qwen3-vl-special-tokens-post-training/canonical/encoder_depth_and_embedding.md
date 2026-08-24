# H3 encoder depth and embedding-precision boundary

**Status:** Source-verified and measured baseline
**Observation date:** 2026-08-24
**Scope:** the official MiniMax-H3 checkout at commit
`d21241f0a4b3acbb34c97dae47fa417b7065e438`, installed ComfyUI at commit
`b78cec879b9460d5cb25228a83a942fb78d2cd24`, and the four local H3 encoder
variants named below.

## The released output contract is after 50 decoder layers

- **SOURCE:** MiniMax states that H3 uses the full pretrained Qwen3-VL-32B
  weights and provides the hidden state from its 50th layer to the
  H3-Omni-Transformer. See the official
  [`MiniMax-H3 README`](../../../../coderef/MiniMax-H3/README.md).
- **SOURCE:** In installed ComfyUI this means the unnormalized residual after
  decoder layer index 49: `Qwen3VL_32BConfig` instantiates exactly 50 layers,
  with no final language-model norm or LM head. See
  [`llama.py`](../../../../../../comfy/text_encoders/llama.py) and
  [`minimax.py`](../../../../../../comfy/text_encoders/minimax.py).
- **SOURCE:** H3 then applies a learned 5120-to-5376 `condition_proj` and a
  two-layer token refiner before the conditioning joins the H3 packed
  sequence. Those modules were trained against the released layer-50 state.
  See [`model.py`](../../../../../../comfy/ldm/minimax/model.py).
- **UNKNOWN:** MiniMax has not published why layer 50, rather than another
  depth, was selected. The public architecture description states the choice
  but gives no selection ablation or rationale. It must not be presented as a
  proven memory compromise, a visual-semantic optimum, or a language-versus-
  vision tradeoff.

DeepStack visual features are injected only after the first three language
decoder layers. Running beyond layer 50 would therefore apply more language
decoder transformations to an already fused multimodal residual; it would not
add more vision-tower or DeepStack stages. Because H3's learned projection and
token refiner expect layer-50 statistics, a deeper tap is shape-compatible but
distribution-shifted. Whether H3 benefits or degrades is **UNKNOWN** until a
paired render ablation is run.

## Verified local checkpoint inventories

The following was read from safetensors headers without materializing the
weight tensors:

| local variant | bytes | decoder layers on disk | embedding representation | LM head/final norm on disk |
|---|---:|---:|---|---|
| `qwen3vl_32b_minimax_h3_bf16.safetensors` | 66,714,914,484 | 64 (0--63) | BF16 `[151936, 5120]` | yes |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | 50 (0--49) | BF16 `[151936, 5120]` | no |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | 50 (0--49) | INT8 `[151936, 5120]` plus FP32 per-row scale | no |
| `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors` | 20,394,199,288 | 64 (0--63) | BF16 `[151936, 5120]` | yes |

Thus it is not accurate to group both Comfy-Org compressed encoders together:
the INT8 ConvRot variant preserves the embedding table in BF16, while the
NVFP4-AWQ variant quantizes it to INT8. The current custom W4A16 AWQ artifact
also preserves it in BF16.

The input embedding and LM head are separate matrices in the released model
(`tie_word_embeddings=false`), even though both have shape `[151936, 5120]`.
Both Comfy compressed variants remove the unused LM head; that is not removal
of the input embedding.

After normalizing Hugging Face and Comfy key prefixes and treating each
quantized matrix as one logical weight, the first-50-layer H3 subset contains
902 logical tensors: the input embedding, 550 decoder tensors, and 351 vision
tensors. Both the INT8 ConvRot and NVFP4 files contain exactly that logical set,
with zero missing and zero extra logical tensors. All 351 vision tensors remain
BF16 in both files. Their structural pruning is therefore limited to decoder
layers 50--63, the final language-model norm, and the LM head; their remaining
differences are quantization formats, scales, and associated metadata.

The current W4 artifact is a full 64-layer Hugging Face-compatible checkpoint
on disk. Its Comfy adapter drops layers 50--63, the final norm, and LM head
before loading, so it is already structurally pruned to the released H3
boundary in memory. Producing an additional H3-only file could reduce disk and
host-I/O footprint, but with the existing adapter it would not change the
conditioning tensor or the resident H3 weight set. That packaging experiment
is separate from changing the output depth.

The installed Comfy H3 model class independently constructs only 50 decoder
layers, with no final norm or LM head. Consequently, the full BF16 source file
is also pruned at load time even though those unused tensors remain on disk.
The two Comfy compressed files have already applied the same structural cut on
disk, while the custom W4 adapter applies it explicitly during state-dict
adaptation.

## What physical embedding pruning would mean

Neither Comfy compressed variant removes vocabulary rows or embedding
dimensions. Both retain the full `[151936, 5120]` lookup shape, including the
seven H3 token rows. NVFP4 reduces its storage precision; that is quantization,
not pruning.

Removing arbitrary rows would require a remapping tokenizer/custom embedding
lookup and would sacrifice normal Qwen multilingual vocabulary coverage.
Reducing the 5120-wide dimension would also require changing every downstream
decoder layer and H3's learned conditioning interface. Removing only unused
tail rows, if proven safe, would save very little and would break the standard
checkpoint shape. None of those operations was used by the inspected Comfy
H3 encoders.

## Ablation boundary

A useful depth experiment must hold tokenizer realization, media, visual
preprocessing, weights, unnormalized output convention, DiT, sampler, prompt,
and seeds fixed. Vary only the count of Qwen decoder layers whose output is
fed to H3. Layer 50 is the control.

Embedding precision is a second independent axis. A BF16-versus-INT8 embedding
comparison must hold decoder weights and output depth fixed. In particular,
depth, decoder quantization, and embedding precision must not be changed in one
arm and attributed to one of them afterward.

No depth or embedding-precision render result is canonical yet.
