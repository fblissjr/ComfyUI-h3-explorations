# Qwen3-VL 32B W4A16 AWQ Quantization Suite & Artifacts

> **Context, owner, 2026-09-20:** the artifacts this suite produced were badly
> executed, and they lost a layer-50 holdout to a community quant. That is a
> fact about these two candidates and **not** evidence that quantising our own
> encoder is unpromising -- the calibration, the group size and every other knob
> were ours, we did it poorly, and priorities then shifted. Calibration data is
> the untapped area and the part this programme did worst. The code and recipe
> below remain a working reference. `../../../wiki/decisions.md`.

This directory contains the complete source code, configuration files, workflow templates, and visualization tools for the **Qwen3-VL 32B W4A16 AWQ** model developed for the **MiniMax H3** multimodal pipeline.

---

## Directory Structure

```
awq_quantization_suite/
├── code/                   # Python quantization, extraction, conversion & validation scripts
│   ├── quantize_qwen3_vl_32b.py  # Main llm-compressor AWQ sequential pipeline
│   ├── extract_metadata.py       # 16-worker threaded ComfyUI MP4 prompt extractor
│   ├── convert_to_comfyui.py     # Single-file .safetensors builder
│   ├── validate.py               # Comprehensive 1,954-tensor QA verification suite
│   ├── test_layer50_drift.py     # Layer 50 hidden state shape & tap benchmark
│   └── upload_to_hf.py           # Automated Hugging Face repository publisher
├── (config/ removed 2026-09-20 -- stock Qwen3-VL tokenizer and processor
│    snapshots, reproducible from the Hub; the recipe is inlined below)
├── workflows/              # ComfyUI workflow JSONs and standalone custom loader
│   ├── comfyui_minimax_h3_awq_loader.py            # Standalone drop-in loader node
│   ├── comfyui_minimax_h3_awq_text_to_video.json   # Text-to-video workflow
│   ├── comfyui_minimax_h3_awq_image_reference.json # Omni-reference image workflow
│   └── comfyui_minimax_h3_awq_first_frame.json     # First/last keyframe anchor workflow
└── visualization/          # Interactive dashboard
    └── index.html                # Visual model plan, precision tables & architecture diagrams
```

---

## Key Highlights
* **Precision Breakdown:** 100% BF16 ViT (27 layers, 351 tensors), 100% BF16 DeepStack Mergers, 100% BF16 Embeddings & Norms, W4A16 AWQ Language Decoder (64 layers, 350 linears).
* **Footprint:** Compresses 66.7 GB BF16 source to 18.99 GB, enabling single-GPU inference on a 24GB RTX 4090 with ~5.5 GB headroom.
* **Hugging Face Model:** [`fbjr/qwen3-vl-32b-W4A16-AWQ-H3`](https://huggingface.co/fbjr/qwen3-vl-32b-W4A16-AWQ-H3)
* **Historical Technical Report:** [`qwen3vl_32b_w4a16_awq_quantization_report.md`](qwen3vl_32b_w4a16_awq_quantization_report.md) — read its status correction before citing it.

## The recipe, inlined

`config/recipe.yaml` was removed with the rest of the config snapshots on
2026-09-20. It is the only one worth keeping and it is short, so it lives here
as text instead of as a file. This is what produced the artifacts, not a
recommendation -- see the context note at the top of this file.

```yaml
default_stage:
  default_modifiers:
    AWQModifier:
      requires_calibration_data: true
      mappings:
      - smooth_layer: re:.*input_layernorm$
        balance_layers: ['re:.*q_proj$', 're:.*k_proj$', 're:.*v_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*v_proj$
        balance_layers: ['re:.*o_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*post_attention_layernorm$
        balance_layers: ['re:.*gate_proj$', 're:.*up_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*up_proj$
        balance_layers: ['re:.*down_proj$']
        activation_hook_target: null
      duo_scaling: false
      n_grid: 20
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*visual.*', 're:.*embed_tokens', 're:.*input_layernorm$', 're:.*post_attention_layernorm$',
        're:.*norm$']
      scheme: W4A16
      bypass_divisibility_checks: false
      requires_calibration_data: false
```

`bench/probe_awq_recipe_boundary.py` in the repo root graded this recipe before
any weights were loaded, and `bench/results/2026-08-25_awq_recipe_boundary.json`
is what it recorded: the resolved ignore list was `["lm_head",
"re:model\\.visual\\..*"]`, so the vision tower was never quantized here --
by construction, not as a measured choice.
