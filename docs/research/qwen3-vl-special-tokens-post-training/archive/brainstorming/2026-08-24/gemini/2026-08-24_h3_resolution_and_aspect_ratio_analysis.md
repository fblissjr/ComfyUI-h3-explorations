# MiniMax H3 Resolution, Aspect Ratio & Reference Sizing Architecture Analysis

**Date:** 2026-08-24
**Author:** Antigravity (Gemini)
**Location:** `docs/research/qwen3-vl-special-tokens-post-training/brainstorming/gemini/2026-08-24_h3_resolution_and_aspect_ratio_analysis.md`
**Status:** Non-authoritative multi-codebase analysis; use the canonical contract for decisions

---

## 1. Executive Summary: The Resolution Hierarchy

An investigation across all primary MiniMax H3 reference codebases (**DiffSynth-Studio**, **SGLang**, and **ComfyUI**) resolves the relationship between **`1344×768`**, **`768` short edge**, and **`2048` short edge**.

MiniMax H3 does not use a single universal resolution. Instead, it operates a **three-tier resolution architecture** partitioned by data role:

```
                            MINIMAX H3 SIZING HIERARCHY
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Output Video Canvas (Generation Target)                                                │
│   • Base Short Edge: 768 px                                                               │
│   • Default landscape: 1344 × 768 (7:4, near 16:9, ~1.03 MP)                              │
│   • Default portrait: 768 × 1344 (4:7, near 9:16, ~1.03 MP)                               │
│   • Soft Area Cap: 768 × 1344 = 1,032,192 pixels (Multiples of 32)                         │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Reference Video Clips (Temporal / Motion Conditioning)                                 │
│   • Reference Short Edge: 768 px                                                          │
│   • Soft Area Cap: 768 × 1344 (Scaled down proportionally if exceeded)                    │
│   • Temporal Sampling: 24 fps in DiT / VAE; 2 fps two-frame blocks in Qwen3-VL            │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Reference Still Images (Identity / Character / Style Conditioning)                     │
│   • Reference Short Edge: 2048 px                                                         │
│   • Soft Area Cap: NONE (No area cap applies)                                             │
│   • Intended role: high-detail still conditioning; quality mechanism is not established    │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Codebase Evidence & Source Citations

### 2.1 DiffSynth-Studio (`coderef/DiffSynth-Studio/`)

1. **Model documentation ([`MiniMax-H3.md`](../../../../../coderef/DiffSynth-Studio/docs/en/Model_Details/MiniMax-H3.md)):**
   * **Generation Canvas:**
     ```markdown
     * `height`: Height of the video, defaults to 768, must be a multiple of 32.
     * `width`: Width of the video, defaults to 1344, must be a multiple of 32.
     ```
   * **Reference Still Images (`2048` Short Edge):**
     ```markdown
     * `ref_image_short_edge`: Target short edge of a reference image, defaults to 2048.
       A reference image is rescaled onto that short edge with its aspect ratio preserved
       (upscaling allowed) and both axes rounded to the nearest multiple of 32.
       No area cap applies.
     ```
   * **Reference Videos (`768` Short Edge + Area Cap):**
     ```markdown
     * `ref_video_short_edge`: Target short edge of a reference video, defaults to 768.
     * `ref_video_max_pixels`: Soft area cap for a reference video, defaults to 768 * 1344.
       A reference video is first scaled onto the short edge, then scaled back down proportionally
       if its area exceeds the cap, and finally both axes are rounded to a multiple of 32.
     ```

2. **Pipeline implementation ([`minimax_h3_audio_video.py`](../../../../../coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py)):**
   ```python
   def _resolve_reference_image_shape(self, pipe, width: int, height: int, short_edge: int):
       scale = short_edge * 1.0 / min(width, height)
       return self._nearest_multiple(width * scale, 32), self._nearest_multiple(height * scale, 32)

   def _resolve_reference_video_shape(self, pipe, width: int, height: int, short_edge: int, max_pixels: int):
       scale = min(short_edge * 1.0 / min(width, height), float(np.sqrt(max_pixels / (width * height))))
       return self._nearest_multiple(width * scale, 32), self._nearest_multiple(height * scale, 32)
   ```

---

### 2.2 SGLang (`coderef/sglang/`)

1. **Serving implementation:** [`constants.py`](../../../../../coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py)
   records 768 as the only short edge covered by published MiniMax recipes and
   reference outputs. [`reference_encoding.py`](../../../../../coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py)
   separately implements the 2048-short-edge Ref2VA image rule with upscaling,
   nearest-32 rounding, and no canvas-area cap.

---

### 2.3 ComfyUI Reference Nodes (`comfy_extras/` & `ComfyUI-h3-explorations/`)

1. **Native constants ([`comfy_extras/nodes_minimax_h3.py`](../../../../../../../comfy_extras/nodes_minimax_h3.py)):**
   ```python
   CANVAS_MULTIPLE = 32
   BASE_SHORT_EDGE = 768
   MAX_PIXELS = 768 * 1344          # 1,032,192 pixels
   REF_IMAGE_SHORT_EDGE = 2048
   ```

2. **Canvas adaptation function ([`comfy_extras/nodes_minimax_h3.py`](../../../../../../../comfy_extras/nodes_minimax_h3.py)):**
   ```python
   def adapt_canvas(width, height):
       """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
       ratio = width / height
       if ratio >= 1.0:
           nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
       else:
           nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
       if nom_w * nom_h > MAX_PIXELS:
           s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
           nom_w, nom_h = nom_w * s, nom_h * s
       return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
               max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))
   ```

3. **Dual reference sizing modes (`match` vs. `max`) ([`comfy_extras/nodes_minimax_h3.py`](../../../../../../../comfy_extras/nodes_minimax_h3.py)):**
   * **`match` Mode:** Scales each reference image down to the generation's pixel area ($768 \times 1344$). Offers fast execution and reduced VRAM.
   * **`max` Mode:** Uses the full **`2048`px short edge** (`REF_IMAGE_SHORT_EDGE = 2048`). Preserves fine character features, eye details, and fabric textures at the cost of larger VAE latent grids.

---

## 3. Why 2048 Short Edge is Used for Still Images vs 768 for Video

```
                         VAE & DiT CONDITIONING CAPACITY
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Still Reference Image (Identity Conditioning)          │ Video Reference Clip (Motion / Temporal Conditioning)  │
├────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ • Temporal Latent Length: T = 1                        │ • Temporal Latent Length: T = 2 to 37 (124+ frames)   │
│ • VRAM Cost: Low (single 2D spatial slice)             │ • VRAM Cost: Extremely High (3D volumetric tensor)     │
│ • Sizing Policy: 2048 Short Edge, NO area cap          │ • Sizing Policy: 768 Short Edge, Capped at 768 × 1344  │
│ • Hypothesis: more spatial rows may retain detail      │ • Inference: temporal rows make the same spatial size │
│ • Mechanism and quality benefit remain unmeasured      │   substantially more expensive; quality is unmeasured │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 4. Qwen3-VL Text/Vision Encoder Sizing Rules

In addition to the VAE and DiT latents, reference media passes into the **Qwen3-VL 32B multimodal text encoder**. Qwen3-VL uses its own distinct geometry rules:

1. **Spatial Patch Factor ($f = 32\text{ px}$):**
   * `patch_size = 16`
   * `spatial_merge_size = 2`
   * All images are resized to $(H_{\text{bar}}, W_{\text{bar}})$ such that $H_{\text{bar}} \equiv 0 \pmod{32}$ and $W_{\text{bar}} \equiv 0 \pmod{32}$.

2. **Preprocessing Policies:**
   * **Native Release Policy:** Bounds of $65,536$ to $16,777,216$ pixels (min $256 \times 256$, max $4096 \times 4096$).
   * **Constrained AWQ Policy:** Bounds of $200,704$ to $301,056$ pixels. Constrains common still images to emit **264 to 289 `<|image_pad|>` visual tokens** into the language decoder.
   * **Temporal Video Blocks:** Video sampled at 2 fps, grouped into 2-frame blocks. Each block emits $(grid_h // 2) \times (grid_w // 2)$ tokens into the language stream.

---

## 5. Architectural Synthesis & Takeaways

| Domain | Native Short Edge | Area Constraint | Spatial Step | Target Consumer |
| :--- | :---: | :---: | :---: | :--- |
| **Output Video Canvas** | **`768` px** | Max $768 \times 1344$ ($1.03\text{ MP}$) | Multiple of 32 | MiniMax 33B DiT Video Denoiser |
| **Reference Video Clips** | **`768` px** | Max $768 \times 1344$ | Multiple of 32 | MiniMax 3D Video VAE & DiT |
| **Reference Still Images** | **`2048` px** | **None** | Multiple of 32 | MiniMax 3D Video VAE & DiT |
| **Qwen3-VL Still Images** | Dynamic | $200\text{k}$–$301\text{k}$ px (AWQ)<br>$65\text{k}$–$16.8\text{M}$ px (Release) | Multiple of 32 | Qwen3-VL 32B Layer 50 Conditioning |
| **Qwen3-VL Video Blocks** | Dynamic | Policy-owned: release 25.2M px; installed Comfy default 12.8M px | Multiple of 32 | Qwen3-VL 32B Layer 50 Conditioning |

### Conclusion:
* The commonly documented/default landscape and portrait canvases are **`1344×768`** (7:4) and **`768×1344`** (4:7). MiniMax's public repository establishes the 768p output regime, but does not publish the original training population.
* The **`768` short edge** is the governing baseline for generated output videos and temporal video references.
* The **`2048` short edge** is the serving implementations' target for **still reference images** (`Ref2VA`), not a ceiling; upscaling is enabled and no H3 canvas-area cap applies. Its quality rationale is not attested by the released repository.
