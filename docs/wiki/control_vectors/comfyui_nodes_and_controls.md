# ComfyUI Nodes, Field Masks, Step Windows, and Reference Controls

Last updated: 2026-11-11.

[`control_vector_nodes.py`](../../../control_vector_nodes.py) registers four ComfyUI V3 nodes under `model/conditioning/minimax` and automatically registers `<ComfyUI>/models/control_vectors/` in `folder_paths`.

![Runtime steering, field masks, and step windows](assets/runtime_steering_and_masks.svg)

---

## 1. The four nodes at a glance

| Node | Inputs -> Outputs | Role |
|---|---|---|
| **`MiniMaxH3ApplyControlVectors`** | `MODEL -> MODEL` | Primary runtime node. Attaches a zero-extra-pass `DIFFUSION_MODEL` wrapper that steers selected `context` rows across `[start_percent, end_percent)`, supports inline `< 2 ms` J-Lens `words` / `hold_words`, stream scaling (`video_scale` / `audio_scale`), and reference VAE release (`ref_vae_end_percent`, `ref_vae_aug`). Chain multiple nodes in series to stack vectors and `hold` projections in `float64`. |
| **`MiniMaxH3MakeControlVector`** | `MODEL, CLIP[, H3EncodedReferences] -> vector_name, report` | In-graph builder. Runs the exact same [`encode_pair_set`](../../../control_vector_nodes.py) + [`pool_pair_set`](../../../control_vectors.py) pipeline as CLI `build` and `make` (verified bitwise `torch.equal = True` on live `:8188` across `vector.post_refiner`, `vector.encoder`, and `holds.post_refiner`). Accepts either a `pair_file` `.toml` or inline `words = "pole_a -> pole_b"` + `target_prompt`. |
| **`MiniMaxH3MakeMediaControlVector`** | `MODEL, CLIP, IMAGE_A, IMAGE_B -> vector_name, report` | In-graph image-pair builder (`B - A` between two `<Picture 1>` image batches snapped to an identical 32-px M-RoPE grid under `scene_prompt`; defaults to `pooling = "scene_text"`, `short_edge = 512`). |
| **`MiniMaxH3LayerNormMatch`** | `CONDITIONING[, reference_CONDITIONING] -> CONDITIONING` | Diagnostic utility for `CLIPSetLastLayer` probes: rescales tag-`1` text rows (excluding sink row `0`) so their median L2 norm matches layer-50 `reference_conditioning` or `target_median_row_norm`. |

---

## 2. `MiniMaxH3ApplyControlVectors`: modes and strength units

All chained `MiniMaxH3ApplyControlVectors` nodes on a `MODEL` execute inside a single `float64` loop over the active row mask `M`:

| `mode` | Formula on masked rows $\mathbf{r} \in \mathbb{R}^{5376}$ | Typical `strength` | When to use |
|---|---|---|---|
| `add_norm_matched` *(default)* | $\mathbf{r} \leftarrow \mathbf{r} + s \cdot 67.5 \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$ | `0.75 .. 2.25` (`1.0` = `67.5` L2 shift, ~25% of median token norm `~268`) | **Default for almost every render.** Makes `strength = 1.0` comparable across `diff_span`, `scene_text`, `make-media`, and J-Lens `words` regardless of raw pair norm. |
| `add` | $\mathbf{r} \leftarrow \mathbf{r} + s \cdot \mathbf{v}$ | `0.5 .. 1.5` (depends on vector's `raw_norm`) | Reproducing raw mean-pair-difference units (`1.0` = exact `B - A` shift measured during build). |
| `hold` | $\Delta\mathbf{r} \leftarrow \Delta\mathbf{r} - \alpha (\Delta\mathbf{r} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}$ (`strength` $\alpha \in [0, 2]$) | `1.0` (`0.5` for partial scrub) | Chained **after** one or more `add` / `add_norm_matched` nodes: removes nuisance vector $\hat{\mathbf{u}}$ from the **accumulated steering delta** $\Delta\mathbf{r}$ without stripping concepts already written in the base prompt (`Arm 58`: `cursed_cctv` + `hold(chaos_panic)`). |
| `project_out` | $\mathbf{r} \leftarrow \mathbf{r} - s \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}$ | `1.0` | Removes $\mathbf{v}$ directly from the **base prompt's own tokens** $\mathbf{r}$. |
| `amplify` | $\mathbf{r} \leftarrow \mathbf{r} + s \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}$ | `0.5 .. 1.5` | Boosts `v` in proportion to how strongly each token already aligns with `v`. |

- **Recipe safety (`allow_recipe_mismatch = False` default)**: `fl2va` and `ref2va` post-refiner spaces are near-orthogonal (`cos ≈ 0.01`), and LoRAs inside `TokenRefiner` rotate `vector.post_refiner`. By default, `MiniMaxH3ApplyControlVectors` auto-swaps `stem__fl2va...` <-> `stem__ref2va...` when a twin exists in `models/control_vectors/` and refuses true checkpoint/LoRA mismatches unless `allow_recipe_mismatch = True` (CPU J-Lens `j_lens.*` vectors are family-matched and exempt from LoRA hash refusal).

---

## 3. Targeting parts of an H3 prompt: `mask` and `fields`

Every structured H3 prompt is parsed by [`h3_spans.py`](../../../h3_spans.py) so structural tokens are **never** corrupted by default:
- Tag-`0` vision tokens (`<Picture N>`, `<Video N>`),
- Spoken dialogue inside `<d>...</d>` and speaker tags `(S1)`,
- Reference tags (`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`), `[Shot N]` markers, `<0.0 seconds>` timestamps, Part One keyframe alignment lines, and section headers (`summary:`, `subject_definitions:`, etc.).

You can choose which prose sections receive the vector using either the **`mask`** dropdown or the **`fields`** multi-select (`io.MultiCombo`):

### A. Lever masks (`mask` dropdown when `fields` is empty)

| `mask` | Prose sections steered | Notes & fallback behavior |
|---|---|---|
| `scene_only` *(default)* | All visual prose (`look_only \| description_only`: `subject_definitions:`, `summary:`, visual `retention_analysis:`, and `integrated_multimodal_description:` / `detailed_description:`) | Leaves audio sections (`overall_soundscape:`, `non_diegetic_music:`), dialogue, tags, and vision rows untouched. |
| `description_only` | Action & environment prose inside `detailed_description:` (`ref2va`) or `integrated_multimodal_description:` (`t2va`/`fl2va`) | **Best choice on `ref2va` when grading the room/lighting (`film_16mm_1974`, `noir`) while keeping `<Subject 1>` untouched.** |
| `look_only` | `subject_definitions:`, `summary:`, and visual lines of `retention_analysis:` | Falls back to `description_only` on 3-field `t2va`/`fl2va` prompts that lack `ref2va` headers. |
| `soundscape_only` | `overall_soundscape:`, non-`N/A` `non_diegetic_music:`, and `<Audio N>:` lines | **Mandatory for audio/vocal vectors** (`reverb_cathedral`, `pyro_klaxon_audio`, `operatic_bellow`, `whisper_asmr`); under `scene_only`, `0` audio rows would be steered! |
| `after_dialogue` | Visual prose rows occurring after the final `</d>` | Useful when spoken lines sit early and you only want late reaction beats steered. |
| `non_vision` | Every tag-`1` text row (including headers/dialogue), excluding tag-`0` vision rows | **Required for unsectioned 1-line prompts** (where structured header masks match `0` rows). |
| `all_rows` | Every row in `context`, including tag-`0` vision tokens | Diagnostic ablation only; steering vision rows degrades `<Picture 1>` likeness. |

### B. Exact single-field and multi-field targeting (`mask` field names & `fields` multi-select)

- **Single field via `mask`**: `subject_definitions`, `summary`, `retention_analysis`, `description_only`, `overall_soundscape`, `non_diegetic_music`. Unlike `look_only`, a named field **never falls back**: if the prompt lacks that header or its value is `N/A`, `0` rows are steered and ComfyUI logs a clear notice.
- **Multiple fields via `fields` (`io.MultiCombo`)**: tick any combination of `subject_definitions`, `summary`, `retention_analysis`, `description`, `overall_soundscape`, `non_diegetic_music`.
  - When **any** item in `fields` is ticked, it overrides `mask` (`fields_mask(["description", "overall_soundscape"])` -> `"description+overall_soundscape"`).
  - In raw ComfyUI `_api.json` workflows, `io.MultiCombo` serializes as `{"__value__": ["description", "overall_soundscape"]}` (and on the CLI `inspect` / `grade`, pass `--mask description_only+overall_soundscape`).

### What our 2026-11-11 field-targeting renders proved on `ref2va`
On the 6-field diner `ref2va` shot (`s730451892`), we swept `film_16mm_1974` and subject/wardrobe vectors across individual fields:
1. **Putting a film/style vector on `subject_definitions` (or `look_only`) does NOT restyle `<Subject 1>`**—because `<Subject 1>`'s appearance is anchored by `<Picture 1>`'s vision tokens and reference VAE latents, while room/lighting strokes live in `detailed_description:`.
2. **Putting `film_16mm_1974` on `description_only` grades the diner, window light, and table surfaces into 1974 16mm stock while keeping `<Subject 1>` clean.**
3. **Putting a subject-clothing vector on `subject_definitions` alone** did nothing when `subject_definitions:` named no baseline clothes (`"<Subject 1>: A woman from <Picture 1>."`), and produced a cream blouse at `strength >= 2.0` once baseline clothes (`"wearing a plain black knit sweater and blue jeans"`) were written into `subject_definitions:`—whereas [`MiniMaxH3EditGuider`](../edit_guidance.md) hit the exact rust-orange blouse and flared jeans cleanly.

---

## 4. Step windows (`start_percent`, `end_percent`) and fixing high-strength face tint

`MiniMaxH3ApplyControlVectors` checks step progress using the exact same half-open test as `MiniMaxH3Sol`:
$$\text{start\_percent} \le \frac{\text{step\_idx}}{N_{\text{steps}}} < \text{end\_percent}$$

On **6-step Pika** (`step_idx = 0, 1, 2, 3, 4, 5`; ratios `0.000, 0.167, 0.333, 0.500, 0.667, 0.833`):

| `end_percent` (with `start_percent = 0.0`) | Active steps (`step_idx`) | Count | Typical use |
|---|---|---|---|
| `0.16` | `0` | 1 of 6 | Ultra-brief initial layout nudge |
| `0.33` | `0, 1` | 2 of 6 | Early composition push only |
| **`0.50`** *(or `0.34`)* | **`0, 1, 2`** | **3 of 6** | **Sweet spot for strong vectors (`strength = 2.0..3.0`)**: establishes world/wardrobe/material structure in the first half, then lets unsteered steps `3..5` render clean skin tones and fine facial detail |
| `0.66` | `0, 1, 2, 3` | 4 of 6 | Extended push (`0.667` is excluded because `4/6 = 0.6667 > 0.66`) |
| `0.67` | `0, 1, 2, 3, 4` | 5 of 6 | Drops steering only on final step `5` |
| `1.00` *(default)* | `0, 1, 2, 3, 4, 5` | 6 of 6 | Full schedule; clean at moderate strengths (`0.75 .. 1.5`) |

### Verified 2026-11-11 finding: red facial tint at high strength
On `$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_subject_vector_one_input_lowered_at_a_time_s730451892_stacked.mp4`, pushing a vector on `<Subject 1>`'s line to `strength = 2.5` across all 6 steps (`end_percent = 1.0`) caused a reddish color burn around her eyes and cheeks. Testing one knob at a time proved:
- Lowering `ref_vae_end_percent` alone (`0.5`) or lowering `ref_vae_aug` alone (`0.5`) **kept the red tint** on her face while loosening her likeness.
- Lowering **the vector's `end_percent` to `0.5`** (stopping the vector after step `2`, while keeping `<Picture 1>` at `ref_vae_end_percent = 1.0, ref_vae_aug = 0.999`) **completely removed the red eye/cheek tint** while keeping the clothing/scene change.

---

## 5. Decoupled video/audio scaling and reference VAE release

### Stream multipliers (`video_scale`, `audio_scale`, `split_end_percent`)
- At default `(video_scale = 1.0, audio_scale = 1.0)`, the node runs **1 DiT pass per step**.
- When `(video_scale, audio_scale) != (1.0, 1.0)` inside `[0.0, split_end_percent)`, the wrapper runs an unsteered base pass alongside the steered pass and scales the video and audio velocity deltas independently (`(1.0, 0.0)` = visual-only with untouched audio, verified in `Arm 60` at `split_end_percent = 0.34`; `(0.0, 1.5)` = boosted acoustic effect with untouched video).

### Reference & keyframe VAE release (`ref_vae_end_percent`, `ref_vae_aug`)
On `ref2va`, `fl2va`, and guide-latent graphs, clean 3D Video VAE latents (`ref_img` / `cond`) are packed directly into the DiT token sequence alongside noisy video/audio tokens:
- **Keep defaults (`ref_vae_end_percent = 1.0`, `ref_vae_aug = 0.999`)** whenever `<Subject 1>` should remain a recognizable human in the same pose/geometry (film looks, lighting grades, weather, or clothing experiments).
- **Lower `ref_vae_end_percent` to `0.16` or `0.33`** only when a vector intentionally **reshapes 3D silhouette or replaces live-action skin** (`claymation_stopmotion`, `felt_muppet`, `balloon_head_inflate`, `molten_lava_golem`). After `ref_vae_end_percent`, the wrapper zeros visual conditioning rows while keeping `PackedLayout` sequence length, 3D RoPE cursor positions, tag-`0` Qwen vision tokens, and speaker audio references intact (`Arm 66` lesson: dropping `refs` from `PackedLayout` mid-trajectory shifts 3D RoPE cursors and scrambles the clip).
- **`ref_vae_aug`** (`0.0 .. 1.0`, default `0.999`): controls how clean the reference VAE rows are labeled on active steps, clamped by `max(ref_vae_aug, stream_time)` (`dd054d6b`) so reference rows are never marked noisier than the frame currently being denoised.
