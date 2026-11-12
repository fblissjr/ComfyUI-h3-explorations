# ComfyUI Nodes, Field Masks, Step Windows, and Reference Controls

Last updated: 2026-11-11.

[`control_vector_nodes.py`](../../../control_vector_nodes.py) registers the composable ComfyUI V3 steering & reference-release nodes under `model/conditioning/minimax` (plus the legacy 18-input `MiniMaxH3ApplyControlVectors` under `model/conditioning/minimax/legacy` with `is_deprecated = True` for saved PNGs and API graphs) and automatically registers `<ComfyUI>/models/control_vectors/` in `folder_paths`.

![Runtime steering, field masks, and step windows](assets/runtime_steering_and_masks.svg)

---

## 1. The node suite at a glance

### Composable canvas nodes (`model/conditioning/minimax`)

| Node | Inputs -> Outputs | Role |
|---|---|---|
| **`MiniMaxH3Steer`** | `MODEL[, custom_vector, hold_vector] -> MODEL` | **Everyday runtime steering node.** Picks a deduplicated `vector` stem (auto-resolving `fl2va` / `ref2va` and LoRA siblings at run time) or accepts a `custom_vector` wire (`MINIMAX_H3_CONTROL_VECTOR`), applies `strength` in `mode` (`add_norm_matched`, `add`, or `project_out`) onto `target` or ticked `fields` across `[start_percent, end_percent)`, isolates `streams` (`both`, `video_only`, `audio_only`) across the entire active window, and scrubs side effects via `hold_words` or a wired `hold_vector`. Chain multiple `MiniMaxH3Steer` nodes to stack vectors in `float64`. |
| **`MiniMaxH3WordVector`** | `[, hold_vector] -> H3ControlVector` | **Zero-file `< 2 ms` CPU J-Lens word synthesizer.** Builds a family-agnostic word vector handle from `words`, `against_words` (including `@prompt_subject`), `hold_words`, and `hold_alpha`. Wire into `MiniMaxH3Steer.custom_vector` to steer or `MiniMaxH3Steer.hold_vector` to protect a concept. |
| **`MiniMaxH3LoadControlVector`** | `[, hold_vector] -> H3ControlVector` | **Saved vector stem loader.** Emits a family-agnostic `H3ControlVector` handle from `models/control_vectors/` (with optional `hold_words` / `hold_vector`) so a saved vector can be wired into `MiniMaxH3Steer.hold_vector` (replacing fragile 2-node `mode = "hold"` chains) or `custom_vector`. |
| **`MiniMaxH3ReleaseReferences`** | `MODEL -> MODEL` | **Standalone visual reference & keyframe release node** (`end_percent = 1.0`, `clean_level = 0.999`). Controls when clean 3D Video VAE visual conditioning segments (`ref_img` / `cond`) drop from `PackedLayout` and how cleanly they are held (`max(clean_level, stream_time)`). Independent of control vectors: works on plain prompts, `MiniMaxH3EditGuider`, and `MiniMaxH3Steer` alike. |
| **`MiniMaxH3MakeControlVector`** | `MODEL, CLIP[, H3EncodedReferences] -> vector_name, report, control_vector` | In-graph builder. Runs the exact same [`encode_pair_set`](../../../control_vector_nodes.py) + [`pool_pair_set`](../../../control_vectors.py) pipeline as CLI `build` and `make` (verified bitwise `torch.equal = True` across `vector.post_refiner`, `vector.encoder`, and `holds.post_refiner`). Accepts either a `pair_file` `.toml` or inline `words = "pole_a -> pole_b"` + `target_prompt` (`words` overrides `pair_file` when non-empty), and outputs `control_vector` (`H3ControlVector`) so a newly built vector wires directly into `MiniMaxH3Steer.custom_vector` in the same run. |
| **`MiniMaxH3MakeMediaControlVector`** | `MODEL, CLIP, IMAGE_A, IMAGE_B -> vector_name, report, control_vector` | In-graph image-pair builder (`B - A` between two `<Picture 1>` image batches snapped to an identical 32-px M-RoPE grid under `scene_prompt`; defaults to `pooling = "scene_text"`, `short_edge = 512`), also emitting a direct `control_vector` wire. |
| **`MiniMaxH3LayerNormMatch`** | `CONDITIONING[, reference_CONDITIONING] -> CONDITIONING` | Diagnostic utility for `CLIPSetLastLayer` probes: rescales tag-`1` text rows (excluding sink row `0`) so their median L2 norm matches layer-50 `reference_conditioning` or `target_median_row_norm`. |

### Legacy all-in-one node (`model/conditioning/minimax/legacy`, `is_deprecated = True`)

| Node | Inputs -> Outputs | Role |
|---|---|---|
| **`MiniMaxH3ApplyControlVectors`** | `MODEL -> MODEL` | Kept with all 18 original inputs in exact positional order so saved clip PNGs, shipped `*_control_vector_api.json` workflows, `workflows/build_workflows.py`, and the phone app run unchanged while hiding the 18-widget monolith from ComfyUI's search menu. |

---

## 2. Modes, strength units, and hold wires (`MiniMaxH3Steer` & `MiniMaxH3ApplyControlVectors`)

All `MiniMaxH3Steer` (and legacy `MiniMaxH3ApplyControlVectors`) nodes chained on a `MODEL` share the same underlying `attach_control_vector` engine and execute inside a single `float64` loop over the active row mask `M`:

| `mode` | Availability | Formula on masked rows $\mathbf{r} \in \mathbb{R}^{5376}$ | Typical `strength` | When to use |
|---|---|---|---|---|
| `add_norm_matched` *(default)* | `Steer` & legacy | $\mathbf{r} \leftarrow \mathbf{r} + s \cdot 67.5 \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$ | `0.75 .. 2.25` (`1.0` = `67.5` L2 shift, ~25% of median token norm `~268`) | **Default for almost every render.** Makes `strength = 1.0` comparable across `diff_span`, `scene_text`, `make-media`, and J-Lens `words` regardless of raw pair norm. |
| `add` | `Steer` & legacy | $\mathbf{r} \leftarrow \mathbf{r} + s \cdot \mathbf{v}$ | `0.5 .. 1.5` (depends on vector's `raw_norm`) | Reproducing raw mean-pair-difference units (`1.0` = exact `B - A` shift measured during build). |
| `project_out` | `Steer` & legacy | $\mathbf{r} \leftarrow \mathbf{r} - s \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}$ | `1.0` | Removes $\mathbf{v}$ directly from the **base prompt's own tokens** $\mathbf{r}$. |
| `hold` *(via `hold_vector` wire on `Steer`, or `mode = "hold"` on legacy)* | `Steer.hold_vector` & legacy | $\Delta\mathbf{r} \leftarrow \Delta\mathbf{r} - \alpha (\Delta\mathbf{r} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}$ ($\alpha \in [0, 2]$) | `hold_alpha = 1.0` (`0.5` for partial scrub) | Removes nuisance vector $\hat{\mathbf{u}}$ from the **accumulated additive steering delta** $\Delta\mathbf{r}$ on the same target rows without stripping concepts already written in the base prompt (`Arm 58`: `cursed_cctv` + `hold(chaos_panic)`). On `MiniMaxH3Steer`, wiring `hold_vector` automatically inherits the node's `target`/`fields` and `[start_percent, end_percent)` so a dangling or mismatched `hold` is impossible. |
| `amplify` | Legacy only | $\mathbf{r} \leftarrow \mathbf{r} + s \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}$ | `0.5 .. 1.5` | Boosts `v` in proportion to how strongly each token already aligns with `v`. |

- **Stem deduplication & recipe safety (`allow_recipe_mismatch = False` default)**: `fl2va` and `ref2va` post-refiner spaces are near-orthogonal (`cos ≈ 0.01`), and LoRAs inside `TokenRefiner` rotate `vector.post_refiner`. `MiniMaxH3Steer` and `MiniMaxH3LoadControlVector` collapse `<stem>__<family>__<pooling>[__lora_<hash>].safetensors` into clean `<stem>` entries in the dropdown and resolve the loaded checkpoint's family + LoRA sibling at run time (refusing true checkpoint/LoRA mismatches unless `allow_recipe_mismatch = True`; CPU J-Lens `j_lens.*` vectors are family-matched and exempt from LoRA hash refusal).

---

## 3. Targeting parts of an H3 prompt: `target` / `mask` and `fields`

Every structured H3 prompt is parsed by [`h3_spans.py`](../../../h3_spans.py) so structural tokens are **never** corrupted by default:
- Tag-`0` vision tokens (`<Picture N>`, `<Video N>`),
- Spoken dialogue inside `<d>...</d>` and speaker tags `(S1)`,
- Reference tags (`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`), `[Shot N]` markers, `<0.0 seconds>` timestamps, Part One keyframe alignment lines, and section headers (`summary:`, `subject_definitions:`, etc.).

You can choose which prose sections receive the vector using either the **`target`** dropdown (`mask` on legacy `MiniMaxH3ApplyControlVectors`) or the **`fields`** multi-select (`io.MultiCombo`, placed directly below `target` on `MiniMaxH3Steer`):

### A. Lever presets (`MiniMaxH3Steer.target` / `MiniMaxH3ApplyControlVectors.mask` when `fields` is empty)

| `target` (`mask`) | Prose sections steered | Notes & fallback behavior |
|---|---|---|
| `scene_only` *(default)* | All visual prose (`look_only \| description_only`: `subject_definitions:`, `summary:`, visual `retention_analysis:`, and `integrated_multimodal_description:` / `detailed_description:`) | Leaves audio sections (`overall_soundscape:`, `non_diegetic_music:`), dialogue, tags, and vision rows untouched. |
| `description_only` | Action & environment prose inside `detailed_description:` (`ref2va`) or `integrated_multimodal_description:` (`t2va`/`fl2va`) | **Best choice on `ref2va` when grading the room/lighting (`film_16mm_1974`, `noir`) while keeping `<Subject 1>` untouched.** |
| `look_only` | `subject_definitions:`, `summary:`, and visual lines of `retention_analysis:` | Falls back to `description_only` on 3-field `t2va`/`fl2va` prompts that lack `ref2va` headers. |
| `soundscape_only` | `overall_soundscape:`, non-`N/A` `non_diegetic_music:`, and `<Audio N>:` lines | **Mandatory for audio/vocal vectors** (`reverb_cathedral`, `pyro_klaxon_audio`, `operatic_bellow`, `whisper_asmr`); under `scene_only`, `0` audio rows would be steered! |
| `after_dialogue` | Visual prose rows occurring after the final `</d>` | Useful when spoken lines sit early and you only want late reaction beats steered. |
| `non_vision` | Every tag-`1` text row (including headers/dialogue), excluding tag-`0` vision rows | **Required for unsectioned 1-line prompts** (where structured header masks match `0` rows). |
| `all_rows` *(legacy `Apply` / CLI only)* | Every row in `context`, including tag-`0` vision tokens | Diagnostic ablation only; steering vision rows degrades `<Picture 1>` likeness (omitted from `MiniMaxH3Steer.target`). |

### B. Exact single-field and multi-field targeting (`fields` multi-select)

- **One or more fields via `fields` (`io.MultiCombo` on `MiniMaxH3Steer` and legacy `Apply`)**: tick any combination of `subject_definitions`, `summary`, `retention_analysis`, `description`, `overall_soundscape`, `non_diegetic_music`.
  - When **any** item in `fields` is ticked, it overrides `target` / `mask` (`fields_mask(["description", "overall_soundscape"])` -> `"description+overall_soundscape"`) and logs the active field mask. Unlike `look_only`, ticked fields **never fall back**: if the prompt lacks that header or its value is `N/A`, `0` rows are added for that field and ComfyUI logs a warning if the total is `0`.
  - In raw ComfyUI `_api.json` workflows, `io.MultiCombo` serializes as `{"__value__": ["description", "overall_soundscape"]}` (and on the CLI `inspect` / `grade`, pass `--mask description_only+overall_soundscape` or a single field name such as `--mask subject_definitions`).
  - *(Legacy note: `MiniMaxH3ApplyControlVectors.mask` still retains the five single-field strings `subject_definitions`, `summary`, `retention_analysis`, `overall_soundscape`, `non_diegetic_music` for older graphs; `MiniMaxH3Steer` removes that duplication so `target` holds only the 6 preset levers and `fields` holds individual prompt fields.)*

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
On `$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_subject_vector_one_input_lowered_at_a_time_s730451892_stacked.mp4`, pushing a vector on `<Subject 1>`'s line to `strength = 2.5` across all 6 steps (`end_percent = 1.0`) caused a reddish color burn around her eyes and cheeks. One seed, one picture, one input lowered at a time (the arms are `render_diner16_subject_release.py` in `$H3_OUTPUT_DIR/Video/edit_guidance/diner16f_how_it_was_made/`):
- Lowering `ref_vae_end_percent` alone (to `0.34`, a three-step hold) or lowering `ref_vae_aug` alone (to `0.7`) **kept the red tint** on her face. With the weight lowered the room came out as a different draw.
- Lowering **the vector's `end_percent` to `0.5`** (stopping the vector after step `2`, while keeping `<Picture 1>` at `ref_vae_end_percent = 1.0, ref_vae_aug = 0.999`) **removed the red eye/cheek tint** while keeping the clothing change. The owner read the same on playback.
- Releasing the picture was neither what allowed the clothing change nor needed for it: the clothing changed with the picture held, and where it did not change (a prompt that names no clothing) no release made it change.

---

## 5. Stream isolation (`streams`) and `MiniMaxH3ReleaseReferences`

### Stream selection (`MiniMaxH3Steer.streams` vs. legacy `video_scale` / `audio_scale` / `split_end_percent`)
- **On `MiniMaxH3Steer`**: `streams` chooses `"both"` *(default, 1 DiT pass per step)*, `"video_only"` *(steers video and keeps audio unsteered across all active steps; 2 DiT passes per active step)*, or `"audio_only"` *(steers audio and keeps video unsteered; 2 DiT passes per active step)*. Because `MiniMaxH3Steer` fixes `split_end_percent = 1.0` internally, an unsteered stream stays genuinely unsteered for the entire run.
- **On legacy `MiniMaxH3ApplyControlVectors`**:
  - At default `(video_scale = 1.0, audio_scale = 1.0)`, the node runs **1 DiT pass per step**.
  - The two scales multiply the vector's **strength**, one for the pass that makes the video and one for the pass that makes the audio; they do not scale the model's output. When they differ, every step inside `[0.0, split_end_percent)` runs the model **twice** (the video is taken from one pass and the audio from the other), so those steps take twice as long. Equal values are one pass and only scale the strength (`control_vector_nodes.py`, `_effective_spec_strength` and the wrapper's `need_split`). `(1.0, 0.0)` = visual-only with untouched audio (`Arm 60` ran it at `split_end_percent = 0.34`); `(0.0, 1.5)` = a stronger acoustic effect with untouched video.
  - **After `split_end_percent` both streams are steered at the plain strength.** A stream scaled to `0` is steered from that step on, so `split_end_percent` below `1.0` does not keep a stream unsteered for the whole run; it only ends the two-pass cost early.
  - `mode = "hold"` on a legacy node with no additive vector before it in the chain changes nothing (it has nothing to take its direction out of) and logs a warning.
  - `ref_vae_end_percent` and `ref_vae_aug` are settings of the picture, not of one vector: across a chain the lowest value wins on every step, whatever each node's own window.

### Reference & keyframe VAE release (`MiniMaxH3ReleaseReferences`: `end_percent`, `clean_level`)
On `ref2va`, `fl2va`, and guide-latent graphs, clean 3D Video VAE latents (`ref_img` / `cond`) are packed directly into the DiT token sequence alongside noisy video/audio tokens. Use **`MiniMaxH3ReleaseReferences`** on the `MODEL` line (or `ref_vae_end_percent` / `ref_vae_aug` on legacy `MiniMaxH3ApplyControlVectors`):
- **Keep defaults (`end_percent = 1.0`, `clean_level = 0.999`)** whenever `<Subject 1>` should remain a recognizable human in the same pose/geometry (film looks, lighting grades, weather, or clothing experiments).
- **Lower `end_percent` (`ref_vae_end_percent`) to `0.16` or `0.33`** only when you intentionally **reshape 3D silhouette or replace live-action skin** (`claymation_stopmotion`, `felt_muppet`, `balloon_head_inflate`, `molten_lava_golem`). After `end_percent`, `_drop_visual_cond_from_layout` slices the `cond` and `ref_img` visual conditioning segments out of `PackedLayout` while preserving 3D RoPE cursor positions on generated video/audio segments, tag-`0` Qwen vision tokens, and speaker audio references (`Arm 66` lesson: rebuilding `PackedLayout` from scratch without `refs` shifts 3D RoPE cursors mid-trajectory and scrambles the clip).
- **`clean_level` (`ref_vae_aug`)** (`0.0 .. 1.0`, default `0.999`): controls how clean the reference VAE rows are mixed and labeled on the steps where they are held, clamped by `max(clean_level, stream_time)` (`dd054d6b`) so reference rows are never noisier than their timestep label.
