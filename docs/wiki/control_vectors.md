# Steering MiniMax H3: Edit Guidance, Control Vectors, and J-Lens

Last updated: 2026-11-11.

This pack ships three complementary mechanisms for steering, editing, and diagnosing MiniMax H3 renders without retraining weights or training a LoRA:

1. **`MiniMaxH3EditGuider` + `MiniMaxH3PromptEdit`** ([`edit_guidance.md`](edit_guidance.md), [`edit_guidance.py`](../../edit_guidance.py)): 2-pass latent-velocity guidance between a baseline prompt and a rewritten copy of that prompt.
2. **Contrastive Control Vectors** (`MiniMaxH3Steer`, `MiniMaxH3LoadControlVector`, `MiniMaxH3ReleaseReferences`, and legacy `MiniMaxH3ApplyControlVectors`; [`control_vectors.py`](../../control_vectors.py), [`control_vector_nodes.py`](../../control_vector_nodes.py)): 1-pass `5376`-d directions added to selected text rows after the 2-block `TokenRefiner` and before the 50 DiT blocks.
3. **J-Lens** (`MiniMaxH3WordVector`, [`j_lens.py`](../../j_lens.py), `models/j_lens/j_lens_h3.safetensors`): a precomputed `< 15 ms` CPU Jacobian and Wiener-inverse lexical operator that decodes any `5376`-d vector into English words, scrubs unwanted side-effect words (`hold_words`), and synthesizes live zero-file vectors from single-token words (`words`).

![End-to-end control vector pipeline](control_vectors/assets/control_vector_pipeline.svg)

---

## 1. Which tool to reach for (and why)

The single most important architectural distinction in H3 steering is **where the subtraction happens**:

- **Control vectors** (and **J-Lens `words`**) collapse a contrast into **one `5376`-d vector** and broadcast that single vector across every token in the chosen prompt mask *before* the 50 DiT blocks run (`0` extra DiT passes, `1x` render cost).
- **`MiniMaxH3EditGuider`** keeps both prompts **fully tokenized** through all 50 DiT blocks and subtracts their predicted latent velocities *after* cross-attention in pixel/audio latent space (`2` DiT passes per guided step, reducible to `9` or `10` total passes on 6-step Pika via `last_step`).

Because pooling collapses per-word token positions while latent-velocity subtraction preserves them, each tool wins in a distinct regime:

| Goal | Best tool | Why it works / what fails if you pick the wrong one |
|---|---|---|
| **Exact subject wardrobe, prop, or action swap** while keeping `<Picture 1>`'s face, hair, and existing room (`"black sweater -> 1970s rust-orange wide-collar blouse and flared jeans"`) | **`MiniMaxH3EditGuider`** (`scale = 1.5`; the clip the owner passed ran every step in two passes, the default `last_step`) | Every edited token (`"rust-orange"`, `"blouse"`, `"flared"`, `"jeans"`) attends to its own spatial patches across 50 DiT blocks, while unchanged face, hair, booth, and dialogue cancel in $(\mathbf{v}_{\text{edit}} - \mathbf{v}_{\text{orig}})$. Pooling the same `6 -> 22` token rewrite into one vector lost color-to-garment binding (rendering a **cream** blouse instead of rust-orange), and averaging 6 diverse 1970s outfits (`wardrobe_1970s`) canceled specific garments into a vague `"vintage, psychedelic"` room tint that left `<Subject 1>`'s black sweater unchanged. |
| **Continuous optical / film look or lighting grade** (`film_16mm_1974`, `noir`, `vhs_1991_camcorder`, `thermal_infrared`) | **10-pair Contrastive Control Vector** (`mask = "scene_only"`, or `description_only` to protect `<Subject 1>`) | Optical grain, halation, color cast, and contrast apply uniformly across the frame and reuse cleanly across any prompt at `1x` sampling cost with continuous dial control (`strength = 0.5..2.0`). |
| **Coupled sight + sound physics, weather, or pyrotechnics** (`pyro_chain_explosions`, `scorching_heatwave`, `wind_tunnel_gale`, `underwater_submerged`) | **10-pair Contrastive Control Vector** (`video_scale` / `audio_scale`) | H3 generates video and audio jointly inside one packed DiT sequence. Multi-pair GPU vectors built from coupled visual + `overall_soundscape:` contrasts carry multi-token dynamics and synchronized acoustics that unigram word lists drop or miss. |
| **Whole-scene world or material transformation that carries the subject with it** (`claymation_stopmotion`, `felt_muppet`, `balloon_head_inflate`, or seasonal/world shifts) | **Contrastive Vector or J-Lens `words`** on `scene_only` + `end_percent = 0.5` (+ `ref_vae_end_percent = 0.16..0.33` for silhouette changes) | When a coherent single-world push moves the entire environment (`scene_only`), the DiT's world prior carries contextual surfaces and attire along with the scene. Shortening `end_percent` to `0.5` prevents high-strength facial color burn on later steps. |
| **World shift *without* coupled wardrobe drift** (e.g., turning a sunny market into a snowstorm without putting a winter coat on the vendor) | **Control Vector or J-Lens `words` + `hold_words`** (`hold_words = "coat, parka"`, `hold_alpha = 0.5..1.0`) | Used in `$H3_OUTPUT_DIR/Video/cv_e2e_verify/10_t2va_jlens_words_00001-audio.mp4`: J-Lens projects the `"coat, parka"` direction out of the word vector in `< 1 ms` on CPU, and the clip has snowdrifts on the citrus stall with the vendor still in his canvas apron. No arm without the hold was rendered, so how much of that the hold did is not measured. |
| **Pre-render X-ray audit of any built vector** (catching washed-out or off-target builds before spending GPU time) | **J-Lens `inspect`** (automatic at the end of `build`) | Decodes any `5376`-d vector into English words in `< 15 ms` on CPU. On `wardrobe_1970s` the build's own readout was `"vintage, retro, nostalgia, nostalgic, decades, psychedelic"` with **zero garment nouns**. Nobody read it before rendering; read first, it would have said the vector carries an era and not clothing, which is what the renders then showed. |
| **Zero-file `< 2 ms` tinkering & rapid concept sketching** | **`MiniMaxH3WordVector` -> `MiniMaxH3Steer`** (or `words` on legacy `MiniMaxH3ApplyControlVectors`) | Synthesizes a family-matched `5376`-d vector on CPU at step 0 from single-token English words (`cos = +0.30..+0.52` with 10-pair GPU vectors; drops multi-token BPE words such as `muppet`, `halation`, `plasticine`, `pyrotechnics`). |
| **Audio-only acoustic space or vocal delivery** (`reverb_cathedral`, `pyro_klaxon_audio`, `whisper_asmr`, `operatic_bellow`) | **Contrastive Control Vector** with `target = "soundscape_only"` (`streams = "audio_only"` optional) | Steers `overall_soundscape:` and `non_diegetic_music:` (plus `<Audio N>:` lines) while leaving visual prose and dialogue words untouched (`streams = "audio_only"` on `MiniMaxH3Steer` keeps video unsteered across all steps). |

---

## 2. How the three mechanisms fit H3's architecture

Every H3 prompt passes through two text stages before the 50 DiT blocks see it:

1. **Causal encoder (`Qwen3-VL-32B`, `layers.0..49`, width `5120`)**:
   - Strictly causal (`is_causal = True`), with vision tokens tagged `0` and text tokens tagged `1` in `conditioning_token_tags`.
   - Row `0` is a massive attention sink (`||e_0|| ≈ 15,309` vs. `~292.3` median across rows `1..S-1`, a `52.4x` ratio). Every encoder pooling path in this pack excludes row `0`.
2. **Bidirectional `condition_proj` + 2-block `TokenRefiner` (width `5376`)**:
   - Lives inside each DiT (`fl2va` vs. `ref2va`) and runs **with active LoRAs applied** (such as the 6-step Pika LoRA).
   - Mixes tokens bidirectionally across the prompt and dampens attention-sink artifacts (`||r_0|| / median ≈ 2.27x`, post-refiner row median `≈ 266–270`).
   - `fl2va` and `ref2va` do **not** share a `5376`-d coordinate basis (cross-family cosine on identical text is `~0.01`), so every saved `.safetensors` vector records its `model_family` (`fl2va` or `ref2va`) and LoRA recipe, and `MiniMaxH3ApplyControlVectors` auto-resolves the matching family file at step 0.

From that `5376`-d `context` tensor onward:

- **`MiniMaxH3Steer`** (and legacy `MiniMaxH3ApplyControlVectors`) wraps the DiT forward pass (`WrappersMP.DIFFUSION_MODEL`), clones `context`, and modifies only the rows selected by `target` / `mask` / `fields` during `[start_percent, end_percent)`. Multiple chained nodes accumulate in `float64` inside one wrapper (`add_norm_matched` scales a unit vector to `67.5` shift norm at `strength = 1.0`).
- **J-Lens** maps between `5376`-d post-refiner space (`fl2va` or `ref2va`) and `24,912` single-token English words in `Qwen3-VL`'s `lm_head` using precomputed analytical Jacobians ($\bar{J}_{49\to 63}$ and $\bar{J}_{\text{ref}}^{(f)}$) and an empirical-covariance **Wiener inverse** ($J_{\text{Wiener}}^{(f)}$).
- **`MiniMaxH3EditGuider`** leaves `context` unmodified per branch, runs the DiT on `[original]` and `[edited]` sequentially (`h3_edit_pass` prevents ComfyUI from concatenating two `~18k`-token sequences into a `B = 2` OOM), and extrapolates the predicted velocity:
  $$\mathbf{v} = \mathbf{v}_{\text{original}}(x_t) + \text{scale} \cdot \left(\mathbf{v}_{\text{edited}}(x_t) - \mathbf{v}_{\text{original}}(x_t)\right)$$

---

## 3. Summary of verified empirical rules (2026-11-05 .. 2026-11-11)

Every rule below was tested on rendered 6-step Pika `t2va` and `ref2va` clips and verified on playback:

1. **Pool only the changed clause (`pooling = "diff_span"`)**:
   - Averaging across a whole 105-token scene when only 8–12 words changed (`scene_text`) dilutes raw vector norm by `2.1x–8.2x` (`10.35` vs. `85.01` on `head_swell`). `diff_span` (`[first_diff:end_diff]`) is the default across CLI `build`, `make`, and `MiniMaxH3MakeControlVector`.
2. **One pair-build implementation across CLI and ComfyUI (`fb3c978d`)**:
   - [`control_vector_nodes.py::encode_pair_set`](../../control_vector_nodes.py) and [`control_vectors.py::pool_pair_set`](../../control_vectors.py) power CLI `build`, `make`, Studio builds, and `MiniMaxH3MakeControlVector` identically (verified bitwise `torch.equal = True` across `vector.post_refiner`, `vector.encoder`, and `holds.post_refiner`).
3. **Where a pair sits in the build prompt matters far less than what the contrast means and where you apply it**:
   - Building the same wording contrast inside `subject_definitions:` vs. `detailed_description:` vs. a 3-field prompt produces nearly the same direction. At render time, however, **which mask receives the vector decides what moves**:
     - Aiming a film/room look (`film_16mm_1974`) at `subject_definitions` or `look_only` leaves `<Subject 1>` unchanged.
     - Aiming it at `description_only` (or ticking `description` in `fields`) grades the room and lighting while keeping `<Subject 1>` clean.
     - Aiming it at `scene_only` grades the whole shot.
4. **Always read the J-Lens top words—and heed build-time `b_side` alignment warnings—before rendering a new vector**:
   - Coherent single-concept pairs (`film_16mm_1974`, `claymation_stopmotion`, `pyro_chain_explosions`) surface concrete optical, material, or physical nouns at the top of their J-Lens readout.
   - Averaging heterogeneous nouns (`wardrobe_1970s`'s 6 different outfits) or repeating one Pole-A shirt/sweater across all pairs (`pirate_crew`, `wild_west_saloon`, `arctic_expedition`) cancels target attire words. CLI `build` / `make` and `MiniMaxH3MakeControlVector` now run [`j_lens.py::check_b_side_alignment`](../../j_lens.py) automatically at save time to warn whenever target/attire words drop out of `(+)`, and [`h3_pair_compiler.py`](../../h3_pair_compiler.py) supports rotated Pole-A baselines (`a1 | a2 | a3 -> b` or target-only `--bank-scenes 5`).
5. **Default `retention_analysis:` protection + tag-safe `release_subject_attire` on `ref2va`**:
   - Default lever masks (`scene_only`, `look_only`, `soundscape_only` in [`h3_spans.py`](../../h3_spans.py)) leave `retention_analysis:` preservation contracts unsteered so high-strength vectors (`strength = 2.25..3.5`) never scramble `fully_preserved` / `partially_preserved` instructions (tick `retention_analysis` in `fields` only when you explicitly want to steer it). To unlock `<Subject 1>`'s outfit cleanly for a vector or edit without hand-editing tags, toggle `release_subject_attire = True` on `MiniMaxH3PromptEdit`.
6. **High-strength facial red tint comes from late-step text steering, not `<Picture 1>` weight**:
   - When a strong vector (`strength = 2.0..3.0`) causes red burn around `<Subject 1>`'s eyes and cheeks, shortening **the vector's** window to `end_percent = 0.5` (first 3 of 6 Pika steps) removes the red tint on playback while preserving the style/scene effect—with `<Picture 1>` left at default `ref_vae_end_percent = 1.0` and `ref_vae_aug = 0.999`.
   - Only lower `ref_vae_end_percent` (`0.16` or `0.33`) when you want `<Subject 1>`'s **3D silhouette or material** to depart from the clean reference VAE latents (such as claymation, felt puppet, or head inflation).
7. **Project out correlated side effects (`[[hold]]` at build time, `mode = "hold"` or `hold_words` at render time)**:
   - `film_16mm_1974` pulls warm amber cast alongside grain; `film_16mm_1974_neutral` adds a `[[hold]]` group (`warm_colour_cast`) that removes the amber shift while keeping grain and gate weave (`cos = 0.72` between the two).
   - At render time, wiring `MiniMaxH3LoadControlVector` (or `MiniMaxH3WordVector`) into `MiniMaxH3Steer.hold_vector` (e.g., `cursed_cctv` + `hold_vector = chaos_panic`), or setting `hold_words` (`hold_words = "coat, parka"`), scrubs unwanted collateral concepts from the additive delta without altering the base prompt.

---

## 4. Shipped API workflows

Generated by [`workflows/build_workflows.py`](../../workflows/build_workflows.py), wiring `MiniMaxH3Steer` (bare vector stem auto-resolved to the loaded checkpoint's `fl2va`/`ref2va` + LoRA sibling) and `MiniMaxH3ReleaseReferences` where reference or guide VAE latents are present:

| Workflow | Mode | What it demonstrates |
|---|---|---|
| `workflows/h3_text_to_video_pika_control_vector_api.json` | 6-step Pika `t2va` | `MiniMaxH3Steer` (`film_16mm_1974` at `+1.15`, `target = "scene_only"`) |
| `workflows/h3_text_to_video_pika_dual_control_vectors_api.json` | 6-step Pika `t2va` | Two chained `MiniMaxH3Steer` nodes (`film_16mm_1974` at `+1.15` on `scene_only` + `reverb_cathedral` at `+1.25` on `soundscape_only`) |
| `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json` | 6-step Pika `ref2va` | Reference-anchored `<Subject 1>` shot with `MiniMaxH3Steer` (`target = "description_only"` so the kitchen grades while `<Subject 1>` stays clean) + `MiniMaxH3ReleaseReferences` |
| `workflows/h3_text_to_video_pika_edit_guidance_api.json` | 6-step Pika `t2va` | `MiniMaxH3PromptEdit` + `MiniMaxH3EditGuider` (`scale = 1.5`) |
| `workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json` | 6-step Pika `ref2va` | Exact `<Subject 1>` wardrobe/prop/scene edit on a `<Picture 1>` reference shot via `MiniMaxH3PromptEdit` (`release_subject_attire`) + `MiniMaxH3EditGuider` |
| `workflows/image/h3_p_h3_image_t2i_control_vector_api.json` | 8-step `P-H3-Image` `t2i` | `MiniMaxH3Steer` (`film_16mm_1974` at `+1.15`, `target = "scene_only"`) |
| `workflows/image/h3_p_h3_image_edit_control_vector_api.json` | 8-step `P-H3-Image` `edit` | `MiniMaxH3Steer` (`target = "scene_only"`) + `MiniMaxH3ReleaseReferences` on `<Picture 1>`/`<Picture 2>` |
| `workflows/image/h3_pika_image_t2i_control_vector_api.json` | 6-step Pika still `t2i` | `MiniMaxH3Steer` (`film_16mm_1974` at `+1.15`, `target = "scene_only"`) |
| `workflows/image/h3_pika_image_edit_control_vector_api.json` | 6-step Pika still `edit` | `MiniMaxH3Steer` (`target = "description_only"`) + `MiniMaxH3ReleaseReferences` on `<Picture 1>` |
| `workflows/image/h3_probe_pika_image_guide_edit_control_vector_api.json` | 6-step Pika guide `edit` | `MiniMaxH3Steer` (`target = "non_vision"`, core `MiniMaxH3ReferenceToVideo`) + `MiniMaxH3ReleaseReferences` |

---

## 5. Guide map

| Page | What it covers |
|---|---|
| [`edit_guidance.md`](edit_guidance.md) | **2-pass Edit Guidance under the hood**: `MiniMaxH3EditGuider` velocity math, why unpooled tokens preserve color/noun binding, `first_step` & `last_step` pass budgets, `MiniMaxH3PromptEdit` syntax & refusals |
| [`control_vectors/building_pairs_and_vectors.md`](control_vectors/building_pairs_and_vectors.md) | **How contrastive vectors are built**: two-stage H3 text geometry, `diff_span` pooling, `[[hold]]` Gram-Schmidt projection, why coherent concepts beat grab-bag averages, `from-prompt`, `from-toml`, and `make-media` |
| [`control_vectors/comfyui_nodes_and_controls.md`](control_vectors/comfyui_nodes_and_controls.md) | **ComfyUI nodes & runtime knobs**: `MiniMaxH3Steer`, `MiniMaxH3WordVector`, `MiniMaxH3LoadControlVector`, `MiniMaxH3ReleaseReferences` (+ legacy `ApplyControlVectors`), lever `target` vs. `fields`, 6-step `end_percent = 0.5` math, `streams`, and reference release |
| [`control_vectors/j_lens_diagnostics_and_word_lab.md`](control_vectors/j_lens_diagnostics_and_word_lab.md) | **J-Lens under the hood**: analytical Jacobians & Wiener inverse, `< 15 ms` X-ray readouts, `< 1 ms` `hold_words` scrubbing, `< 2 ms` `words` synthesis, single-token BPE limits, and `build-j-lens` |
| [`control_vectors/examples_and_recipes.md`](control_vectors/examples_and_recipes.md) | **Copy-paste recipes**: room-only film grade on `ref2va`, coupled weather/pyro, claymation world shift, snow-without-parka `hold_words`, CPU `sculpt`, and `MiniMaxH3EditGuider` wardrobe swaps |
| [`control_vectors/preset_catalog_and_cli_reference.md`](control_vectors/preset_catalog_and_cli_reference.md) | **CLI & starter bank reference**: every `bench/control_vectors.py` subcommand and flag, `.safetensors` file & metadata contract, and the 38-file `prompt_bank/control_vectors/` catalog |
| [`control_vectors/llm_prompting_guide.md`](control_vectors/llm_prompting_guide.md) | **Authoring contrastive pairs with an LLM**: standalone vs. automated system prompts, single-clause & length-matching discipline, concrete Pole A baselines, and `from-md` import |
| [`control_vectors/studio_ui_guide.md`](control_vectors/studio_ui_guide.md) | **Control Vector Studio (`bench/control_vectors.py ui`)**: browser workbench screens, shared CLI/node build pipeline, GPU serialization, session-token security, and folder confinement |
