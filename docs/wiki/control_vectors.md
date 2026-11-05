# Control Vectors for MiniMax H3: Code, Prompt Bank, LLM Authoring, and Readout Index

last updated: 2026-11-05

One-stop router and index for activation control vectors in this pack: the runtime nodes and CLI, the token-span partitioner, the 14 contrastive-pair `.toml` sets in `prompt_bank/control_vectors/`, the self-contained LLM system prompt that authors them, the ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` activation-readout and batch-labelling pipeline, and the measured Pika H3 Flex and `ref2va` records.

---

## 1. Where Control Vectors Operate in H3

MiniMax H3 encodes text and multimodal references with stock `Qwen3-VL-32B-Instruct`, taps the hidden state after **`model.language_model.layers.49`** (`50`th transformer block, `[L, 5120]`), projects it via `condition_proj` (`5120 -> 5376`), and passes it through a 2-block bidirectional **`token_refiner`** (`[L, 5376]`, ending in `final_norm`) inside `MiniMaxH3.extra_conds` once per sampling call before the 50-block DiT denoising loop begins.

Our control vectors operate at two points:
1. **`vector.post_refiner` (`[5376]`, primary steering point for video+audio generation)**:
   - Extracted from contrastive prompt pairs `(A_i, B_i)` after `condition_proj` + `token_refiner`, mean-pooled over either **`scene_text`** tokens (default for whole-scene style/atmosphere vectors) or **`from_first_diff`** tokens (`--pooling from_first_diff`, recommended for single-clause end-of-section physical/action vectors like `head_swell` and `zero_gravity` to avoid `2.12x–2.28x` prefix dilution across the shared ~70% prefix tokens), always excluding vision blocks `tag == 0`, `<Picture i>: ` / `<Video k>: ` / `<Audio j>: ` labels, `<T.T seconds>` timestamps, `<d>...</d>` dialogue spans, and `(S1)` speaker tags via [`h3_spans.py`](../../h3_spans.py).
   - Applied once per sampling call inside a `DIFFUSION_MODEL` wrapper around `preprocess_text_embeds` ([`control_vectors.py`](../../control_vectors.py)), scaling the unit direction by `strength * median_row_norm` across the selected token mask (`non_vision` or `scene_text`) while leaving vision rows (`tag == 0`) bitwise untouched and sequence length `L` invariant.
2. **`vector.encoder` (`[5120]`, optional pre-projection tap at `layers.49`)**:
   - Mean-pooled over `scene_text` or `from_first_diff` tokens (excluding token index `0`, where Qwen3-VL's attention sink sits at ~`3679` norm vs. ~`191`–`297` median row norm).
   - Used both for pre-refiner vs. post-refiner geometry comparisons and for direct `layers.49 -> layers.50..63 -> lm_head` activation-steered self-report and lexical log-odds readout in `Qwen3-VL-32B-Instruct` ([`docs/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py`](../research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py)).

---

## 2. Code, Nodes, Shipped Workflows, CLI, and Checks

| File | Role | Key Symbols & Entry Points |
|---|---|---|
| [`control_vectors.py`](../../control_vectors.py) | Core math, `.toml` pair loader, `.safetensors` I/O, model family & recipe verification, runtime `DIFFUSION_MODEL` wrapper, and ComfyUI nodes | `MiniMaxH3MakeControlVector`, `MiniMaxH3ApplyControlVectors`, `load_pairs`, `differing_span` (`MAX_SHARED_INTERIOR_WORDS = 5`), `mean_difference`, `flip_null`, `random_null`, `hold_basis`, `project_out`, `apply_post_refiner` (`add`, `subtract`, `amplify`, `project_out`), `extract_model_recipe`, `save_control_vector`, `load_control_vector` |
| [`h3_spans.py`](../../h3_spans.py) | Token-span partitioner for H3 conditioning sequences (`vision`, `label`, `timestamp`, `dialogue`, `speaker`, `scene`) | `spans`, `pooling_mask` (`scene_text`, `all_tag1`, `from_first_diff`), `expand_token_ids`, `first_differing_index`, `verify_tokenizer_constants` |
| [`bench/control_vectors.py`](../../bench/control_vectors.py) | Standalone CLI for building, inspecting, metering, and building gate-ladder workflows for control vectors | Subcommands: `make` (`--pairs`, `--unet`, `--lora`, `--pooling`, `--include-encoder`, `--overwrite`), `inspect`, `meter`, `gate` |
| [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py) | Fast CPU guard covering `[C1]`–`[C7]`: tokenizer constants, span partitioning, `.toml` single-clause diff extraction & refusals, `float64` math & `flip_null`, `.safetensors` round-trip & recipe guards, `apply_post_refiner` bitwise vision preservation, and runtime single/dual wrapper | `check_h3_spans_constants`, `check_h3_spans_partition_and_refusals`, `check_pair_loading_and_refusals`, `check_math_and_synthetic_statistics`, `check_safetensors_and_recipe_guards`, `check_apply_post_refiner_and_vision_invariance`, `check_runtime_wrapper_and_tracked_tomls` |
| [`workflows/build_workflows.py`](../../workflows/build_workflows.py) | Workflow generator helper for wiring one or more chained `MiniMaxH3ApplyControlVectors` nodes (`"115"`, `"125"`, ...) into `t2va`, `fl2va`, and `r2v` (`ref2va`) API graphs | `build_api(..., control_vector=...)`, `build_control_vector_arm` (`mode`, `vector_name`, `strength`, `mask`, `start_percent`, `end_percent`, `apply_holds`, `lora`, `lora_branch`, `ref_images`) |
| [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json) | Shipped `t2va` single-control-vector API workflow (Pika H3 Flex 6-step LCM `PIKA_FL2VA_SVD995_LORA`, `MiniMaxH3ApplyControlVectors` at node `"115"`) | UI companion: [`workflows/h3_text_to_video_pika_control_vector.json`](../../workflows/h3_text_to_video_pika_control_vector.json) |
| [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json) | Shipped `t2va` dual-control-vector API workflow chaining two `MiniMaxH3ApplyControlVectors` nodes in series (`"115"` `film_16mm_1974` at `1.8` $\to$ `"125"` `chaos_panic` at `2.2` $\to$ `BasicGuider`) | UI companion: [`workflows/h3_text_to_video_pika_dual_control_vectors.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors.json) |
| [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) | Shipped `ref2va` character-reference control-vector API workflow (Pika H3 Flex 6-step LCM `PIKA_REF2VA_SVD995_LORA`, `MiniMaxH3ApplyControlVectors` at node `"115"` with `mask="non_vision"` so `<Picture 1>` vision tokens stay bitwise invariant) | UI companion: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json) |

### ComfyUI Nodes ([`control_vectors.py`](../../control_vectors.py))

- **`MiniMaxH3MakeControlVector`**:
  - Extracts a control vector from a `.toml` pair file in `prompt_bank/control_vectors/` using the connected `CLIP` (`Qwen3-VL-32B-Instruct`) and `MODEL` (`token_refiner`, including any active `MiniMaxH3LoRABranch` LoRAs), writes `<name>__<family>__<recipe_hash>.safetensors` into `<model_store>/control_vectors/`, and prints a quality report (`mean_pairwise_cos`, `min_pair_to_mean_cos`, `pc1_explained_var`, `snr_vs_flip_null`, `cos_to_holds`).
- **`MiniMaxH3ApplyControlVectors`**:
  - Wraps `MODEL` to apply up to 3 control vectors (accumulated in `float64`) to the refined text conditioning inside `extra_conds`.
  - Supports `mode` in `{"add", "subtract", "amplify", "project_out"}`, `mask` in `{"non_vision", "scene_text"}`, `apply_holds` (projects out nuisance `[[hold]]` directions before scaling), and `start_percent` / `end_percent` step scheduling via `conditioning_set_values`.

---

## 3. Prompt Bank: Contrastive Pair `.toml` Files (`prompt_bank/control_vectors/`)

Every `.toml` file under `prompt_bank/control_vectors/` defines a `[vector]` header (`name`, `quality`, `a_side`, `b_side`, `mode`, `frame`, `exclude_shots`), 10 or more `[[pair]]` entries (`a`, `b`, optional `weight`), and optional `[[hold]]` nuisance-direction pairs (`group`, `a`, `b`, optional `weight`).

In `frame = "scene"`, [`control_vectors.py::differing_span`](../../control_vectors.py#L138-L215) enforces that `a` and `b` are character-for-character identical outside **one contiguous clause within a single section** (allowing at most `MAX_SHARED_INTERIOR_WORDS - 1 = 4` shared interior words inside that clause, such as a compound surface noun phrase like `"the polished mahogany desk"`).

| `.toml` File | Mode | Frame | Target Quality (`a_side` $\to$ `b_side`) | Holds | Provenance / Notes |
|---|---|---|---|---|---|
| [`chaos_panic.toml`](../../prompt_bank/control_vectors/chaos_panic.toml) | `t2va` | `scene` | Calm, unhurried composure $\to$ frantic wide-eyed panic, trembling hands, and panicked vocal urgency | 0 | De-confounded (`7593e65f`): 10 distinct scenes/characters, single-attribute baseline `a` |
| [`cursed_cctv.toml`](../../prompt_bank/control_vectors/cursed_cctv.toml) | `t2va` | `scene` | Clean rectilinear optics & clear room audio $\to$ 2004 fisheye CCTV distortion, green phosphor tint, scanlines, and 60 Hz mic hum | 0 | De-confounded (`7593e65f`): 10 distinct scenes/characters, single-attribute baseline `a` |
| [`felt_muppet.toml`](../../prompt_bank/control_vectors/felt_muppet.toml) | `t2va` | `scene` | Natural human skin, anatomy, and voice $\to$ fuzzy stitched felt hand-puppet anatomy with ping-pong-ball eyes and squeaky puppet voice | 0 | **Verbatim LLM output (`dde679e5`)** from [`2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) (10/10 single-clause pass, 0 edits) |
| [`fever_melt.toml`](../../prompt_bank/control_vectors/fever_melt.toml) | `t2va` | `scene` | Solid rigid geometry and steady voice $\to$ surreal wax-melting architectural & facial distortion with warped pitch | 0 | De-confounded (`7593e65f`): 10 distinct scenes/characters, single-attribute baseline `a` |
| [`film_16mm_1974.toml`](../../prompt_bank/control_vectors/film_16mm_1974.toml) | `t2va` | `scene` | Clean modern digital cinema camera $\to$ 1974 16mm photochemical film look, heavy grain, warm halation, gate weave | 6 (`warm_tint`, `soft_focus`) | Gate benchmark vector (16 pairs + 6 holds across 2 hold groups); evaluated in [`2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md) |
| [`film_16mm_1974_bare.toml`](../../prompt_bank/control_vectors/film_16mm_1974_bare.toml) | `t2va` | `bare` | Bare keyword phrases for modern digital vs. 1974 16mm film grain | 0 | Ablation pair set (`frame = "bare"`) testing bare phrases vs. full H3 scene framing (`cos = +0.74` at `layers.49`, `+0.56` post-refiner) |
| [`head_swell.toml`](../../prompt_bank/control_vectors/head_swell.toml) | `t2va` | `scene` | Normal human head proportions $\to$ head physically inflating/expanding like a balloon while speaking | 0 | **Verbatim LLM output (`db821224`)** from [`2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) (10/10 single-clause pass, 0 edits, zero target-noun cancellation) |
| [`infomercial_fail.toml`](../../prompt_bank/control_vectors/infomercial_fail.toml) | `t2va` | `scene` | Calm, coordinated task execution $\to$ exaggerated clumsy infomercial disaster (spilling, fumbling, comic distress) | 0 | De-confounded (`7593e65f`): 10 distinct scenes/characters, single-attribute baseline `a` |
| [`lettering.toml`](../../prompt_bank/control_vectors/lettering.toml) | `t2va` | `scene` | Blank, unlettered surfaces $\to$ bold, legible painted/printed/stenciled English signage and typography on surfaces | 0 | 16 scene-framed pairs for surface typography |
| [`lettering_bare.toml`](../../prompt_bank/control_vectors/lettering_bare.toml) | `t2va` | `bare` | Bare keyword phrases for blank surfaces vs. legible English lettering | 0 | Ablation pair set (`frame = "bare"`, `cos = +0.82` with `lettering.toml` at `layers.49`) |
| [`noir.toml`](../../prompt_bank/control_vectors/noir.toml) | `t2va` | `scene` | Even, natural full-color daylight $\to$ high-contrast monochrome 1940s film noir chiaroscuro shadows and silver grain | 0 | 12 scene-framed pairs |
| [`reverb_cathedral.toml`](../../prompt_bank/control_vectors/reverb_cathedral.toml) | `t2va` | `scene` | Dry, close-mic indoor room acoustics $\to$ vast Gothic stone cathedral reverberation and long echoing decay (in `overall_soundscape:`) | 0 | 12 pairs modifying only `overall_soundscape:` while keeping `integrated_multimodal_description:` identical |
| [`vocal_strain.toml`](../../prompt_bank/control_vectors/vocal_strain.toml) | `t2va` | `scene` | Calm, conversational vocal delivery $\to$ vein-bulging, raspy, full-lung operatic screaming | 0 | De-confounded (`7593e65f`): 10 distinct scenes/characters, single-attribute baseline `a` |
| [`zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml) | `ref2va` | `scene` | Loose tabletop objects remaining stationary on surfaces under normal gravity $\to$ loose objects detaching and floating weightlessly upward as gravity ceases | 0 | **Verbatim LLM output (`db821224`)** from [`2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) (10/10 single-clause pass, 0 edits, zero content-word overlap) |

---

## 4. Authoring Scene Prompts and Contrastive Pairs with an LLM

- **System Prompt**: [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md)
  - Self-contained system prompt for `Qwen3-VL-32B-Instruct` (or any frontier LLM) that outputs:
    1. **`### Test Scene Prompt`**: A 100% unsteered baseline prompt featuring the user's requested character, setting, activity, and dialogue line (`3`-section `t2va`/`fl2va` format or `6`-section `ref2va` format when `<Picture 1>` is provided).
    2. **`### Contrastive Pairs (1–10)`**: 10 diverse `Prompt A (Baseline)` vs. `Prompt B (Steered)` pairs set in 10 unrelated locations/professions with 5 women and 5 men, zero overlap with the test scene's setting/character/dialogue, `(S1)` outside `<d>...</d>`, and a single contiguous 8–16 word predicate clause placed at the end of the modified section (after `<d>...</d>`) so all 10 pairs pass `control_vectors.load_pairs` verbatim without hand-editing.
  - **Why `Prompt A` Must Only Negate the Single Target Attribute Without Naming the Target Noun**:
    - Earlier template-generated `.toml` sets stuffed 5 baseline traits (`"clean modern digital cinema camera with realistic human skin, natural fabric textures, solid geometry, and calm unhurried composure"`) into every `Prompt A` and reused the same 10 silent base scenes across 5 vectors. Because $\vec{v} = \text{normalize}(\overline{h(B)} - \overline{h(A)})$, subtracting a multi-trait `Prompt A` subtracted all 5 traits at once, creating `+0.42..+0.79` cross-vector cosine overlap at `layers.49`.
    - Similarly, placing the target body-part noun (`"head"`, `"cranium"`, `"skull"`) or phenomenon keyword (`"gravity"`) inside `Prompt A` or in the shared prefix before `first_diff` cancels that noun in $B - A$, while padding `Prompt B` with generic rate adverbs (`slowly`, `gradually`, `steadily`, `progressively`, `rapidly`) causes rate adverbs to dominate the averaged vector. Describing `Prompt A`'s baseline with general anatomical/physical phrasing and reserving target nouns/verbs exclusively for `Prompt B`'s end-of-section clause (`db821224`, `0edcd08f`) raised causal `from_first_diff` `raw_norm` at `layers.49` to `107.04` (`zero_gravity`) and `112.20` (`head_swell`) and dropped `cos(head_swell, zero_gravity)` from `+0.62` to `+0.30`.

---

## 5. ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` Readout & Batch Labeller

Because H3's text encoder is stock `Qwen3-VL-32B-Instruct` truncated after `layers.49`, loading the full 64-layer `Qwen3-VL-32B-Instruct` model in ExLlamaV3 allows us to inject `vector.encoder` (`[5120]`) directly into the residual stream at `layers.49` and read out what `layers.50..63` + `lm_head` decode in natural language.

| File | Purpose |
|---|---|
| [`docs/research/control_vectors/exl3_qwen3vl/runbook.md`](../research/control_vectors/exl3_qwen3vl/runbook.md) | End-to-end runbook for quantizing `Qwen3-VL-32B-Instruct` on a single 24 GB RTX 4090, hooking `model.language_model.layers.49` (`model.modules[53]`) with `resid_pending` materialization, and avoiding KV page-cache collisions across steering sweeps |
| [`docs/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py`](../research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py) | Batch labeller (`--pooling {scene_text,from_first_diff}`) combining: (1) blind pair-diff synthesis (`--mode blind_pairs`), (2) direct `layers.49 -> lm_head` word-initial log-odds shift (`compute_top_logit_shifts`), (3) coupled-Gumbel (`rand_u32`) `layers.49`-steered rollouts across $\alpha \in \{-0.15, +0.00, +0.15, +0.22\}$ (`--mode self_report`), (4) VLM video/contact-sheet critique (`--mode vlm`), and (5) automatic stale-`.safetensors` fallback (`sf_exl3_cos < 0.985`) |
| [`docs/research/control_vectors/exl3_qwen3vl/build_h3_cal_data.py`](../research/control_vectors/exl3_qwen3vl/build_h3_cal_data.py) | Builds the `250 x 2048` hybrid Hessian calibration dataset (`h3_cal_data.safetensors`) combining standard ExLlamaV3 calibration rows, H3 scene prompts, and `.toml` contrastive pairs |
| [`docs/research/control_vectors/exl3_qwen3vl/h3_readout_5.37bpw.yaml`](../research/control_vectors/exl3_qwen3vl/h3_readout_5.37bpw.yaml) | `5.37 bpw` EXL3 quantization recipe (`6 bpw` on all 64 `self_attn` blocks and all 14 readout blocks `layers.50..63`, `5 bpw` on trunk MLPs `layers.0..49`, `8 bpw` `lm_head`; `99.9972%` cosine similarity at `layers.49` and `99.9989%` at `lm_head` vs. `bf16`) |
| [`docs/research/control_vectors/exl3_qwen3vl/h3_multimodal_4.65bpw.yaml`](../research/control_vectors/exl3_qwen3vl/h3_multimodal_4.65bpw.yaml) | `4.65 bpw` EXL3 quantization recipe (`17.86 GiB` total GPU weights with `-vb 6`, leaving `~5.3–5.5 GiB` VRAM headroom for multi-frame video contact sheets) |
| [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md) | Generated batch-labelling and `layers.49` readout report across all 14 `.toml` control vectors (summary table, `14x14` `layers.49` cosine similarity matrix, top `+v_encoder`/`-v_encoder` boosted words, and coupled $\alpha$ rollouts on Anchors A & B) |

---

## 6. Why Initial Physical-Action Renders (`25..30`) Failed and How We Got Them to Work (`31..40`)

When testing the single-clause end-of-section physical transformations (`head_swell.toml` on `t2va` and `zero_gravity.toml` on `ref2va`) at `1344x768x141` (`5.88s` at `24 fps`) under Pika H3 Flex (`6`-step `lcm`), our initial renders (`arms 25..30` at `s = +2.50, +4.00`) showed only subtle props or slight cranial widening rather than unmistakable mid-air levitation or balloon-scale head expansion.

Systematic ablation across `arms 31..40` ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)) isolated and solved all four root causes:

1. **Post-Dialogue Timeline Budget (`1.5s` Speech + `4.3s` Effect Window vs. `4.8s` Speech)**:
   - In every pair of [`head_swell.toml`](../../prompt_bank/control_vectors/head_swell.toml) and [`zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml), the changed clause sits at the **end** of `integrated_multimodal_description:` **after** the `<d>...</d>` dialogue line (`"...says <d>...</d>, while his skull and cranium inflate..."` / `"...while the brass letter opener, fountain pen, and inkpot drift upward..."`).
   - In `arms 25..30`, the test prompts had 7–13 word `<d>...</d>` sentences (`<d>What did you put in this coffee? My head feels enormous right now!</d>`, `<d>Could I get a refill on this coffee when you get a chance?</d>`) that occupied `00:00..00:04.8` of the `5.88s` clip, leaving `<1.0s` (`~20` frames) after speech for the post-dialogue transformation to unfold.
   - **Fix**: Keep the test prompt's `<d>...</d>` line concise (`<d>Morning!</d>` or `<d>Another pie?</d>`, finishing by `~1.5s`) so the `141`-frame (`5.88s`) window provides `4.0–4.3s` of post-speech runtime (`arms 31..40`), or increase frame length when testing long multi-clause monologues.
2. **Omitting Contradictory Pole-A Grounding Clauses in the Base Prompt**:
   - In `arms 25..27`, the `ref2va` base prompt ended with `"while the porcelain coffee mug and silver fork rest flat on the table"`—an explicit Pole-A static-gravity constraint that fought the injected `zero_gravity` control vector token-for-token.
   - **Fix**: An unsteered `Test Scene Prompt` must describe the scene neutrally and stop after the character's action/dialogue without asserting that tabletop objects remain stationary or that anatomy remains fixed.
3. **`2.12x–2.28x` Prefix Dilution Under `scene_text` Pooling vs. `from_first_diff` Pooling**:
   - Because single-clause `.toml` pairs share ~85–110 identical scene-setup tokens before the 8–12 word end-of-section diff clause, mean-pooling `token_refiner` over all `scene_text` tokens averages ~70% near-zero prefix diffs with ~30% suffix diffs:
     - **`head_swell` (`fl2va`)**: `scene_text` `raw_norm = 10.35` vs. `from_first_diff` `raw_norm = 23.57` (**`2.28x` stronger**, `cos = 0.9427`, `mean_pairwise_cosine` improves `0.7066 -> 0.7208`).
     - **`zero_gravity` (`ref2va`)**: `scene_text` `raw_norm = 10.65` vs. `from_first_diff` `raw_norm = 22.62` (**`2.12x` stronger**, `cos = 0.9403`, `mean_pairwise_cosine` improves `0.7466 -> 0.7558`).
   - **Fix**: Either extract single-clause end-of-section vectors with **`--pooling from_first_diff`** and run at standard **`s = +3.0..+3.5`** (`arms 39, 40`, where $\|\Delta \mathbf{r}_{\text{PR}}\| = 82.49$ and $79.17$), or scale `scene_text`-pooled vectors to norm-matched **`s = +6.5..+8.0`** (`arms 32..33, 35..36`, where $\|\Delta \mathbf{r}_{\text{PR}}\| = 67.3..85.2$).
4. **On-Manifold Physical Reconciliation vs. Unassisted Mid-Air Levitation (`from_first_diff`)**:
   - Because `apply_post_refiner` broadcasts the 1D `[5376]` vector uniformly across all non-vision tokens (`[L, 5376]`), at `scene_text` `s = +6.5..+8.0` (`arms 32..33`) the DiT + Pika LoRA reconciles upward object trajectories with its real-world physical prior by having `<Subject 1>` reach out and lift the mug, spoon, and fork to eye level.
   - Extracting with **`--pooling from_first_diff`** removes the 70% shared-prefix noise: at **`s = +3.50` (`40_ref2va_diner_zero_grav_ffd_s3p5`)**, a ceramic dinner plate, a coffee mug, and a silver fork **levitate unassisted in mid-air** above the diner booth across `f = 42..140` while `<Subject 1>`'s hands hold two other utensils separately. On `head_swell`, both `scene_text` `s = +8.00` (`36`) and `from_first_diff` `s = +3.50` (`39`) match the explicit Pole-B text positive control (`34`) with a **`~1.70x` screen-space cranial ballooning** (`25.8% -> 43.9%` of frame width) and a synthesized pressurized steam-inflation hiss.

---

## 7. How to Run `t2va`, Dual-Control-Vector `t2va`, and `ref2va` Workflows

### 7.1 Single Control Vector on `t2va`
- **Shipped API Workflow**: [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json) (UI companion: [`workflows/h3_text_to_video_pika_control_vector.json`](../../workflows/h3_text_to_video_pika_control_vector.json)).
- **Node Wiring**: `MiniMaxH3LoRABranch (18, pika_h3_flex_fl2va_svd995_bf16.safetensors) -> ModelSamplingSD3 (19, shift=5.0) -> ModelAttentionBackend (58, triton) -> MiniMaxH3Sol (21) -> MiniMaxH3ApplyControlVectors (115) -> BasicGuider (9)`.
- **Recommended Strengths**:
  - Whole-scene style/atmosphere/persona vectors (`film_16mm_1974`, `felt_muppet`, `cursed_cctv`, `chaos_panic`, `vocal_strain`, `reverb_cathedral`, `fever_melt`, `infomercial_fail`): `strength = 1.8 .. 2.8` with default `scene_text`-pooled `.safetensors`.
  - End-of-section physical transformation vectors (`head_swell`): `strength = 3.5` with `--pooling from_first_diff` (`head_swell_ffd__fl2va__from_first_diff__lora_2a8c2b52.safetensors`), or `strength = 6.5 .. 8.0` with `scene_text` pooling (`head_swell__fl2va__lora_2a8c2b52.safetensors`).

### 7.2 Dual Control Vectors on `t2va` (Chained Nodes or Multi-Slot Node)
- **Shipped API Workflow**: [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json) (UI companion: [`workflows/h3_text_to_video_pika_dual_control_vectors.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors.json)).
- **Two Ways to Combine Multiple Control Vectors**:
  1. **Chained `MiniMaxH3ApplyControlVectors` Nodes in Series (`"115"` $\to$ `"125"` $\to$ `BasicGuider (9)`)**:
     - Used in [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json) (and generated by passing a list of specs `control_vector=[(vec1, s1), (vec2, s2)]` to `build_workflows.py::build_api`).
     - Gives each vector its own independent `mode` (`add`, `subtract`, `amplify`, `project_out`), `mask` (`non_vision`, `scene_text`), `apply_holds`, and step schedule (`start_percent`, `end_percent`).
     - Measured examples (`arms 37, 38`): chaining `film_16mm_1974` (`s = +1.80` at node `"115"`) $\to$ `chaos_panic` (`s = +2.20` at node `"125"`) simultaneously applies 1974 16mm celluloid film grain/color/wardrobe and frantic rapid-fire panicked monologue (`38_t2va_coffee_dual_film16mm_plus_chaos`), while chaining `felt_muppet` (`s = +2.20`) $\to$ `reverb_cathedral` (`s = +2.20`) combines stitched felt puppet transformation with echoing stone-hall room acoustics (`37_t2va_coffee_dual_muppet_plus_cathedral`).
  2. **Single `MiniMaxH3ApplyControlVectors` Node (`vector_name` + `vector_name_2` + `vector_name_3`)**:
     - Accumulates up to 3 vectors in `float64` inside one node when all vectors share the same `mode`, `mask`, and step window (`arms 15..17`).

### 7.3 Character-Reference Control Vectors on `ref2va`
- **Shipped API Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) (UI companion: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json)).
- **Step-by-Step Recipe for `ref2va`**:
  1. **Extract a `ref2va`-family `.safetensors` vector** matching the `ref2va` UNet (`h3_27B_ref2va_nt_int8_sn3_v2.safetensors`) and `ref2va` Pika LoRA (`pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0`). Because `ref2va` and `fl2va` have distinct `condition_proj` + `token_refiner` weights, `MiniMaxH3ApplyControlVectors` verifies `model_family == "ref2va"` at runtime and refuses mismatched `fl2va` vectors unless `allow_recipe_mismatch=True`. For end-of-section physical actions (`zero_gravity.toml`), pass `--pooling from_first_diff`:
     ```bash
     $PY bench/control_vectors.py make \
         --pairs prompt_bank/control_vectors/zero_gravity.toml \
         --unet h3_27B_ref2va_nt_int8_sn3_v2.safetensors \
         --lora pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0 \
         --pooling from_first_diff \
         --name zero_gravity_ffd \
         --device default --refiner-device cuda \
         --include-encoder --overwrite
     ```
  2. **Wire `LoadImage` + `MiniMaxH3ReferenceConditioning` (`mode="r2v"`)**:
     - Connect the reference portrait (`<Picture 1>`) to `MiniMaxH3ReferenceConditioning` (`node 10`) with a 6-section `ref2va` prompt (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`). Keep `<d>...</d>` concise (`1–5` words) and do not include any Pole-A static-gravity clause.
  3. **Wire `MiniMaxH3ApplyControlVectors` (`node 115`) with `mask="non_vision"`**:
     - Set `mask="non_vision"` (or `"scene_text"`). `apply_post_refiner` checks `text_vision_ModalMask` (`tag == 0` for `<Picture 1>`'s vision patch tokens, `tag == 1` for text tokens) and leaves all `<Picture 1>` vision embeddings **bitwise identical (`0.0` diff)** so `<Subject 1>`'s face, beard, glasses, and clothing identity remain locked while the text tokens steer the scene (`strength = 3.5` for `zero_gravity_ffd`, or `6.5 .. 8.0` for `scene_text`-pooled `zero_gravity`).

---

## 8. Research Notes & Measured Records

| Document | What It Covers |
|---|---|
| [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md) | Architecture, mathematical formulation (`mean_difference`, `flip_null`, `random_null`, orthonormal `hold_basis`), ComfyUI conditioning & `extra_conds` trace, 3 text-path weight families (`fl2va`, `ref2va`, `fasth3`), prior art (`ComfyUI-MAINodes`/`Dials`, `vlens`, `krea2-explorations`), and experimental design |
| [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) | Copy-pasteable system prompt for generating H3 `t2va`/`fl2va`/`ref2va` scene prompts and 10-pair contrastive `.toml` sets, plus diagnosis of physical-action vectors and workflow instructions |
| [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md) | **40 rendered Pika H3 Flex (`6`-step `lcm`) and `ref2va` arms across 4 waves**: (1) the `film_16mm_1974` gate ladder (`01..08`), (2) joint audio+video steering & multi-vector combinations (`09..17`), (3) `ref2va` character-reference steering with `mask="non_vision"` (`18..24`), and (4) `db821224` physical transformations (`zero_gravity`, `head_swell`), timeline/norm-matching ablations, chained dual-node `t2va` vectors (`37..38`), and `from_first_diff` pooling (`25..40`) |

---

## 9. Quick CLI Reference

Using the ComfyUI venv Python (`$PY`) and paths from `../../start.sh`:

```bash
# 1. Run all CPU checks for control vectors and h3_spans:
$PY bench/check_control_vectors.py

# 2. Build a standard scene_text-pooled control vector (.safetensors):
$PY bench/control_vectors.py make \
    --pairs prompt_bank/control_vectors/film_16mm_1974.toml \
    --unet h3_27B_fl2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_fl2va_svd995_bf16.safetensors:1.0 \
    --device default --refiner-device cuda \
    --include-encoder --overwrite

# 3. Build a from_first_diff-pooled control vector for single-clause end-of-section physical actions (2.12x-2.28x higher raw_norm):
$PY bench/control_vectors.py make \
    --pairs prompt_bank/control_vectors/zero_gravity.toml \
    --unet h3_27B_ref2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0 \
    --pooling from_first_diff \
    --name zero_gravity_ffd \
    --device default --refiner-device cuda \
    --include-encoder --overwrite

# 4. Inspect a saved .safetensors control vector's quality report and recipe metadata:
$PY bench/control_vectors.py inspect \
    "$H3_MODEL_STORE/control_vectors/zero_gravity_ffd__ref2va__from_first_diff__lora_865ee189.safetensors"

# 5. Score arbitrary prompts against a saved control vector (projection meter):
$PY bench/control_vectors.py meter \
    --vector "$H3_MODEL_STORE/control_vectors/film_16mm_1974__fl2va__lora_2a8c2b52.safetensors" \
    --unet h3_27B_fl2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_fl2va_svd995_bf16.safetensors:1.0 \
    --prompts prompt_bank/control_vectors/film_16mm_1974.toml
```
