# Control Vectors for MiniMax H3: Code, Prompt Bank, LLM Authoring, and Readout Index

last updated: 2026-11-05

One-stop router and index for activation control vectors in this pack: the runtime nodes and CLI, the token-span partitioner, the 14 contrastive-pair `.toml` sets in `prompt_bank/control_vectors/`, the self-contained LLM system prompt that authors them, the ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` activation-readout and batch-labelling pipeline, and the measured Pika H3 Flex and `ref2va` records.

---

## 1. Where Control Vectors Operate in H3

MiniMax H3 encodes text and multimodal references with stock `Qwen3-VL-32B-Instruct`, taps the hidden state after **`model.language_model.layers.49`** (`50`th transformer block, `[L, 5120]`), projects it via `condition_proj` (`5120 -> 5376`), and passes it through a 2-block bidirectional **`token_refiner`** (`[L, 5376]`, ending in `final_norm`) inside `MiniMaxH3.extra_conds` once per sampling call before the 50-block DiT denoising loop begins.

Our control vectors operate at two points:
1. **`vector.post_refiner` (`[5376]`, primary steering point for video+audio generation)**:
   - Extracted from contrastive prompt pairs `(A_i, B_i)` after `condition_proj` + `token_refiner`, mean-pooled over **`scene_text`** tokens only (excluding vision blocks `tag == 0`, `<Picture i>: ` / `<Video k>: ` / `<Audio j>: ` labels, `<T.T seconds>` timestamps, `<d>...</d>` dialogue spans, and `(S1)` speaker tags via [`h3_spans.py`](../../h3_spans.py)).
   - Applied once per sampling call inside a `DIFFUSION_MODEL` wrapper around `preprocess_text_embeds` ([`control_vectors.py`](../../control_vectors.py)), scaling the unit direction by `strength * median_row_norm` across the selected token mask (`non_vision` or `scene_text`) while leaving vision rows (`tag == 0`) bitwise untouched and sequence length `L` invariant.
2. **`vector.encoder` (`[5120]`, optional pre-projection tap at `layers.49`)**:
   - Mean-pooled over `scene_text` tokens (excluding token index `0`, where Qwen3-VL's attention sink sits at ~`3679` norm vs. ~`191`–`297` median row norm).
   - Used both for pre-refiner vs. post-refiner geometry comparisons and for direct `layers.49 -> layers.50..63 -> lm_head` activation-steered self-report and lexical log-odds readout in `Qwen3-VL-32B-Instruct` ([`docs/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py`](../research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py)).

---

## 2. Code, Nodes, CLI, and Checks

| File | Role | Key Symbols & Entry Points |
|---|---|---|
| [`control_vectors.py`](../../control_vectors.py) | Core math, `.toml` pair loader, `.safetensors` I/O, model family & recipe verification, runtime `DIFFUSION_MODEL` wrapper, and ComfyUI nodes | `MiniMaxH3MakeControlVector`, `MiniMaxH3ApplyControlVectors`, `load_pairs`, `differing_span` (`MAX_SHARED_INTERIOR_WORDS = 5`), `mean_difference`, `flip_null`, `random_null`, `hold_basis`, `project_out`, `apply_post_refiner` (`add`, `subtract`, `amplify`, `project_out`), `extract_model_recipe`, `save_control_vector`, `load_control_vector` |
| [`h3_spans.py`](../../h3_spans.py) | Token-span partitioner for H3 conditioning sequences (`vision`, `label`, `timestamp`, `dialogue`, `speaker`, `scene`) | `spans`, `pooling_mask` (`scene_text`, `all_tag1`, `from_first_diff`), `expand_token_ids`, `first_differing_index`, `verify_tokenizer_constants` |
| [`bench/control_vectors.py`](../../bench/control_vectors.py) | Standalone CLI for building, inspecting, metering, and building gate-ladder workflows for control vectors | Subcommands: `make` (`--pairs`, `--unet`, `--lora`, `--pooling`, `--include-encoder`, `--overwrite`), `inspect`, `meter`, `gate` |
| [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py) | Fast CPU guard covering `[C1]`–`[C7]`: tokenizer constants, span partitioning, `.toml` single-clause diff extraction & refusals, `float64` math & `flip_null`, `.safetensors` round-trip & recipe guards, `apply_post_refiner` bitwise vision preservation, and runtime wrapper | `check_h3_spans_constants`, `check_h3_spans_partition_and_refusals`, `check_pair_loading_and_refusals`, `check_math_and_synthetic_statistics`, `check_safetensors_and_recipe_guards`, `check_apply_post_refiner_and_vision_invariance`, `check_runtime_wrapper_and_tracked_tomls` |
| [`workflows/build_workflows.py`](../../workflows/build_workflows.py) | Workflow generator helper for wiring `MiniMaxH3ApplyControlVectors` into `t2va`, `fl2va`, and `r2v` (`ref2va`) API graphs | `build_control_vector_arm` (`mode`, `vector_name`, `strength`, `mask`, `start_percent`, `end_percent`, `apply_holds`, `lora`, `lora_branch`, `ref_images`) |

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

## 6. Research Notes & Measured Records

| Document | What It Covers |
|---|---|
| [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md) | Architecture, mathematical formulation (`mean_difference`, `flip_null`, `random_null`, orthonormal `hold_basis`), ComfyUI conditioning & `extra_conds` trace, 3 text-path weight families (`fl2va`, `ref2va`, `fasth3`), prior art (`ComfyUI-MAINodes`/`Dials`, `vlens`, `krea2-explorations`), and experimental design |
| [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) | Copy-pasteable system prompt for generating H3 `t2va`/`fl2va`/`ref2va` scene prompts and 10-pair contrastive `.toml` sets |
| [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md) | 24 rendered Pika H3 Flex (`6`-step `lcm`) and `ref2va` arms across 3 waves: the `film_16mm_1974` gate ladder (`s=0, +1, +2, +3, -2`, `flip_null`, `random_null`, explicit prompt), joint audio+video steering (`felt_muppet`, `cursed_cctv`, `infomercial_fail`, `vocal_strain`, `fever_melt`, `chaos_panic`, plus stacked multi-vector combinations), and `ref2va` character-reference steering with `mask="non_vision"` |

---

## 7. Quick CLI Reference

Using the ComfyUI venv Python (`$PY`) and paths from `../../start.sh`:

```bash
# 1. Run all CPU checks for control vectors and h3_spans:
$PY bench/check_control_vectors.py

# 2. Build a control vector (.safetensors) with both post_refiner [5376] and encoder [5120] directions:
$PY bench/control_vectors.py make \
    --pairs prompt_bank/control_vectors/zero_gravity.toml \
    --unet h3_27B_ref2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_ref2va_comfy_svd995_bf16.safetensors:1.0 \
    --device default --refiner-device cuda \
    --include-encoder --overwrite

# 3. Inspect a saved .safetensors control vector's quality report and recipe metadata:
$PY bench/control_vectors.py inspect \
    "$H3_MODEL_STORE/control_vectors/zero_gravity__ref2va__lora_865ee189.safetensors"

# 4. Score arbitrary prompts against a saved control vector (projection meter):
$PY bench/control_vectors.py meter \
    --vector "$H3_MODEL_STORE/control_vectors/film_16mm_1974__fl2va__lora_82f01909.safetensors" \
    --unet h3_27B_fl2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_t2va_comfy_svd995_bf16.safetensors:1.0 \
    --prompts prompt_bank/control_vectors/film_16mm_1974.toml
```
