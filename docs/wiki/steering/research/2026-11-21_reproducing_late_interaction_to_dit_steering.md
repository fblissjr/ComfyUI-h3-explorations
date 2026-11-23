# Reproducing *Breaking the Frozen-Anchor Steering Paradox* (Code & LLM Wiki Reference Map)

**Date:** 2026-11-21  
**Companion Monograph (Paper v4):** [`2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md)  
**Benchmark Result Ledgers:** [`bench/results/2026-11-21_multivector_erasure_benchmark.md`](../../../../bench/results/2026-11-21_multivector_erasure_benchmark.md) · [`bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md)  
**Practitioner LLM Wiki:** [`docs/wiki/steering.md`](../../steering.md) · [`how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md) · [`tutorials_progressive_guide.md`](../tutorials_progressive_guide.md)

Every theorem, census, CPU mathematical ablation, controlled GPU ablation, 50-block DiT mechanistic trace, causal block-band intervention, `YIN` $F_0$ vocal stability probe, and vector figure in [`2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md) reproduces from tracked repository scripts, `.toml` pair sets, and generated ComfyUI API workflows — no external dataset bundle is required.

---

## 1. Shell & Environment Bootstrap (Zero `uv`)

Per [`AGENTS.md`](../../../../AGENTS.md) and [`docs/wiki/ops/comfy_notes.md`](../../ops/comfy_notes.md), run every command from the repository root using ComfyUI's `.venv/bin/python` and `h3.env` (`$H3_INPUT_DIR`, `$H3_OUTPUT_DIR`, `$H3_MODEL_STORE`):

```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"
set -a; . ./h3.env; set +a
```

---

## 2. Tier 1: Zero-GPU Algebraic Proofs & Unit Checks (`< 5 s` CPU)

Reproduces **Propositions 1–4**, **Theorem 1** (exact `float64` SVD ridge split `max_abs_err <= 1e-13`), **Proposition 2** (uniform-marginal Sinkhorn OT collapse `cos == 1.000000` vs. asymmetric row-softmax $\tau_{\text{attn}} = 0.15$), **Theorem 2** ($s = 0$ bitwise identity, $s < 0$ exact 1D fallback, $|s| \cdot 67.5$ RMS shift matching, orthogonal `hold_words`, rank-16 `compose_erasure_payload`), Gates A–D, `MIN_ERASURE_PAIRS = 6`, and `build_spatial_vision_erasure`:

```bash
# Exact float64 algebraic proofs (Theorems 1-2 & Propositions 1-2)
CUDA_VISIBLE_DEVICES="" "$PY" bench/verify_pole_a_erasure.py --cpu-only

# Unit & schema suite for control_vectors.py, Gates A-D, erasure.mu_a, and runtime operators
CUDA_VISIBLE_DEVICES="" "$PY" bench/check_control_vectors.py
```

- **Owning modules & symbols:** [`control_vectors.py`](../../../../control_vectors.py) (`build_pole_a_erasure`, `build_spatial_vision_erasure`, `compose_erasure_payload`, `orthogonalize_erasure_against_holds`, `apply_post_refiner`, `per_token_erasure_profile`, `lint_pair_set`), [`j_lens.py`](../../../../j_lens.py) (`decode_erasure_payload`, `load_model2vec_lexicon`), [`bench/verify_pole_a_erasure.py`](../../../../bench/verify_pole_a_erasure.py), [`bench/check_control_vectors.py`](../../../../bench/check_control_vectors.py).

---

## 3. Tier 2: Building the Bank, `Model2Vec` Lexicon, & `< 15 ms` CPU Censuses

All `39` contrastive `.toml` pair definitions (`38` original concepts + `de_llm_slop.toml`) live in [`prompt_bank/control_vectors/`](../../../../prompt_bank/control_vectors/) (indexed in [`preset_catalog_and_cli_reference.md`](../preset_catalog_and_cli_reference.md)):

```bash
# 1. Build the 39-concept control-vector bank into $H3_MODEL_STORE/control_vectors/bank/
"$PY" bench/control_vectors.py build --all --families fl2va ref2va
"$PY" bench/control_vectors.py build prompt_bank/control_vectors/noir.toml prompt_bank/control_vectors/de_llm_slop.toml --families h3_image

# 2. Build the 180-term contextual Model2Vec lexicon ($H3_MODEL_STORE/j_lens/model2vec_h3_lexicon.safetensors)
"$PY" bench/control_vector/build_model2vec_lexicon.py

# 3. Run < 15 ms CPU Subspace X-Ray on any built concept (§5.4)
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vectors.py inspect noir

# 4. Run 38-bank spectral, Grassmannian, & J-Lens rescue census (§6.3, Fig. 3; < 5 s CPU)
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vector/eval_multivector_erasure.py --stage census

# 5. Run float64 tau_attn, V_r construction, & pair-count coverage sweeps (§6.2, §6.4, Figs. 2 & 4; < 5 s CPU)
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vector/eval_multivector_erasure.py --stage cpu_math
```

---

## 4. Tier 3: Controlled GPU Benchmark, 50-Block DiT Trace, & Vector Figures

[`bench/control_vector/eval_multivector_erasure.py`](../../../../bench/control_vector/eval_multivector_erasure.py) executes the RMS-matched (`101.25` at $s = 1.5$) GPU ablations, logs metrics to `data/exp_unpooled_sliders/paper_2026_11_21/*.json`, writes contact sheets and `.1fps` companions to `$H3_OUTPUT_DIR/{Image,Video}/paper_erasure_2026-11-21/`, and compiles the deterministic `.svg` plots in [`figures/`](figures/):

```bash
# Ablations 1-5: tau_attn Goldilocks plateau, V_r variants (fl2va + h3_image), rank r & ridge, pair count P, step windows (§6.2, §6.4)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage ablations

# Ablations 6-7: MiniMaxH3WordVector (Model2Vec + against-SVD) & MakeMediaControlVector patch erasure (§5.3, §6.4)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage word_media

# Ablation 8a-8c: 192f/90f 24 fps + 40 Hz AV (Clean SIF, Ward+WARP, audio-stream erasure, stratified bank) (§6.5)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage audio_video

# Ablation 9: Zero-shot transfer to undistilled 50-step Base H3 (loras=[]) (§6.5)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage base50

# 50-Block DiT Mechanistic Modality Trace across blocks.0..49 (§6.6, Fig. 6)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage dit_blocks

# Regenerate vector figures (figures/fig1..fig4_*.svg) on CPU
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vector/eval_multivector_erasure.py --stage figures
```

- **Recorded results:** [`bench/results/2026-11-21_multivector_erasure_benchmark.md`](../../../../bench/results/2026-11-21_multivector_erasure_benchmark.md) and [`2026-11-20_unpooled_multivector_sliders.md`](2026-11-20_unpooled_multivector_sliders.md).

---

## 5. Tier 4: Phase 3 Causal Block-Band K/V Routing, `de_llm_slop`, & Dual-Lock `fl2va`

[`bench/control_vector/run_phase3_steering_experiments.py`](../../../../bench/control_vector/run_phase3_steering_experiments.py) reproduces Stages A–D (metrics in `data/exp_unpooled_sliders/phase3_2026_11_21/*.json`, media in `$H3_OUTPUT_DIR/{Image,Video}/phase3_2026-11-21/`):

```bash
# Stage A: 12-pair de_llm_slop Anti-Slop Master EQ across 4 diverse scenes + 90f speaking AV (§6.4, §7.2.3)
"$PY" bench/control_vector/run_phase3_steering_experiments.py --stage de_slop

# Stage B: Per-pair contested_svd_r8 vs. uncentered global_contested_svd_r8, strength ramp, & 3-slider stack (§6.4, §7.2.3)
"$PY" bench/control_vector/run_phase3_steering_experiments.py --stage exp1

# Stage C: 10-pair pooling="vision_tokens" (erasure.mu_a) on ref2va Selective-Color Lock & Dual-Lock fl2va (§7.2.2)
"$PY" bench/control_vector/run_phase3_steering_experiments.py --stage exp2

# Stage D: Causal Shadow-Text K/V block-band routing across [0..19], [20..49], [30..49], [0..49] (§6.6, Table 12b)
"$PY" bench/control_vector/run_phase3_steering_experiments.py --stage exp3
```

- **Recorded results:** [`bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md).

---

## 6. Tier 5: Objective `YIN` $F_0$ Vocal Pitch Stability (`0` GPU)

[`bench/probe_yin_f0_drift.py`](../../../../bench/probe_yin_f0_drift.py) computes pure-PyTorch `rfft`/`irfft` `YIN` fundamental frequency (`65..600 Hz`) on any rendered baseline vs. steered `-audio.mp4` pair (§6.5):

```bash
CUDA_VISIBLE_DEVICES="" "$PY" bench/probe_yin_f0_drift.py \
  --baseline "$H3_OUTPUT_DIR/Video/paper_erasure_2026-11-21/ab8a_192f_noir_A1_1d-audio.mp4" \
  --steered  "$H3_OUTPUT_DIR/Video/paper_erasure_2026-11-21/ab8a_192f_noir_A2_erasure-audio.mp4"
```

---

## 7. Paper Section -> Code Symbol -> Workflow -> LLM Wiki Map

| Paper Section / Claim | Owning Code & Symbols | Reproducing CLI / API Workflow | Companion LLM Wiki Record |
| :--- | :--- | :--- | :--- |
| **§3 Retrieval–Steering Duality & Propositions 1–4** | `steering/extract.py::build_pole_a_erasure`, `bench/verify_pole_a_erasure.py::check_algebraic_ridge_split` | `bench/verify_pole_a_erasure.py --cpu-only` | [`2026-11-20_multivector_retrieval_sota_to_dit_steering.md`](2026-11-20_multivector_retrieval_sota_to_dit_steering.md) |
| **§4–§5 Theorems 1–2, Algorithms 1–2, &Rank-16 Union (`sculpt`)** | `steering/runtime.py::apply_post_refiner`, `steering/extract.py::compose_erasure_payload`, `steering/nodes.py::MiniMaxH3Steer` | `bench/check_control_vectors.py`, [`workflows/h3_text_to_video_pika_stacked_erasure_api.json`](../../../../workflows/h3_text_to_video_pika_stacked_erasure_api.json) | [`how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md), [`tutorials_progressive_guide.md`](../tutorials_progressive_guide.md) (Labs 1 & 3) |
| **§5.3–§5.4 `Model2Vec` Lexicon, `ColQwen` Media Contrast, & `< 15 ms` X-Ray** | `steering/xray.py::decode_erasure_payload`, `steering/extract.py::per_token_erasure_profile`, `steering/nodes.py::MiniMaxH3WordVector`, `steering/nodes.py::MiniMaxH3MakeMediaControlVector` | `bench/control_vector/build_model2vec_lexicon.py`, `eval_multivector_erasure.py --stage word_media`, [`workflows/h3_text_to_video_pika_word_vector_erasure_api.json`](../../../../workflows/h3_text_to_video_pika_word_vector_erasure_api.json), [`workflows/h3_image_ref_make_media_erasure_pika_api.json`](../../../../workflows/h3_image_ref_make_media_erasure_pika_api.json) | [`tutorials_progressive_guide.md`](../tutorials_progressive_guide.md) (Labs 2 & 4), [`j_lens_diagnostics_and_word_lab.md`](../j_lens_diagnostics_and_word_lab.md) |
| **§6.1–§6.4 $2\times 2$ Synergy, 6-Arm Specificity, `38`-Bank Census, & Hyperparameter Sweeps** | `steering/extract.py::erasure_spectral_summary`, `prompt_bank/control_vectors/*.toml` | `eval_multivector_erasure.py --stage {census,cpu_math,ablations}`, [`workflows/h3_text_to_video_pika_1d_vs_erasure_ab_api.json`](../../../../workflows/h3_text_to_video_pika_1d_vs_erasure_ab_api.json) | [`2026-11-20_unpooled_multivector_sliders.md`](2026-11-20_unpooled_multivector_sliders.md), [`2026-11-21_multivector_erasure_benchmark.md`](../../../../bench/results/2026-11-21_multivector_erasure_benchmark.md) |
| **§6.5 Joint `24 fps` + `40 Hz` AV, `YIN` $F_0$, `MiniMax-H3-Image`, `P-H3-Image`, & Base 50-Step H3** | `bench/probe_yin_f0_drift.py::yin_f0`, `h3_spans.py` (`scene_only`, `soundscape_only`) | `eval_multivector_erasure.py --stage {audio_video,base50}`, [`workflows/image/h3_image_t2i_control_vector_api.json`](../../../../workflows/image/h3_image_t2i_control_vector_api.json), [`workflows/image/h3_p_h3_image_t2i_control_vector_api.json`](../../../../workflows/image/h3_p_h3_image_t2i_control_vector_api.json) | [`docs/wiki/steering.md`](../../steering.md), [`docs/wiki/image/h3_image.md`](../../image/h3_image.md) |
| **§6.6 50-Block DiT Trace & Causal Block-Band Shadow-Text K/V Proof (`blocks.20..49`)** | `bench/control_vector/eval_multivector_erasure.py`, `bench/control_vector/run_phase3_steering_experiments.py` | `eval_multivector_erasure.py --stage dit_blocks`, `run_phase3_steering_experiments.py --stage exp3` | [`2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md) |
| **§7.2 Boundaries: Dual-Lock `fl2va` (`erasure.mu_a` + `cond_video_latents`), `de_llm_slop`, & `>= 30%` Spatial Masks** | `steering/extract.py::build_spatial_vision_erasure`, `conditioning.py::MiniMaxH3Conditioning`, `edit_guidance.py::MiniMaxH3EditGuider` | `run_phase3_steering_experiments.py --stage {de_slop,exp1,exp2}`, [`workflows/h3_spatial_portal_pika_erasure_api.json`](../../../../workflows/h3_spatial_portal_pika_erasure_api.json) | [`docs/wiki/edit_guidance.md`](../../edit_guidance.md), [`2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md) |
