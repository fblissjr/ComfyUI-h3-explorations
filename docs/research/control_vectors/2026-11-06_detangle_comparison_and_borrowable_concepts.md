# DeTangle (`matlowai/DeTangle`) Compared Against H3 Control Vectors: What Transfers, What Does Not, and What We Should Change

**Date**: 2026-11-06  
**Reference credit**: [`DeTangle`](https://github.com/matlowai/DeTangle) by Dylan Matlow (MatlowAI), inspected in `coderef/DeTangle` at commit `9945346` (`2026-10-21`).  
**Related docs**: [`docs/wiki/control_vectors.md`](../../wiki/control_vectors.md), [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md), [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](2026-11-05_control_vector_system_prompt.md), [`docs/research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md), [`docs/wiki/references.md`](../../wiki/references.md).

---

## 1. Executive Summary & Architectural Comparison

**DeTangle** ([`matlowai/DeTangle`](https://github.com/matlowai/DeTangle), `coderef/DeTangle/README.md:1-60`) is an open-source toolkit and case-study suite by Dylan Matlow (MatlowAI) for **post-hoc conditional least-squares surgery on LoRA weights ("delta surgery")** and **paired baseline-subtracted evaluation (`DeltaScope`)**. Its primary motivating problem is **generic class bleed in identity/concept LoRAs**: a character LoRA trained with trigger token `rlsa` leaks the trained person's face onto generic prompts like `"a portrait of a woman"` even when the trigger token is omitted (`coderef/DeTangle/docs/mechanisms.md:3-12`).

While both `DeTangle` and our H3 control vectors (`control_vectors.py`, `control_vector_nodes.py`, `h3_spans.py`, `bench/control_vectors.py`) use **training-free `float64` linear algebra on captured paired activations** to isolate a desired concept while suppressing collateral bleed, they operate on **different mathematical objects at different stages of the model**:

| Axis | `DeTangle` (`matlowai/DeTangle`, `9945346`) | H3 Control Vectors (`ComfyUI-h3-explorations`) |
|---|---|---|
| **Primary artifact** | **Edited LoRA checkpoint** (`.safetensors` with refitted `lora_A.weight` matrices across all adapted linear layers; `lora_B.weight` frozen) (`coderef/DeTangle/src/delta_surgery_seed/cli.py:55-114`) | **Single 5376-d steering vector** (`~24 KB` `.safetensors` storing `vector.post_refiner` in `float32[5376]`) (`control_vectors.py:1842-1887`) |
| **Where it acts in the network** | Multiplicative low-rank weight update $\Delta(\mathbf{x}) = B A' \mathbf{x}$ inside every adapted DiT/UNet linear layer across diffusion timesteps (`coderef/DeTangle/docs/mechanisms.md:5-11`) | Additive conditioning shift $\mathbf{r}' = \mathbf{r} + s \mathbf{v}_{\text{PR}}$ on the 5376-d post-`token_refiner` text stream before the 50 DiT blocks run (`control_vector_nodes.py:1025-1114`, `control_vectors.py:2020-2169`) |
| **Contrastive inputs** | Two activation reservoirs captured over 72 paired prompts $\times$ 6 flow timesteps: **Class** $X_c$ (generic `"woman..."`) vs. **Trigger** $X_t$ (`"rlsa, woman..."`) (`coderef/DeTangle/docs/case-study.md:40-43`) | 10 stratified contrastive scene pairs $(A_i, B_i)$ differing in a single contiguous clause (`diff_span`), plus optional `[[hold]]` nuisance pairs (`control_vectors.py:942-1121`) |
| **How the target is extracted** | Refits $A'$ via ridge regression to preserve $A X_t^\top$ while zeroing $A' X_c^\top$: $\min_{A'} \lVert A' X_c^\top\rVert_F^2 + \mu \lVert(A' - A) X_t^\top\rVert_F^2 + \epsilon \lVert A' - A_0\rVert_F^2$ (`coderef/DeTangle/src/delta_surgery_seed/core.py:65-108`) | Computes pair-by-pair difference $\mathbf{v}_{\text{raw}} = \frac{1}{N}\sum_{i=1}^N w_i(\bar{\mathbf{r}}_{B,i} - \bar{\mathbf{r}}_{A,i})$ over the differing BPE token span (`diff_span`) (`control_vectors.py:942-970`) |
| **How collateral bleed is suppressed** | Soft Gram penalty $\lVert A' X_c^\top\rVert_F^2$ balanced against $\mu \lVert(A' - A)X_t^\top\rVert_F^2$ (`coderef/DeTangle/docs/mechanisms.md:18-36`) | (1) Hard Gram-Schmidt projection $\mathbf{v}_\perp = (I - U_{\text{col}} U_{\text{col}}^\top)\mathbf{v}_{\text{raw}}$ against `[[hold]]` basis $U$ (`control_vectors.py:1093-1121`), and (2) token-tag masks (`non_vision`, `scene_only`, `after_dialogue`) (`h3_spans.py:350-382`, `control_vector_nodes.py:1001-1022`) |
| **How pre-render quality is graded** | In-sample & held-out Frobenius response retention $\lVert B A' X^\top\rVert_F / \lVert B A X^\top\rVert_F$ and cosine on $X_c$ vs. $X_t$ (`coderef/DeTangle/src/delta_surgery_seed/core.py:111-130`) | `mean_pairwise_cosine`, `per_pair_cosine` (leave-one-pair-out), `split_half_cosine`, `channel_census` (`top1_energy_share`), and `flip_null` / `random_null` controls (`control_vectors.py:973-1042, 1144-1234, 2200-2221`) |
| **How rendered outputs are graded** | **Paired two-number score** on every seed: `triggered_likeness` (target effect) alongside `generic_excess = generic_likeness(adapter) - generic_likeness(matched_base)` (`coderef/DeTangle/docs/case-study.md:9-36`) | Seed-matched `strength = 0.0` vs. `strength > 0` Pika `t2va` and `ref2va` render sweeps + ExLlamaV3 `Qwen3-VL-32B` `layers.49` readout (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`) |

An audit of `DeTangle` against our on-disk `.toml` banks and `.safetensors` vectors leads to three conclusions:
1. **Three concepts from `DeTangle` are viable and practical** when adapted to H3: soft ridge hold attenuation ($\alpha \in [0, 2]$), paired two-number evaluation against a seed-matched `strength = 0.0` base, and side-by-side lockstep comparison in `bench/control_vectors.py ui` (Section 2).
2. **Given the structural difference between a black-box LoRA and H3's structured conditioning, we should support composable inference-time holds and section-aware masks in H3**: where `30/32` shipped `.toml` banks have `0` holds, we allow orthogonalizing at **inference time** against any of our 32 already-built 10-pair `.toml` vectors (or inline `hold_words`) via `mode = "hold"` with a tunable $\alpha \in [0, 2]$, and use **section-aware token masks** (`description_only` vs. `soundscape_only` in `h3_spans.py::inference_mask`) to separate visual and audio steering before reaching for subspace projection (Section 3).
3. **Two `DeTangle` mechanisms must NOT be borrowed**: full-token baseline covariance whitening $(X_A^\top X_A + \epsilon I)^{-1}\mathbf{v}_{\text{raw}}$ in 5376-d space and pre-render cosine against an unsteered `Test Scene Prompt` (Section 4).

---

## 2. Tier 1: Viable Concepts from `DeTangle` (With On-Disk Data & Code Coupling Audit)

### 2.1 Soft Ridge Hold Attenuation ($\alpha \in [0, 1]$ vs. Hard `project_out`)

- **Source in `DeTangle`**: `coderef/DeTangle/docs/mechanisms.md:18-36` and `coderef/DeTangle/src/delta_surgery_seed/core.py:65-108` (`solve`).
- **Current H3 code**: `control_vectors.py::hold_basis` (`control_vectors.py:1045-1090`) and `control_vectors.py::project_out` (`control_vectors.py:1093-1121`).

#### What the on-disk vectors show about hard Gram-Schmidt ($\alpha = 1.0$)
When a `.toml` bank defines `[[hold]]` nuisance pairs, `control_vectors.py::hold_basis` constructs an orthonormal basis $U = [\mathbf{u}_1, \dots, \mathbf{u}_K]^\top \in \mathbb{R}^{K \times 5376}$ ($K \le 6$) via two-pass modified Gram-Schmidt in `float64` (`control_vectors.py:1079-1088`), and `control_vectors.py::project_out` (`control_vectors.py:1119-1120`) applies hard orthogonal elimination:

$$\mathbf{v}_{\perp} = \mathbf{v} - \sum_{k=1}^{K} (\mathbf{v} \cdot \mathbf{u}_k) \mathbf{u}_k = (I_d - U_{\text{col}} U_{\text{col}}^\top)\mathbf{v}$$

Measuring the built `fl2va` `diff_span` vectors on disk reveals how much signal hard projection removes when a hold axis is partially correlated with the target effect:
- **`head_swell` (`prompt_bank/control_vectors/head_swell.toml:130-154`)**: Defines 1 hold group (`pressurized_steam_acoustics`, 2 hold pairs). The unprojected `diff_span` vector (`apply_holds=False`, `n_holds = 0`) has `raw_norm = 85.0101`, and its cosine with `holds.post_refiner` (`[1, 5376]`) is **$\cos(\mathbf{v}_{\text{raw}}, \mathbf{u}_{\text{steam}}) = +0.4070$**. Hard Gram-Schmidt ($\alpha = 1.0$) removes $1 - \sqrt{1 - 0.4070^2} = 0.0866$ (`8.66%`) of the vector's L2 norm (`85.01 -> 76.04`).
- **`film_16mm_1974` (`prompt_bank/control_vectors/film_16mm_1974.toml:178-254`)**: Defines 2 hold groups (`warm_colour_cast` and `soft_focus`, 6 hold pairs total, originally written for the E2.4 sub-attribute ablation). Because `film_16mm_1974`'s own `b_side` includes *"faded warm colour, and soft blooming highlights"*, the unprojected vector (`apply_holds=False`, `raw_norm = 63.8924`) has cosines **$+0.2949$** and **$+0.1489$** with the two hold rows, so hard projection drops `raw_norm` by **`15.38%`** (`63.89 -> 54.07`)—which is why `05_film16mm_s2p71_double` (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md:34`) had to be built with `apply_holds=False`.
- **Cross-vector correlations across the other 30 `.toml` banks on disk**: Even on banks without `[[hold]]` pairs, `chaos_panic` has $\cos(\mathbf{v}_{\text{panic}}, \mathbf{v}_{\text{strain}}) = +0.6539$ (`vocal_strain`) and $\cos(\mathbf{v}_{\text{panic}}, \mathbf{v}_{\text{handheld}}) = +0.5752$ (`handheld_documentary_zoom`); `elastic_neck_stretch` has $\cos = +0.3562$ with `vocal_strain`; `felt_muppet` has $\cos = +0.3295$ with `vocal_strain`.

#### Woodbury reduction of `DeTangle`'s ridge objective on an orthonormal hold basis
Replacing hard projection with `DeTangle`'s ridge penalty (`coderef/DeTangle/docs/mechanisms.md:18-26`) on an orthonormal hold basis $U_{\text{col}} = [\mathbf{u}_1, \dots, \mathbf{u}_K] \in \mathbb{R}^{d \times K}$ ($U_{\text{col}}^\top U_{\text{col}} = I_K$):

$$\min_{\mathbf{v}'} \quad \lVert\mathbf{v}' - \mathbf{v}\rVert_2^2 + \lambda \lVert U_{\text{col}}^\top \mathbf{v}'\rVert_2^2 \quad \Longrightarrow \quad (I_d + \lambda U_{\text{col}} U_{\text{col}}^\top)\mathbf{v}' = \mathbf{v}$$

By the Woodbury identity, the $5376 \times 5376$ inverse collapses to a single scalar attenuation factor $\alpha = \frac{\lambda}{1 + \lambda} \in [0, 1]$:

$$\mathbf{v}' = (I_d + \lambda U_{\text{col}} U_{\text{col}}^\top)^{-1}\mathbf{v} = \mathbf{v} - \frac{\lambda}{1 + \lambda} U_{\text{col}} U_{\text{col}}^\top \mathbf{v} = \mathbf{v} - \alpha \sum_{k=1}^{K} (\mathbf{v} \cdot \mathbf{u}_k) \mathbf{u}_k$$

#### Coupled call sites and schema invariants (`file:line`)
If `alpha < 1.0` is applied at build time, six code sites read or validate the resulting payload and must move together:
1. `control_vectors.py::project_out` (`control_vectors.py:1093-1121`): accepts `alpha: float = 1.0` (`0.0 <= alpha <= 1.0`).
2. `control_vectors.py::_validate_vector_payload` (`control_vectors.py:1587-1595`, guarded by `bench/check_control_vectors.py:1412-1416`): currently refuses with `ValueError("tensor 'vector.post_refiner' is not orthogonal to 'holds.post_refiner'")` whenever `|(holds.post_refiner @ vector.post_refiner).max()| > 1e-4 * max(1.0, v_norm)`. Allowing `alpha < 1.0` at build time requires recording `hold_alpha` in `metadata["pairs"]` and relaxing `_validate_vector_payload` when `hold_alpha < 1.0` (or keeping `vector.post_refiner` unprojected and applying `alpha` at inference time; see Section 3.1).
3. `control_vectors.py::default_vector_filename` (`control_vectors.py:1598-1631`): currently distinguishes only `apply_holds=True` vs. `apply_holds=False` (`__noholds`).
4. `control_vectors.py::format_vector_sidecar_md` (`control_vectors.py:1735-1774`): should record `pre_hold_raw_norm` and per-hold cosines $\cos(\mathbf{v}_{\text{raw}}, \mathbf{u}_k)$ so the `.md` sidecar shows how much energy each hold group removed.
5. Builders in `control_vector_nodes.py::make_control_vector_from_pairs` (`control_vector_nodes.py:814-826`) and `bench/control_vectors.py` (`cmd_make` at `bench/control_vectors.py:356-367`, `cmd_build` at `bench/control_vectors.py:1815-1826`).

---

### 2.2 Paired Two-Number Render Scoring (`Target Effect` + `Collateral Drift` vs. Seed-Matched `strength = 0.0` Base)

- **Source in `DeTangle`**: `coderef/DeTangle/docs/case-study.md:9-36` and `coderef/DeTangle/lab/README.md:28-51`.
- **Current H3 data & tools**: 66 rendered Pika control-vector clips with seed-matched `strength = 0.0` baselines (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`), `bench/measure_dialogue_transcription.py`, `bench/voice_spans.py`, `bench/measure_pair_alignment.py`, `subject_track.py`, and `docs/research/control_vectors/exl3_qwen3vl/runbook.md`.

#### The "Space Station Trap" (`coderef/DeTangle/docs/case-study.md:13-24, 48`)
In `DeTangle`'s case study, a repaired LoRA achieved near-zero generic bleed (`generic_excess = -0.019`) on a space-station portrait—which looks like a complete win if you only measure collateral suppression—but simultaneously collapsed on the target effect (`triggered_likeness = 0.134` vs. `0.667` naive) (`coderef/DeTangle/docs/case-study.md:15`). Because of this, `DeTangle` reports every arm as a **paired two-number cell**:

$$\text{Score}(s, \text{seed}) = \bigl(\text{Target Effect}(s, \text{seed}), \text{Collateral Drift vs. Matched Base}(s, \text{seed})\bigr)$$

#### How to implement this in H3 using tools already in the repo (zero new dependencies)
Unlike `DeTangle`, which uses external ONNX InsightFace (`buffalo_l` / ArcFace) for face scoring (`coderef/DeTangle/docs/case-study.md:28-35`, `coderef/DeTangle/lab/README.md:56-68`—neither `insightface` nor `onnxruntime` is installed in our `.venv`, by the rule in `docs/wiki/who_is_who_evidence.md:245-249`), this repo already has the tools needed for paired scoring on seed-matched `strength = 0.0` vs. `strength = s` clips:

1. **Target Effect Score**:
   - Scored on the rendered video frames and audio via `label_h3_control_vectors.py` (`docs/research/control_vectors/exl3_qwen3vl/runbook.md:35`).
2. **Voice Collateral Drift Score (on `<d>[Language] ...</d>` spans)**:
   - **Word intelligibility**: `bench/measure_dialogue_transcription.py` (`bench/measure_dialogue_transcription.py:1-120`) runs local `openai/whisper-medium` against the `<d>` lines embedded in the clip's workflow metadata (`script_lines`, `bench/measure_dialogue_transcription.py:81-89`), catching word drops or unscripted hallucinations (such as `chaos_panic` arm `38`).
   - **Pitch & timbre stability**: Because Whisper WER alone stays `0.0` even when voice pitch shifts dramatically (such as `head_swell` arm `30`, where `<d>Hi, how are you?</d>` remained intelligible while median F0 jumped `+67%` from `131.2 Hz -> 219.2 Hz`, `bench/results/2026-11-05_control_vectors_pika_and_ref2va.md:80, 108`), combine WER with voiced-span F0 / log-mel cosine against the `strength = 0.0` baseline using `bench/voice_spans.py` (`bench/voice_spans.py:81-100`).
3. **Visual / Subject Collateral Drift Score (`t2va` take alignment & `ref2va` subject stability)**:
   - `bench/measure_pair_alignment.py` (`bench/measure_pair_alignment.py:1-98`) already measures per-frame luma SSIM (`SAME_TAKE_MIN_LUMA_SSIM = 0.70`) and 20 ms audio RMS envelope Pearson correlation (`SOUND_ALIGNED_MIN_ENVELOPE_R = 0.90`) between two clips of the same seed. *(Note: `bench/measure_pair_alignment.py:115-117` expects the silent `.mp4` path and resolves its `-audio.mp4` sibling.)*
   - In H3 `ref2va`, `<Subject 1>` identity is anchored by both `tag == 0` vision tokens (`mask = "non_vision"`) and 3D Video VAE `ref_latents` (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md:65-70`), and subject likeness can be checked with SAM 3.1 mask-pooled head/top-third trunk features (`subject_track.py:60-82`) without any external face-recognition dependency.

---

### 2.3 Side-by-Side Lockstep Comparison View in `bench/control_vectors.py ui`

- **Source in `DeTangle`**: `DeltaScope` (`coderef/DeTangle/lab/README.md:18-21`).
- **Current H3 code**: `bench/control_vectors.py::cmd_ui` (`bench/control_vectors.py:2401-2451`), `bench/control_vectors.py::handle_ui_request` (`bench/control_vectors.py:2210-2399`), and `bench/control_vector_ui.html`.

`bench/control_vectors.py::_Handler._dispatch` (`bench/control_vectors.py:2413-2425`) currently serves only UTF-8 JSON/HTML responses and intentionally has no binary `HTTP 206 Partial Content` (`Range`) video streamer or access to `H3_OUTPUT_DIR`. The cleanest, zero-friction way to add a `DeltaScope`-style side-by-side comparison tab to `bench/control_vector_ui.html` is **pure client-side HTML5 file loading** (`<input type="file" accept="video/mp4">` / drag-and-drop bound via `URL.createObjectURL(file)` to two synchronized `<video>` elements with locked `play`, `pause`, `currentTime`, and `±1/24s` frame-step controls). This requires zero backend changes in `bench/control_vectors.py`, supports instant native browser byte-seeking on `blob:` URLs, and never transmits local filesystem paths over HTTP.

---

## 3. What We Should Change About Our Approach (And Why H3 Is Different from `DeTangle`)

Auditing our data on disk reveals a critical gap: **only 2 out of 32 shipped `.toml` banks** (`prompt_bank/control_vectors/film_16mm_1974.toml` and `prompt_bank/control_vectors/head_swell.toml`) define `[[hold]]` pairs today; **30 of 32 `.toml` banks have `0` holds**.

Why did `DeTangle` do offline matrix surgery, and why should H3 take a different approach?
- **`DeTangle`'s constraint**: A LoRA (`delta(x) = B A x`) modifies weights inside 50+ DiT layers across all timesteps and tokens. At inference time, a LoRA cannot tell which tokens belong to a reference image, which are visual scene descriptions, which are `<d>...</d>` dialogue, or which are `overall_soundscape:`. Therefore, `DeTangle` *had* to bake suppression into the $A$ matrix offline.
- **Our advantage in H3**: We steer the structured conditioning sequence right after `token_refiner` (where exact BPE token roles and section boundaries are known), and we already have **32 pre-built 10-pair `.toml` control vectors (`~24 KB` each, `64` files across `fl2va` + `ref2va`, `90` total on disk)**.

### Change 1: Composable Inference-Time Cross-Vector Orthogonalization (`mode = "hold"` & `hold_words`)
1. **Why bespoke per-`.toml` `[[hold]]` authoring alone does not scale**:
   - Writing custom `[[hold]]` pairs inside every `.toml` file does not scale (`30/32` banks have `0` holds, and 2-pair single-group holds like `head_swell`'s have high variance and correlate at `0.407–0.569` with visual/wind vectors).
   - Meanwhile, `control_vectors.py::apply_post_refiner(..., mode="project_out")` (`control_vectors.py:2172-2335`) projects a vector out of **the prompt's own conditioning rows $\mathbf{r}$** ($\mathbf{r} \leftarrow \mathbf{r} - s \frac{\mathbf{r}\cdot\mathbf{u}}{\lVert\mathbf{u}\rVert^2}\mathbf{u}$), which strips features that the prompt itself asked for, rather than orthogonalizing the *steering vector* $\mathbf{v}_{\text{target}}$.
2. **H3-native inference-time design (`control_vector_nodes.py:2401-2548`)**:
   - At inference time in `MiniMaxH3ApplyControlVectors`, allow projecting out **any existing 10-pair control vector from `models/control_vectors/`** (e.g., `chaos_panic` held out of `cursed_cctv` in `Arm 58`) by chaining a node with `mode = "hold"` (where `strength` acts as $\alpha \in [0, 2]$) or by passing `hold_words` + `hold_alpha` ($\alpha \in [0, 2]$) directly on the node:
     $$\mathbf{v}_{\text{eff}} = \mathbf{v}_{\text{target}} - \alpha \sum_{k=1}^{K} (\mathbf{v}_{\text{target}} \cdot \hat{\mathbf{u}}_k) \hat{\mathbf{u}}_k$$
   - This immediately gives all **32** `.toml` banks (and any compiled vector) access to high-stability 10-pair hold directions (`split_half_cosine = 0.91–0.97`) with zero new `.toml` authoring and `< 0.1 ms` runtime adjustment.

### Change 2: Use Section-Aware Token Masks (`description_only` vs. `soundscape_only`) Before Subspace Projection—Paired with Early-Step Velocity Splitting When DiT Cross-Modal Attention Couples the Streams
- In H3 prompts, visual action lives in `integrated_multimodal_description:` (or `detailed_description:`), spoken lines live in `<d>...</d>`, and environmental audio lives in `overall_soundscape:`.
- Trying to remove acoustic bleed from a visual vector via 5376-d subspace projection (`head_swell` vs. `pressurized_steam_acoustics`, $\cos = 0.407$) throws away global vector energy because `token_refiner`'s 2 bidirectional layers entangle physical tension and sound.
- Because `h3_spans.py::spans` (`h3_spans.py:254-325`) already parses the tokenized prompt at inference time, extending `h3_spans.py::inference_mask` (`h3_spans.py:350-382`, currently `non_vision`, `scene_only`, `after_dialogue`, `all_rows`) with **`description_only`** (steers only non-dialogue visual description tokens, leaving `overall_soundscape:` and `<d>...</d>` untouched) and **`soundscape_only`** (steers only `overall_soundscape:` tokens, leaving visual description and `<d>...</d>` untouched) prevents direct text-token shifts on the non-target section with **zero hold pairs and zero loss of `raw_norm`**.
- **Important architectural boundary (`comfy/ldm/minimax/model.py:591-604, 760-773`)**: Because all 50 H3 `DiTBlock`s run full joint self-attention over the single packed sequence `h = [text, cond, ref_img, ref_audio, audio, video]`, `audio` latent tokens still attend directly to `integrated_multimodal_description:` text rows (as seen in `head_swell` `Arms 30, 36`, which shifted vocal F0 `+67%` despite 100% identical `overall_soundscape:` sections across all 10 pairs). When complete visual-vs-audio isolation is required on Pika's 6-step schedule, combine `description_only` / `soundscape_only` token masks with an early-step (`steps 0..1`) velocity split (`video_scale`, `audio_scale`) on `MiniMaxH3Model._forward`'s `[-video_out, -audio_out]` output (`comfy/ldm/minimax/model.py:792`; see [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md) §3.2).

---

## 4. Tier 2: Concepts That Do Not Transfer (And Why They Are Mathematical Traps Here)

### 4.1 Trap 1: Full-Token Baseline Covariance Whitening $(X_A^\top X_A + \epsilon I)^{-1} \mathbf{v}_{\text{raw}}$

- **Source in `DeTangle`**: `coderef/DeTangle/src/delta_surgery_seed/core.py:65-108` (`solve`) and `coderef/DeTangle/docs/mechanisms.md:18-36`.

Multiplying $\mathbf{v}_{\text{raw}}$ by $(X_A^\top X_A + \epsilon I)^{-1}$ using all tokens from `Prompt A` would degrade H3 control vectors for three independent reasons:
1. **Multiplicative operator ($B A \mathbf{x}$) vs. additive vector ($\mathbf{r} + s \mathbf{v}_{\text{PR}}$)**:
   - `DeTangle` solves for a matrix $A' \in \mathbb{R}^{r \times d}$ that **multiplies** each input token $\mathbf{x}$ at inference time so $A' \mathbf{x}_c \approx \mathbf{0}$ on generic tokens and $A' \mathbf{x}_t \approx A \mathbf{x}_t$ on triggered tokens (`coderef/DeTangle/docs/mechanisms.md:18-29`).
   - An H3 control vector $\mathbf{v}_{\text{PR}} \in \mathbb{R}^{5376}$ is **added** ($\mathbf{r} + s \mathbf{v}_{\text{PR}}$), not multiplied with $\mathbf{r}$. Pre-multiplying a constant vector by $(X_A^\top X_A + \epsilon I)^{-1}$ cannot make an additive shift turn on or off per token; `h3_spans.py::inference_mask` (`h3_spans.py:350-382`) and `control_vector_nodes.py::_resolve_custom_inference_mask` (`control_vector_nodes.py:1001-1022`) already handle token selectivity directly.
2. **Within-pair subtraction already cancels shared `Prompt A` context**:
   - Under `--pooling diff_span` (`control_vectors.py:942-970`), each pair's difference $\mathbf{d}_i = \bar{\mathbf{r}}_{B,i} - \bar{\mathbf{r}}_{A,i}$ already subtracts $\bar{\mathbf{r}}_{A,i}$ within pair $i$.
3. **Rank deficiency in $\mathbb{R}^{5376}$ amplifies off-manifold noise by $1/\epsilon$**:
   - With $d = 5376$ (`control_vectors.py:32`) and only $N = 10$ pairs ($\le 1,200$ total `Prompt A` tokens), $X_A^\top X_A \in \mathbb{R}^{5376 \times 5376}$ has rank $\le 1,200$ and a **$> 4,100$-dimensional null space**.
   - Applying $(X_A^\top X_A + \epsilon I)^{-1}$ shrinks components inside the natural-language subspace of `token_refiner` by $\frac{1}{\sigma_j^2 + \epsilon}$ while multiplying residual noise in the $> 4,100$ unmeasured dimensions by $\frac{1}{\epsilon}$, rotating $\mathbf{v}_{\text{PR}}$ off the `token_refiner` manifold toward `control_vectors.py::random_null` (`control_vectors.py:1026-1042`).

---

### 4.2 Trap 2: Pre-Render Cosine of $\mathbf{v}_{\text{PR}}$ Against an Unsteered `Test Scene Prompt`

- **Source in `DeTangle`**: `coderef/DeTangle/src/delta_surgery_seed/cli.py:117-155` (`evaluate`).

`DeTangle`'s `evaluate` checks held-out activation captures where `--trigger-capture` **contains the trigger word `rlsa` in the prompt** (`coderef/DeTangle/src/delta_surgery_seed/core.py:111-130`). By Principle 1 of `docs/wiki/system_prompts/h3_control_vector_system_prompt.md` (`docs/wiki/system_prompts/h3_control_vector_system_prompt.md:120-123`), an H3 `Test Scene Prompt` is an **unsteered neutral scene** with zero Pole $B$ tokens, so its post-refiner tokens $\mathbf{r}_{\text{test}}$ have near-zero cosine (`cos ≈ 0`) with $\mathbf{v}_{\text{PR}} = \bar{\mathbf{r}}_B - \bar{\mathbf{r}}_A$ by design. Our existing leave-one-pair-out `control_vectors.py::per_pair_cosine` (`control_vectors.py:1158-1172`) and `control_vectors.py::split_half_cosine` (`control_vectors.py:1175-1214`) inside `control_vectors.py::quality_report` (`control_vectors.py:1217-1234`) are already the mathematically sound held-out activation tests.

---

## 5. Engineering Hygiene Parallels Already Shared by Both Codebases

1. **CPU `float64` linear algebra with `float32`/checkpoint serialization**:
   - `DeTangle` converts all inputs to CPU `float64` (`coderef/DeTangle/src/delta_surgery_seed/core.py:8-15, 65-108`) after finding that CUDA `float32` Cholesky solves (`historical_solve`, `coderef/DeTangle/src/delta_surgery_seed/core.py:17-62`) drift on ill-conditioned Gram matrices (`coderef/DeTangle/docs/mechanisms.md:52-60`).
   - `control_vectors.py` likewise computes all pair differences, Gram-Schmidt bases, projections, and cosines in `float64` (`control_vectors.py:925-970, 1052-1121, 1124-1214`) before saving `float32`.
2. **Cryptographic provenance in `.safetensors` headers & refusal on recipe mismatch**:
   - `DeTangle` records `source_sha256`, `class_capture_sha256`, `trigger_capture_sha256`, and `output_sha256` (`coderef/DeTangle/src/delta_surgery_seed/cli.py:66-106`), and `evaluate` refuses to reuse a fitting capture file (`coderef/DeTangle/src/delta_surgery_seed/cli.py:140-144`).
   - `control_vectors.py` and `control_vector_nodes.py` record SHA256 digests of the checkpoint, encoder, and LoRA recipe (`control_vectors.py::safetensors_recipe_digest` at `control_vectors.py:1351-1397`, `control_vector_nodes.py::extract_model_recipe` at `control_vector_nodes.py:159-351`, `control_vectors.py::save_vector` at `control_vectors.py:1842-1887`), and `control_vectors.py::check_vector_compatibility` (`control_vectors.py:1914-1992`) refuses cross-family or incompatible recipes unless overridden.
3. **Zero fuzzy matching**:
   - `DeTangle` refuses ambiguous layer mappings (`coderef/DeTangle/src/delta_surgery_seed/core.py:133-158`) and publishes files atomically via hard links (`coderef/DeTangle/src/delta_surgery_seed/cli.py:101-114`).
   - `control_vectors.py` enforces exact section and BPE diff-span alignment (`control_vectors.py::differing_span` at `control_vectors.py:140-243`, `h3_spans.py::differing_token_bounds` at `h3_spans.py:530-550`, `control_vectors.py::lint_pair_set` at `control_vectors.py:816-910`).

---

## 6. Summary Table of Borrowable Concepts & H3 Architectural Redesign

| # | Concept / Design Decision | Verdict for H3 Control Vectors | Data Needed & Implementation Path |
|---|---|---|---|
| **1** | **Soft Ridge Hold Attenuation ($\alpha = \frac{\lambda}{1+\lambda} \in [0, 2]$) + Inference-Time Cross-Vector Orthogonalization** (`coderef/DeTangle/docs/mechanisms.md:18-36`) | **Adopt (`mode = "hold"` & `hold_words` at Inference Time)** | **Zero new `.toml` data needed**: allow `MiniMaxH3ApplyControlVectors` to project out any of the 32 built 10-pair `.toml` vectors (`64` files, or `hold_words`) from $\mathbf{v}_{\text{target}}$ with tunable $\alpha \in [0, 2]$ (`Arm 58`). |
| **2** | **Section-Aware Token Masks (`description_only` vs. `soundscape_only`)** (`h3_spans.py:254-382`) | **Adopt (H3-Native Advantage Over LoRA Surgery)** | **Zero new data needed**: extend `h3_spans.py::inference_mask` to mask `integrated_multimodal_description:` separately from `overall_soundscape:`, preventing visual/audio cross-bleed without losing `raw_norm`. |
| **3** | **Paired Two-Number Render Scoring (`Target Effect` + `Collateral Drift` vs. `strength = 0.0` Base)** (`coderef/DeTangle/docs/case-study.md:9-36`) | **Adopt for Render Sweeps** | Uses existing seed-matched `strength = 0.0` clips + `bench/measure_dialogue_transcription.py` (WER), `bench/voice_spans.py` (F0/timbre drift), `bench/measure_pair_alignment.py` (SSIM/audio envelope), `subject_track.py:60-82`, and `label_h3_control_vectors.py`. |
| **4** | **Side-by-Side Lockstep Video Comparison Tab (`DeltaScope` pattern)** (`coderef/DeTangle/lab/README.md:18-21`) | **Adopt (Client-Side `URL.createObjectURL`)** | Uses client-side `<input type="file" accept="video/mp4">` drag-and-drop in `bench/control_vector_ui.html` so no binary `Range` server or server path exposure is needed in `bench/control_vectors.py`. |
| **5** | **Full-Token Baseline Covariance Whitening $(X_A^\top X_A + \epsilon I)^{-1}\mathbf{v}_{\text{raw}}$** (`coderef/DeTangle/src/delta_surgery_seed/core.py:65-108`) | **Reject (Mathematical Trap)** | Rank-$\le 1200$ covariance in $\mathbb{R}^{5376}$ leaves $>4,100$ null dimensions where $1/\epsilon$ amplifies off-manifold noise; within-pair $B - A$ already cancels shared `Prompt A` context. |
| **6** | **Pre-Render Cosine Against an Unsteered `Test Scene Prompt`** (`coderef/DeTangle/src/delta_surgery_seed/cli.py:117-155`) | **Reject (Measures Wrong Quantity)** | Unsteered H3 `Test Scene Prompt` has zero Pole $B$ tokens (`cos ≈ 0` by construction); use `control_vectors.py::per_pair_cosine` (`1158-1172`) and `split_half_cosine` (`1175-1214`) instead. |
