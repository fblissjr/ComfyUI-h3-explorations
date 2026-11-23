# Cross-Domain Synthesis: H3 DiT Steering, `Qwen3.8-27B` (`heylook`), `vgel.me`, Nov 2026 SOTA, & Strategic Audit

**Date:** `2026-11-23`  
**Scope:** Synthesis connecting (1) MiniMax H3 joint audio-video DiT steering (`50`-stem active bank, Wave 2–4 empirical gates, Paper v5), (2) the `heylookitsanllm` `Qwen3.8-27B` / `Swift 1.5` forensic findings from the local Apple Silicon LLM host (external measurements noted explicitly by source), (3) November 2026 SOTA across DiT, LLM, and late-interaction/static retrieval, (4) Theia Vogel's (`vgel.me`) November 12–20, 2026 interpretability & neuron-engineering essays (`making-neurons-by-hand`, `shapes-of-model-words`, `words-all-the-way-down`), and (5) a direct five-question strategic audit of what works, what hits walls, and what to prioritize next.

**Key Primary Anchors on Disk:** [`control_vectors.py`](../../../../control_vectors.py) · [`control_vector_nodes.py`](../../../../control_vector_nodes.py) · [`steer_attention_hooks.py`](../../../../steer_attention_hooks.py) · [`lora_branch.py`](../../../../lora_branch.py) · [`j_lens.py`](../../../../j_lens.py) · [`heylook_client.py`](../../../../heylook_client.py) · [`manuscript.md`](../paper/manuscript.md) · [`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md) · [`2026-11-23_wave3b_behavioral_concepts_overhype_and_takeoff.md`](../../../../bench/results/2026-11-23_wave3b_behavioral_concepts_overhype_and_takeoff.md) · [`2026-11-23_wave4_post_restart_live_verification.md`](../../../../bench/results/2026-11-23_wave4_post_restart_live_verification.md)

---

## Executive Summary (Plain English)

1. **Both projects uncovered the exact same 3-Regime Control Law across DiTs and autoregressive LLMs:**
   - **Regime A (Style / Manifold Replacement — e.g. `noir`, `charcoal_sketch_animated`, `reverb_cathedral`):** You know both what to remove from a static prompt and where you want to go. Prompt-centered subspace clearing (`-P_{V_r}`) plus a target push (`+mu_d`), per-peer `ROS`, and an RMS ceiling works cleanly in one `1x` pass.
   - **Regime B (Bad-Habit Suppression — e.g. `de_llm_slop` in H3, anxious `"Wait..."` self-doubt at `L63` in `Qwen3.8`):** There is **no single Pole-B target** (`ADDS` is pure noise at `Qwen3.8` `L63`, and `de_llm_slop` has `mu_d_positive_words = []` in `j_lens.decode_erasure_payload`, drifting into `document/institutional` nouns). Symmetric centering plus a fixed RMS floor harms already-clean inputs; you need a **one-way noise gate** with `boost = 0` and an upper safety cap only — anchored at dataset `mu_clean` (`ACE` `h_t - mu_clean`) for causal `T = 1` LLMs, or keeping **prompt-local centering (`h_i - h_bar`) + a half-wave `-mu_d` cosine gate** on full multi-token H3 prompts so noun/verb POS offsets do not misfire.
   - **Regime C (Multi-Layer Contextual Policy — e.g. `Pika` 6-step trajectory in H3, `Swift 1.5` RL reasoning in `Qwen3.8`):** Multi-layer contextual gating (`L00..L59` knowing *when* a `"Wait"` check catches a real bug vs. pointless spinning) cannot be collapsed into a single terminal or prelude vector. It belongs in an **unmerged compute-dtype SVD LoRA (`lora_branch.py`) with phase/step gating (`_Window` / `in_thinking == True`)**.
2. **CPU metrics on H3's `50` `fl2va` bank stems verify MiniMax's `L49/64` truncation directly:** Passing H3's `50` concept directions through `Qwen3-VL`'s chopped-off final 14 layers (`J_49_to_63`) and `lm_head` (`M_union`) cuts effective dimensionality more than in half (`effR = 31.48 -> 18.99 -> 14.04`) and nearly doubles cross-concept cosine overlap (`0.2438 -> 0.4416`), matching `Qwen3.8`'s abrupt `L51..L63` token-projection collapse (`effR_D: 622 -> 52.6`).
3. **Why `vgel.me` (`Nov 12–20, 2026`) clicks into place:** `shapes-of-model-words` ("self-steering phrases") explains *why* suppressing 20 doubt pivot words works without moving `</think>`, `making-neurons-by-hand` (`SwiGLU` `gate(off, on)` on `Qwen3.6-27B`) matches Swift 1.5's top deltas (`mlp.gate_proj`, `mlp.up_proj`) and gives the exact nonlinear conditional blueprint for `Regime B`, and H3's `L49 -> lm_head` collapse + Gate 2B VAE Dual-Lock answer `words-all-the-way-down`'s open question on where non-linguistic geometry takes over.

---

## 1. Five-Question Strategic Audit (Zero Sugarcoating)

### Q1. What are we investing time in that has a big wall waiting for us?
1. **Pushing `1x` Prelude control vectors past physical walls Wave 2 just proved:**
   - **Gate 2B (`Path 1C` `ColQwen` `vision_tokens` Dual-Lock):** Empirical null (`hsv_sat = 0.2432 / 0.2485` vs. `ref2va_base` `0.2477 / 0.2549`, [`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md)). Tweaking tag-0 Qwen embeddings cannot override raw 24-ch VAE `reference_latent` pixels inside DiT self-attention.
   - **Gate 2C (`1x Q_vid[mask]`):** Clear fail (`1.047x–1.090x` portal contrast with `K_vid, V_vid` washout and empty-stove layout drift vs. `1.421x` crisp localized charcoal on `2x EditGuider`; `1836.1 MB` peak VRAM vs. `263.8 MB` for `1.07x–1.11x` speed on temporal splits).
   - **CPU `Path 1B` implicit token-cosine routing:** `< 1.1%` LOPO MSE gain (`-0.09%` to `-1.08%`, [`manuscript.md`](../paper/manuscript.md#L147)) because `TokenRefiner` self-attention smears stop-word contexts.
2. **Codebase monolith & multi-repo coordination tax:**
   - The repo now holds **`80` root `.py` files**, **`509` flat `bench/*.py` scripts**, **`135` workflow JSONs**, and **`23` research docs** in `docs/wiki/steering/research/` (`~960 KB`), with **`13,137` LOC across just three steering files** ([`control_vector_nodes.py`](../../../../control_vector_nodes.py) `5,835` LOC, [`control_vectors.py`](../../../../control_vectors.py) `4,221` LOC, [`j_lens.py`](../../../../j_lens.py) `3,081` LOC).
   - Keeping root, nested `app/.git` (`825d27d`), and remote `heylookitsanllm` aligned already causes drift (e.g., Wave 4 confirmed `heylook_client.py` bumped `DEFAULT_HEYLOOK_MODEL` while frozen `app/` still points elsewhere and remote `heylook` returned `502` offline during Wave 4).
3. **Scaling `bank/` past `50` single-look stems:**
   - `50.9%` shared live-action scaffolding (`103.5x` random Grassmann overlap), `stained_glass_cathedral` failing `0/2` visually despite highest CPU `mpc = 0.762`, and scene-dependent partiality on dim prompts (`charcoal_stairwell_192f` `rgb_mae = 39.27` vs. `kitchen` `77.30`) show diminishing returns on adding dozens more static look vectors.
4. **In `heylook` (`Qwen3.8-27B`):** Unconditional `L63`-only activation erasure faces the exact same wall as `NoWait`—by `L63`, `L00..L62` have already run unsteered Stock weights, so `L63` cannot tell *"Wait, I spotted a real off-by-one bug"* apart from *"Wait, let me re-verify 2+2"*. Hard `thinking_budget` guillotines also truncate mid-sentence.

### Q2. What SHOULD we spend our time on instead?
1. **In `heylook` (highest leverage for *both* repos):**
   - Ship the **`<think>`-gated unmerged rank-32 Swift-1.5 LoRA** over 8-bit `Qwen3.8-27B` (per-component FP64 QR-SVD, zero GDN hooks, `s = 1.0` inside `<think>`, `s = 0.0` on visible answers) so H3's `MiniMaxH3PromptExpander`, `MiniMaxH3AutoEditPrompt`, and `/lab` pair generation get `-55%` thinking latency without `thinking_budget` truncation.
   - Run the decisive **3-arm comparison** (`<think>`-gated Rank-32 LoRA vs. `L63`-only `ACE` hook vs. 20-token `NoWait` logit damper, plus an optional `vgel`-style layer-33 conditional SwiGLU sidecar) on `code_bug`, `knights_and_knaves`, and a held-out 30–50 prompt coding/math/tool-call set.
2. **In H3 (`ComfyUI-h3-explorations`):**
   - **Unfreeze `Track App-2.0` (`app/`)** around the verified 3-tool creator pipeline: (`1x` fast `heylook` prompt expansion) -> (`1x` 4D-routed `MiniMaxH3Steer` on verified buses) -> (`2x` `EditGuider` / `SpatialDirector` portals + `ReleaseReferences`).
   - **Implement ONE surgical `Regime B` cleanup mode** only if needed for real utility suppressors: keep H3's **prompt-local centering `h_i - h_bar` + half-wave `-mu_d` cosine gate** (`boost = 0`, upper safety cap only; `1.36–1.77` clean-prompt RMS, `38x–50x` quieter than `67.50`, zero `.safetensors` schema rebuild) rather than dataset `mu_clean` (which fits causal `T = 1` LLMs and isolated `diff_span` clauses at `8.6x–30.5x` damping, but trips across full mixed-POS H3 prompts), and extract `V_r` from paired shifts `D_p = P @ B_p - A_p` rather than unsupervised `A_c` on future Regime-B builds.
   - **Pay down Phase-2 structural cleanup** (`control_vector_nodes.py` / `control_vectors.py` split + `bench/` subpackages per [`2026-11-13_repo_audit_and_atlas/index.md`](../../projects/2026-11-13_repo_audit_and_atlas/index.md#L105-L112)).
   - **Ship Paper v5 as-is and publish the `heylook` Qwen3.8 work separately** instead of over-polishing or merging two distinct architectures into one overclaimed manuscript.

### Q3. What are we building that is total bullshit or smoke and mirrors?
1. **"Retrieval–Steering Duality" / ColBERT mythology:** `control_vectors.py:1333-1358` builds `erasure.vr` **strictly from `a_spans`**, never touching `b_spans` or ColBERT; `48/52` bank stems satisfy `cos(mu_d, v_1d) >= 0.980` (`median = 0.9935`). Paper v4 peer review ([`2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md`](2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md)) already caught this: the real engine was baseline erasure (`-P_{V_r}`) + 4D AV DiT routing, not IR duality.
2. **Treating CPU vector metrics (`mpc`, `j_lens` `b_cov`, Grassmann overlap) as render quality proofs:**
   - `stained_glass_cathedral` had the highest CPU `mpc = 0.762` (`sh = 0.942`) -> **`0/2` fail** on GPU.
   - `Bus-Scoped V_r` (`pc8`/`pc16`) cut CPU Grassmann overlap `7.94x` -> **blew out `uv+reverb` exposure (`mid_lap 43.44 -> 7.95`)** in Gate 2A.
   - `youtube_overhype_influencer` had **`1/12` `b_side` J-Lens `top_k` coverage** -> **crushed GPU Wave 3B** (`3.37x` motion, `+10.88 dB`, `+17.43 st`).
   - CPU metrics are **bug/regression detectors**, not visual emergence predictors.
3. **Current `de_llm_slop` ("Anti-Slop EQ") as shipped:** An ordinary 12-pair documentary vector (`b_side_alignment.coverage = 0.071`, `1/14`) with `document / chronological / official / institutional` drift ([`2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md`](2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md#L98-L108)) that blurs clean scenes `-33%..-39%`, blows highlights `3.58x–6.83x` on `sitcom_kitchen`, injects metal filing cabinets, and drops audio `-13.38 dB` (Phase 3) / `-2.45 dB` (Phase 4). Keep retired/rewritten behind a conditional `Regime B` gate or replace it cleanly.
4. **Multi-agent ledger/index ceremony (`~960 KB` research prose):** Dozens of planning cycles auditing and indexing prose cost far more than running a 30-second GPU smoke.
5. **Exposing `target="vision_tokens"` to creator UI workflows** after Gate 2B falsified it as an empirical null (`hsv_sat = 0.2432` vs. `0.2477` base). Keep `STEER_TARGETS[6] = "vision_tokens"` intact in [`control_vector_nodes.py:93`](../../../../control_vector_nodes.py#L93) (`AGENTS.md` §1.5 combo/schema stability, `bench/check_control_vectors.py`, Wave 4 `/object_info`), while leaving `"vision_tokens"` out of `app/` creator dropdowns during `Track App-2.0` (`app/` already has zero references on disk).

### Q4. What aren't we thinking enough about?
1. **Real end-to-end creator multi-shot output:** Making *one* genuinely awesome 30-second multi-shot clip in our own Studio instead of tuning hyperparameters on single-seed `seed = 730451892` `kitchen`/`stairwell` clips via `hsv_sat` and `laplacian_var`.
2. **Input-conditional nonlinear steering (`vgel`'s SwiGLU `gate(off, on)` / one-way `ReLU`):** The missing middle tier between blunt `193.5 KB` unconditional linear vectors and `300 MB–1 GB` 64-layer LoRAs.
3. **Multi-seed & held-out evaluation discipline:** Almost every H3 gate rests on `seed = 730451892` (`kitchen` / `stairwell`), and `heylook` used `14` micro-eval prompts.
4. **When NOT to steer at all:** Fast local `heylook` (`gemma-4-26B` or Swift-LoRA `Qwen3.8`) rewrites prose in ~1s off-GPU and beats control vectors on every noun, wardrobe, blocking, and lighting specification.
5. **Cross-repo API/capability contracts:** Synchronizing root [`heylook_client.py`](../../../../heylook_client.py), nested `app/`, and Mac `heylookitsanllm` `/v1/models` so client timeouts, `thinking_budget`, and default model IDs don't drift when servers restart.

### Q5. What have we dabbled in, failed to pursue further, and SHOULD?
1. **Embedded Image-in-Video + mid-clip `guide_end_percent = 0.33` ([`2026-11-14_embedded_image_in_video_plan.md`](../../research/image/2026-11-14_embedded_image_in_video_plan.md) Phases 1–2):** Verified once on `clockmaker` (`1008`-row zero-seam mid-clip release); shares the already-loaded `Qwen3-VL-32B` with 3-second `MiniMax-H3-Image` edits and solves exact pose, wardrobe, and `fl2va` color palette control without Dual-Lock fights—yet never tested on a 2nd scene or packaged into Phase 2 workflows.
2. **2D Sprite Trajectory D-Pad + zero-file `MiniMaxH3SoundCues` ([`2026-11-12_sound_cues_dpad_and_033_release_law.md`](2026-11-12_sound_cues_dpad_and_033_release_law.md), [`next_steps.md`](../../next_steps.md#L9-L21)):** Delivered `124/124` tracked `+420 / -444 / -392 px` locomotion and Wave 3B's `+9.39 dB` half-2 takeoff roar, yet remains mostly single-scene (`tortoise` + `moto`), with 2-subject trajectories and SAM 3.1 sprite cutouts still parked.
3. **`MiniMaxH3EditGuider.mask` (`[T, H, W]` SAM 3.1 subject-protected regional video editing, [`next_steps.md`](../../next_steps.md#L12-L20,L45)):** Shipped Nov 14 (`e89e1554`), proved `GrowMask(24)` shell/candle protection, then stalled before `eg_unmasked` SAM tracking and word-gated SAM 3.1 region edits (`F1`).
4. **`audio_freeze` loose-mask (`0.25`) + stem freeze + `AudioRefineMask` + `FrozenVideoCache` ([`h3_audio_freeze.md`](../../core/h3_audio_freeze.md), [`next_steps.md`](../../next_steps.md#L30-L31)):** `1,737` lines of song-loop machinery built, while the high-signal hybrid music-stem + live-foley renders sit unexecuted.
5. **`M3` `Ref2VA` pose-freeze & `< 94 px` face-floor sweep + `attention.py::attach_lse_reference_balance` ([`next_steps.md`](../../next_steps.md#L42)):** Already written; just `4–6` renders away from diagnosing `<Picture 1>` pose-freeze in `ref2va`.

---

## 2. November 2026 SOTA Novelty Map & The 3-Regime Control Law

To keep claims sharp in both repos, separate established 2023–Nov 2026 prior art from genuinely new engineering:

| Domain | Established Nov 2026 Prior Art (Cite, Do Not Claim as New) | Genuinely Net-New Contributions Here |
| :--- | :--- | :--- |
| **DiT Steering & Erasure** | Training-free SVD/subspace erasure (`Safree` ICLR, `CURE` NeurIPS, `IPErase`), OT style sliders (`Linear-AcT`), single-target retain projection (`RASteer`, `arXiv:2610.01969`), video-only DiT steering/unlearning (`LA-LQR`, `STAS`, `MUTE`, `FADE`, `CLEAR`), audio-only DiT steering (`TADA!` ICLR 2026), unmerged compute-dtype LoRAs. | **1) Causal 4D single-stream AV DiT map:** `blocks.0..19` layout lock, `blocks.20..29` visual breakout, `10`-block audio lag (`blocks.31..47`), `block 49` sink.<br>**2) Compiler-safe `1x` query slicing:** `1x Q_aud -> K_txt_base` (`1.003x` latency, `+0` `malloc_graph` breaks) and `1x audio_only + late_30_49` (`rgb_mae = 7.461`), plus falsifying `1x Q_vid[mask]` (`Gate 2C`).<br>**3) Multi-slider compositional algebra & norm cliff:** `50.9%` scaffolding census, runtime per-peer `ROS`, `[0, 1]` clamped union SVD, `95.5 -> 98.5` smooth RMS knee.<br>**4) Multimodal Dual-Lock Law (`Prop 1` + `Gate 2B`).** |
| **Retrieval $\to$ Generation** | `ColBERTv2`, `WARP` (SIGIR 2025), LightOn `PyLate` Ward pooling (`arXiv:2511.00444`), Google `MUVERA`, `ColQwen`, Mixedbread *Latent Terms* (`arXiv:2605.29384`), Minish `Model2Vec` / `POTION`. | **1) Generative translation failure bounds:** Sinkhorn mean-pool collapse (`8.88e-16`), `MUVERA`/appended-bag softmax denominator drift, small-bank sparse atom starvation.<br>**2) Zero-GPU `< 15 ms` Wiener `J-Lens` (`j_lens.py`) + `< 3 ms` `Model2Vec` synthesis** across a **truncated VLM (`L49/64`) + bidirectional `TokenRefiner`** with no `lm_head`. |
| **LLM & Reasoning (`heylook`)** | `QLoRA`, weight-delta SVD LoRA extraction, `LEACE` (NeurIPS 2023), `Abliteration` (Arditi 2024), `ACE` (Marshall et al. 2024), `SEAL` (2025), `NoWait` (`arXiv:2506.08343`, 2025), `EAT` (2025). | **1) Hybrid `3:1` GDN+FA forensics (`Swift 1.5`):** Proving all `48` GDN layers untouched (`0.000000` on `A_log`/`dt_bias`/`norm`) and isolating a rank-32 7-module LoRA.<br>**2) `<think>`-only unmerged LoRA gating** over 8-bit MLX (preserving stock answer precision, APC, and MTP).<br>**3) Mechanistic unification of `NoWait`:** Proving `Swift 1.5` leaves `</think>` flat (`+0.0021`) while collapsing `effR_D` `11x` at `L63` to suppress the top-20 self-doubt pivot words by `-4.0..-4.3` nats (`ADDS = noise`). |

### The Unified 3-Regime Control Law

| Regime | Examples | Geometry Observed | Correct Operator | Why Naive Defaults Break |
| :--- | :--- | :--- | :--- | :--- |
| **Regime A: Attribute Manifold Replacement** | H3 `noir`, `charcoal_sketch_animated`, `reverb_cathedral`, `sudden_takeoff_flight` | Opposing Pole-A baseline inside static prompt + coherent Pole-B destination | Prompt-centered `-P_{V_r,\lambda}` + `+s * mu_d` + `peer_ros` + clamped union + RMS norm cap | `1D` addition alone gets vetoed by static Pole-A tokens inside `H_prelude`. |
| **Regime B: Pure Failure-Mode Suppression** | H3 `de_llm_slop`; LLM `L63` doubt suppression, sycophancy, boilerplate | No single Pole-B target (`L63 ADDS = noise`; `de_llm_slop` `mu_d_positive_words = []`) | **LLM (`T=1`): clean-anchor `ACE` (`h_t - mu_clean`) + one-way `ReLU`; H3 (`scene_only`): prompt-centered `h_i - h_bar` + half-wave `-mu_d` ray gate + paired-`D_p` `V_r`** (both `boost = 0`, upper RMS cap only) | Centering LLMs at `mu_a` perturbs clean math; fixed RMS `67.5` blurs clean H3 scenes `-35%`; dataset `mu_clean` on full H3 prompts trips on noun/verb POS offsets. |
| **Regime C: Contextual Multi-Layer Policy** | H3 `Pika` 6-step distill; LLM `Swift 1.5` RL reasoning (`L00..L59`) | Input-conditional `B(A x)` circuit gating *when* to check vs. proceed | **Unmerged compute-dtype SVD `lora_branch` (`sv_fro` per component) + step/phase gate (`_Window` / `in_thinking`)** | Static readout/prelude vectors cannot see upstream computation (`L00..L59` error detection). |

---

## 3. Empirical Cross-Domain Findings, Peer Catches, & Local CPU Verification

### 3.1 External `heylook` `Qwen3.8-27B` / `Swift 1.5` Findings *(Measured on Mac Host, `2026-11-23`)*
*Note: The figures in §3.1 come directly from the external `heylookitsanllm` session (`2026-11-23` handoff + `L43..L63` log), audited against H3's implementation here.*

1. **4-Bucket Micro-Eval (`14` prompts, `seed=42`, `temp=1.0`, MTP active):**
   - `Stock 8-bit xhigh`: `14/14` pass, `7,494` think tok (`535.3` mean), `983` answer tok, `215.4s` (`~39 tok/s`, MTP `0.596`).
   - `Stock 8-bit medium / low`: `14/14` pass, `5,359 / 4,338` think tok, **but visible answers bloated >2x (`2,310 / 2,063` tok)**, burning back time and still spending `1,205` think tok on `sys_design` at `low`.
   - `Swift 1.5 BF16 xhigh`: `13/14` pass (`code_brackets` missed `{` in `"(["`), **`3,380` think tok (`-54.9%`)**, **`939` answer tok (`-4%`)**.
2. **Exact BF16 Weight-Delta SVD (`W_Swift1.5 - W_Stock`, `851` `language_model` tensors):**
   - Active rank-32 LoRA on 7 standard modules: `mlp.gate_proj` (`0.004987` mean rel Frob), `mlp.up_proj` (`0.004962`), `self_attn.q_proj` (`0.004076`), `k_proj` (`0.003998`, `r32 = 85.3%` energy vs `r64 = 86.4%`), `o_proj` (`0.003199`, `r32 = 78.5%` vs `r64 = 79.2%`), `v_proj` (`0.002898`, `r32 = 78.5%`), `mlp.down_proj` (`0.002618`).
   - **All 48 Gated DeltaNet (`linear_attn`) layers untouched:** `A_log`, `dt_bias`, `linear_attn.norm` literally `0.000000` diff (`144` tensors); GDN projections (`0.00036–0.00058`, `eff_rank ~ 2500–2600`) flat isotropic BF16 rounding noise.
3. **Teacher-Forced `<think>` Logit Diff & `L03..L63` Activation/PCRR Phase Transition (`3,195` tokens, `xhigh` prompt):**
   - Mean `<think>` entropy: `0.5315` (Stock) -> `0.3978` (`Swift 1.5`, `-25.2%`); `</think>` (`id=248069`) boundary logit nearly unchanged (`+0.0021` at paragraph boundaries, `d_logprob_all = -2.5593`).
   - Top-20 suppressed tokens inside `<think>` (`Δ logprob` `-3.99` to `-4.32` nats, ~`55x–75x` suppression): `' actually' (-4.32)`, `' wait' (-4.26)`, `' however' (-4.21)`, `' alternatively' (-4.14)`, `' hold' (-4.14)`, `' possibly' (-4.13)`, `' instead' (-4.11)`, `' perhaps' (-4.10)`, `' nonetheless' (-4.06)`, `' hmm' (-4.06)`, `' nevertheless' (-4.06)`, `' might' (-4.05)`, `' either' (-4.05)`, `' error' (-4.02)`, `' mistake' (-4.02)`, `' rather' (-4.01)`, `' otherwise' (-3.99)`, `' rethink' (-3.99)`, `' different' (-3.99)`, `' doubt' (-3.99)`.
   - **`L03..L63` PCRR spectrum (`L43..L63` terminal phase transition):**

| Layer | Type | `rel_d` | `r1_E%` | `sh_cos` | `effR_A` | `effR_D` | `dR8_E` | `PCRR_R2` | `E_cos` | Sink |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L03..L39` *(range)* | FA/GDN | `0.005–0.020` | `0.062–0.196` | `+0.899..+0.985` | `59–513` | `23–885` | `0.167–0.676` | `0.014–0.044` | `-0.24..+0.22` | `3.4–4.7x` |
| `L43` | FA | `0.021` | `0.049` | `+0.882` | `455.5` | `588.0` | `0.185` | `+0.032` | `-0.056` | `3.0x` |
| `L47` | FA | `0.026` | `0.073` | `+0.892` | `519.5` | `577.4` | `0.172` | `+0.037` | `-0.037` | `2.2x` |
| `L51` | FA | `0.044` | `0.168` | `+0.925` | `520.6` | `622.1` | `0.147` | `+0.067` | `+0.270` | `1.1x` |
| `L55` | FA | `0.051` | `0.152` | `+0.906` | `530.7` | `546.0` | `0.166` | `+0.074` | `+0.278` | `0.8x` |
| `L59` | FA | `0.056` | `0.133` | `+0.880` | `579.3` | `568.4` | `0.166` | `+0.069` | `+0.210` | `0.8x` |
| `L62` | GDN | `0.084` | `0.299` | `+0.945` | `575.5` | `355.1` | `0.277` | `+0.173` | `+0.277` | `1.1x` |
| **`L63`** | **FA** | **`0.187`** | **`0.515`** | **`+0.984`** | **`343.4`** | **`52.6`** | **`0.627`** | **`+0.340`** | **`+0.826`** | **`1.9x`** |

   - **`L63` `lm_head` readout:** `L63 mu_d ERASES` = `' actually' (-16.3), ' maybe' (-16.0), ' wait' (-16.0), ' perhaps' (-15.5), ' possibly' (-15.2), ' instead' (-15.0), ' either' (-15.0), ' however' (-14.9)`; `L63 mu_d ADDS` = pure noise (`'' (+7.4), '和' (+6.9), '-Cds' (+6.7)`).

### 3.2 Four Technical Catches Verified Against H3 Code & Causal Topology
1. **`h - mu_a` anchor inversion bug (fixed):** Without a constant `+mu_d` push, centering causal erasure at `mu_a` (Pole A / hesitation) makes `h - mu_a ≈ 0` *during hesitation* (zeroing out erasure when needed) and `h - mu_a != 0` *during clean math* (perturbing clean tokens). Causal Regime B must anchor at **`mu_clean`** (`h' = h - P_{V_r}(h - mu_clean)` where `V_r` spans `H_bad - mu_clean`), plus a one-way ray gate and upper clamp only.
2. **`_pack_3band_block_diag` ([`lora_branch.py:84-118`](../../../../lora_branch.py#L84-L118)) mismatch:** Hardcodes 3 equal-height bands (`out_dim % 3 == 0`). `Qwen3.8-27B` uses asymmetric GQA (`q_proj = 12,288` vs. `k_proj = v_proj = 1,024`) and 2-band `gate+up`, and MLX keeps all 7 projections as separate `nn.Linear` modules. What transfers 1-to-1 is [`bench/convert_pika_lora.py:48-100`](../../../../bench/convert_pika_lora.py#L48-L100) per-component FP64 QR-SVD truncation so large `q`/`k` norms do not starve `v`.
3. **8-bit vs. BF16 dtype confound in early layers:** Comparing `Stock 8-bit` against `Swift 1.5 BF16` mixes 8-bit quantization residual into small early-layer diffs (`rel_d = 0.005–0.056` in `L03..L59`), inflating `effR_D ~ 600–880`. Re-check `Stock 8-bit` vs. `Stock 8-bit + rank-32 LoRA` to isolate true `B(Ax)` activation rank.
4. **`L63`-only `ACE` vs. `L00..L63` LoRA causal catch:** Only `final_norm` + `lm_head` (plus the 1-layer MTP drafter) sit after `L63`. An `L63`-only hook acts *after* `L00..L62` ran unsteered Stock weights, so it cannot replicate upstream RL error-detection features and behaves like an internal `NoWait` logit damper. Test **3 arms** (`<think>`-gated Rank-32 LoRA vs. `L63`-only `ACE` vs. 20-token `NoWait`) on `code_bug` and `knights_and_knaves` to verify whether `L00..L62` preserves real bug-catching accuracy where `L63`-only fails.

### 3.3 Two Brand-New Local CPU Verification Proofs on H3 Disk (`2026-11-23`)

1. **`L49` vs. `TokenRefiner` vs. `L63` vs. `lm_head` Bank Rank Collapse (`50` `fl2va` bank stems via `j_lens_h3.safetensors`):**
   Measured directly across all `50` active `fl2va` stems in `bank/` (`J_49_to_63` `eff_rank = 1129.9`, `J_ref_fl2va` `eff_rank = 363.7`):

| Representation Space (`N = 50` `fl2va` Bank Stems) | `eff_rank` (max `49`) | `top5_PC` Share | `mean\|cos\|` | `p90\|cos\|` | Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. `Qwen3-VL` `L49` (`vector.encoder`, `5120`-D)** | **`31.48`** | `0.379` | `0.2438` | `0.4244` | High-rank semantic/visual workspace |
| **2. H3 Post-Refiner (`vector.post_refiner`, `5376`-D)** | **`30.04`** (`95.4%` kept) | `0.382` | **`0.2235`** | **`0.4185`** | Bidirectional refiner preserves diversity & *lowers* overlap |
| **3. `Qwen3-VL` `L63` pre-head (`J_49_to_63 @ v_enc`, `5120`-D)** | **`18.99`** (`-40%`) | `0.571` | `0.3423` | `0.6247` | Funnel into token-selection bottleneck |
| **4. `Qwen3-VL` `L63` `lm_head` logits (`M_union @ v_enc`, `27407`-D)** | **`14.04`** (`-55%`) | `0.644` | `0.4416` | `0.7633` | Next-token vocabulary bottleneck compresses concepts together |

   *Repo impact:* Explains empirically why MiniMax truncated `Qwen3-VL` at `L49/64` and lowers priority of [`next_steps.md`](../../next_steps.md#L25) GPU Item 5 (`layers.50..63` chain rollouts) given our `94.0%`-coverage `L49 + TokenRefiner` lexicon.

2. **Regime A vs. Regime B Self-Damping (`diff_span` `.pt` Banks vs. Full-Prompt H3 Split):**
   Testing `cv._erasure_row_shift64` (`s = 1.0`, [`control_vectors.py:3580-3641`](../../../../control_vectors.py#L3580-L3641)) against a clean-anchor ray-gated `ACE` suppressor (`mu_clean = B.mean(0)`, `vr_bad = top-8 SVD(A - mu_clean)`, scalar `clamp(<H - mu_clean, u_ray> / ray_len, 0, 1)` gate, `boost = 0`) on isolated `diff_span` tokens vs. full `91–383`-token `scene_only` prompts:

| `.pt` Bank (`data/exp_unpooled_sliders/banks/`) | Shipped Regime-A `RMS(Bad A / Clean B)` | Ray-Gated `ACE` `RMS(Bad A / Clean B)` | Self-Damping Ratio | `cos` Toward `mu_clean` |
| :--- | :---: | :---: | :---: | :---: |
| `chaos_panic` | `67.5 / 67.5` (`1.00x`) | `60.3 / 6.9` | **`8.7x`** | `+0.705` |
| `look_direct_1990s_flash` | `67.5 / 67.5` (`1.00x`) | `71.0 / 2.3` | **`30.5x`** | `+0.870` |
| `noir` | `67.5 / 67.5` (`1.00x`) | `57.1 / 6.6` | **`8.6x`** | `+0.715` |

   - **Full-Prompt H3 Trap vs. Zero-Rebuild Fix (`covered_market.pt`, `indoor_evening.pt`, `indoor_neutral.pt`):** Dataset-centroid `ACE` (`h - mu_clean`) works cleanly for **causal LLMs (`T = 1` next-token decoding)** and isolated `diff_span` clauses above, **but** across full `383`-token `covered_market.pt` (`scene_only`), subtracting optical-adjective `mu_clean = mean(diff_span_B)` from nouns, verbs, and punctuation injects a part-of-speech offset (`mu_noun - mu_adj`) that falsely fires on **`100.0%` of tokens** (`RMS = 12.2–31.6`, `mean_g = 0.372–0.720`). Keeping **H3 prompt-local centering `h_i - h_bar`** with a half-wave cosine gate along `-mu_d` (`g_i = clip((cos(h_i - h_bar, -mu_d) - 0.08) / 0.22, 0, 1)`, `boost = 0`, upper RMS cap only) drops clean-prompt RMS on shipped `de_llm_slop__fl2va` from `67.50` (and symmetric raw `e_own_rms = 22.72–23.38`) down to **`1.75`** (`covered_market`, `M = 266`), **`1.36`** (`indoor_evening`, `M = 91`), and **`1.77`** (`indoor_neutral`, `M = 98`) — **`38x–50x` quieter**, leaving `'Live'`, `'focus'`, `'wide'`, and commas untouched with **zero `.safetensors` schema rebuild** (`erasure.mu_clean` not needed on H3).
   - **Why current `de_llm_slop__fl2va` still needs paired-`D_p` `V_r` for future builds:** CPU check shows `cos(mu_d, v_ref) = 0.9719`, `decode_erasure_payload` yields `mu_d_positive_words = []`, and `vr[2] (d = 0.822)` symmetrically spans `focus (+1.20), foreground (+1.13), depth (+1.06), blur (+0.98), ambient (+0.94), focused (+0.93)` — explaining `-33%..-39%` clean-scene blurring. Future dedicated H3 `Regime B` vectors should extract `V_r` from paired shifts `D_p = P @ B_p - A_p` rather than unsupervised `A_c` so neutral optical axes (`focus/depth`) never enter the eraser.

---

## 4. Integration With `vgel.me` (Nov 12–20, 2026)

1. **`making-neurons-by-hand` (Nov 20, 2026 — `Qwen3.6-27B`):**
   - Hand-wiring one SwiGLU neuron (`gate_proj = up_proj = gate(off, on)`, `down_proj = clean_write`) makes the MLP compute `silu(z) * z * write`: `~0` on `off`, `+write` on `on`.
   - **Connection:** Explains why `Swift 1.5`'s largest weight deltas sit in `mlp.gate_proj` (`0.004987`) and `mlp.up_proj` (`0.004962`), and provides the exact conditional blueprint for H3 `Regime B` (note H3's `TokenRefiner` also uses SwiGLU MLPs) plus an ultra-cheap `heylook` layer-33 SwiGLU sidecar experiment.
   - **Technical caveats:** `vgel` evaluated 1-line greedy completions (`stop_strings=["\n"]`) and noted punctuation stuttering on the rude neuron; mathematically, setting `gate_proj = up_proj = read` computes `silu(z) * z`, which leaks a positive bump (`+0.48` at `z = -2`) when `z << 0`—an explicit `ReLU` gate eliminates that leak.
2. **`shapes-of-model-words` (Nov 19, 2026):**
   - Mechanism #1 (**"self-steering phrases"**—RL teaches short pivot phrases like `"No, that's not correct"` or `"Let me check instead of guess"` that condition the base simulator into a behavior) directly explains Qwen3.8's `"Wait / Actually / However / Alternatively / Hmm"` reasoning loops and why turning those 20 pivots down by `55x–75x` collapses CoT without touching `</think>`.
   - Mechanisms #2–#3 (concept compression + isolated RL drift) explain why LLM prompt generators default to cliché imagery ("gates", "bells", "teal-orange HDR glow") unless constrained by zero-lore mode prompts like [`heylook_auto_edit_guidance_design.md`](../heylook_auto_edit_guidance_design.md).
3. **`words-all-the-way-down` (Nov 12, 2026):**
   - Argues the `d_model`-width vector is the universal "Lego brick" of transformers, substantially made of words—matching H3's unified `5376`-D stack (`control_vectors.py` + `j_lens.py` + `Model2Vec`).
   - Where `vgel` asks *"at what point will we see evidence of non-linguistic substructure?"*, our H3 measurements provide two concrete answers: (a) projecting `50` H3 vectors through `L50..L63` into `lm_head` words crushes `eff_rank` from `31.48` to `14.04`, and (b) `Gate 2B` proved raw 24-ch Video VAE `reference_latent` pixels override VLM/language steering until explicitly released.

---

## 5. Cross-Session Handoff Briefs (`2026-11-23`)

### Brief A — Sibling Session Here (`ComfyUI-h3-explorations`, Post–Wave 4)
1. Finish committing Wave 4 normally; **do not touch `control_vectors.py` or `app/.git@825d27d` today.**
2. **Do not reopen Gate 2B (`Path 1C` `vision_tokens`), Gate 2C (`1x Q_vid[mask]`), or Gate 2A (`Bus-Scoped V_r` `pc8`/`pc16`).** Leave `STEER_TARGETS[6] = "vision_tokens"` intact in [`control_vector_nodes.py:93`](../../../../control_vector_nodes.py#L93) (`AGENTS.md` §1.5 combo stability, `bench/check_control_vectors.py`, Wave 4 `/object_info` compatibility), and simply keep `"vision_tokens"` hidden from `app/` creator dropdowns during `Track App-2.0` (`app/` already has zero references on disk).
3. Unfreeze `Track App-2.0` around the verified 3-tool pipeline (`1x` `heylook` expansion -> `1x` 4D-routed `MiniMaxH3Steer` -> `2x` `EditGuider`/`SpatialDirector`), deprioritize [`next_steps.md`](../../next_steps.md#L25) GPU Item 5 (`layers.50..63` chains), and prioritize our high-leverage unfinished lanes: Embedded Image-in-Video Phase 2 + `EditGuider.mask` SAM3 (`F1`) + `audio_freeze` loose-mask + `M3` `attach_lse_reference_balance`.
4. If revisiting deferred **Regime B (`de_llm_slop` / pure suppressors)** later: do not patch `add_norm_matched_erasure`; add a separate H3 `Regime B` mode using **prompt-local centering `h_i - h_bar` + half-wave `-mu_d` cosine gate (`boost = 0`, upper clamp only)** — needing zero `.safetensors` schema rebuild (`erasure.mu_clean` is for causal `T = 1` LLMs, not mixed-POS H3 prompts) — and extract `V_r` from paired shifts `D_p = P @ B_p - A_p` instead of unsupervised `A_c` on future dedicated builds.

### Brief B — `heylook` Session on Mac (`heylookitsanllm`, `Qwen3.8-27B`)
1. **Fix the 3 transfers before writing code:** (a) anchor causal self-damping `ACE` at `mu_clean` (never `mu_a` without `+mu_d`) with a one-way ray gate + upper cap only; (b) skip `_pack_3band_block_diag` (hardcodes 3 equal bands) and port [`bench/convert_pika_lora.py:48-100`](../../../../bench/convert_pika_lora.py#L48-L100) per-component FP64 SVD instead; (c) re-run activation diagnostics on `Stock 8-bit` vs. `Stock 8-bit + Rank-32 LoRA` to remove the `8-bit` vs. `BF16` confound.
2. **Causal test on `L63`:** Because only `final_norm + lm_head` (+ MTP) follow `L63`, test **4 arms on `code_bug`, `knights_and_knaves`, and a 30–50 prompt held-out coding/math suite**: (i) `<think>`-gated Rank-32 Swift LoRA (`L00..L63`), (ii) `L63`-only `ACE`, (iii) 20-token `NoWait` logit damper, and optionally (iv) a `vgel`-style layer-33 one-way `ReLU` SwiGLU sidecar.
3. **Keep H3 paper separate:** Publish the `Qwen3.8` / `Swift 1.5` 0-GDN + `<think>`-gated LoRA + `L63` `NoWait` findings as a standalone write-up citing `QLoRA`, `LEACE`, `ACE`, `SEAL`, `NoWait`, and `EAT`.

### Brief C — All Active Agents Working Today
1. **Stop trusting CPU cosine/`mpc` metrics as render quality proofs** (remember `stained_glass_cathedral` `0.762 -> 0/2` vs. `youtube_overhype_influencer` `1/12` word match -> Wave 3B win).
2. **Enforce the 3-Regime split:** Regime A (`add_norm_matched_erasure` + `peer_ros`), Regime B (LLM: `mu_clean`-anchored one-way `ACE`; H3: prompt-centered `h_i - h_bar` + half-wave `-mu_d` gate + paired-`D_p` `V_r`, both `boost = 0` + upper cap only), Regime C (`lora_branch` phase-gated SVD LoRA). Never use `Regime A` or `thinking_budget` guillotines for `Regime B`/`C` jobs.
3. **Protect creator runtime contracts across repos:** Keep `heylook` server defaults and root [`heylook_client.py`](../../../../heylook_client.py) / `app/` aligned, leave `STEER_TARGETS[6] = "vision_tokens"` intact in Python while keeping `vision_tokens` out of `app/` creator UI, and keep Paper v5 scoped strictly to MiniMax H3.
