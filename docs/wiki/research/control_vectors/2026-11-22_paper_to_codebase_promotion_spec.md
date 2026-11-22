# Paper-to-Codebase Promotion Spec & 9-Item Gap-Closure Ledger (`2026-11-22`)

**Date:** `2026-11-22`  
**Companion Monograph (Paper v4):** [`2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md) · [`2026-11-21_reproducing_late_interaction_to_dit_steering.md`](2026-11-21_reproducing_late_interaction_to_dit_steering.md)  
**Benchmark Ledgers:** [`bench/results/2026-11-21_multivector_erasure_benchmark.md`](../../../../bench/results/2026-11-21_multivector_erasure_benchmark.md) · [`bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md)  
**Scope:** Tracks every paper mechanism, experimental boundary, and bank/tooling gap identified after Paper v4 (`52a62318`) with exact disk-verified code symbols, owners across Lanes A–D, empirical CPU triage measurements, and acceptance criteria for GPU Window 1 (`faa62ef5`), GPU Window 2 (`90e9d2bd` Wave 2), and Window 3.  
**Policy (`2026-11-22`):** Focus exclusively on multi-vector Pole-A subspace erasure (`add_norm_matched_erasure`) — zero new 1D work or 1D comparison renders (historical Paper v4 1D failure baselines remain as archived provenance only).

---

## 1. Master 9-Item Gap-Closure & Ownership Matrix

| # | Gap / Unfinished Item From Paper v4 | Current Code / Data Reality on Disk | Owner & Schedule | Promotion / Closure Criterion |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Block-Band Text K/V Routing (`blocks.20..49`) in Production** | Proved on `paper_origami` in bench (`bench/control_vector/run_phase3_steering_experiments.py` `_attach_shadow_text_kv_band`); requires `disable_comfy_compiler = True` + `pause_malloc_graph()` due to per-block arena frees (`comfy/ldm/minimax/model.py:658`). Absent from [`MiniMaxH3Steer`](../../../../control_vector_nodes.py#L4561). | **Phase 4 (`blocks20`):** Lane A (`faa62ef5`)<br>**Spec A below:** Lane B (`085d496e`) | **Window 1:** Inverted shadow (`felt_muppet +1.5`, `noir +1.5` on `90f` `ref2va`). If both preserve facial lip-sync/identity while isolating stylization, append `block_band` at the **end** of `MiniMaxH3Steer.define_schema` (Rule 5). |
| **2** | **`1x` Single-Pass `Q_aud` Stream-Split Audio Protection** | Heavy dark reductions (`de_llm_slop` / Base50 `noir`) drop un-CFG'd audio RMS `-11.1` to `-13.4 dB`, while `MiniMaxH3Steer.streams = "video_only"` still pays `2.00x` DiT passes. | **Phase 4 (`qaud`):** Lane A (`faa62ef5`)<br>**Spec B below:** Lane B | **Window 1:** Compare `Steered Both` vs. `1x Q_aud -> K_txt_base` vs. `2x streams="video_only"` on `90f` `t2va_box_office` (`de_llm_slop +1.15`). Quantifies 1-hop (`Q_aud -> K_txt`) vs. 2-hop (`Q_aud -> K_vid`) coupling before wiring. |
| **3** | **Runtime Clamped Union Multi-Slider Stacking + `erasure.mu_a` & `MIN_ERASURE_PAIRS` Parity** | Live stacking sums `_erasure_row_shift64` outputs linearly (`P_sum` on `noir + chaos_panic + de_llm_slop` has **`7` eigenvalues `> 1.0`**, unweighted `λ_max = 1.9262`, `1.8568` weighted); `per_token_erasure_profile` (`control_vectors.py:1596`) & `compose_erasure_payload` (`:1510`) drop `erasure.mu_a`; `build_pole_a_erasure` (`:1289`) & `MakeMedia` (`control_vector_nodes.py:3525`) bypass `MIN_ERASURE_PAIRS = 6`; `bench/control_vector/cli.py:615` defaults `grade --mode` to `1D`. | **Lane A** (`faa62ef5` implementation + `stack3` + `bench/control_vector/cli.py:615` + `bench/check_control_vectors.py:4376`)<br>**Spec C below:** Lane B math review | Land `_stacked_erasure_block_shift64` with continuous `e_budget` + `||add_total||_2 <= 1e-6` guard, `per_token_erasure_profile(..., mu_a=None)`, `bench/control_vector/cli.py:615` lockstep fix, and `stack3` (`t=1` + `90f` `4`-slider `1x` AV showcase). |
| **4** | **Native `MiniMax-H3-Image` Bank Coverage (`42 / 52` Missing)** | `101` bank `.safetensors` exist (`52` `fl2va`, `39` `ref2va`, `10` `h3_image`, `0` dedicated `fl2va_base` for `P-H3-Image`). `{fl2va, ref2va}` share `erasure.vr` intact (`control_vector_nodes.py:2825-2848`); only `h3_image` falls back to 1D encoder projection (`:2793-2824`). Exact split of the `42` missing stems: **`32` static visual/look/material/expression**, `4` dynamic motion/blocking, `6` audio/vocal-only. | **GPU builds:** Lane A (`faa62ef5` missing visual `--families h3_image` in Window 1 + Showcase D)<br>**Relay for `app/`:** Lane D (`90e9d2bd`) -> `61cf9813` | Build `--families h3_image` for core visual/material concepts; showcase native `4.0 MP` `MiniMax-H3-Image` rank-8 multi-vector erasure (`base` vs. native `r=8` single & stacked sliders, zero 1D comparison renders); add `"Erasure r=8"` vs. `"1D fallback"` badge in `app/`. |
| **5** | **Legacy `.toml` & Concept Hygiene (`31 / 39` Warn + `data/edit_concepts/`)** | All `51` TOMLs pass Gate D except `crystallize_ice.toml` Pair 8 (`'bright'`). Across `15` prefix-collapsed `.toml`s in `census_38_bank.json`, **`13 / 15` already work cleanly** (`mpc = 0.574..0.807`), leaving only `2` weak multi-facet prefix-collapsed `.toml`s (`wes_anderson_symmetry` `mpc = 0.453`, `handheld_documentary_zoom` `mpc = 0.525`). Separate cleanup: `10` `[vector]` TOMLs mixed into `data/edit_concepts/`. | **`crystallize_ice` & `data/edit_concepts/` move:** Lane A (`faa62ef5`)<br>**Policy rule:** Do NOT bulk-rewrite working `.toml`s | Fix `crystallize_ice.toml` Pair 8 (`0` Gate D warnings); move `10` `[vector]` TOMLs to `data/control_vectors/`. Any future rewrite of `wes_anderson_symmetry` / `handheld_documentary_zoom` requires paired visual A/B grading before overwriting `bank/`. |
| **6** | **Multi-Pair `MakeMedia` (`N >= 6..10`) & Dual-Lock `fl2va` Unlock** | Lab 4 (`workflows/h3_image_ref_make_media_erasure_pika_api.json`) feeds `N = 1` pair (`40.6%` Grassmann coverage); zero saved `vision_tokens` vectors in `bank/`. `fl2va` (`conditioning.py:169-185`) pins raw 24-ch VAE `cond_video_latents` alongside `tags == 0` vision rows, holding keyframe RGB chroma across `[0, 0.33]`. | **Spec D below:** Lane B<br>**GPU Execution:** **Window 3** (queued after `faa62ef5` Window 1 + `90e9d2bd` Wave 2) | Build `6–10`-pair `noir_vis` (`pooling="vision_tokens"`, `erasure.mu_a`) + Lab 4 multi-batch doc; test Dual-Lock `fl2va` pre-graded endpoints vs. `guide_end_percent = 0.17` early release. |
| **7** | **`app/` Vibe Search `-2PC` Symmetry Check (`app/cv_lab.py`)** | `app/cv_lab.py:1197-1212,1299-1301` scores `max(s_v1, mat_mu @ q_vec)` where lexicon queries (`bench/control_vector/build_model2vec_lexicon.py:394`) have top-2 common PCs removed while bank `mu_d` / `v1` vectors do not. **Measured CPU verdict (§3.2 below): keep `app/cv_lab.py` AS SHIPPED** — de-coning the `52`-stem bank **hurts** Top-1 (`88.9% -> 77.8%`). | **Closed (`085d496e` CPU benchmark verified)**<br>Relay via `90e9d2bd` -> `61cf9813` | Zero code change needed in `app/cv_lab.py`; `-2PC` bank subtraction verified counter-productive on contrastive difference matrices. |
| **8** | **Second-Architecture External Validation (Local `Qwen-Image-2.1`)** | Paper §7.3 transfers Prelude EQ algebraically to *Looped-DiT*, Masked DLMs, and *Huginn*/*Ouro*, while `100%` of GPU runs use MiniMax-H3. Local `$H3_MODEL_STORE/diffusion_models/qwen_image_2.1_bf16.safetensors` (`comfy/ldm/qwen_image/model.py:435-450` `WrappersMP.DIFFUSION_MODEL`) is already on disk (`0` downloads). | **Spec E below:** Lane B<br>**GPU Execution:** **Window 3** (required only for external paper publication / cross-DiT generality) | Minimal 3-pair `noir` `Base` vs. `Three-Subspace Multi-Vector EQ` still check on local `qwen_image_2.1_bf16.safetensors` (`~2 min` GPU, zero weights trained). |
| **9** | **Named-Object `cfg = 1.0` Color Clauses & Objective Perceptual Audio/Identity Probes** | `cfg = 1.0` prompts with explicit noun-color bindings (`t2va_sitcom_kitchen.txt`: `"brown polyester knit vest"`, `"golden retriever"`) retain partial garment/fur tint under `scene_only` (`Paper §7.2.4`). CPU `YIN` $F_0$ audit (§3.1) and `M1` DINOv3 `stage_c`/`stage_d` copy-freeze & identity audit (§3.4) are now complete on existing Phase 3 clips. | **CPU `YIN` + `M1` DINOv3 Phase 3 DONE:** Lane B (§3.1, §3.4)<br>**Phase 4 `M1` / `voice_spans` sweep:** Lane B post-Window 1<br>**`sitcom_kitchen` `subject_definitions` vs. `2x` A/B:** Window 3 | Quantify `subject_definitions + scene_only` vs. `2x EditGuider` on `sitcom_kitchen`, plus `bench/voice_spans.py`-gated `YIN` $F_0$ and `M1` DINOv3 scores across Phase 3 & Phase 4 clips in Paper v5. |

---

## 2. Detailed Code-Grounded Promotion Specs (Specs A–E)

### Spec A — Production `blocks.20..49` Shadow-Text K/V Routing (`MiniMaxH3Steer`)
- **Problem proven in Stage D ([`stage_d_shadow_text_kv.json`](../../../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_d_shadow_text_kv.json)):** Steering `blocks.20..49` (`border_lap = 140.66`, `center_vs_ref_sat_diff = 0.0192`) achieves crisp material emergence while leaving `blocks.0..19` unsteered to preserve live-action face anatomy and lip-sync.
- **Technical hurdle to clear before wiring into `control_vector_nodes.py`:** `bench/control_vector/run_phase3_steering_experiments.py` sets `disable_comfy_compiler = True` and wraps per-block writes in `pause_malloc_graph()` because ComfyUI's block prefetch cleanup (`comfy/ldm/minimax/model.py:658`, `prefetch_queue_pop(..., malloc_scope="block")`) frees per-block compiler arenas between `block[i]` and `block[i+1]`.
- **Target inverted-shadow production architecture (verified against `comfy/ldm/minimax/model.py:648-671`):**
  1. Run the main DiT `h_txt` stream **unsteered** across `blocks.0..19` (`0` extra work on early blocks; zero arena lifetime violation while unsteered).
  2. Pre-allocate a single reusable `[1, L_txt, 5376]` steered text buffer **outside** `_run_blocks` (or run the `[L_txt, 5376]` `2`-block `TokenRefiner` / block-wise text shadow inside `optimized_attention_override` under `pause_malloc_graph()`) and only patch text K/V slices `[tx_a:tx_b]` on `block_index >= 20`.
  3. Because `block 49` has `text_rel_inc == 0.000` (terminal sink, Paper §6.6), verify whether `blocks.20..48` gives identical output to `blocks.20..49` without interfering with dense tail block `49` (`SOL_DENSE_TAIL = "49"`).
  4. **Rule 5 UI constraint (`AGENTS.md` §1.5):** Never reorder `MiniMaxH3Steer.define_schema` (`control_vector_nodes.py:4561`) inputs; **append** optional `block_band` (`["all", "mid_late_20_49", "late_30_49"]`, default `"all"`) strictly at the **end** of `inputs`, and run `bench/check_node_ids.py` + `bench/check_widget_deviations.py`.

### Spec B — `1x` Single-Pass `Q_aud -> K_txt_base` Stream-Split Decision Gate
- Currently `MiniMaxH3Steer(streams="video_only" | "audio_only")` (`control_vector_nodes.py:4662`) runs `2` DiT forwards per step (`2.00x`).
- In Phase 4 `qaud`, `faa62ef5` compares `Steered Both`, `1x Q_aud -> K_txt_base` (passing steered `K_txt, V_txt` to text+video queries and unsteered `K_txt_base, V_txt_base` strictly to `[aud_a:aud_b]` queries inside `optimized_attention_override`), and `2x streams="video_only"` on `90f` `t2va_box_office` (`de_llm_slop +1.15`).
- **Decision rule:**
  - If `1x Q_aud -> K_txt_base` recovers `>= 70%` of the `-13.4 dB` ambience/RMS drop without altering `hsv_sat`, wire query-sliced `optimized_attention_override` as a `1.00x` implementation path for `streams` inside `MiniMaxH3Steer` (plus compiler compatibility test under `MiniMaxH3Sol`).
  - If `Q_aud -> K_vid` cross-attention through single-stream blocks (`blocks.31..48`) accounts for most of the attenuation, keep `2.00x` splitting for strict stream invariance and document the `1x` `soundscape_only` counter-anchor recipe instead.

### Spec C — Clamped Union Stacking Theorem 2.6 + `erasure.mu_a` & `MIN_ERASURE_PAIRS` Parity
Verified against `control_vectors.py:1510-1621,3409-3724` and live `fl2va` `.safetensors` (`noir`, `chaos_panic`, `de_llm_slop`):
- **Why parallel sum diluted `noir`:** Adding per-slider `_erasure_row_shift64` outputs sums `P_sum = sum_k min(s_k, 1.0) * P_k`, yielding `7` unweighted eigenvalues `> 1.0` (`[1.9262, 1.6659, 1.5505, 1.3366, 1.2610, 1.1384, 1.0435]`; `1.8568` at `s = 1.25, 1.0, 0.9`) and suppressing `cos(P_after, P_base)` in `V_r(noir)` down to `+0.108`.
- **Exact runtime clamped union specification (Lane A implementing in `control_vectors.py`):**
  1. Stack `W = cat_k sqrt(min(s_k, 1.0) * diag_k) * vr_k`, run thin SVD without rank-16 truncation, and clamp `d_union = (s_vals ** 2).clamp(0.0, 1.0)` (`> 1e-9`), lifting simulated `cos(P_after, P_base)` from `+0.108 -> +0.903`.
  2. Use the continuous total erasure budget **`e_budget = min(sum_k min(s_k, 1.0) * NORM_MATCHED_TARGET, max(NORM_MATCHED_TARGET, ||add_total||_2))`** — algebraically exact at `K = 1`, continuous under splitting `s = 1.0` into `0.5 + 0.5` (`67.5 == 67.5`, avoiding `sqrt(sum)`'s `30%` step drop), and protective when opposing boosts reduce `||add_total||_2`.
  3. Guard `||add_total||_2 <= 1e-6` by returning `-e` directly instead of collapsing raw RMS to zero.
  4. Partition additive blocks cleanly (`K >= 2` prompt-centered erasure specs in `_stacked_erasure_block_shift64` vs. `1D` / `s <= 0` / `mu_a` entries + trailing `mode="hold"`), add optional `mu_a: torch.Tensor | None = None` to `per_token_erasure_profile` (`control_vectors.py:1596`, used by `bench/control_vector/grading.py:206`), preserve `erasure.mu_a` cleanly in `compose_erasure_payload`, and enforce `MIN_ERASURE_PAIRS = 6` across compiled builds consistently.

### Spec D — Multi-Pair `MakeMedia` (`N >= 6..10`) & Dual-Lock `fl2va` Unlock (Window 3)
1. **Multi-pair `MakeMedia` bank vector:** Build a reusable `6–10`-pair `noir_vis__fl2va` / `noir_vis__ref2va` vector via `MiniMaxH3MakeMediaControlVector(pooling="vision_tokens")` (storing `erasure.{mu_d,vr,diag,mu_a}` via `build_spatial_vision_erasure`) and update Lab 4 (`tutorials_progressive_guide.md` + `workflows/h3_image_ref_make_media_erasure_pika_api.json`) to show batch `[N >= 6, H, W, C]` vs. single-pair `40.6%` Grassmann coverage.
2. **In-graph Dual-Lock `fl2va` remedy (`conditioning.py:169-185`):** Test (a) pre-grading `first_frame` + `last_frame` through `MiniMax-H3-Image` `noir` (`0.2355 -> 0.0330` B&W) or grayscale transform before `MiniMaxH3Conditioning`, vs. (b) early keyframe release (`guide_end_percent = 0.17` + `Return-LSE`) on `fl2va_noodle_141f` (`cond_video_latents` drop right at step 1, freeing `[0, 0.34]` palette commit without mutating `MiniMaxH3Conditioning` defaults).

### Spec E — Second-Architecture Non-H3 Proof (`Qwen-Image-2.1`) & `cfg = 1.0` Clause Test (Window 3)
1. **Zero-download local DiT test:** `$H3_MODEL_STORE/diffusion_models/qwen_image_2.1_bf16.safetensors` (`comfy/ldm/qwen_image/model.py:435-450` wraps `_forward` cleanly via `WrappersMP.DIFFUSION_MODEL` and receives `[B, L, D]` `context` directly, no weight training needed). Running `Base` vs. `Three-Subspace Multi-Vector EQ` across a `3`-prompt conflicting still matrix directly validates Paper §7.3's Prelude–Core–Coda generality outside MiniMax-H3 (no new `1D` arm).
2. **Named-object `cfg = 1.0` clause boundary test:** On `t2va_sitcom_kitchen.txt` (`"brown polyester knit vest"`, `"golden retriever"`, `"avocado green kitchen cabinetry"`), score `scene_only` vs. `scene_only + subject_definitions` (`fields=["subject_definitions","summary","description"]`) vs. `2x EditGuider` (`scale = 1.5`) so Paper §7.2.4 specifies when switching `fields` resolves noun-bound colors in `1x` vs. requiring `2x`.

---

## 3. Verified Zero-GPU CPU Measurements (`2026-11-22`)

### 3.1 Objective `YIN` $F_0$ Pitch Audit on Existing Phase 3 AV Clips (`probe_yin_f0_drift.py`)

Measured via `CUDA_VISIBLE_DEVICES="" "$PY" bench/probe_yin_f0_drift.py` (JSON records written to `data/exp_unpooled_sliders/phase3_2026_11_21/yin_f0_phase3_{stage_a,stage_c,stage_d}.json`):

| Phase 3 Stage & Clip | Baseline $F_0$ (Hz) | Steered $F_0$ (Hz) | Global Shift (`st`) | Paired Shift (`st`) | Voiced (`base -> steer`) | Diagnostic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Stage C` `woman_blonde` `gear1_selective_color_erasure`** | `209.0` | `205.8` | `-0.27` | `-0.20` | `0.30 -> 0.31` | Clean vocal stability (`0` alerts); selectively removes green shop awning while keeping voice untouched. |
| **`Stage C` `gear2a_vis_erasure_rel033`** | `209.0` | `208.1` | `-0.08` | `-0.45` | `0.30 -> 0.29` | Clean vocal stability (`0` alerts) under `tags == 0` vision erasure. |
| **`Stage C` `gear2b_vis_erasure_rel017_clean085`** | `209.0` | `216.7` | `+0.62` | `+0.26` | `0.30 -> 0.34` | Clean (`0` alerts). |
| **`Stage D` `origami` `early_only_0_19`** | `210.7` | `207.0` | `-0.31` | `+0.14` | `0.33 -> 0.30` | Clean (`0` alerts). |
| **`Stage D` `origami` `mid_late_20_49`** | `210.7` | `212.8` | **`+0.17`** | `-1.90` | **`0.33 -> 0.33`** | **Winning `blocks.20..49` band preserves global vocal pitch & voiced fraction (`0.33`) with `0` alerts.** |
| **`Stage D` `origami` `late_only_30_49`** | `210.7` | `217.4` | `+0.54` | `+0.74` | `0.33 -> 0.31` | Clean (`0` alerts). |
| **`Stage D` `origami` `all_blocks_0_49` / `+lse_ref_balance`** | `210.7` | `210.7` / `214.4` | `+0.00` / `+0.30` | `-2.32` / `-1.90` | `0.33 -> 0.33` | Clean global pitch (`0` alerts). |
| **`Stage A` `box_office` `de_llm_slop` `1d` / `erasure`** | `109.96` | `174.29` / `163.52` | `+7.97` / `+6.87` | `+3.13` / `+3.24` | `0.18 -> 0.15` / `0.18` | **Methodological insight:** `base` carries a heavy `65.7 Hz` trailer/music bed (`2.5–3.75 s`, `dbfs_p50 = -21.6`) that lowers un-gated median $F_0$; `erasure` drops background ambience (`dbfs_p50 = -42.8`, zero `2.5–3.75 s` bed) leaving speech clear (`143.7–172.4 Hz`). Always pass `bench/voice_spans.py` `--voice-spans` when comparing clips with shifted score/ambience beds. |

### 3.2 CPU `-2PC` Symmetry Benchmark for `app/cv_lab.py` Vibe Search (`18` Queries $\times$ `52` Bank Stems)

Measured strictly on CPU against `j_lens.load_model2vec_lexicon()` (`180` terms, `-2PC` centered, `post_refiner.fl2va`) and the `52` `fl2va` `.safetensors` in `$COMFY/models/control_vectors/bank/`:

| Scoring Matrix vs. `Model2Vec` (`-2PC`) Query `q` | Top-1 Accuracy | Top-3 Accuracy | Mean Target Cosine | Specific Effect on Problem Cases |
| :--- | :---: | :---: | :---: | :--- |
| `1D_only` (`M_v1 @ q`) | `15 / 18` (`83.3%`) | `17 / 18` (`94.4%`) | `+0.3358` | Misses weak 1D targets where `mu_d` rescues target vocabulary |
| **`current_app_max(v1, mu)` (Shipped `app/cv_lab.py:1299-1301`)** | **`16 / 18` (`88.9%`)** | **`17 / 18` (`94.4%`)** | **`+0.3396`** | **Highest recall & target cosine (`noir` #1 `+0.349`, `amber` #1 `+0.394`)** |
| `v1_only_2pc` | `14 / 18` (`77.8%`) | `17 / 18` (`94.4%`) | `+0.3053` | Over-subtracts legitimate strong contrast directions |
| `max(v1_2pc, mu_2pc)` | `14 / 18` (`77.8%`) | `17 / 18` (`94.4%`) | `+0.3124` | Drops `"noir"` (`Rank 1 -> 2`) & `"amber"` (`Rank 1 -> 2`) |
| `mu_2pc_only` | `14 / 18` (`77.8%`) | `16 / 18` (`88.9%`) | `+0.2942` | Worst overall recall and cosine |

**Conclusion for `app/` (`61cf9813`):** Keep `app/cv_lab.py:1197-1212,1299-1301` as shipped. Bank `B - A` difference vectors are already centered contrastive directions; extracting top-2 PCs over only `52` diverse difference vectors treats high-leverage style axes as nuisance common modes.

### 3.3 Exact Breakdown of Missing `--families h3_image` (`42 / 52`) & Prefix-Collapsed `.toml`s (`15 / 39`)

- **Out of `42` stems present in `fl2va` (`52`) without `__h3_image` (`10`):**
  - **`32` visual / style / material / expression / VFX stems (build priority):** `balloon_head_inflate`, `bioluminescent_glow`, `claymation_stopmotion`, `crystallize_ice`, `cursed_cctv`, `elastic_neck_stretch`, `env_airborne_dust_flour_haze`, `env_torrential_rain_drench`, `env_wet_skin_oil_sheen`, `expr_deadpan_stonefaced`, `expr_flushed_breathless_tension`, `felt_muppet`, `fever_melt`, `head_swell`, `heavy_blizzard`, `knitted_wool_world`, `lettering`, `look_amber_candlelight`, `look_clinical_nordic_fluorescent`, `magnetic_attraction`, `miniature_tiltshift`, `molten_lava_golem`, `paper_origami`, `pyro_chain_explosions`, `rubber_hose_1930s`, `smoke_dissolve`, `strobe_club_pulse`, `sunken_coral_reef`, `toy_brick_world`, `tv_1950s_broadcast`, `underwater_submerged`, `wind_tunnel_gale`.
  - **`4` dynamic motion / blocking stems (low still priority):** `chaos_panic`, `handheld_documentary_zoom`, `infomercial_fail`, `zero_gravity`.
  - **`6` audio / vocal-only stems (skip `h3_image`, no audio stream):** `auctioneer_rapid_patter`, `operatic_bellow`, `pyro_klaxon_audio`, `reverb_cathedral`, `vocal_strain`, `whisper_asmr`.
- **Out of `15` prefix-collapsed `.toml`s (`median word first_diff < 10`) in `prompt_bank/control_vectors/`:**
  - `13 / 15` achieve `mpc = 0.574..0.807` (`noir` `0.574`, `paper_origami` `0.699`, `claymation_stopmotion` `0.711`, `film_16mm_1974` `0.807`) and are already visually verified — **never bulk-rewrite working `.toml`s blindly**; only `wes_anderson_symmetry` (`mpc = 0.453`) and `handheld_documentary_zoom` (`mpc = 0.525`) fall in the weak multi-facet prefix-collapsed tail (`< 0.55`).

### 3.4 Objective `M1` DINOv3 `<Picture 1>` Copy-Freeze & Geometric Stylization Audit (`probe_dinov3_copy_freeze.py`)

Measured strictly on CPU (`CUDA_VISIBLE_DEVICES=""`, `dino_v3_L_naf_fp32.safetensors`, `crop=False`, `9` frames per `90f` clip against `$H3_INPUT_DIR/woman_blonde.jpg`, saved to `data/exp_unpooled_sliders/phase3_2026_11_21/dinov3_m1_phase3_stage_cd.json`):

| Phase 3 `ref2va` Clip (`90f`) | Ref CLS (`f0 / mean / max`) | Ref Patch Mean | Temporal Self-Cos (`mean / min`) | Quantitative Diagnostic |
| :--- | :---: | :---: | :---: | :--- |
| `stage_c` `gear0_1d_text` | `0.187 / 0.463 / 0.731` | `0.331` | `0.9416 / 0.8947` | Natural motion (`< 0.985` freeze threshold), no static trace (`0` alerts). |
| `stage_c` `gear1_selective_color_erasure` | `0.137 / 0.377 / 0.671` | `0.317` | `0.8707 / 0.5867` | Active motion + background desaturation (`min = 0.5867`), `0` alerts. |
| `stage_c` `gear2a_vis_erasure_rel033` | `0.137 / 0.388 / 0.622` | `0.304` | `0.8960 / 0.7564` | Preserves motion & live-action structure (`0` alerts). |
| `stage_c` `gear2b_vis_erasure_rel017_clean085` | `0.157 / 0.384 / 0.655` | `0.308` | `0.8730 / 0.7551` | Clean (`0` alerts). |
| `stage_d` `origami` `unsteered_base` | `0.169 / 0.399 / 0.657` | `0.326` | `0.9501 / 0.9297` | Unsteered baseline identity/motion floor (`0` alerts). |
| `stage_d` `origami` `early_only_0_19` | `0.164 / 0.387 / 0.661` | `0.325` | `0.9438 / 0.9151` | Identical to unsteered (`0.661` vs. `0.657`) — confirms `blocks.0..19` do not induce origami geometry. |
| **`stage_d` `origami` `mid_late_20_49`** | **`0.046 / 0.161 / 0.323`** | **`0.203`** | **`0.8807 / 0.6517`** | **Synthesizes crisp origami glasses (`0.657 -> 0.323` CLS shift) while retaining `1.76x` higher peak DINOv3 reference CLS similarity than `all_blocks_0_49` (`0.184`).** |
| `stage_d` `origami` `late_only_30_49` | `0.123 / 0.421 / 0.706` | `0.311` | `0.9227 / 0.8484` | Matches unsteered baseline — misses the `blocks.20..29` transition band. |
| `stage_d` `origami` `all_blocks_0_49` | `0.032 / 0.106 / 0.184` | `0.143` | `0.9318 / 0.8600` | Over-covers nose & mouth (`0.184` max CLS), erasing lower-face likeness. |
