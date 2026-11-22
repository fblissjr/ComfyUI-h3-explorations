# Paper v5 Figure Manifest (`docs/wiki/steering/paper/figures/`)

Vector SVG figures for **Causal 4D Prelude Steering in Joint Audio-Video Diffusion Transformers** (`manuscript.md`), generated reproducibly by [`../scripts/make_v5_figures.py`](../scripts/make_v5_figures.py) with zero owner paths or hardcoded scratch directories.

---

## 1. Active Tracked v5 Figures

| Filename | Status | Generator | Description |
| :--- | :---: | :--- | :--- |
| [`fig_v5_4d_causal_architecture.svg`](fig_v5_4d_causal_architecture.svg) | **Generated & Tracked** | [`make_v5_figures.py`](../scripts/make_v5_figures.py) | Dual-panel architecture diagram: **(a)** `1x` Affine Prelude Transport & Multi-Slider Algebra (`-P_{V_r}`, `ROS`, `95.5 -> 98.5` smooth knee, `Bus-Scoped V_r`), and **(b)** `50`-block Single-Stream AV DiT Causal Routing (`blocks.0..19`, `20..29`, `30..49`, Stage 2 `1x Q_aud -> K_txt_base` at `1.003x` / `+0` graph breaks). |

---

## 2. Planned Phase 2 Empirical v5 Charts & Frozen v4 Disposition

Once `[PHASE 2 GPU]` gates close, [`../scripts/make_v5_figures.py`](../scripts/make_v5_figures.py) will emit updated quantitative v5 SVGs into this folder without mutating Paper v4's frozen provenance under [`../../research/figures/`](../../research/figures/fig1_tau_goldilocks_and_deconing.svg):

| Planned v5 Slot | v4 Reference (`../../research/figures/`) | Change for Paper v5 |
| :--- | :--- | :--- |
| **Fig 1 (Architecture)** | Replaces `fig6_prelude_eq_architecture.svg` | **Tracked now** as [`fig_v5_4d_causal_architecture.svg`](fig_v5_4d_causal_architecture.svg) — drops untested *Looped-DiT* / *LLaDA* claims and adds `ROS`, norm knee, `blocks.20..49`, and `1x Q_aud`. |
| **Fig 2 (`52`-Stem Bank & Scaffolding)** | Updates `fig2_38bank_regimes_and_grassmann.svg` | Drop `"28 / 38"` word-rescue jitter; chart `52`-stem `cos(mu_d, v_1d)`, `42.4%` cross-erasure, `15.9%` self-overlap, `50.9%` shared PCs, and `Bus-Scoped drop_look_pc16` `7.94x` reduction. |
| **Fig 3 (`50`-Block Causal Trace & `1x Q_aud`)** | Extends [`fig4_dit_50block_modality_trace.svg`](../../research/figures/fig4_dit_50block_modality_trace.svg) | Keep `50`-block video-before-audio emergence curve (`fig4` reusable directly) and add Stage 2 `1x Q_aud` vs. `2x` speed/`YIN` $F_0$ / DINOv3 M1 panel. |
| **Fig 4 (Corrected Synergy & Double Dissociation)** | Updates `fig5_factorial_and_specificity.svg` | Pair `noir` with `paper_origami` parity in Panel A and clarify Table 11 `v1d_erase` provenance under `no-new-1D` lock. |
| **Fig 5 (Norm Cliff, `ROS`, & Track 2 LOPO)** | Net-new v5 empirical figure | Plot `soft_cap_stacked_rms` (`95.5 -> 98.5` eliminating `lighthouse` diptych) and Track 2 `35%–49%` unexplained residual + `Path 1B` falsification vs. `Path 1A` / `Path 1C`. |
| **Fig 6 (Diverse `192f`/`243f` Non-`noir` & `345f` Check)** | Replaces `noir`-heavy v4 sweeps (`fig1`, `fig3`) | Adult-only contrasting multi-shot benchmarks + `Path 1C` `ColQwen` Dual-Lock unlock (`Phase 2`). |
