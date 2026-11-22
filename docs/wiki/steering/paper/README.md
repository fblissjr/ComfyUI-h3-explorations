# Paper v5 Hub: Causal 4D Prelude Steering in Joint Audio-Video Diffusion Transformers

**Canonical Paper Title (v5):** **Causal 4D Prelude Steering in Joint Audio-Video Diffusion Transformers**  
**Subtitle:** *Resolving the `1x` Frozen-Anchor Prompt Veto via Affine Subspace Clearing, Depth-and-Audio Query Routing, and Conflict-Free Multi-Slider Composition*  
**Status:** Active v5 home, revision plan, manuscript stub, LaTeX scaffold, vector diagram, and promoted CPU/build code (`2026-11-22`). Ready for Phase 2 GPU data drop without re-architecting the paper.  
**Historical Predecessor (Frozen Provenance):** Paper v4 ([`../research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](../research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md)) remains untouched in [`../research/`](../research/2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md) as dated provenance alongside its reproduction guide ([`2026-11-21_reproducing_late_interaction_to_dit_steering.md`](../research/2026-11-21_reproducing_late_interaction_to_dit_steering.md)).

---

## 1. Repository Map (Strictly Tracked Paper Artifacts)

Raw `.pt` token banks, untracked per-experiment JSON records under `data/exp_unpooled_sliders/`, and exported binary PDF/plate bundles under `$H3_OUTPUT_DIR/` intentionally stay outside git. Everything required to write, verify on CPU, regenerate SVG figures, compile LaTeX, and export clean HTML/PDF bundles is tracked here:

| Tracked File | Role |
| :--- | :--- |
| [`README.md`](README.md) | This Paper v5 hub, v4-to-v5 structural changelog, Phase 2 freeze gates, and reproducible CLI commands |
| [`manuscript.md`](manuscript.md) | Canonical Paper v5 manuscript stub pre-populated with verified Stage 1–2 + CPU audit facts and `[PHASE 2 GPU]` insertion slots |
| [`main.tex`](main.tex) | Generated arXiv / `pdflatex` LaTeX manuscript built directly from [`manuscript.md`](manuscript.md) |
| [`figures/README.md`](figures/README.md) | Paper v5 figure inventory (`fig_v5_4d_causal_architecture.svg` + planned empirical charts) |
| [`figures/fig_v5_4d_causal_architecture.svg`](figures/fig_v5_4d_causal_architecture.svg) | Paper v5 Figure 1: Affine Prelude Composition + `50`-Block Single-Stream AV DiT & `1x Q_aud` Causal Routing |
| [`scripts/build_paper.py`](scripts/build_paper.py) | Standalone stdlib Markdown-to-LaTeX compiler (`main.tex`) + optional SVG-math HTML & Chromium PDF renderer |
| [`scripts/make_v5_figures.py`](scripts/make_v5_figures.py) | Zero-dependency vector SVG generator for Paper v5 figures (with optional `--from-data` local readiness inspection) |
| [`scripts/export_paper_bundle.py`](scripts/export_paper_bundle.py) | Portable relative-URL HTML/PDF/LaTeX bundle exporter targeting `$H3_OUTPUT_DIR/` |
| [`scripts/audit_track1_cross_interference.py`](scripts/audit_track1_cross_interference.py) | Promoted CPU Track 1 `52`-stem cross-interference census (`2,652` pairs) & `ROS` multi-slider simulator |
| [`scripts/audit_track2_multivector_pole_b.py`](scripts/audit_track2_multivector_pole_b.py) | Promoted CPU Track 2 Leave-One-Pair-Out CV audit across `7` cached per-pair `.pt` banks |
| [`scripts/run_stage3_cpu_prototypes.py`](scripts/run_stage3_cpu_prototypes.py) | Promoted CPU Stage 3 prototype suite: Bus-Scoped `V_r`, `soft_tanh`/oblique norm cap, and `Path 1A` vs. `Path 1B` falsification |

---

## 2. Why Paper v5 Drops the v4 Framing (Six Verified Structural Shifts)

Consolidated directly from the peer review ([`2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md`](../research/2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md)), fresh-eye audit ([`2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md`](../research/2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md)), promotion spec ([`2026-11-22_paper_to_codebase_promotion_spec.md`](../research/2026-11-22_paper_to_codebase_promotion_spec.md)), Track 1/2 blueprint ([`2026-11-22_track1_track2_multivector_erasure_blueprint.md`](../research/2026-11-22_track1_track2_multivector_erasure_blueprint.md)), Stage 1–2 verification ledger ([`bench/results/2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md`](../../../../bench/results/2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md)), and stabilization roadmap ([`../../projects/2026-11-22_stabilization_and_phase2_roadmap/index.md`](../../projects/2026-11-22_stabilization_and_phase2_roadmap/index.md)):

1. **Invert the Thesis & Title (Lead with Joint AV 4D Routing):** Paper v4 buried its 100% net-new discovery (`blocks.20..29` visual commitment, `10`-block audio lag in `blocks.31..47`, terminal `block 49` sink, `1.52x`/`1.75x` DINOv3 face preservation, and `1x Q_aud -> K_txt_base`) in §6.8 while leading with an overclaimed *"Retrieval–Steering Duality"* and untested external architectures (*Looped-DiT*, *LLaDA*, *Dream*, *Huginn*, *Ouro*, *LoopFormer*). With Stage 2 landing compiler-safe `1.003x` hooks (`+0` `malloc_graph_breaks` across `4` backends) and `Spec E` (`Qwen-Image-2.1`) deferred, v5 leads cleanly with MiniMax H3 Joint AV 4D Causal Routing and keeps external recurrent/masked-diffusion architectures strictly in Discussion hypotheses.
2. **Own Prior Art & Fix Mathematical Bugs:** Explicitly cite **ACE**, **LEACE**, **CURE**, **Safree**, **IPErase**, **Linear-AcT**, **RASteer**, **MUTE**, **FADE**, **STAS**, **TADA!**, and **DATE**. Demote v4 `Theorem 1` to **Lemma 1 (PCRR Empirical Diagnostic)** with the corrected constraint $\operatorname{row}(\mathbf{W}) \subseteq \operatorname{row}(\mathbf{V}_r)$ (`control_vectors.py::build_pole_a_erasure` uses only `a_spans`), state the post-RMS $\gamma \le 1$ non-inversion condition and `[0, 1]` union clamp explicitly, collapse former Propositions 1–3 into **Design Eliminations**, and keep fixed-$P_0$ `MakeMedia` bottleneck **Proposition 1** (also newly closed in code via `control_vector_nodes.py::make_control_vector_from_media_pairs` `scene_prompts`).
3. **Close All Five v4 Reporting & Provenance Errata (Zero New `1D` GPU Runs):** Remove jittered `"28 / 38"` counts (report bank-wide `median cos(mu_d, v_1d) = 0.9935`, `4/52 < 0.980`); label Table 4 (`noir` + `paper_origami`) and Table 11 (`check_c`/`check_b` `v1d_erase`) accurately from existing provenance; explain Table 10 `gate2b2` H.264 chroma floor and include `ref2va__noir`; reframe `de_llm_slop` as an honest Regime A vs. Regime B case study; and document the `effective_coverage` fix in `j_lens.py::check_b_side_alignment`.
4. **Add Multi-Vector Composition Algebra & Norm-Cliff Fixes:** Retract v4's refuted `"5+ simultaneous sliders"` claim and replace it with the `52`-stem `2,652`-pair cross-interference census (`42.4%` max peer hit, `15.9%` self-overlap trap, `50.9%` shared scaffolding), Stage 1+2 proof (`peer_ros`, resolved-mask non-consecutive grouping, `95.5 -> 98.5` `soft_cap_stacked_rms` eliminating the `4.0 MP` `lighthouse` diptych), and CPU-verified `Bus-Scoped drop_look_pc16` (`7.94x` overlap cut at `100%` Look retention).
5. **Document Honest Track 2 Push-Side Falsification (`Path 1B` -> `Path 1A` + `Path 1C`):** Report the CPU LOPO finding that unsupervised semantic-role cosine routing (`Path 1B`) gains `< 1.1%` MSE from `TokenRefiner` stop-word bleed, focusing Track 2 cleanly on explicit `Path 1A` atomic sub-faders + syntactic `h3_spans.py` clauses + `Path 1C` `ColQwen` `tags == 0` vision patches (`face_retain` `ROS` + Dual-Lock release).
6. **Broaden Non-`noir` Evaluation & Hardware Boundaries:** Retire `paper_origami`, `wes_anderson_symmetry`, weak/correlated duds, and child/teen prompts from new flagship evaluation; pair contrasting adult `192f`/`243f` multi-shot scenes with `voice_spans`-gated `YIN` $F_0$, DINOv3 `M1`, optical flow, `345f` `FADE` reactivation checks, `FrozenVideoCache` Mode 0 limits, and `1x Q_vid[mask]` vs. `2x` `MiniMaxH3Sol` gates.

---

## 3. Readiness Ledger: Verified Now vs. Queued for Phase 2 GPU Drop

| Milestone / Gate | Status | Notes & Anchors |
| :--- | :---: | :--- |
| Math fixes (`PCRR`, post-RMS, clamp), prior art, & v4 errata corrections | **Done** | Ready in [`manuscript.md`](manuscript.md) |
| Stage 1 composition fixes (`peer_ros`, barrier grouping, `soft_cap_stacked_rms`) | **Done (`dd45600e` / `340e8ded`)** | `4.0 MP` `lighthouse` diptych eliminated (`clip 0.07255 -> 0.00790`); `box_office` `0.3187 -> 0.1661` |
| Stage 2 4D hooks (`steer_attention_hooks.py`, `block_band`, `1x Q_aud`) | **Done (`340e8ded`)** | `1.003x` base latency, `+0` `malloc_graph_breaks` / `0` rogues across 4 backends, `YIN` $F_0 = -0.014\text{ st}$ |
| CPU Track 1/2/Stage-3 audits (`52`-stem census, `Bus-Scoped V_r`, `Path 1B` falsification) | **Done** | Reproducible via [`scripts/`](scripts/audit_track1_cross_interference.py) |
| Media-pair per-prompt fix + `--hold-words` erasure + `j_lens` `effective_coverage` | **Done (`48031b0e`)** | `control_vector_nodes.py::make_control_vector_from_media_pairs`, `j_lens.py::check_b_side_alignment` |
| **Phase 2 Gate A:** Live `192f` `streams="audio_only" + block_band="late_30_49"` + `192f` `1x Q_aud` `YIN` $F_0$ | **Pending GPU** | Slot in [`manuscript.md`](manuscript.md) §5 & §6.5 |
| **Phase 2 Gate B:** `Bus-Scoped V_r` (`drop_look_pc16`) GPU multi-bus A/B | **Pending GPU** | Slot in [`manuscript.md`](manuscript.md) §6.2 |
| **Phase 2 Gate C:** Track 2 `Path 1C` `ColQwen` `tags == 0` Dual-Lock breaker (`N >= 6` pairs) | **Pending GPU** | Slot in [`manuscript.md`](manuscript.md) §6.4 |
| **Phase 2 Gate D:** Adult-only `192f`/`243f` non-`noir` showcase suite, `345f` `FADE` check, & `~90s` `1x Q_vid[mask]` vs. `2x` `MiniMaxH3Sol` probe | **Pending GPU** | Slots in [`manuscript.md`](manuscript.md) §5, §6.5, §7.1 |

---

## 4. Build & Reproduction Commands (Never Run `uv`)

```bash
COMFY="$(cd ../.. && pwd)" && PY="$COMFY/.venv/bin/python"
set -a; . ./h3.env; set +a

# 1. Regenerate Paper v5 vector figure(s) (< 1 s CPU, zero external data required)
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/make_v5_figures.py

# 2. Rebuild Paper v5 LaTeX scaffold `main.tex` from `manuscript.md` (< 1 s CPU)
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/build_paper.py --tex-only

# 3. Optional: inspect local untracked JSON readiness & export portable HTML/PDF bundle to $H3_OUTPUT_DIR/
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/make_v5_figures.py --from-data
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/export_paper_bundle.py --skip-pdf

# 4. Run repo CPU verification & promoted Paper v5 audits (when local bank + data/ pt files exist)
CUDA_VISIBLE_DEVICES="" "$PY" bench/verify_pole_a_erasure.py --cpu-only
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/audit_track1_cross_interference.py
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/audit_track2_multivector_pole_b.py
CUDA_VISIBLE_DEVICES="" "$PY" docs/wiki/steering/paper/scripts/run_stage3_cpu_prototypes.py
```
