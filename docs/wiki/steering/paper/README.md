# Paper v5 Hub: Causal 4D Prelude Steering in a Single-Stream Joint Audio-Video Diffusion Transformer

**Canonical Paper Title (v5):** **Causal 4D Prelude Steering in a Single-Stream Joint Audio-Video Diffusion Transformer**  
**Subtitle:** *Resolving the `1x` Frozen-Anchor Prompt Veto in MiniMax H3 via Affine Subspace Clearing, Depth-and-Audio Query Routing, and Reduced-Interference Multi-Slider Composition*  
**Status:** Active v5 home, revision plan, manuscript stub, LaTeX scaffold, vector diagram, and promoted CPU/build code (`2026-11-22`, aligned with `b657c6a8` `48`-stem active bank). Ready for Phase 2 GPU data drop without re-architecting the paper.  
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

Consolidated directly from the peer review ([`2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md`](../research/2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md)), fresh-eye audit ([`2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md`](../research/2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md)), promotion spec ([`2026-11-22_paper_to_codebase_promotion_spec.md`](../research/2026-11-22_paper_to_codebase_promotion_spec.md)), Track 1/2 blueprint ([`2026-11-22_track1_track2_multivector_erasure_blueprint.md`](../research/2026-11-22_track1_track2_multivector_erasure_blueprint.md)), Stage 1–2 verification ledger ([`bench/results/2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md`](../../../../bench/results/2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md)), stabilization commits (`48031b0e`, `6f7131dd`, `b657c6a8`), and roadmap ([`../../projects/2026-11-22_stabilization_and_phase2_roadmap/index.md`](../../projects/2026-11-22_stabilization_and_phase2_roadmap/index.md)):

1. **Invert the Thesis & Title (Strict Single-Backbone Scope):** Paper v4 buried a finding that, to our knowledge, was previously unmapped in unified single-stream AV DiTs (`blocks.20..29` visual commitment, `10`-block audio lag in `blocks.31..47`, terminal `block 49` sink, `1.52x`/`1.75x` `Stage D` `N=1` DINOv3 ratio, and `1x Q_aud -> K_txt_base`) in §6.8 while leading with an overclaimed *"Retrieval–Steering Duality"* and untested external architectures (*Looped-DiT*, *LLaDA*, *Dream*, *Huginn*, *Ouro*, *LoopFormer*). With Stage 2 landing compiler-safe `1.003x` hooks (`+0` `malloc_graph_breaks` across `4` backends) and `Spec E` (`Qwen-Image-2.1`) deferred, v5 locks its title, abstract, and claims strictly to MiniMax H3 and drops untested cross-architecture speculation.
2. **Own Prior Art & Fix Mathematical Bugs:** Explicitly cite **ACE**, **LEACE**, **CURE**, **Safree**, **IPErase**, **Linear-AcT**, **RASteer**, **MUTE**, **FADE**, **STAS**, **TADA!**, and **DATE**. Demote v4 `Theorem 1` to **Lemma 1 (PCRR Empirical Diagnostic)** with the corrected constraint $\operatorname{row}(\mathbf{W}) \subseteq \operatorname{row}(\mathbf{V}_r)$ (`control_vectors.py::build_pole_a_erasure` uses only `a_spans`), state the post-RMS $\gamma \le 1$ non-inversion condition, `[0, 1]` union clamp, and pure mathematical RMS bound (`<= 98.5` for `soft_cap_stacked_rms`) explicitly, collapse former Propositions 1–3 into **Design Eliminations**, and keep fixed-$P_0$ `MakeMedia` bottleneck **Proposition 1**.
3. **Close v4 Reporting Errata & Sync `6f7131dd` Consolidation (Zero New `1D` GPU Runs):** Remove jittered `"28 / 38"` counts (report `median cos(mu_d, v_1d) = 0.9935`, `4/52 < 0.980`); label Table 4 (`noir` + `paper_origami`) and Table 11 (`check_c`/`check_b` `v1d_erase`) accurately from existing provenance; explain Table 10 `gate2b2` H.264 chroma floor and include `ref2va__noir`; reframe `de_llm_slop` as an honest Regime A vs. Regime B case study; document the `48`-stem stabilized bank (`48 fl2va`, `48 ref2va`, `41 h3_image` = `137` `.safetensors`, `5` retired including `stained_glass_cathedral` at `mpc=0.762` -> `0/2` `t=1` falsification) plus the rebuilt `204`-term lexicon (`decone_pcs=0`, `47/48`, `top1=0.8298`, `top3=0.9574`, `mrr=0.9007`); and track both landed `48031b0e` fixes (`scene_prompts` Python API, `j_lens.py::check_b_side_alignment` `effective_coverage`) and the remaining open CLI `--scene-prompts` ([`bench/control_vector/make_cmds.py`](../../../../bench/control_vector/make_cmds.py#L115)) and soft-alpha/`erasure.mu_a` hold gaps ([`bench/control_vector/helpers.py::_apply_erasure_holds`](../../../../bench/control_vector/helpers.py#L420)).
4. **Add Reduced-Interference Multi-Slider Algebra & Norm-Cliff Fixes:** Retract v4's refuted `"5+ simultaneous sliders"` claim and replace it with the pre-consolidation `52`-stem `2,652`-pair cross-interference census (`42.4%` max peer hit, `15.9%` self-overlap trap, `50.9%` shared scaffolding), Stage 1+2 proof (`peer_ros`, resolved-mask non-consecutive grouping, `95.5 -> 98.5` `soft_cap_stacked_rms` eliminating the Phase 4 `4.0 MP` `lighthouse` diptych), and CPU-verified `Bus-Scoped drop_look_pc16` (`7.94x` overlap cut at `100%` Look retention — GPU promotion queued first in Phase 2 `B.1`).
5. **Document Honest Track 2 Push-Side Falsification (`Path 1B` -> `Path 1A` + `Path 1C` Unproven on GPU):** Report the CPU LOPO finding that unsupervised semantic-role cosine routing (`Path 1B`) gains `< 1.1%` MSE from `TokenRefiner` stop-word bleed, and mark explicit `Path 1A` atomic sub-faders and `Path 1C` `ColQwen` `tags == 0` vision patches explicitly as **unverified on GPU** until Phase 2.
6. **Broaden Non-`noir` Evaluation & Hardware Boundaries:** Evaluate the `4` `t=1`-verified visual winners (`charcoal_sketch_animated`, `x_ray_fluoroscope`, `mercury_liquid_chrome`, `blacklight_uv_rave`) across contrasting adult `192f`/`243f` multi-shot scenes with `voice_spans`-gated `YIN` $F_0$, DINOv3 `M1`, optical flow, `345f` `FADE` reactivation checks, `FrozenVideoCache` Mode 0 limits, and a gated `1x Q_vid[mask]` vs. `2x` `MiniMaxH3Sol` probe. Treat rigid-body event inventions (e.g., `sudden_takeoff_flight`) as explicitly gated tests, not assumed standalone `1x` capabilities.

---

## 3. Readiness Ledger: Verified Now vs. Phase 2 (`B.1` Engine & Winners First -> `B.2` Net-New Packs)

| Milestone / Gate | Status | Notes & Anchors |
| :--- | :---: | :--- |
| Math fixes (`PCRR`, post-RMS, clamp, RMS cap), prior art, & v4 errata corrections | **Done** | Ready in [`manuscript.md`](manuscript.md) |
| Stage 1 composition fixes (`peer_ros`, barrier grouping, `soft_cap_stacked_rms`) | **Done (`dd45600e` / `340e8ded`)** | `4.0 MP` `lighthouse` diptych eliminated (`clip 0.07255 -> 0.00790`); `box_office` `0.3187 -> 0.1661` |
| Stage 2 4D hooks (`steer_attention_hooks.py`, `block_band`, `1x Q_aud`) | **Done (`340e8ded`)** | `Gate 1`: `1.003x` latency, `+0` `malloc_graph_breaks`, `-0.014 st`; `Showcase C`: `+0.630 st` (`Q_aud -> K_vid` bleed noted) |
| CPU Track 1/2/Stage-3 audits (`52`-stem census, `Bus-Scoped V_r`, `Path 1B` falsification) | **Done (CPU)** | Reproducible via [`scripts/`](scripts/audit_track1_cross_interference.py) (`Bus-Scoped V_r` & `Path 1A` not yet GPU-tested) |
| Active bank consolidation (`48` stems), `t=1` look screen (`4` winners, `stained_glass_cathedral` `0/2`), & `decone_pcs=0` lexicon | **Done (`6f7131dd` / `b657c6a8`)** | `48031b0e` landed Python `scene_prompts` & `j_lens` `effective_coverage`; CLI `--scene-prompts` & soft-hold/`mu_a` gaps stay open |
| **Phase 2 `B.1.1` (First):** Promote `Bus-Scoped V_r` (`drop_look_pc16`) + wire `make_cmds.py` `--scene-prompts` & soft-hold/`mu_a` | **Pending Phase 2** | Lands engine & `Path 1C` prerequisites before new flagship multi-bus GPU renders |
| **Phase 2 `B.1.2`:** Proven `4`-winner `192f` showcases (`charcoal_sketch_animated`, `x_ray_fluoroscope`, `mercury_liquid_chrome`, `blacklight_uv_rave`) + `192f` `1x Q_aud` `YIN` $F_0$ | **Pending GPU** | Contrasting adult scenes (`prompt_bank/t2va_stairwell_dialogue.txt`, `prompt_bank/t2va_restaurant_kitchen.txt`), slot in [`manuscript.md`](manuscript.md) §6.5 |
| **Phase 2 `B.1.3`:** Track 2 `Path 1C` `ColQwen` `tags == 0` Dual-Lock breaker (`N >= 6` pairs) | **Pending GPU** | Requires CLI `--scene-prompts` + `erasure.mu_a` holds (`helpers.py`) first; slot in [`manuscript.md`](manuscript.md) §6.4 |
| **Phase 2 `B.1.4`:** Timeboxed `~90s` `90f` `1x Q_vid[mask]` bleed & wall-clock probe vs. `2x` `MiniMaxH3Sol` | **Pending GPU** | Settles whether portals run at `1x` or `2x`; slot in [`manuscript.md`](manuscript.md) §5.2 & §7.1 |
| **Phase 2 `B.2` (After `B.1`):** Net-new behavioral packs (`youtube_overhype_influencer`, live `late_30_49`, gated `sudden_takeoff_flight`) & `345f` `FADE` reactivation check | **Pending GPU** | Built on verified `B.1` engine baseline; rigid-body takeoff stays explicitly gated |

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
