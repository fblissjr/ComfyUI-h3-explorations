# Project Atlas & Audit (`2026-11-13`)

Cross-session audit (`HEAD` `7630c8ff`) synthesizing **16 `agy` sessions** (repo + CLI project), **11 Claude root sessions (`+ 51` subagents)** from `2026-11-10`..`11-13` (out of `78` Nov 4–13 DuckDB rows), **`352` commits** since `2026-11-04` (`215` since `2026-11-09`), **`65` registered nodes**, **`108` workflow graphs**, and verified `$H3_OUTPUT_DIR` renders.

- **Offline Interactive Report:** [`report.html`](report.html) (zero-CDN self-contained HTML viewer)
- **Machine-Queryable Ledger:** [`ledger.json`](ledger.json) (`schema_version: "1.0"`)
- **Folder Contract & Portfolio Index:** [`../AGENTS.md`](../AGENTS.md) · [`../index.md`](../index.md) · [`../../index.md`](../../index.md)

---

## 1. Agent Query Cookbook (`ledger.json`)

Agents should query [`ledger.json`](ledger.json) rather than scraping HTML or long prose:

```bash
# Top unfinished/forgotten ideas with exact file:line anchors
jq '.unfinished_and_forgotten_ideas[] | {rank, id, leverage, effort, code_anchors}' docs/wiki/projects/2026-11-13_repo_audit_and_atlas/ledger.json

# Filter non-parked tracks & their owning wiki hubs
jq '.tracks[] | select(.status != "parked_closed") | {id, status, wiki_owners}' docs/wiki/projects/2026-11-13_repo_audit_and_atlas/ledger.json

# List verified $H3_OUTPUT_DIR render folders (excluding parked 'fun')
jq '.render_catalogue[] | {date, batch, highlights}' docs/wiki/projects/2026-11-13_repo_audit_and_atlas/ledger.json
```

---

## 2. Codebase Footprint & LLM Wiki Consolidation

| Surface | Measured Count | Notes |
| :--- | :--- | :--- |
| **Custom Nodes** | **`64`** node IDs | Locked in [`bench/node_id_manifest.json`](../../../../bench/node_id_manifest.json) / [`nodes.py`](../../../../nodes.py) (`65` at AM audit; `MiniMaxH3SolChunked` retired PM `2026-11-13`) |
| **Root `.py` Modules** | **`74`** files (`48,003` lines) | Largest: [`control_vector_nodes.py`](../../../../control_vector_nodes.py) (`4,698`), [`control_vectors.py`](../../../../control_vectors.py) (`3,024`), [`subject_track.py`](../../../../subject_track.py) (`3,005`) |
| **`bench/` `.py` Scripts** | **`511`** files (`243,374` lines) | `480` flat `bench/*.py` (`232,652` lines, `169` `check_*.py`) + 5 subpackages |
| **Shipped Workflows** | **`108`** `.json` graphs | Generated from [`workflows/h3_config.py`](../../../../workflows/h3_config.py) via [`workflows/build_workflows.py`](../../../../workflows/build_workflows.py) |
| **LLM Wiki (`docs/wiki/`)** | **`288`** files | `206` moved from legacy `docs/*`; `decisions.md` sharded into `6` files |

---

## 3. Portfolio Status Across 8 Tracks (`'fun'` Masked V2V Parked)

| Track | Status | Primary Wiki Hub | Key Shipped Nodes / Modules |
| :--- | :--- | :--- | :--- |
| **Control Vectors & J-Lens** | Active | [`../../control_vectors.md`](../../control_vectors.md) | `MiniMaxH3Steer`, `WordVector`, `ReleaseReferences`, [`j_lens.py`](../../../../j_lens.py), [`bench/control_vector/ui_server.py`](../../../../bench/control_vector/ui_server.py) |
| **Edit Guidance** | Active | [`../../edit_guidance.md`](../../edit_guidance.md) | [`edit_guidance.py`](../../../../edit_guidance.py) (`MiniMaxH3EditGuider`, `e0a5bb35`), [`h3_prompt_edit.py`](../../../../h3_prompt_edit.py) |
| **Sound Cues & 2D D-Pad** | Active | [`2026-11-12_sound_cues_dpad_and_033_release_law.md`](../../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md) | [`sound_cues.py`](../../../../sound_cues.py), [`sprite_trajectory.py`](../../../../sprite_trajectory.py), `video_end_percent=0.33` |
| **H3-Image & P-H3-Image** | Shipped | [`../../image/h3_image.md`](../../image/h3_image.md), [`../../image/p_h3_image.md`](../../image/p_h3_image.md) | [`h3_image_nodes.py`](../../../../h3_image_nodes.py), [`ph3_image_nodes.py`](../../../../ph3_image_nodes.py), `MiniMaxH3EncodeImageReferences` |
| **Sol Sparse & Precision** | Active | [`../../attention/SOLATTN.md`](../../attention/SOLATTN.md) | [`sol_attn_h3.py`](../../../../sol_attn_h3.py) (`SOL_DENSE_TAIL="49"`), [`archive/sol_chunked_h3.py`](../../../../archive/sol_chunked_h3.py) (retired), [`attention.py`](../../../../attention.py) |
| **Core Rules & Prompting** | Active | [`../../core/h3_frame_rules.md`](../../core/h3_frame_rules.md) | [`h3_rules.py`](../../../../h3_rules.py), [`AGENTS.md`](../../../../AGENTS.md), [`conditioning.py`](../../../../conditioning.py) |
| **Audio Freeze & Song Loops** | Queued Exps | [`../../core/h3_audio_freeze.md`](../../core/h3_audio_freeze.md) | [`audio_freeze.py`](../../../../audio_freeze.py), [`audio_freeze_song.py`](../../../../audio_freeze_song.py), [`audio_refine.py`](../../../../audio_refine.py) |
| **Masked V2V & `lane_db`** | **Parked (`'fun'`)** | [`../../masked_v2v.md`](../../masked_v2v.md) | [`video_mask.py`](../../../../video_mask.py), [`subject_track.py`](../../../../subject_track.py), [`bench/lane_db/db.py`](../../../../bench/lane_db/db.py) (Tier-1 SQLite ready for future `'fun2'`) |

---

## 4. Top 14 Unfinished & Forgotten High-Leverage Directions

| # | Direction | Leverage | Effort | Verified Anchor(s) | Sessions |
|---:|---|---|---|---|---|
| **1** | Spatiotemporal `[T, H, W]` `MASK` & Curve Modulation on `MiniMaxH3EditGuider` | Critical | `S (~40 lines)` | `edit_guidance.py:196`, `edit_guidance.py:278` | `3b57a539`, `a9f31858` |
| **2** | Interactive Canvas/Web UI for 2D Sprite Trajectory D-Pad & Sound Cues | High | `M` | `sprite_trajectory.py:492`, `sound_cues.py:337` | `3b57a539`, `a9f31858` |
| **3** | 2-Subject Sprite D-Pad (`<Video 1>` + `<Video 2>`), SAM-3 Cutouts, & Combined `ref2va` Ablation | High | `S–M` | `sprite_trajectory.py:492`, `sam31_corrections.py:1` | `3b57a539`, `a9f31858` |
| **4** | Whitened Whole-Word J-Lens Table & Neighbor-Hold Contrastive Verification | High | `S (CPU + 4h GPU)` | `j_lens.py:351`, `bench/control_vector_readout_score.py:119` | `ad7e5777`, `346a1e5c` |
| **5** | Fix `DEFAULT_AUX_BASELINES` Length/Domain Imbalance (`cos = 0.72` Shared Bias) | High | `XS` | `h3_pair_compiler.py:59`, `h3_pair_compiler.py:716` | `a9f31858`, `3b57a539` |
| **6** | Control Vector Studio 3-Way Builder Unification, Thin-Wrapper CLI, & Open Studio UI Bugs | High | `M` | `control_vector_nodes.py:2628`, `bench/control_vector/ui_server.py:95` | `4ebaab8d`, `f77c2cfa` |
| **7** | Multimodal Differential `EditGuider` (`plain_cut` vs `drawn_cut`) & Event-Gated Facial Envelopes | High | `M` | `edit_guidance.py:76`, `edit_guidance.py:278` | `3b57a539`, `a9f31858` |
| **8** | ~~Forward `quantizer` in `MiniMaxH3SolChunked`~~ (**CLOSED PM 2026-11-13**: node retired to `archive/sol_chunked_h3.py`) & Rebuild `comfy-kitchen` Wheel | Medium-High (wheel only) | `S` | `archive/sol_chunked_h3.py`, `sol_attn_h3.py:166` | `a8949cba`, `23d096c4` |
| **9** | Upper-Layer Chain Readout (`layers.50..63` + `lm_head`) on Qwen3-VL-32B | Medium-High | `M (GPU pass)` | `j_lens.py:1896`, `docs/wiki/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md:1` | `ad7e5777`, `346a1e5c` |
| **10** | Mobile Studio (`app/`): `/queue` + `/history` Payload Privacy Trim & 'Edit From Render' Button | Medium | `S` | `workflows/h3_config.py:1658`, `docs/wiki/image/h3_image.md:1` | `c5132fe6`, `53392f0e` |
| **11** | Unrendered `audio_freeze` Stem/0.25-Release & `AudioRefineMask` + `FrozenVideoCache` Tests | Medium | `S (GPU run)` | `audio_freeze_song.py:502`, `audio_refine.py:32` | `79a5bba6` |
| **12** | `lane_db` Tier-2/3 Provenance, DuckDB Analytical Views, & `perception-lhf` for Future `'fun2'` | Medium | `M` | `bench/lane_db/db.py:48`, `bench/lane_db/cli.py:46` | `553f263b`, `683575dd` |
| **13** | Cross-Session / Multi-Harness Agent Bus (`Claude` <-> `agy`) & Shared Transcript Indexer | Medium | `S–M` | `docs/wiki/projects/AGENTS.md:1`, `bench/build_wiki_index.py:1` | `3b57a539`, `a9f31858` |
| **14** | Phase-2 Internal Package Grouping for `75` Root `.py` (`48.2k` LOC) & `480` Flat `bench/*.py` (`232.9k` LOC) | High (Maintainability) | `L (Staged PRs)` | `bench/check_node_ids.py:1`, `control_vector_nodes.py:1` | `79a5bba6`, `4ebaab8d` |

Full acceptance criteria, drop reasons, and next actions for all 14 items are stored in [`ledger.json`](ledger.json) and rendered in [`report.html`](report.html).

---

## 5. Code-vs-Prose Truth Ledger

| Status | Area | Discrepancy Summary |
|---|---|---|
| **OPEN** | Temporal Visual Guides | **Prose:** `docs/wiki/research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md:170` writes that feeding drawn HUD bars as a full-clip `cond` visual guide paints literal bars *'unless released early in the step schedule'*. **Code/Evidence:** `docs/wiki/research/2026-11-12_guide_signal_full_length_bars.md:5-8` actually tested both held AND early-released full-length bar guides (`bars_a_rel`, `bars_b_rel`) and verified the picture STILL comes back as literal bars — because `comfy/ldm/minimax/model.py:34,608,754` pins `minimax_keyframes` `cond` via `video_patch_proj` at `VISUAL_COND_TIMESTEP = 0.999`. `NKD_v153`/`NKD_v173` overlaid HUD strips post-render and used short `until`-released localized keyframes (`cond_audio` / image). |
| **RESOLVED (PM 2026-11-13)** | Attention / SolChunked | **Prose:** `MiniMaxH3SolChunked` exposed a `quantizer` widget parallel to `MiniMaxH3Sol` without forwarding it (`sol_chunked_h3.py:158-161`). **Resolution:** `MiniMaxH3SolChunked` and `bench/check_sol_chunked.py` were retired to [`archive/sol_chunked_h3.py`](../../../../archive/sol_chunked_h3.py) and [`archive/bench/check_sol_chunked.py`](../../../../archive/bench/check_sol_chunked.py) (`65 -> 64` registered nodes in `bench/node_id_manifest.json`), closing the forwarding gap with zero unused code retained. |
| **FIXED `11-13`** | Audio Freeze / Plate Restore | **Prose:** `docs/wiki/core/h3_audio_freeze.md:290` previously claimed no pack node restored clean plates between two samplers. **Code/Evidence:** `plate_restore.py` (`MiniMaxH3RestorePlate`) implements clean plate restoration between two samplers. |
| **FIXED `11-13`** | Bench Checks | **Prose:** Pre-existing failures in `bench/check_audio_freeze.py:836` (`docs/wiki/core/h3_audio_freeze.md` line check), `bench/check_two_shot_load.py:178` (stale `spliced.mp4` doc assert), `docs/wiki/research/sglang_comparison.md:512`, `docs/wiki/research/sglang_h3_pipeline.md:545`, `docs/wiki/research/sglang_h3_pipeline.md:637`, and `docs/wiki/masked_v2v/fun_render_backlog.md:304`. **Code/Evidence:** Verified actual targets and line ranges on disk (`audio_freeze_song.py`, `bench/shot_to_load.py`, `comfy/...`). |
| **FIXED `11-13`** | Wiki Routers | **Prose:** `docs/wiki/decisions.md` (`~204 KB`) and `docs/wiki/index.md` (`~50.5 KB`) exceeded `ROUTER_MAX_BYTES = 45,000` (`read_file` 46,080 B limit) and generated `prompt_bank.md`/`prompt_catalogue.md` tripped `build_wiki_index.py`. **Code/Evidence:** Sharded `2026-10-24`–`2026-11-10` into 6 `<= 40 KB` files under `docs/wiki/decisions/`, slimmed `docs/wiki/index.md` to `~17 KB`, and updated `bench/build_wiki_index.py:159` to exclude non-router generated catalogs. |
| **FIXED `11-13`** | Control Vectors vs EditGuider | **Prose:** Earlier notes treated Control Vectors and `MiniMaxH3EditGuider` interchangeably on localized scene/optical shifts. **Code/Evidence:** `exp2`–`exp5` + `fedcdff1` established distinct strengths: Control Vectors excel as `1x` broadcast world/style/medium dials (`cos=0.72` Pole-A caveat on single-phrase builds), while `MiniMaxH3EditGuider(first_step=1)` preserves anatomy and handles subtle continuous or split-stream edits (`1.5x`). |

---

## 6. Verified Render Batches (`$H3_OUTPUT_DIR`, `'fun'` Excluded)

| Date | Batch | Verified Key Clip / Folder | Takeaway |
|---|---|---|---|
| `2026-11-12` | Shipped Zero-File Sound Cues + 2D Sprite D-Pad Smoke | `$H3_OUTPUT_DIR/Video/h3_r2v_dpad_sound_pika_00001-audio.mp4` | Cold-start zero-file graphs: `f51` hop apex, `-434 px` facing-right backward walk with 3 leg cycles, organic foley (`r=+0.0238`). |
| `2026-11-12` | Sound Islands & Leash-Yank Multi-Actor Choreography | `$H3_OUTPUT_DIR/Video/guide_sound_2026-11-12/exp1/stack_5arms.mp4` | `e0a5bb35` packed-tensor `streams="video_only"` fix eliminated `.wav` bleed (`r=-0.0812` vs `+0.8414`) while slamming Hop 2 at `f54..57`. |
| `2026-11-12` | 2D Kinematic D-Pad (`<Picture 1> + <Video 1>`) | `$H3_OUTPUT_DIR/Video/ref_signal_2026-11-12/exp2_dpad/stack_3arms_dpad.mp4` | 4-way trajectory D-pad (`+420 px` right, `-444 px` left, `-392 px` backward facing right) with zero backdrop bleed (`124/124` tracked). |
| `2026-11-12` | Control Vector Word Share & `exp2`–`exp5` Ladder | `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/exp2_film_16mm_1974_stacked.mp4` | Wait-on-owner-eye 8-clip J-Lens split (`shared`/`named`/`rest`) + 11 `exp3` single-phrase builds isolating the `cos=0.72` Pole-A shift. |
| `2026-11-11..12` | Edit Guidance 3-Way Showcase & Diner Fun Vectors | `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/exp1_hat_off_guide_s2.2_s20261112_00001-audio.mp4` | 20 showcase clips (`edit_guidance_showcase`, 60 files incl. png/mp4) + 20 diner concept renders (`fun_vectors`) + 316 files in `Video/edit_guidance/`. |
| `2026-11-11` | 6-Step Pika Sol `SOL_DENSE_TAIL="49"` Panel (15 measured clips) | `$H3_OUTPUT_DIR/Video/h3_t2v_pika_pika_market__s0_b49_00001-audio.mp4` | Blind Pass 2 proved `s0_b49` (`55/300` dense calls) ties or beats `dense` while saving `10.1–12.6 s` per 345-frame render. |
| `2026-11-10..11` | MiniMax-H3-Image Int8/BF16 Probes & Pika Still / Field Mask Trials | `$H3_OUTPUT_DIR/Image/h3_image_probe/shipped_h3_image_t2i_00001_.png` | 29 probe stills in `h3_image_probe`, 135 files in `pika_still_trial`, 22 in `field_mask_trial`, and 15 in `Image/2026-11-10_cv_comparisons`. |

---

## 7. Phase-2 Code Reorganization Blueprint (Follow-Up PRs)

Phase 1 (`docs/wiki/`) is complete. Phase 2 tackles the `74` root `.py` modules and `480` flat `bench/*.py` scripts in small compatibility-preserving PRs:

1. **Hard invariants:** Never rename any of the `64` active `node_id` values ([`bench/node_id_manifest.json`](../../../../bench/node_id_manifest.json)), never reorder `get_node_list` or input/output tuples, and keep thin top-level re-export shims at repo root during migration.
2. **Root -> `h3_pack/`:** Group internal implementations into `h3_pack/{loaders,attention,conditioning,steering,guidance_and_motion,image,audio,masked_v2v,probes}/` while splitting the `4,698`-line [`control_vector_nodes.py`](../../../../control_vector_nodes.py) monolith (`node defs` vs tensor math) and folding `4ebaab8d`'s 3 duplicated pair builders (`node` / `cmd_build` / `cmd_make`) into one core helper.
3. **Bench -> Subpackages:** Move flat `bench/*.py` into `bench/{checks,eval,masked_v2v,ops}/` alongside the existing 5 clean subpackages (`_lib`, `comfy_capture_nodes`, `control_vector`, `h3_image`, `lane_db`), quarantining `'fun'`-era capture/shot scripts (`capture_masked_run.py`, `shot_to_load.py`, `assemble_delivery.py`) into `bench/masked_v2v/` while keeping [`bench/run_checks.py`](../../../../bench/run_checks.py) as the single test runner.

## Post-Audit Updates (2026-11-13 PM)

- Active follow-up queues after this audit snapshot (Path A `MiniMaxH3SolChunked` retirement landed; parked Path B `EditGuider` mask spec, Path C `h3_pair_compiler.py` notes, immediate sprite/sound-cue follow-up tests, standing evaluation rules, and the unreproduced shipped-graph sprite `cx` footnote) are tracked in [`../../next_steps.md`](../../next_steps.md).
