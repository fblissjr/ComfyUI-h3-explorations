# Stage 1 & Stage 2 Runtime Promotion: Per-Slider `ROS`, Smooth Norm Knee (`95.5 -> 98.5`), and Compiler-Safe Unified Attention Hooks

**Date:** `2026-11-22`
**Lane A Commits:** Stage 1 [`dd45600e`](../../control_vectors.py) · Stage 2 (`steer_attention_hooks.py`, `attention.py`, `exact_blocks.py`, `control_vector_nodes.py`, `verify_stage2_unified_steer_hooks.py`)
**Scope:** Multi-Vector & Clamped Pole-A Subspace Erasure (`add_norm_matched_erasure`) across `MiniMax-H3` (`fl2va` Pika SVD995, `ref2va` Pika SVD995) and native `4.0 MP` `MiniMax-H3-Image` (`h3_image`). Zero 1D comparison arms; retired stems excluded.

---

## 1. Stage 1: Per-Slider `ROS`, Seam Fixes, and Smooth Stacked Norm Knee (`95.5 -> 98.5`)

Stage 1 promoted three runtime stabilization fixes in [`control_vectors.py`](../../control_vectors.py) and [`control_vector_nodes.py`](../../control_vector_nodes.py) (`dd45600e`), verified against [`data/exp_unpooled_sliders/stage1_2026_11_22/stage1_ros_norm_cliff.json`](../../data/exp_unpooled_sliders/stage1_2026_11_22/stage1_ros_norm_cliff.json):

1. **Per-slider Residual-Orthogonal Steering (`peer_ros=True`):** Removes cross-slider interference (`V_r^(i)` orthogonalized against peer boosted directions) inside stacked additive blocks.
2. **Smooth conditioning-norm knee (`STACK_SMOOTH_KNEE_NORM = 95.5`, `STACK_SMOOTH_CEIL_NORM = 98.5`):** Applied strictly when `K >= 2` distinct non-collinear erasure directions stack on the same rows via `soft_cap_stacked_rms`, eliminating norm-driven posterization / diptych fractures without altering single sliders (`0.5 + 0.5`), orthogonal pairs (`<= 95.5`), or explicit `1D + erasure` mixed blocks.
3. **Seam fixes:** Separated prompt vs. `erasure.mu_a` vision clamped-union passes and enabled barrier-aware non-consecutive grouping across resolved masks (`max_stacked_rms="smooth"` default on `MiniMaxH3Steer`).

### Stage 1 Measured Sweep (`stage1_ros_norm_cliff.json`)

| Suite / Prompt (`seed=730451892`) | Arm | `net_row_rms` | `hsv_sat` | `laplacian_var` | `rms_contrast` | `highlight_clip_frac` | Elapsed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fl2va` `box_office` (`t=1`, baseline) | `base` | `0.00` | `0.2920` | `215.31` | `58.24` | `0.00020` | `2.99 s` |
| `stack3`: `noir 1.25 + chaos_panic 1.0 + de_llm_slop 0.9` | `no_ros_nocap` | `134.64` | `0.3083` | `198.38` | `79.33` | `0.09746` | `1.37 s` |
| `stack3`: `noir 1.25 + chaos_panic 1.0 + de_llm_slop 0.9` | `ros_nocap` | `134.64` | `0.1772` | `218.17` | `80.06` | `0.10114` | `1.42 s` |
| `stack3` capped `120 / 110 / 100` | `ros_cap100` | `100.00` | `0.1523` | `238.93` | `84.23` | `0.12315` | `1.38 s` |
| `cyberpunk_neon_rain 1.25 + noir 1.25` | `ros_cap110` | `110.00` | `0.4790` | `457.14` | `87.50` | `0.12840` | `1.38 s` |
| `MiniMax-H3-Image` `4.0 MP` `lighthouse` | `base` | `0.00` | `0.2384` | `196.67` | `66.58` | `0.00087` | `12.07 s` |
| `lighthouse` (`claymation 1.20 + noir 1.15`) | `no_ros_nocap` | `113.07` | `0.0895` | `773.36` | `73.83` | `0.07255` (diptych) | `3.04 s` |
| `lighthouse` (`claymation 1.20 + noir 1.15`) | `ros_cap105` | `105.00` | `0.0773` | `536.57` | `73.87` | `0.06775` (diptych) | `3.06 s` |
| `lighthouse` (`claymation 1.20 + noir 1.15`) | `ros_cap98` | `98.00` | `0.1114` | `436.31` | `75.84` | `0.00520` (clean) | `3.04 s` |

---

## 2. Stage 2: Compiler-Safe Unified Attention Hooks (`MiniMaxH3Steer`)

Stage 2 promotes [`steer_attention_hooks.py`](../../steer_attention_hooks.py) alongside seams in [`attention.py`](../../attention.py) and [`exact_blocks.py`](../../exact_blocks.py), exposing two optional trailing inputs on `MiniMaxH3Steer` in [`control_vector_nodes.py`](../../control_vector_nodes.py):

- `block_band` (`"all"`, `"mid_late_20_49"`, `"late_30_49"`, `"early_0_19"`): Routes shadow text K/V updates across block subsets (Mode 1) inside Comfy's active `malloc_graph` compiler (`args.disable_comfy_compiler = False`), recycling per-clone `H3SteerStepWorkspace` buffers (`max 2` LRU entries, cleared via `CallbacksMP.ON_CLEANUP`) under `pause_malloc_graph_ctx()`.
- `split_backend` (`"auto_1x_kv"`, `"legacy_2x_pass"`): Executes `streams="video_only"` / `"audio_only"` in **one DiT pass** (`1.0x`) by running `Q_aud` against unsteered/targeted text K/V via fused SDPA and splicing `aud_out_bhnd` before `out_proj` (Mode 2 standalone, Mode 3 dual-shadow combined with `block_band`).
- Multi-clone isolation, mutex guards against `MiniMaxH3FrozenVideoCache` / `nodes_sparse_attention`, and FunControl `H3SteerBlockPatch.previous` forwarding (`to`/`cleanup`/`models`).

Verified on live RTX 4090 by [`bench/control_vector/verify_stage2_unified_steer_hooks.py`](../../bench/control_vector/verify_stage2_unified_steer_hooks.py) -> [`data/exp_unpooled_sliders/stage2_2026_11_22/stage2_verification.json`](../../data/exp_unpooled_sliders/stage2_2026_11_22/stage2_verification.json) (`failures: []`, exit `0`):

### 2.1 Gate 1 & Gate 3: Compiler Graph Parity, `1x` Audio Preservation, & Peak PyTorch VRAM Delta (`90f` AV)

Note: Under dynamic VRAM on 24 GB RTX 4090 (`cudaMallocAsync`), unsteered `MiniMaxH3Model` baselines record `99` (`fl2va`) and `149` (`ref2va` due to `ref_vae_end_percent=0.50` layout shrink `28857 -> 24657`) `malloc_graph_breaks` during warmup steps `0..1`. Every Stage 2 Mode 1/2/3 arm added **`+0` graph breaks** and **`0` rogues**.

| Gate / Arm (`90f`, `6`-step Pika, `seed=730451892`) | Mode / Hooks | Compiler Breaks / Rogues | PyTorch Peak (`Δ` vs Base) | `audio_dbfs` / `YIN` `F0` Δ | Visual Metrics & Timing |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Gate 1 Base (`fl2va` `box_office`)** | Unsteered | `99 / 1` (`used=True`) | `142.03 MiB` (`+0.0`) | `-23.11 dBFS` / ref | `sat=0.3646`, `lap=167.98`, `26.82 s` |
| **Gate 1 `de_llm_slop +1.15` (`video_only`, `auto_1x_kv`)** | Mode 2 (`6` steps, `300` splices, `50` blks, `blk49=True`) | **`99 / 0`** (`+0` added) | `217.83 MiB` (**`+75.80 MiB`**) | **`-23.19 dBFS` (`-0.014 st`)** | `sat=0.2803`, `lap=193.17`, **`26.91 s` (`1.003x`)** |
| **Gate 3 Base (`ref2va` `woman_blonde`)** | Unsteered (`end=0.50`) | `149 / 0` (`used=True`) | `175.90 MiB` (`+0.0`) | `-20.79 dBFS` | `center_diff=0.0644`, `border_sat=0.2254`, `34.04 s` |
| **Gate 3 `noir +1.5` (`description_only`, `band=20..49`, `both`)** | Mode 1 (`6` steps, `180` swaps, `0` splices, `blk49=True`) | **`149 / 0`** (`+0` added) | `355.25 MiB` (`+179.35 MiB`) | `-22.35 dBFS` | `center_diff=0.0515`, `border_sat=0.1901`, `33.97 s` |
| **Gate 3 `noir +1.5` (`description_only`, `band=20..49`, `video_only`)** | Mode 3 (`6` steps, `180` swaps, `180` splices, `blk49=True`) | **`149 / 0`** (`+0` added) | `412.11 MiB` (`+236.21 MiB`) | **`-21.29 dBFS`** (`+1.06 dB` vs M1) | `center_diff=0.0501`, `border_sat=0.1931`, `35.97 s` |

Multi-clone isolation verified on `fl2va`: unsteered base clone retained `0` `patches_replace["dit"]` entries while steered clone held `50` (`clone_isolation_ok = true`).

### 2.2 Gate 2: 4-Backend Parity Matrix (`fl2va` `90f`, `2`-step Pika Schedule)

All 4 attention configurations matched exact hook counts, zero added breaks (`99`), zero rogues, and `block_49_touched = True`:

| Attention Backend | Mode 2 (`video_only`, `1x`) | Mode 1 (`mid_late_20_49`) | Elapsed (M2 / M1) |
| :--- | :--- | :--- | :---: |
| `MiniMaxH3Sol` + `MiniMaxH3SageAttention` (`fp8++ rotated`) | `finite=True`, `calls=2`, `splices=100`, `blk49=True`, `breaks=99/0` | `finite=True`, `calls=2`, `swaps=60`, `blk49=True`, `breaks=99/0` | `9.34 s` / `10.46 s` |
| `MiniMaxH3SageAttention` standalone | `finite=True`, `calls=2`, `splices=100`, `blk49=True`, `breaks=99/0` | `finite=True`, `calls=2`, `swaps=60`, `blk49=True`, `breaks=99/0` | `11.74 s` / `11.57 s` |
| `ModelAttentionBackend` (`comfy kitchen attention`) | `finite=True`, `calls=2`, `splices=100`, `blk49=True`, `breaks=99/0` | `finite=True`, `calls=2`, `swaps=60`, `blk49=True`, `breaks=99/0` | `11.94 s` / `11.88 s` |
| `MiniMaxH3ExactBlocks(blocks="49")` + `Sol` + `Sage` | `finite=True`, `calls=2`, `splices=100`, `blk49=True`, `breaks=99/0` | `finite=True`, `calls=2`, `swaps=60`, `blk49=True`, `breaks=99/0` | `10.70 s` / `10.65 s` |

### 2.3 Gate 4: Production `MiniMaxH3Steer` Smooth Norm Knee (`max_stacked_rms="smooth"`, `peer_ros=True`)

Visual inspection of `$H3_OUTPUT_DIR/Image/stage2_2026-11-22/stage2_prod_box_office_stack3.png` and `stage2_prod_h3i_lighthouse.png` (and `$H3_OUTPUT_DIR/Video/stage2_2026-11-22/strips/*.png`):

- **`fl2va` `t=1` `box_office` `stack3` (`noir 1.25 + chaos_panic 1.0 + de_llm_slop 0.9`):** `base` (`hsv_sat=0.3187`, `lap=227.31`, `clip=0.00012`) -> `prod_stack3_ros_smooth` (`hsv_sat=0.1661`, `lap=219.96`, `clip=0.09792`). Visually crisp ticket booth scene with clerk preserved at the ticket window and strong monochrome noir shift (`0` collapse).
- **`4.0 MP` `MiniMax-H3-Image` `lighthouse` (`claymation_stopmotion 1.20 + noir 1.15`):** `base` (`hsv_sat=0.2384`, `lap=196.67`, `clip=0.00087`, `11.51 s`) -> `prod_clay_noir_smooth_ros` (**`hsv_sat=0.1136`**, **`lap=437.15`**, **`clip=0.00790`**, `3.07 s`). Completely eliminates the Stage 1 diptych/border-fracture cliff without over-softening stop-motion clay texture (`STACK_SMOOTH_KNEE_NORM = 95.5 -> STACK_SMOOTH_CEIL_NORM = 98.5`).
- **`fl2va` `90f` Mode 2 (`gate1_fl_mode2_deslop_vidonly_1x_90f.png`) & `ref2va` `90f` Modes 1/3 (`gate3_*.png`):** `de_llm_slop +1.15` removes glossy warm-yellow bias while preserving lip-sync speech (`F0_delta = -0.014 st`), and `ref2va` `mid_late_20_49` cools/noir-tints background buildings (`border_sat 0.2254 -> 0.1901/0.1931`) while retaining natural blonde hair and warm skin tones on the subject (`center_vs_ref_sat_diff = 0.0515 / 0.0501`).

---

## 3. Verification Checks Run

- CPU gates: `verify_pole_a_erasure.py --cpu-only`, `check_control_vectors.py` (`C1–C7` incl. Stage 2 mock DiT hooks, clone isolation, FunControl `.previous` forwarding, and refusal gates), `check_auto_edit_prompt.py`, `check_exact_blocks.py`, `check_node_ids.py`, `check_schema_defaults.py`, `check_literal_widgets.py`, `check_doc_links.py`, `check_no_owner_paths.py`, and `check_commit_ready.py`.
- GPU suite: `bench/control_vector/verify_stage2_unified_steer_hooks.py` (exit `0`, `failures: []`).
