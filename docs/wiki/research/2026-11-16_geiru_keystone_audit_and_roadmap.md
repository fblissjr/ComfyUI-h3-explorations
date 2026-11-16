# Audit of `geiru_context_suite` / `NerdNotes` & Holistic Integration Plan (`2026-11-16`)

Clean-room technical evaluation of [`roycho87/geiru_context_suite`](https://github.com/roycho87/geiru_context_suite) (GPL-3.0), [`roycho87/NerdNotes`](https://github.com/roycho87/NerdNotes) (CC BY-SA 4.0), [`ckinpdx/ComfyUI-MMH3Tools`](https://github.com/ckinpdx/ComfyUI-MMH3Tools) (MIT), and [`ethanfel/ComfyUI-MiniMaxH3-Context-Loop`](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) (`Lumifel`), verified directly against the shared `1632x720`, `24.0 fps`, `4,629`-frame (`192.88 s`, `30`-clip) comparison render and an RTX 4090 PyTorch / SageAttention benchmark.

> **License boundary & attribution policy:** This pack ([`LICENSE`](../../../LICENSE)) is **MIT**. No GPL-3.0 code from `geiru_context_suite` or CC BY-SA 4.0 prose from `NerdNotes` is copied here. Every adopted concept is re-derived clean-room onto MIT repo primitives ([`attention.py`](../../../attention.py), [`audio_freeze.py`](../../../audio_freeze.py), [`audio_freeze_song.py`](../../../audio_freeze_song.py)) and credited in docstrings, tooltips, [`../upstream/periphery.md`](../upstream/periphery.md), and owning lane docs.

---

## 1. Empirical Audit of the `30`-Clip Comparison Render

Measured on the `1632x720` (`4 x 408x720` horizontal split, `32 kHz` stereo AAC) `30`-clip beach render (`192` frames in Clip 1 + `29 x 153` new stitched frames with `39`-frame `on_both_clocks` overlap = `4,629` total frames), evaluating full-frame ROI (`y = 42..668` excluding burned-in HUD labels), Laplacian variance sharpness, upper-cliff pixel distance to Clip 1 Frame `0`, and decoded PCM levels:

| Panel | Clips `11–30` Mean Luma vs. Clip 1 | Worst Clip (`2–30`) | Clip 30 Sharpness vs. Clip 1 | Mean Inter-Frame Motion (`clips 2–30`) | Visual & Audio State at Clip `20–30` |
|---|---:|---:|---:|---:|---|
| `0: No protection` | `-24.82%` | `-29.04%` (`clip 23`) | `+111.5%` (`136.8 -> 289.4`) | `14.853` | Fried shadows, severe sky color banding & vertical streaking |
| `1: Keystone + capstone` | `-15.76%` | `-21.17%` (`clip 26`) | `+23.8%` (`135.8 -> 168.2`) | `13.622` | Sawtooth exposure oscillation (`-9%` to `-21%`), horizontal sand fry around clip `28` |
| `2: Keystone + balance` | **`-7.43%`** | **`-8.78%`** (`clip 21`) | `+7.5%` (`135.8 -> 145.9`) | `9.929` | Exposure & color hold cleanly through Clip `30`; `~26%` motion damping |
| `3: Balance + ctx noise` | **`-7.81%`** | **`-9.25%`** (`clip 3`) | **`-1.7%`** (`137.4 -> 135.2`) | `10.112` | Tightest sharpness hold (`25.4%` intra-clip swing vs. `31.3%` in panel 1) |

### Verified Trade-Offs Observed in Playback & Measurements

1. **Groundhog-Day scenery lock on moving cameras:** Upper-cliff pixel `L1` distance relative to Clip 1 Frame `0` grows naturally inside Clip 1 as the shot moves (`0.00 -> 29.77` at `f185`), then **snaps back to Frame `0`'s cliff framing during Clip 2** (`29.26` at `f195` -> `5.92` at `f340` for `Keystone + balance`; `5.74` for `Balance + ctx noise`) and stays anchored near Frame `0` through `f4550` (`8.75`). Anchoring Frame `0` across all `30` clips holds illumination by pulling background geometry back to the anchor.
2. **Audio runaway from carried audio latents (`hold_audio`):** All four visual arms share the comparison audio track, which balloons from `-29.20 dBFS` RMS in Clip `1` (`2.21%` energy `>4 kHz`) to **`-11.58 dBFS` RMS** by Clip `15` (`+17.62 dB`, treble collapsing `25x` to `0.09%`), leaving warbling tonal distortion in Clips `20–30`. Pre-encoded master-track slicing + lossless PCM muxing ([`audio_freeze.py`](../../../audio_freeze.py), [`audio_freeze_song.py`](../../../audio_freeze_song.py)) completely avoids this failure mode.
3. **Motion damping (`~26%` lower inter-frame delta):** Mean inter-frame pixel delta across Clips `2–30` falls from `13.622` (`Keystone + capstone`) to `9.929` (`Keystone + balance`) and `10.112` (`Balance + ctx noise`).
4. **Silent PyTorch SDPA kernel fallback (`D_v = 136`):** Appending `2` indicator channels (`+ 6` zero padding channels) to `V` forces `D_qk = 128 != D_v = 136`, which PyTorch `FLASH_ATTENTION` and `CUDNN_ATTENTION` reject, silently falling back to `EFFICIENT_ATTENTION` + two extra subset calls.

---

## 2. Clean-Room Math & Kernel Benchmarks on This RTX 4090

### 2.1 Mass-Conserving Within-Slice Subset Balance via Native `return_lse=True`

Given two disjoint key index sets $A$ and $B$ in H3's single-stream packed attention sequence with shared total softmax probability $S = p_A + p_B$, shifting attention mass strictly between $A$ and $B$ by factor $b$ (`balance`) preserves all attention weights outside $A \cup B$ (prompt text, `<Picture>`/`<Video>` refs, audio, and target frames):

$$p'_A = \frac{(p_A + p_B)\, b\, p_A}{b\, p_A + p_B}, \qquad \text{out}' = \text{out}_{\text{full}} + (p'_A - p_A)(o_A - o_B)$$

Instead of mutating `V` to `D_v = 136`, our installed SageAttention fork (`sageattn_qk_int8_pv_fp8_cuda(..., qk_rotate=True, smooth_k=True, return_lse=True)`) natively returns natural-log $\text{lse}_{\text{all}}$ with `smooth_k` compensated (`mean |diff| = 0.00089` vs. fp32 exact on a `S = 512` check), letting FlashAttention-2 compute $(o_A, \text{lse}_A)$ and $(o_B, \text{lse}_B)$ cleanly at $D = 128$:

$$p_A = \exp(\text{lse}_A - \text{lse}_{\text{all}}), \qquad p_B = \exp(\text{lse}_B - \text{lse}_{\text{all}})$$

Measured on this machine's RTX 4090 (`S = 28,650` tokens, `56` heads, `head_dim = 128`, `bfloat16`, `|A| = 450`, `|B| = 2,700`):

| Implementation (Per DiT Block) | Time (ms) | Relative Cost vs. Plain Sage | Notes |
|---|---:|---:|---|
| Plain `sageattn_qk_int8_pv_fp8_cuda(qk_rotate=True)` | `57.59` | `1.00x` | Baseline [`attention.py`](../../../attention.py) kernel |
| **Sage `return_lse=True` + Flash subset passes** | **`88.19`** | **`1.53x`** (`5.89x` faster than Geiru) | Keeps `D=128`, `qk_rotate`, and `sageattn_consume` release |
| Standard bf16 PyTorch SDPA (`FlashAttention`) | `147.28` | `2.56x` | Standard dense attention baseline (`1,971 MiB` peak) |
| Flash `return_lse` (`aten._scaled_dot_product_flash_attention`) | `177.38` | `3.08x` | `0.00270` relative `L2` diff vs. fp32 exact |
| `V`-padded `D_v = 136` SDPA (`EFFICIENT_ATTENTION` fallback) | `519.03` | `9.01x` | Rejected by `FLASH_ATTENTION` & `CUDNN_ATTENTION` (`3,057 MiB` peak) |

Further verified at `S = 105,000` fused QKV tokens (`stride_seq = 21,504`, above the `99,864` `int32` crossing): `aten._scaled_dot_product_flash_attention` over fused `q` views matches contiguous `.contiguous()` tensors bit-identically (`0.0` tail `max diff` on both output and `lse`).

### 2.2 Causal Block Separation Corroboration

`NerdNotes`'s 1-clip perturbed-anchor sweep isolates brightness/exposure transfer in H3 blocks `31–33, 36–44, 47` (`38` & `44` strongest; `0–29` inert for brightness). This independently aligns with [`sparse/sol_dense_blocks_reanalysis.md`](sparse/sol_dense_blocks_reanalysis.md#L27-L36), which previously identified blocks `38–43` via Sol-Attn DuckDB telemetry as H3's diffuse global-integration cluster.

### 2.3 Causal VAE Seam-Lock Theorem (`c % 5 == 2`)

Reading ComfyUI's H3 Video VAE (`comfy/ldm/minimax/vae.py:597-602` & `comfy/ldm/minimax/vae.py:933-952`), temporal decoding uses `tokens_chunk_size = 5` and `token_overlap = 2`. Every `on_both_clocks` context window (`39f -> c = 12`, `90f -> c = 27`, `141f -> c = 42`) satisfies **`c % 5 == 2`**, so the chunk decoding the first newly generated frame (`frame = trim`) slices exactly `z[:, :, c - 2 : c + 5]`. Pinning the final two context latents (`c - 2`, `c - 1`) to mask `0.0` (while tapering earlier context latents smoothly) locks the causal decoder's overlap state identically across the cut without per-step sampler callbacks.

---

## 3. Five-Bucket Holistic Integration Roadmap

1. **Bucket 1 (Active, zero node edits):** Document clean-room upstream attribution; add [`bench/probe_anchor_transfer.py`](../../../bench/probe_anchor_transfer.py) to run `141f` 6-step Pika probes for block-scoped `MiniMaxH3AudioAttentionGain` (`0-29` vs. `all`) and perturbed Frame-0 release thresholds (`1.00`, `0.50`, `0.33`).
2. **Bucket 2 (Simplifies & closes queued `M4` + `S2`):** Tapered `context_noise` with `c - 2 .. c - 1` causal seam lock in [`MiniMaxH3FreezeAudioWindow`](../../../audio_freeze.py#L433), plus pre-decode clean context restore + `soft_empty_cache()` in [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py#L964-L975).
3. **Bucket 3 (Phase 3A/3B):** Timeline-scoped (`--- label`) section anchors in `MiniMaxH3AudioFreezeSong` with Gate 1's `end_percent` release so unmasked multi-minute song chains hold exposure without snapping across song sections.
4. **Bucket 4 (Shared DiT engine):** Native `return_lse=True` subset balance inside [`attention.py`](../../../attention.py) (`sageattn_consume` + Flash subset pre-slice) + automatic dense routing in [`sol_attn_h3.py`](../../../sol_attn_h3.py), serving `AudioAttentionGain`, `Ref2VA` (`M3`), and optionally Tier 2 song loops.
5. **Bucket 5 (Sync & gates):** Regenerate modified shipped workflows (`workflows/build_workflows.py`), indexes (`bench/build_index.py`), docs ([`../core/h3_audio_freeze.md`](../core/h3_audio_freeze.md), [`../decisions.md`](../decisions.md), [`../next_steps.md`](../next_steps.md)), and pass the full check suite (`bench/run_checks.py`).
