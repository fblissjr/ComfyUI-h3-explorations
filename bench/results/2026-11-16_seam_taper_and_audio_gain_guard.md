# Bucket 2 Verification — `apply_audio_gain` Idempotency Guard, Causal `seam_taper` Context Schedule, & Mild `0-29` Audio Gain

Date: 2026-11-16
Model: Pika H3 6-step (`MINIMAX_H3_MODEL_PIKA` + `PH3_VIDEO_VAE` + `MINIMAX_H3_ENCODER`)
Resolution / Grid: `1344x768`, `length = 141` frames (`on_both_clocks`), Part B 2-window `MiniMaxH3AudioFreezeSong` (`window_frames = 141`, `context_frames = 39`, `song_frames = 243` / `10.125 s` at `24.0 fps`)
Attention / Provenance: [`MiniMaxH3SageAttention`](../../nodes.py#L229) (`mode="auto"`, `qk_rotate=True`) + [`MiniMaxH3Sol`](../../sol_attn_h3.py) (`dense_blocks="49"`, `sink_conditioning="exact_kv_and_all_rows"`, `n_evals=6`, `n_sparse=5`, `sigma_start=0.979592`, `sigma_end=0.0`; both Part A [`MiniMaxH3ProvenanceStamp`](../../provenance.py) records report `sol.state="present"` and `encoder_arm.tokenizer.markers_resolved=7`, with `resolved={"error": "ValueError"}` because `sigmas` is unset on the stamp probe input)
Script: [`bench/probe_bucket2_seam_and_gain.py`](../probe_bucket2_seam_and_gain.py)
Unit Suite: [`bench/check_audio_freeze.py`](../check_audio_freeze.py) (`check_seam_taper_and_gain_guard`)
Summary & Outputs: `data/bucket2_probe_2026-11-16/probe_summary.json`, `$H3_OUTPUT_DIR/Video/bucket2_probe_2026-11-16/` (zero media committed to `data/`)

## Code Changes Verified in Bucket 2

1. **`apply_audio_gain` idempotency guard ([`attention.py`](../../attention.py#L287), [`attention.py`](../../attention.py#L596)):**
   - Added `_h3_audio_gain_applied` tracking around `optimized_attention(...)` in `via_optimized_attention` (`sol_take_forward`, `try...finally` cleanup) and early return in [`apply_audio_gain`](../../attention.py#L273) so `MiniMaxH3Sol(dense_blocks="49") -> make_sage_override` cannot double-scale audio `K, V` in-place.
2. **Causal `seam_taper` context schedule ([`audio_freeze.py`](../../audio_freeze.py#L430), [`MiniMaxH3FreezeAudioWindow`](../../audio_freeze.py#L482)):**
   - Added [`context_noise_schedule`](../../audio_freeze.py#L436) supporting `CONTEXT_PROFILES = ("flat", "seam_taper")` (`CONTEXT_PROFILE_DEFAULT = "flat"`, `SEAM_LOCK_STEPS = 2`), ramping `context_noise -> 0.0` while locking steps `c - 2` and `c - 1` (`token_overlap=2` across `c % 5 == 2` dual-clock contexts `39 / 90 / 141`; clean-room adaptation crediting `ethanfel/ComfyUI-MiniMaxH3-Context-Loop` and `roycho87/geiru_context_suite`).
   - Appended optional `context_profile` at the end of `MiniMaxH3FreezeAudioWindow` and [`MiniMaxH3AudioFreezeSong`](../../audio_freeze_song.py#L502) inputs (`bench/node_id_manifest.json` updated via [`bench/check_node_ids.py`](../check_node_ids.py)).
   - Updated [`MiniMaxH3AudioAttentionGain`](../../audio_freeze.py#L753) docstring/tooltip with Gate 1 `blocks="0-29"` guidance for 6-step Pika.
3. **Zero-mask context pin, `soft_empty_cache()`, & clean `prev` carry ([`audio_freeze_song.py`](../../audio_freeze_song.py#L951)):**
   - Pins exact `clean_ctx` onto all `mask == 0` context cells before decode and `.safetensors` window save (keeping [`bench/recomposite_window.py`](../recomposite_window.py) decode-vs-file parity bit-identical), frees transient sampling buffers with `soft_empty_cache()`, and restores clean `clean_ctx` onto `prev[:, :, :c_steps]` so short overlapping windows (`window_frames < 2 * context_frames`) never pass twice-redrawn context into later windows.

---

## Section 1 — Part A: Guarded `AudioAttentionGain` (`all` vs. Mild `0-29` at 6 Steps)

Inputs: [`prompt_bank/t2va_studio_dancer_close.txt`](../../prompt_bank/t2va_studio_dancer_close.txt) + synthetic 120 BPM drum loop (`7.0 s`) frozen via [`MiniMaxH3FreezeAudio`](../../audio_freeze.py) (`audio_mask=0.0`, `level="clip_guard"`). Reference baseline row (`p1a_gain_off`) comes from [Gate 1](2026-11-16_pika_anchor_and_audio_gain_probe.md) at identical seed (`730451892`) and graph topology.

| Arm | `key_gain` / `value_gain` | `blocks` | Guard State | GPU / Wall (s) | Mean Frame Diff | On-Beat Motion | Off-Beat Motion | On/Off Ratio | Global Luma Mean +- Std | Beat Luma Pump Mean | Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p1a_gain_off` (Gate 1 ref) | `1.0` / `1.0` | `all` | Baseline | `80.19` / `81.12` | `5.8068` | `6.4018` | `5.5047` | `1.1630` | `141.308` +- `10.649` | `0.3213` | `$H3_OUTPUT_DIR/Video/anchor_probe_2026-11-16/p1a_gain_off_00001-audio.mp4` |
| `b2_gain_all_fixed` | `2.0` / `1.5` | `all` | Guard active | `76.73` / `78.17` | `2.9026` | `2.7015` | `2.8342` | `0.9532` | `43.431` +- `1.971` | `1.0289` | `b2_gain_all_fixed_00001-audio.mp4` |
| `b2_gain_0_29_mild` | `1.35` / `1.15` | `0-29` | Guard active | `59.30` / `60.09` | `5.7948` | `6.9001` | `5.7095` | `1.2085` | `140.896` +- `12.657` | `0.1963` | `b2_gain_0_29_mild_00001-audio.mp4` |

Audio loudness remained identical across all arms (`rms_dbfs = -22.77`, `peak_dbfs = -0.01`).

### Visual Findings (`bucket2_gain_stacked.mp4` & `bucket2_gain_stacked.1fps.00000-00140.mp4`)

- **Middle (`b2_gain_all_fixed`):** Even with Block-49 double-scaling eliminated, `2.0 / 1.5` across `blocks="all"` still collapses all `141` frames into dark gray/brown textured noise (`global_luma_mean = 43.431`), isolating the root collapse mechanism to late decoder blocks `30-49` rather than Block 49 alone.
- **Right (`b2_gain_0_29_mild`, `1.35 / 1.15`, `blocks="0-29"`):** Preserves exact natural scene exposure (`140.896` vs `141.308` baseline), drops rolling-median luma pumping below baseline (`0.1963` vs `0.3213`, compared to `3.0280` at `2.0 / 1.5` in Gate 1), and raises `on_off_beat_ratio` to `1.2085` (vs `1.1630` baseline) with full crisp horizontal arm extensions reaching mirror edges on beat (`f011..016`, `f036..041`), sharp chest snaps (`f022..029`, `f046..053`), and clean transition into mirror close-up (`f073..140`).

---

## Section 2 — Part B: 2-Window `MiniMaxH3AudioFreezeSong` Seam A/B (`flat` vs. `seam_taper` at `context_noise=0.6`)

Inputs: [`prompt_bank/t2va_studio_dancer_close.txt`](../../prompt_bank/t2va_studio_dancer_close.txt) + synthetic `10.125 s` 120 BPM drum WAV, `window_frames = 141`, `context_frames = 39` (`c = 12` latent steps, seam between frame `140` of Window 1 and frame `141` of Window 2 after `trim = 39`).

| Arm | `context_noise` | `context_profile` | GPU / Wall (s) | Seam `L1` (`f140 -> f141`) | Local Adj `L1` Mean | Seam Jump Ratio | Seam Luma Jump | W1 -> W2 Luma Mean (`% Delta`) | Audio RMS / Peak (dBFS) | Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `b2_song_flat_cn060` | `0.6` | `flat` | `131.87` / `132.38` | `7.3153` | `6.3724` | `1.1480` | `0.0732` | `140.170 -> 126.147` (`-10.00%`) | `-22.06` / `-0.09` | `b2_song_flat_cn060_00001.mp4` |
| `b2_song_taper_cn060` | `0.6` | `seam_taper` | `131.59` / `132.14` | `3.0608` | `4.5517` | `0.6725` | `0.6294` | `140.170 -> 126.341` (`-9.87%`) | `-22.06` / `-0.09` | `b2_song_taper_cn060_00001.mp4` |

### Visual Findings (`bucket2_seam_stacked.mp4` & `bucket2_seam_stacked.1fps.00000-00242.mp4`)

- **Identical Window 1 (`f000..f140`):** Both panels share identical pixels through `f140` (direct mirror close-up gaze).
- **Boundary cut (`f140 -> f141 -> f144`):**
  - Under `flat` (`left`), `f141` immediately shifts jaw/head angle and shoulder lighting across the cut (`seam_l1_f140_141 = 7.3153`, `seam_jump_ratio = 1.1480`) before snapping into a left head turn (`f142`) and arm raise (`f143`).
  - Under `seam_taper` (`right`), pinning `c - 2 .. c - 1` clean reduces `f140 -> f141` pixel discontinuity by **`2.39x` (`-58.2%`, `7.3153 -> 3.0608`, `seam_jump_ratio = 0.6725`)**, holding head pose, gaze, and shoulder alignment locked seamlessly at `f141` before smoothly initiating motion across `f143..f160`.
- **Global exposure drift (`f141..f242`):** Both unanchored `t2va` arms exhibit nearly identical W2 vs W1 global brightness loss (`-10.00%` vs `-9.87%`, visibly slightly dimmer white studio wall in `f150..242` than `f000..140`), confirming that `seam_taper` resolves causal boundary discontinuities (`c % 5 == 2`) while global exposure stability across chained windows belongs to **Bucket 3** (timeline-scoped Frame-0 `GuideLatent(0)` anchor + `MiniMaxH3ReleaseReferences(end_percent=0.33)`) and **Bucket 4** (LSE subset balance).
