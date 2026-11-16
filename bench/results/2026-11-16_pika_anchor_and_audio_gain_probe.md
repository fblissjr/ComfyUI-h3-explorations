# Pika 6-Step Gate 1 Probe — `AudioAttentionGain` Block Scoping & Perturbed Frame-0 Anchor Transfer

Date: 2026-11-16
Model: Pika H3 6-step (`MINIMAX_H3_MODEL_PIKA` + `PH3_VIDEO_VAE` + `MINIMAX_H3_ENCODER`)
Resolution / Length: `1344x768`, `length = 141` frames (`on_both_clocks`, `24.0 fps` / `5.875 s`)
Attention / Provenance: [`MiniMaxH3SageAttention`](../../nodes.py#L229) (`mode="auto"`, `qk_rotate=True`) + [`MiniMaxH3Sol`](../../sol_attn_h3.py) (`dense_blocks="49"`, `sink_conditioning="exact_kv_and_all_rows"`, `n_evals=6`, `n_sparse=5`, `sigma_start=0.979592`, `sigma_end=0.0`; all 10 [`MiniMaxH3ProvenanceStamp`](../../provenance.py) records report `sol.state="present"` and `encoder_arm.tokenizer.markers_resolved=7`, with `resolved={"error": "ValueError"}` because `sigmas` is unset in the test harness)
Seeds: Probe 1A `seed = 730451892`; Probe 1B Scout `seed = 730451892`, Continuation anchors `seed = 730451893`
Script: [`bench/probe_anchor_transfer.py`](../probe_anchor_transfer.py)
Summary & Outputs: `data/anchor_probe_2026-11-16/probe_summary.json`, `$H3_OUTPUT_DIR/Video/anchor_probe_2026-11-16/` (zero media committed to `data/`)

## Why This Gate Ran

Before changing chained-loop code ([`audio_freeze_song.py`](../../audio_freeze_song.py)), attention scaling ([`attention.py`](../../attention.py)), or `M4` release wiring ([Bucket 2 & Bucket 3 roadmap](../../docs/wiki/research/2026-11-16_geiru_keystone_audit_and_roadmap.md)), Gate 1 measured two upstream assumptions borrowed from [`roycho87/geiru_context_suite`](../../docs/wiki/upstream/periphery.md) on live `:8188` without editing custom nodes or restarting ComfyUI:

1. **Probe 1A (`AudioAttentionGain` block scoping):** Does restricting [`MiniMaxH3AudioAttentionGain`](../../audio_freeze.py#L681) from `blocks="all"` to early/mid blocks `blocks="0-29"` prevent visual collapse while driving rhythm-locked motion on frozen drum audio in 6-step Pika?
2. **Probe 1B (Frame-0 luma transfer & release):** How much of a `+29.5%` Frame-0 brightness perturbation propagates past Frame 0's 17-frame causal block (`f024..f140`), and how do [`MiniMaxH3GuideLatent`](../../guide_latent.py) (`frame_idx=0`, DiT-only) and [`MiniMaxH3Conditioning`](../../conditioning.py) (`first_frame` `i2va` + Qwen `<Picture 1>`) trade off exposure transfer, camera orbit freedom, and `M4` [`MiniMaxH3ReleaseReferences`](../../control_vector_nodes.py) release (`end_percent` in `{1.00, 0.50, 0.33}`)?

---

## Section 1 — Probe 1A: `AudioAttentionGain` Block Scoping on 6-Step Pika

Inputs: [`prompt_bank/t2va_studio_dancer_close.txt`](../../prompt_bank/t2va_studio_dancer_close.txt) + synthetic 120 BPM drum loop (`h3_probe_20261116_drum_120bpm.wav`) frozen with [`MiniMaxH3FreezeAudio`](../../audio_freeze.py) (`audio_mask=0.0`, `level="clip_guard"`). Graph node `"58"` (`ModelAttentionBackend`) was replaced by `MiniMaxH3SageAttention` so `apply_audio_gain` executes.

### Metrics (`probe_summary.json`)

| Arm | `key_gain` / `value_gain` | `blocks` | GPU / Wall (s) | Mean Frame Diff | On-Beat Motion | Off-Beat Motion | On/Off Ratio | Global Luma Mean +- Std | Beat Luma Pump Mean | Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p1a_gain_off` | `1.0` / `1.0` | `all` (inert) | `80.19` / `81.12` | `5.8068` | `6.4018` | `5.5047` | `1.1630` | `141.308` +- `10.649` | `0.3213` (`0.1776` on / `0.4717` off) | `p1a_gain_off_00001-audio.mp4` |
| `p1a_gain_all` | `2.0` / `1.5` | `all` | `57.68` / `58.62` | `2.5171` | `2.3954` | `2.5244` | `0.9489` | `45.701` +- `2.241` | `1.1578` (`1.0554` on / `1.2180` off) | `p1a_gain_all_00001-audio.mp4` |
| `p1a_gain_0_29` | `2.0` / `1.5` | `0-29` | `59.24` / `60.08` | `9.2486` | `8.1360` | `9.9049` | `0.8214` | `164.973` +- `9.573` | `3.0280` (`2.6378` on / `3.3855` off) | `p1a_gain_0_29_00001-audio.mp4` |

All three arms preserved identical frozen audio loudness (`rms_dbfs = -22.77`, `peak_dbfs = -0.01`).

### Visual Inspection (`probe1a_stacked.mp4` & `probe1a_stacked.1fps.00000-00140.mp4`)

- **Left (`gain_off`, `f000..f140`):** Clean, well-exposed dance studio framing. The dancer starts with hands on hips (`f000..f005`), snaps both arms outward horizontally in fists around beats (`f007..f018`, returns `f021..f025`, repeats `f031..f046`), leans forward (`f056..f071`), then cuts cleanly at `f072` to a close-up profile/mirror gaze holding steady from `f084..f140`. Minimal rolling-median exposure flicker (`0.3213`).
- **Center (`gain_all`, `f000..f140`):** Complete visual collapse across every single frame (`f000..f140`) into dark gray/brown static fabric-like noise (`global_luma_mean = 45.701`, zero studio or dancer structure).
- **Right (`gain_0_29`, `f000..f140`):** Scoping `AudioAttentionGain` to `blocks="0-29"` completely rescues coherent scene formation (`f000..f036` sparring/jabbing in front of multi-panel studio mirrors; `f038..f067` continuous spin with hair whipping horizontally instead of the `f072` jump cut; `f068..f140` direct mirror close-up). At `key_gain=2.0, value_gain=1.5`, however, even `0-29` brightens/washes highlights (`164.973` vs `141.308`, `+23.665` luma), induces `9.4x` larger rolling-median luma pumping (`3.0280` vs `0.3213`), and raises off-beat motion above on-beat motion (`on_off_beat_ratio = 0.8214` vs `1.1630` baseline).

### Code-Verified Root Cause Behind `blocks="all"` Collapse

Inspection of [`nodes.py`](../../nodes.py#L304), [`attention.py`](../../attention.py#L392), and [`sol_attn_h3.py`](../../sol_attn_h3.py#L693) shows two compounding mechanisms when `AudioAttentionGain` runs under `MiniMaxH3SageAttention` + `MiniMaxH3Sol(dense_blocks="49")`:

1. [`sol_take_forward`](../../nodes.py#L304-L307) calls [`apply_audio_gain`](../../attention.py#L573) **in-place** (`k.narrow(...).mul_(kg)`, `v.narrow(...).mul_(vg)`) before dispatching to `optimized_attention` ([`attention.py`](../../attention.py#L606)). Blocks `0..48` run `Sol` directly (`1x` gain).
2. For `dense_blocks=[49]`, `Sol`'s [`dense()`](../../sol_attn_h3.py#L693-L705) falls back to [`make_sage_override`](../../attention.py#L392), which calls `apply_audio_gain` a **second time** in-place on the same underlying storage — scaling Block 49's audio keys/values by `(kg^2, vg^2) = (4.0, 2.25)` on top of unguarded late-decoder scaling across blocks `30..48`.
3. `blocks="0-29"` bypasses blocks `30..49` completely (eliminating both late-block decoder overwrite and Block 49 double-scaling), explaining why `0-29` recovers coherent visuals while `all` collapses into uniform noise.

---

## Section 2 — Probe 1B: Perturbed Frame-0 Luma Transfer vs. Camera Orbit & `M4` Release

Inputs: [`prompt_bank/t2va_pika_balcony_orbit.txt`](../../prompt_bank/t2va_pika_balcony_orbit.txt). Frame `0` from `p1b_0_scout_t2v` (`f0_luma = 161.213`) was saved as `h3_probe_20261116_f0_clean.png` (`luma=161.21`) and multiplied by `1.3x` (`BRIGHT_GAIN`) to `h3_probe_20261116_f0_bright130.png` (`luma=208.83`, `+29.5%` lift). All six anchor arms shared continuation `seed = 730451893` (clean `p1b_1` vs. bright `p1b_2..p1b_6`). Body metrics exclude Frame 0's 17-frame causal block (`f024..f140`).

### Metrics (`probe_summary.json`)

| Arm | Mode / Policy | GPU (s) | `f0` Luma | Body Luma (`f24..140`) | vs. Clean Hold | `q1`..`q4` vs. Clean (%) | Mean L1 to `f0` (`f24..140`) | `f105` / `f140` L1 | Motion (`f24..140`) | Sharpness (Lap Var) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p1b_0_scout_t2v` | T2VA Scout (`seed`) | `67.68` | `161.213` | `147.814` | `+1.77%` | `+3.18 / +3.96 / -2.06 / +1.83` | `32.989` | `52.333` / `24.168` | `13.7881` | `268.90` |
| `p1b_1_guide0_clean_hold100` | `GuideLatent(0)`, `ep=1.00` | `63.87` | `158.607` | `145.250` | `0.00%` | `0.00 / 0.00 / 0.00 / 0.00` | `33.136` | `42.718` / `24.779` | `13.2673` | `296.08` |
| `p1b_2_guide0_bright_hold100` | `GuideLatent(0)`, `ep=1.00` | `64.23` | `206.027` | `182.337` | `+25.53%` | `+29.19 / +28.31 / +19.85 / +24.51` | `58.407` | `60.551` / `50.071` | `16.2454` | `422.49` |
| `p1b_3_guide0_bright_rel050` | `GuideLatent(0)`, `ep=0.50` | `62.40` | `202.860` | `181.171` | `+24.73%` | `+28.03 / +27.36 / +19.96 / +23.38` | `57.144` | `59.188` / `48.580` | `16.1749` | `416.06` |
| `p1b_4_guide0_bright_rel033` | `GuideLatent(0)`, `ep=0.33` | `61.80` | `196.792` | `178.734` | `+23.05%` | `+24.67 / +24.88 / +18.24 / +24.15` | `54.296` | `57.405` / `48.480` | `15.4570` | `364.49` |
| `p1b_5_i2va_bright_hold100` | `i2va` `first_frame`, `ep=1.00` | `75.49` | `205.997` | `182.752` | `+25.82%` | `+29.29 / +29.26 / +16.93 / +27.29` | `59.147` | `62.751` / `53.608` | `13.8135` | `418.20` |
| `p1b_6_i2va_bright_rel033` | `i2va` `first_frame`, `ep=0.33` | `64.58` | `205.304` | `182.987` | `+25.98%` | `+29.23 / +29.13 / +17.11 / +27.93` | `59.776` | `63.512` / `55.200` | `13.4742` | `395.96` |

### Visual Inspection (`probe1b_stacked.mp4`, `probe1b_stacked.1fps.00000-00140.mp4`, & `p1b_0_scout_t2v_00001-audio.mp4`)

- **Exposure & palette propagation without decay (`f000..f140`):** Every `+29.5%` bright arm (`p1b_2..p1b_6`) holds its brighter cream facade and lighter sky tones cleanly across the entire `141`-frame generation rather than snapping dark after Frame `16`. Even at `end_percent = 0.33` (`p1b_4`), tail quarter `q4_f111_140` retains `+24.15%` body brightness (`+23.05%` overall body lift — ~78% of the `+29.5%` input brightness differential), showing a single Frame-0 row transfers scene lighting/exposure cleanly across the DiT window.
- **`MiniMaxH3ReleaseReferences` (`M4`) effectiveness & release smoothness:** Both `guide0` (`p1b_2` vs `p1b_3` vs `p1b_4`) and `i2va` (`p1b_5` vs `p1b_6`) diverged cleanly when `end_percent` stepped down to `0.50` and `0.33`. No arm exhibited any visible exposure jump or frame pop anywhere in `f000..f140`. On `guide0`, releasing at `0.33` allowed subtle relaxation (`f0_luma` `206.027 -> 196.792`) and avoided the sharper edge halo/over-sharpening seen under `hold100` (`sharpness` `364.49` vs `422.49`, closer to clean baseline `296.08`).
- **Camera orbit freedom (`guide0` vs `i2va` `first_frame`):**
  - Across `f085..f112`, all four `GuideLatent(0)` arms (`p1b_1..p1b_4`) pushed/tilted upward to frame the top balcony purple-suit dancer (`f090..f102`) before pulling down to red-suit foreground (`f103..f110`) and widening back to the three-balcony exterior (`f115..f140`), with `mean_motion_body = 15.46..16.25` on bright `guide0` (`13.27` on clean `guide0`). Notice `+29.5%` global exposure did not suppress orbit travel vs clean `guide0`.
  - Both `i2va` arms (`p1b_5`, `p1b_6`) included Qwen `<Picture 1>` opening-frame semantics and stayed anchored lower on the middle red-suit balcony throughout `f085..f112` (`mean_motion_body = 13.81` / `13.47`), only catching the top balcony legs briefly around `f098..f105`, even after `end_percent = 0.33` release (`p1b_6`).

---

## Section 3 — Gate 1 Decisions Feeding Bucket 2 & Bucket 3

1. **`AudioAttentionGain` fix & scoping policy (Bucket 2):**
   - **Must fix in [`attention.py`](../../attention.py):** Add an idempotency guard so `sol_take_forward` -> `optimized_attention` -> `make_sage_override` never executes `apply_audio_gain` twice on `dense_blocks` (such as Block `49`), eliminating the `(kg^2, vg^2)` double-scaling trap.
   - **Recommended default block scope & sweep in [`audio_freeze_song.py`](../../audio_freeze_song.py):** Wire `AudioAttentionGain` with explicit `blocks="0-29"` default (excluding `30-49` where late-block overwrite occurs) and test milder gains (`1.15..1.4`) rather than `2.0 / 1.5` on 6-step Pika to get beat articulation without `+23.7` luma lift or `3.0` luma pumping.
2. **Global Frame-0 anchor & `M4` release policy (Bucket 2):**
   - **Gate 1 passed for opt-in Frame-0 exposure/palette anchoring:** A clean `1`-frame `GuideLatent(frame_idx=0)` row paired with [`MiniMaxH3ReleaseReferences`](../../control_vector_nodes.py) (`end_percent = 0.33`) transfers `+23.05%` (`+24.15%` in `q4_f111_140`) of a `+29.5%` Frame-0 brightness lift across a full `141`-frame Pika clip without Qwen `<Picture 1>` camera-orbit suppression (`p1b_4` `motion_body=15.46` vs `i2va` `13.47`) and with lower edge over-sharpening (`364.49` vs `422.49` `hold100`).
   - **`M4` policy finding:** On `i2va` (`first_frame` with `<Picture 1>`), releasing `cond_latents` at `end_percent=0.33` still preserved full `+25.98%` tail exposure while shaving over-sharpening (`418.20 -> 395.96`), whereas Qwen's visual prefix remained responsible for tighter shot framing. In [`audio_freeze_song.py`](../../audio_freeze_song.py), exposing opt-in Frame-0 anchor + `release_end_percent` (`0.33` default on Pika 6-step, `1.0` off) gives chained windows exposure stability without fighting shot-to-shot camera motion.
3. **Keystone Sink-Norm Balancing (Bucket 3):**
   - Because `MiniMaxH3SageAttention` + `MiniMaxH3Sol(dense_blocks="49")` ran every Probe 1A/1B render cleanly at `61.8..80.2 s` with `0` OOMs and verified Sol provenance (`n_sparse=5`), Bucket 3 can implement the clean-room dual-pass LSE sink-scaling kernel inside [`attention.py`](../../attention.py) directly on top of the validated `6`-step Pika probe suite.
