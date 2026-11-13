# Short timeline sound islands (`cond_audio`) and `<Picture 1> + <Video 1>` kinematic D-pads (2026-11-12)

Date: 2026-11-12

Follow-up to [`2026-11-12_guide_signal_full_length_bars.md`](2026-11-12_guide_signal_full_length_bars.md), which established two inverse facts on the 6-step Pika `t2va` zebra-crossing scene (`seed = 730451892`, `124` frames at `1344x768`):
1. A full-length `124`-frame visual guide (`cond` rows) overwrites the entire video canvas with the black-and-white bar signal even when released early.
2. The unguided audio stream (`x_audio`) spontaneously locked its acoustic events onto the held visual windows—raising the converse question: **can short timeline audio islands (`cond_audio`, `0` visual `cond` rows) lock sub-frame multi-actor visual choreography on `t2va` without touching the visual canvas, and how does 1D audio-island timing divide labor with `<Video 1>` reference signals on `ref2va`?**

Across two sound-island batches (`data/guide_sound_2026-11-12/`, 11 rendered clips), two `<Video 1>` reference batches (`data/ref_signal_2026-11-12/`, 6 rendered clips), two pack commits (`d19ba65c`, `e0a5bb35`), and the shipped workflow [`workflows/h3_text_to_video_pika_sound_guide_api.json`](../../workflows/h3_text_to_video_pika_sound_guide_api.json), every piece of that mechanism is now measured on disk and verified on 1:1 native pixel strips.

---

## 1. Architecture & node mechanics (`d19ba65c`, `e0a5bb35`)

### 1.1 Why `cond_audio` islands leave `100%` of the video canvas free
In `comfy/ldm/minimax/model.py` (`PackedLayout`), each audio-only keyframe entry attached by [`MiniMaxH3GuideLatent`](../../guide_latent.py) (`guide=None, audio=..., frame_idx=F`) appends **zero** visual `cond` rows (`cond_rows = 0`) and packs a short `cond_audio` segment of length $L_a = \text{round}(T_{\text{sec}} \cdot 40)$ (`40 Hz` Audio VAE latent tokens) at 1D temporal RoPE origin:
$$t_{\text{audio}} = \text{round}\!\left(F \cdot \frac{40}{24}\right)$$
When $F$ is a multiple of `3` video frames (`125 ms`), $t_{\text{audio}} = 5 \cdot (F / 3)$ lands on an exact integer Audio VAE token with zero rounding jitter (`F = 24 -> 40`, `F = 54 -> 90`, `F = 96 -> 160`). Inside the 50 joint AV DiT blocks, noisy video patches near video RoPE frame $F$ attend directly to those high-certainty `cond_audio` tokens (`seg_t["cond_audio"] = 1.0`) and place physical impacts to explain the acoustic transient.

### 1.2 Step release (`MiniMaxH3ReleaseReferences`) & 0-vs-K `streams="video_only"` (`MiniMaxH3EditGuider`)
Two runtime failure modes appeared during `exp1` and `exp2` and were solved in pack code:

1. **Audio-guide step release (`d19ba65c`, [`control_vector_nodes.py`](../../control_vector_nodes.py))**:
   - [`MiniMaxH3ReleaseReferences`](../../control_vector_nodes.py) previously dropped only visual `ref_img`/`cond` segments. Commit `d19ba65c` added `audio_end_percent` (default `1.0`) and `audio_clean_level` (default `1.0`), slicing `cond_audio` segments out of `PackedLayout` once `step_idx / total_steps >= audio_end_percent` while preserving `<Audio N>` speaker/reference segments (`ref_audio`) and keeping all generated video/audio 3D RoPE cursors intact (`_drop_audio_cond_from_layout`).
2. **0-vs-K keyframe-differential guidance & live `CFGGuider.sample` packed-tensor split (`d19ba65c` + `e0a5bb35`, [`edit_guidance.py`](../../edit_guidance.py))**:
   - [`MiniMaxH3EditGuider`](../../edit_guidance.py) now accepts conditioning pairs that share **identical** prompt tokens (`1x` Qwen3-VL 32B encode) and differ only in `minimax_keyframes` (`0` keyframes on `original` vs. `K` `cond_audio` islands on `edited`), automatically switching steps `> last_step` to 1-pass `[original]` once `audio_end_percent` has already dropped `cond_audio` on `[edited]` (`_release_to_original_after`, `10` DiT calls instead of `12` at `audio_end_percent = 0.60, last_step = 3`).
   - **Live `pack_latents` catch (`e0a5bb35`)**: Unit mocks pass `NestedTensor((v, a))` directly into `EditGuider.predict_noise`, whereas live `CFGGuider.sample` (`comfy/samplers.py`) packs `NestedTensor` latents into a single flat 3D tensor `[B, 1, C_flat]` (`comfy/utils.py::pack_latents`) *before* `sample_lcm` calls `predict_noise`, unpacking only after sampling finishes. Because `isinstance(out[0], NestedTensor)` was `False` during live sampling, pre-`e0a5bb35` `streams="video_only"` fell through to `out[0] + eff_v * (out[1] - out[0])` and extrapolated `a1` at `1.6x` anyway (`corr_with_input_yank_wav_raw = +0.8414`). Commit `e0a5bb35` resolves `inner_model.latent_shapes` via `comfy.utils.unpack_latents` / `pack_latents`, splicing `(v0 + eff_v * (v1 - v0), a0 + eff_a * (a1 - a0))` on the live flat tensor (`corr_with_input_yank_wav_raw = -0.0812`).

---

## 2. Sound-island experiments (`data/guide_sound_2026-11-12/`)

All clips run `124` frames (`1344x768`, `24 fps`) on `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `h3/pika_h3_flex_fl2va_svd995_bf16.safetensors`, `seed = 730451892`, outputs under `$H3_OUTPUT_DIR/Video/guide_sound_2026-11-12/`.

### 2.1 `exp1`: 3-hop `200 ms` `100 Hz` thuds (`[24, 60, 96]` vs. `+12f` `[36, 72, 108]`)
Recorded in `exp1_sound_record.json` and `exp1_sound_measure.json` (`guide_signal/thud_200ms.wav`, `8` Audio VAE tokens per island, pure `100 Hz` exponentially damped sine wave tapering to `0.0`):

| Arm | Guide islands (`frame_idx`) | Guider | `mean_brightness` | `40–250 Hz` peak frames | `median_dbfs` / `min_dbfs` | Visual & acoustic outcome |
|---|---|---|---:|---|---:|---|
| `hop_clean` | None | `BasicGuider` (`6` calls) | `104.21` | `[21, 22, 24, 43, 51, 64]` | `-38.06` / `-44.15` | Prompt asks for three hops, but tortoise hops **once** (`f26..29`) and plods flat on the road across `f50..123`. Continuous street rumble. |
| `thuds_a_rel` | `[24, 60, 96]` (`t_a=[40,100,160]`) | `BasicGuider` (`6` calls) | `113.48` | `[24, 25, 60, 61, 96, 97]` | **`-90.90` / `-93.62`** | **Zero visual canvas overwrite**; tortoise executes **all `3/3` hops** landing on `[24, 60, 96]`, but inter-island audio collapses to `-94 dBFS` digital silence. |
| `thuds_b_rel` | `[36, 72, 108]` (`+12f`, `t_a=[60,120,180]`) | `BasicGuider` (`6` calls) | `112.49` | `[9, 36, 45, 72, 108, 117]` | `-64.16` / `-94.76` | All three hops and low-frequency landings shift later by `+12` frames (`+0.50 s`). Inter-island audio still drops to `-95 dBFS`. |
| `thuds_a_diff_s1.6` | `0` vs. `[24, 60, 96]` | Core `CFGGuider` `1.6` (`both`) | `118.64` | `[9, 24, 46, 60, 96, 118]` | `-92.62` / `-94.87` | Higher hop clearance, but steering both streams (`a0 + 1.6*(a1 - a0)`) amplifies the `-95 dBFS` silent tails as well. |
| `thuds_b_diff_s1.6` | `0` vs. `[36, 72, 108]` | Core `CFGGuider` `1.6` (`both`) | `117.25` | `[9, 36, 45, 72, 108, 117]` | `-56.89` / `-95.42` | `+12f`-shifted counterpart of `thuds_a_diff_s1.6`. |

**Why `-94 dBFS` silence happened in `exp1`**: `thud_200ms.wav` decayed to mathematical zero (`0.0`) at the tail of each `8`-token `cond_audio` island. At `seg_t["cond_audio"] = 1.0`, the DiT interpolated between those three zero-floor boundaries and muted street ambience across steps `0..2`.

### 2.2 `exp2`: Multi-actor mid-hop leash yank + `e0a5bb35` `streams="video_only"` fix
To test coupled two-actor choreography (free hop 1 -> mid-air taut red leather leash yank slamming Hop 2 down early -> free rebound hop 3) AND solve both `-94 dBFS` muting and synthetic `.wav` bleed, `exp2` (`exp2_yank_record.json`, `exp2_measure.json`, [`bench/make_sound_guide_wavs.py`](../../bench/make_sound_guide_wavs.py)) introduced two changes:
1. **Single compound `300 ms` (`12`-token) foley island (`h3_guide_yank_slam_300ms.wav`) over a `-28 dBFS` street-rumble floor**:
   - Packing two separate back-to-back islands `3` frames apart (`t_a = 90` len `8` and `t_a = 95` len `8`) creates overlapping temporal RoPE positions (`90..97` vs. `95..102`). Encoding the two-actor collision inside **one** `300 ms` (`12`-token) clip keeps RoPE strictly monotonic: a taut leather-strap snap (`160–720 Hz` burst) at `t = 0 ms` (`frame_idx = 54`, `t_a = 90`) followed at `t = +125 ms` (`+5` audio tokens = **exact `+3` video frames**, `frame 57`, `t_a = 95`) by a `220 -> 88 Hz` slammed shell impact, both riding over a continuous `-28 dBFS` low-passed diesel rumble floor so `cond_audio` boundaries never touch digital zero.
2. **Post-`e0a5bb35` `MiniMaxH3EditGuider(streams="video_only", scale=1.6, last_step=3)` + `MiniMaxH3ReleaseReferences(audio_end_percent=0.60)`**:
   - Steers `x_video` along `v0 + 1.6 * (v1 - v0)` on steps `0..3` while holding `x_audio = a0` strictly on the `0`-keyframe `original` pass (`10` DiT calls total).

| Arm | Guide islands (`frame_idx`) | Guider (`DiT` calls) | `median_dbfs` (outside events) | Raw `.wav` bleed `r` (`corr_with_input_yank_wav_raw`) | `40–250 Hz` peaks | Measured trajectory & 1:1 visual inspection |
|---|---|---|---:|---:|---|---|
| `yank_clean` | None | `BasicGuider` (`6`) | `-33.74` (`-33.74`) | `-0.0114` | `[18, 61, 111]` | Uncontrolled drift (`f18, f61, f111`), no timed leash yank at `f54..57`. |
| `yank_a_rel` | `[24, 54, 96]` | `BasicGuider` (`6`) | **`-32.15` (`-33.37`)** | `+0.8586` | `[20,24, 54,57, 92,96]` | `-28 dBFS` floor **eliminates `-94 dBFS` silence** (`-33.37 dBFS` between events!), but synthetic `.wav` waveform copies directly into output audio (`r = +0.8586`). |
| `yank_b_rel` | `[36, 66, 108]` (`+12f`) | `BasicGuider` (`6`) | `-31.23` (`-31.89`) | `+0.8199` | `[7,11, 33,36, 66,69, 105,108]` | Whole 3-event compound sequence shifts `+12` frames cleanly (`54->57` moves to `66->69`). |
| `yank_a_vidonly_s1.6` *(pre-`e0a5bb35`)* | `0` vs. `[24, 54, 96]` | `EditGuider` pre-fix (`10`) | `-27.64` (`-28.57`) | `+0.8414` | `[20,24, 54,57, 92,96]` | Exposed the `CFGGuider.sample` `pack_latents` bug: `r = +0.8414` proved `a1` was still extrapolated before `e0a5bb35`. |
| **`yank_a_vidonly_fixed_s1.6`** *(post-`e0a5bb35`, shipped recipe)* | **`0` vs. `[24, 54, 96]`** | **`EditGuider` `video_only` (`10`)** | **`-30.76` (`-30.82`)** | **`-0.0812`** | **`[24, 57, 96]`** | **Zero synthetic `.wav` bleed (`r = -0.0812`)**, full `-30.8 dBFS` street ambience, and **100% organic H3 foley peaking on `[24, 57, 96]`** via `x_video -> x_audio` coupling! |

**1:1 native pixel inspection of `yank_a_vidonly_fixed_s1.6` (verified without downscaling)**:
- **Hop 1 (`f14..24`)**: Tortoise launches freely to **`+33.5 px` shell clearance** (`f17`, `8` frames airborne), red leather leash intact from right hand to collar, landing cleanly on `f22..24` right as native H3 audio hits `-6.8 dB` on **`f24`**.
- **Hop 2 + Leash Yank (`f48..57`)**: Tortoise launches at `f49`, reaches **`+25.8 px` at `f51` (`-23%` lower than Hop 1, `-31%` lower than Hop 3)**; across `f50..54` the red leather strap stays visibly connected from the man's raised right hand to the harness as his hand whips upward, pitching the shell nose-down onto the asphalt by **`f56` (`-10.3 dB`) / `f57` (`-8.4 dB`)**.
- **Hop 3 (`f88..98`)**: Full rebound hop to **`+37.6 px` clearance** (`f91`), red leash intact, landing on **`f96` (`-9.8 dB`)**.
- **Honest 1D-vs-2D boundary on `f56..57`**: Because `cond_audio` supplies 1D `(time, spectrum)` and zero 2D spatial vectors, at `f56..57` the man's raised forearm swings forward (`+x`, continuing his rightward walking stride) rather than pulling backward (`-x`), and the thin lower strap blurs briefly against dark asphalt during the fastest downward snap before re-forming cleanly.

---

## 3. Complementary `<Video 1>` reference experiments (`data/ref_signal_2026-11-12/`)

Run in parallel on `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `h3/pika_h3_flex_ref2va_svd995_bf16.safetensors` (`124` frames, `1344x768`, outputs under `$H3_OUTPUT_DIR/Video/ref_signal_2026-11-12/`):

### 3.1 `exp1_ref`: Abstract boolean flashes in `<Video 1>` do NOT transfer hop timing (`exp1_ref_measure.json`)
- Three arms (`ref_dark` solid black control, `ref_a` white flashes at `[20, 56, 92]`, `ref_b` white flashes at `[32, 68, 104]`) wired via `MiniMaxH3AppendRefVideo` + `MiniMaxH3ReferenceConditioning`.
- **Result**: Unlike `cond` guide rows, `<Video 1>` (`ref_vid` at negative temporal RoPE ahead of `t = 0`) caused **zero black-canvas bleed** (`black_bar_bleed_frames = 0`, `picture_mean_brightness = 93.8..94.2`). However, stationary on/off brightness flashes carry **zero vertical optical flow**, and Qwen3-VL samples `<Video 1>` at `2 fps` (`12`-frame stride, missing `ref_b`'s `[32, 68, 104]` windows completely): tortoise motion stayed locked to spontaneous text-driven hops around `f43..52` and `f100..111` across all three arms, while the DiT routed the `ref_vid` brightness pulses into `384 Hz` pedestrian-crossing beeps (`250–2000 Hz` inside/outside energy ratio `1.88x` on `ref_a`, `5.63x` on `ref_b`).

### 3.2 `exp2_dpad`: `<Picture 1>` actor anchor + `<Video 1>` 2D kinematic D-pad (`exp2_dpad_measure.json`)
- Replacing stationary flashes with a **`118x84` GrabCut RGBA cutout sprite of the tortoise + lit candle** (`<Video 1>`, `384x224`, `124` frames, fading ground contact shadow over a neutral `134 -> 106` overcast grey gradient; `MiniMaxH3AppendRefVideo(use_vae=True)`, `retention_analysis: attribute_transfer`) paired with a clean leash-free `384x288` `<Picture 1>` tortoise portrait (`MiniMaxH3AppendRefImage(size_policy="max", allow_upscale=False, qwen_view="separate")`, `retention_analysis: fully_preserved`) under one identical direction-neutral static-camera prompt turned `<Video 1>` into a **4-way `(LEFT, RIGHT, UP, DOWN)` + `-x` backward D-pad** with zero flat-grey backdrop bleed (`mean_brightness = 99.5..102.2`, `rgb_std = 49.4..50.1`, `tracked_frames = 124/124`) and organic native foley:

| Arm | `<Video 1>` cutout sprite command | Tortoise `cx` (`f00..10 -> f44..54 -> f110..123`) | Peak hop apex (`min_cy_frame`) | Native audio (`median_dbfs` / `audio_peak_frame`) |
|---|---|---|---|---|
| `dpad_right_earlyhop` | Walk `+x` (`0.22 -> 0.78`, `flip_x=False`) + `UP` arc `f30..42` (peak `f36`) | **`0.310 -> 0.398 -> 0.623`** (`+420 px` rightward, faces right) | **`f36`** (`cy = 0.523`, `0f` error; airborne `f31..41`) | `-39.66 dBFS` / landing thud at **`f43`** (`-13.46 dBFS`) |
| `dpad_left_latehop` | Walk `-x` (`0.78 -> 0.22`, `flip_x=True`) + `UP` arc `f78..90` (peak `f84`) | **`0.619 -> 0.526 -> 0.289`** (`-444 px` leftward, faces left) | **`f85`** (`cy = 0.522`, `+1f` vs. `f84`, `+49f` shift; airborne `f80..91`) | `-41.93 dBFS` / landing thud at **`f93`** (`-9.35 dBFS`) |
| `dpad_mid_reverse` | Walk `+x` `f0..48`, `UP` arc `f44..54` (peak `f49`), **pull `-x` backward** `f48..92` (`0.64 -> 0.26`, `flip_x=False`) | **`0.280 -> 0.468 -> 0.222`** (`-330 px` **facing-right `-x` backpedal**!) | **`f56`** (`cy = 0.590`, `+7f` lag vs. `f49`; airborne `f52..61`) | `-41.00 dBFS` / landing thud at **`f64`** (`-15.18 dBFS`) |

- **Why cutout sprite orientation (`flip_x`) matters**: A rotationally symmetric dot cannot separate facing-left locomotion (`flip_x=True`, Arm 2) from a facing-right `-x` backward pull/retreat (`flip_x=False`, Arm 3). Holding `flip_x=False` while reversing `cx` caused the rendered tortoise to glide backward leftward in a static stance across the zebra stripes while keeping its head facing right (`+x`) and its candle lit (with minor `+7f` jump lag on the simultaneous reversal, a 3-frame flame elongation on `f59..61`, and static-sprite leg lock during `f68..92` while `<Video 1>` remained packed through step `5`).

---

## 4. Division of labor & shipped workflows

For the follow-up `0.33` flow-midpoint gates (`Gate 1`, `Gate 2`, `Gate 2b`), zero-file nodes ([`MiniMaxH3SoundCues`](../../sound_cues.py) and [`MiniMaxH3AppendSpriteTrajectory`](../../sprite_trajectory.py)), and failure-mode guide, see **[`docs/research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md)**.

| Control task | Best mechanism | Shipped workflow | Why |
|---|---|---|---|
| **Exact sub-frame multi-event timing & collision intensity on `t2va`** (no reference image/video needed) | [`MiniMaxH3SoundCues`](../../sound_cues.py) (or chained `MiniMaxH3GuideLatent` `audio` islands on `3f` boundaries) + `MiniMaxH3ReleaseReferences(audio_end_percent=0.33)` on `BasicGuider` (`1x`, `6` passes), or `audio_end_percent=0.60` + `MiniMaxH3EditGuider(streams="video_only", scale=1.6, last_step=3)` (`10` passes) | [`workflows/h3_text_to_video_pika_sound_guide_api.json`](../../workflows/h3_text_to_video_pika_sound_guide_api.json) | Zero visual `cond` rows (`100%` open canvas), `1x` Qwen3-VL encode, zero `.wav` bleed (`r = -0.1578` at `0.33` `BasicGuider`; `r = -0.0086` at `0.60` `EditGuider`), and native AV sync via `x_video -> x_audio`. Cannot specify 2D `(-x vs +x)` vector direction. |
| **Continuous 2D `(x, y)` path, facing polarity, `-x` retreat, & timed jump** for a `<Picture 1>` subject | `<Picture 1>` (`MiniMaxH3AppendRefImage`) + `<Video 1>` [`MiniMaxH3AppendSpriteTrajectory`](../../sprite_trajectory.py) + [`MiniMaxH3SoundCues`](../../sound_cues.py) + `MiniMaxH3ReleaseReferences(end_percent=1.0, video_end_percent=0.33, audio_end_percent=0.60)` + `MiniMaxH3EditGuider(streams="video_only", scale=1.6, last_step=3)` | [`workflows/h3_image_ref_dpad_sound_pika_api.json`](../../workflows/h3_image_ref_dpad_sound_pika_api.json) | `<Video 1>` + touchdown `cond_audio` steer coarse trajectory, `-x` reversal, and jump apex across steps `0..1` (`b: 1.00 -> 0.50`), then `video_end_percent = 0.33` releases the static cutout so H3 articulates natural alternating backward leg strides across steps `2..5` while `<Picture 1>` holds identity at `1.0`. |
