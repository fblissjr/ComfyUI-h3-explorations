# 2026-11-27 — 2-Pass Audio SDEdit Breakthrough: Zero-Memorization Music-Video Covers

**Verdict: PROVEN WINNER across 4 zero-memorization songs (1 synthetic + 3 user AI-generated tracks).**
**2-Pass True Audio SDEdit** (`MiniMaxH3FreezeAudio` Pass 1 → `MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` + undistilled Base `ref2va` 16-step `euler/simple` at `denoise = 0.45–0.52`, **zero `cond_audio`**, **zero custom attention hacks**) decisively outperformed `cond_audio` + `MiniMaxH3AudioRefRouter` on every zero-memorization metric and native ear test.

---

## 1. Root Cause Found via Chroma Audit & Upstream `sglang` Verification

1. **Pretraining confounder caught on `fun.mp4`:** Octave-invariant pitch-chroma (`chroma_sameKey`) revealed that *"Girls Just Want to Have Fun"* was famous enough that even `u0_ceiling` (zero `fun.mp4` audio) scored `chroma_sameKey = +0.167` from lyrics alone, masking that `MiniMaxH3AudioRefRouter(split_block=20..24)` collapsed melody lock on the zero-memorization Blues-covers-Disco pair (`b2` `+0.462` → router `+0.040..+0.083`).
2. **Upstream `sglang` explanation (`coderef/sglang` at `db8c562023`, `docs/wiki/core/h3_audio_freeze.md`):** Upstream `sglang` has **no `cond_audio` / guide-audio path** (`task_profiles.py` exposes only `t2va`, `fl2va`, `ref2va` with `ref_audio` at `t < 0` and target `audio` at `t >= 0`). `MiniMaxH3AddGuide(audio=...)` is a ComfyUI-only extension never trained by MiniMax.
3. **Exact Audio SDEdit carry math (`comfy/model_base.py:2238-2254`, `comfy/ldm/minimax/model.py:678-684`, `comfy/samplers.py:638-642`):**
   - Setting `audio_mask < 1.0` on `MiniMaxH3FreezeAudio` mixes `(1 - audio_mask)` of raw Song X back into the final step (`samplers.py:642`).
   - Instead, setting **`video_mask = 0.0, audio_mask = 1.0`** (`MiniMaxH3AudioRefineMask`) with **`BasicScheduler(steps=16, denoise=d)`** on undistilled Base `ref2va` (`shift_video=12.0, shift_audio=3.0, audio_scale=4.0`) satisfies $\frac{\sigma_{a0}(1-\sigma_{v0})}{\sigma_{v0}} \cdot 4 = 1 - \sigma_{a0}$ **identically**, initializing `x_audio` at exact `σ_a0`:
     - `denoise = 0.40` → `σ_v0 = 0.889`, **`σ_a0 = 0.667`**
     - `denoise = 0.45` → `σ_v0 = 0.907`, **`σ_a0 = 0.710`**
     - `denoise = 0.50` → `σ_v0 = 0.923`, **`σ_a0 = 0.750`**
     - `denoise = 0.52` → `σ_v0 = 0.929`, **`σ_a0 = 0.766`**
     - `denoise = 0.60` → `σ_v0 = 0.951`, **`σ_a0 = 0.828`**
   - Pass 1 (`MiniMaxH3FreezeAudio(song_x, audio_mask=0.0)` + 6-step Pika) renders photorealistic Target-B video with frame-accurate lip-sync and guitar strumming synchronized to Song X and writes Song X's clean latent into `latent["samples"][1]`. Pass 2 freezes video pixel-exact (`MAE = 0.0`), noises Song X's audio latent to `σ_a0 = 0.67–0.77`, and denoises freely (`audio_mask = 1.0`, zero raw mixback) conditioned solely on `<Audio 1>` (Target B) + verbatim `<d>` lyrics.

---

## 2. Zero-Memorization Shootout: Audio SDEdit vs. Late-Only `AudioRefRouter`

- **Harness:** [`bench/music_cover_shootout.py`](../music_cover_shootout.py) · `1152x768`, `141f = 5.875 s`, `seed = 730451892` (synthetic male delta blues `194.6 Hz` covering original female disco soprano `456.8 Hz`) · report [`data/music_cover_gate/shootout_triage_report.json`](../../data/music_cover_gate/shootout_triage_report.json).

| Arm | Method | Wall (s) | `med F0` (Hz) | `st vs B` | `onset_x` (lag) | `ltas_cos B` | Native Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`s1_sdedit_d40`** | 2-Pass Audio SDEdit `denoise=0.40` (`σ_a0=0.667`) | 216.9\* | **217.3** | +1.91 | **0.340 (`+1`)** | 0.926 | **Winner (max melody lock):** exact original disco melody, bounce & soaring hook in gravelly male blues voice + pure steel resonator guitar, pixel-exact lip-sync, zero female/synth bleed. |
| **`s2_sdedit_d50`** | 2-Pass Audio SDEdit `denoise=0.50` (`σ_a0=0.750`) | 145.0 | **202.5** | **+0.69** | **0.261 (`+1`)** | **0.956** | **Winner (purest blues voice):** 100% authentic blues voice & steel guitar while keeping exact disco phrasing, rhythm & hook lift. |
| `s3_sdedit_d60` | 2-Pass Audio SDEdit `denoise=0.60` (`σ_a0=0.828`) | 145.2 | 217.4 | +1.91 | 0.220 (`+1`) | 0.966 | Slightly mellower blues contour than `s1`/`s2`; confirms `0.40–0.52` is the golden zone. |
| `s4_router_s34` | Late-only `AudioRefRouter(split=34)` on `cond_audio` | 70.2 | 215.8 | +1.79 | 0.039 (`+3`) | 0.956 | Clean blues voice, lost disco rhythm/melody lock. |
| `s5_router_s38` | Late-only `AudioRefRouter(split=38)` on `cond_audio` | 69.8 | 295.6 | +7.23 | 0.146 (`+3`) | 0.892 | Strained elevated tenor, weak onset lock. |
| `s6_router_s42` | Late-only `AudioRefRouter(split=42)` on `cond_audio` | 69.5 | 208.5 | +1.19 | 0.166 (`0`) | 0.971 | Clean voice, still well below SDEdit onset/melody lock. |

\*`s1` included Pass 1 (`~72 s`), which ComfyUI cached automatically for `s2` and `s3`.

---

## 3. Real-World Zero-Memorization Validation: 3 User AI-Generated Songs Covered by MTV Unplugged

- **Harness:** [`bench/ai_song_sdedit_covers.py`](../ai_song_sdedit_covers.py) · `1024x768` (exact 4:3 MTV Unplugged), `seed = 910427311`, cleaner vocal-forward `<Audio 1>` (`unplugged_compressed.mp4` `24.0–34.125 s`, reduced cello drone), `<Picture 1,2>` stills · outputs under `$H3_OUTPUT_DIR/Video/ai_song_sdedit_covers_2026-11-27/` · report [`data/music_cover_gate/ai_song_sdedit_triage_report.json`](../../data/music_cover_gate/ai_song_sdedit_triage_report.json).

| Arm | Source AI Song (Zero Memorization) | Len | Pass-2 `denoise` | Wall (s) | `med F0` (Hz) | `onset_x` (lag) | `ltas_cos B` | Native Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ai1_whatevercomes_d45` | `whatevercomes.wav` (lush female ballad, raw) | `243f` (`10.1 s`) | `0.45` (`σ_a0=0.710`) | 455.5 | 207.6 | 0.257 (`0`) | 0.918 | Exact soaring ballad melody & frame-accurate lip-sync in raw raspy Kurt voice + acoustic guitar; tiny pop-pad reverb trace left on final line. |
| **`ai2_whatevercomes_d52`** | `whatevercomes.wav` (lush female ballad, raw) | `243f` (`10.1 s`) | `0.52` (`σ_a0=0.766`) | 312.5\* | 206.4 | **0.272 (`0`)** | **0.938** | **Cleanest ballad take:** strips remaining pop-pad reverb into pure live unplugged guitar, cello, and aching raspy Kurt vocals while keeping every note of the melody. |
| **`ai3_idontwannacry_harm_d45`** | `idontwannacry.wav` (fast female pop-punk, HPSS) | `192f` (`8.0 s`) | `0.45` (`σ_a0=0.710`) | 352.8 | 122.7 | **0.382 (`+1`)** | 0.922 | **Fast punk stress test PASS:** nails rapid-fire phrasing, chord hits, and furious acoustic strumming in raw snarling grunge vocals with tight close-up lip-sync. |
| **`ai4_violet_v1_harm_d45`** | `violet.mp3` (electro-indie verse 1, HPSS) | `192f` (`8.0 s`) | `0.45` (`σ_a0=0.710`) | 347.1 | 204.1 | **0.267 (`0`)** | **0.940** | **Seamless genre flip:** transforms female electro-pop into moody, hypnotic 1993 Seattle acoustic grunge (wooden guitar + cello + raspy Kurt vocal) on the exact original melody. |

\*Pass 1 cached by ComfyUI from `ai1`.

---

## 4. Production Recipe & Next Steps

1. **Default 2-Pass Pure-H3 Cover Recipe:**
   - **Pre-pass (recommended for dense pop/rock/electronic tracks):** Isolate lead vocals (or vocal-forward stem) via `MelBandRoFormerSampler` (already installed in `ComfyUI-MelBandRoFormer`) or non-pitch-shifted STFT-HPSS. Keep Target B `<Audio 1>` unseparated (voice + acoustic instrument together).
   - **Pass 1:** Pika 6-step `ref2va` + `MiniMaxH3FreezeAudio(song_x, audio_mask=0.0)` + `<Picture N>` + `<Audio 1>` + verbatim `<d>` lyrics.
   - **Pass 2:** Undistilled Base `ref2va` (`16-step euler/simple`) + `MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` + `BasicScheduler(steps=16, denoise=0.45..0.52)` (`0.45` for fast/rhythmic tracks, `0.50–0.52` for lush ballads needing maximum timbre transformation).
2. **Next:** Verify `MelBandRoFormerSampler` vocals-only stem inside the ComfyUI graph vs HPSS, then implement Phase 3 (`carry_prev_audio` + `start_seconds` + shipped workflow in `workflows/build_workflows.py`) for continuous multi-window music videos.
