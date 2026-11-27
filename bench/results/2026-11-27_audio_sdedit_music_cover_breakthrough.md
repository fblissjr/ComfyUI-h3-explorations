# 2026-11-27 — 2-Pass Audio SDEdit Music-Cover Audit: Rhythm/Texture Win, Musical Pitch Failure

**Verdict: MIXED / FAILED MUSICAL PITCH BAR (Corrected `2026-11-27` after owner ear-audit + pitch-chroma verification).**
While **2-Pass Audio SDEdit** (`MiniMaxH3FreezeAudio` Pass 1 → `MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` + undistilled Base `ref2va` 16-step `euler/simple` at `denoise = 0.45–0.52`) beat `cond_audio` + `MiniMaxH3AudioRefRouter` on onset alignment (`onset_lag_x = 0..1`) and acoustic timbre (`ltas_cos_unplugged_b = 0.918–0.940`), **all 4 AI-song covers and the `s1`–`s3` shootout arms failed musical pitch/key fidelity** (`ai1` `chroma_sameKey_x = 0.129` / `+4 st` drift, `ai2` `0.099` / `+4 st`, `ai3` `0.188` / `+5 st`, `ai4` `-0.087` off-key; `s1` `0.113`, `s2` `0.086`, `s3` `0.064`).

**Three verified root causes (not yet isolated in this run):**
1. **`<Audio 1>` Drop-C# chord-drone pull:** `unplugged_b_solo_vocal_243f.wav` (*"All Apologies"*) rings a `C# (0.81) + G# (0.49)` drone across all steps, pulling `whatevercomes.wav` (`F#/G`) sharp (`cos_vs_ref_kurt = 0.632 → 0.757`) and pulling `violet.mp3` (`A = 0.71`) one semitone flat onto `G# (0.68)`.
2. **Unseparated pop drums/synths & octave mismatch in `x_0[audio]`:** Raw/HPSS mixes in a female octave forced `denoise = 0.45–0.52` (`σ_a0 = 0.71–0.77`) without isolating a clean male-octave vocal stem (`MelBand-RoFormer` + clean octave shift).
3. **`[Shot 1]` speaker misattribution (`00:00–00:03`):** Freezing a full pop mix onto the multi-person `unplugged_b_stage.png` without `MiniMaxH3QueryRouter` or solo-performer framing caused the background guitarist in the grey T-shirt to sing `[Shot 1]` while Kurt's mouth stayed closed until `00:04`.

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

| Arm | Method | Wall (s) | `med F0` (Hz) | `st vs B` | `chroma_sameKey` | `onset_x` (lag) | `ltas_cos B` | Native Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `s1_sdedit_d40` | 2-Pass Audio SDEdit `denoise=0.40` (`σ_a0=0.667`) | 216.9\* | 217.3 | +1.91 | **0.113** | 0.340 (`+1`) | 0.926 | Blues baritone + steel guitar on disco rhythm, **but misses original soprano pitch contour (`chroma_sameKey = 0.113` vs `b2`'s `0.462`, drones low)**. |
| `s2_sdedit_d50` | 2-Pass Audio SDEdit `denoise=0.50` (`σ_a0=0.750`) | 145.0 | 202.5 | +0.69 | **0.086** | 0.261 (`+1`) | 0.956 | Authentic blues voice & steel guitar on disco phrasing, **but fails pitch/melody lock (`chroma_sameKey = 0.086`, near `c0` `0.089` baseline)**. |
| `s3_sdedit_d60` | 2-Pass Audio SDEdit `denoise=0.60` (`σ_a0=0.828`) | 145.2 | 217.4 | +1.91 | **0.064** | 0.220 (`+1`) | 0.966 | Mellower blues contour, out-of-key (`chroma_sameKey = 0.064`). |
| `s4_router_s34` | Late-only `AudioRefRouter(split=34)` on `cond_audio` | 70.2 | 215.8 | +1.79 | 0.060 | 0.039 (`+3`) | 0.956 | Clean blues voice, lost disco rhythm/melody lock. |
| `s5_router_s38` | Late-only `AudioRefRouter(split=38)` on `cond_audio` | 69.8 | 295.6 | +7.23 | 0.083 | 0.146 (`+3`) | 0.892 | Strained elevated tenor, weak onset lock. |
| `s6_router_s42` | Late-only `AudioRefRouter(split=42)` on `cond_audio` | 69.5 | 208.5 | +1.19 | 0.110 | 0.166 (`0`) | 0.971 | Clean voice, still well below SDEdit onset lock and off-melody. |

\*`s1` included Pass 1 (`~72 s`), which ComfyUI cached automatically for `s2` and `s3`.

---

## 3. Real-World Zero-Memorization Validation: 3 User AI-Generated Songs Covered by MTV Unplugged

- **Harness:** [`bench/ai_song_sdedit_covers.py`](../ai_song_sdedit_covers.py) · `1024x768` (exact 4:3 MTV Unplugged), `seed = 910427311`, `<Audio 1>` (`unplugged_compressed.mp4` `24.0–34.125 s`, Drop-C# `C#/G#` drone), `<Picture 1,2>` stills · outputs under `$H3_OUTPUT_DIR/Video/ai_song_sdedit_covers_2026-11-27/` · report [`data/music_cover_gate/ai_song_sdedit_triage_report.json`](../../data/music_cover_gate/ai_song_sdedit_triage_report.json).

| Arm | Source AI Song (Zero Memorization) | Len | Pass-2 `denoise` | Wall (s) | `med F0` (Hz) | `chroma_sameKey` (`tr_st`) | `onset_x` (lag) | `ltas_cos B` | Native Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ai1_whatevercomes_d45` | `whatevercomes.wav` (lush female ballad, raw) | `243f` (`10.1 s`) | `0.45` (`σ_a0=0.710`) | 455.5 | 207.6 | **0.129 (`+4 st`)** | 0.257 (`0`) | 0.918 | Acoustic Kurt timbre & onset timing, **but pitch drifts sharp/off-key, pop drums/pads bleed through, and grey-shirt guitarist sings `[Shot 1]` (`00:00–00:03`)**. |
| `ai2_whatevercomes_d52` | `whatevercomes.wav` (lush female ballad, raw) | `243f` (`10.1 s`) | `0.52` (`σ_a0=0.766`) | 312.5\* | 206.4 | **0.099 (`+4 st`)** | 0.272 (`0`) | 0.938 | Cleaner unplugged texture, **but pitch still off-key (`+4 st` drift, flattens melody on line 2), subtle drum trace, and grey-shirt guitarist sings `[Shot 1]`**. |
| `ai3_idontwannacry_harm_d45` | `idontwannacry.wav` (fast female pop-punk, HPSS) | `192f` (`8.0 s`) | `0.45` (`σ_a0=0.710`) | 352.8 | 122.7 | **0.188 (`+5 st`)** | 0.382 (`+1`) | 0.922 | Tight rhythmic hits & grunge strumming, **but vocals sit off-key (`+5 st` drift) and grey-shirt guitarist sings `[Shot 1]`**. |
| `ai4_violet_v1_harm_d45` | `violet.mp3` (electro-indie verse 1, HPSS) | `192f` (`8.0 s`) | `0.45` (`σ_a0=0.710`) | 347.1 | 204.1 | **-0.087 (`+1 st`)** | 0.267 (`0`) | 0.940 | Unplugged texture, **but severely out-of-tune (`G#` Kurt drone clashes `1 st` against `A` song tonic) and grey-shirt guitarist sings `[Shot 1]`**. |

\*Pass 1 cached by ComfyUI from `ai1`.

---

## 4. Next Gate (Pitch-Lock Isolation Before Multi-Window Chaining)

1. **Do NOT ship raw/HPSS 2-Pass SDEdit as a working cover pipeline yet.**
2. Isolate and test the three suspected pitch/attribution confounders on `:8188`:
   - Strip `<Audio 1>`'s `C#/G#` guitar drone via `MelBandRoformer_fp16.safetensors` (and test minimal `±st` capo key-matching when full vocal+guitar `<Audio 1>` is kept),
   - Feed Song X as a clean octave-matched `MelBand-RoFormer` vocal stem (+ optional quiet in-key harmonic chord root bed),
   - Fix `[Shot 1]` speaker misattribution with solo-Kurt `<Picture>` framing + `MiniMaxH3QueryRouter`, and test `MiniMaxH3EditGuider(streams="audio_only")` on Pass 2.
