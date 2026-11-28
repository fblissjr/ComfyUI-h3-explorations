# Pure-H3 Stem Cascade (`c1`–`c9`) & 4-Window `44.125 s` Chained Cover Song (`2026-11-27`)

- **Verdict:** On the 4-window (`4 x 294f`, `1059f`, `44.125 s`, `8.875..53.000 s`) pure-H3 Verse+Chorus cover of `$H3_INPUT_DIR/justaflicker.mp3` at `denoise = 0.29` (`41` schedule steps, $\sigma_{\text{aud}} = 0.5538$, `-14.5 dB` harmonic / `-12.0 dB` bass scaffold, stock-RoPE `TrimAudioDuration + AudioConcat` handoff with zero `MiniMaxH3GuideLatent` collision), `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/kurt_unplugged_justaflicker_verse_chorus_44s-audio.mp4` holds zero key drift (`best_tr_st = 0`, `chroma_drift_st = 0`), **`med_abs_cent_err = 17.9 c`** (`in_tune_frac_50c = 0.704`, `voiced_overlap_frac = 0.172` [`419/2439`], `vocal_f0_r_one_shift = 0.510`), beating **`100%` (`169/169`)** of time-shifted nulls on all four untransposed and transposed error/in-tune columns (`lag_null`), and holding **`37.7 c`** (`42.2%` overlap, **`169/169`** shifts beaten) under relaxed `threshold = 0.28` YIN! Across `c1`–`c9`, `c1` (`d=0.26`) passes `33/33` null shifts cleanly with a smoother baritone vocal, while `c3` / `c6` / `c7` (`d=0.30`) and `song_w1`–`song_w4` (`d=0.29`) trade lower strict YIN periodicity for gravelly Kurt vocal fry and chord accompaniment in vocal rests. Two prior overclaims (`c8` having audible acoustic guitar strumming, and raw `ltas_cos_kurt_unplugged` measuring singer conversion) are explicitly withdrawn below.
- **Date:** `2026-11-27`
- **Extends & Cross-Links:**
  - [`2026-11-27_music_cover_pitch_lock_gate.md`](2026-11-27_music_cover_pitch_lock_gate.md) (`2f8723f0`)
  - [`2026-11-27_music_cover_pitch_lock_metric_null.md`](2026-11-27_music_cover_pitch_lock_metric_null.md) (`8515059a`, `c6e2125b`)
  - [`2026-11-27_music_cover_pitch_coverage.md`](2026-11-27_music_cover_pitch_coverage.md) (`4ec035d3`–`b4fa3341`, independent coverage, bass-rest, high-pass, and `ltas` controls)
- **Scripts & Checks:** [`bench/h3_stem_sep.py`](../h3_stem_sep.py), [`bench/check_h3_stem_sep.py`](../check_h3_stem_sep.py), [`bench/pure_h3_cover_song.py`](../pure_h3_cover_song.py), [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py), [`bench/pitch_lock_coverage.py`](../pitch_lock_coverage.py)
- **Media & Reports:** Outputs under `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/`; JSON/prompt data under [`data/pure_h3_cover_song/`](../../data/pure_h3_cover_song/); cascade stems under `$H3_INPUT_DIR/music_cover_stems/`

---

## 1. Native GPU Stem Cascade & On-Disk Null Tests (`bench/h3_stem_sep.py`)

[`bench/h3_stem_sep.py`](../h3_stem_sep.py) separates 44.1 kHz stereo mixes natively on `:8188`-aware CUDA (`POST /free` when idle, `< 1.6 GiB` VRAM) with `MelBandRoformer_fp16` (`vocals` / residual `inst`) followed by `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS` (`drums`, `bass`, `harmonic = inst - drums - bass`), quantizing the PCM16 residual onto `mix.wav` so all 4 exported stems sum cleanly back to `mix.wav` (offline [`bench/check_h3_stem_sep.py`](../check_h3_stem_sep.py) passes 4/4 cases).

| Song | Duration (`s`) | `mix.wav` Null Depth (`dB`) | `mix` Max Abs Diff | `no_drums.wav` Bed Null (`dB`) | Bed Max Abs Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `justaflicker.mp3` | `162.80` | `-224.001` | `0.0` | `-79.965` | `3.1e-05` (`1` PCM16 LSB) |
| `thatsnotakite.mp3` | `163.58` | `-223.832` | `0.0` | `-79.700` | `3.1e-05` (`1` PCM16 LSB) |
| `inthefridge.mp3` | `175.04` | `-223.712` | `0.0` | `-78.097` | `3.1e-05` (`1` PCM16 LSB) |

- **Scaffold recipes (`librubberband` `pitchq=quality`):**
  - `justaflicker` (`-5.0 st`, `vocal_formant=False` = `0.749x` male vocal-tract scaling): calibration arms `c1`–`c7` use `harmonic_db = -9.5, bass_db = -13.0`; `--song` uses **`harmonic_db = -14.5, bass_db = -12.0`** so `harmonic.wav` anchors the `w1` intro (`8.875..10.500 s`: `vocals = -240 dB`, `harmonic = -21.0 dB`, `bass = -58.7 dB`) and `w3` turnaround (`36.2..42.375 s`: `harmonic = -21.7 dB`) while giving Pass-2 diffusion more room for live acoustic chords during sung phrases.
  - `thatsnotakite` (`c8`, `-6.0 st`) & `inthefridge` (`c9`, `+2.0 st`): zero-synth test (`harmonic_db = -120.0`, `bass_db = -11.0`, `vocals + bass` only). Note: in [`2026-11-27_music_cover_pitch_coverage.md`](2026-11-27_music_cover_pitch_coverage.md) §6.2 & native playback of `$H3_INPUT_DIR/music_cover_stems/kite_raw_orig_243f.wav` and `$H3_INPUT_DIR/music_cover_stems/fridge_raw_orig_243f.wav`, both source choruses carry multi-tracked gang harmonies and overlapping spoken/shouted ad-libs (*"OH! Swing it!"*, *"Hahaha! / Keep it fresh!"*), making their isolated `vocals.wav` targets largely non-periodic under single-voice YIN even before shifting (`0.313` and `0.243` active periodicity).

---

## 2. Noise-Schedule Discretization, Formant Scaling, & Meter Controls

Under `ModelSamplingAV(shift_video=12.0, shift_audio=3.0)`, `BasicScheduler(steps=12, denoise=d)` computes `total_steps = int(12 / d)`, mapping nominal `denoise` to discrete actual audio noise $\sigma_{\text{aud}} = \frac{3 d_{\text{act}}}{1 + 2 d_{\text{act}}}$:

| Nominal `denoise` | `total_steps = int(12 / d)` | Actual `denoise` | Start $\sigma_{\text{aud}}$ | Start $\sigma_{\text{vid}}$ | Role |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0.26` | `46` | `0.2609` | `0.5143` | `0.8090` | `c1` (tight pitch lock, smoother baritone) |
| `0.29` | `41` | `0.2927` | `0.5538` | `0.8324` | `--song` (`w1`–`w4`, `44.125 s`) |
| `0.30` | `40` | `0.3000` | `0.5625` | `0.8372` | `c3`–`c9` (1 schedule step from `0.29`) |
| `0.32` | `37` | `0.3243` | `0.5902` | `0.8521` | `c2` (`0.28` YIN cliff) |

- **Meter Calibration Controls (`cal_triage_report.json`):**
  - **Positive `AudioVAE` round-trip (`flicker_m5_verse_cal_243f.wav` -> `encode`/`decode` -> `MelBand-RoFormer`):** `vocal_med_f0_hz = 174.67 Hz`, `voiced_overlap_frac = 0.995` (`624/627`), `med_abs_cent_err = 2.0 c`, `in_tune_frac_50c = 0.997`, `best_tr_st = 0`, `vocal_f0_r_one_shift = 0.981`, beating `33/33` (`1.00`) `lag_null` shifts across every column (`null_best = 164.1 c`).
  - **Negative 6-pair cross-song control (all pairs across Verse / Chorus / Kite / Fridge targets under identical `best_tr_st` search):** median raw `288.8 c` (`0.129` `in50c`), median best-transposed `167.4 c` (`0.087` `in50c`), none passing `lag_null`.
  - **Withdrawal of `ltas_cos_kurt_unplugged` as a voice-conversion metric:** Independent controls in [`2026-11-27_music_cover_pitch_coverage.md`](2026-11-27_music_cover_pitch_coverage.md) §9 (`440b4ff5`) showed raw `ltas_cos` moves sharply with `< 150 Hz` bass level (`-0.023` verse scaffold vs `+0.636` unconverted verse vocal target), and unconverted chorus/bridge targets (`+0.841` / `+0.890`) score higher against `kurt_vocal_clean_192f.wav` than Kurt's own self-split (`+0.804`). High-passed `150 Hz` comparisons (`4e27b500`) show H3 genuinely adds `1–4 kHz` vocal presence/grit with higher `denoise`, which correlates with fewer strict-YIN voiced frames rather than proving identity alone. Timbre conversion (`c1` smoother baritone vs `c5`/`c6`/`w1`–`w4` gravelly Kurt fry) rests on direct ear audit.

---

## 3. 9-Arm `243f` (`10.125 s`) Single-Window Gate (`c1`–`c9`)

All nine arms use `kurt_closeup.png`, `kurt_solo_medium.png` (except `c4` single-pic rain diner), and `kurt_vocal_clean_192f.wav`. Each `lag_null` cell prints `aligned / null_med / null_best / beats_frac` from [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py).

| Arm | Target Med (`Hz`) / `best_tr_st` | `voiced_overlap_frac` (`mut / tgt`) | `med_abs_cent_err` (`lag_null`) | `in_tune_frac_50c` (`lag_null`) | `rel_med_abs_cent_err` (`lag_null`) | `rel_in_tune_frac_50c` (`lag_null`) | `r_one_shift` | Relaxed `thr=0.28`: `ov` / `err` / `beats` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `175.3` / `+0` | `0.308` (`193/627`) | `8.7` / `256.9` / `108.4` / **`1.00`** | `0.917` / `0.154` / `0.423` / **`1.00`** | `8.7` / `141.7` / `85.5` / **`1.00`** | `0.917` / `0.143` / `0.294` / **`1.00`** | `+0.348` | `0.628` / `20.3 c` / **`1.00`** |
| `c2_verse_m5_d32` | `175.3` / `+0` | `0.089` (`56/627`) | `39.9` / `342.6` / `86.9` / `1.00` | `0.536` / `0.000` / `0.256` / `1.00` | `39.9` / `34.1` / `10.1` / **`0.42`** | `0.536` / `0.590` / `0.889` / **`0.39`** | `+0.033` | `0.348` / `193.5 c` / **`0.909`** |
| `c3_chorus_m5_d30` | `232.3` / `+0` | `0.292` (`192/657`) | `40.3` / `335.8` / `191.2` / **`1.00`** | `0.542` / `0.026` / `0.190` / **`1.00`** | `40.3` / `139.8` / `78.2` / **`1.00`** | `0.542` / `0.141` / `0.463` / **`1.00`** | `+0.688` | `0.598` / `58.9 c` / **`1.00`** |
| `c4_verse_m5_rain_a2v0_d30` | `175.3` / `+0` | `0.367` (`230/627`) | `48.9` / `217.6` / `98.7` / **`1.00`** | `0.513` / `0.096` / `0.336` / **`1.00`** | `48.9` / `143.6` / `93.9` / **`1.00`** | `0.513` / `0.133` / `0.348` / **`1.00`** | `+0.260` | `0.681` / `139.2 c` / `0.879` |
| `c5_verse_m5_d30_s1` | `175.3` / `+0` | `0.214` (`134/627`) | `60.6` / `277.7` / `52.3` / `0.97` | `0.463` / `0.073` / `0.494` / `0.97` | `60.6` / `132.0` / `35.6` / `0.88` | `0.463` / `0.100` / `0.591` / `0.88` | `+0.560` | `0.579` / `237.9 c` / `0.606` |
| `c6_verse_m5_d30_s2` | `175.3` / `+0` | `0.169` (`106/627`) | `48.2` / `227.9` / `80.1` / **`1.00`** | `0.566` / `0.083` / `0.394` / **`1.00`** | `48.2` / `101.1` / `21.6` / `0.91` | `0.566` / `0.261` / `0.673` / `0.94` | `+0.548` | `0.423` / `55.9 c` / **`1.00`** |
| `c7_bridge_peak_m5_d30` | `261.9` / `+0` | `0.377` (`239/634`) | `20.3` / `271.0` / `129.5` / **`1.00`** | `0.644` / `0.058` / `0.349` / **`1.00`** | `20.3` / `131.0` / `84.9` / **`1.00`** | `0.644` / `0.124` / `0.417` / **`1.00`** | **`+0.758`** | `0.692` / `30.8 c` / **`1.00`** |
| `c8_kite_chorus_d30` | `230.7` / `+1` | `0.074` (`21/283`) | `109.5` / `183.5` / `16.3` / `0.80` | `0.381` / `0.113` / `0.833` / `0.80` | `96.9` / `104.9` / `16.3` / `0.60` | `0.190` / `0.225` / `0.917` / `0.40` | `-0.190` | `0.343` / `132.7 c` / `0.939` |
| `c9_fridge_chorus_d30` | `208.2` / `+0` | `0.098` (`22/224`) | `16.0` (no null) | `0.591` (no null) | `16.0` (no null) | `0.591` (no null) | `-0.227` | `0.415` / `157.3 c` / `0.871` |

- **Fractional-semitone `in50` sweep (`next_steps.md` item `000`, `0.25 st` steps in `[-6, +6]`):** Integer `in_tune_frac_50c` maximization picks `+0 st` on all `justaflicker` arms (`0.917`, `0.536`, `0.542`, `0.513`, `0.463`, `0.566`, `0.644`), matching `best_tr_st = 0`. Allowing `+0.25 st` fractional tuning lifts `in50` slightly on sharp-biased arms (`c2`: `0.536 -> 0.571`; `c3`: `0.542 -> 0.630`; `c5`: `0.463 -> 0.530`; `c6`: `0.566 -> 0.585`).
- **Why low `voiced_overlap_frac` occurs (observable YIN detector bins + §11 controls):** Across `c1`–`c7`, `below_min_dbfs_frac_of_tgt` (`rms < -45 dBFS`) is `0.000–0.008`, `cmndf_015_to_028_frac_of_tgt` holds `0.254–0.365` (`+159..+229` frames), and `cmndf_ge_028_frac_of_tgt` holds `0.300–0.652`. As shown independently in [`2026-11-27_music_cover_pitch_coverage.md`](2026-11-27_music_cover_pitch_coverage.md) §11, stripping `< 150 Hz` bass rescues raw scaffolds (`414 -> 535`) while leaving render stems unchanged (`193 -> 164` on `c1`), and mixing a `-10 dB` bed leaves `85.0%` pitched — proving the periodicity drop lives inside H3's `> 150 Hz` synthesized vocal waveform itself.

### Honest Ear & Visual Audit (`c1`–`c9`)
- **`c1` vs `c5`/`c6` (`d=0.26` vs `0.30` seed pair):** `c1` (`8.7 c`) sounds closer to the smoother source baritone on *"Grey light"*; `c5` and `c6` both add audible husky Kurt fry (`c6` holds `55.9 c` and `1.00` null beat at `thr=0.28`, while `c5` sharpens toward `203.9 Hz` [`+52.6 c` bias] and misses `thr=0.28` null at `0.606`).
- **`c4` (`Rain-Diner` + `a2v=0.0`):** `[Shot 1]` (`00:00–00:04`) renders the rainy midnight neon diner sidewalk with bone-dry studio vocals; **`[Shot 2]` (`00:05–00:09` tight close-up) snaps back to `<Picture 1>`'s violet Unplugged stage backdrop**. Fix for custom-location `[Shot 2]`: neutral/cutout reference portraits or `MiniMaxH3ReleaseReferences(end_percent=0.50)` (note `video_end_percent` only releases `<Video N>` refs and is a no-op on `<Picture N>`).
- **`c7` (`Bridge Peak`, `261.9 Hz` med / `301.8 Hz` `p90`):** Holds `20.3 c` / `1.00` null on `0.15` and `30.8 c` / `1.00` on `0.28`, thinning slightly above `300 Hz` and slurring `[Shot 2]` (*"just a bit"* -> *"just got a bit"*) because `103.0..113.125 s` sings *"just a flicker, just a bit"* across `00:04–00:08` before *"Every—"* enters at `00:09`, colliding with `ref2va_cal_unplugged_bridge.txt`'s `[Shot 2]` `"Everything we try to quit"` line.
- **`c8` (`thatsnotakite`, `harmonic_db=-120.0`) & `c9` (`inthefridge`, `harmonic_db=-120.0`) — Correction:** Inspecting isolated `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/vocal_stems/c8_kite_chorus_d30_voc.wav` and `c8`'s last `2.5 s` without visuals **retracts** my initial impression of synthesized steel-string guitar: **`c8` has NO audible guitar strumming** — it is a near-a-cappella belted grunge vocal with reduced `< 150 Hz` rest sub-bass (`-0.7 dB` of rest energy, matching §11 where `sepaudio six` misfiled `+0.0 dB` of that sub-bass under `vocals`). By contrast, `c9` (`+2 st`, zero chipmunk artifact) has an audible plucked bass/low-guitar groove during its `00:03.2–00:05.0` vocal rest (`-14.9 dB` bass / `-12.15 dB` other). By ear, H3 sings both clips as a single lead voice rather than gang vocals (`0.034` / `0.058` active periodicity leaves single-voice YIN unscoreable).

---

## 4. 4-Window `44.125 s` (`1059f`) Chained Pure-H3 Cover (`--song-denoise 0.29`)

### Graph Architecture Fixes (`bench/pure_h3_cover_song.py`)
1. **Removed `MiniMaxH3GuideLatent(frame_idx=0)` duplicate-RoPE collision:** Window `k > 1` chains purely through `TrimAudioDuration(duration=1.625)` + `AudioConcat -> MiniMaxH3FreezeAudioWindow(context_frames=39, context_noise=0.35, context_profile="seam_taper")` (`start_seconds=0.0`, `previous=["110_w{k-1}", 0]`) on 100% stock RoPE.
2. **Boundary alignment (`SONG_START_S = 8.875` vs `9.500`):** At `9.500`, every `39f` overlap ran `0.38–0.50 s` into the next sung word (`-13.7..-15.9 dBFS` across the last 5 `seam_taper` lock frames `f34..38` and mid-word cutoff at `w4`'s end). Shifting `SONG_START_S = 8.875` (`8.875..53.000 s`) places all three seams (`21.125 s`, `31.750 s`, `42.375 s`) and `w4`'s ending inside `-34.0..-59.3 dBFS` vocal pauses (`f34..38`).

### Per-Window (`w1`–`w4`) & Full `44.125 s` Pitch-Lock + Coverage Results

Rendered in `1911.2 s` (`~31.9 min`) to `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/kurt_unplugged_justaflicker_verse_chorus_44s-audio.mp4` (`1059f`, `44.125 s`, zero external audio mux; every clip scored from cached `MelBand-RoFormer` stems via [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py) and [`bench/pitch_lock_coverage.py`](../pitch_lock_coverage.py)):

| Segment | `voiced_overlap_frac` (`mut / tgt`) | `best_tr_st` | `med_abs_cent_err` (`lag_null`) | `in_tune_frac_50c` (`lag_null`) | `rel_med_abs_cent_err` (`lag_null`) | `rel_in_tune_frac_50c` (`lag_null`) | `r_one_shift` | Relaxed `thr=0.28`: `ov` / `err` / `beats` | `coverage` sweep `0.25 / 0.35 / 0.50` (`beats`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w1` (`8.875..21.125 s`, Verse 1) | `0.114` (`71/623`) | `+0` | `28.0` / `311.9` / `52.3` / **`1.00`** | `0.676` / `0.085` / `0.406` / **`1.00`** | `28.0` / `76.3` / `5.8` / `0.83` | `0.676` / `0.364` / `0.917` / `0.90` | `+0.119` | `0.385` / `49.3 c` / **`1.00`** | `0.321 / 0.543 / 0.852` (**`1.00`**) |
| `w2` (`19.500..31.750 s`, Verse 2) | `0.086` (`58/678`) | `+0` | `27.3` / `224.7` / `50.8` / **`1.00`** | `0.586` / `0.077` / `0.500` / **`1.00`** | `27.3` / `95.2` / `22.0` / **`0.98`** | `0.586` / `0.250` / `0.882` / **`0.98`** | `-0.155` | `0.378` / `33.7 c` / **`1.00`** | `0.305 / 0.546 / 0.801` (**`1.00`**) |
| `w3` (`30.125..42.375 s`, Chorus+Break) | `0.059` (`32/543`) | `+0` | `29.2` / `311.1` / `18.6` / `0.90` | `0.594` / `0.000` / `0.842` / `0.90` | `29.2` / `68.7` / `14.7` / `0.79` | `0.594` / `0.500` / `0.842` / `0.76` | `+0.700` | `0.239` / `67.1 c` / `0.976` | `0.186 / 0.389 / 0.656` (**`1.00`**) |
| `w4` (`40.750..53.000 s`, Chorus Tag) | `0.384` (`273/711`) | `+0` | `15.2` / `307.2` / `170.7` / **`1.00`** | `0.722` / `0.056` / `0.207` / **`1.00`** | `15.2` / `133.3` / `59.4` / **`1.00`** | `0.722` / `0.144` / `0.478` / **`1.00`** | `+0.460` | `0.633` / `30.3 c` / **`1.00`** | `0.581 / 0.744 / 0.858` (**`1.00`**) |
| **Stitched `44.125 s` (`1059f`)** | **`0.172`** (`419/2439`) | **`+0`** | **`17.9`** / `271.3` / `111.3` / **`1.00`** | **`0.704`** / `0.086` / `0.332` / **`1.00`** | **`17.9`** / `175.1` / `63.0` / **`1.00`** | **`0.704`** / `0.100` / `0.476` / **`1.00`** | **`+0.510`** | **`0.422` / `37.7 c` / `1.00`** | **`0.361 / 0.572 / 0.823`** (**`1.00`**) |

- **Window Seam & AV Continuity Metrics (`song_stitch_meta.json`, mean adjacent L1 across 1059f = `3.543`):**
  - `w1 -> w2` (`f293 -> f294`, `12.25 s`): `seam_l1 = 7.145`, `post_10f_adj_mean = 4.132`, `seam_ratio = 1.729`.
  - `w2 -> w3` (`f548 -> f549`, `22.875 s`): `seam_l1 = 5.863`, `post_10f_adj_mean = 5.547`, `seam_ratio = 1.057`.
  - `w3 -> w4` (`f803 -> f804`, `33.50 s`): `seam_l1 = 6.550`, `post_10f_adj_mean = 3.802`, `seam_ratio = 1.723`.
- **Direct AV Playback (`read_file` on `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/kurt_unplugged_justaflicker_verse_chorus_44s-audio.mp4` and `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/vocal_stems/song_44s_voc.wav`):**
  - `00:00–00:02` opens with warm unplugged chords and calm solo stage guitar posture before gravelly Kurt vocals enter cleanly on *"Grey light hits the concrete block / Watching how the shadows walk"*. All three `300 ms` equal-power audio joins (`11.3 s`, `21.9 s`, `32.5 s`) sound completely seamless without clicks, echoes, or key drift (`chroma_drift_st = 0`), transitioning smoothly out of `w1`/`w2` verse (`175 Hz`) into the belted Chorus *"It's just a flicker, an echo in the lane"* (`w3`, `232 Hz` target) and closing with `w4`'s clean outro sustain. In `w3` `[Shot 2]` (`00:27.5–00:32.0`), Kurt's lips stay closed over clean instrumental chords without stray vocal bleed (visual framing stayed at the medium stage shot rather than cutting strictly down to hands-only), while `w4` `[Shot 2]` cleanly preserves the source's echo/backing whisper tail on *"like the static on the pane"*. Visually, because `seam_taper` pins 5 boundary frames (`seam_ratio = 1.057–1.729`), lighting and wardrobe stay completely continuous with only a subtle 1-beat motion inflection at `f294` and `f804`.
- **Next optional follow-up:** Using the same harness (`--song`), a multi-window cover of `thatsnotakite` (or testing `justaflicker` Verse `d=0.26` -> Chorus `d=0.29` per-window denoise to raise strict `voiced_overlap_frac` in `w1`/`w2` from `0.114`/`0.086` toward `0.30+`) is ready when needed.
