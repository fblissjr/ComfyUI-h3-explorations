# Phase 1 Zero-Code Silicon Gate: H3 Music-Video Voice & Style Adaptation (`2026-11-27`)

> **Harness:** [`archive/bench/music_cover_gate.py`](../../archive/bench/music_cover_gate.py) · **Prompts:** [`data/music_cover_gate/`](../../data/music_cover_gate/) (`0 FAIL, 0 WARN`) · **Triage JSON:** [`data/music_cover_gate/phase1_triage_report.json`](../../data/music_cover_gate/phase1_triage_report.json)
> **Observable media location:** `$H3_OUTPUT_DIR/Video/music_cover_gate_2026-11-27/` (all 9 rendered `-audio.mp4` files inspected natively via `read_file`).

---

## 1. Controlled Extreme-Contrast Test Pair (Zero Memorization Shortcut)

All 8 arms ran on `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (`1152x768`, `141f = 5.875 s`, `seed = 730451892`, `MiniMaxH3Sol`, `PIKA_REF2VA_SVD995_LORA` 6-step `lcm` except `b4` on undistilled Base 16-step `euler/simple`):

- **Target B (Performer & Acoustic Style):** Weathered 50s male delta-blues singer-guitarist (`target_b_solo.png` + `target_b_audio.wav` extracted from `p2_blues_base_en_00001-audio.mp4`, exact `188,000` samples = `235.0` 40 Hz steps; `median_f0 = 194.63 Hz`, `hf4k_share = 0.0020`, solo steel resonator guitar, zero drums).
- **Song X (Source Song to Adapt):** Soaring female mezzo-soprano 1970s disco performance (`song_x_raw.wav` extracted from `s2_disco_erasure_clean_00001-audio.mp4`, exact `235.0` steps; `median_f0 = 456.81 Hz` — **`+14.77 st` above Target B**, `hf4k_share = 0.0236`, full four-on-the-floor disco drums/bass/strings), singing verbatim lyrics:
  - `[Shot 1]` `<d>[English] Spin me round the floor until the morning light!</d>`
  - `[Shot 2]` `<d>[English] Feel the groove take flight!</d>`
- **Preprocessed Scaffold Variant (`song_x_baritone_scaffold.wav`):** Center-channel STFT-HPSS (`120..4000 Hz`, `hf4k_share = 0.0`) + `-12 st` `torchaudio.functional.pitch_shift` (`median_f0 = 258.24 Hz`).

---

## 2. Quantitative Triage + Honest Native Audio-Visual Verdicts

| Arm | Time | `rms_dbfs` | `median_f0` (`st` vs B) | `env_corr` vs Song X | `f0_corr` vs Song X | `f0_corr` vs Target B | `ltas_cos` vs B | `frame_mel` vs Song X | Honest Native AV Verdict (`read_file`) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **`c0_voice_ceiling`** (`<Audio 1>` only) | `90.2 s` | `-27.18` | `194.50 Hz` (`-0.01`) | `0.220` | `-0.009` | `0.024` | `0.972` | `0.162` | **Voice-clone ceiling = 10/10.** Unmistakable Target-B gravelly baritone + metallic steel resonator guitar + clean new lyrics & lip-sync; sings a slow blues tune (no Song X audio). |
| **`a1_native_dual_raw`** (dual `<Audio>` raw) | `95.4 s` | `-27.27` | `189.66 Hz` (`-0.45`) | `0.038` | `0.354` | **`0.795`** | `0.972` | `0.188` | Clean Target-B baritone + steel guitar (zero disco bleed), **but `<Audio 1> + <Picture 1>` drowned out `<Audio 2>`** — still sang Target B's slow blues tune. |
| **`a2_native_dual_scaffold`** (dual `<Audio>` scaf) | `81.0 s` | `-26.84` | `192.95 Hz` (`-0.15`) | `0.226` | `0.431` | `0.685` | `0.981` | `0.195` | Clean Target-B baritone + steel guitar, slight rhythmic pickup, but still dominated by Target B's blues melody. |
| **`b1_cond_release_33_raw`** (`cond_audio` raw @ `0.33`, 2 steps) | `80.1 s` | `-26.17` | `216.49 Hz` (`+1.84`) | `0.290` | `0.069` | `0.268` | `0.962` | `0.210` | Clean Target-B baritone (slightly brighter/lifted) + pure steel guitar, zero disco drums; **released too early (`2/6` steps, `σ_a >= 0.90`) to lock Song X's melody.** |
| **`b2_cond_release_50_raw`** (`cond_audio` raw @ `0.50`, 3 steps) | `69.1 s` | `-17.16` | **`406.47 Hz` (`+12.75`)** | **`0.814`** | `0.476` | `-0.209` | `0.921` | **`0.512`** | **SMOKING GUN:** Locked Song X's **exact recognizable disco melody & timing**, *but* leaked Person A's high female soprano voice + disco backing through `blocks.20..49` because `cond_audio` stayed clean (`1.0`) across all 50 blocks. |
| **`b3_cond_release_50_scaffold`** (`cond_audio` scaf @ `0.50`) | `69.4 s` | `-27.18` | `196.07 Hz` (`+0.13`) | `0.296` | `0.117` | `0.250` | `0.976` | `0.194` | Clean Target-B baritone + steel guitar, but H3 rejected the muffled/phase-vocoded `-12 st` scaffold after release and snapped back to blues. |
| **`c1_twopass_rescore`** (`FreezeAudio` → `AudioRefineMask`) | `137.8 s` | `-27.39` | `196.54 Hz` (`+0.17`) | `0.362` | **`0.676`** | `0.640` | `0.976` | `0.187` | **Pixel-exact video lock (`video_mae = 0.0`)** between Pass 1 & Pass 2; Pass 2 completely erased the Pass-1 scaffold into clean Target-B baritone + steel guitar with tight lip-sync & strumming, but still leaned bluesy due to the muffled scaffold. |
| **`b4_base16_scaffold`** (Base 16-step `euler/simple`) | `134.3 s` | `-25.10` | `199.89 Hz` (`+0.46`) | `0.178` | `0.060` | `0.037` | `0.971` | `0.163` | Clean Target-B baritone + steel guitar; confirms the muffled `-12 st` phase-vocoded scaffold itself (not just Pika's 6-step schedule) failed to anchor melody. |

---

## 3. Root-Cause Diagnosis & Phase 2 Decision Gate

Phase 1 closed every empirical blind spot cleanly:

1. **Voice & acoustic instrument cloning (`<Audio 1>: reference`) works natively at high fidelity (`c0`, `-0.01 st`, `ltas_cos = 0.972`).**
2. **Putting Song X at `t < 0` (`<Audio 2>`) loses to `<Audio 1> + <Picture 1>` (`a1`, `a2`).** Frame-aligned `t >= 0` `cond_audio` is required to force Song X's exact melody.
3. **`b1` vs. `b2` exposes a sharp 1-step phase transition on 6-step Pika (`0.33` = Steps `0,1` vs. `0.50` = Steps `0,1,2`):**
   - At 2 steps (`b1`), Target B's baritone holds cleanly, but Song X's melody hasn't locked yet.
   - At 3 steps (`b2`), **Song X's exact melody locks (`env_corr = 0.814`)**, **but Person A's female soprano timbre (`406.5 Hz`) leaks in** because:
     1. `Q_aud` attends to `K_cond_audio` across **all 50 DiT blocks (`0..49`)**, including the vocal-pitch & acoustic-timbre blocks (`20..49`), and
     2. `cond_audio` is injected at full `audio_clean_level = 1.0` (preserving crisp female formants & disco hi-hats).
4. **Why `b3`/`b4` lost melody:** Heavy `-12 st` phase-vocoder + `4 kHz` low-pass (`hf4k_share = 0.0`) created an out-of-distribution warble that H3 discarded as soon as `cond_audio` dropped.

### Phase 2 Target (Triggered Directly by `b2`)
Build **Phase 2A + 2B** cleanly (zero peer-file collisions) and re-run matched-seed A/B directly against **`b2_cond_release_50_raw`**:
1. **`MiniMaxH3AudioRefRouter` (`audio_ref_router.py` — `1x` per-ref `Q_aud` block-band attention router + per-block `cond_audio_clean_level`):**
   - Keeps `Q_aud -> K_cond_audio` (Song X) strong in **early structure/melody/timing blocks (`0 .. split_block-1`)** while **suppressing `Q_aud -> K_cond_audio` and boosting `Q_aud -> K_ref_audio` (Target B) in timbre/acoustic blocks (`split_block .. 49`)**, strictly via additive `Q_aud` side-SDPA logit bias (never mutating `V[ref_audio]` in-place).
   - Independently adds deterministic Gaussian noise ONLY to `cond_audio` (`cond_audio_clean_level < 1.0`, blurring Person A's sharp formants/hi-hats) while leaving Target B's `<Audio 1>` `ref_audio` clean (`1.0`).
   - Optionally extends `attention.py::_resolve_span_spec` for `ref_audio`/`cond_audio` LSE balancing.
2. **Base 16-step Raw Song X control (`audio_end_percent = 0.38` & `0.44`):** Maps the fine-grained `σ_a` release window between Pika's 2-step (`0.33`) and 3-step (`0.50`) cliff on crisp, in-distribution `song_x_raw.wav` (plus raw `song_x_raw.wav` in Pass-1 of `c1_twopass_rescore`).
