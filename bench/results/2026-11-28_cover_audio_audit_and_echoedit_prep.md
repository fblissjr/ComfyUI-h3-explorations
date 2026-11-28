# Cover Audio End-to-End Audit, Independent Metrics, Gate A Shootout, & `MiniMaxH3EchoEdit` (`2026-11-28`)

- **Verdict:** Audited the entire cover-song audio path from `sglang` reference encoding through stem separation, pitch shifting, Pass-2 sampling, and evaluation metrics. Fixing `librubberband` R2's stereo `channels=apart` phase decorrelation with mono + `ffmpeg` `atempo` WSOLA jumped full-band `L/R_r` from `0.4450` to `0.7502` (`arm1a`, `med_abs_cent_err = 16.9 c`, `chroma_beats = 1.000`) and decoupling the H3 backing pass (`arm3_decoupled`, `backing_denoise = 0.48`) replaced the pop synth bed with pure live acoustic guitar (`timbre_sim_b` `0.885 -> 0.922`, `med_abs_cent_err = 16.1 c`). Across 6 matched-seed Gate A arms evaluated with our new RoFormer-free / YIN-free [`bench/cover_audio_audit.py`](../cover_audio_audit.py) and native listening, **`denoise = 1.0` (`arm2`, `arm2b`, `arm2c`) removes all vocal compression sheen and flips `timbre_margin` positive (`+0.043..+0.047`) even with zero stem separation, but loses unseen melodies (`chroma_beats = 0.765..0.873 < 0.90`), whereas `SDEdit d = 0.29` (`arm1a`, `arm3`) locks melody (`1.000`) while retaining mild vocal sheen and negative `timbre_margin` (`-0.026..-0.072`).** Implemented and unit-tested [`echo_edit.py`](../../echo_edit.py) (`MiniMaxH3EchoEdit`, `72`th node) implementing inversion-free `EchoEdit+` (`arXiv:2606.15149`) + `AV-FlowEdit` correlated noise, wired `--render-mode echoedit` in [`bench/h3_cover_workflow.py`](../h3_cover_workflow.py) (`--dry-run` preflight PASS), and paused before live `arm4_echoedit_raw_m5` / Gate B for a ComfyUI `:8188` reload of the fixed `echo_edit.py` schedule + `CFGGuider.sample` lifecycle.
- **Date:** `2026-11-28`
- **Scripts & Nodes:** [`guide_latent.py`](../../guide_latent.py), [`echo_edit.py`](../../echo_edit.py), [`nodes.py`](../../nodes.py), [`bench/h3_stem_sep.py`](../h3_stem_sep.py), [`bench/h3_cover_workflow.py`](../h3_cover_workflow.py), [`bench/cover_audio_audit.py`](../cover_audio_audit.py)
- **Checks:** [`bench/check_guide_latent.py`](../check_guide_latent.py), [`bench/check_h3_stem_sep.py`](../check_h3_stem_sep.py), [`bench/check_h3_cover_workflow.py`](../check_h3_cover_workflow.py), [`bench/check_cover_audio_audit.py`](../check_cover_audio_audit.py), [`bench/check_echo_edit.py`](../check_echo_edit.py), [`bench/check_node_ids.py`](../check_node_ids.py), [`bench/check_doc_links.py`](../check_doc_links.py)
- **Archived Narrow Scripts:** Moved 7 single-pair cover harnesses (`unplugged_fun_cover.py`, `pure_h3_cover_song.py`, `ai_song_sdedit_covers.py`, `music_cover_gate.py`, `music_cover_shootout.py`, `music_cover_pitch_lock_gate.py`, `prepare_pitch_lock_assets.py`) into `archive/bench/` via `git mv`.
- **Observable Media:** `$H3_OUTPUT_DIR/Video/cover_jobs/ctrl_kurt_chorus/` (`arm1a_sdedit_mono_vocref`, `arm1b_sdedit_mono_rawref`, `arm2_resynth_d100`, `arm2b_router_raw_m5_r56`, `arm2c_router_raw_m5_r70`, `arm3_decoupled`, and `_shifter_audit/`).

---

## 1. Upstream Parity Audit (`sglang`, `guide_latent.py`, RoFormer, & `ffmpeg` Shifters)

1. **`sglang` vs. pack reference & guide audio encoding:**
   - Verified `reference_conditioning.py` matches `sglang`'s `minimax_h3` reference path (`stereo float32 [-1, 1]`, mono duplicated to 2 ch, bandlimited sinc resample to `32 kHz`, right-pad to `hop = 800`, `fp32 AudioVAE` posterior mean, `(z - mean) / std`, `AUDIO_COND_TIMESTEP = 1.0`).
   - Found and fixed an alignment gap in [`guide_latent.py`](../../guide_latent.py): added `_encode_guide_audio_aligned` (mono→stereo duplication, `audio_resample.resample`, right-pad to `hop = 800` before `_encode_ref_audio`). Verified in [`bench/check_guide_latent.py`](../check_guide_latent.py) (`5/5` groups).
2. **MelBand-RoFormer windowing & opt-in dereverb ([`bench/h3_stem_sep.py`](../h3_stem_sep.py)):**
   - Aligned `_run_roformer_on_device` with `ComfyUI-MelBandRoFormer` (`fp32`, flat-top `_trapezoidal_window` with `fade_size = chunk // 10`, `1.0` at chunk `0` head and final chunk tail).
   - Added `dereverb_vocals_roformer` loading `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` (`weights_only=True`, `684` keys), exposed opt-in via `--dereverb-vocal`.
3. **CPU `ffmpeg` pitch-shifter shootout (`$H3_OUTPUT_DIR/Video/cover_jobs/_shifter_audit/`):**
   - Yesterday's `librubberband` R2 defaulted to `channels=apart`, decorrelating stereo channels on `-5 st` (`L/R_r = 0.3940`, `Side/Mid = -3.62 dB`, `periodic = 0.735`, `med_cmndf = 0.060`) and `-12 st` (`L/R_r = 0.7376`, `Side/Mid = -8.13 dB`, `periodic = 0.786`).
   - Collapsing lead vocals to mono (`mono_vocal=True`) and switching to `ffmpeg` `asetrate + aresample=soxr + atempo` WSOLA (`--shifter atempo`, default) achieved `L/R_r = 1.0000`, `Side/Mid = -98.96..-101.23 dB`, and higher periodicity (`0.771` on `-5 st`, `0.787` on `-12 st`, `med_cmndf = 0.040..0.050`), audibly cleaner than both `rubberband` `apart` and `rubberband` `together` (`mono`).

---

## 2. Independent Audio Audit (`bench/cover_audio_audit.py`) & Gate A Results

To avoid overfitting or RoFormer-on-output / YIN drop-out artifacts, [`bench/cover_audio_audit.py`](../cover_audio_audit.py) (`4/4` synthetic self-tests in [`bench/check_cover_audio_audit.py`](../check_cover_audio_audit.py)) decodes any `.mp4`/`.flac`/`.wav` at `32 kHz` stereo via `ffmpeg` and scores 4 independent axes directly on the full mix:
- **Naturalness / Anti-Sheen:** multi-band `L/R_r` & `Side/Mid dB`, `0..4500 Hz` Hillenbrand Cepstral Peak Prominence (`cpp_db_med`), `1–4 kHz` spectral flatness, mid-presence / HF air ratios, and crest factor.
- **Melody / Harmony Lock:** `40 Hz` 12-bin harmonic-salience chromagram cosine similarity vs. key-shifted raw target (`raw_m5_32k.wav`) against `102` circular time-lag nulls (`chroma_lift`, `chroma_beats_frac`).
- **Rhythm / Onset Lock:** positive log-spectral flux onset envelope Pearson `r` (`±50 ms` tolerance) vs. circular lag null.
- **Timbre / Style Transfer:** active-frame 12-dim MFCC (`c1..c12`) centroid cosine similarity to Target B raw audio (`ref_raw_192f.wav`) minus Source A shifted mix (`timbre_margin = sim_b - sim_a`).

All Gate A arms ran on `ctrl_kurt_chorus` (`1024x768`, `294f = 12.25 s`, `seed = 910427311`, `justaflicker.mp3` chorus `41.6667..53.9167 s` shifted `-5.0 st`):

| Arm | Pipeline / Guide / `<Audio 1>` | `L/R_r` (full / voc) | `cpp_db` | `flat_1_4k` | `chroma_lift` (`beats`) | `onset_r0` (`beats`) | `timbre_margin` (`sim_b` / `sim_a`) | `YIN med_err` (`lag_null`) | Native AV Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `u0_ceiling` (11-27) | Pika 6-step, raw `<Audio 1>` only | `0.735 / 0.755` | `1.312` | `0.1924` | `+0.004 (0.542)` | `0.045 (0.169)` | **`+0.1159`** (`0.987 / 0.871`) | N/A | Clean unplugged Kurt + guitar, zero sheen (no melody target). |
| `u8_router` (11-27) | Base 16-step `AudioRefRouter`, HPSS `fun` | `0.524 / 0.471` | `1.116` | `0.1917` | `+0.007 (0.678)` | `0.026 (0.203)` | **`+0.1175`** (`0.978 / 0.860`) | N/A | Authentic Kurt voice & guitar covering `fun` cleanly. |
| `yesterday_ctrl` (11-27) | `sdedit d=0.29`, `rubberband apart`, voc ref | `0.445 / 0.422` | `1.227` | `0.1533` | `+0.245 (1.000)` | `0.479 (1.000)` | `-0.0832` (`0.883 / 0.966`) | `39.8c (1.000)` | Widened phasey stereo swirl (`L/R_r = 0.445`). |
| `arm1a_sdedit_mono_vocref` | `sdedit d=0.29`, `atempo mono`, voc ref | `0.750 / 0.737` | `1.343` | `0.1354` | **`+0.265 (1.000)`** | **`0.507 (1.000)`** | `-0.0722` (`0.885 / 0.957`) | `16.9c (1.000)` | Stereo phase swirl gone, accurate melody, mild `d=0.29` sheen remains. |
| `arm1b_sdedit_mono_rawref` | `sdedit d=0.29`, `atempo mono`, raw ref | `0.597 / 0.567` | `1.144` | `0.1835` | `+0.167 (1.000)` | `0.534 (1.000)` | `-0.0132` (`0.913 / 0.926`) | `48.2c (1.000)` | Shimmer-doubled harmony & weaker melody lock than `1a`. |
| `arm2_resynth_d100` | `resynth d=1.0`, Stage-A guide, `ReleaseRefs(0.42)` | **`0.818 / 0.833`** | **`1.482`** | **`0.1016`** | `+0.024 (0.843)` | `0.110 (0.961)` | `-0.0644` (`0.862 / 0.927`) | `140.5c (0.780)` | Cleanest waveform (`0` sheen), wandered off-melody. |
| `arm2b_router_raw_m5_r56` | `resynth d=1.0`, **0-stem-sep** `raw_m5`, raw ref, `rel=0.56` | `0.559 / 0.495` | `1.076` | `0.2261` | `+0.043 (0.873)` | `0.063 (0.627)` | **`+0.0428`** (`0.961 / 0.918`) | `312.5c (0.317)` | Zero compression sheen, unplugged Kurt timbre, drifted to folk chord. |
| `arm2c_router_raw_m5_r70` | `resynth d=1.0`, **0-stem-sep** `raw_m5`, raw ref, `rel=0.70` | `0.572 / 0.519` | `1.135` | `0.2339` | `+0.034 (0.765)` | `0.088 (0.794)` | **`+0.0466`** (`0.964 / 0.917`) | `220.7c (0.732)` | Same clean unplugged timbre (`+0.047`), still missed `justaflicker` melody. |
| `arm3_decoupled` | `decoupled` (`backing d=0.48` + `arm1a` voc -> `FreezeAudio`) | `0.753 / 0.738` | `1.132` | `0.2062` | `+0.210 (1.000)` | `0.276 (1.000)` | `-0.0256` (`0.922 / 0.947`) | `16.1c (1.000)` | Pure H3 acoustic guitar backing (`d=0.48`), locked melody, lead vocal still inherits `arm1a`'s `d=0.29` sheen. |

---

## 3. `MiniMaxH3EchoEdit` (`echo_edit.py`) & Next SOTA Steps

- **Why `EchoEdit+` (`arXiv:2606.15149`) + `AV-FlowEdit` (`arXiv:2610.13072`):** Gate A isolated the exact Pareto wall: `denoise = 1.0` (`arm2b`/`arm2c`) has zero sheen and positive timbre transfer but drops unseen melodies; `SDEdit d = 0.29` (`arm1a`/`arm3`) locks melody (`1.000`) but retains mild vocal sheen and Source A's envelope. `EchoEdit+` constructs on-manifold OT-coupled queries $z_a^{\text{src}} = (1-\sigma_a)x_a^{\text{src}} + \sigma_a\epsilon$, $z_a^{\text{tar}} = (1-\sigma_a)y_a + \sigma_a\epsilon$ with AR(1) correlated noise (`noise_corr = 0.5`) and integrates $y_a \leftarrow y_a - \Delta\sigma_a \cdot \text{edit\_scale} \cdot (v_a^{\text{tar}} - v_a^{\text{src}})$ (`2×` NFE, no CFG, exact `ModelSamplingAV` `audio_scale = shift_v / shift_a = 4.0` cancellation, frozen video `MAE = 0.0`).
- **Pre-launch audit catches fixed before GPU run:**
  1. Replaced an ascending/truncated `percent_to_sigma` schedule with `build_echo_sigmas(n_steps, t_max, t_min, shift_v, shift_a)` over native audio sigma $\sigma_a$ (strictly descending, all `n_steps` executed, exact `time_shift_sigma(sig_v, 12.0, 3.0) == sig_a`).
  2. Delegated execution through `_EchoEditGuider(CFGGuider)` + `_EchoEditSampler(Sampler)` calling `guider.sample(...)` so `prepare_model_patcher`, `MiniMaxH3Sol`, `ModelAttentionBackend`, `process_latent_in/out`, and hook cleanup match `SamplerCustomAdvanced` without patching ComfyUI core.
- **Immediate GPU queue after `:8188` reload:**
  1. Run `arm4_echoedit_raw_m5` (zero stem separation on `raw_m5_32k.wav` + raw `<Audio 1>`) on `ctrl_kurt_chorus` (`294f`, `seed = 910427311`), audit with `cover_audio_audit.py` + native listening.
  2. Run Gate B diverse 3-pair matrix (`fun_covers_flicker`, `blues_covers_whatever`, `unplugged_covers_violet`, `294f` each) with the winning Gate A recipe.
- **Other Nov-2026 SOTA follow-ups tracked:** `Soundwich` (`arXiv:2610.00691v2`) quiet-reference carrier audio masking & SAM 3.1 bitmask entity routing (`coderef/Soundwich/minimax-h3/soundwich_h3/routing.py`), `SAM-Audio` (`arXiv:2512.18099`) visual-prompt singer isolation & `sam-audio-judge`, `YingMusic-SVC` (`arXiv:2512.04793`), and post-H3 `Apollo` / `FlashSR` `32 -> 48 kHz` restoration.
