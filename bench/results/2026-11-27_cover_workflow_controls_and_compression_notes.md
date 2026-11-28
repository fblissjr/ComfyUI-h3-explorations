# Pure-H3 Cover Workflow Controls, Octave Folding, & Perceptual Compression Notes (`2026-11-27`)

VERDICT: PARTIAL PASS — Generalized workflow controls reproduce `lag_null beats_frac = 1.00` melody lock across a `1`-window (`12.25 s`, `-5.0 st`) chorus control and a `2`-window (`22.875 s`, `-12.0 st` vocal / `0.0 st` accompaniment) octave-folded cross-track run; further multi-song sweeps are paused pending resolution of a persistent Pass-2 vocal compression / phase-vocoder sheen identified by ear on both lossless `.flac` and `.mp4` outputs.

## 1. Workflow, Stem Separation, & Diagnostic Cleanup

- **Archived one-off coverage modes (`bench/pitch_lock_coverage.py`, `bench/check_pitch_lock_coverage.py`):** Slimmed [`bench/pitch_lock_coverage.py`](../pitch_lock_coverage.py) cleanly to the two reusable subcommands (`frames` with `--highpass`, and `periodic`) verified by [`bench/check_pitch_lock_coverage.py`](../check_pitch_lock_coverage.py) (`all 6 case(s) ok`). Archived the five one-off diagnostic modes (`agree`, `levels`, `ltas`, `rests`, `under`) out of `bench/` into `data/pure_h3_cover_song/pitch_lock_coverage_oneoff.py`.
- **Hardened generalized CLI ([`bench/h3_cover_workflow.py`](../h3_cover_workflow.py), [`bench/check_h3_cover_workflow.py`](../check_h3_cover_workflow.py)):**
  - **Cache mtime invalidation (`score_cover_job::_separate_cached`, `cmd_prepare`):** Cached separated stems now re-run whenever the source audio/video `mtime` is newer than the cached `vocals.wav`, preventing stale-stem scoring across repeated runs under one `--job` slug.
  - **Register-ceiling guard (`choose_cover_semitones`):** Tightened upper-quartile/p90 checks against `ref_f0_stats` (`p75` vs `max(ref_p90, ref_p75 * 1.06)` and `p90` vs `ref_p90 * 1.12`) so wide-dynamic-range high-belt targets select `-5.0 st` automatically for a baritone acoustic reference instead of stopping prematurely at `-2.0 st` or `-4.0 st`.
  - **Reference scanning & pinning (`--ref-start-s`, `--ref-scan-s` default `90.0`):** Bounds reference scanning on multi-minute tracks and allows deterministic `8.0 s` (`192f`) vocal slice pinning.
  - **Formant-preservation ear check (`--vocal-formant auto|true|false`):** Direct ear comparison of a `+5 st` upward vocal scaffold with `vocal_formant=True` vs `vocal_formant=False` showed RubberBand formant preservation introduces hollow/nasal phase warping, whereas `vocal_formant=False` sounds natural and bright even on `+5 st` shifts. Kept `False` as the `auto` default.
  - **Pass-2 `MiniMaxH3FrozenVideoCache` deferred to opt-in (`--frozen-video-cache`):** Changed `build_cover_graph` default `use_frozen_video_cache` to `False` (matching the uncached topology) while preserving opt-in `--frozen-video-cache` (`all 4 case(s) ok`).
- **Accompaniment octave folding (`bench/h3_stem_sep.py::_fold_inst_semitones`, [`bench/check_h3_stem_sep.py`](../check_h3_stem_sep.py)):**
  - Added `_fold_inst_semitones(semitones)` folding accompaniment key shifts (`harmonic` + `bass`) into `[-6.0, +6.0]` st (`inst_semitones = 0.0` when vocal shift is `-12.0 st`; `inst_semitones = +4.0` when vocal shift is `-8.0 st`). This preserves exact musical key harmony without turning bass and chord beds into sub-audible mud on large downward vocal transpositions (`all 4 case(s) ok`).

## 2. GPU Workflow Runs (`denoise = 0.29`, `pass2_steps = 12`, `seed = 910427311`, FVC off)

All paths live under `$H3_OUTPUT_DIR/Video/cover_jobs/` and `$H3_INPUT_DIR/cover_jobs/`.

| Run | Span / Windows | Vocal shift / Inst shift | Wall (s) | `mut_voiced` / `tgt_voiced` (`overlap`) | `med_abs_cent_err` (signed) | `in_tune_50c` | `vocal_f0_r` | `lag_null` `beats_frac` (4 core cols) | `0.50` sweep overlap (gained `50c`, `beats_frac`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1`-win chorus control (`baritone acoustic ref -> high-belt target`) | `12.25 s` (`294f`, chorus) | `-5.0 st` / `-5.0 st` (`formant=False`) | `485.44` | `200 / 734` (`0.272`) | `39.8 c` (`+8.7 c`) | `0.525` | `0.961` (`one_shift=0.477`) | `1.00 / 1.00 / 0.927 / 0.927` (`rel_one_shift=0.902`) | `0.846` (`0.277`, `1.00` over `41` lags) |
| `2`-win octave-folded cover (`baritone acoustic ref -> upbeat pop target`) | `22.875 s` (`549f`, verse->hook) | `-12.0 st` / `0.0 st` (`formant=False`) | `956.59` | `374 / 1387` (`0.270`) | `26.8 c` (`+0.3 c`) | `0.671` | `0.981` (`one_shift=0.226`) | `1.00 / 1.00 / 1.00 / 1.00` (`rel_one_shift=0.929`) | `0.828` (`0.439`, `1.00` over `84` lags) |

Window breakdown on the `2`-window run (`w1->w2` visual seam `L1 = 9.101`, `post_10f_adj_mean = 3.613`, `seam_ratio = 2.519`, `mean_adj_l1 = 4.037`):
- `w1`: `mut_voiced = 247 / 803` (`0.308`), `med_abs_cent_err = 24.8 c` (`-3.0 c`), `in_tune_50c = 0.680`, `vocal_f0_r_one_shift = 0.200` (`beats_frac = 0.878`), core 4 `lag_null beats_frac = 1.00`.
- `w2`: `mut_voiced = 210 / 690` (`0.304`), `med_abs_cent_err = 33.7 c` (`+11.6 c`), `in_tune_50c = 0.610`, `vocal_f0_r_one_shift = 0.265` (`beats_frac = 0.780`), core 4 `lag_null beats_frac = 1.00`, `below_min_dbfs = 0.154` (clean silent turnaround tail in `[Shot 2]`).

## 3. Direct Visual & Ear Audit Findings

- **Visual & lip-sync quality:** Both rendered `.mp4` clips show clean acoustic-concert continuity matching `<Picture 1>` and `<Picture 2>`, natural shot cuts (`0:04` in the `1`-window chorus control, `w1->w2` continuity + `[Shot 2]` instrumental mouth-closure turnaround in `0:16..0:22` of the `2`-window run), and crisp acoustic accompaniment where `-12.0 st` folded to `0.0 st` kept synth/guitar harmonics natural rather than muddied.
- **Compressed / phasey / auto-tune-like vocal texture (cross-stage diagnosis):**
  - Listener feedback on the baseline flagged a compressed / auto-tune-like vocal quality.
  - Direct inspection of both raw `VAEDecodeAudio` lossless `32 kHz` `.flac` outputs (`song_aud_w1_00001.flac`) before `192 kbps` AAC muxing **and** the `-12.0 st` cross-track test confirms the sheen is **not** AAC-only and **not** specific to one target track.
  - Likely contributing mechanisms in [`bench/h3_cover_workflow.py`](../h3_cover_workflow.py) and [`bench/h3_stem_sep.py`](../h3_stem_sep.py):
    1. **Upstream `librubberband` R2 phase-vocoder scaffold (`~71%` retained at `denoise = 0.29`):** `ffmpeg`'s built-in `rubberband` filter exposes only the R2 phase-vocoder (`phase=laminar`, not R3 `finer`). At `denoise = 0.29`, Pass 2 retains ~71% of the phase-vocoded guide structure.
    2. **`MiniMaxH3Sol` + `int8_convrot` + `12`-step Euler `BasicGuider` in Pass 2:** `_wire_sol` configures `start_percent = 0.2`, `tau = 1.0`, `dense_blocks = "49"`, `quantizer = "rotated"`, and `rows.audio_rows = "exact"` over Base `unet_ref2va` (`int8_convrot`), meaning Pass 2 (`percent = 0.71..1.0`) runs `100%` through sparse-routed quantized attention for only `12` Euler steps (`sigma_audio(0.29) = 0.55 -> 0`) at `cfg = 1.0`.
    3. **`AudioVAE` (`32 kHz` / `40 Hz` latent frame bottleneck):** Native `16 kHz` Nyquist ceiling and `25 ms` frame quantization across mixed stems.
  - **The SDEdit Catch-22:** `denoise = 0.29` holds melody (`lag_null = 1.00`) precisely because it retains `~71%` of the phase-vocoded scaffold through quantized sparse Pass-2 attention; raising `denoise` high enough (`0.40+`) to overwrite the phase-vocoder texture causes H3 (which lacks explicit F0 pitch conditioning) to drift off melody.

## 4. Decision & Tomorrow's 3-Step Decision Gate (`~15 min` to Clear Verdict)

- **Pause further multi-song GPU runs tonight:** Deferred additional multi-song / cross-gender covers until audio quality is isolated cleanly.
- **Ordered triage & pivot criteria when resuming:**
  1. **`AudioVAE` encode-decode null (`0` DiT steps, ~10 s):** Pass an unshifted dry studio vocal and a `-5.0 st` RubberBand R2 scaffold through `minimax_h3_audio_vae_fp32.safetensors` encode -> decode only, listening side-by-side to isolate how much compression comes from `AudioVAE` (`32 kHz / 40 Hz`) vs `librubberband` R2 vs Pass-2 DiT.
  2. **One "all-approximations-off" Pass-2 ceiling render (`1` window `294f`, ~5 min on `:8188`):** Re-render the `1`-window `-5.0 st` chorus control at matched seed bundling three Pass-2 upgrades together as a single upper-bound check:
     - **Bypass `MiniMaxH3Sol` on Pass 2** (`294f` fits cleanly below the `99,864`-token int32 threshold -> `100%` dense attention, zero rotated KV quantization).
     - **Swap checkpoint `int8_convrot` -> `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`** (higher precision weights).
     - **Increase `pass2_steps` from `12` -> `28`** (finer Euler trajectory down to `sigma = 0`).
     - *Decision rule:* If the ceiling render sounds substantially cleaner and less compressed, ablate which of the three upgrades drove it; if it still carries the compressed / auto-tune sheen, stop tuning Pass-2 attention and step counts because the ceiling lies in low-denoise RubberBand SDEdit itself.
  3. **If Step 2 hits the SDEdit ceiling — decouple vocal conversion from video generation (or pivot queues):** Move vocal timbre/pitch conversion out of H3 Pass-2 SDEdit onto a dedicated pitch-locked zero-shot F0-conditioned SVC (`44.1/48 kHz`, or RubberBand R3 `engine=finer` upstream), mix a clean studio-grade cover track, and feed that audio straight into `MiniMaxH3FreezeAudio(audio_mask = 0.0)` on 1-pass 6-step Pika so H3 handles video, camera performance, and 1:1 lip-sync without degrading the vocals — or pivot GPU focus to the remaining `step0_critic` / `self_mask` / Studio queues.
