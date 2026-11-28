# 6-Arm Pure-H3 Music-Video Cover Pitch-Lock Gate (`2026-11-27`)

- **Verdict:** **PARTIAL UNLOCK — VISUAL ATTRIBUTION & ACOUSTIC GUITAR SYNTHESIS SOLVED; ABSOLUTE PITCH LOCK (`128.2–457.6 c`) STILL ABOVE `< 80 c` BAR AT `denoise = 0.36–0.40`, WITH THREE ISOLATED CAUSAL LAWS EXTRACTED.**
- **Corrected (`2026-11-27`):** the pitch columns of section 2 were read without a chance level, and `vocal_f0_r` does not measure contour agreement. Rescored against time-shifted targets, no arm beats its own null, and the "interval lock" (`p1`/`p2`) and "highest contour" (`p3`) readings do not stand as written: [`2026-11-27_music_cover_pitch_lock_metric_null.md`](2026-11-27_music_cover_pitch_lock_metric_null.md). The text below is left as first recorded.
- **Date:** `2026-11-27`
- **Scripts:** [`bench/prepare_pitch_lock_assets.py`](../prepare_pitch_lock_assets.py), [`bench/music_cover_pitch_lock_gate.py`](../music_cover_pitch_lock_gate.py)
- **Data & Prompts:** `data/music_cover_pitch_lock/stem_prep_report.json`, `data/music_cover_pitch_lock/pitch_lock_triage_report.json`, `data/music_cover_pitch_lock/ref2va_unplugged_{flicker,flicker_neutral,whatever_192f,fridge}.txt`
- **Outputs:** `$H3_OUTPUT_DIR/Video/music_cover_pitch_lock_2026-11-27/p1..p6_00001-audio.mp4` and `vocal_stems/p1..p6_voc.wav`
- **Grid & Topology:** `1024x768`, `192` frames (`8.000 s` at `24.0 fps`, `320` `40 Hz` audio steps, `on_both_clocks`), `seed = 910427311`. Pass 1: 6-step Pika H3 Flex (`minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `pika_h3_flex_ref2va_svd995_bf16.safetensors` + `MiniMaxH3Sol` + `MiniMaxH3QueryRouter(section_split="soundscape_to_audio")` + `MiniMaxH3FreezeAudio(audio_mask=0.0)`). Pass 2: Base `ref2va` 12-step `euler/simple` + `MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` at `denoise = 0.36–0.40` (plus `2x MiniMaxH3EditGuider(streams="audio_only", scale=1.75, last_step=8)` on `p2`).

---

## 1. Hypotheses Tested Against Prior `ai_song_sdedit_covers` (`ai1`–`ai4`) Failures

1. **Solo-Kurt `<Picture 1,2>` + `QueryRouter(section_split="soundscape_to_audio")`:** Replace `unplugged_b_stage.png` (which had a foreground background guitarist on screen right who stole `[Shot 1]` lip-sync across `ai1`–`ai4`) with clean 4:3 solo-Kurt frames (`kurt_closeup.png` at `t=72 s`, `kurt_solo_medium.png` at `t=195 s`).
2. **De-droned `<Audio 1>` (`kurt_vocal_only_192f.wav`) vs. Capo-Matched `<Audio 1>` (`kurt_vocal_guitar_192f.wav`):** Strip the Drop-C# (`C# + G#`) guitar drone from `<Audio 1>` via `MelBandRoformer_fp16.safetensors` (`194.69 Hz` solo vocal vs. `69.69 Hz` full mix) or transpose Song X by `-5 st` (`formant=preserved`) onto C#.
3. **Clean `MelBand-RoFormer` Vocal Stem vs. `-11 dB` `STFT-HPSS (80–1200 Hz)` Chord Bed:** Test whether feeding a dry vocal stem alone (`p3`) vs. vocal + HPSS harmonic bed (`p1`, `p2`, `p4`, `p5`, `p6`) holds both melody and clean acoustic guitar accompaniment.

---

## 2. Measured 6-Arm Results (`data/music_cover_pitch_lock/pitch_lock_triage_report.json`)

Vocal cent errors are computed via `probe_yin_f0_drift.yin_f0` (`24 kHz`) on `MelBand-RoFormer`-isolated output vocal stems (`vocal_stems/<arm>_voc.wav`) against each arm's target vocal stem (`src_voc`). Both **unshifted target-key error** (`med_abs_cent_err`, `med_signed_cent_bias`) and **best-semitone-transposed interval error** (`best_tr_st`, `rel_med_abs_cent_err`, `rel_in_tune_frac_50c`) are reported so global register/key pull is cleanly separated from internal melodic interval fidelity.

| Arm | Song & Scaffold (`x_0[audio]`) | `<Audio 1>` Ref | `denoise` (`sigma_aud`) | Wall (s) | Scaf -> Out Med `F0` (Hz) | Target-Key `med_abs_cent_err` (Bias) | Best Transp `best_tr_st` -> `rel_med_err` (`in50c`) | Contour `vocal_f0_r` | `chroma_sameKey` / `anyKey` (`drift`) | Kurt `ltas_cos` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `p1_flicker_chord_d36` | `flicker` (`-12 st` voc + HPSS) | `kurt_vocal_guitar` | `0.36` (`0.628`) | `319.0` | `132.1 -> 185.4` | `455.7 c` (`+444.7 c`) | **`+5 st -> 71.4 c` (`39.3%`)** | `0.495` | `0.165` / `0.237` (`+5 st`) | `0.923` |
| `p2_flicker_editguider_d40` | `flicker` (`-12 st` voc + HPSS + `2x EditGuider`) | `kurt_vocal_guitar` | `0.40` (`0.667`) | `368.6` | `132.1 -> 201.5` | `457.6 c` (`+434.4 c`) | **`+5 st -> 68.6 c` (`40.4%`)** | `0.606` | `0.145` / `0.245` (`+5 st`) | **`0.936`** |
| `p3_flicker_voconly_d40` | `flicker` (**`-12 st` voc ONLY**, zero HPSS) | `kurt_vocal_only` | `0.40` (`0.667`) | `309.0` | `155.0 -> 189.6` | `250.2 c` (`+250.2 c`) | `+4 st -> 149.8 c` (`8.8%`) | **`0.788`** | **`0.243`** / `0.243` (`0 st`) | `0.830` |
| `p4_whatever_dedroned_origkey_d38` | `whatever` (`-12 st` voc + HPSS, orig F#) | `kurt_vocal_only` | `0.38` (`0.648`) | `298.8` | `123.8 -> 183.6` | `382.0 c` (`+200.0 c`) | `+5 st -> 167.7 c` (`10.2%`) | `0.671` | `0.126` / `0.198` (`+4 st`) | `0.793` |
| `p5_whatever_capo_csharp_d38` | `whatever` (**`-5 st` `formant=preserved`** + HPSS) | `kurt_vocal_guitar` | `0.38` (`0.648`) | `306.5` | **`186.7 -> 203.6`** | **`128.2 c` (`-21.5 c`)** | `+1 st -> 121.5 c` (`8.6%`) | `0.139` | `0.040` / `0.160` (`-3 st`) | `0.903` |
| `p6_fridge_gfunk_unplugged_d40` | `In The Fridge` (`0 st` male voc + HPSS) | `kurt_vocal_only` | `0.40` (`0.667`) | `307.3` | `184.9 -> 193.6` | `300.7 c` (`+165.1 c`) | `+3 st -> 142.2 c` (`10.7%`) | `0.582` | `0.132` / **`0.303`** (`+2 st`) | `0.667` |

---

## 3. Native Audio-Visual Audit (`read_file` Across All 6 `-audio.mp4` Renders)

- **Visual Attribution (`p1`–`p6`):** **6/6 PASS.** Replacing `unplugged_b_stage.png` with solo `kurt_solo_medium.png` (`[Shot 1]`, `00:00–00:03`) and `kurt_closeup.png` (`[Shot 2]`, `00:04–00:07`) under Pass-1 `MiniMaxH3QueryRouter(section_split="soundscape_to_audio")` completely eliminated background-singer lip-sync theft. Kurt sits center-frame strumming the acoustic guitar and mouthing lyrics into the vocal mic across `[Shot 1]`, cutting cleanly at `00:04` to the tight close-up for `[Shot 2]`.
- **`p1_flicker_chord_d36` & `p2_flicker_editguider_d40`:** Audibly in-key *with themselves* (`rel_med_abs_cent_err = 71.4 c` and `68.6 c`, `39.3–40.4%` within `+-50 c`), because H3 transposed the entire song up a clean fourth (`+5 st`, lifting the `-12 st` `132–155 Hz` baritone scaffold into Kurt's natural `185–202 Hz` chest register). However, two artifacts degrade vocal naturalness: (1) shifting a high female vocal `-12 st` without `:formant=preserved` left a darkened, slowed formant warble underneath, and (2) the `STFT-HPSS (80–1200 Hz)` bed creates a phasey choral wash that blurs acoustic guitar transients.
- **`p3_flicker_voconly_d40` (Dry Vocal Stem Only, Zero HPSS):** **Cleanest live acoustic guitar accompaniment in the gate.** Removing the HPSS wash from `x_0[audio]` allowed H3's `40 Hz` AudioVAE + Unplugged `<Picture 1,2>` / `<Audio 1>` priors to synthesize a crisp, genuine steel-string acoustic guitar strum from scratch (`00:00–00:04` *"Just a flicker... in the rain"*) while holding the strongest F0 contour correlation (`vocal_f0_r = 0.788`, `chroma_drift_st = 0`). Without a clean harmonic stem in `x_0[audio]`, H3's invented guitar chooses its own bright major chord in `00:04–00:07`, and the un-formant-preserved `-12 st` vocal slurs the second line at `d=0.40`.
- **`p4_whatever_dedroned_origkey_d38` vs. `p5_whatever_capo_csharp_d38`:**
  - `p4` (`-12 st` un-formant-preserved, `123.8 Hz` scaffold + de-droned `kurt_vocal_only`): Crisp intelligible lyrics (*"Whatever comes, whoever he chooses to love / Whether joy finds him or not, I'll let him go"*, `onset_lag = -2`) and solid melodic contour shape (`vocal_f0_r = 0.671`), **but** even without the guitar drone in `<Audio 1>`, Kurt's `194.7 Hz` vocal prior pulled the `123.8 Hz` scaffold up `+4..+5 st` to `183.6 Hz` (`med_abs_cent_err = 382.0 c`).
  - `p5` (`-5 st` **with `formant=preserved`**, `186.7 Hz` scaffold + `kurt_vocal_guitar`): Best Kurt vocal timbre (`ltas_cos = 0.903`) and **near-zero signed register bias (`-21.5 c`, `in_tune_frac_50c = 32.3%`, `3.0x` lower `med_abs_cent_err = 128.2 c` than `p4`)**, proving that matching the input vocal median directly into Kurt's `185–205 Hz` chest register eliminates global upward pitch pull. However, pairing fast sixteenth-note pop phrasing with `kurt_vocal_guitar`'s `C#` drone at `denoise = 0.38` (`sigma_aud = 0.648`) flattened the melodic contour (`vocal_f0_r = 0.139`) into a gritty grunge chant.
- **`p6_fridge_gfunk_unplugged_d40`:** Rhythmic alignment is frame-exact (`onset_lag_x = 0`) and transforms West Coast G-funk rap-singing (*"Why is the vinegar in the fridge? ... I'm just tryna wipe down the ridge"*) into a snarling acoustic grunge delivery over strummed guitar, shifted `+2..+3 st` (`184.9 -> 193.6 Hz`, `chroma_anyKey_x = 0.303`).

---

## 4. Three Extracted Laws for Pure-H3 Cover & Full-Song Synthesis

1. **Law 1 — `ModelSamplingAV` `4x` Audio Shift Amplification (`denoise = 0.38` is `64.8%` Audio Noise):**
   - In `comfy/model_sampling.py::ModelSamplingAV` (`shift_video = 12.0`, `shift_audio = 3.0`, `audio_scale = 4.0`), `BasicScheduler(denoise = d)` schedules `sigma_vid = 12d / (1 + 11d)`, which places the audio stream at effective flow noise $\sigma_{\text{aud}} = 3d / (1 + 2d)$:
     - `denoise = 0.25` -> $\sigma_{\text{aud}} = 0.500$ (`0 dB` 1:1 signal/noise crossover)
     - `denoise = 0.36..0.40` -> $\sigma_{\text{aud}} = 0.628..0.667$ (`63–67%` Gaussian noise, burying fine harmonic F0 spacing on fast melodies)
   - Once $\sigma_{\text{aud}} > 0.60$, early flow steps re-anchor global pitch to the target singer's natural vocal register (`183.6–203.6 Hz` across all 6 arms).
2. **Law 2 — Register-Match (`185–200 Hz`) with `formant=preserved` (Never `-12 st` Unpreserved):**
   - Shifting soprano vocals `-12 st` (`124–155 Hz`) undershoots Kurt's `194.7 Hz` Unplugged register by `4–5 st`, forcing H3 to transpose upward (`+250..+445 c`) while leaving slowed formant warble. Shifting `-5` to `-7 st` **with `rubberband=formant=preserved`** lands directly on `185–200 Hz` (`med_signed_cent_bias = -21.5 c` on `p5`).
3. **Law 3 — Never Put `STFT-HPSS` Beds in `x_0[audio]`; Use Dry Vocal or Phase-Clean Multi-Stem (`guitar + bass`):**
   - `p3` proved H3 synthesizes crisp steel-string acoustic guitar directly when `x_0[audio]` has zero HPSS smear, while `p1`/`p2` proved a subtle harmonic root reference locks internal melodic intervals (`68.6–71.4 c`). Multi-stem separation (`vocals`, `guitar`, `bass`, `piano`, `drums`, `other` with `drums` + synth `other` dropped, rather than `STFT-HPSS`) supplies phase-clean chord roots without smearing guitar transients, ready to chain across `39 + 51m` (`243f` / `345f`) windows with `context_profile="seam_taper"`.
