# What the Unscored Frames of the 243-Frame Cover Gate Hold (`2026-11-27`)

- **Verdict:** In `c1`..`c4` the target's voiced frames that go unscored are not silent, not off the beat and not lost by the separator: the output is as loud there as in its scored frames and YIN finds no pitch in it. They are mostly the frames where the target itself is least steady, and there are more of them at higher `denoise`. What the unpitched sound is has to be judged by ear; no session has listened.
- **Date:** `2026-11-27`
- **Extends:** [`2026-11-27_music_cover_pitch_lock_metric_null.md`](2026-11-27_music_cover_pitch_lock_metric_null.md), section 5 (it could not say why most frames have no pitch) and section 6 (scored from a different separator, on another machine).
- **Scripts:** [`bench/pitch_lock_coverage.py`](../pitch_lock_coverage.py), [`bench/check_pitch_lock_coverage.py`](../check_pitch_lock_coverage.py), [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py)
- **Inputs:** the gate's own files, no new renders. Under `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/`: `<arm>_00001-audio.mp4` and the gate's `MelBand-RoFormer` stems `vocal_stems/<arm>_voc.wav` ("gate" below). Under `$H3_INPUT_DIR/music_cover_stems/`: the targets `flicker_voc_m5_{verse,chorus,bridge}_243f.wav`, the scaffolds `flicker_m5_{verse,chorus}_cal_243f.wav` and the reference voice `kurt_vocal_clean_192f.wav`.
- **Second separator:** `sepaudio stems <clip> -r six` (the `sepaudio` project at `f789100`, whose code is that of `2d3570b`), written to a scratch folder; "`six`" below is its `vocals.wav`.
- **Conditions:** CPU on the card machine while the gate's `c5`..`c9` run held the card, `sepaudio` with the card hidden. `c1`..`c4` are scored from gate stems. `c5`, `c6` and `c7` are scored from `six` vocals, because the gate had not yet written stems for them. `c8` and `c9` had not rendered.

---

## 1. The gate's own stems against the time-shift null

Every cell is aligned / null median / null best / share of `33` shifts beaten, as in the earlier record.

| Arm | Vocal stem | `voiced_overlap_frac` (frames) | `best_tr_st` | `med_abs_cent_err` | `in_tune_frac_50c` | `rel_med_abs_cent_err` | `rel_in_tune_frac_50c` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | gate | `0.308` (`193` of `627`) | `+0` | `8.7` / `256.9` / `108.4` / `1.00` | `0.917` / `0.154` / `0.423` / `1.00` | `8.7` / `141.7` / `85.5` / `1.00` | `0.917` / `0.143` / `0.294` / `1.00` |
| `c2_verse_m5_d32` | gate | `0.089` (`56` of `627`) | `+0` | `39.9` / `342.6` / `86.9` / `1.00` | `0.536` / `0.000` / `0.256` / `1.00` | `39.9` / `34.1` / `10.1` / `0.42` | `0.536` / `0.590` / `0.889` / `0.39` |
| `c3_chorus_m5_d30` | gate | `0.292` (`192` of `657`) | `+0` | `40.3` / `335.8` / `191.2` / `1.00` | `0.542` / `0.026` / `0.190` / `1.00` | `40.3` / `139.8` / `78.2` / `1.00` | `0.542` / `0.141` / `0.463` / `1.00` |
| `c4_verse_m5_rain_a2v0_d30` | gate | `0.367` (`230` of `627`) | `+0` | `48.9` / `217.6` / `98.7` / `1.00` | `0.513` / `0.096` / `0.336` / `1.00` | `48.9` / `143.6` / `93.9` / `1.00` | `0.513` / `0.133` / `0.348` / `1.00` |
| `c5_verse_m5_d30_s1` | `six` | `0.183` (`115` of `627`) | `+0` | `54.2` / `278.3` / `46.5` / `0.97` | `0.470` / `0.073` / `0.514` / `0.97` | `54.2` / `122.5` / `32.5` / `0.82` | `0.470` / `0.284` / `0.618` / `0.76` |
| `c6_verse_m5_d30_s2` | `six` | `0.164` (`103` of `627`) | `+0` | `49.2` / `250.8` / `70.4` / `1.00` | `0.544` / `0.059` / `0.443` / `1.00` | `49.2` / `82.3` / `17.6` / `0.79` | `0.544` / `0.299` / `0.760` / `0.88` |
| `c7_bridge_peak_m5_d30` | `six` | `0.364` (`231` of `634`) | `+0` | `23.6` / `269.5` / `129.9` / `1.00` | `0.623` / `0.060` / `0.350` / `1.00` | `23.6` / `135.9` / `90.5` / `1.00` | `0.623` / `0.121` / `0.396` / `1.00` |

1. **The gate's stems give the reading the other separator gave.** `c1` and `c3` beat every shift on all four columns and `c2` cannot be read, as in section 6 of the earlier record. The pitch columns of the four gate rows equal the gate's own `data/pure_h3_cover_song/cal_triage_report.json` as it stood before the `c5`..`c9` run, so the files and the scoring are the gate's. Only `vocal_med_f0_hz` of `c1` differs, in its last digit: the gate scores each stem in memory, and this reads the 16-bit file it wrote.
2. **`c4` and `c7` also beat every shift on all four columns.** `c7` is the window above the chorus that the earlier record listed as untested.
3. **The two verse seeds at `denoise` `0.30` sit at the edge.** `c6` beats every shift on the two untransposed columns and `c5` misses one shift on each. Both sit inside their null on the transposed columns, on fewer scored frames than `c1`.

## 2. Where the unscored frames go

`frames` splits the target's voiced frames into scored (a pitch in both), silent (output under `probe_yin_f0_drift.DEFAULT_MIN_DBFS`) and unpitched (output above it, no pitch found). Levels are frame medians in dBFS; `min_cmndf` is YIN's periodicity figure, lower when a frame is more periodic.

| Arm | Target voiced | Scored | Silent | Unpitched | Output pitch outside target | Best overlap shift | Output level, scored / missed | Output `min_cmndf`, scored / missed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `627` | `193` | `0` | `434` | `8` | `0 ms` | `-18.46` / `-18.59` | `0.07` / `0.29` |
| `c2_verse_m5_d32` | `627` | `56` | `0` | `571` | `6` | `-20 ms` | `-17.96` / `-18.5` | `0.08` / `0.36` |
| `c3_chorus_m5_d30` | `657` | `192` | `3` | `462` | `6` | `10 ms` | `-17.1` / `-20.38` | `0.1` / `0.32` |
| `c4_verse_m5_rain_a2v0_d30` | `627` | `230` | `0` | `397` | `10` | `0 ms` | `-18.87` / `-19.56` | `0.1` / `0.28` |

1. **Not silence.** Almost no missed frame is under the level gate, and in the three verse arms the output is as loud in its missed frames as in its scored ones. In `c3` the missed frames are quieter, though still above the gate.
2. **Not a shifted rhythm.** The two voicings overlap most at no shift or within a few frames of it, and the output has a pitch in almost no frame where the target has none.

## 3. Two separators and the unseparated mix name the same frames

`agree`, within the target's voiced frames.

| Arm | Gate stem and `six` vocals: both / gate only / `six` only / neither | Same pitch where both | Gate stem and unseparated mix: both / gate only / mix only / neither | Same pitch where both |
| :--- | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `167` / `26` / `0` / `434` | `1.0` | `163` / `30` / `0` / `434` | `1.0` |
| `c2_verse_m5_d32` | `53` / `3` / `1` / `570` | `1.0` | `42` / `14` / `0` / `571` | `1.0` |
| `c3_chorus_m5_d30` | `173` / `19` / `5` / `460` | `1.0` | `104` / `88` / `0` / `465` | `0.952` |
| `c4_verse_m5_rain_a2v0_d30` | `222` / `8` / `6` / `391` | `0.995` | `208` / `22` / `0` / `397` | `0.995` |

**Not a weak stem.** The frames with no pitch in the gate's stem have none in the other separator's stem, bar a handful, and none in the mix before separation. Separation adds frames to what the mix already gives, most in `c3`; the mix has a pitch in no frame where the gate's stem lacks one.

## 4. Whether the unpitched frames still follow the melody

`frames` loosens YIN's periodicity threshold on the output only and scores the frames that gain a pitch, aligned and against the target rotated in time.

| Arm | Threshold | `voiced_overlap_frac` | Gained frames | Gained frames in tune: aligned / null median / null best / share of shifts beaten |
| :--- | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.25` | `0.53` | `139` | `0.468` / `0.125` / `0.351` / `1.00` |
| `c1_verse_m5_d26` | `0.35` | `0.766` | `287` | `0.362` / `0.128` / `0.329` / `1.00` |
| `c1_verse_m5_d26` | `0.5` | `0.903` | `373` | `0.354` / `0.121` / `0.296` / `1.00` |
| `c2_verse_m5_d32` | `0.25` | `0.276` | `117` | `0.094` / `0.042` / `0.186` / `0.73` |
| `c2_verse_m5_d32` | `0.35` | `0.528` | `275` | `0.084` / `0.063` / `0.175` / `0.67` |
| `c2_verse_m5_d32` | `0.5` | `0.829` | `464` | `0.097` / `0.065` / `0.154` / `0.82` |
| `c3_chorus_m5_d30` | `0.25` | `0.534` | `159` | `0.384` / `0.037` / `0.207` / `1.00` |
| `c3_chorus_m5_d30` | `0.35` | `0.671` | `249` | `0.402` / `0.056` / `0.169` / `1.00` |
| `c3_chorus_m5_d30` | `0.5` | `0.776` | `318` | `0.371` / `0.051` / `0.177` / `1.00` |
| `c4_verse_m5_rain_a2v0_d30` | `0.25` | `0.609` | `152` | `0.237` / `0.091` / `0.493` / `0.91` |
| `c4_verse_m5_rain_a2v0_d30` | `0.35` | `0.775` | `256` | `0.215` / `0.102` / `0.418` / `0.97` |
| `c4_verse_m5_rain_a2v0_d30` | `0.5` | `0.911` | `341` | `0.229` / `0.099` / `0.374` / `0.97` |

1. **In `c1` and `c3` the unpitched frames follow the target melody, less closely than the scored ones.** The frames that gain a pitch beat every shift at each threshold. Their in-tune share is below that of the scored frames (section 1) and below what the check reads for a known right melody under noise (`check_rough_voice_is_credited_and_beats_its_null` prints it).
2. **In `c2` they cannot be tied to the melody.** Its gained frames sit inside their null at every threshold, so nothing in the unscored part of `c2` shows the target's notes.
3. **`c4` is between the two.** Its gained frames beat most shifts and not all.

## 5. Which of the target's frames keep their pitch

Scored share in each quarter of the target's voiced frames, ranked by the target's own `min_cmndf`, level and pitch.

| Arm | Vocal stem | By target `min_cmndf` (steadiest first) | By target level (quietest first) | By target pitch (lowest first) |
| :--- | :--- | :---: | :---: | :---: |
| `c1_verse_m5_d26` | gate | `0.459` / `0.439` / `0.229` / `0.103` | `0.204` / `0.197` / `0.401` / `0.429` | `0.057` / `0.280` / `0.382` / `0.513` |
| `c2_verse_m5_d32` | gate | `0.153` / `0.115` / `0.076` / `0.013` | `0.102` / `0.089` / `0.083` / `0.083` | `0.006` / `0.032` / `0.057` / `0.263` |
| `c3_chorus_m5_d30` | gate | `0.497` / `0.335` / `0.213` / `0.122` | `0.085` / `0.268` / `0.372` / `0.445` | `0.279` / `0.268` / `0.463` / `0.159` |
| `c4_verse_m5_rain_a2v0_d30` | gate | `0.510` / `0.401` / `0.344` / `0.212` | `0.261` / `0.376` / `0.471` / `0.359` | `0.376` / `0.274` / `0.382` / `0.436` |
| `c5_verse_m5_d30_s1` | `six` | `0.318` / `0.172` / `0.172` / `0.071` | `0.191` / `0.159` / `0.204` / `0.179` | `0.127` / `0.172` / `0.127` / `0.308` |
| `c6_verse_m5_d30_s2` | `six` | `0.242` / `0.261` / `0.102` / `0.051` | `0.172` / `0.083` / `0.127` / `0.276` | `0.038` / `0.057` / `0.261` / `0.301` |
| `c7_bridge_peak_m5_d30` | `six` | `0.642` / `0.352` / `0.304` / `0.158` | `0.069` / `0.314` / `0.456` / `0.620` | `0.321` / `0.409` / `0.399` / `0.329` |

1. **The target's steadiest frames are the ones that survive, in every arm.** The quarter of frames where the target vocal is most periodic is scored several times as often as the quarter where it is least. Every one of these target frames already passes YIN, so this is a ranking among clean frames.
2. **Level and pitch each matter in some arms and not in others.** The second and third columns do not order the same way from arm to arm: level makes no difference in `c2` and `c5`, pitch makes none in `c7`, and in `c3` it is the highest quarter that is lost where in `c1`, `c2` and `c6` it is the lowest. Within a phrase the three rankings move together, and nothing here separates them.

The same four verse arms at one prompt, from `six` vocals so that one separator covers the series:

| Arm | `denoise` | Seed | Scored of target voiced | Gained at `0.35`: frames | In tune: aligned / null median / null best / share beaten |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.26` | `SEED` | `167` of `627` | `284` | `0.394` / `0.121` / `0.344` / `1.00` |
| `c5_verse_m5_d30_s1` | `0.30` | `SEED` | `115` of `627` | `321` | `0.231` / `0.091` / `0.438` / `0.94` |
| `c6_verse_m5_d30_s2` | `0.30` | `SEED + 1000` | `103` of `627` | `256` | `0.445` / `0.113` / `0.371` / `1.00` |
| `c2_verse_m5_d32` | `0.32` | `SEED` | `54` of `627` | `272` | `0.088` / `0.059` / `0.181` / `0.76` |

3. **Scored frames fall as `denoise` rises.** At one seed and one prompt the count drops at each step, and the second seed at `0.30` lands beside the first. That is one pair of seeds.

## 6. How periodic the voices are

`periodic`: the share of a file's active frames (within `ACTIVE_BELOW_LOUD_DB` of its loud frames) in which YIN finds a pitch.

| File | Active frames | Share with a pitch | `min_cmndf` quartiles |
| :--- | :---: | :---: | :---: |
| `flicker_voc_m5_verse_243f.wav` | `786` | `0.782` | `0.01` / `0.03` / `0.10` |
| `flicker_voc_m5_chorus_243f.wav` | `853` | `0.728` | `0.03` / `0.07` / `0.17` |
| `kurt_vocal_clean_192f.wav` | `486` | `0.714` | `0.04` / `0.07` / `0.19` |
| `c1_verse_m5_d26_voc.wav` | `793` | `0.253` | `0.14` / `0.27` / `0.46` |
| `c2_verse_m5_d32_voc.wav` | `816` | `0.075` | `0.26` / `0.37` / `0.51` |
| `c3_chorus_m5_d30_voc.wav` | `819` | `0.242` | `0.15` / `0.27` / `0.49` |
| `c4_verse_m5_rain_a2v0_d30_voc.wav` | `819` | `0.293` | `0.13` / `0.25` / `0.44` |

**The rendered voice is far less periodic than the voice it imitates.** The reference recording and both targets have a pitch in most of their active frames; the four output vocals have one in a minority. The scaffold keeps nearly every frame through the second separator (earlier record, section 6) and through the audio VAE followed by the gate's separator (`controls.audiovae_roundtrip` in the gate's report), so neither the separation nor the VAE does this.

## 7. Stem levels

`levels`, in dB against each clip's own mix. The last six columns are the `six` stems.

| Mix | Gate stem | Mix minus gate stem | `six` vocals | guitar | bass | other | piano | drums |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `-0.28` | `-13.58` | `-0.10` | `-23.14` | `-90.95` | `-63.65` | `-101.55` | `-85.95` |
| `c2_verse_m5_d32` | `-0.27` | `-14.59` | `-0.21` | `-18.74` | `-70.93` | `-37.48` | `-28.24` | `-38.37` |
| `c3_chorus_m5_d30` | `-0.74` | `-10.15` | `-0.57` | `-28.80` | `-94.09` | `-16.13` | `-24.19` | `-37.07` |
| `c4_verse_m5_rain_a2v0_d30` | `-0.16` | `-15.98` | `-0.17` | `-19.52` | `-30.70` | `-26.04` | `-100.95` | `-85.56` |
| verse scaffold | | | `-0.51` | `-20.02` | `-10.48` | `-94.39` | `-99.94` | `-92.03` |
| chorus scaffold | | | `-0.38` | `-23.18` | `-13.17` | `-18.47` | `-97.79` | `-95.49` |

By level only, not by ear:

1. **Both separators put nearly the whole render in the vocal stem.** Each vocal stem is within a decibel of the mix, and taking the gate's stem out leaves a residue well under it.
2. **The scaffold's bass is not in the renders' bass stem.** The two scaffold rows show a bass under the vocal. In `c1`..`c3` the bass stem is close to empty, and in `c4` it is well below the scaffold's.
3. **The guitar stem of the renders is at about the level of the scaffolds' own.**

These are two separators trained on recorded music reading generated audio. Whether a guitar or a bass can be heard is a listening question.

## 8. Where to listen

The longest spans in each gate stem, in seconds and 24 fps frames of the clip.

| Arm | Unpitched while the target sings | Scored |
| :--- | :--- | :--- |
| `c1_verse_m5_d26` | `0.79`–`1.40 s` (`f18`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `5.64`–`6.06 s` (`f135`–`f145`), `7.97`–`9.05 s` (`f191`–`f217`) | `2.69`–`3.12 s` (`f64`–`f74`), `3.29`–`3.72 s` (`f78`–`f89`), `6.60`–`6.81 s` (`f158`–`f163`), `7.08`–`7.58 s` (`f169`–`f181`) |
| `c2_verse_m5_d32` | `0.27`–`1.40 s` (`f6`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `6.97`–`7.70 s` (`f167`–`f184`), `7.97`–`9.05 s` (`f191`–`f217`) | `1.63`–`1.77 s` (`f39`–`f42`), `2.71`–`3.10 s` (`f65`–`f74`), `3.32`–`3.38 s` (`f79`–`f81`), `6.33`–`6.38 s` (`f151`–`f153`) |
| `c3_chorus_m5_d30` | `4.57`–`5.00 s` (`f109`–`f120`), `5.21`–`5.65 s` (`f125`–`f135`), `7.20`–`7.70 s` (`f172`–`f184`), `9.06`–`10.07 s` (`f217`–`f241`) | `0.87`–`1.18 s` (`f20`–`f28`), `1.24`–`1.48 s` (`f29`–`f35`), `2.46`–`2.85 s` (`f59`–`f68`), `8.63`–`9.06 s` (`f207`–`f217`) |
| `c4_verse_m5_rain_a2v0_d30` | `0.47`–`1.40 s` (`f11`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `5.64`–`6.06 s` (`f135`–`f145`), `8.65`–`9.05 s` (`f207`–`f217`) | `2.66`–`3.11 s` (`f63`–`f74`), `3.29`–`3.64 s` (`f78`–`f87`), `6.59`–`6.82 s` (`f158`–`f163`), `8.32`–`8.68 s` (`f199`–`f208`) |

The three verse arms lose the same passages, which is section 5 seen in time. What to listen for in an unpitched span against a scored one in the same clip: rasp or breath on a held note, two voices at once, an instrument inside the voice, or a note that wanders.

## 9. Commands

```bash
V="$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27"; S="$V/vocal_stems"; I="$H3_INPUT_DIR/music_cover_stems"
X=<scratch folder>   # sepaudio stems "$V/<arm>_00001-audio.mp4" -r six -o "$X", run in the sepaudio checkout's own environment
"$PY" bench/pitch_lock_metrics.py "$S/c1_verse_m5_d26_voc.wav" "$I/flicker_voc_m5_verse_243f.wav"      # section 1
"$PY" bench/pitch_lock_coverage.py frames "$S/c1_verse_m5_d26_voc.wav" "$I/flicker_voc_m5_verse_243f.wav"  # sections 2, 4, 5, 8
"$PY" bench/pitch_lock_coverage.py agree "$S/c1_verse_m5_d26_voc.wav" "$X/c1_verse_m5_d26_00001-audio/vocals.wav" "$I/flicker_voc_m5_verse_243f.wav"  # section 3
"$PY" bench/pitch_lock_coverage.py periodic "$I/kurt_vocal_clean_192f.wav" "$S/c1_verse_m5_d26_voc.wav"  # section 6
"$PY" bench/pitch_lock_coverage.py levels "$V/c1_verse_m5_d26_00001-audio.mp4" "$S/c1_verse_m5_d26_voc.wav" "$X/c1_verse_m5_d26_00001-audio/"{vocals,guitar,bass,other,piano,drums}.wav  # section 7
```

The chorus arm takes `flicker_voc_m5_chorus_243f.wav` and `c7` takes `flicker_voc_m5_bridge_243f.wav`.

## 10. Limits

- No one has listened. "Unpitched" is YIN's judgment at `probe_yin_f0_drift.DEFAULT_THRESHOLD`. It does not tell rasp from breath, from two voices at once, or from an instrument a separator left in the vocal stem.
- A better separator might recover frames that neither of these two does. Section 3 shows the two agree, not that they are right.
- The in-tune share of gained frames is a floor. A looser threshold takes harmonics for the pitch more often, and the check's control is a tone under white noise, which a rough voice is not.
- One render per arm, apart from the one pair of seeds in section 5.
- The quarters of section 5 describe where pitch survives. They do not say why, and the three rankings are not independent.
- `c5`, `c6` and `c7` are scored from a different separator than `c1`..`c4`. Section 3 is the evidence that this changes little, on the four arms where both exist.
- The time-shift null has the limits listed in the earlier record, section 5.

## 11. Not covered

- `c8`, `c9` and the 4-window song.
- `c5`..`c7` from the gate's own stems, once it writes them.
- The choice of transposition by median error (earlier record, section 1, item 3). Every arm here is scored at no transposition, so it did not arise.
