# What the Unscored Frames of the 243-Frame Cover Gate Hold (`2026-11-27`)

- **Verdict:** In the seven `justaflicker` arms the target's voiced frames that go unscored are not silent, not off the beat and not lost by the separator: the output is as loud there as in its scored frames and YIN finds no pitch in it. They are mostly the frames where the target itself is least steady, and there are more of them at higher `denoise`. The two other-song arms cannot be read, because their target vocals are mostly unpitched before any render. Accompaniment is left in the vocal stems, and it is too quiet to explain the loss (section 11). The gate's timbre figure does not show whether the singer changed (section 9). What the unpitched sound is has to be judged by ear; no session has listened.
- **Date:** `2026-11-27`
- **Corrected the same day:** section 7 first read the six-stem levels as the bass having left the renders and the guitar stem gaining nothing. Section 11 measures the vocal rests and shows otherwise; items 2 and 3 of section 7 are rewritten. The other session's listening prompted the check.
- **Extends:** [`2026-11-27_music_cover_pitch_lock_metric_null.md`](2026-11-27_music_cover_pitch_lock_metric_null.md), section 5 (it could not say why most frames have no pitch) and section 6 (scored from a different separator, on another machine).
- **Scripts:** [`bench/pitch_lock_coverage.py`](../pitch_lock_coverage.py), [`bench/check_pitch_lock_coverage.py`](../check_pitch_lock_coverage.py), [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py)
- **Inputs:** the gate's own files, no new renders. Under `$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27/`: `<arm>_00001-audio.mp4` and the gate's `MelBand-RoFormer` stems `vocal_stems/<arm>_voc.wav` ("gate stem" below). Under `$H3_INPUT_DIR/music_cover_stems/`: each arm's target vocal and scaffold, the three songs' stem folders, and the reference voice `kurt_vocal_clean_192f.wav`.
- **Second separator:** `sepaudio stems <clip> -r six` (the `sepaudio` project at `f789100`, whose code is that of `2d3570b`), written to a scratch folder; "`six`" below is one of its six stems.
- **Conditions:** CPU on the card machine, `sepaudio` with the card hidden. All nine arms are scored from the gate stems as the gate's analysis of all arms left them. The first commit of this record scored `c5`..`c7` from `six` vocals, before those gate stems existed; the reading did not change.

---

## 1. The gate's own stems against the time-shift null

Every cell is aligned / null median / null best / share of shifts beaten, as in the earlier record.

| Arm | `voiced_overlap_frac` (frames) | Shifts | `best_tr_st` | `med_abs_cent_err` | `in_tune_frac_50c` | `rel_med_abs_cent_err` | `rel_in_tune_frac_50c` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.308` (`193` of `627`) | `33` | `+0` | `8.7` / `256.9` / `108.4` / `1.00` | `0.917` / `0.154` / `0.423` / `1.00` | `8.7` / `141.7` / `85.5` / `1.00` | `0.917` / `0.143` / `0.294` / `1.00` |
| `c2_verse_m5_d32` | `0.089` (`56` of `627`) | `33` | `+0` | `39.9` / `342.6` / `86.9` / `1.00` | `0.536` / `0.000` / `0.256` / `1.00` | `39.9` / `34.1` / `10.1` / `0.42` | `0.536` / `0.590` / `0.889` / `0.39` |
| `c3_chorus_m5_d30` | `0.292` (`192` of `657`) | `33` | `+0` | `40.3` / `335.8` / `191.2` / `1.00` | `0.542` / `0.026` / `0.190` / `1.00` | `40.3` / `139.8` / `78.2` / `1.00` | `0.542` / `0.141` / `0.463` / `1.00` |
| `c4_verse_m5_rain_a2v0_d30` | `0.367` (`230` of `627`) | `33` | `+0` | `48.9` / `217.6` / `98.7` / `1.00` | `0.513` / `0.096` / `0.336` / `1.00` | `48.9` / `143.6` / `93.9` / `1.00` | `0.513` / `0.133` / `0.348` / `1.00` |
| `c5_verse_m5_d30_s1` | `0.214` (`134` of `627`) | `33` | `+0` | `60.6` / `277.7` / `52.3` / `0.97` | `0.463` / `0.073` / `0.494` / `0.97` | `60.6` / `132.0` / `35.6` / `0.88` | `0.463` / `0.100` / `0.591` / `0.88` |
| `c6_verse_m5_d30_s2` | `0.169` (`106` of `627`) | `33` | `+0` | `48.2` / `227.9` / `80.1` / `1.00` | `0.566` / `0.083` / `0.394` / `1.00` | `48.2` / `101.1` / `21.6` / `0.91` | `0.566` / `0.261` / `0.673` / `0.94` |
| `c7_bridge_peak_m5_d30` | `0.377` (`239` of `634`) | `33` | `+0` | `20.3` / `271.0` / `129.5` / `1.00` | `0.644` / `0.058` / `0.349` / `1.00` | `20.3` / `131.0` / `84.9` / `1.00` | `0.644` / `0.124` / `0.417` / `1.00` |
| `c8_kite_chorus_d30` | `0.074` (`21` of `283`) | `10` | `+1` | `109.5` / `183.5` / `16.3` / `0.80` | `0.381` / `0.113` / `0.833` / `0.80` | `96.9` / `104.9` / `16.3` / `0.60` | `0.190` / `0.225` / `0.917` / `0.40` |
| `c9_fridge_chorus_d30` | `0.098` (`22` of `224`) | - | `+0` | no null | no null | no null | no null |

Targets: `flicker_voc_m5_verse_243f.wav` for `c1`, `c2`, `c4`, `c5` and `c6`; `flicker_voc_m5_chorus_243f.wav` for `c3`; `flicker_voc_m5_bridge_243f.wav` for `c7`; `kite_voc_m6_243f.wav` for `c8`; `fridge_voc_p2_243f.wav` for `c9`.

1. **The gate's stems give the reading the other separator gave.** `c1` and `c3` beat every shift on all four columns and `c2` cannot be read, as in section 6 of the earlier record. The pitch columns here equal those of the gate's own `data/pure_h3_cover_song/cal_triage_report.json` in every arm but `c9`, apart from the last digit of `med_signed_cent_bias` in `c5`. In `c9` one frame differs, and on a score of about twenty frames that moves every column: the gate scores each stem in memory, and this reads the 16-bit file it wrote.
2. **`c4` and `c7` also beat every shift on all four columns.** `c7` is the window above the chorus that the earlier record listed as untested.
3. **The two verse seeds at `denoise` `0.30` sit at the edge.** `c6` beats every shift on the two untransposed columns and `c5` misses one shift on each. Neither beats every shift on the transposed columns, and both rest on fewer scored frames than `c1`.
4. **`c8` and `c9` cannot be read.** Each rests on about twenty frames; `c8` has few shifts to compare against and `c9` too few for a null at all. Section 6 says why.

## 2. Where the unscored frames go

`frames` splits the target's voiced frames into scored (a pitch in both), silent (output under `probe_yin_f0_drift.DEFAULT_MIN_DBFS`) and unpitched (output above it, no pitch found). Levels are frame medians in dBFS; `min_cmndf` is YIN's periodicity figure, lower when a frame is more periodic. The overlap column gives the shift at which the two voicings share the most frames, and that count against the count at no shift.

| Arm | Target voiced | Scored | Silent | Unpitched | Output pitch outside target | Best overlap shift | Output level, scored / missed | Output `min_cmndf`, scored / missed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `627` | `193` | `0` | `434` | `8` | `0 ms` (`193` against `193`) | `-18.46` / `-18.59` | `0.07` / `0.29` |
| `c2_verse_m5_d32` | `627` | `56` | `0` | `571` | `6` | `-20 ms` (`58` against `56`) | `-17.96` / `-18.50` | `0.08` / `0.36` |
| `c3_chorus_m5_d30` | `657` | `192` | `3` | `462` | `6` | `10 ms` (`194` against `192`) | `-17.10` / `-20.38` | `0.10` / `0.32` |
| `c4_verse_m5_rain_a2v0_d30` | `627` | `230` | `0` | `397` | `10` | `0 ms` (`230` against `230`) | `-18.87` / `-19.56` | `0.10` / `0.28` |
| `c5_verse_m5_d30_s1` | `627` | `134` | `0` | `493` | `13` | `-20 ms` (`137` against `134`) | `-18.02` / `-18.75` | `0.10` / `0.30` |
| `c6_verse_m5_d30_s2` | `627` | `106` | `0` | `521` | `12` | `0 ms` (`106` against `106`) | `-18.99` / `-19.41` | `0.08` / `0.36` |
| `c7_bridge_peak_m5_d30` | `634` | `239` | `5` | `390` | `0` | `0 ms` (`239` against `239`) | `-17.01` / `-19.43` | `0.10` / `0.28` |
| `c8_kite_chorus_d30` | `283` | `21` | `0` | `262` | `12` | `-170 ms` (`24` against `21`) | `-17.87` / `-17.55` | `0.13` / `0.34` |
| `c9_fridge_chorus_d30` | `224` | `22` | `8` | `194` | `15` | `-70 ms` (`28` against `22`) | `-15.26` / `-17.30` | `0.11` / `0.33` |

1. **Not silence.** Almost no missed frame is under the level gate. In the verse arms the output is as loud in its missed frames as in its scored ones; in `c3` and `c7` the missed frames are quieter, though still above the gate.
2. **Not a shifted rhythm.** In the `justaflicker` arms the two voicings overlap most at no shift or within a few frames of it, and no shift adds more than a few frames. In those arms the output has a pitch in few frames where the target has none.

## 3. Two separators and the unseparated mix name the same frames

`agree`, within the target's voiced frames.

| Arm | Gate stem and `six` vocals: both / gate only / `six` only / neither | Same pitch where both | Gate stem and unseparated mix: both / gate only / mix only / neither | Same pitch where both |
| :--- | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `167` / `26` / `0` / `434` | `1.0` | `163` / `30` / `0` / `434` | `1.0` |
| `c2_verse_m5_d32` | `53` / `3` / `1` / `570` | `1.0` | `42` / `14` / `0` / `571` | `1.0` |
| `c3_chorus_m5_d30` | `173` / `19` / `5` / `460` | `1.0` | `104` / `88` / `0` / `465` | `0.952` |
| `c4_verse_m5_rain_a2v0_d30` | `222` / `8` / `6` / `391` | `0.995` | `208` / `22` / `0` / `397` | `0.995` |
| `c5_verse_m5_d30_s1` | `115` / `19` / `0` / `493` | `0.983` | `77` / `57` / `0` / `493` | `0.974` |
| `c6_verse_m5_d30_s2` | `101` / `5` / `2` / `519` | `1.0` | `68` / `38` / `0` / `521` | `1.0` |
| `c7_bridge_peak_m5_d30` | `226` / `13` / `5` / `390` | `0.996` | `169` / `70` / `0` / `395` | `1.0` |
| `c8_kite_chorus_d30` | `17` / `4` / `0` / `262` | `1.0` | `17` / `4` / `0` / `262` | `1.0` |
| `c9_fridge_chorus_d30` | `22` / `0` / `0` / `202` | `1.0` | `21` / `1` / `0` / `202` | `1.0` |

**Not a stem that one separator got wrong.** The frames with no pitch in the gate stem have none in the other separator's stem, bar a handful, and none in the mix before separation. Separation adds frames to what the mix already gives; the mix has a pitch in no frame where the gate stem lacks one. Whether both stems carry accompaniment is section 11's question.

## 4. Whether the unpitched frames still follow the melody

`frames` loosens YIN's periodicity threshold on the output only and scores the frames that gain a pitch, aligned and against the target rotated in time.

| Arm | Threshold | `voiced_overlap_frac` | Gained frames | Gained frames in tune: aligned / null median / null best / share of shifts beaten |
| :--- | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.25` | `0.530` | `139` | `0.468` / `0.125` / `0.351` / `1.00` |
| `c1_verse_m5_d26` | `0.35` | `0.766` | `287` | `0.362` / `0.128` / `0.329` / `1.00` |
| `c1_verse_m5_d26` | `0.50` | `0.903` | `373` | `0.354` / `0.121` / `0.296` / `1.00` |
| `c2_verse_m5_d32` | `0.25` | `0.276` | `117` | `0.094` / `0.042` / `0.186` / `0.73` |
| `c2_verse_m5_d32` | `0.35` | `0.528` | `275` | `0.084` / `0.063` / `0.175` / `0.67` |
| `c2_verse_m5_d32` | `0.50` | `0.829` | `464` | `0.097` / `0.065` / `0.154` / `0.82` |
| `c3_chorus_m5_d30` | `0.25` | `0.534` | `159` | `0.384` / `0.037` / `0.207` / `1.00` |
| `c3_chorus_m5_d30` | `0.35` | `0.671` | `249` | `0.402` / `0.056` / `0.169` / `1.00` |
| `c3_chorus_m5_d30` | `0.50` | `0.776` | `318` | `0.371` / `0.051` / `0.177` / `1.00` |
| `c4_verse_m5_rain_a2v0_d30` | `0.25` | `0.609` | `152` | `0.237` / `0.091` / `0.493` / `0.91` |
| `c4_verse_m5_rain_a2v0_d30` | `0.35` | `0.775` | `256` | `0.215` / `0.102` / `0.418` / `0.97` |
| `c4_verse_m5_rain_a2v0_d30` | `0.50` | `0.911` | `341` | `0.229` / `0.099` / `0.374` / `0.97` |
| `c5_verse_m5_d30_s1` | `0.25` | `0.498` | `178` | `0.264` / `0.095` / `0.495` / `0.94` |
| `c5_verse_m5_d30_s1` | `0.35` | `0.707` | `309` | `0.227` / `0.089` / `0.434` / `0.94` |
| `c5_verse_m5_d30_s1` | `0.50` | `0.884` | `420` | `0.195` / `0.090` / `0.343` / `0.94` |
| `c6_verse_m5_d30_s2` | `0.25` | `0.356` | `117` | `0.444` / `0.090` / `0.383` / `1.00` |
| `c6_verse_m5_d30_s2` | `0.35` | `0.571` | `252` | `0.448` / `0.112` / `0.376` / `1.00` |
| `c6_verse_m5_d30_s2` | `0.50` | `0.847` | `425` | `0.414` / `0.109` / `0.310` / `1.00` |
| `c7_bridge_peak_m5_d30` | `0.25` | `0.651` | `174` | `0.511` / `0.096` / `0.237` / `1.00` |
| `c7_bridge_peak_m5_d30` | `0.35` | `0.792` | `263` | `0.502` / `0.112` / `0.219` / `1.00` |
| `c7_bridge_peak_m5_d30` | `0.50` | `0.874` | `315` | `0.457` / `0.103` / `0.213` / `1.00` |
| `c8_kite_chorus_d30` | `0.25` | `0.276` | `57` | `0.351` / `0.111` / `0.600` / `0.88` |
| `c8_kite_chorus_d30` | `0.35` | `0.565` | `139` | `0.273` / `0.128` / `0.340` / `0.88` |
| `c8_kite_chorus_d30` | `0.50` | `0.820` | `211` | `0.256` / `0.120` / `0.328` / `0.97` |
| `c9_fridge_chorus_d30` | `0.25` | `0.348` | `56` | `0.321` / `0.220` / `0.600` / `0.71` |
| `c9_fridge_chorus_d30` | `0.35` | `0.598` | `112` | `0.295` / `0.182` / `0.486` / `0.84` |
| `c9_fridge_chorus_d30` | `0.50` | `0.866` | `172` | `0.285` / `0.122` / `0.549` / `0.83` |

1. **In `c1`, `c3`, `c6` and `c7` the unpitched frames follow the target melody, less closely than the scored ones.** The frames that gain a pitch beat every shift at each threshold. Their in-tune share is below that of the scored frames (section 1) and below what the check reads for a known right melody under noise (`check_rough_voice_is_credited_and_beats_its_null` prints it).
2. **In `c2` they cannot be tied to the melody.** Its gained frames sit inside their null at every threshold, so nothing in the unscored part of `c2` shows the target's notes.
3. **`c4` and `c5` are between the two.** Their gained frames beat most shifts and not all. `c5` and `c6` are two seeds of one arm and fall on either side of this line.
4. **`c8` and `c9` stay inside their null**, on targets that give the null little to work with.

## 5. Which of the target's frames keep their pitch

Scored share in each quarter of the target's voiced frames, ranked by the target's own `min_cmndf`, level and pitch.

| Arm | By target `min_cmndf` (steadiest first) | By target level (quietest first) | By target pitch (lowest first) |
| :--- | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.459` / `0.439` / `0.229` / `0.103` | `0.204` / `0.197` / `0.401` / `0.429` | `0.057` / `0.280` / `0.382` / `0.513` |
| `c2_verse_m5_d32` | `0.153` / `0.115` / `0.076` / `0.013` | `0.102` / `0.089` / `0.083` / `0.083` | `0.006` / `0.032` / `0.057` / `0.263` |
| `c3_chorus_m5_d30` | `0.497` / `0.335` / `0.213` / `0.122` | `0.085` / `0.268` / `0.372` / `0.445` | `0.279` / `0.268` / `0.463` / `0.159` |
| `c4_verse_m5_rain_a2v0_d30` | `0.510` / `0.401` / `0.344` / `0.212` | `0.261` / `0.376` / `0.471` / `0.359` | `0.376` / `0.274` / `0.382` / `0.436` |
| `c5_verse_m5_d30_s1` | `0.376` / `0.204` / `0.185` / `0.090` | `0.223` / `0.178` / `0.236` / `0.218` | `0.166` / `0.204` / `0.146` / `0.340` |
| `c6_verse_m5_d30_s2` | `0.261` / `0.261` / `0.096` / `0.058` | `0.172` / `0.089` / `0.134` / `0.282` | `0.038` / `0.051` / `0.280` / `0.308` |
| `c7_bridge_peak_m5_d30` | `0.648` / `0.377` / `0.323` / `0.158` | `0.082` / `0.327` / `0.487` / `0.614` | `0.340` / `0.434` / `0.392` / `0.342` |
| `c8_kite_chorus_d30` | `0.070` / `0.028` / `0.113` / `0.086` | `0.085` / `0.141` / `0.070` / `0.000` | `0.155` / `0.056` / `0.056` / `0.029` |
| `c9_fridge_chorus_d30` | `0.089` / `0.071` / `0.071` / `0.161` | `0.054` / `0.071` / `0.125` / `0.143` | `0.089` / `0.179` / `0.089` / `0.036` |

1. **In each of the seven `justaflicker` arms the target's steadiest frames are the ones that survive.** The quarter of frames where the target vocal is most periodic is scored far more often than the quarter where it is least. Every one of these target frames already passes YIN, so this is a ranking among clean frames. `c8` and `c9` show no order, on about twenty scored frames each.
2. **Level and pitch each matter in some arms and not in others.** The second and third columns do not order the same way from arm to arm: level makes no difference in `c2` and `c5`, pitch makes none in `c7`, and in `c3` it is the highest quarter that is lost where in `c1`, `c2` and `c6` it is the lowest. Within a phrase the three rankings move together, and nothing here separates them.

The four verse arms that share one prompt:

| Arm | `denoise` | Seed | Scored of target voiced | Gained at `0.35`: frames | In tune: aligned / null median / null best / share beaten |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `0.26` | `SEED` | `193` of `627` | `287` | `0.362` / `0.128` / `0.329` / `1.00` |
| `c5_verse_m5_d30_s1` | `0.30` | `SEED` | `134` of `627` | `309` | `0.227` / `0.089` / `0.434` / `0.94` |
| `c6_verse_m5_d30_s2` | `0.30` | `SEED + 1000` | `106` of `627` | `252` | `0.448` / `0.112` / `0.376` / `1.00` |
| `c2_verse_m5_d32` | `0.32` | `SEED` | `56` of `627` | `275` | `0.084` / `0.063` / `0.175` / `0.67` |

3. **Scored frames fall as `denoise` rises.** At one seed and one prompt the count drops at each step. The second seed at `0.30` lands below the first and above the `0.32` arm. That is one pair of seeds.

## 6. How periodic the voices are

`periodic`: the share of a file's active frames (within `ACTIVE_BELOW_LOUD_DB` of its loud frames) in which YIN finds a pitch.

| File | Active frames | Share with a pitch | `min_cmndf` quartiles |
| :--- | :---: | :---: | :---: |
| `flicker_voc_m5_verse_243f.wav` | `786` | `0.782` | `0.01` / `0.03` / `0.10` |
| `flicker_voc_m5_chorus_243f.wav` | `853` | `0.728` | `0.03` / `0.07` / `0.17` |
| `flicker_voc_m5_bridge_243f.wav` | `822` | `0.713` | `0.01` / `0.04` / `0.24` |
| `kite_voc_m6_243f.wav` | `925` | `0.293` | `0.13` / `0.27` / `0.53` |
| `fridge_voc_p2_243f.wav` | `830` | `0.265` | `0.14` / `0.29` / `0.56` |
| `kurt_vocal_clean_192f.wav` | `486` | `0.714` | `0.04` / `0.07` / `0.19` |
| `c1_verse_m5_d26_voc.wav` | `793` | `0.253` | `0.14` / `0.27` / `0.46` |
| `c2_verse_m5_d32_voc.wav` | `816` | `0.075` | `0.26` / `0.37` / `0.51` |
| `c3_chorus_m5_d30_voc.wav` | `819` | `0.242` | `0.15` / `0.27` / `0.49` |
| `c4_verse_m5_rain_a2v0_d30_voc.wav` | `819` | `0.293` | `0.13` / `0.25` / `0.44` |
| `c5_verse_m5_d30_s1_voc.wav` | `808` | `0.179` | `0.18` / `0.29` / `0.47` |
| `c6_verse_m5_d30_s2_voc.wav` | `844` | `0.140` | `0.22` / `0.35` / `0.50` |
| `c7_bridge_peak_m5_d30_voc.wav` | `837` | `0.286` | `0.14` / `0.25` / `0.47` |
| `c8_kite_chorus_d30_voc.wav` | `958` | `0.034` | `0.29` / `0.43` / `0.57` |
| `c9_fridge_chorus_d30_voc.wav` | `637` | `0.058` | `0.26` / `0.40` / `0.59` |

The same figure for each source vocal before the gate's pitch shift, cut from the song's `vocals.wav` at the arm's window:

| Source vocal | Active frames | Share with a pitch | `min_cmndf` quartiles |
| :--- | :---: | :---: | :---: |
| `justaflicker`, verse window | `788` | `0.782` | `0.02` / `0.03` / `0.08` |
| `justaflicker`, chorus window | `880` | `0.769` | `0.02` / `0.04` / `0.14` |
| `justaflicker`, bridge window | `837` | `0.717` | `0.01` / `0.02` / `0.26` |
| `thatsnotakite`, `c8` window | `926` | `0.313` | `0.12` / `0.25` / `0.51` |
| `inthefridge`, `c9` window | `839` | `0.243` | `0.15` / `0.28` / `0.57` |

1. **The output vocal stems are far less periodic than the voice they imitate.** The reference recording and the three `justaflicker` targets have a pitch in most of their active frames; the output vocal stems have one in a minority. Section 11 shows the stems carry some accompaniment, and that it is too quiet to account for this. The scaffold keeps nearly every frame through the second separator (earlier record, section 6) and through the audio VAE followed by the gate's separator (`controls.audiovae_roundtrip` in the gate's report), so neither the separation nor the VAE does this.
2. **The `thatsnotakite` and `inthefridge` targets are themselves mostly unpitched.** They read like the renders, not like the `justaflicker` targets, and their source vocals read the same before the pitch shift, so the shift is not the cause. That leaves the scoring of `c8` and `c9` with few target frames to stand on. Whether those vocals are layered, harmonized or processed is a listening question; a pitch tracker that follows one voice will not score them as they are.

## 7. Stem levels

`levels`, in dB against each clip's own mix. The last six columns are the `six` stems.

| Mix | Gate stem | Mix minus gate stem | `six` vocals | guitar | bass | other | piano | drums |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `-0.28` | `-13.58` | `-0.10` | `-23.14` | `-90.95` | `-63.65` | `-101.55` | `-85.95` |
| `c2_verse_m5_d32` | `-0.27` | `-14.59` | `-0.21` | `-18.74` | `-70.93` | `-37.48` | `-28.24` | `-38.37` |
| `c3_chorus_m5_d30` | `-0.74` | `-10.15` | `-0.57` | `-28.80` | `-94.09` | `-16.13` | `-24.19` | `-37.07` |
| `c4_verse_m5_rain_a2v0_d30` | `-0.16` | `-15.98` | `-0.17` | `-19.52` | `-30.70` | `-26.04` | `-100.95` | `-85.56` |
| `c5_verse_m5_d30_s1` | `-0.33` | `-13.98` | `-0.26` | `-19.61` | `-26.73` | `-25.42` | `-95.55` | `-43.52` |
| `c6_verse_m5_d30_s2` | `-0.26` | `-15.07` | `-0.28` | `-22.30` | `-95.19` | `-17.81` | `-99.93` | `-92.19` |
| `c7_bridge_peak_m5_d30` | `-0.53` | `-10.89` | `-0.47` | `-31.34` | `-40.54` | `-18.33` | `-32.29` | `-86.46` |
| `c8_kite_chorus_d30` | `-0.25` | `-14.59` | `-0.00` | `-98.57` | `-97.39` | `-100.07` | `-104.14` | `-91.66` |
| `c9_fridge_chorus_d30` | `-0.44` | `-10.48` | `-0.50` | `-82.55` | `-14.90` | `-12.15` | `-102.27` | `-47.26` |
| scaffold `flicker_m5_verse_cal_243f.wav` | | | `-0.51` | `-20.02` | `-10.48` | `-94.39` | `-99.94` | `-92.03` |
| scaffold `flicker_m5_chorus_cal_243f.wav` | | | `-0.38` | `-23.18` | `-13.17` | `-18.47` | `-97.79` | `-95.49` |
| scaffold `flicker_m5_bridge_cal_243f.wav` | | | `-0.38` | `-20.81` | `-12.40` | `-21.17` | `-97.89` | `-96.33` |
| scaffold `kite_m6_voc_bass_cal_243f.wav` | | | `-0.31` | `-100.93` | `-11.68` | `-100.76` | `-103.99` | `-98.70` |
| scaffold `fridge_p2_voc_bass_cal_243f.wav` | | | `-0.31` | `-100.30` | `-11.68` | `-100.07` | `-104.52` | `-100.07` |

By level only, not by ear:

1. **Both separators put nearly the whole render in the vocal stem.** Each vocal stem is within a decibel of the mix, and taking the gate stem out leaves a residue well under it.
2. **The `six` bass stem is close to empty in five arms and well below its scaffold's in three more.** That is not the bass leaving the render. Section 11 finds low-band sound in the vocal rests of every arm, and `six` files it under vocals; in `c8` all of it.
3. **Over the whole clip the guitar stem of each `justaflicker` render is near its scaffold's level, and it is empty in `c8` and `c9`, whose scaffolds have none.** In the vocal rests it is a different picture (section 11).
4. **`other` holds something the scaffold does not have in several arms, most in `c6` and `c9`.**

These are two separators trained on recorded music reading generated audio. Whether a guitar or a bass can be heard is a listening question.

## 8. Where to listen

The longest spans in each gate stem, in seconds and 24 fps frames of the clip.

| Arm | Unpitched while the target sings | Scored |
| :--- | :--- | :--- |
| `c1_verse_m5_d26` | `0.79`–`1.40 s` (`f18`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `5.64`–`6.06 s` (`f135`–`f145`), `7.97`–`9.05 s` (`f191`–`f217`) | `2.69`–`3.12 s` (`f64`–`f74`), `3.29`–`3.72 s` (`f78`–`f89`), `6.60`–`6.81 s` (`f158`–`f163`), `7.08`–`7.58 s` (`f169`–`f181`) |
| `c2_verse_m5_d32` | `0.27`–`1.40 s` (`f6`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `6.97`–`7.70 s` (`f167`–`f184`), `7.97`–`9.05 s` (`f191`–`f217`) | `1.63`–`1.77 s` (`f39`–`f42`), `2.71`–`3.10 s` (`f65`–`f74`), `3.32`–`3.38 s` (`f79`–`f81`), `6.33`–`6.38 s` (`f151`–`f153`) |
| `c3_chorus_m5_d30` | `4.57`–`5.00 s` (`f109`–`f120`), `5.21`–`5.65 s` (`f125`–`f135`), `7.20`–`7.70 s` (`f172`–`f184`), `9.06`–`10.07 s` (`f217`–`f241`) | `0.87`–`1.18 s` (`f20`–`f28`), `1.24`–`1.48 s` (`f29`–`f35`), `2.46`–`2.85 s` (`f59`–`f68`), `8.63`–`9.06 s` (`f207`–`f217`) |
| `c4_verse_m5_rain_a2v0_d30` | `0.47`–`1.40 s` (`f11`–`f33`), `3.85`–`4.42 s` (`f92`–`f106`), `5.64`–`6.06 s` (`f135`–`f145`), `8.65`–`9.05 s` (`f207`–`f217`) | `2.66`–`3.11 s` (`f63`–`f74`), `3.29`–`3.64 s` (`f78`–`f87`), `6.59`–`6.82 s` (`f158`–`f163`), `8.32`–`8.68 s` (`f199`–`f208`) |
| `c5_verse_m5_d30_s1` | `0.27`–`1.40 s` (`f6`–`f33`), `3.29`–`3.72 s` (`f78`–`f89`), `3.85`–`4.42 s` (`f92`–`f106`), `7.97`–`8.72 s` (`f191`–`f209`) | `1.62`–`1.78 s` (`f38`–`f42`), `2.71`–`3.11 s` (`f65`–`f74`), `6.59`–`6.81 s` (`f158`–`f163`), `7.40`–`7.58 s` (`f177`–`f181`) |
| `c6_verse_m5_d30_s2` | `0.27`–`1.40 s` (`f6`–`f33`), `6.27`–`6.84 s` (`f150`–`f164`), `6.97`–`7.70 s` (`f167`–`f184`), `7.97`–`9.05 s` (`f191`–`f217`) | `1.62`–`1.89 s` (`f38`–`f45`), `2.87`–`3.12 s` (`f68`–`f74`), `3.30`–`3.72 s` (`f79`–`f89`), `6.79`–`6.83 s` (`f162`–`f163`) |
| `c7_bridge_peak_m5_d30` | `0.00`–`0.60 s` (`f0`–`f14`), `2.45`–`2.80 s` (`f58`–`f67`), `4.76`–`5.28 s` (`f114`–`f126`), `7.69`–`8.20 s` (`f184`–`f196`) | `1.42`–`1.90 s` (`f34`–`f45`), `5.60`–`5.92 s` (`f134`–`f142`), `6.77`–`7.19 s` (`f162`–`f172`), `9.39`–`9.67 s` (`f225`–`f232`) |
| `c8_kite_chorus_d30` | `0.06`–`0.48 s` (`f1`–`f11`), `2.71`–`3.30 s` (`f65`–`f79`), `5.49`–`5.74 s` (`f131`–`f137`), `7.60`–`8.20 s` (`f182`–`f196`) | `0.46`–`0.55 s` (`f11`–`f13`), `2.56`–`2.71 s` (`f61`–`f65`), `5.60`–`5.65 s` (`f134`–`f135`), `5.87`–`5.90 s` (`f140`–`f141`) |
| `c9_fridge_chorus_d30` | `2.19`–`2.63 s` (`f52`–`f63`), `6.22`–`6.35 s` (`f149`–`f152`), `7.06`–`7.63 s` (`f169`–`f183`), `8.39`–`8.51 s` (`f201`–`f204`) | `1.16`–`1.33 s` (`f27`–`f31`), `2.14`–`2.19 s` (`f51`–`f52`), `2.58`–`2.59 s` (`f61`–`f62`), `7.55`–`7.57 s` (`f181`–`f181`) |

The verse arms lose the same passages, which is section 5 seen in time. What to listen for in an unpitched span against a scored one in the same clip: rasp or breath on a held note, two voices at once, an instrument inside the voice, or a note that wanders.

## 9. The gate's timbre figure against controls

`ltas`: the cosine of two long-term mel spectra, here each file against `kurt_vocal_clean_192f.wav`, computed as the gate computes `ltas_cos_kurt_unplugged`. The "Render mix" column equals the gate's report in all nine arms. The reference's first half against its second reads `0.804`.

| Arm | Its scaffold | Render mix (the gate's figure) | Its target vocal | Gate stem | Gate stem minus target vocal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `c1_verse_m5_d26` | `-0.023` | `+0.308` | `+0.636` | `+0.608` | `-0.028` |
| `c2_verse_m5_d32` | `-0.023` | `+0.503` | `+0.636` | `+0.755` | `+0.119` |
| `c3_chorus_m5_d30` | `+0.062` | `+0.423` | `+0.841` | `+0.788` | `-0.053` |
| `c4_verse_m5_rain_a2v0_d30` | `-0.023` | `+0.538` | `+0.636` | `+0.765` | `+0.129` |
| `c5_verse_m5_d30_s1` | `-0.023` | `+0.415` | `+0.636` | `+0.685` | `+0.049` |
| `c6_verse_m5_d30_s2` | `-0.023` | `+0.357` | `+0.636` | `+0.598` | `-0.038` |
| `c7_bridge_peak_m5_d30` | `-0.087` | `+0.314` | `+0.890` | `+0.832` | `-0.058` |
| `c8_kite_chorus_d30` | `+0.233` | `+0.533` | `+0.780` | `+0.739` | `-0.041` |
| `c9_fridge_chorus_d30` | `+0.247` | `+0.585` | `+0.787` | `+0.858` | `+0.071` |

1. **The figure follows what is in the clip more than whose voice it is.** Every scaffold reads far below its own target vocal, with the same singer in both: the difference is the bass and bed under the voice. `check_ltas_moves_with_a_bass_under_the_same_voice` holds the same on a synthetic voice.
2. **The rise from scaffold to render cannot be read as a change of singer.** Section 11 shows the low band is weaker in every render than in its scaffold, and that alone moves this figure the same way. Every render mix still reads below its unconverted target vocal.
3. **Like for like, about half the stems read no closer to the reference than the unconverted target vocal does.** The last column is negative in `c1`, `c3`, `c6`, `c7` and `c8`, and positive in `c2`, `c4`, `c5` and `c9`.
4. **The scale does not separate singers.** The chorus and bridge target vocals, sung by the original singer, read higher against the reference than the reference's two halves read against each other.

This figure does not measure whether the singer changed. That stays a listening question, the one the earlier handoff asked of `c1`.

## 10. Where the song's `denoise` sits among the calibrated arms

The audio pass of every graph is a `BasicScheduler` with `steps` `12` (node `160` in `data/pure_h3_cover_song/graphs/`), and core turns `denoise` into a schedule by `total_steps = int(steps / denoise)`, keeping the last `steps + 1` sigmas (`comfy_extras/nodes_custom_sampler.py`, `BasicScheduler`).

| `denoise` | `total_steps` | Used by |
| :---: | :---: | :--- |
| `0.26` | `46` | `c1` |
| `0.29` | `41` | the 4-window song, `song_4win_d29.json` |
| `0.30` | `40` | `c3` to `c9` |
| `0.32` | `37` | `c2` |

**The song's setting is one schedule step from the `0.30` arms and five from `c1`.** Between `c1` and `c5` the scored count falls by about ten frames for each schedule step, and the two seeds at `0.30` differ by more than that (section 5). By these arms the song's verse windows should land like the `0.30` pair, where one seed held the melody and the other partly lost it, and not like `c1`. No arm was rendered at `0.29`.

## 11. Whether accompaniment left in the vocal stem explains the unpitched frames

The other session, listening, hears chords under the voice in `c3` and a bass riff in the rests of `c9`, and holds that these separators leave such sound in the vocal stem. If they do, a frame could be unpitched because two sources sound at once. Three measurements.

**What the vocal rests hold.** `rests`: the frames where the target vocal is more than `REST_BELOW_LOUD_DB` under its loud frames, so the scaffold has no voice there. Columns: the clip's level in the rests against its level while the vocal sings; how the rests' sound is spread over three bands; and each stem's level against the clip, within the rests.

| Clip | Rest frames | Rests against sung, dB | Rests' sound under `150 Hz` / `150`–`1000 Hz` / above, dB | In the rests, against the clip: gate stem | `six` vocals | `six` bass | `six` guitar | `six` other |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| scaffold `flicker_m5_verse_cal_243f.wav` | `141` | `-9.8` | `-0.5` / `-9.5` / `-26.0` |  | `-19.1` | `-0.6` | `-11.3` | `-99.5` |
| scaffold `flicker_m5_chorus_cal_243f.wav` | `27` | `-8.3` | `-0.8` / `-7.9` / `-17.8` |  | `-18.3` | `-1.0` | `-16.2` | `-9.9` |
| scaffold `flicker_m5_bridge_cal_243f.wav` | `18` | `-10.1` | `-1.0` / `-7.0` / `-21.7` |  | `-18.1` | `-1.2` | `-13.6` | `-9.2` |
| scaffold `kite_m6_voc_bass_cal_243f.wav` | `35` | `-7.5` | `-0.1` / `-17.1` / `-28.0` |  | `-24.0` | `-0.0` | `-106.5` | `-104.4` |
| scaffold `fridge_p2_voc_bass_cal_243f.wav` | `30` | `-10.8` | `-0.3` / `-13.3` / `-19.4` |  | `-18.2` | `-0.1` | `-102.9` | `-103.1` |
| `c1_verse_m5_d26` | `140` | `-10.3` | `-3.1` / `-3.2` / `-14.5` | `-14.1` | `-3.2` | `-76.8` | `-8.0` | `-48.7` |
| `c2_verse_m5_d32` | `140` | `-9.6` | `-7.0` / `-1.7` / `-9.4` | `-11.2` | `-9.2` | `-55.2` | `-3.4` | `-21.5` |
| `c3_chorus_m5_d30` | `27` | `-6.8` | `-7.3` / `-1.7` / `-8.6` | `-20.3` | `-8.3` | `-87.6` | `-27.5` | `-8.2` |
| `c4_verse_m5_rain_a2v0_d30` | `140` | `-10.7` | `-6.1` / `-1.8` / `-10.4` | `-11.5` | `-10.7` | `-16.0` | `-3.3` | `-9.6` |
| `c5_verse_m5_d30_s1` | `140` | `-9.9` | `-5.6` / `-1.9` / `-10.7` | `-10.6` | `-9.0` | `-11.5` | `-5.5` | `-11.3` |
| `c6_verse_m5_d30_s2` | `140` | `-11.5` | `-4.3` / `-2.8` / `-10.0` | `-8.8` | `-8.3` | `-82.6` | `-7.8` | `-4.4` |
| `c7_bridge_peak_m5_d30` | `18` | `-6.9` | `-2.3` / `-4.7` / `-11.6` | `-12.3` | `-11.8` | `-27.6` | `-24.0` | `-11.1` |
| `c8_kite_chorus_d30` | `35` | `-9.2` | `-0.7` / `-8.9` / `-19.3` | `-8.4` | `+0.0` | `-85.5` | `-92.0` | `-92.6` |
| `c9_fridge_chorus_d30` | `30` | `-8.5` | `-2.3` / `-4.2` / `-14.9` | `-20.2` | `-8.9` | `-2.0` | `-92.3` | `-16.1` |

1. **Every render sounds through the vocal rests, at about the level its scaffold does.**
2. **The scaffolds' rests are bass; the renders' rests are not only bass.** In the scaffolds nearly all of the rests' sound is under `150 Hz` and `six` files it under bass. In the `justaflicker` renders much of it sits between `150` and `1000 Hz`, and `six` files a good part of that under guitar or other. So where the voice rests, these renders hold more mid-range sound than their scaffold did, most in the verse arms at `0.30` and `0.32`.
3. **Both separators leave some of it in the vocal stem.** The gate stem carries the rests' sound at the levels in its column, least in `c3` and `c9`, most in `c6` and `c8`. `six` calls all of `c8`'s rests vocal, though they are almost entirely under `150 Hz`.

The share of each clip's sound under `150 Hz`, over all frames, in dB against the clip:

| Arm | Its scaffold | Render |
| :--- | :---: | :---: |
| `c1_verse_m5_d26` | `-8.8` | `-14.7` |
| `c2_verse_m5_d32` | `-8.8` | `-19.4` |
| `c3_chorus_m5_d30` | `-11.5` | `-16.5` |
| `c4_verse_m5_rain_a2v0_d30` | `-8.8` | `-19.6` |
| `c5_verse_m5_d30_s1` | `-8.8` | `-17.3` |
| `c6_verse_m5_d30_s2` | `-8.8` | `-15.8` |
| `c7_bridge_peak_m5_d30` | `-10.9` | `-13.3` |
| `c8_kite_chorus_d30` | `-9.5` | `-17.6` |
| `c9_fridge_chorus_d30` | `-10.6` | `-14.1` |

4. **The low band is weaker in every render than in its scaffold, and not gone.**

**Whether the low band hides the pitch.** `frames --highpass 150`, which filters the output before scoring. The unseparated scaffolds are the control: voice, bass and bed together.

| Clip scored against its target | Scored frames as is | High-passed at `150 Hz` | Target voiced |
| :--- | :---: | :---: | :---: |
| scaffold `flicker_m5_verse_cal_243f.wav`, unseparated | `414` | `535` | `627` |
| scaffold `flicker_m5_chorus_cal_243f.wav`, unseparated | `342` | `442` | `659` |
| scaffold `flicker_m5_bridge_cal_243f.wav`, unseparated | `450` | `569` | `634` |
| scaffold `kite_m6_voc_bass_cal_243f.wav`, unseparated | `142` | `266` | `283` |
| scaffold `fridge_p2_voc_bass_cal_243f.wav`, unseparated | `154` | `201` | `226` |
| `c1_verse_m5_d26`, gate stem | `193` | `164` | `627` |
| `c2_verse_m5_d32`, gate stem | `56` | `56` | `627` |
| `c3_chorus_m5_d30`, gate stem | `192` | `142` | `657` |
| `c4_verse_m5_rain_a2v0_d30`, gate stem | `230` | `185` | `627` |
| `c5_verse_m5_d30_s1`, gate stem | `134` | `108` | `627` |
| `c6_verse_m5_d30_s2`, gate stem | `106` | `91` | `627` |
| `c7_bridge_peak_m5_d30`, gate stem | `239` | `213` | `634` |
| `c8_kite_chorus_d30`, gate stem | `21` | `14` | `283` |
| `c9_fridge_chorus_d30`, gate stem | `22` | `23` | `224` |

5. **Taking the low band out restores the scaffolds and does nothing for the renders.** With its bass in, an unseparated scaffold loses a good part of its frames, and the filter brings many of them back (`check_rests_and_highpass_find_a_bass_left_in_the_stem` holds the same on a synthetic voice). No render stem gains more than a frame; most lose some, as a clean vocal does under the same filter. The pitch is lost above `150 Hz`.

**How loud a bed must be.** `under`: the `justaflicker` verse vocal before the pitch shift, with the same song's `harmonic.wav` mixed under it.

| Bed level against the voice | `-20 dB` | `-15 dB` | `-10 dB` | `-5 dB` | `+0 dB` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Share of the voice's pitched frames that keep a pitch | `0.978` | `0.946` | `0.850` | `0.625` | `0.473` |

6. **Even a bed as loud as the voice costs less than the renders lose, and one well under the voice costs little.** In the rests, the gate stem carries the leftover sound far below that, once its column in the first table is added to the rests' own level against the sung parts.

**Reading.** The renders carry accompaniment, with more mid-range sound between phrases than their scaffolds had, and neither separator removes all of it from the vocal stem. That does not explain the unpitched frames: the low band is not what hides the pitch, and the mid-range sound the stems carry between phrases is too quiet to. What this cannot rule out is an accompaniment much louder under the voice than between phrases.

## 12. Commands

```bash
V="$H3_OUTPUT_DIR/Video/pure_h3_cover_song_2026-11-27"; S="$V/vocal_stems"; I="$H3_INPUT_DIR/music_cover_stems"
X=<scratch folder>   # sepaudio stems "$V/<arm>_00001-audio.mp4" -r six -o "$X", run in the sepaudio checkout's own environment
"$PY" bench/pitch_lock_metrics.py "$S/c1_verse_m5_d26_voc.wav" "$I/flicker_voc_m5_verse_243f.wav"      # section 1
"$PY" bench/pitch_lock_coverage.py frames "$S/c1_verse_m5_d26_voc.wav" "$I/flicker_voc_m5_verse_243f.wav"  # sections 2, 4, 5, 8
"$PY" bench/pitch_lock_coverage.py agree "$S/c1_verse_m5_d26_voc.wav" "$X/c1_verse_m5_d26_00001-audio/vocals.wav" "$I/flicker_voc_m5_verse_243f.wav"  # section 3
"$PY" bench/pitch_lock_coverage.py agree "$S/c1_verse_m5_d26_voc.wav" "$V/c1_verse_m5_d26_00001-audio.mp4" "$I/flicker_voc_m5_verse_243f.wav"  # section 3, the mix
"$PY" bench/pitch_lock_coverage.py periodic "$I/kurt_vocal_clean_192f.wav" "$I/kite_voc_m6_243f.wav" "$S/c1_verse_m5_d26_voc.wav"  # section 6
ffmpeg -ss 63.625 -t 10.125 -i "$I/thatsnotakite/vocals.wav" "$X/kite_chorus_src_voc.wav"   # section 6, a source window; inthefridge starts at 60.500
"$PY" bench/pitch_lock_coverage.py levels "$V/c1_verse_m5_d26_00001-audio.mp4" "$S/c1_verse_m5_d26_voc.wav" "$X/c1_verse_m5_d26_00001-audio/"{vocals,guitar,bass,other,piano,drums}.wav  # section 7
"$PY" bench/pitch_lock_coverage.py ltas "$I/kurt_vocal_clean_192f.wav" "$I/flicker_m5_verse_cal_243f.wav" "$I/flicker_voc_m5_verse_243f.wav" "$V/c1_verse_m5_d26_00001-audio.mp4" "$S/c1_verse_m5_d26_voc.wav"  # section 9
"$PY" bench/pitch_lock_coverage.py rests "$I/flicker_voc_m5_verse_243f.wav" "$V/c1_verse_m5_d26_00001-audio.mp4" "$S/c1_verse_m5_d26_voc.wav" "$X/c1_verse_m5_d26_00001-audio/"{vocals,bass,guitar,other}.wav  # section 11
"$PY" bench/pitch_lock_coverage.py frames --highpass 150 "$S/c1_verse_m5_d26_voc.wav" "$I/flicker_voc_m5_verse_243f.wav"  # section 11
"$PY" bench/pitch_lock_coverage.py under "$X/flicker_verse_src_voc.wav" "$X/flicker_verse_src_bed.wav"  # section 11; both cut at 10.500 as below, from vocals.wav and harmonic.wav
```

The `justaflicker` source windows start at `10.500`, `42.200` and `103.000` for the verse, chorus and bridge.

## 13. Limits

- No one has listened. "Unpitched" is YIN's judgment at `probe_yin_f0_drift.DEFAULT_THRESHOLD`. It does not tell rasp from breath or from two voices at once. Section 11 bounds how much an instrument left in the stem can account for, by its level between phrases; it cannot see the level under the voice.
- A better separator might recover frames that neither of these two does. Section 3 shows the two agree, not that they are right.
- The in-tune share of gained frames is a floor. A looser threshold takes harmonics for the pitch more often, and the check's control is a tone under white noise, which a rough voice is not.
- One render per arm, apart from the one pair of seeds.
- The quarters of section 5 describe where pitch survives. They do not say why, and the three rankings are not independent.
- The time-shift null has the limits listed in the earlier record, section 5.
- Section 9 shows what the timbre figure cannot say. It offers no measure of the singer in its place.

## 14. Not covered

- The 4-window song.
- The other two songs, beyond showing that their targets cannot carry this scoring.
- The choice of transposition by median error (earlier record, section 1, item 3). Every `justaflicker` arm here is scored at no transposition, so it did not arise.
