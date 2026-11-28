# Pitch-Lock Scoring Rescored Against Time-Shifted Targets (`2026-11-27`)

- **Verdict:** No arm of the 6-arm pitch-lock gate beats its own time-shifted null on any column, so none has shown pitch lock. Which arm looks best depends on the column read. `vocal_f0_r` does not measure contour agreement and is withdrawn as evidence.
- **Date:** `2026-11-27`
- **Corrects:** [`2026-11-27_music_cover_pitch_lock_gate.md`](2026-11-27_music_cover_pitch_lock_gate.md), the pitch columns of section 2 and the "interval lock" and "contour" readings built on them. Its attribution, guitar-synthesis and output-register findings are not touched.
- **Scripts:** [`bench/pitch_lock_metrics.py`](../pitch_lock_metrics.py), [`bench/check_pitch_lock_metrics.py`](../check_pitch_lock_metrics.py)
- **Inputs:** the gate's own files, no new renders: `$H3_OUTPUT_DIR/Video/music_cover_pitch_lock_2026-11-27/vocal_stems/<arm>_voc.wav` against each arm's `src_voc` under `$H3_INPUT_DIR/music_cover_pitch_lock/` (`ARM_SPECS` in [`bench/music_cover_pitch_lock_gate.py`](../music_cover_pitch_lock_gate.py)).
- **Conditions:** scored on CPU on a second machine, reading the same files over the shared mount, with `torch`, `numpy` and `ffmpeg` only. The legacy columns below reproduce the gate's table to the printed digit, so the files and the scoring are the ones the gate used.

---

## 1. What the gate's table could not say

1. **No column had a chance level.** A cent error or an in-tune share was read by its size. Nothing recorded what the same output scores against a target it is not following.
2. **`vocal_f0_r` builds the target into the output.** It moves every output frame to the octave nearest its target frame and then correlates the two. After that step the moved track is the target plus a bounded error, so it correlates with the target whatever was sung. `check_pitch_lock_metrics.py::check_unrelated_pair_sits_inside_its_null` holds this on two unrelated synthetic melodies.
3. **The transposition is chosen by median error.** `best_tr_st` is the whole-semitone shift with the lowest median error, and `rel_in_tune_frac_50c` is read at that shift. Where the errors fall in two groups, that shift can leave the position where most frames were in tune.

## 2. Method

`pitch_lock_metrics.py::lag_null` scores the output against its own target rotated in time, every `LAG_STEP_S` from `LAG_EDGE_S` after the start to `LAG_EDGE_S` before the end. Rotation keeps the target's key, range and amount of singing and removes only the alignment. Each shift gets the same transposition search as the aligned score. For each column it reports the null's median, the null's best value, and the share of shifts the aligned score is better than.

`vocal_f0_r_one_shift` is the same correlation with one octave count for the whole clip.

```bash
S="$H3_OUTPUT_DIR/Video/music_cover_pitch_lock_2026-11-27/vocal_stems"; I="$H3_INPUT_DIR/music_cover_pitch_lock"
"$PY" bench/pitch_lock_metrics.py \
  "$S/p1_flicker_chord_d36_voc.wav" "$I/flicker_voc_orig_192f.wav" \
  "$S/p2_flicker_editguider_d40_voc.wav" "$I/flicker_voc_orig_192f.wav" \
  "$S/p3_flicker_voconly_d40_voc.wav" "$I/flicker_voc_orig_192f.wav" \
  "$S/p4_whatever_dedroned_origkey_d38_voc.wav" "$I/whatever_voc_orig_192f.wav" \
  "$S/p5_whatever_capo_csharp_d38_voc.wav" "$I/whatever_voc_capo_csharp_192f.wav" \
  "$S/p6_fridge_gfunk_unplugged_d40_voc.wav" "$I/fridge_voc_orig_192f.wav"
```

## 3. Results

Every cell is aligned / null median / null best / share of shifts beaten. Cent columns are better when lower, the others when higher.

No transposition:

| Arm | `med_abs_cent_err` | `in_tune_frac_50c` |
| :--- | :---: | :---: |
| `p1_flicker_chord_d36` | `455.7` / `412.9` / `112.7` / `0.25` | `0.111` / `0.050` / `0.244` / `0.67` |
| `p2_flicker_editguider_d40` | `457.6` / `412.8` / `211.9` / `0.17` | `0.028` / `0.058` / `0.267` / `0.42` |
| `p3_flicker_voconly_d40` | `250.2` / `416.3` / `69.7` / `0.75` | `0.307` / `0.000` / `0.439` / `0.88` |
| `p4_whatever_dedroned_origkey_d38` | `382.0` / `396.4` / `291.5` / `0.62` | `0.012` / `0.012` / `0.136` / `0.50` |
| `p5_whatever_capo_csharp_d38` | `128.2` / `168.1` / `55.2` / `0.83` | `0.323` / `0.127` / `0.481` / `0.88` |
| `p6_fridge_gfunk_unplugged_d40` | `300.7` / `395.0` / `15.6` / `0.59` | `0.036` / `0.025` / `0.625` / `0.53` |

Best whole-semitone transposition, chosen separately for the aligned pair and for each shift:

| Arm | `rel_med_abs_cent_err` | `rel_in_tune_frac_50c` |
| :--- | :---: | :---: |
| `p1_flicker_chord_d36` | `71.4` / `92.9` / `42.5` / `0.88` | `0.393` / `0.270` / `0.562` / `0.83` |
| `p2_flicker_editguider_d40` | `68.6` / `111.5` / `39.4` / `0.96` | `0.404` / `0.147` / `0.556` / `0.79` |
| `p3_flicker_voconly_d40` | `149.8` / `96.0` / `32.6` / `0.08` | `0.088` / `0.300` / `0.607` / `0.12` |
| `p4_whatever_dedroned_origkey_d38` | `167.7` / `147.7` / `96.3` / `0.33` | `0.102` / `0.113` / `0.278` / `0.50` |
| `p5_whatever_capo_csharp_d38` | `121.5` / `114.9` / `38.4` / `0.42` | `0.086` / `0.152` / `0.551` / `0.42` |
| `p6_fridge_gfunk_unplugged_d40` | `142.2` / `80.5` / `15.6` / `0.18` | `0.107` / `0.429` / `0.737` / `0.18` |

Contour correlation, one octave count for the clip and the legacy per-frame version:

| Arm | `vocal_f0_r_one_shift` | `vocal_f0_r` |
| :--- | :---: | :---: |
| `p1_flicker_chord_d36` | `0.200` / `0.003` / `0.495` / `0.79` | `0.495` / `0.789` / `0.919` / `0.08` |
| `p2_flicker_editguider_d40` | `-0.138` / `-0.055` / `0.525` / `0.42` | `0.606` / `0.786` / `0.914` / `0.00` |
| `p3_flicker_voconly_d40` | `0.352` / `-0.013` / `0.565` / `0.88` | `0.788` / `0.821` / `0.973` / `0.38` |
| `p4_whatever_dedroned_origkey_d38` | `-0.110` / `-0.023` / `0.506` / `0.46` | `0.671` / `0.709` / `0.842` / `0.33` |
| `p5_whatever_capo_csharp_d38` | `0.109` / `-0.072` / `0.174` / `0.92` | `0.139` / `0.547` / `0.833` / `0.21` |
| `p6_fridge_gfunk_unplugged_d40` | `0.122` / `-0.323` / `0.769` / `0.65` | `0.582` / `0.807` / `0.985` / `0.23` |

What each score rests on:

| Arm | Time shifts | Frames voiced in both | `best_tr_st` | `med_signed_cent_bias` |
| :--- | :---: | :---: | :---: | :---: |
| `p1_flicker_chord_d36` | `24` | `135` | `+5` | `+444.7` |
| `p2_flicker_editguider_d40` | `24` | `141` | `+5` | `+434.4` |
| `p3_flicker_voconly_d40` | `24` | `114` | `+4` | `+250.2` |
| `p4_whatever_dedroned_origkey_d38` | `24` | `255` | `+5` | `+200.0` |
| `p5_whatever_capo_csharp_d38` | `24` | `93` | `+1` | `-21.5` |
| `p6_fridge_gfunk_unplugged_d40` | `17` | `28` | `+3` | `+165.1` |

## 4. Reading

1. **No arm beats every shift on any column.** In every arm, on every column, at least one misaligned position of the target scores better than the aligned one. A copy of a melody, and a copy three semitones up, beat all of theirs (`check_identical_and_transposed_pairs_beat_every_shift`).
2. **`p1` and `p2` come closest only after transposition.** Their aligned error at `+5 st` is better than most shifts. With no transposition their median error is worse than their null's median. If that is lock, it is to a re-keyed melody.
3. **`p3` and `p5` come closest with no transposition, and the gate's table ranked them last.** About a third of their frames are in tune in the target key, more than at most shifts. The transposition search then moves both arms off that position, which is why their `rel_in_tune_frac_50c` reads low.
4. **`vocal_f0_r` is below its null's median in all six arms.** `p3`'s figure, read in the gate record as the highest contour correlation, is lower than the same output scores against its target at most misaligned positions.
5. **`p4` is at its null on every column. `p6` rests on too few frames to read.**

## 5. Limits

- One render per arm and at most `24` shifts, so a share moves in steps of one shift and nothing here says how much two seeds of one arm differ.
- The frames compared are a minority of each clip: only frames where YIN finds a pitch in both the output stem and the target.
- A melody that repeats inside the clip scores well at the shift of its own period. That raises the null and can hide real lock; it cannot create one.
- The transposition search is whole semitones. An output that follows the target a fraction of a semitone off gets no credit.
- The rotated target is the null for "follows this melody in time". It does not test whether the output merely stays in the target's key.

## 6. Not done

- `music_cover_pitch_lock_gate.py::_vocal_pitch_metrics` still computes its own columns. `pitch_lock_metrics.py::vocal_pitch_metrics` takes the same two arguments and returns the same keys plus `vocal_f0_r_one_shift` and `lag_null`, so it can replace the body. Not swapped here because the gate could not be run on the machine this was written on.
- The gate record's section 2 table and the laws drawn from it are left as written, with a dated pointer to this record.
