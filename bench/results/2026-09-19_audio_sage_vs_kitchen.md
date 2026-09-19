# Audio: sage chain vs kitchen chain, MiniMax H3

Date: 2026-09-19. Model: MiniMax H3, `minimax_h3_fl2va_pruned_int8_convrot`,
345 frames at 24 fps, 1344x768, audio 32 kHz stereo out of
`minimax_h3_audio_vae_fp32`. Every number in this file is also in
`bench/results/2026-09-19_audio_sage_vs_kitchen.json`, which carries the full
per-clip rows, per-pair deltas and the per-second per-band grids. CPU only; no
GPU work, no server contact.

Question: on 2026-09-18 the owner heard one same-prompt same-seed pair and said
the sage chain's audio was "just more natural", and could not otherwise tell the
chains apart. This is the measurement.

## Headline

Nothing clears the floor.

Across the two clean kitchen-vs-sage pairs, no audio metric moves in a
consistent direction, and every chain difference sits inside the spread produced
by flipping a knob that has nothing to do with the chain (Sol's token reorder) on
the same chain. The chain differences are *real* -- renders here are
bit-deterministic, so the numbers are not noise -- but they are scene-specific,
they change sign between the two scenes, and they are the same size as or smaller
than an ordinary reorder. On this evidence the data agrees with the owner's own
report that the chains are hard to tell apart.

Two things the reader should take away before the tables:

1. **The brief's kitchen-scene pair is not a chain comparison.** See below.
2. **There is no noise floor to clear.** Two executions of the same graph give
   bit-identical audio. That makes the comparison sharp, and it also means the
   usual "is it above the noise" question has to be replaced with "is it bigger
   than a change we already treat as irrelevant".

## Correction to the brief's chain labels

The brief named `Video/block49_kitchen/default_s730451892_00001-audio.mp4` as
the kitchen-chain arm of the kitchen scene. Its embedded workflow says otherwise:

| clip | dense attention | Sol qk_balance |
|---|---|---|
| `Video/block49_kitchen/default_s730451892_00001-audio.mp4` | `MiniMaxH3SageAttention` mode `auto` | false |
| `Video/sage_chain_panel/kitchen_sage_chain_plain_kitchen_sage_chain_plain_00001-audio.mp4` | `MiniMaxH3SageAttention` mode `fp8++ balanced` | true |

Both arms are sage. Neither carries `ModelAttentionBackend`. That pair is a
sage-mode plus `qk_balance` contrast and it is reported here under its own
label, not as a chain result.

No clean kitchen-vs-sage pair exists for the kitchen scene at all. Prompt sha
`12578a7626` has exactly one kitchen-chain clip in the output tree,
`Video/reorder_panel/kitchen_3d_s730451892_kitchen_3d_00001-audio.mp4`, and that
one also has Sol morton on, so it cannot isolate the chain.

Which pair the owner actually heard on 2026-09-18 is not determined here. Every
candidate already existed by then and file times cannot answer it. Worth asking
the owner: if the impression came from the kitchen scene, it was formed on a
sage-vs-sage comparison.

Chain definitions, from `workflows/h3_config.py`:

- kitchen = `ModelAttentionBackend attention="comfy kitchen attention"` (ComfyUI
  core's node; kitchen's rotated INT8 dense attention), Sol chained on top.
- sage = `MiniMaxH3SageAttention mode="fp8++ balanced"`, Sol chained on top.

## The clips

Clean chain pairs. A full-graph diff of the embedded workflows shows only the
backend node itself and Sol's rewired `model` input; prompt, seed, resolution,
length, Sol settings and every other node are identical.

| scene | kitchen arm | sage arm |
|---|---|---|
| meerkat | `Video/reported_clips/meerkat_default_plain_meerkat_default_plain_00001-audio.mp4` | `Video/reported_clips/meerkat_sage_chain_clean_meerkat_sage_chain_clean_00001-audio.mp4` |
| diner | `Video/reorder_panel/diner_plain_k0235_s730451892_diner_plain_k0235_00001-audio.mp4` | `Video/sage_chain_panel/diner_sage_chain_plain_diner_sage_chain_plain_00001-audio.mp4` |

The meerkat pair was not in the brief. It is the cleanest pair in the tree and
is used as the primary.

The brief's diner kitchen arm,
`Video/block49_repro/diner_default_s730451892_00001-audio.mp4`, turns out to be
**bit-identical** to `diner_plain_k0235` in both audio and video, despite a
different Sol node build, a `SageChainAssert` node and a different `verbose`
flag. So the two diner rows are the same measurement and the brief's arm carries
no build confound after all.

Relabelled, kept because the brief named it:
`Video/block49_kitchen/default_s730451892_00001-audio.mp4` vs
`Video/sage_chain_panel/kitchen_sage_chain_plain_kitchen_sage_chain_plain_00001-audio.mp4`.

Confounded corroboration (sage plus the `MiniMaxH3ChannelBalance` weights
rebalance, against kitchen), in `Video/block49_repro/`: diner, market seed
20260915, hardware_aisle_short. The noodle_bar and post_office pairs there were
excluded as instructed; both were rendered well past their prompt's declared
length.

## The floor, and how it was built

Three reference perturbations, weakest to strongest.

**Measurement chain.** Decode plus an AAC round trip with no filter: integrated
loudness moves 0.10 LU, spectral centroid 1.66 Hz, and the largest cell of the
per-second per-band grid moves 0.17 dB. That is the resolution of the
instrument, not of the model.

**Render determinism: exactly zero.** Four pairs whose workflow graphs are
identical apart from the output filename, plus one pair
(`diner_plain_k0235` against `block49_repro/diner_default`) whose graphs differ
by `SageChainAssert`, Sol's `verbose` flag and the Sol node build. All of them
come back at exactly 0.000 on every metric, on every cell of the band grid, with
waveform correlation 1.0000. Verified independently: the sha256 of the decoded
audio stream (`ffmpeg -map 0:a:0 -f f32le`) matches, and so does the decoded
video stream.

So H3 renders on this box are deterministic for a fixed graph and seed, and the
listed graph differences are inert. There is no run-to-run noise. Every non-zero
number below is caused by the knob that changed.

**Token order (the reference knob).** Same chain, Sol's morton reorder flipped
on and off, seven scenes. This is the closest thing available to "two takes that
differ by something irrelevant" that actually produces a non-zero difference.
Five of the seven stay aligned with the original take (20 ms envelope
correlation at or above 0.90) and are the comparand used in the table. Two
(`cafe_kids`, `cafe_kids_v2`) diverged into different performances, correlation
0.04 and 0.11; they bound what a re-sample does, not what a filter does, and are
reported separately.

| pair | envelope r | band grid mean abs dB | band grid max abs dB |
|---|---|---|---|
| chain/meerkat | 0.921 | 0.722 | 6.109 |
| chain/diner | 0.986 | 1.001 | 4.577 |
| floor meerkat morton | 0.935 | 1.309 | 5.296 |
| floor diner morton | 0.971 | 1.249 | 7.715 |
| floor market morton | 0.979 | 1.992 | 8.351 |
| floor hardware_aisle_short morton | 0.985 | 1.191 | 5.379 |
| floor crowd_churn_long morton | 0.933 | 2.976 | 4.831 |
| floor cafe_kids morton (diverged) | 0.037 | 6.429 | 29.762 |
| floor cafe_kids_v2 morton (diverged) | 0.114 | 8.156 | 30.785 |
| relabelled kitchen scene (diverged) | 0.275 | 5.379 | 24.923 |

Also on file, for the level question: `bench/results/2026-08-28_audio_seed_spread.json`
carries how far audio rms moves across four seeds of one prompt on the stock
path. Read the range and sd from that record. The determinism result above
strengthens it rather than contradicting it: since two executions of one graph
are bit-identical, the spread in that record is entirely seed effect, with no
run-to-run component mixed in.

## Metric against floor

Chain delta is sage minus kitchen. Floor columns are the absolute delta over the
five aligned token-order pairs.

| metric | meerkat | diner | signs | floor max | floor median | verdict |
|---|---|---|---|---|---|---|
| integrated loudness LUFS | 0 | +0.2 | split | 2.9 | 0.6 | within |
| loudness range LU | -0.6 | -0.3 | same | 0.8 | 0.4 | within |
| true peak dBFS | +0.1 | +0.3 | same | 1.7 | 0.5 | within |
| rms | +0.00022 | +0.00036 | same | 0.0116 | 0.0046 | within |
| band 20-200 Hz share | +0.0263 | -0.0245 | split | 0.0486 | 0.0130 | within |
| band 200 Hz-1 kHz share | -0.0199 | +0.0147 | split | 0.0464 | 0.0393 | within |
| band 1-4 kHz share | -0.0060 | +0.0088 | split | 0.0442 | 0.0134 | within |
| band 4-16 kHz share | -0.00042 | +0.00110 | split | 0.00515 | 0.00090 | within |
| spectral centroid Hz | -9.2 | +28.4 | split | 84.0 | 23.6 | within |
| 85% rolloff Hz | 0 | +46.9 | split | 250 | 46.9 | within |
| top band during speech | -0.00046 | +0.00136 | split | 0.0136 | 0.00099 | within |
| top band in gaps | +0.00244 | -0.00842 | split | 0.0247 | 0.0186 | within |
| energy fraction above 6 kHz | -0.00034 | +0.00016 | split | 0.00060 | 0.00025 | within |
| energy fraction above 8 kHz | -0.00015 | +0.00001 | split | 0.00078 | 0.00027 | within |
| spectral flatness, loud frames | -0.00020 | +0.00006 | split | 0.00473 | 0.00017 | within |
| spectral flatness, quiet frames | +0.00147 | +0.00063 | same | 0.0132 | 0.0040 | within |
| voiced fraction | -0.0084 | +0.0070 | split | 0.0432 | 0.0042 | within |
| periodicity, loud frames | +0.0046 | -0.0149 | split | 0.0189 | 0.0049 | within |
| HNR dB, voiced frames | +0.043 | -0.034 | split | 0.314 | 0.211 | within |
| stereo correlation | -0.00026 | +0.00013 | split | 0.0678 | 0.0020 | within |
| stereo L-R level dB | +0.0137 | -0.0068 | split | 0.0227 | 0.0069 | within |
| dropout window fraction | +0.0028 | 0 | split | 0.0070 | 0 | within |
| absolute peak | +0.0058 | +0.0121 | same | 0.0609 | 0.0158 | within |
| samples at or over 0 dBFS | 0 | 0 | -- | 0 | 0 | not measurable (all zero) |
| DC offset | +1.3e-07 | +3.2e-06 | same | 1.8e-05 | 6.2e-06 | within |
| hum prominence 400 Hz dB | +0.71 | -1.31 | split | 3.16 | 1.05 | within |
| hum prominence 800 Hz dB | +0.05 | +0.28 | same | 1.81 | 0.78 | within |
| hum prominence 1600 Hz dB | -1.02 | -0.66 | same | 1.39 | 0.61 | within |
| centroid slope Hz/s | -2.04 | -1.81 | same | 27.5 | 5.72 | within |
| rms slope dB/s | +0.011 | +0.030 | same | 0.091 | 0.038 | within |
| centroid first quarter Hz | +1.7 | +20.4 | same | 119 | 43.0 | within |
| centroid last quarter Hz | -25.9 | +0.6 | split | 470 | 106 | within |

**Read "within the floor" as "smaller than a token reorder", not as
"indistinguishable from nothing".** The floor here is a knob effect, not noise:
the noise floor is exactly zero. And the reference knob is not weak. On the
crowd_churn_long pair the reorder takes 3.41 dB off the 63 Hz band in 14 of 14
one-second windows, and on market 2.61 dB in 14 of 14 -- large, perfectly
sign-consistent, in aligned pairs. Sol's token reorder moves this model's audio
more, and more systematically, than the choice of dense attention chain does.
That is a side finding of this run, and morton is currently held pending the
short-clip panel; it deserves its own measurement rather than a decision taken
from this table.

No row clears the floor. Twenty of the thirty-two rows split in sign between the
two scenes. Twelve agree, and one of those twelve (samples at full scale) is two
zeros. The other eleven agreeing in sign is two draws agreeing, not a result;
they are listed in the JSON under
`metrics_agreeing_in_sign_across_both_clean_scenes` as the only rows worth
re-measuring if more clean pairs get rendered. The closest of them to the floor
is hum prominence at 1600 Hz: sage is lower in both scenes (-1.02 and -0.66 dB)
against a floor median of 0.61 and a maximum of 1.39.

The confounded sagelevers arm does not rescue any of this. On the diner it
agrees with the clean pair (more 4 kHz energy on the sage side, +0.76 dB against
the clean pair's +1.30 dB), but on market and hardware_aisle_short the 4 kHz
delta is -1.27 and -1.48 dB, the opposite direction.

## Where to listen

No difference clears the floor, so this section asserts nothing. These are the
loudest cells the data has, offered so a listener can check the strongest places
rather than hunt.

Both clean pairs are aligned well enough for a per-second read (envelope
correlation 0.986 diner, 0.921 meerkat): the two takes say the same words at the
same times, so a per-second band difference is timbre rather than "which word
landed when". The relabelled kitchen pair at 0.275 is **not** aligned and its
per-second grid is not interpretable.

**Diner, the sage chain is brighter in the back half.** Sage carries more
4 kHz energy than kitchen in 13 of the 14 one-second windows, +1.30 dB on
average over the clip, and more 8 kHz energy in 11 of 14, +0.80 dB. It
concentrates from about 8 s to 12 s: the shot-3 wide exterior that starts at
9.8 s, which is rain on plate glass and the music tail, no dialogue. The single
loudest second is 10 to 11 s, where sage is up in every band at once, by 0.8 to
4.6 dB -- that is a level bump in that second, not a colour change. The clearest
colour change is 8 to 9 s, where sage is +3.4 dB at 2 kHz and +2.4 dB at 4 kHz
but only +0.7 dB at 125 Hz. Clips:
`Video/reorder_panel/diner_plain_k0235_s730451892_diner_plain_k0235_00001-audio.mp4`
against
`Video/sage_chain_panel/diner_sage_chain_plain_diner_sage_chain_plain_00001-audio.mp4`.

**Meerkat, the direction reverses and the biggest cell is sub-bass.** Sage is
slightly darker on average (-0.28 dB at 8 kHz, -0.20 dB at 2 kHz) and carries
+1.14 dB more 63 Hz. The largest single cell in either clean pair is 7 to 8 s at
63 Hz, +6.11 dB on the sage side -- that falls inside shot 2, which the prompt
gives no dialogue, so it is rumble in a gap and most rigs will not reproduce it.
The one cell in a spoken stretch is 5 to 6 s at 2 kHz, where kitchen is +3.91 dB
over sage, right at the shot-1 to shot-2 cut as the narrator's line ends. Clips:
`Video/reported_clips/meerkat_default_plain_meerkat_default_plain_00001-audio.mp4`
against
`Video/reported_clips/meerkat_sage_chain_clean_meerkat_sage_chain_clean_00001-audio.mp4`.

If the owner wants one A/B to settle the impression, the diner 8 to 12 s window
is the strongest candidate in the data: it is the largest sign-consistent
band-limited difference in an aligned pair. It is still inside the token-order
floor.

## Power

Two clean pairs and five aligned floor pairs. What that can and cannot do:

**Can.** Detect any difference that survives as a consistent sign across both
scenes and exceeds the aligned token-order spread. In practice that means, per
metric, roughly: 2.9 LU of integrated loudness, 0.05 of a band's share of total
power, 84 Hz of spectral centroid, 0.0006 of the energy fraction above 6 kHz,
0.31 dB of HNR in voiced frames, 3.2 dB of hum prominence at 400 Hz, and about
1.2 to 3.0 dB of mean absolute per-second per-band difference. Those are the
floor maxima in the table; a real chain effect would have to be larger than that
*and* point the same way twice.

**Cannot.** Establish anything from two scenes agreeing in sign. The pack's own
rule for this shape of question (`bench/grade_arm_audio_spectrum.py`, which
refuses fewer than three clips per arm, and refused this data for exactly that
reason) is that the most a record can say at this count is how many draws share
a sign. Two of two is not significance. Nor can it see anything that is neither
a level, a spectrum, a periodicity nor a hum: prosody, word choice, emphasis,
delivery, or how convincing a voice sounds are not in any of these numbers, and
"more natural" is most plausibly a claim about exactly those. This method could
not have detected it.

**Also cannot.** Say anything about the kitchen scene, because no clean pair
exists for it.

The honest summary: nothing clears the floor. The owner said they could not
easily hear a difference, and the instruments agree.

## Instrument check

The subject is the meerkat kitchen clip. Directions were written before the run
(they are in the JSON under
`instrument_check.predictions_written_before_the_run`).

| arm | integrated LUFS | true peak | centroid Hz | band 4-16k share | fraction above 6 kHz | band grid mean dB |
|---|---|---|---|---|---|---|
| itself (exact copy) | 0.00 | 0.00 | 0.00 | 0.00000 | 0.00000 | 0.000 |
| re-encode, no filter | -0.10 | 0.00 | -1.66 | -0.00015 | -0.00011 | -0.025 |
| -3 dB gain | -3.10 | -3.00 | -2.04 | -0.00018 | -0.00015 | -3.034 |
| 6 kHz low-pass | -0.10 | -0.10 | -48.18 | -0.00536 | -0.00457 | -0.918 |

Against itself: exactly zero on every metric and every cell of the grid. Passes.

Minus 3 dB: moves what must move (loudness -3.10 LU, true peak -3.00 dB, rms
ratio 0.7073 which is -3.01 dB, every octave band of the grid between -3.00 and
-3.11 dB) and leaves alone what must not (band shares, centroid, rolloff, top
band at speech and in gaps, flatness, voicing, stereo correlation all move by
less than the re-encode null). Passes both ways.

Six kilohertz low-pass: 8 kHz band -6.68 dB, 4 kHz band -0.53 dB, everything at
or below 2 kHz within 0.07 dB; centroid -48 Hz; the fraction above 6 kHz falls
from 0.0055 to 0.0009, a drop of 83 per cent; the 4-16 kHz share falls 68 per
cent; integrated loudness moves only -0.10 LU, as predicted for a K-weighted
measure. Passes.

One prediction was too generous. I wrote that the 85% rolloff must move under
the low-pass; it moved 15.6 Hz, which is exactly one FFT bin at
`n_fft=2048`, 32 kHz. On this material 85 per cent of the energy sits around
560 Hz, so a 6 kHz filter barely touches it. Read the rolloff row in the main
table as near-blind to top-end changes, and the fraction above 6 and 8 kHz as
the rows that carry that question.

## Tools and settings

- `bench/measure_clip_loudness.py::ebur128` -- `ffmpeg -af ebur128=peak=true` on
  stream `0:a:0`; integrated LUFS, loudness range, true peak.
- `bench/grade_arm_audio_spectrum.py::_descriptors` -- mono 32 kHz, stft
  `n_fft=2048 hop=512`; four band shares (20-200, 200-1k, 1k-4k, 4k-16k Hz),
  centroid, 85 per cent rolloff, and the top-band share split at the 75th and
  25th percentiles of frame energy (speech against gaps). The script's grader
  path was not used: it refuses fewer than three clips per arm and every arm
  here has one. That refusal is correct for this data.
- `bench/analyze_audio_spectral_tilt.py::frames` -- `NFFT=2048 HOP=1024
  BAND_SPLIT=2000`; per-frame centroid, high-over-low ratio and rms, summarised
  as first and last quarter means and least-squares slopes. Its own `decode()`
  was not used; see the defect below.
- `bench/analyze_audio_hum.py::prominences` -- Welch `nperseg=8192` on mono
  32 kHz, dB minus a 51-point median filter of itself, peak within 25 Hz of 400,
  800 and 1600 Hz.
- New in this run, because no tool in the pack carries them: octave-band energy
  (63 Hz to 8 kHz, edges at the centre over and times root two) in one-second
  windows, which is the per-second per-band grid; third-octave shares from 63 Hz
  to 12.5 kHz; energy fraction above 6 and 8 kHz; spectral flatness split by
  frame energy; voiced fraction and an HNR proxy from the normalised
  autocorrelation peak over 70 to 400 Hz on 40 ms frames; dropout windows 40 dB
  below the clip median; DC offset, absolute peak and samples at full scale;
  stereo correlation and left-right level; and per pair the waveform and 20 ms
  envelope correlation with the best lag searched over half a second either way.
- Not used: `bench/measure_audio_video_coupling.py` measures picture against
  sound, which is a different question. `bench/measure_audio_seed_spread.py`
  needs renders that do not exist for these scenes; its existing record is cited
  instead.
- ffmpeg 6.1.1. Everything ran on CPU with the GPU hidden.

The driver and the per-clip helpers live outside the repo, in this session's
scratchpad. Nothing here was run against a GPU and no server was contacted.

## Defect found, not fixed

`bench/analyze_audio_spectral_tilt.py` cannot read a muxed `-audio.mp4`. Its
`decode()` calls `ffprobe -show_entries stream=sample_rate,channels` without
`-select_streams a:0` and then reads `streams[0]`, which for these files is the
video stream; it dies with `KeyError: 'sample_rate'`. It works on the audio-only
files it was written for. This run replaced only the decode and used `frames()`
verbatim, so the measurement is the tool's. Left unfixed on purpose.
