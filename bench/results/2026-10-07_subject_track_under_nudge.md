# Today's Subject Track with every input value moved by one level of 255 (2026-10-07)

lane: masked
verdict: the automatic pass alone, without the owner's typed corrections: on a thirty-second stretch with five cuts and on a calm stretch of a crowd clip every decision is the same as fed and nudged, with every likeness far from its line; on the first six seconds of a hard crowd stretch the nudge makes the node skip one of four regains, leaving nine frames with nobody, and no regain in either arm lands on another person; there the nudge moves a regain's lead over the next person by about the lead the node requires

**Read this first.** This is the Subject Track's AUTOMATIC pass alone. The
owner's typed corrections on these clips are not applied. A shot the node
leaves alone here is not a statement that a render made with corrections
was wrong, and nothing here judges which person is the right one.

**What was asked.** The owner, 2026-10-07: whether yesterday's handling
between shots (cuts, the match across a cut, regain after a loss) still
stands after a day that found ComfyUI's tracker sensitive to changes of the
input too small to see, and what to do "in cases where scores are thin
margins". The node's lines have a few hundredths of margin and were set on
one clip (`subject_track.py`: `REGAIN_SAME`, `REGAIN_MARGIN`,
`MATCH_FLOOR`).

**How.** [`bench/subject_track_under_nudge.py`](../subject_track_under_nudge.py)
drives the node's own functions at `h3_config.SUBJECT_TRACK` on ComfyUI's
SAM 3.1 as ComfyUI runs it, on frames from the loader node the masked
graphs use, twice: as fed, and with every value moved one level of 255 at
random sign (the nudge of
[`2026-10-07_sam3_precision_arms.md`](2026-10-07_sam3_precision_arms.md),
same seed). Where an arm was run twice the two runs are identical to the
last digit, so the floor is zero and the later runs leave the repeat out.
Outside the server, the card otherwise idle. Every number is in
[`2026-10-07_subject_track_under_nudge.json`](2026-10-07_subject_track_under_nudge.json),
one key per run, and every table below is printed from it by the tool's
`render --key`.

**The reading, written before the first run** (it is the tool's
docstring). A decision is, per shot: the cuts that bound it; whether the
subject was taken as the pick, over the line, as the only person, or not at
all; which person, read as the overlap of the two arms' masks (under a half
is another person or another place); and each regain. HOLDS: every decision
the same in both arms. HOLDS BY A THIN MARGIN: the same, but a likeness
moved by more than half its distance to its line. DOES NOT HOLD: a decision
differs, which is a finding about that line on that shot.

## Many cuts: `thrill_2160.mkv`

`thrill_2160.mkv` from 56.0 s, 720 frames at 24.0 a second, 1024x768; the node's lines: {'MATCH_FLOOR': 0.8, 'PLAIN_FLOOR': 0.91, 'MATCH_THRESHOLD': 0.82, 'PLAIN_SAME': 0.93, 'REGAIN_SAME': 0.88, 'REGAIN_MARGIN': 0.03, 'MIN_GAP': 0.1}.

| arm | cuts found | pick frame | others on it | match line | shots taken | shots absent | times seeded again |
|---|---|---|---|---|---|---|---|
| as fed | [501, 544, 611, 626, 705] | 505 | 6 | 0.8 | 2 | 4 | 0 |
| as fed, again | [501, 544, 611, 626, 705] | 505 | 6 | 0.8 | 2 | 4 | 0 |
| nudged | [501, 544, 611, 626, 705] | 505 | 5 | 0.8 | 2 | 4 | 0 |
| nudged, again | [501, 544, 611, 626, 705] | 505 | 5 | 0.8 | 2 | 4 | 0 |

### as fed | as fed, again

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | taken over the line / taken over the line | True | 0.9188 / 0.9188 | 0.0 | 0.1188 | None / None | 0 / 0 | 1.0 | 501 |
| 2 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 1.0 | 43 |
| 3 | absent / absent | True | 0.6491 / 0.6491 | 0.0 | 0.1509 | 0.4494 / 0.4494 | 0 / 0 | None | 0 |
| 4 | absent / absent | True | 0.6762 / 0.6762 | 0.0 | 0.1238 | 0.6393 / 0.6393 | 0 / 0 | None | 0 |
| 5 | absent / absent | True | 0.6222 / 0.6222 | 0.0 | 0.1778 | 0.5163 / 0.5163 | 0 / 0 | None | 0 |
| 6 | absent / absent | True | 0.6308 / 0.6308 | 0.0 | 0.1692 | 0.5925 / 0.5925 | 0 / 0 | None | 0 |

### nudged | nudged, again

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | taken over the line / taken over the line | True | 0.9202 / 0.9202 | 0.0 | 0.1202 | None / None | 0 / 0 | 1.0 | 501 |
| 2 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 1.0 | 43 |
| 3 | absent / absent | True | 0.6448 / 0.6448 | 0.0 | 0.1552 | 0.4548 / 0.4548 | 0 / 0 | None | 0 |
| 4 | absent / absent | True | 0.6737 / 0.6737 | 0.0 | 0.1263 | 0.642 / 0.642 | 0 / 0 | None | 0 |
| 5 | absent / absent | True | 0.6243 / 0.6243 | 0.0 | 0.1757 | 0.5205 / 0.5205 | 0 / 0 | None | 0 |
| 6 | absent / absent | True | 0.6389 / 0.6389 | 0.0 | 0.1611 | 0.6036 / 0.6036 | 0 / 0 | None | 0 |

### as fed | nudged

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | taken over the line / taken over the line | True | 0.9188 / 0.9202 | 0.0014 | 0.1188 | None / None | 0 / 0 | 0.9965 | 501 |
| 2 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 0.9949 | 43 |
| 3 | absent / absent | True | 0.6491 / 0.6448 | 0.0043 | 0.1509 | 0.4494 / 0.4548 | 0 / 0 | None | 0 |
| 4 | absent / absent | True | 0.6762 / 0.6737 | 0.0025 | 0.1238 | 0.6393 / 0.642 | 0 / 0 | None | 0 |
| 5 | absent / absent | True | 0.6222 / 0.6243 | 0.0021 | 0.1778 | 0.5163 / 0.5205 | 0 / 0 | None | 0 |
| 6 | absent / absent | True | 0.6308 / 0.6389 | 0.0081 | 0.1692 | 0.5925 / 0.6036 | 0 / 0 | None | 0 |

**HOLDS.** The best likeness moves by under a hundredth in every shot and
the nearest shot sits more than a tenth from the match line. The nudge
changed what the detector returned on the pick frame (five other people
against six) without changing the pick. Four of the six shots are left
alone by the automatic pass in both arms.

## A crowd, calm: `lotsofpeopledance_0414_0720.mkv` from 16 s

`lotsofpeopledance_0414_0720.mkv` from 16.0 s, 144 frames at 24.0 a second, 1344x760; the node's lines: {'MATCH_FLOOR': 0.8, 'PLAIN_FLOOR': 0.91, 'MATCH_THRESHOLD': 0.82, 'PLAIN_SAME': 0.93, 'REGAIN_SAME': 0.88, 'REGAIN_MARGIN': 0.03, 'MIN_GAP': 0.1}.

| arm | cuts found | pick frame | others on it | match line | shots taken | shots absent | times seeded again |
|---|---|---|---|---|---|---|---|
| as fed | [] | 4 | 15 | 0.8 | 1 | 0 | 0 |
| as fed, again | [] | 4 | 15 | 0.8 | 1 | 0 | 0 |
| nudged | [] | 4 | 15 | 0.8 | 1 | 0 | 0 |
| nudged, again | [] | 4 | 15 | 0.8 | 1 | 0 | 0 |

### as fed | as fed, again

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 1.0 | 144 |

### nudged | nudged, again

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 1.0 | 144 |

### as fed | nudged

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 0 / 0 | 0.9959 | 144 |

**HOLDS.** One shot, fifteen other people on the pick frame, the subject
followed alone on every frame in every arm.

## A crowd, hard: the same clip from 138 s, six seconds

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the node's lines: {'MATCH_FLOOR': 0.8, 'PLAIN_FLOOR': 0.91, 'MATCH_THRESHOLD': 0.82, 'PLAIN_SAME': 0.93, 'REGAIN_SAME': 0.88, 'REGAIN_MARGIN': 0.03, 'MIN_GAP': 0.1}.

| arm | cuts found | pick frame | others on it | match line | shots taken | shots absent | times seeded again |
|---|---|---|---|---|---|---|---|
| as fed | [] | 4 | 15 | 0.8 | 1 | 0 | 4 |
| nudged | [] | 4 | 15 | 0.8 | 1 | 0 | 3 |

### as fed | nudged

Same cuts: True; same pick frame: True; match line 0.8 and 0.8.

| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | the pick / the pick | True | 1.0 / 1.0 | 0.0 | None | None / None | 4 / 3 | 0.7881 | 124 |

Every frame the `as fed` arm looked at after a loss (the line is 0.88, the lead required 0.03):

| shot | frame | detections | best | next person | lead | taken |
|---|---|---|---|---|---|---|
| 1 | 23 | 16 | 0.9632 | 0.8737 | 0.0895 | yes |
| 1 | 71 | 16 | 0.8675 | 0.8553 | 0.0122 | no |
| 1 | 83 | 16 | 0.9244 | 0.8791 | 0.0453 | yes |
| 1 | 92 | 16 | 0.9227 | 0.8923 | 0.0304 | yes |
| 1 | 105 | 16 | 0.8974 | 0.8885 | 0.0089 | no |
| 1 | 117 | 16 | 0.8754 | 0.8731 | 0.0023 | no |
| 1 | 129 | 16 | 0.8967 | 0.8875 | 0.0092 | no |
| 1 | 141 | 16 | 0.9308 | 0.8975 | 0.0333 | yes |

Every frame the `nudged` arm looked at after a loss (the line is 0.88, the lead required 0.03):

| shot | frame | detections | best | next person | lead | taken |
|---|---|---|---|---|---|---|
| 1 | 23 | 16 | 0.9641 | 0.8752 | 0.0889 | yes |
| 1 | 68 | 16 | 0.8898 | 0.8741 | 0.0157 | no |
| 1 | 80 | 16 | 0.8704 | 0.869 | 0.0014 | no |
| 1 | 92 | 16 | 0.9197 | 0.8573 | 0.0624 | yes |
| 1 | 105 | 16 | 0.8927 | 0.8901 | 0.0026 | no |
| 1 | 117 | 16 | 0.8768 | 0.8691 | 0.0077 | no |
| 1 | 129 | 16 | 0.9023 | 0.8878 | 0.0145 | no |
| 1 | 141 | 16 | 0.9315 | 0.8968 | 0.0347 | yes |

Each time an arm seeded the track again (the line is 0.88, the lead required 0.03):

| shot | arm | empty from | seeded on | best | next person | lead | the other arm seeded within a stride | overlap of the two arms over the next second | frames compared |
|---|---|---|---|---|---|---|---|---|---|
| 1 | as fed | 23 | 23 | 0.9632 | 0.8737 | 0.0895 | True | 0.987 | 24 |
| 1 | nudged | 23 | 23 | 0.9641 | 0.8752 | 0.0889 | True | 0.987 | 24 |
| 1 | as fed | 71 | 83 | 0.9244 | 0.8791 | 0.0453 | True | 0.5792 | 22 |
| 1 | as fed | 92 | 92 | 0.9227 | 0.8923 | 0.0304 | True | 0.9802 | 13 |
| 1 | nudged | 68 | 92 | 0.9197 | 0.8573 | 0.0624 | True | 0.9802 | 13 |
| 1 | as fed | 105 | 141 | 0.9308 | 0.8975 | 0.0333 | True | 0.9873 | 3 |
| 1 | nudged | 105 | 141 | 0.9315 | 0.8968 | 0.0347 | True | 0.9873 | 3 |

**DOES NOT HOLD for one regain.** The three regains both arms make land on
the same place. The fourth, on frame 83 as fed, is not made in the nudged
arm, which loses the subject three frames earlier and has nobody until
frame 92. So the nudge costs a gap, not a wrong person. The thinnest lead
as fed (frame 92) doubles in the nudged arm because the next person's
score moved: the nudge moves a lead by about the lead the node requires.
The tables of every look show why the arms part: they lose the subject
three frames apart, so they look at different frames, and at most of the
looks that take nobody the best person is over the line and is refused on
the lead alone. Every look returns sixteen detections, the most the node
asks for: on this clip the frame holds more people than that.

## The same stretch for fourteen seconds, as fed only

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 336 frames at 24.0 a second, 1344x760; the node's lines: {'MATCH_FLOOR': 0.8, 'PLAIN_FLOOR': 0.91, 'MATCH_THRESHOLD': 0.82, 'PLAIN_SAME': 0.93, 'REGAIN_SAME': 0.88, 'REGAIN_MARGIN': 0.03, 'MIN_GAP': 0.1}.

| arm | cuts found | pick frame | others on it | match line | shots taken | shots absent | times seeded again |
|---|---|---|---|---|---|---|---|
| as fed | [] | 4 | 15 | 0.8 | 1 | 0 | 7 |

Each time the `as fed` arm seeded the track again (the line is 0.88, the lead required 0.03):

| shot | empty from | seeded on | best | next person | lead |
|---|---|---|---|---|---|
| 1 | 23 | 23 | 0.9632 | 0.8737 | 0.0895 |
| 1 | 71 | 83 | 0.9244 | 0.8791 | 0.0453 |
| 1 | 92 | 92 | 0.9227 | 0.8923 | 0.0304 |
| 1 | 105 | 141 | 0.9308 | 0.8975 | 0.0333 |
| 1 | 150 | 162 | 0.8846 | 0.8412 | 0.0434 |
| 1 | 167 | 179 | 0.9209 | 0.8901 | 0.0308 |
| 1 | 191 | 275 | 0.9001 | 0.864 | 0.0361 |

Shot 1: 179 of 336 frames with a mask; searched and found nobody over [[277, 335]].

One arm: the run was stopped after it (it took about nine minutes an arm
and the card was shared). It is here for its regains: seven, five of them
leading the next person by less than half as much again as the lead
required.

## What this shows

- **Between shots, on these two clips, the automatic decisions do not
  depend on a one-level change of the input** (measured, two stretches).
- **On hard crowd footage a regain does** (measured, one stretch): the lead
  over the next person is on the noise. The node's required lead is a
  module constant today, not an input.
- **Every regain taken in both arms was the same place in both**
  (measured). That is agreement between arms, not a check that the place is
  the subject.
- **Repeats are bit-identical** (measured, three runs), so nothing here is
  run-to-run noise.

## What it does not show

- That either arm follows the right person: there are no labels here.
- Anything about a render.
- More than one nudge, one stretch per kind of footage, or the footage
  with the owner's corrections applied.
- The Subject Track on a corrected model: these runs are stock ComfyUI
  SAM 3.1, as the shipped node runs.

## Reproducing it

With `T` for `bench/subject_track_under_nudge.py` and `J` a json path:

```
<python> T run --clip thrill_2160.mkv --second 56 --seconds 30 --width 1024 --rate 24 --json J
<python> T run --clip lotsofpeopledance_0414_0720.mkv --second 16 --seconds 6 --width 1344 --rate 24 --json J
<python> T run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --no-repeats --json J
<python> T render --json J
```

## Files

- `2026-10-07_subject_track_under_nudge.json`: four keys, one per run; the
  fourteen-second key holds the one arm that ran, taken from its log.
