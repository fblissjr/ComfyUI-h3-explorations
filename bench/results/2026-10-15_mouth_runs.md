# The head reference's mouth over time: mouth v2 (2026-10-15)

lane: masked video to video
verdict: data, against the outside reviewer's watched table (G6). Holding the form against its neighbours and filling short drop-outs takes 2823's form switches from 46 to 8 and its one-frame runs from 16 to 0, and leaves no open frame without a mouth on any of the five references but 630's 658-667, the known trade. The hollow outline on a closed mouth (item 1) is not built: the class map cannot tell those mouths from open ones.

**Read this first.** The figures are made with `bench/head_reference.py`'s own functions on the five references remade
today (2823's from its folder moved aside), with no reference made again. The watched states are G6's
(`internal/gemini/2026-10-15_watch_mouth.md`, section 1). Every value is a starting value set on these frames: a fit,
not a test. Renders decide what a viewer sees; 2823b's (G3) shows whether 46 switches were visible at all.

## The rules (`MOUTH_RUNS`, `CLASS_MOUTH_GATE`'s `agreed_*`)

- A frame enters the lip lines at `MOUTH_CHECK`'s share (0.8) and leaves them only under `leave_lines` (0.6): the
  share comes in steps of 1/15 and jitters round 0.8 on a big close-up.
- A run of at most `fill_frames` (2) frames with no mouth between frames with one takes the class map's mouth where
  it lies at least `fill_on_body` (0.45) on the body model and within the gate's `near` of the pose mouth. On these five
  references it fills no frame: its case (805's 822-823 inside one sung vowel) is now kept by the agreed floor below. It
  stays as a guard.
- Where the lines' share is at least `agreed_share` (0.60) and the class map's mouth is within `agreed_near` (0.20 of a
  face height) of the pose mouth, `on_body`'s floor is `agreed_on_body` (0.47).
- No rule drops a one-frame mouth: on these frames it would drop 805's open 824 to clear 1010's closed 1021.

## Per reference

| reference | rule | switches | one-frame runs | open frames with no mouth |
|---|---|---|---|---|
| lead_0630_0680 | landed (ca86c125) | 12 | 6 | 10 (658, 659, 660, 661, 662, 663, 664, 665, 666, 667) |
| lead_0630_0680 | v2 | 6 | 0 | 10 (658, 659, 660, 661, 662, 663, 664, 665, 666, 667) |
| lead_0805_0833 | landed (ca86c125) | 12 | 6 | 3 (819, 822, 823) |
| lead_0805_0833 | v2 | 8 | 3 | 0 |
| second_1010_1031 | landed (ca86c125) | 5 | 2 | 0 |
| second_1010_1031 | v2 | 5 | 2 | 0 |
| second_5114_5129 | landed (ca86c125) | 2 | 0 | 0 |
| second_5114_5129 | v2 | 2 | 0 | 0 |
| lead_2823_3097 | landed (ca86c125) | 46 | 16 | 0 |
| lead_2823_3097 | v2 | 8 | 0 | 0 |

## Hysteresis, how far to leave the lines (2823)

| leave the lines under | switches | one-frame runs | frames on the lines |
|---|---|---|---|
| 0.8 (landed) | 46 | 16 | 214 |
| 0.7 | 32 | 7 | 232 |
| 0.65 | 18 | 3 | 245 |
| 0.6 | 8 | 0 | 264 |
| 0.5 | 6 | 0 | 266 |

At 0.6, of the 54 frames v2 holds on the lines under 0.8 (all on 2823), G6 calls 50 open, 2866-2867 pursed or
borderline, and 2979-2980 closed or clenched.

## Item 1, the hollow outline on a closed mouth: the class map cannot tell these apart

On the frames given the class map's own mouth, against G6:
- the open inside (teeth or tongue) is 0 px on 12 of 18 closed frames, and also on 14 of 83 open ones (631, 641-644,
  649, 5114-5115, 5124-5129);
- the gap between the upper and lower lip pieces, column by column, is 0 px on the closed frames with no inside (657:
  2 px), and also 0 px on 9 of those 14 open frames (631, 641-643, 649, 5115, 5124, 5126, 5127), 1-4 px on the rest.

So "no inside, no mouth" would mend 12 closed frames and lose 14 open ones, 5114's talking among them (under the landed
rule; under v2, which holds 649 on the lines, 13 open: 631, 641-644, 5114-5115, 5124-5129). Not built as a default. The
lead's later word of 2026-10-15, from 5114d's and 630d's renders, where the model opened her mouth with the source on
frames given none: `--class-mouth-needs-hole`, an option for one test (5114e), judged on teeth or tongue labelled, applied
before the fill so the fill never draws a frame it dropped.

## What it cannot see

Whether a switch between the lip lines and the class map's outline shows in a render at all. Whether a mouth drawn is
open when hers is closed. The picture.
