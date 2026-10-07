# SAM 3.1 at half and full precision through ComfyUI's own nodes, with floors and a yardstick (2026-10-07)

lane: masked
verdict: on three stretches of one crowd clip, sixteen people seeded, float16 (what ComfyUI computes SAM in on this card, from either checkpoint file) against true float32 is inside both floors and closer to float32 than bf16 is; on the hard stretch a one-level change of the input moves the tracked masks about four times more than the precision does, and every arm loses everyone on the same frame; true float32 costs about four times the seconds per frame and a third more memory; by the rule fixed before the run, precision is not a lever and the default stays ComfyUI's

**What was asked.** The owner, 2026-10-07: whether ComfyUI running SAM 3.1
at half precision matters, whether our float32 checkpoint is in fact running
at float32, and to "have the data and stacked renders showing how the two
behave on the same conditions", SAM only, with no video model.

**Who measured it.** An independent session (mrgrass), working from the
primary sources only, without the notes of the sessions whose findings it
was also checking. This record is assembled from its report and result
files. The tool is `bench/sam3_precision_arms.py`; every figure is in
[`2026-10-07_sam3_precision_arms.json`](2026-10-07_sam3_precision_arms.json).

**Corrected the same day, by the session that measured it, on reading this
record against its own files.** Three wordings below say more than was
measured. (1) Departure 2 says the detections are "visibly different": no
picture was looked at; read "measurably". (2) The verdict and "Where the arms
part" say every arm loses everyone "on the same frame": the float16 and
float32 arms do; the nudged arm loses them one frame earlier; read "within a
frame". (3) "Exact from our file and rounded from Comfy-Org's" holds on the
server's loading path only; on ComfyUI's other loading path both files give
the same weights. Also not in this record: the same session's later run of
the image range against a one-level floor, which found the range a lever on
the tracked masks well above its floor and a failure a count of non-empty
masks hides (a mask sliding onto the frame's edge); it is not yet in a
tracked record.

**How to read it.** Two floors and a yardstick, fixed before the run. Float16
run twice is the run-to-run floor. True float32 against itself with every
input pixel moved by one level of 8, at random sign, is the sensitivity
floor. bf16 against float32 is the yardstick, since Meta's own code computes
under bf16 (this is ComfyUI loaded at bf16, not Meta's code). An eight-level
nudge is the positive control the harness must be able to see. Inside the
floors: precision is not a lever. Above them but no larger than bf16's
difference: a perturbation the reference tolerates. Above bf16's, or
one-sided: name the layer. **No picture was looked at for this record**: the
stacked videos were checked by their numbers only.

## What ran

- **Clip and frames:** `lotsofpeopledance_0414_0720.mkv`, its own frames, no
  resize beyond the one ComfyUI's nodes make. Three stretches chosen from
  that session's own cut and motion statistics: a calm one (frames 400 to
  549), a hard one (3450 to 3799, a long steady zoom with large local
  change), and one with a cut (50 to 174, the cut at stretch frame 38).
- **Seeds:** ComfyUI's detect node on each stretch's first frame, once, in
  true float32; sixteen people; the same masks in every arm. Then ComfyUI's
  track node with all sixteen at once and no detecting again.
- **Arms:** the same weights file (ours) in every arm but one, loaded through
  ComfyUI's public loader with its precision override, which is what a
  loader node would pass; the text encoder left at ComfyUI's default in
  every arm.
- **True float32:** PyTorch ships with reduced-precision convolutions on. A
  self-test in the tool shows it (a neck-shaped convolution is off by a few
  parts in ten thousand as shipped, and by under one part in a million with
  both flags off) and each float32 arm logs both. A smaller convolution could
  not see the difference; the self-test's shape was changed before any arm
  ran.

## The masks, pair by pair

One minus the mean overlap of two arms' masks, over the person-frames where
either has a mask (0 is identical):

| pair | calm | hard, all 350 frames | hard, first 4 s | cut |
|---|--:|--:|--:|--:|
| float16 twice | 0 | 0 | 0 | 0 |
| float32 against float32, input nudged one level | 0.0045 | 0.218 | 0.215 | 0.0045 |
| **float16 against float32** | 0.00025 | 0.058 | 0.056 | 0.00034 |
| bf16 against float32 | 0.0016 | 0.179 | 0.178 | 0.0018 |
| float32 against float32, input nudged eight levels | 0.026 | 0.424 | 0.415 | not run |
| Comfy-Org's file against ours, both as ComfyUI runs them | not run | 0.033 | 0.031 | not run |

- **Float16 twice is bit-identical**: masks, the tracker's per-person scores
  and layer outputs. The run-to-run floor is exactly zero.
- **Float16 against float32 is inside the floors on every view**, and closer
  to float32 than bf16 is.
- **Not one-sided.** On the hard stretch the two arms' mask areas agree to
  two parts in a thousand, and each has more people than the other on a
  handful of frames. bf16's masks run about an eighth larger than float32's.
- **The positive control is seen** on both stretches it ran on.
- **The two checkpoint files** differ, as ComfyUI runs them, by about as
  much as float16 differs from float32: the text encoder's weights and two
  positional tables are exact from our file and rounded from Comfy-Org's.
  Inside the floors.

## Where the arms part

Only on the hard stretch, person by person, and as a one-level nudge does.
Float16 against float32: 16 of 1,394 person-frames disagree about a person
having a mask at all, and 10 of 5,584 of the tracker's per-person "still
here" decisions flip. Both arms lose every person on the same frame, about
four seconds in, and end every person's track on the same frame but one.
The difference inside the tracker does not grow with time. On the stretch
with a cut every arm loses all sixteen exactly at the cut and none takes
anyone up again.

**What the size of the sensitivity floor says.** On the hard stretch a
one-level change of the input moves the tracked masks by about a fifth,
against under half a percent on the calm one. The tracker there is unstable
to anything; precision is the smallest perturbation tried.

## The prediction, made before the comparison

From a detector-only pass on twenty frames of the hard stretch, the largest
one-step difference between float16 and float32 was written down, and from
it a predicted count of "still here" decisions close enough to their
threshold to flip. The first prediction, from the calm stretch's scores,
said none and was wrong: the calm stretch has no decision anywhere near the
threshold and the hard one has dozens. The second, from the hard stretch's
own float16 run and written before its other arms, gave 56 as its figure;
10 flipped. It held as an upper bound, not as an estimate. Both are in the
json with their times.

## What full precision costs

Sixteen people on this card: about four times the seconds per tracked frame,
about four times per detector call, and a third more memory, for no
difference outside the floors. The largest value leaving any block of the
image network on the hard stretch is under one percent of float16's limit,
so there is no overflow to buy safety from.

## Is the float32 file running at float32?

Not in the model. ComfyUI computes the image network, the detector and the
tracker in float16 from either file on this card, on both of its loading
paths, and nothing in this pack sets a precision. Our float32 file is used
at float32 only by the text encoder (which ComfyUI always computes in
float32, and whose weights stay exact on the server's loading path) and in
two positional tables the tracker holds in float32. Forcing the model to
float32 through ComfyUI's loader override works, and is what the float32
arms here are.

## Also measured by the same pass: three departures of ComfyUI from Meta

Found without the other sessions' notes, so an independent confirmation of
the first two (`departures_probe` in the json; seven frames from the three
stretches, ComfyUI's own nodes, four conditions).

1. **The text encoder's activation.** ComfyUI's config runs a different
   activation than Meta built the encoder with. Same weights and tokens: the
   features differ by a twentieth for `person` and by a sixth for an
   eight-word phrase; with the activation switched in memory they match to
   rounding. On real frames the detected masks barely move, except on the
   one frame where the model's presence score is low.
2. **The image range.** Read at the model's first layer: 0 to 1 from the
   detect node and very nearly that from the track node, where Meta feeds -1
   to 1. Fed Meta's range, the same frames give visibly different
   detections (one to five of sixteen masks no longer matching), while
   tracking from the same seeds on sixty calm frames barely moves. The
   detector is sensitive to the range and the tracker, on calm frames, much
   less. Which input is better was not judged.
3. **The detection score.** Meta thresholds the class score multiplied by
   the presence score, and removes overlapping detections; ComfyUI's node
   thresholds the class score alone and applies neither. On a frame with a
   low presence score Meta's rule returns nobody where ComfyUI's node
   returns sixteen. What that rule would do to the seeds was not measured.

## What this does and does not show

- One clip, three stretches, one card, sixteen people, one run per arm
  beside the repeated float16 arm. It says precision is not what the tracker
  is sensitive to here, not that it never matters.
- On the hard stretch every arm has lost everyone by about four seconds, so
  the tracked part is its opening; the table gives that part on its own.
- The nudge is random per pixel and channel, a stand-in for any small change
  of the input, not a model of a real one.
- Not run: Meta's own predictor as an arm; more than sixteen people; this
  pack's Subject Track end to end, whose match across a cut rests on
  thresholds with hundredths of margin (held as a second case).
- The stacked videos, one per stretch, are on the output share for the owner
  (`Video/mrgrass/mrgrass_sam3_precision_*`): each arm a row on the same
  frames, each person one colour, and a last row lit where the first three
  arms disagree.

## Files

- `2026-10-07_sam3_precision_arms.json`: the four pairwise summaries with
  their verdicts, the detector's one-step differences, both predictions as
  written, the cost figures, the self-test, and the departures probe.
