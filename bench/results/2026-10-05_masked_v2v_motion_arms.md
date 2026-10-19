# 2026-10-05: the turn arms on the masked lane (encoder-only motion reference)

Session mryellow. Rows: `2026-10-05_masked_v2v_motion_arms.jsonl` (one per
arm, wall and per-node seconds, from `bench/run_graph_arms.py`). The graphs and
contact sheets are not tracked. The question the arms answer is the session's
research note's first: why the replaced subject does not
turn with the band when the original does.

## Setup, shared by every arm

The band clip from 112 s, 360 source frames, one window of 345, the
Masked Source with the subject's SAM 3.1 mask, the generic male prompt
(`internal/internal_prompt_bank/mask/swap_male.txt`, mrhf) and the still. The
motion arms add the Masked Source's `motion_reference = subject only` at
`motion_short_edge = 384` with `motion_vae` off (the subject's own frames on
grey reach the encoder only, no DiT rows; 0.190.7) and the prompt's
relationship sentence on `<Video 1>` (`swap_male_motion.txt`). Seed one is
mrhf's seed; seed two is one above it. Judged on contact sheets of frames
237, 247, 257, 267 and 277 (the turn shot) against mrhf's generic-prompt
control at seed one and `control_s2` at seed two. One window, so the
multi-window carry is not exercised.

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `motion_s1` | fl2va PDD8 bake (the shipped graph) | 8 | subject frames, encoder only | no | kept |
| `motion_s2` | same, seed two | 8 | same | no | drifted (no cap, longer hair) |
| `control_s2` | fl2va PDD8 bake, generic prompt, no reference | 8 | none | no | same drift as `motion_s2` |
| `ref2va_base_s1` | ref2va undistilled | 16 | same as motion | **yes**, back to camera from about frame 267 | kept |
| `fl2va_base_s1` | fl2va undistilled | 16 | same as motion | no | kept |

Pending at the time of writing (appended below as they land):
`ref2va_pdd8_s1` (ref2va with its PDD8 bake, 8 steps), `ref2va_pdd6_s1` (the
same bake on `h3_config.PDD_MANUAL_SIGMAS`, six evaluations; a `steps`
request of 6 is refused by the node and the knot list is the sanctioned
form, `docs/distills/pdd.md`), `ref2va_base_s2` (the base at seed two).

## What the five rows say

- The encoder-only video reference carries the turn on ref2va and nothing
  on fl2va. fl2va ignores it as the PDD8 bake (two seeds) and undistilled at
  sixteen steps (one seed), so the checkpoint is the variable, not the
  distill or the step count.
- The seed-two look drift on the bake is the seed's, not the reference's:
  the control at that seed drifts the same way with no reference wired.
- A same-clip reference of the subject does not confuse ref2va: the look
  stays the prompt's (cap, grey T-shirt) while the pose follows the
  reference. This answers the board's `q-same-clip-reference` for one seed.
- Wall times in the rows are queue-inclusive where arms shared the server
  (`control_s2` sat behind the ref2va base); the song node's own seconds
  are the per-node field.

## Caveats

One seed on each base arm; the bake arms have two. The turn is read by eye
on five frames; no motion metric. The band clip's turn is the only motion
tested. Cache state: every arm ran on one server started at 14:02 with the
models staged once, the encoder reloaded per prompt change.

## Appended 14:55: the ref2va PDD8 bake

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_pdd8_s1` | ref2va with its PDD8 bake (`PDD_REF2VA_LORA`) | 8 | same as motion | no; he faces the camera and from frame 257 opens his hands in a gesture the source does not make | kept |

The bake on ref2va loses the turn the base had at the same seed, so the
reference's hold on the pose does not survive the distill at eight steps, or
does not survive eight steps. Those two are separated by `ref2va_base8_s1`,
queued: the undistilled base at eight steps. If it turns him, the distill is
what loses the reference; if it does not, it is the step count. The gesture
is new on this arm and may be the bake's own habit under a video reference;
one seed.

## Appended 15:10: the ref2va base at a second seed

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base_s2` | ref2va undistilled, seed two | 16 | same as motion | **yes**, back to camera from about frame 257 | kept (cap, short hair, dark T-shirt; the control at this seed has long hair) |

Both seeds on the base turn him. The seed-two drift seen on the fl2va bake
does not appear here with the reference wired.

## Appended 15:15: the ref2va bake on six evaluations

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_pdd6_s1` | ref2va with its PDD8 bake on `PDD_MANUAL_SIGMAS` | 6 evaluations | same as motion | no; arms at his sides, no gesture | kept |

Same as eight on the turn (none) and cleaner on this seed: the open-hands
gesture the eight-step bake added is absent. The song-node seconds are in the
rows for the cost difference. One seed.

## Appended 15:22: the ref2va base at eight steps

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base8_s1` | ref2va undistilled | 8 | same as motion | no; faces the camera, hands lifted a little from frame 257 | kept |

The undistilled base at eight steps loses the turn the same base carried at
sixteen, so it is the step count, not the distill, that loses the reference:
the PDD8 bake at eight behaves as the base at eight does, gesture included.
The reference's hold on the pose needs more evaluations than eight on this
seed. `ref2va_base12_s1` is queued to narrow where between eight and sixteen
the turn appears.

## Appended 15:35: the ref2va base at twelve steps, and the set complete

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base12_s1` | ref2va undistilled | 12 | same as motion | **yes**, back to camera from about frame 257, as at sixteen | kept |

Twelve carries the turn as sixteen does; eight does not. The threshold on
this seed is between eight and twelve evaluations. The set is complete:
eleven rows, the server stopped at 15:35.

## Verdict

- fl2va does not take an encoder-only video reference of the subject here,
  distilled or not, at eight or sixteen steps.
- ref2va takes it at twelve and sixteen steps on the base and loses it at
  eight with or without its PDD8 bake, and at six evaluations. The step
  count is the variable, not the distill.
- The route therefore exists today on ref2va at twelve steps, at that
  cost; the song-node seconds in the rows price it against the shipped
  eight-step fl2va graph. Whether the lane moves its moving shots there is
  the owner's call on the board.

## Appended 16:45: the control with no reference, and the shipped graph's own render

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base12_noref_s1` | ref2va undistilled, the generic prompt as mrhf wrote it, `motion_reference` none | 12 | **none** | no; faces the camera through the shot | kept |
| `ship_ref2va_motion_s1` | `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json` as first generated (its prompt the shipped masked text with the three `<Video 1>` lines grafted on), the band clip, seed one, one window | 12 | same as motion | **no.** Corrected 2026-10-05 after mrteal's yaw metric disagreed (`2026-10-05_subject_yaw_calibration.md`: largest turn 34 degrees against 180 for the arm) and tight crops confirmed it: the owner's judgment of the clip: "he doesnt turn his head he bobs it side to side"; on tight crops the shoulders angle around frames 257 to 263 and are square to the camera again by 273. This row first said "yes, a partial one: side-on by frame 257, three-quarter back by 267", a misread of the contact sheet. | kept |

The control is the question the owner asked ("if you just use the ref2va
model and dont pass in the video reference"): on ref2va at twelve steps with
nothing but the plate, the still and the generic prompt he does not turn. So
the reference is what carries the turn, and ref2va above eight steps is what
lets it through. The shipped graph's own render, on the first shipped prompt, does not turn
him; the arm at the same seed does. The two differ only in the prompt's base
text: the arm ran mrhf's generic subject-swap text with the three `<Video 1>`
lines, the first shipped prompt grafted the same lines onto
`ref2va_masked_subject_swap`. So the base text is not a knob but part of
what was measured, and the same evening the shipped prompt
(`prompt_bank/ref2va_masked_subject_motion.txt`) was replaced by the arm's
exact text and both motion graphs rebuilt; the arm `ref2va_base12_s1` is
then the shipped chain and prompt at the shipped count, and a fresh render
of the regenerated graph is owed before "verified" is said of it again.

## Verdict, revised

- fl2va does not take an encoder-only video reference of the subject here,
  distilled or not, at eight or sixteen steps.
- ref2va takes it at twelve and sixteen steps and loses it at eight with or
  without its PDD8 bake, and at six evaluations. Without the reference,
  ref2va at twelve does not turn him. The step count and the reference are
  both needed; the distill is not the variable.
- The route ships as `h3_video_to_video_masked_song_ref2va_motion_api.json`
  (and `daily/h3_mask_ref2va_motion_api.json`) with the measured prompt text
  since the evening's correction; its first render on the first prompt did
  not turn him (see the corrected row), so the evidence for the shipped
  chain is the arm `ref2va_base12_s1`, not a render of the generated file.
  The fl2va PDD8 graph stays the default for a shot that needs no movement
  from the source.

## Appended 17:50: can a distill carry it, and is Sol the cause at eight

Owner's questions: whether any distill or knot list carries the reference,
and whether Sol-Attn's sparse steps, rather than the count, lose it at eight.
Same window, seed, still, reference and prompt as the arms above.

| arm | chain | evaluations | Sol | turn | look |
|---|---|---|---|---|---|
| `ref2va_pdd8mid_s1` | ref2va PDD8 LoRA on an eight-knot list that moves two knots into the mid range (`1.0, 0.972973, 0.923077, 0.878049, 0.8, 0.734694, 0.631579, 0.444444, 0.0`, widths `[8,8,4,4,2,2,2,2]` on the PDD grid) | 8 | on, start 0.0 (the PDD recipe of the day) | no; no gesture either | kept |
| `ref2va_base10_s1` | ref2va undistilled | 10 | on, start 0.2 | no; faces the camera, hands near the waist | kept |
| `ref2va_pdd8_nosol_s1` | ref2va PDD8 LoRA, the file's own knots, the Sol node bypassed (dense kitchen attention every step) | 8 | off | no; a glance to the side around frame 257, then the camera | kept |
| `ref2va_pdd8_sol02_s1` | ref2va PDD8 LoRA, the file's own knots, Sol started at 0.2 (what the reverted shipped chain runs) | 8 | on, start 0.2 | no | kept |

Not rendered: `ref2va_flashgen4_s1` (FlashGen four-step, the rank-64 ref2va
LoRA) and `ref2va_base8_nosol_s1` were killed mid-render when my own wait
stopped the server with them still queued, three runners having interleaved
on one queue; and then dropped, not requeued, on the owner's word the same
evening: "we need to have a path to being comfortable and confident doing
those instead of optimizing for this one off." The lane moves to a benchmark
across diverse clips (the board's `build-lane-benchmark`); the band clip's
schedule ladder ends here. The
`ref2va_pdd8mid_s1` row with zero song-node seconds is the mux failure of the
first attempt (its window folder was removed mid-run), not a render.

What the four say:

- The threshold on the undistilled base is between ten and twelve
  evaluations on this seed: eight and ten face the camera, twelve and
  sixteen turn.
- Moving the PDD8 knots into the mid range does not help; nor does the
  six-evaluation list above it. The bake behaves as the base at the same
  count, so a knot list cannot buy back what eight evaluations lose.
- Sol is not the cause at eight: with the Sol node bypassed he still faces
  the camera, and with Sol started at 0.2 instead of 0.0 he still does. The
  owner's reversal of the 0.0 start (0.194.2) stands on its own reasoning;
  it is not what the reference needed.
- So no shipped distill carries an encoder-only video reference on ref2va,
  and the dense-block question the owner raised for the case where Sol was
  the cause does not arise. The price of the reference is the step count,
  twelve on this seed.
