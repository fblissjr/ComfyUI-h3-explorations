# The turn without a prompt: a late start from a softened source (2026-10-04)

Session mrhf, in the evening, on the owner's word ("If you know how to solve
those, do it"). Rows: `2026-10-04_masked_v2v_turn_soft_arms.jsonl`. The code
the arms ran on is NOT in the tree: it is the uncommitted hunk saved beside
this file as `2026-10-04_masked_v2v_turn_soft_arms_patch.txt` (apply to
`video_mask.py` at 3204fde5), so the rows are not re-runnable from a clean
checkout. The arm graphs are local and untracked.

**Verdict: not a fix.** The softened late start turns him on both seeds
tried and does not keep the reference.

## What was asked

mrpink's three arms (`2026-10-04_masked_v2v_band_arms.jsonl`, labels
`turn_*`) showed that dropping knots from the start of the PDD8 schedule
makes the new subject follow the original's turn, and brings the original's
hair and clothes with it. The idea tested here: give the late start less to
carry, a copy of the original that keeps only its coarse shape inside the
region.

## Setup

mrpink's window and settings, so the arms compare with theirs: the band clip
from 112 s, 360 frames loaded, one window, the plain masked prompt, the
reference still, seed 730451892 unless said. The subject track at 0.186.45
with frame 48 named. Core's `SplitSigmas` drops the knots. The softening
replaces the pixels under the subject's mask (grown by half of
`grow_pixels`) before the encode; what is composited still comes from the
real frames. One window; one seed except where said.

## Arms, as seen on contact sheets against the source

The shot with the turn is frames 237 to 279 of the window. "Turns" means his
back is to the camera when the group's is.

| label | softening | knots dropped | turns | what he looks like |
|---|---|---|---|---|
| `turn_control` (mrpink) | none | 0 | no | the reference: cap, short hair, T-shirt |
| `turn_late1` (mrpink, their report) | none | 1 | yes | cap and face of the reference, the original's long hair, a grey sweatshirt |
| `turn_soft_late1` | grey, 16 px blur | 1 | yes | cap, face and short hair of the reference; a pale yellow hoodie |
| `turn_soft_late1_seed2` | the same, seed + 1 | 1 | yes | cap, long hair, a yellow jumper with a collar |
| `turn_soft_late2` | grey, 16 px blur | 2 | yes | broken: a flat white body and a flat brown hair shape |
| `turn_soft_blur32_late1` | grey, 32 px blur | 1 | no | cap, short hair, pale hoodie |
| `turn_soft_head30_late1` | grey 16 px on the top 30% of his height, the rest filled from the surroundings | 1 | half: three-quarter view, face still visible | cap, short hair, pale crewneck |
| `turn_soft_level40_late1` | grey 16 px, mean brightness moved to 0.4 | 1 | yes | as `turn_soft_late1` |
| `turn_soft_level40_gain50_late1` | the same with the contrast halved | 1 | no | cap, short hair, a yellow T-shirt |
| `turn_soft_colour_late1` | 16 px blur, colour kept | 1 | no | cap, a mint T-shirt carrying the original sweatshirt's graphic |

## What it says

- (seen) The facing cue survives a 16 px blur without colour and is gone at
  32 px, at half contrast, and, oddly, when the blur keeps its colour.
- (seen) Every arm that turns him also changes his clothes, and one of the
  two seeds brought long hair back. The turn and the reference's look did
  not come apart in any arm.
- (seen) Softening only the head and filling the body gave half a turn: the
  body's coarse shape is part of the cue. On this clip that may be the
  graphic on the chest, which a back does not have (inferred).
- (inferred) `h3_config.PDD8_SIGMAS` puts the second knot close enough to
  the first that the source enters at about a hundredth of the latent, and
  that is already enough to fix the garment's shape.
- (withdrawn) That the garment takes the brightness of what the region
  starts from: levelling the brightness changed nothing.
- (unexplained) The yellow of the garment in every grey arm.

## Not tried

A start between the first two knots; a different blur between 16 and 32 px;
more seeds; any other clip.
