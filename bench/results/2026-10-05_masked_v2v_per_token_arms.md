# The per-token late start on the band clip's turn window (2026-10-05)

lane: masking
verdict: it turns him on three of the four chains and keeps his face, cap and hair; it changes his clothes on every arm; fl2va PDD8 with no reference does not turn. Not adopted; the option stays in the node, off by default.

Session mrhf. Rows: `2026-10-05_masked_v2v_per_token_arms.jsonl`. The code is
in the tree since 0.193.3 (`video_mask.py`: `start_from`, `soften_subject`,
`top_of`, `start_zero_tokens`; the song node slices the schedule). The arm
graphs are local and untracked; each is
a copy of its control with the four `start_*` inputs set and Sol's
`start_percent` at 0.2. Renders: `Video/mrhf/masked_v2v_pt_*` on the share.

## What was asked

Every late start tried before put the whole subject under the mask and
carried the original's clothes with its pose
(`2026-10-04_masked_v2v_turn_soft_arms.md`). This one decides by token: the
schedule starts one knot late, the top three tenths of the subject's height
starts from a grey 16 px blur of the original, and the rest of the subject
starts from a zero latent. The owner approved it on the masking board, then
asked for "the permutations with what we know works / doesnt work".

## Setup

The band clip from 112 s, 360 frames loaded, one window, the subject picked
automatically, the generic prompt (the motion arms use the measured motion
prompt, as their controls do). Seed 730451892 unless said. Sol starts at 0.2
on every arm, by the owner's rule of the same day. The sampler's progress
bars in the server log show one step fewer than each control, so the late
start was in effect. Read from contact sheets against the source; "turns"
means his back is to the camera when the group's is. One window.

## Arms

| label | chain | motion reference | turns | clothes | hair | control, and what it did |
|---|---|---|---|---|---|---|
| `pt_fl2va_pdd8_control_sol02_s1` | fl2va PDD8, no late start | none | no | the reference's T-shirt | short | (this is the control) |
| `pt_fl2va_pdd8_s1` | fl2va PDD8 | none | no | a striped knit sweater, long sleeves | short | the row above |
| `pt_fl2va_pdd8_s2` | fl2va PDD8, seed + 1 | none | yes, late in the shot | a yellow and black striped sweater | long | mryellow's `control_s2` (Sol from 0.0): no turn, dark T-shirt, long hair |
| `pt_ref2va12_noref_s1` | ref2va base, 12 steps | none | yes | a pale striped T-shirt | short | `ref2va_base12_noref_s1`: no turn, the T-shirt |
| `pt_ref2va8_motion_s1` | ref2va base, 8 steps | subject only | yes | a grey striped T-shirt | short | `ref2va_base8_s1`: no turn, the T-shirt |
| `pt_fl2va_pdd8_motion_s1` | fl2va PDD8 | subject only | yes | a striped polo shirt | short | mryellow's `motion_s1` (Sol from 0.0): no turn, a pale T-shirt |

## The owner's verdicts

From the clips, the same evening, as relayed by mryellow and held on the
masking board as finding my-28. They outrank my reading of contact sheets.

| label | the owner |
|---|---|
| `pt_fl2va_pdd8_motion_s1` | "turns but has the wrong clothes but maybe thats ok if the prompt isnt specific enough" |
| `pt_ref2va8_motion_s1` | "also turns ... but same shirt difference but again maybe needs more specific prompt" |
| `pt_ref2va12_noref_s1` | turns ("they all turn so far") |
| `pt_fl2va_pdd8_s2` | "totally different clothes and has the real video's singer's long hair and does NOT turn" |
| `pt_fl2va_pdd8_control_sol02_s1` | "has the right clothes but no turn" |
| `pt_fl2va_pdd8_s1` | "has similar looking clothes as the first 3 but its a hoodie instead of a sweater but does NOT turn" |

**Corrected by this:** the table above says `pt_fl2va_pdd8_s2` turns late in
the shot. I read that off the last two tiles of a contact sheet; the owner,
watching the clip, says he does not turn. So fl2va PDD8 with neither
reference does not turn on either seed.

Two explanations of the clothes are on the table and neither is tested: the
owner's, that the prompt does not name the clothes specifically enough; and
mine, the zeroed start below. Each is one arm to separate: the motion arm
with a prompt that names the T-shirt by colour and cut, and the same arm
with the body's tokens started from the softened source.

## What it says

- (seen) The softened top of the original is a facing cue the model reads.
  On ref2va it carries the turn with no reference at all, where the control
  faces the camera; with the motion reference it carries the turn at eight
  steps, the count that loses the reference alone; and on fl2va PDD8 it
  carries it together with the motion reference, which did nothing there by
  itself.
- (seen) fl2va PDD8 with neither reference is the weak chain: no turn on
  either seed, by the owner's viewing (my sheet reading of a late turn on
  the second seed is corrected above).
- (seen) The clothes change on every late-start arm, and to a striped
  garment each time. The reference's short sleeves survive on the three
  arms that have ref2va or the motion reference; its plain T-shirt does not
  survive on any.
- (seen) The face, the cap and the short hair hold on every first-seed arm.
  On the second seed the hair is long in the control as well, so the long
  hair there is the seed's and not this method's. That also weakens what
  `2026-10-04_masked_v2v_turn_soft_arms.md` said of its second seed.
- (measured, one decode) A zero latent is not nothing to the model. Decoded
  by the H3 video VAE it is a warm mid grey with a fine regular texture at
  the latent cell's pitch. Read with the row above (inferred): the body's
  tokens start from a faint weave, and a knit or a stripe is what came out.
  So "the body from nothing" was the wrong description of this arm, and the
  board's finding hf-01 is right about the source and wrong about neutral.
- (confound, stated) `pt_fl2va_pdd8_s2` and `pt_fl2va_pdd8_motion_s1` differ
  from their controls in Sol's start as well as in the late start.
- (seen, not this method's) Both ref2va rows at eight steps with the motion
  reference, the control included, show him with his back to the camera
  about sixty frames in, where the source faces it.

## Not tried

A start for the body that is neither the source nor zero: the cue written
into the head's initial noise with no knot dropped, so the first step still
commits the reference's clothes; other shares and blurs; any clip but this
one. The owner moved the lane to a benchmark across clips the same evening,
so none of these was queued.
