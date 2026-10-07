# One window of a fourth clip: what each arm changed, and what the day did and did not explain (2026-10-06)

lane: masked
verdict: owner, on playback: the base chain with the part node's head and upper body as the region and the motion reference on is "solid" at both window lengths; every whole-subject arm on the base chain at the short window is broken; the fast chain is "pretty decent" on the whole subject and "pretty broken" on the recipe that works on the base chain. The mask, the day's code changes and the three caches are cleared as causes. Why the whole-subject arms fail is not explained.

**What this is.** The record of one working day on one window of
`thrill_2160.mkv`, written from the run folder of the session that rendered
it (mrteal_jr, closed) by a session that rendered none of it (mrop). The
numbers are in `2026-10-06_masked_v2v_body_window_arms.json`; the tables
below are copied from it by a script, and where a sentence and the file
could disagree the file is right. Two things in it were re-derived from the
files on disk when it was written, and are marked so; the rest is as the
runner's rows and the song node's reports hold it.

**What it leaves out, on purpose.** Nothing here says what the clip shows.
The clip is its file name and its technical facts, and an arm is what our
tools did to it. The typed prompts describe the still's person and are not
in this repo; the record names their files, which are under
`internal/claude/2026-10-06_mryellow_jr/thrill/prompts/`. Two sibling
records hold the day's other renders of this clip and are not repeated:
`2026-10-06_masked_v2v_4x3_canvas_pair.md` (thirty seconds on two canvases)
and `2026-10-06_masked_v2v_margin_and_words_variations.md` (eight short
renders, of which the body window's base arm is the control here).

## What was asked

The lane's fourth clip is 4:3 and cut into shots. The owner wanted to know,
by eye, what the masked graphs do with it: which chain, how wide a margin,
how much of the subject to replace, and how to say it in the prompt. The
arms were queued through the day as each verdict came back, so this is a
sequence of questions and not a designed grid.

## Setup, shared by every arm

`thrill_2160.mkv` from 67.0 s on the 4:3 canvas the owner chose, 1024x768
(the loader's width and the song node's width and height patched; the
json has the clip's facts), the lane's still, the graph's own seed, one window per arm,
the subject picked as the largest `person` on a named frame. Two loads of
the same start: a short one that is a single shot, for the 141-frame
window, and a longer one with three cuts in it, for the 345-frame window.
On the longer load the tracker at defaults leaves two of the later shots
without a subject, and two typed corrections read off its tile assign
them; a correction's numbers belong to one load. Two chains: `base` is the
ref2va motion graph at `h3_config.MASKED_MOTION_STEPS`, `fast` is the fl2va
graph with its PDD8 bake at `h3_config.PDD_STEPS`. The margin is the Masked
Source's `grow_pixels`. Three server processes served the day and each
arm's row says which; the first was armed with `H3_TELEMETRY` and nothing
else. One seed throughout.

The seconds are soft. Other sessions' sweeps and builds ran on the
processors beside several arms, and every arm after the first of a stretch
read encodes kept from an earlier prompt, which the json counts per arm.
The lane's timing record is `2026-10-06_masked_render_time_breakdown.md`.

## The arms

Each row is one render. "Its one change" is against the arm in the next
column; where an arm changed more than one thing the cell says both.

At 141 frames:

| arm | chain | region | margin | motion reference | prompt | its one change | control | tokens that regenerate | sampling, s |
|---|---|---|---|---|---|---|---|---|---|
| `choose_body_base` | base | whole subject | 64 | subject only | node, a noun | the base of the eight variations: margin 64, the subject a noun | (a control) | 31.8% | 98.6 |
| `diag_body_base_rerun` | base | whole subject | 64 | subject only | node, a noun | choose_body_base again on a later server and later code | `choose_body_base` | 31.8% | 98.5 |
| `diag_body_orig1344` | base | whole subject | 64 | subject only | node, a noun | the same window at the shipped 1344x768 canvas, the 4:3 source centre-cropped | `choose_body_base` | 31.1% | 136.7 |
| `body_p1` | base | whole subject | 64 | subject only | typed, `p1_described.txt` | a typed text in place of the prompt node's | `choose_body_base` | 31.8% | 98.1 |
| `body_p2` | base | whole subject | 64 | subject only | typed, `p2_described_rest_from_video.txt` | a typed text | `choose_body_base` | 31.8% | 99.0 |
| `body_p3` | base | whole subject | 64 | subject only | typed, `p3_partially_preserved_named.txt` | a typed text | `choose_body_base` | 31.8% | 98.8 |
| `body_p4` | base | whole subject | 64 | whole frame | typed, `p4_video_edit_whole_frame.txt` | a typed text, and the motion reference as the whole frame | `choose_body_base` | 31.8% | 98.9 |
| `body_p5` | base | whole subject | 64 | subject only | typed, `p5_fully_preserved_named.txt` | a typed text | `choose_body_base` | 31.8% | 99.5 |
| `body_p6` | base | whole subject | 64 | subject only | typed, `p6_named_whole_figure_proportions.txt` | a typed text | `choose_body_base` | 31.8% | 99.2 |
| `body_pdd8` | fast | whole subject | 64 | none | node, a noun | the fast chain as it ships: no motion reference | `choose_body_base` | 31.8% | 65.7 |
| `body_p7` | base | parts: hair, face and neck, upper clothing, hands | 64 | subject only | typed, `p7_upper_body_parts.txt` | the part node's hair, face and neck, upper clothing and hands as the region, and a typed text for it | `choose_body_base` | 28.6% | 99.0 |

At 345 frames:

| arm | chain | region | margin | motion reference | prompt | its one change | control | tokens that regenerate | sampling, s |
|---|---|---|---|---|---|---|---|---|---|
| `body_345` | base | whole subject | 64 | subject only | node, a noun | the base chain as it ships at the full window | (a control) | 35.7% | 314.9 |
| `body_345_p7` | base | parts: hair, face and neck, upper clothing, hands | 64 | subject only | typed, `p7_upper_body_parts.txt` | body_p7's recipe at the full window | `body_345` | 33.3% | 316.4 |
| `body_345_pdd8` | fast | whole subject | 64 | none | node, a noun | the fast chain as it ships | `body_345` | 35.7% | 193.9 |
| `body_345_pdd8_grow32` | fast | whole subject | 32 | none | node, a noun | margin 32 | `body_345_pdd8` | 26.6% | 189.7 |
| `body_345_pdd8_grow16` | fast | whole subject | 16 | none | node, a noun | margin 16 | `body_345_pdd8` | 22.1% | 194.1 |
| `body_345_pdd8_motion` | fast | whole subject | 64 | subject only | node, a noun | the motion reference on (subject only) | `body_345_pdd8` | 35.7% | 210.4 |
| `body_345_pdd8_parts` | fast | parts: hair, face and neck, upper clothing, hands | 64 | none | typed, `p7_fast_upper_body_parts_no_video.txt` | body_345_p7's parts on the fast chain, its text without the <Video 1> lines | `body_345_pdd8` | 33.3% | 190.8 |
| `body_345_pdd8_parts_motion` | fast | parts: hair, face and neck, upper clothing, hands | 64 | subject only | typed, `p7_upper_body_parts.txt` | the motion reference on, and the text with its <Video 1> lines | `body_345_pdd8_parts` | 33.3% | 210.0 |
| `ladder_a_head` | fast | parts: hair, face and neck | 64 | none | node, described | the region is hair, face and neck | `body_345_pdd8_parts` | 16.0% | 190.5 |
| `ladder_c_to_ankles` | fast | parts: hair, face and neck, upper clothing, lower clothing, hands | 64 | none | typed, `p8_body_to_ankles_parts.txt` | the region adds lower clothing | `body_345_pdd8_parts` | 35.6% | 191.1 |
| `ladder_d_every_class` | fast | parts: hair, face and neck, upper clothing, lower clothing, hands, mouth, every other class by name | 64 | none | node, described | the region is every class of the part model | `body_345_pdd8_parts` | 36.1% | 189.8 |

Two more 345-frame renders are controls and not arms: `body_345_pdd8_grow32`
again with nothing kept from an earlier prompt read, and again as the first
prompt on a restarted server with the stored mask not read either. They
are under "The three-way cache control".

## The owner's verdicts

From playback, relayed by the lane's lead the same day. Quoted where the
words judge the render; put in tool terms where the owner's words would
say what the clip shows. They outrank every reading of stills in this
record.

| arm | the owner |
|---|---|
| `choose_body_base` | "broken": the still's own framing is drawn into the region, a head at the size of a region that holds a whole figure in the source |
| `body_p1` to `body_p4`, with `choose_body_base` as the first row of one stack | "every single one ... is broken horribly" |
| `body_p5`, `body_p6` | not judged separately; rendered after that verdict |
| `body_pdd8` | "a pretty good test if its way faster"; the top is not the still's colour |
| `body_p7` | "solid"; the movement followed best of the short arms and the clothes are the still's |
| `body_345` | no verdict on quality: the owner asked why the figure is missing through most of the first shot |
| `body_345_pdd8`, with margins 32 and 16 beside it | "pretty decent for a pdd/distill"; the top's colour changed, the still's headwear is gone, and there is a flicker that follows the region's edge |
| `body_345_p7` | "yeah that one is solid" |
| `body_345_pdd8_motion` | a yellow top, and some detail of the source's clothing kept on the figure |
| `body_345_pdd8_parts` | "pretty broken": too thin, changing shape on every movement, the face compressed |
| `ladder_a_head`, `ladder_c_to_ankles`, `ladder_d_every_class`, `body_345_pdd8_parts_motion` | not watched when the session that rendered them closed |

## What was measured beside the renders

### The mask check

The owner asked that the mask be looked at and not only reused. A fresh
track, with the stored mask not read and the node cache defeated so the
tracker really ran, was saved as a lossless mask video and compared with
the stored mask frame by frame, on this window's short load and on the
thirty-second stretch the day's first render used. (measured) Both are
identical to the stored mask on every frame. On the short load the mask is
never empty and neither its area nor its centre jumps; on the stretch it
is empty only on frames of two shots that no correction assigns a subject,
and every large change in its area is at a cut. The figures are under
`mask_check` in the json. (seen) That the mask is on the subject and not a
neighbour is a reading of pictures: the lead read sheets of every eighth
frame of the stretch's second half; the rendering session read eighteen
frames of `body_345` over its region; the session writing this laid the
recipe arm's region and a whole-subject mask over every twelfth frame of
the 345-frame window. All found it on the subject. The stretch's first
half rests on the numbers.

### Where sampling's seconds go

| pair | sampling, s | evaluations | s per evaluation |
|---|---|---|---|
| `body_345` (base, reference subject only) | 314.9 | 12 | 26.24 |
| `body_345_pdd8` (fast, reference none) | 193.9 | 8 | 24.24 |
| `body_345_pdd8_motion` (fast, reference subject only) | 210.4 | 8 | 26.3 |
| `body_345_p7` (base, reference subject only) | 316.4 | 12 | 26.37 |
| `body_345_pdd8_parts_motion` (fast, reference subject only) | 210.0 | 8 | 26.25 |
| `body_345_pdd8_parts` (fast, reference none) | 190.8 | 8 | 23.85 |

(measured, one seed, soft seconds) An evaluation costs the same on both
chains when both carry the motion reference, on the whole subject and on
the parts. The fast chain is faster by its count of evaluations and by
shipping without the reference: turning the reference on costs it a
little more on every evaluation, on both regions. And the margin does not
move sampling at all: on the fast chain the margins sample in the same
seconds while the share of tokens that regenerate falls with the margin
(the arms table). The stock sampler runs every row either way, which
is the case for the frozen-row cache.

### Frame-to-frame change, by zone

The owner's "flicker that follows the region's edge", as a number: the
mean change in brightness between consecutive frames over a zone, in the
render less the same in the source, in 8-bit levels. The zones are the
mask, the margin around it, a thin ring on the region's outer edge, and
the rest of the frame. Cuts are left out.

| arm | chain | margin | mask | band | edge | plate |
|---|---|---|---|---|---|---|
| `body_345` | base | 64 | -0.21 | -0.58 | -0.47 | -0.10 |
| `body_345_p7` | base | 64 | -0.11 | +0.02 | -0.56 | -0.10 |
| `body_345_pdd8` | fast | 64 | +1.49 | +1.01 | +0.20 | -0.03 |
| `body_345_pdd8_grow32` | fast | 32 | +1.54 | +0.60 | +0.32 | -0.04 |
| `body_345_pdd8_grow16` | fast | 16 | +0.93 | +0.01 | +0.41 | -0.06 |

(measured) The fast chain's region is busier in time than the source,
most in the mask itself and next in the margin; the base chain's is not,
on the whole subject or on the parts. The ring on the edge carries less of
it than the inside does, and the excess does not gather at one place in
the VAE's four-frame cycle (the json has it by cycle position). (inferred)
So it is the fast chain's picture that is unsteady, not the composite's
seam. A narrower margin takes the excess out of the margin and leaves it
in the mask. These are means over a window; they do not say where on the
edge the eye catches it.

### The three-way cache control

The owner asked whether a cache was handing an arm something stale. One
arm, `body_345_pdd8_grow32`, was rendered three ways from one graph: as it
first ran, reading a conditioning and a source latent kept from an earlier
prompt; with `reuse_windows` off; and with `reuse_windows` and
`reuse_mask` off as the first prompt on a restarted server. (measured,
re-derived for this record) The stored window's video and audio latents
are equal value for value across the three, and the three joined videos
decode to the same bytes. The rendering session also found the three shot
tables equal on every key, and the cold prompt's report shows the tracker
and the source encode ran. That clears the window keep, the stored mask,
ComfyUI's own node cache and the frame-cap code the restart loaded, for
this arm. It also says the fast chain is reproducible bit for bit on this
server. The matched pair for the keep itself is
`2026-10-06_window_keep_matched_pair.json`.

(measured, re-derived) The day's code changes are cleared the same way for
one arm: `diag_body_base_rerun` is `choose_body_base` again on a later
server and later code, and the two decode to the same bytes.

### The parts ladder

A run of regions on the fast chain, from the head alone to every class the
part model has, each with the prompt node's role or a typed text to match
(rows `ladder_a_head`, `body_345_pdd8_parts`, `ladder_c_to_ankles`,
`ladder_d_every_class` above). (measured) The share of tokens that
regenerate grows from the head alone to every class, and sampling takes
the same seconds on every rung. Only `body_345_pdd8_parts` has a verdict.

### An island at the feet

| single-class pass | frames with a piece at the feet, of 345 | median pixels on those frames |
|---|---|---|
| face_and_neck | 290 | 341 |
| mouth | 0 | 0 |
| eyeglass | 0 | 0 |
| hair | 0 | 0 |
| hands | 9 | 5 |
| upper_clothing | 250 | 39.5 |

(measured, re-derived) The part model's `Face_Neck` class fires on a
small patch at the subject's feet on most frames of this window, so the
face tick puts an island into the region there. The mouth classes, the
glasses and the hair never do. (measured, new in this record) The upper
clothing class does it too, on nearly as many frames, with a much smaller
patch. The count is of pieces of a single-class pass's mask that do not
touch the main piece of the recipe arm's region and sit in the lowest
quarter of the subject's height; the json has the definition and the
files. A filter that drops a piece of the face classes not connected to
the head's was proposed by the rendering session and is not built.

## What it says

- (verdict) **One recipe works on this window, at both lengths.** The base
  chain, the motion reference on, the region cut by the part node to hair,
  face and neck, upper clothing and hands, the legs kept from the source,
  margin 64, and a typed text written for that region. A verdict at
  each length, one seed.
- (verdict) **Every whole-subject arm on the base chain at 141 frames
  that the owner watched is broken, and the same way**: the base and the
  typed prompts of one stack, one of which also showed the encoder the
  whole frame as its motion reference. (seen, not judged) The arm at the
  shipped canvas shows the same on the frames the lead read, and the two
  later typed prompts were not watched. Neither the margin nor more words
  about the still was judged apart from it (the variations record).
  (unexplained) Nothing rendered says why.
- (seen, not judged) **At 345 frames the whole-subject arm on the base
  chain fails differently.** On every twentieth frame of `body_345` the
  region holds no figure through most of the first shot and holds one in
  the later shots. (inferred) So the short window is not what breaks the
  whole-subject arms, and a longer one does not mend them.
- (inferred, confounded) What separates the working arms from the failing
  ones on the base chain is the region and the text written for it, which
  changed together in `body_p7`. No arm has one without the other.
- (inferred from the verdicts) **The fast chain does not predict the base
  chain.** On the whole subject it has the better verdict of the two
  here; on the recipe that is solid on the base chain it is broken.
- (verdict, seen) **A yellow top on the fast chain.** The whole-subject
  fast arms dress the figure in a yellow top where the still's is another
  colour. Two earlier records report a yellow garment from the same chain
  on another clip: `2026-10-04_masked_v2v_turn_soft_arms.md` and
  `2026-10-05_masked_v2v_per_token_arms.md`. Neither says which still it
  used, and this record did not establish that it was the same one.
  (unexplained)
- (seen on stills, one seed, and one verdict) **On the fast chain the
  motion reference brings the source's appearance with it.** A pair on
  each region differs only in the reference. On the whole subject the arm
  with it takes the colour of the source's lower clothing, and the owner saw detail of
  the source's clothing on the figure. On the parts the arm with it shows
  the source's own clothing in the region through the first part of the
  window and the still's after. On the base chain the recipe arm showed
  the still's top from the first sampled frame.
- (read from code) What the encoder is shown as that reference is the
  source's pixels under the Masked Source's region mask, widened by half of
  `grow_pixels` with a square kernel, on grey
  (`video_mask.py::motion_reference`, cut with the mask
  `video_mask.py::window_frames` returns); and it is shown every twelfth
  frame of the window (`reference_conditioning.py`, `sample_indices`). So
  on a parts arm the reference is the parts and nothing else of the
  subject.
- (inferred) The fast chain is the checkpoint that did not take the
  subject's turn from this reference on another clip
  (`2026-10-05_masked_v2v_motion_arms.md`). On that chain the reference
  has now been seen to carry appearance and has not been seen to carry
  movement.

## What the day ruled out

- **The mask.** Fresh equals stored on both loads, and it is on the
  subject on every picture read.
- **The day's code.** One arm rendered again on later code decodes to the
  same bytes.
- **The caches.** The three-way control above.
- **The composite's seam as the source of the flicker.** The excess is
  inside the mask on the fast chain and absent on the base chain through
  the same composite.
- **The margin as a lever on sampling time.** Sampling does not move
  with it.
- **Withdrawn the same day**, each by the session that said it: that the
  wider reference of a padded source is scaled down before the encoder
  (core's budget per pair of frames is far above either size,
  `comfy/text_encoders/minimax.py::process_video_block`); and readings of
  stills that the owner's verdicts on playback then contradicted. The
  day's rule from that: a still is reported as seen and never as a
  verdict on a clip.

## What it did not explain

- Why the whole-subject arms on the base chain draw the still's framing at
  141 frames, and why at 345 the region holds no figure through most of
  the first shot.
- Why the recipe that is solid on the base chain is broken on the fast
  chain.
- The yellow top.
- Why the padded source fails; that is the canvas pair's record.

## Not tried

A second seed, for any arm. The base chain on this window with the motion
reference off, the recipe's region with the prompt node's text in place
of the typed one, and a typed text with one block per shot: all three were
queued when the rendering session closed and had not rendered. **Note,
2026-10-07:** the second of those three did render that afternoon, by
another session after this record's rows were taken: `body_345_p7_node`, the
recipe arm's graph and patches with the prompt node wired, its role set to
the head and upper body and the subject described. It and the typed arm are
stacked on the output share
(`Video/compare_stacked/body_345_p7_typed_beside_node_over_mask.mp4`). Seen
on stills by that session, not judged; its row is in that session's
untracked run folder, so no figure of it is in this record's json.
(verdict) The owner watched the pair on playback on 2026-10-07: the
node-text arm is good, and they could not tell it from the typed arm. A motion
reference with no appearance in it, which is queued as one arm on the
recipe. Any window of this clip but this one and the thirty-second stretch.
