# A clean plate from VOID on the band clip's turn window: a first probe (2026-10-05)

lane: masking
verdict: tested and removed at the owner's word: "Not a single clip showed any improvement." No arm gave a plate on this clip

**The clips this record names under `Video/mrorange/` were deleted by the
owner on 2026-10-05, after the verdict; the frame readings and the verdict
below are what remains of them.** Session mrorange, the board's `build-void-plate`. **Status: closed by the
owner on 2026-10-05 and parked; see "Where it ended" first.** The first four
clips were judged one by one (below). What is described under "What the frames
show" was read by me from still frames at VOID's size, three per shot and a
ten-frame sheet, before the judgment; **where it and the owner's judgment
differ, the judgment is right**, and for shot 1 they differ. One clip, one
window, one seed, two prompts.

## Judged by the owner, 2026-10-05

On playback, in the owner's words, as relayed by mryellow and as the board's
`build-void-plate` card quotes them:

- the turn shot, empty prompt: "some ghosting at the start and end and
  shadows are still there but gone... but you can see glitchy at the very
  last frames";
- the turn shot, scene prompt: "looks horrible. deformed / morphing
  faces/bodies everywhere";
- the whole window, scene prompt: "has people disappearing and appearing and
  morphing";
- the whole window, empty prompt: "does too but not as much but... still
  really bad".

corrected: my reading of shot 1 below, from three stills, was that the plate
looks plausible there. On playback it is not: people disappear, appear and
morph over time, which a still cannot show. Stills were the wrong instrument
for a plate that has to hold over time.

mryellow, from contact sheets of every sixth frame, agrees, and adds for the
empty arm: a flat translucent panel exactly the size of the subject's cell
region with the neon tube drawn across it, the man behind him smeared beside
the woman on the right, grey streaks in the last frames, and **the cast
shadow outside the one-cell grey ring**. So the affected region did not hold
the shadow: one cell beyond the cells he touches covers the lane's hole and
not what he casts on the wall. The ring's width is `GRID_GROW` in the probe.

The numbers are in `2026-10-05_void_plate_turn.json`. The script and its
graph (`void_plate_turn.py`) are not
tracked: the graph was built by hand from core's nodes and is in no
generator.

**Later the same day the code built for this was removed, at the owner's
word**: "Park VOID entirely, lets remove the code we built from it - it was built for cogvideox and its ghosting and not worth keeping or preserving. Note we tested it and thats it. No current state docs or workflows with it please." `void_conditioning.py`
(`MiniMaxH3VoidConditioning`), `bench/convert_void_checkpoint.py`,
`bench/check_void_conversion.py` and `bench/check_void_conditioning.py` were
last present at commit df3bd21b. Where this record says they stay, they do
not. The board's cards are `ruled_out`. This record is what is kept.

## Where it ended

**The owner's verdict on every VOID arm rendered today, in their words, as
relayed by mryellow: "Not a single clip showed any improvement."** That
includes the arms in which the panel no longer appears on the strips; the
readings from strips further down are what I saw, and they stand under that
judgment, not beside it.

The owner then closed the lane. Their reason, as relayed and not in their
words: VOID is trained on CogVideoX, so this may be a waste of time; and the
clips they saw were not doing anything useful. The last pair of arms was
stopped on that word while its second arm was sampling. **Parked, not ruled
out**: a clean plate under the hole is still something the lane wants; this
tool, on this base model, through core's port, is set aside. No pass 2, no
run of upstream's own code on the same input, no blended windows.

What the day established about it, for whoever picks it up:

| difference between core's port and upstream's inference | tested | what the arms showed (strips, mine) |
|---|---|---|
| 1. the pixel range the mask is encoded at | yes, twice: the quadmask halved through core's node, and the pack's node | the flat panel in the shape of the affected region goes away |
| 2. what the model is shown under the mask | yes: black (core), half brightness, whole, and upstream's other path, black with upstream's mask | none of them is clean; the more of him is shown, the more of him stays |
| 3. guidance | could not be tested with empty prompts: the two predictions are one tensor | at upstream's value of 1 there is no guidance at all |
| 4. the window length | yes: 85 frames, a short shot padded by ping-pong | the ghost is still there, darker on the turn shot |

**The trend across what the model sees of him** is the one regularity in the
arms: blacked out under core's mask range, no ghost and a panel; at half
brightness, some ghost; shown whole, more ghost, on both window lengths;
blacked out with upstream's mask range, the first half of the turn shot is
about as clean as the best arm and residue returns in the second half (one
arm, the room-only prompt).

**What no arm showed** is the model making a plate of this clip. Every arm was
a port. Whether the port or the model is at fault here is exactly what a run
of upstream's own code on the same input would say, and that run was not
made.

What stays in the tree: `MiniMaxH3VoidConditioning` and its check, which hold
the two conditioning differences whatever becomes of the lane;
`bench/convert_void_checkpoint.py` and its check; this record. The candidate
defect in core (difference 1, which reaches core's own template) is for the
owner to report or not.

All clips, under `Video/mrorange/` on the output share, each also as
`_turnshot`: `void_plate_turn_empty`, `_scene` (first round, core's node);
`_empty_cfg1_up`, `_empty_cfg1` (second round, core's node); `_pack_empty`,
`_pack_scene`, `_pack_room` (third round, the pack's node); `_native_room`,
`_native_empty` (fourth round, 85-frame windows); `_black_room` (the last
pair's finished arm).

## For the owner to look at

On the output share under `Video/mrorange/`, each the source, the plate and
the quadmask side by side at 672x384:

- `void_plate_turn_empty_turnshot.mp4` and
  `void_plate_turn_scene_turnshot.mp4`: the shot with the turn in it, frames
  237 to 279 of the window.
- `void_plate_turn_empty.mp4` and `void_plate_turn_scene.mp4`: the whole
  window. The plate is VOID's where the lead is on screen and the source
  elsewhere.

## What ran

The window `turn_control` loads: `thinkaboutthings_compressed.mp4` from 112 s,
345 frames at 24 fps, loaded at width 1344.

- **Who to remove**: `MiniMaxH3SubjectTrack`, everything automatic. Four
  shots, cuts at 186, 237 and 280. The lead is taken in shot 1 (frames 0 to
  185) and shot 3 (237 to 279, where the group turns its back); shots 2 and 4
  are cutaways without him. The node's report is in the record.
- **The model**: core's VOID nodes, pass 1 only, at the settings of core's own
  template (`utility_void_video_inpainting.json`): 672x384, cfg 6, 30
  `simple` steps, `VOIDSampler`, the T5 encoder and the CogVideoX VAE. Pass 2
  and the optical-flow file were left out
  (the masking note of 2026-10-04, "VOID for the plate"). The
  checkpoint is upstream's, converted:
  `2026-10-05_void_checkpoint_conversion.md`.
- **The quadmask**, built from core's mask nodes at VOID's size so its levels
  are exact. The subject's mask, any pixel it touches, grown by 6 pixels, is
  the object to remove. The 32-pixel grid cells the subject touches, grown by
  one cell, are the affected region. One cell at this size is the lane's
  `grow_pixels` at 1344 wide (`h3_config.MASKED_SOURCE`), so the affected
  region covers the hole the lane cuts. Core's template wires a plain mask
  straight into `VOIDInpaintConditioning`, so 1.0 there is "remove"; the
  affected level is the one `VOIDQuadmaskPreprocess` produces.
- **The windows**: one per shot the lead is in. VOID takes lengths of 8k + 5
  frames (`comfy_extras/nodes_void.py::_valid_void_length`), so each window is
  the shot rounded up and runs past its cut: frames 0 to 188, **dropping 186,
  187 and 188**, and frames 237 to 281, **dropping 280 and 281**. Neither shot
  needed splitting.
- **Two arms on seed 43**: an empty prompt, and one sentence describing what
  remains: "A small group of young people in matching green sweatshirts stand
  and dance in a living room with framed pictures on the wall and table
  lamps, under coloured light."

## What the frames show

**Shot 1, 189 frames** (stills; see the correction above). The lead is gone in both arms. The wall, the pictures
and the neon tube behind him are filled in, and the people he stood in front
of are completed where he covered them. The two arms give nearly the same
plate. Around frame 170 both arms appear to put a second copy of the woman
with glasses where he stood, and the man who stood behind him is not there.
If that reading holds on playback, the plate under the hole holds an invented
person in that part of the shot.

**The turn shot, 45 frames.** Not clean in either arm.

- Empty prompt: where he stood there is a tall, door-like panel on the wall
  that the room does not have, the neon tube is drawn through it, and there
  are smeared dark figures beside the woman on the right.
- The scene sentence: two new people and a brown sofa are painted into the
  hole.

So on this shot the prompt decides what fills the hole, and a sentence that
mentions people puts people there. On shot 1 it barely matters. In the
record, `arms_compared`: the two arms differ inside the quadmask's region
about two and a half times as much on the turn shot as on shot 1.

**The quadmask** is as designed on the frames looked at: his silhouette, and
a blocky ring one cell wide. A stray speck of tracker mask at frame 237
became a small square of "remove".

## Three things about core's VOID nodes, found on the way

1. **The output is three frames longer than the input, and shifted.** Each
   window came back with three frames more than it was given. Matching the
   kept pixels of the source against the output at each offset puts input
   frame j at output frame j + 3 on both windows (`output_offset` in the
   record). The assembly here applies that. Core's template does not, so its
   pass 1 clip is three frames late against its source.
2. **The kept area is not the source's pixels.** VOID decodes the whole frame,
   and outside the quadmask its output differs from the source by a few
   levels (`arms_compared`). A plate has to be composited under the hole; it
   cannot replace the frame.
3. **Core cannot load the upstream checkpoint.** See the conversion record.

## The quadmask's polarity, read

Asked by mryellow after the judgment: a flat panel in the shape of the
subject's region is what an inverted or self-repainting mask would give. Read
in core and in upstream's own inference code (`netflix/void-model`, `main` at
e3914f8f551d, fetched 2026-10-05; local copies, untracked). **The polarity is
right.**

- Upstream (`get_video_mask_input`, in its own utils module): the quadmask
  on disk is 0 on the object, 63 on the overlap, 127 on the affected area and
  255 where the frame is kept. It is quantised to those four values and then
  inverted, `255 - mask`, and divided by 255. So the mask the pipeline
  receives is 1.0 on the object, about 0.5 on the affected area and 0.0 on
  what is kept.
- Core's `VOIDQuadmaskPreprocess` is the same quantisation and inversion, and
  its docstring says 1.0 is "remove". Core's template skips that node and
  wires a plain mask, 1 on the subject, into the conditioning node.
- The probe's quadmask is 1.0 on the object, 0.5 on the ring and 0.0
  elsewhere: the same convention.
- Both then encode `1 - mask` with the VAE as the mask channels (core:
  `VOIDInpaintConditioning`, `vae.encode(inverted_mask_3ch)`; upstream:
  `1 - mask_condition_tile` with `use_vae_mask`), which is the paper's
  picture: black object, grey affected area, white kept.
- seen, and independent of the reading: in shot 1, on the same nodes, the lead
  is gone and the wall behind him is drawn. An inverted mask would have kept
  him.

## Where core's port differs from upstream's own inference

Found by the same reading, in two passes. Four differences; the first was
found last and is the one the arms below test.

1. **The pixel range the mask is encoded at.** Both sides hand `1 - mask` to
   the VAE as the mask channels. Upstream's mask processor does not normalise
   (`VaeImageProcessor(do_normalize=False)`), so its VAE, whose pixel
   convention is -1 to 1, sees the mask in 0 to 1. Core's node calls
   `VAE.encode`, which scales every input by `x * 2 - 1` for this VAE
   (`comfy/sd.py`, `process_input`; checked on the loaded VAE). So:

   | quadmask value | upstream's VAE sees | core's VAE sees |
   |---|---|---|
   | object, 1.0 | 0.0 | -1.0 |
   | affected, about 0.5 | about 0.5 | about 0.0 |
   | kept, 0.0 | 1.0 | 1.0 |

   In core, the affected ring is encoded at the value upstream uses for the
   object to remove, and the object at a value upstream never produces. For
   the probe that means the whole blocky cell region was marked as an object,
   which fits a flat panel of exactly that shape.

   **This is a candidate defect in core, and it reaches core's own template**:
   a plain subject mask wired into the conditioning node, as the template
   does, is encoded at -1 where upstream uses 0. Not established here:
   nobody has compared core's output with upstream's on the same input.
   Whether to report it upstream is the owner's call.

   It can be tested through core's node with no new code: halve the quadmask
   before the node. Core then encodes `1 - m/2`, which after its scaling is
   0.0 on the object, 0.5 on the ring and 1.0 on what is kept: upstream's
   mask channels.
2. **What the model is shown under the mask.** Upstream's shipped config sets
   `zero_out_mask_region = False`, and the pipeline then conditions on the
   whole, unmasked video (`masked_video = init_video`). Core's node always
   multiplies the video by `1 - mask`: the object is blacked out and the
   affected ring is shown at half brightness. Even upstream's other path
   (`zero_out_mask_region = True`) blacks only the object and leaves the
   affected area as it is. With the quadmask halved as in 1, core shows the
   object at half brightness and the ring at three quarters, which is nearer
   upstream's and still not it.
3. **Guidance.** Upstream's config has `guidance_scale = 1.0`, with a negative
   prompt about quality; core's template uses cfg 6 with an empty negative.
   Both use 30 steps (upstream's script passes 30 and ignores its config's
   50).
4. **Windows.** Upstream's script calls the pipeline with `num_frames` set to
   its `temporal_window_size`, 85, and a multidiffusion stride in the config.
   Core's node samples the whole window at once; the probe gave it 189 and 45
   frames. How upstream stitches its windows was not read.

Also read, with nothing following for the probe: upstream quantises the mask
to three levels inside the pipeline whatever the config says
(`use_trimask = True` is passed literally), so its overlap level becomes
"object". The probe has no overlap level.

Showing the model the whole unmasked video (2) is not reachable through
core's node. Nothing in this pack patches core, so it is a conditioning node
here: `MiniMaxH3VoidConditioning` (`void_conditioning.py`), which does 1 and 2
as upstream does and is held to upstream's arithmetic, with core's node as
the control, by `bench/check_void_conditioning.py`. It has not rendered yet.

## Second round: the mask's range, and guidance

Two more arms on the empty prompt, seed 43, the same windows, through core's
node (`second_round` in the record; clips beside the others, named
`void_plate_turn_empty_cfg1_up` and `void_plate_turn_empty_cfg1`). **Not
judged.** Read by me from strips of eight frames a shot
(`void_AB_*_strip.jpg`, untracked), which the first
round showed is not enough to judge a plate.

- **B, the quadmask halved, cfg 1.** Core's VAE is then given upstream's mask
  levels. On the turn shot the panel is gone: the wall where he stood is
  drawn through with its pictures and the neon tube. From about the middle
  of the shot a dark teal smear of him stays where he stands. On shot 1 a
  smeared figure stays where he stood in roughly the second quarter of the
  shot, where the first arm's stills showed none.
- **A, the quadmask as before, cfg 1.** The first empty arm again: the panel
  is there, and the two plates differ by about one level after encoding
  (`plates_compared`). **This says nothing about guidance.** With an empty
  positive and an empty negative prompt the two predictions are the same
  tensor, so cfg cannot change the result; the arm should not have been
  asked for. For the same reason arm C (cfg 6, mask halved) would have been B
  again and was removed from the queue unrun. Guidance matters only once the
  two prompts differ, as upstream's do.

So of the four differences the arms have tested one: **the mask's pixel range
is what draws the panel.** What B leaves, a ghost of him, has one candidate
that the reading offers and no arm has tested: with the mask halved core
still shows the model the object at half brightness, where upstream shows it
whole. The node above removes that difference.


## Third round: the pack's node, upstream's mask and the whole video

Three arms through `MiniMaxH3VoidConditioning`, cfg 1 as upstream's config
has it, seed 43, the same two windows (`third_round` in the record; clips
`void_plate_turn_pack_empty`, `_pack_scene`, `_pack_room`, each with its
`_turnshot`). **Not judged.** Read by me from strips of eight frames a shot.

- **The panel is gone in all three**, as in B.
- **The empty prompt**: a dark teal ghost where he stands through the whole
  turn shot, more of it than in B. So the dimmed video was not what left the
  ghost in B: shown whole, more of him stays.
- **The scene sentence**: a smaller figure painted where he stood through the
  turn shot. A person again.
- **A room-only sentence** ("A living room with framed pictures on the wall,
  table lamps and a neon tube, under coloured light."): the cleanest first
  half of the turn shot of any arm, the wall drawn through with only a faint
  smear; the ghost returns in the second half, when he has turned and fills
  more of the frame.
- **Shot 1**: the three arms alike, and like B: clean at both ends, a mottled
  figure where he stood through roughly frames 50 to 105.

So with the mask and the video as upstream has them, the plate is still not
clean on this clip. What remains of the four differences is the window
(difference 4), and it was under-read above: upstream's script pads any clip
shorter than its window of 85 frames to that length by appending the clip
reversed, and samples a longer one in windows of 85 blended at every step. Its
model is therefore always given exactly 85 frames; core was given 189 and 45
here. Both failures sit where that would matter: the short shot, and the
middle of the long one.

## Fourth round: upstream's window length

One window of 85 frames per shot through the pack's node, the whole video,
cfg 1, seed 43, the room-only sentence and the empty prompt (`fourth_round`
in the record). The turn shot is 43 frames, so it was padded as upstream's
`temporal_padding` does it: the clip, then the clip reversed, cut to 85. Only
its forward half is read; the reversed tail is a motion the clip never had.
Shot 1 was cut to the 85 frames round its middle, 50 to 134, which holds the
stretch where the long window left a figure. **This tests the length. It is
not a way to cover a long shot**, which would need upstream's windows blended
at every step.

Read by me from strips: the ghost is still there, in both shots and with both
prompts. On the turn shot it is darker and more solid than with the 45-frame
window. On the middle of shot 1 it is present through the whole window, where
the 189-frame window was clean again from about frame 110. So the length is
not the cause.

## The last pair: the object blacked out, upstream's mask

The pack's node with `the object blacked out`, which is upstream's other path
(`zero_out_mask_region = True`): the mask at upstream's range, nothing of him
shown, the affected area left as it is. The room-only arm finished; on the
strip its turn shot is like the third round's room-only arm in the first
half and has residue in the second. The empty-prompt arm was stopped while
sampling, on the owner's word, and has no output (`last_pair`).

## Time

Sampling took about four minutes for the 189-frame window and about forty
seconds for the 45-frame one, the same in both arms, on the 4090
(`per_node_s`). A whole arm, with its loads and decodes, was between five and
six minutes.

## What this does not say

- Whether the plate is good enough under the hole in shot 1: nobody has
  watched it play, and the hole is about a fifth of the frame.
- Anything about removing a shadow. corrected: this line first said the lead
  casts no clear shadow at this size. He does in the turn shot, on the wall,
  and it lies outside the ring (the owner's judgment above), so the probe
  never asked VOID to remove it.
- Whether the turn shot fails because it is short, because the subject is
  close and moving fast, because of the prompt, or because of how core's
  node conditions (the section above). The paper names close subjects as a
  weak spot. Untried: upstream's guidance; a ring wide enough to hold the
  shadow; a sentence that describes the room and no people; a window that
  carries the turn shot together with more context; pass 2; a second seed.
- Whether a window that runs past its cut harms the shot's last frames. The
  sheet's last two columns for each window are the shot's last frames and
  show nothing the earlier ones do not; that is five frames, by eye.
- Anything at the lane's size. The plate is 672x384 and would be enlarged
  twofold to sit under a hole at 1344 wide.
