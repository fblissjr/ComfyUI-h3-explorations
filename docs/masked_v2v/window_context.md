# What ties a window to the window before it, and what loosens it

**A continuation window follows the frames it inherits more than it follows its motion reference.** The kept frames
are conditioning rows, as clean and as trusted as a reference, and core gives no way to set their level per
reference. The lever that loosens them without patching core is `context_noise`: at 0.8 on one stretch the head made
the turn with almost no seam (`h3_config.MASKED_JOB_CONTEXT_NOISE`, a starting value from that stretch); at 0.5 it did
not; at 1.0 it turned and broke the seam. A short load's held tail is the same mechanism after the new frames, and is
always held with its region open. The lane is [`overview.md`](overview.md); the signals a window can be shown are
[`state_signals.md`](state_signals.md).

## The mechanism

Read from code on 2026-10-10; nothing run.


- The window node copies the previous window's SAMPLED video latent tail
  into this window's first latent steps and gives those rows a mask of
  `audio_freeze.CONTEXT_NOISE` (`audio_freeze.py`,
  `MiniMaxH3FreezeAudioWindow.execute`). With a source wired the song node
  takes the minimum with the region's tokens, so the plate's rows are 0.
- To core a row at mask 0 is conditioning inside the picture. Every step
  the sampler feeds it the clean latent with core's small fixed noise
  (`comfy/model_base.py`, `MiniMaxH3.scale_latent_inpaint`), the model
  labels it at the conditioning timestep (`comfy/ldm/minimax/model.py`,
  `_forward`: `t_pin_v`, `rows_t`), and its velocity is multiplied by the
  mask. It never moves and is never noised.
- A reference's rows (a still, a reference video's own copy) get the same
  label by another path (`_cond_video_rows`, `seg_t`). Their level is one
  scalar in the payload for value and label together, the same for every
  reference: a still's rows are not told from a video's (`"ref_img"` is
  both). `reference_noise.py` sets that scalar. It cannot reach a masked
  row: both places a masked row is treated read core's module constant.
- So the level of the references cannot be set per reference without a
  patch to core, and the context's level could not be set at all until
  `context_noise`.
- Where the rows sit: the context at the head of the target's own time
  grid, next to the frames being drawn; a reference block ahead of the
  target on the time axis (`_ref_t_span`).
- How many: a context of `context_frames` is a fixed count of latent steps
  at the canvas's full size, whatever the window's length, so it is a
  larger share of a short window than of a long one; a motion video's rows
  follow `motion_short_edge`. Rows are not attention.
  `bench/map_attention_mass_on_capture.py` measures mass by segment on a
  capture, and today refuses a capture with reference rows.

## The levers

Kind: INPUT (settable today), CONSTANT (in code), CODE (a change in this
pack). None needs core patched.

| lever | where | kind | status |
|---|---|---|---|
| fewer context frames | `context_frames` on the song node; `loop_plan.check_window_settings` lists the values, `audio_freeze.window_geometry` refuses fewer | INPUT | tried 2026-10-10 |
| no context | a load of its own; `continue_from` unwired | INPUT | 2026-10-10: see results |
| the context shown part noised | `context_noise` on the song node and the window node (0.271.0); core runs a mask value between 0 and 1 as that row's own strength | INPUT since 0.271.0 | 0.8 works on one stretch, 0.5 does not, 1.0 breaks the seam; per load on a person's word (results below) |
| the context dropped for part of the schedule | the model patcher's `denoise_mask_function` for the value and a diffusion-model wrapper for the label, both together | CODE | not built; only if the noised context helps |
| the motion video over the context's own frames | a prepared video on `motion_video` | INPUT | not run |
| a larger motion video | `motion_short_edge`, `motion_vae` | INPUT | tried 2026-10-10 |
| a longer window | `window_frames` and where the plan puts a boundary | INPUT | not run |
| the seed | `seed`; each window takes seed plus its index | INPUT | tried 2026-10-10 |
| a text for the window | the timeline and its blocks; one text a load with a load per shot | INPUT | tried 2026-10-10 |
| the attention kernel | the graph | INPUT | not run; only if nothing else moves it |

Not levers here: `reference_noise.py` (it weakens references, the still
with them, and cannot reach the context); the frozen video cache (no
masked graph wires it); a late start (it is about the regenerated rows'
start, not the context).

## What a weaker context costs

The frames the previous window already wrote are not redrawn. With
`context_noise` above 0 a window redraws its own copy of the context and
continues from that, so its first new frame can differ from the last frame
written before it: a step at the seam. Every arm is read for that step as
well as for what it was run to move.

## The held tail of a short load: the same mechanism, after the new frames

**A window past its load's last frame holds that frame with its mask, so the region stays open over the held tail**
(`video_mask.held_mask`; the owner asked for one behaviour, 2026-10-10, [`docs/state/decisions.md`](../state/decisions.md)).
Held as plate, the tail showed the model the source's last frame with the ORIGINAL in it, a long run of clean
conditioning rows after the new frames, and the render faded back to the original.

- **What was seen with the plate tail**: a 29-frame load capped at its shot's end drew the still's person on its first
  frames and the original on its last. It is a FADE, not a flip: on the hair the render moves toward the source's tone
  on every frame and is the source's on the last (the capture `data/2026-10-10_fun_kitchen_shot2_1010`). The stored
  latent, decoded and laid again, says the fade is the SAMPLER's, not the composite giving the source back
  (`bench/results/2026-10-10_saved_windows_laid_again.md`, the short load). The tail's reach is more than two cycles of
  latent steps back from the first held frame.
- **What the open tail did**: on all three loads that had ended on the source the fall at the end is gone, the back
  half of the two face loads is lifted as well, and nothing reads worse (`data/2026-10-10_fun_face_street_0438_tail`,
  `data/2026-10-10_fun_kitchen_shot2_tails_1010`; one seed each). In the joined file the street face then read as
  swapped on the owner's playback. One clip; a second has not been run.
- **On the composite's side**: a held frame with no mask of its own is not laid (`video_mask.unlaid_frames`, 8952c545).
- **The tail is short now**: a load one window holds is given the shortest length of the video grid that holds it
  (`masked_job_graph.window_for`, the owner, 2026-10-13), so its held tail is under one video step. The audio clock is
  a rule of the seam between two windows, and a window that is the whole run has none (`loop_plan.stands_alone`).
- **A short load's last step is part held**: a load that starts on a cut ends on a step's last frame only when its
  length leaves a remainder that is one of the cycle's step starts (`loop_plan.step_span`, `loop_plan.STEP_CYCLE`), so
  most short loads end inside a step that holds real and held frames. The open tail carries the last frame's mask into
  it.
- **A small subject did not follow her motion video on that load** with or without the zoomed motion video; one change
  of the reference still made it take the pose. A still that shows another pose is a pull of the same kind as a kept
  frame, and is not among the levers below (one load; `data/2026-10-10_fun_pose/`).
- **Open**: the tail's pull against the subject's share of the frame. Two short loads with a large subject and a longer
  tail did not fade; the one that faded had her small and partly hidden. Not separated.
- **Not built, if the open tail leaves a doubt**: fill the tail with the shot played back in time, frames, mask and
  motion video mirrored past the last frame, so the window asks one thing from first frame to last. Reasoned only.
- **Not scoped**: a window shorter than the planner's shortest. The vendor's README gives the output duration's lower
  end as four seconds (`coderef/MiniMax-H3/README.md`, "Output duration"), and core's `length` tooltip gives the trained
  range (`comfy_extras/nodes_minimax_h3.py`; [`docs/h3/geometry.md`](../h3/geometry.md)); the
  lengths below the planner's are a little or far under it.

## Results, dated

- 2026-10-08, one clip (the postmortem under `docs/research/postmortems/`):
  a subject stood still in a render's last windows; a window that began on
  frames of the new subject held him where he was small. Nobody isolated
  the context's pull that day.
- 2026-10-10, one stretch of one clip, two renders (the lead's pose table):
  the rendered head stays with the frozen context on a continuation's own
  new frames while the source's head and the motion video's turn.
- 2026-10-10, the same stretch, one render each, read from stills and not
  on playback (the lead): the stretch as a load of its own, with no kept
  frames, the same motion video, text and seed, made the turn; the same
  stretch as a continuation behind its kept context did not. So the motion
  video carried the movement and the kept frames outweighed it. The load of
  its own is not a fix there: the source has no cut at that frame, and the
  load drew the subject's clothing differently from the pass it would join.
- 2026-10-10, the same stretch measured (a peer's pose records under
  `data/2026-10-10_fun_pose/`; `docs/masked_v2v/state_signals.md` has the
  figures): the load of its own is within a few degrees of the source's
  head through the turn and after it, and its chin and wrist read as the
  source's; every render that carries kept frames reads tens of degrees
  off, with the chin up and the hand away, and reads the same to a degree
  whether the motion video was shown at the larger or the smaller short
  edge and under either attention sink. So on that stretch the motion
  video's size changed nothing, and the kept frames decide it.
- 2026-10-10, `context_noise` 1.0 on that continuation, its own load, seed
  and attention sink, one render, read from stills and not on playback (the
  lead) with the seam measured off the card (a peer): the head turns to the
  camera, the chin lifts and the hand comes to the face, as the source and
  the motion video do, where the clean-context twin stays turned away
  throughout; the subject is still the pass's person in the pass's
  clothing. The price is at the seam: the first new frame steps away from
  the last frame the pass wrote about four times as far as the source
  itself moves there, and the window's own copy of the frame before the
  seam is as far again from the pass's, where the clean-context
  continuations keep both at the source's own step (the figures are that
  peer's to record). After the seam it moves as the source does. So the
  hold is in the context's subject rows, which is what this lever reaches.
- 2026-10-10, `context_noise` 1.0, 0.8 and none against a load with no kept
  frames, measured on that stretch (a peer's pose records,
  `data/2026-10-10_fun_pose/kitchen_cn_*`; one stretch, one seed each; the
  figures are that session's to record). Both noised continuations make
  the turn, the lowered head and the hand at the face; the clean one makes
  none of them. 0.8 is better than 1.0 on every column (head, chin, wrist,
  how far the movement follows and how nearly in step) and has almost no
  seam: its step onto the first new frame is a little over the source's
  own there, where 1.0's is several times it, and its copy of the frame
  before the join is as close to the pass's as the source's own step. The
  load with no kept frames is still the closest to the source. Both noised
  ones are a little LATE: the movement fits best a few frames behind, more
  at 1.0 than at 0.8. So at 0.8 the kept frames still tie the first new
  frame to what came before and no longer pin the pose.
- 2026-10-10, `context_noise` 0.5 on the same continuation (the lead's
  relay of a peer's reading; one stretch, one seed): it did not make the
  turn. So on this stretch the level that loosens the pose lies between
  0.5 and 0.8, and 0.8 is the lowest tried that works. It is a starting
  value from one stretch of one clip (`h3_config.MASKED_JOB_CONTEXT_NOISE`),
  not a default measured for another clip.
- 2026-10-10, the lateness, two READINGS and no measurement. It is not the
  clean share of the context pulling the first frames back, because it is
  larger with none of that share (1.0) than with some (0.8). What the
  mechanism does offer: (a) the movement begins a frame after the join,
  inside the first new latent step, and one token then has to hold the old
  pose and the start of the turn, so the turn can only begin at the step's
  grain; (b) at 1.0 the context is redrawn whole, the model follows the
  motion video through it on its own clock, and nothing ties that clock to
  the pass's frames at the join, where at 0.8 something does. The test that
  separates them: the same continuation with its join moved by one latent
  step. Not run.
- 2026-10-10, a level that ramps across the kept frames: possible in core's
  terms, not with today's input. Core takes a mask value per token row, so
  each of the context's latent steps can carry its own; `context_noise`
  writes one value on all of them (`audio_freeze.py`, the one line). A ramp
  is that line writing a vector and a second number on the node. The
  direction the results point to: clean at the join, where the seam is
  made (1.0's price was the frame before the join being redrawn), and
  noisier far from it, where the rows only add to the count showing the old
  pose. That is a softer form of a shorter context, and the clean
  short-context arm on the card is its limit. NOT BUILT.
- 2026-10-10: the same morning's two per-shot loads kept the still's person
  with no frozen context (two shots, one seed;
  `bench/results/2026-10-10_one_load_per_shot_against_one_long_load.md`).
- 2026-10-15, a ruling and no measurement: a load longer than one window
  may take `context_noise` for its inner windows, per load, on a person's
  word (`bench/shot_to_load.py job --context-noise X --context-noise-why
  WHY`; the lead's ruling, A (a)). Until then only a continuation
  (`continue_from`) could set it, though the song node shows an inner window
  its kept frames the same way. Without it the inner windows see them
  clean, the node's own. Not a default yet: 0.8 rests on the one stretch
  above, and its cost is a seam that may show. The kitchen's second subject
  over 604-1009 (seam 898, context 90, 0.8) is the second data point; if its
  seam holds on the owner's playback, it becomes the default for every
  window that keeps frames, with that render as the reason.
