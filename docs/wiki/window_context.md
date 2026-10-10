# What ties a window to the window before it, and what loosens it

A ledger for one question in the masked lane: a continuation window follows
the frames it inherits more than it follows its motion reference. The
mechanism is read from code and cited; every result is a dated line with
its scope, and a change is a new dated line under the old one, never a
rewrite. `docs/wiki/masked_v2v.md` has the lane; `docs/wiki/state_signals.md`
has the signals a window can be shown.

## The mechanism (read from code 2026-10-10, nothing run)

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
| fewer context frames | `context_frames` on the song node; `loop_plan.check_window_settings` lists the values, `audio_freeze.window_geometry` refuses fewer | INPUT | on the card 2026-10-10 |
| no context | a load of its own; `continue_from` unwired | INPUT | 2026-10-10: see results |
| the context shown part noised | `context_noise` on the song node and the window node (0.271.0); core runs a mask value between 0 and 1 as that row's own strength | INPUT since 0.271.0 | 2026-10-10: at 1.0 it turned the head and broke the seam (results below); the levels between are on the card |
| the context dropped for part of the schedule | the model patcher's `denoise_mask_function` for the value and a diffusion-model wrapper for the label, both together | CODE | not built; only if the noised context helps |
| the motion video over the context's own frames | a prepared video on `motion_video` | INPUT | not run |
| a larger motion video | `motion_short_edge`, `motion_vae` | INPUT | on the card 2026-10-10 |
| a longer window | `window_frames` and where the plan puts a boundary | INPUT | not run |
| the seed | `seed`; each window takes seed plus its index | INPUT | on the card 2026-10-10 |
| a text for the window | the timeline and its blocks; one text a load with a load per shot | INPUT | on the card 2026-10-10 |
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

A load capped at its shot's last frame is shorter than its window, and the
node holds the shot's last frame for the rest with an empty mask
(`video_mask.window_frames`). An empty mask is plate, and the plate there is
the source's last frame with the ORIGINAL subject in it. So a short load's
window is its own frames with the region open, followed by a long run of
clean, conditioning-labelled rows that show the original where the new
subject has to end up: the rows that outweigh a motion video on a
continuation, placed after the new frames where a context is placed before
them.

- 2026-10-10, NOT BUILT, scope only: hold the mask with the frame, so the
  region stays open over the held tail and the model draws the new subject
  standing still there. It would cost no sampling time (every step runs the
  whole sequence either way; only the regenerating share in the report
  changes), the held frames are never written, and it takes the original's
  look out of the window. It is a change to `video_mask.window_frames` with
  a choice to switch it and a check, because "a held plate is what a frozen
  row should see" was decided on purpose (the function's docstring).
  Evidence that the unmasked tail is survivable: two per-shot loads of that
  morning, each with a held tail several times its own length, kept the
  still's person on the lead's stills (two shots, one seed). Evidence that
  it could matter: every result below. What to read on a short load: its
  last frames, for the subject's look drifting toward the original as the
  held frames approach.

- 2026-10-10, later the same day, one render, one seed, read from stills
  and not on playback (the lead): a 29-frame load capped at its shot's end
  drew the still's person on its first frames and the ORIGINAL on its last,
  with the region open on every latent step of the shot (the render's own
  saved region) and a held tail four times the shot's length. Two readings
  the stills cannot tell apart: the sampler drew the original toward the
  held frames, or it drew something else and `only what changed` gave the
  source back. The window's stored latent, decoded and laid again
  (`bench/recomposite_window.py`), separates them; pending a decode.
- 2026-10-10: BUILT, not committed and not on a server: `held_tail` on the
  Masked Source, `the last frame, with its region open`
  (`video_mask.held_mask`). The default is the plate, as before.
- 2026-10-10, read from code: a shorter window is possible in core's
  terms. Core's lengths are one clip of the VAE plus a remainder
  (`comfy/sd.py`, the video VAE's ratios), and the lengths on both the
  video and the audio clock start below the planner's shortest
  (`loop_plan.CHAIN_LENGTHS`; the two shorter ones are the values
  `check_window_settings` allows as a context). What refuses them as a
  window is this pack's planner and the song node's own minimum, not core.
  Whether the model renders a window that short well is not known here: no
  render of one exists. Not built.

- 2026-10-10, the reading asked for before any patch, and it changes the
  line above from "possible" to "possible and outside what the model is
  documented to make". The vendor's README gives the output duration as a
  range whose lower end is four seconds (`coderef/MiniMax-H3/README.md`,
  the capabilities table, "Output duration"). Core's own node says where
  the trained range of frame counts starts and ends
  (`comfy_extras/nodes_minimax_h3.py`, the `length` input's tooltip;
  `docs/h3_geometry_and_nodes.md` restates it). The planner's shortest
  window is the shortest length on both clocks that is inside both
  statements, which is why `loop_plan.CHAIN_LENGTHS` starts where it does
  (`docs/h3_audio_freeze.md`, "short windows sit inside the trained
  range"). Of the two shorter lengths on both clocks, one is a little under
  the vendor's lower end and the other far under it. And the one that is
  nearly inside would still leave a load as short as the one above with a
  held tail about twice its own length. So a shorter window is not scoped
  as a patch: it buys little where it is nearly allowed and is unsupported
  where it would buy much. The open tail is the lever for a short load.

- 2026-10-10, the same short load measured (a peer's capture,
  `data/2026-10-10_fun_kitchen_shot2_1010`; the tracked figures are that
  session's to write): it is a FADE, not a flip. On the subject's hair,
  where the still and the original differ most in tone, the render holds
  the still's tone on the shot's first frames, moves toward the source's
  on every frame after, is about half way before the shot's last third, and
  is the source's on its last frames; the last frame of the shot is the
  source's to within the plate's own floor. So the tail's reach is long:
  more than two cycles of latent steps back from the first held frame.
- 2026-10-10, OPEN QUESTION: the tail's pull against the subject's share of
  the frame. The morning's two short loads, each with a longer tail in
  proportion, did not fade by the same session's look figure, and in those
  the subject was large in the frame; in the load that faded she is small
  and partly hidden. Not separated: one load of each kind.

- 2026-10-10, NOT BUILT, the next thing if the open tail leaves a doubt:
  fill the tail with the shot played back, the frames, the mask and the
  motion video mirrored in time past the last frame, with the region open.
  The reasoning: an open tail over a held frame still asks the model to draw
  the subject standing still for most of the window, with only a held plate
  and one held frame of the motion video to say so, and where that frame
  shows little of the subject the signal is weak. Played back, the window
  asks one thing from its first frame to its last, the subject moving as the
  motion video moves, and holds no frozen picture. It would be a third
  `held_tail` choice. Reasoned only.
- 2026-10-10, the order agreed with the lead: the open tail is the arm that
  tests the fade (the same load, window, seed, motion video and text, the
  one choice changed); the zoomed motion video on the same shot is read for
  the pose on the shot's first frames only, where the tail has not reached.

- 2026-10-10, the same short load's stored latent decoded and laid from the
  region the render saved: the fade is the SAMPLER's. The composite kept
  all of the render under the subject's mask on every frame, and the
  decode itself moves toward the source on every latent step as the held
  frames approach (`bench/results/2026-10-10_saved_windows_laid_again.md`,
  the short load). The other reading, the composite giving the source back,
  is ruled out for this render.
- 2026-10-10, the same load's pose (a peer's records,
  `data/2026-10-10_fun_pose/`, and `docs/wiki/state_signals.md`): on its
  first frames, where the render still shows the still's person, the motion
  video held the source's pose and the render did not take it. A load of
  its own that did NOT follow its motion video. What differs from the load
  that did, on the other stretch: the subject is a far smaller share of the
  frame, and most of the window is held tail. The open tail and the zoomed
  motion video are the two arms that separate those.

- 2026-10-10, the grid of a short load's last step, confirmed from
  `loop_plan.step_span`: a load that starts on a cut ends on a latent
  step's last frame only when its length, counted in cycles of
  `loop_plan.STEP_CYCLE`, leaves a remainder that is one of the cycle's own
  step starts. A shot's length is the shot's, so most short loads end
  inside a step, and that step then holds real frames and held ones: with
  the tail as plate it is part plate by construction, before any pull from
  the frames after it. The load that faded ends three real frames into a
  four-frame step. Under the open tail the held frame in that step carries
  the last frame's mask, so the one arm tests both. A second render of the
  same shot, a large, well-lit face, held until that last step and lost
  only it (a peer's figures), where the small subject lost twenty frames.

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
  `data/2026-10-10_fun_pose/`; `docs/wiki/state_signals.md` has the
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
  THE QUESTION THE NEXT TWO ARMS ANSWER: is there a level below 1.0 that
  turns the head and keeps the seam's step near the source's.
- 2026-10-10: the same morning's two per-shot loads kept the still's person
  with no frozen context (two shots, one seed;
  `bench/results/2026-10-10_one_load_per_shot_against_one_long_load.md`).
