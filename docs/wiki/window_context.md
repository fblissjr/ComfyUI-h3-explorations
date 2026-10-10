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
| the context shown part noised | `context_noise` on the song node and the window node (0.271.0); core runs a mask value between 0 and 1 as that row's own strength | INPUT since 0.271.0 | untested; first arms after the next restart |
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
- 2026-10-10: the same morning's two per-shot loads kept the still's person
  with no frozen context (two shots, one seed;
  `bench/results/2026-10-10_one_load_per_shot_against_one_long_load.md`).
