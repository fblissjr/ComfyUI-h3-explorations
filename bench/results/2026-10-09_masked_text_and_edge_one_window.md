# Masked video to video on one window where the subject goes from large to small: a text written to what the window shows, and the finer edge (2026-10-09)

lane: masked video to video
verdict: not judged. One window, one seed, three renders, read from stills only: the control and the finer-edge render show the lead facing the camera at the start, where the source shows a back, and a bare-headed figure in a white shirt from about the middle; the render whose text names only what the window shows keeps the cap and T-shirt on every still read. Nothing here is a default.

**Read this first.** One window of one clip, one seed, one render a cell: one
source video, one lead (one still, one set of clothing) and one scene. What
it shows is about this window; whether it holds on another clip, another
person or another setting is not known and is not claimed. Every
reading below is from stills on thirteen frame numbers, set against the
source's frame of the same number. The owner has watched none of these; a
still cannot show what must hold over time, so no line here is a verdict on a
clip. The text render differs from the control in several sentences at once
(listed below), so what it shows cannot be put down to any one of them.

**What was asked.** Two of the changes the upstream cross-check of this date
raised ([`docs/research/masking/2026-10-09_mrfetch.md`](../../docs/research/masking/2026-10-09_mrfetch.md)),
each against a control on the same window, seed and code: a window's text
that names only what that window's source shows, and the region's mask
handed to the sampler per latent cell (`MiniMaxH3MaskedSource`, `edge`). The
window was chosen for a subject who goes from large to small in the frame
and is seen from behind for most of it.

## How

Everything is in the json beside this file
([`2026-10-09_masked_text_and_edge_one_window.json`](2026-10-09_masked_text_and_edge_one_window.json)):
the window, the settings the three renders share, one row a render, both
texts in full, and the stills read. In short: the final masked graph of
2026-10-08, cut to one window of the length the json gives, with the newer
still and the margin set by the subject's size in every render. The control
carries the text `masked_prompt_text.assemble` writes for that recipe.

*Added later on 2026-10-09: the control is not what that final graph
rendered with.* The graph stored in the 2026-10-08 final carries a
hand-written text for each of its windows, several hundred words apiece,
and each says which way he faces and when he turns; its first window also
opens on his face. The control here drops those texts for the node's own,
which says nothing about the view and keeps the face "in every frame", on a
window that opens on his back. So the control asks what the node's text does
by itself on such a window, and the text render puts back, in a few
sentences, what the final's texts already said at length. The owner, on
playback of the control: he faces the wrong way at the start and walks
backward. The
subject was picked the way that graph picks, and the pick was confirmed on
the tracker's tile before anything was queued.

| render | differs from the control in |
|---|---|
| control | nothing |
| text | the Song node's prompt |
| edge | `edge` set to `latent cells` on the Masked Source |

The text render's prompt was written from the window's frames and read twice
against them by the session on the whole-frame lane. Against the control it:
says in the summary that he is seen from behind for most of the window and
faces the camera only at its end; limits the retained face to where it shows
at the end; replaces the one sentence placing him in the scene with three
that say which way he faces through the window, naming the back of the cap
and of the T-shirt; and adds that he is the same size in the frame as the
person in the motion video at every moment.

## What the stills show

Not judged. Frame numbers are the window's own, from zero.

| frames | source | control | text | edge |
|---|---|---|---|---|
| 0, 30 | a back | faces the camera | a back, cap and T-shirt | faces the camera |
| 90 to 128 | a back | a back, cap and T-shirt | a back, cap and T-shirt | a back, cap and T-shirt |
| 136, 144 | turned to the side | bare head, pale then white shirt | cap and T-shirt, turned to the side | bare head, pale then white shirt |
| 150, 200 | among the rows, part hidden | bare head, white shirt | cap and T-shirt | bare head, white shirt |
| 242 | faces the camera, head bowed | bare head, white shirt, faces the camera | cap and T-shirt, faces the camera | bare head, white shirt, faces the camera |

Three things a reader should not take from this table:

- **That the figure in the control's later frames is the source put back.**
  On the frames read it is in a different pose from the source's frame of the
  same number, so it was drawn. It has the look of the person in the motion
  video, which the text encoder sees and the video model does not; that is a
  reading of what it resembles, not a measured cause.
- **That the finer edge does nothing.** Its own report line has fewer tokens
  regenerated and more of the regenerated pixels kept than the control's (the
  json has both), and on the stills read it is the control. Its aim was the
  halo round a small subject, which these stills do not resolve and this
  window's control does not isolate, because the lead is not the wanted
  person in the frames where he is small.
- **That the text rule is carried.** One seed. The same rule decided the
  framing in the whole-frame pairs of this date
  ([`2026-10-09_whole_frame_text_and_copy_size.md`](2026-10-09_whole_frame_text_and_copy_size.md)),
  where the video model has its own copy of the source; here it sees the
  source only outside the mask, so this is the rule on a different arm.

## The owner's playback of the control (2026-10-09)

Only the control has been watched. At the start he faces the wrong way and
walks backward. At about the five-second mark the lead becomes the other
man, and never becomes the owner's likeness again. At the end, when he turns
to the camera, he is a third person, neither the lead nor the source's man.
A close sheet of the last two seconds agrees on stills: bare head, white
shirt and a tie, a face that is not the source's, and the turn a few frames
ahead of the source's. The text render on the same frames keeps the cap and
T-shirt through the turn; whether its face is the lead's is for playback.

## Where the graph comes from, and what is not the final's

The 2026-10-08 final is the shipped graph
`workflows/h3_video_to_video_masked_upper_song_ref2va_motion_api.json` with
its prompt node swapped for a typed text per window and its loader for a
path loader, the pick set to the largest person on a named frame, and the
motion reference at the canvas's short edge. The tracker, the part model,
the Masked Source's other settings, Sol and the schedule are the shipped
graph's. This trial keeps all of that. Beside the prompt, its control
differs from the final in: one shorter window that opens on his back (the
final's first window opens on his face, and each later one continues from
the one before); the newer still; the margin rule; a newer core
(`git reflog` in the ComfyUI checkout has the two pulls); and one tracker
change that landed after the final rendered (`git log -- subject_track.py`).
So the control is not the final with a different prompt, and its failure is
not shown to be the prompt's alone. The text render shares every one of
those with the control and differs from it only in the prompt.

Nothing was reused by this pack: `audio_freeze_song.WINDOW_REUSE_ENABLED`
and `video_mask.MASK_REUSE_ENABLED` are off and each render's report says so
(the json has the lines). ComfyUI's own node cache served the track and the
parts to all three renders from the pick preview run before them, which is
read from the timings, the server's history being gone; all three therefore
rendered from one and the same mask.

## Caveats

- The text render carries several changes at once: the view, the garments
  named in the shot sentences, a narrower retained face, and the size
  sentence. A render apiece would be needed to say which mattered.
- The control's text is the node's own, so a result in the text render's
  favour is a result about what that text leaves unsaid on a window like
  this one, where the lead never shows the face for most of it.
- The part model reused a trusted frame's part on a run of frames in the
  window's second half (the json has them), in all three renders alike.
- Wall times in the json are rough: not a timed run, and a peer's CPU work
  ran beside the later two.
- The motion reference was the whole frame at the canvas's own short edge,
  where `video_mask._reference_size` returns the canvas's shape, so the
  proposed change that keeps a smaller copy in the canvas's shape would have
  changed nothing here. It is not built.

## Not run

The crop. One fixed box has to hold the subject where he is largest, which
on this window is nearly the whole frame, so it enlarges him by almost
nothing. It is deferred to a window where he is small throughout, with a
control of its own (owner, 2026-10-09).

## What would change a default

Nothing does yet. The owner set the bar on 2026-10-09: no default moves
unless a pair is a marked improvement on playback, and the pairs have not
been watched. If the text pair holds on playback, it has held on one clip
with one lead in one scene; it would need to repeat on a different clip and
a different lead before the node's text changes. Then the questions are
which of its sentences did it, and whether the node can write them from what
the tracker already knows (which way the subject faces, and when the face
shows).
