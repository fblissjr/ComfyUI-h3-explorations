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
carries the text `masked_prompt_text.assemble` writes for that recipe. The
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
