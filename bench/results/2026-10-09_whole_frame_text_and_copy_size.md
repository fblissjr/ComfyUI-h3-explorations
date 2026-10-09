# Whole-frame video to video on one clip: what the text decides, how small the video model's copy can be, and when the still is needed (2026-10-09)

lane: whole-frame
verdict: on one clip, read from stills and one timing measure and not judged on playback: a window's text decided whether the render kept its source's framing in five matched pairs; the video model's copy of the source held all five windows at a 576 short edge and followed more loosely at 384; a lead who is already in the source kept his face with no still wired when the video model had its copy, and lost it when it did not

**Read this first.** One clip, one seed a window, one render a cell. Every
reading below is from eight matched stills a window and one timing measure.
The owner has watched two of the window-a renders and nothing else here, so
no line of this record is a verdict on a clip. A pair differs in the one
thing named; anything said across pairs is a reading, not a result. The
renders are on a newer core than the 2026-10-08 renders, which are therefore
not controls for these.

**What was asked.** The owner, 2026-10-09: a surreal version of the
whole-frame clip that 2026-10-08 made as swimmers; then how to cut the row
count, which they suspected of hurting that clip; then the swimmers again by
text alone, with no still, as fast as it will go.

## How

All of it is in the json beside this file
([`2026-10-09_whole_frame_text_and_copy_size.json`](2026-10-09_whole_frame_text_and_copy_size.json)):
the source, the settings every render shares, the five windows, one row a
render, the measure and its rule, the held frames, the joins. In short: the
Song node with a Masked Source whose mask covers the whole frame, one window
a run, each continued from the stored window before it. The source is the
finished one-person render of 2026-10-08, so the lead is already the person
wanted. The arms change how the source reaches the model (its own copy for
the video model at a short edge, the text encoder only, or a grey late
start), the text, whether a still is wired, and whether Sol is on.

## The text decides whether a window keeps its source's framing

Each pair is one graph and one seed, with the text changed as named.

| window | arm | what the text said | what the stills show |
|---|---|---|---|
| a, copy 576 | `surreal_copy576_a` | the summary described the new place, which the rows hide for the whole window | follows the input for about a hundred frames, then pulls out to a distant shot of that place |
| a, copy 576 | `surreal2_copy576_a` | no place named; "still seen from the chest or the waist up and still filling the frame from edge to edge" | the input's framing in all eight |
| c, copy 576 | `surreal2_copy576_c` | the place "comes into view"; the camera "pulls out" | the beats follow; from the second bow the camera is far past the input's |
| c, copy 576 | `surreal2_copy576_c_pinned` | "the same size in the frame as he is in `<Video 1>`"; the place "in the part of the frame where the background of `<Video 1>` shows"; an end state for the camera | the input's scale in all eight |
| e, copy 576 | `surreal2_copy576_e` | a strip of sand named for the bottom edge of the last frames | the rows drawn narrower than the input's, with sand round them |
| e, copy 576 | `surreal2_copy576_e_nosand` | no sand; the rows "span the full width of the frame ... as wide as they are in `<Video 1>`" | the rows nearer the input's width |
| a, copy 384 | `swimtext_copy384_a` | 2026-10-08's first text: the rows named as a kind of person, no framing | part of the rows unchanged, part other people; the framing leaves the input |
| a, copy 384 | `swim_copy384_a` | the rows named as what the source shows, the garment by its look, the scale pinned | the input's people and framing |
| a, late start | `e1_v2text_knots4_243` | no place named | the input's look to the last frame |
| a, late start | `e3_v2text_plus_place_knots4_243` | the same text with the one place clause put back | the text's look breaks in on the window's last latent frame |

What the pairs share: a text that names a thing its window's source does not
show, or leaves open how much of the frame a thing takes, was followed in
place of the source. The wording that held states sizes as a relationship to
the source video and names only what that window shows. The vendor's guide
says the second half already
([`docs/research/masking/2026-10-09_mrfetch.md`](../../docs/research/masking/2026-10-09_mrfetch.md)).
All of this is one clip.

The last pair settles a smaller question. In `surreal_zero_knots4_a` the
look flipped on frames 239 to 242. The text encoder's odd, repeated last
sample was a candidate. It is neither needed (a 192-frame window with an
even count flips) nor enough (a 243-frame and a 294-frame window with odd
counts do not): the json's `flip_probes`.

## How small the video model's copy of the source can be

| copy | windows | wall time for window a | reading |
|---|---|---|---|
| 576 | a to e, the surreal text | 1155 s | every window's framing holds once its text is right |
| 384 | a, the surreal text | 781 s | holds; the lock is a little looser |
| 384 with Sol | a to e, the swimmers text | 586 s | a holds; from b on the beats and the lead follow and the rows are drawn at fuller length than the input's |
| none (text encoder only) | a | 687 s at a 768 view, 486 s at 384 | the shot's arc follows; timing does not, about 29 frames ahead on a wide search |
| none, with a grey late start | a | 552 s at 4 knots | movement locks; the input's look stays, at 4 knots and at 1 |

The node rounds each side of the copy to 32, so a 576 short edge is 1024x576
on a 1344x768 canvas: a little wider in shape than the target. 384 is the
canvas's shape exactly. Both are smaller than anything the public trainer
prepares (same research page).

The 2026-10-08 failure of a 384 copy is not explained by its size:
`swimtext_copy384_a` is that day's text on today's graph and it also leaves
the input, though not in the same way.

## The lead without a still

The source already shows the lead. With no still wired:

- `swim_copy384_a` and `swim2_copy384_a` (the video model has its copy): the
  lead's face is the input's through the close-up and the turn.
- `swim_enc384_a` (the text encoder only): the lead is another person.

So the still is not needed when the source already shows the person and the
video model has its own copy of the source, and it is needed when either is
missing. Two renders against one, one clip, stills.
[`docs/h3_references.md`](../../docs/h3_references.md), "Edit a source
video", states the rule; the owner had it recorded on 2026-10-09 and no
default changed.

## Sol on these graphs

Two twins on the swimmers text, Sol at the node's defaults: 586 s against
675 s with the 384 copy, and 355 s against 486 s with no copy. In matched
stills each Sol render shows the same picture as its twin. On 2026-10-08, at
the full-size copy, the same mode was slower than dense. A pair of renders
is not a grade of Sol; a capture is.

## What did not work

- **A painted medium on the people.** Not from the style sentence, not from
  the medium written on each subject as the vendor's guide lays out
  (`surreal2_copy576_a_painted`), not from showing the references less clean
  (`MiniMaxH3ReferenceNoise` at 0.85 and 0.6). Changes stated on a thing (a
  garment's colour, a place) took every time. Untried: a style picture.
- **A restyle with no copy in the video model.** The grey late start holds
  brightness and structure, so the look does not change; weakening it
  loosened the movement first.

## Not explained

- **One held frame in the clip's last two hundred frames**, in every version
  of that stretch on both days (the json's `held_frames`). It is in the
  window's own file. It is not the join, the loader (each window's frames
  were matched to the source's by index), the window's length, the head of
  the last window, a fixed phase of the decoder's 17-frame clips, or the
  seed: another seed moved it five frames.
- **Windows b and c run about two frames ahead of their input** on the 576
  copy. Window b rendered cold does the same, so it is not the carry between
  windows.

## The two clips

`Video/mrship/surreal2_copy576_full_906_v2_with_audio.mp4` (the 576 copy,
dense, with the still) and `Video/mrship/swim2_copy384_full_906_with_audio.mp4`
(the 384 copy, Sol, no still), each 906 frames; the json's `joins` has the
windows and the wall time of each. The frame-to-frame change at each seam is
inside the range of its neighbours, except the first seam of the swimmers
clip, a little above.
