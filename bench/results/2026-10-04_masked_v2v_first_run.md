# Masked video to video: the first renders

last updated: 2026-10-04

**Status: rendered, not judged.** The owner has not played these yet. What is
below is what the session read from low-resolution contact sheets and three
single frames, which can show whether the subject changed and the plate held
and cannot show lip sync, the edge at full size, or the audio.

## What ran

`bench/masked_v2v_arms.json`, on `workflows/h3_video_to_video_masked_song_pdd8_api.json`:
the PDD8 song chain, the source's audio frozen hard, one reference still, the
subject tracked by SAM 3.1 from the word "man". Rows, timings and substrate:
`data/results/2026-10-04_masked_v2v_arms.jsonl`. Clips are in the output
share under `Video/mrpink/`.

| arm | prompt | covers |
|---|---|---|
| `generic_w1` | `ref2va_masked_subject_swap`, which names no setting or shot | the first window |
| `specific_w1` | `ref2va_masked_stage_singer`, the first window's three shots in the source's order | the first window, same seed |
| `generic_30s` | `ref2va_masked_subject_swap` | the first thirty seconds, three windows |

`generic_w1` first ran alone as a throwaway to prove the path, and that is
the record's first row. The manifest's copy of it came back from the server's
cache, which is the second `generic_w1` row and times nothing.

**Rendered on the code as it stood before a peer's review was applied.** The
review's changes do not alter what these arms compute: the mask fit takes its
interpolating branch on this source, as before, and no window here is
unmasked. Reasoned from the two code paths, not re-rendered.

**After the review and a restart**, three short runs on the reviewed code
(rows in `internal/`, not tracked: they are smoke runs): one masked window
rendered; a window whose mask was empty was written from the source and not
sampled, as its report line says; and a tracker prompt for something not in
the clip still matched the microphone, so an empty mask had to be forced with
the threshold.

## The tracker, before any render

Two probe runs of the tracker alone, read from overlay sheets at one frame
per second over the first 32 seconds:

- The mask is on the singer in every shot, wide and close, across the clip's
  cuts, at core's default threshold and at the raised one.
- At the default threshold it also marks the hanging microphone as the man
  while the frame is still empty. The raised threshold clears the first
  second and not the second. The arms use the raised one
  (`SAM3_VideoTrack.detection_threshold` in the manifest).
- **The tracker kept the singer as one object across every cut.** Each
  object's mask saved on its own, at the raised threshold: the first object
  is the microphone, present only in the opening shot before the singer
  enters, and the second is the singer, present in every frame of every
  shot after he enters. No third object was made, so the cap
  (`h3_config.SEGMENTER_TRACK`) was not spent. Selecting the second object
  alone (`SAM3_TrackToMask.object_indices`) would drop the microphone; the
  arms did not, so its region regenerates for about a second of the opening.

## What the frames show

- **The path works.** The mask reached the sampler, Sol stayed on, the window
  wrote, and the audio stream is in the file. The song node's report gives
  the share of each window's video tokens that regenerate.
- **`generic_w1`:** the man from the still, in its cap and T-shirt, is in
  every shot, at close and full-length scale. The backdrop, the microphone,
  the framing and the cut times are the source's. No trace of the original
  singer's hair or clothing at this scale.
- **The original's shadow is still on the backdrop.** The mask is the body.
- **`specific_w1`:** the same, and the subject follows the source's blocking
  more closely: profile to the microphone, chin lifted, eyes closing in the
  close-up. On `generic_w1` the close-up has him looking toward the lens.
  One seed, so this is an observation about two clips and not about the
  prompts.
- **`generic_30s`:** the same subject through all three windows and all six
  shots. Six consecutive frames across each of the two joins show no jump in
  the subject. The node's report gives each window's share of regenerated
  video tokens; under three tenths in each, the rest frozen.

## For the owner, on playback

1. Likeness against the still, close and wide.
2. The mouth against the frozen vocal: does it follow, and does it rest when
   the voice does.
3. The edge of the subject at full size, and whether the composite shows.
4. On `generic_30s`: the two joins between windows, on the subject.
5. `generic_w1` against `specific_w1`: is the prompt that describes the
   clip's shots worth writing per clip.
6. Whether the audio is the source's, unchanged.

## The owner's verdicts, and the paint-out

- On the three first clips: "pretty good". Flicker beside the subject near
  the twenty-second mark of `generic_30s`, a shot the fourteen-second clips
  do not reach, so the pair of prompts says nothing about flicker. In the
  last second of `generic_30s` "the person is only partially there", with
  dark lines during movement.
- A peer session measured that flicker and the cut-off subject: a faint remnant of the
  original beside the legs with the mask edge fixed; the cap and the back cut
  on the region's edge in the close-ups, because the region is the old
  subject's outline and the replacement sits differently; the microphone
  regenerated wherever the tracker marks it.
- **`generic_paint_out`** (the subject filled in before the encode, same
  seeds, two windows): by that session's patch probe not a pass, better above
  the cap and worse at the shin and the microphone. The owner, on its second
  window: "tons of flicker ... cut off hat, black lines appearing to the left
  of the man every time he moves - much worse than before", and "looks like
  someone used a lasso tool and cut the man out poorly and quickly and pasted
  him onto the video". The fill also cost a large share of a window's time on
  the CPU (the song node's seconds in the rows). `paint_out` stays off.
- **`generic_changed`** is the response: the composite keeps the render only
  where it differs from the source or the old subject stood, the margin is
  doubled, the singer's object is selected alone and the still is the
  cleaned one. Its result is recorded below when it lands.

## Not run

- The unmasked trained path on the same clip
  (`workflows/h3_ref_video_swap_api.json`), the comparison the exploration
  proposed as the bar. Not rendered here: the owner asked for the build.
- A second seed.
- The source also wired as a `<Video 1>` reference.
