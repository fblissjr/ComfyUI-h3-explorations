# Masked video to video: research notes

Research notes on replacing one subject in a source video and keeping the
rest: tracking and masking the subject, what the mask does on the token grid,
the composite, and the models that could help. The owner asked on 2026-10-04
that every session working on this keep its notes here, so they are tracked
and shared.

This folder holds research and working notes. It is not the authority for how
the node behaves: `video_mask.py` and `audio_freeze_song.py` are, and the arms
that were rendered are `bench/masked_v2v_arms.json` and
`bench/masked_v2v_band_arms.json`.

## How to add notes

- One file per session per day: `YYYY-MM-DD_<session>.md`. The date in the
  name makes it a dated record, so measured numbers may live in it
  (`docs/prose_measurements.md`).
- Write in your own file. Another session's file is theirs to change; answer
  it from yours and link to it.
- Say how you know each thing. The labels used so far: card (a model card or
  the Hub API), paper, code (read in this pack or in ComfyUI core), seen
  (frames looked at), measured (a script's output), reported (another
  session's result), inferred, unverified.
- When a claim is corrected, keep it and mark it, so a reader who heard the
  first version finds the correction beside it.
- No media in git. The source clips, the reference still, renders and contact
  sheets stay on the shares or under `internal/`, named here by filename only.
- Add your file to the list below.

## Notes

- [`2026-10-07_mryolk.md`](2026-10-07_mryolk.md): SAM 3.1 in ComfyUI from input to output, stage by
  stage, for someone who knows ComfyUI and not SAM: what goes in and comes
  out of each stage, which file does it, what Meta's code does at the same
  point, and a verdict for each (same by reading, same by running,
  different as a choice or as a departure, not established). Its section 7
  collects what was measured on 2026-10-07 and section 8 what was not.
  [`2026-10-07_mryolk_stage_table.md`](2026-10-07_mryolk_stage_table.md) is its stage table: one row per
  stage with the lines on both sides and the tensor to compare.
- [`2026-10-07_mryolk_phrases.md`](2026-10-07_mryolk_phrases.md): what phrase to
  give SAM 3.1 and how specific to be, from Meta's examples, tests, eval
  configs and dataset cards: short plain nouns; one thing among several
  is chosen with a box or a click on top of the phrase, never with a
  longer phrase; presence is part of Meta's score and rejects describing
  phrases; what ComfyUI's detect node can and cannot do of that today; and
  a small measurement, planned and not run.
- [`2026-10-06_mrsun.md`](2026-10-06_mrsun.md): SAM 3.1's video pipeline as
  Meta built it, frame by frame, with the builder's values; where core's port
  departs (its own shorter session logic, the presence score dropped); which
  of the Subject Track's complaints are core's and which are Meta's; what a
  port would stand on and what Meta's code needs to run as a reference. A
  reading; nothing was run.
- [`2026-10-04_mrhf.md`](2026-10-04_mrhf.md): which Hugging Face models help
  beyond SAM 3.1 (effects outside the silhouette, matting, sync and identity
  instruments, sub-part masks); how core's SAM3 tracker behaves with several
  people; the shot table for the band segment; what the flicker beside the
  subject in the first clip is, and a fix to test.
- [`2026-10-04_mrblue.md`](2026-10-04_mrblue.md): what a code review of the
  mask path established and what it leaves open: a test that needs no
  sampling to tell whether the remnant is painted at decode or by the model,
  the one node that would let the two-sampler graphs carry a mask, what a
  soft band of mask values does, and two attributions nobody has made.
- [`2026-10-04_mrpink.md`](2026-10-04_mrpink.md): the build session's notes:
  what the mechanism is, what the two clips showed, why one word does not
  isolate one person among several, the turn that a generic prompt missed,
  the `replace` choice (whole subject, or head and hair), why the margin
  steers where the new subject stands, what version one of the shipped
  workflow is, and the late start that carries the turn and the original's
  look with it.
- [`2026-10-05_mryellow.md`](2026-10-05_mryellow.md): the lane read broadly
  the next day: why every method that puts the source into the target rows
  carries its look with the pose, the routes left for the turn in order
  (the source as a reference video, a mannequin from SAM 3D Body, automatic
  per-shot captions), and what the build can do beyond a subject swap
  (reshoot one shot, keep the person and replace the world, a part menu
  from Sapiens2, a redub, a clean plate from VOID, hand corrections from
  point prompts).
