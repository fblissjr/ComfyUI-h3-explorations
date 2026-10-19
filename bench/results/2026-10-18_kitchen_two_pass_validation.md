# Validating a two-pass scene by hand: the kitchen at 751-785, after the 24 fps fix (2026-10-18)

lane: masked video to video
verdict: a draft; the results section is filled when the night's run finishes

Written by mrted (the Claude lead) on 2026-10-18, before the run, at the owner's ask: what was run, how and why, the
scene's background, how a two-pass scene was validated by hand tonight, and what should probably be done on every
scene. The exact commands, as run, are kept outside git because they carry machine paths:
`internal/gemini/2026-10-18_kitchen_24fps_check.md` (linked from `docs/wiki/handover.md`). **This is a record of one
scene's check, not a recipe.** Only the three points under "Always true" hold everywhere.

## The scene, and why it needed two passes

The kitchen, source frames 604-1009: the mom (`second`) at the table and the lead walking behind her. Each person is
replaced in a pass of their own (two or more subjects in one pass is a closed lane, `docs/roadmap.md`). The mom's pass,
`second_hfp_0604h`, was kept and laid on 2026-10-18 (`bench/results/2026-10-18_second_hfp_0604h_recipe.md`); the lead's,
`lead_fp_0606`, was kept and laid the same day. Both had been rendered against the original video, so where the lead
passes behind the mom (about 755-775) the lead's chin was generated beside the original mom's dark hair, and when the
blonde mom was laid beside it a dark smear showed on the lead's chin (gem-smear, `internal/gemini/2026-10-18_worker_gem-smear.md`).

The fix tried was the second pass of the order the leads agreed for crossings: the person in front first, then the
person behind rendered against the cut that already holds the new front person. That run, `lead_fp_0751` (751-785),
was not kept: an open mouth from the old drawing, and hard square edges round her head.

## What was found tonight

1. **The cut was 23.976 fps.** The original `fun.mp4` is 23.976 fps; every job renders from a 24 fps every-frame copy,
   and every kept piece did. But `bench/rebuild_delivery.py` took the original before the copy when it made the joined
   cut, so the cut, and the stacks and review clips made from it, were written at 23.976
   (`docs/research/postmortems/2026-10-18_24fps_joined_cut.md`, with both leads' words).
2. **So the second pass loaded one frame early.** The job loader forces no rate (`h3_config.MASKED_JOB_FORCE_RATE`) and
   turns a frame into a time assuming 24 fps (`workflows/masked_job_graph.py::seek`); from about frame 600 a 23.976 file
   starts a frame early. `lead_fp_0751` loaded frames 750-784 of the cut against masks and a reference on 751-785, so
   every regenerated square carried the previous frame's background (mrgemson's trace, G30).
3. **The four-frame step and the margin also cost squares.** A keep-out or a margin that touches a cell on one frame of
   a four-frame latent step takes it for the whole step (`video_mask.py`, the keep-out and `run_lengths`); with the
   subject's own mask winning every cell it touches (`video_mask.py:2005`), a keep-out acts only on margin cells.

## How the scene was validated by hand tonight

In order (the commands are in the untracked file named above):
1. The fix on its own branch: the rebuild takes the 24 fps copy; the job step refuses a source not recorded at 24;
   the stack tool reads a saved start time at the rate it was written at. Cleared, landed.
2. The cut rebuilt: it must probe 24/1 with the same 6,395 frames, registered like the input's 24 fps copy.
3. `lead_fp_0751` rendered again against the new cut, every setting as before except the source, the same seed.
4. Old and new run measured with the same, fixed tools, in one pass:
   - each loader's first frame (`bench/what_h3_got.py`): source, masks and reference must agree;
   - the render against its source at offsets -2..+2, outside the region (`bench/stack_delivery.py::offsets_mae`,
     pointed at the render): best offset 0 when the frames line up;
   - pixel diffs: the new cut against the old, and frame to frame inside the region;
   - the capture's per-frame record (`frame_table.jsonl`, `manifest.json`): cells each keep-out gave back, depth;
   - the stack's yellow squares and kept pixels (`bench/stack_delivery.py`);
   - the assembler's findings (`bench/assemble_delivery.py`), `d001` and `d002` among them;
   - the block measure on the picture itself (`bench/measure_blocks.py`, written tonight);
   - the render's face points against the source's (the land step's pose pass);
   - the gap check on the span (`bench/frame_coverage.py`);
   - a Gemini reader watching both natively, writing a card that cites the figures.
5. The candidate on Start here for the owner; laid only on the owner's keep.

## Always true (the owner's rules, `docs/wiki/decisions.md`, 2026-10-18)

- H3 gets 24 fps from input to output, and a cut used as a source is made and loaded exactly like the real input's
  24 fps copy.
- Derived data (`.npz` files, tracks, class maps, depth, references) is made again after any change to the code that
  makes it.
- Old and new are measured with the same tools.

## What should probably be done on every scene (a proposal, for the owner and both leads)

- Two-person shots: the person in front first; the person behind rendered against the cut that holds them, at 24 fps.
- Before any render: the loaders' first frames agree (source, masks, reference).
- After every render: the facts beside the clip, in its JSON sidecars, read before anyone watches (`docs/wiki/handover.md`,
  item 1c): the offset check against the source, the pixel diffs, the block measure, the mouth opening against the
  source's, the face points, the gaps, and the reader's card citing them.
- The margin and the keep-outs read against the subject's own mask before a crossing renders (item 2b).
- On the back pass of a crossing, keep out only the front person's laid matte (where depth puts them in front), not
  their whole-body track (mrgemson, G31): on `lead_fp_0751` the mom's whole track reached into the lead's 16 px face
  margin where her laid piece had replaced nothing.
- Before a crossing renders, count the cells a keep-out or margin takes per four-frame step, not per frame (mrgemson,
  G31): a collision across a step can show 0 on every single frame. His example on 773-776 was traced with the stack
  tool before tonight's fix, which indexed the old run a frame off; re-check it with the fixed tools before relying on it.

## Results

(Filled after the run: the old-against-new table, the reader's card, and what the owner said.)
