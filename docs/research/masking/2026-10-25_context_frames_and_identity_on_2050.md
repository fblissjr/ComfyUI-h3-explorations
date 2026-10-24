# 2026-10-25: the lead's face did not change on 2050, and what the window showed the model

A dated record (`docs/AGENTS.md`: what was true then). Written by the Claude reviewer at about 00:45Z from the
evening's reads with the Gemini lead; the Gemini lead lands it by pathspec with the close. Where this page and a
record disagree, the record is right: the job files under `data/2026-10-24_gemlead_jobs/`, the render graphs under
`data/2026-10-24_lead_fp_2050*_job/`, the node's own reports in each `*.render.queued.json` (`said["75"]`), and the
sheets under `data/2026-10-24_lead_fp_2050e/`.

## What happened

- `lead_fp_2050c` (2050-2247, 198 frames, zoom 1.42x) and `lead_fp_2050e` (2050-2094, 45 frames, zoom 1.86x, the
  size probe) both rendered the singer's own face in the hole, not the still's. Inside the `parts_held` face mask,
  only `0.04%–0.16%` of pixels (`8–32 px` out of `19,306–25,499 px`) were 1:1 unchanged (`diff == 0`), while `79.8%–85.9%`
  changed by `> 5 / 255` (`mean_diff = 14.01–22.51`, `max_diff = 121–178`), and the mouth and eyes changed to follow the
  drawing (open mouth on 2050 where the film's is nearly closed; upper teeth replaced by a dark oral cavity on 2056;
  open eyes with dark irises on 2085 where the film's are shut; `face_closeup_src_vs_render.jpg`). The model synthesised
  a face from noise (`start_from: noise`, "nothing of the source is under the mask") and the face it synthesised was hers.
- Ruled out as the cause, each with its evidence: size (her face 380 px on the canvas at 1.86x, above the node's
  300 px target, 3.6x the kept 4310c's); the still (the same 1800x1800 file handed at 2048, 4,096 tokens, first in
  the reference order, on every lead render); the text (byte-identical, sha256 39e04f598c3190da, on thirteen lead
  renders including kept ones); the reference order (one appended picture, then the wired motion video, on every
  load); the audio (her voice, frozen, on every lead shot); hair and clothing words (the same on every kept lead piece).

## What the window showed the model

- The window is a legal length (39, 90, 141, 192, 243, 294, 345 on both clocks), and a shot shorter than it is
  padded after with real frames of the source, shown "as filmed" since 2026-10-21: nothing regenerated on them, the
  original subject in the window unmasked (`video_mask.py::pad_context`, `CONTEXT_AS_FILMED`). Padding that crosses
  the shot's recorded span is held copies of the edge frame with its mask (`workflows/masked_job_graph.py::held_context`,
  the 2026-10-22 seam rule). The shots file records 2050 as 2050-2622, so both renders' padding fell inside the shot
  and stayed real.
- 2050e: 45 of its 90 frames were the singer unmasked, 2095-2139. 2050c: the mirror swings back and her face is
  unmasked on 21 of its 45 padding frames, 2272-2292 (`2050c_context_2248_2292_grid.jpg`).
- Every kept lead render was checked the same way. None had her unmasked face in its window: 22 older pieces had
  windows sized to the load; 11 had padding wholly held past a cut; 4310c's 9 real padding frames are after she
  leaves the frame. Two pieces the reviewer first named as counterexamples, 3098b and 4053c, were withdrawn on the
  frames (`3098b_ctx.jpg`, `4053c_ctx.jpg`): she has left the frame on their padding, and the tracker's mask the
  reviewer had counted was on a bubble, another woman, and the dancers (bug 4 of 2026-10-23: a track following
  another person after the subject leaves; the lead's tracks on 3098 past 3109 and on 4053 past 4097 are on other
  people and must not be read as her).
- One more rule of the padding that bears on the fix (`docs/masked_v2v/frames_rules.md` §4, the guard): where the
  shot's last frame has no mask, or less than `video_mask.HELD_EDGE_LEAST` of the shot's median region, held copies
  fall back to real frames and the plan row says so (`held_fell_back`); `second_hfp_4912e` is why. On 2050 her face is
  leaving the mirror at 2247 (a forehead sliver at 2248), so a boundary recorded there may trip the guard and put real
  frames back. Measured (Gemini, 2026-10-25 about 01:10Z, `video_mask.py::held_edge_falls_back`, `HELD_EDGE_LEAST`
  0.25, on `parts_held__shot_c_2050f.mkv`): the median region over 2050-2247 is 21,829 px, the floor 5,457, and
  frame 2247 carries 4,578, so a boundary recorded at 2247 WOULD have tripped the guard and put the 45 real frames
  back. The three paths below avoid the guard: a load that is its own window has no padding to fall back on.
- The face-covered context word does not remove this: it stamps the shot's last mask onto the moving padding frames,
  covering 41-61% of her face there (measured on 2095-2139 against `parts_held`).

So 2050 is the only lead render ever made with the singer's unmasked face in its window, and it is the only one
whose face did not change. The second thing particular to 2050, look-alike faces pressed against her box and her
face a mirror reflection, has no counterexample either and stays the second suspect.

## The result (2026-10-25, 02:40Z; the owner's keep at 22:41Z local, "AMAZING! PERFECT!")

`lead_fp_2050f` (2050-2139, 90 frames, its own window, every frame masked, no padding, zoom 1.86x in [210, 180,
762, 594], seed 730451892, the same text and still) put the still's face in on every frame read (2056, 2085, 2120;
`lead_fp_2050f_2050_2139_stacked.mp4`), with her own hair kept round it, where `lead_fp_2050e` (the same 90-frame
window, 45 masked and 45 unmasked) drew the singer. One variable changed, the 45 unmasked frames of her in the window,
and the face changed with it. The owner kept it on the review page.

So the cause is the original subject shown unmasked in the window, and on 2050 nothing else was: the same seven
faces press against her box in 2050f; the ring, the drawing, the text, the still and the audio were unchanged. The
four tests queued behind this one are not run for 2050. The real piece follows the same way: 2050-2292 as a legal
243-frame window masked end to end, laid as 2050-2247.

What it means for the rest of the lead: any load whose padding fell inside its own shot with her in frame rendered
with her original face in the window. 5935, 6105 and 3759 have their padding read before they render, and the
padding rule's build is the recipe for finishing them.

## The owner's ruling (2026-10-25, about 00:10Z to 00:20Z)

Padding frames exist only to reach a legal window length. The original subject is never shown unmasked in a
window. Three paths:

1. A shot up to one window (345 on both clocks, `docs/h3_resolutions.md` "The length axis": 362 is the training
   ceiling and ends with the sound off its last frame, so masked loads stop at 345) runs as one load to the window's length, masked end to end from the
   preview's masks, and is laid short (the laid list already separates the shot from the frames laid).
2. A shot longer than a window runs as chained windows inside one load, each re-denoising a 90-frame overlap on the
   previous window's render and tossing it: what the chain does today.
3. The padding side is chosen by the footage: masked real frames where the footage continues (2050's mirror spin),
   held copies of the edge frame across a hard cut (the 2026-10-22 seam; `frames_rules.md` §2, a shot's frame
   outnumbered in its 17-frame clip, 4479e's bright square). Nothing in front except a shot at the clip's end, as
   since 2026-10-22 (`frames_rules.md` §4). Every padding frame is dropped after the decode (§5); a frame of the load
   past the shot is dropped at the lay by the laid row's `frames` field, which already lays a sub-span.
   One clause the owner's ruling needs (from 4912e, 2026-10-21, the case that made the old default): where the edge
   frame's mask is lost (4946 had none beside 13.5k px on 4942-4945), held copies of that frame show the subject
   untouched with the hole closed. So held copies across a cut take the LAST GOOD mask, not the edge frame's empty
   one, and the hole stays open on the padding. Masked real frames (path 1) carry their own tracked mask per frame; a
   lost mask there is the tracker's fault to fix, not the padding's.

This reverses the 2026-10-21 "as filmed" default, kept on 2026-10-22 for the mom's lowered gaze beside the shot; if
her gaze loses something, it shows on her next render and gets its own look. One shot is never split; the seam rule
across a cut stands; the laid row still names the shot and the frames laid.

## The test, and what follows it

- The proof, no build: a load of exactly 90 frames (2050-2139) is its own window with no padding, every frame
  masked, one variable against 2050e (the same 90 frames, 45 masked and 45 unmasked). Its zoom box is the union over
  90 frames, so its zoom is below 2050e's 1.86x; size is ruled out, so that does not cloud the answer.
- If the still's face appears: the real piece the same way, 2050-2292 as a legal 243 window masked throughout, laid
  as 2050-2247; and the job tool choosing the span as the window length by itself when the shot continues.
- If it does not: the crowd test (the other faces painted out on a copy, never the film, on the owner's word), then
  `paint_out` (the ring: the encoder is convolutional and kept cells beside the mask carry the old face; core's own
  inpaint hides the original before its encode; off here "until a matched render says it helps"), then the still
  cropped to her head at the same 2048, then the drawing off at the cost of gaze. Each against the proof as baseline.

## What else the evening found, for the leftovers list

- The zoom enlarges frames already shrunk to the canvas: the loader hands 1024x768 of a 1424x1080 clip, and the
  node's report says a loader handing the clip's own frames would give the box more pixels. The zoom buys tokens, not
  detail. With it: the zoom's height measure counts stray mask pieces (5935c: 323 px raw, 241 the largest piece), and
  both bounds read the same per-frame boxes, so the fix is the largest piece per frame, with the box per step.
- The mark check measures position, not state: an open mouth drawn on a closed one and open eyes on shut ones passed
  it on 2050 and 2085. A third measure, the eye line's roll, caught 2175 (15 degrees) where nose distance and pitch
  did not.
- After a render, the take reads the node's own writes (the mask-review video, the raw zoomed canvas), not the
  chain's record; on 2050e they agreed, and the rule is that nobody finds out by assuming it.
