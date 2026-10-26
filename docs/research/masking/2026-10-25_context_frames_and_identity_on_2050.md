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
  *(Dated record; one later change bears on it: since `449b17f1` on 2026-10-25 a held edge reaches back at most
  `video_mask.HELD_EDGE_EMPTY_RUN` frames, one latent step, where before it had no limit; `HELD_EDGE_LEAST` is 0.25 as
  written here. Verified against the code on 2026-10-26.)*
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

## The test, and what followed it

- **The 90-frame proof (`lead_fp_2050f`, `2050–2139`, rendered and kept 2026-10-24T22:41:51Z, `data/fun/owner_answers.jsonl:1619-1620`)**:
  a load of exactly 90 frames (`2050–2139`) as its own legal window (`89 = 1 + 88`) with `0` unmasked padding frames,
  every frame masked (`--no-draw-face-on 2050-2055,2083-2086`), one variable against `2050e` (the same 90 frames, 45
  masked and 45 unmasked). Zoom held at `1.86x` in `[210, 180, 762, 594]`, `7.9%` regenerated tokens, median rendered
  face height `125 px`, median lash gap `5.69 px`. The still's face transferred cleanly across all 90 frames (`"AMAZING! PERFECT!"`).
- **The full 243-frame piece (`lead_fp_2050g`, `2050–2292` laid as `2050–2247`, rendered and kept 2026-10-24T23:39:32Z, `data/fun/owner_answers.jsonl:1621-1622`)**:
  a legal 243-frame window (`241 = 1 + 240`, padded by 2 duplicated tail frames inside the node) masked end to end
  across the shot (`2050–2247`) and the 45-frame tail (`2248–2292`), with `--no-draw-face-on 2050-2055,2083-2086,2170-2177,2235-2270`,
  `--no-head-on 2249-2270`, and `--drop 2249-2270` for the 22 mirror-spin frames (`2249–2270`) where her reflection
  leaves the mirror before returning on `2271–2292`. Zoom was `1.27x` in `[120, 100, 928, 706]`, `8.1%` regenerated
  tokens, `0` unmasked padding frames, `0` held frames, median rendered face height `125 px`, median lash gap `5.41 px`,
  lid/lip score `0.98`. Kept on sight (`"Everything looks amazing here, even the spinning ball!!!"`) and laid on
  `2050–2247` in `data/fun/laid.jsonl` for the `2341Z` cut (then extended to `2050–2248`, the last frame before the
  `2249–2270` mirror spin), replacing `lead_fp_2050f`.

### Seam evidence across all six pieces laid on 2026-10-24

Inside the laid mask region (`parts_held > 0`), the frame-to-frame mean absolute difference (MAD, `0–255` scale) in the
render/composite against the raw source video across the exact same frame pairs confirms that every internal boundary
step (`f0 -> f0+1` and `f1-1 -> f1`) and the soft mirror-spin joins on `2050g` (`2247 -> 2248` and `2248 -> 2249`) sit
beside or below the source video's own change:

| Piece (`Span`) | Join Before (`f0-1 -> f0`) Comp vs. Src | First Step (`f0 -> f0+1`) Render vs. Source | Last Step (`f1-1 -> f1`) Render vs. Source | Join After (`f1 -> f1+1`) Comp vs. Source |
|---|---|---|---|---|
| `lead_fp_2050g` (`2050–2247`) | `88.25` vs. `87.51` (hard cut) | `28.26` vs. `33.19` | `29.65` vs. `28.89` | `48.81` vs. `47.31` (`2247->2248`) |
| `lead_fp_2050g` (`2050–2248`) | `88.25` vs. `87.51` (hard cut) | `28.26` vs. `33.19` | `34.12` vs. `37.06` (`2247->2248`) | `35.10` vs. `38.72` (`2248->2249` spin) |
| `third_fp_4647e` (`4647–4725`) | `102.38` vs. `100.77` (hard cut) | `3.44` vs. `3.75` | `16.92` vs. `16.16` | `35.46` vs. `39.34` (hard cut) |
| `lead_fp_4310c` (`4310–4389`) | `128.56` vs. `106.18` (hard cut) | `29.99` vs. `29.07` | `24.14` vs. `22.60` | `30.19` vs. `28.06` (`4389->4390` exit) |
| `third_fp_5675g` (`5675–5715`) | `103.96` vs. `87.99` (hard cut) | `14.21` vs. `23.42` | `11.05` vs. `6.87` | `65.02` vs. `62.25` (hard cut) |
| `third_fp_5011g` (`5011–5043`) | `59.55` vs. `66.21` (hard cut) | `19.48` vs. `24.70` | `8.66` vs. `12.55` | `37.68` vs. `36.46` (hard cut) |
| `third_fp_5461e` (`5461–5497`) | `36.15` vs. `42.16` (hard cut) | `19.00` vs. `27.98` | `25.75` vs. `22.15` | `66.34` vs. `68.95` (hard cut) |

Reading: at a hard camera cut (`f0-1 -> f0`), the composite's step across the cut can exceed the source's (as on
`4310c` and `5675g`) because the frame after the cut carries the new swapped face where the source had the original
actor, and the camera cut itself is large on both sides. That is the face swap across a scene cut, not a seam flash.
A `4479e`-style bright-square fault appears inside a shot (`f0 -> f0+1` or `f1-1 -> f1`) or at a mid-shot chop (`2248 -> 2249`),
and none of the six pieces shows one. On `2050g`'s mirror-rim patch (`[y=330:510, x=700:810]`), mean luminance steps
smoothly across `2245..2250` (`127.12 -> 102.01 -> 86.62 -> 74.81 -> 73.60 -> 71.37`), changing by `1.21` luma units
across the `2248 -> 2249` chop where `carried` hits `0 px`.

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
