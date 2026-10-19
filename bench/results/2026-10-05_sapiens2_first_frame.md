# Sapiens2 parts on the band clip: class order, time per frame, hair against SAM 3 (2026-10-05)

lane: masking
verdict: the class order is upstream's for the classes on the frame; hair found on every frame, where SAM 3's phrase missed four of 48

Session mrorange. The first real run of `MiniMaxH3Sapiens2Loader` and
`MiniMaxH3SubjectParts` (`sapiens2_parts.py`, landed in 8b451ebc). The numbers
are in `2026-10-05_sapiens2_first_frame.json`; this page says what was run and
what was seen. The script, its graph and the pictures (`sapiens2_first_frame.py`,
`first_frame_label_sheet.png`, `first_frame_hair_tile.png`) are not tracked.

**Everything here is one span of one clip**: 48 consecutive frames of one
shot, one subject, facing the camera at mid distance.

## What ran

Two seconds of `thinkaboutthings_compressed.mp4` from 101.0 s, 48 frames at
24 fps and width 1344, which is how the lane loads it: the warm group shot,
six people in one sweatshirt. One graph, built by hand, through a server
started for it:

- the lead from `MiniMaxH3SubjectTrack` with everything automatic;
- `MiniMaxH3SubjectParts` with `hair` alone and the matting model loaded
  (node 201 in the record);
- the same with the node's defaults, hair with face and neck, and no matting
  model (node 203);
- core's `SAM3_Detect` for the phrase `hair` at threshold 0.5 on the same
  frames (node 211), which is the call the Masked Source's `detect_part` makes
  for each of its phrases.

Models: `facebook_sapiens2-seg-1b` and `facebook_sapiens2-matting-1b` under
`models/sapiens2/`, the installed transformers' classes, bf16 autocast on the
card. The graph ran twice. The first run loaded every model cold from disk,
and its per-node timings were lost when the script failed on a path after the
run; its node reports are kept in the record as `first_run_reports`. The
second run followed a `/free` with `unload_models` and `free_memory`, so
nothing was cached and nothing was on the card, with the weights in the page
cache. No DiT was loaded in either.

## The class index order

The checkpoint's config names its classes `LABEL_0` to `LABEL_28`. Read with
upstream's list (`coderef/sapiens2/docs/SEG.md`, the node's `CLASS_NAMES`),
the label map puts each name on the right thing for the classes present on
the lead across the 48 frames: Hair, Face_Neck, Left_Hand and Right_Hand on
the correct sides, Upper_Clothing, Lower_Clothing, and Apparel on his watch.
Eyeglass lands on the glasses of a neighbour in the crop. The two lip classes,
the two teeth classes and Tongue all lie inside his mouth; at the sheet's size
they are not told apart from one another. Seen on the label sheet, eight
frames.

**Not shown by this clip**: the feet, shoes, socks, the bare arm and leg
classes and Torso. Nobody in the shot shows any of them, so their indices are
taken from upstream's list and have not been looked at. The order holding for
classes spread from index 1 to index 28 makes a shuffle among the rest
unlikely; it is not a check of them.

## Time per frame

From the node's own report, which times the crops, the forward passes and the
map back, and not the model load:

| | seconds per frame |
|---|---|
| segmentation alone (node 203) | 0.15 |
| segmentation and matting (node 201) | 0.33 |

Both runs gave the same figures to within 0.01. As whole nodes in the second
run, with their model loads, for the 48 frames (`per_node_s`):

| node | seconds for 48 frames | per frame |
|---|---|---|
| Subject Parts, segmentation alone | 8.2 | 0.17 |
| Subject Parts, segmentation and matting | 17.8 | 0.37 |
| `SAM3_Detect`, the one phrase `hair` | 8.5 | 0.18 |
| Subject Track | 5.6 | 0.12 |

The Masked Source asks SAM 3 for two phrases by default
(`video_mask.PART_PHRASES`), so its part detection is twice the `SAM3_Detect`
row. Against that, segmentation alone is **about twice as fast, not ten
times**; with matting it is about level. Where the 0.15 s goes has not been
profiled; the crop, the resize of 29 channels of logits and the per-frame
dilation all run one frame at a time. What the node buys is the next section,
and every part from the one pass.

## Hair against SAM 3

Sapiens2's Hair on the subject against SAM 3's `hair` cut to the subject's
mask widened by `subject_margin`, which is what the Masked Source's
`select_part` keeps. Both read back from 8-bit previews.

- **Where SAM 3 found the lead's hair, on 44 of the 48 frames, the two masks
  agree at an intersection over union of about 0.9** (0.89 to 0.92, median
  0.91). Sapiens2's is the larger by about a thousand pixels on a mask of
  about twenty-two thousand: on the tile for frame 6 it follows the loose
  strands at the hair's lower edge further down the chest.
- **On the other 4 frames (36, 37, 39, 45) SAM 3's detection was on somebody
  else**: it returns one detection for a phrase with no count, and on those
  frames it was a neighbour's hair, with nothing on the lead. These are the
  frames the Masked Source carries from a neighbour. Sapiens2 found hair on
  the lead on all 48, and its area moves by a few percent across them.
- The matte for hair has three to four thousand pixels strictly between 0 and
  1 per frame, along the hair's outer edge, and is hard where the hair meets
  the face and the sweatshirt, as designed.

What this does not say: which hair mask is the more correct along the edge
(nobody has judged it at full size), and anything about a second clip, a
close-up, a subject seen from behind or a small, distant subject, where the
crop is upscaled far more than here.

## Not tested

- The node with the DiT on the card. It ran beside SAM 3.1 only; the load goes
  through core's `load_models_gpu`, which is what moves the DiT out.
- The node upstream of a Masked Source, and the kept mask's key with it there.
- `hold_missing` on real frames: no frame of this span lacked the part.
