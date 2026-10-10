# ComfyUI core's SAM 3D Body against Meta's own inference code, on the same image, boxes and camera (2026-10-10)

lane: sam3d
verdict: on two public samples, on the CPU in float32: core's port computes what Meta's code computes once both are given the same crop, to under the difference between Meta's own two precisions for five boxes of six; as shipped core is off Meta by millimetres, all of it from how core samples the crop, smallest when the person's crop is about the model's input size and larger when it is shrunk or enlarged; one hand moved by centimetres where the two sides disagreed on using the hand decoder; the camera is the same on both sides; expression is zero on both

**What was asked.** The render lane uses a SAM 3D Body mesh as a motion
signal ([`docs/wiki/masked_v2v.md`](../../docs/wiki/masked_v2v.md)), through
ComfyUI core's port, and the owner does not take core's ports of Meta's
models on trust (core's SAM 3 port had three departures found on
2026-10-07, [`2026-10-07_sam3_core_against_meta.md`](2026-10-07_sam3_core_against_meta.md)).
The lead asked for the same comparison here: core's whole prediction
against Meta's on the same frames, the camera each used, and whether hand
refinement ran on each side. The reading of the two code paths that came
first is [`docs/wiki/meta_perception_models.md`](../../docs/wiki/meta_perception_models.md),
"Core's port against Meta's inference"; this record is the run that page
said was missing.

**How.** [`bench/compare_sam3d_body_core_against_meta.py`](../compare_sam3d_body_core_against_meta.py)
runs core's `SAM3DBody_Loader` and `SAM3DBody_Predict` in its own process on
the DINOv3 file, and Meta's `SAM3DBodyEstimator.process_one_image` from
`coderef/sam-3d-body` in a child process on Meta's original `model.ckpt` and
rig file, with a Python outside this repo that has Meta's dependencies.
Both on the CPU, the card masked. The child changes three things, none of
them arithmetic (the tool's docstring lists them): the backbone's source is
the local `coderef/dinov3`, moves to the card are no-ops, and the rig loads
from its TorchScript file by Meta's own switch. Each side gets the same
image and the same boxes.

Four comparisons a row group:

- **core against Meta**: core as it ships, against Meta in float32.
- **the control**: core's model again, with the one function that samples
  the crop (`comfy/ldm/sam3d_body/utils.py::warp_affine_batched`) replaced
  in that process by OpenCV's call as Meta makes it
  (`coderef/sam-3d-body/sam_3d_body/data/transforms/common.py:305`).
  Nothing on disk is changed.
- **the floor**: Meta as released (the backbone in the precision its config
  names) against Meta in float32. A difference under this is under what
  Meta's own precision choice moves.
- two **cameras**: `default`, where neither side is given intrinsics, and
  `fov`, where both are given the same vertical field of view.

**The inputs are Meta's own published samples**, so nothing here is
anybody's media: `coderef/sam-3d-body/notebook/images/dancing.jpg` with a
whole-frame box and the bounding box of the mask beside it
(`dancing_mask.png`), and frame 60 of
`coderef/sam-audio/examples/assets/office.mp4`, a low-resolution frame, with
a whole-frame box and three boxes typed by hand, one a person. A frame is
its file name and its technical facts; what it shows is in no file here.

**What this does not say.** It is core on the CPU in float32. The server
runs core in half precision on the card through a different model patcher;
[`2026-10-05_sam3d_body_vith.md`](2026-10-05_sam3d_body_vith.md), item 1, is
how far those two were apart on one frame. No detector, no mask
conditioning, no track, no smoothing, no ViT-H release, no clip. Two
images, six boxes. There is no ground truth: nothing here says which side
is closer to the person, only how far apart they are and why.

Every number is in
[`2026-10-10_sam3d_body_core_against_meta.json`](2026-10-10_sam3d_body_core_against_meta.json)
and both tables below are printed from it by the tool's `render`; where a
sentence and the file disagree the file is right.

## How far apart, and what closes it

Distances between the two bodies for one person. "Model pixels per source
pixel" is how much the box is shrunk (under one) or enlarged (over one) to
reach the model's input. The 2D columns are in pixels of the source image;
the last column is each hand's keypoints about its own wrist, which is the
hand's shape apart from where the arm put it.

| image | person | box, model pixels per source pixel | camera | comparison | vertices, mean (mm) | after centring (mm) | 2D keypoints, mean (px) | head (px) | body (px) | right hand (px) | left hand (px) | hand shape about the wrist, right, left (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dancing | 0 | whole frame, 0.14 | default | core against Meta | 17.76 | 13.72 | 4.46 | 9.42 | 5.99 | 1.75 | 4.30 | 0.70, 1.31 |
| dancing | 0 | whole frame, 0.14 | default | core against Meta, body decoder only | 19.40 | 15.55 | 6.83 | 10.34 | 6.51 | 3.98 | 9.19 | 6.93, 8.40 |
| dancing | 0 | whole frame, 0.14 | default | core with OpenCV's crop against Meta (the control) | 0.08 | 0.08 | 0.04 | 0.04 | 0.03 | 0.06 | 0.04 | 0.19, 0.09 |
| dancing | 0 | whole frame, 0.14 | default | Meta as released against Meta in float32 (the floor) | 0.79 | 0.75 | 0.23 | 0.72 | 0.17 | 0.10 | 0.29 | 0.38, 0.45 |
| dancing | 0 | whole frame, 0.14 | fov | core against Meta | 17.01 | 12.94 | 3.97 | 8.86 | 6.01 | 1.27 | 3.27 | 1.08, 2.19 |
| dancing | 0 | whole frame, 0.14 | fov | core against Meta, body decoder only | 18.14 | 14.48 | 6.44 | 10.28 | 6.43 | 4.07 | 7.92 | 6.49, 8.20 |
| dancing | 0 | whole frame, 0.14 | fov | core with OpenCV's crop against Meta (the control) | 0.07 | 0.06 | 0.05 | 0.04 | 0.03 | 0.09 | 0.04 | 0.09, 0.13 |
| dancing | 0 | whole frame, 0.14 | fov | Meta as released against Meta in float32 (the floor) | 0.73 | 0.67 | 0.21 | 0.71 | 0.16 | 0.13 | 0.23 | 0.15, 0.56 |
| dancing | 1 | on the person, 0.31 | default | core against Meta | 4.07 | 3.91 | 1.41 | 2.65 | 1.55 | 0.79 | 1.58 | 0.59, 1.24 |
| dancing | 1 | on the person, 0.31 | default | core against Meta, body decoder only | 3.56 | 3.35 | 1.84 | 2.61 | 1.38 | 2.09 | 1.90 | 1.38, 3.19 |
| dancing | 1 | on the person, 0.31 | default | core with OpenCV's crop against Meta (the control) | 0.05 | 0.05 | 0.03 | 0.01 | 0.01 | 0.03 | 0.04 | 0.07, 0.06 |
| dancing | 1 | on the person, 0.31 | default | Meta as released against Meta in float32 (the floor) | 0.39 | 0.33 | 0.13 | 0.13 | 0.11 | 0.07 | 0.20 | 0.12, 0.54 |
| dancing | 1 | on the person, 0.31 | fov | core against Meta | 3.96 | 3.74 | 1.57 | 2.53 | 1.53 | 1.59 | 1.37 | 0.79, 1.03 |
| dancing | 1 | on the person, 0.31 | fov | core against Meta, body decoder only | 3.49 | 3.36 | 1.90 | 2.51 | 1.43 | 2.21 | 1.97 | 1.43, 3.31 |
| dancing | 1 | on the person, 0.31 | fov | core with OpenCV's crop against Meta (the control) | 0.07 | 0.06 | 0.03 | 0.01 | 0.02 | 0.02 | 0.08 | 0.04, 0.13 |
| dancing | 1 | on the person, 0.31 | fov | Meta as released against Meta in float32 (the floor) | 0.35 | 0.33 | 0.12 | 0.11 | 0.09 | 0.08 | 0.20 | 0.10, 0.54 |
| office_frame60 | 0 | whole frame, 0.48 | default | core against Meta | 4.56 | 4.01 | 1.17 | 1.01 | 0.98 | 2.05 | 0.54 | 9.96, 1.13 |
| office_frame60 | 0 | whole frame, 0.48 | default | core against Meta, body decoder only | 3.69 | 3.02 | 1.28 | 1.04 | 1.08 | 1.71 | 1.12 | 1.69, 1.59 |
| office_frame60 | 0 | whole frame, 0.48 | default | core with OpenCV's crop against Meta (the control) | 1.25 | 0.81 | 0.21 | 0.04 | 0.21 | 0.36 | 0.08 | 1.05, 0.68 |
| office_frame60 | 0 | whole frame, 0.48 | default | Meta as released against Meta in float32 (the floor) | 0.77 | 0.64 | 0.24 | 0.06 | 0.25 | 0.27 | 0.26 | 0.61, 0.81 |
| office_frame60 | 0 | whole frame, 0.48 | fov | core against Meta | 4.11 | 3.62 | 1.06 | 0.96 | 1.15 | 0.90 | 1.15 | 9.03, 1.61 |
| office_frame60 | 0 | whole frame, 0.48 | fov | core against Meta, body decoder only | 3.46 | 2.98 | 1.27 | 0.91 | 1.26 | 1.53 | 1.10 | 1.73, 1.60 |
| office_frame60 | 0 | whole frame, 0.48 | fov | core with OpenCV's crop against Meta (the control) | 1.35 | 0.89 | 0.29 | 0.09 | 0.24 | 0.40 | 0.29 | 0.92, 0.43 |
| office_frame60 | 0 | whole frame, 0.48 | fov | Meta as released against Meta in float32 (the floor) | 0.97 | 0.83 | 0.28 | 0.06 | 0.17 | 0.45 | 0.27 | 1.81, 0.81 |
| office_frame60 | 1 | on the person, 2.12 | default | core against Meta | 2.47 | 1.76 | 0.42 | 0.38 | 0.36 | 0.55 | 0.38 | 0.85, 0.51 |
| office_frame60 | 1 | on the person, 2.12 | default | core against Meta, body decoder only | 2.44 | 1.75 | 0.43 | 0.39 | 0.37 | 0.58 | 0.37 | 0.83, 0.50 |
| office_frame60 | 1 | on the person, 2.12 | default | core with OpenCV's crop against Meta (the control) | 0.14 | 0.08 | 0.01 | 0.00 | 0.02 | 0.01 | 0.00 | 0.02, 0.02 |
| office_frame60 | 1 | on the person, 2.12 | default | Meta as released against Meta in float32 (the floor) | 0.94 | 0.52 | 0.08 | 0.02 | 0.14 | 0.04 | 0.08 | 0.12, 0.16 |
| office_frame60 | 1 | on the person, 2.12 | fov | core against Meta | 2.63 | 1.87 | 0.43 | 0.37 | 0.37 | 0.53 | 0.40 | 0.95, 0.63 |
| office_frame60 | 1 | on the person, 2.12 | fov | core against Meta, body decoder only | 2.63 | 1.86 | 0.43 | 0.37 | 0.37 | 0.55 | 0.39 | 0.90, 0.65 |
| office_frame60 | 1 | on the person, 2.12 | fov | core with OpenCV's crop against Meta (the control) | 0.13 | 0.08 | 0.01 | 0.00 | 0.03 | 0.01 | 0.01 | 0.02, 0.02 |
| office_frame60 | 1 | on the person, 2.12 | fov | Meta as released against Meta in float32 (the floor) | 0.97 | 0.51 | 0.09 | 0.02 | 0.13 | 0.05 | 0.10 | 0.11, 0.19 |
| office_frame60 | 2 | on the person, 1.10 | default | core against Meta | 11.23 | 11.67 | 3.66 | 0.71 | 2.26 | 8.93 | 0.64 | 25.56, 0.52 |
| office_frame60 | 2 | on the person, 1.10 | default | core against Meta, body decoder only | 0.87 | 0.51 | 0.11 | 0.06 | 0.21 | 0.05 | 0.08 | 0.33, 0.14 |
| office_frame60 | 2 | on the person, 1.10 | default | core with OpenCV's crop against Meta (the control) | 0.05 | 0.06 | 0.01 | 0.00 | 0.03 | 0.00 | 0.01 | 0.01, 0.02 |
| office_frame60 | 2 | on the person, 1.10 | default | Meta as released against Meta in float32 (the floor) | 0.53 | 0.46 | 0.17 | 0.04 | 0.19 | 0.15 | 0.19 | 0.23, 0.34 |
| office_frame60 | 2 | on the person, 1.10 | fov | core against Meta | 2.30 | 2.41 | 0.94 | 0.13 | 0.27 | 2.06 | 0.74 | 1.64, 0.42 |
| office_frame60 | 2 | on the person, 1.10 | fov | core against Meta, body decoder only | 0.84 | 0.52 | 0.11 | 0.05 | 0.22 | 0.04 | 0.08 | 0.31, 0.12 |
| office_frame60 | 2 | on the person, 1.10 | fov | core with OpenCV's crop against Meta (the control) | 0.11 | 0.12 | 0.05 | 0.00 | 0.03 | 0.12 | 0.01 | 0.25, 0.05 |
| office_frame60 | 2 | on the person, 1.10 | fov | Meta as released against Meta in float32 (the floor) | 0.74 | 0.69 | 0.20 | 0.05 | 0.18 | 0.33 | 0.13 | 0.58, 0.37 |
| office_frame60 | 3 | on the person, 3.23 | default | core against Meta | 5.57 | 4.31 | 0.64 | 0.40 | 0.77 | 0.39 | 0.81 | 1.35, 1.74 |
| office_frame60 | 3 | on the person, 3.23 | default | core against Meta, body decoder only | 5.57 | 4.34 | 0.65 | 0.37 | 0.79 | 0.37 | 0.85 | 1.35, 1.77 |
| office_frame60 | 3 | on the person, 3.23 | default | core with OpenCV's crop against Meta (the control) | 0.05 | 0.04 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01, 0.01 |
| office_frame60 | 3 | on the person, 3.23 | default | Meta as released against Meta in float32 (the floor) | 0.56 | 0.45 | 0.08 | 0.01 | 0.08 | 0.06 | 0.11 | 0.06, 0.16 |
| office_frame60 | 3 | on the person, 3.23 | fov | core against Meta | 5.66 | 4.32 | 0.62 | 0.36 | 0.75 | 0.42 | 0.72 | 1.47, 1.90 |
| office_frame60 | 3 | on the person, 3.23 | fov | core against Meta, body decoder only | 5.72 | 4.44 | 0.62 | 0.41 | 0.77 | 0.37 | 0.77 | 1.42, 1.91 |
| office_frame60 | 3 | on the person, 3.23 | fov | core with OpenCV's crop against Meta (the control) | 0.05 | 0.04 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01, 0.01 |
| office_frame60 | 3 | on the person, 3.23 | fov | Meta as released against Meta in float32 (the floor) | 0.54 | 0.45 | 0.08 | 0.01 | 0.07 | 0.07 | 0.12 | 0.06, 0.19 |

Reading it:

- **The control closes the gap.** With OpenCV's crop, core is under the
  floor for five boxes of the six, in both cameras, body-only and with
  hands, on every joint group. So what follows the crop in core's port (the
  backbone, both decoders, the hand path, the rig, the camera) computes what
  Meta's code computes. The sixth is the whole-frame box of the frame with
  three people in it: the control there is above the floor and several times
  under core as shipped. My reading, not tested: one crop holds three
  people in that row, so which body comes out is unstable on either side,
  and the control's crop is not bit-identical to Meta's when the box's scale
  is not a round number.
- **As shipped, core is off Meta in every row, by more than the floor.** The
  table's first row of each group against its last. It is millimetres, and
  all of it is the crop's sampling: core uses `grid_sample` at pixel centres
  and floors the result, Meta uses OpenCV on the 8-bit image.
- **The gap is smallest where the crop is about one model pixel per source
  pixel, and larger both ways.** The body-only rows: the person whose crop
  is close to one-to-one is within a millimetre of Meta with core's own
  sampling, close to the floor; the
  whole-frame box on the large image, shrunk the most, is the furthest
  apart; the smallest person, enlarged the most, is in between. This is what
  a half-pixel difference in convention predicts (no offset at one-to-one),
  and the mean offset of the 2D keypoints points up and left as it would;
  but it is not a pure shift, the pose moves by a similar amount. Six boxes
  are not a curve: the direction is established, the shape is not.
- **A box on the person helps when the frame is large and the person is
  not.** On the large image the box on the person is several times closer to
  Meta than the whole-frame box. It is not a rule for every size: a small
  person in a small frame is enlarged whatever the box.
- **The camera is the same on both sides.** The JSON's `focal_length_px` is
  identical for core and Meta in every row, with no intrinsics given and
  with the same field of view given to each. Core's `fov` input and Meta's
  `cam_int` argument are the same thing.

## Hands and expression

Whether the hand decoder's result was used for each hand, read from what
each side's `run_inference` returns (the final hand pose equals the hand
decoder's output exactly when it was used), with the side of that hand's
crop in source pixels: the quantity Meta's second test compares with
`hand_box_size_thresh`
(`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1317`).

| image | person | camera | side | left hand: decoder used, crop side (px) | right hand: decoder used, crop side (px) | expression all zero |
|---|---|---|---|---|---|---|
| dancing | 0 | default | core | yes, 189 | yes, 200 | yes |
| dancing | 0 | default | core with OpenCV's crop | yes, 186 | yes, 196 | yes |
| dancing | 0 | default | Meta float32 | yes, 186 | yes, 196 | yes |
| dancing | 0 | default | Meta as released | yes, 186 | yes, 196 | yes |
| dancing | 0 | fov | core | yes, 198 | yes, 194 | yes |
| dancing | 0 | fov | core with OpenCV's crop | yes, 194 | yes, 191 | yes |
| dancing | 0 | fov | Meta float32 | yes, 194 | yes, 191 | yes |
| dancing | 0 | fov | Meta as released | yes, 194 | yes, 191 | yes |
| dancing | 1 | default | core | yes, 159 | yes, 186 | yes |
| dancing | 1 | default | core with OpenCV's crop | yes, 157 | yes, 185 | yes |
| dancing | 1 | default | Meta float32 | yes, 157 | yes, 185 | yes |
| dancing | 1 | default | Meta as released | yes, 157 | yes, 185 | yes |
| dancing | 1 | fov | core | yes, 160 | yes, 181 | yes |
| dancing | 1 | fov | core with OpenCV's crop | yes, 159 | yes, 180 | yes |
| dancing | 1 | fov | Meta float32 | yes, 159 | yes, 180 | yes |
| dancing | 1 | fov | Meta as released | yes, 159 | yes, 180 | yes |
| office_frame60 | 0 | default | core | yes, 92 | yes, 91 | yes |
| office_frame60 | 0 | default | core with OpenCV's crop | yes, 91 | yes, 90 | yes |
| office_frame60 | 0 | default | Meta float32 | yes, 91 | yes, 91 | yes |
| office_frame60 | 0 | default | Meta as released | yes, 91 | yes, 90 | yes |
| office_frame60 | 0 | fov | core | yes, 91 | yes, 90 | yes |
| office_frame60 | 0 | fov | core with OpenCV's crop | yes, 90 | yes, 89 | yes |
| office_frame60 | 0 | fov | Meta float32 | yes, 90 | yes, 89 | yes |
| office_frame60 | 0 | fov | Meta as released | yes, 90 | yes, 89 | yes |
| office_frame60 | 1 | default | core | no, 100 | no, 72 | yes |
| office_frame60 | 1 | default | core with OpenCV's crop | no, 100 | no, 72 | yes |
| office_frame60 | 1 | default | Meta float32 | no, 100 | no, 72 | yes |
| office_frame60 | 1 | default | Meta as released | no, 100 | no, 72 | yes |
| office_frame60 | 1 | fov | core | no, 108 | no, 73 | yes |
| office_frame60 | 1 | fov | core with OpenCV's crop | no, 108 | no, 73 | yes |
| office_frame60 | 1 | fov | Meta float32 | no, 108 | no, 73 | yes |
| office_frame60 | 1 | fov | Meta as released | no, 108 | no, 73 | yes |
| office_frame60 | 2 | default | core | yes, 92 | yes, 97 | yes |
| office_frame60 | 2 | default | core with OpenCV's crop | yes, 92 | no, 97 | yes |
| office_frame60 | 2 | default | Meta float32 | yes, 92 | no, 97 | yes |
| office_frame60 | 2 | default | Meta as released | yes, 92 | no, 97 | yes |
| office_frame60 | 2 | fov | core | yes, 90 | yes, 91 | yes |
| office_frame60 | 2 | fov | core with OpenCV's crop | yes, 90 | yes, 92 | yes |
| office_frame60 | 2 | fov | Meta float32 | yes, 90 | yes, 92 | yes |
| office_frame60 | 2 | fov | Meta as released | yes, 90 | yes, 92 | yes |
| office_frame60 | 3 | default | core | no, 43 | no, 28 | yes |
| office_frame60 | 3 | default | core with OpenCV's crop | no, 42 | no, 28 | yes |
| office_frame60 | 3 | default | Meta float32 | no, 42 | no, 28 | yes |
| office_frame60 | 3 | default | Meta as released | no, 42 | no, 28 | yes |
| office_frame60 | 3 | fov | core | no, 42 | no, 30 | yes |
| office_frame60 | 3 | fov | core with OpenCV's crop | no, 42 | no, 29 | yes |
| office_frame60 | 3 | fov | Meta float32 | no, 42 | no, 29 | yes |
| office_frame60 | 3 | fov | Meta as released | no, 42 | no, 29 | yes |

- **Under the constant, the hand decoder is never used.** The smallest
  person's hand crops are under it and no side used the hand decoder, in
  either camera. The fingers there are the body decoder's.
- **Over the constant is not enough.** One person's hand crops are over it
  and no side used the hand decoder: one of the other three tests refused
  it. The output does not say which.
- **Core and Meta disagreed on one hand, and that is the largest hand
  difference in this record.** For one person under the default camera,
  core used the hand decoder for the right hand and Meta did not; core with
  OpenCV's crop agrees with Meta. In that row the body-only distance is
  close to the floor and the full run's is many times it, and the right
  hand's shape differs by centimetres. So a threshold decision near its edge is flipped by the
  crop's sampling alone. Under the other camera the same hand is used by
  every side.
- **A first version of this reading was wrong and is not in the file.** It
  compared a full run with a body-only run and reported the hand decoder as
  used everywhere, because a full run re-runs the body decoder with wrist
  and elbow prompts and the hand pose changes either way. The tool's
  docstring keeps the note.
- **Expression is all zero on every side in every row**, as the code reads
  (`coderef/sam-3d-body/sam_3d_body/models/heads/mhr_head.py:316`). The mesh
  has no mouth and no expression from either implementation.

## What a mesh from core can be trusted for

Priors for reading a mesh-driven render that went wrong, each from the
tables above and no wider than two images.

- **The body, as a motion signal: yes.** Core's body differs from Meta's by
  millimetres, and for scale the model's two releases differ from each other
  by more on one frame ([`2026-10-05_sam3d_body_vith.md`](2026-10-05_sam3d_body_vith.md)).
  A body that is wrong in a render is not explained by core's port.
- **Hands: the position follows the arm; the fingers are the weak part.**
  Whether the fingers come from the hand decoder is a set of threshold tests
  the output does not report, a hand crop under the constant never gets it,
  and near a threshold the answer can differ between two runs that differ by
  a sub-pixel. When fingers are wrong, look at the hand crop's size first.
- **The mouth and the face: never.** Zero by construction.
- **A whole-frame box is the less accurate box on a large frame, and with
  more than one person it is not one person's box at all.**
- **There is no confidence to read.** Neither side returns a per-joint
  visibility or a score; the hand-presence logits exist in Meta's model and
  are unused by both.

Nothing here calls for a change on our side beyond a box or track per
person, which more than one person needs anyway
(`subject_boxes.py::MiniMaxH3SubjectBoxes`). The sampling difference is in
core; this pack patches nothing in core.

## Later the same day: this pack's own node, given Meta's crop

The owner's rule is that no ComfyUI SAM node is wired into a graph of ours,
so the finding above was built on rather than worked around:
`body_pose.py` calls ComfyUI's model code as a library and gives it Meta's
crop, transcribed from Meta's transforms, for the body and for both hands.
[`bench/check_body_pose.py`](../check_body_pose.py) holds it, against
[`bench/fixtures/sam3d_body_meta_reference.json`](../fixtures/sam3d_body_meta_reference.json),
which the tool's `fixture` mode writes from the same Meta runs as the tables
above plus the hash of the body crop Meta's own transform makes of each box.
From that check's run on 2026-10-10, on the CPU in float32, default camera:

| image | person | our body crop against Meta's | our 2D keypoints against Meta's, mean (px) | the floor for that box (px) | hands' decoders used, ours and Meta's |
|---|---|---|---|---|---|
| dancing | 0 | the same bytes | 0.032 | 0.227 | both, both |
| dancing | 1 | the same bytes | 0.028 | 0.126 | both, both |
| office_frame60 | 0 | the same bytes | 0.083 | 0.243 | both, both |
| office_frame60 | 1 | the same bytes | 0.009 | 0.082 | neither, neither |
| office_frame60 | 2 | the same bytes | 0.014 | 0.168 | left only, left only |
| office_frame60 | 3 | the same bytes | 0.007 | 0.079 | neither, neither |

- **Under the floor on all six boxes**, the whole-frame box of the
  three-person frame included. That box was the one the control above did
  not close; the control used ComfyUI's closed-form matrix with OpenCV's
  sampling, and the check shows that matrix is not Meta's on every box (a
  break that swaps it in fails the crop case on one of the six). So the
  reading labelled "not tested" above holds for its second half.
- **The hand that ComfyUI's port decided differently is decided as Meta
  decides it.** Person 2's right hand, default camera.
- ComfyUI's own crop hashes to Meta's on none of the six; it is the check's
  control.

Not covered, as above: the card, half precision, a clip.

## To run it again

The second image is one frame cut from Meta's example video:

    ffmpeg -i coderef/sam-audio/examples/assets/office.mp4 -vf "select='eq(n\,60)'" -fps_mode passthrough office_frame60.png

Then, with the card masked, the tool's own docstring for the Python that
runs Meta's side, and `<store>` the folder holding Meta's release
(`model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt`):

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/compare_sam3d_body_core_against_meta.py \
        --image coderef/sam-3d-body/notebook/images/dancing.jpg --label dancing \
        --boxes whole --boxes-from-mask coderef/sam-3d-body/notebook/images/dancing_mask.png \
        --meta-python <python> --meta-weights <store> --out-dir <dir>
    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/compare_sam3d_body_core_against_meta.py \
        --image office_frame60.png --label office_frame60 \
        --boxes whole --boxes 45,95,190,245 --boxes 320,55,600,345 --boxes 345,100,440,215 \
        --meta-python <python> --meta-weights <store> --out-dir <dir>

`gather` joins the two JSON files into the one beside this record and
`render` prints the tables from it.

## Not done

- The field of view from MoGe on core's side against Meta's own MoGe call.
  The weights were not on disk on the day
  (`ComfyUI/models/geometry_estimation` held none). What is established is
  narrower: given the same field of view, both sides use the same focal
  length.
- A clip. Nothing here is about time: jitter between frames, or
  `SAM3DBody_Smooth`.
- A SAM 3 track as the box source, with its mask conditioning, and a
  tracked person whose mask is empty on a frame.
- Core as the server runs it: half precision, on the card.
