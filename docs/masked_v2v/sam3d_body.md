# SAM 3D Body in the masked lane: the body-model video, and what it cannot show

**SAM 3D Body gives the lane one thing: a posed body mesh of the tracked person on every frame, drawn as a video, from
which the head reference of every face, head or hair load is cut.** It carries where the head and body are and which
way they turn. It has no mouth and no expression: its face is a neutral modelled face on every frame. It runs in every
preview through this pack's own nodes, `MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose` and `MiniMaxH3BodyMeshVideo`
(`../../body_pose.py`, since 2026-10-10), on the DINOv3 release (`h3_config.MASKED_JOB_BODY_MODEL`), in float32. The
overview of every perception model is [`perception_models.md`](perception_models.md); how the reference is cut and
drawn, and what it carried in renders, is [`renders_by_form.md`](renders_by_form.md).

## How it is wired

- **In a job.** `../../workflows/masked_job_graph.py` puts the three nodes in every preview graph, after
  `MiniMaxH3SubjectBoxes` (`../../subject_boxes.py`), which gives one box a frame round the tracked mask and no box on
  a frame the mask is empty on (`subject_boxes.py::frame_boxes`, the floor `subject_boxes.py::SMALLEST_MASK_PX`).
  `MiniMaxH3BodyPose` also takes the person's mask, so a box that holds two people poses the tracked one
  (2026-10-12, `e5242d8a`; [the record](../../bench/results/2026-10-12_body_model_posed_the_other_person.md)).
- **What it writes.** The mesh video, and a pose table (`body_pose.py::pose_table`, as text and as a file) with each
  body's box, its source, each hand's crop and whether its decoder was used, and the 3D keypoints in camera space. The
  capture reads the table (`pose=` on `--mask`) for its preflight (`bench/capture_masked_run.py::flag_pose`) and for
  the body's state per frame (`bench/capture_masked_run.py::pose_state`; [`state_signals.md`](state_signals.md)).
- **What reads the video.** `bench/head_reference.py` cuts it to the head and draws the pose model's face points on
  it ([`sapiens2_pose.md`](sapiens2_pose.md)); the cut video goes to the Masked Source as `motion_video`
  ([`pieces.md`](pieces.md)). `bench/shot_to_load.py` refuses a reference when the preview made no pose pass for the
  subject.
- **The other release.** The ViT-H backbone loads through the same loader (`../../sam3d_body_vith.py`, held by
  `../../bench/check_sam3d_body_vith.py`); no job uses it ([`perception_trials.md`](perception_trials.md)).

## The face

- **The mesh has no mouth movement and no expression.** The head predicts expression components
  (`MHRHead.num_face_comps`, `coderef/sam-3d-body/sam_3d_body/models/heads/mhr_head.py:53`) and then multiplies them by
  zero (`:316`), and zeroes the jaw (`:306`). The estimator returns them as `expr_params`
  (`coderef/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py:210`), always zero. Core's port does the same
  (`comfy/ldm/sam3d_body/mhr/mhr_head.py::MHRHead.forward`).
- **The face is neutral, not blank.** With the body-model video taken away, and the sentences about it, her head held
  nearly still; with it her head followed the source's
  ([`2026-10-11_kitchen_shot_head_and_eyes_two_tests.md`](../../bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md),
  one shot, one seed). What the eyes and mouth get is drawn on top from the pose model.
- **Core's `SAM3DBody_FaceExpression` is core's own, and is not used.** It runs MediaPipe's face landmarker on the
  frame and maps its blendshapes to the rig's expression axes with a table its comment says was derived by visual
  inspection, because the rig's axes ship unnamed (`comfy_extras/sam3d_body/face_expression.py`, the comment above the
  table). A reference driven by it broke the mouth, the eyes and the iris when tried, and it is not proposed again
  ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-18).

## Hands

- A second decoder runs on a crop of each hand, found by a box head in the body decoder, the left one mirrored
  (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py::run_inference`).
- **Its result replaces the body decoder's only when four tests pass**: the wrist rotation the two decoders imply
  agrees, the hand box is large enough, every hand keypoint falls inside the crop, and the two wrist positions agree
  (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1314`, `:1317`, `:1331`, `:1347`). Then wrists and
  elbows go back in as keypoint prompts for one more body pass.
- **The paper says more than the code does.** It says an uncertainty on the hand box turns the hand decoder off for
  occluded hands. The released code computes those logits
  (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1158`) and never reads them; core drops the weights
  at load (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Loader.execute`). An occluded hand is gated by the four tests
  alone.
- **Run, on two images** ([the record](../../bench/results/2026-10-10_sam3d_body_core_against_meta.md), "Hands and
  expression"): a hand whose crop is under the size constant never got the hand decoder on either side; a hand over
  it was still refused on both sides once; and on one hand core and Meta decided differently, with the crop's sampling
  the only difference between them. Our node's table says which hands were refined; neither upstream does.
- How reliable Meta says hands are: the benchmark row in `coderef/sam-3d-body/README.md` ("SAM 3D Body checkpoints")
  and the paper's per-case table, which has rows for crossed fingers, held objects and self-occluded hands.

## More than one person

- **Meta's pipeline has no identity across frames.** It processes one image
  (`coderef/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py::process_one_image`). Boxes come from the caller, from a
  detector, or are the whole image when there is neither (`:125`). Its two detectors are a COCO person detector, whose
  boxes are sorted by coordinate (`coderef/sam-3d-body/tools/build_detector.py:133`), and SAM 3 prompted with one word,
  each box enlarged (`:40`, `:49`).
- **In this pack, identity is the track's.** The boxes come from the tracked mask and the mask goes in with them, so
  the body is the tracked person's ([`sam31.md`](sam31.md) says how the track is chosen).
- **In core, identity is the index.** With a SAM 3 track it is the track's object index; with detector boxes it is the
  order of the list on that frame. `SAM3DBody_Smooth` and the face node both treat the index as one person through the
  clip (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Smooth.execute`). With neither a track nor boxes, core makes one
  box of the whole frame (`SAM3DBody_Predict.execute`), and with two people in the frame which of them comes out is not
  defined. A tracked person with an empty mask on a frame gets a whole-frame box, not no box
  (`comfy_extras/sam3d_body/utils.py::_bbox_from_mask`), and the empty mask is passed as a confident one
  (`comfy_extras/sam3d_body/utils.py::run_batched_single_chunk`). Read, not run. Our boxes node gives no box there.

## Core's port against Meta's inference

What the checks hold: the key mapping (`../../bench/check_sam3d_body_conversion.py`), the ViT-H backbone's forward
against Meta's file (`../../bench/check_sam3d_body_vith.py`), and our nodes' crop against the crop Meta's own code made,
bit for bit, with ComfyUI's crop as the control (`../../bench/check_body_pose.py`).

**The whole prediction has been run against Meta's on two public images**
([`2026-10-10_sam3d_body_core_against_meta.md`](../../bench/results/2026-10-10_sam3d_body_core_against_meta.md), by
`../../bench/compare_sam3d_body_core_against_meta.py`, whose docstring holds the recipe for Meta's side), on the CPU in
float32. What it found; the record has the tables:

- **Given the same crop, core computes what Meta computes.** With core's one crop-sampling function replaced by OpenCV's
  call as Meta makes it, core is closer to Meta than Meta's two precisions are to each other, for every box but one
  that held three people.
- **As shipped, core is off Meta by a small amount, and all of it is the crop's sampling.** The gap is smallest when the
  person's crop is about the model's input size in source pixels, and grows when a large crop is shrunk (a whole-frame
  box on a large frame) and when a small person is enlarged.
- **A box on the person is the better box on a large frame**, and with more than one person it is the only box that
  means one person. That is why the pack has `subject_boxes.py::MiniMaxH3SubjectBoxes`.
- **The camera is the same on both sides** given the same field of view.

**What we run instead** (2026-10-10, `f8538b1c`): this pack does not wire ComfyUI's SAM nodes into its graphs. Our three
nodes call ComfyUI's model code and its rasteriser as a library, since the run above shows the model is right; the crop
is Meta's, transcribed, for the body and for both hands. On the two samples our keypoints sit inside Meta's own
precision floor on every box, the three-person one included, and every hand's decoder is used exactly where Meta's code
used it. On the card the same holds in float32, which is what the loader loads in; in the half precision ComfyUI's own
loader picks there the body is outside that floor on most boxes (`42aaa7ee`; the record's last sections).

A second session read the nodes cold the same day and broke them with a stand-in model (its cases are in the check). A
box with no area, off the frame or not a number is now dropped (`body_pose.py::unusable`) and named under
`boxes_refused`; a one-pixel speck of mask gives no box (`subject_boxes.py::SMALLEST_MASK_PX`). Still true:
`first_source_frame` is a label the caller supplies, and nothing ties it to where a loader started.

**The `marked` style** (2026-10-10, `86febba3`): the grey body with the face side of the head, the back of it and each
hand in flat colours (`body_pose.py::MARK_COLOURS`). A text that uses these frames takes the sentence naming the colours
from the node's second output, `legend` (`body_pose.py::legend`). Which vertex is which part is kept in
`body_marks.json`. No job uses this style; the reference form is the plain mesh with the face points drawn
([`renders_by_form.md`](renders_by_form.md)).

The differences found by reading, with what the run says of each:

| stage | Meta | core | measured? |
|---|---|---|---|
| camera | the demo estimates the field of view with MoGe-2 by default (`coderef/sam-3d-body/demo.py:148`, `coderef/sam-3d-body/tools/build_fov_estimator.py::run_moge`); without an estimator the focal length is the image diagonal (`coderef/sam-3d-body/sam_3d_body/data/utils/prepare_batch.py:66`) | `fov` left at its default is Meta's fallback, not Meta's default (`comfy/ldm/sam3d_body/utils.py::prepare_batch`); ours takes `fov_degrees` (`body_pose.py::FOV_DEFAULT`) | the same focal length on both sides; MoGe's own estimate not given to the mesh ([`depth.md`](depth.md), "The camera for the body mesh") |
| precision | the backbone in the type the config names, then back to full precision for the decoder (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:163`, `:165`, `:1103`) | half precision through the decoder on this card (`comfy/ldm/sam3d_body/model/model.py::forward_pose_branch`) | outside Meta's floor on most boxes; ours loads in float32 |
| the crop | the same box rule (`coderef/sam-3d-body/sam_3d_body/data/transforms/common.py:110`, `:229`), sampled with OpenCV (`:305`) | the same rule, sampled with `grid_sample` at pixel centres, then floored (`comfy/ldm/sam3d_body/utils.py::warp_affine_batched`) | yes: it is the whole difference; ours is Meta's |
| the mask | off unless asked (`coderef/sam-3d-body/demo.py:184`) | on whenever a track is wired, as a confident mask | ours passes the tracked mask |
| the detector | inside the pipeline | none; the caller wires boxes or a track | not applicable |
| hands | the four tests above | the same tests and constants (`comfy/ldm/sam3d_body/model/model.py::run_inference`), both hands in one batch | the same decision on every hand but one, where the crop's sampling flipped it |
| face, extra face keypoints, smoothing | none of the three | all three are core's (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Smooth` filters every parameter in time) | no; none is in our graphs |

## Candidate preflight flags

What the capture checks on a pose table before its mesh is used as a motion video, all in
`bench/capture_masked_run.py::flag_pose` since 2026-10-10 (`252b6f8e`), read from `MiniMaxH3BodyPose`'s table:

| flag | what it predicts | state |
|---|---|---|
| `hand_not_refined` | the fingers on that frame are the body decoder's, never the hand decoder's (a crop under Meta's `hand_box_size_thresh`, `coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1317`, is one cause) | built; confirmed on one frame, both sides, in the record's "Hands and expression" |
| `pose_fitted_to_the_whole_frame` | one crop holds everyone; which person comes out is not defined. Top level when another subject is in the frame | built; for our graphs a whole-frame box comes only from a box given as the whole frame |
| `several_bodies_under_one_name`, `no_pose_where_the_subject_is` | the table and the track disagree on who is there | built |
| `pose_mostly_outside_the_frame` | joints the model placed where it saw nothing (`capture_masked_run.POSE_OUTSIDE`, reasoned) | built |

**There is no confidence to read.** Neither Meta's code nor core returns a score for a detection, a visibility for a
joint, or whether a hand was refined; our node recovers the last of these from the model's own return and nothing else.
Meta's model has hand-presence logits; its inference never reads them and core drops their weights. A flag here is
geometry, not certainty.

## Limits

- No mouth, no expression, no hair, clothing or held things in the mesh.
- A head seen from behind still gets a nose point, because the model completes what it cannot see
  ([`state_signals.md`](state_signals.md), "Which way the head faces").
- The grey head's generic face and pose sit a few pixels off the source's features; the fix is in the drawing, not the
  box or the camera (measured on 45 references, 2026-10-18:
  [`2026-10-18_feature_drift_finding.md`](../../internal/gemini/2026-10-18_feature_drift_finding.md)).
- Not run: a tracked person leaving the frame in core's predict node; the mesh with MoGe-2's field of view.
