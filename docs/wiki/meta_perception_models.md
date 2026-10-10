# SAM-Audio, PE-AV, SAM 3D Body and Sapiens2: what each gives the masked lane, and what it does not

last updated: 2026-10-10 (first written, from a reading of Meta's code, the three papers and ComfyUI core's SAM 3D Body port; later the same day, core's SAM 3D Body prediction run against Meta's code, a section of candidate preflight flags, and which weights are now on disk)

Written by hand. One claim a line, each with the file that says it. It
carries no numbers of its own: a threshold, a rate or a size is cited by the
line or the command that holds it. Where this page and the code disagree,
the code is right. **Everything here was read, not run**, unless a line
names a record; one record exists so far, for SAM 3D Body. What a run would
have to show for the rest is under "Not run".

The two questions this page was written for come from the masked lane
([`masked_v2v.md`](masked_v2v.md)):

1. When only a face is redrawn, what can show the model the original
   performer's mouth, or score a render's lips against its own track?
2. When a whole person is redrawn, what can carry the movement without
   words, and does it hold with more than one person in the frame?

## The short answers

| question | answer | where |
|---|---|---|
| Does a SAM 3D Body mesh carry the mouth? | **No.** Meta's head sets the jaw and every expression parameter to zero before the rig is posed. A mesh from Meta's model has a neutral face on every frame | "SAM 3D Body", "The face" |
| Where does a mouth on core's mesh come from, then? | A node core added, driven by MediaPipe through a table its own comment calls hand-derived. It is not Meta's and is not in Meta's repository | the same section |
| Does PE-AV score lip sync? | **No.** Its audio-to-video outputs are one pooled vector for a whole clip. Nothing in it compares a video frame with an audio frame | "PE-AV" |
| Can SAM-Audio take "the voice of the person under this mask"? | Yes, it is a designed case, and Meta calls it the weaker prompt. Meta's own code disagrees with itself about which way the mask goes | "SAM-Audio", "The visual prompt" |
| Does SAM 3D Body follow more than one person? | Meta's pipeline is one image at a time with no identity between frames. Identity is whatever the caller's boxes or track carry | "SAM 3D Body", "More than one person" |
| What published by Meta does read a mouth? | Sapiens2: the lip, teeth and tongue classes of its part segmentation, which this pack already loads, and its keypoint model, which nothing here runs yet | "Sapiens2" |

## Where each was read

| what | checkout | revision | weights here |
|---|---|---|---|
| SAM-Audio | `coderef/sam-audio` | `bb4c699` | the large model, one `-tv` variant and the judge, in the owner's model store since 2026-10-10 as Meta's original files; nothing here loads them yet |
| PE-AV and PE-A-Frame | upstream `facebookresearch/perception_models` at `3e352cc`, not under `coderef/`; a port is in `coderef/transformers/src/transformers/models/pe_audio_video` and in the ComfyUI venv's `transformers` | | none |
| SAM 3D Body | `coderef/sam-3d-body` | `b5c765a` | both releases, with Meta's originals and the rig file, in the owner's model store; `../../bench/results/2026-10-05_sam3d_body_conversion.md` |
| core's SAM 3D Body port | the ComfyUI checkout, `comfy_extras/nodes_sam3d_body.py`, `comfy_extras/sam3d_body/`, `comfy/ldm/sam3d_body/` | `0df64eb24` | the converted files above |
| Sapiens2 | `coderef/sapiens2` | `7e5bae8` | part segmentation and matting; the keypoint model since 2026-10-10, as Meta's original file, which nothing here loads yet |

The papers: SAM Audio is arXiv 2512.18099 and SAM 3D Body is arXiv
2602.15989, both linked from their READMEs; the PE-AV report is linked from
`coderef/sam-audio/README.md`. Sizes, gating and licence of a file on the
Hub are read with `hf download --dry-run <repo>` or the Hub's model API, not
from this page.

## SAM-Audio

A flow-matching model that takes a mixture and returns the part a prompt
names and the rest (`coderef/sam-audio/sam_audio/model/model.py::separate`).

- **A sung voice from a full mix is a designed case.** Meta's own evaluation
  on a studio multitrack set prompts with the stem's bare name
  (`coderef/sam-audio/eval/dataset/musdb.py:29`). The paper reports
  listeners preferring it to the Demucs family on that set (section on
  instrument separation). No comparison of the two has been run here.
- **Three prompts, usable together**
  (`coderef/sam-audio/sam_audio/processor.py::SAMAudioProcessor.__call__`):
  text, time spans, and a masked video.
- **Text is a lowercase noun or verb phrase**, not a sentence
  (`coderef/sam-audio/README.md:74`). The encoder is the one named by
  `T5EncoderConfig.name` in `coderef/sam-audio/sam_audio/model/config.py`.
- **A span is a sign and two times in seconds**
  (`coderef/sam-audio/sam_audio/processor.py::Batch.process_anchors`): where the sound is, or where it
  is not. The model can also predict spans from the text
  (`coderef/sam-audio/sam_audio/model/model.py::predict_spans`), with a second model named by
  `SAMAudioConfig.span_predictor`.
- **Any input is resampled and mixed down to one channel**
  (`coderef/sam-audio/sam_audio/processor.py::batch_audio`; the rate is `DACVAEConfig.sample_rate`).
  The output is the same rate, one channel, a target and a residual. A
  stereo track comes back mono.
- **The released code separates in one pass.** The paper describes long
  audio as overlapping windows merged at every step and says training clips
  were capped; `coderef/sam-audio/sam_audio/model/model.py::separate` has no windows. A whole song is cut by
  the caller.

### The visual prompt

- **What it is given:** the video's frames multiplied by a mask, as
  `masked_videos`. Each frame is resized to a square with its aspect not
  kept (`coderef/sam-audio/sam_audio/model/vision_encoder.py::PerceptionEncoder.get_transform`)
  and becomes one vector from an image encoder
  (`PerceptionEncoderConfig.name`); the frame nearest each audio step is the
  one used (`coderef/sam-audio/sam_audio/processor.py::load_video`). In Meta's example the description
  is left empty and the mask comes from SAM 3 prompted with a phrase
  (`coderef/sam-audio/examples/visual_prompting.ipynb`).
- **Which way the mask goes is contradicted inside Meta's code.** The
  helper the README shows keeps the pixels where the mask is zero, so it
  blacks out the object (`coderef/sam-audio/sam_audio/processor.py:204`). The paper says the model sees
  the frames times the mask, the object kept (its section on the visual
  prompt), and Meta's evaluation loader does that
  (`coderef/sam-audio/eval/dataset/sam_audio_bench.py:118`). Upstream's main
  branch carried both lines on 2026-10-10. A run here builds the masked
  video the evaluation's way and tries the helper's way beside it.
- **Meta says it is the weaker prompt.** The paper's conclusion calls it
  noticeably less effective than text, and its results section says a mask
  of a person is ambiguous because a person makes more than one sound. It
  names one case where the mask is what decides: several speakers a text
  cannot tell apart. Use the mask with a text, not in place of one.
- The `-tv` checkpoints are the ones Meta recommends for the visual prompt
  (`coderef/sam-audio/README.md:124`).

### What loading it brings

Loading a checkpoint as released also loads the rankers and the span
predictor its `config.json` names (`coderef/sam-audio/sam_audio/model/model.py:94`, `:95`, `:97`): an
image-audio model, an audio-text model, Meta's judge and an audio-frame
model. They serve reranking and span prediction only. A keyword of the same
name set to `None` at load drops each (`coderef/sam-audio/sam_audio/model/base.py:50`).
Whether the largest checkpoint fits the card with them dropped is not
measured; it does not fit beside a loaded H3 server.

Licence: the SAM License in `coderef/sam-audio/LICENSE`. Its dependencies
are in `coderef/sam-audio/pyproject.toml`; the ComfyUI venv has none of the
audio ones (checked 2026-10-10 with `importlib.util.find_spec`), so it runs
from an environment of its own, not the server's.

## PE-AV

A contrastive encoder that puts audio, video, audio with video, and text in
one space (upstream, the class `PEAudioVisual` in `pe.py` under `core/audio_visual_encoder`; the files named in this section are that directory's and are not on disk in this repo).

- **Every embedding is pooled over the clip.** Each tower's output head is
  applied to its class token (`PEAudioVisual.forward`; `Transformer.forward` in its `transformer.py`
  returns the first position as the pooled output). Audio against video is one number for a clip.
- **Nothing in it is trained or evaluated on lip sync.** The paper's tasks
  are retrieval and classification; it mentions lip movement once, in its
  introduction, as motivation.
- **The frame-level model is audio against text.** PE-A-Frame scores each
  audio step against a description and returns the spans where it holds
  (`PEAudioFrame.forward`). No video enters it.
- The video tower takes every frame, or a fixed sample of them for the
  checkpoints named for it (`VideoProcessor._sample_frames` in its `transforms.py`),
  each resized to a square.

So it does not replace a measure of the mouth as shipped. Scoring a face
crop against an isolated voice over short windows is an idea, not a
feature, and has an entry condition: the same clip with its audio shifted by
a few frames must score lower. Until that control is run, a number from it
means nothing.

What depends on it: SAM-Audio's span predictor is a PE-A-Frame model
(`coderef/sam-audio/sam_audio/model/model.py:97`) and SAM-Audio's judge is
built on PE-AV's transformer class (`coderef/sam-audio/sam_audio/model/judge.py:8`).
SAM-Audio's own video encoder is the image model, not PE-AV
(`coderef/sam-audio/sam_audio/model/vision_encoder.py:86`). SAM 3D Body uses neither. Whether
SAM 3's backbone is from the same family was not confirmed.

## SAM 3D Body

One image and a box in, one posed body mesh out, on Meta's Momentum Human
Rig. `../../bench/results/2026-10-05_sam3d_body_vith.md` is what has been
run here: one frame through both releases behind core's nodes.

### The face

- **The mesh has no mouth movement and no expression.** The head predicts
  expression components (`MHRHead.num_face_comps`,
  `coderef/sam-3d-body/sam_3d_body/models/heads/mhr_head.py:53`) and then
  multiplies them by zero (`:316`), and zeroes the jaw (`:306`). The
  estimator returns them as `expr_params`
  (`coderef/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py:210`), always zero. Core's port
  does the same (`comfy/ldm/sam3d_body/mhr/mhr_head.py::MHRHead.forward`).
- **Core's `SAM3DBody_FaceExpression` is core's own.** It runs MediaPipe's
  face landmarker on the frame and maps its blendshapes to the rig's
  expression axes with a table its comment says was derived by visual
  inspection, because the rig's axes ship unnamed
  (`comfy_extras/sam3d_body/face_expression.py`, the comment above the
  table). It reads the picture, not the audio. The owner does not build on
  it (2026-10-10).

### Hands

- A second decoder runs on a crop of each hand, found by a box head in the
  body decoder, the left one mirrored
  (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py::run_inference`).
- **Its result replaces the body decoder's only when four tests pass**: the
  wrist rotation the two decoders imply agrees, the hand box is large
  enough, every hand keypoint falls inside the crop, and the two wrist
  positions agree (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1314`, `:1317`, `:1331`, `:1347`). Then
  wrists and elbows go back in as keypoint prompts for one more body pass.
- **The paper says more than the code does.** It says an uncertainty on the
  hand box turns the hand decoder off for occluded hands. The released code
  computes those logits (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1158`) and never reads them; core
  drops the weights at load (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Loader.execute`).
  An occluded hand is gated by the four tests alone.
- **Run, on two images** ([the record](../../bench/results/2026-10-10_sam3d_body_core_against_meta.md), "Hands and
  expression"): a hand whose crop is under the size constant never got the
  hand decoder on either side; a hand over it was still refused on both
  sides once; and on one hand core and Meta decided differently, with the
  crop's sampling the only difference between them. Neither output says
  which hands were refined.
- How reliable Meta says hands are: the benchmark row in
  `coderef/sam-3d-body/README.md` ("SAM 3D Body checkpoints") and the
  paper's per-case table, which has rows for crossed fingers, held objects
  and self-occluded hands.

### More than one person

- **Meta's pipeline has no identity across frames.** It processes one image
  (`coderef/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py::process_one_image`). Boxes come from the
  caller, from a detector, or are the whole image when there is neither
  (`:125`). Its two detectors are a COCO person detector, whose boxes are
  sorted by coordinate (`coderef/sam-3d-body/tools/build_detector.py:133`), and SAM 3 prompted
  with one word, each box enlarged (`:40`, `:49`).
- **In core, identity is the index.** With a SAM 3 track it is the track's
  object index; with detector boxes it is the order of the list on that
  frame. `SAM3DBody_Smooth` and the face node both treat the index as one
  person through the clip (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Smooth.execute`).
- **With neither a track nor boxes, core makes one box of the whole frame**
  (`SAM3DBody_Predict.execute`). Two people in the frame are then one crop,
  and which of them comes out is not defined.
- **A tracked person with an empty mask on a frame gets a whole-frame box,
  not no box** (`comfy_extras/sam3d_body/utils.py::_bbox_from_mask`), and
  the empty mask is passed as a confident one
  (`comfy_extras/sam3d_body/utils.py::run_batched_single_chunk`). Read, not run: expect a body on
  frames where that person is absent, and look at the mesh there.

### Core's port against Meta's inference

What the checks hold: the key mapping
(`../../bench/check_sam3d_body_conversion.py`) and the ViT-H backbone's
forward against Meta's file (`../../bench/check_sam3d_body_vith.py`).

**The whole prediction has been run against Meta's on two public images**
([`2026-10-10_sam3d_body_core_against_meta.md`](../../bench/results/2026-10-10_sam3d_body_core_against_meta.md),
by `../../bench/compare_sam3d_body_core_against_meta.py`), on the CPU in
float32. What it found, in words; the record has the tables:

- **Given the same crop, core computes what Meta computes.** With core's
  one crop-sampling function replaced by OpenCV's call as Meta makes it,
  core is closer to Meta than Meta's two precisions are to each other, for
  every box but one that held three people.
- **As shipped, core is off Meta by a small amount, and all of it is the
  crop's sampling.** The gap is smallest when the person's crop is about the
  model's input size in source pixels, and grows when a large crop is shrunk
  (a whole-frame box on a large frame) and when a small person is enlarged.
- **A box on the person is the better box on a large frame**, and with more
  than one person it is the only box that means one person. This pack makes
  one from a tracked mask: `subject_boxes.py::MiniMaxH3SubjectBoxes`.
- **The camera is the same on both sides** given the same field of view.
- Not covered by that run: the card, half precision, a track with its mask,
  a clip, MoGe's field of view.

The differences found by reading, with what the run says of each:

| stage | Meta | core | measured? |
|---|---|---|---|
| camera | the demo estimates the field of view with MoGe-2 by default (`coderef/sam-3d-body/demo.py:148`, `coderef/sam-3d-body/tools/build_fov_estimator.py::run_moge`), and the paper does too; without an estimator the focal length is the image diagonal (`coderef/sam-3d-body/sam_3d_body/data/utils/prepare_batch.py:66`) | `fov` left at its default is Meta's fallback, not Meta's default (`comfy/ldm/sam3d_body/utils.py::prepare_batch`). Core has the node that matches Meta: `comfy_extras/nodes_moge.py`, the one whose docstring names `SAM3DBody_Predict`. The tooltip's angle is the diagonal one, not the vertical one the input takes | the same focal length on both sides, with no intrinsics and with one field of view given to both; MoGe's own estimate not run |
| precision | the backbone in the type the config names, then back to full precision for the decoder (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:163`, `:165`, `:1103`) | half precision through the decoder on this card (`comfy/ldm/sam3d_body/model/model.py::forward_pose_branch`) | on one frame, negligible: `../../bench/results/2026-10-05_sam3d_body_vith.md`, item 1 |
| the crop | the same box rule (`coderef/sam-3d-body/sam_3d_body/data/transforms/common.py:110`, `:229`), sampled with OpenCV (`:305`) | the same rule, sampled with `grid_sample` at pixel centres, then floored (`comfy/ldm/sam3d_body/utils.py::warp_affine_batched`) | yes: it is the whole difference between the two |
| the mask | off unless asked (`coderef/sam-3d-body/demo.py:184`) | on whenever a track is wired, as a confident mask | no |
| the detector | inside the pipeline | none; the caller wires boxes or a track | not applicable |
| hands | the four tests above | the same tests and constants (`comfy/ldm/sam3d_body/model/model.py::run_inference`), both hands in one batch | the same decision on every hand but one, where the crop's sampling flipped it |
| face, extra face keypoints, smoothing | none of the three | all three are core's. `SAM3DBody_Smooth` filters every parameter in time, hands and expression included, and backs off when the root turns fast (`comfy_extras/nodes_sam3d_body.py::SAM3DBody_Smooth`) | no |

## Sapiens2

The Meta model that does read a mouth, two ways:

- **Part segmentation has classes for the lips, the teeth and the tongue**
  (`coderef/sapiens2/docs/SEG.md`, the class list). This pack loads it
  (`../../sapiens2_parts.py`) and the masked lane's part menu is built on
  it.
- **The keypoint model's layout includes points on the lips**
  (`coderef/sapiens2/docs/POSE.md`;
  `coderef/sapiens2/sapiens/pose/configs/_base_/keypoints308.py`). SAM 3D
  Body takes its keypoints from a mapping of the same size that its comment
  names after Sapiens (`coderef/sam-3d-body/sam_3d_body/models/heads/mhr_head.py::mhr_forward`); whether the two orders
  match was not checked. Its weights are in the owner's model store and
  nothing here runs it yet.

Both read the original performer's mouth from the picture, which is the
signal a face-only swap is missing: the model hears the frozen track and is
shown nothing of the mouth that made it.

## Candidate preflight flags

Signals a mesh-driven render could be checked on before it samples
([`masked_v2v.md`](masked_v2v.md), "The rule: data before a render, and the
same data after"). Each is readable from core's pose data
(`comfy_extras/sam3d_body/utils.py::run_batched_single_chunk` lists what it
holds per person per frame). None is a rule until a preview saves that data;
`../../bench/capture_masked_run.py` is where a rule would live.

| flag | read from | what it predicts | state |
|---|---|---|---|
| a hand crop under Meta's size constant | the width of `lhand_bbox` or `rhand_bbox`, against `hand_box_size_thresh` (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1317`; core's copy is in `comfy/ldm/sam3d_body/model/model.py::run_inference`) | the fingers on that frame are the body decoder's, never the hand decoder's | confirmed on one frame, both sides: the record's "Hands and expression" |
| a person's `bbox` equal to the whole frame while a track is wired | `bbox` against `image_size` | that person's mask was empty on the frame and core substituted the frame (`comfy_extras/sam3d_body/utils.py::_bbox_from_mask`); a body is drawn from whatever is there | read, not run |
| one whole-frame box and more than one person in the frame | the box, and the tracker's person count | one crop holds everyone; which person comes out is not defined | read; the one such box in the record is also the only one where the control did not reach the floor |
| a keypoint outside the frame | `pred_keypoints_2d` against `image_size` | a joint the model placed where it saw nothing | read, not run |

**There is no confidence to read.** Neither Meta's code nor core returns a
score for a detection, a visibility for a joint, or whether a hand was
refined. Meta's model has hand-presence logits; its inference never reads
them and core drops their weights. A flag here is geometry, not certainty.

Not measured, so not a flag: the size of a face below which Sapiens2's lip,
teeth and tongue classes stop being reliable.

## Not run

Each line is a claim above that only a run settles.

- SAM-Audio on any audio of ours: whether it fits the card, how a mask of
  one singer does against the word alone, and which mask direction is right.
- The shift control that would let PE-AV be tried as a sync score.
- Core's SAM 3D Body prediction with MoGe's field of view against Meta's own
  MoGe call, and core as the server runs it (the card, half precision).
- A tracked person leaving the frame in core's predict node.
- The Sapiens2 keypoint model.
