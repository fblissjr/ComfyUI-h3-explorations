# SAM-Audio, PE-AV, SAM 3D Body and Sapiens2: what each gives the masked lane, and what it does not

last updated: 2026-10-11 (the section "What is on disk, what loads it, and what to try next", with depth and geometry models that are not Meta's perception stack; dated notes where the keypoint model's first run overtook a sentence). Before that, 2026-10-10 (first written, from a reading of Meta's code, the three papers and ComfyUI core's SAM 3D Body port; later the same day, core's SAM 3D Body prediction run against Meta's code, a section of candidate preflight flags with the tool each belongs in, where the probes are, and which weights are now on disk; then this pack's own body pose nodes, which replace ComfyUI's three in our graphs; a cold read's fixes to them: a box that is not a person gets no body; a `marked` drawing style))

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
| What published by Meta does read a mouth? | Sapiens2: the lip, teeth and tongue classes of its part segmentation, which this pack already loads, and its keypoint model, which no node runs (*2026-10-11: run once as a one-off on eight frames; the next section*) | "Sapiens2" |
| Which of these models are on disk, and which does any code load? | The tracker, the part model and the body model are loaded by the masked lane; the keypoint model has been tried once; nothing loads the rest | "What is on disk, what loads it, and what to try next" |

## What is on disk, what loads it, and what to try next

Written 2026-10-11 by a session that read the code and the store that day
and ran none of the models. The store is the owner's model folder, one
`facebook_<name>/` folder a Hugging Face repository; list it to see what is
there now, since this table is a reading of one day. "Loaded by" names the
constant or the file, which is the thing to read; a recommendation here is
one session's, and the owner's ruling of the same day sets the order (the
keypoint model first, tried on a handful of plain frames; depth and audio
wait: [`decisions.md`](decisions.md), 2026-10-11).

| folder in the store | what it gives | loaded by | state | recommendation |
|---|---|---|---|---|
| `facebook_sam3.1` | people and things found by a phrase, and one mask followed through a shot | `h3_config.SEGMENTER`, the file `../../bench/convert_sam3_checkpoint.py` writes from it; `../../subject_track.py` | in every masked job | keep. It says where a person is, not who: [`who_is_who.md`](who_is_who.md) |
| `facebook_sapiens2-seg-1b` | a class for every pixel of one person: `sapiens2_parts.CLASS_NAMES` | `h3_config.SAPIENS2["segmentation"]`; `../../sapiens2_parts.py` | in every masked job that takes a part | keep. Its classes are not the things a person would choose: read the note under the table before building a choice on them |
| `facebook_sam-3d-body-dinov3` | a posed body mesh and a table of joints, from one frame and a box | `h3_config.MASKED_JOB_BODY_MODEL`; `../../body_pose.py` | the motion video of a whole-person load | keep for a whole person. It has no face ("SAM 3D Body", "The face"). Whether it helps or hinders a load that redraws only a face and hair is an open question of 2026-10-11 with one render proposed to answer it: the same load with the motion video taken away |
| `facebook_sam-3d-body-vith` | the same model on the other backbone | a loader exists (`../../sam3d_body_vith.py`, held by `../../bench/check_sam3d_body_vith.py`); the job builder names the other file | on disk, used by no job | do not switch without a failing case. One frame through both is all that has been compared: `../../bench/results/2026-10-05_sam3d_body_vith.md` |
| `facebook_sapiens2-pose-1b` | named points on the body, hands and face, the lids and the lips among them | no node. A one-off run: `../../bench/results/2026-10-11_sapiens2_pose_first_frames.md` | tried once, on eight frames | use it to MEASURE: how open each eye is and whether a mouth is open or shut, on the source and on a render. Do not cut a mask from its lip points on a small face, and do not use its pupil points; the record says why. Next for it: frames where the answer is not at either end (a blink, a half-open mouth) |
| `facebook_sapiens2-matting-1b` | a soft edge for the whole person, not for a part | the part node has a slot for it (`sapiens2_parts.MATTE_ARCH`); `h3_config.SAPIENS2["matting"]` is `"none"` | on disk, switched off | leave it off until something reads a matte. The comment beside `h3_config.SAPIENS2` gives the reason: the Masked Source reads the hard part mask, so loading it costs a pass a frame and changes nothing. It becomes worth a try the day the composite's edge is the fault, and then on hair against a wall first. One earlier run loaded it: `../../bench/results/2026-10-05_sapiens2_first_frame.md` |
| `facebook_sapiens2-pointmap-5b` | a position in space for every pixel of a person | nothing | on disk, loaded by nothing | the only model in the store that could say which of two people is in front. Its file is about the size of the card's whole memory (`ls -la` on the folder against `nvidia-smi`), so it cannot sit beside a loaded H3 server. The smaller release of the same head is the one to try first: "Depth and geometry" below |
| `facebook_sam-audio-large`, `facebook_sam-audio-base-tv`, `facebook_sam-audio-judge` | one sound taken out of a mix by a phrase, a time span or a masked video; the judge scores a separation | nothing; the server's environment lacks its dependencies ("SAM-Audio", "What loading it brings") | on disk, loaded by nothing | wait, as ruled. Its first use here would be a voice a person: who is heard on a shot is typed by a person today, and the voice table is made with another model (`../../bench/voice_spans.py`) |
| `facebook_SACo-Gold`, `facebook_SACo-Silver`, `facebook_SACo-VEval`, `facebook_SA-FARI` | annotation sets, not models | `../../bench/sam3_dataset_phrases.py` | read by that one script | nothing to wire |

Not in the store and named on this page: PE-AV (no weights here, "PE-AV"),
MoGe (ComfyUI has a node for it, `comfy_extras/nodes_moge.py`; no weights
here), MediaPipe (ComfyUI's face node uses it; the owner does not build on
that node).

**Looked at and set aside: MoE-ViE** (`facebookresearch/moe_vie`, read on
GitHub on 2026-10-11; nothing fetched). It is a contrastive image and video
encoder scored against text, of the family that ranks a picture against a
caption: one vector for a whole image or clip. It finds nothing in a frame
and follows nothing, so it has no place in making a mask. The one question
it could be pointed at is whether two crops show the same person, and that
road is closed here for a reason that does not depend on the encoder: who is
in a scene is a person's answer to a numbered picture, because appearance
alone could not tell two people of this lane's clips apart
([`who_is_who.md`](who_is_who.md); [`decisions.md`](decisions.md),
2026-10-10).

**The part classes are not things a person would choose.** Seen on
2026-10-11 by one session, on stills of two shots of one clip, in the class
maps of one preview (the per-shot table is in the untracked
`data/fun/poc/segments.json`): one class, apparel, held a hat, a collar,
gloves and a belt together; a whole dress was one class; bare forearms were
labelled hands; on a shot of a dancer many of the classes that appeared did so
on a handful of frames only; and a share of the hair lay
outside the tracked person on every shot. The face and the hair were the two
parts that looked right. So a choice of "what to replace" offered from this
model's classes is safe for a face and for hair, and needs a second look
before it offers clothing, a hat or hands.

### Depth and geometry

Nothing in this pack reads depth today. The question depth would answer here
is narrow: of two masks that touch or cross on a frame, which is nearer the
camera (two people crossing; a hand over a face; somebody walking through the
front of the shot). That is an order between two regions, not a distance.
The owner's rule for the masked lane stands whatever is used: no control
adapter, so depth would be data for planning a mask and never an input to
the video model.

What could give it. The order is one session's recommendation, changed the
same day it was written: it first put the model that needs no download at
the top, and that was an argument about effort, not about which answers the
question best.

| model | where | what it is for | state |
|---|---|---|---|
| `facebook/sapiens2-pointmap-1b` | in the store since 2026-10-11 as `facebook_sapiens2-pointmap-1b`; loaded by nothing | the same head as the 5b in the store, on the backbone size this pack already runs its part model and its keypoint model at | the one to fetch if a person-only reading is wanted. Trained for pictures of people (the card's "Intended Use"), so it is for person against person and a hand against a face, not for a bowl or a telephone. The 5b in the store then serves as the reference the 1b is checked against on the same frames; the two smaller sizes are for speed and nothing here is short of it yet |
| `facebook/hyden-mogev2-metric-point` | in the store since 2026-10-11 as `facebook_hyden-mogev2-metric-point`; loaded by nothing | Meta's variant of MoGe-2: a position in space for every pixel of a whole scene, in real units | fetch first. It is the one model here with two uses: the order between any two masks, things as well as people, and the camera the body mesh wants (the paragraph on MoGe below). For an order inside one frame its real units are not needed; they would matter only for comparing one frame with another |
| `facebook/hyden-da2-relative-depth` | in the store since 2026-10-11 as `facebook_hyden-da2-relative-depth`; loaded by nothing | Meta's sharper-edged variant of Depth Anything V2 Large; relative depth | fetch with it, as a second reading on the same frames. Two models looking at the same pixels are not two independent votes, so agreement proves little; a frame where they DISAGREE is one to look at by eye. Their base models differ (the cards) and their encoder design is the same |
| Depth Anything V2 | already in the ComfyUI checkout's `models/depthanything/` | relative depth of a whole scene | on disk; no code of this pack loads it. The base model of the row above, so it adds nothing once that one is here; worth a run only if neither HyDen checkpoint can be made to load |
| `facebook/hyden-mogev2-surface-normal`, and the Sapiens2 normal models | not on disk | which way each surface faces | no use in this lane that anybody has named |

What fetching does not settle: the code that loads these checkpoints is
Meta's `facebookresearch/metadepth`, which is not under `coderef/` and which
nobody here has read, and no node exists for either. So the first run is a
one-off against that reference, on a handful of frames where the order is
plain by eye, before anything reads its output.

The three `hyden` repositories were read on the Hub on 2026-10-11, before
two of them were fetched: each card calls the encoder a hybrid of a low-resolution
transformer path and a full-resolution convolution path, names
`facebookresearch/metadepth` as the code, and carries a non-commercial
research licence behind an access form. Read a card again before fetching;
this paragraph is one day's reading.

**Why MoGe has come up before.** Not for depth. Meta's own SAM 3D Body demo
estimates the camera's field of view with MoGe-2 before it poses a body, and
without an estimator it falls back to a focal length taken from the image's
diagonal ("Core's port against Meta's inference", the row "camera"). This
pack's body node runs on that fallback, and the comparison against Meta's
code was made with one field of view given to both sides, so MoGe's own
estimate has never been run here ("Not run"). A model of that family would
be the way to give the body mesh the camera Meta's pipeline expects. Whether
that changes a mesh enough to see is not measured, and it matters only while
a mesh is used at all.

## Where each was read

| what | checkout | revision | weights here |
|---|---|---|---|
| SAM-Audio | `coderef/sam-audio` | `bb4c699` | the large model, one `-tv` variant and the judge, in the owner's model store since 2026-10-10 as Meta's original files; nothing here loads them yet |
| PE-AV and PE-A-Frame | upstream `facebookresearch/perception_models` at `3e352cc`, not under `coderef/`; a port is in `coderef/transformers/src/transformers/models/pe_audio_video` and in the ComfyUI venv's `transformers` | | none |
| SAM 3D Body | `coderef/sam-3d-body` | `b5c765a` | both releases, with Meta's originals and the rig file, in the owner's model store; `../../bench/results/2026-10-05_sam3d_body_conversion.md` |
| core's SAM 3D Body port | the ComfyUI checkout, `comfy_extras/nodes_sam3d_body.py`, `comfy_extras/sam3d_body/`, `comfy/ldm/sam3d_body/` | `0df64eb24` | the converted files above |
| Sapiens2 | `coderef/sapiens2` | `7e5bae8` | part segmentation and matting; the keypoint model since 2026-10-10, as Meta's original file, which nothing here loads yet (*2026-10-11: no node loads it; a one-off ran it on eight frames, and the largest pointmap model is in the store too: the section above*) |

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
- Not covered by that run: a track with its mask, a clip, MoGe's field of
  view. The card and half precision were run later the same day for our own
  node: the record's "On the card".

**What we run instead** (2026-10-10; the owner: this pack does not wire
ComfyUI's SAM nodes into its graphs). `../../body_pose.py` holds three nodes,
`MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose` and `MiniMaxH3BodyMeshVideo`.
ComfyUI's model code and its rasteriser are called as a library, since the
run above shows the model is right; the crop is Meta's, transcribed, for the
body and for both hands, and `../../bench/check_body_pose.py` holds it to
the crop Meta's own code made, bit for bit, with ComfyUI's crop as the
control. On the two samples our keypoints sit inside Meta's own precision
floor on every box, the three-person one included, and every hand's decoder
is used exactly where Meta's code used it. On the card the same holds in
float32, which is what the loader loads in; in the half precision ComfyUI's
own loader picks there the body is outside that floor on most boxes. The
record's last sections have the figures. The table that node writes is the input the flags below were
missing.

A second session read the nodes cold the same day and broke them with a
stand-in model (its cases are in the check). What it found and what changed:
a box with no area, off the frame or not a number was given a body and drawn,
and the boxes node made a one-pixel box from a one-pixel speck of mask. Now
`body_pose.py::unusable` drops such a box and the table names it under
`boxes_refused`, and `subject_boxes.py::SMALLEST_MASK_PX` is a floor under
which a frame has no box, with the day's figures beside the constant. One
box list used for every frame is said in the report. The table also carries
each person's 3D keypoints in camera space, so one pose pass serves the mesh,
the preflight and a motion measure. Still true and not fixed by the node:
`first_source_frame` is a label the caller supplies, and nothing ties it to
where a loader started.

The mesh video has a third style, `marked` (2026-10-10, for the lane that
wants state carried by a signal and not by a sentence): the grey body with
the face side of the head, the back of it and each hand in flat colours
(`body_pose.py::MARK_COLOURS`): the face side yellow, the back of the head
blue, the person's right hand red, the left green. A text that uses these
frames has to name the colours, and takes the sentence from the node's
second output, `legend` (`body_pose.py::legend`), never from memory: the node
is the one place that knows them. Which way a head faces is then an area of
colour: on a public sample turned in the check, the face side is nearly all
of the head towards the camera and nearly none away. Which vertex is which part is a fact
about the rig, kept in `body_marks.json` and read only when the style is
drawn, so a pose pass does nothing for it. It is drawn with pieces
private to ComfyUI's rasteriser, since its public function lights any colour
it is given; the check holds that with nothing marked our draw is the public
one, pixel for pixel. Not known: whether the video model reads it at the few
tokens a head covers, and whether the colours leak into a render. The mesh
cannot show a mouth or an expression in any style.

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
  nothing here runs it yet. *2026-10-11: run once, outside any node, on
  eight frames: `../../bench/results/2026-10-11_sapiens2_pose_first_frames.md`.
  On a small, soft face its lip points sat above the lips the part model
  found, so of the two ways to read a mouth the part classes are the one to
  cut a mask from, and the points are for measuring an opening.*

Both read the original performer's mouth from the picture, which is the
signal a face-only swap is missing: the model hears the frozen track and is
shown nothing of the mouth that made it.

## Candidate preflight flags

Signals a mesh-driven render could be checked on before it samples
([`masked_v2v.md`](masked_v2v.md), "The rule: data before a render, and the
same data after"). Each row says which tracked tool the rule belongs in and
what that tool still lacks to read it; the masking board's card
`build-mesh-preflight-flags` carries the owners. Core's pose data holds
everything they need
(`comfy_extras/sam3d_body/utils.py::run_batched_single_chunk` lists it per
person per frame). Since 2026-10-10 this pack's own node writes what the
rules read as a table (`body_pose.py::pose_table`, as text and as a file
beside the render); what is missing now is the rules.

| flag | what it predicts | belongs in | still lacks | state |
|---|---|---|---|---|
| a hand crop under Meta's size constant: the width of `lhand_bbox` or `rhand_bbox` against `hand_box_size_thresh` (`coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1317`; core's copy is in `comfy/ldm/sam3d_body/model/model.py::run_inference`) | the fingers on that frame are the body decoder's, never the hand decoder's; a little over the constant is not safe either | the preflight of `../../bench/capture_masked_run.py` | nothing from the node: `MiniMaxH3BodyPose`'s table has each hand's crop side and, better, whether its decoder was used (`body_pose.py::pose_table`). The rule itself is not written yet | confirmed on one frame, both sides: the record's "Hands and expression" |
| a person's `bbox` equal to the whole frame while a track is wired | that person's mask was empty on the frame and core substituted the frame (`comfy_extras/sam3d_body/utils.py::_bbox_from_mask`); a body is drawn from whatever is there | closed for a graph that takes its boxes from `subject_boxes.py::MiniMaxH3SubjectBoxes`, which gives no box on such a frame (`subject_boxes.py::frame_boxes`, held by `../../bench/check_video_mask.py`). For a graph that wires core's SAM 3 track straight into the predict node: the same preflight | nothing for our graphs: `MiniMaxH3BodyPose` never substitutes a frame for an empty box; its table says `whole frame` only for a box that was given as the whole frame | read, not run, for ComfyUI's node; held for ours by `../../bench/check_body_pose.py` |
| one whole-frame box and more than one person in the frame | one crop holds everyone; which person comes out is not defined | the same preflight, which already has the person count per shot | nothing from the node: the table's `box_source` says `whole frame` for such a box. The rule is not written yet | read; the one such box in the record is also the only one where the control did not reach the floor |
| a keypoint outside the frame | a joint the model placed where it saw nothing | the same preflight | nothing from the node: the table has the keypoints and a count of those outside the frame. The rule is not written yet | read, not run |

**There is no confidence to read.** Neither Meta's code nor core returns a
score for a detection, a visibility for a joint, or whether a hand was
refined; our node recovers the last of these from the model's own return
and nothing else. Meta's model has hand-presence logits; its inference never reads
them and core drops their weights. A flag here is geometry, not certainty,
and the preflight should print that with the flags.

Not a flag yet, because nothing is measured: the size of a face below which
Sapiens2's lip, teeth and tongue classes stop being reliable. It belongs in
the part node's own report (`../../sapiens2_parts.py`) and lacks a probe: one
face at a ladder of sizes, the lip classes' area read at each.

## Where the probes are

Every sentence above rests on a file in `coderef/`, on a tracked tool, or on
a one-line probe whose command is here. Nothing rests on a session folder.

- Core against Meta: `../../bench/compare_sam3d_body_core_against_meta.py`,
  whose docstring holds the recipe for the Python that runs Meta's side.
- Our own nodes against Meta: `../../bench/check_body_pose.py`, against a
  fixture that tool writes from Meta's code.
- What the ComfyUI venv lacks for SAM-Audio: `importlib.util.find_spec` on
  each name in `coderef/sam-audio/pyproject.toml`, run with the venv's
  Python.
- Upstream's two mask lines on a given day: fetch `sam_audio/processor.py`
  and `eval/dataset/sam_audio_bench.py` from the repository's main branch
  and search for `eq(0)` and `video_frames * mask`.
- A Hub file's size, gating and licence: `hf download --dry-run <repo>`.
- The papers were read as text; a claim from one names its section.

## Not run

Each line is a claim above that only a run settles.

- SAM-Audio on any audio of ours: whether it fits the card, how a mask of
  one singer does against the word alone, and which mask direction is right.
- The shift control that would let PE-AV be tried as a sync score.
- Core's SAM 3D Body prediction with MoGe's field of view against Meta's own
  MoGe call, and core as the server runs it (the card, half precision).
- A tracked person leaving the frame in core's predict node.
- The Sapiens2 keypoint model. *2026-10-11: run once on eight frames (the
  record above); not yet on a blink or a half-open mouth, and not as a node.*
- Any depth or pointmap model on a frame of ours: whether the nearer of two
  crossing masks is the nearer one by its figures.
- The matting model with something reading its matte.
