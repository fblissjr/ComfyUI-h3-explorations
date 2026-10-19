# Perception trials: the models the lane does not run, and what each trial would decide

**Of the perception models read or fetched for the lane, these are not in any job: SAM-Audio and the HyDen relative
depth model are candidates on disk; PE-AV was read and has no use as shipped; SAM 3D Body's ViT-H backbone waits for a
failing case; Sapiens2 matting is closed, the Sapiens2 pointmap was tried and dropped, core's face-expression node is
not to be tried again, and MoE-ViE and a likeness score were set aside.** Each has a short section below with the
reading or the run behind it. A model gets card time only through a row of "The trials" that says, before it runs, what
it buys a render, what passes and what fails; the owner sets the order. The models the lane does run are in
[`perception_models.md`](perception_models.md).

## The trials

The same rules for every row, from this lane's own:

- **A handful of frames where the answer is plain by eye**, picked from captures whose masks a person has looked at. A
  model that is wrong where a person is sure is not worth a second handful.
- **Whoever runs it does not judge it.** A second session looks at the pictures and the saved figures before it reads
  the first one's conclusion, and the record says who did which.
- **A one-off first, a tool only on a pass.** The first run is a script beside its data under the untracked `data/`,
  with a record under `bench/results/`; nothing reads its output until the row has passed.
- **A figure is never an input.** A measure plans a mask or scores a render; what is drawn from one may be shown to the
  model as a layer of the load's one reference video, and nothing goes through a control adapter
  ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-11).

| trial | the question, and what an answer buys a render | state |
|---|---|---|
| 1. The keypoint model | Do the eye and mouth openings follow the picture? It buys a number for expression on a source and its render | **run 2026-10-11**: the eye figure passes at its two ends, the mouth figure fails. Verdict and what was built: [`sapiens2_pose.md`](sapiens2_pose.md), "Trial 1's verdict". Still open: consecutive frames of a blink and a half-open mouth |
| 2. Who is in front | Of two masks that touch or cross, which is nearer? It buys whose pixels are protected when two people cross | **answered** by MoGe-2 on overlap frames ([`depth.md`](depth.md)); the pointmap was tried against it and dropped (below); the relative depth model not run |
| 3. The camera for the body mesh | Does a field of view from the metric point model put the mesh closer on the tracked person than the fallback does? Passes if the drawn mesh overlaps the tracked mask more with it, on frames of more than one shot | not run. A single-frame field of view swings on close-ups, so a shot-level value is what would be tried ([`depth.md`](depth.md), "The camera for the body mesh") |
| 4. Whose voice | Can one voice be taken out of the track by naming or pointing at its singer? It buys "who is heard on this shot" as data where a person types it today. On a stretch where two people sing in turn: the phrase alone, the masked video alone, both, and the current vocal stem beside it. Passes if each singer's line lands in that singer's part | not run. Needs an environment of its own and a card with no H3 server on it ("SAM-Audio", below) |
| 5. How small a face the part model can read | Below what size do the lip, teeth and tongue classes stop being found? One face at a ladder of sizes; a measurement, not a pass | not run. Belongs in the part node's own report ([`sapiens2_parts.md`](sapiens2_parts.md)) |
| 6. A soft edge | Does a matte at the hairline look better than the hard part mask? | **closed** with the matting lane on 2026-10-18 (below) |
| 7. The other body backbone | no question yet; run only on a frame where the mesh in use is wrong | not run |

## SAM-Audio

**A candidate for one job, whose voice is heard on a shot; on disk and loaded by nothing.** A flow-matching model that
takes a mixture and returns the part a prompt names and the rest
(`coderef/sam-audio/sam_audio/model/model.py::separate`). The large model, one `-tv` variant and the judge are in the
owner's store as `facebook_sam-audio-large`, `facebook_sam-audio-base-tv`, `facebook_sam-audio-judge`. Today the voice
table comes from Hybrid Demucs ([`voice_stem.md`](voice_stem.md)).

- **A sung voice from a full mix is a designed case.** Meta's own evaluation on a studio multitrack set prompts with
  the stem's bare name (`coderef/sam-audio/eval/dataset/musdb.py:29`). The paper reports listeners preferring it to the
  Demucs family on that set (section on instrument separation). No comparison of the two has been run here.
- **Three prompts, usable together**
  (`coderef/sam-audio/sam_audio/processor.py::SAMAudioProcessor.__call__`): text, time spans, and a masked video.
- **Text is a lowercase noun or verb phrase**, not a sentence (`coderef/sam-audio/README.md:74`). The encoder is the
  one named by `T5EncoderConfig.name` in `coderef/sam-audio/sam_audio/model/config.py`.
- **A span is a sign and two times in seconds**
  (`coderef/sam-audio/sam_audio/processor.py::Batch.process_anchors`): where the sound is, or where it is not. The
  model can also predict spans from the text (`coderef/sam-audio/sam_audio/model/model.py::predict_spans`), with a
  second model named by `SAMAudioConfig.span_predictor`.
- **Any input is resampled and mixed down to one channel**
  (`coderef/sam-audio/sam_audio/processor.py::batch_audio`; the rate is `DACVAEConfig.sample_rate`). The output is the
  same rate, one channel, a target and a residual. A stereo track comes back mono.
- **The released code separates in one pass.** The paper describes long audio as overlapping windows merged at every
  step; `coderef/sam-audio/sam_audio/model/model.py::separate` has no windows. A whole song is cut by the caller.

### The visual prompt

- **What it is given:** the video's frames multiplied by a mask, as `masked_videos`. Each frame is resized to a square
  with its aspect not kept (`coderef/sam-audio/sam_audio/model/vision_encoder.py::PerceptionEncoder.get_transform`) and
  becomes one vector from an image encoder (`PerceptionEncoderConfig.name`); the frame nearest each audio step is the
  one used (`coderef/sam-audio/sam_audio/processor.py::load_video`). In Meta's example the description is left empty
  and the mask comes from SAM 3 prompted with a phrase (`coderef/sam-audio/examples/visual_prompting.ipynb`).
- **Which way the mask goes is contradicted inside Meta's code.** The helper the README shows keeps the pixels where
  the mask is zero, so it blacks out the object (`coderef/sam-audio/sam_audio/processor.py:204`). The paper says the
  model sees the frames times the mask, the object kept, and Meta's evaluation loader does that
  (`coderef/sam-audio/eval/dataset/sam_audio_bench.py:118`). Upstream's main branch carried both lines on 2026-10-10. A
  run here builds the masked video the evaluation's way and tries the helper's way beside it.
- **Meta says it is the weaker prompt.** The paper's conclusion calls it noticeably less effective than text, and says
  a mask of a person is ambiguous because a person makes more than one sound. It names one case where the mask decides:
  several speakers a text cannot tell apart. Use the mask with a text, not in place of one.
- The `-tv` checkpoints are the ones Meta recommends for the visual prompt (`coderef/sam-audio/README.md:124`).

### What loading it brings

Loading a checkpoint as released also loads the rankers and the span predictor its `config.json` names
(`coderef/sam-audio/sam_audio/model/model.py:94`, `:95`, `:97`): an image-audio model, an audio-text model, Meta's
judge and an audio-frame model. They serve reranking and span prediction only. A keyword of the same name set to
`None` at load drops each (`coderef/sam-audio/sam_audio/model/base.py:50`). Whether the largest checkpoint fits the
card with them dropped is not measured; it does not fit beside a loaded H3 server. Licence: the SAM License in
`coderef/sam-audio/LICENSE`. The ComfyUI venv has none of its audio dependencies (`coderef/sam-audio/pyproject.toml`,
checked 2026-10-10 with `importlib.util.find_spec`), so it runs from an environment of its own.

## PE-AV

**Read on 2026-10-10 and of no use to the lane as shipped: it does not score lip sync.** A contrastive encoder that
puts audio, video, audio with video, and text in one space (upstream `facebookresearch/perception_models`, the class
`PEAudioVisual` in `pe.py` under `core/audio_visual_encoder`; not on disk in this repo, no weights here).

- **Every embedding is pooled over the clip.** Each tower's output head is applied to its class token
  (`PEAudioVisual.forward`; `Transformer.forward` in its `transformer.py` returns the first position). Audio against
  video is one number for a clip.
- **Nothing in it is trained or evaluated on lip sync.** The paper's tasks are retrieval and classification.
- **The frame-level model is audio against text.** PE-A-Frame scores each audio step against a description
  (`PEAudioFrame.forward`). No video enters it.
- Scoring a face crop against an isolated voice over short windows is an idea with an entry condition: the same clip
  with its audio shifted by a few frames must score lower. Until that control is run, a number from it means nothing.
- What depends on it: SAM-Audio's span predictor is a PE-A-Frame model (`coderef/sam-audio/sam_audio/model/model.py:97`)
  and SAM-Audio's judge is built on PE-AV's transformer class (`coderef/sam-audio/sam_audio/model/judge.py:8`).
  SAM-Audio's own video encoder is the image model, not PE-AV
  (`coderef/sam-audio/sam_audio/model/vision_encoder.py:86`).

## Sapiens2 matting

**Closed on 2026-10-18, not refuted** ([`docs/state/roadmap.md`](../state/roadmap.md), "Closed lanes"). The loader has a
slot for it (`sapiens2_parts.MATTE_ARCH`) and `h3_config.SAPIENS2["matting"]` is `"none"`: the Masked Source reads the
hard part mask, so loading it would cost a pass a frame and change nothing. One run loaded it on 2026-10-05
([`2026-10-05_sapiens2_first_frame.md`](../../bench/results/2026-10-05_sapiens2_first_frame.md)); a review of what a
hair-edge composite would take is
[`2026-10-17_review_sapiens2_matting.md`](../../internal/gemini/2026-10-17_review_sapiens2_matting.md). What stands
instead: the pack's own per-piece mattes under one compositing law (`bench/assemble_delivery.py::lay_plane`).

## Sapiens2 pointmap

**Tried and dropped on 2026-10-18.** The 1b and the 5b, on CPU on 1306-1310: the 1b gives a flat sheet, the 5b a soft
edge at 18 times MoGe-2's time; MoGe-2 separates the fist from the face well
([`2026-10-18_pointmap_finding.md`](../../internal/gemini/2026-10-18_pointmap_finding.md), `6aebcbf6`). It is trained for
pictures of people (its card's "Intended Use"), and its 5b file is about the size of the card's whole memory, so it
cannot sit beside a loaded H3 server.

## HyDen relative depth, and Depth Anything V2

**A candidate, in the store since 2026-10-11 as `facebook_hyden-da2-relative-depth`, loaded by nothing.** Meta's
sharper-edged variant of Depth Anything V2 Large; relative depth. Its use would be a second reading beside MoGe-2 on the
same frames. Two models looking at the same pixels are not two independent votes, so agreement proves little; a frame
where they disagree is one to look at by eye. Its code is `coderef/metadepth`, which loads it with torch and torchvision
alone. Depth Anything V2 itself is in the ComfyUI checkout's `models/depthanything/` and adds nothing once the HyDen
variant is here. The three `hyden` model cards were read on the Hub on 2026-10-11; read a card again before relying on
it.

## SAM 3D Body's other backbone

**On disk, loadable, used by no job.** `facebook_sam-3d-body-vith` loads through `MiniMaxH3BodyModelLoader`
(`../../sam3d_body_vith.py`, held by `../../bench/check_sam3d_body_vith.py`). One frame through both releases is all
that has been compared ([`2026-10-05_sam3d_body_vith.md`](../../bench/results/2026-10-05_sam3d_body_vith.md)); do not
switch without a failing case.

## Core's face expression node

**Not to be tried again** ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-18): a reference driven by
`SAM3DBody_FaceExpression` broke the mouth, the eyes and the iris when tried. What it is: [`sam3d_body.md`](sam3d_body.md),
"The face". The expression the reference carries is drawn from the keypoint model instead
([`sapiens2_pose.md`](sapiens2_pose.md)).

## Looked at and set aside: MoE-ViE

`facebookresearch/moe_vie`, read on GitHub on 2026-10-11, nothing fetched. A contrastive image and video encoder scored
against text: one vector for a whole image or clip. It finds nothing in a frame and follows nothing, so it has no place
in making a mask. The one question it could be pointed at, whether two crops show the same person, is closed here for a
reason that does not depend on the encoder: who is in a scene is a person's answer to a numbered picture
([`who_is_who.md`](who_is_who.md); [`docs/state/decisions.md`](../state/decisions.md), 2026-10-10).

## Not on the list unless the owner reopens it: telling who is who by a likeness score

A score between two crops was the first idea of 2026-10-10 and failed there on this lane's own clip, with the tracker's
features: a line could not name a person, and wrong takes scored inside the range of right ones
([`2026-10-10_who_is_who_across_shots.md`](../../bench/results/2026-10-10_who_is_who_across_shots.md)). The ruling that
followed stands: the answer is a person's, given once a video. What is still open to a fair trial is narrower, and is a
second reading shown beside a numbered picture, never written into the cast file: on the shots whose answers a person
has already given, does some other feature (the part model's own backbone under the mask costs no download) rank the
right person first on the shots the tracker's features got wrong? It passes only if it does that on those shots;
agreeing on the easy ones says nothing. A likeness cannot know a person who has turned away or changed clothes,
whichever model it comes from.

## Where each was read

| what | checkout | revision | weights here |
|---|---|---|---|
| SAM-Audio | `coderef/sam-audio` | `bb4c699` | in the store since 2026-10-10, Meta's original files; nothing loads them |
| PE-AV and PE-A-Frame | upstream `facebookresearch/perception_models` at `3e352cc`, not under `coderef/`; a port is in `coderef/transformers/src/transformers/models/pe_audio_video` and in the ComfyUI venv's `transformers` | | none |
| SAM 3D Body | `coderef/sam-3d-body` | `b5c765a` | both releases ([`2026-10-05_sam3d_body_conversion.md`](../../bench/results/2026-10-05_sam3d_body_conversion.md)) |
| core's SAM 3D Body port | the ComfyUI checkout, `comfy_extras/nodes_sam3d_body.py`, `comfy_extras/sam3d_body/`, `comfy/ldm/sam3d_body/` | `0df64eb24` | the converted files above |
| Sapiens2 | `coderef/sapiens2` | `7e5bae8` | segmentation, matting, the keypoint model, the pointmaps |
| HyDen MoGe-2 and relative depth | `coderef/metadepth` | | in the store since 2026-10-11 |

The papers: SAM Audio is arXiv 2512.18099 and SAM 3D Body is arXiv 2602.15989, both linked from their READMEs; the PE-AV
report is linked from `coderef/sam-audio/README.md`. A claim from a paper names its section. Sizes, gating and licence of
a file on the Hub are read with `hf download --dry-run <repo>`, not from this page.

## Where the probes are

- What the ComfyUI venv lacks for SAM-Audio: `importlib.util.find_spec` on each name in
  `coderef/sam-audio/pyproject.toml`, run with the venv's Python.
- Upstream's two mask lines on a given day: fetch `sam_audio/processor.py` and `eval/dataset/sam_audio_bench.py` from
  the repository's main branch and search for `eq(0)` and `video_frames * mask`.

## Not run

- SAM-Audio on any audio of ours: whether it fits the card, how a mask of one singer does against the word alone, and
  which mask direction is right.
- The shift control that would let PE-AV be tried as a sync score.
- The HyDen relative depth model on a frame of ours.
- Trials 3, 4, 5 and 7 above.
