# Perception models in the masked lane: what each gives, and which are wired

**The lane runs on four perception models: SAM 3.1 says which pixels are the person and, since 2026-10-19, which are
the head; Sapiens2's part model labels every pixel of that person; SAM 3D Body poses a body mesh whose video, cut to the
head, is every face load's reference; and Sapiens2's keypoint model draws the source's face on that reference and
measures the eyes.** MoGe-2 depth says who is in front where two masks overlap, behind a switch that is off by default.
A Hybrid Demucs vocal stem says where a voice is. None of them carries a mouth into the body mesh or scores lip sync;
the mouth on a reference is drawn from the part and keypoint models where the person sings or talks. SAM-Audio, PE-AV,
the Sapiens2 pointmap and matting, and core's face-expression node were read, tried or closed, and run in no job. Where
this page and the code disagree, the code is right.

## The models

| model (store folder or file) | what it gives the lane | status today | since | evidence |
|---|---|---|---|---|
| SAM 3.1 (`h3_config.SEGMENTER`) | the person's mask shot by shot; a head track inside it | **wired** into `MiniMaxH3SubjectTrack` (`subject_track.py`) in every job; the head track on by default (`subject_track.py::TRACK_HEAD`) | before 2026-10-07; head track 2026-10-19 | `8b50cecf`, `b7927da1`; [`sam31.md`](sam31.md), [`sam31_core.md`](sam31_core.md) |
| Sapiens2 part segmentation (`facebook_sapiens2-seg-1b`) | a class for every pixel: a part load's region, the class map each capture grades against, the reference's mouth, the strangers keep-out; face against hair inside the head track | **wired** into `MiniMaxH3Sapiens2Loader`, `MiniMaxH3SubjectParts` (`sapiens2_parts.py`) in every preview; read by `bench/capture_masked_run.py` and `bench/head_tube_parts.py` | 2026-10-05; parts graph 2026-10-06; head tube 2026-10-17 | `8b451ebc`, `9ae35f54`, `c4b395bf`; [`sapiens2_parts.md`](sapiens2_parts.md) |
| SAM 3D Body, DINOv3 (`h3_config.MASKED_JOB_BODY_MODEL`) | a posed body mesh and joints table; the body-model video the head reference is cut from; head direction | **wired** into `MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose`, `MiniMaxH3BodyMeshVideo` (`body_pose.py`) in every preview | 2026-10-10 | `f8538b1c`, `42aaa7ee`; [`sam3d_body.md`](sam3d_body.md) |
| Sapiens2 keypoints (`h3_config.POSE_MODEL`) | the face points drawn on the head reference; the eye opening at its two ends; the face state in the head tube | **wired** into `bench/pose_along_shot.py`, run by `bench/shot_to_load.py reference` and `bench/load_chain.py land`; no node | 2026-10-11; the reference form 2026-10-12 | `08429b9b`, `0a16f8e6`, `503f9cf3`; [`sapiens2_pose.md`](sapiens2_pose.md) |
| HyDen MoGe-2 metric point (`facebook_hyden-mogev2-metric-point`) | metric depth on overlap frames: who owns a shared pixel | **wired, off by default**: `bench/shot_to_load.py job --depth`, `bench/measure_moge2_depth.py`, `vendor/meta_moge2/` | tool 2026-10-17; chain 2026-10-18 | `4725ff5c`, `1c213f41`, `3980ed5c`; [`depth.md`](depth.md) |
| Hybrid Demucs (torchaudio's cache) | the vocals stem; voiced frames | **wired** into `bench/voice_spans.py`, read by `bench/load_text.py` and the capture | 2026-10-10 | `544dbe81`; [`voice_stem.md`](voice_stem.md) |
| SAM 3D Body, ViT-H (`facebook_sam-3d-body-vith`) | the same mesh on the other backbone | candidate: loadable, used by no job | 2026-10-05 | `9d2392af`; [`perception_trials.md`](perception_trials.md) |
| HyDen relative depth (`facebook_hyden-da2-relative-depth`) | a second depth reading | candidate: in the store, loaded by nothing | 2026-10-11 | `253db186`; the same page |
| SAM-Audio (`facebook_sam-audio-*`) | one voice out of the mix by a phrase, a span or a masked video | candidate: on disk, loaded by nothing; needs its own environment | read 2026-10-10 | `5d1cc0c6`; the same page |
| PE-AV | one pooled audio-video vector a clip | read and set aside: no lip-sync score; no weights here | 2026-10-10 | the same page |
| Sapiens2 matting (`facebook_sapiens2-matting-1b`) | a soft edge for the whole person | closed, not refuted: a slot in the loader, `h3_config.SAPIENS2["matting"]` is `"none"` | node 2026-10-05; closed 2026-10-18 | `23c32dbb`; the same page |
| Sapiens2 pointmap (`facebook_sapiens2-pointmap-1b`, `-5b`) | a position in space for every pixel of a person | tried and dropped: a flat sheet at 1b, a soft edge at 18 times MoGe-2's time at 5b | 2026-10-18 | `6aebcbf6`; the same page |
| core's `SAM3DBody_FaceExpression` (MediaPipe) | an expression on the mesh | tried and dropped, not to be tried again: broke the mouth, eyes and iris | 2026-10-18 | `1b3e8dea`; [`sam3d_body.md`](sam3d_body.md), "The face" |
| MoE-ViE | one vector a picture or clip | set aside: finds nothing in a frame | 2026-10-11 | `be311502`; [`perception_trials.md`](perception_trials.md) |

## The two questions behind this page

1. **When only a face is redrawn, what shows the model the original performer's mouth?** The head reference: the pose
   model's lip points where they lie on the part model's mouth, drawn on windows where the person sings or talks
   ([`sapiens2_pose.md`](sapiens2_pose.md)); what that carried in renders is [`renders_by_form.md`](renders_by_form.md).
   Nothing scores a render's lips against its track; a mouth is read by eye.
2. **When a whole person is redrawn, what carries the movement, with more than one person in the frame?** The
   body-model video, posed from the tracked person's box and mask, so identity is the track's
   ([`sam3d_body.md`](sam3d_body.md), "More than one person"). What each signal can say about a subject is
   [`state_signals.md`](state_signals.md); how a load is given them is [`running_a_job.md`](running_a_job.md).

## The store

The owner's model folder holds one `facebook_<name>/` folder a Hugging Face repository; list it to see what is there
now. Besides the models above it holds the annotation sets `facebook_SACo-Gold`, `facebook_SACo-Silver`,
`facebook_SACo-VEval` and `facebook_SA-FARI`, read by `../../bench/sam3_dataset_phrases.py` and nothing else. Named on
these pages and not in the store: PE-AV's weights, MoGe for ComfyUI's own node (`comfy_extras/nodes_moge.py`), and
MediaPipe, which only core's face-expression node uses. Where each model's code was read, at which revision:
[`perception_trials.md`](perception_trials.md), "Where each was read".
