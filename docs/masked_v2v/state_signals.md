# State by signal: what code can say about a subject, and what it cannot yet

**A subject's state (which way the head faces, whether the face shows, a hand at the face, who is in front, an open
mouth) comes from a model's measurement, never from a session reading the scene into a prompt** (the owner,
2026-10-10). It reaches H3 through the reference video, never the text: a generated text takes no place-and-action
sentence (`bench/load_text.py::NO_SCENE`; [`text.md`](text.md)). The capture (`bench/capture_masked_run.py`) writes
each signal as a column per frame and per subject; the gate raises what is at risk. What each model behind a signal
gives and does not is [`perception_models.md`](perception_models.md).

## The map

| state | what measures it | trust | how it reaches the model | cost on record |
|---|---|---|---|---|
| which way the head faces | `bench/measure_subject_yaw.py::head_yaw`; the capture's `pose_state` | points measured against Meta's code; the reading untested on a frame of known answer | the body-model reference video | yes, both ways: omitted and guessed |
| head lowered or raised | `head_lift`; the capture's `pose_state` | untested | the same video | timing only |
| a hand at the face | each wrist's distance from the nose in torso units (`pose_state`) | no distance set for "at the face" | the same video | yes: drawn early and long |
| which hand holds what | the part node's `held` mask: something is held, not which hand, not what | untested | the plate, when it does not touch the subject; given back at assembly | yes: a second hand; a thing left where it was |
| whether the face shows | the part's face class on the frame, with the head angle (`grade_holds`) | measured on one shot | an empty part on turned-away frames | yes: a face shape drawn on hair |
| mouth open, singing | lip, teeth and tongue classes; voiced spans | one stretch | the audio; "sings" in the text where the plan says so | yes: a voice sentence for shots with none |
| a mouth open with no voice | `bench/capture_masked_run.py mouth` | one shot, two renders | none yet | yes: the opening is not carried |
| whose voice | nothing run; SAM Audio takes a person's mask over the frames with a text | on disk, unwired | n/a | n/a |
| who is in front of whom | MoGe-2 depth on overlap frames (`--depth`) | figures on crossings, 2026-10-18 | the plate: the back person's region clipped, the other given back | none traced to a text |
| how large in the frame | `track_share`, the boxes | measured | the hole; the zoomed render | none in this lane |
| what in the region is not the subject | the region's share that is not the person | measured; the things are unnamed | keep them out of the region, or give them back | yes, both ways: redrawn as other things; drawn in the next shot |
| hair over the face | hair against face classes | untested | none | none traced |
| whether a render looks like the reference | nothing | | | yes: a small face that does not read as the reference |

Order and timing words in a text ("then", "near the end") need no measurement, they need deleting.

## Which way the head faces

- **Measured** by `bench/measure_subject_yaw.py::head_yaw`, the angle of the line between the ears in the body model's
  camera-space keypoints, on the scale `yaw_of` documents. `MiniMaxH3BodyPose` (`body_pose.py`) saves those keypoints
  in its pose table and the capture takes it (`pose=` on `--mask`), so the angle can be read from a no-sampling
  preview. `bench/capture_masked_run.py::pose_state` writes the body's, hips' and head's facing and the chin's lift per
  frame and subject.
- **Record**: `data/2026-10-10_fun_pose/kitchen_0604_yaw.json` (`source.curve[].head_yaw`; how closely the mesh video
  follows, `clips.mesh_video.mean_difference` and `end_difference`). One stretch of one clip: the head turns from well
  away to facing the camera, the mesh video follows closely, and the two whole-subject renders did not make the turn
  (`clips.mesh_pass`, `clips.morning_pass`). The mesh pass was slow on the turn, not deaf to it: its
  `clips.mesh_pass.curve[].head_yaw` ends well short of the source. Read the curve before calling a render one that
  ignores a signal.
- **Not verified**: left against right (`yaw_of`'s sign has not been checked on a frame of known answer); and the far
  end of the scale. A head seen from behind still gets a nose point, because the body model completes what it cannot
  see, so "the face shows" must come from the angle and the face class, never from a keypoint existing.
- **Two crops**: the record's angles are from ComfyUI's own body nodes (`SAM3DBody_Predict`), not this pack's
  `MiniMaxH3BodyPose`, whose crop is Meta's ([`sam3d_body.md`](sam3d_body.md)). The two have not been
  compared on a frame.
- **The dense pose model** (Sapiens2 pose 1b) names the lips point by point, both eyelids, the iris and pupil, the
  nose, each ear's outline and each finger's joints
  (`coderef/sapiens2/sapiens/pose/configs/_base_/keypoints308.py`). It is run along shots by
  `bench/pose_along_shot.py`; its eye figure is trusted at its two ends and its mouth figure is not
  ([`sapiens2_pose.md`](sapiens2_pose.md), "Trial 1's verdict").

## Whether the face shows

- **The rule is code.** `bench/capture_masked_run.py::grade_holds` (`part_grades.json` in a capture) keeps a part or a
  fill on a frame only where it lies on the part's own classes. Turned-away frames' face part is emptied and never
  filled (`data/CAPTURE_GAPS.md` item 39). It reads the class map, so a frame where the class map is wrong is still the
  caller's to empty (`--no-face-on`, [`running_a_job.md`](running_a_job.md), "A shot, made whole").
- **Evidence**: one street shot rendered twice, the part kept on four turned-away frames in one and emptied in the
  other. Kept, the render redrew the hair and drew no face; emptied, the frames are the source's
  (`bench/results/2026-10-10_face_part_kept_or_emptied_on_a_turn.md`, one shot, one seed). The "face shape on hair"
  seen before was the mask's shape.
- **Not built**: the column on the render's side, how much of a face a render shows against its source
  (`data/CAPTURE_GAPS.md` item 37).

## Head lift, and a hand at the face

Both are columns of the capture for a pose table with 3D keypoints (`bench/capture_masked_run.py::pose_state`): the
chin's lift by `measure_subject_yaw.py`'s own arithmetic, and each wrist's distance from the nose in torso units.
Pinned on a hand-built skeleton; no distance has been set for "at the face".

## Who is in front of whom

**Measured with depth, on the frames where two masks overlap**: `bench/shot_to_load.py job --depth` runs
`bench/measure_moge2_depth.py` on a load's overlap frames, and the capture settles shared pixels by foreground depth
([`running_a_job.md`](running_a_job.md), "Depth: who is in front";
[`2026-10-18_moge2_depth_on_crossings.md`](../../bench/results/2026-10-18_moge2_depth_on_crossings.md)). It cannot
separate faces side by side at the same distance, or a hand on a face.

- **Why it is needed**: the owner map (`owners.npz`) settles a contested pixel by the part model's classes, with no
  notion of nearer; the pixels it could not settle on 2026-10-10 were hair and a cuff.
- **The body model's order** (every person on a frame given the same camera) put three people in the right order on a
  public frame (`bench/fixtures/sam3d_body_meta_reference.json`, `office_frame60`); the figures are not kept and
  become a case in `bench/check_body_pose.py` with a depth field. A whole-frame box on a frame of several people
  comes back as one body at about the nearest person's distance, so depth is read only from a box on one person. A
  mesh has no hair, clothing or held things, which is what the unsettled pixels were.
- **The Sapiens2 pointmap** was tried and dropped on 2026-10-18 (`6aebcbf6`): a flat sheet at 1b, a soft edge at
  far more than MoGe-2's time ([`perception_trials.md`](perception_trials.md), "Sapiens2 pointmap"). MoGe-2 is the
  depth the lane uses ([`depth.md`](depth.md)).
- **Objects** need the body model's sibling scene model (`coderef/sam-3d-body/README.md`), which is not in the store.

## A mouth open with no voice

**The case holds, by measure: a face-only pass with no motion video keeps the mouth at its ordinary opening where the
source's opens wide on frames with no voice.** Its timing follows the source's throughout; what is missing is how far
it opens. Figures: `mouth__singer.json` in `data/2026-10-10_fun_face_street_0294_run` (untracked),
`open_with_no_voice`. One shot, two renders, one seed. The gate raises the run before a render
(`mouth_open_with_no_voice`, iffy).

- **Why it has no channel**: the mouth is the audio's to carry; with no voice the audio says nothing, the mesh has no
  mouth, and a face pass wires no motion video.
- **Which frames**: `bench/capture_masked_run.py mouth` reports the source's runs of a dozen frames or more with the
  opening in the shot's top quarter on unvoiced frames (`open_runs`, `OPEN_SHARE`, `OPEN_RUN`; `--voice` for a capture
  made without the table), with each render's opening over the run as a share of the source's. On the two street face
  shots it finds one run on the first and none on the second
  (`bench/results/2026-10-10_face_part_kept_or_emptied_on_a_turn.json`, `mouth_open_with_no_voice`). The render's
  column needs the part model's class map of each render.
- **First test, reasoned, nothing rendered: the source's head cut out, zoomed, blurred, to the video model.** An open
  mouth is about twice the size of the features that carry who a face is, so a narrow blur removes the second and
  keeps the first: by the Gaussian's falloff a width of about a sixteenth of the head's width leaves fine features a
  few percent of their contrast and the open mouth about half; at an eighth the mouth is gone too. So two renders, the
  sixteenth and the eighth as the control. Read `look` first (does the original's face come back), then `mouth`. The
  blur option once built on the motion reference was removed unrendered on 2026-10-10
  ([`docs/state/decisions.md`](../state/decisions.md), "What the owner cut from the masked lane's nodes"), so this
  needs a prepared video on `motion_video`.
- **Second, if the blur leaks: a mouth map**, the part model's lip, teeth and tongue classes drawn flat on grey, zoomed
  to the head. It cannot bring a face back. One was tried on 2026-10-10 on a voiced stretch from a session's script,
  with the lip sync of three arms inside each other's noise; no record, so it says nothing about a shot with no voice.
- **Not a candidate: a sentence.** "Mouth open" is an event with a time, and a laugh written from stills is on record
  as drawn on every frame.

## The degraded original as a reference

The subject cut out of the source carries facing, hair and hands that a grey mesh does not, and brings the original's
look with it (the owner's idea, 2026-10-10). Untested; nothing in the nodes carries it since the blur and grey options
were removed. Noise at the model cannot serve: core has one level for every visual reference, the still included
(`comfy/ldm/minimax/model.py`). The ranking, all reasoned: (1) the clean cut-out to the video model, which sets the
size of the leak; (2) a blur half of one of the video model's tokens wide at the size the readers see; (3) the same in
grey, since blur removes fine detail first and the look that came back on record was mostly hair and clothing; (4) a
blur of one whole token, expected to be where facing starts to go; (5) the blurred cut-out to the text encoder with
the grey mesh to the video model, which needs a second window-aligned video (`motion_video_2` now exists,
[`pieces.md`](pieces.md)). Accepted only if `look` is no worse than the grey mesh's on two seeds and a state the mesh
cannot carry (a mouth, a held thing, hair) comes through.

## How a signal reaches the model

- **Which model reads a motion video, and how often.** The text encoder always gets it, every twelfth frame
  (`reference_conditioning.py`, the `sample_indices` line), joined in pairs into one picture block a second
  (`comfy/text_encoders/minimax.py`, the `kind == "video"` branch). The video model gets every frame, only with
  `motion_vae` on (`reference_conditioning.py`, the `record.use_vae` gate). So anything inside a second rides only the
  video model's copy. On the opening shot the mesh to the text encoder alone followed far less than the mesh in the
  video model.
- **Its size**: `motion_short_edge`, default `video_mask.MOTION_SHORT_EDGE` (reasoned for cost, not measured). Every
  arm since 2026-10-10 runs at the canvas's short edge; [`docs/h3/references.md`](../h3/references.md), "Video
  references", has all three vendor implementations putting a reference video on the canvas rule, so the default is a
  candidate to move (a schema change, not made).
- **Kept frames outweigh the motion video.** On the kitchen stretch the same mesh, text and seed made the turn, the
  lowered head and the hand at the face as a load of its own, and did not as a continuation behind kept frames of her
  turned away; the mesh's size and the Sol sink changed nothing
  ([`bench/results/2026-10-10_kept_frames_against_the_mesh.md`](../../bench/results/2026-10-10_kept_frames_against_the_mesh.md);
  `data/2026-10-10_fun_pose/kitchen_0950_*`, `kitchen_0936_*`, `open_0014_*`). Two one-window patches on the pass's
  own render failed the same way, which says "kept frames" and not "a continuation". A marked or zoomed mesh belongs
on a stretch where the plain mesh failed with no kept frames in the way: the hands and elbows of the kitchen pass
(`data/2026-10-10_fun_pose/kitchen_0604_motion.json`, the mesh pass against the morning pass). The levers on kept frames are
  [`window_context.md`](window_context.md).
- **A small subject did not follow her mesh** on a 29-frame load with no kept frames
  (`data/2026-10-10_fun_pose/shot_1010_*`; the capture `data/2026-10-10_fun_kitchen_shot2_1010`). The zoomed motion
  video changed nothing there; one change of the reference still made the render take the pose, so a still showing
  another pose pulls like a kept frame ([`window_context.md`](window_context.md)).
- **Drawn marks.** The `marked` mesh style (face side, back of head and each hand in flat colours, `body_pose.py::legend`
  naming them, `body_marks.json` saying which vertex is which, `body_pose.py::FACE_FROM`; 86febba3) has not been
  rendered. The source's lid openings drawn on the body model's face did render, and the face points drawn on the body
  model cut to the head are the lane's standard form: [`renders_by_form.md`](renders_by_form.md).
- **A wired video in the subject's box**: `motion_reference` has a choice for it (`video_mask.MOTIONS`, f7e5de99).

## Whether a render looks like the reference

**Nothing measures it.** Every reading the capture makes of a render asks whether it differs from the source
(`changed`, `look`, `ends`); a face that is new and nobody's in particular passes all of them.

- **The tracker's own signature cannot**: on a few frames each of two face passes, one accepted by the owner and one
  in question, a render is nearer the reference than its source by an amount small on both and no larger on the
  accepted one (`data/2026-10-10_fun_likeness_try.json`, untracked; a one-off script). Do not try it again for this.
  It was read under a whole head, which a face pass leaves alone, and was built to tell people apart across a cut.
- **Not ruled out**: a measure on the face part's own pixels by a model made for faces; none is in the pack.
- **In its place**: `face_small_in_tokens`, a warning when the face to be replaced is only a few tokens tall
  (`bench/capture_masked_run.py::flag_face_size`, `FACE_TOKENS`; its line is unmeasured).
- **A related state**: a region that holds the whole head takes the head's angle and the eyes from the still when
  nothing says where the head points (one hair-and-face load, read from stills); the still's own angle is measured
  nowhere.
