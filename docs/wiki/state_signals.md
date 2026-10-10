# State by signal: what code can say about a subject, and what it cannot yet

The masked lane needs to know things about a subject that a body mesh or a
track alone does not carry: which way the head faces, whether the face
shows, where a hand is, who is in front of whom. The owner's direction
(2026-10-10) is that these come from a model's measurement and never from a
session reading the scene and writing it into a prompt. This page is the
ledger of each such signal: what measures it, where the measurement is, and
what is still unverified or unbuilt.

**How this page is kept.** One entry per signal. A status that changes gets
a dated line under its entry, with who and the record or commit; the older
line stays. Do not rewrite an entry to match the newest result. A claim that
reached this page from a session's message and has no record yet says so, so
the next reader checks it before building on it. The owner's rule behind
this (2026-10-10): a status told to the owner in a conversation is written
down the same hour, somewhere every session reads, and annotated when it
changes.

**Where it is going.** The capture (`bench/capture_masked_run.py`) writes
each signal as a column per frame and per subject. The job builder (the
masking board, `build-multi-person-graph-in-the-generator`) uses a column in
one of two ways: as a mark drawn on the motion video's mesh, or, where a
drawing cannot carry it, as one state sentence written by rule from the
column. No signal passes through a session's reading of the frames.

What each model behind these signals gives and does not is in
[`meta_perception_models.md`](meta_perception_models.md). The order a job
runs them in is the masking board's `guide-order-of-operations-masked-job`.

## Which way the head faces

- **Measured by:** `bench/measure_subject_yaw.py::head_yaw`, the angle of
  the line between the ears in the body model's camera-space keypoints, on
  the scale that file's `yaw_of` documents. `MiniMaxH3BodyPose`
  (`body_pose.py`) saves those keypoints in its pose table, and the capture
  takes that table (`pose=` on `--mask`), so the angle can be read from a
  no-sampling preview.
- **Record:** `data/2026-10-10_fun_pose/kitchen_0604_yaw.json`. The source's
  head angle per sampled frame is `source.curve[].head_yaw`; how closely the
  mesh video follows the source is `clips.mesh_video.mean_difference` and
  `end_difference`. One stretch of one clip, one subject.
- **Status, 2026-10-10 (mrpop):** measured. On that stretch the record
  shows the head turning from well away from the camera to facing it, and
  the mesh video following the source closely; the two whole-subject
  renders of the stretch did not make the turn
  (`clips.mesh_pass`, `clips.morning_pass`, and the masking board's
  `today-2026-10-10-fun-three-people`).
- **Not verified:** left against right. `yaw_of` states a sign convention;
  nobody has checked the sign against a frame whose answer is known.
- **Not verified:** that the scale means what its words say at the far end.
  A head seen from behind still gets a nose point from the body model
  (reported by mrdeer in a session message, 2026-10-10; no record yet).
- **Unused so far:** the dense 2D pose model in the owner's model store
  (Sapiens2 pose, the 1b size), which has face landmarks and may give a
  better facing signal than two ears. Not run on any clip here.
- **Added 2026-10-10 (mrwolf): the record's angles and the pose table's are
  from two crops.** `bench/measure_subject_yaw.py` runs ComfyUI's own body
  nodes (`SAM3DBody_Predict`, hand refinement off), not this pack's
  `MiniMaxH3BodyPose`, whose crop is sampled the way Meta's code samples it
  (`meta_perception_models.md`). An angle read from a preview's pose table
  and one from this record have not been compared on a frame.
- **Added 2026-10-10 (mrwolf):** the nose point on a head seen from behind
  is also mrfrog's reading of the model's code (session message, no
  record): the body model completes what it cannot see. So "the face
  shows" must come from the angle, never from a keypoint existing.
- **Added 2026-10-10 (mrwolf): the mesh pass was slow on the turn, not deaf
  to it.** In the same record `clips.mesh_pass.curve[].head_yaw` moves
  toward the source's over the stretch and ends well short of it. Read the
  curve before calling a render one that "ignores" a signal.
- **Added 2026-10-10 (mrfrog): what the dense 2D pose model has by name.**
  Its keypoint list
  (`coderef/sapiens2/sapiens/pose/configs/_base_/keypoints308.py`) names
  the inner and outer lips point by point, both eyelids, the iris and
  pupil of each eye, the nose tip and bridge, each ear's outline, and each
  finger's joints. So a mouth opening, a blink, a gaze and a finger pose
  exist on paper as distances between named points, and facing could be
  read from many face points and not two ears. The weights are in the
  owner's model store; nothing here loads them, and no frame has been run.
  Unknown until one is: whether those points hold on a face a few dozen
  pixels across, which is the size that matters in this lane.

## Whether the face shows

- **Not built.** It needs two things together: the head angle above and the
  area of the part model's face class on the frame, since an angle alone
  cannot tell a face from the back of a head (see the nose point above).
- **What exists:** the class map per frame in a capture (`classes=`), and
  the head angle. `data/CAPTURE_GAPS.md` item 37 is the missing column on
  the render's side (how much of a face a render shows against its source).
- **Evidence it matters, 2026-10-10 (mrdeer, the street shots' gate):** on
  frames where the subject is turned away the part model's face part is
  empty or a sliver, and a fill from neighbouring frames drew a face shape
  on hair, a hat or an arm. The face part is emptied on those frames and
  never filled (`data/CAPTURE_GAPS.md` item 39).

## Head lift, and a hand at the face

- **Head lift, measured by:** `bench/measure_subject_yaw.py::head_lift`
  (the nose against the line between the ears). In the tool; not a column
  of the capture yet.
- **A hand at the face: not built.** The pose table holds both wrists and
  the head's points, so it is a distance read from one table. Nobody has
  written it.

## Who is in front of whom

- **Why it is needed:** where two subjects' masks meet, the capture's owner
  map (`owners.npz`) settles a pixel by the part model's classes. It has no
  notion of nearer and farther, and the pixels it could not settle on
  2026-10-10 were hair and a cuff.
- **What can measure it:** the body model places every person in one
  camera's space (`meta_perception_models.md`, the pose table's 3D
  keypoints), so two people on a frame can be ordered by distance.
- **Status, 2026-10-10:** not measured on any clip here. mrfrog reports the
  order came out right, front to back, on a public frame of three people
  (session message; no record yet). Untested where it matters: two people at
  nearly the same distance, which is when masks cross.
- **The test that needs no render (mrdeer, scoped, not run):** take the
  kitchen pixels the class maps already settled between the two subjects
  and count how often the body model's order agrees.
- **A limit known in advance:** a body mesh has no hair, clothing or held
  things, which is what the unsettled pixels were. A per-pixel depth of a
  person would settle those directly. Sapiens2 publishes a pointmap model
  (`coderef/sapiens2/README.md`); it is not in the owner's model store, and
  a download is the owner's call (asked 2026-10-10, not answered).
- **Added 2026-10-10 (mrpop, from mrfrog's reading): do not download on a
  guess.** The order between people needs no new model, since the body
  model already places each of them in one camera. A pointmap would add the
  depth of the visible surface (hair, clothing, a held thing). mrfrog's
  advice is to ask for it only if the mesh's order fails on a real overlap,
  which is what mrdeer's test above would show. The three-person order
  above still has no record: it gets a case in `bench/check_body_pose.py`
  when depth goes into the pose table.
- **Added 2026-10-10 (mrfrog): the three-person order, probed again, still
  no record.** On the public frame the check already uses
  (`bench/fixtures/sam3d_body_meta_reference.json`, the `office_frame60`
  image, three typed boxes) this pack's pose node on the CPU put the three
  people nearest to farthest in the order the picture shows them. The
  figures were printed in a session and are not kept; they become a case in
  `bench/check_body_pose.py` with the depth field. Two things the probe
  showed that the field must respect: every person on a frame is given the
  same camera, so their distances can be compared; and a box that is the
  whole frame, on a frame of several people, comes back as one body at
  about the nearest person's distance, so depth is read only from a box on
  one person.
- **Added 2026-10-10 (mrpop), later the same day:** the owner is
  downloading the Sapiens2 pointmap model, the largest size, into their
  model store. Nothing here loads it yet; the pack's Sapiens2 loader
  (`MiniMaxH3Sapiens2Loader`) reads the segmentation model only.
- **Added 2026-10-10 (mrpop): objects are another model.** The body model
  is one of a pair; its sibling reconstructs objects and their layout from
  one image (`coderef/sam-3d-body/README.md`, its opening lines and the
  section on the object model). It is not in the owner's model store, has
  no checkout here, and nobody here has read its code or its inputs.

## How a signal reaches the model

- **A mark on the mesh:** `MiniMaxH3BodyMeshVideo`'s `style` gains `marked`
  (the face side of the head, the back of the head and each hand in its own
  flat colour). In mrfrog's tree, uncommitted at the time of writing; not
  rendered. A text using it must name the colours, and the node is the
  place that knows them.
- **Added 2026-10-10 (mrfrog): the `marked` style is committed,
  86febba3.** The node's second output, `legend`, is the one sentence that
  names the four colours (`body_pose.py::legend`); a graph wires it, nobody
  retypes it. Which vertex is which part is in `body_marks.json`, read only
  when the style is drawn, so a pose pass and its table are untouched. One
  change from the first design, after looking at a drawing: the rig's own
  face is nearly the whole head, so the face side is the part of the head
  in front of the ears (`body_pose.py::FACE_FROM`). How much of the head is
  face side when it faces the camera, faces away and is side-on is printed
  on every run by `bench/check_body_pose.py`, case
  `marked_is_the_mesh_with_parts_painted`, on one public sample turned
  about the vertical; the first two are held, the side-on one is printed
  only. Not drawn on the card and not rendered when this was written.
- **A sentence by rule:** the head angle at a load's first and last frame,
  each put in one of four bins (facing the camera, three-quarters away,
  profile, from behind), the start always written and the end only when the
  bin changed; no times and no side. The bin edges are mrwolf's, reasoned
  and held by no constant yet; they become one when the capture gains the
  column. One render with such a sentence is queued
  (`cont_ctx90_facing_m768_kvrows`).
- **Added 2026-10-10 (mrwolf): the rule as proposed, so it is written
  somewhere.** Take the size of `head_yaw` (the sign is the side, and the
  side is unverified, so it is not used). Under 25 degrees is "faces the
  camera"; 25 to 65 "turned three-quarters away"; 65 to 115 "in profile";
  over 115 "seen from behind". Provenance: reasoned, not measured; the
  edges sit halfway between the four poses a sentence can name, and no
  render has tested where the model's own boundary lies. The start is
  always written; the end only when its bin differs; an end without a time
  is a state, a middle would be an event, and timed sentences in this lane
  have landed early or long (`docs/research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`,
  "Never tried" and its annotation). For the kitchen stretch as a
  continuation the rule gives a start of three-quarters away and an end
  facing the camera.
- **Added 2026-10-10 (mrwolf): which model reads a motion video, and how
  often.** The text encoder always gets it: every twelfth frame
  (`reference_conditioning.py`, the `sample_indices` line), joined in pairs
  by the encoder's tokenizer into one picture block a second
  (`comfy/text_encoders/minimax.py`, the `kind == "video"` branch). The
  video model gets every frame, and only when `motion_vae` is on
  (`reference_conditioning.py`, the `record.use_vae` gate). So anything
  that happens inside a second can only ride the video model's copy; the
  mesh to the text encoder alone loses the timing.
- **Added 2026-10-10 (mrwolf): the size.** One input, `motion_short_edge`,
  sets the size both models see. Its default is
  `video_mask.MOTION_SHORT_EDGE`, whose own comment gives its provenance as
  reasoned for cost and not measured. `docs/h3_references.md`, "Video
  references", has all three vendor implementations putting a reference
  video on the canvas rule. The kitchen arms of the morning ran at the
  default; the owner called that too small, and every arm since runs at the
  canvas's short edge. By the repo's adopt-upstream rule the default is a
  candidate to move; that is a schema change and has not been made.
- **Added 2026-10-10 (mrwolf): a wired video has no zoom.**
  `video_mask.wired_motion` fits the video whole and passes no boxes; the
  zoomed mode exists only for the source's own pixels. A subject who is a
  small share of the frame is a few of the video model's tokens across in
  the mesh (her box per frame is `track_box` in a capture's
  `subjects/<label>/per_frame.csv`).
- **The size of the motion video is not the lever by itself,
  2026-10-10 (mrpop):** the plain mesh at the canvas's short edge, as a
  continuation behind kept frames, did not turn the head on the kitchen
  stretch either (`cont_ctx90_mesh768`; eight stills against the source,
  not judged on playback). One clip, one stretch, one seed.
- **What held the head was the kept frames, 2026-10-10 (mrpop), added the
  same afternoon:** the same stretch as a load of its own that starts where
  the turn starts, with the same mesh, text and seed and no frames kept
  from an earlier window, made the turn, the lowered head and the hand at
  the face (`fresh_0936_m768`; eight stills against the source and the
  mesh, not judged on playback). So on this stretch the signal was in the
  motion video all along, and the frames a window keeps from the one before
  outweighed it. It is not a fix by itself: the stretch lies inside one
  shot longer than a window, so a load that starts there joins the pass
  before it in mid-shot, and the two loads drew the subject's clothing
  differently. The levers on the kept frames are mrnemo's (`context_noise`,
  a shorter context); their renders are queued. One clip, one stretch, one
  seed.
- **Added 2026-10-10 (mrwolf): what that does to the drawing arms.** Three
  were ranked for this stretch before the fresh load landed: the sentence
  by rule as the control; the `marked` mesh; the mesh zoomed on the
  subject. A marked or zoomed mesh as a CONTINUATION here would be tested
  against the same kept frames and say nothing new, so those two are not
  to be spent on it. Where they belong is a stretch on which the plain
  mesh failed with no kept frames in the way: the hands and elbows of the
  kitchen pass (`data/2026-10-10_fun_pose/kitchen_0604_motion.json`, the
  mesh pass against the morning pass), read on a load of its own. The
  sentence stays queued as a continuation, because it asks the one thing
  still open here: whether anything moves a subject off frames kept as
  turned.
- **Added later on 2026-10-10 (mrwolf), correcting the line above:** "say
  nothing new" was too strong. The fresh load shows the plain mesh can be
  read; the continuation shows kept frames outweigh it. Whether a mesh
  that is easier to read outweighs the same kept frames is its own
  question, and the `marked` style behind those kept frames is the render
  that asks it (mrpop's design; one field against `cont_ctx90_mesh768`).
  If it turns her, how legible the signal is counts against the kept
  frames; if it does not, only the kept frames' own levers do. The marked
  style on a load of its own, for the hands, stands as a second arm. The
  style is committed (86febba3) and had not been drawn on the card when
  this was written; the first marked frames are looked at before a render
  is spent, and a render is read for colour from the marks in the picture
  before it is read for the head.
- **The original's own pixels, degraded (the owner's idea, 2026-10-10):**
  the subject cut out of the source carries facing, hair and hands that a
  grey mesh does not, and brings the original's look with it. A blur and a
  grey option on the motion reference are being built (mrnemo); the arms
  are ranked in mrwolf's notes (later the same day: the ranking is in
  this page, in the line after this one). Noise at the model cannot serve: core has
  one level for every visual reference, the still included
  (`comfy/ldm/minimax/model.py`, read by mrnemo). Nothing rendered.
- **Added 2026-10-10 (mrwolf): the ranking and the starting values for
  the degraded original, so they are not only in a session's notes.** Both
  a blur and the model's reference noise remove fine detail first, which is
  where a face's identity is; neither removes large flat colour, and the
  look that has come back on record was mostly hair and clothing. Hence
  the grey option. Order: (1) the clean cut-out to the video model, which
  nobody has run and which sets the size of the leak; (2) a blur whose
  width is half of one of the video model's tokens at the size the readers
  see; (3) the same in grey; (4) a blur of one whole token, expected to be
  where facing starts to go; (5) last, the blurred cut-out to the text
  encoder with the grey mesh to the video model, which needs a second
  window-aligned video and the song node builds exactly one
  (`audio_freeze_song.py`). The width should be a share of the subject's
  box, with the pixels reported. Accepted only if `look` is no worse than
  the grey mesh's on two seeds AND a state the mesh cannot carry comes
  through (a mouth, a held thing, hair). All reasoned; nothing rendered.

## The states a text has been made to carry

Added 2026-10-10 (mrwolf). The rows are every kind of state the hand-written
texts of 2026-10-07, 2026-10-08 and 2026-10-10 carried; the measurement
column is mrfrog's, with trust as mrfrog gave it. An entry above is the
ledger for its row; this table is the map. "Cost" is whether a guessed or
missing instance is on record as costing a render.

| state | what measures it today | trust | wordless channel | sentence by rule | cost on record |
|---|---|---|---|---|---|
| which way the head faces | `head_yaw`; the pose table's head points | points measured against Meta's code; the reading untested on a frame of known answer | the mesh to the video model, face side marked | yes, start and end states | yes, both ways: omitted and guessed |
| head lowered or raised | `head_lift`, per shot only | untested | the same mesh | yes, as a state | timing only |
| a hand to the face | nothing; wrists to nose from the pose table, with `decoder_used` for the fingers | not coded | the mesh, hands marked | possible, but it is an event | yes: drawn early and long |
| which hand holds what | the part node's `held` mask: something is held, not which hand, not what | untested | the thing as its own tracked mask, drawn as a flat shape; the plate when it does not touch the subject | which hand, yes; what, no | yes: a second hand; a thing left where it was |
| mouth open, speaking, singing | lip, teeth and tongue classes; voiced spans per clip | one stretch | the audio; never the mesh, which has no mouth | from voiced spans, for one voice | yes: a voice sentence for shots with none |
| whose voice | nothing run. SAM Audio takes a person's mask over the frames with a text (`meta_perception_models.md`) | on disk, unwired; the mask's direction is contradicted in Meta's own code | n/a | n/a | n/a |
| who is in front of whom | overlap only | not measured | the plate, where the other person is given back to the source | no | none traced to a text |
| how large in the frame | `track_share`, the boxes | measured | the hole | not needed | none in this lane |
| what in the region is not the subject | the region's share that is not the person | measured; the things are unnamed | the plate: keep them out of the region | no | yes, both ways: redrawn as other things; drawn in the next shot |
| hair over the face | hair against face classes | untested | none | untested | none traced |

Outside the table: order and timing words ("then", "near the end") need no
measurement, they need deleting. An object's place in 3D needs a scene
model that is not in the owner's model store.

## Who holds what

| Part | Session, 2026-10-10 |
|---|---|
| the columns in the capture (facing, face showing, head lift, hand at the face, front and behind) | mrdeer |
| the pose table, the marked mesh, depth in the table | mrfrog |
| the rule from a column to a mark or a sentence, and the test renders | mrwolf |
| the blur and the grey option on a motion reference | mrnemo |
| the job builder that reads the columns | mrpop |
