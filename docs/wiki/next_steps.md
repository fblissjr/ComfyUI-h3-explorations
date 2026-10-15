# Next steps

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept short on purpose: when it
disagrees with [`../roadmap.md`](../roadmap.md), the roadmap is right and this
page is stale.

**Starting a day: read [`handover.md`](handover.md) first.** It is what the
last working day left, in the order its lead would take it up, and it opens
with what not to assume.

## To-do (the owner's action list, 2026-09-05)

**Pointers only, one line each, consolidated across the sessions working this
tree.** Each names the record that closes it. The reasoning lives where the
pointer goes; this list is stale the moment the roadmap disagrees with it.
Items from the `mrpink` helper lane first; `evalman`'s render lane appends its
own below the rule.

**The queue at the close of 2026-10-14 (the lead, mrtwo). The card's order and the tooling order are
[`handover.md`](handover.md), "Tomorrow, in order", and are not repeated here; the day's rulings are
[`decisions.md`](decisions.md), 2026-10-14. Of the list below, items 1 and 2 and the floor of item 6 were done on
2026-10-14 (`--parts-on`, the track hold, `--track-not`, the review tools as tracked tools, the zoom-held floor); the
rest is carried.**

Tooling, in order (each is a row further down this page unless it says otherwise).
Before any tool for a new kind of load: one dry run of that load to its gate on
the card, and build from what it stops on.
1. What stands between the owner and the gaps they saw on the joined clip: a
   part that changes inside one load (`--parts-on`); the reference step held to
   the subject's own track; a track that lies on another subject, and a way to
   say so on named frames; a zoomed load whose keep-outs block its first
   capture.
2. What the owner reviews from, as tracked tools with a check each (built
   2026-10-13 as session scripts under `internal/`): the per-shot clip with
   every piece laid; the local review file; the provenance page, with a way to
   write to the sessions from it once the owner says what they want to send;
   the review page's server and its page check.
3. The gaps print in `shot_parts.py` and in the job step (the table prints it).
4. The sound signals for a window's edges (`bench/clip_signals.py`): its third
   run read by somebody who did not build it, then the figures into a load's
   details and onto the plan strip, then the picture rows of
   [`window_edges_data.md`](window_edges_data.md).
5. A pair measure as a tracked tool; a zoom box that follows the subject.
6. The rows of 2026-10-13 not started: a zoomed load's floor at the size it
   renders at (built 2026-10-14); overlapping keeps ordered by the keep's time; a shot between the
   planner's longest length and the model's; the plan strip drawing a zoomed
   load's box (its words are done); `break_checks.py` and `check_masked_job`; choosing among several
   boxes in one cut; how much of its box a person fills on a who picture.

Carried from 2026-10-12: the third subject's other shots, face only, and
1222-1305 as the first two-subject render on a day the owner can look. (The
hair band as the fix for 693-703 is now the row "Her hair comes and goes beside
her head" below: the fault was found elsewhere on 2026-10-14.)

**Masked video to video, 2026-10-15: rows of the day.**
- **The lead's face-only pieces change her hair too, so each piece's edge is a visible seam** (2026-10-15, the outside
  reviewer's watch of the 604-1009 cut with sound, `internal/gemini/2026-10-15_watch_0604_1009_and_5675b.md`, section
  1.2; not yet checked by a session's eye). At 750/751, 778/779, 851/852 and 965/966 her hair snaps between the
  original's curly strawberry-blonde and the render's darker auburn, not only her face. Likely, NOT VERIFIED: the hair
  band (a chain default, `h3_config.MASKED_JOB_HAIR_BAND`) regenerates hair round the face, so a piece edge inside a
  shot shows. Wanted: the hair band's share of each lead piece's region, and the readers of 630d and 805d told to look
  at her hair on each load's first and last frame. A row, not a job; the owner's eye on the cut decides whether it
  matters.
- **The outside reviewer sees video at one frame a second** (its own report, 2026-10-15,
  `internal/gemini/2026-10-15_clips_review.md` section 1 and C1b; not checked independently). A frame-level question
  goes to it as frames or a 1 fps companion strip (mrvid's review-clip build), and its answers say which frames it saw.
- **The strangers keep-out keeps the subject's own neck, dress and hair beyond her track as strangers, on a close-up with
  nobody else** (2026-10-15, the lead's ruling; option 2 of mrlol's three). bench/strangers.py hands back a piece that
  touches a subject and holds no body core, and keeps one that holds a core: ruling (A) of 2026-10-14, another person's
  arm or face against her stays a stranger. On lead_fp_2823b's strangers record, 263 pieces were kept though touching
  her, 7,657,313 px, by core: Face_Neck alone 90, Upper_Clothing 38, Hair not joined to hers 31, Hair with
  Upper_Clothing 29, Face_Neck with Upper_Clothing 24, with an arm 11. Option 1, a person's word per load (`shot_to_load
  job --no-strangers WHY`), is built today. THE ROW, for a day the card can spare a part-model rerun: extend the hair rule
  to every core class (a piece whose Face_Neck, clothing or arm pixels join the same class inside her own, pixel by pixel
  through her grown band, is hers) and measure it, since the whole-frame class map is not kept. FIRST LOOK: mrgem's G9
  (`internal/gemini/2026-10-15_watch_strangers_0630d.md`) on lead_fp_0630d's pieces touching her, 654-680, sheet
  `data/2026-10-15_strangers_touching_0630d/`: every one is her own (fingertips on 654, her hair or scarf tail on
  669-673, her dress under her wrist on 678-680); the pieces touching nobody are the flower pitcher's handle and base
  and a speck by a dish, not a person. Figures: `internal/claude/2026-10-15_mrlol_scripts/strangers_on_her.txt`.

**Masked video to video, 2026-10-14: rows of the day.**
- **Gaze is a known limit: the render does not follow a horizontal gaze it is shown, and the pose model misses a
  vertical eye roll** (2026-10-14). On lead_fp_1097b at 1119-1133 her iris sits at +0.25 eye widths to screen right
  (median, both eyes) and the render's at -0.02; two sweeps with only the reference changed, the iris filled and the
  iris in contrast (white sclera, black iris), rendered on the whole frame, put it at -0.03 each: the reference's eye
  form moved nothing by this measure (the lead's `iris_offset.py` on each render's held pose pass;
  `internal/claude/2026-10-14_mrdrill_scripts/iris_offset_render_iris*_1097_1133.csv`). A zoomed sweep would have
  rendered on the whole frame (her raised arms on 1097-1104 widen the box), so the zoomed case is open: a load on fewer
  frames, if the node's box allows. Upstream of the drawing: on lead_fp_1156b's held reference at 1158 and 1160 her
  eyes roll up with the whites below the iris, and the pose points put the iris centred in an open white eye
  (mrcircle's sheet `data/2026-10-14_fun_batch_r_inputs/lead_fp_1156b.eyes_1158_1160_1170.jpg`), beside 415's
  lowering (`internal/claude/2026-10-14_mrtwo_scripts/eyes/findings.md`, section 3). The zoomed case came out the same:
  lead_fp_1097c, zoomed 1.80x with the chain's defaults, leaves her iris near the middle of the eye where hers is to
  the right (`internal/claude/2026-10-14_mrdrill_scripts/iris_offset_render_zoomed_1097c_1097_1133.csv` against the
  original's), so the eye form gets no more spend. PARKED, NOT A JOB: the outside reviewer's section 11
  (`internal/gemini/2026-10-14_relay_gaze_hair_expressions.md`, lines 263-293) reads it by the latent grid: at that
  zoom her whole sideways shift is about half of one 16 px cell, while the open mouth that did carry on 1156c spans
  several, and the still, fed large with its irises centred over many cells, outvotes a sub-cell cue (its table has
  the figures). If gaze is revisited: a 3.5-4x zoom on a short, nearly still glance window (1121-1133) puts the shift
  over a full cell; and a still with the irises turned to the glance. Its third idea, naming the glance in the text,
  is out by the owner's one-form rule (no per-load face words).
- **The pose model's lid points sometimes miss a lowered or closing eye, and the render opens it** (2026-10-14, the
  lead's reading; the owner's old fault on the second subject, recorded with its cause). On second_hfp_0278c the
  render's eyes open on 290-293 while hers stay lowered, and the reference draws them open there because the lid
  points open on 288-293 (278b's reader). On second_hfp_0415d at 437 hers close and the render's stay open. The lid
  gap a frame, original against render (the lead's `eye_open_table.py`):
  `data/2026-10-14_second_hfp_0278c/eyes/lid_gap_render_vs_original.txt` and
  `data/2026-10-14_second_hfp_0415d/eyes/lid_gap_render_vs_original.txt`. Upstream of the drawing, beside the gaze
  row above and 415's lowering. Not a job today.
- **On a laughing squint the drawn lids stay open, and the body model's eyes are open under every drawing**
  (2026-10-14, the lead's look at the sheets of the iris drawn between the lids, the lead's 3240-3276). Where
  she squeezes her eyes shut (3247, 3261, 3276) the lid outline stays an open almond: the pose model's lid points
  do not close fully on a laughing squint. And the body model's own sculpted eyes are open on every frame,
  whatever is drawn over them. Both are in the drawing's record ("not_known"). A row, not a job: a render of
  3240 carrying the iris between the lids decides whether either matters. Not started.
- **The reference draws her eyes open where she laughs with them shut**
  (2026-10-14, the lead, from `lead_fp_3167`'s blind reading,
  `data/2026-10-14_lead_fp_3167/look_blind/reading.md`). On 3167-3213 the
  held reference draws the lead's eyes open on all 47 frames while she laughs
  with her eyes shut. Likely, NOT VERIFIED: the pose model's lid points on a
  face about 109 px tall (the held pass's median, 2026-10-14) do not close.
  The same kind of fault as the second subject's eyes that do not look down
  (the owner's "wrong" on three pieces of 10/12); her test on 484-515 is the
  next reading. Wanted, once that test is read: the eye opening the drawing
  uses, against the original's on the same frames, as a figure a frame
  (`bench/_lib/face_points.py::eye_opening`). A row only. Not started.
- **The drawing of a face in profile draws the far side and stray lines**
  (2026-10-14, the lead, from the card's sheet of the held reference of
  `lead_fp_1466b` and its blind reading,
  `data/2026-10-14_lead_fp_1466b/look_blind/reading.md`). The lead in
  profile, looking up at the third subject, 1466-1534: the far eye and brow
  are drawn in front of the body model's face, off its outline, and long
  straight stray lines run off the face on 1478, 1496-1497, 1503-1506,
  1509-1510, 1512, 1515 and 1524-1533. Likely cause, NOT VERIFIED: points of
  the hidden side, scored low, are still joined into lines by
  `bench/draw_face_points.py`. Wanted: the drawing leaves out a line whose
  points are scored under the reading value, or that leaves the body
  model's own face, and prints how many it left out on which frames.
  Built after the track-not-that-person row (3b) unless it shows on the
  owner's playback. Not started.

**Masked video to video, 2026-10-13: two rows from the day's last hours.**
- **The who page numbers a shot on one frame only, so a subject who comes back late cannot be clicked where he
  returns** (2026-10-14, parked by the lead: 14 frames of a far third subject are not worth a card pass today). On
  the record's shot 4647-4725 the one preview (shot_c_4647b) numbers each tracker's one cut at 4666 only; the third
  subject is hidden behind the lead on about 4679-4710 and shows again at the right edge from about 4712, where no
  preview saved a detection. The quarter frames give a frame to say he is there, but no box to seed his tracker.
  The shape when it is wanted: a preview whose table numbers a frame in 4712-4725 (a card pass), and `who_page`
  showing, for a frame named on the command line, that preview's numbered picture beside the shot's own, since the
  page shows one picture a shot today and would drop one (about 40 lines). A plain `--frame` with the two frame
  buttons is the cheaper half and records presence only.
- **A zoomed load has one box, so a small face that travels cannot be
  zoomed** (2026-10-13, from the holder of the card on the lead's 168-184).
  The node zooms a load into one box that must hold the subject with its
  room on every frame; where she walks across more than about half the
  frame the box is larger than the frame allows and the node refuses the
  zoom (the node's own line in
  `data/2026-10-13_lead_fp_0168_job/lead_fp_0168.plan.queued.json`: the
  load is rendered on the whole frame, with the sizes it compared), and on
  the whole frame her face is too small to render.
  Wanted: a box that follows the subject a frame at a time, or a box a run
  inside one load. Not known: what a moving box does to the reference and
  to the lay-back at a frame where the box steps; that is the first thing
  to read, on a plan, before anything renders. Not started.
- **The review page's server is an untracked throwaway** (2026-10-13,
  from the owner asking how to start it by hand). `serve.py` and
  `check_page.py` under a video's untracked `data/<video>/poc/` folder
  have served the owner's review page every day since 2026-10-11, and no
  tracked file held how to start or stop them until
  [`../comfy_notes.md`](../comfy_notes.md) said so that evening. Wanted:
  the server and the page check as tracked tools with a check each, the
  page's folder handed in by path, nothing of a video or an address in
  the code. Not started.
- **A pair measure is two session scripts** (2026-10-13). Two renders of
  the same frames were set side by side twice that day, each time by a
  script of the session that asked: sharpness inside the region as
  delivered, how the mouth's opening follows the source's, the step across
  a frame where a piece begins
  (`internal/claude/2026-10-13_mrtower_scripts/measure_pair_1920c_1920d.py`,
  untracked, and the two-shot session's beside its notes). The owner's
  rule for two keeps on the same frames asks for exactly this where the
  other grounds do not settle it ([`decisions.md`](decisions.md),
  2026-10-13). Wanted: one tracked tool that takes two landed loads and a
  span and prints those figures with what each was measured on and what it
  cannot see, and a check; no verdict, no threshold. Not started.

**Masked video to video, 2026-10-12: gaps in the tools, each found by a run; several
fixed the same day, said inline below.**
- **A sweep tool: the same loads again with one thing changed** (the owner, 2026-10-12,
  19:35Z on the Open List's inbox and 19:40Z in the lead's terminal: once the windows are settled, run the
  same windows on the same input with only the two stills changed; "make sure we have on the backlog a tool
  to build that repeatable process"). Every load is already a job file naming its stills, capture, motion
  video, seed and settings by path, so the sweep is: a list of job files, one change named (a subject's
  still, a text, a motion video), each copied under a new name with that change and nothing else, built,
  looked at and queued through `bench/masked_job.py` on the same captures, with a record that says which
  loads, which change, and where each render landed. `bench/sweep_loads.py`, with a check that the copies
  differ from their sources in the named key only. About one afternoon. Not started; after the forms settle.
  **Built 2026-10-12, its first cut**, at the lead's go once the owner had set the form: `bench/sweep_loads.py make`
  writes the copies beside their sources and the sweep's record under `data/<date>_sweep_<tag>/`, and prints
  `bench/masked_job.py`'s commands in order for the loads the change touches; `landed` reads where each render
  landed into the record. It builds and queues nothing: the queue is the card holder's and a look sits between a
  build and a queue. Held by `bench/check_sweep_loads.py`. Not yet run on a real sweep; what is left is to use it.
  **Its first real run, 2026-10-12** (the card holder, the lead's three hall loads into the cut-reference form)
  met three things, each answered in the tool the same evening: a new reference always needs the text that
  describes it changed with it, so `--motion-file` with `--text` is now one change; a copy kept its source's
  capture, where the lane keeps one capture a load, so `--capture JOB=FOLDER` names the copy's own and a copy
  without one is told what a build there can write over; and a new reference that begins on another frame is
  given as `FILE@N`. A copy on other frames (`first`, `frames`, `left_out`) is another load and stays by hand.
- **The capture tool reads a two-subject load as one subject's** (2026-10-12, when the load was built). A
  load's `subjects` reaches `bench/capture_masked_run.py` as its FIRST subject's load
  (`workflows/masked_job_graph.py::one_subject_view`): the second subject's part is inside the region and is no
  subject of the plan, so the rule about a region over another subject fires on it and is answered by hand,
  and nothing the capture measures of a region (what is inside it that is not its subject, the part's
  coverage, the pose rows) is taken for the second subject. To build: a plan with two subjects, each rule read
  per subject of the load, and the strip drawn with both. Starts at `loads_from_file` and the roughly sixty
  places that take a plan's `subject` as one label. **Partly built 2026-10-12** (the first real two-subject
  load met it: the look's strip refused the load, and the capture's own plan was the first subject's alone):
  the plan the capture works out is from both parts, the run records `subjects`, and the look's strip takes
  the load, says which label is whose and draws both. **The rules, the same day**, in the shape the lead ruled
  (a run keeps `subject` and gains `subjects`; `subjects_of`, `by_subject`): no subject of the load is "another
  subject" to it, so the rule about a region over another subject and the rule about another subject's segment
  inside the region no longer fire on the second subject and are no longer answered by hand; and a rule that is
  a fact about a subject runs once per subject of the load (a shot of the load without them, their size over
  the load and at its start, a face a few tokens tall, an open mouth with no voice, the pose ahead of kept
  frames, the class map for a planned part, a statement that excuses frames a load covers). A capture has to be
  built with the tool as it is now for any of it to apply: the rules read `subjects` off the run. Met no real
  pair but one trial capture of the long kitchen shot's load, built and gated in a session's scratchpad.
- **A few rules still read a two-subject load as its first subject's** (2026-10-12, named when the rest was
  built), in three groups. Each reads the ONE mask the node is given, both parts together, and cannot say whose a missing part
  is without splitting that mask by track, which nobody has decided how to do where the two tracks overlap.
  (1) `load_lays_nothing_on_frames` and the lent-frame rule: a frame where the second subject's part is empty
  and the first's is not is named by neither, since something is carried on it; the per-subject rule
  `part_empty_on_the_subject` does name it. (2) `original_shown_in_the_held_tail`: read for the first subject's
  mask on the load's last frame only. (3) `region_carried_across_a_cut` and `kept_pixels_inside_the_part`: the
  first subject's shot table and part. To build when a real pair shows which of them matters; start at
  `bench/capture_masked_run.py::by_subject`, whose docstring says why these do not walk it.
- **Three people in one pass has no route** (the lead, 2026-10-12; frames 1920 to 6394 have shots with all
  three subjects briefly). The Masked Source has two reference inputs (`motion_video`, `motion_video_2`) and
  the builder's `subjects` list takes two. Two routes, neither started: a third reference input and a third
  entry in the list, which is node code and a restart, with one margin for three; or a two-subject pass and a
  single pass on the same frames, merged by the delivery, which is what the tools already do and costs a
  second render and a join inside one picture. Nothing is built before the previews of those frames say which
  shots hold three and for how long.
- **Her hair comes and goes beside her head on the long kitchen shot (604-948):
  a keep-out's mis-cut strip, and the band that did not reach it** (2026-10-14,
  the lead's diagnosis of the owner's annotated frames 689, 693 and 703:
  `internal/claude/2026-10-14_mrtwo_scripts/hair_0604/findings.md`). The clothes
  keep-out's track (`data/2026-10-12_second_hf_0604_b/subjects/clothes/`) has a
  thin strip down the original woman's hair edge. Every cell it touches beside
  her head is handed back to the source (`video_mask.py`, the keep-out rule), on
  90 frames of the render, stepping every latent step. **LANDED, in two steps.**
  fa5c58ac: the hair band starts at the part box's TOP
  (`h3_config.MASKED_JOB_HAIR_BAND["starts"]`), so the clothes keep-out is
  emptied beside the head as well as under it. 7504bbc4: the band reaches the
  load's own margin past the part's box each side (`wider: "margin"`), where a
  tenth of the box's width had left a column of clothes cells at each shoulder,
  inside the margin, for the clothes keep-out to hand back (second_hfp_0484b's
  frame 488, the owner's annotation; the read is
  `internal/claude/2026-10-14_mrdrill_hair_read.md`). **THE OWNER'S EYE on the
  first render carrying both, second_hfp_0484c (2026-10-14, about 18:15Z): "hair
  stays put. nice work."** Left there by the band: her held thing and her hands
  (the necklace and pendant under her chin, by the lead's look, stay the
  original). Open: 604-948 itself made again with the band; four held cells
  beside her neck on 486-490 of 484b, the lead's look on 484c's render.
  WHAT THIS ROW CLAIMED UNTIL ABOUT 18:15Z 2026-10-14: "BUILT, NOT LANDED" (worktree
  branch `hair-band-from-top`, d2591ad5), the band in no plan, and no render with it.
  WHAT THIS ROW USED TO CLAIM (2026-10-12, the row "Her long hair flickers at the
  region's edge"): that the region's LOWER edge was the clothes keep-out giving
  the margin's cells back under the jaw, so the hair was drawn where the region
  reached and cut where it did not; and that a band under the part (`hair_band`,
  built that day from the box's bottom) answered it, untested. The hole is
  beside the head, not under it, and a band from the bottom never reached it.
- **The composite's `only what changed` decides, per pixel and per frame, where a drawn hair end stops: a
  node-composite question, not a job today** (mrpre, 2026-10-14, on the owner's "hair clips a bit on bottom left but
  its ok move on" on second_hfp_0604d; the read is `internal/claude/2026-10-14_mrpre_hair_room_0604d_read.md`, its
  last section). At her right shoulder on 604-622 the node's region reaches well below her drawn hair, and its edge
  cuts almost none of it. But 35-52% of the region there is shown as the original, nearly all of it outside her part,
  because `video_mask.changed_alpha` keeps the render only where it differs from the source or where her original part
  was. The line between drawn hair and original is the composite's threshold, frame by frame: the white "kept" outline
  of the render's own mask view (crops `internal/claude/2026-10-14_mrpre/crops/hair_0604d_bottom_left_604_612.jpg`
  and `_614_622.jpg`). It is a candidate cause of more than one come-and-go of the day. The question for the node:
  whether the changed mask should be held steady over a latent step's frames, or its threshold softened toward the
  region's margin. A look-spec room (`bench/look_rooms.py`) is today the one place the composite is told to keep the
  render whatever was drawn, since a room is joined to the carried mask. What it cannot see: the drawing against the
  composite. The window's video on disk is composited, and telling them apart needs a decode of the window's latent
  before the composite, on the card.
  **A second shot, second_hfp_0014d** (the owner's frames 57 and 163: a glitch at her neck where her hair ends, at the
  right of the picture; the owner's screenshots and crops are in `internal/claude/2026-10-14_mrpre/`, `neck_0014d_*.jpg`).
  It splits two ways, and neither is the keep-out the owner's frame suggests:
  - **On 50-65 it is the drawing.** The region holds no hole there, and the composite keeps the render over the
    place. Beside her drawn hair's end the model draws a dark patch where the original has its dark hair.
  - **On 159-166 it is a hole in the node's own region:** 432 px on 159-162, then 84-144 px on 163-166, exactly two
    latent steps. Held pixels lie in it on 161-162 and the strangers keep-out on 164-165. The region is one per
    latent step, and a keep-out cell on one frame of a step gives the cell back on all four. So the strangers' 22 px
    on 164-165 hole the region on 163 as well, where the strangers mask is empty.
  That is a third candidate beside the composite and the drawing: a keep-out's one-frame speck shared across its
  step. Read on 2026-10-14 at the day's close, from the plan, the render and the capture. Not checked: which cell of
  the zoomed grid each speck falls in.
- **Nothing prints whether a body model is on its subject.** A preview posed
  the person behind the subject for half a second and every table read the
  same. The node is fixed (`../../bench/results/2026-10-12_body_model_posed_the_other_person.md`);
  the figure that would have shown it is not built: the body model's head
  against the subject's own part or track, a column a frame in the capture's
  pose rows, printed with what it was measured on. Body-model videos made
  before 2026-10-12 on shots where two people overlap have not been measured.
- **`bench/capture_masked_run.py files --overwrite` leaves a subject's files
  from the build before** when the new build gives that subject fewer inputs
  (a class map, a part), and the gate's last step then reads them: it crashed
  on a class map of another length after writing `flags.json`, and before
  that it raised flags about a part the build no longer had. Until it clears
  what it no longer writes, build a changed capture in a new folder.
- **`bench/draw_lid_marks.py`'s read-back fails on a frame left bare when the
  body model has pixels near white**: the re-encode pushes a few of them over
  its line and the test wants none. Read on one run: the pixels were spread
  over the body and none was at an eye. **Fixed 2026-10-12**: a pixel is a
  mark's white only where the body model under it was well under white
  (`bench/draw_lid_marks.py::NEAR_WHITE`, shared with
  `bench/draw_face_points.py`, which met the same thing); the check holds a
  bare frame with a lifted highlight to no white.
- **`bench/capture_masked_run.py video` will not stack a render that starts
  inside the capture's span** ("the render starts after the span's first
  frame; capture the span the render covers"). A run that continues a stored
  window saves only the frames it writes, so its file starts where its kept
  frames end, while its capture spans the frames its text was written from.
  Every continuation meets this. For the first one the two renders were
  joined into a viewing copy and stacked on the first load's capture, whose
  region on the new frames equals the continuation's. The strip already has
  an offset for a load's own strip (`--load`); a render placed by
  `--render-first` inside the span wants the same, with a case in
  `bench/check_capture_masked_run.py`. **Fixed 2026-10-12**: the strip is
  the frames the span and the render both hold
  (`bench/capture_masked_run.py::render_span`), every frame of it names in
  its panel the span's frames it does not show, and the command prints
  which source frame it took the render's frame 0 to be and on whose word.
  Run once on a real load's own capture (the hall shot whose first five
  frames are left as the source); a render that is not one of the
  capture's runs still needs `--render-first`, and without it is placed on
  the span's first frame, which the printed line says.
- **`bench/recomposite_window.py` writes tables and no pictures.** It lays a
  stored window again with one setting changed and prints what moved; to see
  the layings, a session script called the tool's own functions (the saved
  region, the kept decode, `video_mask.lay_window`) and wrote the frames.
  Run once, on the first window of the long kitchen shot, for three layings
  of 33 frames and a box of them side by side with frame numbers; it will be
  run again. Wanted as an option of the tool: write the frames of a laying
  as a video, and several layings side by side over a box, with the source
  frame number burned in. Not started.
- **A line of the page's answers file is the owner's because it says so.**
  `bench/page_answers_to_cast.py` takes a line whose `who` is `owner`, and
  `who` is whatever the page's "Answering as" box held when the click was
  posted; the page server does not know who clicked. A session's test click
  left with the box saying `owner` would be stored as the owner's, in the
  cast file and, since 2026-10-12, as `present` in a shots record. Nothing
  such is known to have happened; sessions' test clicks have carried a
  `who` of their own, which the tool lists as left out. What would close it:
  the page server stamps on each line what it can know of where the click
  came from, and the tool reads that beside `who`. Not started (found by a
  review of the commit that added the shots record, 2026-10-12).
- **The check that a render's graph is its plan's graph is a session script, and it stops on a hair band.** Before
  a render is queued the card holder compares the render graph with the plan graph and lets them differ in the
  song node's preview switch and file prefix only. That comparison lives in an untracked script handed from one
  card holder to the next, with no check of its own. The builder writes a hair band's cut keep-out once a stage,
  under the stage's name (`<load>.plan.<label>_less_hair_band.mkv` and `<load>.render.<...>.mkv`,
  `bench/masked_job.py::hair_band`), so on the first band load the two graphs differed in that one input and the
  script stopped; the card holder decoded both files, found every frame equal, and queued by hand. The rule to
  build, as a tracked tool or inside the builder's own `queue` of a render: an input that is a file the build
  writes per stage is compared by content, equal content passes with both names printed, different content
  stops, and anything else that differs stops as now. Not started (2026-10-12).

**Masked video to video, 2026-10-11: eyelids from marks on the body-model video held on one shot and did not repeat on a second.**
- Start from [`handover.md`](handover.md): what not to assume, where the
  day's folders, tools and one-offs are, and the order for the next day.
- The two records: `../../bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`
  (it worked there, as a level; the owner picked a render) and
  `../../bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md`
  (it did nothing for her eyes; what differs between the two loads).
- **Owed, on the owner's word: one render**, the close-up with the marks
  at three times and nothing else changed; before it, a look at what of
  the reference the model is shown, which costs no render. Built and
  held behind it: the shorter wording, and the third subject's shot.
- **Not to be done until the form has held on two shots**: the builder's
  motion choice and a generated workflow for it.
- Owed and not started: the record of a video on the share with the tool
  that takes the page's raw answers into it; trial 2 of
  [`meta_perception_models.md`](meta_perception_models.md), "The trials".
  *The pose pass along a shot and a tracked home for the pose model's
  point names (`vendor/meta_sapiens2/`), both listed here as owed when
  this block was written, landed the same night.*
- Owed after the owner's cleanup of `vendor/`: the comments in code
  that cite the removed Sol reference file
  ([`handover.md`](handover.md), "State when the day closed").
- **`bench/check_sol_kernel.py` is to be slimmed**, agreed with the owner
  and not done: it keeps its printout of which kernel build answered,
  `observable` and `schema`; its other cases go, with the tables and
  header paragraphs only they used (two of them read the reference file
  that is no longer in the tree). A node-adjacent check: whoever does it
  reads each case before it goes, and does the stale comments above in
  the same stretch.
- **The direction for the capture tool, from the owner at the close**
  ([`decisions.md`](decisions.md), 2026-10-11, the last entry): the whole
  path of each thing a node uses, source to target, every change with
  the node and values that made it, for every window; and a capture that
  says up front whether the source's shape is the canvas's. Not sized.
  First step: list what the saved graph, the plan, the saved region and
  the tables already hold.
- **The first two things to build, both wanted and neither started**
  ([`handover.md`](handover.md), "Tomorrow", has each in full): the
  layered view of the preprocessing, shared helpers first; then a
  generated workflow for the lid-marks form, after the close-up's next
  render is read. What exists in its place:
  `../../bench/examples/masked_job_lid_marks/`.
- **The local page reorganized** (the owner, 2026-10-12 about 17:23Z, in
  the lead's terminal: it is impossible to find what is important and what
  is stale). Three sections and nothing else: look at now; the video a
  shot at a time over the whole span with each shot's plan, renders and
  verdicts; the stale kept folded and dated. The shape is in the lead's
  handoff (`internal/claude/2026-10-12_mrfern_handoff.md`, the last
  section). First tooling job for the lead; the page's code then goes into
  the tree as `bench/review_page/`.
- **The lead's four singing loads** (the owner, Open List rows p05 and t66,
  2026-10-12 17:06-17:07Z): her face only on the three hall shots with
  the singing sentence, and the street shot 294-414 with none, built on
  the record as it stands. *Corrected 2026-10-12 about 18:46Z by the
  lead: the three hall shots are the rows' own, by their times (about
  47 s, 49 s, 52 s): 1093-1133, 1156-1188 and 1222-1305. The list that
  stood here (with 1134-1155) came from a session's reply on row p05.
  The still cut to head and hair is not part of these four: it is row
  g05's test of the second subject's still on 14-154, a load of its own
  after them; the lead's still is already a head crop.* Found while
  building them, each a gap and none fixed:
  - **1134-1155 is left as the source.** Her face is hidden under her
    hair for a third of the shot (the preview's part is empty on
    1145-1146 and a few hundred pixels on 1144 and 1147-1149); sheet on
    the share, `Video/fun/mrhorn/lead_looked_at/lead_part_1141_1154.jpg`.
    No face load of this shot is built.
  - **The voice table has no voice on 1093-1188**
    (`data/2026-10-10_fun_audio/voice_spans.json`: nothing between 1049
    and 1201), though her mouth is open in the picture on much of it.
    What it cannot see: a voice the separator put in the other stem.
    Each load's note carries the figures.
  - **On the street shot the body model is not her pose on 316-320**,
    where a cyclist hides her (sheet beside the one above,
    `lead_part_0315_0358.jpg`); that load takes no motion video. And her
    face part is under the capture's least size
    (`bench/capture_masked_run.py::PART_LEAST_PX`) on about half the
    frames, so the gate blocks: nothing would be laid there. Waiting on
    the lead's ruling between a zoomed load, the part as saved, or
    holding the shot for row p04's zoomed-load work.
  - **A capture over a preview of several shots asks who is who** on
    every shot its tracker took by likeness, though the owner answered
    those shots on the who page (`data/fun/shots.json`). Nothing carries
    those answers into a preview's corrections; today each such shot got
    a preview of its own, picked on the owner's frame.
  After the second subject's loads on the card. The zoomed load that keeps
  its motion video (row p04, 17:05Z) is the node change after that, for
  the street walk zoomed and the phone insets, and for the second subject
  on 1010-1038 (the owner's row 11, 17:47Z: wait for the zoomed load).
  **Built 2026-10-13 (mrolaf), the node and the builder, not rendered**: a
  zoomed load takes a `motion_file` it names and the node hands that video
  on uncut (`video_mask.zoomed_source`; [`masked_v2v.md`](masked_v2v.md),
  "A small subject, rendered zoomed in"). The preview's own whole-frame
  body model stays refused on a zoomed load.
  **The zoomed preview stage is NOT to be built on what was known**
  (2026-10-13, one run at the lead's go; this paragraph used to say the
  preview stage "has no zoom", that on 1010-1038 the part model "at
  whole-frame size calls her hands face from about 1025", and that the
  preview run on an enlarged picture with its tables mapped back "is a
  tool to write"). Two things were wrong in it. Every model of the
  preview but the tracker already works on a crop of the subject resized
  to its own input (`sapiens2_parts.take_crops`; `body_pose.py`;
  `bench/pose_along_shot.py` takes a box), so enlarging the picture first
  hands them the same pixels. And the run says so: the same preview on
  those frames cut to a box round her and enlarged twice gives the face
  class the same area and the hands as little as before
  (`data/2026-10-13_second_cz_1010/READING_mrolaf.md` and the two tables
  beside it). In a look at five frames of the cut, from about 1025 her
  head is bowed and no face shows: what is called face is her hands under
  her hair. That stretch needs a decision (the source, or her whole
  person), not a tool. Not tested by it: a shot where a small face does
  show, a box round the head alone, the source's pixels without the
  canvas's reduction.
  The plan of a zoomed load (`render_region: part`) is cheap today and
  says the box the node would take; it was run on 1010-1038 as
  `second_hf_1010_z` (mrhorn).
- **The stacked form of a joined clip was a one-off twice**
  (`data/2026-10-11_fun_joined_b/`, then
  `data/2026-10-13_fun_joined_stacked/build_stacked_join.py`). **Built
  2026-10-13 as a tracked tool**: `bench/stack_delivery.py`, reading the
  table the assembler built the clip from, with
  `bench/check_stack_delivery.py`. Not in it: the parts in their own
  colours, and a zoomed render's mask at its whole-frame place.
- **One load from its preview to the page is one tool** (2026-10-13):
  `bench/load_chain.py`, a step each for first, plan, on-plan, render,
  land and page, reading the job file. It replaces the five session
  scripts every load went through, which took thirteen typed arguments
  and held the day, one video's paths and one machine's folders in their
  text. A job file needs `preview` (the preview's job folder) for it, and
  the video's manifest its voice tables (`clip_analysis.py voice`). Not
  in it: more than one load a job file (refused); the loads made before
  it ran with `pieces=largest`
  typed into the script, which their job files do not say.
- **The step before the chain is one tool** (2026-10-13):
  `bench/shot_to_load.py`, a step each for the shot's preview, a load's
  job file and its head reference, from the shots record, the plan and
  the cast file; it stops for the text. It needs each subject's `still`
  and `parts` in the plan, and reads a tracker's pick frame from the
  shots record's `frames_on` and `frames_boxed`, which the record has
  only once `bench/page_answers_to_cast.py --shots` has made it again;
  until then every subject the cast file has no box for is picked on the
  shot's middle, and the job's note says so. Not in it: a whole-person
  load, a load of two subjects. It has run no real load.
- **The delivery table is generated, never typed** (2026-10-13):
  `bench/render_ledger.py` (every render, how it was made), then
  `bench/delivery_table.py` (which render stands on each shot by one
  stated rule, the reading of why, and the assembler's command whole).
  Owed to it: a field in the job file, carried by the build record, that
  says why a load took the plain cut (today the reading says the reason
  was not read, and the lead hands it in with `--prefer`).
- **The shots record has no frame fields yet, in either half, and a
  marked frame is not a pick** (2026-10-13, from the chain's first real
  preview). `bench/shot_to_load.py` picks an unseeded subject on a frame
  the shots record names as a field (`frames_on`, `frames_boxed`), and
  `data/fun/shots.json` has neither on any entry, so every unseeded
  subject is picked on its shot's middle and the job's note says so.
  - *Frames 1920-6394.* Those entries were written by
    `bench/page_answers_to_cast.py --shots` the evening before the fields
    existed. Run on the same answers again it writes them; nothing else
    about those entries changes.
  - *Frames 0-1919.* That half was consolidated by hand from several
    sources and the tool cannot make it again. Its frames are in the cast
    file: every line there with a box is a frame a person outlined the
    subject on. What is wanted is a mode of the same tool that adds the
    two fields to the entries that are there, from the cast file's
    standing boxes and the plain answers, and leaves `present`, `how` and
    `by` as they are.
  - *What the fields do not buy.* A frame the subject is marked ON says
    they are in the picture, not which person they are: the tracker still
    takes the largest person on it. On the first real preview the
    unseeded subject, picked that way, took the person another subject
    was seeded on; the tool now runs seeded subjects first, which mends
    that case and no other. The lead's count on the previews of
    1920-6394, made with typed picks before the tool: about half the
    picks are the wrong person, nearly all the largest person where the
    subject is smaller. A pick is right when a person has outlined the
    subject, which is a seed: the open question is how the owner gives
    one box a shot at the least cost, and it is not answered here.
  - *The tile says two things for a seeded subject.* Its heading reads
    "the largest person, picked on frame N" (`shot_table.py`, where the
    heading is written from the job's `pick`) while its row reads "taken
    (seeded)" (`subject_track.py`). The job tool cannot leave `pick` out:
    the builder asks one of every subject. Saying one thing there is a
    change to node code and waits for a landing.
- **A load rendered zoomed in: the capture reads it; the job tool
  cannot ask for one yet** (2026-10-13; the reader built that
  evening, the rest named). Why it matters: by the saved part masks and
  the node's own box rule (`video_mask.zoom_box`), the lead's face on
  438-483 is the one small face of the three shots thought to need it
  (the owner kept the zoomed render of it made on 2026-10-10 by a
  session's own builder, which has no build record and is not laid);
  1010-1038 is a large face and 604-1009 cannot zoom as one load, her
  part's union being the whole frame. A second customer since: the
  stretch of the kitchen shot where the second subject's face part is
  under the capture's floor.
  - *What was built.* `bench/capture_masked_run.py::zoomed_on_the_canvas`
    lays a zoomed window's fitted mask and token region on the canvas
    with the node's own `video_mask.lay_back` and cuts at a half;
    `read_planned_regions` and `read_saved_regions`, which refused a
    zoomed plan and a zoomed render's windows, use it. A cell is in the
    region when any pixel of it is, and the run's record counts its
    cells by three rules (`zoomed_cells_counted`), since a box is on no
    grid of the canvas's. The run carries its `zoom`, and its margin
    twice: `margin_px` in canvas pixels and the node's own figure, of
    the zoomed picture, beside it. A window of another box than its
    run's, a zoom for another canvas, and the review picture of a
    zoomed run (which is of the box) are refused.
  - *Seen on the one real zoomed plan* (the second subject's 1010-1038
    of 2026-10-12, read into a scratch folder): `files`, `verify` and
    the gate run on it; the node's mask laid back is the capture's own
    carried mask on every frame, no pixel in one and not the other; the
    region is smaller than the one the 12th's capture worked out from
    the masks, which grew by the node's figure as canvas pixels.
  - *What the job tool lacks.* `bench/shot_to_load.py` never writes
    `render_region`. Ruled by the lead the same evening: a flag with
    words on `job`, the reason kept in the job file and its note, with
    `least=` said in the same place when a zoomed load needs its floor
    set, and no default for either.
  - *What stays wrong unless it is touched too.* The part floor
    (`least=`), the face-size flag and the rule inputs measure the part
    at whole-frame size, smaller than it is drawn by the zoom squared,
    and the capture's zoomed line says so; the strip and
    `bench/stack_delivery.py` show the node's mask picture, which is of
    the box; the head reference's box and the render's box are worked
    out apart.
  - *Not verified.* That the capture's fit of the source and the node's
    crop agree when the frames handed are not the canvas's size (the
    assembler refuses that case; the reader refuses a zoom whose canvas
    is not the capture's). Where the node's cut and `lay_back` put a
    pixel's centre was the other doubt: on the real plan, a box of even
    pixels at a zoom of 2, they agree to the pixel, and a made case on
    a box on no grid agrees too; a render has not been laid against it.
    The node's own remark on that plan stands: handed canvas-size
    frames, a zoom enlarges those pixels and is "not a sharper
    picture"; whether a job can hand the node the clip's own frames
    has not been read.
- **A picks batch holds one line for a shot of the record, however many
  of the tracker's cuts the owner answered on** (2026-10-13, the cast run
  for the page of 5935-6394 shown cut by cut). The owner answered on
  seven pictures of that one record shot; all seven are in the cast file,
  and the batch the run wrote holds the one clicked last
  (`bench/picks_against_answers.py::lines_of` walks a subject of a
  preview and a shot of the RECORD, and reads one answered picture
  there). So six answers are kept and not held against the tracker's
  picks. The fix: a line a tracker cut that has an answer of its own
  (the answer's table shot is on the answers line), with the record's
  shot beside it; the counts then say how many cuts a line stands for.
  It goes away by itself for a shot whose record is split to the cuts.
  Not started.
- **Two shots in one window: the tool is built, and no load has rendered
  that way** (2026-10-13, from the owner's questions on windows; the
  owner's word to build it came later the same afternoon,
  [`decisions.md`](decisions.md)). Every load made so far is one shot with
  a text that has one `[Shot 1]`
  (`grep -l '\[Shot 2\]'` over the job folders' `*.render.txt`). What a
  window holding two neighbouring shots could buy, in the owner's words:
  one face for the subject across the cut, and on a tough shot the help of
  an easier one beside it. What it costs and risks is in
  [`masked_v2v.md`](masked_v2v.md), "Two neighbouring shots in one load: a
  tool, not a default" and "A latent step is several frames, and its grid
  is fixed for a whole load": a cut inside a step, a reference video whose
  framing jumps at the cut, a text that describes both shots, and sampling
  that grows faster than the frames. Built: `bench/two_shot_load.py` with
  `bench/check_two_shot_load.py`, and the opt-in in the builder. Wanted
  next, in this order: one pair made both ways with nothing else changed,
  for the owner's eye, on two shots where the subject's face is on both
  sides of the cut and the load's first frame puts the cut on a step's
  edge; then a tough shot beside an easy one. Not built, each refused by
  name by the tool: two loads that do not meet at the cut (a part emptied
  between them), a subject one preview tracks and the other does not, two
  subjects, a zoomed or continued load, keep-outs and a hair band, and a
  tracker that crosses the cut in one preview. Not known: what the model
  draws in a latent step that holds frames of two shots; whether the
  node's own cut detector finds every cut a job names
  (`bench/two_shot_load.py plan-cuts` prints it for each load).
- **A shot longer than the planner's longest length and no longer than the
  model's is cut into two windows** (2026-10-13, read by the tooling
  session while fitting a single window to its load). The builder gives
  such a load the last of `loop_plan.CHAIN_LENGTHS` and the song node's
  schema tops out at the same value (`audio_freeze_song.py`, the
  `window_frames` input), so the node makes two windows where one of
  `h3_rules.MAX_LENGTH` would hold the shot. The owner's word of the same
  day is that a shot is split only when it has to be
  ([`masked_v2v.md`](masked_v2v.md), "Rules of the lane";
  [`decisions.md`](decisions.md)). Wanted: a load that fits within
  `h3_rules.MAX_LENGTH` takes one window of the grid's length that holds
  it. No shot of `fun.mp4` is in that case (`bench/shot_parts.py` prints
  each row's window), so nothing waits on it. Not started.
- **A plan strip does not say that its load renders zoomed in**
  (2026-10-13, from the blind reader of the first zoomed load's plan,
  `lead_fp_1920b`). The strip shows the region on the whole frame, as the
  capture reads it back onto the canvas, and nothing on it says the node
  will cut a box round the region and enlarge it, nor which box. A plan
  reader therefore cannot see the zoom or judge the box. The capture
  knows both (`bench/capture_masked_run.py::zoomed_on_the_canvas`, the
  run's `_zoomed.json` once it exists). Wanted: a line on the strip
  saying the load is zoomed, by how much and in which box, and the box
  drawn. DONE 2026-10-14 except the box drawn: once its capture is on the
  node's plan, a load's strip says zoomed in by how much and on which box,
  or the whole frame with the node's words why, each window's frames and
  held frames, and the job's defaults
  (`bench/capture_masked_run.py::render_said`, commit b543d899). The box
  is not drawn on the picture yet.
- **A flag's frame count and the frames burned for it can differ by the
  frames between its runs** (2026-10-13, from the render reader of
  `lead_fp_1920c`: a flag that says fourteen frames is burned on a span of
  fifteen). The cause is read, not guessed: a flag's `source_frames` are
  made by `capture_masked_run.frame_spans` with its default `join`, which
  makes one run of two runs a frame apart, so a span can hold a frame the
  flag does not count; the flag's own figures hold the exact count. The
  flag's key is made from those spans (`flag_key`), so exact spans would
  change the key of every standing override: that is why it is a row and
  not a fix made beside running loads. The small way: the strip's label
  says "N of these frames" where a span holds more than the flag counts.
  Not started.
- **The gaps between pieces are printed by the table and not yet by the two
  lists or the job step** (2026-10-13, late; the owner's ask after
  watching the joined clip: the frames no generated clip covers, sorted,
  loud, by itself wherever a session finds out what to build next).
  `bench/piece_gaps.py` lists them and `bench/delivery_table.py` prints
  them at the end of every table. NOT YET: the same runs for a shot's
  subjects in `bench/shot_parts.py` (the two lists a session reads for
  what each shot needs), and in `bench/shot_to_load.py job` for the shot
  being loaded ("after this load these frames of the shot are still the
  original of this subject"), which is where a load's frames are
  chosen. Also wanted: the tool's own `--ledger` read is of each load's
  job file for the reason it left frames; the capture says only that it
  emptied the part, not why (the grade, the floor, a keep-out), and the
  manifest holds which. Not started.
- **A preview's track for one subject can lie on another after the first
  has left the picture** (2026-10-13, late, from the holder of the card).
  In the preview of 604-1009 the lead is out of the picture from 999,
  and her track comes back on 1003-1006 and 1008-1009 lying almost
  wholly inside the second subject's track (counted from the two track
  files by the card holder). A load of the second subject there is then
  blocked for a region over another subject, and with the lead taken out
  of the load's others by hand the capture's verify fails on a track the
  shot table says is taken, since both read the preview's tracks
  whatever the load's others are. Very likely why her kept piece of
  that shot ends where it does. Wanted: the preview's own tables say where
  one subject's track lies inside another's (a figure a frame), so it
  is seen at the preview and not at a load's gate; and a way, with a
  person's word, to end a subject's track at a frame for the loads of
  that preview. Not started.
- **A zoomed load with keep-outs is blocked at its first capture**
  (2026-10-13, late, same source). A zoomed load of the second subject
  with her three keep-outs was blocked at the gate of its FIRST capture
  for a region over another subject, the others being her own
  keep-outs: on a small head with her hands at her face they take most
  of the region (the gate's print in that load's job folder has the
  shares; an earlier load of hers read far lower by the same rule). The
  lead's reading: the same wall as her load on 1093 that afternoon. Not
  known: whether the first capture's region for a zoomed
  load is the whole-frame estimate, the on-plan step being where a
  zoomed load's region is laid back; if so the gate blocks such a load
  before the step that would measure it rightly. To read first:
  `bench/capture_masked_run.py::zoomed_on_the_canvas` and which region
  the first gate reads. Not started.
- **A part that changes inside one load** (2026-10-13, late; THE NEXT
  WORKING DAY'S FIRST TOOLING JOB, the lead's ruling). `shot_to_load.py
  job --parts` makes a load whose part is some of its subject's parts
  on EVERY frame. The first one (the second subject's hair alone on
  1025-1038) was stopped at its reference: seven of its frames are all
  that the reference follows her on, under the builder's shortest
  load, and a face-and-hair load ending on 1024 beside it would put the
  change inside the shot, which is what the owner saw and disliked.
  Wanted: `--parts-on FIRST-LAST=NAME`, so ONE load 1010-1031 carries
  face and hair to 1024 and the hair alone on 1025-1031, cut from the
  class map a frame at a time, with one reference made on 1010-1031;
  1032-1038 left the source with the reason. Until then she is the
  original on the whole of 1010-1038. No hair-alone render exists yet,
  so the table's taking of a named part as the plan's region is unread
  on a real piece. Not started.
- **The reference step passes a body model that is not the subject**
  (2026-10-13, late, from the holder of the card on that load). On
  1032-1038 the lead crosses in front of the second subject, the body
  model goes elsewhere (upright, an arm raised, the face marks flung off
  the head), and `shot_to_load.py reference` printed PASSED: its
  read-back measures slip against the pose, not whose body it is. Its
  own print carried the sign (a face a few pixels tall on a head with no
  face showing). Wanted: the step holds the body model's head to the
  subject's own track a frame at a time, prints the frames on which it
  leaves it, and does not say passed over them; and the gate-alone
  path's list (`load_chain.FORCING_NOT_READ`) stops having to say that a
  head leaving its box is read only as "passed". Not started.
- **The delivery table orders overlapping keeps by the render's time, not
  the keep's** (2026-10-13). Where the owner kept more than one render
  on the same frames, `bench/delivery_table.py` takes the newer RENDER
  first. The owner's rule of that day ([`decisions.md`](decisions.md),
  its last entry of 2026-10-13) starts from which KEEP is the more
  recent and their words on each. The two agreed on the one case of the
  day and would not where an older render is kept after a newer one.
  Wanted: keeps on the same frames ordered by the time of the keep,
  both keeps and the owner's words on each printed, and the rule's
  other grounds left to the lead's `--prefer` with its reason. Not
  started.
- **After a cut is mended, the step that carries page answers into the
  record refuses** (2026-10-13). `bench/page_answers_to_cast.py --shots`
  holds every plain-frame answer to the shot the page showed it under,
  and an answer given under a shot the record has since joined, cut or
  moved is refused by name (the first met: frame 1926, shown under
  1920-1944, the record now 1898-1944). Right for an answer whose shot
  is gone; but it stops the whole run, so no answer of that day's page
  reached the record through it and the cast file was written without
  `--shots`. The one shot that had to change was written by
  `bench/shots_record.py answer`, from the owner's lines by number.
  Wanted: an answer under a shot the record no longer has is counted
  and listed, each with the shot that now holds its frame, and the run
  goes on; whether such an answer is carried to the new shot is a
  person's ruling, shot by shot. Not started.
- **`bench/break_checks.py` cannot run `check_masked_job`** (2026-10-13,
  from the build of the two-shot load). Its copy of the repo leaves
  `.git` out and that check asks git for HEAD, so the control is red in
  the copy whatever the break; the builder worked round it with a
  commit made inside the copy. Wanted: the copy carries enough of a
  repository for a check that asks for HEAD, or the check takes the
  commit from its caller. Not started.
- **The voice table's level is the two channels' mean** (2026-10-13,
  from the blind reader of `bench/clip_signals.py`'s first run).
  `bench/voice_spans.py` reads each stem as the mean of its two
  channels, and its line and its voiced runs are on that scale. Where
  the channels differ the mean reads low: on `fun.mp4` the power of
  both channels is above it by the figures in the reader's file
  (`reader_first_run/reading.md` in the clip's audio folder, section
  2). The signals tool now reads the power of both channels and says
  in its record that the voice table does not. NOT CHANGED, on the
  lead's word: every load's text leans on the table's `voiced` column,
  so the table is made again only with a read of which frames change
  sides and which texts that touches. Wanted: that read, then the
  table by the power of both channels with its line set again on the
  new scale. Not started.
- **Of several boxes in one cut, the seed is the one clicked last**
  (2026-10-13, from the preview of 6105-6394, where it was a frame inside
  a fade to black). `bench/cast_file.py::seeds` gives a cut one seed and,
  of several boxes in it, takes the one latest in the cast file: the only
  order nobody can type wrong, and no judgement of the picture. The
  preview's note names the box used and the boxes not used. What the tool
  could read to choose by the picture, none of it built and none of it a
  rule yet: the box's area on the canvas (a larger figure gives the
  tracker's look more to hold); the frame's own brightness and its
  spread, from the source frame (a fade, a flash); how far the frame is
  from the cut's ends (the tracker's cuts end where a head drops or an
  arm crosses); and whether the node's detection on that frame overlaps
  the box well, which the saved table of an earlier preview holds for a
  frame it showed. Each is a figure to print beside the boxes first; a
  preference among them is a person's. Not started.
- **A record's cut mended costs a preview of each shot it changes**
  (2026-10-13, from the holder of the card). `shot_to_load.py job`
  refuses a preview that is not of the shot's own frames, so the preview
  of 2050-2711 could not serve 2050-2622 and that long shot was tracked
  again. The refusal is right as built: a load's files are cut from the
  preview by frame, and a tracker run on other frames can pick and cut
  otherwise. Wanted, if mending goes on: a job step that takes a preview
  whose frames hold the shot's and whose tracker's cut falls on the
  record's new cut, saying so in the job. Not started.
- **A who picture does not say how much of its box a person fills**
  (2026-10-13, from the cut-by-cut page of 5935-6394, frame 6360). A thin
  detection's box lay inside another person's, and on the numbered
  picture the owner could not tell alone whose a number was; the lead
  read it with them. The page already puts each number on ground its own
  box owns (`bench/who_page.py`, the smallest box owns a point), which
  does not help when the picture inside the box is mostly somebody else.
  Wanted, one of two, a person's choice: the picture says for each
  number how much of its box its own mask fills, or a box whose mask
  fills little of it is left off the picture and listed under it. The
  masks are in the preview's saved tables' files; nothing new is tracked.
  Not started.
- **A zoomed load's part floor is counted at the canvas, not at the size
  the load renders at** (2026-10-13, from the holder of the card on
  `lead_fp_1802`, a load the node enlarges). The capture empties a part
  left under its floor (`bench/capture_masked_run.py::PART_LEAST_PX`, a
  starting value, or the load's own `least`) and counts that size on the
  canvas; a load rendered zoomed in gives the model the same part at the
  zoom's scale squared, so a part under the floor on the canvas can be
  well over it where the model works. On that load one frame was lost to
  it, by the margin its capture prints, and the load was made again with
  `--least` set by hand. Wanted: the floor is held to the part's size at
  the size the load renders at (the capture has the zoom,
  `zoomed_on_the_canvas`), with the canvas figure still printed beside
  it. A change to the capture, so built in a worktree and landed when no
  load is between its plan and its render. BUILT 2026-10-14
  (`bench/capture_masked_run.py::starting_floor`): with no `least=`, the
  starting value is held at the zoom the node's `zoom_box` gives the
  part's boxes over the span; `zoom=off` (the job's `whole_frame`, told by
  `bench/load_chain.py`) keeps it; at on-plan a run at another zoom prints
  both and the frames whose emptying would change. Taken up for the
  card's queue past 2049 (`internal/claude/2026-10-14_mrdrill_runs_past_2049.md`:
  each run's px and zoom), whose small zoomed runs the canvas floor
  emptied.
- **The shots record calls 5935-6394 one shot, and the tracker cut it
  into twenty-five** (2026-10-13, from the who page of the previews of
  1920-6394). The record's shots of that half are from scene scores of
  the copy (`cuts_from` in `data/fun/shots.json`); the preview of that
  shot (`shot_c_5935`, the lead's table) holds the tracker's own cuts.
  - *What the table says.* Fifteen of the twenty-five are one frame
    long; seven are as long as the shortest load the song node takes
    (`bench/masked_job.py::shortest_load`) or longer, and hold nearly
    all of the stretch. The tracker has the lead on some of the long
    ones and on others not.
  - *What it did to the who page.* The page shows one picture a shot,
    the one with the most people outlined, and here that was a one-frame
    cut on which the tracker holds nobody. The owner answered on it and
    said in words that they could not tell whose a number was. An answer
    on that frame, a box or not, speaks for one frame.
  - *What is not known.* Whether the tracker's cuts are real cuts (a
    strobe, a flash, a fast edit) or its own trouble with the picture.
    Nobody has looked at them as cuts. Whether the record should be
    split is a person's ruling after that look; a split changes which
    frames a preview, a load and an answer are of.
  - *Until then.* No seed work on that stretch from the one answer. A
    page of the stretch with one picture a long cut is the way the
    owner offered to help, and is listed with the lead.
  - *Looked at, the same day.* A blind reader looked at the frames either
    side of each of the tracker's cuts and of the picture's reading
    (`data/fun/cuts_look_2026-10-13/reading.md`, with its second list):
    one is a cut, at 6105, and none of the others is (a pan, a head
    going down and coming up, an arm across the lens, a fade to black at
    the end). The record was split there on the lead's ruling with
    `bench/shots_record.py`, each half one window, and the owner's
    answers of the cut-by-cut page divided by the frame each is on. What
    the reader did not look at: the frames between its sheets.
- **The owner's typed words on the shots of 1920-6394 are in no
  shot's note** (2026-10-13; a row, waiting on a read). The owner asked
  that day that their free-text words are read with their clicks.
  `bench/page_answers_to_cast.py --words-to-notes` carries each words
  line of a who row into the note of the shot that holds its frame,
  whole, once; it was run on 604-1009 and on 1039-1465 after the lead
  had read those lines. A dry run over the whole record then lists
  what is left: the lines typed on the plain-frame page of 1920-6394
  on 2026-10-12, each naming its shot's first frame. They were read
  on the 12th into `data/fun/who_1920_words_read.md`, by a cold reader,
  and not against the record's entries since.
  - *What it takes.* The lead reads the dry run's list (the command
    with `--dry-run` and no `--span`), then the same command without
    `--dry-run` on `--span 1920-6394`, after a copy of the shots file.
    Carrying the words changes no `present`, `how` or `by`.
  - *What it does not do.* Read the words against the buttons. Where
    the two disagree on a shot, that is still a person's to settle;
    the note only puts the words where the shot is read.
- **A capture's times are the machine's local time with no zone, and
  that clock is load-bearing** (2026-10-13; a row, the stored form not
  changed). Seen as `looks.json`'s `at` reading five hours behind the
  day's UTC stamps.
  - *How wide.* It is the capture tool's one clock: every time
    `bench/capture_masked_run.py` writes is
    `datetime.now().isoformat()` (status, flags, outcomes, looks, the
    region and plan records; `grep -n "datetime.datetime.now()"` in that
    file lists them).
  - *Who depends on it.* `bench/masked_job.py::_when` REFUSES a time
    that carries a zone, so that a look and its capture are compared on
    one clock; a look written in UTC alone would be refused by the
    builder. `bench/load_chain.py` reads a zoneless time as this
    machine's local time (`in_utc`, `later`) and writes UTC in the go
    file.
  - *What was done.* Where one of these times is printed to a person it
    is printed in UTC with a Z: the builder's look line and its two
    refusals (`utc_said`), the strip's time, the chain's refusal of an
    earlier look. What is stored is as it was.
  - *What is wanted, on a quiet card and in one change.* The capture
    tool writing an offset with its times, `_when` taking both forms
    and comparing as times, and a word on captures already on disk,
    which hold the old form. Not piecemeal: a half-changed clock makes
    a look read hours newer or older than it is.
- **The text grader has no donor for a masked load, so its command fails
  every masked text** (2026-10-13, from the chain's first real load; a
  row, nothing built). `docs/prompting.md` gives one command to hold a
  text's form, `bench/grade_prompt_text.py --mode ref2va --length N
  TEXT`, and a masked load's text is a `ref2va` text that names
  `<Picture 1>` and `<Video 1>`.
  - *What it does today.* The tool picks its donor graph by mode from
    the shipped graphs, and the `ref2va` donor wires pictures and no
    video, so the text FAILs on "prompt names `<Video 1>`, which no
    socket wires", whatever it says. The tool's own hint calls that a
    property of the donor. `--like` with a masked graph's stem is
    refused: no shipped masked graph carries a literal prompt.
  - *The length.* Asked for 61 frames it graded at 73, and asked for 50
    at 56: it snaps to the lengths a generation takes, and a masked load
    is the length of its shot. The word count it warns on is the
    guide's for a generation, not for a text beside a source video.
  - *What is wanted.* A donor that wires one picture and one video, or a
    mode that reads the sockets off the load's own built graph (the job
    folder's `<load>.plan.json`), graded at the load's own frame count.
    Until then a masked text's form is held by a reader, and the
    command's FAIL on such a text says nothing about the text.
- **A delivery record's `claimed_by_more_than_one_track` is of two
  people** (2026-10-13, traced and closed the same day). It counted any
  two tracks, and a keep-out is a track of things on a person, lying on
  that person's own track by design; the 13th's captures carry four
  keep-out labels on the second subject where the 12th's had one, which
  is all of the rise between the two days' records. It is now counted
  between people (`Captures.people`: a subject with a class map), with
  a person and a keep-out counted beside it by name
  (`a_person_and_a_keep_out_by_design`). The recount of both days is in
  `Subjects`' own words in `bench/assemble_delivery.py`. A record
  written before the change holds the old figure under the same key,
  and `--compare` says so of it.
- **The four reds of the green-and-red census are cut** (the owner, row
  z08, 2026-10-12 15:52:12Z): `check_correctness`'s bar, `check_sol_chunked`'s
  cosine and peak, `check_channel_balance`'s fragile-selection red,
  `draw_lid_marks`' fit verdict; each its own commit with its docstring and
  `docs/checks.md` row; `draw_lid_marks` on a day it is not on the card.
  Queued by the lead; not started.
- **A two-subject load, both stills referenced per the vendor guide**
  (the owner, 2026-10-12, in the lead's terminal: several people at once
  have been changed before, whole frame; it needs the references specified
  per `vendor_guides/ref_en.md`; Open List, today's row 9). The guide: one
  subject per picture, pictures and videos numbered independently. The
  masked lane builds one subject a load today (`bench/masked_job.py`,
  `one["subject"]`, one still, one region, one motion video); the reference
  chain already takes several pictures in order. The shape: two subjects in
  one load, the region the union of the two, one motion video with both
  body models on the black ground, the text with `<Subject 1>` and
  `<Subject 2>`; first test on the kitchen shot 604-1009 beside the two
  single passes that exist for it. Queued for the owner's word on the row;
  not started.
- **The builder's window choice reads the cuts** (the owner's rule of
  2026-10-12, [`decisions.md`](decisions.md)): shorter windows are fine,
  boundaries align with cuts where one is near, and no window begins within
  about a second of a cut. `bench/masked_job.py` leaves the window length
  to the node today and reads no cuts; the change gives it
  `data/<video>/shots.json` and a guard of about 24 frames, with a case in
  its check. Queued by the lead (mrfern); not started.
- **A layered view of the preprocessing** (the same entry): shared
  library code first (the capture tool's drawing helpers into
  `bench/_lib`, their own change), then the view where it fits an
  existing smaller tool, nothing duplicated. Wanted, by the owner's word
  on the one clip that exists (the Open List's row t76); which of its
  panels earn their place is asked when it is built. Not started.
- **For new checks and tools, from the owner at the close** (the same
  entry): green or red only for a hard failure; anything of degree is
  data with its context, no verdict. Whether that reaches the existing
  checks and the capture gate's levels is open with the owner; nothing
  existing is changed on it yet.

**Whole-frame video to video (2026-10-09): two full clips on one source; what is owed.**
- What was rendered, what each showed and what is unexplained:
  `../../bench/results/2026-10-09_whole_frame_text_and_copy_size.md`. Not
  judged on playback beyond the owner's remarks on single renders.
- **Owed, the owner's "another day": the swimmers clip by text with a men's
  suit.** The rendered text drew the rows in a women's one-piece. The
  replacement texts (swim shorts by their look, plain caps, how much of each
  body shows pinned in the second and third windows) and their graphs are
  written and not rendered: `internal/claude/2026-10-09_mrship/`,
  `prompt_swim3_*.txt`, `swim3_copy384_*_api.json`, built by
  `make_swim3_prompts.py` and `make_swim3_graphs.py` there. Each window is
  its own run, continued from the one before; the session's notes beside
  them say how they were queued.
- Open with the owner: whether the rule the text pairs point at goes into
  `../prompting.md`; a repair for the one held frame.

**The removals listed in `decisions.md`, 2026-10-10, "What the owner cut
from the masked lane's nodes", landed that evening** (`e9de8cc3` to
`368fbaa8`, the rebuild `ac35766a`); the one addition, the song node's input
that reads the analysis folder, is held for the next day
(`data/2026-10-10_cuts_piece5_analysis_input/`). *2026-10-10, evening: this
paragraph said the change was owed and gave its order as one sweep, then a
rebuild; the order that works after a node loses an input is restart,
rebuild, sweep, because the generator validates against the running
server.*

**Masked video to video (owner, 2026-10-04): version one shipped; movement from the source is answered on ref2va (2026-10-05) and open on fl2va.**
- **The 2026-10-08 crowd render, looked back on (2026-10-10):**
  `../research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`.
  Written without the session transcripts. Its nine "next time" items are
  checkable, and a session that holds a transcript annotates it.
- A shot that needs the original's movement renders on
  `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json`; why, and
  at what cost: `../../bench/results/2026-10-05_masked_v2v_motion_arms.md`
  and `masked_v2v.md`, "What it is". The fl2va-side route (mrhf's per-token
  late start), the Sapiens2 part menu wired into the Masked Source, click
  corrections per shot and the per-shot table are the four items the owner
  set running on 2026-10-05; the masking board carries them.
- **2026-10-06, generalising the lane** (the owner's direction: beyond one
  clip and one shot). The prompt is written by a node
  (`../../masked_prompt.py`), a review graph shows the tracker's picks
  before a render, and a parts graph wires Sapiens2 in; none of the three
  has rendered. Owed, in order: a first render on the node's text; a
  clothing text for the parts graph; several stills of one subject.
  `decisions.md`, 2026-10-06.
- **2026-10-07, SAM 3.1 read against Meta's code, and the tracker's rebuild
  begun.** Where each thing stands, pointers only:
  - *What to type when SAM is asked for something*, and the four places
    ComfyUI treats a phrase differently from Meta:
    [`sam3_prompting.md`](sam3_prompting.md).
  - *The corrections node* (`../../sam31_corrections.py`): the image range
    and the text activation set right for whatever loader feeds it, accepted
    on a server's queue
    (`../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md`). No
    shipped graph wires it yet.
  - *The default pick is the most central person* (`decisions.md`,
    2026-10-07; `h3_config.SUBJECT_TRACK`).
  - *What today's Subject Track does on a crowd*, under a change too small
    to see and alone against a group:
    `../../bench/results/2026-10-07_subject_track_under_nudge.md`,
    `../../bench/results/2026-10-07_subject_alone_or_in_a_group.md`. Owed
    from them, in order: a test on every frame that a track's mask has not
    jumped to another person; a re-find judged by where the subject was as
    well as by likeness; a higher detection count with that rule. The design
    and its build order are on the masking board (card
    `q-sam3-route-after-the-read`) and not yet a tracked page.
  - *Keeping what the original holds*: the Masked Source's `keep` input and
    the part node's `held` output (`masked_v2v.md`, "The region"). Built and
    rendered once each; the owner's look is owed, and no graph wires them.
  - *The frozen video cache on a masked window*: the bullet below.
  - *The reference still to the text encoder alone*: tried and closed for
    the masked graphs
    (`../../bench/results/2026-10-07_masked_still_encoder_only.md`).
  - *The prompt node's upper-body text* is in the vendor guide's form and
    says nothing it cannot know (`decisions.md`, 2026-10-07;
    `masked_v2v.md`, the prompt). The head and whole-person roles have not
    been judged in that form.
  - *A region switched on partway through a window* lands late and not by a
    fixed amount, and on a cut when one is ahead; *a patch of the source
    kept inside the region* brings the original person back:
    `../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`.
  - **The evening of 2026-10-07, in the order to pick it up:**
    0. **FIRST: the region is too big for a small subject.** The mask is on
       the right person; the margin added round it is a fixed pixel count,
       so when the subject is small most of the region is other people. On a
       window that starts with the subject small and has the region open on
       every frame, the render put the new subject on the person standing in
       front and filled the subject's own place with crowd (the owner, on
       playback; the pick and the mask were checked and are the same as in
       the render before it: the record below, section 5, and
       `data/2026-10-07_capture_windows/README.md` for how often the margin
       swamps the subject across four clips). In order: the margin as a
       share of the subject, capped at today's value
       (`h3_config.MASKED_SOURCE`, `grow_pixels`); the person in front kept
       out of the region (`subject_tracks.in_the_way`, not wired, into the
       Masked Source's `keep`); then that window again beside the last
       render of it. One render, one seed: a reading until that pair exists.
    1. *Nothing an earlier run made is reused by a key*: the kept-mask
       store and stored-window reuse were REMOVED on 2026-10-10 (the
       owner's rule; `masked_v2v.md`, "Nothing is kept between runs by a
       key"). The key that changes when the code does is not being built.
    2. *A subject who is small*: the part mask collapses, the fixed margin
       swamps him and the motion reference barely shows him; with the mask
       continuous over a window that starts with him small, the render does
       not show the new subject (the record above, section 5; the captured
       masks are in the untracked `data/` folder, with their own READMEs).
       Owed: the part step and the margin scaled to the subject, the zoomed
       motion reference on the upper-body recipe, then that window again.
    3. *The source is loaded in one resize* (`decisions.md`); what that does
       to SAM's masks is not measured.
    4. *A corrected shot is searched again* after a loss; the search is by
       likeness, and the search by place in `subject_tracks.py` is not wired
       in. *The pick*: no rule is right everywhere (`masked_v2v.md`, the
       pick).
    5. Asked for and not done: the still shared with the text encoder at
       full size on the masked graphs; a trace written by the workflow
       (the masking board, `plan-trace-through-the-workflow`); the benchmark
       records' SA-V tables regenerated at the corrected frame rate.
- How it works, its limits and the wider set of directions:
  [`masked_v2v.md`](masked_v2v.md). The dated account:
  [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 4. What was
  decided and against what: `decisions.md`, 2026-10-04.
  The renders and the owner's verdicts:
  `../../bench/results/2026-10-04_masked_v2v_band.md` and
  `../../bench/results/2026-10-04_masked_v2v_first_run.md`. Each session's
  notes: [`../research/masking/`](../research/masking/).
- The Subject Track with nothing to set (automatic pick frame, match and
  cuts, a manual value beside each) is landed and in the shipped graph,
  validated by three sessions on two clips. A third clip then broke it (two
  other women taken as the lead), and the node now compares a person in two
  places, the top of the mask and the head SAM 3 finds, and asks a lone
  person for a head: `../../bench/results/2026-10-04_subject_track_three_clips.md`.
  What that leaves: the rule was chosen on the three clips it passes, so a
  fourth; people who look alike, where one shot of the car clip cannot be
  settled by these features (an identity model, or a way to correct one shot
  by hand).
- **The turn, with no prompt text** (owner: does not want to prompt
  movement). A late start carries the pose and the original's look together
  (band record, "The turn without a prompt"). The same late start from a
  source softened inside the region was run as eight arms and is not a fix:
  he turns, and his clothes change, and on one of two seeds his hair
  (`../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`, which also
  lists what was not tried). What remains is the owner's to choose: the
  generic clause narrowed to facing, or accepting different clothes.
- **The scene's clothes on the new subject** (owner: where they want the band
  clip to go). Untried: a crop of the original's torso as a second
  reference.
- Owed, lower: the original subject's shadow stays in the plate; the source
  is resampled twice on its way to the canvas (the loader, then the fit); the
  margin as a share of the subject's size and not a pixel count.
- **The frozen-row cache on a masked window: back in the tree, measured on
  one window (2026-10-07).** Its first masked run saved no time and the
  owner retired it (`../../bench/results/2026-10-06_frozen_cache_masked_window.md`).
  The cause was one call: the cached block removed the attention override
  that carries the kitchen backend, so its attention ran on torch's kernel
  (`../../bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`). With
  the override left in place the same pair saves what the design predicted
  (`../../bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`),
  the owner could not tell the two renders apart, and they restored the
  masked use themselves the same day. It lives in one probe graph
  (`h3_config.FROZEN_VIDEO_CACHE`, the generator's `masked_cache`); no
  shipped or daily graph carries it. A second window, the upper-body recipe
  with a third of the tokens regenerating, saves less, as the design says
  (`../../bench/results/2026-10-07_frozen_cache_masked_upper_window.md`).
  Owed, in order: the owner's eye on that pair; a clip of several windows
  as a like-for-like pair; the
  departure of a cached step from a stock one measured on the right kernel
  (`verify`); the card's memory during a cached step; then whether a masked
  graph carries it by default, which is the owner's call. The refine pass's
  cached steps run on the graph's backend through the same change and have
  not been timed again.
- Parked by the owner, 2026-10-04: two samplers. The node that restores the
  plate between them is written and held to core's sampler on a stub model
  (`../../plate_restore.py`, `../../bench/check_plate_restore.py`), is in no
  graph and has not run on H3.
- Not being built, by the owner's call: protecting the neighbours' pixels,
  `paint_out` as a default, a ControlNet, the baseline chain for this lane.

**Encoder int8 against bf16 (owner, 2026-09-27): closed.** `ENCODER_INT8`
stays the shipped default.
- At the DiT, int8 moves the prediction no more than any small change, and a
  dose-response confirms that is a floor of the int8 DiT itself: noise far
  below int8's error reads the same
  (`../../data/results/2026-09-27_encoder_quant_dit.json`,
  `../../data/results/2026-09-27_encoder_quant_floor.json`).
- Floor-free, at the DiT's bf16 conditioning path, int8's image-token error is
  no worse than random and its text-token error is somewhat worse
  (`../../data/results/2026-09-27_encoder_quant_refiner.json`). A better int8
  encoder would start on the text side.
- The distill version (`../open_experiments.md` #47) closes unbuilt. Any
  numerical A/B read at an int8 DiT's output needs a tiny-noise arm beside it.

**Open after the 2026-09-27 night**, from the handoffs of the four sessions
(lookingdude, fastdude, vaedude, encoderdude). The board
(the owner's H3 Distill Board artifact) carries the same items as directions.

- **Decided 2026-09-27, owner** (each with its record):
  - Sol's default quantizer is `rotated`: 0.166.0,
    `../../bench/results/2026-09-27_sol_redesign_test2.md`.
  - The save format stays H.264 8-bit crf 19 (#38,
    `../../bench/results/2026-09-27_encode_format_ab.md`). O1's splotchy
    blacks are still open: the measured dark-blocking excess (+0.05) is below
    the owner's eye (vd-m12) and is not the codec's 8 px blocking. The test is
    a lossless decode of a clip the owner names.
  - PDD8 then full FlashGen from sigma 0.8 is the owner's t2v pick, added as
    `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json` (vaedude).
    Whether it replaces the default t2v PDD graph is still the owner's call.
    On fresh runs, where both passes sampled, it samples in the same time as
    PDD8 alone at the same length (6 PDD8 plus 2 FlashGen evaluations, 8 like
    PDD8's own), and the FlashGen LoRA node adds 0.1-3 s
    (vaedude, fastdude; `../../bench/results/2026-09-27_finisher_grid.md`).
    On i2v that review kept PDD8 alone; since 2026-10-03 the daily i2v graph
    is the finish (`workflows/h3_config.py::DAILY_GRAPHS`; `decisions.md`,
    2026-10-05).
  - PDD6 is a low-motion and close-up option, not the default
    (`../h3_distills.md`, "Rules of thumb, provisional").
  - The base finish (#37) and late-only FlashGen as a finisher (#34) close
    with no change. The base finish is not inert: it partly fixed
    spec_unusual's sign text and matched on noodle_bar, so PDD8's tail is part
    of its weakness (vd-v01).
  - Late-only FlashGen alone (#46) trades coherence for naturalness, so no
    stripped file is built.
  - The research model files are deleted, and so is the unpruned FastH3 V2
    download (`../../bench/results/2026-09-27_research_files_deleted.md`).
    Kept for #35's routes 2 and 3: the bf16 pruned FastH3 and fl2va, and
    `internal/pdd_shims/shim_{fasth3,fl2va}_temb`. `h3_research/` now holds
    only #35's two new checkpoints.
- **The Sol node redesign**
  (`../research/2026-09-27_sol_node_redesign.md`, lookingdude):
  - Tests 0 to 2 are done: output-neutral, `qk_balance` not inert, and
    `rotated` best while kitchen int8 wins the dense tail.
  - The dense tail still beats Sol on blocks 45, 48 and 49 on PDD8, so
    `dense_blocks` stays (the same record). *(2026-10-02, re-evaluated with full
    50-block telemetry: while dense beats sparse locally on 45, 48, 49, rotation
    dropped block 48 error to 7.14% and 45 to 9.55%. In contrast, middle blocks
    38–43 suffer 19%–26% routing sparsity error [block 42 at 25.67%, block 39 at
    25.22%]. Protecting block 48 while leaving 39–42 sparse wastes dense compute.
    Option A "39,41,42,49" and Option B "39,40,41,42,49" under trial; see
    [`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)).*
    *(2026-10-02, later: Options A, B and C were adopted in turn on local
    probe error alone, with an unfixed seed and no render. On the cells every
    run measured, the configs sit within prompt-and-seed noise.
    `../../bench/results/2026-10-02_sol_campaign_reanalysis.md` corrects the
    campaign and lists what would settle `dense_blocks`: fixed seed and
    prompt with two seeds per arm, routing `off` as shipped, an output-level
    score, the `exact_kv_and_all_rows` arm, a per-head price for block 49,
    and the FlashGen finisher measured on its own.)*
    *(2026-10-02, evening: the output-level panel,
    `../../bench/results/2026-10-02_sol_dense_blocks_panel.md`, was stopped
    early by the owner and decided (0.185.0): `sink_conditioning` is
    `exact_kv_and_all_rows`, `dense_blocks` stays Option C. `dense_blocks`
    does not move the owner's verdict; Sol itself changes the action on an
    action-heavy scene. Not being pursued: the record's open questions
    (PDD-node `start_percent` against dense, Sol on the finisher only,
    distance from dense as a scene screen) stand if the lane reopens.)*
  - Next: test 3, the token-routing re-grade on the fixed kernel, including
    whether "all blocks" still needs the balance. It runs on the test 2
    captures (keep until 2026-10-31), with no render.
  - Our sage attention in place of kitchen int8 as Sol's dense fallback:
    reopened and settled 2026-10-01. Kitchen is the more accurate kernel on
    every dense cell graded, the two steps before Sol's window included, and
    sage wins about one percent of the sampler, so kitchen stays
    (`../../bench/results/2026-10-01_sage_floor_early_steps.md`).
- **Approved and queued** (owner, 2026-09-27 night, relayed by fastdude:
  "ok to 4 and 5"):
  - #40, a second seed on the finalists: the t2v finish
    (`h3_text_to_video_pdd8_flashgen_finish`) and PDD8 alone on i2v. It gates
    every one-seed recommendation (vaedude, fastdude). Both arms render fresh
    on the same code, so no quantizer pin.
  - #43's scene pass (fastdude): the x0 twins of PDD8 and base Euler 32 on
    slapstick and samurai (high delta) and courtroom and radio (low delta).
    The pilot found that on subway, base Euler 32 settles its high-change
    frames late and PDD8 does not
    (`../../bench/results/2026-09-27_settle_pilot.md`). If that holds on the
    four scenes, the next step is the causal test: a finer PDD tail on one
    action scene. It renders its own baselines, so no pin.
  - #44, timestamps inside a shot (fastdude): first the timed and untimed
    bank pair and their checklist (`../open_experiments.md` #44), then the
    renders. It compares within itself, so no pin.
- **FastH3 (2026-09-29):**
  - #35, FastH3's gates against backbone drift, is rendered and read: the
    look travels with the gates (`../../bench/results/2026-09-29_fasth3_gates.md`).
    Open in that record: whether the gate values or only the coarse branch's
    presence matter, and whether the gates work off fl2va.
  - #36, the gate dial (`gate_scale` on `MiniMaxH3OverlayLoader`), is rendered
    and read: it is a look-and-motion dial, not a detail-only control, and scale
    0 is bit-identical to the no-gates checkpoint
    (`../../bench/results/2026-09-29_fasth3_gate_dial.md`; reread from the saved
    latents in `../../bench/results/2026-09-29_gate_dial_reanalysis.md`). It can
    only act on a checkpoint with `to_gate_compress` layers and sparse attention
    on: in `models/diffusion_models` and `models/loras/h3` only the two FastVideo
    VSA checkpoints have them, and none of the adapters (`mb-03` on the board). What remains is the owner's
    eye on the blind batch `fasth3_gate_dial_look_anchor`.
  - FastH3 as exact pieces on fl2va, and the node that applies them
    (`MiniMaxH3OverlayLoader`): see the index row for the overlay code and
    `../../bench/results/2026-09-29_fasth3_overlay_exact.md`.
  - PDD8 finished by FastH3's own checkpoint, t2v and ref2va: graphs built
    (`step_switch_to="fasth3"`), rendered once each, scored by the owner and
    parked (`../../bench/results/2026-09-29_blind_sessions_read.md`;
    `decisions.md`, 2026-09-29).
  - A general FastH3 backbone adapter is not built: the backbone change is
    not low-rank (`../../bench/results/2026-09-29_fasth3_bf16_rank.md`), and
    with #35 it is not what carries the look.
- **ref2va (2026-09-29):** the board's `ref2va-investigation` is h3lora's.
  Two of its findings were re-checked by a second pass and mostly hold
  (`../../bench/results/2026-09-29_ref2va_partition_delta_verify.md`,
  `../../bench/results/2026-09-29_ref2va_block49_verify.md`; the corrections are
  in `decisions.md`, 2026-09-29). Still open:
  - Grade block 49 on video and audio query rows. `bench/grade_dense_kernels_on_captures.py`
    now prints and records that grade under each cell's all-row grade
    (`rows_video_audio` in its JSON; the selector is tested on the CPU by
    `bench/probe_grade_row_selection.py`). Running it on the real kernels over the
    ref2va and t2v test 2 captures is what remains, and it needs the card.
  - Regenerate `../../data/results/2026-09-29_partition_delta_map.jsonl`, which
    predates its script; do not read `cos` from it.
  - Comfy-Org/ComfyUI issue 16604 (a reported ref-row shape mismatch above 16384
    packed rows) does not reproduce on the layout on the CPU
    (`../../bench/results/2026-09-29_ref_rows_16604.md`). Not tested: CUDA or
    ROCm, and the reporter's stack. The one weak spot it found in this pack,
    the reference-video branch taking its latent grid from `canvas // 16`, was
    fixed in 0.176.2.
- **Comparing across 0.166.0:** anything compared against a render made
  before it pins `quantizer=balanced` or re-renders its baseline. FastH3's
  contract graphs use core's attention and are unaffected.
- **Owner's calls with no render:**
  - #45's reference node: kitchen's VSA selection matches FastVideo's on
    blocks 0 and 24. Block 49's gap is int8 on text rows
    (`../../bench/results/2026-09-27_vsa_selection_grade.md`). Build the node
    only to see whether that is visible; otherwise close #45.
  - `check_vsa_core_patch.py`'s logic fix, so that "nothing patches core"
    has a working check again.
  - The working rules the sessions proposed (the board's
    cross-session-patterns).
- **Hygiene, no card** (vaedude):
  - waiter-locks: the substring-matching queue waiter has bitten twice.
  - The dataset builder should raise on an output claimed by two renders.
  - always-red-checks, down to two: `check_doc_links` on
    `../h3_audio_freeze.md`'s sister-repo citations, and
    `check_audio_freeze`'s ImportError.

**Idea saved for later, 2026-09-26: fix PDD's weakness with the other
distills' strengths.** The owner: "i wonder if you could make pdd better by
fixing its bad with the complements from flashgen and/or fasth3 ... and maybe
the other way around but my gut says pdd".
- **PDD first, FlashGen finishing (the reverse of route 3 below).**
  - PDD's early blocks integrate the smallest share of the trajectory
    (`../h3_pdd.md`, the arc table), so they are its most accurate, and PDD is
    the distill that follows the teacher's layout.
  - FlashGen's mode-seeking final jump commits moving things instead of
    hedging, which is where the working model in
    [`../h3_distills.md`](../h3_distills.md) puts PDD's clones.
  - The same graph also answers "the other way around": FlashGen's weakness
    is adherence, and the layout comes from the early steps.
  - Two handoffs, both on PDD's own knots: 0.632 (PDD 7 of 8 steps, then
    FlashGen's one step) and 0.8 (PDD 6 steps, FlashGen two).
  - The x0 capture on `subway_chase__pdd8` (the follow-up batch) shows the
    step where the clone enters, and so which handoff to try.
  - Built only on 0.154.8 or later: two branch models in one graph.
- **PDD video, FastH3 audio.** Re-render the audio on FastH3 with the video
  frozen (the audio-refine pass). The owner rates FastH3's audio highest.
- *Annotation, 2026-09-27: the two figures below are the prompt's suspect and
  agent, not a clone* (`../../bench/results/2026-09-27_clone_base_control.md`).
  The base draws the same two. The open question is only PDD8's overlap at
  1.1 s, by the owner's eye. The reverse switch's value is its highlight
  repair (`../../bench/results/2026-09-27_reverse_switch.md`).
- **Update, the same night: the subway clone is born early.** The x0 capture
  shows two figures already at PDD8's step 1 of 8
  (`../../data/results/2026-09-26_x0_steps_subway_pdd8.json`).
  - A late handoff (PDD first, FlashGen finishing) should not remove it.
  - Route 3 (FlashGen first, PDD finishing) or a different first block
    might.
  - The base's early x0 at this seed is the control: does the base make the
    same choice?
  - The PDD-first graph is still the right shape if other failures turn out
    to be late.
- **Not weight blending.** On 2026-09-26 the stacking bug ran FlashGen with
  PDD's backbone half and no PDD heads, and the owner saw it as "blocky as
  hell" (`../../bench/results/2026-09-26_followup_contamination.md`).

**Idea saved for later, 2026-09-26: fit the schedule to how much of the frame
changes.** The owner, on PDD's coarse last steps: "our sigma schedule needs to
adjust based on a ton of pixels all needing to change at the same time
(action scene) vs. just regions (dialogue scene) except during hardcoded
prompted cuts". The owner also hypothesised that the same wholesale change is
why cuts, and changes inside a shot, have to be prompted: a cut is "like a
pixel wipe".
- **What supports it.** PDD quality follows the coarseness of the schedule's
  tail (`../evidence.md`, "Settled about H3"). The working model in
  [`../h3_distills.md`](../h3_distills.md) puts the clones in the coarse late
  steps: the heads predict a block's mean velocity from its start state, and
  hedge when that state does not yet decide where a moving thing goes.
- **What exists to build it from.**
  - Schedules are per render and global: one sigma for every token.
  - PDD can take any partition of its 32-point grid inside the trained
    envelope (`pdd_lora.envelope_partition`), so a scene can buy a finer tail
    where it moves.
  - The DiT already takes a timestep per token row, which is how frozen rows
    run at a different t. A per-region schedule is expressible in principle,
    but PDD fuses one head per step for every token.
- **Cheapest first version.** Pick the PDD partition per scene from its
  motion, read from the prompt's metadata or from an early x0's inter-frame
  delta, with cut frames masked out (`../../bench/measure_clip_delta.py`
  already finds cuts). The PDD 4/6/8/16 test in `h3_distills.md` says first
  whether the tail is the lever.
- **About prompting cuts: inference, not tested.** A cut has to be prompted on
  the undistilled base too, which runs a stochastic 16-step sampler. So that
  part is more likely what the model learned from captioned cuts than a
  sampler effect.

**Idea saved for later, 2026-09-26: pick the distill by what the shot does.**
From the owner's unblinded looks at one seed:
- **PDD8** "looks great when its closeups or medium shots - lots of detail and
  good color - sucks at motion", and "seems to do great at low
  movement/deltas" (`../../bench/results/2026-09-26_subway_v2_s1.md`,
  `../../bench/results/2026-09-26_distill_compare_s1.md`). An older finding
  agrees: PDD's artifact severity tracks inter-frame delta
  (`../../bench/measure_clip_delta.py`, docstring).
- **FlashGen and FastH3** hold large motion better. FlashGen keeps the base's
  colour but loses the prompt's roles on subway. FastH3 runs warmer and more
  saturated than the base (`../../bench/results/2026-09-26_distill_tone.md`).

**The owner's idea:** route adapters by shot, PDD on the still shots and
FlashGen or FastH3 on the moving ones. Or at least document which distill
suits which kind of shot. The owner's own caveat: switching adapters changes
attention along the way, unless the strength is low.

**What is in the way.** H3 generates every frame of a clip in one joint pass,
so a clip has no "shot 2" to give its own weights. Three routes could work,
none tried:
1. **Render shots separately, each on its distill, and join them.** This
   needs the continuation or keyframe path for the join, and a cut hides the
   seam.
2. **Row-masked adapters.** `lora_branch.py` already adds the LoRA per token
   row, so it could scale it by each row's frame index. That is untrained, and
   every row still attends to every other, so the owner's attention concern
   applies in full.
3. **Switch by step instead of by shot**, for example a motion distill's early
   steps for layout and PDD's late steps for detail. Every frame carries the
   same adapter within each pass, so nothing mixes across frames. The risk is
   the handoff instead: the second model receives the first one's
   intermediate state ([`../h3_ref2v_distillation.md`](../h3_ref2v_distillation.md)
   §3, written for base and distill).

   **How it would be built** (agreed with the VAE session, 2026-09-26):
   - **A new generator arm, not `split_at`.** `split_at` splits one
     `BasicScheduler` schedule between a base pass and a single LoRA pass,
     and bypasses the PDD node's own sigmas. This needs two model chains:
     FlashGen through `MiniMaxH3LoRABranch` and PDD through
     `MiniMaxH3PDDLoRA`, each with its own ManualSigmas, and `DisableNoise` on
     pass 2. Both chains keep the same `MiniMaxH3SigmaShift` at 12/3, or the
     audio stream's noise levels drift from the video's.
   - **The handoff sigma.** Pass 1 is FlashGen's own first two steps: 1.0,
     0.9655, 0.8889 (`h3_config.FLASHGEN_MANUAL_SIGMAS`). Pass 2 starts where
     pass 1 left the latent, so there is no noise mismatch: 0.8889, then 0.8,
     0.632, 0. The last two intervals are PDD8's own knots (`simple` at shift
     12, `bench/check_pdd_sigmas.py`). That is five evaluations. Bending
     FlashGen's second step to PDD8's 0.878 was the alternative, and it would
     move a 4-step student off its trained points.
   - **PDD's heads absorb the offset.**
     - Core's `FinalLayer` picks heads from each pass's own `sample_sigmas`,
       and `pdd_lora`'s tracker reads the same sub-schedule.
     - `pdd_math.schedule_knots([0.8889, 0.8, 0.6316, 0], 12, 32)` gives
       [19, 24, 28, 32], against [20, 24, 28, 32] from 0.878. So the first
       step fuses heads 19 to 23.
     - Head 19 spans 0.8914 to 0.878 and gets its whole interval's dt,
       although the step starts at 0.8889, so it is slightly overweighted.
   - **Time the handoff.** Pass 1's branch-patched model and pass 2's
     int8-merged PDD model are two patched states, so the card re-patches,
     reloads or re-stages between them. The five-evaluation cost leaves that
     out. Arm `H3_TELEMETRY` for the first render: it records per-node model
     residency and the load and staging lines.

**Routes 1 and 2 in depth, and three routes of H3's own** (by noise level per row, by stream, by component), ranked cheapest evidence first: [`../research/2026-09-26_distill_routing.md`](../research/2026-09-26_distill_routing.md). Route 2 has a constraint worse than the attention leak: every row shares one sigma schedule, so a masked-in distill always runs off its own.

The owner's other half is written: `../h3_distills.md` (2026-09-26, one seed; revised when the second seed lands).

**2026-10-05: PDMD is retired** (owner: "lets retire it. i didnt see anything
from it worth keeping"). Its graphs, constants and note are out of the tree;
`../roadmap.md`, "Closed lanes". The 2026-10-01 entry below is as written, and
the section and file it points at are gone.

**2026-10-01 (PDMD, FlashGen's closest sibling): parked.** Kijai's PDMD LoRAs
were researched, wired, and judged once, blind, against FlashGen
(`../h3_distills.md`, "PDMD"). FlashGen was preferred. Kijai's resize was
indistinguishable from full rank, so it is the default file. The owner parked the
lane ("nothing else to do with it"); `../roadmap.md`, "Closed lanes", says what
would reopen it. Nothing is owed.

**2026-09-26 (the prompt bank fix, 0.151.1 and 0.151.2).** Every flagged prompt is
fixed (`../../CHANGELOG.md`, 0.151.2). One gap is still open:
`bench/grade_prompt_text.py` passed the refview2 twins while their unmarked
vocal events lacked `(Sx)` reuse, which ref2va requires. A check for it would
have caught three prompts.

**2026-09-26 (FastH3 before judging it).** The owner: make FastH3 as good as it
can be before calling it worse than PDD or FlashGen. The first probe copied
ComfyUI's template, and FastVideo's own contract differs on the sampler, the
VSA kept fraction and a dense warm-up (0.148.0, `h3_config.FASTH3_CONTRACT_*`).
- **The owner's look** at `contract` against `template`
  (`../../bench/results/2026-09-26_fasth3_contract_s1.md`). The attention change
  decides the take and is also faster.
- *Rendered 2026-09-26:* FastH3 at the contract against PDD8 and FlashGen, one
  seed, two scenes (`../../bench/results/2026-09-26_distill_compare_s1.md`).
  Still owed: the owner's three-way look, then a blind session with the dense
  baseline if one of them is to be the default.

**2026-09-26 (FlashGen before audio refine).** The owner: make FlashGen work as
well as it can before refining its audio. Where every setting comes from and how
upstream runs it: `../research/2026-09-26_flashgen.md`.
- *Shipped 2026-09-26:* `h3_text_to_video_flashgen` (rank 64 at the call), on
  the owner's "best stab" before they could look (`decisions.md`).
- **The owner's look** at `r64` against `r64_branch`
  (`../../bench/results/2026-09-26_flashgen_lora_path_s1.md`). A merged LoRA on
  the int8 checkpoint loses most of FlashGen's delta
  (`../../bench/results/2026-09-26_int8_lora_requant.json`). Applied at the call,
  it renders a different take.
- **Then:** a second seed and scene; a 125-frame arm at FlashGen's trained
  length.
- **FlashGen off T2VA** (0.147.0): i2v and ref2va probes rendered and hold up by
  eye (`../../bench/results/2026-09-26_flashgen_tasks_s1.md`). Still owed: the
  owner's look, same-seed base or PDD renders to compare, and prompts with sound
  in them.

**2026-09-25, late night (0.143.0, the frozen-video cache).**
`MiniMaxH3FrozenVideoCache` ran live on the FlashGen refine arm
(`../../bench/results/2026-09-25_frozen_cache_s1.md`). It roughly halves the
refine pass, and the finished audio stays close to the uncached pass.
- *Done 2026-09-26:* the owner heard no difference, and the shipped refine
  graphs carry the cache (0.144.0, `decisions.md`).
- **Fewer refine steps** is the cheapest remaining cut: `h3_config.AUDIO_REFINE`
  is inherited and untuned, and whether fewer steps keep the audio lift is a
  listen.
- **Noise floor:** a same-graph repeat of
  `h3_probe_t2v_flashgen_4step_audio_refine_uncached` after a cache clear.
- **Shorter cached steps:** a VRAM store or K/V contents, both untried.

**2026-09-25, night (0.142.0).** Sol now defaults to `dense_blocks = 45,48,49`,
and `token_routing` is one dropdown with `off` as its default. *(2026-10-02: Under
`quantizer="rotated"`, block 48 error dropped to 7.14% while middle blocks 38–43
spike to ~25% error; `dense_blocks="45,48,49"` is under trial for replacement by
Option A `"39,41,42,49"` or Option B `"39,40,41,42,49"`. See
[`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)).*
- **Red since 0.139.0, found 2026-09-26:** `bench/check_audio_freeze.py`
  imports `audio_freeze.py` as a top-level module. That module's relative import
  of `audio_resample` fails, so the check never reaches its cases.
- **Owner decision:** `bench/check_schema_defaults.py` has been red since this
  change. `token_routing` is `'off'` in the schema and `None` in `execute`,
  which keeps a pre-dropdown API graph on its list. Align them, which silently
  drops such a graph's routing, or exempt the input with that reason.
- *Done 2026-09-25:* the server was restarted at 0.143.0, all graphs validate
  against it, and `bench/check_widget_deviations.py` passes. **Still owed:** the
  Sol kernel checks run with the GPU visible.
- **The tail default is unscored.** `h3_probe_t2v_no_dense_tail` against
  `h3_text_to_video` is the pair that scores it.

**2026-09-25, evening (the distill and audio-recovery throwaway run).** Wired
and rendered at 0.141.0 (`../../bench/results/2026-09-25_distill_audio_s1.md`):
PDD8, PDD8 plus the audio-only refine pass, FastH3 8-step V2, and FlashGen
4-step with and without the refine pass, on the diner and subway scenes, one
seed.

- **Owner: listen to the throwaway clips** (the record lists them). A wiring
  run, not a verdict. The refine pass raised loudness on every pair, and
  FastH3 was the loudest arm on both scenes. Loudness is not quality.
- **Then the quoted session** through `h3-ab-session`: the same arms plus the
  dense baseline, a second seed, and blinded pairs. The throwaway timings put
  it at a few hours of GPU with the baseline included.
- **Open, a cheap probe:** do LoRA'd layers stay on the int8 path at run time?
  That decides whether the PDD and FlashGen arms run the base graphs'
  numerics (`../research/2026-09-25_temporal_offset_and_adaln_rounding.md`,
  last section).
- **Open, a cheap control:** the refine arms' video is about 46 dB from their
  base arms, not bit-identical. A same-seed repeat of one graph after a cache
  clear says whether that is decode nondeterminism or the masked blend.
- **Step caching (owner question):** worth reopening only if the 16-step base
  graphs are still rendered. Then build a DPCache-style node
  (`../research/2026-09-25_step_caching_survey.md`). At distilled step counts,
  nothing can be skipped.
- **Temporal offset and AdaLN rounding: nothing to do.** The offset weakens
  the prompt and does nothing for seams. The rounding is real but below the
  int8 noise (`../research/2026-09-25_temporal_offset_and_adaln_rounding.md`).

**2026-09-25, later (the PDD, continuation and upstream digs).** Each result is
mapped to the owner decision it feeds. None of the decisions is made here.

- **Does our PDD node keep its head half?** The evidence favours keeping it
  (`../research/pdd/2026-09-25_upstream_pdd_comparison.md`, "Evidence for the
  open decision"):
  - core never engages for our files;
  - core has no fingerprint or shift guard;
  - on the song graphs' masked audio, core would pick blocks the row never
    visits.

  Precision is equal either way. Owner decision.
- **Can PDD do better than core's and ours?** Not from any upstream: we are at
  the precision floor and agree with core on every shipped schedule. Three arms
  are left, each needing a render go:
  - head selection for mask-pinned audio rows (the song graphs, where our
    off-schedule warning would fire once per render; inferred from the probe,
    not seen in a log);
  - audio refine on the base model after PDD;
  - PDD on the unpruned base.
- **Report upstream?** All outward-facing; the owner's call:
  - sglang's fc1 half-swap in its PDD builder, shown on one block's MLP at a
    time (`../../bench/results/2026-09-25_sglang_pdd_fc1_merge.md`);
  - sglang's bf16 plan and storage, which cost a large share of the distilled
    correction (the PDD record, section 1, has the figures);
  - UtilsCollection's four-argument `final_layer` patch against merged core's
    seven.
- **Build guide-row continuation (arm B)?** The evidence is mixed
  (`../research/2026-09-25_continuation_guide_rows.md`):
  - core expresses it without a patch, and a matched-pair plan is written;
  - under a hard audio freeze, A and B differ only in video;
  - LongMedia reports B's guide span repeating as a motion motif;
  - A already has a positive verdict.

  Worth one pair only if seams are a live complaint.
- *Done 2026-09-25:* `bench/check_sol_observe.py` had failed since 0.123.0
  because its stub model lacked the modules `install_h3_morton` hooks. The
  stub now carries them, and the check passes.
- *Done 2026-09-25:* the four stale Kijai PDD symlinks were removed from
  `models/loras/h3/` (owner's call). The files remain in the Storage share.
- **INT8 video VAE:** nothing new bears on it. The tile-seam change is inside
  the overlap bands and fidelity-neutral
  (`../../bench/results/2026-09-25_vae_tile_seam_blend.md`); it only means
  pre-2026-09-22 renders are not bit-comparable. *Superseded 2026-09-26: the
  owner reopened the lane, and `h3_config.MODELS["video_vae"]` is the INT8
  build since 0.151.0.*

**2026-10-03 (sparse attention: the output check and the per-head table).**
- The output check is closed: no measurement predicts the owner's verdict,
  no automatic judge, the shipped default stays
  (`../../bench/results/2026-10-03_sol_output_check.md`). A dense finisher
  is the remedy for one render whose action came out wrong.
- The per-head tau table waits on the owner: a different reference did not
  break it, it is worth little, and it needs a guard margin and a re-fit on
  the 512 layout before any use
  (`../../bench/results/2026-10-03_per_head_ref_kill.md`). The instrument is
  built (`../../bench/results/2026-10-03_tau_sweep.md`).
- Every attention timing and routed share of 2026-10-03 is on the shared
  2048-view layout; none has been re-taken on the 512 default.
- Open from the 2026-10-03 review of the sparse node: the items listed under
  "Not done" in CHANGELOG 0.185.20.

**2026-10-02 (the upstream survey session).** Found by the read in
[`../sol_upstream.md`](../sol_upstream.md), "comfy-kitchen and core,
2026-10-02", [`references.md`](references.md), "What moved by 2026-10-02",
and [`../research/sglang_comparison.md`](../research/sglang_comparison.md),
"Eighth read". Nothing triggered the adopt-upstream rule. Most urgent first;
none of it is done.

- **Core PR 16681 would break `MiniMaxH3LoRABranch` on fc2 if it merges.**
  `lora_branch.py::_mlp_forward` (and its copy in
  `standalone/h3_mutant_distill/exact_lora.py`) patches `mlp.forward` with a
  one-argument function, and the PR calls the MLP with `residual`, `gate` and
  `segments`. The fix is to accept the three keywords and apply core's
  `_mod_gate` when `residual` is given, with `bench/check_lora_branch.py`
  calling it both ways, and the full `bench/check_*.py` sweep before commit.
  Reasoned from the diff, not run. Do it when the PR merges, or before as a
  forward-compatible change if the PR's signature settles. *Done 2026-10-02
  in 0.184.10 (owner): both copies accept the keywords, and
  `bench/check_lora_branch.py` runs the PR's call. If the PR's signature
  changes before merge, that case is what to update.*
- **If core PR 16713 merges,** re-run `bench/probe_ref_rows_16604.py`, which
  calls `_cond_video_rows`.
- **Two optional decoders now exist, neither measured here.** Core loads
  LynnReal's light H3 VAE (a distilled shorter decoder, same latent space;
  `986c4d15`), and vllm-omni wires a community latent upscaler between denoise
  and decode (`a038b3817`). Either would go through the VAE lane's decode
  comparison (`bench/compare_vae_decoders.py`) before any default moved.
  The owner's call whether to look.
- **Not proposed:** sglang's new ComfyUI integrated mode replaces core's
  diffusion model, so none of our DiT patches would run under it; Spectrum
  stays behind the step-caching survey's verdict; vllm-omni's no-resize
  reference stills are one engine leaving parity for speed, and
  `bench/refview2_arms.json` is still the test that would settle
  `allow_upscale`.

**2026-09-25 (the upstream survey session).** Found by the read in
[`../sol_upstream.md`](../sol_upstream.md), "comfy-kitchen and core,
2026-09-25", and [`references.md`](references.md), "What moved by
2026-09-25". Most urgent first; none of it is done.

- *Done 2026-09-25 in 0.139.0:* the torchaudio resample sites call
  `comfy.audio.resample` through `audio_resample.py`, and the Sol node refuses
  an H3 model that will not compute in bf16, which is what core PR 16508 would
  cause on a launcher with `--fast fp16_accumulation` (this one dropped it
  2026-09-28). Still worth watching 16508; if it merges, the
  refusal is what users will see, and `--bf16-unet` is the fix it names.
- **The PDD head-half question is live.** Core's head bank merged 2026-08-29
  (`../h3_pdd.md`, "Core is learning this"): should `MiniMaxH3PDDLoRA` keep
  its own head swap, or narrow to conversion plus the partition guard? sglang
  now fuses PDD heads with the same formula as ours, which corroborates the
  math, not the choice. Owner decision.
- **Core's VAE tile blending changed in the 2026-09-22 pull** (#16436). Any
  bit-identity check against a clip rendered before it will fail for that
  reason alone. The next rebuild record re-renders its reference clip first.
- **Continuation has a third design worth one pair.** vllm-omni feeds the
  previous window's latent tail as guide rows and regenerates the overlap.
  That is `../h3_audio_freeze.md` section 5 idea 5, and core can express the
  guide half as `minimax_keyframes` latent rows. Whenever the continuation lane
  resumes: one matched pair against `MiniMaxH3FreezeAudioWindow`, through
  [`../eval_comparison.md`](../eval_comparison.md). *Now
  `../open_experiments.md` #32 (2026-09-26).*
- **On the next kitchen tag,** #192's fp16-accumulate depth gate reaches H3's
  video VAE encode on this launcher, so the rebuild record for that tag says
  so and checks an encode. *Now `../open_experiments.md` #33 (2026-09-26).*
- **Not proposed:** ComfyUI's templates moved to the INT8 video VAE the owner
  removed on 2026-08-21. That decision stands unless reopened. *Reopened and
  adopted 2026-09-26 (0.151.0).* sglang's
  probable fc1 swap in its PDD builder is theirs to hear about; reporting it
  upstream is outward-facing and the owner's call.

**2026-09-20 (the prompt-rules session):**

- **Documented pointers to scripts are unchecked, and six have rotted.**
  `docs/checks.md` and `docs/prompting.md` section 11 name `.py` files that no
  longer exist: `check_portable_standard.py` (renamed 2026-09-01),
  `check_workflow_schema.py`, `check_mono_ref_audio.py` and
  `check_uncontrolled_claims.py` (each deleted with a reason), and
  `show_red_preflight_guide_split.py` and `show_red_reference_runtime.py`
  (left with `bench/red/` in `0a764a33`). One of the six was corrected by hand
  in `9e7d7377`; the rest stand. `bench/check_skill_routes.py` already does
  exactly this for `.claude/skills/`, both directions, so the instrument
  exists and is pointed at one directory. Done when it also reads the two rule
  tables, or when the owner decides documented pointers are not worth gating.
  **Why it is more than tidiness:** a rule table is consulted only while it is
  trusted, and an index that names missing files is one people re-derive
  around instead of reading. The sister project landed its own citation
  resolution on 2026-09-20 and found the same class. **What separates this
  from a speculative check:** the list above is one hand pass's catch, and
  one of the six had already been found and repaired by hand in `9e7d7377`
  long after it rotted. The other five are still there as you read this.
  *Corrected 2026-09-20, same session:* the item above said
  `check_skill_routes.py` "is the right shape", and a scan of every tracked
  doc for backticked paths says that is too simple. A skill file's paths are
  repo-relative by convention, which is the whole reason that check is cheap.
  These documents legitimately cite four roots -- this repo, a `coderef/`
  checkout, ComfyUI core, and the model release -- so a bare
  `backends/eager/sol_attn.py` is correct prose and unresolvable without
  knowing which root it hangs from. A resolver that does not model the roots
  reports hundreds of them, which is the false-positive twin of the
  empty-search trap and worse than no check. Two further findings from the
  scan: the tracked docs cite paths that exist on disk but are untracked,
  concentrated in `coderef/` and `internal/`, which are followable by whoever
  wrote them and by nobody else -- the sister project separates that from rot
  with a per-document allowlist that GRANTS, and our equivalent is prose
  disclosure applied uncertainly. And the oracle matters: ask the filesystem
  and those read as fine, ask `git ls-files` and they do not.

**2026-09-19 (the research and instrument session):**

- **The blind panel is built and unscored.** Six real contests (noodle bar at
  107 frames, diner, market, hardware aisle, crowd, meerkat: the default
  against a fully dense render at one seed), a two-seed decoy, an identical
  pair and a low anchor, one session
  (`bench/blind_panel.py`, key sealed under `internal/blind_keys/`). **The
  profiler and the `dense_blocks` contest are held on its result**: if the
  owner cannot separate the default from fully dense, the levers inside Sol
  are invisible a fortiori and the remaining question is seconds. Score it,
  then `bench/join_panel_verdicts.py`, which applies the stop rules of
  [`../research/2026-09-19_evaluation_one_judge.md`](../research/2026-09-19_evaluation_one_judge.md)
  section 0.3 and refuses a conclusion until a reader marks which verdicts
  name a defect.
- **Most full-length pairs are two takes, not one take with a difference.**
  Picture and sound are judged apart, and they come apart in both directions
  (`bench/results/2026-09-19_pair_alignment.md`). Only a few scenes stay
  pixel-aligned; on the rest a reference metric against another render
  measures the performance, not the knob. **Consequence for the largest speed
  lever:** a `start_percent` ladder cannot be scored by distance to a dense
  render, because removing the dense warm-up moves the sample into another
  basin; it needs the blind panel
  ([`../research/2026-09-19_where_approximation_is_tolerated.md`](../research/2026-09-19_where_approximation_is_tolerated.md)
  section 1.1).
- **The judge's tie rate is on file, and the panel supplies the first
  decoys.** Ties are about as common as decisive verdicts across the
  structured verdict files, and the slot split is within chance
  (`bench/results/2026-09-19_judge_tie_rate.md`). The tally now counts decoy
  verdicts apart, where a picked winner is the false positive nothing has
  measured yet.
- **Every fifth latent frame draws more attention mass, and Sol keeps up with
  it: CLOSED.** The one-pixel-frame latents draw more exact mass than their
  share of keys at every depth but block 0, with a residue control
  (`bench/results/2026-09-19_attention_mass_covered_market_depth.md`), but
  Sol does not miss them more than their mass warrants
  (`bench/results/2026-09-19_sol_block_grouping.md`, section H). No routing
  prior is needed.
- **The text encoder's RoPE moved onto comfy-kitchen on 2026-09-15 and is not
  bit-identical.** A rounding-order difference of equal accuracy
  (`bench/results/2026-09-19_encoder_rope_kitchen.md`), so a same-seed clip
  rendered before that morning's core pull is not guaranteed to reproduce
  after it. Say so in any record that pairs clips across it.

**2026-09-15:**

- **TaoMate-H3: deprecated by the owner on 2026-09-27 and removed, not
  pursued** ([`decisions.md`](decisions.md), 2026-09-27). What was tried that
  day, the whole-clip arms and the ported streaming runtime, is in the
  `bench/results/2026-09-15_taomate_*` records and in git.
- **Cafe prompt pair: the rewrite wins, and so does 8 steps.** An outside
  prompt verbatim against its house-structure rewrite (`t2va_cafe_kids`), PDD
  at 5 and 8 steps. The owner judged the rewrite way better at both counts,
  and PDD 8-step better by far (`bench/results/2026-09-15_cafe_kids_verdict.json`).
  - **Worth re-judging:** with the dancer verdict the same day, 8 steps beat
    5 (tail5) on two scenes under the new default chain. That runs against
    the earlier tail5 preference, which was judged under sage.

**2026-09-14:**

- **M3-to-H3 bridge: run the image lane next.** The text lane closed (M3
  maps onto Qwen's text rows up to a cheap aligned map, an order of magnitude
  outside the tolerance two Qwen runtimes set at the DiT's refiner), the
  fifteen image pairs are captured on both sides, and the fit was stopped
  unfinished. The take, the results and the numbered next-session list are
  `internal/claude/2026-09-14_m3-h3-bridge_independent-take.md` (gitignored,
  this checkout); the fixed script is `internal/claude/m3-h3-adapter/fit_image_lane.py`
  and needs a GPU window agreed with the session holding the card.
- **Two duplicate Sol-on probes retired** (owner). `h3_probe_sol_on` and
  `h3_probe_sol_on_i2v` were Sol-on twins of graphs that were sage-only when
  written; with Sol on by default they matched `h3_text_to_video` and
  `h3_first_frame_to_video` in everything but the output name.
  `h3_probe_sol_on_all_refs` stays: its references (a video with its soundtrack
  and a standalone clip) make it a different workload, and the EasyCache and
  euler probes name it as their control. **Not yet examined:**
  `workflows/h3_probe_sol_on_refs_api.json`; diff it against the shipped
  reference graph it mirrors, then retire it or record why it stays.
- **API-only workflows: the editor side is unverified.** The editor loads an
  API file by input name (read from the frontend source, not exercised). Done
  when Export (API) from the editor diffs clean against the shipped file for
  `h3_text_to_video_api.json` (Resolution and Sol members),
  `h3_image_ref_plus_text_to_video_api.json` (reference chain, `size_policy`),
  `h3_text_to_video_pdd_api.json` (the steps input, control `fixed`) and
  `h3_text_to_video_audio_freeze_song_ref_pdd8_api.json` (`extent` members,
  seed control `fixed`).
- **Found, not fixed: two stale Sol readers.** (a)
  `bench/check_provenance_stamp.py` is red on `no_missing_knobs`:
  `sol_attn_h3.py`'s `token_aug_profile` is not recorded by `provenance.py`.
  Red at `1345791` too, so it predates the API-only commits. (b)
  `bench/analyze_sol_error.py`'s CUDA arm passes `centroid_tail=` to
  `comfy_kitchen.sol_attn`, which the installed kernel rejects since the
  centroid form became unconditional; its eager and dense arms still run
  (found by the channel-balance session). Observable for each: run the script.
- **Frozen-audio loop and prompt lists: what was built, by commit.** Stage 1,
  encode first, references, metadata and the window folder: `22b046c`
  (smoke record `b3ab31f`). Shot workflows and `MiniMaxH3JoinWindows`
  retired: `a68be96`. Live preview node dropped: `889b0a1`. Stage 2, resume:
  `0a2380f`. Stage 3, prompt lists and wildcard files: `1270a2d`, `3fb743b`.
  One fill path for every loop node, `MiniMaxH3FillPromptLists` and the example
  `workflows/h3_text_to_video_audio_freeze_song_lists_pdd8_api.json`: `eac1a6f`.
  Song graphs' own notes and the API list guard: `e07c462`. **The simplification
  the owner agreed, built the same day** (CHANGELOG 0.108.0,
  `git log -- loop_plan.py`): the cuts, the list node's `shuffle`, Fill Prompt
  Lists' fixed `index`, the `timeline` and the `preview` switch. Preview, and a
  render of the first 30 seconds (three windows of two lengths, joined),
  exercised on the served example graph:
  `data/results/2026-09-14_audio_freeze_song_timeline_smoke.jsonl` (`83d2633`).
  In that render the third window, which starts inside the verse, kept the
  medium shot it inherited instead of opening on the prompt's close-up; one
  sample. **Still owed:** the full-length example render (a throwaway first
  run; if most within-section seams do the same, the "each window opens on a
  close-up" premise in the generator's comment on that graph is wrong), resume
  on the reference song graph, one queue from the editor for the workflow
  chunk, and the preview seen on the canvas's Preview as Text node; song graphs
  the owner saved from the editor need re-making. Postmortem:
  `internal/postmortems/2026-09-14_session_api-only-and-loop-simplification.md`.
- **Frozen-audio loop, stage 1 of 4 built** (`audio_freeze_song.py`,
  `loop_output.py`, `workflows/h3_text_to_video_audio_freeze_song_ref_pdd8_api.json`):
  the track and each distinct prompt encoded before any window samples,
  reference stills with every window's prompt, window files in
  `<prefix>_windows/`, the prompt and workflow in the finished file plus an
  optional PNG (the workflow chunk arrives only from an editor queue and has
  not been exercised yet), and the seed control slot the song UI graphs were
  missing.
  Harness smoke, two short windows per arm, reference and plain:
  `data/results/2026-09-14_audio_freeze_song_stage1_smoke.jsonl`. Its first
  full-length run is still a throwaway. Next, as decided on 2026-09-14:
  (2) built the same day: resume from the first changed window, the seed
  held fixed (`loop_resume.py`; harness smoke
  `data/results/2026-09-14_audio_freeze_song_resume_smoke.jsonl`); (3)
  built the same day: a prompt list node filling
  `__name__` placeholders, one list per node with its own fixed seed, in
  order or reshuffled on each pass so no value repeats before the list is
  used up (an explicit `random` mode allows repeats), advancing only when a
  window uses its placeholder; wildcard `.txt` and JSON files from
  `<input>/wildcards/`, selectable on the node (`prompt_lists.py`,
  `bench/check_prompt_lists.py`; harness smoke
  `data/results/2026-09-14_audio_freeze_song_lists_smoke.jsonl`). The owner
  then agreed that a placeholder with no list node reads a wildcard file of
  its name directly, shuffled at seed 0, and that lists serve every loop, not
  the song alone: loop nodes fill through `prompt_lists.fill_windows`
  (enforced by `bench/check_prompt_lists.py`), and `MiniMaxH3FillPromptLists`
  fills any prompt input by index for chained or one-clip graphs. The example
  is `workflows/h3_text_to_video_audio_freeze_song_lists_pdd8_api.json` on
  `just-a-flicker.mp3`; its first full render is owed and a throwaway. Still
  owed on the loop: a full-length
  song, resume on the reference graph, and an editor queue to exercise the
  workflow chunk in the finished file. The shot-per-window
  workflows and `MiniMaxH3JoinWindows` retired unrendered the same day.
  Decisions: [`decisions.md`](decisions.md),
  2026-09-14.
- **Why the workflows ship as UI and API twins**: a read-only subagent review
  found no recorded reason; the API form is what every runner and most checks
  read, and the UI form carries the notes, titles, layout and bypass state
  that the frontend's deprecated API import does not rebuild. Open for the
  owner: which graphs get opened in the editor, and whether the notes are
  used. The owner then chose API-only workflows (2026-09-14); the
  implementation and verification plan for a later session is
  `internal/2026-09-14_api_only_workflows_plan.md` (not shipped), and it
  recommends converting before loop stage 2. **Done the same day**: shipped
  workflows are API format only ([`decisions.md`](decisions.md), 2026-09-14).

**2026-09-13:**

- **Reference-view ablation, second edition: rendered, awaiting the owner's
  blind verdicts.** `bench/refview2_arms.json`: five ref2va scene graphs
  (`h3_config.REFVIEW2_SCENES`, `workflows/h3_probe_refview2_*.json`) built
  at `MiniMaxH3AppendRefImage`'s new defaults (vendor parity: 2048 short
  edge, upscale on, one copy for both the video VAE and Qwen3-VL). Four arms
  per scene after the owner cut the middle tier mid-run (`parity`,
  `noup_shared`, `noup_q512`, `noup_q2048`); one seed, by the owner's call.
  Rows: `data/results/2026-09-13_refview2_arms.jsonl`, no failures. The
  blind session is built: `Video/blind/refview2_2026-09-13/` on the output
  share, every render as a neutral single plus fifteen stacked pairs (parity
  against each other arm per scene), `score.html` beside them, key sealed
  under `internal/blind_keys/`. What it settles: the `qwen_view` default
  (if `noup_q512` beats `parity` across scenes, the 2026-08-27 separate view
  was right and returns; if not, `shared` stands; *2026-10-03: the owner
  returned it without this session, on two clips of the finish graph,
  `decisions.md`*) and the `allow_upscale`
  default (if `noup_shared` matches `parity` on identity, the upscale buys
  rows and nothing else). One seed is one sample per pair; the reading is
  across the five scenes. Nothing is claimed until scored. The old three-arm
  Gate 6 family (`h3_probe_refview_{a_source,b_qwen2048,c_parity}`,
  `bench/gate6_refview_arms.json`) was never rendered and is deleted; the
  owner's decisions of the day are in [`decisions.md`](decisions.md).
- **From the sage-fork session, 2026-09-13** (its CHANGELOG decision-log
  entry "sm89 q/k quantization: per-thread Triton stays; per-warp CUDA
  measured slower end to end -- 2026-09-13"): the fused CUDA quantizers now
  form 64-bit offsets, bit-identical to the old build at every H3 length
  tried and free; per-thread q/k stays on this card because per-warp is
  slower for the whole call and grades worse on two captured cells; the
  quantization pass is a small fraction of the attention call at every
  length, so no fusion there can move a render. **One item for this repo:**
  on the late cell (block 49, step 15) `smooth_k=True` improved both fp8++
  and fp16 by several percent where the fork's 2026-08-05 early-cell run
  found it inert. `attention.py` passes `smooth_k=False`, and the consume
  path's memory saving depends on it (the comment beside it), so this is a
  graded trade to run on captures (`bench/grade_sage_on_capture.py`) across
  early and late cells, not a flip. The 2026-09-03 base-16 capture that
  served the fork's grading is retention-extended to 2026-10-31 (the owner,
  2026-09-18: every capture-graded record since uses its cells as the common
  yardstick) and is due for `bench/recycle_captures.py` after that; the
  owner's call.
  **The served sage build** since the 17:16 restart is the fork's tag
  `served/2026-09-13` (commit 1408254; the process started on the 071b186
  tree, comments apart), with `sageattention.quant.ELEMENT_OFFSET_BITS`
  64. Nothing on the H3 path changed numerically: the fork's table has the
  new kernels bit-equal to the previous build at every shape this card
  renders. **Owner principle, same day**: the default at every layer is the
  measured best and the chain is checked end to end, kernel build to node
  default to generated graph. Not yet done for the kernel layer: the sage
  build is not in `bench/run_graph_arms.py`'s `substrate`, and no check
  asserts the served build equals the tagged one. Done when a row carries
  the fork commit or tag and a check reads it off the installed module.
- **Rendered, awaiting the owner's verdict**: `untold_loose_pdd8` at a
  second seed (2203), `Video/h3_candidate_t2v_pdd8_baked_audio_freeze_untold_loose_pdd8_00002-audio.mp4`,
  row in `data/results/2026-09-13_audio_freeze_untold_seed2.jsonl`. The
  first seed lost lip sync after about two seconds; this one says whether
  loose mask with neither transcript nor guide rows fails every time or did
  once. One line into the step 2 verdict file closes it.
- **What changed on 2026-09-13, and where each is recorded** (the day's
  session log is `internal/log/log_2026-09-13.md`; the decisions are in
  [`decisions.md`](decisions.md); the changelog is 0.100.0):
  - *Node defaults.* `MiniMaxH3AppendRefImage`: `size_policy=max`,
    `dit_short_edge=2048`, `allow_upscale=True`, `qwen_view=shared` (vendor
    parity). `MiniMaxH3ReferenceConditioning`: `video_policy` and
    `image_policy` offer `comfy` (default) and `release`; `encoder` removed.
    `MiniMaxH3FreezeAudioWindow.context_frames` min 39 step 51, no zero
    mode; `MiniMaxH3AudioFreezeSong.max_seconds` replaced by the combo
    `extent`. Observables: each node's `define_schema`, served by
    `/object_info`; `bench/check_workflow_schema.py` green against it.
  - *New node.* `MiniMaxH3ReferenceReport` (`reference_report.py`).
  - *Rules and config.* `h3_rules.REF_QWEN_SHORT_EDGE` is now only the
    value under `separate`; `h3_config.REFVIEW2_SCENES` added;
    `ENCODER_V1`/`ENCODER_V2` removed; `REF_VIDEO_BUDGET` still turns
    upscale off on the video-bearing reference arms.
  - *Generator.* `ref_upscale` default True, `ref_video_policy` default
    `comfy`, `ref_qwen_short_edge` default 0 (shared); the AWQ loader branch
    gone; `--dump-prompts` emits a per-window list for `freeze_shots`
    graphs; the refview2 family replaces the Gate 6 family;
    `h3_probe_reference_upscale` now turns upscale OFF. Every graph
    regenerated and validated against the live server.
  - *Deleted.* `h3_awq_encoder.py`, `MiniMaxH3AWQEncoderLoader`, `config/`,
    `docs/h3_awq_encoder.md`, seven AWQ-only bench tools, the three Gate 6
    graphs and their manifest.
  - *Owner rule.* A numeric widget never means a mode by being 0
    (`bench/check_literal_widgets.py` enforces it; green).
  - *Learned.* The two sage quantizer ceilings and which path this card runs
    (`docs/h3_input_impacts.md`); the vendor's reference sizing is one 2048
    copy for both towers (`docs/research/sglang_h3_pipeline.md`); reference
    rows cannot be pre-encoded per step because attention is unmasked
    (`comfy/ldm/minimax/model.py`); PR Comfy-Org/ComfyUI#16187 changes VAE
    numerics only, not reference cost or geometry; on the audio-freeze lane
    the words are not required to hold a mouth to a frozen clip (the step 2
    verdict file, entries dated 2026-09-13); the prompt describer had
    recorded no prompt id for reference graphs (`workflows/prompts.py`).
  - *Good.* Every check touched is green; the ablation rendered with no
    failures including the three-still parity arm on this card; the report
    node's numbers match the static pricer's
    (`data/results/2026-09-13_reference_settings_three_stills.json`).
- Before queueing a reference render, read `MiniMaxH3ReferenceConditioning`'s
  preview or run `bench/preflight_graph.py`: both price what each reader sees
  (the `MiniMaxH3ReferenceReport` node this used to name was deleted in 0.174.0).

  *2026-10-05: closed unscored. The sealed key under `internal/blind_keys/`
  is no longer on disk (found by the memory audit of 2026-10-05), so the
  session cannot be unblinded, and the owner chose to move on ("Im not sure
  how much value it added anyway"). The `qwen_view` default it was built to
  settle was decided on 2026-10-03 by eye instead (`decisions.md`). A future
  blind session seals a new key; `bench/blind_batch.py::KEY_DIR` is the
  place and the directory is made on demand.*

**2026-09-12:**

- Audio-freeze lane: [`../h3_audio_freeze.md`](../h3_audio_freeze.md)
  section 4 is the plan, rewritten after the first verdicts. Step 1 built
  and controlled; step 2's mechanism question closed on the owner's word for
  both scenes (dancer follows the beat, speaker lip-syncs;
  `bench/results/2026-09-12_audio_freeze_step2_verdict.json`). Next: PDD8
  as the iteration chain (`bench/audio_freeze_pdd_arms.json`), the two open
  pairs (loose mask at a second seed, transcript or not), first-frame, then
  the loop.
- **Morning brief, 2026-09-13.** The lane doc's "For the owner to judge"
  table lists seven clips (the seam, three motion arms, three untold arms)
  and the verdict file takes one line each. Then the first run of the
  whole-track node `MiniMaxH3AudioFreezeSong`
  (`workflows/h3_text_to_video_audio_freeze_song_api.json`, 30 s look; its
  one-window smoke row is `data/results/2026-09-12_audio_freeze_song_smoke.jsonl`),
  the shot chain graphs, and `bench/audio_freeze_gain_arms.json` on PDD8.
  PDD8 is the iteration chain; the base chain is for keepers. The mouth
  instrument needs a face model before it says anything.

**2026-09-11:**

- Graphs regenerated on the rebuilt venv and validated against a live server;
  `bench/check_attention_defaults.py` passes, with `end_percent` 1.0 on every
  Sol node except `h3_candidate_t2v_pdd8_sol_narrow`'s.
- `bench/check_node_ids.py` fails on `MiniMaxH3SolAttn` since `f980aeb`
  (2026-09-08) because it compares declaration order. The server's
  `input_order` keeps the required inputs in the manifest's order and adds
  `token_aug_blocks` as the only optional input after them. Fix the check to
  compare required then optional, then record the append. *2026-09-12: the
  append is recorded (`bench/node_id_manifest.json`, regenerated with
  `--write` when the freeze node was added) and the check passes against
  it; whether the check should also tolerate a required-then-optional
  reorder is still open.*

**2026-09-10 (upstream survey session):**

- Core vs ours Sol A/B, rendered 2026-09-10 with no verdict record
  (`data/results/2026-09-10_sol_core_ab_arms.jsonl`; this line said "BUILT not
  rendered" until 2026-09-15): courtroom, disco, hacker; two seeds; pairs A
  ours vs core, B ours at core's values vs core, C ours vs all-rows.
  *Retired 2026-09-15*, manifest `bench/sol_core_ab_arms.json` and graph
  `h3_probe_t2v_sol_core` together: the default floor moved to the kitchen
  backend (`28d8ee5`), so a re-run would set ours on kitchen against core's
  node on sage under this A/B's name. The clips remain on disk and unscored,
  in the ComfyUI output folder under the basenames
  `data/results/2026-09-10_sol_core_ab_outputs.json` lists, and the scoring
  below is still possible from those files. Ours against core again is a new
  manifest, core's node over the backend node, not a re-run.
- All-rows sink switch: planned, gated on pair C (`docs/roadmap.md`, forward
  plan step 3).
- Pair B, GRADED on activations
  (`data/results/2026-09-10_sol_impl_capture_grade.json`, capture
  `2026-09-10_sol_impl_courtroom`, its repo records beside it): at matched
  knobs core's chunked producer and our node land the same distance from
  exact in every cell, carried statistics do not move that distance, and the
  rebuilt Q/K/V equal the capture. Against each other they sit above the
  kernel's all-routed floor in most cells (every carried cell), so pair B's
  rendered pairs mostly show sample divergence. Ours at the then-shipped
  policy is not uniformly closer to exact than core's defaults: it loses on
  the block-0 cells and the per-row mean and wins the whole-tensor mean
  through block 49 (`docs/SOLATTN.md`, the 2026-09-10 core section; corrected
  2026-09-27); pair A's scores say whether the policy difference is visible.
  The text segment is the worst segment in every arm, which is the all-rows
  argument again.
- A/B scoring: the owner scores `sol_core_ab_2026-09-10`; then
  `bench/score_session.py`, the loudness record
  (`bench/measure_clip_loudness.py --outputs
  data/results/2026-09-10_sol_core_ab_outputs.json --baseline-rung ours
  --out bench/results/2026-09-10_sol_core_ab_audio_loudness.json`),
  the frontier table, and the all-rows decision from pair C. Whoever writes
  the verdict up says two things `score_session.py` has no field for: pair
  B was settled on activations before scoring, so its verdicts read as
  sample divergence; and every row's code is "server started 15:55, node code
  `8870f8e`", because the rows stamp the repo HEAD at render time, not the
  code the server loaded.
- Run rows need a code stamp from the server. The rows
  `bench/run_graph_arms.py` writes carry `substrate.git_commit` and
  `git_dirty`, read from the working tree when each row is written, so a
  commit during a batch relabels every later row. In
  `data/results/2026-09-10_sol_core_ab_arms.jsonl` the stamp moves from
  `3e553f6` to `311cffa` mid-batch, while the server ran `8870f8e`
  throughout. Done when rows also carry the serving process's start time,
  or the commit at its start. Wrong premise if the owner rules that no
  commit is made during a batch. Until then, verdict write-ups state the
  server's start time and loaded code by hand.
- The Base16 capture passed its `keep_until` on 2026-09-10 and the recycler
  lists it recyclable; deleting it is the owner's call.
- `bench/check_dit_prefix_attention.py` is indexed as CPU-only but cannot run
  with CUDA masked: it imports `comfy_extras.nodes_minimax_h3`, whose import
  chain reaches `comfy.model_management`, which asks CUDA for a device at
  import. Run it with the card free, or give it ComfyUI's `--cpu` the way
  `bench/audit_ref_audio_crop.py` does.
- Keyframe-encode determinism: `bench/measure_keyframe_encode_determinism.py`,
  written, not yet run (open experiment 30).
- Reference audio now end-padded to the audio VAE's hop in our path
  (`reference_conditioning.py::_encode_ref_audio_aligned`), closing gap 16 for
  this pack; core's own nodes still crop until Comfy-Org/ComfyUI#15972 lands.

- **Owner decisions 2026-09-05 evening** (`docs/roadmap.md`, the decisions
  block): composition counts as prompt adherence; PDD parked; the turbo
  rung goes with two LoRAs at two seeds (bake session owns it, plus a sage
  floor render at the second seed); token routing stage one goes (survey
  session offline at the time, pick it up from the plan page).
- The bake pair: SCORED and joined 2026-09-05, pairs only (`data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`, read its
  `join_notes_2026-09-05`). The baked arm lost to the sage floor on four
  scenes and read same on one, on the same properties the merged arm lost
  on the day before (audio level, lighting, framing, the subway sign
  text); merged against baked read same on two, merged won two (standoff, subway,
  the latter after the owner corrected a mis-clicked verdict) and baked won
  diner on composition. Reading: at
  this seed the PDD8 defects are the schedule's, not the merge's; the
  bake is a hygiene and cost change and stays the PDD8 checkpoint of
  choice for that reason alone. Next lever for PDD quality is the
  partition (the tail-weighted six-evaluation schedule the node already
  accepts, `h3_config.PDD_MANUAL_SIGMAS`), not the weights. Records beside it:
  `data/results/2026-09-05_pdd_bake_outputs.json`,
  `..._pdd_bake_audio_loudness.json` (absolute levels; the merged twins'
  are in the 2026-09-04 ladder record) and
  `..._pdd_bake_2026-09-05_frontier.json`, where the baked arm's sampler
  time sits beside the merged arm's, as predicted.
- Scoring page rubric (`bench/rubrics/default.json`): the pair tab's per-half
  quick tags reuse the singles' option list, so "same" and "can't tell"
  appear as tags on one half of a pair, where they mean nothing (the owner
  asked what they meant, 2026-09-05). Give the pair tab its own tag list
  (good / not good / off per half) before the next session is built; the
  three built pages are unaffected and the join ignores those two tags on
  a half.
- PDD ladder: SCORED and joined 2026-09-05, pairs only, singles unscored by
  the owner's choice (`data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`). Reading, one seed, five scenes: the sage
  16-step floor beat PDD8 under sage alone on every scene and beat PDD8 with
  the narrow Sol window on four with one can't-tell, so the PDD8 loss is
  PDD's own, not Sol's; Sol on two of eight steps read *same* as sage alone
  on all five; the shipped four-step window lost to sage alone on standoff
  (background definition) and subway (scrambled first-frame text,
  artifacts) and read *same* on three. The owner heard the floor louder than
  PDD on every scene compared; `data/results/2026-09-04_pdd_ladder_audio_loudness.json`
  is the level record. Merge against schedule is what the bake pair decides.
- Frontier table on that verdict: DONE 2026-09-05,
  `data/results/2026-09-05_pdd_ladder_2026-09-04_frontier.json`, speed against
  the sage floor beside the owner's note per pair (the tool now skips
  warmup rows, which shared their arm's label).
- The chain assert's call-time proof on the no-sage graph: the exercise
  skips for headroom on this card with the DiT staged, so the Sol-over-stock
  state is proved at registration only. Size the exercise to the probes the
  state actually fires (`assert_chain.py::_exercise`); a branch, after the
  card is down. Closes when a just-Sol render's log carries the "no sage:
  probes ... reached no sage kernel" line. Item 7 of the 2026-09-04
  postmortem under `internal/postmortems/`.
- The fp16-sage rung's lost time (`data/results/2026-09-03_ladder_outputs.json`
  shows it slower than Sol as shipped on every scene): moot since the
  2026-09-04 plan dropped the rung (annotated in the 2026-09-03 postmortem
  under `internal/postmortems/` on 2026-09-05); reopens only if the rung
  returns.

---

Items from the `evalman` render lane:

- Just-Sol session: SCORED and joined 2026-09-05, pairs only (`data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`, read its
  `join_notes_2026-09-05`). The sage floor beat just-Sol on four scenes with
  one can't-tell; Sol as shipped beat it on three and read same on two; the
  tau-raised arm lost both its subway pairs. Nearly every win is a
  composition difference (sample divergence), the one quality loss is
  stairwell (lighting, skin, level), so the direction is against running
  Sol without sage and the evidence is weak. The shipped chain (sage
  outer steps, Sol inside) stays the leader. Frontier table: UNBLOCKED and
  written 2026-09-08,
  `data/results/2026-09-08_sol_nosage_2026-09-04_frontier.json` (the session's
  own outputs record plus the ladder's, which carries the sage floor rows).
  The split now takes the FIRST underscore, so a rung may carry one, with
  `--scene` to declare a scene name that does; `bench/frontier_table.py
  --controls` covers the underscored rung, the declared scene and the
  unchanged plain labels. `bench/measure_clip_loudness.py` took the same fix,
  prophylactically: its 2026-09-04 record never included the tau-raised arm,
  so nothing measured was wrong -- but the frontier table now carries a row
  whose loudness has no entry. Closes when that arm has a loudness row or the
  record says why it is absent.
- One pair, no card time: the all-rows sink mode's armed subway clip
  against the 2026-09-03 subway Sol clip
  (`data/results/2026-09-04_probe_allrows_vs_shipped_pixels.json` says the
  change is the size of dense-versus-sage;
  `data/results/2026-09-04_probe_render_vs_unarmed_pixels.json` says an
  armed clip is a valid sample). Closes with a scored one-pair session.
- An armed render of the just-Sol graph under `H3_SOL_PROBE`, so the
  probe's counterfactual is stock attention and the record ranks Sol
  against near-exact attention directly (roadmap 2026-09-04, step 1's next
  footing). Closes with a probe record and its comparison against
  `data/results/2026-09-04_sol_probe_base16_subway.json`.
- Then the vendor's start-schedule arm, open experiment 27 in
  `../open_experiments.md` (`upman`'s lane), in the step-policy slot: probe
  first, then one blind pair.
- The PDD bake's hypothesis, written 2026-09-05 in the roadmap's bake
  paragraph: the defects named blind on the shipped PDD8 rung are the kind
  a requantised merge produces; the baked checkpoint on the same scenes at
  the same seed is the test. The plan, owner-approved 2026-09-05 and under
  the render lane's review, is
  [`../research/pdd/2026-09-05_bake_plan.md`](../research/pdd/2026-09-05_bake_plan.md);
  closes with the baked pair on disk, the contract check green on it, and
  the merged-versus-baked pair blinded.
- Unscheduled: sage at more steps against dense at sixteen, open
  experiment 6 in `../open_experiments.md`, the base-model question the
  owner raised.

Items from the PDD backbone bake lane (the alibaba-pai PDD LoRA folded into the fl2va int8 checkpoint; the peer session that built it happens to share a name with the larryvrh turbo LoRA, which is a different artifact and not part of this bake):

- **The turbo rung closes unscored** (2026-09-26, with the lightx2v lane;
  `../roadmap.md`, "Closed lanes"). Both of its arms were turbo LoRAs (the
  larryvrh pack, gone 2026-09-23, and lightx2v v1.2). The run records
  `bench/results/2026-09-05_turbo_rung_*.jsonl` and the two blind sessions
  in the output folder stay as they are; nothing was deleted.

- The bake script's open review candidates: a control-before-bake gate,
  the contract check's retyped filenames now that `h3_config` names the
  bake and the stripped sidecar, the dirty-tree marker in the bake's
  metadata, and the duplicated helpers. The list with a state per row is
  in the session's internal notes (2026-09-05, bake code review) and
  forward item 4 of the postmortem under `internal/postmortems/`. Closes
  when each row is fixed or dismissed with a reason in `CHANGELOG.md`.
- The stripped sidecar into the HF sidecar staging tree beside the four
  full ones; the checkpoint's publication is the owner's separate call.
  Closes when the bundle notice names it.
- The ref2va bake: `bench/bake_pdd_checkpoint.py` pointed at the other
  partition, after the fl2va pair is judged. Closes with the pair on disk
  and the three verifications green.

## Candidates on trial (2026-09-05): what to render your own prompts on

The owner asked for canonical graphs carrying the lane's current best
guess. Generated by `workflows/build_workflows.py` like every graph here
(never hand-edited; the Sol fields that differ from the recipe are declared
in `bench/check_attention_defaults.py::DEVIATIONS`, which grades each
deviation as real). Drag the `_api.json` into the editor, replace the prompt, render.
Rows are ordered fastest first by their step mix; the per-step costs that
turn a mix into a time are the medians in
`data/results/2026-09-03_ladder_outputs.json`. Nothing below is a verdict,
and the last column says why it is on the list and what would make it one.

**The step mixes below predate `07b903c` (2026-09-11)**, which moved
`end_percent` to 1.0 and removed the dense last step. The leader and the
all-rows candidate now run four sage and twelve Sol of sixteen, reasoned from
the Sol window's sigma band at shift 12 rather than re-rendered; the PDD rows
changed too and are not recounted here.

| graph | what it is | step mix (of 16, or of 8 under PDD) | why it is here, and what settles it |
|---|---|---|---|
| `workflows/h3_candidate_t2v_pdd8_sol_narrow_api.json` | PDD8 with Sol on two of the eight steps | six sage, two Sol | the shipped PDD8 rung lost every scene blind and carried Sol on four steps; this asks whether Sol on the coarse schedule is what lost. Blinded as `pdd_ladder_2026-09-04`; unjudged |
| `workflows/h3_candidate_t2v_pdd8_baked_api.json` | the shipped PDD8 graph on the baked checkpoint with the stripped sidecar | four sage, four Sol | same work as the shipped PDD8 graph with the merge noise gone: the PDD backbone is quantised once with the weights instead of merged and requantised at load (`docs/research/pdd/2026-09-05_bake_plan.md`). Settled by the merged-versus-baked pair, `pdd_bake_2026-09-05`, rendered under sage alone so Sol is not in it |
| `workflows/distill_experiments/h3_probe_t2v_pdd8_sage_api.json` | PDD8 under sage alone, no Sol | eight sage | the PDD floor under the sage decision; if it reads the same as the sage floor, PDD8 is a near-doubling over it. Blinded as `pdd_ladder_2026-09-04`; unjudged |
| `workflows/h3_candidate_t2v_sol_only_api.json` | just Sol, from the first step to the last-but-one, no sage node | one stock, fifteen Sol | the fastest base-model arm: the probe records put Sol's disagreement lowest early in the window, so the dense warm-up may be unneeded, and the owner asked what Sol does without sage under it. Blinded as `sol_nosage_2026-09-04`; unjudged |
| `workflows/h3_text_to_video_api.json` | **the leader**: sage auto plus Sol as shipped | five sage, eleven Sol | the only arm with a blind verdict behind it: same as dense on four scenes of five at one seed, and the fifth scene's slip belonged to sage (`data/results/2026-09-04_ladder_2026-09-03_frontier.json`, `docs/roadmap.md` step 1) |
| `workflows/h3_candidate_t2v_sol_allrows_api.json` | the leader's chain with the text rows dense too | five sage, eleven Sol, plus a few hundred dense rows | free on the probe (the text segment's disagreement collapses to the audio floor) and the pixels move as far as dense-versus-sage; one blind pair away, already rendered |

Not on the list: the shipped PDD8 graph (`workflows/h3_text_to_video_pdd_api.json`),
which lost to the true baseline on every scene of the 2026-09-03 ladder; it
stays the reference the two PDD candidates are judged against. PDD
parameters on every PDD graph are the vendor's contract as
`workflows/h3_config.py` states them (the fl2va eight-step LoRA at its
shipped strength, heads on, the sampler's own step count); the baked
candidate above changes the weights those parameters act on, not the
parameters. The chunked Sol node is not wired into any of these:
it is a memory lever (`sol_chunked_h3.py`), and at the trained canvas and
length every arm above rendered on this card without it.

## Now

- **Sol dense blocks reanalysis and 5-test matrix, 2026-10-02.** Test 1's all-block
  instrumentation run on Ref2VA (119k tokens) under `quantizer="rotated"` proved that
  Hadamard rotation resolved the $K$-norm outlier issue on blocks 45–48 (Block 48 exhibits
  7.14% relative error). However, sparsity truncation error revealed a massive middle-network
  spike across blocks 38–43 (peaking at Block 42 with 25.67% error and Block 39 with 25.22%).
  `dense_blocks="45,48,49"` is under trial for replacement by Option A (`"39,41,42,49"`) or
  Option B (`"39,40,41,42,49"`). Details and the active 5-test evaluation matrix are in
  [`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)
  and [`decisions.md`](decisions.md).
- **Prompting docs, 2026-10-01.** `../prompting.md` is the only home of a prompting
  rule and its section 16 is specificity. The dead code is removed and the song node's
  prompts are graded. What is still open for the owner (the song prompt's mood word,
  the closed-record scripts, the stale published portable copy, and fixes no render has
  confirmed) is in `../../bench/results/2026-10-01_prompting_docs_audit.md`.
- **Distill weights and routes, from the 2026-09-26 overnight batch.** The
  summary is fastdude's `../../bench/results/2026-09-27_followup_summary.md`;
  the ranked takeaways are in
  `../../bench/results/2026-09-27_evening_takeaways.md`. What those left open
  (#34 to #46) was run, reviewed by the owner or decided on 2026-09-27. What
  remains is in "Open after the 2026-09-27 night" at the top of this page.

*2026-09-19, two corrections to the paragraph below.* (1) The "sage's audio more natural" lead is WITHDRAWN as a lead
about chains: the pair the owner heard was sage against sage (the 2026-09-15 kitchen-scene "default" clip predates the chain
move and is sage `auto`); on the true chain pairs every measured audio difference is inside the floor
(`bench/results/2026-09-19_audio_sage_vs_kitchen.md`), while Sol's token reorder moves the audio MORE than the chain does.
(2) For the same reason the reorder panel has five valid pairs, not six: plain order better on two, `3d` on one, two ties.
`bench/diff_clip_graphs.py` now shows what actually differs between two clips before a stack is built. Also since then: the
sage node's `auto` is the rotated mode and the sage chain names it (0.129.0); 362 frames is legal, 345 is the default length.

*End of 2026-09-18, where things stand.* The reorder works under core's memory compiler and stays OFF by default: six
full-length pairs scored blind showed no benefit (`bench/results/2026-09-18_sol_reorder_panel.md`), while the noodle bar at
its declared 107 frames showed plain order ghosting a figure in and the reorder not
(`bench/results/2026-09-18_off_length_prompts.md`); a short-clip panel decides. Most of what looked like attention trouble
was prompt trouble: two prompts rendered at three times their length, an action with no agent, a speaker with no seat, a
plural that doubled an object. The bank lost every shot-header timestamp (the owner's rule; one same-seed pair says it is
safe, `bench/results/2026-09-18_timestamps_diner.md`) and is getting one-clause wording fixes. Sage against kitchen on the
same prompt: no visual difference, and the owner found sage's audio more natural on one scene
(`bench/results/2026-09-18_sage_chain_panel.md`), which is the live lead on which chain should be the default. Sol `rotate`
and the sage set's rotated mode are HELD: no visible benefit shown, and a default flip costs every reference clip its
bit-identity. Next: score the short-clip panel, the sage diner pair and the six 2026-09-15 pairs by ear; take the two
approved captures; rewrite the 345-frame prompts that script only a few seconds.

*2026-09-18, corrected: the noodle bar result below came from an over-length render* (a 107-frame prompt at 345; the owner), so it shows how Sol behaves while the model improvises past a script, not a defect of normal use (`bench/results/2026-09-18_off_length_prompts.md`). The panel and an on-length rerun decide what is left of it.

*2026-09-17: the reorder is the lever the eye can see.* On the noodle bar the
owner found every plain-order arm morphing and the two `3d` reorder arms clean
([`decisions.md`](decisions.md)). *2026-09-18: the reorder now works with
core's memory compiler, bit-identical to the compiler-off render
(`bench/results/2026-09-18_sol_reorder_under_memory_compiler.md`).* Next: a
panel of four or five bank scenes, plain order against the reorder, before any
default moves ([`../eval_comparison.md`](../eval_comparison.md), "A stress
scene is not a typical scene"). Waiting on the owner's go: Sol `rotate` on, and
the sage chain to `fp8++ rotated`; `start_percent` stays at 0.2. Two clips the
owner reported as bad on 2026-09-17 are open, one a lattice artifact on a
PDD-baked checkpoint run off its recipe, one the shipped sage graph on a
nature prompt (`bench/results/2026-09-17_owner_reported_bad_clips.md`). The
paragraphs below are older states.

*2026-09-15: the floor moved.* The default dense kernel under Sol is core's
Model Attention Backend on kitchen int8 (`h3_config.DENSE_BACKEND_NODE`), sage
off, Sol's `qk_balance` on (owner; [`decisions.md`](decisions.md)). The
paragraph below is the 2026-09-04 state.

**The routing policy is the lane, and sage is the floor.** Two owner
decisions on 2026-09-04, recorded with what they change and the sequence
they set in [`../roadmap.md`](../roadmap.md) (forward plan 2026-09-04): the
next render on the card is the online probe on the subway scene, the one
scene where Sol lost blind, set beside the standoff record; then the PDD
ladder below; then the three policy axes the node already exposes, segment,
block and step, each measured on the probe before one blind pair.

**Two sessions wait on the owner's eyes, and one pair costs no card time.**
`pdd_ladder_2026-09-04` (PDD8 under sage alone and with Sol on two of eight
steps, against the ladder's sage and shipped-PDD8 clips) and, once its
renders land, the just-Sol session (`bench/sol_nosage_arms.json`, against the
ladder's sage and Sol clips). A third, one pair on the subway scene, needs no
render: the all-rows sink mode's armed clip against the 2026-09-03 Sol clip,
since arming the probe does not perturb a render
(`data/results/2026-09-04_probe_render_vs_unarmed_pixels.json`). After the
just-Sol renders, an ARMED render of the just-Sol graph is the probe's next
footing: with no sage, its counterfactual is stock attention, so the record
measures Sol against near-exact attention directly rather than against sage.

**Read the ladder verdict, as a frontier.**
`data/results/2026-09-04_ladder_2026-09-03_frontier.json`
(`bench/frontier_table.py`) puts each arm's speed against the sage floor
beside the owner's own words from every pair it sat in; that is the form
every ladder is read in from now on, speed beside what was noticed, never
pass or fail.
The owner scored every stacked pair of session `ladder_2026-09-03` on
2026-09-04 (five bank scenes by five rungs, matched seed, trained canvas;
`data/results/2026-09-03_ladder_outputs.json` maps arm to clip and holds
the timings). `data/results/2026-09-04_ladder_2026-09-03_verdict.json`
holds the tally per contest, the owner's note on each pair, and which arm
sat in each slot, joined with the sealed key after the key's rows and labels
were checked against the render record. Two reading rules the record cannot
enforce: within a scene every contest shares one clip per rung, so the
verdicts in a scene are one sample of each arm judged more than once, not
independent votes; and no single was scored, so nothing audible was judged
and the manifest's audio questions (the diner's speech against its sound
bed, the opera's sung line) stand open. The direction the record gives: the
PDD8 rung lost to the dense baseline on every scene, on a named defect each
time, and the owner named it as PDD unprompted on two pairs; on the four
scenes without fast motion, sage alone and Sol as shipped were called the
same as dense or could not be told from it; on the subway chase every rung
lost to dense, narrowly for sage and Sol in the owner's words; the fp16-sage
rung went both ways. One seed anchors; it does not decide. **The PDD8 rung
is confounded, as the owner pointed out on reading the result:** every rung
above dense carries sage, the sol, solfp16 and pdd8 rungs all carry Sol, and
pdd8 also carries the PDD-specific Sol `end_percent`
(`h3_config.sol_for_graph`). No arm renders PDD8 without Sol or without
sage, so a pdd8 loss does not attribute to the step distillation, the
kernels, or their interaction. The audio half is the owner's report in
conversation, recorded with its provenance in the verdict record's
`owner_report_after_join`: good across the board, a bit quieter on the PDD
clips; `data/results/2026-09-04_ladder_audio_loudness.json`
(`bench/measure_clip_loudness.py`, EBU R128 per clip) is the level beside
that sentence, descriptive at one clip per arm, and its sign summary says
on how many scenes each rung sits below dense. The owner does not want
more seeds; the question is what runs faster at quality the owner cannot
tell from dense, and the ladder's answer so far is sage alone and Sol as
shipped on four scenes of five. The blind page for this session
on the share was edited by hand after it was built (its pair questions
export under `clip_rubric`); `bench/score_session.py` reads that key and
records which it read. Regenerate a page rather than editing it.

**The online Sol-versus-Sage instrument is validated; read it, then widen
it.** `sol_block_probe.py` passed its fixture controls, reproduced the
offline record on every retained Base16 cell
(`data/results/2026-09-03_probe_replay_base16.txt`), and its first canonical
record holds every invariant
(`data/results/2026-09-03_sol_probe_base16_standoff.json`, summary as data;
the closing section of
[`../research/2026-09-03_sol_exact_pquant_and_base_capture.md`](../research/2026-09-03_sol_exact_pquant_and_base_capture.md)
says what it ranks and what it cannot say). Next, first on the card under
the 2026-09-04 plan: the subway scene on the same footing, then legal PDD8
through the node's own sigmas, then Ref2VA, each its own population. Arm
with `H3_SOL_PROBE` on a restart you own; timings from an armed server are
void, and the 2026-09-03 probe render's row shows the render itself takes
about three times an unarmed Sol render.

**Render the PDD ladder, then blind it against the ladder's sage clips.**
Built 2026-09-04 at the owner's ask (PDD8 without Sol, and PDD8 with Sol at
more conservative settings) and cut to three rungs under the sage-always
decision: `bench/pdd_ladder_arms.json`, on the five ladder scenes at the
ladder's seed. PDD8 under sage alone is a new probe graph
(`h3_probe_t2v_pdd8_sage`, generated, never edited); the shipped PDD8 graph
renders twice, once as shipped and once with Sol's window narrowed to two of
the eight steps, which is reasoned from `docs/SOLATTN.md`'s sigma-window
arithmetic rather than measured. The manifest names the controlled pairs,
the reading rules, the sage rows to append from the ladder's JSONL, and the
one caveat: the ladder's clips predate the 2026-09-04 ComfyUI pull, so pairs
against them are cross-regime until a rung re-rendered after the pull is
shown pixel-identical to its 2026-09-03 clip. The probe graphs were built
with `--no-validate` on a stopped server; validate against the live
`/object_info` before the first render. Score the singles this time.

**The reference pathway verdict is in; read it before touching a reference
graph's cost.** Scored and joined 2026-09-04
(`data/results/2026-09-04_ref_pathway_2026-09-03_verdict.json`, one seed,
pairs only): the owner could not tell both pathways from encoder only on
either conditioner, named none of the identity items as differing, and
named one signature on every encoder-only arm, static people in the
background. Open experiment 26 in
[`../open_experiments.md`](../open_experiments.md) carries the reading and
what would close it; it leans to the cheaper arm and is not closed. No
action is queued on it; a second seed is the manifest's `run` line if the
owner wants one.

**Block 49's token-routing instability: the variation has the selection unit's
shape, and the kernel source names a mechanism.** Two records, read in order:
`data/results/2026-09-08_token_aug_determinism_shape.json` establishes that
the variation is not accumulation order and eliminates the two obvious causes
(centroid fidelity, refuted by our own morton measurement; the high-norm-row
hotspot, refuted by measurement the same day), and
`data/results/2026-09-08_token_aug_selection_structure.json`
(`bench/probe_token_aug_selection_structure.py`) says the moving rows carry the
granularity of one token-routing centroid, against a scattered control the run
measures on its own rows and a shifted-grid control for the alignment. All five
captured blocks have now been tested at every captured step and 49 is the only
unstable one, at all of them; every budget behaves the same way. Each control
cell carries its own plain arm, so a deterministic block is one where the lever
engaged and stayed stable rather than one where it admitted nothing.
**Its predecessor's "unobservable from outside the kernel" is half wrong and
open experiment 29 carries the correction**: the kernel's source says the
selection boundary is a histogram bin edge, not a rank cut, and that a token
clearing it takes a slot by `atomicAdd` and is dropped if the slot exceeds the
budget -- so the docstring's "the set never depends on scheduling" holds only
while the count clearing the threshold fits. Read entry 29 before proposing a
cause. **It is upstream's defect, measured and not merely argued**: it reproduces on a
plain upstream wheel holding none of our code, with a positive control on each
arm that it is the wheel it claims to be
(`data/results/2026-09-08_token_aug_stock_wheel_control.json`). A standalone
repro and the smallest input that still moves are in
`bench/repro_token_aug_nondeterminism.py` and
`data/results/2026-09-08_token_aug_repro_shapes.json`; reproduction is
non-monotone in head count and sequence length, so **a negative on one shape is
not evidence of correctness** and candidates must be tested in fresh processes.
**The cause is measured** (arm 5, which needed no kernel change after all --
the workspace and the count's offset are both reachable from Python): the
admitted token count exceeds the budget by orders of magnitude on a few hundred
centroid groups, the same groups every launch, and every group whose output
moves is one that overflowed
(`data/results/2026-09-08_token_aug_admitted_count.json`). Above the budget the
kernel's slot assignment is decided by atomic arrival order. Note the first
proposed fix is retracted in the entry: the overshoot is far too large for the
boundary-rounding explanation.

**How much of an H3 render is attention: a floor, derived rather than
profiled.** `bench/derive_attention_share.py` writes
`data/results/2026-09-08_attention_share_bound.json` from the 2026-09-03
ladder's paired arms. Read it before ranking any attention work, because it
carries two different floors and they answer different questions: the dense
one is what "attention dominates H3" means, and the sage one is the
denominator an Amdahl argument actually needs, since it is the configuration
that ships. The record states its own limits, including that both are floors
and neither bounds the share from above.

**Token routing: stages 1, 2 and 3 all done 2026-09-08.** The kernel is rebased onto v0.2.33 and installed
(`0.2.33+sol.990ae4c`), and `token_aug_blocks` exists on `MiniMaxH3SolAttn`
as a per-block knob that ships off in every graph. Neither touched the card.
Stage 2 reproduced: every aggregate not involving `token_aug` is bit-identical
to kijai's graded build, so the rebase changed no arithmetic. It also found
that the plan's stop condition was unachievable as written, because the
`token_aug` arms are nondeterministic run to run, and that the nondeterminism
sits on exactly the one block where the lever hurts. Both in
[`../research/2026-09-05_token_aug_plan.md`](../research/2026-09-05_token_aug_plan.md)
and its record. What has NOT happened is any render with it on; stage 4 is
the expensive one and needs the owner's go.

**Both upstream-survey checks landed 2026-09-04.** Token routing
(Comfy-Org/comfy-kitchen PR 156) graded on the Base16 cells:
[`../research/2026-09-04_sol_token_aug_grade.md`](../research/2026-09-04_sol_token_aug_grade.md)
owns it; better on four blocks, worse on block 49 at every step, budget
nearly inert, so it is a per-block candidate for the block-policy step and
not the frontier move roadmap step 7 asked about; nothing installed. The
post-pull core regime (the Comfy Compiler's malloc graph, on whenever aimdo
is) was checked on one dense rung
(`data/results/2026-09-04_stairwell_dense_retime.jsonl`): sampler time
inside the 2026-09-03 rows and the clip pixel-identical to the 2026-09-03
one by the scoring lane's comparison, so the 09-03 clips remain valid blind
references. `docs/sol_upstream.md` holds the kitchen state;
`docs/research/sglang_comparison.md` the rest of the survey.

**Small fixes the ladder exposed, before the next one.** The blind tool's
default output root landed 2026-09-04: `bench/_paths.py::comfy_output` reads
the launcher's `--output-directory` from the live server's command line and
refuses with no server and no `H3_COMFY_OUTPUT`, rather than answering with
the local directory. The fp16-sage rung is slower than Sol as shipped on every scene,
and a per-step timing would say where. Each is a checkable item in the
session postmortem under `internal/postmortems/` dated 2026-09-03. The
audio-spectrum grader's refusal below three clips per arm landed 2026-09-04
(changelog 0.99.40).

## Then, in the roadmap's order

The forward plan in [`../roadmap.md`](../roadmap.md) (section "Current
forward plan — 2026-09-03") owns the sequence, the decision standard, and
what would count as finding it. The PDD bake is a parallel lane there, not a
step behind the Sol work.

## Where the reasoning lives

- What was measured and what was withdrawn:
  [`../research/2026-09-03_sol_exact_pquant_and_base_capture.md`](../research/2026-09-03_sol_exact_pquant_and_base_capture.md)
  and [`../evidence.md`](../evidence.md).
- The goal in the owner's words: [`../../VISION.md`](../../VISION.md).
- How a rendered comparison is judged: [`../eval_comparison.md`](../eval_comparison.md).

## Updating this page

When a step above is done, replace it here and in the roadmap in the same
commit; a done step left standing is the drift this repo names most often.
