# Handover: read this before anything else

**Rewritten at the end of each working day by the Claude lead; the day before is in git.** Written for 2026-10-18 by
`mrted` (the Claude lead) at 22:05Z, while the card still ran; the Gemini lead `mrgemson` adds the close's state at the
end of "State when closed" and hands off to a new Gemini lead, who starts from this page. Where this page and a record
disagree, the record is right (`decisions.md`, newest first; `data/fun/laid.jsonl`; the newest gap check; `git log`).
Today's mailbox: `internal/gemini/2026-10-18_claude_to_gemini.md` (C1-C127) and `2026-10-18_gemini_to_claude.md`
(G1-G21); orders to the drawing builder: `internal/gemini/2026-10-18_mrted_to_gem-mouth.md` (M1-M21).

**Read first tonight's hand-run check**: [kitchen_24fps_check](../../internal/gemini/2026-10-18_kitchen_24fps_check.md),
every command run to fix the 24 fps bug and verify it on the kitchen (751-785), old against new. Hand-made for one
scene, not a recipe; only what it marks ALWAYS (24 fps end to end, derived data remade, the same tools for old and new)
holds everywhere. Tomorrow's facts pass (item 1c) automates it. The tracked write-up of the same check (what, how,
why, the scene's background, and what every scene should probably get): `bench/results/2026-10-18_kitchen_two_pass_validation.md`.

## State when closed

- **Kept and laid today** (the owner's keeps; `data/fun/laid.jsonl`, 38 lines): the mom's whole kitchen
  `second_hfp_0604h` (604-1009; recipe frozen in `bench/results/2026-10-18_second_hfp_0604h_recipe.md`, why it worked
  in `adf51858`'s postmortem), the lead's kitchen `lead_fp_0606`, the lead's hall `lead_fp_1306t`, the lead's street
  spin `lead_fp_0438e`.
- **Landed** (`git log --since=2026-10-18T04:15Z`): MoGe-2 depth for who is in front where tracks overlap, behind
  `--depth`, off by default (`1c213f41`, `3980ed5c`, `01b75141`); the tracker cuts frames that sit on another person
  before regain (10a, `f6281edc`, MASK_VERSION 15); the head tube no longer takes back off-head islands, the dad's bicep
  patch among them (`ffca8acd`); the pages rebuild themselves when `laid.jsonl` is written, with a stale-page banner
  (`6508f64e`); the reader's four-field card on Start here (`d3911162`); the mouth shape behind `--closed-lip-line`, off
  (`6ead729f`); the comfy-kitchen kernel fix for nanobind 2.10.x (`2e6f0d04`, `fd34e1aa`; upstream PR
  Comfy-Org/comfy-kitchen#256); the read-in order and one-lead rule (`bcf73037`).
- **Two root causes found, each with its control render** (`decisions.md`, 2026-10-18):
  1. The dad's glasses and grey beard in `third_fp_1306d`: his original face showed inside the window (under his fist
     on 1306-1309; the dropped 1409-1413), and H3 copies what the window holds. `third_fp_1340b`, the same reference on
     clean frames, came out 0/21 glasses and 0/21 beard. The floating contrast eye in profile then shaped the glasses.
  2. The mom's open mouth on 278-293: the drawn closed mouth sat 8-12 px above the grey head's own sculpted lips, which
     reads as parted lips. `second_hfp_0278f`, with no mouth drawn, kept her lips closed on all 16 frames.
  gem-dad measured the drift between the drawn points and the grey head on 45 references (3,503 frames;
  `internal/gemini/2026-10-18_feature_drift_finding.md`): the zoom, the camera focal and Meta's box enlargement each
  move it by 2 px or less. It is the grey head's generic face and pose, so the fix is in the drawing.
- **The drawing is the main line** (the owner, about 21:50-22:00Z; M10-M21): every mark clipped to the grey head's
  silhouette (a far eye shrinks into the edge, never switches off); the grey head a little darker with its shading
  kept; filled eyes; lines at today's width; no dots; no colour per part (the owner sided with gem-smear: cleaner, not
  pose-model colours); the mouth decided per WINDOW by voice, none when silent, drawn on every face frame when the
  person sings or talks, with the grey head's own lips covered and the mouth kept above its chin.
- **In flight at writing:** the lead's kitchen chin 751-785 rendered against the cut that has 0604h laid (C106; the
  smear was 0606 rendered beside the original mom's dark hair); gem-mouth's drawing build (the owner sees the reference
  sheet before any of its renders); `gem-review-kept-today` (C111, Kept today on the review page, old sections out);
  `third_fp_1306e` (1306-1327, the region grown over
  the beard under his fist) rendered before the owner's skip reached the card: a reader saw 0/22 glasses and 0/22 beard
  against 22/22 in 1306d (`data/2026-10-18_third_fp_1306e/playback_reading_gemread-0278h.md`); not laid, the owner's
  look not asked for.
- **Closed** (`docs/roadmap.md`, "Closed lanes"): Sapiens matting, SAM 3 base. Not again: a reference driven by core's
  SAM 3D Body face expression (broke the mouth, eyes and iris before).
- **The kitchen 604-1009 is locked as laid.** `lead_fp_0751` (751-785 against the cut, the one try, C131) was not kept
  (the owner, about 22:30Z): an open mouth from the old drawn mouth, and square blocks round her head where the mom's
  keep-out with depth carved whole cells out of the lead's region (C133).
- **Late night, as of 02:08Z on 2026-10-19** (mrted's last write; mrgemson completes it):
  - The 24 fps fix landed (`e44af676`): the cut is rebuilt at 24/1 with 6,395 frames and registered at rate 24.0;
    jobs refuse a source not at 24. gem-mouth's drawing landed (`5ebaecfe`, behind `--closed-lip-line` and
    `--darken-head`, off by default). The block measure is cleared (C168).
  - `lead_fp_0751d` (751-785 against the 24 fps cut): "not nearly as bad, still an outline" (the owner). The kept
    `lead_fp_0606` had the outline too. Cause found: Sapiens mislabels her face as clothing or background on 759-761,
    and the "SAM 3.1 head tube" never runs SAM in the capture (2b''); off_head adds a cut on 772 (2b').
  - In flight: `lead_fp_0606b` (the full kitchen window against the 24 fps cut, C178); gem-mouth building the real SAM
    3.1 head track (`internal/gemini/2026-10-18_brief_sam_head_track.md`, C190), then its mask test on 751-785 with no
    render; then the drawing tests (C166); then mrgemson's close (C134, C163).
  - If the Claude lead is out of credits, mrgemson is the one designated lead (`AGENTS.md`) and finishes this list,
    with a blind read by a session that did not build in place of mrted's clearance, and asks the owner where unsure.
- **The chin smear, corrected cause** (the owner on 760, about 03:10Z: the same dark smudge at the lead's chin in
  `0606` and `0606b`): the stack marks squares kept as the original at her chin in both runs, cut by the mom's keep-out
  (her whole-body track) and outside the mom's laid piece, so the cut shows the source's own pixels there (the lead's
  original chin against the original mom's dark hair). Rendering against the cut cannot change squares that are never
  regenerated; the earlier reading (H3 blending her chin into the dark hair, gem-smear) is not the whole story. Fix:
  the back pass keeps out the mom's laid matte, not her whole track (designed tomorrow), with the SAM 3.1 head track.
- **Dropped by the owner today:** the dad's beard work (the region grown over his beard, the full hall remake, the
  2050-2622 window test); wider lines; colour per part. The dad's hall stays as laid (`third_fp_1310c`, with its gaps).
- **Open List:** r18_05 (which kept pieces to remake with the old mouth) waits on the drawing test; r18_02 and r18_03
  are answered with conditions (below); r18_06 is tonight's state.

## What not to assume

- **A drawn mouth is not neutral.** Drawing a closed mouth opened hers in 0278d, 0278e and 0278h; taking the mouth out
  of the text alone (0278h) did not close it. The 24 older kept pieces drew no mouth and never had this.
- **A leak check that counts face labels only is blind.** A wrist in front of a chin hides the beard behind it (on
  1306-1309 most of the "hand" in the chin box is his wrist; only a 10-25 px blurred fringe of beard is mislabelled hand),
  and a turned beard is labelled hair (1408-1411); gem-track's shrinking-face measure (`pinched_head_face`) catches both.
- **Sapiens2 Pointmap is not worth adopting** (`internal/gemini/2026-10-18_pointmap_finding.md`, CPU, 1306-1310): 1B
  gives a flat sheet, 5B a soft edge at 18 times MoGe-2's time; MoGe-2 separates the fist from the face well, and no
  single depth map sees the beard behind a wrist.
- **A keep-out at a crossing carves whole latent cells out of the subject's own face**, and the original shows in
  squares (0751 at 772, C133). A keep-out holes a cell for its whole latent step (four frames; a window's first frame is
  a step of its own; `video_mask.py:70`, `:654`), so one frame's overlap costs four. Before any crossing load renders,
  count the cells the keep-out takes from her face, per step. But a cell the keep-out touches stays hers if her own mask
  has a pixel in it on any frame of the step (`video_mask.py:2000-2005`), so a carved square is one her own mask missed
  on all four frames. On 0751 at 772 the mom's keep-out gave back 0 cells and the load carried no keep, so the squares
  are NOT TRACED: read what the stack's yellow fill and `kept_px` count first (`bench/stack_delivery.py`, KEPT_INSIDE;
  C141), before building anything for crossings.
- **Sapiens' eyeglass class (2) was in the dad's face part on `1306d` and on the kept `1310c`**, and absent on the clean
  `1340b` and `1306e`, so it does not track his glasses. It is included only when Sapiens labels some of the original
  man's pixels eyeglass, so Sapiens saw something glasses-like on him in 1306-1418: if the original actor wears glasses,
  they leaked in with the beard. A cheap check if the hall is ever reopened: class 2's px per frame in his class map,
  and a reader on the source frames.
- **A dropped stretch is not the back of the head because a label says so**: 1412-1413 show the dad's bearded profile.
- **Old tracks predate today's tracker fixes.** A shot captured before today gets a fresh first preview, never a
  re-run on its saved .mkv (10a and regain run only in the preview).
- **The head tube's default needs one occluded shot.** The drawing tests are unoccluded faces and show nothing about
  it; one hand-or-mic-over-face window must read clean first: inside 2050-2622, 2295-2325 or 2335-2365 (gem-track; 516-603
  is not occluded in its table, and the whole of 2050-2622 is over the 362 cap).
- **Off by default until the owner says yes:** the head tube, `--closed-lip-line`, `--depth`, and every part of the
  drawing build. THE DEFAULTS THIS LOAD CARRIES in each job step is the observable.
- **The kernel venv still has nanobind 2.10.x.** The fork's `.none()` guards work around it; take 2.11 at the next
  kernel rebuild (`docs/comfy_notes.md`, "After a ComfyUI update").
- **`bench/load_chain.py:429-432` (`others_of`) adds the preview's other subjects as keep-outs whatever the job names**
  (mrgemson, G22): check the keep-outs a load actually carries.
- **Derived data is made again after any change to the code that makes it** (the owner, about 23:45Z): an `.npz`, a
  saved track, a class map or a depth file nobody reads in detail is never carried across a code change; the
  hand-bumped version stamps (`subject_track.py` MASK_VERSION, `video_mask.py` REGION_VERSION and others) rely on
  someone remembering. A durable form is tomorrow's item 2c.
- **Times from `date -u`.** The Claude lead wrote five stamps ahead of the clock today; each was corrected.

## Tomorrow, in order

1. Read in by `AGENTS.md`'s order. The new Gemini lead starts on the new mailbox tool, `internal/comms/`
   (`README.md`, `post.py`, `wait.py`); today's files stay as the record.
1b. Everything H3 gets is 24 fps (the owner, about 23:58Z). The joined cut is written at the original's 23.976
   (`data/fun/manifest.json`, rate 23.976024), the loader forces no rate (`h3_config.MASKED_JOB_FORCE_RATE` 0.0) and
   `workflows/masked_job_graph.py::seek` assumes 24, so past about frame 600 a load from the cut starts a frame early
   (lead_fp_0751 loaded 750 for 751; mrgemson, G29). Only lead_fp_0751 used it; every kept piece used a 24 fps copy.
   The owner's rule: 24 fps from input to output, and a cut is loaded exactly the way the real input is. The input's
   way is `bench/clip_analysis.py init --copy`: an every-frame copy at the model's rate (each frame kept, retimed, not
   resampled), identified and checked frame for frame, recorded in the manifest, loaded by the same loader settings.
   So before any pass is rendered against the cut: the assembler writes the cut at 24 fps as that same kind of
   every-frame copy and registers it the same way, and the job step refuses a source whose manifest rate is not 24,
   naming the 24 fps copy. Then check the owner's question: two joined clips in
   one shot that slip partway through. Every place a frame number becomes a time on a file not at 24 is a suspect
   (`bench/stack_delivery.py::frame_of`, `bench/assemble_delivery.py::audio_input`, the review clips' encodes); one
   joined clip read frame by frame against its pieces settles it.
1c. Facts beside every review clip, in the JSON sidecars that already sit beside each one (`<clip>.mp4.json`, the
   provenance; `<clip>.stack.json`; `<clip>.stack.frames.jsonl`, per frame: `changed_px`, `diff_vs_current`, each
   column's kept squares and keep-outs; `<clip>.alignment.txt`), and beside each render on the share
   (`Video/fun/<session>/<load>/<load>/<load>_00001.matte_reconstructed.json` beside every laid piece, `_zoomed.json`
   where zoomed): add the missing measures as fields there, or as one more sidecar of the same kind beside the render,
   not a new place. (the owner, about 00:20Z: "how else would you know what you're looking at, with
   real data across modalities?"). One pass, run on every clip that goes to Start here, writing a short per-frame table
   the reader reads BEFORE watching and the card cites: the block measure (`bench/measure_blocks.py`, C158), joins
   (`d002`), the render's face points against the source's (the land step's pose pass), the mouth opening per frame in
   the render against the source (Sapiens lip points on both), the voice per frame (`voice_per_frame.csv`) beside it,
   the gaps (`bench/frame_coverage.py`) and who is in front (the depth record). First version gathers what exists plus
   the mouth comparison; flicker and the eyes' offset come after. Pixel diffs too (the owner): the render against its
   source outside the region (near zero when the frames line up), a candidate cut against the current one, and frame
   to frame inside the region. No verdicts: figures with what they were measured on.
2. The owner's looks, batched on Start here: 751-785 against the cut, and the drawing build's reference sheet if it
   did not reach them tonight, then its renders: the mom's 278-293 (silent, no mouth) and the dad's 1340-1360 twice
   (no mouth against the repaired mouth), each against today's drawing, same seed. If the two 1340 arms read equal, the
   same pair on 1156-1182 before the mouth leaves singing windows (M19).
2a. Read gem-genealogy's record (`internal/gemini/2026-10-18_mask_genealogy.md`, started tonight at the owner's ask):
   every change to how masks are made and handled since 2026-09-12, which layers later fixes made redundant, and the
   five leftovers most likely costing us now. It is the frame for 2b.
2b. The buffers, before any crossing load (the owner, about 23:10Z: "maybe we're adding buffer to things we shouldn't
   be because we did it initially when we didn't solve it other ways"). An audit, reading only: every grow, margin,
   feather and whole-token rounding in the chain, its value, its provenance, what it was covering for and whether that
   is fixed now (`video_mask.py::GROW_PIXELS` 64, the owner's pick on one clip and one seed on 2026-10-04;
   `h3_config.MASKED_LOAD_MARGIN` and `MASKED_LOAD_DEFAULTS["feather"]`, inherited and never measured). Then one
   crossing shot at today's margins against the smallest the audit supports, same seed. On 0751 the grown region made
   the 5,810 px overlap that forced the keep-out whose carve showed as squares (C133, C137).
   The proof the lens pays: what fixed the mom's kitchen (0604h) was removing four guardrails added for earlier failures
   (assembly-time restore of her own parts, the largest-piece rule, "only what changed", the self keep-outs), each of
   which had been punching holes in her (`docs/research/postmortems/2026-10-18_mom_kitchen_flicker.md`, 25-28).
   Two-person shots (worked through at about 23:45Z, C145-C146): the subject's own mask already wins every cell it
   touches (`video_mask.py:2005`), so a keep-out only ever acts on MARGIN cells. On 0604h at 772 the lead stood beside
   the mom, and the 106 cells it gave back were the mom's 64 px margin reaching onto the lead's visible body: the keep-out
   protecting the lead, rightly. So a depth rule for keep-outs is not the fix (it was designed and dropped); the margin
   is (this item), plus, for the pass behind, keeping out the front person's laid matte rather than their whole track
   (design it tomorrow).
   The simple first test, no code: the per-load `margin` already exists (`workflows/masked_job_graph.py:739`, into the
   node's `grow_pixels`); render one two-person shot with it lowered against today's, same seed. Next step if that
   helps: the node's size-based margin ("the subject's size", built 2026-10-07, never tried; the chain passes only the
   pixel margin, so it needs one setting wired through). Neither explains 0751's squares, which had a 16 px margin.
   A candidate after 2b, not before (the owner asked about noise, about 23:35Z): a band of partly noised cells at the
   region's edge, so H3 blends a kept square into a regenerated one instead of butting them (the hard joins on 0751).
   It changes how the sampler reads the mask (a runtime patch); removing excess margin may shrink the problem first.
2b'. Her face cut out by off_head (the owner on lead_fp_0751d, about 01:20Z: "are we segmenting out part of her
   again?"): `parts_dropped_off_her_head` took 14,577 px over 26 frames, 8,559 on 772, as "no face point near, outside
   her head box" (`bench/capture_masked_run.py` off_head, a box round the pose face points widened 16 px). Today's
   island fix (`ffca8acd`) stops the head tube taking such pixels back, so a wrong cut now sticks. Check the pieces on
   762, 768, 772 against the frame; then gate off_head by the SAM 3.1 head tube instead of the face-point box
   (gem-genealogy's Rank 3), with a case on 772. The owner's hunch: masks were better before today, the Sapiens labels
   or SAM 3.1. What changed today: memory selection on (`35706971`), 10a (`f6281edc`), depth (`1c213f41`) and the
   island fix (`ffca8acd`); the labels did not. First, CPU only: lead_fp_0751d's capture files step at `ffca8acd`'s
   parent against main, from the same saved preview, the face part per frame (C175); then memory selection off.
   The clips the owner judged it on (the share's `output/`): `Video/fun/mrgemson/lead_fp_0751d/lead_fp_0751d/
   lead_fp_0751d_00001_with_mask.mp4` and `.../lead_fp_0751d_windows/lead_fp_0751d_window_1_with_mask.mp4` (red: the
   face part; cyan: what else regenerates), and the plain window `.../lead_fp_0751d_windows/lead_fp_0751d_window_1.mp4`.
   All four render clips probe 24/1 with 35 frames, and so does the review stack
   (`Video/fun/review_clips/lead_fp_0751d_0751_0785_stacked.mp4`; its `.1fps.` copy is the one-frame-a-second reading
   copy, 1/1). The masked clip's top half matches the plain window within encoding (mean 0.7-0.9 grey levels, at most
   13, on frames 5, 21 and 30). Beside the window clip, read by `ls` and the files' own keys:
   - `lead_fp_0751d_window_1_region.npz`: `mask` (39 frames, 768 by 1024 bit-packed to 128 bytes: the subject's part
     mask), `tokens` (12 latent steps by 48 by 64 cells: what was regenerated, 3.6% of the window), `margin` (16 px),
     `matte` (39 by 768 by 1024, 0-255: the weight the render was blended with, every frame), `kept` (packed like the
     mask: what the composite kept of the render), and `meta` (version 2; grow 16 px "a fixed margin", edge "latent
     cells", feather 8, composite "whole region", 35 frames written). Reading `mask` against `tokens` per frame shows
     where the face part ends and the margin begins, which is the owner's red-against-blue.
   - `lead_fp_0751d_window_1.safetensors`: the window's latent kept for a window that continues from it (metadata
     `nested`, `written` 35, `next_start` 1.625 s): `stream_0` [1, 24, 12, 48, 64], the video latent (12 latent steps at
     48 by 64), and `stream_1` [1, 32, 2, 65], the audio latent. Not a mask. Two readers: `bench/recomposite_piece.py`
     and `bench/recomposite_window.py` decode it with the render's own VAE to lay the window again with one setting
     changed or to save the piece's own matte (how every laid piece's matte was made for the one compositing law), and
     a later load's `continue_from` meets it at a seam. Nothing reads it for viewing; keep it beside any piece that
     might be laid.
   What makes them, end to end (read from the code's own docstrings; `audio_freeze_song.py` "Files and resume"):
   1. The capture (`bench/shot_to_load.py` preview and job, `bench/capture_masked_run.py`): her track, her part mask
      (Sapiens classes, then off_head, the head tube, depth), saved as 24 fps mask videos under `data/<date>_<load>/`.
   2. The reference (`bench/head_reference.py`) and the text (`bench/load_text.py`).
   3. The render (`bench/load_chain.py render`, graph from `workflows/masked_job_graph.py`): loaders on the 24 fps
      source, the masks and the reference; `MiniMaxH3MaskedSource` (`video_mask.py`) fits the mask, grows the margin,
      makes the token region and applies keep-outs; `MiniMaxH3AudioFreezeSong` (`audio_freeze_song.py`) samples each
      window and writes, in `<prefix>_windows/`, the window's video and its latent (`loop_output.py`), the mask review
      `..._with_mask.mp4` (the render over `video_mask.overlay_pieces`) and `..._region.npz`
      (`video_mask.save_window_region`); then joins the windows into `<prefix>_00001.mp4` and `_with_mask.mp4` at 24
      fps, with the first frame as a PNG carrying the prompt and the graph.
   4. The land step (pose pass, record), `bench/stack_delivery.py` (the review stack), `bench/assemble_delivery.py` (a
      candidate cut) and `bench/rebuild_delivery.py` (the cut and the pages).
   Where the owner's red-against-blue comes from is step 1 (the part mask) and step 3's margin; tomorrow reads both.
2b''. Her face cut by the Sapiens labels (mrgemson's trace on lead_fp_0751d, about 01:35Z; the same labels made the
   kept lead_fp_0606, whose outline the owner then saw too). SAM 3.1's track holds her whole head on 759-762 (about
   20,600-22,500 px), but inside it Sapiens2-Seg calls her head and left cheek Upper_Clothing (13,948 px on 759, 16,899
   on 760), a notch down her forehead Hair, and on 761 20,133 px Background, leaving a 2,037 px crescent as her face.
   The head tube did not heal it: it smooths the class map over 3 frames at fixed pixels while her head moves about 18
   px a frame, so a 3-frame mislabel never reaches its floor (`bench/head_tube_parts.py`, `part_prob_floor` 0.35).
   Candidates, smallest first: inside the SAM 3.1 head track, let Sapiens decide only face against hair, never face
   against clothing or background; or smooth along the head's motion, not at fixed pixels. Test on 759-761 and 772.
   Found after (C185): in the capture the "SAM 3.1 head tube" never runs SAM 3.1. `head_tube_for_capture`
   (`bench/head_tube_parts.py:1488`) passes no tracker, so the tube is Sapiens' head classes bounded by the person track
   and smoothed; SAM 3.1 decides only the person (`subject_phrase` "person"). The fix is to give it a real SAM 3.1
   tracker prompted "head", so SAM decides the head and Sapiens only face against hair.
   The owner's principle behind it: what we track and where we let H3 paint should be the same thing. Today "where her
   head is" has five answers (SAM 3.1's person track, Sapiens' face class, Sapiens' pose points, the SAM 3D Body mesh,
   the head tube from Sapiens' classes), and each disagreement is a fault seen this week (off_head on 772, the drawing's
   drift off the mesh, the region missing face on 759-761). Aim: one answer, the SAM 3.1 head track, read by the
   region, off_head, the reference crop and the drawing (C187; the roles audit, C182-C184).
   With it, the owner's proposal (about 02:00Z): one SAM 3.1 and one Sapiens pass per subject per load, made fresh,
   that every step reads (region, off_head, reference crop, drawing, the grey head's crop), in place of passes made at
   different times on different spans and crops. Run over the whole shot that holds the window, not the window alone
   (SAM 3.1's memory needs the shot's run-up). The audit's pass table (C188) gives the cost; the owner decides.
2c. Derived data that goes stale silently (the owner, about 23:45Z): stamp each derived file (the capture's `.npz`
   files, saved tracks, class maps, depth) with a hash of the code that wrote it, and have its readers refuse a file
   whose writer has changed since, saying which command makes it again. Design and build on a branch; it replaces
   remembering to bump a version by hand.
2d. Old masks against new, a measurement (the owner, about 00:15Z): re-capture one or two kept shots with today's
   tools (the head tube, memory selection, 10a, the island fix, at 24 fps) while the card is idle, and compare the new
   masks with the saved ones: the face region, the holds, the keep-outs, frame by frame. No render, nothing remade.
   It says which kept pieces a remake could improve, if the owner ever wants one; regenerating every mask was weighed
   and not chosen (the existing masks were made from the 24 fps copy, and kept pieces are not re-rendered).
2e. Bring today's measuring scripts into bench/ with checks (root `AGENTS.md`: a tool worth keeping is tracked; the
   owner, about 00:40Z: nothing valuable left in a worktree or a scratch folder). Tonight each was committed on a branch
   of its own (`h3_worktrees/<worker>-scripts`; mrgemson's End of day lists them, C163): gem-dad's drift and box
   scripts, gem-track's Pointmap and next-shots scripts, gem-mouth's sheet and brightness script, and the kitchen
   check's scripts. The drawing tests and the old-against-new comparisons (2d) then read them.
3. Defaults, each a row on the Open List for the owner's yes: the new drawing and the per-window mouth (replaces
   r18_03's rule) once the tests read; the head tube (r18_02) after one occluded window (inside 2050-2622); depth after one more real load.
4. r18_05: remake the kept pieces with the old mouth, once the drawing is proven.
4b. Sapiens pose's mirror averaging (`flip_test=True` in Meta's own config; `bench/pose_along_shot.py` runs one pass):
   build it behind a switch, measure the points' frame-to-frame jitter on one profile turn (1306-1418 or 1093-1133)
   before and after, and put the default to the owner. It steadies profile points; it does not touch the grey head's
   8-13 px offset (gem-track, 2026-10-18).
5. The second-half push, worst first (`internal/gemini/2026-10-18_next_shots_track_check.md`): fresh first previews
   for every shot from an old capture; the lost tracks (the dad 1632-1729, the mom 1437-1452); the shots that need depth
   (1466-1729, 4647-4725, 1419-1465); 2050-2622; then the uncaptured shots in song order (the newest gap check).

## Where things are

- Decisions and why: `docs/wiki/decisions.md`, 2026-10-18. The owner's preferences: `docs/wiki/working_with_the_owner.md`.
- The cut and gaps: `data/fun/laid.jsonl`; the newest gap check under `data/2026-10-18_fun_delivery_tables/`
  (`frame_coverage_*`); verdicts `data/fun/verdicts/verdicts.jsonl`; the owner's pages `Video/fun/TO_REVIEW.html` and
  `PROVENANCE.html` on the share, rebuilt by `bench/rebuild_delivery.py`.
- **Each session's notes today** (under `internal/`, not tracked; the links work on this machine):
  - the Gemini lead, mrgemson (G entries; the last is "End of day"): [gemini_to_claude](../../internal/gemini/2026-10-18_gemini_to_claude.md)
  - gem-mouth: the drawing build, both mouth findings, the kept-piece audit: [worker_gem-mouth](../../internal/gemini/2026-10-18_worker_gem-mouth.md), [worker_gem-mouth-b](../../internal/gemini/2026-10-18_worker_gem-mouth-b.md), [mouth_finding](../../internal/gemini/2026-10-18_mouth_finding.md), [mouth_0278e_finding](../../internal/gemini/2026-10-18_mouth_0278e_finding.md), [mouth_kept_audit](../../internal/gemini/2026-10-18_mouth_kept_audit.md)
  - gem-dad: the dad's glasses, the drift measurement: [worker_gem-dad](../../internal/gemini/2026-10-18_worker_gem-dad.md), [dad_glasses_finding](../../internal/gemini/2026-10-18_dad_glasses_finding.md), [feature_drift_finding](../../internal/gemini/2026-10-18_feature_drift_finding.md), [notes_for_gem-smear](../../internal/gemini/2026-10-18_notes_for_gem-smear.md)
  - gem-smear: the kitchen smear, the window options: [worker_gem-smear](../../internal/gemini/2026-10-18_worker_gem-smear.md), [window_leaks_options](../../internal/gemini/2026-10-18_window_leaks_options.md)
  - gem-sam: what our code draws and where: [worker_gem-sam](../../internal/gemini/2026-10-18_worker_gem-sam.md), [drawing_map](../../internal/gemini/2026-10-18_drawing_map.md)
  - gem-h3ref: what H3 reads in the reference: [worker_gem-h3ref](../../internal/gemini/2026-10-18_worker_gem-h3ref.md), [open_reference_view](../../internal/gemini/2026-10-18_open_reference_view.md)
  - gem-track: tracker fix 10a, the next shots, Pointmap, the Sapiens audit: [worker_gem-track](../../internal/gemini/2026-10-18_worker_gem-track.md), [next_shots_track_check](../../internal/gemini/2026-10-18_next_shots_track_check.md), [pointmap_finding](../../internal/gemini/2026-10-18_pointmap_finding.md)
  - gem-depth: MoGe-2 in the chain: [worker_gem-depth](../../internal/gemini/2026-10-18_worker_gem-depth.md), [moge2_integration_design](../../internal/gemini/2026-10-18_moge2_integration_design.md), [depth_crossings_map](../../internal/gemini/2026-10-18_depth_crossings_map.md)
  - gem-env: the ComfyUI update and the kernel: [worker_gem-env](../../internal/gemini/2026-10-18_worker_gem-env.md), [env_finding](../../internal/gemini/2026-10-18_env_finding.md)
  - gem-postmortem: why 0604h worked: [worker_gem-postmortem](../../internal/gemini/2026-10-18_worker_gem-postmortem.md)
  - the builders: the head-tube fix, the pages, the reader card, Kept today, the new mailbox tool, a repro: [worker_gem-headtube-island](../../internal/gemini/2026-10-18_worker_gem-headtube-island.md), [worker_gem-pages](../../internal/gemini/2026-10-18_worker_gem-pages.md), [worker_gem-review-card](../../internal/gemini/2026-10-18_worker_gem-review-card.md), [worker_gem-review-kept-today](../../internal/gemini/2026-10-18_worker_gem-review-kept-today.md), [worker_gem-comms](../../internal/gemini/2026-10-18_worker_gem-comms.md), [worker_gem-repro](../../internal/gemini/2026-10-18_worker_gem-repro.md)
  - the Claude lead's orders: to mrgemson, and to gem-mouth: [claude_to_gemini](../../internal/gemini/2026-10-18_claude_to_gemini.md), [mrted_to_gem-mouth](../../internal/gemini/2026-10-18_mrted_to_gem-mouth.md)
  - **read first: the day's synthesis across every session** (findings that held, hops that did and did not happen,
    contradictions, patterns): [synthesis](../../internal/claude/2026-10-18_mrted/synthesis.md)
  - the Claude lead, mrted: [mentor_notes](../../internal/claude/2026-10-18_mrted/mentor_notes.md), [handover_notes](../../internal/claude/2026-10-18_mrted/handover_notes.md), [built_2026-10-17](../../internal/claude/2026-10-18_mrted/built_2026-10-17.md)
- Commands: `docs/wiki/runbook.md`. The lane: `data/AGENTS.md`, then `data/fun/AGENTS.md` (local).
