# Verifying a Load Against Its Artifacts (`docs/masked_v2v/verifying_raw_frames_and_keeps.md`)

A capture's `.npz` and `.json` files are the chain's cache; **the node's own inputs and writes on disk are the artifact**. Whenever a record's region, zoom box, reference crop, or laid state is in doubt, verify the artifact directly by the procedure below, in order. Each step names the artifact on disk, not the summary record.

Related rules and records:
- Root [`AGENTS.md`](../../AGENTS.md) ("Do not trust prose, including this file's")
- [`docs/masked_v2v/AGENTS.md`](AGENTS.md) (§8, "What a change forces: the chain in trace order")
- [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
- Case record from `2026-10-25`: [`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)
- Multi-pass contamination & mask-step forensics (`2026-10-25`): [`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) (`bench/inspect_mask_steps.py`)

---

## 1. What the Node Was Handed (The Queued Graph & Input `.mkv` Files)

The queued graph (`<job_folder>/<load>_render_api.json` or `<load>_plan_api.json`) and its queue record (`<job_folder>/<load>.render.queued.json`, key `digests`) name the exact `.mkv` mask files and `.mov`/`.mp4` videos handed to the loader nodes on the server.

1. Read `digests` in `<job_folder>/<load>.render.queued.json` (or `<load>.plan.queued.json`) to get the exact file paths and `sha256` hashes the server read (`bench/what_h3_got.py` verifies those hashes). At the plan stage it also checks each mask file's digest against the `<capture>_first` copy and names the frames the on-plan rebuild rewrote in place (`first_copy`, `frames_as_planned`), and since `ac803f92` (2026-10-26) it joins the capture's own records to the graph. A render's prompt is read in full from this report, never from a summary (root [`AGENTS.md`](../../AGENTS.md)).
2. Decode the `.mkv` mask files (`subjects/<subject>/parts_held__<preview>.mkv`, `track__<preview>.mkv`, and each keep-out `subjects/<other>/track__<preview>.mkv`) directly with `ffmpeg` or `cv2.VideoCapture` and count white pixels (`> 127`) per frame.
3. Compare those raw per-frame pixel counts and bounding boxes (`[x0, y0, x1, y1]`) against `subjects/<subject>/per_frame.json` (`parts_share`, `parts_box`, `track_share`, `track_box`). **Always check the `y` range `[y0, y1]` (and draw a visual crop strip) rather than trusting a non-zero pixel count alone** (seen on `shot_c_4647k` `4682–4685`, `2026-10-26`: `second_parts_00001.mkv` held `608–2,123 px` on `4682–4685`, which looked like her emerging head in a count table, but sat at `y=289–367` on her bare forearm crop while her real head emerged `40 px` higher at `y=258–316` on `4686`; [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.9).
4. Two strangers files share a name and differ (2026-10-26): the preview's `<preview>_strangers.mkv` on the share is uncarved and is what `her_hair_in_hole` reads (`as_read` in `bench/capture_masked_run.py`, near line 2567), while the capture's `subjects/strangers/track__<preview>.mkv` is carved against her track (`carve_keep_out`, near line 2575) and is what node `104` reads. Measuring the fill's room on the carved copy gave a few pixels on `4926` where the uncarved file held thousands; name which file a count was read from.
   - *Example (`2050e` / `2050g`)*: Decoding the four `.mkv` mask videos matched the carried record counts to within a pixel on frames `0`, `22`, and `44`, and scanning all `243` frames of `parts_held__shot_c_2050f.mkv` confirmed `min_x = 256` (frame `2272`) and `max_x = 790` (frame `2245`).

---

## 2. What the Node Did (The Node's Own Mask-Review Video & Raw Zoomed Canvas)

Beside every render `<load>_00001.mp4`, the node (`video_mask.py`, `audio_freeze_song.py`) writes its own diagnostic videos on the share under `Video/<video>/<session>/<load>/<load>/`:
- `<load>_mask_00001.mp4`: the node's own mask-review overlay (what part was taken and where the grown margin sits).
- `<load>_00001_zoomed.mp4`: the raw zoomed canvas before `video_mask.lay_back` composites the hole back onto the full frame (`bench/assemble_delivery.py` reads this companion at `--size source`).
- `<load>_windows/<load>_window_N_region.npz`: the node's saved record of window `N` (`video_mask.py::save_window_region`): `tokens` (`[steps, 48, 64]`, the token grid per latent step), `mask` (per frame, bit-packed along the width: `np.unpackbits(z["mask"], axis=-1)` gives `[frames, 768, 1024]`), `matte` (per frame, `uint8` 0–255 over the whole canvas: the weight the composite laid), `kept` (bit-packed like `mask`: where the composite's weight was over one half), `margin`, and `meta` (a JSON string with the source, zoom, padding and `cuts`).

**The composite rule, and how to judge a mask with this file** (read from the node and from `second_hfp_4912g`'s window 1 on 2026-10-26): a latent step's region is the maximum of its four frames' masks (`video_mask.py::token_mask`); the model regenerates it for all four; the composite lays it on every frame of the step whose own mask is non-empty and leaves a frame with no mask as the source (`video_mask.py::unlaid_frames`, `pixel_alpha`). So one white pixel on a frame lays the whole step's region there (`4926`: 5 mask pixels, the same matte as `4927` and `4928`), and an empty frame lays nothing (`4925`). A mask is therefore judged against `matte` (after a render) or the blue region on the inspector's strip (before one), never against the green part on a check strip: a miss matters only where visible hair or face falls outside the matte, and a frame with no matte at all is the one that needs a `head_marks` entry (the owner's rule, 2026-10-26; [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.4). To see it, draw `matte > 0` over the canvas-sized source frames of the span.

To verify what the node actually did on a frame:
1. Pull the same frame from the source working copy, `<load>_mask_00001.mp4`, `<load>_00001_zoomed.mp4`, and `<load>_00001.mp4` side by side.
2. Check `<load>_mask_00001.mp4` to confirm the hole and margin cover the intended part (and do not stop short of the hairline or jaw).
3. Check `<load>_00001_zoomed.mp4` to see whether the DiT regenerated the new subject inside the zoomed canvas or whether the composite (`<load>_00001.mp4`) laid the source back over it.
   - *Example (`2050e`)*: Frame `22` showed the hole open on her face in `<load>_mask_00001.mp4` and a regenerated face in `<load>_00001_zoomed.mp4`.

---

## 3. What the Node Says It Did (The Queue Record's `said` Report & `<load>_plan.json`)

After execution, `<job_folder>/<load>.render.queued.json` (and `<load>.plan.queued.json`) records the node's own execution report under `said`, alongside `<load>_windows/<load>_plan.json`:
- `window_frames`, `context_frames`, `pad_before`, `pad_after`, and `source_frames_held` (`context_before_held`, `context_after_held`, `held_fell_back`).
- `zoom`: `box` (`[x0, y0, x1, y1]` in even `1024×768` canvas pixels), `scale`, `set_by`, and `up_to`.
- `regenerating_share`, `frames_with_the_mask`, `frames_with_no_mask`, and `frames_left_as_source`.

Read `said` and `<load>_plan.json` in full rather than trusting an upstream summary.
- *Example (`2050e`)*: `45` frames padded after, `0` held, `1.86x` zoom in `[210, 180, 762, 594]`, `8.2%` regenerated, frames `45` and `46` (`2095–2096`) left as source.

---

## 4. Where the Box Sits on the Film (Raw Working-Copy Frames + `.mkv` Mask + Zoom Box)

Two coordinate and union-box rules explain why a box in `<load>_plan.json` or `<subject>_head_reference_<first>_<last>.json` has the size it does:

1. **Three coordinate spaces (`1424×1080` source vs. `1024×768` canvas vs. `64×48` zoomed token grid)**:
   - The working copy (`fun_0000_6394_24p_sound_in_step.mov`) is `1424×1080`; `workflows/masked_job_graph.py::load` scales and center-crops it to the `1024×768` canvas (`video_mask.py::_fit_crop`).
   - Every `.mkv` mask (`parts_held__<preview>.mkv`, `track__<preview>.mkv`) and `zoom.box` (`[x0, y0, x1, y1]`) is on the unzoomed `1024×768` canvas.
   - `<load>_window_N_region.npz` (`tokens`, shape `[steps, 48, 64]`) is in `16×16` cells of the **zoomed** `box`, laid back onto the canvas by `video_mask.py::lay_back`.
   - Always resize raw working-copy frames to `1024×768` before drawing `.mkv` masks or `zoom.box` over them.
2. **One fixed union box across the span (`video_mask.py::zoom_box` and `bench/crop_reference_to_head.py::cranium_box`)**:
   - `video_mask.py::zoom_box` (`:1887-1935`) computes **one fixed box** over all frames of the window (`rows = boxes[boxes[:, 0] >= 0]`) with `ZOOM_ROOM` (`0.20`) around it ("a box that followed the subject would turn the subject's travel into the background's", `video_mask.py:1899-1900`).
   - Likewise, `bench/crop_reference_to_head.py::cranium_box` (`:239-303`) computes **one fixed box** over all frames in the reference span where `grey_head` is non-empty (`:263-265`), and in `bench/head_reference.py::run` (`:877-900`) `crop.run` cuts the plain 3D body-model video (`1 of 3`) **before** `bench/draw_face_points.py::run` (`3 of 3`, `:1054-1062`) applies `--no-head-on`.
   - To check whether a `zoom.box` is caused by true subject travel or by a stray mask piece, decode the raw frames from the working copy at the `min_x`, `max_x`, `min_y`, `max_y` frames, draw `parts_held__<preview>.mkv` and `zoom.box` over them, and look.
   - *Example (`2050f` vs. `2050g`)*: Overlaying both boxes on raw frames `2050, 2100, 2190, 2232, 2245, 2272` showed that the `2050f` 90-frame box (`[210, 180, 762, 594]`, `1.86x`) sliced through the middle of the singer's face on frame `2245` (`x1 = 790` at the right rim of the spinning mirror), whereas `2050g`'s 243-frame box (`[120, 100, 928, 706]`, `1.27x`) held her full sweep from the left rim (`x0 = 256` on `2272`) to the right rim (`x1 = 790` on `2245`).

---

## 5. The Take's Rule

When writing the four-line take for a render (`PROTOCOL.md` §4.1), its fifth line (what the node actually did on the frames in question) comes from **Steps 2 and 3** (`<load>_mask_00001.mp4`, `<load>_00001_zoomed.mp4`, and the queue record's `said`), never from the `.npz` region file alone.

---

## 6. Verifying Every Owner Keep & Multi-Subject Composite in `data/<video>/laid.jsonl` (`bench/laid.py verify` & `keeps`)

Before running `bench/rebuild_delivery.py --video <video> --clip`, verify that every kept clip and multi-subject composite stands in `data/<video>/laid.jsonl`:

1. **`bench/laid.py verify`**: Checks that every row in `data/<video>/laid.jsonl` exists on the share (`--share`) and matches its recorded `sha256` and `zoomed_sha256`.
2. **`bench/laid.py keeps`**: Cross-checks every owner choice in `data/<video>/owner_answers.jsonl` and `data/<video>/verdicts/verdicts.jsonl` against `data/<video>/laid.jsonl`:
   - Keys coverage on `(subject, frames)` (`[[f0, f1], ...]`), **not** on `(shot, subject)` alone, so split-span shots with two pieces of the same subject on disjoint frame spans (`1466b` + `1575b` on `1466–1729`, `2823b` + `3098b` on `2823–3166`) pass without false conflict.
   - Follows `replaces` and `laid.removed.jsonl` chains and keep `at` timestamps so earlier kept takes covered by a newer kept row are reported as `superseded by <newer_load>`, not missing.
   - Hard-refuses (`exit 1`) only on the two genuine failure cases:
     1. **`KEEP_NOT_LAID`**: A kept render with no laid row covering its `(subject, frames)` span (and not superseded by a newer keep or recorded removal).
     2. **`LAID_MARKED_WRONG`**: A currently laid row whose load the owner marked wrong (`NOT_THE_ONE`) after the keep/lay.
3. **Matte-based compositing & `restore=` tokens (`bench/assemble_delivery.py::laid_by_of`)**:
   - Confirm in `<clip>.mp4.check.json` and `frame_coverage_<video>_<first>_<last>_<stamp>.md` that every laid piece resolves to `rule: "matte"` (`*_window_*_region.npz`) and every multi-subject shot carries its `restore=<other>` tokens.

---

## 7. Does a Pass-1 Cut Alter the Next Subject? (Compare the Cut Against the Film Inside Her Track)

Before any claim that an earlier pass regenerated, erased or recolored a subject who was not replaced, measure it on the
pixels (added 2026-10-26, after the `4647–4725` claim of 2026-10-25 was found to be a crop holding the other subject's
head; [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.1):

1. Decode the same frames from the raw working copy and from the cut (`<share>/input/.../fun_0000_6394_joined_<stamp>_from_laid.mp4`), and fit both to the `1024x768` canvas as `workflows/masked_job_graph.py::load` does (`video_mask.py::_fit_crop`), so the masks line up.
2. Take her person track from the preview made on the raw film (`<share>/output/Video/<video>/<session>/shot_c_<first><letter>/shot_c_<first><letter>_<subject>_track_00001.mkv`, white `> 127`).
3. Count, per frame, the pixels whose largest channel difference exceeds a small threshold (8 levels separates re-encode grain from a change) in three places: inside her track, in a ring 16 px around it, and over the whole frame. Draw the whole-frame difference once and look at where it lies.
4. Read the earlier pass's `matte` from its window record (§2) and intersect it with her track: an empty intersection means the pass never laid onto her.

What the three counts say: a change inside the track is a change to her; a change in the ring only is the other subject's edge or the margin; a whole-frame change with neither is somebody else. On `4647–4725` the cut's change on her was in single digits on every frame and the whole-frame change was the third subject's face and beard.

---

## 8. Why Did a Track Collapse on the Cut's Preview? (Two Previews' Tracks, and the Tracker's Own Log)

When a subject's track on the pass-2 preview differs from the raw preview's, the pixels are only one of two candidates; the other is the seeds (added 2026-10-26, after the `4697` collapse was traced to one hand-drawn box; [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.2):

1. Decode the two track files (§1, white `> 127`) and print, per frame, the pixel count and the bounding box of each. A collapse with a box that jumps to another place on the canvas is another person, not a shrunken her.
2. Read the tracker's own report, `said["<node id>"]` for the subject's `MiniMaxH3SubjectTrack` node in each preview's `preview.queued.json` (node `1105` on the `4647` previews). It lists every seed by load frame (`load frame = source frame - shot.first`): `seeded on load frame N: took person K ... tracked over A-B and kept on C-D`, `SEED REFUSED on load frame N ...` with the share inside the box it measured, and `THE BOXES' TRACKS HELD DIFFERENT PEOPLE on ...: left empty there` for the frames the conflict rule emptied. Compare the two previews line by line; a box refused on one and taken on the other, by a share near the 0.50 line, is the cause.
3. The same facts are in the preview's shot table on the share (`<preview>_<subject>_shots_00001_shots.json`, written by `shot_table.py`): `seeds` lists each box with `inside` (the share of the person's mask inside it), `covers` (the share of the box it covers), `drawn` and `kept_on`, and `seed_tracks_disagree_on` lists the frames the conflict rule emptied. The line is `subject_track.py::SEED_MOSTLY` (0.5).
4. Find the box in `data/<video>/cast/cast.jsonl` by its frame and `box`, and if it is a bad seed withdraw it there (`bench/cast_file.py withdraw`) before the next preview. Nothing in the job or the capture carries a refused seed or an emptied frame forward; only the preview's log and shot table do.

Checked and verified on 2026-10-26: the `4698` box (`cast.jsonl` line 186) was refused on `shot_c_4647h` and taken on `shot_c_4647j` at a share of 0.56 while covering 0.16 of the box, splitting her track and emptying `4697`.

