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

1. Read `digests` in `<job_folder>/<load>.render.queued.json` (or `<load>.plan.queued.json`) to get the exact file paths and `sha256` hashes the server read (`bench/what_h3_got.py` verifies those hashes).
2. Decode the `.mkv` mask files (`subjects/<subject>/parts_held__<preview>.mkv`, `track__<preview>.mkv`, and each keep-out `subjects/<other>/track__<preview>.mkv`) directly with `ffmpeg` or `cv2.VideoCapture` and count white pixels (`> 127`) per frame.
3. Compare those raw per-frame pixel counts and bounding boxes (`[x0, y0, x1, y1]`) against `subjects/<subject>/per_frame.json` (`parts_share`, `parts_box`, `track_share`, `track_box`).
   - *Example (`2050e` / `2050g`)*: Decoding the four `.mkv` mask videos matched the carried record counts to within a pixel on frames `0`, `22`, and `44`, and scanning all `243` frames of `parts_held__shot_c_2050f.mkv` confirmed `min_x = 256` (frame `2272`) and `max_x = 790` (frame `2245`).

---

## 2. What the Node Did (The Node's Own Mask-Review Video & Raw Zoomed Canvas)

Beside every render `<load>_00001.mp4`, the node (`video_mask.py`, `audio_freeze_song.py`) writes its own diagnostic videos on the share under `Video/<video>/<session>/<load>/<load>/`:
- `<load>_mask_00001.mp4`: the node's own mask-review overlay (what part was taken and where the grown margin sits).
- `<load>_00001_zoomed.mp4`: the raw zoomed canvas before `video_mask.lay_back` composites the hole back onto the full frame (`bench/assemble_delivery.py` reads this companion at `--size source`).
- `<load>_windows/<load>_window_N_region.npz`: the node's saved `[steps, 48, 64]` token grid (`tokens`), pixel `mask`, and `margin` matte for window `N`.

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
