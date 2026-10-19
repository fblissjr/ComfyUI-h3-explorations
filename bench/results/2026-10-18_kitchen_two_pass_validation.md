# Validating a two-pass scene by hand: the kitchen at 751-785, after the 24 fps fix (2026-10-18)

lane: masked video to video
verdict: measured across four kitchen runs (`lead_fp_0751`, `lead_fp_0751d`, `lead_fp_0606`, `lead_fp_0606b`); the 24.0 fps cut fix removes the 1-frame background slip, the 29-frame `d002` seam defect, and the 755-775 dark chin smear; the remaining 759-761 and 772 face holes trace to Sapiens2-Seg mislabelling her chin/face as `Upper_Clothing` and `Background` while SAM 3.1 tracks the head cleanly

Written by mrted (the Claude lead) on 2026-10-18, before the run, at the owner's ask: what was run, how and why, the
scene's background, how a two-pass scene was validated by hand tonight, and what should probably be done on every
scene. The exact commands, as run, are kept outside git because they carry machine paths:
`internal/gemini/2026-10-18_kitchen_24fps_check.md` (linked from `docs/wiki/handover.md`). **This is a record of one
scene's check, not a recipe.** Only the three points under "Always true" hold everywhere.

## The scene, and why it needed two passes

The kitchen, source frames 604-1009: the mom (`second`) at the table and the lead walking behind her. Each person is
replaced in a pass of their own (two or more subjects in one pass is a closed lane, `docs/roadmap.md`). The mom's pass,
`second_hfp_0604h`, was kept and laid on 2026-10-18 (`bench/results/2026-10-18_second_hfp_0604h_recipe.md`); the lead's,
`lead_fp_0606`, was kept and laid the same day. Both had been rendered against the original video, so where the lead
passes behind the mom (about 755-775) the lead's chin was generated beside the original mom's dark hair, and when the
blonde mom was laid beside it a dark smear showed on the lead's chin (gem-smear, `internal/gemini/2026-10-18_worker_gem-smear.md`).

The fix tried was the second pass of the order the leads agreed for crossings: the person in front first, then the
person behind rendered against the cut that already holds the new front person. That run, `lead_fp_0751` (751-785),
was not kept: an open mouth from the old drawing, and hard square edges round her head.

## What was found tonight

1. **The cut was 23.976 fps.** The original `fun.mp4` is 23.976 fps; every job renders from a 24 fps every-frame copy,
   and every kept piece did. But `bench/rebuild_delivery.py` took the original before the copy when it made the joined
   cut, so the cut, and the stacks and review clips made from it, were written at 23.976
   (`docs/research/postmortems/2026-10-18_24fps_joined_cut.md`, with both leads' words).
2. **So the second pass loaded one frame early.** The job loader forces no rate (`h3_config.MASKED_JOB_FORCE_RATE`) and
   turns a frame into a time assuming 24 fps (`workflows/masked_job_graph.py::seek`); from about frame 600 a 23.976 file
   starts a frame early. `lead_fp_0751` loaded frames 750-784 of the cut against masks and a reference on 751-785, so
   every regenerated square carried the previous frame's background (mrgemson's trace, G30).
3. **The four-frame step and the margin also cost squares.** A keep-out or a margin that touches a cell on one frame of
   a four-frame latent step takes it for the whole step (`video_mask.py`, the keep-out and `run_lengths`); with the
   subject's own mask winning every cell it touches (`video_mask.py:2005`), a keep-out acts only on margin cells.

## How the scene was validated by hand tonight

In order (the commands are in the untracked file named above):
1. The fix on its own branch: the rebuild takes the 24 fps copy; the job step refuses a source not recorded at 24;
   the stack tool reads a saved start time at the rate it was written at. Cleared, landed.
2. The cut rebuilt: it must probe 24/1 with the same 6,395 frames, registered like the input's 24 fps copy.
3. `lead_fp_0751` rendered again against the new cut, every setting as before except the source, the same seed.
4. Old and new run measured with the same, fixed tools, in one pass:
   - each loader's first frame (`bench/what_h3_got.py`): source, masks and reference must agree;
   - the render against its source at offsets -2..+2, outside the region (`bench/stack_delivery.py::offsets_mae`,
     pointed at the render): best offset 0 when the frames line up;
   - pixel diffs: the new cut against the old, and frame to frame inside the region;
   - the capture's per-frame record (`frame_table.jsonl`, `manifest.json`): cells each keep-out gave back, depth;
   - the stack's yellow squares and kept pixels (`bench/stack_delivery.py`);
   - the assembler's findings (`bench/assemble_delivery.py`), `d001` and `d002` among them;
   - the block measure on the picture itself (`bench/measure_blocks.py`, written tonight);
   - the render's face points against the source's (the land step's pose pass);
   - the gap check on the span (`bench/frame_coverage.py`);
   - a Gemini reader watching both natively, writing a card that cites the figures.
5. The candidate on Start here for the owner; laid only on the owner's keep.

## Always true (the owner's rules, `docs/wiki/decisions.md`, 2026-10-18)

- H3 gets 24 fps from input to output, and a cut used as a source is made and loaded exactly like the real input's
  24 fps copy.
- Derived data (`.npz` files, tracks, class maps, depth, references) is made again after any change to the code that
  makes it.
- Old and new are measured with the same tools.

## What should probably be done on every scene (a proposal, for the owner and both leads)

- Two-person shots: the person in front first; the person behind rendered against the cut that holds them, at 24 fps.
- Before any render: the loaders' first frames agree (source, masks, reference).
- After every render: the facts beside the clip, in its JSON sidecars, read before anyone watches (`docs/wiki/handover.md`,
  item 1c): the offset check against the source, the pixel diffs, the block measure, the mouth opening against the
  source's, the face points, the gaps, and the reader's card citing them.
- The margin and the keep-outs read against the subject's own mask before a crossing renders (item 2b).
- On the back pass of a crossing, keep out only the front person's laid matte (where depth puts them in front), not
  their whole-body track (mrgemson, G31): on `lead_fp_0751` the mom's whole track reached into the lead's 16 px face
  margin where her laid piece had replaced nothing.
- Before a crossing renders, count the cells a keep-out or margin takes per four-frame step, not per frame (mrgemson,
  G31): a collision across a step can show 0 on every single frame. His example on 773-776 was traced with the stack
  tool before tonight's fix, which indexed the old run a frame off; re-check it with the fixed tools before relying on it.

## Results

### 1. Four-run comparison on 751-785 (and full 606-998 kitchen span)

All four runs were measured with the fixed tools (`bench/what_h3_got.py`, `bench/stack_delivery.py::offsets_mae`,
`bench/assemble_delivery.py`, `bench/frame_coverage.py`, `bench/measure_blocks.py`):

| Check | `lead_fp_0751` (old `23.976` cut, 35f) | `lead_fp_0751d` (new `24/1` cut, 35f) | `lead_fp_0606` (kept, `24/1` orig source, `606-998`) | `lead_fp_0606b` (new `24/1` cut, `606-998`, after `5ebaecfe` + `ffca8acd`) |
|---|---|---|---|---|
| Source file & rate | `...1932Z_from_laid.mp4` (`24000/1001`) | `...0029Z_from_laid.mp4` (`24/1`) | `fun_0000_6394_24p_sound_in_step.mov` (`24/1`) | `...0029Z_from_laid.mp4` (`24/1`) |
| `what_h3_got.py` first frame (source / masks / ref) | **`750` / `751` / `751` (1f early)** | **`751` / `751` / `751` (aligned)** | `606` / `606` / `606` | **`606` / `606` / `606` (aligned)** |
| `offsets_mae` outside region on `751-785` (best offset) | **`-1`** (`4.248` at `-1` vs `12.815` at `0`) | **`+0`** (`4.230` at `0` vs `12.847` at `-1`) | `+0` | **`+0`** (`1.209` at `0` vs `11.158` at `-1`) |
| Assembler `d002` `restored_pixels_beside_a_large_change` | **`29` frames** (`752-772`, `778-785`), max `39.7` | **`0` frames** | `0` frames | **`0` frames** |
| Assembler `d001` `subject_edge_step` | `13` frames, max `20.0` at `762` | `12` frames, max `22.7` at `762` | `16` frames (`606-998`), max `25.0` at `969` | **`7` frames (`606-998`)**, max `18.3` at `614` (`3` frames on `751-785`, max `16.9` at `752`) |
| Gap check (`frame_coverage.py`) `lead.tracked_off_pct` on `751-785` | `21.444%` | `15.274%` | `5.902%` (`3.570%` on `604-1009`) | **`5.902%`** (`3.570%` on `604-1009`) |
| `measure_blocks.py` on render piece (`751-785`, `1024x768`) | `21` squares (`12`f): `7` regen, `4` kept, `10` outside; max grid `25.38`, max step `77.79` | `34` squares (`13`f): `13` regen, `7` kept, `14` outside; max grid `26.81`, max step `67.54` | `111` squares (`19`f): `44` regen, `29` kept, `38` outside; max grid `31.19`, max step `63.69` | **`44` squares (`17`f)**: `25` regen, `10` kept, `9` outside; max grid `29.44`, max step `59.40` |
| `measure_blocks.py` on candidate cut (`751-785`) | `22` squares (`13`f); max grid `25.94`, max step `40.70` | `22` squares (`12`f); max grid `27.38`, max step `38.68` | `25` squares (`13`f); max grid `27.38`, max step `44.93` | **`24` squares (`12`f)**: `10` regen, `6` kept, `8` outside; max grid `25.00`, max step `45.32` |
| Yellow `kept_squares` taken per 4-frame step (`760-763` / `764-767` / `768-771` / `772-775`) | `27` / `28` / `23` / `16` (indexed on aligned `751d` grid) | `27` / `28` / `23` / `16` (`off_head` cut on `772`) | `34` / `41` / `20` / `18` | **`15` / `14` / `7` / `5`** (cut by >50% after `ffca8acd`; `0` `outside` squares on `768-775`) |

### 2. Why the face holes on 759-761 and 772 remained (`C181` / `C185` trace)

Even after `ffca8acd` stopped `off_head` from eroding inside the single-component head (`off_head_said_px = 0` on every
frame of `751-785`), the red regenerated part on `759-761` and `772` still had a missing lower-face bite because
Sapiens2-Seg mislabelled her chin and lower cheek as `Upper_Clothing` (`20,689 px` on `759`, `21,218 px` on `760`,
`16,405 px` on `772`) or `Background` (`22,769 px` on `761`), while SAM 3.1's person track above the shoulders
(`sam_above`, `y <= max(left_shoulder_y, right_shoulder_y)`) held her whole head cleanly (`30,548 px` on `759`,
`27,741 px` on `760`, `24,806 px` on `761`, `38,375 px` on `772`). Because `head_tube_for_capture`
(`bench/head_tube_parts.py:1488`) called `sam31_head_tube` without passing `tracker=`, the head tube was built from
Sapiens' own head classes and collapsed wherever Sapiens dropped the face (`tube_px = 10,277`, `11,182`, `6,969`,
`23,223 px`).

### 3. Reader cards and owner verdicts

- **`lead_fp_0751`** (`data/2026-10-18_lead_fp_0751/playback_reading_gemread-0751d.md`, reader verdict `remake`):
  loaded source `750-784` one frame early from the `23.976` cut; `29` frames of `d002` seam defect; open mouth on the
  silent stretch from the pre-`5ebaecfe` drawing; face holes at `759-761` and `772`. Owner verdict (`data/fun/verdicts/verdicts.jsonl`):
  not kept (open mouth and square edges round her head).
- **`lead_fp_0751d`** (`data/2026-10-18_lead_fp_0751d/playback_reading_gemread-0751d.md`, reader verdict `remake`):
  `24/1` cut aligns `first_frame = 751` (`offsets_mae` best `+0`, `d002 = 0` frames) and removes the 1-frame background
  slip, but the `759-761` and `772` face holes remain visible (`lead_fp_0751d_window_1_with_mask` screenshots, `C179`).
  Owner verdict: root-cause diagnostic showing the mask cutting into her face at `760` and `772`.
- **`lead_fp_0606b`** (`data/2026-10-18_lead_fp_0606b/playback_reading_mrgemson.md`, reader verdict `unsure`):
  full `606-998` render against the `24/1` cut with `5ebaecfe` (silent-window no-mouth drawing + darkened grey head) and
  `ffca8acd` (`off_head` single-component fix). Eliminates the `755-775` dark chin smear against the blonde mom, cuts
  piece blocks on `751-785` from `111` (`lead_fp_0606`) to `44`, and cuts yellow kept squares on `760-775` by more than
  half (`15, 14, 7, 5` vs `34, 41, 20, 18`), while the Sapiens `759-761` / `772` face-hole outline remains until the
  SAM 3.1 head track (`gem-sam-head`, `C186`/`C190`) replaces Sapiens' dropped chin classes.

