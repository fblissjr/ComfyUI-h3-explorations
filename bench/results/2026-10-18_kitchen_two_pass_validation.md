# Validating a two-pass scene by hand: the kitchen at 751-785, after the 24 fps fix (2026-10-18)

lane: masked video to video
verdict: measured across four kitchen runs (`lead_fp_0751`, `lead_fp_0751d`, `lead_fp_0606`, `lead_fp_0606b`); the 24.0 fps cut fix removes the 1-frame background slip and the 29-frame `d002` seam defect (`lead_fp_0606b` owner verdict: "slightly better", not laid); the chin smudge at 760 is the same in `0606` and `0606b` because the chin squares (`y = 207..255`) are kept as the source's own pixels (dropped by Sapiens2-Seg and cut by the mom's whole-track keep-out outside her laid matte)

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

### 2. Why the chin smudge at 760 and face holes on 759-761 / 772 remained (`C181`, `C185`, `C195`)

- **The owner's look on `760` (`C195`)**: watching `lead_fp_0606b_0604_1009_compare.mp4` at frame `760`, the dark smudge
  at the lead's chin is the same in `lead_fp_0606` and `lead_fp_0606b`. Checking `frame_table.jsonl` and `region.npz` on
  the 4-frame latent step `760-763` explains why:
  - On `760`, the red regenerated mask (`part_px = 5,054`) covers only `x = 366..450, y = 94..206` (stopping above her
    chin), while the region's `kept` mask covers `118` cells (`27,016 px`) spanning `x = 288..479, y = 64..255` in
    `0606b` and `0751d` (`154` cells, `34,004 px` in `0606`).
  - Across the `760-763` 4-frame step, `second`'s whole-track keep-out overlaps `12`, `11`, `9`, and `7` region cells on
    `760-763` (`18, 17, 13, 8` in `0606`) and gives back `0, 1, 4, 9` margin cells (`0, 6, 10, 9` in `0606`) outside
    the mom's laid matte.
  - Because every square in `y = 207..255` across her chin is **kept as the source's own pixels** (both where Sapiens
    dropped her chin and where the mom's whole-body keep-out cut the margin outside the mom's laid piece), the original
    source pixels show through at her chin in both `0606` and `0606b`.
- **Sapiens vs SAM 3.1 on `755-775` (`C181` / `C185`)**: even after `ffca8acd` stopped `off_head` from eroding inside the
  single-component head (`off_head_said_px = 0`), Sapiens2-Seg mislabelled her chin and lower cheek as `Upper_Clothing`
  (`20,689 px` on `759`, `21,218 px` on `760`, `16,405 px` on `772`) or `Background` (`22,769 px` on `761`), while
  SAM 3.1's person track above the shoulders (`sam_above`, `y <= max(left_shoulder_y, right_shoulder_y)`) held her whole
  head cleanly (`30,548 px` on `759`, `27,741 px` on `760`, `24,806 px` on `761`, `38,375 px` on `772`). Because
  `head_tube_for_capture` (`bench/head_tube_parts.py:1488`) called `sam31_head_tube` without passing `tracker=`, the head
  tube was built from Sapiens' own head classes and collapsed wherever Sapiens dropped the face (`tube_px = 10,277`,
  `11,182`, `6,969`, `23,223 px`).

### 3. Reader cards and owner verdicts (`C195`, `C196`)

- **`lead_fp_0751`** (`data/2026-10-18_lead_fp_0751/playback_reading_gemread-0751d.md`, reader verdict `remake`):
  loaded source `750-784` one frame early from the `23.976` cut; `29` frames of `d002` seam defect; open mouth on the
  silent stretch from the pre-`5ebaecfe` drawing; face holes at `759-761` and `772`. Owner verdict (`data/fun/verdicts/verdicts.jsonl`):
  not kept (open mouth and square edges round her head).
- **`lead_fp_0751d`** (`data/2026-10-18_lead_fp_0751d/playback_reading_gemread-0751d.md`, reader verdict `remake`):
  `24/1` cut aligns `first_frame = 751` (`offsets_mae` best `+0`, `d002 = 0` frames) and removes the 1-frame background
  slip, but the `759-761` and `772` face holes remain visible (`lead_fp_0751d_window_1_with_mask` screenshots, `C179`).
  Owner verdict: root-cause diagnostic showing the mask cutting into her face at `760` and `772`.
- **`lead_fp_0606b`** (`data/2026-10-18_lead_fp_0606b/playback_reading_mrgemson.md`, reader verdict `unsure`):
  full `606-998` render against the `24/1` cut with `5ebaecfe` and `ffca8acd`. Cuts piece blocks on `751-785` from `111`
  (`lead_fp_0606`) to `44` and yellow kept squares on `760-775` from `34, 41, 20, 18` to `15, 14, 7, 5`, while the dark
  chin smudge at `760` remains the same as `0606` because those chin squares (`y = 207..255`) are kept as the source's
  own pixels (`C195`). **Owner verdict (`C196`)**: `"slightly better"` (recorded in `data/fun/verdicts/verdicts.jsonl` as
  a candidate, not laid: the `24/1` cut and long window helped a little at the chin; what remains needs the SAM 3.1 head
  track, the back pass keeping out the mom's laid matte instead of her whole track, and the silent-window no-mouth
  drawing; the kitchen stays as laid until those are in).

### SAM 3.1 Head Track Mask Test on `751-785` (`lead`, no render; `8b50cecf`)

Running `MiniMaxH3SubjectTrack` (`track_head=True`) on the 406 frames of `shot_c_0604c` (`604-1009`) left the person track `0 / 406` frames (`0 px`) changed; running `capture_masked_run.py files` on `751-785` with `head_track` off was byte-identical to `data/2026-10-18_lead_fp_0751d`. On a 35-frame preview (`751-785`, `1024x768`, 1 subject, warm cache after model load), `/history` and `/system_stats` recorded `MiniMaxH3SubjectTrack` node time `7 s` (`23.60 s` graph wall, `13,994 MiB` peak card VRAM / `8,119 MiB` torch VRAM) with `track_head=False` and `9 s` (`+2 s`, `26.14 s` graph wall, `14,676 MiB` peak card VRAM / `8,801 MiB` torch VRAM, `+682 MiB`) with `track_head=True` (cold `track_head=False`: `8 s` node time, `30.78 s` graph wall, `13,579 MiB` peak card VRAM).

With `head_track` on (`Video/fun/mrgemson/shot_c_0604c_sam_head_test/sheets/compare_751_785_key_frames.jpg`), `0` frames fall under the `480 px` floor (`least = 739 px` on `751`, `median = 8,747 px`, `most = 18,287 px` on `772`; `0` frames held by the floor rule vs `2` frames `784-785` in `0751d`), and `0 px` are lost inside `head_track` on every frame (`clipped_outside_head` on `767-778` is Sapiens `Face_Neck` collar/chest bleed below her chin `y > head_tr_ymax`):

| `sf` | `person_track` (px) | `head_track` (px) | `head / person` | `old_prt` (`0751d`, px) | `new_prt` (SAM head, px) | `gained` (px) | `lost_in_head` (px) | `clipped_outside_head` (px) | `kp_in_old` | `kp_in_new` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 751 | 144,913 | 31,977 | 0.221 | 852 | 739 | 0 | 0 | 113 | 28/65 | 28/65 |
| 752 | 140,241 | 32,610 | 0.233 | 1,823 | 1,710 | 0 | 0 | 113 | 28/65 | 28/65 |
| 753 | 138,209 | 33,133 | 0.240 | 3,585 | 3,557 | 1 | 0 | 29 | 28/65 | 28/65 |
| 754 | 134,516 | 32,413 | 0.241 | 8,217 | 7,261 | 6 | 0 | 962 | 28/66 | 28/66 |
| 755 | 127,578 | 31,878 | 0.250 | 6,894 | 6,807 | 40 | 0 | 127 | 29/65 | 29/65 |
| 756 | 118,055 | 29,138 | 0.247 | 6,709 | 6,291 | 27 | 0 | 445 | 31/65 | 31/65 |
| 757 | 109,721 | 27,461 | 0.250 | 4,929 | 7,063 | 2,334 | 0 | 200 | 29/65 | 47/65 |
| 758 | 95,180 | 25,145 | 0.264 | 7,759 | 7,470 | 629 | 0 | 918 | 56/65 | 56/65 |
| **759** | 83,070 | 23,124 | 0.278 | **4,637** | **10,662** | **+6,025** | **0** | 0 | 15/60 | **51/60** |
| **760** | 68,704 | 21,792 | 0.317 | **5,054** | **13,505** | **+8,457** | **0** | 6 | 0/63 | **54/63** |
| **761** | 60,588 | 20,807 | 0.343 | **2,037** | **12,189** | **+10,152** | **0** | 0 | 1/63 | **54/63** |
| 762 | 47,344 | 20,401 | 0.431 | 7,719 | 8,189 | 493 | 0 | 23 | 50/64 | 55/64 |
| 763 | 29,933 | 20,167 | 0.674 | 8,582 | 8,778 | 399 | 0 | 203 | 59/64 | 59/64 |
| 764 | 49,258 | 19,691 | 0.400 | 8,500 | 8,937 | 539 | 0 | 102 | 44/64 | 54/64 |
| 765 | 49,876 | 20,173 | 0.404 | 10,238 | 10,365 | 181 | 0 | 54 | 66/67 | 66/67 |
| 766 | 48,377 | 21,032 | 0.435 | 9,439 | 11,346 | 2,032 | 0 | 125 | 34/66 | 56/66 |
| 767 | 50,593 | 21,814 | 0.431 | 12,456 | 10,727 | 495 | 0 | 2,224 | 37/64 | 57/64 |
| 768 | 58,139 | 23,025 | 0.396 | 13,813 | 10,893 | 411 | 0 | 3,331 | 45/65 | 45/65 |
| 769 | 64,717 | 23,915 | 0.370 | 15,120 | 10,815 | 45 | 0 | 4,350 | 36/66 | 36/66 |
| 770 | 68,482 | 24,525 | 0.358 | 15,231 | 10,939 | 0 | 0 | 4,292 | 51/65 | 51/65 |
| 771 | 68,622 | 26,987 | 0.393 | 15,307 | 11,636 | 407 | 0 | 4,078 | 42/64 | 45/64 |
| **772** | 65,888 | 31,010 | 0.471 | **15,693** | **18,287** | **+6,117** | **0** | 3,523 | 56/64 | **56/64** |
| 773 | 55,274 | 24,748 | 0.448 | 16,886 | 11,886 | 284 | 0 | 5,284 | 61/61 | 61/61 |
| 774 | 42,895 | 23,632 | 0.551 | 16,103 | 10,989 | 575 | 0 | 5,689 | 41/54 | 54/54 |
| 775 | 37,701 | 22,471 | 0.596 | 14,341 | 10,128 | 310 | 0 | 4,523 | 50/54 | 54/54 |
| 776 | 32,499 | 20,775 | 0.639 | 11,973 | 9,640 | 97 | 0 | 2,430 | 51/54 | 54/54 |
| 777 | 29,877 | 18,876 | 0.632 | 9,921 | 8,747 | 21 | 0 | 1,195 | 50/54 | 49/54 |
| 778 | 30,042 | 18,283 | 0.609 | 8,805 | 8,093 | 12 | 0 | 724 | 27/54 | 27/54 |
| 779 | 31,133 | 18,046 | 0.580 | 5,538 | 7,816 | 2,384 | 0 | 106 | 7/54 | 27/54 |
| 780 | 31,010 | 17,739 | 0.572 | 5,953 | 7,212 | 1,400 | 0 | 141 | 5/54 | 27/54 |
| 781 | 31,219 | 17,956 | 0.575 | 7,557 | 7,190 | 22 | 0 | 389 | 27/54 | 27/54 |
| 782 | 31,228 | 17,651 | 0.565 | 3,837 | 7,030 | 3,235 | 0 | 42 | 2/54 | 27/54 |
| 783 | 30,788 | 17,785 | 0.578 | 3,951 | 7,170 | 3,313 | 0 | 94 | 0/56 | 36/56 |
| **784** | 29,932 | 16,710 | 0.558 | **180** (held 3,987) | **7,405** | **+7,225** | **0** | 0 | 0/54 | **36/54** |
| **785** | 29,646 | 16,769 | 0.566 | **371** (held 4,195) | **8,044** | **+7,676** | **0** | 3 | 0/54 | **41/54** |
