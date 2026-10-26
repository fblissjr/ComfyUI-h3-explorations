# Multi-Pass Contamination, Perception Provenance, and Mask-Step Forensics (`2026-10-25`)

This record documents the root-cause analysis, file creation (`%w`) vs. modification (`%y`) audit, perception/depth model status, environment/folder rules, and step-by-step mask diagnostic tooling ([`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py)) built on `2026-10-25` to explain why multi-subject (`4647–4725`, `1306–1418`, `4479–4509`) and occluded single-subject (`4912–4946`, `2050–2292`) loads failed or needed special handling while unoccluded single-subject loads (`5114–5129`, `1039–1060`, `1061–1092`, `2623–2711`, `3967–4052`) succeeded cleanly. It also specifies **what every tool and rule built today does, what it does NOT do, and how and when to use each across any scene or project**.

Related rules and procedures:
- [`docs/masked_v2v/AGENTS.md`](AGENTS.md)
- [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
- [`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md)
- [`docs/masked_v2v/frames_rules.md`](frames_rules.md)
- [`docs/wiki/handover.md`](../wiki/handover.md)

---

## 0. Read first: what 2026-10-26 confirmed, corrected and added (annotated 2026-10-26)

Measured on 2026-10-26 by a Gemini reader (`gemf`) and reproduced by the Claude session (`mrfoo`); the owner's own
frame readings and marks; the builds the Gemini lead landed. Each item names the file or command that holds its
figures. Where this section and the sections below disagree, this section is the later measurement, and the older
line is marked `[2026-10-26: ...]` where it stands. The matching updates to
[`docs/masked_v2v/AGENTS.md`](AGENTS.md) §1 and [`docs/wiki/decisions.md`](../wiki/decisions.md) (`2026-10-26`)
landed in `b375d53e`, `7c195e68`, `68e8086d`, and `f6028d64`.

### 0.1 Disproved: pass 1 (`third_fp_4647e`) did not alter the second subject in the cut

- The raw working copy and the `1903Z` cut (equal to the `2355Z` cut on `4647–4725`) were compared at the `1024x768`
  canvas, inside her person track from the raw preview (`shot_c_4647h_second_track_00001.mkv`), in a 16 px ring
  around it, and over the whole frame, counting pixels changed by more than 8 levels on every frame `4664–4717`
  (the procedure: [`verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md) §7). Inside her track the
  count is in single digits on every frame; in the ring it stays in the low hundreds at most; the whole-frame change
  is the third subject's own face and beard. Pass 1's `matte` (`third_fp_4647e_window_1_region.npz`, `x = 389..524,
  y = 157..274` on `4689–4706`) does not touch her track or head (`x = 462..551, y = 256..383`).
- So §2.1's "changed `50–551 px` in the 16 px ring" and AGENTS.md §1's "erased `second`'s face and white collar on
  `4698–4715` (`1,673–2,183` altered pixels)" were a crop that held his head, not hers. The cut holds the film on her.
- What follows: `third_fp_4647e` stays kept and there is no order change on `4647–4725`; the "new face with dark
  hair" the owner saw on `4698–4715` is in `second_hfp_4647g_00001.mp4`, her own render, which was never laid
  (`data/fun/laid.jsonl` holds only `third_fp_4647e` on the span). §2.1's remedy of restoring her track from the raw
  preview is withdrawn with it.

### 0.2 Found: the `4697` collapse and the `4698–4706` drift were one hand-drawn seed box

- `data/fun/cast/cast.jsonl` line 186 is the owner's box on source frame `4698` (load frame 51, `[477, 246, 68,
  138]`, drawn 2026-10-23). The tracker's own log is `said["1105"]` in each preview's `preview.queued.json`:
  - raw preview `data/2026-10-24_shot_c_4647h_job`: the box is `SEED REFUSED` ("nobody lies mostly (0.50) inside the
    drawn box ... at most 0.42"), and the seed on load frame 48 (`4695`) is "tracked over 47-78 and kept on 48-68".
  - cut preview `data/2026-10-25_shot_c_4647j_job`: the same box "took person 3 ... (0.56 of their mask inside it ...
    covering 0.16 of the box), tracked over 48-78 and kept on 51-69", the `4695` seed is cut at `4696`, and the
    conflict rule prints "THE BOXES' TRACKS HELD DIFFERENT PEOPLE on 30, 31, 32, 50: left empty there" (50 = `4697`).
  The line is `subject_track.py::SEED_MOSTLY` (0.5); the shot table on the share records each box's `inside`, `covers`
  and `kept_on` and the emptied frames as `seed_tracks_disagree_on`.
- Person 3 is the woman in the black vest. Decoding the two track files shows it (white pixels `> 127`, boxes
  `[x0, y0, x1, y1]` on the canvas; the procedure: [`verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md) §8):

  | frame | raw preview `shot_c_4647h`, her track | cut preview `shot_c_4647j`, "her" track |
  |---|---|---|
  | 4696 | 4,855 px, `[474, 266, 544, 375]` | 6,652 px, `[461, 228, 589, 490]` |
  | 4697 | 2,987 px, `[476, 265, 536, 373]` | 0 px |
  | 4698 | 3,081 px, `[473, 264, 536, 378]` | 2,741 px, `[470, 256, 599, 489]` |
  | 4699 | 4,080 px, `[469, 267, 534, 381]` | 1,892 px, `[472, 259, 640, 490]` |
  | 4702 | 2,979 px, `[478, 265, 547, 381]` | 1,823 px, `[498, 251, 660, 500]` |
  | 4706 | 1,826 px, `[495, 263, 550, 378]` | 1,260 px, `[513, 255, 622, 487]` |

  Her head track and Sapiens parts are clipped to the person track, so on `4699–4702` they were clipped against the
  wrong person (`1,430 -> 75 px` on `4699` in §1's row is this, not DiT contamination). The picture on her had not
  changed (0.1); one rough box on an occluded frame crossed the tracker's 0.50 line by six hundredths.
- What follows: a load for her on the current cut is not built on `shot_c_4647j`. Withdrawing the `4698` box (`cast.jsonl`
  line 210) produced `shot_c_4647k` and `second_hfp_4647h` (§0.9 below). By the owner's ruling at the close of
  `2026-10-26` ([`docs/wiki/decisions.md`](../wiki/decisions.md)), only HTML `who-is-who` page clicks (`cast.jsonl`
  lines `1–174`, each identifying which detected person is the subject on that frame) stand as tracker identifications;
  boxes from pasted screenshots (`lines 175–192, 194–196`, `"just to give you a rough idea"`) and session-entered
  contact-sheet boxes (`lines 197–209`) are human hints and never tracker seeds, to be withdrawn together with
  tomorrow's tracker post-occlusion re-finding build.

### 0.3 Confirmed by reading the code: the capture's order, and what each consumer is handed

- The keep-out carve runs first, inside the per-mask loop, against her SAM track (`bench/capture_masked_run.py`,
  `carve_keep_out` near line 2575); `parts_held__<by>.mkv` is written by the grades (line 2843) and written again by
  `her_hair_in_hole`, which runs last, after the loop (line 5557). The fill is handed the other subjects' tracks
  uncarved, as read (`as_read`, near line 2567), while node `104` is handed the carved `subjects/<other>/track__<by>.mkv`.
  The head tube's bridged output reaches the fill at line 2841 (b461584e). Line numbers are of main at `45f3eaf1`.
- At the job step her part is held to her own SAM track grown 8 px before the capture sees it
  (`bench/shot_to_load.py::hold_to_track`, `h3_config.MASKED_STRANGERS["grow_px"]`), so a frame where SAM lost her
  head has an empty part even where the class map saw her.
- `subjects/<s>/part_grades.json` is written by the grades (line 2816) before the fill and never again, so
  `bench/frame_table.py` reports the floor state from before the fill: `emptied` where the fill later added.
- `head_marks` are read for the load's own subject only (`bench/shot_to_load.py::head_marks_of`, line 3625): a mark
  never reaches a keep-out subject's track.
- `--head-tube` exists on `job` (it records a word; the tube runs by default when a head track exists, line 2598).
  No minimum-fill constant exists in the capture tool. `facebook_hyden-da2-relative-depth` is absent from the code
  (§4.2 holds); MoGe-2 loads from `H3_MODEL_STORE` (`<model-store>`).

### 0.4 The composite rule, read from the node and the render's own region file; the judging rule it gives

- A latent step's region is the maximum over its four frames (`video_mask.py::token_mask`); the model regenerates it
  for all four; the composite lays it on every frame of the step whose own mask is non-empty and leaves a frame with
  no mask as the source (`video_mask.py::unlaid_frames`, `pixel_alpha`). Read on
  `second_hfp_4912g_windows/second_hfp_4912g_window_1_region.npz` (keys in
  [`verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md) §2; `mask` and `kept` are bit-packed):
  `4925` mask 0 px and matte 0; `4926` mask 5 px and matte identical to `4927`'s and `4928`'s (21,872 px above zero,
  15,400 at full weight). One white pixel lays the whole step; an empty frame lays nothing.
- The judging rule the owner settled on 2026-10-26: a mask is judged against the region the node regenerates (the
  blue on the inspector's strip before a render; `matte` in the window record after one), never against the green
  `prt` on the check strips, which is the part before the floor hold and understates both. A miss matters only where
  visible hair or face falls outside the region; a frame with no region at all is the one that needs a mark. On
  `4647g`'s `4688–4703` the laid region covers her whole head on every frame but `4697` and `4698`.
- The cost the rule carries: on a same-color crossing the step's region on the crossing frame includes some of the
  bystander (the woman in blue's hair on `4926`). Inherent to per-step regions; not a fault of a load.

### 0.5 Added: her hair is lost at the preview, before any mask tool runs (a blind spot of every ledger)

Measured on `Video/fun/gemlead/shot_c_4912h/shot_c_4912h_second_parts_00001.mkv` against
`Video/fun/parts/shot_c_4912h/shot_c_4912h_second_part_saved_on_track_8px.mkv`: the clip to her track costs a few
hundred pixels a frame (`4918`: 539; `4943`: 381), but the loss is in the preview's part itself: where her hair runs
edge to edge into another woman's hair of the same color, the part model, run on her SAM crop with everyone else
blanked, finds half (`4939`) or almost none (`4921`, `4943–4946`) of it. SAM's person boundary against a same-colored
neighbor is where her hair stops being hers. Recorded nowhere as a loss; compensated three stages later by a `hers`
mark and the fill. General to every shot where she stands against dark hair. (`start_from` is `noise` in every
queued graph, so the softened-source start plays no part.)

### 0.6 The owner's marks, final (2026-10-26), and what the fill does with each

- `4647–4725` (`second`): `hidden` on `4674–4685` and `4716–4725`; `hers` on `4697` (hair shows) and `4698` (face and
  hair show); nothing else, carried as the capture has it. Read off the restored strips in
  `data/2026-10-25_second_hfp_4647g/strips/` (`wide_4672_4687`, `check_4688_4703`, `check_4704_4725_gemf`,
  `zoom_4688_4703`). This replaces §2.1's `--hold-part 4674-4685`: her head is not in view there.
- `4912–4946` (`second`): `hers` on `4920`, `4921`, `4926`, `4939`, `4943–4946`; `hidden` on `4922–4925` and
  `4941–4942`; `not hers` on `4917`, `4919`, `4927–4928`; `4940` unmarked.
- On a fresh preview a `hers` frame comes out one of three ways, all handled (`bench/capture_masked_run.py::her_hair_in_hole`,
  the loops near lines 5283 and 5347): tracked whole, the mark adds nothing; partial, completed from the nearest whole
  head; lost, the nearest whole head placed with crown room read from her own track. `hidden` frames are skipped by
  the fill and not counted by the gate (b461584e).

### 0.7 Landed on 2026-10-26 (`git log`, main)

- `b461584e`: the head tube's bridged output is handed to `her_hair_in_hole` and to `bench/inspect_mask_steps.py`
  (closing §1's and §2.1's "line 2797 handed raw `head_track_arr`"); `hidden` frames skipped in the fill's parts 1
  and 3; the gate counts only frames the fill asked about; the full `~taken[k]` subtraction restored (§2.2 item 2).
  Replayed on the kept `4912` capture before landing.
- `ac803f92`: `bench/what_h3_got.py` joins the capture's records; a git-less check fake; the active-step 0 px test.
- `00a3f1d8` and `45f3eaf1`: held pixels kept in the track mkv where the fill extends beyond the track; a 4-connected
  bridge carved in `grid_carve` where a keep-out pinches a `hers` unit.
- `cbd39fbe` (Build D): moved settled `margin` and `zoom_up_to` recipes into `data/fun/plan.json` (`recipe` and
  `shot_overrides`, recording `second_hfp_1093k` at `margin = 128`) and made `bench/shot_to_load.py job` and
  `bench/flags_before_render.py::recipe_check_of` hard-refuse any differing value with no `--*-why` bypass.
- `HELD_EDGE_EMPTY_RUN` (§2.1 item 3, §6): holds as written, the parent of `449b17f1` had no limit.

### 0.8 Environment facts settled on 2026-10-26

- A build worktree must sit at `<ComfyUI checkout>/h3_worktrees/<builder>-<item>`, two levels below the ComfyUI root,
  so `bench/_lib` and `h3_config` resolve the checkout (`bench/_lib/__init__.py`, the `COMFY` resolution near line
  55; `workflows/h3_config.py::COMFY_ROOT`): a worktree under the home folder breaks `import comfy`. The empty
  `custom_nodes/h3_worktrees` was removed by the owner. §5.2's rule stands with this added.
- The server's output folder comes from ComfyUI's own `--output-directory` argument (`folder_paths` in
  `audio_freeze_song.py`), not from `H3_OUTPUT_DIR`; the variable guards the bench tools. §5.1's table is right about
  the readers; the server is guarded by its start script.
- The working copy and the lead's still live under `input/mrpop/` by history (`data/fun/manifest.json`, `plan.json`);
  the chain finds them by the manifest and the plan, not by the folder.

### 0.9 The render on the fresh preview, `second_hfp_4647h` (2026-10-26 evening): her hair dark again from 4684, and why

The owner's playback: blonde and right from 4647 until about 4684-4685 (`+36.9` BGR luma over source across `4647–4673`,
proving `2.17x` zoom in the single 79-frame window swaps her hair to blonde once unmasked dark hair is removed), dark
from there to the end (`+7.4` luma on `4686–4705`). Measured by the Gemini lead and the Claude session on the capture
(`data/2026-10-26_second_hfp_4647h`), the preview's files on the share (`shot_c_4647k`) and the render's window record;
the step mapping is the 17-frame cycle 1, 4, 4, 4, 4 repeating, read from the record's `tokens` (27 steps for 90 frames).
Four facts, none of them the causes §1 and §2.1 named:

- **4682-4685 (one step `t=11`, no token on)**: her person track there (`765–1,082 px`) is her bare forearm (`y=297–359`);
  the part Sapiens found on that crop (608 to 2,123 px a frame at `y=289–367`) sits on the forearm too, labeled neither
  face nor hair (`0` face points), and the head tube suppressed it rightly (`head_tube__shot_c_4647k.json`,
  `off_tube_suppressed_px`). Her head, peeking out from behind the lead's shoulder on 4684-4685, was found by nothing in
  the chain: not SAM, not Sapiens on a crop of her arm. The owner's `hidden` on 4674-4685 was read off strips that could
  not show it. These frames were in the window as filmed.
- **4686 and 4693-4695**: Sapiens on her crop found her whole head and hair (about 1,923 px at rows 258-316 on 4686,
  `2,203–3,607 px` on 4693-4695), and `hold_to_track` kept almost all of it (`1,833 px` on 4686); the tube cut over a
  thousand pixels of it (`945–1,354 px`) on each of these frames (`1,833 -> 562 px` on 4686; `1,923 -> 836 px` in
  `parts_held`), since her head track was empty 4682-4695 and `--head-per-stretch` was off. Her dark hair stayed in the
  window on the kept side of the boundary.
- **4686-4693 (`t=12–13`)**: the laid matte reached 40 to 70 px above her head (`y=181–258`) into another person SAM's
  people file (`shot_c_4647k_people.npz`) holds on every frame (2,000 to 3,500 px at scores 0.54 to 0.81), and the
  strangers file held zero pixels there: `bench/strangers.py::run` writes only what whole-frame Sapiens calls human, and
  it found nobody there. That head was inside her hole.
- **The fill (`bench/capture_masked_run.py:5132–5407`, `HAIR_IN_HOLE_CANNOT_SEE`) could not repair any of it**: on this
  capture it read hair from the whole-frame class map (the record's `hair_from`, item 2 of `HAIR_IN_HOLE_CANNOT_SEE`)
  rather than her crop class map (`classes__shot_c_4647k.npz`), subtracts her own person track (`~track[f]`) from what it
  may add (line 5384, item 3 of `HAIR_IN_HOLE_CANNOT_SEE` at line 5132), and added zero hair pixels over 79 frames; its
  head additions were all on 4702-4714, while Stage 2 (item 4) triggers on arm-only `person_track` frames (`4682–4685`)
  and copies a neighbor's head (`hers`) when `head_track` is empty. In addition (`lines 5303–5312`), `hole(f)` tests
  "inside the hole" against `out[f]` dilated by `margin_px = 30` canvas px **before** the keep-out carve and feather: on
  `4686` (`frame 39`), `head[39] & ~hole(39)` was `0 px` so Part 1 added nothing to `out[39]`, leaving `130 px` of
  `head_tube[39]` (`932 px` total) in the feathered rim where `matte <= 127` in `second_hfp_4647h_window_1_region.npz`.

Not a cause: her held hair on 4688, 4691-4693 (`726–1,460 px` of class `4` `Hair` on her own class map).
Withdrawn: a split of the shot across the occlusion (§6 of `AGENTS.md` forbids it). The rules this forces, with the
Gemini lead's two guardrails, are in `docs/wiki/decisions.md` 2026-10-26 and the handover's "Tomorrow, in order": her
part at the job step from her Sapiens face and hair on her crop with her head track, in a room anchored to head
detections (never to an arm-only crop, never unioned with her person track on a face-and-hair subject), less every other
person's SAM detection, the clip to the person track retired; SAM's other-person detections in the strangers keep-out;
the head track re-seeded per stretch by default; the pasted-screenshot and sheet boxes withdrawn as seeds.

---

## 1. Ten-Capture Forensic Matrix: Why `5114k`, `1039h`, `1061b`, `2623b`, `3967c`, `4479e`, and `1306h` Succeeded While `4647g` and `4912g` Failed

Running [`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py) with `H3_INPUT_DIR` and `H3_OUTPUT_DIR` set across all `10` captures (`4647g`, `4912g`, `2050g`, and the `7` renders kept on `2026-10-25` in `data/fun/owner_answers.jsonl`) exposes the exact structural difference between the loads that worked and the two that failed:

| Capture | Verdict | Subject & Span (`replaces`) | `[0]` Pass # & Provenance | `median_carried_px` (`whole_head_frames`) | `[1]` Pad Bleed | `[2]` Blue-Only (`0 px`) | `[3]` Specks in Blue | `[4]` Edge-Shrunk Whole | `[5]` `strangers` Swallow | `[6]` Missed SAM 3.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `2026-10-25_second_hfp_5114k` | **KEPT** | `second` (`5114..5129`, 16f, hair+face) | Pass #1 (raw 24p; `shot_c_5114f` from `10-22`, code updated after capture) | `11,315 px` (`16/16`) | `0` | `0` | `0` | `0` | `0` | `0` |
| `2026-10-25_second_hfp_1039h` | **KEPT** | `second` (`1039..1060`, 22f, hair+face) | Pass #1 (raw 24p; `shot_c_1039d` from `10-22`, code updated after capture) | `22,292 px` (`22/22`) | `0` | `0` | `0` | `0` | `0` | `0` |
| `2026-10-25_lead_fp_1061b` | **KEPT** | `lead` (`1061..1092`, 32f, face-only) | Pass #1 (raw 24p; `shot_c_1061c` from `10-25 15:33`) | `9,439.5 px` (`30/32`) | `0` | `0` | `0` | `4` (`1069, 1072, 1080, 1082`) | `0` | `0` |
| `2026-10-25_lead_fp_2623b` | **KEPT** | `lead` (`2623..2711`, 89f, face-only) | Pass #1 (raw 24p; **stale at build**: `shot_c_2623d` from `10-21 15:03` < commit `96f9989e`) | `73,575 px` (`86/89`) | `0` | `0` | `0` | `1` (`2624`) | `0` | `0` |
| `2026-10-25_lead_fp_3967c` | **KEPT** | `lead` (`3967..4052`, 86f, face-only) | Pass #1 (raw 24p; `shot_c_3967e` from `10-25 15:34`, `carved_keep_out=145 px`) | `9,602.5 px` (`67/86`) | `0` | `1` (`4039`, before `4039..4052` drop) | `0` | `8` (`3969..3970, 3981, 4010..4011, 4033, 4035, 4037`) | `0` | `9` (`4004..4012`) |
| `2026-10-25_third_fp_4479e` | **KEPT** | `third` (`4479..4509`, 31f, face-only) | Pass #2 (`2341Z` cut holding `second_hfp_4479h`; `strangers=None`; prior region overlapped `26f` up to `964 px`, `0` SAM drift frames) | `3,457 px` (`31/31`) | `0` | `0` | `0` | `0` | `0` | `0` |
| `2026-10-25_third_fp_1306h` | **KEPT** | `third` (`1306..1418`, 113f, face-only) | Pass #2 (`2341Z` cut holding `lead_fp_1306t`; `contested_px=149,666` without depth, `carved_keep_out=8 px`; `0` SAM drift frames) | `57,616 px` (`100/113`) | `0` | `0` | `0` | `5` (`1309, 1320, 1353, 1397, 1400`) | `0` | `14` (`1375..1401`) |
| `2026-10-24_lead_fp_2050g` | **KEPT (`2050..2248`)** | `lead` (`2050..2292`, 243f, face-only) | Pass #1 (raw 24p; `carved_keep_out=2,305 px`) | `21,932 px` (`216/243`) | `0` | `1` (`2249`, at spin cut) | `0` | `2` (`2099, 2244`) | `11` (`2251..2257, 2262..2265`, outside kept `2050..2248`) | `123` (`person#1..8`) |
| `2026-10-25_second_hfp_4647g` | **WRONG** | `second` (`4647..4725`, 79f, hair+face) | Pass #2 (`1903Z` cut holding `third_fp_4647e`; `contested_px=186` (`claimed_by_more_than_one_px=17,792`) without depth; **prior region overlapped `32f` (`4647..4689`) while `head_track=0` on `4674..4685`**) | `1,759 px` (`20/79`) | `1` (`win 0 step 23`: `4724, 4725, PAD_79, PAD_80`) | `6` (`4674..4676, 4697, 4724..4725`) | `1` (`4699`, `73 px` vs `701 px` step max) | `1` (`4694`, `1,033 px` vs `1,999 px`) | `0` | `33` (`4686..4725`) |
| `2026-10-25_second_hfp_4912g` | **WRONG** | `second` (`4912..4946`, 35f, hair+face) | Pass #1 (raw 24p; `shot_c_4912h` from `10-25 15:31`, `strangers` on raw 24p) | `9,078.5 px` (`22/35`) | `0` | `2` (`4925, 4941`) | `1` (`4926`, `1 px` vs `8,536 px` step max) | `2` (`4920, 4939`) | `13` (`4921..4926, 4940..4946`, woman-in-blue & un-tracked hair in `strangers`) | `24` (`4912..4938`) |

**Why the 7 kept renders succeeded today while `4647g` and `4912g` failed**:
1. **Unoccluded Pass-1 loads (`5114k`, `1039h`, `1061b`, `2623b`, `3967c`)**:
   - Have `0` prior passes laid on their spans, `0` `[1]` pad-bleed steps, `0` `[3]` speck frames, and `0` `[5]` `strangers` self-swallow frames.
   - Even where `5114k`, `1039h`, and `2623b` reused `2026-10-21` or `2026-10-22` Pass-1 previews, the subject was continuously tracked (`whole_head_frames = 16/16`, `22/22`, `86/89`) with no occlusion gap where `her_hair_in_hole()` or `strangers` needed to recover a lost head.
2. **Benign Pass-2 face-only loads (`4479e` and `1306h`) vs. occluded Pass-2 hair+face load (`4647g`)**:
   - On `4479e` (`third` after `second_hfp_4479h`) and `1306h` (`third` after `lead_fp_1306t`), Pass 1's regenerated region barely touched `third`'s periphery (`0` SAM track collapse frames between the raw preview and Pass-2 preview, `0` blue-only frames, `0` specks, `0` swallows).
   - On `4647g` (`second` after `third_fp_4647e`), Pass 1's `16x16` token grid overlapped `second` on `32` frames (`4647–4689`), while `second` lost her `head_track` (`0 px`) on `4674–4685` and `capture_masked_run.py:2797` handed raw `head_track_arr` (`0 px` on `4674–4695`) instead of `head_tube`'s bridged output (`907–2,867 px` on `4686–4695`) to `her_hair_in_hole`, reducing `whole_head_frames` to only `20 / 79` frames and `median_carried_px` to `1,759 px`. `[2026-10-26: the hand-off landed in b461584e (§0.7). The `1,430 -> 75 px` head-track collapse on `4699` in the row above is the seed box of §0.2, not pass-1 contamination.]`

---

## 2. Two Compounding Failure Chains on `4647–4725` and `4912–4946`

### 2.1 Multi-Subject Sequential Pass Contamination (`third_fp_4647e` → `second_hfp_4647g`) vs. The Kitchen `604–1009` Bouffant Hair Rule (`second_hfp_0604h` → `lead_fp_0606d`)

On shot `4647–4725` (`79` frames), three subjects appear (`lead` on the left, `second` in the middle, `third` on the right).
1. **What `third_fp_4647e` (Pass 1) changed in the `1903Z` cut around `second`**:
   - In `data/2026-10-24_third_fp_4647e_job/third_fp_4647e.render.json` (`node 104 MiniMaxH3MaskedSource`), Pass 1 ran with `grow_pixels: 16`, `composite: "whole region"`, and `others: ["210", 0]` (`lead`, `second`, and `strangers` tracks), and `assemble_delivery.py` laid `third_fp_4647e` into `fun_0000_6394_joined_2026-10-25T1903Z_from_laid.mp4` with `restore=lead:whole restore=second:whole`.
   - Because `restore=second:whole` gave back `second`'s tracked mask (`changed_inside_second = 0..36 px`), `second`'s pixels inside her raw track stayed the raw video's, while `third_fp_4647e`'s `16x16` token grid (`third_fp_4647e_window_1_region.npz`) overlapped `second` on `32` frames (`4647–4689`) and changed `50–551 px` in the `16 px` ring immediately outside `second`'s track (`changed_within_16px_of_second`, plus `third`'s face and beard right beside `second`). `[2026-10-26: disproved as a change to her; measured again at the canvas with a threshold of 8 levels the ring holds at most a couple of hundred changed pixels and the track single digits, and the change near her is his face and beard (§0.1). Pass 1's matte never touches her track.]`
2. **Pass 2 (`shot_c_4647j` → `second_hfp_4647g`) ran on `1903Z` with `head_tube: true`, but `capture_masked_run.py:2797` passed raw `head_track_arr` to `her_hair_in_hole`, and ran without `--hold-part` or `depth.npz`**:
   - On `4647–4673`, `shot_c_4647j` tracked `second`'s body cleanly (`22,521–51,297 px`) and her head through `4673` (`1,210–6,929 px`, including `6,929 px` track / `1,272 px` tube on `4672`). On `4696–4703`, `shot_c_4647j` tracked her head at `1,416–6,652 px` (`0 px` only on `4697`). `[2026-10-26: those are person-track figures, and on `4698–4706` that track is the woman in the black vest, not her (§0.2).]`
   - On `4674–4685` (where `second` is in the background with her head turned away), her raw `head_track` dropped to `0 px` (`4674–4695`), while `head_tube` (enabled by default in `capture_masked_run.py:2558`, `head_tube: true` in `manifest.json`) regained and bridged `4686–4695` (`907–2,867 px`). However, `capture_masked_run.py:2797` handed raw `head_track_arr` (`0 px` across `4674–4695`) to `her_hair_in_hole` instead of `head_tube__shot_c_4647j.npz["tube"]`, and `second_hfp_4647g` ran without `--hold-part` or `depth.npz` (`contested_px = 186`, `claimed_by_more_than_one_px = 17,792` unsettled).
3. **Why `second`'s mask dropped out on `4694–4702` and bled into `PAD_79–80` (`4724–4725`)**:
   - `[2026-10-26: the cause of both is the owner's seed box on `4698`, refused on the raw preview and taken on the cut's (§0.2); the box is withdrawn before any preview of her on this span.]`
   - **Blue-only frame on `4697` (`[2]`) and speck-in-blue frame on `4699` (`[3]`)**: Frame `4697` had `carried_px = 0 px` inside active latent step `14` (`45` canvas cells, `163` zoomed tokens, peers `4694: 1033px, 4695: 1197px, 4696: 1183px, 4697: 0px`), so `video_mask.unlaid_frames` left `4697` as unswapped source while the DiT regenerated step `14`. Frame `4699` had `carried_px = 73 px` (`10.4%` of step `16`'s max `701 px`), pasting only a `73 px` speck.
   - **Edge-shrunk `whole` reference on `4694` (`[4]`)**: Frame `4694` (`ref_px = 1,033 px`, `51.7%` of `4693`'s `1,999 px` as her head crossed behind `third`) was chosen as the nearest non-empty `whole` reference for the gap, shrinking the recovered mask.
   - **Tail pad-bleed on `4724–4725` into `PAD_79, PAD_80` (`[1]`)**: On `4724–4725` (window `0` latent step `23`: `['4724', '4725', 'PAD_79', 'PAD_80']`), `second` had already exited the frame (`mask_px = [0, 0, 2089, 2089]`). Because `video_mask.py::held_edge_source` had no empty-run limit after `f5e0d66d`, it stepped backward `11` frames across the `4715–4725` empty tail to `4714` (`2,089 px` > `HELD_EDGE_LEAST = 1,024`) and copied `4714`'s mask into `PAD_79` and `PAD_80`, activating `7` canvas cells (`16` zoomed tokens) on latent step `23`. Commit `449b17f1` introduced `HELD_EDGE_EMPTY_RUN = max(FRAME_PER_TOKEN)` (`4`, one 4-frame latent step) to cap that reach-back.
4. **Why we must NOT revert to raw-source keep-out masks or graft tracks across plates (The Kitchen `604–1009` Bouffant Hair Rule, `lead_fp_0606d` over `second_hfp_0604h`, frames `755–774`)**:
   - In Kitchen `604–1009`, `second` (`mom`) was replaced in Pass 1 (`second_hfp_0604h`): to replace all of the 1980s mom's dark bouffant hair (`ymin = 188..209`), Pass 1's own mask had to cover her entire tall bouffant hairdo (and at the bottom near her shoulders, `hair_band` / `her_hair_in_hole` / `--margin` expand a subject's own mask when raw segmentation misses dark hair tips). Because the new blonde woman (`second_hfp_0604h`) has flatter, straighter hair (`ymin = 211..221`), Pass 1 filled the upper ~60 px dome of the old bouffant mask above her new blonde crown with kitchen wall/curtain background.
   - In the older `lead_fp_0751c (10/15)` (frames `760–761`), `lead` was rendered and assembled using `second`'s **old raw bouffant hair mask**: `second`'s old bouffant keep-out carved a bite out of `lead`'s chin/mouth during `lead`'s render, and at assembly (`two_pieces_change_the_same_pixels`) `second_hfp_0604h`'s kitchen-wall fill in that upper 60 px bouffant dome overwrote `lead`'s lower face so `lead` disappeared into the invisible old bouffant hair mask as she walked behind `second`.
   - Running Pass 2's preview (`shot_c_0604e`) and render (`lead_fp_0606d`) **on the post-Pass-1 cut** (`fun_0000_6394_joined_2026-10-19T0029Z_from_laid.mp4`) with `head_tube` and `--depth` solved Kitchen `604–1009` because:
     1. On the post-Pass-1 cut, SAM 3.1 + Sapiens segmented `second`'s **new lower blonde head** (so `second`'s keep-out no longer held the upper 60 px bouffant dome); and
     2. `lead`'s mask (`head_tube`) masked **over** Pass 1's kitchen-wall fill in that 60 px band right down to the new blonde crown.
   - **How the rules work together**:
     - **A subject's own mask (Pass 1)** expands as needed (`hair_band`, `her_hair_in_hole`, `--margin`, `--hold-part`) to cover all of the original actor's hair/beard so no dark hair tips or beard fringes are left behind near the shoulders or jaw.
     - **Pass 2's preview and keep-out masks (`others`)** **always** come from the **post-Pass-1 cut** so they hug the new replaced subject's actual silhouette, and tracks are **never** grafted from a raw preview onto a post-Pass-1 capture.
     - **Pass 2's own target subject (`second` on `4647–4725`)**: `her_hair_in_hole` reads `head_tube`'s bridged output (`head_tube__<by>.npz["tube"]`), paired with owner `head_marks` (or single-span `--hold-part 4674-4685`) and `measure_moge2_depth.py` (`depth.npz`). `[2026-10-26: the owner marked `4674–4685` `hidden` (§0.6): nothing is held there, and `--hold-part` is withdrawn for this span.]`

### 2.2 Occlusion Gaps, Woman-in-Blue Keep-Out, and Class-Filter Pitfall on `second_hfp_4912g` (`4912–4946`)

On `second_hfp_4912g` (`4912–4946`, 35 frames), `second` is occluded twice: on `4921–4926` (by the woman in blue) and on `4941–4946`.
1. **What happened on `4921–4926` and `4940–4946`**:
   - On `4922–4925`, `second` is occluded behind the woman in blue (`strangers` pixels on `4922–4925` belong to the woman in blue in front of `second`, not a false self-swallow of `second`'s visible head). On `4921–4925`, `hair_in_hole__shot_c_4912h.json` recorded `lost_asked_on_source_frames: [[4921, 4925], [4941, 4946]]` and `hair_undecided_on_source_frames: [[4917, 4922], [4925, 4928]]` with `marks: {"hidden": [], "hers": [], "not hers": []}`, and all `6` blocking flags in `flags.json` (`f001..f006`) were overridden rather than answered with `head_marks`.
   - On `4926`, `second`'s raw part mask was `103 px` (`< floor_px = 1,024`), which `capture_masked_run.py` emptied to `0 px` (`emptied_by_the_floor_on_source_frames`), and `her_hair_in_hole` part 1 (`near_h & ~taken`) added back a single `1 px` speck (`added_px = 1` on `4926` in `hair_in_hole__shot_c_4912h.json:15, 228-231`), while `4925` (`0 px` inside active latent step `4`: `4925..4928`) and `4941` (`0 px` inside active latent step `8`: `4938..4941`) were left as `0 px` (`left_as_the_source_with_no_mask: [[4925, 4925], [4941, 4941]]`).
   - On `4940` (`1,907 px`), `second`'s hair is already behind the hat brim so only her face (`1,907 px`) is visible; `4940` is a valid visible face frame, whereas `4920` (`5,279 px`) and `4939` (`5,231 px`) are transition frames before the gaps.
   - **Why class-filtering `taken_for_hers` in `449b17f1` was wrong and reverted**: commit `449b17f1` tried filtering `taken` down to `Apparel` + non-subject `face_ids` (`taken_for_hers`). When `4912h` ran with that change, bystander hair of the same colour (`Hair` class, not `Apparel` or `Face_Neck`) on `4943–4946` was no longer in `taken_for_hers` and was claimed as `second`'s hair (`2,381–3,777 px`), painting blonde hair onto the bystanders. `her_hair_in_hole` therefore subtracts every other person's mask (`~taken[k]`) and relies on `head_tube` + owner `head_marks` (`hers` / `hidden` / `not hers`) for occlusion gaps. `[2026-10-26: the full subtraction landed in b461584e; the owner's final marks for `4912–4946` are in §0.6, and `4926` is `hers` by the composite rule of §0.4: its few pixels of mask lay the whole step's region, which is her `4927–4928` head grown.]`

---

## 3. Audit of File Creation (`%w`) vs. Modification (`%y`) Dates and Code Commit Skew

`bench/inspect_mask_steps.py` check `[0]` inspects three distinct timestamps for every input (`manifest.json`, the source video on `<share>/input/`, the preview's `preview.queued.json` and `.mkv` masks on `<share>/output/`, and `strangers.mkv`):
- **File creation / birth time (`stat().st_birthtime` / `%w`)** and **modification time (`stat().st_mtime` / `%y`)** on disk.
- **`written` timestamp** recorded in `manifest.json` (`subjects[s][k]["written"]`) vs. the **latest git commit timestamp** touching the producer code (`subject_track.py`, `sam31_corrections.py`, `shot_table.py`, `sapiens2_parts.py`, `strangers.py`).

Crucially, check `[0]` separates two cases that would otherwise be conflated:
1. **`stale_at_build` (`! STALE AT BUILD`)**: Input files that were **already older than a producer-code commit at the time the capture was built** (`in_ts < code_ts <= cap_created_ts`):
   - Seen on `2026-10-25_lead_fp_2623b`: built at `2026-10-25 13:09:42`, but its `track`, `head_track`, and `classes` came from `shot_c_2623d` (`written = 2026-10-21T15:03`, `disk_created = 2026-10-21 15:03`), which predated `2026-10-25` commits to `subject_track.py`, `sam31_corrections.py`, and `sapiens2_parts.py`.
2. **`code_updated_since_capture` (`* INFO (code updated after capture built)`)**: Captures whose inputs were current (or benignly reused) before a **later** code commit landed on the repo after the capture was built (`in_ts < cap_created_ts < code_ts`):
   - Seen on `2026-10-25_second_hfp_5114k` (`shot_c_5114f` from `2026-10-22T16:50`) and `2026-10-25_second_hfp_1039h` (`shot_c_1039d` from `2026-10-22T12:12`), where commit `96f9989e` landed after those captures were built at `14:39` and `14:41`.

---

## 4. Audit of Parked / Forgotten Perception Flags (`--depth` / `MoGe-2` Metric Depth, `--head-tube`, & `Sapiens2` Matting)

### 4.1 `facebook_hyden-mogev2-metric-point` (`bench/shot_to_load.py job --depth` / `bench/measure_moge2_depth.py` → `depth.npz`) and `head_tube`

- **When `--depth` and `head_tube` were built and how each runs**:
  - Built and measured on **`2026-10-17` and `2026-10-18`** ([`bench/results/2026-10-18_moge2_depth_on_crossings.md`](../../bench/results/2026-10-18_moge2_depth_on_crossings.md)):
    - `workflows/h3_config.py:137` defines `MOGE2_MODEL = "facebook_hyden-mogev2-metric-point/model.pt"` (`vendor/meta_moge2/hyden_moge2.py::load_hyden_moge2`).
    - [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) (`:2558`) runs `head_tube` **by default** (`not args.no_head_tube`, recording `head_tube: true` in `manifest.json`) whenever a `head_track` exists; [`bench/shot_to_load.py`](../../bench/shot_to_load.py) (`:5251-5256`, `overlap_frames_of`) accepts `--head-tube [WHY]` (which records the word in `job.json`) and `--depth [WHY]`: when `--depth` is passed, `shot_to_load.py job` runs [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) on CPU (`~3.97 s/frame`) **only on frames where two subject tracks or a keep-out overlap**, writes `depth.npz` into the capture folder, and passes `--depth <capture>/depth.npz` through `bench/load_chain.py::capture_args` to `bench/capture_masked_run.py files`.
    - [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) (`:236-267`, `:1383-1410`, `DEPTH_WIN_M = 0.05`, `DEPTH_BOUNDARY_SLACK_M = 0.02`) writes `owners.npz` and settles contested pixels (`contested & ~front`), and [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) (`:1349-1369`, `restore_weight`) gates `restore=<other>:whole` by `captures.depth_settled` and `captures.in_front`.
  - Used on **`2026-10-18` to `2026-10-20`** on `data/2026-10-18_lead_fp_0751d/depth.npz` (`contested_px = 1,670`, `100%` front-kept) and Kitchen `data/2026-10-19_lead_fp_0606c/depth.npz` / `lead_fp_0606d` (`contested_px = 1,995`, `86%` front-kept; `data/2026-10-20_lead_fp_0606d_job/job.render.json:4` carried both `--head-tube` and `--depth`).
- **Why `--depth` stopped running after `2026-10-20` (and why `head_tube`'s bridged output was missed by `her_hair_in_hole`)**:
  - In `bench/shot_to_load.py:5251-5256`, `--depth` is an **opt-in CLI flag** (`default=None`), and `--carry-words-from` does not carry it from an earlier job. Meanwhile, `head_tube` was already running by default in `capture_masked_run.py:2558` (`head_tube: true`), but `capture_masked_run.py:2797` handed raw `head_track_arr` (`0 px` on `4674–4695` in `4647g`) instead of `head_tube__<by>.npz["tube"]` to `her_hair_in_hole`. `[2026-10-26: fixed in b461584e.]`
  - Because no pre-render check flagged `contested_px > 0` with `depth_used == False`, sessions after `2026-10-20` omitted `--depth` on multi-person crossing shots:
    - On `2026-10-25_second_hfp_4647g` (`depth_used = False`): `186` contested pixels (`17,792` claimed by more than one) were unsettled (`front_kept_px = 0`).
    - On `2026-10-25_third_fp_1306h` (`depth_used = False`): `149,666` contested pixels between `third` and `lead` were unsettled (`front_kept_px = 0`).
- **How we verify and use `--depth` and `head_tube` going forward (Input-Creation & Consumption Provenance)**:
  - Because `bench/shot_to_load.py::overlap_frames_of` runs `bench/measure_moge2_depth.py` **only** on frames where two tracks or a keep-out overlap, `--depth` is useful across every shot with overlapping tracks/keep-outs (`0` cost on non-overlapping frames).
  - Check `[0]` of [`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py) and [`bench/what_h3_got.py`](../../bench/what_h3_got.py) explicitly report `depth_used`, `contested_px`, `head_tube`, and `hair_in_hole`.

### 4.2 `facebook_hyden-da2-relative-depth` & `Sapiens2ForImageMatting`

- **`facebook_hyden-da2-relative-depth`**: Not present in the repo (`0` references in code or `workflows/h3_config.py`); MoGe-2 (`facebook_hyden-mogev2-metric-point`) is the metric depth model used here.
- **`Sapiens2ForImageMatting` (`sapiens2_parts.py:36-39, 55-60`)**:
  - `sapiens2_parts.py` imports `Sapiens2ForImageMatting` alongside `Sapiens2ForSemanticSegmentation`, and defines `MATTING_WEIGHTS`, with `NO_MATTING = "none"` as the default when matting weights are not selected.
  - Every masked preview graph built by `workflows/masked_job_graph.py` passes `matting: "none"`, so alpha matting has never been active in the masked lane.

---

## 5. Environment Variables, Share Paths, Worktree Safety, and the Stray `./Video/` Directory

### 5.1 Canonical Environment Variables (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`)

| Variable | Role | Readers (`file:line`) | Fallback / Required Setting |
|---|---|---|---|
| `H3_OUTPUT_DIR` | Root output directory on the share (`<share>/output`, holding `Video/<video>/`) | `workflows/h3_config.py::output_dir` (`:2077-2085`), `bench/shot_to_load.py`, `bench/load_chain.py`, `bench/rebuild_delivery.py`, `bench/inspect_mask_steps.py` | ComfyUI's local `output/` directory (`<comfy-root>/output`) |
| `H3_INPUT_DIR` | Root input directory on the share (`<share>/input`, holding working copies, `*_from_laid.mp4` cuts, and reference stills) | `bench/masked_job.py::input_dir` (`:160-171`), `bench/rebuild_delivery.py`, `bench/inspect_mask_steps.py` | ComfyUI's local `input/` directory (`<comfy-root>/input`) |
| `H3_MODEL_STORE` | Local perception/pose model store (`$H3_MODEL_STORE`, `<model-store>`; never `<comfy-root>/models`) | `bench/load_chain.py` (`os.environ.get("H3_MODEL_STORE")`), `docs/comfy_notes.md` | Must be set to `$H3_MODEL_STORE` (`<model-store>`) |
| `H3_CAPTURE_ROOT` | Optional capture root override | `subject_track.py`, `sapiens2_parts.py`, `video_mask.py` | Repo's `data/` directory (`Path(__file__).parent / "data"`) |

### 5.2 Why `./Video/fun/gemlead/` Appeared Inside the Repo on `2026-10-24`

- On `2026-10-24 20:51`, a session invoked `bench/shot_to_load.py` with a relative `--output Video/fun/gemlead` without `H3_OUTPUT_DIR` resolved against the share, creating `./Video/fun/gemlead/lead_fp_2050f` and `./Video/fun/gemlead/lead_fp_2050g` inside the repo root (`ComfyUI-h3-explorations/Video/`).
- Those two folders held only stale `2026-10-24` intermediate `.mkv` masks (`4.7 MB`), while the real renders and `.mkv` masks used by the server live on `<share>/output/Video/fun/gemlead/`. The stray `./Video/` directory has been removed.
- **Worktree rule**: Never run `shot_to_load.py`, `masked_job.py`, `load_chain.py`, or `rebuild_delivery.py` from inside a worktree (`h3_worktrees/<branch>`) without pointing `--shots`, `--plan`, and capture paths at the main repo's `data/` and setting `H3_INPUT_DIR` and `H3_OUTPUT_DIR` to the share. Worktrees are strictly for editing and testing code (`bench/check_*.py`) before landing on `main`. `[2026-10-26: and they sit at `<ComfyUI checkout>/h3_worktrees/<builder>-<item>`, never under the home folder (§0.8).]`

---

## 6. Reference Guide: Everything Built & Landed Today (`2026-10-25`) — What It Does, What It Does Not Do, and When to Use It

| Tool / Module (`commit`) | What It Does | What It Does NOT Do | When & How to Use It (Across Any Scene or Project) |
|---|---|---|---|
| **[`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py)** & [`bench/check_inspect_mask_steps.py`](../../bench/check_inspect_mask_steps.py) (`449b17f1`) | Audits any capture across `[0]` provenance (Pass 1/2/3 vs `laid.jsonl`, birth `%w` + mtime `%y`, `stale_at_build` vs `code_updated_since_capture`, `depth.npz` usage, and prior-pass region overlap / SAM track collapse), `[1]` `(1,4,4,4,4)` latent-step `PAD` bleed, `[2]` blue-only `0 px` frames, `[3]` speck-in-blue frames, `[4]` `<75%` edge-shrunk `whole` frames, `[5]` `strangers` self-swallow, `[6]` missed SAM 3.1 objects in the blue region, and `[7]` per-frame gap recovery + `--strips <out.jpg>` visual overlay (`red` mask, `blue` latent step, `green` strangers, `yellow` recoverable whole head). | Does **not** modify any capture `.npz`/`.mkv` file, does **not** run GPU models (`SAM 3.1`, `Sapiens`, `MoGe-2`), and does **not** judge whether a rendered face looks like the reference still (only whether the masks and latent steps handed to the DiT and `lay_back` are geometrically sound). | Run **before every render** (right after `load_chain.py on-plan`) and **immediately when debugging any failed render**: `H3_INPUT_DIR=<share>/input H3_OUTPUT_DIR=<share>/output <python> bench/inspect_mask_steps.py --capture data/<day>_<load> --problems-only --strips /tmp/<load>_strips.jpg`. |
| **`video_mask.py::HELD_EDGE_EMPTY_RUN = 4`** (`449b17f1`) | Introduces an empty-edge reach-back limit in `held_edge_source()` at `4` (`max(FRAME_PER_TOKEN)`, `1` DiT temporal token; no empty-run limit existed after `f5e0d66d`). When a subject is absent (`0 px`) at the start or end of a shot, `PAD` frames hold the empty edge (`0 px`) unless a real mask cleared `HELD_EDGE_LEAST` (`1,024 px`) within the last `4` frames. | Does **not** fill mid-shot occlusion gaps inside the shot (those are handled by `head_tube_parts.py` and `her_hair_in_hole`), and takes effect only when ComfyUI (`:8188`) executes `MiniMaxH3VideoMask` during `plan` or `render`. | Automatic on every `plan` and `render` once `:8188` is restarted on `449b17f1+`. Prevents tail/lead pad-bleed (`[1]`, such as `4714 -> 4724..4725, PAD_79..80` on `4647g`) on any shot where a subject enters late or exits early. |
| **`bench/capture_masked_run.py` & `bench/head_tube_parts.py::her_hair_in_hole`** (`clean_whole`, `abs(k - src) <= 2`, `head_tube` handoff, `~taken[k]`, `unmasked_head_hair_context`) | 1) Drops `< 0.75 * neighbor` edge-shrunk frames from `clean_whole` so gap recovery pulls from full unshrunk heads; 2) widens `near_h` to `abs(k - src) <= 2`; 3) passes `head_tube`'s bridged output (`head_tube__<by>.npz["tube"]`) into `her_hair_in_hole` and subtracts every other person's mask (`~taken[k]`, reverting `449b17f1`'s class-filtered `taken_for_hers` which missed same-colour bystander hair on `4943–4946`) `[2026-10-26: this item landed on 2026-10-26 in b461584e, not on 2026-10-25; §1 and §2.1 of this record describe the state before it]`; 4) records `unmasked_head_hair_context` joining `part_grades`, `head_tube`, and `hair_in_hole`. | Only modifies `parts_held__<preview>.mkv` for **subjects whose plan replaces `Hair`** (`hair_in_hole_files` returns early for face-only subjects like `lead` or `third`). Multi-frame occlusion gaps where both raw `head_track` and `head_tube` are empty (`lost_asked_on_source_frames`) require owner `head_marks` in `marks.json` (or single-span `--hold-part A-B`). | Automatic during `capture_masked_run.py` (`load_chain.py first` and `on-plan`) for every hair+face subject (`second`). Pair with owner `head_marks` (`hers` / `hidden` / `not hers`) when `hair_in_hole` asks on lost-head frames. |
| **[`bench/flags_before_render.py`](../../bench/flags_before_render.py)** (`SUBJECT_RECIPE_DEFAULTS`, `unlaid_while_tracked_of`, `unresolved_hair_in_hole_of`) (`79b7b6da`, `e5be4a20`, `449b17f1`, `cbd39fbe`) | Enforces pre-render gates: `[0]` recipe gate against `data/fun/plan.json` (`recipe` and `shot_overrides`; `[2026-10-26: prior to cbd39fbe, recipe_check_of was a diff that passed with any written --*-why, which let third_fp_5114d run at margin 64 against 16; Build D (cbd39fbe) moved settled margin and zoom_up_to into plan.json and made both shot_to_load.py job and recipe_check_of hard-refuse any differing value with no --*-why bypass (docs/wiki/decisions.md, 2026-10-26)]`), `mouthless_reference` check, `partial_shot` check, `unlaid_while_tracked` check (using `runs/<load>/region.npz` `carried` as authoritative), and `unresolved_hair_in_hole` check (`PAUSE — VISIBLE HEAD/HAIR SIGNALS ON UNMASKED FRAME(S)`). | Cannot see prior-pass image contamination baked into a Pass-2 `--source` cut (use `inspect_mask_steps.py` `[0]` for that), and cannot see missed SAM 3.1 strangers inside the blue region unless they are in `keep_outs` (use `inspect_mask_steps.py` `[6]`). | Mandatory before every render (`bench/flags_before_render.py --capture data/<day>_<load> --load <load> --initial --by <session>`). Refuses `--initial` and `load_chain.py render` until every blocking gate is resolved. |
| **[`bench/shot_to_load.py`](../../bench/shot_to_load.py)** & **[`bench/laid.py`](../../bench/laid.py)** Pass-2 & Partial-Lay Guards (`79b7b6da`, `e5be4a20`) | `shot_to_load.py` refuses building a Pass-2 load on raw source when another subject is already laid on the shot, or on a cut that already holds the **same** subject's piece; `laid.py` refuses laying a raw-rendered piece over a multi-subject shot (`--allow-parallel-raw WHY`) or laying a partial span without `--allow-partial-lay WHY`. | Does **not** settle contested pixels between overlapping subjects unless `--depth` (`depth.npz`) is run. Never graft raw-preview tracks onto a post-Pass-1 capture. | Active on every `shot_to_load.py preview/job` and `laid.py lay/replace`. Before building Pass 2 on a cut, run `inspect_mask_steps.py` `[0]` and pass `--depth` when tracks or keep-outs overlap. |
| **[`bench/inventory.py`](../../bench/inventory.py)** & **[`bench/timeline_page.py`](../../bench/timeline_page.py)** (`ec9a113f`) | Builds the whole-video per-`(shot, subject)` inventory (`inventory_<video>_<stamp>.md/.json`) and interactive NLE timeline (`TIMELINE.html` / `timeline_<video>_<stamp>.json`) showing every frame's status (`carried`, `held`, `left_out_by_a_look`, `owed`, `not_present`), age tier (`pre-10/20`, `10/20-10/22`, `10/23+`), mid-shot seams, and frame-pinned owner notes (`bench/said_on_shot.py`). | Reports coverage at the **delivery-table / cut level** after loads are built or laid; does **not** inspect inside an unlaid capture's `.mkv` or `.npz` mask steps (that is `inspect_mask_steps.py`'s job). | Run automatically as part of `bench/rebuild_delivery.py` (or standalone when planning the next batch of shots) to see which shots are `done`, `partial`, or `nothing laid`, and what the owner previously said on each shot. |
| **Eye-line roll & bare-frame reference guards** (`bench/head_reference.py`, `bench/crop_reference_to_head.py`) (`86e65d1f`) | Flags eye-line roll disagreement (`>= 15 deg`) between 2D Sapiens pose and 3D SAM-Body mesh in `marks_against_the_body_model`, and skips bare (`--no-head-on`) and dropped (`--drop`) frames when computing `cranium_box` so empty/headless frames do not bloat or shift the reference crop box. | Does **not** change the source video's mask or region (`parts_held` / `region.npz`); operates strictly on `<Video 1>` (the 3D body-model reference video). | Automatic on every `shot_to_load.py reference` / `head_reference.py` run. Check `[2]` in `flags_before_render.py` (`marks_against_the_body_model` and `face_on_the_track`) before rendering. |
