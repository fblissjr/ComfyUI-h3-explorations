# Preserving the recipe for `second_hfp_0604h`, the mom's kitchen shot on frames 604–1009 (2026-10-18)

lane: masked video to video
verdict: kept by the owner on 2026-10-18 ("mom looks amazing. nice work. like 100x better", `data/fun/verdicts/verdicts.jsonl` line 386), replacing `second_hfp_0604d`, `second_hfp_0949d` and `second_hfp_0604g` across source frames 604–1009 (406 frames).

## 1. The owner's verdict and what this record preserves

On 2026-10-18 at `15:31:00Z`, after watching the candidate-against-current comparison clip `output/Video/fun/review_clips/second_hfp_0604h_0604_1009_compare.mp4`, the owner kept `second_hfp_0604h` over the three prior kitchen pieces for the second subject (`second_hfp_0604d` on `604–948`, `second_hfp_0949d` on `949–998`, and `second_hfp_0604g` on `999–1009`):

- **Verdict record** (`data/fun/verdicts/verdicts.jsonl`, line 386):
  - `at`: `2026-10-18T15:31:00Z`
  - `row`: `second_hfp_0604h`
  - `render`: `Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_00001.mp4`
  - `first`: `604`, `frames`: `406` (`604–1009`)
  - `chosen`: `"keep this one"`
  - `words`: `"mom looks amazing. nice work. like 100x better"`

Because the mom's kitchen flicker had been the central open fault on this shot for days and `second_hfp_0604h` resolved it cleanly across all 406 frames, the owner asked that its code, workflow graphs, plan state, and data be frozen so `second_hfp_0604h` can be reproduced byte for byte.

---

## 2. Code commit, local git tag, and what `second_hfp_0604h` ran with

- **Commit recorded in both the plan and render graphs** (`second_hfp_0604h.plan.built.json`, `second_hfp_0604h.render.built.json`, `second_hfp_0604h.render.queued.json`, and `data/2026-10-17_second_hfp_0604h/manifest.json`): **`8d007f54`** (`uncommitted_node_code: []`, `builder_not_committed: []`).
- **Local git tag** (never pushed): `recipe/second_hfp_0604h` at `8d007f54`.
- **Queue time**:
  - `plan` queued at `2026-10-17T15:45:38-05:00` (`2026-10-17T20:45:38Z`), prompt id `37a09e65-e0cf-459a-9a0b-cc618327c805`.
  - `render` queued at `2026-10-17T16:14:14-05:00` (`2026-10-17T21:14:14Z`), prompt id `d749e477-8209-44a4-846a-2f3c376d6061`, completed in `2,131 s` (`2,108 s` node stage total: `1,853.7 s` sampling, `148.0 s` conditioning, `57.7 s` source encode, `26.1 s` decode, `10.1 s` mask review, `5.2 s` write, `4.4 s` composite, `1.5 s` motion reference, `0.5 s` join and mux, `0.4 s` track encode).
- **Which changes were in tree at `8d007f54` when `second_hfp_0604h` ran**:
  - **Ran WITH**:
    1. The `N2` plan edit (`2026-10-17T14:13Z` in `data/fun/plan.json`, retiring `second`'s default `held`, `hands`, and `clothes` keep-outs and `hair_band`, and naming her apron and floral blouse in `plan.json` `clothing`).
    2. The whole-shot single load over `604–1009` in two chained windows (`[294, 202]` in source frames: `202` source frames `808–1009` = `90` context `808–897` + `112` written `898–1009`; `[294, 243]` as sampled by the node: `202` source frames + `41` held frames past `1009` trimmed before write) meeting at `898` with `context: 90` at `context_noise: 0.8`.
    3. `--carry-words-from` (`2196e8d1`), carrying the human span words from `second_hfp_0604g.json` (`--eyes-shut 938-953` and `968-1009` in `lids` form, `--track-not lead:1003-1009`, and `--seam 898`).
    4. `part_on_track` (`77035465`, from 2026-10-15), bounding `second`'s part mask to her own SAM 3.1 track grown by `8 px` (`h3_config.MASKED_STRANGERS`).
    5. The per-frame class-map mouth gate and hysteresis (`bench/results/2026-10-15_class_mouth_gate.md` and `bench/results/2026-10-15_mouth_runs.md`, from 2026-10-15).
    6. The one compositing law (`cecd8db1` and `save_kept: true` / `composite: "whole region"`), laying the piece by its own saved matte and comparing it in `bench/stack_delivery.py`.
  - **Ran WITHOUT** (landed after `8d007f54`, or not re-run because `0604h` used the 2026-10-14 preview `shot_c_0604b`):
    - `D22b` floor holds and off-head filtering (`9f139d0c`, landed after `0604h` was queued).
    - The head tube, the three-state face label, and the closed-lip line (landed later on 2026-10-17/18).
    - SAM 3.1 `memory_selection` (landed 2026-10-17, whereas `0604h` read the 2026-10-14 preview `shot_c_0604b`).

---

## 3. The command sequence as run, with each step's flags

All commands run with the ComfyUI virtualenv Python and `H3_INPUT_DIR`, `H3_OUTPUT_DIR`, and `H3_MODEL_STORE` set in the environment (`docs/working/comfy_notes.md`).

### Step 0: Shot preview (`shot_c_0604b`, run 2026-10-14 by `mrcircle`)

`second_hfp_0604h` takes its track, part, class map, body mesh, and pose table from the existing preview `data/2026-10-14_shot_c_0604b_job` (seeded on source frame `806` from `data/fun/cast/cast.jsonl` for both `second` and `lead`):

```bash
<python> bench/shot_to_load.py preview \
  --shots data/fun/shots.json \
  --plan data/fun/plan.json \
  --shot 604 \
  --folder data/2026-10-14_shot_c_0604b_job \
  --by mrcircle \
  --cast data/fun/cast/cast.jsonl \
  --remake \
  --queue
```

### Step 1: `job` (`2026-10-17T20:33:20Z`, by `mrgemsol`)

Builds `data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json` over the whole shot `604–1009` (`406` frames) under the `N2` plan, carrying the span words (`--seam 898`, `--context-noise 0.8`, `--track-not lead:1003-1009`, and `--eyes-shut 938-953` and `968-1009` in `lids` form) from `data/2026-10-15_fun_kitchen_inputs/second_hfp_0604g.json`:

```bash
<python> bench/shot_to_load.py job \
  --shots data/fun/shots.json \
  --plan data/fun/plan.json \
  --shot 604 \
  --subject second \
  --preview data/2026-10-14_shot_c_0604b_job \
  --folder data/2026-10-17_fun_kitchen_inputs \
  --by mrgemsol \
  --remake \
  --carry-words-from data/2026-10-15_fun_kitchen_inputs/second_hfp_0604g.json
```

What `job` wrote and bound:
- Bound `second`'s part to `second`'s track grown `8 px` (`output/Video/fun/parts/shot_c_0604b/shot_c_0604b_second_part_saved_on_track_8px.mkv`).
- Bound `lead`'s keep-out track to the copy with `1003–1009` emptied (`output/Video/fun/tracks/shot_c_0604b/shot_c_0604b_lead_track_not_1003-1009.mkv`).
- Wired `others: ["lead", "strangers"]` (with `held`, `hands`, `clothes`, and `hair_band` absent under `N2`).
- Assigned new reference folder `second_0604_1009_b` (since `second_0604_1009` had been read by `second_hfp_0604g`).

### Step 2: `reference` (`2026-10-17T20:42:46Z`, by `mrgemsol`)

Runs the pose pass inside `second`'s tracked box over `604–1009` and draws the head reference into `output/Video/fun/references/second_0604_1009_b/second_head_reference_0604_1009.mp4` (`sha256: 80a841d8f39494e1d1a95325f4ac8232ccdbc8a25f5ecc551ebc4693f279ed13`), with `iris="contrast"`, `mouth="auto"` (`323` of `406` frames drawn as the class map, `51` with open inside; `32` frames left out by `CLASS_MOUTH_GATE` on `645–650`, `652–659`, `681–686`, `725–733`, `745–747`), and `eyes_shut` drawn as single closed lines (`form: "lids"`) on `938–953` and `968–1009`:

```bash
<python> bench/shot_to_load.py reference \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol
```

### Step 3: `text` (`2026-10-17T20:42:56Z`, by `mrgemsol`)

Generates `data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.txt` (`2,163` characters, `sha256: 96f2c02d40250a591750e5face1a2290ef68149bbb9ec3eec588036d6e0fb6e1`) from the reference record, `data/fun/plan.json`, and the voice table (`talks, not heard` on `604–1009`, so no voice sentence is written) with `load_text.NO_SCENE`:

```bash
<python> bench/shot_to_load.py text \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol \
  --plan data/fun/plan.json
```

### Step 4: `first` (`2026-10-17T20:44:50Z`, by `mrgemsol`)

Runs `who_is_who`, builds the initial capture in `data/2026-10-17_second_hfp_0604h`, runs `verify` (`0`), runs `gate` (`clear`, `22` flags: `2` likely fine, `20` iffy, `0` likely to fail), and draws the initial strip under `output/Video/fun/looks/2026-10-17_second_hfp_0604h_first/`:

```bash
<python> bench/load_chain.py first \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol
```

### Step 5: `plan` (`2026-10-17T20:45:14Z` built, `20:46:20Z` finished, by `mrgemsol`)

Builds `data/2026-10-17_second_hfp_0604h_job/second_hfp_0604h.plan.json` (`sha256: 1154d58b6b9f6fd142ccc93edea94cc1a8e2dd25271102059a6cd7a398744efe`) at commit `8d007f54` and queues it on ComfyUI (40 s, nothing sampled), writing `second_hfp_0604h_plan.json`, `second_hfp_0604h_window_1_planned_region.npz`, and `second_hfp_0604h_window_2_planned_region.npz`:

```bash
<python> bench/load_chain.py plan \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol
```

### Step 6: `on-plan`, flag override, blind look, and `table` (`2026-10-17T20:46:34Z–21:13:25Z`, by `mrgemsol`)

Copies the initial capture to `data/2026-10-17_second_hfp_0604h_first`, rebuilds `data/2026-10-17_second_hfp_0604h` on the node's planned windows (carving `27 px` / `8` cells on `8` frames of `strangers` out of `second`'s own track), verifies (`0`), gates (`24` flags, blocked only on `f001`: `region_over_another_subject` on `759–773` where `lead` passes behind `second`'s chair), and draws the plan strip `output/Video/fun/looks/2026-10-17_second_hfp_0604h/second_hfp_0604h.mp4`:

```bash
<python> bench/load_chain.py on-plan \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol

# Override f001 (lead's crossing behind second's chair on 759-773 is restored at assembly by restore=lead:whole) and re-gate clear:
<python> bench/capture_masked_run.py outcome \
  data/2026-10-17_second_hfp_0604h \
  --flag f001 \
  --override \
  --by "mrgemsol (verified on second_hfp_0604h.mp4 frame 759; standing restore=lead:whole from second_hfp_0604g f001)" \
  --note "759-773: the lead's fur wrap, right arm and left hair/cheek edge pass behind the seated woman's chair inside her 16px margin (up to 15525 px on the node's plan); given back at assembly by restore=lead:whole on the table, as on second_hfp_0604g's f001 and second_hfp_0604d's f002."

<python> bench/capture_masked_run.py gate \
  data/2026-10-17_second_hfp_0604h

# Record the independent blind reader's look on the 406-frame plan strip:
<python> bench/capture_masked_run.py looked \
  data/2026-10-17_second_hfp_0604h \
  --by "a blind reader (subagent cb4bf919 of mrgemsol, given the strip and the text only), recorded by mrgemsol" \
  --loads second_hfp_0604h \
  --note "Every frame (406) of the plan strip of second_hfp_0604h: data/2026-10-17_second_hfp_0604h/look_blind/reading.md. Across all 406 frames (604-1009), the solid green mask and light-green margin form a continuous, uncarved dome over the seated woman's head, hair, neck, shoulders, and white apron bib down to the mixing bowl, with zero internal keep-out holes from held, hands, clothes, or hair_band. When lead crosses behind her chair on 759-773 (f001, overridden: given back at assembly by restore=lead:whole) and when the woman's own left hand rises to her mouth and brow on 988-1009, second's head and hair mask remains intact and hole-free for generation. The <Video 1> reference model tracks her head pose and speaking mouth movements smoothly across both profile turns (633-758 right, 774-937 left) and matches her eye state throughout the closing sequence (938-953 shut lids, 954-967 open eyes looking forward, and 968-1009 shut lids)."

# Rebuild frame_table.jsonl and record its sha256 in second_hfp_0604h.plan.built.json:
<python> bench/load_chain.py table \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol
```

### Step 7: `render` (`2026-10-17T21:13:48Z` built, `21:14:14Z` queued, by `mrgemsol`)

Writes `data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.render.go.md`, builds `data/2026-10-17_second_hfp_0604h_job/second_hfp_0604h.render.json` (`sha256: ed6b22382aff2102685ebbb26e1d6afc68a5b534f7649ce805892d8d3b4033f1`, differing from `second_hfp_0604h.plan.json` only in the song node's `preview` and `filename_prefix`), verifies all mask videos decode equal to the plan's, and queues the render on ComfyUI:

```bash
<python> bench/load_chain.py render \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol \
  --grounds data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.grounds.txt \
  --show data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.show.txt
```

### Step 8: `land` (`2026-10-17T21:50Z–21:52:55Z`, by `mrgemsol`)

Frees ComfyUI's resident models, runs the pose pass on the rendered video (`data/2026-10-17_second_hfp_0604h/second_hfp_0604h_render_points.npz`), and builds the stacked review clip `output/Video/fun/review_clips/second_hfp_0604h_0604_1009_stacked.mp4` (and four `.1fps.` clips covering `604–1009`):

```bash
<python> bench/load_chain.py land \
  data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json \
  --by mrgemsol
```

---

## 4. Windows, seam, and node settings

| setting | value | provenance / note |
|---|---|---|
| source span | `604–1009` (`406` frames, `16.90 s` at `24 fps`) | full kitchen shot `604–1009` |
| canvas | `1024x768` | source `1424x1080` centre-cropped to `1424x1068` at `(0, 6)`, scaled to `1024x768` |
| `render_region` | `"part"` (`zoomed in on what is replaced`) | subject with margin takes `1216x912` of `1024x768` (`< 1.25x`), so the node renders on the whole `1024x768` frame (`scale: 1.0`) |
| `replace` / `parts` | `["hair", "face_and_neck"]` | `parts_held__shot_c_0604b.mkv` (`Face_Neck`, `Hair`, `Eyeglass`, `Lower_Lip`, `Upper_Lip`, `Lower_Teeth`, `Upper_Teeth`, `Tongue`), held to `second`'s track + `8 px` |
| `others` (keep-outs) | `["lead", "strangers"]` | `lead` (`1003–1009` emptied) and `strangers`; zero self keep-outs (`held`, `hands`, `clothes`, `hair_band` retired) |
| `margin` / `feather` / `edge` | `64 px` / `8 px` / `"latent cells"` | `workflows/h3_config.py` defaults for `face and hair` |
| `composite` / `save_kept` | `"whole region"` / `true` | saves exact compositing matte into each window's `*_region.npz` |
| `seed` | `730451892` | `workflows/h3_config.py::SEED` |
| `window` / `context` / `context_noise` | `294` / `90` / `0.8` | 2 chained windows meeting at source frame `898`: `[294, 202]` in source frames (`808–1009`), `[294, 243]` as sampled (`202` source + `41` held past `1009` trimmed before write) |
| **Window 1** | `294` frames (`0.000s–12.250s`), writes **`604–897`** | `context: 0`; `16.65%` of video tokens regenerate; sampling `1,067.1 s` |
| **Window 2** | `202` source frames (`808–1009`) / `243` sampled frames (`8.500s–18.625s`), writes **`898–1009`** (`112` new frames) | keeps `90` frames (`808–897`, `27` latent steps) from Window 1 at `context_noise: 0.8` + writes `112` new frames (`898–1009`) = `202` source frames, plus holds last frame `41` frames past track end (`243` sampled, trimmed before write); `20.57%` of video tokens regenerate; sampling `786.6 s` |
| **Seam (`898`)** | Window 1 ends at `897`, Window 2 writes from `898` | Chosen (`mrgo` / `mrarc`, 2026-10-15) because on `894–901` she sits steadily in left profile at the table and the kept context `808–897` is calm, whereas `949` fell mid-turn to the camera (`945–952`) |

---

## 5. The five changes against the old kitchen pieces (`0604d`, `0949d`, `0604g`)

Both the old pieces (`second_hfp_0604d` on `604–948`, `second_hfp_0949d` on `949–998`, `second_hfp_0604g` on `999–1009`) and `second_hfp_0604h` (`604–1009`) read the same 2026-10-14 shot preview (`shot_c_0604b`). Five things changed between the old pieces and `second_hfp_0604h` (`C31`):

1. **`N2` — Self keep-outs (`held`, `hands`, `clothes`) and `hair_band` retired (`data/fun/plan.json` at `2026-10-17T14:13Z`)**:
   - In `second_hfp_0604d`, `held` (`shot_c_0604b_second_held_things_00001.mkv`), `hands` (`Left/Right_Hand` + `Left/Right_Lower/Upper_Arm`), and `clothes` (`Apparel` + `Upper_Clothing`), plus the `0.6` `hair_band` box (or `0604g`'s 9-frame `clothing` stay keep-out), were wired into `others` and carved the original video back into her region frame by frame as Sapiens per-frame part labels jittered along her blonde hair ends, neck, shoulders, and white apron straps.
   - In `second_hfp_0604h`, all self keep-outs and `hair_band` are gone (`others: ["lead", "strangers"]`), and `pieces=largest` is also retired so smaller face/hair islands are never dropped; instead, her clothing is named in the prompt (`"a white apron with a bib, its straps over her shoulders, over a sleeveless floral blouse"`), allowing H3 to generate a continuous, uncarved dome over her head, hair, neck, and apron bib.
2. **One load over the whole shot (`604–1009`) in two windows with context, against three separate pieces with hard joins**:
   - Previously, the kitchen shot was assembled from three separate loads (`second_hfp_0604d` on `604–948`, `second_hfp_0949d` on `949–998`, and `second_hfp_0604g` on `999–1009`) with hard piece boundaries at `948/949` (right in the middle of her head turn to the camera) and `998/999`.
   - `second_hfp_0604h` runs as a single 406-frame load across `604–1009` in two chained windows (`[294, 202]` in source frames: `90` context `808–897` + `112` written `898–1009`; `[294, 243]` as sampled with `41` held frames past `1009` trimmed before write) with a 90-frame context (`808–897` at `context_noise: 0.8`) and its seam placed at `898` while she sits calmly in left profile.
3. **The part bounded by her own track (`part_on_track`, `77035465`)**:
   - `second_hfp_0604d` loaded `shot_c_0604b_second_parts_00001.mkv` without clipping to her track.
   - `second_hfp_0604h` (like `0604g`) bounds `second`'s part mask to `second`'s own SAM 3.1 track grown by `8 px` (`shot_c_0604b_second_part_saved_on_track_8px.mkv`), and empties `lead`'s false track on `1003–1009` (`shot_c_0604b_lead_track_not_1003-1009.mkv`) where `lead`'s tracker jumped onto the seated mom after `lead` exited the frame at `998`.
4. **Laid by its own matte under one compositing law (`composite: "whole region"`, `bench/assemble_delivery.py::lay_plane`)**:
   - The old pieces were originally rendered or laid under `"only what changed"` (pixel-difference thresholding against the source), which left low-contrast patches of the original dark hair and skin inside the region.
   - `second_hfp_0604h` is rendered with `composite: "whole region"` and `save_kept: true`, and laid into the delivery cut by its own saved feathered region matte (`second_hfp_0604h_window_{1,2}_region.npz`), with `restore=lead:whole` on `759–773` where the lead walks behind her chair.
5. **The mouth drawn by the gated class map and her eyes shut drawn as closed lid lines (`lids`)**:
   - In `second_hfp_0604d`, the mouth was left out of the prompt (`parts_left_out: ["mouth"]`, `0604d.json` line 58) and her eyes were drawn open across the entire span (`604–948`), with no `CLASS_MOUTH_GATE` or temporal hysteresis (`MOUTH_RUNS`).
   - In `second_hfp_0604h`, her reference (`second_0604_1009_b`) draws the mouth on `323` of `406` frames (`51` with open inside, gated frame by frame by `CLASS_MOUTH_GATE` and smoothed by `MOUTH_RUNS` so off-body mouth blobs on `645–650`, `652–659`, `681–686`, `725–733`, and `745–747` are suppressed) and draws her eyes shut as single closed lines (`form: "lids"`) on `938–953` and `968–1009`, matching both her head turn and her bow at the end of the shot, with both behaviours described in the generated text.

---

## 6. Where the frozen bundle lives

The frozen data bundle is stored (read-only, `chmod a-w`, regular file copies only, no symlinks) in:

- **`data/fun/recipes/second_hfp_0604h/`**
- **Bundle manifest**: `data/fun/recipes/second_hfp_0604h/manifest.json`

### Files copied into `data/fun/recipes/second_hfp_0604h/`

| file in `data/fun/recipes/second_hfp_0604h/` | role | sha256 |
|---|---|---|
| `kitchen_inputs/second_hfp_0604h.json` | job file (`data/2026-10-17_fun_kitchen_inputs/second_hfp_0604h.json`) | `d2c3c974f956308f505a0d4e566574e68f62317ca1eeb3012db579aff7d46247` |
| `kitchen_inputs/second_hfp_0604h.txt` | generated prompt text (`2,164` bytes) | `96f2c02d40250a591750e5face1a2290ef68149bbb9ec3eec588036d6e0fb6e1` |
| `kitchen_inputs/second_hfp_0604h.grounds.txt` | render grounds text | `c2ac51b15f18843e5c22b4472d876095fb36cd4cfddfc2ea3dac4191a308244b` |
| `kitchen_inputs/second_hfp_0604h.render.go.md` | pre-render go record | `5ae136f7a610ac492ea7578e1637086f89678d2aa6dca477ff0c94e1eb92e948` |
| `kitchen_inputs/second_hfp_0604h.show.txt` | render show text | `6403bdb3c0350303a1dab2e714591818cf5fe4907bfb0467d3cee4685fa055c5` |
| `job/job.plan.json` | job file snapshot at plan build | `d2c3c974f956308f505a0d4e566574e68f62317ca1eeb3012db579aff7d46247` |
| `job/job.render.json` | job file snapshot at render build | `d2c3c974f956308f505a0d4e566574e68f62317ca1eeb3012db579aff7d46247` |
| `job/second_hfp_0604h.plan.built.json` | plan build record | `a78b6f6998deb801cd5021cad47c62a47e91b31c574e38e8ee2bd37d5f2d58c2` |
| `job/second_hfp_0604h.plan.json` | ComfyUI plan graph as queued | `1154d58b6b9f6fd142ccc93edea94cc1a8e2dd25271102059a6cd7a398744efe` |
| `job/second_hfp_0604h.plan.masks.json` | on-plan mask per-frame digest record | `bbc8efab6be1e3f023faa98dc1ba2cb07f75627244f0e0de00d2dbae5876b5ee` |
| `job/second_hfp_0604h.plan.queued.json` | plan queue and history record | `402314bacaaa9adc34267ed6889f4d80bd5b17a551abee953fd1540f46fb6310` |
| `job/second_hfp_0604h.plan.txt` | plan text snapshot | `96f2c02d40250a591750e5face1a2290ef68149bbb9ec3eec588036d6e0fb6e1` |
| `job/second_hfp_0604h.render.build_printed.txt` | builder printed output for render | `9d47d6889428c82bd1d4a4c84f4db7efff800c169378dabfaa46c4a83225d014` |
| `job/second_hfp_0604h.render.built.json` | render build record | `ce95e177b1850df5aa735f442fd59d2ba55a576cb38c4b8cdab09f53c0221174` |
| `job/second_hfp_0604h.render.chain.json` | render chain pre-queue verification record | `578f037170829de5d824528231bee75fb60736ae290a9b645dec30dba5e98211` |
| `job/second_hfp_0604h.render.json` | ComfyUI render graph as queued | `ed6b22382aff2102685ebbb26e1d6afc68a5b534f7649ce805892d8d3b4033f1` |
| `job/second_hfp_0604h.render.queued.json` | render queue, digests, and stage timing record | `5b7f38fa19ce439ff191375f0d4c14b3231d895c24337a7954c36ee213091166` |
| `job/second_hfp_0604h.render.txt` | render text snapshot | `96f2c02d40250a591750e5face1a2290ef68149bbb9ec3eec588036d6e0fb6e1` |
| `job/second_hfp_0604h.who_is_who.json` | shot-table `who_is_who` record | `1606b81423f6ba78ac69ce1ffa83b3bf0cb4425c25a734ef7c64a15ad63fd6e7` |
| `preview_job_shot_c_0604b/job.json` | preview job file (`data/2026-10-14_shot_c_0604b_job/job.json`) | `e4c957ad5559e4d5ad81326b7b4dbad3554b4cd7b5e9a28078f973ae0b964907` |
| `preview_job_shot_c_0604b/preview.built.json` | preview built record (`data/2026-10-14_shot_c_0604b_job/preview.built.json`) | `f13e3f51ce55587b409b41be048748045458aec79f7019191c940dad800724dc` |
| `preview_job_shot_c_0604b/preview.json` | preview ComfyUI graph (`data/2026-10-14_shot_c_0604b_job/preview.json`) | `0b03b411c7d2ee613452aa3e76de6e592548a6028e37eb4bcca31f6c20e7728b` |
| `preview_job_shot_c_0604b/preview.queued.json` | preview queued record (`data/2026-10-14_shot_c_0604b_job/preview.queued.json`) | `e1d154d05a5dee42a46b0b50985a6df7a64dd828c4ecca3f8da30276e6046d19` |
| `plan.2026-10-17T141409Z.json` | `data/fun/plan.json` at render time (`2026-10-17T14:14:09Z`, saved as `data/fun/plan.before_no_stubble_2026-10-18T1630Z.json`) | `8f746dd9e5d6252bbe2e32f1a898f694715ee2d3e6596edc1db723560563cadd` |
| `plan.json` | identical copy of `plan.2026-10-17T141409Z.json` (`data/fun/plan.before_no_stubble_2026-10-18T1630Z.json`) | `8f746dd9e5d6252bbe2e32f1a898f694715ee2d3e6596edc1db723560563cadd` |
| `plan.before_n2_keepouts_2026-10-17T1413Z.json` | `data/fun/plan.json` snapshot immediately before the `N2` edit | `522dfc1d0f1d8f1b865a299cc06484c7af6ded89bb97e6f9e6e6887603a37dc0` |
| `reference/second_head_reference_0604_1009.json` | head reference record (`second_0604_1009_b`) | `24fed98104290abb2783343a1843a217f1826b30314e9f22cecc378be7f8bf6a` |
| `reference/second_0604_1009_points.json` | pose pass points metadata in `second`'s tracked box (`406` frames) | `dae660bc91ff2085fde5db7f1285b43b2fb2cc8c8e00165f7b994250784a3387` |
| `reference/second_0604_1009_points.npz` | pose pass points arrays in `second`'s tracked box (`406` frames) | `1e40c8c5b301ae1a426674eca5cfc49b752345ad5e70d230e79836f01d1d15f8` |
| `reference/second_head_reference_0604_1009_points_filled.json` | gap-filled head reference face points metadata | `daa9499d4c6a9edb136fd1e5666e0ad76e31f44ec646fa91c00078578b897b12` |
| `reference/second_head_reference_0604_1009_points_filled.npz` | gap-filled head reference face points arrays | `a73c296ff2e000caf853f8ad5d425ca49830e208336394556322141057035612` |
| `reference/second_head_reference_0604_1009_plain_cut.json` | plain-cut body model cut record | `257550f843125926e37b9685c3b5927bc2ab78d01f7ff909750c401d022f8f73` |
| `reference/second_head_reference_0604_1009_plain_cut_points.npz` | plain-cut body model points arrays | `0ee30241575db4fa993be808534a57b928b6891e367491ceb2263f3d8461de84` |
| `reference/second_head_reference_0604_1009_source_cut.json` | source-cut record | `94ac746d7f244d57d026055c311b8338da6500b68d9cfc1f74c5973115dc1deb` |
| `reference/second_head_reference_0604_1009_source_cut_points.npz` | source-cut points arrays | `0ee30241575db4fa993be808534a57b928b6891e367491ceb2263f3d8461de84` |
| `capture/manifest.json` | capture manifest (`data/2026-10-17_second_hfp_0604h/manifest.json`) | `1862a8a034de5a93f3239d64dce5e7f7398b3f3c5c98b8b5db0d0b6730461ad7` |
| `capture/verify.json` | capture verification record (`0` hard failures) | `0789b0d429550a8f1a6c32d6b929f2d3de19a3aecde490bb02910101b1f87fb1` |
| `capture/flags.json` | capture gate flags (`24` flags on-plan) | `1fafc12757481466f3af985118c73d69081c19f1b20f71a846aa7bf6f7a9ad5b` |
| `capture/outcomes.json` | capture flag outcome override (`f001`) | `2774fb9025151815207c83065a8f1fcb24f9abd482e7013b9fba2a768140dbeb` |
| `capture/looks.json` | capture blind reader look record | `da1f4ebcfaa378a97abe573220bfb2370ee3c38ac6436156ae209bba682884a2` |
| `capture/status.json` | capture status record | `e615e8ed6de2bd22b863c24b67ad5772a3f28c9f76e2cc6defb2b638eac093b3` |
| `capture/rules_input.json` | capture rules input record | `b59b7857d7e274118cb094ddf9068f8c691a72aa084957f4243949635f9a0180` |
| `capture/delivery__cand_0604_1009.json` | capture delivery candidate check record | `166514c5d5c02db09131e650542ca821c66f460bd3ed4270abccfffdc41d4386` |
| `capture/who_is_who.json` | capture `who_is_who` record | `1606b81423f6ba78ac69ce1ffa83b3bf0cb4425c25a734ef7c64a15ad63fd6e7` |
| `capture/second_hfp_0604h_render_points.json` | landed render pose points metadata (`406` frames) | `84290a44e163898ece13dc62903bac04cc93e1090eb32dfc3357c786e20a5d28` |
| `capture/second_hfp_0604h_render_points.npz` | landed render pose points arrays (`406` frames) | `7c24ddfd3dea5df329690a0440960b5f9eab2474418cfc906ea1abe6146b9727` |
| `capture/second_hfp_0604h_plan.json` | node's planned windows file (`runs/second_hfp_0604h/predicted/second_hfp_0604h_plan.json`) | `d1892a96010d6748885f4ee0d89a8506483a8bafe4ceedccc744b07e15736d41` |

### Large media, masks, and window files recorded by `sha256` in `data/fun/recipes/second_hfp_0604h/manifest.json`

| relative path (under repo or share `input/` / `output/`) | role | sha256 |
|---|---|---|
| `input/mrpop/fun_0000_6394_24p_sound_in_step.mov` | `source_video` (`2,759,232,567` bytes) | `9bd4451fe86e9e3c8541c81d9297c5d26f0cc0c7990323b78804ba8c69742048` |
| `input/fun_mom_edit.jpg` | `identity_image` (`725,801` bytes) | `11f68ec9b9c475859aa5fac9e708b29ffee8af2ccc3a66a43f80dc8d112e1bac` |
| `data/2026-10-17_second_hfp_0604h/subjects/second/track__shot_c_0604b.mkv` | `capture_second_track_mkv` (`563,159` bytes) | `7d1cce4aebad7da58ac30e048ea951084aa211cc5e1820b592e92ebc0517989b` |
| `data/2026-10-17_second_hfp_0604h/subjects/second/parts__shot_c_0604b.mkv` | `capture_second_parts_mkv` (`415,568` bytes) | `286b2d4ada0980ba251c179eb8987a8ec405b2f874db3c328fabed1868e5c61d` |
| `data/2026-10-17_second_hfp_0604h/subjects/second/parts_held__shot_c_0604b.mkv` | `capture_second_parts_held_mkv` (`415,568` bytes) | `ff8e79761de336364c1f3d3ff9a353df7daca3cb32180f73d1463c9e7cb92541` |
| `data/2026-10-17_second_hfp_0604h/subjects/lead/track__shot_c_0604b.mkv` | `capture_lead_track_mkv` (`604,035` bytes) | `acbc4f275b4cb45467d1c6e957a603156bb9a316e6016dc8de0964e3cea040fc` |
| `data/2026-10-17_second_hfp_0604h/subjects/strangers/track__shot_c_0604b.mkv` | `capture_strangers_track_mkv` (`233,722` bytes) | `035dcd8e7c680d2d8d4c44df7b72f9f07040e3512e375bbc84a85ee95f455be5` |
| `data/2026-10-17_second_hfp_0604h/runs/second_hfp_0604h/carried.mkv` | `capture_run_carried_mkv` (`415,568` bytes) | `9524b453cb5058796740aca9fa244231cd7b0619849b4c8d93208bf4fe22b37b` |
| `data/2026-10-17_second_hfp_0604h/runs/second_hfp_0604h/region.mkv` | `capture_run_region_mkv` (`276,421` bytes) | `e98c3e2a39f8b27390d1fe143791ddcc10ed0627e02b8ee8ab54ef700b704eba` |
| `data/2026-10-17_second_hfp_0604h/runs/second_hfp_0604h/region.npz` | `capture_run_region_npz` (`424,066` bytes) | `8efbfda06dd1b3674a8463f22e7879d9bffc6e8bdb2b482ee501664025ddf09a` |
| `data/2026-10-17_second_hfp_0604h/runs/second_hfp_0604h/predicted/second_hfp_0604h_window_1_planned_region.npz` | `capture_predicted_w1_region_npz` (`280,465` bytes) | `abf3cb802f1f810da632e1b0d80640d25010f573511f86fd7c7cd2797a84b0ea` |
| `data/2026-10-17_second_hfp_0604h/runs/second_hfp_0604h/predicted/second_hfp_0604h_window_2_planned_region.npz` | `capture_predicted_w2_region_npz` (`289,492` bytes) | `8a00e801007c05e0512043a27a8693ea42e3fd414185f0bf0d91ad3be4fc70b1` |
| `output/Video/fun/keep_outs/shot_c_0604b/shot_c_0604b_second_clothes.mkv` | `share_keepout_second_clothes_mkv` (`625,932` bytes) | `d5a844f6e4429551c50b48a8e634ab7a110c9bfb2e45c1ff5de83922e4535095` |
| `output/Video/fun/keep_outs/shot_c_0604b/shot_c_0604b_second_hands.mkv` | `share_keepout_second_hands_mkv` (`449,645` bytes) | `ef906043e19bfd55498ee9c968558184c522a907defb4300bc4b7509049553dc` |
| `output/Video/fun/keep_outs/shot_c_0604b/shot_c_0604b_strangers.mkv` | `share_keepout_strangers_mkv` (`234,122` bytes) | `96c299f56f52ef37d55a356ef9da158cbd36cb380cf96c0da67398953da207c0` |
| `output/Video/fun/references/second_0604_1009_b/second_head_reference_0604_1009.mp4` | `reference_video` (`5,549,840` bytes) | `80a841d8f39494e1d1a95325f4ac8232ccdbc8a25f5ecc551ebc4693f279ed13` |
| `output/Video/fun/references/second_0604_1009_b/second_head_reference_0604_1009_plain_cut.mp4` | `reference_plain_cut_video` (`5,351,913` bytes) | `a3276294fbac35eeb85d4732f242dc5579b5fb8d5d430793bb33aec14af8aa63` |
| `output/Video/fun/references/second_0604_1009_b/second_head_reference_0604_1009_source_cut.mp4` | `reference_source_cut_video` (`18,755,010` bytes) | `e447aec4ae3e6981bf63e6def2d3406bd4023300333225941e5bf95072766389` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_00001.mp4` | `render_mp4` (`15,215,506` bytes) | `091585a719389062ff1ecaa62feff8bfc31b0bb2c861d8e5230959fa67938ad6` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_00001_with_mask.mp4` | `render_with_mask_mp4` (`29,414,442` bytes) | `c83efa8d8c089046d550cc53d0a3a32a19fd2764fccce8b56b12b7412d85f83a` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_1.mp4` | `window_1_mp4` (`10,268,095` bytes) | `24d93dfc61a325acbbdfd309b2ae4739094eceb761be3233db5d18088f9a6617` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_1.safetensors` | `window_1_safetensors` (`25,783,040` bytes) | `c1c629df38b60e2ad83fa6a02d8daa63325d5d3e95128bec5d2991bb19fdb754` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_1_region.npz` | `window_1_region_npz` (`3,281,729` bytes) | `e69f412f6e8ced1e1f84004d6705b198d814b0015030a93866c53620d893c9fe` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_1_with_mask.mp4` | `window_1_with_mask_mp4` (`20,215,993` bytes) | `680fd3a2da3e6756bfaedf2c5d52d6cd52014afce90f1529305c123eab11eb6e` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_2.mp4` | `window_2_mp4` (`4,397,086` bytes) | `5bccbb9a5c757dd07e2f76c4d6b42ffc56d35a6d815fe183f621067afa22b3b9` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_2.safetensors` | `window_2_safetensors` (`21,337,600` bytes) | `d844f0baba22c3a341bd06e5394c111c8d838ada46faaacb5668ae741c615c5c` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_2_region.npz` | `window_2_region_npz` (`2,994,008` bytes) | `5b754b2546a5313b8e6855a0c27933aec45bc008082b8d3a640cc6f0856c8c0a` |
| `output/Video/fun/mrgemsol/second_hfp_0604h/second_hfp_0604h/second_hfp_0604h_windows/second_hfp_0604h_window_2_with_mask.mp4` | `window_2_with_mask_mp4` (`8,648,298` bytes) | `92ada3f562fa1339eef9a5dc3b64abdb096ebd44b7c7e0d41ff8a94d9851b8e3` |
