# Self-Mask Sensor: Controls, Harvest Step, One Head, a Second Reading, and Two People (`2026-11-27`)

Six read-only probe passes and thirteen routed renders asking why the `text_phrase` self-mask lags a moving
subject, and what fixes it. Plain-English summary and what to build next:
[`docs/wiki/self_attention.md`](../../docs/wiki/self_attention.md).

**Short version.** On a small figure crossing a locked-off wide shot, the shipped reading (all heads, blocks
`35 + 40`, first sampler step) does not follow the figure. Subtracting a control map and judging frames against
the whole clip do not fix it. Reading two or three steps later does, but routing that starts that late no longer
changes the figure. One head at block `45` (head `43` of `56`) already marks the figure at the first step, and it
was the best head probed on three clips it was not chosen on. Reading the mask once more on the first routed
pass keeps it on her for the whole walk (section 8). With two people in frame that head marks both, and the
region prompt, not the target words, decides which one changes (section 9).

## 1. Conditions

- **Card and code:** `RTX 4090` (`24 GB`); pack working tree between `19a47212` and `cf53cb9c` (the
  `text_phrase` harvest path in [`steering/live_policy.py`](../../steering/live_policy.py) is identical at both);
  ComfyUI core `7c78b5641`. Run in-process after `POST /free` on an idle `:8188`.
- **Recipe (all runs):** `minimax_h3_fl2va_pruned_int8_convrot` + Pika H3 Flex LoRA (`MiniMaxH3LoRABranch`,
  strength `1.0`) + `MiniMaxH3SigmaShift(12.0, 3.0)` + `comfy kitchen attention` + `MiniMaxH3Sol`; `lcm`, six
  steps on `ManualSigmas "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"`; text-to-video (no frame
  wired), `1344x768`, `141f`. Prompt, seed `730451892`, and router settings copied from the Gemini session's
  scratch runner for the `2026-11-26` Track 2 arms `c0_walk_base` / `c2_walk_woman_coat_replay_e024`
  ([`2026-11-26_h3_4sensor_calibration.md`](2026-11-26_h3_4sensor_calibration.md) section 6).
- **Reading:** [`bench/probe_h3_sensors.py`](../probe_h3_sensors.py) attached last, unmodified. Per-head and
  single-head variants replace one method on the recorder or the harvester in the probe process only; no repo
  file was changed.
- **Builds for sections 8 and 9:** the arms there need `harvest_heads` and `reharvest` on
  `attach_self_mask_router`, so they ran from a worktree build of the pack: `f1`, `f2` and the two-person probe on
  `488e3a8b`; `h1..h3` and `i1..i3` on `66605ac8` (the same plus a `harvest_contrast` option that `h3` uses and
  that was not landed; kept on branch `claude-self-mask-contrast`); `j1` on the rebased build that landed as `1368f40d`.
- **Scripts, prompts, arrays, metrics (untracked):** `data/self_mask_probe_2026-11-27/` (`run_control_probe.py`,
  `run_late_harvest.py`, `run_head43.py`, `run_reharvest_two_subjects.py`, the `analyze_*.py` files,
  `two_people.py`, `prompts/`, `*.npz`, `*.json`).
- **Clips, sheets, reports:** `$H3_OUTPUT_DIR/Video/self_mask_late_harvest_2026-11-27/`
  (`compare_harvest_step_and_head_grid.mp4` stacks every walk arm with its mask;
  `compare_harvest_step_and_head_sheet.png`, `control_words_sheet.png` and `two_people_sheet.png` are the stills).

### How each number is measured

- **Where the subject is.** The camera is locked off, so the per-pixel median over time is the empty scene; pixels
  that differ from it are the subject and its cast shadow (`data/self_mask_probe_2026-11-27/analyze_control_probe.py::coat_patch_truth`, largest
  connected blob per latent step). Each routed arm is scored against the subject in its own clip.
- **`covered`**: share of the subject's patches under the mask. **`on subject`**: share of the mask's mass on the
  subject's patches grown by one patch. The subject is about `4%` of the frame in every clip here, so `0.04` on
  subject is chance.
- **`band AUC`**: inside the rows the subject ever occupies, how well the raw map separates subject patches from
  the rest, per latent frame (`0.5` is chance). A map that is merely high near the ground earns nothing.
- **`peak hit`**: share of latent frames where the map's strongest column inside that band is within a tenth of
  the frame width of the subject.
- **`coat-colored`**: share of the walker's pixels with the coat's red; the base clip is the reference, lower
  means transformed.
- **early / late**: latent steps starting before frame `72` / from frame `72` on. Masks use the shipped
  `synthesize_4d_attention_mask` at `threshold_z = 1.80` unless a row says otherwise.

## 2. This probe reproduces the masks the walk arms harvested

Rebuilding the mask from the probe's own maps gives the same `coverage_mean`, `hard_coverage_mean`, `peak_xy`,
`centroid_xy` and `bbox_xyxy` as both run reports, to every printed digit, for `c1` (`"woman"`, `1.35`) and `c2`
(`"woman, crimson, coat"`, `1.80`): `data/self_mask_probe_2026-11-27/metrics.json::reproduction`.

## 3. First step, shipped blocks: two fixes that did not work

Three target words, step `0`, blocks `35 + 40`, scored on `c0_walk_base`.

| Map | Whole-frame AUC early / late | On subject early / late | Covered early / late |
| :--- | :-: | :-: | :-: |
| As shipped | `0.621` / `0.526` | `0.129` / `0.052` | `0.344` / `0.054` |
| Minus the mean of all text rows | `0.541` / `0.441` | `0.091` / `0.016` | `0.139` / `0.009` |
| Minus eight function words | `0.490` / `0.435` | `0.058` / `0.007` | `0.033` / `0.001` |
| Minus nine scene words | `0.611` / `0.524` | `0.141` / `0.064` | `0.425` / `0.097` |
| As shipped, frames judged against the whole clip | not applicable | `0.132` / `0.054` | `0.304` / `0.060` |

- **Control subtraction makes the first-step mask worse.** The all-text map does not light the arch openings the
  target map lights (`control_words_sheet.png`, rows 1 and 2), so there is nothing generic to take out.
- **Judging against the whole clip changes nothing.**
- **The mask is offset as well as late:** its centroid sits at `y = 0.654` while the walker is centered near
  `0.83`.

## 4. The same words follow her two steps later

Three target words, blocks `35 + 40`, inside her band, on `c0_walk_base`.

| Sampler step | Band AUC early / late | Peak hit | Minus all text: band AUC early / late | Minus all text: peak hit |
| :-: | :-: | :-: | :-: | :-: |
| `0` | `0.594` / `0.491` | `0.375` | `0.597` / `0.493` | `0.400` |
| `1` | `0.733` / `0.626` | `0.750` | `0.757` / `0.647` | `0.700` |
| `2` | `0.742` / `0.794` | `0.850` | `0.827` / `0.799` | `0.925` |
| `3` | `0.809` / `0.754` | `1.000` | `0.859` / `0.790` | `1.000` |
| `4` | `0.774` / `0.715` | `0.950` | `0.839` / `0.768` | `1.000` |
| `5` | `0.789` / `0.711` | `0.950` | `0.852` / `0.749` | `1.000` |

- The all-text map alone never follows her at any step (band AUC `0.46..0.60`, peak hit `0.15..0.40`), so it is a
  fair control, and from step `2` on subtracting it helps a little.
- **By word.** At step `0` no word follows her (band AUC `0.39..0.68` for all `31` words read). By step `2` the
  words about the subject and what she does all do (`woman` `0.706 / 0.786`, `walks` `0.806 / 0.795`, `footsteps`
  `0.813 / 0.765`, `lone` `0.796 / 0.756`), and the scene words point away from her (`arches` `0.213 / 0.193`,
  `limestone` `0.133 / 0.131`, `stones` `0.158 / 0.135`). The subject noun is not the best locator among them.
  (`data/self_mask_probe_2026-11-27/tracking.json`.)

## 5. A later harvest gives a better mask and a weaker edit

Router settings as `c2` (`"woman, crimson, coat"`, `1.80`, `early_0_24_shield`, `replay_step0 = True`, chrome
region prompt). Only `start_percent` changes, which moves the step the router harvests and first routes on.
Each arm scored in its own clip.

| Arm | Covered early / late | On subject early / late | Mask `y` | Coat-colored at `f24 f48 f72 f96 f120` |
| :--- | :-: | :-: | :-: | :--- |
| `c0_walk_base` (no routing) | not applicable | not applicable | not applicable | `0.488 0.580 0.667 0.571 0.603` |
| `c2` harvest at step `0`, as shipped | `0.471` / `0.069` | `0.275` / `0.085` | `0.654` | `0.331 0.393 0.461 0.669 0.690` |
| `e1` harvest at step `0`, block `45` head `43` only | `0.705` / `0.500` | `0.438` / `0.447` | `0.872` | `0.395 0.009 0.015 0.483 0.617` |
| `d1` harvest at step `1` (`start_percent = 0.15`) | `0.745` / `0.462` | `0.407` / `0.460` | `0.799` | `0.036 0.235 0.274 0.351 0.484` |
| `d2` harvest at step `2` (`start_percent = 0.30`) | `0.785` / `0.908` | `0.339` / `0.498` | `0.825` | `0.456 0.619 0.660 0.583 0.580` |
| `d3` harvest at step `3` (`start_percent = 0.45`) | `0.864` / `0.649` | `0.448` / `0.464` | `0.802` | `0.347 0.473 0.570 0.530 0.519` |

- **`d2` and `d3`: the mask is on her and she stays a woman in a crimson coat.** Routing that begins at step `2`
  or later does not change her look with this isolation mode.
- **`d1` is the only late-harvest arm that keeps a partial effect across the walk.**
- **`e1` gives the strongest effect of any arm through `f72`** (whole figure, not only head and shoulders), then
  she reverts. In `c2` and `e1` she ends farther right than in the base clip (`x = 0.88` at `f120` against
  `0.79`): routing changes how fast she walks, and the mask was read from an unrouted pass, so it trails her.
- Read from stills (`compare_harvest_step_and_head_sheet.png`); not judged on playback.

## 6. One head at block `45` marks the subject at the first step

On `c0_walk_base`, step `0`, masks from different head choices (`data/self_mask_probe_2026-11-27/analyze_blocks.py`, `data/self_mask_probe_2026-11-27/analyze_heads.py`).

| Mask from | Covered early / late | On subject early / late | Mask `y` |
| :--- | :-: | :-: | :-: |
| Blocks `35 + 40`, all heads (shipped) | `0.344` / `0.054` | `0.129` / `0.052` | `0.654` |
| Block `40`, all heads | `0.425` / `0.092` | `0.181` / `0.101` | `0.691` |
| Block `45`, all heads | `0.009` / `0.002` | `0.072` / `0.026` | `0.663` |
| Block `45`, head `43` only | `0.952` / `0.925` | `0.437` / `0.491` | `0.871` |

- **The all-head average at block `45` yields almost no mask**, which agrees with the earlier record's "periodic
  horizontal RoPE stripes" at `S0 B45` ([`2026-11-26_h3_4sensor_calibration.md`](2026-11-26_h3_4sensor_calibration.md)
  section 2). The clean signal is inside that block, in one head.
- **At blocks `30` and `35` no head follows her at step `0`.** Heads that score well on the early half score badly
  on the late half (correlation `-0.88` and `-0.60` across the `56` heads): they prefer a side of the frame.
- **Picking heads beats random sets only where heads agree across the two halves.** Eight heads picked on one half
  and scored on the other, against `200` random sets of eight: no gain at step `0` at any block read; a gain
  beyond the random sets' 95th percentile at, for example, block `45` from step `2` and block `30` at step `3`
  (`data/self_mask_probe_2026-11-27/heads.json`).

### Held out: three clips the head was not chosen on

Head `43` was chosen on `c0_walk_base`. Step `0`, per-head readings at blocks `35, 40, 43..48`.

| Clip | Mask from | Covered first / second half | On subject first / second half | Band AUC |
| :--- | :--- | :-: | :-: | :-: |
| `g0` walk prompt, seed `20261127` | shipped `35 + 40` | `0.513` / `0.132` | `0.205` / `0.116` | `0.607` |
| | block `45` head `43` | `0.994` / `0.946` | `0.369` / `0.376` | `0.839` |
| | block `45` heads `43, 40, 34` | `0.963` / `0.858` | `0.485` / `0.601` | `0.860` |
| `g1` man, yellow raincoat, right to left | shipped `35 + 40` | `0.328` / `0.190` | `0.300` / `0.192` | `0.580` |
| | block `45` head `43` | `0.889` / `0.869` | `0.647` / `0.592` | `0.813` |
| | block `45` heads `43, 40, 34` | `0.926` / `0.796` | `0.535` / `0.591` | `0.831` |
| `g2` child, green jacket, gallery, right to left | shipped `35 + 40` | `0.148` / `0.045` | `0.229` / `0.072` | `0.566` |
| | block `45` head `43` | `0.721` / `0.991` | `0.461` / `0.418` | `0.842` |
| | block `45` heads `43, 40, 34` | `0.739` / `0.990` | `0.609` / `0.595` | `0.872` |

- On all three, head `43` at block `45` ranks first of `56` at its block and is the best single head across all
  eight blocks read (`data/self_mask_probe_2026-11-27/generalize.json`).
- Two of the three subjects move right to left, so it is not a left-to-right habit.

### It follows the subject more than the words

Block `45`, head `43`, on `c0_walk_base` (`data/self_mask_probe_2026-11-27/analyze_h43_words.py`).

| Step | Rows scored against | Band AUC | Covered | On subject |
| :-: | :--- | :-: | :-: | :-: |
| `0` | the three target rows | `0.818` | `0.936` | `0.467` |
| `0` | the mean of all text rows | `0.789` | `0.920` | `0.415` |
| `0` | the target rows' share among all text rows | `0.753` | `0.861` | `0.464` |
| `2` | the three target rows | `0.892` | `0.976` | `0.512` |
| `2` | the mean of all text rows | `0.845` | `0.911` | `0.569` |

The head marks her almost as well against text in general as against her own words. On a clip with one subject
that is a good mask. Whether it can tell two subjects apart by word is not tested here.

## 8. Reading the mask again on the routed pass

Walk prompt and seed, router settings as `c2`, harvest from block `45` head `43`. `reharvest` reads the same head
again while the first routed pass runs (peek only) and uses that mask from the next step. Each arm scored in its
own clip; the mask columns describe the last mask the run held.

| Arm | `reharvest` | Covered early / late | On subject early / late | Mask `y` | Coat-colored at `f24 f48 f72 f96 f120` |
| :--- | :--- | :-: | :-: | :-: | :--- |
| `e1` | `off` | `0.705` / `0.500` | `0.438` / `0.447` | `0.872` | `0.395 0.009 0.015 0.483 0.617` |
| `f1` | `first_routed_step` | `0.907` / `0.878` | `0.441` / `0.478` | `0.863` | `0.015 0.016 0.010 0.003 0.005` |
| `f2` | `every_step` | `1.000` / `0.983` | `0.416` / `0.472` | `0.651` | `0.000 0.000 0.000 0.000 0.004` |

- **One extra reading is enough.** In `f1` she is changed on every frame measured, where `e1` reverts from `f84`.
- **Reading on every pass also holds her, with a larger mask that sits higher** (coverage `0.137` / `0.156`
  against `0.104` / `0.124` for `f1`). `first_routed_step` is the tighter of the two.
- **The second reading is free at this scale:** sampling took `75.4 s` (`e1`), `79.6 s` (`f1`) and `74.0 s` (`f2`).
- **`j1` repeats `f1` on the build that landed** (rebased onto `eb02b2c6`): every figure in the `f1` row came out the same to the printed digit
  (`data/self_mask_probe_2026-11-27/late_harvest_metrics.json`).

## 9. Two people in frame

Prompt `data/self_mask_probe_2026-11-27/prompts/p3_courtyard_woman_and_man_crossing.txt`: a woman in a crimson
coat walks left to right while a man in a yellow raincoat walks right to left; they pass mid-frame. Each walker is
tracked from the side they enter on (`data/self_mask_probe_2026-11-27/two_people.py`), so the reading does not
depend on how they look after an edit. Steps where they overlap are left out.

### The head finds people; other heads tell them apart

Unrouted base clip, step `0` (`data/self_mask_probe_2026-11-27/two_people_heads.json`). `her vs him` asks whether
the map for her words minus the map for his words ranks her patches above his (`0.5` is chance).

| Mask from | Woman covered | Man covered |
| :--- | :-: | :-: |
| Shipped `35 + 40`, all heads, her words | `0.211` | `0.020` |
| Block `45` head `43`, her words | `0.827` | `0.744` |
| Block `45` head `43`, his words | `0.833` | `0.855` |
| Block `45` head `43`, her words minus all text | `0.615` | `0.145` |
| Both must agree: block `45` head `43` (her words) and block `46` head `13` (her minus his) | `0.549` | `0.000` |

- **Head `43` marks both people whichever words it is read against** (people-vs-background AUC `0.765`, first at
  its block; her-vs-him `0.719`).
- **Heads that tell them apart by word exist and are different heads:** block `46` head `13` (`0.914`), block `48`
  head `3` (`0.897`), block `48` head `29` (`0.885`), block `47` head `17` (`0.873`). They are poor at finding
  people at all (`0.58..0.62`). **They were picked on this clip and are not validated on another.**

### In the renders, the region prompt picks who changes

All arms: block `45` head `43`, `reharvest = "first_routed_step"`, unless the row says otherwise. The region
prompt is the chrome figure "striding briskly from left to right" (her direction) or the same text with "from
right to left" (his). `still red` / `still yellow` are means over the clean steps; the base clip is the reference.

| Arm | Target words | Region prompt says | Woman still red | Man still yellow | Last mask covers woman / man |
| :--- | :--- | :--- | :-: | :-: | :-: |
| base clip | not applicable | not applicable | `0.672` | `0.542` | not applicable |
| `h1` shipped reading, no second reading | hers | left to right | `0.520` | `0.540` | `0.195` / `0.025` |
| `h2` | hers | left to right | `0.010` | `0.486` | `0.842` / `0.644` |
| `i1` | his | left to right | `0.123` | `0.497` | `0.877` / `0.705` |
| `i2` | his | right to left | `0.429` | `0.040` | `0.707` / `0.833` |
| `i3` | hers | right to left | `0.356` | `0.067` | `0.727` / `0.775` |
| `h3` her words minus all text | hers | left to right | `0.175` | `0.420` | `0.646` / `0.327` |

- **The mask covers both people in every single-head arm, and one person changes.** Which one follows the
  direction named in the region prompt (`h2` and `i1` change her; `i2` and `i3` change him) and not the target
  words (`i1` used his words and changed her; `i3` used hers and changed him).
- **`h2` is the cleanest:** she is changed for the whole clip and he is close to the base clip.
- **With the region prompt pointing his way, her coat reads less red than in the base clip** though on the stills
  she is still the woman in the crimson coat (`two_people_sheet.png`); the mask is on her too, so some spill is
  likely.
- **The contrast against all text (`h3`) narrowed the mask and weakened the edit on her without sparing him
  more,** so that option was not landed.

## 10. What these runs cannot say

- **One kind of shot:** one or two small figures crossing a locked-off wide frame, one checkpoint and LoRA, one
  canvas and length. No still subjects, no close-ups, no `MiniMax-H3-Image`, no `ref2va`.
- **The two-person finding rests on one prompt and one seed,** and the two region prompts differ only in the
  direction they name. Whether another cue in the region prompt (a color, a size) picks a person the same way is
  not tested.
- **One seed per clip**, and one routed render per arm.
- **The subject finder includes her cast shadow** and cannot be used on a moving camera or a still subject.
- **Raw scores, not attention weights:** no softmax over all keys was computed.
- **`coat-colored` measures red pixels**, which is a proxy for "still looks like the base walker"; the edit was
  read on stills and not on playback.

## 11. Read later on close-up stills

Three sheets cut from the clips at full size, in `$H3_OUTPUT_DIR/Video/self_mask_late_harvest_2026-11-27/`:
`walk_closeup_sheet.png` (lower part of the frame at `f36 f72 f96 f120`), `two_people_closeup_sheet.png` (the same
at `f48 f76 f84 f110`) and `walk_upper_right_zoom.png` (base, `f1` and `f2` at `f72 f96 f120`). Stills, not
playback.

- **A routed render is a different take of the scene, not the base clip with the subject changed.** In every
  routed walk arm the walker is larger in frame than in `c0_walk_base`. The courtyard differs in detail too: `f1`
  and `f2` have hanging lamps in the arches and a downpipe on the wall that the base clip does not have, and `f2`
  has a bar across the right-hand arches that neither of the other two has. Section 5's "routing changes how fast
  she walks" is too narrow a reading: where she is differs because the whole shot differs.
- **The arms that changed her least are the ones nearest the base shot.** `d2` and `d3` (section 5) leave her as
  filmed. Whether the edit can take without the shot changing was not tested.
- **The target words do not pick who changes, but they show in how complete the change is.** `i1` (his words, her
  direction) carries a red patch behind the chrome figure on all four frames. `i3` (her words, his direction)
  keeps a yellow collar on all four and a human head on the earlier ones. `h2` and `i2`, whose words name the
  person the region prompt describes, show neither. This agrees in direction with the `still red` and
  `still yellow` columns in section 9. One seed, four frames per arm.
- **The person who is not changed is not the person in the base clip either:** in `i2` and `i3` the woman's coat
  is darker and longer than in `p3_two_people_base`, and `i2` is framed closer.
