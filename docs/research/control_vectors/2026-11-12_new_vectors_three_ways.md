# 2026-11-12: Eight new control vectors, and the same concepts by rewrite and by edit guidance

Eight control vectors were built fresh, at the owner's word, and each concept was then rendered three ways on
one prompt and seed: as the vector, as a plain rewrite of the prompt, and by edit guidance. This record says
what each vector was made from, what settings every clip was rendered with, and what the equivalent is in each
method. **No clip has been judged.** Section 8 lists where to look, from one still of each clip.

The scripts are throwaways with no check, beside their outputs in the untracked
`data/control_vectors/new_vectors_2026-11-12/`. The vectors themselves are ordinary files in
`<comfy>/models/control_vectors/`, each with its `.md` card, and each file's metadata holds the full text of
every pair it was built from (`pairs.pairs_ab`).

## 1. The methods, and which is which

There are two ways to steer in this pack, and one baseline that uses neither:

| | What changes | Reusable on another prompt | Model passes |
|---|---|---|---|
| **Control vector** (`MiniMaxH3Steer`) | One direction is added to the prompt's text rows | Yes | One |
| **Edit guidance** (`MiniMaxH3EditGuider`) | The original and an edited prompt both run each step; the result is moved from the original toward the edited | No | Two on every guided step |
| **Rewrite** (the baseline) | The prompt text itself; rendered normally | No | One |

A fourth thing is sometimes counted as a method and is not one: a **word vector** (`MiniMaxH3WordVector`) is a
control vector made from a word list instead of from pairs. It goes through the same `MiniMaxH3Steer` node.
Section 6 gives the word-vector equivalent of each concept; none was rendered here.

[`../../wiki/control_vectors/which_tool_and_what_to_call_it.md`](../../wiki/control_vectors/which_tool_and_what_to_call_it.md)
has the comparison in full.

## 2. The eight concepts

Each concept is one phrase. The same phrase is what the vector was built from (section 3) and what the
rewrite and edit guidance add to the prompt (section 5).

| Vector | Phrase |
|---|---|
| `knitted_wool_world` | A hand-knitted wool world where every surface is chunky yarn stitches |
| `heavy_blizzard` | A heavy blizzard with thick snow piling on every surface and wind-driven flakes |
| `golden_hour_glow` | Low golden-hour sunlight with long warm shadows and soft lens flare |
| `overgrown_jungle_ruin` | An overgrown ruin reclaimed by jungle vines, moss, and giant ferns |
| `tv_1950s_broadcast` | A 1950s black-and-white television broadcast with scan lines and soft ghosting |
| `ankle_deep_flood` | Ankle-deep floodwater rippling across the ground and mirroring everything above it |
| `toy_brick_world` | A toy world built entirely from glossy plastic building bricks with studded surfaces |
| `watercolor_wash` | A watercolor painting with soft bleeding pigment washes on textured paper |

The phrases were written for this run. They are not from the pair bank.

## 3. What each vector was built from

- **Node.** `MiniMaxH3MakeControlVector` on the running ComfyUI server, with `words` set to the bare phrase,
  `target_prompt` empty, `mode = t2va`, `pooling = diff_span`, `apply_holds = true`. With no target prompt the
  node compiles the phrase across five `prompt_bank` scenes (`h3_pair_compiler.py`).
  `build_new_vectors.py` in the run folder queued them; `graph_<name>.json` beside it is each graph.
- **Loaders.** The shipped Pika `t2va` control-vector graph's own: the `fl2va` checkpoint with the Pika H3
  Flex LoRA at strength 1.0. Each file's `recipe` metadata names the checkpoint, encoder and LoRA with hashes.
- **The five scenes**, the same for every concept:
  `t2va_appalachian_folk_cabin`, `t2va_arena_rock_video`, `t2va_box_office`, `t2va_cable_car`,
  `t2va_cafe_kids`.
- **The pair made from each scene.** Both sides are the whole scene prompt with one short phrase put at the
  opening of its first shot. Side B gets the concept phrase. Side A gets one of the compiler's neutral
  phrases, a different one per scene (`h3_pair_compiler.py::DEFAULT_AUX_BASELINES`).

For `heavy_blizzard` the five pairs differ like this, and in nothing else:

| Scene | Side A opens with | Side B opens with |
|---|---|---|
| `t2va_appalachian_folk_cabin` | Everyday contemporary realism, | A heavy blizzard with thick snow piling on every surface and wind-driven flakes, |
| `t2va_arena_rock_video` | Ordinary natural-light setting, | the same |
| `t2va_box_office` | Unstylized candid footage, | the same |
| `t2va_cable_car` | Plain matter-of-fact framing, | the same |
| `t2va_cafe_kids` | Neutral quotidian atmosphere, | the same |

So a pair reads, for the first scene: "[Shot 1] **Everyday contemporary realism,** Live-action, cinematic, a
rustic folk music documentary style captures..." against "[Shot 1] **A heavy blizzard with thick snow piling
on every surface and wind-driven flakes,** Live-action, cinematic, a rustic folk music documentary style
captures...".

The other seven concepts use the same scenes and the same neutral phrases with their own phrase on side B.
(`ankle_deep_flood` drew the neutral phrases in a slightly different rotation; its file records which.)

- **What is pooled.** Only the rows of the differing phrase, on each side (`diff_span`). Side A's phrase is a
  few rows long; side B's is several times that. The vector is the mean of side B's rows minus the mean of
  side A's, averaged over the five pairs.

What each build recorded (from each file's metadata and card):

| Vector | Rows pooled, side A (per pair) | Rows pooled, side B | Length of the vector | Agreement between its five pairs |
|---|---|---:|---:|---:|
| `knitted_wool_world` | 3 to 6 | 15 | 80.4 | 0.61 |
| `heavy_blizzard` | 3 to 6 | 16 | 83.1 | 0.63 |
| `golden_hour_glow` | 3 to 6 | 12 | 75.5 | 0.57 |
| `overgrown_jungle_ruin` | 3 to 6 | 15 | 83.3 | 0.62 |
| `tv_1950s_broadcast` | 3 to 6 | 19 | 65.3 | 0.49 |
| `ankle_deep_flood` | 4 to 6 | 17 | 76.6 | 0.60 |
| `toy_brick_world` | 3 to 6 | 14 | 89.4 | 0.65 |
| `watercolor_wash` | 3 to 6 | 13 | 77.9 | 0.60 |

"Agreement" is the card's `mean_pairwise_cosine`. Each card's own table gives the range it calls healthy:
three of these are inside it and five are below it, `tv_1950s_broadcast` furthest.

### 3.1 The words each vector reads as

J-Lens readout, eight words each way, with the bundle as rebuilt on 2026-11-12 (`shared_new.json`):

| Vector | Toward | Away from |
|---|---|---|
| `knitted_wool_world` | texture, fuzzy, textured, hairy, wool, intricate, furry, plush | shot, outdoors, daytime, nighttime, straightforward, indoor, shots, indoors |
| `heavy_blizzard` | snow, dust, storm, cloud, rain, layers, thick, frost | shot, documentary, straightforward, unnamed, minimalist, fictional, converse, casual |
| `golden_hour_glow` | sunlight, warm, lighting, glowing, golden, shadows, sunset, light | fictional, narrative, confront, explicitly, contemporary, nihilism, dialogue, concerning |
| `overgrown_jungle_ruin` | jungle, moss, lush, walls, exotic, towering, forest, bamboo | straightforward, shot, converse, normal, description, conventional, standard, daytime |
| `tv_1950s_broadcast` | vintage, faded, nostalgic, grain, timestamp, retrod, decades, dated | indoors, outdoor, natural, indoor, outdoors, quiet, nighttime, daytime |
| `ankle_deep_flood` | shimmer, water, rain, floor, pool, above, submerged, surface | style, straightforward, directive, genre, converse, natural, faithful, documentary |
| `toy_brick_world` | plastic, material, intricate, populated, intricacy, metal, primitive, colorful | shot, quiet, daytime, nighttime, documentary, shots, evening, observational |
| `watercolor_wash` | water, sketch, texture, layers, paper, textured, delicate, shimmer | nighttime, daytime, unidentified, protagonist, unnamed, straightforward, fictional, confront |

The "toward" words fit each concept. The "away" words are much the same across the batch and echo the neutral
phrases on side A.

### 3.2 What the batch shares

The eight vectors are more alike than the older bank's vectors are (`shared_new.json`):

| | Older bank | This batch |
|---|---:|---:|
| A typical vector's cosine with the mean of its own set | 0.55 | 0.72 |
| A typical cosine between two vectors of the set | 0.26 | 0.44 |

- The mean of this batch reads toward "layers, shimmer, texture, textured, walls, swirling" and away from
  "straightforward, daytime, quiet, unspecified, shot, pragmatic, observational".
- Its cosine with the older bank's mean is 0.42: related, not the same.
- Every vector here was set against the same few neutral phrases, and all eight are a world or a look, so
  both the method and the choice of concepts push this figure up. This run does not separate the two.

## 4. How the vector clips were rendered

One `MiniMaxH3Steer` per clip, on the shipped Pika `t2va` control-vector graph
(`workflows/h3_text_to_video_pika_control_vector_api.json`), through `bench/edit_guidance_arms.py`
`--vector-arm`. The arm graphs are in `renders/` in the run folder.

| Input | Value |
|---|---|
| `vector` | the concept's name |
| `strength` | 1.15, the shipped graph's value |
| `mode` | `add_norm_matched`: every vector is rescaled to one length before it is added |
| `target` | `scene_only` |
| `start_percent`, `end_percent` | 0.0, 1.0: on for every step |
| `streams` | `both` |
| `hold_words` | empty |
| `allow_recipe_mismatch` | false |

The rest of the graph is as shipped: six steps, `lcm` sampler, the graph's manual sigmas, 1344x768, Sol as
the graph sets it.

- **Prompt.** `prompt_bank/t2va_offpath_tortoise.txt`, unchanged: one shot of an elderly man walking a
  tortoise across a zebra crossing past a waiting bus.
- **Length and seed.** The bank's frame count for that prompt (`prompt_bank/bank.json`); seed 730451892.
- **What `scene_only` covers on this prompt.** The visual prose rows: 212 of the prompt's 276. The
  soundscape rows (41) and the headers are left alone. On a three-section prompt like this one
  `description_only` and `look_only` select the same 212 rows (`h3_spans.py::inference_mask`).

**These settings are defaults, not findings.** Every vector was tried at this one setting. The strength is
inside the node's recommended range and below what the wiki's recipes use for a change of world; the target
is narrower than the `non_vision` most early verified renders used. A sweep that would say whether a weak
clip is the vector or the setting has not been run.

### 4.1 The two shared-direction clips

- `shared_new`: the mean of the eight vectors' unit vectors, through the same node and settings.
- `shared_new_reversed`: the same file at `strength = -1.15`.

The file was written for the renders and removed afterwards; `shared_new.py` rebuilds it.

### 4.2 Two more unsteered clips

The prompt alone at seeds 11 and 12, to go with the unsteered clip at the main seed from
[`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../bench/results/2026-11-12_control_vector_word_share.md).

## 5. The equivalents that were rendered

Both use one edit, the same for every concept apart from the phrase. It puts the phrase where the vector's
own pairs had it, at the opening of the first shot:

```text
Live-action, cinematic, -> <phrase>, Live-action, cinematic,
```

For `heavy_blizzard` the prompt then opens: "[Shot 1] **A heavy blizzard with thick snow piling on every
surface and wind-driven flakes,** Live-action, cinematic, a medium-wide shot frames a zebra crossing...".
Everything after that is the original prompt.

| | Rewrite | Edit guidance |
|---|---|---|
| Arm in `bench/edit_guidance_arms.py` | `rewrite` | `node_s1.5` (with `--node-arms --scales 1.5`) |
| Prompt given to the model | the edited prompt only | the original and the edited prompt |
| Guider | `BasicGuider`, as shipped | `MiniMaxH3EditGuider` |
| Guider settings | none | `scale = 1.5`, `first_step = 0`, `last_step = 1000` (every step), `align_positions = false`: the node's defaults |
| `MiniMaxH3Steer` | none | none |
| Everything else | as section 4 | as section 4 |

In a graph built by hand, the same edit is one line in `MiniMaxH3PromptEdit`'s `edits` feeding
`MiniMaxH3EditGuider`; [`../../wiki/edit_guidance.md`](../../wiki/edit_guidance.md) section 2.

One thing to weigh when watching: for the three style concepts the edited prompt still says "Live-action,
cinematic" straight after the phrase. That was kept because the vectors' own pairs have the same clash, so
the comparison stays like for like. It may hold those rewrites back.

As with the Steer settings in section 4, these are the node's defaults and were not tuned. In particular
`first_step = 0` lets the edit shape the first step, where the node's own tooltip says a later first step
keeps the original framing and pose.

The arm graphs and a record per concept are in `renders_edit/` in the run folder.

## 6. The word-vector equivalent, not rendered

A word vector would be `MiniMaxH3WordVector` wired into `MiniMaxH3Steer`'s `custom_vector`, with the Steer
node's `vector` set to `none` and the other settings as in section 4. Its `words` must be whole single-token
words. Each vector's own "toward" words in section 3.1 already are, so the nearest word vector to, say,
`heavy_blizzard` is `words = "snow, storm, frost, thick"` against the node's neutral default.

How close such a vector comes to the pair-built one is the question
[`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../bench/results/2026-11-12_control_vector_word_share.md) section 3 measured for
the older bank. It was not measured for this batch.

## 7. Where the clips are

Under `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/`:

- `exp3/`: the eight vector clips, the two shared-direction clips, and the two extra unsteered seeds.
- `exp4/`: for each concept, `exp4_<name>_rewrite` and `exp4_<name>_node_s1.5`.
- `exp2/exp2_base_s730451892`: the unsteered clip at the main seed.
- `exp3_shared_directions_stacked.mp4`: unsteered, this batch's shared direction, the same reversed, and the
  older bank's shared direction.
- `exp3_unsteered_three_seeds_stacked.mp4`: the prompt alone at three seeds.
- `exp4_<name>_three_ways_stacked.mp4`, one per concept: unsteered, the vector, the rewrite, edit guidance.

The stacks carry no sound (`bench/stack_labeled_clips.py`); the singles do.

## 8. Where to look

From one still of each clip. A list of where to look, not a judgement.

### 8.1 The vector clips

- **`golden_hour_glow`**: the sky and the light. In its still they are warm and low.
- **`overgrown_jungle_ruin`**: the tortoise's shell, which carries moss in its still; the street does not.
- **`heavy_blizzard`**: the road and the shell. The still shows a wet street and something white on the
  shell, and no falling snow.
- **The other five**: whether anything of the concept shows at all. Their stills show an ordinary street in
  ordinary color.
- **The harness and the candle.** The prompt asks for both. In stills the three unsteered clips show no
  harness and a small or missing flame; several steered clips show a harness and a larger flame, the
  shared-direction ones included.
- **`shared_new_reversed`**: its still looks plainer than the unsteered one.

### 8.2 The three ways, concept by concept

In each `exp4_<name>_three_ways_stacked.mp4`, rows two to four are the vector, the rewrite and edit guidance.

- **`heavy_blizzard`**: the ground and the railings. In the rewrite and edit-guidance stills they are
  white with snow; in the vector still they are not.
- **`overgrown_jungle_ruin`**: the buildings. In the rewrite and edit-guidance stills they are hung with
  ferns and vines; in the vector still only the tortoise's shell is green.
- **`golden_hour_glow`**: the light in all three rows. All three stills are warm; the rewrite and
  edit-guidance ones more so, down to the road.
- **`ankle_deep_flood`**: the road around the tortoise. The rewrite and edit-guidance stills show standing
  water with reflections; the vector still shows a damp road.
- **`knitted_wool_world`**: the tortoise's shell and the man's cardigan in the edit-guidance row, which
  look patterned and chunky in its still. The other two rows look ordinary.
- **`tv_1950s_broadcast`, `toy_brick_world`, `watercolor_wash`**: whether any row shows the concept. No
  still of any of the three methods is black-and-white, built of bricks or painted. These are the three
  whose edited prompt still says "Live-action, cinematic" after the phrase (section 5).
- **The rewrite against edit guidance.** In stills the two rows of a concept look alike. What edit guidance
  is for, keeping the original take's framing and action, is a question for playback against the top row.

## 9. What this cannot see

- **Whether a weak vector clip is the vector or its settings** (section 4).
- **More than one prompt and one seed** for every steered clip.
- **Sound.** Not looked at.
- **Anything about these vectors on `ref2va`.** Only the `fl2va` files were built.
- **What the J-Lens words would have been on the older bundle.**
