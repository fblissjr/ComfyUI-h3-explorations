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

[`../../wiki/control_vectors/which_tool_and_what_to_call_it.md`](../../control_vectors/which_tool_and_what_to_call_it.md)
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
  `t2va_cafe_kids`. They are not plain scenes: section 11.3 says what their own openings carry.
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
[`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../../bench/results/2026-11-12_control_vector_word_share.md).

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
`MiniMaxH3EditGuider`; [`../../wiki/edit_guidance.md`](../../edit_guidance.md) section 2.

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
[`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../../bench/results/2026-11-12_control_vector_word_share.md) section 3 measured for
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

## 10. What would be done differently, knowing this

Written the same day, after the clips were made and before most were judged. It is about how this batch was
run, not about what the clips show.

### 10.1 The vectors

- **Read the whole build card before rendering, not only the words.** Each card prints how well a vector's
  pairs agree and the range it calls healthy. Five of the eight were below that range (section 3). Only the
  readout words were read first, and they looked right. A vector whose pairs disagree should be rebuilt or
  set aside before a render is spent on it.
- **Look at how the build method makes its contrast before using it.** A bare phrase compiles into pairs
  with a few neutral words on one side and a long phrase on the other, against the same few neutral phrases
  every time (section 3). That is where the batch's large shared part comes from (section 3.2). It was found
  after the vectors were built. A real plain side, written for each concept and about as long as the concept
  side ("clear dry weather with bare pavements" against the blizzard), over more scenes than five, would
  have been a fairer build.
- **Keep trial vectors out of the bank's folder under plain names.** The eight sit in
  `<comfy>/models/control_vectors/` beside the bank and look like any other vector in the node's list. They
  are untested and several are weak. A prefix that marks them as trials, or removing them once the clips
  were made, would have been better. The owner has since decided to keep them (section 12).

### 10.2 The renders

- **Find a working setting on one vector before running eight.** Every vector got one try at the node's
  defaults (section 4). The wiki's own recipes use a higher strength and a shorter window for a change of
  world. One vector taken through strength, target and window first would have shown what setting to use, or
  that none works. Eight concepts at one setting taught less than two concepts at five settings would have.
- **Give edit guidance the arm that tests what it is for.** It was run with `first_step = 0`, where it and a
  plain rewrite do nearly the same thing, and their stills look alike (section 8.2). The arm that holds the
  original take, `first_step = 1`, was left out, so the comparison says "words in the prompt beat this
  vector" and says little about edit guidance itself.
- **Change one thing between neighboring arms.** Vector against rewrite differs in the method, in where the
  concept comes from (five other scenes against this scene), and in settings that were never matched. A
  ladder in which each step changes one of those is the design the second lead proposed afterwards, and it
  is the better one.
- **Fit the edit to the concept.** For the three style concepts the phrase was added in front of
  "Live-action, cinematic", which contradicts it, and no method showed the concept (section 8.2). The clash
  was kept for a like-for-like comparison with the vectors' pairs, which had it too. The better answer was to
  fix it in both: replace that clause for a style concept, in the pairs and in the edit.
- **Use more than one seed for anything that will be argued from.** A guess that the tortoise's harness
  came and went with the seed was made from one seed and was wrong on three (section 8.1).

### 10.3 What held up

- Rendering the plain rewrite beside edit guidance. Without it the edit-guidance clips would have been read
  as showing what edit guidance adds.
- Keeping the prompt, seed, length and graph fixed across every batch, so any clip stacks against any other.
- Writing down every input each clip was made with, and what the stills cannot say.

The same question for the day as a whole is section 14 of
[`2026-11-12_chain_lens_digest_and_experiment_plan.md`](2026-11-12_chain_lens_digest_and_experiment_plan.md).

## 11. A second session's cross-check, on playback and sound

Another session watched and listened to the single clips of `exp2`, `exp3` and `exp4` in full, beside an
edit-guidance showcase it had rendered the same day, and passed its notes on through the owner. They cover
what sections 8 and 9 said this record could not see. Each point below says what was checked here afterwards
and how it came out.

### 11.1 Rewrite against edit guidance

- **Its reading.** On playback the rewrite and edit guidance are close to the same clip. Edit guidance adds
  a little density (snow, ferns). Against the vectors at their default settings both are clearly ahead;
  against each other, not meaningfully. On `knitted_wool_world` edit guidance patterns the shell and thickens
  the cardigan, which it puts down to the prompt already saying "green knitted cardigan".
- **Checked here.** The prompt does say that. The rest agrees with section 8.2's stills. Not checked
  further: this record's session cannot watch playback.
- **Where it found edit guidance does separate from a rewrite**, in its own renders: a later `first_step`
  with an early `last_step`, which holds the original take; a `scale` above the default when the picture
  mutes a new soundscape; and a `scale` under 1 for a partial change. None of those was in `exp4`
  (section 10.2). [`../../wiki/edit_guidance.md`](../../edit_guidance.md) has its clips.

### 11.2 Sound, which the stacks do not carry

- **Its reading.** For `heavy_blizzard` and `ankle_deep_flood` the rewrite and edit-guidance clips are much
  quieter than the unsteered clip, with no wind or water added: the prompt's soundscape section still
  describes a dry street, and the edit touched only the picture. For `tv_1950s_broadcast` the picture stayed
  in color but the sound gained broadcast hiss.
- **Checked here, holds.** `ffmpeg`'s `volumedetect` on the single clips: the two blizzard clips and the two
  flood clips are quieter than the unsteered clip by roughly the amount it gave, and above 2.5 kHz the two
  TV clips are louder than the unsteered clip by roughly the amount it gave. The absolute levels differ from
  its figures, as two ways of measuring loudness will. What the sound is cannot be checked from levels.
- **Also seen in the same check.** The vector clips for the blizzard and the flood are quieter than the
  unsteered clip too, though `scene_only` leaves the soundscape rows unsteered.

So an edit that changes the weather in the picture and says nothing in `overall_soundscape:` changes the
sound anyway, and not toward the weather. A fair comparison of any of the three methods on such a concept
has to edit, or steer, the soundscape too.

### 11.3 Two things mixed into the vectors

- **"Live-action, cinematic" blocked the concepts that change the medium, in every method.** It counts four:
  the three in section 8.2 and `knitted_wool_world`, where on playback the bus, street and buildings stay
  live-action. "On an overcast morning" likewise stays in the prompt after the blizzard and golden-hour
  phrases. This is the clash section 5 names; its cost was larger than that section allowed.
- **The five bank scenes bring their own looks.** **Checked here, holds:** `t2va_box_office` opens "Vintage
  film, 16mm grain with warm faded colour", `t2va_cable_car` opens "2D-animated, flat cel shading" above "a
  snowed valley", `t2va_cafe_kids` has "golden hour" in its opening, and `t2va_arena_rock_video` opens as a
  1980s music video with haze. So a pair reads "Unstylized candid footage, Vintage film, 16mm grain..."
  against the concept phrase in front of the same words. Three of the eight concepts (the TV broadcast,
  golden hour, the blizzard) meet a scene that already says something close to them.
- **Its account of the shared part.** It puts the batch's large shared direction down to those openings
  together with the lopsided phrases, and notes that this is why so many steered clips turn the candle into
  a torch. **Cannot tell from here:** the openings are on both sides of every pair, so they should mostly
  cancel, and nothing run separates their effect from the lopsided phrases'. Both belong on the list of
  things a fairer build removes.

### 11.4 What follows for a next test

Its suggestions, with what this record's session would add:

- **A higher strength on these files also multiplies their shared part.** So a strength sweep on the
  current `heavy_blizzard` cannot by itself say the vector was under-driven. An arm with the batch's shared
  direction held out of the vector, at the same strengths, would separate the two.
- **Build one vector in place and edit without the clash.** A vector made with the tortoise prompt as its
  target, from an edit that replaces the conflicting words ("on an overcast morning" for the blizzard,
  "Live-action, cinematic" for a medium) and includes a soundscape clause, beside edit guidance on the same
  edit with a later `first_step`. Each tool is then used as it is meant to be. One caution: a vector built
  from one pair on its own target prompt is close to that rewrite stored as a direction. It tests whether a
  vector can carry the edit at all, not whether it carries it to another prompt, which is what a vector is
  for.
- **If the edit has a soundscape clause, the vector arm needs a target that reaches the soundscape rows**,
  or the two arms differ on sound for a reason that is not the method.
- **Two seeds**, as section 10.2 says.

### 11.5 A later, longer review from the same session

It then reviewed the whole of `exp2`, `exp3` and `exp4` against the scripts and passed that on too. Most of it
repeats sections 11.1 to 11.4 at more length. Three points are new, and each was checked here:

- **The candle. Does not hold as stated.** It reads the shared-direction clips as lighting a candle the
  prompt calls unlit, and builds on that. `prompt_bank/t2va_offpath_tortoise.txt` says "a single lit white
  birthday candle ... its small flame flickering", and has since it was added. So the prompt asks for a
  small flame; the unsteered clips show little or none, and many steered clips show a large one. That the
  shared direction turns the flame into a torch stands as something seen. That it overrides the prompt does
  not: on this detail the steered clips are closer to the prompt than the unsteered ones (section 8.1).
- **The parts in the first three-way split were each given the whole vector's length. Holds, and was
  stated.** A part that is a small share of its vector was therefore pushed several times harder than it
  is inside the vector. That was the design, to compare directions, and
  [`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../../bench/results/2026-11-12_control_vector_word_share.md)
  section 6.5 names it as a limit. It adds, fairly, that `head_swell` is a close-up vector tried on a wide
  shot.
- **Building in place on the target prompt removes most of the shared part.** It built two vectors with the
  tortoise prompt as target, from edits that replace the conflicting words, and reports a much lower cosine
  with this batch's shared direction and readout words on the concept. Not checked here; the figures are
  its own. If they hold, the batch's shared part was the build, as section 3.2 could not say.

It then ran a ladder of ten renders on that design, in `exp5/` beside the others. Its results are its own to
record.

## 12. The vectors are kept

The owner's decision, 2026-11-12: keep the eight, where the other vectors are. Some did better than others;
all are worth keeping. What was added so that they stand like the others:

- **A pair file each**, `data/control_vectors/<name>.toml`, beside the other generated pair files. Each was
  written from the exact pairs recorded in its vector file's metadata and loads with
  `control_vectors.py::load_pairs` (`write_pair_files.py` in the run folder).
- **The `ref2va` file and its card**, built from that pair file with `MiniMaxH3MakeControlVector` on the
  shipped Pika `ref2va` control-vector graph's loaders, in one queue item (`build_ref2va_siblings.py`). Pair
  agreement on the `ref2va` cards is within a hundredth of each vector's `fl2va` figure in section 3.

They remain what section 10.1 says they are: built from lopsided pairs on scenes with looks of their own,
tried at one setting. Their names do not mark that. This record does.

