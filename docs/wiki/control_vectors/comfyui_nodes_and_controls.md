# The control vector nodes

last updated: 2026-11-10

[Back to the start page](../control_vectors.md)

![Runtime steering and 3-lever span masks](assets/runtime_steering_and_masks.svg)

Four nodes in `control_vector_nodes.py`. Each node's inputs, ranges, defaults and tooltips are its own
`define_schema`, which is what ComfyUI shows; this page says what each node is for and what to know before
using it. "The plan" is
[`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).

| Node | Takes | Gives | For |
|---|---|---|---|
| `MiniMaxH3ApplyControlVectors` | a model and a vector file | a model | Applying a vector at render time. It goes on the model wire before the guider. Several can be chained |
| `MiniMaxH3MakeControlVector` | a model, an encoder, and a pair file or a prompt with edit clauses | the built file's name and a report | Building a vector inside a graph. It can take encoded references for a prompt that names a picture |
| `MiniMaxH3MakeMediaControlVector` | a model, an encoder and two pictures | the built file's name and a report | Building a vector from two pictures |
| `MiniMaxH3LayerNormMatch` | a conditioning and a reference conditioning | a conditioning | Scaling the text rows of a conditioning taken from an earlier encoder layer so their typical length matches the reference's; an experiment's helper, not part of steering |

Of the two build nodes, neither was run by a session on 2026-11-09; the vectors that day were built from
the command line, which runs the same build code. The second and the fourth are untested here.

## The apply node

Its defaults are `h3_rules.CONTROL_VECTOR_DEFAULTS` and `h3_rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS`. What
each input is for:

- `vector_name`: the file under the models folder's `control_vectors/`. A file is built for one family and
  one recipe, and its name says which. The node refuses a file whose recipe does not match the loaded model
  unless `allow_recipe_mismatch` is on.
- `mode`: how the vector is used. `add_norm_matched`, the default, rescales the vector to one fixed length
  (`control_vectors.NORM_MATCHED_TARGET`) and adds it times `strength`, so a strength is the same push for
  any vector. `add` adds the vector as stored. `hold` takes this vector's direction out of the push the
  nodes before it in the chain have added. `project_out` and `amplify` remove or scale what the prompt's
  own rows already carry along the vector. `control_vectors.py::apply_post_refiner` is the definition.
- `strength`: the multiplier. A build prints a recommended range for the default mode
  (`control_vectors.strength_ranges`). A negative strength pushes the other way.
- `mask`: which rows are pushed (`h3_spans.py::inference_mask`). On sectioned H3 prompts, the **3-lever
  section masks** separate Visual Look, Events/Actions, and Audio while always protecting `<Picture N>` /
  `<Video N>` / `<Audio N>` / `<Subject N>`, `[Shot N]`, `<d>dialogue</d>`, `(S1)` speaker tags, section
  headers, `ref_en.md` task brackets (`[reference generation]`, etc.), and `retention_analysis:` relationship
  prefixes (`fully_preserved -`, etc.):

  | Mask mode | Lever | Rows steered |
  |---|---|---|
  | `look_only` | **Visual Look** | `subject_definitions:` + `summary:` + visual lines of `retention_analysis:` (excludes `<Audio N>:` retention lines; falls back to `description_only` on 3-field `t2va`/`fl2va` prompts) |
  | `scene_only` *(default)* | **All Visual Prose** | `look_only \| description_only` — all visual prose across the prompt, excluding `overall_soundscape:` and `non_diegetic_music:` |
  | `description_only` | **Events / Actions** | `detailed_description:` / `integrated_multimodal_description:` non-dialogue prose only |
  | `soundscape_only` | **Audio** | `overall_soundscape:` + `non_diegetic_music:` (when its body is not `"N/A"` / `"None"`) + `<Audio N>:` lines in `retention_analysis:` |
  | `after_dialogue` | Post-speech visual tail | `scene_only` rows strictly after the last `<d>...</d>` dialogue token |
  | `non_vision` | Every text row | All tag-1 non-vision tokens (including headers, tags, and dialogue) |
  | `all_rows` | Text + vision | Every row, including tag-0 vision prefix tokens |
- `start_percent`, `end_percent`: the part of the run in which the vector is on.
- `video_scale`, `audio_scale`, `split_end_percent`: separate multipliers for the video and audio streams.
- `ref_vae_end_percent`, `ref_vae_aug`: for a reference shot, how long and how strongly the reference's VAE
  latents stay in force. The defaults equal core's own value, and the node leaves that road alone unless
  one is lowered (`bench/check_control_vectors.py` holds the default to core's).
- `words`, `against_words`, `hold_words`, `hold_alpha`: build a vector from words at run time, or take
  words out of a loaded one, through J-Lens. See the J-Lens page for what that can and cannot be trusted
  for.

## What to know before turning things

- The push is one size on every row it reaches. A wider mask is therefore a larger total change at the
  same strength, not only a change of where (the plan, section 24).
- Which mask is right depends on the aim. `scene_only` (`look_only | description_only`) is the default
  because it steers all visual prose while protecting structural/reference/dialogue tags and excluding
  `overall_soundscape:` / `non_diegetic_music:` (visual-vs-audio section isolation verified in `Arms 59..60`
  of [`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)).
  On a `ref2va` shot, keeping a vector off `subject_definitions:` and `retention_analysis:` (`description_only`)
  leans toward preserving the reference's prose description, while reaching them (`scene_only` or `look_only`)
  leans toward changing the look; one round compared those on a reference shot and found small metric
  differences for a bank vector, which the owner has not yet visually judged (the plan, section 25).
- On a reference shot the subject also reaches the DiT as VAE latents, which no mask touches. In that round
  the subject read as the same person in every arm, including the one that pushed the picture's rows. The
  two `ref_vae_` inputs are the handle on that road and were not varied.
- A `hold` node with nothing before it in the chain has nothing to take a direction out of.
- The shipped graphs set a strength above the node's default on purpose, so a first render shows the grade
  plainly; `bench/check_widget_deviations.py` declares it.

## Where the node acts

The node wraps the model so that, at each step it is on, the refined text rows are changed before the main
blocks run (`control_vector_nodes.py`, the wrapper; `control_vectors.py::apply_post_refiner`). It reads
which rows are which from the span masks the conditioning node attached, and refuses when they are missing.
Nothing in ComfyUI core is patched.

With edit guidance (`MiniMaxH3EditGuider`), both `[base]` and `[edited]` carry `h3_edit_pass` metadata on
`minimax_payload` and are treated as conditioned passes (never misclassified as unconditioned CFG passes)
across active steps.
