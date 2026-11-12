# Edit Guidance: Steering a Render by Editing Your Own Prompt

Last updated: 2026-11-12.

**Edit Guidance** lets you change specific parts of a shot—an outfit, a prop, the lighting, the weather, or the sound—simply by writing `old wording -> new wording` and turning a strength dial. You do not need to train anything, write prompt pairs, or build a control-vector file first.

At each sampling step, the model looks at both your **original prompt** and your **edited prompt**, compares what they want to draw, and nudges the render toward your edit by the `scale` you choose. Everything you did *not* change—the character's face, their spoken lines, the camera movement, and the room around them—cancels out and stays anchored.

---

## 1. Why use Edit Guidance?

| Situation | Why Edit Guidance helps |
|---|---|
| **A normal prompt edit gets ignored** | When a reference photo (`<Picture 1>`) or a dense scene strongly locks a character's existing sweater or lighting, simply changing the prompt (`scale = 1.0`) often isn't strong enough. Turning `scale` up to `1.5` boosts *only* what you changed so the new outfit or prop actually lands. |
| **Detailed, multi-part outfits or props** | Control vectors average an entire phrase into one blended push, so `"a rust-orange blouse and blue flared jeans"` loses which color belongs to which piece of clothing. Edit Guidance reads both full prompts word for word, keeping every color, material, and noun attached to the right item. |
| **A normal prompt edit goes too far** | Setting `scale` between `0.0` and `1.0` (for example `0.5`) gives you a gentle halfway blend instead of an all-or-nothing rewrite. |
| **You want the exact same camera angle and pose** | Rewriting a prompt at the start of a render often makes the model pick a new framing or pose on the same seed. Setting `first_step = 1` lets step `0` lock in the original shot's layout first, then applies your edit on top. |
| **Removing something unwanted ("negative prompt" trick)** | Fast 6-step Pika workflows have no negative prompt box. Replacing an unwanted detail with its clean opposite (for example `steam crossing the lens -> crystal-clear air across the lens`) at `scale = 1.5..2.0` actively pushes the render away from it. |

**When is a [Control Vector](control_vectors.md) better instead?**
Reach for a Control Vector when you want a **reusable camera or world look** (like `film_16mm_1974`, `noir`, or `reverb_cathedral`) that you can drop onto any prompt at `1x` render speed without running two model passes per step. You can also use both together (see Example 7 below).

---

## 2. Quick start in ComfyUI

Open either of the two ready-to-run workflows:

- **Text-to-video (`t2va`)**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Reference image + text-to-video (`ref2va`)**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json)

You only need to touch three places in the graph:

1. **The Prompt text box** (`PrimitiveStringMultiline`): Write your normal scene prompt once here. It automatically feeds both the original branch and the edit node so the two never drift out of sync.
2. **`MiniMax H3 Prompt Edit`**: Type one `old wording -> new wording` change per line in `edits` (and turn on `release_subject_attire` if you are changing `<Subject 1>`'s clothes on a reference-image shot).
3. **`MiniMax H3 Edit Guider`** (wired into `SamplerCustomAdvanced` where `BasicGuider` normally sits): Adjust `scale`, `first_step`, or `last_step`.

---

## 3. What the dials and edit rules do

### `MiniMax H3 Edit Guider` dials

| Dial | Default | Plain-English guide |
|---|---|---|
| `scale` | `1.5` (`0.0 .. 4.0`) | **How strongly to push the edit.** `0.0` gives the original prompt; `0.5` gives a subtle halfway blend; `1.0` equals the edited prompt on its own; `1.5` (default) gives an extra push that reliably locks in new clothes, props, or lighting. Above about `2.5`, unrequested color saturation starts creeping into skin tones and highlights. |
| `first_step` | `0` | **Which step turns the edit on** (counting from `0`). Leave at `0` when your edit changes silhouettes, outfits, or major action. Set to `1` on 6-step Pika to let step `0` lock the original camera framing and actor pose before your edit kicks in. |
| `last_step` | `1000` (all steps) | **Which step stops running the extra pass.** Every guided step runs the model twice. On a 6-step Pika render (steps `0` to `5`), setting `last_step = 2` runs the 2-pass boost on steps `0, 1, 2` and finishes steps `3, 4, 5` with a single pass on the edited prompt—cutting total passes from `12` down to `9` (`1.5x` render time instead of `2x`) while keeping the edit locked in. |
| `align_positions` | `False` | Advanced toggle that aligns internal token positions when the original and edited prompts have different lengths. Leave `False` (no benefit seen on tested edits). |

### `MiniMax H3 Prompt Edit` rules & shortcuts

- **Basic rule (`old wording -> new wording`)**: Put one change per line. The `old wording` must appear **exactly once** so the node knows the exact spot to replace. If a word appears twice in your prompt, include a couple of neighboring words to make it unique (for example `white ceramic mug -> bright yellow glazed mug` instead of just `mug -> yellow mug`).
- **Aiming at one prompt section (optional prefixes)**: If a phrase appears in more than one section, start the line with a section name so it only searches there:
  - `description: old -> new` (the main visual scene description)
  - `soundscape: old -> new` (aliases `audio:`, `foley:`, `sound:`)
  - `music: old -> new` (aliases `score:`, `non_diegetic_music:`)
  - `subject_definitions: old -> new` (alias `subjects:`)
  - `summary: old -> new`
  - `retention_analysis: old -> new` (alias `retention:`)
- **Unlocking a reference character's clothes (`release_subject_attire`)**:
  Reference-image (`ref2va`) prompts normally include a `retention_analysis:` line telling the model `"fully_preserved - face, hair, and clothing are retained"`, which fights clothing edits.
  - Turn **`release_subject_attire = True`** on the node to automatically unlock `<Subject 1>`'s clothes while keeping their face and hair locked to `<Picture 1>`.
  - Have more than one character? Write `release_attire: <Subject 2>` (or `release_attire: all`) on its own line in `edits`.
- **Built-in guardrails (what it refuses on purpose)**:
  To keep you from accidentally breaking dialogue sync or reference wiring, `old -> new` lines will refuse to touch spoken `<d>...</d>` dialogue, speaker labels like `(S1)`, reference tags like `<Subject 1>` or `<Picture 1>`, `[Shot 1]` markers, or section headers. Always match the plain descriptive words *next to* those tags—never the tags themselves.

> **Tip when a `ref2va` prompt doesn't mention current clothes**: If `subject_definitions:` says `<Subject 1>: A woman with shoulder-length blonde hair from <Picture 1>.`, do **not** include `<Picture 1>` in your match (tags are protected). Match the plain words next to it instead:
> `subject_definitions: shoulder-length blonde hair -> shoulder-length blonde hair, wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans`

---

## 4. Examples across diverse use cases

### Example 1 — Changing a reference character's outfit (`ref2va`, verified on renders)
Swap `<Subject 1>` into a specific multi-piece outfit while keeping her `<Picture 1>` face, hair, voice, and spoken lines intact (verified on the diner 1970s outfit clip and parking-lot samurai armor clip).

- **Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json)
- **Node settings**: `release_subject_attire = True` on `MiniMax H3 Prompt Edit`; `scale = 1.5`, `first_step = 0` on `MiniMax H3 Edit Guider` (add `last_step = 2` for `1.5x` speed).
- **`edits`**:
  ```text
  subject_definitions: face, hair, and clothing are carried -> face and hair are carried, now wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans
  description: preserving the face, hair, wardrobe and build established in the reference -> preserving the face, hair, and build established in the reference while wearing the rust-orange wide-collar blouse and flared jeans
  ```
- **Good to know**: Leave `MiniMaxH3ReleaseReferences` (if present) at its default `end_percent = 1.0` for clothing swaps on the same person—keeping the clean `<Picture 1>` reference active preserves facial likeness while `EditGuider` cleanly replaces the outfit.

### Example 2 — Swapping a prop or gesture across shots (`t2va`, pattern verified on renders)
Replace a specific object or action (like turning a white mug into a bright yellow glazed ceramic mug, stopping a character from checking a watch, or changing market crates across cuts) while keeping camera framing and actor blocking locked.

- **Workflow**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Node settings**: `scale = 1.5`, `first_step = 1`, `last_step = 2`.
- **`edits`**:
  ```text
  crates of citrus -> wicker baskets of glowing blue bioluminescent mushrooms
  stacks fruit into a pyramid -> arranges glowing blue mushrooms into a cluster
  ```

### Example 3 — Pose-locked style, lighting, or medium shift (verified on renders)
Keep the original shot's room layout and choreography from step `0`, then shift the visual medium or lighting style—or use `scale = 0.5` from step `0` when you want a subtle half-strength blend (verified on noodle-bar claymation and diner sitcom-lighting clips).

- **Node settings**: `scale = 1.5`, `first_step = 1`, `last_step = 2` (pose-locked restyle) **or** `scale = 0.5`, `first_step = 0` (gentle 50% blend).
- **`edits`**:
  ```text
  Live-action, cinematic, -> Claymation stop-motion with fingerprinted plasticine surfaces,
  ```

### Example 4 — Synchronized weather / pyrotechnics + sound effects + score (coupled A/V verified on renders)
Because H3 generates video and audio together in one pass, editing visual prose alongside `soundscape:` and `music:` keeps lightning/explosions, foley, and music locked to the same beat.

- **Node settings**: `scale = 1.5..2.5`, `first_step = 0`, `last_step = 2`.
- **`edits`**:
  ```text
  late morning -> a midnight thunderstorm with rain lashing the aisle
  dust turning slowly in a shaft of light from the roof vents -> rainwater streaming from the roof vents under blue lightning flashes
  soundscape: Loose crowd murmur under a high roof -> Heavy rain hammering corrugated tin overhead with rolling thunderclaps
  music: N/A -> Tense, pulsing analog synth bassline.
  ```

### Example 5 — Removing an unwanted detail ("negative prompt" recipe, unrendered example)
Simply deleting `"steam"` from a busy kitchen prompt often still produces steam because `"four burners lit"` and `"flat top"` imply it. Replacing the unwanted phrase with its clean opposite at `scale = 1.5..2.0` actively steers the prediction away from steam:

- **Node settings**: `scale = 1.5..2.0`, `first_step = 0`, `last_step = 2`.
- **`edits`**:
  ```text
  four burners lit, steam crossing the lens -> four unlit cold burners, crystal-clear air across the lens
  steam crossing the lens twice -> clear air across the pass
  in a blur of steam -> in sharp, unobstructed focus
  ```

### Example 6 — Changing vocal timbre around locked spoken dialogue (unrendered example)
`MiniMax H3 Prompt Edit` protects `<d>...</d>` dialogue and `(S1)`/`(S2)` speaker tags so spoken words and lip-sync stay intact, while letting you rewrite the voice description right next to them:

- **Node settings**: `scale = 1.5`, `first_step = 0`.
- **`edits`**:
  ```text
  a warm, gravelly alto -> a breathless, conspiratorial whisper
  a quick, bright tenor -> a deep, booming bass-baritone
  ```

### Example 7 — Combining Edit Guidance with a Control Vector
`MiniMax H3 Edit Guider` (on `SamplerCustomAdvanced`) and `MiniMax H3 Steer` (on the `MODEL` wire upstream) stack cleanly on the same workflow. For example, use `MiniMax H3 Edit Guider` (`scale = 1.5`) to put `<Subject 1>` in a specific outfit while `MiniMax H3 Steer` applies `film_16mm_1974` (`target = "description_only"`, `strength = 1.0`) to give the room 1974 16mm film grain.

---

## 5. Under the hood (technical reference)

For readers inspecting [`edit_guidance.py`](../../edit_guidance.py), [`h3_prompt_edit.py`](../../h3_prompt_edit.py), or [`h3_rules.py`](../../h3_rules.py):

- **`CFGGuider` subclass (`EditGuider`)**: Subclasses `comfy.samplers.CFGGuider` without patching ComfyUI core (following `Guider_DualCFG` in `comfy_extras/nodes_custom_sampler.py`). On steps inside `[first_step, last_step]` when `scale` is neither `0` nor `1`, `predict_noise` computes `v_orig + scale * (v_edit - v_orig)`. Steps `< first_step` run 1-pass `[original]`; steps `> last_step` run 1-pass `[edited]`.
- **Single-pass media encoding & safety checks (`check_same_but_text`)**:
  - Verifies both branches carry `minimax_token_tags` and matching `minimax_refs` / `minimax_keyframes` counts before sampling begins.
  - Reference workflows encode vision tokens and VAE latents **once** via [`MiniMaxH3EncodeReferences`](../../reference_encode.py) (`single_frame=True` for Pika still edits) and feed two [`MiniMaxH3PromptOnReferences`](../../reference_encode.py) nodes.
  - Guide-latent workflows encode guide media **once** via [`MiniMaxH3GuideLatent`](../../guide_latent.py) (`edited_positive` input -> output slot `2`).
  - [`MiniMaxH3Preflight`](../../preflight.py) (`edited_conditioning` input) prices peak sequence length across both branches. *(Single-frame `P-H3-Image` encodes fast `seed=42` references across two `MiniMaxH3PH3ImageConditioning` nodes; official `MiniMax-H3-Image` uses 6 per-step distilled heads and has not been tested with Edit Guidance.)*
- **Sequential `B = 1` execution (`h3_edit_pass`)**: `_prepare` tags each branch's `minimax_payload` with `h3_edit_pass = "original"` or `"edited"`, preventing ComfyUI from concatenating equal-length `~18k`-token sequences into a `B = 2` OOM and letting wrappers identify each pass.
- **Verification & comparison tools**:
  ```bash
  <comfy-venv-python> bench/check_edit_guidance.py
  <comfy-venv-python> bench/edit_guidance_arms.py --help
  ```
