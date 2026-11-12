# Edit Guidance: Steering a Render by Editing Your Own Prompt

Last updated: 2026-11-12.

**Edit Guidance** lets you change specific parts of a shot—an outfit, a prop, the lighting, or the sound—simply by writing `old wording -> new wording` and turning a strength dial. You do not need to train anything, write prompt pairs, or build a file first.

At each denoising step, the model looks at the same in-progress video/audio under both your **original prompt** and your **edited prompt**, compares how each prompt wants to update that step, and scales the difference by the `scale` you choose. What you did *not* change—the character's face, their spoken lines, the camera movement, and the room around them—mostly cancels out.

---

## 1. Which option should I use?

Before paying for a 2-pass guider, pick the lightest tool that solves your shot:

### Option A — Just edit the prompt (`BasicGuider`, `1x` render speed)
Best when you are generating a fresh shot and the visuals and audio **agree** with each other (for example, changing both the scene description and the soundscape into a thunderstorm from step `0`).
- *Tip*: You can still wire **`MiniMax H3 Prompt Edit`** into a normal single-pass conditioning + `BasicGuider` workflow whenever you want safe `old -> new` replacements that protect `<d>...</d>` dialogue and `<Picture 1>` tags without paying `EditGuider`'s 2-pass cost.

### Option B — [Control Vectors](control_vectors.md) (`MiniMax H3 Steer`, `1x` speed by default)
Our control vectors operate in H3's LoRA-aware post-refiner space and can be built from multi-scene prompt pairs (with `[[hold]]` side-effect scrubbing), extracted from reference image pairs (`MiniMax H3 Make Media Control Vector`), or synthesized on the fly in `< 2 ms` from words via J-Lens (`MiniMax H3 Word Vector`). Reach for a Control Vector when you want:
- **Reusable camera, mood, physics, or acoustic dials** (`film_16mm_1974`, `noir`, `pyro_chain_explosions`, `reverb_cathedral`) across any prompt at `1x` speed,
- **Separate video vs. audio control or section masks** (`streams = "video_only"` or `"audio_only"`, `target = "description_only"`, `"soundscape_only"`, etc.), or
- **Geometric concept removal / holds** (`mode = "project_out"`, `hold_words`, `hold_vector`—for example adding a snowstorm while holding out `"coat, parka"`), which Edit Guidance cannot do.

### Option C — Edit Guidance (`MiniMax H3 Edit Guider`, `1.5x–2x` render time)
Whereas a single `MiniMax H3 Steer` node pushes along one concept direction across its target section, Edit Guidance runs **both full prompts** through all 50 DiT blocks and scales the difference in their predicted video/audio updates. It earns its extra pass cost in **four specific situations**:

| Situation | Why Edit Guidance beats a plain prompt edit or single vector |
|---|---|
| **1. Something else in the shot is fighting your edit (`scale = 1.5..2.0`)** | When `<Picture 1>` strongly locks a character's old sweater on `ref2va`, or when calm elevator visuals mute a loud `soundscape:` edit, a normal prompt edit (`scale = 1.0`) gets damped. Turning `scale` to `1.5` amplifies *only* the edit difference so it punches through. |
| **2. Detailed, multi-part outfits or multiple distinct props at once** | Because both prompts run through cross-attention word for word, `"rust-orange"` stays bound to `"blouse"` while `"blue denim"` stays bound to `"flared jeans"`—without needing to build a vector first. |
| **3. Keeping an existing take's exact camera framing and pose (`first_step = 1`)** | Changing 3 words to 7 words at step `0` shifts token positions and often reshuffles camera framing and actor pose on the same seed. Setting `first_step = 1` locks step `0` to your original prompt first, then applies your edit on top. |
| **4. Partial blends (`0 < scale < 1`) or "negative prompt" cleanup** | `scale = 0.5` gives a 50% blend without hunting for weaker adjectives; replacing an unwanted implied trait with its opposite (like `steam -> crystal-clear air` in a hot kitchen) at `scale = 1.5..2.0` actively pushes the render away from it. |

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
| `scale` | `1.5` (`0.0 .. 4.0`) | **How strongly to push the edit.** `0.0` gives the original prompt; `0.5` gives a subtle halfway blend; `1.0` equals the edited prompt on its own; `1.5` (default) gives an extra push that reliably locks in new clothes, props, or fighting audio edits. Above about `2.5`, collateral differences get amplified too (adding extra color saturation to skin tones or room acoustics to audio). |
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

## 4. Examples (and why each one uses Edit Guidance)

### Example 1 — Changing a reference character's outfit (`ref2va`, verified on renders)
- **Why use Edit Guidance over a plain prompt edit?** `<Picture 1>`'s vision and reference latents strongly anchor the character's original clothes, so a `scale = 1.0` rewrite often bleeds the old sweater back in. Pushing `scale = 1.5` overrides `<Picture 1>`'s clothing inertia while keeping her `<Picture 1>` face, hair, and spoken dialogue intact (verified on the diner 1970s outfit clip and parking-lot samurai armor clip).
- **Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json)
- **Node settings**: `release_subject_attire = True` on `MiniMax H3 Prompt Edit`; `scale = 1.5`, `first_step = 0` on `MiniMax H3 Edit Guider` (add `last_step = 2` for `1.5x` speed).
- **`edits`**:
  ```text
  subject_definitions: face, hair, and clothing are carried -> face and hair are carried, now wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans
  description: preserving the face, hair, wardrobe and build established in the reference -> preserving the face, hair, and build established in the reference while wearing the rust-orange wide-collar blouse and flared jeans
  ```
- **Good to know**: Leave `MiniMaxH3ReleaseReferences` (if present) at its default `end_percent = 1.0` when changing clothes on the same person—keeping `<Picture 1>` pinned preserves facial likeness while `EditGuider` replaces the outfit.

### Example 2 — Swapping a prop or action while keeping an existing take's framing (`t2va`, pattern verified on renders)
- **Why use Edit Guidance over a plain prompt edit?** Replacing 3 words with 7 words at step `0` shifts token offsets and changes camera framing, background layout, and actor pose even on the same seed. Setting `first_step = 1` runs only your original prompt on step `0`—locking the exact shot framing and blocking—and then swaps the prop on steps `1..2` (pattern verified on mug-swap and watch-check edits).
- **Workflow**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Node settings**: `scale = 1.5`, `first_step = 1`, `last_step = 2`.
- **`edits`**:
  ```text
  crates of citrus -> wicker baskets of glowing blue bioluminescent mushrooms
  stacks fruit into a pyramid -> arranges glowing blue mushrooms into a cluster
  ```

### Example 3 — Pushing a loud audio change past calm visuals (`t2va`, verified on renders)
- **Why use Edit Guidance over a plain prompt edit?** Because H3 generates video and audio together, calm visual prose actively dampens a dramatic audio-only edit! On [`prompt_bank/t2va_night_porter.txt`](../../prompt_bank/t2va_night_porter.txt) (where the visuals stay a quiet, unhurried elevator ride), simply rewriting `soundscape:` (`scale = 1.0`) reached `-24.8 dBFS`, whereas `MiniMax H3 Edit Guider` (`scale = 1.5, last_step = 2`) punched **`+6.2 dB` louder** (`-18.6 dBFS`, with **`+8.5 dB` more `< 250 Hz` sub-bass**) while keeping the elevator visually calm and all 7 sung words clean. *(By contrast, if you rewrite **both** the visuals and the soundscape to explode together, a plain `1x` prompt edit already gets most of the way there.)*
- **Node settings**: `scale = 1.5`, `first_step = 0`, `last_step = 2`.
- **`edits`** (on [`prompt_bank/t2va_night_porter.txt`](../../prompt_bank/t2va_night_porter.txt)):
  ```text
  soundscape: A low mechanical hum fills the lift car under the soft rattle of trolley castors. Cables tick behind the wall panel, fabric shifts against the handrail, and the doors part with a slow pneumatic sigh. -> Blaring industrial klaxon alarm sirens wail loudly inside the lift car over thunderous concussive sub-bass explosions and metallic shockwave reverberations as the doors part.
  ```

### Example 4 — Pose-locked medium shift or 50% style blend (verified on renders)
- **Why use Edit Guidance over a plain prompt edit?** If you want a 100% claymation shot from scratch and don't care about matching a live-action take's layout, just edit the prompt at `1x` speed. Use `EditGuider` when you either want to **keep the live-action take's step-0 framing** (`first_step = 1, scale = 1.5`) or dial in a **halfway 50% blend** (`first_step = 0, scale = 0.5`).
- **Node settings**: `scale = 1.5`, `first_step = 1`, `last_step = 2` (pose-locked) **or** `scale = 0.5`, `first_step = 0` (50% blend).
- **`edits`**:
  ```text
  Live-action, cinematic, -> Claymation stop-motion with fingerprinted plasticine surfaces,
  ```

### Example 5 — Removing an unwanted implied detail ("negative prompt" recipe, unrendered example)
- **Why use Edit Guidance over a plain prompt edit?** Even after you delete the word `"steam"` from a restaurant kitchen prompt, `"four burners lit"` and `"tossing a pan"` still make the model draw hazy steam at `scale = 1.0`. Replacing `"steam"` with `"crystal-clear air"` at `scale = 1.5..2.0` actively subtracts the steam direction:
- **Node settings**: `scale = 1.5..2.0`, `first_step = 0`, `last_step = 2`.
- **`edits`**:
  ```text
  four burners lit, steam crossing the lens -> four unlit cold burners, crystal-clear air across the lens
  steam crossing the lens twice -> clear air across the pass
  in a blur of steam -> in sharp, unobstructed focus
  ```

### Example 6 — Changing vocal timbre around locked spoken dialogue (unrendered example)
- **Why use it?** `MiniMax H3 Prompt Edit` protects `<d>...</d>` dialogue and `(S1)`/`(S2)` tags so spoken words stay intact while you change the voice description right before them. Start at `1x` speed with `BasicGuider` first; switch to `MiniMax H3 Edit Guider` (`scale = 1.5` or `first_step = 1`) when noisy/quiet room prose overrides the voice change at `scale = 1.0` or when you want to preserve exact step-0 camera framing:
- **`edits`**:
  ```text
  a warm, gravelly alto -> a breathless, conspiratorial whisper
  a quick, bright tenor -> a deep, booming bass-baritone
  ```

### Example 7 — Combining Edit Guidance with a Control Vector
`MiniMax H3 Edit Guider` (on `SamplerCustomAdvanced`) and `MiniMax H3 Steer` (on the `MODEL` wire upstream) compose cleanly on one graph—for example, using `MiniMax H3 Edit Guider` (`scale = 1.5`) for a multi-piece `<Subject 1>` outfit swap while `MiniMax H3 Steer` applies `film_16mm_1974` (`target = "description_only"`, `strength = 1.0`) for reusable 1974 16mm film grain.

---

## 5. Under the hood (technical reference)

For readers inspecting [`edit_guidance.py`](../../edit_guidance.py), [`h3_prompt_edit.py`](../../h3_prompt_edit.py), or [`h3_rules.py`](../../h3_rules.py):

- **`CFGGuider` subclass (`EditGuider`)**: Subclasses `comfy.samplers.CFGGuider` without patching ComfyUI core (following `Guider_DualCFG` in `comfy_extras/nodes_custom_sampler.py`). On steps inside `[first_step, last_step]` when `scale` is neither `0` nor `1`, `predict_noise` computes `v_orig + scale * (v_edit - v_orig)`. Steps `< first_step` run 1-pass `[original]`; steps `> last_step` run 1-pass `[edited]`. Note that `scale` currently multiplies the full `NestedTensor((video, audio))` delta uniformly (unlike `MiniMaxH3Steer`, which offers `streams = "video_only" | "audio_only"`).
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
