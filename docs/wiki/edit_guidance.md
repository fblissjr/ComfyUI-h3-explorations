# Steering with an Edited Prompt: `MiniMaxH3EditGuider` and `MiniMaxH3PromptEdit`

Last updated: 2026-11-11.

[`edit_guidance.py`](../../edit_guidance.py) steers a render by comparing two versions of the **same shot's prompt**—`original` and `edited`—directly in the DiT's predicted latent velocity space, with no control vector file required.

For **discrete, multi-attribute edits to a subject, garment, prop, or action** (such as changing `<Subject 1>` from a black knit sweater into `"a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans"` while keeping her likeness from `<Picture 1>` and keeping the diner around her), `MiniMaxH3EditGuider` is the verified primary route in this repo.

---

## 1. How `MiniMaxH3EditGuider` works under the hood

`EditGuider` ([`edit_guidance.py:132-167`](../../edit_guidance.py#L132-L167)) subclasses ComfyUI's `CFGGuider` (`comfy.samplers.CFGGuider`), replacing the standard `(uncond, cond)` classifier-free guidance pair with `(original, edited)`:

1. **Identical non-text conditioning check (`check_same_but_text`)**:
   - Before sampling starts, [`check_same_but_text`](../../edit_guidance.py#L43-L79) verifies that both conditioning lists carry `conditioning_token_tags`, contain the exact same number of tag-`0` vision rows (`<Picture N>` / `<Video N>`), and match on every auxiliary conditioning entry (`reference_latents`, `guide_latents`, audio references, canvas layout).
   - If you accidentally wire different reference images or canvases into the two branches, `MiniMaxH3EditGuider` refuses immediately rather than producing silent spatial drift.
2. **Sequential DiT execution without `B = 2` OOM (`h3_edit_pass`)**:
   - A packed 121-frame H3 audio+video sequence is `~18,000` tokens long. If ComfyUI concatenated `original` and `edited` into a `B = 2` batch (`~36,000` tokens), attention memory would spike and OOM.
   - `set_conds` tags the two conditioning dicts with `h3_edit_pass = "original"` and `h3_edit_pass = "edited"` ([`edit_guidance.py:114-121`](../../edit_guidance.py#L114-L121)). ComfyUI's batch splitter sees unequal keys and runs two clean `B = 1` DiT forward passes sequentially on the identical noisy latent $x_t$.
3. **Latent-velocity extrapolation (`predict_noise`)**:
   - At sampling step index `step` (`0`-based):
     - **Before `first_step` (`step < first_step`) or when `scale == 0.0`**: runs **1 pass** (`[original]`). Early high-sigma steps lock the baseline shot's camera framing, room layout, and `<Subject 1>` pose.
     - **After `last_step` (`step > last_step`) or when `scale == 1.0`**: runs **1 pass** (`[edited]`). Once composition and garment structure are established, late refinement steps follow `[edited]` at `1x` cost.
     - **Inside `[first_step, last_step]` with `scale != 0, 1`**: runs **2 passes** (`[original]` and `[edited]`) on the same $x_t$ and combines their predicted velocities via `comfy.samplers.cfg_function`:
       $$\mathbf{v}(x_t) = \mathbf{v}_{\text{original}}(x_t) + \text{scale} \cdot \left(\mathbf{v}_{\text{edited}}(x_t) - \mathbf{v}_{\text{original}}(x_t)\right)$$

### Why velocity subtraction succeeds where pooled control vectors smear

In a contrastive control vector, `22` tokens describing `"a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans"` are mean-pooled into **one** `5376`-d vector and added equally to every token in `subject_definitions:`. Token-level binding (that `rust-orange` modifies `blouse` while `flared` modifies `jeans`) is collapsed before the DiT ever runs; on our 2026-11-11 diner shot (`s730451892`), that pooled vector produced a cream button blouse rather than rust-orange.

In `MiniMaxH3EditGuider`, **nothing is pooled**:
- Both prompts pass through `Qwen3-VL`, `TokenRefiner`, and all 50 joint audio-video DiT blocks with every token intact.
- Inside cross-attention, `"rust-orange"` and `"blouse"` attend to torso patches, `"flared"` and `"jeans"` attend to leg patches, and `<Picture 1>`'s tag-`0` vision tokens + unchanged diner prose produce matching velocities in both passes.
- Subtracting $\mathbf{v}_{\text{edited}} - \mathbf{v}_{\text{original}}$ cancels the unchanged face, hair, booth, lighting, and spoken dialogue while amplifying *only* the spatial/acoustic regions where the edited words alter the DiT's prediction.

---

## 2. Nodes and parameters

### `MiniMaxH3PromptEdit` (`conditioning/minimax`)

Applies deterministic, validated `old -> new` text replacements to an H3 prompt string so `original` and `edited` stay identical everywhere else ([`edit_guidance.py:170-199`](../../edit_guidance.py#L170-L199)).

- **Inputs**:
  - `original` (`STRING`): the baseline H3 scene prompt.
  - `edits` (`STRING`, multiline): one `old -> new` rule per line (`#` lines and blank lines are ignored).
- **Outputs**:
  - `edited_prompt` (`STRING`): wire to a second text-encode node sharing the same `clip` (and same `encoded_references` on `ref2va`).
  - `report` (`STRING`): per-line summary showing which section matched and word counts (`old_words -> new_words`).
- **Scoping prefixes**:
  - Unprefixed `old -> new`: searches outside `overall_soundscape:` and `non_diegetic_music:`.
  - `soundscape: old -> new`: searches only inside `overall_soundscape:`.
  - `music: old -> new`: searches only inside `non_diegetic_music:`.
  - `<header_name>: old -> new` (e.g., `subject_definitions: ...` or `summary: ...`): searches only inside that named H3 section (`summary`, `retention_analysis`, `subject_definitions`, `integrated_multimodal_description`, `detailed_description`).
- **Built-in safety refusals** ([`edit_guidance.py:17-40`](../../edit_guidance.py#L17-L40)):
  - Refuses if `old` does not appear or appears **more than once** in the scoped span (add a neighbor word to disambiguate).
  - Refuses if an edit touches protected H3 structure: `<d>...</d>` dialogue, `(S1)` speaker tags, `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` references, `[Shot N]` markers, `<0.0 seconds>` timestamps, or section headers.

> **When baseline prose does not name the attribute yet**: On `ref2va` prompts, `subject_definitions:` often anchors `<Subject 1>` via `<Picture 1>` without describing her current sweater. You can either match the end of her `<Subject 1>` definition line in `MiniMaxH3PromptEdit` (`from <Picture 1>. -> from <Picture 1>, wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans.`) or pass a dedicated edited prompt string directly into the second `MiniMaxH3ReferenceTextEncode` node.

### `MiniMaxH3EditGuider` (`sampling/custom_sampling/guiders`)

Drop-in replacement for `BasicGuider` on `SamplerCustomAdvanced` ([`edit_guidance.py:202-251`](../../edit_guidance.py#L202-L251); defaults in [`h3_rules.py:281-286`](../../h3_rules.py#L281-L286)).

| Input | Default | Range | Effect & empirical guidance |
|---|---|---|---|
| `model` | — | `MODEL` | Loaded H3 / Pika DiT model |
| `original` | — | `CONDITIONING` | Encoded baseline prompt |
| `edited` | — | `CONDITIONING` | Encoded edited prompt (must share encoder, references, and canvas with `original`) |
| `scale` | `1.5` | `0.0 .. 5.0` | `0.0` = `original` (`1` pass/step); `1.0` = `edited` (`1` pass/step); `1.5` (default) cleanly locks in wardrobe/material edits more decisively than `scale = 1.0` alone; above `~2.5`, unrequested color saturation creeps into skin and highlights |
| `first_step` | `0` | `0 .. 1000` | `0`-based first step where 2-pass guidance turns on. `0` lets the edit shape silhouette/garment geometry from step 0; setting `first_step = 1` on 6-step Pika locks step-0 camera framing and body pose from `original` before applying the edit on steps `1..last_step` |
| `last_step` | `1000` | `0 .. 1000` | `0`-based last step that runs 2-pass extrapolation. Steps `> last_step` run **1-pass `[edited]`** (`scale = 1.0`). On 6-step Pika (`steps 0..5`), setting `last_step = 2` runs 2-pass on steps `0, 1, 2` and 1-pass `[edited]` on steps `3, 4, 5` (**9 DiT passes** instead of 12—`1.5x` baseline instead of `2.0x`) while keeping the edit locked in |
| `align_positions` | `False` | `BOOLEAN` | Pads the shorter prompt's text span in 3D RoPE cursor space so audio/video tokens start at identical RoPE indices when `len(original) != len(edited)`. Leave `False` unless testing large length deltas (one-shot check on 2026-11-09 showed no benefit on short edits) |

---

## 3. Worked examples

### Example A — Exact `<Subject 1>` clothing swap on `ref2va` (verified winner, 2026-11-11)

- **Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance.json`
- **Setup**:
  - `MiniMaxH3EncodeReferences` encodes `<Picture 1>` **once** and feeds **both** `MiniMaxH3ReferenceTextEncode` nodes (`original` and `edited`), so vision tokens are computed once and match bit-for-bit.
  - `original` keeps the diner `ref2va` prompt as written.
  - `edited` appends `<Subject 1>`'s target outfit to her `subject_definitions:` / `detailed_description:` prose (`"wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans"`).
  - `MiniMaxH3EditGuider`: `scale = 1.5`, `first_step = 0`, `last_step = 2` (or `1000` for full 6-step 2-pass).
- **Verified playback result** (`$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_edit_guidance_style_vs_subject_s730451892_stacked.mp4`):
  - `<Subject 1>` wears the exact rust-orange wide-collar blouse and high-waisted flared jeans in motion, while her face, hair, spoken line, and diner booth stay faithful to `<Picture 1>` and the baseline shot.
  - Note: on `ref2va`, keep `ref_vae_end_percent = 1.0` (default) when changing clothes on the same person—the clean `<Picture 1>` reference latents sit at their own reference time slot (`t = text_len`), so `EditGuider` cleanly overrides her outfit on generated video frames while preserving her facial identity.

### Example B — Coupled lighting + acoustic edit on `t2va`

- **Workflow**: `workflows/h3_text_to_video_pika_edit_guidance.json`
- **`MiniMaxH3PromptEdit` `edits`**:
  ```text
  warm tungsten pendant lamps -> flickering red emergency strobes and drifting white steam
  soundscape: quiet clink of porcelain cups -> rhythmic electronic klaxon pulses and hissing steam pipes
  ```
- **Why use `EditGuider` here**: When an edit replaces specific named light fixtures and specific foley sources inside one shot without needing a reusable library `.safetensors` file, `MiniMaxH3PromptEdit` + `MiniMaxH3EditGuider` (`scale = 1.5`) executes immediately with zero vector build step.

---

## 4. Combining Edit Guidance with Control Vectors

`MiniMaxH3EditGuider` (a `GUIDER` on `SamplerCustomAdvanced`) and `MiniMaxH3ApplyControlVectors` (a `MODEL` wrapper upstream of the guider) compose cleanly on the same graph—for example, using `MiniMaxH3EditGuider` to put `<Subject 1>` in a specific outfit while applying `film_16mm_1974` (`mask = "description_only"`, `strength = 1.0`) on the `MODEL` line to give the room 1974 16mm celluloid grain.

Verification script for edit-guidance invariants and graph wiring:
```bash
/path/to/comfy-venv/bin/python bench/check_edit_guidance.py
```
