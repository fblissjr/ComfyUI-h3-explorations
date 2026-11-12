# Steering with an Edited Prompt: `MiniMaxH3EditGuider` and `MiniMaxH3PromptEdit`

Last updated: 2026-11-11.

[`edit_guidance.py`](../../edit_guidance.py) steers a render by comparing two versions of the **same shot's prompt**—`original` and `edited`—directly in the DiT's predicted latent velocity space, with no control vector file required.

For **discrete, multi-attribute edits to a subject, garment, prop, or action** (such as changing `<Subject 1>` from a black knit sweater into `"a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans"` while keeping her likeness from `<Picture 1>` and keeping the diner around her), `MiniMaxH3EditGuider` is the verified primary route in this repo.

---

## 1. How `MiniMaxH3EditGuider` works under the hood

`EditGuider` ([`edit_guidance.py:95-162`](../../edit_guidance.py#L95-L162)) subclasses ComfyUI's `CFGGuider` (`comfy.samplers.CFGGuider`), replacing the standard `(uncond, cond)` classifier-free guidance pair with `(original, edited)`:

1. **Identical non-text conditioning check (`check_same_but_text`) & `1x` media encoding (`02a20fe3`)**:
   - Before sampling starts, [`check_same_but_text`](../../edit_guidance.py#L72-L79) verifies that both single-entry conditioning lists carry `minimax_token_tags` and match on reference and keyframe counts (`minimax_refs`, `minimax_keyframes`).
   - If you accidentally wire non-H3 conditioning or mismatched references/keyframes into the two branches, `MiniMaxH3EditGuider` refuses immediately rather than producing silent spatial drift.
   - Heavy vision and VAE conditioning work executes **once per run**, never twice:
     - **Reference video & single-frame reference edits**: [`MiniMaxH3EncodeReferences`](../../reference_encode.py#L441-L558) (with `single_frame=True` for Pika still edits) encodes Qwen3-VL vision spans and reference VAE latents once on node `132` and feeds two lightweight [`MiniMaxH3PromptOnReferences`](../../reference_encode.py) text passes (`5` for `original` and `131` for `edited`, reusing the vision KV cache).
     - **Guide-latent video & single-frame edits**: [`MiniMaxH3GuideLatent`](../../guide_latent.py#L142-L312) accepts an optional `edited_positive` conditioning input and emits `edited_positive` on output slot `2` (`outputs = [positive, latent, edited_positive]`), encoding the guide image/video/mask/audio latents once and attaching identical `minimax_keyframes` to both branches.
     - **Preflight peak pricing**: [`MiniMaxH3Preflight`](../../preflight.py#L120-L155) accepts an optional `edited_conditioning` input (`"26".edited_conditioning`), pricing peak `max(seq_orig, seq_edit)` and reporting both branches' prompt and packed sequence lengths.
     - *(Single-frame `P-H3-Image` uses fast 1-frame `seed=42` deterministic reference encoding on two `MiniMaxH3PH3ImageConditioning` nodes without an intermediate split node; official `MiniMax-H3-Image` uses 6 per-step output heads distilled without CFG and has not been tested with Edit Guidance.)*
2. **Sequential DiT execution without `B = 2` OOM (`h3_edit_pass`)**:
   - A packed 121-frame H3 audio+video sequence is `~18,000` tokens long. If ComfyUI concatenated `original` and `edited` into a `B = 2` batch (`~36,000` tokens), attention memory would spike and OOM.
   - `_prepare` tags the two `minimax_payload` dicts with `h3_edit_pass = "original"` and `h3_edit_pass = "edited"` ([`edit_guidance.py:112-131`](../../edit_guidance.py#L112-L131)). ComfyUI's batch splitter sees unequal keys and runs two clean `B = 1` DiT forward passes sequentially on the identical noisy latent $x_t$.
3. **Latent-velocity extrapolation (`predict_noise`)**:
   - At sampling step index `step` (`0`-based):
     - **Before `first_step` (`step < first_step`) or when `scale == 0.0`**: runs **1 pass** (`[original]`). Early high-sigma steps lock the baseline shot's camera framing, room layout, and `<Subject 1>` pose.
     - **After `last_step` (`step > last_step`) or when `scale == 1.0`**: runs **1 pass** (`[edited]`). Once composition and garment structure are established, late refinement steps follow `[edited]` at `1x` cost.
     - **Inside `[first_step, last_step]` with `scale != 0, 1`**: runs **2 passes** (`[original]` and `[edited]`) on the same $x_t$ and extrapolates their predicted velocities directly (`out[0] + (out[1] - out[0]) * self.edit_scale`):
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

Applies deterministic, validated `old -> new` text replacements (and tag-safe `ref2va` attire unlocks) to an H3 prompt string so `original` and `edited` stay identical everywhere else ([`edit_guidance.py:164-213`](../../edit_guidance.py#L164-L213), [`h3_prompt_edit.py`](../../h3_prompt_edit.py)).

- **Inputs**:
  - `original` (`STRING`): the baseline H3 scene prompt.
  - `edits` (`STRING`, multiline): one `old -> new` rule or `release_attire` directive per line (or `;`-separated on one line; blank lines are ignored).
  - `release_subject_attire` (`BOOLEAN`, optional, default `False`): one-click tag-safe unlock for `<Subject 1>` on `ref2va` prompts. Rewrites `<Subject 1>`'s line in `retention_analysis:` from `fully_preserved` to `partially_preserved`, strips contradictory `, and clothing` list items when present, and appends `; clothing is not retained and follows the setting of the scene.` without touching `<Subject N>` / `<Picture N>` tags (`h3_prompt_edit.py:220-278`). Works equally for `MiniMaxH3EditGuider` or as a single-conditioning preprocessor upstream of `MiniMaxH3Steer`!
- **Outputs**:
  - `edited_prompt` (`STRING`): wire to a conditioning node (either the second `MiniMaxH3PromptOnReferences` / `MiniMaxH3Conditioning` node for `MiniMaxH3EditGuider`, or your single `MiniMaxH3ReferenceConditioning` / `MiniMaxH3PromptOnReferences` node when unlocking attire upstream of `MiniMaxH3Steer`).
  - `report` (`STRING`): per-line summary showing which section matched and word counts (`old_words -> new_words`).
- **Scoping prefixes & directives**:
  - Unprefixed `old -> new`: searches the full prompt text (must match uniquely across the entire prompt).
  - `subject_definitions: old -> new` (alias `subjects:`): searches only inside `subject_definitions:`.
  - `summary: old -> new`: searches only inside `summary:`.
  - `retention_analysis: old -> new` (alias `retention:`): searches only inside `retention_analysis:`.
  - `description: old -> new` (aliases `detailed_description:`, `integrated_multimodal_description:`): searches only inside the main visual description section.
  - `soundscape: old -> new` (aliases `audio:`, `foley:`, `sound:`): searches only inside `overall_soundscape:`.
  - `music: old -> new` (aliases `non_diegetic_music:`, `score:`): searches only inside `non_diegetic_music:`.
  - `release_attire: <Subject N>` (aliases `unlock_attire:`, `release_clothing:`; supports comma-separated subjects or `release_attire: all`): unlocks reference clothing in `retention_analysis:` for any `<Subject N>` (not just `<Subject 1>`).
- **Built-in safety refusals** ([`h3_prompt_edit.py:281-379`](../../h3_prompt_edit.py#L281-L379)):
  - Refuses if `old` does not appear or appears **more than once** in the scoped span (add a neighbor word to disambiguate).
  - Refuses if an `old -> new` edit touches protected H3 structure: `<d>...</d>` dialogue, `(S1)` speaker tags, `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` references, `<|...|>` special tokens, `[Shot N]` markers, or section headers.

> **When baseline prose does not name the attribute yet**: On `ref2va` prompts, `subject_definitions:` often anchors `<Subject 1>` via `<Picture 1>` without describing her current sweater. You can either match the end of her `<Subject 1>` definition line in `MiniMaxH3PromptEdit` (`subject_definitions: from <Picture 1>. -> from <Picture 1>, wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans.`) or pass a dedicated edited prompt string directly into the second `MiniMaxH3PromptOnReferences` node.

### `MiniMaxH3EditGuider` (`sampling/custom_sampling/guiders`)

Drop-in replacement for `BasicGuider` on `SamplerCustomAdvanced` ([`edit_guidance.py:215-271`](../../edit_guidance.py#L215-L271); defaults in [`h3_rules.py:281-286`](../../h3_rules.py#L281-L286)).

| Input | Default | Range | Effect & empirical guidance |
|---|---|---|---|
| `model` | — | `MODEL` | Loaded H3 / Pika DiT model |
| `original` | — | `CONDITIONING` | Encoded baseline prompt |
| `edited` | — | `CONDITIONING` | Encoded edited prompt (must share encoder, references, and canvas with `original`) |
| `scale` | `1.5` | `0.0 .. 4.0` | `0.0` = `original` (`1` pass/step); `1.0` = `edited` (`1` pass/step); `1.5` (default) cleanly locks in wardrobe/material edits more decisively than `scale = 1.0` alone; above `~2.5`, unrequested color saturation creeps into skin and highlights |
| `first_step` | `0` | `0 .. 1000` | `0`-based first step where 2-pass guidance turns on. `0` lets the edit shape silhouette/garment geometry from step 0; setting `first_step = 1` on 6-step Pika locks step-0 camera framing and body pose from `original` before applying the edit on steps `1..last_step` |
| `last_step` | `1000` | `0 .. 1000` | `0`-based last step that runs 2-pass extrapolation. Steps `> last_step` run **1-pass `[edited]`** (`scale = 1.0`). On 6-step Pika (`steps 0..5`), setting `last_step = 2` runs 2-pass on steps `0, 1, 2` and 1-pass `[edited]` on steps `3, 4, 5` (**9 DiT passes** instead of 12—`1.5x` baseline instead of `2.0x`) while keeping the edit locked in |
| `align_positions` | `False` | `BOOLEAN` | Pads the shorter prompt's text span in 3D RoPE cursor space so audio/video tokens start at identical RoPE indices when `len(original) != len(edited)`. Leave `False` unless testing large length deltas (one-shot check on 2026-11-09 showed no benefit on short edits) |

---

## 3. Worked examples

### Example A — Exact `<Subject 1>` clothing swap on `ref2va` (verified winner, 2026-11-11)

- **Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`
- **Setup**:
  - `MiniMaxH3EncodeReferences` encodes `<Picture 1>` **once** and feeds **both** `MiniMaxH3PromptOnReferences` nodes (`original` and `edited`), so vision tokens and reference VAE latents are computed once and match bit-for-bit.
  - `original` keeps the diner `ref2va` prompt as written.
  - `edited` appends `<Subject 1>`'s target outfit to her `subject_definitions:` / `detailed_description:` prose (`"wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans"`).
  - `MiniMaxH3EditGuider`: `scale = 1.5`, `first_step = 0`, and the default `last_step` (every step in two passes), as read from the clip's own workflow. A shorter `last_step` was not rendered for this case.
- **Verified playback result** (`$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_edit_guidance_style_vs_subject_s730451892_stacked.mp4`):
  - `<Subject 1>` wears the exact rust-orange wide-collar blouse and high-waisted flared jeans in motion, while her face, hair, spoken line, and diner booth stay faithful to `<Picture 1>` and the baseline shot.
  - Note: on `ref2va`, keep reference release at its default (`end_percent = 1.0` on `MiniMaxH3ReleaseReferences`) when changing clothes on the same person—the clean `<Picture 1>` reference latents sit at their own reference time slot (`t = text_len`), so `EditGuider` cleanly overrides her outfit on generated video frames while preserving her facial identity.

### Example B — Coupled lighting + acoustic edit on `t2va`

- **Workflow**: `workflows/h3_text_to_video_pika_edit_guidance_api.json`
- **`MiniMaxH3PromptEdit` `edits`**:
  ```text
  warm tungsten pendant lamps -> flickering red emergency strobes and drifting white steam
  soundscape: quiet clink of porcelain cups -> rhythmic electronic klaxon pulses and hissing steam pipes
  ```
- **Why use `EditGuider` here**: When an edit replaces specific named light fixtures and specific foley sources inside one shot without needing a reusable library `.safetensors` file, `MiniMaxH3PromptEdit` + `MiniMaxH3EditGuider` (`scale = 1.5`) executes immediately with zero vector build step.

---

## 4. Combining Edit Guidance with Control Vectors

`MiniMaxH3EditGuider` (a `GUIDER` on `SamplerCustomAdvanced`) and `MiniMaxH3Steer` / `MiniMaxH3ReleaseReferences` (or legacy `MiniMaxH3ApplyControlVectors` on the `MODEL` line upstream of the guider) compose cleanly on the same graph—for example, using `MiniMaxH3EditGuider` to put `<Subject 1>` in a specific outfit while applying `film_16mm_1974` (`target = "description_only"`, `strength = 1.0`) on `MiniMaxH3Steer` to give the room 1974 16mm celluloid grain.

Verification script for edit-guidance invariants and graph wiring:
```bash
/path/to/comfy-venv/bin/python bench/check_edit_guidance.py
```
