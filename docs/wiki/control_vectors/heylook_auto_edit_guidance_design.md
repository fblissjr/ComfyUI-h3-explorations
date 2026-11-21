# Design Note: `heylook` Video Prompt Expander (`1x`) & Auto-Edit Guidance (`2x`)

Last updated: 2026-11-20.

Status: **Implemented (`2026-11-19`..`2026-11-20`)** as a two-node separation sharing [`h3_prompt_contract.py`](../../../h3_prompt_contract.py), [`heylook_client.py`](../../../heylook_client.py), [`web/h3_auto_edit_heylook.js`](../../../web/h3_auto_edit_heylook.js), and [`bench/check_auto_edit_prompt.py`](../../../bench/check_auto_edit_prompt.py):
- **`1x` Fresh Prompt Authoring:** [`MiniMaxH3PromptExpander`](../../../h3_prompt_expander.py) on standard `BasicGuider` graphs (`1152x768 @ 192f / 8.00s` default).
- **`2x` Surgical Rewrites:** [`MiniMaxH3AutoEditPrompt`](../../../auto_edit_prompt.py) (`21` inputs) paired with [`edit_guidance.py`](../../../edit_guidance.py) (`MiniMaxH3EditGuider`, `lazy=True` on `original`) and [`control_vector_nodes.py`](../../../control_vector_nodes.py).

## 0. Implemented Architecture (Separated `1x` Expander + `2x` `EditPlan` + Shared Contract)

The shipped implementation separates `1x` fresh scene authoring (native plain-text H3 sections via single-mode zero-lore prompts in [`docs/wiki/system_prompts/h3_video/README.md`](../system_prompts/h3_video/README.md)) from `2x` surgical differential editing (plain Markdown `## section:` + `## axes` blocks; dropping raw JSON replies and Phase 2 `[K, 5376]` concept tokens):

1. **`MiniMaxH3PromptExpander` + [`h3_prompt_contract.py`](../../../h3_prompt_contract.py) (`1x` Easy Button)**:
   - Covers all six H3 modes (`t2va`, `i2va`, `fl2va`, `l2va`, multimodal `ref2va`, and `single_frame`) via standalone zero-lore mode prompts under [`docs/wiki/system_prompts/h3_video/README.md`](../system_prompts/h3_video/README.md) (`auto_edit_prompt.py::resolve_fresh_system_prompt`), falling back to [`fresh_prompt_system_prompt.md`](../system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md) only for combined custom overrides and letting an explicit `heylook` preset's `system_prompt` override win.
   - Python deterministically inspects `H3References` via [`reference_order.py::assign_labels`](../../../reference_order.py) (`<Picture N>`, `<Video N>`, `<Audio N>`), samples video keyframes for `heylook` vision, stamps exact `vendor_guides/base_en.md` Part One lines and `S.SS` durations from `length`, enforces `fl2va` single-continuous-`[Shot 1]` transition pacing, routes synthetic trajectories to `video_end_percent = 0.33`, and enforces `bench/preflight_graph.py` mechanical checks (`350–500` words on `ref2va` with `[video editing]` exempt, `(Sx)` before `<d>`, zero `(Sx)` in `retention_analysis:`) with zero internal repo paths in repair feedback.
   - Ships three `1x` `BasicGuider` workflows (`1152x768 @ 192f`, `PreviewAny` prompt display, `PrimitiveInt` shared `length` on `fl2v`): [`workflows/h3_text_to_video_pika_prompt_expander_api.json`](../../../workflows/h3_text_to_video_pika_prompt_expander_api.json), [`workflows/h3_first_last_frame_pika_prompt_expander_api.json`](../../../workflows/h3_first_last_frame_pika_prompt_expander_api.json), and split-cached `MiniMaxH3EncodeReferences` + `MiniMaxH3PromptOnReferences` [`workflows/h3_image_ref_plus_text_to_video_pika_prompt_expander_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_prompt_expander_api.json), with live `model`/`preset`/`concept` dropdowns in [`web/h3_auto_edit_heylook.js`](../../../web/h3_auto_edit_heylook.js).
2. **`MiniMaxH3AutoEditPrompt` + `EditPlan` (`2x` Surgical Rewriter, `auto_edit_prompt.py`)**:
   - Stays focused at `21` inputs on surgical rewrites of an existing `original` prompt using `section_edits` (default, plain Markdown `## section:` blocks, [`section_edits_system_prompt.md`](../system_prompts/h3_auto_edit/section_edits_system_prompt.md)) or `edit_lines` ([`edit_lines_system_prompt.md`](../system_prompts/h3_auto_edit/edit_lines_system_prompt.md)) routed through [`h3_prompt_edit.py`](../../../h3_prompt_edit.py).
   - Validates rewrites via `h3_prompt_contract.py` + tag/dialogue invariants with 1-shot APC repair, refreshes Part One `S.SS` on keyframe prompts, and emits `EditPlan` (`MINIMAX_H3_EDIT_PLAN`) driving `MiniMaxH3EditGuider` (`lazy=True` on `original`, with `Node 26` `MiniMaxH3Preflight` routed on `edited` so unmasked `1x` `fast_1x` runs skip the duplicate `Qwen3-VL` pass) and `MiniMaxH3ReleaseReferences`.
3. **Shared `heylook` Client & Clean Concept Presets**:
   - [`heylook_client.py`](../../../heylook_client.py) is shared across `MiniMaxH3ImagePromptExpander`, `MiniMaxH3PromptExpander`, and `MiniMaxH3AutoEditPrompt` (`resolve_heylook_base_url`, capability-gated `reasoning_effort`, client-side `2048 px` image resize, `503` backoff, interrupt `DELETE /v1/requests/{id}`).
   - Loads curated concept cards from `data/edit_concepts/*.toml` first (incl. lighting presets `look_crimson_darkroom`, `look_direct_1990s_flash`, `look_lurid_magenta_neon` and narrative/costume presets), followed by `data/control_vectors/*.toml` and `prompt_bank/control_vectors/*.toml`, filtering out short `< 25`-word fragment exemplars during fresh authoring.
4. **Shipped `2x` Auto-Edit API Workflows**: [`workflows/h3_text_to_video_pika_auto_edit_guidance_api.json`](../../../workflows/h3_text_to_video_pika_auto_edit_guidance_api.json) and [`workflows/h3_image_ref_plus_text_to_video_pika_auto_edit_guidance_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_auto_edit_guidance_api.json) leave `base_url=""` and `model=""` so no host-specific addresses are tracked.

This note explains the two-stage H3 text/DiT mechanism behind **Edit Guidance** ([`edit_guidance.md`](../edit_guidance.md)) vs. **Control Vectors** ([`control_vectors.md`](../control_vectors.md))—plugging a local `heylook` (`heylookitsanllm`) server upstream via [`heylook_client.py`](../../../heylook_client.py).

---

## 1. Why This Works Architecturally

### The two-stage H3 text bottleneck

Every MiniMax H3 prompt runs through two text stages before the 50 DiT blocks see it ([`control_vectors.md`](../control_vectors.md#2-how-the-three-mechanisms-fit-h3s-architecture)):

1. **Causal Qwen3-VL-32B encoder (`5120`-d)** + **bidirectional `condition_proj` & 2-block `TokenRefiner` (`5376`-d)** produce a sequence of `L` post-refiner context rows `[B, L, 5376]`.
2. **50 joint audio-video DiT blocks** cross-attend video and audio queries to every token row independently across `L`.

Because of **where subtraction happens**, single-vector steering and Edit Guidance operate in fundamentally different regimes:

| Mechanism | Where the delta happens | Dimensionality seen by DiT | Binding & Failure Mode |
|---|---|---|---|
| **Contrastive Control Vector (`MiniMaxH3Steer`)** | Averages `diff_span` tokens to **one `[5376]` vector** and broadcasts it across selected rows *before* the 50 DiT blocks (`1x` DiT pass). | `1` global direction (rank-1 shift per row) | Strong for global optical/atmosphere/physics shifts (`film_16mm_1974`, `cyberpunk_neon_rain`, `pyro_chain_explosions`), but loses word order and per-noun binding (e.g., swapping `"black sweater -> 1970s rust-orange wide-collar blouse and flared jeans"` blurred colors into a cream blouse). |
| **Edit Guidance (`MiniMaxH3EditGuider`)** | Keeps both `original` and `edited` prompts **fully tokenized** through all 50 DiT blocks and subtracts predicted latent velocities *after* cross-attention (`2x` guided passes, `9–10` total with `last_step = 3`). | Full `[L_orig, 5376]` vs. `[L_edit, 5376]` sequences | Unchanged subjects, dialogue, camera, and room layout cancel out in $(\mathbf{v}_{\text{edited}} - \mathbf{v}_{\text{original}})$, while every rewritten token (`"rust-orange"`, `"blouse"`, `"flared"`, `"jeans"`) binds directly to its target spatial/temporal patches and supports `[F, H, W]` masking (`MiniMaxH3SpatialDirector`). |

### The core insight

Control vectors are convenient **not** because `5376`-d vector addition is inherently better than tokenized text—often it is worse—but because you pick a named concept once (for example, `claymation_stopmotion`, `cyberpunk_neon_rain`, or a wardrobe/material transformation) and apply it to *any* prompt without hand-authoring `old wording -> new wording` lines.

If a fast local LLM (`heylook`) sits *upstream* of `MiniMaxH3EditGuider` and translates `(base_prompt, concept_spec)` into a surgical, tag-preserving `edited_prompt` on the fly:

1. **You keep Control Vector UX**: pick a `.toml` stem (or short natural-language concept) plus a `scale` dial, with zero manual prompt surgery.
2. **You get Edit Guidance mechanics**: the DiT sees full tokenized cross-attention across all 50 blocks, preserving noun-to-attribute binding, clean `first_step` / `last_step` scheduling, `streams` (`both` / `video_only` / `audio_only` / `independent`), and `[F, H, W]` spatial/temporal masks.
3. **You stay faithful to the `.toml` bank's intent**: a `.toml` bank's `[vector]` metadata (`quality`, `a_side`, `b_side`) and `[[pair]]` examples already specify the exact transformation—serving as structured few-shot context for `heylook` rather than collapsing them to a single `5376`-d mean.

---

## 2. End-to-End Architecture

```text
  base_prompt ─────────────────────────┬──▶ [original] Qwen3-VL encode (1x) ──────┐
                                       │                                          ▼
  concept (.toml stem / short spec) ───┼──▶ heylook POST /v1/messages ──▶ [edited] Qwen3-VL encode (1x)
                                       │    (rewrites only allowed sections,       │
                                       │     relaxes retention_analysis if needed) ▼
                                       └──────────────────────────────────▶ MiniMaxH3EditGuider
                                                                            (v0 + scale * (v1 - v0) * mask)
```

### Component 1: Concept Source (Reusing Existing `.toml` Banks + Ad-Hoc Specs)

Any file under `prompt_bank/control_vectors/*.toml` (or user-authored `.toml` banks) already contains:
- `[vector].quality`, `a_side`, `b_side` — concise specification of the transition.
- `[[pair]]` (`a`, `b`) — concrete H3-formatted before/after examples showing exact clause phrasing across `integrated_multimodal_description:` and `overall_soundscape:`.

A proposed `MiniMaxH3AutoEditPrompt` node accepts either:
- a `concept_toml` dropdown (reading `[vector]` + 2–3 `[[pair]]` exemplars as few-shot demonstrations), **or**
- a short free-text `edit_intent` string (e.g., `"turn the kitchen into a rain-slicked neon night exterior and switch his jacket to rust-orange corduroy"`).

### Component 2: `heylook` Rewrite Client (Reusing `h3_image_nodes.py` Helpers)

[`h3_image_nodes.py`](../../../h3_image_nodes.py) already implements the full `heylookitsanllm` `/v1/messages` client contract. An auto-edit prompt node should reuse those exact primitives rather than duplicating HTTP logic:

| Existing Helper in `h3_image_nodes.py` | Role in Auto-Edit Prompt Rewriting |
|---|---|
| `normalize_heylook_base_url` | Validates `http(s)://<host>:<port>` without trailing slash (`127.0.0.1` / LAN IP so `heylook`'s `Host` check passes). |
| `discover_heylook_model` | Queries `GET /v1/models` at runtime (`capabilities` gated on `"chat"`—no hardcoded model IDs). |
| `expand_heylook_preset` | Optional preset support from `GET /v1/presets`. |
| `build_heylook_request_body` | Capability-gates `thinking` / `reasoning_effort`, omits `max_tokens` when using model/preset defaults, applies `seed`, strips unsupported `request_fields`. |
| `with_heylook_backoff` | Retries `503` (`model_overloaded`) with `Retry-After` backoff on single-user local servers. |
| `stream_heylook_message` | Sends a fresh `X-Request-ID`, streams `/v1/messages` SSE, cancels via `DELETE /v1/requests/{id}` on ComfyUI interrupt, maps `event: error` cleanly, and joins only `text` blocks (dropping `thinking` blocks). |

*(Note: When `heylook` runs on a separate Apple Silicon host on the LAN, the rewrite step adds ~0.3–1.5s of off-GPU latency while keeping 100% of local NVIDIA VRAM dedicated to ComfyUI's H3 models.)*

### Component 3: Structural Contract & Deterministic Guardrails

Just like `MiniMaxH3PromptEdit` in [`edit_guidance.py`](../../../edit_guidance.py#L480-L518), an LLM rewriter must never silently corrupt H3's structural markers. Before returning `edited_prompt`, a deterministic post-validator should enforce:

1. **Section invariance**: Output contains the exact same top-level sections as `base_prompt` (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`, or `integrated_multimodal_description:` on `t2va`).
2. **Scope masking (`target_scope`)**:
   - `"scene_only"` / `"description_only"` / `"look_only"` / `"soundscape_only"` / `"all_visual"`: Unselected sections are copied **byte-for-byte** from `base_prompt`, even if the LLM drifted.
3. **Tag & dialogue protection**:
   - All `[Shot N]`, `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`, `(S1)/(S2)`, and `<d>...</d>` spoken dialogue spans in `edited_prompt` must match `base_prompt` verbatim unless an explicit dialogue rewrite flag is enabled.
   - Because `MiniMaxH3EditGuider` aligns best when unchanged prose stays identical, the system prompt instructs `heylook` to make **minimal local edits** (mirroring `edit_guidance.md` §3) rather than paraphrasing untouched sentences—so `v_edited - v_original` isolates only the intended change.
4. **Automatic `retention_analysis:` unlocking on `ref2va`**:
   - When `base_prompt` is a `ref2va` prompt containing `fully_preserved - ... clothing are retained` (or hair/age/silhouette) and the concept touches appearance or wardrobe, keeping `fully_preserved` untouched causes the DiT's preservation contract to fight the edit.
   - The rewriter can either output an updated `retention_analysis:` line (`partially_preserved - facial identity is retained while ... adapts`) or emit an explicit `retention_analysis` edit string alongside `edited_prompt`.

### Component 4: `ref2va` Split-Cache & `MiniMaxH3EditGuider` Wiring

On `ref2va` workflows ([`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json)):

1. **`MiniMaxH3EncodeReferences`** runs **once** on `<Picture 1>` to cache the Qwen3-VL vision prefix and reference VAE latents.
2. **Two `MiniMaxH3PromptOnReferences` nodes** encode `base_prompt` (`original`) and `edited_prompt` (`edited`) cheaply against that cached prefix.
3. **`MiniMaxH3EditGuider`** runs the 2-pass subtraction across `first_step .. last_step` (e.g., `first_step = 1, last_step = 3, scale = 1.5`), optionally paired with:
   - **`streams = "independent", audio_scale = 1.0`** when visual edits exceed `scale >= 1.65` (preventing audio clipping), or **`streams = "video_only"`** when soundscape was left unmodified.
   - **`MiniMaxH3SpatialDirector`** `.mask` (`[F, H, W]`) to restrict the auto-generated edit to a moving region or subject mask.
   - **`MiniMaxH3ReleaseReferences`** (`end_percent = 0.25 .. 0.33` when changing 3D silhouette/material like `claymation_stopmotion`, or `1.0` for lighting/color/wardrobe grades).

---

## 3. Worked Neutral Examples

### Example A — Neutral `t2va` Style + Environment Shift (`cyberpunk_neon_rain`)

Suppose `base_prompt` is a quiet daylight market shot (`t2va`):

- **What `heylook` receives**: `base_prompt` + `[vector]` summary + 2 `[[pair]]` excerpts from `prompt_bank/control_vectors/cyberpunk_neon_rain.toml`, with `target_scope = "all_visual"` and instruction to keep `[Shot N]`, `(S1)/(S2)`, and `<d>...</d>` dialogue untouched.
- **What `heylook` produces (`edited_prompt`)**: Swaps `"crates of citrus ... roof vents ... late morning"` $\rightarrow$ `"rain-slicked market stalls under flickering cyan and magenta neon signs at midnight, puddles reflecting holographic signage"`, leaving `"A stallholder in her fifties ... (S1) ... says: <d>[English] Last of the good ones. After this it is all imports.</d>"` word-for-word identical.
- **Why this beats a single `5376`-d vector**: Every new noun (`"puddles"`, `"cyan"`, `"magenta"`, `"neon signs"`) binds to the actual market counter and wet floor patches via 50-block cross-attention, and wiring `first_step = 1, last_step = 3, scale = 1.5` locks step `0`'s actor blocking and stall geometry before pouring neon rain onto steps `1..3` (the exact mechanism verified in [`edit_guidance.md` §4 Example 3](../edit_guidance.md#4-verified-examples-where-a-plain-prompt-edit-fails)).

### Example B — Neutral `ref2va` Multi-Attribute Wardrobe Swap

Suppose `base_prompt` has `<Subject 1>` in `subject_definitions:` wearing `"a black sweater"` and `retention_analysis:` says `fully_preserved - the face, hair, and black sweater are retained`.

- **What `heylook` receives**: `base_prompt` + `edit_intent = "swap sweater for a 1970s rust-orange wide-collar blouse and flared jeans"`, `allow_retention_relax = True`.
- **What `heylook` produces (`edited_prompt`)**:
  - `subject_definitions:`: replaces `"black sweater"` $\rightarrow$ `"1970s rust-orange wide-collar blouse and flared blue denim jeans"`.
  - `retention_analysis:`: updates `"and black sweater are retained"` $\rightarrow$ `"are retained; clothing changes to a 1970s rust-orange wide-collar blouse and flared jeans"`.
  - `detailed_description:` / `overall_soundscape:`: left byte-identical so the room, camera motion, and acoustics cancel out cleanly in $(\mathbf{v}_{\text{edited}} - \mathbf{v}_{\text{original}})$.
- **Why this beats a pooled vector**: In [`control_vectors.md` §1](../control_vectors.md#1-which-tool-to-reach-for-and-why), pooling the `6 -> 22` token rewrite into one `5376`-d vector lost color-to-garment binding (rendering a cream blouse), whereas tokenized `MiniMaxH3EditGuider` at `scale = 1.5` preserved `<Subject 1>`'s face, hair, room, and exact rust-orange/blue garment binding.

---

## 4. Known Failure Modes & Mitigations

| Risk | Why it happens | Mitigation in the node |
|---|---|---|
| **LLM paraphrases unchanged lines** | Rewording untouched sentences shifts token indices and injects noisy `(v_edit - v_orig)` velocity across the background. | Instruct minimal in-place span edits; fallback option to output `old -> new` diff lines and run them through `MiniMaxH3PromptEdit` (which refuses out-of-scope edits and protects tags deterministically). |
| **Thinking tokens leak into `edited_prompt`** | Thinking-capable models on `heylook` emit `thinking` blocks before `text` blocks. | `stream_heylook_message` already separates block types and joins only `text` blocks; optionally gate `thinking = "off"` via `build_heylook_request_body` for sub-second rewrites. |
| **Unrequested audio / dialogue drift** | Changing scene mood tempts the LLM to edit `<d>...</d>` lines or add music. | Deterministic validator rejects/reverts any diff inside `<d>...</d>` or locked sections (`non_diegetic_music:`, `overall_soundscape:` when not targeted). |
| **`2x` DiT pass cost on long clips** | `EditGuider` evaluates `[original, edited]` on guided steps. | Default `last_step = 3` (`9–10` DiT passes on 6-step Pika instead of `12`), and pair with `MiniMaxH3EncodeReferences` so Qwen3-VL vision tokens are encoded once. |
| **ComfyUI node cache vs. non-deterministic LLM** | Re-running with `seed` fixed should reproduce `edited_prompt`. | Forward `seed` into `build_heylook_request_body` (gated on `"seed" in request_fields`) and display the resolved `edited_prompt` string on the node's UI output / report for inspectability. |

---

## 5. Complementary Path: Pairing with Unpooled Concept Tokens (`1x` Pass)

Even with `heylook` auto-rewriting for `MiniMaxH3EditGuider`, there is a useful `1x`-pass middle ground for fast previewing before paying `2x` guided passes:

- Instead of pooling `diff_span` tokens down to a single `[1, 5376]` vector, a build-time variant can retain `K` (`~8–16`) clustered/orthogonal post-refiner concept token rows (`[K, 5376]`) and append them to `context` after `TokenRefiner` (`[B, L + K, 5376]`).
- That preserves separation across distinct concepts in one `1x` pass—serving as a quick draft dial before flipping the exact same `.toml` stem to `heylook` + `MiniMaxH3EditGuider` when full 50-block syntactic binding or `[F, H, W]` spatial masking is needed.
