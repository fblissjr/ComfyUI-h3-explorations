# MiniMax-H3-Image Standalone System Prompts (`docs/wiki/system_prompts/h3_image/`)

Self-contained, English-only system prompts for expanding short requests into finished **MiniMax-H3-Image** captions (`workflows/image/h3_image_{t2i,edit}_api.json` and `h3_image_{t2i,edit}_heylook_api.json`, documented in [`docs/wiki/image/h3_image.md`](../../image/h3_image.md)).

Every prompt file in this folder is **pure standalone prompt text**—no outer markdown fences, no repo paths, no internal model lore, and no external cloud API assumptions—ready to paste as the system prompt in a local thinking LLM/VLM (such as `heylookitsanllm`).

> **Model boundary:** These prompts target MiniMax's official 6-step **MiniMax-H3-Image** checkpoint. For PrunaAI's 8-step **P-H3-Image** model (`Edit <Picture 1>. ... Keep ... unchanged.`), use [`../h3_p_h3_image_system_prompt.md`](../h3_p_h3_image_system_prompt.md). The unmodified, hash-pinned vendor middleware prompts from the release live in [`vendor_guides/h3_image/prompt/`](../../../../vendor_guides/h3_image/prompt/).

---

## Which Prompt to Pick

| File | Mode & Scope | Output Format | When to Use |
| :--- | :--- | :--- | :--- |
| [`t2i_system_prompt.md`](t2i_system_prompt.md) | **General T2I** — scenes, photography, film stills, characters, creatures, environments, illustration (plus brief in-scene signs if requested) | Single plain-text paragraph | Manual copy-paste into `MiniMaxH3ImageConditioning.prompt` when you pick the ratio yourself on `MiniMax H3 Image Size` |
| [`t2i_aspect_ratio_system_prompt.md`](t2i_aspect_ratio_system_prompt.md) | Same as `t2i_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <10 presets>` | Local scripts or automated `heylookitsanllm` runs that parse both prompt and canvas shape |
| [`t2i_design_system_prompt.md`](t2i_design_system_prompt.md) | **Design & Typography T2I** — posters, covers, typography/logic boards, infographics, charts, UI/dashboards, character/product sheets, multi-panel diagrams | Single layout-mapped plain-text paragraph | Manual copy-paste for layout-heavy, multi-slot, or data-rich graphics without polluting general scene prompts |
| [`t2i_design_aspect_ratio_system_prompt.md`](t2i_design_aspect_ratio_system_prompt.md) | Same as `t2i_design_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <10 presets>` | Automated runs for posters, infographics, UI screens, and design sheets |
| [`edit_system_prompt.md`](edit_system_prompt.md) | **Image Editing** (`1–16` `<Picture N>` references) — 3-part continuous `<Subject N>` visual delta (`Part 1` bindings $\rightarrow$ `The target image ...` + 4 preservation clauses $\rightarrow$ `Part 3` changed/new delta only) | Single plain-text paragraph | Manual copy-paste into `MiniMaxH3ImageConditioning.prompt` with `MiniMax H3 Image Size` set to `same as the picture` (requires a vision-capable model that sees the input images) |
| [`edit_aspect_ratio_system_prompt.md`](edit_aspect_ratio_system_prompt.md) | Same as `edit_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <same as the picture \| W:H>` | Automated VLM edit runs where crops, outpaints, or explicit target shapes may change the canvas ratio |

---

## Why These Differ from `vendor_guides/h3_image/prompt/`

MiniMax's raw middleware files (`15.8 KB` for T2I, `38.5 KB` for Edit) are built for their hosted `/v1/h3_image_context_ir` wrapper and XGrammar JSON schema decoding. This suite preserves **100% of the target caption structure** the DiT expects (verified against [`vendor_guides/h3_image/examples/revised_prompt.t2i.txt`](../../../../vendor_guides/h3_image/examples/revised_prompt.t2i.txt) and [`revised_prompt.edit.txt`](../../../../vendor_guides/h3_image/examples/revised_prompt.edit.txt)) while fixing four local-LLM bottlenecks:

1. **Unconstrained plain-text generation (no JSON diversity collapse):** Instead of forcing `{ "expanded_caption": "..." }` with `\"` quote-escaping across 400 words under constrained JSON decoding, the model writes natural plain text first and (in the ratio variants) appends one clean `aspect_ratio: ...` line at the end.
2. **Scene vs. Design/Infographic separation:** Splitting the 6 infographic/UI/board contracts into `t2i_design_*` keeps general `t2i_*` prompts compact (~430 words) so design-board callouts, headers, and swatch strips never bleed into photographic or artistic renders.
3. **Zero synthetic API pre-parts for Edit:** You do not need to wrap edit requests in caller-generated `Reference Media: [aspect_ratio W:H]` JSON metadata blocks. Attach the reference images in order (`<Picture 1>`, `<Picture 2>`, ...) with your plain English instruction; when canvas framing stays unchanged, `edit_aspect_ratio_system_prompt.md` emits `aspect_ratio: same as the picture` rather than guessing pixel ratios from resized vision previews.
4. **Fluff & benchmark-patch removal:** Strips hopscotch chalk-grid benchmark hacks, radar/pie-chart sum checks, Chinese/multilingual translation scaffolding, and non-image text-response rules.

---

## Recommended Local Thinking-Model Settings

Matches the release's [`generation_config.json`](../../../../vendor_guides/h3_image/README.md) sampling defaults without JSON grammar masking:

- **Thinking / reasoning:** Enabled natively via the model's chat template (do **not** force JSON `response_format`). The image DiT never sees `<think>...</think>`—only the post-thinking caption goes to `MiniMaxH3ImageConditioning`.
- **Sampling:** `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `repetition_penalty = 1.0`, `max_tokens = 4096` (covers both `<think>` reasoning and the final caption).
- **Quoting convention preserved from upstream:** T2I captions quote visible in-image text in `"double quotes"`; Edit captions quote visible source and target text in `'single quotes'`.

---

## Parsing Replies & ComfyUI Expander Node

- [`h3_image_nodes.py::MiniMaxH3ImagePromptExpander`](../../../../h3_image_nodes.py) (`MiniMax H3 Image Prompt Expander (heylook)`) loads these prompt files directly (`mode = auto | t2i + aspect_ratio | t2i | t2i_design + aspect_ratio | t2i_design | edit + aspect_ratio | edit | custom`), discovers served `chat` / `vision` models at runtime from `GET /v1/models`, downscales wired `<Picture 1>..<Picture 16>` reference images client-side (`<= 2048 px`), streams `POST /v1/messages`, and wires `(prompt, shape)` into `MiniMaxH3ImageConditioning.prompt` and `MiniMaxH3ImageSize.shape_from_expander` (example workflows: [`workflows/image/h3_image_t2i_heylook_api.json`](../../../../workflows/image/h3_image_t2i_heylook_api.json) and [`workflows/image/h3_image_edit_heylook_api.json`](../../../../workflows/image/h3_image_edit_heylook_api.json)).
- [`h3_image_nodes.py::parse_expander_reply`](../../../../h3_image_nodes.py) is the single shared offline parser for expander outputs (used alongside [`resolve_size`](../../../../h3_image_nodes.py)). It strips any unparsed `<think>...</think>` block or code fence and accepts plain captions, trailing `aspect_ratio: <same as the picture | W:H>` lines, and vendor `expanded_caption` JSON objects, returning `(clean_prompt, shape_or_none)`.
