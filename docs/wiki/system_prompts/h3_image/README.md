# MiniMax-H3-Image Standalone System Prompts (`docs/wiki/system_prompts/h3_image/`)

Self-contained, English-only system prompts for expanding short requests into finished **MiniMax-H3-Image** captions (`workflows/image/h3_image_{t2i,edit}_api.json` and `h3_image_{t2i,edit}_heylook_api.json`, documented in [`docs/wiki/image/h3_image.md`](../../image/h3_image.md)).

Every prompt file in this folder is **pure standalone prompt text**—no outer markdown fences, no repo paths, no internal model lore, and no external cloud API assumptions—ready to paste as the system prompt in a local thinking LLM/VLM (such as `heylookitsanllm`).

> **Model boundary:** These prompts target MiniMax's official 6-step **MiniMax-H3-Image** checkpoint. For PrunaAI's 8-step **P-H3-Image** model (`Edit <Picture 1>. ... Keep ... unchanged.`), use [`../h3_p_h3_image_system_prompt.md`](../h3_p_h3_image_system_prompt.md). The unmodified, hash-pinned vendor middleware prompts from the release live in [`vendor_guides/h3_image/prompt/`](../../../../vendor_guides/h3_image/prompt/).

---

## Which Prompt to Pick

| File | Mode & Scope | Output Format | When to Use |
| :--- | :--- | :--- | :--- |
| [`t2i_system_prompt.md`](t2i_system_prompt.md) | **General T2I** — length-proportional photography/scene briefs (45–90 words for focused requests, scaling organically with real scene complexity; double-quoted signage only when requested) | Single plain-text paragraph | Manual copy-paste into local LLM apps or `MiniMaxH3ImageConditioning.prompt` when you pick the ratio yourself on `MiniMax H3 Image Size` |
| [`t2i_aspect_ratio_system_prompt.md`](t2i_aspect_ratio_system_prompt.md) | Same as `t2i_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <10 presets>` | Local scripts or automated `heylookitsanllm` runs that parse both prompt and canvas shape |
| [`t2i_design_system_prompt.md`](t2i_design_system_prompt.md) | **Design & Typography T2I** — posters, covers, typography/logic boards, infographics, charts, UI/dashboards, character/product sheets, multi-panel diagrams | Single layout-mapped plain-text paragraph | Manual copy-paste for layout-heavy, multi-slot, or data-rich graphics without polluting general scene prompts |
| [`t2i_design_aspect_ratio_system_prompt.md`](t2i_design_aspect_ratio_system_prompt.md) | Same as `t2i_design_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <10 presets>` | Automated runs for posters, infographics, UI screens, and design sheets |
| [`edit_system_prompt.md`](edit_system_prompt.md) | **Image Editing** (`1–16` `<Picture N>` references) — natural 3-beat prose paragraph (`Preserve -> Change -> Finished Image`, referencing attached photos sequentially as `<Picture 1>`, `<Picture 2>`, ... with `"double-quoted"` visible text) | Single plain-text paragraph | Manual copy-paste into vision-capable LLM apps (works both with unlabeled chat attachments in order and ComfyUI's `<Picture N>:`-labeled expander node) with `MiniMax H3 Image Size` set to `same as the picture` |
| [`edit_aspect_ratio_system_prompt.md`](edit_aspect_ratio_system_prompt.md) | Same as `edit_system_prompt.md` | Single paragraph + trailing `aspect_ratio: <same as the picture \| W:H>` | Automated VLM edit runs where crops, outpaints, or explicit target shapes may change the canvas ratio |

---

## Why These Differ from `vendor_guides/h3_image/prompt/`

MiniMax's raw `vendor_guides/h3_image/prompt/` middleware files (`15.8 KB` for T2I, `38.5 KB` for Edit) were SFT data-compiler / hosted `/v1/h3_image_context_ir` wrapper scaffolding with XGrammar JSON schema decoding—and the upstream Edit prompt-expansion guide itself is unfinished (`prompt only`, producing verbose `<Subject 1..N> is fully preserved.` inventories and forced `'single quotes'`). Meanwhile, open-source inference stacks (`diffusers`, `vllm-omni`, `sglang`, `LightX2V`) and MiniMax's own official edit examples ([`vendor_guides/h3_image/examples/prompt.edit.txt`](../../../../vendor_guides/h3_image/examples/prompt.edit.txt)) feed natural `"double-quoted"` captions directly to the Qwen3-VL text encoder. This suite fixes four local-LLM bottlenecks:

1. **Unconstrained plain-text generation (no JSON diversity collapse):** Instead of forcing `{ "expanded_caption": "..." }` with `\"` quote-escaping across 400 words under constrained JSON decoding, the model writes natural plain text first and (in the ratio variants) appends one clean `aspect_ratio: ...` line at the end.
2. **Proportional length & Scene vs. Design separation:** General `t2i_*` acts like an experienced photographer/art director—roughly `45–90` words for focused requests without inventing unrequested props or "8k/masterpiece" buzzwords—while keeping layout-heavy posters, UIs, and charts isolated in `t2i_design_*`.
3. **Natural sequential `<Picture N>` edits (zero synthetic scaffolding):** Attach images in order (`<Picture 1>`, `<Picture 2>`, ...) in any raw vision chat UI or via ComfyUI's expander node and describe only what stays unchanged, what changes or transfers, and how the finished image reads—avoiding caller-generated `Reference Media: [aspect_ratio W:H]` pre-parts, synthetic `<Subject N>` tags, and `'single quotes'` translation noise. When canvas framing stays unchanged, `edit_aspect_ratio_system_prompt.md` outputs `aspect_ratio: same as the picture` rather than estimating ratios from resized previews.
4. **Fluff & benchmark-patch removal:** Strips hopscotch chalk-grid benchmark hacks, radar/pie-chart sum checks, Chinese/multilingual translation scaffolding, and non-image text-response rules.

---

## Recommended Local Thinking-Model Settings

Matches the release's [`generation_config.json`](../../../../vendor_guides/h3_image/README.md) sampling defaults without JSON grammar masking:

- **Thinking / reasoning:** Works cleanly with both thinking models (`Qwen3.8-27B`) and non-thinking VLMs (`Qwen3-VL-32B-Instruct`). Leave JSON `response_format` disabled; the image DiT never sees `<think>...</think>`—only the post-thinking caption goes to `MiniMaxH3ImageConditioning`.
- **Sampling:** `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `repetition_penalty = 1.0`, `max_tokens = 4096` (covers both `<think>` reasoning and the final caption).
- **Quoting convention:** All prompts in this suite (`t2i_*`, `t2i_design_*`, and `edit_*`) quote visible in-image text in standard `"double quotes"`.

---

## Parsing Replies & ComfyUI Expander Node

- [`h3_image_nodes.py::MiniMaxH3ImagePromptExpander`](../../../../h3_image_nodes.py) (`MiniMax H3 Image Prompt Expander (heylook)`) loads these prompt files directly (`mode = auto | t2i + aspect_ratio | t2i | t2i_design + aspect_ratio | t2i_design | edit + aspect_ratio | edit | custom`), discovers served `chat` / `vision` models at runtime from `GET /v1/models`, downscales wired `<Picture 1>..<Picture 16>` reference images client-side (`<= 2048 px`), streams `POST /v1/messages`, and wires `(prompt, shape)` into `MiniMaxH3ImageConditioning.prompt` and `MiniMaxH3ImageSize.shape_from_expander` (example workflows: [`workflows/image/h3_image_t2i_heylook_api.json`](../../../../workflows/image/h3_image_t2i_heylook_api.json) and [`workflows/image/h3_image_edit_heylook_api.json`](../../../../workflows/image/h3_image_edit_heylook_api.json)).
- [`h3_image_nodes.py::parse_expander_reply`](../../../../h3_image_nodes.py) is the single shared offline parser for expander outputs (used alongside [`resolve_size`](../../../../h3_image_nodes.py)). It strips any unparsed `<think>...</think>` block or code fence and accepts plain captions, trailing `aspect_ratio: <same as the picture | W:H>` lines, and vendor `expanded_caption` JSON objects, returning `(clean_prompt, shape_or_none)`.
