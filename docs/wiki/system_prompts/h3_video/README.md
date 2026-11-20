# H3 Video Standalone System Prompts (`docs/wiki/system_prompts/h3_video/`)

Self-contained, zero-lore, single-purpose system prompts for expanding short scene ideas (plus attached images/videos/audio) into complete joint audio-video prompts across all five video modes—mirroring [`docs/wiki/system_prompts/h3_image/`](../h3_image/README.md).

Every `*_system_prompt.md` file in this folder is **pure standalone prompt text**—zero outer markdown fences, zero repo paths, zero model/engine lore, and zero frame-math instructions—ready to paste straight into any open-weights or frontier LLM/VLM, configure as reusable presets on a `heylookitsanllm` server (`/v1/presets`), or load automatically inside `auto_edit_prompt.py`, `h3_prompt_expander.py`, and `app/`.

---

## Which Prompt & Preset to Pick

| File | Suggested `heylook` Preset | Mode & Inputs | Output Contract |
| :--- | :--- | :--- | :--- |
| [`ref2va_system_prompt.md`](ref2va_system_prompt.md) | `h3-video-ref2va` | **Reference Video (`ref2va`):** `1–N` `<Picture N>`, `<Video N>`, and/or `<Audio N>` refs | 6 plain-text sections (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:` `350–500` words, `overall_soundscape:`, `non_diegetic_music:`) |
| [`t2va_system_prompt.md`](t2va_system_prompt.md) | `h3-video-t2va` | **Text-to-Video (`t2va`):** text idea only | 3 plain-text sections (`integrated_multimodal_description:` ~`220–360` words, `overall_soundscape:`, `non_diegetic_music:`) |
| [`i2va_system_prompt.md`](i2va_system_prompt.md) | `h3-video-i2va` | **First Frame (`i2va`):** `<Picture 1>` at `0.00s` + text | 3 plain-text sections developing forward from `<Picture 1>` (`stamp_part_one` inserts the `I2VA` Part One line automatically) |
| [`fl2va_system_prompt.md`](fl2va_system_prompt.md) | `h3-video-fl2va` | **First & Last Frame (`fl2va`):** bare `Picture 1` (start) + bare `Picture 2` (end) | 3 plain-text sections bridging `Picture 1` continuously into `Picture 2` (`stamp_part_one` inserts exact `[Shot N]` + `S.SS` Part One line automatically) |
| [`l2va_system_prompt.md`](l2va_system_prompt.md) | `h3-video-l2va` | **Last Frame (`l2va`):** `<Picture 1>` at final beat + text | 3 plain-text sections converging smoothly onto `<Picture 1>` (`stamp_part_one` inserts exact `[Shot N]` + `S.SS` Part One line automatically) |

---

## Why Native Single-Mode Prompts + Automatic Part One Stamping

1. **Zero prior model knowledge needed:** Open-weights LLMs (`gemma-4`, `Qwen3-VL`, etc.) without web search or repo access need one unambiguous structural skeleton per mode rather than one 6-mode switch statement—eliminating cross-mode leaks (`<Subject N>` in `fl2va`, `<Picture 1>` brackets in `fl2va`, `[Shot 1, Shot 2]` commas or `(Sx)` inside `retention_analysis:`, and short `< 350`-word `ref2va` descriptions).
2. **Native plain-text sections everywhere:** Every prompt emits exact H3 sections directly (`subject_definitions:` / `integrated_multimodal_description:`), so the exact same preset works identically in a local chat client, `app/`'s iPhone Expand button, and ComfyUI's `MiniMaxH3PromptExpander` (while `auto_edit_prompt.py::validate_fresh_h3_prompt` still accepts legacy `## section:` + `## axes` replies too).
3. **Deterministic frame math & validation in code:** Prompt expansion leaves `Part One` `S.SS` math to [`h3_prompt_contract.py::stamp_part_one`](../../../../h3_prompt_contract.py) and enforces every vendor guide constraint via [`h3_prompt_contract.py::validate_h3_prompt_mechanics`](../../../../h3_prompt_contract.py) (with 1-shot repair when needed). Verified by [`bench/check_prompt_docs_sync.py`](../../../../bench/check_prompt_docs_sync.py) and [`bench/check_auto_edit_prompt.py`](../../../../bench/check_auto_edit_prompt.py).
