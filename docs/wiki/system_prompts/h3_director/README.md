# H3 Director Standalone System Prompts (`docs/wiki/system_prompts/h3_director/`)

Self-contained, zero-lore system prompts powering interactive spatial direction, reticle grounding, and closed-loop multi-shot continuation presets synced to `heylook` via [`bench/sync_heylook_presets.py`](../../../../bench/sync_heylook_presets.py).

| File | `heylook` Preset Slug | Purpose | Output Contract |
| :--- | :--- | :--- | :--- |
| [`reticle_suggest_system_prompt.md`](reticle_suggest_system_prompt.md) | `h3-director-reticle-suggest` | Spatial reticle + `512px` ROI crop grounding | Plain Markdown (`scene_caption:`, `subject_label:`, `## suggestions`) |
| [`scene_suggest_system_prompt.md`](scene_suggest_system_prompt.md) | `h3-director-scene-suggest` | Scene-grounded world-sandbox action suggestions | Plain Markdown (`## suggestions` bullets) |
| [`shot_continuation_system_prompt.md`](shot_continuation_system_prompt.md) | `h3-director-shot-continue` | Closed-loop tail-frame VLM prompt grounding for multi-shot continuity | Plain-text 6-section or 3-section H3 video prompt |

Synced presets live on `/v1/presets` (synced manually on demand via `bench/sync_heylook_presets.py --apply`) and are inherited automatically downstream by [`director/server.py`](../../../../director/server.py) and `heylook` callers, with local `.md` fallbacks when unsynced. Verified by [`bench/check_prompt_docs_sync.py`](../../../../bench/check_prompt_docs_sync.py) and [`bench/check_sync_heylook_presets.py`](../../../../bench/check_sync_heylook_presets.py).
