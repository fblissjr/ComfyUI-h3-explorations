# H3 Query Router Standalone System Prompts (`docs/wiki/system_prompts/h3_query_router/`)

Self-contained, zero-lore, zero-visual-edit prompt-diff system prompts for generating paired `audio_conditioning` prompts consumed by [`MiniMaxH3QueryRouter`](../../../../steering/query_router.py) in one-pass `1x` workflows ([`docs/wiki/query_routing_recipes.md`](../../query_routing_recipes.md)).

Every `*_system_prompt.md` file in this folder is **pure standalone prompt text**—zero outer markdown fences, zero repo paths, and zero model/engine lore—ready to paste into any local LLM/VLM or sync to a `heylookitsanllm` `/v1/presets` server via [`bench/sync_heylook_presets.py`](../../../../bench/sync_heylook_presets.py).

---

## Which Prompt & Preset to Pick

| File | `heylook` Preset Slug | What Changes | Verbatim Lock |
| :--- | :--- | :--- | :--- |
| [`dub_dialogue_system_prompt.md`](dub_dialogue_system_prompt.md) | `h3-query-router-dub` | **Strictly `<d>[Language] ...</d>` spans** (full-clip dubbing or per-speaker `(S1)`/`(S2)` code-switching) | Every visual section, `[Shot N]` header, speaker ID, tag (`<|lyrics_start|>`, `<|caption_start|>`, `<|cutoff|>`), `overall_soundscape:`, and `non_diegetic_music:` stays 100% verbatim |
| [`swap_soundscape_system_prompt.md`](swap_soundscape_system_prompt.md) | `h3-query-router-soundscape` | **Strictly `overall_soundscape:`** (and `non_diegetic_music:` only when a music/score change is requested) | Every visual section (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `integrated_multimodal_description:`) and `<d>` dialogue stays 100% verbatim |

Verified by [`bench/check_prompt_docs_sync.py`](../../../../bench/check_prompt_docs_sync.py), [`bench/check_heylook_prompts.py`](../../../../bench/check_heylook_prompts.py), and [`bench/check_sync_heylook_presets.py`](../../../../bench/check_sync_heylook_presets.py).
