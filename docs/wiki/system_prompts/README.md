# H3 System Prompts (`docs/wiki/system_prompts/`)

Canonical system prompts for generating MiniMax H3 video+audio scene prompts and control-vector contrastive pair sets.

| System Prompt | Purpose | Verified By |
|---|---|---|
| [`h3_system_prompt.md`](h3_system_prompt.md) | General H3 multimodal scene-prompt writer (`CORE` + `MODE` blocks for `t2va`, `i2va`, `fl2va`, `l2va`, `ref2va`) derived from [`docs/prompting.md`](../../prompting.md) and [`vendor_guides/`](../../../vendor_guides/) | [`bench/check_prompt_docs_sync.py`](../../../bench/check_prompt_docs_sync.py) (`RULE_PINS` + `BASE_ALIGNMENT`), [`bench/grade_prompt_text.py`](../../../bench/grade_prompt_text.py) |
| [`h3_control_vector_system_prompt.md`](h3_control_vector_system_prompt.md) | Canonical control-vector authoring system prompt (`### Test Scene Prompt` + stratified `### Contrastive Pairs (1–10)` + optional `### Hold Pairs`) consumed by [`bench/control_vectors.py`](../../../bench/control_vectors.py) (`build` / `from-md`) | [`bench/check_prompt_docs_sync.py`](../../../bench/check_prompt_docs_sync.py) (`BASE_ALIGNMENT`), [`bench/check_control_vectors.py`](../../../bench/check_control_vectors.py) |

## Related Documentation
- **Prompting Manual (Single Source of Truth for Prompt Rules)**: [`docs/prompting.md`](../../prompting.md)
- **Control Vectors Wiki & End-to-End Guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
- **Historical `2026-11-05` Control-Vector System Prompt Research Record**: [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../../research/control_vectors/2026-11-05_control_vector_system_prompt.md)
