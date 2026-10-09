bump: patch

### Changed

- **The references page records what moved upstream by 2026-10-09, with a
  cross-check of upstream's mask editing against the masked lane.** A fetch
  of every sister checkout, the ComfyUI checkout since its 2026-10-02 read
  and kitchen's upstream. No upstream moved a default this pack ships.
  `docs/wiki/references.md`, "What moved by 2026-10-09", holds it; the
  parts a reader acts on:
  - Core can run kitchen's Sol-Attn on a model file's or a LoRA's say-so,
    with no node in the graph. A graph with one of our attention nodes
    wired does not consult that choice; a graph on stock attention would
    follow it. No file `workflows/h3_config.py` or a generated graph names
    carries the key (a one-off header scan, not a check).
  - sglang's FastH3 entry is the 8-step V2 checkpoint on the trainer's
    contract, the values `h3_config.FASTH3_CONTRACT_*` already hold. Its
    default quality tier kept its name and changed meaning.
  - FastVideo measured a single 4090 on a private pruned checkpoint with an
    experimental kernel; the V2 contract did not move.
  - The hybrid attention in sglang and now vllm-omni is VDN-H3, a separate
    trained checkpoint recorded here on 2026-09-19.
  - "Masks and edited video upstream, beside the masked lane": where
    vllm-omni's mask editing, core and `MaskVidExperiments` agree with the
    lane, where they differ, and three things they do that it has not
    tried. `MaskVidExperiments` and `ComfyUI-NKD-Basic-Tools` get rows in
    the page's table.
- `docs/research/sglang_h3_pipeline.md`: three line citations that sglang's
  refactor moved are by symbol, which turns `bench/check_doc_links.py`
  green again. `docs/sol_upstream.md` gets a dated pointer section,
  `docs/wiki/masked_v2v.md` a dated note on what "clean" means in a kept
  token, and `docs/wiki/decisions.md` the log of what was corrected.
  No code, graph or default changes.
