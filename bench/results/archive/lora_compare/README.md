# LoRA and DiT file comparisons

A finished byte-level survey, measured 2026-08-25: the H3 Turbo LoRAs compared
across the lightx2v and DBM publishers and across key conventions, plus the
fl2va DiT variants compared pruned against rank-8 against int8 against bf16.

Each record is a header read plus range-fetched tensors -- key sets, alphas,
module counts, dtypes. No render, no GPU, nothing cached.

**The lane is finished, not broken.** `bench/compare_lora_files.py` and
`bench/build_hybrid.py` both still run; if a publisher ships a new artifact,
re-run the tool rather than reading these. `docs/evidence.md` carries what the
survey concluded.
