# 1. Control Vector Studio UI Guide

[← Back to Control Vectors Overview](../control_vectors.md)

The Control Vector CLI (`bench/control_vectors.py`) is the primary end-to-end interface for creating, inspecting, sculpting, and rebuilding control vectors (`from-prompt`, `from-toml`, `from-words`, `from-images`, `sculpt`, `rebuild`, `build-j-lens`, `inspect`).

An optional local browser workbench is also available via:
```bash
COMFY="$(cd ../.. && pwd)"
"$COMFY/.venv/bin/python" bench/control_vectors.py ui
```

---

## Core Screens & Corresponding CLI Subcommands

Every action in the browser UI maps 1-to-1 to a CLI subcommand in [`bench/control_vector/cli.py`](../../../bench/control_vector/cli.py):

| Studio Screen | Equivalent CLI Subcommand | Purpose |
|---|---|---|
| **1. Vector Library & Quick Actions** | `inspect`, `rebuild` | Browse built `.safetensors` in `models/control_vectors/` and `.toml` banks in `prompt_bank/control_vectors/`, inspect `split_half_cosine` and J-Lens readouts, or trigger `rebuild`. |
| **2. In-Place Prompt Contrast (`from-prompt`)** | `from-prompt --target-prompt <prompt> --words "pole_a -> pole_b"` | Paste your target scene prompt and substitute `pole_a -> pole_b` clauses in place (with optional `soundscape:` and `hold:` clauses). |
| **3. Multi-Scene Pair & LLM Studio (`from-toml` / `from-md`)** | `from-toml`, `from-md`, `generate-pairs` | Inspect or edit 10-pair `.toml` / `.md` sets with token-diff highlighting and `< 1 ms` CPU pair re-pooling (`/api/repool`). |
| **4. Reference Image Contrast (`from-images`)** | `from-images --image-a ... --image-b ...` | Contrast two reference images (`Image A` baseline vs. `Image B` target look) on a matched 32-px M-RoPE grid. |
| **5. Subspace Sculptor (`< 5 ms` CPU)** | `sculpt <vec1> [<vec2> ...] [--invert] [--hold-words ...]` | Invert (`-1.0x`), blend, or orthogonalize (`hold`) existing vectors in `float64` on CPU in `< 5 ms`. |
| **6. J-Lens & Prompt Receptor Meter (`from-words` / `inspect`)** | `from-words`, `inspect --prompt`, `build-j-lens` | Score per-token cosine alignment on a target prompt or synthesize `< 2 ms` CPU unigram vectors via `models/j_lens/j_lens_h3.safetensors`. |
