# Documentation (`docs/`)

All active repository documentation lives in the **LLM Wiki** under [`wiki/index.md`](wiki/index.md). Root [`../AGENTS.md`](../AGENTS.md) carries the runtime rules, H3 clock/frame invariants, and 65-node pack replacement map; `docs/wiki/` organizes deep technical ownership, operations, project atlases, dated research, decisions, and retired records into modular subtrees (`<= 45 KB` per router page).

## Directory Map

| Path | Role | Start Here |
| :--- | :--- | :--- |
| `docs/wiki/` | Front-and-center LLM Wiki router & top-level mode guides | [`wiki/index.md`](wiki/index.md), [`wiki/h3_uses.md`](wiki/h3_uses.md), [`wiki/stages.md`](wiki/stages.md) |
| `docs/wiki/core/` | Core H3 temporal/VAE clocks, conditioning, references, distills, audio freeze | [`wiki/core/h3_frame_rules.md`](wiki/core/h3_frame_rules.md) |
| `docs/wiki/image/` | Single-frame image generation & editing (`MiniMax-H3-Image`, `P-H3-Image`, Pika `t=1`) | [`wiki/image/h3_image.md`](wiki/image/h3_image.md), [`wiki/image/p_h3_image.md`](wiki/image/p_h3_image.md) |
| `docs/wiki/control_vectors/` | User/agent guides for post-refiner steering, J-Lens readouts, Studio UI, CLI | [`wiki/control_vectors.md`](wiki/control_vectors.md), [`wiki/edit_guidance.md`](wiki/edit_guidance.md) |
| `docs/wiki/attention/` | `MiniMaxH3Sol`, `comfy-kitchen` kernels, quantization policy, Block 49 | [`wiki/attention/SOLATTN.md`](wiki/attention/SOLATTN.md), [`wiki/attention/comfy_kitchen.md`](wiki/attention/comfy_kitchen.md) |
| `docs/wiki/prompting/` | Prompt rules, `grade_prompt_text.py` checks, sharded manual, prompt bank | [`wiki/prompting/prompting.md`](wiki/prompting/prompting.md) |
| `docs/wiki/system_prompts/` | Runtime system prompt templates read by ComfyUI nodes & bench compilers | [`wiki/system_prompts/README.md`](wiki/system_prompts/README.md) |
| `docs/wiki/masked_v2v/` | Masked V2V perception/inpaint pipeline & `lane_db` SQLite architecture (`'fun'` parked) | [`wiki/masked_v2v.md`](wiki/masked_v2v.md), [`wiki/masked_v2v/AGENTS.md`](wiki/masked_v2v/AGENTS.md) |
| `docs/wiki/ops/` | Running ComfyUI, check suites, evidence/retraction standards, hardware, roadmap | [`wiki/ops/comfy_notes.md`](wiki/ops/comfy_notes.md), [`wiki/ops/checks.md`](wiki/ops/checks.md) |
| `docs/wiki/projects/` | Cross-cutting project dossiers (`index.md` + machine `ledger.json` + offline `report.html`) | [`wiki/projects/index.md`](wiki/projects/index.md), [`wiki/projects/2026-11-13_repo_audit_and_atlas/index.md`](wiki/projects/2026-11-13_repo_audit_and_atlas/index.md) |
| `docs/wiki/research/` | Dated experiments, studies, postmortems, sparse/PDD/control-vector/masking memos | [`wiki/research/INDEX.md`](wiki/research/INDEX.md) |
| `docs/wiki/decisions/` | Dated decisions/reversals sharded into `< 42 KB` files | [`wiki/decisions.md`](wiki/decisions.md) |
| `docs/wiki/archive/` | Frozen superseded docs preserved for link/provenance continuity | [`wiki/archive/INDEX.md`](wiki/archive/INDEX.md) |
