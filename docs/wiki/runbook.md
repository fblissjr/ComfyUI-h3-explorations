# The card and ledger runbook: router to the single home of each procedure

Last updated: 2026-10-26 (replaced the 2026-10-17 / 2026-10-21 duplicate tables and historical session-improvisation notes with direct pointers to the single authoritative home of each procedure; root [`AGENTS.md`](../../AGENTS.md), "Where state lives: one home per kind of fact, not two").

| Topic | Authoritative Home |
|---|---|
| **Today's order, what not to assume, and close-of-day state** | [`docs/wiki/handover.md`](handover.md) and [`docs/wiki/decisions.md`](decisions.md) |
| **Starting, checking, and stopping ComfyUI (`:8188`) and the review server (`:8765`), environment keys (`H3_*`), and cache states** | [`docs/comfy_notes.md`](../comfy_notes.md) and [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) Stage 0 |
| **Building, auditing, queueing, and landing a masked V2V load (`preview` → `strangers` → `job` → `reference` → `text` → `first` → `plan` → `on-plan` → `table` → `facts_pass.py` → `inspect_mask_steps.py` → `flags_before_render.py` → `what_h3_got.py` → `render` → `land` → `page`)** | [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) Stages 1–5 and [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §1–§11 |
| **What a change to code, a reference, a prompt, or a Pass-1 cut forces (in trace order)** | [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §8 ("What a change forces: the chain in trace order") |
| **Settled recipes (`margin`, `zoom_up_to`, `parts`, `reference`, `head_marks`) and `cast.jsonl` Group 1 / Group 2 / Group 3 rules** | `data/<video>/plan.json` (enforced by [`bench/shot_to_load.py`](../../bench/shot_to_load.py) and [`bench/flags_before_render.py`](../../bench/flags_before_render.py)), [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §3 & §8, and [`docs/wiki/decisions.md`](decisions.md) (`2026-10-26`) |
| **Laying a kept piece (`laid.jsonl`) and rebuilding the delivery cut, gap check, stack, and review pages (`TO_REVIEW.html`, `PROVENANCE.html`)** | [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) Stage 5 ([`bench/laid.py`](../../bench/laid.py) and [`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py)) |
| **End-of-day closeout (handover, `next_steps.md`, daily changelog fragment, stopping `:8188`)** | Root [`AGENTS.md`](../../AGENTS.md) ("Where state lives" & "Operative rules") and [`changelog.d/README.md`](../../changelog.d/README.md) |
