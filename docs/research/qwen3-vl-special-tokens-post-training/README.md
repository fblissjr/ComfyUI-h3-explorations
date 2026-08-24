# Qwen3-VL special-token and AWQ v2 research

This directory separates accepted evidence from working conversation and
rejected historical artifacts.

| Location | Meaning |
|---|---|
| [`canonical/`](canonical/README.md) | Authoritative facts, owner decisions, contracts, and accepted plans. |
| [`brainstorming/`](brainstorming/README.md) | Current agent handoffs only. Nothing here overrides canonical. |
| [`archive/`](archive/README.md) | Superseded communication and compact rejected evidence retained for audit. |
| [`h3_special_tokens_post_training.md`](h3_special_tokens_post_training.md) | Original research proposal and hypothesis framing; canonical wins on conflict. |

Executable probes and builders live in [`bench/`](../../../bench/), and their
portable outputs live in [`bench/results/`](../../../bench/results/). Do not add
active code, generated datasets, or launchable preflight directories under this
documentation tree.

The deployed AWQ checkpoint and symlink are outside this directory and are not
changed by reorganizing these records.

At the 2026-08-24 stopping point, resume with the authoritative
[`Gate 2 readiness record`](canonical/2026-08-24_gate2_readiness.md) and
[`active plan`](canonical/active_plan.md). Gate 1/1B are accepted; the corrected
Gate 2A measurement and non-exporting Gate 2B modifier pilot remain before any
v2 quantization launch package.
