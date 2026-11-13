# `docs/`: LLM-Wiki-First Documentation

Loaded when a session opens a file here; root [`../AGENTS.md`](../AGENTS.md) still governs.

- **[`wiki/index.md`](wiki/index.md) is the master router** for the LLM wiki (`docs/wiki/`), which is the single documentation surface of the repository (`core/`, `image/`, `attention/`, `prompting/`, `control_vectors/`, `system_prompts/`, `masked_v2v/`, `ops/`, `projects/`, `research/`, `decisions/`, `archive/`).
- **Prose is a cache of the code.** Where prose disagrees with `.py` modules on disk, verify against the code (`file:line` / symbol / constant), fix the page, and log the change in [`wiki/decisions.md`](wiki/decisions.md).
- **[`wiki/projects/`](wiki/projects/index.md)** holds structured multi-format project atlases (`index.md`, `ledger.json`, `report.html`) built for both human reading and machine/agent querying.
- **[`wiki/research/`](wiki/research/INDEX.md)** holds dated studies, campaign logs, and postmortems (what was observed on a date, not current rules).
- **[`wiki/archive/`](wiki/archive/INDEX.md)** holds frozen/superseded notes left unmodified for provenance.
- **[`wiki/ops/checks.md`](wiki/ops/checks.md)** defines the check standard and inventory; [`wiki/working_with_the_owner.md`](wiki/working_with_the_owner.md) governs style and path privacy (plain American English, repo-relative paths and `$H3_*` env vars only; never owner/mount paths or quotes of passing remarks).
