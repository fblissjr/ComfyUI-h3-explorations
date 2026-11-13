# docs/wiki/projects/ — Cross-Cutting Project Dossiers

Owns structured, machine-and-human-readable project dossiers inside the LLM Wiki ([`../index.md`](../index.md)). Use this folder for portfolio audits, multi-track roadmaps, and end-to-end pipeline atlases rather than scattering ad-hoc reports at repo root or in untracked folders.

---

## 1. Dossier Schema (`<YYYY-MM-DD>_<slug>/`)

Every project folder under `docs/wiki/projects/` MUST follow one layout:

1. **`index.md` (`<= 45,000 B`)**: LLM + human entry router containing an executive summary, runnable `jq`/Python query recipes over `ledger.json`, tracked `file:line` anchors, and clickable links to sibling wiki hubs (`../../...` or `../...`) and `report.html`.
2. **`ledger.json`**: Canonical structured JSON (`json.tool`-valid) holding exact measured metrics, track inventories, ranked unfinished/forgotten items (`file:line` anchored), verified code-vs-prose discrepancies, session provenance, and `$H3_*`-normalized render catalogues. Agents querying structured facts MUST read `ledger.json` rather than scraping prose.
3. **`report.html` (optional)**: Self-contained offline interactive viewer generated deterministically from `ledger.json` (inline `<script id="atlas-ledger" type="application/json">`, zero external CDN `<link>`/`<script src>`/`@import` calls, dark/light toggle, search/filter controls).

---

## 2. Mandatory Invariants

1. **Code beats prose (`bench/check_doc_links.py`)**: Every cited `.py` path, symbol, line number, commit hash, workflow filename, and `node_id` MUST match tracked files on disk. When code and prose disagree, trust code and log the drift in `ledger.json` and [`../decisions.md`](../decisions.md). Do not cite gitignored `app/` or untracked `data/` files with `:line` suffixes in tracked Markdown (`bench/check_doc_links.py` validates every `path:line`).
2. **Path privacy (`bench/check_no_owner_paths.py`)**: Never write absolute home-directory paths, mount paths, owner names, or file-share names in `index.md`, `ledger.json`, `report.html`, or commit messages. Always use repo-relative paths or `$H3_OUTPUT_DIR` / `$H3_INPUT_DIR` / `$H3_MODEL_STORE` / `$H3_CAPTURE_ROOT`.
3. **Router size cap (`<= 45,000 B`)**: `bench/build_wiki_index.py` enforces `ROUTER_MAX_BYTES = 45,000` on router Markdown pages and verifies every file under `docs/wiki/` is reachable from [`../../../AGENTS.md`](../../../AGENTS.md), [`../../AGENTS.md`](../../AGENTS.md), [`../index.md`](../index.md), and [`index.md`](index.md).
4. **Stage new files and rebuild indexes**: Whenever adding a dossier file, run `git add -- <new_files>` so [`../../../bench/build_index.py`](../../../bench/build_index.py) indexes it, then regenerate [`../../../INDEX.md`](../../../INDEX.md) and validate with [`../../../bench/check_doc_links.py`](../../../bench/check_doc_links.py) and [`../../../bench/check_no_owner_paths.py`](../../../bench/check_no_owner_paths.py).
