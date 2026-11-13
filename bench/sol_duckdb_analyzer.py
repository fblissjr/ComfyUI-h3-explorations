#!/usr/bin/env python3
"""DuckDB ingest and static dashboard for Sol probe and observer captures.

Ingests `sol_probe_*.jsonl` (H3_SOL_PROBE, `sol_block_probe.py`) and
`sol_observe_*.jsonl` (H3_SOL_OBSERVE, `sol_observe.py`) into a DuckDB file,
defines the views below, and writes a dashboard of tables computed from them.
The dashboard carries no prose findings: every number on it is a query result,
and the query is printed beside it.

**A run is a capture directory.** Every row carries `run`, the name of the
directory its file sits in, so runs are never pooled by accident. Rewritten
2026-10-02: the earlier version keyed rows by `prompt_id` alone (the probe's
`capture` label is null on older records), defaulted a missing schedule index
to step 0, and pooled every run into one block profile on the dashboard.

Views, each keyed by `run`:

  sol_runs              one row per run: settings, settings digest, seed,
                        prompt hash, cell/skip counts, probe granularity
  sol_probe_cells       one row per compared call (whole-call metrics)
  sol_probe_heads       per-head metrics, unnested
  sol_probe_segments    per-segment metrics, unnested
  sol_observe_calls     one row per observed DiT call, with route and density
  sol_joined_analysis   probe cells joined to their observer call
  sol_matched_steps     sigmas that every loaded run measured
  sol_matched_cells     cells on (sigma, block) pairs every loaded run measured
  sol_run_summary       per run: all-cell and matched-population error
  sol_run_cost          per run: attention cost in block-equivalents
  sol_block_profile     per run and block, on matched steps: relative and
                        absolute error
  sol_replicates        per block, runs that share a settings digest

**Why the matched population.** `dense_blocks` and `start_percent` remove
cells from what the probe measures, so the all-cell average and maximum fall
when the worst blocks or steps go dense even if no measured cell changed.
`sol_matched_cells` compares runs on the cells all of them measured.

**Why block-equivalents.** A dense block or an out-of-range step costs a full
attention call; a Sol call costs roughly its kernel density (sinks included).
Routed density over Sol calls alone ignores the dense calls a config adds.

**What the probe cannot see.** Its reference is the chained fallback on the
identical q/k/v (`sol_block_probe.py::_header`), so a cell is local error at
one call. Error carried into later steps or blocks shows up only as a change
in their inputs, never as a term in any cell.

    uv run python bench/sol_duckdb_analyzer.py --logs-dir data/sparse/captures
    uv run python bench/sol_duckdb_analyzer.py --query "SELECT * FROM sol_run_summary"

Needs the `duckdb` CLI (`DUCKDB_BIN`, PATH, or ~/.local/bin/duckdb).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_duckdb_binary() -> str:
    """Locate the duckdb binary without hardcoding user home paths."""
    env_bin = os.environ.get("DUCKDB_BIN")
    if env_bin and shutil.which(env_bin):
        return env_bin
    which_bin = shutil.which("duckdb")
    if which_bin:
        return which_bin
    user_local = Path.home() / ".local" / "bin" / "duckdb"
    if user_local.is_file():
        return str(user_local)
    return "duckdb"


DUCKDB_BIN = find_duckdb_binary()

# A probe cell is ~20-40 KB of JSON; DuckDB's default object limit is 16 MB,
# which holds, but render records grew once and this keeps headroom (reasoned).
MAX_OBJECT_BYTES = 64 * 1024 * 1024


def run_duckdb_sql(db_path: Path, sql: str) -> list[dict]:
    """Execute SQL against a DuckDB database and return the last statement's rows."""
    cmd = [DUCKDB_BIN, "-json", str(db_path), "-c", sql]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"DuckDB SQL error:\n{res.stderr}\nQuery (first 2000 chars):\n{sql[:2000]}")
    out = res.stdout.strip()
    if not out:
        return []
    # One JSON array per statement that returns rows; keep the last.
    last = out[out.rfind("\n[") + 1:] if "\n[" in out else out
    return json.loads(last)


def _reader(files: list[Path]) -> str:
    files_str = ", ".join(f"'{p.as_posix()}'" for p in files)
    return (f"read_json_auto([{files_str}], filename=true, union_by_name=true, "
            f"sample_size=-1, maximum_object_size={MAX_OBJECT_BYTES})")


# The run label: the directory holding the file.
RUN_EXPR = r"regexp_extract(filename, '([^/]+)/[^/]+$', 1)"


PROBE_SQL = """
CREATE OR REPLACE TABLE sol_probe_raw AS
SELECT {run} AS run, * EXCLUDE (filename) FROM {reader};

CREATE OR REPLACE VIEW sol_runs AS
WITH cells AS (
    SELECT run, prompt_id,
           any_value(config) AS settings_digest,
           any_value(to_json(settings)) AS settings,
           count(*) FILTER (WHERE kind = 'cell') AS cells,
           count(*) FILTER (WHERE kind = 'skip') AS skips,
           bool_or((to_json(metrics)->'per_segment'->0->>'worst_heads') IS NOT NULL) AS granular
    FROM sol_probe_raw WHERE kind IN ('cell', 'skip')
    GROUP BY run, prompt_id
), renders AS (
    SELECT run, prompt_id,
           to_json(rendered)->>'seed' AS seed,
           left(to_json(rendered)->>'prompt_sha256', 12) AS prompt_sha,
           to_json(rendered)->>'canvas' AS canvas,
           try_cast(to_json(rendered)->>'length' AS INTEGER) AS length
    FROM sol_probe_raw WHERE kind = 'render'
)
SELECT c.run, c.prompt_id, c.settings_digest,
       try_cast(c.settings->>'tau' AS DOUBLE) AS tau,
       c.settings->>'dense_blocks' AS dense_blocks,
       try_cast(c.settings->>'start_percent' AS DOUBLE) AS start_percent,
       c.settings->>'token_routing' AS token_routing,
       c.settings->>'sink_conditioning' AS sink_conditioning,
       c.settings->>'quantizer' AS quantizer,
       r.seed, r.prompt_sha, r.canvas, r.length,
       c.cells, c.skips, c.granular
FROM cells c LEFT JOIN renders r USING (run, prompt_id);

CREATE OR REPLACE VIEW sol_probe_cells AS
SELECT run, prompt_id, executing_node_id, seq, block,
       schedule.schedule_index AS step,
       sigma, round(sigma, 4) AS sigma_key,
       T AS tokens, cond_or_uncond, trajectory, config AS settings_digest,
       compare_status, compare_reason, returned_backend,
       metrics.whole.rel_l2 AS rel_l2, metrics.whole.cos AS cos,
       metrics.whole.diff_rms AS diff_rms, metrics.whole.ref_rms AS ref_rms,
       metrics.whole.numerator AS numerator, metrics.whole.denominator AS denominator,
       metrics.per_head AS per_head, metrics.per_segment AS per_segment
FROM sol_probe_raw
WHERE kind = 'cell' AND metrics.whole.rel_l2 IS NOT NULL;

CREATE OR REPLACE VIEW sol_probe_heads AS
WITH u AS (SELECT run, prompt_id, executing_node_id, step, sigma, sigma_key, block, unnest(per_head) AS h
           FROM sol_probe_cells)
SELECT run, prompt_id, executing_node_id, step, sigma, sigma_key, block,
       h.head AS head, h.rel_l2 AS head_rel_l2, h.cos AS head_cos,
       h.numerator AS head_numerator, h.denominator AS head_denominator,
       try_cast(to_json(h.rows)->>'p99' AS DOUBLE) AS row_p99,
       try_cast(to_json(h.rows)->>'p99_9' AS DOUBLE) AS row_p99_9,
       try_cast(to_json(h.rows)->>'max' AS DOUBLE) AS row_max
FROM u;

CREATE OR REPLACE VIEW sol_probe_segments AS
WITH u AS (SELECT run, prompt_id, executing_node_id, step, sigma, sigma_key, block, unnest(per_segment) AS s
           FROM sol_probe_cells)
SELECT run, prompt_id, executing_node_id, step, sigma, sigma_key, block,
       s.kind AS segment_kind, s.start AS seg_start, s."end" AS seg_end,
       s.rel_l2 AS seg_rel_l2, s.cos AS seg_cos,
       s.numerator AS seg_numerator, s.denominator AS seg_denominator,
       try_cast(to_json(s.rows)->>'p99' AS DOUBLE) AS row_p99,
       try_cast(to_json(s.rows)->>'p99_9' AS DOUBLE) AS row_p99_9,
       try_cast(to_json(s.rows)->>'max' AS DOUBLE) AS row_max,
       to_json(s)->'worst_heads' AS worst_heads
FROM u;

CREATE OR REPLACE VIEW sol_matched_steps AS
SELECT sigma_key FROM sol_probe_cells GROUP BY sigma_key
HAVING count(DISTINCT run) = (SELECT count(DISTINCT run) FROM sol_probe_cells);

CREATE OR REPLACE VIEW sol_matched_cells AS
WITH pairs AS (
    SELECT sigma_key, block FROM sol_probe_cells GROUP BY sigma_key, block
    HAVING count(DISTINCT run) = (SELECT count(DISTINCT run) FROM sol_probe_cells)
)
SELECT c.* FROM sol_probe_cells c JOIN pairs USING (sigma_key, block);

CREATE OR REPLACE VIEW sol_run_summary AS
WITH a AS (
    SELECT run, count(*) AS cells, avg(rel_l2) AS avg_rel_l2, max(rel_l2) AS max_rel_l2,
           min(cos) AS min_cos
    FROM sol_probe_cells GROUP BY run
), m AS (
    SELECT run, count(*) AS matched_cells, avg(rel_l2) AS matched_avg_rel_l2,
           max(rel_l2) AS matched_max_rel_l2, min(cos) AS matched_min_cos,
           sqrt(sum(numerator) / sum(denominator)) AS matched_pooled_rel_l2
    FROM sol_matched_cells GROUP BY run
)
SELECT r.run, r.settings_digest, r.tau, r.dense_blocks, r.start_percent, r.prompt_sha, r.seed,
       a.cells, a.avg_rel_l2, a.max_rel_l2, a.min_cos,
       m.matched_cells, m.matched_avg_rel_l2, m.matched_max_rel_l2, m.matched_min_cos,
       m.matched_pooled_rel_l2
FROM sol_runs r JOIN a USING (run) LEFT JOIN m USING (run);

CREATE OR REPLACE VIEW sol_block_profile AS
SELECT run, block, count(*) AS cells,
       avg(rel_l2) AS avg_rel_l2, max(rel_l2) AS max_rel_l2, min(cos) AS min_cos,
       avg(diff_rms) AS avg_diff_rms, avg(ref_rms) AS avg_ref_rms
FROM sol_probe_cells
WHERE sigma_key IN (SELECT sigma_key FROM sol_matched_steps)
GROUP BY run, block;

CREATE OR REPLACE VIEW sol_replicates AS
WITH g AS (SELECT settings_digest FROM sol_runs GROUP BY settings_digest HAVING count(*) > 1),
     b AS (SELECT p.run, r.settings_digest, r.prompt_sha, r.seed, p.block, p.avg_rel_l2
           FROM sol_block_profile p JOIN sol_runs r USING (run)
           WHERE r.settings_digest IN (SELECT settings_digest FROM g))
SELECT x.settings_digest, x.block, x.run AS run_a, y.run AS run_b,
       x.prompt_sha = y.prompt_sha AS same_prompt,
       x.avg_rel_l2 AS a_rel_l2, y.avg_rel_l2 AS b_rel_l2, y.avg_rel_l2 - x.avg_rel_l2 AS diff
FROM b x JOIN b y ON x.settings_digest = y.settings_digest AND x.block = y.block AND x.run < y.run;
"""

OBSERVE_SQL = """
CREATE OR REPLACE TABLE sol_observe_raw AS
SELECT {run} AS run, * EXCLUDE (filename) FROM {reader};

CREATE OR REPLACE VIEW sol_observe_calls AS
SELECT run, prompt_id, executing_node_id, seq, block,
       schedule.schedule_index AS step,
       sigma, round(sigma, 4) AS sigma_key, cond_or_uncond, route, T AS tokens,
       routed_density.mean AS mean_routed_density,
       routed_density.p50 AS p50_routed_density,
       routed_density.max AS max_routed_density,
       kernel_density.mean AS mean_kernel_density,
       try_cast(to_json(ordering_effect_density)->>'overall' AS DOUBLE) AS ordering_effect_density,
       peak_alloc_bytes / (1024.0 * 1024.0) AS peak_vram_mb
FROM sol_observe_raw
WHERE kind = 'call' AND scope = 'dit';

-- A Sol call costs its kernel density; any other route on a DiT call ran the
-- dense fallback and costs 1. Block-equivalents, not seconds: timings from an
-- armed render are void (sol_block_probe.py::_header).
CREATE OR REPLACE VIEW sol_run_cost AS
SELECT run, count(*) AS dit_calls,
       count(*) FILTER (WHERE route = 'sol') AS sol_calls,
       count(*) FILTER (WHERE route <> 'sol') AS dense_calls,
       sum(CASE WHEN route = 'sol' THEN mean_kernel_density ELSE 1.0 END) AS block_equivalents,
       sum(CASE WHEN route = 'sol' THEN mean_kernel_density ELSE 1.0 END) / count(*) AS fraction_of_dense,
       avg(mean_kernel_density) FILTER (WHERE route = 'sol') AS sol_kernel_density,
       avg(mean_routed_density) FILTER (WHERE route = 'sol') AS sol_routed_density
FROM sol_observe_calls GROUP BY run;
"""

JOIN_SQL = """
CREATE OR REPLACE VIEW sol_joined_analysis AS
SELECT p.run, p.prompt_id, p.executing_node_id, p.seq, p.step, p.sigma, p.block,
       p.rel_l2, p.cos, p.diff_rms, p.ref_rms,
       o.route, o.mean_routed_density, o.mean_kernel_density, o.ordering_effect_density, o.peak_vram_mb
FROM sol_probe_cells p
LEFT JOIN sol_observe_calls o
  ON p.run = o.run AND p.prompt_id = o.prompt_id AND p.executing_node_id = o.executing_node_id
 AND p.block = o.block AND p.sigma_key = o.sigma_key AND p.cond_or_uncond = o.cond_or_uncond;
"""


def setup_duckdb_tables(db_path: Path, probe_files: list[Path], observe_files: list[Path]):
    """Load the JSONL files and (re)define every view."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = []
    if probe_files:
        sql.append(PROBE_SQL.format(run=RUN_EXPR, reader=_reader(probe_files)))
    if observe_files:
        sql.append(OBSERVE_SQL.format(run=RUN_EXPR, reader=_reader(observe_files)))
    if probe_files and observe_files:
        sql.append(JOIN_SQL)
    if sql:
        run_duckdb_sql(db_path, "\n".join(sql))


# --- dashboard -------------------------------------------------------------

def _fmt(v, kind: str) -> str:
    if v is None:
        return "&ndash;"
    if kind == "pct":
        return f"{float(v) * 100:.2f}"
    if kind == "f4":
        return f"{float(v):.4f}"
    if kind == "f1":
        return f"{float(v):.1f}"
    return html.escape(str(v))


def _table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    if not rows:
        return '<p class="muted">No rows.</p>'
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r.get(key), kind)}</td>" for key, _, kind in cols) + "</tr>"
        for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _section(title: str, note: str, sql: str, rows: list[dict], cols) -> str:
    return (f'<section><h2>{html.escape(title)}</h2><p class="muted">{html.escape(note)}</p>'
            f'<pre>{html.escape(sql.strip())}</pre>{_table(rows, cols)}</section>')


def _block_grid(db_path: Path) -> str:
    """Blocks down, runs across: mean relative L2 (%) on matched steps."""
    runs = [r["run"] for r in run_duckdb_sql(db_path, "SELECT run FROM sol_runs ORDER BY run")]
    rows = run_duckdb_sql(db_path, "SELECT run, block, avg_rel_l2 FROM sol_block_profile")
    grid: dict[int, dict[str, float]] = {}
    for r in rows:
        grid.setdefault(r["block"], {})[r["run"]] = r["avg_rel_l2"]
    if not grid:
        return ""
    vmax = max(v for b in grid.values() for v in b.values())
    head = "<th>block</th>" + "".join(f"<th class='rot'>{html.escape(r)}</th>" for r in runs)
    body = []
    for block in sorted(grid):
        cells = []
        for run in runs:
            v = grid[block].get(run)
            if v is None:
                cells.append('<td class="dense">dense</td>')
            else:
                a = 0.08 + 0.72 * (v / vmax)
                cells.append(f'<td style="background: rgb(var(--heat) / {a:.2f})">{v * 100:.1f}</td>')
        body.append(f"<tr><td>{block}</td>{''.join(cells)}</tr>")
    sql = "SELECT run, block, avg_rel_l2 FROM sol_block_profile  -- matched steps only"
    return (f"<section><h2>Per-block relative L2 (%), matched steps</h2>"
            f"<p class='muted'>Steps every run measured. A cell marked dense was not measured in that run.</p>"
            f"<pre>{html.escape(sql)}</pre><div class='scroll'><table class='grid'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></section>")


def generate_html_dashboard(db_path: Path, out_html: Path):
    """Render the dashboard from the views; no prose findings."""
    views = {r["name"] for r in run_duckdb_sql(db_path, "SHOW TABLES;")}
    parts = []
    if "sol_runs" in views:
        q = """SELECT run, settings_digest, tau, dense_blocks, start_percent, token_routing,
       sink_conditioning, prompt_sha, seed, cells, skips, granular FROM sol_runs ORDER BY run"""
        parts.append(_section("Runs", "One row per capture directory. A shared settings digest means identical Sol settings.",
                              q, run_duckdb_sql(db_path, q), [
            ("run", "run", "s"), ("settings_digest", "digest", "s"), ("tau", "tau", "s"),
            ("dense_blocks", "dense_blocks", "s"), ("start_percent", "start", "s"),
            ("token_routing", "routing", "s"), ("sink_conditioning", "sink", "s"),
            ("prompt_sha", "prompt", "s"), ("seed", "seed", "s"), ("cells", "cells", "s"),
            ("skips", "skips", "s"), ("granular", "granular", "s")]))
    if "sol_run_summary" in views:
        q = "SELECT * FROM sol_run_summary ORDER BY run"
        parts.append(_section(
            "Error per run: all cells against the matched population",
            "All-cell figures move when dense_blocks or start_percent remove cells. Matched figures use only (sigma, block) pairs every run measured. Relative L2 in %.",
            q, run_duckdb_sql(db_path, q), [
                ("run", "run", "s"), ("cells", "cells", "s"), ("avg_rel_l2", "avg", "pct"),
                ("max_rel_l2", "max", "pct"), ("min_cos", "min cos", "f4"),
                ("matched_cells", "matched cells", "s"), ("matched_avg_rel_l2", "matched avg", "pct"),
                ("matched_max_rel_l2", "matched max", "pct"), ("matched_pooled_rel_l2", "matched pooled", "pct"),
                ("matched_min_cos", "matched min cos", "f4")]))
    if "sol_run_cost" in views:
        q = "SELECT * FROM sol_run_cost ORDER BY run"
        parts.append(_section(
            "Attention cost per run",
            "Block-equivalents: a Sol call counts its kernel density, any dense call counts 1. Fraction of dense is that sum over the DiT call count.",
            q, run_duckdb_sql(db_path, q), [
                ("run", "run", "s"), ("dit_calls", "DiT calls", "s"), ("sol_calls", "Sol", "s"),
                ("dense_calls", "dense", "s"), ("block_equivalents", "block-equiv", "f1"),
                ("fraction_of_dense", "of dense (%)", "pct"), ("sol_kernel_density", "Sol kernel (%)", "pct"),
                ("sol_routed_density", "Sol routed (%)", "pct")]))
    if "sol_replicates" in views:
        q = """SELECT settings_digest, run_a, run_b, same_prompt, count(*) AS blocks,
       avg(abs(diff)) AS mean_abs_diff, max(abs(diff)) AS max_abs_diff
FROM sol_replicates GROUP BY ALL ORDER BY settings_digest"""
        parts.append(_section(
            "Replicates: runs with identical Sol settings",
            "Per-block mean relative L2 on matched steps, run B minus run A. This is the noise a config difference has to clear.",
            q, run_duckdb_sql(db_path, q), [
                ("settings_digest", "digest", "s"), ("run_a", "run A", "s"), ("run_b", "run B", "s"),
                ("same_prompt", "same prompt", "s"), ("blocks", "blocks", "s"),
                ("mean_abs_diff", "mean |diff| (pts)", "pct"), ("max_abs_diff", "max |diff| (pts)", "pct")]))
    if "sol_block_profile" in views:
        parts.append(_block_grid(db_path))
    if "sol_probe_segments" in views:
        q = """SELECT run, segment_kind, sqrt(sum(seg_numerator) / sum(seg_denominator)) AS pooled_rel_l2,
       min(seg_cos) AS min_cos FROM sol_probe_segments
WHERE sigma_key IN (SELECT sigma_key FROM sol_matched_steps) GROUP BY ALL ORDER BY run, segment_kind"""
        parts.append(_section("Per-segment error, matched steps",
                              "Pooled relative L2 (sum of squared error over sum of squared reference) per packed segment.",
                              q, run_duckdb_sql(db_path, q), [
            ("run", "run", "s"), ("segment_kind", "segment", "s"),
            ("pooled_rel_l2", "pooled rel L2 (%)", "pct"), ("min_cos", "min cos", "f4")]))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sol Capture Tables</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#66655f; --line:#dedcd5; --code:#f1f0ec; --heat:214 84 52; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --fg:#ecebe6; --muted:#a3a29b; --line:#33322e; --code:#1d1d1b; --heat:236 112 72; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 4px; }}
.muted {{ color:var(--muted); margin:0 0 8px; }}
pre {{ background:var(--code); padding:10px 12px; border-radius:6px; overflow-x:auto; font-size:12.5px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 10px; text-align:right; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; }}
table.grid td {{ padding:3px 8px; }} td.dense {{ color:var(--muted); font-size:11px; }}
</style>
</head>
<body><main>
<h1>Sol capture tables</h1>
<p class="muted">Generated by <code>bench/sol_duckdb_analyzer.py</code> from <code>{html.escape(db_path.name)}</code>. Every table is the result of the query above it.</p>
{''.join(parts)}
</main></body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=str, nargs="+", default=["data/sparse/captures"],
                        help="Directories searched recursively for sol_probe_*.jsonl and sol_observe_*.jsonl")
    parser.add_argument("--exclude-dirs", type=str, nargs="*", default=[],
                        help="Skip files whose path contains any of these substrings")
    parser.add_argument("--db", type=str, default="data/sparse/sol_analysis.duckdb", help="DuckDB file")
    parser.add_argument("--out", type=str, default="docs/wiki/research/sparse/sol_duckdb_dashboard.html",
                        help="Dashboard HTML output")
    parser.add_argument("--query", type=str, help="Run one SQL query against the existing database and exit")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.query:
        print(json.dumps(run_duckdb_sql(db_path, args.query), indent=2))
        return 0

    probe_files, observe_files = [], []
    for ld in args.logs_dir:
        for f in sorted(Path(ld).glob("**/sol_probe_*.jsonl")):
            if not any(ex in f.as_posix() for ex in args.exclude_dirs):
                probe_files.append(f)
        for f in sorted(Path(ld).glob("**/sol_observe_*.jsonl")):
            if not any(ex in f.as_posix() for ex in args.exclude_dirs):
                observe_files.append(f)

    print(f"[sol_duckdb] {len(probe_files)} probe and {len(observe_files)} observe files -> {args.db}")
    setup_duckdb_tables(db_path, probe_files, observe_files)
    generate_html_dashboard(db_path, Path(args.out))
    print(f"[sol_duckdb] dashboard: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
