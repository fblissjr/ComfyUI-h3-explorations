#!/usr/bin/env python3
"""Read a Sol route record and say what it holds, per render, step and block.

A reader, not a check: it asserts nothing about the numbers and exits 0 on
any well-formed file. What it refuses to do is summarise past an `error` row
without being told to, because a file that stopped is not a complete result
for the rows it has (`pdd_observe.py`'s lesson).

    python bench/sol_observe_report.py <dir-or-jsonl> [--join http://127.0.0.1:8188]
        [--past-errors] [--segments] [--block-table]

`--join` asks the named server's `/history/<prompt_id>` for every prompt id
in the file and reports which are known to it -- the live acceptance the
uncontrolled row in `docs/checks.md` is waiting on. A row whose prompt id the
server does not know was either rendered by another server process or
recorded under no executing context; the report says which by the row's
`identity_source`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("sol_observe_*.jsonl"))
        if not files:
            sys.exit(f"no sol_observe_*.jsonl under {path}")
        if len(files) > 1:
            print(f"note: {len(files)} record files in {path}; reading the newest, {files[-1].name}")
        path = files[-1]
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"note: line {i} is not JSON (an interrupted append?); stopping there")
                break
    return rows


def _fmt(x, width=7):
    return f"{x:{width}.4f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("path", help="record directory or a sol_observe_*.jsonl")
    ap.add_argument("--join", metavar="URL", help="ComfyUI server to join prompt ids against")
    ap.add_argument("--past-errors", action="store_true", help="summarise even after an error row")
    ap.add_argument("--segments", action="store_true", help="per-segment table per (config, step)")
    ap.add_argument("--block-table", action="store_true", help="block x step adaptive density")
    ap.add_argument("--json", metavar="PATH", help="also write a tracked summary: header, renders, configs, "
                    "per-forward routes, whole-call densities and the block x step ordering-effect table")
    ap.add_argument("--prompt", metavar="ID", help="summarise this prompt id only")
    args = ap.parse_args()

    rows = _load(Path(args.path))
    headers = [r for r in rows if r.get("kind") == "header"]
    configs = {r["digest"]: r["settings"] for r in rows if r.get("kind") == "config"}
    calls = [r for r in rows if r.get("kind") == "call"]
    errors = [r for r in rows if r.get("kind") == "error"]
    if args.prompt:
        calls = [r for r in calls if r.get("prompt_id") == args.prompt]
        errors = [r for r in errors if r.get("prompt_id") == args.prompt]
    summary = {"produced_by": "bench/sol_observe_report.py", "record": str(Path(args.path)),
               "headers": headers, "renders": [r for r in rows if r.get("kind") == "render"
                                               and (not args.prompt or r.get("prompt_id") == args.prompt)],
               "configs": configs, "errors": len(errors), "forwards": [], "sol": None,
               "not_quotable": "timings; every Sol call synchronized the stream to copy its counts"}

    for h in headers:
        print(f"header: schema {h.get('schema')}, comfy-kitchen {h.get('comfy_kitchen_version')}, "
              f"pack {h.get('pack_git_head')}, pid {h.get('pid')} on {h.get('host')}, "
              f"device {h.get('device')}, raw sidecar {'on' if h.get('raw_sidecar') else 'off'}, "
              f"timing quotable: {h.get('timing_quotable')}")
    for r in rows:
        if r.get("kind") == "render":
            sm = r.get("summary") or {}
            pdd = sm.get("pdd")
            print(f"render {r['prompt_id']}: workflow {r.get('workflow_file') or '(unmatched)'} [{r.get('match')}], "
                  f"process render index {r.get('process_render_index')} "
                  f"({'cold, first in this process' if r.get('process_render_index') == 0 else 'warm'}), "
                  f"{'PDD ' + str(pdd.get('steps')) + ' evaluations, ' + str(pdd.get('lora_name')) if pdd else 'no PDD node'}, "
                  f"sampler {sm.get('sampler')}, scheduler {sm.get('scheduler')}, unet {sm.get('unet')}")
    for d, s in configs.items():
        sel = f"topk {s.get('topk_ratio')}" if s.get("topk_ratio") else f"tau {s.get('tau')}"
        print(f"config {d}: {sel}, tail {s.get('tail')}, window sigma [{s.get('sigma_end')}, "
              f"{s.get('sigma_start')}], dense_blocks {s.get('dense_blocks')}, sink "
              f"{s.get('sink_conditioning')}, min_tokens {s.get('min_tokens')}, blocks {s.get('n_blocks')}")
    if errors:
        print(f"\nERROR rows: {len(errors)}")
        for e in errors[:5]:
            print(f"  seq {e.get('seq')} stage {e.get('stage')}: {e.get('message')}")
        if not args.past_errors:
            print("this record STOPPED; nothing below it is a complete result. "
                  "--past-errors to summarise anyway")
            return 0
    if not calls:
        print("no call rows")
        return 0

    # identity
    by_prompt = Counter(r.get("prompt_id") for r in calls)
    sources = Counter(r.get("identity_source") for r in calls)
    print(f"\ncalls: {len(calls)} across {len(by_prompt)} prompt id(s); identity sources {dict(sources)}")
    if args.join:
        for pid in by_prompt:
            if pid is None:
                print(f"  {by_prompt[pid]} row(s) carry no prompt id (identity_source above says why)")
                continue
            url = f"{args.join.rstrip('/')}/history/{pid}"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = json.load(resp)
                known = pid in body
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                print(f"  {pid}: could not ask {url} ({exc})")
                continue
            print(f"  {pid}: {'KNOWN to' if known else 'NOT known to'} {args.join}, {by_prompt[pid]} row(s)")

    # Rows with no sigma did not come from a sampler forward: the chain
    # assert's probes, or a call outside any executing context. They are
    # counted apart, never as a forward.
    probes = [r for r in calls if r.get("sigma") is None]
    if probes:
        print(f"  {len(probes)} row(s) with no sigma (probes / outside a forward): routes "
              f"{dict(Counter(r['route'] for r in probes))}, executing nodes "
              f"{sorted({str(r.get('executing_node_id')) for r in probes})}")
    summary["probe_rows"] = {"count": len(probes), "routes": dict(Counter(r["route"] for r in probes))}
    # per (prompt, schedule step): routes and the block set
    steps = defaultdict(list)
    for r in calls:
        if r.get("sigma") is None:
            continue
        key = (r.get("prompt_id"), r.get("schedule", {}).get("schedule_index"),
               r.get("sigma"))
        steps[key].append(r)
    print("\nper (prompt, step): rows by route, DiT blocks seen, rows without a block")
    for (pid, idx, sigma), rs in sorted(steps.items(), key=lambda kv: (str(kv[0][0]), -(kv[0][2] or 0))):
        routes = Counter(r["route"] for r in rs)
        blocks = sorted({r["block"] for r in rs if r.get("block") is not None})
        unknown = sum(1 for r in rs if r.get("block") is None)
        span = f"{blocks[0]}..{blocks[-1]} ({len(blocks)})" if blocks else "-"
        print(f"  {str(pid)[:8]:>8} step {str(idx):>4} sigma {sigma if sigma is None else round(sigma, 4)!s:>8}: "
              f"{dict(routes)}, blocks {span}, no-block rows {unknown}")
        summary["forwards"].append({"prompt_id": pid, "schedule_index": idx, "sigma": sigma,
                                    "routes": dict(routes), "dit_blocks": len(blocks), "no_block_rows": unknown})

    sol = [r for r in calls if r["route"] in ("sol", "sol_chunked") and r.get("sigma") is not None]

    def _finish():
        if args.json:
            Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
            print(f"\nsummary written to {args.json}")
        return 0

    if not sol:
        print("\nno Sol rows")
        return _finish()
    kd = [r["kernel_density"]["mean"] for r in sol if r.get("kernel_density")]
    oe = [r["ordering_effect_density"]["overall"] for r in sol
          if r.get("ordering_effect_density") and r["ordering_effect_density"].get("overall") is not None]
    rd = [r["routed_density"]["mean"] for r in sol if r.get("routed_density")]
    print(f"\nSol rows: {len(sol)}; mean over rows of: kernel density {_fmt(sum(kd) / len(kd)) if kd else '-'}, "
          f"ordering-effect density (pair-weighted, the analyze_routing.py number) "
          f"{_fmt(sum(oe) / len(oe)) if oe else '-'}, routed density (query-weighted) "
          f"{_fmt(sum(rd) / len(rd)) if rd else '-'}; shape_ok false on "
          f"{sum(1 for r in sol if not r.get('shape_ok'))} row(s)")
    summary["sol"] = {"rows": len(sol), "rows_note": "forward rows only; probe rows are under probe_rows",
                      "kernel_density_mean_over_rows": sum(kd) / len(kd) if kd else None,
                      "ordering_effect_density_mean_over_rows": sum(oe) / len(oe) if oe else None,
                      "routed_density_query_weighted_mean_over_rows": sum(rd) / len(rd) if rd else None,
                      "shape_ok_false": sum(1 for r in sol if not r.get("shape_ok")),
                      "geometry": {k: sol[0].get(k) for k in ("B", "H", "T", "NQ", "NTB")},
                      "block_by_step_ordering_effect": {}}
    for r in sol:
        b, i = r.get("block"), r.get("schedule", {}).get("schedule_index")
        oev = (r.get("ordering_effect_density") or {}).get("overall")
        if b is not None and oev is not None:
            summary["sol"]["block_by_step_ordering_effect"].setdefault(str(i), {})[str(b)] = oev
    if args.block_table:
        by_block = defaultdict(list)
        for r in sol:
            by_block[(r.get("block"), r.get("schedule", {}).get("schedule_index"))].append(r)
        blocks = sorted({b for b, _ in by_block if b is not None})
        idxs = sorted({i for _, i in by_block if i is not None})
        print("\nordering-effect density (pair-weighted), block x step:")
        print("  block " + "".join(f"{i:>8}" for i in idxs))
        for b in blocks:
            cells = []
            for i in idxs:
                rs = by_block.get((b, i), [])
                vals = [r["ordering_effect_density"]["overall"] for r in rs
                        if r.get("ordering_effect_density") and r["ordering_effect_density"].get("overall") is not None]
                cells.append(_fmt(sum(vals) / len(vals), 8) if vals else f"{'-':>8}")
            print(f"  {b:>5} " + "".join(cells))
    if args.segments:
        print("\nper-segment ordering-effect density (pair-weighted with row overlaps), mean over Sol rows per step:")
        by_step = defaultdict(list)
        for r in sol:
            if r.get("per_segment"):
                by_step[r.get("schedule", {}).get("schedule_index")].append(r)
        for i, rs in sorted(by_step.items(), key=lambda kv: (kv[0] is None, kv[0])):
            acc = defaultdict(list)
            for r in rs:
                for s in r["per_segment"]:
                    if s.get("ordering_effect") is not None:
                        acc[(s["kind"], s["start"], s["stop"])].append(s["ordering_effect"])
            cells = ", ".join(f"{k[0]}[{k[1]}:{k[2]}) {sum(v) / len(v):.4f}" for k, v in acc.items())
            print(f"  step {i}: {cells if cells else '(no defined segment rows)'}")
            summary["sol"].setdefault("per_segment_ordering_effect_by_step", {})[str(i)] = {
                f"{k[0]}[{k[1]}:{k[2]})": sum(v) / len(v) for k, v in acc.items()}
    return _finish()


if __name__ == "__main__":
    sys.exit(main())
