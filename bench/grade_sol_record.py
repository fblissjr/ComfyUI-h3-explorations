#!/usr/bin/env python3
"""Grade a live Sol route record structurally, per render and per forward.

The acceptance Codex set for the first armed capture (2026-09-01), as
assertions rather than a row total, because execution batching can put more
than one forward at the same sigma and a fixed count would then be wrong in
both directions:

  per (prompt, sigma occurrence)
      every DiT block 0..n_blocks-1 appears exactly once
      each DiT row is `sol` or `dense_block` inside the configured window and
      `composed_patch` outside it, with dense_blocks routed `dense_block`
      the sampler's non-DiT rows (token refiner) have no block label
  per prompt
      one prompt id per render group, joinable to /history when asked
      no `unknown`-scope row carries a DiT route, no error rows, no shape
      failures
      `sol` rows carry counts (and a raw pointer when the sidecar is on) and
      every other row carries none; every raw pointer's CRC verifies

Rows whose executing node is not the sampler (the chain-assert probe, for
one) are reported separately and not graded as forwards.

Two expectations are PARAMETERS because they depend on the graph and the
run, not on the observer, and the first live record proved both defaults
wrong for the canonical PDD graph:

  --refiner-rows   sampler-owned no-block rows per forward. Two when the
                   sampler receives raw Qwen states and `_forward` runs the
                   token refiner; ZERO when the context arrives already
                   projected, which is what the shipped t2v graphs do.
  --probe-rows     rows from executing nodes other than the sampler. One
                   when the chain assert's live probe runs; zero when its
                   VRAM headroom guard skips it, which the server log says.

A file shared by several renders (one armed process, several prompts) is
graded one prompt at a time with `--prompt <id>`; the other prompts' rows,
complete or still being written, are left aside.

    python bench/grade_sol_record.py <dir-or-jsonl> [--join http://127.0.0.1:8188]
        [--prompt ID] [--refiner-rows N] [--probe-rows N] [--single]

Exit 0 when every graded assertion holds, 1 otherwise, 2 when there is
nothing to grade.
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

from sol_observe_report import _load  # noqa: E402

FAILED: list[str] = []


def fail(msg):
    FAILED.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--join", metavar="URL")
    ap.add_argument("--refiner-rows", type=int, default=2,
                    help="sampler-owned no-block rows expected per forward (H3's two "
                         "token-refiner attention calls)")
    ap.add_argument("--prompt", metavar="ID",
                    help="grade only this prompt id (a file shared by several renders, "
                         "one of which may still be running)")
    ap.add_argument("--single", action="store_true",
                    help="a fresh single-render directory: exactly one prompt id")
    ap.add_argument("--probe-rows", type=int, default=1,
                    help="rows expected from executing nodes other than the sampler "
                         "(the chain-assert probe is one); -1 to not grade")
    args = ap.parse_args()

    path = Path(args.path)
    jsonl = path if path.is_file() else sorted(path.glob("sol_observe_*.jsonl"))[-1:]
    if isinstance(jsonl, list):
        if not jsonl:
            print("nothing to grade: no record file")
            return 2
        jsonl = jsonl[0]
    rows = _load(jsonl)
    headers = [r for r in rows if r.get("kind") == "header"]
    configs = {r["digest"]: r["settings"] for r in rows if r.get("kind") == "config"}
    calls = [r for r in rows if r.get("kind") == "call"]
    errors = [r for r in rows if r.get("kind") == "error"]
    if args.prompt:
        skipped = len({r.get("prompt_id") for r in calls if r.get("prompt_id") != args.prompt})
        calls = [r for r in calls if r.get("prompt_id") == args.prompt]
        errors = [r for r in errors if r.get("prompt_id") == args.prompt]
        print(f"grading prompt {args.prompt} only; {skipped} other prompt id(s) in the file left ungraded")
    if not calls:
        print("nothing to grade: no call rows")
        return 2
    raw_on = bool(headers and headers[0].get("raw_sidecar"))
    print(f"record {jsonl.name}: {len(calls)} call rows, {len(configs)} config(s), raw sidecar {'on' if raw_on else 'off'}")

    if errors:
        fail(f"{len(errors)} error row(s): first is seq {errors[0].get('seq')} {errors[0].get('stage')}: {errors[0].get('message')}")
    else:
        ok("no error rows")
    bad_shape = [r for r in calls if r["route"] == "sol" and not r.get("shape_ok")]
    if bad_shape:
        fail(f"{len(bad_shape)} sol row(s) with shape_ok false")
    else:
        ok("no shape failures")

    # counts only on sol rows; raw pointers verify
    from _live_sol import sol_observe as _load_observer
    sol_observe = _load_observer()
    wrong_counts = [r for r in calls if (r["route"] == "sol") != ("kernel_density" in r)]
    if wrong_counts:
        fail(f"{len(wrong_counts)} row(s) where counts are present iff route is sol is violated")
    else:
        ok("counts present on sol rows and only there")
    stray_raw = [r for r in calls if r["route"] != "sol" and (r.get("raw") or {}).get("offset") is not None]
    if stray_raw:
        fail(f"{len(stray_raw)} non-sol row(s) carry a raw pointer")
    else:
        ok("no raw pointer on a non-sol row")
    if raw_on:
        sol_rows = [r for r in calls if r["route"] == "sol"]
        missing = [r for r in sol_rows if not (r.get("raw") or {}).get("offset") and (r.get("raw") or {}).get("offset") != 0]
        crc_bad = 0
        for r in sol_rows:
            if r in missing:
                continue
            try:
                sol_observe.read_raw(jsonl, r)
            except ValueError:
                crc_bad += 1
        if missing or crc_bad:
            fail(f"raw pointers: {len(missing)} missing, {crc_bad} failed CRC/length")
        else:
            ok(f"every raw pointer verifies ({len(sol_rows)} sol rows)")

    # group by prompt
    by_prompt = defaultdict(list)
    for r in calls:
        by_prompt[r.get("prompt_id")].append(r)
    if None in by_prompt:
        fail(f"{len(by_prompt[None])} row(s) carry no prompt id (identity_source "
             f"{Counter(r.get('identity_source') for r in by_prompt[None])})")
    named = [p for p in by_prompt if p is not None]
    if args.single and len(named) != 1:
        fail(f"--single: {len(named)} prompt id(s) in a directory that should hold one render")
    elif args.single:
        ok("one prompt id in the directory")
    for pid, rs in by_prompt.items():
        if pid is None:
            continue
        print(f"\nprompt {pid}: {len(rs)} rows")
        cfgs = {r["config"] for r in rs}
        if len(cfgs) != 1:
            fail(f"{pid}: rows reference {len(cfgs)} configs")
            continue
        cfg = configs[next(iter(cfgs))]
        n_blocks = cfg.get("n_blocks") or 0
        dense = set(cfg.get("dense_blocks") or [])
        s_start, s_end = cfg.get("sigma_start"), cfg.get("sigma_end")
        # the sampler is the executing node that owns the DiT rows
        by_node = Counter(r.get("executing_node_id") for r in rs if r.get("block") is not None)
        if not by_node:
            fail(f"{pid}: no row carries a block label")
            continue
        sampler = by_node.most_common(1)[0][0]
        other = [r for r in rs if r.get("executing_node_id") != sampler]
        if other:
            print(f"  note  {len(other)} row(s) from other executing node(s) "
                  f"{sorted({r.get('executing_node_id') for r in other})}, routes "
                  f"{dict(Counter(r['route'] for r in other))}: not graded as forwards")
        if args.probe_rows >= 0:
            if len(other) != args.probe_rows or any(r.get("block") is not None for r in other):
                fail(f"{pid}: expected {args.probe_rows} non-sampler probe row(s) with no block, "
                     f"found {len(other)}")
            else:
                ok(f"{pid}: {args.probe_rows} non-sampler probe row(s), no block label")
        srows = [r for r in rs if r.get("executing_node_id") == sampler]
        if args.join:
            url = f"{args.join.rstrip('/')}/history/{pid}"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = json.load(resp)
                if pid in body:
                    ok(f"{pid} is known to {args.join}/history")
                else:
                    fail(f"{pid} is NOT known to {args.join}/history")
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                fail(f"{pid}: could not ask {url} ({exc})")

        # forwards: group sampler rows by sigma occurrence, in sequence order
        forwards = []
        current = None
        for r in sorted(srows, key=lambda r: r["seq"]):
            sigma = r.get("sigma")
            block = r.get("block")
            # A forward is one sigma occurrence: the refiner rows (no block)
            # come first, then blocks 0..n-1. A block-0 row after blocks have
            # already been seen at the same sigma is a second forward there.
            new_forward = (current is None or current["sigma"] != sigma
                           or (block == 0 and current["blocks_seen"]))
            if new_forward:
                current = {"sigma": sigma, "rows": [], "blocks_seen": set()}
                forwards.append(current)
            current["rows"].append(r)
            if block is not None:
                current["blocks_seen"].add(block)
        print(f"  {len(forwards)} forward(s) at {len({f['sigma'] for f in forwards})} distinct sigma(s)")
        all_good = True
        for i, f in enumerate(forwards):
            sigma = f["sigma"]
            inside = sigma is not None and s_start is not None and s_end is not None and s_end <= sigma <= s_start
            dit = [r for r in f["rows"] if r.get("block") is not None]
            nodit = [r for r in f["rows"] if r.get("block") is None]
            blocks = Counter(r["block"] for r in dit)
            problems = []
            if sorted(blocks) != list(range(n_blocks)) or any(c != 1 for c in blocks.values()):
                dup = [b for b, c in blocks.items() if c > 1]
                missing = [b for b in range(n_blocks) if b not in blocks]
                problems.append(f"blocks: missing {missing[:6]}{'...' if len(missing) > 6 else ''}, duplicated {dup[:6]}")
            for r in dit:
                want = ("dense_block" if r["block"] in dense else "sol") if inside else "composed_patch"
                if r["route"] != want:
                    problems.append(f"block {r['block']} route {r['route']}, want {want}")
                    if len(problems) > 6:
                        break
            for r in nodit:
                if r["route"] in ("sol", "dense_block"):
                    problems.append(f"no-block row with DiT route {r['route']}")
            if len(nodit) != args.refiner_rows:
                problems.append(f"{len(nodit)} no-block rows, want {args.refiner_rows}")
            routes = dict(Counter(r["route"] for r in f["rows"]))
            label = "inside window" if inside else "outside window"
            if problems:
                all_good = False
                fail(f"forward {i} sigma {sigma!r} ({label}): {routes}; " + "; ".join(problems[:6]))
            else:
                ok(f"forward {i} sigma {sigma!r} ({label}): {len(dit)} DiT rows {routes}, {len(nodit)} no-block row(s)")
        if all_good:
            ok(f"{pid}: every forward complete and routed as configured")

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} assertion(s)")
        return 1
    print("all graded assertions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
