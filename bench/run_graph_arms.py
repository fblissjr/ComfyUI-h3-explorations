#!/usr/bin/env python3
"""Time shipped graph files as bench arms, with widget patches, to JSONL.

`bench_e2e_h3.py` builds its own graph, which is right for the attention
matrix it serves but means every new arm needs runner code. This runner is
the complement: it submits GRAPH FILES -- the exact JSON the repo ships and
the UI runs -- so an arm is a file plus zero or more widget patches, and a
new experiment is a command line rather than a code change.

What it records per render, one JSON object per line:

  - sampler seconds (`SamplerCustomAdvanced`, found by class not by id),
    decode seconds (`VAEDecode`), total submit-to-finish seconds, and the
    full per-node timing map, all from the websocket node-transition feed
    `bench_e2e_h3.run_once` already implements;
  - the arm label, graph path, applied patches, and seed, so the row is
    re-runnable from itself;
  - the substrate (GPU, driver, power limit vs stock, torch, git commit),
    read live at each render via `make_attention_defaults_json.substrate` --
    docs/hardware.md records that a power limit changes render times and
    appears in no workflow JSON, and that bench runs historically persisted
    nothing about their own conditions. This runner exists partly to end
    that: a timing row without its substrate cannot be compared later.

Arms alternate (A B A B ...) by default, same reasoning as bench_e2e_h3:
drift in clocks, thermals and allocator state is shared rather than
attributed to whichever arm ran second. The first render of the session
pays model load and autotune; use `--warmup` to run one discarded render
first (any arm), or accept that run 0 of the first arm is hot-start dirty
and drop it at analysis time -- the row records `"warmup": true` either way.

Usage:

  bench/run_graph_arms.py \
      --arm control=workflows/h3_probe_sol_on_all_refs_api.json \
      --arm cache=workflows/h3_probe_cache_easy_api.json \
      --runs 2 --seed 730451892 \
      --out bench/results/2026-08-18_cache_arms.jsonl

  # widget patch: arm label, node CLASS or id, input name, JSON value
  bench/run_graph_arms.py \
      --arm t03=workflows/h3_probe_cache_easy_api.json \
      --set t03:EasyCache.reuse_threshold=0.3 \
      ...

Needs a running ComfyUI. Runs are queued one at a time; the runner waits
for each to finish before submitting the next, so it can share a server
with nothing and nobody.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
import prompts as _prompts  # noqa: E402  -- bank id, hash, length, canvas, seed per row
sys.path.insert(0, str(HERE / "results"))

from bench_e2e_h3 import run_once  # noqa: E402
from make_attention_defaults_json import substrate  # noqa: E402


def _parse_value(s: str):
    # `@bank:<id>` is the bank entry's text, so a manifest names a prompt
    # instead of carrying a second copy of it (the owner's single-source rule,
    # 2026-09-03); a missing id fails here, before any render is queued.
    if s.startswith("@bank:"):
        return _prompts.text(s[len("@bank:"):])
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s  # bare string widget value


def _find_nodes(graph: dict, key: str) -> list[str]:
    """Node ids matching `key` as an exact id or a class_type."""
    if key in graph:
        return [key]
    return [nid for nid, n in graph.items() if n.get("class_type") == key]


def split_target(graph: dict, target: str) -> tuple[str, str]:
    """Split `NODE.FIELD` when FIELD may itself contain dots.

    A dynamic combo writes its branch widget as `shape.square_resolution`, so
    `rpartition(".")` -- which this used until 2026-08-21 -- took the LAST dot
    and produced the node key `27.shape`, which matches nothing. The tool could
    therefore not express a canvas change at all, which is why the arms for
    `docs/open_experiments.md` #22 could not be written as a command line.

    Split at the FIRST dot whose left side actually names a node, so the node
    key stays greedy-shortest and everything after it is the field. Falls back
    to the old behaviour so an unmatched target still reports against the last
    dot, which is the reading a typo in a plain field deserves.
    """
    parts = target.split(".")
    for i in range(1, len(parts)):
        head = ".".join(parts[:i])
        if _find_nodes(graph, head):
            return head, ".".join(parts[i:])
    node_key, _, field = target.rpartition(".")
    return node_key, field


def apply_patch(graph: dict, node_key: str, field: str, value) -> list[str]:
    nids = _find_nodes(graph, node_key)
    if not nids:
        raise SystemExit(f"patch target {node_key!r} matches no node")
    for nid in nids:
        inputs = graph[nid]["inputs"]
        if field not in inputs and "." in field:
            # A dynamic combo's branch widget. `shape=square` and
            # `shape.square_resolution=...` are one edit in the UI: the branch
            # selector changes and the previous branch's widget goes away. A
            # patch that only added the new key would leave the old band's
            # widget behind, and the graph would carry two.
            #
            # Guarded on the SELECTOR existing, not on the dot: a bare typo in
            # a plain field still fails loudly below, which is what the
            # unknown-input guard is for.
            selector = field.split(".", 1)[0]
            if selector in inputs:
                for stale in [k for k in inputs
                              if k.startswith(selector + ".") and k != field]:
                    del inputs[stale]
                inputs[field] = value
                continue
        if field not in inputs:
            raise SystemExit(
                f"node {nid} ({graph[nid]['class_type']}) has no input "
                f"{field!r}; it has {sorted(inputs)}")
        if isinstance(inputs[field], list):
            raise SystemExit(
                f"{node_key}.{field} is a link, not a widget; refusing")
        inputs[field] = value
    return nids


def _class_seconds(graph: dict, per_node: dict, class_type: str) -> float | None:
    nids = [nid for nid, n in graph.items()
            if n.get("class_type") == class_type]
    vals = [per_node[n] for n in nids if n in per_node]
    return sum(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[],
                    metavar="LABEL=GRAPH.json",
                    help="an arm: label=path to an API-format graph")
    ap.add_argument("--manifest", metavar="ARMS.json",
                    help="a JSON arm set: {\"arms\": {LABEL: repo-relative "
                         "graph path}, \"patches\": [LABEL:NODE.FIELD=VALUE, "
                         "...]} -- appended after any --arm/--set given here. "
                         "`bench/gate6_refview_arms.json` is one")
    ap.add_argument("--set", action="append", default=[], dest="patches",
                    metavar="LABEL:NODE.FIELD=VALUE",
                    help="widget patch for one arm; NODE is an id or a "
                         "class_type (patches every match); VALUE is JSON "
                         "or a bare string")
    ap.add_argument("--runs", type=int, default=1,
                    help="renders per arm (default 1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="set every RandomNoise node to this seed; without "
                         "it, arms keep whatever the graph ships")
    ap.add_argument("--hold-seed", action="store_true",
                    help="use the same seed for every repeat run of an arm. "
                         "Default is seed+run_index, because ComfyUI's "
                         "node-output cache returns a cached sampler result "
                         "for a byte-identical resubmission -- a 'timing' of "
                         "0.0s. Measured 2026-08-18: an identical control "
                         "arm came back in 3.0s total, sampler 0.0s. Hold "
                         "the seed only when a cache hit is the thing under "
                         "test")
    ap.add_argument("--warmup", metavar="LABEL", default=None,
                    help="run this arm once first and mark the row warmup")
    ap.add_argument("--no-alternate", action="store_true",
                    help="run each arm's runs as a block instead of A B A B")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-render timeout seconds (default 3600)")
    ap.add_argument("--out", required=True,
                    help="JSONL output path, appended to")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        for label, rel in man.get("arms", {}).items():
            args.arm.append(f"{label}={repo / rel}")
        args.patches = list(man.get("patches", [])) + list(args.patches)
    if not args.arm:
        ap.error("no arms: pass --arm LABEL=GRAPH.json or --manifest ARMS.json")

    arms: dict[str, dict] = {}
    order: list[str] = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm wants LABEL=PATH, got {spec!r}")
        raw = Path(path).read_bytes()
        graph = json.loads(raw)
        # Rows are committed records: a graph outside the repo is recorded
        # by basename only (an absolute scratch path leaks the machine's
        # layout into tracked content and points at files that will not
        # exist later), and every row carries the graph's content hash so
        # it stays identifiable either way.
        try:
            rec = str(Path(path).resolve().relative_to(repo))
        except ValueError:
            rec = Path(path).name
        arms[label] = {"path": rec, "graph": graph, "patches": [],
                       "sha": hashlib.sha256(raw).hexdigest()[:16]}
        order.append(label)

    for spec in args.patches:
        label, _, rest = spec.partition(":")
        target, _, raw = rest.partition("=")
        if label not in arms:
            raise SystemExit(f"--set names arm {label!r}, which has no --arm")
        node_key, field = split_target(arms[label]["graph"], target)
        if label not in arms or not node_key or not raw:
            raise SystemExit(f"--set wants LABEL:NODE.FIELD=VALUE, got {spec!r}")
        value = _parse_value(raw)
        nids = apply_patch(arms[label]["graph"], node_key, field, value)
        arms[label]["patches"].append(
            {"nodes": nids, "field": f"{node_key}.{field}", "value": value})

    if args.seed is None and args.runs > 1:
        raise SystemExit(
            "--runs > 1 needs --seed: without one, repeat runs are "
            "byte-identical resubmissions and the server's node-output "
            "cache returns stored results as 0.0s 'renders'")

    for label, arm in arms.items():
        # Distinct output prefix per arm, so renders land findably and no
        # arm overwrites another's clip. Any writer with a string
        # filename_prefix -- VHS_VideoCombine, SaveImage, whatever else --
        # not just the video muxer.
        for node in arm["graph"].values():
            w = node.get("inputs", {})
            if isinstance(w.get("filename_prefix"), str):
                w["filename_prefix"] += f"_{label}"

    schedule: list[tuple[str, bool]] = []          # (label, warmup)
    if args.warmup:
        if args.warmup not in arms:
            raise SystemExit(f"--warmup {args.warmup!r} is not an arm label")
        schedule.append((args.warmup, True))
    if args.no_alternate:
        for label in order:
            schedule += [(label, False)] * args.runs
    else:
        for i in range(args.runs):
            schedule += [(label, False) for label in order]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_index: dict[str, int] = {}
    for i, (label, warmup) in enumerate(schedule):
        arm = arms[label]
        seed_used = None
        if args.seed is not None:
            # See --hold-seed: repeats bump the seed so the node-output
            # cache cannot hand back a 0.0s "render". The warmup gets its
            # own seed (seed-1) and does NOT consume a run index -- letting
            # it consume one desynchronized seeds across arms (the warmup
            # arm's real runs at seed+1.. while the others sat at seed+0..),
            # which broke the seed-matched pairing this tool exists for.
            if warmup:
                seed_used = args.seed - 1
            else:
                seed_used = args.seed + (0 if args.hold_seed
                                         else run_index.get(label, 0))
            for nid in _find_nodes(arm["graph"], "RandomNoise"):
                arm["graph"][nid]["inputs"]["noise_seed"] = seed_used
                if "control_after_generate" in arm["graph"][nid]["inputs"]:
                    arm["graph"][nid]["inputs"]["control_after_generate"] = "fixed"
        if not warmup:
            run_index[label] = run_index.get(label, 0) + 1
        client_id = str(uuid.uuid4())
        print(f"[{i + 1}/{len(schedule)}] {label}"
              f"{' (warmup, discard)' if warmup else ''} ...", flush=True)
        t0 = time.time()
        prompt_id = None
        try:
            total_s, per_node, err, prompt_id = asyncio.run(
                run_once(args.host, arm["graph"], client_id, args.timeout,
                         return_prompt_id=True))
        except Exception as exc:
            # A submission-path failure (HTTP 400 on validation, server
            # down) must leave a row saying the arm was attempted, same as
            # an execution error -- a schedule that dies with a traceback
            # and no row is evidence that evaporated.
            total_s, per_node, err = None, {}, f"submit/transport: {exc}"
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": label,
            "graph": arm["path"],
            "graph_sha256": arm["sha"],
            "patches": arm["patches"],
            "seed": seed_used,
            "prompt_id": prompt_id,
            # What was rendered, not only which file (owner's rule
            # 2026-09-03): bank id or the full text when foreign, its hash,
            # length, canvas. `seed` above is the one actually used.
            "rendered": _prompts.describe(arm["graph"]),
            "warmup": warmup,
            "total_s": total_s,
            "sampler_s": _class_seconds(arm["graph"], per_node,
                                        "SamplerCustomAdvanced"),
            "decode_s": _class_seconds(arm["graph"], per_node, "VAEDecode"),
            "per_node_s": {k: round(v, 3) for k, v in per_node.items()},
            "error": err,
            "wall_s": round(time.time() - t0, 1),
            "substrate": substrate(),
        }
        # A sampler that "ran" in under a second did not run: the server's
        # node-output cache returned a stored result. The row is kept (it is
        # evidence about the harness), but no one should average it.
        #
        # **The `sampler_s is None` case was routed away from this flag until
        # 2026-08-22, and that is where the common cache hit actually lands.**
        # The old reasoning was that None means "no SamplerCustomAdvanced was
        # timed -- another sampler class, or the run never reached it", and
        # flagging it would be red-while-correct. Sound, except a fully cached
        # graph is exactly this shape: the server replays stored outputs, the
        # sampler never TRANSITIONS, so it never appears in per_node_s at all.
        # The escaped instance: on 2026-08-22 an unpatched arm re-run at a seed
        # already rendered an hour earlier came back at 2.9s with only the
        # save node timed, and this field said False.
        #
        # The ambiguity the old comment worried about is removable rather than
        # unavoidable: read the GRAPH. If it wires a SamplerCustomAdvanced and
        # that node produced no timing, nothing honest explains it -- an
        # arm using a different sampler class has no such node and is still
        # correctly exempt.
        has_sampler = any(n.get("class_type") == "SamplerCustomAdvanced"
                          for n in arm["graph"].values())
        row["sampler_untimed"] = (not err and row["sampler_s"] is None)
        row["suspect_cache_hit"] = not err and (
            (row["sampler_s"] is not None and row["sampler_s"] < 1.0)
            or (row["sampler_s"] is None and has_sampler))
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        status = (f"ERROR: {err}" if err else
                  f"total {total_s:.1f}s sampler {row['sampler_s'] or 0:.1f}s "
                  f"decode {row['decode_s'] or 0:.1f}s")
        print(f"    {status}", flush=True)
        if err:
            # A failed arm invalidates the pairing; stop rather than time
            # the survivor against nothing.
            return 1
    print(f"rows appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
