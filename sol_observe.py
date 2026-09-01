"""Record what the live Sol-Attn override actually did, one row per call.

## Why this exists

Every routed-density number in this repo before 2026-09-01 was an OFFLINE
approximation: `bench/analyze_routing.py` re-derives the router's decision
from the eager reference on captured activations, in float arithmetic the
kernel does not use, on the handful of blocks and steps a capture happened to
hold. The kernel itself computes the real thing on every call -- the route
stage writes one int32 per (batch, head, query block) that the exact stage
walks -- and the public API discarded it with the workspace. comfy-kitchen's
`sol_attn` now takes an optional `blk_cnt` out-parameter (our branch
`sol-blk-cnt`), so the count leaves the SAME invocation that produced the
attention output; nothing here re-runs routing to look at it.

What a row can answer: how dense the exact walk actually was, by transformer
block, sampling step, head and 64-row query block, and by QUERY segment once
the packed layout is joined. What it cannot answer: WHICH key blocks or key
segments were chosen (that is `blk_idx`, not recorded), and how much error the
route cost (no reference output is involved). Do not read a density here as
a quality claim; it is the cost side of the trade.

## Inert unless `H3_SOL_OBSERVE` is set

    H3_SOL_OBSERVE="dir=/path[,raw=0]" <comfy>/start.sh

Read once at import, like `h3_capture.py`: arming is a property of the SERVER
PROCESS, which is why `bench/restart_comfy.sh` lists the key in `ARMING_KEYS`
and refuses to restart an armed server. Unarmed, the node passes no `blk_cnt`
keyword at all, so an older installed wheel still renders and the default
path allocates, copies and synchronizes nothing. Armed with a wheel that lacks
the argument, `sol_attn_h3._require_kernel` fails the node at patch time
rather than recording nothing.

Arming is a named boolean (`_armed`), never a property of a path's spelling:
`pdd_observe.py` shipped `bool(str(Path("")))`, which is `True`, and armed
itself against the working directory.

## Files, both append-only

    <dir>/sol_observe_<start>.jsonl     one JSON object per line
    <dir>/blk_cnt_<start>.u16           raw counts, uint16, appended per Sol call

Row kinds, all carrying `schema`:

  header    once per process: schema, installed comfy-kitchen distribution
            version, pack git head, pid/host/device/torch, the arming spec,
            and `timing_quotable: false` -- an armed render synchronizes the
            stream once per Sol call, so nothing timed in it is a measurement.
  config    once per distinct node configuration, keyed by a digest every
            call row references. Settings are NOT process-global: one server
            can execute patched models from different Sol nodes, so they do
            not belong in the header.
  render    once per prompt id, the first time a call carries it: WHICH
            WORKFLOW the render ran under. The running prompt's graph is read
            from the server's queue, hashed the way `provenance.py` hashes a
            graph, and matched against every shipped file under `workflows/`
            (through `h3_config.graph_paths`), so `workflow_file` names the
            shipped graph when the submitted one is byte-for-byte a shipped
            one and is null with a reason otherwise. A summary of the graph
            travels with it -- PDD LoRA and its step count, UNET, sampler,
            scheduler steps, canvas and length -- so a reader can tell a PDD
            render from a base one without the file. `process_render_index`
            says how many prompts this process had already run (0 = cold, the
            first render after a restart; a restart is a new file with its own
            header and pid), because ComfyUI keeps models and node outputs
            resident between prompts and a warm render is a different cache
            state from a cold one. Routed counts are a function of the
            inputs alone; the index is there so that claim can be checked.
  call      one per override call, every route, not only Sol. Identity, step,
            block, shape, selection, sinks, the route actually taken, the
            allocator's high-water mark so far (`peak_alloc_bytes`), and for
            a Sol route the density summaries and the raw pointer. A
            `sol_chunked` route is the chunked producer (`sol_chunked_h3.py`)
            taking the call through Sol's gate delegate; it carries counts
            from its own launch like a `sol` row.
  error     the observer's own failure, written before it raises.

The raw append goes first and is flushed; the JSONL row that references it
(absolute offset, byte count, shape, dtype, CRC32) is committed after, under
one lock. A reader ignores an unreferenced raw tail left by an interrupted
render.

## Identity, read at CALL time and never at patch time

ComfyUI caches a node's output by input hash, so the patched model a sampler
receives may have been built for an earlier prompt. Everything identifying
here is read inside the call:

  prompt_id, executing_node_id, list_index
        `comfy_execution.utils.get_executing_context()`, the contextvar
        `execution.py` sets around every node execution. `prompt_id` is the
        key `/history` uses, so a row joins the submitted graph without any
        hashing. `executing_node_id` is the node RUNNING the forward --
        normally the sampler -- not the Sol patch node.
  conditioning_uuids, cond_or_uncond
        the complete lists from `transformer_options`; the UUIDs are minted
        when ComfyUI converts the conditioning for a sampler run.
  sigma, schedule
        `sigma` is authoritative. The schedule index is derived by isclose
        against `sample_sigmas` and carries its residual and a state
        (`matched`, `ambiguous`, `no_match`, `no_schedule`); it is null when it
        cannot be justified. `schedule_len` counts entries including the
        terminal value; `n_intervals` is one fewer.
  block
        `transformer_options["sol_block"]`, published by the block pre-hook and
        CLEARED by its paired post-hook (both in `sol_attn_h3.py`). Absent, the
        row is `scope: "unknown"` -- not "token refiner", which is an inference
        the row cannot support on its own.

## Denominators, stated in every row

`NTB = ceil(T / 64)` key blocks, and `forced[q] = |sink range| + |{q-1, q,
q+1} that exist and are not sink|` as a set, with sink_q rows at NTB. Three
figures, and the weighting is part of each name because two of them were
once called "the same density" and are not (Codex's review, 2026-09-01):

  kernel_density
      cnt / NTB over every query block; forced pairs INCLUDED. What the
      exact stage actually walked: the cost number. Query- and pair-weighted
      means coincide here because the denominator is constant.
  ordering_effect_density
      sum(cnt - forced) / sum(NTB - forced) over query blocks NOT in sink_q:
      PAIR-weighted, one ratio per call and one per head. This is the number
      `bench/analyze_routing.py::densities` computes and the one that joins
      the tables in `docs/SOLATTN.md`. Null where the denominator is zero.
  routed_density
      the distribution of the per-query-block fraction
      (cnt - forced) / (NTB - forced) over the same rows: min, p50, p95, max,
      mean. QUERY-weighted: every query block weighs the same regardless of
      how many free pairs it has, so its mean differs from the ratio above
      wherever forced varies -- at the sequence edges and where the diagonal
      meets a sink. Distribution statistics, not the join key.

sink_q rows are NTB by construction and are counted, never averaged into an
adaptive figure. Segment figures are QUERY-segment quantities, overlap-
weighted: a 64-row block crossing a boundary contributes to each side by the
rows it holds there; `ordering_effect` in a segment entry aggregates
numerator and denominator with those weights, `routed` averages the
per-block fractions. The full segment table is stored on the
row so the raw counts can be re-reduced any other way later.

`route: composed_patch` is a call Sol's composition gate declined, so the
composed foreign forward -- Sage on the canonical graphs -- ran directly: no
kernel, no counts, and the gate's own verdict (`outside_range` or
`ineligible`) leads the reason. `path` carries the same distinction
(`override` or `composed_patch`) on every row. Without those rows the
outside-window DiT calls of a canonical render were absent from the file,
which is what the first implementation did.

## The producer asserts its own shape, and a failure aborts the capture

Per Sol row, before anything is written: the count tensor is int32 of shape
(B, H, ceil(T/64)), every row is at least its forced floor, at most NTB, and
every sink_q row is exactly NTB. A wrong workspace slice fails these with
certainty on a real call; a recorder returning a constant LEGAL tensor does
not, which is what the multi-tau kernel test is for. On failure an `error` row
is appended and `SolObserveError` is raised -- and the node calls this OUTSIDE
the `except` that turns a kernel failure into a dense fallback, so an armed
render with a bad observer fails instead of finishing with a plausible partial
record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import socket
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch

SCHEMA = 1
BLOCK = 64
#: routes whose rows carry counts from a kernel launch: the direct path, and
#: the chunked producer taking the call through Sol's gate delegate
COUNT_ROUTES = ("sol", "sol_chunked")

_LOG = "[h3-sol-observe]"


class SolObserveError(RuntimeError):
    """The observer could not record truthfully; the armed capture must stop."""


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------

_armed: bool | None = None
_spec: dict = {}
_raw_spec: str = ""


def _parse(spec: str) -> dict:
    out = {"dir": None, "raw": True}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = os.path.expanduser(val)
        elif key == "raw" and val:
            out["raw"] = val.lower() not in ("0", "false", "no", "off")
    return out


def arm(spec: str | None) -> bool:
    """Set the arming state from a spec string. The server reads the
    environment once at import through this; tests call it directly."""
    global _armed, _spec, _raw_spec
    _raw_spec = spec or ""
    parsed = _parse(_raw_spec) if _raw_spec else {"dir": None, "raw": True}
    _armed = bool(_raw_spec) and bool(parsed["dir"])
    if _raw_spec and not parsed["dir"]:
        print(f"{_LOG} H3_SOL_OBSERVE set but no dir=; observation disabled")
    _spec = parsed
    _writer_reset()
    if _armed:
        print(f"{_LOG} ARMED: dir={parsed['dir']} raw={'on' if parsed['raw'] else 'off'}")
    return _armed


def enabled() -> bool:
    global _armed
    if _armed is None:
        arm(os.environ.get("H3_SOL_OBSERVE", ""))
    return bool(_armed)


def spec() -> dict:
    enabled()
    return dict(_spec, spec=_raw_spec)


# ---------------------------------------------------------------------------
# Writer: one lock, two append-only files, header once, configs by digest
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_jsonl = None
_raw = None
_paths: dict = {}
_seq = 0
_header_written = False
_configs_written: set = set()
_estimates_logged: set = set()
_renders_seen: dict = {}          # prompt_id -> process render index
_shipped_hashes: dict | None = None


def _writer_reset() -> None:
    global _jsonl, _raw, _paths, _seq, _header_written
    for f in (_jsonl, _raw):
        try:
            if f is not None:
                f.close()
        except OSError:
            pass
    _jsonl = _raw = None
    _paths = {}
    _seq = 0
    _header_written = False
    _configs_written.clear()
    _estimates_logged.clear()
    _renders_seen.clear()


def paths() -> dict:
    return dict(_paths)


def _open() -> None:
    global _jsonl, _raw, _paths
    if _jsonl is not None:
        return
    d = Path(_spec["dir"])
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _paths = {"jsonl": str(d / f"sol_observe_{stamp}.jsonl"),
              "raw": str(d / f"blk_cnt_{stamp}.u16") if _spec["raw"] else None}
    _jsonl = open(_paths["jsonl"], "a", encoding="utf-8")
    if _paths["raw"]:
        _raw = open(_paths["raw"], "ab")


def _kitchen_version() -> str | None:
    import importlib.metadata as md
    for name in ("comfy-kitchen", "comfy_kitchen"):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return None


def _header() -> dict:
    try:
        from .substrate import git_head
    except ImportError:                      # loaded by path, not as a package member
        from substrate import git_head
    device = None
    try:
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:                        # noqa: BLE001 -- header only
        device = None
    return {
        "kind": "header", "schema": SCHEMA, "produced_by": "sol_observe.py",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "comfy_kitchen_version": _kitchen_version(),
        "pack_git_head": git_head(Path(__file__).resolve().parent),
        "pid": os.getpid(), "host": socket.gethostname(), "device": device,
        "torch": torch.__version__, "arming_spec": _raw_spec,
        "raw_sidecar": bool(_spec.get("raw")),
        "timing_quotable": False,
        "why_not_quotable": ("every Sol call synchronizes the stream to copy "
                             "its counts to the host; wall times from an "
                             "armed render are not measurements"),
        "denominators": {
            "forced": "|sink range clamped to [0, NTB)| + |{q-1,q,q+1} existing and not sink| as a set; sink_q rows are NTB",
            "kernel_density": "cnt / NTB over every query block; forced pairs INCLUDED",
            "ordering_effect_density": ("sum(cnt - forced) / sum(NTB - forced) over query blocks not in sink_q: "
                                        "PAIR-weighted, the bench/analyze_routing.py number; null on a zero denominator"),
            "routed_density": ("distribution of (cnt - forced) / (NTB - forced) per query block not in sink_q: "
                               "QUERY-weighted, every block weighs the same; statistics, not the join key"),
            "per_segment": ("overlap-weighted QUERY-segment figures: `ordering_effect` aggregates numerator and "
                            "denominator with the row weights, `routed` averages per-block fractions"),
            "path": ("override = reached optimized_attention_override; composed_patch = Sol's gate declined "
                     "and the composed foreign forward ran, no counts (route composed_patch, gate verdict in reason)"),
        },
    }


def _write_row(row: dict, raw_bytes: bytes | None = None) -> dict | None:
    """Append `raw_bytes` (if any) then the row that references them. Returns
    the raw pointer written into the row, or None. Caller holds no lock."""
    global _seq, _header_written
    with _lock:
        _open()
        assert _jsonl is not None
        if not _header_written:
            _jsonl.write(json.dumps(_header()) + "\n")
            _header_written = True
        pointer = None
        if raw_bytes is not None and _raw is not None:
            offset = _raw.tell()
            _raw.write(raw_bytes)
            _raw.flush()
            pointer = {"file": os.path.basename(_paths["raw"]), "offset": offset,
                       "nbytes": len(raw_bytes), "crc32": zlib.crc32(raw_bytes) & 0xFFFFFFFF}
        _seq += 1
        row = dict(row, seq=_seq, schema=SCHEMA)
        if pointer is not None:
            row["raw"] = dict(row.get("raw") or {}, **pointer)
        _jsonl.write(json.dumps(row) + "\n")
        _jsonl.flush()
        return pointer


def graph_sha256(prompt) -> str:
    """The graph's identity, hashed exactly as `provenance.py` and
    `substrate.graph` hash it, so the three records join."""
    blob = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _shipped_graph_hashes() -> dict:
    """sha256 -> shipped file name for every graph `h3_config.graph_paths`
    walks, bench graphs included. Computed once per process."""
    global _shipped_hashes
    if _shipped_hashes is not None:
        return _shipped_hashes
    table = {}
    try:
        import importlib.util
        wf = Path(__file__).resolve().parent / "workflows"
        spec = importlib.util.spec_from_file_location("_h3_config_for_observe", wf / "h3_config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        for path in cfg.graph_paths(wf, include_bench=True):
            try:
                table[graph_sha256(json.loads(path.read_text()))] = str(path.relative_to(wf))
            except (OSError, ValueError):
                continue
    except Exception as exc:                          # noqa: BLE001 -- identity, not the render
        logging.warning(f"{_LOG} could not hash the shipped graphs: {exc}")
    _shipped_hashes = table
    return table


def _running_prompt(prompt_id: str):
    """(prompt graph, extra_data) for the prompt the server is executing,
    or (None, why). The queue keeps `(number, prompt_id, prompt, extra_data,
    outputs, ...)` per running item."""
    try:
        import server
        queue = server.PromptServer.instance.prompt_queue
        running = list(queue.currently_running.values())
    except Exception as exc:                          # noqa: BLE001
        return None, f"prompt unavailable: {type(exc).__name__}: {exc}"[:200]
    for item in running:
        if len(item) > 2 and item[1] == prompt_id:
            return item[2], None
    return None, "prompt unavailable: not in the server's running queue"


def _linked(graph: dict, value, key="value"):
    """Resolve a literal or a [node_id, slot] link to a literal, one hop."""
    if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
        return graph[str(value[0])].get("inputs", {}).get(key)
    return value


def graph_summary(graph: dict) -> dict:
    """What kind of render a graph is, from its nodes. Best-effort and
    literal: fields are null when the graph has no such node."""
    by_type: dict = {}
    for k, n in graph.items():
        if isinstance(n, dict) and "class_type" in n:
            by_type.setdefault(n["class_type"], []).append(n)
    first = lambda ct: (by_type.get(ct) or [None])[0]    # noqa: E731
    pdd = first("MiniMaxH3PDDLoRA")
    res = first("MiniMaxH3Resolution")
    sched = first("BasicScheduler")
    return {
        "class_types": sorted(by_type),
        "pdd": None if pdd is None else {
            "lora_name": pdd["inputs"].get("lora_name"),
            "steps": _linked(graph, pdd["inputs"].get("steps")),
            "strength": pdd["inputs"].get("strength"),
            "nfe": pdd["inputs"].get("nfe")},
        "unet": (first("UNETLoader") or {}).get("inputs", {}).get("unet_name"),
        "sampler": (first("KSamplerSelect") or {}).get("inputs", {}).get("sampler_name"),
        "scheduler": None if sched is None else {
            "scheduler": sched["inputs"].get("scheduler"), "steps": _linked(graph, sched["inputs"].get("steps"))},
        "resolution": None if res is None else {
            k2: v for k2, v in res["inputs"].items() if not isinstance(v, list)},
        "sol_nodes": len(by_type.get("MiniMaxH3SolAttn", [])),
        "sage_nodes": len(by_type.get("MiniMaxH3SageAttention", [])),
    }


def _ensure_render(prompt_id: str | None) -> None:
    """Write the `render` row the first time a prompt id is seen."""
    if prompt_id is None or prompt_id in _renders_seen:
        return
    index = len(_renders_seen)
    _renders_seen[prompt_id] = index
    graph, why = _running_prompt(prompt_id)
    row = {"kind": "render", "prompt_id": prompt_id, "process_render_index": index,
           "prior_prompt_ids": [p for p, i in _renders_seen.items() if i < index],
           "graph_sha256": None, "workflow_file": None, "match": why, "summary": None}
    if graph is not None:
        sha = graph_sha256(graph)
        shipped = _shipped_graph_hashes()
        row.update({"graph_sha256": sha, "workflow_file": shipped.get(sha),
                    "match": ("shipped graph, byte-identical" if sha in shipped
                              else "no shipped graph matches this hash; a modified or foreign graph"),
                    "summary": graph_summary(graph)})
    _write_row(row)


def config_digest(settings: dict) -> str:
    canon = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def _ensure_config(settings: dict) -> str:
    digest = config_digest(settings)
    if digest not in _configs_written:
        _write_row({"kind": "config", "digest": digest, "settings": settings})
        _configs_written.add(digest)
    return digest


# ---------------------------------------------------------------------------
# Semantics: forced pairs, densities, segments, schedule index
# ---------------------------------------------------------------------------

def forced_counts(ntb: int, sink_blocks, sink_q) -> torch.Tensor:
    """Per query block, the count the route kernel produces when nothing
    clears the threshold: the sink range clamped to [0, NTB), plus the
    diagonal {q-1, q, q+1} restricted to blocks that exist and are not sink,
    as a set union; sink_q rows attend everything. int32 (NTB,), CPU."""
    s0 = max(0, min(int(sink_blocks[0]), ntb))
    s1 = max(s0, min(int(sink_blocks[1]), ntb))
    q = torch.arange(ntb)
    forced = torch.full((ntb,), s1 - s0, dtype=torch.int32)
    for off in (-1, 0, 1):
        b = q + off
        forced += ((b >= 0) & (b < ntb) & ~((b >= s0) & (b < s1))).to(torch.int32)
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    forced[q0:q1] = ntb
    return forced


def shape_check(counts: torch.Tensor, batch: int, heads: int, tokens: int,
                sink_blocks, sink_q) -> tuple[bool, str | None, torch.Tensor, int]:
    """(ok, why, forced, NTB). `counts` on CPU. Every clause is one the route
    kernel guarantees, so a violation is a wrong slice, not a strange input."""
    ntb = (int(tokens) + BLOCK - 1) // BLOCK
    want = (int(batch), int(heads), ntb)
    forced = forced_counts(ntb, sink_blocks, sink_q)
    if counts.dtype != torch.int32:
        return False, f"dtype {counts.dtype}, want int32", forced, ntb
    if tuple(counts.shape) != want:
        return False, f"shape {tuple(counts.shape)}, want {want}", forced, ntb
    if int(counts.max()) > ntb:
        return False, f"max {int(counts.max())} exceeds NTB {ntb}", forced, ntb
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    if q1 > q0 and bool((counts[:, :, q0:q1] != ntb).any()):
        return False, f"a sink_q row in [{q0}, {q1}) is not NTB {ntb}", forced, ntb
    floor = forced.view(1, 1, ntb)
    below = counts < floor
    if bool(below.any()):
        i = int(below.flatten().nonzero()[0])
        return False, (f"count below its forced floor at flat index {i}: "
                       f"{int(counts.flatten()[i])} < {int(floor.expand(want).flatten()[i])}"), forced, ntb
    return True, None, forced, ntb


def _stats(x: torch.Tensor) -> dict | None:
    x = x[~torch.isnan(x)].double()
    if x.numel() == 0:
        return None
    return {"min": float(x.min()), "p50": float(x.quantile(0.5)),
            "p95": float(x.quantile(0.95)), "max": float(x.max()),
            "mean": float(x.mean()), "n": int(x.numel())}


def _nanmean_or_none(x: torch.Tensor) -> float | None:
    x = x[~torch.isnan(x)]
    return float(x.double().mean()) if x.numel() else None


def densities(counts: torch.Tensor, forced: torch.Tensor, ntb: int,
              sink_q) -> tuple[torch.Tensor, torch.Tensor]:
    """(kernel (B,H,NQ), adaptive (B,H,NQ) per-query fraction with NaN where
    undefined -- sink_q rows and zero denominators). Query-weighted by
    construction: reduce `adaptive` with a mean and every block weighs the
    same. For the pair-weighted ratio use `ordering_effect`."""
    c = counts.double()
    kernel = c / ntb
    f = forced.double().view(1, 1, -1)
    den = ntb - f
    adaptive = torch.where(den > 0, (c - f) / den.clamp_min(1), torch.full_like(c, float("nan")))
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    if q1 > q0:
        adaptive[:, :, q0:q1] = float("nan")
    return kernel, adaptive


def ordering_effect(counts: torch.Tensor, forced: torch.Tensor, ntb: int, sink_q,
                    weights: torch.Tensor | None = None) -> tuple[float | None, list, int, int]:
    """PAIR-weighted adaptive density: sum(cnt - forced) / sum(NTB - forced)
    over query blocks outside sink_q, overall and per head, plus the two
    integer sums. `weights` (NQ,) multiplies both sums per query block --
    the segment reducer passes row overlaps. This is
    `bench/analyze_routing.py::densities`'s ordering-effect density."""
    c = counts.double()
    f = forced.double().view(1, 1, -1)
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    live = torch.ones(ntb, dtype=torch.float64)
    if q1 > q0:
        live[q0:q1] = 0.0
    if weights is not None:
        live = live * weights.double()
    num = ((c - f) * live).sum(dim=(0, 2))                       # per head
    den = ((ntb - f).expand_as(c) * live).sum(dim=(0, 2))
    per_head = [float(n / d) if float(d) > 0 else None for n, d in zip(num, den)]
    n_all, d_all = float(num.sum()), float(den.sum())
    overall = n_all / d_all if d_all > 0 else None
    return overall, per_head, int(round(n_all)), int(round(d_all))


def segment_densities(kernel: torch.Tensor, adaptive: torch.Tensor, segments, tokens: int,
                      counts: torch.Tensor | None = None, forced: torch.Tensor | None = None,
                      ntb: int | None = None, sink_q=(0, 0)) -> list | None:
    """Overlap-weighted QUERY-segment figures. A 64-row block straddling a
    boundary counts toward each side by the rows it holds there. `kernel` and
    `adaptive_query_weighted` average per-block values; `ordering_effect`
    aggregates numerator and denominator with the same weights. Preserves
    every occurrence and kind in the table, in order."""
    if not segments:
        return None
    out = []
    nq = kernel.shape[-1]
    for entry in segments:
        a, b, kind = int(entry[0]), int(entry[1]), str(entry[2])
        b = min(b, int(tokens))
        if b <= a:
            out.append({"kind": kind, "start": a, "stop": b, "rows": 0, "query_blocks": None,
                        "kernel": None, "routed": None, "ordering_effect": None})
            continue
        q0, q1 = a // BLOCK, (b - 1) // BLOCK
        q1 = min(q1, nq - 1)
        qs = torch.arange(q0, q1 + 1)
        w = (torch.minimum(torch.tensor(b), (qs + 1) * BLOCK)
             - torch.maximum(torch.tensor(a), qs * BLOCK)).double()     # rows of this segment per block
        ks = kernel[:, :, q0:q1 + 1]
        rs = adaptive[:, :, q0:q1 + 1]
        wk = w.view(1, 1, -1).expand_as(ks)
        kd = float((ks * wk).sum() / wk.sum())
        mask = ~torch.isnan(rs)
        wr = (wk * mask.float()).sum()
        rd = float((torch.nan_to_num(rs) * wk).sum() / wr) if float(wr) > 0 else None
        oe = None
        if counts is not None and forced is not None and ntb is not None:
            full_w = torch.zeros(nq, dtype=torch.float64)
            full_w[q0:q1 + 1] = w
            oe = ordering_effect(counts, forced, ntb, sink_q, weights=full_w)[0]
        out.append({"kind": kind, "start": a, "stop": b, "rows": b - a,
                    "query_blocks": [q0, q1 + 1], "kernel": kd,
                    "routed": rd, "ordering_effect": oe})
    return out


def schedule_index(sigma: float | None, sample_sigmas) -> dict:
    if sigma is None:
        return {"state": "no_sigma", "schedule_index": None, "nearest_index": None,
                "residual": None, "schedule_len": None, "n_intervals": None}
    if sample_sigmas is None:
        return {"state": "no_schedule", "schedule_index": None, "nearest_index": None,
                "residual": None, "schedule_len": None, "n_intervals": None}
    try:
        s = [float(x) for x in sample_sigmas]
    except TypeError:
        return {"state": "no_schedule", "schedule_index": None, "nearest_index": None,
                "residual": None, "schedule_len": None, "n_intervals": None}
    if not s:
        return {"state": "no_schedule", "schedule_index": None, "nearest_index": None,
                "residual": None, "schedule_len": 0, "n_intervals": None}
    diffs = [abs(x - sigma) for x in s]
    nearest = min(range(len(s)), key=diffs.__getitem__)
    close = [j for j, x in enumerate(s) if math.isclose(x, sigma, rel_tol=1e-6, abs_tol=1e-6)]
    if len(close) == 1:
        state, idx = "matched", close[0]
    elif close:
        state, idx = "ambiguous", None
    else:
        state, idx = "no_match", None
    return {"state": state, "schedule_index": idx, "nearest_index": nearest,
            "residual": diffs[nearest], "schedule_len": len(s), "n_intervals": len(s) - 1}


def _identity() -> dict:
    try:
        from comfy_execution.utils import get_executing_context
    except Exception:                                  # noqa: BLE001
        return {"prompt_id": None, "executing_node_id": None, "list_index": None,
                "identity_source": "unavailable"}
    ctx = get_executing_context()
    if ctx is None:
        return {"prompt_id": None, "executing_node_id": None, "list_index": None,
                "identity_source": "no_executing_context"}
    return {"prompt_id": str(ctx.prompt_id), "executing_node_id": str(ctx.node_id),
            "list_index": ctx.list_index, "identity_source": "comfy_execution.utils"}


def _uuids(options) -> list | None:
    u = options.get("uuids") if isinstance(options, dict) else None
    if u is None:
        return None
    try:
        return [str(x) for x in u]
    except TypeError:
        return [str(u)]


def _cond_or_uncond(options) -> list | None:
    c = options.get("cond_or_uncond") if isinstance(options, dict) else None
    if c is None:
        return None
    try:
        return [int(x) for x in c]
    except TypeError:
        return None


def _segments(options) -> list | None:
    segs = options.get("h3_segments") if isinstance(options, dict) else None
    if not segs:
        return None
    return [[int(a), int(b), str(kind)] for a, b, kind in segs]


def _topk_budget(ntb: int, sink_blocks, ratio: float) -> int | None:
    try:
        from comfy_kitchen.backends.eager.sol_attn import _sink_count, _topk_count
    except Exception:                                  # noqa: BLE001
        return None
    return int(_topk_count(ntb - _sink_count(ntb, int(sink_blocks[0]), int(sink_blocks[1])), ratio))


# ---------------------------------------------------------------------------
# The entry the node calls
# ---------------------------------------------------------------------------

def record(*, route: str, reason: str | None, counts: torch.Tensor | None, options,
           settings: dict, block, block_tau, tokens: int, batch: int, heads: int,
           sink, sink_q, tail: bool, topk_ratio: float, min_tokens: int,
           path: str = "override") -> None:
    """One call row. Raises SolObserveError rather than writing a row it
    cannot stand behind; the caller must not catch that inside the kernel
    fallback. `path` names the code that executed the call: `override`, or
    `composed_patch` when Sol's composition gate declined it and the composed
    foreign forward ran instead (no kernel, no counts)."""
    if not enabled():
        return
    digest = _ensure_config(settings)
    identity = _identity()
    _ensure_render(identity.get("prompt_id"))
    sigmas = options.get("sigmas") if isinstance(options, dict) else None
    sigma = None
    if sigmas is not None:
        try:
            sigma = float(sigmas[0]) if getattr(sigmas, "ndim", 1) else float(sigmas)
        except (TypeError, IndexError, ValueError):
            sigma = None
    ntb = (int(tokens) + BLOCK - 1) // BLOCK
    row = {
        "kind": "call", "t_wall": time.time(), "config": digest,
        **identity,
        "conditioning_uuids": _uuids(options), "cond_or_uncond": _cond_or_uncond(options),
        "block": None if block is None else int(block),
        "scope": "dit" if block is not None else "unknown",
        "sigma": sigma,
        "schedule": schedule_index(sigma, options.get("sample_sigmas") if isinstance(options, dict) else None),
        "B": int(batch), "H": int(heads), "T": int(tokens), "NQ": ntb, "NTB": ntb,
        "selection": ({"mode": "topk", "topk_ratio": float(topk_ratio), "tau": None,
                       "budget": _topk_budget(ntb, sink, topk_ratio)}
                      if topk_ratio else
                      {"mode": "tau", "tau": None if block_tau is None else float(block_tau),
                       "topk_ratio": 0.0, "budget": None}),
        "tail": bool(tail), "sink_blocks": [int(sink[0]), int(sink[1])],
        "sink_q": [int(sink_q[0]), int(sink_q[1])], "min_tokens": int(min_tokens),
        "route": route, "reason": reason, "path": path,
        # The allocator's high-water mark so far in this process, read for
        # free. Per render, the last row's value is the peak; the memory
        # lever the chunked producer claims is graded on it.
        "peak_alloc_bytes": (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
    }
    if route not in COUNT_ROUTES:
        _write_row(row)
        return
    if counts is None:
        _write_row({**row, "kind": "error", "stage": "record",
                    "message": f"route {route} with no count tensor"})
        raise SolObserveError(f"route {route} recorded with no blk_cnt tensor")

    # The one synchronization an armed render pays: the copy to host.
    c = counts.detach().to("cpu")
    ok, why, forced, _ = shape_check(c, batch, heads, tokens, sink, sink_q)
    if not ok:
        _write_row({**row, "kind": "error", "stage": "shape_check", "message": why,
                    "counts_shape": list(c.shape), "counts_dtype": str(c.dtype)})
        raise SolObserveError(f"blk_cnt failed its shape check: {why}")

    kernel, adaptive = densities(c, forced, ntb, sink_q)
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    # The named decomposition is computed from its own definitions, never
    # from the minimum of the sum: with no sink the minimum forced count is
    # still an edge diagonal, and reporting it as the sink was the defect
    # Codex's review found on the first revision.
    s0 = max(0, min(int(sink[0]), ntb))
    s1 = max(s0, min(int(sink[1]), ntb))
    sink_count = s1 - s0
    outside_q = [i for i in range(ntb) if not (q0 <= i < q1)]
    diag = (forced[outside_q] - sink_count) if outside_q else None
    oe_overall, oe_heads, oe_num, oe_den = ordering_effect(c, forced, ntb, sink_q)
    # Stamp the weighting only on a DEFINED distribution. `dict(None or {},
    # weighting=...)` is a truthy {"weighting": "query"} with no mean, which
    # broke the null contract the header states (Codex's follow-up review).
    routed_stats = _stats(adaptive.flatten())
    if routed_stats is not None:
        routed_stats["weighting"] = "query"
    segments = _segments(options)
    row.update({
        "forced": {"sink": sink_count, "sink_range_clamped": [s0, s1],
                   "diag_min": int(diag.min()) if diag is not None else None,
                   "diag_max": int(diag.max()) if diag is not None else None,
                   "rows_outside_sink_q": len(outside_q)},
        "sink_q_rows": q1 - q0,
        "kernel_density": _stats(kernel.flatten()),
        "ordering_effect_density": {"overall": oe_overall, "numerator": oe_num,
                                    "denominator": oe_den, "weighting": "pair"},
        "routed_density": routed_stats,
        "per_head": {"kernel_mean": [float(x) for x in kernel.mean(dim=(0, 2))],   # float64 reductions
                     "ordering_effect": oe_heads,
                     "routed_mean": [_nanmean_or_none(adaptive[:, h]) for h in range(adaptive.shape[1])]},
        "segments": segments,
        "per_segment": segment_densities(kernel, adaptive, segments, tokens,
                                         counts=c, forced=forced, ntb=ntb, sink_q=sink_q),
        "shape_ok": True, "shape_why": None,
    })
    raw_bytes = None
    if _spec.get("raw"):
        arr = np.ascontiguousarray(c.numpy().astype(np.uint16))      # NTB <= 65535 by shape_check
        raw_bytes = arr.tobytes()
        row["raw"] = {"shape": list(c.shape), "dtype": "<u2", "order": "C"}
        key = (digest, int(batch), int(heads), ntb)
        if key not in _estimates_logged:
            _estimates_logged.add(key)
            sched = row["schedule"]
            per_call = len(raw_bytes)
            blocks = settings.get("n_blocks")
            dense = len(settings.get("dense_blocks") or [])
            per_step = per_call * max(0, (blocks or 0) - dense) if blocks else None
            total = (per_step * sched["n_intervals"]) if (per_step and sched.get("n_intervals")) else None
            logging.info(
                f"{_LOG} raw sidecar at this geometry: {per_call / 1024:.0f} KiB per Sol call"
                + (f", up to {per_step / 2**20:.1f} MiB per Sol-active step" if per_step else "")
                + (f", about {total / 2**20:.0f} MiB if every step routed" if total else "")
                + "; an estimate from live NQ/H and the schedule, not a fixed cost")
    try:
        _write_row(row, raw_bytes)
    except OSError as exc:
        # Cannot append the error row either if the disk is the problem; say so on the log.
        logging.error(f"{_LOG} write failed: {exc}")
        raise SolObserveError(f"could not append the observation row: {exc}") from exc


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------

def read_rows(jsonl_path) -> list[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_raw(jsonl_path, row: dict) -> torch.Tensor:
    """The (B, H, NQ) int32 counts a call row points at, checksum verified."""
    ptr = row.get("raw")
    if not ptr or "offset" not in ptr:
        raise ValueError("row carries no raw pointer")
    path = Path(jsonl_path).parent / ptr["file"]
    with open(path, "rb") as f:
        f.seek(ptr["offset"])
        data = f.read(ptr["nbytes"])
    if len(data) != ptr["nbytes"]:
        raise ValueError("raw sidecar is shorter than the row's pointer; an unreferenced tail is fine, a short referenced block is not")
    if (zlib.crc32(data) & 0xFFFFFFFF) != ptr["crc32"]:
        raise ValueError("raw block CRC mismatch")
    arr = np.frombuffer(data, dtype=np.dtype(ptr["dtype"])).reshape(ptr["shape"])
    return torch.from_numpy(arr.astype(np.int32))
