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
  call      one per override call, every route, not only Sol. Identity, step,
            block, shape, selection, sinks, the route actually taken, and for
            a Sol route the density summaries and the raw pointer.
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

`NTB = ceil(T / 64)` key blocks. Two densities, the same two
`bench/analyze_routing.py` prints, so the vocabulary does not fork:

  kernel_density   cnt / NTB over every query block. Forced pairs INCLUDED:
                   what the exact stage actually walked. The cost number.
  routed_density   (cnt - forced) / (NTB - forced) over query blocks NOT in
                   sink_q, where forced = |sink range| + |{q-1, q, q+1} that
                   exist and are not sink|, as a set. The ADAPTIVE share --
                   what tau or top-k selected beyond the mandatory pairs.
                   Null where the denominator is zero.

sink_q rows are NTB by construction and are counted, never averaged into the
routed figure. Segment figures are QUERY-segment densities, overlap-weighted:
a 64-row block crossing a segment boundary contributes to each side by the
rows it holds there. The full segment table is stored on the row so the raw
counts can be re-reduced any other way later.

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
            "kernel_density": "cnt / NTB over every query block; sink and diagonal pairs INCLUDED",
            "routed_density": ("(cnt - forced) / (NTB - forced) over query blocks not in sink_q, "
                               "forced = |sink| + |{q-1,q,q+1} existing and not sink| as a set; "
                               "null where the denominator is zero"),
            "per_segment": "overlap-weighted QUERY-segment means of the two above",
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
    """(kernel (B,H,NQ), routed (B,H,NQ) with NaN where undefined)."""
    c = counts.double()
    kernel = c / ntb
    f = forced.double().view(1, 1, -1)
    den = ntb - f
    routed = torch.where(den > 0, (c - f) / den.clamp_min(1), torch.full_like(c, float("nan")))
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    if q1 > q0:
        routed[:, :, q0:q1] = float("nan")
    return kernel, routed


def segment_densities(kernel: torch.Tensor, routed: torch.Tensor, segments, tokens: int) -> list | None:
    """Overlap-weighted QUERY-segment means. A 64-row block straddling a
    boundary counts toward each side by the rows it holds there. Preserves
    every occurrence and kind in the table, in order."""
    if not segments:
        return None
    out = []
    nq = kernel.shape[-1]
    for entry in segments:
        a, b, kind = int(entry[0]), int(entry[1]), str(entry[2])
        b = min(b, int(tokens))
        if b <= a:
            out.append({"kind": kind, "start": a, "stop": b, "rows": 0,
                        "query_blocks": None, "kernel": None, "routed": None})
            continue
        q0, q1 = a // BLOCK, (b - 1) // BLOCK
        q1 = min(q1, nq - 1)
        qs = torch.arange(q0, q1 + 1)
        w = (torch.minimum(torch.tensor(b), (qs + 1) * BLOCK)
             - torch.maximum(torch.tensor(a), qs * BLOCK)).double()     # rows of this segment per block
        ks = kernel[:, :, q0:q1 + 1]
        rs = routed[:, :, q0:q1 + 1]
        wk = w.view(1, 1, -1).expand_as(ks)
        kd = float((ks * wk).sum() / wk.sum())
        mask = ~torch.isnan(rs)
        wr = (wk * mask.float()).sum()
        rd = float((torch.nan_to_num(rs) * wk).sum() / wr) if float(wr) > 0 else None
        out.append({"kind": kind, "start": a, "stop": b, "rows": b - a,
                    "query_blocks": [q0, q1 + 1], "kernel": kd, "routed": rd})
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
           sink, sink_q, tail: bool, topk_ratio: float, min_tokens: int) -> None:
    """One call row. Raises SolObserveError rather than writing a row it
    cannot stand behind; the caller must not catch that inside the kernel
    fallback."""
    if not enabled():
        return
    digest = _ensure_config(settings)
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
        **_identity(),
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
        "route": route, "reason": reason,
    }
    if route != "sol":
        _write_row(row)
        return
    if counts is None:
        _write_row({**row, "kind": "error", "stage": "record",
                    "message": "route sol with no count tensor"})
        raise SolObserveError("route sol recorded with no blk_cnt tensor")

    # The one synchronization an armed render pays: the copy to host.
    c = counts.detach().to("cpu")
    ok, why, forced, _ = shape_check(c, batch, heads, tokens, sink, sink_q)
    if not ok:
        _write_row({**row, "kind": "error", "stage": "shape_check", "message": why,
                    "counts_shape": list(c.shape), "counts_dtype": str(c.dtype)})
        raise SolObserveError(f"blk_cnt failed its shape check: {why}")

    kernel, routed = densities(c, forced, ntb, sink_q)
    q0 = max(0, min(int(sink_q[0]), ntb))
    q1 = max(q0, min(int(sink_q[1]), ntb))
    segments = _segments(options)
    row.update({
        "forced": {"sink": int(forced.min()) if ntb else 0,
                   "sink_range_clamped": [max(0, min(int(sink[0]), ntb)), max(0, min(int(sink[1]), ntb))],
                   "diag_min": int((forced[[i for i in range(ntb) if not (q0 <= i < q1)]]
                                    - forced.min()).min()) if q1 - q0 < ntb else None,
                   "diag_max": int((forced[[i for i in range(ntb) if not (q0 <= i < q1)]]
                                    - forced.min()).max()) if q1 - q0 < ntb else None},
        "sink_q_rows": q1 - q0,
        "kernel_density": _stats(kernel.flatten()),
        "routed_density": _stats(routed.flatten()),
        "per_head": {"kernel_mean": [float(x) for x in kernel.mean(dim=(0, 2))],   # float64 reductions
                     "routed_mean": [_nanmean_or_none(routed[:, h]) for h in range(routed.shape[1])]},
        "segments": segments,
        "per_segment": segment_densities(kernel, routed, segments, tokens),
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
