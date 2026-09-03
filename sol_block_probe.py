"""The Sol-versus-fallback probe: per-block, per-head, per-segment error of
Sol's output against the shipped fallback on identical q/k/v, recorded as
summaries, no tensors, on a live render.

**Implemented 2026-09-03**, replacing the scaffold this file was from
2026-08-16 (its history and the specification it was built to are below).
Environment-gated like every capture instrument here:

    H3_SOL_PROBE="dir=$H3_CAPTURE_ROOT/<name>[,trajectory=sol|sage][,capture=<label>]" <comfy>/start.sh

`trajectory=sol` (default) returns Sol's output and computes the fallback as
the counterfactual -- the shipping measurement; `trajectory=sage` returns the
fallback's output -- the control. The record is `sol_probe_<stamp>.jsonl` in
`dir`; `bench/check_sol_probe.py --record` reads it, validates every
invariant and prints the per-block table, and `--controls` proves the
behaviour on fixtures. Timings from an armed render are void: every Sol call
also ran the fallback.

## History: the 2026-08-16 scaffold this replaced

The original plan (Track A2 of a since-retired plan) left every function a
stub so the node id and input order would be decided deliberately. What
follows is that scaffold's reasoning, kept because the specification at the
end supersedes it point by point and the reader should see which points.

## Why this exists

The Triton pack carried `SolAttnBlockProbe`, which computed every attention
call both sparse and dense and logged per-block relative error worst-first. It
is the instrument for choosing a `dense_blocks` list, and it was the last live
reason that pack existed here.

**The pack was deleted on 2026-08-16 before this was written**, so the port
target is upstream at a pinned commit rather than a local path:
`https://github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`. Fetch it into a
scratch dir to read; do not reinstall it into `custom_nodes/`.

`docs/SOLATTN.md` names the thing it is for: at `tau` above roughly 1.5 a small
persistent object can dissolve partway through a clip. The stated fix is to
force the most approximation-sensitive blocks dense, and `SOL_ARTIFACT_INSURANCE`
in `workflows/h3_config.py:280` is a guessed starting set that is deliberately
not wired, pending a probe run nobody has done.

## What is being ported, and what is not

**Ported unchanged**, from `kijai/ComfyUI-SolAttn_triton@842c4ea:__init__.py:323-342`:
the wrapper calls the installed override for the sparse result, calls `func`
for the dense reference, records the pair, and **returns the dense result**.
That last detail is load-bearing and is easy to lose: returning the sparse
result would let each block's error compound into every later block's number.

**Superseded 2026-09-03 as the only mode (see the specification section
below):** returning the fallback is now `trajectory=sage`, the control; the
shipping measurement is `trajectory=sol`, which RETURNS SOL so the cells are
the production population, upstream compounding included on purpose. The
record says which mode wrote it and the two are never pooled.

**Not ported:** anything Triton. `make_probe_override` contains none -- it wraps
whatever override is installed, so it works against the CUDA node in principle.

## Three additions, each from something measured since the original was written

1. **Routed density per block.** ~~Nothing in this repo measures it.~~
   **Measured since 2026-09-01:** the fork's `blk_cnt` out-parameter reports
   the routed count per (batch, head, query block) and `sol_observe.py`
   records it per call; the instrument reads the same buffer. The rest of
   this item describes the state before that.
   `sol_attn_stats()` (`vendor/sol_attn_minimax.py:102-104`) counts dispatches,
   not blocks. Density is what decides whether Sol's exact branch is a thin
   slice or most of the work, which is the open question behind the 16-bit PV
   decision (plan Track B, gate B0a).
2. **Report the tail, not only the mean.** `docs/morton.md` found the p10
   mattered where the mean did not: `2d_frame`'s p10 centroid fidelity sits
   below raster at blocks 0 and 49 while its mean sits above.
3. **Per-segment error.** H3 packs text, audio, references and video into one
   sequence. The audio rows are thin -- `docs/SOLATTN.md` calls them the shape a
   block-sparse router drops first -- and a whole-tensor error number cannot see
   them. Split by the `PackedLayout` spans the compose hooks already publish.

## THE RISK THAT MUST BE RUN, NOT REASONED

**Unverified: whether a probe wrapping `optimized_attention_override` sees the
CUDA node's DiT calls at all.**

Our sage node object-patches `diffusion_model.blocks.{i}.attn.forward`, which
deletes the `optimized_attention` call site for all 50 DiT blocks. Sol's
`_compose_module_patch` gates that patch and calls `stock()` inside the sigma
window, which should reach the override. **Should.** That is an inference from
source, and `docs/SOLATTN.md`'s Ordering section documents two separate nodes
that look like they compose and do not. The same shape has cost this repo
several confident wrong claims.

So the first thing this node needs is not a feature, it is a control: **assert
the number of distinct blocks seen equals the number the compose hook reports
patching.** An empty or short list must be a loud failure, never a quiet
"no errors found" -- a probe that measures nothing looks exactly like a kernel
with no error, and it is most convincing when it is emptiest.

## Node id and input order are permanent

CLAUDE.md's one rule that matters. `node_id` is baked into every saved graph's
`type` field and inputs are matched positionally. **Append only.** No shipped
graph should ever wire this -- it is a diagnostic that roughly doubles render
cost, since every call is computed twice.

## Specification as of 2026-09-03 (Codex's, adopted by the owner)

Two modes, explicit record fields `trajectory` and `returned_backend`:
`trajectory=sol` computes Sol and Sage on identical q/k/v and RETURNS SOL
(the production population; upstream Sol error in later inputs is
intentional); `trajectory=sage` computes both and RETURNS SAGE (the
fallback trajectory, which is what a dense block runs; never call it
"dense exact"). The two renders' cells are never aggregated. The retained
2026-09-03 Base16 capture is the validation fixture: the instrument's
summaries must reproduce its cells within a stated tolerance before a new
render is trusted.

Per measured cell, record: identity (capture id, prompt id, render index,
schedule occurrence and index, sigma, schedule length, block, actual
route, compare status and reason); shape and layout (B/H/T/D and the
authoritative `PackedLayout.segments`, published by a neutral helper that
Sol or `h3_capture.py` can arm -- not derived from geometry);
configuration (tau or top-k, tail, resolved sink and sink-query ranges,
ordering, window, min_tokens, dense_blocks, requested Sage mode, the Sage
kernel actually dispatched, both kernel build ids); cost (blk_cnt, kernel
density, pair-weighted density, with their different denominators kept);
Sol-versus-the-actual-shipped-Sage error on identical q/k/v (whole-call
relative L2 and cosine; absolute-difference RMS and reference RMS; per-head
numerator, denominator, relative L2 and cosine; per-segment aggregates;
per-head and per-segment row-distribution summaries: count, mean, p50,
p90, p99, max). A zero denominator yields null with numerator and
denominator retained.

Behaviour: compare only calls that routed through Sol, recording skip
reasons elsewhere; confirm the counterfactual is the configured Sage auto
kernel on this box, not generic dense attention; write no q/k/v; join to
`sol_observe` by capture id, prompt, schedule occurrence and block; treat
the render's timing as void; unarmed, add no allocation, kernel call, copy
or synchronisation beyond the branch. Memory: do not clone production
q/k/v blindly; if the Sage counterfactual is head-chunked, first prove on
the fixture that it matches the 56-head call within a stated tolerance.

Controls before trusting a record: armed and unarmed outputs bitwise equal
to canonical Sol; unarmed produces no records and no extra Sage call;
Sage dispatch telemetry observed, a stock-attention substitution fails;
identical and deliberately perturbed fixtures validate every metric;
metrics reproduce the retained capture's cells; completeness -- every
expected active schedule occurrence x blocks 0-49 exactly once; segments
contiguous over [0, T), wrong or shuffled boundaries fail; duplicate or
missing blocks, mixed prompt ids, NaN/Inf and wrong schedule populations
fail; PDD completeness derived from the PDD node's actual SIGMAS, never a
nominal step count.

The scaffold's original note -- return the DENSE result so a block's error
cannot compound into later blocks -- is the `trajectory=sage` mode above,
kept as a named mode rather than the only one.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_LOG = "[h3-sol-probe]"
SCHEMA = 1
_IMPLEMENTED = True

# ---------------------------------------------------------------------------
# Arming. Environment only, read once, cached: the unarmed path is one bool.
#
#   H3_SOL_PROBE="dir=/path[,trajectory=sol|sage][,capture=<label>]"
#
# trajectory=sol   compute Sol and the fallback on identical q/k/v, RETURN SOL
#                  (the shipping measurement; upstream Sol error compounds
#                  into later blocks on purpose -- that is the production
#                  population)
# trajectory=sage  compute both, RETURN THE FALLBACK (the control: no block's
#                  input carries an earlier Sol approximation)
# capture=<label>  a free label joining this record to a capture or a session
# ---------------------------------------------------------------------------

_armed: bool | None = None
_spec: dict = {}
_raw_spec: str = ""


def _parse(spec: str) -> dict:
    out = {"dir": None, "trajectory": "sol", "capture": None}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = os.path.expanduser(val)
        elif key == "trajectory" and val:
            out["trajectory"] = val.lower()
        elif key == "capture" and val:
            out["capture"] = val
    return out


def arm(spec: str | None) -> bool:
    """Set the arming state from a spec string; the server reads the
    environment through this once, tests call it directly."""
    global _armed, _spec, _raw_spec
    _raw_spec = spec or ""
    parsed = _parse(_raw_spec) if _raw_spec else _parse("")
    if parsed["trajectory"] not in ("sol", "sage"):
        print(f"{_LOG} H3_SOL_PROBE trajectory must be sol or sage, got {parsed['trajectory']!r}; probe disabled")
        parsed["dir"] = None
    _armed = bool(_raw_spec) and bool(parsed["dir"])
    if _raw_spec and not parsed["dir"]:
        print(f"{_LOG} H3_SOL_PROBE set but no dir=; probe disabled")
    _spec = parsed
    _writer_reset()
    if _armed:
        print(f"{_LOG} ARMED: dir={parsed['dir']} trajectory={parsed['trajectory']} "
              f"capture={parsed['capture']}; every Sol call also runs the fallback and "
              f"timings from this render are void")
    return _armed


def enabled() -> bool:
    global _armed
    if _armed is None:
        arm(os.environ.get("H3_SOL_PROBE", ""))
    return bool(_armed)


def spec() -> dict:
    enabled()
    return dict(_spec, spec=_raw_spec)


# ---------------------------------------------------------------------------
# Writer: one lock, one append-only jsonl, header once, a render row per prompt
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_jsonl = None
_path: str | None = None
_seq = 0
_header_written = False
_renders_seen: dict = {}


def _writer_reset() -> None:
    global _jsonl, _path, _seq, _header_written
    try:
        if _jsonl is not None:
            _jsonl.close()
    except OSError:
        pass
    _jsonl = None
    _path = None
    _seq = 0
    _header_written = False
    _renders_seen.clear()


def path() -> str | None:
    return _path


def _builds() -> dict:
    """Both kernel builds under the comparison, read from the process."""
    out = {"comfy_kitchen": None, "sageattention": None, "sageattention_path": None,
           "sageattention_git_head": None, "pack_commit": None}
    try:
        import importlib.metadata as md
        for name in ("comfy-kitchen", "comfy_kitchen"):
            try:
                out["comfy_kitchen"] = md.version(name); break
            except md.PackageNotFoundError:
                continue
        try:
            out["sageattention"] = md.version("sageattention")
        except md.PackageNotFoundError:
            pass
    except Exception:                                  # noqa: BLE001
        pass
    try:
        import subprocess
        import sageattention as _sa
        d = os.path.dirname(os.path.abspath(_sa.__file__))
        out["sageattention_path"] = d
        r = subprocess.run(["git", "-C", d, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5)
        out["sageattention_git_head"] = r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                  # noqa: BLE001
        pass
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5)
        out["pack_commit"] = r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                  # noqa: BLE001
        pass
    return out


def _header() -> dict:
    return {"kind": "header", "schema": SCHEMA, "spec": _raw_spec, "trajectory": _spec.get("trajectory"),
            "capture": _spec.get("capture"), "builds": _builds(), "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metric": {
                "rel_l2": "sqrt(sum |sol - ref|^2 / sum |ref|^2) over the named scope, float32 inputs, float64 sums",
                "cos": "sum(sol . ref) / (|sol| |ref|) over the named scope",
                "diff_rms": "sqrt(mean |sol - ref|^2)", "ref_rms": "sqrt(mean |ref|^2)",
                "rows": "per-row (one head, one token) relative L2 = |sol_row - ref_row| / |ref_row|, summarised as count, mean, p50, p90, p99, max; rows with a zero reference norm are excluded and counted",
                "null_rule": "a zero denominator yields null with numerator and denominator retained",
                "reference": "the CHAINED fallback the Sol override would have run for this call (the shipped sage override on the canonical graphs), on the identical q/k/v; not exact attention"},
            "note": "timings from an armed render are void: every Sol call also ran the fallback"}


def _open() -> None:
    global _jsonl, _path
    if _jsonl is not None:
        return
    d = Path(_spec["dir"]); d.mkdir(parents=True, exist_ok=True)
    _path = str(d / f"sol_probe_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    _jsonl = open(_path, "a", encoding="utf-8")


def _write(row: dict) -> None:
    global _seq, _header_written
    with _lock:
        _open()
        assert _jsonl is not None
        if not _header_written:
            _jsonl.write(json.dumps(_header()) + "\n")
            _header_written = True
        _seq += 1
        _jsonl.write(json.dumps(dict(row, seq=_seq, schema=SCHEMA)) + "\n")
        _jsonl.flush()


def _ensure_render(prompt_id) -> None:
    """A render row the first time a prompt id is seen: the graph's hash, the
    shipped file it matches, and what was rendered (bank id, prompt hash,
    length, canvas, seed), through the route recorder's own helpers so the
    two records describe a render the same way."""
    if prompt_id is None or prompt_id in _renders_seen:
        return
    _renders_seen[prompt_id] = len(_renders_seen)
    row = {"kind": "render", "prompt_id": prompt_id, "process_render_index": _renders_seen[prompt_id],
           "trajectory": _spec.get("trajectory"), "graph_sha256": None, "workflow_file": None, "rendered": None}
    try:
        from . import sol_observe as _obs
    except ImportError:
        import sol_observe as _obs  # type: ignore
    try:
        graph, why = _obs._running_prompt(prompt_id)
        row["match"] = why
        if graph is not None:
            sha = _obs.graph_sha256(graph)
            row.update({"graph_sha256": sha, "workflow_file": _obs._shipped_graph_hashes().get(sha),
                        "rendered": _obs._describe_prompt(graph)})
    except Exception as exc:                          # noqa: BLE001 -- identity, not the render
        row["match"] = f"could not read the running prompt: {exc}"
    _write(row)


# ---------------------------------------------------------------------------
# Metrics. Everything streams per head so the temporaries are one head's
# worth (about 50 MiB at 104k tokens), never a whole-call float32 copy.
# ---------------------------------------------------------------------------

def _bhnd(x: torch.Tensor, heads: int, skip_output_reshape: bool) -> torch.Tensor:
    """(B, H, N, D) view of an override's output in either layout."""
    if skip_output_reshape:
        return x                                              # already BHND
    b, n, hd = x.shape
    return x.view(b, n, heads, hd // heads).transpose(1, 2)   # BHND view


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0.0 else num / den


def _summ(rows: torch.Tensor) -> dict:
    """count, mean, p50, p90, p99, max of a 1-D tensor of per-row relative
    errors; None-valued when empty."""
    if rows.numel() == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    r = rows.double()
    return {"count": int(r.numel()), "mean": float(r.mean()), "p50": float(r.quantile(0.5)),
            "p90": float(r.quantile(0.9)), "p99": float(r.quantile(0.99)), "max": float(r.max())}


def _segment_spans(segments, tokens: int) -> list[tuple[str, int, int]]:
    """[(kind, start, end)] clipped to [0, tokens); empty when no table."""
    out = []
    for a, b, kind in (segments or []):
        a, b = max(0, int(a)), min(int(tokens), int(b))
        if b > a:
            out.append((str(kind), a, b))
    return out


def metrics(sol: torch.Tensor, ref: torch.Tensor, *, heads: int, skip_output_reshape: bool,
            segments=None) -> dict:
    """Sol-versus-reference error on one call, whole-call, per head, per
    segment, with per-head and per-segment row distributions. `sol` and `ref`
    are the two override outputs in the same layout."""
    a = _bhnd(sol, heads, skip_output_reshape)
    r = _bhnd(ref, heads, skip_output_reshape)
    if a.shape != r.shape:
        raise ValueError(f"probe: sol {tuple(a.shape)} and reference {tuple(r.shape)} differ in shape")
    B, H, N, D = a.shape
    spans = _segment_spans(segments, N)
    # accumulators (float64 python floats)
    diff2 = ref2 = dot = sol2 = 0.0
    per_head = []
    seg_acc = {kind: {"diff2": 0.0, "ref2": 0.0, "dot": 0.0, "sol2": 0.0, "rows": []} for kind, _, _ in spans}
    zero_rows = 0
    for h in range(H):
        x = a[:, h].float()                       # (B, N, D)
        y = r[:, h].float()
        d = x - y
        hd2 = float((d * d).sum(dtype=torch.float64)); hr2 = float((y * y).sum(dtype=torch.float64))
        hdot = float((x * y).sum(dtype=torch.float64)); hs2 = float((x * x).sum(dtype=torch.float64))
        diff2 += hd2; ref2 += hr2; dot += hdot; sol2 += hs2
        row_d = d.norm(dim=-1)                    # (B, N)
        row_r = y.norm(dim=-1)
        live = row_r > 0
        zero_rows += int((~live).sum())
        rel_rows = torch.where(live, row_d / row_r.clamp_min(1e-30), torch.zeros_like(row_d))
        per_head.append({
            "head": h, "numerator": hd2, "denominator": hr2,
            "rel_l2": _ratio(math.sqrt(hd2), math.sqrt(hr2)) if hr2 > 0 else None,
            "cos": _ratio(hdot, math.sqrt(hs2 * hr2)) if hs2 > 0 and hr2 > 0 else None,
            "rows": _summ(rel_rows[live]),
        })
        for kind, s0, s1 in spans:
            xs, ys, ds = x[:, s0:s1], y[:, s0:s1], d[:, s0:s1]
            acc = seg_acc[kind]
            acc["diff2"] += float((ds * ds).sum(dtype=torch.float64))
            acc["ref2"] += float((ys * ys).sum(dtype=torch.float64))
            acc["dot"] += float((xs * ys).sum(dtype=torch.float64))
            acc["sol2"] += float((xs * xs).sum(dtype=torch.float64))
            lv = live[:, s0:s1]
            acc["rows"].append(rel_rows[:, s0:s1][lv].detach().cpu())
        del x, y, d, row_d, row_r, rel_rows
    per_segment = []
    for kind, s0, s1 in spans:
        acc = seg_acc[kind]
        rows = torch.cat(acc["rows"]) if acc["rows"] else torch.zeros(0)
        per_segment.append({
            "kind": kind, "start": s0, "end": s1, "numerator": acc["diff2"], "denominator": acc["ref2"],
            "rel_l2": _ratio(math.sqrt(acc["diff2"]), math.sqrt(acc["ref2"])) if acc["ref2"] > 0 else None,
            "cos": (_ratio(acc["dot"], math.sqrt(acc["sol2"] * acc["ref2"]))
                    if acc["sol2"] > 0 and acc["ref2"] > 0 else None),
            "rows": _summ(rows),
        })
    n_elem = float(B * H * N * D)
    return {
        "shape": {"B": B, "H": H, "N": N, "D": D},
        "whole": {"numerator": diff2, "denominator": ref2,
                  "rel_l2": _ratio(math.sqrt(diff2), math.sqrt(ref2)) if ref2 > 0 else None,
                  "cos": _ratio(dot, math.sqrt(sol2 * ref2)) if sol2 > 0 and ref2 > 0 else None,
                  "diff_rms": math.sqrt(diff2 / n_elem), "ref_rms": math.sqrt(ref2 / n_elem),
                  "zero_reference_rows": zero_rows},
        "per_head": per_head,
        "per_segment": per_segment,
        "segments_recorded": bool(spans),
        "finite": bool(math.isfinite(diff2) and math.isfinite(ref2) and math.isfinite(dot)),
    }


# ---------------------------------------------------------------------------
# The two entries the Sol override calls
# ---------------------------------------------------------------------------

def _sage_counts():
    try:
        import sageattention as _sa
        c = _sa.get_dispatch_counts()
        return dict(c) if isinstance(c, dict) else c
    except Exception:                                  # noqa: BLE001
        return None


def _sage_last():
    try:
        import sageattention as _sa
        return _sa.get_last_dispatched_kernel()
    except Exception:                                  # noqa: BLE001
        return None


def _count_total(c) -> int | None:
    if c is None:
        return None
    if isinstance(c, dict):
        return int(sum(int(v) for v in c.values() if isinstance(v, (int, float))))
    try:
        return int(c)
    except (TypeError, ValueError):
        return None


def _identity_bits(options, settings, block, block_tau, tokens, batch, heads, sink, sink_q,
                   tail, topk_ratio, min_tokens) -> dict:
    try:
        from . import sol_observe as _obs
    except ImportError:
        import sol_observe as _obs  # type: ignore
    identity = _obs._identity()
    sigmas = options.get("sigmas") if isinstance(options, dict) else None
    sigma = None
    if sigmas is not None:
        try:
            sigma = float(sigmas[0]) if getattr(sigmas, "ndim", 1) else float(sigmas)
        except (TypeError, IndexError, ValueError):
            sigma = None
    ntb = (int(tokens) + 63) // 64
    return {
        **identity, "capture": _spec.get("capture"), "trajectory": _spec.get("trajectory"),
        "config": _obs.config_digest(settings), "settings": settings,
        "block": None if block is None else int(block),
        "sigma": sigma,
        "schedule": _obs.schedule_index(sigma, options.get("sample_sigmas") if isinstance(options, dict) else None),
        "cond_or_uncond": _obs._cond_or_uncond(options),
        "B": int(batch), "H": int(heads), "T": int(tokens), "NTB": ntb,
        "selection": ({"mode": "topk", "topk_ratio": float(topk_ratio)} if topk_ratio
                      else {"mode": "tau", "tau": None if block_tau is None else float(block_tau)}),
        "tail": bool(tail), "sink_blocks": [int(sink[0]), int(sink[1])],
        "sink_q": [int(sink_q[0]), int(sink_q[1])], "min_tokens": int(min_tokens),
        "segments": _obs._segments(options),
    }


def skip(*, route: str, reason, options, settings, block, block_tau, tokens, batch, heads,
         sink, sink_q, tail, topk_ratio, min_tokens) -> None:
    """A call that did not route through Sol is recorded as skipped, with its
    reason, so completeness can be checked against the schedule."""
    if not enabled():
        return
    bits = _identity_bits(options, settings, block, block_tau, tokens, batch, heads,
                          sink, sink_q, tail, topk_ratio, min_tokens)
    _ensure_render(bits.get("prompt_id"))
    _write({"kind": "skip", "t_wall": time.time(), **bits, "route": route, "reason": reason})


def compare(out: torch.Tensor, dense_fn, *, skip_output_reshape: bool, options, settings, block,
            block_tau, tokens, batch, heads, sink, sink_q, tail, topk_ratio, min_tokens,
            counts=None) -> torch.Tensor:
    """Called by the Sol override after a call it took. Runs the chained
    fallback on the same q/k/v, records the comparison, and returns the
    output the configured trajectory names. `counts` is the kernel's blk_cnt
    tensor when the route recorder is armed too (cost lives in that record;
    here only its per-head mean is copied for convenience)."""
    if not enabled():
        return out
    bits = _identity_bits(options, settings, block, block_tau, tokens, batch, heads,
                          sink, sink_q, tail, topk_ratio, min_tokens)
    _ensure_render(bits.get("prompt_id"))
    before = _sage_counts()
    t0 = time.time()
    status, why, ref = "compared", None, None
    try:
        ref = dense_fn()
    except torch.cuda.OutOfMemoryError as exc:
        status, why = "oom", f"fallback OOM: {exc}"[:200]
        torch.cuda.empty_cache()
    except Exception as exc:                          # noqa: BLE001 -- recorded, never swallowed silently
        status, why = "fallback_error", f"{type(exc).__name__}: {exc}"[:200]
    after = _sage_counts()
    b0, a0 = _count_total(before), _count_total(after)
    dispatched = (a0 - b0) if (a0 is not None and b0 is not None) else None
    telemetry = {"sage_dispatch_delta": dispatched, "sage_last_kernel": _sage_last(),
                 "sage_counts_after": after}
    row = {"kind": "cell", "t_wall": time.time(), **bits, "route": "sol",
           "compare_status": status, "compare_reason": why, "counterfactual": telemetry,
           "fallback_seconds": time.time() - t0, "returned_backend": None, "metrics": None,
           "blk_cnt_per_head_mean": None}
    if counts is not None:
        try:
            row["blk_cnt_per_head_mean"] = [float(x) for x in counts.detach().to("cpu").double().mean(dim=(0, 2))]
        except Exception:                              # noqa: BLE001
            pass
    if ref is not None:
        try:
            row["metrics"] = metrics(out, ref, heads=heads, skip_output_reshape=skip_output_reshape,
                                     segments=bits.get("segments"))
            if not row["metrics"]["finite"]:
                row["compare_status"], row["compare_reason"] = "non_finite", "NaN or Inf in the sums"
        except Exception as exc:                      # noqa: BLE001
            row["compare_status"], row["compare_reason"] = "metric_error", f"{type(exc).__name__}: {exc}"[:200]
    if dispatched == 0 and ref is not None:
        # the reference did not go through sage: a stock-attention substitution,
        # or a sage build without the counter -- either way not the shipped
        # fallback, and the row says so rather than describing the wrong thing
        row["compare_status"], row["compare_reason"] = "reference_not_sage", "sage dispatch counter did not move"
    trajectory = _spec.get("trajectory", "sol")
    if trajectory == "sage":
        if ref is None:
            _write({**row, "returned_backend": None})
            raise RuntimeError(f"{_LOG} trajectory=sage but the fallback failed ({why}); cannot continue honestly")
        row["returned_backend"] = "sage"
        _write(row)
        del out
        return ref
    row["returned_backend"] = "sol"
    _write(row)
    del ref
    return out


def summarize() -> dict | None:
    """Where the record went; the offline reader does the per-block table."""
    return {"path": _path, "rows": _seq, "renders": len(_renders_seen)} if _path else None
