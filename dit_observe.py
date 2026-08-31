"""Record what each quantised linear SEES, at every block and every step.

Tier 1 of `docs/open_experiments.md` #23. Every quantisation record in this
repo is a stored-WEIGHT distance, and `int8_convrot` is W8A8 -- the activation
is rotated online and quantised per TOKEN before the int8 GEMM, so the error a
module carries at run time is two roundings and we have only ever seen one.
This observes the side nobody has looked at.

## What it records, and why each field

Per `(block, kind, step)`, all reduced online -- nothing large is written:

  chan_absmax   [in_features]  per-input-channel |x| max. **This vector IS the
                SmoothQuant scale** (`docs/research/quant_levers.md`), and it
                is what a channel-permutation design needs. A median cannot
                substitute: Tier 0 recorded one and consequently cannot inform
                the permutation lever at all.
  chan_rms      [in_features]  per-channel RMS. With absmax it gives the
                per-channel outlier ratio, which is the quantity the Hadamard
                rotation exists to flatten.
  token_absmax  quantiles + a 64-bin histogram of the PER-ROW max. The runtime
                quantiser is dynamic per token, so this is the shape of the
                rounding problem the weight side cannot see. Quantiles rather
                than the raw vector because 104k floats x 200 cells x steps is
                not a reduction.
  x_norm/out_norm, rows, dtype, and the kernel that actually RAN.

## Three traps this file exists to not fall into

**`mlp.fc2.forward` is never called.** H3's MLP is
`comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`, so a patch on
`fc2.forward` records nothing while the other three kinds look complete. That
already cost this repo once -- `unmerged_blocks` silently dropped fc2 on
2026-08-30 -- and an external review caught the same design here before it ran.
fc2 is reached through an `MLP.forward` wrapper instead; see `nodes.py`.

**No full-tensor `.abs()` or `.float()`.** At production geometry fc2's input
is 104361 x 14336, about 2.79 GiB in bf16, and an `abs()` doubles it. The
previous observer OOM'd its two widest modules at exactly this point and
reported a clean-looking result for the third. Everything here reduces in row
chunks, in the tensor's own dtype, promoting only the small vectors.

**A short capture must not look complete.** `_shape_check` asserts
`blocks x kinds x steps` at WRITE time and names what is missing, because
nothing downstream knows what should have been there.

## Inert unless `H3_QUANT_OBSERVE` is set

    H3_QUANT_OBSERVE=/path/to/dir <comfy>/start.sh

The node is still required -- the env var alone installs nothing. Two gates
because this must not be reachable by opening a workflow, and because a node
left in a graph must not arm a later render.

**An instrumented run's TIMING is void.** The reductions and CPU copies do not
change deterministic tensor values, but they do change wall time and peak
pressure. Never quote a sampler time, a stage time, or peak VRAM from a run
with this armed; the `convrot_groupsize` timing arm #23 gates needs its own
uninstrumented process.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

#: Rows per reduction chunk. Bounds peak scratch to CHUNK x in_features rather
#: than rows x in_features -- 8192 x 14336 in bf16 is ~224 MiB against 2.79 GiB
#: for the whole thing, and the reduction is exact either way.
CHUNK_ROWS = 8192

#: Bins for the per-token amax histogram. Linear over the observed range, with
#: the range recorded, so a log view is recoverable offline.
N_TOKEN_BINS = 64

_rows: list[dict] = []
_failures: list[dict] = []
_dir: Path | None = None
#: Arming is its own flag; see `enabled`. A path spelling is not a state.
_armed: bool | None = None
_path: Path | None = None
#: Set once per forward by the outer patch. `PackedLayout.segments` is the
#: authority for row modalities and is NOT inferable from a module hook, which
#: is why the observer needs a second patch point at all.
_context: dict = {}


def enabled() -> bool:
    """Armed only by the environment, and `""` is not a directory.

    **Corrected 2026-08-31.** This read `return bool(str(_dir))` with `_dir`
    set to `Path("")` when the variable was unset -- and `str(Path(""))` is
    `"."`, not `""`, so the disabled state evaluated TRUE and the module armed
    itself against the current working directory. Found by
    `bench/check_quant_observe.py::inert_without_the_env_var`, written for the
    other observer, on the first run of a case that looked like a formality.

    The arming decision is now its own boolean rather than a property of a
    path's spelling, because that spelling is where the bug lived.
    """
    global _dir, _armed
    if _armed is None:
        d = os.environ.get("H3_QUANT_OBSERVE", "").strip()
        _armed = bool(d)
        _dir = Path(d) if d else None
    return _armed


def set_context(**kw) -> None:
    """What the module hooks cannot see: step, knot, sigma, segments.

    Called by the `diffusion_model.forward` patch. Kept as one dict rather
    than threaded through every hook so the two instruments cannot disagree
    about which step they are in -- the joint-capture contract's "one shared
    authority for indices and layout".
    """
    _context.update({k: v for k, v in kw.items() if v is not None})


def _channel_stats(x: torch.Tensor):
    """Per-channel absmax and sum-of-squares, in row chunks.

    Returns fp32 CPU vectors of length `in_features`. The accumulation is fp32
    because a sum over 104k rows in bf16 loses the tail; the OPERANDS stay in
    their own dtype, which is the distinction that keeps this inside memory.
    """
    n_in = x.shape[-1]
    amax = torch.zeros(n_in, dtype=torch.float32, device=x.device)
    sumsq = torch.zeros(n_in, dtype=torch.float32, device=x.device)
    tok = []
    for i in range(0, x.shape[0], CHUNK_ROWS):
        c = x[i:i + CHUNK_ROWS]
        a = c.abs()
        amax = torch.maximum(amax, a.amax(dim=0).float())
        sumsq += (c.float() ** 2).sum(dim=0)
        tok.append(a.amax(dim=1).float().cpu())
        del a
    rows = x.shape[0]
    rms = (sumsq / max(rows, 1)).sqrt()
    return amax.cpu(), rms.cpu(), torch.cat(tok) if tok else torch.empty(0)


def _token_summary(tok: torch.Tensor) -> dict:
    """Quantiles and a histogram of the per-token max. Small by construction."""
    if tok.numel() == 0:
        return {"n": 0}
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    quant = torch.quantile(tok, torch.tensor(qs, dtype=tok.dtype))
    lo, hi = float(tok.min()), float(tok.max())
    hist = torch.histc(tok, bins=N_TOKEN_BINS, min=lo, max=hi if hi > lo else lo + 1.0)
    return {
        "n": int(tok.numel()),
        "quantile_points": qs,
        "quantiles": [float(v) for v in quant],
        "mean": float(tok.mean()),
        "hist_range": [lo, hi],
        "hist": [int(v) for v in hist],
    }


def record(block: int, kind: str, x: torch.Tensor,
           out: torch.Tensor | None = None, kernel: str | None = None) -> None:
    """One `(block, kind, step)` observation of a linear's INPUT.

    Never raises into a render -- but never silently either. A swallowed
    failure is recorded in `failures` and reddens `_shape_check`, because the
    2026-08-30 observer's `except` produced a file indistinguishable from a
    working one.
    """
    if not enabled():
        return
    try:
        with torch.no_grad():
            t = x.detach()
            if t.ndim != 2:
                t = t.reshape(-1, t.shape[-1])
            amax, rms, tok = _channel_stats(t)
            row = {
                "block": int(block),
                "kind": kind,
                "step": _context.get("step"),
                "knot": _context.get("knot"),
                "sigma": _context.get("sigma"),
                "kernel": kernel,
                "rows": int(t.shape[0]),
                "in_features": int(t.shape[-1]),
                "dtype": str(t.dtype),
                "x_norm": float(torch.linalg.vector_norm(t.float())
                                if t.numel() < 2 ** 24 else
                                torch.linalg.vector_norm(rms) * (t.shape[0] ** 0.5)),
                "chan_absmax": [float(v) for v in amax],
                "chan_rms": [float(v) for v in rms],
                "token_absmax": _token_summary(tok),
            }
            if out is not None:
                o = out.detach()
                o = o if o.ndim == 2 else o.reshape(-1, o.shape[-1])
                row["out_features"] = int(o.shape[-1])
                row["out_norm"] = float(
                    torch.linalg.vector_norm(
                        torch.linalg.vector_norm(o, dim=1).float()))
            _rows.append(row)
    except Exception as exc:
        _failures.append({"block": int(block), "kind": kind,
                          "step": _context.get("step"),
                          "error": f"{type(exc).__name__}: {exc}"[:200]})


def _shape_check(expect_blocks: int = 0, expect_kinds: int = 4) -> dict:
    """Assert `blocks x kinds x steps` at write time, not at read time.

    `expect_blocks` is passed by the caller rather than inferred: inferring it
    from what reported makes a capture that lost a whole block indistinguishable
    from one that was never asked for it. Zero means "infer, and say so".
    """
    if not _rows:
        return {"ok": None, "why": "nothing recorded"}
    blocks = sorted({r["block"] for r in _rows})
    kinds = sorted({r["kind"] for r in _rows})
    steps = sorted({r["step"] for r in _rows if r["step"] is not None})
    n_steps = max(len(steps), 1)
    n_blocks = expect_blocks or len(blocks)
    want = n_blocks * expect_kinds * n_steps
    seen = {(r["block"], r["kind"], r["step"]) for r in _rows}
    short = [{"block": b, "kind": k}
             for b in blocks for k in kinds
             if sum(1 for s in steps if (b, k, s) in seen) != n_steps]
    return {
        "ok": (len(_rows) == want and not short and not _failures
               and len(kinds) == expect_kinds),
        "rows_recorded": len(_rows), "rows_expected": want,
        "blocks_seen": len(blocks), "blocks_expected": n_blocks,
        "blocks_inferred": not expect_blocks,
        "kinds_seen": kinds, "kinds_expected": expect_kinds,
        "steps": n_steps,
        "incomplete": short[:20], "n_incomplete": len(short),
        "why": ("`ok` false means this capture is SHORT and is NOT a complete "
                "result for the cells that did report. The specific failure "
                "this asserts against: `mlp.fc2` is unreachable through "
                "`fc2.forward`, so a capture missing it looks like a clean "
                "three-kind result. `kinds_expected` is 4 and is not inferred."),
    }


def flush(meta: dict | None = None, expect_blocks: int = 0) -> str | None:
    """Write everything observed so far. Idempotent, does NOT clear.

    Once per forward rather than once per render, for `pdd_observe.py`'s
    reason: nothing in this pack sees a render end, and a run interrupted
    part-way still leaves its completed steps on disk.
    """
    global _path
    if not enabled() or not _rows or _dir is None:
        return None
    _dir.mkdir(parents=True, exist_ok=True)
    if _path is None:
        _path = _dir / f"dit_observe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    _path.write_text(json.dumps({
        "produced_by": "dit_observe.py",
        "what": ("per (block, kind, step): what the quantised linear's INPUT "
                 "looks like -- per-channel absmax and RMS, and the per-token "
                 "max distribution the dynamic activation quantiser acts on"),
        "is_not": ("a weight measurement, and not yet an error measurement. "
                   "This is the ACTIVATION side that every existing quant "
                   "record in this repo is blind to. The error decomposition "
                   "is an offline pass over these plus the weights; see "
                   "docs/open_experiments.md #23"),
        "timing_void": ("wall time, stage time and peak VRAM from a run with "
                        "this armed are meaningless -- reductions and CPU "
                        "copies change them. Values are unaffected."),
        "chunk_rows": CHUNK_ROWS,
        "n_token_bins": N_TOKEN_BINS,
        "segments": _context.get("segments"),
        "segments_note": ("PackedLayout.segments as (start, end, kind), the "
                          "authority for row modalities. Recorded rather than "
                          "inferred from positions -- a capture that reduces a "
                          "grouping it did not record has lost its shape."),
        "capture_id": _context.get("capture_id"),
        "meta": meta or {},
        "shape_check": _shape_check(expect_blocks),
        "observations": _rows,
        "failures": _failures,
        "failure_note": ("non-empty means the capture LOST observations and "
                         "`shape_check.ok` is false. Not a complete result for "
                         "the cells that survived."),
    }, indent=2) + "\n")
    return str(_path)
