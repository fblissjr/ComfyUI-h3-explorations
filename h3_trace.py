"""Record the shape of every H3 attention call, for someone else's bench.

The sage fork's accuracy and speed benches cover LTX 2.3 and Z-Image shapes
and have **zero H3 coverage**, so every rtol and every speed number either
project cites is measured on the wrong workload for this one. Their own
discipline forbids adding bench shapes without profiling a real trace first.
This produces that trace.

Deliberately env-gated and inert by default: importing it costs nothing and
patches nothing. `H3_TRACE=/path/to/out.json` turns it on for one process.

What it records, and why each field is here rather than inferred:

  seq, heads, head_dim   the actual attention shape. Previously the fork had
                         two points for H3, both inferred from a docstring and
                         a log line in this repo.
  dtype                  bf16 is assumed everywhere; assumption, not evidence.
  module                 `dit` or `refiner`. H3 has NO cross-attention -- both
                         DiTBlock and RefinerBlock hold one self-attn over the
                         packed sequence -- but the refiner runs a different
                         and much shorter one, and folding the two together
                         would report a bimodal workload as one average.
  has_mask, has_scale    both currently cause our override to DECLINE, so
                         they are the coverage question: any call carrying
                         either leaves sage silently.
  route                  which path actually ran. `sage` counts are also
                         available from the fork's own get_dispatch_counts;
                         the value here is the DENOMINATOR, because a
                         fallback-heavy render and a fully-sage one are
                         indistinguishable in our log today (one warning is
                         emitted whether 1 or 10,000 calls fell back).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
from collections import Counter

_PATH = os.environ.get("H3_TRACE")
_LOCK = threading.Lock()
_CALLS: Counter = Counter()
_ROUTES: Counter = Counter()

enabled = bool(_PATH)


_CURRENT = threading.local()


def record(*, seq, heads, head_dim, dtype, module, has_mask, has_scale, route):
    """One attention call, at ENTRY. Keyed rather than appended: a 16-step
    render makes 50*16 identical DiT calls, and a list of 800 copies of one
    dict is a worse artifact than a count.

    Stashes the key thread-locally so `route()` can attribute the outcome
    without the call site having to carry the shape to every exit."""
    if not enabled:
        return
    key = (module, int(seq), int(heads), int(head_dim), str(dtype),
           bool(has_mask), bool(has_scale))
    _CURRENT.key = key
    with _LOCK:
        _CALLS[key] += 1
        n = sum(_CALLS.values())
    # Dump periodically rather than relying on shutdown. atexit does NOT run
    # on SIGTERM -- Python's default handler terminates without unwinding --
    # and ComfyUI did not reach it on SIGINT either, so a whole trace run
    # produced no file at all. 50 blocks per step means this lands within the
    # first few steps and then refreshes, so the artifact exists whether the
    # process is stopped politely, killed, or left running.
    if n % 200 == 0:
        dump()


def route(seq, outcome):
    """Which path the call recorded at entry actually took.

    `seq` is accepted and ignored on purpose: it makes the call sites
    self-documenting and lets a future assertion catch an entry/exit mismatch,
    but the attribution uses the stashed key so it cannot drift from what was
    counted."""
    if not enabled:
        return
    key = getattr(_CURRENT, "key", None)
    if key is None:
        return
    with _LOCK:
        _ROUTES[(key, outcome)] += 1


def dump():
    """Write the trace. Safe to call repeatedly; last write wins."""
    if not enabled:
        return
    shapes = []
    for key, n in sorted(_CALLS.items(), key=lambda kv: -kv[1]):
        module, seq, heads, head_dim, dtype, has_mask, has_scale = key
        routes = {r: c for (k, r), c in _ROUTES.items() if k == key}
        shapes.append({
            "module": module, "seq": seq, "heads": heads,
            "head_dim": head_dim, "dtype": dtype,
            "has_mask": has_mask, "has_scale": has_scale,
            "calls": n, "routes": routes,
        })
    payload = {
        "model": "MiniMax-H3",
        "note": ("H3 packs [text | refs | audio | video] into ONE sequence and "
                 "attends it with self-attention only. There is no cross-"
                 "attention in either DiTBlock or RefinerBlock, so a "
                 "self-vs-cross split does not apply to this architecture."),
        "attention_shapes": shapes,
        "total_calls": sum(_CALLS.values()),
    }
    try:
        from sageattention import get_dispatch_counts
        payload["sage_dispatch_counts"] = dict(get_dispatch_counts())
    except Exception as exc:
        payload["sage_dispatch_counts"] = f"unavailable: {exc}"
    path = _PATH
    assert path is not None  # `enabled` already guarantees it; this is for the type checker
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


# Dumped when the process exits, which is the only moment guaranteed to be
# after every render. Registered unconditionally but inert when disabled --
# a bare atexit hook costs nothing and cannot be forgotten the way a
# "remember to dump" step can.
atexit.register(dump)
