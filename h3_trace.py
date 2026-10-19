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
  module                 **`unknown`, always, and corrected here 2026-09-08.**
                         This used to say "`dit` or `refiner`", and the one
                         call site passed "dit" for every call, so every
                         refiner call was labelled a DiT one. The forward we
                         patch replaces `Attention.forward` on the CLASS, and
                         both DiTBlock and RefinerBlock instantiate the same
                         `Attention`, so the call site cannot tell them apart.
                         The point the old text made is still true -- the
                         refiner runs a much shorter sequence and folding the
                         two would report a bimodal workload as one average --
                         but `seq` is what separates them, not this field, and
                         `seq` is part of the counter key, so no data was ever
                         lost. Split a trace on seq; do not trust this field.
  has_mask, has_scale    constants, and true by construction rather than
                         assumed: `Attention.forward(self, x, rope_freqs=None,
                         transformer_options={})` has nowhere to carry a mask
                         or a scale. The old text called them "the coverage
                         question", which described a risk this path cannot
                         run: a call carrying either would have to come from a
                         different signature.
  route                  TWO different things share this name, which is worth
                         reading twice. `record(route=...)` is passed
                         "entered" at the single call site and is the
                         DENOMINATOR: every attention call, whatever happened
                         next. Attribution is `h3_trace.route(seq, outcome)`,
                         a separate function called at five exits in
                         attention.py -- sage, sage_chunked, fallback_kernel,
                         fallback_chunked, sol_delegate -- which uses the key
                         stashed at entry. So the numerator does NOT have to
                         come from the fork's get_dispatch_counts, though
                         comparing the two would be a real cross-check and
                         nothing here does it yet.

**Nothing covers this file.** No `bench/check_*.py` drives it, and
`docs/method/checks.md` has no row for it, so the five outcomes above are
correct-by-construction: no one has forced each case and confirmed the field
lands where it should. Treat a share computed from it as unverified until
that exists.
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
