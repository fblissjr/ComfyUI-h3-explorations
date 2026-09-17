"""Device time of every attention call the Sol override handles, inside a real render.

Env-gated and inert by default, like `h3_trace.py`: `H3_SOL_TIME=<out.jsonl>`
arms it for one server process. Unarmed, `enabled()` is one bool and nothing
else runs.

It exists to check a replay against the thing it replays.
`bench/profile_sol_stages.py` splits a Sol call into stages on captured q/k/v
with the card to itself; whether that call costs the same inside a render,
beside three other resident models and weight streaming, is a separate fact,
and this records it: one row per call with its route (`sol`, or `dense` for
the chained fallback), block, sigma, sequence length and device milliseconds.
The `dense` rows are the fallback kernel's live cost on the steps outside
Sol's window, which no record had either.

Timing is a CUDA event pair on the current stream and **never synchronises**:
a pair is read only once `query()` says it has completed, on a later call. So
an armed render runs the same kernels in the same order with no added sync
point, and the cost of arming is two event records per call. The last call of
a process can stay unread; one row in several thousand, by construction.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

_PATH = os.environ.get("H3_SOL_TIME")
_pending: list = []


def enabled() -> bool:
    return bool(_PATH)


def _drain() -> None:
    import orjson
    done = []
    while _pending and _pending[0][2].query():
        meta, start, end = _pending.pop(0)
        meta["ms"] = round(start.elapsed_time(end), 3)
        done.append(meta)
    if done:
        with open(_PATH, "ab") as fh:
            for row in done:
                fh.write(orjson.dumps(row) + b"\n")


@contextmanager
def span(route: str, *, block, sigma, tokens: int, batch: int, heads: int):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _pending.append(({"route": route, "block": block, "sigma": sigma,
                          "tokens": int(tokens), "batch": int(batch), "heads": int(heads)},
                         start, end))
        _drain()
