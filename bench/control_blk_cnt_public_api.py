#!/usr/bin/env python3
"""One positive and one negative control on `blk_cnt`, through the PUBLIC API.

The joint plan for the Sol observability lane (2026-09-01) had Codex
independently reproduce one valid count and one rejected buffer on the
installed CUDA wheel before any production render. Codex's process lost CUDA
access mid-review, so it ran the pair on the eager backend only and left the
CUDA half owed. This is that half, written as a bench record rather than a
test so the result lands in `bench/results/` with the wheel that answered.

Deliberately NOT the observer check's fixtures and not the upstream tests'
either: a five-block sequence with a two-block sink and one sink_q row, so
the closed-form count is short enough to write by hand --

    forced[q] = |sink| + |{q-1, q, q+1} that exist and are not sink|,
    sink_q rows = NTB

-- and a request that clears every threshold (tau minus 1e9) so every row is
NTB. Positive: the count equals the hand-written vector on every head, and
the output is bitwise what it is without the buffer. Negative: a buffer one
block too long, and one of the wrong dtype, are refused by public dispatch
rather than filled or broadcast. A third arm asks the eager backend the same
question on the same tensors, so the two implementations are compared on
this fixture too.

    python bench/control_blk_cnt_public_api.py [--out bench/results/<file>.json]

Exit 0 when every arm holds, 1 otherwise, 2 without CUDA or without a wheel
whose `sol_attn` takes `blk_cnt`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", help="write the record here (JSON)")
    args = ap.parse_args()
    try:
        import importlib.metadata as md
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.sol_attn import sol_attn as eager
        from comfy_kitchen.exceptions import NoCapableBackendError
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 2
    if "blk_cnt" not in inspect.signature(ck.sol_attn).parameters:
        print("SKIP: the installed comfy_kitchen.sol_attn has no blk_cnt")
        return 2
    version = md.version("comfy_kitchen")
    record = {"produced_by": "bench/control_blk_cnt_public_api.py", "comfy_kitchen": version,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": {}}
    failed = []

    def arm(name, ok, detail):
        record["arms"][name] = {"ok": bool(ok), "detail": detail}
        print(f"  {'ok  ' if ok else 'FAIL'} {name}   {detail}")
        if not ok:
            failed.append(name)

    print(f"blk_cnt controls through comfy_kitchen.sol_attn, {version} on {record['device']}\n")
    torch.manual_seed(7)
    t, h, n = 320, 2, 5
    q, k, v = (torch.randn(1, t, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    sink, sink_q = [0, 2], [3, 4]
    # by hand: sink 2 blocks; diagonal outside the sink -> q0: {} (0,1 are sink) ->2; q1: {2} ->3;
    # q2: {1 sink,2,3} ->4; q3: sink_q -> NTB 5; q4: {3,4} -> 4
    want = torch.tensor([2, 3, 4, 5, 4], dtype=torch.int32, device="cuda").view(1, 1, n).expand(1, h, n)

    ref = ck.sol_attn(q, k, v, tau=1e9, sink_blocks=sink, sink_q=sink_q)
    cnt = torch.empty(1, h, n, dtype=torch.int32, device="cuda")
    out = ck.sol_attn(q, k, v, tau=1e9, sink_blocks=sink, sink_q=sink_q, blk_cnt=cnt)
    arm("positive: forced-pair count by hand, both heads", torch.equal(cnt, want),
        f"got {cnt[0, 0].tolist()} / {cnt[0, 1].tolist()}, want {want[0, 0].tolist()}")
    arm("positive: output bitwise unchanged by the buffer", torch.equal(out, ref), "same bytes")
    cnt_all = torch.empty(1, h, n, dtype=torch.int32, device="cuda")
    ck.sol_attn(q, k, v, tau=-1e9, sink_blocks=sink, sink_q=sink_q, blk_cnt=cnt_all)
    arm("positive: everything routed at tau -1e9", bool((cnt_all == n).all()), f"got {cnt_all[0, 0].tolist()}")
    cnt_e = torch.empty(1, h, n, dtype=torch.int32, device="cuda")
    eager(q, k, v, tau=1e9, sink_blocks=sink, sink_q=sink_q, blk_cnt=cnt_e)
    arm("eager agrees on the same tensors", torch.equal(cnt_e, want), f"eager {cnt_e[0, 0].tolist()}")

    for label, bad in (("one block too long", torch.empty(1, h, n + 1, dtype=torch.int32, device="cuda")),
                       ("int64 dtype", torch.empty(1, h, n, dtype=torch.int64, device="cuda")),
                       ("cpu device", torch.empty(1, h, n, dtype=torch.int32))):
        try:
            ck.sol_attn(q, k, v, tau=1e9, sink_blocks=sink, sink_q=sink_q, blk_cnt=bad)
            arm(f"negative: {label} refused", False, "ACCEPTED")
        except (ValueError, NoCapableBackendError) as exc:
            arm(f"negative: {label} refused", True, f"{type(exc).__name__}")

    record["verdict"] = "all arms hold" if not failed else f"FAILED: {failed}"
    print(f"\n{record['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(f"record written to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
