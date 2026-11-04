#!/usr/bin/env python3
"""Where a LoRA applied at the call spends its time, one block's modules at a time.

    <comfy venv python> bench/profile_lora_branch.py --lora h3/<file> [--tokens 32768]
        [--scale-to 110000] [--block 25] [--iters 10]

`lora_branch.py` adds each module's delta at the call: `out += s * B (A x)`,
in place with `addmm_`, with A and B copied from pinned host RAM per call. A
FlashGen render pays a sampler-time cost for this that the 2026-09-26 record
(`bench/results/2026-09-26_flashgen_lora_path_s1.md`) calls "not optimized"
without saying where it goes. This times the branch's own operations on real A
and B from the file, at the shapes the DiT feeds them, so the cost can be
split before anything is changed:

- `copy`     the per-call host-to-device copy of A and B, alone;
- `branch`   `_Branch.add_into` as it runs in a render, the copy included;
- `resident` the same with A and B already on the card, so `branch - resident`
             is what the copy costs inside the call;
- `fc2`      `_mlp_forward`'s path: swiglu recomputed from fc1's output in
             `FC2_CHUNK_ROWS` chunks, then the branch on it.

Activations are random at the real widths; the branch's cost does not depend
on their values. Every operation here is linear in the token count (the ranks
and widths are fixed), so `--tokens` can stay small enough to share a card and
`--scale-to` reports the figure at a render's length. **That scaling is an
assumption this tool states, not one it checks**: run two `--tokens` values if
it matters. It runs the module code itself (`lora_branch._Branch`,
`lora_branch._mlp_forward`'s swiglu), never a copy of it.

Needs CUDA and nothing else: no server, no checkpoint. Peak memory is about
`tokens * 28672 * 2 bytes * 3` (fc1's output, the swiglu chunk, the branch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

if not torch.cuda.is_available():     # before comfy's import, which insists on a device
    print("no CUDA; this tool times GPU work and has nothing to do without it")
    raise SystemExit(2)

import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import lora_branch as lb  # noqa: E402

#: The modules a block's branch touches, with the width of the input each one
#: is fed and the width of the output it adds into. Inherited: read off the
#: LoRA files' A and B shapes and `comfy/ldm/minimax/model.py` (hidden 5376,
#: attention inner 7168, MLP 14336 with a 2x fc1 for swiglu).
MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _time(fn, iters: int) -> float:
    """Median milliseconds of `fn` over `iters` runs, after one untimed run."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def profile_module(name: str, br: "lb._Branch", T: int, iters: int) -> dict:
    """One module's row: the copy alone, the branch as it runs, the branch with A and B resident."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    swiglu = comfy.ops.INPUT_ACT_EAGER["swiglu"]
    assert br.a is not None and br.b is not None
    rank, d_in = br.a.shape
    d_out = br.b.shape[0]
    a_dev, b_dev = br.a.to(dev).to(dt), br.b.to(dev).to(dt)
    out = torch.zeros(T, d_out, device=dev, dtype=dt)
    chunks = [(i, min(i + lb.FC2_CHUNK_ROWS, T)) for i in range(0, T, lb.FC2_CHUNK_ROWS)]

    def copy():
        br._dev(br.a, out)
        br._dev(br.b, out)

    extra = {}
    if name == "mlp.fc2":
        h = torch.randn(T, 2 * d_in, device=dev, dtype=dt)

        def branch():
            factors = br.device_factors(out)
            for i, j in chunks:
                br.add_into(swiglu(h[i:j]), out[i:j], factors)

        def resident():
            for i, j in chunks:
                out[i:j].addmm_(swiglu(h[i:j]) @ a_dev.T, b_dev.T)

        def act_only():
            for i, j in chunks:
                swiglu(h[i:j])
        extra["swiglu_ms"] = _time(act_only, iters)
    else:
        x = torch.randn(T, d_in, device=dev, dtype=dt)

        def branch():
            br.add_into(x, out)

        def resident():
            out.addmm_(x @ a_dev.T, b_dev.T)
    return {"module": name, "rank": rank, "d_in": d_in, "d_out": d_out, "tokens": T,
            "copy_ms": _time(copy, iters), "branch_ms": _time(branch, iters),
            "resident_ms": _time(resident, iters), **extra}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required since 2026-11-04: its default was the FlashGen r64 file for
    # fl2va, retired with that lane, and no kept graph names a LoRA for this
    # node to default to.
    ap.add_argument("--lora", required=True,
                    help="a file under ComfyUI's loras folder, as `MiniMaxH3LoRABranch` names it")
    ap.add_argument("--tokens", type=int, default=32768, help="rows fed to each module")
    ap.add_argument("--scale-to", type=int, default=None,
                    help="also report each figure scaled linearly to this many rows")
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--blocks-per-forward", type=int, default=50)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", type=Path, default=None, help="write the rows here too")
    args = ap.parse_args(argv)

    path = folder_paths.get_full_path_or_raise("loras", args.lora)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    prefix = f"{lb.PREFIX}blocks.{args.block}."
    branches = lb.parse_lora({k: v for k, v in sd.items() if k.startswith(prefix)}, 1.0)
    T = args.tokens
    rows = []
    for name in MODULES:
        br = branches.get(f"blocks.{args.block}.{name}")
        if br is None or br.a is None:
            continue
        rows.append(profile_module(name, br, T, args.iters))
        torch.cuda.empty_cache()

    keys = [k for k in ("copy_ms", "branch_ms", "resident_ms", "swiglu_ms")]
    print(f"{Path(args.lora).name}, block {args.block}, {T} rows; median ms of {args.iters}")
    print(f"  {'module':16s} {'rank':>4s} " + " ".join(f"{k:>12s}" for k in keys))
    for r in rows:
        print(f"  {r['module']:16s} {r['rank']:4d} " +
              " ".join(f"{r[k]:12.2f}" if k in r else f"{'-':>12s}" for k in keys))
    tot = {k: sum(r.get(k, 0.0) for r in rows) for k in keys}
    n = args.blocks_per_forward
    print(f"  {'block total':16s} {'':4s} " + " ".join(f"{tot[k]:12.2f}" for k in keys))
    print(f"  x {n} blocks, s/forward at {T} rows: " +
          ", ".join(f"{k[:-3]} {tot[k] * n / 1000:.2f}" for k in keys))
    if args.scale_to:
        f = args.scale_to / T
        print(f"  scaled linearly to {args.scale_to} rows (assumed, see docstring): " +
              ", ".join(f"{k[:-3]} {tot[k] * n * f / 1000:.2f}" for k in keys if k != "copy_ms") +
              f", copy {tot['copy_ms'] * n / 1000:.2f} (does not scale with rows)")
    if args.json:
        args.json.write_text(json.dumps({"lora": args.lora, "block": args.block, "rows": rows,
                                         "gpu": torch.cuda.get_device_name()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
