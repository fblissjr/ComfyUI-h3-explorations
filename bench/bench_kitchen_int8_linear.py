#!/usr/bin/env python3
"""comfy-kitchen's `int8_linear` on H3's real block-0 layers: time per call, and a bitwise digest.

    <comfy venv python> bench/bench_kitchen_int8_linear.py --label new --m 104103 120666 \
        --out data/results/<date>_<what>.json

Runs the layer ComfyUI runs for the pruned int8 convrot checkpoint
(`comfy/ops.py`: `ck.int8_linear(x, qdata, scale, bias, dtype, convrot=True,
convrot_groupsize=256)`), on the four linear shapes of block 0 and the
checkpoint's own weights and scales, with seeded random bf16 activations at
each token count `--m`. The two token counts of the 2026-10-01 record are
H3's packed lengths at 1344x768 and 345 frames, without and with two 2048x2048
references; `bench/preflight_graph.py` prices any other graph.

**To compare two builds, run it twice and compare `digest`.** The digest is a
position-weighted sum of the output's bf16 bits, so it is equal only when the
output is bit-identical (it is not an error measure). To measure a wheel other
than the venv's, install it beside the venv and put that directory first:

    uv pip install --python <comfy venv python> --target <dir> --no-deps <wheel>
    PYTHONPATH=<dir> <comfy venv python> bench/bench_kitchen_int8_linear.py ...

**Do not let it import the kitchen clone.** The clone's own `comfy_kitchen/`
carries whatever `_C.abi3.so` an in-place build last left there, which is not
the installed wheel (on 2026-10-01 it was a build from 2026-09-28, and the
kitchen's pytest imported it for both "builds" until the tests were copied out
of the clone: pytest, `python -m` and `python -c` put the clone first when run
from it, and `PYTHONPATH` can too). Each row names the version it imported, and
the run refuses a source tree.

When the build has `cutlass_int8_selected_config` (kitchen 2026-09-30, #215),
each row also carries the GEMM tile config the shape heuristic picked.

Timing is events around five calls after two warmups, on an idle card, with
the allocator state of one process; it is a comparison between two runs of
this script, not an absolute. No server needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "workflows"))
import h3_config as C  # noqa: E402

#: Block 0's four linears. Provenance: read from the checkpoint's own header (2026-10-01); every
#: block shares them.
LAYERS = ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2")
WARMUP, ITERS = 2, 5


def digest(t: torch.Tensor) -> int:
    """Position-weighted sum of the output's bf16 bits: equal only for a bit-identical output."""
    v = t.reshape(-1).view(torch.int16)
    acc, step = 0, 1 << 24
    for i in range(0, v.numel(), step):
        c = v[i:i + step].to(torch.int64)
        idx = (torch.arange(i, i + c.numel(), device=c.device, dtype=torch.int64) % 1000003) + 1
        acc = (acc + int((c * idx).sum().item())) % (1 << 61)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="names the build in each row")
    ap.add_argument("--m", type=int, nargs="+", required=True, help="token counts (activation rows)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import comfy_kitchen as ck
    from safetensors import safe_open
    pkg = Path(str(ck.__file__)).resolve().parent
    if (pkg.parent / "pyproject.toml").exists():
        print(f"REFUSED: comfy_kitchen was imported from a source tree ({pkg.parent.name}/), not an "
              f"installed wheel; run this from outside the clone")
        return 1
    # The dist-info beside the imported package names the build, in site-packages and in a `--target`
    # directory alike; `importlib.metadata` could answer for another install on the path.
    infos = sorted(pkg.parent.glob("comfy_kitchen-*.dist-info"))
    version = infos[0].name[len("comfy_kitchen-"):-len(".dist-info")] if infos else "unknown"
    print(f"comfy_kitchen {version}", flush=True)

    ckpt = C.COMFY_ROOT / "models" / "diffusion_models" / C.MODELS["unet_fl2va"]
    weights = {}
    with safe_open(str(ckpt), "pt", device="cpu") as f:
        for name in LAYERS:
            weights[name] = (f.get_tensor(name + ".weight").cuda(), f.get_tensor(name + ".weight_scale").cuda())
    select = getattr(getattr(getattr(ck, "backends", None), "cuda", None), "_C", None)
    select = getattr(select, "cutlass_int8_selected_config", None)

    rows = []
    for m in args.m:
        for name in LAYERS:
            q, s = weights[name]
            n, k = q.shape
            g = torch.Generator(device="cuda")
            g.manual_seed(1234 + k)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)

            def run():
                return ck.int8_linear(x, q, s, None, torch.bfloat16, convrot=True, convrot_groupsize=256)

            out = run()
            for _ in range(WARMUP - 1):
                out = run()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(ITERS):
                out = run()
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / ITERS
            row = {"label": args.label, "kitchen": version, "m": m, "layer": name.removeprefix("blocks.0."),
                   "n": n, "k": k, "ms": round(ms, 3), "tops": round(2 * m * n * k / ms / 1e9, 1),
                   "digest": digest(out), "tile_config": select(m, n, k) if select else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del x, out
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
