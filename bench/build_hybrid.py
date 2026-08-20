#!/usr/bin/env python3
"""Build an fl2va/ref2va hybrid checkpoint by copying tensors, and prove the builder first.

The HF hybrids (`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`) are fl2va
with ref2va's `blocks.{N..49}.adaln_proj.linear.{weight,bias}` swapped in,
everything else fl2va, all four leaving `final_layer.adaln_proj` on fl2va
(byte-compared 2026-08-16, `docs/roadmap.md`). This builds the same kind of
file locally from the two parents on disk, with the cut as an argument, so a
hybrid nobody published -- every block plus the final layer -- is one command
rather than an 84 GB download that does not exist.

## The control comes before the build

A builder that writes a plausible 21 GB safetensors is the shape of tool whose
first production use is its first test. So `--verify-against` takes an
existing hybrid and compares EVERY tensor's bytes between what this would
write and that file, without writing anything. The intended first run is
`--blocks 30-49` against the HF `b30-49` file: it must match on all tensors.
Then the deliberate violation: the same command with `--blocks 25-49` must
NOT match (and must say which tensors differ), or the comparison is not
comparing. Only after both is `--out` used.

Comparison is per tensor by header offset, not whole-file, because the header
JSON (key order, metadata) can legitimately differ between two files holding
identical tensors.

## What the output is, exactly

For every tensor name in fl2va's header: if the name is
`blocks.<b>.adaln_proj.linear.{weight,bias}` with `<b>` in `--blocks`, or
`final_layer.adaln_proj.linear.{weight,bias}` with `--final-layer`, the bytes
come from ref2va; otherwise from fl2va. Dtypes and shapes are asserted equal
across the parents for every swapped tensor. `adaln_t_table` stays fl2va's,
as in the HF files: the two parents' tables differ in sign on basis columns
4-7 (`bench/results/2026-08-20_dit_internals.json`), which sounds fatal and is
not -- those columns carry ~0.1% of the modulation norm, so applying ref2va's
coefficients to fl2va's table costs ~0.2% at the modulation output (measured
2026-08-20 on the HF b15/b20/b25/b30 files, scratch, reproduced by
`analyze_checkpoint_delta.py`'s method). Metadata records the recipe.

The output lands wherever `--out` points; the repo convention is a
`Storage`-side folder symlinked into `models/diffusion_models`, and the path
is typed in the shell, never here.

    # control, no write
    python bench/build_hybrid.py --blocks 30-49 \\
        --verify-against models/diffusion_models/minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors
    # violation: must report a mismatch
    python bench/build_hybrid.py --blocks 25-49 --verify-against <same file>
    # the build
    python bench/build_hybrid.py --blocks 0-49 --final-layer --out <dir>/minimax_h3_hybrid_fl2va_ref2va_adaln_all-int8.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_checkpoint_delta import FL_NAME, RE_NAME, header  # noqa: E402

CHUNK = 64 << 20


def parse_blocks(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out or min(out) < 0 or max(out) > 49:
        sys.exit(f"refuse: --blocks {spec!r} is not a set of block indices in 0-49")
    return out


def swapped(name: str, blocks: set[int], final_layer: bool) -> bool:
    if name.startswith("blocks.") and ".adaln_proj.linear." in name:
        return int(name.split(".")[1]) in blocks
    if name.startswith("final_layer.adaln_proj.linear."):
        return final_layer
    return False


def raw(path: str, base: int, info: dict) -> bytes:
    o0, o1 = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return f.read(o1 - o0)


def plan(fl: str, re_: str, blocks: set[int], final_layer: bool):
    fl_h, fl_b = header(fl)
    re_h, re_b = header(re_)
    if set(fl_h) != set(re_h):
        sys.exit("refuse: the two parents do not carry the same tensor names")
    names = sorted(fl_h)
    items = []
    for name in names:
        src, h, b = (re_, re_h, re_b) if swapped(name, blocks, final_layer) else (fl, fl_h, fl_b)
        if swapped(name, blocks, final_layer):
            if fl_h[name]["dtype"] != re_h[name]["dtype"] or fl_h[name]["shape"] != re_h[name]["shape"]:
                sys.exit(f"refuse: {name} differs in dtype/shape between the parents")
        items.append((name, src, b, h[name]))
    return items


def verify(items, target: str) -> int:
    t_h, t_b = header(target)
    if set(t_h) != {n for n, *_ in items}:
        print("MISMATCH: tensor name sets differ")
        return 1
    bad = []
    for name, src, base, info in items:
        if raw(src, base, info) != raw(target, t_b, t_h[name]):
            bad.append(name)
    if bad:
        print(f"MISMATCH: {len(bad)} tensor(s) differ from {Path(target).name}; first: {bad[:5]}")
        return 1
    print(f"MATCH: every tensor of {Path(target).name} reproduced byte-for-byte ({len(items)} tensors)")
    return 0


def write(items, out: Path, meta: dict) -> None:
    hdr: dict = {"__metadata__": meta}
    off = 0
    for name, _src, _base, info in items:
        n = info["data_offsets"][1] - info["data_offsets"][0]
        hdr[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = out.with_suffix(out.suffix + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        for name, src, base, info in items:
            o0, o1 = info["data_offsets"]
            with open(src, "rb") as f:
                f.seek(base + o0)
                left = o1 - o0
                while left:
                    buf = f.read(min(CHUNK, left))
                    w.write(buf)
                    left -= len(buf)
    tmp.rename(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", required=True, help="block indices taking ref2va adaln, e.g. 30-49 or 0-49")
    ap.add_argument("--final-layer", action="store_true", help="also take final_layer.adaln_proj from ref2va")
    ap.add_argument("--models-dir", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify-against", help="an existing hybrid; compare every tensor, write nothing")
    g.add_argument("--out", help="output safetensors path")
    args = ap.parse_args()

    models = Path(args.models_dir) if args.models_dir else HERE.parents[2] / "models" / "diffusion_models"
    fl, re_ = str(models / FL_NAME), str(models / RE_NAME)
    blocks = parse_blocks(args.blocks)
    items = plan(fl, re_, blocks, args.final_layer)
    n_swap = sum(1 for name, *_ in items if swapped(name, blocks, args.final_layer))
    print(f"recipe: ref2va adaln in blocks {args.blocks}{' + final_layer' if args.final_layer else ''}; "
          f"{n_swap} tensors from ref2va, {len(items) - n_swap} from fl2va")

    if args.verify_against:
        return verify(items, args.verify_against)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"refuse: {out} exists; delete it yourself if you mean to rebuild")
    meta = {"built_by": "bench/build_hybrid.py", "fl2va": FL_NAME, "ref2va": RE_NAME,
            "adaln_from_ref2va_blocks": args.blocks,
            "final_layer_adaln_from_ref2va": str(bool(args.final_layer)).lower(),
            "adaln_t_table": "fl2va"}
    write(items, out, meta)
    print("written", out)
    h, _ = header(str(out))
    print(f"  {len(h)} tensors; re-verifying against the recipe ...")
    return verify(items, str(out))


if __name__ == "__main__":
    sys.exit(main())
