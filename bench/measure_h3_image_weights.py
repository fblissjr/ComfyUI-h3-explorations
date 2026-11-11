#!/usr/bin/env python3
"""What the MiniMax-H3-Image weights are: where they sit beside fl2va and ref2va, and what a ComfyUI int8 file keeps of them.

Three questions, each answered from the weights and nothing else. One JSON row
per tensor is written as it is measured, so a stopped run keeps its rows.

**parentage** (`--fl2va`, `--ref2va`): is the image DiT a fine-tune of the fl2va
or the ref2va video DiT? Per block linear, norm and AdaLN tensor:

- `img_fl`, `img_ref`: relative distance of the image weight from each video
  weight, and `fl_ref`, the two video weights from each other, as the scale.
- The same three distances for the tensors outside the blocks (`OUTSIDE_BLOCKS`):
  the text projection and the token refiner a prompt's rows pass through
  first, and the picture's patch projection. `--blocks none` measures these
  alone.
- `alpha` (the four linears): where the image weight projects on the line from
  fl2va to ref2va, `<img - fl, ref - fl> / |ref - fl|^2`. A fine-tune of fl2va
  gives about 0, a fine-tune of ref2va about 1; a model that left both from
  further back gives something between, with `cos_delta` (the cosine of
  `img - fl` with `ref - fl`) near 0.

**conversion** (`--int8`): is the int8 file a faithful copy? Per quantized
linear:

- `err_shipped`: relative error of the file's dequantized weight against the
  release weight in ComfyUI's row order (`bench/h3_image/release.py`).
- `codes_equal`: the share of int8 codes that equal what comfy-kitchen's own
  quantizer gives from the release weight with the format the file declares
  (int8, per output row, Hadamard rotation in groups of 256, no stochastic
  rounding). 1.0 means the file is that recipe run on the release and making it
  again would give the same bytes.
- `err_release_order` (fused qkv only): the same error with the release's row
  order left as it is, the control that says the reordering is real.

**adaln** (`--int8`): a pruned file replaces the timestep embedder and the
full-width AdaLN projections with a small table. Per block, the relative error
of the table's modulation against the release's own, at the timesteps the image
model is ever run at (its six-step grid, and the level reference rows are held
at), and over the table's whole grid.

The unquantized tensors and the output heads are compared byte for byte by
`bench/convert_h3_image_dit.py --verify`, not here.

Needs a CUDA card with a few GB free (one tensor at a time); reads every file
it is given once.

    <comfy venv python> bench/measure_h3_image_weights.py --release <MiniMax-H3-Image folder> \\
        --fl2va <H3 release>/FL2VA/transformer --ref2va <H3 release>/Ref2VA/transformer \\
        --int8 <minimax_h3_image_pruned_int8_convrot.safetensors> --out bench/results/<date>_h3_image_weights.jsonl

    <comfy venv python> bench/measure_h3_image_weights.py --summarize bench/results/<date>_h3_image_weights.jsonl

The second form reads a finished run's rows and prints, per kind of tensor, the
median and range of each figure. It needs no card and no model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h3_image import release as rel  # noqa: E402

LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SMALL = ("norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight",
         "adaln_proj.linear.weight", "adaln_proj.linear.bias")
#: The tensors a prompt's rows pass through before the first block, and the picture's patch projection.
OUTSIDE_BLOCKS = ("condition_proj.weight", "condition_proj.bias", "video_patch_proj.weight",
                  *(f"token_refiner.blocks.{b}.{kind}.weight" for b in range(2) for kind in LINEARS),
                  "token_refiner.final_norm.weight")
INT8_FORMAT = dict(is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256, stochastic_rounding=0)
#: reference rows are held at this clean time (`comfy/ldm/minimax/model.py::VISUAL_COND_TIMESTEP`)
REFERENCE_T = 0.999


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum(dtype=torch.float64))


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """|a - b| / |b|."""
    return math.sqrt(dot(a - b, a - b) / dot(b, b))


def triangulate(img: torch.Tensor, fl: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Where `img` projects on the line from `fl` to `ref`, and how aligned its move from `fl` is with that line."""
    d_img, d_ref = img - fl, ref - fl
    cross, ref2 = dot(d_img, d_ref), dot(d_ref, d_ref)
    return {"alpha": cross / ref2, "cos_delta": cross / math.sqrt(dot(d_img, d_img) * ref2)}


def time_embedding(release: rel.Sharded, t: torch.Tensor) -> torch.Tensor:
    """The release's timestep embedding as the AdaLN projections consume it (`TimeEmbedder`, then AdalnProj's SiLU)."""
    dev = t.device
    w1 = release.get("time_embedder.proj_in.weight").to(dev).double()
    b1 = release.get("time_embedder.proj_in.bias").to(dev).double()
    w2 = release.get("time_embedder.proj_out.weight").to(dev).double()
    b2 = release.get("time_embedder.proj_out.bias").to(dev).double()
    half = w1.shape[1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64, device=dev) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    silu = torch.nn.functional.silu
    return silu(silu(emb @ w1.T + b1) @ w2.T + b2)


def curve(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A pruned file's table read at `t`, as core reads it (`_time_embedding`)."""
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def summarize(path: Path) -> int:
    """The rows of a finished run as one table: per kind of tensor, the median and range of each figure."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kinds: dict[str, list[dict]] = {}
    for row in rows:
        parts = row["key"].split(".")
        kinds.setdefault(".".join(parts[2:]) if parts[0] == "blocks" else row["key"], []).append(row)
    fields = ("img_fl", "img_ref", "fl_ref", "alpha", "cos_delta", "err_shipped", "err_release_order", "codes_equal",
              "adaln_err_grid_median", "adaln_err_grid_max")
    for kind, group in kinds.items():
        print(f"{kind}  ({len(group)} tensors)")
        for field in fields:
            values = sorted(r[field] for r in group if field in r)
            if values:
                print(f"    {field:<22} median {values[len(values) // 2]:.5f}   min {values[0]:.5f}   max {values[-1]:.5f}")
        for field in ("adaln_err_at_used_t", "adaln_err_moving_part_at_used_t"):
            values = sorted(max(r[field]) for r in group if field in r)
            if values:
                print(f"    {field:<22} worst timestep, median over tensors {values[len(values) // 2]:.5f}   max {values[-1]:.5f}")
        if all("img_fl" in r for r in group):
            closer = sum(r["img_fl"] < r["img_ref"] for r in group)
            print(f"    fl2va is the nearer of the two in {closer} of {len(group)}")
    last = [r for r in rows if r["key"].startswith("blocks.") and "alpha" in r]
    if last:
        by_block: dict[int, list[float]] = {}
        for r in last:
            by_block.setdefault(int(r["key"].split(".")[1]), []).append(r["alpha"])
        print("alpha by block (mean of the four linears):")
        print("    " + "  ".join(f"{b}:{sum(v) / len(v):.2f}" for b, v in sorted(by_block.items())))
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        return summarize(Path(sys.argv[2]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True, help="the MiniMax-H3-Image release folder")
    ap.add_argument("--fl2va", type=Path, help="the H3 release's FL2VA/transformer folder")
    ap.add_argument("--ref2va", type=Path, help="the H3 release's Ref2VA/transformer folder")
    ap.add_argument("--int8", type=Path, help="a ComfyUI int8 file of the image model")
    ap.add_argument("--blocks", default="all", help="'all', 'none', or a comma list of block indices")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.fl2va is None) != (args.ref2va is None):
        ap.error("--fl2va and --ref2va go together")
    if args.fl2va is None and args.int8 is None:
        ap.error("nothing to measure: give --fl2va and --ref2va, --int8, or all three")

    dev = "cuda"
    img = rel.Sharded(args.release / "transformer")
    fl = rel.Sharded(args.fl2va) if args.fl2va else None
    rf = rel.Sharded(args.ref2va) if args.ref2va else None
    int8 = safe_open(str(args.int8), "pt") if args.int8 else None
    if int8 is not None:
        from comfy_kitchen.tensor import TensorWiseINT8Layout as Int8
    depth = sum(1 for k in img.where if k.endswith(".mlp.fc1.weight") and k.startswith("blocks."))
    blocks = (range(depth) if args.blocks == "all" else
              [] if args.blocks == "none" else [int(b) for b in args.blocks.split(",")])
    started = time.time()

    with open(args.out, "w") as out:
        def write(row: dict) -> None:
            out.write(json.dumps(row) + "\n")
            out.flush()

        for b in blocks:
            for kind in LINEARS:
                key = f"blocks.{b}.{kind}.weight"
                wi = img.get(key).to(dev).float()
                row: dict = {"key": key}
                if fl is not None:
                    wf, wr = fl.get(key).to(dev).float(), rf.get(key).to(dev).float()
                    row.update(img_fl=relative(wi, wf), img_ref=relative(wi, wr), fl_ref=relative(wr, wf),
                               **triangulate(wi, wf, wr))
                    del wf, wr
                if int8 is not None:
                    codes, scale = int8.get_tensor(key).to(dev), int8.get_tensor(key + "_scale").to(dev)
                    target = rel.comfy_qkv_order(wi) if rel.is_fused_qkv(key) else wi
                    params = Int8.Params(scale=scale, orig_dtype=torch.float32, orig_shape=tuple(wi.shape),
                                         is_weight=True, convrot=True, convrot_groupsize=256)
                    shipped = Int8.dequantize(codes, params).float()
                    row["err_shipped"] = relative(shipped, target)
                    if rel.is_fused_qkv(key):
                        row["err_release_order"] = relative(shipped, wi)
                    ours, _ = Int8.quantize(target.to(torch.bfloat16), **INT8_FORMAT)
                    row["codes_equal"] = float((ours == codes).float().mean())
                    del codes, shipped, ours, target
                del wi
                write(row)
            if fl is not None:
                for kind in SMALL:
                    key = f"blocks.{b}.{kind}"
                    wi, wf, wr = (s.get(key).to(dev).float() for s in (img, fl, rf))
                    write({"key": key, "img_fl": relative(wi, wf), "img_ref": relative(wi, wr),
                           "fl_ref": relative(wr, wf)})
            for s in (img, fl, rf):
                if s is not None:
                    s.close()
            print(f"block {b} ({time.time() - started:.0f}s)", flush=True)

        if fl is not None:
            # what a prompt passes through before the blocks: the text projection and the token refiner
            for key in OUTSIDE_BLOCKS:
                wi, wf, wr = (s.get(key).to(dev).float() for s in (img, fl, rf))
                write({"key": key, "img_fl": relative(wi, wf), "img_ref": relative(wi, wr),
                       "fl_ref": relative(wr, wf)})
            for s in (img, fl, rf):
                s.close()
            print(f"outside the blocks ({time.time() - started:.0f}s)", flush=True)

        if int8 is not None and blocks and "adaln_t_table" in int8.keys():
            sigmas = json.loads(int8.metadata()["config"])["transformer"]["pdd_sigmas"]
            used = sorted({1.0 - s for s in sigmas[:-1]} | {REFERENCE_T})
            table = int8.get_tensor("adaln_t_table").to(dev).double()
            t_used = torch.tensor(used, dtype=torch.float64, device=dev)
            t_grid = torch.linspace(0.0, 1.0, table.shape[0], dtype=torch.float64, device=dev)
            e_used, e_grid = time_embedding(img, t_used), time_embedding(img, t_grid)
            c_used = curve(table, t_used)
            names = [f"blocks.{b}.adaln_proj.linear" for b in blocks] + ["final_layer.adaln_proj.linear"]
            for name in names:
                w, bias = img.get(name + ".weight").to(dev).double(), img.get(name + ".bias").to(dev).double()
                wc = int8.get_tensor(name + ".weight").to(dev).double()
                bc = int8.get_tensor(name + ".bias").to(dev).double()
                true_used, file_used = e_used @ w.T + bias, c_used @ wc.T + bc
                true_grid, file_grid = e_grid @ w.T + bias, table @ wc.T + bc
                at_used = (true_used - file_used).norm(dim=1) / true_used.norm(dim=1)
                # the part that moves with the timestep, the constant removed from both
                moving = true_used - true_grid.mean(0)
                at_used_moving = (moving - (file_used - file_grid.mean(0))).norm(dim=1) / moving.norm(dim=1)
                on_grid = (true_grid - file_grid).norm(dim=1) / true_grid.norm(dim=1)
                write({"key": name + ".modulation", "t": used, "adaln_err_at_used_t": at_used.tolist(),
                       "adaln_err_moving_part_at_used_t": at_used_moving.tolist(),
                       "adaln_err_grid_median": float(on_grid.median()), "adaln_err_grid_max": float(on_grid.max())})
                del w, wc
                img.close()
            print(f"adaln ({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
