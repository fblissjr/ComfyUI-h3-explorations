#!/usr/bin/env python3
"""Convert Beidouqixing's FlashGen 4-step LoRA, at its full rank, for the pruned fl2va checkpoint.

`Beidouqixing/minimax-h3-4step-lora-flashgen` (Apache-2.0) ships a rank-64
PEFT LoRA over MiniMax H3's native modules: `blocks.N.attn.qkv_proj`,
`attn.out_proj`, `mlp.fc1`, `mlp.fc2`, `adaln_proj.linear`, the token refiner's
and the final layer's AdaLN. Its own `merge_lora_ckpt.py` merges
`W + scale * lora_B @ lora_A` into the release's `FL2VA/transformer` shards at
scale 1.0. That is the ground truth this converter reproduces. kijai's
conversion (HF `Kijai/MiniMax-H3-experimental`) is a dynamic rank resize of the
same file at sv_fro 0.95; this one keeps the full rank.

## The mapping

- **Names.** `{module}.lora_A.default.weight` becomes
  `diffusion_model.{module}.lora_A.weight`, and likewise for B. One `.alpha` =
  rank per module, so ComfyUI's alpha / rank is the publisher's scale 1.0.
  ComfyUI reads alpha from a tensor, never from metadata
  (`bench/check_lora_alpha.py`).
- **qkv rows.** The release interleaves q, k and v per head. ComfyUI's
  `qkv_proj` is `cat([q; k; v])` in bands (`bench/convert_pdd_lora.py`, the
  qkv block). So `lora_B` rows are permuted
  `[head, qkv, dim] -> [qkv, head, dim]`. This is **checked on the weights**,
  not assumed: the release's `qkv_proj.weight`, permuted the same way, must
  match the pruned checkpoint's dequantised one row for row, and must NOT
  match it unpermuted.
- **AdaLN.** The pruned checkpoint's `adaln_proj.linear` takes an 8-column
  time basis (`adaln_t_table`), not the 2688-wide `silu(time_embedder(t))`. So
  `lora_A` is re-expressed in that basis. The LoRA's time curve
  `grid @ A.T`, over the 1025-row grid from the RELEASE's time embedder, is
  least-squares fitted onto `[adaln_t_table, 1]`. The basis part becomes the
  new `lora_A`, and the constant part becomes a `.diff_b` on the bias, because
  the pruned form keeps the curve's mean in its bias. This is
  `bench/convert_pdd_lora.py`'s method, with its per-block reconstruction
  bound. It is partition-specific: the fl2va table, for the fl2va checkpoint.
- **fc1, fc2, out_proj.** Renamed. The publisher trains on the native layout,
  and native fc1 is ComfyUI's `[gate; up]`; the layout check confirms fc1 too.

## Checks it runs

1. Layout: qkv (block and token-refiner) and fc1, release against checkpoint.
2. AdaLN fit per module, refused above 1e-3 relative.
3. The output's delta against the source delta, reconstructed on both sides,
   for one module of each kind.
4. With `--compare`, each module's delta against kijai's lossy file (cosine),
   as a description, not a gate.

CPU only. Reads a few shards of the release and the pruned checkpoint's
headers and small tensors.

    CUDA_VISIBLE_DEVICES="" python bench/convert_flashgen_lora.py \\
        --lora <flashgen>/minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors \\
        --release <MiniMaxAI_MiniMax-H3> \\
        --pruned <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --out <models>/loras/h3/<name>.safetensors \\
        [--compare <kijai file>] [--record bench/results/<date>_flashgen_lora_conversion.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
from pdd_math import silu_temb_grid  # noqa: E402
from h3_config import FLASHGEN_MANUAL_SIGMAS  # noqa: E402

A_SUF, B_SUF = ".lora_A.default.weight", ".lora_B.default.weight"
HEAD_DIM = 128          # inherited: vendor_config attention_head_dim
GRID_ROWS = 1025        # inherited: adaln_t_table's first dimension
ADALN_FIT_BOUND = 1e-3  # inherited from bench/convert_pdd_lora.py
#: The publisher's base_schedule (merge_lora_ckpt.py DEFAULT_BASE_SCHEDULE).
BASE_SCHEDULE = (1.0, 0.7, 0.4, 0.15, 0.0)


def shifted(shift, t):
    return shift * t / (1 + (shift - 1) * t)


#: The release partition the pruned checkpoint was cut from, set by
#: `--partition`. FlashGen was trained on FL2VA; converting it for Ref2VA fits
#: its adaln onto that partition's own time basis and embedder.
PARTITION = "FL2VA"


def release_tensor(release: Path, key: str) -> torch.Tensor:
    d = release / PARTITION / "transformer"
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(d / idx[key]), "pt") as f:
        return f.get_tensor(key).float()


def dequantised(handle, keys, module) -> torch.Tensor:
    w = handle.get_tensor(f"{module}.weight")
    if f"{module}.weight_scale" not in keys:
        return w.float()
    cfg = json.loads(bytes(handle.get_tensor(f"{module}.comfy_quant").tolist()).decode())
    if cfg.get("format") != "int8_tensorwise" or not cfg.get("convrot"):
        raise SystemExit(f"{module}: quantised as {cfg}; this converter does not dequantise that")
    sys.path.append(str(COMFY))
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    return dequantize_int8_convrot_weight(w, handle.get_tensor(f"{module}.weight_scale"),
                                          int(cfg["convrot_groupsize"])).float()


def qkv_to_bands(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Rows [head, qkv, dim] -> [qkv, head, dim]."""
    rest = x.shape[1:]
    return x.reshape(heads, 3, HEAD_DIM, *rest).transpose(0, 1).reshape(3 * heads * HEAD_DIM, *rest)


def row_cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).median())


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--pruned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--partition", choices=("FL2VA", "Ref2VA"), default="FL2VA",
                    help="the release partition --pruned was cut from (default FL2VA)")
    args = ap.parse_args()
    global PARTITION
    PARTITION = args.partition
    rec = {"source": args.lora.name, "checks": {}}

    with safe_open(str(args.lora), "pt") as f:
        src = {k: f.get_tensor(k) for k in f.keys()}
    modules = sorted({k[: -len(A_SUF)] for k in src if k.endswith(A_SUF)})
    stray = [k for k in src if not (k.endswith(A_SUF) or k.endswith(B_SUF))]
    if stray or any(m + B_SUF not in src for m in modules):
        raise SystemExit(f"unpaired or unexpected tensors: {stray[:4]}")
    ranks = {src[m + A_SUF].shape[0] for m in modules}
    if len(ranks) != 1:
        raise SystemExit(f"mixed ranks {sorted(ranks)}")
    rank = ranks.pop()

    pruned = safe_open(str(args.pruned), "pt")
    pkeys = set(pruned.keys())
    table = pruned.get_tensor("adaln_t_table").to(torch.float64)          # [1025, 8]
    heads = pruned.get_slice("blocks.0.attn.qkv_proj.weight").get_shape()[0] // (3 * HEAD_DIM)

    # 1. layout, on the weights
    layout = {}
    for mod in ("blocks.0.attn.qkv_proj", "blocks.49.attn.qkv_proj", "token_refiner.blocks.0.attn.qkv_proj"):
        rel, ck = release_tensor(args.release, f"{mod}.weight"), dequantised(pruned, pkeys, mod)
        layout[mod] = {"permuted": row_cos(qkv_to_bands(rel, heads), ck), "as_stored": row_cos(rel, ck)}
        if not (layout[mod]["permuted"] > 0.99 and layout[mod]["as_stored"] < 0.9):
            raise SystemExit(f"{mod}: qkv layout is not the per-head interleave this converter assumes: {layout[mod]}")
    rel, ck = release_tensor(args.release, "blocks.0.mlp.fc1.weight"), dequantised(pruned, pkeys, "blocks.0.mlp.fc1")
    layout["blocks.0.mlp.fc1"] = {"as_stored": row_cos(rel, ck)}
    if layout["blocks.0.mlp.fc1"]["as_stored"] < 0.99:
        raise SystemExit(f"fc1 is not stored in ComfyUI's order: {layout['blocks.0.mlp.fc1']}")
    rec["checks"]["layout_median_row_cosine"] = layout

    # 2. the time grid from the release's own time embedder
    te = {k: release_tensor(args.release, f"time_embedder.{k}") for k in
          ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")}
    grid = silu_temb_grid(te["proj_in.weight"], te["proj_in.bias"], te["proj_out.weight"],
                          te["proj_out.bias"], rows=GRID_ROWS).to(torch.float64)
    design = torch.cat([table, torch.ones(table.shape[0], 1, dtype=torch.float64)], dim=1)

    out, worst_fit = {}, 0.0
    for m in modules:
        a, b = src[m + A_SUF], src[m + B_SUF]
        dst = f"diffusion_model.{m}"
        if m.endswith("attn.qkv_proj"):
            b = qkv_to_bands(b, heads)
        if m.endswith("adaln_proj.linear"):
            if f"{m}.weight" not in pkeys or pruned.get_slice(f"{m}.weight").get_shape()[1] != table.shape[1]:
                raise SystemExit(f"{m}: the pruned checkpoint's adaln is not on the {table.shape[1]}-column basis")
            curve = grid @ a.to(torch.float64).T                                 # [rows, rank]
            coef = torch.linalg.lstsq(design, curve).solution                    # [9, rank]
            true = curve @ b.to(torch.float64).T
            got = (design @ coef) @ b.to(torch.float64).T
            err = float((got - true).norm() / true.norm())
            worst_fit = max(worst_fit, err)
            if err > ADALN_FIT_BOUND:
                raise SystemExit(f"{m}: adaln delta does not fit the basis (rel {err:.2e})")
            a = coef[:-1].T.to(torch.float32)                                    # [rank, 8]
            out[f"{dst}.diff_b"] = (b.to(torch.float64) @ coef[-1]).to(torch.float32)
        target = f"{m}.weight"
        if target not in pkeys:
            raise SystemExit(f"the pruned checkpoint has no {target}")
        out[f"{dst}.lora_A.weight"] = a.contiguous()
        out[f"{dst}.lora_B.weight"] = b.contiguous()
        out[f"{dst}.alpha"] = torch.tensor(float(rank))
    rec["checks"]["adaln_worst_relative_fit"] = worst_fit

    # 3. output delta against the source delta
    recon = {}
    for m in ("blocks.7.attn.qkv_proj", "blocks.7.mlp.fc1", "blocks.7.attn.out_proj", "blocks.7.mlp.fc2"):
        s = src[m + B_SUF].float() @ src[m + A_SUF].float()
        if m.endswith("qkv_proj"):
            s = qkv_to_bands(s, heads)
        o = out[f"diffusion_model.{m}.lora_B.weight"].float() @ out[f"diffusion_model.{m}.lora_A.weight"].float()
        recon[m] = float((o - s).norm() / s.norm())
        if recon[m] > 1e-6:
            raise SystemExit(f"{m}: emitted delta differs from the source ({recon[m]:.2e})")
    rec["checks"]["reconstruction_relative"] = recon

    # 4. against kijai's resize, descriptive
    if args.compare:
        cmp = {}
        with safe_open(str(args.compare), "pt") as k:
            kk = set(k.keys())
            for m in ("blocks.0.attn.qkv_proj", "blocks.24.mlp.fc1", "blocks.49.mlp.fc2",
                      "blocks.49.attn.out_proj", "blocks.24.adaln_proj.linear"):
                p = f"diffusion_model.{m}"
                if f"{p}.lora_A.weight" not in kk:
                    continue
                ka = k.get_tensor(f"{p}.lora_A.weight").float()
                kb = k.get_tensor(f"{p}.lora_B.weight").float()
                kal = float(k.get_tensor(f"{p}.alpha")) if f"{p}.alpha" in kk else ka.shape[0]
                # float64: the flattened deltas run to 1e8 elements, and fp32
                # norms over that many drift enough to print a cosine above 1.
                kd = ((kb.double() @ ka.double()) * (kal / ka.shape[0])).flatten()
                od = (out[f"{p}.lora_B.weight"].double() @ out[f"{p}.lora_A.weight"].double()).flatten()
                cmp[m] = {"cosine": float((kd @ od) / (kd.norm() * od.norm())),
                          "norm_ratio_kijai_over_full": float(kd.norm() / od.norm()),
                          "kijai_rank": int(ka.shape[0])}
        rec["checks"]["vs_kijai_resize"] = cmp

    sig = [shifted(12.0, t) for t in BASE_SCHEDULE]
    const = [float(x) for x in FLASHGEN_MANUAL_SIGMAS.split(",")]
    if any(abs(a - b) > 1e-6 for a, b in zip(sig, const)) or len(sig) != len(const):
        raise SystemExit(f"the publisher's schedule at shift 12 {sig} disagrees with "
                         f"h3_config.FLASHGEN_MANUAL_SIGMAS {const}")
    meta = {
        "format": "pt",
        "source": f"Beidouqixing/minimax-h3-4step-lora-flashgen {args.lora.name}",
        "conversion": ("bench/convert_flashgen_lora.py: full rank; qkv lora_B rows [head,qkv,dim]->[qkv,head,dim]; "
                       f"adaln lora_A projected onto the pruned {PARTITION.lower()} adaln_t_table basis with the mean in diff_b; "
                       "alpha = rank (publisher scale 1.0)"),
        "adaln_curve_basis": PARTITION.lower(),
        "sampler_steps": "4",
        "manual_sigmas_shift12": FLASHGEN_MANUAL_SIGMAS,
        "base_schedule": ",".join(str(x) for x in BASE_SCHEDULE),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=meta)
    rec.update(out=args.out.name, rank=rank, modules=len(modules), metadata=meta)
    print(json.dumps(rec["checks"], indent=1))
    print(f"wrote {args.out.name}: {len(modules)} modules at rank {rank}")
    if args.record:
        args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
