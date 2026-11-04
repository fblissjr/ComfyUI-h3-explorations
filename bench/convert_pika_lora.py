#!/usr/bin/env python3
"""Convert Pika H3 Flex PEFT LoRAs (`fl2va`, `ref2va`) to ComfyUI format.

Supports two conversion modes from `adapter.safetensors` + `manifest.json`:

- `--mode full`: exact rank-128 (`qkv_proj` rank 384) BF16 conversion with
  explicit per-module `.alpha` tensors (`alpha == rank`, unit scale 1.0) and
  Pika's sampling/numerics contract recorded in `__metadata__`.
- `--mode svd_per_component`: truncates `to_q`, `to_k`, `to_v`, `to_out.0`,
  `ff.net.0.proj`, and `ff.net.2` individually via exact FP64 QR-SVD before
  fusing `qkv_proj` and swapping `mlp.fc1`'s SwiGLU halves. Running `sv_fro`
  before `qkv_proj` fusion prevents the larger pre-RMSNorm `to_q` and `to_k`
  deltas from starving the unnormalized `to_v` slice in late blocks (where
  post-fusion `sv_fro=0.99` drops `blocks.49.attn.qkv_proj` to rank 4-5 and
  loses 94.8% of `to_v` on `fl2va` and 72.2% on `ref2va`).

Usage:

    <comfy venv python> bench/convert_pika_lora.py \\
        --source-dir <Pika-H3-Flex-Early-Access-weights>/fl2va \\
        --out <out>.safetensors [--mode svd_per_component --sv-fro 0.995] [--verify]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _gram_rel_err(b1: torch.Tensor, a1: torch.Tensor, b2: torch.Tensor, a2: torch.Tensor) -> tuple[float, float]:
    """Return `(||b1 a1 - b2 a2||_F / ||b1 a1||_F, ||b1 a1||_F)` via r x r Gram products."""
    b1d, a1d = b1.double(), a1.double()
    b2d, a2d = b2.double(), a2.double()
    n11 = ((b1d.T @ b1d) * (a1d @ a1d.T)).sum()
    n22 = ((b2d.T @ b2d) * (a2d @ a2d.T)).sum()
    n12 = ((b1d.T @ b2d) * (a1d @ a2d.T)).sum()
    norm = float(n11.clamp(min=1e-30).sqrt().item())
    err = float(((n11 + n22 - 2.0 * n12).clamp(min=0.0).sqrt() / norm).item())
    return err, norm


def _qr_svd_truncate(
    b: torch.Tensor,
    a: torch.Tensor,
    sv_fro: float,
    min_rank: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Exact thin SVD of `b @ a` via QR on `b` and `a.T`, keeping `sv_fro` Frobenius norm."""
    qb, rb = torch.linalg.qr(b.double(), mode="reduced")
    qa, ra = torch.linalg.qr(a.double().T, mode="reduced")
    m = rb @ ra.T
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    cum = torch.cumsum(s.pow(2), dim=0) / s.pow(2).sum().clamp(min=1e-30)
    k = int((cum < (sv_fro ** 2)).sum().item()) + 1
    k = max(min_rank, min(k, int(s.shape[0])))
    s_sqrt = s[:k].sqrt()
    b_new = (qb @ (u[:, :k] * s_sqrt)).to(dtype).contiguous()
    a_new = ((s_sqrt.unsqueeze(1) * vh[:k, :]) @ qa.T).to(dtype).contiguous()
    return b_new, a_new, k


def _manual_sigmas_str(knots: list[int], shift: float) -> str:
    vals = []
    for k in knots:
        t = float(k) / 1000.0
        s = 0.0 if t <= 0.0 else (1.0 if t >= 1.0 else (shift * t) / (1.0 + (shift - 1.0) * t))
        vals.append(f"{s:.6f}".rstrip("0").rstrip(".") if s in (0.0, 1.0) else f"{s:.6f}")
    return ", ".join(vals)


def convert(
    source_dir: Path,
    out_path: Path,
    mode: str = "svd_per_component",
    sv_fro: float = 0.995,
    min_rank: int = 1,
    device: str = "cuda",
    verify: bool = True,
) -> dict:
    manifest_path = source_dir / "manifest.json"
    adapter_path = source_dir / "adapter.safetensors"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sched = manifest.get("sampling_schedule", {})
    knots = sched.get("knots", [1000, 750, 500, 250, 100, 40, 0])
    v_shift = float(sched.get("video_shift", 12.0))
    a_shift = float(sched.get("audio_shift", 3.0))

    blocks = [(f"transformer_blocks.{i}", f"diffusion_model.blocks.{i}") for i in range(50)] + [
        (f"token_refiner.refiner_blocks.{i}", f"diffusion_model.token_refiner.blocks.{i}") for i in range(2)
    ]

    out_tensors: dict[str, torch.Tensor] = {}
    ranks: list[int] = []
    comp_errs: dict[str, list[float]] = {k: [] for k in ("to_q", "to_k", "to_v", "out_proj", "fc1", "fc2")}

    with safe_open(adapter_path, framework="pt", device=device) as fo:
        src_meta = dict(fo.metadata() or {})
        for ob, cb in blocks:
            bs_qkv, as_qkv = [], []
            for x in ("to_q", "to_k", "to_v"):
                b_orig = fo.get_tensor(f"{ob}.attn.{x}.lora_B.default.weight")
                a_orig = fo.get_tensor(f"{ob}.attn.{x}.lora_A.default.weight")
                if mode == "full":
                    bn = b_orig.to(torch.bfloat16).contiguous()
                    an = a_orig.to(torch.bfloat16).contiguous()
                else:
                    bn, an, _ = _qr_svd_truncate(b_orig, a_orig, sv_fro=sv_fro, min_rank=min_rank)
                bs_qkv.append(bn)
                as_qkv.append(an)
                if verify:
                    err, _ = _gram_rel_err(b_orig, a_orig, bn, an)
                    comp_errs[x].append(err)

            b_qkv = torch.block_diag(*bs_qkv).to(torch.bfloat16).contiguous().cpu()
            a_qkv = torch.cat(as_qkv, dim=0).to(torch.bfloat16).contiguous().cpu()
            r_qkv = int(a_qkv.shape[0])
            ranks.append(r_qkv)
            out_tensors[f"{cb}.attn.qkv_proj.lora_B.weight"] = b_qkv
            out_tensors[f"{cb}.attn.qkv_proj.lora_A.weight"] = a_qkv
            out_tensors[f"{cb}.attn.qkv_proj.alpha"] = torch.tensor(float(r_qkv), dtype=torch.bfloat16)

            for okind, ckind, swap in (
                ("attn.to_out.0", "out_proj", False),
                ("ff.net.0.proj", "fc1", True),
                ("ff.net.2", "fc2", False),
            ):
                b_orig = fo.get_tensor(f"{ob}.{okind}.lora_B.default.weight")
                a_orig = fo.get_tensor(f"{ob}.{okind}.lora_A.default.weight")
                if swap:
                    h = b_orig.shape[0] // 2
                    b_orig = torch.cat([b_orig[h:], b_orig[:h]], dim=0)
                if mode == "full":
                    bn = b_orig.to(torch.bfloat16).contiguous()
                    an = a_orig.to(torch.bfloat16).contiguous()
                else:
                    bn, an, _ = _qr_svd_truncate(b_orig, a_orig, sv_fro=sv_fro, min_rank=min_rank)
                if verify:
                    err, _ = _gram_rel_err(b_orig, a_orig, bn, an)
                    comp_errs[ckind].append(err)
                bn_cpu = bn.cpu()
                an_cpu = an.cpu()
                rk = int(an_cpu.shape[0])
                ranks.append(rk)
                cpre = f"{cb}.attn.{ckind}" if ckind == "out_proj" else f"{cb}.mlp.{ckind}"
                out_tensors[f"{cpre}.lora_B.weight"] = bn_cpu
                out_tensors[f"{cpre}.lora_A.weight"] = an_cpu
                out_tensors[f"{cpre}.alpha"] = torch.tensor(float(rk), dtype=torch.bfloat16)

    avg_rank = st.mean(ranks)
    meta = dict(src_meta)
    meta.update({
        "source_adapter": "adapter.safetensors",
        "converted_for": "comfyui-minimax-h3",
        "adapter_id": str(manifest.get("adapter_id", source_dir.name)),
        "task": str(manifest.get("task", source_dir.name)),
        "base_model": str(src_meta.get("base_model") or manifest.get("base_model_hf_repo", "MiniMaxAI/MiniMax-H3")),
        "base_revision": str(src_meta.get("base_revision") or manifest.get("base_model_hf_revision", "42ed227e")),
        "numerics_policy": str(manifest.get("numerics_policy", "")),
        "knots": json.dumps(knots),
        "video_shift": str(v_shift),
        "audio_shift": str(a_shift),
        "method": str(sched.get("method", "renoise")),
        "comfy_sampler": "lcm",
        "manual_sigmas_shift12": _manual_sigmas_str(knots, v_shift),
        "baked_scale": "1.0",
        "conversion": (
            f"mode={mode}; sv_fro={sv_fro if mode != 'full' else 1.0}; "
            "per-component QR-SVD before qkv block-diag fusion (contiguous q|k|v); "
            "mlp.fc1 swiglu halves swapped; explicit .alpha == rank (scale 1.0)"
        ),
        "avg_rank": f"{avg_rank:.2f}",
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(out_path), metadata=meta)
    out_path.chmod(0o644)

    summary = {
        "out": out_path.name,
        "mode": mode,
        "sv_fro": sv_fro if mode != "full" else 1.0,
        "tensors": len(out_tensors),
        "avg_rank": round(avg_rank, 2),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
    }
    if verify:
        summary["component_rel_err"] = {
            k: {"mean": round(st.mean(v), 5), "max": round(max(v), 5)} for k, v in comp_errs.items()
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True, help="Directory holding adapter.safetensors and manifest.json")
    ap.add_argument("--out", type=Path, required=True, help="Output ComfyUI safetensors path")
    ap.add_argument("--mode", choices=("full", "svd_per_component"), default="svd_per_component")
    ap.add_argument("--sv-fro", type=float, default=0.995, help="Per-component cumulative Frobenius norm threshold")
    ap.add_argument("--min-rank", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    res = convert(
        source_dir=args.source_dir,
        out_path=args.out,
        mode=args.mode,
        sv_fro=args.sv_fro,
        min_rank=args.min_rank,
        device=args.device,
        verify=args.verify,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
