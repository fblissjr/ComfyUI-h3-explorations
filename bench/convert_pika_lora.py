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
    switch_rank: int = 3,
    switch_scale: float = 1.0,
    body_scale: float = 1.0,
    b_peer: torch.Tensor | None = None,
    a_peer: torch.Tensor | None = None,
    task_boost: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Exact thin SVD of `b @ a` (optionally task-boosted vs `b_peer @ a_peer`) with band scaling."""
    bd = b.double()
    ad = a.double()
    if b_peer is not None and a_peer is not None and task_boost != 0.0:
        bd = torch.cat([(1.0 + task_boost) * bd, (-task_boost) * b_peer.double()], dim=1)
        ad = torch.cat([ad, a_peer.double()], dim=0)
    qb, rb = torch.linalg.qr(bd, mode="reduced")
    qa, ra = torch.linalg.qr(ad.T, mode="reduced")
    m = rb @ ra.T
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    if sv_fro >= 1.0:
        k = int(s.shape[0])
    else:
        cum = torch.cumsum(s.pow(2), dim=0) / s.pow(2).sum().clamp(min=1e-30)
        k = int((cum < (sv_fro ** 2)).sum().item()) + 1
        k = max(min_rank, min(k, int(s.shape[0])))

    sr = max(0, min(int(switch_rank), k))
    if switch_scale != 1.0 or body_scale != 1.0:
        u = u.clone()
        s = s.clone()
        if switch_scale < 0.0:
            u[:, :sr] = -u[:, :sr]
        if body_scale < 0.0:
            u[:, sr:k] = -u[:, sr:k]
        s[:sr] = s[:sr] * abs(switch_scale)
        s[sr:k] = s[sr:k] * abs(body_scale)

    if switch_scale == 0.0 and body_scale != 0.0 and k > sr:
        idx = slice(sr, k)
    elif body_scale == 0.0 and switch_scale != 0.0 and sr > 0:
        idx = slice(0, sr)
    else:
        idx = slice(0, k)

    s_sel = s[idx]
    s_sqrt = s_sel.sqrt()
    b_new = (qb @ (u[:, idx] * s_sqrt)).to(dtype).contiguous()
    a_new = ((s_sqrt.unsqueeze(1) * vh[idx, :]) @ qa.T).to(dtype).contiguous()
    return b_new, a_new, int(s_sel.shape[0])


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
    switch_rank: int = 3,
    switch_scale: float = 1.0,
    body_scale: float = 1.0,
    peer_dir: Path | None = None,
    task_boost: float = 0.0,
    device: str = "cuda",
    verify: bool = True,
) -> dict:
    manifest_path = source_dir / "manifest.json"
    adapter_path = source_dir / "adapter.safetensors"
    peer_adapter_path = (peer_dir / "adapter.safetensors") if peer_dir is not None else None
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
    use_band_or_boost = (switch_scale != 1.0) or (body_scale != 1.0) or (peer_adapter_path is not None and task_boost != 0.0)
    eff_sv_fro = 1.0 if mode == "full" else sv_fro

    with safe_open(adapter_path, framework="pt", device=device) as fo:
        fp = safe_open(peer_adapter_path, framework="pt", device=device) if (peer_adapter_path is not None and task_boost != 0.0) else None
        try:
            src_meta = dict(fo.metadata() or {})
            for ob, cb in blocks:
                bs_qkv, as_qkv = [], []
                for x in ("to_q", "to_k", "to_v"):
                    b_orig = fo.get_tensor(f"{ob}.attn.{x}.lora_B.default.weight")
                    a_orig = fo.get_tensor(f"{ob}.attn.{x}.lora_A.default.weight")
                    b_peer = fp.get_tensor(f"{ob}.attn.{x}.lora_B.default.weight") if fp is not None else None
                    a_peer = fp.get_tensor(f"{ob}.attn.{x}.lora_A.default.weight") if fp is not None else None
                    if mode == "full" and not use_band_or_boost:
                        bn = b_orig.to(torch.bfloat16).contiguous()
                        an = a_orig.to(torch.bfloat16).contiguous()
                    else:
                        bn, an, _ = _qr_svd_truncate(
                            b_orig,
                            a_orig,
                            sv_fro=eff_sv_fro,
                            min_rank=min_rank,
                            switch_rank=switch_rank,
                            switch_scale=switch_scale,
                            body_scale=body_scale,
                            b_peer=b_peer,
                            a_peer=a_peer,
                            task_boost=task_boost,
                        )
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
                    b_peer = fp.get_tensor(f"{ob}.{okind}.lora_B.default.weight") if fp is not None else None
                    a_peer = fp.get_tensor(f"{ob}.{okind}.lora_A.default.weight") if fp is not None else None
                    if swap:
                        h = b_orig.shape[0] // 2
                        b_orig = torch.cat([b_orig[h:], b_orig[:h]], dim=0)
                        if b_peer is not None:
                            b_peer = torch.cat([b_peer[h:], b_peer[:h]], dim=0)
                    if mode == "full" and not use_band_or_boost:
                        bn = b_orig.to(torch.bfloat16).contiguous()
                        an = a_orig.to(torch.bfloat16).contiguous()
                    else:
                        bn, an, _ = _qr_svd_truncate(
                            b_orig,
                            a_orig,
                            sv_fro=eff_sv_fro,
                            min_rank=min_rank,
                            switch_rank=switch_rank,
                            switch_scale=switch_scale,
                            body_scale=body_scale,
                            b_peer=b_peer,
                            a_peer=a_peer,
                            task_boost=task_boost,
                        )
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
        finally:
            if fp is not None:
                del fp

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
        "switch_rank": str(switch_rank),
        "switch_scale": str(switch_scale),
        "body_scale": str(body_scale),
        "task_boost": str(task_boost),
        "conversion": (
            f"mode={mode}; sv_fro={eff_sv_fro}; switch_rank={switch_rank}; "
            f"switch_scale={switch_scale}; body_scale={body_scale}; task_boost={task_boost}; "
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
        "sv_fro": eff_sv_fro,
        "switch_rank": switch_rank,
        "switch_scale": switch_scale,
        "body_scale": body_scale,
        "task_boost": task_boost,
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


_COMP_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("to_q", "attn.to_q", False),
    ("to_k", "attn.to_k", False),
    ("to_v", "attn.to_v", False),
    ("out_proj", "attn.to_out.0", False),
    ("fc1", "ff.net.0.proj", True),
    ("fc2", "ff.net.2", False),
)


def _load_comp(fo, prefix: str, okind: str, swap: bool) -> tuple[torch.Tensor, torch.Tensor]:
    b = fo.get_tensor(f"{prefix}.{okind}.lora_B.default.weight").double()
    a = fo.get_tensor(f"{prefix}.{okind}.lora_A.default.weight").double()
    if swap:
        h = b.shape[0] // 2
        b = torch.cat([b[h:], b[:h]], dim=0)
    return b, a


def _thin_svd(b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    return qb @ u, s, vh @ qa.T


def analyze(
    source_dir: Path,
    peer_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    switch_rank: int = 3,
    device: str = "cuda",
) -> dict:
    """Compute Section 7 spectral, task-difference, and AdaLN alignment metrics."""
    adapter_path = source_dir / "adapter.safetensors"
    peer_adapter_path = (peer_dir / "adapter.safetensors") if peer_dir is not None else None

    comp_names = [c[0] for c in _COMP_SPECS]
    spec_by_comp: dict[str, list[dict[str, float]]] = {k: [] for k in comp_names}
    groups = ("0-9", "10-19", "20-29", "30-39", "40-47", "48-49", "refiner")
    spec_by_group: dict[str, dict[str, list[dict[str, float]]]] = {
        g: {k: [] for k in comp_names} for g in groups
    }

    with safe_open(adapter_path, framework="pt", device=device) as fo:
        all_blocks = [(f"transformer_blocks.{i}", i) for i in range(50)] + [
            (f"token_refiner.refiner_blocks.{i}", -1) for i in range(2)
        ]
        for ob, bidx in all_blocks:
            grp = (
                "refiner"
                if bidx < 0
                else ("48-49" if bidx >= 48 else ("40-47" if bidx >= 40 else f"{(bidx // 10) * 10}-{(bidx // 10) * 10 + 9}"))
            )
            for cname, okind, swap in _COMP_SPECS:
                b, a = _load_comp(fo, ob, okind, swap)
                _, s, _ = _thin_svd(b, a)
                s2 = s.pow(2)
                tot_e = float(s2.sum().item())
                cum = torch.cumsum(s2, dim=0) / max(tot_e, 1e-30)
                sr = min(switch_rank, int(s.shape[0]))
                top_e = float(cum[sr - 1].item())
                tail_e = max(0.0, 1.0 - top_e)
                r99 = int((cum < (0.99 ** 2)).sum().item()) + 1
                r995 = int((cum < (0.995 ** 2)).sum().item()) + 1
                kept_tail_995 = (
                    float(s2[sr:r995].sum().item() / max(s2[sr:].sum().item(), 1e-30))
                    if r995 > sr and s2[sr:].sum().item() > 0
                    else 0.0
                )
                rec = {
                    "fro_norm": float(tot_e ** 0.5),
                    "stable_rank": float(tot_e / max(float(s2[0].item()), 1e-30)),
                    "top1_energy": float(cum[0].item()),
                    "top3_energy": top_e,
                    "top3_norm": float(top_e ** 0.5),
                    "tail_energy": tail_e,
                    "tail_norm": float(tail_e ** 0.5),
                    "r99": float(r99),
                    "r995": float(r995),
                    "tail_energy_kept_svd995": kept_tail_995,
                }
                if bidx >= 0:
                    spec_by_comp[cname].append(rec)
                spec_by_group[grp][cname].append(rec)

    def _agg(recs: list[dict[str, float]]) -> dict[str, float]:
        if not recs:
            return {}
        return {k: round(st.mean(r[k] for r in recs), 5) for k in recs[0]}

    out: dict = {
        "source_adapter": source_dir.name,
        "switch_rank": switch_rank,
        "spectrum_by_component_50blocks": {k: _agg(v) for k, v in spec_by_comp.items()},
        "spectrum_by_block_group": {
            g: {k: _agg(v) for k, v in spec_by_group[g].items()} for g in groups
        },
    }

    if peer_adapter_path is not None and peer_adapter_path.exists():
        task_split: dict[str, list[dict[str, float]]] = {k: [] for k in comp_names}
        trunc_metrics: dict[str, list[dict[str, float]]] = {"0.99": [], "0.995": []}
        rank_mismatch_counts = {"0.99": 0, "0.995": 0}
        with safe_open(adapter_path, framework="pt", device=device) as f_src, safe_open(
            peer_adapter_path, framework="pt", device=device
        ) as f_peer:
            for i in range(50):
                ob = f"transformer_blocks.{i}"
                for cname, okind, swap in _COMP_SPECS:
                    b1, a1 = _load_comp(f_src, ob, okind, swap)
                    b2, a2 = _load_comp(f_peer, ob, okind, swap)
                    u1, s1, vh1 = _thin_svd(b1, a1)
                    u2, s2, vh2 = _thin_svd(b2, a2)
                    w1 = u1 * s1
                    w2 = u2 * s2
                    n1 = float(s1.pow(2).sum().sqrt().item())
                    n2 = float(s2.pow(2).sum().sqrt().item())
                    inner_full = float(((w1.T @ w2) * (vh1 @ vh2.T)).sum().item())
                    cos_full = inner_full / max(n1 * n2, 1e-30)
                    d_sq = max(0.0, n1 * n1 + n2 * n2 - 2.0 * inner_full)
                    d_norm = d_sq ** 0.5

                    sr = min(switch_rank, int(s1.shape[0]))
                    u2_sr = u2[:, :sr]
                    vh2_sr = vh2[:sr, :]
                    core = (u2_sr.T @ w1) @ (vh1 @ vh2_sr.T) - torch.diag(s2[:sr])
                    core_sq = float(core.pow(2).sum().item())
                    d_in_top3 = core_sq ** 0.5
                    d_out_top3 = max(0.0, d_sq - core_sq) ** 0.5

                    task_split[cname].append({
                        "cos_full": cos_full,
                        "rel_diff_full": d_norm / max(n2, 1e-30),
                        "diff_in_top3_subspace_of_total": d_in_top3 / max(n2, 1e-30),
                        "diff_outside_top3_subspace_of_total": d_out_top3 / max(n2, 1e-30),
                    })

                    c1 = torch.cumsum(s1.pow(2), dim=0) / max(n1 * n1, 1e-30)
                    c2 = torch.cumsum(s2.pow(2), dim=0) / max(n2 * n2, 1e-30)
                    for tau_s, tau in (("0.99", 0.99), ("0.995", 0.995)):
                        k1 = int((c1 < (tau ** 2)).sum().item()) + 1
                        k2 = int((c2 < (tau ** 2)).sum().item()) + 1
                        if k1 != k2:
                            rank_mismatch_counts[tau_s] += 1
                        # E_1 = tail(k1:) of source adapter; D = W_src - W_peer
                        e1_sq = float(s1[k1:].pow(2).sum().item())
                        in_e1_w2 = float(((w1[:, k1:].T @ w2) * (vh1[k1:] @ vh2.T)).sum().item())
                        # <E_1, W_src - W_peer> = ||E_1||_F^2 - <E_1, W_peer>
                        proj_on_d = (e1_sq - in_e1_w2) / max(d_sq, 1e-30)
                        t1_sq = float(s1[:k1].pow(2).sum().item())
                        in_t1_w2 = inner_full - in_e1_w2
                        cos_t1_w2 = in_t1_w2 / max((t1_sq ** 0.5) * n2, 1e-30)
                        dist_t1_w2 = (max(0.0, t1_sq + n2 * n2 - 2.0 * in_t1_w2) ** 0.5) / max(d_norm, 1e-30)
                        trunc_metrics[tau_s].append({
                            "proj_loss_onto_task_diff": proj_on_d,
                            "cos_full": cos_full,
                            "cos_trunc_src_to_peer": cos_t1_w2,
                            "dist_trunc_src_to_peer_over_d": dist_t1_w2,
                        })

        out["peer_adapter"] = peer_dir.name if peer_dir is not None else None
        out["task_similarity_and_split_50blocks"] = {k: _agg(v) for k, v in task_split.items()}
        out["truncation_vs_task_diff_300modules"] = {
            tau_s: {
                "rank_mismatch_modules_out_of_300": rank_mismatch_counts[tau_s],
                "proj_loss_onto_task_diff_median": round(
                    st.median(r["proj_loss_onto_task_diff"] for r in recs), 5
                ),
                "proj_loss_onto_task_diff_mean": round(
                    st.mean(r["proj_loss_onto_task_diff"] for r in recs), 5
                ),
                "proj_loss_onto_task_diff_max": round(
                    max(r["proj_loss_onto_task_diff"] for r in recs), 5
                ),
                "cos_full_median": round(
                    st.median(r["cos_full"] for r in recs), 5
                ),
                "cos_trunc_src_to_peer_median": round(
                    st.median(r["cos_trunc_src_to_peer"] for r in recs), 5
                ),
                "cos_trunc_src_to_peer_mean": round(
                    st.mean(r["cos_trunc_src_to_peer"] for r in recs), 5
                ),
                "dist_trunc_src_to_peer_over_d_median": round(
                    st.median(r["dist_trunc_src_to_peer_over_d"] for r in recs), 5
                ),
                "dist_trunc_src_to_peer_over_d_mean": round(
                    st.mean(r["dist_trunc_src_to_peer_over_d"] for r in recs), 5
                ),
            }
            for tau_s, recs in trunc_metrics.items()
        }

    if checkpoint_path is not None and checkpoint_path.exists():
        shift_align: dict[str, dict[str, list[float]]] = {
            k: {b: [] for b in ("v1", "v2", "v3", "v4_5", "v6_128")} for k in ("to_q", "to_k", "to_v", "fc1")
        }
        scale_align: dict[str, list[float]] = {k: [] for k in ("to_q", "to_k", "to_v", "fc1")}
        prop_align: dict[str, dict[str, list[float]]] = {
            "out_proj_from_wv_shift": {"all_v1": [], "b48_49_v1": [], "all_tail": []},
            "fc2_from_wfc1_shift": {"all_v1": [], "b48_49_v1": [], "all_tail": []},
        }
        with safe_open(adapter_path, framework="pt", device=device) as fo, safe_open(
            checkpoint_path, framework="pt", device=device
        ) as fc:
            ck_keys = set(fc.keys())
            ck_pref = "diffusion_model.blocks" if "diffusion_model.blocks.0.adaln_proj.linear.weight" in ck_keys else "blocks"
            for i in range(50):
                ob = f"transformer_blocks.{i}"
                cb = f"{ck_pref}.{i}"
                w_ada = fc.get_tensor(f"{cb}.adaln_proj.linear.weight").double().view(3, 6, 5376, 8)
                b_ada = fc.get_tensor(f"{cb}.adaln_proj.linear.bias").double().view(3, 6, 5376, 1)
                wb_ada = torch.cat([w_ada, b_ada], dim=-1)  # (3, 6, 5376, 9)
                w_norm1 = fc.get_tensor(f"{cb}.norm1.weight").double()
                w_norm2 = fc.get_tensor(f"{cb}.norm2.weight").double()

                for cname, okind, swap, sh_idx, sc_idx, w_norm in (
                    ("to_q", "attn.to_q", False, 0, 1, w_norm1),
                    ("to_k", "attn.to_k", False, 0, 1, w_norm1),
                    ("to_v", "attn.to_v", False, 0, 1, w_norm1),
                    ("fc1", "ff.net.0.proj", True, 3, 4, w_norm2),
                ):
                    b_l, a_l = _load_comp(fo, ob, okind, swap)
                    _, _, vh = _thin_svd(b_l, a_l)
                    sh_vecs = wb_ada[:, sh_idx, :, :].permute(1, 0, 2).reshape(5376, 27)
                    q_sh, _ = torch.linalg.qr(sh_vecs, mode="reduced")
                    p_sh = (vh @ q_sh).pow(2).sum(dim=1)
                    shift_align[cname]["v1"].append(float(p_sh[0].item()))
                    shift_align[cname]["v2"].append(float(p_sh[1].item()))
                    shift_align[cname]["v3"].append(float(p_sh[2].item()))
                    shift_align[cname]["v4_5"].append(float(p_sh[3:5].mean().item()))
                    shift_align[cname]["v6_128"].append(float(p_sh[5:].mean().item()))

                    sc_vecs = (w_norm[:, None, None] * wb_ada[:, sc_idx, :, :].permute(1, 0, 2)).reshape(5376, 27)
                    sc_vecs = torch.cat([w_norm.unsqueeze(1), sc_vecs], dim=1)
                    q_sc, _ = torch.linalg.qr(sc_vecs, mode="reduced")
                    p_sc = (vh @ q_sc).pow(2).sum(dim=1)
                    scale_align[cname].append(float(p_sc[0].item()))

                # Propagated shift into out_proj (via W_V) and fc2 (linearized via W_fc1)
                w_qkv = fc.get_tensor(f"{cb}.attn.qkv_proj.weight").double()
                d_qkv = w_qkv.shape[0] // 3
                w_v = w_qkv[2 * d_qkv :, :]
                sh_msa = wb_ada[:, 0, :, :].permute(1, 0, 2).reshape(5376, 27)
                q_wv, _ = torch.linalg.qr(w_v @ sh_msa, mode="reduced")
                b_o, a_o = _load_comp(fo, ob, "attn.to_out.0", False)
                _, _, vh_o = _thin_svd(b_o, a_o)
                p_o = (vh_o @ q_wv).pow(2).sum(dim=1)
                chance_o = 27.0 / float(d_qkv)
                prop_align["out_proj_from_wv_shift"]["all_v1"].append(float(p_o[0].item() / chance_o))
                prop_align["out_proj_from_wv_shift"]["all_tail"].append(float(p_o[5:].mean().item() / chance_o))
                if i >= 48:
                    prop_align["out_proj_from_wv_shift"]["b48_49_v1"].append(float(p_o[0].item() / chance_o))

                w_fc1 = fc.get_tensor(f"{cb}.mlp.fc1.weight").double()
                h_fc = w_fc1.shape[0] // 2
                sh_mlp = wb_ada[:, 3, :, :].permute(1, 0, 2).reshape(5376, 27)
                fc1_vecs = torch.cat([w_fc1[:h_fc] @ sh_mlp, w_fc1[h_fc:] @ sh_mlp], dim=1)
                q_fc1, _ = torch.linalg.qr(fc1_vecs, mode="reduced")
                b_f2, a_f2 = _load_comp(fo, ob, "ff.net.2", False)
                _, _, vh_f2 = _thin_svd(b_f2, a_f2)
                p_f2 = (vh_f2 @ q_fc1).pow(2).sum(dim=1)
                chance_f2 = 54.0 / float(h_fc)
                prop_align["fc2_from_wfc1_shift"]["all_v1"].append(float(p_f2[0].item() / chance_f2))
                prop_align["fc2_from_wfc1_shift"]["all_tail"].append(float(p_f2[5:].mean().item() / chance_f2))
                if i >= 48:
                    prop_align["fc2_from_wfc1_shift"]["b48_49_v1"].append(float(p_f2[0].item() / chance_f2))

        chance_sh = 27.0 / 5376.0
        chance_sc = 28.0 / 5376.0
        out["adaln_alignment_50blocks"] = {
            "shift_subspace_dim": 27,
            "input_dim": 5376,
            "random_chance_energy": round(chance_sh, 6),
            "shift_alignment_by_component": {
                cname: {
                    band: {
                        "mean_energy": round(st.mean(vals), 5),
                        "max_energy": round(max(vals), 5),
                        "enrichment_vs_chance": round(st.mean(vals) / chance_sh, 2),
                    }
                    for band, vals in bands.items()
                }
                for cname, bands in shift_align.items()
            },
            "scale_v1_alignment_by_component": {
                cname: {
                    "mean_energy": round(st.mean(vals), 5),
                    "enrichment_vs_chance": round(st.mean(vals) / chance_sc, 2),
                }
                for cname, vals in scale_align.items()
            },
            "propagated_shift_enrichment_vs_chance": {
                k: {m: round(st.mean(vals), 2) for m, vals in d.items() if vals}
                for k, d in prop_align.items()
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True, help="Directory holding adapter.safetensors and manifest.json")
    ap.add_argument("--out", type=Path, default=None, help="Output ComfyUI safetensors path (required unless --analyze)")
    ap.add_argument("--mode", choices=("full", "svd_per_component"), default="svd_per_component")
    ap.add_argument("--sv-fro", type=float, default=0.995, help="Per-component cumulative Frobenius norm threshold")
    ap.add_argument("--min-rank", type=int, default=1)
    ap.add_argument("--switch-rank", type=int, default=3, help="Number of top singular values in the switch band (default 3)")
    ap.add_argument("--switch-scale", type=float, default=1.0, help="Multiplier for top switch_rank singular values (0.0 = tail-only)")
    ap.add_argument("--body-scale", type=float, default=1.0, help="Multiplier for singular values above switch_rank (0.0 = spike-only)")
    ap.add_argument("--peer-dir", type=Path, default=None, help="Optional peer task directory for task-difference boost or --analyze")
    ap.add_argument("--task-boost", type=float, default=0.0, help="Boost factor beta in dW + beta*(dW - dW_peer)")
    ap.add_argument("--analyze", action="store_true", help="Run Section 7 spectral, task-difference, and AdaLN alignment analysis")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional MiniMax-H3 base checkpoint for AdaLN alignment in --analyze")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to save JSON output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    if args.analyze:
        res = analyze(
            source_dir=args.source_dir,
            peer_dir=args.peer_dir,
            checkpoint_path=args.checkpoint,
            switch_rank=args.switch_rank,
            device=args.device,
        )
    else:
        if args.out is None:
            ap.error("--out is required unless --analyze is specified")
        res = convert(
            source_dir=args.source_dir,
            out_path=args.out,
            mode=args.mode,
            sv_fro=args.sv_fro,
            min_rank=args.min_rank,
            switch_rank=args.switch_rank,
            switch_scale=args.switch_scale,
            body_scale=args.body_scale,
            peer_dir=args.peer_dir,
            task_boost=args.task_boost,
            device=args.device,
            verify=args.verify,
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
