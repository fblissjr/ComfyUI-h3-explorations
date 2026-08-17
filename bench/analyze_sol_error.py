#!/usr/bin/env python3
"""Gate B0b: decompose Sol-Attn's error into Sparsity vs. Quantization components.

Answers whether a 16-bit PV matmul in sol_attn_exact.cu is worth building by
decomposing total error on real production captures:

    total error  =  sparsity error  +  quantization error
                    (the algorithm)    (INT8 arithmetic)

If Quantization Error / Sparsity Error < 0.05, quantization error is negligible
against algorithmic sparsity error, and Track B (16-bit PV) is retired.

## What this has established

Run on the 2026-08-17 reference-heavy captures, the ratio came back between
roughly 15% and 62% at block level -- far above 0.05. Quantization is a real
share of total error, so Track B is NOT retired on this evidence.

The more useful result is the spread, not the average: within one block at one
step, per-head ratios range from about 24% to about 1695%, and one head's
sparsity error exceeds 1.0 while its neighbours sit near 0.1. A relative L2 of
1.0 is what you get from emitting zeros, so that head's Sol output is further
from dense attention than nothing would be. That heterogeneity is the finding
that should drive `dense_blocks` and any per-head escape work, and it is robust
to every defect listed below because it is about spread rather than absolute
magnitude.

## What is wrong with it, and still worth keeping

Three defects, none of which invalidate the block-level ratio (a ratio of two
quantities computed the same way over the same subset), all of which bound what
the per-head columns can carry.

1. `rho` IS NOT A COSINE. `quant_l2` is normalised by ||out_eager|| while
   `sparsity_l2` and `total_l2` are normalised by ||out_dense||, because
   `rel_l2_error` divides by its second argument. The identity
       ||e_t||^2 = ||e_s||^2 + ||e_q||^2 + 2||e_s||||e_q||rho
   requires all three in the same units. Writing r = ||out_eager||/||out_dense||,
   what is actually computed is
       rho_calc = [||e_t||^2 - ||e_s||^2 - ||e_q||^2/r^2] / (2||e_s||||e_q||/r)
   which equals the true correlation only at r = 1. The numerator is a heavily
   cancelling difference -- for block 49 step 14 it is 2.06e-4 from terms of size
   4e-3, a 20x cancellation -- so a few percent of r-drift moves rho by tens of
   percent. Small |rho| values are therefore not resolved, and the sign of a
   near-zero rho is not trustworthy. Large ones (|rho| ~ 0.45) survive.
   FIX: normalise `quant_l2` by `out_dense`. One line, marked below.

2. `--heads` DEFAULTS TO A SUBSET, AND IT IS NOT A SAMPLE. `decompose_single`
   slices `q[:, :head_subset]` -- the FIRST n heads of 56, not n drawn from 56.
   Every aggregate row this prints is therefore a first-n-heads figure even
   though it is labelled by block, and any claim of the form "head X is the
   worst" surveys only the measured prefix. The banner now prints the subset so
   a log cannot be misread, but the default is still a prefix.

3. NOTHING PROVES THE EAGER REFERENCE IS SOL. `eager_sol_reference` is a
   reimplementation, and it has no calibration gate against the vendored
   `bench/_sol_attn_reference.py`. Its sibling `bench/simulate_track_b_lite.py`
   has exactly that gate and currently REFUSES to report, at rel_l2 0.97,
   because its own reimplementation drifted. This file's version does not share
   the three divergences named in that refusal -- it routes per-query-block on
   `colmean > thr` like the oracle, it corrects the ragged final block through
   `lengths[-1]`, and it has no online-softmax recurrence at all -- so it is
   likely in better shape. "Likely" is not a gate. Until one exists, every
   number here is reproducible but unvalidated, and those are different things.

## How to make it more useful

- ADD THE CALIBRATION GATE FIRST. Copy `calibrate_against_oracle` from
  `bench/simulate_track_b_lite.py`. It is ~30 lines and it is the difference
  between "these numbers reproduce" and "these numbers are about Sol". Nothing
  else on this list matters until it passes.
- RUN `--control`. The `tau=-1e9` dense-limit arm measures the floor of this
  apparatus. Given that a head reported relative L2 above 1.0, that arm is what
  distinguishes "this head is genuinely broken" from "this instrument is". It
  already exists and has never been run.
- The whole 12-row matrix at all 56 heads costs about half an hour of GPU
  (measured: 16.4 s per block/step at 8 heads, and the per-head work is a plain
  loop). Cost is not the reason to measure a prefix.
- Emit JSON alongside the printed table. Every consumer so far has re-typed
  numbers out of a terminal scrollback, which is how a transcription error
  becomes a finding.
- Report per-head magnitudes next to per-head ratios everywhere. A ratio of
  1695% means "this head is nearly perfect and the residue is quantization"
  when its sparsity error is 0.0015. Without the magnitude it reads as alarm.
  The `note` tags below exist for this and should not be dropped.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _sol_attn_reference import BLOCK, _LOG2E, _pool  # noqa: E402


def load_cuda_kernel():
    """Load comfy_kitchen.sol_attn."""
    try:
        ck = importlib.import_module("comfy_kitchen")
    except Exception as exc:
        raise RuntimeError(f"comfy_kitchen not importable: {exc}")
    if not hasattr(ck, "sol_attn"):
        raise RuntimeError("comfy_kitchen has no sol_attn op")
    return ck.sol_attn


def load_capture(filepath: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one `qkv_L*_S*_b{block}_s{step}.pt` as (q, k, v), each [B, H, S, D] in bf16."""
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    q = data["q"]
    k = data["k"]
    v = data["v"]
    return q, k, v


def dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    chunk_q: int = 256,
) -> torch.Tensor:
    """Exact fp32 attention, computed head-by-head and chunked over query tokens.

    Inputs: [B, H, S, D]
    Returns: [B, H, S, D] in float32
    """
    b, h, s, d = q.shape
    if scale is None:
        scale = d ** -0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outs_heads = []

    for head_idx in range(h):
        q_h = q[:, head_idx : head_idx + 1, :, :].to(dtype=torch.float32)
        k_h = k[:, head_idx : head_idx + 1, :, :].to(device=device, dtype=torch.float32)
        v_h = v[:, head_idx : head_idx + 1, :, :].to(device=device, dtype=torch.float32)

        outs_chunks = []
        for start in range(0, s, chunk_q):
            end = min(start + chunk_q, s)
            q_chunk = q_h[:, :, start:end, :].to(device=device)
            # scores: [B, 1, chunk_q, S]
            scores = torch.matmul(q_chunk, k_h.transpose(-1, -2)) * scale
            attn = F.softmax(scores, dim=-1)
            out_chunk = torch.matmul(attn, v_h)
            outs_chunks.append(out_chunk.cpu())
            del scores, attn, out_chunk

        del k_h, v_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outs_heads.append(torch.cat(outs_chunks, dim=2))

    return torch.cat(outs_heads, dim=1)


def eager_sol_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.3,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    centroid_tail: bool = True,
    chunk_q: int = 256,
) -> torch.Tensor:
    """Exact Sol-Attn algorithm in float32 without INT8 quantization, head-by-head and query-chunked.

    Inputs: [B, H, S, D]
    Returns: [B, H, S, D] in float32
    """
    b, h, t, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d ** -0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outs_heads = []

    lengths = torch.full((n,), float(BLOCK), device=device, dtype=torch.float32)
    if n * BLOCK - t:
        lengths[-1] = float(t - (n - 1) * BLOCK)

    idx = torch.arange(n, device=device)
    valid_blk = (idx * BLOCK < t).view(1, 1, 1, n)
    diag_mask = ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    sink_kv_mask = ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    sink_q_mask = ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)

    neg = torch.finfo(torch.float32).min
    qblk_all = torch.arange(t, device=device) // BLOCK

    for head_idx in range(h):
        q_h = q[:, head_idx : head_idx + 1, :, :].to(dtype=torch.float32)
        k_h = k[:, head_idx : head_idx + 1, :, :].to(device=device, dtype=torch.float32)
        v_h = v[:, head_idx : head_idx + 1, :, :].to(device=device, dtype=torch.float32)

        # (B, T, 1, D) for _pool
        fk_bthd = k_h.permute(0, 2, 1, 3).contiguous()
        fv_bthd = v_h.permute(0, 2, 1, 3).contiguous()

        kc = _pool(fk_bthd, n, "mean")                       # (B, N, 1, D)
        vc = _pool(fv_bthd, n, "sum")                        # (B, N, 1, D)

        k_mean = kc.mean(dim=1, keepdim=True)               # (B, 1, 1, D)
        kcc = kc - k_mean
        kc_var = kcc.pow(2).mean(dim=1)                     # (B, 1, D)

        kh = (fk_bthd - k_mean.squeeze(1).unsqueeze(1)).permute(0, 2, 1, 3) # (B, 1, T, D)
        vh = fv_bthd.permute(0, 2, 1, 3)                                    # (B, 1, T, D)
        kch = kcc.permute(0, 2, 1, 3)                                       # (B, 1, N, D)
        vch = vc.permute(0, 2, 1, 3)                                        # (B, 1, N, D)

        fq_bthd = q_h.to(device=device).permute(0, 2, 1, 3).contiguous()
        centroid = _pool(fq_bthd, n, "mean")                                # (B, N, 1, D)
        var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)              # (B, N, 1)
        thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)                  # (B, N, 1)

        outs_chunks = []
        for start in range(0, t, chunk_q):
            end = min(start + chunk_q, t)
            q_chunk_bthd = fq_bthd[:, start:end, :, :]
            qh_chunk = q_chunk_bthd.permute(0, 2, 1, 3)                     # (B, 1, Q_c, D)
            q_len = end - start

            s_tok = (qh_chunk @ kh.transpose(-1, -2)) * log2s
            s_blk = (qh_chunk @ kch.transpose(-1, -2)) * log2s

            qblk_chunk = qblk_all[start:end]
            qblk_min = qblk_chunk[0]
            qblk_max = qblk_chunk[-1]
            num_qblks = int(qblk_max - qblk_min + 1)

            colmean_chunk = torch.zeros(b, 1, num_qblks, n, device=device, dtype=torch.float32)
            rel_qblk = (qblk_chunk - qblk_min).view(1, 1, q_len, 1).expand(b, 1, q_len, n)
            colmean_chunk.scatter_add_(2, rel_qblk, s_blk)
            colmean_chunk = colmean_chunk / lengths.view(1, 1, 1, n)

            thr_chunk = thr[:, qblk_min:qblk_max + 1, :].permute(0, 2, 1).unsqueeze(-1)
            exact_chunk = colmean_chunk > thr_chunk
            exact_chunk |= diag_mask[:, :, qblk_min:qblk_max + 1, :]
            exact_chunk |= sink_kv_mask
            exact_chunk |= sink_q_mask[:, :, qblk_min:qblk_max + 1, :]
            exact_chunk &= valid_blk

            ex_tok = exact_chunk.gather(2, (qblk_chunk - qblk_min).view(1, 1, q_len, 1).expand(b, 1, q_len, n))
            keep_tok = ex_tok.repeat_interleave(BLOCK, dim=-1)[..., :t]
            s_tok = s_tok.masked_fill(~keep_tok, neg)

            if centroid_tail:
                s_blk = colmean_chunk.gather(2, (qblk_chunk - qblk_min).view(1, 1, q_len, 1).expand(b, 1, q_len, n))
            s_blk = s_blk.masked_fill(ex_tok | ~valid_blk, neg)

            logits = torch.cat([s_tok, s_blk], dim=-1)
            p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
            p = p.masked_fill(logits <= neg, 0.0)

            num = p[..., :t] @ vh + p[..., t:] @ vch
            den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, 1, n)).sum(-1)
            out_chunk = (num / den.clamp_min(1e-30).unsqueeze(-1)).permute(0, 2, 1, 3)
            outs_chunks.append(out_chunk.permute(0, 2, 1, 3).cpu())
            del s_tok, s_blk, logits, p, num, den, out_chunk

        del k_h, v_h, kh, vh, kch, vch, fq_bthd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outs_heads.append(torch.cat(outs_chunks, dim=2))

    return torch.cat(outs_heads, dim=1)


def cuda_sol_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.3,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    centroid_tail: bool = True,
) -> torch.Tensor:
    """Run comfy_kitchen.sol_attn CUDA kernel.

    Inputs: [B, H, S, D] in bf16
    Returns: [B, H, S, D] in float32
    """
    sol_fn = load_cuda_kernel()
    device = torch.device("cuda:0")

    # Sol kernel takes (B, T, H, D)
    q_bthd = q.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    k_bthd = k.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    v_bthd = v.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3).contiguous()

    out_bthd = sol_fn(
        q_bthd,
        k_bthd,
        v_bthd,
        tau=tau,
        scale=scale,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
        centroid_tail=centroid_tail,
    )
    # out_bthd is [B, T, H, D] -> return [B, H, T, D] in float32
    return out_bthd.permute(0, 2, 1, 3).to(dtype=torch.float32).cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity accumulated in float64, chunked.

    In fp32 at production tensor size this returns values ABOVE 1.0 -- 1.047609
    was printed on a real run -- which Cauchy-Schwarz forbids, so it was pure
    accumulation error over ~1e8 terms. Measured 2026-08-17 on a synthetic case
    at 100,861,952 elements: fp32 gives 1.021457 where float64 gives 0.998752,
    a +2.27% error that crosses the bound.

    `rel_l2_error` is NOT affected to the same degree (+0.053% on the same
    control) because the two norms' errors largely cancel in the ratio. That is
    why the tables reproduce while this did not. It also means the reported
    figures are supported to about three significant figures, not six.

    Chunked rather than a whole-tensor `.double()`, which would allocate ~800 MB
    per operand at this size.
    """
    a_f, b_f = a.flatten(), b.flatten()
    chunk = 1 << 24
    dot = sq_a = sq_b = 0.0
    for start in range(0, a_f.numel(), chunk):
        x = a_f[start : start + chunk].double()
        y = b_f[start : start + chunk].double()
        dot += float(x @ y)
        sq_a += float(x @ x)
        sq_b += float(y @ y)
    return dot / ((sq_a ** 0.5) * (sq_b ** 0.5) + 1e-12)


def rel_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """||pred - target|| / ||target||.

    Note the denominator is the SECOND argument. Passing a different target per
    call therefore changes units per call, which is what broke `rho` -- use
    `rel_l2_against` when several errors must be comparable.
    """
    pred_f = pred.float()
    target_f = target.float()
    diff_norm = (pred_f - target_f).norm().item()
    target_norm = target_f.norm().item()
    return diff_norm / (target_norm + 1e-12)


def rel_l2_against(pred: torch.Tensor, target: torch.Tensor, denom_norm: float) -> float:
    """||pred - target|| / denom_norm, for errors that must share one denominator.

    The quadrature identity the `rho` below rests on,

        ||e_t||^2 = ||e_s||^2 + ||e_q||^2 + 2||e_s|| ||e_q|| rho

    only holds when all three magnitudes are in the same units. Until 2026-08-17
    the quantization term was normalised by ||out_eager|| while the other two
    used ||out_dense||, so it never held, and the residual that got attributed
    to vector alignment was partly just the denominator mismatch.
    """
    return (pred.float() - target.float()).norm().item() / (denom_norm + 1e-12)


def decompose_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.3,
    head_subset: int = 0,
) -> dict:
    """Run full 3-way decomposition on (q, k, v).

    `head_subset` takes a PREFIX, not a sample: `q[:, :n]` is heads 0..n-1 of
    however many the capture has. Every aggregate this returns is therefore a
    figure over that prefix, even where a caller labels it by block. `heads_total`
    is returned so no printed row can be read as covering the whole block.
    """
    heads_total = q.shape[1]
    if head_subset > 0 and head_subset < q.shape[1]:
        q = q[:, :head_subset, :, :]
        k = k[:, :head_subset, :, :]
        v = v[:, :head_subset, :, :]

    out_dense = dense_reference(q, k, v)
    out_eager = eager_sol_reference(q, k, v, tau=tau)
    out_cuda = cuda_sol_kernel(q, k, v, tau=tau)

    # ONE denominator, ||out_dense||, for all three. See `rel_l2_against`.
    dense_norm = out_dense.float().norm().item()
    sparsity_l2 = rel_l2_against(out_eager, out_dense, dense_norm)
    quant_l2 = rel_l2_against(out_cuda, out_eager, dense_norm)
    total_l2 = rel_l2_against(out_cuda, out_dense, dense_norm)

    sparsity_cos = cosine_sim(out_eager, out_dense)
    quant_cos = cosine_sim(out_cuda, out_eager)
    total_cos = cosine_sim(out_cuda, out_dense)

    ratio = quant_l2 / (sparsity_l2 + 1e-12)

    # Error vector correlation and quadrature analysis
    quad_total = math.sqrt(sparsity_l2**2 + quant_l2**2)
    denom = 2.0 * sparsity_l2 * quant_l2
    rho = (total_l2**2 - (sparsity_l2**2 + quant_l2**2)) / denom if denom > 1e-12 else 0.0

    # Per-head error breakdown
    num_heads = q.shape[1]
    # Per head, the same rule: one denominator per head, that head's dense norm.
    per_head_dense = [out_dense[:, i : i + 1].float().norm().item() for i in range(num_heads)]
    per_head_sparsity = [rel_l2_against(out_eager[:, i : i + 1], out_dense[:, i : i + 1], per_head_dense[i]) for i in range(num_heads)]
    per_head_quant = [rel_l2_against(out_cuda[:, i : i + 1], out_eager[:, i : i + 1], per_head_dense[i]) for i in range(num_heads)]
    per_head_total = [rel_l2_against(out_cuda[:, i : i + 1], out_dense[:, i : i + 1], per_head_dense[i]) for i in range(num_heads)]
    per_head_ratios = [per_head_quant[i] / (per_head_sparsity[i] + 1e-12) for i in range(num_heads)]
    per_head_rho = []
    for i in range(num_heads):
        s_h = per_head_sparsity[i]
        q_h = per_head_quant[i]
        t_h = per_head_total[i]
        den_h = 2.0 * s_h * q_h
        rho_h = (t_h**2 - (s_h**2 + q_h**2)) / den_h if den_h > 1e-12 else 0.0
        per_head_rho.append(rho_h)

    return {
        "sparsity_l2": sparsity_l2,
        "quant_l2": quant_l2,
        "total_l2": total_l2,
        "quad_total": quad_total,
        "rho": rho,
        "sparsity_cos": sparsity_cos,
        "quant_cos": quant_cos,
        "total_cos": total_cos,
        "ratio": ratio,
        "heads_measured": num_heads,
        "heads_total": heads_total,
        "seq_len": q.shape[2],
        "per_head_sparsity": per_head_sparsity,
        "per_head_quant": per_head_quant,
        "per_head_total": per_head_total,
        "per_head_ratios": per_head_ratios,
        "per_head_rho": per_head_rho,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="Path to capture directory")
    parser.add_argument("--blocks", default="0,8,16,24,32,40,49")
    parser.add_argument("--steps", default="3,8,14")
    parser.add_argument("--tau", type=float, default=1.3)
    parser.add_argument("--heads", type=int, default=8, help="Number of heads to measure (0 = all 56)")
    parser.add_argument("--control", action="store_true", help="Run control cases (dense limit vs high tau)")
    args = parser.parse_args()

    capture_dir = Path(os.path.expanduser(args.capture))
    if not capture_dir.is_dir():
        print(f"Error: capture directory not found: {capture_dir}", file=sys.stderr)
        return 1

    pattern = str(capture_dir / "qkv_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Error: no capture files matching {pattern}", file=sys.stderr)
        return 1

    print(f"================================================================================")
    print(f" Sol Error Decomposition: Sparsity (Algorithm) vs. Quantization (INT8 PV)")
    print(f" Capture Directory: {capture_dir}")
    print(f" Total Capture Files Found: {len(files)}")
    print(f" Measuring Heads: {args.heads if args.heads > 0 else 'All 56'} | Tau: {args.tau}")
    print(f"================================================================================\n")

    target_blocks = {int(b) for b in args.blocks.split(",") if b.strip()}
    target_steps = {int(s) for s in args.steps.split(",") if s.strip()}

    results = []

    for fpath in files:
        fname = Path(fpath).name
        # Parse block and step from filename: qkv_L{length}_S{seq}_b{block}_s{step}.pt
        parts = fname.replace(".pt", "").split("_")
        block_val = None
        step_val = None
        for p in parts:
            if p.startswith("b") and p[1:].isdigit():
                block_val = int(p[1:])
            elif p.startswith("s") and p[1:].isdigit():
                step_val = int(p[1:])

        if block_val not in target_blocks or step_val not in target_steps:
            continue

        print(f"Analyzing {fname} (Block {block_val}, Step {step_val})...")
        q, k, v = load_capture(fpath)

        if args.control:
            print("  [CONTROL] Running at Tau = -1e9 (Dense Limit)...")
            res_dense = decompose_single(q, k, v, tau=-1e9, head_subset=args.heads)
            print(f"    Sparsity L2: {res_dense['sparsity_l2']:.6e} | Quant L2: {res_dense['quant_l2']:.6e} | Total L2: {res_dense['total_l2']:.6e}")

            print("  [CONTROL] Running at Tau = 26.0 (High Sparsity Limit)...")
            res_high = decompose_single(q, k, v, tau=26.0, head_subset=args.heads)
            print(f"    Sparsity L2: {res_high['sparsity_l2']:.6e} | Quant L2: {res_high['quant_l2']:.6e} | Total L2: {res_high['total_l2']:.6e}")
            continue

        res = decompose_single(q, k, v, tau=args.tau, head_subset=args.heads)
        res["file"] = fname
        res["block"] = block_val
        res["step"] = step_val
        results.append(res)

        measured, total = res["heads_measured"], res["heads_total"]
        scope = ("ALL heads" if measured == total
                 else f"heads 0-{measured - 1} of {total} -- A PREFIX, NOT A SAMPLE; "
                      f"the rows below describe this prefix, not the block")
        print(f"  Seq Length: {res['seq_len']} tokens | {measured}/{total} heads ({scope})")
        print(f"  Sparsity Error (L2 rel):     {res['sparsity_l2']:.6f}  (Cos: {res['sparsity_cos']:.6f})")
        print(f"  Quantization Error (L2 rel): {res['quant_l2']:.6f}  (Cos: {res['quant_cos']:.6f})")
        print(f"  Total Error (L2 rel):        {res['total_l2']:.6f}  (Cos: {res['total_cos']:.6f})")
        print(f"  Quadrature Pred Total:       {res['quad_total']:.6f}  (Vector alignment rho: {res['rho']:+.4f})")
        print(f"  Quant / Sparsity Ratio:      {res['ratio']:.4f} ({res['ratio']*100:.2f}%)")
        print(f"  Per-Head Distribution:")
        for h_idx in range(res["heads_measured"]):
            s_val = res['per_head_sparsity'][h_idx]
            q_val = res['per_head_quant'][h_idx]
            t_val = res['per_head_total'][h_idx]
            r_val = res['per_head_ratios'][h_idx] * 100
            note = ""
            if s_val < 0.01 and q_val >= 0.01:
                note = " [Low-Sparsity Base: Quantization Dominant]"
            elif s_val < 0.01 and q_val < 0.01:
                note = " [Near-Zero Overall Error]"
            elif r_val > 100.0:
                note = " [Quantization > Sparsity]"
            rho_h = res['per_head_rho'][h_idx]
            print(f"    Head {h_idx}: Sparsity = {s_val:.6f} | Quant = {q_val:.6f} | Total = {t_val:.6f} | Ratio = {r_val:6.2f}% | rho = {rho_h:+.4f}{note}")
        print("-" * 80)

    if not results:
        return 0

    avg_ratio = sum(r["ratio"] for r in results) / len(results)
    max_ratio = max(r["ratio"] for r in results)
    avg_sparsity = sum(r["sparsity_l2"] for r in results) / len(results)
    avg_quant = sum(r["quant_l2"] for r in results) / len(results)
    avg_total = sum(r["total_l2"] for r in results) / len(results)
    avg_quad = sum(r["quad_total"] for r in results) / len(results)
    avg_rho = sum(r["rho"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print(" SUMMARY DECOMPOSITION ACROSS MEASURED PRODUCTION ACTIVATIONS")
    print("=" * 80)
    print(f"Average Sparsity Error:       {avg_sparsity:.6f}")
    print(f"Average Quantization Error:   {avg_quant:.6f}")
    print(f"Average Measured Total Error: {avg_total:.6f}")
    print(f"Average Quadrature Total:     {avg_quad:.6f} (Mean Alignment rho: {avg_rho:+.4f})")
    print(f"Average Quant/Sparsity Ratio: {avg_ratio:.4f} ({avg_ratio * 100:.2f}%)")
    print(f"Max Quant/Sparsity Ratio:     {max_ratio:.4f} ({max_ratio * 100:.2f}%)")
    print("-" * 80)

    if max_ratio < 0.05:
        print("DECISION: Ratio < 0.05 across all blocks/steps.")
        print("-> Quantization error is NEGLIGIBLE against algorithmic sparsity error.")
        print("-> Track B (16-bit PV matmul in sol_attn_exact.cu) is RETIRED as mathematically unjustified.")
    else:
        print("DECISION: Ratio >= 0.05.")
        print("-> Quantization error contributes measurably to total error.")
        print(f"-> Proceed with Track B with {max_ratio*100:.2f}% error ceiling.")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
