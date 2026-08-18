"""Priority 3d: Track B-Lite Eager FP16-PV Simulation.

Compares INT8 PV accumulation against unquantized PV on specific heads:
- Block 49: Heads 2 & 3 (quantization-dominated) + Head 4 (sparsity-dominated)
- Block 0: Heads 0 & 1 (entry-layer, quantization-sensitive)

## READ THIS BEFORE QUOTING A "RECOVERY %"

**The `fp16_pv=True` arm is not an FP16 PV matmul.** `int8_quant_pv_head` returns
`eager_sol_reference_head(...)` outright when that flag is set -- an arm with no
quantization anywhere, not one with the PV product widened. The two arms are
otherwise identical in routing, so the difference between them really is
attributable to PV quantization alone, which makes this a legitimate UPPER BOUND
on what a perfect PV matmul could recover. It is not a measurement of one.

Two consequences that were reported as findings and are not:

1. The "FP16 PV error" this prints IS the sparsity error, by construction. It is
   the same quantity `bench/analyze_sol_error.py` calls `sparsity_l2`. So "the
   residual converges to the theoretical sparsity floor" is a restatement of the
   arm's definition, not a result -- the residual IS the floor, and there is
   nothing for it to converge to.

   Stated as a code fact, because that is what it is. It was corroborated
   numerically on 2026-08-17 by matching the figures in a report that claimed
   this file as their source against `analyze_sol_error.py`'s sparsity column,
   which agreed to six digits -- but this file REFUSES to run (see below), so
   its own output was never observed, and that corroboration is about the
   report's numbers, not about a run of this code.
2. `recovery = (err_int8 - err_fp16) / err_int8` therefore reduces to
   `1 - sparsity/total`, which is fully determined by the quant/sparsity ratio
   that `analyze_sol_error.py` already prints. This file adds no information
   beyond that ratio. A head showing "-94% recovery" is a head whose error is 94%
   quantization; those are one fact, not two.

A real FP16 PV matmul carries fp16 rounding error, so its true recovery would be
LESS than what this prints. Treat every figure here as optimistic.

## Current state

This file REFUSES TO RUN. Its `eager_sol_reference_head` disagrees with the
vendored `bench/_sol_attn_reference.py` by rel_l2 0.97, and `calibrate_against_oracle`
below exits rather than printing figures measured against a function that is not
the algorithm it names. The three known divergences are listed in that refusal
message. Fix them, or delete this file and drive the study from the oracle.

## How to make it worth keeping

- Fix the reimplementation until the calibration gate passes. Until then nothing
  here can be quoted, which is the correct state.
- Then make the `fp16_pv` arm actually quantize QK and widen only PV, so it
  measures what it names and stops being an upper bound.
- Or delete the arm and state the bound analytically from `analyze_sol_error.py`'s
  ratio column, which is where it comes from anyway. That is less code and the
  same information, and it would have prevented the misreading above.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Shared rather than reimplemented. `find_one` rejects an ambiguous match instead
# of taking `list(glob(...))[0]`, which returned `os.scandir` order over a pattern
# that matches every sequence length in a directory -- so a directory holding two
# runs picked whichever file the filesystem happened to yield first.
from verify_multistep_capture import find_one  # noqa: E402

BLOCK = 64
_LOG2E = 1.4426950408889634

def _pool(x: torch.Tensor, n: int, op: str = "mean") -> torch.Tensor:
    b, t, h, d = x.shape
    pad = n * BLOCK - t
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
    grouped = x.view(b, n, BLOCK, h, d)
    if op == "mean":
        if pad > 0:
            s = grouped.sum(dim=2)
            denom = torch.full((n,), float(BLOCK), device=x.device, dtype=x.dtype)
            denom[-1] = float(BLOCK - pad)
            return s / denom.view(1, n, 1, 1)
        return grouped.mean(dim=2)
    elif op == "sum":
        return grouped.sum(dim=2)
    raise ValueError(f"unknown op {op}")

def dense_reference_head(q_h: torch.Tensor, k_h: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
    # [1, 1, S, D] -> [1, 1, S, D] float32
    d = q_h.shape[-1]
    scale = d ** -0.5
    scores = (q_h.float() @ k_h.float().transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v_h.float()

def eager_sol_reference_head(q_h: torch.Tensor, k_h: torch.Tensor, v_h: torch.Tensor, tau: float = 1.3) -> torch.Tensor:
    # Algorithmic exact sparse routing in FP32
    b, _, t, d = q_h.shape
    n = (t + BLOCK - 1) // BLOCK
    scale = d ** -0.5
    log2s = scale * _LOG2E

    fk_bthd = k_h.float().permute(0, 2, 1, 3).contiguous()
    fv_bthd = v_h.float().permute(0, 2, 1, 3).contiguous()
    kc = _pool(fk_bthd, n, "mean")
    vc = _pool(fv_bthd, n, "sum")
    k_mean = kc.mean(dim=1, keepdim=True)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)
    kh = (fk_bthd - k_mean.squeeze(1).unsqueeze(1)).permute(0, 2, 1, 3)
    vh = fv_bthd.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)
    vch = vc.permute(0, 2, 1, 3)

    fq_bthd = q_h.float().permute(0, 2, 1, 3).contiguous()
    centroid = _pool(fq_bthd, n, "mean")
    var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

    idx = torch.arange(n)
    diag_mask = ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    qblk_all = torch.arange(t) // BLOCK

    chunk_q = 256
    out_chunks = []
    for q_start in range(0, t, chunk_q):
        q_end = min(q_start + chunk_q, t)
        qc = q_h[:, :, q_start:q_end, :].float()
        qblk = qblk_all[q_start:q_end]
        sc = (qc @ kch.transpose(-1, -2)) * log2s
        # `thr` is (b, n_qblocks, h): ONE threshold per query block. The
        # original `thr.unsqueeze(2)[:, :, q_start:q_end, :]` sliced the size-1
        # head axis as though it were the query axis, so every chunk after the
        # first sliced an empty tensor and the call raised -- i.e. this function
        # could not run on any sequence longer than chunk_q. Index by each
        # query's own block instead, then broadcast over the key-block axis.
        thr_q = thr.permute(0, 2, 1)[:, :, qblk].unsqueeze(-1)
        gate = (sc > thr_q) | diag_mask[:, :, qblk, :]

        acc = torch.zeros((b, 1, q_end - q_start, d), dtype=torch.float32)
        lse = torch.full((b, 1, q_end - q_start, 1), float("-inf"), dtype=torch.float32)

        for kb in range(n):
            mask = gate[:, :, :, kb : kb + 1]
            k_st = kb * BLOCK
            k_en = min(k_st + BLOCK, t)
            
            # Exact route
            if mask.any():
                qk = (qc @ kh[:, :, k_st:k_en, :].transpose(-1, -2)) * log2s
                m_blk = qk.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(lse, m_blk)
                alpha = torch.exp2(lse - m_new)
                p = torch.exp2(qk - m_new)
                acc = acc * alpha + torch.where(mask, p @ vh[:, :, k_st:k_en, :], 0.0)
                lse = m_new + torch.log2(alpha + torch.where(mask, p.sum(dim=-1, keepdim=True), 0.0))
            else:
                sc_b = sc[:, :, :, kb : kb + 1]
                m_new = torch.maximum(lse, sc_b)
                alpha = torch.exp2(lse - m_new)
                p = torch.exp2(sc_b - m_new)
                acc = acc * alpha + p * vch[:, :, kb : kb + 1, :]
                lse = m_new + torch.log2(alpha + p * BLOCK)

        out_chunks.append(acc * torch.exp2(-lse))
    return torch.cat(out_chunks, dim=2)

def int8_quant_pv_head(q_h: torch.Tensor, k_h: torch.Tensor, v_h: torch.Tensor, tau: float = 1.3, fp16_pv: bool = False) -> torch.Tensor:
    # `fp16_pv=True` does NOT widen the PV product inside this function -- it
    # hands the whole call to the unquantized reference. Since the two share
    # routing exactly, the delta is still attributable to PV quantization, so
    # this is a valid upper bound. But the value it returns is the SPARSITY
    # ERROR, not an FP16-PV error, and every "recovery %" computed from it is
    # `1 - sparsity/total`. See the module docstring before quoting any of it.
    if fp16_pv:
        return eager_sol_reference_head(q_h, k_h, v_h, tau=tau)
    
    # Otherwise simulate INT8 PV quantization noise
    b, _, t, d = q_h.shape
    n = (t + BLOCK - 1) // BLOCK
    scale = d ** -0.5
    log2s = scale * _LOG2E

    fk_bthd = k_h.float().permute(0, 2, 1, 3).contiguous()
    fv_bthd = v_h.float().permute(0, 2, 1, 3).contiguous()
    kc = _pool(fk_bthd, n, "mean")
    vc = _pool(fv_bthd, n, "sum")
    k_mean = kc.mean(dim=1, keepdim=True)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)
    kh = (fk_bthd - k_mean.squeeze(1).unsqueeze(1)).permute(0, 2, 1, 3)
    vh = fv_bthd.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)
    vch = vc.permute(0, 2, 1, 3)

    # Quantize V per 64-token block to int8
    v_blocks = []
    v_scales = []
    for kb in range(n):
        k_st = kb * BLOCK
        k_en = min(k_st + BLOCK, t)
        v_sub = vh[:, :, k_st:k_en, :]
        v_max = v_sub.abs().max().clamp(min=1e-6)
        v_scale = v_max / 127.0
        v_q = (v_sub / v_scale).round().clamp(-128, 127)
        v_blocks.append(v_q)
        v_scales.append(v_scale)

    fq_bthd = q_h.float().permute(0, 2, 1, 3).contiguous()
    centroid = _pool(fq_bthd, n, "mean")
    var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

    idx = torch.arange(n)
    diag_mask = ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    qblk_all = torch.arange(t) // BLOCK

    chunk_q = 256
    out_chunks = []
    for q_start in range(0, t, chunk_q):
        q_end = min(q_start + chunk_q, t)
        qc = q_h[:, :, q_start:q_end, :].float()
        qblk = qblk_all[q_start:q_end]
        sc = (qc @ kch.transpose(-1, -2)) * log2s
        # `thr` is (b, n_qblocks, h): ONE threshold per query block. The
        # original `thr.unsqueeze(2)[:, :, q_start:q_end, :]` sliced the size-1
        # head axis as though it were the query axis, so every chunk after the
        # first sliced an empty tensor and the call raised -- i.e. this function
        # could not run on any sequence longer than chunk_q. Index by each
        # query's own block instead, then broadcast over the key-block axis.
        thr_q = thr.permute(0, 2, 1)[:, :, qblk].unsqueeze(-1)
        gate = (sc > thr_q) | diag_mask[:, :, qblk, :]

        acc = torch.zeros((b, 1, q_end - q_start, d), dtype=torch.float32)
        lse = torch.full((b, 1, q_end - q_start, 1), float("-inf"), dtype=torch.float32)

        for kb in range(n):
            mask = gate[:, :, :, kb : kb + 1]
            k_st = kb * BLOCK
            k_en = min(k_st + BLOCK, t)

            if mask.any():
                qk = (qc @ kh[:, :, k_st:k_en, :].transpose(-1, -2)) * log2s
                m_blk = qk.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(lse, m_blk)
                alpha = torch.exp2(lse - m_new)
                p = torch.exp2(qk - m_new)

                # INT8 PV matmul emulation
                p_max = p.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
                p_scale = p_max / 127.0
                p_q = (p / p_scale).round().clamp(0, 127)
                pv_int32 = p_q @ v_blocks[kb]
                pv_recon = pv_int32 * (p_scale * v_scales[kb])

                acc = acc * alpha + torch.where(mask, pv_recon, 0.0)
                lse = m_new + torch.log2(alpha + torch.where(mask, p.sum(dim=-1, keepdim=True), 0.0))
            else:
                sc_b = sc[:, :, :, kb : kb + 1]
                m_new = torch.maximum(lse, sc_b)
                alpha = torch.exp2(lse - m_new)
                p = torch.exp2(sc_b - m_new)
                acc = acc * alpha + p * vch[:, :, kb : kb + 1, :]
                lse = m_new + torch.log2(alpha + p * BLOCK)

        out_chunks.append(acc * torch.exp2(-lse))
    return torch.cat(out_chunks, dim=2)

def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Shapes asserted, not broadcast. Without this, a pred of [1,3,192,64]
    # against a target of [1,1,192,64] broadcasts to a plausible-looking scalar
    # instead of raising -- which is exactly how the query/head axis confusion
    # in the gate produced a printable "recovery %" rather than an error.
    if pred.shape != target.shape:
        raise ValueError(
            f"rel_l2 shape mismatch: pred {tuple(pred.shape)} vs target "
            f"{tuple(target.shape)}. These would broadcast into a meaningless "
            f"number; a divergence here is the finding, not a nuisance.")
    diff = (pred.float() - target.float()).norm().item()
    tgt = target.float().norm().item()
    return diff / (tgt + 1e-12)


def calibrate_against_oracle(tau: float = 1.3, tol: float | None = None) -> float:
    """This file's eager Sol must agree with the vendored oracle, or it reports nothing.

    `bench/_sol_attn_reference.py` is the vendored upstream implementation and
    its docstring says: DO NOT EDIT to fix a disagreement -- if it and something
    else disagree, that is the finding. This module re-implements the same
    algorithm rather than importing it, because it needs to vary the PV
    accumulation dtype, which the oracle does not expose. That is a legitimate
    reason to have a second implementation, and the price of one is a control.

    Returns the rel_l2 between the two on a synthetic case. `main()` refuses to
    print recovery figures when it exceeds `tol`, because a recovery percentage
    measured against a baseline that is not Sol-Attn is not a measurement of
    anything.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _sol_attn_reference as _oracle

    torch.manual_seed(0)
    b, h, t, d = 1, 1, 320, 64
    q, k, v = (torch.randn(b, t, h, d) for _ in range(3))
    qh, kh, vh = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    ora = _oracle.sol_attn(q, k, v, tau=tau, centroid_tail=True)
    ora = ora.permute(0, 2, 1, 3).float()
    mine = eager_sol_reference_head(qh, kh, vh, tau=tau).float()
    return rel_l2(mine, ora)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Required, not defaulted to a home-directory path: this repo's captures
    # live outside the tree and the location differs per machine, so a baked-in
    # default is both a path leak and unusable anywhere else. `H3_CAPTURE` is
    # the variable `bench/red/` already uses for the same purpose.
    parser.add_argument("--capture-dir", default=os.environ.get("H3_CAPTURE"),
                        help="capture directory; defaults to $H3_CAPTURE")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--tol", type=float, default=0.02,
                        help="max rel_l2 between this file's eager Sol and the "
                             "vendored oracle before it refuses to report")
    args = parser.parse_args()

    if not args.capture_dir:
        sys.exit("no capture directory: pass --capture-dir or set H3_CAPTURE.")

    cdir = Path(os.path.expanduser(args.capture_dir))
    print("=" * 80)
    print(f" Priority 3d: Track B-Lite Selective FP16-PV Eager Simulation (Step {args.step})")
    print(f" Source Directory: {cdir}")
    print("=" * 80)

    # Calibration gate. Everything below is a recovery figure measured against
    # this file's own eager Sol; if that is not Sol-Attn, the figures describe
    # nothing. Checked before any capture is read so the failure is cheap.
    drift = calibrate_against_oracle(tol=args.tol)
    if drift > args.tol:
        sys.exit(
            f"\nREFUSING TO REPORT: this file's eager Sol disagrees with the "
            f"vendored oracle\n(`bench/_sol_attn_reference.py`) by rel_l2 "
            f"{drift:.4f}, tolerance {args.tol}.\n\n"
            f"Every 'recovery %' below is measured against that baseline, so "
            f"they would be\nnumbers about a function that is not the algorithm "
            f"they name. Known divergences,\nfrom an xhigh review on 2026-08-17: "
            f"routing per-query on `sc > thr` where the\noracle routes "
            f"per-query-block on `colmean > thr`; weighting the ragged final "
            f"block\nby `p * BLOCK` instead of the oracle's corrected "
            f"`lengths[-1]`; and an online-softmax\nrecurrence that rescales "
            f"`acc` while it is still in the previous block's frame.\n\n"
            f"Fix the reimplementation until this gate passes, or delete it and "
            f"drive the study\nfrom the oracle. Do NOT raise --tol to make this "
            f"go away.")
    print(f"\n  calibration ok: eager Sol agrees with the vendored oracle "
          f"(rel_l2 {drift:.4f})\n")

    # 1. Block 49 Analysis
    f49 = find_one(cdir, 49, args.step)
    t49 = torch.load(f49, map_location="cpu")
    q49, k49, v49 = t49["q"], t49["k"], t49["v"]

    print("\n[Block 49 — Exit Layer]")
    for h_idx in [2, 3, 4]:
        qh = q49[:, h_idx : h_idx + 1, :, :]
        kh = k49[:, h_idx : h_idx + 1, :, :]
        vh = v49[:, h_idx : h_idx + 1, :, :]

        ref_dense = dense_reference_head(qh, kh, vh)
        sol_int8 = int8_quant_pv_head(qh, kh, vh, tau=1.3, fp16_pv=False)
        sol_fp16 = int8_quant_pv_head(qh, kh, vh, tau=1.3, fp16_pv=True)

        err_int8 = rel_l2(sol_int8, ref_dense)
        err_fp16 = rel_l2(sol_fp16, ref_dense)
        recovery = (err_int8 - err_fp16) / err_int8 * 100

        label = "Quant-Dominated Target" if h_idx in [2, 3] else "NULL CONTROL (Sparsity-Dominated)"
        print(f"  Head {h_idx} [{label}]:")
        print(f"    Baseline INT8 PV Total Error: {err_int8:.6f}")
        print(f"    Track B-Lite FP16 PV Error:   {err_fp16:.6f}")
        print(f"    Error Reduction / Recovery:   {recovery:.2f}% ({err_int8:.6f} -> {err_fp16:.6f})")

    # 2. Block 0 Analysis
    f0 = find_one(cdir, 0, args.step)
    t0 = torch.load(f0, map_location="cpu")
    q0, k0, v0 = t0["q"], t0["k"], t0["v"]

    print("\n[Block 0 — Input Layer]")
    for h_idx in [0, 1]:
        qh = q0[:, h_idx : h_idx + 1, :, :]
        kh = k0[:, h_idx : h_idx + 1, :, :]
        vh = v0[:, h_idx : h_idx + 1, :, :]

        ref_dense = dense_reference_head(qh, kh, vh)
        sol_int8 = int8_quant_pv_head(qh, kh, vh, tau=1.3, fp16_pv=False)
        sol_fp16 = int8_quant_pv_head(qh, kh, vh, tau=1.3, fp16_pv=True)

        err_int8 = rel_l2(sol_int8, ref_dense)
        err_fp16 = rel_l2(sol_fp16, ref_dense)
        recovery = (err_int8 - err_fp16) / err_int8 * 100

        print(f"  Head {h_idx} [Entry Quantization-Sensitive]:")
        print(f"    Baseline INT8 PV Total Error: {err_int8:.6f}")
        print(f"    Track B-Lite FP16 PV Error:   {err_fp16:.6f}")
        print(f"    Error Reduction / Recovery:   {recovery:.2f}% ({err_int8:.6f} -> {err_fp16:.6f})")

    print("\n" + "=" * 80)
    print(" P3d SIMULATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
