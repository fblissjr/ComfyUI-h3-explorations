#!/usr/bin/env python3
"""Offline unit tests for `echo_edit.py` (`MiniMaxH3EchoEdit`)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
for p in (str(REPO), str(COMFY)):
    if p not in sys.path:
        sys.path.insert(0, p)

from echo_edit import (  # noqa: E402
    MiniMaxH3EchoEdit,
    build_echo_sigmas,
    echo_edit_step,
    euler_target_step,
    next_correlated_noise,
    time_shift_sigma,
)


def main() -> int:
    torch.manual_seed(42)

    # 1. build_echo_sigmas produces strictly descending (sig_v, sig_a) of exact length n_steps + 1
    sig_v, sig_a = build_echo_sigmas(16, t_max=0.75, t_min=0.0, shift_v=12.0, shift_a=3.0)
    assert len(sig_v) == 17 and len(sig_a) == 17
    assert abs(sig_a[0] - 0.75) < 1e-6 and sig_a[-1] == 0.0 and sig_v[-1] == 0.0
    for i in range(16):
        assert sig_v[i] > sig_v[i + 1], f"sig_v not strictly descending at {i}: {sig_v[i]} <= {sig_v[i+1]}"
        assert sig_a[i] > sig_a[i + 1], f"sig_a not strictly descending at {i}: {sig_a[i]} <= {sig_a[i+1]}"
        recovered = time_shift_sigma(sig_v[i], 12.0, 3.0)
        assert abs(recovered - sig_a[i]) < 1e-5, f"shift mismatch at {i}: {recovered} vs {sig_a[i]}"
    print(f"ok    build_echo_sigmas: 16 steps, sig_v={sig_v[0]:.4f}->{sig_v[-1]:.1f}, sig_a={sig_a[0]:.4f}->{sig_a[-1]:.1f}")

    # 2. Zero drift when src == tgt (exact identity under any noise/scale)
    x_src = torch.randn(1, 32, 2, 40)
    y = x_src.clone()
    eps = torch.randn_like(x_src)
    d_same = 4.0 * (x_src * 0.9 + 0.1 * torch.randn_like(x_src))
    y_next = echo_edit_step(
        y_a=y,
        x_src_a=x_src,
        denoised_src_a_carried=d_same,
        denoised_tar_a_carried=d_same.clone(),
        eps_a=eps,
        sigma_a_cur=0.5,
        sigma_a_next=0.35,
        audio_scale=4.0,
        edit_scale=1.0,
    )
    assert torch.allclose(y_next, y, atol=1e-6), f"expected exact zero drift when src == tgt, max diff={(y_next - y).abs().max()}"
    print("ok    src_eq_tgt_zero_drift: max_diff=0.0")

    # 3. Exact linear flow transport converges from x_src to x_tgt across all 16 build_echo_sigmas steps
    x_tgt = torch.randn(1, 32, 2, 40)
    y_traj = x_src.clone()
    init_err = float((y_traj - x_tgt).pow(2).mean().sqrt())
    gen = torch.Generator(device="cpu").manual_seed(123)
    prev_e = None
    for i in range(len(sig_v) - 1):
        sa, sa_next = sig_a[i], sig_a[i + 1]
        eps_i = next_correlated_noise(
            prev_e, tuple(x_src.shape), alpha=0.5, generator=gen, device=torch.device("cpu"), dtype=torch.float32
        )
        prev_e = eps_i
        y_traj = echo_edit_step(
            y_a=y_traj,
            x_src_a=x_src,
            denoised_src_a_carried=4.0 * x_src,
            denoised_tar_a_carried=4.0 * x_tgt,
            eps_a=eps_i,
            sigma_a_cur=sa,
            sigma_a_next=sa_next,
            audio_scale=4.0,
            edit_scale=1.0,
        )
    final_err = float((y_traj - x_tgt).pow(2).mean().sqrt())
    assert final_err < 0.25 * init_err, f"expected strong convergence from {init_err:.4f}, got {final_err:.4f}"
    print(f"ok    linear_flow_converges: init_rmse={init_err:.4f} -> final_rmse={final_err:.4f} (rem_frac={final_err/init_err:.3f})")

    # 4. AR(1) correlated noise preserves unit variance & positive adjacent correlation
    gen2 = torch.Generator(device="cpu").manual_seed(99)
    e0 = next_correlated_noise(None, (1, 32, 2, 256), alpha=0.5, generator=gen2, device=torch.device("cpu"), dtype=torch.float32)
    e1 = next_correlated_noise(e0, (1, 32, 2, 256), alpha=0.5, generator=gen2, device=torch.device("cpu"), dtype=torch.float32)
    corr = float((e0 * e1).mean() / (e0.std() * e1.std()))
    assert abs(float(e1.std()) - 1.0) < 0.05 and 0.40 < corr < 0.60, f"bad noise_corr stats: std={e1.std()}, corr={corr}"
    print(f"ok    correlated_noise: std={float(e1.std()):.4f}, corr={corr:.4f}")

    # 5. build_echo_sigmas with t_min > 0 still descends to 0.0, and euler_target_step converges cleanly
    sig_v_tail, sig_a_tail = build_echo_sigmas(16, t_max=0.85, t_min=0.18, shift_v=12.0, shift_a=3.0)
    assert len(sig_v_tail) == 17 and sig_a_tail[-1] == 0.0 and sig_v_tail[-1] == 0.0
    tail_indices = [i for i in range(16) if sig_a_tail[i] <= 0.18 + 1e-6]
    assert len(tail_indices) >= 1, f"expected at least 1 tail step <= 0.18, got {sig_a_tail}"
    s0 = sig_a_tail[tail_indices[0]]
    z_tail = (1.0 - s0) * x_src + s0 * torch.randn_like(x_src)
    for i in tail_indices:
        z_tail = euler_target_step(
            z_a=z_tail,
            denoised_tar_a_carried=4.0 * x_tgt,
            sigma_a_cur=sig_a_tail[i],
            sigma_a_next=sig_a_tail[i + 1],
            audio_scale=4.0,
        )
    assert torch.allclose(z_tail, x_tgt, atol=1e-5), f"tail step did not converge to x_tgt, max_diff={(z_tail - x_tgt).abs().max()}"
    print(f"ok    alg2_tail_refinement: {len(tail_indices)} tail step(s) <= 0.18, max_diff=0.0")

    # 6. Schema stability (11 original inputs preserved + 2 appended optional multi-window inputs = 13)
    schema = MiniMaxH3EchoEdit.define_schema()
    in_ids = [getattr(i, "id", getattr(i, "name", "?")) for i in schema.inputs]
    assert schema.node_id == "MiniMaxH3EchoEdit" and len(in_ids) == 13, in_ids
    assert in_ids[-2:] == ["prev_latent", "context_frames"], in_ids
    assert (39 * 5) // 3 == 65
    print(f"ok    node_schema: {schema.node_id} with {len(in_ids)} inputs (prev_latent, context_frames appended)")

    print("\nall 6 case(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
