"""Inversion-free EchoEdit+ / AV-FlowEdit sampler for frozen MiniMax H3 (`MiniMaxH3EchoEdit`).

Operationalizes EchoEdit+ (arXiv:2606.15149) OT-coupled velocity-difference transport
plus AV-FlowEdit (arXiv:2610.13072) cross-step correlated noise on H3's ModelSamplingAV
schedule (`shift_v / shift_a` carried audio scale) without patching ComfyUI core.

Core update at native audio sigma `sigma_a(u)` with shared correlated noise `eps_i`:
    z_src_c = (1 - sigma_v) * (s * x_src) + sigma_v * eps_a
    z_tar_c = (1 - sigma_v) * (s * y_a)   + sigma_v * eps_a
    v_a     = ((1 - sigma_a) * latent_a + sigma_a * eps_a - denoised_a_c / s) / sigma_a
    y_a    <- y_a - (sigma_a[i] - sigma_a[i+1]) * edit_scale * (v_tar_a - v_src_a)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from comfy_api.latest import io

try:
    from .h3_rules import on_both_clocks
except ImportError:
    from h3_rules import on_both_clocks

logger = logging.getLogger(__name__)

PASS_KEY = "h3_edit_pass"


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def build_echo_sigmas(
    n_steps: int,
    t_max: float,
    t_min: float = 0.0,
    shift_v: float = 12.0,
    shift_a: float = 3.0,
) -> tuple[list[float], list[float]]:
    """Build strictly descending `(sig_v, sig_a)` schedules of length `n_steps + 1` down to `0.0`.

    `t_max` is the starting native audio sigma in `[0.05, 1.0]`. It is inverted onto the
    uniform base flow grid `u`, linearly spaced from `u_hi` all the way down to `0.0`,
    and mapped back onto `(shift_v, shift_a)` so `time_shift_sigma(sig_v[i], shift_v, shift_a) == sig_a[i]`
    holds identically. `t_min` is accepted for signature compatibility and used by the sampler
    loop as the transition threshold from 2x-NFE EchoEdit+ OT transport to Algorithm 2's
    1x-NFE target-only Euler refinement tail.
    """
    _ = t_min
    steps = max(2, int(n_steps))
    t_hi = max(0.05, min(1.0, float(t_max)))
    u_hi = time_shift_sigma(t_hi, shift_a, 1.0) if t_hi < 1.0 else 1.0
    us = [u_hi * (1.0 - (i / steps)) for i in range(steps + 1)]
    sig_a = [time_shift_sigma(u, 1.0, shift_a) if u > 1e-9 else 0.0 for u in us]
    sig_v = [time_shift_sigma(u, 1.0, shift_v) if u > 1e-9 else 0.0 for u in us]
    return sig_v, sig_a


def _unpack_av(latent: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("MiniMaxH3EchoEdit requires a NestedTensor H3 audio-video latent.")
    streams = list(samples.unbind())
    if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
        raise ValueError(
            "MiniMaxH3EchoEdit expects [video 5D, audio 4D] AV streams, "
            f"got {[tuple(s.shape) for s in streams]}"
        )
    return streams[0], streams[1]


def _mark_cond_pass(conds: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in conds or []:
        c_copy = dict(c)
        mc = dict(c_copy.get("model_conds") or {})
        wrapped = mc.get("minimax_payload")
        if wrapped is not None and isinstance(getattr(wrapped, "cond", None), dict):
            payload = dict(wrapped.cond)
            payload[PASS_KEY] = label
            mc["minimax_payload"] = wrapped.__class__(payload)
            c_copy["model_conds"] = mc
        out.append(c_copy)
    return out


def echo_edit_step(
    *,
    y_a: torch.Tensor,
    x_src_a: torch.Tensor,
    denoised_src_a_carried: torch.Tensor,
    denoised_tar_a_carried: torch.Tensor,
    eps_a: torch.Tensor,
    sigma_a_cur: float,
    sigma_a_next: float,
    audio_scale: float,
    edit_scale: float,
) -> torch.Tensor:
    """Apply one EchoEdit+ OT-coupled velocity-difference step in native audio space."""
    dt_a = float(sigma_a_cur - sigma_a_next)
    if dt_a == 0.0 or edit_scale == 0.0:
        return y_a
    s = float(audio_scale) if audio_scale != 0.0 else 1.0
    sig = max(float(sigma_a_cur), 1e-6)
    x0_src = denoised_src_a_carried / s
    x0_tar = denoised_tar_a_carried / s
    z_src = (1.0 - sig) * x_src_a + sig * eps_a
    z_tar = (1.0 - sig) * y_a + sig * eps_a
    v_src = (z_src - x0_src) / sig
    v_tar = (z_tar - x0_tar) / sig
    return y_a - (dt_a * float(edit_scale)) * (v_tar - v_src)


def euler_target_step(
    *,
    z_a: torch.Tensor,
    denoised_tar_a_carried: torch.Tensor,
    sigma_a_cur: float,
    sigma_a_next: float,
    audio_scale: float,
) -> torch.Tensor:
    """Apply one 1x-NFE target-only Euler refinement step in native audio space (EchoEdit+ Alg. 2)."""
    s = float(audio_scale) if audio_scale != 0.0 else 1.0
    sig = max(float(sigma_a_cur), 1e-6)
    x0_tar = denoised_tar_a_carried / s
    if sigma_a_next <= 1e-9:
        return x0_tar
    v_tar = (z_a - x0_tar) / sig
    return z_a - (sig - float(sigma_a_next)) * v_tar


def next_correlated_noise(
    prev_eps: torch.Tensor | None,
    shape: tuple[int, ...],
    *,
    alpha: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw fresh unit Gaussian noise or cross-step AR(1) correlated noise (`AV-FlowEdit`)."""
    fresh = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)
    a = max(0.0, min(0.999, float(alpha)))
    if prev_eps is None or a == 0.0:
        return fresh
    return a * prev_eps + math.sqrt(max(0.0, 1.0 - a * a)) * fresh


class MiniMaxH3EchoEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EchoEdit",
            display_name="MiniMax H3 EchoEdit+ Sampler",
            category="sampling/minimax",
            description=(
                "Inversion-free EchoEdit+ (OT-coupled velocity difference) + AV-FlowEdit "
                "correlated noise sampler on frozen H3 video+audio latents. Preserves target "
                "melody/rhythm without SDEdit low-denoise compression smear."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("src_conditioning"),
                io.Conditioning.Input("tgt_conditioning"),
                io.Latent.Input("latent"),
                io.Int.Input("steps", default=16, min=4, max=64, step=1),
                io.Float.Input("t_max", default=0.80, min=0.05, max=1.0, step=0.01),
                io.Float.Input("t_min", default=0.0, min=0.0, max=0.95, step=0.01),
                io.Float.Input("edit_scale", default=1.0, min=0.0, max=3.0, step=0.05),
                io.Float.Input("noise_corr", default=0.50, min=0.0, max=0.95, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("freeze_video", default=True),
                io.Latent.Input("prev_latent", optional=True),
                io.Int.Input("context_frames", default=39, min=39, max=345, step=51, optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        src_conditioning,
        tgt_conditioning,
        latent,
        steps: int = 16,
        t_max: float = 0.80,
        t_min: float = 0.0,
        edit_scale: float = 1.0,
        noise_corr: float = 0.50,
        seed: int = 0,
        freeze_video: bool = True,
        prev_latent=None,
        context_frames: int = 39,
    ) -> io.NodeOutput:
        import comfy.nested_tensor
        import comfy.samplers
        import comfy.utils
        import latent_preview
        from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP

        v_orig, a_orig = _unpack_av(latent)
        ctx_a = 0
        if prev_latent is not None:
            ctx_f = int(context_frames)
            if not on_both_clocks(ctx_f):
                raise ValueError(
                    f"MiniMaxH3EchoEdit context_frames must lie on both H3 clocks (39 + 51k), got {ctx_f}"
                )
            ctx_a = (ctx_f * 5) // 3
            if ctx_a <= 0 or ctx_a >= int(a_orig.shape[-1]):
                raise ValueError(
                    f"MiniMaxH3EchoEdit context_frames={ctx_f} ({ctx_a} audio steps) out of range for T_a={int(a_orig.shape[-1])}"
                )
            _, prev_a = _unpack_av(prev_latent)
            if int(prev_a.shape[-1]) < ctx_a:
                raise ValueError(
                    f"MiniMaxH3EchoEdit prev_latent audio length {int(prev_a.shape[-1])} < ctx_a={ctx_a}"
                )
            a_orig = a_orig.clone()
            a_orig[:, :, :, :ctx_a] = prev_a[:, :, :, -ctx_a:].to(device=a_orig.device, dtype=a_orig.dtype)

        ms = model.get_model_object("model_sampling")
        shift_v = float(getattr(ms, "shift", 12.0))
        shift_a = float(getattr(ms, "audio_shift", None) or 3.0)
        sig_v_list, _ = build_echo_sigmas(steps, t_max, t_min, shift_v=shift_v, shift_a=shift_a)
        sigmas_t = torch.tensor(sig_v_list, dtype=torch.float32)

        class _EchoEditGuider(comfy.samplers.CFGGuider):
            pass

        class _EchoEditSampler(comfy.samplers.Sampler):
            def sample(
                self,
                model_wrap,
                sigmas,
                extra_args,
                callback,
                noise,
                latent_image=None,
                denoise_mask=None,
                disable_pbar=False,
            ):
                inner_model = model_wrap.inner_model
                latent_shapes = inner_model.latent_shapes
                sv_shift = float(getattr(inner_model.model_sampling, "shift", 12.0))
                sa_shift = float(getattr(inner_model.model_sampling, "audio_shift", None) or 3.0)
                a_scale = float(inner_model.audio_scale()) if hasattr(inner_model, "audio_scale") else (sv_shift / sa_shift)

                v_dev, a_dev_carried = comfy.utils.unpack_latents(latent_image, latent_shapes)
                a_dev = a_dev_carried / a_scale
                ctx_tail_dev = a_dev[:, :, :, :ctx_a].clone() if ctx_a > 0 else None
                y_v = v_dev.clone()
                y_a = a_dev.clone()

                c_src = _mark_cond_pass(model_wrap.conds["negative"], "original")
                c_tgt = _mark_cond_pass(model_wrap.conds["positive"], "edited")
                extra_opts = extra_args["model_options"]

                gen = torch.Generator(device="cpu").manual_seed(int(extra_args.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)
                prev_eps_v: torch.Tensor | None = None
                prev_eps_a: torch.Tensor | None = None
                static_eps_v: torch.Tensor | None = (
                    torch.randn(tuple(v_dev.shape), generator=gen, device="cpu", dtype=torch.float32).to(
                        device=v_dev.device, dtype=torch.float32
                    )
                    if freeze_video
                    else None
                )
                t_min_clamped = max(0.0, min(float(t_max) - 1e-3, float(t_min)))
                in_tail = False
                z_tail_a: torch.Tensor | None = None
                z_tail_v: torch.Tensor | None = None
                total_steps = len(sigmas) - 1

                for idx in range(total_steps):
                    sig_v = float(sigmas[idx].item())
                    sig_v_next = float(sigmas[idx + 1].item())
                    if sig_v <= 1e-6:
                        continue
                    sig_a = float(time_shift_sigma(sig_v, sv_shift, sa_shift))
                    sig_a_next = float(time_shift_sigma(sig_v_next, sv_shift, sa_shift)) if sig_v_next > 1e-6 else 0.0

                    t_step = torch.tensor([sig_v], device=v_dev.device, dtype=torch.float32)

                    # EchoEdit+ Appendix F Algorithm 2: 1x-NFE target-only Euler refinement tail when t_min > 0.
                    if t_min_clamped > 0.0 and sig_a <= t_min_clamped + 1e-6:
                        if not in_tail:
                            in_tail = True
                            eps_tail_a = torch.randn(
                                tuple(a_dev.shape), generator=gen, device="cpu", dtype=torch.float32
                            ).to(device=a_dev.device, dtype=torch.float32)
                            z_tail_a = (1.0 - sig_a) * y_a + sig_a * eps_tail_a
                            if not freeze_video:
                                eps_tail_v = torch.randn(
                                    tuple(v_dev.shape), generator=gen, device="cpu", dtype=torch.float32
                                ).to(device=v_dev.device, dtype=torch.float32)
                                z_tail_v = (1.0 - sig_v) * y_v + sig_v * eps_tail_v

                        assert z_tail_a is not None
                        v_tar_in = (
                            VISUAL_COND_TIMESTEP * v_dev + (1.0 - VISUAL_COND_TIMESTEP) * static_eps_v
                            if freeze_video
                            else z_tail_v
                        )
                        assert v_tar_in is not None
                        # Carry algebra: z_a_carried = (sig_v / sig_a) * z_a_model => model sees exact z_tail_a.
                        z_tail_a_c = z_tail_a * (sig_v / max(sig_a, 1e-6))
                        if ctx_tail_dev is not None:
                            z_tail_a_c[:, :, :, :ctx_a] = ctx_tail_dev * (sig_v / max(sig_a, 1e-6))
                        x_tar_packed = comfy.utils.pack_latents([v_tar_in, z_tail_a_c])[0]
                        d_tar = comfy.samplers.calc_cond_batch(inner_model, [c_tgt], x_tar_packed, t_step, extra_opts)[0]
                        d_tar_v, d_tar_a_c = comfy.utils.unpack_latents(d_tar, latent_shapes)

                        z_tail_a = euler_target_step(
                            z_a=z_tail_a,
                            denoised_tar_a_carried=d_tar_a_c.float(),
                            sigma_a_cur=sig_a,
                            sigma_a_next=sig_a_next,
                            audio_scale=a_scale,
                        )
                        if ctx_tail_dev is not None:
                            z_tail_a[:, :, :, :ctx_a] = ctx_tail_dev
                        y_a = z_tail_a
                        if not freeze_video:
                            assert z_tail_v is not None
                            if sig_v_next <= 1e-9:
                                z_tail_v = d_tar_v.float()
                            else:
                                v_tar_v = (z_tail_v - d_tar_v.float()) / max(sig_v, 1e-6)
                                z_tail_v = z_tail_v - (sig_v - sig_v_next) * v_tar_v
                            y_v = z_tail_v

                        if callback is not None:
                            cur_x0 = comfy.utils.pack_latents([v_dev if freeze_video else y_v, a_scale * y_a])[0]
                            callback(idx, cur_x0, x_tar_packed, total_steps)
                        continue

                    if freeze_video:
                        assert static_eps_v is not None
                        v_src_in = VISUAL_COND_TIMESTEP * v_dev + (1.0 - VISUAL_COND_TIMESTEP) * static_eps_v
                        v_tar_in = v_src_in
                    else:
                        eps_v = next_correlated_noise(
                            prev_eps_v,
                            tuple(v_dev.shape),
                            alpha=noise_corr,
                            generator=gen,
                            device=v_dev.device,
                            dtype=torch.float32,
                        )
                        prev_eps_v = eps_v
                        v_src_in = (1.0 - sig_v) * v_dev + sig_v * eps_v
                        v_tar_in = (1.0 - sig_v) * y_v + sig_v * eps_v

                    eps_a = next_correlated_noise(
                        prev_eps_a,
                        tuple(a_dev.shape),
                        alpha=noise_corr,
                        generator=gen,
                        device=a_dev.device,
                        dtype=torch.float32,
                    )
                    prev_eps_a = eps_a

                    z_src_a_c = (1.0 - sig_v) * (a_scale * a_dev) + sig_v * eps_a
                    z_tar_a_c = (1.0 - sig_v) * (a_scale * y_a) + sig_v * eps_a
                    if ctx_tail_dev is not None:
                        z_ctx_c = ctx_tail_dev * (sig_v / max(sig_a, 1e-6))
                        z_src_a_c[:, :, :, :ctx_a] = z_ctx_c
                        z_tar_a_c[:, :, :, :ctx_a] = z_ctx_c

                    x_src_packed = comfy.utils.pack_latents([v_src_in, z_src_a_c])[0]
                    x_tar_packed = comfy.utils.pack_latents([v_tar_in, z_tar_a_c])[0]

                    d_src = comfy.samplers.calc_cond_batch(inner_model, [c_src], x_src_packed, t_step, extra_opts)[0]
                    d_tar = comfy.samplers.calc_cond_batch(inner_model, [c_tgt], x_tar_packed, t_step, extra_opts)[0]

                    d_src_v, d_src_a_c = comfy.utils.unpack_latents(d_src, latent_shapes)
                    d_tar_v, d_tar_a_c = comfy.utils.unpack_latents(d_tar, latent_shapes)

                    y_a = echo_edit_step(
                        y_a=y_a,
                        x_src_a=a_dev,
                        denoised_src_a_carried=d_src_a_c.float(),
                        denoised_tar_a_carried=d_tar_a_c.float(),
                        eps_a=eps_a,
                        sigma_a_cur=sig_a,
                        sigma_a_next=sig_a_next,
                        audio_scale=a_scale,
                        edit_scale=edit_scale,
                    )
                    if ctx_tail_dev is not None:
                        y_a[:, :, :, :ctx_a] = ctx_tail_dev
                    if not freeze_video:
                        v_src_v = (v_src_in - d_src_v.float()) / max(sig_v, 1e-6)
                        v_tar_v = (v_tar_in - d_tar_v.float()) / max(sig_v, 1e-6)
                        y_v = y_v - ((sig_v - sig_v_next) * float(edit_scale)) * (v_tar_v - v_src_v)

                    if callback is not None:
                        cur_x0 = comfy.utils.pack_latents([v_dev if freeze_video else y_v, a_scale * y_a])[0]
                        callback(idx, cur_x0, x_tar_packed, total_steps)

                return comfy.utils.pack_latents([v_dev if freeze_video else y_v, a_scale * y_a])[0]

        guider = _EchoEditGuider(model)
        guider.set_conds(tgt_conditioning, src_conditioning)
        guider.set_cfg(1.0)

        latent_nested = comfy.nested_tensor.NestedTensor((v_orig, a_orig))
        noise_nested = comfy.nested_tensor.NestedTensor((torch.zeros_like(v_orig), torch.zeros_like(a_orig)))
        if ctx_a > 0:
            audio_mask = torch.ones((1, 1, 2, int(a_orig.shape[-1])), dtype=a_orig.dtype, device=a_orig.device)
            audio_mask[:, :, :, :ctx_a] = 0.0
            video_mask = torch.zeros_like(v_orig) if freeze_video else torch.ones_like(v_orig)
            mask_nested = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        elif freeze_video:
            mask_nested = comfy.nested_tensor.NestedTensor((torch.zeros_like(v_orig), torch.ones_like(a_orig)))
        else:
            mask_nested = None
        cb = latent_preview.prepare_callback(model, len(sig_v_list) - 1)
        result = guider.sample(
            noise_nested,
            latent_nested,
            _EchoEditSampler(),
            sigmas_t,
            denoise_mask=mask_nested,
            callback=cb,
            seed=int(seed),
        )
        res_v, res_a = result.unbind()
        out_v = v_orig.clone() if freeze_video else res_v.to(device=v_orig.device, dtype=v_orig.dtype)
        out_a = res_a.to(device=a_orig.device, dtype=a_orig.dtype)
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = comfy.nested_tensor.NestedTensor((out_v, out_a))
        return io.NodeOutput(out)
