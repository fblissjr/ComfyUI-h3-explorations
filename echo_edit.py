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

logger = logging.getLogger(__name__)

PASS_KEY = "h3_edit_pass"


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def build_echo_sigmas(
    n_steps: int,
    t_max: float,
    t_min: float,
    shift_v: float = 12.0,
    shift_a: float = 3.0,
) -> tuple[list[float], list[float]]:
    """Build strictly descending `(sig_v, sig_a)` schedules of length `n_steps + 1`.

    `t_max` and `t_min` are native audio sigmas in `[0, 1]`. Both endpoints are
    inverted onto the uniform base flow grid `u`, linearly spaced from `u_hi` down
    to `u_lo`, and mapped back onto `(shift_v, shift_a)` so all `n_steps` run and
    `time_shift_sigma(sig_v[i], shift_v, shift_a) == sig_a[i]` holds identically.
    """
    steps = max(2, int(n_steps))
    t_hi = max(0.05, min(1.0, float(t_max)))
    t_lo = max(0.0, min(t_hi - 1e-3, float(t_min)))
    u_hi = time_shift_sigma(t_hi, shift_a, 1.0) if t_hi < 1.0 else 1.0
    u_lo = time_shift_sigma(t_lo, shift_a, 1.0) if t_lo > 0.0 else 0.0
    us = [u_hi - (u_hi - u_lo) * (i / steps) for i in range(steps + 1)]
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
    ) -> io.NodeOutput:
        import comfy.nested_tensor
        import comfy.samplers
        import comfy.utils
        import latent_preview
        from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP

        v_orig, a_orig = _unpack_av(latent)
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
                y_v = v_dev.clone()
                y_a = a_dev.clone()

                c_src = _mark_cond_pass(model_wrap.conds["negative"], "original")
                c_tgt = _mark_cond_pass(model_wrap.conds["positive"], "edited")
                extra_opts = extra_args["model_options"]

                gen = torch.Generator(device="cpu").manual_seed(int(extra_args.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)
                prev_eps_v: torch.Tensor | None = None
                prev_eps_a: torch.Tensor | None = None
                total_steps = len(sigmas) - 1

                for idx in range(total_steps):
                    sig_v = float(sigmas[idx].item())
                    sig_v_next = float(sigmas[idx + 1].item())
                    if sig_v <= 1e-6:
                        continue
                    sig_a = float(time_shift_sigma(sig_v, sv_shift, sa_shift))
                    sig_a_next = float(time_shift_sigma(sig_v_next, sv_shift, sa_shift)) if sig_v_next > 1e-6 else 0.0

                    eps_v = next_correlated_noise(
                        prev_eps_v,
                        tuple(v_dev.shape),
                        alpha=noise_corr,
                        generator=gen,
                        device=v_dev.device,
                        dtype=torch.float32,
                    )
                    eps_a = next_correlated_noise(
                        prev_eps_a,
                        tuple(a_dev.shape),
                        alpha=noise_corr,
                        generator=gen,
                        device=a_dev.device,
                        dtype=torch.float32,
                    )
                    prev_eps_v, prev_eps_a = eps_v, eps_a

                    if freeze_video:
                        v_in = VISUAL_COND_TIMESTEP * v_dev + (1.0 - VISUAL_COND_TIMESTEP) * eps_v
                    else:
                        v_in = (1.0 - sig_v) * y_v + sig_v * eps_v

                    z_src_a_c = (1.0 - sig_v) * (a_scale * a_dev) + sig_v * eps_a
                    z_tar_a_c = (1.0 - sig_v) * (a_scale * y_a) + sig_v * eps_a

                    x_src_packed = comfy.utils.pack_latents([v_in, z_src_a_c])[0]
                    x_tar_packed = comfy.utils.pack_latents([v_in, z_tar_a_c])[0]
                    t_step = torch.tensor([sig_v], device=v_dev.device, dtype=torch.float32)

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
                    if not freeze_video:
                        v_src_v = (v_in - d_src_v.float()) / max(sig_v, 1e-6)
                        v_tar_v = (v_in - d_tar_v.float()) / max(sig_v, 1e-6)
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
        mask_nested = (
            comfy.nested_tensor.NestedTensor((torch.zeros_like(v_orig), torch.ones_like(a_orig)))
            if freeze_video
            else None
        )
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
        out["samples"] = comfy.nested_tensor.NestedTensor((out_v, out_a))
        return io.NodeOutput(out)
