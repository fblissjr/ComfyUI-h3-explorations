#!/usr/bin/env python3
"""That bench/convert_h3_image_vae_from_diffusers.py writes a diffusers MiniMax-H3-Image VAE in core's layout.

What each case defends, i.e. what breaks if it is deleted:

- `core still reads the layouts the converter writes` -- the image VAE's attention takes q, k and v
  interleaved per head (`comfy/ldm/minimax/vae.py`, `Attention.attend`, the `3 * self.dim_head`
  view) and core's swiglu gates the first half (`comfy/ops.py::_swiglu_eager`). Read from core's
  source, so a change in core turns this red rather than silently scrambling a converted file.
- `q, k and v come back per head` -- a random one-block diffusers decoder converted by `to_comfy`
  and split the way that view splits gives diffusers' own to_q, to_k and to_v outputs. Control: the
  three stacked whole (`[q_all; k_all; v_all]`, the DiT's layout) must NOT pass.
- `the gated FFN computes the same` -- core's swiglu on the converted `ff.w1` equals diffusers'
  SwiGLU (`hidden * silu(gate)`, `[up; gate]`) on the source. Control: `ff.net.0.proj` copied
  unswapped must NOT pass.
- `names and shapes are core's` -- core's own `MiniMaxH3ImageVAE`, built on the meta device,
  converted to diffusers names by diffusers' `convert_video_vae_key` and back by `to_comfy`, gives
  core's persistent state-dict names and shapes exactly. Not graded without `coderef/diffusers`.

Does NOT establish that a converted fine-tune decodes well: the converter's own `--base` report and
a decode are what show that.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_h3_image_vae_convert.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import bootstrap, case, finish, skip  # noqa: E402

bootstrap(cpu=True)

import comfy.ldm.minimax.vae as core_vae  # noqa: E402
import comfy.ops  # noqa: E402

import convert_h3_image_vae_from_diffusers as conv  # noqa: E402

HEADS, DIM_HEAD, HIDDEN = 3, 4, 7
DIM = HEADS * DIM_HEAD
P = "decoder.transformer_blocks.0"
g = torch.Generator().manual_seed(0)


def layouts():
    attend = inspect.getsource(core_vae.Attention.attend)
    assert "view(batch_size, seq_len, -1, 3 * self.dim_head)" in attend, "core's VAE attention no longer views qkv per head"
    assert "torch.chunk(qkv, 3, dim=-1)" in attend, "core's VAE attention no longer chunks q, k, v from each head"
    swiglu = inspect.getsource(comfy.ops._swiglu_eager)
    assert "gate, up = x.chunk(2, dim=-1)" in swiglu and "silu(gate)" in swiglu, "core's swiglu no longer gates the first half"


def diffusers_block() -> dict[str, torch.Tensor]:
    sd = {f"{P}.attn.to_{c}.{s}": torch.randn(*((DIM, DIM) if s == "weight" else (DIM,)), generator=g)
          for c in "qkv" for s in ("weight", "bias")}
    sd[f"{P}.ff.net.0.proj.weight"] = torch.randn(2 * HIDDEN, DIM, generator=g)
    sd[f"{P}.ff.net.0.proj.bias"] = torch.randn(2 * HIDDEN, generator=g)
    return sd


def core_qkv(w, b, x):
    """q, k, v as core's VAE attention takes them: one fused row, viewed [heads, 3 * dim_head], chunked."""
    return (x @ w.T + b).view(HEADS, 3 * DIM_HEAD).chunk(3, dim=-1)


def qkv():
    sd = diffusers_block()
    out = conv.to_comfy(sd, HEADS, DIM_HEAD)
    x = torch.randn(DIM, generator=g)
    want = [(x @ sd[f"{P}.attn.to_{c}.weight"].T + sd[f"{P}.attn.to_{c}.bias"]).view(HEADS, DIM_HEAD) for c in "qkv"]
    got = core_qkv(out[f"{P}.attn.to_qkv.weight"], out[f"{P}.attn.to_qkv.bias"], x)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, want)), "converted to_qkv does not give diffusers' q, k, v"
    w_flat = torch.cat([sd[f"{P}.attn.to_{c}.weight"] for c in "qkv"])
    b_flat = torch.cat([sd[f"{P}.attn.to_{c}.bias"] for c in "qkv"])
    flat = core_qkv(w_flat, b_flat, x)
    assert not all(torch.allclose(a, b, atol=1e-5) for a, b in zip(flat, want)), "control passed: [q_all; k_all; v_all] read as per-head"
    return "control ([q_all; k_all; v_all]) fails as it should"


def ffn():
    sd = diffusers_block()
    out = conv.to_comfy(sd, HEADS, DIM_HEAD)
    x = torch.randn(DIM, generator=g)
    h_src = x @ sd[f"{P}.ff.net.0.proj.weight"].T + sd[f"{P}.ff.net.0.proj.bias"]
    hidden, gate = h_src.chunk(2, dim=-1)
    want = hidden * F.silu(gate)                                   # diffusers' SwiGLU
    got = comfy.ops._swiglu_eager(x @ out[f"{P}.ff.w1.weight"].T + out[f"{P}.ff.w1.bias"])
    assert torch.allclose(got, want, atol=1e-5), "converted ff.w1 does not compute diffusers' SwiGLU"
    unswapped = comfy.ops._swiglu_eager(h_src.clone())
    assert not torch.allclose(unswapped, want, atol=1e-5), "control passed: unswapped ff.net.0.proj"
    return "control (halves left in diffusers' order) fails as it should"


def names():
    dconv = conv.diffusers_converter()
    if dconv is None:
        skip(f"{conv.DIFFUSERS_CONVERTER.relative_to(conv.PACK)} not on disk")
    assert dconv is not None
    with torch.device("meta"):
        model = core_vae.MiniMaxH3ImageVAE(operations=comfy.ops.disable_weight_init, num_layers=2)
    core = {k: t for k, t in model.state_dict().items() if k not in ("latents_mean", "latents_std")}
    config = {"decoder_num_attention_heads": 16, "decoder_attention_head_dim": 64}
    diffusers_sd = {}
    for k, t in core.items():
        for nk, nt in dconv.convert_video_vae_key(k, t, config):
            diffusers_sd[nk] = nt
    back = conv.to_comfy(diffusers_sd, 16, 64)
    assert back.keys() == core.keys(), f"only written {sorted(back.keys() - core.keys())}, only core's {sorted(core.keys() - back.keys())}"
    shapes = [k for k in core if back[k].shape != core[k].shape]
    assert not shapes, f"shape differs: {shapes}"
    return f"{len(core)} tensors (two decoder blocks)"


case("core still reads the layouts the converter writes", layouts)
case("q, k and v come back per head", qkv)
case("the gated FFN computes the same", ffn)
case("names and shapes are core's", names)
sys.exit(finish())
