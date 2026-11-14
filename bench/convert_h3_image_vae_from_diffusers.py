#!/usr/bin/env python3
"""Write a diffusers MiniMax-H3-Image VAE (AutoencoderKLMiniMaxH3Image) as the file ComfyUI's VAELoader loads.

For VAE fine-tunes published in the diffusers layout. This undoes, key by key,
what diffusers' `scripts/convert_minimax_h3_to_diffusers.py::convert_video_vae_key`
does to the release:

  encoder.down_blocks.{i}.resnets.{j}   -> encoder.down.{i}.block.{j}
  conv_shortcut / downsamplers.0        -> nin_shortcut / downsample
  decoder.proj_in                       -> decoder.x_embedder
  attn.to_q, to_k, to_v                 -> attn.to_qkv, rows interleaved per head [q k v, q k v, ...]
  attn.to_out.0                         -> attn.to_out
  ff.net.0.proj [up; gate]              -> ff.w1 [gate; up]
  ff.net.2                              -> ff.w2
  (dropped by diffusers)                -> decoder.mask_token, zeros (unused at inference)

The latent statistics stay out of the file, as in the release: core supplies
them (`comfy/sd.py`, the MiniMax-H3-Image branch), and the script refuses a
`config.json` whose statistics differ from core's `LATENTS_MEAN` / `LATENTS_STD`.

    <comfy venv python> bench/convert_h3_image_vae_from_diffusers.py --src <diffusers vae folder> --out <file> [--dtype fp32|fp16|bf16]

After writing, the file is converted back with diffusers' own function and
compared with the source bit for bit (fp32 only; a cast file cannot round-trip).
That needs `coderef/diffusers`; without it the line says skipped. `--base <file>`
also reports, per tensor, whether the written file equals an existing ComfyUI
MiniMax-H3-Image VAE and how far the rest moved: a fine-tune with a frozen
encoder shows every encoder tensor equal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PACK = Path(__file__).resolve().parent.parent
DIFFUSERS_CONVERTER = PACK / "coderef/diffusers/scripts/convert_minimax_h3_to_diffusers.py"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
QKV = re.compile(r"(decoder\.transformer_blocks\.\d+)\.attn\.to_([qkv])\.(weight|bias)$")


def load(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def to_comfy(src: dict[str, torch.Tensor], heads: int, dim_head: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    qkv: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, t in src.items():
        m = QKV.match(key)
        if m:
            qkv.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = t
            continue
        new = key
        if new.startswith("encoder.down_blocks."):
            level, rest = new.removeprefix("encoder.down_blocks.").split(".", 1)
            rest = rest.replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1)
            rest = rest.replace("downsamplers.0.", "downsample.", 1)
            new = f"encoder.down.{level}.{rest}"
        new = new.replace("decoder.proj_in.", "decoder.x_embedder.")
        new = new.replace(".attn.to_out.0.", ".attn.to_out.")
        if ".ff.net.0.proj." in new:
            new = new.replace(".ff.net.0.proj.", ".ff.w1.")
            up, gate = t.chunk(2, dim=0)
            t = torch.cat([gate, up], dim=0)
        new = new.replace(".ff.net.2.", ".ff.w2.")
        if new in out:
            raise SystemExit(f"two source keys map to {new}")
        out[new] = t.contiguous()
    for (prefix, suffix), parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise SystemExit(f"{prefix}.attn is missing one of to_q, to_k, to_v")
        rows = [parts[c].reshape(heads, dim_head, *parts[c].shape[1:]) for c in "qkv"]
        fused = torch.cat(rows, dim=1).reshape(3 * heads * dim_head, *parts["q"].shape[1:])
        out[f"{prefix}.attn.to_qkv.{suffix}"] = fused.contiguous()
    out["decoder.mask_token"] = torch.zeros(1, 1, heads * dim_head)
    return out


def diffusers_converter():
    """diffusers' converter module with its two package imports stubbed; only its pure key function is used."""
    if not DIFFUSERS_CONVERTER.is_file():
        return None
    stubs = {
        "diffusers": None, "diffusers.models": None, "diffusers.models.autoencoders": None,
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio": {"AutoencoderKLMiniMaxH3Audio": object},
        "diffusers.utils": None, "diffusers.utils.constants": {"SAFE_WEIGHTS_INDEX_NAME": ""},
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        for name, attrs in stubs.items():
            mod = types.ModuleType(name)
            for attr, value in (attrs or {}).items():
                setattr(mod, attr, value)
            sys.modules[name] = mod
        spec = importlib.util.spec_from_file_location("_diffusers_h3_converter", DIFFUSERS_CONVERTER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def round_trip(out: dict[str, torch.Tensor], src: dict[str, torch.Tensor], config: dict) -> None:
    conv = diffusers_converter()
    if conv is None:
        print(f"round trip through diffusers' converter: skipped ({DIFFUSERS_CONVERTER.relative_to(PACK)} not on disk)")
        return
    back: dict[str, torch.Tensor] = {}
    for key, t in out.items():
        for new, nt in conv.convert_video_vae_key(key, t, config):
            back[new] = nt
    missing, extra = src.keys() - back.keys(), back.keys() - src.keys()
    unequal = [k for k in src.keys() & back.keys() if not torch.equal(src[k], back[k])]
    print(f"round trip through diffusers' converter: {len(src) - len(unequal) - len(missing)} of {len(src)} "
          f"source tensors back bit for bit; unequal {unequal}, missing {sorted(missing)}, extra {sorted(extra)}")


def compare(out: dict[str, torch.Tensor], base_path: Path) -> None:
    base = load(base_path)
    if out.keys() != base.keys():
        print(f"keys differ from {base_path.name}: only here {sorted(out.keys() - base.keys())}, "
              f"only there {sorted(base.keys() - out.keys())}")
    shared = sorted(out.keys() & base.keys())
    shapes = [k for k in shared if out[k].shape != base[k].shape]
    print(f"shape mismatches against {base_path.name}: {shapes}")
    equal, moved = [], []
    for k in shared:
        if k in shapes:
            continue
        a, b = out[k].float(), base[k].float()
        if torch.equal(a, b):
            equal.append(k)
        else:
            moved.append((float((a - b).norm() / b.norm().clamp_min(1e-12)), k))
    groups: dict[str, int] = {}
    for k in equal:
        groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + 1
    print(f"tensors equal to {base_path.name}: {len(equal)} {groups}")
    if moved:
        moved.sort()
        print(f"tensors that moved: {len(moved)}; relative L2 change min {moved[0][0]:.3g} ({moved[0][1]}), "
              f"median {moved[len(moved) // 2][0]:.3g}, max {moved[-1][0]:.3g} ({moved[-1][1]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder holding config.json and diffusion_pytorch_model.safetensors")
    ap.add_argument("--out", type=Path, required=True, help="the .safetensors file to write")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32", help="dtype of every tensor written (default fp32)")
    ap.add_argument("--base", type=Path, help="an existing ComfyUI MiniMax-H3-Image VAE to compare against")
    args = ap.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    if config.get("_class_name") != "AutoencoderKLMiniMaxH3Image":
        ap.error(f"{args.src}/config.json is a {config.get('_class_name')}, not an AutoencoderKLMiniMaxH3Image")
    sys.path.insert(0, str(PACK.parent.parent))
    import comfy.ldm.minimax.vae as core_vae  # noqa: E402  (pure module; touches no device)
    for name, ours in (("latents_mean", core_vae.LATENTS_MEAN), ("latents_std", core_vae.LATENTS_STD)):
        if config.get(name) is not None and config[name] != list(ours):
            ap.error(f"{name} in config.json differs from core's; core would decode this VAE's latents wrongly")

    src = load(args.src / "diffusion_pytorch_model.safetensors")
    out = to_comfy(src, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"])
    dtype = DTYPES[args.dtype]
    if dtype == torch.float16:
        over = [k for k, t in out.items() if t.is_floating_point() and t.abs().max() > torch.finfo(torch.float16).max]
        if over:
            ap.error(f"values beyond the fp16 range in {over}; write fp32 or bf16")
    if args.dtype == "fp32":
        round_trip(out, src, config)
    else:
        print("round trip through diffusers' converter: skipped (a cast file cannot return the source bits)")
    out = {k: t.to(dtype) for k, t in out.items()}
    if args.base:
        compare(out, args.base)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata={"format": "pt"})
    print(f"wrote {args.out}: {len(src)} diffusers tensors as {len(out)} ComfyUI tensors, {args.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
