"""Shared ComfyUI SAM 3.1 helpers moved from `bench/sam3_parity_ladder.py` (2026-10-17, r17_05)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO / "workflows"))

#: measured (`bench/convert_sam3_checkpoint.py` output): the one tensor core's loader refuses outside the server; neither the
#: model nor the text features use it.
TEXT_PROJECTION_KEY = "detector.backbone.language_backbone.encoder.text_projection"
#: inherited: `bench/sam3_precision_arms.py::NUDGE_SEED`, so the nudge is the independent session's, frame for frame.
NUDGE_SEED = 1234


def boot(float32: bool, cpu: bool = False):
    """Import core outside the server. `float32` asks core for float32 model and text encoder and switches TF32 off."""
    sys.argv = sys.argv[:1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(COMFY))
    import comfy.cli_args
    comfy.cli_args.args.cpu = cpu
    if float32:
        comfy.cli_args.args.fp32_unet = True
        comfy.cli_args.args.fp32_text_enc = True
    import torch
    if float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def tf32_selftest(torch) -> dict:
    """A float32 convolution and matmul on the card against float64 on the CPU: a relative error near 1e-6 is true float32,
    near 1e-3 is TF32. The convolution has the neck's shape, 256 channels and 3x3, because a narrower one does not show
    cuDNN's TF32. After `bench/sam3_precision_arms.py::tf32_selftest`, the independent session's."""
    if not torch.cuda.is_available():
        return {}
    g = torch.Generator().manual_seed(7)
    x, w = torch.randn(1, 256, 72, 72, generator=g), torch.randn(256, 256, 3, 3, generator=g) * 0.05
    a, b = torch.randn(512, 1024, generator=g), torch.randn(1024, 512, generator=g)
    conv = torch.nn.functional.conv2d
    rel = lambda got, ref: float((got.double().cpu() - ref).abs().max() / ref.abs().max())   # noqa: E731
    return {"conv_relative_error": rel(conv(x.cuda(), w.cuda(), padding=1), conv(x.double(), w.double(), padding=1)),
            "matmul_relative_error": rel(a.cuda() @ b.cuda(), a.double() @ b.double())}


def environment(torch, selftest: bool = False) -> dict:
    head = subprocess.run(["git", "-C", str(COMFY), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"torch": torch.__version__, "core_commit": head,
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    if selftest:
        out["tf32_selftest"] = tf32_selftest(torch)
    return out


def load_core(ckpt: str):
    import comfy.sd
    import comfy.utils
    import folder_paths
    sd, md = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("checkpoints", ckpt), return_metadata=True)
    sd.pop(TEXT_PROJECTION_KEY, None)
    model, clip, _, _ = comfy.sd.load_state_dict_guess_config(sd, output_vae=False, output_clip=True, output_model=True, metadata=md)
    return model, clip


def nudged(u8, levels: int):
    """Every value moved by `levels` of 255 at random sign, seeded per frame (`bench/sam3_precision_arms.py::nudged`)."""
    import numpy as np
    if levels == 0:
        return u8
    out = np.empty_like(u8)
    for i in range(u8.shape[0]):
        sign = np.random.default_rng(NUDGE_SEED + i).integers(0, 2, size=u8.shape[1:], dtype=np.int8) * 2 - 1
        out[i] = np.clip(u8[i].astype(np.int16) + sign * levels, 0, 255).astype(np.uint8)
    return out
