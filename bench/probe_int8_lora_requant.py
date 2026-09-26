#!/usr/bin/env python3
"""How much of a LoRA survives being baked into the int8 convrot checkpoint.

Under dynamic VRAM, ComfyUI applies a LoRA to a quantized layer in
`comfy/ops.py::resolve_cast_module_with_vbar`: dequantize the int8 weight
(which un-rotates it), add the LoRA delta, and requantize with the layer's own
layout at a recalculated per-row scale with stochastic rounding. The
requantized weight is written back and stays resident, so every step of every
render uses it. This probe runs that exact sequence on real layers and
compares the delta the model ends up with against the delta the LoRA asked
for.

Measured per layer, for each LoRA named on the command line:

- `delta_rel`: the LoRA delta's norm over the base weight's.
- `delta_over_step`: its mean magnitude over the mean int8 step of its row.
- `cos` and `rel_err`: the surviving delta against the requested one, with
  ComfyUI's stochastic rounding and with round-to-nearest.
- `out_rel_err`: the same on the layer's output, for random inputs.

    <comfy venv python> bench/probe_int8_lora_requant.py \\
        --checkpoint <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --lora <models>/loras/h3/<lora>.safetensors [--lora ...] --out <results json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = HERE.parent.parent.parent
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = False
import comfy.quant_ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

LAYERS = [f"blocks.{b}.{m}" for b in (0, 24, 49)
          for m in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]
#: ComfyUI seeds the stochastic rounding per layer from its key; any seed
#: shows the distribution. **Reasoned.**
SEED = 1234


def load_base(f, key, device):
    meta = json.loads(bytes(f.get_tensor(f"{key}.comfy_quant").tolist()).decode())
    q = f.get_tensor(f"{key}.weight").to(device)
    scale = f.get_tensor(f"{key}.weight_scale").to(device)
    layout = comfy.quant_ops.TensorWiseINT8Layout
    params = layout.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape),
                           is_weight=True, convrot=bool(meta.get("convrot")),
                           convrot_groupsize=int(meta.get("convrot_groupsize", 256)))
    return QuantizedTensor(q, "TensorWiseINT8Layout", params), meta


def lora_delta(f, key, device):
    k = f"diffusion_model.{key}"
    a = f.get_tensor(f"{k}.lora_A.weight").to(device, torch.float32)
    b = f.get_tensor(f"{k}.lora_B.weight").to(device, torch.float32)
    alpha = float(f.get_tensor(f"{k}.alpha")) if f"{k}.alpha" in f.keys() else a.shape[0]
    return (b @ a) * (alpha / a.shape[0])


def measure(base, delta, stochastic):
    w0 = base.dequantize().float()
    target = (w0 + delta).to(torch.bfloat16)   # the compute dtype the cast adds in
    q1 = base.requantize_from_float(target, scale="recalculate",
                                    stochastic_rounding=SEED if stochastic else 0)
    got = q1.dequantize().float() - w0
    cos = float(torch.nn.functional.cosine_similarity(got.flatten(), delta.flatten(), dim=0))
    rel = float((got - delta).norm() / delta.norm())
    x = torch.randn(256, delta.shape[1], device=delta.device, generator=torch.Generator(delta.device).manual_seed(0))
    out_rel = float(((x @ got.T) - (x @ delta.T)).norm() / (x @ delta.T).norm())
    return {"cos": round(cos, 5), "rel_err": round(rel, 4), "out_rel_err": round(out_rel, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    with safe_open(args.checkpoint, "pt") as fc:
        for key in LAYERS:
            base, meta = load_base(fc, key, dev)
            w0 = base.dequantize().float()
            step = (base._params.scale.float().flatten()).mean()   # per-row scale is the int8 step
            for lp in args.lora:
                with safe_open(lp, "pt") as fl:
                    delta = lora_delta(fl, key, dev)
                row = {"layer": key, "lora": Path(lp).name, "quant": meta,
                       "delta_rel": round(float(delta.norm() / w0.norm()), 5),
                       "delta_over_step": round(float(delta.abs().mean() / step), 4),
                       "stochastic": measure(base, delta, True),
                       "nearest": measure(base, delta, False)}
                rows.append(row)
                print(json.dumps(row), flush=True)
            del w0, base
    Path(args.out).write_text(json.dumps({"checkpoint": Path(args.checkpoint).name,
                                          "seed": SEED, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
