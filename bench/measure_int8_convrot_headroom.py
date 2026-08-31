#!/usr/bin/env python3
"""What a rebuilt `int8_convrot` encoder could gain over the shipped one.

The question this answers is the one the AWQ lane asked and got wrong for a
different format: **is the shipped artifact at its format's floor, or did the
build leave something on the table?** For W4A16 AWQ the lever was the
calibration population. `int8_convrot` has no calibration step at all -- it is
a deterministic, data-free weight transform -- so the levers are different and
have to be enumerated against the code rather than assumed from the AWQ
experience.

Three arms, all against the released BF16 encoder as reference:

1. **the shipped file**, dequantized through its own layout;
2. **a reproduction** through ComfyUI's own
   ``TensorWiseINT8Layout.quantize(..., per_channel=True, convrot=True)``, from
   an FP32 and from a BF16 source, which says whether the shipped bytes are
   reachable by the stock recipe and which source precision was used;
3. **a sweep of ``convrot_groupsize``** and a no-rotation control, which says
   whether the one exposed knob is worth turning.

Everything is measured on the weights, not on a render. That is deliberate:
`CLAUDE.md` records that a rendered clip cannot A/B a numerical change, and a
weight comparison is controlled by construction.

**What that costs, stated 2026-08-31 after this record was cited past its
scope.** `int8_convrot` is W8A8. `int8_linear` rotates the ACTIVATION online
with the same Hadamard and quantises it per token before an int8 GEMM whose
accumulation is exact, so the error a module carries at run time is two
roundings and every arm below sees one. The groupsize sweep in particular is
weakened by this rather than merely narrowed: the rotation exists to spread
outliers before rounding, and the activation is the side with the outliers. A
flat groupsize result on stored weights does NOT establish that the knob is
inert. See `docs/open_experiments.md` #23.

Needs both encoder files and a CUDA device (the convrot kernels are CUDA-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h3_producer_provenance import producer_provenance  # noqa: E402
from _paths import _comfy_root  # noqa: E402

_COMFY = _comfy_root()
if _COMFY is not None:
    # `comfy.quant_ops` owns the layout; importing it is the whole point of this
    # script, so resolve the checkout the same way every other bench tool does
    # rather than requiring the caller to set PYTHONPATH.
    sys.path.insert(0, str(_COMFY))

#: The module kinds each decoder layer quantizes, and the layers sampled. Both
#: are a sample, not the population: 350 linears at this size is minutes of
#: host-to-device traffic for a number that does not move between layers.
MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.down_proj")
LAYERS = (0, 25, 49)
#: Powers of four that divide every input dimension in this model. The kernel
#: requires a power-of-four group that divides the row length
#: (`comfy/ldm/wan/model_animate2.py`), so this is the legal set, not a taste.
GROUP_SIZES = (64, 256, 1024)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).float().norm() / b.float().norm()).item()


def measure(int8_path: Path, bf16_path: Path) -> dict:
    from comfy.quant_ops import TensorWiseINT8Layout as layout

    per_module: list[dict] = []
    with safe_open(str(int8_path), framework="pt") as fi, \
            safe_open(str(bf16_path), framework="pt") as fb:
        for index in LAYERS:
            for module in MODULES:
                quant_name = f"model.layers.{index}.{module}"
                weight = fb.get_tensor(
                    f"model.language_model.layers.{index}.{module}.weight")
                qdata = fi.get_tensor(quant_name + ".weight").cuda()
                scale = fi.get_tensor(quant_name + ".weight_scale").cuda()
                params = layout.Params(
                    scale=scale, orig_dtype=torch.bfloat16,
                    orig_shape=tuple(weight.shape), is_weight=True,
                    convrot=True, convrot_groupsize=256,
                )
                row = {
                    "layer": index,
                    "module": module,
                    "shape": list(weight.shape),
                    "shipped": _rel(layout.dequantize(qdata, params).cpu(), weight),
                }

                for label, source in (("reproduced_from_fp32", weight.float().cuda()),
                                      ("reproduced_from_bf16", weight.clone().cuda())):
                    produced, produced_params = layout.quantize(
                        source, is_weight=True, per_channel=True,
                        convrot=True, convrot_groupsize=256)
                    row[label] = _rel(
                        layout.dequantize(produced, produced_params).cpu(), weight)
                    # How far the produced bytes are from the shipped ones is a
                    # sharper answer than the error alone: a recipe that lands on
                    # the same integers IS the recipe that built the file.
                    delta = produced.int() - qdata.int()
                    row[label + "_int8_values_differing_pct"] = round(
                        (delta != 0).float().mean().item() * 100, 4)
                    row[label + "_int8_max_abs_diff"] = int(delta.abs().max().item())

                for group in GROUP_SIZES:
                    produced, produced_params = layout.quantize(
                        weight.float().cuda(), is_weight=True, per_channel=True,
                        convrot=True, convrot_groupsize=group)
                    row[f"convrot_groupsize_{group}"] = _rel(
                        layout.dequantize(produced, produced_params).cpu(), weight)

                produced, produced_params = layout.quantize(
                    weight.float().cuda(), is_weight=True, per_channel=True)
                row["no_rotation"] = _rel(
                    layout.dequantize(produced, produced_params).cpu(), weight)

                per_module.append(row)
                del weight, qdata, scale

    keys = [k for k in per_module[0] if isinstance(per_module[0][k], float)]
    means = {k: sum(r[k] for r in per_module) / len(per_module) for k in keys}
    return {"per_module": per_module, "mean_relative_l2": means}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    encoders = (_COMFY / "models" / "text_encoders") if _COMFY else Path(".")
    parser.add_argument("--int8", type=Path,
                        default=encoders / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                        help="the shipped int8_convrot encoder")
    parser.add_argument("--bf16", type=Path,
                        default=encoders / "qwen3vl_32b_minimax_h3_bf16.safetensors",
                        help="the released BF16 encoder, the reference")
    parser.add_argument("--out", type=Path, help="write the record here")
    args = parser.parse_args(argv)

    for path in (args.int8, args.bf16):
        if not path.exists():
            print(f"SKIP  encoder not found: {path.name}")
            return 2
    if not torch.cuda.is_available():
        print("SKIP  the convrot kernels are CUDA-only")
        return 2

    measured = measure(args.int8, args.bf16)
    record = {
        "question": "can a rebuilt int8_convrot encoder beat the shipped one",
        "method": (
            "per-weight relative L2 against the released BF16 encoder, for the "
            "shipped file and for reproductions through ComfyUI's own "
            "TensorWiseINT8Layout.quantize; a sample of module kinds and layers, "
            "not the full 350 linears"
        ),
        "path_policy": "logical identifiers only; encoders named by file name",
        "reference": args.bf16.name,
        "subject": args.int8.name,
        "layers_sampled": list(LAYERS),
        "modules_sampled": list(MODULES),
        "not_established": (
            "end-to-end encoder output fidelity, which is "
            "2026-08-25_four_encoders_holdout_layer50.json's; whether a "
            "per-group scale would help, which this format cannot express; and "
            "-- added 2026-08-31 -- ANY runtime behaviour at all. int8_convrot "
            "is W8A8 and this measures the weight rounding only, so a flat "
            "groupsize result here does not establish that convrot_groupsize is "
            "inert. The rotation's purpose is to spread outliers before "
            "rounding and the activation is the side with the outliers. "
            "docs/open_experiments.md #23"
        ),
        "producer": producer_provenance(__file__),
        **measured,
    }
    text = json.dumps(record, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    means = record["mean_relative_l2"]
    for key in sorted(means):
        print(f"  {key:42s} {means[key]:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
