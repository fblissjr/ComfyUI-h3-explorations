#!/usr/bin/env python3
"""Where the two stacks' vision features actually diverge, on released weights.

`canonical/2026-08-24_transformers_comfy_parity.md` closed vision-tower
arithmetic at `2.384e-7` and told the next lane to consume it rather than repeat
it. That probe used a small seeded float32 configuration, and the repo's own
rule says an assumption that has only ever met one implementation is not a
tested assumption -- here, one *configuration*. Driving the real released tower
produced a difference five orders of magnitude larger, so this exists to say
what it is.

The four arms, on identical real patches and the released vision weights:

- **ComfyUI with float32 parameters** and **ComfyUI with bfloat16 parameters**.
  The second is what a deployed H3 encoder actually runs, and the difference
  between them is the finding: `Qwen35VisionModel.fast_pos_embed_interpolate`
  calls `ops.Embedding` with no `out_dtype` and builds its bilinear weights at
  `self.pos_embed.weight.dtype`, so the position-embedding lookup, the
  interpolation coefficients and their product all run at the *stored* dtype
  even though every `manual_cast` linear upcasts to the float32 activation.
- **Transformers with float32 weights** and **with bfloat16 weights**, which
  compute that interpolation at whatever dtype the model was loaded in.

So "ComfyUI runs the vision tower in float32" is true of the linears and false
of the position embedding, and a Transformers arm cannot match the deployed path
on both at once. Which arm to calibrate in is a decision this measurement
informs; it does not make it.

The control is a deliberately wrong weight, which must move the comparison well
past the honest arms' spread. Without it these numbers only prove the script ran.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`), which has both
implementations. Needs a GPU and the released text-encoder directory via
`--source-dir` or `H3_BF16_ENCODER_DIR`; only the vision tensors are read, so
it costs about 2 GiB rather than the whole checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
COMFY = BENCH.parents[2]
REPORT = BENCH / "results" / "2026-08-24_released_vision_precision.json"
VISION_PREFIX = "model.visual."


def released_vision_state(root: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        if name.startswith(VISION_PREFIX):
            by_shard.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for name in sorted(names):
                state[name[len(VISION_PREFIX):]] = handle.get_tensor(name)
    if not state:
        raise SystemExit("the index names no model.visual.* tensors")
    return state


def comfy_tower(state: dict, out_hidden_size: int, dtype: torch.dtype, device: str):
    import comfy.ops
    from comfy.text_encoders.qwen3vl import (
        QWEN3VL_VISION,
        QWEN3VL_VISION_COMMON,
        Qwen3VLVisionModel,
    )

    config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_32b"],
              "out_hidden_size": out_hidden_size}
    model = Qwen3VLVisionModel(config, device="cpu", dtype=dtype,
                               ops=comfy.ops.manual_cast)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    return model.eval().to(device)


def transformers_tower(state: dict, vision_config, dtype: torch.dtype, device: str):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    model = Qwen3VLVisionModel(vision_config).to(dtype).eval()
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    return model.to(device)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a, b = reference.double(), candidate.double()
    difference = a - b
    tokenwise = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {
        "relative_l2": float(difference.norm() / a.norm()),
        "flattened_cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) /
                                  (a.norm() * b.norm())),
        "max_abs_delta": float(difference.abs().max()),
        "tokenwise_cosine_min": float(tokenwise.min()),
        "reference_rms": float((a ** 2).mean().sqrt()),
        "candidate_rms": float((b ** 2).mean().sqrt()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True,
                        help="calibration bundle; its patches are the input")
    parser.add_argument("--row", help="bundle row id; defaults to the first")
    parser.add_argument("--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir or H3_BF16_ENCODER_DIR is required")

    sys.path.insert(0, str(COMFY))
    import nodes  # noqa: F401  ComfyUI's, before any comfy_extras
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig

    root = Path(args.source_dir).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / "presentation.json").read_text())
    record = next((r for r in manifest["rows"] if r["row_id"] == args.row),
                  manifest["rows"][0])
    batch = load_file(bundle / record["batch_file"])
    patches = batch["pixel_values"].to(args.device).float()
    grid = batch["image_grid_thw"].to(args.device)

    config = AutoConfig.from_pretrained(root)
    state = released_vision_state(root)
    print(f"{len(state)} released vision tensors, "
          f"{record['row_id']} grids {grid.tolist()}")

    outputs: dict[str, torch.Tensor] = {}
    arms = [
        ("comfy-float32-params", "comfy", torch.float32),
        ("comfy-bfloat16-params", "comfy", torch.bfloat16),
        ("transformers-float32", "transformers", torch.float32),
        ("transformers-bfloat16", "transformers", torch.bfloat16),
    ]
    for label, kind, dtype in arms:
        if kind == "comfy":
            # float32 patches into both ComfyUI arms, because that is what the
            # deployed path does: `h3_awq_encoder.py::install_source_processors`
            # hands `preprocess_embed` a float32 tensor regardless of how the
            # weights are stored. Casting the input here instead would measure a
            # configuration nothing runs and would hide the position-embedding
            # effect behind a much larger one.
            model = comfy_tower(state, config.text_config.hidden_size, dtype, args.device)
            with torch.no_grad():
                merged = model(patches, grid)[0]
        else:
            model = transformers_tower(state, config.vision_config, dtype, args.device)
            with torch.no_grad():
                merged = model(patches.to(dtype), grid_thw=grid,
                               return_dict=True).pooler_output
        outputs[label] = merged.double().cpu()
        del model
        torch.cuda.empty_cache()
        print(f"  {label:<24} {tuple(outputs[label].shape)}")

    # The deployed arm is the reference: it is what H3 actually receives.
    reference = "comfy-bfloat16-params"
    report: dict = {
        "probe": "released Qwen3-VL vision tower, ComfyUI versus Transformers, "
                 "by parameter precision",
        "path_policy": "logical identifiers only",
        "source": {"logical_name": root.name, "vision_tensors": len(state)},
        "input": {
            "bundle_row": record["row_id"],
            "grids": grid.tolist(),
            "patch_rows": int(patches.shape[0]),
            "merged_tokens": int(outputs[reference].shape[0]),
            "pixel_values_dtype": "float32",
        },
        "reference_arm": reference,
        "reference_note": "the deployed ComfyUI encoder stores vision weights in "
                          "bfloat16, so this arm is what H3 conditioning is "
                          "actually built from",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "against_deployed": {},
        "matched_precision": {},
    }
    for label in outputs:
        if label == reference:
            continue
        report["against_deployed"][label] = _metrics(outputs[reference], outputs[label])
    report["matched_precision"]["comfy-float32 vs transformers-float32"] = _metrics(
        outputs["comfy-float32-params"], outputs["transformers-float32"]
    )
    report["matched_precision"]["comfy-bfloat16 vs comfy-float32"] = _metrics(
        outputs["comfy-bfloat16-params"], outputs["comfy-float32-params"]
    )

    print(f"\nagainst the deployed arm ({reference}):")
    for label, metrics in report["against_deployed"].items():
        print(f"  {label:<24} relL2 {metrics['relative_l2']:.6g}  "
              f"cos {metrics['flattened_cosine']:.8f}")
    print("matched precision:")
    for label, metrics in report["matched_precision"].items():
        print(f"  {label:<44} relL2 {metrics['relative_l2']:.6g}")

    # Control. It guards the tightest claim this probe makes -- that the two
    # implementations agree to `matched_precision` when their weights carry the
    # same dtype -- so it perturbs one weight in the Transformers arm and
    # requires that comparison to move well past its honest value. Grading it
    # against the widest arm instead would have let a 2 percent weight error
    # hide behind the bfloat16 precision gap, which is how the first version of
    # this control passed nothing.
    perturbed = {k: v.clone() for k, v in state.items()}
    target = "merger.linear_fc2.weight" if "merger.linear_fc2.weight" in perturbed else \
        next(k for k in perturbed if k.endswith(".weight") and perturbed[k].ndim == 2)
    # 2 percent, not the 0.1 percent tried first: bfloat16's relative step is
    # about 0.4 percent, so a smaller nudge rounds straight back to the original
    # value and an earlier version of this control reported an exact zero.
    scale = 0.02
    perturbed[target] = perturbed[target].float().mul(1.0 + scale).to(perturbed[target].dtype)
    if torch.equal(perturbed[target], state[target]):
        raise SystemExit("the control perturbation was lost to rounding")
    model = transformers_tower(perturbed, config.vision_config, torch.float32, args.device)
    with torch.no_grad():
        merged = model(patches, grid_thw=grid, return_dict=True).pooler_output.double().cpu()
    del model
    torch.cuda.empty_cache()
    control = _metrics(outputs["comfy-float32-params"], merged)
    honest = report["matched_precision"]["comfy-float32 vs transformers-float32"]["relative_l2"]
    report["control"] = {
        "guards": "comfy-float32 vs transformers-float32",
        "perturbed_tensor": target,
        "perturbed_in_arm": "transformers-float32",
        "relative_scale": scale,
        **control,
        "honest_relative_l2": honest,
        "ratio": control["relative_l2"] / honest if honest else None,
        "detected": control["relative_l2"] > 10 * honest,
    }
    print(f"control: {target} x{1 + scale} in the Transformers arm -> matched-precision "
          f"relL2 {control['relative_l2']:.6g} against the honest {honest:.6g} "
          f"({report['control']['ratio']:.1f}x)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{REPORT.name}")
    if not report["control"]["detected"]:
        print("RED: a 2 percent weight error did not move the matched-precision "
              "comparison by 10x; it cannot distinguish a wrong weight from noise")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
