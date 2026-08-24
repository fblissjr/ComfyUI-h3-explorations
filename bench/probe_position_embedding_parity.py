#!/usr/bin/env python3
"""Where exactly the two position-embedding interpolations diverge.

Gate 1B forensics. The hybrid calibration policy reproduces deployed ComfyUI's
*dtype* configuration -- BF16 position interpolation, FP32 active compute -- and
a control has shown that split is the whole dtype story inside ComfyUI's tower:
an FP32 tower with only `pos_embed` in BF16 reproduces the deployed all-BF16
tower bit-for-bit. Yet the hybrid arm's residual against deployed is
grid-dependent, so something other than dtype differs.

The two implementations build the same bilinear resample differently:

- `comfy/text_encoders/qwen35.py::Qwen35VisionModel.fast_pos_embed_interpolate`
  walks each image with `torch.linspace`, takes `int()` floor and a clipped
  ceil, materialises four index and four weight lists **through Python lists**,
  casts the weights to the stored dtype, and reduces with an explicit
  `pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]`.
- transformers' `get_vision_interpolation_indices_and_weights` is a vectorised
  closed form reduced with `.sum(1)`.

This probe compares them at four points, in order, so a divergence is located
rather than inferred:

1. the gather indices, exactly;
2. the interpolation weights, at FP32 and at BF16;
3. the position-embedding tensor each produces, before the patch embedding is
   added; and
4a. the position-embedding tensor rebuilt from ComfyUI's own coefficients and
   reduced two ways -- `.sum(1)` and the explicit four-term add -- which
   isolates the reduction and nothing else; and
4b. the real tower forward fed those coefficients through the documented
   `interp_indices` / `interp_weights` kwargs, compared on the merged output
   and every DeepStack feature.

The split matters. **4a is arithmetic on coefficients and says nothing about
whether the tower accepts them or what it then produces**; an earlier version of
this file conflated the two, which would have let a coefficient-level result be
read as a tower-level one. 4b is the tower claim.

4a also carries the red control: one index and one coefficient are corrupted,
and the comparison must move. Without it every "exact" verdict above is
unfalsifiable.

**Naming caveat.** The BF16 arms here cast the interpolation coefficients to
BF16. Transformers does not do that on its own -- its helper returns FP32
coefficients whatever dtype the model is -- so these are the *proposed*
BF16-coefficient arms, not generic Transformers BF16.
`bench/h3_calibration_precision.py` keeps `bfloat16_native` separate for the
same reason.

**How ComfyUI's indices and weights are recovered without copying its code.**
Its `weight_tensor` is a local, so the probe substitutes `pos_embed` with a
stub and reads the quantities back out of the real function's own output:

- a stub returning the one-hot basis for tap `k` makes the weighted sum emit
  `w[k]` in channel `k`, at whatever dtype the module holds; and
- a stub returning the scalar source position makes the sum emit that position,
  because bilinear weights sum to one -- which recovers the permutation the
  function applies at the end.

The indices come straight from the stub's input. Everything is produced by
executing the installed function; nothing about its arithmetic is restated here.

Run it with the ComfyUI venv python, which has both implementations. Needs a GPU
and the released text-encoder directory via `--source-dir` or
`H3_BF16_ENCODER_DIR`; it reads only the vision tensors.
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
REPORT = BENCH / "results" / "2026-08-24_position_embedding_parity.json"
VISION_PREFIX = "model.visual."


class _TapStub(torch.nn.Module):
    """Stands in for `pos_embed`, recording its input and returning a probe value.

    `mode="lookup"` passes the real table through, so the function computes what
    it normally computes. `mode="onehot"` returns the basis vector for each tap,
    turning the weighted sum into the weights themselves. `mode="position"`
    returns the source position, which survives the sum because bilinear
    weights sum to one, and so reveals the permutation.
    """

    def __init__(self, table: torch.Tensor, mode: str, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(table.to(dtype), requires_grad=False)
        self.mode = mode
        self.captured: torch.Tensor | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        self.captured = indices.detach().clone()
        taps, count = indices.shape
        width = self.weight.shape[1]
        if self.mode == "lookup":
            return torch.nn.functional.embedding(indices, self.weight)
        out = torch.zeros(taps, count, width, dtype=self.weight.dtype,
                          device=self.weight.device)
        if self.mode == "onehot":
            for tap in range(taps):
                out[tap, :, tap] = 1.0
        elif self.mode == "position":
            out[:, :, 0] = torch.arange(count, dtype=self.weight.dtype,
                                        device=self.weight.device)
        else:
            raise ValueError(f"unknown stub mode {self.mode!r}")
        return out


def comfy_interpolation(table: torch.Tensor, grid: torch.Tensor, dtype: torch.dtype,
                        device: str) -> dict:  # noqa: D401
    """Indices, weights, permutation and pos_embeds, all from the installed function."""
    import comfy.ops
    from comfy.text_encoders.qwen3vl import (
        QWEN3VL_VISION,
        QWEN3VL_VISION_COMMON,
        Qwen3VLVisionModel,
    )

    config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_32b"],
              "out_hidden_size": 5120}
    model = Qwen3VLVisionModel(config, device="cpu", dtype=dtype,
                               ops=comfy.ops.manual_cast).eval().to(device)

    results = {}
    for mode, mode_dtype in (("lookup", dtype), ("onehot", dtype),
                             ("position", torch.float32)):
        stub = _TapStub(table, mode, mode_dtype).to(device)
        model.pos_embed = stub
        with torch.no_grad():
            out = model.fast_pos_embed_interpolate(grid.to(device))
        results[mode] = out
        results[f"{mode}_indices"] = stub.captured

    permutation = results["position"][:, 0].round().long()
    # Extraction invariants. The one-hot and position stubs are clever enough
    # that a silent failure would look like a result, so each recovered
    # quantity is checked for the property it must have if the recovery worked.
    count = int(permutation.numel())
    sorted_permutation = torch.sort(permutation)[0]
    invariants = {
        "permutation_is_bijection": bool(
            torch.equal(sorted_permutation.cpu(),
                        torch.arange(count, device="cpu"))
        ),
        "permutation_elements": count,
        "indices_in_range": bool(
            (results["lookup_indices"] >= 0).all()
            and (results["lookup_indices"] < table.shape[0]).all()
        ),
        "index_max": int(results["lookup_indices"].max()),
        "table_rows": int(table.shape[0]),
        # Bilinear weights must sum to one per position. The error is reported
        # rather than asserted away, because at BF16 it is not exactly zero and
        # the size of it bounds what the recovery can resolve.
        "weight_sum_max_abs_error": float(
            (results["onehot"][:, :4].double().sum(1) - 1.0).abs().max()
        ),
    }
    return {
        "invariants": invariants,
        "indices_raw": results["lookup_indices"],          # [4, N], source order
        "weights_permuted": results["onehot"][:, :4],      # [N, 4], output order
        "permutation": permutation,                        # output pos -> source pos
        "pos_embeds": results["lookup"],                   # [N, D], output order
    }


def transformers_interpolation(table: torch.Tensor, grid: torch.Tensor,
                               vision_config, dtype: torch.dtype, device: str) -> dict:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
        get_vision_interpolation_indices_and_weights,
    )

    model = Qwen3VLVisionModel(vision_config).to(dtype).eval().to(device)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid.to(device),
        num_grid_per_side=model.num_grid_per_side,
        mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners,
        spatial_merge_size=vision_config.spatial_merge_size,
    )
    embedding = torch.nn.Embedding(table.shape[0], table.shape[1], dtype=dtype).to(device)
    embedding.weight.data = table.to(dtype).to(device)
    with torch.no_grad():
        pos_embeds = (embedding(indices) * weights.to(dtype)[:, :, None]).sum(1)
    return {
        "indices": indices,
        "weights": weights,
        "pos_embeds": pos_embeds,
        "interpolation_mode": model.interpolation_mode,
        "align_corners": bool(model.interpolation_align_corners),
        "num_grid_per_side": int(model.num_grid_per_side),
    }


def _compare(name: str, left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape:
        return {"field": name, "shape_left": list(left.shape),
                "shape_right": list(right.shape), "comparable": False}
    if not left.is_floating_point():
        mismatches = int((left != right).sum())
        return {"field": name, "comparable": True, "exactly_equal": mismatches == 0,
                "mismatches": mismatches, "elements": int(left.numel())}
    a, b = left.double(), right.double()
    difference = (a - b).abs()
    return {
        "field": name,
        "comparable": True,
        "exactly_equal": bool(torch.equal(left, right)),
        "max_abs_delta": float(difference.max()),
        "relative_l2": float((a - b).norm() / a.norm()) if float(a.norm()) else None,
        "elements_differing": int((left != right).sum()),
        "elements": int(left.numel()),
    }


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
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--row", action="append",
                        help="bundle row id; repeatable, defaults to every row")
    parser.add_argument("--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir or H3_BF16_ENCODER_DIR is required")

    sys.path.insert(0, str(COMFY))
    import nodes  # noqa: F401
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig

    root = Path(args.source_dir).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / "presentation.json").read_text())
    wanted = args.row or [r["row_id"] for r in manifest["rows"]]
    config = AutoConfig.from_pretrained(root)
    state = released_vision_state(root)
    table = state["pos_embed.weight"]
    print(f"pos_embed table {tuple(table.shape)} {table.dtype}")

    report: dict = {
        "probe": "Qwen3-VL vision position-embedding interpolation, ComfyUI "
                 "versus Transformers, located field by field",
        "path_policy": "logical identifiers only",
        "source": {"logical_name": root.name},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "rows": {},
    }

    for row_id in wanted:
        record = next(r for r in manifest["rows"] if r["row_id"] == row_id)
        batch = load_file(bundle / record["batch_file"])
        grid = batch["image_grid_thw"]
        entry: dict = {"grids": grid.tolist(), "steps": {}}

        comfy32 = comfy_interpolation(table, grid, torch.float32, args.device)
        comfy16 = comfy_interpolation(table, grid, torch.bfloat16, args.device)
        tf32 = transformers_interpolation(table, grid, config.vision_config,
                                          torch.float32, args.device)
        tf16 = transformers_interpolation(table, grid, config.vision_config,
                                          torch.bfloat16, args.device)
        entry["comfy_extraction_invariants"] = {
            "float32": comfy32["invariants"],
            "bfloat16": comfy16["invariants"],
        }
        entry["transformers_interpolation"] = {
            "mode": tf32["interpolation_mode"],
            "align_corners": tf32["align_corners"],
            "num_grid_per_side": tf32["num_grid_per_side"],
        }

        # 1. indices, in the same order. ComfyUI records them in source order and
        #    permutes its result at the end, so its permutation is applied here
        #    rather than assuming the two orders agree.
        permutation = comfy32["permutation"].to(comfy32["indices_raw"].device)
        comfy_indices = comfy32["indices_raw"].T[permutation]      # [N, 4]
        entry["steps"]["1_indices"] = _compare(
            "interp_indices", comfy_indices.cpu(), tf32["indices"].cpu()
        )

        # 2. weights, at both dtypes
        entry["steps"]["2_weights_float32"] = _compare(
            "interp_weights fp32", comfy32["weights_permuted"].cpu(),
            tf32["weights"].to(torch.float32).cpu()
        )
        entry["steps"]["2_weights_bfloat16"] = _compare(
            "interp_weights bf16", comfy16["weights_permuted"].cpu(),
            tf16["weights"].to(torch.bfloat16).cpu()
        )

        # 3. the position-embedding tensor itself, before patch embeddings
        entry["steps"]["3_pos_embeds_float32"] = _compare(
            "pos_embeds fp32", comfy32["pos_embeds"].cpu(), tf32["pos_embeds"].cpu()
        )
        entry["steps"]["3_pos_embeds_bfloat16"] = _compare(
            "pos_embeds bf16", comfy16["pos_embeds"].cpu(), tf16["pos_embeds"].cpu()
        )

        # 4a. the coefficients alone, reduced two ways. This isolates the
        #     reduction; it is NOT a statement about the tower, and an earlier
        #     version of this file's docstring claimed otherwise.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        embedding = torch.nn.Embedding(table.shape[0], table.shape[1],
                                       dtype=torch.bfloat16).to(args.device)
        embedding.weight.data = table.to(torch.bfloat16).to(args.device)
        weights16 = comfy16["weights_permuted"].to(args.device)
        with torch.no_grad():
            taps = embedding(comfy_indices.to(args.device)) * weights16[:, :, None]
            generic = taps.sum(1)
            explicit = ((taps[:, 0] + taps[:, 1]) + taps[:, 2]) + taps[:, 3]
        entry["steps"]["4a_generic_sum_reduction"] = _compare(
            "pos_embeds bf16, comfy coefficients, .sum(1)",
            comfy16["pos_embeds"].cpu(), generic.cpu()
        )
        entry["steps"]["4a_explicit_four_term_reduction"] = _compare(
            "pos_embeds bf16, comfy coefficients, four-term add",
            comfy16["pos_embeds"].cpu(), explicit.cpu()
        )

        # 4a red control. If corrupting one index and one coefficient did not
        # move this comparison, the whole chain above would be measuring
        # nothing, and every "exact" verdict would be unfalsifiable.
        corrupt_indices = comfy_indices.clone()
        corrupt_indices[0, 0] = (int(corrupt_indices[0, 0]) + 1) % table.shape[0]
        corrupt_weights = weights16.clone()
        corrupt_weights[0, 0] = corrupt_weights[0, 0] * 1.5 + 0.25
        with torch.no_grad():
            bad_index = ((embedding(corrupt_indices.to(args.device)) * weights16[:, :, None])
                         .sum(1))
            bad_weight = ((embedding(comfy_indices.to(args.device))
                           * corrupt_weights[:, :, None]).sum(1))
        entry["steps"]["4a_control_corrupt_index"] = _compare(
            "one index corrupted", generic.cpu(), bad_index.cpu()
        )
        entry["steps"]["4a_control_corrupt_weight"] = _compare(
            "one coefficient corrupted", generic.cpu(), bad_weight.cpu()
        )
        del embedding
        torch.cuda.empty_cache()

        # 4b. the real tower forward, fed ComfyUI's own coefficients through
        #     the documented kwargs, compared on the merged output and every
        #     DeepStack feature. 4a says nothing about either.
        state = released_vision_state(root)
        tower = Qwen3VLVisionModel(config.vision_config).to(torch.bfloat16).eval()
        tower.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()},
                              strict=True)
        tower = tower.to(args.device)
        patches = batch["pixel_values"].to(args.device).float()
        with torch.no_grad():
            fed_out = tower(patches, grid_thw=grid.to(args.device), return_dict=True,
                            interp_indices=comfy_indices.to(args.device),
                            interp_weights=weights16)
            plain_out = tower(patches, grid_thw=grid.to(args.device), return_dict=True)
        entry["steps"]["4b_tower_merged_fed_vs_plain"] = _compare(
            "tower pooler_output, comfy coefficients vs library coefficients",
            plain_out.pooler_output.cpu(), fed_out.pooler_output.cpu()
        )
        deepstack_fed = fed_out.deepstack_features
        deepstack_plain = plain_out.deepstack_features
        entry["steps"]["4b_tower_deepstack_fed_vs_plain"] = [
            _compare(f"deepstack[{i}]", a.cpu(), b.cpu())
            for i, (a, b) in enumerate(zip(deepstack_plain, deepstack_fed))
        ]
        entry["steps"]["4b_kwargs_were_consumed"] = {
            "note": "if the tower had ignored interp_indices/interp_weights the "
                    "two forwards would be bit-identical",
            "outputs_differ": not bool(
                torch.equal(plain_out.pooler_output, fed_out.pooler_output)
            ),
        }
        del tower
        torch.cuda.empty_cache()

        report["rows"][row_id] = entry
        print(f"\n{row_id}  grids {grid.tolist()}")
        inv = entry["comfy_extraction_invariants"]["bfloat16"]
        print(f"   extraction: bijection {inv['permutation_is_bijection']}, "
              f"indices in range {inv['indices_in_range']}, "
              f"weight-sum error {inv['weight_sum_max_abs_error']:.3g}")
        for step, result in entry["steps"].items():
            if isinstance(result, list):
                exact = all(r.get("exactly_equal") for r in result)
                print(f"   {step:<42} {'all exact' if exact else 'DIFFER'}")
                continue
            if "outputs_differ" in result:
                print(f"   {step:<42} outputs differ: {result['outputs_differ']}")
                continue
            summary = ("exact" if result.get("exactly_equal") else
                       f"differs on {result.get('elements_differing', '?')}/"
                       f"{result.get('elements', '?')}, relL2 "
                       f"{result.get('relative_l2')}")
            print(f"   {step:<42} {summary}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
