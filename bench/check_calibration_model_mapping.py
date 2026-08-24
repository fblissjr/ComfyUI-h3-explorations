#!/usr/bin/env python3
"""Prove the released checkpoint maps and loads strictly into the calibration model.

Gate 1, checkpoint half. Everything else in this gate is about the inputs; this
is about the model those inputs are fed to. `active_plan.md` asks for
"full-checkpoint strict loading into the Transformers calibration model", and
the handoff adds that small random-model arithmetic parity does not cover this
boundary -- so this uses the real released weights.

Two arms, cheap first:

1. **Mapping.** The model is instantiated on the meta device from the released
   `config.json`, and its parameter and buffer inventory is compared against the
   checkpoint index -- names, shapes and dtypes, in both directions. A tensor the
   model wants and the checkpoint lacks is a missing key; one the checkpoint
   carries and the model does not want is unexpected. Neither is tolerated. This
   reads only safetensors headers, so it costs seconds and no memory.

2. **Load.** The same weights are then loaded for real with
   `from_pretrained(..., output_loading_info=True)`, and its reported missing,
   unexpected and mismatched keys must all be empty. That call maps safetensors
   lazily, so returning quickly proves the keys resolved and nothing more; the
   arm therefore also touches the first and last element of every state-dict
   tensor, which forces each mapping to a real page and would catch a tensor
   left on `meta` or a shard that cannot be read. `--no-load` skips the arm and
   says so in the report, because a mapping-only run is a weaker claim and must
   not read like the full one.

It also records what the candidate is allowed to quantize: all 64 decoder
layers are present, and the input embedding, the vision tower and the DeepStack
mergers are the tensors that must stay BF16.

The control is a deliberately broken inventory -- a renamed key, a reshaped key
and a dtype change -- which the mapping arm must reject with the right reason.
Without it a green mapping arm only proves the comparison ran.

Run it with the `llm-compressor` virtualenv python. Point `--source-dir` (or
`H3_BF16_ENCODER_DIR`) at the released text-encoder directory; no path from it
is written to the report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open

BENCH = Path(__file__).resolve().parent
REPORT = BENCH / "results" / "2026-08-24_calibration_model_mapping.json"

EXPECTED_DECODER_LAYERS = 64
BF16_PREFIXES = ("model.visual.", "model.language_model.embed_tokens.")


def checkpoint_inventory(root: Path) -> dict[str, dict]:
    """Name -> shape and dtype, read from the shard headers only."""
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"no model.safetensors.index.json in {root.name}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    inventory: dict[str, dict] = {}
    for shard, names in sorted(by_shard.items()):
        path = root / shard
        if not path.is_file():
            raise SystemExit(f"shard {shard} named by the index is not present")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                slice_ = handle.get_slice(name)
                inventory[name] = {
                    "shape": list(slice_.get_shape()),
                    "dtype": slice_.get_dtype(),
                    "shard": shard,
                }
    return inventory


def model_inventory(root: Path) -> tuple[dict[str, dict], dict, object]:
    """Name -> shape and dtype for the calibration model, built on meta.

    The inventory is `state_dict()`, not parameters plus buffers, because
    strict loading is defined against the state dict and the two differ: the
    three RoPE inverse-frequency buffers are non-persistent, recomputed at
    init, and present in no checkpoint. Comparing against the wider set reports
    them as missing every time, which is a false red -- and the rule here is
    that a check crying red while the state is correct is worse than no check.
    They are recorded separately so their absence stays visible rather than
    silently filtered.
    """
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(root)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    inventory = {
        name: {"shape": list(tensor.shape),
               "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in model.state_dict().items()
    }
    persistent = set(inventory)
    non_persistent = {
        name: list(tensor.shape)
        for name, tensor in model.named_buffers()
        if name not in persistent
    }
    return inventory, non_persistent, config


_DTYPE_ALIASES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32",
                  "I64": "int64", "I32": "int32", "U8": "uint8", "BOOL": "bool"}


def _normalise(dtype: str) -> str:
    return _DTYPE_ALIASES.get(dtype, dtype)


def compare(checkpoint: dict, model: dict, declared_dtype: str) -> dict:
    """Both directions, with the meta model's dtype normalised to the declared one.

    An empty-model instantiation takes its dtype from the config's `dtype`
    field, so comparing raw dtypes would flag every tensor. What matters is that
    the checkpoint's stored dtype is the one the config declares, so the model
    side is normalised to that and the checkpoint side is checked against it.
    """
    missing = sorted(set(model) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(model))
    shape_mismatch, dtype_mismatch = [], []
    for name in sorted(set(model) & set(checkpoint)):
        if model[name]["shape"] != checkpoint[name]["shape"]:
            shape_mismatch.append({
                "name": name, "model": model[name]["shape"],
                "checkpoint": checkpoint[name]["shape"],
            })
        stored = _normalise(checkpoint[name]["dtype"])
        if stored != declared_dtype:
            dtype_mismatch.append({
                "name": name, "stored": stored, "declared": declared_dtype,
            })
    return {
        "checkpoint_tensors": len(checkpoint),
        "model_tensors": len(model),
        "missing_from_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
    }


def quantization_surface(checkpoint: dict) -> dict:
    """What the candidate targets, and what must stay BF16.

    Read off the checkpoint rather than asserted, so a released file with a
    different layer count would show up here instead of in a launcher.
    """
    layers = set()
    for name in checkpoint:
        prefix = "model.language_model.layers."
        if name.startswith(prefix):
            head = name[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                layers.add(int(head))
    bf16_only = sorted(
        name for name in checkpoint if name.startswith(BF16_PREFIXES)
    )
    deepstack = [n for n in bf16_only if "deepstack_merger_list" in n]
    return {
        "decoder_layers_present": len(layers),
        "decoder_layer_range": [min(layers), max(layers)] if layers else None,
        "contiguous": bool(layers) and sorted(layers) == list(range(min(layers), max(layers) + 1)),
        "h3_consumes_layers": "0-49",
        "must_stay_bf16_tensors": len(bf16_only),
        "vision_tower_tensors": sum(1 for n in bf16_only if n.startswith("model.visual.")),
        "deepstack_merger_tensors": len(deepstack),
        "input_embedding_present": "model.language_model.embed_tokens.weight" in checkpoint,
    }


def violation_arm(checkpoint: dict, model: dict, declared_dtype: str) -> list[str]:
    """Three deliberate defects in the inventory. Each must be reported."""
    failures = []
    name = next(iter(sorted(model)))

    renamed = dict(checkpoint)
    renamed[name + ".renamed"] = renamed.pop(name)
    result = compare(renamed, model, declared_dtype)
    if name not in result["missing_from_checkpoint"]:
        failures.append("a renamed checkpoint key was not reported missing")
    if name + ".renamed" not in result["unexpected_in_checkpoint"]:
        failures.append("a renamed checkpoint key was not reported unexpected")

    reshaped = dict(checkpoint)
    reshaped[name] = {**reshaped[name], "shape": reshaped[name]["shape"] + [1]}
    result = compare(reshaped, model, declared_dtype)
    if not any(m["name"] == name for m in result["shape_mismatch"]):
        failures.append("a reshaped checkpoint key was not reported")

    recast = dict(checkpoint)
    recast[name] = {**recast[name], "dtype": "F32"}
    result = compare(recast, model, declared_dtype)
    if not any(m["name"] == name for m in result["dtype_mismatch"]):
        failures.append("a recast checkpoint key was not reported")
    return failures


def load_arm(root: Path) -> dict:
    from transformers import Qwen3VLForConditionalGeneration

    started = time.time()
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        root, dtype=torch.bfloat16, device_map="cpu", output_loading_info=True,
    )
    elapsed = time.time() - started
    parameters = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters()})

    # `from_pretrained` maps safetensors lazily, so a fast return proves the
    # keys resolved, not that any byte was read. Touching the first and last
    # element of every tensor forces each mapping to a real page and catches a
    # truncated or unreadable shard, at the cost of two pages per tensor rather
    # than the whole 65 GiB. `meta` would mean nothing was loaded at all.
    touched, on_meta, non_finite = 0, [], []
    for name, tensor in model.state_dict().items():
        if tensor.device.type == "meta":
            on_meta.append(name)
            continue
        flat = tensor.reshape(-1)
        if flat.numel() == 0:
            continue
        ends = torch.stack([flat[0], flat[-1]]).float()
        if not bool(torch.isfinite(ends).all()):
            non_finite.append(name)
        touched += 1

    result = {
        "loaded": True,
        "seconds": round(elapsed, 1),
        "load_is_mmap_backed": True,
        "tensors_touched": touched,
        "tensors_left_on_meta": on_meta,
        "tensors_with_non_finite_ends": non_finite,
        "parameters": parameters,
        "parameter_dtypes": dtypes,
        "loading_info": {k: (v if isinstance(v, (int, str)) else list(v))
                         for k, v in info.items()},
        "decoder_layers_built": len(model.model.language_model.layers),
        "deepstack_mergers_built": len(model.model.visual.deepstack_merger_list),
    }
    del model
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"),
        help="released text-encoder directory; defaults to H3_BF16_ENCODER_DIR",
    )
    parser.add_argument("--no-load", action="store_true",
                        help="skip the real materialising load (weaker claim)")
    args = parser.parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir or H3_BF16_ENCODER_DIR is required")
    root = Path(args.source_dir).expanduser().resolve()

    print("reading the checkpoint index and shard headers")
    checkpoint = checkpoint_inventory(root)
    print(f"  {len(checkpoint)} tensors")

    print("instantiating the calibration model on the meta device")
    model, non_persistent, config = model_inventory(root)
    declared_dtype = str(getattr(config.text_config, "dtype", "bfloat16"))
    declared_dtype = declared_dtype.removeprefix("torch.")
    print(f"  {len(model)} state-dict tensors, config declares {declared_dtype}; "
          f"{len(non_persistent)} non-persistent buffers excluded")

    report: dict = {
        "check": "released checkpoint maps and loads strictly into the "
                 "Transformers calibration model",
        "path_policy": "logical identifiers only; the source directory is not recorded",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "source": {"logical_name": root.name, "shards": len({
            entry["shard"] for entry in checkpoint.values()})},
        "declared_dtype": declared_dtype,
        "non_persistent_buffers": non_persistent,
    }
    failures: list[str] = []

    report["mapping"] = compare(checkpoint, model, declared_dtype)
    for key in ("missing_from_checkpoint", "unexpected_in_checkpoint",
                "shape_mismatch", "dtype_mismatch"):
        if report["mapping"][key]:
            failures.append(f"{key}: {report['mapping'][key][:4]}")
    print(f"  missing={len(report['mapping']['missing_from_checkpoint'])} "
          f"unexpected={len(report['mapping']['unexpected_in_checkpoint'])} "
          f"shape={len(report['mapping']['shape_mismatch'])} "
          f"dtype={len(report['mapping']['dtype_mismatch'])}")

    report["quantization_surface"] = quantization_surface(checkpoint)
    surface = report["quantization_surface"]
    if surface["decoder_layers_present"] != EXPECTED_DECODER_LAYERS:
        failures.append(
            f"the checkpoint carries {surface['decoder_layers_present']} decoder "
            f"layers, the candidate targets {EXPECTED_DECODER_LAYERS}"
        )
    if not surface["contiguous"]:
        failures.append("decoder layer indices are not contiguous")
    if not surface["input_embedding_present"]:
        failures.append("no input embedding in the checkpoint")
    print(f"  decoder layers {surface['decoder_layer_range']}, "
          f"{surface['must_stay_bf16_tensors']} tensors must stay BF16 "
          f"({surface['vision_tower_tensors']} vision, "
          f"{surface['deepstack_merger_tensors']} DeepStack)")

    print("violation arm")
    violations = violation_arm(checkpoint, model, declared_dtype)
    report["violation_arm"] = {"failures": violations,
                               "passed": not violations}
    failures += violations
    print(f"  {'all three defects reported' if not violations else violations}")

    if args.no_load:
        report["load"] = {
            "loaded": False,
            "reason": "--no-load: the mapping arm ran, the materialising load "
                      "did not. This is a weaker claim than a full strict load.",
        }
        print("skipping the materialising load (--no-load)")
    else:
        print("loading the full checkpoint; this takes minutes and ~65 GiB of RAM")
        report["load"] = load_arm(root)
        info = report["load"]["loading_info"]
        for key, value in info.items():
            if isinstance(value, list) and value:
                failures.append(f"from_pretrained reported {key}: {value[:4]}")
        if report["load"]["tensors_left_on_meta"]:
            failures.append(
                f"tensors left on the meta device after loading: "
                f"{report['load']['tensors_left_on_meta'][:4]}"
            )
        if report["load"]["tensors_with_non_finite_ends"]:
            failures.append(
                f"non-finite values at the ends of "
                f"{report['load']['tensors_with_non_finite_ends'][:4]}"
            )
        if report["load"]["decoder_layers_built"] != EXPECTED_DECODER_LAYERS:
            failures.append(
                f"the loaded model built {report['load']['decoder_layers_built']} "
                f"decoder layers"
            )
        print(f"  loaded {report['load']['parameters']:,} parameters in "
              f"{report['load']['seconds']}s (mmap-backed), touched "
              f"{report['load']['tensors_touched']} tensors, "
              f"dtypes {report['load']['parameter_dtypes']}, "
              f"loading_info { {k: len(v) if isinstance(v, list) else v for k, v in info.items()} }")

    report["failures"] = failures
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{REPORT.name}")

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: every released tensor maps to a tensor the calibration model "
          "wants, at the declared shape and dtype, and the comparison fails on "
          "a renamed, reshaped or recast key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
