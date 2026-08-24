#!/usr/bin/env python3
"""Which SDPA kernel actually runs, named by the profiler rather than inferred.

Availability is not selection. `can_use_flash_attention`,
`can_use_efficient_attention` and `can_use_cudnn_attention` bound what *could*
have run; none of them says what did. That distinction was got wrong once in
this lane: a 512-token availability matrix that omitted cuDNN was read as proof
that the math backend must be running, and the conclusion was used to propose a
Gate 2B lever. Two defects pointed the same way -- an incomplete matrix and a
shape that did not match the failure -- which is the combination that makes a
wrong conclusion feel well founded.

So this probe names the operation. `torch.profiler` records the dispatched
`aten::_scaled_dot_product_*` op, which is the only direct evidence available
short of reading kernel launches.

**Deliberately separate from the Gate 2A pilot.** The profiler has its own
overhead and allocations; running it inside the feasibility harness would
contaminate exactly the peak and timing numbers that harness exists to measure.

Two arms, because neither alone is enough:

1. **Direct calls.** `scaled_dot_product_attention` invoked at the real head
   counts, head dimension and sequence lengths, with the exact keyword
   arguments `transformers.integrations.sdpa_attention` would pass -- causal,
   no mask, `enable_gqa` on or off. This isolates the dispatch decision from
   everything else, and it can sweep shapes cheaply.
2. **In situ.** A real slice of the released checkpoint -- few layers, but the
   released `head_dim`, head counts and vision geometry -- driven with a real
   calibration batch. **A reduced-width model would be worthless here**: head
   dimension is one of the things the backends gate on, so shrinking it changes
   the answer. This confirms the model's own call dispatches where arm 1 says.

It also records, per shape, the numerical difference between an explicitly
forced backend and the free choice. A forced arm that matches `auto` bit for
bit is evidence about which one `auto` took; one that differs is evidence it
did not.

Run it with the `llm-compressor` virtualenv python and a free GPU. Point
`--source-dir` or `H3_BF16_ENCODER_DIR` at the released text encoder.
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
sys.path.insert(0, str(BENCH))
REPORT = BENCH / "results" / "2026-08-24_sdpa_backend_selection.json"

from h3_calibration_precision import POLICIES, compute_dtype  # noqa: E402
from h3_effective_batch import effective_batch  # noqa: E402

FORCED = ("math", "flash", "efficient", "cudnn")


def _backend(name: str):
    from torch.nn.attention import SDPBackend

    return {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[name]


def availability(query, key, is_causal: bool, gqa: bool) -> dict:
    from torch.backends.cuda import (
        SDPAParams,
        can_use_cudnn_attention,
        can_use_efficient_attention,
        can_use_flash_attention,
    )

    params = SDPAParams(query, key, key, None, 0.0, is_causal, gqa)
    return {
        "flash": bool(can_use_flash_attention(params, False)),
        "efficient": bool(can_use_efficient_attention(params, False)),
        "cudnn": bool(can_use_cudnn_attention(params, False)),
    }


def profiled_ops(fn) -> list[str]:
    """The `aten::_scaled_dot_product_*` operations one call dispatched."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {
        event.name for event in prof.events()
        if "_scaled_dot_product" in event.name
    }
    return sorted(names)


def direct_arm(shapes: list[dict], dtype: torch.dtype) -> list[dict]:
    """Arm 1: SDPA called directly at real shapes, with transformers' kwargs."""
    results = []
    for shape in shapes:
        heads, kv_heads = shape["heads"], shape["kv_heads"]
        length, head_dim = shape["length"], shape["head_dim"]
        gqa = kv_heads != heads
        q = torch.randn(1, heads, length, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(1, kv_heads, length, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(1, kv_heads, length, head_dim, dtype=dtype, device="cuda")
        kwargs = {"attn_mask": None, "dropout_p": 0.0,
                  "is_causal": shape["is_causal"]}
        if gqa:
            kwargs["enable_gqa"] = True

        entry = {
            **{k2: v2 for k2, v2 in shape.items()},
            "enable_gqa": gqa,
            "availability": availability(q, k, shape["is_causal"], gqa),
        }
        call = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)  # noqa: E731
        try:
            entry["auto_dispatched_ops"] = profiled_ops(call)
            reference = call().double()
        except Exception as exc:
            entry["auto"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            results.append(entry)
            del q, k, v
            torch.cuda.empty_cache()
            continue

        entry["forced"] = {}
        for name in FORCED:
            from torch.nn.attention import sdpa_kernel

            try:
                with sdpa_kernel([_backend(name)]):
                    out = call().double()
                    ops = profiled_ops(call)
                delta = float((reference - out).abs().max())
                entry["forced"][name] = {
                    "ran": True,
                    "dispatched_ops": ops,
                    "max_abs_delta_vs_auto": delta,
                    "bit_identical_to_auto": delta == 0.0,
                }
                del out
            except Exception as exc:
                entry["forced"][name] = {
                    "ran": False,
                    "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}",
                }
        # Which forced arms reproduce `auto` exactly. If exactly one does, that
        # is the strongest available evidence for what `auto` selected; if
        # several do, the evidence does not separate them and says so.
        matches = [n for n, r in entry["forced"].items()
                   if r.get("bit_identical_to_auto")]
        entry["auto_matched_bit_for_bit_by"] = matches
        entry["selection_evidence"] = (
            "one forced backend reproduces auto exactly" if len(matches) == 1
            else "ambiguous: several or no forced backends reproduce auto"
        )
        results.append(entry)
        del q, k, v, reference
        torch.cuda.empty_cache()
    return results


def in_situ_arm(source: Path, bundle: Path, row_id: str | None, policy: str,
                layers: int) -> dict:
    """Arm 2: a real slice of the released checkpoint, real geometry."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    from h3_calibration_precision import calibration_precision

    manifest = json.loads((bundle / "presentation.json").read_text())
    record = next((r for r in manifest["rows"] if r["row_id"] == row_id),
                  manifest["rows"][0])
    raw = load_file(bundle / record["batch_file"])
    batch, _ = effective_batch(raw, row_id=record["row_id"])

    config = AutoConfig.from_pretrained(source)
    config.text_config.num_hidden_layers = layers
    dtype = compute_dtype(policy)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, config=config, dtype=dtype, attn_implementation="sdpa",
    ).eval().to("cuda")

    inputs = {k: (v.to("cuda", dtype) if v.is_floating_point() else v.to("cuda"))
              for k, v in batch.items()}

    def call():
        with torch.no_grad():
            model(**inputs, use_cache=False)

    with calibration_precision(model, policy):
        ops = profiled_ops(call)
    result = {
        "row_id": record["row_id"],
        "sequence_tokens": record["sequence_length"],
        "vision_block_grids": [b["grid_thw"][0][1:] for b in record["vision_blocks"]],
        "decoder_layers_built": layers,
        "released_head_dim": config.text_config.head_dim,
        "released_heads": config.text_config.num_attention_heads,
        "released_kv_heads": config.text_config.num_key_value_heads,
        "vision_heads": config.vision_config.num_heads,
        "dispatched_ops": ops,
        "note": "real released head_dim and head counts; a reduced-width model "
                "would change the answer because the backends gate on head "
                "dimension",
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--row")
    parser.add_argument("--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"))
    parser.add_argument("--policy", default="comfy_exact", choices=sorted(POLICIES))
    parser.add_argument("--layers", type=int, default=2,
                        help="decoder layers for the in-situ arm; the released "
                             "head geometry is what matters, not the depth")
    parser.add_argument("--skip-in-situ", action="store_true")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir or H3_BF16_ENCODER_DIR is required")
    if not torch.cuda.is_available():
        raise SystemExit("this probe is about CUDA kernel selection")

    from transformers import AutoConfig

    source = Path(args.source_dir).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / "presentation.json").read_text())
    config = AutoConfig.from_pretrained(source)
    text, vision = config.text_config, config.vision_config
    dtype = compute_dtype(args.policy)

    lengths = sorted({r["sequence_length"] for r in manifest["rows"]})
    patches = sorted({
        int(b["grid_thw"][0][1]) * int(b["grid_thw"][0][2])
        for r in manifest["rows"] for b in r["vision_blocks"]
    })
    shapes = [
        {"kind": "text", "length": length, "heads": text.num_attention_heads,
         "kv_heads": text.num_key_value_heads, "head_dim": text.head_dim,
         "is_causal": True}
        for length in lengths
    ] + [
        {"kind": "vision", "length": count, "heads": vision.num_heads,
         "kv_heads": vision.num_heads,
         "head_dim": vision.hidden_size // vision.num_heads, "is_causal": False}
        for count in patches
    ]

    print(f"policy {args.policy} -> compute dtype {dtype}")
    print(f"{len(shapes)} real shapes: {len(lengths)} text lengths, "
          f"{len(patches)} vision block sizes")

    report: dict = {
        "probe": "which SDPA kernel the calibration path actually dispatches",
        "separate_from": "the Gate 2A feasibility pilot, deliberately: the "
                         "profiler's overhead would contaminate its peak and "
                         "timing measurements",
        "availability_is_not_selection": True,
        "path_policy": "logical identifiers only",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "policy": args.policy,
        "compute_dtype": str(dtype).removeprefix("torch."),
        "direct": direct_arm(shapes, dtype),
    }
    for entry in report["direct"]:
        print(f"  {entry['kind']:<7} n={entry['length']:<6} gqa={entry['enable_gqa']} "
              f"avail={entry['availability']}  auto={entry.get('auto_dispatched_ops')}  "
              f"matched_by={entry.get('auto_matched_bit_for_bit_by')}")

    if not args.skip_in_situ:
        print("in-situ arm: real released head geometry")
        report["in_situ"] = in_situ_arm(source, bundle, args.row, args.policy,
                                        args.layers)
        print(f"  dispatched: {report['in_situ']['dispatched_ops']}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
