#!/usr/bin/env python3
"""What the Sol override's container entry changes, on captured q/k/v.

    cd <ComfyUI root>
    PYTHONPATH=. <comfy venv python> custom_nodes/<pack>/bench/measure_sol_container_protocol.py \
        --out OUT.json --dense <capture.pt> [...] --sol <capture.pt> [...]

`sol_attn_h3.py::make_override` gives its override a `container_function` when
the fallback under it has one, so `wrap_attn` hands q, k and v over in core's
single-owner containers. Each cell here goes through core's `optimized_attention`
as the H3 forward calls it, over the fallback `set_model_optimized_attention`
builds for kitchen's int8 backend, twice: with the container entry, and with
the attribute removed, which is the override as it was before 2026-10-03.

Per cell it records the route that ran, whether the two outputs are the same
bytes, the peak allocation of each call, and the time of a second call of each.
`--dense` cells run with their block in `dense_blocks`, `--sol` cells with none.
For the dense cells it also compares kitchen's two entries directly
(`int8_attention` against `prequantize_int8_attention` then
`int8_attention_from_prequantized`), since the container entry swaps one for
the other. Needs a quiet card with room for one call at the capture's length.
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def load(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def kitchen_fallback():
    """The override core installs for kitchen's int8 backend, built by core."""
    import comfy.ldm.modules.attention as attention
    import comfy.model_patcher
    holder = types.SimpleNamespace(model_options={"transformer_options": {}})
    comfy.model_patcher.ModelPatcher.set_model_optimized_attention(
        holder, attention.attention_comfy_kitchen_int8)
    return holder.model_options["transformer_options"]["optimized_attention_override"]


def one_call(attention, override, cell, timed):
    """(output on host, peak bytes, ms or None, route). q, k and v go up fresh
    each time: both entries consume them."""
    Box = attention.AttentionTensorContainer
    q, k, v = (Box(cell[n].cuda()) for n in ("q", "k", "v"))        # [1, H, S, D]
    heads, tokens = cell["q"].shape[1], cell["q"].shape[2]
    options = {"minimax_h3_layout": types.SimpleNamespace(
                   seq_len=tokens, segments=[tuple(s) for s in cell["segments"]]),
               "block_index": int(cell["block"]),
               "sigmas": torch.tensor([float(cell["sigma"])]),
               "optimized_attention_override": override}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    out = attention.optimized_attention(q, k, v, heads, mask=None, skip_reshape=True,
                                        transformer_options=options)
    e1.record()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    host = out.cpu()
    del out, q, k, v
    torch.cuda.empty_cache()
    return host, peak, round(e0.elapsed_time(e1), 1) if timed else None, options.get("h3_attn_route")


def kitchen_direct(cell):
    import comfy_kitchen as ck
    q, k, v = (cell[n].cuda() for n in ("q", "k", "v"))
    a = ck.int8_attention(q, k, v).cpu()
    quantized = ck.prequantize_int8_attention(q, k, v)
    del q, k, v
    b = ck.int8_attention_from_prequantized(quantized).cpu()
    del quantized
    torch.cuda.empty_cache()
    return bool(torch.equal(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense", nargs="*", default=[])
    ap.add_argument("--sol", nargs="*", default=[])
    args = ap.parse_args()

    load("h3x", REPO / "__init__.py", package_dir=REPO)
    node = sys.modules["h3x.sol_attn_h3"]
    import comfy.ldm.modules.attention as attention
    import comfy_kitchen as ck
    qk_balance, rotate = node.SOL_QUANTIZERS[node.SOL_QUANTIZER_DEFAULT]
    fallback = kitchen_fallback()

    cells = {}
    for kind, paths in (("dense", args.dense), ("sol", args.sol)):
        for path in paths:
            name = path.split("/")[-1]
            cell = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            dense_blocks = frozenset({int(cell["block"])}) if kind == "dense" else frozenset()

            def make():
                return node.make_override(
                    tau=1.0, min_tokens=12288, sink_conditioning=node.SOL_SINK_DEFAULT,
                    dense_blocks=dense_blocks, previous=fallback,
                    qk_balance=qk_balance, rotate=rotate)
            container, tensor = make(), make()
            del tensor.container_function                   # the override as it was
            row = {"expected": kind, "shape": list(cell["q"].shape),
                   "qkv_bytes": sum(cell[n].numel() * cell[n].element_size() for n in "qkv"),
                   "has_container_entry": hasattr(container, "container_function")}
            for label, override in (("tensor_entry", tensor), ("container_entry", container)):
                host, peak, _ms, route = one_call(attention, override, cell, timed=False)
                _again, _peak, ms, _route = one_call(attention, override, cell, timed=True)
                row[label] = {"route": route, "peak_bytes": peak, "ms_second_call": ms,
                              "repeat_equal": bool(torch.equal(host, _again))}
                if label == "tensor_entry":
                    before = host
                else:
                    row["outputs_equal"] = bool(torch.equal(before, host))
                del _again
            row["peak_saved_bytes"] = row["tensor_entry"]["peak_bytes"] - row["container_entry"]["peak_bytes"]
            if kind == "dense":
                row["kitchen_entries_equal"] = kitchen_direct(cell)
            cells[name] = row
            print(name, json.dumps(row), flush=True)
            del before, host, cell

    record = {
        "what": "MiniMaxH3Sol's override with and without core's container entry, on captured q/k/v",
        "tool": "bench/measure_sol_container_protocol.py",
        "fallback": "core's set_model_optimized_attention(attention_comfy_kitchen_int8)",
        "settings": {"tau": 1.0, "quantizer": node.SOL_QUANTIZER_DEFAULT,
                     "sink_conditioning": node.SOL_SINK_DEFAULT},
        "comfy_kitchen": getattr(ck, "__version__", None),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "peak_note": "torch.cuda.max_memory_allocated over the one attention call, "
                     "q, k and v included; nothing else on the card",
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
