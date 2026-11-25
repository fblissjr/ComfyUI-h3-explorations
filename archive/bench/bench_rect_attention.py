"""One block's attention, alone on the card, for a rectangle of queries against every key.

Random tensors in the H3 block's layout (`comfy/ldm/minimax/model.py::Attention.forward`:
`(rows, heads, head_dim)`, transposed, with a batch axis), no model and no
server. Three kernels on the same tensors: comfy-kitchen's `int8_attention`
through both of core's entries (`comfy/ldm/modules/attention.py`: the plain
one and the container one, which prequantizes), and torch's
`scaled_dot_product_attention` as core's `attention_pytorch` calls it
(`comfy/ops.py::SDPA_BACKEND_PRIORITY`).

It exists to say which kernel a measured attention stage ran on, and what the
other would have cost at the same sizes. First use:
`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`.

    cd <ComfyUI> && .venv/bin/python -I custom_nodes/<pack>/bench/bench_rect_attention.py --out <file.json>

The sizes default to that record's: the masked window's packed rows with the
live rows of its cached step, and the audio-refine pass's. A step is
`BLOCKS` calls.
"""
import argparse
import importlib.metadata
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import comfy_kitchen

#: The DiT's defaults, `comfy/ldm/minimax/model.py` (inherited).
HEADS, HEAD_DIM, BLOCKS = 56, 128, 50
#: core's order, `comfy/ops.py::SDPA_BACKEND_PRIORITY` (inherited).
SDPA_PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                 SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
#: keys, then the query counts timed against them. The first row is the masked
#: window of `2026-10-06_frozen_cache_masked_window.md` (its live rows, its
#: ring, and the square); the second the refine pass of
#: `2026-10-06_frozen_cache_stage_split.md` (measured, both).
SIZES = ((113072, (1699, 7062, 28689, 56536, 113072)), (104515, (1699,)))
REPEATS = 4


def _rows(n, dtype, device, gen):
    x = torch.randn((n, HEADS, HEAD_DIM), generator=gen, device=device, dtype=torch.float32)
    return (x / x.pow(2).mean(-1, keepdim=True).sqrt()).to(dtype)  # RMS-normed, as q and k are


def _time(fn):
    seconds = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(REPEATS + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        del out
        if i:  # the first call is a warm-up
            seconds.append(dt)
    return seconds, torch.cuda.max_memory_allocated() / 2**30


def _sdpa(q, k, v):
    with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


KERNELS = {
    "kitchen int8, plain": lambda q, k, v: comfy_kitchen.int8_attention(q, k, v),
    "kitchen int8, prequantized": lambda q, k, v: comfy_kitchen.int8_attention_from_prequantized(
        comfy_kitchen.prequantize_int8_attention(q, k, v)),
    "torch sdpa": _sdpa,
}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--out", help="write the rows and the environment here as json")
    args = ap.parse_args()
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    gen = torch.Generator(device=device).manual_seed(0)
    rows = []
    for keys, query_counts in SIZES:
        k = _rows(keys, dtype, device, gen).transpose(0, 1).unsqueeze(0)
        v = _rows(keys, dtype, device, gen).transpose(0, 1).unsqueeze(0)
        q_all = _rows(keys, dtype, device, gen)
        for queries in query_counts:
            # the cached block's own selection: an index_select of the live rows, then the transpose
            idx = torch.linspace(0, keys - 1, queries, device=device).long()
            q = (q_all.index_select(0, idx) if queries < keys else q_all).transpose(0, 1).unsqueeze(0)
            for name, fn in KERNELS.items():
                if name == "torch sdpa" and queries > 28689:
                    continue  # not a size any measured stage ran it at
                seconds, peak = _time(lambda: fn(q, k, v))
                best = min(seconds)
                rows.append({"kernel": name, "keys": keys, "queries": queries, "dtype": args.dtype,
                             "seconds_a_call": [round(s, 4) for s in seconds], "best_a_call": round(best, 4),
                             "seconds_a_step": round(best * BLOCKS, 2),
                             "ns_per_pair": round(best / (queries * keys) * 1e9, 4),
                             "peak_gib": round(peak, 2)})
                print(json.dumps(rows[-1]), flush=True)
            del q
        del k, v, q_all
        torch.cuda.empty_cache()
    doc = {"environment": {"torch": torch.__version__, "comfy_kitchen": importlib.metadata.version("comfy_kitchen"),
                           "card": torch.cuda.get_device_name(0)},
           "layout": {"heads": HEADS, "head_dim": HEAD_DIM, "blocks": BLOCKS, "repeats": REPEATS}, "rows": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
