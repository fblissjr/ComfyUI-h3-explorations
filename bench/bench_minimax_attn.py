#!/usr/bin/env python3
"""A/B the patched MiniMax H3 attention forward against the stock one.

Builds a single real `comfy.ldm.minimax.model.Attention` module at H3's
config and drives it directly, so this measures the thing the node
actually replaces -- including the fused qkv projection whose output q, k
and v are all views of, which is what decides how much memory can
actually be released mid-call.

Run one arm per process. Both arms allocate multi-GiB tensors, and a
prior arm trains the caching allocator in a way that biases whatever runs
second:

    python bench/bench_minimax_attn.py stock
    python bench/bench_minimax_attn.py sage

Needs ComfyUI importable (run from the ComfyUI root or with it on
PYTHONPATH) and about 8 GiB of free VRAM at the default shape.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# H3's attention config, and the packed sequence length for fl2va at the
# node's default canvas (1344x768, 124 frames).
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
SEQ_DEFAULT = 41822
# H3 ropes 96 of its 128 head dims; the table carries pair angles, so the
# model's `rot_dim = rope_freqs.shape[-3] * 2` recovers this from [.., 48, 2, 2].
ROT_DIM = 96
MiB = 2**20


def build_attention(device, dtype):
    import comfy.ops
    from comfy.ldm.minimax.model import Attention

    # requires_grad_(False) mirrors a loaded model. `cast_to` hands the rope
    # kernel the norm weights untouched when they are already the right dtype
    # and device, so a parameter that still wants grad reaches its
    # inference-only guard as a readonly input and trips it.
    return Attention(
        HIDDEN, HEADS, HEAD_DIM, 1e-5,
        dtype=dtype, device=device, operations=comfy.ops.manual_cast,
    ).to(device).requires_grad_(False)


def build_rope(seq, device, dtype):
    """A rope table of the shape the model builds, at H3's 96 of 128 rot dims.

    The values are junk; the shape and the dtype are not. Passing a table at
    all is what decides the aliasing this bench exists to measure: with one,
    `rms_rope_split_half_` runs in place and q, k and v stay three views of
    the fused qkv buffer, which is the real inference path. With
    `rope_freqs=None` the eager branch runs `q_norm(q.view(...))`, and
    RMSNorm returns fresh tensors -- so q and k are separate allocations and
    only v still pins the fused buffer. That is a different memory question
    than the one this file is asking.
    """
    from comfy.ldm.minimax.model import rope_rotation_table

    angles = torch.zeros(seq, ROT_DIM, device=device, dtype=torch.float32)
    return rope_rotation_table(angles, dtype)


def timed(fn, warmup=1, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def inference_only():
    """The in-place rope kernel refuses to run on tensors that require grad.

    ComfyUI samples under no_grad, so its Attention never meets that check.
    A bench that drives the module directly has to say so itself, or the
    module's own parameters make every activation require grad.
    """
    torch.set_grad_enabled(False)


def probe(args):
    """Report whether q, k and v share one allocation on the way to the kernel.

    Runs the real patched forward with the kernel swapped for a recorder, at
    a small sequence length, so it costs nothing and needs no GPU.
    """
    from attention import build_kernel, make_minimax_attn_forward

    inference_only()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = build_attention(device, torch.bfloat16)
    seen = {}

    def recorder(qkv, **_kw):
        q, k, v = qkv
        qkv.clear()
        seen.update(q=q.untyped_storage().data_ptr(),
                    k=k.untyped_storage().data_ptr(),
                    v=v.untyped_storage().data_ptr())
        return torch.zeros_like(q)

    kernel_kwargs = build_kernel(args.mode)[1]
    forward = make_minimax_attn_forward(recorder, kernel_kwargs,
                                        clone_v=args.clone_v)
    attn.forward = forward.__get__(attn, attn.__class__)
    seq = 64
    rope = None if args.no_rope else build_rope(seq, device, torch.bfloat16)
    attn(torch.randn(seq, HIDDEN, device=device, dtype=torch.bfloat16),
         rope_freqs=rope)

    qk_shared = seen["q"] == seen["k"]
    v_shared = seen["v"] == seen["q"]
    print(f"clone_v={args.clone_v} rope={not args.no_rope}  "
          f"q={seen['q']:#x} k={seen['k']:#x} v={seen['v']:#x}\n"
          f"  q and k share one allocation: {qk_shared}\n"
          f"  v shares it too:              {v_shared}")
    # The flag's whole claim is that it takes v out of the fused buffer. If v
    # is not in there to begin with, or is still in there afterwards, any peak
    # reading taken alongside it is measuring something other than the claim.
    if not qk_shared:
        print("PROBE FAILED: q and k are not views of one buffer, so this is "
              "not the fused-qkv case the clone is about")
        return 1
    if v_shared == args.clone_v:
        print("PROBE FAILED: --clone-v did not change v's aliasing")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["stock", "sage"])
    ap.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--clone-v", action="store_true",
                    help="sage arm only: give v its own storage before the "
                         "hand-off, so releasing q and k frees the fused qkv "
                         "buffer. Costs a third of that buffer up front.")
    ap.add_argument("--no-rope", action="store_true",
                    help="Take the eager q_norm/k_norm branch instead of the "
                         "fused in-place rope. Changes the qkv aliasing, so it "
                         "is not the shape of the real inference path -- kept "
                         "only to isolate the fused rope kernel's own cost.")
    ap.add_argument("--probe", action="store_true",
                    help="Print the storage each of q, k and v lands in and "
                         "exit, without benchmarking. This is the control for "
                         "--clone-v: a null result in the peak numbers means "
                         "nothing unless the flag is known to have changed the "
                         "aliasing it claims to change.")
    args = ap.parse_args()

    if args.probe:
        return probe(args)

    inference_only()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    attn = build_attention(device, dtype)
    x = torch.randn(args.seq, HIDDEN, device=device, dtype=dtype)

    if args.arm == "sage":
        from attention import build_kernel
        from attention import make_minimax_attn_forward

        kernel_fn, kernel_kwargs = build_kernel(args.mode)
        forward = make_minimax_attn_forward(kernel_fn, kernel_kwargs,
                                            clone_v=args.clone_v)
        attn.forward = forward.__get__(attn, attn.__class__)

    # The rope table is what keeps q, k and v three views of the fused qkv
    # buffer, which is the aliasing the peak number is about. See build_rope:
    # the eager branch is a different memory question, not a cheaper way to
    # ask this one.
    rope = None if args.no_rope else build_rope(args.seq, device, dtype)
    call = lambda: attn(x, rope_freqs=rope)

    call()  # allocate autotune scratch before the peak is recorded
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = call()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MiB
    del out

    ms = timed(call)
    label = args.arm + ("+clone_v" if args.clone_v else "")
    print(
        f"{label:14s} seq={args.seq} mode={args.mode}  "
        f"{ms:8.2f} ms   peak {peak:7.0f} MiB"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
