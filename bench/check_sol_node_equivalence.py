#!/usr/bin/env python3
"""Grade our Sol node's DISPATCH against the algorithm's own eager reference.

## What this covers that `check_solattn_correctness.py` does not

That file grades the KERNEL against the algorithm's eager reference. This
grades everything our node does AROUND the kernel: the BHND-to-BTHD transpose
and back, the scale it forwards, the sink pair it derives, and the tail flag it
passes. A defect in any of those produces a plausible tensor of the right shape
and a successful render.

The distinction is the reshape. `optimized_attention` hands H3's attention over
as BHND with `skip_reshape=True`, the kernel wants BTHD, and the output goes
back as BHND. Getting that wrong transposes heads against tokens -- which does
not raise, because both are legal sizes.

## The oracle is the KERNEL, not the algorithm, and the first draft got that wrong

This file was first written to compare the dispatch against the eager
reference, and the numbers looked like a marginal failure: cosine ~0.994 to
0.998 depending on shape and seed, sometimes under the bar. The instinct was
to loosen the bar. That would have been wrong twice over.

Measured instead: **the dispatch is BITWISE identical to a direct
`comfy_kitchen.sol_attn` call** on the same inputs. Every bit of that spread
was the kernel's INT8 arithmetic against fp32 -- which is not the node's doing,
is a property `check_solattn_correctness.py` already owns, and would have been
silently absorbed into a loosened tolerance here.

So the oracle is the kernel. That gives an exact claim rather than a tolerance,
it isolates the layer this file is about, and it needs no O(T^2) score tensor,
so it runs at a realistic sequence length instead of a toy one. **A tolerance
where an equality is available is a check that cannot see small defects.**

## Why it no longer compares against the vendored node

**It used to, and that comparison is finished rather than broken.** Until
2026-08-30 this file asserted that our forked node produced the SAME BYTES as
the vendored upstream one at the shipped settings, which is what made migrating
145 graphs safe. It passed, at both selections, and the result is recorded in
`bench/results/2026-08-30_sol_node_equivalence.json`.

That comparison cannot be re-run and should not be resurrected. `vendor/`
now holds the PRE-MERGE upstream drop, restored to be a pristine reference:
its `_run` passes `centroid_tail` to a kernel that no longer accepts it, so it
raises rather than producing a baseline. Keeping a check that can only skip
would be worse than none -- so the baseline moved to the algorithm, which is
the more durable control anyway and one this repo already trusts.

Claims, i.e. what breaks if a case is deleted:

  dispatch == kernel       our node's `_run` produces the SAME BYTES as calling
                           `comfy_kitchen.sol_attn` directly with the transpose
                           done by hand. Catches a transpose, a dropped scale,
                           or a sink pair built wrong.
  top-k dispatch           the same through the other selection, which no
                           shipped graph uses and a bench arm can reach.
  sink pair reaches        a non-zero sink must change the output. The sink is
    the kernel             derived from H3's layout and passed through two
                           call frames; if it stopped arriving, every
                           conditioning row would be routed sparsely and the
                           render would merely look worse.
  pooled_tail reaches      RED CONTROL. `tail` is the argument the fork added.
    the kernel             If turning it off does not move the output, it is
                           not connected and every case above is comparing a
                           knob that does nothing.
  (an OOM exits 2, not 1)  a resident model or a render in flight can leave
                           too little VRAM for these shapes. That is an
                           environment state, not a result. Until 2026-09-04
                           the guard for it wrapped nothing and the check
                           died on a raw traceback with exit 1; now the
                           kernel cases run under `graded_kernel_cases`,
                           which catches the allocator's error, names the
                           cases that were not graded, and exits 2.
                           `--oom-control` proves that with the card masked:
                           a kernel stub that raises on its first call must
                           come back as 2 with no case marked.

  a transposed oracle      RED CONTROL, and the one that earns this file.
    is caught              Compares against a kernel call with heads and tokens
                           swapped. If that still matches, the equality above
                           is not seeing layout at all.

  the sink pair per mode   CPU, pure, run BEFORE the CUDA gate. `_sink_blocks` on a
    (no kernel)            fixture layout for every `sink_conditioning` mode:
                           off is zeros; exact_kv has no dense-query range;
                           exact_kv_and_rows starts the range at the target
                           audio and leaves reference rows sparse;
                           exact_kv_and_all_rows covers every conditioning
                           row; the no-audio-span fallback of exact_kv_and_rows
                           IS the all-rows range; on a t2v-shaped layout the
                           two ranges differ by the text rows alone and on a
                           ref2v-shaped one by the reference rows; a missing
                           video span or a short sequence is zeros in every
                           mode; an unknown mode is refused; and the node's
                           combo lists exactly the modes the function accepts,
                           with the shipped default among them.

Needs CUDA and a comfy_kitchen carrying the merged `sol_attn` for the kernel
cases; the sink cases run anywhere the node imports. Exit 0 all passed, 1 a
case failed, 2 the kernel cases were not graded (no CUDA, no kernel, OOM),
even when the sink cases ran and passed.

    python bench/check_sol_node_equivalence.py
    CUDA_VISIBLE_DEVICES= python bench/check_sol_node_equivalence.py   # sink cases only, touches no card
    CUDA_VISIBLE_DEVICES= python bench/check_sol_node_equivalence.py --oom-control
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent


def load(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def sink_cases(node, check):
    """`_sink_blocks` per mode on fixture layouts. Pure, so no tensor and no
    kernel: the sink pair is derived from the published spans alone."""
    B = node.BLOCK_SIZE
    modes = node.SINK_CONDITIONING_MODES
    # [text][audio][video]: t2v-shaped, text immediately before the target audio
    t2v = dict(sol_h3_video_span=(640, 4096), sol_h3_audio_span=(320, 640))
    # [text][ref][audio][video]: ref2v-shaped, reference rows between them
    ref = dict(sol_h3_video_span=(4160, 8192), sol_h3_audio_span=(3840, 4160))
    T2V, REF = 4096, 8192
    kv = (0, 10)                       # ceil(640 / 64) and ceil(4160 / 64) == 65
    kv_ref = (0, 65)

    def pair(opts, tokens, mode):
        return node._sink_blocks(opts, tokens, mode)

    check("off: zeros", pair(t2v, T2V, "off") == ((0, 0), (0, 0)))
    check("exact_kv: exact keys over conditioning, no dense queries",
          pair(t2v, T2V, "exact_kv") == (kv, (0, 0)) and pair(ref, REF, "exact_kv") == (kv_ref, (0, 0)))
    check("exact_kv_and_rows: dense queries from the target audio to the end of conditioning",
          pair(t2v, T2V, "exact_kv_and_rows") == (kv, (320 // B, 10))
          and pair(ref, REF, "exact_kv_and_rows") == (kv_ref, (3840 // B, 65)))
    check("exact_kv_and_all_rows: dense queries over every conditioning row",
          pair(t2v, T2V, "exact_kv_and_all_rows") == (kv, kv)
          and pair(ref, REF, "exact_kv_and_all_rows") == (kv_ref, kv_ref))
    no_audio_t2v = {k: v for k, v in t2v.items() if k != "sol_h3_audio_span"}
    check("exact_kv_and_rows without an audio span falls back to the all-rows range",
          pair(no_audio_t2v, T2V, "exact_kv_and_rows") == pair(t2v, T2V, "exact_kv_and_all_rows") == (kv, kv))
    rows_t2v = pair(t2v, T2V, "exact_kv_and_rows")[1]
    all_t2v = pair(t2v, T2V, "exact_kv_and_all_rows")[1]
    rows_ref = pair(ref, REF, "exact_kv_and_rows")[1]
    all_ref = pair(ref, REF, "exact_kv_and_all_rows")[1]
    text_blocks = 320 // B
    ref_blocks = (3840 - 320) // B
    check("the two dense ranges differ by the text rows on t2v and by the reference rows on ref2v",
          rows_t2v[0] - all_t2v[0] == text_blocks
          and rows_ref[0] - all_ref[0] == text_blocks + ref_blocks
          and rows_t2v[1] == all_t2v[1] and rows_ref[1] == all_ref[1],
          f"t2v {rows_t2v[0] - all_t2v[0]} blocks, ref2v {rows_ref[0] - all_ref[0]} blocks")
    check("no video span, or a sequence shorter than it, is zeros in every mode",
          all(pair({}, T2V, m) == ((0, 0), (0, 0)) for m in modes)
          and all(pair(t2v, 640 - 1, m) == ((0, 0), (0, 0)) for m in modes))
    try:
        pair(t2v, T2V, "exact_kv_rows"); refused = False
    except ValueError:
        refused = True
    check("an unknown mode is refused, not run as exact_kv", refused)
    try:
        schema = node.MiniMaxH3SolAttn.define_schema()
        combo = next(i for i in schema.inputs if i.id == "sink_conditioning")
        options, default = list(combo.options), combo.default
    except Exception as exc:                                  # noqa: BLE001
        print(f"  SKIP the combo's options   define_schema not readable here: {exc}")
    else:
        check("the combo lists exactly the modes the function accepts, default among them",
              options == list(modes) and default in modes and default == "exact_kv_and_rows",
              f"{options}, default {default}")


def load_node():
    """The Sol node module. With a card, through the pack's entrypoint, as the
    server loads it. Without one, the single module under a bare package:
    the entrypoint imports ComfyUI's model management, which insists on a
    device at import, and the sink cases need no device."""
    if torch.cuda.is_available():
        load("h3x", REPO / "__init__.py", package_dir=REPO)
    else:
        import types
        import comfy.cli_args as cli_args
        cli_args.args.cpu = True
        pkg = types.ModuleType("h3x")
        pkg.__path__ = [str(REPO)]
        sys.modules["h3x"] = pkg
    return load("h3x.sol_attn_h3", REPO / "sol_attn_h3.py")

# Every kernel case by name, in the order they run, so an OOM can say which
# were not graded rather than "every case".
KERNEL_CASES = (
    "dispatch == kernel at the shipped selection",
    "dispatch == kernel under top-k",
    "dispatch with blk_cnt == dispatch without",
    "sink pair reaches the kernel",
    "pooled_tail reaches the kernel",
    "a transposed oracle is caught",
)


def kernel_cases(node, ck, check, device="cuda", shape=(1, 8, 16384, 128)):
    """The dispatch against the kernel call it should be making, bitwise."""
    torch.manual_seed(0)
    # A realistic length, which the kernel oracle allows and an O(T^2) eager
    # oracle would not.
    b, h, t, d = shape
    # BHND, which is how `optimized_attention` hands H3's attention over.
    q, k, v = (torch.randn(b, h, t, d, device=device, dtype=torch.bfloat16)
               for _ in range(3))
    common = dict(skip_reshape=True, skip_output_reshape=True, scale=None,
                  min_tokens=12288, verbose=False)

    def dispatch(**kw):
        return node._run(q, k, v, h, **{**common, "tau": 1.0, **kw})

    def kernel(transpose_oracle=False, **kw):
        """The kernel call the dispatch should be making, done by hand."""
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        if transpose_oracle:                      # heads against tokens
            qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (qs, ks, vs))
        out = ck.sol_attn(qs, ks, vs, **kw)
        return out.transpose(1, 2)

    print("our node's dispatch against the kernel call it should be making:")
    print(f"  B={b} H={h} T={t} D={d} bf16, bitwise\n")

    for label, kw in (("at the shipped selection", dict(tau=1.0)),
                      ("under top-k", dict(tau=1.0, topk_ratio=0.10))):
        got, want = dispatch(**kw), kernel(**{**kw, "tail": True})
        same = torch.equal(got, want)
        check(f"dispatch == kernel {label}", same,
              "same bytes" if same else
              f"DIFFER: max abs "
              f"{float((got.float() - want.float()).abs().max()):.3e}")

    # The observer's passthrough: a count buffer handed to `_run` reaches the
    # kernel, comes back bounded, and moves no byte of the output. Graded here
    # rather than in check_sol_observe.py because THIS file owns "dispatch ==
    # kernel", and observation is a second way to call the dispatch.
    n = (t + 63) // 64
    buf = torch.empty(b, h, n, dtype=torch.int32, device=device)
    with_counts = dispatch(tau=1.0, blk_cnt=buf)
    check("dispatch with blk_cnt == dispatch without",
          torch.equal(with_counts, dispatch(tau=1.0))
          and 1 <= int(buf.min()) and int(buf.max()) <= n,
          f"same bytes; counts in [{int(buf.min())}, {int(buf.max())}] of {n} blocks")

    sink = dispatch(tau=1.0, sink_blocks=(0, 4), sink_q=(0, 4))
    check("sink pair reaches the kernel",
          torch.equal(sink, kernel(tau=1.0, tail=True,
                                   sink_blocks=[0, 4], sink_q=[0, 4]))
          and not torch.equal(sink, dispatch(tau=1.0)),
          "a non-zero sink both arrives and changes the output")

    print("\nred controls:")
    base = dispatch(tau=1.0)
    off = dispatch(tau=1.0, tail=False)
    moved = not torch.equal(base, off)
    c = cosine(base, off)
    check("pooled_tail reaches the kernel", moved,
          f"cos {c:.6f} against tail=True -- connected" if moved else
          "turning the pooled tail off changed nothing; it is not reaching "
          "the kernel and every case above is vacuous")

    swapped = kernel(transpose_oracle=True, tau=1.0, tail=True)
    caught = not (swapped.shape == base.shape and torch.equal(base, swapped))
    check("a transposed oracle is caught", caught,
          "a heads/tokens swap does not match, so the equality above is "
          "actually seeing layout"
          if caught else
          "a transposed oracle still matches; this file cannot see the defect "
          "class it exists for")


def graded_kernel_cases(node, ck, check, **kw):
    """Run the kernel cases; 0 when they all ran, 2 when the card was busy.

    **An OOM here is not a failure and must not print as one.** This box runs
    a resident ComfyUI, so a model left loaded from a render leaves under a
    GiB free while these shapes want about two, and a render in flight leaves
    less. Before 2026-09-04 the guard for this wrapped nothing, so the check
    died on a torch traceback with exit 1, which reads exactly like a real
    mismatch -- and a check that goes red while the state is correct trains a
    reader to ignore red, which is the one thing docs/checks.md says is worse
    than having no check. Exit 2, naming the cases that were not graded and
    the fix. The sink cases before this point stand on their own result."""
    graded = []

    def recording(name, ok, detail=""):
        graded.append(name)
        check(name, ok, detail)

    try:
        kernel_cases(node, ck, recording, **kw)
    except torch.OutOfMemoryError as exc:
        missing = [c for c in KERNEL_CASES if c not in graded]
        try:
            free, total = torch.cuda.mem_get_info()
            vram = f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}"
        except Exception:                                    # noqa: BLE001
            vram = "VRAM figures unavailable"
        print(f"\n  SKIP  the card was busy: {type(exc).__name__} ({vram}).\n"
              f"        This is an environment state, not a result. A resident model or a\n"
              f"        render in flight is the usual cause; free the card (POST /free with\n"
              f"        unload_models, or wait for the render), then re-run.\n"
              f"        Not graded ({len(missing)} of {len(KERNEL_CASES)}): {', '.join(missing)}")
        return 2
    return 0


def oom_control(node):
    """RED CONTROL for the wrapper, with the card masked: a kernel that raises
    the allocator's error on its first call must come back as 2 with every
    kernel case named as not graded, and no case marked ok or FAIL."""
    class Busy:
        @staticmethod
        def sol_attn(*a, **kw):
            raise torch.OutOfMemoryError("fixture: CUDA out of memory")

    def run(*a, **kw):
        raise torch.OutOfMemoryError("fixture: CUDA out of memory")

    saved = node._run
    node._run = run
    marks = []
    try:
        rc = graded_kernel_cases(node, Busy, lambda n, ok, d="": marks.append(n),
                                 device="cpu", shape=(1, 2, 256, 128))
    except torch.OutOfMemoryError:
        rc = "escaped"                     # the wrapper let the allocator's error through
    finally:
        node._run = saved
    ok = rc == 2 and not marks
    print(f"  {'ok  ' if ok else 'FAIL'} a busy card exits 2 with nothing marked"
          + ("" if ok else f"   rc {rc}, marked {marks}"))
    return 0 if ok else 1


def main():
    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(REPO / "bench"))

    node = load_node()

    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("the sink pair per mode, on CPU:")
    sink_cases(node, check)
    print()

    if "--oom-control" in sys.argv[1:]:
        print("the OOM wrapper, card masked:")
        return oom_control(node) or (1 if failures else 0)

    if not torch.cuda.is_available():
        print("no CUDA; the kernel cannot run. The kernel cases were not graded.")
        if failures:
            print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
            return 1
        return 2
    import comfy_kitchen as ck                              # noqa: E402
    if not hasattr(ck, "sol_attn"):
        print("this comfy_kitchen has no sol_attn; the kernel cases were not graded.")
        return 1 if failures else 2

    rc = graded_kernel_cases(node, ck, check, device="cuda")
    if rc == 2:
        return 1 if failures else 2

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
