#!/usr/bin/env python3
"""Check that `clone_v` reaches the forward, and only on modes that earn it.

Cloning v before the hand-off cuts peak attention VRAM 9.1% -- but only
when the kernel underneath releases q and k as soon as it quantizes them.
On a kernel that holds them for the whole call it is a flat cost: +571 MiB
at seq=41822, measured in `bench_minimax_attn.py`, and +572 MiB in the
fork's own `tests/test_sageattn_consume.py`.

So the saving depends on two things that live in different files and are
joined by one argument in `nodes.py`. Nothing else notices if that
argument is dropped: every existing check passes with `clone_v` gone,
because none of them look at storage. The rendered frames are identical
either way. It is a silent 9.1% regression, which is exactly the kind
this repo keeps finding after the fact.

Three cases, in the order a regression would reach them:

  predicate_matches_kernel : `mode_releases_qkv` agrees with what
                             `build_kernel` actually returns for that mode.
                             Delete and a new MODES entry can claim a
                             release it does not get.
  node_wires_it            : the node's own `execute` passes the predicate's
                             answer down. Delete and the argument can be
                             dropped in `nodes.py` with nothing to notice.
  clone_v_changes_storage  : `clone_v=True` actually takes v out of the
                             fused buffer. Delete and the flag can become
                             decorative while both cases above still pass.
  device_gate_is_honoured  : a False from sage stops the clone. Delete and
                             the forward can ignore the predicate entirely
                             without anything on this arch noticing, since
                             sm89 answers True.
  chunked_path_does_not_clone : chunking switches the clone off, by both the
                             argument and the transformer_options route.
                             Delete and the KJNodes configuration pays 572
                             MiB for a copy nothing can recover.

Needs CUDA (sage resolves the device arch at import) but no model, no
sampling, and no meaningful VRAM.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

# `nodes.py` imports its siblings relatively, and its name collides with
# ComfyUI's own top-level `nodes`. Import it as this package instead, from
# the custom_nodes directory above -- which also keeps `attention` a single
# module object shared with it, rather than a second copy under its own name.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parent))

attn_mod = importlib.import_module(f"{_REPO.name}.attention")
node_mod = importlib.import_module(f"{_REPO.name}.nodes")

MODES = attn_mod.MODES
build_kernel = attn_mod.build_kernel
make_minimax_attn_forward = attn_mod.make_minimax_attn_forward
mode_releases_qkv = attn_mod.mode_releases_qkv

HIDDEN, HEADS, HEAD_DIM = 5376, 56, 128


class _StubAttn:
    """Just enough of `Attention` for `forward.__get__` and the patch loop."""


class _StubDiffusionModel:
    def __init__(self, blocks=2):
        self.blocks = [type("B", (), {"attn": _StubAttn()})() for _ in range(blocks)]


class _StubModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.patched = []

    def get_model_object(self, _name):
        return _StubDiffusionModel()

    def clone(self):
        return self

    def add_object_patch(self, key, _value):
        self.patched.append(key)


def check_predicate_matches_kernel():
    """The predicate must describe the kernel the mode actually builds."""
    import sageattention

    consume = getattr(sageattention, "sageattn_consume", None)
    failures = []
    for mode in MODES:
        kernel_fn, _kwargs = build_kernel(mode)
        actually_consumes = kernel_fn is consume
        claimed = mode_releases_qkv(mode)
        ok = claimed == actually_consumes
        print(f"  {'ok  ' if ok else 'FAIL'} {mode:22s} "
              f"releases={claimed} kernel_is_consume={actually_consumes}")
        if not ok:
            failures.append(mode)
    return failures


def check_node_wires_it():
    """`execute` must hand the predicate's answer to the forward builder."""
    seen = {}
    real_builder = node_mod.make_minimax_attn_forward
    real_guard = node_mod._is_minimax_h3

    def recording_builder(kernel_fn, kernel_kwargs, head_chunks=1, clone_v=False):
        seen["clone_v"] = clone_v
        return real_builder(kernel_fn, kernel_kwargs, head_chunks=head_chunks,
                            clone_v=clone_v)

    # setattr rather than attribute assignment: these are module globals, and
    # a type checker has no way to know nodes.py defines them.
    setattr(node_mod, "make_minimax_attn_forward", recording_builder)
    setattr(node_mod, "_is_minimax_h3", lambda _m: True)
    try:
        failures = []
        for mode in MODES:
            seen.clear()
            node_mod.MiniMaxH3SageAttention.execute(_StubModel(), mode=mode)
            want = mode_releases_qkv(mode)
            got = seen.get("clone_v")
            ok = got == want
            print(f"  {'ok  ' if ok else 'FAIL'} {mode:22s} "
                  f"node passed clone_v={got}, predicate says {want}")
            if not ok:
                failures.append(mode)
        return failures
    finally:
        setattr(node_mod, "make_minimax_attn_forward", real_builder)
        setattr(node_mod, "_is_minimax_h3", real_guard)


def check_clone_v_changes_storage():
    """The flag has to move v out of the fused buffer, or it is decorative."""
    import comfy.ops
    from comfy.ldm.minimax.model import Attention, rope_rotation_table

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = 64
    failures = []
    for clone_v in (False, True):
        module = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-5, dtype=torch.bfloat16,
                           device=device, operations=comfy.ops.manual_cast)
        module = module.to(device).requires_grad_(False)
        seen = {}

        def recorder(qkv, **_kw):
            q, k, v = qkv
            qkv.clear()
            seen["q"] = q.untyped_storage().data_ptr()
            seen["v"] = v.untyped_storage().data_ptr()
            return torch.zeros_like(q)

        forward = make_minimax_attn_forward(recorder, {}, clone_v=clone_v)
        module.forward = forward.__get__(module, module.__class__)
        rope = rope_rotation_table(
            torch.zeros(seq, 96, device=device, dtype=torch.float32),
            torch.bfloat16)
        module(torch.randn(seq, HIDDEN, device=device, dtype=torch.bfloat16),
               rope_freqs=rope)

        shares = seen["q"] == seen["v"]
        # clone_v=True must break the sharing; clone_v=False must keep it,
        # or the fused case this is all about is not what is being measured.
        ok = shares != clone_v
        print(f"  {'ok  ' if ok else 'FAIL'} clone_v={clone_v!s:5s} "
              f"v shares the fused buffer: {shares}")
        if not ok:
            failures.append(f"clone_v={clone_v}")
    return failures


def check_device_gate_is_honoured():
    """A False from sage must stop the clone, even with the mode saying yes.

    This box answers True, so a forward that ignored the predicate outright
    would be indistinguishable from a correct one in every case above. The
    only way to see the gate from here is to make sage disagree and watch the
    clone not happen -- which is also what keeps this file from grading our
    own answer against itself once the node reads the same predicate.
    """
    import comfy.ops
    from comfy.ldm.minimax.model import Attention, rope_rotation_table

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    seq = 64
    real = attn_mod._prefers_cloned_v
    failures = []
    try:
        for answer in (True, False):
            setattr(attn_mod, "_prefers_cloned_v", lambda _d, a=answer: a)
            module = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-5, dtype=torch.bfloat16,
                               device=device, operations=comfy.ops.manual_cast)
            module = module.to(device).requires_grad_(False)
            seen = {}

            def recorder(qkv, **_kw):
                q, _k, v = qkv
                qkv.clear()
                seen["q"] = q.untyped_storage().data_ptr()
                seen["v"] = v.untyped_storage().data_ptr()
                return torch.zeros_like(q)

            # clone_v=True throughout: the mode says yes, only sage's answer
            # varies, so the storage difference is the gate and nothing else.
            forward = make_minimax_attn_forward(recorder, {}, clone_v=True)
            module.forward = forward.__get__(module, module.__class__)
            rope = rope_rotation_table(
                torch.zeros(seq, 96, device=device, dtype=torch.float32),
                torch.bfloat16)
            module(torch.randn(seq, HIDDEN, device=device, dtype=torch.bfloat16),
                   rope_freqs=rope)

            cloned = seen["q"] != seen["v"]
            ok = cloned == answer
            print(f"  {'ok  ' if ok else 'FAIL'} sage says {answer!s:5s} "
                  f"-> forward cloned: {cloned}")
            if not ok:
                failures.append(f"predicate={answer}")
    finally:
        setattr(attn_mod, "_prefers_cloned_v", real)
    return failures


def check_chunked_path_does_not_clone():
    """Chunking must switch the clone off, by both routes that set it.

    `_chunked_heads` holds q, k and v across every group, so the kernel's
    per-group release frees nothing and the clone is a flat +572 MiB at
    seq=41822 -- worse than chunking nothing and cloning nothing. Storage
    aliasing settles it exactly and needs 64 rows instead of 8 GiB, so the
    peak measurement stays a one-off in the changelog and this is what holds
    the behaviour.

    Both routes are here because they are different code reaching the same
    `n`, and the one that matters is the one we do not control: KJNodes
    publishes `minimax_head_chunks` through transformer_options while our own
    widget stays at 1. A gate written against `head_chunks` passes the
    argument route and lets the published one straight through.
    """
    import comfy.ops
    from comfy.ldm.minimax.model import Attention, rope_rotation_table

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    seq = 64
    failures = []
    # (label, forward's own head_chunks, transformer_options, expect a clone)
    cases = [
        ("chunks=1", 1, {}, True),
        ("chunks=4 via argument", 4, {}, False),
        ("chunks=4 via options", 1, {"minimax_head_chunks": 4}, False),
    ]
    for label, head_chunks, options, want_clone in cases:
        module = Attention(HIDDEN, HEADS, HEAD_DIM, 1e-5, dtype=torch.bfloat16,
                           device=device, operations=comfy.ops.manual_cast)
        module = module.to(device).requires_grad_(False)
        seen = {}

        def recorder(qkv, **_kw):
            q, _k, v = qkv
            qkv.clear()
            # The chunked path calls once per group; the first call is enough
            # to see whether v came out of the fused buffer.
            seen.setdefault("q", q.untyped_storage().data_ptr())
            seen.setdefault("v", v.untyped_storage().data_ptr())
            return torch.zeros_like(q)

        forward = make_minimax_attn_forward(recorder, {}, head_chunks=head_chunks,
                                            clone_v=True)
        module.forward = forward.__get__(module, module.__class__)
        rope = rope_rotation_table(
            torch.zeros(seq, 96, device=device, dtype=torch.float32),
            torch.bfloat16)
        module(torch.randn(seq, HIDDEN, device=device, dtype=torch.bfloat16),
               rope_freqs=rope, transformer_options=options)

        cloned = seen["q"] != seen["v"]
        ok = cloned == want_clone
        print(f"  {'ok  ' if ok else 'FAIL'} {label:22s} cloned={cloned} "
              f"(want {want_clone})")
        if not ok:
            failures.append(label)
    return failures


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.", file=sys.stderr)
        return 0
    # Surface the module under check, so a rename shows up here rather than
    # as a quietly skipped case.
    assert hasattr(attn_mod, "mode_releases_qkv"), (
        "attention.mode_releases_qkv is what nodes.py gates the clone on"
    )

    failures = []
    for fn in (check_predicate_matches_kernel, check_node_wires_it,
               check_clone_v_changes_storage, check_device_gate_is_honoured,
               check_chunked_path_does_not_clone):
        print(f"{fn.__name__}:")
        failures += fn()

    if failures:
        print(f"\n{len(failures)} case(s) failed: {', '.join(map(str, failures))}")
        return 1
    print("\nclone_v is wired to the modes that earn it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
