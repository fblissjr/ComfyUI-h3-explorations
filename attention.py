"""SageAttention forward for MiniMax H3's packed self-attention.

H3 runs one unmasked self-attention per DiT block over the whole packed
`[text | cond | audio | video]` sequence -- 56 heads, head_dim 128, no
mask anywhere. That is exactly the shape SageAttention's sm89 INT8-QK /
FP8-PV kernel is built for, and at the node's default canvas it lands
about 2.7x ahead of torch's flash backend.

Replacing the block's `forward` rather than going through ComfyUI's
`optimized_attention` buys two things:

  - q/k/v stay in NHD, the layout `qkv_proj` already produces, instead of
    being transposed into HND and back.
  - the float q/k/v can be handed to sage as a list it takes ownership
    of, so they are released as soon as their quantized forms exist
    rather than at the end of the call.

Everything else -- the fused RMSNorm + split-half RoPE, the output
projection -- is left exactly as the stock forward's *inference* path does
it, including running in place on the qkv buffer. The stock forward also
has a `comfy.model_management.in_training` branch that calls the
non-in-place `rms_rope_split_half`; this file does not mirror it, because
sageattn has no backward and a training step through this forward would
fail at the kernel regardless of which rope variant ran.
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_LOGGED = False


def _log_fallback_once(exc):
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED:
        _FALLBACK_LOGGED = True
        logger.warning(
            "[h3] sage kernel raised (%s: %s); this block and any "
            "later failure fall back to ComfyUI's attention for the rest of "
            "the run. The render continues, just without sage.",
            type(exc).__name__, exc,
        )


def reset_fallback_state():
    """Let the next run report a fallback again instead of staying quiet."""
    global _FALLBACK_LOGGED
    _FALLBACK_LOGGED = False


def _sage():
    """Import sageattention lazily so a missing install is a node-time
    error the user can read, not an import-time traceback at startup."""
    try:
        import sageattention
    except ImportError as exc:
        raise RuntimeError(
            "sageattention is not installed in this ComfyUI's environment. "
            "This node needs the Ada fork (SageAttention-ada) built from "
            "source; see the README."
        ) from exc
    return sageattention


# mode -> (attribute on sageattention, extra kwargs). "auto" lets sage's
# own dispatcher pick, which is correct on every card we support; the
# explicit entries exist so a suspected accuracy problem can be bisected
# without editing code.
#
# A `None` attribute means "go through sageattn_consume", which releases
# the float q/k/v as soon as they are quantized. sageattn_consume applies
# its pv_accum_dtype with setdefault, so naming one here overrides the
# dispatcher's pick without giving up that release -- which matters,
# because on Ada "auto" already resolves to fp8++ and a user picking it
# explicitly would otherwise silently lose ~435 MiB per call for nothing.
# Only the fp16 kernel has no consuming entry point.
MODES = {
    "auto": (None, {}),
    "fp8++ (fastest)": (None, {"pv_accum_dtype": "fp32+fp16"}),
    "fp8": (None, {"pv_accum_dtype": "fp32+fp32"}),
    "fp16 (most accurate)": ("sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
}


_CLONE_V_BY_DEVICE = {}


def _prefers_cloned_v(device):
    """Ask sage whether cloning v pays on `device`, once per device.

    This is the half of the clone decision we cannot answer ourselves.
    `mode_releases_qkv` covers the mode; sage covers the arch, and owns
    flipping the answer if its own fused-case peak ever drops below what a
    cloning caller can reach -- at which point the clone becomes a cost while
    the release it depends on is still happening. Gating on the predicate
    rather than on an arch set means that reaches us on upgrade.

    Asked with the device the tensors are actually on, not the one the model
    was patched from: ComfyUI patches before loading or casting, so at patch
    time the model can still be on the CPU, and on a multi-GPU box that bakes
    an answer for the wrong card. The predicate is a list index behind an
    import-time table, so calling it per forward would be free anyway; the
    cache is only here to keep the hot path free of attribute lookups.
    """
    if device.type != "cuda":
        # Our kernels are CUDA-only, so this call is already on its way to a
        # fallback. Sage raises on a non-CUDA device rather than answering,
        # deliberately -- and a raise here would escape the try/except that
        # makes that fallback graceful.
        return False

    index = device.index if device.index is not None else torch.cuda.current_device()
    hit = _CLONE_V_BY_DEVICE.get(index)
    if hit is None:
        predicate = getattr(_sage(), "sageattn_consume_prefers_cloned_v", None)
        # A fork old enough to have `sageattn_consume` but not the predicate
        # keeps the behaviour it already had rather than silently losing the
        # saving. build_kernel is what enforces the floor on fork age.
        hit = True if predicate is None else bool(predicate(index))
        _CLONE_V_BY_DEVICE[index] = hit
    return hit


def mode_releases_qkv(mode):
    """Whether `mode`'s kernel frees the float q/k/v as soon as it quantizes.

    Only the modes that go through `sageattn_consume` do. It decides whether
    cloning v is worth paying for: measured at seq=41822, the clone saves
    286 MiB on a releasing kernel and costs 571 MiB on one that holds q/k/v
    for the whole call. Note that sage itself only takes the early-release
    path on the sm89-family fp8 kernels; on an arch that falls back to the
    ordinary path this still returns True and the clone is a small loss.
    """
    return MODES[mode][0] is None


def build_kernel(mode):
    """Return `(fn(qkv_list, **kw) -> NHD output, kwargs)` for `mode`."""
    sa = _sage()
    if not hasattr(sa, "sageattn_consume"):
        raise RuntimeError(
            "The installed sageattention has no sageattn_consume(). This "
            "node needs the Ada fork at a version that provides it; a stock "
            "SageAttention install will not work."
        )

    attr, extra = MODES[mode]
    # A note for anyone arriving from KJNodes' "pad V to CTA_K=128 in H3 mem-eff
    # sage sm90" fix: that bug is not reachable from here. It comes from
    # reimplementing sage's internals and skipping the kv_len pad that the
    # top-level sm90 entry point does. We go through sageattn_consume, whose
    # fp8 dispatch is gated on arch in {sm89, sm100, sm120, sm121} -- sm90
    # never reaches the fp8 kernel on this path.
    base_kwargs = {
        "tensor_layout": "NHD",
        "is_causal": False,
        # H3 self-attention is unmasked, and ComfyUI passes smooth_k=False
        # on its own sage path. Keeping it off avoids a K-mean pass that
        # buys nothing measurable at these shapes.
        #
        # It is also what makes the `clone_v` below pay. With smooth_k=True,
        # `per_thread_int8` allocates q_int8/k_int8 before evaluating
        # `k = k - km`, so a full bf16 copy of K sits on top of them and eats
        # the saving: the clone goes from -286 MiB to +286 MiB at seq=41822.
        # Turning this on means turning that off. Measured in the fork's
        # `tests/test_sageattn_consume.py`.
        "smooth_k": False,
        **extra,
    }

    if attr is None:
        return sa.sageattn_consume, base_kwargs

    # No consuming entry point for this kernel: unpacking the list binds
    # q/k/v into this frame for the duration of the call, so the tensors
    # stay alive and the memory saving is lost. Correct, just heavier --
    # acceptable for a diagnostic mode, and called out in the tooltip.
    kernel = getattr(sa, attr)

    def call_without_release(qkv, **kw):
        q, k, v = qkv
        qkv.clear()
        return kernel(q, k, v, **kw)

    return call_without_release, base_kwargs


def make_sage_override(kernel_fn, kernel_kwargs, previous=None):
    """An `optimized_attention_override` that routes eligible calls to sage.

    The forward patch below is the fast path and handles everything on its
    own -- it calls sage directly and never reaches `optimized_attention`.
    This exists for the case where another patch runs ComfyUI's *stock*
    forward instead of ours, which is how Sol-Attn takes a call: its
    compose gate hands eligible calls to the stock forward so they reach
    its own override.

    Without an override of ours in place, every path Sol-Attn declines
    *after* that point -- a mask, its kernel returning None, or a kernel
    error -- falls through to ComfyUI's default attention rather than back
    to sage. Installing this makes sage the fallback instead, which is
    what Sol-Attn's own `previous` chaining is for.

    `previous` preserves any override already on the model, so this
    composes rather than clobbers.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def fallback():
            target = func if previous is None else functools.partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        # Sage has no mask support on this path, and a custom softmax scale
        # is not plumbed through here. Both are rare on H3 (its self-attn
        # passes neither) but wrong silently if assumed.
        if mask is not None or kwargs.get("scale") is not None:
            return fallback()

        # Shape joins mask and scale as a reason to decline rather than to
        # raise. This override exists to catch calls another patch handed
        # back, so the one case where a safety net earns its keep is a caller
        # nobody predicted -- and until 2026-08-11 an unexpected ndim killed
        # the render instead of degrading. Our own chain probe was that
        # caller, which is how it was found.
        want_ndim = 4 if skip_reshape else 3
        if q.ndim != want_ndim or k.ndim != want_ndim or v.ndim != want_ndim:
            _log_fallback_once(ValueError(
                f"expected {want_ndim}D q/k/v for skip_reshape={skip_reshape}, "
                f"got {q.ndim}D/{k.ndim}D/{v.ndim}D"))
            return fallback()

        if skip_reshape:
            b, _, _, dim_head = q.shape
            layout = "HND"
        else:
            b, _, dim = q.shape
            dim_head = dim // heads
            q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
            layout = "NHD"
        if q.dtype not in (torch.bfloat16, torch.float16) or dim_head > 128:
            return fallback()

        # Deliberately does NOT hand ownership to the kernel here. Dropping
        # q/k/v would free them earlier, but `fallback` closes over those
        # names, so releasing them turns a kernel failure into a NameError
        # instead of a graceful degrade. This path only runs when another
        # patch is driving; a working fallback is worth more here than the
        # per-call saving, which the forward patch still gets.
        try:
            out = kernel_fn([q, k, v], **dict(kernel_kwargs, tensor_layout=layout))
        except Exception as exc:
            _log_fallback_once(exc)
            return fallback()

        if layout == "HND":
            return out if skip_output_reshape else \
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out.transpose(1, 2) if skip_output_reshape else \
            out.reshape(b, -1, heads * dim_head)

    return override


def make_minimax_attn_forward(kernel_fn, kernel_kwargs, head_chunks=1,
                              clone_v=False):
    """Build a replacement `Attention.forward` bound to one sage kernel.

    `kernel_fn(qkv_list, **kernel_kwargs)` must consume the `[q, k, v]`
    list and return an NHD-shaped output.

    `clone_v` gives v its own storage before the hand-off. q, k and v are
    three views of one fused qkv buffer, so a kernel that releases q and k
    as soon as they are quantized frees nothing -- v still pins the whole
    allocation. `sageattn_consume`'s own docstring measures this: the same
    call that saves 858 MiB on separately allocated q/k/v saves exactly
    zero on fused views. Cloning v costs one third of the buffer to let
    the other two thirds go early. ComfyUI does the same thing upstream
    (`comfy/ldm/minimax/model.py`, "Fix peak memory issue with H3").

    Measured on this repo's `bench_minimax_attn.py`, head_chunks=1, one arm
    per process, four canvases at 124 frames: a 9.1% lower peak at every
    shape (-286 MiB at seq=41822 down to -174 MiB at seq=25406) for 0.7-1.0%
    more time. That lands sage a few MiB under the stock forward's own peak
    while staying ~2.1x faster.

    It is not free everywhere, which is why `nodes.py` ties it to
    `mode_releases_qkv` rather than turning it on outright: on the fp16 mode,
    whose kernel holds q/k/v for the whole call, the same clone costs 571 MiB
    at seq=41822.

    It is also off whenever the heads are chunked, measured rather than
    assumed: at seq=41822, chunks=4 costs +572 MiB with the clone (3217 vs
    2645), which is the clone's own size recovered by nothing, and lands
    above the 3148 of chunking nothing and cloning nothing. `_chunked_heads`
    holds q, k and v for the whole loop, so the kernel's per-group release
    has nothing to free.

    `clone_v=True` is a permission, not an instruction: the forward asks
    `_prefers_cloned_v` about the device it is actually running on, and
    checks the resolved chunk count, before paying for the copy.

    **Every number above is per attention call, and that is not a render.**
    Measured 2026-08-11 and worth stating plainly, because the whole reason
    to want the clone was an assumption this contradicts. An e2e run at 124
    frames put head_chunks=4 *higher* on process peak than head_chunks=1 by
    1186 MiB, the opposite direction from the per-call figures, and a second
    run on the same box measured a 2265 MiB spread across two runs of one
    unchanged configuration. So the excursion is larger than the effect and
    the sign cannot be settled at that sample size either way.

    Fragmentation at the call is not the explanation: this bench now reports
    reserved beside allocated and they track within 8 MiB on all three arms,
    so a single call on a clean allocator does not fragment. Whatever drives
    process peak lives in the interaction across 50 blocks and every step
    with a model resident and ComfyUI's dynamic VRAM reallocating -- exactly
    what a per-module bench excludes by construction.

    What survives: the clone is free (0.7% wall-clock, bit-identical output)
    and it lowers the attention call's peak. What does not: any claim that it
    lowers what a user's card reports. Do not spend anything to get it.

    `head_chunks` > 1 runs the heads in that many groups, quantizing and
    attending one group at a time so the kernel's internal transients shrink
    by roughly the group count. It costs that many kernel launches per call
    instead of one. On a 24 GB 4090 the headroom this buys was measured to
    convert to wall-clock at a ~2.6% ceiling (`workflows/h3_config.py`), so
    this defaults off and exists to make the 1-vs-4 A/B that config asks for
    runnable through our node rather than only KJNodes'.

    A `transformer_options["minimax_head_chunks"]` published by KJNodes'
    MiniMaxLowVRAMAttention is honoured when the node's own input is left at
    1. Without that, installing their node alongside ours silently does
    nothing: their head chunking lives in a forward that ours displaces.
    """

    def forward(self, x, rope_freqs=None, transformer_options={}):
        import comfy.model_management
        import comfy.quant_ops

        # KJNodes' MiniMaxLowVRAMAttention patches the *block* forward to hand
        # `x` over in a single-item list, so attention can free the block's
        # normed h right after the qkv GEMM. That block patch is installed
        # whether or not its attn patch won the object-patch key, so this
        # forward sees the list in either node order -- and Sol-Attn's compose
        # gate passes the list through untouched on calls it declines.
        #
        # We take the tensor out but keep holding it, giving up the release
        # KJNodes is buying. `_stock_forward` recomputes from x, and a working
        # fallback is worth more here than ~250 MiB per call -- same trade the
        # override path makes below.
        if isinstance(x, list):
            x = x.pop()

        s = x.shape[0]
        # One fused projection, split into three views of the same buffer.
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        v = v.view(1, s, self.heads, self.head_dim)

        if rope_freqs is not None:
            # Same fused per-head RMSNorm + partial split-half rope the stock
            # forward uses, in place on the qkv buffer.
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2,
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        n = head_chunks
        if n <= 1 and isinstance(transformer_options, dict):
            n = transformer_options.get("minimax_head_chunks", 1)
        n = max(1, min(int(n), self.heads))

        # Deliberately below the whole of the `n` resolution above, including
        # the transformer_options read -- not beside the clone's old home. The
        # chunked path keeps q, k and v alive across every group, so the
        # kernel's per-group release frees nothing and the clone is a flat
        # cost with nothing to recover it: measured +572 MiB at seq=41822,
        # which lands chunking-plus-cloning above doing neither. Gating this
        # on `head_chunks` instead of `n` would read 1 in exactly the case
        # that motivated the fix, since KJNodes delivers its value through
        # transformer_options while our own widget stays at 1.
        if n == 1 and clone_v and _prefers_cloned_v(x.device):
            # After rope, which only writes q and k. v's own storage is
            # what lets the fused buffer die once the kernel has consumed
            # q and k; see the docstring above. Three gates, because the
            # question has three halves: `clone_v` is the mode's answer,
            # settled when this forward was built; the predicate is the
            # device's; and `n` is the path's, known only once resolved.
            v = v.clone()

        if n > 1:
            try:
                out = _chunked_heads(self, q, k, v, s, n,
                                     kernel_fn, kernel_kwargs)
            except Exception as exc:
                _log_fallback_once(exc)
                del q, k, v
                return _stock_forward(self, x, rope_freqs, transformer_options)
            del q, k, v  # last refs to the fused qkv buffer, before out_proj
            return self.out_proj(out.view(s, self.heads * self.head_dim))

        # This branch is load-bearing, not an optimisation for the trivial
        # case. sage measured the release suppressed by a slicing loop of
        # *one* group as completely as by four: the saving tracks whether the
        # caller still holds the parents, not the group count. Routing n=1
        # through `_chunked_heads` would turn -286 MiB into +572 with nothing
        # visible in the output. `check_clone_v_wiring.py` pins it.
        qkv = [q, k, v]
        del q, k, v  # the list is now the only owner

        try:
            out = kernel_fn(qkv, **kernel_kwargs)
        except Exception as exc:
            # The kernel consumes the list, so there is nothing left to
            # retry with -- recompute from x through ComfyUI's own forward.
            # Wasteful, but this path only runs when sage has already
            # failed and the alternative is failing the render.
            _log_fallback_once(exc)
            del qkv
            return _stock_forward(self, x, rope_freqs, transformer_options)

        return self.out_proj(out.view(s, self.heads * self.head_dim))

    return forward


def _chunked_heads(self, q, k, v, s, n, kernel_fn, kernel_kwargs):
    """Attend `n` head groups in turn, writing into one output buffer.

    The kernel takes ownership of each group's list, but the groups are
    *views* into the fused qkv buffer, so nothing is actually freed until
    the caller drops q/k/v. The saving here is in the kernel's own
    transients -- the int8/fp8 copies it makes per call -- which scale with
    the head count it is handed, not with what the caller still holds.

    Uneven splits go to the earlier groups (`i < heads % n`), so 56 heads
    over 5 groups is 12/11/11/11/11 rather than a ragged final group of 4.
    """
    out = torch.empty((1, s, self.heads, self.head_dim),
                      dtype=q.dtype, device=q.device)
    start = 0
    for i in range(n):
        end = start + self.heads // n + (1 if i < self.heads % n else 0)
        out[:, :, start:end] = kernel_fn(
            [q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]],
            **kernel_kwargs)
        start = end
    return out


def _stock_forward(self, x, rope_freqs, transformer_options):
    """Re-run ComfyUI's own unpatched Attention.forward from scratch."""
    from comfy.ldm.minimax.model import Attention

    return Attention.forward(
        self, x, rope_freqs=rope_freqs, transformer_options=transformer_options
    )
