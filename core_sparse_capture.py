"""Capture q/k/v on core's sparse producer path, and change nothing.

Written 2026-09-27 for `docs/state/open_experiments.md` #45 (is part of FastH3's
over-polish the kitchen's VSA selection?). FastH3's contract graph runs core's
`BlockSparseAttention` (`comfy_extras/nodes_sparse_attention.py`). On the calls
it takes, core swaps each H3 block's attention for its sparse producer, which
projects q/k/v in row chunks inside the kitchen kernel. No full q/k/v tensor
ever exists, and none of this pack's attention paths runs, so
`h3_capture.maybe_capture` is never called. A capture of that graph wrote
nothing.

This node sits after `BlockSparseAttention` and wraps, per block, the block
patch core installed (`patches_replace["dit"][("double_block", i)]`). The
wrapper hands core's patch an `original_block` that swaps the block's
`attention` callable for a capturing one. That callable then:
- projects the fused qkv in core's own `PRODUCER_CHUNK` row chunks, straight
  to host memory, only on a call `H3_CAPTURE` asks for (`h3_capture.wants_pre`);
- writes it through `h3_capture.maybe_capture_pre`, with the rope table and the
  q/k norm weights: everything a grader needs to rebuild the q and k that VSA
  selection sees;
- on a model with VSA gates, also writes the coarse-branch gate's output for
  the same call (`gate_*.pt`), which the qkvpre file cannot rebuild;
- counts the call (`h3_capture.count_call`), since no `maybe_capture` runs on
  this path to advance the step counter;
- calls the attention core chose, unchanged.

The projection is built in core's chunk size. On the VSA path core first
reorders rows into its tiles, so the chunk contents differ from these. The
capture equals core's bytes where the projection is row-independent, which
`h3_capture`'s `chunk_check` measures on the real weights and records in each
file. The GPU cost is one chunk; the host holds the full projection for the
calls written. It stamps `_h3_block_index` on each attention module, as `nodes.py`
does, so block indices survive a model swap within one server.

**Needs `pre=1` or `pre=only` in `H3_CAPTURE`**; `maybe_capture_pre` writes
nothing otherwise. With `H3_CAPTURE` unset, the wrapper calls straight through
and the node is inert. Never put it in a graph where one of this pack's
attention nodes also captures: both would count each call.

Nothing here patches core; the hook is the model patcher's own block-replace
slot, chained onto core's.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

try:
    from . import h3_capture as _capture
except ImportError:   # run as a script or from a bench harness
    import h3_capture as _capture

logger = logging.getLogger(__name__)
TAG = "[h3 core sparse capture]"


def _host_chunks(linear, h):
    """`linear(h)` built on the host in core's producer chunk size."""
    try:
        from comfy_extras.nodes_sparse_attention import PRODUCER_CHUNK
        chunk = int(PRODUCER_CHUNK)
    except Exception:                                  # noqa: BLE001 -- a size, any will do
        chunk = 8192
    parts = []
    with torch.no_grad():
        for i in range(0, h.shape[0], chunk):
            parts.append(linear(h[i:i + chunk]).cpu())
    return torch.cat(parts)


def _host_projection(attn, h):
    """The fused qkv projection of `h`, built on the host in core's producer chunks."""
    return _host_chunks(attn.qkv_proj, h)


def _write_gate(attn, h, transformer_options=None):
    """Save the VSA coarse-branch gate's output for this call, beside its qkvpre file.

    VSA's attention output is fine + coarse * gate(x) (core's producer, from
    `attn.to_gate_compress`, no activation). The qkvpre file rebuilds q, k and
    v but not the gate, whose weights are int8 in the checkpoint. The model's
    own gate, applied to this call's input, is what core used. Same (block,
    step, render) indices as the qkvpre file, read without advancing. It
    carries the same server stamp, prompt id and branch fields as the qkvpre
    record, so a capture manifest sees one process (lookingdude, 2026-09-27:
    without them the manifest tool refused the first #45 capture as mixed
    stamps)."""
    gate = getattr(attn, "to_gate_compress", None)
    if gate is None:
        return
    import os
    with _capture._lock:
        block, step = _capture._block_step(attn, advance=False)
        render = _capture._render
    g = _host_chunks(gate, h)
    suffix = f"_r{render}" if render else ""
    name = f"gate_L{int(h.shape[0])}_S{int(h.shape[0])}_b{block}_s{step}{suffix}.pt"
    to = transformer_options if isinstance(transformer_options, dict) else {}
    sigmas = to.get("sigmas")
    try:
        sigma = float(sigmas[0]) if sigmas is not None else None
    except (TypeError, IndexError, ValueError):
        sigma = None
    torch.save({"kind": "coarse_gate", "gate": g, "block": int(block), "step": int(step),
                "render": int(render), "seq_len": int(h.shape[0]), "sigma": sigma,
                "uuids": [str(u) for u in (to.get("uuids") or [])] or None,
                "cond_or_uncond": [int(c) for c in (to.get("cond_or_uncond") or [])] or None,
                "prompt_id": _capture._prompt_id(), "server": _capture._server_stamp(),
                "source": "the model's to_gate_compress on this call's attention input, core's chunk size"},
               os.path.join(_capture._config["dir"], name))
    print(f"[h3_capture] wrote {name}  gate{tuple(g.shape)} {g.dtype}", flush=True)


def _capturing(attn, inner):
    def attention(h, rope_freqs=None, transformer_options={}):
        # Both calls re-read H3_CAPTURE and return at once when it is unset.
        if _capture.wants_pre(attn):
            _write_gate(attn, h, transformer_options)
            qkv = _host_projection(attn, h)
            _capture.maybe_capture_pre(attn, qkv, h, rope_freqs, transformer_options,
                                       length_hint=int(h.shape[0]))
            del qkv
        _capture.count_call(attn)
        return inner(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    return attention


def _wrap(block, existing):
    def block_patch(args, extra):
        original = extra["original_block"]

        def original_capturing(a):
            inner = a.get("attention") or block.attn
            return original({**a, "attention": _capturing(block.attn, inner)})

        if existing is None:
            return original_capturing(args)
        return existing(args, {**extra, "original_block": original_capturing})
    return block_patch


class MiniMaxH3CoreSparseCapture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CoreSparseCapture",
            display_name="MiniMax H3 Core Sparse Capture",
            category="MiniMax H3/debug",
            description=(
                "Put after core's Block Sparse Attention to capture the fused q/k/v it "
                "never materialises, when H3_CAPTURE (with pre=1 or pre=only) is set in "
                "the server's environment. Inert otherwise. A diagnostic: it changes no "
                "output."),
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if blocks is None:
            raise ValueError(f"{TAG} needs a MiniMax-H3 model (no .blocks on {type(dm).__name__})")
        replaces = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        wrapped = 0
        for i, block in enumerate(blocks):
            block.attn._h3_block_index = i
            existing = replaces.get(("double_block", i))
            wrapped += existing is not None
            m.set_model_patch_replace(_wrap(block, existing), "dit", "double_block", i)
        logger.info("%s %d blocks wrapped, %d of them over an existing block patch; capture %s",
                    TAG, len(blocks), wrapped, "ARMED" if _capture.enabled else "off (H3_CAPTURE unset)")
        return io.NodeOutput(m)
