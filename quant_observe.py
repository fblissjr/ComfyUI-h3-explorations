"""Install the Tier 1 activation observer on an H3 DiT.

`dit_observe.py` is the recorder; this is the wiring. Two gates, deliberately:
the node must be in the graph AND `H3_QUANT_OBSERVE` must be set. Either alone
does nothing. A node left in a saved graph must not arm a later render, and an
armed server must not instrument a graph that did not ask for it.

## The patch set, and why it is three per block and not four

    blocks.N.attn.qkv_proj.forward   ordinary call  -> wrapper
    blocks.N.attn.out_proj.forward   ordinary call  -> wrapper
    blocks.N.mlp.forward             wrapper covering BOTH mlp kinds
    diffusion_model.forward          step, sigma, PackedLayout.segments

**`mlp.fc2.forward` is never called on the shipped INT8 path.** H3's MLP body
is `comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`
(`comfy/ldm/minimax/model.py:205-206`), so the fused helper reads `fc2.weight`
directly and a patch on `fc2.forward` records nothing. This repo has already
paid for that once -- `unmerged_blocks` silently dropped fc2 on 2026-08-30 --
and an external review caught the same design here before it ran. So the MLP
seam is wrapped instead, and `dit_observe._shape_check` asserts four kinds
rather than inferring them.

**The wrapper REPLICATES core's two-line MLP body rather than calling it.**
Calling the original and then recomputing `fc1` for statistics would double a
104361x5376x28672 matmul per block per step, which is not affordable. So the
wrapper computes `fc1` once, uses it for both, and pays only one extra elementwise
swiglu for the statistics. The cost of that choice is a drift risk: if core
changes `MLP.forward`, this diverges silently. `assert_mlp_shape` below checks
the structure it depends on at install time, and the recorded output names the
replication so a reader knows to check it.

**The outer patch is not optional.** `PackedLayout.segments` reaches the model
through `minimax_payload` and is NOT in `transformer_options`, so a module-level
hook cannot see it. Without it the capture bins positionally and loses the
grouping -- exactly what leaves `grade_sage_on_capture.py` unable to answer a
segment question today.

Sol-Attn composes with `.forward` patches whose owner segment contains "attn",
read as `key.rsplit(".", 2)[-2]`. For these keys that is `qkv_proj`, `out_proj`
and `mlp`, none of which contain "attn", so Sol leaves them alone -- the same
reasoning `pdd_lora.py` records for its own patches.
"""

from __future__ import annotations

import logging
import uuid

from comfy_api.latest import io

from .block_spec import parse_blocks
from . import dit_observe

logger = logging.getLogger(__name__)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _is_minimax_h3(diffusion_model) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def assert_mlp_shape(mlp) -> None:
    """The structure the MLP wrapper replicates, checked at install time.

    Cheap insurance against the one real risk of replicating core's body: a
    core change that renames or restructures these would otherwise produce a
    wrapper that still runs and records something subtly different.
    """
    for attr in ("fc1", "fc2"):
        if not hasattr(mlp, attr):
            raise RuntimeError(
                f"this model's MLP has no `{attr}`, so the fc2 seam this "
                f"observer replicates is not the one core runs. Refusing "
                f"rather than recording a different computation.")


def make_linear_forward(base_forward, block: int, kind: str):
    """Observe a linear's input, then call the original. Changes nothing."""
    def forward(inp, *a, **kw):
        out = base_forward(inp, *a, **kw)
        dit_observe.record(block, kind, inp, out=out)
        return out
    return forward


def make_mlp_forward(mlp, block: int):
    """Both MLP kinds from one seam. Replicates `MLP.forward`; see the header.

    `fc1` is computed ONCE and used for the real call and the statistics. The
    only added arithmetic is a second swiglu over the fc1 output, which is
    elementwise and does not touch the result -- `linear_input_act` still
    receives the un-activated `h` exactly as core hands it.
    """
    import comfy.ops

    def forward(x):
        dit_observe.record(block, "mlp.fc1", x)
        h = mlp.fc1(x)
        try:
            act = comfy.ops.INPUT_ACT_EAGER["swiglu"](h)
            dit_observe.record(block, "mlp.fc2", act)
            del act
        except Exception as exc:                     # noqa: BLE001
            # Recorded, never swallowed: a missing fc2 is the exact failure
            # `_shape_check` exists to catch, so it must reach the record.
            dit_observe._failures.append(
                {"block": block, "kind": "mlp.fc2",
                 "error": f"{type(exc).__name__}: {exc}"[:200]})
        return comfy.ops.linear_input_act(mlp.fc2, h, "swiglu")
    return forward


def make_outer_forward(base_forward, expect_blocks: int, capture_id: str):
    """Publish step, sigma and the packed layout, then flush.

    The module hooks cannot see any of this from where they sit, and the two
    instruments must not infer indices independently -- that is how two
    plausible files end up with different counters.
    """
    def forward(x, timestep, context, transformer_options={}, *a, **kw):
        try:
            sig = float(timestep.flatten()[0]) if timestep is not None else None
            sched = transformer_options.get("sample_sigmas")
            step = None
            if sched is not None and sig is not None:
                diffs = [abs(float(s) - sig) for s in sched]
                step = int(min(range(len(diffs)), key=diffs.__getitem__))
            payload = kw.get("minimax_payload")
            layout = payload.get("layout") if isinstance(payload, dict) else None
            segs = getattr(layout, "segments", None)
            dit_observe.set_context(
                capture_id=capture_id, step=step, sigma=sig,
                # The schedule itself rather than a derived knot index: the
                # grid mapping belongs to the PDD tracker, and re-deriving it
                # here would be a second authority that can disagree. Recorded
                # so the knot is computable offline by whoever owns the grid.
                sample_sigmas=[float(s_) for s_ in sched]
                if sched is not None else None,
                segments=[[int(a_), int(b_), str(k_)] for a_, b_, k_ in segs]
                if segs else None)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("[h3-quant-observe] context unavailable: %s", exc)
        out = base_forward(x, timestep, context, transformer_options, *a, **kw)
        try:
            dit_observe.flush(expect_blocks=expect_blocks)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("[h3-quant-observe] flush failed: %s", exc)
        return out
    return forward


class MiniMaxH3QuantObserve(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3QuantObserve",
            display_name="MiniMax H3 Quant Observe (activation statistics)",
            category="model_patches/minimax",
            description=(
                "Records what each quantised linear's INPUT looks like, per "
                "block and per step, so the ACTIVATION half of int8_convrot's "
                "two roundings can be measured offline. Every quantisation "
                "record in this repo is a stored-weight distance; this is the "
                "other one.\n\n"
                "INERT unless the server was started with H3_QUANT_OBSERVE "
                "set to an output directory. The node alone does nothing, so "
                "a saved graph cannot arm a later render.\n\n"
                "Wire it like an attention node: after the loader, before the "
                "sampler. It changes no tensor the model computes.\n\n"
                "TIMING FROM AN ARMED RUN IS VOID. The reductions and CPU "
                "copies do not change values but do change wall time and peak "
                "memory. Profile in a separate uninstrumented process."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "blocks", default="-1", optional=True,
                    tooltip=(
                        "Which blocks to observe. Same grammar as Sol-Attn's "
                        "dense_blocks and the PDD node's unmerged_blocks: "
                        "'0-2,49', '32', '-1' for all.\n\n"
                        "'-1' is the default and is what the Tier 1 contract "
                        "asks for -- the quantisation sidecar wants all 50 "
                        "blocks, independently of the much narrower block "
                        "filter the attention capture uses. The statistics are "
                        "kilobytes per cell, so all 50 is affordable where "
                        "saving inputs would not be."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, blocks="-1") -> io.NodeOutput:
        dm = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(dm):
            raise RuntimeError(
                f"This node only observes MiniMax H3; got {type(dm).__name__}.")

        if not dit_observe.enabled():
            logger.info(
                "[h3-quant-observe] H3_QUANT_OBSERVE is not set; this node is "
                "inert and the render is unchanged.")
            return io.NodeOutput(model)

        want = parse_blocks(blocks, len(dm.blocks))
        if not want:
            raise RuntimeError(
                f"blocks={blocks!r} selected nothing of this model's "
                f"{len(dm.blocks)} blocks.")

        m = model.clone()
        # **Chain from the PATCHED forward, never from the raw bound method.**
        # `add_object_patch` is last-writer-wins and `get_model_object` returns
        # whatever patch is already installed, so `m.get_model_object(key)` is
        # the only correct base -- it composes with an owner that got there
        # first instead of replacing it.
        #
        # Corrected 2026-08-31 before this ever ran, by an external review. The
        # first version took `dm.forward`, the UNPATCHED method, for the outer
        # patch: on any graph wiring `MiniMaxH3PDDLoRA` -- which patches the
        # same key to publish its step tracker -- this observer would have
        # silently replaced it and broken head selection. Worse, that version
        # carried a collision REFUSAL for the 150 block keys and neither
        # chained nor refused on the one key most likely to collide.
        for i in sorted(want):
            blk = dm.blocks[i]
            assert_mlp_shape(blk.mlp)
            for kind in ("attn.qkv_proj", "attn.out_proj"):
                key = f"diffusion_model.blocks.{i}.{kind}.forward"
                m.add_object_patch(
                    key, make_linear_forward(m.get_model_object(key), i, kind))
            key = f"diffusion_model.blocks.{i}.mlp.forward"
            m.add_object_patch(key, make_mlp_forward(blk.mlp, i))

        capture_id = f"quant-{uuid.uuid4().hex[:12]}"
        m.add_object_patch(
            "diffusion_model.forward",
            make_outer_forward(m.get_model_object("diffusion_model.forward"),
                               expect_blocks=len(want),
                               capture_id=capture_id))

        logger.info(
            "[h3-quant-observe] armed: %d block(s), 4 kinds, writing to %s. "
            "Timing from this run is void.", len(want), dit_observe._dir)
        return io.NodeOutput(m)
