"""Run named DiT blocks on EXACT attention -- neither sage nor Sol-Attn.

## Why this node exists, and what `dense_blocks` does not do

`SolAttnMiniMax`'s `dense_blocks` reads as "keep these blocks exact". It does
not. `vendor/sol_attn_minimax.py::make_override`'s `dense()` hands the call to
`previous`, and on every shipped graph here `previous` is **sage**. So the
knob chooses sage over Sol at those blocks and never reaches exact attention.
Nothing in this pack did, before this node.

That matters most where a graph would most want it. Measured on real captures
(`bench/results/2026-08-18_sage_accuracy_on_capture.json`), sage's error grows
with depth and **block 49 is where it breaks down**: rel L2 ~0.031 against
~0.005 at block 0, and `cos_min` goes NEGATIVE (-0.04 to -0.11), meaning that
on some rows sage's output is anti-correlated with the exact answer. The last
block is both the one a distilled output head reads directly and the one the
fallback kernel is worst at.

**This node is a knob, not a recommendation.** No shipped graph wires it, and
what it buys end to end has not been measured here -- only what it removes at
the call. See `docs/SOLATTN.md`.

## What "exact" means, precisely

ComfyUI's own `Attention.forward` with the attention OVERRIDE removed from
`transformer_options`, so the call lands on whatever backend ComfyUI resolves
for the device -- flash or SDPA in bf16, the same path a graph with no
attention node at all would take. It is not float64, and it is not a reference
implementation; it is this box's unaccelerated answer.

## Ordering, and why it survives being wrapped

Sol-Attn re-wraps any foreign `attn.forward` from a block pre-hook, so a patch
installed downstream of it does not simply win by being later. Both of Sol's
composition sites -- the patch-time loop in `_apply_patch` and the run-time
`pre_hook` in `_install_compose_hooks` -- skip a forward carrying
`_uses_optimized_attention`, so this forward sets it and is left alone in
either node order.

**That flag's name promises slightly more than this forward delivers, and the
mismatch is deliberate.** Sol's own comment reads "patch routes through
optimized_attention; the override composes directly". The first half is true
here -- `Attention.forward` calls `optimized_attention` and so does this. The
second is not: the override is stripped, on purpose, because reaching it is
exactly what this node exists to prevent. Branching on the flag is branching on
a name rather than an observable, which this repo warns against, so the
behaviour is asserted rather than trusted: `bench/check_exact_blocks.py` builds
the real chain and checks that a named block reaches neither kernel.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io


def _exact_forward(self, x, rope_freqs=None, transformer_options={}):
    """ComfyUI's stock Attention.forward with the attention override removed.

    Imported inside the call rather than at module scope, so that importing
    this pack cannot fail on a ComfyUI without the H3 model module.
    """
    from comfy.ldm.minimax.model import Attention

    # KJNodes' low-VRAM block patch hands `x` over in a single-item list. The
    # stock forward does not expect that, so unwrap exactly as this pack's sage
    # forward does; the release that list is buying is given up either way.
    if isinstance(x, list):
        x = x.pop()

    # Copy before mutating. `transformer_options` is the live dict the sampler
    # threads through every block -- removing the override in place would
    # disable sage and Sol for the REST of the model, turning a two-block
    # request into a whole-model change with nothing to show it.
    if isinstance(transformer_options, dict) and \
            "optimized_attention_override" in transformer_options:
        transformer_options = {k: v for k, v in transformer_options.items()
                               if k != "optimized_attention_override"}

    return Attention.forward(self, x, rope_freqs=rope_freqs,
                             transformer_options=transformer_options)


# Tells Sol-Attn's compose sites to leave this forward alone. See the module
# docstring for why the name overstates what this does. `setattr` rather than
# an attribute assignment only to keep type checkers quiet about a function
# growing a field, which is legal and is the protocol Sol reads.
setattr(_exact_forward, "_uses_optimized_attention", True)


class MiniMaxH3ExactBlocks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ExactBlocks",
            display_name="MiniMax H3 Exact Blocks",
            category="model/attention/minimax",
            is_experimental=True,
            description=(
                "Run the named DiT blocks on ComfyUI's own attention, with "
                "neither SageAttention nor Sol-Attn. Sol's `dense_blocks` does "
                "NOT do this: a block it keeps 'dense' falls through to sage. "
                "Place this AFTER the sage node; either side of Sol-Attn "
                "works. Costs roughly 1.7x the sage time on the blocks it "
                "names, on every step."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "blocks", default="",
                    tooltip=(
                        "Blocks to run exactly, same syntax as Sol's "
                        "dense_blocks: '48,49', '0-2,-1'. Negative indices "
                        "count from the end, so -1 is the last block. Empty "
                        "means this node does nothing. The measured case for "
                        "this is the LAST block, where sage's cosine against "
                        "an exact reference goes negative on some rows; see "
                        "docs/SOLATTN.md."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, blocks="") -> io.NodeOutput:
        diffusion_model = model.get_model_object("diffusion_model")
        dit_blocks = getattr(diffusion_model, "blocks", None)
        if dit_blocks is None:
            raise RuntimeError(
                f"{type(diffusion_model).__name__} has no .blocks to index; "
                f"this node only patches a MiniMax H3 DiT."
            )
        # Deferred import, matching how `nodes.py` reaches the other vendored
        # module: importing the Sol module at pack-import time would pull
        # comfy_kitchen in for every user, including those who never wire it.
        # Shared rather than reimplemented, so the two nodes cannot disagree
        # about what "0-2,-1" means.
        from .vendor.sol_attn_minimax import parse_blocks

        count = len(dit_blocks)
        wanted = parse_blocks(blocks, count)

        m = model.clone()
        if not wanted:
            # An empty spec is the node's default and is a legitimate state,
            # not a mistake -- say so at the same volume as a hit, so a graph
            # that wires this node and forgot to fill it in is visible in the
            # log rather than silently inert.
            logging.info("[h3] exact blocks: none named, this node is inert")
            return io.NodeOutput(m)

        for i in sorted(wanted):
            m.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward",
                               _exact_forward.__get__(dit_blocks[i].attn,
                                                      type(dit_blocks[i].attn)))
        logging.info(
            f"[h3] exact blocks: {sorted(wanted)} of {count} run on ComfyUI's "
            f"own attention -- neither sage nor Sol")
        return io.NodeOutput(m)
