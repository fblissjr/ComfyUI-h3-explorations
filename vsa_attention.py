"""VSA (Video Sparse Attention) for MiniMax-H3, on comfy-kitchen's Sol kernel.

## EXPERIMENTAL, AND NOT FINISHED. Read this before quoting anything from it

**Every input this node needs is a draft or an experiment**, and none of it is
a release. Nothing here has rendered end to end. Treat a result from this node
as a report about draft code, not about VSA.

  core support   `github.com/comfyanonymous/ComfyUI` PR #15958, "Minimax-H3:
                 support FastVideo VSA", by kijai. **A DRAFT**, head `10febb01`
                 on base `0a33ed6c`, applied to this box on 2026-08-30 as an
                 UNCOMMITTED working-tree change on master. So the H3 model
                 this box builds is not the one stock ComfyUI builds, and any
                 H3 measurement taken here has to say so.
                 `bench/check_vsa_core_patch.py` is the provenance record and
                 the thing that notices a half-applied patch.
  the checkpoint `huggingface.co/Kijai/MiniMax-H3-experimental`,
                 `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot`.
                 The repository says experimental in its name. The artifact
                 carries NO metadata at all -- no training record, no schedule,
                 no step count -- so everything it claims about itself lives in
                 its filename, including the "4step".
                 `docs/research/vsa/fastvideo_vsa_checkpoint.md` takes it apart.
  the kernel     `comfy_kitchen`'s `coarse_gate`, from
                 Comfy-Org/comfy-kitchen#117. This half IS merged and released,
                 and is the only one of the three that is.
  the method     VSA, "Faster Video Diffusion with Trainable Sparse Attention"
                 (arXiv 2505.13389). Not read here beyond its abstract and a
                 summary; this node follows the KERNEL's contract and the one
                 other H3 implementation, not the paper.

**What is unexercised.** The gate projection, the kernel call and the output
reordering under a real forward. `bench/check_vsa_geometry.py` asserts the
reorder and the refusals, and that is all it asserts.

## Read this first: VSA is a regime, not a knob on the Sol node

Three arguments arrived on `comfy_kitchen.sol_attn` when Sol-Attn merged
upstream -- `tail`, `block_len` and `coarse_gate` -- and they read like three
independent knobs. They are one feature, used together, and upstream's own
tests group them as "VSA-style pieces: padded tiles, no tail, gated coarse
branch". VSA is:

    tail=False        no pooled correction term; softmax over routed blocks only
    coarse_gate=G     plus a gated coarse branch, G * softmax(qm km^T) vm
    block_len         because the tiles are PADDED, one cube per 64-row block
    topk_ratio        SLA-style selection rather than a tau threshold

`MiniMaxH3SolAttn` cannot reach the last two, and not because nobody wired
them. It installs an `optimized_attention_override`, which is handed Q, K and V
already built -- but the gate is a projection of the BLOCK INPUT, taken before
`qkv_proj`, and the cube tiling has to reorder and pad the sequence before the
projection too. **Both need the block forward replaced.** That is this node.

## What it needs, and what happens today without it

**A VSA-trained checkpoint.** The gate is a learned `to_gate_compress` linear
per block, not a scalar anyone can dial. A constant gate is not "VSA mode": it
adds an untrained global-average term to every output and keeps the sparsity.
`docs/research/vsa/fastvideo_vsa_checkpoint.md` takes the published one apart.

**Core support for loading it.** As of 2026-08-30, ComfyUI master has no
`gate_compress` in `comfy/ldm/minimax/model.py` and no detection for it in
`comfy/model_detection.py`; Comfy-Org/ComfyUI#15958 is a draft that adds both.
Without it the checkpoint's gate keys have no slot on the constructed model and
are dropped with a warning -- **the render then succeeds and gives you the
dense base checkpoint**. That is the failure mode this node exists to make
loud: `_gate_modules` refuses by name rather than running something that looks
like VSA and is not.

Installing that PR is necessary and NOT sufficient. Its own comment says the
gate is "unused by the dense forward; consumed by sparse attention patches" --
so core loads the weight and something else has to compute it and pass it.

## What is asserted here rather than assumed

  video is the last segment    core's own comment ("target audio then target
                               video, always the last two segments") and its
                               `PackedLayout.segments`. Asserted anyway, because
                               the cube grid is built from the tail of the
                               sequence and a layout with anything after video
                               would tile the wrong rows.
  the grid matches the rows    `latent_t * (latent_h//2) * (latent_w//2)` must
                               equal the video segment's length, or the reorder
                               is addressing tokens that are not there.
  every main block has a gate  all 50 or none. A checkpoint with a partial set
                               would silently run some blocks VSA and some
                               dense, which is neither thing.

## Credit, and where this differs

The shape of the integration follows
`coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py`, read
2026-08-30, which got there first and is the only other implementation of VSA
for H3 anywhere. Two deliberate differences:

  - **It restricts itself to plain text/audio/video packing and falls back to
    the dense block otherwise.** This one accepts any prefix, because the
    prefix is chunked into 64-row blocks and made a sink either way -- what the
    geometry actually needs is that VIDEO IS LAST, which core guarantees. So
    reference graphs work here.
  - It refuses when an `optimized_attention_override` is present. This one
    warns instead, because every shipped graph here wires sage: the override is
    correctly bypassed on the 50 main blocks this node replaces, and correctly
    still runs on the 2 token-refiner blocks, which have no gate and are not
    VSA's business.
"""

from __future__ import annotations

import logging
import math

import torch

from comfy_api.latest import io

logger = logging.getLogger(__name__)

# One VSA cube per 64-row kernel block. 4*4*4 = 64 exactly, which is why the
# cube shape and the kernel's block size are not independent choices.
CUBE = (4, 4, 4)
BLOCK = 64
assert math.prod(CUBE) == BLOCK, "a cube must fill exactly one kernel block"

_GEOMETRY_CACHE: dict = {}


def _geometry(prefix_lengths, video_grid):
    """(destination_by_source, block_len, prefix_blocks, padded_rows), on CPU.

    `destination_by_source[i]` is the padded row that source row `i` moves to.
    Prefix segments are chunked 64 rows at a time and keep their order; video
    rows are grouped into 4x4x4 cubes, one cube per block, zero-padded to 64.

    Padding is why `block_len` exists: a cube at the edge of the grid holds
    fewer than 64 real rows, and the kernel must be told so or it will treat
    zeros as keys and fold them into the block means.
    """
    key = (tuple(prefix_lengths), tuple(video_grid))
    hit = _GEOMETRY_CACHE.get(key)
    if hit is not None:
        return hit

    order: list[int] = []
    lengths: list[int] = []
    offset = 0
    for length in prefix_lengths:
        for start in range(0, length, BLOCK):
            count = min(BLOCK, length - start)
            order.extend(range(offset + start, offset + start + count))
            lengths.append(count)
        offset += length
    prefix_blocks = len(lengths)

    frames, height, width = video_grid
    ct, ch, cw = CUBE
    for t0 in range(0, frames, ct):
        for h0 in range(0, height, ch):
            for w0 in range(0, width, cw):
                cube = [offset + (t * height + h) * width + w
                        for t in range(t0, min(t0 + ct, frames))
                        for h in range(h0, min(h0 + ch, height))
                        for w in range(w0, min(w0 + cw, width))]
                order.extend(cube)
                lengths.append(len(cube))

    total = offset + frames * height * width
    if len(order) != total:
        raise RuntimeError(
            f"VSA geometry covered {len(order)} rows of {total}; the cube walk "
            f"and the packed layout disagree")

    destination = torch.empty(total, dtype=torch.long)
    cursor = 0
    for index, count in enumerate(lengths):
        rows = torch.tensor(order[cursor:cursor + count], dtype=torch.long)
        destination[rows] = index * BLOCK + torch.arange(count, dtype=torch.long)
        cursor += count

    hit = (destination, torch.tensor(lengths, dtype=torch.int32),
           prefix_blocks, len(lengths) * BLOCK)
    _GEOMETRY_CACHE[key] = hit
    return hit


def _layout_geometry(layout, tokens, device):
    """Geometry for one packed layout, or a RuntimeError naming what is wrong."""
    segments = list(getattr(layout, "segments", ()) or ())
    if not segments:
        raise RuntimeError("the packed layout published no segments")
    if segments[-1][2] != "video":
        raise RuntimeError(
            f"VSA tiles the video segment and expects it last; this layout ends "
            f"with {segments[-1][2]!r}. Core packs target audio then target "
            f"video as the last two segments, so a layout that does not is "
            f"either a new packing or a different model.")
    signature = tuple(getattr(layout, "signature", ()) or ())
    if len(signature) != 5:
        raise RuntimeError("the packed layout carries no 5-tuple signature")
    _text, latent_t, latent_h, latent_w, _audio = (int(v) for v in signature)
    grid = (latent_t, latent_h // 2, latent_w // 2)

    start, stop, _ = segments[-1]
    if math.prod(grid) != stop - start:
        raise RuntimeError(
            f"video grid {grid} is {math.prod(grid)} rows but the segment is "
            f"{stop - start}; the cube walk would address rows that are not there")
    if stop != tokens:
        raise RuntimeError(
            f"the video segment ends at {stop} and the sequence is {tokens} rows")

    prefix = tuple(int(b - a) for a, b, _ in segments[:-1])
    destination, block_len, prefix_blocks, padded = _geometry(prefix, grid)
    return (destination.to(device), block_len.to(device), prefix_blocks, padded)


def _scatter_rows(value, destination, padded_rows):
    """Move rows into padded, cube-major order. Padding rows stay zero."""
    out = value.new_zeros((padded_rows, *value.shape[1:]))
    out[destination] = value
    return out


def _gate_modules(model, block_count):
    """The per-block `to_gate_compress` linears, or (None, why).

    Refuses on a partial set as loudly as on an empty one. A checkpoint
    carrying gates for some blocks would run those VSA and the rest dense,
    which is neither regime and would look like a quality result.
    """
    # `get_model_object` falls through to `comfy.utils.get_attr`, which is a
    # plain `getattr` loop -- so BOTH a missing attribute and an out-of-range
    # block index raise AttributeError, the latter because `nn.ModuleList`
    # registers its children as string-named attributes. Verified by execution
    # against this install's core, not carried over from the pack this node's
    # shape follows. KeyError is caught too, for the object-patch branch above
    # it, and costs nothing.
    gates = []
    for index in range(block_count):
        path = f"diffusion_model.blocks.{index}.attn.to_gate_compress"
        try:
            gate = model.get_model_object(path)
        except (AttributeError, KeyError):
            if index == 0:
                return None, (
                    "this model has no `to_gate_compress`, so it is not a "
                    "VSA-trained checkpoint -- or ComfyUI cannot build the "
                    "slot for one.\n\n"
                    "If you loaded a VSA checkpoint and got this, that is the "
                    "expected result on stock ComfyUI: `gate_compress` reaches "
                    "`comfy/ldm/minimax/model.py` only through "
                    "Comfy-Org/ComfyUI#15958, which is still a draft. Without "
                    "it the gate weights have nowhere to go and are dropped on "
                    "load with a warning -- and the render then SUCCEEDS, "
                    "giving you the dense base checkpoint. This node refuses "
                    "rather than let that pass for VSA.")
            return None, (
                f"block {index} has no `to_gate_compress` while block 0 does. "
                f"A partial gate set would run some blocks VSA and some dense.")
        if not isinstance(getattr(gate, "weight", None), torch.Tensor):
            return None, f"the gate at block {index} carries no weight tensor"
        gates.append(gate)
    return gates, None


def _vsa_attention(attn, gate, x, rope_freqs, geometry, topk_ratio, tail):
    """One block's attention, VSA-style. Mirrors `Attention.forward`.

    Reproduced rather than called because every difference is upstream of
    `optimized_attention`: the rows are reordered and padded before `qkv_proj`,
    and the gate is a projection of `x` itself.
    """
    from comfy.ldm.minimax import model as h3
    import comfy_kitchen

    destination, block_len, prefix_blocks, padded = geometry
    x = _scatter_rows(x, destination, padded)
    if rope_freqs is not None:
        rope = rope_freqs.new_zeros(
            (rope_freqs.shape[0], padded, *rope_freqs.shape[2:]))
        rope[:, destination] = rope_freqs
    else:
        rope = None

    heads, head_dim = attn.heads, attn.head_dim
    q, k, v = attn.qkv_proj(x).split(heads * head_dim, dim=-1)
    v = v.view(padded, heads, head_dim)
    if rope is not None:
        # Same fused RMSNorm + split-half rope core runs, on the padded rows.
        q = q.view(1, padded, heads, head_dim)
        k = k.view(1, padded, heads, head_dim)
        qw = h3.comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = h3.comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope.shape[-3] * 2
        if h3.comfy.model_management.in_training:
            q, k = h3.comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            h3.comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q.view(padded, heads, head_dim))
        k = attn.k_norm(k.view(padded, heads, head_dim))
    v = v.clone()

    # The gate: a projection of the BLOCK INPUT, which is the whole reason this
    # cannot be done from an attention override. Same padded row order as q/k/v,
    # and q's exact shape, which `sol_attn_common_call_rule` requires.
    coarse_gate = gate(x).view(1, padded, heads, head_dim)

    out = comfy_kitchen.sol_attn(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        topk_ratio=topk_ratio, tail=tail,
        block_len=block_len, coarse_gate=coarse_gate,
        sink_blocks=[0, prefix_blocks], sink_q=[0, prefix_blocks])

    # Back to source order, dropping the padding rows, and flattened to the
    # (rows, heads*head_dim) the output projection consumes.
    return attn.out_proj(out[0, destination].flatten(-2))


_LAYOUTS: dict = {}
_LAYOUT_PATCHED = set()
_FORWARD_PATCHED = set()


def _publish_layout(diffusion_model):
    """Make the packed layout reachable from inside a block replacement.

    A block replacement is handed `img`, `t_emb`, `mod_segments`, `rope_freqs`
    and `transformer_options` -- not the layout, and not `position_ids`. So the
    layout is registered by the identity of the `position_ids` tensor it built
    (there is one per distinct shape, and the layout is kept alive by the same
    entry so the id cannot be recycled underneath us), and republished into
    `transformer_options` from `rope_freqs`, which is the one call that receives
    that tensor and runs once per forward.

    The same trick as `sol_attn_h3.install_h3_morton`, and deliberately its own
    copy: the two nodes are alternatives, so VSA must not need the Sol node
    installed to see a layout.
    """
    import sys
    module = sys.modules[type(diffusion_model).__module__]
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")

    if id(layout_cls) not in _LAYOUT_PATCHED:
        original_init = layout_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if torch.is_tensor(getattr(self, "position_ids", None)):
                _LAYOUTS[id(self.position_ids)] = self

        layout_cls.__init__ = __init__
        _LAYOUT_PATCHED.add(id(layout_cls))

    if id(diffusion_model) in _FORWARD_PATCHED:
        return
    original_forward = diffusion_model._forward
    original_rope = diffusion_model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        diffusion_model._vsa_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            diffusion_model._vsa_options = None
            transformer_options.pop("h3_vsa_layout", None)

    def rope_freqs(position_ids, device):
        options = getattr(diffusion_model, "_vsa_options", None)
        if isinstance(options, dict):
            options["h3_vsa_layout"] = _LAYOUTS.get(id(position_ids))
        return original_rope(position_ids, device)

    diffusion_model._forward = _forward
    diffusion_model.rope_freqs = rope_freqs
    _FORWARD_PATCHED.add(id(diffusion_model))


def _make_block(block, gate, topk_ratio, tail, index):
    """The replacement for one DiT block. Mirrors `DiTBlock.forward`."""
    from comfy.ldm.minimax import model as h3

    def replacement(args, extra):
        layout = args["transformer_options"].get("h3_vsa_layout")
        if layout is None:
            if index == 0:
                logger.warning("[h3-vsa] no packed layout published this "
                               "forward; running dense")
            return extra["original_block"](args)
        try:
            geometry = _layout_geometry(layout, args["img"].shape[0],
                                        args["img"].device)
        except RuntimeError as exc:
            if index == 0:
                logger.warning(f"[h3-vsa] {exc}; running dense")
            return extra["original_block"](args)

        x, t_emb, mod_segments = args["img"], args["t_emb"], args["mod_segments"]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaln_proj(t_emb)
        h = h3._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
        x = h3._mod_gate(x, gate_msa,
                         _vsa_attention(block.attn, gate, h, args["rope_freqs"],
                                        geometry, topk_ratio, tail),
                         mod_segments)
        h = h3._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return {"img": h3._mod_gate(x, gate_mlp, block.mlp(h), mod_segments)}

    return replacement


class MiniMaxH3VSAAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VSAAttention",
            display_name="MiniMax H3 VSA Attention",
            is_experimental=True,
            category="model/attention/minimax",
            description=(
                "EXPERIMENTAL AND UNFINISHED. Needs a DRAFT ComfyUI PR "
                "(comfyanonymous/ComfyUI#15958) for core to load the gate at "
                "all, and an experimental checkpoint from "
                "Kijai/MiniMax-H3-experimental whose only self-description is "
                "its filename. Nothing has been rendered through this node. "
                "See docs/research/vsa/.\n\n"
                "FastVideo VSA (Video Sparse Attention) for MiniMax-H3, on "
                "comfy_kitchen's Sol kernel. Replaces the 50 main DiT blocks: "
                "video tokens are grouped into 4x4x4 cubes, one cube per "
                "64-row kernel block, and each block's learned "
                "`to_gate_compress` supplies the gate on VSA's coarse "
                "branch.\n\n"
                "REQUIRES a VSA-trained checkpoint. On any other model this "
                "node refuses rather than running -- a constant gate is not "
                "VSA, and a checkpoint whose gate weights failed to load "
                "renders as the dense base without saying so.\n\n"
                "Do not combine with a Sol-Attn node: both decide how the same "
                "50 blocks attend. A sage node upstream is fine and is left to "
                "handle the 2 token-refiner blocks, which have no gate."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("keep_percent", default=10.0, min=0.5, max=95.0,
                               step=0.5,
                               tooltip="Percent of key blocks each query block "
                                       "attends exactly. VSA's published "
                                       "sparsity is 0.90, which is 10 here. "
                                       "This is the fraction the checkpoint was "
                                       "distilled at, not a free quality dial: "
                                       "moving it away from the training value "
                                       "is off-distribution in the direction "
                                       "the distillation cannot help with."),
                io.Boolean.Input("pooled_tail", default=False,
                                 tooltip="Leave OFF. VSA has no Sol-style "
                                         "pooled correction -- its coarse "
                                         "branch is what covers the unselected "
                                         "blocks, and adding a pooled term on "
                                         "top gives the model two global paths "
                                         "where it was trained with one. "
                                         "Exposed so the arm can be run, not "
                                         "because it should be."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, keep_percent, pooled_tail) -> io.NodeOutput:
        import comfy_kitchen
        for name in ("sol_attn",):
            if not hasattr(comfy_kitchen, name):
                raise RuntimeError(
                    "this comfy_kitchen has no sol_attn; VSA runs on the same "
                    "kernel Sol-Attn does. bench/check_sol_kernel.py reports "
                    "which build is installed.")
        import inspect
        params = inspect.signature(comfy_kitchen.sol_attn).parameters
        missing = [k for k in ("topk_ratio", "tail", "block_len", "coarse_gate")
                   if k not in params]
        if missing:
            raise RuntimeError(
                f"comfy_kitchen.sol_attn does not accept {missing}. VSA needs "
                f"the merged kernel (Comfy-Org/comfy-kitchen#117); a pre-merge "
                f"build cannot express the coarse branch at all. Both report "
                f"version 0.2.31, so read the local version segment.")

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None or not hasattr(diffusion_model, "rope_freqs"):
            raise RuntimeError(
                f"this node only patches MiniMax-H3; got "
                f"{type(diffusion_model).__name__}")

        gates, why = _gate_modules(model, len(blocks))
        if gates is None:
            raise RuntimeError(f"VSA is not available on this model: {why}")

        options = model.model_options.get("transformer_options", {})
        if options.get("patches_replace", {}).get("dit"):
            raise RuntimeError(
                "another node already replaces the H3 DiT blocks; VSA needs "
                "the block forward and cannot share it.")
        if "optimized_attention_override" in options:
            logger.warning(
                "[h3-vsa] an attention override is installed (sage, or a "
                "Sol-Attn node). VSA replaces the block forward, so that "
                "override is BYPASSED on all %d main blocks and still runs on "
                "the 2 token-refiner blocks. If that override is a Sol-Attn "
                "node, remove it -- both decide how these blocks attend.",
                len(blocks))

        _publish_layout(diffusion_model)

        m = model.clone()
        replace = m.model_options["transformer_options"].setdefault(
            "patches_replace", {}).setdefault("dit", {})
        for index, block in enumerate(blocks):
            replace[("double_block", index)] = _make_block(
                block, gates[index], keep_percent / 100.0, pooled_tail, index)

        logger.info("[h3-vsa] VSA on %d blocks, keep %.1f%% of key blocks, "
                    "4x4x4 cubes, pooled_tail=%s", len(blocks), keep_percent,
                    pooled_tail)
        return io.NodeOutput(m)
