"""Fit a reference image the way the reference pipeline does, including up.

`MiniMaxH3ReferenceToVideo` sizes reference images with

    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))     # nodes_minimax_h3.py:226

and the reference pipeline with

    scale = reference_image_short_edge / min(width, height)  # diffusers
                                    # modular_pipelines/minimax_h3/before_encoder.py:490

Same constant (2048), same round-to-32, one difference: the `min(1.0, ...)`.
ComfyUI never upscales. So a reference smaller than 2048 on its short edge
reaches the DiT at its original size, where the reference would have enlarged
it -- and reference tokens are latent rows, so a 512px reference contributes
16x fewer of them than the released pipeline gives it. Identity fidelity is
the whole job of a reference image, and it is being conditioned on a fraction
of the rows the model was built to see.

This node closes that gap the same way `MiniMaxH3KeyframeCanvas` closes the
canvas one: it does the resize itself, so the stock node's own scale becomes
`min(1.0, 2048/2048) = 1.0` and its resize is a no-op. It composes rather
than replacing.

**Upscaling is not free and the node says so.** Reference rows ride through
every sampling step, so quadrupling the short edge multiplies those rows by
16 and the stock node's own tooltip already warns that `max` "can be several
times slower". The `latent_rows` output is there to make that visible before
you queue a render rather than after.

It also carries the reference's 1:4..4:1 check on reference images
(`before_encoder.py:488`), which ComfyUI does not have -- the same limit
`h3_rules` holds for the canvas.
"""

from __future__ import annotations

import contextlib
import logging

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE,
                                           REF_IMAGE_SHORT_EDGE, _resize)

try:
    from .h3_rules import aspect_in_range, describe_aspect_range
except ImportError:
    from h3_rules import aspect_in_range, describe_aspect_range  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class MiniMaxH3ReferenceFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceFit",
            display_name="MiniMax H3 Reference Resolution",
            category="model/conditioning/minimax",
            description=(
                "Scales a reference image to MiniMax H3's 2048 short edge, "
                "upscaling when it is smaller -- which the reference pipeline "
                "does and ComfyUI does not. Wire it between Load Image and "
                "MiniMax H3 Reference to Video, with that node's "
                "ref_image_size on 'max', and its own resize becomes a no-op. "
                "Reference rows ride every sampling step, so read latent_rows "
                "before queueing."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input(
                    "allow_upscale", default=True, tooltip=(
                        "Scale the image until its shorter side reaches "
                        "short_edge, enlarging it if it is smaller. "
                        "On: matches the released pipeline, which upscales "
                        "unconditionally. A 1280x720 reference becomes "
                        "3648x2048, going from 880 to 7296 vision tokens. "
                        "Off: matches ComfyUI, which only ever shrinks. "
                        "The two differ by exactly this clamp and nothing "
                        "else. Upscaling adds tokens, not detail, so whether "
                        "it helps an already-small source is unmeasured here."
                    )),
                io.Int.Input(
                    "short_edge", default=REF_IMAGE_SHORT_EDGE, min=256,
                    max=4096, step=32, tooltip=(
                        "Scale the image until its SHORTER side reaches this, "
                        "then round to 32. A 1280x720 reference has a short "
                        "side of 720, so 2048 scales it by 2.844 in both "
                        "directions, giving 3648x2048 and 7296 vision tokens "
                        "against 880 unscaled. "
                        "2048 is a property of the released checkpoint, not a "
                        "derivation: the reference pipeline carries it beside "
                        "the canvas rules as `reference_image_short_edge` and "
                        "would change it for a different checkpoint. "
                        "Image references are deliberately exempt from the "
                        "768x1344 area cap that binds the video, which is why "
                        "a reference may legitimately reach 7.5 megapixels "
                        "when the video itself cannot exceed about one. "
                        "Lowering it trades identity fidelity for sequence "
                        "length, which is the actual cost knob here -- and "
                        "note that upscaling adds tokens, not detail, so "
                        "whether it helps an already-small source is "
                        "unmeasured."
                    )),
                # APPENDED, and it has to stay last: widget values map
                # positionally in every saved graph.
                io.Boolean.Input(
                    "lift_downstream_clamp", default=False, optional=True,
                    display_name="EXPERIMENTAL: lift the 2048 clamp",
                    tooltip=(
                        "EXPERIMENTAL. Leave this off unless you are running "
                        "an experiment and expect to throw the result away. "
                        "It monkeypatches a core ComfyUI node for one call and "
                        "pushes the model outside the distribution it was "
                        "trained on; nothing downstream is tested there. "
                        "Only matters above 2048. MiniMax H3 Reference to "
                        "Video clamps with min(1.0, 2048/short_edge), so "
                        "anything larger this node produces is scaled straight "
                        "back and the setting appears to do nothing. Turning "
                        "this on lifts that clamp for exactly one downstream "
                        "call, then restores it. "
                        "Above 2048 is off-distribution: 2048 is what the "
                        "released checkpoint conditioned image references at. "
                        "Cost climbs quadratically -- 3072 is 16,416 vision "
                        "tokens against 7,296 -- per reference, on a sequence "
                        "that already crosses the int32 threshold at 345 "
                        "frames. Requires ref_image_size on 'max'; under "
                        "'match' the constant is never read and this logs that "
                        "it did nothing."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                # Named `latent_rows` since 2026-08-13. `_tokens` returns the
                # DiT's packed rows, `(h//32)*(w//32)`, not Qwen vision tokens
                # -- the two coincide below Qwen's max_pixels and diverge above
                # it. The description and the module docstring both already
                # called this `latent_rows`; the output did not. display_name
                # is free to change, unlike node_id and position.
                io.Int.Output(display_name="latent_rows"),
            ],
            # Hidden inputs are not part of `inputs=[]`, so adding them moves
            # no widget position. `prompt` is what lets this node see whether
            # the downstream node is actually on 'max'.
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, lift_downstream_clamp=False, **kwargs):
        """Force re-execution whenever the experimental clamp lift is armed.

        The arm is a side effect of `execute`, and a cached node does not run.
        So editing only the prompt text downstream left this node cached, the
        arm never fired, and the render silently reverted to the 2048 clamp
        with the checkbox still ticked. Returning a changing value here keeps
        the node out of the cache for exactly the configuration that depends
        on it, and leaves normal use cached.
        """
        return float("nan") if lift_downstream_clamp else None

    @classmethod
    def execute(cls, image, allow_upscale=True, short_edge=REF_IMAGE_SHORT_EDGE,
                lift_downstream_clamp=False
                ) -> io.NodeOutput:
        if image.shape[0] > 1:
            logger.warning(
                "[h3] reference carries %d images; using the first. Wire one "
                "node per reference instead.", image.shape[0])
        src_h, src_w = int(image.shape[1]), int(image.shape[2])

        # The reference refuses out-of-range reference images outright, the
        # same limit it applies to the canvas.
        if not aspect_in_range(src_w, src_h):
            raise RuntimeError(
                f"A MiniMax H3 reference image must be within "
                f"{describe_aspect_range()}; this one is {src_w}x{src_h} "
                f"({src_w / src_h:.3g}). Crop it before referencing it."
            )

        full = short_edge / min(src_w, src_h)
        scale = full if allow_upscale else min(1.0, full)

        tw, th = _fit(src_w, src_h, scale)
        out = _resize(image[:1], tw, th, "disabled")
        # What the DiT actually pays: reference latents are 16x downsampled
        # spatially, and every one of these rows is attended at every step.
        tokens = _tokens(tw, th)
        # Always against ComfyUI's current behaviour, so the log answers "what
        # is this node changing" rather than restating what it just did.
        stock_tokens = _tokens(*_fit(src_w, src_h, min(1.0, full)))

        # Clear this node's own arm FIRST, unconditionally. The previous code
        # only disarmed when the checkbox was off, so the "nothing to lift"
        # branch -- the one that says out loud it is doing nothing -- was the
        # branch that let a previous prompt's 3072 through.
        node_id = getattr(cls.hidden, "unique_id", None)
        disarm_short_edge_override(node_id)

        # Does the node we feed actually read the constant we are about to
        # lift? Under core's default `ref_image_size='match'` it never does:
        # references are sized from the video's pixel area instead, this node's
        # resize is undone, and the log below would otherwise claim an
        # improvement that did not happen.
        downstream = _downstream_ref_image_size(
            getattr(cls.hidden, "prompt", None), node_id)
        effective = downstream in (None, "max")

        logger.info(
            "[h3] reference %dx%d -> %dx%d (allow_upscale=%s, short_edge=%d): "
            "%d latent rows, %.2gx ComfyUI's own sizing%s",
            src_w, src_h, tw, th, allow_upscale, short_edge, tokens,
            tokens / stock_tokens,
            "" if effective else "  -- BUT SEE THE WARNING BELOW",
        )
        if not effective:
            logger.warning(
                "[h3] MiniMax H3 Reference to Video is on ref_image_size=%r, "
                "so it sizes references from the video's pixel area and never "
                "reads the %d constant. This node's resize is undone "
                "downstream: the %d rows above will NOT be what the DiT sees, "
                "and you are paying two lanczos resamples for nothing. Set it "
                "to 'max'.", downstream, REF_IMAGE_SHORT_EDGE, tokens)

        if lift_downstream_clamp:
            if short_edge > REF_IMAGE_SHORT_EDGE:
                arm_short_edge_override(short_edge, node_id)
            else:
                logger.info(
                    "[h3] lift_downstream_clamp is on but short_edge is %d, at "
                    "or below the %d clamp, so there is nothing to lift.",
                    short_edge, REF_IMAGE_SHORT_EDGE)

        return io.NodeOutput(out, tokens)


def _downstream_ref_image_size(prompt, node_id):
    """`ref_image_size` of the ReferenceToVideo this node feeds, or None.

    None means "could not tell" -- no prompt, no consumer found, or a graph
    shape this does not understand -- and is deliberately treated as "fine"
    rather than as a warning, because a false alarm on every render would be
    worse than the silence it replaces.
    """
    if not prompt or node_id is None:
        return None
    node_id = str(node_id)
    sizes = set()
    for spec in prompt.values():
        if not isinstance(spec, dict) or spec.get("class_type") != "MiniMaxH3ReferenceToVideo":
            continue
        inputs = spec.get("inputs") or {}
        feeds = any(isinstance(v, list) and v and str(v[0]) == node_id
                    for v in inputs.values())
        if feeds:
            sizes.add(inputs.get("ref_image_size", "match"))
    if len(sizes) != 1:
        return None
    return sizes.pop()


def _fit(w, h, scale):
    def snap(v):
        return max(CANVAS_MULTIPLE,
                   round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def _tokens(w, h):
    """Vision tokens a reference of this pixel size contributes to the DiT.

    Two stages, not one. The VAE compresses space by 16, then the DiT
    patchifies that latent with `patch_size=(1, 2, 2)` before anything is
    attended (`comfy/ldm/minimax/model.py`, `patchify_video`). Counting only
    the VAE stage reports four times what the sequence actually carries,
    which is what this function did until 2026-08-11 and what the number in
    the 0.3.0 changelog entry came from.
    """
    return (h // 32) * (w // 32)


# --------------------------------------------------------------------------
# Lifting ComfyUI's 2048 clamp, for one downstream call
# --------------------------------------------------------------------------
#
# `MiniMaxH3ReferenceToVideo.execute` sizes image references with
# `min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))` (`nodes_minimax_h3.py:226`).
# The clamp is what lets this node work at all below 2048: we resize first,
# the stock scale resolves to 1.0, and its resize is a no-op. Above 2048 the
# same clamp scales our work back down, so `short_edge` is one-directional
# and a sweep past the default silently does nothing.
#
# `REF_IMAGE_SHORT_EDGE` is a module attribute read inside `execute` at call
# time, so rebinding it changes behaviour. Rebinding it *globally* would be
# the wrong fix: it is sticky for the process, invisible in the UI, and would
# reach graphs that do not contain this node -- the same silent-contamination
# class `nodes.py` guards against when it copies transformer_options. So the
# override is armed by this node, consumed by exactly one downstream call,
# and cleared in a `finally`.
#
# Above 2048 is off-distribution by construction: 2048 is what the released
# checkpoint conditioned image references at, carried in the reference
# pipeline as `ConfigSpec("reference_image_short_edge", 2048)`. This exists to
# make that measurable, not because bigger is better.

# Keyed by the arming node's unique_id, not a bare value. With one global
# value, two fit nodes in one graph resolved by execution order: the one with
# the checkbox OFF called a global disarm and silently cancelled the other's
# arm, and ComfyUI's order between independent nodes is not the graph's visual
# order and not settable. Per-node entries make disarm affect only its own.
_PENDING_SHORT_EDGE: dict[str, int] = {}
_WRAP_MARKER = "_h3_explorations_short_edge_wrapper"


@contextlib.contextmanager
def _rebound_short_edge(value):
    """Swap the module constant for the duration of one call."""
    import comfy_extras.nodes_minimax_h3 as core

    previous = core.REF_IMAGE_SHORT_EDGE
    core.REF_IMAGE_SHORT_EDGE = value
    try:
        yield
    finally:
        core.REF_IMAGE_SHORT_EDGE = previous


def _make_wrapper(original):
    """Wrap `ReferenceToVideo.execute` so an armed override applies once.

    Factored out so the behaviour is testable without a VAE, a CLIP or a
    model: `bench/check_short_edge_override.py` calls this with a stub.
    """
    def wrapper(*args, **kwargs):
        # Consume and clear on EVERY call, armed or not, in a finally. An arm
        # that is never consumed is an arm that reaches a later prompt, and
        # clearing only on the armed path is what let that happen.
        armed = dict(_PENDING_SHORT_EDGE)
        _PENDING_SHORT_EDGE.clear()
        pending = max(armed.values()) if armed else None
        if len(set(armed.values())) > 1:
            logger.warning(
                "[h3] two Reference Resolution nodes armed different short "
                "edges (%s); the downstream node reads ONE value for all of "
                "them, so %d is being used for every reference in this graph.",
                sorted(set(armed.values())), pending)
        if pending is None:
            return original(*args, **kwargs)
        if kwargs.get("ref_image_size", "match") == "match":
            # In `match` the stock node never reads the constant -- it scales
            # to the generation's pixel area instead. An override that
            # silently does nothing in the default configuration is worse
            # than no override, so say so rather than appear to work.
            logger.warning(
                "[h3] short_edge override of %d ignored: ref_image_size is "
                "'match', which sizes references from the video's pixel area "
                "and never reads the 2048 constant. Set it to 'max'.", pending)
            return original(*args, **kwargs)
        logger.warning(
            "[h3] EXPERIMENTAL: lifting ComfyUI's reference clamp to %d for "
            "one call. This monkeypatches a core node and pushes image "
            "references past the %d the released checkpoint was conditioned "
            "at. Results are not comparable to anything measured at the "
            "default, and nothing downstream is tested here.",
            pending, REF_IMAGE_SHORT_EDGE)
        with _rebound_short_edge(pending):
            return original(*args, **kwargs)

    setattr(wrapper, _WRAP_MARKER, True)
    return wrapper


def _install_wrapper():
    """Wrap the stock node once. Idempotent, and says so if someone else won.

    The chaining packs established the marker convention for exactly this;
    two packs each wrapping unaware of the other is the collision class this
    ecosystem keeps producing.
    """
    import comfy_extras.nodes_minimax_h3 as core

    node = core.MiniMaxH3ReferenceToVideo
    current = node.__dict__.get("execute")
    if current is None:
        # Inherited from a base or a mixin rather than defined here. Wrapping
        # `None` would install `classmethod(_make_wrapper(None))` and kill
        # every reference render in the process, including graphs that never
        # touched the experimental flag, since the wrapper is global.
        logger.warning(
            "[h3] cannot install the short-edge override: "
            "MiniMaxH3ReferenceToVideo.execute is not defined on the class. "
            "Upstream moved it; the override will not apply.")
        return
    inner = current.__func__ if isinstance(current, classmethod) else current
    if getattr(inner, _WRAP_MARKER, False):
        return
    node.execute = classmethod(_make_wrapper(inner))


def arm_short_edge_override(value, node_id=None):
    """Arm the override for the next downstream ReferenceToVideo call.

    Arming is per fit node and consumption is per downstream call, and a graph
    has one `ReferenceToVideo` for however many references. The entry is keyed
    by `node_id` so a sibling fit node with the checkbox off cannot cancel it;
    the wrapper reconciles multiple arms and warns if they disagree.

    **Known residual, and it is not closable through the public surface.** If a
    prompt arms and no `ReferenceToVideo` runs -- the node is muted, execution
    is interrupted, an unrelated branch raises -- the entry survives until the
    next call to that node, which may be in a later prompt that never asked
    for it. Closing it properly needs a prompt identity the wrapper can compare
    against, and ComfyUI does not expose `prompt_id` to nodes (see
    `provenance.py`, same finding). What is closed: the wrapper now clears on
    every call rather than only the armed path, and every fit node clears its
    own entry before arming, so the window is one prompt that contains the fit
    node, does not reach the downstream node, and is followed by a prompt that
    reaches the downstream node without the fit node.
    """
    _install_wrapper()
    key = str(node_id) if node_id is not None else "_anonymous"
    previous = _PENDING_SHORT_EDGE.get(key)
    if previous not in (None, value):
        logger.warning(
            "[h3] short-edge override for node %s was armed at %d and is now "
            "%d.", key, previous, value)
    _PENDING_SHORT_EDGE[key] = value


def disarm_short_edge_override(node_id=None):
    """Drop this node's arm, or every arm when no node is named.

    Called unconditionally at the top of each `execute`, which is what stops a
    previous prompt's value surviving the checkbox being switched off -- the
    old code only disarmed on the checkbox-off path, so the branch that logged
    "there is nothing to lift" was the branch that let 3072 through.
    """
    if node_id is None:
        _PENDING_SHORT_EDGE.clear()
    else:
        _PENDING_SHORT_EDGE.pop(str(node_id), None)
