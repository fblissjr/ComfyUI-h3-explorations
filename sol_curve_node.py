"""Select a token ordering Sol-Attn's own node does not offer.

kijai's `SolAttnMiniMax` exposes `morton_curve` as a combo of `3d` and
`2d_frame`, so a third ordering cannot be requested through it even though the
machinery underneath takes an arbitrary string. This node supplies one, without
editing his file and without rebuilding a kernel.

**Place it AFTER SolAttnMiniMax.** It overwrites
`transformer_options["sol_morton_curve"]`, which that node sets, so it has to
run second or its value is the one that gets replaced. It does nothing at all
unless Sol-Attn's own `morton` toggle is on: the reorder is gated on
`sol_morton`, which only that node publishes.

## Why a node and not a fork

`_perm_for` inside the vendored node resolves `morton_perm` as a plain module
global, so rebinding that one name is enough for a new curve to reach the
permutation cache, the start-offset rotation and the block hooks unchanged.
`sol_curves.install()` does the rebinding and resolves the live module by
identity rather than by name, because a running ComfyUI can hold two module
objects for one file. See `vendor/README.md` for the preference order this
follows, and CLAUDE.md for the two-module trap.

## What the curves are worth, measured

On captured H3 activations (`bench/analyze_capture.py`, blocks 24 and 49 of a
dense 124-frame 1344x768 render), against the shipped `2d_frame`:

    centroid fidelity   3d best, hilbert second, both above 2d_frame
    mass concentration  hilbert best, 3d second, both above 2d_frame

The two metrics disagree about which of `3d` and `hilbert` leads, and neither
has been rendered. **This node exists so that comparison can be made, not
because an answer is known.** `3d` needs no node; it is already selectable on
Sol-Attn's own combo.

## What this node silently changes besides the ordering

**It moves the operating point Sol-Attn was configured at, and no A/B run here
has accounted for that.** The router's threshold is
`tau * sqrt(sum_d c_d^2 * kcvar_d * log2s^2)`, and `kcvar` is the variance
across the block centroids -- which the permutation defines. So a curve change
moves the threshold, and the fraction of blocks routed exact at a fixed `tau`
moves with it. Direction is not derivable from the formula: the scores are
computed against the same pooled centroids, so numerator and denominator both
respond to block coherence. It has to be measured per curve and per depth.

Two consequences for anyone wiring this node:

  - **A fixed-`tau` comparison of two curves varies two things.** Whatever it
    shows is a mix of "different blocks" and "different sparsity", and the split
    is unknown until `bench/analyze_routing.py` exists.
  - **Compensation is expressible but not measured.** `SolAttnMiniMax` takes a
    `tau_profile`, which is keyed per transformer block, so a depth-dependent
    correction fits. It cannot express a sigma-dependent one, and whether the
    divergence is sigma-dependent is unmeasured -- the only capture is step 1
    of 6. Do not build the compensation table before that is known.

See `docs/morton.md`, "Holding `tau` fixed does not hold sparsity fixed".
"""

from __future__ import annotations

import logging
from pathlib import Path

from comfy_api.latest import io

try:                                    # loaded as a package by ComfyUI
    from . import sol_curves
except ImportError:                      # loaded as a bare module by a script
    import sol_curves

_VENDOR = Path(__file__).resolve().parent / "vendor" / "sol_attn_minimax.py"

logger = logging.getLogger(__name__)


class MiniMaxH3SolAttnCurve(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolAttnCurve",
            display_name="MiniMax H3 Sol-Attn Curve",
            is_experimental=True,
            category="model/patch/minimax",
            description=(
                "Override Sol-Attn's token ordering with a curve its own node "
                "does not list. Place AFTER SolAttnMiniMax, and turn that "
                "node's `morton` on -- this is inert otherwise. `hilbert` "
                "almost never jumps between consecutive points (6 steps of "
                "1007 at latent 24x42) where Z-order jumps on half of them, "
                "so its 64-token blocks stay nearly connected on grids whose "
                "latent dims are not multiples of 8. "
                "CHANGING THE CURVE CHANGES THE OPERATING POINT: block "
                "membership feeds the router's threshold, so the fraction of "
                "blocks routed exact moves at a fixed tau. A curve A/B at "
                "fixed tau is not a controlled comparison. See docs/morton.md."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "curve", options=["hilbert"], default="hilbert",
                    tooltip="hilbert: 2D Hilbert within each latent frame, "
                            "frames left in original order. The direct "
                            "replacement for 2d_frame. For 3d, use Sol-Attn's "
                            "own morton_curve instead; this node is only for "
                            "orderings that one cannot express.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, curve="hilbert") -> io.NodeOutput:
        # Default mirrors the schema's. ComfyUI does not inject a schema
        # default for an input an API prompt omits, so the two are independent
        # and a split means the UI and API paths see different values.
        # check_schema_defaults.py enforces this across every node here.
        patched = sol_curves.install(_VENDOR)
        if not patched:
            # Zero is a failure, not a no-op: the render would silently run
            # whatever curve Sol-Attn's own combo was left on, and look fine.
            raise RuntimeError(
                "Sol-Attn's node is not loaded, so its token ordering cannot "
                "be overridden. Install ComfyUI-SolAttn-cuda, or remove this "
                f"node from the graph. Looked for a live module whose file is "
                f"{_VENDOR.name}."
            )

        m = model.clone()
        to = m.model_options["transformer_options"] = \
            m.model_options.get("transformer_options", {}).copy()
        if not to.get("sol_morton"):
            logger.warning(
                "[sol_curves] curve set to %r but Sol-Attn's `morton` is off, "
                "so no reordering will happen. Turn morton on in "
                "SolAttnMiniMax, and put that node before this one.", curve)
        to["sol_morton_curve"] = curve
        logger.info("[sol_curves] token ordering overridden to %r", curve)
        return io.NodeOutput(m)
