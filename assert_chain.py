"""Fail the render if the attention chain is not composed as intended.

The problem this exists for: our node registers a replacement `forward` on
each DiT attention module *and* an `optimized_attention_override`. A sparse
attention patch applied afterwards is supposed to find that override and
chain onto it. Whether it does is negotiated between two third-party
packages through a duck-typed attribute and a delegate key that neither
formally owns, and both rewrote that seam within a minute of each other on
2026-08-06.

If either side renames something, composition silently takes a different
path. There is no error and no warning. The render succeeds, looks fine, and
is slower or numerically different than the one you meant to run -- which is
indistinguishable from success unless you go read the log.

Guarding that by hand, with a three-line log grep after every restart, only
works while someone remembers to do it. This promotes the check to a hard
gate: wire it after the last model patch and the graph refuses to run when
the contract is broken.

Scope: it asserts *our* routing contract -- that sage is installed and that
anything layered on top chained rather than overwrote. It deliberately does
not assert anything about what the other package does internally, which is
not ours to pin.

**What it proves, and what it does not.** The structural checks run at patch
time and prove *registration*: the override object exists, the forward
patches are on the keys we expect. They cannot prove the composed path is
taken when attention actually fires -- the same gap as a log line that
confirms a block list parsed rather than that an exemption fired. That
distinction has cost this project several measurements.

`exercise=True` closes it by pushing one tensor through the composed
attention and reading the sage fork's own dispatch telemetry
(`get_last_dispatched_kernel`), which names the kernel that took the call.
That is call-time evidence about *sage*, and it degrades to a warning rather
than a failure whenever the probe could not be run at all.

Until 2026-08-13 it read the SPARSE package's counters instead, which made it
a control that could not fail: it passed when that package routed, in graphs
that may not even use it, and reported "path not taken" on a sage-only graph
where sage was working perfectly.
"""
from __future__ import annotations

import logging

from comfy_api.latest import io

logger = logging.getLogger(__name__)


class SageChainAssert(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SageChainAssert",
            display_name="Assert Sage Attention Chain",
            category="model/attention/minimax",
            description=(
                "Raises if the attention chain is not composed as intended, "
                "instead of letting a silently-bypassed patch render "
                "successfully. Place after the last node that patches "
                "attention."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "require_override", default=True,
                    tooltip="Require our optimized_attention_override to still be "
                            "installed. Fails when a later patch replaced it "
                            "outright instead of chaining onto it."),
                io.Boolean.Input(
                    "require_forward_patch", default=True,
                    tooltip="Require the per-block attention forward patches to "
                            "still be present. A later patch may legitimately "
                            "own this key and cooperate -- see the note in the "
                            "failure message before turning this off."),
                io.Boolean.Input(
                    "exercise", default=True,
                    tooltip="Push one tensor through the composed attention and "
                            "assert on what actually routed, rather than only on "
                            "what is registered. This is the difference between "
                            "install-time and call-time evidence. Costs a "
                            "fraction of a second and ~176 MiB transiently."),
                io.Boolean.Input(
                    "warn_only", default=False,
                    tooltip="Log instead of raising. Defeats the point of the "
                            "node; use only while diagnosing."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @staticmethod
    def _exercise(override, to):
        """Fire one attention call through the composed path; report what ran.

        Returns (ok, detail). Structural checks above prove registration;
        this proves routing, and now also identity -- it reports which sage
        kernel took the call rather than only that something happened.

        `None` means "cannot tell" and is returned for every reason the probe
        could not be run: no CUDA, no telemetry, not enough free VRAM.
        `False` is reserved for "ran, and sage did not take it". Conflating
        those is the error this whole node exists to prevent, and the
        pre-2026-08-13 version committed it -- it read a THIRD package's
        counter, so "that package is not in this graph" arrived here as
        "the composed path was not taken".
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            return None, f"torch unavailable: {exc}"
        if not torch.cuda.is_available():
            return None, "no CUDA device; skipped"

        # WHAT THIS READS, and why it is not the sparse package's counters.
        #
        # Until 2026-08-13 this probe required a counter named
        # `sol_attn_stats` to move -- the SPARSE pack's, never sage's, and
        # never ours. Two consequences, both live for months:
        #
        #   * On a graph with no sparse patch the probe fired, sage routed it,
        #     nothing named sol_attn_stats moved, and this node reported "the
        #     composed path was not taken". Sage was fine; the instrument
        #     could not see it. That made a sage-only graph unrunnable.
        #   * When it PASSED, what it confirmed was that the sparse pack
        #     routed. It never said anything at call time about sage, which is
        #     the node this file is named for. And because that pack is
        #     imported process-wide once installed, the symbol resolved even in
        #     graphs that do not use it -- so the check could not tell "not in
        #     this graph" from "path not taken". A control that cannot fail.
        #
        # The fork exports the right primitive as public API:
        # `get_last_dispatched_kernel()` is set on every sage call and names
        # the kernel, so this asserts routing AND identity in one read.
        try:
            from sageattention import (get_last_dispatched_kernel,
                                       KNOWN_KERNEL_NAMES)
        except Exception as exc:
            return None, (f"sage dispatch telemetry unavailable ({exc}), so "
                          "routing could not be confirmed at call time; "
                          "registration checks above still passed")

        # [B, H, S, D] with skip_reshape=True, which is what
        # `comfy/ldm/minimax/model.py:172` sends. The first version of this
        # probe used [B, S, H, D] and passed no skip_reshape, which is the
        # layout the override *produces* internally rather than one it
        # accepts, so it took the 3D branch and died unpacking four values
        # into three. It had never executed, so nothing caught that.
        # SIZE THE PROBE BELOW THE SPARSE GATE, and note that the correct
        # size INVERTED when this check stopped reading the sparse package's
        # counters.
        #
        # Old check: a large probe was required, because it needed the sparse
        # kernel to fire so its counter would move. A small probe was declined
        # for being small and looked identical to a broken chain.
        #
        # New check: it asks whether SAGE took the call. On a composed graph a
        # large probe is taken by the sparse patch, which runs its own kernel
        # and never reaches sage -- so a large probe now proves the opposite of
        # what is wanted and fails on every correctly composed graph. Measured
        # 2026-08-13: at 4608 the log shows `[sol_attn] sparse (1, 4608, ...)`
        # and sage dispatch stays None. The chain was fine.
        #
        # Below the gate the sparse patch declines and the call falls through
        # to sage, which is exactly the composition claim: *sage handles what
        # the sparse patch does not*. Read the gate's own threshold when it is
        # published rather than assuming the default, so a graph that lowers
        # min_tokens does not silently push the probe back above it.
        BATCH, HEADS, HEAD_DIM = 1, 56, 128
        gate = to.get("sol_compose") if isinstance(to, dict) else None
        min_tokens = 4096
        if isinstance(gate, dict) and isinstance(gate.get("min_tokens"), int):
            min_tokens = gate["min_tokens"]
        SEQ = max(256, (min_tokens // 2) // 64 * 64)
        dt = torch.bfloat16

        # 3 x 66 MB of probe, plus the kernel's own output and workspace, at the
        # moment the model is being staged. On a card already oversubscribed by
        # H3's stack that is a bad time to be the allocation that fails, so give
        # up the call-time evidence rather than the render.
        need = 4 * BATCH * SEQ * HEADS * HEAD_DIM * 2
        free = torch.cuda.mem_get_info()[0]
        if free < need * 4:
            return None, (f"skipped the probe: {free / 2**20:.0f} MiB free, want "
                          f"{need * 4 / 2**20:.0f} MiB headroom for a "
                          f"{need / 2**20:.0f} MiB probe. Registration checks "
                          f"above still passed; routing was not confirmed at "
                          f"call time")

        q, k, v = (torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device="cuda", dtype=dt)
                   for _ in range(3))

        # Fire on a FRESH THREAD, which is what makes this sound without any
        # private reset. The dispatch value lives on a `threading.local`, so a
        # thread that has never made a sage call returns None BY CONSTRUCTION.
        # Read on the execution thread instead and a value left by an earlier
        # prompt is indistinguishable from this probe's -- a false negative on
        # exactly the graphs that route consistently, which is the defect
        # above wearing a better API.
        #
        # The module patch is per-module and the CUDA context is process-wide,
        # so the composed forward is still traversed off-thread.
        #
        # Do NOT read stream behaviour from this. PyTorch's current stream is
        # itself thread-local, so this thread sees the default stream and not
        # the sampler's. Sound for "did it route", misleading for anything else.
        import threading

        outcome = {}

        def _probe():
            try:
                with torch.inference_mode():
                    override(lambda *a, **kw: torch.zeros_like(q),
                             q, k, v, HEADS, skip_reshape=True,
                             transformer_options=dict(to))
                outcome["name"] = get_last_dispatched_kernel()
            except Exception as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=_probe, name="sage-chain-probe")
        worker.start()
        worker.join()
        del q, k, v
        torch.cuda.empty_cache()

        if "error" in outcome:
            return False, f"composed attention raised on a probe call: {outcome['error']!r}"
        name = outcome.get("name")
        if name is None:
            return False, (f"a probe of {SEQ} tokens -- deliberately below the "
                           f"sparse gate at {min_tokens} so it should fall "
                           "through -- did not reach sage: the fork's dispatch "
                           "telemetry is still None on a thread that made "
                           "exactly this one call")
        if name not in KNOWN_KERNEL_NAMES:
            return True, (f"routed to {name!r}, which is not in the fork's "
                          "KNOWN_KERNEL_NAMES -- newer kernel, or a rename")
        return True, f"sage routed a {SEQ}-token probe on {name}"

    @classmethod
    def execute(cls, model, require_override, require_forward_patch,
                exercise, warn_only):
        problems = []

        to = model.model_options.get("transformer_options", {})
        override = to.get("optimized_attention_override")
        if require_override and override is None:
            problems.append(
                "no optimized_attention_override is installed. Either the sage "
                "node is not in this graph, or a later patch replaced the "
                "override instead of chaining onto it. A sparse-attention patch "
                "that declines a call would then fall through to ComfyUI's "
                "default attention rather than sage.")

        patches = getattr(model, "object_patches", {}) or {}
        attn_forwards = [k for k in patches
                         if k.startswith("diffusion_model.blocks.")
                         and k.endswith(".attn.forward")]
        if require_forward_patch and not attn_forwards:
            problems.append(
                "no per-block attention forward patches are installed on "
                "diffusion_model.blocks.*.attn.forward. If a cooperating "
                "low-VRAM or sparse patch has legitimately taken ownership of "
                "that key, this check is too strict for your graph and "
                "require_forward_patch can be turned off -- but confirm from "
                "the log that sage still runs before you do.")

        if exercise and override is not None:
            ok, detail = cls._exercise(override, to)
            if ok is None:
                logger.warning("[h3] chain assert: %s", detail)
            elif ok:
                logger.info("[h3] chain assert, call-time: %s", detail)
            else:
                problems.append(detail)

        if problems:
            detail = "\n".join(f"  - {p}" for p in problems)
            msg = (f"SageChainAssert: attention chain is not composed as "
                   f"intended.\n{detail}\n"
                   f"  Node order matters: the sage node must come before any "
                   f"sparse-attention patch. Reversed, the sparse patch "
                   f"overwrites it and the render silently uses sage only.")
            if warn_only:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        else:
            logger.info(
                "[h3] chain assert ok: override installed, "
                "%d attention forward patch(es) present", len(attn_forwards))

        return io.NodeOutput(model)
