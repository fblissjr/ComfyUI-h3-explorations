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
        # `comfy/ldm/minimax/model.py` sends. That is the model's only
        # `optimized_attention` call in SOURCE, but it is reached by **52
        # modules** -- `num_layers=50` DiT blocks plus
        # `token_refiner_num_layers=2` -- since they all share one
        # `Attention.forward`. Do not read "one call site" as "one call":
        # patching or gating that line touches all 52, and the two refiner
        # blocks see only the text span while the 50 see the full packed
        # sequence. The first version of this
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
        # A PAIR OF PROBES, because either one alone is ambiguous.
        #
        # The sparse gate falls THROUGH to the foreign patch -- us -- whenever
        # it declines: `take = gate is not None and ...` then
        # `return patched_forward(...)`. So a call reaching sage is consistent
        # with two very different worlds:
        #
        #   composed and healthy   the gate is live and declined this call
        #   composition dead       the gate never engaged, sage gets everything
        #
        # A small probe alone reports green in both. That is the same shape as
        # the counter bug this file just replaced: evidence that cannot
        # separate "working as designed" from "the mechanism is absent".
        #
        # Pinning it from both sides:
        #   small (below min_tokens)  MUST reach sage -> the fall-through works
        #   large (above min_tokens)  must NOT reach sage -> the gate is taking
        #
        # The second assertion is only sound because of the fresh thread. A
        # None normally means "cannot tell"; on a thread that has made exactly
        # one call it cannot mean anything else, so None after a large probe is
        # positive evidence that sage did not route it.
        BATCH, HEADS, HEAD_DIM = 1, 56, 128
        dt = torch.bfloat16

        # Whether a sparse patch is in this graph at all, taken from the same
        # dict its gate reads. Do NOT default a missing key to a size: an
        # absent `sol_compose` is exactly the dead-composition case above, and
        # silently substituting 4096 would size a probe for a gate that is not
        # there and call the result green.
        gate = to.get("sol_compose") if isinstance(to, dict) else None
        sparse_expected = isinstance(gate, dict)
        if sparse_expected and not isinstance(gate.get("min_tokens"), int):
            return None, ("a sparse patch published `sol_compose` without an "
                          "int `min_tokens`, so the probe cannot be sized "
                          "against its gate; registration checks above still "
                          "passed")
        min_tokens = gate["min_tokens"] if sparse_expected else 4096

        # Sized from the LARGEST probe this actually fires, which is
        # `min_tokens + 512` below -- not `2 * min_tokens`, which is what this
        # computed until 2026-08-27 and is nearly double the real peak. The two
        # probes run in sequence on threads that exit, so the peak is the larger
        # one alone.
        #
        # The overestimate did not matter while `min_tokens` was 4096. It
        # started to when the shipped Sol recipe took the node's 12288 the same
        # day: the demand went from ~1.75 GiB free to ~5.25 GiB, and this
        # assert is wired in most shipped graphs, so a second render in a
        # session with H3 resident would quietly return "skipped" instead of
        # confirming the sage/Sol composition. A check that goes silent when the
        # state is fine is the mirror of one that goes red when it is fine.
        probe_peak = 4 * BATCH * (min_tokens + 512) * HEADS * HEAD_DIM * 2
        free = torch.cuda.mem_get_info()[0]
        if free < probe_peak * 4:
            return None, (f"skipped the probe: {free / 2**20:.0f} MiB free, want "
                          f"{probe_peak * 4 / 2**20:.0f} MiB headroom for a "
                          f"{min_tokens + 512}-token probe. Registration "
                          "checks above still passed; routing was not confirmed "
                          "at call time")

        import threading

        def fire(seq):
            """One probe of `seq` tokens on a FRESH thread; the sage kernel or None.

            Fresh thread is what makes the None sound in both directions. The
            dispatch value is a `threading.local`, so a thread that has never
            made a sage call starts at None by construction -- no reset needed,
            and no chance of reading a value another prompt left behind.

            Do NOT read stream behaviour from this. PyTorch's current stream is
            thread-local too, so this thread sees the default stream, not the
            sampler's. Sound for "did it route", misleading for anything else.
            """
            q, k, v = (torch.randn(BATCH, HEADS, seq, HEAD_DIM, device="cuda",
                                   dtype=dt) for _ in range(3))
            out = {}

            def _probe():
                try:
                    with torch.inference_mode():
                        override(lambda *a, **kw: torch.zeros_like(q),
                                 q, k, v, HEADS, skip_reshape=True,
                                 transformer_options=dict(to))
                    out["name"] = get_last_dispatched_kernel()
                except Exception as exc:
                    out["error"] = exc

            # The fresh thread is for sage's thread-local dispatch value (above);
            # it must still carry ComfyUI's executing context, which is a
            # contextvar and does NOT cross a Thread on its own. Without the
            # copy, an armed `sol_observe` records this probe with no prompt
            # id -- which is exactly what the first live record showed.
            import contextvars
            worker = threading.Thread(target=contextvars.copy_context().run, args=(_probe,),
                                      name=f"sage-chain-probe-{seq}")
            worker.start()
            worker.join()
            del q, k, v
            torch.cuda.empty_cache()
            return out

        small_n = max(256, (min_tokens // 2) // 64 * 64)
        small = fire(small_n)
        if "error" in small:
            return False, f"composed attention raised on a probe call: {small['error']!r}"
        small_name = small.get("name")
        if small_name is None:
            return False, (f"a {small_n}-token probe, below the sparse gate at "
                           f"{min_tokens} so it should fall through, did not "
                           "reach sage. The fall-through to our patch is broken")
        if small_name not in KNOWN_KERNEL_NAMES:
            return True, (f"routed to {small_name!r}, not in the fork's "
                          "KNOWN_KERNEL_NAMES -- newer kernel, or a rename")

        if not sparse_expected:
            return True, (f"sage routed a {small_n}-token probe on {small_name}; "
                          "no sparse patch published `sol_compose`, so this "
                          "graph is sage-only and nothing should be taking "
                          "calls ahead of it")

        large_n = min_tokens + 512
        large = fire(large_n)
        if "error" in large:
            return False, f"composed attention raised on a probe call: {large['error']!r}"
        if large.get("name") is not None:
            return False, (f"a {large_n}-token probe, ABOVE the sparse gate at "
                           f"{min_tokens}, still reached sage on "
                           f"{large['name']}. A sparse patch published "
                           "`sol_compose` but is not taking calls it should "
                           "take, so composition is registered and dead -- the "
                           "render will look fine and be dense")
        return True, (f"sage routed a {small_n}-token probe on {small_name} and "
                      f"correctly did NOT get the {large_n}-token one, so the "
                      f"sparse gate at {min_tokens} is live and sage is taking "
                      "what it declines")

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

        # The resolved Sol window, which nothing printed before 2026-08-26.
        #
        # `start_percent`/`end_percent` are widgets, but what actually gates a
        # step is the SIGMA pair they resolve to, and that conversion happens
        # inside the Sol node at patch time. So the graph shows percentages, the
        # log showed nothing, and the thing that decides which steps run sparse
        # was invisible in both. It is also step-count dependent in effect --
        # a fixed sigma floor covers a different fraction of a 16-step run than
        # of an 8-step one -- which is how the PDD arms silently lost their
        # dense final step.
        #
        # This node is where it belongs: it already exists to report what the
        # composition ended up as rather than what any one node intended, and
        # it is the only node downstream of Sol that holds the model.
        # MiniMaxH3Preflight cannot -- it takes conditioning and samples, not a
        # model, so it can price the sequence and nothing about attention.
        gate = to.get("sol_compose") if isinstance(to, dict) else None
        if isinstance(gate, dict):
            lo, hi = gate.get("sigma_start"), gate.get("sigma_end")
            if isinstance(lo, float) and isinstance(hi, float):
                logger.info(
                    "[h3] sol window: a step runs SPARSE while sigma is in "
                    "[%.4f, %.4f], and dense outside it. Fewer steps means a "
                    "coarser schedule, so check the tail: if the last sigma is "
                    "above %.4f the final step runs sparse.", hi, lo, hi)

        return io.NodeOutput(model)
