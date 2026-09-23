#!/usr/bin/env python3
"""Check every shipped graph carries the attention configuration `h3_config` declares.

## What was uncontrolled, and what this replaces

`docs/checks.md`'s standing audit carried this row: *"Sol-Attn is on by default
in every shipped video workflow"* — enforced by **one graph, by accident**.
`check_bench_matches_shipped.py` compared the bench harness against
`h3_probe_sol_on_api.json` (retired 2026-09-14), so stripping Sol from that one
file turned it red while stripping it from any other went unnoticed. That was measured by
deliberate violation on 2026-08-17, not inferred.

It does **not** replace `check_bench_matches_shipped.py`, and the distinction is
the whole reason both exist. That check's subject is the **bench harness** —
whether `bench_e2e_h3.py`'s hardcoded arms still describe what the graphs ship.
This check's subject is the **graphs**, against `h3_config`. Neither implies the
other: on 2026-08-13 the config and the graphs moved together while the bench
stayed behind, which is the failure that check was written for; the failure
*this* one is written for is a graph drifting from the config the generator
reads.

## Why values and not just presence

Presence was the stated rule, but a graph with Sol wired at the wrong `tau` or
sage at the wrong mode is a silently different experiment, and on 2026-08-18 the
sage mode changed in every graph at once. Nothing verified that the rewrite
took everywhere; the verification was an ad-hoc script run by hand, which is the
shape this repo keeps finding after the fact.

## Reachability, not presence

`docs/evidence.md` records a capture provenance field that asked whether a Sol
node **existed** when the question was whether it **ran**, and reported the
situation backwards. ComfyUI seeds execution from output nodes and walks
backwards, so a node nothing consumes never executes. An ACTIVE
`SolAttnMiniMax` whose MODEL output feeds nothing renders dense and looks
entirely normal — measured on 2026-08-18 at a real cost, in
`bench/results/2026-08-18_attention_defaults.json`.

The shipped graphs are API form, where a node that is off is simply absent, so
omission is how a graph disables Sol **deliberately**, and is not a defect. Only
a present-but-unreachable node is.

## Why the exemption list cannot rot

Several kinds of graph legitimately ship without Sol -- `SOL_EXEMPT_STEMS`
is the list and counting them here would be a second copy that goes stale the
next time one is added, which is what happened when the PDD arms landed. A
hand-written list of
exempt filenames is the shape that goes stale silently, which is the objection
`docs/checks.md` raised against this check existing at all. So:

  * the single-frame class is taken from `h3_config.GRAPH_DIRS`, not from
    filenames — it is whatever the generator routes to a subdirectory. That
    class is **empty while the single-frame lane is parked** (2026-08-27), and
    the run says so on its own line rather than leaving it to be inferred from
    an exemption nothing printed: correctly-empty and never-computed have to
    look different from a green run;
  * every exemption is asserted **necessary**. If an exempt graph turns out to
    have live Sol, that is a failure, not a pass. So an exemption that stops
    being true goes red instead of quietly covering a graph nobody checks.

## The dense floor, since 2026-09-15

Sol chains onto whatever attention override is already installed, so the
kernel under it is part of the configuration too. Until 2026-09-15 that was
sage on every graph; since then the default is ComfyUI core's
`ModelAttentionBackend` at `h3_config.DENSE_BACKEND_NODE`. Each graph's floor
is read from its live nodes (sage, backend, or neither for stock attention),
the backend's values are graded, and a graph off the default must be named in
`FLOOR_STEMS` with a mechanism, held to the same necessity rule as the
exemptions: a graph that keeps sage without saying why fails, and so does a
declaration the graph no longer needs.

`workflows/bench/*_stamped_api.json`, the dense baselines, are outside
`graph_paths()` entirely and so outside this check's scope. That is stated
rather than exempted, because an exemption implies coverage.

## Shown red

2026-08-18, four ways, each reverted: an `SolAttnMiniMax` re-pointed so its
output feeds nothing (`no_orphans`); Sol's `tau` edited in one graph
(`sol_values`); a sage `mode` edited in one graph (`sage_values`); and Sol
wired into a graph on the exemption list (`exemptions_necessary`).

    python bench/check_attention_defaults.py

No CUDA, no server, no model. Reads graphs and `h3_config` only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "workflows"))

import h3_config  # noqa: E402

WORKFLOWS = _REPO / "workflows"
# Ours since 2026-08-30, replacing the vendored `SolAttnMiniMax`. One name,
# because this check grades what the SHIPPED graphs carry and the generator
# emits exactly one node id -- a graph on the old id is a stale regeneration,
# which is a finding for this check rather than a second case to accept.
SOL = "MiniMaxH3SolAttn"
SAGE = "MiniMaxH3SageAttention"
# ComfyUI core's node, the dense kernel under Sol since 2026-09-15; graded
# against h3_config.DENSE_BACKEND_NODE.
BACKEND = "ModelAttentionBackend"

OUTPUT_TYPES = {"VHS_VideoCombine", "SaveImage", "PreviewImage", "SaveAudio",
                "SaveAnimatedWEBP", "SaveWEBM", "SaveVideo", "PreviewAny",
                # this pack's own file writer, an output node by declaration
                "MiniMaxH3AudioFreezeSong"}

#: {graph stem: (node fields, reason)}. A per-graph deviation that is the point
#: of the graph. `node fields` is a tuple of widget names on the sage or Sol
#: node (the names do not collide); each named field is skipped when that
#: graph is graded against h3_config, and asserted to actually deviate, so a
#: declaration cannot outlive the graph it describes. A reason naming a file
#: rather than a mechanism is not a reason.
DEVIATIONS = {
    "h3_probe_head_chunks": (("head_chunks",),
                             "this graph exists to run the 1-vs-N head-chunking "
                             "A/B that h3_config asks for; the deviating value "
                             "IS its subject"),
    "h3_candidate_t2v_sol_only": (("start_percent",),
                                  "candidate graph (2026-09-05): Sol from the first "
                                  "step, no sage; the widened window IS the arm, "
                                  "blinded as sol_nosage_2026-09-04"),
    "h3_candidate_t2v_sol_allrows": (("sink_conditioning",),
                                     "candidate graph (2026-09-05): every conditioning "
                                     "query row dense; the sink mode IS the arm, "
                                     "measured on the 2026-09-04 subway probe pair"),
    "h3_candidate_t2v_pdd8_sol_narrow": (("start_percent", "end_percent"),
                                         "candidate graph (2026-09-05): Sol on two of "
                                         "the eight PDD steps; the window IS the arm, "
                                         "blinded as pdd_ladder_2026-09-04"),
    "h3_probe_t2v_policy": (("mode",),
                            "block-49 policy graph (2026-09-15, docs/h3_quant_policy.md "
                            "Tier 0/1), kept on the sage chain while its pair is scored: "
                            "sage in 'fp8++ balanced' plus the balance node and exact "
                            "tail blocks; the mode IS the arm"),
    "h3_probe_t2v_ck": (("qk_balance",),
                        "community-chain control (2026-09-15): kitchen int8 dense + Sol "
                        "with qk_balance OFF, the chain as most people run it; the "
                        "switch IS the arm"),
    "h3_probe_t2v_rotate": (("rotate",),
                            "Tier 2 witness (2026-09-15): the default chain with Sol's "
                            "Hadamard rotation on; the switch IS the arm"),
    "h3_probe_t2v_ck_dense_tail": (("dense_blocks",),
                                   "community-chain probe (2026-09-15): blocks 45/48/49 handed to "
                                   "the kitchen dense kernel; the list IS the arm"),
    "h3_probe_t2v_exact_tail": (("qk_balance",),
                                "the scored ceiling arm of the sage chain (2026-09-15, "
                                "bench/results/2026-09-15_block49_*): kept as it rendered, "
                                "so Sol's qk_balance stays off; the switch predates the "
                                "recipe's"),
    "h3_probe_t2v_sage_rotate": (("mode", "rotate"),
                                 "the sage chain with every lever plus Sol rotate (2026-09-15 "
                                 "night); the mode and the switch ARE the arm"),
    "h3_probe_t2v_levers": (("mode",),
                            "block-49 Tier 1 witness (2026-09-15, docs/h3_quant_policy.md), "
                            "kept on the sage chain while its pair is scored: every free "
                            "lever on and no exact blocks; the mode IS the arm"),
}

#: Graphs that legitimately ship without live Sol, by MECHANISM. The
#: single-frame class is not listed here -- it is derived from GRAPH_DIRS below.
SOL_EXEMPT_STEMS = {
    "h3_probe_t2v_dense":
        "the fully dense control (2026-09-15): no Sol, no sage, no dense node, "
        "ComfyUI's own attention on every step; the ceiling for the whole chain "
        "and the arm that separates a kernel flaw from a take",
    "h3_probe_vsa":
        "VSA and Sol-Attn are mutually exclusive, not merely redundant: VSA "
        "replaces the DiT block forward on the 50 main blocks and Sol-Attn "
        "overrides attention on the same 50, so a Sol node here would be "
        "SILENTLY INERT rather than additive. The generator refuses the pair "
        "outright (`build_workflows.py`, the vsa branch). sage is still wired "
        "and is not decoration -- it takes the 2 token-refiner blocks, which "
        "carry no gate and are not VSA's business",
    "h3_probe_vsa_dense":
        "the control for the arm above, and it must run the VSA CHECKPOINT "
        "with no sparse attention at all. Sol here would make it a "
        "Sol-against-VSA comparison instead of the sparse-against-dense one it "
        "exists to be, and its whole job is to show the checkpoint is a "
        "working H3 model independently of the attention regime",
    "h3_probe_capture_ref3":
        "activation capture: h3_capture.py records the attention inputs a dense "
        "baseline is measured from, and Sol gives sage only the steps outside "
        "its sigma window, so a capture taken through Sol is a different "
        "trajectory than the analysis assumes",
    "h3_probe_capture_ref3_fl2va":
        "the same capture on the fl2va checkpoint with no LoRA, the control "
        "for whether block 49's input structure is a property of the released "
        "weights rather than of ref2va; Sol off for the same reason as its twin",
    "h3_probe_t2v_pdd8_baked_sage":
        "the baked arm of the merged-versus-baked PDD8 pair "
        "(bench/pdd_bake_arms.json): the twin of h3_probe_t2v_pdd8_sage with "
        "the checkpoint and sidecar swapped for the bake, sage on every step "
        "and Sol absent so the pair differs in the weights alone. Its twin "
        "with Sol as shipped is h3_candidate_t2v_pdd8_baked, not exempt",
    "h3_probe_t2v_turbo_lx12_sage":
        "the turbo rung's lightx2v arm (bench/turbo_rung_arms.json): the v1.2 "
        "768p 4-step file at the vendor's count and strength under sage "
        "alone, Sol absent, the same footing as h3_probe_t2v_pdd8_sage",
    "h3_probe_t2v_pdd8_sage":
        "the sage-alone rung of the PDD ladder (bench/pdd_ladder_arms.json): "
        "PDD8 with sage on every step and Sol absent, so that the shipped "
        "PDD8 graph's loss to the true baseline in the 2026-09-03 ladder can "
        "be attributed. Its twin with Sol is h3_text_to_video_pdd; its twin "
        "with neither kernel is h3_probe_t2v_pdd8_dense, exempt by mechanism "
        "below rather than by this list",
    "h3_probe_turbo_768p_sla_dense":
        "comparative arm: the Turbo-SLA LoRA under sage alone, the repo's "
        "dense-baseline convention, one of three attention regimes the SLA "
        "probe set spans (Sol, router, dense)",
}

#: {graph stem: (floor, reason)} for graphs whose dense attention kernel is NOT
#: `h3_config.DENSE_BACKEND_NODE` (core's ModelAttentionBackend on kitchen
#: int8), the default chain since 2026-09-15. `floor` is what the graph must
#: carry: "sage" (a live MiniMaxH3SageAttention) or "stock" (neither node, so
#: ComfyUI's own attention). Held to the same two rules as the lists above: a
#: graph off the default floor that is not listed fails, and a listed graph
#: that is back on the default fails. The PDD reference-replication arms are
#: derived, not listed, by the same mechanism as their Sol exemption.
FLOOR_STEMS = {
    # Rungs and controls that were rendered as sage alone, so their pair holds.
    "h3_probe_t2v_turbo_lx12_sage":
        ("sage", "the turbo rung's lightx2v arm, sage alone by construction "
                 "(bench/turbo_rung_arms.json)"),
    "h3_probe_t2v_pdd8_sage":
        ("sage", "the sage-alone rung of the PDD ladder (bench/pdd_ladder_arms.json)"),
    "h3_probe_t2v_pdd8_baked_sage":
        ("sage", "the baked twin of h3_probe_t2v_pdd8_sage (bench/pdd_bake_arms.json)"),
    "h3_probe_turbo_768p_sla_dense":
        ("sage", "the SLA LoRA under sage alone, the comparative arm "
                 "SOL_EXEMPT_STEMS describes"),
    # Graphs where the sage node is part of the mechanism.
    "h3_probe_capture_ref3":
        ("sage", "h3_capture.py records from inside the sage forward, so a "
                 "capture target needs the sage node"),
    "h3_probe_capture_ref3_fl2va":
        ("sage", "the capture twin on fl2va; the same reason"),
    "h3_probe_vsa":
        ("sage", "sage takes the two token-refiner blocks VSA does not replace, "
                 "and the generator refuses the kitchen backend on a VSA arm"),
    "h3_probe_vsa_dense":
        ("sage", "the VSA arm's control, on the same floor as the arm"),
    "h3_probe_head_chunks":
        ("sage", "head_chunks is an input of MiniMaxH3SageAttention; the arm "
                 "means nothing without the node"),
    # Block-49 arms on the sage chain, scored or still being scored there.
    "h3_probe_t2v_sage_rotate":
        ("sage", "the sage chain with every lever plus Sol rotate, against the "
                 "kitchen chain (2026-09-15 night)"),
    "h3_probe_t2v_levers":
        ("sage", "block-49 Tier 1 witness on the sage chain; its pair with "
                 "h3_probe_t2v_policy is still being scored"),
    "h3_probe_t2v_policy":
        ("sage", "block-49 Tier 0/1 policy graph on the sage chain; the same pair"),
    "h3_probe_t2v_exact_tail":
        ("sage", "the scored ceiling arm of the sage chain, kept as it rendered"),
    # Sol over ComfyUI's own attention.
    "h3_probe_t2v_dense":
        ("stock", "the fully dense control (2026-09-15): stock attention everywhere, "
                  "by construction"),
    "h3_probe_t2v_sol_nosage":
        ("stock", "the just-Sol arm of 2026-09-04 (bench/sol_nosage_arms.json): "
                  "Sol over stock attention, which is not the kitchen chain"),
    "h3_candidate_t2v_sol_only":
        ("stock", "the just-Sol candidate blinded as sol_nosage_2026-09-04: "
                  "stock attention on the step Sol leaves"),
}


def load(path):
    return json.loads(path.read_text())


def reachable(g):
    """Node ids that feed an output node."""
    nodes = {i: n for i, n in g.items() if isinstance(n, dict)}
    edges = {}
    for i, n in nodes.items():
        for val in (n.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2:
                edges.setdefault(i, set()).add(str(val[0]))
    seeds = [i for i, n in nodes.items() if n.get("class_type") in OUTPUT_TYPES]
    seen, stack = set(), list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    return seen


def attn_nodes(g, want):
    """[(node_id, values_by_name, state)] for every `want` node in `g`.

    state is `live` or `orphaned` (present and unreachable -- the defect). A
    node that is off is absent from an API graph, so it never appears here.
    """
    live = reachable(g)
    out = []
    for i, n in g.items():
        if not isinstance(n, dict) or n.get("class_type") != want:
            continue
        vals = {k: v for k, v in (n.get("inputs") or {}).items()
                if not isinstance(v, list)}
        if want == SOL:
            # The API form keys the selected option's inputs under the combo
            # (`selection.tau`). Strip the prefix so the values are graded in
            # h3_config's vocabulary -- left dotted, `tau` is simply absent
            # from `vals`, every `if k in vals` comparison below skips it, and
            # the check goes green having graded nothing about the knob it
            # exists for.
            vals = {k.split(".", 1)[1] if k.startswith("selection.") else k: v
                    for k, v in vals.items()}
        out.append((i, vals, "live" if i in live else "orphaned"))
    return out


def single_frame_dirs():
    """The image class, from the generator's own routing rather than filenames.

    Empty while the single-frame lane is parked, which is the CORRECT answer
    and not a broken derivation -- `GRAPH_DIRS` is `("",)`, so no graph sits in
    a subdirectory and nothing is exempted on this ground. `main` prints the
    size either way; an exemption class that silently covers nothing is the
    shape this file's header argues against.
    """
    return {d for d in h3_config.GRAPH_DIRS if d}


def wires_dense_kernel(graph) -> bool:
    """Whether this graph has a dense attention node at all, sage or backend.

    The discriminator between the two PDD classes: a reference-replication arm
    runs stock attention and carries no attention node; a canonical arm runs
    the repo default and carries a dense node and Sol. It read the sage node
    alone until 2026-09-15, when the default floor moved to the backend node
    and every canonical PDD arm would have read as a reference arm.
    """
    return any(isinstance(n, dict) and n.get("class_type") in (SAGE, BACKEND)
               for n in graph.values())


def _steps_of(graph):
    """The graph's step count, or None.

    Delegates to `h3_config.graph_schedule` rather than reading
    `BasicScheduler` itself. A PDD graph has not carried one since 0.83.0 --
    the PDD node emits the schedule -- and the local reader this replaced
    returned None on every one of them. That read here as "no step count", took
    `SOL_END_PERCENT_BY_STEPS` to its 0.9 default, and reported eleven
    correctly-wired graphs as wrong.
    """
    return h3_config.graph_schedule(graph)[0]


def loads_pdd(graph) -> bool:
    """Whether this graph loads a Parallel Decoding Distillation LoRA.

    A MECHANISM, not a list of filenames. The PDD arms replicate the vendor's
    reference path, which runs Diffusers' stock SDPA -- so they wire neither
    sage nor Sol, and they are a class rather than four graphs. Listing their
    stems is the shape this file's own header argues against, and it went stale
    within hours: the four ref2va arms were listed on 2026-08-26 and the two
    fl2va ones added the same day turned the check red.

    Decided by the loader class, so a fifth PDD arm is exempt the moment it
    exists and a graph that stops loading one stops being exempt. The
    necessity assertion still applies -- an exempt graph with live Sol is a
    failure, not a pass.
    """
    return any(isinstance(n, dict) and n.get("class_type") == "MiniMaxH3PDDLoRA"
               for n in graph.values())


def main() -> int:
    # API form only: this reads `class_type`/`inputs`, and a graph in another
    # form would read as one with no Sol at all.
    paths = h3_config.graph_paths(WORKFLOWS, "*_api.json")
    img_dirs = single_frame_dirs()
    problems, checked = [], {"sol": 0, "sage": 0, "backend": 0, "graphs": 0,
                             "single_frame": 0}
    exempt_seen = {k: False for k in SOL_EXEMPT_STEMS}
    dev_seen = {k: False for k in DEVIATIONS}
    floor_seen = {k: False for k in FLOOR_STEMS}
    floors = {"ck": 0, "sage": 0, "stock": 0}

    for p in paths:
        g = load(p)
        stem = p.stem[:-4] if p.stem.endswith("_api") else p.stem
        in_image = p.parent.name in img_dirs
        exempt_reason = SOL_EXEMPT_STEMS.get(stem)
        pdd_reference = loads_pdd(g) and not wires_dense_kernel(g)
        if exempt_reason is None and pdd_reference:
            # Narrowed 2026-08-26, hours after it was written. It exempted
            # EVERY PDD graph, which was right while they all ran dense and
            # wrong the moment the canonical arms took the repo default -- a
            # PDD arm with sage is an ordinary graph and should carry Sol like
            # any other. The mechanism is the reference-replication config
            # (neither sage nor Sol, which is what the vendor runs), not the
            # loader class.
            exempt_reason = (
                "PDD reference-replication arm: wires neither sage nor Sol, "
                "which is what the vendor's own Diffusers pipeline runs. "
                "Derived from the graph's attention config rather than listed, "
                "so a new one is covered the moment it exists")
        if exempt_reason:
            exempt_seen[stem] = True
        checked["graphs"] += 1
        checked["single_frame"] += int(in_image)

        sol = attn_nodes(g, SOL)
        sage = attn_nodes(g, SAGE)
        backend = attn_nodes(g, BACKEND)
        checked["sol"] += len(sol)
        checked["sage"] += len(sage)
        checked["backend"] += len(backend)

        # --- dense_floor ----------------------------------------------------
        # Which dense kernel the graph runs under Sol, read from its live
        # nodes, against FLOOR_STEMS or the default. Since 2026-09-15 the
        # default is the backend node; a graph that quietly kept sage, or lost
        # its dense node and fell to stock attention, is a different
        # experiment wearing the shipped name.
        live_sage = [n for n in sage if n[2] == "live"]
        live_backend = [n for n in backend if n[2] == "live"]
        floor = ("both" if live_sage and live_backend else "sage" if live_sage
                 else "ck" if live_backend else "stock")
        declared = FLOOR_STEMS.get(stem)
        if declared is not None:
            floor_seen[stem] = True
        if floor == "both":
            problems.append(
                f"{p.relative_to(_REPO)}: carries a live {SAGE} AND a live "
                f"{BACKEND}; each installs the attention override, so one of "
                f"them silently decides the kernel. Wire one.")
        elif declared is not None and floor != declared[0]:
            problems.append(
                f"{p.relative_to(_REPO)}: FLOOR_STEMS declares {declared[0]!r} "
                f"but the graph runs {floor!r}. The declaration is stale -- "
                f"remove it or restore the arm.")
        elif (declared is None and floor != "ck" and not in_image
              and not (pdd_reference and floor == "stock")):
            problems.append(
                f"{p.relative_to(_REPO)}: dense floor is {floor!r}, the default "
                f"is {BACKEND} at h3_config.DENSE_BACKEND_NODE. If this graph "
                f"is a new exception, add it to FLOOR_STEMS with a mechanism.")
        if floor in floors:
            floors[floor] += 1
        for nid, vals, _state in live_backend:
            for k, want in h3_config.DENSE_BACKEND_NODE.items():
                if vals.get(k) != want:
                    problems.append(
                        f"{p.relative_to(_REPO)}: node {nid} {BACKEND}.{k} is "
                        f"{vals.get(k)!r}, h3_config.DENSE_BACKEND_NODE says {want!r}")

        # --- no_orphans: applies to EVERY graph, exempt or not -------------
        for nid, _vals, state in sol + sage + backend:
            if state == "orphaned":
                problems.append(
                    f"{p.relative_to(_REPO)}: node {nid} is present but its MODEL "
                    f"output nothing consumes. ComfyUI walks backwards from the "
                    f"output nodes, so it never executes. To disable it on "
                    f"purpose, leave it out of the graph.")

        live_sol = [n for n in sol if n[2] == "live"]

        # --- sol_reachable / exemptions_necessary --------------------------
        if in_image or exempt_reason:
            if live_sol:
                why = "single-frame (from GRAPH_DIRS)" if in_image else exempt_reason
                problems.append(
                    f"{p.relative_to(_REPO)}: exempted from the Sol-on rule "
                    f"({why}) but has LIVE Sol. The exemption is stale -- remove "
                    f"it or the graph, do not leave both.")
        elif not live_sol:
            problems.append(
                f"{p.relative_to(_REPO)}: no reachable {SOL}. CLAUDE.md: Sol-Attn "
                f"is on by default in every shipped video workflow. If this graph "
                f"is a new exception, add it to SOL_EXEMPT_STEMS with a mechanism.")

        # --- sol_values ----------------------------------------------------
        # Two things make the expected config vary per graph, and BOTH are
        # resolved by `h3_config.sol_for_graph` rather than restated here.
        # That is deliberate: this block used to re-derive the `end_percent`
        # lookup, which is a second copy of the generator's rule, and a check
        # that computes its own expectation grades itself (CLAUDE.md, prefer
        # a control the check compares against).
        #
        #   step count   the window used to move with the count so the last
        #                step stayed dense (h3_config.SOL_END_PERCENT_BY_STEPS).
        #                Empty since 2026-09-11, so every count expects
        #                SOL_RECOMMENDED_CUDA's 1.0; the lookup stays in the
        #                resolver.
        #   PDD          a distilled arm takes h3_config.SOL_PDD_CUDA whole,
        #                owner decision 2026-08-29, at every step count.
        #                Decided by `loads_pdd`, the same MECHANISM the
        #                exemption above uses, so a new PDD arm is graded on
        #                the right recipe the moment it exists.
        graph_steps = _steps_of(g)
        graph_pdd = loads_pdd(g)
        expected = h3_config.sol_for_graph(graph_pdd, graph_steps)
        source = "SOL_PDD_CUDA" if graph_pdd else "SOL_RECOMMENDED_CUDA"
        dev_fields, _dev_why = DEVIATIONS.get(stem, ((), None))
        deviated = set()
        for nid, vals, state in sol:
            if state != "live":
                continue
            for k, want in expected.items():
                if k in dev_fields:
                    if k in vals and vals[k] != want:
                        deviated.add(k)
                    continue
                if k in vals and vals[k] != want:
                    problems.append(
                        f"{p.relative_to(_REPO)}: node {nid} {SOL}.{k} is "
                        f"{vals[k]!r}, h3_config.{source} says {want!r}"
                        + (f" at {graph_steps} steps" if k == "end_percent"
                           and not graph_pdd else ""))

        # --- sage_values ---------------------------------------------------
        for nid, vals, state in sage:
            if state != "live":
                continue
            for k, want in h3_config.SAGE_NODE.items():
                if k in dev_fields:
                    if k in vals and vals[k] != want:
                        deviated.add(k)
                    continue
                if k in vals and vals[k] != want:
                    problems.append(
                        f"{p.relative_to(_REPO)}: node {nid} {SAGE}.{k} is "
                        f"{vals[k]!r}, h3_config.SAGE_NODE says {want!r}")

        # A declared deviation must deviate, or the declaration has outlived
        # the graph it described and would hide a real drift on that field.
        if stem in DEVIATIONS:
            dev_seen[stem] = True
            stale = sorted(set(dev_fields) - deviated)
            if stale:
                problems.append(
                    f"{p.relative_to(_REPO)}: DEVIATIONS declares {stale} for this "
                    f"graph but the widget carries the recipe's own value; remove "
                    f"the field from the declaration or restore the arm")

    # A declared exemption for a graph that does not exist is also rot.
    for stem, seen in exempt_seen.items():
        if not seen:
            problems.append(
                f"SOL_EXEMPT_STEMS names {stem!r}, which matches no graph under "
                f"graph_paths(). Remove the entry.")
    for stem, seen in dev_seen.items():
        if not seen:
            problems.append(
                f"DEVIATIONS names {stem!r}, which matches no graph under "
                f"graph_paths(). Remove the entry.")
    for stem, seen in floor_seen.items():
        if not seen:
            problems.append(
                f"FLOOR_STEMS names {stem!r}, which matches no graph under "
                f"graph_paths(). Remove the entry.")

    print(f"  {checked['graphs']} graphs, {checked['sol']} {SOL}, "
          f"{checked['backend']} {BACKEND} and {checked['sage']} {SAGE} node(s)")
    print(f"  dense floor: {floors['ck']} on {BACKEND} "
          f"{h3_config.DENSE_BACKEND_NODE['attention']!r} (the default), "
          f"{floors['sage']} on sage and {floors['stock']} on stock attention; "
          f"{len(FLOOR_STEMS)} declared in FLOOR_STEMS, the PDD reference arms derived")
    print(f"  declared: sage mode {h3_config.SAGE_NODE['mode']!r}, "
          f"sol tau {h3_config.SOL_RECOMMENDED_CUDA['tau']}, "
          f"min_tokens {h3_config.SOL_RECOMMENDED_CUDA['min_tokens']}")
    print(f"  PDD arms graded on SOL_PDD_CUDA instead: "
          f"end_percent {h3_config.SOL_PDD_CUDA['end_percent']}, "
          f"min_tokens {h3_config.SOL_PDD_CUDA['min_tokens']}, "
          f"dense_blocks {h3_config.SOL_PDD_CUDA['dense_blocks']!r}")
    print("  single-frame class: "
          + (f"{checked['single_frame']} graph(s) from GRAPH_DIRS "
             f"{sorted(img_dirs)}" if img_dirs else
             "EMPTY -- h3_config.GRAPH_DIRS routes no graph to a "
             "subdirectory, so nothing is exempted on single-frame grounds "
             "(the lane is parked; docs/h3_image_editing.md)"))
    print(f"  out of scope: workflows/bench/*_stamped_api.json are the dense "
          f"baselines and are outside graph_paths()")
    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for pr in problems:
            print(f"    - {pr}")
        return 1
    print("  ok    every live node matches h3_config, every exemption is necessary, "
          "no orphans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
