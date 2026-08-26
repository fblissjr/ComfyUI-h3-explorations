#!/usr/bin/env python3
"""Check every shipped graph carries the attention configuration `h3_config` declares.

## What was uncontrolled, and what this replaces

`docs/checks.md`'s standing audit carried this row: *"Sol-Attn is on by default
in every shipped video workflow"* — enforced by **one graph, by accident**.
`check_bench_matches_shipped.py` compares the bench harness against
`h3_probe_sol_on_api.json`, so stripping Sol from that one file turns it red
while stripping it from any other goes unnoticed. That was measured by
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

Bypass (`mode=4`) is how the shipped graphs disable Sol **deliberately**, and is
not a defect. Only an active-but-unreachable node is.

## Why the exemption list cannot rot

Three kinds of graph legitimately ship without Sol. A hand-written list of
exempt filenames is the shape that goes stale silently, which is the objection
`docs/checks.md` raised against this check existing at all. So:

  * the single-frame class is taken from `h3_config.GRAPH_DIRS`, not from
    filenames — it is whatever the generator routes to `image/`;
  * every exemption is asserted **necessary**. If an exempt graph turns out to
    have live Sol, that is a failure, not a pass. So an exemption that stops
    being true goes red instead of quietly covering a graph nobody checks.

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
from build_workflows import SOL_SELECTION_INPUTS, sol_widget_order  # noqa: E402

WORKFLOWS = _REPO / "workflows"
SOL = "SolAttnMiniMax"
SAGE = "MiniMaxH3SageAttention"

# `h3_config` states that SAGE_NODE's key order IS the node's declared input
# order, because a UI graph maps widget values positionally. Derived rather than
# repeated, so a reordering there cannot leave a stale copy here.
SAGE_WIDGET_ORDER = tuple(h3_config.SAGE_NODE)

OUTPUT_TYPES = {"VHS_VideoCombine", "SaveImage", "PreviewImage", "SaveAudio",
                "SaveAnimatedWEBP", "SaveWEBM", "SaveVideo", "PreviewAny"}

#: {graph stem: (node field, reason)}. A per-graph deviation that is the point
#: of the graph. A reason naming a file rather than a mechanism is not a reason.
DEVIATIONS = {
    "h3_probe_head_chunks": ("head_chunks",
                             "this graph exists to run the 1-vs-N head-chunking "
                             "A/B that h3_config asks for; the deviating value "
                             "IS its subject"),
}

#: Graphs that legitimately ship without live Sol, by MECHANISM. The
#: single-frame class is not listed here -- it is derived from GRAPH_DIRS below.
SOL_EXEMPT_STEMS = {
    "h3_probe_capture_ref3":
        "activation capture: h3_capture.py records the attention inputs a dense "
        "baseline is measured from, and Sol gives sage only the steps outside "
        "its sigma window, so a capture taken through Sol is a different "
        "trajectory than the analysis assumes",
    "h3_probe_capture_ref3_fl2va":
        "the same capture on the fl2va checkpoint with no LoRA, the control "
        "for whether block 49's input structure is a property of the released "
        "weights rather than of ref2va; Sol off for the same reason as its twin",
    "h3_probe_turbo_768p_sla_dense":
        "comparative arm: the Turbo-SLA LoRA under sage alone, the repo's "
        "dense-baseline convention, one of three attention regimes the SLA "
        "probe set spans (Sol, router, dense)",
    "h3_probe_turbo_768p_sla_router":
        "comparative arm: the Turbo-SLA LoRA under the sparse top-k router it "
        "was distilled with (MiniMaxH3SLARouter), which replaces the sage+Sol "
        "chain outright; a Sol node after it would route a second time",
    **{stem: (
        "numerical probe: the PDD arms patch attention NOT AT ALL -- no sage "
        "and no Sol, unlike the sla_dense arm which is sage-only. Both change "
        "attention numerics and the subject of these graphs is a numerical "
        "mechanism in the output head, so either one wired in puts a second "
        "approximation in the path of an experiment about the first. Owner "
        "decision 2026-08-26, after a wrong-head defect that these would have "
        "made unattributable")
       for stem in ("h3_probe_ref2v_pdd", "h3_probe_ref2v_pdd_headfree",
                    "h3_probe_ref2v_pdd_345", "h3_probe_ref2v_pdd_8s")},
}


def load(path):
    return json.loads(path.read_text())


def is_ui(g):
    return isinstance(g.get("nodes"), list)


def reachable(g):
    """Node ids that feed an output node. Both graph formats."""
    if is_ui(g):
        nodes = {n["id"]: n for n in g["nodes"]}
        edges = {}
        for link in g.get("links", []):
            if isinstance(link, list) and len(link) >= 5:
                edges.setdefault(link[3], set()).add(link[1])
        seeds = [i for i, n in nodes.items() if n.get("type") in OUTPUT_TYPES]
    else:
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


def _sol_ui_order(widgets):
    """Widget ids for a UI-form Sol node, whose order depends on its own
    `selection` value (widget 0).

    An unrecognised selection returns the selector alone rather than guessing
    an order: pairing the remaining values against the wrong names would
    invent settings the graph does not have. The lone `selection` entry is
    enough for the comparison against h3_config to report the real problem.
    """
    selected = widgets[0] if widgets else None
    if selected not in SOL_SELECTION_INPUTS:
        return ("selection",)
    return sol_widget_order({"selection": selected})


def attn_nodes(g, want):
    """[(node_id, values_by_name, state)] for every `want` node in `g`.

    state is `live`, `orphaned` (ACTIVE and unreachable -- the defect),
    `bypassed` or `muted`.
    """
    live = reachable(g)
    out = []
    if is_ui(g):
        for n in g["nodes"]:
            if n.get("type") != want:
                continue
            widgets = n.get("widgets_values") or []
            if want == SOL:
                order = _sol_ui_order(widgets)
            else:
                order = SAGE_WIDGET_ORDER
            vals = dict(zip(order, widgets))
            mode = n.get("mode", 0)
            state = ("bypassed" if mode == 4 else "muted" if mode == 2
                     else "live" if n["id"] in live else "orphaned")
            out.append((n["id"], vals, state))
    else:
        for i, n in g.items():
            if not isinstance(n, dict) or n.get("class_type") != want:
                continue
            vals = {k: v for k, v in (n.get("inputs") or {}).items()
                    if not isinstance(v, list)}
            if want == SOL:
                # The API form keys the selected option's inputs under the
                # combo (`selection.tau`). Strip the prefix so both forms are
                # graded in h3_config's vocabulary -- left dotted, `tau` is
                # simply absent from `vals`, every `if k in vals` comparison
                # below skips it, and the check goes green having graded
                # nothing about the knob it exists for.
                vals = {k.split(".", 1)[1] if k.startswith("selection.") else k: v
                        for k, v in vals.items()}
            out.append((i, vals, "live" if i in live else "orphaned"))
    return out


def single_frame_dirs():
    """The image class, from the generator's own routing rather than filenames."""
    return {d for d in h3_config.GRAPH_DIRS if d}


def main() -> int:
    paths = h3_config.graph_paths(WORKFLOWS)
    img_dirs = single_frame_dirs()
    problems, checked = [], {"sol": 0, "sage": 0, "graphs": 0}
    exempt_seen = {k: False for k in SOL_EXEMPT_STEMS}

    for p in paths:
        g = load(p)
        stem = p.stem[:-4] if p.stem.endswith("_api") else p.stem
        in_image = p.parent.name in img_dirs
        exempt_reason = SOL_EXEMPT_STEMS.get(stem)
        if exempt_reason:
            exempt_seen[stem] = True
        checked["graphs"] += 1

        sol = attn_nodes(g, SOL)
        sage = attn_nodes(g, SAGE)
        checked["sol"] += len(sol)
        checked["sage"] += len(sage)

        # --- no_orphans: applies to EVERY graph, exempt or not -------------
        for nid, _vals, state in sol + sage:
            if state == "orphaned":
                problems.append(
                    f"{p.relative_to(_REPO)}: node {nid} is an ACTIVE node whose "
                    f"MODEL output nothing consumes. ComfyUI walks backwards from "
                    f"the output nodes, so it never executes. To disable it on "
                    f"purpose, bypass it (mode=4).")

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
        for nid, vals, state in sol:
            if state != "live":
                continue
            for k, want in h3_config.SOL_RECOMMENDED_CUDA.items():
                if k in vals and vals[k] != want:
                    problems.append(
                        f"{p.relative_to(_REPO)}: node {nid} {SOL}.{k} is "
                        f"{vals[k]!r}, h3_config.SOL_RECOMMENDED_CUDA says {want!r}")

        # --- sage_values ---------------------------------------------------
        dev_field, _dev_why = DEVIATIONS.get(stem, (None, None))
        for nid, vals, state in sage:
            if state != "live":
                continue
            for k, want in h3_config.SAGE_NODE.items():
                if k == dev_field:
                    continue
                if k in vals and vals[k] != want:
                    problems.append(
                        f"{p.relative_to(_REPO)}: node {nid} {SAGE}.{k} is "
                        f"{vals[k]!r}, h3_config.SAGE_NODE says {want!r}")

    # A declared exemption for a graph that does not exist is also rot.
    for stem, seen in exempt_seen.items():
        if not seen:
            problems.append(
                f"SOL_EXEMPT_STEMS names {stem!r}, which matches no graph under "
                f"graph_paths(). Remove the entry.")

    print(f"  {checked['graphs']} graphs, {checked['sol']} {SOL} and "
          f"{checked['sage']} {SAGE} node(s)")
    print(f"  declared: sage mode {h3_config.SAGE_NODE['mode']!r}, "
          f"sol tau {h3_config.SOL_RECOMMENDED_CUDA['tau']}, "
          f"min_tokens {h3_config.SOL_RECOMMENDED_CUDA['min_tokens']}")
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
