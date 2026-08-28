#!/usr/bin/env python3
"""A static reader must follow a linked widget, or it goes red on a good graph.

## Why this exists

A node input in an API graph is EITHER a literal (`"steps": 4`) OR a link to
another node's output (`"steps": ["50", 0]`). Every static reader in `bench/`
reads the first form. `check_distill_settings._literal` carried the consequence
as a comment: a link "is not a value this file can grade", so it returned
`None`, and the caller then reports "could not read the sampler's step count",
which is a FAIL. **Link a widget a check reads and the check goes red on a
graph that is completely fine.**

That is a repeat, not a hypothesis. When `MiniMaxH3PDDLoRA` began emitting
SIGMAS the PDD graphs stopped carrying a `BasicScheduler`, three checks' local
step readers each returned `None`, and `check_attention_defaults._steps_of`
records the result: it "reported eleven correctly-wired graphs as wrong". The
escaped instance is the same shape both times -- the value is present and
unambiguous, the reader cannot see where it comes from, and the check blames
the graph. No existing gate could have caught the link form, because every
shipped graph writes these widgets as literals and every check is green today.
`check_graph_discovery.py` prints the boundary itself: it "covers WHICH FILES a
scan sees, never which FIELDS it reads", and records that a scan reading only
`inputs` and missing every UI graph's `widgets_values` happened on 2026-08-27
with nothing to catch it. This file is that other half.

`h3_config.resolve_link` / `resolve_widget` is the one walker. This file is
what says it works, and what says it can fail.

## The controls, and why they are controls

**The link rule is graded against ComfyUI's own.** `h3_config._is_link` mirrors
`comfy_execution.graph_utils.is_link` -- the predicate the executor itself
applies -- because `h3_config` must stay importable with no ComfyUI on
`sys.path`. A mirrored rule is a second copy, so `link rule matches core` runs
both over one battery. Not a battery of expected answers: the two
implementations are compared to each other, and core is the authority.

**The output map is graded against declarations nobody writes twice.**
`OUTPUT_SOURCES` says how many outputs each class has and which input a
pass-through slot carries. For this pack's nodes that is checked against
`bench/node_id_manifest.json`, the committed baseline `check_node_ids.py`
maintains precisely so a schema change cannot rewrite its own expectation; for
core's primitives, against `define_schema()`. If `MiniMaxH3Resolution` grows an
eighth output, a link to slot 7 would read as "slot out of range" -- a
MALFORMED verdict on a fine graph, this file's whole subject -- and that case
goes red first.

**The link/literal equivalence is graded against the shipped tree.** Every
shipped graph that names a step count is read twice: once as it ships, and once
through a copy whose `steps` widget has been rewired to a `PrimitiveInt`
carrying the same number. The expectation is the graph's own answer, never a
number written here, and the population is every graph rather than one fixture.
No shipped graph links a widget today, which is exactly why the linked arm has
to be synthesised -- and exactly why it is cheap to get right now.

## What this asserts, i.e. what breaks if a case is deleted

  link rule matches core   `_is_link` and `comfy_execution.graph_utils.is_link`
                           agree on every shape in the battery. Delete it and
                           the mirrored copy is free to drift from the rule the
                           executor applies, which is how a link gets read as a
                           literal (or the reverse) with nothing said
  output map matches       every `OUTPUT_SOURCES` row agrees with an
                           independent declaration on slot count, on the
                           pass-through input's name, and on its widget index.
                           Delete it and a node that appends an output turns
                           every link past the old end into MALFORMED
  literal and link agree   `graph_schedule` returns the identical
                           `(steps, scheduler)` for a shipped graph and for the
                           same graph with `steps` fed from a constant node, in
                           BOTH graph forms. Delete it and the resolver can
                           stop following links while every other case here
                           still passes -- this is the one that grades the
                           defect
  ui link beats widget     a linked UI widget leaves a STALE literal in
                           `widgets_values`, and the shipped reference graphs
                           are full of them. Delete it and a reader may return
                           the stale number, which is worse than returning
                           nothing because it is confidently wrong
  states are distinct      COMPUTED, OPAQUE and MALFORMED are reported for the
                           things they name, and none of them is RESOLVED. Each
                           mutation is judged against the SAME graph unmutated,
                           so a case that never reached the resolver reads as a
                           failure rather than as proof. Delete it and the four
                           states collapse back into `None`, which is the API
                           this file exists to replace
  cycle terminates         a link chain that closes on itself is caught by the
                           CYCLE SET -- no node walked twice, and well short of
                           the depth bound -- inside a wall-clock alarm. The
                           first version asserted only "reports MALFORMED" and
                           stayed green with the cycle set deleted, because
                           `MAX_LINK_HOPS` reported the same verdict for the
                           other reason. Delete this and the two guards cover
                           for each other until neither is there, at which
                           point the file hangs rather than going red

Needs ComfyUI importable for `comfy_execution.graph_utils` and
`comfy_extras.nodes_primitive`; no CUDA, no server, no model, no checkpoint on
disk. The graph cases need no ComfyUI, so a partial environment still grades
something.

Exit codes: 0 all cases passed, 1 a case failed.

    python bench/check_graph_values.py
"""

from __future__ import annotations

import copy
import json
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))                 # ComfyUI root
sys.path.insert(0, str(REPO / "workflows"))

import h3_config as cfg  # noqa: E402

WORKFLOWS = REPO / "workflows"
MANIFEST = HERE / "node_id_manifest.json"

failures: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        failures.append(name)
        print(f"  FAIL  {name}: {exc}")
    else:
        print(f"  ok    {name}" + (f"   {detail}" if detail else ""))


# --------------------------------------------------------------------------
# the link rule, against the executor's own
# --------------------------------------------------------------------------

#: Shapes chosen so a wrong rule cannot agree by accident: the true links
#: differ in slot type and id, and the near misses each break exactly one
#: clause of the rule (length, id type, slot type, nesting, not-a-list).
LINK_BATTERY = [
    ["50", 0], ["50", 1], ["7", 0.0], ["node-with-dashes", 12],
    ["50"], ["50", 0, 0], [50, 0], ["50", "0"], ["50", None],
    [["50", 0]], [], 4, 4.0, "4", None, True, {"0": 50}, ("50", 0),
]


def case_link_rule_matches_core():
    """`h3_config._is_link` is the executor's rule, not a lookalike.

    The authority is `comfy_execution.graph_utils.is_link`, which is what
    `execution.get_input_data` branches on when it decides whether an input is
    a socket or a value. Ours is a mirror because `h3_config` may not import
    ComfyUI. Comparing the two is the only thing that keeps the mirror honest;
    comparing ours against answers typed here would pass on any rule both this
    file and the resolver happened to share.
    """
    from comfy_execution.graph_utils import is_link as core_is_link

    disagree = [v for v in LINK_BATTERY
                if bool(core_is_link(v)) != bool(cfg._is_link(v))]
    assert not disagree, (
        "core and h3_config disagree about whether these are links: "
        + ", ".join(repr(v) for v in disagree[:4]))
    positives = sum(1 for v in LINK_BATTERY if core_is_link(v))
    assert positives, "the battery holds no links; agreement on it proves nothing"
    assert positives < len(LINK_BATTERY), (
        "the battery holds no non-links; a rule that says yes to everything "
        "would pass")
    return f"{len(LINK_BATTERY)} shapes, {positives} of them links"


# --------------------------------------------------------------------------
# the output map, against declarations it does not own
# --------------------------------------------------------------------------

def _core_schema(class_type):
    """`(input ids, output count)` for a core node, from its own schema.

    Only the classes `OUTPUT_SOURCES` actually names are looked up. Import is
    by module rather than through `nodes.py`: `docs/comfy_notes.md`'s
    `import nodes` trap makes the wide path expensive and nothing here needs
    this pack registered.
    """
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy_extras.nodes_primitive as prim

    for cls in (prim.Int, prim.Float, prim.String, prim.StringMultiline,
                prim.Boolean):
        schema = cls.define_schema()
        if schema.node_id == class_type:
            return [i.id for i in schema.inputs], len(schema.outputs)
    return None


def case_output_map_matches_declarations():
    """Every `OUTPUT_SOURCES` row agrees with somebody else's declaration.

    Two independent sources, one per family. This pack's nodes are graded
    against `bench/node_id_manifest.json`, which `check_node_ids.py` keeps as a
    committed baseline exactly so it cannot be rewritten by the change it is
    supposed to catch. Core's primitives are graded against `define_schema()`.

    A row for a class NEITHER source knows is a failure rather than a skip: an
    unverifiable row is the state this file exists to prevent, and a silent
    skip is indistinguishable from a pass.

    The widget index is derived, not compared to a constant: it is the position
    of the pass-through input among the class's declared inputs. That
    derivation is only exact while every declared input is a widget, so a row
    whose class declares more than one input is refused rather than guessed at.
    """
    manifest = json.loads(MANIFEST.read_text())
    by_node_id = {v["node_id"]: v for v in manifest.values()}
    problems, graded = [], []
    for class_type, slots in cfg.OUTPUT_SOURCES.items():
        declared = by_node_id.get(class_type)
        if declared is not None:
            inputs, n_out = declared["inputs"], len(declared["outputs"])
            source = "node_id_manifest.json"
        else:
            core = _core_schema(class_type)
            if core is None:
                problems.append(
                    f"{class_type}: no independent declaration to grade the "
                    f"row against")
                continue
            inputs, n_out = core
            source = "define_schema()"
        graded.append(f"{class_type} ({source})")
        if len(slots) != n_out:
            problems.append(
                f"{class_type}: OUTPUT_SOURCES declares {len(slots)} slot(s), "
                f"{source} declares {n_out}")
        for i, slot in enumerate(slots):
            if slot is None:
                continue
            if slot.input_name not in inputs:
                problems.append(
                    f"{class_type} slot {i} passes through "
                    f"{slot.input_name!r}, which {source} does not declare")
                continue
            if slot.ui_widget is None:
                continue
            if len(inputs) != 1:
                problems.append(
                    f"{class_type} slot {i} pins ui_widget="
                    f"{slot.ui_widget} on a class with {len(inputs)} inputs; "
                    f"the index cannot be derived, so it cannot be graded")
                continue
            if slot.ui_widget != inputs.index(slot.input_name):
                problems.append(
                    f"{class_type} slot {i} pins ui_widget="
                    f"{slot.ui_widget}; {source} puts {slot.input_name!r} at "
                    f"{inputs.index(slot.input_name)}")
    assert not problems, "; ".join(problems[:4])
    assert graded, "OUTPUT_SOURCES is empty, so this case asserts nothing"
    return f"{len(graded)} row(s) graded"


# --------------------------------------------------------------------------
# the equivalence, against the shipped tree
# --------------------------------------------------------------------------

def _api_graphs():
    return [p for p in cfg.graph_paths(WORKFLOWS, include_bench=True)
            if p.name.endswith("_api.json")]


def _ui_graphs():
    return [p for p in cfg.graph_paths(WORKFLOWS, include_bench=True)
            if not p.name.endswith("_api.json")]


def _free_api_id(graph):
    return str(max((int(k) for k in graph if k.isdigit()), default=0) + 1)


def _link_api_steps(graph):
    """A copy of an API graph whose `BasicScheduler.steps` arrives over a link.

    The value is unchanged and moved, not rewritten: whatever the graph said is
    what the new `PrimitiveInt` carries. So the two copies are the same graph
    said two ways, and any difference in what a reader gets out of them is the
    reader's.
    """
    out = copy.deepcopy(graph)
    for node in out.values():
        if not isinstance(node, dict) or node.get("class_type") != "BasicScheduler":
            continue
        value = node.get("inputs", {}).get("steps")
        if not isinstance(value, int):
            continue
        new_id = _free_api_id(out)
        out[new_id] = {"class_type": "PrimitiveInt", "inputs": {"value": value}}
        node["inputs"]["steps"] = [new_id, 0]
        return out
    return None


def _link_ui_steps(graph):
    """The same rewiring in UI form, with the stale widget left in place.

    Leaving `widgets_values[1]` alone is the point: that is what the frontend
    does, and a reader that prefers it to the link reads a value the graph no
    longer uses.
    """
    out = copy.deepcopy(graph)
    for node in out.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "BasicScheduler":
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list) or len(widgets) < 2:
            continue
        if not isinstance(widgets[1], int):
            continue
        new_id = max(n.get("id", 0) for n in out["nodes"]) + 1
        new_link = max([row[0] for row in out.get("links", [])] or [0]) + 1
        out["nodes"].append({
            "id": new_id, "type": "PrimitiveInt", "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "links": [new_link]}],
            "widgets_values": [widgets[1]],
        })
        out.setdefault("links", []).append(
            [new_link, new_id, 0, node["id"], len(node.get("inputs") or []), "INT"])
        node.setdefault("inputs", []).append(
            {"name": "steps", "type": "INT", "link": new_link,
             "widget": {"name": "steps"}})
        return out
    return None


def case_literal_and_link_agree():
    """A linked `steps` reads as the same schedule the literal did.

    The expectation is the shipped graph's own `graph_schedule` result. That is
    the control the repo's rule asks for -- a second statement of the same
    fact, not a number this file computed -- and it is taken over every graph
    in the tree rather than over one fixture, so a reader that works on the
    shape of one graph and not another has nowhere to hide.

    Both forms are exercised because they take different paths: the API form
    resolves a named `[id, slot]`, the UI form resolves a link id through the
    link table and must beat the stale widget beside it.
    """
    problems, pairs = [], {"api": 0, "ui": 0}
    for path in _api_graphs():
        graph = json.loads(path.read_text())
        linked = _link_api_steps(graph)
        if linked is None:
            continue
        pairs["api"] += 1
        want, got = cfg.graph_schedule(graph), cfg.graph_schedule(linked)
        if want != got:
            problems.append(f"{path.name} (api): literal reads {want}, "
                            f"linked reads {got}")
    for path in _ui_graphs():
        graph = json.loads(path.read_text())
        linked = _link_ui_steps(graph)
        if linked is None:
            continue
        pairs["ui"] += 1
        want, got = cfg.graph_schedule(graph), cfg.graph_schedule(linked)
        if want != got:
            problems.append(f"{path.name} (ui): literal reads {want}, "
                            f"linked reads {got}")
    assert not problems, "; ".join(problems[:4])
    assert pairs["api"] and pairs["ui"], (
        f"nothing was rewired ({pairs}); the case compared a graph with itself")
    return f"{pairs['api']} api + {pairs['ui']} ui graph(s) agree either way"


def case_ui_link_beats_stale_widget():
    """A UI widget fed by a link is read from the link, not from the widget.

    The control is shipped, not synthesised: every reference graph wires
    `width`/`height`/`length` into its conditioner from `MiniMaxH3Resolution`
    and still carries the numbers in `widgets_values`. Those numbers are what a
    positional reader returns and they are not what the graph computes -- the
    resolution node parses its DynamicCombo label at run time, so the honest
    answer is COMPUTED.

    A resolver that ignored links here would return an int and look right,
    which is why this is a separate case from the equivalence above: that one
    can only see a reader that returns NOTHING, and this one sees a reader that
    returns something plausible.
    """
    seen, problems = 0, []
    for path in _ui_graphs():
        graph = json.loads(path.read_text())
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            entry = cfg._ui_input_entry(node, "width")
            if entry is None or entry.get("link") is None:
                continue
            widgets = node.get("widgets_values") or []
            stale = [w for w in widgets if isinstance(w, int)]
            got = cfg.resolve_widget(graph, node, "width",
                                     stale[0] if stale else None)
            seen += 1
            if got.state == cfg.RESOLVED:
                problems.append(
                    f"{path.name} node {node.get('id')}: read a linked width "
                    f"as the literal {got.value!r}")
    assert not problems, "; ".join(problems[:4])
    assert seen, ("no shipped UI graph links a `width` widget, so this case "
                  "compared nothing")
    return f"{seen} linked width widget(s), none read from the stale value"


# --------------------------------------------------------------------------
# the four states
# --------------------------------------------------------------------------

def _base_graph():
    """A minimal API graph whose `steps` resolves through one constant node."""
    return {
        "1": {"class_type": "BasicScheduler",
              "inputs": {"scheduler": "simple", "steps": ["2", 0]}},
        "2": {"class_type": "PrimitiveInt", "inputs": {"value": 4}},
        "3": {"class_type": "MiniMaxH3Resolution",
              "inputs": {"shape": "wide", "length": 362}},
    }


def _steps(graph):
    return cfg.resolve_widget(graph, graph["1"], "steps")


#: `(label, expected state, mutation)`. Each mutation is applied to a fresh
#: copy of `_base_graph()`, whose unmutated verdict is asserted RESOLVED first
#: -- so a mutation that never reached the resolver shows up as an unchanged
#: verdict rather than as evidence the state works.
def _mut_missing(g):
    del g["2"]


def _mut_slot(g):
    g["1"]["inputs"]["steps"] = ["3", 99]


def _mut_cycle(g):
    g["2"]["inputs"]["value"] = ["4", 0]
    g["4"] = {"class_type": "PrimitiveInt", "inputs": {"value": ["2", 0]}}


def _mut_deep(g):
    n = cfg.MAX_LINK_HOPS + 3
    g["1"]["inputs"]["steps"] = ["10", 0]
    for i in range(n):
        g[str(10 + i)] = {"class_type": "PrimitiveInt",
                          "inputs": {"value": [str(11 + i), 0]}}
    g[str(10 + n)] = {"class_type": "PrimitiveInt", "inputs": {"value": 4}}


def _mut_computed(g):
    g["1"]["inputs"]["steps"] = ["3", 0]


def _mut_unmapped(g):
    g["2"] = {"class_type": "SomeNodeNobodyDescribed", "inputs": {"value": 4}}


def _mut_wrong_input(g):
    del g["2"]["inputs"]["value"]


MUTATIONS = [
    ("absent node", cfg.MALFORMED, _mut_missing),
    ("slot past the last output", cfg.MALFORMED, _mut_slot),
    ("chain longer than MAX_LINK_HOPS", cfg.MALFORMED, _mut_deep),
    ("pass-through input the node lacks", cfg.MALFORMED, _mut_wrong_input),
    ("a run-time output", cfg.COMPUTED, _mut_computed),
    ("a class with no OUTPUT_SOURCES row", cfg.OPAQUE, _mut_unmapped),
]


def case_states_are_distinct():
    """Each unresolvable shape reports its own state, and none reports a value.

    The baseline is the same graph unmutated, and it must RESOLVE. That is what
    makes each row evidence: a mutation that left the verdict alone would be
    reporting on a path the resolver never took, which is the family
    `docs/checks.md` calls a check whose input cannot fail.

    Three distinct states appear here rather than one `None`, and that is the
    load-bearing part. MALFORMED says the graph is broken and a caller should
    go red; COMPUTED says the graph is fine and no static reader will ever know
    the value; OPAQUE says this resolver has not been taught the node. The
    three take opposite next actions and the old API could not tell them apart.
    """
    base = _steps(_base_graph())
    assert base.state == cfg.RESOLVED and base.value == 4, (
        f"the unmutated graph does not resolve ({base}); every row below "
        f"would be judged against a broken baseline")
    problems = []
    for label, want, mutate in MUTATIONS:
        graph = _base_graph()
        mutate(graph)
        got = _steps(graph)
        if got.state != want:
            problems.append(f"{label}: expected {want}, got {got.state} "
                            f"({got.reason or got.value!r})")
        elif got.value is not None:
            problems.append(f"{label}: reported {want} but still carried "
                            f"the value {got.value!r}")
        elif not got.reason:
            problems.append(f"{label}: reported {want} with no reason, so a "
                            f"caller cannot say what went wrong")
    assert not problems, "; ".join(problems[:4])
    states = sorted({want for _, want, _ in MUTATIONS})
    return f"{len(MUTATIONS)} shapes over {len(states)} states: {', '.join(states)}"


def case_cycle_terminates():
    """A cycle is caught AS A CYCLE: no node is walked twice.

    **The obvious version of this case could not fail, and a deliberate
    violation is what showed it.** Written as "a cyclic chain reports
    MALFORMED", it stayed green with the cycle set deleted: the walk went round
    and round until `MAX_LINK_HOPS` stopped it and reported MALFORMED for the
    other reason. Two guards, one verdict, and the case was reading whichever
    fired -- the family `docs/checks.md` calls a check whose input already
    satisfies the expected outcome.

    So the assertion is the property rather than the verdict: the reported
    chain visits each node once, and it is shorter than the depth bound. A
    two-node cycle must be reported after two nodes. With the cycle set gone
    the chain comes back as `2 -> 4 -> 2 -> 4 -> ...` for sixteen entries and
    both assertions go red.

    The alarm stays, and covers the third way this can go wrong: remove BOTH
    guards and the resolver does not fail, it hangs, and a hung check is not a
    red one.
    """
    graph = _base_graph()
    _mut_cycle(graph)

    def _fire(signum, frame):
        raise AssertionError(
            "the resolver did not return within 10 s on a cyclic link chain; "
            "nothing bounds the walk and every caller of resolve_widget hangs")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(10)
    try:
        got = _steps(graph)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert got.state == cfg.MALFORMED, (
        f"a cyclic chain reported {got.state}, not {cfg.MALFORMED}")
    assert got.via, "the cycle report names no node, so it cannot be found"
    assert len(got.via) == len(set(got.via)), (
        f"the walk visited a node twice before stopping ({' -> '.join(got.via)}); "
        f"the cycle set is not what caught this")
    assert len(got.via) < cfg.MAX_LINK_HOPS, (
        f"a {len(set(got.via))}-node cycle was reported after "
        f"{len(got.via)} nodes, at the MAX_LINK_HOPS bound; the depth guard "
        f"caught it, not the cycle set")
    return f"reported at {' -> '.join(got.via)}"


print("GRAPH VALUES output: a reader that can follow a linked widget")
check("link rule matches core", case_link_rule_matches_core)
check("output map matches declarations", case_output_map_matches_declarations)
check("literal and link agree", case_literal_and_link_agree)
check("ui link beats stale widget", case_ui_link_beats_stale_widget)
check("states are distinct", case_states_are_distinct)
check("cycle terminates", case_cycle_terminates)

if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("\nall ok -- a linked widget reads as the value behind it")
