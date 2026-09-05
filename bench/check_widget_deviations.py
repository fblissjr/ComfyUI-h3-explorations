#!/usr/bin/env python3
"""Every shipped widget value that differs from its node's own default is declared.

**The escaped instance this exists for.** `MiniMaxH3PDDLoRA.head_strength`
shipped at a literal `1.0` in all 20 PDD graphs while the node's schema default
was `-1.0`, the sentinel meaning "follow `strength`". Identical behaviour at
`strength` 1.0 and silently divergent the moment anyone edited it -- a shipped
graph and a freshly dragged-in node disagreeing. Every existing gate was green:
`check_literal_widgets.py` and `check_pdd_head_selection.py` read the NODE,
`check_pdd_sigmas.py`'s ui/api case compares the STEP COUNT only, and nothing
compared a graph's widget values against the node's own defaults. Two sessions
found it by hand, on the owner's rule that a workflow's values should not differ
from the node's without a reason.

**Why this shape and not a resolver.** The obvious design was to generalise
`h3_config.sol_for_graph` -- take what a graph IS, return what it should carry,
per widget. Measured before building: across all 80 API graphs, 34 (class,
widget) pairs match the node default everywhere, **25 deviate**, 23 have no
declared default. Of the 25, **eighteen take exactly one value in every graph**;
only four vary by graph, two of those are identity, and the fourth is
`MiniMaxH3SolAttn.end_percent`, which `sol_for_graph` already resolves. And
`head_strength` was single-valued -- so a resolver keyed on graph kind returns
the same wrong value for every kind and catches nothing. The failure was an
UNDECLARED DEVIATION, so declaration is what is graded.

**The allowlist carries the judgement, exactly as `check_literal_widgets.py`'s
`LITERAL_ZERO`/`SENTINELS` do**, and for the same reason: no predicate can tell
a deliberate house value from a mistake. A row cannot be added without saying
which KIND of deviation it is and why. Rows whose reason this repo does not
actually record say `UNRECORDED` rather than carrying an invented one, and the
count of those is printed -- a reason nobody wrote is a gap, not a formatting
slip.

**It goes red six ways.** An undeclared deviation; a `HOUSE` value moved off
its pinned value (including moved BACK to the node default, which stops being a
deviation and would otherwise be silently fine); a `HOUSE` row with no pinned
value; a declared row that no longer deviates anywhere (the fix landed and the
row is now a lie); and a stem naming a step count the graph does not run.

**What it does NOT cover, and it is a real boundary rather than a caveat: the
UI graphs.** This reads `*_api.json` exclusively. A deviation present in a UI
graph and absent from its API twin is INVISIBLE here -- demonstrated by an
independent red proof on 2026-08-31 that put the `head_strength` defect in the
UI form only and got green. Both forms come from one generator so they normally
agree, but "normally" is already doing work: `h3_text_to_video_pdd_manual_sigmas`
carries `steps` 6 in the UI and 0 in the API, legitimately. Teaching this check
to read UI graphs is the wrong fix -- `widgets_values` is POSITIONAL and the
DynamicCombo resolution below is name-based, so it would mean rebuilding the
frontend's widget-order mapping. **Enforced by nothing today.** The two
candidates, neither built: assert UI and API agree on every widget value rather
than only the step count (`bench/check_pdd_sigmas.py` does the narrow version),
or regenerate into a temp tree and diff, which would also catch the generator
having been edited without a rebuild -- an instance of which occurred on
2026-08-31, when `build_workflows.py` emitted `head_strength: -1.0` while all 20
shipped graphs still carried `1.0`.

    python bench/check_widget_deviations.py
    python bench/check_widget_deviations.py --object-info /tmp/oi.json

Exit codes follow `check_workflow_schema.py`'s convention, and the distinction
is the point:

    0   graphs were checked and every deviation is declared
    1   a real finding -- undeclared deviation, stale row, or stem mismatch
    2   this check DID NOT RUN and nothing was validated (no reachable
        /object_info, or no graphs found)

**Node defaults come from the live schema, never from a mirror kept here.**
`h3_config` cannot hold them: it imports without torch by design and node
modules do not. A mirror would also inherit the failure
`bench/check_sol_kernel.py` exists to catch -- `SOL_CUDA_DEFAULTS` drifted from
the node's own defaults on three keys with nothing asserting either.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "workflows"))

import h3_config  # noqa: E402

#: Inputs that name WHICH render this is rather than how it behaves: a scene, a
#: file, a seed, an output name. A defaults table has no business owning these
#: and they are excluded by rule rather than by ending up in the table with a
#: reason like "it is the prompt". Kept explicit -- an input added here stops
#: being graded, so it is a judgement and reads as one.
IDENTITY = {
    "filename_prefix", "prompt", "text", "image", "video", "audio_prompt",
    "unet_name", "lora_name", "vae_name", "clip_name", "type",
    "seed", "noise_seed", "value",
}

#: Declared deviations: `(class, widget) -> (kind, reason)`.
#:
#: `kind` is the part that cannot be omitted, and it is the whole discipline:
#:   HOUSE      a deliberate house value, the SAME in every graph that carries
#:              the widget. A HOUSE row MUST carry that value as a third
#:              element, and every graph is asserted against it -- see
#:              `case: a HOUSE widget silently moved` below.
#:   ARM        set per experimental arm, so it differs by design across graphs
#:   RESOLVED   another authority already owns and grades it; the reason names it
#:   UNRECORDED it deviates and this repo does not record why. Not a pass --
#:              a disclosed gap, counted and printed on every run.
#:
#: **Both rows that were UNRECORDED at first turned out to be recorded, in
#: files the author had not searched** -- `head_chunks` in
#: `docs/h3_geometry_and_nodes.md` and a timing table in `h3_config.py`,
#: `video_policy` in `docs/h3_references.md`. So treat an UNRECORDED row as a
#: prompt to look harder before treating it as a gap in the repo; twice out of
#: twice it was a gap in the search.
#: **Why HOUSE rows pin a value and the others do not.** Grading "is this
#: deviation declared" leaves a hole that is the mirror image of the escape
#: this check was built for: a graph moved BACK to the node's default stops
#: deviating, so it is silently fine. Found by a red proof that expected a
#: graph pushed off the trained canvas to go red and got green -- correctly,
#: since 41 other graphs still deviated and the row was not stale. A HOUSE
#: value is single by definition, so pinning it closes the hole. ARM values
#: vary by graph on purpose and RESOLVED values belong to another authority, so
#: neither can be pinned here without becoming a second copy.
#:
#: **HOUSE grades EVERY graph carrying the widget, not only the deviating
#: ones**, which is what lets it catch a value moved back to the node default.
#: The price is that HOUSE is only available where the default never
#: legitimately appears. `head_chunks` deviates in one graph and sits at the
#: default in 76, so it cannot be HOUSE, and a change from 4 to 8 on its probe
#: is NOT caught -- the row asserts that a deviation is declared, not which
#: value it takes. Establishing that would need per-graph intent, which is the
#: resolver this check exists instead of.
#:
#: **The rule has caught four of this table's own misclassifications**, which
#: is the argument for keeping it: `strength_model`, `shape` and
#: `shape.wide_resolution` were written HOUSE and take more than one value;
#: `head_chunks` was then written HOUSE on the reasoning that it takes exactly
#: one, and 76 graphs at the default went red. The KIND claim is graded, not
#: trusted, and it caught its author four times in one sitting.
DECLARED: dict[tuple[str, str], tuple] = {
    ("BasicScheduler", "steps"):
        ("ARM", "the arm's evaluation count; graded against the LoRA it loads "
                "by bench/check_distill_settings.py"),
    ("MiniMaxH3Resolution", "length"):
        ("ARM", "the arm's frame count; h3_config.LENGTH / LONG_LENGTH"),
    ("MiniMaxH3AppendRefImage", "size_policy"):
        ("HOUSE", "'max' matches the vendor -- it caps the short edge and "
                  "applies no area cap, where the node default 'match' sizes to "
                  "the generation's pixel area and is several times smaller on a "
                  "landscape canvas. docs/h3_references.md is the authority. "
                  "Load-bearing in 42 graphs and enforced by nothing until this "
                  "check.", "max"),
    ("MiniMaxH3AppendRefImage", "qwen_view"):
        ("ARM", "'shared' on the A and C arms of the reference-view ablation "
                "(h3_probe_refview_*), which exist to render the state "
                "docs/h3_references.md warns about. Deliberate on those two "
                "graphs and nowhere else."),
    ("MiniMaxH3Resolution", "shape"):
        ("ARM", "the arm's aspect: 'wide' on 41, 'standard' on 27, and one "
                "each of tall/ultrawide/square/custom. The node's first option "
                "is 'ultrawide'. docs/h3_resolutions.md owns the list."),
    ("MiniMaxH3Resolution", "shape.wide_resolution"):
        ("ARM", "1344x768 in 39 graphs -- a trained canvas, and what anything "
                "informing a shipped decision is measured at -- against "
                "1152x768 on two cheaper probe arms. Classified HOUSE first; "
                "the pinned-value rule caught that it takes two values. "
                "docs/h3_resolutions.md owns the list. NOTE: this is why a "
                "graph moved off 1344x768 does NOT go red here and is what anything informing "
                  "a shipped decision is measured at; the node's first option "
                "."),
    ("MiniMaxH3Resolution", "shape.standard_resolution"):
        ("HOUSE", "1024x768 on every 4:3 arm, cheaper per frame than the "
                  "node's first option. docs/h3_resolutions.md.",
         "1024x768  4/3  768 tok/frame  0.58x"),
    ("MiniMaxH3Resolution", "shape.tall_resolution"):
        ("ARM", "768x1344 on the portrait canvas probe -- the trained canvas "
                "rotated, not the node's first tall option."),
    ("MiniMaxH3Resolution", "shape.ultrawide_resolution"):
        ("ARM", "1536x672 on the ultrawide canvas probe, at the same "
                "1008 tok/frame budget as the trained canvas."),
    ("MiniMaxH3Resolution", "shape.width"):
        ("ARM", "960 on the turbo home-canvas probe; h3_config.TURBO_HOME_CANVAS"),
    ("MiniMaxH3Resolution", "shape.height"):
        ("ARM", "544 on the turbo home-canvas probe; h3_config.TURBO_HOME_CANVAS"),
    ("MiniMaxH3AppendRefImage", "size_policy.allow_upscale"):
        ("ARM", "True only on the two probes that exist to price upscaling "
                "(h3_probe_reference_upscale, h3_probe_refview_c_parity). It "
                "was flipped OFF everywhere else on 2026-08-28 after an audit "
                "found it costing ~6,300 extra reference rows per step for a "
                "benefit this repo has never measured."),
    ("MiniMaxH3AppendRefImage", "qwen_view.qwen_short_edge"):
        ("ARM", "2048 on the B arm of the reference-view ablation, which is "
                "what that arm varies. The shipped default 512 is a PRIOR "
                "resting on one render at one seed -- docs/h3_references.md."),
    ("MiniMaxH3SolAttn", "end_percent"):
        ("RESOLVED", "h3_config.sol_for_graph(pdd, steps); graded by "
                     "bench/check_attention_defaults.py"),
    ("ManualSigmas", "sigmas"):
        ("RESOLVED", "h3_config.PDD_MANUAL_SIGMAS, the six-block tail-weighted "
                     "partition; its 6dp rounding is load-bearing and "
                     "bench/check_pdd_sigmas.py asserts the derivation"),
    ("MiniMaxH3SigmaShift", "shift_video"):
        ("ARM", "6.0 on the 768p turbo arms, which were distilled at that "
                "shift; h3_config.TURBO_768P_SHIFT"),
    ("MiniMaxH3PDDLoRA", "patch_heads"):
        ("ARM", "False on the headfree control arm, which is the whole point "
                "of that graph"),
    ("MiniMaxH3PDDLoRA", "steps"):
        ("ARM", "0 on the manual_sigmas graph, where ManualSigmas supplies the "
                "schedule and the node must not emit its own"),
    ("MiniMaxH3Conditioning", "canvas"):
        ("ARM", "'explicit' where the arm pins width/height rather than taking "
                "the resolution node's shape"),
    ("MiniMaxH3Conditioning", "width"):
        ("ARM", "the arm's canvas, set with canvas='explicit'"),
    ("MiniMaxH3Conditioning", "length"):
        ("ARM", "the arm's frame count, set with canvas='explicit'"),
    ("VHS_VideoCombine", "frame_rate"):
        ("HOUSE", "h3_config.FPS -- the rate H3 renders at, not the node's",
         24.0),
    ("VHS_VideoCombine", "format"):
        ("HOUSE", "h264-mp4 so clips play in the blind-scoring app",
         "video/h264-mp4"),
    ("VHS_LoadVideo", "force_rate"):
        ("HOUSE", "h3_config.FPS, so a reference video is resampled to the "
                  "rate the model works at", 24.0),
    ("VHS_LoadVideo", "frame_load_cap"):
        ("ARM", "the arm's frame count; h3_config.REF_VIDEO_LENGTH"),
    ("SplitSigmas", "step"):
        ("HOUSE", "h3_config.SPLIT_AT, the two-pass split point", 2),
    ("MiniMaxH3SageAttention", "head_chunks"):
        ("ARM", "4 on h3_probe_head_chunks, the one graph that exists to "
                  "exercise head chunking. It is a MEASURED group count, not "
                  "a guess: docs/h3_geometry_and_nodes.md records ~3227 MiB "
                  "saved at 4 groups, and workflows/h3_config.py's table has the "
                "head4/ffn1 row at 396.5s / 13475 MiB / -3227 / 0.998x -- the "
                  "saving is real and the speed cost is nil. NOT House: 76 "
                  "graphs carry this input at the node default, and a HOUSE "
                  "row grades every carrier."),
    ("MiniMaxH3ReferenceConditioning", "video_policy"):
        ("ARM", "'release' on four graphs -- h3_probe_release_video_policy and "
                "the three refview probes -- against 'encoder' on 43. "
                "docs/h3_references.md: 'release' handles the vendor's own "
                "video sizing locally AND enables the coupled Qwen stage, "
                "while 'encoder' keeps native-compatible VAE sizing, which is "
                "why generated graphs use it. The probes are the arms that "
                "vary it."),
    ("LoraLoaderModelOnly", "strength_model"):
        ("ARM", "1.0 on eight graphs and h3_config.TURBO_OWNER_STRENGTH 0.75 "
                "on five -- the owner's turbo recipe. Classified HOUSE first; "
                "the pinned-value rule caught the second value."),
    ("EasyCache", "verbose"):
        ("HOUSE", "h3_config.CACHE_NODE['verbose'] -- the cache logs what it "
                  "skipped, which is the only way to see it worked", True),
    ("SageChainAssert", "exercise"):
        ("ARM", "the assert node's own probe knobs, set per instrumentation "
                "graph"),
    ("SageChainAssert", "require_forward_patch"):
        ("ARM", "as above"),
    ("SageChainAssert", "require_override"):
        ("ARM", "as above"),
    ("SageChainAssert", "require_absent"):
        ("ARM", "True on every arm that patches attention not at all -- the "
                "true baseline and the PDD reference arms -- so the node proves "
                "the graph is the baseline it claims to be; False wherever sage "
                "or Sol is wired. Set by the generator's `_assert_inputs` from "
                "the chain, since 2026-09-03 (Sol-alone state added 2026-09-04)"),
}

#: **There is deliberately no exception table here.** One was written on
#: 2026-08-31 holding the two `turbo_4step_768p` stems, whose `_4step` named
#: the LoRA file while the graphs ran `TURBO_768P_STEPS` = 6. The owner
#: refused it: a stem saying `_Nstep` names the evaluations the graph
#: denoises, full stop, and the two graphs were removed instead.
#:
#: The history is worth keeping because it shows the exception would have
#: been a lie. Those stems were ACCURATE when written --
#: `bench/results/2026-08-20_power_limit_pair_verdict.json` describes that
#: graph as "4 steps" -- and went stale on 2026-08-23 when the recipe moved
#: to six. So the rationale the table would have carried ("the 4step names
#: the LoRA") was a post-hoc reading of ordinary drift. An exception
#: mechanism here would have made every future rename optional.

_STEP_NODES = ("BasicScheduler", "MiniMaxH3PDDLoRA")


def object_info(url: str | None, cached: str | None) -> dict | None:
    if cached:
        return json.loads(Path(cached).read_text())
    try:
        with urllib.request.urlopen(f"{url}/object_info", timeout=60) as fh:
            return json.load(fh)
    except Exception:
        return None


def _entry(spec: dict, key: str):
    """One input declaration out of a node spec OR a DynamicCombo option.

    **Both spellings, and that is not defensive padding.** A node spec nests
    its sections under `"input"`; a DynamicCombo option nests its own under
    `"inputs"`. A first version read only `"input"`, so every sub-input
    resolved to "no declared default" and dropped silently into the ungradeable
    bucket -- the check reported them in its count and graded none of them.
    Found by a red proof that expected `dit_short_edge` 2048 -> 512 to go red
    and got green; nothing about the output said anything was wrong.
    """
    for holder in ("input", "inputs"):
        section_map = spec.get(holder)
        if not isinstance(section_map, dict):
            continue
        for section in ("required", "optional"):
            entry = (section_map.get(section) or {}).get(key)
            if entry is not None:
                return entry
    return None


def _default_of_entry(entry):
    """`(True, value)` for a plain input declaration.

    A combo's first option IS its default -- ComfyUI selects it when a prompt
    omits the key -- so it counts, which is what makes `format` and
    `video_policy` gradeable at all.
    """
    if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], dict) \
            and "default" in entry[1]:
        return (True, entry[1]["default"])
    if isinstance(entry, list) and entry and isinstance(entry[0], list):
        return (True, entry[0][0] if entry[0] else None)
    return (False, None)


def declared_default(oi: dict, cls: str, key: str, node_inputs: dict | None = None):
    """`(True, value)` when the node declares a default for this input.

    **DynamicCombo inputs are resolved, not skipped, and that is the whole
    reason this function is not three lines.** A `COMFY_DYNAMICCOMBO_V3` input
    carries its options inline and each option brings its own sub-inputs, which
    a graph writes as `parent.child`. A first version treated both as
    "no declared default" and reported 23 ungradeable inputs -- and they were
    not a random 23. They were `qwen_view`/`qwen_short_edge`,
    `size_policy`/`allow_upscale`/`dit_short_edge`, `MiniMaxH3Resolution.shape`
    and `MiniMaxH3SolAttn.selection`/`tau`: precisely the knobs `CLAUDE.md`
    spends the most words warning about. A check blind to exactly the inputs
    with the worst history is worse than no check, because its green reads as
    coverage.

    The parent's default is its FIRST option's key. A sub-input's default comes
    from the option the graph actually CHOSE, which is why `node_inputs` is
    needed: `dit_short_edge` exists only under `max`, and grading it against
    `match`'s (absent) declaration would be grading a different input.
    """
    spec = oi.get(cls)
    if not spec:
        return (False, None)

    if "." in key:
        parent, child = key.split(".", 1)
        pentry = _entry(spec, parent)
        if not (isinstance(pentry, list) and pentry and pentry[0] == "COMFY_DYNAMICCOMBO_V3"):
            return (False, None)
        options = (pentry[1] or {}).get("options") or []
        chosen = (node_inputs or {}).get(parent)
        if chosen is None and options:
            chosen = options[0].get("key")
        for opt in options:
            if opt.get("key") != chosen:
                continue
            centry = _entry(opt, child)
            return _default_of_entry(centry) if centry is not None else (False, None)
        return (False, None)

    entry = _entry(spec, key)
    if entry is None:
        return (False, None)
    if isinstance(entry, list) and entry and entry[0] == "COMFY_DYNAMICCOMBO_V3":
        options = (entry[1] or {}).get("options") or []
        return (True, options[0].get("key")) if options else (False, None)
    return _default_of_entry(entry)


def graph_step_count(graph: dict):
    """Evaluations this graph runs, resolved through a PrimitiveInt link."""
    prim = {nid: n.get("inputs", {}).get("value")
            for nid, n in graph.items() if n.get("class_type") == "PrimitiveInt"}
    for node in graph.values():
        if node.get("class_type") not in _STEP_NODES:
            continue
        raw = node.get("inputs", {}).get("steps")
        val = prim.get(raw[0]) if isinstance(raw, list) else raw
        if val:
            return int(val)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--object-info", help="cached /object_info JSON instead of --url")
    args = ap.parse_args(argv)

    oi = object_info(args.url, args.object_info)
    if oi is None:
        print("DID NOT RUN: no reachable /object_info. Start ComfyUI, or pass\n"
              "--object-info with a saved copy. Nothing was validated, which is\n"
              "not the same as passing.")
        return 2

    paths = [p for p in h3_config.graph_paths(_REPO / "workflows")
             if str(p).endswith("_api.json")]
    if not paths:
        print("DID NOT RUN: no API graphs found through h3_config.graph_paths.")
        return 2

    undeclared: dict[tuple[str, str], list] = {}
    seen_deviating: set[tuple[str, str]] = set()
    stem_bad: list[str] = []
    house_bad: list[str] = []
    no_default: set[tuple[str, str]] = set()
    checked = 0

    for path in paths:
        graph = json.loads(Path(path).read_text())
        stem = Path(path).name[: -len("_api.json")]

        m = re.search(r"_(\d+)step", stem)
        if m:
            got = graph_step_count(graph)
            if got is not None and got != int(m.group(1)):
                stem_bad.append(f"{stem}: stem says {m.group(1)} step(s), graph runs {got}")

        for node in graph.values():
            cls = node.get("class_type", "")
            for key, val in node.get("inputs", {}).items():
                if isinstance(val, list) or key in IDENTITY:
                    continue
                has, default = declared_default(oi, cls, key,
                                                node.get("inputs", {}))
                if not has:
                    no_default.add((cls, key))
                    continue
                checked += 1
                if val == default:
                    continue
                seen_deviating.add((cls, key))
                if (cls, key) not in DECLARED:
                    undeclared.setdefault((cls, key), []).append((stem, val, default))

    # A HOUSE widget is single-valued by declaration, so every graph carrying
    # it is graded against the pinned value -- including graphs where it equals
    # the node default and therefore never entered `seen_deviating`.
    for path in paths:
        graph = json.loads(Path(path).read_text())
        stem = Path(path).name[: -len("_api.json")]
        for node in graph.values():
            cls = node.get("class_type", "")
            for key, val in node.get("inputs", {}).items():
                if isinstance(val, list) or key in IDENTITY:
                    continue
                row = DECLARED.get((cls, key))
                if not row or row[0] != "HOUSE" or len(row) < 3:
                    continue
                if val != row[2]:
                    house_bad.append(
                        f"{stem}: {cls}.{key} is {val!r}, the declared house "
                        f"value is {row[2]!r}")

    missing_value = sorted(k for k, r in DECLARED.items()
                           if r[0] == "HOUSE" and len(r) < 3)
    stale = sorted(set(DECLARED) - seen_deviating)
    unrecorded = sorted(k for k, row in DECLARED.items() if row[0] == "UNRECORDED")

    print(f"{len(paths)} API graph(s), {checked} gradeable widget value(s) against "
          f"the live schema")
    print(f"  declared deviations exercised : {len(seen_deviating & set(DECLARED))}"
          f" of {len(DECLARED)}")
    print(f"  inputs the node declares no default for (ungradeable): {len(no_default)}")
    if unrecorded:
        print(f"  UNRECORDED reasons, disclosed rather than invented: {len(unrecorded)}")
        for cls, key in unrecorded:
            print(f"      {cls}.{key} -- {DECLARED[(cls, key)][1]}")

    if not undeclared and not stale and not stem_bad and not house_bad \
            and not missing_value:
        print("\nok -- every deviation from a node default is declared with a reason")
        return 0

    for (cls, key), rows in sorted(undeclared.items()):
        stems = sorted({r[0] for r in rows})
        val, default = rows[0][1], rows[0][2]
        print(f"\nFAIL  undeclared deviation: {cls}.{key}")
        print(f"      graphs carry {val!r}, the node's default is {default!r}")
        print(f"      {len(stems)} graph(s), e.g. {stems[0]}")
        print(f"      Add a row to DECLARED naming the KIND and why, or change "
              f"the generator so the graphs stop deviating.")
    for cls, key in stale:
        print(f"\nFAIL  stale DECLARED row: {cls}.{key} no longer deviates in any "
              f"graph.\n      The row now claims a deviation that is not there. "
              f"Delete it.")
    for line in house_bad:
        print(f"\nFAIL  HOUSE value moved: {line}\n      A HOUSE row pins one "
              f"value for every graph. Change it back, or if the graph is right, "
              f"the row is wrong.")
    for cls, key in missing_value:
        print(f"\nFAIL  HOUSE row without a pinned value: {cls}.{key}\n      "
              f"A HOUSE deviation is single-valued by declaration, so the row "
              f"must carry it as a third element. Use ARM if it legitimately "
              f"varies.")
    for line in stem_bad:
        print(f"\nFAIL  stem/step mismatch: {line}\n      Rename the graph so the "
              f"stem names the evaluations it denoises, or remove it. There is no "
              f"exception table; see the note above _STEP_NODES for why.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
