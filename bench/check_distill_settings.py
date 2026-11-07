#!/usr/bin/env python3
"""Check every distill is loaded at the shift and step count it was trained at.

A distill runs on its trainer's contract (`docs/wiki/references.md`): Pika H3
Flex on its own six shift-12 sigmas on `lcm`. It does not carry its shift in
the file, so a graph at the wrong one renders plausibly wrong and **nothing
errors**.

Claims, i.e. what breaks if a case is deleted:
  loaders complete      every node of ours with a `lora_name` input is in
                        `LORA_LOADER_CLASSES`, or its graph reads as a base
                        graph and every rule below passes it
  graphs are consistent  EVERY shipped API graph. Each distill is graded on
                        its own contract; a graph with none must sit at the
                        base checkpoint's own 12/3. It FAILS if either
                        population is empty: a walk that matches no file
                        grades nothing, and nothing reads as a pass
  no retired turbo      a graph loading any lightx2v or other turbo LoRA
                        fails: that lane is closed (`docs/roadmap.md`,
                        "Closed lanes", 2026-09-26). Its vendor-row grading
                        (the `LEGAL` table, `UNATTESTED`, `OWNER_RECIPE`) went
                        with it and is in git
  no retired PDMD       a graph loading a PDMD LoRA fails the same way:
                        retired 2026-10-05, and the contract grading it
                        had (one file, at the call, `simple` at the
                        trained step count) is in git
  no FlashGen or FastH3 a graph loading a FlashGen LoRA, or any model loader
                        naming a FastH3 checkpoint, fails the same way:
                        retired 2026-11-04 with every fast distill. Their
                        contract grading (FlashGen's file, shift, sigmas,
                        sampler and strength; FastH3's 10/3, template and
                        contract sampling and VSA; the step switches through
                        either) is in git
  no retired PDD        a graph loading a PDD LoRA or `MiniMaxH3PDDLoRA` fails
                        the same way: retired 2026-11-06 (`docs/roadmap.md`,
                        "Closed lanes"). Its 32-point grid, manual-knot, and
                        `nfe` contract grading went with it and is in git

No CUDA, no model, no ComfyUI import. Reads API-form JSON (`*_api.json`,
{node_id: {class_type, inputs}}) and a config module.

Exit codes: 0 all cases passed, 1 a case failed.

    python bench/check_distill_settings.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

WORKFLOWS = REPO / "workflows"

# A bare `WORKFLOWS.glob` is non-recursive and quietly stops covering any
# directory `GRAPH_DIRS` routes to -- as `workflows/image/` was from 2026-08-16
# until the single-frame lane was parked on 2026-08-27. See h3_config.GRAPH_DIRS.
from h3_config import (LORA_LOADER_CLASSES, graph_paths, graph_schedule,  # noqa: E402
                        resolve_link)


class Found(NamedTuple):
    loras: list[str]
    shift: tuple[float, float] | None
    steps: int | None
    # Read here, graded nowhere in this file. `check_distill_grid.py` imports
    # this reader rather than growing a second graph walk, and the scheduler
    # is the field it needs: a LoRA loaded at the right shift and step count is
    # still off its distillation grid if the scheduler places the steps
    # somewhere else.
    scheduler: str | None = None
    # EVERY shift node, in iteration order. `shift` above keeps the last one
    # for the callers that want a single answer; a split graph carries two
    # (`build_workflows.py::_plain_model_chain` builds node 40 beside node 19)
    # and reading one of two silently grades half the graph.
    shifts: tuple[tuple[float, float], ...] = ()
    # {lora filename: strength_model}. Strength was read by nothing until
    # 2026-08-23: a graph at the right file, shift and steps but the wrong
    # strength is a different arm, and three of four fields staying right is
    # exactly how the fourth drifts unnoticed.
    strengths: dict[str, float] | None = None


# The base checkpoint's own training shifts. Every graph that loads no distill
# must sit here, which is what makes "every shipped graph" true rather
# than "the two graphs that happen to load a LoRA".
BASE_SHIFT = (12.0, 3.0)

_PDD_NAME = re.compile(r"_pdd_(\d+)step_", re.IGNORECASE)


def classify_pdd(lora_name):
    return bool(_PDD_NAME.search(lora_name.replace("-", "_")) or "pdd" in lora_name.lower())


def is_turbo(lora_name):
    """Any turbo LoRA, a closed lane since 2026-09-26: lightx2v's family and
    the larryvrh pack alike match on the word."""
    return "turbo" in lora_name.lower()


def classify_pika(lora_name):
    """Pika H3 Flex 6-step LCM LoRAs (h3_config.PIKA_*). Without its own row,
    a Pika graph would fall through to the base branch and pass on the base
    12/3 shift without grading partition, 6 steps, lcm, or PIKA_MANUAL_SIGMAS."""
    return "pika_h3_flex_" in lora_name.lower()


def classify_ph3_image(doc, found: Found) -> bool:
    """PrunaAI P-H3-Image single-frame 8-step distilled graphs (`workflows/image/`).
    Recognised by either the standalone `p-h3-image_int8_convrot.safetensors`
    checkpoint, the `p-h3-image_lora_bf16.safetensors` adapter, or
    `PH3ImageConditioning`."""
    if any("p-h3-image" in l.lower() for l in found.loras):
        return True
    for n in doc.values():
        if not isinstance(n, dict):
            continue
        if n.get("class_type") == "PH3ImageConditioning":
            return True
        if "p-h3-image" in str((n.get("inputs") or {}).get("unet_name", "")).lower():
            return True
    return False


def is_pdmd(lora_name):
    """Any PDMD LoRA, a retired lane since 2026-10-05 (`docs/roadmap.md`,
    "Closed lanes"): pdmd2026's files and kijai's resizes of them."""
    return "pdmd" in lora_name.lower()


def is_flashgen(lora_name):
    """Any FlashGen LoRA, a retired lane since 2026-11-04 (`docs/roadmap.md`,
    "Closed lanes"): kijai's resize and our rank-64 conversions alike match
    on the word."""
    return "flashgen" in lora_name.lower()


def fasth3_models(doc):
    """Every model file a node of `doc` loads whose name says FastH3, a
    retired lane since 2026-11-04. Read from any `unet_name` or `ckpt_name`
    input, so a FastH3 checkpoint behind another loader class is caught too."""
    out = []
    for n in doc.values():
        if not isinstance(n, dict):
            continue
        for key in ("unet_name", "ckpt_name"):
            v = (n.get("inputs") or {}).get(key)
            if isinstance(v, str) and "fasth3" in v.lower():
                out.append(v)
    return out


# --------------------------------------------------------------------------
# graph reader -- API form, the only form this file reads
# --------------------------------------------------------------------------

def _literal(doc, value):
    """The value behind an API-form input, or None if it cannot be read.

    An input is either a literal or a `[node_id, slot]` link, and this
    returned None for every link until it was pointed at the resolver --
    coercing one with `float()` raises TypeError, which surfaces as a FAIL on a
    graph that is actually fine. That traded one wrong answer for a quieter
    one: the day anyone wires
    a `strength_model` or a `shift_video` from a constant node, every graph
    that does it reads here as ungradeable and this file goes red on correct
    state.

    So the link is followed, once, by `h3_config.resolve_link`, which is the
    only walker in the repo. None still means "could not read", and it now
    means what it says: the chain is genuinely unresolvable -- a computed
    output, a class the resolver has no row for, or a broken link. The caller's
    existing "could not read" path is unchanged for all three, deliberately;
    telling them apart is `GraphValue.reason`'s job and no case here needs it
    yet.
    """
    got = resolve_link(doc, value)
    return got.value if got.ok else None


def read_api(doc) -> Found:
    """API form: {node_id: {class_type, inputs{}}}."""
    loras: list[str] = []
    shift: tuple[float, float] | None = None
    steps: int | None = None
    scheduler: str | None = None
    shifts: list[tuple[float, float]] = []
    strengths: dict[str, float] = {}
    for node in doc.values():
        ct, inp = node.get("class_type"), node.get("inputs", {})
        if ct in LORA_LOADER_CLASSES:
            loras.append(inp.get("lora_name", ""))
            s = _literal(doc, inp.get("strength_model"))
            if s is None:
                s = _literal(doc, inp.get("strength"))
            if s is not None:
                strengths[str(inp.get("lora_name", ""))] = float(s)
        elif ct == "MiniMaxH3SigmaShift":
            sv, sa = (_literal(doc, inp.get("shift_video")),
                      _literal(doc, inp.get("shift_audio")))
            shift = None if sv is None or sa is None else (float(sv), float(sa))
            if shift is not None:
                shifts.append(shift)
    steps, scheduler = graph_schedule(doc)
    return Found(loras, shift, steps, scheduler, tuple(shifts), strengths)


def main():
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("distill shift and step pairing")

    # ---- every loader of ours is read as one --------------------------------
    def loaders_complete():
        """A node of ours with a `lora_name` input that LORA_LOADER_CLASSES
        misses reads its graph as a base graph, which every rule below then
        passes. `MiniMaxH3LoRABranch` did on 2026-09-26."""
        manifest = json.loads((HERE / "node_id_manifest.json").read_text())
        takes = sorted(n for n, e in manifest.items() if "lora_name" in e.get("inputs", []))
        missing = [n for n in takes if n not in LORA_LOADER_CLASSES]
        assert takes and not missing, (
            f"nodes with a lora_name input missing from LORA_LOADER_CLASSES: {missing}")
    check("every node of ours with a lora_name input is a known loader", loaders_complete)

    # ---- every shipped API graph -----------------------------------------
    def graphs_are_consistent():
        import h3_config as cfg
        distilled_graphs, ph3_image_graphs, base_graphs = {}, {}, {}
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            found = read_api(doc)
            retired = [l for l in found.loras if is_turbo(l)]
            assert not retired, (
                f"{path.name}: loads {retired}. Turbo LoRAs are a closed lane "
                f"(docs/roadmap.md, 'Closed lanes', 2026-09-26)")
            retired = [l for l in found.loras if is_pdmd(l)]
            assert not retired, (
                f"{path.name}: loads {retired}. PDMD is a retired lane "
                f"(docs/roadmap.md, 'Closed lanes', 2026-10-05)")
            retired = [l for l in found.loras if is_flashgen(l)] + fasth3_models(doc)
            assert not retired, (
                f"{path.name}: loads {retired}. FlashGen and FastH3 are a retired "
                f"lane (docs/roadmap.md, 'Closed lanes', 2026-11-04)")
            pdd_nodes = [n["class_type"] for n in doc.values()
                         if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3PDDLoRA"]
            retired = [l for l in found.loras if classify_pdd(l)] + pdd_nodes
            assert not retired, (
                f"{path.name}: loads {retired}. PDD is a retired lane "
                f"(docs/roadmap.md, 'Closed lanes', 2026-11-06)")
            if classify_ph3_image(doc, found):
                nodes = [n for n in doc.values() if isinstance(n, dict)]
                unets = {n["inputs"].get("unet_name") for n in nodes
                         if n.get("class_type") == "UNETLoader"}
                if found.loras:
                    assert found.loras == [cfg.PH3_IMAGE_LORA], (
                        f"{path.name}: P-H3-Image LoRA arm must load "
                        f"{[cfg.PH3_IMAGE_LORA]}, has {found.loras}")
                    assert unets == {cfg.MODELS["unet_fl2va"]}, (
                        f"{path.name}: {cfg.PH3_IMAGE_LORA} is trained on "
                        f"unet_fl2va ({cfg.MODELS['unet_fl2va']}), has {sorted(map(str, unets))}")
                    got_s = (found.strengths or {}).get(cfg.PH3_IMAGE_LORA)
                    assert got_s == cfg.PH3_IMAGE_STRENGTH, (
                        f"{path.name}: P-H3-Image LoRA strength {got_s}, "
                        f"want {cfg.PH3_IMAGE_STRENGTH}")
                else:
                    assert unets == {cfg.PH3_IMAGE_UNET}, (
                        f"{path.name}: standalone P-H3-Image arm must load "
                        f"{cfg.PH3_IMAGE_UNET}, has {sorted(map(str, unets))}")
                assert found.shift == cfg.PH3_IMAGE_SHIFT, (
                    f"{path.name}: P-H3-Image uses unshifted 8-step sigmas at "
                    f"shift {cfg.PH3_IMAGE_SHIFT}, has {found.shift}")
                assert (found.scheduler, found.steps) == ("manual", cfg.PH3_IMAGE_STEPS), (
                    f"{path.name}: P-H3-Image runs its {cfg.PH3_IMAGE_STEPS} sigmas "
                    f"through ManualSigmas; graph has {found.scheduler!r}/{found.steps}")
                manual = [n["inputs"].get("sigmas") for n in nodes
                          if n.get("class_type") == "ManualSigmas"]
                assert manual == [cfg.PH3_IMAGE_MANUAL_SIGMAS], (
                    f"{path.name}: ManualSigmas {manual}, want {[cfg.PH3_IMAGE_MANUAL_SIGMAS]}")
                main_samplers = {doc[str(n["inputs"]["sampler"][0])]["inputs"].get("sampler_name")
                                 for n in nodes if n.get("class_type") == "SamplerCustomAdvanced"}
                assert main_samplers == {cfg.PH3_IMAGE_SAMPLER}, (
                    f"{path.name}: P-H3-Image steps {cfg.PH3_IMAGE_SAMPLER}, "
                    f"graph has {sorted(map(str, main_samplers))}")
                ph3_image_graphs[path.name] = found
                continue
            pika = [l for l in found.loras if classify_pika(l)]
            if pika:
                nodes = [n for n in doc.values() if isinstance(n, dict)]
                pika_files = {
                    cfg.PIKA_FL2VA_LORA: "unet_fl2va",
                    cfg.PIKA_FL2VA_SVD995_LORA: "unet_fl2va",
                    cfg.PIKA_FL2VA_SVD99_LORA: "unet_fl2va",
                    cfg.PIKA_FL2VA_KIJAI_LORA: "unet_fl2va",
                    cfg.PIKA_REF2VA_LORA: "unet_ref2va",
                    cfg.PIKA_REF2VA_SVD995_LORA: "unet_ref2va",
                    cfg.PIKA_REF2VA_SVD99_LORA: "unet_ref2va",
                    cfg.PIKA_REF2VA_KIJAI_LORA: "unet_ref2va",
                }
                assert len(found.loras) == 1 and found.loras[0] in pika_files, (
                    f"{path.name}: Pika H3 Flex must load exactly one of "
                    f"{sorted(pika_files)} and nothing beside it, has {found.loras}")
                unets = {n["inputs"].get("unet_name") for n in nodes
                         if n.get("class_type") == "UNETLoader"}
                want_unet = cfg.MODELS[pika_files[found.loras[0]]]
                assert unets == {want_unet}, (
                    f"{path.name}: {found.loras[0]} is trained for "
                    f"{pika_files[found.loras[0]]}, so it loads on "
                    f"{want_unet} only, has {sorted(map(str, unets))}")
                effective = BASE_SHIFT if found.shift is None else found.shift
                assert effective == BASE_SHIFT, (
                    f"{path.name}: Pika H3 Flex's sigmas are shifted at 12 and "
                    f"core derives audio shift 3 from {BASE_SHIFT}, has {effective}")
                assert (found.scheduler, found.steps) == ("manual", cfg.PIKA_STEPS), (
                    f"{path.name}: Pika H3 Flex runs its own {cfg.PIKA_STEPS} sigmas "
                    f"through ManualSigmas; graph has {found.scheduler!r}/{found.steps}")
                manual = [n["inputs"].get("sigmas") for n in nodes
                          if n.get("class_type") == "ManualSigmas"]
                assert manual == [cfg.PIKA_MANUAL_SIGMAS], (
                    f"{path.name}: ManualSigmas {manual}, want {[cfg.PIKA_MANUAL_SIGMAS]}")
                main_samplers = {doc[str(n["inputs"]["sampler"][0])]["inputs"].get("sampler_name")
                                 for n in nodes if n.get("class_type") == "SamplerCustomAdvanced"
                                 and doc.get(str(n["inputs"].get("latent_image", [None])[0]), {})
                                 .get("class_type") != cfg.AUDIO_REFINE_MASK_NODE}
                assert main_samplers == {cfg.PIKA_SAMPLER}, (
                    f"{path.name}: Pika H3 Flex steps {cfg.PIKA_SAMPLER}, graph has "
                    f"{sorted(map(str, main_samplers))}")
                got = (found.strengths or {}).get(found.loras[0])
                assert got == cfg.PIKA_STRENGTH, (
                    f"{path.name}: Pika H3 Flex strength {got}, want {cfg.PIKA_STRENGTH}")
                distilled_graphs[path.name] = (found, pika)
                continue

            assert not found.loras, f"{path.name}: loads unknown LoRA {found.loras}"
            # A base graph is still policed: it must sit at the base
            # checkpoint's own training shifts. Skipping it is how
            # "every shipped graph" quietly became "the two with a LoRA".
            if found.shift is not None:
                base_graphs[path.name] = found
                assert found.shift == BASE_SHIFT, (
                    f"{path.name}: no distill, so it must sit at the base "
                    f"shift {BASE_SHIFT[0]}/{BASE_SHIFT[1]}, "
                    f"has {found.shift[0]}/{found.shift[1]}")

        # Printed before the empty-set guards, so a red on an empty walk still
        # says which population came back empty. An assertion that fires
        # first on the loop above cannot reach here; that is fine, because it
        # names the graph instead.
        print(f"        ({len(distilled_graphs)} Pika, "
              f"{len(ph3_image_graphs)} P-H3-Image, "
              f"{len(base_graphs)} base API graph(s) graded)")
        assert distilled_graphs, (
            "no shipped API graph loads a Pika LoRA; this check saw "
            "nothing, and grading an empty set is not a pass")
        assert ph3_image_graphs, (
            "no shipped API graph was recognised as P-H3-Image; this check saw "
            "nothing, and grading an empty set is not a pass")
        assert base_graphs, (
            "no shipped base API graph was examined; the base arm is unpoliced")

    check("every shipped graph sits at the right shift and steps", graphs_are_consistent)

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall ok -- every graph sits at the shift and steps its arm was distilled at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
