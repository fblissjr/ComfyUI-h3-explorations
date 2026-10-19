#!/usr/bin/env python3
"""Check every distill is loaded at the shift and step count it was trained at.

A distill runs on its trainer's contract (`docs/working/coderef.md`): PDD and
FlashGen at the base 12/3 on their own grids, FastH3 V2 at `FASTH3_SHIFT`.
None of them carries its shift in the file, so a graph at the wrong one
renders plausibly wrong and **nothing errors**.

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
                        fails: that lane is closed (`docs/state/roadmap.md`,
                        "Closed lanes", 2026-09-26). Its vendor-row grading
                        (the `LEGAL` table, `UNATTESTED`, `OWNER_RECIPE`) went
                        with it and is in git
  no retired PDMD       a graph loading a PDMD LoRA fails the same way:
                        retired 2026-10-05, and the contract grading it
                        had (one file, at the call, `simple` at the
                        trained step count) is in git
  refine undistilled    an audio refine pass reaches its UNETLoader through
                        no LoRA loader

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
    # The `nfe` OVERRIDE a PDD graph sets, or None at its default of 0.
    # Added 2026-08-26 when the node began fusing heads at load for any
    # divisor; repurposed 2026-08-27 when it began deriving the count from
    # `sample_sigmas` instead. It is no longer the evaluation count -- the
    # sampler's `steps` is -- so this exists only to catch a graph that forces
    # one partition while stepping another.
    pdd_nfe: int | None = None


# The base checkpoint's own training shifts. Every graph that loads no distill
# must sit here, which is what makes "every shipped graph" true rather
# than "the two graphs that happen to load a LoRA".
BASE_SHIFT = (12.0, 3.0)

# Parallel Decoding Distillation, converted by bench/convert_pdd_lora.py.
# Graded on its own terms: PDD carries no
# shift of its own to inherit, because its block boundaries ARE the base
# checkpoint's own schedule -- so the shift must be the BASE one, and the step
# count must be the `nfe` the converted artifact was actually fused for.
#
# That step count is read from the FILE, not from h3_config. A PDD arm at the
# wrong step count evaluates the model off the boundaries its fused heads were
# built for, and the only other thing that would notice is a runtime warning
# in a log nobody reads. Grading it against our own constant would be the
# check deriving its expectation from the thing it is checking.
_PDD_NAME = re.compile(r"_pdd_(\d+)step_", re.IGNORECASE)


def _manual_knots(doc, grid):
    """Grid indices a `ManualSigmas` vector lands on, or None.

    Uses the same `schedule_knots` the NODE uses at run time rather than
    restating the mapping -- the repo's rule about grading against the real
    expression instead of a copy of it.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pdd_math import schedule_knots
    # `doc` is an API graph, {node_id: {class_type, inputs}}.
    for n in doc.values():
        if not isinstance(n, dict) or n.get("class_type") != "ManualSigmas":
            continue
        ins = n.get("inputs")
        raw = ins.get("sigmas") if isinstance(ins, dict) else None
        if not isinstance(raw, str):
            return None
        try:
            vec = [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            return None
        import h3_config
        shift = float(h3_config.SIGMA_SHIFT["shift_video"])
        return schedule_knots(torch.tensor(vec), shift, grid)
    return None


def classify_pdd(lora_name):
    return bool(_PDD_NAME.search(lora_name.replace("-", "_")))


def lora_path(name):
    """Absolute path for a LoRA a graph names, or None.

    Prefers ComfyUI's own resolver so `extra_model_paths.yaml` is honoured,
    and falls back to the stock directory. Same shape as
    `bench/check_lora_alpha.py`'s resolver; kept local rather than imported
    because this file otherwise needs no ComfyUI at all and a shared helper
    would drag that requirement into every case here.
    """
    comfy_root = HERE.parents[2]
    try:
        sys.path.insert(0, str(comfy_root))
        import folder_paths  # noqa: WPS433
        found = folder_paths.get_full_path("loras", name)
        if found:
            return Path(found)
    except Exception:
        pass
    candidate = comfy_root / "models" / "loras" / name
    return candidate if candidate.exists() else None


def pdd_grid(lora_name):
    """`pdd_num_steps` from the converted file, or None."""
    return _pdd_meta(lora_name, "pdd_num_steps")


def pdd_block_size(lora_name):
    """`pdd_block_size` -- the width the head bank was distilled at, or None.

    Needed to grade a step count that does NOT divide the grid: those tile it
    unevenly, and whether a given uneven tiling is legal is a question about
    the trained envelope `[trained, 2*trained]` rather than about divisibility.
    """
    return _pdd_meta(lora_name, "pdd_block_size")


def pdd_nfe(lora_name):
    """`pdd_nfe` from the converted file's own metadata, or None if unreadable.

    None is not a pass: the caller fails on it. A PDD arm whose artifact cannot
    be read is one whose schedule cannot be graded, and the filename's own
    `_8step_` is not evidence -- the converter writes both, and only the
    metadata is what the node actually consumes.
    """
    return _pdd_meta(lora_name, "pdd_nfe")


def _pdd_meta(lora_name, key):
    import json as _json
    import struct as _struct
    path = lora_path(lora_name)
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            n = _struct.unpack("<Q", handle.read(8))[0]
            meta = _json.loads(handle.read(n)).get("__metadata__") or {}
        return int(meta[key])
    except Exception:
        return None


def is_turbo(lora_name):
    """Any turbo LoRA, a closed lane since 2026-09-26: lightx2v's family and
    the larryvrh pack alike match on the word."""
    return "turbo" in lora_name.lower()


def classify_flashgen(lora_name):
    """The FlashGen 4-step LoRA (h3_config.FLASHGEN_*). Before 2026-09-25 this
    file had no row for it, so a FlashGen graph fell through to the base branch
    and passed on the base shift it happens to share -- graded on nothing that
    belongs to it."""
    return "flashgen" in lora_name.lower()


def is_pdmd(lora_name):
    """Any PDMD LoRA, a retired lane since 2026-10-05 (`docs/state/roadmap.md`,
    "Closed lanes"): pdmd2026's files and kijai's resizes of them."""
    return "pdmd" in lora_name.lower()


def _flashgen_header_sigmas():
    """`manual_sigmas_shift12` from the FlashGen file's header, or None if the
    file is not on this box (a missing file is `check_model_files.py`'s)."""
    import h3_config as cfg
    path = lora_path(cfg.FLASHGEN_LORA)
    if path is None:
        return None
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return (f.metadata() or {}).get("manual_sigmas_shift12")


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
    pdd_nfe: int | None = None
    for node in doc.values():
        ct, inp = node.get("class_type"), node.get("inputs", {})
        if ct in LORA_LOADER_CLASSES:
            # The pack node is a turbo loader too. Matching only the stock
            # one made its graphs read as BASE graphs -- policed for shift,
            # which they happened to satisfy, and never graded on steps.
            # A pass for the wrong reason is what this file exists to stop.
            loras.append(inp.get("lora_name", ""))
            s = _literal(doc, inp.get("strength_model"))
            if s is None:
                s = _literal(doc, inp.get("strength"))     # MiniMaxH3PDDLoRA
            if ct == "MiniMaxH3PDDLoRA":
                pdd_nfe = _literal(doc, inp.get("nfe")) or None
            if s is not None:
                strengths[str(inp.get("lora_name", ""))] = float(s)
        elif ct == "MiniMaxH3SigmaShift":
            sv, sa = (_literal(doc, inp.get("shift_video")),
                      _literal(doc, inp.get("shift_audio")))
            shift = None if sv is None or sa is None else (float(sv), float(sa))
            if shift is not None:
                shifts.append(shift)
    # Steps and scheduler come from `h3_config.graph_schedule`, not from a
    # local `BasicScheduler` branch: since 0.83.0 a PDD graph carries no
    # scheduler node at all -- `MiniMaxH3PDDLoRA` emits SIGMAS -- and reading
    # only `BasicScheduler` reported "could not read the sampler's step count"
    # on every PDD graph, which this file treats as a failure.
    steps, scheduler = graph_schedule(doc)
    return Found(loras, shift, steps, scheduler, tuple(shifts), strengths, pdd_nfe)


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
        distilled_graphs, base_graphs = {}, {}
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            found = read_api(doc)
            retired = [l for l in found.loras if is_turbo(l)]
            assert not retired, (
                f"{path.name}: loads {retired}. Turbo LoRAs are a closed lane "
                f"(docs/state/roadmap.md, 'Closed lanes', 2026-09-26)")
            retired = [l for l in found.loras if is_pdmd(l)]
            assert not retired, (
                f"{path.name}: loads {retired}. PDMD is a retired lane "
                f"(docs/state/roadmap.md, 'Closed lanes', 2026-10-05)")
            nodes_all = [n for n in doc.values() if isinstance(n, dict)]
            unets_all = {n["inputs"].get("unet_name") for n in nodes_all
                         if n.get("class_type") == "UNETLoader"}
            if (cfg.MODELS["unet_fasth3_v2"] in unets_all
                    and set(found.loras) in ({cfg.PDD_FL2VA_LORA}, {cfg.PDD_REF2VA_LORA})
                    and any(n.get("class_type") == "DisableNoise" for n in nodes_all)):
                # PDD8 then FastH3's own checkpoint (step_switch_to="fasth3"):
                # the one LoRA is pass 1's, on fl2va. The pair and the shift
                # travel together (h3_config.STEP_SWITCH_FASTH3): the shift
                # decides the audio's noise level at the handoff.
                manual = sorted(n["inputs"]["sigmas"] for n in nodes_all
                                if n.get("class_type") == "ManualSigmas")
                declared = {tuple(sorted(p[:2])): (p[2]["shift_video"], p[2]["shift_audio"])
                            for p in cfg.STEP_SWITCH_FASTH3.values()}
                assert tuple(manual) in declared, (
                    f"{path.name}: a PDD8-then-FastH3 switch must sample one of "
                    f"h3_config.STEP_SWITCH_FASTH3, has {manual}")
                assert found.shifts == (declared[tuple(manual)],), (
                    f"{path.name}: its finish pass runs at shift {declared[tuple(manual)]}, "
                    f"graph has {found.shifts}")
                vsa = [n["inputs"] for n in nodes_all if n.get("class_type") == cfg.SOL_CORE_NODE]
                assert vsa and all(v.get("selection") == "vsa" for v in vsa), (
                    f"{path.name}: FastH3 V2 finishes on core's VSA node; graph has {vsa}")
                continue
            if cfg.MODELS["unet_fasth3_v2"] in unets_all:
                # A distilled CHECKPOINT, not a LoRA, so it is keyed on the
                # unet. Without this row it read as a base graph and failed on
                # its own 10/3 shift -- or, had it sat at 12/3, passed wrongly.
                assert not found.loras, (
                    f"{path.name}: FastH3 V2 is a full distilled model; stacking "
                    f"{found.loras} on it is outside anything its publishers ship")
                want_shift = (cfg.FASTH3_SHIFT["shift_video"], cfg.FASTH3_SHIFT["shift_audio"])
                assert found.shift == want_shift, (
                    f"{path.name}: FastH3 V2 samples at {want_shift} (its card: "
                    f"video shift 10), graph has {found.shift}")
                # Two sampling setups are legal: ComfyUI's template (8 `simple`
                # steps on res_multistep) and FastVideo's contract (the
                # release's positions through ManualSigmas, on Euler). Each is
                # graded whole; a graph mixing their halves is neither.
                samplers = {n["inputs"].get("sampler_name") for n in nodes_all
                            if n.get("class_type") == "KSamplerSelect"}
                manual = [n["inputs"].get("sigmas") for n in nodes_all
                          if n.get("class_type") == "ManualSigmas"]
                template = ((found.scheduler, found.steps) == (cfg.FASTH3_SCHEDULER, cfg.FASTH3_STEPS)
                            and samplers == {cfg.FASTH3_SAMPLER})
                contract = ((found.scheduler, found.steps) == ("manual", cfg.FASTH3_STEPS)
                            and manual == [cfg.FASTH3_CONTRACT_SIGMAS]
                            and samplers == {cfg.FASTH3_CONTRACT_SAMPLER})
                assert template or contract, (
                    f"{path.name}: FastH3 V2 runs either the template's {cfg.FASTH3_STEPS} "
                    f"{cfg.FASTH3_SCHEDULER} steps on {cfg.FASTH3_SAMPLER} or the contract's "
                    f"sigmas on {cfg.FASTH3_CONTRACT_SAMPLER}; graph has "
                    f"{found.scheduler!r}/{found.steps}, samplers {sorted(map(str, samplers))}, "
                    f"ManualSigmas {manual}")
                vsa = [n["inputs"] for n in nodes_all if n.get("class_type") == cfg.SOL_CORE_NODE]
                if path.name.removesuffix("_api.json").removesuffix("_savelat") == \
                        "h3_probe_t2v_fasth3_8step_contract_novsa":
                    # The texture probe (docs/distills/which_distill.md): VSA off by design,
                    # so it must carry NO VSA node, the inverse of the rule below.
                    assert not vsa, f"{path.name}: the VSA-off probe carries a VSA node {vsa}"
                    continue
                assert vsa and all(v.get("selection") == "vsa" for v in vsa), (
                    f"{path.name}: FastH3 V2 was trained with VSA and ships with "
                    f"core's VSA node; graph has {vsa}")
                continue
            flashgen = [l for l in found.loras if classify_flashgen(l)]
            switched = any(isinstance(n, dict) and n.get("class_type") == "DisableNoise"
                           for n in doc.values())
            if switched and set(found.loras) == {cfg.PDD_FL2VA_LORA}:
                # PDD8 then the undistilled base (step_switch_to="base",
                # open_experiments #37): one LoRA, on pass 1 only. Without this
                # it fell to the PDD rule below and failed as a truncated
                # trajectory. Its pair must be one STEP_SWITCH_BASE declares.
                manual = sorted(n["inputs"]["sigmas"] for n in doc.values()
                                if isinstance(n, dict) and n.get("class_type") == "ManualSigmas")
                assert manual in [sorted(p) for p in cfg.STEP_SWITCH_BASE.values()], (
                    f"{path.name}: a PDD8-then-base switch must sample one of "
                    f"h3_config.STEP_SWITCH_BASE, has {manual}")
                continue
            if (set(found.loras) in ({cfg.FLASHGEN_R64_LORA, cfg.PDD_FL2VA_LORA},
                                     {cfg.FLASHGEN_R64_REF2VA_LORA, cfg.PDD_REF2VA_LORA}) and switched):
                # A step switch (build_api(step_switch=True), either direction)
                # loads both on purpose, one per pass. Its settings are one of the
                # declared sigma pairs in h3_config.STEP_SWITCH_PAIRS, exactly.
                manual = sorted(n["inputs"]["sigmas"] for n in doc.values()
                                if isinstance(n, dict) and n.get("class_type") == "ManualSigmas")
                assert manual in [sorted(p) for p in cfg.STEP_SWITCH_PAIRS], (
                    f"{path.name}: a step switch must sample one of h3_config.STEP_SWITCH_PAIRS "
                    f"(route 3 or a reverse handoff), has {manual}")
                continue
            if flashgen:
                nodes = [n for n in doc.values() if isinstance(n, dict)]
                # Each file's adaln was fitted onto one pruned checkpoint's curve
                # basis, so each file loads on that checkpoint and no other.
                flashgen_files = {cfg.FLASHGEN_LORA: "unet_fl2va",
                                  cfg.FLASHGEN_R64_LORA: "unet_fl2va",
                                  cfg.FLASHGEN_R64_REF2VA_LORA: "unet_ref2va"}
                assert len(found.loras) == 1 and found.loras[0] in flashgen_files, (
                    f"{path.name}: FlashGen must load exactly one of "
                    f"{sorted(flashgen_files)} and nothing beside it, has {found.loras}")
                unets = {n["inputs"].get("unet_name") for n in nodes
                         if n.get("class_type") == "UNETLoader"}
                want_unet = cfg.MODELS[flashgen_files[found.loras[0]]]
                assert unets == {want_unet}, (
                    f"{path.name}: {found.loras[0]}'s adaln is fitted to "
                    f"{flashgen_files[found.loras[0]]}'s curve basis, so it loads on "
                    f"{want_unet} only, has {sorted(map(str, unets))}")
                effective = BASE_SHIFT if found.shift is None else found.shift
                assert effective == BASE_SHIFT, (
                    f"{path.name}: FlashGen's sigmas are its base_schedule at shift "
                    f"12, so the model samples at the base {BASE_SHIFT}, has {effective}")
                assert (found.scheduler, found.steps) == ("manual", cfg.FLASHGEN_STEPS), (
                    f"{path.name}: FlashGen runs its own {cfg.FLASHGEN_STEPS} sigmas "
                    f"through ManualSigmas; graph has {found.scheduler!r}/{found.steps}")
                # The vector is graded against the FILE, not only the constant:
                # the constant is a copy of the header, and a re-download that
                # moved the schedule should not leave the graphs on the old one.
                header = _flashgen_header_sigmas()
                manual = [n["inputs"].get("sigmas") for n in nodes
                          if n.get("class_type") == "ManualSigmas"]
                assert manual == [cfg.FLASHGEN_MANUAL_SIGMAS] and (
                        header is None or header == cfg.FLASHGEN_MANUAL_SIGMAS), (
                    f"{path.name}: ManualSigmas {manual}, constant "
                    f"{cfg.FLASHGEN_MANUAL_SIGMAS!r}, file header {header!r}")
                main_samplers = {doc[str(n["inputs"]["sampler"][0])]["inputs"].get("sampler_name")
                                 for n in nodes if n.get("class_type") == "SamplerCustomAdvanced"
                                 and doc.get(str(n["inputs"].get("latent_image", [None])[0]), {})
                                 .get("class_type") != cfg.AUDIO_REFINE_MASK_NODE}
                assert main_samplers == {cfg.FLASHGEN_SAMPLER}, (
                    f"{path.name}: FlashGen steps {cfg.FLASHGEN_SAMPLER}, graph has "
                    f"{sorted(map(str, main_samplers))}")
                got = (found.strengths or {}).get(found.loras[0])
                assert got == cfg.FLASHGEN_STRENGTH, (
                    f"{path.name}: FlashGen strength {got}, want {cfg.FLASHGEN_STRENGTH}")
                continue
            pdd = [l for l in found.loras if classify_pdd(l)]

            if not pdd:
                # A base graph is still policed: it must sit at the base
                # checkpoint's own training shifts. Skipping it is how
                # "every shipped graph" quietly became "the two with a LoRA".
                if found.shift is not None:
                    base_graphs[path.name] = found
                    assert found.shift == BASE_SHIFT, (
                        f"{path.name}: no distill, so it must sit at the base "
                        f"shift {BASE_SHIFT[0]}/{BASE_SHIFT[1]}, "
                        f"has {found.shift[0]}/{found.shift[1]}")
                continue

            distilled_graphs[path.name] = (found, pdd)
            for lora in pdd:
                # PDD: base shift, and the step count the artifact was fused
                # for.
                # `None` means the graph carries no `MiniMaxH3SigmaShift`,
                # which since 2026-08-31 is how the PDD graphs ship: at the
                # checkpoint's own 12/3 the node patched the model into
                # what it already was, so it was dropped rather than left
                # as a knob that `pdd_lora.py::check_shift` raises on. An
                # absent node therefore RUNS the base shift, and that is
                # what this compares. `BASE_SHIFT` here is a retyped copy
                # of ComfyUI's `MiniMaxH3.sampling_settings`; the check
                # that grades the absent case against the real one is
                # `check_pdd_sigmas.py::_checkpoint_default_shift`.
                effective = BASE_SHIFT if found.shift is None else found.shift
                assert effective == BASE_SHIFT, (
                    f"{path.name}: {lora} is a PDD arm, whose block "
                    f"boundaries ARE the base schedule -- it must sit at "
                    f"{BASE_SHIFT[0]}/{BASE_SHIFT[1]}, has {found.shift}")
                # **The SAMPLER's step count is the evaluation count.**
                # Since 2026-08-27 the node reads the block boundaries off
                # `sample_sigmas` at run time, so the graph's `steps` is
                # what runs and the file's `pdd_nfe` is only a fallback for
                # a sampler that publishes no schedule. This used to grade
                # `steps` against the file and went red on every correct
                # 4-step arm the moment the widget stopped carrying 4.
                grid = pdd_grid(lora)
                assert grid, (
                    f"{path.name}: could not read `pdd_num_steps` from "
                    f"{lora}. A PDD arm whose grid cannot be read cannot "
                    "be graded; the filename is not evidence.")
                assert found.steps is not None, (
                    f"{path.name}: could not read the sampler's step count, "
                    f"which IS the evaluation count since 2026-08-27. A PDD "
                    f"arm whose schedule cannot be read cannot be graded; "
                    f"the filename is not evidence.")
                # **Uniformity is not the requirement; landing on the grid
                # is.** This asserted `grid % steps == 0` until 2026-08-28,
                # which was right while every arm was uniform and wrong the
                # moment one was not. `[8,8,4,4,4,4]` is six evaluations, so
                # it fails that test, and it is on-grid at every knot --
                # every block boundary is a grid point and every width is
                # within the trained envelope. Divisibility was standing in
                # for on-grid because for a UNIFORM partition the two are
                # the same thing.
                #
                # A `manual` schedule is checked against the grid it names;
                # anything else still has to tile, because a non-uniform
                # schedule can only arrive through ManualSigmas and its
                # absence means the count came from a node that emits
                # uniform blocks.
                if found.scheduler == "manual":
                    knots = _manual_knots(doc, grid)
                    assert knots is not None, (
                        f"{path.name}: a manual schedule that cannot be "
                        f"read cannot be graded.")
                    assert knots == sorted(set(knots)), (
                        f"{path.name}: manual sigmas do not ascend through "
                        f"the grid: {knots}.")
                    widths = [b - a for a, b in zip(knots, knots[1:])]
                    assert knots[0] == 0 and knots[-1] == grid, (
                        f"{path.name}: manual sigmas span {knots[0]}.."
                        f"{knots[-1]} of a {grid}-point grid; a shipped arm "
                        f"runs the whole trajectory.")
                    assert all(w <= 8 for w in widths), (
                        f"{path.name}: block widths {widths} exceed the "
                        f"trained envelope (L_max 8 under the inferred "
                        f"4/8). Legal to render, not to ship.")
                else:
                    assert grid % found.steps == 0, (
                        f"{path.name}: {found.steps} evaluations do not "
                        f"divide the file's {grid}-point grid, and this arm "
                        f"has no ManualSigmas to name an explicit partition, "
                        f"so its blocks come out uneven. The node takes them "
                        f"anyway and says so, but a SHIPPED arm should tile: "
                        f"{sorted(n for n in range(1, grid + 1) if grid % n == 0)}.")
                # `nfe` is an override that forces uniform blocks and
                # ignores the schedule. Legal, but it means the arm decodes
                # one partition while stepping another -- an experiment, not
                # something to ship. Every shipped graph carries 0.
                assert not found.pdd_nfe or found.pdd_nfe == found.steps, (
                    f"{path.name}: the node's `nfe` override is "
                    f"{found.pdd_nfe} while the sampler runs {found.steps} "
                    f"steps. That decodes the blocks of one partition while "
                    f"stepping another. Leave `nfe` at 0 so it follows the "
                    f"schedule, or change the sampler's steps to match.")

        # Printed before the empty-set guards, so a red on an empty walk still
        # says which population came back empty. An assertion that fires
        # first on the loop above cannot reach here; that is fine, because it
        # names the graph instead.
        print(f"        ({len(distilled_graphs)} PDD, "
              f"{len(base_graphs)} base API graph(s) graded)")
        assert distilled_graphs, (
            "no shipped API graph loads a PDD LoRA; this check saw "
            "nothing, and grading an empty set is not a pass")
        assert base_graphs, (
            "no shipped base API graph was examined; the base arm is unpoliced")

    check("every shipped graph sits at the right shift and steps", graphs_are_consistent)

    def refine_passes_run_undistilled():
        """An audio-only refine pass exists to run UNDISTILLED steps: its model
        must reach the UNETLoader with no LoRA loader on the way."""
        import h3_config as cfg
        seen = 0
        for path in graph_paths(WORKFLOWS, "*_api.json", include_bench=True):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for n in doc.values():
                if not (isinstance(n, dict) and n.get("class_type") == "SamplerCustomAdvanced"):
                    continue
                lat = n["inputs"].get("latent_image")
                if not (isinstance(lat, list) and doc.get(str(lat[0]), {}).get("class_type")
                        == cfg.AUDIO_REFINE_MASK_NODE):
                    continue
                seen += 1
                guider = doc[str(n["inputs"]["guider"][0])]
                ref = guider["inputs"]["model"]
                chain = []
                while isinstance(ref, list):
                    node = doc[str(ref[0])]
                    chain.append(node["class_type"])
                    ref = node["inputs"].get("model")
                assert chain and chain[-1] == "UNETLoader", (
                    f"{path.name}: the refine pass's model chain {chain} does not end at a UNETLoader")
                bad = [c for c in chain if c in LORA_LOADER_CLASSES]
                assert not bad, (
                    f"{path.name}: the refine pass runs through {bad}, so its "
                    f"steps are distilled -- the pass exists to be undistilled")
        assert seen, "no refine pass found; the graphs that carry one would go ungraded"

    check("an audio refine pass runs undistilled", refine_passes_run_undistilled)

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall ok -- every graph sits at the shift and steps its arm was distilled at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
