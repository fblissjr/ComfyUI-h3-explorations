#!/usr/bin/env python3
"""Grade the W4A16 GPTQ recipe's boundary and its calibration loop, on the CPU.

`bench/h3_gptq_recipe.py` is a proposal until something drives it. This runs the
pinned `llm-compressor` session against the reduced-width Qwen3-VL from
`bench/check_calibration_precision_policy.py::tiny_full_model` and the tiny
native batch beside it, and asserts what the recipe claims -- by letting the
library apply the config and run the algorithm, never by reimplementing either.

**GPTQ's whole loop runs here, not just its construction.** Unlike the AWQ
boundary probe, which stops at config application, this drives
`calibration_start` -> forward -> `sequential_epoch_end` -> `calibration_end`
through `LifecycleCallbacks`, which is the same event sequence
`SequentialPipeline` emits. So the Hessian allocation, the per-layer free, the
quantize step and the rewritten weight are all observed rather than inferred.
That matters for the budget claim in the recipe's docstring: the per-layer free
is what makes GPTQ's state independent of the population size, and a docstring
is not evidence of it.

The arms:

1. **Construction.** Both methods build in the pinned environment; `awq_gptq` is
   `[AWQModifier, GPTQModifier]` in that order, which is the order the lifecycle
   iterates; a deprecated `actorder` and an unknown method are refused.
2. **Boundary and attachment.** With the real recipe applied, exactly the text
   decoder projections carry a weight scheme (the imported
   `assert_decoder_only_boundary`), and the GPTQ modifier's own resolved
   attachment set -- `_module_names`, built from `match_named_modules` in
   `on_initialize` -- is that same set and nothing else, seven per decoder layer.
   Both are asserted, because the boundary answers "what carries a scheme" and
   the attachment answers "what will GPTQ actually build a Hessian for", and a
   recipe can get one right while getting the other wrong.
3. **Calibration.** One forward accumulates one FP32 square per targeted Linear,
   sized in that Linear's input width; the epoch end quantizes and frees every
   one of them; the weight changes and a group scale appears.
4. **Budget method.** `gptq_hessian_budget` is graded against the Hessians the
   modifier actually allocated, so the arithmetic in the recipe's docstring is
   checked as a *method* even though the released figures are unmeasured.
5. **Three mutations, each required to go red.** A widened ignore list (a
   decoder projection protected) must be caught; a narrowed one that forgets the
   DeepStack mergers must be caught -- that is the defect the boundary assertion
   was written for, and this recipe reuses the AWQ list rather than copying it,
   so the control has to be re-run here to say the reuse is intact; and a
   `targets` list missing one decoder leaf must be caught by the boundary *and*
   show up as six attachments per layer instead of seven.
6. **Mixed precision.** A W8-on-`down_proj` override must produce two schemes
   over the same population -- and `assert_decoder_only_boundary` must *refuse*
   it, because that assertion requires one scheme to describe the artifact. The
   refusal is asserted rather than discovered later: a mixed-precision variant
   needs its own boundary assertion before it can ship.

**No CUDA.** The visible device list is emptied before torch is imported. The
library's own `CompressionLogger` reads accelerator properties during the
quantize step, which would initialise a CUDA context on a card another job owns;
emptying the list makes "CPU only" a property of the run rather than a hope.

**The calibration arm runs at a group size the tiny widths divide.** The reduced
model's hidden size is narrower than the released group size, so the real recipe
warns there and the quantize step cannot form groups at all. The boundary and
attachment arms therefore run at the real group size -- what they grade does not
depend on it -- and the calibration arm at one that divides. That substitution is
recorded in the output rather than hidden, and it is the reason the card probe
below is still required.

**What this cannot prove, and the card probe must.** Everything that scales:
Hessian residency and device at 25,600 columns rather than 128, the quantize
step's time and transient at that width, whether any layer fell back to
round-to-nearest on a failed Cholesky, whether `offload_hessians` is needed to
fit the card, and the host peak against the prediction that GPTQ's only
token-scaled cost is the pipeline's own intermediates cache. That probe is
`bench/pilot_sequential_feasibility.py --layers 2 --prefix 8 --modifier gptq
--offload auto_offload`, on the GPU, on the calibration bundle.

Run it with the pinned `llm-compressor` virtualenv. No model, no checkpoint, no
server, no output file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Before torch. See the CUDA note above; also keep the hub offline, so a config
# lookup can never become a download.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from h3_awq_recipe import BoundaryViolation  # noqa: E402
from h3_gptq_recipe import (  # noqa: E402
    TEXT_DECODER_LINEAR_LEAVES,
    assert_decoder_only_boundary,
    build_recipe,
    describe_recipe,
    gptq_hessian_budget,
    ignore_patterns,
)

# The reduced model's widths are narrower than the released group size. The
# quantize step needs a size that divides them; the boundary does not care.
CALIBRATION_GROUP_SIZE = 32


def tiny_model():
    from check_calibration_precision_policy import tiny_full_model

    return tiny_full_model()


def tiny_input() -> dict:
    from check_calibration_precision_policy import tiny_batch

    return tiny_batch()


def decoder_projection_names(model) -> set[str]:
    """The population the recipe is allowed to touch, read off the model.

    Derived from the model's own module tree rather than from the recipe, so a
    recipe that quantizes the wrong set cannot also define what the right set
    was.
    """
    layers = [
        name
        for name, module in model.named_modules()
        if type(module).__name__ == "Qwen3VLTextDecoderLayer"
    ]
    return {f"{layer}.{leaf}" for layer in layers for leaf in TEXT_DECODER_LINEAR_LEAVES}


def gptq_modifier(session):
    modifiers = session.lifecycle.recipe.modifiers
    return next(m for m in modifiers if type(m).__name__ == "GPTQModifier")


def check_construction(failures: list[str]) -> None:
    recipe = build_recipe("gptq")
    names = [type(m).__name__ for m in recipe]
    if names != ["GPTQModifier"]:
        failures.append(f"construction: `gptq` built {names}")
    composed = [type(m).__name__ for m in build_recipe("awq_gptq")]
    if composed != ["AWQModifier", "GPTQModifier"]:
        failures.append(
            f"construction: `awq_gptq` built {composed}, not AWQ then GPTQ. The "
            "lifecycle iterates the recipe list in order, so this order is the "
            "claim that AWQ smooths before GPTQ quantizes"
        )
    modifier = recipe[0]
    declared = {
        "scheme": modifier.scheme,
        "targets": list(modifier.targets),
        "ignore": list(modifier.ignore),
        "block_size": modifier.block_size,
        "dampening_frac": modifier.dampening_frac,
        "actorder": str(getattr(modifier.actorder, "value", modifier.actorder)),
        "offload_hessians": modifier.offload_hessians,
    }
    if declared["ignore"] != ignore_patterns():
        failures.append(
            "construction: the GPTQ modifier does not carry the imported "
            f"boundary patterns: {declared['ignore']}"
        )
    weights = next(iter(modifier.resolved_config.config_groups.values())).weights
    if (weights.num_bits, weights.group_size) != (4, 128):
        failures.append(
            f"construction: the resolved scheme is {weights.num_bits} bits at "
            f"group {weights.group_size}, not W4A16 at group 128"
        )
    if str(getattr(weights.actorder, "value", weights.actorder)) != "static":
        failures.append(
            "construction: the resolved scheme's actorder is "
            f"{weights.actorder!r}; the recipe sets it explicitly so the emitted "
            "config is not a function of which preset is in force"
        )
    print(f"  gptq fields {declared}")
    print(f"  awq_gptq    {composed}")

    for bad, why in (
        (dict(method="sgptq"), "an unknown method"),
        (dict(method="gptq", actorder="group"), "a deprecated actorder"),
    ):
        try:
            build_recipe(**bad)
            failures.append(f"construction: {why} was accepted: {bad}")
        except ValueError:
            pass
    # And the refusal is not unconditional.
    build_recipe("gptq", actorder=None)


def check_boundary_and_attachment(failures: list[str]) -> dict:
    from llmcompressor.core.session_functions import create_session

    model = tiny_model()
    expected = decoder_projection_names(model)
    with create_session() as session:
        session.initialize(model=model, start=-1, recipe=build_recipe("gptq"))
        modifier = gptq_modifier(session)
        record = assert_decoder_only_boundary(model)
        attached = set(modifier._module_names.values())
        session.finalize()

    unexpected = sorted(attached - expected)
    missing = sorted(expected - attached)
    if unexpected or missing:
        failures.append(
            "attachment: the GPTQ modifier resolved onto the wrong modules. "
            f"Outside the decoder projections: {unexpected[:6]}. Decoder "
            f"projections it will build no Hessian for: {missing[:6]}"
        )
    per_layer = record["linears_per_decoder_layer"]
    if len(attached) != record["text_decoder_layers"] * per_layer:
        failures.append(
            f"attachment: {len(attached)} attachments for "
            f"{record['text_decoder_layers']} decoder layers at {per_layer} "
            "projections each"
        )
    print(f"  boundary {record['linear_counts']}")
    print(f"  attached {len(attached)} = {record['text_decoder_layers']} layers "
          f"x {per_layer}")
    return record


def check_calibration(failures: list[str]) -> dict:
    """Drive the whole modifier loop and read the Hessians while they exist."""
    from llmcompressor.core.session_functions import callbacks, create_session

    model = tiny_model()
    batch = tiny_input()
    probe = model.model.language_model.layers[0].self_attn.q_proj
    before = probe.weight.detach().clone()

    with create_session() as session:
        session.initialize(
            model=model, start=-1,
            recipe=build_recipe("gptq", group_size=CALIBRATION_GROUP_SIZE),
        )
        modifier = gptq_modifier(session)
        callbacks.calibration_start()
        with torch.no_grad():
            model(**batch, use_cache=False)
        census = {}
        for module, hessian in modifier._hessians.items():
            census[modifier._module_names[module]] = {
                "shape": tuple(hessian.shape),
                "in_features": int(module.weight.shape[1]),
                "dtype": str(hessian.dtype),
                "device": str(hessian.device),
                "bytes": hessian.numel() * hessian.element_size(),
            }
        modules = list(modifier._num_samples.keys())
        callbacks.sequential_epoch_end(modules=modules)
        left = len(modifier._hessians)
        samples_left = len(modifier._num_samples)
        callbacks.calibration_end()
        session.finalize()

    text_config = model.config.text_config
    if not census:
        failures.append(
            "calibration: no Hessian was allocated, so the forward never reached "
            "a GPTQ hook and every assertion below is vacuous"
        )
    for name, entry in census.items():
        if entry["shape"] != (entry["in_features"], entry["in_features"]):
            failures.append(
                f"calibration: {name}'s Hessian is {entry['shape']} for input "
                f"width {entry['in_features']}"
            )
        if entry["dtype"] != "torch.float32":
            failures.append(f"calibration: {name}'s Hessian is {entry['dtype']}")
        if entry["device"] != "cpu":
            failures.append(
                f"calibration: {name}'s Hessian is on {entry['device']}; this "
                "check runs with no visible accelerator"
            )
    if left or samples_left:
        failures.append(
            f"calibration: {left} Hessians and {samples_left} sample counters "
            "survived the sequential epoch end. GPTQ's state is only independent "
            "of the population size because they are freed per layer"
        )
    after = probe.weight.detach()
    if torch.equal(before, after):
        failures.append(
            "calibration: the probed weight is unchanged, so the quantize step "
            "did not reach it and the loop above proved nothing"
        )
    if not hasattr(probe, "weight_scale"):
        failures.append("calibration: no group scale was attached to the probe")
    print(f"  {len(census)} Hessians, all FP32 on cpu, freed at the epoch end: "
          f"{left} left")
    print(f"  probed weight moved by "
          f"{float((before - after).abs().max()):.4g}, scale "
          f"{tuple(probe.weight_scale.shape)}")
    return census, text_config


def check_budget_method(failures: list[str], census: dict, text_config,
                        boundary: dict) -> None:
    """Grade the budget arithmetic against the Hessians that were allocated.

    This grades the *method*. The released-shape figures the recipe's docstring
    derives from the same function are INFERENCE and stay so until the card
    probe; what is checkable here is that the function's widths and its per-layer
    total are the ones a real model produced.
    """
    budget = gptq_hessian_budget(text_config)
    layers = boundary["text_decoder_layers"]

    predicted = budget["hessian_bytes_per_decoder_layer"] * layers
    observed = sum(entry["bytes"] for entry in census.values())
    if predicted != observed:
        failures.append(
            f"budget: gptq_hessian_budget predicts {predicted} bytes over "
            f"{layers} layers, the modifier allocated {observed}"
        )
    for leaf, entry in budget["per_leaf"].items():
        widths = {
            record["in_features"]
            for name, record in census.items() if name.endswith("." + leaf)
        }
        if widths and widths != {entry["input_width"]}:
            failures.append(
                f"budget: {leaf} is predicted at input width "
                f"{entry['input_width']}, the model has {sorted(widths)}"
            )
    if budget["scales_with_population_tokens"]:
        failures.append("budget: the record claims GPTQ state scales with tokens")
    print(f"  predicted {predicted} B over {layers} layers, allocated {observed} B")
    print(f"  widest leaf {budget['widest_leaf']} at input width "
          f"{budget['per_leaf'][budget['widest_leaf']]['input_width']}; the same "
          "function at the released widths is the recipe docstring's budget, "
          "which no arm here measures")


def _apply_and_grade(recipe) -> tuple[str | None, set[str]]:
    """Apply a recipe to a fresh model; return the boundary's complaint, if any."""
    from llmcompressor.core.session_functions import create_session

    model = tiny_model()
    with create_session() as session:
        session.initialize(model=model, start=-1, recipe=recipe)
        attached = set(gptq_modifier(session)._module_names.values())
        try:
            assert_decoder_only_boundary(model)
            message = None
        except BoundaryViolation as exc:
            message = " ".join(str(exc).split())[:300]
        session.finalize()
    return message, attached


def check_mutations(failures: list[str], boundary: dict) -> list[dict]:
    """Three deliberate defects, each required to be caught and to name itself."""
    layers = boundary["text_decoder_layers"]
    per_layer = boundary["linears_per_decoder_layer"]

    widened = ignore_patterns() + [r"re:.*\.mlp\.down_proj$"]
    narrowed = ["lm_head", r"re:model\.visual\.blocks\..*", r"re:model\.visual\.merger\..*"]
    narrowed_targets = [
        r"re:.*\.self_attn\.(q|k|v|o)_proj$",
        r"re:.*\.mlp\.(gate|up)_proj$",
    ]

    cases = [
        {
            "control": "ignore list widened over a decoder projection",
            "recipe": build_recipe("gptq", ignore=widened),
            "expect_attached": layers * (per_layer - 1),
            "must_name": "down_proj",
        },
        {
            "control": "ignore list narrowed: the DeepStack mergers forgotten",
            "recipe": build_recipe("gptq", ignore=narrowed),
            "expect_attached": None,
            "must_name": "deepstack_merger_list",
        },
        {
            "control": "targets missing one decoder leaf",
            "recipe": build_recipe("gptq", targets=narrowed_targets),
            "expect_attached": layers * (per_layer - 1),
            "must_name": "down_proj",
        },
    ]

    outcomes = []
    for case in cases:
        message, attached = _apply_and_grade(case["recipe"])
        fired = message is not None
        named = bool(message) and case["must_name"] in message
        if not fired:
            failures.append(
                f"red control: `{case['control']}` passed the boundary assertion, "
                "so the assertion is blind to the defect it exists for"
            )
        elif not named:
            failures.append(
                f"red control: `{case['control']}` was caught but the message does "
                f"not name `{case['must_name']}`, so it would not lead a reader to "
                f"the defect: {message}"
            )
        if case["expect_attached"] is not None and len(attached) != case["expect_attached"]:
            failures.append(
                f"red control: `{case['control']}` left {len(attached)} GPTQ "
                f"attachments, expected {case['expect_attached']}"
            )
        outcomes.append({
            "control": case["control"], "fired": fired,
            "attachments": len(attached), "message": message,
        })
        print(f"  {case['control']}: fired={fired} attachments={len(attached)}")
    return outcomes


def check_mixed_precision(failures: list[str], boundary: dict) -> None:
    """The per-layer override, and the boundary assertion that must refuse it."""
    from llmcompressor.core.session_functions import create_session

    overrides = {r"re:.*\.mlp\.down_proj$": "W8A16"}
    model = tiny_model()
    expected = decoder_projection_names(model)
    with create_session() as session:
        session.initialize(
            model=model, start=-1,
            recipe=build_recipe("gptq", overrides=overrides),
        )
        bits = {}
        for name, module in model.named_modules():
            scheme = getattr(module, "quantization_scheme", None)
            weights = None if scheme is None else scheme.weights
            if weights is not None:
                bits[name] = weights.num_bits
        try:
            assert_decoder_only_boundary(model)
            refused = None
        except BoundaryViolation as exc:
            refused = " ".join(str(exc).split())[:200]
        session.finalize()

    if set(bits) != expected:
        failures.append(
            "mixed precision: the override changed the quantized population; "
            f"unexpected {sorted(set(bits) - expected)[:4]}, missing "
            f"{sorted(expected - set(bits))[:4]}"
        )
    eight = {name for name, n in bits.items() if n == 8}
    four = {name for name, n in bits.items() if n == 4}
    if eight != {name for name in expected if name.endswith(".mlp.down_proj")}:
        failures.append(
            f"mixed precision: the 8-bit population is {sorted(eight)[:4]}, not "
            "the down_proj set. A `re:` override must outrank the base group's "
            "`Linear` class target under match_targets' ordering"
        )
    if four != expected - eight:
        failures.append("mixed precision: the remaining population is not 4-bit")
    if refused is None:
        failures.append(
            "mixed precision: assert_decoder_only_boundary accepted two schemes "
            "over one population. It requires a single scheme so that one set of "
            "fields describes the artifact; if that requirement has gone, the "
            "recipe's docstring and any mixed-precision variant are both wrong"
        )
    print(f"  {len(eight)} at 8 bits, {len(four)} at 4; boundary refused the mix: "
          f"{refused is not None}")
    # The library logs `Could not match ...` for the override pattern while
    # applying the config, and the population above shows it matched anyway.
    # Read the census, not the warning.
    print("  (the library's `Could not match` line above is not the answer; the "
          "8-bit population is)")


def main() -> int:
    failures: list[str] = []

    if torch.cuda.is_available():
        failures.append(
            "this check runs with no visible accelerator by construction, and "
            "torch still reports CUDA available"
        )

    print("construction:")
    check_construction(failures)

    print("boundary and attachment:")
    boundary = check_boundary_and_attachment(failures)

    print(f"calibration (group size {CALIBRATION_GROUP_SIZE}, the reduced widths "
          "do not divide the released one):")
    census, text_config = check_calibration(failures)

    print("budget method:")
    check_budget_method(failures, census, text_config, boundary)

    print("red controls:")
    check_mutations(failures, boundary)

    print("mixed precision:")
    check_mixed_precision(failures, boundary)

    print("recipe as recorded:",
          " + ".join(entry["class"] for entry in describe_recipe(build_recipe("awq_gptq"))))

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print(
        "GREEN: both methods construct in the pinned environment, GPTQ attaches "
        "to exactly the text decoder projections and quantizes them, its "
        "Hessians are one FP32 square per input width and are freed at the "
        "sequential epoch end, the budget arithmetic matches what was allocated, "
        "all three mutations were caught by name, and the mixed-precision "
        "override lands on down_proj alone while the single-scheme boundary "
        "assertion refuses it. Everything that scales -- Hessian residency at "
        "25,600 columns, the quantize transient, a failed Cholesky, the host "
        "peak -- is still the card probe's: "
        "pilot_sequential_feasibility.py --layers 2 --prefix 8 --modifier gptq "
        "--offload auto_offload"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
