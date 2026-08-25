#!/usr/bin/env python3
"""The W4A16 AWQ v2 candidate recipe, and the boundary it is allowed to touch.

The deployed artifact quantizes the Qwen3-VL text decoder and nothing else: the
vision tower, the patch merger, the DeepStack mergers and the output head stay
at released precision. That boundary is the whole point of the checkpoint -- the
H3 conditioning path reads the vision side -- and in the previous launcher it
was expressed as a hand-written `re:.*visual.*` in a recipe nobody could grade
until the 32B model had already loaded.

This module separates the two halves so the boundary can be graded first:

- `build_recipe` constructs the modifier list. Construction is the first gate:
  the rejected preflight recipe (`docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_awq_v2_preflight_review.md`,
  finding 1) died here, on a nonexistent `AWQModifier` field and on supplying
  both `scheme` and `config_groups`, and it died only because someone
  constructed it by hand. Nothing in this module swallows a construction error.
- `assert_decoder_only_boundary` grades a model the quantization config has
  already been applied to. It answers the question the recipe cannot: which
  modules actually came out carrying a weight quantization scheme.

**The ignore patterns are a proposal; the assertion is the control.** A pattern
list is only ever correct against the module names of the implementation it was
written for, and this repo has already been bitten once by branching on a name
that a later version changed. So the patterns here are deliberately short and
the boundary check does not consult them to decide what is correct: it recovers
the vision tower, the decoder layers, the output head and the input embedding
from the model itself -- by module class and by the model's own
`get_input_embeddings` / `get_output_embeddings` -- and then asserts that the
set of modules carrying a weight scheme is exactly the text decoder projections.
If a future transformers renames `model.visual`, the patterns stop matching, the
vision tower is quantized, and the assertion goes red naming the modules. That
is the failure mode worth catching; a broader regex would only have hidden it.

**Why the input embedding needs no ignore entry.** The scheme targets
`["Linear"]`, and the embedding is an `nn.Embedding`, so it is never a target
and an ignore entry for it can only ever be inert. The previous launcher
carried `re:.*embed_tokens` and three norm patterns for the same reason it
carried `lm_head` -- caution -- and the evidence that they were inert is the
deployed artifact's own resolved ignore list, which contains neither the
embedding nor any norm: compressed-tensors writes that list from the modules
that were skipped *among the types it targeted*, so a non-Linear cannot appear
in it. `lm_head` is different and does need its entry: the released config sets
`tie_word_embeddings=False`, so the output head is a real `nn.Linear` and would
otherwise be quantized. The boundary check asserts both -- that the embedding
carries no scheme and that the head carries none -- rather than trusting this
paragraph.

Importable from the pinned `llm-compressor` virtualenv: `torch`,
`llmcompressor` and `compressed_tensors` only, no ComfyUI, no transformers
import at module scope, and no model load. Nothing here reads or writes a
checkpoint.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import torch

__all__ = [
    "BoundaryViolation",
    "IGNORE_PATTERNS",
    "TEXT_DECODER_LAYER_CLASSES",
    "TEXT_DECODER_LINEAR_LEAVES",
    "VISION_TOWER_CLASSES",
    "assert_decoder_only_boundary",
    "build_recipe",
    "describe_recipe",
    "ignore_patterns",
    "resolved_awq_mappings",
]


# Every Linear outside the text decoder stack, by module-name pattern. Anchored
# at the start because compressed-tensors matches an `re:` target with
# `re.match`, not `re.fullmatch` -- a trailing `.*` is load-bearing and a
# leading one would be a silent widening.
IGNORE_PATTERNS: tuple[str, ...] = (
    "lm_head",
    r"re:model\.visual\..*",
)

# The class the real run names in `sequential_targets`. Recovering the decoder
# layers by class rather than by the attribute path `model.language_model.layers`
# keeps the boundary check keyed to the same identity the pipeline uses.
TEXT_DECODER_LAYER_CLASSES: tuple[str, ...] = ("Qwen3VLTextDecoderLayer",)
VISION_TOWER_CLASSES: tuple[str, ...] = ("Qwen3VLVisionModel",)

# The projections inside one decoder layer, by leaf path relative to the layer.
# This is the population the candidate is allowed to quantize.
TEXT_DECODER_LINEAR_LEAVES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_VISION_BLOCK = re.compile(r"^blocks\.\d+\.")
_VISION_DEEPSTACK = re.compile(r"^deepstack_merger_list\.\d+\.")
_VISION_MERGER = re.compile(r"^merger\.")


class BoundaryViolation(AssertionError):
    """The applied quantization config reached outside the text decoder."""


def ignore_patterns() -> list[str]:
    """The candidate's ignore list, as a fresh mutable copy.

    Returned as a copy because `QuantizationModifier` stores the list it is
    given and a caller mutating it afterwards would edit the recipe.
    """
    return list(IGNORE_PATTERNS)


def build_recipe(
    *,
    offload_device: str | torch.device = "cpu",
    duo_scaling: bool | str = False,
    n_grid: int = 20,
    ignore: list[str] | None = None,
    scheme: str = "W4A16",
    targets: list[str] | None = None,
) -> list:
    """The v2 candidate modifier list, fully constructed.

    Constructing this is the cheapest gate the launcher has, and it is the one
    the rejected preflight failed: both of its modifier definitions raised
    before `oneshot` could reach a model. Every argument is therefore passed
    through to the pinned classes untouched, and no exception is caught here --
    a recipe that cannot be built must stop the caller, not degrade into a
    default.

    `offload_device` is the only offload knob the pinned `AWQModifier` exposes:
    it is where the cached parent-module forward arguments, from which the
    smoothing statistics and the grid search are computed, are held between the
    activation hook and the sequential epoch end. Left unset it is a sentinel
    that resolves to CPU only when the model is detected as MoE, and to no
    offloading otherwise -- so on this dense model the default is to keep the
    activation cache on the execution device. It is set explicitly here because
    the run is memory-bound, and explicitly rather than by relying on that
    detection because "the default happened to be right" is not a decision
    anybody can review later.

    `duo_scaling=False` restricts the grid search to activation statistics,
    matching the arm the deployed artifact was produced under.

    :param ignore: override the candidate ignore patterns. Present so a red
        control can build a deliberately incomplete recipe and watch
        `assert_decoder_only_boundary` catch it; leave it None for the candidate.
    :return: `[AWQModifier, QuantizationModifier]`, in the order the pipeline
        must see them -- AWQ smooths before quantization observes.
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    return [
        AWQModifier(
            offload_device=torch.device(offload_device),
            duo_scaling=duo_scaling,
            n_grid=n_grid,
        ),
        QuantizationModifier(
            scheme=scheme,
            targets=list(targets) if targets is not None else ["Linear"],
            ignore=ignore_patterns() if ignore is None else list(ignore),
        ),
    ]


def _jsonable(value: Any) -> Any:
    """Anything a modifier field can hold, reduced to JSON.

    Falls through to `repr` rather than raising: a provenance record that
    refuses to be written because one field held an unexpected type is worse
    than a record that says what the field looked like.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.nn.Module):
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump())
    return repr(value)


def describe_recipe(recipe: list) -> list[dict]:
    """A JSON-serialisable description of a constructed recipe.

    Every declared field of every modifier, not a curated subset: the point of
    the record is that a later reader can tell which knobs were at their
    defaults, and a curated subset silently answers "the ones I thought
    mattered". The resolved quantization config is included where the modifier
    has one, because that -- not the `scheme` string -- is what is applied.
    """
    described = []
    for modifier in recipe:
        fields = type(modifier).model_fields
        entry: dict[str, Any] = {
            "class": type(modifier).__name__,
            "module": type(modifier).__module__,
            "fields": {
                name: _jsonable(getattr(modifier, name, None)) for name in fields
            },
        }
        if hasattr(modifier, "resolved_config"):
            entry["resolved_quantization_config"] = _jsonable(modifier.resolved_config)
        described.append(entry)
    return described


def resolved_awq_mappings(modifier, model) -> list[dict]:
    """The AWQ modifier's own smooth-layer to balance-layer resolution.

    The modifier resolves its mappings inside `on_calibration_start`, which also
    installs the activation-cache hooks. This calls the resolver directly so a
    record can be taken without hooks, without calibration data and without a
    forward pass; it is the modifier's own code, not a reimplementation of the
    matching rules.

    Mappings are inferred from the model architecture during the modifier's
    `on_initialize`, so the session must have been initialized first. Raises if
    it was not, because an empty result would otherwise read as "this model has
    no mappings", which is a very different claim.
    """
    if getattr(modifier, "mappings", None) is None:
        raise RuntimeError(
            "the AWQ modifier has no mappings yet; initialize the session with "
            "this recipe before asking for the resolved mappings"
        )
    if not modifier._resolved_mappings:
        modifier._set_resolved_mappings(model)
    return [
        {
            "smooth_layer": mapping.smooth_name,
            "balance_layers": list(mapping.balance_names),
            "parent": mapping.parent_name,
            "activation_hook_target": mapping.activation_hook_target,
        }
        for mapping in modifier._resolved_mappings
    ]


def _named_modules_of_class(model, class_names: tuple[str, ...]) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ in class_names
    ]


def _classify_linears(model) -> dict[str, list[str]]:
    """Every `nn.Linear` in the model, bucketed by where it lives.

    Structure is recovered from the model, not from name patterns: the decoder
    layers by the class the pipeline sequences on, the vision tower by its class,
    the output head by `get_output_embeddings`. Names are then used only
    *within* a recovered subtree, where they are that subtree's own layout.

    A Linear that lands in no bucket is reported under `unclassified` rather
    than quietly ignored -- it means the architecture moved, and the ignore
    patterns cannot be assumed to still cover it.
    """
    decoder_layers = _named_modules_of_class(model, TEXT_DECODER_LAYER_CLASSES)
    towers = _named_modules_of_class(model, VISION_TOWER_CLASSES)
    if not decoder_layers:
        raise BoundaryViolation(
            f"no module of class {TEXT_DECODER_LAYER_CLASSES} was found; the text "
            "decoder cannot be located, so nothing about the boundary can be "
            "asserted. If the class was renamed upstream, update "
            "TEXT_DECODER_LAYER_CLASSES and the run's sequential_targets together"
        )
    if len(towers) != 1:
        raise BoundaryViolation(
            f"expected exactly one module of class {VISION_TOWER_CLASSES}, found "
            f"{[name for name, _ in towers]}"
        )
    tower_name = towers[0][0]

    head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()

    buckets: dict[str, list[str]] = {
        "text_decoder": [],
        "vision_block": [],
        "vision_merger": [],
        "vision_deepstack_merger": [],
        "vision_other": [],
        "lm_head": [],
        "unclassified": [],
    }
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if head is not None and module is head:
            buckets["lm_head"].append(name)
            continue
        if embedding is not None and module is embedding:
            # Only reachable if a future architecture makes the input embedding
            # a Linear. Recorded rather than assumed away.
            buckets["unclassified"].append(name)
            continue
        if name == tower_name or name.startswith(tower_name + "."):
            inside = name[len(tower_name) :].lstrip(".")
            if _VISION_BLOCK.match(inside):
                buckets["vision_block"].append(name)
            elif _VISION_DEEPSTACK.match(inside):
                buckets["vision_deepstack_merger"].append(name)
            elif _VISION_MERGER.match(inside):
                buckets["vision_merger"].append(name)
            else:
                buckets["vision_other"].append(name)
            continue
        for layer_name, _ in decoder_layers:
            if name.startswith(layer_name + "."):
                leaf = name[len(layer_name) + 1 :]
                if leaf in TEXT_DECODER_LINEAR_LEAVES:
                    buckets["text_decoder"].append(name)
                else:
                    buckets["unclassified"].append(name)
                break
        else:
            buckets["unclassified"].append(name)
    buckets["_decoder_layer_names"] = [name for name, _ in decoder_layers]
    buckets["_vision_tower_name"] = [tower_name]
    return buckets


def _weight_scheme(module):
    scheme = getattr(module, "quantization_scheme", None)
    if scheme is None:
        return None
    return getattr(scheme, "weights", None)


def assert_decoder_only_boundary(model, *, ignore: list[str] | None = None) -> dict:
    """Assert that only the text decoder projections came out quantized.

    Call this on a model a `QuantizationModifier`'s config has already been
    applied to -- applying the config is what attaches `quantization_scheme` to
    a module. Driving the real session initialize path is the caller's job; see
    `bench/probe_awq_recipe_boundary.py`. On a model whose config was never
    applied this raises rather than passing vacuously, because the decoder
    projections it expects to find quantized are not: measured, not assumed.

    What must hold:

    - the set of modules carrying a *weight* quantization scheme is exactly the
      `TEXT_DECODER_LINEAR_LEAVES` projections inside the decoder layers;
    - every decoder layer contributes all of them, so the count is the layer
      count times the leaf count and no layer was partly skipped;
    - nothing in the vision tower, the merger, the DeepStack mergers, the output
      head or the input embedding carries a scheme;
    - no non-Linear module carries a weight scheme;
    - every targeted module carries the *same* scheme, so a single set of scheme
      fields describes the artifact.

    :param ignore: the patterns the recipe was built with, recorded in the
        returned dict. Not consulted to decide correctness -- see the module
        docstring.
    :return: the record: counts per category, the ignore patterns, and the
        scheme fields.
    :raises BoundaryViolation: on any of the above.
    """
    buckets = _classify_linears(model)
    decoder_layer_names = buckets.pop("_decoder_layer_names")
    tower_name = buckets.pop("_vision_tower_name")[0]

    expected = {
        f"{layer}.{leaf}"
        for layer in decoder_layer_names
        for leaf in TEXT_DECODER_LINEAR_LEAVES
    }
    missing_from_model = expected - set(buckets["text_decoder"])
    if missing_from_model:
        raise BoundaryViolation(
            "the model does not have the decoder projections this boundary is "
            f"defined over: {sorted(missing_from_model)[:8]}"
        )

    if buckets["unclassified"]:
        raise BoundaryViolation(
            "Linear modules landed in no known category: "
            f"{sorted(buckets['unclassified'])[:8]}. The architecture has moved "
            "under the ignore patterns, so whether they are covered is unknown "
            "-- classify them before trusting this boundary"
        )

    scheme_carriers = {
        name: module
        for name, module in model.named_modules()
        if _weight_scheme(module) is not None
    }
    non_linear = sorted(
        name
        for name, module in scheme_carriers.items()
        if not isinstance(module, torch.nn.Linear)
    )
    if non_linear:
        raise BoundaryViolation(
            f"non-Linear modules carry a weight quantization scheme: {non_linear[:8]}"
        )

    targeted = set(scheme_carriers)
    unexpected = sorted(targeted - expected)
    unquantized = sorted(expected - targeted)
    if unexpected or unquantized:
        raise BoundaryViolation(
            "the quantized population is not the text decoder projections. "
            f"Quantized outside the decoder ({len(unexpected)}): {unexpected[:8]}. "
            f"Decoder projections left unquantized ({len(unquantized)}): "
            f"{unquantized[:8]}"
        )

    embedding = model.get_input_embeddings()
    if embedding is not None and _weight_scheme(embedding) is not None:
        raise BoundaryViolation("the input embedding carries a weight scheme")
    head = model.get_output_embeddings()
    if head is not None and _weight_scheme(head) is not None:
        raise BoundaryViolation("the output head carries a weight scheme")

    schemes = {repr(_weight_scheme(module)) for module in scheme_carriers.values()}
    if len(schemes) != 1:
        raise BoundaryViolation(
            f"targeted modules carry {len(schemes)} different weight schemes; a "
            "single set of scheme fields cannot describe the artifact"
        )
    weights = _weight_scheme(next(iter(scheme_carriers.values())))

    def _plain(value):
        return getattr(value, "value", value)

    return {
        "vision_tower_module": tower_name,
        "text_decoder_layers": len(decoder_layer_names),
        "linears_per_decoder_layer": len(TEXT_DECODER_LINEAR_LEAVES),
        "ignore_patterns": ignore_patterns() if ignore is None else list(ignore),
        "linear_counts": {key: len(value) for key, value in buckets.items()},
        "linears_total": sum(len(value) for value in buckets.values()),
        "targeted_linears": len(targeted),
        "ignored_linears": sum(len(value) for value in buckets.values()) - len(targeted),
        "scheme": {
            "num_bits": weights.num_bits,
            "group_size": weights.group_size,
            "symmetric": weights.symmetric,
            "strategy": _plain(weights.strategy),
            "type": _plain(weights.type),
            "observer": weights.observer,
            "dynamic": _plain(weights.dynamic),
            "actorder": _plain(weights.actorder),
        },
    }
