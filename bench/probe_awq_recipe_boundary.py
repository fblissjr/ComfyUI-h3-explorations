#!/usr/bin/env python3
"""Grade the W4A16 AWQ v2 recipe's boundary before any weights are loaded.

The expensive way to find out which modules a recipe quantizes is to run it.
This runs the pinned `llm-compressor` session's own initialize path -- the same
`session.initialize(model=..., recipe=...)` call `oneshot` makes -- against two
models that cost nothing:

1. the reduced-width Qwen3-VL built by
   `bench/check_calibration_precision_policy.py::tiny_full_model`, which keeps
   the released vision tower's shape ratios;
2. the released config instantiated under `init_empty_weights`, so every
   parameter is a meta tensor: real module names, real widths, no checkpoint
   read and no download.

The config application is not reimplemented here. `QuantizationModifier`
attaches a `quantization_scheme` to a module inside
`QuantizationMixin.initialize_quantization`, and the only honest way to ask what
the recipe does is to let the session call it.

**The control is the deployed artifact.** The candidate's resolved ignore list
and scheme fields are compared against
`config/qwen3vl_32b_minimax_h3_w4a16_awq/config.json`, and the candidate's list
is produced by `compressed_tensors.quantization.QuantizationConfig.from_pretrained`
-- the same serializer that wrote that file. So the comparison is between two
runs of one writer, not between a file and a list this script assembled to match
it. A difference in either direction fails the probe.

Three red controls run and must each fail, because a probe whose assertions have
never been seen to fire is a probe nobody can distinguish from `return 0`:

- an ignore list missing the DeepStack mergers must be caught by the boundary
  assertion, not by a pattern comparison;
- the rejected preflight's `scheme` *and* `config_groups` must fail at
  construction;
- a nonexistent `AWQModifier` field must fail at construction.

CPU only, no CUDA, no checkpoint, no output directory. Run it with the pinned
`llm-compressor` virtualenv. Writes exactly one file,
`bench/results/2026-08-25_awq_recipe_boundary.json`, and records the working
directory's entries before and after the session work to say so.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Snapshot before any heavy import: a library that creates a log directory at
# import time would otherwise be invisible to the check at the bottom.
_CWD_AT_START = sorted(p.name for p in Path.cwd().iterdir())

# No network. The released config is read from a local directory; if that
# directory ever went missing, a silent hub fetch would turn this probe into a
# download and the failure would be reported as a config mismatch.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

from h3_awq_recipe import (  # noqa: E402
    BoundaryViolation,
    assert_decoder_only_boundary,
    build_recipe,
    describe_recipe,
    ignore_patterns,
    resolved_awq_mappings,
)
from h3_producer_provenance import producer_provenance  # noqa: E402

_ROOT_AT_START = sorted(p.name for p in ROOT.iterdir())

DEPLOYED_CONFIG = ROOT / "config" / "qwen3vl_32b_minimax_h3_w4a16_awq" / "config.json"
RELEASED_CONFIG_DIR = ROOT / "coderef" / "llm-compressor" / "models" / "qwen3-vl-32b-bf16"
LLM_COMPRESSOR = ROOT / "coderef" / "llm-compressor"
RESULT = BENCH / "results" / "2026-08-25_awq_recipe_boundary.json"

# Normative: the released text stack is 64 decoder layers and the candidate
# quantizes seven projections in each. Both halves are read back off the model
# and the config in `check_released_arm`; this constant is what the probe
# refuses to pass without.
EXPECTED_TARGETED_LINEARS = 448

# Compared strictly against the deployed artifact. `format` is deliberately not
# in this set: the deployed group carries "pack-quantized" because it was set
# when the checkpoint was compressed and saved, which this probe never does.
COMPARED_SCHEME_FIELDS = (
    "num_bits", "type", "symmetric", "group_size", "strategy",
    "observer", "dynamic", "actorder",
)


def versions() -> dict:
    import importlib.metadata as metadata

    record = {"python": ".".join(str(v) for v in sys.version_info[:3])}
    for package in ("torch", "transformers", "llmcompressor", "compressed-tensors"):
        try:
            record[package] = metadata.version(package)
        except Exception as exc:
            record[package] = f"unavailable: {type(exc).__name__}"
    try:
        record["llm_compressor_commit"] = subprocess.run(
            ["git", "-C", str(LLM_COMPRESSOR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        record["llm_compressor_dirty"] = bool(subprocess.run(
            ["git", "-C", str(LLM_COMPRESSOR), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except Exception as exc:
        record["llm_compressor_commit"] = None
        record["git_error"] = f"{type(exc).__name__}: {exc}"
    return record


def tiny_model():
    from check_calibration_precision_policy import tiny_full_model

    return tiny_full_model()


def released_meta_model():
    """The released architecture with no weights anywhere.

    `init_empty_weights` puts every parameter on the meta device, so the module
    tree, the class names and every width are the released ones while nothing is
    read from disk and nothing is allocated. The quantization config attaches
    scales and zero points to meta tensors, which is all the boundary question
    needs.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    if not (RELEASED_CONFIG_DIR / "config.json").is_file():
        raise FileNotFoundError(
            "the released config is not present at the expected repo-relative "
            "location; this probe reads a config, never weights, and will not "
            "fall back to the hub"
        )
    config = AutoConfig.from_pretrained(str(RELEASED_CONFIG_DIR))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    return model, config


def apply_recipe(model, recipe):
    """Drive the real session initialize path and read the applied state back.

    Returns the boundary record, the resolved AWQ mappings and the quantization
    config compressed-tensors would serialise for this model. Everything is
    captured before `finalize`, which clears the modifier's resolved mappings.
    """
    from compressed_tensors.quantization import QuantizationConfig
    from llmcompressor.core.session_functions import create_session

    with create_session() as session:
        session.initialize(model=model, start=-1, recipe=recipe)
        awq, quant = session.lifecycle.recipe.modifiers
        record = assert_decoder_only_boundary(model, ignore=quant.ignore)
        declared = [
            {"smooth_layer": m.smooth_layer, "balance_layers": list(m.balance_layers)}
            for m in awq.mappings
        ]
        mappings = resolved_awq_mappings(awq, model)
        serialised = QuantizationConfig.from_pretrained(model)
        session.finalize()
    return record, declared, mappings, serialised


def mappings_for_first_layer(mappings: list[dict], prefix: str) -> list[dict]:
    return [m for m in mappings if m["smooth_layer"].startswith(prefix)]


def _matches(target: str, name: str) -> bool:
    """compressed-tensors' own target semantics: `re:` means `re.match`, else equality."""
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    return target == name


def mapping_census(declared: list[dict], mappings: list[dict], layers: int,
                   failures: list[str], arm: str) -> dict:
    """Where AWQ's smoothing lands, and whether it lands the same way everywhere.

    AWQ rewrites the smooth layer's weights as well as the balance layers', so a
    mapping resolving outside the decoder stack would move weights the ignore
    list is protecting -- a second boundary, and one the quantization scheme
    census cannot see. It is asserted here rather than assumed from the fact
    that untargeted mappings are skipped.

    A layer that contributes fewer mappings than its neighbours is the other
    failure worth naming: the run would smooth most of the stack and quietly
    leave one layer alone.
    """
    prefix = "model.language_model.layers."
    outside = [m["smooth_layer"] for m in mappings
               if not m["smooth_layer"].startswith(prefix)]
    per_layer: dict[str, int] = {}
    shapes: dict[str, int] = {}
    for mapping in mappings:
        smooth = mapping["smooth_layer"]
        if smooth in outside:
            continue
        index = smooth[len(prefix):].split(".", 1)[0]
        per_layer[index] = per_layer.get(index, 0) + 1
        leaf = smooth[len(prefix) + len(index) + 1:]
        balances = tuple(
            name.split(".", 4)[-1] for name in mapping["balance_layers"]
        )
        shapes[f"{leaf} -> {', '.join(balances)}"] = (
            shapes.get(f"{leaf} -> {', '.join(balances)}", 0) + 1
        )
    counts = sorted(set(per_layer.values()))
    if outside:
        failures.append(
            f"{arm}: AWQ resolved {len(outside)} mappings outside the decoder "
            f"stack, so smoothing would rewrite protected weights: {outside[:4]}"
        )
    if len(per_layer) != layers:
        failures.append(
            f"{arm}: AWQ resolved mappings onto {len(per_layer)} of {layers} "
            "decoder layers"
        )
    if len(counts) != 1:
        failures.append(
            f"{arm}: decoder layers do not all get the same number of AWQ "
            f"mappings; counts seen {counts}"
        )
    return {
        "declared": declared,
        "declared_but_never_resolved": [
            entry["smooth_layer"] for entry in declared
            if not any(_matches(entry["smooth_layer"], m["smooth_layer"])
                       for m in mappings)
        ],
        "total": len(mappings),
        "per_decoder_layer": counts[0] if len(counts) == 1 else counts,
        "layers_covered": len(per_layer),
        "outside_decoder_layers": outside,
        "shapes": shapes,
        "why_some_declared_shapes_are_absent":
            "source read of the pinned resolver, corroborated by its own skipped "
            "count: a declared smooth layer ending in `.v_proj` is dropped when "
            "its out_features differ from the balance layer's in_features. Under "
            "grouped-query attention v_proj is narrower than o_proj by the "
            "head-to-kv-head ratio, so the v_proj -> o_proj mapping never "
            "resolves and o_proj is quantized without being smoothed",
    }


def check_construction_gate(failures: list[str]) -> list[dict]:
    """The two rejected-preflight defects, each built and each required to fail."""
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    outcomes = []

    try:
        AWQModifier(duality=False)
        failures.append(
            "red control: AWQModifier accepted a nonexistent field, so the "
            "construction gate the rejected preflight died on is gone"
        )
        outcomes.append({"control": "nonexistent AWQModifier field", "fired": False})
    except Exception as exc:
        outcomes.append({
            "control": "nonexistent AWQModifier field",
            "constructed": "AWQModifier(duality=False)",
            "fired": True,
            "exception": type(exc).__name__,
            "message": " ".join(str(exc).split())[:300],
        })

    both = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True,
                        "group_size": 128, "strategy": "group"},
        }
    }
    try:
        QuantizationModifier(
            scheme="W4A16", targets=["Linear"], ignore=ignore_patterns(),
            config_groups=both,
        )
        failures.append(
            "red control: QuantizationModifier accepted both `scheme` and "
            "`config_groups`, which the installed resolver is supposed to forbid"
        )
        outcomes.append({"control": "scheme and config_groups together", "fired": False})
    except Exception as exc:
        outcomes.append({
            "control": "scheme and config_groups together",
            "constructed": "QuantizationModifier(scheme=..., config_groups=...)",
            "fired": True,
            "exception": type(exc).__name__,
            "message": " ".join(str(exc).split())[:300],
        })
    return outcomes


def check_missing_deepstack_control(failures: list[str], deployed_ignore: list[str]) -> dict:
    """An ignore list that forgets the DeepStack mergers must be caught.

    This is the defect the boundary assertion exists for: the mergers sit under
    the vision tower but are not vision blocks, so a pattern written by looking
    at the block list omits them, and every count except theirs still looks
    right. The assertion must name them, and the deployed-artifact comparison
    must also see the difference -- if it did not, the comparison would be
    passing on a list it cannot discriminate.
    """
    from compressed_tensors.quantization import QuantizationConfig
    from llmcompressor.core.session_functions import create_session

    mutilated = [
        "lm_head",
        r"re:model\.visual\.blocks\..*",
        r"re:model\.visual\.merger\..*",
    ]
    model = tiny_model()
    recipe = build_recipe(ignore=mutilated)
    outcome: dict = {
        "control": "ignore list missing the DeepStack mergers",
        "ignore": mutilated,
        "fired": False,
    }
    with create_session() as session:
        session.initialize(model=model, start=-1, recipe=recipe)
        try:
            assert_decoder_only_boundary(model, ignore=mutilated)
            failures.append(
                "red control: an ignore list without the DeepStack mergers passed "
                "assert_decoder_only_boundary, so the boundary assertion is blind "
                "to the defect it was written for"
            )
        except BoundaryViolation as exc:
            outcome["fired"] = True
            outcome["exception"] = type(exc).__name__
            outcome["message"] = " ".join(str(exc).split())[:400]
        serialised = QuantizationConfig.from_pretrained(model)
        session.finalize()
    outcome["serialised_ignore_len"] = len(serialised.ignore)
    outcome["serialised_ignore_matches_deployed"] = serialised.ignore == deployed_ignore
    if outcome["serialised_ignore_matches_deployed"]:
        failures.append(
            "red control: the mutilated recipe serialises the same ignore list as "
            "the deployed artifact, so the deployed comparison cannot discriminate"
        )
    return outcome


def compare_with_deployed(serialised, failures: list[str]) -> dict:
    deployed = json.loads(DEPLOYED_CONFIG.read_text())["quantization_config"]
    deployed_ignore = deployed["ignore"]
    candidate_ignore = list(serialised.ignore)
    candidate_group = next(iter(serialised.config_groups.values()))
    candidate_weights = candidate_group.weights.model_dump()
    deployed_weights = deployed["config_groups"]["group_0"]["weights"]

    comparison: dict = {
        "deployed_config": "config/qwen3vl_32b_minimax_h3_w4a16_awq/config.json",
        "deployed_ignore_len": len(deployed_ignore),
        "candidate_ignore_len": len(candidate_ignore),
        "ignore_identical": candidate_ignore == deployed_ignore,
        "ignore_only_in_candidate": sorted(set(candidate_ignore) - set(deployed_ignore)),
        "ignore_only_in_deployed": sorted(set(deployed_ignore) - set(candidate_ignore)),
        "deployed_targets": deployed["config_groups"]["group_0"]["targets"],
        "candidate_targets": list(candidate_group.targets),
        "compared_scheme_fields": list(COMPARED_SCHEME_FIELDS),
        "scheme_field_deltas": {},
        "not_compared": {
            "format": {
                "deployed": deployed["config_groups"]["group_0"].get("format"),
                "candidate": candidate_group.format,
                "why": "set when the checkpoint is compressed and saved, which "
                       "this probe never does",
            },
            "quantization_status": {
                "deployed": deployed.get("quantization_status"),
                "candidate": serialised.quantization_status,
                "why": "the deployed artifact is a finished checkpoint; this "
                       "model has only had the config applied",
            },
        },
    }
    if not comparison["ignore_identical"]:
        failures.append(
            "the candidate's resolved ignore list differs from the deployed "
            f"artifact's: only in candidate {comparison['ignore_only_in_candidate'][:6]}, "
            f"only in deployed {comparison['ignore_only_in_deployed'][:6]}"
        )
    if comparison["candidate_targets"] != comparison["deployed_targets"]:
        failures.append(
            f"targets differ: candidate {comparison['candidate_targets']} against "
            f"deployed {comparison['deployed_targets']}"
        )
    for field in COMPARED_SCHEME_FIELDS:
        mine = candidate_weights.get(field)
        theirs = deployed_weights.get(field)
        if mine != theirs:
            comparison["scheme_field_deltas"][field] = {
                "candidate": mine, "deployed": theirs,
            }
            failures.append(
                f"scheme field `{field}` differs: candidate {mine!r} against "
                f"deployed {theirs!r}"
            )
    comparison["scheme_fields_identical"] = not comparison["scheme_field_deltas"]
    return comparison


def check_tiny_arm(failures: list[str], recipe) -> dict:
    model = tiny_model()
    record, declared, mappings, serialised = apply_recipe(model, recipe)
    layers = record["text_decoder_layers"]
    expected = layers * record["linears_per_decoder_layer"]
    if record["targeted_linears"] != expected:
        failures.append(
            f"tiny arm: {record['targeted_linears']} targeted Linears, expected "
            f"{expected} for {layers} decoder layers"
        )
    first = mappings_for_first_layer(mappings, "model.language_model.layers.0.")
    if not first:
        failures.append("tiny arm: no AWQ mapping resolved onto the first decoder layer")
    return {
        "model": "reduced-width Qwen3-VL from "
                 "bench/check_calibration_precision_policy.py::tiny_full_model",
        "note": "the reduced-width text stack is narrower than the group size, so "
                "compressed-tensors warns that the group does not divide the "
                "weight columns. That is a warning about the reduced widths, not "
                "about the recipe; the released arm has no such warning and the "
                "boundary this probe grades does not depend on it",
        "boundary": record,
        "awq_mapping_census": mapping_census(
            declared, mappings, record["text_decoder_layers"], failures, "tiny arm"),
        "resolved_awq_mappings_first_layer": first,
        "serialised_ignore_len": len(serialised.ignore),
        "serialised_ignore": list(serialised.ignore),
    }


def check_released_arm(failures: list[str], recipe, deployed_ignore: list[str]) -> tuple[dict, object]:
    model, config = released_meta_model()
    record, declared, mappings, serialised = apply_recipe(model, recipe)

    declared_layers = config.text_config.num_hidden_layers
    if record["text_decoder_layers"] != declared_layers:
        failures.append(
            f"released arm: found {record['text_decoder_layers']} decoder layers, "
            f"the config declares {declared_layers}"
        )
    if record["targeted_linears"] != EXPECTED_TARGETED_LINEARS:
        failures.append(
            f"released arm: {record['targeted_linears']} targeted Linears, the "
            f"probe requires {EXPECTED_TARGETED_LINEARS}"
        )
    if record["targeted_linears"] != declared_layers * record["linears_per_decoder_layer"]:
        failures.append(
            "released arm: the targeted count is not the declared layer count "
            "times the per-layer projection count"
        )
    if record["ignored_linears"] != len(deployed_ignore):
        failures.append(
            f"released arm: {record['ignored_linears']} ignored Linears against "
            f"the deployed artifact's ignore list of {len(deployed_ignore)}"
        )
    declared_blocks = config.vision_config.depth
    counts = record["linear_counts"]
    if counts["vision_block"] != declared_blocks * 4:
        failures.append(
            f"released arm: {counts['vision_block']} vision-block Linears for "
            f"{declared_blocks} declared blocks"
        )
    declared_deepstack = len(config.vision_config.deepstack_visual_indexes)
    if counts["vision_deepstack_merger"] != declared_deepstack * 2:
        failures.append(
            f"released arm: {counts['vision_deepstack_merger']} DeepStack merger "
            f"Linears for {declared_deepstack} declared taps"
        )
    first = mappings_for_first_layer(mappings, "model.language_model.layers.0.")
    if not first:
        failures.append("released arm: no AWQ mapping resolved onto the first decoder layer")

    arm = {
        "model": "released config under init_empty_weights (meta tensors, no weights)",
        "config_source": "coderef/llm-compressor/models/qwen3-vl-32b-bf16 "
                         "(config only; gitignored sister checkout)",
        "declared_text_layers": declared_layers,
        "declared_vision_blocks": declared_blocks,
        "declared_deepstack_taps": declared_deepstack,
        "tie_word_embeddings": config.tie_word_embeddings,
        "boundary": record,
        "awq_mapping_census": mapping_census(
            declared, mappings, record["text_decoder_layers"], failures, "released arm"),
        "resolved_awq_mappings_first_layer": first,
        "serialised_ignore_len": len(serialised.ignore),
    }
    return arm, serialised


def main() -> int:
    failures: list[str] = []
    deployed = json.loads(DEPLOYED_CONFIG.read_text())["quantization_config"]
    deployed_ignore = deployed["ignore"]

    recipe = build_recipe()
    description = describe_recipe(recipe)
    print("recipe:", " + ".join(entry["class"] for entry in description))

    print("tiny arm:")
    tiny = check_tiny_arm(failures, recipe)
    print(f"  targeted {tiny['boundary']['targeted_linears']}, "
          f"ignored {tiny['boundary']['ignored_linears']}, "
          f"serialised ignore {tiny['serialised_ignore_len']}")

    print("released arm:")
    # A second, unused recipe object: the tiny arm's modifiers now carry that
    # model's inferred mappings and a resolved config, and reusing them would
    # measure the first arm's leftovers as the second arm's result.
    released, serialised = check_released_arm(failures, build_recipe(), deployed_ignore)
    print(f"  targeted {released['boundary']['targeted_linears']}, "
          f"ignored {released['boundary']['ignored_linears']}")

    print("deployed comparison:")
    comparison = compare_with_deployed(serialised, failures)
    print(f"  ignore identical {comparison['ignore_identical']}, "
          f"scheme fields identical {comparison['scheme_fields_identical']}")

    print("red controls:")
    controls = check_construction_gate(failures)
    controls.append(check_missing_deepstack_control(failures, deployed_ignore))
    for control in controls:
        print(f"  {control['control']}: fired={control['fired']}")

    cwd_now = sorted(p.name for p in Path.cwd().iterdir())
    root_now = sorted(p.name for p in ROOT.iterdir())
    filesystem = {
        "claim": "the session initialize/finalize path used here writes no file",
        "checked": "entries of the process working directory and of the repository "
                   "root, snapshotted before any llmcompressor import and again "
                   "after every session, before this probe writes its result",
        "working_directory_created": sorted(set(cwd_now) - set(_CWD_AT_START)),
        "working_directory_removed": sorted(set(_CWD_AT_START) - set(cwd_now)),
        "repository_root_created": sorted(set(root_now) - set(_ROOT_AT_START)),
        "repository_root_removed": sorted(set(_ROOT_AT_START) - set(root_now)),
    }
    for key in ("working_directory_created", "repository_root_created"):
        if filesystem[key]:
            failures.append(f"the session path created {filesystem[key]} ({key})")
    filesystem["nothing_written"] = not any(
        filesystem[key] for key in
        ("working_directory_created", "working_directory_removed",
         "repository_root_created", "repository_root_removed")
    )
    print(f"filesystem: nothing written {filesystem['nothing_written']}")

    result = {
        "probe": "which modules the W4A16 AWQ v2 candidate recipe quantizes, "
                 "graded before any weights are loaded",
        "path_policy": "logical identifiers only",
        "producer": producer_provenance(__file__),
        "environment": versions(),
        "cuda_used": False,
        "recipe": description,
        "arms": {"tiny_full_model": tiny, "released_config_meta": released},
        "deployed_comparison": comparison,
        "red_controls": controls,
        "filesystem": filesystem,
        "failures": failures,
        "verdict": "green" if not failures else "red",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {RESULT.relative_to(ROOT)}")

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print(
        "GREEN: the candidate recipe constructs in the pinned environment, "
        "quantizes exactly the text decoder projections on both the "
        "reduced-width model and the released architecture, resolves to the "
        "deployed artifact's ignore list and scheme fields, and all three red "
        "controls fired"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
