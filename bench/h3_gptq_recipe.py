#!/usr/bin/env python3
"""The W4A16 GPTQ recipe, and the AWQ-then-GPTQ composition, on the v2 boundary.

Gate 5 rejected the AWQ v2 candidate against v1: both artifacts sit at relative
L2 0.33 to 0.38 against BF16 at the layer-49 output on vision rows, and swapping
the calibration data moved that by about a tenth in either direction
(`docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-25_v2_launch_record.md`,
Gate 5 result). The reading recorded there is that the recipe owns the result and
the data decides the sign of a small term. AWQ is a *transform*: it rescales
channels so the 4-bit grid lands where the activations are large, then rounds
each weight independently. It moves rounding error between channels; it never
compensates it. GPTQ does compensate -- it quantizes column by column and pushes
each column's error into the columns it has not reached yet, using the inverse
Hessian of the layer's own input covariance. That is the axis Gate 5 did not
vary, so this module exists to vary it.

Nothing here quantizes anything or reads a checkpoint. It builds modifier lists.

## What this reuses rather than restates

The boundary is `bench/h3_awq_recipe.py`'s and is imported, not retyped: the same
`IGNORE_PATTERNS`, the same `ignore_patterns()`, and the same
`assert_decoder_only_boundary` as the control. A second copy of that list would
be a second thing to keep correct, and the escaped instance behind the AWQ
module's own docstring -- an ignore list written by looking at the vision block
names and silently missing the DeepStack mergers -- is exactly what a copy
re-earns. `bench/check_h3_gptq_recipe.py` is the red control for this file and
mutates that list in both directions.

## The GPTQ fields, and which of them are defaults

Line citations in this module are into `coderef/llm-compressor/src/`, the
pinned editable install, at `llmcompressor 0.13.1.dev38+g501f432bf` with
`compressed-tensors 0.18.1a20260821`. `gptq/base.py` is
`llmcompressor/modifiers/gptq/base.py`, `gptq_quantize.py` is its sibling,
`awq/base.py` is `llmcompressor/modifiers/transform/awq/base.py`, and
`mixin.py` is `llmcompressor/modifiers/quantization/quantization/mixin.py`.

Read off `GPTQModifier` in the pinned tree; every value below is passed
explicitly, including the ones that equal the installed default, because "the
default happened to be right" is not a decision a later reader can review.

- **`scheme="W4A16"`** (`QuantizationMixin.scheme` default is None). The preset
  resolves to int4, symmetric, group 128, `memoryless_minmax`
  (`compressed_tensors.quantization.quant_scheme.PRESET_SCHEMES`), which is the
  scheme the deployed v1 artifact carries. Holding it fixed is what makes a GPTQ
  arm a comparison against v1 rather than a second uncontrolled change.
- **`targets=["Linear"]`** (installed default is `["Linear"]`). Set explicitly so
  the red control that narrows it has something to narrow.
- **`ignore=ignore_patterns()`** (installed default is `[]`). The decoder-only
  boundary. With the default the vision tower, both merger families and the
  output head would all be quantized.
- **`block_size=128`** (installed default 128). GPTQ's lazy-batch column width:
  `quantize_weight` processes `[i1:i2]` column blocks and defers the error
  update of the columns beyond `i2` to the end of the block
  (`gptq_quantize.py:167-244`). Held equal to
  `group_size` so that one column block lies inside exactly one quantization
  group -- `g_idx = arange(num_columns) // group_size` -- which means no block
  ever straddles two scales. It is a working-set and blocking choice, not an
  accuracy knob: the per-column update is the same either way.
- **`dampening_frac=0.01`** (installed default 0.01; the class docstring's sample
  YAML shows 0.001). This is `percdamp`: a ridge of `percdamp * mean(diag(H))`
  added to the Hessian diagonal before the Cholesky
  (`gptq_quantize.py:150-152`). It is load-bearing here rather than cosmetic.
  The failure it guards is not subtle -- on `_LinAlgError` the installed code
  falls back to **round-to-nearest for that module** and only logs a warning
  (`gptq_quantize.py:157-164`), so an under-damped run produces a candidate in
  which some layers had no GPTQ at all and nothing in the artifact says which.
  The library default is taken rather than the docstring's smaller value because
  this model's widest input is 25,600 columns accumulated in FP32 over a
  six-figure token count, which is where conditioning goes wrong; a smaller
  ridge is a thing to try *after* a run that is known to have inverted every
  Hessian, not before. **The pilot must report the fallback count**, and until
  it does, "no warning was seen" is not evidence.
- **`actorder="static"`** (installed default is the sentinel `Sentinel("static")`,
  which resolves to `ActivationOrdering.STATIC` only when the scheme carries no
  `actorder`; the W4A16 preset carries None, so the sentinel would resolve the
  same way). Passed as a real value so the resolved config is not a function of
  which preset is in force. STATIC permutes the columns by Hessian diagonal for
  the quantization order and inverts the permutation before the weight is
  written (`gptq_quantize.py:107-108, 247-250`), so the stored layout and the
  runtime cost are unchanged -- the class docstring says the same
  (`gptq/base.py:93-96`). `GROUP` and `DYNAMIC` are refused by `build_recipe`: the
  installed source deprecates both and warns that `GROUP` will be removed
  (`gptq/base.py:96`, and the warning it emits at `gptq/base.py:181-184`), and
  `GROUP` additionally writes a `weight_g_idx` the deployed adapter has never
  been asked to load.
- **`offload_hessians=False`** (installed default False). Where the Hessians sit
  *between* forwards. See the budget below: this is the field the card probe
  decides, and it is the one field here whose right value is currently unknown.

## Does AWQ compose with GPTQ in one recipe, and in what order

**SOURCE, from the pinned tree, three independent readings.**

1. `AWQModifier`'s own class docstring says it "does not perform quantization or
   compression on its own" and "must be applied in conjunction with a modifier
   that inherits from `QuantizationMixin`"
   (`awq/base.py:73-76`). `GPTQModifier` is
   declared `class GPTQModifier(Modifier, QuantizationMixin)`
   (`gptq/base.py:47`), so it is such a modifier and no separate
   `QuantizationModifier` is needed or wanted -- two config-bearing modifiers in
   one recipe would apply the config twice.
2. The pinned tree ships the composition as an example:
   `examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py` builds
   `recipe = [AWQModifier(...), GPTQModifier(...)]` and hands it to `oneshot`.
3. Order is the recipe list's order, at every lifecycle point:
   `CompressionLifecycle.initialize`, `.finalize` and `.event` each iterate
   `for mod in self.recipe.modifiers` with no sort
   (`llmcompressor/core/lifecycle.py:103, 136, 203`). So with
   `[AWQModifier, GPTQModifier]`: AWQ's `on_initialize` infers its mappings
   first, GPTQ's `on_initialize` then applies the quantization config, AWQ's
   `on_calibration_start` resolves mappings against modules that already carry a
   scheme, and at each `SEQUENTIAL_EPOCH_END` **AWQ smooths first and GPTQ
   quantizes second** (`awq/base.py:249`, `gptq/base.py:240`).

**The composition has a seam, and it is not hypothetical.** GPTQ accumulates its
Hessian inside a forward hook (`gptq/base.py:256-291`), so a layer's Hessian is
built from the activations of the forwards that ran *before* AWQ rewrote that
layer's weights at the epoch end. The weight GPTQ then quantizes is the smoothed
one, while the input covariance it compensates against is the unsmoothed one.
Under a smoothing scale `s` the true input becomes `X/s`, and the matching
Hessian would be `diag(1/s) H diag(1/s)`; nothing in the installed code applies
that. **INFERENCE, from a source read and not from a run:** this makes
`awq_gptq` a weaker composition than either half suggests, and it is a reason to
measure `gptq` alone first rather than assuming the stack is additive. The
shipped example above is evidence that the pair is *supported*, not that the
seam is closed.

## Per-layer schemes, if a mixed-precision variant is wanted later

`overrides` exists so a W8-on-named-layers variant needs no new file. In this
version of compressed-tensors a per-layer scheme is expressed as
`config_groups`: a `dict[str, QuantizationScheme]`, each scheme carrying its own
`targets`. `QuantizationMixin.resolve_quantization_config` refuses `scheme` and
`config_groups` together (`mixin.py:345-346`), so passing an override switches
this builder from the `scheme=` form to the `config_groups=` form. The pinned
tree's own worked example of exactly this shape --  int8 on `down_proj`, int4
elsewhere -- is
`examples/quantization_non_uniform/quantization_int4_int8.py`.

Precedence when a module matches two groups is **not** dict order.
`apply_quantization_config` builds one `target_to_scheme` map and takes
`targets[0]` from `match_targets`, which sorts its matches most-specific-first:
exact name, then `re:` pattern, then class name
(`compressed_tensors/utils/match.py::match_targets`,
`.../quantization/lifecycle/apply.py::_scheme_from_targets`). So a `re:` override
beats the base group's `"Linear"` class target regardless of where it sits in the
dict, and this builder keeps the base group on `"Linear"` for that reason.

**A mixed recipe cannot be graded by the imported boundary assertion.**
`assert_decoder_only_boundary` requires every targeted module to carry the *same*
weight scheme, so that one set of scheme fields describes the artifact; a W4/W8
mix raises `BoundaryViolation` there by design. That refusal is asserted in
`bench/check_h3_gptq_recipe.py` rather than left as a surprise, and a
mixed-precision variant needs its own boundary assertion before it can ship.

## Host and device budget: INFERENCE, unmeasured until the card probe

**What GPTQ holds, from source.** One FP32 Hessian per targeted Linear, square in
that Linear's *input* width: `make_empty_hessian` allocates
`(weight.shape[1], weight.shape[1])` at `GPTQ_PRECISION = torch.float32`
(`gptq_quantize.py:19-25`). It is allocated on the execution device unless
`offload_hessians` is set, in which case it is allocated on the CPU and moved to
the execution device around each accumulate and the quantize step
(`gptq/base.py:256-291`, and the onload context at `gptq/base.py:389-400`).

**What GPTQ does not hold: the batch.** `accumulate_hessian` folds the input into
`H` with `H += inp.matmul(inp.t())` and returns; nothing retains the activation
(`gptq_quantize.py:28-64`). This is the whole difference from AWQ, which keeps
every batch's FP32 parent inputs in `_parent_args_cache` so the grid search can
re-run the parents (`awq/base.py:156-158, 411`) and was measured at about 430 KB
of host per population token
(`bench/results/2026-08-25_gate2b_host_budget_prefix8_2layers.json`). **GPTQ's
state does not scale with population tokens at all.** It scales with layer width,
and it is freed per layer: `compress_modules` pops each module's Hessian at the
sequential epoch end (`gptq/base.py:320-343`), and `on_finalize` raises if any
survive.

**The arithmetic, at this model's shapes.** Qwen3-VL-32B's text config declares
`hidden_size` 5120, `intermediate_size` 25600, 64 attention heads of
`head_dim` 128. One decoder layer's seven targeted inputs are therefore 5120
(q, k, v, gate, up), 8192 (o_proj, = heads x head_dim) and 25600 (down_proj), and
`4 * n^2` bytes each:

    5 x 4 x 5120^2  =    524,288,000 B
    1 x 4 x 8192^2  =    268,435,456 B
    1 x 4 x 25600^2 =  2,621,440,000 B
    per decoder layer  3,414,163,456 B  = 3.18 GiB

`down_proj` alone is 2.44 GiB of that, and the quantize step transiently doubles
it: the Cholesky chain rebinds `H` three times and each call allocates
(`gptq_quantize.py:153-155`), on top of an FP32 clone of the weight
(`gptq_quantize.py:88-92`, 0.49 GiB for `down_proj`). So the peak around one
`down_proj` is on the order of 5.4 GiB, wherever the module is executing.

`gptq_hessian_budget()` computes all of the above from a config rather than
restating it, so the method is checkable on a model even where the released
figures are not measured; `bench/check_h3_gptq_recipe.py` grades it against a
real model's attached modules.

**What this predicts, and what it does not.** Against the AWQ v2 run
(214,187 tokens, host peak 82.24 GiB, CUDA 9.83 GiB allocated / 17.53 reserved on
a 23.5 GiB card, `bench/results/2026-08-25_v2_calibration_run.json`), the same
population under GPTQ should lose AWQ's token-scaled host term entirely and keep
only the pipeline's own intermediates cache, measured at 78,457 B/token at two
layers (`..._gate2b_host_budget_prefix8_2layers.json`) -- about 16 GiB at that
population against about 86 GiB. The cost moves onto the card instead: 3.18 GiB
resident for the duration of each layer plus the transient above, against a run
that already reserved 17.5 of 23.5 GiB. **That is the open question, and it is
what `offload_hessians` is for**: setting it moves the 3.18 GiB to the host,
where the freed AWQ budget can absorb it, at the price of a 2.44 GiB host/device
round trip per `down_proj` forward. Every number in this section is arithmetic
from a source read. None of it is measured, and the pilot's own record is what
settles it:

    bench/pilot_sequential_feasibility.py --layers 2 --prefix 8 \
        --modifier gptq --offload auto_offload --host-reserve-gib <as for v2>

on the calibration bundle, on the GPU. A modifier-bearing run refuses any other
offload arrangement, and the reserve is what decides the weight tier, so both
belong in the command rather than in a reader's memory.

Importable from the pinned `llm-compressor` virtualenv: `torch`, `llmcompressor`
and `compressed_tensors` only. No ComfyUI, no transformers import at module
scope, no model load, no checkpoint read or write.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# `bench/` is a flat directory of top-level modules, not a package, and this one
# is the first of them to import a sibling at module scope. Adding its own
# directory keeps `import h3_gptq_recipe` working for a caller that has not
# already inserted `bench/`; it adds nothing else to the path and reaches
# nothing outside this file's own directory.
_BENCH = str(Path(__file__).resolve().parent)
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from h3_awq_recipe import (  # noqa: E402
    IGNORE_PATTERNS,
    TEXT_DECODER_LINEAR_LEAVES,
    assert_decoder_only_boundary,
    describe_recipe,
    ignore_patterns,
)

__all__ = [
    "ACTORDERS",
    "IGNORE_PATTERNS",
    "METHODS",
    "TEXT_DECODER_LINEAR_LEAVES",
    "assert_decoder_only_boundary",
    "build_recipe",
    "describe_recipe",
    "gptq_hessian_budget",
    "ignore_patterns",
]

# The two arms this file builds. `gptq` is the one Gate 5 did not try;
# `awq_gptq` is the shipped composition, with the seam named in the docstring.
METHODS: tuple[str, ...] = ("gptq", "awq_gptq")

# What `actorder` may be here, which is narrower than what the field accepts.
# `group` and `dynamic` are deprecated in the installed source and `group`
# additionally emits a `weight_g_idx` the deployed adapter has never loaded.
ACTORDERS: tuple[str | None, ...] = ("static", "weight", None)


def _quantization_config_kwargs(
    *,
    scheme: str,
    targets: list[str] | None,
    group_size: int,
    overrides: dict[str, str] | None,
    ignore: list[str] | None,
) -> dict[str, Any]:
    """The scheme half of the modifier's arguments, in whichever form it needs.

    Two forms, and the installed resolver refuses both at once
    (`mixin.py:345-346`):

    - no per-layer override and the preset's own group size: the plain
      `scheme=` / `targets=` form, which is what the deployed v1 artifact was
      produced under and what makes the emitted config comparable to it;
    - anything else: `config_groups`, built from the same presets, with the base
      group left on the class target `"Linear"` so every `re:` override outranks
      it under `match_targets`' most-specific-first ordering.
    """
    from compressed_tensors.quantization import preset_name_to_scheme

    resolved_targets = ["Linear"] if targets is None else list(targets)
    resolved_ignore = ignore_patterns() if ignore is None else list(ignore)

    base = preset_name_to_scheme(scheme, resolved_targets)
    if not overrides and base.weights is not None and base.weights.group_size == group_size:
        return {
            "scheme": scheme,
            "targets": resolved_targets,
            "ignore": resolved_ignore,
        }

    def _with_group_size(built):
        if built.weights is not None and built.weights.group_size != group_size:
            built.weights.group_size = group_size
        return built

    config_groups = {"group_0": _with_group_size(base)}
    for index, (pattern, preset) in enumerate(sorted((overrides or {}).items()), start=1):
        config_groups[f"group_{index}"] = _with_group_size(
            preset_name_to_scheme(preset, [pattern])
        )
    return {"config_groups": config_groups, "ignore": resolved_ignore}


def build_recipe(
    method: str = "gptq",
    *,
    group_size: int = 128,
    block_size: int = 128,
    dampening_frac: float = 0.01,
    actorder: str | None = "static",
    offload_hessians: bool = False,
    overrides: dict[str, str] | None = None,
    ignore: list[str] | None = None,
    scheme: str = "W4A16",
    targets: list[str] | None = None,
    awq_offload_device: str | torch.device = "cpu",
    awq_duo_scaling: bool | str = False,
    awq_n_grid: int = 20,
) -> list:
    """The GPTQ modifier list, fully constructed.

    Construction is the cheapest gate a launcher has and the one the rejected
    AWQ preflight failed, so nothing here is caught or defaulted away: an
    argument the pinned classes reject must stop the caller.

    Every field's value and the reason for it is in the module docstring; this
    signature only exposes them.

    :param method: `gptq` for GPTQ alone, `awq_gptq` for AWQ smoothing followed
        by GPTQ. The composition's order and its seam are in the module
        docstring.
    :param overrides: `{target pattern: preset scheme name}`, e.g.
        `{r"re:.*\\.mlp\\.down_proj$": "W8A16"}`. Switches the builder to the
        `config_groups` form. Note that the imported boundary assertion refuses
        a mixed-scheme population by design.
    :param ignore: override the boundary patterns. Present so a red control can
        build a deliberately wrong recipe; leave it None for a real run.
    :return: `[GPTQModifier]`, or `[AWQModifier, GPTQModifier]` in the order the
        lifecycle must see them -- AWQ smooths before GPTQ quantizes.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if actorder not in ACTORDERS:
        raise ValueError(
            f"actorder {actorder!r} is not offered here; expected one of "
            f"{ACTORDERS}. `group` and `dynamic` are deprecated in the installed "
            "GPTQModifier and `group` writes a weight_g_idx the deployed adapter "
            "has never been asked to load"
        )

    from llmcompressor.modifiers.gptq import GPTQModifier

    gptq = GPTQModifier(
        block_size=block_size,
        dampening_frac=dampening_frac,
        actorder=actorder,
        offload_hessians=offload_hessians,
        **_quantization_config_kwargs(
            scheme=scheme,
            targets=targets,
            group_size=group_size,
            overrides=overrides,
            ignore=ignore,
        ),
    )
    if method == "gptq":
        return [gptq]

    from llmcompressor.modifiers.transform.awq import AWQModifier

    return [
        AWQModifier(
            offload_device=torch.device(awq_offload_device),
            duo_scaling=awq_duo_scaling,
            n_grid=awq_n_grid,
        ),
        gptq,
    ]


def gptq_hessian_budget(text_config, *, leaves: tuple[str, ...] | None = None) -> dict:
    """What one decoder layer's GPTQ Hessians cost, derived from a config.

    Derived rather than restated: the module docstring's arithmetic is an
    INFERENCE about the released shapes, and this is the function that produces
    it, so a check can grade the *method* against a real model's attached
    modules even where the released figures are not measured
    (`bench/check_h3_gptq_recipe.py`).

    Per targeted Linear the modifier holds one FP32 square matrix in that
    Linear's input width (`gptq_quantize.py:19-25`), freed per layer at the
    sequential epoch end (`gptq/base.py:320-343`). The peak entry is the
    quantize-step transient: the Cholesky chain rebinds `H` three times, each
    allocating, on top of an FP32 clone of the weight.

    :param text_config: anything carrying `hidden_size`, `intermediate_size`,
        `num_attention_heads` and `head_dim` -- the released text config, or a
        reduced-width one.
    :return: input width and bytes per leaf, the per-layer total, and the widest
        leaf's quantize-step transient.
    """
    hidden = int(text_config.hidden_size)
    intermediate = int(text_config.intermediate_size)
    heads = int(text_config.num_attention_heads)
    head_dim = int(getattr(text_config, "head_dim", hidden // heads))
    element = torch.finfo(torch.float32).bits // 8

    widths = {
        "self_attn.q_proj": hidden,
        "self_attn.k_proj": hidden,
        "self_attn.v_proj": hidden,
        "self_attn.o_proj": heads * head_dim,
        "mlp.gate_proj": hidden,
        "mlp.up_proj": hidden,
        "mlp.down_proj": intermediate,
    }
    if leaves is not None:
        widths = {leaf: widths[leaf] for leaf in leaves}

    per_leaf = {
        leaf: {"input_width": width, "hessian_bytes": element * width * width}
        for leaf, width in widths.items()
    }
    total = sum(entry["hessian_bytes"] for entry in per_leaf.values())
    widest_leaf = max(per_leaf, key=lambda leaf: per_leaf[leaf]["hessian_bytes"])
    widest = per_leaf[widest_leaf]
    # The weight clone is (out_features x in_features) FP32; out_features is not
    # in this function's inputs for every leaf, so the clone is priced only for
    # the widest one, where it is `hidden x intermediate` by construction.
    clone_bytes = element * hidden * widest["input_width"]
    return {
        "element_bytes": element,
        "per_leaf": per_leaf,
        "hessian_bytes_per_decoder_layer": total,
        "hessian_gib_per_decoder_layer": round(total / 2**30, 3),
        "widest_leaf": widest_leaf,
        "quantize_transient_bytes_at_widest_leaf": 2 * widest["hessian_bytes"] + clone_bytes,
        "quantize_transient_gib_at_widest_leaf": round(
            (2 * widest["hessian_bytes"] + clone_bytes) / 2**30, 3
        ),
        "scales_with_population_tokens": False,
        "note": "one FP32 square per targeted Linear in its input width, freed "
                "per layer at the sequential epoch end. The transient doubles "
                "the widest Hessian for the Cholesky chain and adds an FP32 "
                "clone of the weight. INFERENCE from a source read of the "
                "pinned tree, not a measurement",
    }
