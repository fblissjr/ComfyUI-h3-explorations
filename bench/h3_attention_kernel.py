#!/usr/bin/env python3
"""Which form of the key/value heads reaches SDPA: grouped-query, or expanded.

The kernel axis of the Gate 2 design. Measured on 2026-08-25
(`bench/results/2026-08-25_sdpa_backend_selection_*.json`): at FP32 on the
RTX 4090 the grouped-query call the model declares (`enable_gqa=True`, 8 KV
heads against 64 query heads) has no fused SDPA kernel and dispatches to the
math backend, which materialises the full `[64, L, L]` logit tensor and fails
at 8,981 tokens. The same call with the KV heads explicitly repeated to 64
selects the memory-efficient kernel at every real length in the population.

transformers decides between the two inside
`transformers.integrations.sdpa_attention.sdpa_attention_forward`: it calls
`use_gqa_in_sdpa(attention_mask, key, value)` and either passes
`enable_gqa=True` or runs `repeat_kv` first. This module makes that decision a
named, revertible, scoped choice instead of a property of the library version:

| kind | what reaches SDPA |
|---|---|
| `grouped_query` | the library's own decision: `enable_gqa=True` when there is no mask |
| `expanded_kv` | `repeat_kv` always; SDPA sees 64 KV heads and no `enable_gqa` |

**The expansion is semantically a copy and is not assumed to be numerically
free.** Whether grouped-query math and expanded-KV math agree bit for bit is a
hypothesis until measured on released weights; a reduction-order difference of
the kind Gate 1B found in the position embedding is exactly the class of thing
that could separate them. `bench/compare_transformers_comfy_layer50.py
--attention` is where that is measured, and this module only makes the arms
constructible.

Scoped like `h3_calibration_precision.calibration_precision`: the helper is
module-level, so it is patched process-wide but gated by hooks on this model's
own attention modules and by the entering thread. The gate is on the attention
modules rather than the root model because `llm-compressor`'s sequential
pipeline never calls the root forward; it executes traced subgraphs that call
the submodules directly.

Importable from either virtualenv: pure `torch` and `transformers`.
"""

from __future__ import annotations

import contextlib
import inspect
import threading

ATTENTION_KINDS = ("grouped_query", "expanded_kv")

ATTENTION_INTENT = {
    "grouped_query": "the library's own decision: enable_gqa=True with no mask, "
                     "which at FP32 on this card selects the math backend and "
                     "materialises the full logit tensor",
    "expanded_kv": "repeat_kv before SDPA, unconditionally: SDPA sees the full "
                   "head count and no enable_gqa, which at FP32 on this card "
                   "selects the memory-efficient kernel. A measured candidate, "
                   "not an accepted arithmetic",
}

# The two lines this switch depends on. If the integration is rewritten so
# that the decision is made elsewhere, patching `use_gqa_in_sdpa` would change
# nothing while the record claimed expansion, so the switch refuses instead.
_EXPECTED_DECISION = "if not use_gqa_in_sdpa(attention_mask, key, value):"
_EXPECTED_EXPANSION = "key = repeat_kv(key, module.num_key_value_groups)"


def _assert_supported_source() -> None:
    from transformers.integrations import sdpa_attention

    source = " ".join(inspect.getsource(sdpa_attention.sdpa_attention_forward).split())
    for expected in (_EXPECTED_DECISION, _EXPECTED_EXPANSION):
        if " ".join(expected.split()) not in source:
            raise RuntimeError(
                "the attention-kernel switch patches `use_gqa_in_sdpa` inside "
                "`sdpa_attention_forward`, and the installed transformers no "
                f"longer contains `{expected}`. Refusing to run: the switch "
                "would report expansion while changing nothing."
            )


def _attention_modules(model) -> list:
    """Every module that would consult `use_gqa_in_sdpa`: those with KV groups."""
    return [m for m in model.modules()
            if getattr(m, "num_key_value_groups", 1) > 1]


@contextlib.contextmanager
def attention_kernel(model, kind: str):
    """Fix which KV form reaches SDPA for this model's attention, then undo it.

    Yields the record that belongs beside any number the forward produces,
    including a count of the attention calls the switch actually governed.
    """
    from transformers.integrations import sdpa_attention

    if kind not in ATTENTION_KINDS:
        raise ValueError(f"unknown attention kind {kind!r}; expected {ATTENTION_KINDS}")
    modules = _attention_modules(model)
    if not modules:
        raise ValueError(
            "no module on this model has num_key_value_groups > 1, so there is "
            "no grouped-query attention for the switch to govern"
        )
    _assert_supported_source()
    original = sdpa_attention.use_gqa_in_sdpa
    state: dict = {"depth": 0, "owner": threading.get_ident()}
    counts: dict = {"decisions_governed": 0, "expanded": 0, "left_to_library": 0,
                    "governed_modules": len(modules)}

    def _require_owner(where: str) -> None:
        if threading.get_ident() != state["owner"]:
            raise RuntimeError(
                f"attention_kernel({kind!r}) was entered on thread "
                f"{state['owner']} and {where} ran on {threading.get_ident()}; "
                "single-threaded forward execution only"
            )

    def use_gqa_in_sdpa(attention_mask, key, value):
        if state["depth"] <= 0:
            return original(attention_mask, key, value)
        _require_owner("the SDPA decision")
        counts["decisions_governed"] += 1
        if kind == "expanded_kv":
            counts["expanded"] += 1
            return False
        counts["left_to_library"] += 1
        return original(attention_mask, key, value)

    def _enter(_module, _args, _kwargs=None):
        _require_owner("an attention forward")
        state["depth"] += 1
        return None

    def _exit(_module, _args, _output=None):
        state["depth"] = max(0, state["depth"] - 1)
        return None

    sdpa_attention.use_gqa_in_sdpa = use_gqa_in_sdpa
    handles = []
    for module in modules:
        handles.append(module.register_forward_pre_hook(_enter))
        handles.append(module.register_forward_hook(_exit, always_call=True))
    try:
        yield {
            "kind": kind,
            "intent": ATTENTION_INTENT[kind],
            "counts": counts,
            "patched": "transformers.integrations.sdpa_attention.use_gqa_in_sdpa, "
                       "gated by this model's attention modules on the entering thread",
            "expansion_is_assumed_numerically_free": False,
            "modifies_checkpoint_or_deployment": False,
            "modifies_installed_packages": False,
        }
    finally:
        for handle in handles:
            handle.remove()
        sdpa_attention.use_gqa_in_sdpa = original
