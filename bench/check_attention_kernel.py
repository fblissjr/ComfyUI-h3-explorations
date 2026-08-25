#!/usr/bin/env python3
"""Hold the attention-kernel switch to what it promises.

`bench/h3_attention_kernel.py` decides whether SDPA sees grouped-query or
expanded key/value heads. Each promise is watched failing here, on the tiny
full Qwen3-VL from the precision-policy check, CPU only:

1. **It reaches the call.** What arrives at
   `torch.nn.functional.scaled_dot_product_attention` is observed directly:
   under `grouped_query` the key carries the model's KV head count and
   `enable_gqa=True`; under `expanded_kv` the key carries the full head count
   and no `enable_gqa`. A switch that only changed a flag in its own record
   would fail this.
2. **It governs every attention call and counts them.** The record's count
   equals the number of decoder layers times the forwards run.
3. **Instance scoping.** A second model in the same process sees the library's
   own decision while the switch is active on the first.
4. **Restoration.** The module-level helper and the hook counts come back
   identical, including when the forward raises.
5. **Source guard.** With the decision line absent from the installed source,
   the switch refuses.
6. **Refusal.** A model with no grouped-query attention is refused.

What this check does not claim: that grouped-query and expanded-KV outputs
agree. That is measured on released weights by the layer-49 comparison, and
the tiny-model delta is printed here only as an observation.

Run it with either virtualenv's python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import h3_attention_kernel as kernel_module  # noqa: E402
from check_calibration_precision_policy import tiny_batch, tiny_full_model  # noqa: E402
from h3_attention_kernel import attention_kernel  # noqa: E402


class _SdpaObserver:
    """Records the KV head count and kwargs of every SDPA call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._original = torch.nn.functional.scaled_dot_product_attention

    def __enter__(self):
        observer = self

        def observed(query, key, value, *args, **kwargs):
            observer.calls.append({"q_heads": int(query.shape[1]),
                                   "kv_heads": int(key.shape[1]),
                                   "enable_gqa": bool(kwargs.get("enable_gqa", False))})
            return observer._original(query, key, value, *args, **kwargs)

        torch.nn.functional.scaled_dot_product_attention = observed
        return self

    def __exit__(self, *_exc):
        torch.nn.functional.scaled_dot_product_attention = self._original
        return False


def _run(model, batch) -> torch.Tensor:
    captured = []
    handle = model.model.language_model.layers[-1].register_forward_hook(
        lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out).detach().clone())
    )
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    finally:
        handle.remove()
    return captured[0]


def _text_calls(calls: list[dict], heads: int) -> list[dict]:
    """The language stack's calls: the vision tower has no KV groups."""
    return [c for c in calls if c["q_heads"] == heads]


def check_reaches_the_call() -> list[str]:
    failures = []
    model = tiny_full_model()
    batch = tiny_batch()
    heads = model.config.text_config.num_attention_heads
    kv_heads = model.config.text_config.num_key_value_heads
    layers = model.config.text_config.num_hidden_layers

    with _SdpaObserver() as seen, attention_kernel(model, "grouped_query") as record:
        grouped = _run(model, batch)
    text = _text_calls(seen.calls, heads)
    if len(text) != layers:
        failures.append(f"grouped_query: {len(text)} language SDPA calls, expected {layers}")
    if any(c["kv_heads"] != kv_heads or not c["enable_gqa"] for c in text):
        failures.append(f"grouped_query: SDPA did not see {kv_heads} KV heads with enable_gqa: {text}")
    if record["counts"]["decisions_governed"] != layers or record["counts"]["left_to_library"] != layers:
        failures.append(f"grouped_query counts wrong: {record['counts']}")

    with _SdpaObserver() as seen, attention_kernel(model, "expanded_kv") as record:
        expanded = _run(model, batch)
    text = _text_calls(seen.calls, heads)
    if len(text) != layers:
        failures.append(f"expanded_kv: {len(text)} language SDPA calls, expected {layers}")
    if any(c["kv_heads"] != heads or c["enable_gqa"] for c in text):
        failures.append(f"expanded_kv: SDPA did not see {heads} KV heads without enable_gqa: {text}")
    if record["counts"]["expanded"] != layers:
        failures.append(f"expanded_kv counts wrong: {record['counts']}")

    delta = float((grouped - expanded).abs().max())
    print(f"  tiny-model grouped-vs-expanded max abs delta {delta:.3e} "
          f"(an observation; the released-weight measurement is the claim)")
    return failures


def check_instance_scoping() -> list[str]:
    failures = []
    first, second = tiny_full_model(), tiny_full_model(seed=1)
    batch = tiny_batch()
    heads = first.config.text_config.num_attention_heads
    with attention_kernel(first, "expanded_kv"):
        with _SdpaObserver() as seen:
            _run(second, batch)
        text = _text_calls(seen.calls, heads)
        if any(c["kv_heads"] == heads for c in text):
            failures.append("the switch on the first model expanded the second model's KV")
    return failures


def check_restoration() -> list[str]:
    from transformers.integrations import sdpa_attention

    failures = []
    model = tiny_full_model()
    batch = tiny_batch()
    attention = [m for m in model.modules() if getattr(m, "num_key_value_groups", 1) > 1]
    before = (sdpa_attention.use_gqa_in_sdpa,
              [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in attention])
    with attention_kernel(model, "expanded_kv"):
        if sdpa_attention.use_gqa_in_sdpa is before[0]:
            failures.append("the helper was not patched while the switch was active")
    after = (sdpa_attention.use_gqa_in_sdpa,
             [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in attention])
    if after != before:
        failures.append("helper or hooks not restored")

    raising = attention[0].register_forward_pre_hook(
        lambda _m, _a: (_ for _ in ()).throw(RuntimeError("deliberate"))
    )
    try:
        with attention_kernel(model, "expanded_kv"):
            try:
                _run(model, batch)
            except RuntimeError:
                pass
    finally:
        raising.remove()
    after = (sdpa_attention.use_gqa_in_sdpa,
             [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in attention])
    if after != before:
        failures.append("helper or hooks not restored after a raising forward")
    return failures


def check_source_guard() -> list[str]:
    failures = []
    model = tiny_full_model()
    saved = kernel_module._EXPECTED_DECISION
    kernel_module._EXPECTED_DECISION = "if not this_line_does_not_exist(attention_mask):"
    try:
        try:
            with attention_kernel(model, "expanded_kv"):
                failures.append("the switch ran against a source without the decision line")
        except RuntimeError as exc:
            if "Refusing" not in str(exc):
                failures.append(f"the guard raised something else: {exc}")
    finally:
        kernel_module._EXPECTED_DECISION = saved
    return failures


def check_refusal() -> list[str]:
    failures = []
    model = tiny_full_model()
    try:
        with attention_kernel(model.model.visual, "expanded_kv"):
            failures.append("the vision tower, which has no KV groups, was accepted")
    except ValueError:
        pass
    try:
        with attention_kernel(model, "flash_please"):
            failures.append("an unknown kind was accepted")
    except ValueError:
        pass
    return failures


def main() -> int:
    failures: list[str] = []
    for name, arm in (("reaches the call", check_reaches_the_call),
                      ("instance scoping", check_instance_scoping),
                      ("restoration", check_restoration),
                      ("source guard", check_source_guard),
                      ("refusal", check_refusal)):
        print(f"{name}:")
        found = arm()
        failures.extend(found)
        print(f"  {'ok' if not found else found}")
    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: the switch changes what reaches SDPA, governs every language "
          "attention call, is scoped to one model, restores itself, and refuses "
          "an unrecognised source or a model without grouped-query attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
