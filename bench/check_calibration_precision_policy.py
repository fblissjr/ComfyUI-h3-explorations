#!/usr/bin/env python3
"""Hold the calibration precision policies to what they promise.

`bench/h3_calibration_precision.py` substitutes a reduction inside a third-party
model's forward. That is a sharp tool: it is scoped to one instance, it restores
everything on exit, and it refuses to run if the expression it substitutes for is
not there. Each of those is a promise, and a promise nobody has watched break is
not evidence.

Four arms, all on a reduced-width Qwen3-VL so this runs in seconds with no
released weights:

1. **Restoration.** Every mutable thing the context manager touches -- the
   module-level helper, the `pos_embed` forward, the position-embedding dtype,
   and the instance's hook count -- is captured before and after and must come
   back identical, including when the body raises.
2. **Instance scoping.** A second Qwen3-VL in the same process must be
   unaffected while the policy is active on the first. A module-level patch that
   changed both would be invisible in any single-model test.
3. **Source guard.** With the expected reduction absent from the installed
   source, `comfy_exact` must refuse rather than substitute into whatever is
   there instead.
4. **Red control.** `comfy_exact_corrupt_tap` must move the tower output away
   from `comfy_exact`; if it does not, the substitution is not reaching the
   computation and every parity number above it is unfalsifiable.
5. **Raising forward.** A forward that raises must still close the gate, which
   is what `always_call=True` on the cleanup hook is for. Without it a caught
   exception leaves the gate open inside a still-open context and the next
   unrelated forward silently runs under this policy.

Run it with either virtualenv's python. CPU only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import h3_calibration_precision as policy_module  # noqa: E402
from h3_calibration_precision import calibration_precision  # noqa: E402

RELEASED_VISION = {
    "depth": 27, "patch_size": 16, "temporal_patch_size": 2,
    "spatial_merge_size": 2, "deepstack_visual_indexes": [8, 16, 24],
    "num_position_embeddings": 2304,
}


def tiny_vision(dtype: torch.dtype, seed: int = 0):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(
        hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
        **RELEASED_VISION,
    )
    generator = torch.Generator().manual_seed(seed)
    model = Qwen3VLVisionModel(config).to(dtype).eval()
    model.load_state_dict(
        {k: (torch.randn(v.shape, generator=generator) * 0.02).to(dtype)
         for k, v in model.state_dict().items()},
        strict=True,
    )
    return model


def _snapshot(model) -> dict:
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    return {
        "helper": modeling_qwen3_vl.get_vision_interpolation_indices_and_weights,
        "pos_forward": model.pos_embed.forward,
        "pos_dtype": model.pos_embed.weight.dtype,
        "pre_hooks": len(model._forward_pre_hooks),
        "hooks": len(model._forward_hooks),
    }


def check_restoration() -> list[str]:
    failures = []
    model = tiny_vision(torch.float32)
    before = _snapshot(model)
    with calibration_precision(model, "comfy_exact"):
        during = _snapshot(model)
        if during["helper"] is before["helper"]:
            failures.append("the helper was not wrapped while the policy was active")
        if during["pos_forward"] is before["pos_forward"]:
            failures.append("pos_embed.forward was not substituted")
        if during["pos_dtype"] != torch.bfloat16:
            failures.append(f"pos_embed dtype is {during['pos_dtype']}, expected bfloat16")
    after = _snapshot(model)
    for key in before:
        if after[key] is not before[key] and after[key] != before[key]:
            failures.append(f"{key} was not restored: {before[key]} -> {after[key]}")

    # And when the body raises. A context manager that only restores on the
    # happy path leaves a poisoned model behind exactly when someone is
    # debugging, which is the worst moment for it.
    try:
        with calibration_precision(model, "comfy_exact"):
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass
    raised = _snapshot(model)
    for key in before:
        if raised[key] is not before[key] and raised[key] != before[key]:
            failures.append(f"{key} was not restored after an exception")
    return failures


def check_raising_forward() -> list[str]:
    """A forward that raises must still close the gate.

    The cleanup hook is registered with `always_call=True` precisely for this.
    Without it a caught forward exception leaves the gate open inside a
    still-open context, and the next unrelated forward runs under this policy
    with nothing saying so -- a silent wrong-policy run, which is the worst
    shape of failure this module can have.
    """
    failures = []
    model = tiny_vision(torch.float32)
    patches = torch.randn(64, 3 * 2 * 16 * 16)
    grid = torch.tensor([[1, 8, 8]])

    original_blocks = model.blocks
    with calibration_precision(model, "comfy_exact") as record:
        model.blocks = _Exploding()
        try:
            with torch.no_grad():
                model(patches, grid_thw=grid, return_dict=True)
        except _DeliberateForwardError:
            pass
        finally:
            model.blocks = original_blocks
        # The gate must be closed again even though the forward never returned.
        indices, weights = _probe_helper_state(grid)
        if indices is None:
            failures.append("could not read the helper state back")
        elif indices.shape[-1] == 1:
            failures.append(
                "the gate was left open after a raising forward: the helper is "
                "still returning the single trivial tap"
            )
        # And the policy must still work on the next honest forward.
        with torch.no_grad():
            model(patches, grid_thw=grid, return_dict=True)
    if record["modifies_installed_packages"] is not False:
        failures.append("the record claims it modifies installed packages")
    return failures


class _DeliberateForwardError(RuntimeError):
    pass


class _Exploding(torch.nn.Module):
    """Stands in for the vision blocks and raises once reached."""

    def __iter__(self):
        raise _DeliberateForwardError("deliberate failure inside the vision forward")

    def __len__(self):
        return 0


def _probe_helper_state(grid):
    """Ask the currently bound helper what it returns outside a forward."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    try:
        return modeling_qwen3_vl.get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=False,
            spatial_merge_size=2,
        )
    except Exception:
        return None, None


def check_instance_scoping() -> list[str]:
    """A second model in the same process must be untouched."""
    failures = []
    first, second = tiny_vision(torch.float32), tiny_vision(torch.float32)
    patches = torch.randn(64, 3 * 2 * 16 * 16)
    grid = torch.tensor([[1, 8, 8]])
    with torch.no_grad():
        baseline = second(patches, grid_thw=grid, return_dict=True).pooler_output.clone()
    with calibration_precision(first, "comfy_exact"):
        with torch.no_grad():
            during = second(patches, grid_thw=grid, return_dict=True).pooler_output
            changed = first(patches, grid_thw=grid, return_dict=True).pooler_output
        if not torch.equal(baseline, during):
            failures.append(
                "the policy changed a second Qwen3-VL instance in the same "
                "process; the module-level wrap is not scoped"
            )
    with torch.no_grad():
        plain = first(patches, grid_thw=grid, return_dict=True).pooler_output
    if torch.equal(plain, changed):
        failures.append(
            "the policy did not change the instance it was applied to; the "
            "scoping gate is closed when it should be open"
        )
    return failures


def check_offload_dispatch_compatibility() -> list[str]:
    """The substituted forward must survive `compressed_tensors`' offload hooks.

    `compressed_tensors.offload.module.offload_module` reads
    `module.forward.__func__` when it installs offload hooks, so a forward
    replaced with a plain function breaks the sequential pipeline before the
    first batch. That is how the first real Gate 2 run failed, and nothing in
    the numerical checks above could have caught it: the substitution is
    numerically perfect and structurally wrong.
    """
    failures = []
    model = tiny_vision(torch.float32)
    with calibration_precision(model, "comfy_exact"):
        forward = model.pos_embed.forward
        if not hasattr(forward, "__func__"):
            failures.append(
                "the substituted pos_embed.forward is not a bound method, so "
                "compressed_tensors' offload dispatch cannot install its hooks"
            )
        if getattr(forward, "__self__", None) is not model.pos_embed:
            failures.append("the substituted forward is bound to the wrong object")
    if "forward" in model.pos_embed.__dict__:
        failures.append(
            "a `forward` instance attribute was left behind, shadowing the "
            "class method for the rest of this model's life"
        )
    return failures


def check_source_guard() -> list[str]:
    """With the expected expression gone, comfy_exact must refuse."""
    failures = []
    original = policy_module._EXPECTED_REDUCTION
    model = tiny_vision(torch.float32)
    policy_module._EXPECTED_REDUCTION = (
        "(self.pos_embed(interp_indices) * interp_weights).mean(1)  # not present"
    )
    try:
        with calibration_precision(model, "comfy_exact"):
            failures.append(
                "comfy_exact ran with its substituted expression absent from the "
                "installed source; the guard does not fail closed"
            )
    except RuntimeError as exc:
        if "Refusing to run" not in str(exc):
            failures.append(f"the guard raised something else: {exc}")
        else:
            print(f"  guard refused: {str(exc).split('Refusing to run: ')[-1][:90]}")
    finally:
        policy_module._EXPECTED_REDUCTION = original

    # And it must not be refusing unconditionally.
    try:
        with calibration_precision(model, "comfy_exact"):
            pass
    except RuntimeError as exc:
        failures.append(f"the guard refuses the installed source too: {exc}")
    return failures


def check_corrupt_tap_control() -> list[str]:
    failures = []
    model = tiny_vision(torch.float32)
    patches = torch.randn(64, 3 * 2 * 16 * 16)
    grid = torch.tensor([[1, 8, 8]])
    outputs = {}
    for policy in ("comfy_exact", "comfy_exact_corrupt_tap"):
        with calibration_precision(model, policy):
            with torch.no_grad():
                outputs[policy] = model(patches, grid_thw=grid,
                                        return_dict=True).pooler_output.clone()
    if torch.equal(outputs["comfy_exact"], outputs["comfy_exact_corrupt_tap"]):
        failures.append(
            "scaling one interpolation tap changed nothing; the substitution is "
            "not reaching the computation"
        )
    else:
        delta = float((outputs["comfy_exact"].double()
                       - outputs["comfy_exact_corrupt_tap"].double()).abs().max())
        print(f"  corrupt-tap control moved the tower output by {delta:.4g}")
    return failures


def main() -> int:
    failures: list[str] = []
    for name, arm in (("restoration", check_restoration),
                      ("raising forward", check_raising_forward),
                      ("instance scoping", check_instance_scoping),
                      ("offload dispatch compatibility",
                       check_offload_dispatch_compatibility),
                      ("source guard", check_source_guard),
                      ("corrupt-tap red control", check_corrupt_tap_control)):
        print(f"{name}:")
        found = arm()
        failures.extend(found)
        print(f"  {'ok' if not found else found}")
    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: the policy is scoped to one instance, restores everything it "
          "touched including after an exception, refuses an unrecognised source, "
          "and its red control moves the output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
