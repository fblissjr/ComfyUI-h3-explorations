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


# --------------------------------------------------------------------------
# the manual-cast policy: BF16 storage, FP32 compute


def tiny_full_model(seed: int = 0):
    """A whole Qwen3-VL at the released shape ratios, scaled down.

    Grouped-query heads, interleaved M-RoPE, a 27-block tower with three
    DeepStack taps and a patch-merger: the parts whose dtype handling the
    policy has to get right. Weights are seeded and rounded to BF16-exact
    values so the same numbers can be loaded as FP32 and as BF16.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "rope_parameters": {"rope_type": "default", "rope_theta": 5000000,
                                "mrope_section": [4, 2, 2], "mrope_interleaved": True},
        },
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": 64, **RELEASED_VISION},
        image_token_id=5, video_token_id=6, vision_start_token_id=3,
        vision_end_token_id=4, tie_word_embeddings=False,
    )
    generator = torch.Generator().manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(config).eval()
    state = {
        k: (torch.randn(v.shape, generator=generator) * 0.02).to(torch.bfloat16).float()
        for k, v in model.state_dict().items()
    }
    model.load_state_dict(state, strict=True)
    return model


def tiny_batch(seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.tensor([[1, 2, 3, 5, 5, 5, 5, 4, 7, 8, 9]])
    return {
        "input_ids": input_ids,
        "mm_token_type_ids": (input_ids == 5).to(torch.int64),
        "pixel_values": torch.randn(16, 3 * 2 * 16 * 16, generator=generator),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


def _last_layer_state(model, batch) -> torch.Tensor:
    captured = []
    layers = model.model.language_model.layers
    handle = layers[-1].register_forward_hook(
        lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out).detach().clone())
    )
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"the tap fired {len(captured)} times")
    return captured[0]


def _functional_snapshot() -> dict:
    functional = torch.nn.functional
    return {name: getattr(functional, name)
            for name in ("linear", "embedding", "layer_norm", "conv3d")}


def check_bf16_store_policy() -> list[str]:
    """The manual-cast policy: load path, refusal, identity, leaks, restoration.

    The claim under test is the one that was nearly stated as a fact: BF16
    weights upcast per call give the same numbers as the same weights loaded
    in FP32. It is measured as bit identity at the last decoder layer on a
    real multimodal batch through `from_pretrained`, so the load-time
    `_keep_in_fp32_modules_strict` path is the one exercised, not a hand cast.
    """
    import tempfile

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    from h3_calibration_precision import (
        KEEP_IN_FP32_MODULES,
        PrecisionLeak,
        storage_dtype,
        storage_policy,
    )

    failures = []
    policy = "comfy_exact_bf16_store"
    cls = Qwen3VLForConditionalGeneration
    batch = tiny_batch()

    with tempfile.TemporaryDirectory(prefix="h3-tiny-qwen3vl-") as tmp:
        tiny_full_model().save_pretrained(tmp)
        fp32 = cls.from_pretrained(tmp, dtype=torch.float32).eval()
        attribute_before = vars(cls).get("_keep_in_fp32_modules_strict", "<inherited>")
        with storage_policy(cls, policy) as loading:
            bf16 = cls.from_pretrained(tmp, dtype=storage_dtype(policy)).eval()
        attribute_after = vars(cls).get("_keep_in_fp32_modules_strict", "<inherited>")
        # The library's own BF16 load, for the control: with the patch embed
        # kept in FP32 the un-policied model is inconsistent by construction
        # (`visual.dtype` reports float32, so FP32 activations reach BF16
        # LayerNorm weights and torch refuses), so the plain forward has to
        # come from a plain load.
        plain = cls.from_pretrained(tmp, dtype=torch.bfloat16).eval()
        if attribute_after != attribute_before:
            failures.append(f"storage_policy did not restore the class attribute: "
                            f"{attribute_before!r} -> {attribute_after!r}")
        if loading["keep_in_fp32_modules"] != list(KEEP_IN_FP32_MODULES):
            failures.append("storage_policy did not report the modules it kept in FP32")

    # The load did what the policy needs, and nothing more.
    patch = bf16.model.visual.patch_embed.proj.weight.dtype
    if patch != torch.float32:
        failures.append(f"patch_embed.proj.weight loaded as {patch}, not float32")
    if bf16.model.visual.dtype != torch.float32:
        failures.append(f"visual.dtype reports {bf16.model.visual.dtype}")
    others = {name: p.dtype for name, p in bf16.named_parameters()
              if not any(keep in name for keep in KEEP_IN_FP32_MODULES)}
    not_bf16 = {n: d for n, d in others.items() if d != torch.bfloat16}
    if not_bf16:
        failures.append(f"parameters outside the keep set are not bf16: {list(not_bf16)[:5]}")
    buffers = {n: b.dtype for n, b in bf16.named_buffers() if b.is_floating_point()}
    not_fp32 = {n: d for n, d in buffers.items() if d != torch.float32}
    if not_fp32:
        failures.append(f"floating buffers are not float32 under BF16 storage: {not_fp32}")

    # The tower alone is refused: the cast covers the language stack.
    try:
        with calibration_precision(bf16.model.visual, policy):
            pass
        failures.append("the policy accepted the vision tower alone")
    except ValueError:
        pass

    # Identity against the FP32-stored arm under comfy_exact, and a record
    # that says the cast actually happened.
    functional_before = _functional_snapshot()
    root_hooks_before = (len(bf16._forward_pre_hooks), len(bf16._forward_hooks))
    with calibration_precision(fp32, "comfy_exact"):
        reference = _last_layer_state(fp32, batch)
    with calibration_precision(bf16, policy) as record:
        candidate = _last_layer_state(bf16, batch)
        counts = dict(record["manual_cast_counts"])
    if candidate.dtype != torch.float32:
        failures.append(f"the manual-cast state is {candidate.dtype}, not float32")
    if not torch.equal(reference, candidate):
        delta = float((reference - candidate).abs().max())
        failures.append(f"BF16-stored / FP32-computed state is not bit-identical to the "
                        f"FP32-stored state: max abs delta {delta:.3e}")
    for op in ("linear", "embedding", "layer_norm"):
        if counts.get(op, 0) == 0:
            failures.append(f"the policy never cast an F.{op} weight; the gate did not open")
    if counts.get("conv3d", 0) != 0:
        failures.append("the patch-embed conv was cast at call time; it should have "
                        "loaded as float32")
    if counts.get("already_float32", 0) == 0:
        failures.append("no op saw the kept-FP32 patch embed")
    print(f"  bit-identical to the FP32-stored arm; casts {counts}")

    # The gate must open without the root forward: the sequential pipeline
    # calls submodules directly from traced subgraphs. A decoder layer called
    # on its own under the policy must compute, and in FP32.
    layer0 = bf16.model.language_model.layers[0]
    hidden = torch.randn(1, 11, 64)
    position_ids = torch.arange(11).view(1, 1, 11).expand(3, 1, 11)
    rotary = bf16.model.language_model.rotary_emb(hidden, position_ids)
    with calibration_precision(bf16, policy) as record:
        try:
            with torch.no_grad():
                direct = layer0(hidden, position_embeddings=rotary)
            direct = direct[0] if isinstance(direct, tuple) else direct
            if direct.dtype != torch.float32:
                failures.append(f"a directly called decoder layer returned {direct.dtype}")
        except RuntimeError as exc:
            failures.append(f"a directly called decoder layer failed under the policy: "
                            f"{str(exc).splitlines()[0][:100]}")
        if record["manual_cast_counts"].get("gated_modules", 0) == 0:
            failures.append("no modules were gated")

    # Restoration: the functional patches and hooks are gone.
    for name, fn in _functional_snapshot().items():
        if fn is not functional_before[name]:
            failures.append(f"torch.nn.functional.{name} was not restored")
    if (len(bf16._forward_pre_hooks), len(bf16._forward_hooks)) != root_hooks_before:
        failures.append("root hooks were not removed")
    sample = bf16.model.language_model.layers[0].mlp.down_proj
    if sample._forward_pre_hooks or sample._forward_hooks:
        failures.append("module gate hooks were not removed")
    # After the patches are gone, the kept-FP32 model cannot run under the
    # library's own functional layer: FP32 activations meet BF16 LayerNorm
    # weights and torch refuses with a mixed-dtype error. That refusal is the
    # proof the patches were removed; a PrecisionLeak here would mean the
    # gate was still installed.
    try:
        _last_layer_state(bf16, batch)
        failures.append("the kept-FP32 model ran without the policy, so a patch leaked")
    except PrecisionLeak:
        failures.append("the gate was still installed after the context exited")
    except RuntimeError:
        pass
    # The control that the policy changed something: the library's plain BF16
    # forward on a plain BF16 load differs from the manual-cast state.
    native = _last_layer_state(plain, batch)
    if native.dtype != torch.bfloat16:
        failures.append(f"the plain BF16 load runs at {native.dtype}")
    if torch.equal(native.float(), candidate):
        failures.append("the plain BF16 forward equals the manual-cast state, so the "
                        "policy changed nothing")

    # Leak controls. A downcast upstream of any parameterised op must stop
    # the run, at entry when it is the patch embed, and inside the forward
    # when it is anywhere else.
    leaked = bf16.model.language_model.layers[0].register_forward_pre_hook(
        lambda _m, args: (args[0].to(torch.bfloat16),) + tuple(args[1:])
    )
    try:
        with calibration_precision(bf16, policy):
            try:
                _last_layer_state(bf16, batch)
                failures.append("a BF16 activation reached F.linear without a PrecisionLeak")
            except PrecisionLeak:
                pass
    finally:
        leaked.remove()
    for name, fn in _functional_snapshot().items():
        if fn is not functional_before[name]:
            failures.append(f"torch.nn.functional.{name} was not restored after the leak")
    bf16.model.visual.patch_embed.to(torch.bfloat16)
    try:
        with calibration_precision(bf16, policy):
            pass
        failures.append("a BF16 patch embed was accepted at entry")
    except PrecisionLeak:
        pass
    return failures


def main() -> int:
    failures: list[str] = []
    for name, arm in (("restoration", check_restoration),
                      ("raising forward", check_raising_forward),
                      ("instance scoping", check_instance_scoping),
                      ("offload dispatch compatibility",
                       check_offload_dispatch_compatibility),
                      ("source guard", check_source_guard),
                      ("corrupt-tap red control", check_corrupt_tap_control),
                      ("bf16-store manual cast", check_bf16_store_policy)):
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
          "its red control moves the output, and the BF16-stored manual-cast "
          "arm is bit-identical to FP32 storage and refuses a downcast")
    return 0


if __name__ == "__main__":
    sys.exit(main())
