#!/usr/bin/env python3
"""The calibration-only precision policies, and what each one claims to model.

Gate 1B of [`active_plan.md`]. Deployed ComfyUI does not run the Qwen3-VL vision
tower at one dtype. It stores the released weights in BF16, casts each
`manual_cast` linear to the FP32 activation dtype, and then performs the
position-embedding lookup and its interpolation at the *stored* BF16 dtype,
because `Qwen35VisionModel.fast_pos_embed_interpolate` calls `ops.Embedding`
with no `out_dtype` and builds its bilinear coefficients at
`self.pos_embed.weight.dtype`. Plain Transformers picks one dtype for both, so
neither plain arm reproduces the combination -- measured in
`canonical/2026-08-24_transformers_comfy_parity.md`.

This module makes the split an explicit, named, revertible policy rather than a
property of a dtype flag:

| policy | coefficients | reduction | active linears |
|---|---|---|---|
| `float32` | FP32 | `.sum(1)` | FP32 |
| `bfloat16_native` | FP32, as the library returns them | `.sum(1)` | BF16 |
| `bfloat16` | BF16 | `.sum(1)` | BF16 |
| `hybrid` | BF16 | `.sum(1)` | FP32 |
| `comfy_exact` | BF16 | ComfyUI's four-term add | FP32 |
| `comfy_exact_corrupt_tap` | BF16, one tap scaled | four-term add | FP32 |
| `hybrid_fp32_posembed` | FP32 | `.sum(1)` | FP32 |
| `hybrid_bf16_linear` | BF16 | `.sum(1)` | BF16 |

**`bfloat16_native` and `bfloat16` are different arms and the distinction is
load-bearing.** Transformers' helper computes the interpolation coefficients in
FP32 whatever dtype the model is, so a plain BF16 model rounds once, after the
weighted sum. Casting the coefficients to BF16 first is a *proposed* behaviour
that matches ComfyUI, not something the library does. Reporting the second as
"Transformers BF16" would attribute a choice made here to the library.

The last two are single-axis reverts of the hybrid dtype split, used as
controls. They are *constructed from the hybrid path* rather than aliased to the
plain arms, so each reverts exactly one half. That they then come out
bit-identical to the corresponding plain arm is asserted, not assumed: if the
switch had a side effect, that assertion is where it shows. `hybrid` plays the
same role for `comfy_exact`, reverting only the reduction order, and
`comfy_exact_corrupt_tap` is its red control.

**Nothing here modifies a checkpoint, a saved config, the deployed encoder, its
symlink, or any ComfyUI node.** The position-embedding weight is cast in memory
only, and the cast is exact: the values came from a BF16 checkpoint, so
FP32-to-BF16 is a round trip, not a quantization. The interpolation
coefficients are redirected by wrapping the *name as bound inside the Qwen3-VL
modeling module*, so no other model's vision path is touched and the real
transformers helper still computes the coefficients.

Importable from either virtualenv: pure `torch` and `transformers`, no ComfyUI
and no `llmcompressor`.
"""

from __future__ import annotations

import contextlib
import threading
import types

import torch

POLICIES = (
    "float32",
    "bfloat16_native",
    "bfloat16",
    "hybrid",
    "comfy_exact",
    "comfy_exact_corrupt_tap",
    "hybrid_fp32_posembed",
    "hybrid_bf16_linear",
    "comfy_exact_bf16_store",
)

# `comfy_exact_bf16_store` is `comfy_exact` with the deployed *storage*
# arrangement as well as the deployed arithmetic: the weights stay BF16 where
# they are stored and offloaded, and every parameterised op computes with a
# transient FP32 copy of its weight -- ComfyUI's `manual_cast`, done at the
# functional layer so that it reaches the computation under any loader.
# Instance- and class-level `forward` overrides do not: Accelerate's device
# hooks and compressed-tensors' offload wrappers both capture the original
# `forward` when they wrap a module, so a later override is never called.
# `torch.nn.functional.linear` and friends are what every wrapped forward
# ends up calling, and they are patched here only while this model's own
# forward is running.
#
# Two places in transformers downcast the vision input to the tower's dtype
# before any hook can see it: `get_image_features` casts `pixel_values` to
# `visual.dtype`, and `Qwen3VLVisionPatchEmbed.forward` casts to its conv
# weight's dtype. Both key off the patch-embed weight, and `visual.dtype` is
# read from the first floating parameter, which is that weight. Keeping that
# one module in FP32 at load time makes both casts no-ops. The policy refuses
# to run unless it finds them so.
MANUAL_CAST_POLICIES = ("comfy_exact_bf16_store",)
KEEP_IN_FP32_MODULES = ("visual.patch_embed",)

# What each policy claims to model, quoted back into every report so a number
# cannot be read without the claim attached.
POLICY_INTENT = {
    "float32": "plain Transformers FP32; a control, not the deployed configuration",
    "bfloat16_native": "plain Transformers BF16 exactly as the library runs it: "
                       "the model is BF16 but the interpolation coefficients "
                       "come back from the helper in FP32 and are only rounded "
                       "once, after the weighted sum. This is the honest "
                       "plain-BF16 control",
    "bfloat16": "BF16 model with the interpolation coefficients ALSO cast to "
                "BF16. This is a proposed arm, not what plain Transformers "
                "does, and must not be reported as generic Transformers BF16",
    "hybrid": "BF16 position interpolation with FP32 active compute, reduced "
              "with transformers' own `.sum(1)`. Also the reduction-order "
              "control for comfy_exact",
    "comfy_exact": "BF16 position interpolation reduced in ComfyUI's explicit "
                   "four-term order, with FP32 active compute; reproduces the "
                   "deployed ComfyUI position embedding bit-for-bit",
    "comfy_exact_corrupt_tap": "red control: comfy_exact with one interpolation "
                               "tap scaled, so a comparison blind to it is "
                               "blind to a real substitution defect",
    "hybrid_fp32_posembed": "control: the hybrid arm with the position "
                            "interpolation reverted to FP32",
    "hybrid_bf16_linear": "control: the hybrid arm with the active linears and "
                          "residuals reverted to BF16",
    "comfy_exact_bf16_store": "comfy_exact arithmetic over BF16-stored weights: "
                              "every parameterised op computes with a transient "
                              "FP32 copy of its weight, as ComfyUI's manual_cast "
                              "does, while the stored and offloaded weights stay "
                              "BF16. The patch-embed conv is kept FP32 at load so "
                              "the vision input is never downcast",
}

_COMPUTE_DTYPE = {
    "float32": torch.float32,
    "bfloat16_native": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "hybrid": torch.float32,
    "comfy_exact": torch.float32,
    "comfy_exact_corrupt_tap": torch.float32,
    "hybrid_fp32_posembed": torch.float32,
    "hybrid_bf16_linear": torch.bfloat16,
    "comfy_exact_bf16_store": torch.float32,
}

# The dtype the checkpoint is loaded and stored in. It equals the compute
# dtype for every policy except the manual-cast one, where storage is BF16
# and compute is FP32 -- the split that lets the 64-layer model sit in host
# memory at half the size while the active arithmetic is unchanged.
_STORAGE_DTYPE = {policy: dtype for policy, dtype in _COMPUTE_DTYPE.items()}
_STORAGE_DTYPE["comfy_exact_bf16_store"] = torch.bfloat16

# `None` means "leave the coefficients at whatever the library returns", which
# is the only way to express native Transformers behaviour: its helper computes
# them in FP32 regardless of the model dtype.
_POSITION_DTYPE = {
    "float32": torch.float32,
    "bfloat16_native": None,
    "bfloat16": torch.bfloat16,
    "hybrid": torch.bfloat16,
    "comfy_exact": torch.bfloat16,
    "comfy_exact_corrupt_tap": torch.bfloat16,
    "hybrid_fp32_posembed": torch.float32,
    "hybrid_bf16_linear": torch.bfloat16,
    "comfy_exact_bf16_store": torch.bfloat16,
}


def compute_dtype(policy: str) -> torch.dtype:
    """The dtype the linears and residuals compute in.

    For every policy but the manual-cast one this is also the dtype the model
    loads in; `storage_dtype` is the one to pass to `from_pretrained`.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; expected {POLICIES}")
    return _COMPUTE_DTYPE[policy]


def storage_dtype(policy: str) -> torch.dtype:
    """The dtype to load and store the checkpoint in."""
    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; expected {POLICIES}")
    return _STORAGE_DTYPE[policy]


class PrecisionLeak(RuntimeError):
    """An activation reached a parameterised op below the compute dtype.

    Raised rather than cast: a silent upcast here would hide exactly the
    defect the policy exists to exclude -- a downcast somewhere upstream that
    the FP32 arithmetic then computes on faithfully.
    """


@contextlib.contextmanager
def storage_policy(model_cls, policy: str):
    """Make `from_pretrained` load the way the policy stores.

    Wrap the `from_pretrained` call in this. For the manual-cast policy it
    sets the class's `_keep_in_fp32_modules_strict` so the patch-embed conv
    loads as FP32 under a BF16 `dtype`; for every other policy it changes
    nothing. The class attribute is restored on exit either way, because it is
    process-wide and a later load in the same process must not inherit it.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; expected {POLICIES}")
    if policy not in MANUAL_CAST_POLICIES:
        yield {"keep_in_fp32_modules": []}
        return
    attribute = "_keep_in_fp32_modules_strict"
    had = attribute in vars(model_cls)
    previous = getattr(model_cls, attribute, None)
    setattr(model_cls, attribute, list(KEEP_IN_FP32_MODULES))
    try:
        yield {"keep_in_fp32_modules": list(KEEP_IN_FP32_MODULES)}
    finally:
        if had:
            setattr(model_cls, attribute, previous)
        else:
            delattr(model_cls, attribute)


def position_dtype(policy: str) -> torch.dtype:
    """The dtype the position-embedding lookup and interpolation run at."""
    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; expected {POLICIES}")
    return _POSITION_DTYPE[policy]


def _vision_module(model):
    """The Qwen3-VL vision tower, whether given a whole model or the tower."""
    for path in (("model", "visual"), ("visual",)):
        node = model
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        else:
            return node
    if hasattr(model, "pos_embed"):
        return model
    raise ValueError("could not find the Qwen3-VL vision tower on this object")


# The exact expression the `comfy_exact` policy substitutes into. If a future
# transformers release changes this line, the substitution would silently apply
# to something else, so the policy refuses to run rather than guess.
_EXPECTED_REDUCTION = "(self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)"


def _assert_supported_source() -> str:
    """Fail closed if the vision forward no longer contains what we substitute.

    Branch on the observable, not on a version string: a backport or a fork can
    carry any version number, and what matters is whether the reduction this
    policy replaces is still written the way it is written here.
    """
    import inspect

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    source = inspect.getsource(Qwen3VLVisionModel.forward)
    normalised = " ".join(source.split())
    if " ".join(_EXPECTED_REDUCTION.split()) not in normalised:
        raise RuntimeError(
            "the comfy_exact policy substitutes for "
            f"`{_EXPECTED_REDUCTION}` inside Qwen3VLVisionModel.forward, and "
            "that expression is not in the installed transformers. Refusing to "
            "run: the substitution would be applied to a different computation "
            "and the resulting numbers would be measuring something nobody "
            "chose."
        )
    return _EXPECTED_REDUCTION


@contextlib.contextmanager
def calibration_precision(model, policy: str):
    """Apply one named policy for the duration of a forward, then undo it.

    Yields the record that belongs beside any number the forward produces.

    Everything is restored on the way out -- the position-embedding dtype, the
    module-level helper, the `pos_embed` forward and the scoping hooks -- so a
    caller that reuses a model cannot silently inherit the previous arm's
    policy, which is the failure mode that would make two arms agree for the
    wrong reason.

    **Scoped to this vision instance.** The helper transformers calls is a
    module-level function, so wrapping it would otherwise change every Qwen3-VL
    in the process. The wrapper therefore only alters behaviour while this
    instance's own forward is running, gated by hooks on that instance.
    """
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; expected {POLICIES}")
    vision = _vision_module(model)
    wanted = _POSITION_DTYPE[policy]
    exact = policy in ("comfy_exact", "comfy_exact_corrupt_tap") or policy in MANUAL_CAST_POLICIES
    manual_cast = policy in MANUAL_CAST_POLICIES
    reduction_source = _assert_supported_source() if exact else None
    original = vision.pos_embed.weight.dtype

    if manual_cast:
        # The whole model, not the tower: the gate has to open for the language
        # stack's linears too, and the two vision-input downcasts are decided
        # by attributes only a whole-model load sets.
        if vision is model:
            raise ValueError(
                f"{policy!r} needs the whole Qwen3-VL model, not the vision tower: "
                "its manual cast covers the language stack and it checks the "
                "load-time dtype of the patch embed"
            )
        patch_dtype = vision.patch_embed.proj.weight.dtype
        if patch_dtype != torch.float32:
            raise PrecisionLeak(
                f"{policy!r}: visual.patch_embed.proj.weight is {patch_dtype}, not "
                "float32. transformers casts pixel_values to that dtype before "
                "any hook can see them, so the vision input would be downcast. "
                "Load under `storage_policy(model_cls, policy)`."
            )
        reported = vision.dtype
        if reported != torch.float32:
            raise PrecisionLeak(
                f"{policy!r}: visual.dtype reports {reported}; get_image_features "
                "casts pixel_values to it. The patch embed must be the first "
                "floating parameter of the tower and must be float32."
            )

    # Exactness check, not decoration. The released values are BF16, so casting
    # to BF16 must be a round trip. If it ever is not, the source is not what
    # this policy assumes and the number would be measuring the cast instead.
    reference = vision.pos_embed.weight.detach()
    round_trip_exact = (
        True if wanted is None
        else bool(torch.equal(reference.to(wanted).to(original), reference))
    )

    original_helper = modeling_qwen3_vl.get_vision_interpolation_indices_and_weights
    original_pos_forward = vision.pos_embed.forward
    functional = torch.nn.functional
    _original_embedding = functional.embedding
    _original_linear = functional.linear
    _original_layer_norm = functional.layer_norm
    _original_conv3d = functional.conv3d
    # Whether `forward` was already an instance attribute decides how to put it
    # back: assigning the bound method would otherwise leave a permanent
    # instance attribute where the class method used to be found.
    had_instance_forward = "forward" in vision.pos_embed.__dict__
    # `depth`, not a boolean: a nested forward on the same instance would
    # otherwise clear the gate on the inner exit and silently run the outer
    # forward under the library's own behaviour. `owner` makes concurrent use
    # loud rather than silently wrong -- the helper this wraps is module-level,
    # so a forward on another thread would read a gate it does not own.
    state: dict = {"depth": 0, "owner": threading.get_ident()}

    def _require_owner(where: str) -> None:
        if threading.get_ident() != state["owner"]:
            raise RuntimeError(
                f"calibration_precision({policy!r}) was entered on thread "
                f"{state['owner']} and {where} ran on {threading.get_ident()}. "
                "This policy gates a module-level helper and supports "
                "single-threaded forward execution only; running it "
                "concurrently would apply the wrong policy to one of the "
                "forwards without saying so."
            )

    def helper(*args, **kwargs):
        indices, weights = original_helper(*args, **kwargs)
        if state["depth"] <= 0:
            return indices, weights
        _require_owner("the interpolation helper")
        if wanted is not None:
            weights = weights.to(wanted)
        if not exact:
            return indices, weights
        # Hand the real indices and weights to the substitute below and give the
        # caller a single trivial tap, so the model's own
        # `(pos_embed(i) * w[:, :, None]).sum(1)` reduces to exactly what the
        # substitute returns. Multiplying by one and summing a single term is
        # exact at BF16, so this replaces the reduction and nothing else in the
        # forward.
        state["indices"], state["weights"] = indices, weights
        rows = indices.shape[0]
        return (torch.zeros(rows, 1, dtype=indices.dtype, device=indices.device),
                torch.ones(rows, 1, dtype=weights.dtype, device=weights.device))

    def exact_pos_embed(_self, dummy_indices):  # noqa: D401
        """ComfyUI's four-term reduction, over transformers' own coefficients.

        The indices and the BF16 weights were measured identical between the two
        implementations on every grid tried, so nothing about the interpolation
        is restated here -- only the order the four taps are added in, which
        `probe_position_embedding_parity.py` isolated as the entire remaining
        difference: `comfy/text_encoders/qwen35.py` writes
        `pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]`,
        transformers writes `.sum(1)`, and at BF16 those are not the same number
        on some grids.
        """
        if state["depth"] <= 0 or "indices" not in state:
            return original_pos_forward(dummy_indices)
        indices, weights = state["indices"], state["weights"]
        table = vision.pos_embed.weight
        # The unpatched embedding, deliberately: under the manual-cast policy
        # `F.embedding` upcasts its result, and the four-term reduction below
        # has to stay at BF16 to reproduce ComfyUI's.
        taps = _original_embedding(indices, table) * weights[:, :, None]
        if policy == "comfy_exact_corrupt_tap":
            # The red control. One tap is scaled, so a comparison that cannot
            # see this cannot see a real substitution defect either.
            taps = taps.clone()
            taps[:, 0] = taps[:, 0] * 1.5
        reduced = ((taps[:, 0] + taps[:, 1]) + taps[:, 2]) + taps[:, 3]
        return reduced.unsqueeze(1)

    def _enter(_module, _args, _kwargs=None):
        _require_owner("the vision forward")
        state["depth"] += 1
        return None

    def _exit(_module, _args, _output=None):
        # Registered with `always_call=True`, so this runs even when the
        # forward raises. Without that, a caught exception would leave the gate
        # open inside a still-open context and the next unrelated forward would
        # silently run under this policy.
        state["depth"] = max(0, state["depth"] - 1)
        if state["depth"] == 0:
            state.pop("indices", None)
            state.pop("weights", None)
        return None

    # The manual-cast gate. Open only while the whole model's forward runs, on
    # the entering thread, so the process-wide functional patches are inert
    # for any other model in the process -- the same scoping the interpolation
    # helper uses above, one level up.
    cast_state: dict = {"depth": 0}
    cast_counts: dict = {"linear": 0, "embedding": 0, "layer_norm": 0, "conv3d": 0,
                         "already_float32": 0}

    def _leak(op: str, dtype: torch.dtype) -> PrecisionLeak:
        return PrecisionLeak(
            f"{policy!r}: F.{op} received {dtype} activations, so something "
            "upstream downcast them and FP32 arithmetic here would compute on a "
            "rounded input. Refusing rather than upcasting."
        )

    def _fp32(tensor):
        return None if tensor is None else tensor.float()

    def cast_linear(input, weight, bias=None):
        if cast_state["depth"] <= 0:
            return _original_linear(input, weight, bias)
        _require_owner("F.linear")
        if input.dtype != torch.float32:
            raise _leak("linear", input.dtype)
        if weight.dtype == torch.float32:
            cast_counts["already_float32"] += 1
            return _original_linear(input, weight, bias)
        cast_counts["linear"] += 1
        return _original_linear(input, weight.float(), _fp32(bias))

    def cast_embedding(input, weight, *args, **kwargs):
        if cast_state["depth"] <= 0:
            return _original_embedding(input, weight, *args, **kwargs)
        _require_owner("F.embedding")
        out = _original_embedding(input, weight, *args, **kwargs)
        if out.dtype == torch.float32:
            cast_counts["already_float32"] += 1
            return out
        # Gather the BF16 rows, then upcast: the same values as upcasting the
        # whole table first, without a transient copy of the 151,936-row table.
        cast_counts["embedding"] += 1
        return out.float()

    def cast_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if cast_state["depth"] <= 0:
            return _original_layer_norm(input, normalized_shape, weight, bias, eps)
        _require_owner("F.layer_norm")
        if input.dtype != torch.float32:
            raise _leak("layer_norm", input.dtype)
        if weight is None or weight.dtype == torch.float32:
            cast_counts["already_float32"] += 1
            return _original_layer_norm(input, normalized_shape, weight, bias, eps)
        cast_counts["layer_norm"] += 1
        return _original_layer_norm(input, normalized_shape, weight.float(), _fp32(bias), eps)

    def cast_conv3d(input, weight, bias=None, *args, **kwargs):
        if cast_state["depth"] <= 0:
            return _original_conv3d(input, weight, bias, *args, **kwargs)
        _require_owner("F.conv3d")
        if input.dtype != torch.float32:
            raise _leak("conv3d", input.dtype)
        if weight.dtype == torch.float32:
            cast_counts["already_float32"] += 1
            return _original_conv3d(input, weight, bias, *args, **kwargs)
        cast_counts["conv3d"] += 1
        return _original_conv3d(input, weight.float(), _fp32(bias), *args, **kwargs)

    def _cast_enter(_module, _args, _kwargs=None):
        _require_owner("the model forward")
        cast_state["depth"] += 1
        return None

    def _cast_exit(_module, _args, _output=None):
        cast_state["depth"] = max(0, cast_state["depth"] - 1)
        return None

    if wanted is not None:
        vision.pos_embed.to(wanted)
    modeling_qwen3_vl.get_vision_interpolation_indices_and_weights = helper
    if exact:
        # Bound as a method, not assigned as a plain function.
        # `compressed_tensors.offload.module.offload_module` reads
        # `module.forward.__func__` when it installs offload hooks, and a plain
        # function has no `__func__` -- which is exactly how the first real
        # sequential-pipeline run failed. Anything that substitutes a forward
        # has to remain a method to survive the offload dispatch.
        vision.pos_embed.forward = types.MethodType(exact_pos_embed, vision.pos_embed)
    pre_handle = vision.register_forward_pre_hook(_enter)
    post_handle = vision.register_forward_hook(_exit, always_call=True)
    cast_handles = []
    if manual_cast:
        functional.linear = cast_linear
        functional.embedding = cast_embedding
        functional.layer_norm = cast_layer_norm
        functional.conv3d = cast_conv3d
        cast_handles = [
            model.register_forward_pre_hook(_cast_enter),
            model.register_forward_hook(_cast_exit, always_call=True),
        ]
    try:
        yield {
            "policy": policy,
            "intent": POLICY_INTENT[policy],
            "compute_dtype": str(_COMPUTE_DTYPE[policy]).removeprefix("torch."),
            "storage_dtype": str(_STORAGE_DTYPE[policy]).removeprefix("torch."),
            "manual_cast": manual_cast,
            "manual_cast_ops": (["linear", "embedding", "layer_norm", "conv3d"]
                                if manual_cast else []),
            "manual_cast_counts": cast_counts if manual_cast else None,
            "keep_in_fp32_modules": list(KEEP_IN_FP32_MODULES) if manual_cast else [],
            "position_interpolation_dtype": ("library default (float32 "
                                             "coefficients)" if wanted is None
                                             else str(wanted).removeprefix("torch.")),
            "position_reduction": ("comfy four-term add" if exact
                                   else "transformers .sum(1)"),
            "position_embedding_source_dtype": str(original).removeprefix("torch."),
            "position_cast_is_round_trip_exact": round_trip_exact,
            "substituted_expression": reduction_source,
            "scoped_to": "this vision instance only, gated by its own forward "
                         "hooks, on the thread that entered the context",
            "concurrency": "single-threaded forward execution only; a forward "
                           "on another thread raises rather than silently "
                           "running under the wrong policy",
            "forward_substitution_is_bound_method": exact,
            "applied_to": "pos_embed weight, its forward, and the interpolation "
                          "coefficients bound inside transformers' qwen3_vl "
                          "modeling module",
            "modifies_checkpoint_or_deployment": False,
            "modifies_installed_packages": False,
        }
    finally:
        for handle in cast_handles:
            handle.remove()
        if manual_cast:
            functional.linear = _original_linear
            functional.embedding = _original_embedding
            functional.layer_norm = _original_layer_norm
            functional.conv3d = _original_conv3d
        pre_handle.remove()
        post_handle.remove()
        modeling_qwen3_vl.get_vision_interpolation_indices_and_weights = original_helper
        if had_instance_forward:
            vision.pos_embed.forward = original_pos_forward
        else:
            vision.pos_embed.__dict__.pop("forward", None)
        vision.pos_embed.to(original)
