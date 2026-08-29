"""Load a compatible compressed-tensors AWQ Qwen3-VL checkpoint as H3.

This is deliberately a custom loader instead of a patch to ``CLIPLoader``.
ComfyUI natively supplies the H3 architecture and tokenizer (including the
seven H3 tokens), and comfy-kitchen supplies the CUDA W4A16 operator.  Core
does not currently recognize compressed-tensors' Hugging Face namespace,
packing, or metadata.  This module is the repo-local adapter for that gap.

The source stays symlinked under ``models/text_encoders``.  Adaptation is
in-memory and view-based for the 4-bit weights; it does not write a second
multi-gigabyte checkpoint.  The authoritative small source configs are
snapshotted under ``config/qwen3vl_32b_minimax_h3_w4a16_awq``.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import re
import sys
import types
from pathlib import Path

from comfy_api.latest import io

logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "config"
CONFIG_DIR = SNAPSHOT_ROOT / "qwen3vl_32b_minimax_h3_w4a16_awq"
CONFIG_SOURCE = str(CONFIG_DIR)
QUANT_FORMAT = "h3_awq_w4a16"
H3_LAYERS = 50
GROUP_SIZE = 128
EXPECTED_QUANTIZED_LINEARS = H3_LAYERS * 7

# v1 ships `processor_config.json`, a Qwen3VLProcessor container carrying an
# `image_processor` object. The release ships `preprocessor_config.json`, the
# image processor's own settings at the top level, and a candidate calibrated
# against the release carries that file instead. Read whichever is present and
# take the settings from wherever they sit inside it, so the adapter branches
# on what the file contains and not on which generation wrote it.
STILL_CONFIG_NAMES = ("processor_config.json", "preprocessor_config.json")

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


@functools.lru_cache(maxsize=None)
def _config(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Re-copy the file from the encoder source "
            "directory; do not reconstruct its settings in code."
        )
    return json.loads(path.read_text())


def _snapshot_json(snapshot, name: str) -> dict:
    """One small config from a snapshot; ``None`` is this module's own.

    ``None`` routes through :func:`_config` so the standalone build, which
    embeds its configs and has no directory, resolves every reader the same
    way. A path reads that directory's copy.
    """
    if snapshot is None:
        return _config(name)
    return json.loads((Path(snapshot) / name).read_text())


def _snapshot_has(snapshot, name: str) -> bool:
    if snapshot is None:
        try:
            _config(name)
        except FileNotFoundError:
            return False
        return True
    return (Path(snapshot) / name).is_file()


def _still_config_name(snapshot=None) -> str:
    """Which of ``STILL_CONFIG_NAMES`` this artifact actually carries.

    Split out of :func:`_still_settings` so the bind-time report can name the
    file it read instead of restating the branch. Which one answered is the
    first thing to check when a reference arrives at an unexpected size: the
    two filenames belong to generations that disagree about the bounds inside.
    """
    for name in STILL_CONFIG_NAMES:
        if _snapshot_has(snapshot, name):
            return name
    raise FileNotFoundError(
        f"{snapshot or CONFIG_SOURCE} carries none of {STILL_CONFIG_NAMES}; a "
        "snapshot must be copied from the artifact's own files."
    )


def _still_settings(snapshot=None) -> dict:
    """This artifact's still-image processor settings, whatever it calls them."""
    cfg = _snapshot_json(snapshot, _still_config_name(snapshot))
    return dict(cfg.get("image_processor", cfg))


def _snapshot_dirs(root=None) -> tuple:
    """Every artifact snapshot under ``root``, in name order.

    Empty in the standalone build, whose ``config`` directory does not exist:
    the published one-file loader answers for the artifact whose configs it
    embeds and refuses anything else by name of its own config. ``root``
    exists so a harness can point resolution at a synthetic set of snapshots
    without editing the installed ones.
    """
    root = SNAPSHOT_ROOT if root is None else Path(root)
    if not root.is_dir():
        return ()
    return tuple(sorted(
        path for path in root.iterdir() if (path / "config.json").is_file()
    ))


def _resolve_snapshot(embedded: dict, root=None):
    """Which versioned snapshot the selected artifact declares itself to be.

    Recognition is by content: the artifact's own config has to equal one
    snapshot's ``config.json`` exactly. That is what refuses a v2 file selected
    while only the v1 snapshot is installed, and what refuses it against a v1
    snapshot somebody widened -- neither artifact has to be named anywhere for
    the wrong pairing to fail.

    This module's own snapshot answers first and as ``None``, so the standalone
    build -- which embeds one artifact's configs and has no directory to scan --
    resolves its own file and refuses every other by the same code.
    """
    if root is None and embedded == _config("config.json"):
        return None
    for config_dir in _snapshot_dirs(root):
        if json.loads((config_dir / "config.json").read_text()) == embedded:
            return config_dir
    carried = ", ".join(path.name for path in _snapshot_dirs(root))
    raise ValueError(
        "checkpoint embedded config matches no versioned config snapshot "
        f"(this adapter carries: {carried or Path(CONFIG_SOURCE).name}). A "
        "candidate needs its own snapshot, copied from its own files by "
        "bench/convert_h3_awq_candidate.py."
    )


def _quant_declaration(cfg: dict) -> dict:
    """The W4A16 fields a config declares, in the form the contract compares.

    One extraction, read by the check below and by the bind-time report, so the
    report cannot describe a different set of fields than the one enforced.
    """
    group = ((cfg.get("quantization_config") or {}).get("config_groups") or {}).get(
        "group_0", {}
    )
    weights = group.get("weights") or {}
    text = cfg.get("text_config") or {}
    return {
        "format": group.get("format"),
        "bits": weights.get("num_bits"),
        "group_size": weights.get("group_size"),
        "symmetric": weights.get("symmetric"),
        "strategy": weights.get("strategy"),
        "text_dtype": text.get("dtype"),
        "hidden_size": text.get("hidden_size"),
    }


def _quant_contract(snapshot=None) -> dict:
    cfg = _snapshot_json(snapshot, "config.json")
    text = cfg.get("text_config") or {}
    required = _quant_declaration(cfg)
    expected = {
        "format": "pack-quantized", "bits": 4, "group_size": GROUP_SIZE,
        "symmetric": True, "strategy": "group", "text_dtype": "bfloat16",
        "hidden_size": 5120,
    }
    if required != expected:
        raise ValueError(
            "vendored encoder config is not the W4A16 H3 contract this "
            f"adapter implements: got {required!r}, expected {expected!r}"
        )
    layers = text.get("num_hidden_layers")
    if not isinstance(layers, int) or layers < 1:
        raise ValueError(
            f"encoder config declares {layers!r} decoder layers, not a count"
        )
    # The checkpoint's top-level storage dtype is not the decoder's. A
    # candidate whose recipe keeps the vision patch embed in FP32 declares
    # `float32` here while every decoder tensor it quantizes still carries BF16
    # scales, which is why the decoder's dtype is the one pinned above. What
    # the adapter actually requires of the tensors is asserted on the tensors,
    # in `adapt_compressed_state_dict`.
    if cfg.get("dtype") not in ("bfloat16", "float32"):
        raise ValueError(
            f"encoder config declares storage dtype {cfg.get('dtype')!r}; this "
            "adapter has only met bfloat16 and float32 checkpoints"
        )
    return cfg


def artifact_depth(snapshot=None) -> tuple[int, int]:
    """``(decoder layers the artifact carries, H3 depth it can populate)``.

    The source depth is the artifact's declaration, not a constant here: v1 and
    the v2 candidate both carry 64 and H3 consumes the first 50, while a
    reduced-layer smoke artifact carries fewer and populates all of them.
    """
    layers = int(_quant_contract(snapshot)["text_config"]["num_hidden_layers"])
    return layers, min(H3_LAYERS, layers)


def source_image_pixel_bounds(snapshot=None) -> tuple[int, int]:
    """Return the selected encoder artifact's declared still-image pixel budget.

    Exported for the same reason as the video pair below: a caller that has to
    reason about this artifact's ceiling should read it rather than assume it.
    ``reference_fit.py`` in particular introspects Comfy's native
    ``process_qwen2vl_images`` default, which is the wrong ceiling whenever this
    adapter has replaced ``preprocess_embed``.
    """
    return _bounds_from(_still_settings(snapshot), "source image")


def source_image_patch_geometry(snapshot=None) -> dict:
    """Return still patch/normalization settings from the encoder's snapshot.

    The sibling of :func:`source_video_patch_geometry`, and separate from it
    for the same reason: the still and video processors own their own configs,
    and borrowing one for the other turns an upstream divergence into a silent
    local assumption. They agree today.
    """
    return _geometry_from(_still_settings(snapshot), "source image")


def source_video_pixel_bounds(snapshot=None) -> tuple[int, int]:
    """Return the selected encoder artifact's declared video pixel budget."""
    return _bounds_from(
        _snapshot_json(snapshot, "video_preprocessor_config.json"), "source video"
    )


def source_video_patch_geometry(snapshot=None) -> dict:
    """Return patch/normalization settings from the encoder's own snapshot."""
    return _geometry_from(
        _snapshot_json(snapshot, "video_preprocessor_config.json"), "source video"
    )


def _bounds_from(cfg: dict, what: str) -> tuple[int, int]:
    size = cfg.get("size") or {}
    lo, hi = size.get("shortest_edge"), size.get("longest_edge")
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 < lo < hi:
        raise ValueError(f"{what} processor has invalid size bounds: {size!r}")
    return lo, hi


def _geometry_from(cfg: dict, what: str) -> dict:
    keys = ("patch_size", "temporal_patch_size", "merge_size",
            "image_mean", "image_std")
    geometry = {key: cfg[key] for key in keys if key in cfg}
    if set(geometry) != set(keys):
        raise ValueError(f"{what} processor is missing patch or normalization settings")
    return geometry


def snapshot_contract(config_dir: Path | None = None) -> dict:
    """The processor contract one artifact declares, as one record.

    This is what the loader stamps on the CLIP it builds, and what
    `reference_geometry.encoder_contract_from_clip` reads back. It is a
    property of the *artifact*, read from the artifact's own
    `processor_config.json` and `video_preprocessor_config.json`: the current
    W4 file carries its snapshot under `config/`, and a candidate shipped as
    an HF directory carries the same two files beside its weights. Neither
    the conditioner nor the preflight needs to know which, and neither may
    fall back to this module's default when handed a CLIP that declares
    nothing -- that silent substitution is what this record exists to end.

    `None` means this module's own snapshot, read through `_config` so the
    standalone build, which embeds the configs and has no directory, resolves
    it the same way.
    """
    image = _still_settings(config_dir)
    video = _snapshot_json(config_dir, "video_preprocessor_config.json")
    source = (Path(CONFIG_SOURCE).name if config_dir is None
              else Path(config_dir).name)
    return {
        "source": source,
        "image_bounds": _bounds_from(image, "source image"),
        "image_geometry": _geometry_from(image, "source image"),
        "video_bounds": _bounds_from(video, "source video"),
        "video_geometry": _geometry_from(video, "source video"),
    }


# Which snapshot each single-file artifact this adapter knows is described by.
# `None` is "this module's own snapshot"; a path names one of the other
# directories under `config/`. This table is read only by the *static* reader
# below, which sees a graph's `encoder_name` string and never opens the file;
# the loader itself recognizes an artifact by its embedded config, not by its
# name. A name absent from the table resolves to `None`, never to a guess, and
# so does a name whose snapshot directory is not installed.
ARTIFACT_SNAPSHOTS: dict[str, Path | None] = {
    "qwen3vl_32b_minimax_h3_w4a16_awq.safetensors": None,
    "qwen3vl_32b_minimax_h3_w4a16_awq_v1-comfy.safetensors": None,
    "qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors":
        SNAPSHOT_ROOT / "qwen3vl_32b_minimax_h3_w4a16_awq_v2",
    "qwen3vl_32b_minimax_h3_w4a16_awq_v2_smoke-comfy.safetensors":
        SNAPSHOT_ROOT / "qwen3vl_32b_minimax_h3_w4a16_awq_v2_smoke",
}


def encoder_contract_from_artifact(encoder_name: str) -> dict | None:
    """Resolve the contract for a loader's `encoder_name` without loading it.

    For the static readers (`bench/preflight_graph.py`), which see a graph and
    not a CLIP. Returns `None` for a name this adapter does not know, so the
    caller reports "no contract" rather than pricing somebody else's bounds.
    """
    name = str(encoder_name)
    if Path(name).name in ARTIFACT_SNAPSHOTS:
        config_dir = ARTIFACT_SNAPSHOTS[Path(name).name]
        if config_dir is None or Path(config_dir).is_dir():
            return snapshot_contract(config_dir)
        return None
    candidate = Path(name)
    if candidate.is_dir() and any(
        (candidate / still).exists() for still in STILL_CONFIG_NAMES
    ):
        return snapshot_contract(candidate)
    return None


def _validate_metadata(metadata: dict | None):
    """Check the AWQ scheme and resolve which snapshot this artifact declares.

    Returns the resolved snapshot -- ``None`` for this module's own -- so every
    later stage reads the selected artifact's configs rather than the default
    ones. Before 2026-08-25 there was one artifact and no resolution step; a
    second generation with different processor bounds is exactly the case where
    a silent default would bind the wrong ceiling.
    """
    metadata = metadata or {}
    if metadata.get("scheme") != "w4a16" or metadata.get("quantization") != "awq":
        raise ValueError(
            "checkpoint is not the expected AWQ W4A16 artifact: safetensors "
            f"metadata says scheme={metadata.get('scheme')!r}, "
            f"quantization={metadata.get('quantization')!r}"
        )
    raw = metadata.get("config")
    if not isinstance(raw, str):
        raise ValueError("checkpoint safetensors metadata has no embedded config")
    try:
        embedded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint embedded config is not valid JSON") from exc
    snapshot = _resolve_snapshot(embedded)
    if embedded != _quant_contract(snapshot):
        raise ValueError(
            "checkpoint embedded config differs from the versioned config "
            f"snapshot from {snapshot or CONFIG_SOURCE}"
        )
    return snapshot


def _drop_source_key(name: str, depth: int) -> bool:
    """True for full-Qwen tensors H3 intentionally does not consume."""
    match = _LAYER.match(name)
    if match and int(match.group(1)) >= depth:
        return True
    return name.startswith("lm_head.") or name.startswith("model.language_model.norm.")


def adapt_compressed_state_dict(state_dict: dict, metadata: dict | None) -> dict:
    """Destructively adapt the raw full-depth HF state dict to Comfy H3.

    compressed-tensors packs eight signed int4 values into each int32 after
    adding eight.  On little-endian hosts an int8 view yields four consecutive
    bytes, each already containing the two unsigned nibbles comfy-kitchen
    expects.  No weight-sized unpack or copy is needed.

    The truncation depth is the artifact's, not a constant: the released
    checkpoints carry 64 decoder layers and H3 consumes the first 50, while a
    reduced-layer smoke artifact carries fewer and keeps all of them. The
    quantized-linear count is then asserted exactly against that depth.
    """
    import torch

    snapshot = _validate_metadata(metadata)
    source_layers, depth = artifact_depth(snapshot)
    expected_linears = depth * 7
    if sys.byteorder != "little":
        raise RuntimeError("the zero-copy AWQ repack is only defined on little-endian hosts")

    shapes = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_shape") and not _drop_source_key(name, depth):
            prefix = name[:-len(".weight_shape")]
            shape = tuple(int(x) for x in tensor.tolist())
            if len(shape) != 2:
                raise ValueError(f"{name} declares non-matrix shape {shape!r}")
            shapes[prefix] = shape

    out = {}
    quantized = set()
    for source_name in list(state_dict):
        tensor = state_dict.pop(source_name)
        if _drop_source_key(source_name, depth):
            continue

        name = source_name
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        elif name.startswith("model.visual."):
            name = "visual." + name[len("model.visual."):]

        if source_name.endswith(".weight_shape"):
            continue
        if source_name.endswith(".weight_packed"):
            source_prefix = source_name[:-len(".weight_packed")]
            logical = shapes.get(source_prefix)
            if tensor.dtype != torch.int32 or logical is None:
                raise ValueError(
                    f"{source_name} needs int32 storage and a weight_shape companion"
                )
            expected = (logical[0], logical[1] // 8)
            if tuple(tensor.shape) != expected or logical[1] % GROUP_SIZE:
                raise ValueError(
                    f"{source_name} shape {tuple(tensor.shape)} does not encode "
                    f"declared {logical} at 4-bit/group-{GROUP_SIZE}"
                )
            target_prefix = name[:-len(".weight_packed")]
            qweight = tensor.view(torch.int8)
            if tuple(qweight.shape) != (logical[0], logical[1] // 2):
                raise AssertionError("int32-to-int8 view did not preserve packed rows")
            out[f"{target_prefix}.weight"] = qweight
            conf = {"format": QUANT_FORMAT, "group_size": GROUP_SIZE}
            out[f"{target_prefix}.comfy_quant"] = torch.tensor(
                list(json.dumps(conf, sort_keys=True).encode("utf-8")),
                dtype=torch.uint8,
            )
            quantized.add(target_prefix)
            continue
        if source_name.endswith(".weight_scale"):
            source_prefix = source_name[:-len(".weight_scale")]
            logical = shapes.get(source_prefix)
            if tensor.dtype != torch.bfloat16 or logical is None:
                raise ValueError(f"{source_name} has no usable BF16 scale/shape pair")
            expected = (logical[0], logical[1] // GROUP_SIZE)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{source_name} shape {tuple(tensor.shape)} != {expected}"
                )
            target_prefix = name[:-len(".weight_scale")]
            # kitchen owns scales as (K/group, N); compressed-tensors stores
            # (N, K/group). CUDA consumes flat row-major data, so materialize
            # the transpose once rather than handing it a strided view.
            out[f"{target_prefix}.weight_scale"] = tensor.t().contiguous()
            continue

        out[name] = tensor

    if len(quantized) != expected_linears:
        raise ValueError(
            f"adapted {len(quantized)} quantized linears; this artifact needs "
            f"{expected_linears} (7 in each of {depth} layers)"
        )
    missing_scales = [p for p in quantized if f"{p}.weight_scale" not in out]
    if missing_scales:
        raise ValueError(f"quantized linears missing scales: {missing_scales[:3]}")
    if "visual.deepstack_merger_list.0.norm.weight" not in out:
        raise ValueError("adapted checkpoint has no Qwen3-VL DeepStack vision tower")
    if f"model.layers.{depth - 1}.self_attn.q_proj.weight" not in out:
        raise ValueError(f"adapted checkpoint does not reach layer {depth - 1}")
    if any(k.startswith(f"model.layers.{depth}.") for k in out):
        raise AssertionError(f"source layer {depth} escaped the H3 truncation")

    logger.info(
        "[h3-awq] adapted compressed-tensors checkpoint in memory: %d "
        "W4A16 linears, first %d/%d language layers, BF16 vision/embedding",
        len(quantized), depth, source_layers,
    )
    return out


def _install_quant_format():
    """Register the kitchen layout only when this custom loader executes."""
    import torch
    import comfy.ops
    import comfy.quant_ops
    from comfy_kitchen.tensor import TensorCoreAWQW4A16Layout  # noqa: F401

    spec = {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_zeros"},
        "comfy_tensor_layout": "TensorCoreAWQW4A16Layout",
        "quantize_input": False,
    }
    existing = comfy.quant_ops.QUANT_ALGOS.get(QUANT_FORMAT)
    if existing is not None and existing != spec:
        raise RuntimeError(
            f"ComfyUI already registered {QUANT_FORMAT!r} with a different contract"
        )
    comfy.quant_ops.QUANT_ALGOS[QUANT_FORMAT] = spec
    comfy.ops.QUANT_ALGOS[QUANT_FORMAT] = spec
    return spec


@functools.lru_cache(maxsize=1)
def awq_operations():
    """Comfy mixed ops with one local load branch for symmetric AWQ W4A16."""
    import torch
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreAWQW4A16Layout

    spec = _install_quant_format()
    base = comfy.ops.mixed_precision_ops(
        compute_dtype=torch.bfloat16, full_precision_mm=False
    )

    class H3AWQOperations(base):
        class Linear(base.Linear):
            def _forward(self, input, weight, bias):
                if (isinstance(weight, QuantizedTensor)
                        and weight._layout_cls == "TensorCoreAWQW4A16Layout"):
                    # SDClipModel intentionally constructs and forwards H3
                    # embeddings as FP32. AWQ is W4A16: kitchen's fused CUDA
                    # backend accepts BF16/FP16 activation, while an FP32 x
                    # silently selects the eager dequantization backend. Cast
                    # only across the quantized matmul, then restore the
                    # caller's dtype for residual arithmetic.
                    output_dtype = input.dtype
                    kernel_dtype = weight._params.scale.dtype
                    input = input.to(dtype=kernel_dtype)
                    if bias is not None:
                        bias = bias.to(dtype=kernel_dtype)

                    if input.device.type == "cuda" and not getattr(
                            H3AWQOperations, "_awq_backend_logged", False):
                        from comfy_kitchen.registry import registry
                        backend = registry.get_capable_backend(
                            "gemv_awq_w4a16",
                            kwargs={
                                "x": input,
                                "qweight": weight._qdata,
                                "wscales": weight._params.scale,
                                "wzeros": weight._params.zeros,
                                "bias": bias,
                                "group_size": weight._params.group_size,
                            },
                        )
                        logger.info(
                            "[h3-awq] W4A16 dispatch backend=%s, "
                            "activation %s -> %s",
                            backend, output_dtype, kernel_dtype,
                        )
                        H3AWQOperations._awq_backend_logged = True

                    return torch.nn.functional.linear(
                        input, weight, bias
                    ).to(dtype=output_dtype)
                return super()._forward(input, weight, bias)

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                conf_key = f"{prefix}comfy_quant"
                conf_tensor = state_dict.get(conf_key)
                conf = None
                if conf_tensor is not None:
                    conf = json.loads(conf_tensor.numpy().tobytes())
                if not conf or conf.get("format") != QUANT_FORMAT:
                    return super()._load_from_state_dict(
                        state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs,
                    )

                weight_key = f"{prefix}weight"
                scale_key = f"{prefix}weight_scale"
                zeros_key = f"{prefix}weight_zeros"
                weight = state_dict.pop(weight_key, None)
                scale = state_dict.pop(scale_key, None)
                zeros = state_dict.pop(zeros_key, None)
                state_dict.pop(conf_key, None)
                if weight is None or scale is None:
                    raise ValueError(
                        f"{prefix.rstrip('.')} is missing AWQ weight or scale"
                    )
                if tuple(weight.shape) != (
                    self._orig_shape[0], self._orig_shape[1] // 2
                ):
                    raise ValueError(
                        f"{prefix} packed shape {tuple(weight.shape)} does not "
                        f"match linear {self._orig_shape}"
                    )

                device = self.factory_kwargs["device"]
                dtype = self.factory_kwargs["dtype"]
                weight = weight.to(device=device, dtype=spec["storage_t"])
                scale = scale.to(device=device, dtype=dtype)
                if zeros is None:
                    # The source config is symmetric, so the affine zero term
                    # is exactly zero. kitchen's general AWQ ABI still takes
                    # the tensor explicitly.
                    zeros = torch.zeros_like(scale, device=device, dtype=dtype)
                else:
                    zeros = zeros.to(device=device, dtype=dtype)

                params = TensorCoreAWQW4A16Layout.Params(
                    scale=scale, zeros=zeros,
                    group_size=int(conf.get("group_size", GROUP_SIZE)),
                    transposed=False, orig_dtype=dtype,
                    orig_shape=self._orig_shape,
                )
                self.quant_format = QUANT_FORMAT
                self.layout_type = spec["comfy_tensor_layout"]
                self._full_precision_mm_config = False
                self.weight = torch.nn.Parameter(
                    QuantizedTensor(weight, self.layout_type, params),
                    requires_grad=False,
                )

                # Let torch load the ordinary bias, then erase the missing
                # report for the weight we deliberately consumed ourselves.
                torch.nn.Module._load_from_state_dict(
                    self, state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs,
                )
                for key in (weight_key, scale_key, zeros_key, conf_key):
                    if key in missing_keys:
                        missing_keys.remove(key)

    H3AWQOperations.__name__ = "H3AWQOperations"
    return H3AWQOperations


@functools.lru_cache(maxsize=8)
def _image_processor(snapshot=None, bounds: tuple[int, int] | None = None):
    """Build the artifact's still-image processor, or one at overridden bounds.

    ``bounds`` exists so a caller can measure this artifact under a different
    still-image budget without editing the snapshot, which is hash-guarded by
    ``bench/check_h3_awq_encoder.py`` for good reason: the snapshot records what
    the artifact *declares*. Passing bounds does not change that declaration,
    and the default remains the declared one.

    The release's flat ``preprocessor_config.json`` and v1's nested
    ``processor_config.json`` both construct the same slow processor: the
    release file omits ``resample`` and the ``do_*`` flags and the class
    defaults supply v1's values, so only ``size`` differs between them.
    Measured on the installed transformers, 2026-08-25.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    settings = _still_settings(snapshot)
    if bounds is not None:
        lo, hi = bounds
        if not (isinstance(lo, int) and isinstance(hi, int) and 0 < lo < hi):
            raise ValueError(
                f"still-image pixel bounds must be ints with 0 < min < max, got {bounds!r}"
            )
        settings["size"] = {"shortest_edge": lo, "longest_edge": hi}
    return Qwen2VLImageProcessor(**settings)


_clamp_reported: set = set()


def _report_clamp(width: int, height: int, grid, bounds: tuple[int, int],
                  snapshot=None) -> None:
    """Say so, once per distinct case, when the budget actually reduces a reference.

    The still-image budget is a ceiling applied *after* whatever sizing the
    reference nodes performed, and it governs only the conditioner. When it
    binds, the DiT's view of a reference and Qwen's view diverge — a reference
    deliberately upscaled upstream can still reach layer 50 heavily reduced.
    Until 2026-08-24 nothing reported that at runtime, in this module or in the
    conditioning path.
    """
    key = (width, height, bounds)
    if key in _clamp_reported:
        return
    patch = int(_still_settings(snapshot)["patch_size"])
    grid_h, grid_w = int(grid[0][1]), int(grid[0][2])
    out_h, out_w = grid_h * patch, grid_w * patch
    if out_h * out_w >= width * height:
        return
    _clamp_reported.add(key)
    logger.info(
        "[h3-awq] still-image budget %d..%d px reduced a %dx%d reference to "
        "%dx%d for the conditioner (%d merged tokens). The DiT's view of this "
        "reference is unaffected; only the Qwen view is capped.",
        bounds[0], bounds[1], width, height, out_w, out_h,
        (grid_h // 2) * (grid_w // 2),
    )


def _source_image_patches(images, device, bounds: tuple[int, int] | None = None,
                          snapshot=None):
    """Run the source checkpoint's declared still-image processor.

    ``bounds`` overrides the declared budget for this call only; see
    ``_image_processor``.
    """
    import torch

    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError(f"Qwen image must be [B,H,W,3], got {tuple(images.shape)}")
    height, width = int(images.shape[1]), int(images.shape[2])
    image = images[0].detach().permute(2, 0, 1).to("cpu")
    if image.is_floating_point():
        image = image.mul(255).round().clamp_(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    batch = _image_processor(snapshot, bounds).preprocess(image, return_tensors="pt")
    grid = batch["image_grid_thw"].to(device=device)
    _report_clamp(width, height, grid,
                  bounds or source_image_pixel_bounds(snapshot), snapshot)
    return batch["pixel_values"], grid


def _source_video_block_patches(frames, device, snapshot=None):
    """Patchify an already duration-fitted two-frame Qwen video block."""
    import torch

    cfg = _snapshot_json(snapshot, "video_preprocessor_config.json")
    temporal = int(cfg["temporal_patch_size"])
    patch = int(cfg["patch_size"])
    merge = int(cfg["merge_size"])
    if (frames.ndim != 4 or frames.shape[-1] != 3 or
            frames.shape[0] != temporal):
        raise ValueError(
            f"Qwen video block must be [{temporal},H,W,3], got {tuple(frames.shape)}"
        )
    height, width = int(frames.shape[1]), int(frames.shape[2])
    factor = patch * merge
    if height % factor or width % factor:
        raise ValueError(
            f"Qwen video block {width}x{height} was not fitted to the source "
            f"processor's {factor}-pixel grid. Use "
            "MiniMaxH3ReferenceConditioning video_policy='encoder' or 'release'."
        )

    imgs = frames.permute(0, 3, 1, 2)
    mean = torch.tensor(cfg["image_mean"], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image_std"], device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std
    grid_h, grid_w = height // patch, width // patch
    patches = imgs.reshape(
        1, temporal, 3, grid_h // merge, merge, patch,
        grid_w // merge, merge, patch,
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_h * grid_w, 3 * temporal * patch * patch
    )
    grid = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)
    return flatten, grid


def _describe(settings: dict) -> str:
    """Every key/value of one config block, in a stable order."""
    return ", ".join(f"{key}={settings[key]!r}" for key in sorted(settings))


def _source_configs_report(snapshot, bounds: tuple[int, int],
                           overridden: bool) -> str:
    """Which config files this bind read, and the values it took from each.

    Unconditional, because the artifact generations this adapter accepts are
    shaped identically and differ ONLY in these files -- the still-image budget
    alone moves a 1344x768 reference between a few hundred and a few thousand
    merged tokens, and nothing downstream of the bind can tell you which budget
    it ran under. Every value is read back out of the file at the moment it is
    bound, so this is the artifact's own declaration and not a second copy of
    one.
    """
    source = CONFIG_SOURCE if snapshot is None else str(snapshot)
    lines = [f"[h3-awq] source configs bound from {source}"]

    still_name = _still_config_name(snapshot)
    raw = _snapshot_json(snapshot, still_name)
    shape = "image_processor object" if "image_processor" in raw else "flat"
    lines.append(f"[h3-awq]   still  {still_name} ({shape}): "
                 + _describe(_still_settings(snapshot)))

    video_name = "video_preprocessor_config.json"
    lines.append(f"[h3-awq]   video  {video_name}: "
                 + _describe(_snapshot_json(snapshot, video_name)))

    tokenizer_name = "tokenizer_config.json"
    tokenizer = _snapshot_json(snapshot, tokenizer_name)
    key = ("extra_special_tokens" if tokenizer.get("extra_special_tokens")
           else "additional_special_tokens")
    declared = tokenizer.get(key) or []
    lines.append(f"[h3-awq]   tokens {tokenizer_name}: {len(declared)} special "
                 f'tokens under "{key}", H3 markers {declared[13:]}')

    config_name = "config.json"
    cfg = _snapshot_json(snapshot, config_name)
    layers = (cfg.get("text_config") or {}).get("num_hidden_layers")
    consumed = min(H3_LAYERS, layers) if isinstance(layers, int) else None
    lines.append(f"[h3-awq]   model  {config_name}: {layers} decoder layers, "
                 f"first {consumed} consumed by H3, storage "
                 f"dtype={cfg.get('dtype')!r}, " + _describe(_quant_declaration(cfg)))

    budget = f"[h3-awq]   still-image budget in force: {bounds[0]}..{bounds[1]} px"
    if overridden:
        declared_lo, declared_hi = source_image_pixel_bounds(snapshot)
        budget += (" -- OVERRIDDEN for this CLIP instance; the artifact still "
                   f"declares {declared_lo}..{declared_hi}")
    lines.append(budget)
    return "\n".join(lines)


def install_source_processors(clip, image_bounds: tuple[int, int] | None = None,
                              snapshot=None) -> None:
    """Bind source-config preprocessing to this CLIP instance only.

    ``image_bounds`` overrides the artifact's declared still-image budget for
    this CLIP instance. It exists for measurement — comparing one set of weights
    under two processor policies — and is not a way to redeclare the artifact.
    The snapshot on disk stays authoritative and unchanged; ``_h3_image_bounds``
    records what this instance was actually bound with so a capture can report
    it rather than assume the declared value.

    ``snapshot`` is which artifact's configs to bind, resolved by the loader
    from the file it opened. ``None`` is this module's own, which is what a
    caller re-binding a policy onto an already-loaded v1 CLIP wants.
    """
    import torch

    model = clip.cond_stage_model.qwen3vl_32b.transformer
    bounds = image_bounds or source_image_pixel_bounds(snapshot)

    def preprocess_embed(this, embed, device):
        if embed.get("type") != "image":
            return None, None
        if embed.get("minimax_video_block", False):
            flatten, grid = _source_video_block_patches(
                embed["data"], device, snapshot
            )
        else:
            flatten, grid = _source_image_patches(
                embed["data"], device, image_bounds, snapshot
            )
        merged, deepstack = this.visual(
            flatten.to(device=device, dtype=torch.float32), grid
        )
        return merged, {"grid": grid, "deepstack": deepstack}

    model.preprocess_embed = types.MethodType(preprocess_embed, model)
    model._h3_processor_source = CONFIG_SOURCE if snapshot is None else str(snapshot)
    model._h3_image_bounds = bounds
    # The contract the conditioner and the static readers bind to. The image
    # bounds are the ones this INSTANCE was bound with, override included, so
    # a capture at overridden bounds is priced at the bounds it actually ran.
    contract = snapshot_contract(snapshot)
    contract["image_bounds"] = tuple(bounds)
    model._h3_encoder_contract = contract
    try:
        logger.info(
            "%s", _source_configs_report(snapshot, bounds, image_bounds is not None)
        )
    except Exception as exc:  # a report may not be the thing that fails a load
        logger.info("[h3-awq] could not report the bound source configs: %r", exc)


def _validate_loaded_state_contract(clip, provided_shapes: dict[str, tuple]) -> None:
    """Reject missing, extra, or shape-incompatible adapted model tensors.

    Core intentionally loads text encoders with ``strict=False``. That is a
    useful general policy, but unsafe for this format adapter: an absent
    ordinary vision/norm weight otherwise leaves a factory-created parameter
    behind and the loader still returns a CLIP. Compare the adapted inventory
    with the concrete native H3 module that core selected.

    Symmetric AWQ is the single intentional exception. The source omits affine
    zero tensors and ``H3AWQOperations`` constructs exact zeros while loading.
    """
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    expected_state = model.state_dict()
    expected = set(expected_state)
    provided = set(provided_shapes)
    quantized = {
        name[:-len(".comfy_quant")]
        for name in provided
        if name.endswith(".comfy_quant")
    }
    synthesized = {f"{prefix}.weight_zeros" for prefix in quantized}
    missing = sorted(expected - provided - synthesized)
    unexpected = sorted(provided - expected)
    mismatched = []
    for name in sorted(expected & provided):
        # This byte tensor serializes configuration rather than model data.
        # Presence is part of the inventory; its decoded format/group values
        # were validated while adapting and loading the quantized linear.
        if name.endswith(".comfy_quant"):
            continue
        actual = tuple(provided_shapes[name])
        wanted = tuple(expected_state[name].shape)
        if actual != wanted:
            mismatched.append((name, actual, wanted))
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:3]}")
        raise ValueError(
            "selected checkpoint does not exactly populate the native H3 "
            "architecture after adaptation: " + "; ".join(details)
        )


def _validate_native_tokenizer(clip, snapshot=None) -> None:
    """Prove native Comfy's tokenizer realizes the snapshotted token list.

    The declaration is the ordered list of 20 special tokens; which key holds
    it is a property of the tokenizer_config generation, not of the artifact.
    v1's file writes ``extra_special_tokens``; a candidate saved by a newer
    Transformers writes the same 20 in the same order under
    ``additional_special_tokens`` and stops its ``added_tokens_decoder`` before
    the seven H3 ids. Reading only the first key would report "no tokens
    declared" for the second file and refuse a correct artifact, so read
    whichever carries the list and assert the list itself.
    """
    cfg = _snapshot_json(snapshot, "tokenizer_config.json")
    declared = (cfg.get("extra_special_tokens")
                or cfg.get("additional_special_tokens") or [])
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    vocab = tokenizer.get_vocab()
    if len(declared) != 20:
        raise ValueError(
            f"source encoder declares {len(declared)} special tokens, expected 20"
        )
    if len(set(declared)) != len(declared):
        raise ValueError("source encoder declares duplicate special tokens")
    expected = {
        **{token: 151644 + index for index, token in enumerate(declared[:13])},
        **{token: 151669 + index for index, token in enumerate(declared[13:])},
    }
    actual = {token: vocab.get(token) for token in declared}
    if actual != expected:
        raise ValueError(
            "native ComfyUI tokenizer token ids disagree with the selected "
            f"encoder config: got {actual}, expected {expected}"
        )
    cfg = _snapshot_json(snapshot, "config.json")
    declared_roles = {
        "<|vision_start|>": cfg.get("vision_start_token_id"),
        "<|vision_end|>": cfg.get("vision_end_token_id"),
        "<|image_pad|>": cfg.get("image_token_id"),
        "<|video_pad|>": cfg.get("video_token_id"),
    }
    role_mismatches = {
        token: (token_id, expected[token])
        for token, token_id in declared_roles.items()
        if token_id != expected[token]
    }
    if role_mismatches:
        raise ValueError(
            "source config token roles disagree with tokenizer_config: "
            f"{role_mismatches} (config, tokenizer)"
        )


def _load_clip(path: str, embedding_directory, device: str = "default",
               disable_dynamic: bool = False, install_cache: bool = True):
    import torch
    import comfy.sd
    import comfy.utils

    state_dict, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )
    snapshot = _validate_metadata(metadata)
    source_layers, depth = artifact_depth(snapshot)
    if depth != H3_LAYERS:
        # Core recognizes the H3 encoder by the presence of decoder layer 49
        # (`comfy/sd.py::detect_te_model`) and builds a fixed 50-layer model, so
        # a reduced-depth artifact cannot be constructed here at all. Say that,
        # rather than letting core misdetect it as a different Qwen3-VL and fail
        # on a width mismatch the way the 2026-08-23 escape did. The adaptation
        # itself is depth-parametric and can still be inspected directly.
        raise ValueError(
            f"{Path(path).name} declares {source_layers} decoder layers, which "
            f"populates {depth} of native ComfyUI's {H3_LAYERS} H3 layers. Only "
            "a full-depth artifact can be constructed; adapt this one with "
            "adapt_compressed_state_dict and inspect the result instead."
        )
    state_dict = adapt_compressed_state_dict(state_dict, metadata)
    provided_shapes = {name: tuple(tensor.shape) for name, tensor in state_dict.items()}
    model_options = {"custom_operations": awq_operations()}
    if device == "cpu":
        cpu = torch.device("cpu")
        model_options["load_device"] = model_options["offload_device"] = cpu
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state_dict], embedding_directory=embedding_directory,
        clip_type=comfy.sd.CLIPType.MINIMAX, model_options=model_options,
        disable_dynamic=disable_dynamic,
    )
    _validate_loaded_state_contract(clip, provided_shapes)
    install_source_processors(clip, snapshot=snapshot)
    _validate_native_tokenizer(clip, snapshot)
    if install_cache:
        clip.patcher.cached_patcher_init = (
            load_h3_awq_model_patcher,
            (path, embedding_directory, device),
        )
    logger.info(
        "[h3-awq] loaded %s through repo-local compressed-tensors adapter; "
        "architecture/tokenizer are native ComfyUI, W4A16 execution is "
        "comfy-kitchen, preprocessing is source-config driven",
        Path(path).name,
    )
    return clip


def load_h3_awq_model_patcher(path: str, embedding_directory,
                              device: str = "default", disable_dynamic=False):
    return _load_clip(
        path, embedding_directory, device=device,
        disable_dynamic=disable_dynamic, install_cache=False,
    ).patcher


@functools.lru_cache(maxsize=1)
def shipped_encoder_name() -> str | None:
    """A W4A16 artifact THIS ADAPTER CAN OPEN, read from `workflows/h3_config.py`.

    **Corrected 2026-08-27 (late).** This read `MODELS["clip"]` on the premise
    that the menu should start from "the artifact the shipped graphs load".
    That premise died the same day: the graphs moved to the ComfyUI-native INT8
    build through core's `CLIPLoader`, so `MODELS["clip"]` became a file this
    loader REFUSES, and the default it produced would have started every
    freshly dragged node on an error. A menu default has to name something the
    node can open, which is a property of the FILE and not of what ships.

    So it reads `ENCODER_V1` -- still a real W4A16 artifact on this box -- and
    falls back to `MODELS["clip"]` only when that is itself adapter-openable,
    which keeps the original intent alive for anyone who ships one again.

    Read, never copied. This is only a menu default, but a second copy of the
    filename here would drift from the graphs the first time one of them moved,
    which is the rule `h3_config` exists to enforce. Loaded from its path rather
    than imported: nothing is added to `sys.path`, and that file declares no
    imports of its own, so this costs no dependency in either direction.

    `None` is ordinary, not an error -- the standalone distribution carries this
    module without a `workflows/` beside it, and a name absent from the
    directory's real population must not become a menu item either way.
    """
    path = Path(__file__).resolve().parent / "workflows" / "h3_config.py"
    try:
        spec = importlib.util.spec_from_file_location("_h3_config_menu_default", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = set(getattr(module, "CORE_LOADED_ENCODERS", ()) or ())
        shipped = module.MODELS.get("clip")
        # Prefer the shipped encoder only if this adapter owns its format.
        if isinstance(shipped, str) and shipped not in native:
            name = shipped
        else:
            name = getattr(module, "ENCODER_V1", None)
    except Exception:
        return None
    return name if isinstance(name, str) else None


class MiniMaxH3AWQEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        # Filename is not the format contract. Offer the directory's real
        # safetensors population and validate compressed-tensors AWQ metadata
        # after the user selects one; never manufacture a missing menu item.
        names = sorted(
            n for n in folder_paths.get_filename_list("text_encoders")
            if n.endswith(".safetensors")
        )
        # Select the artifact the shipped graphs load, but only when this
        # directory actually offers it. Without a default ComfyUI selects
        # `options[0]`, which is whatever sorts first in `text_encoders` and on
        # this box was `clip_l.safetensors` -- not an H3 encoder at all. A
        # default outside `options` would be the manufactured menu item the
        # comment above refuses, so a miss leaves the menu as it was.
        shipped = shipped_encoder_name()
        encoder_input = (
            io.Combo.Input("encoder_name", options=names, default=shipped)
            if shipped in names
            else io.Combo.Input("encoder_name", options=names)
        )
        return io.Schema(
            node_id="MiniMaxH3AWQEncoderLoader",
            display_name="Load MiniMax H3 Compressed-Tensors AWQ Encoder",
            category="MiniMaxH3/loaders",
            description=(
                "Repo-local adapter for Qwen3-VL-32B H3 checkpoints using "
                "compressed-tensors W4A16 AWQ. ComfyUI supplies native H3 "
                "architecture/tokenizer; "
                "comfy-kitchen supplies W4A16 CUDA execution. This node "
                "converts compressed-tensors packing/metadata in memory and "
                "uses the source image/video processor configs. Core "
                "CLIPLoader only lists this file; it cannot load this format."
            ),
            inputs=[
                encoder_input,
                io.Combo.Input(
                    "device", options=["default", "cpu"], default="default",
                    optional=True,
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, encoder_name, device="default"):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("text_encoders", encoder_name)
        clip = _load_clip(
            path, folder_paths.get_folder_paths("embeddings"), device=device
        )
        return io.NodeOutput(clip)
