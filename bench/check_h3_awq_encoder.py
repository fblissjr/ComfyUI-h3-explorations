#!/usr/bin/env python3
"""Exercise the custom compressed-tensors W4A16 H3 loader on CPU or CUDA.

## The escape this owns

On 2026-08-23 core ``CLIPLoader`` offered the canonical W4A16 file, so the
existing model-file check was green. At execution core detected the full
Hugging Face namespace as Qwen3-VL-8B and instantiated width 4096; the first
H3 norm is width 5120, so the queued job failed at the loader. Menu discovery
was being mistaken for format support.

This check loads the whole checkpoint through the repo adapter on CPU (mmap;
no encode), proves all 350 H3 language linears use the kitchen W4A16 layout,
and exercises the real FP32 H3 activation boundary on a tiny exact matmul.
The optional ``--gpu`` case forces kitchen's CUDA backend. It also drives
core's public loader on the real artifact so the reason for the custom node
cannot quietly disappear. If core later recognizes this format natively, that
case turns red and prompts retirement rather than leaving a redundant adapter.

AWQ is the calibration method, not one storage ABI. When the official
``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`` is installed, a companion
control proves core recognizes its native H3 namespace and all 350 of its
``nvfp4`` layer records. That file working in ``CLIPLoader`` therefore does not
imply that compressed-tensors W4A16 works there too.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO / "workflows"))

import comfy.cli_args  # noqa: E402
from build_h3_awq_standalone import (  # noqa: E402
    COMPARE_WORKFLOW_FILENAME,
    MODEL_FILENAME,
    WORKFLOW_SUBDIR,
    FIRST_LAST_WORKFLOW,
    NODE_ID,
    REF2V_TURBO_LORA_PATH,
    RUNTIME_CONFIGS,
    STANDALONE_FILENAME,
    TURBO_768P_LORA,
    TURBO_768P_SHIFT,
    TURBO_768P_STEPS,
    TURBO_768P_STRENGTH,
    WORKFLOWS,
    build as build_standalone,
    render_standalone_loader,
)
from h3_config import MODELS  # noqa: E402

comfy.cli_args.args.cpu = True


def _module():
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules["_h3pack"] = pkg
    spec = importlib.util.spec_from_file_location(
        "_h3pack.h3_awq_encoder", REPO / "h3_awq_encoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = _module()
NATIVE_NVFP4_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def _path() -> Path:
    raw = os.environ.get("H3_AWQ_ENCODER")
    if raw:
        return Path(os.path.expanduser(raw))
    # The shipped graphs choose a canonical artifact, but the loader itself
    # accepts any selected filename whose metadata and full tensor inventory
    # satisfy the format contract.
    return COMFY / "models" / "text_encoders" / MODELS["clip"]


def _native_path() -> Path:
    raw = os.environ.get("H3_NATIVE_NVFP4_ENCODER")
    if raw:
        return Path(os.path.expanduser(raw))
    return COMFY / "models" / "text_encoders" / NATIVE_NVFP4_NAME


def core_natively_recognizes_nvfp4_awq(path: Path):
    """The working AWQ control: Comfy namespace plus Comfy quant metadata."""
    from collections import Counter
    from safetensors import safe_open
    import comfy.quant_ops
    import comfy.sd

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        detection_state = {
            "visual.deepstack_merger_list.0.norm.weight": None,
            "model.layers.49.self_attn.q_proj.weight": None,
        }
        assert detection_state.keys() <= keys
        detected = comfy.sd.detect_te_model(detection_state)
        assert detected == comfy.sd.TEModel.QWEN3VL_32B, detected

        formats = Counter()
        for key in keys:
            if key.endswith(".comfy_quant"):
                conf = json.loads(f.get_tensor(key).numpy().tobytes())
                formats[conf.get("format")] += 1

    assert formats == {"nvfp4": 350, "int8_tensorwise": 1}, formats
    assert "nvfp4" in comfy.quant_ops.QUANT_ALGOS


def source_config_snapshot_matches_digests():
    recorded = json.loads((H.CONFIG_DIR / "sha256.json").read_text())
    expected = {
        "config.json", "tokenizer_config.json", "processor_config.json",
        "video_preprocessor_config.json", "recipe.yaml",
    }
    assert expected <= recorded.keys(), recorded.keys()
    for name in expected:
        digest = hashlib.sha256((H.CONFIG_DIR / name).read_bytes()).hexdigest()
        assert digest == recorded[name], (name, digest, recorded[name])


def standalone_distribution_contract():
    """Prove the one-file build stays code/config-identical and discoverable."""
    rendered = render_standalone_loader()
    assert rendered == render_standalone_loader(), "standalone render is not deterministic"
    with tempfile.TemporaryDirectory(prefix="h3-awq-standalone-") as raw:
        output_dir = Path(raw)
        written = build_standalone(output_dir)
        # The loader is emitted at the root and the workflows one level down;
        # see build_h3_awq_standalone.build() for why that split is load-bearing.
        expected_names = {STANDALONE_FILENAME,
                          f"{WORKFLOW_SUBDIR}/{COMPARE_WORKFLOW_FILENAME}",
                          *(f"{WORKFLOW_SUBDIR}/{n}" for n in WORKFLOWS)}
        assert {str(path.relative_to(output_dir)) for path in written} == expected_names
        standalone_path = output_dir / STANDALONE_FILENAME
        assert standalone_path.read_text() == rendered

        spec = importlib.util.spec_from_file_location(
            "_h3_awq_standalone_check", standalone_path
        )
        standalone = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = standalone
        spec.loader.exec_module(standalone)

        for name in RUNTIME_CONFIGS:
            source_path = H.CONFIG_DIR / name
            assert standalone._config(name) == json.loads(source_path.read_text())
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            assert standalone._EMBEDDED_CONFIG_SHA256[name] == digest

        # These functions own the format, execution, preprocessing, and load
        # boundaries. The standalone builder copies them from the authoritative
        # module; only config storage, wording, and registration may differ.
        for name in (
            "_quant_contract",
            "adapt_compressed_state_dict",
            "awq_operations",
            "_source_image_patches",
            "_source_video_block_patches",
            "install_source_processors",
            "_validate_loaded_state_contract",
            "_validate_native_tokenizer",
            "_load_clip",
        ):
            standalone_source = inspect.getsource(getattr(standalone, name)).replace(
                "through standalone compressed-tensors adapter",
                "through repo-local compressed-tensors adapter",
            )
            assert standalone_source == inspect.getsource(getattr(H, name)), name

        extension = asyncio.run(standalone.comfy_entrypoint())
        node_list = asyncio.run(extension.get_node_list())
        assert node_list == [standalone.MiniMaxH3AWQEncoderLoader]
        assert NODE_ID in rendered

        for workflow_name in WORKFLOWS:
            workflow = json.loads((output_dir / WORKFLOW_SUBDIR / workflow_name).read_text())
            nodes = list(workflow.get("nodes", []))
            for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
                nodes.extend(subgraph.get("nodes", []))
            types_ = [node.get("type") for node in nodes]
            assert types_.count(NODE_ID) == 1, (workflow_name, types_.count(NODE_ID))
            assert "CLIPLoader" not in types_, workflow_name
            assert not ({
                "MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning",
                "MiniMaxH3Resolution", "MiniMaxH3Preflight",
                "MiniMaxH3SageAttention", "SageChainAssert", "SolAttnMiniMax",
            } & set(types_)), workflow_name
            for create_video in (n for n in nodes if n.get("type") == "CreateVideo"):
                assert create_video.get("widgets_values") == [24, 8, "sRGB"]
                assert create_video.get("widgets_values_named", {}).get(
                    "color_space"
                ) == "sRGB"

            loaders = [node for node in nodes if node.get("type") == NODE_ID]
            # Read the name from the generator rather than restating it: a
            # hardcoded copy here went stale the first time the published file
            # was renamed, and the shipped workflows would have pointed at a
            # file the README no longer tells anyone to download.
            assert loaders[0]["widgets_values"] == [MODEL_FILENAME, "default"]
            if workflow_name != "comfyui_minimax_h3_awq_image_reference.json":
                lora = next(n for n in nodes if n.get("type") == "LoraLoaderModelOnly")
                assert lora["widgets_values"] == [
                    TURBO_768P_LORA, TURBO_768P_STRENGTH,
                ]
                shift = next(n for n in nodes if n.get("type") == "MiniMaxH3SigmaShift")
                assert shift["widgets_values"] == [
                    TURBO_768P_SHIFT["shift_video"],
                    TURBO_768P_SHIFT["shift_audio"],
                ]
                assert any(
                    n.get("type") == "PrimitiveBoolean"
                    and n.get("widgets_values") == [True]
                    for n in nodes
                )
                assert any(
                    n.get("type") == "PrimitiveInt"
                    and n.get("widgets_values", [None])[0] == TURBO_768P_STEPS
                    for n in nodes
                )
            else:
                lora = next(n for n in nodes if n.get("type") == "LoraLoaderModelOnly")
                assert lora["widgets_values"][0] == REF2V_TURBO_LORA_PATH

            if workflow_name == FIRST_LAST_WORKFLOW:
                assert types_.count("LoadImage") == 2
                outer = workflow["nodes"]
                subgraph_ids = {
                    subgraph["id"]
                    for subgraph in workflow["definitions"]["subgraphs"]
                }
                instance = next(n for n in outer if n.get("type") in subgraph_ids)
                last_frame = next(
                    input_ for input_ in instance["inputs"]
                    if input_["name"] == "last_frame"
                )
                assert last_frame["link"] is not None
                prompt = instance["widgets_values_named"]["prompt"]
                assert prompt.startswith(
                    "How the reference pictures align with the target video — "
                    "Picture 1 (from Shot 1) aligns with the 0.00-second mark "
                    "of the target video; Picture 2 (from Shot 1) aligns with "
                    "the 5.17-second mark of the target video.\n\n"
                )
                assert all(
                    section in prompt for section in (
                        "integrated_multimodal_description:",
                        "overall_soundscape:",
                        "non_diegetic_music:",
                    )
                )
                note_text = "\n".join(
                    str(value)
                    for node in outer if node.get("type") == "MarkdownNote"
                    for value in node.get("widgets_values", [])
                )
                assert "default 5-second duration snaps to 124 frames" in note_text

        compare = json.loads((output_dir / WORKFLOW_SUBDIR / COMPARE_WORKFLOW_FILENAME).read_text())
        compare_types = [node["type"] for node in compare["nodes"]]
        assert compare_types.count("VHS_LoadVideo") == 2
        assert compare_types.count("ImageConcatMulti") == 1
        assert compare_types.count("VHS_VideoCombine") == 1
        concat = next(node for node in compare["nodes"]
                      if node["type"] == "ImageConcatMulti")
        assert concat["widgets_values"] == [2, "right", True]
        combine = next(node for node in compare["nodes"]
                       if node["type"] == "VHS_VideoCombine")
        assert combine["widgets_values"]["frame_rate"] == 24.0

        external = os.environ.get("H3_AWQ_STANDALONE")
        if external:
            external_path = Path(os.path.expanduser(external))
            assert external_path.read_text() == rendered, external_path
        return standalone


def external_model_digest_matches(path: Path):
    """Optional slow integrity pass over the external multi-gigabyte file."""
    recorded = json.loads((H.CONFIG_DIR / "sha256.json").read_text())
    expected = recorded.get(path.name)
    if expected is None:
        raise AssertionError(
            f"sha256.json has no external model digest for selected {path.name!r}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == expected, (path.name, digest.hexdigest(), expected)


def core_still_misdetects_the_real_format(path: Path):
    from safetensors import safe_open
    import comfy.sd

    with safe_open(path, framework="pt", device="cpu") as f:
        state = {
            "model.visual.deepstack_merger_list.0.norm.weight":
                f.get_tensor("model.visual.deepstack_merger_list.0.norm.weight"),
            "model.visual.merger.linear_fc2.weight":
                f.get_tensor("model.visual.merger.linear_fc2.weight"),
        }
    detected = comfy.sd.detect_te_model(state)
    assert detected == comfy.sd.TEModel.QWEN3VL_8B, detected
    assert detected != comfy.sd.TEModel.QWEN3VL_32B


def core_loader_still_rejects_the_real_format(path: Path):
    """Exercise core's public load boundary, not detection alone.

    This intentionally passes the real path through ``load_clip`` so embedded
    metadata, old-quant conversion, and every future pre-detection conversion
    participate. A reduced state-dict fixture would remain red if core added
    support in one of those earlier stages and would hide the retirement event.
    """
    import torch
    import comfy.sd

    try:
        comfy.sd.load_clip(
            [str(path)], embedding_directory=[],
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={"load_device": torch.device("cpu"),
                           "offload_device": torch.device("cpu")},
            disable_dynamic=True,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "size mismatch" in message, message[:500]
        assert "4096" in message and "5120" in message, message[:500]
    else:
        raise AssertionError(
            "core loaded the compressed-tensors namespace; reassess whether "
            "MiniMaxH3AWQEncoderLoader is still needed"
        )


def compressed_nibbles_are_kitchen_order(path: Path):
    """Independent bit extraction versus the adapter's zero-copy byte view."""
    import torch
    from safetensors import safe_open

    key = "model.language_model.layers.0.self_attn.q_proj.weight_packed"
    with safe_open(path, framework="pt", device="cpu") as f:
        packed = f.get_tensor(key)[:2]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    source = ((packed.unsqueeze(-1) >> shifts) & 15).reshape(2, -1)
    bytes_ = packed.view(torch.int8).to(torch.int32)
    kitchen = torch.stack([bytes_ & 15, (bytes_ >> 4) & 15], dim=-1).reshape(2, -1)
    assert torch.equal(source, kitchen)


def kitchen_layout_casts_fp32_and_executes_exactly_on_cpu():
    import torch

    ops = H.awq_operations()
    linear = ops.Linear(
        128, 2, bias=False, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    generator = torch.Generator().manual_seed(23)
    qweight = torch.randint(0, 16, (2, 64), dtype=torch.int8, generator=generator)
    scale = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)
    conf = torch.tensor(
        list(json.dumps({"format": H.QUANT_FORMAT,
                         "group_size": H.GROUP_SIZE}).encode()),
        dtype=torch.uint8,
    )
    linear.load_state_dict({
        "weight": qweight, "weight_scale": scale, "comfy_quant": conf,
    }, strict=True)
    # H3's SDClipModel forwards FP32 activations. The W4A16 CUDA kernel accepts
    # BF16/FP16, so the local operation must cast across the quantized matmul
    # and restore FP32 for residual arithmetic. A BF16-only check missed the
    # regression where kitchen silently selected eager for the real workload.
    x = torch.randn(3, 128, dtype=torch.float32, generator=generator)
    got = linear(x)
    q32 = qweight.to(torch.int32)
    values = torch.stack([q32 & 15, (q32 >> 4) & 15], -1)
    values = values.reshape(2, 128).to(torch.bfloat16)
    kernel_x = x.to(torch.bfloat16)
    expected = kernel_x @ ((values - 8) * scale.t()).t()
    assert got.dtype == torch.float32, got.dtype
    assert torch.equal(got, expected.float()), float((got - expected).abs().max())


def strict_inventory_controls():
    """Core's non-strict CLIP load cannot mask absent or stray tensors."""
    import torch

    class _Model:
        def state_dict(self):
            return {"visual.required.weight": torch.empty(2, 3)}

    class _Clip:
        pass

    clip = _Clip()
    clip.cond_stage_model = types.SimpleNamespace(
        qwen3vl_32b=types.SimpleNamespace(transformer=_Model())
    )
    H._validate_loaded_state_contract(
        clip, {"visual.required.weight": (2, 3)}
    )
    for provided, fragment in (
        ({}, "missing="),
        ({"visual.required.weight": (2, 3), "visual.stray": (1,)},
         "unexpected="),
        ({"visual.required.weight": (3, 2)}, "shape_mismatch="),
    ):
        try:
            H._validate_loaded_state_contract(clip, provided)
        except ValueError as exc:
            assert fragment in str(exc), exc
        else:
            raise AssertionError(f"inventory control did not reject {fragment}")


def kitchen_cuda_dispatches_fp32_h3_input():
    """Force kitchen's CUDA backend across the real FP32 H3 call boundary."""
    import torch
    from comfy_kitchen.registry import registry

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    ops = H.awq_operations()
    linear = ops.Linear(
        128, 2, bias=False, device=torch.device("cuda"), dtype=torch.bfloat16
    )
    generator = torch.Generator(device="cuda").manual_seed(23)
    qweight = torch.randint(
        0, 16, (2, 64), dtype=torch.int8, device="cuda", generator=generator
    )
    scale = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16, device="cuda")
    conf = torch.tensor(
        list(json.dumps({"format": H.QUANT_FORMAT,
                         "group_size": H.GROUP_SIZE}).encode()),
        dtype=torch.uint8,
    )
    linear.load_state_dict({
        "weight": qweight, "weight_scale": scale, "comfy_quant": conf,
    }, strict=True)
    x = torch.randn(
        3, 128, device="cuda", dtype=torch.float32, generator=generator
    )
    with registry.use_backend("cuda"):
        got = linear(x)
    torch.cuda.synchronize()
    q32 = qweight.to(torch.int32)
    values = torch.stack([q32 & 15, (q32 >> 4) & 15], -1)
    values = values.reshape(2, 128).to(torch.bfloat16)
    expected = x.to(torch.bfloat16) @ ((values - 8) * scale.t()).t()
    assert got.dtype == torch.float32 and tuple(got.shape) == (3, 2), (
        got.dtype, got.shape)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, expected.float(), rtol=0.02, atol=0.25), (
        got, expected)


def full_loader_contract(path: Path, module=H):
    clip = module._load_clip(str(path), [], device="cpu")
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    linears = [m for m in model.modules()
               if getattr(m, "quant_format", None) == module.QUANT_FORMAT]
    assert len(linears) == module.EXPECTED_QUANTIZED_LINEARS, len(linears)
    assert all(m.weight._params.group_size == module.GROUP_SIZE for m in linears)
    assert model.num_layers == module.H3_LAYERS
    assert model._h3_processor_source == module.CONFIG_SOURCE
    vocab = clip.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    assert [vocab[t] for t in (
        "<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>",
        "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>",
    )] == list(range(151669, 151676))
    return len(linears)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="store_true",
        help="also force a synthetic FP32 H3 activation through kitchen CUDA",
    )
    parser.add_argument(
        "--verify-model-hash", action="store_true",
        help="stream and verify the selected external model (slow; ~19 GB)",
    )
    args = parser.parse_args(argv)
    path = _path()
    if not path.exists():
        print(f"SKIP  custom AWQ encoder not found at {path}; set H3_AWQ_ENCODER")
        return 2
    standalone_module = {}

    def standalone_case():
        standalone_module["module"] = standalone_distribution_contract()

    cases = [
        ("source config snapshot", source_config_snapshot_matches_digests),
        ("standalone loader/workflows", standalone_case),
        ("core format control", lambda: core_still_misdetects_the_real_format(path)),
        ("core load boundary", lambda: core_loader_still_rejects_the_real_format(path)),
        ("compressed nibble order", lambda: compressed_nibbles_are_kitchen_order(path)),
        ("kitchen CPU FP32 boundary",
         kitchen_layout_casts_fp32_and_executes_exactly_on_cpu),
        ("strict full inventory", strict_inventory_controls),
    ]
    if args.gpu:
        cases.append(("kitchen CUDA FP32 boundary",
                      kitchen_cuda_dispatches_fp32_h3_input))
    if args.verify_model_hash:
        cases.append(("external model sha256",
                      lambda: external_model_digest_matches(path)))
    else:
        print("  SKIP  external model sha256: use --verify-model-hash for the ~19 GB pass")
    native = _native_path()
    if native.exists():
        cases.insert(
            0,
            ("native NVFP4-AWQ control",
             lambda: core_natively_recognizes_nvfp4_awq(native)),
        )
    else:
        print(f"  SKIP  native NVFP4-AWQ control: not installed at {native}")
    ok = True
    for label, case in cases:
        try:
            case()
        except Exception as exc:
            ok = False
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {label}")
    started = time.monotonic()
    try:
        # Load through the generated one-file module when its parity/discovery
        # control passed. This makes the large CPU construction an artifact
        # test, not merely another test of the repo-local import.
        module = standalone_module.get("module", H)
        count = full_loader_contract(path, module=module)
    except Exception as exc:
        ok = False
        print(f"  FAIL  full CPU loader: {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    full CPU loader: {count} W4A16 linears, "
              f"{time.monotonic() - started:.1f}s, no encode/CUDA")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
