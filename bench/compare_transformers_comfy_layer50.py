#!/usr/bin/env python3
"""Do the two stacks land on the same layer-50 state from the same batch?

Gate 1, output-boundary half, and the only part of this gate that uses the real
released weights on both sides. Calibration drives Transformers'
`Qwen3VLForConditionalGeneration` through `llm-compressor`; deployed H3
conditioning runs ComfyUI's 50-layer implementation. If those two disagree at
the H3 boundary, AWQ collects statistics from a distribution the deployed model
never produces, and every later fidelity number is measured against the wrong
reference.

`canonical/2026-08-24_transformers_comfy_parity.md` closed M-RoPE and
vision-tower arithmetic on seeded small models. This is the different claim: the
whole stack, released BF16 weights, one real multi-block multimodal row, at the
raw unnormalized residual after decoder layer 49.

**The isolation.** Both arms consume the same bundle: the same token ids,
attention mask, token-type ids, grids and -- byte for byte -- the same
`pixel_values`. The ComfyUI arm does not recompute patches; its
`preprocess_embed` replays the bundle's tensors, so no processor difference can
enter. The comparator refuses to report metrics unless the presentation hashes
agree.

**The controls.** A number this expensive is worth nothing if the comparison
cannot fail, so `--compare` also grades two deliberate defects: the state tapped
after decoder layer 48 instead of 49, and one decoder weight perturbed. Both
must move the metrics well outside the honest arms' agreement.

Three commands, in separate processes so the two 32B stacks are never resident
together:

    # ComfyUI venv
    python bench/compare_transformers_comfy_layer50.py --arm comfy \\
        --bundle <dir> --row <id> --out <dir>
    # llm-compressor venv
    python bench/compare_transformers_comfy_layer50.py --arm transformers \\
        --bundle <dir> --row <id> --out <dir> --source-dir <released text encoder>
    # either venv
    python bench/compare_transformers_comfy_layer50.py --compare <a> <b> \\
        --out bench/results/<name>.json

`--dtype` selects the Transformers compute precision. ComfyUI runs the language
stack in float32 with BF16-stored weights (`comfy/sd1_clip.py` passes
`dtype=torch.float32` and `manual_cast` upcasts each weight), so `float32` is
the dtype-matched comparison and `bfloat16` measures what a bf16 calibration
run would additionally cost. Whichever is used is recorded, and a comparison
across two different dtypes is reported as such rather than as implementation
drift.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
COMFY = REPO.parents[1]
SCHEMA = "h3-crossstack-layer50-v1"
H3_TAP_LAYER = 49

sys.path.insert(0, str(BENCH))

from h3_attention_kernel import ATTENTION_KINDS, attention_kernel  # noqa: E402
from h3_calibration_precision import (  # noqa: E402
    POLICIES,
    calibration_precision,
    compute_dtype,
    storage_dtype,
    storage_policy,
)


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bundle_row(bundle: Path, row_id: str | None) -> tuple[dict, dict, dict]:
    manifest = json.loads((bundle / "presentation.json").read_text())
    rows = {r["row_id"]: r for r in manifest["rows"]}
    if row_id is None:
        multi = [r for r in manifest["rows"] if len(r["vision_blocks"]) > 1]
        if not multi:
            raise SystemExit("no multi-block row in the bundle; pass --row")
        record = multi[0]
    else:
        if row_id not in rows:
            raise SystemExit(f"{row_id} is not in the bundle")
        record = rows[row_id]
    from safetensors.torch import load_file

    batch = load_file(bundle / record["batch_file"])
    media = load_file(bundle / record["media_file"]) if record.get("media_file") else {}
    return manifest, record, {"batch": batch, "media": media}


def _split_blocks(record: dict, batch: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The bundle's concatenated patches, back into per-block tensors.

    The split sizes are the recorded grids' own products, so a mismatch between
    what the record claims and what the tensor holds fails here rather than
    silently misaligning a block.
    """
    if not record["vision_blocks"]:
        # A text-only row: no patches to split, and a batch that carries some
        # anyway is a builder defect, not a row to grade.
        if "pixel_values" in batch:
            raise ValueError("the record declares no vision block but the batch "
                             "carries pixel_values")
        return []
    patches, grids = batch["pixel_values"], batch["image_grid_thw"]
    blocks, offset = [], 0
    for index, block in enumerate(record["vision_blocks"]):
        grid = torch.tensor(block["grid_thw"], dtype=torch.long)
        rows = int(grid.prod(-1).sum())
        blocks.append((patches[offset:offset + rows], grid))
        if not torch.equal(grid[0], grids[index]):
            raise ValueError(
                f"block {index}: recorded grid {grid.tolist()} != batch grid "
                f"{grids[index].tolist()}"
            )
        offset += rows
    if offset != patches.shape[0]:
        raise ValueError(
            f"the recorded grids account for {offset} patch rows, the batch has "
            f"{patches.shape[0]}"
        )
    return blocks


def _presentation_hashes(record: dict, batch: dict) -> dict:
    """What both arms must agree on before any metric is computed."""
    return {
        "row_id": record["row_id"],
        "prompt_sha256": record["prompt_sha256"],
        "sequence_length": record["sequence_length"],
        "input_ids_sha256": _tensor_sha(batch["input_ids"]),
        "attention_mask_sha256": _tensor_sha(batch["attention_mask"]),
        "mm_token_type_ids_sha256": _tensor_sha(batch["mm_token_type_ids"]),
        # A text-only row carries no patches; the absence is part of the
        # presentation both arms must agree on, so it is recorded as such.
        "pixel_values_sha256": (_tensor_sha(batch["pixel_values"])
                                if "pixel_values" in batch else None),
        "image_grid_thw_sha256": (_tensor_sha(batch["image_grid_thw"])
                                  if "image_grid_thw" in batch else None),
        "grids": batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else [],
        "token_tags_sha256": record["token_tags_sha256"],
        "comfy_position_ids_sha256": record["position_ids_sha256"],
    }


# --------------------------------------------------------------------------
# ComfyUI arm


def _capture_module():
    """Reuse the BF16 loader the layer-50 benchmark already owns."""
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    import nodes  # noqa: F401

    package = types.ModuleType("_h3_crossstack_pkg")
    package.__path__ = [str(REPO)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "_h3_crossstack_capture", BENCH / "capture_h3_encoder_states.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load capture_h3_encoder_states.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _embedding_tap(layer) -> tuple[object, list]:
    """Capture the hidden state entering decoder layer 0.

    A single number at layer 50 says the two stacks differ; it cannot say
    where. This tap splits the difference in two: whatever is already present
    here comes from the token embedding and the vision tower, and whatever
    appears only at the output accumulated through the language stack.
    DeepStack injects *after* layers 0, 1 and 2, so this state is embeddings
    plus merged vision features and nothing else.

    ComfyUI calls its decoder layers with `x=` as a keyword and Transformers
    passes `hidden_states` positionally, so the hook reads both.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, args, kwargs):
        state = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
        if state is None:
            raise ValueError("could not find the hidden state entering layer 0")
        captured.append(state.detach().float().cpu())
        return None

    return layer.register_forward_pre_hook(hook, with_kwargs=True), captured


def _artifact_record(path: Path, model) -> dict:
    """Name a W4 artifact without reading its 19 GB: path, size, mtime, and
    the contract the loader stamped on the model, which is what identifies the
    artifact's snapshot. Bit-level identity is the check's job at convert time."""
    stat = path.stat()
    contract = getattr(model, "_h3_encoder_contract", None)
    if contract is not None and not isinstance(contract, (dict, str, int, float, list)):
        contract = str(contract)
    return {"logical_name": path.name, "artifact": "w4a16", "bytes": stat.st_size,
            "mtime": int(stat.st_mtime), "encoder_contract": contract}


def run_comfy_arm(bundle: Path, row_id: str | None, source: Path, out: Path,
                  reserve_gib: float, tap_layer: int, w4_path: Path | None = None,
                  all_rows: bool = False) -> dict:
    """The deployed stack on a bundle row: BF16 stored weights by default, or a
    W4 artifact through the same loader the capture instrument uses. With
    `all_rows` the model is loaded once and every row of the bundle is written
    to `out/<row_id>/`, which is how a holdout is graded."""
    bundle_manifest = json.loads((bundle / "presentation.json").read_text())
    row_ids = [r["row_id"] for r in bundle_manifest["rows"]] if all_rows else [row_id]

    capture = _capture_module()
    h3 = capture._h3_module()
    import comfy.model_management as model_management
    import folder_paths

    model_management.EXTRA_RESERVED_VRAM = int(reserve_gib * 1024 * 1024 * 1024)
    embedding_directory = folder_paths.get_folder_paths("embeddings")
    if w4_path is None:
        clip, inventory = capture._load_bf16(source, h3, embedding_directory)
        model = clip.cond_stage_model.qwen3vl_32b.transformer
        source_record = {"logical_name": source.name,
                         "mapped_tensors": inventory["selected_tensor_count"]}
        dtype_label = "float32 compute, bfloat16 stored weights"
    else:
        clip = capture._load_w4(w4_path, h3, embedding_directory)
        model = clip.cond_stage_model.qwen3vl_32b.transformer
        source_record = _artifact_record(w4_path, model)
        dtype_label = "float32 compute, w4a16 stored weights"

    if tap_layer != H3_TAP_LAYER:
        # The wrong-layer control. ComfyUI builds exactly 50 layers, so the tap
        # is moved by truncating the stack rather than by reading a different
        # hidden state -- which is what a builder that got the depth wrong would
        # actually have produced.
        model.model.layers = model.model.layers[: tap_layer + 1]

    last: dict = {}
    for rid in row_ids:
        if all_rows and (out / rid / "manifest.json").exists():
            # A rerun after a failed row: rows already captured are kept, since
            # the model and the bundle are the same and the capture is
            # deterministic; the manifest records the reserve used per row.
            print(f"skip {rid}: already captured")
            continue
        manifest, record, tensors = _bundle_row(bundle, rid)
        batch, media = tensors["batch"], tensors["media"]
        blocks = _split_blocks(record, batch)
        replay = list(blocks)
        consumed: list[dict] = []

        def preprocess_embed(this, embed, device, _replay=replay, _consumed=consumed):
            """Replay the bundle's patches; recompute nothing."""
            if embed.get("type") != "image":
                return None, None
            if not _replay:
                raise ValueError("the presentation asked for more vision blocks than "
                                 "the bundle recorded")
            patches, grid = _replay.pop(0)
            _consumed.append({"grid_thw": grid.tolist(),
                              "patches_sha256": _tensor_sha(patches)})
            merged, deepstack = this.visual(
                patches.to(device=device, dtype=torch.float32), grid.to(device)
            )
            return merged, {"grid": grid.to(device), "deepstack": deepstack}

        model.preprocess_embed = types.MethodType(preprocess_embed, model)

        ref_items = []
        for item in record["ordered_media"]:
            if item["type"] == "audio":
                ref_items.append({"type": "audio"})
                continue
            pixels = media[item["upstream_media_key"]].float() / 255.0
            if item["type"] == "image":
                ref_items.append({"type": "image", "data": pixels})
            else:
                ref_items.append({"type": "video", "data": pixels,
                                  "timestamps": list(item["timestamps"])})

        prompt = _prompt_for(record, bundle)
        embed_handle, embed_captured = _embedding_tap(model.model.layers[0])
        started = time.time()
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        output = clip.encode_from_tokens(tokens, return_dict=True)
        elapsed = time.time() - started
        embed_handle.remove()

        hidden = output["cond"].detach().float().cpu()[0].contiguous()
        tags = [int(x) for x in output["minimax_token_tags"].detach().cpu().flatten().tolist()]
        # The arm replays the bundle's patches through the replaced
        # `preprocess_embed`, so the artifact's own processor bounds never run
        # and cannot reach what is graded. What this guards is the replacement
        # itself: a `preprocess_embed` that silently did not take (wrong
        # attribute path, a later overwrite) would let the stack build its own
        # view of the media, and the count would differ from the bundle's. A
        # green run says the replacement held, not anything about bounds.
        if int(hidden.shape[0]) != int(record["sequence_length"]):
            raise ValueError(
                f"{rid}: the stack built {int(hidden.shape[0])} tokens, the bundle "
                f"recorded {record['sequence_length']}; the arm is not grading the "
                f"bundle's presentation"
            )
        if replay:
            raise ValueError(f"{len(replay)} recorded vision blocks were never consumed")
        if len(embed_captured) != 1:
            raise ValueError(f"the layer-0 tap fired {len(embed_captured)} times")
        embeddings = embed_captured[0][0].contiguous()

        manifest_out = {
            "schema": SCHEMA,
            "arm": "comfy",
            "dtype": dtype_label,
            "tap_layer": tap_layer,
            "row_id": rid,
            "reserve_vram_gib": reserve_gib,
            "seconds": round(elapsed, 1),
            "presentation": _presentation_hashes(record, batch),
            "comfy_sequence_length": int(hidden.shape[0]),
            "comfy_token_tags_sha256": _json_sha(tags),
            "consumed_vision_blocks": consumed,
            "hidden_state": {"shape": list(hidden.shape),
                             "dtype": "float32", "sha256": _tensor_sha(hidden)},
            "layer0_input": {"shape": list(embeddings.shape),
                             "dtype": "float32", "sha256": _tensor_sha(embeddings)},
            "bundle_provenance": manifest["provenance"],
            "source": source_record,
            "environment": _environment(),
        }
        _write_arm(out / rid if all_rows else out, hidden, manifest_out, embeddings)
        last = manifest_out
    return last

def _prompt_for(record: dict, bundle: Path) -> str:
    """The row's raw prompt, from the pinned dataset, checked against the record.

    The bundle stores the prompt's hash rather than its several kilobytes, so
    the text is re-read from the dataset and must hash to what the builder saw.
    """
    spec = importlib.util.spec_from_file_location(
        "_h3_pool_builder_crossstack", BENCH / "build_h3_calibration_pool.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load build_h3_calibration_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, _ = module.pinned_snapshot()
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] == record["row_id"]:
            prompt = row.get("target_ir") or ""
            actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if actual != record["prompt_sha256"]:
                raise ValueError(
                    f"{record['row_id']}: the dataset prompt no longer hashes to "
                    f"what the bundle recorded"
                )
            return prompt
    raise SystemExit(f"{record['row_id']} is not in the pinned snapshot")


# --------------------------------------------------------------------------
# Transformers arm


def _sdpa_availability(dtype: torch.dtype, config) -> dict:
    """Which fused SDPA backends exist for this model's attention shape.

    Torch exposes availability but not which kernel a given call actually
    selected, so this bounds the question rather than answering it: an arm
    cannot have used a backend that is unavailable. Both the grouped-query
    shape the model declares and the expanded shape a `repeat_kv` would produce
    are reported, because which of the two reaches SDPA decides the answer and
    is not visible from outside.
    """
    if not torch.cuda.is_available():
        return {"available": None, "reason": "no CUDA"}
    from torch.backends.cuda import (
        SDPAParams,
        can_use_cudnn_attention,
        can_use_efficient_attention,
        can_use_flash_attention,
    )

    text = config.text_config
    heads, kv_heads = text.num_attention_heads, text.num_key_value_heads
    head_dim, seq = text.head_dim, 512
    out = {}
    for label, k_heads, gqa in (("grouped_query", kv_heads, True),
                                ("expanded_kv", heads, False)):
        q = torch.zeros(1, heads, seq, head_dim, dtype=dtype, device="cuda")
        k = torch.zeros(1, k_heads, seq, head_dim, dtype=dtype, device="cuda")
        params = SDPAParams(q, k, k, None, 0.0, True, gqa)
        out[label] = {
            "flash": bool(can_use_flash_attention(params, False)),
            "efficient": bool(can_use_efficient_attention(params, False)),
            "cudnn": bool(can_use_cudnn_attention(params, False)),
        }
        del q, k
    torch.cuda.empty_cache()
    out["note"] = ("availability, not selection: torch does not expose which "
                   "kernel a call chose")
    return out


@contextlib.contextmanager
def _sdpa_backend(name: str):
    """Pin SDPA's kernel, so a mask experiment is not also a kernel experiment.

    Passing an all-ones `attention_mask` sends SDPA to its math backend;
    omitting the mask lets it pick a memory-efficient one. Comparing those two
    directly changes two things at once, and a difference from the kernel would
    be attributed to the mask. Forcing the backend separates the questions.
    """
    if name == "auto":
        yield {"requested": "auto"}
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {"math": [SDPBackend.MATH],
                "efficient": [SDPBackend.EFFICIENT_ATTENTION]}
    if name not in backends:
        raise ValueError(f"unknown sdpa backend {name!r}")
    with sdpa_kernel(backends[name]):
        yield {"requested": name}


def run_transformers_arm(bundle: Path, row_id: str | None, source: Path, out: Path,
                         policy: str, tap_layer: int, gpu_gib: float,
                         perturb: str | None, keep_mask: bool,
                         sdpa_backend: str = "auto",
                         attention: str = "grouped_query") -> dict:
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    manifest, record, tensors = _bundle_row(bundle, row_id)
    batch = tensors["batch"]

    config = AutoConfig.from_pretrained(source)
    # Only the layers H3 consumes are built, so the run costs 50/64 of the
    # checkpoint instead of all of it. The tap is the last layer's raw output,
    # taken by hook rather than from `output_hidden_states`, whose final entry
    # is post-norm and would not be the H3 boundary.
    config.text_config.num_hidden_layers = tap_layer + 1
    # The policy sets two things independently: the dtype the model loads in,
    # which is the linear and residual compute, and the dtype the vision
    # position interpolation runs at. `bench/h3_calibration_precision.py` owns
    # that split and records which deployed behaviour each half models.
    torch_dtype = compute_dtype(policy)
    started = time.time()
    # SDPA, explicitly. The config's default is eager, which materialises a
    # `seq x seq x heads` score matrix -- 9 GiB in float32 at 6,189 tokens, an
    # OOM rather than a result. ComfyUI reaches SDPA through
    # `optimized_attention_for_device`, so this is also the closer match.
    # Loaded at the policy's storage dtype; `torch_dtype` above is the compute
    # dtype the inputs are cast to. They differ only under the manual-cast
    # policy, where `storage_policy` also keeps the patch embed in FP32.
    with storage_policy(Qwen3VLForConditionalGeneration, policy):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, config=config, dtype=storage_dtype(policy), device_map="auto",
            attn_implementation="sdpa",
            max_memory={0: f"{gpu_gib:.0f}GiB", "cpu": "100GiB"},
        ).eval()
    load_seconds = time.time() - started

    layers = model.model.language_model.layers
    if len(layers) != tap_layer + 1:
        raise ValueError(f"built {len(layers)} layers, expected {tap_layer + 1}")

    perturbed = None
    perturb_handle = None
    if perturb:
        # The altered-weight control, applied as an output scale on one
        # bias-free projection rather than an in-place edit of its weight.
        # `device_map="auto"` leaves offloaded parameters as placeholders that
        # accelerate refills from its own map at call time, so an in-place
        # `weight.mul_` is discarded before the forward -- the first version of
        # this control did exactly that and returned a state bit-identical to
        # the unperturbed arm, a control that could not fail. For a linear with
        # no bias, scaling the output is scaling the weight.
        # Layer 0, not the tapped layer: a scale applied at the last layer
        # reaches the output through one residual add and moved the metric by
        # 2 percent of an already-small honest value, which is not a control.
        # Applied at layer 0 it propagates through the whole stack, which is
        # what a real calibration-time weight defect would do.
        scale = float(perturb)
        perturb_layer = 0
        target = layers[perturb_layer].mlp.down_proj
        if getattr(target, "bias", None) is not None:
            raise ValueError("down_proj has a bias; output scaling is not "
                             "equivalent to weight scaling here")
        perturb_handle = target.register_forward_hook(
            lambda _m, _i, output: output * (1.0 + scale)
        )
        perturbed = {"module": f"layers.{perturb_layer}.mlp.down_proj",
                     "applied_as": "output scale, equivalent to scaling the "
                                   "bias-free weight",
                     "relative_scale": scale}

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        captured.append(state.detach().float().cpu())

    handle = layers[tap_layer].register_forward_hook(hook)
    embed_handle, embed_captured = _embedding_tap(layers[0])
    device = next(model.parameters()).device
    all_ones = bool((batch["attention_mask"] == 1).all())
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "mm_token_type_ids": batch["mm_token_type_ids"].to(device),
        "pixel_values": batch["pixel_values"].to(device=device, dtype=torch_dtype),
        "image_grid_thw": batch["image_grid_thw"].to(device),
    }
    # An explicit all-ones `attention_mask` becomes a float 4D mask, which sends
    # SDPA to its math backend and materialises `heads x seq x seq` -- 9.13 GiB
    # in float32 at 6,189 tokens, and quadratic from there. Dropping it when it
    # masks nothing leaves the attention causal and identical while letting the
    # memory-efficient kernel run. `--keep-attention-mask` measures the cost.
    dropped_mask = all_ones and not keep_mask
    if not dropped_mask:
        inputs["attention_mask"] = batch["attention_mask"].to(device)
    started = time.time()
    with calibration_precision(model, policy) as precision:
        with attention_kernel(model, attention) as kernel:
            with _sdpa_backend(sdpa_backend) as backend:
                with torch.no_grad():
                    model(**inputs, use_cache=False)
    forward_seconds = time.time() - started
    handle.remove()
    embed_handle.remove()
    if perturb_handle is not None:
        perturb_handle.remove()

    if len(captured) != 1:
        raise ValueError(f"the tap fired {len(captured)} times, expected once")
    hidden = captured[0][0].contiguous()
    if len(embed_captured) != 1:
        raise ValueError(f"the layer-0 tap fired {len(embed_captured)} times")
    embeddings = embed_captured[0][0].contiguous()

    manifest_out = {
        "schema": SCHEMA,
        "arm": "transformers",
        "dtype": policy,
        "precision_policy": precision,
        "attn_implementation": model.config._attn_implementation,
        "attention_mask_all_ones": all_ones,
        "attention_mask_passed": not dropped_mask,
        "sdpa_backend": backend,
        "attention_kernel": kernel,
        "sdpa_availability": _sdpa_availability(torch_dtype, config),
        "tap_layer": tap_layer,
        "tap": f"forward hook on language_model.layers[{tap_layer}], raw residual",
        "load_seconds": round(load_seconds, 1),
        "forward_seconds": round(forward_seconds, 1),
        "presentation": _presentation_hashes(record, batch),
        "transformers_sequence_length": int(hidden.shape[0]),
        "layers_built": len(layers),
        "deepstack_mergers": len(model.model.visual.deepstack_merger_list),
        "perturbed_weight": perturbed,
        "hidden_state": {"shape": list(hidden.shape),
                         "dtype": "float32", "sha256": _tensor_sha(hidden)},
        "layer0_input": {"shape": list(embeddings.shape),
                         "dtype": "float32", "sha256": _tensor_sha(embeddings)},
        "bundle_provenance": manifest["provenance"],
        "source": {"logical_name": source.name},
        "peak_cuda_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                          if torch.cuda.is_available() else None),
        "environment": _environment(),
    }
    _write_arm(out, hidden, manifest_out, embeddings)
    return manifest_out


# --------------------------------------------------------------------------


def _environment() -> dict:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _write_arm(out: Path, hidden: torch.Tensor, manifest: dict,
               embeddings: torch.Tensor | None = None) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=False)
    tensors = {"hidden_state": hidden}
    if embeddings is not None:
        tensors["layer0_input"] = embeddings
    save_file(tensors, out / "hidden_state.safetensors")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("bundle_provenance",)}, indent=2))


def _metrics(reference: torch.Tensor, candidate: torch.Tensor,
             mask: torch.Tensor | None = None) -> dict:
    a = reference.double()
    b = candidate.double()
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.numel() == 0:
        return {"rows": 0}
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    cosine = float(torch.dot(flat_a, flat_b) /
                   (flat_a.norm() * flat_b.norm()))
    difference = a - b
    tokenwise = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {
        "rows": int(a.shape[0]),
        "flattened_cosine": cosine,
        "relative_l2": float(difference.norm() / flat_a.norm()),
        "mse": float((difference ** 2).mean()),
        "rmse": float((difference ** 2).mean().sqrt()),
        "reference_activation_rms": float((a ** 2).mean().sqrt()),
        "candidate_activation_rms": float((b ** 2).mean().sqrt()),
        "tokenwise_cosine_mean": float(tokenwise.mean()),
        "tokenwise_cosine_min": float(tokenwise.min()),
        "tokenwise_cosine_p01": float(torch.quantile(tokenwise, 0.01)),
        "tokenwise_cosine_p50": float(torch.quantile(tokenwise, 0.50)),
    }


def _attention_kind(manifest: dict) -> str | None:
    """Which KV form the arm's SDPA saw.

    The ComfyUI arm has no such field: its attention is its own. A Transformers
    capture from before the switch existed has none either, and could only
    have been the library's grouped-query decision, which is what is returned
    for it.
    """
    if manifest.get("arm") != "transformers":
        return None
    kernel = manifest.get("attention_kernel")
    return kernel["kind"] if kernel else "grouped_query"


def compare(paths: list[Path], bundle: Path | None, out: Path,
            reference_dir: Path | None = None,
            field_under_test: list[str] | None = None) -> int:
    from safetensors.torch import load_file

    arms = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        loaded = load_file(path / "hidden_state.safetensors")
        hidden = loaded["hidden_state"]
        if _tensor_sha(hidden) != manifest["hidden_state"]["sha256"]:
            raise SystemExit(f"{path.name}: hidden state does not match its manifest")
        arms.append((manifest, hidden, loaded.get("layer0_input"), path.resolve()))

    if reference_dir is not None:
        # Naming a reference makes this comparator usable for questions that are
        # not "how far is Transformers from deployed ComfyUI" -- the
        # mask-omission equivalence proof compares two Transformers arms to each
        # other. The choice is recorded so such a report can never be misread as
        # the default comparison.
        resolved = reference_dir.resolve()
        reference = next((a for a in arms if a[3] == resolved), None)
        if reference is None:
            raise SystemExit(
                f"--reference {reference_dir.name} is not among the compared captures"
            )
    else:
        reference = next((a for a in arms if a[0]["arm"] == "comfy"), None)
        if reference is None:
            raise SystemExit("no ComfyUI arm among the captures; it is the reference")

    # Which field the arms are allowed to differ on. Against the ComfyUI arm
    # the policy is the question by construction; between two Transformers
    # arms it has to be declared, so a policy comparison cannot be mistaken
    # for a backend or mask one. The declaration is recorded in the report.
    under_test = set(field_under_test or [])
    if reference[0]["arm"] == "comfy":
        under_test.update({"dtype", "attention"})

    report: dict = {
        "comparison": "ComfyUI versus Transformers at the H3 layer-50 boundary",
        "reference_arm": {k: v for k, v in reference[0].items()
                          if k not in ("bundle_provenance",)},
        "reference_chosen_explicitly": reference_dir is not None,
        "field_under_test": sorted(under_test),
        "arms": {},
        "refusals": [],
    }
    base_manifest, base_hidden, base_embeddings, _base_path = reference

    for manifest, hidden, embeddings, _path in arms:
        if manifest is base_manifest:
            continue
        label = f"{manifest['arm']}-{manifest['dtype']}-tap{manifest['tap_layer']}"
        if manifest.get("arm") == "transformers":
            label += ("-mask" if manifest.get("attention_mask_passed") else "-nomask")
            label += f"-{manifest.get('sdpa_backend', {}).get('requested', '?')}"
            label += f"-{_attention_kind(manifest)}"
        if manifest.get("perturbed_weight"):
            label += "-perturbed"
        if manifest.get("arm") == "comfy" and "encoder" in under_test:
            label += "-" + str(manifest.get("source", {}).get("logical_name"))
        entry: dict = {"manifest": {k: v for k, v in manifest.items()
                                    if k not in ("bundle_provenance",)}}
        # Refuse on anything that would make the two arms answer different
        # questions, not just on presentation. A backend experiment that also
        # moved the policy, the tap or the weights would attribute the whole
        # difference to the backend.
        mismatched = [
            field for field in ("dtype", "tap_layer")
            if manifest.get(field) != base_manifest.get(field)
            and field not in under_test
        ]
        if manifest.get("perturbed_weight") != base_manifest.get("perturbed_weight"):
            mismatched.append("perturbed_weight")
        if (_attention_kind(manifest) != _attention_kind(base_manifest)
                and "attention" not in under_test):
            mismatched.append("attention")
        if manifest.get("source", {}).get("logical_name") != \
                base_manifest.get("source", {}).get("logical_name") \
                and "encoder" not in under_test:
            mismatched.append("source")
        if mismatched:
            entry["refused"] = (
                f"the arms differ on {mismatched} as well, so a difference "
                f"could not be attributed to the field under test"
            )
            report["refusals"].append(f"{label}: {entry['refused']}")
            report["arms"][label] = entry
            continue
        if manifest["presentation"] != base_manifest["presentation"]:
            differing = sorted(
                k for k, v in manifest["presentation"].items()
                if base_manifest["presentation"].get(k) != v
            )
            entry["refused"] = (
                f"presentation differs on {differing}; a rowwise metric would be "
                f"invalid"
            )
            report["refusals"].append(f"{label}: {entry['refused']}")
            report["arms"][label] = entry
            continue
        if hidden.shape != base_hidden.shape:
            entry["refused"] = (
                f"shape {tuple(hidden.shape)} != {tuple(base_hidden.shape)}"
            )
            report["refusals"].append(f"{label}: {entry['refused']}")
            report["arms"][label] = entry
            continue

        tags = _tags_for(base_manifest, bundle)
        entry["all"] = _metrics(base_hidden, hidden)
        if tags is not None:
            entry["text_rows"] = _metrics(base_hidden, hidden,
                                          torch.tensor([t == 1 for t in tags]))
            entry["vision_rows"] = _metrics(base_hidden, hidden,
                                            torch.tensor([t == 0 for t in tags]))
        if base_embeddings is not None and embeddings is not None \
                and base_embeddings.shape == embeddings.shape:
            entry["layer0_input"] = _metrics(base_embeddings, embeddings)
            if tags is not None:
                # Split here too. The aggregate hides which half of the input
                # diverges, and that is the whole question: the token embedding
                # or the vision tower.
                entry["layer0_input_text_rows"] = _metrics(
                    base_embeddings, embeddings, torch.tensor([t == 1 for t in tags])
                )
                entry["layer0_input_vision_rows"] = _metrics(
                    base_embeddings, embeddings, torch.tensor([t == 0 for t in tags])
                )
        report["arms"][label] = entry

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"reference: {base_manifest['arm']} {base_manifest['dtype']} "
          f"tap {base_manifest['tap_layer']}, {base_hidden.shape[0]} rows")
    for label, entry in report["arms"].items():
        if "refused" in entry:
            print(f"  {label:<44} REFUSED: {entry['refused']}")
            continue
        row = entry["all"]
        layer0 = entry.get("layer0_input")
        print(f"  {label:<44} layer50 cos {row['flattened_cosine']:.6f}  relL2 "
              f"{row['relative_l2']:.6f}  rmse {row['rmse']:.4g}  "
              f"tok-cos min {row['tokenwise_cosine_min']:.6f}")
        if layer0:
            text0 = entry.get("layer0_input_text_rows", {})
            vision0 = entry.get("layer0_input_vision_rows", {})
            print(f"  {'':<44} layer0  cos {layer0['flattened_cosine']:.6f}  relL2 "
                  f"{layer0['relative_l2']:.6f}"
                  + (f"  (text {text0['relative_l2']:.6g}, "
                     f"vision {vision0['relative_l2']:.6f})" if text0 else ""))
            print(f"  {'':<44} layer50 text relL2 {entry['text_rows']['relative_l2']:.6f}  "
                  f"vision relL2 {entry['vision_rows']['relative_l2']:.6f}")
    print(f"\nwrote {out}")
    return 0


def _tags_for(manifest: dict, bundle: Path | None) -> list[int] | None:
    if bundle is None:
        return None
    presentation = json.loads((bundle / "presentation.json").read_text())
    for record in presentation["rows"]:
        if record["row_id"] == manifest["presentation"]["row_id"]:
            # The record stores the tag hash, not the tags; rebuild them from
            # the recorded vision spans so the split is the bundle's, not a
            # second guess at it.
            tags = [1] * record["sequence_length"]
            for start, end in record["vision_spans"]:
                for index in range(start, end + 1):
                    tags[index] = 0
            if _json_sha(tags) != record["token_tags_sha256"]:
                raise SystemExit(
                    "tags rebuilt from the recorded vision spans do not hash to "
                    "the recorded tags"
                )
            return tags
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("comfy", "transformers"))
    parser.add_argument("--compare", nargs="+", metavar="DIR")
    parser.add_argument("--reference", metavar="DIR",
                        help="use this capture as the reference instead of the "
                             "ComfyUI arm; recorded in the report")
    parser.add_argument("--attention", default="grouped_query", choices=ATTENTION_KINDS,
                        help="which KV form reaches SDPA; see bench/h3_attention_kernel.py")
    parser.add_argument("--w4-path", default=None,
                        help="ComfyUI arm only: load this W4 artifact through the "
                             "capture instrument's loader instead of the BF16 shards")
    parser.add_argument("--all-rows", action="store_true",
                        help="ComfyUI arm only: one model load, every bundle row "
                             "captured to OUT/<row_id>/")
    parser.add_argument("--field-under-test", action="append", choices=("dtype", "attention", "encoder"),
                        help="declare which field the compared arms may differ "
                             "on; a policy comparison between two Transformers "
                             "arms needs `dtype`. Recorded in the report")
    parser.add_argument("--bundle")
    parser.add_argument("--row")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"))
    parser.add_argument("--dtype", default="float32", choices=sorted(POLICIES),
                        help="calibration precision policy; see "
                             "bench/h3_calibration_precision.py for what each models")
    parser.add_argument("--tap-layer", type=int, default=H3_TAP_LAYER,
                        help="control: tap a different decoder layer")
    parser.add_argument("--perturb-weight", default=None,
                        help="control: relative nudge to one decoder projection")
    parser.add_argument(
        "--gpu-gib", type=float, default=14.0,
        help="weight budget on the accelerator; the rest offloads. The default "
             "leaves room for SDPA's transient allocation, which at float32 is "
             "quadratic in sequence length and OOMs at 20 GiB on long rows",
    )
    parser.add_argument("--sdpa-backend", default="auto",
                        choices=("auto", "math", "efficient"),
                        help="pin SDPA's kernel so a mask experiment is not "
                             "also a kernel experiment")
    parser.add_argument("--keep-attention-mask", action="store_true",
                        help="pass the all-ones mask instead of dropping it; "
                             "measures the SDPA math-backend cost")
    parser.add_argument("--reserve-vram-gib", type=float, default=0.0)
    args = parser.parse_args()

    if args.compare:
        return compare([Path(p).expanduser().resolve() for p in args.compare],
                       Path(args.bundle).expanduser().resolve() if args.bundle else None,
                       Path(args.out).expanduser().resolve(),
                       Path(args.reference).expanduser().resolve() if args.reference else None,
                       args.field_under_test)

    if not args.arm or not args.bundle:
        parser.error("--arm and --bundle are required unless --compare is used")
    if not args.source_dir:
        parser.error("--source-dir or H3_BF16_ENCODER_DIR is required")
    bundle = Path(args.bundle).expanduser().resolve()
    source = Path(args.source_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not (args.arm == "comfy" and args.all_rows):
        # With --all-rows the directory holds one subdirectory per row and a
        # rerun skips the rows already captured; a single-row capture still
        # refuses to overwrite.
        raise SystemExit(f"refuse to overwrite existing capture directory: {out}")

    if args.arm == "comfy":
        if args.all_rows and args.row:
            parser.error("--all-rows and --row are exclusive")
        if not args.all_rows and args.row is None:
            parser.error("the ComfyUI arm needs --row or --all-rows")
        run_comfy_arm(bundle, args.row, source, out, args.reserve_vram_gib,
                      args.tap_layer,
                      Path(args.w4_path).expanduser().resolve() if args.w4_path else None,
                      args.all_rows)
    else:
        if args.w4_path or args.all_rows:
            parser.error("--w4-path and --all-rows apply to the ComfyUI arm only")
        run_transformers_arm(bundle, args.row, source, out, args.dtype,
                             args.tap_layer, args.gpu_gib, args.perturb_weight,
                             args.keep_attention_mask, args.sdpa_backend,
                             args.attention)
    return 0


if __name__ == "__main__":
    sys.exit(main())
