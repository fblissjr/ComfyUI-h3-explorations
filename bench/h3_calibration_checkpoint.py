#!/usr/bin/env python3
"""Checkpoint a sequential AWQ calibration at a subgraph boundary, and resume.

A ten-hour run that dies costs the run. With this it costs one layer.
`docs/research/calibration_checkpoint_resume.md` is the design and carries the
state inventory this implements; the two findings that shaped it are that
`AWQModifier._parent_args_cache` holds zero filled batches at every boundary
(structure, not data, so it is not checkpointed) and that `_error_metrics`
accumulates for the whole run (so it is).

## The resume is two seams, not a reimplemented loop

`SequentialPipeline` iterates every subgraph from zero, so resuming needs a
start index and a preloaded cache. Both fall out of seams the pipeline already
has, and neither requires copying its loop:

- `trace_subgraphs` is wrapped to return `subgraphs[start:]`. The loop then
  begins at the boundary, and its own `subgraph_index < num_subgraphs - 1`
  test still correctly identifies the final subgraph.
- `IntermediatesCache.from_dataloader` is wrapped to return the restored cache
  instead of building one from the dataloader.

Copying the loop to add an index parameter would have been a second
implementation of the pipeline, drifting from the installed one on the first
upstream change. `bench/pilot_sequential_feasibility.py` already patches these
same two names for its instrumentation, so the seams are load-bearing and
observed, not invented here.

## Cadence, and why the default is not every layer

MEASURED 2026-08-25. Writing safetensors to this filesystem and syncing runs at
2.7 GB/s; the 6.0 GB/s an unsynced write reports is the page cache, not the
device. At the population's roughly 16 GiB cache a checkpoint is about 6 s, so
every-layer cadence costs under 7 minutes of writing across 64 layers --
nothing against ten hours.

Time is not what decides it. Overwriting one checkpoint 64 times writes about
1.1 TB per run, which is drive endurance, so the default is **every 4 layers**:
at most four layers lost on a failure, about 18 minutes at the measured rate,
and roughly 280 GB written. `every=1` stays available for a run that has
already failed once and should not lose four more. The staged-then-renamed
write means a second cache-sized copy exists on disk at the moment of the
rename, and that is the same size at any cadence.

## What refuses

A checkpoint carries the recipe description, the bundle identity and the model
identity. `verify_compatible` compares them before anything is loaded, because
a resume against a different bundle or a different recipe silently produces an
artifact that is neither run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "h3-calibration-checkpoint-v1"

# The closed grammar `IntermediatesCache._offload_value` can produce, read from
# the installed package on 2026-08-25. Anything outside it is refused loudly
# rather than dropped, because a cache restored with a value missing produces a
# run that is subtly not the interrupted one.
_PRIMITIVES = (int, str, float, bool, type(None))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# The intermediates cache: a shape tree plus one flat tensor store.


def _encode(value: Any, tensors: dict, path: str) -> Any:
    """One cache value to JSON, tensors lifted into `tensors` by path."""
    import torch

    if isinstance(value, torch.Tensor):
        tensors[path] = value
        return {"k": "tensor", "at": path}
    if isinstance(value, (list, tuple)):
        return {"k": "list" if isinstance(value, list) else "tuple",
                "v": [_encode(v, tensors, f"{path}.{i}")
                      for i, v in enumerate(value)]}
    if isinstance(value, dict):
        return {"k": "dict",
                "v": {str(key): _encode(v, tensors, f"{path}.{key}")
                      for key, v in value.items()}}
    if isinstance(value, torch.dtype):
        return {"k": "dtype", "v": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"k": "device", "v": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {"k": "dataclass",
                "cls": f"{cls.__module__}:{cls.__qualname__}",
                "v": {f.name: _encode(getattr(value, f.name), tensors,
                                      f"{path}.{f.name}")
                      for f in dataclasses.fields(value)}}
    if isinstance(value, _PRIMITIVES):
        return {"k": "primitive", "v": value}
    raise TypeError(
        f"cache value at {path} is {type(value).__name__}, outside the grammar "
        "IntermediatesCache produces. Refusing rather than dropping it: a "
        "restored cache missing a value is a run that is not the interrupted one."
    )


def _resolve(spec: str):
    module_name, _, qualname = spec.partition(":")
    import importlib

    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _decode(node: Any, tensors: dict) -> Any:
    import torch

    kind = node["k"]
    if kind == "tensor":
        return tensors[node["at"]]
    if kind in ("list", "tuple"):
        out = [_decode(v, tensors) for v in node["v"]]
        return out if kind == "list" else tuple(out)
    if kind == "dict":
        return {k: _decode(v, tensors) for k, v in node["v"].items()}
    if kind == "dtype":
        return getattr(torch, node["v"])
    if kind == "device":
        return torch.device(node["v"])
    if kind == "dataclass":
        cls = _resolve(node["cls"])
        return cls(**{k: _decode(v, tensors) for k, v in node["v"].items()})
    if kind == "primitive":
        return node["v"]
    raise ValueError(f"unknown cache node kind {kind!r}")


def encode_cache(cache) -> tuple[dict, dict]:
    """`(shape tree, tensors by path)` for an `IntermediatesCache`.

    The `IntermediateValue` wrapper is encoded explicitly rather than through
    the dataclass branch, so its `device` field survives as a device and the
    restored cache onloads to where the interrupted one would have.
    """
    tensors: dict = {}
    batches = []
    for batch_index, values in enumerate(cache.batch_intermediates):
        entry = {}
        for name, intermediate in values.items():
            base = f"b{batch_index}.{name}"
            entry[name] = {
                "value": _encode(intermediate.value, tensors, base),
                "device": (str(intermediate.device)
                           if intermediate.device is not None else None),
            }
        batches.append(entry)
    tree = {
        "batches": batches,
        "offload_device": (str(cache.offload_device)
                           if cache.offload_device is not None else None),
    }
    return tree, tensors


def decode_cache(tree: dict, tensors: dict):
    import torch
    from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue

    batches = []
    for entry in tree["batches"]:
        values = {}
        for name, node in entry.items():
            values[name] = IntermediateValue(
                value=_decode(node["value"], tensors),
                device=(torch.device(node["device"])
                        if node["device"] is not None else None),
            )
        batches.append(values)
    offload = tree["offload_device"]
    return IntermediatesCache(
        batch_intermediates=batches,
        offload_device=torch.device(offload) if offload is not None else None,
    )


# --------------------------------------------------------------------------


def layer_modules(model) -> list[tuple[str, Any]]:
    """`(name, module)` for each decoder layer, in order.

    Located by walking for the longest `ModuleList` of identically-typed
    modules rather than by a path string, so a model whose decoder sits
    somewhere other than `model.layers` still checkpoints.
    """
    import torch

    best: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        kinds = {type(child).__name__ for child in module}
        if len(kinds) != 1:
            continue
        if len(module) > len(best):
            best = [(f"{name}.{i}", child) for i, child in enumerate(module)]
    if not best:
        raise ValueError("no decoder ModuleList found on this model")
    return best


class CheckpointStore:
    """One directory: a manifest, one safetensors per layer, and the cache."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def layer_path(self, index: int) -> Path:
        return self.root / "layers" / f"layer_{index:04d}.safetensors"

    def identity(self, *, recipe_description, bundle_id: str,
                 model_id: str, subgraph_count: int) -> dict:
        """What a resume must match before it loads anything.

        Describe the recipe as AUTHORED, before `session.initialize`.
        `AWQModifier.on_initialize` fills in `mappings` inferred from the
        model, so a description taken afterwards differs from one taken before
        and every resume would refuse itself. Measured 2026-08-25. The resolved
        mappings are a function of the model, which `model_id` identifies
        separately, so nothing is lost by describing the authored form.
        """
        return {
            "recipe_sha256": _sha256_text(
                json.dumps(recipe_description, sort_keys=True, default=str)),
            "bundle_id": bundle_id,
            "model_id": model_id,
            "subgraph_count": int(subgraph_count),
        }

    def save(self, *, model, cache, next_subgraph: int, completed_layers: int,
             identity: dict, error_metrics: list, extra: dict | None = None) -> dict:
        """Write a boundary. Overwrites in place; the previous one is the cost.

        Written to a sibling directory and moved into place, so a kill during
        the write leaves the previous checkpoint intact rather than a truncated
        one. A checkpoint that can be half-written is worse than none: the
        resume would load it and produce an artifact nobody could account for.
        """
        import torch
        from safetensors.torch import save_file

        staging = self.root.with_name(self.root.name + ".partial")
        if staging.exists():
            _rmtree(staging)
        (staging / "layers").mkdir(parents=True)

        layers = layer_modules(model)
        for index, (name, module) in enumerate(layers[:completed_layers]):
            state = {k: v.detach().to("cpu").contiguous()
                     for k, v in module.state_dict().items()}
            save_file(state, str(staging / "layers" / f"layer_{index:04d}.safetensors"),
                      metadata={"module": name, "layer_index": str(index)})

        tree, tensors = encode_cache(cache)
        save_file({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()},
                  str(staging / "cache.safetensors"))
        (staging / "cache_shape.json").write_text(json.dumps(tree, indent=1))

        manifest = {
            "schema": SCHEMA,
            "identity": identity,
            "next_subgraph": int(next_subgraph),
            "completed_layers": int(completed_layers),
            "layer_names": [n for n, _ in layers[:completed_layers]],
            "cache_batches": len(cache.batch_intermediates),
            "cache_keys": sorted({k for b in cache.batch_intermediates for k in b}),
            "cache_tensors": len(tensors),
            "error_metrics": error_metrics,
            "pid": os.getpid(),
            **(extra or {}),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))

        if self.root.exists():
            _rmtree(self.root)
        staging.rename(self.root)
        return manifest

    def read_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"no checkpoint manifest at {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(
                f"checkpoint schema {manifest.get('schema')!r} is not {SCHEMA!r}")
        return manifest

    def verify_compatible(self, identity: dict) -> dict:
        """Refuse a checkpoint from a different bundle, recipe or model.

        Before anything is loaded. A resume across a mismatch does not fail
        loudly on its own -- the shapes agree -- it produces an artifact that is
        neither of the two runs.
        """
        manifest = self.read_manifest()
        theirs = manifest.get("identity") or {}
        differing = sorted(k for k in set(identity) | set(theirs)
                           if identity.get(k) != theirs.get(k))
        if differing:
            raise ValueError(
                "this checkpoint was not written by this run: "
                + "; ".join(f"{k}: checkpoint {theirs.get(k)!r} != now "
                            f"{identity.get(k)!r}" for k in differing)
            )
        return manifest

    def restore(self, model, identity: dict) -> dict:
        """Apply the saved layers and return the manifest plus the cache."""
        from safetensors.torch import load_file

        manifest = self.verify_compatible(identity)
        layers = layer_modules(model)
        for index in range(manifest["completed_layers"]):
            path = self.layer_path(index)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint is missing {path.name}")
            name, module = layers[index]
            if name != manifest["layer_names"][index]:
                raise ValueError(
                    f"layer {index} is {name!r} now and was "
                    f"{manifest['layer_names'][index]!r} at checkpoint time")
            state = load_file(str(path), device="cpu")
            missing, unexpected = module.load_state_dict(state, strict=False)
            # `strict=False` is required because the saved layer carries the
            # quantization parameters the fresh model does not have until the
            # config is applied. Anything else absent is a real mismatch.
            unresolved = [k for k in missing
                          if not k.endswith(("weight_scale", "weight_zero_point"))]
            if unresolved or unexpected:
                raise ValueError(
                    f"layer {index} did not restore cleanly: missing="
                    f"{unresolved[:4]} unexpected={list(unexpected)[:4]}")
        tensors = load_file(str(self.root / "cache.safetensors"), device="cpu")
        tree = json.loads((self.root / "cache_shape.json").read_text())
        manifest["_cache"] = decode_cache(tree, tensors)
        return manifest


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


# --------------------------------------------------------------------------
# The two seams.


@contextlib.contextmanager
def resume_at(start_subgraph: int, cache):
    """Enter `SequentialPipeline`'s loop at `start_subgraph` with `cache` loaded.

    Wraps `trace_subgraphs` to drop the completed subgraphs and
    `IntermediatesCache.from_dataloader` to hand back the restored cache. The
    pipeline's own loop is untouched, so it does not drift from the installed
    one. Both names are restored on exit even if the run raises.
    """
    from llmcompressor.pipelines.sequential import pipeline as sp

    original_trace = sp.trace_subgraphs
    original_from = sp.IntermediatesCache.from_dataloader
    seen: dict = {}

    def trace(model, sample_input, *args, **kwargs):
        subgraphs = original_trace(model, sample_input, *args, **kwargs)
        seen["total"] = len(subgraphs)
        if not 0 <= start_subgraph <= len(subgraphs):
            raise ValueError(
                f"checkpoint resumes at subgraph {start_subgraph} but this "
                f"model traces to {len(subgraphs)}; the checkpoint is not for "
                "this model or this tracing configuration")
        kept = subgraphs[start_subgraph:]
        declared = sorted({n for s in kept[:1] for n in s.input_names})
        have = sorted({k for b in cache.batch_intermediates for k in b})
        absent = [n for n in declared if n not in have]
        if absent:
            raise ValueError(
                f"the restored cache does not carry subgraph {start_subgraph}'s "
                f"inputs {absent}; the traced names moved, so resuming here "
                "would feed the wrong activations")
        seen["kept"] = len(kept)
        return kept

    def from_dataloader(dataloader, model_device=None, offload_device=None):
        return cache

    sp.trace_subgraphs = trace
    sp.IntermediatesCache.from_dataloader = staticmethod(from_dataloader)
    try:
        yield seen
    finally:
        sp.trace_subgraphs = original_trace
        sp.IntermediatesCache.from_dataloader = original_from


@contextlib.contextmanager
def checkpoint_each_boundary(store: CheckpointStore, *, model, identity,
                             modifiers, cache_holder, first_subgraph: int = 0,
                             extra: dict | None = None, on_write=None,
                             every: int = 4):
    """Write a checkpoint every `every` completed layers.

    `every` is a cadence in LAYERS, not subgraphs, because that is the unit of
    lost work: a failure costs at most `every` layers. The default trades a
    little more redone work for a lot less disk written; see the cadence
    section above for the measurement behind it. `every=1` writes at every
    boundary.

    `cache_holder` is a one-element list the caller fills with the live
    `IntermediatesCache`; the pipeline builds it internally, so it is captured
    through the same `from_dataloader` seam rather than passed in.

    A subgraph's epoch end is the boundary: smoothing and the weight observers
    have run for the layer it covers, and the cache holds the inputs to the
    next one. `docs/research/calibration_checkpoint_resume.md` records why
    nothing else has to be captured there.
    """
    from llmcompressor.pipelines.sequential import pipeline as sp

    original_end = sp.LifecycleCallbacks.sequential_epoch_end
    awq = next((m for m in modifiers if hasattr(m, "_error_metrics")), None)
    if int(every) < 1:
        raise ValueError(f"checkpoint cadence must be at least 1 layer, got {every!r}")
    state = {"index": first_subgraph, "written": [], "every": int(every)}

    def sequential_epoch_end(modules, **kwargs):
        # BEFORE the callback, not after. Measured on the installed pipeline:
        # at the top of subgraph k's epoch end the cache still holds the inputs
        # to subgraph k (the calibration pass reads without updating, and the
        # propagation pass that writes subgraph k's outputs runs after this
        # call), and layer k-1 has not yet been smoothed. So this instant is
        # exactly resumable at subgraph k with layers 0..k-2 complete.
        #
        # Snapshotting after the callback instead records `_error_metrics`
        # that already include layer k-1, while `completed_layers` says k-1
        # layers are done -- so the resumed run re-runs that layer and reports
        # it twice. Measured, not assumed: the weights are unaffected, because
        # layer k-1 is not among the ones restored either way. The claim that
        # it would be double-smoothed was wrong and a mutation refuted it;
        # `bench/check_calibration_checkpoint.py` now owns the real one.
        index = state["index"]
        cache = cache_holder[0] if cache_holder else None
        completed = index - 1
        due = completed >= 1 and completed % max(1, int(every)) == 0
        if cache is not None and index >= 1 and due:
            manifest = store.save(
                model=model, cache=cache, next_subgraph=index,
                completed_layers=index - 1, identity=identity,
                error_metrics=list(getattr(awq, "_error_metrics", []) or []),
                extra=extra,
            )
            state["written"].append(manifest["next_subgraph"])
            if on_write is not None:
                on_write(manifest)
        state["index"] = index + 1
        return original_end(modules, **kwargs)

    sp.LifecycleCallbacks.sequential_epoch_end = staticmethod(sequential_epoch_end)
    try:
        yield state
    finally:
        sp.LifecycleCallbacks.sequential_epoch_end = original_end


@contextlib.contextmanager
def capture_cache(holder: list):
    """Hand the caller the `IntermediatesCache` the pipeline builds."""
    from llmcompressor.pipelines.sequential import pipeline as sp

    original = sp.IntermediatesCache.from_dataloader

    def from_dataloader(dataloader, model_device=None, offload_device=None):
        cache = original(dataloader, model_device, offload_device)
        holder.clear()
        holder.append(cache)
        return cache

    sp.IntermediatesCache.from_dataloader = staticmethod(from_dataloader)
    try:
        yield holder
    finally:
        sp.IntermediatesCache.from_dataloader = original
