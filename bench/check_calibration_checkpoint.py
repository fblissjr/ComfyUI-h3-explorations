#!/usr/bin/env python3
"""A resumed calibration must produce the same weights as an uninterrupted one.

`bench/h3_calibration_checkpoint.py` exists so a ten-hour run that dies costs
one layer. That is only worth having if the resumed run is the same run, so the
case this file is built around is the one the design note names as the proof:
an uninterrupted run produces A, a run killed at a boundary and resumed
produces B, and every tensor must be bit-identical.

**Run here at fixture scale on CPU, not on the candidate.** The full proof on
the real model needs the card and comes after Gate 5. This version drives the
real `SequentialPipeline` with the real recipe from `bench/h3_awq_recipe.py`
over a small random model, which exercises every seam the real one uses --
tracing, the subgraph slice, the restored cache, the layer restore, the
observers -- and can run any time. A green here does not license skipping the
card run; it means the mechanism is right and the card run is checking scale
and the real weights.

## One concession, stated because it changes what this proves

`AWQModifier._apply_smoothing` calls `IntermediatesCache.pin_memory`, which
raises without CUDA, so every case here neuters that call. Pinning is a
host-to-device transfer optimisation and does not enter the arithmetic, so the
weights compared below are the weights the pipeline computes. It does mean this
file has never run the AWQ arm exactly as the card runs it.

## What the red control owns

A resume against a checkpoint from a different bundle, recipe or model must
refuse before loading anything. Nothing about the mismatch is loud on its own:
the shapes agree, the run completes, and the artifact is neither of the two
runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

SEED = 20260825
LAYERS = 3


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _module("_h3_ckpt", REPO / "bench" / "h3_calibration_checkpoint.py")


def _neuter_pin_memory():
    from llmcompressor.pipelines.sequential import pipeline as sp

    sp.IntermediatesCache.pin_memory = lambda self, index: None


def _model():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(SEED)
    cfg = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64,
                      max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    # Uniform random weights give AWQ nothing to do: the grid picks the
    # identity scale, every mapping reports reduction 1.0, and no weight moves.
    # On such a fixture "resumed equals uninterrupted" passes even if resuming
    # re-smoothed a layer, because re-applying the identity is idempotent --
    # the check would be green over the exact defect it exists to catch. So the
    # fixture is given the per-channel imbalance AWQ is built to correct.
    generator = torch.Generator().manual_seed(SEED + 2)
    with torch.no_grad():
        for layer in model.model.layers:
            skew = torch.exp(torch.randn(cfg.hidden_size, generator=generator) * 2.5)
            layer.input_layernorm.weight.mul_(skew.to(torch.bfloat16))
            layer.post_attention_layernorm.weight.mul_(skew.to(torch.bfloat16))
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                         layer.self_attn.v_proj, layer.mlp.gate_proj,
                         layer.mlp.up_proj):
                proj.weight.div_(skew.to(torch.bfloat16).unsqueeze(0))
    return model


def _loader(batches: int = 2):
    import torch

    torch.manual_seed(SEED + 1)
    rows = [{"input_ids": torch.randint(0, 256, (1, 16)),
             "attention_mask": torch.ones(1, 16, dtype=torch.long)}
            for _ in range(batches)]

    class Loader:
        def __init__(self, rows):
            self.rows = rows

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

    return Loader(rows)


def _recipe():
    from h3_awq_recipe import build_recipe

    return build_recipe(offload_device="cpu", duo_scaling=False, n_grid=2,
                        ignore=["lm_head"])


def _fresh_session(model, recipe):
    from llmcompressor.core import active_session, reset_session

    reset_session()
    session = active_session()
    session.initialize(model=model, start=-1, recipe=recipe)
    return session


def _dataset_args():
    from llmcompressor.args import DatasetArguments

    return DatasetArguments(sequential_targets=["Qwen3DecoderLayer"])


def _identity(store, recipe, subgraphs: int, bundle="fixture-A", model_id="qwen3-tiny"):
    from h3_awq_recipe import describe_recipe

    return store.identity(recipe_description=describe_recipe(recipe),
                          bundle_id=bundle, model_id=model_id,
                          subgraph_count=subgraphs)


def _final_state(model):
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def _run_uninterrupted():
    from llmcompressor.pipelines.sequential import pipeline as sp

    model, recipe = _model(), _recipe()
    _fresh_session(model, recipe)
    sp.SequentialPipeline()(model, _loader(), _dataset_args())
    return _final_state(model)


class _Killed(RuntimeError):
    pass


def _run_until(store, kill_after_boundary: int, bundle="fixture-A"):
    """Checkpoint at every boundary and die after one, like a real failure."""
    from llmcompressor.pipelines.sequential import pipeline as sp

    model, recipe = _model(), _recipe()
    # Before initialize: `AWQModifier.on_initialize` fills in mappings inferred
    # from the model, so an identity taken afterwards never matches one taken
    # by a fresh process and every resume refuses itself.
    identity = _identity(store, recipe, LAYERS + 1, bundle=bundle)
    _fresh_session(model, recipe)
    holder: list = []

    def on_write(manifest):
        if manifest["next_subgraph"] > kill_after_boundary:
            raise _Killed(f"killed after subgraph {kill_after_boundary}")

    with C.capture_cache(holder), C.checkpoint_each_boundary(
            store, model=model, identity=identity, modifiers=recipe,
            cache_holder=holder, on_write=on_write):
        try:
            sp.SequentialPipeline()(model, _loader(), _dataset_args())
        except _Killed:
            pass
        else:
            raise AssertionError("the run was not interrupted; nothing to resume")
    return identity


def _run_resumed(store, bundle="fixture-A"):
    from llmcompressor.pipelines.sequential import pipeline as sp

    model, recipe = _model(), _recipe()
    identity = _identity(store, recipe, LAYERS + 1, bundle=bundle)
    _fresh_session(model, recipe)
    manifest = store.restore(model, identity)
    cache = manifest.pop("_cache")
    with C.resume_at(manifest["next_subgraph"], cache) as seen:
        sp.SequentialPipeline()(model, _loader(), _dataset_args())
    return _final_state(model), manifest, seen


# --------------------------------------------------------------------------


def cache_round_trip_is_exact():
    """The serializer's grammar covers what the real cache holds."""
    import torch
    from llmcompressor.pipelines.sequential import pipeline as sp

    model, recipe = _model(), _recipe()
    _fresh_session(model, recipe)
    holder: list = []
    captured = {}
    original_end = sp.LifecycleCallbacks.sequential_epoch_end

    def spy(modules, **kwargs):
        out = original_end(modules, **kwargs)
        if "cache" not in captured and holder:
            # Decode immediately: the live cache mutates in place as the run
            # proceeds, so comparing a snapshot against it later compares two
            # different instants and fails for the wrong reason.
            tree, tensors = C.encode_cache(holder[0])
            frozen = {k: v.detach().clone() for k, v in tensors.items()}
            captured["cache"] = (C.decode_cache(tree, frozen),
                                 _snapshot(holder[0]), len(tensors))
        return out

    sp.LifecycleCallbacks.sequential_epoch_end = staticmethod(spy)
    try:
        with C.capture_cache(holder):
            sp.SequentialPipeline()(model, _loader(), _dataset_args())
    finally:
        sp.LifecycleCallbacks.sequential_epoch_end = original_end

    restored, live, tensor_count = captured["cache"]
    assert tensor_count, "the cache carried no tensors; this case checked nothing"
    assert len(restored.batch_intermediates) == len(live.batch_intermediates)
    compared = 0
    for got, want in zip(restored.batch_intermediates, live.batch_intermediates):
        assert set(got) == set(want), (sorted(got), sorted(want))
        for name in want:
            a, b = got[name], want[name]
            assert a.device == b.device, (name, a.device, b.device)
            fa, fb = _flatten(a.value), _flatten(b.value)
            assert [type(x).__name__ for x in fa] == [type(x).__name__ for x in fb]
            for x, y in zip(fa, fb):
                if isinstance(x, torch.Tensor):
                    assert torch.equal(x, y), name
                    compared += 1
                else:
                    assert x == y, (name, x, y)
    return f"{tensor_count} tensor(s), {compared} compared exactly"


def _snapshot(cache):
    """A deep copy of the cache as it stands now, tensors included."""
    import copy

    import torch

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, (list, tuple)):
            out = [clone(v) for v in value]
            return out if isinstance(value, list) else tuple(out)
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        import dataclasses

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return type(value)(**{f.name: clone(getattr(value, f.name))
                                  for f in dataclasses.fields(value)})
        return copy.copy(value)

    from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue

    batches = [{k: IntermediateValue(value=clone(v.value), device=v.device)
                for k, v in entry.items()}
               for entry in cache.batch_intermediates]
    return IntermediatesCache(batch_intermediates=batches,
                              offload_device=cache.offload_device)


def _flatten(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flatten(v)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flatten(v)]
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [x for f in dataclasses.fields(value)
                for x in _flatten(getattr(value, f.name))]
    return [value]


def a_resumed_run_equals_an_uninterrupted_one():
    """The case this file exists for."""
    import torch

    reference = _run_uninterrupted()
    with tempfile.TemporaryDirectory(prefix="h3-ckpt-") as raw:
        store = C.CheckpointStore(Path(raw) / "ckpt")
        _run_until(store, kill_after_boundary=2)
        manifest = store.read_manifest()
        candidate, restored_manifest, seen = _run_resumed(store)

    assert restored_manifest["completed_layers"] >= 1, restored_manifest
    assert seen["kept"] == seen["total"] - manifest["next_subgraph"], seen
    assert set(candidate) == set(reference), "the two runs produced different tensors"
    moved = sorted(k for k in reference
                   if not torch.equal(reference[k].float(), candidate[k].float()))
    assert not moved, (
        f"{len(moved)} tensor(s) differ between the uninterrupted and resumed "
        f"runs, first: {moved[:4]}")
    return (f"{len(reference)} tensors identical; resumed at subgraph "
            f"{manifest['next_subgraph']} of {seen['total']}, "
            f"{manifest['completed_layers']} layer(s) restored")


def a_partial_resume_is_not_a_whole_one():
    """The proof above is only worth something if a wrong resume differs.

    Restores the checkpoint and then enters at the WRONG subgraph -- the one
    already completed. If that produced identical weights too, the comparison
    above would be passing on a pipeline whose start index does nothing.
    """
    import torch
    from llmcompressor.pipelines.sequential import pipeline as sp

    reference = _run_uninterrupted()
    with tempfile.TemporaryDirectory(prefix="h3-ckpt-wrong-") as raw:
        store = C.CheckpointStore(Path(raw) / "ckpt")
        _run_until(store, kill_after_boundary=2)
        manifest = store.read_manifest()
        model, recipe = _model(), _recipe()
        identity = _identity(store, recipe, LAYERS + 1)
        _fresh_session(model, recipe)
        restored = store.restore(model, identity)
        cache = restored.pop("_cache")
        wrong = manifest["next_subgraph"] - 1
        try:
            with C.resume_at(wrong, cache):
                sp.SequentialPipeline()(model, _loader(), _dataset_args())
        except ValueError as exc:
            # Also an acceptable outcome, and the better one: the cache does
            # not carry that subgraph's declared inputs, so it refuses.
            assert "inputs" in str(exc) or "subgraph" in str(exc), exc
            return f"entering at {wrong} was refused: {str(exc)[:60]}"
        candidate = _final_state(model)
    moved = [k for k in reference
             if not torch.equal(reference[k].float(), candidate[k].float())]
    assert moved, (
        "entering at the wrong subgraph produced identical weights, so the "
        "start index is not changing what runs and the equivalence case above "
        "proves nothing")
    return f"entering at {wrong} moved {len(moved)} tensor(s), as it must"


def the_report_covers_exactly_the_completed_layers():
    """A checkpoint's error metrics must not describe a layer it did not save.

    `_error_metrics` accumulates for the whole run, so where the snapshot sits
    relative to the epoch-end callback decides whether it already contains the
    layer that callback is about to smooth. Taken afterwards, the checkpoint
    claims metrics for a layer `completed_layers` excludes -- the resumed run
    re-runs it and reports it twice, and the run record says a layer was
    calibrated more often than it was.

    Measured 2026-08-25: this is the whole consequence of the placement. The
    weights are identical either way, because that layer is not restored in
    either case, so `resumed equals uninterrupted` cannot see this and it needs
    its own case.
    """
    with tempfile.TemporaryDirectory(prefix="h3-ckpt-report-") as raw:
        store = C.CheckpointStore(Path(raw) / "ckpt")
        _run_until(store, kill_after_boundary=2)
        manifest = store.read_manifest()
    metrics = manifest["error_metrics"]
    assert metrics, "no error metrics were carried; this case checked nothing"
    described = {e["layer_name"].split(".mlp")[0].rsplit(".", 1)[0]
                 if e["layer_name"].count(".") > 2 else e["layer_name"]
                 for e in metrics}
    layers = sorted({name.rsplit(".", 1)[0] if ".mlp" in name else name
                     for name in (e["layer_name"] for e in metrics)})
    indices = sorted({int(part) for name in layers
                      for part in name.split(".") if part.isdigit()})
    assert indices == list(range(manifest["completed_layers"])), (
        f"the checkpoint says {manifest['completed_layers']} layer(s) are "
        f"complete but carries metrics for layers {indices}; the snapshot is "
        "on the wrong side of the epoch-end callback")
    return (f"{len(metrics)} metric(s) covering layers {indices}, matching "
            f"completed_layers={manifest['completed_layers']}")


def a_foreign_checkpoint_is_refused():
    """The red control: a different bundle, recipe or model must not resume."""
    with tempfile.TemporaryDirectory(prefix="h3-ckpt-foreign-") as raw:
        store = C.CheckpointStore(Path(raw) / "ckpt")
        _run_until(store, kill_after_boundary=2, bundle="fixture-A")
        recipe = _recipe()

        refused = []
        for label, identity in (
            ("bundle", _identity(store, recipe, LAYERS + 1, bundle="fixture-B")),
            ("model", _identity(store, recipe, LAYERS + 1, model_id="other")),
            ("subgraph count", _identity(store, recipe, LAYERS + 99)),
        ):
            try:
                store.verify_compatible(identity)
            except ValueError as exc:
                assert "not written by this run" in str(exc), exc
                refused.append(label)
            else:
                raise AssertionError(f"a checkpoint from a different {label} resumed")

        # And a different recipe, built for real rather than by editing the id.
        from h3_awq_recipe import build_recipe, describe_recipe

        other = build_recipe(offload_device="cpu", duo_scaling=False, n_grid=7,
                             ignore=["lm_head"])
        different = store.identity(recipe_description=describe_recipe(other),
                                   bundle_id="fixture-A", model_id="qwen3-tiny",
                                   subgraph_count=LAYERS + 1)
        try:
            store.verify_compatible(different)
        except ValueError as exc:
            assert "recipe_sha256" in str(exc), exc
            refused.append("recipe")
        else:
            raise AssertionError("a checkpoint from a different recipe resumed")

        # The matching identity still passes, so the guard is not refusing all.
        store.verify_compatible(_identity(store, recipe, LAYERS + 1))
    return f"refused on {', '.join(refused)}; the matching identity still loads"


def an_interrupted_write_leaves_the_previous_checkpoint():
    """A half-written checkpoint is worse than none, so writes are atomic."""
    with tempfile.TemporaryDirectory(prefix="h3-ckpt-atomic-") as raw:
        store = C.CheckpointStore(Path(raw) / "ckpt")
        _run_until(store, kill_after_boundary=2)
        good = store.read_manifest()
        assert not store.root.with_name(store.root.name + ".partial").exists(), (
            "a staging directory survived a clean write")
        # A kill during the next write leaves the committed one untouched: the
        # staging directory is what gets abandoned.
        staging = store.root.with_name(store.root.name + ".partial")
        (staging / "layers").mkdir(parents=True)
        (staging / "manifest.json").write_text("{}")
        assert store.read_manifest() == good, (
            "an abandoned partial write changed what the store reads back")
    return f"committed manifest survives an abandoned write at subgraph {good['next_subgraph']}"


def main() -> int:
    try:
        _neuter_pin_memory()
    except Exception as exc:  # noqa: BLE001
        print(f"could not import the pinned llm-compressor pipeline: "
              f"{type(exc).__name__}: {exc}")
        return 2

    cases = [
        ("cache round trip", cache_round_trip_is_exact),
        ("resumed equals uninterrupted", a_resumed_run_equals_an_uninterrupted_one),
        ("wrong entry point differs", a_partial_resume_is_not_a_whole_one),
        ("report matches completed layers", the_report_covers_exactly_the_completed_layers),
        ("foreign checkpoint refused", a_foreign_checkpoint_is_refused),
        ("write is atomic", an_interrupted_write_leaves_the_previous_checkpoint),
    ]
    ok = True
    for label, case in cases:
        try:
            detail = case()
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {label}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
