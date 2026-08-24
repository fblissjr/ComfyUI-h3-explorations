#!/usr/bin/env python3
"""Prove the bundle is what `oneshot` consumes, and show the proof can fail.

Gate 1, seam half. `build_native_h3_calibration_batch.py` produces a bundle in
the ComfyUI virtualenv; this consumes it in the pinned `llm-compressor` one and
follows each batch through the exact path a calibration run puts it through:

    bundle file -> DataLoader -> IntermediatesCache -> traced subgraph inputs

re-deriving the hash at every hop. Those two virtualenvs cannot import each
other -- ComfyUI has no `llmcompressor`, and `comfy` needs `comfy_aimdo`, which
the pinned quantization environment does not have and must not acquire for a
convenience. So the seam is a hashed bundle rather than one live object, and
this script exists to make that a checkable claim instead of an assumption.

What it establishes:

1. **File identity.** Every batch file and every tensor in it hashes to what the
   builder recorded.
2. **Dataloader identity.** `next(iter(dataloader))` -- the object
   `SequentialPipeline` traces from -- and every later batch re-hash to the
   same values. A preconstructed `DataLoader` is returned verbatim by
   `get_calibration_dataloader`, so no sampler or collator sits in between.
3. **Cache identity.** After `IntermediatesCache.from_dataloader`, what
   `fetch` returns for subgraph 0's declared inputs still hashes the same.
4. **Trace envelope.** The traced graph declares `pixel_values`,
   `image_grid_thw` and `mm_token_type_ids`, every row runs against the graph
   traced from the first one, and a perturbation assay shows the media actually
   reaches the language stack rather than being silently dropped.
5. **M-RoPE agreement on the real batches.** Transformers' own
   `get_rope_index` on each batch against the ComfyUI position ids the builder
   recorded. The general implementation-parity gate is closed
   (`canonical/2026-08-24_transformers_comfy_parity.md`); this is its instance
   on the actual calibration objects, which is a different claim.
6. **DeepStack placement.** The released tower declares three mergers, so
   features are injected after decoder layers 0, 1 and 2 -- inside the 0--49
   window H3 consumes and AWQ quantizes.
7. **Image-keyed versus video-keyed blocks.** The vendor's serving stack labels
   H3 two-frame blocks with `<|video_pad|>` while this batch labels them
   `<|image_pad|>`, matching ComfyUI. Whether that changes anything is measured
   here rather than argued from source.

Every one of those has a matching violation. Mutated bundles built by
`--mutate` must fail check 1--3, a text-only row must fail the trace envelope,
and the builder-disconnect control feeds the traced graph a batch the record
does not describe.

Run it with the `llm-compressor` virtualenv python. CPU only, no 32B weights;
the trace uses a reduced-width Qwen3-VL at released vision geometry.

    coderef/llm-compressor/.venv/bin/python bench/prove_calibration_seam.py \\
        --bundle <clean bundle> --mutated-bundle <name>=<dir> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from torch.utils.data import DataLoader

import llmcompressor
from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

BENCH = Path(__file__).resolve().parent
REPORT = BENCH / "results" / "2026-08-24_calibration_seam_proof.json"
SEQUENTIAL_TARGETS = ["Qwen3VLTextDecoderLayer"]
IMAGE_PAD, VIDEO_PAD = 151655, 151656
VISION_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids")


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# the dataloader a launcher would hand to oneshot


class _BundleDataset(torch.utils.data.Dataset):
    """The bundle's rows, in the manifest's declared order.

    Deterministic by construction: the order is the manifest's, not a sampler's.
    `active_plan.md` requires the row trace to come from the exact post-policy
    dataloader the launcher consumes, so there is only one ordering and it is
    written down.
    """

    def __init__(self, bundle: Path, manifest: dict):
        self.bundle = bundle
        self.records = {r["row_id"]: r for r in manifest["rows"]}
        self.order = [r for r in manifest["order"] if r in self.records]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> dict:
        record = self.records[self.order[index]]
        return load_file(self.bundle / record["batch_file"])


def _identity_collate(rows: list[dict]) -> dict:
    """One row per batch, passed through untouched.

    A callable collator short-circuits `_make_collate_fn`, which is also why
    `max_seq_length` is inert on this path -- measured and recorded in
    `canonical/2026-08-24_calibration_input_seam.md`. Sequence policy is the
    manifest's job, and batching more than one row would pad or truncate,
    so this refuses instead.
    """
    if len(rows) != 1:
        raise ValueError(
            f"the H3 calibration dataloader is one row per batch; got {len(rows)}. "
            "Batching would require padding or truncation, which would calibrate "
            "on positions the deployed path never produces."
        )
    return rows[0]


def build_dataloader(bundle: Path, manifest: dict) -> DataLoader:
    """The dataloader, routed through `llm-compressor`'s own entry point.

    Passing it as `DatasetArguments.dataset` and letting
    `get_calibration_dataloader` hand it back is the point: it demonstrates
    that the library returns this object verbatim, rather than asserting so
    from a source reading.
    """
    dataset = _BundleDataset(bundle, manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=_identity_collate)
    returned = get_calibration_dataloader(
        DatasetArguments(dataset=loader), processor=None
    )
    if returned is not loader:
        raise AssertionError(
            "get_calibration_dataloader did not return the preconstructed "
            "DataLoader; the seam assumption is wrong for this revision"
        )
    return returned


# --------------------------------------------------------------------------
# the reduced-width model the trace runs against


def tiny_qwen3vl(released: dict):
    """Released vision geometry and DeepStack depth, reduced hidden width.

    The trace envelope is decided by which keys exist in the first batch and by
    the module structure, not by hidden size. `deepstack_visual_indexes` keeps
    its released length, because how many decoder layers receive an injection
    is one of the things this proves.
    """
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=262144),
        vision_config=dict(depth=released["depth"], hidden_size=32,
                           intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=released["patch_size"],
                           temporal_patch_size=released["temporal_patch_size"],
                           spatial_merge_size=released["spatial_merge_size"],
                           deepstack_visual_indexes=released["deepstack_visual_indexes"],
                           num_position_embeddings=released["num_position_embeddings"]),
        image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD,
        vision_start_token_id=151652, vision_end_token_id=151653,
    )
    return Qwen3VLForConditionalGeneration(config).eval(), config


RELEASED_VISION = {
    "depth": 27,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "deepstack_visual_indexes": [8, 16, 24],
    "num_position_embeddings": 2304,
}


# --------------------------------------------------------------------------
# checks


def check_file_identity(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    failures, rows = [], {}
    for record in manifest["rows"]:
        path = bundle / record["batch_file"]
        actual = _file_sha(path)
        entry = {"batch_file_sha256": actual,
                 "matches_record": actual == record["batch_file_sha256"],
                 "tensors": {}}
        if not entry["matches_record"]:
            failures.append(f"{record['row_id']}: batch file hash differs from the record")
        loaded = load_file(path)
        declared = record["batch_tensors"]
        if sorted(loaded) != sorted(declared):
            failures.append(
                f"{record['row_id']}: batch keys {sorted(loaded)} != recorded {sorted(declared)}"
            )
        for key, value in loaded.items():
            got = _tensor_sha(value)
            want = declared.get(key, {}).get("sha256")
            entry["tensors"][key] = {"sha256": got, "matches_record": got == want}
            if got != want:
                failures.append(f"{record['row_id']}: tensor {key} hash differs from the record")
        rows[record["row_id"]] = entry
    return rows, failures


def check_dataloader_and_cache(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    failures = []
    loader = build_dataloader(bundle, manifest)
    dataset = loader.dataset
    declared = {r["row_id"]: r["batch_tensors"] for r in manifest["rows"]}

    first = next(iter(loader))
    first_row = dataset.order[0]
    for key, value in first.items():
        if _tensor_sha(value) != declared[first_row][key]["sha256"]:
            failures.append(
                f"{first_row}: the batch the pipeline traces from does not match "
                f"the record on {key}"
            )

    yielded = {}
    for index, batch in enumerate(loader):
        row_id = dataset.order[index]
        yielded[row_id] = {k: _tensor_sha(v) for k, v in batch.items()}
        for key, sha in yielded[row_id].items():
            if sha != declared[row_id][key]["sha256"]:
                failures.append(f"{row_id}: dataloader batch differs from the record on {key}")

    cache = IntermediatesCache.from_dataloader(loader, torch.device("cpu"), torch.device("cpu"))
    cached = {}
    for index, row_id in enumerate(dataset.order):
        fetched = cache.fetch(index, list(declared[row_id]))
        cached[row_id] = {k: _tensor_sha(v) for k, v in fetched.items()}
        if sorted(fetched) != sorted(declared[row_id]):
            failures.append(
                f"{row_id}: IntermediatesCache returned {sorted(fetched)}, "
                f"recorded {sorted(declared[row_id])}"
            )
        for key, sha in cached[row_id].items():
            if sha != declared[row_id][key]["sha256"]:
                failures.append(f"{row_id}: cache fetch differs from the record on {key}")

    return {"dataloader": yielded, "intermediates_cache": cached,
            "rows": len(dataset)}, failures


def check_trace_envelope(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    failures = []
    model, _ = tiny_qwen3vl(RELEASED_VISION)
    loader = build_dataloader(bundle, manifest)
    dataset = loader.dataset

    sample = next(iter(loader))
    subgraphs = trace_subgraphs(model, sample, SEQUENTIAL_TARGETS, [], 1)
    first = subgraphs[0]
    declared_inputs = sorted(first.input_names)
    result = {
        "traced_from": dataset.order[0],
        "subgraph_count": len(subgraphs),
        "subgraph0_input_names": declared_inputs,
        "rows": {},
    }
    for key in VISION_KEYS:
        if key not in first.input_names:
            failures.append(
                f"the traced graph does not declare {key}; media cannot reach "
                f"the language stack (this is the silent-drop configuration)"
            )

    for index, batch in enumerate(loader):
        row_id = dataset.order[index]
        inputs = {k: v for k, v in batch.items() if k in first.input_names}
        dropped = sorted(set(batch) - set(inputs))
        entry = {"dropped_by_trace": dropped,
                 "missing_for_trace": sorted(set(first.input_names) - set(inputs))}
        try:
            with torch.no_grad():
                outputs = first.forward(model, **inputs)
            entry["outcome"] = "ran"
            entry["media_influence"] = _media_influence(model, first, batch)
            if entry["media_influence"] is not None and entry["media_influence"] == 0.0:
                failures.append(
                    f"{row_id}: perturbing the pixels changed nothing; the media "
                    f"never reached the language stack"
                )
            del outputs
        except Exception as exc:  # a raise here is a real failure, not a control
            entry["outcome"] = "raised"
            entry["error"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
            failures.append(f"{row_id}: subgraph 0 raised: {entry['error']}")
        if dropped:
            failures.append(f"{row_id}: the trace dropped {dropped}")
        result["rows"][row_id] = entry
    return result, failures


def _media_influence(model, subgraph, batch: dict) -> float | None:
    """Identical tokens, perturbed pixels. Zero means the media was dropped."""
    if "pixel_values" not in batch:
        return None
    outputs = []
    for scale in (1.0, -1.0):
        perturbed = dict(batch)
        perturbed["pixel_values"] = batch["pixel_values"] * scale
        inputs = {k: v for k, v in perturbed.items() if k in subgraph.input_names}
        with torch.no_grad():
            out = subgraph.forward(model, **inputs)
        tensors = [v for v in out.values()
                   if torch.is_tensor(v) and v.is_floating_point() and v.dim() >= 2]
        outputs.append(max(tensors, key=lambda t: t.numel()))
    return float((outputs[0] - outputs[1]).abs().max())


def check_mrope(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    """Transformers' own position ids against the ComfyUI ones, per row."""
    model, _ = tiny_qwen3vl(RELEASED_VISION)
    rows = {}
    for record in manifest["rows"]:
        batch = load_file(bundle / record["batch_file"])
        if "image_grid_thw" not in batch:
            rows[record["row_id"]] = {"covered": False, "reason": "no vision block"}
            continue
        position_ids, _ = model.model.get_rope_index(
            input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=batch["attention_mask"],
        )
        # ComfyUI stores (3, seq) float; transformers returns (3, batch, seq) long.
        theirs = position_ids[:, 0, :].to(torch.float32)
        sha = _tensor_sha(theirs)
        rows[record["row_id"]] = {
            "covered": True,
            "transformers_position_ids_sha256": sha,
            "comfy_position_ids_sha256": record["position_ids_sha256"],
            "equal": sha == record["position_ids_sha256"],
            "max_position": int(position_ids.max()),
        }
    bad = [row for row, value in rows.items() if value.get("covered") and not value["equal"]]
    return rows, ([f"M-RoPE differs from the ComfyUI record on {bad}"] if bad else [])


def check_deepstack(manifest: dict) -> tuple[dict, list[str]]:
    """How many decoder layers receive an injection, and which.

    `Qwen3VLTextModel` injects DeepStack features after decoder layer `i` for
    each `i` below the number of mergers, so the released three mergers land on
    layers 0, 1 and 2. Those are inside the 0--49 window H3 consumes and inside
    the 64 layers the candidate quantizes, which is why bypassing the vision
    tower would collect statistics from a distribution the deployed model never
    produces.
    """
    model, _ = tiny_qwen3vl(RELEASED_VISION)
    mergers = len(model.model.visual.deepstack_merger_list)
    recorded = {r["row_id"]: r["deepstack_feature_count"] for r in manifest["rows"]}
    failures = []
    if mergers != len(RELEASED_VISION["deepstack_visual_indexes"]):
        failures.append(
            f"the model built {mergers} DeepStack mergers, released geometry "
            f"declares {len(RELEASED_VISION['deepstack_visual_indexes'])}"
        )
    injected = list(range(mergers))
    if max(injected) >= 50:
        failures.append("DeepStack injects outside the 0-49 window H3 consumes")
    mismatched = [row for row, count in recorded.items() if count != mergers]
    if mismatched:
        failures.append(
            f"the ComfyUI record carries a different DeepStack feature count on {mismatched}"
        )
    return {
        "vision_mergers": mergers,
        "injected_after_decoder_layers": injected,
        "h3_consumes_layers": "0-49",
        "candidate_quantizes_layers": "0-63",
        "comfy_recorded_feature_counts": recorded,
    }, failures


def check_video_keying(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    """Does labelling an H3 block video rather than image change anything?

    The vendor's serving stack emits `<|video_pad|>` for reference-video blocks
    (`coderef/sglang/.../minimax_h3/presentation.py`), while ComfyUI splices
    embeddings and never materialises a pad id at all, so the calibration batch
    has to choose one. `get_rope_index` reads modality 1 and 2 through the same
    `get_vision_position_ids` call and `get_video_features` is documented as
    "same implementation as for images", which predicts no difference at
    `grid_t = 1`. That is a source reading; this measures it.
    """
    video_rows = [
        r for r in manifest["rows"]
        if any(item["type"] == "video" for item in r["ordered_media"])
    ]
    if not video_rows:
        return {"covered": False, "reason": "no reference-video row in the bundle"}, []

    model, _ = tiny_qwen3vl(RELEASED_VISION)
    record = video_rows[0]
    batch = load_file(bundle / record["batch_file"])
    image_keyed, _ = model.model.get_rope_index(
        input_ids=batch["input_ids"],
        mm_token_type_ids=batch["mm_token_type_ids"],
        image_grid_thw=batch["image_grid_thw"],
        attention_mask=batch["attention_mask"],
    )
    video_ids = batch["input_ids"].clone()
    video_ids[video_ids == IMAGE_PAD] = VIDEO_PAD
    video_types = batch["mm_token_type_ids"].clone()
    video_types[video_types == 1] = 2
    video_keyed, _ = model.model.get_rope_index(
        input_ids=video_ids,
        mm_token_type_ids=video_types,
        video_grid_thw=batch["image_grid_thw"].clone(),
        attention_mask=batch["attention_mask"],
    )
    equal = bool(torch.equal(image_keyed, video_keyed))
    return {
        "covered": True,
        "row_id": record["row_id"],
        "blocks": len(record["vision_blocks"]),
        "grid_t": sorted({b["grid_thw"][0][0] for b in record["vision_blocks"]}),
        "position_ids_identical": equal,
        "max_abs_delta": float((image_keyed - video_keyed).abs().max()),
        "reading": "vendor serving labels H3 video blocks <|video_pad|>; this "
                   "batch labels them <|image_pad|>, matching ComfyUI",
    }, []


# Which gate owns which defect. A mutation that changes only the H3 token tags
# cannot break this chain and must not be asked to: token tags are returned to
# the DiT as `minimax_token_tags` for adaLN and never enter Qwen, so they are
# not part of the object `oneshot` consumes. `check_native_h3_presentation.py`
# owns that one. Listing it here as "expected not detected" is a specificity
# claim as well as an honest one: a chain that flagged every mutation would be
# flagging the bundle rather than the defect.
SEAM_BLIND = {
    "token-tags-flip": "H3 token tags are a DiT-side quantity and are absent "
                       "from the calibration batch; graded by "
                       "check_native_h3_presentation.py instead",
}


def check_mutated(bundle: Path, manifest: dict, name: str,
                  mutated: Path) -> tuple[dict, list[str]]:
    """A mutated bundle graded against the clean record.

    The clean manifest is the declared truth; the mutated bundle is what a
    defective builder would have produced. Every mutation that touches the
    batch must break at least one hop of the identity chain, and
    `first-image-only` and `drop-media` must additionally strip a row's vision
    keys, which is the silent-drop configuration `active_plan.md` names as a
    stop condition. A mutation in `SEAM_BLIND` must do the opposite: leave the
    chain intact, because its defect lives outside the batch.
    """
    mutated_manifest = json.loads((mutated / "presentation.json").read_text())
    clean = {r["row_id"]: r for r in manifest["rows"]}
    broken, vision_lost = [], []
    for record in mutated_manifest["rows"]:
        row_id = record["row_id"]
        if row_id not in clean:
            continue
        loaded = load_file(mutated / record["batch_file"])
        declared = clean[row_id]["batch_tensors"]
        if sorted(loaded) != sorted(declared):
            broken.append(f"{row_id}: keys {sorted(loaded)} != {sorted(declared)}")
            if "pixel_values" in declared and "pixel_values" not in loaded:
                vision_lost.append(row_id)
            continue
        for key, value in loaded.items():
            if _tensor_sha(value) != declared[key]["sha256"]:
                broken.append(f"{row_id}: {key}")
    detected = bool(broken)
    expected = name not in SEAM_BLIND
    entry = {
        "mutation": name,
        "detected": detected,
        "expected_detected": expected,
        "broken_fields": broken[:8],
        "rows_that_lost_vision_keys": vision_lost,
    }
    if name in SEAM_BLIND:
        entry["out_of_scope_reason"] = SEAM_BLIND[name]
    if detected == expected:
        return entry, []
    if expected:
        return entry, [
            f"mutation {name} passed the seam identity chain; the proof cannot fail"
        ]
    return entry, [
        f"mutation {name} was expected to be invisible to the seam chain but "
        f"broke it; the chain is reacting to something other than the batch"
    ]


def check_builder_disconnect(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    """Feed the traced graph a batch the record does not describe.

    The rejected preflight validated a builder the launcher never called. The
    equivalent defect here is a dataloader whose rows are not the validated
    ones, so this swaps two rows' batch files behind the manifest's back and
    requires the identity chain to say so.
    """
    if len(manifest["rows"]) < 2:
        return {"covered": False, "reason": "needs two rows"}, []
    swapped = json.loads(json.dumps(manifest))
    a, b = swapped["rows"][0], swapped["rows"][1]
    a["batch_file"], b["batch_file"] = b["batch_file"], a["batch_file"]
    _, failures = check_file_identity(bundle, swapped)
    detected = bool(failures)
    return {
        "covered": True,
        "swapped": [a["row_id"], b["row_id"]],
        "detected": detected,
        "example": failures[0] if failures else None,
    }, ([] if detected else
        ["swapping two rows' batch files went undetected; the record does not "
         "bind a row to its tensors"])


def check_text_only_row(bundle: Path, manifest: dict) -> tuple[dict, list[str]]:
    """The stop condition, reproduced on real batches.

    `canonical/2026-08-24_calibration_input_seam.md` measured this on synthetic
    rows: a text-only row fed to a vision trace raises, and a vision row fed to
    a text-only trace is silently stripped. This repeats it on the real bundle,
    because the population this run traces is the one that matters.
    """
    model, _ = tiny_qwen3vl(RELEASED_VISION)
    loader = build_dataloader(bundle, manifest)
    sample = next(iter(loader))
    subgraphs = trace_subgraphs(model, sample, SEQUENTIAL_TARGETS, [], 1)
    first = subgraphs[0]

    text_only = {k: v for k, v in sample.items() if k not in ("pixel_values", "image_grid_thw")}
    text_only["mm_token_type_ids"] = torch.zeros_like(sample["mm_token_type_ids"])
    inputs = {k: v for k, v in text_only.items() if k in first.input_names}
    try:
        with torch.no_grad():
            first.forward(model, **inputs)
        outcome, error = "ran", None
    except Exception as exc:
        outcome = "raised"
        error = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
    return {
        "outcome": outcome,
        "error": error,
        "expected": "raised",
    }, ([] if outcome == "raised" else
        ["a text-only row ran against the vision trace without raising; the "
         "loud failure this population depends on is absent"])


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--mutated-bundle", action="append", default=[],
                        metavar="NAME=DIR", help="a deliberately defective bundle")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / "presentation.json").read_text())

    report: dict = {
        "proof": "native-H3 calibration seam identity through llm-compressor",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "llmcompressor": llmcompressor.__version__,
        },
        "bundle_provenance": manifest["provenance"],
    }
    failures: list[str] = []

    print("1. batch files and tensors against the record")
    report["file_identity"], f = check_file_identity(bundle, manifest)
    failures += f
    print(f"   {len(report['file_identity'])} rows, {len(f)} problems")

    print("2. dataloader and intermediates cache")
    report["dataloader_identity"], f = check_dataloader_and_cache(bundle, manifest)
    failures += f
    print(f"   {report['dataloader_identity']['rows']} rows, {len(f)} problems")

    print("3. trace envelope and media influence")
    report["trace"], f = check_trace_envelope(bundle, manifest)
    failures += f
    print(f"   subgraph0 declares {report['trace']['subgraph0_input_names']}")
    for row_id, entry in report["trace"]["rows"].items():
        print(f"   {row_id:<34} {entry['outcome']:<7} "
              f"media influence {entry.get('media_influence')}")

    print("4. M-RoPE against the ComfyUI record")
    report["mrope"], f = check_mrope(bundle, manifest)
    failures += f
    for row_id, entry in report["mrope"].items():
        print(f"   {row_id:<34} "
              f"{'equal' if entry.get('equal') else entry.get('reason', 'DIFFERS')}")

    print("5. DeepStack placement")
    report["deepstack"], f = check_deepstack(manifest)
    failures += f
    print(f"   {report['deepstack']['vision_mergers']} mergers, injected after "
          f"layers {report['deepstack']['injected_after_decoder_layers']}")

    print("6. image-keyed versus video-keyed H3 blocks")
    report["video_keying"], f = check_video_keying(bundle, manifest)
    failures += f
    print(f"   {report['video_keying']}")

    print("7. controls")
    report["controls"] = {}
    report["controls"]["text_only_row"], f = check_text_only_row(bundle, manifest)
    failures += f
    print(f"   text-only row against the vision trace: "
          f"{report['controls']['text_only_row']['outcome']}")

    report["controls"]["builder_disconnect"], f = check_builder_disconnect(bundle, manifest)
    failures += f
    print(f"   builder disconnect detected: "
          f"{report['controls']['builder_disconnect'].get('detected')}")

    report["controls"]["mutations"] = {}
    for spec in args.mutated_bundle:
        name, _, directory = spec.partition("=")
        entry, f = check_mutated(bundle, manifest, name, Path(directory).expanduser().resolve())
        failures += f
        report["controls"]["mutations"][name] = entry
        verdict = "as expected" if entry["detected"] == entry["expected_detected"] else "WRONG"
        print(f"   mutation {name:<22} detected={str(entry['detected']):<5} "
              f"expected={str(entry['expected_detected']):<5} {verdict}; "
              f"lost vision keys on {entry['rows_that_lost_vision_keys'] or '-'}")

    report["failures"] = failures
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{REPORT.name}")

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: every hop from bundle file to traced subgraph preserves the "
          "recorded batch, and every control fails as intended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
