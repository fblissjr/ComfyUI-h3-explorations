#!/usr/bin/env python3
"""Compare aligned MiniMax H3 layer-50 encoder captures.

This is the numerical half of the BF16-versus-W4 benchmark.  It never loads a
model.  Each capture arm must have been written independently by
``capture_h3_encoder_states.py``.

The escaped defect that justifies this gate is the rejected 2026-08-24 AWQ v2
preflight: its report called six fixtures parity tests although five never
compared the claimed reference path, and its row trace did not describe the
dataset the launcher would consume.  This comparator therefore joins arms only
on recorded observables, not fixture names or intent.

For a weight-only comparison, direct rowwise metrics are refused unless both
arms record identical:

* expanded token ids and sequence length;
* ordered raw-media hashes;
* exact normalized visual-patch hashes and ``grid_thw``;
* vision spans, token tags, attention mask and MRoPE position ids; and
* the complete pre-language-layer input embedding tensor.

The last condition is intentionally stricter than merely using the same image.
It proves the BF16 vision/embedding path and the W4-preserved BF16 path handed
the language decoder identical floating-point inputs.  A mismatch is a useful
result, but it is not isolated weight-quantization drift.

No quality threshold lives here.  The tool reports float64-accumulated metrics
for all, text, vision and marker rows.  Interpretation belongs to the dated
result record after a real capture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


SCHEMA = "h3-layer50-capture-v1"
ALIGNMENT_FIELDS = (
    "expanded_token_ids_sha256",
    "sequence_length",
    "ordered_media",
    "visual_patches",
    "grid_thw",
    "vision_spans",
    "token_tags_sha256",
    "attention_mask_sha256",
    "position_ids_sha256",
    "input_embeds_sha256",
    "text_positions",
    "vision_positions",
    "marker_positions",
)
PERCENTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
REQUIRED_TOP = {
    "schema_version",
    "arm",
    "comparison_kind",
    "processor_policy",
    "processor_policy_record",
    "output_tap",
    "fixture_population",
    "tokenizer_vocab_sha256",
    "model",
    "provenance",
    "fixtures",
}
REQUIRED_PROVENANCE = {
    "capture_script_sha256",
    "h3_awq_adapter_sha256",
    "implementation_files",
    "comfyui_commit",
    "torch",
    "cuda_runtime",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _finite(value: float | int) -> float | int | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"capture has no manifest.json: {directory}")
    data = json.loads(path.read_text())
    missing_top = sorted(REQUIRED_TOP - set(data))
    if missing_top:
        raise ValueError(f"{directory.name}: manifest missing {missing_top}")
    if data.get("schema_version") != SCHEMA:
        raise ValueError(
            f"{directory.name}: schema {data.get('schema_version')!r}, "
            f"expected {SCHEMA!r}"
        )
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError(f"{directory.name}: manifest has no fixture records")
    ids = [row.get("fixture_id") for row in fixtures]
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"{directory.name}: fixture ids must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{directory.name}: duplicate fixture ids")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{directory.name}: provenance must be an object")
    missing_provenance = sorted(REQUIRED_PROVENANCE - set(provenance))
    if missing_provenance:
        raise ValueError(
            f"{directory.name}: provenance missing {missing_provenance}"
        )
    configs = (data.get("model") or {}).get("shared_processor_configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"{directory.name}: shared processor configs are absent")
    for row in fixtures:
        _validate_fixture_record(directory.name, row)
    return data


def _validate_fixture_record(capture_name: str, row: dict) -> None:
    fixture_id = row.get("fixture_id")
    prefix = f"{capture_name}/{fixture_id}"
    length = row.get("sequence_length")
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"{prefix}: invalid sequence length {length!r}")

    ids = row.get("expanded_token_ids")
    if not isinstance(ids, list) or len(ids) != length:
        raise ValueError(f"{prefix}: expanded token ids do not match sequence length")
    if _json_sha(ids) != row.get("expanded_token_ids_sha256"):
        raise ValueError(f"{prefix}: expanded token id hash is not self-consistent")

    tags = row.get("token_tags")
    if not isinstance(tags, list) or len(tags) != length or set(tags) - {0, 1}:
        raise ValueError(f"{prefix}: invalid token tags")
    if _json_sha(tags) != row.get("token_tags_sha256"):
        raise ValueError(f"{prefix}: token tag hash is not self-consistent")

    attention = row.get("attention_mask")
    if not isinstance(attention, list) or len(attention) != length:
        raise ValueError(f"{prefix}: invalid attention mask")
    if _json_sha(attention) != row.get("attention_mask_sha256"):
        raise ValueError(f"{prefix}: attention hash is not self-consistent")

    expected_text = [index for index, tag in enumerate(tags) if tag == 1]
    expected_vision = [index for index, tag in enumerate(tags) if tag == 0]
    expected_markers = [
        index for index, token in enumerate(ids) if 151669 <= int(token) <= 151675
    ]
    if row.get("text_positions") != expected_text:
        raise ValueError(f"{prefix}: text positions do not derive from token tags")
    if row.get("vision_positions") != expected_vision:
        raise ValueError(f"{prefix}: vision positions do not derive from token tags")
    if row.get("marker_positions") != expected_markers:
        raise ValueError(f"{prefix}: marker positions do not derive from token ids")

    spans = []
    start = None
    for index, tag in enumerate(tags):
        if tag == 0 and start is None:
            start = index
        elif tag == 1 and start is not None:
            spans.append([start, index - 1])
            start = None
    if start is not None:
        spans.append([start, length - 1])
    if row.get("vision_spans") != spans:
        raise ValueError(f"{prefix}: vision spans do not derive from token tags")

    embeds = row.get("input_embeds") or {}
    if embeds.get("sha256") != row.get("input_embeds_sha256"):
        raise ValueError(f"{prefix}: input embedding hashes disagree")


def _fixture_map(manifest: dict) -> dict[str, dict]:
    return {row["fixture_id"]: row for row in manifest["fixtures"]}


def _alignment_errors(reference: dict, candidate: dict) -> list[str]:
    errors = []
    for field in ALIGNMENT_FIELDS:
        if reference.get(field) != candidate.get(field):
            errors.append(field)
    return errors


def _load_hidden(directory: Path, record: dict) -> torch.Tensor:
    rel = record.get("hidden_state_file")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"{record.get('fixture_id')}: unsafe hidden-state path {rel!r}")
    path = directory / rel
    if not path.is_file():
        raise ValueError(f"{record.get('fixture_id')}: missing {rel}")
    expected = record.get("hidden_state_sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{record.get('fixture_id')}: hidden-state hash {actual} != {expected}"
        )
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"hidden"}:
        raise ValueError(f"{record.get('fixture_id')}: expected only tensor 'hidden'")
    hidden = tensors["hidden"]
    declared = record.get("hidden_state") or {}
    if list(hidden.shape) != declared.get("shape"):
        raise ValueError(
            f"{record.get('fixture_id')}: tensor shape {list(hidden.shape)} != "
            f"{declared.get('shape')}"
        )
    if str(hidden.dtype).removeprefix("torch.") != declared.get("dtype"):
        raise ValueError(f"{record.get('fixture_id')}: tensor dtype changed")
    if hidden.ndim != 2:
        raise ValueError(f"{record.get('fixture_id')}: expected [tokens, hidden]")
    return hidden


def _summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"hidden-state shape mismatch {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        return {"rows": 0, "elements": 0, "available": False}

    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    delta = cand - ref
    ref_flat = ref.flatten()
    cand_flat = cand.flatten()
    delta_flat = delta.flatten()

    ref_norm = torch.linalg.vector_norm(ref_flat)
    cand_norm = torch.linalg.vector_norm(cand_flat)
    delta_norm = torch.linalg.vector_norm(delta_flat)
    denom = ref_norm * cand_norm
    cosine = (ref_flat @ cand_flat) / denom if denom else torch.tensor(float("nan"))
    mse = torch.mean(delta_flat.square())

    row_ref_norm = torch.linalg.vector_norm(ref, dim=-1)
    row_cand_norm = torch.linalg.vector_norm(cand, dim=-1)
    row_denom = row_ref_norm * row_cand_norm
    row_dot = torch.sum(ref * cand, dim=-1)
    row_cos = torch.where(
        row_denom > 0,
        row_dot / row_denom,
        torch.full_like(row_dot, float("nan")),
    )
    finite_cos = row_cos[torch.isfinite(row_cos)]
    q = {}
    if finite_cos.numel():
        vals = torch.quantile(finite_cos, torch.tensor(PERCENTILES, dtype=torch.float64))
        q = {f"p{int(p * 100):02d}": _finite(v.item()) for p, v in zip(PERCENTILES, vals)}

    return {
        "available": True,
        "rows": int(ref.shape[0]),
        "elements": int(ref.numel()),
        "flattened_cosine": _finite(cosine.item()),
        "mse": _finite(mse.item()),
        "rmse": _finite(torch.sqrt(mse).item()),
        "relative_l2": _finite((delta_norm / ref_norm).item()) if ref_norm else None,
        "reference_rms": _finite(torch.sqrt(torch.mean(ref_flat.square())).item()),
        "candidate_rms": _finite(torch.sqrt(torch.mean(cand_flat.square())).item()),
        "max_abs_error": _finite(torch.max(torch.abs(delta_flat)).item()),
        "tokenwise_cosine": {
            "finite_rows": int(finite_cos.numel()),
            "minimum": _finite(torch.min(finite_cos).item()) if finite_cos.numel() else None,
            "mean": _finite(torch.mean(finite_cos).item()) if finite_cos.numel() else None,
            **q,
            "maximum": _finite(torch.max(finite_cos).item()) if finite_cos.numel() else None,
        },
    }


def _positions(record: dict, kind: str) -> list[int]:
    length = int(record["sequence_length"])
    if kind == "all":
        return list(range(length))
    if kind == "text":
        return [int(x) for x in record.get("text_positions", [])]
    if kind == "vision":
        return [int(x) for x in record.get("vision_positions", [])]
    if kind == "markers":
        return [int(x) for x in record.get("marker_positions", [])]
    raise AssertionError(kind)


def _selected(tensor: torch.Tensor, positions: Iterable[int]) -> torch.Tensor:
    positions = list(positions)
    if not positions:
        return tensor.new_empty((0, tensor.shape[-1]))
    if min(positions) < 0 or max(positions) >= tensor.shape[0]:
        raise ValueError("recorded row position is outside the hidden-state tensor")
    return tensor[positions]


def _compare_fixture(
    reference_dir: Path,
    candidate_dir: Path,
    reference_record: dict,
    candidate_record: dict,
) -> tuple[dict, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    errors = _alignment_errors(reference_record, candidate_record)
    if errors:
        fid = reference_record.get("fixture_id")
        raise ValueError(
            f"{fid}: refusing direct metrics; arms differ in " + ", ".join(errors)
        )
    ref = _load_hidden(reference_dir, reference_record)
    cand = _load_hidden(candidate_dir, candidate_record)
    if ref.shape[0] != reference_record["sequence_length"]:
        raise ValueError(
            f"{reference_record['fixture_id']}: hidden rows do not equal sequence length"
        )
    metrics = {}
    selected = {}
    for kind in ("all", "text", "vision", "markers"):
        pos = _positions(reference_record, kind)
        a, b = _selected(ref, pos), _selected(cand, pos)
        metrics[kind] = _summary(a, b)
        selected[kind] = (a, b)
    return metrics, selected


def compare(reference_dir: Path, candidate_dir: Path) -> dict:
    reference = _load_manifest(reference_dir)
    candidate = _load_manifest(candidate_dir)

    if reference.get("comparison_kind") != candidate.get("comparison_kind"):
        raise ValueError("arms declare different comparison kinds")
    if reference.get("comparison_kind") != "weight_only":
        raise ValueError(
            "direct rowwise comparison is implemented only for weight_only captures; "
            "deployed-path captures need semantic alignment when presentation differs"
        )
    if reference.get("processor_policy") != candidate.get("processor_policy"):
        raise ValueError("weight-only arms declare different processor policies")
    if reference.get("arm") != "bf16":
        raise ValueError("--reference must be the bf16 arm")
    if candidate.get("arm") != "w4":
        raise ValueError("--candidate must be the w4 arm")

    top_level_equal = (
        "output_tap",
        "fixture_population",
        "tokenizer_vocab_sha256",
        "processor_policy_record",
    )
    for field in top_level_equal:
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"capture arms disagree in top-level field {field}")
    provenance_equal = (
        "capture_script_sha256",
        "h3_awq_adapter_sha256",
        "implementation_files",
        "comfyui_commit",
        "torch",
        "cuda_runtime",
        "extra_reserved_vram_bytes",
    )
    for field in provenance_equal:
        if (reference.get("provenance") or {}).get(field) != (
            candidate.get("provenance") or {}
        ).get(field):
            raise ValueError(f"capture arms used different provenance field {field}")
    ref_configs = (reference.get("model") or {}).get("shared_processor_configs")
    cand_configs = (candidate.get("model") or {}).get("shared_processor_configs")
    if ref_configs != cand_configs:
        raise ValueError("capture arms used different shared processor configs")

    ref_rows = _fixture_map(reference)
    cand_rows = _fixture_map(candidate)
    if set(ref_rows) != set(cand_rows):
        missing = sorted(set(ref_rows) - set(cand_rows))
        extra = sorted(set(cand_rows) - set(ref_rows))
        raise ValueError(f"fixture populations differ; missing={missing}, extra={extra}")

    fixtures = []
    aggregate = {kind: [[], []] for kind in ("all", "text", "vision", "markers")}
    for fixture_id in sorted(ref_rows):
        metrics, selected = _compare_fixture(
            reference_dir, candidate_dir, ref_rows[fixture_id], cand_rows[fixture_id]
        )
        source = ref_rows[fixture_id]
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "family": source.get("family"),
                "prompt_sha256": source.get("prompt_sha256"),
                "source_files": source.get("source_files", []),
                "sequence_length": source.get("sequence_length"),
                "text_rows": len(source.get("text_positions", [])),
                "vision_rows": len(source.get("vision_positions", [])),
                "marker_rows": len(source.get("marker_positions", [])),
                "expanded_token_ids_sha256": source.get(
                    "expanded_token_ids_sha256"
                ),
                "ordered_media": source.get("ordered_media"),
                "visual_patches": source.get("visual_patches"),
                "grid_thw": source.get("grid_thw"),
                "vision_spans": source.get("vision_spans"),
                "input_embeds_sha256": source.get("input_embeds_sha256"),
                "metrics": metrics,
            }
        )
        for kind, (ref_tensor, cand_tensor) in selected.items():
            aggregate[kind][0].append(ref_tensor)
            aggregate[kind][1].append(cand_tensor)

    aggregate_metrics = {}
    hidden = next(iter(ref_rows.values()))["hidden_state"]["shape"][-1]
    for kind, (refs, cands) in aggregate.items():
        ref = torch.cat(refs, dim=0) if refs else torch.empty((0, hidden))
        cand = torch.cat(cands, dim=0) if cands else torch.empty((0, hidden))
        aggregate_metrics[kind] = _summary(ref, cand)

    return {
        "schema_version": "h3-layer50-comparison-v1",
        "comparison_kind": "weight_only",
        "processor_policy": reference["processor_policy"],
        "processor_policy_record": reference["processor_policy_record"],
        "accumulation_dtype": "float64",
        "comparison_provenance": {
            "compared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "comparison_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "reference_capture": {
            "directory": reference_dir.name,
            "manifest_sha256": _sha256(reference_dir / "manifest.json"),
            "model": reference.get("model"),
            "provenance": reference.get("provenance"),
        },
        "candidate_capture": {
            "directory": candidate_dir.name,
            "manifest_sha256": _sha256(candidate_dir / "manifest.json"),
            "model": candidate.get("model"),
            "provenance": candidate.get("provenance"),
        },
        "alignment_fields_required": list(ALIGNMENT_FIELDS),
        "aggregate": aggregate_metrics,
        "fixtures": fixtures,
    }


def _self_record(fixture_id: str, hidden_file: str, hidden_sha: str) -> dict:
    ids = [10, 151669, 20, 151670]
    tags = [1, 1, 0, 1]
    return {
        "fixture_id": fixture_id,
        "expanded_token_ids_sha256": _json_sha(ids),
        "expanded_token_ids": ids,
        "sequence_length": 4,
        "ordered_media": [{"type": "image", "sha256": "a" * 64}],
        "visual_patches": [{"sha256": "b" * 64, "shape": [4, 6], "dtype": "float32"}],
        "grid_thw": [[1, 4, 4]],
        "vision_spans": [[2, 2]],
        "token_tags_sha256": _json_sha(tags),
        "token_tags": tags,
        "attention_mask": [1, 1, 1, 1],
        "attention_mask_sha256": _json_sha([1, 1, 1, 1]),
        "position_ids_sha256": "c" * 64,
        "input_embeds_sha256": "d" * 64,
        "input_embeds": {"sha256": "d" * 64, "shape": [1, 4, 3], "dtype": "float32"},
        "text_positions": [0, 1, 3],
        "vision_positions": [2],
        "marker_positions": [1, 3],
        "hidden_state_file": hidden_file,
        "hidden_state_sha256": hidden_sha,
        "hidden_state": {"shape": [4, 3], "dtype": "float32"},
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="h3-layer50-compare-") as raw:
        root = Path(raw)
        ref_dir, cand_dir = root / "bf16", root / "w4"
        ref_dir.mkdir()
        cand_dir.mkdir()
        ref = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7
        cand = ref.clone()
        cand[2, 1] += 0.25
        for directory, tensor in ((ref_dir, ref), (cand_dir, cand)):
            path = directory / "fixture.safetensors"
            save_file({"hidden": tensor}, str(path))
            rec = _self_record("fixture", path.name, _sha256(path))
            manifest = {
                "schema_version": SCHEMA,
                "arm": "bf16" if directory == ref_dir else "w4",
                "comparison_kind": "weight_only",
                "processor_policy": "shared_current_w4",
                "processor_policy_record": {
                    "name": "shared_current_w4",
                    "scope": "self-test",
                    "changed_from_current_w4": [],
                    "effective_still_bounds": {
                        "shortest_edge": 200704,
                        "longest_edge": 301056,
                    },
                    "effective_image_processor_config": {"test": True},
                    "effective_image_processor_config_sha256": "6" * 64,
                    "effective_video_policy": "self-test",
                    "source_files": [
                        {
                            "role": "self-test",
                            "path": "test",
                            "sha256": "1" * 64,
                        }
                    ],
                },
                "output_tap": "self-test",
                "fixture_population": "self-test",
                "tokenizer_vocab_sha256": "0" * 64,
                "model": {
                    "test": True,
                    "shared_processor_configs": [{"name": "test", "sha256": "1" * 64}],
                },
                "provenance": {
                    "capture_script_sha256": "2" * 64,
                    "h3_awq_adapter_sha256": "3" * 64,
                    "implementation_files": [{"path": "test", "sha256": "4" * 64}],
                    "comfyui_commit": "5" * 40,
                    "torch": torch.__version__,
                    "cuda_runtime": None,
                },
                "fixtures": [rec],
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )

        result = compare(ref_dir, cand_dir)
        assert result["aggregate"]["all"]["relative_l2"] > 0
        assert result["aggregate"]["vision"]["relative_l2"] > 0
        assert result["aggregate"]["markers"]["relative_l2"] == 0

        # Mutation control 1: a changed patch input must invalidate the join.
        candidate_manifest_path = cand_dir / "manifest.json"
        original = json.loads(candidate_manifest_path.read_text())
        mutated = json.loads(candidate_manifest_path.read_text())
        mutated["fixtures"][0]["visual_patches"][0]["sha256"] = "e" * 64
        candidate_manifest_path.write_text(json.dumps(mutated, indent=2) + "\n")
        try:
            compare(ref_dir, cand_dir)
        except ValueError as exc:
            assert "visual_patches" in str(exc)
        else:
            raise AssertionError("changed visual patches did not invalidate comparison")

        # Mutation control 2: a changed tag mask must also invalidate the join.
        mutated = json.loads(json.dumps(original))
        mutated["fixtures"][0]["token_tags_sha256"] = "f" * 64
        candidate_manifest_path.write_text(json.dumps(mutated, indent=2) + "\n")
        try:
            compare(ref_dir, cand_dir)
        except ValueError as exc:
            assert "token tag hash" in str(exc)
        else:
            raise AssertionError("changed token tags did not invalidate comparison")

    print("ok: comparator detects a real delta and refuses patch/tag misalignment")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="BF16 capture directory")
    parser.add_argument("--candidate", help="W4 capture directory")
    parser.add_argument("--out", help="comparison JSON path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (args.reference and args.candidate and args.out):
        parser.error("--reference, --candidate and --out are required")
    result = compare(Path(args.reference).resolve(), Path(args.candidate).resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
