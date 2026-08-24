#!/usr/bin/env python3
"""Grade the calibration bundle against an independent arm, and make it fail.

Gate 1, presentation half. `build_native_h3_calibration_batch.py` drives the
installed native-H3 path; this asks two questions about what it produced.

**1. Does a second implementation agree?** The rejected preflight's central
defect was six "parity" fixtures of which five compared the new builder to
values the same builder produced. So there are two reference arms here, both
built from the bundle's own upstream-sized media and the row's raw prompt,
through code the builder never calls. The installed native path stays the
authority for what H3 actually receives; a disagreement is a finding on
whichever side is wrong and says nothing by itself about which.

- **Segmented ids** follow the vendor's own algorithm in
  `coderef/sglang/.../minimax_h3/presentation.py`, which tokenizes each label
  segment separately. This arm covers every family, reference video included.
- **`Qwen3VLProcessor`** assembles the whole batch end to end, so it also
  produces `pixel_values`, `image_grid_thw` and `mm_token_type_ids`
  independently. It has two structural limits, reported per row rather than
  quietly skipped: it cannot emit H3's two-frame block presentation, and its
  single-shot tokenization crosses a BPE merge boundary that segmented
  tokenization does not, so on a row whose last ordered item is an audio label
  its sequence is one token shorter by construction. Video blocks are instead
  compared patch-for-patch against `comfy/text_encoders/minimax.py::
  process_video_block` in the bundle's own record.

**2. Can each control fail?** Nine deliberate defects from
`build_native_h3_calibration_batch.py::MUTATIONS` are built for real and each
must change a *named* field of the presentation record -- not merely some
field. Two of them (`timestamp-shift`, `mm-types-zero`) leave every shape and
count identical, which is exactly why the record hashes rather than counts.
Rows a mutation cannot reach must come back byte-identical to clean; a control
that changes everything proves nothing.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`); it needs a GPU and
a few minutes, and writes its report to `bench/results/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
COMFY = REPO.parents[1]
BUILDER = BENCH / "build_native_h3_calibration_batch.py"
REPORT = BENCH / "results" / "2026-08-24_native_h3_presentation_parity.json"

# Which row exercises which defect, and which record field must move. A
# mutation graded on "something changed" would pass on an unrelated change.
MUTATION_PLAN = {
    "chat-framing": ("single-image", "presentation_scaffold_or_length"),
    "first-image-only": ("multi-image-2-3", "vision_block_count"),
    "reorder-references": ("multi-image-2-3", "vision_block_order"),
    "timestamp-shift": ("video-reference", "presentation_scaffold"),
    "drop-temporal-repeat": ("video-reference", "vision_block_count"),
    "grid-shrink": ("single-image", "vision_block_grids"),
    "drop-media": ("single-image", "vision_block_count"),
    "token-tags-flip": ("single-image", "token_tags_sha256"),
    "mm-types-zero": ("single-image", "mm_token_type_ids_sha256"),
}

# Fields that a mutation must leave alone on a row it cannot reach.
INVARIANT_FIELDS = (
    "prompt_sha256",
    "expanded_token_ids_sha256",
    "token_tags_sha256",
    "attention_mask_sha256",
    "mm_token_type_ids_sha256",
    "position_ids_sha256",
    "sequence_length",
)


def _run_builder(out: Path, row_id: str | None = None, mutation: str | None = None,
                 families: list[str] | None = None) -> Path:
    command = [sys.executable, str(BUILDER), "--out", str(out)]
    if row_id:
        command += ["--row", row_id]
    for family in families or []:
        command += ["--family", family]
    if mutation:
        command += ["--mutate", mutation]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"builder failed for mutation={mutation!r} row={row_id!r}:\n{result.stderr[-2000:]}"
        )
    return out / "presentation.json"


def _load(bundle: Path) -> dict:
    return json.loads((bundle / "presentation.json").read_text())


def _row(manifest: dict, row_id: str) -> dict:
    for record in manifest["rows"]:
        if record["row_id"] == row_id:
            return record
    raise KeyError(row_id)


def _block_signature(record: dict) -> dict:
    return {
        "count": len(record["vision_blocks"]),
        "grids": [b["grid_thw"] for b in record["vision_blocks"]],
        "patch_hashes": [b["patches"]["sha256"] for b in record["vision_blocks"]],
    }


# --------------------------------------------------------------------------
# released special-token ids


def check_release_special_tokens() -> dict:
    """The installed tokenizer's marker ids, against ids derived from the release.

    The release does not state these seven ids anywhere directly: its
    `added_tokens_decoder` stops at the last stock Qwen entry, and the seven
    appear only in `additional_special_tokens`, whose *order* then decides the
    id each one receives when it is appended past that entry. So the expected
    ids are derived from the two vendored declarations rather than compared
    against `MINIMAX_EXTRA_TOKENS`, which is the installed constant and would
    be checking the installed path against itself.

    Deriving them this way also makes the check sensitive to a reordering of
    that list, which would silently reassign every marker.
    """
    sys.path.insert(0, str(COMFY))
    import nodes  # noqa: F401
    from comfy.text_encoders.minimax import MINIMAX_EXTRA_TOKENS, MiniMaxH3Tokenizer

    vendor_config = json.loads(
        (REPO / "vendor_config" / "tokenizer_config.json").read_text()
    )
    declared = vendor_config["additional_special_tokens"]
    stock = {int(k) for k in vendor_config["added_tokens_decoder"]}
    appended = [token for token in declared if token not in
                {v["content"] for v in vendor_config["added_tokens_decoder"].values()}]
    expected = {token: max(stock) + 1 + index for index, token in enumerate(appended)}

    tokenizer = MiniMaxH3Tokenizer()
    vocab = tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    resolved = {token: vocab.get(token) for token in appended}

    failures = []
    for token, want in expected.items():
        if resolved.get(token) != want:
            failures.append(
                f"{token}: installed id {resolved.get(token)} != release-derived {want}"
            )
    if set(appended) != set(MINIMAX_EXTRA_TOKENS):
        failures.append(
            "the installed tokenizer's extra tokens are not the release's "
            f"appended set: installed={sorted(MINIMAX_EXTRA_TOKENS)}, "
            f"release={sorted(appended)}"
        )
    return {
        "release_declared_special_tokens": len(declared),
        "release_stock_added_tokens": len(stock),
        "release_appended_tokens": appended,
        "release_derived_ids": expected,
        "installed_ids": resolved,
        "failures": failures,
    }


# --------------------------------------------------------------------------
# independent reference arm


def _independent_processor():
    from transformers import AutoTokenizer, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )
    from comfy.text_encoders.minimax import MINIMAX_EXTRA_TOKENS

    tokenizer = AutoTokenizer.from_pretrained(
        str(COMFY / "comfy" / "text_encoders" / "qwen25_tokenizer")
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)}
    )
    image_declared = json.loads(
        (REPO / "vendor_config" / "preprocessor_config.json").read_text()
    )
    video_declared = json.loads(
        (REPO / "vendor_config" / "video_preprocessor_config.json").read_text()
    )
    image_processor = Qwen2VLImageProcessor(**{
        k: v for k, v in image_declared.items()
        if k not in ("processor_class", "image_processor_type")
    })
    video_processor = Qwen3VLVideoProcessor(**{
        k: v for k, v in video_declared.items()
        if k not in ("processor_class", "video_processor_type")
    })
    return Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer,
        video_processor=video_processor,
    )


def _segmented_ids(tokenizer, record: dict, prompt: str) -> list[int]:
    """Assemble ids the way the vendor's own serving stack assembles them.

    `coderef/sglang/.../minimax_h3/presentation.py` tokenizes each label
    segment on its own -- `_text_ids(tokenizer, f"<Picture {i}>: ")`, then the
    explicit vision block, then `_text_ids(tokenizer, prompt)` -- and never
    concatenates before tokenizing. That is not cosmetic. A label ending in a
    space, immediately followed by prompt text, merges into a single `Ġword`
    token when the two are tokenized together and stays two tokens when they
    are not. In H3 presentation the case arises exactly once: an `<Audio j>: `
    label as the last ordered item, because every other label is followed by a
    special token, which breaks the merge anyway.

    So a builder that concatenates first emits a sequence one token shorter
    than both the installed ComfyUI path and the vendor's. This arm reproduces
    the vendor's algorithm to have something real to disagree with.
    """
    ids: list[int] = []
    for item in record["ordered_media"]:
        ids += tokenizer(f"{item['label']}: ", add_special_tokens=False)["input_ids"]
        if item["type"] == "audio":
            continue
        blocks = [
            block for block in record["vision_blocks"]
        ] if item["type"] == "video" else None
        counts = (
            [b["merged_tokens"] for b in blocks]
            if blocks is not None
            else [_merged_for(record, item)]
        )
        stamps = item.get("timestamps")
        if item["type"] == "video":
            # The tokenizer repeat-pads an odd sampled count, so the block
            # count is the record's, not len(timestamps).
            padded = list(stamps) + ([stamps[-1]] if len(stamps) % 2 else [])
            for index, count in enumerate(counts):
                middle = (padded[2 * index] + padded[2 * index + 1]) / 2.0
                ids += tokenizer(f"<{middle:.1f} seconds>",
                                 add_special_tokens=False)["input_ids"]
                ids += _block_ids(tokenizer, count)
        else:
            ids += _block_ids(tokenizer, counts[0])
    ids += tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return ids


def _block_ids(tokenizer, count: int) -> list[int]:
    return (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * int(count)
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )


def _merged_for(record: dict, item: dict) -> int:
    index = record["ordered_media"].index(item)
    visual = [i for i in record["ordered_media"][: index + 1] if i["type"] != "audio"]
    return record["vision_blocks"][len(visual) - 1]["merged_tokens"]


def independent_arm(bundle: Path, record: dict, prompt: str) -> dict:
    """Reassemble the batch outside the builder, two ways, and compare.

    Neither sub-arm calls anything the builder calls beyond the release
    processor configuration and the pixels themselves.

    - `segmented_ids` follows the vendor's own presentation algorithm and
      covers every family, reference video included.
    - `processor` runs `Qwen3VLProcessor` end to end on a single assembled
      text. It is the stronger check because it independently produces
      `pixel_values`, `image_grid_thw` and `mm_token_type_ids` as well, but it
      cannot express H3's two-frame video presentation, and on a row whose last
      ordered item is an audio label its single-shot tokenization crosses the
      merge boundary above. Both limits are reported per row rather than
      silently excluded.
    """
    from safetensors.torch import load_file

    processor = _independent_processor()
    tokenizer = processor.tokenizer
    builder = load_file(bundle / record["batch_file"])
    media = load_file(bundle / record["media_file"]) if record.get("media_file") else {}
    result: dict = {}

    theirs = torch.tensor([_segmented_ids(tokenizer, record, prompt)], dtype=torch.long)
    mine = builder["input_ids"]
    result["segmented_ids"] = {
        "equal": tuple(mine.shape) == tuple(theirs.shape) and bool(torch.equal(mine, theirs)),
        "builder_length": int(mine.shape[1]),
        "arm_length": int(theirs.shape[1]),
        "mismatches": (
            int((mine != theirs).sum()) if tuple(mine.shape) == tuple(theirs.shape) else None
        ),
    }

    has_video = any(item["type"] == "video" for item in record["ordered_media"])
    trailing_audio = bool(record["ordered_media"]) and record["ordered_media"][-1]["type"] == "audio"
    if has_video:
        result["processor"] = {
            "covered": False,
            "reason": "Qwen3VLProcessor cannot emit H3's two-frame block "
                      "presentation; those blocks are compared patch-for-patch "
                      "against comfy process_video_block in the bundle record",
        }
        return result

    pieces, images = [], []
    for item in record["ordered_media"]:
        if item["type"] == "audio":
            pieces.append(f"{item['label']}: ")
            continue
        pieces.append(f"{item['label']}: <|vision_start|><|image_pad|><|vision_end|>")
        images.append(media[item["upstream_media_key"]][0].permute(2, 0, 1))
    text = "".join(pieces) + prompt
    batch = processor(text=[text], images=images or None, return_tensors="pt")

    compared = ("pixel_values", "image_grid_thw") if trailing_audio else (
        "input_ids", "attention_mask", "mm_token_type_ids",
        "pixel_values", "image_grid_thw",
    )
    arm: dict = {
        "covered": True,
        "text_bytes": len(text.encode("utf-8")),
        "sequence_fields_excluded": trailing_audio,
        "exclusion_reason": (
            "row ends on an audio label, so this arm's single-shot "
            "tokenization merges the label's trailing space into the prompt "
            "and is one token shorter by construction"
        ) if trailing_audio else None,
    }
    for key in compared:
        left, right = builder.get(key), batch.get(key)
        if left is None or right is None:
            arm[key] = f"present in only one arm (builder={left is not None})"
            continue
        if tuple(left.shape) != tuple(right.shape):
            arm[key] = f"shape {tuple(left.shape)} != {tuple(right.shape)}"
            continue
        if left.is_floating_point():
            delta = float((left.float() - right.float()).abs().max())
            arm[key] = {"equal": delta == 0.0, "max_abs_delta": delta}
        else:
            equal = bool(torch.equal(left.to(right.dtype), right))
            arm[key] = {"equal": equal,
                        "mismatches": int((left.to(right.dtype) != right).sum())}
    result["processor"] = arm
    return result


SKIP_KEYS = {"covered", "text_bytes", "reason", "sequence_fields_excluded",
             "exclusion_reason", "builder_length", "arm_length", "mismatches"}


def _arm_failures(result: dict) -> list[str]:
    failures = []
    segmented = result.get("segmented_ids", {})
    if not segmented.get("equal"):
        failures.append(f"segmented_ids: {segmented}")
    arm = result.get("processor", {})
    if not arm.get("covered"):
        return failures
    for key, value in arm.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(value, str):
            failures.append(f"processor {key}: {value}")
        elif isinstance(value, dict) and not value.get("equal"):
            failures.append(f"processor {key}: {value}")
    return failures


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True,
                        help="scratch directory for the bundles this builds")
    parser.add_argument("--keep", action="store_true",
                        help="keep the built bundles instead of deleting them")
    args = parser.parse_args()

    work = Path(args.work).expanduser().resolve()
    if work.exists():
        raise SystemExit(f"refuse to reuse an existing work directory: {work}")
    work.mkdir(parents=True)

    from build_native_h3_calibration_batch import MUTATIONS  # noqa: E402

    failures: list[str] = []
    report: dict = {
        "check": "native-H3 calibration presentation parity and mutation controls",
        "clean_bundle_rows": [],
        "independent_arm": {},
        "mutations": {},
    }

    try:
        print("building the clean bundle")
        clean_dir = work / "clean"
        _run_builder(clean_dir)
        clean = _load(clean_dir)
        report["provenance"] = clean["provenance"]

        tokens = check_release_special_tokens()
        report["release_special_tokens"] = tokens
        failures.extend(tokens["failures"])

        root, _ = _dataset_root()
        prompts = {}
        for line in (root / "data" / "train.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["id"] in clean["order"]:
                prompts[row["id"]] = row.get("target_ir") or ""

        print("independent Qwen3VLProcessor arm")
        for record in clean["rows"]:
            report["clean_bundle_rows"].append({
                "row_id": record["row_id"],
                "primary_role": record["primary_role"],
                "sequence_length": record["sequence_length"],
                "vision_positions": record["vision_positions"],
                "blocks": _block_signature(record),
            })
            arm = independent_arm(clean_dir, record, prompts[record["row_id"]])
            report["independent_arm"][record["row_id"]] = arm
            for message in _arm_failures(arm):
                failures.append(f"independent arm, {record['row_id']}: {message}")
            processor_arm = arm.get("processor", {})
            state = "processor arm covered" if processor_arm.get("covered") else (
                "processor arm not applicable")
            if processor_arm.get("sequence_fields_excluded"):
                state += ", sequence fields excluded"
            print(f"  {record['row_id']:<34} segmented ids "
                  f"{'match' if arm['segmented_ids']['equal'] else 'DIFFER'}; {state}")

        for name, (family, field) in MUTATION_PLAN.items():
            row_id = next(
                r["row_id"] for r in clean["rows"] if r["primary_role"] == family
            )
            print(f"mutation {name} on {row_id}")
            mutated_dir = work / f"mut-{name}"
            _run_builder(mutated_dir, row_id=row_id, mutation=name)
            mutated = _row(_load(mutated_dir), row_id)
            base = _row(clean, row_id)
            detected = _detect(base, mutated, field)
            report["mutations"][name] = {
                "intent": MUTATIONS[name],
                "row_id": row_id,
                "graded_field": field,
                **detected,
            }
            if not detected["changed"]:
                failures.append(
                    f"mutation {name} did not change {field} on {row_id}; "
                    f"the control cannot fail"
                )
            print(f"  {field}: {'changed' if detected['changed'] else 'UNCHANGED'} "
                  f"({detected['detail']})")

            unreachable = _unreachable_rows(name, clean)
            if unreachable:
                other_dir = work / f"mut-{name}-unreached"
                _run_builder(other_dir, row_id=unreachable, mutation=name)
                other = _row(_load(other_dir), unreachable)
                base_other = _row(clean, unreachable)
                drifted = [
                    key for key in INVARIANT_FIELDS
                    if base_other.get(key) != other.get(key)
                ]
                report["mutations"][name]["unreachable_row"] = unreachable
                report["mutations"][name]["unreachable_row_drift"] = drifted
                if drifted:
                    failures.append(
                        f"mutation {name} also changed {drifted} on {unreachable}, "
                        f"a row it cannot reach; the control is not specific"
                    )
                print(f"  unreached row {unreachable}: "
                      f"{'clean' if not drifted else 'DRIFTED ' + str(drifted)}")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    report["failures"] = failures
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{REPORT.name}")

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: the independent arm agrees on every row it covers, and every "
          "mutation moves its own field without disturbing a row it cannot reach")
    return 0


def _detect(base: dict, mutated: dict, field: str) -> dict:
    base_blocks, mutated_blocks = _block_signature(base), _block_signature(mutated)
    if field == "vision_block_count":
        changed = base_blocks["count"] != mutated_blocks["count"]
        detail = f"{base_blocks['count']} -> {mutated_blocks['count']}"
    elif field == "vision_block_order":
        changed = (base_blocks["patch_hashes"] != mutated_blocks["patch_hashes"]
                   and sorted(base_blocks["patch_hashes"]) == sorted(mutated_blocks["patch_hashes"]))
        detail = f"grids {base_blocks['grids']} -> {mutated_blocks['grids']}"
    elif field == "vision_block_grids":
        changed = base_blocks["grids"] != mutated_blocks["grids"]
        detail = f"{base_blocks['grids']} -> {mutated_blocks['grids']}"
    elif field == "presentation_scaffold":
        changed = base["presentation_scaffold"] != mutated["presentation_scaffold"]
        detail = f"{base['presentation_scaffold'][:60]!r} -> {mutated['presentation_scaffold'][:60]!r}"
    elif field == "presentation_scaffold_or_length":
        changed = (base["sequence_length"] != mutated["sequence_length"]
                   or base["expanded_token_ids_sha256"] != mutated["expanded_token_ids_sha256"])
        detail = f"length {base['sequence_length']} -> {mutated['sequence_length']}"
    else:
        changed = base.get(field) != mutated.get(field)
        detail = f"{base.get(field)} -> {mutated.get(field)}"
    return {"changed": bool(changed), "detail": detail}


def _unreachable_rows(mutation: str, clean: dict) -> str | None:
    """A row the mutation structurally cannot touch, or None if every row is hit.

    `grid-shrink` moves the still processor, so a video-only row must be
    untouched. `timestamp-shift` and `drop-temporal-repeat` are video-only, so a
    still row must be untouched. The rest reach every row by construction.
    """
    by_role = {r["primary_role"]: r["row_id"] for r in clean["rows"]}
    if mutation == "grid-shrink":
        return by_role.get("video-reference")
    if mutation in ("timestamp-shift", "drop-temporal-repeat"):
        return by_role.get("single-image")
    return None


def _dataset_root():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_h3_pool_builder_check", BENCH / "build_h3_calibration_pool.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load build_h3_calibration_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pinned_snapshot()


if __name__ == "__main__":
    sys.path.insert(0, str(BENCH))
    sys.exit(main())
