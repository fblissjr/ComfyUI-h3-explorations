#!/usr/bin/env python3
"""Sequence-length cost of each candidate still-image policy, on real source media.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no model weights.

Decision 2 of the AWQ v2 lane picks the still-image preprocessing policy the
candidate will own. The choice is usually argued on fidelity. It also sets how
many visual tokens each calibration row carries, and therefore how much
activation cache `oneshot` must hold, so it is a feasibility decision too.

This measures that cost on the accepted, rights-clean H3-IR candidate pool
rather than on representative fixtures. It calls the two real implementations -- it does not
restate their resize rules:

  native      comfy.text_encoders.qwen_vl.process_qwen2vl_images, with the
              arguments comfy/text_encoders/qwen3vl.py::Qwen3VL.preprocess_embed
              passes (patch_size 16, mean/std 0.5, default 3,136..12,845,056)
  constrained h3_awq_encoder._source_image_patches, i.e. the current W4
              artifact's snapshotted Qwen2VLImageProcessor (200,704..301,056)

Geometry depends only on height and width, so each distinct source dimension is
evaluated once on a zeros tensor of that size and mapped back to every row that
uses it. Pixel content cannot change a grid.

Row totals combine measured visual tokens with prompt and "<Picture i>: " label
tokens from the installed MiniMaxH3Tokenizer. That row total is a construction,
not a capture of a launcher's sequences -- it assembles the native presentation
described in comfy/text_encoders/minimax.py rather than observing it. Treat the
visual counts as measured and the row totals as an estimate whose assembly is
stated here. Video rows are excluded: their geometry is governed by the video
policy and the reference node, not by this decision.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
POOL = REPO / "bench/results/archive/v2_encoder/2026-08-24_h3_calibration_pool.jsonl"
CACHE = Path.home() / ".cache/huggingface/hub/datasets--StellarVoyager--H3-IR"
REPORT = REPO / "bench/results/archive/v2_encoder/2026-08-24_still_policy_token_cost.json"

HIDDEN_SIZE = 5120
CACHE_DTYPE_BYTES = 2  # bfloat16 activations in IntermediatesCache

sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO))

import comfy.options  # noqa: E402

# Standalone Comfy imports ignore CLI flags unless parsing is explicitly
# enabled. This probe is CPU-only, so `--cpu` must reach model_management.
comfy.options.enable_args_parsing()

import torch  # noqa: E402

import comfy.text_encoders.qwen_vl as qwen_vl  # noqa: E402
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer  # noqa: E402
import h3_awq_encoder  # noqa: E402


def merged_tokens(grid) -> int:
    """Merged visual tokens for one block, from its grid_thw."""
    g = grid[0] if grid.dim() == 2 else grid
    return int(g[1]) // 2 * (int(g[2]) // 2)


def measure_dimension(width: int, height: int) -> dict:
    """Run both real still-image paths on one source dimension."""
    canvas = torch.zeros(1, height, width, 3, dtype=torch.float32)

    _, native_grid = qwen_vl.process_qwen2vl_images(
        canvas, patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]
    )
    _, constrained_grid = h3_awq_encoder._source_image_patches(
        canvas, torch.device("cpu")
    )
    return {
        "width": width,
        "height": height,
        "source_pixels": width * height,
        "native_grid_thw": native_grid.tolist(),
        "native_merged_tokens": merged_tokens(native_grid),
        "constrained_grid_thw": constrained_grid.tolist(),
        "constrained_merged_tokens": merged_tokens(constrained_grid),
    }


def text_token_count(tokenizer, text: str) -> int:
    """Length of one text segment under the installed native tokenizer."""
    batches = tokenizer.qwen3vl_32b.tokenize_with_weights(
        text, return_word_ids=False, disable_weights=True
    )
    return sum(len(b) for b in batches)


def describe(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "mean": round(statistics.mean(ordered), 1),
        "p90": pct(90),
        "p99": pct(99),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def main() -> int:
    rows = [json.loads(line) for line in POOL.read_text().splitlines()]
    revisions = {r["source_revision"] for r in rows}
    if len(revisions) != 1:
        raise ValueError(f"candidate pool spans revisions: {sorted(revisions)}")
    revision = next(iter(revisions))
    train = CACHE / "snapshots" / revision / "data" / "train.jsonl"
    source_rows = [json.loads(line) for line in train.read_text().splitlines()]
    image_rows = [r for r in rows if r.get("image_count", 0) > 0]

    dimensions = Counter()
    for row in image_rows:
        if len(row["image_dimensions"]) != row["image_count"]:
            raise ValueError(f"incomplete image dimensions for {row['id']}")
        for dims in row["image_dimensions"]:
            dimensions[tuple(dims)] += 1

    print(f"{len(image_rows)} image-bearing rows, {len(dimensions)} distinct dimensions")
    print("measuring both real implementations per distinct dimension ...")
    measured = {}
    for i, (width, height) in enumerate(sorted(dimensions), 1):
        measured[(width, height)] = measure_dimension(width, height)
        if i % 50 == 0:
            print(f"  {i}/{len(dimensions)}")

    tokenizer = MiniMaxH3Tokenizer()
    label_tokens = {n: text_token_count(tokenizer, f"<Picture {n}>: ")
                    for n in range(1, 33)}

    per_image = {"native": [], "constrained": []}
    per_row = {"native": [], "constrained": []}
    row_records = []
    for row in image_rows:
        source = source_rows[row["source_index"]]
        if source["id"] != row["id"]:
            raise ValueError(f"source index mismatch for {row['id']}")
        raw_prompt = source.get("target_ir") or ""
        prompt_tokens = text_token_count(tokenizer, raw_prompt)
        totals = {"native": prompt_tokens, "constrained": prompt_tokens}
        images = row["image_dimensions"]
        for ordinal, dims in enumerate(images, 1):
            entry = measured[tuple(dims)]
            for policy in ("native", "constrained"):
                tokens = entry[f"{policy}_merged_tokens"]
                per_image[policy].append(tokens)
                # vision block = start token + merged pads + end token, after its label
                totals[policy] += label_tokens[ordinal] + tokens + 2
        for policy in ("native", "constrained"):
            per_row[policy].append(totals[policy])
        row_records.append({
            "stable_row_id": row["id"],
            "image_count": len(images),
            "prompt_tokens": prompt_tokens,
            "native_seq_len": totals["native"],
            "constrained_seq_len": totals["constrained"],
        })

    def cache_bytes(seq_lens: list[int]) -> int:
        return sum(seq_lens) * HIDDEN_SIZE * CACHE_DTYPE_BYTES

    summary = {
        "population": {
            "image_bearing_rows": len(image_rows),
            "image_media_records": sum(dimensions.values()),
            "distinct_dimensions": len(dimensions),
        },
        "per_image_merged_tokens": {
            policy: describe(per_image[policy]) for policy in per_image
        },
        "per_row_sequence_length": {
            policy: describe(per_row[policy]) for policy in per_row
        },
        "whole_population_activation_cache_bytes": {
            policy: cache_bytes(per_row[policy]) for policy in per_row
        },
    }

    print("\nmerged visual tokens per image")
    for policy in ("constrained", "native"):
        d = summary["per_image_merged_tokens"][policy]
        print(f"  {policy:<12} median {d['median']:>6}  mean {d['mean']:>8}  "
              f"p99 {d['p99']:>6}  max {d['max']:>6}")

    print("\nnative-H3 sequence length per row (prompt + labels + vision blocks)")
    for policy in ("constrained", "native"):
        d = summary["per_row_sequence_length"][policy]
        print(f"  {policy:<12} median {d['median']:>6}  mean {d['mean']:>8}  "
              f"p99 {d['p99']:>6}  max {d['max']:>6}")

    ratio = (summary["per_row_sequence_length"]["native"]["mean"] /
             summary["per_row_sequence_length"]["constrained"]["mean"])
    print(f"\nnative / constrained mean sequence length: {ratio:.2f}x")

    print("\nactivation cache for the whole image-bearing population, "
          "one hidden state per row")
    for policy in ("constrained", "native"):
        gib = summary["whole_population_activation_cache_bytes"][policy] / 2**30
        print(f"  {policy:<12} {gib:8.2f} GiB")

    print("\nrows exceeding a candidate per-row token budget, native policy")
    for budget in (4096, 8192, 16384, 32768):
        over = sum(1 for r in row_records if r["native_seq_len"] > budget)
        print(f"  > {budget:>6} tokens: {over:>5} of {len(row_records)} rows")

    # The two named options are endpoints. A v2 candidate may instead declare an
    # intermediate max_pixels, applied identically at calibration and serving.
    # Sweep it so the choice can be made against numbers rather than adjectives.
    print("\nfrontier: declared max_pixels between the two endpoints")
    print(f"  {'max_pixels':>12}  {'med tok/img':>11}  {'med row':>8}  "
          f"{'max row':>8}  {'cache GiB':>9}")
    frontier = []
    for max_pixels in (301056, 602112, 1204224, 2408448, 4816896, 12845056):
        by_dimension = {}
        for width, height in dimensions:
            canvas = torch.zeros(1, height, width, 3, dtype=torch.float32)
            _, grid = qwen_vl.process_qwen2vl_images(
                canvas, patch_size=16, max_pixels=max_pixels,
                image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]
            )
            by_dimension[(width, height)] = merged_tokens(grid)

        image_tokens, row_lengths = [], []
        for row in image_rows:
            source = source_rows[row["source_index"]]
            total = text_token_count(tokenizer, source.get("target_ir") or "")
            images = row["image_dimensions"]
            for ordinal, dims in enumerate(images, 1):
                tokens = by_dimension[tuple(dims)]
                image_tokens.append(tokens)
                total += label_tokens[ordinal] + tokens + 2
            row_lengths.append(total)

        entry = {
            "max_pixels": max_pixels,
            "per_image_merged_tokens": describe(image_tokens),
            "per_row_sequence_length": describe(row_lengths),
            "activation_cache_bytes": cache_bytes(row_lengths),
        }
        frontier.append(entry)
        print(f"  {max_pixels:>12}  {entry['per_image_merged_tokens']['median']:>11}  "
              f"{entry['per_row_sequence_length']['median']:>8}  "
              f"{entry['per_row_sequence_length']['max']:>8}  "
              f"{entry['activation_cache_bytes'] / 2**30:>9.2f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "measurement": "still-image policy sequence-length cost on accepted H3-IR pool",
        "candidate_pool": str(POOL.relative_to(REPO)),
        "source_revision": revision,
        "policies": {
            "native": "comfy.text_encoders.qwen_vl.process_qwen2vl_images, "
                      "patch_size 16, defaults 3,136..12,845,056 px",
            "constrained": "h3_awq_encoder._source_image_patches, current W4 "
                           "artifact snapshot, 200,704..301,056 px",
        },
        "caveat": "visual token counts are measured from the real "
                  "implementations; row sequence lengths are constructed by "
                  "assembling the native presentation and are not captured "
                  "launcher sequences",
        "summary": summary,
        "per_dimension": sorted(measured.values(),
                                key=lambda e: -e["source_pixels"]),
        "per_row": row_records,
        "frontier": frontier,
    }, indent=2) + "\n")
    print(f"\nwrote {REPORT.parent.name}/{REPORT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
