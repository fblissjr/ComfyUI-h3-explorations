#!/usr/bin/env python3
"""How much does each artifact move the seven H3 marker embedding rows?

## What this decides

`bench/results/2026-08-27_marker_tokenization_alignment.json` raised a concern:
AWQ never touches `embed_tokens` (an `nn.Embedding` is not a `Linear` target,
and `layer0_input` relative L2 is exactly 0.0 for v1 and v2), but **NVFP4 ships
the embedding table quantized**, so it is the one artifact that perturbs the
marker rows -- and five of the seven have never appeared in any population it
was graded on.

That concern is answerable by reading the rows rather than by rendering
anything, and the scale to read it against already exists: the
`mean_init_rows` arm replaces each row with the table mean, and
`bench/results/2026-08-27_marker_epsilon.json` measured the DiT's response to
that substitution at up to 13.5%.

## Why the ratio is the number and the relative L2 is not

A relative L2 of 0.008 means nothing on its own -- the reader cannot tell
whether that is a lot for an embedding row. Against an arm whose perturbation
has a MEASURED downstream consequence, it is interpretable: it says what
fraction of a known-consequential change this artifact applies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
# ComfyUI last so it lands first: this repo has a `nodes.py` and marker_arms
# imports comfy_api. See docs/comfy_notes.md for the `import nodes` trap.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import marker_arms as M  # noqa: E402

FIRST_MARKER_ID = 151669
# Trained and untrained controls, so "how big is 0.008" has two anchors that do
# not come from this script's own arithmetic.
TRAINED_CONTROL = (151644, 151657)   # Qwen's own specials
UNTRAINED_CONTROL = (151700, 151960)  # padding tail


def _rows(path: str, key: str, lo: int, hi: int, scale_key: str | None = None):
    with safe_open(path, framework="pt") as f:
        rows = f.get_slice(key)[lo:hi].to(torch.float32)
        if scale_key is not None:
            rows = rows * f.get_slice(scale_key)[lo:hi].to(torch.float32)
    return rows


def _table_mean(path: str, key: str, width: int, rows_total: int):
    total = torch.zeros(width, dtype=torch.float64)
    with safe_open(path, framework="pt") as f:
        sl = f.get_slice(key)
        for lo in range(0, rows_total, 8192):
            total += sl[lo:lo + 8192].to(torch.float64).sum(0)
    return (total / rows_total).to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="the BF16 release encoder")
    ap.add_argument("--candidate", required=True, help="the artifact to grade")
    ap.add_argument("--reference-key", default="model.language_model.embed_tokens.weight")
    ap.add_argument("--candidate-key", default="model.embed_tokens.weight")
    ap.add_argument("--candidate-scale-key", default="model.embed_tokens.weight_scale",
                    help="omit with '' for an unquantized candidate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = M.marker_tokens()
    n = len(names)
    lo, hi = FIRST_MARKER_ID, FIRST_MARKER_ID + n

    with safe_open(args.reference, framework="pt") as f:
        shape = f.get_slice(args.reference_key).get_shape()
    ref = _rows(args.reference, args.reference_key, lo, hi)
    cand = _rows(args.candidate, args.candidate_key, lo, hi,
                 args.candidate_scale_key or None)

    mean = _table_mean(args.reference, args.reference_key, shape[1], shape[0])

    per_marker, cand_rel, mean_rel = [], [], []
    for i, name in enumerate(names):
        c = float((cand[i] - ref[i]).norm() / ref[i].norm())
        m = float((mean - ref[i]).norm() / ref[i].norm())
        cand_rel.append(c)
        mean_rel.append(m)
        per_marker.append({
            "marker": name, "id": lo + i,
            "candidate_relative_l2": c,
            "candidate_cosine": float(torch.nn.functional.cosine_similarity(
                cand[i], ref[i], dim=0)),
            "mean_init_relative_l2": m,
            "reference_row_norm": float(ref[i].norm()),
        })

    trained = _rows(args.reference, args.reference_key, *TRAINED_CONTROL)
    untrained = _rows(args.reference, args.reference_key, *UNTRAINED_CONTROL)
    avg_c = sum(cand_rel) / n
    avg_m = sum(mean_rel) / n

    record = {
        "measurement": "per-row perturbation of the seven H3 marker embeddings",
        "reference": Path(args.reference).name,
        "candidate": Path(args.candidate).name,
        "per_marker": per_marker,
        "candidate_mean_relative_l2": avg_c,
        "mean_init_rows_mean_relative_l2": avg_m,
        "candidate_as_fraction_of_mean_init": avg_c / avg_m,
        "the_scale_this_is_read_against": (
            "mean_init_rows replaces each row with the table mean and moved the "
            "DiT's prediction by up to 13.5% "
            "(bench/results/2026-08-27_marker_epsilon.json). This candidate's "
            "perturbation expressed as a fraction of that one is the number to "
            "read; the raw relative L2 is not interpretable alone."
        ),
        "row_norm_controls": {
            "trained_qwen_specials": float(trained.norm(dim=1).mean()),
            "the_seven_markers": float(ref.norm(dim=1).mean()),
            "untrained_padding_tail": float(untrained.norm(dim=1).mean()),
        },
        "does_not_establish": (
            "that the downstream effect scales linearly with row perturbation. "
            "A small fraction of a consequential change is evidence of a small "
            "effect only under an assumption this does not test."
        ),
    }

    out = Path(args.out) if args.out else (
        REPO / "bench" / "results" / "2026-08-27_marker_row_perturbation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")

    print(f"{'marker':<20} {'rel L2':>10} {'cosine':>10}")
    for row in per_marker:
        print(f"{row['marker']:<20} {row['candidate_relative_l2']:>10.5f} "
              f"{row['candidate_cosine']:>10.6f}")
    print(f"\ncandidate mean rel L2        : {avg_c:.5f}")
    print(f"mean_init_rows mean rel L2   : {avg_m:.5f}")
    print(f"candidate is {100*avg_c/avg_m:.1f}% of the mean_init_rows perturbation")
    try:
        print(f"wrote {out.relative_to(REPO)}")
    except ValueError:
        print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
