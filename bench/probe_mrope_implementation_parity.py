#!/usr/bin/env python3
"""Do ComfyUI and Transformers compute the same M-RoPE position ids?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`), which must also be
able to import `transformers` (the llm-compressor venv has both if this one does
not). CPU only, no weights.

Why this matters for AWQ v2. Codex's capture harness runs entirely inside the
installed ComfyUI path, so both of its arms inherit whatever ComfyUI computes and
it never needs this check. The calibration path does not: `llm-compressor` drives
`Qwen3VLForConditionalGeneration`, so v2's activation statistics would be
collected under Transformers' position ids and then served under ComfyUI's. Two
independent implementations of the same idea either agree or they do not. If they
do not, every calibration statistic is gathered from a distribution inference
never produces, and nothing in either codebase would report it.

Position ids are a good place to look first because they need no weights, they
feed every decoder layer through rotary embedding, and the two implementations
were written from different starting points:

  ComfyUI      comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids
               walks `embeds_info` spans and assigns grid positions per span
  Transformers Qwen3VLModel.get_rope_index, groups `mm_token_type_ids` runs and
               consumes an iterator of grids per modality

The probe builds the same presentation for both, compares (3, seq_len) integer
outputs exactly, and ends with a mutation control: a deliberately shifted vision
span must make the comparison fail. A parity check that cannot go red proves
nothing.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
REPORT = REPO / "bench/results/2026-08-24_mrope_implementation_parity.json"

sys.path.insert(0, str(COMFY))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import Qwen3VLConfig, Qwen3VLModel  # noqa: E402

import comfy.text_encoders.qwen_vl as qwen_vl  # noqa: E402

VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
MERGE = 2


def build_presentation(blocks, lead=3, gap=2, tail=4):
    """One native-H3-shaped row: text, then per block a vision span, then text.

    `blocks` is a list of (grid_h, grid_w). Returns the pieces each
    implementation needs, built once so neither can be fed a different sequence.
    """
    ids, embeds_info = [], []
    ids += list(range(1000, 1000 + lead))
    for gh, gw in blocks:
        merged = (gh // MERGE) * (gw // MERGE)
        ids.append(VISION_START)
        start = len(ids)                      # first expanded embedding position
        ids += [IMAGE_PAD] * merged
        ids.append(VISION_END)
        embeds_info.append({
            "type": "image",
            "index": start,
            "size": merged,
            "extra": {"grid": torch.tensor([[1, gh, gw]], dtype=torch.long)},
        })
        ids += list(range(2000, 2000 + gap))
    ids += list(range(3000, 3000 + tail))

    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "embeds_info": embeds_info,
        "image_grid_thw": torch.tensor([[1, gh, gw] for gh, gw in blocks],
                                       dtype=torch.long),
        "mm_token_type_ids": torch.tensor(
            [[1 if t == IMAGE_PAD else 0 for t in ids]], dtype=torch.int),
        "seq_len": len(ids),
    }


def comfy_position_ids(p):
    out = qwen_vl.qwen2vl_mrope_position_ids(
        p["embeds_info"], p["seq_len"], torch.device("cpu")
    )
    return None if out is None else out.to(torch.long)


def transformers_position_ids(model, p):
    pos, _ = model.get_rope_index(
        input_ids=p["input_ids"],
        mm_token_type_ids=p["mm_token_type_ids"],
        image_grid_thw=p["image_grid_thw"],
        video_grid_thw=None,
        attention_mask=torch.ones_like(p["input_ids"]),
    )
    return pos[:, 0, :].to(torch.long)          # (3, B, S) -> (3, S)


def compare(a, b):
    if a is None or b is None:
        return {"agree": False, "reason": "one implementation returned None"}
    if a.shape != b.shape:
        return {"agree": False, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a != b)
    n = int(diff.sum())
    result = {"agree": n == 0, "shape": list(a.shape), "mismatched_positions": n}
    if n:
        idx = diff.any(dim=0).nonzero().flatten()
        first, last = int(idx[0]), int(idx[-1])
        result["first_mismatch"] = first
        result["last_mismatch"] = last
        result["comfy_at_first"] = a[:, first].tolist()
        result["transformers_at_first"] = b[:, first].tolist()
        # a constant offset is a different finding from a structural disagreement
        deltas = {tuple((a[:, i] - b[:, i]).tolist()) for i in idx.tolist()}
        result["distinct_deltas"] = len(deltas)
        result["sample_deltas"] = [list(d) for d in list(deltas)[:4]]
    return result


def main() -> int:
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                           num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0]),
        image_token_id=IMAGE_PAD, vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )
    model = Qwen3VLModel(cfg).eval()

    fixtures = [
        ("single square block", [(4, 4)]),
        ("single wide block", [(4, 12)]),
        ("single tall block", [(12, 4)]),
        ("two equal blocks", [(4, 4), (4, 4)]),
        ("three mixed blocks", [(4, 4), (6, 8), (8, 6)]),
        ("nine blocks, Ref2VA shaped", [(4, 4)] * 5 + [(6, 6)] * 4),
    ]

    cases = []
    for name, blocks in fixtures:
        p = build_presentation(blocks)
        result = compare(comfy_position_ids(p), transformers_position_ids(model, p))
        result.update(case=name, blocks=blocks, seq_len=p["seq_len"])
        cases.append(result)
        verdict = "agree" if result["agree"] else \
            f"DISAGREE ({result.get('mismatched_positions', '?')} positions)"
        print(f"  {name:<28} seq={p['seq_len']:>4}  {verdict}")

    # Mutation control: shift one vision span so the two are fed different
    # structure. This must fail; a check that cannot go red proves nothing.
    p = build_presentation([(4, 4), (6, 8)])
    p["embeds_info"][1]["index"] += 1
    mutated = compare(comfy_position_ids(p), transformers_position_ids(model, p))
    print(f"\n  mutation control (shifted span): "
          f"{'FAILED TO DETECT' if mutated['agree'] else 'detected, as required'}")

    report = {
        "probe": "ComfyUI vs Transformers M-RoPE position-id parity",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "cases": cases,
        "mutation_control": {**mutated, "detects_mutation": not mutated["agree"]},
        "all_agree": all(c["agree"] for c in cases),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n  all fixtures agree: {report['all_agree']}")
    print(f"  wrote {REPORT.parent.name}/{REPORT.name}")
    return 0 if report["mutation_control"]["detects_mutation"] else 1


if __name__ == "__main__":
    print("ComfyUI vs Transformers M-RoPE position ids\n")
    sys.exit(main())
