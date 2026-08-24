#!/usr/bin/env python3
"""Modality envelope of llm-compressor's sequential calibration trace on Qwen3-VL.

`oneshot`'s sequential pipeline traces the model once, from the first batch the
dataloader yields, and every later batch is executed against that one traced
graph. `populate_concrete_args` bakes any forward parameter absent from the
trace sample in as a constant, so the trace sample's key set -- not the recipe,
not the manifest -- decides which modalities can reach the language stack.

This probe measures that envelope on a tiny random-init Qwen3-VL. It answers:

  Q1  can text-only and vision-bearing rows share one calibration run?
  Q2  does a trace made from a one-image row accept a nine-image row?
  Q3  do H3 two-frame video blocks ride the image keys under that same trace?
  Q4  can an empty pixel_values/[0,3] grid carry a text-only row into a vision
      trace without fabricating media?

A traced subgraph returning without raising proves only that it ran. Each case
therefore also runs a perturbation assay: two batches with identical token ids
and a different image. If the media reached the language stack the subgraph
output moves; if the trace dropped it the output is bit-identical. The
text-only-trace case is the deliberate violation -- it must report zero.

Run under the llm-compressor virtualenv (it needs that checkout's transformers
and llmcompressor). CPU only, no real weights, seconds to run:

    coderef/llm-compressor/.venv/bin/python bench/probe_calibration_input_seam.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

import llmcompressor
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 151652, 151653, 151655, 151656
SEQUENTIAL_TARGETS = ["Qwen3VLTextDecoderLayer"]
REPORT = (Path(__file__).resolve().parent / "results" /
          "2026-08-24_calibration_input_seam_probe.json")


def tiny_qwen3vl():
    """Smallest model that keeps the real forward structure: vision tower,
    DeepStack merger, image-pad scatter, M-RoPE, and text decoder layers."""
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=4096),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
                           num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0],
                           num_position_embeddings=1024),
        image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    return Qwen3VLForConditionalGeneration(cfg).eval(), cfg


def mm_token_type_ids(ids):
    """text 0 / image 1 / video 2, the derivation transformers' processor uses.
    Required since transformers 5.x whenever a grid is passed."""
    t = torch.zeros(1, len(ids), dtype=torch.int)
    for i, v in enumerate(ids):
        t[0, i] = 1 if v == IMAGE_PAD else (2 if v == VIDEO_PAD else 0)
    return t


def vision_row(cfg, grids, seed=0):
    """One row carrying len(grids) grid_t=1 vision blocks.

    An H3 two-frame video block is also a grid_t=1 block whose patch vector
    holds both frames, so image rows and video rows share this one key set --
    see comfy/text_encoders/minimax.py::process_video_block.
    """
    vc = cfg.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    ids, n_patches = [], 0
    for gh, gw in grids:
        merged = (gh // vc.spatial_merge_size) * (gw // vc.spatial_merge_size)
        ids += [100, 101, VISION_START] + [IMAGE_PAD] * merged + [VISION_END]
        n_patches += gh * gw
    ids += list(range(200, 206))
    gen = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "mm_token_type_ids": mm_token_type_ids(ids),
        "pixel_values": torch.randn(n_patches, patch_dim, generator=gen),
        "image_grid_thw": torch.tensor([[1, gh, gw] for gh, gw in grids]),
    }


def text_row(cfg, n=12, empty_vision=False):
    ids = list(range(300, 300 + n))
    row = {"input_ids": torch.tensor([ids]),
           "attention_mask": torch.ones(1, n, dtype=torch.long),
           "mm_token_type_ids": mm_token_type_ids(ids)}
    if empty_vision:
        vc = cfg.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
        row["pixel_values"] = torch.zeros(0, patch_dim)
        row["image_grid_thw"] = torch.zeros(0, 3, dtype=torch.long)
    return row


def _largest_float_output(outputs):
    tensors = [v for v in outputs.values()
               if torch.is_tensor(v) and v.is_floating_point() and v.dim() >= 2]
    return max(tensors, key=lambda t: t.numel())


def feed(model, subgraph, batch):
    """Run one batch exactly as SequentialPipeline does: fetch only the keys the
    traced graph declares, then call it."""
    inputs = {k: v for k, v in batch.items() if k in subgraph.input_names}
    result = {"passed": sorted(inputs),
              "dropped_by_trace": sorted(set(batch) - set(inputs)),
              "missing_for_trace": sorted(set(subgraph.input_names) - set(inputs))}
    try:
        with torch.no_grad():
            out = subgraph.forward(model, **inputs)
        result["outcome"] = "ran"
        result["_out"] = out
    except Exception as exc:
        result["outcome"] = "raised"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).strip().splitlines()[-1][:160]
    return result


def media_influence(model, subgraph, cfg, grids):
    """Identical tokens, different image. Returns max|h_A - h_B| on subgraph-0's
    hidden-state output. Zero means the media never reached the language stack."""
    a, b = vision_row(cfg, grids, seed=1), vision_row(cfg, grids, seed=999)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert not torch.equal(a["pixel_values"], b["pixel_values"])
    ra, rb = feed(model, subgraph, a), feed(model, subgraph, b)
    if ra["outcome"] != "ran" or rb["outcome"] != "ran":
        return None
    return (_largest_float_output(ra["_out"]) -
            _largest_float_output(rb["_out"])).abs().max().item()


def run_case(name, question, make_trace_row, feeds, assay_grids=None):
    model, cfg = tiny_qwen3vl()
    trace_row = make_trace_row(cfg)
    case = {"case": name, "question": question, "trace_row_keys": sorted(trace_row)}
    try:
        subgraphs = trace_subgraphs(model, trace_row, SEQUENTIAL_TARGETS, [], 1)
    except Exception as exc:
        case["trace"] = "failed"
        case["error"] = f"{type(exc).__name__}: {exc}"
        return case
    sg0 = subgraphs[0]
    case["trace"] = "ok"
    case["num_subgraphs"] = len(subgraphs)
    case["subgraph0_input_names"] = sorted(sg0.input_names)
    case["feeds"] = []
    for label, make_row in feeds:
        r = feed(model, sg0, make_row(cfg))
        r.pop("_out", None)
        r["row"] = label
        case["feeds"].append(r)
    if assay_grids is not None:
        delta = media_influence(model, sg0, cfg, assay_grids)
        case["media_influence_max_abs_delta"] = delta
        case["media_reached_language_stack"] = None if delta is None else bool(delta > 0)
    return case


def main():
    torch.manual_seed(0)
    one_image = lambda cfg: vision_row(cfg, [(4, 4)])
    nine_images = lambda cfg: vision_row(cfg, [(4, 4)] * 4 + [(6, 6)] * 5)
    three_blocks = lambda cfg: vision_row(cfg, [(6, 8)] * 3)
    text_only = lambda cfg: text_row(cfg)
    text_empty_vision = lambda cfg: text_row(cfg, empty_vision=True)

    cases = [
        run_case("Q1a vision trace, text-only row",
                 "can a text-only row join a run traced from a vision row?",
                 one_image, [("one-image", one_image), ("text-only", text_only)],
                 assay_grids=[(4, 4)]),
        run_case("Q1b text-only trace, vision row",
                 "can a vision row join a run traced from a text-only row?",
                 text_only, [("text-only", text_only), ("one-image", one_image)],
                 assay_grids=[(4, 4)]),
        run_case("Q2 vision trace, nine-image row",
                 "does a one-image trace accept a nine-image row?",
                 one_image, [("nine-image", nine_images)],
                 assay_grids=[(4, 4)] * 4 + [(6, 6)] * 5),
        run_case("Q3 vision trace, three two-frame video blocks",
                 "do H3 two-frame video blocks ride the image keys?",
                 one_image, [("three two-frame blocks", three_blocks)],
                 assay_grids=[(6, 8)] * 3),
        run_case("Q4 vision trace, text-only row with empty vision tensors",
                 "can empty pixel_values/[0,3] grid carry a text-only row?",
                 one_image, [("text-only + empty vision", text_empty_vision)]),
    ]

    report = {
        "probe": "llm-compressor sequential-trace modality envelope (Qwen3-VL)",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "llmcompressor": llmcompressor.__version__,
        },
        "cases": cases,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["environment"], indent=2))
    silent_drops = []
    for c in cases:
        print(f"\n{c['case']}  --  {c['question']}")
        if c["trace"] != "ok":
            print(f"  TRACE FAILED: {c.get('error')}")
            continue
        print(f"  subgraph0 declares: {c['subgraph0_input_names']}")
        for f in c["feeds"]:
            detail = f.get("error", "") if f["outcome"] == "raised" else ""
            print(f"  feed {f['row']:<26} dropped={f['dropped_by_trace'] or '-'} "
                  f"missing={f['missing_for_trace'] or '-'} -> {f['outcome']} {detail}")
        if "media_influence_max_abs_delta" in c:
            d = c["media_influence_max_abs_delta"]
            if d is None:
                print("  media influence: not measurable (a feed raised)")
                continue
            verdict = ("media REACHED the language stack" if d > 0
                       else "media NEVER REACHED it -- SILENT DROP")
            print(f"  media influence: max|h(imgA)-h(imgB)| = {d:.6g}  ->  {verdict}")
            if d == 0:
                silent_drops.append(c["case"])

    print(f"\nwrote {REPORT.parent.name}/{REPORT.name}")
    if silent_drops:
        print("silent-media-drop configurations (must never be used for "
              f"calibration): {silent_drops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
