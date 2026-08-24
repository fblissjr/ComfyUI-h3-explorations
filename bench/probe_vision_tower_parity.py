#!/usr/bin/env python3
"""Do ComfyUI and Transformers compute the same Qwen3-VL vision tower output?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`); it also needs
`transformers` importable. CPU only, random-init weights, no checkpoint.

Why this matters for AWQ v2. Codex's capture harness runs entirely inside the
installed ComfyUI path, so both of its arms inherit whatever ComfyUI computes and
it never needs this check. The calibration path does: `llm-compressor` drives
`Qwen3VLForConditionalGeneration`, so v2's activation statistics would be
gathered under Transformers' vision tower and then served under ComfyUI's. If the
two disagree, every calibration statistic is collected from a distribution
inference never produces, and nothing in either codebase would report it.

The vision tower is the half that bites hardest. Its merged output lands on the
`<|image_pad|>` positions, and its DeepStack features are injected into decoder
layers 0..2 -- inside the 50 layers H3 consumes and AWQ quantizes. This is the
companion to `probe_mrope_implementation_parity.py`, which covers position ids.

Both implementations expose the same 39 state-dict keys under matching names, so
weights transfer by a direct `load_state_dict` with no name mapping. That is
itself worth asserting: a silent rename would make every later comparison
meaningless, so the probe checks the key sets agree before it compares anything.

Ends with a mutation control: perturbing one weight must make the comparison
fail. A parity check that cannot go red proves nothing.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
REPORT = REPO / "bench/results/2026-08-24_vision_tower_parity.json"

sys.path.insert(0, str(COMFY))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import Qwen3VLConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLVisionModel as HFVisionModel,
)

import comfy.ops  # noqa: E402
from comfy.text_encoders.qwen3vl import (  # noqa: E402
    QWEN3VL_VISION_COMMON,
    Qwen3VLVisionModel as ComfyVisionModel,
)

VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
              out_hidden_size=64, deepstack_visual_indexes=[0],
              num_position_embeddings=256)


def build_pair():
    """One random-init tower per implementation, weights copied Comfy -> HF."""
    comfy_model = ComfyVisionModel(
        {**QWEN3VL_VISION_COMMON, **VISION},
        device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast,
    ).eval()
    hf_model = HFVisionModel(Qwen3VLConfig(vision_config=VISION).vision_config).eval()

    comfy_keys, hf_keys = set(comfy_model.state_dict()), set(hf_model.state_dict())
    if comfy_keys != hf_keys:
        raise AssertionError(
            "vision state-dict keys diverge, so a weight transfer would be a "
            f"guess: comfy-only={sorted(comfy_keys - hf_keys)[:5]}, "
            f"hf-only={sorted(hf_keys - comfy_keys)[:5]}"
        )

    # Neither framework's own initialization is used. Comfy's ops allocate
    # parameters uninitialized because they expect a checkpoint, so reading them
    # yields garbage that propagates to NaN; HF would initialize differently
    # again. Both models are loaded from one explicitly seeded state dict, so the
    # only thing under test is the arithmetic.
    gen = torch.Generator().manual_seed(1234)
    seeded = {}
    for key, ref in sorted(comfy_model.state_dict().items()):
        if key.endswith("norm1.weight") or key.endswith("norm2.weight") or \
                key.endswith("norm.weight"):
            seeded[key] = torch.ones_like(ref)          # LayerNorm gain
        elif key.endswith(".bias"):
            seeded[key] = torch.zeros_like(ref)
        else:
            seeded[key] = torch.randn(ref.shape, generator=gen) * 0.05
    comfy_model.load_state_dict(seeded, strict=True)
    hf_model.load_state_dict(seeded, strict=True)

    for name, model in (("comfy", comfy_model), ("hf", hf_model)):
        bad = [k for k, v in model.state_dict().items() if not torch.isfinite(v).all()]
        if bad:
            raise AssertionError(f"{name} holds non-finite weights after seeding: {bad[:3]}")
    return comfy_model, hf_model, sorted(comfy_keys)


def patches_for(grids, cfg_patch=16, temporal=2, channels=3):
    """Flat patch tensor and grid_thw for a set of (grid_h, grid_w) blocks."""
    dim = channels * temporal * cfg_patch * cfg_patch
    rows = sum(gh * gw for gh, gw in grids)
    gen = torch.Generator().manual_seed(0)
    return (torch.randn(rows, dim, generator=gen),
            torch.tensor([[1, gh, gw] for gh, gw in grids], dtype=torch.long))


def run_pair(comfy_model, hf_model, patches, grid):
    with torch.no_grad():
        comfy_merged, comfy_deep = comfy_model(patches, grid)
        hf_out = hf_model(patches, grid)
    if isinstance(hf_out, tuple):
        hf_merged, hf_deep = hf_out
    else:
        # transformers returns BaseModelOutputWithDeepstackFeatures; comfy
        # returns a bare (merged, deepstack) pair. Unwrap by field name, not by
        # position -- and note WHICH field: the post-merger output is
        # `pooler_output`, while `last_hidden_state` is the tower's pre-merge
        # state at hidden_size rather than out_hidden_size. Matching those two
        # would compare different tensors and, at the released 1152/5120 sizes,
        # would not even fail on shape the way it does here.
        hf_merged = getattr(hf_out, "pooler_output", None)
        hf_deep = getattr(hf_out, "deepstack_features", None)
    return (comfy_merged, comfy_deep), (hf_merged, hf_deep)


def compare(a, b, label):
    if a is None or b is None:
        return {"tensor": label, "comparable": False,
                "reason": f"one side returned None (comfy={a is None}, hf={b is None})"}
    if tuple(a.shape) != tuple(b.shape):
        return {"tensor": label, "comparable": False,
                "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a.float() - b.float()).abs()
    denom = a.float().norm()
    return {
        "tensor": label,
        "comparable": True,
        "shape": list(a.shape),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "relative_l2": ((a.float() - b.float()).norm() / denom).item() if denom > 0 else None,
        "exact": bool(torch.equal(a, b)),
    }


def case(name, grids, comfy_model, hf_model):
    patches, grid = patches_for(grids)
    (cm, cd), (hm, hd) = run_pair(comfy_model, hf_model, patches, grid)
    results = [compare(cm, hm, "merged")]
    if cd is not None and hd is not None:
        n = min(len(cd), len(hd))
        if len(cd) != len(hd):
            results.append({"tensor": "deepstack", "comparable": False,
                            "reason": f"{len(cd)} features vs {len(hd)}"})
        for i in range(n):
            results.append(compare(cd[i], hd[i], f"deepstack[{i}]"))
    elif (cd is None) != (hd is None):
        results.append({"tensor": "deepstack", "comparable": False,
                        "reason": f"comfy={cd is None}, hf={hd is None}"})
    return {"case": name, "grids": grids,
            "patch_rows": int(patches.shape[0]), "tensors": results}


def worst(entry):
    vals = [t["max_abs_diff"] for t in entry["tensors"] if t.get("comparable")]
    return max(vals) if vals else None


def main() -> int:
    torch.manual_seed(0)
    comfy_model, hf_model, keys = build_pair()

    fixtures = [
        ("single square block", [(4, 4)]),
        ("single wide block", [(4, 12)]),
        ("single tall block", [(12, 4)]),
        ("two equal blocks", [(4, 4), (4, 4)]),
        ("three mixed blocks", [(4, 4), (6, 8), (8, 6)]),
        ("nine blocks, Ref2VA shaped", [(4, 4)] * 5 + [(6, 6)] * 4),
    ]
    cases = [case(name, grids, comfy_model, hf_model) for name, grids in fixtures]

    for c in cases:
        w = worst(c)
        bad = [t for t in c["tensors"] if not t.get("comparable")]
        note = f"NOT COMPARABLE: {bad[0]['reason']}" if bad else f"max|diff| = {w:.3e}"
        print(f"  {c['case']:<28} rows={c['patch_rows']:>4}  {note}")

    # Mutation control: perturb one HF weight. The comparison must notice.
    with torch.no_grad():
        p = dict(hf_model.named_parameters())["merger.linear_fc2.weight"]
        p[0, 0] += 0.5
    mutated = case("mutation control", [(4, 4)], comfy_model, hf_model)
    detected = (worst(mutated) or 0) > 0
    print(f"\n  mutation control (one merger weight): "
          f"{'detected, as required' if detected else 'FAILED TO DETECT'}")

    report = {
        "probe": "ComfyUI vs Transformers Qwen3-VL vision-tower parity",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "state_dict_keys_agree": True,
        "num_shared_keys": len(keys),
        "cases": cases,
        "mutation_control": {"detects_mutation": detected,
                             "max_abs_diff": worst(mutated)},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  {len(keys)} state-dict keys shared, transferred without mapping")
    print(f"  wrote {REPORT.parent.name}/{REPORT.name}")
    return 0 if detected else 1


if __name__ == "__main__":
    print("ComfyUI vs Transformers Qwen3-VL vision tower\n")
    sys.exit(main())
