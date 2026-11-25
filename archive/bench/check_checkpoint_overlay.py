#!/usr/bin/env python3
"""An overlay plus its base is the target, tensor for tensor, and the pieces select what they name.

`checkpoint_overlay.py` claims: base plus every piece of an overlay equals the
target it was built from, an overlay refuses any other base, and a selection
of pieces gives the checkpoint that piece list describes. This checks it.

**Synthetic** (always runs, no model file): tiny safetensors files with the
shapes of each case an H3 overlay meets (a coded int8 linear, a rescaled one,
a bf16 tensor that became int8, gates that exist only in the target, adaln, a
norm). Each claim is then broken on purpose, and the check must go red:

    flipped delta   one code delta of the overlay changed by 1
    dropped piece   the piece holding a coded linear left out of the selection
    wrong base      a base file with one code changed: refused by content
    extra tensor    a target that lacks a base tensor: refused at build

**Real files** (`--models`, needs the checkpoints on disk, CPU only, tens of
GB read): the FastH3 V2 overlay on fl2va applies to FastH3 V2 whole; the
selection "gates only" equals #35's `hybrid__fl2va-weights__plus-fasth3v2-gates`
and "everything but the gates" equals `hybrid__fasth3v2-weights__no-gates`. A
base with one code flipped must NOT give FastH3 V2, and hybrid A as the base
must be refused. Also `gate_scale`: 0.5 halves every gate row scale and touches
nothing else; a scale with no gates present is an error.

Claims, i.e. what breaks if a case is deleted:

- `full overlay on its base equals the target` -- the format is exact on every
  case an H3 overlay meets: a coded linear (±2 deltas included), a rescaled one,
  bf16 to int8, tensors only the target has, adaln, a norm.
- `the base is not modified` -- a coded linear is cloned before its positions
  are written; the loaded base can be reused.
- `gates alone ...`, `everything but the gates ...`, `one block's diff ...` --
  a selection is exactly the pieces it names, and no others.
- `pieces are the ones the rule names`, `every overlay tensor sits in the piece
  its key names` -- the piece table is the rule's.
- `gate_scale ...` -- the dial touches gate row scales only, exactly, and has
  no silent no-op.
- the four controls -- each of the checks above can go red: a wrong delta, a
  dropped piece, a wrong base and a target missing a base tensor.
- on real files, FastH3 V2 whole, and the two hybrids by selection, and the two
  controls that a flipped code and a wrong base file are seen.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_checkpoint_overlay.py [--models <models dir>]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from checkpoint_overlay import (GATE_SCALE_SUFFIX, apply_overlay, build_overlay, check_base,  # noqa: E402
                                piece_of, read_overlay, select_pieces)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(name)


def raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def differs(applied: dict, target: dict | str) -> list[str]:
    """Keys where `applied` is not the target: absent, extra, or unequal in dtype, shape or value."""
    if isinstance(target, str):
        with safe_open(target, "pt") as f:
            keys, get = set(f.keys()), f.get_tensor
            bad = sorted(keys ^ set(applied))
            for k in sorted(keys & set(applied)):
                t, a = get(k), applied[k]
                if t.dtype != a.dtype or t.shape != a.shape or not torch.equal(t, a):
                    bad.append(k)
            return bad
    bad = sorted(set(target) ^ set(applied))
    return bad + [k for k in sorted(set(target) & set(applied))
                  if target[k].dtype != applied[k].dtype or target[k].shape != applied[k].shape
                  or not torch.equal(target[k], applied[k])]


def synthetic() -> None:
    print("synthetic")
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-127, 127, (64, 32), dtype=torch.int8, generator=g)
    base = {
        "blocks.0.attn.qkv_proj.weight": codes.clone(),
        "blocks.0.attn.qkv_proj.weight_scale": torch.rand(64, 1, generator=g),
        "blocks.1.mlp.fc1.weight": codes.clone(),
        "blocks.1.mlp.fc1.weight_scale": torch.rand(64, 1, generator=g),
        "blocks.1.norm1.weight": torch.rand(8, generator=g).bfloat16(),
        "blocks.1.adaln_proj.linear.weight": torch.rand(4, 8, generator=g).bfloat16(),
        "adaln_t_table": torch.rand(4, generator=g).bfloat16(),
        "token_refiner.blocks.0.attn.qkv_proj.weight": torch.rand(16, 8, generator=g).bfloat16(),
        "video_patch_proj.weight": torch.rand(8, 8, generator=g).bfloat16(),
    }
    target = {k: v.clone() for k, v in base.items()}
    moved = codes.clone().view(-1)
    moved[[3, 700, 2000]] += torch.tensor([1, -1, 2], dtype=torch.int8)
    target["blocks.0.attn.qkv_proj.weight"] = moved.view(64, 32)
    target["blocks.0.attn.qkv_proj.weight_scale"] = base["blocks.0.attn.qkv_proj.weight_scale"] * 1.01
    target["blocks.1.norm1.weight"] = base["blocks.1.norm1.weight"] + 1
    target["blocks.1.adaln_proj.linear.weight"] = base["blocks.1.adaln_proj.linear.weight"] * 2
    target["adaln_t_table"] = base["adaln_t_table"] * 2
    target["token_refiner.blocks.0.attn.qkv_proj.weight"] = torch.randint(-127, 127, (16, 8), dtype=torch.int8, generator=g)
    target["token_refiner.blocks.0.attn.qkv_proj.weight_scale"] = torch.rand(16, 1, generator=g)
    for b in (0, 1):
        target[f"blocks.{b}.attn.to_gate_compress.weight"] = torch.randint(-127, 127, (8, 8), dtype=torch.int8, generator=g)
        target[f"blocks.{b}.attn.to_gate_compress.weight_scale"] = torch.rand(8, 1, generator=g)
    target["video_patch_proj.weight"] = base["video_patch_proj.weight"] + 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = {n: str(tmp / f"{n}.safetensors") for n in ("base", "target", "overlay", "other_base", "short")}
        save_file(base, paths["base"])
        save_file(target, paths["target"])
        moved_base = {k: v.clone() for k, v in base.items()}
        moved_base["blocks.1.mlp.fc1.weight"].view(-1)[5] += 1
        save_file(moved_base, paths["other_base"])
        save_file({k: v for k, v in target.items() if k != "video_patch_proj.weight"}, paths["short"])

        stats = build_overlay(paths["base"], paths["target"], paths["overlay"])
        meta, pieces = read_overlay(paths["overlay"])
        everything = select_pieces(pieces, frozenset(range(2)), True, True, True, True)
        applied = apply_overlay(base, paths["overlay"], everything)
        check("full overlay on its base equals the target", not differs(applied, target), str(differs(applied, target)))
        check("the coded linear is coded, the small one whole",
              "idx.blocks.0.attn.qkv_proj.weight" in pieces["backbone.0"] and "full.blocks.0.attn.qkv_proj.weight_scale" in pieces["backbone.0"])
        check("pieces are the ones the rule names", set(pieces) == {"backbone.0", "backbone.1", "gates", "refiner", "adaln", "io"}, str(sorted(pieces)))
        check("every overlay tensor sits in the piece its key names",
              all(piece_of(n.split(".", 1)[1]) == p for p, names in pieces.items() for n in names))
        check("the delta is the true difference, ±2 included", stats["max_abs_delta"] == 2, str(stats))
        check("the base is not modified", not differs(base, load_file(paths["base"])))

        only_gates = apply_overlay(base, paths["overlay"], select_pieces(pieces, frozenset(), True, False, False, False))
        expect = dict(base) | {k: v for k, v in target.items() if "to_gate_compress" in k}
        check("gates alone is the base plus the gates", not differs(only_gates, expect))
        no_gates = apply_overlay(base, paths["overlay"], select_pieces(pieces, frozenset(range(2)), False, True, True, True))
        expect = {k: v for k, v in target.items() if "to_gate_compress" not in k}
        check("everything but the gates is the target without them", not differs(no_gates, expect))
        block1 = apply_overlay(base, paths["overlay"], select_pieces(pieces, frozenset({1}), False, False, False, False))
        check("one block's diff touches that block alone",
              set(differs(block1, base)) == {"blocks.1.norm1.weight"}, str(differs(block1, base)))

        half = apply_overlay(base, paths["overlay"], everything, gate_scale=0.5)
        wrong = [k for k in differs(half, applied) if not k.endswith(GATE_SCALE_SUFFIX)]
        check("gate_scale 0.5 changes the gate row scales and nothing else", not wrong and
              all(torch.equal(half[k], applied[k] * 0.5) for k in applied if k.endswith(GATE_SCALE_SUFFIX)))
        check("gate_scale with no gates present is an error",
              raises(lambda: apply_overlay(base, paths["overlay"], [], gate_scale=0.5)))

        print("controls, each must go red")
        flipped = {k: v.clone() for k, v in load_file(paths["overlay"]).items()}
        flipped["delta.blocks.0.attn.qkv_proj.weight"][0] += 1
        save_file(flipped, paths["overlay"], metadata=meta)
        check("control: a flipped delta no longer gives the target",
              bool(differs(apply_overlay(base, paths["overlay"], everything), target)))
        check("control: leaving out the coded piece no longer gives the target",
              bool(differs(apply_overlay(base, paths["overlay"], [p for p in everything if p != "backbone.0"]), target)))
        check("control: a base with one code changed is refused",
              raises(lambda: check_base(meta, paths["other_base"])))
        check("control: a target that lacks a base tensor is refused",
              raises(lambda: build_overlay(paths["base"], paths["short"], paths["overlay"])))
        check("the right base is accepted", not raises(lambda: check_base(meta, paths["base"])))


def real(models: Path) -> None:
    print("real files")
    m = models / "diffusion_models"
    base_path = str(m / "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    overlay_path = str(models / "h3_overlays" / "fasth3_v2_on_fl2va.h3overlay.safetensors")
    fasth3 = str(m / "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors")
    hybrid_a = str(m / "h3_research" / "hybrid__fl2va-weights__plus-fasth3v2-gates__int8.safetensors")
    hybrid_b = str(m / "h3_research" / "hybrid__fasth3v2-weights__no-gates__int8.safetensors")
    meta, pieces = read_overlay(overlay_path)
    allb = frozenset(range(1 + max(int(p.split(".")[1]) for p in pieces if p.startswith("backbone."))))
    base = load_file(base_path)

    check("the overlay names this base by content", not raises(lambda: check_base(meta, base_path)))
    a = apply_overlay(base, overlay_path, select_pieces(pieces, allb, True, True, True, True))
    check("base + full overlay == FastH3 V2, tensor for tensor", not differs(a, fasth3), str(differs(a, fasth3)[:5]))
    a = apply_overlay(base, overlay_path, select_pieces(pieces, frozenset(), True, False, False, False))
    check("gates only == hybrid: fl2va weights + FastH3 V2 gates", not differs(a, hybrid_a), str(differs(a, hybrid_a)[:5]))
    a = apply_overlay(base, overlay_path, select_pieces(pieces, allb, False, True, True, True))
    check("all but gates == hybrid: FastH3 V2 weights, no gates", not differs(a, hybrid_b), str(differs(a, hybrid_b)[:5]))

    a = apply_overlay(base, overlay_path, select_pieces(pieces, allb, True, True, True, True), gate_scale=0.5)
    full = apply_overlay(base, overlay_path, select_pieces(pieces, allb, True, True, True, True))
    moved = [k for k in differs(a, full)]
    check("gate_scale 0.5 moves only the gate row scales", moved and all(k.endswith(GATE_SCALE_SUFFIX) for k in moved),
          str(moved[:3]))
    check("gate_scale 0.5 halves them exactly", all(torch.equal(a[k], full[k] * 0.5) for k in moved))
    check("gate_scale on a selection with no gates is an error",
          raises(lambda: apply_overlay(base, overlay_path, select_pieces(pieces, allb, False, False, False, False), gate_scale=0.5)))
    del a, full

    print("controls, each must go red")
    code_key = "blocks.7.attn.qkv_proj.weight"
    hurt = dict(base)
    hurt[code_key] = base[code_key].clone()
    hurt[code_key].view(-1)[0] += 1
    a = apply_overlay(hurt, overlay_path, select_pieces(pieces, allb, True, True, True, True))
    check("control: a base with one code changed does not give FastH3 V2", differs(a, fasth3) == [code_key], str(differs(a, fasth3)[:5]))
    check("control: hybrid A as the base is refused", raises(lambda: check_base(meta, hybrid_a)))
    del a, hurt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", type=Path, default=None, help="the ComfyUI models directory; runs the real-file half")
    args = ap.parse_args()
    synthetic()
    if args.models:
        real(args.models)
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall ok")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
