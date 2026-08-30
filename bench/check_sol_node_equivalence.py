#!/usr/bin/env python3
"""Check that our forked Sol node computes what the vendored one computed.

`sol_attn_h3.py` is a fork of `vendor/sol_attn_minimax.py` (see that file's
header for the four local changes). Three of the four are schema or plumbing;
the fourth removed two arguments. This asserts what that is supposed to mean:
**at the settings this repo ships, the two dispatch paths produce the SAME
BYTES**, so migrating a graph from one node to the other cannot move a render.

Why bit-identical rather than a tolerance. Both paths call
`comfy_kitchen.sol_attn` with the same arguments -- the vendored node stopped
passing `centroid_tail` the moment the merged kernel stopped accepting it, and
the fork passes `tail=True`, which is that kernel's default. So there is no
approximation between them to tolerate. A tolerance here would pass on a real
divergence; equality is the only bar that says what we mean.

This is the control `docs/eval_comparison.md` cannot give us. A rendered pair
cannot A/B a numerical change -- the trajectory diverges completely from any
perturbation -- so the comparison has to happen at the CALL. That is the same
argument `bench/grade_sage_on_capture.py` rests on.

What each case claims:

  vendored node imports    the fork's baseline still exists and still loads.
                           When `vendor/sol_attn_minimax.py` is retired to a
                           read-only reference this SKIPS rather than fails,
                           and a skip exits 2 -- a check that did not run must
                           not read as one that passed.
  identical at shipped     the migration is output-neutral at the settings
    settings               `h3_config.SOL_RECOMMENDED_CUDA` carries.
  identical under top-k    the other selection, which no shipped graph uses
                           and which a bench arm can reach.
  pooled_tail changes it   the red control. `tail` is the one argument the
                           fork added, so if turning it off does NOT move the
                           output, this file is comparing a knob that is not
                           connected and every case above proves nothing.

**Scope, and it is narrower than "the fork is safe".** This grades `_run`,
the dispatch. It does not exercise the sink derivation, the sigma window,
`dense_blocks`, the Morton reorder or the override chaining -- those are
unmodified by the fork rather than verified by this file, and "unmodified" is
a claim about a diff, not a measurement. The node's composition seam is
covered by `bench/smoke_h3.py` and by nothing else.

Needs CUDA and a comfy_kitchen carrying the merged `sol_attn`. Exit 0 all
passed, 1 a case failed, 2 nothing was graded.

    python bench/check_sol_node_equivalence.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent


def load_module(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if not torch.cuda.is_available():
        print("no CUDA; the kernel cannot run. Nothing checked.")
        return 2
    sys.path.insert(0, str(COMFY))

    # Our node, loaded as a package member so its relative imports resolve.
    load_module("h3x", REPO / "__init__.py", package_dir=REPO)
    ours = load_module("h3x.sol_attn_h3", REPO / "sol_attn_h3.py")

    vendored_path = REPO / "vendor" / "sol_attn_minimax.py"
    theirs = None
    if vendored_path.is_file():
        try:
            theirs = load_module("_vendored_sol", vendored_path)
        except Exception as exc:
            print(f"  SKIP vendored node imports   {exc}")
    else:
        print(f"  SKIP vendored node imports   no file at vendor/{vendored_path.name}")

    # A shape in the regime the gate actually selects: above min_tokens, more
    # than one block, and head_dim 128 because the kernel requires it.
    torch.manual_seed(0)
    b, h, t, d = 1, 8, 16384, 128
    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    common = dict(skip_reshape=True, skip_output_reshape=True, scale=None,
                  min_tokens=12288, verbose=False)

    failures, skipped, record = [], [], {}

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def ours_run(**kw):
        # tau is positional-required on both sides; the top-k case does not
        # name it and the kernel ignores it there, so pin the shipped value.
        return ours._run(q, k, v, h, **{**common, "tau": 1.0, **kw})

    def theirs_run(**kw):
        # Positional, because the vendored signature still carries the two
        # arguments the merged kernel dropped and we must not pass them by name.
        return theirs._run(q, k, v, h, common["skip_reshape"],
                           common["skip_output_reshape"], common["scale"],
                           kw.get("tau", 1.0), common["min_tokens"],
                           common["verbose"], (0, 0), (0, 0), True, False,
                           kw.get("topk_ratio", 0.0))

    if theirs is None:
        skipped.append("vendored node imports")
        skipped.append("identical at shipped settings")
        skipped.append("identical under top-k")
    else:
        check("vendored node imports", True, "the fork's baseline is loadable")
        for label, kw in (("identical at shipped settings", dict(tau=1.0)),
                          ("identical under top-k", dict(topk_ratio=0.10))):
            a, bb = ours_run(**kw), theirs_run(**kw)
            same = torch.equal(a, bb)
            record[label] = {"bit_identical": bool(same), **kw}
            check(label, same,
                  "same bytes" if same else
                  f"DIFFER: max abs {float((a.float() - bb.float()).abs().max()):.3e}")

    # Red control: the argument the fork added must be connected to something.
    on, off = ours_run(tau=1.0, tail=True), ours_run(tau=1.0, tail=False)
    moved = not torch.equal(on, off)
    a, bb = on.float().flatten(), off.float().flatten()
    cos = float(a @ bb / (a.norm() * bb.norm()))
    record["pooled_tail_cos_on_vs_off"] = cos
    check("pooled_tail changes the output", moved,
          f"cos {cos:.6f} on synthetic input -- a floor, not a quality figure"
          if moved else
          "pooled_tail is not reaching the kernel; every case above is vacuous")

    out = REPO / "bench" / "results" / f"{date.today()}_sol_node_equivalence.json"
    try:
        import importlib.metadata
        build = importlib.metadata.version("comfy-kitchen")
    except Exception:
        build = "unknown"

    out.write_text(json.dumps({
        "what": "our forked Sol node's dispatch against the vendored node's, "
                "at matched settings, on synthetic input",
        "comfy_kitchen": build,
        "shape": {"batch": b, "heads": h, "tokens": t, "head_dim": d,
                  "dtype": "bfloat16"},
        "scope": "the dispatch only. The sink, the sigma window, dense_blocks, "
                 "Morton and the override chaining are unmodified by the fork "
                 "and are not exercised here.",
        "not_a_quality_measurement": "synthetic torch.randn gives a "
                                     "near-uniform softmax, so there is nothing "
                                     "for a block router to find. Cosines here "
                                     "are floors.",
        "cases": record,
    }, indent=2) + "\n")
    print(f"\nrecord: bench/results/{out.name}")

    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    if skipped:
        print(f"INCOMPLETE: {len(skipped)} case(s) skipped: {', '.join(skipped)}")
        return 2
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
