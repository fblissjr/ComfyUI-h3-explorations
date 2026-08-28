#!/usr/bin/env python3
"""The converter refuses a delta-encoded head stack, and takes a verbatim one.

## Why this exists when the repo bars new checks

`CLAUDE.md`: no new check until a drift instance appears that the existing gates
provably could not have caught. The instance and the proof:

**The instance.** `Comfy-Org/ComfyUI#15908` changed its head formula after
commit `bd016b75ff9b` to one that is correct only if the stored rows are deltas
from head 0, and the HF repo's `lastModified` sits two minutes after that
commit. So an upstream re-upload in the other encoding is a thing that has
nearly happened once.

**Why nothing installed could catch it.** Two gates touch this area and neither
can see it:

  * `compare_pdd_conversions.kijai_bank` DOES branch between two encodings --
    but on the KEYS PRESENT, which works only because his layouts ship
    different tensor names. A re-upload of the alibaba-pai source keeps
    `proj_out.weight` exactly and changes what is inside it, so both encodings
    have identical keys and key-branching is blind by construction.
  * `compare_pdd_conversions.py`'s `our_bank_is_the_published_stack` compares
    our bank against the same file it was copied from. That agrees with itself
    under either encoding -- it is a copy check, not an encoding check.

The partition guard compares the base checkpoint's head, not the bank. So the
converter's verbatim copy was unguarded end to end.

## What it grades, and why not against its own arithmetic

`CLAUDE.md`: prefer a control the check compares against over asserting against
numbers the test computed itself, INCLUDING through a helper the check defines.
So this file defines no ratio of its own -- it calls the converter's
`bank_row_ratio` and `assert_bank_verbatim`, and deleting the guard makes it go
red rather than leaving it green against a restated copy.

The control is a stack built the two ways upstream can ship one, from the same
underlying heads, so the ONLY difference between the passing and failing case is
the encoding.

Runs with no model files. If the published partitions happen to be on the box it
grades those too, but it does not need them -- requiring a checkout is how
`analyze_routing.py` made itself inert.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convert_pdd_lora import (  # noqa: E402
    BANK_VERBATIM_RATIO, assert_bank_verbatim, bank_row_ratio)

RELEASE = Path.home() / "Storage" / "alibaba-pai_MiniMax-H3-Acc-LoRAs"
KEYS = ("proj_out.weight", "audio_proj_out.weight")


def synth_heads(n=32, out=96, inp=512, spread=0.03, seed=0):
    """A stack shaped like a real one: near-identical consecutive heads.

    `spread` is what makes this a control rather than a toy. The published
    stacks differ head-to-head by 2-3% of a head's norm, so a delta encoding of
    THEM is ~35x below a verbatim one. Building the synthetic heads at the same
    spread means the separation here is the separation there; a stack of
    unrelated random heads would make the test far easier than reality.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(out, inp, generator=g, dtype=torch.float64)
    base /= base.norm()
    step = torch.randn(n, out, inp, generator=g, dtype=torch.float64)
    step /= step.flatten(1).norm(dim=1)[:, None, None]
    return base[None] + spread * step


def main() -> int:
    fails, notes, measured = [], [], {}
    heads = synth_heads()

    # --- the two encodings, from ONE set of heads --------------------------
    verbatim = heads.clone()
    delta = heads.clone()
    delta[1:] = heads[1:] - heads[0]          # rows 1.. become differences

    r_verb = bank_row_ratio(verbatim)
    r_delta = bank_row_ratio(delta)

    if r_verb > BANK_VERBATIM_RATIO > r_delta:
        notes.append(f"  ok    the observable separates   verbatim {r_verb:.4f} "
                     f"> {BANK_VERBATIM_RATIO} > delta {r_delta:.4f}, "
                     f"{r_verb / r_delta:.0f}x apart on heads built at the "
                     f"published 3% spread")
    else:
        fails.append(f"  FAIL  the observable separates   verbatim {r_verb:.4f}, "
                     f"delta {r_delta:.4f}, threshold {BANK_VERBATIM_RATIO}")

    # --- the guard takes the good one and refuses the bad one --------------
    src_ok = {k: verbatim for k in KEYS}
    try:
        ratios = assert_bank_verbatim(src_ok, KEYS)
        notes.append("  ok    verbatim is accepted     "
                     + ", ".join(f"{k} {v:.4f}" for k, v in sorted(ratios.items())))
    except SystemExit as e:
        fails.append(f"  FAIL  verbatim is accepted     refused a correct "
                     f"stack: {e}")

    # THE DELIBERATE VIOLATION. If this passes, the guard is decorative.
    for bad_key in KEYS:
        src_bad = {k: (delta if k == bad_key else verbatim) for k in KEYS}
        try:
            assert_bank_verbatim(src_bad, KEYS)
            fails.append(f"  FAIL  delta is refused         a delta-encoded "
                         f"{bad_key} CONVERTED CLEAN -- the guard is inert")
        except SystemExit as e:
            if bad_key in str(e) and "DELTA" in str(e):
                notes.append(f"  ok    delta is refused         {bad_key} "
                             f"alone is caught, and the message names it")
            else:
                fails.append(f"  FAIL  delta is refused         raised without "
                             f"naming {bad_key}: {e}")

    # --- the published files, when they are here ---------------------------
    if RELEASE.is_dir():
        from safetensors import safe_open
        for part in ("FL2VA", "Ref2VA"):
            f = RELEASE / f"MiniMax-H3-{part}-Acc-8Step.safetensors"
            if not f.exists():
                continue
            with safe_open(str(f), framework="pt") as h:
                src = {k: h.get_tensor(k) for k in KEYS}
            try:
                ratios = assert_bank_verbatim(src, KEYS)
                measured[part] = {k: round(v, 6) for k, v in ratios.items()}
                notes.append(f"  ok    {part} ships verbatim    "
                             + ", ".join(f"{k.split('.')[0]} {v:.4f}"
                                         for k, v in sorted(ratios.items())))
            except SystemExit as e:
                fails.append(f"  FAIL  {part} ships verbatim    {e}")
    else:
        notes.append("  note  published files absent   synthetic cases only; "
                     "the guard is graded, the artifacts are not")

    print("PDD head-bank encoding: verbatim heads, not deltas from head 0")
    for line in notes:
        print(line)
    for line in fails:
        print(line)
    if fails:
        print("\nthe converter would copy a wrongly-encoded bank across intact")
        return 1
    if measured and "--json" in sys.argv:
        import json
        rec = Path(__file__).resolve().parents[1] / "bench/results"
        rec /= "2026-08-28_pdd_bank_encoding.json"
        rec.write_text(json.dumps({
            "date": "2026-08-28",
            "script": "bench/check_pdd_bank_encoding.py",
            "what": "median ||row_i||/||row_0||, i>=1, of the published "
                    "per-interval head stacks. ~1.0 is verbatim heads; a "
                    "delta-from-head-0 encoding would sit near 0.02.",
            "threshold": BANK_VERBATIM_RATIO,
            "ratios": measured,
            "why_it_matters": "bench/convert_pdd_lora.py copies this stack "
                              "into h3_pdd.bank.* unchanged. Upstream "
                              "(Comfy-Org/ComfyUI#15908) adopted a formula "
                              "correct only under the delta encoding, so a "
                              "re-upload in that form would convert silently. "
                              "Re-run against any re-fetched artifact BEFORE "
                              "converting it, and compare to these values.",
            "do_not_rely_on": [
                "This says the ENCODING is verbatim. It says nothing about "
                "whether the weights are the same weights -- a re-upload can "
                "keep the encoding and change the values, which "
                "bench/compare_pdd_conversions.py is for.",
                "The threshold is a discriminator, not a precision tolerance. "
                "Any value in (0.05, 0.9) decides identically; a number tuned "
                "finer than that is a number pretending to measure something.",
            ],
        }, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {rec.relative_to(rec.parents[2])}")
    print("\nall ok -- a delta-encoded stack cannot reach the bank silently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
