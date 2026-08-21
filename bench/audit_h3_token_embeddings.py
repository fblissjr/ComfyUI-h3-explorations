#!/usr/bin/env python3
"""Do the seven H3 marker rows carry trained values, or are they init noise?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Reads weights off
disk; no CUDA, no model load.

`bench/compare_h3_tokenizers.py` establishes that ComfyUI cannot emit IDs
151669-151675. This answers the question that one leaves open and that
`vendor_tokens.py` names in its own docstring: whether routing to those rows
would reach anything.

**The two controls are what make the answer readable**, and neither is
optional. The stock Qwen special tokens at 151643-151668 are rows that are
certainly trained -- if they do not separate from the H3 markers, the norm is
not a discriminator and this script says nothing. The padding tail past 151676
is rows that are certainly *not* trained, because the vocabulary and added
tokens stop at 151676 while the table runs to `vocab_size`. A marker row is
then read against those two poles rather than against a threshold picked here.

**A null result is the informative one.** If the seven land on the untrained
pole, making them reachable changes which sequence the encoder sees without
making the markers mean anything, and that is worth knowing before anyone
treats the tokenizer gap as a fidelity defect.

Covers the official release, every repacked encoder present, and -- when a
stock Qwen3-VL checkout is on the box -- upstream Qwen itself. **That last one
is what makes the result interpretable rather than merely true.** The release's
README says the H3-Encoder "uses the full pretrained weights of Qwen3-VL-32B",
so if the same rows are untrained in a stock Qwen the release never touched,
they are untrained because nobody ever trained them: MiniMax pointed seven
tokenizer entries at Qwen's existing padding rows. Without that arm the reading
"MiniMax trained them and the values happen to look like noise" survives.

The nvfp4 file stores an int8 table with a per-row scale, which has to be
applied before any norm is comparable -- without it every row is scaled
differently and the comparison is noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

RELEASE = Path.home() / "Storage" / "MiniMaxAI_MiniMax-H3" / "text_encoder"
ENCODERS = Path.home() / "ComfyUI" / "models" / "text_encoders"
UPSTREAM = Path.home() / "Storage"
REPACKED = ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]

# Where the vocabulary stops, where the seven sit, and where the table ends.
# `vendor_config` owns the marker list; these are the row indices the release's
# tokenizer assigns them, which is a fact about the loader's append order.
FIRST_MARKER, PAST_MARKER = 151669, 151676
STOCK_SPECIALS = (151643, 151669)


def _rows(path: Path):
    """(embedding table as float, source label) or (None, reason)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as fh:
        keys = [k for k in fh.keys() if "embed_tokens" in k]
        main = [k for k in keys if k.endswith("embed_tokens.weight")]
        if not main:
            return None, f"no embed_tokens.weight in {path.name}"
        w = fh.get_tensor(main[0]).float()
        # An int8 table is meaningless before its per-row scale is applied.
        scale = [k for k in keys if k.endswith("weight_scale")]
        if scale:
            s = fh.get_tensor(scale[0]).float()
            if s.numel() == w.shape[0]:
                w = w * s.view(-1, 1)
    return w, None


def _report(label: str, w) -> dict:
    n = w.norm(dim=1)
    tail_lo = PAST_MARKER
    out = {}
    print(f"\n=== {label}   table {tuple(w.shape)}")
    for name, lo, hi in (("trained control (Qwen specials)", *STOCK_SPECIALS),
                         ("the seven H3 markers", FIRST_MARKER, PAST_MARKER),
                         ("untrained control (padding tail)", tail_lo, w.shape[0])):
        s = n[lo:hi]
        out[name] = {"mean": float(s.mean()), "min": float(s.min()),
                     "max": float(s.max()), "rows": int(hi - lo)}
        print(f"  {name:<34} mean {s.mean():.4f}  "
              f"[{s.min():.4f}, {s.max():.4f}]  n={hi - lo}")

    trained = out["trained control (Qwen specials)"]["mean"]
    markers = out["the seven H3 markers"]["mean"]
    untrained = out["untrained control (padding tail)"]["mean"]
    span = trained - untrained
    if span <= 0:
        print("  -> the controls did not separate; norm says nothing here")
        out["verdict"] = "controls did not separate"
        return out
    # Where the markers sit between the two poles. 0.0 is the untrained pole.
    where = (markers - untrained) / span
    out["position_between_controls"] = where
    verdict = ("indistinguishable from untrained" if where < 0.1 else
               "indistinguishable from trained" if where > 0.9 else
               "between the controls -- read the numbers, not this line")
    out["verdict"] = verdict
    print(f"  -> markers sit at {where:+.3f} of the way from the untrained "
          f"pole to the trained one: {verdict}")
    return out


def main() -> int:
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("safetensors is not importable; run this with the ComfyUI venv "
              "python (see docs/comfy_notes.md)")
        return 2

    record, seen = {}, 0
    targets = []
    if RELEASE.exists():
        idx = RELEASE / "model.safetensors.index.json"
        if idx.exists():
            wmap = json.loads(idx.read_text())["weight_map"]
            shard = next((v for k, v in wmap.items() if "embed_tokens" in k), None)
            if shard:
                targets.append(("official release", RELEASE / shard))
    for name in REPACKED:
        p = ENCODERS / name
        if p.exists():
            targets.append((name.replace(".safetensors", ""), p))
    # Any stock Qwen3-VL will do -- the question is whether the rows are
    # untrained upstream, and that does not depend on the parameter count.
    for d in sorted(UPSTREAM.glob("Qwen3-VL-*")):
        idx = d / "model.safetensors.index.json"
        if not idx.exists():
            continue
        wmap = json.loads(idx.read_text())["weight_map"]
        shard = next((v for k, v in wmap.items() if "embed_tokens" in k), None)
        if shard:
            targets.append((f"upstream {d.name} (H3 never touched this)",
                            d / shard))
            break

    if not targets:
        print("no encoder weights reachable on this box; nothing to audit")
        return 2

    for label, path in targets:
        w, err = _rows(path)
        if err:
            print(f"\n=== {label}: {err}")
            continue
        record[label] = _report(label, w)
        seen += 1

    out = _REPO / "bench" / "results" / "2026-08-21_h3_token_embeddings.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\naudited {seen} encoder(s); wrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
