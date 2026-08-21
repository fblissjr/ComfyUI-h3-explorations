#!/usr/bin/env python3
"""What the release's tokenizer emits for the seven H3 markers, against ComfyUI's.

Run it with the ComfyUI venv python, not a repo-local one -- it needs
`transformers` (`docs/comfy_notes.md` has the invocation).

Not a check -- a measurement, and the evidence behind
`docs/research/official_weights_metadata.md`'s finding 1. It reports; it never
exits non-zero on a divergence, because the divergence is the finding.

Reads the release tokenizer from `coderef/MiniMax-H3/tokenizer/`, which is the
repo-relative copy of what ships beside the weights -- all four files md5
identical to the copy on the share, checked 2026-08-21. ComfyUI's side is the
bundled directory its H3 tokenizer actually resolves to
(`comfy/text_encoders/qwen3vl.py:149`), not a Qwen3-VL download: pointing this
at the hub would compare the wrong thing and agree.

**The control is the third block.** Any comparison of two tokenizers finds
differences somewhere; what makes this one a finding is that the labels the
conditioner emits on every reference render -- `<Picture 1>: `, `<Video 1>: `,
`<Audio 1>: `, `<T.T seconds>` -- are ID-identical, so the seven markers are
the whole of it. If that block ever goes non-identical, the story in the doc
changes and so does what needs fixing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

RELEASE = _REPO / "coderef" / "MiniMax-H3" / "tokenizer"
COMFY = Path.home() / "ComfyUI" / "comfy" / "text_encoders" / "qwen25_tokenizer"

# The release's `additional_special_tokens` minus ComfyUI's. Hardcoded rather
# than diffed at run time on purpose: this list is what the doc claims, so the
# script has to be able to disagree with it.
H3_MARKERS = ["<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>",
              "<|caption_start|>", "<|caption_end|>"]

# Everything the conditioner emits around a reference. These must stay
# identical or the finding is much larger than seven markers.
CONTROL = ["<Picture 1>: ", "<Video 1>: ", "<Audio 1>: ", "<0.2 seconds>",
           "<5.2 seconds>", "A medium shot establishes the room.",
           "subject_definitions:", "retention_analysis:"]

# The line `workflows/build_workflows.py:1718` writes into the voice graph.
SHIPPED = "<d>[English] I thought you would have gone by now."


def main() -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not importable; run this with the ComfyUI venv "
              "python (see docs/comfy_notes.md)")
        return 2
    if not RELEASE.exists():
        print(f"{RELEASE.relative_to(_REPO)} is absent -- coderef/ is "
              f"gitignored and this needs the MiniMax-H3 clone")
        return 2

    rel = AutoTokenizer.from_pretrained(str(RELEASE))
    cfy = AutoTokenizer.from_pretrained(str(COMFY))

    def ids(tk, s):
        return tk(s, add_special_tokens=False)["input_ids"]

    record = {"release_len": len(rel), "comfy_len": len(cfy),
              "markers": {}, "control": {}, "shipped_line": {}}

    print(f"tokenizer length   release {len(rel):,}   comfy {len(cfy):,}\n")

    print("the seven H3 markers:")
    for m in H3_MARKERS:
        a, b = ids(rel, m), ids(cfy, m)
        record["markers"][m] = {"release": a, "comfy": b, "same": a == b}
        flag = "SAME" if a == b else "split"
        print(f"  {m:<18} release {str(a):<12} comfy {len(b)} pieces {b}  {flag}")

    print("\ncontrol -- what the conditioner emits on every render:")
    all_same = True
    for s in CONTROL:
        a, b = ids(rel, s), ids(cfy, s)
        record["control"][s] = {"same": a == b, "n": len(a)}
        all_same &= a == b
        print(f"  identical={a == b}  n={len(a):<3} {s!r}")
    print("  -> control block " + ("holds" if all_same else
          "BROKEN, the finding is larger than seven markers"))

    # `<d>` closes the sentence, so the trailing `.` is part of what gets
    # shredded. Printing the decoded pieces is the point: the split does not
    # respect the marker boundary, which the ID counts alone do not show.
    line = SHIPPED + "</d>"
    print(f"\nthe line the voice graph ships:\n  {line}")
    for name, tk in (("release", rel), ("comfy", cfy)):
        i = ids(tk, line)
        record["shipped_line"][name] = {"ids": i,
                                        "pieces": [tk.decode([t]) for t in i]}
        print(f"  {name:<8} {len(i)} ids  {[tk.decode([t]) for t in i]}")

    out = _REPO / "bench" / "results" / "2026-08-21_h3_tokenizer_markers.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
