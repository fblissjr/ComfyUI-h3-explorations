#!/usr/bin/env python3
"""Does an audio reference degrade what the H3 encoder produces?

The report this exists for: a user passed a reference with audio through the
W4A16 AWQ encoder and got video that ignored the prompt, where ComfyUI's INT8
encoder on the same workflow did not.  Three surfaces were cleared by reading
first -- the native tokenizer keeps the prompt intact with audio present, our
`preprocess_embed` override returns exactly what the base class returns for a
non-image embed, and our ref_items construction is character-identical to
core's -- which leaves the weights.

**This asks an ABSOLUTE question, deliberately.**  The obvious comparison,
"how far do the rows move when the audio marker is added", has no meaning
without a reference encoder: the marker is real text, it legitimately shifts
every downstream row, and there is no baseline here that says how much is too
much.  So this measures only properties that are wrong on their own terms --
non-finite values, collapsed rows, a dead vision span -- which need no
reference and are decisive when they trip.  A clean result does not clear the
encoder; it escalates the question to a BF16 comparison, and says so.

The audio arm and the no-audio arm carry the SAME images and the SAME prompt,
so any difference is the audio item and nothing else.  Audio never enters Qwen
(`comfy/text_encoders/minimax.py`, "audio never enters Qwen"); it becomes the
text "<Audio 1>: ".  That is the whole mechanism under test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent.parent))          # ComfyUI
sys.path.insert(0, str(REPO))                        # this pack, for h3_awq_encoder
sys.path.insert(0, str(REPO / "workflows"))

PROMPT = (
    "detailed_description:\n"
    "[Shot 1] A red fox walks across a snowy field at dawn while the camera "
    "trucks right at slow speed with small amplitude.\n"
)


def _image(seed: int, h: int = 448, w: int = 448):
    """A deterministic coordinate pattern. No random or dummy media."""
    import torch
    ys = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    base = (ys * (seed + 1) + xs * (seed + 2)) % 1.0
    return torch.stack([base, 1.0 - base, (base * 0.5) % 1.0], dim=-1)[None]


def _audio(seconds: float = 4.0, rate: int = 48000):
    import torch
    t = torch.linspace(0, seconds, int(seconds * rate))
    wave = torch.sin(2 * 3.14159 * 220.0 * t) * 0.2
    return {"waveform": torch.stack([wave, wave])[None], "sample_rate": rate}


def health(tensor) -> dict:
    """Properties that are wrong on their own terms, with no reference needed."""
    import torch
    x = tensor.detach().to(torch.float32).reshape(-1, tensor.shape[-1])
    norms = x.norm(dim=-1)
    finite = bool(torch.isfinite(x).all())
    # A collapsed encoding shows up as rows that are all nearly the same
    # direction; mean pairwise cosine against the row mean catches it without
    # an O(n^2) pass.
    mean_dir = torch.nn.functional.normalize(x.mean(0, keepdim=True), dim=-1)
    align = torch.nn.functional.normalize(x, dim=-1) @ mean_dir.T
    return {
        "rows": int(x.shape[0]),
        "width": int(x.shape[-1]),
        "all_finite": finite,
        "nan_count": int(torch.isnan(x).sum()),
        "inf_count": int(torch.isinf(x).sum()),
        "row_norm": {
            "min": float(norms.min()), "median": float(norms.median()),
            "max": float(norms.max()), "mean": float(norms.mean()),
        },
        "near_zero_rows": int((norms < 1e-3).sum()),
        "mean_alignment_to_centroid": float(align.mean()),
        "max_alignment_to_centroid": float(align.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, help="filename under models/text_encoders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="default")
    args = ap.parse_args(argv)

    import torch  # noqa: F401
    import folder_paths
    import h3_awq_encoder as H

    path = folder_paths.get_full_path_or_raise("text_encoders", args.encoder)
    clip = H._load_clip(path, folder_paths.get_folder_paths("embeddings"),
                        device=args.device)
    contract = getattr(
        clip.cond_stage_model.qwen3vl_32b.transformer, "_h3_encoder_contract", None)

    img_a, img_b, aud = _image(0), _image(3), _audio()
    arms = {
        "images_only": [{"type": "image", "data": img_a},
                        {"type": "image", "data": img_b}],
        "images_with_audio": [{"type": "image", "data": img_a},
                              {"type": "audio"},
                              {"type": "image", "data": img_b}],
        "audio_only": [{"type": "audio"}],
        "no_refs": None,
    }

    out = {
        "purpose": "absolute health of H3 layer-50 output with and without an "
                   "audio reference; escalates rather than clears",
        "encoder": args.encoder,
        "encoder_contract": contract,
        "prompt_sha256": __import__("hashlib").sha256(PROMPT.encode()).hexdigest(),
        "arms": {},
    }
    for name, items in arms.items():
        tokens = clip.tokenize(PROMPT, minimax_ref_items=items)
        cond = clip.encode_from_tokens(tokens)
        if isinstance(cond, (tuple, list)):
            cond = cond[0]
        out["arms"][name] = health(cond)
        print(f"  {name:20s} {json.dumps(out['arms'][name])}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
