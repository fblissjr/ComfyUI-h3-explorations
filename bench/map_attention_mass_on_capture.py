#!/usr/bin/env python3
"""Where does exact attention mass go, per head? Segment masses, per-latent-frame mass and the heaviest keys, on a capture.

    H3_CAPTURE_ROOT=... python bench/map_attention_mass_on_capture.py <capture-name> \\
        [--cells b49_s15 ...] [--heads N] [--threads 4] --out bench/results/<date>_attention_mass_<name>.json

Reads `qkv_*.pt` cells under `$H3_CAPTURE_ROOT/<capture-name>/` and that
directory's `manifest.json` (token accounting). CPU only; the capture is read,
never written.

## Why

Every per-head number this pack has recorded used a head prefix of 8
(`internal/2026-09-19_question_review.md` C.4), and none says WHERE attention
mass goes by segment or by latent frame. Three questions need it
(`docs/research/2026-09-19_in_context_reference_and_sinks.md`, item 1):

- C.4: are some heads long-range and the rest local? Per head, the share of a
  query's exact mass that lands outside Sol's forced-exact ranges (the
  conditioning prefix and the query's own 64-block band) is the mass Sol has
  to route or pool.
- C.5: the audio rows. Per head, where audio queries put their mass, and how
  much mass video queries put on audio keys.
- Attention sinks: is there a latent frame that draws a disproportionate share
  of every query's mass? Two hypotheses, tested by the same per-frame curve: a
  first-frame sink (reported for Mochi and Wan), and a periodic one on every
  fifth latent frame, whose RoPE time span is one pixel frame against four for
  the rest (`comfy/ldm/minimax/model.py`, `FRAME_PER_TOKEN`). A sink inside the
  video span is outside Sol's forced prefix, which is a node-level question.

## What it computes

For a fixed, seeded, stratified sample of query rows (text, target audio, and
an equal number per latent frame of video), per head, the exact fp32 softmax
row over every key, and from it:

- `segment_mass`: mean mass on text / audio / video keys, per query segment;
- `forced_prefix_mass`: mean mass on the keys Sol always attends exactly as
  the conditioning prefix (whole 64-blocks up to the first block boundary at
  or after the video start, `sol_attn_h3.py`, `_sink_blocks`);
- `diagonal_band_mass`: mean mass on the query's own 64-block and its two
  neighbours, which Sol also always attends exactly;
- `frame_mass`: mean mass on each latent frame's keys, per query segment;
- `top_keys`: the keys with the most mass summed over all sampled queries,
  with segment, latent frame, whether inside the forced prefix, and the value
  norm;
- `sink_summary`: frame 0's mass against the mean over frames; the mean over
  frames whose index is a multiple of 5 against the others; and, as that
  hypothesis's control, the mean for each residue of the frame index mod 5
  against the overall mean, for video queries. Ratios only; what they mean is
  for the record to say.

Limits: sampled query rows, not all; one capture's scene, blocks and steps;
text-to-video layouts only (a capture with reference rows is refused until the
segment order for references is handled). No GPU.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import orjson
import torch

REPO = Path(__file__).resolve().parent.parent
BLOCK = 64                 # Sol's block size (kitchen's sol_attn); the forced ranges are in its units
SAMPLE_TEXT = 256          # query rows sampled per segment (all of them when the segment is smaller)
SAMPLE_AUDIO = 256
SAMPLE_PER_FRAME = 16      # video query rows per latent frame
TOP_KEYS = 32
CHUNK = 512                # query rows per softmax chunk; bounds the fp32 score slab
SEED = 20260919
CELL = re.compile(r"qkv_L(\d+)_S(\d+)_b(\d+)_s(\d+)(?:_(\w+))?\.pt$")


def capture_dir(name: str) -> Path:
    root = os.environ.get("H3_CAPTURE_ROOT")
    if not root:
        raise SystemExit("refuse: set H3_CAPTURE_ROOT (the launcher exports it; captures live outside the repo)")
    path = Path(os.path.expanduser(root)) / name
    if not path.is_dir():
        raise SystemExit(f"refuse: no capture named {name!r} under H3_CAPTURE_ROOT")
    return path


def layout(manifest: dict) -> dict:
    acc = manifest["token_accounting"]
    if acc.get("reference_tokens"):
        raise SystemExit("refuse: this capture has reference rows; their segment order is not handled yet")
    text, audio, video, total = (acc["text_tokens"], acc["audio_tokens"], acc["video_tokens"],
                                 acc["total_sequence_length"])
    if text + audio + video != total:
        raise SystemExit(f"refuse: segments {text}+{audio}+{video} do not sum to {total}")
    frames = manifest["workload"]["canvas"]["latent_frames"]
    if video % frames:
        raise SystemExit(f"refuse: {video} video tokens do not divide into {frames} latent frames")
    video_start = text + audio
    prefix_end = math.ceil(video_start / BLOCK) * BLOCK
    return {"text": (0, text), "audio": (text, video_start), "video": (video_start, total),
            "frames": frames, "frame_tokens": video // frames, "total": total,
            "forced_prefix_end": prefix_end}


def sample_rows(lay: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)

    def pick(lo, hi, n):
        idx = np.arange(lo, hi)
        return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))

    v0, ft = lay["video"][0], lay["frame_tokens"]
    video = np.concatenate([pick(v0 + f * ft, v0 + (f + 1) * ft, SAMPLE_PER_FRAME) for f in range(lay["frames"])])
    return {"text": pick(*lay["text"], SAMPLE_TEXT), "audio": pick(*lay["audio"], SAMPLE_AUDIO), "video": video}


def segment_of(i: int, lay: dict) -> str:
    return "text" if i < lay["text"][1] else "audio" if i < lay["audio"][1] else "video"


def frame_of(i: int, lay: dict) -> int | None:
    v0 = lay["video"][0]
    return None if i < v0 else (i - v0) // lay["frame_tokens"]


def head_pass(q_h: torch.Tensor, k_h: torch.Tensor, v_h: torch.Tensor, rows: dict, lay: dict) -> dict:
    scale = q_h.shape[-1] ** -0.5
    n = lay["total"]
    (t0, t1), (a0, a1), (v0, v1) = lay["text"], lay["audio"], lay["video"]
    frames, ft = lay["frames"], lay["frame_tokens"]
    key_in = torch.zeros(n, dtype=torch.float64)
    out = {}
    for seg, idx in rows.items():
        seg_sum = torch.zeros(3, dtype=torch.float64)
        prefix_sum = 0.0
        band_sum = 0.0
        frame_sum = torch.zeros(frames, dtype=torch.float64)
        for c in range(0, len(idx), CHUNK):
            r = torch.from_numpy(idx[c:c + CHUNK])
            p = torch.softmax((q_h[r] @ k_h.T) * scale, dim=-1)            # [m, n] fp32
            seg_sum += torch.stack((p[:, t0:t1].sum(-1), p[:, a0:a1].sum(-1),
                                    p[:, v0:v1].sum(-1))).sum(-1).double()
            prefix_sum += float(p[:, :lay["forced_prefix_end"]].sum())
            cs = torch.nn.functional.pad(p.cumsum(-1), (1, 0))               # cs[:, j] = mass on keys < j
            blk = torch.div(r, BLOCK, rounding_mode="floor")
            lo = ((blk - 1).clamp(min=0) * BLOCK).clamp(max=n)
            hi = ((blk + 2) * BLOCK).clamp(max=n)
            rows_ix = torch.arange(len(r))
            band_sum += float((cs[rows_ix, hi] - cs[rows_ix, lo]).sum())
            frame_sum += p[:, v0:v1].reshape(len(r), frames, ft).sum(-1).sum(0).double()
            key_in += p.sum(0).double()
        m = len(idx)
        out[seg] = {"n_queries": m,
                    "segment_mass": {name: float(val) for name, val in zip(("text", "audio", "video"), seg_sum / m)},
                    "forced_prefix_mass": prefix_sum / m,
                    "diagonal_band_mass": band_sum / m,
                    "frame_mass": (frame_sum / m).tolist()}
    top = torch.topk(key_in, TOP_KEYS)
    out["top_keys"] = [{"index": int(i), "segment": segment_of(int(i), lay), "frame": frame_of(int(i), lay),
                        "in_forced_prefix": int(i) < lay["forced_prefix_end"],
                        "incoming_mass": float(w), "v_norm": float(v_h[int(i)].norm())}
                       for w, i in zip(top.values, top.indices)]
    fm = np.asarray(out["video"]["frame_mass"])
    mult5 = np.arange(frames) % 5 == 0
    # The control for the period-5 hypothesis: the same ratio for every residue
    # class. A real effect of the one-pixel-frame latents lifts residue 0 alone;
    # an artifact of binning or of the scene lifts any of them.
    residue = [float(fm[np.arange(frames) % 5 == j].mean() / fm.mean()) for j in range(5)]
    out["sink_summary"] = {"frame0_over_mean": float(fm[0] / fm.mean()),
                           "mult5_over_rest": float(fm[mult5].mean() / fm[~mult5].mean()),
                           "residue_over_mean": residue,
                           "max_frame": int(fm.argmax()), "max_over_mean": float(fm.max() / fm.mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--cells", nargs="*", help="e.g. b49_s15; default every qkv cell")
    ap.add_argument("--heads", type=int, default=None, help="head PREFIX, for a smoke test; default all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cdir = capture_dir(args.capture)
    manifest = orjson.loads((cdir / "manifest.json").read_bytes())
    lay = layout(manifest)
    rows = sample_rows(lay)

    cells = []
    for f in sorted(cdir.glob("qkv_*.pt")):
        m = CELL.match(f.name)
        if not m:
            continue
        tag = f"b{m.group(3)}_s{m.group(4)}"
        if args.cells and tag not in args.cells:
            continue
        cells.append((int(m.group(3)), int(m.group(4)), tag, f))
    if not cells:
        raise SystemExit("refuse: no matching qkv cells")

    record = {"tool": "bench/map_attention_mass_on_capture.py", "capture": args.capture,
              "layout": {k: v for k, v in lay.items()},
              "sampling": {"text": len(rows["text"]), "audio": len(rows["audio"]),
                           "video_per_frame": SAMPLE_PER_FRAME, "seed": SEED},
              "block": BLOCK, "cells": []}
    for block, step, tag, f in sorted(cells):
        data = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        q, k, v = data["q"], data["k"], data["v"]
        if q.shape[2] != lay["total"]:
            raise SystemExit(f"refuse: {f.name} has {q.shape[2]} rows, the manifest says {lay['total']}")
        n_heads = q.shape[1] if args.heads is None else min(args.heads, q.shape[1])
        heads = []
        for h in range(n_heads):
            res = head_pass(q[0, h].float(), k[0, h].float(), v[0, h].float(), rows, lay)
            res["head"] = h
            heads.append(res)
            print(f"{tag} head {h:2d}: prefix(v) {res['video']['forced_prefix_mass']:.3f} "
                  f"band(v) {res['video']['diagonal_band_mass']:.3f} "
                  f"f0/mean {res['sink_summary']['frame0_over_mean']:.2f} "
                  f"m5/rest {res['sink_summary']['mult5_over_rest']:.2f} residues "
                  f"{' '.join(f'{x:.2f}' for x in res['sink_summary']['residue_over_mean'])}", flush=True)
        record["cells"].append({"block": block, "step": step, "file": f.name, "heads": heads})
        del data, q, k, v
        args.out.write_bytes(orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY))   # partial runs stay usable
    return 0


if __name__ == "__main__":
    sys.exit(main())
