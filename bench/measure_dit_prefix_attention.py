#!/usr/bin/env python3
"""What the H3 DiT reads from the encoder prefix: attention mass, by key class.

The DiT attends over ONE packed sequence containing the Qwen3-VL prefix and the
latent streams together. Nothing in this repo has measured how much of a latent
query's attention actually lands on that prefix, so every statement about "the
prompt is being read" has been an inference from the architecture. This measures
it on a capture, per block and per sampler step, and writes a per-prefix-position
importance vector that a later loss can weight by.

## The packed order, read from source, not assumed

`comfy/ldm/minimax/model.py::PackedLayout` builds the segment table:

  model.py:325      `segments = [("text", text_len)]` -- the encoder prefix is
                    ALWAYS the first segment, starting at row 0.
  model.py:340-361  fl2va keyframe rows (`cond`, `cond_audio`) come next.
  model.py:363-406  ref2va reference blocks (`ref_img`, `ref_audio`) come next,
                    in request order.
  model.py:408-416  "target audio then target video, always the last two
                    segments" -- `audio` then `video`. The latent VIDEO tokens
                    are therefore the trailing span, and they are the queries
                    this script samples.
  model.py:565      `text_len = context.shape[1]`, i.e. the prefix length is the
                    encoder's own output length, including its vision positions.

Per-position modality tags inside the prefix come from
`payload["text_token_tags"]` (model.py:624), which is `minimax_token_tags` built
by `comfy/text_encoders/minimax.py::token_tags_from_embeds_info` (minimax.py:75):
text positions carry 1, and a whole vision block carries 0 **including its
flanking `<|vision_start|>` / `<|vision_end|>`**, because that function widens
each embed span by one on each side.

**There is no attention mask.** `model.py::Attention.forward` calls
`optimized_attention(q, k, v, self.heads, mask=None, ...)` (model.py:195), so the softmax
denominator runs over every one of the S keys and the class masses of a single
query sum to exactly 1. That is what makes "fraction of attention mass" a
well-posed quantity here, and control 1 in `bench/check_dit_prefix_attention.py`
asserts the partition is total rather than assuming it.

## What the capture does and does not carry

`h3_capture.py` writes only `{"q","k","v"}` per block and step. It records no
segment boundaries and no token tags. So the layout is RECONSTRUCTED, by two
independent routes that must agree:

  geometric  `comfy_extras.nodes_minimax_h3::temporal_shape` and `PackedLayout`
             give every non-text segment's row count from canvas and length; the
             prefix length is the residual `S - non_text_rows`.
  tokenised  the manifest's prompt through ComfyUI's own `MiniMaxH3Tokenizer`
             gives the prefix length directly, for a text-only prefix.

Either route alone would accept an off-by-one silently. Together they cannot:
the residual moves with the canvas and the token count moves with the prompt, so
one being wrong by a row disagrees with the other. When they disagree this
refuses to report. When the prefix contains vision blocks the tokenised route
cannot be completed without the vision encoder, and the record says so as
`UNKNOWN` rather than reporting a cross-check that did not happen.

**The tag-derived check is not available at all from a capture.** Nothing in the
`.pt` files distinguishes a tag-1 row from a tag-0 row, so the record carries
`prefix_tag_source: "UNKNOWN (not derivable from the capture)"` unless tags are
supplied with `--token-tags`.

## Classes

Non-prefix classes are the `PackedLayout` segment kinds themselves, so they
cannot drift from the model. Prefix classes are derived from the token ids and
from the presentation forms in `comfy/text_encoders/minimax.py`:

  prefix_marker_token  a position whose id is one of the special tokens the
                       release declares (`vendor_config.additional_special_tokens`),
                       resolved through the LOADED tokenizer's vocabulary rather
                       than through any constant's name -- CLAUDE.md's rule about
                       branching on the observable. `<|vision_start|>` and
                       `<|vision_end|>` are excluded here because the tag rule
                       above already counts them as vision.
  prefix_marker_span   positions strictly inside a `<d>...</d>` (or `_start` /
                       `_end`) pair.
  prefix_label         the label forms `add_text` emits ahead of a vision block:
                       `<Picture i>: `, `<Audio j>: `, `<Video k>: `,
                       `<T.T seconds>` (minimax.py:169-194).
  prefix_section_key   a `name:` section header, names imported from
                       `bench/preflight_graph.py`'s two guide section lists.
  prefix_vision        tag-0 positions, only when tags are supplied.
  prefix_text_other    everything else in the prefix.

`--span-regex NAME=PATTERN` adds an operator-defined class. Those are recorded in
the output as operator-supplied, because nothing in the release defines them.

## Cost

CPU only by default. Query rows are sampled and scored against ALL keys, which
is exact for those rows -- the same argument `bench/grade_sage_on_capture.py`
makes, and its `sample_rows` is imported rather than restated. Scores are fp32
by default with the softmax and every accumulation in float64;
`--score-dtype float64` runs the whole thing in float64 and the run records the
agreement between the two on one head.

    python bench/measure_dit_prefix_attention.py <capture_dir> \
        --json bench/results/<date>_dit_prefix_attention_<task>.json

Needs `PYTHONPATH` reaching ComfyUI, or nothing at all -- it inserts the roots
itself, ComfyUI ahead of this repo (the `import nodes` trap in
`docs/comfy_notes.md`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
_COMFY = _HERE.parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))
sys.path.insert(0, str(_COMFY))          # ahead of the repo: `import nodes`

import torch  # noqa: E402

# THIS IMPORT MUST COME FIRST, and the reason is not style. `comfy_extras.
# nodes_minimax_h3` does a bare `import nodes`, and this repo has a `nodes.py`
# that dies on a relative import when found first -- the trap in
# `docs/comfy_notes.md`. The sys.path block above puts ComfyUI ahead, but every
# sibling bench module below (`preflight_graph`, `grade_sage_on_capture`)
# re-inserts the repo root at position 0 when IT is imported, which puts the
# repo back in front. So resolve `nodes` while ComfyUI is still leading.
from comfy_extras.nodes_minimax_h3 import temporal_shape  # noqa: E402

import vendor_config  # noqa: E402
from count_packed_rows import _layout, _ref_blocks  # noqa: E402
from grade_sage_on_capture import sample_rows  # noqa: E402
from h3_producer_provenance import producer_provenance  # noqa: E402
from preflight_graph import BASE_SECTIONS, REF_SECTIONS  # noqa: E402

# The label forms `MiniMaxH3Tokenizer.tokenize_with_weights` emits with
# `add_text` ahead of each vision block (comfy/text_encoders/minimax.py:169-194).
# Written as patterns over the reconstructed presentation text because the
# tokeniser splits them across several ids ("<P","icture"," ","1",">:"," ").
LABEL_PATTERNS = (
    r"<Picture \d+>: ",
    r"<Audio \d+>: ",
    r"<Video \d+>: ",
    r"<\d+\.\d seconds>",
)
# Vision-block delimiters. Excluded from the marker class because
# `token_tags_from_embeds_info` (minimax.py:75) already tags them as vision.
VISION_DELIMITERS = ("<|vision_start|>", "<|vision_end|>")

CAPTURE_NAME = re.compile(r"^qkv_L(\d+)_S(\d+)_b(\d+)_s(\d+)(?:_r(\d+))?\.pt$")


def _repo_relative(path: Path) -> str:
    """Repo-relative when it is inside the repo, otherwise the bare name.

    Nothing this writes may carry a machine-owner home path;
    `bench/check_no_owner_paths.py` is the gate and the capture share is
    outside the tree, so an absolute path here would be a leak, not a detail.
    """
    path = Path(path).resolve()
    return str(path.relative_to(_REPO)) if path.is_relative_to(_REPO) else path.name


# --------------------------------------------------------------------------
# tokenizer-side derivation


def special_token_ids(tokenizer) -> dict[str, int]:
    """The release's declared special tokens, resolved in the LOADED vocabulary.

    Keyed on the token STRINGS the release declares, never on the name of the
    constant that happens to hold them in one implementation. CLAUDE.md adopted
    that rule on 2026-08-22 after two harnesses branched on a constant's name and
    silently read the wrong tokenizer when upstream renamed it. The strings are
    the observable; a rename cannot move them.
    """
    vocab = tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    out = {}
    for name in vendor_config.additional_special_tokens():
        tid = vocab.get(name)
        if tid is not None:
            out[name] = int(tid)
    if not out:
        raise SystemExit(
            "FAIL: none of the release's declared special tokens resolve in the "
            "loaded tokenizer's vocabulary. Refusing to class markers by id."
        )
    return out


def tokenize_prefix(prompt: str, tokenizer):
    """(ids, pieces) for a TEXT-ONLY presentation, through ComfyUI's own path.

    Raises when the presentation contains a vision entry, because the prefix
    length then depends on the vision encoder and cannot be had statically.
    """
    entries = tokenizer.tokenize_with_weights(prompt)["qwen3vl_32b"][0]
    ids = []
    for tok, *_ in entries:
        if not isinstance(tok, int):
            raise ValueError("presentation contains a vision entry")
        ids.append(int(tok))
    hf = tokenizer.qwen3vl_32b.tokenizer
    pieces = [hf.decode([i]) for i in ids]
    return ids, pieces


def _char_spans(pieces: list[str]) -> list[tuple[int, int]]:
    spans, off = [], 0
    for p in pieces:
        spans.append((off, off + len(p)))
        off += len(p)
    return spans


def prefix_classes(ids, pieces, specials, extra_spans=(), tags=None,
                   source_text=None):
    """Per-prefix-position class name, and the evidence for each assignment.

    Precedence: vision > marker token > label > section key > operator span >
    marker span > other. Vision leads because it comes from the payload's own
    tags; everything below it is derived from the id stream.

    Every span class is placed by character offset over the tokens decoded one
    at a time and concatenated. That is exact for byte-level BPE on text whose
    multi-byte characters do not straddle a token boundary, and silently wrong
    when they do -- one replacement character shifts every offset after it. So
    when `source_text` is given the concatenation is compared against it, and a
    mismatch DISABLES the span classes rather than placing them somewhere
    plausible. The id-based marker class and the prefix length are unaffected,
    so the top-level prefix-versus-latent answer survives the degradation; the
    record says which happened.
    """
    n = len(ids)
    cls = ["prefix_text_other"] * n
    text = "".join(pieces)
    spans = _char_spans(pieces)
    faithful = source_text is None or text == source_text

    def paint(lo_char, hi_char, name, only_if=("prefix_text_other",)):
        hit = 0
        for i, (a, b) in enumerate(spans):
            if b > lo_char and a < hi_char and cls[i] in only_if:
                cls[i] = name
                hit += 1
        return hit

    evidence = {"marker_tokens": {}, "labels": [], "section_keys": [],
                "operator_spans": {}, "marker_spans": [],
                "char_span_placement": (
                    "exact" if faithful else
                    "DISABLED: decoding the tokens one at a time did not "
                    "reconstruct the prompt, so no character offset can be "
                    "trusted; only the id-based marker class was applied")}

    # marker spans first (lowest precedence), so later passes can overwrite
    marker_ids = {v: k for k, v in specials.items()
                  if k not in VISION_DELIMITERS}
    positions = {}
    for i, tid in enumerate(ids):
        if tid in marker_ids:
            positions.setdefault(marker_ids[tid], []).append(i)
    for opener, closer in _marker_pairs(specials):
        for a, b in zip(positions.get(opener, []), positions.get(closer, [])):
            if b > a + 1:
                for i in range(a + 1, b):
                    cls[i] = "prefix_marker_span"
                evidence["marker_spans"].append([opener, a, b])

    overwritable = ("prefix_text_other", "prefix_marker_span")

    for pattern in LABEL_PATTERNS if faithful else ():
        for m in re.finditer(pattern, text):
            hit = paint(m.start(), m.end(), "prefix_label", only_if=overwritable)
            if hit:
                evidence["labels"].append([m.group(0), hit])

    for key in sorted(set(BASE_SECTIONS) | set(REF_SECTIONS)) if faithful else ():
        for m in re.finditer(rf"^{re.escape(key)}:", text, re.M):
            hit = paint(m.start(), m.end(), "prefix_section_key",
                        only_if=overwritable)
            if hit:
                evidence["section_keys"].append([key, hit])

    for name, pattern in extra_spans if faithful else ():
        hits = 0
        for m in re.finditer(pattern, text, re.M):
            hits += paint(m.start(), m.end(), f"prefix_x_{name}",
                          only_if=overwritable)
        evidence["operator_spans"][name] = {"pattern": pattern, "positions": hits}

    for i, tid in enumerate(ids):
        if tid in marker_ids:
            cls[i] = "prefix_marker_token"
            evidence["marker_tokens"][marker_ids[tid]] = \
                evidence["marker_tokens"].get(marker_ids[tid], 0) + 1

    if tags is not None:
        if len(tags) != n:
            raise SystemExit(f"FAIL: --token-tags has {len(tags)} entries, "
                             f"prefix is {n}")
        for i, t in enumerate(tags):
            if int(t) == 0:
                cls[i] = "prefix_vision"

    return cls, evidence


def _marker_pairs(specials):
    """(opener, closer) names among the declared specials, by their own spelling.

    `<d>`/`</d>` and every `<|x_start|>`/`<|x_end|>` couple. Derived from the
    strings, so a release that adds another pair is covered without an edit.
    """
    names = set(specials)
    pairs = []
    for name in sorted(names):
        if name.startswith("</") and name.endswith(">"):
            opener = "<" + name[2:]
            if opener in names:
                pairs.append((opener, name))
        elif name.endswith("_start|>"):
            closer = name[: -len("_start|>")] + "_end|>"
            if closer in names:
                pairs.append((name, closer))
    return pairs


# --------------------------------------------------------------------------
# capture-side derivation


def capture_files(directory: Path):
    out = []
    for path in sorted(directory.glob("qkv_*.pt")):
        m = CAPTURE_NAME.match(path.name)
        if not m:
            continue
        out.append({"path": path, "name": path.name, "length": int(m.group(1)),
                    "seq": int(m.group(2)), "block": int(m.group(3)),
                    "step": int(m.group(4)),
                    "render": int(m.group(5)) if m.group(5) else 0})
    if not out:
        raise SystemExit(f"FAIL: no qkv_*.pt under {directory.name}")
    seqs = {f["seq"] for f in out}
    if len(seqs) != 1:
        raise SystemExit(f"FAIL: capture holds mixed sequence lengths {sorted(seqs)}")
    return out


def derive_layout(seq_len, width, height, length, refs_spec, prompt, tokenizer):
    """Segment table for the capture, plus the two-route prefix-length check.

    The residual route and the tokenised route are computed independently and
    compared. Disagreement is fatal: a prefix boundary wrong by one row silently
    reclassifies a key, and no number below would announce it.
    """
    _frames, latent_t, audio_t = temporal_shape(length)
    refs = _ref_blocks([s for s in refs_spec.split(",") if s])
    empty = _layout(0, latent_t, height // 16, width // 16, audio_t, refs)
    residual = seq_len - empty.seq_len
    if residual < 0:
        raise SystemExit(
            f"FAIL: non-text segments alone are {empty.seq_len:,} rows for "
            f"{width}x{height} x {length}, but the capture is {seq_len:,}. "
            "The canvas, the length or the reference list is wrong."
        )

    check = {"residual_prefix_len": residual, "tokenised_prefix_len": None,
             "agreement": "UNKNOWN"}
    if prompt is not None:
        try:
            ids, pieces = tokenize_prefix(prompt, tokenizer)
        except ValueError:
            check["agreement"] = ("UNKNOWN: the presentation carries vision "
                                  "entries, whose row count needs the vision "
                                  "encoder")
            ids = pieces = None
        else:
            check["tokenised_prefix_len"] = len(ids)
            if len(ids) != residual:
                raise SystemExit(
                    f"FAIL prefix_length_cross_check: the geometric residual "
                    f"says {residual:,} prefix rows and the tokenizer says "
                    f"{len(ids):,}. One of the capture's canvas, length, "
                    "reference list or recorded prompt does not belong to "
                    "these tensors. Refusing to report."
                )
            check["agreement"] = "two independent routes agree"
    else:
        ids = pieces = None
        check["agreement"] = "UNKNOWN: no prompt supplied"

    layout = _layout(residual, latent_t, height // 16, width // 16, audio_t, refs)
    if layout.seq_len != seq_len:
        raise SystemExit(f"FAIL: rebuilt layout is {layout.seq_len:,} rows, "
                         f"capture is {seq_len:,}")
    segments = [(int(a), int(b), str(kind)) for a, b, kind in layout.segments]
    return {"segments": segments, "latent_t": latent_t, "audio_t": audio_t,
            "prefix_len": residual, "cross_check": check,
            "refs": len(refs)}, ids, pieces


def class_map(segments, prefix_cls, seq_len):
    """(class names in order, LongTensor[S] of class index).

    Non-prefix classes are the segment kinds themselves, so they cannot drift
    from `PackedLayout`. A key that ended in no class is fatal, not silently
    dropped -- that is the hole control 1 exists to see.
    """
    names, index = [], {}

    def cid(name):
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    out = torch.full((seq_len,), -1, dtype=torch.long)
    for a, b, kind in segments:
        if kind == "text":
            if len(prefix_cls) != b - a:
                raise SystemExit("FAIL: prefix class vector length mismatch")
            for i, name in enumerate(prefix_cls):
                out[a + i] = cid(name)
        else:
            out[a:b] = cid(f"seg_{kind}")
    if int((out < 0).sum()) != 0:
        raise SystemExit("FAIL: the class map does not cover every key")
    return names, out


# --------------------------------------------------------------------------
# the measurement


def measure_masses(q, k, rows, class_of, n_classes, prefix_len,
                   head_chunk=4, score_dtype=torch.float32,
                   acc_dtype=torch.float64, device=None):
    """Softmax attention mass of `rows` over all keys, aggregated by class.

    Layout [B, H, S, D], as `h3_capture.py` writes it. The output values `v` are
    never touched: mass is a property of q and k alone.

    Chunked over heads for memory only; the maths is unchunked per head, so each
    sampled row's softmax runs against the complete key set and is exact for that
    row, not an approximation of it.
    """
    b, h, s, d = q.shape
    if b != 1:
        raise SystemExit("FAIL: capture is not single-batch")
    device = torch.device(device) if device is not None else class_of.device
    scale = 1.0 / math.sqrt(d)
    per_head = torch.zeros((h, n_classes), dtype=acc_dtype, device=device)
    importance = torch.zeros(prefix_len, dtype=acc_dtype, device=device)
    row_sum_lo, row_sum_hi = float("inf"), float("-inf")
    n = rows.numel()
    rows = rows.to(q.device)
    class_of = class_of.to(device)

    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        qs = q[0, h0:h1].index_select(1, rows).to(device=device, dtype=score_dtype)
        ks = k[0, h0:h1].to(device=device, dtype=score_dtype)
        scores = torch.bmm(qs, ks.transpose(1, 2)) * scale        # [hc, n, S]
        del qs, ks
        probs = torch.softmax(scores.to(acc_dtype), dim=-1)
        del scores
        m = torch.zeros((h1 - h0, n, n_classes), dtype=acc_dtype, device=device)
        m.index_add_(2, class_of, probs)
        per_head[h0:h1] = m.mean(dim=1)
        sums = m.sum(dim=-1)
        row_sum_lo = min(row_sum_lo, float(sums.min()))
        row_sum_hi = max(row_sum_hi, float(sums.max()))
        if prefix_len:
            importance += probs[:, :, :prefix_len].sum(dim=(0, 1))
        del probs, m, sums
    if prefix_len:
        importance /= float(h * n)
    per_head = per_head.cpu()
    importance = importance.cpu()
    return {"per_head": per_head, "importance": importance,
            "row_sum_min": row_sum_lo, "row_sum_max": row_sum_hi}


def score_dtype_agreement(q, k, rows, class_of, n_classes, head=0):
    """One head, fp32 scores against float64 scores. A number, not a promise."""
    a = measure_masses(q[:, head:head + 1], k[:, head:head + 1], rows, class_of,
                       n_classes, 0, head_chunk=1,
                       score_dtype=torch.float32)["per_head"][0]
    b = measure_masses(q[:, head:head + 1], k[:, head:head + 1], rows, class_of,
                       n_classes, 0, head_chunk=1,
                       score_dtype=torch.float64)["per_head"][0]
    return float((a - b).abs().max())


def summarise(per_head, names, segments):
    """Head-aggregated and per-head fractions, in the terms the question asks."""
    mean = per_head.mean(dim=0)
    total = float(mean.sum())
    by_class = {n: float(mean[i]) for i, n in enumerate(names)}

    prefix_ids = [i for i, n in enumerate(names) if n.startswith("prefix_")]
    video_ids = [i for i, n in enumerate(names) if n == "seg_video"]
    audio_ids = [i for i, n in enumerate(names) if n == "seg_audio"]
    cond_ids = [i for i, n in enumerate(names)
                if n.startswith("seg_") and n not in ("seg_video", "seg_audio")]

    def frac(ids, vec):
        return float(vec[ids].sum()) if ids else 0.0

    prefix_head = per_head[:, prefix_ids].sum(dim=1) if prefix_ids else \
        torch.zeros(per_head.shape[0], dtype=per_head.dtype)
    prefix_mass = frac(prefix_ids, mean)
    within = {n: (float(mean[i]) / prefix_mass if prefix_mass > 0 else 0.0)
              for i, n in enumerate(names) if n.startswith("prefix_")}
    return {
        "class_mass_sums_to": total,
        "mass_by_class": by_class,
        "top_level": {
            "prefix": prefix_mass,
            "latent_video": frac(video_ids, mean),
            "latent_audio": frac(audio_ids, mean),
            "conditioning_latents": frac(cond_ids, mean),
        },
        "within_prefix": within,
        "per_head_prefix_fraction": [round(float(x), 8) for x in prefix_head],
        "per_head_prefix_summary": {
            "min": float(prefix_head.min()), "max": float(prefix_head.max()),
            "mean": float(prefix_head.mean()),
            "median": float(prefix_head.median()),
        },
    }


def top_positions(importance, pieces, k=32):
    if pieces is None or importance.numel() == 0:
        return []
    order = torch.argsort(importance, descending=True)[:k]
    return [{"position": int(i), "mass": float(importance[i]),
             "token": pieces[int(i)]} for i in order]


# --------------------------------------------------------------------------


def _manifest(directory: Path):
    path = directory / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _graph(directory: Path):
    path = directory / "workflow_api.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def manifest_agreement(manifest, layout, seq_len, width, height, length):
    """Whether the manifest beside the tensors describes THESE tensors.

    It is not a formality. On the 2026-08-20 T2VA capture the manifest's
    `workload.canvas` and `token_accounting` describe a different render from
    the tensors in the same directory -- a different canvas, a different text
    length and a total 18,282 rows short of the S in the filenames -- while its
    `prompt` and `captured_tensors` are correct. A reader who priced that
    capture from its manifest would have priced something else, so the
    disagreement travels inside the record rather than being fixed silently
    here. This script reads the canvas from `workflow_api.json` for that reason.
    """
    if not manifest:
        return {"status": "UNKNOWN: no manifest.json beside the tensors"}
    out = {"status": "agrees", "disagreements": []}
    canvas = (manifest.get("workload") or {}).get("canvas") or {}
    for key, got in (("width", width), ("height", height), ("length", length)):
        if key in canvas and int(canvas[key]) != got:
            out["disagreements"].append(
                f"workload.canvas.{key} says {canvas[key]}, the graph and the "
                f"tensors say {got}")
    acc = manifest.get("token_accounting") or {}
    derived = {"total_sequence_length": seq_len,
               "text_tokens": layout["prefix_len"]}
    for a_, b_, kind in layout["segments"]:
        if kind == "video":
            derived["video_tokens"] = b_ - a_
        elif kind == "audio":
            derived["audio_tokens"] = b_ - a_
    for key, got in derived.items():
        if key in acc and int(acc[key]) != got:
            out["disagreements"].append(
                f"token_accounting.{key} says {acc[key]:,}, the layout says "
                f"{got:,}")
    if out["disagreements"]:
        out["status"] = ("DISAGREES: the manifest does not describe these "
                         "tensors; the layout above comes from the graph and "
                         "the filenames, not from it")
    return out


def _canvas_from_graph(graph):
    """(width, height, length) from a capture's own graph, when it has one.

    Preferred over the manifest, and that is not a style choice: on the
    2026-08-20 T2VA capture the manifest's `workload` block describes a
    DIFFERENT render from the tensors beside it (see the run record). The graph
    is the thing that was queued.
    """
    if not graph:
        return None
    for node in graph.values():
        if node.get("class_type") == "MiniMaxH3Resolution":
            ins = node.get("inputs", {})
            label = ins.get("shape.%s_resolution" % ins.get("shape", ""))
            if isinstance(label, str):
                m = re.match(r"\s*(\d+)x(\d+)", label)
                if m and isinstance(ins.get("length"), int):
                    return int(m.group(1)), int(m.group(2)), int(ins["length"])
    for node in graph.values():
        ins = node.get("inputs", {})
        if all(isinstance(ins.get(key), int) for key in ("width", "height", "length")):
            return int(ins["width"]), int(ins["height"]), int(ins["length"])
    return None


def _prompt_from_graph(graph):
    if not graph:
        return None
    for node in graph.values():
        p = node.get("inputs", {}).get("prompt")
        if isinstance(p, str) and p:
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--rows", type=int, default=256,
                    help="sampled latent-video query rows (default matches "
                         "bench/grade_sage_on_capture.py)")
    ap.add_argument("--strata", type=int, default=8)
    ap.add_argument("--head-chunk", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score-dtype", default="float32",
                    choices=("float32", "float64"))
    ap.add_argument("--prompt", default=None,
                    help="override the prompt recorded in the capture")
    ap.add_argument("--canvas", default=None, help="WIDTHxHEIGHT override")
    ap.add_argument("--length", type=int, default=None, help="frame count override")
    ap.add_argument("--refs", default="", help="comma-separated WxH references")
    ap.add_argument("--token-tags", type=Path, default=None,
                    help="JSON list of per-prefix-position tags (1 text, 0 "
                         "vision); without it the vision class is UNKNOWN")
    ap.add_argument("--span-regex", action="append", default=[],
                    metavar="NAME=PATTERN",
                    help="operator-defined prefix class, recorded as such")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--safetensors", type=Path, default=None,
                    help="default: the --json path with _importance.safetensors")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    score_dtype = getattr(torch, a.score_dtype)

    extra_spans = []
    for spec in a.span_regex:
        name, _, pattern = spec.partition("=")
        if not name or not pattern:
            raise SystemExit(f"FAIL: --span-regex wants NAME=PATTERN, got {spec!r}")
        extra_spans.append((name, pattern))

    directory = a.capture_dir.resolve()
    files = capture_files(directory)
    manifest = _manifest(directory)
    graph = _graph(directory)

    canvas = _canvas_from_graph(graph)
    if a.canvas and a.length:
        w, h = (int(v) for v in a.canvas.lower().split("x"))
        canvas = (w, h, a.length)
        canvas_source = "command line"
    elif canvas:
        canvas_source = "workflow_api.json beside the tensors"
    elif manifest:
        wl = (manifest.get("workload") or {}).get("canvas") or {}
        if not all(k in wl for k in ("width", "height", "length")):
            raise SystemExit("FAIL: no canvas in the graph or the manifest; "
                            "pass --canvas and --length")
        canvas = (int(wl["width"]), int(wl["height"]), int(wl["length"]))
        canvas_source = "manifest.json workload.canvas"
    else:
        raise SystemExit("FAIL: no canvas available; pass --canvas and --length")

    prompt = a.prompt or _prompt_from_graph(graph)
    prompt_source = "command line" if a.prompt else (
        "workflow_api.json" if prompt else None)
    if prompt is None and manifest:
        prompt = (manifest.get("prompt") or {}).get("full_prompt_text")
        prompt_source = "manifest.json prompt.full_prompt_text" if prompt else None

    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tokenizer = MiniMaxH3Tokenizer()
    specials = special_token_ids(tokenizer)

    seq_len = files[0]["seq"]
    width, height, length = canvas
    layout, ids, pieces = derive_layout(seq_len, width, height, length,
                                        a.refs, prompt, tokenizer)

    tags = None
    tag_source = "UNKNOWN (not derivable from the capture)"
    if a.token_tags:
        tags = json.loads(a.token_tags.read_text())
        tag_source = f"--token-tags {a.token_tags.name}"

    if ids is None:
        prefix_cls = ["prefix_text_other"] * layout["prefix_len"]
        evidence = {"note": "UNKNOWN: prefix token ids unavailable, so every "
                            "prefix position is unclassified"}
    else:
        prefix_cls, evidence = prefix_classes(ids, pieces, specials,
                                              extra_spans, tags,
                                              source_text=prompt)

    names, class_of = class_map(layout["segments"], prefix_cls, seq_len)
    class_of = class_of.to(device)
    counts = {n: int((class_of == i).sum()) for i, n in enumerate(names)}

    video_seg = next((s for s in layout["segments"] if s[2] == "video"), None)
    if video_seg is None:
        raise SystemExit("FAIL: no video segment in the layout")
    v_lo, v_hi, _ = video_seg
    rows = (sample_rows(v_hi - v_lo, a.rows, strata=a.strata,
                        device=str(device)) + v_lo)

    records, tensors = [], {}
    for f in files:
        t0 = time.time()
        blob = torch.load(f["path"], map_location="cpu", weights_only=True,
                          mmap=True)
        q, k = blob["q"], blob["k"]
        if q.shape[2] != seq_len:
            raise SystemExit(f"FAIL: {f['name']} is S={q.shape[2]}, expected {seq_len}")
        out = measure_masses(q, k, rows, class_of, len(names),
                             layout["prefix_len"], head_chunk=a.head_chunk,
                             score_dtype=score_dtype)
        agreement = score_dtype_agreement(q, k, rows[:32], class_of, len(names))
        rec = {"file": f["name"], "block": f["block"], "step": f["step"],
               "render": f["render"], "heads": int(q.shape[1]),
               "head_dim": int(q.shape[3]),
               "seconds": round(time.time() - t0, 1),
               "per_query_class_mass_sum": {"min": out["row_sum_min"],
                                            "max": out["row_sum_max"]},
               "fp32_vs_fp64_score_path_max_abs_class_mass_delta": agreement}
        rec.update(summarise(out["per_head"], names, layout["segments"]))
        rec["top_prefix_positions"] = top_positions(out["importance"], pieces)
        records.append(rec)
        tensors[f"b{f['block']}_s{f['step']}"] = \
            out["importance"].to(torch.float32).contiguous()
        print(f"  {f['name']}  prefix {rec['top_level']['prefix']:.4%}  "
              f"video {rec['top_level']['latent_video']:.4%}  "
              f"audio {rec['top_level']['latent_audio']:.4%}  "
              f"({rec['seconds']}s)")
        del blob, q, k, out

    report = {
        "measurement": "latent-video query attention mass by key class, per "
                       "captured DiT block and sampler step",
        "produced_by": producer_provenance(_HERE),
        "capture": {
            "directory_name": directory.name,
            "files": [f["name"] for f in files],
            "sequence_length": seq_len,
            "manifest_agreement": manifest_agreement(
                manifest, layout, seq_len, width, height, length),
        },
        "layout": {
            "segments": [[a_, b_, kind] for a_, b_, kind in layout["segments"]],
            "canvas": {"width": width, "height": height, "length": length,
                       "source": canvas_source},
            "latent_t": layout["latent_t"], "audio_t": layout["audio_t"],
            "image_references": layout["refs"],
            "prefix_length_cross_check": layout["cross_check"],
            "derivation": "comfy/ldm/minimax/model.py::PackedLayout via "
                          "bench/count_packed_rows.py::_layout; temporal grid "
                          "via comfy_extras/nodes_minimax_h3.py::temporal_shape",
        },
        "classes": {
            "names": names,
            "position_counts": counts,
            "prefix_tag_source": tag_source,
            "prefix_evidence": evidence,
            "operator_defined": [n for n, _ in extra_spans],
            "special_token_ids": specials,
            "derivation": "non-prefix classes are PackedLayout segment kinds; "
                          "prefix classes are token ids resolved through the "
                          "loaded tokenizer's vocabulary against "
                          "vendor_config/tokenizer_config.json, plus the label "
                          "forms in comfy/text_encoders/minimax.py",
        },
        "sampling": {
            "query_segment": "video",
            "query_span": [v_lo, v_hi],
            "rows": int(rows.numel()),
            "strata": a.strata,
            "row_indices_first_last": [int(rows[0]), int(rows[-1])],
            "derivation": "bench/grade_sage_on_capture.py::sample_rows, "
                          "offset into the video segment",
        },
        "numerics": {"score_dtype": a.score_dtype,
                     "softmax_and_accumulation_dtype": "float64",
                     "device": str(device),
                     "head_chunk": a.head_chunk},
        "prompt_source": prompt_source,
        "records": records,
    }

    if a.json:
        out_path = a.json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n  wrote {_repo_relative(out_path)}")
        st = a.safetensors or out_path.with_name(
            out_path.stem + "_importance.safetensors")
    else:
        print(json.dumps(report, indent=2))
        st = a.safetensors

    if st and tensors:
        from safetensors.torch import save_file
        tensors["prefix_class_id"] = class_of[:layout["prefix_len"]].to(
            torch.int16).contiguous().cpu()
        st = Path(st).resolve()
        save_file(tensors, str(st), metadata={
            "capture_directory_name": directory.name,
            "class_names": json.dumps(names),
            "prefix_length": str(layout["prefix_len"]),
            "meaning": "mean softmax attention mass received at each prefix "
                       "position, averaged over sampled latent-video queries "
                       "and all heads; one tensor per block/step",
        })
        print(f"  wrote {_repo_relative(st)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
