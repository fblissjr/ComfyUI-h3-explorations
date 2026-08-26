"""The release's own configuration, vendored, and the readers for it.

Three numbers and one token list in this repo used to be literals in our code
or, worse, inherited from a shared helper written for a different model. They
are properties of the published MiniMax H3 release, so they live here as the
release's own files rather than as constants somebody typed.

`vendor_config/` holds, copied verbatim on 2026-08-21 from the published repo:

  tokenizer_config.json          the twenty `additional_special_tokens`
  preprocessor_config.json       image min/max pixels, patch geometry
  video_preprocessor_config.json video min/max pixels
  {fl2va,ref2va}_model_index.json  the `_minimax_h3` partition and task lists
  sha256.json                    what each of the above hashed to when copied

**Why vendored rather than read from the weights.** The weights are ~200 GB and
live outside the repo; a graph that needs to know the image floor cannot depend
on them being downloaded. These files are 13 KB and they are the part that
matters. `bench/check_vendor_config.py` compares them against the release when
it is on disk and says so loudly when it is not, which is the difference
between "verified identical" and "nobody looked".

**The fast image processor spells min/max pixels as `size.shortest_edge` and
`size.longest_edge`.** Those are pixel COUNTS, not edge lengths -- 65,536 is
256x256, not a 65,536-pixel edge. Read that wrong and every bound here is
nonsense, which is why the readers below name what they return.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

DIR = Path(__file__).resolve().parent / "vendor_config"


@functools.lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It ships with this repo; if it is gone, "
            "re-copy it from the release rather than typing the values."
        )
    return json.loads(path.read_text())


def additional_special_tokens() -> list[str]:
    """Every special token the release declares, in its declared order.

    Order is load-bearing: these are appended past the end of the vocabulary,
    so the order here decides which id each one gets.
    """
    toks = _load("tokenizer_config.json").get("additional_special_tokens")
    if not isinstance(toks, list) or not toks:
        raise ValueError("vendored tokenizer_config declares no "
                         "additional_special_tokens")
    return list(toks)


def h3_markers() -> list[str]:
    """The markers the release appends past the stock vocabulary, in order.

    Derived from the vendored declaration rather than typed: a declared
    `additional_special_tokens` entry with no `added_tokens_decoder` entry is
    one the release adds past the last stock Qwen id. That is the observable --
    the release's own two lists disagreeing -- rather than a name, a count or a
    hardcoded string list, so a release that renames one or adds an eighth is
    read correctly instead of matching what somebody expected. The same
    derivation is what `bench/check_native_h3_presentation.py` uses to compute
    the ids; this is the token list half of it.
    """
    cfg = _load("tokenizer_config.json")
    decoder = cfg.get("added_tokens_decoder") or {}
    stock = {entry.get("content") for entry in decoder.values()}
    markers = [tok for tok in additional_special_tokens() if tok not in stock]
    if not markers:
        raise ValueError(
            "vendored tokenizer_config declares no special token past its own "
            "added_tokens_decoder; either the copy is stale or the release "
            "stopped appending markers"
        )
    return markers


def _pixels(cfg: dict, label: str) -> tuple[int, int]:
    size = cfg.get("size") or {}
    lo, hi = size.get("shortest_edge"), size.get("longest_edge")
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 < lo < hi:
        raise ValueError(f"vendored {label} has no usable size bounds: {size!r}")
    return lo, hi


def image_pixel_bounds() -> tuple[int, int]:
    """(min_pixels, max_pixels) for a still, as the release declares them."""
    return _pixels(_load("preprocessor_config.json"), "preprocessor_config.json")


def video_pixel_bounds() -> tuple[int, int]:
    """(min_pixels, max_pixels) for a video block."""
    return _pixels(_load("video_preprocessor_config.json"),
                   "video_preprocessor_config.json")


def patch_geometry() -> dict:
    """patch_size / temporal_patch_size / merge_size, and the normalisation."""
    cfg = _load("preprocessor_config.json")
    return {k: cfg[k] for k in
            ("patch_size", "temporal_patch_size", "merge_size",
             "image_mean", "image_std") if k in cfg}


def video_patch_geometry() -> dict:
    """Video patch geometry and normalisation declared by the release.

    Keep this separate from :func:`patch_geometry` even while the two files
    agree.  The duration-aware video processor owns its own config, and
    borrowing the still-image values would turn an upstream divergence into a
    silent local assumption.
    """
    cfg = _load("video_preprocessor_config.json")
    return {k: cfg[k] for k in
            ("patch_size", "temporal_patch_size", "merge_size",
             "image_mean", "image_std") if k in cfg}


def partition_tasks() -> dict[str, list[str]]:
    """Which tasks each released partition serves, from its own model_index."""
    out = {}
    for variant in ("fl2va", "ref2va"):
        block = _load(f"{variant}_model_index.json").get("_minimax_h3") or {}
        if block.get("partition") != variant:
            raise ValueError(
                f"vendored {variant}_model_index.json declares partition "
                f"{block.get('partition')!r}")
        out[variant] = list(block.get("tasks") or ())
    return out
