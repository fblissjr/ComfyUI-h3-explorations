"""MiniMax H3 clean-slate `steering` package (`10` focused modules + Sound-Mixer compiler).

Modules:
- `constants`: settled dimensions (`5376`-D post-refiner, `5120`-D encoder), `67.5` target RMS,
  `95.5 -> 98.5` smooth knee, 4-bus enum (`look`, `motion`, `audio`, `hold`), Day-1 `regime = ("a",)`.
- `data`: `[vector]` TOML loader (`[[pair]]`, `[[hold]]`, `[routing]`, `[provenance]`), Gates A-D.
- `encode`: LoRA-aware Qwen3-VL L49 + `TokenRefiner` encoding & media-pair geometry alignment.
- `extract`: `< 50 ms` CPU extraction (`unit_mean=False`), Pole-A erasure SVD.
- `runtime`: Point C prelude (`h_bar`, `peer_ros=True`, clamped `[0, 1]` union SVD, `s=0` exact identity).
- `hooks`: `auto_1x_kv` stream split, `block_band`, and attention seam helpers.
- `nodes` & `legacy_nodes`: Option-B ComfyUI nodes (zero `node_id_manifest.json` diff).
- `xray`: compact CPU J-Lens + `Model2Vec` lexicon decoder & word synthesizer.
- `mixer`: unified Sound-Mixer (`MixerCompiler`, `1x` static vs `2x` animated/rewrite, Macro Presets).
"""

from __future__ import annotations

from . import constants, data, extract, hooks, mixer, query_router, runtime, xray
from .constants import (
    CANONICAL_PROVENANCE_KEYS,
    COMPILER_EXCLUDE_SHOTS,
    ENCODER_DIM,
    ERASURE_MATCH_TAU,
    ERASURE_MAX_RANK,
    ERASURE_RIDGE_REL,
    FADER_MAX,
    FADER_MIN,
    FORMAT_VERSION,
    MIN_ERASURE_PAIRS,
    NON_H3_IMAGE_STEMS,
    NORM_MATCHED_TARGET,
    POST_REFINER_DIM,
    STACK_SMOOTH_CEIL_NORM,
    STACK_SMOOTH_KNEE_NORM,
    STEER_BUSES,
    STEER_REGIMES,
)
from .data import HoldItem, PairItem, PairSet, RoutingSpec, infer_routing_from_pairs, load_pairs
from .extract import PooledPairSet, load_vector, pool_pair_set, save_vector
from .mixer import (
    BUILT_IN_MACRO_PRESETS,
    ChannelStrip,
    CompiledMixerPlan,
    CompiledSteerNode,
    MacroChannelPreset,
    MixerCompiler,
    filter_vectors_for_family,
    resolve_director_steer_entry,
    resolve_vector_routing,
    validate_fader_strength,
)
from .query_router import SECTION_SPLIT_MODES, MiniMaxH3QueryRouter, attach_query_router
from .runtime import apply_post_refiner, soft_cap_stacked_rms

__all__ = [
    "BUILT_IN_MACRO_PRESETS",
    "CANONICAL_PROVENANCE_KEYS",
    "COMPILER_EXCLUDE_SHOTS",
    "ChannelStrip",
    "CompiledMixerPlan",
    "CompiledSteerNode",
    "ENCODER_DIM",
    "ERASURE_MATCH_TAU",
    "ERASURE_MAX_RANK",
    "ERASURE_RIDGE_REL",
    "FADER_MAX",
    "FADER_MIN",
    "FORMAT_VERSION",
    "HoldItem",
    "MIN_ERASURE_PAIRS",
    "MacroChannelPreset",
    "MiniMaxH3QueryRouter",
    "MixerCompiler",
    "NON_H3_IMAGE_STEMS",
    "NORM_MATCHED_TARGET",
    "POST_REFINER_DIM",
    "PairItem",
    "PairSet",
    "PooledPairSet",
    "RoutingSpec",
    "SECTION_SPLIT_MODES",
    "STACK_SMOOTH_CEIL_NORM",
    "STACK_SMOOTH_KNEE_NORM",
    "STEER_BUSES",
    "STEER_REGIMES",
    "apply_post_refiner",
    "attach_query_router",
    "constants",
    "data",
    "extract",
    "filter_vectors_for_family",
    "hooks",
    "infer_routing_from_pairs",
    "load_pairs",
    "load_vector",
    "mixer",
    "pool_pair_set",
    "query_router",
    "resolve_director_steer_entry",
    "resolve_vector_routing",
    "runtime",
    "save_vector",
    "soft_cap_stacked_rms",
    "validate_fader_strength",
    "xray",
]

def __getattr__(name: str):
    if name in ("nodes", "legacy_nodes", "encode"):
        import importlib

        if name == "legacy_nodes":
            importlib.import_module(".nodes", __name__)
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
