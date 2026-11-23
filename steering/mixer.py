"""Unified Sound-Mixer runtime compiler (`MixerCompiler`) for H3 control vectors.

Resolves `[routing]` (`bus`, `regime`, `target`, `fields`, `streams`, `block_band`,
`needs_ref_release`, `families`, `default_strength`; fader `-1.5..+2.0`), expands Macro
Channel Presets (`1 .safetensors = 1 single-bus vector`), compiles static whole-clip
faders to `1x` Single-Pass (`BasicGuider` + `peer_ros` + `95.5->98.5` smooth knee +
`auto_1x_kv`) and timeline `[F,1,1]` / spatial `[F,H,W]` / `[concept]` rewrites to `2x`
`MiniMaxH3EditGuider`, auto-lowers `MiniMaxH3ReleaseReferences(end_percent=0.33)` when
`needs_ref_release=True` alongside `<Picture 1>`, and filters `[Image]` (`h3_image`) stems.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import (
    DEFAULT_RELEASE_END_PERCENT,
    FADER_MAX,
    FADER_MIN,
    NON_H3_IMAGE_STEMS,
    STACK_SMOOTH_CEIL_NORM,
    STACK_SMOOTH_KNEE_NORM,
    STEER_BUSES,
    STEER_EDIT_PASSES,
    STEER_MODES,
    STEER_REGIMES,
    STEER_SPLIT_BACKENDS,
    STEER_STREAMS,
    STEER_TARGETS,
)
from .data import RoutingSpec, parse_routing_table
from .hooks import validate_block_band

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIR_BANK_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
USER_PAIR_BANK_DIR = REPO_ROOT / "data" / "control_vectors"

_ROUTING_TOML_CACHE: dict[tuple[str, int, int], RoutingSpec | None] = {}


@dataclass(frozen=True)
class ChannelStrip:
    """One fader channel on the Sound-Mixer board (pointing at a single-bus `.safetensors` stem)."""

    vector: str
    strength: float | None = None
    mode: str = "add_norm_matched_erasure"
    bus: str | None = None
    target: str | None = None
    fields: tuple[str, ...] | None = None
    streams: str | None = None
    block_band: str | None = None
    needs_ref_release: bool | None = None
    start_percent: float = 0.0
    end_percent: float = 1.0
    hold_words: str = ""
    hold_alpha: float = 1.0
    allow_recipe_mismatch: bool = True
    edit_pass: str | None = None
    # Optional timeline `[F, 1, 1]` or spatial `[F, H, W]` envelope marker triggering 2x EditGuider
    envelope_kind: str = "static"  # "static" | "timeline" | "spatial"


@dataclass(frozen=True)
class MacroChannelPreset:
    """Multi-strip macro preset combining single-bus vectors + optional `prompt_patch`."""

    name: str
    description: str
    strips: tuple[ChannelStrip, ...]
    prompt_patch: str = ""


@dataclass(frozen=True)
class CompiledSteerNode:
    """Resolved `MiniMaxH3Steer` node specification emitted by `MixerCompiler`."""

    vector: str
    bus: str
    regime: str
    strength: float
    mode: str
    target: str
    fields: tuple[str, ...]
    streams: str
    block_band: str
    split_backend: str
    start_percent: float
    end_percent: float
    hold_words: str
    hold_alpha: float
    allow_recipe_mismatch: bool
    edit_pass: str
    needs_ref_release: bool

    def to_comfy_inputs(self, model_link: list[Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "model": copy.deepcopy(model_link),
            "vector": self.vector,
            "strength": float(self.strength),
            "mode": self.mode,
            "target": self.target,
            "start_percent": float(self.start_percent),
            "end_percent": float(self.end_percent),
            "streams": self.streams,
            "hold_words": self.hold_words,
            "hold_alpha": float(self.hold_alpha),
            "allow_recipe_mismatch": bool(self.allow_recipe_mismatch),
            "edit_pass": self.edit_pass,
            "block_band": self.block_band,
            "split_backend": self.split_backend,
        }
        if self.fields:
            inputs["fields"] = list(self.fields)
        return inputs


@dataclass(frozen=True)
class CompiledMixerPlan:
    """Complete Sound-Mixer compilation output (`1x` BasicGuider vs `2x` EditGuider + ReleaseReferences)."""

    pass_mode: str  # "1x" or "2x"
    guider_class: str  # "BasicGuider" or "MiniMaxH3EditGuider"
    steer_nodes: tuple[CompiledSteerNode, ...]
    peer_ros: bool = True
    smooth_knee: tuple[float, float] = (STACK_SMOOTH_KNEE_NORM, STACK_SMOOTH_CEIL_NORM)
    split_backend: str = "auto_1x_kv"
    needs_ref_release: bool = False
    release_end_percent: float = 1.0
    prompt_patches: tuple[str, ...] = field(default_factory=tuple)
    envelope_kind: str = "static"


# Diverse built-in Macro Channel Presets spanning physical/pyro, vocal/acoustic, weather,
# lighting/club, and tactile craft families (keeping 1 .safetensors = 1 single-bus vector).
BUILT_IN_MACRO_PRESETS: dict[str, MacroChannelPreset] = {
    "pyro_action": MacroChannelPreset(
        name="pyro_action",
        description="Pyrotechnic fireball chain explosions (Motion/Look) + klaxon & blast shockwave (Audio).",
        strips=(
            ChannelStrip(vector="pyro_chain_explosions", strength=1.15),
            ChannelStrip(vector="pyro_klaxon_audio", strength=1.10),
        ),
    ),
    "cathedral_opera": MacroChannelPreset(
        name="cathedral_opera",
        description="Full-chest operatic vocal projection (Audio vocal) + vast stone cathedral acoustics (Audio space).",
        strips=(
            ChannelStrip(vector="operatic_bellow", strength=1.15),
            ChannelStrip(vector="reverb_cathedral", strength=1.10),
        ),
    ),
    "blizzard_gale": MacroChannelPreset(
        name="blizzard_gale",
        description="Whiteout snowdrifts (Look) + horizontal hurricane wind gale (Motion).",
        strips=(
            ChannelStrip(vector="heavy_blizzard", strength=1.10),
            ChannelStrip(vector="wind_tunnel_gale", strength=1.05),
        ),
    ),
    "uv_club_rave": MacroChannelPreset(
        name="uv_club_rave",
        description="Ultraviolet fluorescence (Look) + club strobe pulse with optional ambient bass patch.",
        strips=(
            ChannelStrip(vector="blacklight_uv_rave", strength=1.15),
            ChannelStrip(vector="strobe_club_pulse", strength=0.95),
        ),
        prompt_patch="overall_soundscape: Deep muffled club sub-bass thumping beneath crisp room acoustics.",
    ),
    "wool_whisper": MacroChannelPreset(
        name="wool_whisper",
        description="Tactile knitted wool world (Look) + intimate close-mic ASMR foley & whisper (Audio).",
        strips=(
            ChannelStrip(vector="knitted_wool_world", strength=1.20),
            ChannelStrip(vector="whisper_asmr", strength=1.00),
        ),
    ),
}


def extract_stem_from_vector_name(vector_name: str | Path) -> str:
    norm = str(vector_name).strip().replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    if base.endswith(".safetensors"):
        base = base[: -len(".safetensors")]
    for fam in ("fl2va", "ref2va", "fasth3", "h3_image"):
        tag = f"__{fam}"
        if tag in base:
            cand = base.split(tag, 1)[0]
            if cand:
                return cand
    return base


def _read_routing_from_safetensors(path: Path) -> RoutingSpec | None:
    if not path.is_file():
        return None
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as sf:
            md = sf.metadata() or {}
        raw_rt = md.get("routing")
        if raw_rt:
            parsed = json.loads(raw_rt) if isinstance(raw_rt, str) else raw_rt
            if isinstance(parsed, Mapping):
                return parse_routing_table(parsed, source=str(path))
    except Exception:
        return None
    return None


def _read_routing_from_toml(stem: str, toml_dirs: Sequence[Path]) -> RoutingSpec | None:
    for d in toml_dirs:
        p = d / f"{stem}.toml"
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        key = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
        if key in _ROUTING_TOML_CACHE:
            return _ROUTING_TOML_CACHE[key]
        spec: RoutingSpec | None = None
        try:
            doc = tomllib.loads(p.read_text(encoding="utf-8"))
            if isinstance(doc.get("routing"), Mapping):
                spec = parse_routing_table(doc["routing"], source=str(p))
        except Exception:
            spec = None
        _ROUTING_TOML_CACHE[key] = spec
        return spec
    return None


def default_fallback_routing(stem: str) -> RoutingSpec:
    fams = ("fl2va", "ref2va") if stem in NON_H3_IMAGE_STEMS else ("fl2va", "ref2va", "h3_image")
    return RoutingSpec(
        bus="look",
        regime="a",
        target="scene_only",
        fields=(),
        streams="both",
        block_band="all",
        needs_ref_release=False,
        families=fams,
        default_strength=1.0,
    )


def resolve_vector_routing(
    vector_name: str | Path,
    *,
    safetensors_path: str | Path | None = None,
    toml_dirs: Sequence[Path] | None = None,
) -> RoutingSpec:
    """Resolve `RoutingSpec` for `vector_name` from `.safetensors` metadata, TOML `[routing]`, or fallback."""
    cand_path = Path(safetensors_path) if safetensors_path is not None else Path(str(vector_name))
    if cand_path.suffix == ".safetensors" and cand_path.is_file():
        st_rt = _read_routing_from_safetensors(cand_path)
        if st_rt is not None:
            return st_rt

    stem = extract_stem_from_vector_name(vector_name)
    search_dirs = tuple(toml_dirs) if toml_dirs is not None else (USER_PAIR_BANK_DIR, PAIR_BANK_DIR)
    toml_rt = _read_routing_from_toml(stem, search_dirs)
    if toml_rt is not None:
        return toml_rt

    return default_fallback_routing(stem)


def validate_fader_strength(strength: float, *, clamp: bool = False) -> float:
    s = float(strength)
    if not math.isfinite(s):
        raise ValueError(f"fader strength must be finite, got {strength!r}")
    if clamp:
        return max(FADER_MIN, min(FADER_MAX, s))
    if not (FADER_MIN <= s <= FADER_MAX):
        raise ValueError(
            f"fader strength must lie in [{FADER_MIN}, {FADER_MAX}], got {strength!r}"
        )
    return s


def filter_vectors_for_family(
    vectors: Sequence[str | Mapping[str, Any]],
    *,
    family: str,
    toml_dirs: Sequence[Path] | None = None,
) -> list[Any]:
    """Filter selectable control-vector stems or dicts by `routing.families` (e.g. `family='h3_image'`)."""
    fam_norm = str(family or "fl2va").strip().lower()
    if fam_norm in ("image", "h3-image", "minimax-h3-image"):
        fam_norm = "h3_image"
    out: list[Any] = []
    for item in vectors:
        if isinstance(item, Mapping):
            raw_name = str(item.get("vector") or item.get("name") or item.get("file") or "").strip()
            st_path = item.get("path")
        else:
            raw_name = str(item).strip()
            st_path = None
        if not raw_name or raw_name == "none":
            out.append(item)
            continue
        rt = resolve_vector_routing(
            raw_name,
            safetensors_path=st_path if isinstance(st_path, (str, Path)) else None,
            toml_dirs=toml_dirs,
        )
        if fam_norm in rt.families:
            out.append(item)
    return out


class MixerCompiler:
    """Unified Sound-Mixer runtime compiler (`1x` BasicGuider vs `2x` EditGuider + ReleaseReferences)."""

    def __init__(
        self,
        *,
        toml_dirs: Sequence[Path] | None = None,
        macro_presets: Mapping[str, MacroChannelPreset] | None = None,
    ) -> None:
        self.toml_dirs = tuple(toml_dirs) if toml_dirs is not None else (USER_PAIR_BANK_DIR, PAIR_BANK_DIR)
        self.macro_presets: dict[str, MacroChannelPreset] = dict(BUILT_IN_MACRO_PRESETS)
        if macro_presets:
            self.macro_presets.update(macro_presets)

    def resolve_routing(
        self, vector_name: str | Path, *, safetensors_path: str | Path | None = None
    ) -> RoutingSpec:
        return resolve_vector_routing(
            vector_name, safetensors_path=safetensors_path, toml_dirs=self.toml_dirs
        )

    def filter_for_family(self, vectors: Sequence[str | Mapping[str, Any]], family: str) -> list[Any]:
        return filter_vectors_for_family(vectors, family=family, toml_dirs=self.toml_dirs)

    def expand_macro(
        self,
        preset: str | MacroChannelPreset,
        *,
        master_gain: float = 1.0,
        envelope_kind: str = "static",
    ) -> tuple[list[ChannelStrip], str]:
        mac = self.macro_presets.get(preset) if isinstance(preset, str) else preset
        if mac is None:
            raise ValueError(f"unknown MacroChannelPreset {preset!r}; available={sorted(self.macro_presets)}")
        expanded: list[ChannelStrip] = []
        for st in mac.strips:
            rt = self.resolve_routing(st.vector)
            base_s = rt.default_strength if st.strength is None else float(st.strength)
            scaled_s = validate_fader_strength(base_s * float(master_gain), clamp=True)
            eff_env = envelope_kind if st.envelope_kind == "static" else st.envelope_kind
            expanded.append(
                ChannelStrip(
                    vector=st.vector,
                    strength=scaled_s,
                    mode=st.mode,
                    bus=st.bus,
                    target=st.target,
                    fields=st.fields,
                    streams=st.streams,
                    block_band=st.block_band,
                    needs_ref_release=st.needs_ref_release,
                    start_percent=st.start_percent,
                    end_percent=st.end_percent,
                    hold_words=st.hold_words,
                    hold_alpha=st.hold_alpha,
                    allow_recipe_mismatch=st.allow_recipe_mismatch,
                    edit_pass=st.edit_pass,
                    envelope_kind=eff_env,
                )
            )
        return expanded, mac.prompt_patch

    def compile(
        self,
        strips: Sequence[ChannelStrip | Mapping[str, Any]] = (),
        *,
        macros: Sequence[str | MacroChannelPreset | tuple[str | MacroChannelPreset, float]] = (),
        family: str = "fl2va",
        has_picture_ref: bool = False,
        has_concept_rewrite: bool = False,
        shot_envelope: str = "static",  # "static" | "timeline" | "spatial"
        base_release_end_percent: float = 1.0,
        clamp_faders: bool = False,
    ) -> CompiledMixerPlan:
        """Compile channel strips + macros into a deterministic `1x` or `2x` execution plan."""
        if shot_envelope not in ("static", "timeline", "spatial"):
            raise ValueError(f"shot_envelope must be 'static', 'timeline', or 'spatial', got {shot_envelope!r}")

        all_strips: list[ChannelStrip] = []
        prompt_patches: list[str] = []

        for m_item in macros:
            if isinstance(m_item, tuple) and len(m_item) == 2:
                m_spec, m_gain = m_item[0], float(m_item[1])
            else:
                m_spec, m_gain = m_item, 1.0  # type: ignore[assignment]
            exp_strips, p_patch = self.expand_macro(m_spec, master_gain=m_gain)
            all_strips.extend(exp_strips)
            if p_patch and p_patch not in prompt_patches:
                prompt_patches.append(p_patch)

        for raw_st in strips:
            if isinstance(raw_st, ChannelStrip):
                all_strips.append(raw_st)
            elif isinstance(raw_st, Mapping):
                vec_n = str(raw_st.get("vector") or raw_st.get("file") or raw_st.get("name") or "").strip()
                raw_fields = raw_st.get("fields")
                all_strips.append(
                    ChannelStrip(
                        vector=vec_n,
                        strength=float(raw_st["strength"]) if raw_st.get("strength") is not None else None,
                        mode=str(raw_st.get("mode") or "add_norm_matched_erasure"),
                        bus=str(raw_st["bus"]) if raw_st.get("bus") else None,
                        target=str(raw_st["target"]) if raw_st.get("target") else None,
                        fields=tuple(str(x) for x in raw_fields) if raw_fields is not None else None,
                        streams=str(raw_st["streams"]) if raw_st.get("streams") else None,
                        block_band=str(raw_st["block_band"]) if raw_st.get("block_band") else None,
                        needs_ref_release=bool(raw_st["needs_ref_release"]) if "needs_ref_release" in raw_st and raw_st["needs_ref_release"] is not None else None,
                        start_percent=float(raw_st.get("start_percent", 0.0)),
                        end_percent=float(raw_st.get("end_percent", 1.0)),
                        hold_words=str(raw_st.get("hold_words") or ""),
                        hold_alpha=float(raw_st.get("hold_alpha", 1.0)),
                        allow_recipe_mismatch=bool(raw_st.get("allow_recipe_mismatch", True)),
                        edit_pass=str(raw_st["edit_pass"]) if raw_st.get("edit_pass") else None,
                        envelope_kind=str(raw_st.get("envelope_kind") or "static"),
                    )
                )
            else:
                raise TypeError(f"unsupported strip entry type {type(raw_st).__name__}")

        has_animated_strip = any(s.envelope_kind in ("timeline", "spatial") for s in all_strips)
        needs_2x = bool(
            shot_envelope in ("timeline", "spatial")
            or has_animated_strip
            or has_concept_rewrite
            or bool(prompt_patches)
        )
        eff_envelope = (
            shot_envelope
            if shot_envelope != "static"
            else (
                "spatial"
                if any(s.envelope_kind == "spatial" for s in all_strips)
                else ("timeline" if has_animated_strip else "static")
            )
        )

        fam_norm = str(family or "fl2va").strip().lower()
        compiled_nodes: list[CompiledSteerNode] = []
        any_needs_release = False

        for st in all_strips:
            if not st.vector or st.vector == "none":
                continue
            rt = self.resolve_routing(st.vector)
            if fam_norm == "h3_image" and "h3_image" not in rt.families:
                raise ValueError(
                    f"control vector {st.vector!r} does not support family='h3_image' (routing.families={list(rt.families)})"
                )
            eff_strength = validate_fader_strength(
                rt.default_strength if st.strength is None else st.strength,
                clamp=clamp_faders,
            )
            if eff_strength == 0.0:
                continue

            bus = st.bus or rt.bus
            if bus not in STEER_BUSES:
                raise ValueError(f"bus must be one of {STEER_BUSES}, got {bus!r}")
            regime = rt.regime
            if regime not in STEER_REGIMES:
                raise ValueError(f"regime must be one of {STEER_REGIMES}, got {regime!r}")

            mode = st.mode if st.mode != "add_norm_matched_erasure" or bus != "hold" else "project_out"
            if mode not in STEER_MODES:
                raise ValueError(f"mode must be one of {STEER_MODES}, got {mode!r}")

            target = st.target or rt.target
            if target not in STEER_TARGETS:
                raise ValueError(f"target must be one of {STEER_TARGETS}, got {target!r}")
            fields_tup = st.fields if st.fields is not None else rt.fields

            streams = st.streams or rt.streams
            if fam_norm == "h3_image":
                streams = "both"
            if streams not in STEER_STREAMS:
                raise ValueError(f"streams must be one of {STEER_STREAMS}, got {streams!r}")

            block_band = validate_block_band(st.block_band or rt.block_band)
            nrr = rt.needs_ref_release if st.needs_ref_release is None else bool(st.needs_ref_release)
            if nrr and eff_strength > 0.0:
                any_needs_release = True

            if st.edit_pass is not None:
                if st.edit_pass not in STEER_EDIT_PASSES:
                    raise ValueError(f"edit_pass must be one of {STEER_EDIT_PASSES}, got {st.edit_pass!r}")
                eff_edit_pass = st.edit_pass
            elif not needs_2x:
                eff_edit_pass = "both"
            else:
                eff_edit_pass = (
                    "edited_only"
                    if (st.envelope_kind in ("timeline", "spatial") or shot_envelope in ("timeline", "spatial"))
                    else "both"
                )

            compiled_nodes.append(
                CompiledSteerNode(
                    vector=st.vector,
                    bus=bus,
                    regime=regime,
                    strength=eff_strength,
                    mode=mode,
                    target=target,
                    fields=fields_tup,
                    streams=streams,
                    block_band=block_band,
                    split_backend="auto_1x_kv",
                    start_percent=float(st.start_percent),
                    end_percent=float(st.end_percent),
                    hold_words=st.hold_words,
                    hold_alpha=float(st.hold_alpha),
                    allow_recipe_mismatch=bool(st.allow_recipe_mismatch),
                    edit_pass=eff_edit_pass,
                    needs_ref_release=nrr,
                )
            )

        rel_ep = float(base_release_end_percent)
        if has_picture_ref and any_needs_release:
            rel_ep = min(rel_ep, DEFAULT_RELEASE_END_PERCENT)

        return CompiledMixerPlan(
            pass_mode="2x" if needs_2x else "1x",
            guider_class="MiniMaxH3EditGuider" if needs_2x else "BasicGuider",
            steer_nodes=tuple(compiled_nodes),
            peer_ros=True,
            smooth_knee=(STACK_SMOOTH_KNEE_NORM, STACK_SMOOTH_CEIL_NORM),
            split_backend="auto_1x_kv",
            needs_ref_release=bool(has_picture_ref and any_needs_release),
            release_end_percent=rel_ep,
            prompt_patches=tuple(prompt_patches),
            envelope_kind=eff_envelope,
        )


def resolve_director_steer_entry(
    vec: Mapping[str, Any],
    *,
    model_link: list[Any],
    toml_dirs: Sequence[Path] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Resolve a `director/compiler.py` `control_vectors` dict via `steering.mixer` `[routing]`,
    returning `(steer_inputs_dict, needs_ref_release)`."""
    vec_name = str(vec.get("vector") or vec.get("file") or vec.get("name") or "").strip()
    rt = resolve_vector_routing(vec_name, toml_dirs=toml_dirs)
    strength = float(vec.get("strength", rt.default_strength))
    mode = str(vec.get("mode") or ("project_out" if rt.bus == "hold" else "add_norm_matched_erasure"))
    target = str(vec.get("target") or rt.target)
    streams = str(vec.get("streams") or rt.streams)
    block_band = validate_block_band(vec.get("block_band") or rt.block_band)
    fields = list(vec["fields"]) if vec.get("fields") else list(rt.fields)
    nrr = bool(vec["needs_ref_release"]) if "needs_ref_release" in vec and vec["needs_ref_release"] is not None else rt.needs_ref_release

    steer_inputs: dict[str, Any] = {
        "model": copy.deepcopy(model_link),
        "vector": vec_name,
        "strength": strength,
        "mode": mode,
        "target": target,
        "start_percent": float(vec.get("start_percent", 0.0)),
        "end_percent": float(vec.get("end_percent", 1.0)),
        "streams": streams,
        "hold_words": str(vec.get("hold_words") or ""),
        "hold_alpha": float(vec.get("hold_alpha", 1.0)),
        "allow_recipe_mismatch": bool(vec.get("allow_recipe_mismatch", True)),
        "edit_pass": str(vec.get("edit_pass") or "edited_only"),
        "block_band": block_band,
    }
    if fields:
        steer_inputs["fields"] = fields
    return steer_inputs, bool(nrr and strength > 0.0)
