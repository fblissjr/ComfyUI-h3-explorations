"""Short-contrast and simple-words compiler for MiniMax H3 control vectors.

Allows steering H3 from:
  1. Simple words or a single short phrase (`compile_words_to_pairset`, e.g.
     `"erratic handheld camera shake"`, `"awkward eye-contact avoidance"`,
     `"booming operatic bellow"`, `"muffled through-the-wall acoustics"`,
     `"weightless zero gravity"`, `"1974 16mm film grain"`), with zero `.toml`
     authoring and no manual Pole A baseline required, or
  2. Explicit `(pole_a, pole_b)` short contrast clauses (`compile_short_contrast_pairs`
     and `compile_short_contrast_to_toml`).

Why this works mathematically:
Under `pooling = "diff_span"` (`h3_spans.py::pooling_mask`), only the differing
token span `[first_diff:end_diff]` is mean-pooled per pair side. The surrounding
canonical H3 scene in `prompt_bank/control_vectors/canonical_frames.json` is
never averaged into the vector directly; it provides:
  1. Causal Stage-1 prefix context (`0..first_diff`) in `Qwen3-VL-32B`, and
  2. Bidirectional Stage-2 self-attention context (`0..L`) in the 2-block
     `TokenRefiner` (with Pika's 24 `token_refiner` LoRA branches).
Averaging across diverse canonical scenes cancels scene-specific context leakage
while requiring the user to supply only a few words.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import control_vectors as _cv
except ImportError:
    import control_vectors as _cv

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CANONICAL_FRAMES_PATH = (
    _REPO_ROOT / "prompt_bank" / "control_vectors" / "canonical_frames.json"
)

CANONICAL_SLOTS: tuple[str, ...] = (
    "opening_framing",
    "subject_behavior",
    "interpersonal_dynamics",
    "camera_motion",
    "speech_delivery",
    "soundscape",
)

SLOT_ALIASES: dict[str, str] = {
    "auto": "auto",
    # opening_framing
    "opening_framing": "opening_framing",
    "shot_look": "opening_framing",
    "shot_look_clause": "opening_framing",
    "look": "opening_framing",
    "style": "opening_framing",
    "lighting": "opening_framing",
    "framing": "opening_framing",
    # subject_behavior
    "subject_behavior": "subject_behavior",
    "subject_action": "subject_behavior",
    "subject_action_clause": "subject_behavior",
    "subject": "subject_behavior",
    "beat2": "subject_behavior",
    "action": "subject_behavior",
    "behavior": "subject_behavior",
    "gesture": "subject_behavior",
    "posture": "subject_behavior",
    # interpersonal_dynamics
    "interpersonal_dynamics": "interpersonal_dynamics",
    "scene_dynamics": "interpersonal_dynamics",
    "environment_physics": "interpersonal_dynamics",
    "environment_physics_clause": "interpersonal_dynamics",
    "physics": "interpersonal_dynamics",
    "atmosphere": "interpersonal_dynamics",
    "dynamics": "interpersonal_dynamics",
    "environment": "interpersonal_dynamics",
    # camera_motion
    "camera_motion": "camera_motion",
    "camera": "camera_motion",
    "camera_clause": "camera_motion",
    "movement": "camera_motion",
    "rig": "camera_motion",
    # speech_delivery
    "speech_delivery": "speech_delivery",
    "speech_delivery_clause": "speech_delivery",
    "vocal": "speech_delivery",
    "voice": "speech_delivery",
    "prosody": "speech_delivery",
    "delivery": "speech_delivery",
    # soundscape
    "soundscape": "soundscape",
    "soundscape_clause": "soundscape",
    "sound": "soundscape",
    "audio": "soundscape",
    "foley": "soundscape",
    "acoustics": "soundscape",
}

# Neutral per-slot baseline variations (used when the user provides only target
# words/phrases without Pole A; the compiler automatically selects variants that
# share zero content words with the user's target words so nothing cancels in B - A).
_NEUTRAL_BASELINES_BY_SLOT: dict[str, tuple[str, ...]] = {
    "opening_framing": (
        "filmed with balanced neutral contrast and true-to-life tonal rendition",
        "captured under even daylight balance with uncoloured highlights",
        "rendered with clean standard exposure and accurate surface tones",
        "recorded in neutral studio balance with natural shadow detail",
        "imaged with straightforward optical clarity and balanced midtones",
        "shot with unadorned natural colorimetry and neutral highlight roll-off",
        "presented in clean reference grading with faithful shadow separation",
        "registered with transparent lens rendering and unshifted spectral balance",
        "framed in standard broadcast color balance with lifelike skin and surface fidelity",
        "exposed for neutral midtone density and clean, unclipped tonal range",
        "photographed with flat optical neutrality and undistorted luminance transitions",
        "captured in unstylized reference light with crisp, natural contrast",
    ),
    "subject_behavior": (
        "maintains a composed, unhurried posture and steady hand coordination",
        "keeps a relaxed upright stance with calm, deliberate movements",
        "works with steady composure and natural, unforced physical poise",
        "holds a calm, attentive bearing with smooth, ordinary gestures",
        "moves with relaxed, everyday coordination and balanced posture",
        "carries out the task with measured, unhurried physical steadiness",
        "stands with natural ease and unforced, practical hand motions",
        "proceeds with quiet physical assurance and smooth motor control",
        "conducts the action with settled posture and ordinary, fluid pacing",
        "remains physically grounded with relaxed shoulders and even timing",
        "operates with straightforward bodily balance and calm, direct focus",
        "attends to the moment with unstrained posture and steady, natural motion",
    ),
    "interpersonal_dynamics": (
        "Everything nearby remains calm and undisturbed",
        "Every loose item and fixture stays settled in normal equilibrium",
        "The interior setting remains steady and ordinary",
        "Nearby surfaces and props rest in quiet, unperturbed stillness",
        "The physical setting stays stable and unshifting",
        "All furnishings and tabletop items sit under normal terrestrial conditions",
        "The ambient indoor setting holds a tranquil, unagitated balance",
        "Objects and fixtures remain anchored in ordinary stillness",
        "The studio architecture and indoor air stay completely placid and regular",
        "Everything in the immediate vicinity rests in undisturbed, everyday stability",
        "Nearby props and structural surfaces hold their normal, motionless state",
        "The spatial atmosphere remains grounded and physically serene",
    ),
    "camera_motion": (
        "holds a locked-off stationary frame on a rigid mount",
        "remains fixed in a steady static composition",
        "stays motionless on a leveled tripod head",
        "maintains an unmoving locked perspective",
        "sits planted in a still, unshifting frame",
        "rests on a geared studio head with zero pan or tilt drift",
        "anchors the composition from a dead-steady fixed viewpoint",
        "keeps a rock-solid static framing throughout the shot",
        "holds an unwavering, tripod-locked optical axis",
        "preserves a completely still, stabilized vantage point",
        "locks the field of view in a balanced, unmoving mount",
        "maintains a rigid, vibration-free stationary framing",
    ),
    "speech_delivery": (
        "says in a calm, even conversational cadence",
        "speaks in a clear, unhurried everyday register",
        "delivers the words in a plain, steady speaking voice",
        "remarks in a relaxed, matter-of-fact tone",
        "states with a composed, neutral vocal delivery",
        "addresses the listener in a balanced, natural pitch",
        "articulates the thought with relaxed, ordinary projection",
        "offers the remark in an unforced, level conversational manner",
        "converses with measured, everyday clarity and even pacing",
        "relays the statement in a settled, conversational timbre",
        "notes in a straightforward, unembellished register",
        "communicates with calm, natural chest resonance",
    ),
    "soundscape": (
        "Balanced indoor recording carries clear, dry close-perspective presence.",
        "Calm, unaltered interior air preserves quiet natural presence.",
        "Clean close-mic studio capture registers subtle everyday stillness.",
        "Neutral dry indoor ambience carries unadorned quiet.",
        "Even, natural background tone sits quietly beneath the scene.",
        "Direct, uncolored interior pickup maintains a natural close-up stage.",
        "Quiet, well-damped indoor air holds a clean, lifelike floor.",
        "Standard close-perspective recording captures plain, undistorted presence.",
        "Natural indoor ambiance remains dry, balanced, and free of coloration.",
        "Subtle, unhurried background tone provides a transparent backdrop.",
        "Clear, intimate interior sound preserves faithful everyday stillness.",
        "Unprocessed indoor ambience rests at a calm, natural listening level.",
    ),
}

# Neutral bare noun-phrase baselines per slot (paired inside the exact same
# `_BARE_TARGET_WRAPPERS_BY_SLOT[slot][vary_idx]` wrapper as a bare Pole B target
# so the grammatical lead-in sits inside the shared prefix `0..first_diff` and
# `diff_span` (`[first_diff:end_diff]`) pools 100% the bare target vs. bare baseline
# with zero wrapper token dilution).
_NEUTRAL_BARE_BASELINES_BY_SLOT: dict[str, tuple[str, ...]] = {
    "opening_framing": (
        "balanced neutral contrast and true-to-life tonal rendition",
        "even daylight balance and uncoloured highlights",
        "clean standard exposure and accurate surface tones",
        "neutral studio balance and natural shadow detail",
        "straightforward optical clarity and balanced midtones",
        "unadorned natural colorimetry and neutral highlight roll-off",
        "clean reference grading and faithful shadow separation",
        "transparent lens rendering and unshifted spectral balance",
        "standard broadcast color balance and lifelike surface fidelity",
        "neutral midtone density and unclipped tonal range",
        "flat optical neutrality and undistorted luminance transitions",
        "unstylized reference light and crisp natural contrast",
    ),
    "subject_behavior": (
        "composed, unhurried posture and steady hand coordination",
        "relaxed upright stance and calm, deliberate movements",
        "steady composure and natural, unforced physical poise",
        "calm, attentive bearing and smooth, ordinary gestures",
        "relaxed, everyday coordination and balanced posture",
        "measured, unhurried physical steadiness and even poise",
        "natural ease and unforced, practical hand motions",
        "quiet physical assurance and smooth motor control",
        "settled posture and ordinary, fluid pacing",
        "grounded physical balance and relaxed shoulders",
        "straightforward bodily poise and calm, direct focus",
        "unstrained posture and steady, natural motion",
    ),
    "interpersonal_dynamics": (
        "calm, undisturbed indoor equilibrium",
        "settled, everyday physical stability",
        "steady, ordinary terrestrial conditions",
        "quiet, unperturbed interior stillness",
        "stable, unshifting room balance",
        "normal, grounded spatial conditions",
        "tranquil, unagitated indoor calm",
        "ordinary, anchored physical stillness",
        "placid, regular interior equilibrium",
        "undisturbed, everyday spatial stability",
        "motionless, normal room conditions",
        "grounded, physically serene stillness",
    ),
    "camera_motion": (
        "locked-off stationary framing on a rigid mount",
        "fixed, steady static composition",
        "motionless, leveled tripod framing",
        "unmoving locked optical perspective",
        "still, unshifting studio framing",
        "geared studio mounting with zero pan or tilt drift",
        "dead-steady fixed viewpoint",
        "rock-solid static composition",
        "unwavering, tripod-locked optical alignment",
        "completely still, stabilized vantage framing",
        "balanced, unmoving camera mounting",
        "rigid, vibration-free stationary perspective",
    ),
    "speech_delivery": (
        "calm, even conversational cadence",
        "clear, unhurried everyday register",
        "plain, steady speaking voice",
        "relaxed, matter-of-fact vocal tone",
        "composed, neutral vocal delivery",
        "balanced, natural speaking pitch",
        "relaxed, ordinary vocal projection",
        "unforced, level conversational manner",
        "measured, everyday vocal clarity",
        "settled, conversational speaking timbre",
        "straightforward, unembellished vocal register",
        "calm, natural chest resonance",
    ),
    "soundscape": (
        "clear, dry close-perspective indoor presence",
        "quiet, unaltered natural interior presence",
        "clean close-mic studio stillness",
        "neutral dry indoor quiet",
        "even, natural background room tone",
        "direct, uncolored interior pickup",
        "quiet, well-damped indoor room tone",
        "standard close-perspective undistorted presence",
        "dry, balanced natural indoor ambiance",
        "subtle, transparent background room tone",
        "clear, intimate everyday interior stillness",
        "unprocessed, calm indoor listening presence",
    ),
}

# Target-phrase templates used when a user passes bare words/nouns/adjectives
# (e.g., "handheld camera shake", "1974 16mm film grain", "cathedral reverb").
# Every template is a prefix lead-in ending at `{target}` (or `{target}.` for
# sentence-final slots) and is applied identically to both Pole A and Pole B in
# pair `i` so the grammatical wrapper sits in the shared prefix (`0..first_diff`)
# and `diff_span` pools 100% `{target}` with zero wrapper-word dilution.
_BARE_TARGET_WRAPPERS_BY_SLOT: dict[str, tuple[str, ...]] = {
    "opening_framing": (
        "rendered with {target}",
        "filmed with {target}",
        "captured with {target}",
        "steeped in {target}",
        "styled with {target}",
        "imaged with {target}",
        "graded for {target}",
        "exposed with {target}",
        "photographed with {target}",
        "presented with {target}",
    ),
    "subject_behavior": (
        "exhibits {target}",
        "displays {target}",
        "shows {target}",
        "manifests {target}",
        "acts with {target}",
        "demonstrates {target}",
        "reveals {target}",
        "performs with {target}",
        "behaves with {target}",
        "reacts with {target}",
    ),
    "interpersonal_dynamics": (
        "Across {surface}, {props} and the room are overtaken by {target}",
        "Around {surface}, {props} and the air experience {target}",
        "Throughout the take, {props} near {surface} manifest {target}",
        "All surrounding elements and {props} on {surface} undergo {target}",
        "The area around {surface} and {props} is filled with {target}",
        "Every object in sight, including {props} on {surface}, is subjected to {target}",
        "Across {surface}, {props} and the entire space are swept up in {target}",
        "Around {props} and {surface}, the location shifts into {target}",
        "Adjacent air currents and {props} along {surface} are governed by {target}",
        "The whole space around {props} and {surface} operates under {target}",
    ),
    "camera_motion": (
        "moves with {target}",
        "operates with {target}",
        "tracks with {target}",
        "shifts with {target}",
        "frames the action with {target}",
        "captures the scene with {target}",
        "follows the subject with {target}",
        "behaves with {target}",
        "is driven by {target}",
        "reacts with {target}",
    ),
    "speech_delivery": (
        "intones the phrase with {target}",
        "voices the sentence with {target}",
        "utters the line with {target}",
        "projects the dialogue with {target}",
        "renders the speech with {target}",
        "expresses the words with {target}",
        "carries the vocal line with {target}",
        "delivers the dialogue with {target}",
        "pronounces the line with {target}",
        "enunciates the phrase with {target}",
    ),
    "soundscape": (
        "The sonic field is filled with {target}.",
        "The acoustic space is dominated by {target}.",
        "Every sound in the space is shaped by {target}.",
        "The auditory perspective is defined by {target}.",
        "The sonic environment resonates with {target}.",
        "Across the take, the soundscape carries {target}.",
        "The entire soundstage is enveloped in {target}.",
        "Every sonic detail is colored by {target}.",
        "Throughout the shot, the acoustic field registers {target}.",
        "The audio profile is saturated with {target}.",
    ),
}

_SLOT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "camera_motion",
        (
            "camera", "handheld", "tripod", "whip-pan", "whip pan", "zoom", "dolly",
            "tilt", "pedestal", "tracking shot", "arc shot", "dutch angle", "roll",
            "gimbal", "steadicam", "lens bob", "frame shake", "camera shake",
            "crash zoom", "dolly zoom", "vertigo", "rack focus", "orbital",
        ),
    ),
    (
        "speech_delivery",
        (
            "whisper", "whispering", "bellow", "bellowing", "operatic", "shout",
            "shouting", "scream", "patter", "auctioneer", "monotone", "deadpan",
            "stutter", "trembling voice", "vocal", "raspy", "hoarse", "falsetto",
            "laughing voice", "suppressed laughter", "breathless voice", "drawl",
            "slurred", "sarcastic tone", "sob", "crying voice", "asmr",
            "helium", "chipmunk", "high-pitched voice", "robotic voice", "chant",
        ),
    ),
    (
        "soundscape",
        (
            "reverb", "echo", "echoing", "acoustics", "foley", "soundscape",
            "muffled", "low-pass", "cathedral", "steam hiss", "hiss", "hum",
            "rumble", "siren", "thunder", "wind howl", "footsteps", "creak",
            "silence", "dead air", "underwater sound", "earplug", "tinny",
            "walkie-talkie", "radio static", "crackling",
        ),
    ),
    (
        "subject_behavior",
        (
            "gaze", "eye contact", "eyes", "blink", "stare", "posture", "slouch",
            "flinch", "recoil", "lean in", "leaning", "nod", "head swell", "cranium",
            "chest swell", "torso", "neck stretch", "jaw", "smile", "frown",
            "shiver", "tremble", "twitch", "gesture", "hands", "shoulders",
            "hesitat", "fidget", "shrug", "grimace", "wince",
        ),
    ),
    (
        "interpersonal_dynamics",
        (
            "gravity", "zero-gravity", "weightless", "levitat", "float", "floating",
            "wind tunnel", "gale", "hurricane", "earthquake", "tremor", "underwater",
            "submerged", "shockwave", "magnetic", "time freeze", "slow motion",
            "explosion", "debris", "smoke", "fog", "blizzard", "downpour",
            "snowdrift", "sandstorm", "tornado", "aquatic", "bubbles", "fireflies",
            "snowstorm", "whiteout", "whirlwind", "vortex", "confetti", "hailstorm",
        ),
    ),
)

_FINITE_VERB_START_RE = re.compile(
    r"^(?:holds|remains|stays|maintains|sits|keeps|rests|anchors|shakes|jolts|lurches|"
    r"quivers|wobbles|shudders|pitches|jostles|moves|pans|tilts|zooms|pushes|pulls|"
    r"trucks|pedestals|rolls|tracks|arcs|says|speaks|whispers|bellows|shouts|murmurs|"
    r"mutters|stammers|laughs|sobs|rasps|chants|snaps|drawls|darts|avoids|locks|"
    r"leans|recoils|flinches|slouches|stiffens|trembles|twitches|freezes|inflates|"
    r"swells|stretches|melts|crystallizes|glitches|filmed|captured|rendered|recorded|"
    r"lit|bathed|drenched|graded|illuminated|washed|tinted|veiled|softened|"
    r"wrapped|diffused|blurred|etched|imaged|exposed|set)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContrastClause:
    a: str
    b: str
    slot: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class HoldClause:
    a: str
    b: str
    group: str = "hold_1"
    slot: str | None = None
    weight: float = 1.0


def _split_contrast_str(text: str) -> tuple[str, str] | None:
    """Split a `'pole_a ||| pole_b'` or `'pole_a -> pole_b'` string if present."""
    for sep in ("|||", " -> "):
        if sep in text:
            parts = [p.strip() for p in text.split(sep, 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"expected 'pole_a {sep.strip()} pole_b' with non-empty sides, got {text!r}"
                )
            return parts[0], parts[1]
    return None


def infer_slot_from_words(text: str) -> str:
    """Infer the best structural H3 slot from a short word list or phrase."""
    lower = re.sub(r"[-_]+", " ", text.strip().lower())
    for slot_name, keywords in _SLOT_KEYWORDS:
        for kw in keywords:
            kw_norm = re.sub(r"[-_]+", " ", kw.lower())
            if kw_norm in lower:
                return slot_name
    return "opening_framing"


def resolve_slot_name(slot: str, *, sample_text: str = "") -> str:
    """Map a canonical slot name or alias to one of `CANONICAL_SLOTS` (supports `'auto'`)."""
    if not isinstance(slot, str) or not slot.strip():
        raise ValueError(f"slot must be a non-empty string; expected one of {CANONICAL_SLOTS}")
    key = slot.strip().lower()
    if key not in SLOT_ALIASES:
        raise ValueError(
            f"unknown slot {slot!r}; expected 'auto' or one of {CANONICAL_SLOTS} "
            f"(or aliases {sorted(SLOT_ALIASES.keys())})"
        )
    resolved = SLOT_ALIASES[key]
    if resolved == "auto":
        return infer_slot_from_words(sample_text)
    return resolved


def load_canonical_frames(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate `prompt_bank/control_vectors/canonical_frames.json`."""
    p = Path(path) if path is not None else DEFAULT_CANONICAL_FRAMES_PATH
    if not p.is_file():
        raise FileNotFoundError(f"canonical frames file does not exist: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "h3-canonical-frames/1":
        raise ValueError(
            f"{p}: expected format 'h3-canonical-frames/1', got {data.get('format')!r}"
        )
    defaults = data.get("slot_defaults")
    if not isinstance(defaults, dict) or set(defaults.keys()) != set(CANONICAL_SLOTS):
        raise ValueError(
            f"{p}: slot_defaults must define exactly {sorted(CANONICAL_SLOTS)}"
        )
    frames_by_mode = data.get("frames")
    if not isinstance(frames_by_mode, dict):
        raise ValueError(f"{p}: 'frames' must be a dict keyed by mode ('t2va', 'ref2va')")
    for mode in ("t2va", "ref2va"):
        flist = frames_by_mode.get(mode)
        if not isinstance(flist, list) or len(flist) < 4:
            raise ValueError(f"{p}: frames[{mode!r}] must be a list of at least 4 frames")
        for idx, entry in enumerate(flist):
            if not isinstance(entry, dict):
                raise ValueError(f"{p}: frames[{mode!r}][{idx}] must be a dict")
            for req in (
                "id",
                "domain",
                "subj",
                "subj_lower",
                "poss",
                "obj",
                "refl",
                "props",
                "surface",
                "template",
            ):
                if not isinstance(entry.get(req), str) or not entry[req].strip():
                    raise ValueError(
                        f"{p}: frames[{mode!r}][{idx}] missing non-empty string {req!r}"
                    )
            tmpl = entry["template"]
            for slot_key in CANONICAL_SLOTS:
                placeholder = "{" + slot_key + "}"
                if placeholder not in tmpl:
                    raise ValueError(
                        f"{p}: frames[{mode!r}][{idx}] ({entry['id']}) missing placeholder {placeholder}"
                    )
    return data


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z]{4,}", text.lower())
        if w not in _cv._CANCEL_STOPLIST
    }


def _edge_words(text: str) -> tuple[str, str]:
    toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return ("", "")
    return (toks[0], toks[-1])


def _is_bare_target(target: str, slot: str) -> bool:
    """Return True when `target` is a bare word list or noun phrase that needs a slot wrapper."""
    cleaned = target.strip()
    if slot in ("interpersonal_dynamics", "soundscape"):
        return not cleaned.endswith((".", "!", "?"))
    return not bool(_FINITE_VERB_START_RE.match(cleaned))


def _pick_non_overlapping_baseline(
    slot: str,
    target_clause: str,
    index: int,
    *,
    used_baselines: set[str] | None = None,
    bare: bool = False,
) -> str:
    """Pick a neutral Pole A for `slot` that shares zero `>=4`-char content words (and no boundary words) with `target_clause`."""
    candidates = _NEUTRAL_BARE_BASELINES_BY_SLOT[slot] if bare else _NEUTRAL_BASELINES_BY_SLOT[slot]
    target_words = _content_words(target_clause)
    t_first, t_last = _edge_words(target_clause)
    used = used_baselines if used_baselines is not None else set()
    n = len(candidates)

    def _edges_disjoint(cand: str) -> bool:
        c_first, c_last = _edge_words(cand)
        return (not t_first or c_first != t_first) and (not t_last or c_last != t_last)

    for offset in range(n):
        cand = candidates[(index + offset) % n]
        if cand not in used and not (_content_words(cand) & target_words) and _edges_disjoint(cand):
            used.add(cand)
            return cand
    for offset in range(n):
        cand = candidates[(index + offset) % n]
        if not (_content_words(cand) & target_words) and _edges_disjoint(cand):
            used.add(cand)
            return cand
    for offset in range(n):
        cand = candidates[(index + offset) % n]
        if not (_content_words(cand) & target_words):
            used.add(cand)
            return cand
    # Fallback: strip any overlapping content word from candidate[index] if needed
    base = candidates[index % n]
    cleaned_tokens = [
        tok for tok in base.split() if re.sub(r"[^a-z]", "", tok.lower()) not in target_words
    ]
    res = " ".join(cleaned_tokens) if cleaned_tokens else "normal neutral state"
    used.add(res)
    return res


def _wrap_bare_target_if_needed(target: str, slot: str, index: int, *, offset: int = 0) -> str:
    """If `target` is a bare word list or noun phrase, wrap it into a grammatical slot clause."""
    cleaned = target.strip()
    if not _is_bare_target(cleaned, slot):
        return cleaned
    wrappers = _BARE_TARGET_WRAPPERS_BY_SLOT[slot]
    w_idx = (index + offset) % len(wrappers)
    return wrappers[w_idx].replace("{target}", cleaned.rstrip(".,;:"))


def _clean_slot_clause(text: str, slot: str, frame_entry: Mapping[str, str]) -> str:
    """Bind per-frame pronouns (`{poss}`, `{obj}`, `{refl}`) and props (`{props}`, `{surface}`) and normalize slot punctuation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"contrast clause for slot {slot!r} must be a non-empty string")
    props_val = frame_entry.get("props", "nearby objects")
    surface_val = frame_entry.get("surface", "the nearby surface")
    bound = (
        text.strip()
        .replace("{poss}", frame_entry["poss"])
        .replace("{Poss}", frame_entry["poss"].capitalize())
        .replace("{obj}", frame_entry["obj"])
        .replace("{refl}", frame_entry["refl"])
        .replace("{props}", props_val)
        .replace("{Props}", props_val[:1].upper() + props_val[1:] if props_val else "")
        .replace("{surface}", surface_val)
        .replace("{Surface}", surface_val[:1].upper() + surface_val[1:] if surface_val else "")
    )
    if slot == "opening_framing":
        bound = bound.rstrip(".,;:")
    elif slot == "subject_behavior":
        for lead in ("He ", "he ", "She ", "she ", "<Subject 1> "):
            if bound.startswith(lead):
                bound = bound[len(lead) :].lstrip()
                break
        bound = bound.rstrip(".,;:")
    elif slot == "interpersonal_dynamics":
        bound = bound.rstrip(".")
    elif slot == "camera_motion":
        for lead in ("The camera ", "the camera "):
            if bound.startswith(lead):
                bound = bound[len(lead) :].lstrip()
                break
        bound = bound.rstrip(".")
    elif slot == "speech_delivery":
        for lead in ("(S1) ", "He ", "he ", "She ", "she ", "<Subject 1> "):
            if bound.startswith(lead):
                bound = bound[len(lead) :].lstrip()
                break
        bound = bound.rstrip(".,;:")
    elif slot == "soundscape":
        if not bound.endswith((".", "!", "?")):
            bound = bound + "."
    return bound


def _render_frame_prompt(
    frame_entry: Mapping[str, str],
    slot_defaults: Mapping[str, str],
    slot_override: str,
    clause_value: str,
    *,
    extra_overrides: Mapping[str, str] | None = None,
) -> str:
    vals = dict(slot_defaults)
    vals[slot_override] = _clean_slot_clause(clause_value, slot_override, frame_entry)
    if extra_overrides:
        for k_slot, v_clause in extra_overrides.items():
            vals[k_slot] = _clean_slot_clause(v_clause, k_slot, frame_entry)
    out = frame_entry["template"].replace("{Subj}", frame_entry["subj"]).replace(
        "{subj}", frame_entry["subj_lower"]
    )
    for k, v in vals.items():
        out = out.replace("{" + k + "}", v)
    return out


_HOLD_PREFIX_RE = re.compile(
    r"^hold(?:\(([^)]+)\)|_([a-z0-9_]+))?[ \t]*:[ \t]*(.+)$",
    flags=re.IGNORECASE,
)


def _split_inline_segments(text: str) -> list[str]:
    """Split a string on newlines or `;` delimiters outside `<d>...</d>` tags."""
    if "\n" not in text and ";" not in text:
        return [text]
    segments: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ";" in line and "<d>" not in line.lower():
            parts = [p.strip() for p in line.split(";") if p.strip()]
            segments.extend(parts)
        else:
            segments.append(line)
    return segments


def _extract_clauses_and_holds(
    contrasts: Sequence[ContrastClause | HoldClause | tuple[str, str] | str | Mapping[str, Any]] | str,
    default_slot: str,
) -> tuple[list[ContrastClause], list[ContrastClause], list[HoldClause]]:
    """Parse `contrasts` into `(main_clauses, inline_soundscape_clauses, inline_hold_clauses)`."""
    if isinstance(contrasts, str):
        stripped = contrasts.strip()
        if not stripped:
            raise ValueError("contrasts must be a non-empty string")
        raw_seq: list[Any] = _split_inline_segments(stripped)
    elif isinstance(contrasts, Sequence) and len(contrasts) > 0:
        raw_seq = []
        for elem in contrasts:
            if isinstance(elem, str) and (";" in elem or "\n" in elem):
                parts = _split_inline_segments(elem)
                raw_seq.extend(parts if parts else [elem])
            else:
                raw_seq.append(elem)
    else:
        raise ValueError(
            "contrasts must be a non-empty string or sequence of target phrases / (a, b) pairs"
        )

    main_out: list[ContrastClause] = []
    sound_out: list[ContrastClause] = []
    hold_out: list[HoldClause] = []

    for idx, item in enumerate(raw_seq):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                raise ValueError(f"contrasts[{idx}] must be a non-empty string")
            m_hold = _HOLD_PREFIX_RE.match(text)
            if m_hold is not None:
                raw_grp = m_hold.group(1) or m_hold.group(2) or f"hold_{len(hold_out) + 1}"
                grp = re.sub(r"[^a-z0-9_]+", "_", raw_grp.strip().lower()).strip("_") or f"hold_{len(hold_out) + 1}"
                if not grp[0].isalpha():
                    grp = "h_" + grp
                body = m_hold.group(3).strip()
                split_ab = _split_contrast_str(body)
                if split_ab is not None:
                    a_raw, b_raw = split_ab
                    slot_res = resolve_slot_name("auto", sample_text=b_raw)
                    hold_out.append(HoldClause(a=a_raw, b=b_raw, group=grp, slot=slot_res, weight=1.0))
                else:
                    slot_res = resolve_slot_name("auto", sample_text=body)
                    hold_out.append(HoldClause(a="", b=body, group=grp, slot=slot_res, weight=1.0))
                continue

            slot_prefix: str | None = None
            body = text
            for prefix_candidate in tuple(SLOT_ALIASES.keys()):
                tag = prefix_candidate + ":"
                if body.lower().startswith(tag):
                    slot_prefix = SLOT_ALIASES[prefix_candidate]
                    body = body[len(tag) :].strip()
                    break

            eff_slot = slot_prefix if slot_prefix is not None else default_slot
            split_ab = _split_contrast_str(body)
            if split_ab is not None:
                a_raw, b_raw = split_ab
                slot_resolved = resolve_slot_name(eff_slot, sample_text=b_raw)
                clause = ContrastClause(a=a_raw, b=b_raw, slot=slot_resolved, weight=1.0)
            else:
                slot_resolved = resolve_slot_name(eff_slot, sample_text=body)
                clause = ContrastClause(a="", b=body, slot=slot_resolved, weight=1.0)

            if slot_prefix == "soundscape":
                sound_out.append(clause)
            else:
                main_out.append(clause)
        elif isinstance(item, HoldClause):
            slot_resolved = resolve_slot_name(
                item.slot if item.slot is not None else "auto",
                sample_text=item.b,
            )
            a_val = item.a.strip() if item.a and item.a.strip() else ""
            grp = re.sub(r"[^a-z0-9_]+", "_", (item.group or f"hold_{len(hold_out) + 1}").strip().lower()).strip("_")
            if not grp or not grp[0].isalpha():
                grp = "h_" + (grp or f"{len(hold_out) + 1}")
            hold_out.append(
                HoldClause(
                    a=a_val,
                    b=str(item.b).strip(),
                    group=grp,
                    slot=slot_resolved,
                    weight=float(item.weight),
                )
            )
        elif isinstance(item, ContrastClause):
            slot_resolved = resolve_slot_name(
                item.slot if item.slot is not None else default_slot,
                sample_text=item.b,
            )
            a_val = item.a.strip() if item.a and item.a.strip() else ""
            main_out.append(
                ContrastClause(
                    a=a_val,
                    b=str(item.b).strip(),
                    slot=slot_resolved,
                    weight=float(item.weight),
                )
            )
        elif isinstance(item, tuple) and len(item) == 2:
            b_val = str(item[1]).strip()
            slot_resolved = resolve_slot_name(default_slot, sample_text=b_val)
            a_val = str(item[0]).strip() if item[0] and str(item[0]).strip() else ""
            main_out.append(
                ContrastClause(
                    a=a_val,
                    b=b_val,
                    slot=slot_resolved,
                    weight=1.0,
                )
            )
        elif isinstance(item, Mapping):
            if "b" not in item:
                raise ValueError(f"contrasts[{idx}] mapping must contain at least 'b' (or 'a' and 'b')")
            b_val = str(item["b"]).strip()
            slot_raw = item.get("slot", default_slot)
            slot_resolved = resolve_slot_name(str(slot_raw), sample_text=b_val)
            a_raw = item.get("a", "")
            a_val = str(a_raw).strip() if (a_raw and str(a_raw).strip()) else ""
            w = float(item.get("weight", 1.0))
            if item.get("hold") or item.get("group"):
                raw_grp = str(item.get("group") or f"hold_{len(hold_out) + 1}")
                grp = re.sub(r"[^a-z0-9_]+", "_", raw_grp.strip().lower()).strip("_") or f"hold_{len(hold_out) + 1}"
                if not grp[0].isalpha():
                    grp = "h_" + grp
                hold_out.append(
                    HoldClause(a=a_val, b=b_val, group=grp, slot=slot_resolved, weight=w)
                )
            else:
                main_out.append(
                    ContrastClause(
                        a=a_val,
                        b=b_val,
                        slot=slot_resolved,
                        weight=w,
                    )
                )
        else:
            raise ValueError(
                f"contrasts[{idx}] must be a target string, (pole_a, pole_b) tuple, Mapping, ContrastClause, or HoldClause, got {type(item).__name__}"
            )
    if not main_out and sound_out:
        main_out = list(sound_out)
        sound_out = []
    if not main_out:
        raise ValueError("contrasts must include at least one non-hold target phrase or contrast clause")
    return main_out, sound_out, hold_out


def _normalize_clause_sequence(
    contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | str,
    default_slot: str,
) -> list[ContrastClause]:
    main_out, _sound_out, _hold_out = _extract_clauses_and_holds(contrasts, default_slot)
    return main_out


def _resolve_clause_endpoints(
    raw_clause: ContrastClause,
    active_slot: str,
    vary_idx: int,
    used_baselines: set[str],
    *,
    vary_realizations: bool = False,
) -> tuple[str, str]:
    """Return `(a_clause, b_clause)` for `raw_clause` at frame index `vary_idx`."""
    b_is_bare = _is_bare_target(raw_clause.b, active_slot)
    b_offset = 1 if vary_realizations else 0
    a_raw_clean = raw_clause.a
    if a_raw_clean:
        b_words = _content_words(raw_clause.b)
        if "{props}" in a_raw_clean.lower() and "{props}" in raw_clause.b.lower():
            prop_alt = "ordinary furnishings" if "furnishings" not in b_words else "stationary items"
            a_raw_clean = a_raw_clean.replace("{props}", prop_alt).replace("{Props}", prop_alt.capitalize())
        if "{surface}" in a_raw_clean.lower() and "{surface}" in raw_clause.b.lower():
            surf_alt = "the immediate area" if "immediate" not in b_words else "the resting plane"
            a_raw_clean = a_raw_clean.replace("{surface}", surf_alt).replace("{Surface}", surf_alt.capitalize())
    b_clause = _wrap_bare_target_if_needed(raw_clause.b, active_slot, vary_idx, offset=b_offset)
    if a_raw_clean:
        a_clause = _wrap_bare_target_if_needed(a_raw_clean, active_slot, vary_idx, offset=0)
    elif b_is_bare:
        a_bare = _pick_non_overlapping_baseline(
            active_slot,
            raw_clause.b,
            vary_idx,
            used_baselines=used_baselines,
            bare=True,
        )
        a_clause = _wrap_bare_target_if_needed(a_bare, active_slot, vary_idx, offset=0)
    else:
        a_clause = _pick_non_overlapping_baseline(
            active_slot,
            b_clause,
            vary_idx,
            used_baselines=used_baselines,
            bare=False,
        )
    if vary_realizations and not _is_bare_target(raw_clause.b, active_slot):
        # When vary_realizations=True on a finite-verb clause without {props}/{surface},
        # append a per-frame scene anchor so all 10 b_spans are distinct.
        if "{props}" not in b_clause and "{surface}" not in b_clause:
            b_clause = b_clause.rstrip(".") + " near {surface}"
        if "{props}" not in a_clause and "{surface}" not in a_clause:
            a_clause = a_clause.rstrip(".") + " near {surface}"
    return a_clause, b_clause


def _load_target_prompt_text(target_prompt: str | Path) -> tuple[str, str | None]:
    """Return `(prompt_text, shot_stem_if_file)` from a Path, file-path string, or raw prompt text."""
    if isinstance(target_prompt, Path):
        if not target_prompt.is_file():
            raise FileNotFoundError(f"target_prompt file not found: {target_prompt}")
        return target_prompt.read_text(encoding="utf-8").strip(), target_prompt.stem
    s = str(target_prompt).strip()
    if not s:
        raise ValueError("target_prompt must be a non-empty string or Path")
    if "\n" not in s:
        for cand in (Path(s), _REPO_ROOT / s, _REPO_ROOT / "prompt_bank" / s, _REPO_ROOT / "prompt_bank" / f"{s}.txt"):
            if cand.is_file():
                return cand.read_text(encoding="utf-8").strip(), cand.stem
    return s, None


def _build_target_prompt_pair(
    target_text: str,
    *,
    active_slot: str,
    a_clause: str,
    b_clause: str,
    raw_clause: ContrastClause,
    sound_ab: tuple[str, str] | None = None,
    weight: float = 2.5,
) -> dict[str, Any]:
    """Construct Pair 0 directly inside `target_text` at `weight`."""
    dummy_frame = {
        "id": "target_scene_00",
        "domain": "target_prompt",
        "subj": "The subject",
        "subj_lower": "the subject",
        "poss": "their",
        "obj": "them",
        "refl": "themselves",
        "props": "the tabletop objects",
        "surface": "the main worktop",
    }
    clean_a = _clean_slot_clause(a_clause, active_slot, dummy_frame)
    clean_b = _clean_slot_clause(b_clause, active_slot, dummy_frame)

    if raw_clause.a and raw_clause.a in target_text:
        prompt_a = target_text
        prompt_b = target_text.replace(raw_clause.a, raw_clause.b, 1)
    elif active_slot == "soundscape":
        if "overall_soundscape:" in target_text:
            prompt_a = target_text.replace("overall_soundscape:", f"overall_soundscape: {clean_a}", 1)
            prompt_b = target_text.replace("overall_soundscape:", f"overall_soundscape: {clean_b}", 1)
        else:
            prompt_a = f"{target_text.rstrip()} {clean_a}"
            prompt_b = f"{target_text.rstrip()} {clean_b}"
    else:
        if active_slot == "opening_framing":
            sent_a = f"The shot is {clean_a}."
            sent_b = f"The shot is {clean_b}."
        elif active_slot == "camera_motion":
            sent_a = f"The camera {clean_a}."
            sent_b = f"The camera {clean_b}."
        elif active_slot == "subject_behavior":
            sent_a = f"The subject {clean_a}."
            sent_b = f"The subject {clean_b}."
        elif active_slot == "speech_delivery":
            sent_a = f"In delivery, the speaker {clean_a}."
            sent_b = f"In delivery, the speaker {clean_b}."
        else:
            sent_a = f"{clean_a}."
            sent_b = f"{clean_b}."

        if "\n\noverall_soundscape:" in target_text:
            prompt_a = target_text.replace("\n\noverall_soundscape:", f" {sent_a}\n\noverall_soundscape:", 1)
            prompt_b = target_text.replace("\n\noverall_soundscape:", f" {sent_b}\n\noverall_soundscape:", 1)
        else:
            prompt_a = f"{target_text.rstrip()} {sent_a}"
            prompt_b = f"{target_text.rstrip()} {sent_b}"

    if sound_ab is not None and active_slot != "soundscape" and "overall_soundscape:" in prompt_a:
        sa_clean = _clean_slot_clause(sound_ab[0], "soundscape", dummy_frame)
        sb_clean = _clean_slot_clause(sound_ab[1], "soundscape", dummy_frame)
        prompt_a = prompt_a.replace("overall_soundscape:", f"overall_soundscape: {sa_clean}", 1)
        prompt_b = prompt_b.replace("overall_soundscape:", f"overall_soundscape: {sb_clean}", 1)

    a_span, b_span = _cv.differing_span(
        prompt_a,
        prompt_b,
        frame="scene",
        label=f"compiled target_prompt pair[0] (slot={active_slot!r})",
    )
    return {
        "index": 0,
        "frame_id": "target_scene_00",
        "domain": "target_prompt",
        "slot": f"{active_slot}+soundscape" if (sound_ab is not None and active_slot != "soundscape") else active_slot,
        "weight": float(weight),
        "a": prompt_a,
        "b": prompt_b,
        "a_span": a_span,
        "b_span": b_span,
    }


def _compile_hold_items(
    hold_clauses: Sequence[HoldClause | ContrastClause | tuple[str, str] | str | Mapping[str, Any]],
    *,
    mode: str,
    canonical_frames_path: str | Path | None = None,
    frames_per_hold: int = 2,
) -> tuple[_cv.HoldItem, ...]:
    """Compile `hold_clauses` into validated `_cv.HoldItem` instances across canonical frames."""
    if not hold_clauses:
        return ()
    bank = load_canonical_frames(canonical_frames_path)
    slot_defaults: Mapping[str, str] = bank["slot_defaults"]
    frames: list[dict[str, str]] = list(bank["frames"][mode])

    normalized_holds: list[HoldClause] = []
    for idx, item in enumerate(hold_clauses):
        if isinstance(item, HoldClause):
            normalized_holds.append(item)
        elif isinstance(item, str):
            text = item.strip()
            if not text.lower().startswith("hold"):
                text = f"hold_{idx + 1}: {text}"
            _, _, h_list = _extract_clauses_and_holds([text], "auto")
            normalized_holds.extend(h_list)
        elif isinstance(item, ContrastClause):
            normalized_holds.append(
                HoldClause(
                    a=item.a,
                    b=item.b,
                    group=f"hold_{idx + 1}",
                    slot=item.slot,
                    weight=item.weight,
                )
            )
        elif isinstance(item, tuple) and len(item) == 2:
            normalized_holds.append(
                HoldClause(
                    a=str(item[0]).strip(),
                    b=str(item[1]).strip(),
                    group=f"hold_{idx + 1}",
                    slot=resolve_slot_name("auto", sample_text=str(item[1])),
                )
            )
        elif isinstance(item, Mapping):
            b_val = str(item["b"]).strip()
            a_val = str(item.get("a", "")).strip()
            grp = str(item.get("group") or f"hold_{idx + 1}").strip()
            slot_res = resolve_slot_name(str(item.get("slot", "auto")), sample_text=b_val)
            normalized_holds.append(
                HoldClause(a=a_val, b=b_val, group=grp, slot=slot_res, weight=float(item.get("weight", 1.0)))
            )
        else:
            raise ValueError(f"unsupported hold_contrasts item: {item!r}")

    out: list[_cv.HoldItem] = []
    used_baselines: set[str] = set()
    n_per = max(1, min(int(frames_per_hold), len(frames)))
    hold_idx = 0
    for h_num, hc in enumerate(normalized_holds):
        h_slot = hc.slot or resolve_slot_name("auto", sample_text=hc.b)
        cc = ContrastClause(a=hc.a, b=hc.b, slot=h_slot, weight=hc.weight)
        for k in range(n_per):
            f_idx = (h_num * n_per + k) % len(frames)
            frame_entry = frames[f_idx]
            a_clause, b_clause = _resolve_clause_endpoints(
                cc, h_slot, f_idx, used_baselines, vary_realizations=False
            )
            prompt_a = _render_frame_prompt(frame_entry, slot_defaults, h_slot, a_clause)
            prompt_b = _render_frame_prompt(frame_entry, slot_defaults, h_slot, b_clause)
            a_span, b_span = _cv.differing_span(
                prompt_a,
                prompt_b,
                frame="scene",
                label=f"compiled hold[{hold_idx}] (group={hc.group!r}, frame={frame_entry['id']})",
            )
            out.append(
                _cv.HoldItem(
                    index=hold_idx,
                    group=hc.group,
                    weight=float(hc.weight),
                    a=prompt_a,
                    b=prompt_b,
                    a_span=a_span,
                    b_span=b_span,
                )
            )
            hold_idx += 1
    return tuple(out)


def compile_short_contrast_pairs(
    contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | str,
    *,
    slot: str = "auto",
    mode: str = "t2va",
    n_pairs: int = 10,
    soundscape_contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | None = None,
    couple_soundscape: bool = True,
    target_prompt: str | Path | None = None,
    target_weight: float = 2.5,
    vary_realizations: bool = False,
    canonical_frames_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compile simple words, short phrases, or `(a, b)` contrast pairs into `n_pairs` H3 scene pairs.

    - If `contrasts` is a single string (e.g. `"erratic handheld camera shake"`,
      `"awkward eye-contact avoidance"`, `"1974 16mm film grain"`) or a list of
      target strings without Pole A, automatically infers the structural slot
      (when `slot='auto'`), wraps bare word lists into grammatical clauses across
      the 10 canonical frames, and selects non-overlapping neutral Pole A baselines.
    - Supports `;`- and `\\n`-separated multi-clause specs, coupled `soundscape:`
      edits on the same pair (`couple_soundscape=True`), per-frame `{props}` and
      `{surface}` bindings, and optional `target_prompt` (`Pair 0` at `target_weight=2.5`).
    - Validates every compiled pair through `control_vectors.differing_span(..., frame='scene')`.
    """
    if mode not in ("t2va", "ref2va"):
        raise ValueError(f"mode must be 't2va' or 'ref2va', got {mode!r}")
    bank = load_canonical_frames(canonical_frames_path)
    slot_defaults: Mapping[str, str] = bank["slot_defaults"]
    frames: list[dict[str, str]] = list(bank["frames"][mode])
    if n_pairs < 1 or n_pairs > len(frames) + (1 if target_prompt is not None else 0):
        raise ValueError(f"n_pairs must be in 1..{len(frames)}, got {n_pairs}")

    main_clauses, inline_sound_clauses, _inline_holds = _extract_clauses_and_holds(contrasts, slot)
    explicit_sound_clauses = (
        _normalize_clause_sequence(soundscape_contrasts, "soundscape")
        if soundscape_contrasts
        else []
    )
    sound_clauses = explicit_sound_clauses + inline_sound_clauses

    compiled: list[dict[str, Any]] = []
    used_baselines: set[str] = set()
    used_sound_baselines: set[str] = set()

    n_canonical = n_pairs
    if target_prompt is not None:
        target_text, _ = _load_target_prompt_text(target_prompt)
        c0 = main_clauses[0]
        slot0 = c0.slot or resolve_slot_name(slot, sample_text=c0.b)
        a0, b0 = _resolve_clause_endpoints(
            c0, slot0, 0, used_baselines, vary_realizations=vary_realizations
        )
        sound0_ab: tuple[str, str] | None = None
        if sound_clauses and couple_soundscape and slot0 != "soundscape":
            sc0 = sound_clauses[0]
            sound0_ab = _resolve_clause_endpoints(
                sc0, "soundscape", 0, used_sound_baselines, vary_realizations=vary_realizations
            )
        compiled.append(
            _build_target_prompt_pair(
                target_text,
                active_slot=slot0,
                a_clause=a0,
                b_clause=b0,
                raw_clause=c0,
                sound_ab=sound0_ab,
                weight=target_weight,
            )
        )
        n_canonical = max(0, n_pairs - 1)

    if couple_soundscape:
        n_sound_only_frames = 0
    else:
        n_sound_only_frames = (
            2 if (sound_clauses and n_canonical >= 6) else (1 if (sound_clauses and n_canonical >= 2) else 0)
        )
    n_main_frames = n_canonical - n_sound_only_frames

    for i in range(n_canonical):
        out_idx = len(compiled)
        frame_entry = frames[i % len(frames)]
        if i < n_main_frames:
            raw_clause = main_clauses[out_idx % len(main_clauses)]
            vary_idx = out_idx
        else:
            s_idx = i - n_main_frames
            raw_clause = sound_clauses[s_idx % len(sound_clauses)]
            vary_idx = s_idx

        active_slot = raw_clause.slot or resolve_slot_name(slot, sample_text=raw_clause.b)
        a_clause, b_clause = _resolve_clause_endpoints(
            raw_clause,
            active_slot,
            vary_idx,
            used_baselines,
            vary_realizations=vary_realizations and len(main_clauses) == 1,
        )

        extra_a: dict[str, str] | None = None
        extra_b: dict[str, str] | None = None
        slot_label = active_slot
        if sound_clauses and couple_soundscape and active_slot != "soundscape":
            s_clause = sound_clauses[out_idx % len(sound_clauses)]
            sa_clause, sb_clause = _resolve_clause_endpoints(
                s_clause,
                "soundscape",
                out_idx,
                used_sound_baselines,
                vary_realizations=vary_realizations and len(sound_clauses) == 1,
            )
            extra_a = {"soundscape": sa_clause}
            extra_b = {"soundscape": sb_clause}
            slot_label = f"{active_slot}+soundscape"

        prompt_a = _render_frame_prompt(
            frame_entry, slot_defaults, active_slot, a_clause, extra_overrides=extra_a
        )
        prompt_b = _render_frame_prompt(
            frame_entry, slot_defaults, active_slot, b_clause, extra_overrides=extra_b
        )
        a_span, b_span = _cv.differing_span(
            prompt_a,
            prompt_b,
            frame="scene",
            label=f"compiled pair[{out_idx}] ({frame_entry['id']}, slot={slot_label!r})",
        )
        compiled.append(
            {
                "index": out_idx,
                "frame_id": frame_entry["id"],
                "domain": frame_entry["domain"],
                "slot": slot_label,
                "weight": float(raw_clause.weight),
                "a": prompt_a,
                "b": prompt_b,
                "a_span": a_span,
                "b_span": b_span,
            }
        )
    return compiled


def _slugify_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    parts = [
        p
        for p in slug.split("_")
        if p not in ("a", "an", "the", "with", "and", "in", "of", "to", "props", "surface")
    ]
    short = "_".join(parts[:4]) if parts else "custom_vector"
    if not short or not short[0].isalpha():
        short = "cv_" + (short or "vector")
    return short[:40].rstrip("_")


def compile_words_to_pairset(
    contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | str,
    *,
    name: str | None = None,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    slot: str = "auto",
    mode: str = "t2va",
    n_pairs: int = 10,
    soundscape_contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | None = None,
    hold_contrasts: Sequence[HoldClause | ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | None = None,
    couple_soundscape: bool = True,
    target_prompt: str | Path | None = None,
    target_weight: float = 2.5,
    vary_realizations: bool = False,
    exclude_shots: Sequence[str] = (),
    canonical_frames_path: str | Path | None = None,
) -> _cv.PairSet:
    """Compile simple words, phrases, or short contrasts directly into an in-memory `PairSet` (zero `.toml` file needed)."""
    main_clauses, _inline_sound, inline_holds = _extract_clauses_and_holds(contrasts, slot)
    compiled_pairs = compile_short_contrast_pairs(
        contrasts,
        slot=slot,
        mode=mode,
        n_pairs=n_pairs,
        soundscape_contrasts=soundscape_contrasts,
        couple_soundscape=couple_soundscape,
        target_prompt=target_prompt,
        target_weight=target_weight,
        vary_realizations=vary_realizations,
        canonical_frames_path=canonical_frames_path,
    )
    all_hold_specs: list[Any] = list(inline_holds)
    if hold_contrasts:
        all_hold_specs.extend(hold_contrasts)
    hold_items = _compile_hold_items(
        all_hold_specs,
        mode=mode,
        canonical_frames_path=canonical_frames_path,
    )

    first_a = compiled_pairs[0]["a_span"].split(" | ")[0]
    first_b = compiled_pairs[0]["b_span"].split(" | ")[0]
    first_clause = main_clauses[0]
    default_a = first_clause.a if first_clause.a else first_a
    default_b = first_clause.b if first_clause.b else first_b
    resolved_b_side = (b_side or default_b).strip()
    resolved_a_side = (a_side or default_a).strip()
    resolved_name = (name or _slugify_name(resolved_b_side)).strip()
    if not _cv.NAME_RE.match(resolved_name):
        raise ValueError(f"vector name must match [a-z0-9_]+, got {resolved_name!r}")
    resolved_quality = (quality or f"{resolved_b_side} vs. {resolved_a_side}").strip()

    pair_items = tuple(
        _cv.PairItem(
            index=p["index"],
            weight=p["weight"],
            a=p["a"],
            b=p["b"],
            a_span=p["a_span"],
            b_span=p["b_span"],
        )
        for p in compiled_pairs
    )
    digest_payload = json.dumps(
        {
            "name": resolved_name,
            "mode": mode,
            "pairs": [(p.weight, p.a, p.b) for p in pair_items],
            "holds": [(h.group, h.weight, h.a, h.b) for h in hold_items],
        },
        sort_keys=True,
    ).encode("utf-8")
    sha256 = hashlib.sha256(digest_payload).hexdigest()

    return _cv.PairSet(
        path=Path(f"<compiled:{resolved_name}>"),
        sha256=sha256,
        name=resolved_name,
        quality=resolved_quality,
        a_side=resolved_a_side,
        b_side=resolved_b_side,
        mode=mode,
        frame="scene",
        exclude_shots=tuple(exclude_shots),
        pairs=pair_items,
        holds=hold_items,
    )


def compile_short_contrast_to_toml(
    *,
    name: str,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | str,
    out_path: str | Path,
    slot: str = "auto",
    mode: str = "t2va",
    n_pairs: int = 10,
    soundscape_contrasts: Sequence[ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | None = None,
    hold_contrasts: Sequence[HoldClause | ContrastClause | tuple[str, str] | str | Mapping[str, Any]] | None = None,
    couple_soundscape: bool = True,
    target_prompt: str | Path | None = None,
    target_weight: float = 2.5,
    vary_realizations: bool = False,
    exclude_shots: Sequence[str] | None = None,
    canonical_frames_path: str | Path | None = None,
    bank_dir: str | Path | None = None,
    overwrite: bool = False,
) -> _cv.PairSet:
    """Compile simple words or short contrast clauses into a validated `.toml` pair file and return its `PairSet`."""
    target = Path(out_path)
    if target.suffix != ".toml":
        target = target.with_suffix(".toml")
    stem = target.stem
    if name != stem:
        raise ValueError(f"vector name {name!r} must match output file stem {stem!r}")
    if not _cv.NAME_RE.match(name):
        raise ValueError(f"vector name must match [a-z0-9_]+, got {name!r}")
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"pair TOML file {target.name!r} already exists at {target} "
            "(pass overwrite=True / --overwrite to replace it)"
        )

    main_clauses, _inline_sound, inline_holds = _extract_clauses_and_holds(contrasts, slot)
    compiled_pairs = compile_short_contrast_pairs(
        contrasts,
        slot=slot,
        mode=mode,
        n_pairs=n_pairs,
        soundscape_contrasts=soundscape_contrasts,
        couple_soundscape=couple_soundscape,
        target_prompt=target_prompt,
        target_weight=target_weight,
        vary_realizations=vary_realizations,
        canonical_frames_path=canonical_frames_path,
    )
    all_hold_specs: list[Any] = list(inline_holds)
    if hold_contrasts:
        all_hold_specs.extend(hold_contrasts)
    hold_items = _compile_hold_items(
        all_hold_specs,
        mode=mode,
        canonical_frames_path=canonical_frames_path,
    )

    first_a = compiled_pairs[0]["a_span"].split(" | ")[0]
    first_b = compiled_pairs[0]["b_span"].split(" | ")[0]
    first_clause = main_clauses[0]
    default_a = first_clause.a if first_clause.a else first_a
    default_b = first_clause.b if first_clause.b else first_b
    resolved_b_side = (b_side or default_b).strip()
    resolved_a_side = (a_side or default_a).strip()
    resolved_quality = (quality or f"{resolved_b_side} vs. {resolved_a_side}").strip()
    active_slot = compiled_pairs[0]["slot"]

    target_shot_stem: str | None = None
    if target_prompt is not None:
        _, target_shot_stem = _load_target_prompt_text(target_prompt)

    resolved_bank_dir = Path(bank_dir) if bank_dir is not None else (_REPO_ROOT / "prompt_bank")
    if exclude_shots is not None:
        resolved_exclude_shots = tuple(exclude_shots)
    else:
        resolved_exclude_shots = tuple(
            shot_id
            for shot_id in _cv.DEFAULT_EXCLUDE_SHOTS
            if shot_id != target_shot_stem and (resolved_bank_dir / f"{shot_id}.txt").is_file()
        )
    excl_toml = "[" + ", ".join(json.dumps(str(s), ensure_ascii=False) for s in resolved_exclude_shots) + "]"
    lines: list[str] = [
        f"# Compiled via h3_pair_compiler (slot={active_slot!r}, mode={mode!r}, n_pairs={len(compiled_pairs)})",
        "[vector]",
        f"name = {json.dumps(name, ensure_ascii=False)}",
        f"quality = {json.dumps(resolved_quality, ensure_ascii=False)}",
        f"a_side = {json.dumps(resolved_a_side, ensure_ascii=False)}",
        f"b_side = {json.dumps(resolved_b_side, ensure_ascii=False)}",
        f"mode = {json.dumps(mode, ensure_ascii=False)}",
        'frame = "scene"',
        f"exclude_shots = {excl_toml}",
        "",
    ]

    for p_item in compiled_pairs:
        lines.append(f"# Frame: {p_item['frame_id']} ({p_item['domain']}, slot={p_item['slot']})")
        lines.append("[[pair]]")
        if p_item["weight"] != 1.0:
            lines.append(f"weight = {p_item['weight']}")
        lines.append(f"a = {_cv._format_toml_multiline(p_item['a'])}")
        lines.append(f"b = {_cv._format_toml_multiline(p_item['b'])}")
        lines.append("")

    for h_item in hold_items:
        lines.append(f"# Hold group: {h_item.group}")
        lines.append("[[hold]]")
        lines.append(f"group = {json.dumps(h_item.group, ensure_ascii=False)}")
        if h_item.weight != 1.0:
            lines.append(f"weight = {h_item.weight}")
        lines.append(f"a = {_cv._format_toml_multiline(h_item.a)}")
        lines.append(f"b = {_cv._format_toml_multiline(h_item.b)}")
        lines.append("")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    try:
        return _cv.load_pairs(target, bank_dir=bank_dir)
    except Exception:
        target.unlink(missing_ok=True)
        raise


_PLACEBO_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("a small slate-grey coaster resting on the left side", "a tiny cobalt-blue saucer sitting on the right edge"),
    ("a plain linen bookmark tucked beside the corner", "a woven cotton ribbon placed along the outer margin"),
    ("an unglazed clay pebble set near the rear border", "a smooth river stone positioned by the front trim"),
    ("a matte pewter paperclip lying on the lower shelf", "a brushed brass fastener resting on the upper ledge"),
    ("a folded buff index card leaning against the base", "a flat ivory note slip propped beside the plinth"),
    ("a charcoal felt pad sitting under the far corner", "a navy wool square resting beneath the near edge"),
    ("a cedar wooden peg resting beside the left trim", "a birch dowel pin placed near the right border"),
    ("a dull zinc washer lying near the back molding", "a copper grommet sitting along the front sill"),
    ("a beige cork stopper set on the side counter", "a maroon rubber cap placed on the end table"),
    ("a pale parchment tag hanging from the lower hook", "a cream cardstock label suspended from the upper peg"),
)


def compile_placebo_pairset(
    *,
    slot: str = "opening_framing",
    mode: str = "t2va",
    n_pairs: int = 10,
    canonical_frames_path: str | Path | None = None,
) -> _cv.PairSet:
    """Build a size-matched benign placebo `PairSet` in the exact same canonical frames.

    Used to verify that a candidate control vector's visual/audio effect is stronger
    than template-level conditioning noise at matched post-refiner norm.
    """
    resolved_slot = resolve_slot_name(slot, sample_text="placebo")
    placebo_contrasts = tuple(
        ContrastClause(a=a_text, b=b_text, slot=resolved_slot)
        for a_text, b_text in _PLACEBO_CONTRASTS[:n_pairs]
    )
    return compile_words_to_pairset(
        placebo_contrasts,
        name=f"placebo_{resolved_slot}",
        quality=f"Size-matched placebo control ({resolved_slot}): benign peripheral prop swaps",
        a_side="slate-grey corner coaster",
        b_side="cobalt-blue corner saucer",
        slot=resolved_slot,
        mode=mode,
        n_pairs=n_pairs,
        canonical_frames_path=canonical_frames_path,
    )


def parse_short_contrast_spec(text: str) -> dict[str, Any]:
    """Parse a lightweight short-contrast or simple-words spec into compiler kwargs.

    Accepts any of:
      1. A single line of words (e.g. `"erratic handheld camera shake"` or
         `"awkward eye-contact avoidance"`),
      2. A single `'pole_a ||| pole_b'` or `'pole_a -> pole_b'` contrast line,
      3. Semicolon-separated or multi-line contrasts including `soundscape: a -> b`
         and `hold: a -> b` (or `hold(group): a -> b`), or
      4. Optional `key: value` header lines (`name`, `slot`, `mode`, `quality`,
         `a_side`, `b_side`, `n_pairs`, `target_prompt`, `target_weight`) + one or
         more target / `'a ||| b'` lines.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("short-contrast spec must be a non-empty string")

    header: dict[str, str] = {}
    raw_items: list[str | ContrastClause] = []
    soundscape_items: list[str | ContrastClause] = []
    hold_items: list[HoldClause] = []

    for lineno, raw_line in enumerate(_split_inline_segments(text), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        body_stripped = line.lstrip("-*0123456789.) ").strip()
        m_hold = _HOLD_PREFIX_RE.match(body_stripped)
        if m_hold is not None:
            raw_grp = m_hold.group(1) or m_hold.group(2) or f"hold_{len(hold_items) + 1}"
            grp = re.sub(r"[^a-z0-9_]+", "_", raw_grp.strip().lower()).strip("_") or f"hold_{len(hold_items) + 1}"
            if not grp[0].isalpha():
                grp = "h_" + grp
            h_body = m_hold.group(3).strip()
            if not h_body:
                raise ValueError(f"line {lineno}: 'hold:' must be non-empty")
            split_ab = _split_contrast_str(h_body)
            if split_ab is not None:
                part_a, part_b = split_ab
                hold_items.append(
                    HoldClause(
                        a=part_a,
                        b=part_b,
                        group=grp,
                        slot=resolve_slot_name("auto", sample_text=part_b),
                    )
                )
            else:
                hold_items.append(
                    HoldClause(
                        a="",
                        b=h_body,
                        group=grp,
                        slot=resolve_slot_name("auto", sample_text=h_body),
                    )
                )
            continue

        if "|||" in line or " -> " in line:
            body = body_stripped
            slot_prefix: str | None = None
            for prefix_candidate in tuple(SLOT_ALIASES.keys()):
                tag = prefix_candidate + ":"
                if body.lower().startswith(tag):
                    slot_prefix = SLOT_ALIASES[prefix_candidate]
                    body = body[len(tag) :].strip()
                    break
            split_ab = _split_contrast_str(body)
            if split_ab is None:
                raise ValueError(
                    f"line {lineno}: expected 'pole_a ||| pole_b' or 'pole_a -> pole_b' with non-empty sides, got {raw_line!r}"
                )
            part_a, part_b = split_ab
            if slot_prefix == "soundscape":
                soundscape_items.append(ContrastClause(a=part_a, b=part_b, slot="soundscape"))
            else:
                raw_items.append(ContrastClause(a=part_a, b=part_b, slot=slot_prefix))
        elif ":" in line and not line.startswith(("-", "*")):
            k, v = line.split(":", 1)
            key = k.strip().lower()
            val = v.strip()
            if key in (
                "name",
                "slot",
                "mode",
                "quality",
                "a_side",
                "b_side",
                "n_pairs",
                "target_prompt",
                "target_weight",
                "couple_soundscape",
                "vary_realizations",
            ):
                header[key] = val
                continue
            elif key in ("target", "words", "phrase", "steer"):
                if not val:
                    raise ValueError(f"line {lineno}: {key!r} must be non-empty")
                raw_items.append(val)
                continue
            elif key in SLOT_ALIASES:
                resolved_prefix = SLOT_ALIASES[key]
                if resolved_prefix == "soundscape":
                    soundscape_items.append(val)
                else:
                    raw_items.append(ContrastClause(a="", b=val, slot=resolved_prefix))
                continue
            else:
                raise ValueError(
                    f"line {lineno}: unknown header key {key!r}"
                )
        else:
            if body_stripped:
                raw_items.append(body_stripped)

    if not raw_items and soundscape_items:
        raw_items = list(soundscape_items)
        soundscape_items = []
    if not raw_items:
        raise ValueError(
            "short-contrast spec must include at least one target phrase or 'pole_a ||| pole_b' line"
        )

    slot = header.get("slot", "auto")
    mode = header.get("mode", "t2va").lower()
    if mode not in ("t2va", "ref2va"):
        raise ValueError(f"mode must be 't2va' or 'ref2va', got {mode!r}")
    n_pairs = int(header.get("n_pairs", "10"))
    target_prompt_val = header.get("target_prompt") or None
    target_weight_val = float(header.get("target_weight", "2.5"))
    couple_val = header.get("couple_soundscape", "true").strip().lower() not in ("0", "false", "no")
    vary_val = header.get("vary_realizations", "false").strip().lower() in ("1", "true", "yes")
    contrasts_arg: Any = raw_items[0] if (len(raw_items) == 1 and isinstance(raw_items[0], str)) else raw_items

    ps = compile_words_to_pairset(
        contrasts_arg,
        name=header.get("name"),
        quality=header.get("quality"),
        a_side=header.get("a_side"),
        b_side=header.get("b_side"),
        slot=slot,
        mode=mode,
        n_pairs=n_pairs,
        soundscape_contrasts=soundscape_items or None,
        hold_contrasts=hold_items or None,
        couple_soundscape=couple_val,
        target_prompt=target_prompt_val,
        target_weight=target_weight_val,
        vary_realizations=vary_val,
    )
    return {
        "name": ps.name,
        "slot": resolve_slot_name(slot, sample_text=ps.b_side),
        "mode": ps.mode,
        "n_pairs": len(ps.pairs),
        "quality": ps.quality,
        "a_side": ps.a_side,
        "b_side": ps.b_side,
        "hold_contrasts": list(hold_items),
        "holds": list(ps.holds),
        "target_prompt": target_prompt_val,
        "pair_set": ps,
    }
