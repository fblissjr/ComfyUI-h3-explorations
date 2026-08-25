#!/usr/bin/env python3
"""Compile one scene specification into every marker arm, deterministically.

`canonical/owner_authored_marker_corpus.md` is the brief. Its central
requirement is that the compared arms request the *same scene*: the rejected
generator's dialogue "positive" asked for audible speech while its "contrast"
asked for silent reading, which tests two briefs rather than one marker.

**So the arms are not written; they are derived.** One scene specification is
serialized to prompt text exactly once, and every arm is a declared
transformation of that single string. Semantic drift between arms is not
checked for here -- it is made unrepresentable, because there is only ever one
piece of prose and no arm may author its own.

An arm is a triple, not a prompt:

    (prompt bytes, tokenizer identity, model transform)

That is what lets the mean-initialised-rows arm exist. It differs from the
release-ID arm only in a model-side change, so a compiler emitting prompt text
alone could not express it, and one that loaded weights to express it would be
doing the model's work in the wrong place. **This file never loads weights and
never touches an embedding row.** It names the transform; whoever runs the arms
applies it.

The four arms:

`release_id`
    The canonical text through the tokenizer that owns the seven H3 markers,
    resolving them to their released ids.

`legacy_bpe`
    The identical bytes through a genuinely unpatched tokenizer, reconstructed
    by `audit_h3_marker_tokenization._unpatched_clip`. The brief forbids
    simulating this by spacing out the marker characters, and this does not:
    it is the real pre-patch constructor, selected by vocabulary rather than by
    a constant's name.

`stripped`
    The marker strings removed and nothing else. Text inside a paired marker
    survives byte for byte, which is what makes this the scale rather than a
    second contrast: it says what "the marker is simply absent" costs.

`mean_init_rows`
    Byte-identical to `release_id`. Declares the model-side transform that
    replaces the seven marker embedding rows with the table mean -- the
    no-training control for whether an untrained marker row carries anything
    at all. Declared only.

Every arm carries the provenance the brief requires, including an ordinary-text
alignment claim that is checkable rather than asserted: strip the markers from
any arm and the remainder must hash identically across all four.

CPU only. No CUDA, no weights, no server. The tokenizer arms need ComfyUI
importable; without it the run still emits prompts and provenance and says that
the token streams were not recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent

DIALOGUE = ("<d>", "</d>")
CAPTION = ("<|caption_start|>", "<|caption_end|>")
LYRICS = ("<|lyrics_start|>", "<|lyrics_end|>")
CUTOFF = "<|cutoff|>"
# Longest first: `</d>` must not be found inside `<d>` removal, and the paired
# markers must not be partially matched.
ALL_MARKERS = tuple(sorted(
    (DIALOGUE[0], DIALOGUE[1], CAPTION[0], CAPTION[1], LYRICS[0], LYRICS[1], CUTOFF),
    key=len, reverse=True))

RELEASE_MARKER_IDS = {
    "<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
    "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
    "<|caption_start|>": 151674, "<|caption_end|>": 151675,
}

ARMS = ("release_id", "legacy_bpe", "stripped", "mean_init_rows")

SECTIONS = ("subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# scene specification


REQUIRED_TOP = ("spec_id", "split", "stratum", "marker_families", "task",
                "canvas", "media", "subjects", "shots", "sound", "render")

# `marker_families` says which markers occur; `stratum` says what a result from
# this row is allowed to speak about. They differ on purpose. A lyrics run
# necessarily wraps a `<d>` block, so a lyrics scene always contains the
# dialogue marker -- but the brief authorizes conclusions per family and treats
# an interaction as its own declared stratum, so reading such a row as evidence
# about dialogue would pool an interaction into a single-family estimate.
STRATA = ("dialogue-only", "dialogue-plus-translated-caption",
          "lyrics-sung-run", "cutoff-at-boundary")


def validate_scene(scene: dict) -> list[str]:
    """Structural validation of a scene specification.

    Deliberately not a schema library: the failures worth catching here are
    semantic (a speaker nobody defined, a keyframe with no time, a lyrics run
    with no sung line), and those need the fields' meanings, not their types.
    """
    problems = []
    for key in REQUIRED_TOP:
        if key not in scene:
            problems.append(f"scene lacks {key}")
    if problems:
        return problems

    if scene["split"] not in ("evaluation", "development"):
        problems.append(f"unknown split {scene['split']!r}")
    if scene["stratum"] not in STRATA:
        problems.append(f"unknown stratum {scene['stratum']!r}")
    for family in scene["marker_families"]:
        if family not in ("dialogue", "caption", "lyrics", "cutoff"):
            problems.append(f"unknown marker family {family!r}")

    subjects = {s["id"] for s in scene["subjects"]}
    labels = [m["label"] for m in scene["media"]]
    for kind in ("Picture", "Video", "Audio"):
        found = [re.search(r"(\d+)", label)
                 for label in labels if label.startswith(f"<{kind}")]
        if any(f is None for f in found):
            problems.append(f"a <{kind} ...> label carries no ordinal")
            continue
        seen = [int(f.group(1)) for f in found if f is not None]
        if seen != list(range(1, len(seen) + 1)):
            problems.append(f"{kind} labels are not a 1..n run: {seen}")

    for item in scene["media"]:
        if item["role"] not in ("reference-still", "reference-video",
                                "reference-audio", "keyframe-first",
                                "keyframe-last"):
            problems.append(f"{item['label']}: unknown role {item['role']!r}")
        if item["role"].startswith("keyframe") and "keyframe_time" not in item:
            problems.append(f"{item['label']}: a keyframe declares no time")
        if item["role"] != "reference-audio":
            for key in ("path", "sha256"):
                if not item.get(key):
                    problems.append(f"{item['label']}: no {key}")

    canvas = scene["canvas"]
    if canvas["frame_count"] % 17 != 5:
        problems.append(
            f"frame_count {canvas['frame_count']} is not on the 17n+5 grid")
    if abs(canvas["frame_count"] / 24.0 - canvas["duration_seconds"]) > 0.05:
        problems.append(
            f"duration {canvas['duration_seconds']}s does not match "
            f"{canvas['frame_count']} frames at 24 fps")
    if canvas["still_policy"] not in ("upscale_2048", "max_no_upscale"):
        problems.append(f"unknown still policy {canvas['still_policy']!r}")

    families = set(scene["marker_families"])
    seen_families = set()
    for shot in scene["shots"]:
        for key in ("index", "start_seconds", "camera", "action"):
            if key not in shot:
                problems.append(f"shot {shot.get('index')}: lacks {key}")
        for event in shot.get("events", []):
            kind = event["kind"]
            if kind == "dialogue":
                seen_families.add("dialogue")
                if event["speaker"] not in subjects:
                    problems.append(
                        f"shot {shot['index']}: dialogue speaker "
                        f"{event['speaker']!r} is not a defined subject")
                for key in ("language", "words", "delivery", "mouth"):
                    if not event.get(key):
                        problems.append(
                            f"shot {shot['index']}: dialogue lacks {key}")
                for marker in ALL_MARKERS:
                    if marker in event.get("words", ""):
                        problems.append(
                            f"shot {shot['index']}: dialogue words contain "
                            f"{marker}; <d> carries the language tag and the "
                            f"words and nothing else")
                if event.get("sung") and "lyrics" not in families:
                    problems.append(
                        f"shot {shot['index']}: a sung line but the scene does "
                        f"not declare the lyrics family")
                if event.get("cutoff"):
                    seen_families.add("cutoff")
            elif kind == "caption":
                seen_families.add("caption")
                for key in ("text", "layout", "prose_request"):
                    if not event.get(key):
                        problems.append(
                            f"shot {shot['index']}: caption lacks {key}")
            elif kind == "silence":
                if not event.get("requirement"):
                    problems.append(f"shot {shot['index']}: silence lacks a requirement")
            else:
                problems.append(f"shot {shot['index']}: unknown event kind {kind!r}")

    sung = [e for shot in scene["shots"] for e in shot.get("events", [])
            if e.get("sung")]
    if "lyrics" in families:
        seen_families.add("lyrics")
        if not sung:
            problems.append(
                "the scene declares the lyrics family but marks no line sung; "
                "a lyrics pair that wraps no <d> block marks nothing")
    if families - seen_families:
        problems.append(
            f"declared families {sorted(families - seen_families)} never occur "
            f"in the shots")
    if seen_families - families:
        problems.append(
            f"families {sorted(seen_families - families)} occur in the shots "
            f"but are not declared; authorization is per family")
    return problems


# ---------------------------------------------------------------------------
# serialization: exactly one place where prose is produced


def _timecode(seconds: float) -> str:
    """`MM:SS.mmm`, the shape the released rows use."""
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _dialogue_span(event: dict) -> str:
    span = f"{DIALOGUE[0]}[{event['language']}] {event['words']}{DIALOGUE[1]}"
    if event.get("cutoff"):
        span += CUTOFF
    if event.get("sung"):
        span = f"{LYRICS[0]}{span}{LYRICS[1]}"
    return span


def _event_prose(event: dict) -> str:
    kind = event["kind"]
    if kind == "dialogue":
        lead = event.get("lead") or (
            f"{event['speaker']} {event['delivery']}")
        # No comma after the closing marker: the released rows run straight on
        # from `</d>` into the mouth clause, and a marker's neighbouring
        # punctuation is part of what a tokenizer arm sees.
        return f"{lead}, {_dialogue_span(event)} {event['mouth']}."
    if kind == "caption":
        return (f"{event['prose_request']} The line reads "
                f"{CAPTION[0]}{event['text']}{CAPTION[1]}.")
    return event["requirement"]


def serialize(scene: dict) -> str:
    """The canonical prompt text. The only place prose is produced.

    Every arm is a transformation of this string, so two arms cannot describe
    two scenes. Marker placement follows `preflight_graph.marker_rules`:
    markers ride inline in the shot prose, only `[Shot N]` opens a line, and a
    caption pair is a sibling of `<d>` rather than a wrapper.
    """
    out = []

    out.append("subject_definitions:")
    for subject in scene["subjects"]:
        out.append(f"{subject['id']} is {subject['definition']}")
    out.append("")

    task = scene["task"]
    out.append("summary:")
    out.append(f"[{task['generation_kind']}] {task['summary']}")
    out.append("")

    out.append("retention_analysis:")
    for subject in scene["subjects"]:
        shots = ", ".join(f"[Shot {i}]" for i in subject["appears_in"])
        out.append(f"{subject['id']} (appears in {shots}): "
                   f"{subject['retention']}")
    out.append("")

    out.append("detailed_description:")
    for shot in scene["shots"]:
        parts = [f"[Shot {shot['index']}]"]
        camera = shot["camera"]
        if shot["index"] > 1:
            parts.append(f"At {_timecode(shot['start_seconds'])},")
        else:
            camera = camera[:1].upper() + camera[1:]
        parts.append(f"{camera}.")
        parts.append(f"{shot['action']}")
        for event in shot.get("events", []):
            parts.append(_event_prose(event))
        if shot.get("closing"):
            parts.append(shot["closing"])
        out.append(" ".join(parts))
    out.append("")

    out.append("overall_soundscape:")
    out.append(scene["sound"]["overall"])
    out.append("")
    out.append("non_diegetic_music:")
    out.append(scene["sound"]["non_diegetic"])
    return "\n".join(out)


def strip_markers(text: str) -> str:
    """Remove only the marker strings. Every other character survives."""
    for marker in ALL_MARKERS:
        text = text.replace(marker, "")
    return text


def marker_spans(text: str) -> list[dict]:
    spans = []
    for marker in ALL_MARKERS:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            spans.append({"marker": marker, "char_start": index,
                          "char_end": index + len(marker),
                          "release_id": RELEASE_MARKER_IDS[marker]})
            start = index + len(marker)
    return sorted(spans, key=lambda s: s["char_start"])


# ---------------------------------------------------------------------------
# arms


ARM_TRANSFORMS = {
    "release_id": {
        "text": "identity",
        "tokenizer": "release: the loaded tokenizer that owns the seven H3 markers",
        "model": "none",
    },
    "legacy_bpe": {
        "text": "identity",
        "tokenizer": "legacy: a genuinely unpatched tokenizer, reconstructed from "
                     "the pre-patch constructor and selected by vocabulary",
        "model": "none",
    },
    "stripped": {
        "text": "remove the marker strings; every other character survives",
        "tokenizer": "release",
        "model": "none",
    },
    "mean_init_rows": {
        "text": "identity",
        "tokenizer": "release",
        "model": "replace the seven marker embedding rows with the mean of the "
                 "embedding table. DECLARED ONLY -- nothing here applies it, and "
                 "no file in this repo may write a token row",
    },
}


def compile_arms(scene: dict) -> tuple[str, dict]:
    canonical = serialize(scene)
    texts = {
        "release_id": canonical,
        "legacy_bpe": canonical,
        "stripped": strip_markers(canonical),
        "mean_init_rows": canonical,
    }
    arms = {}
    for name in ARMS:
        text = texts[name]
        arms[name] = {
            "transform": ARM_TRANSFORMS[name],
            "prompt": text,
            "prompt_bytes": len(text.encode("utf-8")),
            "prompt_sha256": sha(text),
            "ordinary_text_sha256": sha(strip_markers(text)),
            "marker_spans": marker_spans(text),
        }
    return canonical, arms


# ---------------------------------------------------------------------------
# tokenization, optional


def _tokenizers():
    """(release, legacy) tokenizers, or (None, None) with a reason.

    Selected by VOCABULARY, never by a branch or constant name: an installed
    tokenizer that does not own the markers cannot serve as the release arm,
    and saying so is the point of the check.
    """
    try:
        sys.path.insert(0, str(REPO.parents[1]))
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
        import importlib.util
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

        release = MiniMaxH3Tokenizer()
        vocab = release.qwen3vl_32b.tokenizer.get_vocab()
        missing = [m for m in RELEASE_MARKER_IDS if m not in vocab]
        if missing:
            return None, None, (
                f"the installed tokenizer does not own {missing}; it cannot be "
                f"the release arm")
        wrong = {m: vocab[m] for m, i in RELEASE_MARKER_IDS.items()
                 if vocab[m] != i}
        if wrong:
            return None, None, f"released marker ids disagree: {wrong}"

        path = BENCH / "audit_h3_marker_tokenization.py"
        spec = importlib.util.spec_from_file_location("_marker_audit", path)
        if spec is None or spec.loader is None:
            return None, None, f"cannot load {path.name} for the legacy arm"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import comfy.text_encoders.minimax as mmx

        targets = []
        for owner, attr in ((MiniMaxH3Tokenizer, "H3_SPECIAL_TOKENS"),
                            (mmx, "MINIMAX_EXTRA_TOKENS")):
            if getattr(owner, attr, None):
                targets.append((owner, attr, getattr(owner, attr)))
        if not targets:
            return None, None, (
                "no H3 special-token list found on the installed tokenizer, so "
                "the legacy arm cannot be reconstructed; it would silently be a "
                "second copy of the release arm")
        try:
            for owner, attr, _ in targets:
                setattr(owner, attr, [])
            legacy = MiniMaxH3Tokenizer()
        finally:
            for owner, attr, saved in targets:
                setattr(owner, attr, saved)
        legacy_vocab = legacy.qwen3vl_32b.tokenizer.get_vocab()
        still_there = [m for m in RELEASE_MARKER_IDS if m in legacy_vocab]
        if still_there:
            return None, None, (
                f"the reconstructed legacy tokenizer still owns {still_there}; "
                f"it is not an unpatched arm")
        return release, legacy, None
    except Exception as exc:
        return None, None, f"tokenizers unavailable: {exc}"


def _ids(tokenizer, text: str) -> list[int]:
    batches = tokenizer.tokenize_with_weights(text)["qwen3vl_32b"]
    return [int(entry[0]) for row in batches for entry in row]


def tokenize_arms(arms: dict) -> str | None:
    release, legacy, reason = _tokenizers()
    if reason:
        for arm in arms.values():
            arm["tokens"] = {"recorded": False, "reason": reason}
        return reason
    for name, arm in arms.items():
        tokenizer = legacy if name == "legacy_bpe" else release
        ids = _ids(tokenizer, arm["prompt"])
        marker_ids = set(RELEASE_MARKER_IDS.values())
        arm["tokens"] = {
            "recorded": True,
            "tokenizer": "legacy" if name == "legacy_bpe" else "release",
            "count": len(ids),
            "ids_sha256": sha(canonical_json(ids)),
            "ids": ids,
            "marker_positions": [i for i, t in enumerate(ids) if t in marker_ids],
            "marker_ids_present": sorted({t for t in ids if t in marker_ids}),
        }
    return None


def label_positions(arms: dict, scene: dict) -> None:
    """Ordinary-BPE positions of the `<Picture i>` / `<Video i>` labels.

    These are ordinary text, not special tokens, so their positions are a
    property of the surrounding stream. The brief wants them recorded because
    a marker transformation must not move them.
    """
    for arm in arms.values():
        found = {}
        for item in scene["media"]:
            label = item["label"]
            found[label] = [m.start() for m in
                            re.finditer(re.escape(label), arm["prompt"])]
        arm["label_char_positions"] = found


# ---------------------------------------------------------------------------


def compile_scene(scene: dict, tokenize: bool = True) -> dict:
    problems = validate_scene(scene)
    if problems:
        raise ValueError(f"{scene.get('spec_id')}: " + "; ".join(problems))
    canonical, arms = compile_arms(scene)
    label_positions(arms, scene)
    reason = tokenize_arms(arms) if tokenize else "tokenization not requested"
    spec_body = {k: v for k, v in scene.items() if k != "_notes"}
    return {
        "spec_id": scene["spec_id"],
        "split": scene["split"],
        "stratum": scene["stratum"],
        "marker_families": scene["marker_families"],
        "scene_spec_sha256": sha(canonical_json(spec_body)),
        "canonical_prompt": canonical,
        "canonical_prompt_sha256": sha(canonical),
        "ordinary_text_sha256": sha(strip_markers(canonical)),
        "canvas": scene["canvas"],
        "render": scene["render"],
        "media": [
            {"label": m["label"], "role": m["role"], "path": m.get("path"),
             "sha256": m.get("sha256"),
             "keyframe_time": m.get("keyframe_time"),
             "geometry_stage_1": m.get("geometry_stage_1"),
             "geometry_stage_2": m.get("geometry_stage_2")}
            for m in scene["media"]],
        "tokenization_skipped_because": reason,
        "arms": arms,
    }


def load_scenes(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path,
                        default=BENCH / "marker_corpus" / "scenes")
    parser.add_argument("--out", type=Path,
                        default=BENCH / "marker_corpus" / "compiled.json")
    parser.add_argument("--no-tokenize", action="store_true")
    args = parser.parse_args()

    scenes = load_scenes(args.scenes)
    if not scenes:
        print(f"no scene specifications in {args.scenes}")
        return 2
    compiled = []
    for scene in scenes:
        try:
            compiled.append(compile_scene(scene, not args.no_tokenize))
        except ValueError as exc:
            print(f"[invalid] {exc}")
            return 1
    skipped = {c["tokenization_skipped_because"] for c in compiled}
    skipped.discard(None)
    document = {
        "producer": Path(__file__).name,
        "brief": "docs/research/qwen3-vl-special-tokens-post-training/canonical/"
                 "owner_authored_marker_corpus.md",
        "arms": list(ARMS),
        "arm_transforms": ARM_TRANSFORMS,
        "standing": ("a seed set, not the frozen evaluation corpus; freezing is "
                     "the owner's act and requires the coverage the brief names"),
        "declared_missing_cells": [
            {"cell": "standalone reference audio paired with a referenced speaker",
             "why": "no authorized audio reference has been selected and heard; "
                    "authoring speaker prose against unheard audio would be the "
                    "prompt/media mismatch this corpus exists to avoid"},
            {"cell": "reference video and video soundtrack",
             "why": "the pool carries genuine input reference video, but the "
                    "scenes here were authored against media that was looked at, "
                    "and the clips have not been"},
            {"cell": "cutoff in a text-only row",
             "why": "cutoff marks an incomplete vocal event at the video "
                    "boundary; without a video there is no boundary for it to "
                    "sit at, so this is not a missing row but a nonsensical one"},
        ],
        "scenes": compiled,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    print(f"compiled {len(compiled)} scene(s) into {len(ARMS)} arms each")
    for entry in compiled:
        families = entry["stratum"]
        tokens = entry["arms"]["release_id"].get("tokens", {})
        count = tokens.get("count", "-") if tokens.get("recorded") else "-"
        print(f"  {entry['spec_id']:24s} {entry['split']:11s} {families:22s} "
              f"{len(entry['media'])} media, {count} release tokens")
    for reason in sorted(skipped):
        print(f"  token streams not recorded: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
